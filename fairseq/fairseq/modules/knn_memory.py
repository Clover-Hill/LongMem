import torch
import faiss
import math
import numpy as np
from fairseq import utils
import torch.distributed
import time
import os
import ray
import asyncio
from tqdm import tqdm
from multiprocessing.shared_memory import SharedMemory
from fairseq.data import Dictionary

class KNN_Dstore:
    def __init__(self, args):
        self.dimension = args.n_embd
        self.k = args.k
        # Use chuknk for storing memory
        self.dstore_size = args.dstore_size // args.chunk_size
        self.dstore_fp16 = args.dstore_fp16
        self.n_head = args.num_attention_heads
        self.head_dim = self.dimension // self.n_head
        self.dstore_dir = args.dstore_dir
        self.dstore_fp16 = args.dstore_fp16
        # Probe trades speed for performance
        self.probe = args.probe
        
        # Read Index and set IO flag to faiss.IO_FLAG_MMAP
        self.index = []
        for i in tqdm(range(self.n_head),desc=f"Loading Index for rank {self.get_rank()}"):
            if args.faiss_index_mmap:
                cur_index = faiss.read_index(os.path.join(self.dstore_dir, f'{i}.index'), faiss.IO_FLAG_MMAP)
            else:
                cur_index = faiss.read_index(os.path.join(self.dstore_dir, f'{i}.index'))
            cur_index.nprobe = self.probe
            self.index.append(cur_index)
            
        # Keys and Vals
        if self.dstore_fp16:
            self.array_type = np.float16
        else:
            self.array_type = np.float32
        self.array_shape = (self.n_head, self.dstore_size, self.head_dim)
    
    def build_dstore(self):
        if not os.path.exists(self.dstore_dir):
            raise ValueError('Cannot build a datastore without the data.')
        
        keys = []
        vals = []
        
        start_time = time.time()
        
        # Move Dstore to ray storage with multiprocessing
        for i in tqdm(range(self.n_head), desc="Reading Dstore to Memory"):
            cur_key = np.memmap(os.path.join(self.dstore_dir, f'{i}_keys.npy'), dtype=self.array_type, mode='r', shape=(self.dstore_size, self.head_dim))
            cur_val = np.memmap(os.path.join(self.dstore_dir, f'{i}_vals.npy'), dtype=self.array_type, mode='r', shape=(self.dstore_size, self.head_dim))
            keys.append(np.array(cur_key))
            vals.append(np.array(cur_val))
            
        keys = np.array(keys)
        vals = np.array(vals)
        
        shm_keys = SharedMemory(create = True, size = keys.nbytes, name="shm_keys")
        shm_vals = SharedMemory(create = True, size = vals.nbytes, name="shm_vals")
        
        tmp_keys = np.ndarray(self.array_shape, dtype = self.array_type, buffer = shm_keys.buf)
        tmp_vals = np.ndarray(self.array_shape, dtype = self.array_type, buffer = shm_vals.buf)
        tmp_keys[:] = keys[:]
        tmp_vals[:] = vals[:]
        
        print('Reading datastore took {} s'.format(time.time() - start_time))

    def retrieve(self, queries):
        # Load from SharedMemory
        shm_keys = SharedMemory(create = False, name="shm_keys")
        shm_vals = SharedMemory(create = False, name="shm_vals")

        keys = np.ndarray(self.array_shape, dtype = self.array_type, buffer = shm_keys.buf)
        vals = np.ndarray(self.array_shape, dtype = self.array_type, buffer = shm_vals.buf)
        
        # queries is of shape (bsz, seq_len, hidden_dim)
        bsz, seq_len, hidden_dim = queries.shape
        queries = queries.view(seq_len*bsz, self.n_head, self.head_dim).type(torch.float32)
        
        # knn shape: (seq_len*bsz) * k * dimension 
        start_time = time.time()
        # Perform KNN Search and Change query to cpu tensor
        knns = [self.index[i].search(queries[:, i, :].contiguous().detach().cpu().float().numpy(), self.k)[1] for i in range(self.n_head)]
        # print(f'Search for query {queries.shape} cost {time.time()-start_time}s')
        
        start_time = time.time()
        # Perform parallel retrieving vectors based on index
        keys_tgt_index = []
        vals_tgt_index = []
        for i in range(self.n_head):
            keys_tgt_index.append(torch.tensor(keys[i][knns[i]]))
            vals_tgt_index.append(torch.tensor(vals[i][knns[i]]))
        
        # print(f'Finding keys and vals cost {time.time()-start_time}s')

        # torch.stack(): Effectively means adding a new dimension in the dim param
        # list of (seq_len*bsz) * k * head_dim -> (seq_len*bsz) * num_heads * k * head_dim
        keys_tgt_index = torch.stack(keys_tgt_index, dim=1).view(seq_len, bsz*self.n_head, self.k, self.head_dim).transpose(0, 1)
        vals_tgt_index = torch.stack(vals_tgt_index, dim=1).view(seq_len, bsz*self.n_head, self.k, self.head_dim).transpose(0, 1)

        keys_tgt_index = keys_tgt_index.view(bsz, self.n_head, seq_len, self.k, self.head_dim)
        vals_tgt_index = vals_tgt_index.view(bsz, self.n_head, seq_len, self.k, self.head_dim)

        return {"keys": keys_tgt_index, "vals": vals_tgt_index}

    def is_master(self):
        if not torch.distributed.is_initialized():
            return True
        if torch.distributed.get_rank()==0:
            return True 
        return False

    def get_rank(self):
        if not torch.distributed.is_initialized():
            return -1
        else:
            return torch.distributed.get_rank()