import torch
import faiss
import math
import numpy as np
from fairseq import utils
import torch.distributed
import ray
import time
import os
import ray
from tqdm import tqdm
from fairseq.data import Dictionary

# KNN_Dstore is a stateful ray actor
@ray.remote
class KNN_Dstore(object):
    def __init__(self, args):
        self.dimension = args.n_embd
        self.k = args.k
        # Use chunk for storing memory
        self.dstore_size = args.dstore_size // args.chunk_size
        self.dstore_fp16 = args.dstore_fp16
        self.n_head = args.num_attention_heads
        self.head_dim = self.dimension // self.n_head
        self.dstore_dir = args.dstore_dir
        self.dstore_fp16 = args.dstore_fp16
        # Probe trades speed for performance
        self.probe = args.probe
        
        # Store the [RefObject] of keys, vals and index
        self.ref_keys, self.ref_vals, self.index = self.setup_faiss(args)
    
    def load_dstore_parallel(self,i):
        if self.dstore_fp16:
            cur_key = np.memmap(os.path.join(self.dstore_dir, f'{i}_keys.npy'), dtype=np.float16, mode='r', shape=(self.dstore_size, self.head_dim))
            cur_val = np.memmap(os.path.join(self.dstore_dir, f'{i}_vals.npy'), dtype=np.float16, mode='r', shape=(self.dstore_size, self.head_dim))
        else:
            cur_key = np.memmap(os.path.join(self.dstore_dir, f'{i}_keys.npy'), dtype=np.float33, mode='r', shape=(self.dstore_size, self.head_dim))
            cur_val = np.memmap(os.path.join(self.dstore_dir, f'{i}_vals.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.head_dim))
        return  (ray.put(np.array(cur_key)), ray.put(np.array(cur_val)))
    
    def retrieve_parallel(self,ref_key,ref_val,knn_index):
        return (ref_key[knn_index],ref_val[knn_index])
    
    def setup_faiss(self, args):
        if not os.path.exists(self.dstore_dir):
            raise ValueError('Cannot build a datastore without the data.')
        
        keys = []
        vals = []
        key_val = []
        index = []
        
        # Read Index and set IO flag to faiss.IO_FLAG_MMAP
        start = time.time()
        
        for i in tqdm(range(self.n_head), desc="Reading index files"):
            cur_index = faiss.read_index(os.path.join(self.dstore_dir, f'{i}.index'), faiss.IO_FLAG_MMAP)
            cur_index.nprobe = self.probe
            index.append(cur_index)
        
        # Move Dstore to ray storage with multiprocessing
        for i in tqdm(range(self.n_head), desc="Reading Dstore to Memory"):
            key_val.append(self.load_dstore_parallel.remote(i))
        key_val = ray.get(key_val)
        keys = [key_val[i][0] for i in range(self.n_head)]
        vals = [key_val[i][1] for i in range(self.n_head)]
        
        print('Reading datastore took {} s'.format(time.time() - start))
    
        # keys and vals are lists of ray RefObject, Index is a list of FAISS Index
        return keys, vals, index

    async def retrieve(self, queries):
        seq_len, bsz, hidden_dim = queries.shape
        queries = queries.view(seq_len*bsz, self.n_head, self.head_dim).type(torch.float32)
        
        # knn shape: (seq_len*bsz) * k * dimension 
        start_time = time.time()
        # Perform KNN Search and Change query to cpu tensor
        knns = [self.index[i].search(queries[:, i, :].contiguous().detach().cpu().float().numpy(), self.k)[1] for i in range(self.n_head)]
        print(f'Search for query {queries.shape} cost {time.time()-start_time}s')
        
        start_time = time.time()
        # Perform parallel retrieving vectors based on index
        all_tgt_index = []
        keys_tgt_index = []
        vals_tgt_index = []
        for i in range(self.n_head):
                all_tgt_index.append(self.retrieve_parallel.remote(self.ref_keys[i],self.ref_vals[i],knns[i]))
        keys_tgt_index = [torch.tensor(all_tgt_index[i][0]) for i in range(self.n_head)]
        vals_tgt_index = [torch.tensor(all_tgt_index[i][1]) for i in range(self.n_head)]
        print(f'Finding keys and vals cost {time.time()-start_time}s')

        # torch.stack(): Effectively means adding a new dimension in the dim param
        # list of (seq_len*bsz) * k * head_dim -> (seq_len*bsz) * num_heads * k * head_dim
        keys_tgt_index = torch.stack(keys_tgt_index, dim=1).view(seq_len, bsz*self.n_head, self.k, self.head_dim).transpose(0, 1)
        vals_tgt_index = torch.stack(vals_tgt_index, dim=1).view(seq_len, bsz*self.n_head, self.k, self.head_dim).transpose(0, 1)

        keys_tgt_index = keys_tgt_index.view(bsz, self.n_head, seq_len, self.k, self.head_dim)
        vals_tgt_index = vals_tgt_index.view(bsz, self.n_head, seq_len, self.k, self.head_dim)

        return {"keys": keys_tgt_index, "vals": vals_tgt_index}