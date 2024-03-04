import torch
import faiss
import math
import numpy as np
from fairseq import utils
import torch.distributed
import pickle
import time
import os
from tqdm import tqdm
from fairseq.data import Dictionary

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
        # Probe trades speed for performance
        self.probe = args.probe
        
        # Initialize self.keys, self.vals, self.index
        self.keys, self.vals, self.index = self.setup_faiss(args)
    
    def is_master(self):
        if not torch.distributed.is_initialized():
            return True
        if torch.distributed.get_rank()==0:
            return True 
        return False
    
    def setup_faiss(self, args):
        if not os.path.exists(self.dstore_dir):
            raise ValueError('Cannot build a datastore without the data.')
        
        keys = []
        vals = []
        index = []
        
        # Read Index
        start = time.time()
        
        for i in tqdm(range(self.n_head), desc="Reading index files"):
            cur_index = faiss.read_index(os.path.join(self.dstore_dir, f'{i}.index'), faiss.IO_FLAG_ONDISK_SAME_DIR)
            cur_index.nprobe = self.probe
            index.append(cur_index)
        
        # Memmap Dstore
        for i in tqdm(range(self.n_head), desc="Reading Dstore to Memory"):
            if args.dstore_fp16:
                cur_key = np.memmap(os.path.join(self.dstore_dir, f'{i}_keys.npy'), dtype=np.float16, mode='r', shape=(self.dstore_size, self.head_dim))
                cur_val = np.memmap(os.path.join(self.dstore_dir, f'{i}_vals.npy'), dtype=np.float16, mode='r', shape=(self.dstore_size, self.head_dim))
            else:
                cur_key = np.memmap(os.path.join(self.dstore_dir, f'{i}_keys.npy'), dtype=np.float33, mode='r', shape=(self.dstore_size, self.head_dim))
                cur_val = np.memmap(os.path.join(self.dstore_dir, f'{i}_vals.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.head_dim))
            
            keys.append(pickle.dumps(torch.tensor(cur_key).cpu().contiguous()))
            vals.append(pickle.dumps(torch.tensor(cur_val).cpu().contiguous()))
        
        print('Reading datastore took {} s'.format(time.time() - start))
    
        # Current can only load vals;
        # If the memory size exceeds 300GB, we can then also load keys;
        # if args.move_dstore_to_mem:
        #     print('Loading to memory...')
        #     start = time.time()

            # del self.keys
            # self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            # self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
            # self.keys = self.keys_from_memmap[:]
            # self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            # del self.vals
            # self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, self.dimension))
            # self.vals = np.zeros((self.dstore_size, self.dimension), dtype=np.int16 if args.dstore_fp16 else np.int)
            # self.vals = self.vals_from_memmap[:]
            # self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
            # print('Loading to memory took {} s'.format(time.time() - start))
            
        return keys, vals, index

    def retrieve(self, queries):
        seq_len, bsz, hidden_dim = queries.shape
        queries = queries.view(seq_len*bsz, self.n_head, self.head_dim).type(torch.float32)
        
        # knn shape: (seq_len*bsz) * k * dimension 
        # Perform KNN Search 
        start_time = time.time()
        # Change query to cpu tensor
        knns = [self.index[i].search(queries[:, i, :].contiguous().detach().cpu().float().numpy(), self.k)[1] for i in range(self.n_head)]
        print(f'Search for query {queries.shape} cost {time.time()-start_time}s')
        
        import pdb 
        pdb.set_trace()
        
        start_time = time.time()
        keys_tgt_index = []
        vals_tgt_index = []
        for i in tqdm(range(self.n_head), desc=f'Key Value Indexing'):
            self.keys[i]=pickle.loads(self.keys[i])
            self.vals[i]=pickle.loads(self.vals[i])
            if torch.is_tensor(self.keys[i]):
                cur_keys = self.keys[i]
            else:
                cur_keys = np.zeros((self.dstore_size,self.head_dim),dtype=np.float16)
                cur_keys = torch.tensor(self.keys[i][:]).cpu().contiguous()
            
            if torch.is_tensor(self.vals[i]):
                cur_vals = self.vals[i]
            else:
                cur_vals = np.zeros((self.dstore_size,self.head_dim),dtype=np.float16)
                cur_vals = torch.tensor(self.vals[i][:]).cpu().contiguous()
            
            keys_tgt_index.append(cur_keys[knns[i]])
            vals_tgt_index.append(cur_vals[knns[i]])

        print(f'Finding keys and vals cost {time.time()-start_time}s')

        pdb.set_trace()
        
        # torch.stack(): Effectively means adding a new dimension in the dim param
        # list of (seq_len*bsz) * k * head_dim -> (seq_len*bsz) * num_heads * k * head_dim
        keys_tgt_index = torch.stack(keys_tgt_index, dim=1).view(seq_len, bsz*self.n_head, self.k, self.head_dim).transpose(0, 1)
        vals_tgt_index = torch.stack(vals_tgt_index, dim=1).view(seq_len, bsz*self.n_head, self.k, self.head_dim).transpose(0, 1)

        keys_tgt_index = keys_tgt_index.view(bsz, self.n_head, seq_len, self.k, self.head_dim)
        vals_tgt_index = vals_tgt_index.view(bsz, self.n_head, seq_len, self.k, self.head_dim)

        return {"keys": keys_tgt_index, "vals": vals_tgt_index}