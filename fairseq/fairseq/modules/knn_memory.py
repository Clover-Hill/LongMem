import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary

class KNN_Dstore(object):
    def __init__(self, args):
        self.dimension = args.embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.dstore_fp16 = args.dstore_fp16
        self.n_head = args.num_attention_heads
        self.head_dim = self.dimension // self.n_head
        # Probe trades speed for performance
        self.probe = args.probe
        
        # Initialize self.keys, self.vals, self.index
        self.keys, self.vals, self.index = self.setup_faiss(args)

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        keys = []
        vals = []
        index = []
        
        # Read Index
        start = time.time()
        
        for i in range(self.n_head):
            cur_index = faiss.read_index(args.indexfile+f'_{i}.index', faiss.IO_FLAG_ONDISK_SAME_DIR)
            cur_index.nprobe = self.probe
            index.append(cur_index)

        print('Reading datastore took {} s'.format(time.time() - start))

        # Memmap Dstore
        for i in range(self.n_head):
            if args.dstore_fp16:
                print('Keys are fp16 and vals are int16')
                cur_key = np.memmap(args.dstore_filename+f'_keys_{i}.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                cur_val = np.memmap(args.dstore_filename+f'_vals_{i}.npy', dtype=np.int16, mode='r', shape=(self.dstore_size, self.dimension))
            else:
                print('Keys are fp32 and vals are int64')
                cur_key = np.memmap(args.dstore_filename+f'_keys_{i}.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                cur_val = np.memmap(args.dstore_filename+f'_vals_{i}.npy', dtype=np.int, mode='r', shape=(self.dstore_size, self.dimension))
            
            keys.append(cur_key)
            vals.append(cur_val)

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
        knns = [self.index[i].search(queries[:, i, :].contiguous(), self.k)[1] for i in range(self.n_head)]

        keys_tgt_index = [self.keys[i][knns[i]].view(seq_len*bsz, self.k, self.head_dim) for i in range(self.n_head)]
        vals_tgt_index = [self.vals[i][knns[i]].view(seq_len*bsz, self.k, self.head_dim) for i in range(self.n_head)] 

        # torch.stack(): Effectively means adding a new dimension in the dim param
        # list of (seq_len*bsz) * k * head_dim -> (seq_len*bsz) * num_heads * k * head_dim
        keys_tgt_index = torch.stack(keys_tgt_index, dim=1).view(seq_len, bsz*self.n_head, self.k, self.head_dim).transpose(0, 1)
        vals_tgt_index = torch.stack(vals_tgt_index, dim=1).view(seq_len, bsz*self.n_head, self.k, self.head_dim).transpose(0, 1)

        keys_tgt_index = keys_tgt_index.view(bsz, self.n_head, seq_len, self.k, self.head_dim)
        vals_tgt_index = vals_tgt_index.view(bsz, self.n_head, seq_len, self.k, self.head_dim)

        return {"keys": keys_tgt_index, "vals": vals_tgt_index}