import torch
import time
import pickle
import faiss
import ray
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from tqdm import tqdm
import tracemalloc 
import numpy as np
import os

if __name__ == '__main__':
    n_head = 1
    head_dim = 768 // 12
    start_time = time.time()
    dstore_size = 103226509 // 4
    k = 1024
    dstore_dir = "/home/jqcao/projects/memory_transformer/LongMem/key_value_dstore/lr_5e-5_chunk_4"
    # dstore_dir = "/tmp/jqcao/lr_5e-5_chunk_4"
    
    tracemalloc.start()
    
    print(f"Memory Cost: {tracemalloc.get_traced_memory()}")
    tracemalloc.stop()
    print(f"Time cost: {time.time()-start_time}")