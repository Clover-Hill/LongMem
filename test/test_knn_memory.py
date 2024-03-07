import torch
import time
import pickle
import faiss
import ray
from ray.util.multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from tqdm import tqdm
import tracemalloc 
import numpy as np
import os

from fairseq.models.hf_newgpt.configuration_newgpt import NewGPTConfig
from fairseq.modules.knn_memory import KNN_Dstore

def test_multiprocessing(rank):
    start_time = time.time()
    
    print(f"Rank {rank} is starting...")
    knn_memory = ray.get_actor("test")
    
    queries = torch.randn(16,1,768)
    retrieve_result = ray.get(knn_memory.retrieve.remote(queries))
    shape = retrieve_result["keys"].shape

    print(f"Retrieve shape for rank {rank}: {shape}")
    print(f"Time cost for rank {rank}: {time.time()-start_time}")

if __name__ == '__main__':
    n_head = 12
    n_embd = 768
    head_dim = 768 // 12
    start_time = time.time()
    dstore_size = 103226509 // 4
    k = 1024
    dstore_dir = "/home/jqcao/projects/memory_transformer/LongMem/key_value_dstore/lr_5e-5_chunk_4"
    # dstore_dir = "/tmp/jqcao/lr_5e-5_chunk_4"
    
    tracemalloc.start()
    
    ray.shutdown()
    ray.init( object_store_memory = 120 * 10**9 )
    config = NewGPTConfig(n_embd=n_embd,n_head=n_head,dstore_dir=dstore_dir, k=k)
    
    # knn_memory = KNN_Dstore.options(name="testing").remote(config)
    # knn_memory_2 = ray.get_actor("testing")

    # retrieve_result = ray.get([knn_memory.retrieve.remote(torch.randn(1024,8,768)),knn_memory_2.retrieve.remote(torch.randn(1024,8,768))])
    # shape = retrieve_result[1]["keys"].shape

    # print(f"Retrieve shape: {shape}")
    # print(f"Memory Cost: {tracemalloc.get_traced_memory()}")
    # print(f"Time cost: {time.time()-start_time}")
    
    pool = Pool()
    rank = range(2)
    
    dstore_actor = KNN_Dstore.options(name="test").remote(config)
    
    pool.map(test_multiprocessing, rank)
    
    pool.close()
    pool.join()
    print(f"Memory Cost Overall: {tracemalloc.get_traced_memory()}")
    print(f"Time cost Overall: {time.time()-start_time}")
    
    tracemalloc.stop()