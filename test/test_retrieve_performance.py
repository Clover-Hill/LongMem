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

def test_read_index():
    index = []
    
    start_time = time.time()
    for i in tqdm(range(n_head), desc="Reading index files"):
        cur_index = faiss.read_index(os.path.join(dstore_dir, f'{i}.index'))
        index.append(cur_index)
        
    print(f"Reading Index cost {time.time()-start_time}")
    return index

def test_memmap():
    keys = []
    vals = []

    for i in tqdm(range(n_head), desc="Reading Dstore to Memory"):
        cur_key = np.memmap(os.path.join( dstore_dir, f'{i}_keys.npy'), dtype=np.float16, mode='r', shape=( dstore_size,  head_dim))
        cur_val = np.memmap(os.path.join( dstore_dir, f'{i}_vals.npy'), dtype=np.float16, mode='r', shape=( dstore_size,  head_dim))
        
        keys.append(cur_key)
        vals.append(cur_val)        
    
    retrieve(keys,vals)
    
@ray.remote
def load_head_dstore(i):
    cur_key = np.memmap(os.path.join( dstore_dir, f'{i}_keys.npy'), dtype=np.float16, mode='r', shape=( dstore_size,  head_dim))
    cur_val = np.memmap(os.path.join( dstore_dir, f'{i}_vals.npy'), dtype=np.float16, mode='r', shape=( dstore_size,  head_dim))
    return (ray.put(np.array(cur_key)), ray.put(np.array(cur_val)))

def test_ray():
    keys = []
    vals = []
    key_val = []
    
    ray.init(num_cpus = multiprocessing.cpu_count())
    start_time = time.time()

    for i in tqdm(range(n_head), desc="Reading Dstore to Memory"):
        key_val.append(load_head_dstore.remote(i))
    key_val = ray.get(key_val)
    
    for i in range(n_head):
        keys.append(key_val[i][0])
        vals.append(key_val[i][1])
    
    print(f"Load dstore to memory cost {time.time()-start_time}")

    index = test_read_index()
    ray_index = ray.put(index)
    retrieve_ray(keys,vals,ray_index)
    
def test_shared_memory():
    keys = []
    vals = []
    
    start_time = time.time()
    
    for i in tqdm(range(n_head), desc="Reading Dstore to Memory"):
        cur_key = np.memmap(os.path.join( dstore_dir, f'{i}_keys.npy'), dtype=np.float16, mode='r', shape=( dstore_size,  head_dim))
        cur_val = np.memmap(os.path.join( dstore_dir, f'{i}_vals.npy'), dtype=np.float16, mode='r', shape=( dstore_size,  head_dim))
        
        keys.append(np.array(cur_key))
        vals.append(np.array(cur_val))
        
    keys = np.array(keys)
    vals = np.array(vals)
    shm_keys = SharedMemory(name = "keys", create=True, size = keys.nbytes)
    shm_vals = SharedMemory(name = "vals", create=True, size = vals.nbytes)
    
    tmp_keys = np.ndarray(array_shape, dtype = array_type, buffer = shm_keys.buf)
    tmp_vals = np.ndarray(array_shape, dtype = array_type, buffer = shm_vals.buf)
    tmp_keys[:] = keys[:]
    tmp_vals[:] = vals[:]
    
    print(f"Load dstore to memory cost {time.time()-start_time}")
    
    index = test_read_index()
    retrieve(index)

@ray.remote
def retrieve_head(ref_key, ref_val, knn_index):
    return (ref_key[knn_index],ref_val[knn_index])
    
@ray.remote 
def faiss_search_head(head,index_head, queries_head):
    return index_head[head].search(queries_head,k)[1]

def retrieve_ray(keys, vals, index):
    start_time = time.time()
    queries = np.random.uniform(low=-10.0, high=10.0, size=(8192, 12, 64))
    knns = [faiss_search_head.remote(i, index, np.ascontiguousarray(queries[:,i,:]).astype(np.float32)) for i in range(n_head)]
    knns = ray.get(knns)
    # knns = [index[i].search(np.ascontiguousarray(queries[:,i,:]).astype(np.float32), k)[1] for i in range(n_head)]
    print(f"Index searching cost {time.time()-start_time}")
    
    start_time = time.time()
    tgt_index = []
    keys_tgt_index = []
    vals_tgt_index = []
    for i in tqdm(range( n_head), desc=f'Key Value Indexing'):
        tgt_index.append(retrieve_head.remote(keys[i],vals[i],knns[i]))
    tgt_index = ray.get(tgt_index)
    
    keys_tgt_index = [tgt_index[i][0] for i in range(n_head)]
    vals_tgt_index = [tgt_index[i][1] for i in range(n_head)]
    # for i in tqdm(range( n_head), desc=f'Key Value Indexing'):
    #     cur_keys = ray.get(keys[i])
    #     cur_vals = ray.get(vals[i])
        
    #     keys_tgt_index.append(cur_keys[knns[i]])
    #     vals_tgt_index.append(cur_vals[knns[i]])

    print(f'Finding keys and vals cost {time.time()-start_time}s')
    print(f"Memory Cost: {tracemalloc.get_traced_memory()}")
    
def retrieve(index):
    # random_sample = []
    # for i in range( n_head):
    #     random_sample.append(np.random.randint(0,dstore_size-1, size=8192*1024))
    # print(random_sample[1].shape)
    
    start_time = time.time()
    queries = np.random.uniform(low=-10.0, high=10.0, size=(4096, 12, 64))
    knns = [index[i].search(np.ascontiguousarray(queries[:, i, :]).astype(np.float32), k)[1] for i in range(n_head)]
    print(f"Index searching cost {time.time()-start_time}")

    shm_keys = SharedMemory(name = "keys", create=False)
    shm_vals = SharedMemory(name = "vals", create=False)
    keys = np.ndarray(array_shape, dtype = array_type, buffer = shm_keys.buf)
    vals = np.ndarray(array_shape, dtype = array_type, buffer = shm_vals.buf)
    
    start_time = time.time()
    keys_tgt_index = []
    vals_tgt_index = []
    for i in tqdm(range( n_head), desc=f'Key Value Indexing'):
        # if torch.is_tensor( keys[i]):
        #     cur_keys =  keys[i]
        # else:
        #     cur_keys = np.zeros(( dstore_size, head_dim),dtype=np.float16)
        #     cur_keys = torch.tensor( keys[i][:]).cpu().contiguous()
        
        # if torch.is_tensor( vals[i]):
        #     cur_vals =  vals[i]
        # else:
        #     cur_vals = np.zeros(( dstore_size, head_dim),dtype=np.float16)
        #     cur_vals = torch.tensor( vals[i][:]).cpu().contiguous()
        cur_keys = keys[i]
        cur_vals = vals[i]
        
        keys_tgt_index.append(cur_keys[knns[i]])
        vals_tgt_index.append(cur_vals[knns[i]])

    print(keys_tgt_index[0].shape)
    print(f'Finding keys and vals cost {time.time()-start_time}s')
    print(f"Memory Cost: {tracemalloc.get_traced_memory()}")

if __name__ == '__main__':
    n_head = 12
    head_dim = 768 // 12
    start_time = time.time()
    dstore_size = 103226509 // 4
    k = 1024
    dstore_dir = "/home/jqcao/projects/memory_transformer/LongMem/key_value_dstore/lr_5e-5_chunk_4"
    
    array_shape = (n_head, dstore_size, head_dim)
    array_type = np.float16
    # dstore_dir = "/tmp/jqcao/lr_5e-5_chunk_4"
    
    tracemalloc.start()
    
    # test_memmap()
    test_shared_memory()
    # test_ray()
    
    print(f"Memory Cost: {tracemalloc.get_traced_memory()}")
    tracemalloc.stop()
    print(f"Time cost: {time.time()-start_time}")