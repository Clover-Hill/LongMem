import argparse
import os
import numpy as np
import faiss
import time
from tqdm import tqdm

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
    parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
    parser.add_argument('--dimension', type=int, default=768, help='Size of each key')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--dstore_fp16', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
    parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
    parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
    parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
    parser.add_argument('--save_dir', type=str, help='directory to write the faiss index')
    parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
                        help='can only load a certain amount of data to memory at a time.')
    args = parser.parse_args()

    print(args)
    return args

def read_dstore(args):
    # Read data store
    if args.dstore_fp16:
        keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
        vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int16, mode='r', shape=(args.dstore_size, args.dimension))
    else:
        keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
        vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, args.dimension))
    
    return keys,vals

def build_index(args,np_keys,np_vals,head_id,head_dim):
    index_filename = os.path.join(args.save_dir,f"{head_id}.index") 
    keys_filename = os.path.join(args.save_dir,f"{head_id}_keys.npy")
    vals_filename = os.path.join(args.save_dir,f"{head_id}_vals.npy")
    
    # Save split keys and values 
    assert(np_keys.shape==(args.dstore_size,head_dim))
    assert(np_vals.shape==(args.dstore_size,head_dim))
    if args.dstore_fp16:
        keys = np.memmap(keys_filename, dtype=np.float16, mode='w+', shape=(args.dstore_size, head_dim))
        vals = np.memmap(vals_filename, dtype=np.int16, mode='w+', shape=(args.dstore_size, head_dim))
    else:
        keys = np.memmap(keys_filename, dtype=np.float32, mode='w+', shape=(args.dstore_size, head_dim))
        vals = np.memmap(vals_filename, dtype=np.int, mode='w+', shape=(args.dstore_size, head_dim))
    keys[:]=np_keys[:]
    vals[:]=np_vals[:]
    keys.flush()
    vals.flush()
    
    # Initialize faiss index
    if not os.path.exists(index_filename+".trained"):

        quantizer = faiss.IndexFlatL2(head_dim)
        index = faiss.IndexIVFPQ(quantizer, head_dim,
            args.ncentroids, args.code_size, 8)
        index.nprobe = args.probe

        print(f'Training Index for head {head_id}')
        np.random.seed(args.seed)
        random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
        start = time.time()

        # Faiss does not handle adding keys in fp16 as of writing this.
        index.train(keys[random_sample].astype(np.float32))
        print('Training took {} s'.format(time.time() - start))

        print(f'Writing index to {index_filename}.trained after training')
        start = time.time()
        faiss.write_index(index, index_filename+".trained")
        print('Writing index took {} s'.format(time.time()-start))

    # Continue adding keys
    print(f'Adding Keys for head {head_id}')
    index = faiss.read_index(index_filename+".trained")
    start = 0
    start_time = time.time()
    
    progress_bar = tqdm(total=args.dstore_size//args.num_keys_to_add_at_a_time+1, desc='Processing')
    while start < args.dstore_size:
        end = min(args.dstore_size, start+args.num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()
        index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
        start += args.num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            print('Added %d tokens so far' % start)
            print('Writing Index', start)
            faiss.write_index(index, index_filename)
            
        progress_bar.update(1)
    progress_bar.close()

    # Writing index
    print("Adding total %d keys" % start)
    print('Adding took {} s'.format(time.time() - start_time))
    print('Writing Index')
    start = time.time()
    faiss.write_index(index, index_filename)
    print('Writing index took {} s'.format(time.time()-start))

if __name__ == '__main__':
    args = parse_args()
    original_keys, original_vals = read_dstore(args) 
    
    dim = args.dimension
    n_head = args.n_head
    head_dim = dim // n_head 
    
    # Reshape keys and vals 
    new_shape = original_keys.shape()[:-1] + (n_head, head_dim)
    original_keys = original_keys.view(new_shape)
    original_vals = original_vals.view(new_shape)
    
    # Build index for each head
    for i in range(n_head):
        print(f"Building index for head {i}:")
        cur_keys = np.array(original_keys[...,i,:])
        cur_vals = np.array(original_vals[...,i,:])
        
        build_index(args, cur_keys, cur_vals, i, head_dim)
        print("-------------------------------------------")