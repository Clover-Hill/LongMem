import argparse
import os
import numpy as np
import faiss
import logging
import multiprocessing
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
    parser.add_argument('--chunk_size', default=4, type=str, help='how many tokens form a chunk')
    args = parser.parse_args()

    print(args)
    return args

def read_dstore(args):
    # Read data store
    if args.dstore_fp16:
        keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
        vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    else:
        keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
        vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    
    return keys,vals

def build_index(args,head_id,head_dim):
    index_filename = os.path.join(args.save_dir,f"{head_id}.index") 
    keys_filename = os.path.join(args.save_dir,f"{head_id}_keys.npy")
    vals_filename = os.path.join(args.save_dir,f"{head_id}_vals.npy")
    if os.path.exists(index_filename):
        return
    
    logger.info(f"Start building index for head {head_id}")
    # Get np_keys and np_vals
    global original_keys
    global original_vals
    np_keys = np.array(original_keys[:,i,:])
    np_vals = np.array(original_vals[:,i,:])
    
    # Save split keys and values 
    assert(np_keys.shape==(args.dstore_size,head_dim))
    assert(np_vals.shape==(args.dstore_size,head_dim))
    new_dstore_size = args.dstore_size // args.chunk_size

    if args.dstore_fp16:
        keys = np.memmap(keys_filename, dtype=np.float16, mode='w+', shape=(new_dstore_size, head_dim))
        vals = np.memmap(vals_filename, dtype=np.float16, mode='w+', shape=(new_dstore_size, head_dim))
    else:
        keys = np.memmap(keys_filename, dtype=np.float32, mode='w+', shape=(new_dstore_size, head_dim))
        vals = np.memmap(vals_filename, dtype=np.float32, mode='w+', shape=(new_dstore_size, head_dim))

    # Using Chunk Size
    keep_dimension = args.dstore_size // args.chunk_size * args.chunk_size
    start_time = time.time()
    logger.info(f"Starting chunking for head {head_id}")
    np_keys = np_keys[:keep_dimension,:].reshape(args.dstore_size // args.chunk_size, args.chunk_size, head_dim)     
    np_vals = np_vals[:keep_dimension,:].reshape(args.dstore_size // args.chunk_size, args.chunk_size, head_dim)     
    np_keys = np_keys.mean(axis=-2)
    np_vals = np_vals.mean(axis=-2)
    assert(np_keys.shape==(args.dstore_size // args.chunk_size, head_dim))
    assert(np_vals.shape==(args.dstore_size // args.chunk_size, head_dim))
    logger.info(f"Chunking cost {time.time()-start_time}")
    
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

        logger.info(f'Training Index for head {head_id}')
        np.random.seed(args.seed)
        random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
        start = time.time()

        # Faiss does not handle adding keys in fp16 as of writing this.
        index.train(keys[random_sample].astype(np.float32))
        logger.info('Training took {} s'.format(time.time() - start))

        logger.info(f'Writing index to {index_filename}.trained after training')
        start = time.time()
        faiss.write_index(index, index_filename+".trained")
        logger.info('Writing index took {} s'.format(time.time()-start))

    # Continue adding keys
    logger.info(f'Adding Keys for head {head_id}')
    index = faiss.read_index(index_filename+".trained")
    start = 0
    start_time = time.time()
    
    progress_bar = tqdm(total=new_dstore_size//args.num_keys_to_add_at_a_time+1, desc='Processing')
    while start < new_dstore_size:
        end = min(new_dstore_size, start+args.num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()
        index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
        start += args.num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            logger.info('Added %d tokens so far' % start)
            logger.info(f'Writing Index {start}')
            faiss.write_index(index, index_filename)
            
        progress_bar.update(1)
    progress_bar.close()

    # Writing index
    logger.info("Adding total %d keys" % start)
    logger.info('Adding took {} s'.format(time.time() - start_time))
    logger.info('Writing Index')
    start = time.time()
    faiss.write_index(index, index_filename)
    logger.info('Writing index took {} s'.format(time.time()-start))

if __name__ == '__main__':
    args = parse_args()
    
    # Set up logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(processName)s - %(message)s",
        filename=os.path.join(args.save_dir,"build_dstore.log"),
        filemode="w"
    )
    logger = logging.getLogger(__name__)

    original_keys, original_vals = read_dstore(args) 

    dim = args.dimension
    n_head = args.n_head
    head_dim = dim // n_head 
    
    # Reshape keys and vals 
    new_shape = original_keys.shape[:-1] + (n_head, head_dim)
    original_keys = original_keys.reshape(new_shape)
    original_vals = original_vals.reshape(new_shape)
    
    # For the first half of heads
    params = []
    
    for i in range(n_head//2):
        params.append((args, i, head_dim))
        
    num_processes = multiprocessing.cpu_count()
    pool =  multiprocessing.Pool(processes=num_processes)
    pool.starmap(build_index, params)
    
    pool.close()
    pool.join()
    
    # For the second half of heads
    params = [(args,1,head_dim)]
    
    for i in range(n_head//2, n_head):
        params.append((args, i, head_dim))
        
    pool =  multiprocessing.Pool(processes=num_processes)
    pool.starmap(build_index, params)
    
    pool.close()
    pool.join()