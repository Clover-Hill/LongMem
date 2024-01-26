import numpy as np 

dstore_mmap = "/home/jqcao/projects/memory_transformer/LongMem/external_memory/wiki103"
dstore_keys = np.memmap(dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+', shape=(10,10000))
