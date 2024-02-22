import numpy as np 

dstore_mmap = "/home/jqcao/projects/memory_transformer/LongMem/key_value_dstore/5e-5_best"

write = np.memmap(dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+', shape=(10,10))
