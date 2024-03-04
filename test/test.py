import torch
from torch import multiprocessing
a = torch.tensor([.3, .4, 1.2]).cuda()
print(a.is_shared())                       

a.share_memory_()

def test(a):
    print(a)                               
p = multiprocessing.Process(target=test, args=(a, ))
p.start()
p.join()
print(a)                                    