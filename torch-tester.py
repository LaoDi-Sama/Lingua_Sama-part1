import timeit
c = '''import torch

a = torch.cuda.is_available()
print(a)'''
f = timeit.timeit(c)
print(f)
