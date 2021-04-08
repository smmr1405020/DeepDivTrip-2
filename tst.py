import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])


b = np.array([[2,4,5],[3,2,6],[4,2,3]])


print(a/b)

b = np.tile(np.arange(1,3),(28,1))

print(b.T)