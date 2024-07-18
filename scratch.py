# %%
import numpy as np
from utils import Vector3D, Quaternion4D
# %%
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a = np.random.rand(3)
A = Vector3D(a)
b = 5*np.random.rand(3,2,4)
B = Quaternion4D(b)
print(B)
B._normalize()
print(B)
# %%
B*B
# %%
