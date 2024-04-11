import numpy as np
import time
from datetime import datetime
# import pycuda stuff
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

BLOCK_SIZE = 16

n = 8000
ni = np.int32(n)

# matrix A
a = np.random.randn(n, n)*100
a = a.astype(np.float32)
print(np.var(a))

# matrix B
b = np.random.randn(n, n)*100
b = b.astype(np.float32)
print(np.var(b))

# matrix B
c = np.empty([n, n])
c = c.astype(np.float32)

# allocate memory on device
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# copy matrix to memory
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# compile kernel
mod = SourceModule(open("kernels.cu", "r").read())

# get function
matmul = mod.get_function("matmul")


# set grid size
if (n % BLOCK_SIZE) != 0:
    grid = (n // BLOCK_SIZE + 1, n//BLOCK_SIZE + 1, 1)
else:
    grid = (n//BLOCK_SIZE, n//BLOCK_SIZE, 1)

# call gpu function
start = time.time()
start_d = datetime.now()
matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=grid);
# print("GPU Time: %.5f s" % (time.time() - start))
# print(f"GPU Time: {(time.time() - start) * 1000} ms")
print(f"GPU Time: {(datetime.now() - start_d).microseconds} mcs")

# copy back the result
cuda.memcpy_dtoh(c, c_gpu)

start = time.time()
start_d = datetime.now()
np.dot(a, b)
# print(f"CPU Time: {(time.time() - start) * 1000} ms")
print(f"CPU Time: {(datetime.now() - start_d).microseconds} mcs")

# print(np.linalg.norm(c - np.dot(a, b)))
# print(c)
# (np.dot(a, b))
print("Mean squared diff:", (np.abs(c - np.dot(a, b)) ** 2).mean())
