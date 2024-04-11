import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import time


mod = SourceModule("""
#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    int stride;
    int __padding; //In order to align with the 64-bit elements pointer
    float* elements;
} Matrix;

 // read matrix elements
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

 // Assign matrix elements
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

 // Get the 16x16 sub-matrix
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void matrix_mul(Matrix *A, Matrix *B, Matrix *C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    Matrix Csub = GetSubMatrix(*C, blockRow, blockCol);

         // Each thread calculates a value of Csub by accumulating Cvalue
    float Cvalue = 0;

         // In order to calculate Csub traverse all the required Asub and Bsub
    for (int m = 0; m < (A->width / BLOCK_SIZE); ++m)
    {
        Matrix Asub = GetSubMatrix(*A, blockRow, m);
        Matrix Bsub = GetSubMatrix(*B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}
""")


class MatrixStruct(object):
    def __init__(self, array):
        self._cptr = None

        self.shape, self.dtype = array.shape, array.dtype
        self.width = np.int32(self.shape[1])
        self.height = np.int32(self.shape[0])
        self.stride = self.width
        self.elements = cuda.to_device(array)  # allocate memory and copy array data to device, return its address

    def send_to_gpu(self):
        self._cptr = cuda.mem_alloc(self.nbytes())  # allocate the memory occupied by a C structure
        cuda.memcpy_htod(int(self._cptr), self.width.tobytes())  # Copy data to device, the same below
        cuda.memcpy_htod(int(self._cptr)+4, self.height.tobytes())
        cuda.memcpy_htod(int(self._cptr)+8, self.stride.tobytes())
        cuda.memcpy_htod(int(self._cptr)+16, np.intp(int(self.elements)).tobytes())

    def get_from_gpu(self):
        return cuda.from_device(self.elements, self.shape, self.dtype)  # Retrieve array data from device

    def nbytes(self):
        return self.width.nbytes * 4 + np.intp(0).nbytes


MATRIX_SIZE = 4000
a = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
c = np.zeros_like(a)

A = MatrixStruct(a)
B = MatrixStruct(b)
C = MatrixStruct(c)
A.send_to_gpu()
B.send_to_gpu()
C.send_to_gpu()

matrix_mul = mod.get_function("matrix_mul")

start = time.time()
matrix_mul(A._cptr, B._cptr, C._cptr, block=(16, 16, 1), grid=(25, 25))
result_gpu = C.get_from_gpu()
gpu_time = (time.time() - start) * 1000

start = time.time()
result_cpu = np.dot(a, b)
cpu_time = (time.time() - start) * 1000

print("Mean diff:", np.abs((result_cpu - result_gpu)).mean())
print(np.abs(result_gpu).sum())
print(np.abs(result_cpu).sum())

print("-" * 80)
print(f"GPU: {gpu_time} ms \nCPU: {cpu_time} ms")


