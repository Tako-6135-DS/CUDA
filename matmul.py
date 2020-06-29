import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
from string import Template

import timeit

N = 128

code_kernel_temp = Template("""
__global__ void matmul_kernel(float *d_C, float *d_A, float *d_B)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.f;
    for (int e = 0; e < ${MATRIX_SIZE}; e++)
        sum += d_A[idx_y * ${MATRIX_SIZE} + e] * d_B[e * ${MATRIX_SIZE} + idx_x];
    d_C[idx_y * ${MATRIX_SIZE} + idx_x] = sum;
}
""")

def matmul(A, B):
    C = np.zeros(N*N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i * N + j] += A[i * N + k] * B[k * N + j]
    return C

np.random.seed(42)
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)
A = np.reshape(A, (1, -1))[0]
B = np.reshape(B, (1, -1))[0]
C = np.reshape(C, (1, -1))[0]

mod = compiler.SourceModule( \
        code_kernel_temp.substitute(MATRIX_SIZE=N))

matmul_kernel = mod.get_function("matmul_kernel")

dimBlock = 16
dimGrid = int((N + dimBlock - 1) / dimBlock)

start = driver.Event()
stop = driver.Event()

start.record()
matmul_kernel(driver.Out(C), driver.In(A), driver.In(B), block=(dimBlock, dimBlock, 1), grid=(dimGrid, dimGrid))
stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print("Time GPU: %.3f ms" % (gpu_time))

start = timeit.default_timer()
c_cpu = matmul(A, B)
cpu_time = timeit.default_timer() - start
print("Time CPU: %.3f ms" % (cpu_time * 1e3))