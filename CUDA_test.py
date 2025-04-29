import cupy as cp
from CUDA_kernels import dot_product_kernel

a = cp.array([1, 2, 3, 4], dtype=cp.float32)
b = cp.array([1, 2, 3, 4], dtype=cp.float32)
result = cp.zeros(1, dtype=cp.float32)

dot_product_kernel((1,), (4,), (a, b, result, 4))

print(result[0])