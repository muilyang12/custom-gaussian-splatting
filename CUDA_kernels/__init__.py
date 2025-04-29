import cupy as cp

dot_product_kernel = cp.RawKernel(r'''
extern "C"
__global__ void dot_product(const float* a, const float* b, float* result, int n) {
    __shared__ float temp[4];

    int tid = threadIdx.x;

    if (tid < n) temp[tid] = a[tid] * b[tid];
    else temp[tid] = 0.0f;

    __syncthreads();

    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = temp[0];
    }
}
''', 'dot_product')