#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

__global__ void __cuda_add_inplace_fp32_fp16(float* a, half* bias, int n)
{
    float* now = a + blockIdx.x * n;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < n; i += stride)
    {
        now[i] += __half2float(bias[i]);
    }
}

__global__ void __cuda_fp32_to_fp16(float* a, half* b, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        b[idx] = __float2half_rz(a[idx]);
    }
}

__global__ void __cuda_fp16_to_fp32(half* a, float* b, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        b[idx] = __half2float(a[idx]);
    }
}

__global__ void __cuda_rope_fp16(half* output, half* input, int* position_ids, int head_num, int head_dim, int rotation_size,
                                 const float* __restrict__ cos_table, const float* __restrict__ sin_table)
{
    const int p = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int token_pos = position_ids[p];

    input += p * head_num * head_dim + head_idx * head_dim;
    output += p * head_num * head_dim + head_idx * head_dim;

    int j = threadIdx.x;
    if (j >= rotation_size) return;
    float v0 = __half2float(input[j]);
    float v1 = __half2float(input[j + rotation_size]);
    float cos_val = cos_table[token_pos * rotation_size + j];
    float sin_val = sin_table[token_pos * rotation_size + j];

    output[j] = __float2half(v0 * cos_val - v1 * sin_val);
    output[j + rotation_size] = __float2half(v0 * sin_val + v1 * cos_val);
}

__global__ void __cuda_swiglu_fp16(half* output, half* gate, half* up, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n)
    {
#ifdef CUDA_NO_TENSOR_CORE
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        output[idx] = __float2half((g / (1.0f + expf(-g))) * u);

#else
        half g = gate[idx];
        half u = up[idx];
        output[idx] = __hmul(__hdiv(g, __hadd(__float2half(1.0), hexp(-g))), u);
#endif
    }
}

__global__ void __cuda_bf16_to_fp16_inplace(half* buffer, size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count)
    {
        buffer[idx] = __float2half(__bfloat162float(((__nv_bfloat16*)buffer)[idx]));
    }
}