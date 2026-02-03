#pragma once

#include <cuda_fp16.h>

typedef union __align__(16) _union_half_4
{
    uint2 in;
    half out[4];
    half2 out2[2];
}
union_half4;

typedef union __align__(16) _union_half_8
{
    uint4 in;
    half out[8];
    half2 out2[4];
}
union_half8;


template <typename T>
__global__ void __cuda_add_kernel(T* output, T* a, T* b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            output[idx] = a[idx] + b[idx];
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            output[idx] = __float2half( __half2float(a[idx]) + __half2float(b[idx]));
        }
    }
}

__global__ void __cuda_add_inplace_fp32_fp16(float* a, half* bias, int n);

__global__ void __cuda_fp32_to_fp16(float* a, half* b, int n);

__global__ void __cuda_fp16_to_fp32(half* a, float* b, int n);

template <typename T>
__global__ void __cuda_embedding_kernel(T* output, int* input, T* weight, int hidden_size)
{
    input += blockIdx.x;
    output += blockIdx.x * hidden_size;
    weight += *input * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += 128)
    {
        output[i] = weight[i];
    }
}

template <int block_dim>
__global__ void __cuda_RMSNorm_fp16(half* output, half* input, half* weight, int nums, int dims, float eps, float inv_dims)
{
    const int ti = blockIdx.x;
    if (ti >= nums) return;

    input += ti * dims;
    output += ti * dims;

    const int di = threadIdx.x * 8;
    float sum = 0;

    union_half8 vec;
    for (int i = di; i < dims; i += block_dim * 8)
    {
        vec.in = *reinterpret_cast<uint4*>(input + i);
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
            float val = __half2float(vec.out[v]);
            sum += val * val;
        }
    }

    float thread_sum = sum;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

 
    __shared__ float warp_sums[32];
    __shared__ float scale;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (lane_id == 0)
    {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x < 32)
    {
        float total_sum = 0.0f;
        const int num_warps = (block_dim + 31) / 32;
        if (threadIdx.x < num_warps)
        {
            total_sum = warp_sums[threadIdx.x];
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
        }

        if (threadIdx.x == 0)
        {
            scale = rsqrtf(total_sum * inv_dims + eps);
        }
    }
    __syncthreads();

    // __shared__ float shared_sum[block_dim];
    // shared_sum[threadIdx.x] = sum;
    // __syncthreads();

    // if constexpr (block_dim >= 1024)
    // {
    //     if (threadIdx.x < 512) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 512];
    //     __syncthreads();
    // }
    // if constexpr (block_dim >= 512)
    // {
    //     if (threadIdx.x < 256) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 256];
    //     __syncthreads();
    // }
    // if constexpr (block_dim >= 256)
    // {
    //     if (threadIdx.x < 128) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 128];
    //     __syncthreads();
    // }
    // if constexpr (block_dim >= 128)
    // {
    //     if (threadIdx.x < 64) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 64];
    //     __syncthreads();
    // }

    // float val = shared_sum[threadIdx.x];

    // if (threadIdx.x < 32)
    // {
    //     volatile float* smem = shared_sum;
    //     if constexpr (block_dim >= 64) smem[threadIdx.x] += smem[threadIdx.x + 32];
    //     if constexpr (block_dim >= 32) smem[threadIdx.x] += smem[threadIdx.x + 16];
    //     if constexpr (block_dim >= 16) smem[threadIdx.x] += smem[threadIdx.x + 8];
    //     if constexpr (block_dim >= 8) smem[threadIdx.x] += smem[threadIdx.x + 4];
    //     if constexpr (block_dim >= 4) smem[threadIdx.x] += smem[threadIdx.x + 2];
    //     if constexpr (block_dim >= 2) smem[threadIdx.x] += smem[threadIdx.x + 1];

    //     float val = shared_sum[threadIdx.x];
    // }

    // __shared__ float scale;
    // if (threadIdx.x == 0)
    // {
    //     scale = rsqrtf(total_sum * inv_dims + eps);
    // }
    // __syncthreads();

    _union_half_8 weight_vec;
    for (int i = di; i < dims; i += block_dim * 8)
    {
        // Load(vec, &input[i]);
        // Load(weight_vec, &weight[i]);
        vec.in = *reinterpret_cast<uint4*>(input + i);
        weight_vec.in = *reinterpret_cast<uint4*>(weight + i);
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
            float x = __half2float(vec.out[v]);
            float w = __half2float(weight_vec.out[v]);
            vec.out[v] = __float2half(x * scale * w);
        }

        // Store(&output[i], vec);
        *reinterpret_cast<uint4*>(&output[i]) = vec.in;
    }
}



__global__ void __cuda_rope_fp16(half* output,  half* input, int* position_ids, int head_num, int head_dim, int rotation_size, const float* __restrict__ cos_table,
               const float*  __restrict__ sin_table);

__global__ void __cuda_swiglu_fp16(half* output, half* gate,  half* up, int n);


__global__ void __cuda_bf16_to_fp16_inplace(half* buffer, size_t count);