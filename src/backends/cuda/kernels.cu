#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#include "kernels.cuh"
#include "native/sgemm.cuh"
#include "cublas/sgemm.cuh"
#include "native/common.cuh"
#include "native/attention.cuh"
#include "utils/precision.hpp"

namespace spyinfer {

#define CHECK_CUDA_ERROR(op) check((op), #op, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void cuda_add_fp32(float* output, float* a, float* b, int n)
{
    int threadsPerBlock = std::min(n, 256);
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    __cuda_add_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(output, a, b, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_add_fp16(uint16_t* output, uint16_t* a, uint16_t* b, int n)
{
    int threadsPerBlock = std::min(n, 256);
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    __cuda_add_kernel<half><<<blocksPerGrid, threadsPerBlock>>>(reinterpret_cast<half*>(output), reinterpret_cast<half*>(a), reinterpret_cast<half*>(b), n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_linear_fp32_fp16(float* output, float* input, uint16_t* weight, uint16_t* bias, int m, int k, int n, float alpha, float beta)
{
    if (m < 8)
    {
        __cuda_linear_fp32_fp16_m8(output, input, reinterpret_cast<half*>(weight), bias ? reinterpret_cast<half*>(bias) : nullptr, m, k, n);
    }
    else
    {
        __cuda_linear_fp32_fp16_cublas(output, input, reinterpret_cast<half*>(weight), bias ? reinterpret_cast<half*>(bias) : nullptr, m, k, n, alpha,
                                       beta);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
}


void cuda_linear_fp16_fp16(uint16_t* output, uint16_t* input, uint16_t* weight, uint16_t* bias, int m, int k, int n, float alpha, float beta)
{

    if (m < 8)
    {
        __cuda_linear_fp16_fp16_m8(reinterpret_cast<half*>(output), reinterpret_cast<half*>(input), reinterpret_cast<half*>(weight), bias ? reinterpret_cast<half*>(bias) : nullptr, m, k, n);
    }
    else
    {
        __cuda_linear_fp16_cublas(reinterpret_cast<half*>(output), reinterpret_cast<half*>(input), reinterpret_cast<half*>(weight), m, k, n, alpha);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_embedding_fp32(float* output, int* input, float* weight, int batch_size, int hidden_size)
{
    __cuda_embedding_kernel<float><<<batch_size, 128>>>(output, input, weight, hidden_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_embedding_fp16(uint16_t* output, int* input, uint16_t* weight, int batch_size, int hidden_size)
{
    __cuda_embedding_kernel<half><<<batch_size, 128>>>(reinterpret_cast<half*>(output), input, reinterpret_cast<half*>(weight), hidden_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_rms_norm_fp16(uint16_t* output, uint16_t* input, uint16_t* weight, int nums, int len, float eps)
{
    __cuda_RMSNorm_fp16<256>
        <<<nums, 256>>>(reinterpret_cast<half*>(output), reinterpret_cast<half*>(input), reinterpret_cast<half*>(weight), nums, len, eps, 1.0f / len);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_rope_fp16(uint16_t* output, uint16_t* input, int* position_ids, int batch_size, int num_heads, int head_size, float* cos_table,
                    float* sin_table)
{

    __cuda_rope_fp16<<<dim3(batch_size, num_heads), head_size / 2>>>(reinterpret_cast<half*>(output), reinterpret_cast<half*>(input), position_ids,
                                                                num_heads, head_size, head_size / 2, cos_table, sin_table);
    CHECK_CUDA_ERROR(cudaGetLastError());
   
}

void cuda_swiglu_fp16(uint16_t* output, uint16_t* gate, uint16_t* up, int n)
{
    const int block_dim = std::min(n, 512);
    __cuda_swiglu_fp16<<<(n - 1) / block_dim + 1, block_dim>>>(reinterpret_cast<half*>(output), reinterpret_cast<half*>(gate),
                                                               reinterpret_cast<half*>(up), n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_store_kvcache_fp16(uint16_t* k, uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, int* slot_mapping, int num_tokens, int num_kv_heads,
                             int head_dim)
{
    __cuda_store_kvcache_fp16<<<dim3(num_tokens, num_kv_heads), 128>>>(reinterpret_cast<const half*>(k), reinterpret_cast<const half*>(v),
                                                                       reinterpret_cast<half*>(k_cache), reinterpret_cast<half*>(v_cache),
                                                                       slot_mapping, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_flash_attention_prefill_fp16(uint16_t* o, uint16_t* q, uint16_t* k, uint16_t* v, int* cu_seqlens, int num_seqs, int max_seq_len,
                                        int num_heads, int num_kv_heads, int head_dim)
{
    int BLOCK_M = 0, BLOCK_N = 0;

    if (head_dim <= 64)
    {
        BLOCK_M = 64;
        BLOCK_N = 64;
    }
    else if (head_dim <= 128)
    {
        BLOCK_M = 32;
        BLOCK_N = 32;
    }
    else
    {
        BLOCK_M = 16;
        BLOCK_N = 16;
    }

    dim3 grid(((max_seq_len + BLOCK_M - 1) / BLOCK_M), num_heads, num_seqs);
    dim3 blockDim(32, 4);

    //int smem_size = (BLOCK_M * BLOCK_N + BLOCK_M * head_dim + BLOCK_N * head_dim * 2) * 4;
    size_t smem_size = 0;

    smem_size += BLOCK_M * head_dim * sizeof(half);
    smem_size += head_dim * BLOCK_N * sizeof(half); 
    smem_size += BLOCK_M * BLOCK_N * sizeof(float); 
    smem_size += BLOCK_N * head_dim * sizeof(half);
    smem_size += BLOCK_M * head_dim * sizeof(half);
    smem_size += BLOCK_M * sizeof(float);
    smem_size += BLOCK_M * sizeof(float);
    smem_size += BLOCK_M * sizeof(float); 

    
    smem_size = ((smem_size + 15) / 16) * 16;

    __cuda_flash_attention_prefill_fp16<<<grid, blockDim, smem_size>>>(
        reinterpret_cast<const half*>(q), reinterpret_cast<const half*>(k), reinterpret_cast<const half*>(v), reinterpret_cast<half*>(o),
        reinterpret_cast<const int*>(cu_seqlens), 1.f / std::sqrt(head_dim), num_heads, head_dim, num_kv_heads, num_seqs, BLOCK_M, BLOCK_N);

    CHECK_CUDA_ERROR(cudaGetLastError());
   
}

void cuda_paged_attention_decode_fp16(uint16_t* o, uint16_t* q, uint16_t* k_cache, uint16_t* v_cache, int* block_tables, int* context_lens,
                                      int batch_size, int num_heads, int num_kv_heads, int head_dim, int block_size, int max_num_blocks)
{
    dim3 grid(batch_size, num_heads);
    // 假设 head_dim <= 1024，通常为 64 或 128
    // 为了对齐 warp，取 32 的倍数
    int block_dim_x = (head_dim + 31) / 32 * 32;
    dim3 block(block_dim_x);

    // 动态共享内存大小: 存储 Q 向量
    size_t shared_mem_size = head_dim * 4;

    paged_attention_decode_kernel<<<grid, block, shared_mem_size>>>(
        reinterpret_cast<const half*>(q), reinterpret_cast<const half*>(k_cache), reinterpret_cast<const half*>(v_cache), reinterpret_cast<half*>(o),
        reinterpret_cast<const int*>(block_tables), reinterpret_cast<const int*>(context_lens), 1.f / std::sqrt(head_dim), num_heads, num_kv_heads,
        head_dim, block_size, max_num_blocks);

    CHECK_CUDA_ERROR(cudaGetLastError());
   

}

void cuda_bf16_to_fp16_inplace(uint16_t* buffer, size_t count)
{
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;

    // 4. 启动Kernel
    __cuda_bf16_to_fp16_inplace<<<grid_size, block_size>>>(reinterpret_cast<half*>(buffer), count);
    CHECK_CUDA_ERROR(cudaGetLastError());
   

}

} // namespace spyinfer
