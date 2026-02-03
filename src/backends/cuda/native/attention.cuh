#pragma once

#include <cuda_fp16.h>



__global__ void __cuda_store_kvcache_fp16(const half* __restrict__ key_ptr, const half* __restrict__ value_ptr, half* __restrict__ k_cache_ptr,
                                          half* __restrict__ v_cache_ptr, const int* __restrict__ slot_mapping_ptr, int head_dim);


__global__ void __cuda_flash_attention_prefill_fp16(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
                                                    half* __restrict__ O, const int* __restrict__ cu_seqlens, float scale, int num_heads,
                                                    int head_dim, int num_kv_heads, int num_seqs, int BLOCK_M, int BLOCK_N);


__global__ void paged_attention_decode_kernel(
    const half* __restrict__ query_ptr,      // [batch_size, num_heads, head_dim]
    const half* __restrict__ k_cache_ptr,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const half* __restrict__ v_cache_ptr,    // [num_blocks, block_size, num_kv_heads, head_dim]
    half* __restrict__ output_ptr,           // [batch_size, num_heads, head_dim]
    const int* __restrict__ block_tables_ptr, // [batch_size, max_num_blocks]
    const int* __restrict__ context_lens_ptr, // [batch_size]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_num_blocks
);


