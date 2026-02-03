#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace spyinfer {

void cuda_add_fp32(float* output, float* a, float* b, int n);

void cuda_add_fp16(uint16_t* output, uint16_t* a, uint16_t* b, int n);


void cuda_linear_fp32_fp16(float* output, float* input, uint16_t* weight, uint16_t* bias, int m, int k, int n, float alpha = 1.0f, float beta = 1.0f);

void cuda_linear_fp16_fp16(uint16_t* output, uint16_t* input, uint16_t* weight, uint16_t* bias, int m, int k, int n, float alpha = 1.0f, float beta = 1.0f);



void cuda_embedding_fp32(float* output, int* input, float* weight, int batch_size, int hidden_size);
void cuda_embedding_fp16(uint16_t* output, int* input, uint16_t* weight, int batch_size, int hidden_size);

void cuda_rms_norm_fp16(uint16_t* output, uint16_t* input, uint16_t* weight, int nums, int len, float eps);
// void cuda_rms_norm(float* output, const float* input, const uint16_t* weight, int batch_size, int hidden_size, float epsilon);

void cuda_rope_fp16(uint16_t* output, uint16_t* input, int* position_ids, int batch_size, int num_heads, int head_size, float* cos_table,
                    float* sin_table);

void cuda_swiglu_fp16(uint16_t* output, uint16_t* gate, uint16_t* up, int n);

void cuda_store_kvcache_fp16(uint16_t* k, uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, int* slot_mapping, int num_tokens, int num_kv_heads,
                             int head_dim);

void cuda_flash_attention_prefill_fp16(uint16_t* o, uint16_t* q, uint16_t* k, uint16_t* v, int* cu_seqlens, int num_seqs, int max_seq_len,
                                       int num_heads, int num_kv_heads, int head_dim);

void cuda_paged_attention_decode_fp16(uint16_t* o, uint16_t* q, uint16_t* k_cache, uint16_t* v_cache, int* block_tables, int* context_lens,
                                      int batch_size, int num_heads, int num_kv_heads, int head_dim, int block_size, int max_num_blocks);

void cuda_bf16_to_fp16_inplace(uint16_t* buffer, size_t count);

} // namespace spyinfer
