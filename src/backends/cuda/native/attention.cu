#include "attention.cuh"

__host__ __device__ __forceinline__ int ceildiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__global__ void __cuda_store_kvcache_fp16(const half* __restrict__ key_ptr, const half* __restrict__ value_ptr, half* __restrict__ k_cache_ptr,
                                          half* __restrict__ v_cache_ptr, const int* __restrict__ slot_mapping_ptr, int head_dim)
{
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int num_kv_heads = gridDim.y;

    int slot_idx = slot_mapping_ptr[token_idx];
    if (slot_idx == -1)
    {
        return;
    }

    for (int dim_offset = threadIdx.x; dim_offset < head_dim; dim_offset += blockDim.x)
    {
        int input_offset = token_idx * num_kv_heads * head_dim + head_idx * head_dim + dim_offset;

        int cache_offset = slot_idx * num_kv_heads * head_dim + head_idx * head_dim + dim_offset;

        k_cache_ptr[cache_offset] = key_ptr[input_offset];
        v_cache_ptr[cache_offset] = value_ptr[input_offset];
    }
}

/**
 * @brief 变长序列+GQA+因果掩码的FlashAttention CUDA核函数
 * @param Q         查询矩阵，shape=(total_tokens, num_heads, head_dim)
 * @param K         键矩阵，shape=(total_tokens, num_kv_heads, head_dim)
 * @param V         值矩阵，shape=(total_tokens, num_kv_heads, head_dim)
 * @param O         输出矩阵，shape=(total_tokens, num_heads, head_dim
 * @param cu_seqlens 累积序列长度数组，shape=(num_seqs + 1)，int32_t类型
 * @param scale     注意力缩放因子（1/sqrt(head_dim)）
 * @param num_heads Q头数量
 * @param num_kv_heads KV头数量
 * @param total_tokens 所有序列的token总数
 * @param num_seqs 序列数量
 **/
__global__ void __cuda_flash_attention_prefill_fp16(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
                                                    half* __restrict__ O, const int* __restrict__ cu_seqlens, float scale, int num_heads,
                                                    int head_dim, int num_kv_heads, int num_seqs, int BLOCK_M, int BLOCK_N)
{
    extern __shared__ half smem[];
    half* s_q = smem;                                // (BLOCK_M, head_dim)
    half* s_k = s_q + BLOCK_M * head_dim;            // (BLOCK_N, head_dim)
    float* s_s = (float*)(s_k + head_dim * BLOCK_N); // (BLOCK_M, BLOCK_N) float32（分数）
    half* s_v = (half*)(s_s + BLOCK_M * BLOCK_N);    // (BLOCK_N, head_dim)
    half* s_o = s_v + BLOCK_N * head_dim;            // (BLOCK_M, head_dim)
    float* s_l = (float*)(s_o + BLOCK_M * head_dim); //(BLOCK_M)
    float* s_m = s_l + BLOCK_M;                      //(BLOCK_M)
    float* s_expmax = s_m + BLOCK_M;                 //(BLOCK_M)

    int q_block_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.z;
    int dim_m = blockDim.x;
    int dim_n = blockDim.y;
    int tid_m = threadIdx.x;
    int tid_n = threadIdx.y;

    if (seq_idx >= num_seqs || head_idx >= num_heads) return;

    int seq_start = cu_seqlens[seq_idx];
    int seq_end = cu_seqlens[seq_idx + 1];
    int seq_len = seq_end - seq_start;
    if (seq_len <= 0) return;

    int q_block_start = q_block_idx * BLOCK_M;
    if (q_block_start >= seq_len) return;

    int heads_per_kv_head = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / heads_per_kv_head;
    if (kv_head_idx >= num_kv_heads) return;

    for (int i = tid_n * dim_m + tid_m; i < BLOCK_M; i += dim_m * dim_n)
    {
        s_m[i] = -INFINITY;
        s_l[i] = 0.f;
    }

    for (int i = tid_n; i < BLOCK_M; i += dim_n)
    {
        for (int j = tid_m; j < head_dim; j += dim_m)
        {
            s_o[i * head_dim + j] = __float2half(0.f);
        }
    }

    for (int i = tid_n; i < BLOCK_M; i += dim_n)
    {
        int token_global = seq_start + q_block_start + i;
        for (int j = tid_m; j < head_dim; j += dim_m)
        {
            if (token_global < seq_end)
            {
                size_t offset = token_global * num_heads * head_dim + head_idx * head_dim + j;
                s_q[i * head_dim + j] = Q[offset];
            }
        }
    }
    __syncthreads();

    int num_kv_blocks = ceildiv(seq_len, BLOCK_N);
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++)
    {
        int kv_block_start = kv_block_idx * BLOCK_N;
        int kv_token_global = seq_start + kv_block_start;

        for (int i = tid_n; i < BLOCK_N; i += dim_n)
        {
            int token_global = kv_token_global + i;
            for (int j = tid_m; j < head_dim; j += dim_m)
            {
                if (token_global < seq_end)
                {
                    size_t offset = token_global * num_kv_heads * head_dim + kv_head_idx * head_dim + j;
                    s_k[i * head_dim + j] = K[offset];
                    s_v[i * head_dim + j] = V[offset];
                }
            }
        }

        __syncthreads();
        for (int i = tid_n; i < BLOCK_M; i += dim_n)
        {
            for (int j = tid_m; j < BLOCK_N; j += dim_m)
            {
                float qk_score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                {
                    qk_score += __half2float(s_q[i * head_dim + d]) * __half2float(s_k[j * head_dim + d]);
                }
                qk_score *= scale;

                int q_pos = q_block_start + i;
                int kv_pos = kv_block_start + j;
                bool causal_valid = (kv_pos <= q_pos) && (q_pos < seq_len) && (kv_pos < seq_len);
                if (!causal_valid)
                {
                    qk_score = -INFINITY;
                }
                s_s[i * BLOCK_N + j] = qk_score;
            }
        }
        __syncthreads();

        for (int i = tid_n; i < BLOCK_M; i += dim_n)
        {
            float row_max = -INFINITY;
            for (int j = tid_m; j < BLOCK_N; j += dim_m)
            {
                row_max = fmaxf(row_max, s_s[i * BLOCK_N + j]);
            }
            row_max = warpReduceMax(row_max);

            if (tid_m == 0)
            {
                row_max = fmaxf(s_m[i], row_max);
            }

            row_max = __shfl_sync(0xffffffff, row_max, 0);

            for (int j = tid_m; j < BLOCK_N; j += dim_m)
            {
                s_s[i * BLOCK_N + j] = expf(s_s[i * BLOCK_N + j] - row_max);
            }

            if (tid_m == 0)
            {
                s_expmax[i] = expf(s_m[i] - row_max);
                s_m[i] = row_max;
            }
        }

        __syncthreads();

        for (int i = tid_n; i < BLOCK_M; i += dim_n)
        {
            float row_sum = 0;
            for (int j = tid_m; j < BLOCK_N; j += dim_m)
            {
                row_sum += s_s[i * BLOCK_N + j];
            }

            row_sum = warpReduceSum(row_sum);

            if (tid_m == 0)
            {
                s_l[i] = fmaxf(s_l[i] * s_expmax[i] + row_sum, 1e-7f);
            }

            for (int j = tid_m; j < head_dim; j += dim_m)
            {
                float pv = 0.f;
                for (int k = 0; k < BLOCK_N; k++)
                {
                    pv += s_s[i * BLOCK_N + k] * __half2float(s_v[k * head_dim + j]);
                }
                s_o[i * head_dim + j] = __float2half(__half2float(s_o[i * head_dim + j]) * s_expmax[i] + pv);
            }
        }

        __syncthreads();
    }

    for (int i = tid_n; i < BLOCK_M; i += dim_n)
    {
        for (int j = tid_m; j < head_dim; j += dim_m)
        {
            s_o[i * head_dim + j] = __float2half(__half2float(s_o[i * head_dim + j]) / s_l[i]);
        }

        if (tid_m == 0)
        {
            s_l[i] = s_m[i] + logf(s_l[i]);
        }
    }
    __syncthreads();

    for (int i = tid_n; i < BLOCK_M; i += dim_n)
    {
        int token_global = seq_start + q_block_start + i;
        for (int j = tid_m; j < head_dim; j += dim_m)
        {
            if (token_global < seq_end)
            {
                size_t offset = token_global * num_heads * head_dim + head_idx * head_dim + j;
                O[offset] = s_o[i * head_dim + j];
            }
        }
    }
}

__inline__ __device__ float blockReduceSum(float val)
{
    val = warpReduceSum(val);

    __shared__ float warp_sums[32];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + 31) / 32;
        float thread_val = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.0f;
        thread_val = warpReduceSum(thread_val);
        if (lane_id == 0) warp_sums[0] = thread_val;
    }
    __syncthreads();

    return warp_sums[0];
}

__global__ void paged_attention_decode_kernel(const half* __restrict__ query_ptr,       // [batch_size, num_heads, head_dim]
                                              const half* __restrict__ k_cache_ptr,     // [num_blocks, block_size, num_kv_heads, head_dim]
                                              const half* __restrict__ v_cache_ptr,     // [num_blocks, block_size, num_kv_heads, head_dim]
                                              half* __restrict__ output_ptr,            // [batch_size, num_heads, head_dim]
                                              const int* __restrict__ block_tables_ptr, // [batch_size, max_num_blocks]
                                              const int* __restrict__ context_lens_ptr, // [batch_size]
                                              const float scale, const int num_heads, const int num_kv_heads, const int head_dim,
                                              const int block_size, const int max_num_blocks)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    extern __shared__ float s_query[];

    int q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + tid;
    float q_val = 0.0f;
    if (tid < head_dim)
    {
        q_val = __half2float(query_ptr[q_offset]);
        s_query[tid] = q_val;
    }
    __syncthreads();

    float acc = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;

    int context_len = context_lens_ptr[batch_idx];

    int num_logical_blocks = (context_len + block_size - 1) / block_size;

    for (int lb = 0; lb < num_logical_blocks; ++lb)
    {
        int block_table_offset = batch_idx * max_num_blocks + lb;
        int physical_block_idx = block_tables_ptr[block_table_offset];

        int start_offset = 0;
        int end_offset = block_size;
        if (lb == num_logical_blocks - 1)
        {
            end_offset = (context_len - 1) % block_size + 1;
        }

        for (int offset = start_offset; offset < end_offset; ++offset)
        {
            size_t k_offset = (size_t)physical_block_idx * block_size * num_kv_heads * head_dim + (size_t)offset * num_kv_heads * head_dim +
                              (size_t)kv_head_idx * head_dim + tid;

            float k_val = (tid < head_dim) ? __half2float(k_cache_ptr[k_offset]) : 0.0f;

            float dot = q_val * k_val;

            float score = blockReduceSum(dot);

            __shared__ float s_score;
            if (tid == 0)
            {
                s_score = score * scale;
            }
            __syncthreads();
            float current_score = s_score;

            float m_i_new = fmaxf(m_i, current_score);

            float alpha = expf(m_i - m_i_new);

            float p = expf(current_score - m_i_new);

            m_i = m_i_new;

            l_i = l_i * alpha + p;

            size_t v_offset = (size_t)physical_block_idx * block_size * num_kv_heads * head_dim + (size_t)offset * num_kv_heads * head_dim +
                              (size_t)kv_head_idx * head_dim + tid;

            float v_val = (tid < head_dim) ? __half2float(v_cache_ptr[v_offset]) : 0.0f;

            acc = acc * alpha + p * v_val;
        }
    }

    if (tid < head_dim)
    {
        float out_val = acc / l_i;
        int out_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + tid;
        output_ptr[out_offset] = __float2half(out_val);
    }
}