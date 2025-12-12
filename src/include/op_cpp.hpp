#pragma once

#include <cmath>
#include <immintrin.h>
#include <vector>
#include <cassert>
#include <algorithm>
#include <thread>
#include <omp.h>
#include "thread_pool.hpp"

#include "define.hpp"

namespace spyinfer {

template <typename T>
void _rms_norm_fp32(float* out, const float* input, const T* weight, std::size_t size, float eps)
{
    float norm = 0.f;
    for (int32_t i = 0; i < size; i++)
    {
        norm += input[i] * input[i];
    }
    norm = 1.f / std::sqrt(norm / size + eps);
    for (int32_t i = 0; i < size; i++)
    {
        out[i] = norm * input[i] * _cvt_to_fp32<T>(weight[i]);
    }
}

template <typename T, typename U>
void _gemm_native(float* out, const T* input1, const U* input2, std::size_t i, std::size_t j, std::size_t k)
{
    for (std::size_t x = 0; x < i; ++x)
    {
        for (std::size_t y = 0; y < k; ++y)
        {
            float sum = 0.0f; // 点积累加器
            for (std::size_t n = 0; n < j; ++n)
            {
                const T& a = input1[x * j + n];
                float a_fp32 = _cvt_to_fp32<T>(a);

                const U& b = input2[n * k + y];
                float b_fp32 = _cvt_to_fp32<U>(b);

                sum += a_fp32 * b_fp32;
            }
            out[x * k + y] = sum;
        }
    }
}

inline float dot_bf16_fp32_avx512(const bf16_t* a, const float* x, size_t k)
{
    assert(a != nullptr && x != nullptr && "Input pointers cannot be null");
    if (k == 0) return 0.0f; // 空向量直接返回0

    __m512 sum16 = _mm512_setzero_ps();    // 512位浮点累加寄存器（16个fp32）
    __m512i zero = _mm512_setzero_si512(); // 512位整数寄存器（bf16位扩展用）
    const size_t simd_step = 32;           // AVX512单次处理32个bf16元素
    const size_t rem = k % simd_step;      // 剩余元素数量（0 ≤ rem <32）
    const size_t main_loop_end = k - rem;  // 主循环的终止位置

    // 1. 向量化主循环：处理32的整数倍个元素
    for (size_t j = 0; j < main_loop_end; j += simd_step)
    {
        // 加载32个bf16元素到512位整数寄存器
        __m512i a32 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + j));

        __m256i a16_lo = _mm512_castsi512_si256(a32);       // 取低16个bf16元素
        __m256i a16_hi = _mm512_extracti64x4_epi64(a32, 1); // 取高16个bf16元素

        // 将16位无符号整数零扩展（zero-extend）到32位整数。
        // 这一步后，每个bf16数值占据了一个32位整数的低16位，高16位为0。
        __m512i b32_lo = _mm512_cvtepu16_epi32(a16_lo);
        __m512i b32_hi = _mm512_cvtepu16_epi32(a16_hi);

        // 核心转换步骤：通过对每个32位整数逻辑左移16位，将bf16的位模式转换为fp32的位模式。
        // bfloat16的16位包含了fp32的符号位、指数位和高位的尾数位。
        __m512i c32_lo = _mm512_slli_epi32(b32_lo, 16);
        __m512i c32_hi = _mm512_slli_epi32(b32_hi, 16);

        // 整数寄存器→浮点寄存器转换（无开销）
        __m512 a16_lo_f32 = _mm512_castsi512_ps(c32_lo);
        __m512 a16_hi_f32 = _mm512_castsi512_ps(c32_hi);

        // 加载32个fp32元素到512位浮点寄存器
        __m512 x16_lo = _mm512_loadu_ps(x + j);
        __m512 x16_hi = _mm512_loadu_ps(x + j + 16);

        // 512位向量化乘积累加
        sum16 = _mm512_add_ps(sum16, _mm512_mul_ps(a16_lo_f32, x16_lo));
        sum16 = _mm512_add_ps(sum16, _mm512_mul_ps(a16_hi_f32, x16_hi));
    }

    // 2. 剩余元素处理：标量循环兜底（处理最后rem个元素）
    float scalar_sum = 0.0f;
    for (size_t j = main_loop_end; j < k; ++j)
    {
        float a_fp32 = _bf16_to_fp32(a[j]); // bf16标量转fp32
        float x_fp32 = x[j];                // 直接取fp32值
        scalar_sum += a_fp32 * x_fp32;      // 标量乘积累加
    }

    // 3. 合并向量化累加结果与标量累加结果
    float vector_sum = _mm512_reduce_add_ps(sum16);
    return vector_sum + scalar_sum;
}

inline float dot_fp32_avx512(const float* a, const float* x, size_t k)
{
    assert(a != nullptr && x != nullptr && "Input pointers cannot be null");
    if (k == 0) return 0.0f; // 空向量直接返回0

    __m512 sum16 = _mm512_setzero_ps();
    const size_t simd_step = 16;
    const size_t rem = k % simd_step;
    const size_t main_loop_end = k - rem;

    for (size_t j = 0; j < main_loop_end; j += simd_step)
    {
        __m512 a16 = _mm512_loadu_ps(a + j);
        __m512 x16 = _mm512_loadu_ps(x + j);
        sum16 = _mm512_add_ps(sum16, _mm512_mul_ps(a16, x16));
    }

    float scalar_sum = 0.0f;
    for (size_t j = main_loop_end; j < k; ++j)
    {
        scalar_sum += a[j] * x[j]; // 标量乘积累加
    }

    // 3. 合并向量化累加结果与标量累加结果
    float vector_sum = _mm512_reduce_add_ps(sum16);
    return vector_sum + scalar_sum;
}

// input1维度 : [i , j],一般作为权重参数， input2维度:[j, 1], 一般作为输入向量
// 为了避免input2的列访问，所以input2最好是列维度为1，这样正好数据是连续的，方便simd
template <typename T>
void _gemv(float* out, const T* input1, const float* input2, std::size_t i, std::size_t j)
{
    for (std::size_t x = 0; x < i; ++x)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            out[x] = dot_fp32_avx512(input1 + x * j, input2, j);
        }
        else if constexpr (std::is_same_v<T, bf16_t>)
        {
            out[x] = dot_bf16_fp32_avx512(input1 + x * j, input2, j);
        }
    }
}

// [1, i] * [i, j] = [1 , j]
template <typename T>
void _gevm(float* out, const float* input1, const T* input2, std::size_t i, std::size_t j)
{
    // 按行广播相乘然后累加
    for (std::size_t x = 0; x < i; ++x)
    {
        std::size_t rem = j / 16;
        std::size_t loop_end = j % 16;
        if constexpr (std::is_same_v<T, float>)
        {
            __m512 vec_in_1 = _mm512_set1_ps(input1[x]);
            for (std::size_t y = 0; y < rem; y++)
            {
                __m512 vec_in_2 = _mm512_loadu_ps(input2 + x * j + y * 16);
                __m512 out_t = _mm512_mul_ps(vec_in_1, vec_in_2);
                __m512 vec_o = _mm512_loadu_ps(out + y * 16);
                vec_o = _mm512_add_ps(vec_o, out_t);
                _mm512_storeu_ps(out + y * 16, vec_o);
            }

            for (std::size_t y = 0; y < loop_end; y++)
            {
                out[rem * 16 + y] += input1[x] * input2[x * j + rem * 16 + y];
            }
        }
        else if constexpr (std::is_same_v<T, bf16_t>)
        {
        }
    }
}

#include <cblas.h>
template <typename T>
void _gemv_blas(float* out, const T* input1, const float* input2, std::size_t i, std::size_t j)
{
    const float* mat_float = nullptr;
    std::vector<float> mat_temp;

    if constexpr (std::is_same_v<T, float>)
    {
        mat_float = input1;
    }
    else
    {
        mat_temp.resize(i * j);
        for (std::size_t idx = 0; idx < i * j; ++idx)
        {
            mat_temp[idx] = _cvt_to_fp32(input1[idx]);
        }
        mat_float = mat_temp.data();
    }

    // 3. 调用 OpenBLAS 单精度 GEMV 接口 cblas_sgemv
    //    核心参数说明（行优先存储 CblasRowMajor）：
    //    - trans: 矩阵是否转置（CblasNoTrans = 不转置，CblasTrans = 转置）
    //    - m: 矩阵的行维度 (i)
    //    - n: 矩阵的列维度 (j)
    //    - alpha: 矩阵乘法的系数（此处为 1.0）
    //    - A: 转换后的 float 矩阵指针
    //    - lda: 矩阵的列数 (j)，行优先时为矩阵的列维度
    //    - x: 输入向量指针（input2）
    //    - incx: 输入向量的步长（1 = 连续存储）
    //    - beta: 输出向量的累加系数（0.0 = 直接覆盖，不累加）
    //    - y: 输出向量指针（out）
    //    - incy: 输出向量的步长（1 = 连续存储）
    cblas_sgemv(CblasRowMajor,       // 矩阵存储方式：C/C++ 行优先
                CblasNoTrans,        // 矩阵不转置
                static_cast<int>(i), // 矩阵行维度 m = i
                static_cast<int>(j), // 矩阵列维度 n = j
                1.0f,                // alpha 系数
                mat_float,           // 输入矩阵（float 类型）
                static_cast<int>(j), // lda = 矩阵列数 j
                input2,              // 输入向量 x
                1,                   // incx = 1（向量连续存储）
                0.0f,                // beta 系数（0 = 覆盖输出）
                out,                 // 输出向量 y
                1                    // incy = 1（向量连续存储）
    );
}

/*交错配对
    核心思想：将相邻的维度两两配对。
    对于我们的8维向量x，交错配对会形成以下4个二维向量进行旋转：
    * 第0对: (x₀, x₁)
    * 第1对: (x₂, x₃)
    * 第2对: (x₄, x₅)
    * 第3对: (x₆, x₇)
*/
void _rope_cross_inplace_fp32(float* x, std::size_t dim, std::size_t pos, float base)
{
    for (std::size_t j = 0; j < dim / 2; j++)
    {
        float rs = sin(pos * powf(base, -2.0f / dim * j));
        float rc = cos(pos * powf(base, -2.0f / dim * j));
        float even_temp = x[2 * j];
        float odd_temp = x[2 * j + 1];
        x[2 * j] = even_temp * rc - odd_temp * rs;
        x[2 * j + 1] = even_temp * rs + odd_temp * rc;
    }
}

/*拆分配对
将向量拆分两半，前一半和后一半匹配逐元素匹配
*/
void rope_inplace_fp32(float* x, std::size_t head_dim, std::size_t pos, float theta)
{
    const std::size_t half_dim = head_dim / 2;
    for (std::size_t i = 0; i < half_dim; ++i)
    {
        float freq = std::pow(theta, -float(i) / half_dim);
        float val = pos * freq;
        float vc = std::cos(val);
        float vs = std::sin(val);

        float v0 = x[i];
        float v1 = x[i + half_dim];
        x[i] = v0 * vc - v1 * vs;
        x[i + half_dim] = v0 * vs + v1 * vc;
    }
}

void _attention_softmax_fp32_native(float* xout, const float* qh, const float* kh_head_block, const float* vh_head_block, std::size_t kv_len,
                                    std::size_t head_dim)
{
    std::vector<float> attention_scores(kv_len);
    float max_score = -1e9;

    std::fill_n(xout, head_dim, 0.0f);

    // 1. 对于每一个历史token， 计算注意力分数(q_h * k_h^T)
    // q: [1, head_dim], k_h^T [kv_len, head_dim]
    for (std::size_t i = 0; i < kv_len; i++)
    {
        const float* kh_i = kh_head_block + i * head_dim;

        float score = 0.0f;
        for (std::size_t j = 0; j < head_dim; j++)
        {
            score += qh[j] * kh_i[j];
        }
        score /= std::sqrt(static_cast<float>(head_dim));
        attention_scores[i] = score;

        max_score = score > max_score ? score : max_score;
    }

    // 2. 对每一个历史token的分数 计算softmax
    float exp_sum = 0.f;
    for (auto& score : attention_scores)
    {
        score = std::exp(score - max_score);
        exp_sum += score;
    }
    for (auto& score : attention_scores)
    {
        score /= exp_sum;
    }

    // 3. 加权求和 (attention_scores * v_h)
    //  vh_head_block 维度 [kv_len , head_dim] attention_score 维度[1, kv_len]
    for (std::size_t i = 0; i < kv_len; i++)
    {
        const float* vh_i = vh_head_block + i * head_dim;
        for (std::size_t j = 0; j < head_dim; j++)
        {
            xout[j] += attention_scores[i] * vh_i[j];
        }
    }
}

void _attention_softmax_fp32(float* xout, const float* qh, const float* kh_head_block, const float* vh_head_block, std::size_t kv_len,
                             std::size_t head_dim)
{
    std::vector<float> attention_scores(kv_len);
    float max_score = -1e9;

    std::fill_n(xout, head_dim, 0.0f);

    // 1. 对于每一个历史token， 计算注意力分数(q_h * k_h^T)
    // q: [1, head_dim], k_h^T [kv_len, head_dim]

    //----------------------
    _gemv(attention_scores.data(), kh_head_block, qh, kv_len, head_dim);

    std::size_t rem = kv_len / 16;
    std::size_t loop_end = kv_len % 16;

    __m512 inv_const = _mm512_set1_ps(1 / std::sqrt(static_cast<float>(head_dim)));
    for (std::size_t i = 0; i < rem; i++)
    {
        __m512 vec_in = _mm512_loadu_ps(attention_scores.data() + i * 16);
        __m512 vec_out = _mm512_mul_ps(vec_in, inv_const);

        _mm512_storeu_ps(attention_scores.data() + i * 16, vec_out);
    }

    for (std::size_t i = 0; i < loop_end; i++)
    {
        attention_scores[i + rem * 16] /= std::sqrt(static_cast<float>(head_dim));
    }

    max_score = *std::max_element(attention_scores.begin(), attention_scores.end());

    //---------------------------
    //优化1，使用simd计算attention scores， 8.89 tokens/s --> 9.11 tokens/s

    // 2. 对每一个历史token的分数 计算softmax
    float exp_sum = 0.f;
    for (auto& score : attention_scores)
    {
        score = std::exp(score - max_score);
        exp_sum += score;
    }
    for (auto& score : attention_scores)
    {
        score /= exp_sum;
    }

    // 3. 加权求和 (attention_scores * v_h)
    //  vh_head_block 维度 [kv_len , head_dim] attention_score 维度[1, kv_len]

    //------------------
    _gevm(xout, attention_scores.data(), vh_head_block, kv_len, head_dim);
    //------------------
    //优化2，使用simd计算attention_score * v， 9.11 tokens/s --> 9.35 tokens/s
}

// void _attention_softmax_fp32(float* xout, const float* qh, const float* kh_head_block, const float* vh_head_block, std::size_t kv_len,
//                              std::size_t head_dim, std::vector<float>& attention_scores)
// {
//     float max_score = -1e9;

//     std::fill_n(xout, head_dim, 0.0f);

//     // 1. 对于每一个历史token， 计算注意力分数(q_h * k_h^T)
//     // q: [1, head_dim], k_h^T [kv_len, head_dim]

//     //----------------------
//     _gemv(attention_scores.data(), kh_head_block, qh, kv_len, head_dim);

//     std::size_t rem = kv_len / 16;
//     std::size_t loop_end = kv_len % 16;

//     __m512 inv_const = _mm512_set1_ps(1 / std::sqrt(static_cast<float>(head_dim)));
//     for (std::size_t i = 0; i < rem; i++)
//     {
//         __m512 vec_in = _mm512_loadu_ps(attention_scores.data() + i * 16);
//         __m512 vec_out = _mm512_mul_ps(vec_in, inv_const);

//         _mm512_storeu_ps(attention_scores.data() + i * 16, vec_out);
//     }

//     for (std::size_t i = 0; i < loop_end; i++)
//     {
//         attention_scores[i + rem * 16] /= std::sqrt(static_cast<float>(head_dim));
//     }

//     max_score = *std::max_element(attention_scores.begin(), attention_scores.end());

//     //---------------------------
//     //优化1，使用simd计算attention scores， 8.89 tokens/s --> 9.11 tokens/s

//     // 2. 对每一个历史token的分数 计算softmax
//     float exp_sum = 0.f;
//     for (auto& score : attention_scores)
//     {
//         score = std::exp(score - max_score);
//         exp_sum += score;
//     }
//     for (auto& score : attention_scores)
//     {
//         score /= exp_sum;
//     }

//     // 3. 加权求和 (attention_scores * v_h)
//     //  vh_head_block 维度 [kv_len , head_dim] attention_score 维度[1, kv_len]

//     //------------------
//     _gevm(xout, attention_scores.data(), vh_head_block, kv_len, head_dim);
//     //------------------
//     //优化2，使用simd计算attention_score * v， 9.11 tokens/s --> 9.35 tokens/s
// }

void _mh_attention_fp32(float* out, const float* q, const float* k, const float* v, std::size_t num_heads, std::size_t head_dim,
                        std::size_t n_kv_heads, std::size_t kv_size, std::size_t kv_len)
{
    const std::int32_t q_per_head = num_heads / n_kv_heads; //每个kv head 对应的q head数量
    const std::size_t kv_head_block_size = kv_size * head_dim;

    
    for (int h = 0; h < num_heads; h++)
    {
        const float* qh = q + h * head_dim;
        float* outh = out + h * head_dim;
        const float* kh_head_block = k + kv_head_block_size * (h / q_per_head);
        const float* vh_head_block = v + kv_head_block_size * (h / q_per_head);

        //计算单头注意力
        _attention_softmax_fp32(outh, qh, kh_head_block, vh_head_block, kv_len, head_dim);
        
    }

    // const unsigned int num_threads = std::thread::hardware_concurrency();

    // static std::vector<std::vector<float>> workspaces;
    // if (workspaces.size() != num_threads || (workspaces.size() > 0 && workspaces[0].size() != kv_len)) {
    //     workspaces.assign(num_threads, std::vector<float>(kv_len));
    // }


    // const std::int32_t q_per_head = num_heads / n_kv_heads;
    // const std::size_t kv_head_block_size = kv_size * head_dim;
    
    // std::vector<std::future<void>> futures;
    // futures.reserve(num_threads);

    // // --- 任务分块策略 ---
    // const std::size_t heads_per_thread = num_heads / num_threads;
    // const std::size_t remainder_heads = num_heads % num_threads;
    // std::size_t start_head = 0;

    // for (int i = 0; i < num_threads; ++i) {
    //     std::size_t heads_for_this_thread = heads_per_thread + (i < remainder_heads ? 1 : 0);
    //     if (heads_for_this_thread == 0) continue;

    //     std::size_t end_head = start_head + heads_for_this_thread;
        
    //     futures.emplace_back(
    //         pool.enqueue([=, &workspace = workspaces[i]]() mutable {
    //             // 每个线程在自己的 "块" 内循环
    //             for (std::size_t h = start_head; h < end_head; ++h) {
    //                 const float* qh = q + h * head_dim;
    //                 float* outh = out + h * head_dim;

    //                 int kv_h_index = h / q_per_head;
    //                 const float* kh_head_block = k + kv_head_block_size * kv_h_index;
    //                 const float* vh_head_block = v + kv_head_block_size * kv_h_index;
                    
    //                 _attention_softmax_fp32(outh, qh, kh_head_block, vh_head_block, kv_len, head_dim, workspace);
    //             }
    //         })
    //     );
    //     start_head = end_head;
    // }

    // for (auto& f : futures) {
    //     f.get();
    // }
}

} // namespace spyinfer