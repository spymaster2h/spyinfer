#include "kernels.hpp"
#include "utils/constant_table.hpp"

#include <cstring>
#include <cmath>
namespace spyinfer {

uint16_t _fp32_to_bf16(float fp32) { return (uint16_t)(std::bit_cast<uint32_t>(fp32) >> 16); }

#ifdef __AVX512F__
__m512 exp512_ps(__m512 x)
{
    __m512 tmp = _mm512_setzero_ps();
    __m512 fx;
    __m512i imm0;
    const __m512 one = _ps512_1;

    // 步骤1：限制x范围，避免exp溢出（和AVX2版本逻辑一致）
    x = _mm512_min_ps(x, _ps512_exp_hi);
    x = _mm512_max_ps(x, _ps512_exp_lo);

    // 步骤2：指数分解：exp(x) = exp(g + n*ln2) = exp(g) * 2^n
    // 计算 n = round(x / ln2)（整数部分）
    fx = _mm512_mul_ps(x, _ps512_cephes_LOG2EF); // x / ln2
    fx = _mm512_add_ps(fx, _ps512_0p5);          // 加0.5，用于floor取整

    // 步骤3：高精度floor取整（校准误差）
    tmp = _mm512_floor_ps(fx);
    // 检测tmp > fx的情况，生成掩码校准

    __mmask16 gt_mask = _mm512_cmp_ps_mask(tmp, fx, _CMP_GT_OQ); // 返回16位掩码
    __m512 mask = _mm512_mask_blend_ps(gt_mask, _mm512_setzero_ps(), one);
    fx = _mm512_sub_ps(tmp, mask);

    // 步骤4：还原小数部分g = x - n*ln2（校准后）
    tmp = _mm512_mul_ps(fx, _ps512_cephes_exp_C1);      // n * ln2
    __m512 z = _mm512_mul_ps(fx, _ps512_cephes_exp_C2); // n * (ln2)^2/2
    x = _mm512_sub_ps(x, tmp);
    x = _mm512_sub_ps(x, z); // 最终g = x - n*ln2（|g| < ln2）

    // 步骤5：局部多项式拟合exp(g)（Cephes优化系数，非泰勒）
    __m512 y = _ps512_cephes_exp_p0;
    y = _mm512_mul_ps(y, x);                    // p0 * g
    y = _mm512_add_ps(y, _ps512_cephes_exp_p1); // + p1
    y = _mm512_mul_ps(y, x);                    // (p0*g + p1) * g
    y = _mm512_add_ps(y, _ps512_cephes_exp_p2); // + p2
    y = _mm512_mul_ps(y, x);                    // (...)*g
    y = _mm512_add_ps(y, _ps512_cephes_exp_p3); // + p3
    y = _mm512_mul_ps(y, x);                    // (...)*g
    y = _mm512_add_ps(y, _ps512_cephes_exp_p4); // + p4
    y = _mm512_mul_ps(y, x);                    // (...)*g
    y = _mm512_add_ps(y, _ps512_cephes_exp_p5); // + p5

    // 乘g²，加g和1，完成多项式拟合
    __m512 g_sq = _mm512_mul_ps(x, x); // g²
    y = _mm512_mul_ps(y, g_sq);
    y = _mm512_add_ps(y, x);
    y = _mm512_add_ps(y, one);

    // 步骤6：快速计算2^n（利用IEEE754位操作，和AVX2版本逻辑一致）
    imm0 = _mm512_cvttps_epi32(fx);             // 浮点n转整数
    imm0 = _mm512_add_epi32(imm0, _pi512_0x7f); // 加127（float指数偏移）
    imm0 = _mm512_slli_epi32(imm0, 23);         // 左移23位到指数位
    __m512 pow2n = _mm512_castsi512_ps(imm0);   // 整数转浮点，得到2^n

    // 步骤7：合并结果：exp(g) * 2^n = exp(x)
    y = _mm512_mul_ps(y, pow2n);

    return y;
}

#endif

void x64_float32_add(float *float32_out, float *float32_a, float *float32_b, int len)
{
    int i = 0;
    const int batch_size = 16;

#ifdef __AVX512F__
    for (; i + batch_size <= len; i += batch_size)
    {
        __m512 vec_f32_a = _mm512_loadu_ps(&float32_a[i]);
        __m512 vec_f32_b = _mm512_loadu_ps(&float32_b[i]);
        __m512 vec_f32_out = _mm512_add_ps(vec_f32_a, vec_f32_b);
        _mm512_storeu_ps(&float32_out[i], vec_f32_out);
    }
#endif
    // 处理剩余元素
    for (; i < len; i++)
    {
        float32_out[i] = float32_a[i] + float32_b[i];
    }
}

float x64_dot_product_fp32(const float *a, const float *b, int len)
{
    int i = 0;
    const int batch_size = 16;

    float result = 0.f;

#ifdef __AVX512F__
    __m512 sum = _mm512_setzero_ps();
    for (; i + batch_size <= len; i += batch_size)
    {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    result += _mm512_reduce_add_ps(sum);
#endif
    // 处理剩余元素
    for (; i < len; i++)
    {
        result += a[i] * b[i];
    }

    return result;
}

float x64_dot_product_fp32_bf16(const float *a, const uint16_t *b, int len)
{
    int i = 0;
    const int batch_size = 32;

    float result = 0.f;
#ifdef __AVX512F__
    __m512 sum16 = _mm512_setzero_ps();
    __m512i zero = _mm512_setzero_si512();

    for (; i + batch_size <= len; i += batch_size)
    {
        __m512i a32 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + i));
        __m256i a16_lo = _mm512_castsi512_si256(a32);
        __m256i a16_hi = _mm512_extracti64x4_epi64(a32, 1);

        // 将16位无符号整数零扩展（zero-extend）到32位整数。
        // 这一步后，每个bf16数值占据了一个32位整数的低16位，高16位为0。
        __m512i b32_lo = _mm512_cvtepu16_epi32(a16_lo);
        __m512i b32_hi = _mm512_cvtepu16_epi32(a16_hi);

        // 核心转换步骤：通过对每个32位整数逻辑左移16位，将bf16的位模式转换为fp32的位模式。
        __m512i c32_lo = _mm512_slli_epi32(b32_lo, 16);
        __m512i c32_hi = _mm512_slli_epi32(b32_hi, 16);

        // 整数寄存器→浮点寄存器转换（无开销）
        __m512 a16_lo_f32 = _mm512_castsi512_ps(c32_lo);
        __m512 a16_hi_f32 = _mm512_castsi512_ps(c32_hi);

        // 加载32个fp32元素到512位浮点寄存器
        __m512 x16_lo = _mm512_loadu_ps(a + i);
        __m512 x16_hi = _mm512_loadu_ps(a + i + 16);

        // 512位向量化乘积累加
        sum16 = _mm512_fmadd_ps(a16_lo_f32, x16_lo, sum16);
        sum16 = _mm512_fmadd_ps(a16_hi_f32, x16_hi, sum16);
    }
    result += _mm512_reduce_add_ps(sum16);
#endif

    for (; i < len; ++i)
    {
        result += a[i] * ConstantTable::GetInstance().bf16_to_fp32_table_[b[i]];
    }
    return result;
}

void x64_rms_norm_fp32_bf16(float *out, const float *input, const uint16_t *weight, int len, float eps)
{
    float norm = 0.f;
    for (int32_t i = 0; i < len; i++)
    {
        norm += input[i] * input[i];
    }
    norm = 1.f / std::sqrt(norm / len + eps);
    for (int32_t i = 0; i < len; i++)
    {
        out[i] = norm * input[i] * ConstantTable::GetInstance().bf16_to_fp32_table_[weight[i]];
    }
}

float x64_scalar_vector_mul_accumulate_fp32(float scalar, const float *vector, int length)
{
    float result = 0.0f;
    const int batch_size = 16;
    __m512 scalar_vec = _mm512_set1_ps(scalar);

    __m512 sum = _mm512_setzero_ps();

    int i = 0;
    for (; i + batch_size <= length; i += batch_size)
    {
        __m512 vec_batch = _mm512_loadu_ps(vector + i);
        sum = _mm512_fmadd_ps(scalar_vec, vec_batch, sum);
    }
    for (; i < length; ++i)
    {
        result += scalar * vector[i];
    }
    return result + _mm512_reduce_add_ps(sum);
}

void x64_vectorized_accumulate_fp32(float *out, float scalar, const float *input, int len)
{
    const size_t batch_size = 16;
    __m512 scalar_vec = _mm512_set1_ps(scalar);

    int i = 0;
    for (; i + batch_size <= len; i += batch_size)
    {
        __m512 o_batch = _mm512_loadu_ps(out + i);
        __m512 i_batch = _mm512_loadu_ps(input + i);
        __m512 result_batch = _mm512_fmadd_ps(scalar_vec, i_batch, o_batch);

        _mm512_storeu_ps(out + i, result_batch);
    }
    for (; i < len; ++i)
    {
        out[i] += scalar * input[i];
    }
}

// gate * sigmoid(gate) * up
void x64_swiglu_fp32(float *out, const float *gate, const float *up, int len)
{
    int i = 0;
#ifdef __AVX512F__
    const __m512 one = _mm512_set1_ps(1.0f);
    for (; i + 15 < len; i += 16)
    {
        __m512 g_vec = _mm512_loadu_ps(&gate[i]);
        __m512 u_vec = _mm512_loadu_ps(&up[i]);

        __m512 neg_g = _mm512_sub_ps(_mm512_setzero_ps(), g_vec); // -gate
        __m512 exp_neg_g = exp512_ps(neg_g);                      // exp(-gate)
        __m512 denom = _mm512_add_ps(one, exp_neg_g);             // 1 + exp(-gate)
        __m512 silu_g = _mm512_div_ps(g_vec, denom);              // SiLU(gate)
        __m512 res_vec = _mm512_mul_ps(silu_g, u_vec);
        _mm512_storeu_ps(&out[i], res_vec);
    }
#endif
    for (; i < len; i++)
    {
        float g = gate[i];
        float u = up[i];
        out[i] = (g / (1.0f + expf(-g))) * u;
    }
}

void x64_sigmoid_fp32(float *out, float *in, int len)
{
    int i = 0;
#ifdef __AVX512F__
    const __m512 one = _mm512_set1_ps(1.0f);
    for (; i + 15 < len; i += 16)
    {
        __m512 x_vec = _mm512_loadu_ps(&in[i]);
        __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x_vec); // -x
        __m512 exp_neg_x = exp512_ps(neg_x);                      // exp(-x)
        __m512 sigmoid_x = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg_x));
        _mm512_storeu_ps(&out[i], sigmoid_x);
    }
#endif
    for (; i < len; i++)
    {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

} // namespace spyinfer
