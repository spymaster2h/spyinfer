#pragma once

#include <cstdint>
#include <bit>

#include <immintrin.h>

namespace spyinfer {

#ifdef __AVX512F__

#define _PS512_CONST(name, val) static const __m512 _ps512_##name = _mm512_set1_ps(val)

// 基础常量
_PS512_CONST(1, 1.0f);
_PS512_CONST(0p5, 0.5f);
_PS512_CONST(cephes_LOG2EF, 1.4426950408889634f); // 1/ln2
_PS512_CONST(cephes_exp_C1, 0.693359375f);        // ln2 近似值（Cephes 校准）
_PS512_CONST(cephes_exp_C2, -2.12194440e-4f);     // (ln2)^2/2 校准值
_PS512_CONST(exp_hi, 88.3762626647949f);          // exp(x) 上界（避免溢出）
_PS512_CONST(exp_lo, -88.3762626647949f);         // exp(x) 下界

// Cephes 多项式拟合系数（局部优化，针对 |g| < ln2）
_PS512_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS512_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS512_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS512_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS512_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS512_CONST(cephes_exp_p5, 5.0000001201E-1f);

// 整数常量：float指数位偏移（127 << 23）
static const __m512i _pi512_0x7f = _mm512_set1_epi32(0x7F); // 127
static const __m512i _pi512_23shift = _mm512_set1_epi32(23);

__m512 exp512_ps(__m512 x);
#endif

uint16_t _fp32_to_bf16(float fp32);

void x64_sigmoid_fp32(float* out, float* in, int len);

void x64_float32_add(float* float32_out, float* float32_a, float* float32_b, int len);

float x64_dot_product_fp32(const float* a, const float* b, int len);

float x64_dot_product_fp32_bf16(const float* a, const uint16_t* b, int len);

void x64_rms_norm_fp32_bf16(float* out, const float* input, const uint16_t* weight, int len, float eps);

float x64_scalar_vector_mul_accumulate_fp32(float scalar, const float* vector, int length);

void x64_vectorized_accumulate_fp32(float* out, float scalar, const float* input, int len);

void x64_swiglu_fp32(float* out, const float* gate, const float* up, int len);

} // namespace spyinfer
