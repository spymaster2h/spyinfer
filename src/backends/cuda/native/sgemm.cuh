#pragma once

#include <cuda_fp16.h>
#include <cstdint>



void __cuda_linear_fp32_fp16_m8(float* output, float* d_input, half* d_weight, half* h_bias, int m, int k, int n,
                                float alpha = 1.0f, float beta = 1.0f);

void __cuda_linear_fp16_fp16_m8(half* output, half* d_input, half* d_weight, half* h_bias, int m, int k, int n,
                                float alpha = 1.0f, float beta = 1.0f);