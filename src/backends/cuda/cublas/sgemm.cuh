#pragma once

#include <cuda_fp16.h>


void __cuda_linear_fp16_cublas(half *output, half *input,  half *weight, int m, int k, int n, half alpha);

void __cuda_linear_fp32_fp16_cublas(float *output, float *input,  half *weight,  half *bias, int m, int k, int n, float alpha, float beta);
