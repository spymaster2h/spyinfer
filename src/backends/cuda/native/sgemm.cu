#include "sgemm.cuh"
#include "common.cuh"


#include <cuda_fp16.h>
#include <iostream>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <int THREAD_PER_BLOCK, int PART>
__global__ void __cuda_linear_fp32_fp16_kernel(float *A, half *B, float *C, half *bias, int m, int k)
{
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    const half zero = __float2half_rn(0.0);
    float4 regA;
    union_half4 regB;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const half *baseB = B + p * m;
    if (m % 4 == 0)
    {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4)
        {
#pragma unroll
            for (int x = 0; x < PART; x++)
            {
                regA = FETCH_FLOAT4(A[i + x * m]);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m) sum += regA.x * __low2float(regB.out2[0]);
                if (i + 1 < m) sum += regA.y * __high2float(regB.out2[0]);
                if (i + 2 < m) sum += regA.z * __low2float(regB.out2[1]);
                if (i + 3 < m) sum += regA.w * __high2float(regB.out2[1]);
                sdata[x][tid] += sum;
            }
        }
    }
    else
    {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK)
        {
#pragma unroll
            for (int x = 0; x < PART; x++)
            {
                sdata[x][tid] += A[i + x * m] * (float)baseB[i];
            }
        }
    }
    __syncthreads();
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
#pragma unroll
            for (int x = 0; x < PART; x++)
            {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        if (bias == nullptr)
        {
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0];
        }
        else
        {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] + __half2float(__ldg(bias + p));
        }
    }
    __syncthreads();
}

void __cuda_linear_fp32_fp16_m8(float *output, float *input,  half *weight,  half *bias, int m, int k, int n, float alpha, float beta)
{

    if (m == 1)
    {
        __cuda_linear_fp32_fp16_kernel<256, 1><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 2)
    {
        __cuda_linear_fp32_fp16_kernel<256, 2><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 3)
    {
        __cuda_linear_fp32_fp16_kernel<256, 3><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 4)
    {
        __cuda_linear_fp32_fp16_kernel<256, 4><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 5)
    {
        __cuda_linear_fp32_fp16_kernel<256, 5><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 6)
    {
        __cuda_linear_fp32_fp16_kernel<256, 6><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 7)
    {
        __cuda_linear_fp32_fp16_kernel<256, 7><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else
    {
        printf("Error: __cuda_linear_fp32_fp16_m8 must m < 8.\n");
        exit(0);
    }
}




template <int THREAD_PER_BLOCK, int PART>
__global__ void __cuda_linear_fp16_fp16_kernel(half *A, half *B, half *C, half *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    const half zero = __float2half_rn(0.0);
    union_half8 regA;
    union_half8 regB;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
        
    const half *baseB = B + p * m;

    if (m % 8 == 0) {
#pragma unroll
        for (int i = tid * 8; i < m; i += THREAD_PER_BLOCK * 8) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint4 *>(A + x * m + i);
                regB.in = *reinterpret_cast<const uint4 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __low2float(regA.out2[0]) * __low2float(regB.out2[0]);
                if (i + 1 < m)
                    sum += __high2float(regA.out2[0]) * __high2float(regB.out2[0]);
                if (i + 2 < m)
                    sum += __low2float(regA.out2[1]) * __low2float(regB.out2[1]);
                if (i + 3 < m)
                    sum += __high2float(regA.out2[1]) * __high2float(regB.out2[1]);
                if (i + 4 < m)
                    sum += __low2float(regA.out2[2]) * __low2float(regB.out2[2]);
                if (i + 5 < m)
                    sum += __high2float(regA.out2[2]) * __high2float(regB.out2[2]);
                if (i + 6 < m)
                    sum += __low2float(regA.out2[3]) * __low2float(regB.out2[3]);
                if (i + 7 < m)
                    sum += __high2float(regA.out2[3]) * __high2float(regB.out2[3]);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += (float)A[i + x * m] * (float)baseB[i];
            }
        }
    }
    __syncthreads();
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = (half)(sdata[x][0] + (float)(__ldg(bias + p)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = (half)(sdata[x][0]);
        }
    }
    __syncthreads();
}



void __cuda_linear_fp16_fp16_m8(half *output, half *input,  half *weight,  half *bias, int m, int k, int n, float alpha, float beta)
{

    if (m == 1)
    {
        __cuda_linear_fp16_fp16_kernel<256, 1><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 2)
    {
        __cuda_linear_fp16_fp16_kernel<256, 2><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 3)
    {
        __cuda_linear_fp16_fp16_kernel<256, 3><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 4)
    {
        __cuda_linear_fp16_fp16_kernel<256, 4><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 5)
    {
        __cuda_linear_fp16_fp16_kernel<256, 5><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 6)
    {
        __cuda_linear_fp16_fp16_kernel<256, 6><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else if (m == 7)
    {
        __cuda_linear_fp16_fp16_kernel<256, 7><<<n, 256>>>(input, weight, output, bias, k, n);
    }
    else
    {
        printf("Error: __cuda_linear_fp32_fp16_m8 must m < 8.\n");
        exit(0);
    }
}