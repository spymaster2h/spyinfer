#include "sgemm.cuh"
#include "../native/common.cuh"


#include <map>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

static std::map<int, cublasHandle_t> cublasHandleMap;

cublasHandle_t getCublasHandle()
{
    int id = -1;
    cudaGetDevice(&id);
    auto it = cublasHandleMap.find(id);
    if (it != cublasHandleMap.end())
    {
        return it->second;
    }
    cublasHandle_t handler = nullptr;
    auto stat = cublasCreate(&handler);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error: CUBLAS initialization failed. state %d.\n", stat);
        exit(0);
    }
    else
    {
        cublasHandleMap[id] = handler;
    }

    return handler;
}

void __cuda_linear_fp16_cublas(half *output, half *input,  half *weight, int m, int k, int n, half alpha)
{
    auto cublasHandle = getCublasHandle();
    half o_beta = 0.f;
    auto status = cublasGemmEx(cublasHandle, 
                                     CUBLAS_OP_T, 
                                     CUBLAS_OP_N, 
                                     n, m, k, 
                                     &alpha, weight, CUDA_R_16F, k, 
                                     input, CUDA_R_16F, k, &o_beta,
                                     output, CUDA_R_16F, n, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);


    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error: CUBLAS gemmEx failed. state %d.\n", status);
        exit(0);
    }
}

void __cuda_linear_fp32_fp16_cublas(float *output, float *input, half *weight, half *bias, int m, int k, int n, float alpha, float beta)
{

    half *fp16_input = nullptr, *fp16_output = nullptr;
    int input_len = m * k;
    int output_len = m * n;
    cudaMalloc(&fp16_input, sizeof(half) * input_len);
    cudaMalloc(&fp16_output, sizeof(half) * output_len);
    int threadPerBlock = std::min(256, input_len);
    __cuda_fp32_to_fp16<<< (input_len - 1) / threadPerBlock + 1, threadPerBlock>>>(input, fp16_input, input_len);

    __cuda_linear_fp16_cublas(fp16_output, fp16_input, weight, m, k, n, alpha);

    threadPerBlock = std::min(256, output_len);
    __cuda_fp16_to_fp32<<< (output_len - 1) / threadPerBlock + 1, threadPerBlock>>>(fp16_output, output, output_len);

    if (bias != nullptr)
    {
        __cuda_add_inplace_fp32_fp16<<<m, 256>>>(output, bias, n);
    }

    cudaFree(fp16_input);
    cudaFree(fp16_output);
}

