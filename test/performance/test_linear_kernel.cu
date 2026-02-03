#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "backends/cuda/native/sgemm.cuh"

#include "utils/precision.hpp"

#include <vector>
#include <random>
#include <chrono>
#include <iostream>

int main(int argc, char **argv)
{
    const int M = 4; // 必须小于8，因为使用的是m8版本的kernel
    const int K = 1024;
    const int N = 1024;

    // 分配GPU内存
    float *d_input, *d_output;
    uint16_t *d_weight;

    cudaMalloc(&d_input, M * K * sizeof(float));
    cudaMalloc(&d_weight, N * K * sizeof(uint16_t));
    cudaMalloc(&d_output, M * N * sizeof(float));

    // 初始化输入数据
    std::vector<float> h_input(M * K);
    std::vector<float> h_weight(N * K);

    // 使用随机数生成器填充数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i)
    {
        h_input[i] = dis(gen);
    }

    for (int i = 0; i < N * K; ++i)
    {
        h_weight[i] = dis(gen);
    }

    // 将FP32权重转换为FP16
    std::vector<uint16_t> h_weight_fp16(N * K);
    for (int i = 0; i < N * K; ++i)
    {
        h_weight_fp16[i] = float_to_half(h_weight[i]);
    }

    // 复制数据到GPU
    cudaMemcpy(d_input, h_input.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight_fp16.data(), N * K * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // 预热GPU
    for (int i = 0; i < 5; ++i)
    {
        __cuda_linear_fp32_fp16_m8(d_output, // 输出指针
                                   d_input,  // 输入指针
                                   reinterpret_cast<half*>(d_weight), // 权重指针
                                   nullptr,   // 偏置指针（这里为nullptr）
                                   M,        // m维度
                                   K,        // k维度
                                   N,        // n维度
                                   1.0f,     // alpha
                                   0.0f      // beta
        );
    }
    cudaDeviceSynchronize();

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    const int num_iterations = 100;
    for (int i = 0; i < num_iterations; ++i)
    {
        __cuda_linear_fp32_fp16_m8(d_output, // 输出指针
                                   d_input,  // 输入指针
                                   reinterpret_cast<half*>(d_weight), // 权重指针
                                   nullptr,   // 偏置指针（这里为nullptr）
                                   M,        // m维度
                                   K,        // k维度
                                   N,        // n维度
                                   1.0f,     // alpha
                                   0.0f      // beta
        );
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    // 计算平均执行时间
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_us = static_cast<double>(duration.count()) / num_iterations;

    // 计算吞吐量
    size_t flops = static_cast<size_t>(2) * M * N * K; // GEMM操作的FLOPs
    double gflops = (static_cast<double>(flops) / 1e9) / (avg_time_us / 1e6);

    std::cout << "Performance Results:" << std::endl;
    std::cout << "Matrix dimensions: M=" << M << ", K=" << K << ", N=" << N << std::endl;
    std::cout << "Average execution time: " << avg_time_us << " microseconds" << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Total FLOPs: " << flops << std::endl;

    // 验证结果（可选）
    // std::vector<float> h_output(M * N);
    // cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // std::cout << "\nFirst 10 output values:" << std::endl;
    // for (int i = 0; i < std::min(10, static_cast<int>(h_output.size())); ++i)
    // {
    //     std::cout << "output[" << i << "] = " << h_output[i] << std::endl;
    // }

    // 释放GPU内存
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    std::cout << "\nKernel execution completed successfully!" << std::endl;

    return 0;
}
