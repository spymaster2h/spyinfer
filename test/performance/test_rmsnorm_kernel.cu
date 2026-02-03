#include "backends/cuda/kernels.cuh"
#include "backends/cpu/x64/kernels.hpp"
#include "utils/precision.hpp"

#include <random>
#include <iostream>

using namespace spyinfer;
// 用于比较浮点数的辅助函数
bool compare_floats(float a, float b, float tolerance = 1e-4f) { return std::abs(a - b) <= tolerance; }


int main()
{
    const int nums = 2;  // 批处理大小
    const int dims = 512; // 每个向量的维度
    float eps = 1e-6;

    // 创建host内存
    std::vector<float> h_input(nums * dims);
    std::vector<float> fp32_weight(dims);
    std::vector<uint16_t> h_weight(dims);  // CPU端权重为BF16格式
    std::vector<float> h_cpu_output(nums * dims);  // CPU输出

    std::vector<uint16_t> h_gpu_output(nums * dims);  // gPU输出


    std::vector<uint16_t> h_weight_fp16(dims);
    std::vector<uint16_t> h_input_fp16(nums * dims);



    // 使用随机数生成器填充数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_int_distribution<uint16_t> weight_dis(0, 65535);

    // 填充输入数据
    for (int i = 0; i < nums * dims; ++i)
    {
        h_input[i] = dis(gen);
        h_input_fp16[i] = float_to_half(h_input[i]);
    }

    // 填充权重数据
    for (int i = 0; i < dims; ++i)
    {
        fp32_weight[i] = weight_dis(gen);
        h_weight[i] = _fp32_to_bf16(fp32_weight[i]);
        h_weight_fp16[i] = float_to_half(fp32_weight[i]);
    }


    // 在CPU上计算真值
    for (int n = 0; n < nums; ++n)
    {
        x64_rms_norm_fp32_bf16(
            &h_cpu_output[n * dims],
            &h_input[n * dims],
            h_weight.data(),
            dims,
            eps
        );
    }

    // 分配GPU内存
    uint16_t *d_input, *d_output, *d_weight;
    cudaMalloc(&d_input, nums * dims * sizeof(uint16_t));
    cudaMalloc(&d_weight, dims * sizeof(uint16_t));
    cudaMalloc(&d_output, nums * dims * sizeof(uint16_t));

    cudaMemcpy(d_input, h_input_fp16.data(), nums * dims * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight_fp16.data(), dims * sizeof(uint16_t), cudaMemcpyHostToDevice);


    
    cuda_rms_norm_fp16(d_output, d_input, d_weight, nums, dims, eps);

    // 从GPU复制结果
    cudaMemcpy(h_gpu_output.data(), d_output, nums * dims * sizeof(half), cudaMemcpyDeviceToHost);

    // 验证结果
    bool passed = true;
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int diff_count = 0;

    for (int n = 0; n < nums; ++n)
    {
        for (int i = 0; i < dims; ++i)
        {
            float cpu_val = h_cpu_output[n * dims + i];
            float gpu_val = half_to_float(h_gpu_output[n * dims + i]);
            
            if (!compare_floats(gpu_val, cpu_val, 1e-5))

                    std::cout << "Mismatch at index " << i << ": expected " << cpu_val
                    << ", got " << gpu_val;
        }
    }

    // 清理内存
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    return 0;
}