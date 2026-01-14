#pragma once

#include <cblas.h>
#include <cstdint>
#include <vector>
#include <array>

#include "core/tensor.hpp"
#include "backends/cpu/x64/kernels.hpp"
#include "utils/constant_table.hpp"

namespace spyinfer {

/**
 * @brief 使用OpenBLAS执行矩阵乘法，支持fp32和bf16数据类型
 * @param input 输入张量，形状为[1, 1, m, k]
 * @param weight 权重张量，形状为[1, 1, n, k]
 * @param output 输出张量，形状为[1, 1, m, n]
 */
void blas_gemm(const Tensor& input, const Tensor& weight, Tensor& output) {
    // 获取矩阵维度
    int m = static_cast<int>(input.shape()[2]);
    int k = static_cast<int>(input.shape()[3]);
    int n = static_cast<int>(weight.shape()[2]);

    // 检查输入输出形状
    if (input.shape()[3] != weight.shape()[3]) {
        throw std::invalid_argument("Input and weight must have the same number of columns (k dimension)");
    }
    if (output.shape()[2] != input.shape()[2] || output.shape()[3] != weight.shape()[2]) {
        throw std::invalid_argument("Output shape must be [1, 1, m, n]");
    }

    // 处理fp32情况
    if (input.dtype() == DataType::fp32_t && weight.dtype() == DataType::fp32_t) {
        // 获取数据指针
        const float* input_data = input.data_ptr<float>();
        const float* weight_data = weight.data_ptr<float>();
        float* output_data = output.data_ptr<float>();

        // 使用OpenBLAS执行矩阵乘法: output = input * weight^T
        // 注意：这里假设输入是RowMajor，权重需要转置
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    m, n, k, 
                    1.0f, input_data, k, 
                    weight_data, k, 
                    0.0f, output_data, n);
    }
    // 处理bf16情况
    else if (input.dtype() == DataType::fp32_t && weight.dtype() == DataType::bf16_t) {
        // 获取输入数据指针
        const float* input_data = input.data_ptr<float>();
        const uint16_t* weight_bf16_data = weight.data_ptr<uint16_t>();
        float* output_data = output.data_ptr<float>();

        // 将bf16权重转换为fp32
        std::vector<float> weight_fp32_data(weight.numel());
        for (size_t i = 0; i < weight.numel(); ++i) {
            weight_fp32_data[i] = ConstantTable::GetInstance().bf16_to_fp32_table_[weight_bf16_data[i]];
        }

        // 使用OpenBLAS执行矩阵乘法: output = input * weight^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    m, n, k, 
                    1.0f, input_data, k, 
                    weight_fp32_data.data(), k, 
                    0.0f, output_data, n);
    }
    else {
        throw std::invalid_argument("Unsupported data type combination");
    }
}

/**
 * @brief 使用OpenBLAS执行矩阵乘法并添加偏置，支持fp32和bf16数据类型
 * @param input 输入张量，形状为[1, 1, m, k]
 * @param weight 权重张量，形状为[1, 1, n, k]
 * @param bias 偏置张量，形状为[1, 1, m, n]
 * @param output 输出张量，形状为[1, 1, m, n]
 */
void blas_gemm_with_bias(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    // 先执行矩阵乘法
    blas_gemm(input, weight, output);

    // 添加偏置
    if (output.dtype() == DataType::fp32_t && bias.dtype() == DataType::fp32_t) {
        const float* bias_data = bias.data_ptr<float>();
        float* output_data = output.data_ptr<float>();
        
        // 将偏置添加到输出
        for (size_t i = 0; i < output.numel(); ++i) {
            output_data[i] += bias_data[i];
        }
    }
    else {
        throw std::invalid_argument("Unsupported data type combination for bias");
    }
}

} // namespace spyinfer