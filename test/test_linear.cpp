#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <thread>

// 添加OpenBLAS头文件
#include "gt/blas_gemm.hpp"

#include "backends/cpu/cpu_backend.hpp"
#include "backends/cpu/x64/kernels.hpp"
#include "backends/base_operator.hpp"
#include "core/tensor.hpp"
#include "core/base_device.hpp"
#include "backends/cpu/x64/kernels.hpp"
#include "utils/constant_table.hpp"



namespace spyinfer {

class TestLinear : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建默认CPU后端
        backend_ = std::make_shared<CPUBackend>();
    }

    void TearDown() override {
        // 清理资源
        backend_.reset();
    }

    std::shared_ptr<CPUBackend> backend_;
};

// 测试基本的Linear运算
TEST_F(TestLinear, BasicLinear) {
    // 定义张量形状
    std::array<int64_t, 4> input_shape = {1, 1, 2, 3};  // [1, 1, m=2, k=3]
    std::array<int64_t, 4> weight_shape = {1, 1, 4, 3}; // [1, 1, n=4, k=3]
    std::array<int64_t, 4> output_shape = {1, 1, 2, 4}; // [1, 1, m=2, n=4]

    // 创建输入张量input、weight和bias
    auto device = backend_->get_devices()[0];
    auto input = Tensor::create(device, input_shape, DataType::fp32_t);
    auto weight = Tensor::create(device, weight_shape, DataType::fp32_t);
    auto bias = Tensor::create(device, output_shape, DataType::fp32_t);
    auto output = Tensor::create(device, output_shape, DataType::fp32_t);

    // 设置测试数据
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> weight_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    std::vector<float> bias_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // 将数据复制到张量
    memcpy(input->data_ptr<float>(), input_data.data(), input_data.size() * sizeof(float));
    memcpy(weight->data_ptr<float>(), weight_data.data(), weight_data.size() * sizeof(float));
    memcpy(bias->data_ptr<float>(), bias_data.data(), bias_data.size() * sizeof(float));

    // 准备参数
    std::unordered_map<std::string, std::any> params;
    
    params["op_type"] = OperatorType::Linear;
    params["input"] = input;
    params["weight"] = weight;
    params["bias"] = bias;
    params["output"] = output;

    // 执行前向扩展和计算
    backend_->run(params);

    auto output_blas = Tensor::create(device, output_shape, DataType::fp32_t);

    blas_gemm_with_bias(*input.get(), *weight.get(), *bias.get(), *output_blas.get());

    // 验证结果
    const float* output_data = output->data_ptr<float>();
    const float* expected_output = output_blas->data_ptr<float>();
    for (size_t i = 0; i < output_blas->numel(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-6f);
    }
}

// 测试FP32权重的Linear运算
TEST_F(TestLinear, LinearFP32Weight) {
    // 定义张量形状
    std::array<int64_t, 4> input_shape = {1, 1, 10, 20};
    std::array<int64_t, 4> weight_shape = {1, 1, 30, 20};
    std::array<int64_t, 4> output_shape = {1, 1, 10, 30};

    // 创建输入张量
    auto device = backend_->get_devices()[0];
    auto input = Tensor::create(device, input_shape, DataType::fp32_t);
    auto weight = Tensor::create(device, weight_shape, DataType::fp32_t);
    auto output = Tensor::create(device, output_shape, DataType::fp32_t);

    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    size_t input_numel = input->numel();
    size_t weight_numel = weight->numel();

    std::vector<float> input_data(input_numel);
    std::vector<float> weight_data(weight_numel);
    
    for (size_t i = 0; i < input_numel; ++i) {
        input_data[i] = dist(gen);
    }
    for (size_t i = 0; i < weight_numel; ++i) {
        weight_data[i] = dist(gen);
    }

    // 将数据复制到张量
    memcpy(input->data_ptr<float>(), input_data.data(), input_data.size() * sizeof(float));
    memcpy(weight->data_ptr<float>(), weight_data.data(), weight_data.size() * sizeof(float));

    // 准备参数
    std::unordered_map<std::string, std::any> params;
    
    params["op_type"] = OperatorType::Linear;
    params["input"] = input;
    params["weight"] = weight;
    params["bias"] = std::shared_ptr<Tensor>();
    params["output"] = output;

    // 执行前向扩展和计算
    backend_->run(params);

    // 使用OpenBLAS计算预期结果
    std::vector<float> expected_output(output->numel(), 0.0f);
    int M = input_shape[2];
    int N = weight_shape[2];
    int K = input_shape[3];

    // 执行矩阵乘法
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                M, N, K, 
                1.0f, input_data.data(), K, 
                weight_data.data(), K, 
                0.0f, expected_output.data(), N);

    // 验证结果
    const float* output_data = output->data_ptr<float>();
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-3f);
    }
}

// 测试BF16权重的Linear运算
TEST_F(TestLinear, LinearBF16Weight) {
    // 定义张量形状
    std::array<int64_t, 4> input_shape = {1, 1, 2, 1030};
    std::array<int64_t, 4> weight_shape = {1, 1, 1024, 1030};
    std::array<int64_t, 4> output_shape = {1, 1, 2, 1024};

    // 创建输入张量
    auto device = backend_->get_devices()[0];
    auto input = Tensor::create(device, input_shape, DataType::fp32_t);
    auto weight = Tensor::create(device, weight_shape, DataType::bf16_t);
    auto output = Tensor::create(device, output_shape, DataType::fp32_t);

    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t input_numel = input->numel();
    size_t weight_numel = weight->numel();

    std::vector<float> input_data(input_numel);
    std::vector<float> weight_data_fp32(weight_numel);
    std::vector<uint16_t> weight_data_bf16(weight_numel);
    
    for (size_t i = 0; i < input_numel; ++i) {
        input_data[i] = dist(gen);
    }
    for (size_t i = 0; i < weight_numel; ++i) {
        weight_data_fp32[i] = dist(gen);
    }

    for (size_t i = 0; i < weight_numel; ++i) {
        weight_data_bf16[i] = _fp32_to_bf16(weight_data_fp32[i]);
    }

    for (size_t i = 0; i < weight_numel; ++i) {
        EXPECT_NEAR(weight_data_fp32[i], ConstantTable::GetInstance().bf16_to_fp32_table_[weight_data_bf16[i]], 1e-2f);
    }

    // 将数据复制到张量
    memcpy(input->data_ptr<float>(), input_data.data(), input_data.size() * sizeof(float));
    memcpy(weight->data_ptr<uint16_t>(), weight_data_bf16.data(), weight_data_bf16.size() * sizeof(uint16_t));

    // 准备参数
    std::unordered_map<std::string, std::any> params;
    
    params["op_type"] = OperatorType::Linear;
    params["input"] = input;
    params["weight"] = weight;
    params["bias"] = std::shared_ptr<Tensor>();
    params["output"] = output;

    // 执行前向扩展和计算
    backend_->run(params);

    // 使用OpenBLAS计算预期结果
    std::vector<float> expected_output(output->numel(), 0.0f);
    int M = input_shape[2];
    int N = weight_shape[2];
    int K = input_shape[3];

    // 执行矩阵乘法
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                M, N, K, 
                1.0f, input_data.data(), K, 
                weight_data_fp32.data(), K, 
                0.0f, expected_output.data(), N);

    // 验证结果（BF16会有精度损失，所以容忍度更高）
    const float* output_data = output->data_ptr<float>();
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-1f);
    }
}

TEST_F(TestLinear, LinearRowParallel) {
    std::array<int64_t, 4> input_shape = {2, 3, 20, 1000};
    std::array<int64_t, 4> weight_shape = {1, 1, 100, 1000};
    std::array<int64_t, 4> output_shape = {2, 3, 20, 100};

    // 创建输入张量
    auto device = backend_->get_devices()[0];
    auto input = Tensor::create(device, input_shape, DataType::fp32_t);
    auto weight = Tensor::create(device, weight_shape, DataType::fp32_t);
    auto output = Tensor::create(device, output_shape, DataType::fp32_t);

    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 0.1f);

    size_t input_numel = input->numel();
    size_t weight_numel = weight->numel();

    std::vector<float> input_data(input_numel);
    std::vector<float> weight_data(weight_numel);
    
    for (size_t i = 0; i < input_numel; ++i) {
        input_data[i] = dist(gen);
    }
    for (size_t i = 0; i < weight_numel; ++i) {
        weight_data[i] = dist(gen);
    }

    // 将数据复制到张量
    memcpy(input->data_ptr<float>(), input_data.data(), input_data.size() * sizeof(float));
    memcpy(weight->data_ptr<float>(), weight_data.data(), weight_data.size() * sizeof(float));

    // 准备参数
    std::unordered_map<std::string, std::any> params;
    
    params["op_type"] = OperatorType::Linear;
    params["input"] = input;
    params["weight"] = weight;
    params["bias"] = std::shared_ptr<Tensor>();
    params["output"] = output;

    // 执行前向扩展和计算
    backend_->run(params);

    // 使用OpenBLAS计算预期结果
    std::vector<float> expected_output(output->numel(), 0.0f);
    int M = input_shape[2] * input_shape[1] * input_shape[0];
    int N = weight_shape[2];
    int K = input_shape[3];

    // 执行矩阵乘法
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                M, N, K, 
                1.0f, input_data.data(), K, 
                weight_data.data(), K, 
                0.0f, expected_output.data(), N);

    // 验证结果
    const float* output_data = output->data_ptr<float>();
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-3f);
    }

}

TEST_F(TestLinear, LinearcolParallel) {
    std::array<int64_t, 4> input_shape = {1, 1, 1, 1000};
    std::array<int64_t, 4> weight_shape = {1, 1, 100, 1000};
    std::array<int64_t, 4> output_shape = {1, 1, 1, 100};

    // 创建输入张量
    auto device = backend_->get_devices()[0];
    auto input = Tensor::create(device, input_shape, DataType::fp32_t);
    auto weight = Tensor::create(device, weight_shape, DataType::fp32_t);
    auto output = Tensor::create(device, output_shape, DataType::fp32_t);

    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 0.1f);

    size_t input_numel = input->numel();
    size_t weight_numel = weight->numel();

    std::vector<float> input_data(input_numel);
    std::vector<float> weight_data(weight_numel);
    
    for (size_t i = 0; i < input_numel; ++i) {
        input_data[i] = dist(gen);
    }
    for (size_t i = 0; i < weight_numel; ++i) {
        weight_data[i] = dist(gen);
    }

    // 将数据复制到张量
    memcpy(input->data_ptr<float>(), input_data.data(), input_data.size() * sizeof(float));
    memcpy(weight->data_ptr<float>(), weight_data.data(), weight_data.size() * sizeof(float));

    // 准备参数
    std::unordered_map<std::string, std::any> params;
    
    params["op_type"] = OperatorType::Linear;
    params["input"] = input;
    params["weight"] = weight;
    params["bias"] = std::shared_ptr<Tensor>();
    params["output"] = output;

    // 执行前向扩展和计算
    backend_->run(params);

    // 使用OpenBLAS计算预期结果
    std::vector<float> expected_output(output->numel(), 0.0f);
    int M = input_shape[2];
    int N = weight_shape[2];
    int K = input_shape[3];

    // 执行矩阵乘法
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                M, N, K, 
                1.0f, input_data.data(), K, 
                weight_data.data(), K, 
                0.0f, expected_output.data(), N);

    // 验证结果
    const float* output_data = output->data_ptr<float>();
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-3f);
    }
}

} // namespace spyinfer

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}