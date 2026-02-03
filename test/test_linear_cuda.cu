#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "backends/cuda/cuda_backend.hpp"
#include "backends/base_operator.hpp"
#include "core/tensor.hpp"
#include "core/base_device.hpp"

#include "utils/precision.hpp"




using namespace spyinfer;
// 用于比较浮点数的辅助函数
bool compare_floats(float a, float b, float tolerance = 1e-4f) {
    return std::abs(a - b) <= tolerance;
}

class TestCudaLinear : public ::testing::Test {
protected:
    std::shared_ptr<CUDABackend> backend_;

    void SetUp() override {
        backend_ = std::make_shared<CUDABackend>();
    }

    void TearDown() override {
        backend_.reset();
    }
};

// 测试不同形状的CUDA Linear运算
TEST_F(TestCudaLinear, DifferentShapes) {
    std::vector<std::tuple<std::array<int64_t, 4>, std::array<int64_t, 4>>> test_cases = {
        {{1, 1, 1, 8}, {1, 1, 10, 8}},  // m=1, k=8, n=10
        {{1, 2, 2, 1024}, {1, 1, 1024, 1024}}, 
        {{1, 3, 3, 1024}, {1, 1, 1024, 1024}}, 
    };

    for (const auto& [input_shape, weight_shape] : test_cases) {
        std::array<int64_t, 4> output_shape = {input_shape[0], input_shape[1], input_shape[2], weight_shape[2]};
        
        // 创建输入张量
        auto device = backend_->get_devices()[0];
        auto input = Tensor::create(device, input_shape, DataType::fp32_t);
        auto weight = Tensor::create(device, weight_shape, DataType::fp16_t);
        auto output = Tensor::create(device, output_shape, DataType::fp32_t);

        // 生成随机数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

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

        // 将FP32数据转换为FP16
        std::vector<std::uint16_t> weight_data_fp16(weight_numel);
        for (size_t i = 0; i < weight_numel; ++i) {
            weight_data_fp16[i] = float_to_half(weight_data[i]);
        }

        // 将数据复制到张量
        backend_->copy_data_from_cpu(input->data_ptr<float>(), input_data.data(), input_data.size() * sizeof(float));
        backend_->copy_data_from_cpu(weight->data_ptr<uint16_t>(), weight_data_fp16.data(), weight_data_fp16.size() * sizeof(uint16_t));

        // 准备参数
        std::unordered_map<std::string, std::any> params;
        params["op_type"] = OperatorType::Linear;
        params["input"] = input;
        params["weight"] = weight;
        params["bias"] = std::shared_ptr<Tensor>(); // 空的bias
        params["output"] = output;

        // 执行前向扩展和计算
        backend_->run(params);

        // 验证结果：手动计算期望输出
        std::vector<float> expected_output(output->numel(), 0.0f);
        int M = static_cast<int>(input_shape[2] * input_shape[1] * input_shape[0]);
        int N = static_cast<int>(weight_shape[2]);
        int K = static_cast<int>(input_shape[3]);

        // 执行矩阵乘法: output = input * weight^T
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < K; ++k_idx) {
                    float input_val = input_data[i * K + k_idx];
                    float weight_val = half_to_float(weight_data_fp16[j * K + k_idx]);
                    sum += input_val * weight_val;
                }
                expected_output[i * N + j] = sum;
            }
        }

        // 从CUDA设备复制结果到主机
        std::vector<float> actual_output(output->numel());
        backend_->copy_data_to_cpu(actual_output.data(), output->data_ptr<float>(), actual_output.size() * sizeof(float));

        // 验证结果
        for (size_t i = 0; i < expected_output.size(); ++i) {
            EXPECT_NEAR(actual_output[i], expected_output[i], 1e-1f)
                << "Mismatch at index " << i << " for shape (" << M << "," << K << "," << N << ")";
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}