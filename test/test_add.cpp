#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>

#include <cblas.h>

#include "backends/cpu/cpu_backend.hpp"
#include "backends/base_operator.hpp"
#include "core/tensor.hpp"
#include "core/base_device.hpp"

namespace spyinfer {

class TestAdd : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建CPU后端
        backend_ = std::make_shared<CPUBackend>();
    }

    void TearDown() override {
        // 清理资源
        backend_.reset();
    }

    std::shared_ptr<CPUBackend> backend_;
};

// 测试基本的Add运算
TEST_F(TestAdd, BasicAdd) {
    // 定义张量形状
    std::array<int64_t, 4> shape = {1, 1, 3, 3};
    size_t numel = shape[0] * shape[1] * shape[2] * shape[3];

    // 创建输入张量a和b
    auto device = backend_->get_devices()[0];
    auto a = Tensor::create(device, shape, DataType::fp32_t);
    auto b = Tensor::create(device, shape, DataType::fp32_t);
    auto output = Tensor::create(device, shape, DataType::fp32_t);

    // 设置测试数据
    std::vector<float> a_data(numel, 1.0f);
    std::vector<float> b_data(numel, 2.0f);

    // 将数据复制到张量
    memcpy(a->data_ptr<float>(), a_data.data(), numel * sizeof(float));
    memcpy(b->data_ptr<float>(), b_data.data(), numel * sizeof(float));

    // 准备参数
    
    
    // params["op_type"] = OperatorType::Add;
    // params["input_a"] = a;
    // params["input_b"] = b;
    // params["output"] = output;

    // 执行前向扩展和计算
    backend_->run({
        {"op_type", OperatorType::Add},
        {"input_a", a},
        {"input_b", b},
        {"output", output},
    });

    // 验证结果
    const float* output_data = output->data_ptr<float>();
    for (size_t i = 0; i < numel; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], a_data[i] + b_data[i]);
    }
}

// 测试不同形状的Add运算
TEST_F(TestAdd, DifferentShapes) {
    std::vector<std::array<int64_t, 4>> shapes = {
        {1, 1, 1, 1},      // 标量
        {1, 1, 100, 100},  // 中等大小
        {1, 4, 128, 1024}   // 较大大小
    };

    for (const auto& shape : shapes) {
        size_t numel = shape[0] * shape[1] * shape[2] * shape[3];

        // 创建张量
        auto device = backend_->get_devices()[0];
        auto a = Tensor::create(device, shape, DataType::fp32_t);
        auto b = Tensor::create(device, shape, DataType::fp32_t);
        //auto output = std::shared_ptr<Tensor>(nullptr);
        auto output = Tensor::create(device, shape, DataType::fp32_t);

        // 生成随机数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);

        std::vector<float> a_data(numel);
        std::vector<float> b_data(numel);
        for (size_t i = 0; i < numel; ++i) {
            a_data[i] = dist(gen);
            b_data[i] = dist(gen);
        }

        // 将数据复制到张量
        memcpy(a->data_ptr<float>(), a_data.data(), numel * sizeof(float));
        memcpy(b->data_ptr<float>(), b_data.data(), numel * sizeof(float));

        // 准备参数
        std::unordered_map<std::string, std::any> params;
        
        params["op_type"] = OperatorType::Add;
        params["input_a"] = a;
        params["input_b"] = b;
        params["output"] = output;

        // 执行前向扩展和计算
        backend_->run({
            {"op_type", OperatorType::Add},
            {"input_a", a},
            {"input_b", b},
            {"output", output},
        });

        // 验证结果
        const float* output_data = output->data_ptr<float>();
        for (size_t i = 0; i < numel; ++i) {
            EXPECT_FLOAT_EQ(output_data[i], a_data[i] + b_data[i]);
        }
    }
}

} // namespace spyinfer

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}