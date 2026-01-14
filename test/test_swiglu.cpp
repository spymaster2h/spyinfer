#include <gtest/gtest.h>
#include <memory>
#include <cmath>

#include "backends/cpu/cpu_backend.hpp"
#include "core/tensor.hpp"


using namespace spyinfer;


class TestSwiGLU : public ::testing::Test
{
protected:
    std::shared_ptr<CPUBackend> backend;

    void SetUp() override { backend = std::make_shared<CPUBackend>(); }

    void TearDown() override { backend.reset(); }

    // 手动实现SwiGLU计算逻辑用于验证
    void manual_swiglu(float* output, const float* gate, const float* up, int len)
    {
        for (int i = 0; i < len; i++)
        {
            float silu_gate = gate[i] / (1.0f + std::exp(-gate[i])); // SiLU activation
            output[i] = silu_gate * up[i];                           // Element-wise multiplication
        }
    }
};

TEST_F(TestSwiGLU, ParallelSwiGLU) {
    // 测试并行情况，元素数量>100000
    std::array<int64_t, 4> input_shape = {1, 1, 100, 1001};  // 100 * 1001 = 100,100 > 100,000
    auto input_gate = Tensor::create(backend->get_devices()[0], input_shape, DataType::fp32_t);
    auto input_up = Tensor::create(backend->get_devices()[0], input_shape, DataType::fp32_t);
    auto output = Tensor::create(backend->get_devices()[0], input_shape, DataType::fp32_t);

    // 初始化输入数据 - 使用更多数据点
    std::vector<float> gate_data(100100);
    std::vector<float> up_data(100100);
    
    for (int i = 0; i < 100100; i++) {
        gate_data[i] = static_cast<float>(i % 100) / 10.0f - 5.0f;  // Values from -5 to 4.9
        up_data[i] = static_cast<float>(i % 70) / 15.0f - 2.0f;      // Values from -2 to ~2.67
    }

    std::copy(gate_data.begin(), gate_data.end(), input_gate->data_ptr<float>());
    std::copy(up_data.begin(), up_data.end(), input_up->data_ptr<float>());

    // 准备参数
    std::unordered_map<std::string, std::any> params;
    params["op_type"] = OperatorType::SwiGLU;
    params["output"] = output;
    params["input_gate"] = input_gate;
    params["input_up"] = input_up;

    // 执行forward_expand
    backend->run(params);

    // 计算期望结果
    std::vector<float> expected_output(100100);
    manual_swiglu(expected_output.data(), gate_data.data(), up_data.data(), 100100);

    // 验证输出 - 检查一些关键点
    float* actual_output = output->data_ptr<float>();
    for (int i = 0; i < 100100; i += 10000) {  // 每隔10000个元素检查一次
        EXPECT_NEAR(actual_output[i], expected_output[i], 1e-5)
            << "Mismatch at index " << i << ": expected " << expected_output[i] 
            << ", got " << actual_output[i];
    }
    
    // 检查最后一个元素
    EXPECT_NEAR(actual_output[100099], expected_output[100099], 1e-5)
        << "Mismatch at last index: expected " << expected_output[100099] 
        << ", got " << actual_output[100099];
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}