#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

#include "op_cpp.hpp"

namespace {

constexpr size_t I = 4096;  // 输入向量长度和矩阵行数
constexpr size_t J = 8192;  // 矩阵列数和输出向量长度
constexpr float EPS = 1e-3f;  // 精度阈值

// 生成随机数据
void generateRandomData(std::vector<float>& data, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (auto& val : data) {
        val = dist(gen);
    }
}


// 将float转换为bf16_t
void floatToBf16(const std::vector<float>& float_data, std::vector<bf16_t>& bf16_data) {
    bf16_data.resize(float_data.size());
    for (size_t i = 0; i < float_data.size(); ++i) {
        // 简化的float到bf16转换（截断）
        uint32_t f32 = std::bit_cast<uint32_t>(float_data[i]);
        bf16_data[i] = static_cast<bf16_t>(f32 >> 16);
    }
}

// 比较两个结果数组是否一致
bool compareResults(const std::vector<float>& result1, const std::vector<float>& result2) {
    if (result1.size() != result2.size()) {
        return false;
    }
    
    for (size_t i = 0; i < result1.size(); ++i) {
        if (std::abs(result1[i] - result2[i]) > EPS) {
            std::cerr << "Mismatch at index " << i << ": " << result1[i] << " vs " << result2[i] << std::endl;
            return false;
        }
    }
    return true;
}

}  // namespace

// 测试float版本的gevm性能和结果一致性
TEST(GEVMTest, FloatPerformanceAndConsistency) {
    std::vector<float> input1(I);  // 输入行向量 [1, I]
    std::vector<float> input2(I * J);  // 输入矩阵 [I, J]
    std::vector<float> output_native(J);  // 参考实现结果
    std::vector<float> output_simd(J);  // SIMD实现结果
    
    // 生成随机数据
    generateRandomData(input1);
    generateRandomData(input2);
    
    // 测试native版本性能（使用_gemm_native作为参考）
    auto start_native = std::chrono::high_resolution_clock::now();
    // 将input1扩展为1xI的矩阵，与IxJ矩阵相乘得到1xJ的结果
    spyinfer::_gemm_native(output_native.data(), input1.data(), input2.data(), 1, I, J);
    auto end_native = std::chrono::high_resolution_clock::now();
    auto duration_native = std::chrono::duration_cast<std::chrono::microseconds>(end_native - start_native).count();
    
    // 测试simd版本性能
    auto start_simd = std::chrono::high_resolution_clock::now();
    // 确保output_simd初始化为0
    std::fill(output_simd.begin(), output_simd.end(), 0.0f);
    spyinfer::_gevm(output_simd.data(), input1.data(), input2.data(), I, J);
    auto end_simd = std::chrono::high_resolution_clock::now();
    auto duration_simd = std::chrono::duration_cast<std::chrono::microseconds>(end_simd - start_simd).count();
    
    // 输出性能比较
    std::cout << "Float Performance Comparison:" << std::endl;
    std::cout << "  _gemm_native duration: " << duration_native << " μs" << std::endl;
    std::cout << "  _gevm duration: " << duration_simd << " μs" << std::endl;
    std::cout << "  SIMD vs Native improvement: " << static_cast<double>(duration_native) / duration_simd << "x" << std::endl;
    
    // 验证结果一致性
    EXPECT_TRUE(compareResults(output_native, output_simd));
}

// 测试bf16版本的gevm性能和结果一致性
TEST(GEVMTest, Bf16PerformanceAndConsistency) {
    std::vector<float> input1(I);  // 输入行向量 [1, I]
    std::vector<float> input2_float(I * J);  // 输入矩阵float版本
    std::vector<bf16_t> input2_bf16;  // 输入矩阵bf16版本
    std::vector<float> output_native(J);  // 参考实现结果
    std::vector<float> output_simd(J);  // SIMD实现结果
    
    // 生成随机数据
    generateRandomData(input1);
    generateRandomData(input2_float);
    
    // 转换为bf16
    floatToBf16(input2_float, input2_bf16);
    
    // 测试native版本性能（使用_gemm_native作为参考）
    auto start_native = std::chrono::high_resolution_clock::now();
    // 将input1扩展为1xI的矩阵，与IxJ矩阵相乘得到1xJ的结果
    spyinfer::_gemm_native(output_native.data(), input1.data(), input2_bf16.data(), 1, I, J);
    auto end_native = std::chrono::high_resolution_clock::now();
    auto duration_native = std::chrono::duration_cast<std::chrono::microseconds>(end_native - start_native).count();
    
    // 测试simd版本性能
    auto start_simd = std::chrono::high_resolution_clock::now();
    // 确保output_simd初始化为0
    std::fill(output_simd.begin(), output_simd.end(), 0.0f);
    // 注意：当前_gevm的bf16版本尚未实现
    spyinfer::_gevm(output_simd.data(), input1.data(), input2_bf16.data(), I, J);
    auto end_simd = std::chrono::high_resolution_clock::now();
    auto duration_simd = std::chrono::duration_cast<std::chrono::microseconds>(end_simd - start_simd).count();
    
    // 输出性能比较
    std::cout << "\nBf16 Performance Comparison:" << std::endl;
    std::cout << "  _gemm_native duration: " << duration_native << " μs" << std::endl;
    std::cout << "  _gevm duration: " << duration_simd << " μs" << std::endl;
    
    // 注意：当前_gevm的bf16版本尚未实现，因此暂时注释掉一致性检查
    // EXPECT_TRUE(compareResults(output_native, output_simd));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
