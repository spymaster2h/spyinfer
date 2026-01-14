#include "cpu_backend.hpp"
#include "cpu_operator.hpp"
#include "cpu_attention.hpp"

namespace spyinfer {
CPUBackend::CPUBackend(int thread_num, size_t thread_buffer_size) : thread_num_(thread_num), parallel_executor_(std::make_unique<ParallelExecutor>(thread_num, thread_buffer_size))
{
    device_cnt_ = 1;
    device_ids_ = {0};
    device_ids_ratio_ = {1};

    devices_ = {BaseDevice::create(DeviceType::CPU, 0)};

    ops_[OperatorType::Add] = std::make_unique<Add>();
    ops_[OperatorType::RMSNorm] = std::make_unique<RMSNorm>();  
    ops_[OperatorType::Embedding] = std::make_unique<Embedding>();
    ops_[OperatorType::Rope] = std::make_unique<Rope>();
    ops_[OperatorType::Linear] = std::make_unique<Linear>();
    ops_[OperatorType::SwiGLU] = std::make_unique<SwiGLU>();
    ops_[OperatorType::Embedding] = std::make_unique<Embedding>();
    ops_[OperatorType::Sigmoid] = std::make_unique<Sigmoid>();
    ops_[OperatorType::Attention] = std::make_unique<Attention>();
}

std::shared_ptr<MemoryBlock> CPUBackend::allocate(size_t size_bytes)
{
    return std::make_shared<MemoryBlock>(size_bytes, devices_[0]);
}


std::shared_ptr<Tensor> CPUBackend::create_tensor(const std::array<int64_t, 4>& shape, DataType dtype)
{
    return Tensor::create(devices_[0], shape, dtype);
}



} // namespace spyinfer
