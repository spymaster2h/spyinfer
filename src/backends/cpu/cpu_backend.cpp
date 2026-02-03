#include "cpu_backend.hpp"
#include "cpu_operator.hpp"
#include "cpu_attention.hpp"

#include <cstring>

namespace spyinfer {
CPUBackend::CPUBackend(int thread_num, size_t thread_buffer_size)
    : thread_num_(thread_num), parallel_executor_(std::make_unique<ParallelExecutor>(thread_num, thread_buffer_size))
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

void CPUBackend::copy_data_from_cpu(void* dst, const void* src, size_t size_bytes) { memcpy(dst, src, size_bytes); }

void CPUBackend::copy_data_to_cpu(void* dst, const void* src, size_t size_bytes) { memcpy(dst, src, size_bytes); }

} // namespace spyinfer
