#pragma once

#include "cpu_parallel.hpp"
#include "../base_backend.hpp"

namespace spyinfer {

class CPUBackend : public BaseBackend
{
public:
    CPUBackend(int thread_num = std::min(4, (int)(4 + std::thread::hardware_concurrency()) / 2), size_t thread_buffer_size = 0);

    ~CPUBackend() = default;

    // virtual void compute(std::unordered_map<std::string, std::any>& params) override;
    int get_thread_num() const { return thread_num_; }

    ParallelExecutor* get_parallel_executor() { return parallel_executor_.get(); }

    virtual void copy_data_from_cpu(void* dst, const void* src, size_t size_bytes) override;

    virtual void copy_data_to_cpu(void* dst, const void* src, size_t size_bytes) override;

    std::string get_backend_name() const override { return "CPU"; }

private:
    int thread_num_;
    std::unique_ptr<ParallelExecutor> parallel_executor_;
};
} // namespace spyinfer