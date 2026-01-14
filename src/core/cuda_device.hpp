#pragma once

#include "base_device.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace spyinfer {

class CUDADevice : public BaseDevice
{
public:
    explicit CUDADevice(int index = 0) : index_(index) {}

    DeviceType type() const override { return DeviceType::CUDA; }
    int index() const override { return index_; }
    std::string to_string() const override { return "CUDA:" + std::to_string(index_); }

    void* allocate(size_t size_bytes) override;
    void deallocate(void* ptr) override;
    void synchronize() const override;
    size_t get_total_memory() const override;
    size_t get_free_memory() const override;

private:
    int index_;
};

} // namespace spyinfer
