#pragma once

#include "backends/base_backend.hpp"
#include "cuda_device.cuh"

namespace spyinfer {

class CUDABackend : public BaseBackend
{
public:
    CUDABackend();
    ~CUDABackend();
    std::string get_backend_name() const override
    {
        return "CUDA";
    }

    void print_device_properties();


    virtual void copy_data_from_cpu(void* dst, const void* src, size_t size_bytes) override;

    virtual void copy_data_to_cpu(void* dst, const void* src, size_t size_bytes) override;

private:
};

} // namespace spyinfer
