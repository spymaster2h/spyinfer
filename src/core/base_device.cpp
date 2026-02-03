#include "base_device.hpp"
#include "backends/cpu/cpu_device.hpp"

#ifdef USE_CUDA
#include "backends/cuda/cuda_device.cuh"
#endif

namespace spyinfer {

std::shared_ptr<BaseDevice> BaseDevice::create(DeviceType type, int index)
{
    switch (type)
    {
        case DeviceType::CPU:
            return std::make_shared<CPUDevice>(index);
#ifdef USE_CUDA
        case DeviceType::CUDA:
            return std::make_shared<CUDADevice>(index);
#endif
        default:
            throw "Unsupported device type";
    }
}

} // namespace spyinfer