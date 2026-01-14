#include "base_device.hpp"
#include "cpu_device.hpp"
#include "cuda_device.hpp"

namespace spyinfer {

std::shared_ptr<BaseDevice> BaseDevice::create(DeviceType type, int index)
{
    switch (type)
    {
        case DeviceType::CPU:
            return std::make_shared<CPUDevice>(index);
        case DeviceType::CUDA:
            return std::make_shared<CUDADevice>(index);
        default:
            throw std::invalid_argument("Unsupported device type");
    }
}

} // namespace spyinfer