#pragma once

#include "core/base_device.hpp"

namespace spyinfer {

class CPUDevice : public BaseDevice
{
public:
    explicit CPUDevice(int index = 0) : index_(index) {}

    DeviceType type() const override { return DeviceType::CPU; }
    int index() const override { return index_; }
    std::string to_string() const override { return "CPU" + (index_ > 0 ? ":" + std::to_string(index_) : ""); }

    void* allocate(size_t size_bytes) override;
    void deallocate(void* ptr) override;
    void synchronize() const override {}
    size_t get_total_memory() const override;
    size_t get_free_memory() const override;

private:
    int index_;
};

} // namespace spyinfer
