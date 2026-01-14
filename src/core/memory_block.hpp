#pragma once

#include <memory>
#include "base_device.hpp"

namespace spyinfer {

class MemoryBlock
{
public:
    MemoryBlock(size_t size_bytes, std::shared_ptr<BaseDevice> device);

    ~MemoryBlock();

    MemoryBlock(const MemoryBlock&) = delete;
    MemoryBlock& operator=(const MemoryBlock&) = delete;

    // Allow move operations
    MemoryBlock(MemoryBlock&&) noexcept;
    MemoryBlock& operator=(MemoryBlock&&) noexcept;

    void* data() { return data_ptr_; }
    const void* data() const { return data_ptr_; }
    size_t size() const { return size_bytes_; }
    const std::shared_ptr<BaseDevice>& device() const { return device_; }

private:
    void* data_ptr_ = nullptr;
    size_t size_bytes_ = 0;
    std::shared_ptr<BaseDevice> device_;
};
} // namespace spyinfer