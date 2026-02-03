#include "memory_block.hpp"

namespace spyinfer {

MemoryBlock::MemoryBlock(size_t size_bytes, std::shared_ptr<BaseDevice> device) 
    : size_bytes_(size_bytes), device_(std::move(device))
{
    if (size_bytes_ == 0)
    {
        data_ptr_ = nullptr;
        return;
    }

    data_ptr_ = device_->allocate(size_bytes_);
}

MemoryBlock::~MemoryBlock()
{
    if (data_ptr_ != nullptr)
    {
        device_->deallocate(data_ptr_);
    }
}

MemoryBlock::MemoryBlock(MemoryBlock&& other) noexcept
    : data_ptr_(other.data_ptr_), size_bytes_(other.size_bytes_), device_(std::move(other.device_))
{
    other.data_ptr_ = nullptr;
    other.size_bytes_ = 0;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) noexcept
{
    if (this != &other)
    {
        if (data_ptr_ != nullptr)
        {
            device_->deallocate(data_ptr_);
        }

        data_ptr_ = other.data_ptr_;
        size_bytes_ = other.size_bytes_;
        device_ = std::move(other.device_);

        other.data_ptr_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

} // namespace spyinfer