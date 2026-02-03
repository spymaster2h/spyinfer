#include "tensor.hpp"

namespace spyinfer {

size_t get_dtype_size(DataType dtype)
{
    switch (dtype)
    {
        case DataType::bf16_t: return 2;
        case DataType::fp16_t: return 2;
        case DataType::fp32_t: return 4;
        case DataType::int32_t: return 4;
        default: throw std::invalid_argument("Unsupported data type");
    }
}

DataType string_to_dtype(const std::string& dtype_str)
{
    if (dtype_str == "F32")
    {
        return DataType::fp32_t;
    }
    else if (dtype_str == "BF16")
    {
        return DataType::bf16_t;
    }
    else if (dtype_str == "F16")
    {
        return DataType::fp16_t;
    }
    throw std::invalid_argument("Unknown data type: " + dtype_str);
}

Tensor::Tensor(std::shared_ptr<MemoryBlock> memory_block, const std::array<int64_t, 4>& shape, DataType dtype, size_t offset_bytes)
    : memory_block_(std::move(memory_block)), shape_(shape), dtype_(dtype), offset_bytes_(offset_bytes)
{
    numel_ = 1;
    for (const auto& dim : shape_)
    {
        numel_ *= dim;
    }
    compute_strides();

    size_t required_bytes = numel_ * get_dtype_size(dtype_);
    if (offset_bytes_ + required_bytes > memory_block_->size())
    {
        throw std::out_of_range("Tensor view goes beyond MemoryBlock boundaries.");
    }
}

void Tensor::compute_strides()
{
    strides_[3] = 1;
    strides_[2] = shape_[3];
    strides_[1] = shape_[2] * strides_[2];
    strides_[0] = shape_[1] * strides_[1];
}

std::shared_ptr<Tensor> Tensor::create(std::shared_ptr<BaseDevice> device, const std::array<int64_t, 4>& shape, DataType dtype)
{
    size_t numel = 1;
    for (const auto& dim : shape)
    {
        numel *= dim;
    }
    size_t size_bytes = numel * get_dtype_size(dtype);

    return std::make_shared<Tensor>(std::make_shared<MemoryBlock>(size_bytes, device), shape, dtype, (size_t)0);
}


std::shared_ptr<Tensor> Tensor::create_from_buffer(std::shared_ptr<MemoryBlock> memory_block, const std::array<int64_t, 4>& shape, DataType dtype, size_t offset_bytes)
{
    size_t numel_view = 1;
    for (const auto& dim : shape)
    {
        numel_view *= dim;
    }
    size_t required_bytes_for_view = numel_view * get_dtype_size(dtype);
    if (offset_bytes + required_bytes_for_view > memory_block->size())
    {
        throw std::out_of_range("Attempted to create a tensor view that exceeds its parent MemoryBlock boundaries.");
    }

    return std::make_shared<Tensor>(std::move(memory_block), shape, dtype, offset_bytes);
}


void Tensor::reshape(const std::array<int64_t, 4>& new_shape)
{
    shape_ = new_shape;
    compute_strides();
    numel_ = 1;
    for (const auto& dim : shape_)
    {
        numel_ *= dim;
    }
}


std::shared_ptr<Tensor> Tensor::view(const std::array<int64_t, 4>& new_shape)
{
    if (new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3] != numel_)
    {
        throw std::invalid_argument("Reshaped tensor must have the same number of elements as the original.");
    }
    return std::make_shared<Tensor>(memory_block_, new_shape, dtype_, offset_bytes_);
}

} // namespace spyinfer