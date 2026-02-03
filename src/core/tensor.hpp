#pragma once

#include <array>
#include <numeric>
#include <string>
#include <stdexcept>

#include "memory_block.hpp"
#include "base_device.hpp"

namespace spyinfer {

// Enum for data types
enum class DataType
{
    bf16_t,
    fp16_t,
    fp32_t,
    int32_t
};

using bf16_t = uint16_t;

size_t get_dtype_size(DataType dtype);

DataType string_to_dtype(const std::string& dtype_str);

class Tensor
{
public:
    static std::shared_ptr<Tensor> create(std::shared_ptr<BaseDevice> device, const std::array<int64_t, 4>& shape, DataType dtype);
    static std::shared_ptr<Tensor> create_from_buffer(std::shared_ptr<MemoryBlock> memory_block, const std::array<int64_t, 4>& shape, DataType dtype,
                                                      size_t offset_bytes = 0);

    std::shared_ptr<Tensor> view(const std::array<int64_t, 4>& new_shape);
    void reshape(const std::array<int64_t, 4>& new_shape);

    const std::array<int64_t, 4>& shape() const { return shape_; }
    const std::array<int64_t, 4>& strides() const { return strides_; }
    DataType dtype() const { return dtype_; }

    void set_dtype(DataType dtype) { dtype_ = dtype; }
    size_t numel() const { return numel_; }
    size_t offset_bytes() const { return offset_bytes_; }

    std::shared_ptr<MemoryBlock> memory_block() const { return memory_block_; }

    template <typename T>
    T* data_ptr()
    {
        if (get_dtype_size(dtype_) != sizeof(T))
        {
            throw std::runtime_error("Mismatched data type access for Tensor.");
        }
        return reinterpret_cast<T*>(reinterpret_cast<char*>(memory_block_->data()) + offset_bytes_);
    }

    template <typename T>
    const T* data_ptr() const
    {
        if (get_dtype_size(dtype_) != sizeof(T))
        {
            throw std::runtime_error("Mismatched data type access for Tensor.");
        }
        return reinterpret_cast<const T*>(reinterpret_cast<const char*>(memory_block_->data()) + offset_bytes_);
    }

    Tensor(std::shared_ptr<MemoryBlock> memory_block, const std::array<int64_t, 4>& shape, DataType dtype, size_t offset_bytes);

private:
    void compute_strides();

    std::shared_ptr<MemoryBlock> memory_block_; // Owned by shared_ptr
    std::array<int64_t, 4> shape_;
    std::array<int64_t, 4> strides_;
    DataType dtype_;
    size_t numel_;
    size_t offset_bytes_; // Byte offset into the memory_block_
};
} // namespace spyinfer