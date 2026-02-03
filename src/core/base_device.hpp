#pragma once

#include <string>
#include <memory>

namespace spyinfer {

enum class DeviceType
{
    CPU,
    CUDA
};


class BaseDevice
{
public:
    virtual ~BaseDevice() = default;

    virtual DeviceType type() const = 0;
    
    virtual int index() const = 0;
    
    bool is_cpu() const { return type() == DeviceType::CPU; }
    bool is_cuda() const { return type() == DeviceType::CUDA; }

    virtual std::string to_string() const = 0;

    virtual void* allocate(size_t size_bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void synchronize() const = 0;
    virtual size_t get_total_memory() const = 0;
    virtual size_t get_free_memory() const = 0;

    bool operator==(const BaseDevice& other) const {
        return type() == other.type() && index() == other.index();
    }
    bool operator!=(const BaseDevice& other) const { return !(*this == other); }

    struct Hash
    {
        size_t operator()(const BaseDevice& d) const {
            return static_cast<size_t>(d.type()) * 1000 + d.index();
        }
    };

    static std::shared_ptr<BaseDevice> create(DeviceType type, int index = 0);
};

} // namespace spyinfer