#pragma once

#include <string>
#include <memory>

namespace spyinfer {

// Enum for device types
enum class DeviceType
{
    CPU,
    CUDA
};

// Forward declarations
class BaseDevice;
class CPUDevice;
class CUDADevice;

// Abstract base class for devices
class BaseDevice
{
public:
    virtual ~BaseDevice() = default;

    // Get device type
    virtual DeviceType type() const = 0;
    
    // Get device index
    virtual int index() const = 0;
    
    // Check device type
    bool is_cpu() const { return type() == DeviceType::CPU; }
    bool is_cuda() const { return type() == DeviceType::CUDA; }

    // Convert to string representation
    virtual std::string to_string() const = 0;

    // Device management functions
    virtual void* allocate(size_t size_bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void synchronize() const = 0;
    virtual size_t get_total_memory() const = 0;
    virtual size_t get_free_memory() const = 0;

    // Comparison operators
    bool operator==(const BaseDevice& other) const {
        return type() == other.type() && index() == other.index();
    }
    bool operator!=(const BaseDevice& other) const { return !(*this == other); }

    // Hash support
    struct Hash
    {
        size_t operator()(const BaseDevice& d) const {
            return static_cast<size_t>(d.type()) * 1000 + d.index();
        }
    };

    // Factory method to create devices
    static std::shared_ptr<BaseDevice> create(DeviceType type, int index = 0);
};

} // namespace spyinfer