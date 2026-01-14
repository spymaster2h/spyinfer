#include "cuda_device.hpp"

#ifdef USE_CUDA
#define CUDA_CHECK(call)                                                                                \
    do                                                                                                  \
    {                                                                                                   \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess)                                                                         \
        {                                                                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));                                          \
        }                                                                                               \
    } while (0)
#endif

namespace spyinfer {

void* CUDADevice::allocate(size_t size_bytes)
{
    if (size_bytes == 0)
        return nullptr;
    
#ifdef USE_CUDA
    void* ptr;
    CUDA_CHECK(cudaSetDevice(index_));
    CUDA_CHECK(cudaMalloc(&ptr, size_bytes));
    return ptr;
#else
    throw std::runtime_error("This binary was not compiled with CUDA support.");
#endif
}

void CUDADevice::deallocate(void* ptr)
{
    if (ptr)
    {
#ifdef USE_CUDA
        CUDA_CHECK(cudaSetDevice(index_));
        CUDA_CHECK(cudaFree(ptr));
#else
        throw std::runtime_error("This binary was not compiled with CUDA support.");
#endif
    }
}

void CUDADevice::synchronize() const
{
#ifdef USE_CUDA
    CUDA_CHECK(cudaSetDevice(index_));
    CUDA_CHECK(cudaDeviceSynchronize());
#else
    throw std::runtime_error("This binary was not compiled with CUDA support.");
#endif
}

size_t CUDADevice::get_total_memory() const
{
#ifdef USE_CUDA
    size_t total = 0;
    CUDA_CHECK(cudaSetDevice(index_));
    CUDA_CHECK(cudaMemGetInfo(nullptr, &total));
    return total;
#else
    throw std::runtime_error("This binary was not compiled with CUDA support.");
#endif
}

size_t CUDADevice::get_free_memory() const
{
#ifdef USE_CUDA
    size_t free = 0;
    CUDA_CHECK(cudaSetDevice(index_));
    CUDA_CHECK(cudaMemGetInfo(&free, nullptr));
    return free;
#else
    throw std::runtime_error("This binary was not compiled with CUDA support.");
#endif
}

} // namespace spyinfer
