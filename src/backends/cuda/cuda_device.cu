#include "cuda_device.cuh"

namespace spyinfer {

void* CUDADevice::allocate(size_t size_bytes)
{
    if (size_bytes == 0)
        return nullptr;
    
    void* ptr;
    CUDA_CHECK(cudaSetDevice(index_));
    CUDA_CHECK(cudaMalloc(&ptr, size_bytes));
    return ptr;
}

void CUDADevice::deallocate(void* ptr)
{
    if (ptr)
    {
        CUDA_CHECK(cudaSetDevice(index_));
        CUDA_CHECK(cudaFree(ptr));
    }
}

void CUDADevice::synchronize() const
{
    CUDA_CHECK(cudaSetDevice(index_));
    CUDA_CHECK(cudaDeviceSynchronize());
}

size_t CUDADevice::get_total_memory() const
{
    size_t total = 0;
    CUDA_CHECK(cudaSetDevice(index_));
    CUDA_CHECK(cudaMemGetInfo(nullptr, &total));

    return total;
}

size_t CUDADevice::get_free_memory() const
{
    size_t free = 0;
    CUDA_CHECK(cudaSetDevice(index_));
    CUDA_CHECK(cudaMemGetInfo(&free, nullptr));
    return free;
}

} // namespace spyinfer
