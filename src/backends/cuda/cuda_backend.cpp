#include "cuda_backend.hpp"
#include "cuda_operator.hpp"
//#include "cuda_attention.hpp"

namespace spyinfer {

CUDABackend::CUDABackend()
{

    print_device_properties();

    device_cnt_ = 1; // Assuming 1 GPU for now
    device_ids_ = {0};
    devices_ = {BaseDevice::create(DeviceType::CUDA, 0)};

    // Register operators
    ops_[OperatorType::Add] = std::make_unique<CUDAAdd>();
    ops_[OperatorType::Linear] = std::make_unique<CUDALinear>();
    ops_[OperatorType::Embedding] = std::make_unique<CUDAEmbedding>();
    ops_[OperatorType::RMSNorm] = std::make_unique<CUDARMSNorm>();
    ops_[OperatorType::Rope] = std::make_unique<CUDARope>();
    ops_[OperatorType::SwiGLU] = std::make_unique<CUDASwiGLU>();
    ops_[OperatorType::Attention] = std::make_unique<CUDAAttention>();

    ops_[OperatorType::BF16ToFP16] = std::make_unique<CUDABF16ToFP16>();
}

CUDABackend::~CUDABackend() {}



void CUDABackend::copy_data_from_cpu(void* dst, const void* src, size_t size_bytes)
{
    CUDA_CHECK(cudaSetDevice(device_ids_[0]));
    CUDA_CHECK(cudaMemcpy(dst, src, size_bytes, cudaMemcpyHostToDevice));
}

void CUDABackend::copy_data_to_cpu(void* dst, const void* src, size_t size_bytes)
{
    CUDA_CHECK(cudaSetDevice(device_ids_[0]));
    CUDA_CHECK(cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToHost));
}

void CUDABackend::print_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("=== GPU Hardware Organization ===\n\n");
    int i = 0;
    //printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    // for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    
    printf("Device %d: %s\n", i, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SMs (Multiprocessors): %d\n", prop.multiProcessorCount);
    printf("  Warp Size: %d threads\n", prop.warpSize);
    printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Dimensions of a Block: (%d, %d, %d)\n", 
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Dimensions of a Grid: (%d, %d, %d)\n", 
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    printf("\n  === Memory Hierarchy ===\n");
    printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Registers per Block: %d\n", prop.regsPerBlock);
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("  Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth: %.2f GB/s\n\n", 
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    //}
}

} // namespace spyinfer
