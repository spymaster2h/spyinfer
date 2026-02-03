#pragma once

#include <iostream>

#include "core/tensor.hpp"
#include "utils/precision.hpp"


#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif


using namespace spyinfer;



static void print_tensor(const std::shared_ptr<Tensor>& tensor, int len)
{
    std::cout << "Tensor data: ";
    for (int i = 0; i < len; ++i)
    {
        if (tensor->dtype() == DataType::fp32_t)
            std::cout << tensor->data_ptr<float>()[i] << " ";
        else if (tensor->dtype() == DataType::bf16_t)
            std::cout << half_to_float(tensor->data_ptr<uint16_t>()[i]) << " ";
        else if (tensor->dtype() == DataType::int32_t)
            std::cout << tensor->data_ptr<int>()[i] << " ";
    }
    std::cout << std::endl;
}




static void print_tensor_cuda(const std::shared_ptr<Tensor>& tensor, int len)
{
    std::cout << "Tensor data: ";

    #ifdef USE_CUDA
    size_t size = (tensor->numel() >= len ? len : tensor->numel()) * get_dtype_size(tensor->dtype());
    void* cpu_ptr = malloc(size);

    if (tensor->dtype() == DataType::fp32_t || tensor->dtype() == DataType::int32_t)
    {
        cudaMemcpy(cpu_ptr, tensor->data_ptr<float>(), size, cudaMemcpyDeviceToHost);
    }
    else
    {
         cudaMemcpy(cpu_ptr, tensor->data_ptr<uint16_t>(), size, cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < len; ++i)
    {
        if (tensor->dtype() == DataType::fp32_t)
            std::cout << ((float*)cpu_ptr)[i] << " ";
        else if (tensor->dtype() == DataType::fp16_t)
            std::cout << half_to_float(((uint16_t*)cpu_ptr)[i]) << " ";
        else if (tensor->dtype() == DataType::int32_t)
            std::cout << ((int*)cpu_ptr)[i] << " ";
    }
    std::cout << std::endl;

    free(cpu_ptr);
    #endif
}
