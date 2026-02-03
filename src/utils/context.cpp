#include "context.hpp"



#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

Context::Context() = default;
Context& Context::getInstance()
{
    static Context instance;
    return instance;
}

void Context::set_context(bool is_prefill_val, int batch_size_val, int* slot_mapping_val, int* context_lens_val, int* block_tables_val,
                          int* seq_idx_map_val, int *cu_seqlens_val, int max_seq_len_val, int max_blocks_per_seq_val)
{
    is_prefill = is_prefill_val;
    slot_mapping = slot_mapping_val;
    context_lens = context_lens_val;
    block_tables = block_tables_val;
    seq_idx_map = seq_idx_map_val;
    batch_size = batch_size_val;
    max_blocks_per_seq = max_blocks_per_seq_val;
    cu_seqlens = cu_seqlens_val;
    max_seq_len = max_seq_len_val;

}

void Context::set_kv_cache(std::shared_ptr<Tensor> k_cache_val, std::shared_ptr<Tensor> v_cache_val)
{
    k_cache = k_cache_val;
    v_cache = v_cache_val;
}

void Context::reset_context()
{
    is_prefill = false;
    batch_size = 0;
    max_blocks_per_seq = 0;
    slot_mapping = nullptr;
    context_lens = nullptr;
    block_tables = nullptr;
    seq_idx_map = nullptr;
}


void Context::set_context_device(std::vector<int>& slot_mapping_val, std::vector<int>& context_lens_val, std::vector<int>& block_tables_val, std::vector<int>& seq_idx_map_val, std::vector<int>& cu_seqlens_val)
{
    #ifdef USE_CUDA

    auto copy_host_to_device = [](const std::vector<int>& host_vec, int*& device_ptr) {
        if (host_vec.empty()) {
            device_ptr = nullptr;
            return; 
        }
        cudaError_t err = cudaMalloc((void**)&device_ptr, host_vec.size() * sizeof(int));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMemcpy(device_ptr, host_vec.data(), host_vec.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(device_ptr);
            device_ptr = nullptr;
            throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
        }
    };

    copy_host_to_device(slot_mapping_val, slot_mapping_device);
    copy_host_to_device(context_lens_val, context_lens_device);
    copy_host_to_device(block_tables_val, block_tables_device);
    copy_host_to_device(seq_idx_map_val, seq_idx_map_device);
    copy_host_to_device(cu_seqlens_val, cu_seqlens_device);


    #endif
}

