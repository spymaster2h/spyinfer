#pragma once

#include "core/tensor.hpp"

using namespace spyinfer;

class Context
{
public:
    bool is_prefill = false;
    
    std::shared_ptr<Tensor> k_cache = nullptr;
    std::shared_ptr<Tensor> v_cache = nullptr;
    int* slot_mapping = nullptr;
    int* context_lens = nullptr;
    int* block_tables = nullptr;
    int* seq_idx_map = nullptr;

    int batch_size = 0;

    int max_blocks_per_seq = 0;
    static Context& getInstance();

    void set_kv_cache(std::shared_ptr<Tensor> k_cache_val, std::shared_ptr<Tensor> v_cache_val);

    // Member method to set the context for the current inference step
    void set_context(bool is_prefill_val, int batch_size_val, int* slot_mapping_val, int* context_lens_val, int* block_tables_val,
                          int* seq_idx_map_val, int max_blocks_per_seq_val);

    // Member method to reset the context to its default state
    void reset_context();

private:
    Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
};
