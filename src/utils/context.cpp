#include "context.hpp"

Context::Context() = default;
Context& Context::getInstance()
{
    static Context instance;
    return instance;
}

void Context::set_context(bool is_prefill_val, int batch_size_val, int* slot_mapping_val, int* context_lens_val, int* block_tables_val,
                          int* seq_idx_map_val, int max_blocks_per_seq_val)
{
    is_prefill = is_prefill_val;
    slot_mapping = slot_mapping_val;
    context_lens = context_lens_val;
    block_tables = block_tables_val;
    seq_idx_map = seq_idx_map_val;
    batch_size = batch_size_val;
    max_blocks_per_seq = max_blocks_per_seq_val;
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
