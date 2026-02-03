#pragma once


#include <vector>
#include "core/tensor.hpp"

using namespace spyinfer;

class Context
{
public:
    bool is_prefill = true;
    
    std::shared_ptr<Tensor> k_cache = nullptr;
    std::shared_ptr<Tensor> v_cache = nullptr;
    int* slot_mapping = nullptr;  // token id -> physic offset (当前token的kv需要插入的kv cahche位置)
    int* context_lens = nullptr; //seq id -> seq len
    int* block_tables = nullptr; //[seq_id][logic_block_id] --> physic_block_id
    int* seq_idx_map = nullptr; //token id -> seq id
    int* cu_seqlens = nullptr;



    int* slot_mapping_device = nullptr;  // token id -> physic offset (当前token的kv需要插入的kv cahche位置)
    int* context_lens_device = nullptr; //seq id -> seq len
    int* block_tables_device = nullptr; //[seq_id][logic_block_id] --> physic_block_id
    int* seq_idx_map_device = nullptr; //token id -> seq id
    int* cu_seqlens_device = nullptr;

    
    int max_seq_len = 0;
    int batch_size = 0;

    int max_blocks_per_seq = 0;
    static Context& getInstance();

    void set_kv_cache(std::shared_ptr<Tensor> k_cache_val, std::shared_ptr<Tensor> v_cache_val);

    void set_context(bool is_prefill_val, int batch_size_val, int* slot_mapping_val, int* context_lens_val, int* block_tables_val,
                          int* seq_idx_map_val, int *cu_seqlens_val, int max_seq_len_val, int max_blocks_per_seq_val);


    void set_context_device(std::vector<int>& slot_mapping_val, std::vector<int>& context_lens_val, std::vector<int>& block_tables_val, std::vector<int>& seq_idx_map, std::vector<int>& cu_seqlens);
    void reset_context();

private:
    Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
};
