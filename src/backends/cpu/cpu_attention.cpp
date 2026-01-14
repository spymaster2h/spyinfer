#include "cpu_attention.hpp"
#include "core/tensor.hpp"
#include "cpu_backend.hpp"
#include "x64/kernels.hpp"
#include "utils/context.hpp"

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace spyinfer {

void Attention::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    for (const auto& input_name : operator_input_names[OperatorType::Attention])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("Attention operator missing input: " + input_name);
        }
    }
    is_parallelizable_ = true;
}

void Attention::compute(std::unordered_map<std::string, std::any>& params)
{

    const auto& context = Context::getInstance();
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto cpu_backend = dynamic_cast<CPUBackend*>(backend.get());

    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input_q = std::any_cast<std::shared_ptr<Tensor>>(params["input_q"]);
    auto input_k = std::any_cast<std::shared_ptr<Tensor>>(params["input_k"]);
    auto input_v = std::any_cast<std::shared_ptr<Tensor>>(params["input_v"]);
    auto positions = std::any_cast<std::shared_ptr<Tensor>>(params["position_ids"]);
    auto layer = std::any_cast<int>(params["layer_ids"]);




    auto k_cache = context.k_cache;
    auto v_cache = context.v_cache;
    auto block_tables_ptr = context.block_tables; //[seq_id][logic_block_id] --> physic_block_id （历史kvcache位置）
    auto context_lens_ptr = context.context_lens; //seq id -> seq len
    auto slot_mapping_ptr = context.slot_mapping; // token id -> physic offset (当前token的kv需要插入的kv cahche位置)
    auto seq_idx_map_ptr = context.seq_idx_map; //token id -> seq id
    auto batch_size = context.batch_size;
    auto kvcache_block_size = context.k_cache->shape()[2];



    const int num_query_tokens = input_q->shape()[1];
    const int num_q_heads = input_q->shape()[2];
    const int head_size = input_q->shape()[3];
    const int num_kv_heads = input_k->shape()[2];

    float* q_ptr = input_q->data_ptr<float>();
    float* k_ptr = input_k->data_ptr<float>();
    float* v_ptr = input_v->data_ptr<float>();
    float* k_cache_ptr = k_cache->data_ptr<float>() + layer * k_cache->strides()[0];
    float* v_cache_ptr = v_cache->data_ptr<float>() + layer * v_cache->strides()[0];
    const int* positions_ptr = positions->data_ptr<int>();

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

    for (int i = 0; i < num_query_tokens; ++i)
    {
        const int slot = slot_mapping_ptr[i];
        const int cache_offset = slot * num_kv_heads * head_size;
        const int input_offset = i * num_kv_heads * head_size;
        std::copy_n(k_ptr + input_offset, num_kv_heads * head_size, k_cache_ptr + cache_offset);
        std::copy_n(v_ptr + input_offset, num_kv_heads * head_size, v_cache_ptr + cache_offset);
    }
    const int execute_threads = cpu_backend->get_thread_num();
    const int total_work_items = num_query_tokens * num_q_heads;

    //对于每个头计算注意力
    auto attention_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
        const size_t items_per_thread = (total_work_items + execute_threads - 1) / execute_threads;
        const size_t work_start_idx = thread_idx * items_per_thread;
        const size_t work_end_idx = std::min(work_start_idx + items_per_thread, (size_t)total_work_items);
        if (work_start_idx >= work_end_idx) return;

        for (size_t work_idx = work_start_idx; work_idx < work_end_idx; ++work_idx)
        {
            const int current_token_idx = work_idx / num_q_heads;
            const int h = work_idx % num_q_heads;

            const int q_token_offset = current_token_idx * num_q_heads * head_size;
            float* q_head_ptr = q_ptr + q_token_offset + h * head_size;
            float* o_head_ptr = output->data_ptr<float>() + q_token_offset + h * head_size;
            std::fill_n(o_head_ptr, head_size, 0.0f);

            const int seq_idx_in_batch = seq_idx_map_ptr[current_token_idx];
            const int context_len = context_lens_ptr[seq_idx_in_batch];
            const int* block_table_ptr = block_tables_ptr + seq_idx_in_batch * context.max_blocks_per_seq;

            thread_buffer.resize(context_len * sizeof(float));
            std::fill(thread_buffer.begin(), thread_buffer.end(), std::byte(0x00));
            float* scores = reinterpret_cast<float*>(thread_buffer.data());
            const int kv_head_idx = h * num_kv_heads / num_q_heads; // GQA mapping

            for (int ctx_i = 0; ctx_i < context_len; ++ctx_i)
            {
                if (ctx_i > positions_ptr[current_token_idx])
                {
                    scores[ctx_i] = -INFINITY;   //causal mask
                    continue;
                }
                const int logical_block_idx = ctx_i / kvcache_block_size;
                const int offset_in_block = ctx_i % kvcache_block_size;
                const int physical_block_id = block_table_ptr[logical_block_idx];
                const int k_slot = physical_block_id * kvcache_block_size + offset_in_block;
                const int k_offset = k_slot * num_kv_heads * head_size + kv_head_idx * head_size;
                scores[ctx_i] = x64_dot_product_fp32(q_head_ptr, k_cache_ptr + k_offset, head_size) * scale;
            }

            float max_score = -INFINITY;
            for (int i = 0; i < context_len; ++i) max_score = std::max(max_score, scores[i]);
            float score_sum = 0.0f;
            for (int i = 0; i < context_len; ++i)
            {
                scores[i] = std::exp(scores[i] - max_score);
                score_sum += scores[i];
            }
            if (score_sum > 1e-9)
            {
                for (int i = 0; i < context_len; ++i) scores[i] /= score_sum;
            }

            for (int ctx_i = 0; ctx_i < context_len; ++ctx_i)
            {
                const int logical_block_idx = ctx_i / kvcache_block_size;
                const int offset_in_block = ctx_i % kvcache_block_size;
                const int physical_block_id = block_table_ptr[logical_block_idx];
                const int v_slot = physical_block_id * kvcache_block_size + offset_in_block;
                const int v_offset = v_slot * num_kv_heads * head_size + kv_head_idx * head_size;
                float* v_head_ptr = v_cache_ptr + v_offset;
                x64_vectorized_accumulate_fp32(o_head_ptr, scores[ctx_i], v_head_ptr, head_size);
            }
        }
    };
    cpu_backend->get_parallel_executor()->execute(attention_task);

}

} // namespace spyinfer