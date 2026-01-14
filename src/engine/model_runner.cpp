#include <numeric>
#include <algorithm>

#include "model_runner.hpp"
#include "utils/context.hpp" // Add include for Context
#include "utils/constant_table.hpp"

namespace spyinfer {

ModelRunner::ModelRunner(const Config& config) : config_(config)
{
    cpu_backend_ = std::make_shared<CPUBackend>();
    model_ = std::make_unique<Qwen3>(config_.model_config, cpu_backend_);
    model_->load_model(config_.model);


    int num_kvcache_blocks = config_.num_kvcache_blocks;
    int kvcache_block_size = config_.kvcache_block_size;
    
    k_cache_tensor_ = cpu_backend_->create_tensor({config.model_config.num_hidden_layers, num_kvcache_blocks, kvcache_block_size, config_.model_config.num_key_value_heads * config_.model_config.head_dim}, DataType::fp32_t);
    v_cache_tensor_ = cpu_backend_->create_tensor({config.model_config.num_hidden_layers, num_kvcache_blocks, kvcache_block_size, config_.model_config.num_key_value_heads * config_.model_config.head_dim}, DataType::fp32_t);
    Context::getInstance().set_kv_cache(k_cache_tensor_, v_cache_tensor_);

    ConstantTable::GetInstance().BuildRopeTable(config.model_config.max_position_embeddings, config.model_config.head_dim, config.model_config.rope_theta);
}


ModelRunner::~ModelRunner() {}

void ModelRunner::execute_model(ScheduleOutput& schedule_output, std::list<Sequence*>& running_queue)
{
    if (schedule_output.prompt_tokens.empty())
    {
        Context::getInstance().reset_context(); // Reset context if no sequences
        return;
    }

    int total_tokens = schedule_output.prompt_tokens.size();
    int num_seqs = schedule_output.context_lens.size();
    bool has_prefill_seqs = false;

    // 1. Prepare input tensors, especially positions
    std::vector<int> positions_vec;
    std::vector<bool> is_prefill_flags; // Keep track of which sequence was prefilled
    int seq_idx = 0;
    for (const auto& seq : running_queue)
    {
        if (seq->status != SequenceStatus::RUNNING) continue;

        is_prefill_flags.push_back(seq->is_prefill);
        if (seq->is_prefill)
        {
            has_prefill_seqs = true;
            for (size_t i = seq->token_pos; i < seq->get_len(); ++i)
            {
                positions_vec.push_back(i);
            }
        }
        else
        {
            positions_vec.push_back(seq->get_len() - 1);
        }
        seq_idx++;
    }

    // Set global context for attention operator
    Context::getInstance().set_context(has_prefill_seqs, // Set to true if at least one sequence is prefilling
                                       num_seqs, schedule_output.slot_mapping.data(), schedule_output.context_lens.data(),
                                       schedule_output.block_tables.data(), schedule_output.seq_idx_mapping.data(), schedule_output.max_blocks_per_seq); 

    auto input_ids_tensor = cpu_backend_->create_tensor({1, 1, 1, static_cast<int64_t>(total_tokens)}, DataType::int32_t);
    std::copy(schedule_output.prompt_tokens.begin(), schedule_output.prompt_tokens.end(), input_ids_tensor->data_ptr<int>());

    auto positions_tensor = cpu_backend_->create_tensor({1, 1, 1, static_cast<int64_t>(total_tokens)}, DataType::int32_t);
    std::copy(positions_vec.begin(), positions_vec.end(), positions_tensor->data_ptr<int>());

    // 2. Execute model forward pass
    std::shared_ptr<Tensor> logits =model_->compute_logits(model_->forward(input_ids_tensor, positions_tensor));

    // 3. Process logits and sample next tokens
    std::vector<int> next_token_ids;
    float* logits_data = logits->data_ptr<float>();
    int vocab_size = config_.model_config.vocab_size;

    int token_offset = 0;
    for (size_t i = 0; i < num_seqs; ++i)
    {
        auto seq_it = running_queue.begin();
        std::advance(seq_it, i);
        Sequence* current_seq = *seq_it;

        int num_processed_tokens = is_prefill_flags[i] ? current_seq->get_len() - current_seq->token_pos : 1;

        // Logits for the last token of the current sequence
        float* last_token_logits = logits_data + (token_offset + num_processed_tokens - 1) * vocab_size;

        int next_token = std::distance(last_token_logits, std::max_element(last_token_logits, last_token_logits + vocab_size));
        next_token_ids.push_back(next_token);
        token_offset += num_processed_tokens;
    }

    // 4. Update sequences in the running queue
    auto seq_it = running_queue.begin();
    for (size_t i = 0; i < next_token_ids.size(); ++i)
    {
        if (seq_it != running_queue.end())
        {
            (*seq_it)->append_token(next_token_ids[i]);

            // Flip flag after successful prefill
            if ((*seq_it)->is_prefill)
            {
                (*seq_it)->is_prefill = false;
            }

            if (next_token_ids[i] == config_.model_config.eos_token_id || (*seq_it)->get_len() >= config_.max_model_len)
            {
                (*seq_it)->status = SequenceStatus::FINISHED;
            }
            ++seq_it;
        }
    }

    // Reset context after model execution
    Context::getInstance().reset_context();
}

} // namespace spyinfer