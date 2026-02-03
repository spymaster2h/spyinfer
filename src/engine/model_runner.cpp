#include <numeric>
#include <algorithm>

#include "model_runner.hpp"
#include "utils/context.hpp" 
#include "utils/constant_table.hpp"
#include "utils/precision.hpp"

#ifdef USE_CUDA
#include "backends/cuda/cuda_backend.hpp"
#endif

namespace spyinfer {

ModelRunner::ModelRunner(const Config& config) : config_(config)
{
    if (config_.backend_type == "cpu")
    {
        backend_ = std::make_shared<CPUBackend>();
        backend_->set_compute_dtype(DataType::fp32_t);
    }
    else
    {
#ifdef USE_CUDA
        backend_ = std::make_shared<CUDABackend>();
        backend_->set_compute_dtype(DataType::fp16_t);
#endif
    }
    model_ = std::make_unique<Qwen3>(config_.model_config, backend_);
    model_->load_model(config_.model);

    int num_kvcache_blocks = config_.num_kvcache_blocks;
    int kvcache_block_size = config_.kvcache_block_size;

    k_cache_tensor_ = backend_->create_tensor({config.model_config.num_hidden_layers, num_kvcache_blocks, kvcache_block_size,
                                               config_.model_config.num_key_value_heads * config_.model_config.head_dim},
                                              backend_->get_compute_dtype());
    v_cache_tensor_ = backend_->create_tensor({config.model_config.num_hidden_layers, num_kvcache_blocks, kvcache_block_size,
                                               config_.model_config.num_key_value_heads * config_.model_config.head_dim},
                                              backend_->get_compute_dtype());
    Context::getInstance().set_kv_cache(k_cache_tensor_, v_cache_tensor_);

    ConstantTable::GetInstance().BuildRopeTable(config.model_config.max_position_embeddings, config.model_config.head_dim,
                                                config.model_config.rope_theta);
}

ModelRunner::~ModelRunner() {}

void ModelRunner::execute_model(ScheduleOutput& schedule_output, std::list<Sequence*>& running_queue)
{
    if (schedule_output.prompt_tokens.empty())
    {
        Context::getInstance().reset_context();
        return;
    }

    int total_tokens = schedule_output.prompt_tokens.size();
    int num_seqs = schedule_output.context_lens.size();
    bool is_prefill_seqs = true;

    std::vector<int> positions_vec;
    std::vector<bool> is_prefill_flags;
    int seq_idx = 0;
    int max_seq_len = 1;
    for (const auto& seq : running_queue)
    {
        if (seq->status != SequenceStatus::RUNNING) continue;

        is_prefill_flags.push_back(seq->is_prefill);
        if (seq->is_prefill)
        {
            max_seq_len = std::max(max_seq_len, (int)seq->get_len());
            for (size_t i = seq->token_pos; i < seq->get_len(); ++i)
            {
                positions_vec.push_back(i);
            }
        }
        else
        {
            is_prefill_seqs = false;
            positions_vec.push_back(seq->get_len() - 1);
        }
        seq_idx++;
    }
    Context::getInstance().set_context(is_prefill_seqs, num_seqs, schedule_output.slot_mapping.data(), schedule_output.context_lens.data(),
                                       schedule_output.block_tables.data(), schedule_output.seq_idx_mapping.data(), schedule_output.cu_seqlens.data(),
                                       max_seq_len, schedule_output.max_blocks_per_seq);

    Context::getInstance().set_context_device(schedule_output.slot_mapping, schedule_output.context_lens, schedule_output.block_tables,
                                              schedule_output.seq_idx_mapping, schedule_output.cu_seqlens);
    auto input_ids_tensor = backend_->create_tensor({1, 1, 1, static_cast<int64_t>(total_tokens)}, DataType::int32_t);
    backend_->copy_data_from_cpu(input_ids_tensor->data_ptr<int>(), schedule_output.prompt_tokens.data(), total_tokens * sizeof(int));
    auto positions_tensor = backend_->create_tensor({1, 1, 1, static_cast<int64_t>(total_tokens)}, DataType::int32_t);
    backend_->copy_data_from_cpu(positions_tensor->data_ptr<int>(), positions_vec.data(), total_tokens * sizeof(int));

    std::shared_ptr<Tensor> logits = model_->compute_logits(model_->forward(input_ids_tensor, positions_tensor));

    std::vector<int> next_token_ids;

    float* logits_data = nullptr;

    if (backend_->get_backend_name() != "CPU")
    {
        std::vector<float> cpu_logits;
        cpu_logits.resize(logits->numel());
        logits_data = cpu_logits.data();
        backend_->copy_data_to_cpu(logits_data, logits->data_ptr<uint16_t>(), logits->numel() * sizeof(uint16_t));

        for (int i = logits->numel() - 1; i >= 0; i--)
        {
            uint16_t* src = (uint16_t*)logits_data + i;
            float* dst = logits_data + i;
            *dst = half_to_float(*src);
        }
    }
    else
    {
        logits_data = logits->data_ptr<float>();
    }
    int vocab_size = config_.model_config.vocab_size;
    int token_offset = 0;
    for (size_t i = 0; i < num_seqs; ++i)
    {
        auto seq_it = running_queue.begin();
        std::advance(seq_it, i);
        Sequence* current_seq = *seq_it;

        int num_processed_tokens = is_prefill_flags[i] ? current_seq->get_len() - current_seq->token_pos : 1;
        float* last_token_logits = logits_data + (token_offset + num_processed_tokens - 1) * vocab_size;

        int next_token = std::distance(last_token_logits, std::max_element(last_token_logits, last_token_logits + vocab_size));
        next_token_ids.push_back(next_token);
        token_offset += num_processed_tokens;
    }

    auto seq_it = running_queue.begin();
    for (size_t i = 0; i < next_token_ids.size(); ++i)
    {
        if (seq_it != running_queue.end())
        {
            (*seq_it)->append_token(next_token_ids[i]);

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

    Context::getInstance().reset_context();
}

} // namespace spyinfer