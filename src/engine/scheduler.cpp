#include "scheduler.hpp"
#include "core/base_device.hpp"
#include <algorithm>

namespace spyinfer {

Scheduler::Scheduler(const Config& config, BlockManager& block_manager) : config_(config), block_manager_(block_manager) {}

void Scheduler::add_sequence(Sequence* seq) { waiting_queue_.push_back(seq); }

void Scheduler::_append_slot(int seq_id, int physical_block_id, int token_offset_in_block, std::vector<int>& slot_mapping)
{
    int slot = physical_block_id * config_.kvcache_block_size + token_offset_in_block;
    slot_mapping.push_back(slot);
}

ScheduleOutput Scheduler::schedule()
{
    // 1. Promote sequences from waiting to running if space is available
    auto it = waiting_queue_.begin();
    while (it != waiting_queue_.end())
    {
        Sequence* seq = *it;
        size_t total_blocks_needed = (seq->get_len() + config_.kvcache_block_size - 1) / config_.kvcache_block_size;
        size_t current_blocks = seq->block_table.size();
        size_t new_blocks_needed = (total_blocks_needed > current_blocks) ? (total_blocks_needed - current_blocks) : 0;

        if (block_manager_.get_num_free_blocks() >= new_blocks_needed)
        {
            seq->status = SequenceStatus::RUNNING;
            // Allocate initial blocks
            for (size_t i = 0; i < new_blocks_needed; ++i)
            {
                block_manager_.allocate(seq->block_table);
            }
            running_queue_.push_back(seq);
            it = waiting_queue_.erase(it);
        }
        else
        {
            break; // Not enough blocks for this sequence, stop promoting
        }
    }

    ScheduleOutput output;
    if (running_queue_.empty())
    {
        return output;
    }

    // 2. Build the batch for prefill and decode
    int current_seq_idx = 0;
    output.cu_seqlens.push_back(0);
    for (auto* seq : running_queue_)
    {
        output.context_lens.push_back(seq->get_len());
        output.cu_seqlens.push_back(output.cu_seqlens.back() + seq->get_len() - seq->token_pos);

        if (seq->is_prefill)
        {
            const auto& tokens = seq->get_tokens();
            for (size_t i = seq->token_pos; i < tokens.size(); ++i)
            {
                output.prompt_tokens.push_back(tokens[i]);
                output.seq_idx_mapping.push_back(current_seq_idx);
                int logical_block_idx = i / config_.kvcache_block_size;
                int physical_block_id = seq->block_table[logical_block_idx];
                int token_offset_in_block = i % config_.kvcache_block_size;
                _append_slot(seq->id, physical_block_id, token_offset_in_block, output.slot_mapping);
            }
        }
        else
        {
            size_t num_tokens = seq->get_len();
            if (num_tokens > 0)
            {
                if ((num_tokens - 1) % config_.kvcache_block_size == 0 && num_tokens > 1)
                {
                    if (block_manager_.get_num_free_blocks() > 0)
                    {
                        block_manager_.allocate(seq->block_table);
                    }
                    else
                    {
                        continue;
                    }
                }
                output.prompt_tokens.push_back(seq->tokens.back());
                output.seq_idx_mapping.push_back(current_seq_idx);
                int logical_block_idx = (num_tokens - 1) / config_.kvcache_block_size;
                int physical_block_id = seq->block_table[logical_block_idx];
                int token_offset_in_block = (num_tokens - 1) % config_.kvcache_block_size;
                _append_slot(seq->id, physical_block_id, token_offset_in_block, output.slot_mapping);
            }
        }
        current_seq_idx++;
    }

    // 3. Build padded block tables for all running sequences
    if (!running_queue_.empty())
    {
        size_t max_blocks = 0;
        for (const auto* seq : running_queue_)
        {
            max_blocks = std::max(max_blocks, seq->block_table.size());
        }
        output.block_tables.resize(running_queue_.size() * max_blocks, -1);
        output.max_blocks_per_seq = max_blocks;
        int seq_idx = 0;
        for (const auto* seq : running_queue_)
        {
            const auto& table = seq->block_table;
            for (size_t block_idx = 0; block_idx < table.size(); ++block_idx)
            {
                output.block_tables[seq_idx * max_blocks + block_idx] = table[block_idx];
            }
            seq_idx++;
        }
    }

    return output;
}

} // namespace spyinfer