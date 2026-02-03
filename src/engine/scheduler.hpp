#pragma once

#include <list>
#include <vector>
#include <memory>
#include <unordered_map>

#include "config.hpp"
#include "sequence.hpp"
#include "block_manager.hpp"
#include "core/tensor.hpp"

namespace spyinfer {

struct ScheduleOutput
{
    std::vector<int> prompt_tokens;
    std::vector<int> slot_mapping;
    std::vector<int> context_lens;
    std::vector<int> seq_idx_mapping;
    std::vector<int> block_tables;
    std::vector<int> cu_seqlens;
    int max_blocks_per_seq = 0;
};

class Scheduler

{
public:
    Scheduler(const Config& config, BlockManager& block_manager);

    void add_sequence(Sequence* seq);

    ScheduleOutput schedule();

    std::list<Sequence*>& get_running_queue() { return running_queue_; }

    std::list<Sequence*>& get_waiting_queue() { return waiting_queue_; }

private:
    void _append_slot(int seq_id, int physical_block_id, int token_offset_in_block, std::vector<int>& slot_mapping);

    const Config& config_;

    BlockManager& block_manager_;

    std::list<Sequence*> waiting_queue_;
    std::list<Sequence*> running_queue_;

    int next_seq_id_ = 0;
};

} // namespace spyinfer
