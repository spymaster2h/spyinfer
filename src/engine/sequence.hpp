#pragma once

#include <vector>
#include <memory>

namespace spyinfer {

enum class SequenceStatus
{
    WAITING,
    RUNNING,
    FINISHED,
};

class Sequence
{
public:
    Sequence(int seq_id, const std::vector<int>& prompt_tokens) : id(seq_id), status(SequenceStatus::WAITING), tokens(prompt_tokens) {}

    int get_id() const { return id; }
    size_t get_len() const { return tokens.size(); }
    void append_token(int token_id) { tokens.push_back(token_id); }
    const std::vector<int>& get_tokens() const { return tokens; }

public:
    int id;
    SequenceStatus status;
    std::vector<int> tokens;
    std::vector<int> block_table; // Maps logical block to physical block_id
    bool is_prefill = true;
    int token_pos = 0; // 当前处理到的token位置
};

} // namespace spyinfer