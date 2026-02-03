#include <numeric>
#include <iostream>
#include <algorithm>

#include "llm_engine.hpp"

namespace spyinfer {

LLMEngine::LLMEngine(const std::string& model_path, const std::unordered_map<std::string, ConfigValue>& params) : config_(model_path)
{
    if (params.contains("max_num_batched_tokens")) config_.max_num_batched_tokens = std::get<int>(params.at("max_num_batched_tokens"));
    if (params.contains("max_num_seqs")) config_.max_num_seqs = std::get<int>(params.at("max_num_seqs"));
    if (params.contains("max_model_len")) config_.max_model_len = std::get<int>(params.at("max_model_len"));
    if (params.contains("kvcache_block_size")) config_.kvcache_block_size = std::get<int>(params.at("kvcache_block_size"));
    if (params.contains("num_kvcache_blocks")) config_.num_kvcache_blocks = std::get<int>(params.at("num_kvcache_blocks"));
    if (params.contains("backend_type")) config_.backend_type = std::get<std::string>(params.at("backend_type"));

    block_manager_ = std::make_unique<BlockManager>(config_.num_kvcache_blocks);
    scheduler_ = std::make_unique<Scheduler>(config_, *block_manager_);
    model_runner_ = std::make_unique<ModelRunner>(config_);
    tokenizer_ = std::make_unique<Tokenizer>(config_);
}

LLMEngine::~LLMEngine() {}

int LLMEngine::add_request(const std::vector<std::pair<std::string, std::string>>& messages)
{
    const int request_id = next_request_id_++;
    std::string prompt = tokenizer_->apply_chat_template(messages);
    std::vector<int> prompt_tokens = tokenizer_->encode(prompt);

    auto it = all_sequences_.emplace(request_id, Sequence(request_id, prompt_tokens));

    Sequence* seq_ptr = &it.first->second;
    scheduler_->add_sequence(seq_ptr);

    return request_id;
}

void LLMEngine::append_to_request(int request_id, const std::pair<std::string, std::string>& message)
{
    auto it = all_sequences_.find(request_id);
    if (it == all_sequences_.end())
    {
        // Error: request ID not found
        return;
    }

    Sequence& seq = it->second;
    if (seq.status != SequenceStatus::FINISHED)
    {
        // Error: cannot append to a running sequence
        return;
    }
    seq.token_pos = seq.tokens.size();
    std::string new_text_chunk = tokenizer_->apply_chat_template({message});
    auto new_tokens = tokenizer_->encode(new_text_chunk);

    if (!new_tokens.empty() && new_tokens.front() == config_.model_config.bos_token_id)
    {
        new_tokens.erase(new_tokens.begin());
    }
    if (!new_tokens.empty() && new_tokens.back() == config_.model_config.eos_token_id)
    {
        new_tokens.pop_back();
    }

    seq.tokens.insert(seq.tokens.end(), new_tokens.begin(), new_tokens.end());

    seq.status = SequenceStatus::WAITING;
    seq.is_prefill = true;
    scheduler_->add_sequence(&seq);
}

void LLMEngine::step()
{
    ScheduleOutput output = scheduler_->schedule();

    if (output.prompt_tokens.empty())
    {
        return;
    }

    model_runner_->execute_model(output, scheduler_->get_running_queue());

    auto& running_q = scheduler_->get_running_queue();
    for (auto it = running_q.begin(); it != running_q.end();)
    {
        if ((*it)->status == SequenceStatus::FINISHED)
        {
            it = running_q.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool LLMEngine::is_request_finished(int request_id) const
{
    auto it = all_sequences_.find(request_id);
    if (it != all_sequences_.end())
    {
        return it->second.status == SequenceStatus::FINISHED;
    }
    return true;
}

bool LLMEngine::has_unfinished_requests() const { return !scheduler_->get_waiting_queue().empty() || !scheduler_->get_running_queue().empty(); }

std::string LLMEngine::get_output(int request_id) const
{
    auto it = all_sequences_.find(request_id);
    if (it != all_sequences_.end())
    {
        const auto& tokens = it->second.get_tokens();
        auto assistant_prompt = tokenizer_->encode("<|im_start|>assistant\n");
        if (!assistant_prompt.empty() && assistant_prompt.front() == config_.model_config.bos_token_id)
        {
            assistant_prompt.erase(assistant_prompt.begin());
        }
        if (!assistant_prompt.empty() && assistant_prompt.back() == config_.model_config.eos_token_id)
        {
            assistant_prompt.pop_back();
        }

        auto last_match = std::find_end(tokens.begin(), tokens.end(), assistant_prompt.begin(), assistant_prompt.end());
        if (last_match != tokens.end())
        {
            std::vector<int> response_tokens(last_match + assistant_prompt.size(), tokens.end());
            return tokenizer_->decode(response_tokens);
        }
        return tokenizer_->decode(tokens);
    }
    return "";
}

std::vector<int> LLMEngine::get_output_tokens(int request_id) const
{
    auto it = all_sequences_.find(request_id);
    if (it != all_sequences_.end())
    {
        return it->second.get_tokens();
    }
    return {};
}

void LLMEngine::remove_request(int request_id)
{
    auto it = all_sequences_.find(request_id);
    if (it != all_sequences_.end())
    {
        block_manager_->free(it->second.block_table);
        all_sequences_.erase(it);
    }
}

} // namespace spyinfer
