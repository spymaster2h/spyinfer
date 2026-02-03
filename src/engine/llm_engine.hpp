#pragma once
#include <variant>
#include <list>
#include <unordered_map>
#include <string>

#include "config.hpp"
#include "block_manager.hpp"
#include "scheduler.hpp"
#include "model_runner.hpp"
#include "utils/tokenizer.hpp"
#include "utils/context.hpp"

namespace spyinfer {

using ConfigValue = std::variant<int, float, bool, std::string>;

class LLMEngine
{
public:
    LLMEngine(const std::string& model_path, const std::unordered_map<std::string, ConfigValue>& params = {});
    ~LLMEngine();

    // Starts a new conversation and returns its unique ID
    int add_request(const std::vector<std::pair<std::string, std::string>>& messages);
    
    // Appends a new user message to an existing conversation
    void append_to_request(int request_id, const std::pair<std::string, std::string>& message);

    // Runs one step of the engine
    void step();

    // Checks if a specific request is finished
    bool is_request_finished(int request_id) const;

    void remove_request(int request_id);

    // Gets the full output string for a request
    std::string get_output(int request_id) const;

    // Gets the full output tokens for a request
    std::vector<int> get_output_tokens(int request_id) const;
    
    // Checks if there are any pending or running requests
    bool has_unfinished_requests() const;

    Tokenizer* get_tokenizer() { return tokenizer_.get(); }

private:
    Config config_;
    std::unique_ptr<BlockManager> block_manager_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<ModelRunner> model_runner_;
    std::unique_ptr<Tokenizer> tokenizer_;

    std::unordered_map<int, Sequence> all_sequences_;
    int next_request_id_ = 0;
};

} // namespace spyinfer