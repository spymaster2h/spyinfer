#pragma once

#include <string>
#include <unordered_map>
#include <any>


#include <vector>
#include <string>

namespace spyinfer {


enum class OperatorType
{
    Add,
    Linear,
    Embedding,
    RMSNorm,
    Rope,
    Attention,
    SwiGLU,
    Sigmoid,
};



static std::unordered_map<OperatorType,  std::vector<std::string>> operator_input_names = {
    {OperatorType::Add, {"backend", "op_type", "output", "input_a", "input_b"}},
    {OperatorType::Linear, {"backend", "op_type", "output", "input", "weight", "bias"}},
    {OperatorType::Embedding, {"backend", "op_type", "output", "input", "weight"}},
    {OperatorType::RMSNorm, {"backend", "op_type", "output", "input", "weight", "epsilon"}},
    {OperatorType::Rope, {"backend", "op_type", "output", "input", "position_ids"}},
    {OperatorType::Attention, {"backend", "op_type", "output", "input_q", "input_k", "input_v", "position_ids", "layer_ids"}},
    {OperatorType::SwiGLU, {"backend", "op_type", "output", "input_gate", "input_up"}},
    {OperatorType::Sigmoid, {"backend", "op_type", "output", "input"}},

};



class BaseOperator
{
public:

    virtual void forward_expand(std::unordered_map<std::string, std::any>& params) = 0;

    virtual void compute(std::unordered_map<std::string, std::any>& params) = 0;

    virtual bool is_parallelizable() const { return is_parallelizable_; }

protected:
    bool is_parallelizable_ = false;
};
} // namespace spyinfer