#pragma once

#include "../base_operator.hpp"

namespace spyinfer {

class Attention : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};
} // namespace spyinfer