#pragma once

#include "backends/base_operator.hpp"

namespace spyinfer {

class CUDAAdd : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};

class CUDALinear : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};


class CUDAEmbedding : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};

class CUDARMSNorm : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};

class CUDARope : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};

class CUDASwiGLU : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};


class CUDAAttention : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};



class CUDABF16ToFP16 : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};

} // namespace spyinfer
