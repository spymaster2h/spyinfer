#pragma once

#include "../base_operator.hpp"


namespace spyinfer {

class Add : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};


class Linear : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};

class Embedding : public BaseOperator
{
    public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};


class RMSNorm : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};



class Rope : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};

class SwiGLU : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};

class Sigmoid : public BaseOperator
{
public:
    void forward_expand(std::unordered_map<std::string, std::any>& params) override;
    void compute(std::unordered_map<std::string, std::any>& params) override;
};


}

