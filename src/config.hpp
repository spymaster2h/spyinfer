#pragma once

#include "models/model_config.hpp"

class Config
{
public:
    Config(const std::string& model_path)
    {
        model = model_path;
        model_config = ModelConfig(std::filesystem::path(model_path) / "config.json");
        max_model_len = std::min(max_model_len, model_config.max_position_embeddings);  // 限制最大长度不超过模型支持
    }


public:
    std::filesystem::path model;  // 模型目录或名称
    std::string backend_type{"cpu"};  // 后端类型
    int max_num_batched_tokens{16384};  // 单批次最大token数
    int max_num_seqs{512};  // 单批次最大序列数
    int max_model_len{4096};  // 单序列最大长度
    float gpu_memory_utilization{0.9};  // GPU显存利用率上限
    int tensor_parallel_size{1};  // 张量并行进程数
    bool enforce_eager{false};  // 是否强制eager模式
    int eos{-1};  // 终止token id
    int kvcache_block_size{4};  // KV缓存块大小 4token
    int num_kvcache_blocks{100};  // KV缓存块数量


    ModelConfig model_config;  // transformers的模型配置
};