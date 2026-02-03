#pragma once

#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

struct ModelConfig
{
    std::string model_type;
    std::string hidden_act;
    int hidden_size;
    int intermediate_size;
    int num_key_value_heads;
    int num_hidden_layers;
    int num_attention_heads;
    bool tie_word_embeddings;
    float rope_theta;
    float rms_norm_eps;
    int head_dim;
    int bos_token_id;
    int eos_token_id;
    int max_position_embeddings;
    int vocab_size;

    ModelConfig() = default;

    ModelConfig(const std::string& model_config_path)
    {
        nlohmann::json config_json;
        std::ifstream config_file(model_config_path);
        config_file >> config_json;
        model_type = config_json.at("model_type").get<std::string>();
        hidden_act = config_json.at("hidden_act").get<std::string>();
        hidden_size = config_json.value("hidden_size", -1);
        intermediate_size = config_json.value("intermediate_size", -1);
        num_key_value_heads = config_json.value("num_key_value_heads", -1);
        num_hidden_layers = config_json.value("num_hidden_layers", -1);
        num_attention_heads = config_json.value("num_attention_heads", -1);
        tie_word_embeddings = config_json.value("tie_word_embeddings", false);
        rope_theta = config_json.value("rope_theta", 1000000.0f);
        rms_norm_eps = config_json.value("rms_norm_eps", 1e-6f);
        head_dim = config_json.value("head_dim", hidden_size / num_attention_heads);
        bos_token_id = config_json.value("bos_token_id", -1);
        eos_token_id = config_json.value("eos_token_id", -1);
        max_position_embeddings = config_json.value("max_position_embeddings", 16384);
        vocab_size = config_json.value("vocab_size", -1);
    }
};