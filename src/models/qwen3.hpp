#pragma once

#include <optional>

#include "model_config.hpp"
#include "core/tensor.hpp"
#include "utils/safetensors_reader.hpp"
#include "backends/base_backend.hpp"

#include "utils/debug_helper.hpp"
using namespace spyinfer;

#define max_temp_size 100

class Qwen3Weight
{
    struct Block
    {
        Block(const ModelConfig& config, const SafeTensorsReader& safetensors, int layer_index)
            : attn_q(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".self_attn.q_proj.weight")),
              attn_k(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".self_attn.k_proj.weight")),
              attn_v(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".self_attn.v_proj.weight")),
              attn_o(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".self_attn.o_proj.weight")),
              mlp_down(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".mlp.down_proj.weight")),
              mlp_gate(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".mlp.gate_proj.weight")),
              mlp_up(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".mlp.up_proj.weight")),
              input_norm(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".input_layernorm.weight")),
              post_norm(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".post_attention_layernorm.weight")),
              attn_q_norm(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".self_attn.q_norm.weight")),
              attn_k_norm(safetensors.get_tensor("model.layers." + std::to_string(layer_index) + ".self_attn.k_norm.weight"))
        {
        }

        std::shared_ptr<Tensor> attn_q, attn_k, attn_v, attn_o;
        std::shared_ptr<Tensor> attn_q_norm, attn_k_norm;
        std::shared_ptr<Tensor> mlp_down, mlp_gate, mlp_up;
        std::shared_ptr<Tensor> input_norm, post_norm;

        std::shared_ptr<Tensor> attn_q_bias, attn_k_bias, attn_v_bias;
    };

public:
    Qwen3Weight(const ModelConfig& config, const SafeTensorsReader& safetensors)
        : embed(safetensors.get_tensor("model.embed_tokens.weight")),
          norm(safetensors.get_tensor("model.norm.weight")),
          lm_head(config.tie_word_embeddings ? embed : safetensors.get_tensor("lm_head.weight"))
    {
        // blocks赋值
        blocks.reserve(config.num_hidden_layers);
        for (std::size_t i = 0; i < config.num_hidden_layers; ++i)
        {
            blocks.emplace_back(config, safetensors, i);
        }
    }

public:
    std::vector<Block> blocks;
    std::shared_ptr<Tensor> embed;
    std::shared_ptr<Tensor> norm;
    std::shared_ptr<Tensor> lm_head;
};

class Qwen3
{
public:
    Qwen3(const ModelConfig& config, std::shared_ptr<BaseBackend> backend) : config_(config), backend_(backend)
    {
        DataType dtype = DataType::fp32_t;

        if (backend_->get_backend_name() == "CUDA") dtype = DataType::fp16_t;
        hidden_states = backend_->create_tensor({1, 1, max_temp_size, config_.hidden_size}, dtype);
        norm_1 = backend_->create_tensor({1, 1, max_temp_size, config_.hidden_size}, dtype);
        q = backend_->create_tensor({1, 1, max_temp_size, config_.num_attention_heads * config_.head_dim}, dtype);
        k = backend_->create_tensor({1, 1, max_temp_size, config_.num_key_value_heads * config_.head_dim}, dtype);
        v = backend_->create_tensor({1, 1, max_temp_size, config_.num_key_value_heads * config_.head_dim}, dtype);
        att_output = backend_->create_tensor({1, 1, max_temp_size, config_.num_attention_heads * config_.head_dim}, dtype);
        att_proj_o = backend_->create_tensor({1, 1, max_temp_size, config_.hidden_size}, dtype);

        mlp_gate = backend_->create_tensor({1, 1, max_temp_size, config_.intermediate_size}, dtype);
        mlp_up = backend_->create_tensor({1, 1, max_temp_size, config_.intermediate_size}, dtype);
        swiglu_output = backend_->create_tensor({1, 1, max_temp_size, config_.intermediate_size}, dtype);
        logits = backend_->create_tensor({1, 1, max_temp_size, config_.vocab_size}, dtype);
    }
    ~Qwen3() = default;

    void load_model(const std::string& model_path)
    {
        SafeTensorsReader reader;
        reader.load_weights(model_path, backend_);
        weights_ = std::make_unique<Qwen3Weight>(config_, reader);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input_ids, std::shared_ptr<Tensor> positions)
    {
        const int64_t batch_size = input_ids->numel();

        if (batch_size < max_temp_size)
        {
            hidden_states->reshape({1, 1, batch_size, config_.hidden_size});
            norm_1->reshape({1, 1, batch_size, config_.hidden_size});
            q->reshape({1, 1, batch_size, config_.num_attention_heads * config_.head_dim});
            k->reshape({1, 1, batch_size, config_.num_key_value_heads * config_.head_dim});
            v->reshape({1, 1, batch_size, config_.num_key_value_heads * config_.head_dim});
            att_output->reshape({1, 1, batch_size, config_.num_attention_heads * config_.head_dim});
            att_proj_o->reshape({1, 1, batch_size, config_.hidden_size});

            mlp_gate->reshape({1, 1, batch_size, config_.intermediate_size});
            mlp_up->reshape({1, 1, batch_size, config_.intermediate_size});
            swiglu_output->reshape({1, 1, batch_size, config_.intermediate_size});
            
        }

        // 1.embedding
        backend_->run({
            {"op_type", OperatorType::Embedding},
            {"output", hidden_states},
            {"input", input_ids},
            {"weight", weights_->embed},
        });

        // 2. forward block
        for (int i = 0; i < config_.num_hidden_layers; ++i)
        {
            auto& block = weights_->blocks[i];

            // input rmsnorm
            backend_->run({
                {"op_type", OperatorType::RMSNorm},
                {"output", norm_1},
                {"input", hidden_states},
                {"weight", block.input_norm},
                {"epsilon", config_.rms_norm_eps},
            });

            // compute qkv
            q->reshape({1, 1, batch_size, config_.num_attention_heads * config_.head_dim});
            k->reshape({1, 1, batch_size, config_.num_key_value_heads * config_.head_dim});
            backend_->run({
                {"op_type", OperatorType::Linear},
                {"output", q},
                {"input", norm_1},
                {"weight", block.attn_q},
                {"bias", block.attn_q_bias},
            });
            backend_->run({
                {"op_type", OperatorType::Linear},
                {"output", k},
                {"input", norm_1},
                {"weight", block.attn_k},
                {"bias", block.attn_k_bias},
            });
            backend_->run({
                {"op_type", OperatorType::Linear},
                {"output", v},
                {"input", norm_1},
                {"weight", block.attn_v},
                {"bias", block.attn_v_bias},
            });

            // q-k norm
            const bool has_qk_norm = block.attn_q_norm && block.attn_k_norm;
            if (has_qk_norm)
            {
                q->reshape({1, batch_size, config_.num_attention_heads, config_.head_dim});
                k->reshape({1, batch_size, config_.num_key_value_heads, config_.head_dim});
                backend_->run({
                    {"op_type", OperatorType::RMSNorm},
                    {"output", q},
                    {"input", q},
                    {"weight", block.attn_q_norm},
                    {"epsilon", config_.rms_norm_eps},
                });
                backend_->run({
                    {"op_type", OperatorType::RMSNorm},
                    {"output", k},
                    {"input", k},
                    {"weight", block.attn_k_norm},
                    {"epsilon", config_.rms_norm_eps},
                });
            }

            // q-k rope
            backend_->run({
                {"op_type", OperatorType::Rope},
                {"output", q},
                {"input", q},
                {"position_ids", positions},
            });
            backend_->run({
                {"op_type", OperatorType::Rope},
                {"output", k},
                {"input", k},
                {"position_ids", positions},
            });

            // self-attention
            backend_->run({
                {"op_type", OperatorType::Attention},
                {"output", att_output},
                {"input_q", q},
                {"input_k", k},
                {"input_v", v},
                {"position_ids", positions},
                {"layer_ids", i},
            });

            backend_->run({
                {"op_type", OperatorType::Linear},
                {"output", att_proj_o},
                {"input", att_output},
                {"weight", block.attn_o},
                {"bias", std::shared_ptr<Tensor>(nullptr)},
            });

            // Residual connection
            backend_->run({
                {"op_type", OperatorType::Add},
                {"output", hidden_states},
                {"input_a", hidden_states},
                {"input_b", att_proj_o},
            });

            // post rmsnorm
            backend_->run({
                {"op_type", OperatorType::RMSNorm},
                {"output", att_proj_o},
                {"input", hidden_states},
                {"weight", block.post_norm},
                {"epsilon", config_.rms_norm_eps},
            });

            // MLP
            backend_->run({
                {"op_type", OperatorType::Linear},
                {"output", mlp_gate},
                {"input", att_proj_o},
                {"weight", block.mlp_gate},
                {"bias", std::shared_ptr<Tensor>(nullptr)},
            });

            backend_->run({
                {"op_type", OperatorType::Linear},
                {"output", mlp_up},
                {"input", att_proj_o},
                {"weight", block.mlp_up},
                {"bias", std::shared_ptr<Tensor>(nullptr)},
            });

            backend_->run({
                {"op_type", OperatorType::SwiGLU},
                {"output", swiglu_output},
                {"input_gate", mlp_gate},
                {"input_up", mlp_up},
            });


            backend_->run({
                {"op_type", OperatorType::Linear},
                {"output", att_proj_o},
                {"input", swiglu_output},
                {"weight", block.mlp_down},
                {"bias", std::shared_ptr<Tensor>(nullptr)},
            });

            // if(backend_->get_backend_name() != "CPU")
            //     print_tensor_cuda(swiglu_output, 10);
            // else
            //     print_tensor(swiglu_output, 10);
           

            // Residual connection
            backend_->run({
                {"op_type", OperatorType::Add},
                {"output", hidden_states},
                {"input_a", hidden_states},
                {"input_b", att_proj_o},
            });
        }

        return hidden_states;
    }

    std::shared_ptr<Tensor> compute_logits(std::shared_ptr<Tensor> hidden_states)
    {
        const auto batch_size = hidden_states->shape()[2];
        logits->reshape({1, 1, batch_size, config_.vocab_size});
        backend_->run({
            {"op_type", OperatorType::RMSNorm},
            {"output", hidden_states},
            {"input", hidden_states},
            {"weight", weights_->norm},
            {"epsilon", config_.rms_norm_eps},
        });

        backend_->run({
            {"op_type", OperatorType::Linear},
            {"output", logits},
            {"input", hidden_states},
            {"weight", weights_->lm_head},
            {"bias", std::shared_ptr<Tensor>(nullptr)},
        });

        // if(backend_->get_backend_name() != "CPU")
        //     print_tensor_cuda(logits, 10);
        // else
        //     print_tensor(logits, 10);
        return logits;
    }

private:
    ModelConfig config_;
    std::unique_ptr<Qwen3Weight> weights_;
    std::shared_ptr<BaseBackend> backend_;

    std::shared_ptr<Tensor> hidden_states{nullptr};
    std::shared_ptr<Tensor> norm_1{nullptr};
    std::shared_ptr<Tensor> q{nullptr};
    std::shared_ptr<Tensor> k{nullptr};
    std::shared_ptr<Tensor> v{nullptr};
    std::shared_ptr<Tensor> att_output{nullptr};
    std::shared_ptr<Tensor> att_proj_o{nullptr};

    std::shared_ptr<Tensor> mlp_gate{nullptr};
    std::shared_ptr<Tensor> mlp_up{nullptr};
    std::shared_ptr<Tensor> swiglu_output{nullptr};
    std::shared_ptr<Tensor> logits{nullptr}; 

};