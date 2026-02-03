#include "cuda_backend.hpp"
#include "cuda_operator.hpp"
#include "cuda_device.cuh"
#include "kernels.cuh"
#include "utils/context.hpp"
#include "utils/constant_table.hpp"

namespace spyinfer {

void CUDAAdd::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("Add operator missing input: " + input_name);
        }
    }
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto a = std::any_cast<std::shared_ptr<Tensor>>(params["input_a"]);
    auto b = std::any_cast<std::shared_ptr<Tensor>>(params["input_b"]);

    if (a->shape() != b->shape())
    {
        throw std::runtime_error("Input tensors must have the same shape");
    }

    if (output != nullptr && output->shape() != a->shape())
    {
        throw std::runtime_error("Output tensor shape must match input tensor shape");
    }
}

void CUDAAdd::compute(std::unordered_map<std::string, std::any>& params)
{
    auto result = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto a = std::any_cast<std::shared_ptr<Tensor>>(params["input_a"]);
    auto b = std::any_cast<std::shared_ptr<Tensor>>(params["input_b"]);

    if (result->dtype() == DataType::fp32_t)
    {
        cuda_add_fp32(result->data_ptr<float>(), a->data_ptr<float>(), b->data_ptr<float>(), result->numel());
    }
    else if (result->dtype() == DataType::fp16_t)
    {
        cuda_add_fp16(result->data_ptr<uint16_t>(), a->data_ptr<uint16_t>(), b->data_ptr<uint16_t>(), result->numel());
    }
}

void CUDALinear::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("Linear operator missing input: " + input_name);
        }
    }
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);
    auto bias = std::any_cast<std::shared_ptr<Tensor>>(params["bias"]);

    auto input_shape = input->shape();   //[x, x, m, k]
    auto weight_shape = weight->shape(); // [1, 1, n, k]

    if (input_shape[3] != weight_shape[3])
    {
        throw std::runtime_error("Input features dimension mismatch between input and weight");
    }

    std::array<int64_t, 4> output_shape = {input_shape[0], input_shape[1], input_shape[2], weight_shape[2]};

    if (bias != nullptr && bias->shape() != output_shape)
    {
        throw std::runtime_error("Bias tensor shape must match linear operator output shape");
    }

    if (output != nullptr && (output->shape() != output_shape || output->dtype() != input->dtype()))
    {
        throw std::runtime_error("Output tensor shape and dtype must match linear operator output shape and dtype");
    }
}

void CUDALinear::compute(std::unordered_map<std::string, std::any>& params)
{
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);
    auto bias = std::any_cast<std::shared_ptr<Tensor>>(params["bias"]);

    int m = static_cast<int>(input->numel() / input->shape()[3]); // 输入序列长度
    int k = static_cast<int>(input->shape()[3]);                  // 输入特征维度
    int n = static_cast<int>(weight->shape()[2]);                 // 输出特征维度

    if (weight->dtype() == DataType::fp16_t)
    {
        if (input->dtype() == DataType::fp32_t)
            cuda_linear_fp32_fp16(output->data_ptr<float>(), input->data_ptr<float>(), weight->data_ptr<uint16_t>(),
                              bias ? bias->data_ptr<uint16_t>() : nullptr, m, k, n);
        else if (input->dtype() == DataType::fp16_t)
            cuda_linear_fp16_fp16(output->data_ptr<uint16_t>(), input->data_ptr<uint16_t>(), weight->data_ptr<uint16_t>(),
                              bias ? bias->data_ptr<uint16_t>() : nullptr, m, k, n);
    }
}

void CUDAEmbedding::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("Embedding operator missing input: " + input_name);
        }
    }
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]); // indices
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);

    auto input_shape = input->shape();
    auto weight_shape = weight->shape();

    std::array<int64_t, 4> expected_output_shape = {1, 1, input->shape().back(), weight->shape().back()};

    if (output != nullptr && (output->shape() != expected_output_shape || output->dtype() != weight->dtype()))
    {
        throw std::runtime_error("Output tensor shape and dtype must match embedding operator output");
    }
}

void CUDAEmbedding::compute(std::unordered_map<std::string, std::any>& params)
{
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]); // indices
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);

    int batch_size = input->numel(); // Total number of tokens
    int hidden_size = weight->shape().back();

    if (weight->dtype() == DataType::fp16_t)
    {
        cuda_embedding_fp16(output->data_ptr<uint16_t>(), input->data_ptr<int>(), weight->data_ptr<uint16_t>(), batch_size, hidden_size);
    }
    else
    {
        cuda_embedding_fp32(output->data_ptr<float>(), input->data_ptr<int>(), weight->data_ptr<float>(), batch_size, hidden_size);
    }
}

void CUDARMSNorm::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("RMSNorm operator missing input: " + input_name);
        }
    }
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);

    float eps = std::any_cast<float>(params["epsilon"]);

    if (eps < 0)
    {
        throw std::runtime_error("Epsilon must be positive for RMSNorm");
    }

    if (input->shape() != weight->shape())
    {
        if (input->shape()[3] != weight->shape()[3])
        {
            throw std::runtime_error("Weight tensor shape must match last dimension of input tensor");
        }
    }

    if (output != nullptr && output->shape() != input->shape())
    {
        throw std::runtime_error("Output tensor shape must match input tensor shape");
    }
}

void CUDARMSNorm::compute(std::unordered_map<std::string, std::any>& params)
{
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);
    float eps = std::any_cast<float>(params["epsilon"]);

    int nums = static_cast<int>(input->shape()[0] * input->shape()[1] * input->shape()[2]); // batch_size * seq_len * num_heads
    int len = static_cast<int>(input->shape()[3]);                                          // hidden_size

    if (input->dtype() == DataType::fp16_t)
    {
        cuda_rms_norm_fp16(output->data_ptr<uint16_t>(), input->data_ptr<uint16_t>(), weight->data_ptr<uint16_t>(), nums, len, eps);
    }
}

void CUDARope::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("Rope operator missing input: " + input_name);
        }
    }
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto position_ids = std::any_cast<std::shared_ptr<Tensor>>(params["position_ids"]);

    if (output != nullptr && output->shape() != input->shape())
    {
        throw std::runtime_error("Output tensor shape must match input tensor shape");
    }
}

void CUDARope::compute(std::unordered_map<std::string, std::any>& params)
{
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto position_ids = std::any_cast<std::shared_ptr<Tensor>>(params["position_ids"]);

    int batch_size = static_cast<int>(input->shape()[1]);
    int num_heads = static_cast<int>(input->shape()[2]);
    int head_size = static_cast<int>(input->shape()[3]); 

    if (input->dtype() == DataType::fp16_t)
    {
        cuda_rope_fp16(output->data_ptr<uint16_t>(), input->data_ptr<uint16_t>(), position_ids->data_ptr<int>(), batch_size, num_heads, head_size,
                       ConstantTable::GetInstance().rope_cos_table_cuda_, ConstantTable::GetInstance().rope_sin_table_cuda_);
    }
}

void CUDASwiGLU::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("SwiGLU operator missing input: " + input_name);
        }
    }
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto gate = std::any_cast<std::shared_ptr<Tensor>>(params["input_gate"]);
    auto up = std::any_cast<std::shared_ptr<Tensor>>(params["input_up"]);

    if (gate->shape() != up->shape())
    {
        throw std::runtime_error("Gate and Up tensors must have the same shape");
    }

    if (output != nullptr && output->shape() != gate->shape())
    {
        throw std::runtime_error("Output tensor shape must match gate/up tensor shape");
    }
}

void CUDASwiGLU::compute(std::unordered_map<std::string, std::any>& params)
{
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto gate = std::any_cast<std::shared_ptr<Tensor>>(params["input_gate"]);
    auto up = std::any_cast<std::shared_ptr<Tensor>>(params["input_up"]);

    int n = static_cast<int>(gate->numel());

    if (gate->dtype() == DataType::fp16_t)
    {
        cuda_swiglu_fp16(output->data_ptr<uint16_t>(), gate->data_ptr<uint16_t>(), up->data_ptr<uint16_t>(), n);
    }
}

void CUDAAttention::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("Attention operator missing input: " + input_name);
        }
    }
}

void CUDAAttention::compute(std::unordered_map<std::string, std::any>& params)
{
    const auto& context = Context::getInstance();
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);

    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto q = std::any_cast<std::shared_ptr<Tensor>>(params["input_q"]);
    auto k = std::any_cast<std::shared_ptr<Tensor>>(params["input_k"]);
    auto v = std::any_cast<std::shared_ptr<Tensor>>(params["input_v"]);
    auto positions = std::any_cast<std::shared_ptr<Tensor>>(params["position_ids"]);
    auto layer = std::any_cast<int>(params["layer_ids"]);

    auto k_cache = context.k_cache;
    auto v_cache = context.v_cache;
    auto block_tables_ptr = context.block_tables_device; //[seq_id][logic_block_id] --> physic_block_id （历史kvcache位置）
    auto context_lens_ptr = context.context_lens_device; // seq id -> seq len
    auto slot_mapping_ptr = context.slot_mapping_device; // token id -> physic offset (当前token的kv需要插入的kv cahche位置)
    auto seq_idx_map_ptr = context.seq_idx_map_device;   // token id -> seq id
    auto cu_seqlens = context.cu_seqlens_device;


    auto batch_size = context.batch_size;
    auto kvcache_block_size = context.k_cache->shape()[2];

    uint16_t* k_cache_ptr = k_cache->data_ptr<uint16_t>() + layer * k_cache->strides()[0];
    uint16_t* v_cache_ptr = v_cache->data_ptr<uint16_t>() + layer * v_cache->strides()[0];

   
    int max_seq_len = context.max_seq_len;
    int num_seqs = context.batch_size;

    const int num_query_tokens = q->shape()[1];
    const int num_q_heads = q->shape()[2];
    const int head_dim = q->shape()[3];
    const int num_kv_heads = k->shape()[2];

    cuda_store_kvcache_fp16(k->data_ptr<uint16_t>(), v->data_ptr<uint16_t>(), k_cache_ptr, v_cache_ptr,
                            slot_mapping_ptr, num_query_tokens, num_kv_heads, head_dim);

    if (q->dtype() == DataType::fp16_t)
    {
        if (context.is_prefill)
        {
            cuda_flash_attention_prefill_fp16(output->data_ptr<uint16_t>(), q->data_ptr<uint16_t>(), k->data_ptr<uint16_t>(), v->data_ptr<uint16_t>(),
                                              cu_seqlens, num_seqs, max_seq_len, num_q_heads, num_kv_heads, head_dim);
        }
        else
        {
            cuda_paged_attention_decode_fp16(output->data_ptr<uint16_t>(), q->data_ptr<uint16_t>(), k_cache_ptr,
                                             v_cache_ptr, block_tables_ptr, context_lens_ptr, batch_size, num_q_heads, num_kv_heads,
                                             head_dim, kvcache_block_size, context.max_blocks_per_seq);
        }
    }
}

void CUDABF16ToFP16::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("BF16ToFP16 operator missing input: " + input_name);
        }
    }
}

void CUDABF16ToFP16::compute(std::unordered_map<std::string, std::any>& params)
{
    auto input = std::any_cast<uint16_t*>(params["input"]);
    size_t n = std::any_cast<size_t>(params["size"]);
    cuda_bf16_to_fp16_inplace(input, n);
}

} // namespace spyinfer