#include "cpu_operator.hpp"
#include "core/tensor.hpp"
#include "cpu_backend.hpp"
#include "x64/kernels.hpp"
#include "utils/constant_table.hpp"

#include <memory>
#include <iostream>

namespace spyinfer {

void Add::forward_expand(std::unordered_map<std::string, std::any>& params)
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

    is_parallelizable_ = a->numel() > 100000;
}

void Add::compute(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto a = std::any_cast<std::shared_ptr<Tensor>>(params["input_a"]);
    auto b = std::any_cast<std::shared_ptr<Tensor>>(params["input_b"]);

    auto cpu_backend = dynamic_cast<CPUBackend*>(backend.get());

    if (output->dtype() == DataType::fp32_t)
    {
        int execute_threads = 1;
        if (is_parallelizable())
        {
            execute_threads = cpu_backend->get_thread_num();
        }
        size_t chunk_size = (a->numel() + execute_threads - 1) / execute_threads;

        auto add_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
            size_t start = thread_idx * chunk_size;
            size_t end = std::min(start + chunk_size, a->numel());
            if (start >= end) return;

            x64_float32_add(output->data_ptr<float>() + start, a->data_ptr<float>() + start, b->data_ptr<float>() + start, end - start);
        };
        cpu_backend->get_parallel_executor()->execute(add_task);
    }
}

void Linear::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    if (op_type != OperatorType::Linear)
    {
        throw std::invalid_argument("Linear operator type mismatch");
    }
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

    // 检查输入张量的形状
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

    if (output == nullptr)
    {
        output = Tensor::create(backend->get_devices()[0], output_shape, input->dtype());
    }

    size_t m = output->numel() / output->shape().back();
    size_t n = output->shape().back();
    size_t k = weight->shape().back();

    is_parallelizable_ = m * n * k > 1e5;
}

void Linear::compute(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);
    auto bias = std::any_cast<std::shared_ptr<Tensor>>(params["bias"]);

    auto cpu_backend = dynamic_cast<CPUBackend*>(backend.get());

    if (output->dtype() == DataType::fp32_t)
    {
        int execute_threads = is_parallelizable() ? cpu_backend->get_thread_num() : 1;

        size_t m = output->numel() / output->shape().back();
        size_t n = output->shape().back();
        size_t k = weight->shape().back();

        size_t rows_per_thread = (m + execute_threads - 1) / execute_threads;
        auto row_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
            size_t row_start = thread_idx * rows_per_thread;
            size_t row_end = std::min(row_start + rows_per_thread, m);

            if (row_start >= row_end) return;

            for (size_t i = row_start; i < row_end; ++i)
            {
                for (size_t j = 0; j < n; j++)
                {
                    if (weight->dtype() == DataType::fp32_t)
                    {
                        output->data_ptr<float>()[i * n + j] =
                            x64_dot_product_fp32(input->data_ptr<float>() + i * k, weight->data_ptr<float>() + j * k, k) +
                            (bias ? bias->data_ptr<float>()[i * n + j] : 0.f);
                    }
                    else if (weight->dtype() == DataType::bf16_t)
                    {
                        output->data_ptr<float>()[i * n + j] =
                            x64_dot_product_fp32_bf16(input->data_ptr<float>() + i * k, weight->data_ptr<uint16_t>() + j * k, k) +
                            (bias ? bias->data_ptr<float>()[i * n + j] : 0.f);
                    }
                }
            }
        };

        size_t cols_per_thread = (n + execute_threads - 1) / execute_threads;
        auto col_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
            size_t col_start = thread_idx * cols_per_thread;
            size_t col_end = std::min(col_start + cols_per_thread, n);
            if (col_start >= col_end) return;
            for (size_t j = col_start; j < col_end; ++j)
            {
                for (size_t i = 0; i < m; ++i)
                {
                    if (weight->dtype() == DataType::fp32_t)
                    {
                        output->data_ptr<float>()[i * n + j] =
                            x64_dot_product_fp32(input->data_ptr<float>() + i * k, weight->data_ptr<float>() + j * k, k) +
                            (bias ? bias->data_ptr<float>()[i * n + j] : 0.f);
                    }
                    else if (weight->dtype() == DataType::bf16_t)
                    {
                        output->data_ptr<float>()[i * n + j] =
                            x64_dot_product_fp32_bf16(input->data_ptr<float>() + i * k, weight->data_ptr<uint16_t>() + j * k, k) +
                            (bias ? bias->data_ptr<float>()[i * n + j] : 0.f);
                    }
                }
            }
        };

        if (execute_threads <= m)
            cpu_backend->get_parallel_executor()->execute(row_task); // m大于线程数，行并行
        else
            cpu_backend->get_parallel_executor()->execute(col_task);
    }
}

void Embedding::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);

    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("Embedding operator missing input: " + input_name);
        }
    }
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);

    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);

    if (input->dtype() != DataType::int32_t)
    {
        throw std::runtime_error("Input tensor dtype must be int");
    }

    auto output_shape = std::array<int64_t, 4>{1, 1, input->shape().back(), weight->shape().back()};

    if (output != nullptr && (output->shape() != output_shape || output->dtype() != DataType::fp32_t))
    {
        throw std::runtime_error("Output tensor shape must match embedding operator output shape");
    }

    if (output == nullptr)
    {
        output = backend->create_tensor(output_shape, DataType::fp32_t);
    }

    is_parallelizable_ = true;
}
void Embedding::compute(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);

    const int hidden_size = weight->shape().back();
    const int execute_threads = is_parallelizable() ? dynamic_cast<CPUBackend*>(backend.get())->get_thread_num() : 1;
    const int batch_size = input->numel();
    const int batch_per_thread = (batch_size + execute_threads - 1) / execute_threads;

    auto embed_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
        int start = thread_idx * batch_per_thread;
        int end = std::min(start + batch_per_thread, batch_size);
        if (start >= end) return;
        for (size_t i = start; i < end; ++i)
        {
            int offset = input->data_ptr<int>()[i] * hidden_size;
            if (weight->dtype() == DataType::fp32_t)
            {
                std::copy_n(weight->data_ptr<float>() + offset, hidden_size, output->data_ptr<float>() + i * hidden_size);
            }
            else if (weight->dtype() == DataType::bf16_t)
            {
                for (int j = 0; j < hidden_size; ++j)
                {
                    output->data_ptr<float>()[i * hidden_size + j] =
                        ConstantTable::GetInstance().bf16_to_fp32_table_[weight->data_ptr<uint16_t>()[offset + j]];
                }
            }
        }
    };

    dynamic_cast<CPUBackend*>(backend.get())->get_parallel_executor()->execute(embed_task);
}

void RMSNorm::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);

    if (op_type != OperatorType::RMSNorm)
    {
        throw std::invalid_argument("RMSNorm operator type mismatch");
    }
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

    if (input->shape().back() != weight->shape().back())
    {
        throw std::runtime_error("RMSNorm operator input and weight shape mismatch");
    }

    if (output != nullptr && output->shape() != input->shape())
    {
        throw std::runtime_error("Output tensor shape must match RMSNorm operator output shape");
    }

    if (output == nullptr)
    {
        output = Tensor::create(backend->get_devices()[0], input->shape(), input->dtype());
    }

    int batch_size = input->shape()[0] * input->shape()[1] * input->shape()[2];

    is_parallelizable_ = batch_size >= dynamic_cast<CPUBackend*>(backend.get())->get_thread_num();
}

void RMSNorm::compute(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto weight = std::any_cast<std::shared_ptr<Tensor>>(params["weight"]);
    auto epsilon = std::any_cast<float>(params["epsilon"]);

    const int execute_threads = is_parallelizable() ? dynamic_cast<CPUBackend*>(backend.get())->get_thread_num() : 1;

    if (input->dtype() == DataType::fp32_t)
    {
        const int hidden_size = input->shape().back();
        const size_t batch_size = input->shape()[0] * input->shape()[1] * input->shape()[2];
        const size_t batch_per_thread = (batch_size + execute_threads - 1) / execute_threads;

        auto rmsnorm_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
            size_t start = thread_idx * batch_per_thread;
            size_t end = std::min(start + batch_per_thread, batch_size);
            if (start >= end) return;
            for (int i = start; i < end; ++i)
            {
                if (weight->dtype() == DataType::bf16_t)
                {
                    x64_rms_norm_fp32_bf16(output->data_ptr<float>() + i * hidden_size, input->data_ptr<float>() + i * hidden_size,
                                           weight->data_ptr<uint16_t>(), hidden_size, epsilon);
                }
            }
        };

        dynamic_cast<CPUBackend*>(backend.get())->get_parallel_executor()->execute(rmsnorm_task);
    }
}

void Rope::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
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

    if (output != nullptr && (output->shape() != input->shape() || output->dtype() != input->dtype()))
    {
        throw std::runtime_error("Output tensor shape and dtype must match input tensor shape and dtype");
    }

    if (output == nullptr)
    {
        output = Tensor::create(backend->get_devices()[0], input->shape(), input->dtype());
    }
    is_parallelizable_ = input->numel() > 100000;
}
void Rope::compute(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);
    auto position_ids = std::any_cast<std::shared_ptr<Tensor>>(params["position_ids"])->data_ptr<int>();

    const int execute_threads = is_parallelizable() ? dynamic_cast<CPUBackend*>(backend.get())->get_thread_num() : 1;
    const int num_heads = input->shape()[2];
    const int batch_size = input->shape()[0] * input->shape()[1] * input->shape()[2];
    const int hidden_size = input->shape().back();
    const int rotation_size = hidden_size / 2;
    const int batch_per_thread = (batch_size + execute_threads - 1) / execute_threads;

    auto& rope_cos_table = ConstantTable::GetInstance().rope_cos_table_;
    auto& rope_sin_table = ConstantTable::GetInstance().rope_sin_table_;

    auto rope_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
        int start = thread_idx * batch_per_thread;
        int end = std::min(start + batch_per_thread, batch_size);
        if (start >= end) return;
        for (int i = start; i < end; ++i)
        {
            if (output->dtype() == DataType::fp32_t)
            {
                const float* input_ptr = input->data_ptr<float>() + i * hidden_size;
                const int pos = position_ids[i / num_heads];
                float* out_ptr = output->data_ptr<float>() + i * hidden_size;

                // rotation each head_dim
                for (int j = 0; j < rotation_size; ++j)
                {
                    float v0 = input_ptr[j];
                    float v1 = input_ptr[j + rotation_size];
                    out_ptr[j] = v0 * rope_cos_table[pos * rotation_size + j] - v1 * rope_sin_table[pos * rotation_size + j];
                    out_ptr[j + rotation_size] = v0 * rope_sin_table[pos * rotation_size + j] + v1 * rope_cos_table[pos * rotation_size + j];
                }
            }
        }
    };

    dynamic_cast<CPUBackend*>(backend.get())->get_parallel_executor()->execute(rope_task);
}

void SwiGLU::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("SwiGLU operator missing input: " + input_name);
        }
    }

    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input_gate = std::any_cast<std::shared_ptr<Tensor>>(params["input_gate"]);
    auto input_up = std::any_cast<std::shared_ptr<Tensor>>(params["input_up"]);

    // Verify dimensions compatibility
    if (input_gate->shape() != input_up->shape())
    {
        throw std::runtime_error("SwiGLU operator input_gate and input_up shape mismatch");
    }

    // Expected output shape after transformation
    auto output_shape = input_gate->shape();

    if (output != nullptr && (output->shape() != output_shape || output->dtype() != input_gate->dtype()))
    {
        throw std::runtime_error("Output tensor shape and dtype must match expected SwiGLU output shape and dtype");
    }

    if (output == nullptr)
    {
        output = Tensor::create(backend->get_devices()[0], output_shape, input_gate->dtype());
    }

    // Determine if operation is parallelizable based on size
    is_parallelizable_ = input_gate->numel() > 100000;
}

void SwiGLU::compute(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input_gate = std::any_cast<std::shared_ptr<Tensor>>(params["input_gate"]);
    auto input_up = std::any_cast<std::shared_ptr<Tensor>>(params["input_up"]);

    auto cpu_backend = dynamic_cast<CPUBackend*>(backend.get());

    const int execute_threads = is_parallelizable() ? cpu_backend->get_thread_num() : 1;
    const size_t batch_size = input_gate->numel();
    const size_t batch_per_thread = (batch_size + execute_threads - 1) / execute_threads;

    auto swiglu_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
        size_t start = thread_idx * batch_per_thread;
        size_t end = std::min(start + batch_per_thread, batch_size);
        if (start >= end) return;

        if (output->dtype() == DataType::fp32_t)
        {
            float* gate_ptr = input_gate->data_ptr<float>() + start;
            float* up_ptr = input_up->data_ptr<float>() + start;
            float* out_ptr = output->data_ptr<float>() + start;

            x64_swiglu_fp32(out_ptr, gate_ptr, up_ptr, end - start);
        }
    };

    cpu_backend->get_parallel_executor()->execute(swiglu_task);
}

void Sigmoid::forward_expand(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto op_type = std::any_cast<OperatorType>(params["op_type"]);
    for (const auto& input_name : operator_input_names[op_type])
    {
        if (params.find(input_name) == params.end())
        {
            throw std::invalid_argument("Sigmoid operator missing input: " + input_name);
        }
    }
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);

    if (output != nullptr && (output->shape() != input->shape() || output->dtype() != input->dtype()))
    {
        throw std::runtime_error("Output tensor shape and dtype must match input tensor shape and dtype");
    }

    if (output == nullptr)
    {
        output = Tensor::create(backend->get_devices()[0], input->shape(), input->dtype());
    }

    is_parallelizable_ = (input->numel() / input->shape().back()) > 1;
}

void Sigmoid::compute(std::unordered_map<std::string, std::any>& params)
{
    auto backend = std::any_cast<std::shared_ptr<BaseBackend>>(params["backend"]);
    auto output = std::any_cast<std::shared_ptr<Tensor>>(params["output"]);
    auto input = std::any_cast<std::shared_ptr<Tensor>>(params["input"]);

    const int execute_threads = is_parallelizable() ? dynamic_cast<CPUBackend*>(backend.get())->get_thread_num() : 1;
    const size_t batch_size = input->numel() / input->shape().back();
    const size_t batch_per_thread = (batch_size + execute_threads - 1) / execute_threads;

    auto sigmoid_task = [&](size_t thread_idx, std::vector<std::byte>& thread_buffer) {
        size_t start = thread_idx * batch_per_thread;
        size_t end = std::min(start + batch_per_thread, batch_size);
        if (start >= end) return;

        for (size_t i = start; i < end; ++i)
        {
            x64_sigmoid_fp32(output->data_ptr<float>() + i * input->shape().back(), input->data_ptr<float>() + i * input->shape().back(),
                             input->shape().back());
        }
    };

    dynamic_cast<CPUBackend*>(backend.get())->get_parallel_executor()->execute(sigmoid_task);
}

} // namespace spyinfer
