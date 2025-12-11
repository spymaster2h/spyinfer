#include <iostream>
#include <string_view>
#include <span>
#include <string>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <array>
#include <unordered_map>

#include "define.hpp"
#include "op_cpp.hpp"

// 函数定义
std::string _decoded_token_key(const std::string_view key);


template <typename T, typename U>
void copy_with_transform(const std::byte *src, size_t src_elem_count, std::byte *dst, U (*transform)(T))
{
    auto src_ptr = reinterpret_cast<const T *>(src);
    auto dst_ptr = reinterpret_cast<U *>(dst);
    for (size_t i = 0; i < src_elem_count; ++i)
    {
        dst_ptr[i] = transform(src_ptr[i]); // 直接写入目标内存地址
    }
}

struct Alloc
{
    std::size_t total_allocated = 0;
    std::unordered_map<std::string, std::size_t> alloc_records;
    std::vector<std::span<std::byte>> arenas;

    std::span<std::byte> alloc(std::size_t size, std::string name = "default")
    {
        auto span = std::span<std::byte>(new std::byte[size](), size);
        arenas.emplace_back(span);
        total_allocated += size;
        alloc_records[name] += size;
        return span;
    }

    ~Alloc()
    {
        for (auto &[name, size] : alloc_records)
        {
            std::cout << "Allocator " << name << " allocated " << size << " bytes." << std::endl;
        }

        for (auto &arena : arenas)
        {
            delete[] arena.data();
        }
    }
};

Alloc allocator;

struct Tensor
{
    std::array<int32_t, 4> shape{};
    std::span<std::byte> data;

    Tensor() {}
    Tensor(std::span<std::byte> data, std::array<int32_t, 4> shape) : shape(shape), data(data) {}

    Tensor(std::span<std::byte> data, int32_t shape0, int32_t shape1 = 1, int32_t shape2 = 1, int32_t shape3 = 1)
        : shape({shape0, shape1, shape2, shape3}), data(data)
    {
    }

    template <typename T>
    T *as()
    {
        return reinterpret_cast<T *>(data.data());
    }

    template <typename T>
    const T *as() const
    {
        return reinterpret_cast<const T *>(data.data());
    }
};

struct Block
{
    Tensor attn_q, attn_k, attn_v, attn_o;
    Tensor attn_q_bias, attn_k_bias, attn_v_bias;
    Tensor attn_q_norm, attn_k_norm;
    Tensor mlp_down, mlp_gate, mlp_up;
    Tensor input_norm, post_norm;
};

struct ModelWeights
{
    std::vector<Block> blocks;
    Tensor embed;
    Tensor norm;
    Tensor lm_head;
};

class Timer
{
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    void reset() { start_time = std::chrono::high_resolution_clock::now(); }

    double elapsed_seconds() const
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }
    double elapsed_ms() const
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

struct StaticDATrie
{
    struct Node
    {
        int32_t token_id = -1;
        std::array<int32_t, 256> children; //子节点的索引

        Node() { children.fill(-1); }
    };
    std::vector<Node> meta;

    StaticDATrie()
    {
        //新建root节点，默认0节点为root
        Node root;
        meta.push_back(root);
    }

    void insert(const std::string &token, const int32_t &token_id)
    {
        std::uint32_t cur_num = 0;

        for (std::size_t i = 0; i < token.size(); ++i)
        {
            const std::uint8_t c = token[i];
            if (meta[cur_num].children[c] == -1)
            {
                Node node;
                meta.push_back(node);
                meta[cur_num].children[c] = meta.size() - 1;
            }
            cur_num = meta[cur_num].children[c];
        }
        meta[cur_num].token_id = token_id;
    }

    std::pair<int32_t, std::size_t> longest_match(const std::string_view data) const
    {
        std::uint32_t cur_num = 0;
        std::size_t i = 0;
        for (; i < data.size(); ++i)
        {
            const std::uint8_t c = data[i];
            if (meta[cur_num].children[c] == -1)
            {
                break;
            }
            cur_num = meta[cur_num].children[c];
        }

        return {meta[cur_num].token_id, i};
    }
};

struct ModelConfig
{
    std::string model_type;
    std::string hidden_act;
    int32_t hidden_size; // 隐藏层向量维度
    int32_t intermediate_size;
    int32_t num_key_value_heads;
    int32_t num_hidden_layers;
    int32_t num_attention_heads;
    bool tie_word_embeddings;
    float rope_theta;
    float rms_norm_eps;
    int32_t head_dim;
    int32_t bos_token_id;
    int32_t eos_token_id;
    int32_t max_position_embeddings;
    int32_t vocab_size;

    bool do_sample;
    float temperature;
    float top_p;
    float top_k;
};

void test();

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <prompt>" << std::endl;
        return 1;
    }

    test();

    std::string model_path = argv[1];
    std::string prompt = argv[2];

    //读取模型config
    const std::filesystem::path config_path = std::filesystem::path(model_path) / "config.json";
    std::ifstream config_file(config_path);
    nlohmann::json config_json;
    config_file >> config_json;
    ModelConfig config;
    config.model_type = config_json.at("model_type").get<std::string>();
    config.hidden_act = config_json.at("hidden_act").get<std::string>();
    config.hidden_size = config_json.value("hidden_size", -1);
    config.intermediate_size = config_json.value("intermediate_size", -1);
    config.num_key_value_heads = config_json.value("num_key_value_heads", -1);
    config.num_hidden_layers = config_json.value("num_hidden_layers", -1);
    config.num_attention_heads = config_json.value("num_attention_heads", -1);
    config.tie_word_embeddings = config_json.value("tie_word_embeddings", false);
    config.rope_theta = config_json.value("rope_theta", 1000000.0f);
    config.rms_norm_eps = config_json.value("rms_norm_eps", 1e-6f);
    config.head_dim = config_json.value("head_dim", config.hidden_size / config.num_attention_heads);
    config.bos_token_id = config_json.value("bos_token_id", -1);
    config.eos_token_id = config_json.value("eos_token_id", -1);
    config.max_position_embeddings = config_json.value("max_position_embeddings", 16384);
    config.vocab_size = config_json.value("vocab_size", -1);

    //读取模型generation_config
    const std::filesystem::path generation_config_path = std::filesystem::path(model_path) / "generation_config.json";
    std::ifstream gen_file(generation_config_path);
    nlohmann::json gen_json;
    gen_file >> gen_json;

    config.do_sample = gen_json.value("do_sample", false);
    config.temperature = gen_json.value("temperature", 1.0f);
    config.top_p = gen_json.value("top_p", 1.0f);
    config.top_k = gen_json.value("top_k", 0);

    Timer tokenizer_load_timer;

    //加载Tokenizer_config
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    nlohmann::json tokenizer_config_json;
    std::ifstream tokenizer_config_file(std::filesystem::path(model_path) / "tokenizer_config.json");
    tokenizer_config_file >> tokenizer_config_json;
    //控制是否在序列开头添加 <bos>，标记序列的开始
    if (tokenizer_config_json.value("add_bos_token", false))
    {
        bos_token_id = config.bos_token_id;
    }
    //控制是否在序列末尾添加 <eos>，标记序列的结束
    if (tokenizer_config_json.value("add_eos_token", false))
    {
        eos_token_id = config.eos_token_id;
    }

    /*加载词表*/
    nlohmann::json tokenizer_json;
    std::ifstream tokenizer_file(std::filesystem::path(model_path) / "tokenizer.json");
    tokenizer_file >> tokenizer_json;
    const auto &vocab_json = tokenizer_json["model"]["vocab"];
    const auto &added_tokens = tokenizer_json["added_tokens"];
    std::vector<std::string> vocab;
    StaticDATrie trie;
    vocab.resize(vocab_json.size() + added_tokens.size() + 100);
    /*
        byte_fallback 是 model 下的一个布尔值（true 或 false），用于控制分词器遇到未在 vocab 中出现的字符时的处理策略：
        true：启用字节级回退。当字符不在 vocab 中时，将其拆分为 UTF-8 字节序列，每个字节作为独立 token（这些字节 token 也在 vocab 中）。
        false：禁用字节级回退。此时未识别的字符会被替换为 unk_token（如 <unk>）
    */
    bool byte_fallback = tokenizer_json["model"].value("byte_fallback", false);
    for (const auto &[key, value] : vocab_json.items())
    {
        auto token_id = value.get<int32_t>();
        // vocab中存储的内容（中文字符的utf-8编码, 然后按照字节表映射unicode字符）
        //_decoded_token_key 完成的是 逐字节的unicode字符-> utf-8 字节的逆映射
        const auto decoded = _decoded_token_key(key);
        trie.insert(decoded, token_id);
        vocab[token_id] = decoded;
    }

    for (const auto &data : added_tokens)
    {
        auto key = data["content"].get<std::string>();
        auto token_id = data["id"].get<int32_t>();
        const auto decoded = _decoded_token_key(key);
        trie.insert(decoded, token_id);
        vocab[token_id] = decoded;
    }
    std::cout << "[DEBUG] Tokenizer loaded in " << tokenizer_load_timer.elapsed_ms() << "ms" << std::endl;

    /*加载模型*/
    Timer model_load_timer;
    std::filesystem::path weight_path = std::filesystem::path(model_path) / "model.safetensors";
    std::ifstream weight_file(weight_path, std::ios::binary);
    std::size_t file_size = std::filesystem::file_size(weight_path);
    std::uint64_t metadata_size = 0;
    //读取前八字节，获取metadata_size
    weight_file.read(reinterpret_cast<char *>(&metadata_size), sizeof(metadata_size));
    std::vector<char> meta_data(metadata_size);
    weight_file.read(meta_data.data(), meta_data.size());

    //读取metadata json
    const auto metadata_json = nlohmann::json::parse(meta_data);

    //读取weight
    std::size_t raw_data = file_size - metadata_size - 8;
    std::span<std::byte> data = allocator.alloc(raw_data, "model_weights");
    weight_file.seekg(metadata_size + 8);
    weight_file.read(reinterpret_cast<char *>(data.data()), data.size());
    if (weight_file.eof() || weight_file.fail())
    {
        throw std::runtime_error("Failed to read metadata from " + weight_path.string());
    }

    auto get_tensor = [&](const std::string_view name) {
        const auto &meta = metadata_json.at(name);
        std::array<int32_t, 4> shape{1, 1, 1, 1};
        auto shape_tensor = meta.at("shape").get<std::vector<int32_t>>();
        if (shape_tensor.size() > 4 || shape_tensor.size() < 1)
        {
            throw std::runtime_error("Invalid tensor shape for " + std::string(name));
        }
        else
        {
            std::copy(shape_tensor.rbegin(), shape_tensor.rend(), shape.begin());
        }
        auto start = meta.at("data_offsets").at(0).get<std::int64_t>();
        auto end = meta.at("data_offsets").at(1).get<std::int64_t>();

        return Tensor(data.subspan(start, end - start), shape);
    };

    ModelWeights weights;
    weights.embed = get_tensor("model.embed_tokens.weight");
    weights.norm = get_tensor("model.norm.weight");
    weights.blocks.resize(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i)
    {
        auto &block = weights.blocks[i];
        block.attn_q = get_tensor("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight");
        block.attn_k = get_tensor("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight");
        block.attn_v = get_tensor("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight");
        block.attn_o = get_tensor("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight");
        block.mlp_down = get_tensor("model.layers." + std::to_string(i) + ".mlp.down_proj.weight");
        block.mlp_gate = get_tensor("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight");
        block.mlp_up = get_tensor("model.layers." + std::to_string(i) + ".mlp.up_proj.weight");
        block.input_norm = get_tensor("model.layers." + std::to_string(i) + ".input_layernorm.weight");
        block.post_norm = get_tensor("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight");

        block.attn_q_norm = get_tensor("model.layers." + std::to_string(i) + ".self_attn.q_norm.weight");
        block.attn_k_norm = get_tensor("model.layers." + std::to_string(i) + ".self_attn.k_norm.weight");
    }

    if (config.tie_word_embeddings)
    {
        weights.lm_head = weights.embed;
    }
    else
    {
        weights.lm_head = get_tensor("lm_head.weight");
    }
    std::cout << "[DEBUG] model weight loaded in " << model_load_timer.elapsed_ms() << "ms" << std::endl;

    /*推理*/

    // 字符串-> token_id
    auto encode = [&](const std::string_view text) -> std::vector<int32_t> {
        std::vector<int32_t> tokens;
        if (bos_token_id >= 0)
        {
            tokens.push_back(bos_token_id);
        }
        std::size_t i = 0;
        while (i < text.size())
        {
            //最长匹配
            const std::string_view sub = {text.begin() + i, text.end()};
            auto [token_id, length] = trie.longest_match(sub);
            if (token_id >= 0)
            {
                tokens.push_back(token_id);
                i += length;
            }
            else
            {
                tokens.push_back(-1);
                i += 1;
            }
        }
        if (eos_token_id >= 0)
        {
            tokens.push_back(eos_token_id);
        }
        return std::move(tokens);
    };
    prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant";
    const auto tokens = encode(prompt);

    uint32_t kv_size = 4096;

    // prefill
    Tensor x(allocator.alloc(config.hidden_size * sizeof(float), "step1 embedding vector"), config.hidden_size);
    Tensor xb(allocator.alloc(config.hidden_size * sizeof(float), "step2 input RMS Norm vector"), config.hidden_size);
    Tensor q(allocator.alloc(config.head_dim * config.num_attention_heads * sizeof(float), "step3 attention_q"), config.num_attention_heads,
             config.head_dim);
    Tensor k(allocator.alloc(config.head_dim * config.num_key_value_heads * sizeof(float), "step3 attention_k"), config.num_key_value_heads,
             config.head_dim);
    Tensor v(allocator.alloc(config.head_dim * config.num_key_value_heads * sizeof(float), "step3 attention_v"), config.num_key_value_heads,
             config.head_dim);

    //[layer, heads, kv_size, per head_dim]
    Tensor kc(
        allocator.alloc(config.num_hidden_layers * config.head_dim * kv_size * config.num_key_value_heads * sizeof(float), "step4 attention_k cache"),
        config.num_hidden_layers, config.num_key_value_heads, kv_size, config.head_dim);
    Tensor vc(
        allocator.alloc(config.num_hidden_layers * config.head_dim * kv_size * config.num_key_value_heads * sizeof(float), "step4 attention_v cache"),
        config.num_hidden_layers, config.num_key_value_heads, kv_size, config.head_dim);

    Tensor xb2(allocator.alloc(config.head_dim * config.num_attention_heads * sizeof(float), "step5 attention_result"), config.num_attention_heads,
               config.head_dim);

    Tensor xb3(allocator.alloc(config.hidden_size * sizeof(float), "step5-1 attention output(hidden_size)"), config.hidden_size);

    Tensor xb4(allocator.alloc(config.hidden_size * sizeof(float), "step6 attention output(Residual connection + post norm)"), config.hidden_size);
    Tensor hb(allocator.alloc(config.intermediate_size * sizeof(float), "step7 mlp x_gate"), config.intermediate_size);
    Tensor hb2(allocator.alloc(config.intermediate_size * sizeof(float), "step7 mlp x_up"), config.intermediate_size);
    Tensor hb3(allocator.alloc(config.intermediate_size * sizeof(float), "step7 mlp result"), config.intermediate_size);
    Tensor xb5(allocator.alloc(config.hidden_size * sizeof(float), "step8 mlp output(down)"), config.hidden_size);

    Tensor logits(allocator.alloc(config.vocab_size * sizeof(float), "logits"), config.vocab_size);
    const auto forward = [&](int32_t token, int32_t pos) {
        // 1. embedding
        copy_with_transform<bf16_t, float>(weights.embed.data.data() + 2 * token * config.hidden_size, config.hidden_size, x.data.data(), _bf16_to_fp32);

        // 2. Forward through each block
        for (std::int32_t i = 0; i < config.num_hidden_layers; ++i)
        {
            const Block &block = weights.blocks[i];

            // 2.1 rms norm
            spyinfer::_rms_norm_fp32<bf16_t>(xb.as<float>(), x.as<float>(), block.input_norm.as<bf16_t>(), config.hidden_size, config.rms_norm_eps);

            // 2.2 compute qkv (GQA)

            const std::int32_t head_dim = config.head_dim;                     // 128
            const std::int32_t q_dim = config.num_attention_heads * head_dim;  // 16 * 128
            const std::int32_t kv_dim = config.num_key_value_heads * head_dim; // 8 * 128

            const bool has_qkv_bias = !block.attn_k_bias.data.empty();
            const bool has_qk_norm = !block.attn_q_norm.data.empty() && !block.attn_k_norm.data.empty();

            /*
              Wq = (16 * 128) * 1024
              wk =  (8 * 128) * 1024
              wv =  (8 * 128) * 1024

             1Q 对应 2kv, 其中Q输出维度 16 * 128 != 1024(隐藏层维度)， 所以需要输出映射W0 = 128 * 16 * 1024
            */
            spyinfer::_gemv(q.as<float>(), block.attn_q.as<bf16_t>(), xb.as<float>(), q_dim, config.hidden_size);
            spyinfer::_gemv(k.as<float>(), block.attn_k.as<bf16_t>(), xb.as<float>(), kv_dim, config.hidden_size);
            spyinfer::_gemv(v.as<float>(), block.attn_v.as<bf16_t>(), xb.as<float>(), kv_dim, config.hidden_size);

            if (has_qk_norm)
            {
                //对于每一个head 进行q k的norm
                for (std::size_t i = 0; i < config.num_attention_heads; i++) // 16 个 q
                {
                    float *qn = q.as<float>() + i * head_dim;
                    spyinfer::_rms_norm_fp32(qn, qn, block.attn_q_norm.as<bf16_t>(), head_dim, config.rms_norm_eps);
                }

                for (std::size_t i = 0; i < config.num_key_value_heads; i++) // 8 个 k
                {
                    float *kn = k.as<float>() + i * head_dim;
                    spyinfer::_rms_norm_fp32(kn, kn, block.attn_k_norm.as<bf16_t>(), head_dim, config.rms_norm_eps);
                }
            }

            // 2.3 qk rope
            for (std::size_t i = 0; i < config.num_attention_heads; i++) // 16 个 q
            {
                float *qr = q.as<float>() + i * head_dim;
                spyinfer::rope_inplace_fp32(qr, config.head_dim, pos, config.rope_theta);
            }

            for (std::size_t i = 0; i < config.num_key_value_heads; i++) // 8 个 k
            {
                float *kr = k.as<float>() + i * head_dim;
                spyinfer::rope_inplace_fp32(kr, config.head_dim, pos, config.rope_theta);
            }

            // 2.4 kv cache
            std::size_t kv_len = pos + 1;
            for (std::size_t kvh = 0; kvh < config.num_key_value_heads; kvh++)
            {
                const std::size_t kv_offest = (i * config.num_key_value_heads * kv_size + kvh * kv_size + pos) * head_dim;

                std::copy_n(k.as<float>() + kvh * head_dim, head_dim, kc.as<float>() + kv_offest);
                std::copy_n(v.as<float>() + kvh * head_dim, head_dim, vc.as<float>() + kv_offest);
            }

            // 2.5 attention
            spyinfer::_mh_attention_fp32(xb2.as<float>(), q.as<float>(), kc.as<float>() + i * config.num_key_value_heads * kv_size * head_dim, vc.as<float>() + i * config.num_key_value_heads * kv_size * head_dim, config.num_attention_heads, config.head_dim,
                                         config.num_key_value_heads, kv_size, kv_len);

            // xb2和q的维度一样 [num_heads, head_dim], 因为与hidden_size不一样,需要wo映射
            spyinfer::_gemv(xb3.as<float>(), block.attn_o.as<bf16_t>(), xb2.as<float>(), config.hidden_size,
                            config.num_attention_heads * config.head_dim);

            // 2.6 残差连接
            for (std::size_t i = 0; i < config.hidden_size; i++)
            {
                x.as<float>()[i] += xb3.as<float>()[i];
            }

            // 2.7 rms_norm on output
            spyinfer::_rms_norm_fp32(xb4.as<float>(), x.as<float>(), block.post_norm.as<bf16_t>(), config.hidden_size, config.rms_norm_eps);

            // 2.8 MLP

            //2.8.1 升维
            spyinfer::_gemv(hb.as<float>(), block.mlp_gate.as<bf16_t>(), xb4.as<float>(), config.intermediate_size, config.hidden_size);
            spyinfer::_gemv(hb2.as<float>(), block.mlp_up.as<bf16_t>(), xb4.as<float>(), config.intermediate_size, config.hidden_size);

            //2.8.2 门控特征
            for (std::size_t i = 0; i < config.intermediate_size; i++)
            {
                float* x_ga = hb.as<float>();
                float* x_up = hb2.as<float>(); 
                float* out = hb3.as<float>();
                out[i] = x_ga[i] / (1 + std::exp(-x_ga[i])) * x_up[i];
            }

            // 2.8.3 降维
            spyinfer::_gemv(xb5.as<float>(), block.mlp_down.as<bf16_t>(), hb3.as<float>() ,config.hidden_size, config.intermediate_size);


            //2.8.4 残差连接
            for (std::size_t i = 0; i < config.hidden_size; i++)
            {
                x.as<float>()[i] += xb5.as<float>()[i];
            }  

        }

        // 3. final norm
        spyinfer::_rms_norm_fp32(x.as<float>(), x.as<float>(), weights.norm.as<bf16_t>(), config.hidden_size, config.rms_norm_eps);

        //4. compute logits
        spyinfer::_gemv(logits.as<float>(), weights.lm_head.as<bf16_t>(), x.as<float>(), config.vocab_size, config.hidden_size);
    };

    auto sample = [](std::span<float> lg) -> uint32_t
    {
        return static_cast<uint32_t>(std::distance(lg.begin(), std::max_element(lg.begin(), lg.end())));
    };

    Timer prefill_timer;
    Timer answer_timer;
    for (std::size_t i = 0; i < tokens.size(); i++)
    {
        if (i == tokens.size() - 1)
        {
            answer_timer.reset();
        }
        forward(tokens[i], i);
    }
    std::cout << "[DEBUG] prefill cost:  " << model_load_timer.elapsed_ms() << "ms" << std::endl;

    std::string answer;
    int total_generate_token = 0;
    
    for (std::int32_t i = 0; i < kv_size; i++)
    {
        std::uint32_t token = sample({logits.as<float>(), static_cast<size_t>(config.vocab_size)});
        total_generate_token += 1;
        if (token == config.eos_token_id)
        {
            break;
        }
        if (i + 1 != kv_size)
        {
            forward(token, tokens.size() + i);
        }

        std::string decoded_token = vocab[token];
        std::cout << decoded_token << std::flush;
        answer += decoded_token;
    }
    std::cout << std::endl;
    std::cout << "[DEBUG] generate rate:  " << std::fixed << std::setprecision(2) << (double)total_generate_token / (answer_timer.elapsed_ms() / 1000) << " token/s" << std::endl;

    return 0;
}

// input: utf-8 编码的 Unicode字符串
// output: 分词前的原始字节序列
std::string _decoded_token_key(const std::string_view input)
{
    std::string output;

    /* 编码规则：
        对于!到~、¡到¬和®到ÿ这些可见字符，字典键就是这些字符，值为对应的 Byte 值。
        !到~ ASCII可打印字符（33-126）
        ¡到¬ 西班牙语特殊字符（161-172）
        ®到ÿ 其他扩展字符（174-255）

        对于其他的非可见字符（空白和控制字符），字典键为256+序号，值为对应的 Byte 值。 说明码点最高值不超过512，即编码变成utf-8 不超过两字节

        n = 0
        # 遍历所有可能的字节（0-255）
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)  # 超出原始范围的字节映射到更高Unicode码位
                n += 1


        第一个不可见区域（0-32）从0依次往后排， 比如0映射 256, 1映射 257, 2映射 258, 32映射288 映射关系 x + 256， （0-32）---> (256-288)
        第二个不可见区域（127-160）之间，它们之间的差值就变成了256-(126-33+1)=162,  127映射289（接着32映射288的位置继续） 映射关系 x + 162,
       (127-160)
       ---> (289-322) 第三个不可见区域 173， 差值是 162-(172-161+1) = 150, 映射关系x + 150    173->323
        中间可见字符会跳过，但是映射后的位置是顺序往后排的，所以才会导致映射关系不连续
    */
    output.reserve(input.size());
    std::size_t i = 0;
    while (i < input.size())
    {
        std::uint8_t b = static_cast<std::uint8_t>(input[i]);
        if (0x21 <= b && b <= 0x7e) //单字节 ASCII 可打印字符 （33-126）， unicode和ascii码对应
        {
            output += b;
            i += 1;
        }
        else
        {
            // 因为最大映射unicode码点是323 所以只会出现双字节uft-8编码
            std::uint8_t prev = b & 0b00011111;            //前三位补0,
            std::uint8_t next = input[i + 1] & 0b00111111; // 前两位补0

            //转10进制 unicode码点
            std::uint16_t code_point = (static_cast<uint16_t>(prev) << 6) | next;

            if (code_point >= 256 && code_point <= 288)
            {
                output += static_cast<std::uint8_t>(code_point - 256);
            }
            else if (code_point >= 289 && code_point <= 322)
            {
                output += static_cast<std::uint8_t>(code_point - 162);
            }
            else if (code_point == 323)
            {
                output += static_cast<std::uint8_t>(code_point - 150);
            }
            else if (code_point > 323)
            {
                std::cout << "invaild code point: " << code_point << std::endl;
            }
            else
            {
                output += static_cast<std::uint8_t>(code_point);
            }

            i += 2;
        }
    }
    return output;
}

void test()
{
    // std::string test_str = "ä½łå¥½";
    // std::cout << _decoded_token_key(test_str) << std::endl;
}

void pTensor(const Tensor& t)
{
    std::size_t dim = t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3];
    for (int i = 0; i < 20; i++)
    {
        std::cout << t.as<float>()[i] << ",";
    }
    std::cout << std::endl;
}