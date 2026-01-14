#include "tokenizer.hpp"

Tokenizer::Tokenizer(Config &cfg) : config_(cfg)
{
    nlohmann::json tokenizer_config_json;
    std::ifstream file(config_.model / "tokenizer_config.json");
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open tokenizer config file");
    }
    file >> tokenizer_config_json;

    if (tokenizer_config_json.value("add_bos_token", false))
    {
        bos_token_id_ = config_.model_config.bos_token_id;
    }

    if (tokenizer_config_json.value("add_eos_token", false))
    {
        eos_token_id_ = config_.model_config.eos_token_id;
    }

    /*加载词表*/
    build_tire();
}

void Tokenizer::build_tire()
{
    nlohmann::json tokenizer_json;
    std::ifstream tokenizer_file(config_.model / "tokenizer.json");
    tokenizer_file >> tokenizer_json;
    const auto &vocab_json = tokenizer_json["model"]["vocab"];
    const auto &added_tokens = tokenizer_json["added_tokens"];

    vocab_.resize(vocab_json.size() + added_tokens.size() + 100);
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
        trie_.insert(decoded, token_id);
        vocab_[token_id] = decoded;
    }

    for (const auto &data : added_tokens)
    {
        auto key = data["content"].get<std::string>();
        auto token_id = data["id"].get<int32_t>();
        const auto decoded = _decoded_token_key(key);
        trie_.insert(decoded, token_id);
        vocab_[token_id] = decoded;
    }
}

std::string Tokenizer::_decoded_token_key(const std::string_view input)
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

std::vector<int> Tokenizer::encode(const std::string_view text)
{
    std::vector<int32_t> tokens;
    if (bos_token_id_ >= 0)
    {
        tokens.push_back(bos_token_id_);
    }
    std::size_t i = 0;
    while (i < text.size())
    {
        //最长匹配
        const std::string_view sub = {text.begin() + i, text.end()};
        auto [token_id, length] = trie_.longest_match(sub);
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
    if (eos_token_id_ >= 0)
    {
        tokens.push_back(eos_token_id_);
    }
    return std::move(tokens);
}


std::string Tokenizer::decode(const std::vector<int32_t> &tokens)
{
    std::string output;
    for (const auto &token_id : tokens)
    {
        if (token_id == bos_token_id_)
        {
            continue;
        }
        if (token_id == eos_token_id_)
        {
            break;
        }
        if (token_id < 0 || token_id >= vocab_.size())
        {
            continue;
        }
        output += vocab_[token_id];
    }
    return output;
}

std::string Tokenizer::apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages) const
{
    std::stringstream ss;
    for (const auto& msg : messages) {
        ss << "<|im_start|>" << msg.first << "\n" << msg.second << "<|im_end|>" << "\n";
    }
    // Add the prompt for the assistant to start generating
    ss << "<|im_start|>assistant\n";
    return ss.str();
}
