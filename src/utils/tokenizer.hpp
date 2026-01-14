#pragma once

#include <string>
#include <array>
#include <vector>
#include <cstdint>

#include "config.hpp"

class Tokenizer
{
    struct StaticDATrie
    {
        struct Node
        {
            int token_id = -1;
            std::array<int, 256> children; //子节点的索引

            Node() { children.fill(-1); }
        };
        std::vector<Node> meta;

        StaticDATrie()
        {
            //新建root节点，默认0节点为root
            Node root;
            meta.push_back(root);
        }

        void insert(const std::string &token, const int &token_id)
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

        std::pair<int, std::size_t> longest_match(const std::string_view data) const
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

public:
    Tokenizer(Config &cfg);
    std::vector<int> encode(const std::string_view text);
    std::string decode(const std::vector<int32_t> &tokens);
    std::string apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages) const;

private:
    void build_tire();
    std::string _decoded_token_key(const std::string_view input);

private:
    Config &config_;
    std::vector<std::string> vocab_;
    StaticDATrie trie_;
    std::int32_t bos_token_id_ = -1;
    std::int32_t eos_token_id_ = -1;
    bool byte_fallback_ = false;
};