#pragma once

#include <array>
#include <cstdint>

#include <cmath>
#include <vector>


class ConstantTable
{
public:
    static ConstantTable& GetInstance()
    {
        static ConstantTable instance;
        return instance;
    }

    ConstantTable(const ConstantTable&) = delete;
    ConstantTable& operator=(const ConstantTable&) = delete;

    void BuildRopeTable(int max_seq_len, int head_dim, float theta)
    {
        int rotation_size = head_dim / 2;
        //shape : [pos, rotation_size]
        rope_sin_table_.resize(max_seq_len * rotation_size);
        rope_cos_table_.resize(max_seq_len * rotation_size);
        for (int i = 0; i < rope_sin_table_.size(); ++i)
        {
            int pos = i / rotation_size;
            int dim = i % rotation_size;
            float freq = std::pow(theta, -float(dim) / float(rotation_size));
            rope_sin_table_[i] = std::sin(pos * freq);
            rope_cos_table_[i] = std::cos(pos * freq);
        }
    }

public:
    std::array<float, 65536> bf16_to_fp32_table_;
    std::vector<float> rope_sin_table_;
    std::vector<float> rope_cos_table_;


private:
    ConstantTable() { BuildBf16ToFp32Table(); }

    void BuildBf16ToFp32Table()
    {
        for (uint32_t bf16_val = 0; bf16_val < 65536; ++bf16_val)
        {
            bf16_to_fp32_table_[bf16_val] = std::bit_cast<float>(bf16_val << 16);
        }
    }
};