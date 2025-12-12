#pragma once
#include <cstdint>
#include <bit>

//类型定义
using bf16_t = std::uint16_t;

float _bf16_to_fp32(bf16_t bf16) { return std::bit_cast<float>(std::uint32_t(bf16) << 16); }

template <typename T>
float _cvt_to_fp32(T t)
{
    if constexpr (std::is_same_v<T, bf16_t>)
    {
        return _bf16_to_fp32(t);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return t;
    }
}