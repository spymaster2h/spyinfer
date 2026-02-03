#pragma once


#include <filesystem>
#include <unordered_map>
#include <memory>

#include "core/tensor.hpp"
#include "backends/base_backend.hpp"

using namespace spyinfer;

class WeightReader
{

public:
    virtual void load_weights(const std::filesystem::path& weights_path, std::shared_ptr<BaseBackend> backend) = 0;

    std::shared_ptr<Tensor> get_tensor(const std::string& tensor_name) const
    {
        return meta_data_.at(tensor_name);
    }

protected:
    std::unordered_map<std::string, std::shared_ptr<Tensor>> meta_data_;
    std::shared_ptr<MemoryBlock> weight_buffer_;
};