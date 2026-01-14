#pragma once

#include "weight_reader.hpp"
#include "core/tensor.hpp"

#include <filesystem>

using namespace spyinfer;

class SafeTensorsReader : public WeightReader
{
public:
    void load_weights(const std::filesystem::path& model_dir, std::shared_ptr<BaseBackend> backend) override;

private:
    void load_weights_from_safetensors(const std::filesystem::path& weights_path, std::shared_ptr<BaseBackend> backend);
};