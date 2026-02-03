#include "safetensors_reader.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

void SafeTensorsReader::load_weights(const std::filesystem::path &model_dir, std::shared_ptr<BaseBackend> backend)
{
    std::filesystem::path index_path = model_dir / "model.safetensors.index.json";

    std::vector<std::string> files;

    if (std::filesystem::exists(index_path))
    {
        std::ifstream index_file(index_path);

        nlohmann::json index_json;
        index_file >> index_json;

        for (const auto &[tensor_name, file_name] : index_json.at("weight_map").items())
        {
            files.emplace_back(file_name.get<std::string>());
        }
    }
    else
    {
        files.emplace_back("model.safetensors");
    }

    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());

    for (const auto &file_name : files)
    {
        load_weights_from_safetensors(model_dir / file_name, backend);
    }
}

void SafeTensorsReader::load_weights_from_safetensors(const std::filesystem::path &path, std::shared_ptr<BaseBackend> backend)
{
    auto fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
    {
        throw std::runtime_error("Failed to open file (POSIX): " + path.string() + " - " + std::strerror(errno));
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        close(fd);
        throw std::runtime_error("Failed to get file size (POSIX): " + path.string() + " - " + std::strerror(errno));
    }
    std::size_t tensor_size = sb.st_size;

    // mmap(地址, 长度, 保护, 标志, 文件描述符, 偏移量)
    auto data_ptr = static_cast<std::byte *>(mmap(NULL, tensor_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data_ptr == MAP_FAILED)
    {
        close(fd);
        throw std::runtime_error("Failed to memory map file (POSIX): " + path.string() + " - " + std::strerror(errno));
    }

    uint64_t metadata_size = *reinterpret_cast<const uint64_t *>(data_ptr);

    std::string v(reinterpret_cast<const char *>(data_ptr) + 8, metadata_size);
    const auto metadata_json = nlohmann::json::parse(v);
    weight_buffer_ = backend->allocate(tensor_size - metadata_size - 8);

    backend->copy_data_from_cpu(weight_buffer_->data(), reinterpret_cast<const char *>(data_ptr) + 8 + metadata_size, weight_buffer_->size());

    std::string dtype;

    for (const auto &[name, tensor_info] : metadata_json.items())
    {
        if (name == "__metadata__") continue;
        auto data_offsets = tensor_info.at("data_offsets").get<std::vector<std::int64_t>>();
        auto shape = tensor_info.at("shape").get<std::vector<std::int32_t>>();

        std::array<std::int64_t, 4> shape_array{1, 1, 1, 1};
        if (shape.size() > 4 || shape.size() < 1)
        {
            throw std::runtime_error("Invalid tensor shape for " + name);
        }
        else
        {
            std::copy(shape.rbegin(), shape.rend(), shape_array.begin());
        }
        std::reverse(shape_array.begin(), shape_array.end());

        dtype = tensor_info.at("dtype").get<std::string>();

        meta_data_[name] = Tensor::create_from_buffer(weight_buffer_, shape_array, string_to_dtype(dtype), data_offsets.at(0));
    }

    // 转fp16
    #ifdef USE_CUDA
        if (backend->get_backend_name() == "CUDA" && string_to_dtype(dtype) != DataType::fp16_t)
        {
            
            if (string_to_dtype(dtype) == DataType::bf16_t)
            {
                backend->run({
                {"op_type", OperatorType::BF16ToFP16},
                {"input", reinterpret_cast<uint16_t*>(weight_buffer_->data())},
                {"size", weight_buffer_->size() / 2}});

                for (auto &[name, tensor] : meta_data_)
                {
                    if (tensor->dtype() == DataType::bf16_t)
                    {
                        tensor->set_dtype(DataType::fp16_t);
                    }
                }
            }
        }
    #endif

    std::cout << "Loading metadata from: " << path.string() << ", size " << (tensor_size >> 20) << " MB";
}
