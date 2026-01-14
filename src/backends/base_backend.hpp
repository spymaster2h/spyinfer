#pragma once

#include "core/base_device.hpp"
#include "core/tensor.hpp"
#include "base_operator.hpp"

#include <vector>



namespace spyinfer {

class BaseBackend : public std::enable_shared_from_this<BaseBackend>
{

public:
    virtual void run(std::unordered_map<std::string, std::any>&& params)
    {
        forward_expand(params);
        compute(params);
    }
    virtual void run(std::unordered_map<std::string, std::any>& params)
    {
        forward_expand(params);
        compute(params);
    }

    const std::vector<std::shared_ptr<BaseDevice>>& get_devices() const { return devices_; }

    virtual std::shared_ptr<Tensor> create_tensor(const std::array<int64_t, 4>& shape, DataType dtype) = 0;


    virtual std::shared_ptr<MemoryBlock> allocate(size_t size_bytes) = 0;

protected:
    virtual void forward_expand(std::unordered_map<std::string, std::any>& params)
    {
        if (params.find("op_type") == params.end())
        {
            throw std::runtime_error("op_type not found in params");
        }
        params.emplace("backend", shared_from_this());
        ops_[std::any_cast<OperatorType>(params["op_type"])]->forward_expand(params);
    }

    virtual void compute(std::unordered_map<std::string, std::any>& params)
    {
        if (params.find("op_type") == params.end())
        {
            throw std::runtime_error("op_type not found in params");
        }
        ops_[std::any_cast<OperatorType>(params["op_type"])]->compute(params);
    }

   


protected:
    int device_cnt_;
    std::vector<int> device_ids_;
    std::vector<int> device_ids_ratio_;
    std::vector<std::shared_ptr<BaseDevice>> devices_;

    std::unordered_map<OperatorType, std::unique_ptr<BaseOperator>> ops_;
};
} // namespace spyinfer