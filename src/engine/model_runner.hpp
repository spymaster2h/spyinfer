#pragma once

#include "config.hpp"
#include "scheduler.hpp"
#include "models/qwen3.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "core/cpu_device.hpp"

namespace spyinfer {

class ModelRunner
{
public:
    ModelRunner(const Config& config);
    ~ModelRunner();

    void execute_model(ScheduleOutput& schedule_output, std::list<Sequence*>& running_queue);

private:
    Config config_;
    std::shared_ptr<BaseBackend> cpu_backend_;
    std::unique_ptr<Qwen3> model_;

    std::shared_ptr<Tensor> k_cache_tensor_;
    std::shared_ptr<Tensor> v_cache_tensor_;



};

}