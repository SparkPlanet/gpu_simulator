#pragma once

#include "executor.hpp"

#include <memory>
#include <string>

namespace eda_gpu {

[[nodiscard]] bool cuda_runtime_available(std::string& detail) noexcept;
[[nodiscard]] std::unique_ptr<Executor> make_cuda_executor(
    Profiler& profiler,
    int device_id = 0);

}  // namespace eda_gpu
