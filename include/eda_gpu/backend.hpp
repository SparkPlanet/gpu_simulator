#pragma once

#include "eda_gpu/executor.hpp"
#include "eda_gpu/matrix.hpp"
#include "eda_gpu/profiler.hpp"

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace eda_gpu {

struct BackendStatistics {
    std::map<std::string, double> values;
    std::map<std::string, std::string> attributes;
};

struct BackendContext {
    Profiler& profiler;
    Executor& host_executor;
};

// Internal Task 1 lifecycle. A backend object is created for exactly one run
// and destroyed after returning one solution to the host.
class Task1Backend {
public:
    virtual ~Task1Backend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    virtual void analyze(const CscMatrix& matrix, BackendContext& context) = 0;
    virtual void factorize(const CscMatrix& matrix, BackendContext& context) = 0;
    [[nodiscard]] virtual std::vector<double> solve(
        const std::vector<double>& right_hand_side,
        BackendContext& context) = 0;
    [[nodiscard]] virtual BackendStatistics statistics() const = 0;
};

struct BackendDescriptor {
    std::string name;
    bool available{};
    std::string description;
};

[[nodiscard]] std::vector<BackendDescriptor> task1_backends();
[[nodiscard]] std::unique_ptr<Task1Backend> create_task1_backend(
    std::string_view name);

}  // namespace eda_gpu
