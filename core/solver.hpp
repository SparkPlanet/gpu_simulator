#pragma once

#include "matrix.hpp"
#include "profile.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace eda_gpu::core {

struct SolverMetrics {
    std::string execution_model{"cpu"};
    std::string schedule_mode{"host"};
    std::string numeric_factor_mode{"host"};
    bool gpu_compute{};
    std::uint64_t device_memory_bytes{};
    std::uint64_t estimated_numeric_factor_flops{};
    std::uint64_t estimated_triangular_solve_flops{};
    std::int32_t schedule_operations{};
    std::int32_t backend_calls_per_solve{};
    std::int32_t persistent_grid_blocks{};
    std::int32_t fused_operations_per_launch{};
    std::int32_t scheduled_columns{};
    std::int32_t widest_operation_columns{};
    std::int32_t narrow_operations{};
    std::int32_t numeric_factor_blocks{};
    NumericFactorDagProfile numeric_factor_dag;
    double last_cpu_bootstrap_factor_ms{};
    double last_gpu_setup_ms{};
    double last_numeric_factor_cpu_prepare_ms{};
    double last_numeric_factor_h2d_ms{};
    double last_numeric_factor_kernel_ms{};
    double last_numeric_factor_status_d2h_ms{};
    double last_solve_cpu_prepare_ms{};
    double last_solve_h2d_ms{};
    double last_solve_kernel_ms{};
    double last_solve_d2h_ms{};
    double last_solve_cpu_finalize_ms{};
};

// Host-side contract shared by CPU, CUDA and MetaX adapters. A device backend
// may cache allocations and transfers internally; solve() returns a host vector.
class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    virtual void analyze(const CscMatrix& matrix) = 0;
    virtual void factorize(const CscMatrix& matrix) = 0;
    [[nodiscard]] virtual std::vector<double> solve(
        const std::vector<double>& right_hand_side) = 0;
    [[nodiscard]] virtual SolverMetrics metrics() const { return {}; }
};

struct SolverDescriptor {
    std::string name;
    bool available{};
    std::string detail;
};

[[nodiscard]] std::vector<SolverDescriptor> solver_descriptors();
[[nodiscard]] std::unique_ptr<LinearSolver> create_solver(std::string_view name);

}  // namespace eda_gpu::core
