#include "gpu_hybrid_solver.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace eda_gpu::bridge {
namespace {

using core::CscMatrix;
using core::LinearSolver;
using core::SolverMetrics;

class GpuHybridSolver final : public LinearSolver {
public:
    GpuHybridSolver(
        std::string solver_name,
        std::unique_ptr<operators::SparseNumericOperator> numeric_operator,
        operators::cpu::KluInitializationOptions initialization_options)
        : solver_name_(std::move(solver_name)),
          numeric_operator_(std::move(numeric_operator)),
          cpu_initialization_(initialization_options),
          static_symbolic_pattern_(
              initialization_options.static_pivot_symbolic_pattern) {
        if (numeric_operator_ == nullptr) {
            throw std::runtime_error("GPU hybrid solver requires a numerical operator");
        }
        metrics_.execution_model = static_symbolic_pattern_
            ? "cpu-klu-ordering+static-symbolic-fill+" +
                  std::string(numeric_operator_->name()) +
                  "-gpu-first-numeric-lu+solve"
            : "cpu-klu-symbolic+numeric-pattern-bootstrap+" +
                  std::string(numeric_operator_->name()) +
                  "-gpu-numeric-lu+solve";
        metrics_.numeric_factor_mode =
            std::string(numeric_operator_->name()) + "-fixed-pattern-numeric-lu";
        metrics_.gpu_compute = true;
    }

    [[nodiscard]] std::string_view name() const noexcept override {
        return solver_name_;
    }

    void analyze(const CscMatrix& matrix) override {
        clear_iteration_state();
        cpu_initialization_.analyze(matrix);
    }

    void factorize(const CscMatrix& matrix) override {
        initialized_ = false;
        const auto bootstrap_start = std::chrono::steady_clock::now();
        const auto& factorization = cpu_initialization_.factorize(matrix);
        metrics_.last_cpu_bootstrap_factor_ms = static_symbolic_pattern_
            ? 0.0
            : std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - bootstrap_start)
                  .count();
        metrics_.last_gpu_setup_ms =
            numeric_operator_->initialize(matrix, factorization);

        // initialize() uploads the selected fixed pattern. Task 1 requires the
        // GPU backend to perform the first numerical LU, so recompute L/U from
        // the original A values before exposing factorize(). Legacy/reference
        // adapters may still use KLU numerical pivoting to obtain that pattern;
        // their full bootstrap cost remains visible in the host wall clock.
        const auto timings = numeric_operator_->factorize(matrix.values);
        metrics_.last_numeric_factor_cpu_prepare_ms = 0.0;
        metrics_.last_numeric_factor_h2d_ms = timings.h2d_ms;
        metrics_.last_numeric_factor_kernel_ms = timings.kernel_ms;
        metrics_.last_numeric_factor_status_d2h_ms = timings.status_d2h_ms;
        initialized_ = true;
        update_operator_profile();
    }

    [[nodiscard]] std::vector<double> solve(
        const std::vector<double>& right_hand_side) override {
        if (!initialized_) {
            throw std::runtime_error(
                "GPU hybrid solve requires analyzed and factored matrix data");
        }

        std::vector<double> solution;
        const auto timings = numeric_operator_->solve(right_hand_side, solution);
        metrics_.last_solve_cpu_prepare_ms = 0.0;
        metrics_.last_solve_h2d_ms = timings.h2d_ms;
        metrics_.last_solve_kernel_ms = timings.kernel_ms;
        metrics_.last_solve_d2h_ms = timings.d2h_ms;
        metrics_.last_solve_cpu_finalize_ms = 0.0;
        metrics_.backend_calls_per_solve = timings.backend_calls;
        return solution;
    }

    [[nodiscard]] SolverMetrics metrics() const override { return metrics_; }

private:
    void update_operator_profile() {
        const auto profile = numeric_operator_->profile();
        metrics_.schedule_mode = profile.algorithm_mode;
        metrics_.device_memory_bytes = profile.device_memory_bytes;
        metrics_.estimated_numeric_factor_flops =
            profile.estimated_numeric_factor_flops;
        metrics_.estimated_triangular_solve_flops =
            profile.estimated_triangular_solve_flops;
        metrics_.schedule_operations = profile.schedule_operations;
        metrics_.persistent_grid_blocks = profile.persistent_grid_blocks;
        metrics_.fused_operations_per_launch = profile.fused_operations_per_launch;
        metrics_.scheduled_columns = profile.scheduled_columns;
        metrics_.widest_operation_columns = profile.widest_operation_columns;
        metrics_.narrow_operations = profile.narrow_operations;
        metrics_.numeric_factor_blocks = profile.numeric_factor_blocks;
        metrics_.numeric_factor_dag = profile.numeric_factor_dag;
    }

    void clear_iteration_state() {
        initialized_ = false;
        metrics_.schedule_mode = "uninitialized";
        metrics_.device_memory_bytes = 0;
        metrics_.estimated_numeric_factor_flops = 0;
        metrics_.estimated_triangular_solve_flops = 0;
        metrics_.schedule_operations = 0;
        metrics_.backend_calls_per_solve = 0;
        metrics_.persistent_grid_blocks = 0;
        metrics_.fused_operations_per_launch = 0;
        metrics_.scheduled_columns = 0;
        metrics_.widest_operation_columns = 0;
        metrics_.narrow_operations = 0;
        metrics_.numeric_factor_blocks = 0;
        metrics_.numeric_factor_dag = {};
        metrics_.last_cpu_bootstrap_factor_ms = 0.0;
        metrics_.last_gpu_setup_ms = 0.0;
        metrics_.last_numeric_factor_cpu_prepare_ms = 0.0;
        metrics_.last_numeric_factor_h2d_ms = 0.0;
        metrics_.last_numeric_factor_kernel_ms = 0.0;
        metrics_.last_numeric_factor_status_d2h_ms = 0.0;
        metrics_.last_solve_cpu_prepare_ms = 0.0;
        metrics_.last_solve_h2d_ms = 0.0;
        metrics_.last_solve_kernel_ms = 0.0;
        metrics_.last_solve_d2h_ms = 0.0;
        metrics_.last_solve_cpu_finalize_ms = 0.0;
    }

    std::string solver_name_;
    std::unique_ptr<operators::SparseNumericOperator> numeric_operator_;
    operators::cpu::KluInitialization cpu_initialization_;
    bool static_symbolic_pattern_{};
    bool initialized_{};
    SolverMetrics metrics_;
};

}  // namespace

std::unique_ptr<core::LinearSolver> make_gpu_hybrid_solver(
    std::string solver_name,
    std::unique_ptr<operators::SparseNumericOperator> numeric_operator,
    operators::cpu::KluInitializationOptions initialization_options) {
    return std::make_unique<GpuHybridSolver>(
        std::move(solver_name), std::move(numeric_operator), initialization_options);
}

}  // namespace eda_gpu::bridge
