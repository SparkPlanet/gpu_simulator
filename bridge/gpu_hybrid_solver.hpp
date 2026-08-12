#pragma once

#include "core/solver.hpp"
#include "operators/cpu/klu_initialization.hpp"
#include "operators/sparse_operator.hpp"

#include <memory>
#include <string>

namespace eda_gpu::bridge {

[[nodiscard]] std::unique_ptr<core::LinearSolver> make_gpu_hybrid_solver(
    std::string solver_name,
    std::unique_ptr<operators::SparseNumericOperator> numeric_operator,
    operators::cpu::KluInitializationOptions initialization_options);

}  // namespace eda_gpu::bridge
