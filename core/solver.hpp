#pragma once

#include "matrix.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace eda_gpu::core {

// Host-side contract shared by CPU, CUDA and MetaX adapters. A device backend
// may cache allocations and transfers internally; solve() returns a host vector.
class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    virtual void analyze(const CscMatrix& matrix) = 0;
    virtual void factorize(const CscMatrix& matrix) = 0;
    virtual void refactorize(const CscMatrix& matrix) = 0;
    [[nodiscard]] virtual std::vector<double> solve(
        const std::vector<double>& right_hand_side) = 0;
};

struct SolverDescriptor {
    std::string name;
    bool available{};
    std::string detail;
};

[[nodiscard]] std::vector<SolverDescriptor> solver_descriptors();
[[nodiscard]] std::unique_ptr<LinearSolver> create_solver(std::string_view name);

}  // namespace eda_gpu::core
