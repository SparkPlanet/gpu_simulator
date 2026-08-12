#pragma once

#include "core/matrix.hpp"
#include "core/profile.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace eda_gpu::operators {

// Canonical CSC payload exchanged between the CPU preparation stage and a
// device numerical backend. Row indices are sorted within every column.
struct SparseCscData {
    std::vector<std::int32_t> column_offsets;
    std::vector<std::int32_t> row_indices;
    std::vector<double> values;
};

struct InitialFactorization {
    std::int32_t dimension{};
    SparseCscData lower;
    SparseCscData upper;
    SparseCscData off_diagonal;
    std::vector<std::int32_t> row_permutation;
    std::vector<std::int32_t> column_permutation;
    std::vector<double> scale_factors;
    std::vector<std::int32_t> block_boundaries;
};

struct NumericFactorTimings {
    double h2d_ms{};
    double kernel_ms{};
    double status_d2h_ms{};
};

struct SolveTimings {
    double h2d_ms{};
    double kernel_ms{};
    double d2h_ms{};
    std::int32_t backend_calls{};
};

struct OperatorProfile {
    std::string algorithm_mode{"uninitialized"};
    std::uint64_t device_memory_bytes{};
    std::uint64_t estimated_numeric_factor_flops{};
    std::uint64_t estimated_triangular_solve_flops{};
    std::int32_t schedule_operations{};
    std::int32_t persistent_grid_blocks{};
    std::int32_t fused_operations_per_launch{};
    std::int32_t scheduled_columns{};
    std::int32_t widest_operation_columns{};
    std::int32_t narrow_operations{};
    std::int32_t numeric_factor_blocks{};
    core::NumericFactorDagProfile numeric_factor_dag;
};

// Common contract for the CUDA reference, future custom CUDA kernels and the
// future MXMACA implementation. Implementations own all device-resident state.
class SparseNumericOperator {
public:
    virtual ~SparseNumericOperator() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    virtual double initialize(
        const core::CscMatrix& matrix,
        const InitialFactorization& factorization) = 0;
    [[nodiscard]] virtual NumericFactorTimings factorize(
        const std::vector<double>& matrix_values) = 0;
    [[nodiscard]] virtual SolveTimings solve(
        const std::vector<double>& right_hand_side,
        std::vector<double>& solution) = 0;
    [[nodiscard]] virtual OperatorProfile profile() const = 0;
};

}  // namespace eda_gpu::operators
