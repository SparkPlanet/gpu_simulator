#pragma once

#include "matrix.hpp"
#include "profiler.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace eda_gpu {

enum class LookupKind {
    binary_search,
};

struct LookupPlan {
    LookupKind kind{LookupKind::binary_search};
    std::uint64_t storage_bytes{};
};

struct LevelSchedule {
    std::vector<SparseIndex> level_offsets;
    std::vector<SparseIndex> rows;

    [[nodiscard]] SparseIndex levels() const noexcept;
    [[nodiscard]] SparseIndex widest_level() const noexcept;
};

// Immutable structural product of the CPU analysis stage. Permutations use
// new-to-old notation:
//
//   permuted_A(i, j) = A(row_permutation[i], column_permutation[j])
//
// matrix_to_lu maps each value in the input CSC arrays directly to its slot in
// the combined CSR L/U pattern. Diagonal entries are stored once.
struct AnalysisPlan {
    SparseIndex dimension{};
    SparseIndex input_nonzeros{};

    std::vector<SparseIndex> row_permutation;
    std::vector<SparseIndex> inverse_row_permutation;
    std::vector<SparseIndex> column_permutation;
    std::vector<SparseIndex> inverse_column_permutation;
    // Maximum-absolute-value scaling for each row of permuted_A.
    std::vector<double> row_scale_factors;
    std::vector<SparseIndex> block_offsets;

    std::vector<SparseIndex> lu_row_offsets;
    std::vector<SparseIndex> lu_column_indices;
    std::vector<SparseIndex> diagonal_positions;
    std::vector<SparseIndex> matrix_to_lu;

    LevelSchedule factor_schedule;
    LevelSchedule forward_schedule;
    LevelSchedule backward_schedule;
    LookupPlan lookup;

    std::uint64_t symbolic_scalar_updates{};
    double estimated_factor_flops{};
    double ordering_estimated_factor_flops{};
    double ordering_estimated_l_nonzeros{};
    double ordering_estimated_u_nonzeros{};
    SparseIndex structural_rank{-1};

    [[nodiscard]] SparseIndex factor_nonzeros() const noexcept;
    [[nodiscard]] std::uint64_t lower_nonzeros() const noexcept;
    [[nodiscard]] std::uint64_t upper_nonzeros() const noexcept;
    [[nodiscard]] std::uint64_t storage_bytes() const noexcept;
    void validate(const CscMatrix& matrix) const;
};

class CpuSymbolicAnalyzer {
public:
    [[nodiscard]] AnalysisPlan analyze(
        const CscMatrix& matrix,
        Profiler& profiler) const;
};

[[nodiscard]] std::string_view to_string(LookupKind kind) noexcept;

}  // namespace eda_gpu
