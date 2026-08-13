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
    // KLU/AMD estimate for L (including the diagonal) in each BTF block.
    // Reused to preallocate the exact-symbolic lower pattern without growth.
    std::vector<double> block_estimated_l_nonzeros;
    // AMD's structural-symmetry ratio per BTF diagonal block, or -1 when AMD
    // did not analyze a tiny/natural-order block.
    std::vector<double> block_input_symmetry;
    // One flag per BTF diagonal block. This describes the final factor
    // pattern: exactly symmetric input and adaptively symmetrized conservative
    // patterns both admit the elimination-tree/right-looking fast paths.
    std::vector<std::uint8_t> structurally_symmetric_blocks;

    std::vector<SparseIndex> lu_row_offsets;
    std::vector<SparseIndex> lu_column_indices;
    std::vector<SparseIndex> diagonal_positions;
    std::vector<SparseIndex> matrix_to_lu;

    LevelSchedule factor_schedule;
    LevelSchedule forward_schedule;
    LevelSchedule backward_schedule;
    // Exact number of scalar fused multiply-subtract updates performed while
    // factorizing each row. This is structural metadata used for profiling and
    // load-balanced ordering inside a dependency level; it is never uploaded.
    std::vector<std::uint64_t> factor_row_updates;
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
    // Constant-time production guard for the dimensions and outer boundaries
    // consumed by the numerical backends. The exhaustive validator below is
    // intentionally reserved for debug builds because it scans every factor
    // entry, input mapping, and dependency edge.
    void validate_dimensions(const CscMatrix& matrix) const;
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
