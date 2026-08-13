#include "analysis.hpp"

extern "C" {
#include "ngspice/klu.h"
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace eda_gpu {
namespace {

[[nodiscard]] std::vector<SparseIndex> inverse_permutation(
    const std::vector<SparseIndex>& permutation,
    SparseIndex dimension) {
    if (permutation.size() != static_cast<std::size_t>(dimension)) {
        throw std::runtime_error("permutation dimension mismatch");
    }
    std::vector<SparseIndex> inverse(static_cast<std::size_t>(dimension), -1);
    for (SparseIndex index = 0; index < dimension; ++index) {
        const auto original = permutation[static_cast<std::size_t>(index)];
        if (original < 0 || original >= dimension ||
            inverse[static_cast<std::size_t>(original)] != -1) {
            throw std::runtime_error("ordering produced an invalid permutation");
        }
        inverse[static_cast<std::size_t>(original)] = index;
    }
    return inverse;
}

struct PermutedRowPattern {
    std::vector<SparseIndex> row_offsets;
    std::vector<SparseIndex> columns;
    // Parallel to columns: position of the entry in the original CSC arrays.
    std::vector<SparseIndex> original_positions;
};

struct SymmetrizedLowerPattern {
    // Block-local CSR offsets; columns retain global permuted indices.
    std::vector<SparseIndex> row_offsets;
    std::vector<SparseIndex> columns;
};

struct SymbolicPhaseTimings {
    double symmetry_detection_ms{};
    double symmetrized_lower_build_ms{};
    double elimination_tree_ms{};
    double lower_reach_ms{};
    double upper_transpose_ms{};
    double btf_off_diagonal_ms{};
    double factor_assembly_ms{};
    double generic_fallback_ms{};
    std::uint64_t symmetrized_blocks{};
};

// A symmetric envelope remains a correct fixed-pivot LU superset: every
// directed fill edge i->j created through pivot k is also the undirected fill
// edge {i,j} in the elimination graph of A union A^T. Apply it only when AMD
// reports high input symmetry so the conservative pattern cannot inflate much.
constexpr double kSymmetrizedEnvelopeThreshold = 0.98;
constexpr double kSymmetrizedEnvelopeMaximumInputExpansion = 6.0;
constexpr SparseIndex kSymmetrizedEnvelopeMinimumBlockSize = 256;

using AnalysisClock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_ms(AnalysisClock::time_point start) noexcept {
    return std::chrono::duration<double, std::milli>(AnalysisClock::now() - start)
        .count();
}

[[nodiscard]] PermutedRowPattern build_permuted_rows(
    const CscMatrix& matrix,
    AnalysisPlan& plan) {
    PermutedRowPattern pattern;
    pattern.row_offsets.assign(static_cast<std::size_t>(matrix.dimension) + 1U, 0);
    plan.row_scale_factors.assign(
        static_cast<std::size_t>(matrix.dimension), 0.0);
    // Row counting and numerical scaling consume the same input entries. Keep
    // them in one streaming pass so analysis never rereads the full CSC value
    // array solely to build scaling metadata.
    for (std::size_t position = 0; position < matrix.row_indices.size(); ++position) {
        const auto original_row = matrix.row_indices[position];
        const auto new_row =
            plan.inverse_row_permutation[static_cast<std::size_t>(original_row)];
        ++pattern.row_offsets[static_cast<std::size_t>(new_row) + 1U];
        auto& scale = plan.row_scale_factors[static_cast<std::size_t>(new_row)];
        scale = std::max(scale, std::abs(matrix.values[position]));
    }
    for (auto& scale : plan.row_scale_factors) {
        if (scale == 0.0) scale = 1.0;
    }
    std::partial_sum(
        pattern.row_offsets.begin(), pattern.row_offsets.end(),
        pattern.row_offsets.begin());
    pattern.columns.resize(static_cast<std::size_t>(matrix.nonzeros()));
    pattern.original_positions.resize(static_cast<std::size_t>(matrix.nonzeros()));
    auto write_positions = pattern.row_offsets;

    // Visit columns in their final order. Every row is therefore emitted in
    // sorted order directly, without n heap allocations or per-row sorting.
    for (SparseIndex new_column = 0; new_column < matrix.dimension; ++new_column) {
        const auto original_column =
            plan.column_permutation[static_cast<std::size_t>(new_column)];
        for (auto position = matrix.column_offsets[static_cast<std::size_t>(original_column)];
             position < matrix.column_offsets[static_cast<std::size_t>(original_column) + 1U];
             ++position) {
            const auto original_row = matrix.row_indices[static_cast<std::size_t>(position)];
            const auto new_row =
                plan.inverse_row_permutation[static_cast<std::size_t>(original_row)];
            const auto destination =
                write_positions[static_cast<std::size_t>(new_row)]++;
            pattern.columns[static_cast<std::size_t>(destination)] = new_column;
            pattern.original_positions[static_cast<std::size_t>(destination)] = position;
        }
    }
#ifndef NDEBUG
    // The final-column traversal above guarantees sorted, unique rows when
    // the validated input CSC has no duplicates. Recheck that invariant only
    // in debug builds; production analysis must not rescan all input entries.
    for (SparseIndex row = 0; row < matrix.dimension; ++row) {
        const auto begin = pattern.row_offsets[static_cast<std::size_t>(row)];
        const auto end = pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
        if (!std::is_sorted(
                pattern.columns.begin() + begin, pattern.columns.begin() + end) ||
            std::adjacent_find(
                pattern.columns.begin() + begin, pattern.columns.begin() + end) !=
                pattern.columns.begin() + end) {
            throw std::runtime_error("permutation unexpectedly created duplicate entries");
        }
    }
#endif
    return pattern;
}

[[nodiscard]] bool row_contains(
    const PermutedRowPattern& pattern,
    SparseIndex row,
    SparseIndex column) {
    const auto begin = pattern.columns.begin() +
                       pattern.row_offsets[static_cast<std::size_t>(row)];
    const auto end = pattern.columns.begin() +
                     pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
    return std::binary_search(begin, end, column);
}

[[nodiscard]] bool block_is_structurally_symmetric(
    const PermutedRowPattern& pattern,
    SparseIndex block_begin,
    SparseIndex block_end) {
    for (auto row = block_begin; row < block_end; ++row) {
        const auto begin = pattern.row_offsets[static_cast<std::size_t>(row)];
        const auto end = pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
        for (auto position = begin; position < end; ++position) {
            const auto column = pattern.columns[static_cast<std::size_t>(position)];
            if (column < block_begin) return false;
            if (column >= block_end) break;
            if (!row_contains(pattern, column, row)) return false;
        }
    }
    return true;
}

void map_input_row_to_factor(
    const PermutedRowPattern& pattern,
    SparseIndex row,
    AnalysisPlan& plan) {
    auto factor_position = plan.lu_row_offsets[static_cast<std::size_t>(row)];
    const auto factor_end =
        plan.lu_row_offsets[static_cast<std::size_t>(row) + 1U];
    const auto input_begin = pattern.row_offsets[static_cast<std::size_t>(row)];
    const auto input_end = pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
    for (auto input_position = input_begin; input_position < input_end;
         ++input_position) {
        const auto column = pattern.columns[static_cast<std::size_t>(input_position)];
        while (factor_position < factor_end &&
               plan.lu_column_indices[static_cast<std::size_t>(factor_position)] <
                   column) {
            ++factor_position;
        }
        if (factor_position >= factor_end ||
            plan.lu_column_indices[static_cast<std::size_t>(factor_position)] !=
                column) {
            throw std::logic_error("input entry is absent from the symbolic LU pattern");
        }
        const auto original_position =
            pattern.original_positions[static_cast<std::size_t>(input_position)];
        plan.matrix_to_lu[static_cast<std::size_t>(original_position)] =
            factor_position;
    }
}

void append_generic_symbolic_block(
    const PermutedRowPattern& pattern,
    SparseIndex block_begin,
    SparseIndex block_end,
    AnalysisPlan& plan,
    std::vector<SparseIndex>& marker,
    std::vector<SparseIndex>& row_columns,
    std::vector<SparseIndex>& lower_frontier) {
    for (auto row = block_begin; row < block_end; ++row) {
        const auto input_begin = pattern.row_offsets[static_cast<std::size_t>(row)];
        const auto input_end = pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
        row_columns.assign(
            pattern.columns.begin() + input_begin,
            pattern.columns.begin() + input_end);
        const auto input_diagonal =
            std::lower_bound(row_columns.begin(), row_columns.end(), row);
        if (input_diagonal == row_columns.end() || *input_diagonal != row) {
            row_columns.insert(input_diagonal, row);
        }

        lower_frontier.clear();
        for (const auto column : row_columns) {
            marker[static_cast<std::size_t>(column)] = row;
            if (column < row) lower_frontier.push_back(column);
        }
        std::make_heap(
            lower_frontier.begin(), lower_frontier.end(), std::greater<>{});

        while (!lower_frontier.empty()) {
            std::pop_heap(
                lower_frontier.begin(), lower_frontier.end(), std::greater<>{});
            const auto pivot_row = lower_frontier.back();
            lower_frontier.pop_back();
            const auto upper_begin = static_cast<std::size_t>(
                plan.diagonal_positions[static_cast<std::size_t>(pivot_row)] + 1);
            const auto upper_end = static_cast<std::size_t>(
                plan.lu_row_offsets[static_cast<std::size_t>(pivot_row) + 1U]);
            for (auto position = upper_begin; position < upper_end; ++position) {
                const auto column = plan.lu_column_indices[position];
                if (marker[static_cast<std::size_t>(column)] == row) continue;
                marker[static_cast<std::size_t>(column)] = row;
                row_columns.push_back(column);
                if (column < row) {
                    lower_frontier.push_back(column);
                    std::push_heap(
                        lower_frontier.begin(), lower_frontier.end(),
                        std::greater<>{});
                }
            }
        }

        std::sort(row_columns.begin(), row_columns.end());
        if (plan.lu_column_indices.size() + row_columns.size() >
            static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max())) {
            throw std::runtime_error("symbolic LU exceeds the 32-bit factor index ABI");
        }
        const auto row_begin = plan.lu_column_indices.size();
        const auto diagonal = std::lower_bound(row_columns.begin(), row_columns.end(), row);
        if (diagonal == row_columns.end() || *diagonal != row) {
            throw std::logic_error("symbolic LU failed to create a diagonal slot");
        }
        plan.diagonal_positions[static_cast<std::size_t>(row)] =
            static_cast<SparseIndex>(row_begin +
                                     static_cast<std::size_t>(diagonal - row_columns.begin()));
        plan.lu_column_indices.insert(
            plan.lu_column_indices.end(), row_columns.begin(), row_columns.end());
        plan.lu_row_offsets[static_cast<std::size_t>(row) + 1U] =
            static_cast<SparseIndex>(plan.lu_column_indices.size());

    }
}

[[nodiscard]] SymmetrizedLowerPattern build_symmetrized_lower_pattern(
    const PermutedRowPattern& pattern,
    SparseIndex block_begin,
    SparseIndex block_end) {
    const auto block_size = block_end - block_begin;
    SymmetrizedLowerPattern result;
    result.row_offsets.assign(static_cast<std::size_t>(block_size) + 1U, 0);

    // Each directed diagonal-block edge contributes its undirected lower
    // endpoint pair. Reciprocal input edges temporarily duplicate the same
    // pair and are removed after the contiguous counting fill.
    for (auto row = block_begin; row < block_end; ++row) {
        const auto begin = pattern.row_offsets[static_cast<std::size_t>(row)];
        const auto end = pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
        for (auto position = begin; position < end; ++position) {
            const auto column = pattern.columns[static_cast<std::size_t>(position)];
            if (column < block_begin || column >= block_end || column == row) continue;
            const auto high = std::max(row, column);
            ++result.row_offsets[static_cast<std::size_t>(high - block_begin) + 1U];
        }
    }
    std::partial_sum(
        result.row_offsets.begin(), result.row_offsets.end(),
        result.row_offsets.begin());
    std::vector<SparseIndex> raw_columns(
        static_cast<std::size_t>(result.row_offsets.back()));
    auto write_positions = result.row_offsets;
    for (auto row = block_begin; row < block_end; ++row) {
        const auto begin = pattern.row_offsets[static_cast<std::size_t>(row)];
        const auto end = pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
        for (auto position = begin; position < end; ++position) {
            const auto column = pattern.columns[static_cast<std::size_t>(position)];
            if (column < block_begin || column >= block_end || column == row) continue;
            const auto high = std::max(row, column);
            const auto low = std::min(row, column);
            raw_columns[static_cast<std::size_t>(
                write_positions[static_cast<std::size_t>(high - block_begin)]++)] = low;
        }
    }

    auto raw_offsets = std::move(result.row_offsets);
    result.row_offsets.assign(static_cast<std::size_t>(block_size) + 1U, 0);
    result.columns.reserve(raw_columns.size());
    for (SparseIndex local_row = 0; local_row < block_size; ++local_row) {
        const auto begin = raw_offsets[static_cast<std::size_t>(local_row)];
        const auto end = raw_offsets[static_cast<std::size_t>(local_row) + 1U];
        auto first = raw_columns.begin() + begin;
        auto last = raw_columns.begin() + end;
        std::sort(first, last);
        last = std::unique(first, last);
        result.columns.insert(result.columns.end(), first, last);
        if (result.columns.size() >
            static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max())) {
            throw std::runtime_error(
                "symmetrized input pattern exceeds the 32-bit index ABI");
        }
        result.row_offsets[static_cast<std::size_t>(local_row) + 1U] =
            static_cast<SparseIndex>(result.columns.size());
    }
    return result;
}

void append_symmetric_symbolic_block(
    const PermutedRowPattern& pattern,
    SparseIndex block_begin,
    SparseIndex block_end,
    AnalysisPlan& plan,
    std::vector<SparseIndex>& marker,
    double estimated_l_nonzeros,
    SymbolicPhaseTimings& timings,
    const SymmetrizedLowerPattern* symmetrized_lower = nullptr) {
    const auto block_size = block_end - block_begin;
    const auto local_size = static_cast<std::size_t>(block_size);
    std::vector<SparseIndex> parent(local_size, -1);
    std::vector<SparseIndex> ancestor(local_size, -1);

    {
        const auto start = AnalysisClock::now();
        // Elimination tree of the structurally symmetric diagonal block. The
        // lower part of row k is the upper part of CSC column k by symmetry.
        for (SparseIndex local_row = 0; local_row < block_size; ++local_row) {
            const auto row = block_begin + local_row;
            const auto begin = symmetrized_lower != nullptr
                                   ? symmetrized_lower->row_offsets[
                                         static_cast<std::size_t>(local_row)]
                                   : pattern.row_offsets[static_cast<std::size_t>(row)];
            const auto end = symmetrized_lower != nullptr
                                 ? symmetrized_lower->row_offsets[
                                       static_cast<std::size_t>(local_row) + 1U]
                                 : pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
            for (auto position = begin; position < end; ++position) {
                const auto column = symmetrized_lower != nullptr
                                        ? symmetrized_lower->columns[
                                              static_cast<std::size_t>(position)]
                                        : pattern.columns[
                                              static_cast<std::size_t>(position)];
                if (symmetrized_lower == nullptr && column >= row) break;
                auto node = column - block_begin;
                while (node != -1 && node < local_row) {
                    const auto next = ancestor[static_cast<std::size_t>(node)];
                    ancestor[static_cast<std::size_t>(node)] = local_row;
                    if (next == -1) parent[static_cast<std::size_t>(node)] = local_row;
                    node = next;
                }
            }
        }
        timings.elimination_tree_ms += elapsed_ms(start);
    }

    std::vector<SparseIndex> lower_offsets(local_size + 1U, 0);
    std::vector<SparseIndex> lower_columns;
    if (std::isfinite(estimated_l_nonzeros) &&
        estimated_l_nonzeros >= static_cast<double>(block_size)) {
        lower_columns.reserve(static_cast<std::size_t>(
            estimated_l_nonzeros - static_cast<double>(block_size)));
    }
    {
        const auto start = AnalysisClock::now();
        std::vector<SparseIndex> reach;
        std::vector<SparseIndex> path;
        std::vector<SparseIndex> reach_marker(local_size, -1);
        for (SparseIndex local_row = 0; local_row < block_size; ++local_row) {
            const auto row = block_begin + local_row;
            reach.clear();
            const auto begin = symmetrized_lower != nullptr
                                   ? symmetrized_lower->row_offsets[
                                         static_cast<std::size_t>(local_row)]
                                   : pattern.row_offsets[static_cast<std::size_t>(row)];
            const auto end = symmetrized_lower != nullptr
                                 ? symmetrized_lower->row_offsets[
                                       static_cast<std::size_t>(local_row) + 1U]
                                 : pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
            for (auto position = begin; position < end; ++position) {
                const auto column = symmetrized_lower != nullptr
                                        ? symmetrized_lower->columns[
                                              static_cast<std::size_t>(position)]
                                        : pattern.columns[
                                              static_cast<std::size_t>(position)];
                if (symmetrized_lower == nullptr && column >= row) break;
                auto node = column - block_begin;
                path.clear();
                while (node != -1 && node < local_row &&
                       reach_marker[static_cast<std::size_t>(node)] != local_row) {
                    reach_marker[static_cast<std::size_t>(node)] = local_row;
                    path.push_back(node);
                    node = parent[static_cast<std::size_t>(node)];
                }
                for (auto item = path.rbegin(); item != path.rend(); ++item) {
                    reach.push_back(block_begin + *item);
                }
            }
            std::sort(reach.begin(), reach.end());
            lower_columns.insert(lower_columns.end(), reach.begin(), reach.end());
            if (lower_columns.size() >
                static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max())) {
                throw std::runtime_error(
                    "symbolic LU exceeds the 32-bit factor index ABI");
            }
            lower_offsets[static_cast<std::size_t>(local_row) + 1U] =
                static_cast<SparseIndex>(lower_columns.size());
        }
        timings.lower_reach_ms += elapsed_ms(start);
    }

    std::vector<SparseIndex> upper_offsets(local_size + 1U, 0);
    std::vector<SparseIndex> upper_columns(lower_columns.size());
    {
        const auto start = AnalysisClock::now();
        for (const auto column : lower_columns) {
            ++upper_offsets[static_cast<std::size_t>(column - block_begin) + 1U];
        }
        std::partial_sum(
            upper_offsets.begin(), upper_offsets.end(), upper_offsets.begin());
        auto upper_write = upper_offsets;
        for (SparseIndex local_row = 0; local_row < block_size; ++local_row) {
            const auto row = block_begin + local_row;
            const auto begin = lower_offsets[static_cast<std::size_t>(local_row)];
            const auto end = lower_offsets[static_cast<std::size_t>(local_row) + 1U];
            for (auto position = begin; position < end; ++position) {
                const auto column = lower_columns[static_cast<std::size_t>(position)];
                upper_columns[static_cast<std::size_t>(
                    upper_write[static_cast<std::size_t>(column - block_begin)]++)] = row;
            }
        }
        timings.upper_transpose_ms += elapsed_ms(start);
    }

    // Off-diagonal BTF entries belong to U. Propagate only those row patterns
    // through the already-known L reach; the expensive diagonal fill closure
    // has already been handled by the elimination tree above.
    std::vector<SparseIndex> off_diagonal_offsets(local_size + 1U, 0);
    std::vector<SparseIndex> off_diagonal_columns;
    {
        const auto start = AnalysisClock::now();
        for (SparseIndex local_row = 0; local_row < block_size; ++local_row) {
            const auto row = block_begin + local_row;
            const auto current_begin = off_diagonal_columns.size();
            const auto input_begin = pattern.row_offsets[static_cast<std::size_t>(row)];
            const auto input_end = pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
            for (auto position = input_begin; position < input_end; ++position) {
                const auto column = pattern.columns[static_cast<std::size_t>(position)];
                if (column < block_end) continue;
                marker[static_cast<std::size_t>(column)] = row;
                off_diagonal_columns.push_back(column);
            }
            const auto lower_begin = lower_offsets[static_cast<std::size_t>(local_row)];
            const auto lower_end = lower_offsets[static_cast<std::size_t>(local_row) + 1U];
            for (auto lower = lower_begin; lower < lower_end; ++lower) {
                const auto pivot =
                    lower_columns[static_cast<std::size_t>(lower)] - block_begin;
                const auto pivot_begin =
                    off_diagonal_offsets[static_cast<std::size_t>(pivot)];
                const auto pivot_end =
                    off_diagonal_offsets[static_cast<std::size_t>(pivot) + 1U];
                for (auto position = pivot_begin; position < pivot_end; ++position) {
                    const auto column =
                        off_diagonal_columns[static_cast<std::size_t>(position)];
                    if (marker[static_cast<std::size_t>(column)] == row) continue;
                    marker[static_cast<std::size_t>(column)] = row;
                    off_diagonal_columns.push_back(column);
                }
            }
            std::sort(
                off_diagonal_columns.begin() +
                    static_cast<std::ptrdiff_t>(current_begin),
                off_diagonal_columns.end());
            if (off_diagonal_columns.size() >
                static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max())) {
                throw std::runtime_error(
                    "symbolic LU exceeds the 32-bit factor index ABI");
            }
            off_diagonal_offsets[static_cast<std::size_t>(local_row) + 1U] =
                static_cast<SparseIndex>(off_diagonal_columns.size());
        }
        timings.btf_off_diagonal_ms += elapsed_ms(start);
    }

    {
        const auto start = AnalysisClock::now();
        for (SparseIndex local_row = 0; local_row < block_size; ++local_row) {
            const auto row = block_begin + local_row;
            const auto lower_begin = lower_offsets[static_cast<std::size_t>(local_row)];
            const auto lower_end = lower_offsets[static_cast<std::size_t>(local_row) + 1U];
            plan.lu_column_indices.insert(
                plan.lu_column_indices.end(),
                lower_columns.begin() + lower_begin,
                lower_columns.begin() + lower_end);
            plan.diagonal_positions[static_cast<std::size_t>(row)] =
                static_cast<SparseIndex>(plan.lu_column_indices.size());
            plan.lu_column_indices.push_back(row);
            const auto upper_begin = upper_offsets[static_cast<std::size_t>(local_row)];
            const auto upper_end = upper_offsets[static_cast<std::size_t>(local_row) + 1U];
            plan.lu_column_indices.insert(
                plan.lu_column_indices.end(),
                upper_columns.begin() + upper_begin,
                upper_columns.begin() + upper_end);
            const auto off_begin =
                off_diagonal_offsets[static_cast<std::size_t>(local_row)];
            const auto off_end =
                off_diagonal_offsets[static_cast<std::size_t>(local_row) + 1U];
            plan.lu_column_indices.insert(
                plan.lu_column_indices.end(),
                off_diagonal_columns.begin() + off_begin,
                off_diagonal_columns.begin() + off_end);
            if (plan.lu_column_indices.size() >
                static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max())) {
                throw std::runtime_error(
                    "symbolic LU exceeds the 32-bit factor index ABI");
            }
            plan.lu_row_offsets[static_cast<std::size_t>(row) + 1U] =
                static_cast<SparseIndex>(plan.lu_column_indices.size());
        }
        timings.factor_assembly_ms += elapsed_ms(start);
    }
}

void append_symbolic_pattern(
    const PermutedRowPattern& pattern,
    AnalysisPlan& plan,
    SymbolicPhaseTimings& timings) {
    const auto dimension = plan.dimension;
    plan.lu_row_offsets.assign(static_cast<std::size_t>(dimension) + 1U, 0);
    plan.diagonal_positions.assign(static_cast<std::size_t>(dimension), -1);
    auto reserve_nonzeros = static_cast<std::size_t>(
        std::max<SparseIndex>(plan.input_nonzeros, dimension));
    const auto estimated_diagonal_factor_nonzeros =
        plan.ordering_estimated_l_nonzeros + plan.ordering_estimated_u_nonzeros -
        static_cast<double>(dimension);
    if (std::isfinite(estimated_diagonal_factor_nonzeros) &&
        estimated_diagonal_factor_nonzeros > 0.0) {
        // AMD estimates only diagonal BTF factors. Leave headroom for the
        // propagated off-diagonal U pattern so assembly normally writes once.
        const auto estimate_with_off_diagonal_headroom =
            std::ceil(1.5 * estimated_diagonal_factor_nonzeros);
        reserve_nonzeros = std::max(
            reserve_nonzeros,
            static_cast<std::size_t>(std::min(
                estimate_with_off_diagonal_headroom,
                static_cast<double>(std::numeric_limits<SparseIndex>::max()))));
    }
    plan.lu_column_indices.reserve(reserve_nonzeros);
    if (plan.structurally_symmetric_blocks.size() + 1U !=
        plan.block_offsets.size()) {
        plan.structurally_symmetric_blocks.assign(
            plan.block_offsets.size() - 1U, std::uint8_t{2});
    }

    std::vector<SparseIndex> marker(static_cast<std::size_t>(dimension), -1);
    std::vector<SparseIndex> row_columns;
    std::vector<SparseIndex> lower_frontier;
    for (std::size_t block = 0; block + 1U < plan.block_offsets.size(); ++block) {
        const auto block_begin = plan.block_offsets[block];
        const auto block_end = plan.block_offsets[block + 1U];
        auto classification = plan.structurally_symmetric_blocks[block];
        if (classification > 1U) {
            const auto symmetry_start = AnalysisClock::now();
            classification =
                block_is_structurally_symmetric(pattern, block_begin, block_end)
                    ? std::uint8_t{1}
                    : std::uint8_t{0};
            timings.symmetry_detection_ms += elapsed_ms(symmetry_start);
            plan.structurally_symmetric_blocks[block] = classification;
        }
        const auto symmetric = classification != 0U;
        const auto input_symmetry =
            block < plan.block_input_symmetry.size()
                ? plan.block_input_symmetry[block]
                : -1.0;
        std::uint64_t diagonal_input_nonzeros{};
        if (!symmetric && block_end - block_begin >=
                              kSymmetrizedEnvelopeMinimumBlockSize) {
            for (auto row = block_begin; row < block_end; ++row) {
                const auto begin =
                    pattern.row_offsets[static_cast<std::size_t>(row)];
                const auto end =
                    pattern.row_offsets[static_cast<std::size_t>(row) + 1U];
                for (auto position = begin; position < end; ++position) {
                    const auto column =
                        pattern.columns[static_cast<std::size_t>(position)];
                    diagonal_input_nonzeros +=
                        column >= block_begin && column < block_end;
                }
            }
        }
        const auto estimated_symmetric_factor_nonzeros =
            2.0 * plan.block_estimated_l_nonzeros[block] -
            static_cast<double>(block_end - block_begin);
        const auto envelope_fits_fill_budget =
            diagonal_input_nonzeros != 0U &&
            std::isfinite(estimated_symmetric_factor_nonzeros) &&
            estimated_symmetric_factor_nonzeros <=
                kSymmetrizedEnvelopeMaximumInputExpansion *
                    static_cast<double>(diagonal_input_nonzeros);
        const auto use_symmetrized_envelope =
            !symmetric &&
            block_end - block_begin >= kSymmetrizedEnvelopeMinimumBlockSize &&
            (input_symmetry >= kSymmetrizedEnvelopeThreshold ||
             envelope_fits_fill_budget);
        if (symmetric) {
            append_symmetric_symbolic_block(
                pattern, block_begin, block_end, plan, marker,
                plan.block_estimated_l_nonzeros[block], timings);
        } else if (use_symmetrized_envelope) {
            const auto build_start = AnalysisClock::now();
            auto lower = build_symmetrized_lower_pattern(
                pattern, block_begin, block_end);
            timings.symmetrized_lower_build_ms += elapsed_ms(build_start);
            append_symmetric_symbolic_block(
                pattern, block_begin, block_end, plan, marker,
                plan.block_estimated_l_nonzeros[block], timings, &lower);
            // From this point onward the conservative factor pattern itself is
            // symmetric, so ordinary L dependencies are exact for right-looking.
            plan.structurally_symmetric_blocks[block] = 1U;
            ++timings.symmetrized_blocks;
        } else {
            const auto generic_start = AnalysisClock::now();
            append_generic_symbolic_block(
                pattern, block_begin, block_end, plan, marker, row_columns,
                lower_frontier);
            timings.generic_fallback_ms += elapsed_ms(generic_start);
        }
    }
}

void build_matrix_mapping(
    const PermutedRowPattern& pattern,
    AnalysisPlan& plan) {
    plan.matrix_to_lu.assign(static_cast<std::size_t>(plan.input_nonzeros), -1);
    for (SparseIndex row = 0; row < plan.dimension; ++row) {
        map_input_row_to_factor(pattern, row, plan);
    }
}

[[nodiscard]] LevelSchedule make_schedule(const std::vector<SparseIndex>& row_levels) {
    LevelSchedule schedule;
    if (row_levels.empty()) {
        schedule.level_offsets.push_back(0);
        return schedule;
    }
    const auto maximum = *std::max_element(row_levels.begin(), row_levels.end());
    schedule.level_offsets.assign(static_cast<std::size_t>(maximum) + 2U, 0);
    for (const auto level : row_levels) {
        ++schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
    }
    std::partial_sum(
        schedule.level_offsets.begin(),
        schedule.level_offsets.end(),
        schedule.level_offsets.begin());
    schedule.rows.assign(row_levels.size(), 0);
    auto write_positions = schedule.level_offsets;
    for (SparseIndex row = 0; row < static_cast<SparseIndex>(row_levels.size()); ++row) {
        const auto level = row_levels[static_cast<std::size_t>(row)];
        schedule.rows[static_cast<std::size_t>(
            write_positions[static_cast<std::size_t>(level)]++)] = row;
    }
    return schedule;
}

void build_schedules(AnalysisPlan& plan) {
    std::vector<SparseIndex> forward_levels(static_cast<std::size_t>(plan.dimension), 0);
    plan.factor_row_updates.assign(static_cast<std::size_t>(plan.dimension), 0U);
    std::uint64_t updates{};
    std::uint64_t lower_nonzeros{};
    for (SparseIndex row = 0; row < plan.dimension; ++row) {
        SparseIndex level{};
        const auto begin = plan.lu_row_offsets[static_cast<std::size_t>(row)];
        const auto diagonal = plan.diagonal_positions[static_cast<std::size_t>(row)];
        lower_nonzeros += static_cast<std::uint64_t>(diagonal - begin);
        std::uint64_t row_updates{};
        for (auto position = begin;
             position < diagonal; ++position) {
            const auto dependency = plan.lu_column_indices[static_cast<std::size_t>(position)];
            level = std::max(
                level,
                static_cast<SparseIndex>(
                    forward_levels[static_cast<std::size_t>(dependency)] + 1));
            const auto dependency_upper =
                plan.lu_row_offsets[static_cast<std::size_t>(dependency) + 1U] -
                plan.diagonal_positions[static_cast<std::size_t>(dependency)] - 1;
            row_updates += static_cast<std::uint64_t>(dependency_upper);
        }
        forward_levels[static_cast<std::size_t>(row)] = level;
        plan.factor_row_updates[static_cast<std::size_t>(row)] = row_updates;
        updates += row_updates;
    }
    plan.symbolic_scalar_updates = updates;
    plan.estimated_factor_flops =
        static_cast<double>(lower_nonzeros) + 2.0 * static_cast<double>(updates);
    plan.factor_schedule = make_schedule(forward_levels);
    plan.forward_schedule = plan.factor_schedule;

    std::vector<SparseIndex> backward_levels(static_cast<std::size_t>(plan.dimension), 0);
    for (SparseIndex row = plan.dimension; row-- > 0;) {
        SparseIndex level{};
        const auto begin = plan.diagonal_positions[static_cast<std::size_t>(row)] + 1;
        const auto end = plan.lu_row_offsets[static_cast<std::size_t>(row) + 1U];
        for (auto position = begin; position < end; ++position) {
            const auto dependency = plan.lu_column_indices[static_cast<std::size_t>(position)];
            level = std::max(
                level,
                static_cast<SparseIndex>(
                    backward_levels[static_cast<std::size_t>(dependency)] + 1));
        }
        backward_levels[static_cast<std::size_t>(row)] = level;
    }
    plan.backward_schedule = make_schedule(backward_levels);
}

void validate_schedule(
    const LevelSchedule& schedule,
    const AnalysisPlan& plan,
    bool backward) {
    if (schedule.level_offsets.empty() || schedule.level_offsets.front() != 0 ||
        schedule.level_offsets.back() != plan.dimension ||
        schedule.rows.size() != static_cast<std::size_t>(plan.dimension)) {
        throw std::runtime_error("invalid level schedule dimensions");
    }
    std::vector<SparseIndex> row_level(static_cast<std::size_t>(plan.dimension), -1);
    for (SparseIndex level = 0; level < schedule.levels(); ++level) {
        const auto begin = schedule.level_offsets[static_cast<std::size_t>(level)];
        const auto end = schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
        if (begin > end) throw std::runtime_error("level offsets are not monotonic");
        for (auto position = begin; position < end; ++position) {
            const auto row = schedule.rows[static_cast<std::size_t>(position)];
            if (row < 0 || row >= plan.dimension ||
                row_level[static_cast<std::size_t>(row)] != -1) {
                throw std::runtime_error("level schedule is not a row permutation");
            }
            row_level[static_cast<std::size_t>(row)] = level;
        }
    }
    for (SparseIndex row = 0; row < plan.dimension; ++row) {
        const auto diagonal = plan.diagonal_positions[static_cast<std::size_t>(row)];
        const auto begin = backward ? diagonal + 1
                                    : plan.lu_row_offsets[static_cast<std::size_t>(row)];
        const auto end = backward
                             ? plan.lu_row_offsets[static_cast<std::size_t>(row) + 1U]
                             : diagonal;
        for (auto position = begin; position < end; ++position) {
            const auto dependency = plan.lu_column_indices[static_cast<std::size_t>(position)];
            if (row_level[static_cast<std::size_t>(dependency)] >=
                row_level[static_cast<std::size_t>(row)]) {
                throw std::runtime_error("level schedule violates an LU dependency");
            }
        }
    }
}

}  // namespace

SparseIndex LevelSchedule::levels() const noexcept {
    return level_offsets.empty() ? 0 : static_cast<SparseIndex>(level_offsets.size() - 1U);
}

SparseIndex LevelSchedule::widest_level() const noexcept {
    SparseIndex widest{};
    for (std::size_t level = 0; level + 1U < level_offsets.size(); ++level) {
        widest = std::max(widest, level_offsets[level + 1U] - level_offsets[level]);
    }
    return widest;
}

SparseIndex AnalysisPlan::factor_nonzeros() const noexcept {
    return static_cast<SparseIndex>(lu_column_indices.size());
}

std::uint64_t AnalysisPlan::lower_nonzeros() const noexcept {
    std::uint64_t result{};
    for (SparseIndex row = 0; row < dimension; ++row) {
        result += static_cast<std::uint64_t>(
            diagonal_positions[static_cast<std::size_t>(row)] -
            lu_row_offsets[static_cast<std::size_t>(row)]);
    }
    return result;
}

std::uint64_t AnalysisPlan::upper_nonzeros() const noexcept {
    std::uint64_t result{};
    for (SparseIndex row = 0; row < dimension; ++row) {
        result += static_cast<std::uint64_t>(
            lu_row_offsets[static_cast<std::size_t>(row) + 1U] -
            diagonal_positions[static_cast<std::size_t>(row)] - 1);
    }
    return result;
}

std::uint64_t AnalysisPlan::storage_bytes() const noexcept {
    const auto index_count = row_permutation.size() + inverse_row_permutation.size() +
                             column_permutation.size() + inverse_column_permutation.size() +
                             block_offsets.size() + lu_row_offsets.size() +
                             lu_column_indices.size() + diagonal_positions.size() +
                             matrix_to_lu.size() + factor_schedule.level_offsets.size() +
                             factor_schedule.rows.size() +
                             forward_schedule.level_offsets.size() +
                             forward_schedule.rows.size() +
                             backward_schedule.level_offsets.size() +
                             backward_schedule.rows.size();
    return static_cast<std::uint64_t>(index_count) * sizeof(SparseIndex) +
           static_cast<std::uint64_t>(
               row_scale_factors.size() + block_estimated_l_nonzeros.size() +
               block_input_symmetry.size()) *
               sizeof(double) +
           static_cast<std::uint64_t>(structurally_symmetric_blocks.size()) *
               sizeof(std::uint8_t) +
           static_cast<std::uint64_t>(factor_row_updates.size()) *
               sizeof(std::uint64_t) +
           lookup.storage_bytes;
}

void AnalysisPlan::validate_dimensions(const CscMatrix& matrix) const {
    if (dimension != matrix.dimension || input_nonzeros != matrix.nonzeros()) {
        throw std::runtime_error("analysis plan does not match the input matrix");
    }
    const auto expected_dimension = static_cast<std::size_t>(dimension);
    if (row_permutation.size() != expected_dimension ||
        inverse_row_permutation.size() != expected_dimension ||
        column_permutation.size() != expected_dimension ||
        inverse_column_permutation.size() != expected_dimension ||
        row_scale_factors.size() != expected_dimension) {
        throw std::runtime_error("analysis permutation/scaling dimension mismatch");
    }
    if (block_offsets.size() < 2U || block_offsets.front() != 0 ||
        block_offsets.back() != dimension ||
        block_estimated_l_nonzeros.size() + 1U != block_offsets.size() ||
        block_input_symmetry.size() + 1U != block_offsets.size() ||
        structurally_symmetric_blocks.size() + 1U != block_offsets.size()) {
        throw std::runtime_error("invalid BTF metadata dimensions");
    }
    if (lu_row_offsets.size() != expected_dimension + 1U ||
        lu_row_offsets.front() != 0 ||
        lu_row_offsets.back() != static_cast<SparseIndex>(lu_column_indices.size()) ||
        diagonal_positions.size() != expected_dimension ||
        matrix_to_lu.size() != static_cast<std::size_t>(input_nonzeros) ||
        factor_row_updates.size() != expected_dimension) {
        throw std::runtime_error("invalid symbolic LU dimensions");
    }
    const auto validate_schedule_dimensions = [&](const LevelSchedule& schedule) {
        if (schedule.level_offsets.empty() || schedule.level_offsets.front() != 0 ||
            schedule.level_offsets.back() != dimension ||
            schedule.rows.size() != expected_dimension) {
            throw std::runtime_error("invalid level schedule dimensions");
        }
    };
    validate_schedule_dimensions(factor_schedule);
    validate_schedule_dimensions(forward_schedule);
    validate_schedule_dimensions(backward_schedule);
}

void AnalysisPlan::validate(const CscMatrix& matrix) const {
    validate_dimensions(matrix);
    const auto validate_permutation = [&](const std::vector<SparseIndex>& permutation,
                                          const std::vector<SparseIndex>& inverse) {
        for (SparseIndex index = 0; index < dimension; ++index) {
            const auto original = permutation[static_cast<std::size_t>(index)];
            if (original < 0 || original >= dimension ||
                inverse[static_cast<std::size_t>(original)] != index) {
                throw std::runtime_error("analysis permutation is inconsistent");
            }
        }
    };
    validate_permutation(row_permutation, inverse_row_permutation);
    validate_permutation(column_permutation, inverse_column_permutation);
    if (row_scale_factors.size() != static_cast<std::size_t>(dimension) ||
        std::any_of(
            row_scale_factors.begin(), row_scale_factors.end(),
            [](double scale) { return !(scale > 0.0) || !std::isfinite(scale); })) {
        throw std::runtime_error("invalid row scaling factors");
    }

    if (!std::is_sorted(block_offsets.begin(), block_offsets.end())) {
        throw std::runtime_error("invalid BTF block boundaries");
    }
    if (std::any_of(
            block_input_symmetry.begin(), block_input_symmetry.end(),
            [](double value) {
                return !std::isfinite(value) || value < -1.0 || value > 1.0;
            })) {
        throw std::runtime_error("invalid BTF input-symmetry metadata");
    }
    if (std::any_of(
            structurally_symmetric_blocks.begin(),
            structurally_symmetric_blocks.end(),
            [](std::uint8_t value) { return value > 1U; })) {
        throw std::runtime_error("invalid BTF structural-symmetry metadata");
    }
    for (SparseIndex row = 0; row < dimension; ++row) {
        const auto begin = lu_row_offsets[static_cast<std::size_t>(row)];
        const auto end = lu_row_offsets[static_cast<std::size_t>(row) + 1U];
        const auto diagonal = diagonal_positions[static_cast<std::size_t>(row)];
        if (begin > diagonal || diagonal >= end ||
            lu_column_indices[static_cast<std::size_t>(diagonal)] != row ||
            !std::is_sorted(
                lu_column_indices.begin() + begin, lu_column_indices.begin() + end) ||
            std::adjacent_find(
                lu_column_indices.begin() + begin, lu_column_indices.begin() + end) !=
                lu_column_indices.begin() + end) {
            throw std::runtime_error("invalid symbolic LU row");
        }
    }
    if (std::accumulate(
            factor_row_updates.begin(), factor_row_updates.end(), std::uint64_t{0}) !=
            symbolic_scalar_updates) {
        throw std::runtime_error("factor row work estimates are inconsistent");
    }
    for (SparseIndex original_column = 0; original_column < dimension; ++original_column) {
        const auto new_column =
            inverse_column_permutation[static_cast<std::size_t>(original_column)];
        for (auto position = matrix.column_offsets[static_cast<std::size_t>(original_column)];
             position < matrix.column_offsets[static_cast<std::size_t>(original_column) + 1U];
             ++position) {
            const auto mapped = matrix_to_lu[static_cast<std::size_t>(position)];
            const auto new_row = inverse_row_permutation[static_cast<std::size_t>(
                matrix.row_indices[static_cast<std::size_t>(position)])];
            if (mapped < lu_row_offsets[static_cast<std::size_t>(new_row)] ||
                mapped >= lu_row_offsets[static_cast<std::size_t>(new_row) + 1U] ||
                lu_column_indices[static_cast<std::size_t>(mapped)] != new_column) {
                throw std::runtime_error("matrix-to-factor mapping is inconsistent");
            }
        }
    }
    validate_schedule(factor_schedule, *this, false);
    validate_schedule(forward_schedule, *this, false);
    validate_schedule(backward_schedule, *this, true);
}

AnalysisPlan CpuSymbolicAnalyzer::analyze(
    const CscMatrix& matrix,
    Profiler& profiler) const {
    AnalysisPlan plan;
    plan.dimension = matrix.dimension;
    plan.input_nonzeros = matrix.nonzeros();

    {
        auto event = profiler.scoped("klu_btf_ordering", EventKind::event);
        profiler.add_attribute("numerical_factorization", "not called");
        profiler.add_attribute("components", "BTF+AMD ordering only");
        klu_common common{};
        if (klu_defaults(&common) == 0) throw std::runtime_error("klu_defaults failed");
        auto* symbolic = klu_analyze(
            matrix.dimension,
            const_cast<SparseIndex*>(matrix.column_offsets.data()),
            const_cast<SparseIndex*>(matrix.row_indices.data()),
            &common);
        if (symbolic == nullptr || common.status != KLU_OK) {
            const auto status = common.status;
            if (symbolic != nullptr) klu_free_symbolic(&symbolic, &common);
            throw std::runtime_error(
                "KLU structural analysis failed with status " + std::to_string(status));
        }
        plan.row_permutation.assign(symbolic->P, symbolic->P + symbolic->n);
        plan.column_permutation.assign(symbolic->Q, symbolic->Q + symbolic->n);
        plan.block_offsets.assign(symbolic->R, symbolic->R + symbolic->nblocks + 1);
        plan.block_estimated_l_nonzeros.assign(
            symbolic->Lnz, symbolic->Lnz + symbolic->nblocks);
        plan.block_input_symmetry.assign(
            static_cast<std::size_t>(symbolic->nblocks), -1.0);
        plan.structurally_symmetric_blocks.assign(
            static_cast<std::size_t>(symbolic->nblocks), std::uint8_t{2});
        std::uint64_t ordering_classified_blocks{};
        for (SparseIndex block = 0; block < symbolic->nblocks; ++block) {
            const auto symmetry = symbolic->Symmetry[block];
            if (symmetry < 0.0 || !std::isfinite(symmetry)) continue;
            plan.block_input_symmetry[static_cast<std::size_t>(block)] = symmetry;
            plan.structurally_symmetric_blocks[static_cast<std::size_t>(block)] =
                symmetry >= 1.0 ? std::uint8_t{1} : std::uint8_t{0};
            ++ordering_classified_blocks;
        }
        plan.ordering_estimated_factor_flops = symbolic->est_flops;
        plan.ordering_estimated_l_nonzeros = symbolic->lnz;
        plan.ordering_estimated_u_nonzeros = symbolic->unz;
        plan.structural_rank = symbolic->structural_rank;
        profiler.add_value("btf_blocks", symbolic->nblocks);
        profiler.add_value("largest_btf_block", symbolic->maxblock);
        profiler.add_value("structural_rank", symbolic->structural_rank);
        profiler.add_value("ordering_estimated_factor_flops", symbolic->est_flops);
        profiler.add_value("ordering_estimated_l_nonzeros", symbolic->lnz);
        profiler.add_value("ordering_estimated_u_nonzeros", symbolic->unz);
        profiler.add_value("btf_off_diagonal_input_nonzeros", symbolic->nzoff);
        profiler.add_value(
            "ordering_classified_block_symmetry", ordering_classified_blocks);
        profiler.add_value(
            "symbolic_symmetry_fallback_blocks",
            symbolic->nblocks - ordering_classified_blocks);
        klu_free_symbolic(&symbolic, &common);
    }

    {
        auto event = profiler.scoped("permutation_build", EventKind::event);
        plan.inverse_row_permutation = inverse_permutation(plan.row_permutation, plan.dimension);
        plan.inverse_column_permutation =
            inverse_permutation(plan.column_permutation, plan.dimension);
    }
    PermutedRowPattern permuted_rows;
    {
        auto event = profiler.scoped("permuted_matrix_build", EventKind::event);
        permuted_rows = build_permuted_rows(matrix, plan);
        profiler.add_attribute(
            "layout", "contiguous CSR built in final column order");
        profiler.add_attribute(
            "scaling", "row maximum fused with CSR row counting");
    }
    {
        auto event = profiler.scoped("symbolic_lu", EventKind::event);
        profiler.add_attribute(
            "strategy",
            "BTF-block elimination-tree reach with exact generic fallback");
        SymbolicPhaseTimings timings;
        append_symbolic_pattern(permuted_rows, plan, timings);
        const auto symmetric_blocks = static_cast<std::uint64_t>(std::count(
            plan.structurally_symmetric_blocks.begin(),
            plan.structurally_symmetric_blocks.end(),
            std::uint8_t{1}));
        profiler.add_value("structurally_symmetric_btf_blocks", symmetric_blocks);
        profiler.add_value(
            "generic_symbolic_btf_blocks",
            plan.structurally_symmetric_blocks.size() - symmetric_blocks);
        profiler.add_value("factor_nonzeros", plan.factor_nonzeros());
        profiler.add_value(
            "factor_index_capacity", plan.lu_column_indices.capacity());
        profiler.add_value(
            "factor_fill_ratio",
            static_cast<double>(plan.factor_nonzeros()) / plan.input_nonzeros);
        profiler.add_value(
            "phase_symmetry_detection_ms", timings.symmetry_detection_ms);
        profiler.add_value(
            "phase_symmetrized_lower_build_ms",
            timings.symmetrized_lower_build_ms);
        profiler.add_value(
            "symmetrized_envelope_btf_blocks", timings.symmetrized_blocks);
        profiler.add_value(
            "phase_elimination_tree_ms", timings.elimination_tree_ms);
        profiler.add_value("phase_lower_reach_ms", timings.lower_reach_ms);
        profiler.add_value("phase_upper_transpose_ms", timings.upper_transpose_ms);
        profiler.add_value(
            "phase_btf_off_diagonal_ms", timings.btf_off_diagonal_ms);
        profiler.add_value(
            "phase_factor_assembly_ms", timings.factor_assembly_ms);
        profiler.add_value(
            "phase_generic_fallback_ms", timings.generic_fallback_ms);
        profiler.add_attribute(
            "symmetrized_envelope_policy",
            "blocks >= 256; AMD symmetry >= 0.98 or predicted envelope <= 6x "
            "diagonal input nnz; A union A^T conservative LU superset");
    }
    {
        auto event = profiler.scoped("matrix_to_lu_build", EventKind::event);
        build_matrix_mapping(permuted_rows, plan);
        profiler.add_attribute("mapping", "linear sorted-row merge");
    }
    permuted_rows = {};
    {
        auto event = profiler.scoped("dependency_schedule_build", EventKind::event);
        build_schedules(plan);
        profiler.add_attribute(
            "work_estimate", "fused with forward dependency scan");
        profiler.add_value("symbolic_scalar_updates", plan.symbolic_scalar_updates);
        profiler.add_value("estimated_factor_flops", plan.estimated_factor_flops);
        profiler.add_value("factor_levels", plan.factor_schedule.levels());
        profiler.add_value("factor_widest_level", plan.factor_schedule.widest_level());
        profiler.add_value("backward_levels", plan.backward_schedule.levels());
        profiler.add_value("backward_widest_level", plan.backward_schedule.widest_level());

        std::uint64_t maximum_row_updates{};
        SparseIndex maximum_work_row{};
        std::uint64_t short_rows{};
        std::uint64_t warp_rows{};
        std::uint64_t block_rows{};
        std::uint64_t heavy_rows{};
        for (SparseIndex row = 0; row < plan.dimension; ++row) {
            const auto work = plan.factor_row_updates[static_cast<std::size_t>(row)];
            if (work > maximum_row_updates) {
                maximum_row_updates = work;
                maximum_work_row = row;
            }
            if (work < 32U) ++short_rows;
            else if (work < 128U) ++warp_rows;
            else if (work < 4096U) ++block_rows;
            else ++heavy_rows;
        }
        std::uint64_t maximum_level_updates{};
        SparseIndex maximum_work_level{};
        for (SparseIndex level = 0; level < plan.factor_schedule.levels(); ++level) {
            const auto begin =
                plan.factor_schedule.level_offsets[static_cast<std::size_t>(level)];
            const auto end =
                plan.factor_schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
            std::uint64_t level_updates{};
            for (auto position = begin; position < end; ++position) {
                const auto row =
                    plan.factor_schedule.rows[static_cast<std::size_t>(position)];
                level_updates +=
                    plan.factor_row_updates[static_cast<std::size_t>(row)];
            }
            if (level_updates > maximum_level_updates) {
                maximum_level_updates = level_updates;
                maximum_work_level = level;
            }
        }
        profiler.add_value("factor_maximum_row_updates", maximum_row_updates);
        profiler.add_value("factor_maximum_work_row", maximum_work_row);
        profiler.add_value("factor_rows_updates_lt_32", short_rows);
        profiler.add_value("factor_rows_updates_32_to_127", warp_rows);
        profiler.add_value("factor_rows_updates_128_to_4095", block_rows);
        profiler.add_value("factor_rows_updates_ge_4096", heavy_rows);
        profiler.add_value("factor_maximum_level_updates", maximum_level_updates);
        profiler.add_value("factor_maximum_work_level", maximum_work_level);
    }
    {
        auto event = profiler.scoped("lookup_build", EventKind::event);
        plan.lookup = {LookupKind::binary_search, 0};
        profiler.add_attribute("lookup", std::string(to_string(plan.lookup.kind)));
    }
    {
        auto event = profiler.scoped("analysis_plan_validation", EventKind::event);
#ifdef NDEBUG
        plan.validate_dimensions(matrix);
        profiler.add_attribute("validation", "production dimension and boundary checks");
#else
        plan.validate(matrix);
        profiler.add_attribute("validation", "exhaustive debug structural checks");
#endif
        profiler.add_value("plan_storage_bytes", plan.storage_bytes());
    }
    return plan;
}

std::string_view to_string(LookupKind kind) noexcept {
    switch (kind) {
        case LookupKind::binary_search: return "binary-search";
    }
    return "unknown";
}

}  // namespace eda_gpu
