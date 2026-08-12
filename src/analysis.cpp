#include "analysis.hpp"

extern "C" {
#include "ngspice/klu.h"
}

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
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

[[nodiscard]] std::vector<std::vector<SparseIndex>> build_permuted_rows(
    const CscMatrix& matrix,
    const AnalysisPlan& plan) {
    std::vector<std::vector<SparseIndex>> rows(
        static_cast<std::size_t>(matrix.dimension));
    for (SparseIndex original_column = 0; original_column < matrix.dimension;
         ++original_column) {
        const auto new_column = plan.inverse_column_permutation[
            static_cast<std::size_t>(original_column)];
        for (auto position = matrix.column_offsets[static_cast<std::size_t>(original_column)];
             position < matrix.column_offsets[static_cast<std::size_t>(original_column) + 1U];
             ++position) {
            const auto original_row = matrix.row_indices[static_cast<std::size_t>(position)];
            const auto new_row =
                plan.inverse_row_permutation[static_cast<std::size_t>(original_row)];
            rows[static_cast<std::size_t>(new_row)].push_back(new_column);
        }
    }
    for (SparseIndex row = 0; row < matrix.dimension; ++row) {
        auto& columns = rows[static_cast<std::size_t>(row)];
        std::sort(columns.begin(), columns.end());
        if (std::adjacent_find(columns.begin(), columns.end()) != columns.end()) {
            throw std::runtime_error("permutation unexpectedly created duplicate entries");
        }
        const auto diagonal = std::lower_bound(columns.begin(), columns.end(), row);
        if (diagonal == columns.end() || *diagonal != row) columns.insert(diagonal, row);
    }
    return rows;
}

void append_symbolic_pattern(
    const std::vector<std::vector<SparseIndex>>& permuted_rows,
    AnalysisPlan& plan) {
    const auto dimension = plan.dimension;
    plan.lu_row_offsets.assign(static_cast<std::size_t>(dimension) + 1U, 0);
    plan.diagonal_positions.assign(static_cast<std::size_t>(dimension), -1);
    plan.lu_column_indices.reserve(
        static_cast<std::size_t>(std::max<SparseIndex>(plan.input_nonzeros, dimension)));

    std::vector<SparseIndex> marker(static_cast<std::size_t>(dimension), -1);
    std::vector<SparseIndex> row_columns;
    for (SparseIndex row = 0; row < dimension; ++row) {
        row_columns = permuted_rows[static_cast<std::size_t>(row)];
        std::priority_queue<
            SparseIndex,
            std::vector<SparseIndex>,
            std::greater<SparseIndex>> lower_frontier;
        for (const auto column : row_columns) {
            marker[static_cast<std::size_t>(column)] = row;
            if (column < row) lower_frontier.push(column);
        }

        while (!lower_frontier.empty()) {
            const auto pivot_row = lower_frontier.top();
            lower_frontier.pop();
            const auto upper_begin = static_cast<std::size_t>(
                plan.diagonal_positions[static_cast<std::size_t>(pivot_row)] + 1);
            const auto upper_end = static_cast<std::size_t>(
                plan.lu_row_offsets[static_cast<std::size_t>(pivot_row) + 1U]);
            for (auto position = upper_begin; position < upper_end; ++position) {
                const auto column = plan.lu_column_indices[position];
                if (marker[static_cast<std::size_t>(column)] == row) continue;
                marker[static_cast<std::size_t>(column)] = row;
                row_columns.push_back(column);
                if (column < row) lower_frontier.push(column);
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

[[nodiscard]] SparseIndex find_factor_position(
    const AnalysisPlan& plan,
    SparseIndex row,
    SparseIndex column) {
    const auto begin = plan.lu_column_indices.begin() +
                       plan.lu_row_offsets[static_cast<std::size_t>(row)];
    const auto end = plan.lu_column_indices.begin() +
                     plan.lu_row_offsets[static_cast<std::size_t>(row) + 1U];
    const auto found = std::lower_bound(begin, end, column);
    if (found == end || *found != column) {
        throw std::logic_error("input entry is absent from the symbolic LU pattern");
    }
    return static_cast<SparseIndex>(found - plan.lu_column_indices.begin());
}

void build_matrix_mapping(const CscMatrix& matrix, AnalysisPlan& plan) {
    plan.matrix_to_lu.assign(static_cast<std::size_t>(matrix.nonzeros()), -1);
    for (SparseIndex original_column = 0; original_column < matrix.dimension;
         ++original_column) {
        const auto new_column = plan.inverse_column_permutation[
            static_cast<std::size_t>(original_column)];
        for (auto position = matrix.column_offsets[static_cast<std::size_t>(original_column)];
             position < matrix.column_offsets[static_cast<std::size_t>(original_column) + 1U];
             ++position) {
            const auto original_row = matrix.row_indices[static_cast<std::size_t>(position)];
            const auto new_row =
                plan.inverse_row_permutation[static_cast<std::size_t>(original_row)];
            plan.matrix_to_lu[static_cast<std::size_t>(position)] =
                find_factor_position(plan, new_row, new_column);
        }
    }
}

void build_row_scaling(const CscMatrix& matrix, AnalysisPlan& plan) {
    std::vector<double> original_scale(static_cast<std::size_t>(matrix.dimension), 0.0);
    for (std::size_t position = 0; position < matrix.values.size(); ++position) {
        const auto row = matrix.row_indices[position];
        original_scale[static_cast<std::size_t>(row)] = std::max(
            original_scale[static_cast<std::size_t>(row)], std::abs(matrix.values[position]));
    }
    plan.row_scale_factors.resize(static_cast<std::size_t>(matrix.dimension));
    for (SparseIndex new_row = 0; new_row < matrix.dimension; ++new_row) {
        const auto original_row = plan.row_permutation[static_cast<std::size_t>(new_row)];
        const auto scale = original_scale[static_cast<std::size_t>(original_row)];
        plan.row_scale_factors[static_cast<std::size_t>(new_row)] = scale > 0.0 ? scale : 1.0;
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
    for (SparseIndex row = 0; row < plan.dimension; ++row) {
        SparseIndex level{};
        const auto diagonal = plan.diagonal_positions[static_cast<std::size_t>(row)];
        for (auto position = plan.lu_row_offsets[static_cast<std::size_t>(row)];
             position < diagonal; ++position) {
            const auto dependency = plan.lu_column_indices[static_cast<std::size_t>(position)];
            level = std::max(
                level,
                static_cast<SparseIndex>(
                    forward_levels[static_cast<std::size_t>(dependency)] + 1));
        }
        forward_levels[static_cast<std::size_t>(row)] = level;
    }
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

void compute_work_estimate(AnalysisPlan& plan) {
    std::uint64_t updates{};
    std::uint64_t lower{};
    for (SparseIndex row = 0; row < plan.dimension; ++row) {
        const auto begin = plan.lu_row_offsets[static_cast<std::size_t>(row)];
        const auto diagonal = plan.diagonal_positions[static_cast<std::size_t>(row)];
        lower += static_cast<std::uint64_t>(diagonal - begin);
        for (auto position = begin; position < diagonal; ++position) {
            const auto pivot = plan.lu_column_indices[static_cast<std::size_t>(position)];
            const auto pivot_upper =
                plan.lu_row_offsets[static_cast<std::size_t>(pivot) + 1U] -
                plan.diagonal_positions[static_cast<std::size_t>(pivot)] - 1;
            updates += static_cast<std::uint64_t>(pivot_upper);
        }
    }
    plan.symbolic_scalar_updates = updates;
    plan.estimated_factor_flops = static_cast<double>(lower) + 2.0 * updates;
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
           static_cast<std::uint64_t>(row_scale_factors.size()) * sizeof(double) +
           lookup.storage_bytes;
}

void AnalysisPlan::validate(const CscMatrix& matrix) const {
    if (dimension != matrix.dimension || input_nonzeros != matrix.nonzeros()) {
        throw std::runtime_error("analysis plan does not match the input matrix");
    }
    const auto validate_permutation = [&](const std::vector<SparseIndex>& permutation,
                                          const std::vector<SparseIndex>& inverse) {
        if (permutation.size() != static_cast<std::size_t>(dimension) ||
            inverse.size() != static_cast<std::size_t>(dimension)) {
            throw std::runtime_error("analysis permutation size mismatch");
        }
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

    if (block_offsets.size() < 2U || block_offsets.front() != 0 ||
        block_offsets.back() != dimension ||
        !std::is_sorted(block_offsets.begin(), block_offsets.end())) {
        throw std::runtime_error("invalid BTF block boundaries");
    }
    if (lu_row_offsets.size() != static_cast<std::size_t>(dimension) + 1U ||
        lu_row_offsets.front() != 0 ||
        lu_row_offsets.back() != static_cast<SparseIndex>(lu_column_indices.size()) ||
        diagonal_positions.size() != static_cast<std::size_t>(dimension)) {
        throw std::runtime_error("invalid symbolic LU dimensions");
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
    if (matrix_to_lu.size() != static_cast<std::size_t>(input_nonzeros)) {
        throw std::runtime_error("matrix-to-factor mapping size mismatch");
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
        plan.ordering_estimated_factor_flops = symbolic->est_flops;
        plan.ordering_estimated_l_nonzeros = symbolic->lnz;
        plan.ordering_estimated_u_nonzeros = symbolic->unz;
        plan.structural_rank = symbolic->structural_rank;
        profiler.add_value("btf_blocks", symbolic->nblocks);
        profiler.add_value("largest_btf_block", symbolic->maxblock);
        profiler.add_value("structural_rank", symbolic->structural_rank);
        profiler.add_value("ordering_estimated_factor_flops", symbolic->est_flops);
        klu_free_symbolic(&symbolic, &common);
    }

    {
        auto event = profiler.scoped("permutation_build", EventKind::event);
        plan.inverse_row_permutation = inverse_permutation(plan.row_permutation, plan.dimension);
        plan.inverse_column_permutation =
            inverse_permutation(plan.column_permutation, plan.dimension);
    }
    {
        auto event = profiler.scoped("row_scaling_build", EventKind::event);
        build_row_scaling(matrix, plan);
        profiler.add_attribute("scaling", "row maximum absolute value");
    }

    std::vector<std::vector<SparseIndex>> permuted_rows;
    {
        auto event = profiler.scoped("permuted_matrix_build", EventKind::event);
        permuted_rows = build_permuted_rows(matrix, plan);
    }
    {
        auto event = profiler.scoped("symbolic_lu", EventKind::event);
        append_symbolic_pattern(permuted_rows, plan);
        compute_work_estimate(plan);
        profiler.add_value("factor_nonzeros", plan.factor_nonzeros());
        profiler.add_value(
            "factor_fill_ratio",
            static_cast<double>(plan.factor_nonzeros()) / plan.input_nonzeros);
        profiler.add_value("symbolic_scalar_updates", plan.symbolic_scalar_updates);
        profiler.add_value("estimated_factor_flops", plan.estimated_factor_flops);
    }
    permuted_rows.clear();
    permuted_rows.shrink_to_fit();

    {
        auto event = profiler.scoped("matrix_to_lu_build", EventKind::event);
        build_matrix_mapping(matrix, plan);
    }
    {
        auto event = profiler.scoped("dependency_schedule_build", EventKind::event);
        build_schedules(plan);
        profiler.add_value("factor_levels", plan.factor_schedule.levels());
        profiler.add_value("factor_widest_level", plan.factor_schedule.widest_level());
        profiler.add_value("backward_levels", plan.backward_schedule.levels());
        profiler.add_value("backward_widest_level", plan.backward_schedule.widest_level());
    }
    {
        auto event = profiler.scoped("lookup_build", EventKind::event);
        plan.lookup = {LookupKind::binary_search, 0};
        profiler.add_attribute("lookup", std::string(to_string(plan.lookup.kind)));
    }
    {
        auto event = profiler.scoped("analysis_plan_validation", EventKind::event);
        plan.validate(matrix);
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
