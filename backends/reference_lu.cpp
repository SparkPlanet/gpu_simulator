#include "analysis.hpp"
#include "backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace eda_gpu {
namespace {

[[nodiscard]] SparseIndex lookup_factor_position(
    const AnalysisPlan& plan,
    SparseIndex row,
    SparseIndex column) {
    const auto row_begin = plan.lu_row_offsets[static_cast<std::size_t>(row)];
    const auto row_end = plan.lu_row_offsets[static_cast<std::size_t>(row) + 1U];
    const auto begin = plan.lu_column_indices.begin() + row_begin;
    const auto end = plan.lu_column_indices.begin() + row_end;
    const auto found = std::lower_bound(begin, end, column);
    if (found == end || *found != column) {
        throw std::logic_error("numerical LU update is absent from the symbolic pattern");
    }
    return static_cast<SparseIndex>(found - plan.lu_column_indices.begin());
}

class ReferenceLuBackend final : public Task1Backend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "reference-lu";
    }

    void analyze(const CscMatrix& matrix, BackendContext& context) override {
        matrix_ = nullptr;
        factor_values_.clear();
        plan_ = analyzer_.analyze(matrix, context.profiler);
        matrix_ = &matrix;

        statistics_.attributes["algorithm"] = "fixed-pivot row-oriented sparse LU";
        statistics_.attributes["analysis"] =
            "CPU KLU BTF+AMD + blockwise elimination-tree/generic symbolic";
        statistics_.attributes["execution_space"] = "host reference";
        statistics_.attributes["factor_storage"] = "combined CSR L/U";
        statistics_.attributes["lookup"] = std::string(to_string(plan_.lookup.kind));
        statistics_.attributes["pivoting"] = "static diagonal pivot";
        statistics_.attributes["row_scaling"] = "maximum absolute value";
        statistics_.attributes["lifecycle"] = "one-shot Task1";
        statistics_.values["input_nonzeros"] = plan_.input_nonzeros;
        statistics_.values["factor_nonzeros_combined_diagonal"] = plan_.factor_nonzeros();
        statistics_.values["lower_nonzeros_excluding_diagonal"] = plan_.lower_nonzeros();
        statistics_.values["upper_nonzeros_excluding_diagonal"] = plan_.upper_nonzeros();
        statistics_.values["factor_fill_ratio"] =
            static_cast<double>(plan_.factor_nonzeros()) / plan_.input_nonzeros;
        statistics_.values["analysis_plan_bytes"] = plan_.storage_bytes();
        statistics_.values["btf_blocks"] = plan_.block_offsets.size() - 1U;
        statistics_.values["structurally_symmetric_btf_blocks"] = std::count(
            plan_.structurally_symmetric_blocks.begin(),
            plan_.structurally_symmetric_blocks.end(),
            std::uint8_t{1});
        SparseIndex largest_block{};
        for (std::size_t block = 0; block + 1U < plan_.block_offsets.size(); ++block) {
            largest_block = std::max(
                largest_block, plan_.block_offsets[block + 1U] - plan_.block_offsets[block]);
        }
        statistics_.values["largest_btf_block"] = largest_block;
        statistics_.values["factor_levels"] = plan_.factor_schedule.levels();
        statistics_.values["factor_widest_level"] = plan_.factor_schedule.widest_level();
        statistics_.values["backward_levels"] = plan_.backward_schedule.levels();
        statistics_.values["backward_widest_level"] =
            plan_.backward_schedule.widest_level();
        statistics_.values["symbolic_scalar_updates"] = plan_.symbolic_scalar_updates;
        statistics_.values["estimated_factor_flops"] = plan_.estimated_factor_flops;
        statistics_.values["ordering_estimated_factor_flops"] =
            plan_.ordering_estimated_factor_flops;
    }

    void factorize(const CscMatrix& matrix, BackendContext& context) override {
        if (matrix_ != &matrix || plan_.dimension != matrix.dimension) {
            throw std::runtime_error(
                "reference LU requires analysis of the same matrix instance");
        }
        {
            auto event = context.profiler.scoped("lu_value_initialize", EventKind::event);
            factor_values_.assign(static_cast<std::size_t>(plan_.factor_nonzeros()), 0.0);
            for (SparseIndex original_column = 0; original_column < matrix.dimension;
                 ++original_column) {
                for (auto position =
                         matrix.column_offsets[static_cast<std::size_t>(original_column)];
                     position <
                         matrix.column_offsets[static_cast<std::size_t>(original_column) + 1U];
                     ++position) {
                    const auto original_row =
                        matrix.row_indices[static_cast<std::size_t>(position)];
                    const auto new_row = plan_.inverse_row_permutation[
                        static_cast<std::size_t>(original_row)];
                    const auto destination =
                        plan_.matrix_to_lu[static_cast<std::size_t>(position)];
                    factor_values_[static_cast<std::size_t>(destination)] +=
                        matrix.values[static_cast<std::size_t>(position)] /
                        plan_.row_scale_factors[static_cast<std::size_t>(new_row)];
                }
            }
            context.profiler.add_value(
                "factor_value_bytes", factor_values_.size() * sizeof(double));
        }

        auto event = context.profiler.scoped("reference_numeric_lu", EventKind::event);
        context.profiler.add_attribute("arithmetic", "row-oriented fixed-pivot LU");
        context.profiler.add_attribute("schedule", "CPU execution of factor levels");
        context.profiler.add_estimated_flops(plan_.estimated_factor_flops);
        minimum_absolute_pivot_ = std::numeric_limits<double>::infinity();
        const auto pivot_threshold = 64.0 * std::numeric_limits<double>::epsilon();

        for (SparseIndex level = 0; level < plan_.factor_schedule.levels(); ++level) {
            const auto level_begin =
                plan_.factor_schedule.level_offsets[static_cast<std::size_t>(level)];
            const auto level_end =
                plan_.factor_schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
            for (auto scheduled = level_begin; scheduled < level_end; ++scheduled) {
                const auto row =
                    plan_.factor_schedule.rows[static_cast<std::size_t>(scheduled)];
                const auto row_begin = plan_.lu_row_offsets[static_cast<std::size_t>(row)];
                const auto diagonal = plan_.diagonal_positions[static_cast<std::size_t>(row)];
                for (auto lower_position = row_begin; lower_position < diagonal;
                     ++lower_position) {
                    const auto pivot_row =
                        plan_.lu_column_indices[static_cast<std::size_t>(lower_position)];
                    const auto pivot_position =
                        plan_.diagonal_positions[static_cast<std::size_t>(pivot_row)];
                    const auto pivot =
                        factor_values_[static_cast<std::size_t>(pivot_position)];
                    if (!std::isfinite(pivot) || std::abs(pivot) <= pivot_threshold) {
                        throw std::runtime_error(
                            "fixed-pivot LU encountered a zero/tiny pivot at permuted row " +
                            std::to_string(pivot_row));
                    }
                    const auto multiplier =
                        factor_values_[static_cast<std::size_t>(lower_position)] / pivot;
                    factor_values_[static_cast<std::size_t>(lower_position)] = multiplier;
                    const auto upper_begin = pivot_position + 1;
                    const auto upper_end =
                        plan_.lu_row_offsets[static_cast<std::size_t>(pivot_row) + 1U];
                    for (auto upper_position = upper_begin; upper_position < upper_end;
                         ++upper_position) {
                        const auto column =
                            plan_.lu_column_indices[static_cast<std::size_t>(upper_position)];
                        const auto destination =
                            lookup_factor_position(plan_, row, column);
                        factor_values_[static_cast<std::size_t>(destination)] -=
                            multiplier *
                            factor_values_[static_cast<std::size_t>(upper_position)];
                    }
                }
                const auto pivot = factor_values_[static_cast<std::size_t>(diagonal)];
                if (!std::isfinite(pivot) || std::abs(pivot) <= pivot_threshold) {
                    throw std::runtime_error(
                        "fixed-pivot LU encountered a zero/tiny pivot at permuted row " +
                        std::to_string(row));
                }
                minimum_absolute_pivot_ =
                    std::min(minimum_absolute_pivot_, std::abs(pivot));
            }
        }
        context.profiler.add_value("minimum_absolute_pivot", minimum_absolute_pivot_);
        statistics_.values["minimum_absolute_pivot"] = minimum_absolute_pivot_;
        statistics_.values["factor_value_bytes"] = factor_values_.size() * sizeof(double);
    }

    [[nodiscard]] std::vector<double> solve(
        const std::vector<double>& right_hand_side,
        BackendContext& context) override {
        if (factor_values_.size() != static_cast<std::size_t>(plan_.factor_nonzeros())) {
            throw std::runtime_error("reference LU solve requires a numerical factor");
        }
        std::vector<double> permuted_rhs(static_cast<std::size_t>(plan_.dimension));
        {
            auto event = context.profiler.scoped("rhs_permutation_and_scaling", EventKind::event);
            for (SparseIndex row = 0; row < plan_.dimension; ++row) {
                const auto original_row =
                    plan_.row_permutation[static_cast<std::size_t>(row)];
                permuted_rhs[static_cast<std::size_t>(row)] =
                    right_hand_side[static_cast<std::size_t>(original_row)] /
                    plan_.row_scale_factors[static_cast<std::size_t>(row)];
            }
        }

        std::vector<double> intermediate(static_cast<std::size_t>(plan_.dimension));
        {
            auto event = context.profiler.scoped("reference_forward_solve", EventKind::event);
            context.profiler.add_estimated_flops(2.0 * plan_.lower_nonzeros());
            for (SparseIndex level = 0; level < plan_.forward_schedule.levels(); ++level) {
                const auto level_begin =
                    plan_.forward_schedule.level_offsets[static_cast<std::size_t>(level)];
                const auto level_end = plan_.forward_schedule.level_offsets[
                    static_cast<std::size_t>(level) + 1U];
                for (auto scheduled = level_begin; scheduled < level_end; ++scheduled) {
                    const auto row =
                        plan_.forward_schedule.rows[static_cast<std::size_t>(scheduled)];
                    auto value = permuted_rhs[static_cast<std::size_t>(row)];
                    const auto begin = plan_.lu_row_offsets[static_cast<std::size_t>(row)];
                    const auto diagonal =
                        plan_.diagonal_positions[static_cast<std::size_t>(row)];
                    for (auto position = begin; position < diagonal; ++position) {
                        const auto column =
                            plan_.lu_column_indices[static_cast<std::size_t>(position)];
                        value -= factor_values_[static_cast<std::size_t>(position)] *
                                 intermediate[static_cast<std::size_t>(column)];
                    }
                    intermediate[static_cast<std::size_t>(row)] = value;
                }
            }
        }

        std::vector<double> permuted_solution(static_cast<std::size_t>(plan_.dimension));
        {
            auto event = context.profiler.scoped("reference_backward_solve", EventKind::event);
            context.profiler.add_estimated_flops(
                2.0 * plan_.upper_nonzeros() + plan_.dimension);
            for (SparseIndex level = 0; level < plan_.backward_schedule.levels(); ++level) {
                const auto level_begin =
                    plan_.backward_schedule.level_offsets[static_cast<std::size_t>(level)];
                const auto level_end = plan_.backward_schedule.level_offsets[
                    static_cast<std::size_t>(level) + 1U];
                for (auto scheduled = level_begin; scheduled < level_end; ++scheduled) {
                    const auto row =
                        plan_.backward_schedule.rows[static_cast<std::size_t>(scheduled)];
                    auto value = intermediate[static_cast<std::size_t>(row)];
                    const auto diagonal =
                        plan_.diagonal_positions[static_cast<std::size_t>(row)];
                    const auto end =
                        plan_.lu_row_offsets[static_cast<std::size_t>(row) + 1U];
                    for (auto position = diagonal + 1; position < end; ++position) {
                        const auto column =
                            plan_.lu_column_indices[static_cast<std::size_t>(position)];
                        value -= factor_values_[static_cast<std::size_t>(position)] *
                                 permuted_solution[static_cast<std::size_t>(column)];
                    }
                    permuted_solution[static_cast<std::size_t>(row)] =
                        value / factor_values_[static_cast<std::size_t>(diagonal)];
                }
            }
        }

        std::vector<double> solution(static_cast<std::size_t>(plan_.dimension));
        {
            auto event = context.profiler.scoped("solution_inverse_permutation", EventKind::event);
            for (SparseIndex column = 0; column < plan_.dimension; ++column) {
                const auto original_column =
                    plan_.column_permutation[static_cast<std::size_t>(column)];
                solution[static_cast<std::size_t>(original_column)] =
                    permuted_solution[static_cast<std::size_t>(column)];
            }
        }
        statistics_.values["estimated_forward_solve_flops"] =
            2.0 * plan_.lower_nonzeros();
        statistics_.values["estimated_backward_solve_flops"] =
            2.0 * plan_.upper_nonzeros() + plan_.dimension;
        return solution;
    }

    [[nodiscard]] BackendStatistics statistics() const override {
        return statistics_;
    }

private:
    CpuSymbolicAnalyzer analyzer_;
    AnalysisPlan plan_;
    const CscMatrix* matrix_{};
    std::vector<double> factor_values_;
    double minimum_absolute_pivot_{};
    BackendStatistics statistics_;
};

}  // namespace

std::unique_ptr<Task1Backend> make_reference_lu_backend() {
    return std::make_unique<ReferenceLuBackend>();
}

}  // namespace eda_gpu
