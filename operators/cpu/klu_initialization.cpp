#include "klu_initialization.hpp"

extern "C" {
#include "ngspice/klu.h"
}

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eda_gpu::operators::cpu {
namespace {

void sort_csc_columns(SparseCscData& matrix) {
    std::vector<std::pair<std::int32_t, double>> entries;
    for (std::size_t column = 0; column + 1U < matrix.column_offsets.size(); ++column) {
        const auto begin = matrix.column_offsets[column];
        const auto end = matrix.column_offsets[column + 1U];
        entries.clear();
        entries.reserve(static_cast<std::size_t>(end - begin));
        for (auto position = begin; position < end; ++position) {
            entries.emplace_back(matrix.row_indices[static_cast<std::size_t>(position)],
                                 matrix.values[static_cast<std::size_t>(position)]);
        }
        std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        for (auto position = begin; position < end; ++position) {
            const auto& entry = entries[static_cast<std::size_t>(position - begin)];
            matrix.row_indices[static_cast<std::size_t>(position)] = entry.first;
            matrix.values[static_cast<std::size_t>(position)] = entry.second;
        }
    }
}

[[nodiscard]] InitialFactorization build_static_pivot_symbolic_pattern(
    const core::CscMatrix& matrix,
    const klu_symbolic& symbolic) {
    const auto dimension = matrix.rows;
    InitialFactorization factors;
    factors.dimension = dimension;
    factors.row_permutation.assign(symbolic.P, symbolic.P + dimension);
    factors.column_permutation.assign(symbolic.Q, symbolic.Q + dimension);
    factors.scale_factors.assign(static_cast<std::size_t>(dimension), 1.0);
    factors.block_boundaries = {0, dimension};
    factors.lower.column_offsets.reserve(static_cast<std::size_t>(dimension) + 1U);
    factors.upper.column_offsets.reserve(static_cast<std::size_t>(dimension) + 1U);
    factors.off_diagonal.column_offsets.assign(
        static_cast<std::size_t>(dimension) + 1U, 0);
    factors.lower.column_offsets.push_back(0);
    factors.upper.column_offsets.push_back(0);

    std::vector<std::int32_t> inverse_row_permutation(
        static_cast<std::size_t>(dimension));
    std::vector<bool> seen_rows(static_cast<std::size_t>(dimension), false);
    std::vector<bool> seen_columns(static_cast<std::size_t>(dimension), false);
    for (std::int32_t index = 0; index < dimension; ++index) {
        const auto row = factors.row_permutation[static_cast<std::size_t>(index)];
        const auto column = factors.column_permutation[static_cast<std::size_t>(index)];
        if (row < 0 || row >= dimension || column < 0 || column >= dimension ||
            seen_rows[static_cast<std::size_t>(row)] ||
            seen_columns[static_cast<std::size_t>(column)]) {
            throw std::runtime_error("KLU symbolic analysis returned an invalid P/Q");
        }
        seen_rows[static_cast<std::size_t>(row)] = true;
        seen_columns[static_cast<std::size_t>(column)] = true;
        inverse_row_permutation[static_cast<std::size_t>(row)] = index;
    }

    // Gilbert-Peierls symbolic reach for a fixed pivot sequence. L columns are
    // retained in CSC as the graph explored by later columns. l_pend applies
    // the same symmetric-pruning idea as KLU: after row k becomes pivotal,
    // future DFS walks only the already-pivotal prefix of a prior L column.
    std::vector<std::int32_t> marks(static_cast<std::size_t>(dimension), -1);
    std::vector<std::int32_t> l_pend(static_cast<std::size_t>(dimension), -1);
    std::vector<std::int32_t> graph_stack;
    std::vector<std::int32_t> upper_rows;
    std::vector<std::int32_t> lower_rows;

    auto require_int32_storage = [](std::size_t size, const char* factor_name) {
        if (size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error(
                std::string("static symbolic ") + factor_name +
                " pattern exceeds 32-bit CSC storage");
        }
        return static_cast<std::int32_t>(size);
    };

    for (std::int32_t column = 0; column < dimension; ++column) {
        graph_stack.clear();
        upper_rows.clear();
        lower_rows.clear();

        auto visit_prior_pivot = [&](std::int32_t root) {
            marks[static_cast<std::size_t>(root)] = column;
            upper_rows.push_back(root);
            graph_stack.push_back(root);
            while (!graph_stack.empty()) {
                const auto node = graph_stack.back();
                graph_stack.pop_back();
                const auto begin =
                    factors.lower.column_offsets[static_cast<std::size_t>(node)] + 1;
                const auto stored_end =
                    factors.lower.column_offsets[static_cast<std::size_t>(node) + 1U];
                const auto end = l_pend[static_cast<std::size_t>(node)] >= 0
                                     ? l_pend[static_cast<std::size_t>(node)]
                                     : stored_end;
                for (auto position = begin; position < end; ++position) {
                    const auto row = factors.lower.row_indices[
                        static_cast<std::size_t>(position)];
                    if (marks[static_cast<std::size_t>(row)] == column) continue;
                    marks[static_cast<std::size_t>(row)] = column;
                    if (row < column) {
                        upper_rows.push_back(row);
                        graph_stack.push_back(row);
                    } else {
                        lower_rows.push_back(row);
                    }
                }
            }
        };

        const auto input_column =
            factors.column_permutation[static_cast<std::size_t>(column)];
        const auto input_begin =
            matrix.column_offsets[static_cast<std::size_t>(input_column)];
        const auto input_end =
            matrix.column_offsets[static_cast<std::size_t>(input_column) + 1U];
        for (auto position = input_begin; position < input_end; ++position) {
            const auto input_row =
                matrix.row_indices[static_cast<std::size_t>(position)];
            const auto row = inverse_row_permutation[
                static_cast<std::size_t>(input_row)];
            if (marks[static_cast<std::size_t>(row)] == column) continue;
            if (row < column) {
                visit_prior_pivot(row);
            } else {
                marks[static_cast<std::size_t>(row)] = column;
                lower_rows.push_back(row);
            }
        }

        if (marks[static_cast<std::size_t>(column)] != column) {
            throw std::runtime_error(
                "static-pivot symbolic LU has no structural diagonal at permuted column " +
                std::to_string(column));
        }

        std::sort(upper_rows.begin(), upper_rows.end());
        std::sort(lower_rows.begin(), lower_rows.end());
        for (const auto row : upper_rows) {
            factors.upper.row_indices.push_back(row);
            factors.upper.values.push_back(0.0);
        }
        factors.upper.row_indices.push_back(column);
        factors.upper.values.push_back(0.0);
        for (const auto row : lower_rows) {
            factors.lower.row_indices.push_back(row);
            factors.lower.values.push_back(row == column ? 1.0 : 0.0);
        }
        factors.upper.column_offsets.push_back(require_int32_storage(
            factors.upper.row_indices.size(), "U"));
        factors.lower.column_offsets.push_back(require_int32_storage(
            factors.lower.row_indices.size(), "L"));

        // Prune each prior L(:,j) only if the new pivot row k occurs in it.
        // Rows are sorted, so the pivotal prefix ends immediately after k.
        for (const auto prior_column : upper_rows) {
            auto& pend = l_pend[static_cast<std::size_t>(prior_column)];
            if (pend >= 0) continue;
            const auto begin =
                factors.lower.column_offsets[
                    static_cast<std::size_t>(prior_column)] + 1;
            const auto end = factors.lower.column_offsets[
                static_cast<std::size_t>(prior_column) + 1U];
            const auto first = factors.lower.row_indices.begin() + begin;
            const auto last = factors.lower.row_indices.begin() + end;
            const auto found = std::lower_bound(first, last, column);
            if (found != last && *found == column) {
                pend = static_cast<std::int32_t>(
                    std::distance(factors.lower.row_indices.begin(), found) + 1);
            }
        }
    }
    return factors;
}

}  // namespace

struct KluInitialization::Impl {
    explicit Impl(KluInitializationOptions requested_options) : options(requested_options) {
        if (options.scaling < 0 || options.scaling > 2) {
            throw std::runtime_error("KLU scaling mode must be 0, 1 or 2");
        }
        if (klu_defaults(&common) == 0) {
            throw std::runtime_error("klu_defaults failed for accelerator initialization");
        }
        common.btf = options.enable_btf ? 1 : 0;
        common.scale = options.scaling;
    }

    ~Impl() {
        reset_numeric();
        reset_symbolic();
    }

    void reset_numeric() noexcept {
        if (numeric != nullptr) klu_free_numeric(&numeric, &common);
    }

    void reset_symbolic() noexcept {
        if (symbolic != nullptr) klu_free_symbolic(&symbolic, &common);
    }

    [[noreturn]] void throw_status(const char* operation) const {
        throw std::runtime_error(std::string(operation) + " failed with KLU status " +
                                 std::to_string(common.status));
    }

    KluInitializationOptions options;
    klu_common common{};
    klu_symbolic* symbolic{};
    klu_numeric* numeric{};
    std::int32_t dimension{};
    std::vector<std::int32_t> column_offsets;
    std::vector<std::int32_t> row_indices;
    InitialFactorization factors;
};

KluInitialization::KluInitialization(KluInitializationOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

KluInitialization::~KluInitialization() = default;

void KluInitialization::analyze(const core::CscMatrix& matrix) {
    matrix.validate();
    impl_->reset_numeric();
    impl_->reset_symbolic();
    impl_->factors = {};
    impl_->dimension = matrix.rows;
    impl_->column_offsets = matrix.column_offsets;
    impl_->row_indices = matrix.row_indices;
    impl_->symbolic = klu_analyze(
        impl_->dimension, impl_->column_offsets.data(), impl_->row_indices.data(),
        &impl_->common);
    if (impl_->symbolic == nullptr || impl_->common.status != KLU_OK) {
        impl_->throw_status("klu_analyze");
    }
    if (impl_->options.static_pivot_symbolic_pattern) {
        if (impl_->options.enable_btf || impl_->options.scaling != 0 ||
            impl_->symbolic->nblocks != 1) {
            throw std::runtime_error(
                "static-pivot symbolic LU currently requires one unscaled block");
        }
        impl_->factors = build_static_pivot_symbolic_pattern(matrix, *impl_->symbolic);
    }
}

const InitialFactorization& KluInitialization::factorize(
    const core::CscMatrix& matrix) {
    require_pattern(matrix);
    if (impl_->options.static_pivot_symbolic_pattern) {
        return impl_->factors;
    }
    impl_->reset_numeric();
    impl_->numeric = klu_factor(
        impl_->column_offsets.data(), impl_->row_indices.data(),
        const_cast<double*>(matrix.values.data()), impl_->symbolic, &impl_->common);
    if (impl_->numeric == nullptr || impl_->common.status != KLU_OK) {
        impl_->throw_status("klu_factor");
    }

    auto& factors = impl_->factors;
    factors = {};
    factors.dimension = impl_->dimension;
    factors.lower.column_offsets.resize(static_cast<std::size_t>(impl_->dimension) + 1U);
    factors.lower.row_indices.resize(static_cast<std::size_t>(impl_->numeric->lnz));
    factors.lower.values.resize(static_cast<std::size_t>(impl_->numeric->lnz));
    factors.upper.column_offsets.resize(static_cast<std::size_t>(impl_->dimension) + 1U);
    factors.upper.row_indices.resize(static_cast<std::size_t>(impl_->numeric->unz));
    factors.upper.values.resize(static_cast<std::size_t>(impl_->numeric->unz));
    factors.off_diagonal.column_offsets.resize(
        static_cast<std::size_t>(impl_->dimension) + 1U);
    factors.off_diagonal.row_indices.resize(static_cast<std::size_t>(impl_->numeric->nzoff));
    factors.off_diagonal.values.resize(static_cast<std::size_t>(impl_->numeric->nzoff));
    factors.row_permutation.resize(static_cast<std::size_t>(impl_->dimension));
    factors.column_permutation.resize(static_cast<std::size_t>(impl_->dimension));
    factors.scale_factors.resize(static_cast<std::size_t>(impl_->dimension));
    factors.block_boundaries.resize(
        static_cast<std::size_t>(impl_->symbolic->nblocks) + 1U);

    if (klu_extract(
            impl_->numeric, impl_->symbolic,
            factors.lower.column_offsets.data(), factors.lower.row_indices.data(),
            factors.lower.values.data(), factors.upper.column_offsets.data(),
            factors.upper.row_indices.data(), factors.upper.values.data(),
            factors.off_diagonal.column_offsets.data(),
            factors.off_diagonal.row_indices.empty()
                ? nullptr
                : factors.off_diagonal.row_indices.data(),
            factors.off_diagonal.values.empty() ? nullptr : factors.off_diagonal.values.data(),
            factors.row_permutation.data(), factors.column_permutation.data(),
            factors.scale_factors.data(), factors.block_boundaries.data(),
            &impl_->common) == 0 ||
        impl_->common.status != KLU_OK) {
        impl_->throw_status("klu_extract");
    }

    sort_csc_columns(factors.lower);
    sort_csc_columns(factors.upper);
    sort_csc_columns(factors.off_diagonal);
    return factors;
}

void KluInitialization::require_pattern(const core::CscMatrix& matrix) const {
    if (impl_->symbolic == nullptr) {
        throw std::runtime_error("accelerator analyze() must run before factorization");
    }
    if (matrix.rows != impl_->dimension || matrix.columns != impl_->dimension ||
        matrix.column_offsets != impl_->column_offsets ||
        matrix.row_indices != impl_->row_indices ||
        matrix.values.size() != impl_->row_indices.size()) {
        throw std::runtime_error(
            "accelerator matrix must preserve the analyzed CSC pattern");
    }
}

}  // namespace eda_gpu::operators::cpu
