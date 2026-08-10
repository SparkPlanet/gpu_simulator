#include "solver.hpp"

extern "C" {
#include "ngspice/klu.h"
}

#include <stdexcept>
#include <utility>

namespace eda_gpu::core {
namespace {

class KluSolver final : public LinearSolver {
public:
    KluSolver() {
        if (klu_defaults(&common_) == 0) {
            throw std::runtime_error("klu_defaults failed");
        }
    }

    ~KluSolver() override {
        reset_numeric();
        reset_symbolic();
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "klu"; }

    void analyze(const CscMatrix& matrix) override {
        matrix.validate();
        reset_numeric();
        reset_symbolic();
        rows_ = matrix.rows;
        column_offsets_ = matrix.column_offsets;
        row_indices_ = matrix.row_indices;
        symbolic_ = klu_analyze(rows_, column_offsets_.data(), row_indices_.data(), &common_);
        if (symbolic_ == nullptr || common_.status != KLU_OK) {
            throw_status("klu_analyze");
        }
    }

    void factorize(const CscMatrix& matrix) override {
        require_pattern(matrix);
        reset_numeric();
        numeric_ = klu_factor(column_offsets_.data(), row_indices_.data(),
                              const_cast<double*>(matrix.values.data()), symbolic_, &common_);
        if (numeric_ == nullptr || common_.status != KLU_OK) {
            throw_status("klu_factor");
        }
    }

    void refactorize(const CscMatrix& matrix) override {
        require_pattern(matrix);
        if (numeric_ == nullptr) {
            throw std::runtime_error("klu refactorize requires an initial factorization");
        }
        if (klu_refactor(column_offsets_.data(), row_indices_.data(),
                         const_cast<double*>(matrix.values.data()), symbolic_, numeric_,
                         &common_) == 0 || common_.status != KLU_OK) {
            throw_status("klu_refactor");
        }
    }

    [[nodiscard]] std::vector<double> solve(
        const std::vector<double>& right_hand_side) override {
        if (symbolic_ == nullptr || numeric_ == nullptr) {
            throw std::runtime_error("klu solve requires analyzed and factored matrix data");
        }
        if (right_hand_side.size() != static_cast<std::size_t>(rows_)) {
            throw std::runtime_error("right-hand-side dimension mismatch");
        }
        auto solution = right_hand_side;
        if (klu_solve(symbolic_, numeric_, rows_, 1, solution.data(), &common_) == 0 ||
            common_.status != KLU_OK) {
            throw_status("klu_solve");
        }
        return solution;
    }

private:
    void require_pattern(const CscMatrix& matrix) const {
        if (symbolic_ == nullptr) {
            throw std::runtime_error("solver analyze() must be called before factorization");
        }
        if (matrix.rows != rows_ || matrix.columns != rows_ ||
            matrix.column_offsets != column_offsets_ || matrix.row_indices != row_indices_ ||
            matrix.values.size() != row_indices_.size()) {
            throw std::runtime_error("refactorization matrix must preserve the analyzed CSC pattern");
        }
    }

    [[noreturn]] void throw_status(const char* operation) const {
        throw std::runtime_error(std::string(operation) + " failed with KLU status " +
                                 std::to_string(common_.status));
    }

    void reset_numeric() noexcept {
        if (numeric_ != nullptr) {
            klu_free_numeric(&numeric_, &common_);
        }
    }

    void reset_symbolic() noexcept {
        if (symbolic_ != nullptr) {
            klu_free_symbolic(&symbolic_, &common_);
        }
    }

    klu_common common_{};
    klu_symbolic* symbolic_{};
    klu_numeric* numeric_{};
    std::int32_t rows_{};
    std::vector<std::int32_t> column_offsets_;
    std::vector<std::int32_t> row_indices_;
};

}  // namespace

std::unique_ptr<LinearSolver> make_klu_solver() {
    return std::make_unique<KluSolver>();
}

}  // namespace eda_gpu::core
