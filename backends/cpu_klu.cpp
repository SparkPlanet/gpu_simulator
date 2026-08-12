#include "backend.hpp"

extern "C" {
#include "ngspice/klu.h"
}

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace eda_gpu {
namespace {

class CpuKluBackend final : public Task1Backend {
public:
    CpuKluBackend() {
        if (klu_defaults(&common_) == 0) {
            throw std::runtime_error("klu_defaults failed");
        }
    }

    ~CpuKluBackend() override {
        release_numeric();
        release_symbolic();
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "cpu-klu"; }

    void analyze(const CscMatrix& matrix, BackendContext& context) override {
        release_numeric();
        release_symbolic();
        analyzed_matrix_ = nullptr;

        auto event = context.profiler.scoped("klu_analyze", EventKind::event);
        context.profiler.add_attribute("components", "BTF+AMD/COLAMD+symbolic analysis");
        symbolic_ = klu_analyze(
            matrix.dimension,
            const_cast<SparseIndex*>(matrix.column_offsets.data()),
            const_cast<SparseIndex*>(matrix.row_indices.data()),
            &common_);
        require_ok("klu_analyze", symbolic_ != nullptr);
        analyzed_matrix_ = &matrix;

        context.profiler.add_value("btf_blocks", symbolic_->nblocks);
        context.profiler.add_value("largest_btf_block", symbolic_->maxblock);
        context.profiler.add_value("off_diagonal_nonzeros", symbolic_->nzoff);
        context.profiler.add_value("estimated_l_nonzeros", symbolic_->lnz);
        context.profiler.add_value("estimated_u_nonzeros", symbolic_->unz);
        context.profiler.add_value("estimated_factor_flops", symbolic_->est_flops);
        context.profiler.add_value("klu_memory_peak_bytes", common_.mempeak);

        statistics_.attributes["algorithm"] = "KLU sparse direct LU";
        statistics_.attributes["execution_space"] = "host";
        statistics_.attributes["ordering"] =
            symbolic_->ordering == 0 ? "AMD" :
            symbolic_->ordering == 1 ? "COLAMD" : "other";
        statistics_.attributes["btf"] = symbolic_->do_btf ? "enabled" : "disabled";
        statistics_.attributes["index_type"] = "int32";
        statistics_.attributes["value_type"] = "float64";
        statistics_.attributes["lifecycle"] = "one-shot Task1";
        statistics_.values["input_nonzeros"] = matrix.nonzeros();
        statistics_.values["btf_blocks"] = symbolic_->nblocks;
        statistics_.values["largest_btf_block"] = symbolic_->maxblock;
        statistics_.values["off_diagonal_nonzeros"] = symbolic_->nzoff;
        statistics_.values["symbolic_estimated_l_nonzeros"] = symbolic_->lnz;
        statistics_.values["symbolic_estimated_u_nonzeros"] = symbolic_->unz;
        statistics_.values["symbolic_estimated_factor_flops"] = symbolic_->est_flops;
        statistics_.values["analysis_memory_peak_bytes"] = common_.mempeak;
    }

    void factorize(const CscMatrix& matrix, BackendContext& context) override {
        if (symbolic_ == nullptr || analyzed_matrix_ != &matrix) {
            throw std::runtime_error("KLU factorization requires analysis of the same matrix");
        }
        release_numeric();

        auto event = context.profiler.scoped("klu_numeric_lu", EventKind::event);
        context.profiler.add_attribute("pivoting", "partial pivoting with diagonal preference");
        if (symbolic_->est_flops > 0.0) {
            context.profiler.add_estimated_flops(symbolic_->est_flops);
        }
        numeric_ = klu_factor(
            const_cast<SparseIndex*>(matrix.column_offsets.data()),
            const_cast<SparseIndex*>(matrix.row_indices.data()),
            const_cast<double*>(matrix.values.data()),
            symbolic_,
            &common_);
        require_ok("klu_factor", numeric_ != nullptr);

        const auto factor_nonzeros = static_cast<double>(numeric_->lnz) + numeric_->unz;
        const auto fill_ratio = factor_nonzeros / static_cast<double>(matrix.nonzeros());
        context.profiler.add_value("l_nonzeros", numeric_->lnz);
        context.profiler.add_value("u_nonzeros", numeric_->unz);
        context.profiler.add_value("factor_fill_ratio", fill_ratio);
        context.profiler.add_value("off_diagonal_pivots", common_.noffdiag);
        context.profiler.add_value("factor_reallocations", common_.nrealloc);
        context.profiler.add_value("klu_memory_peak_bytes", common_.mempeak);

        statistics_.values["l_nonzeros"] = numeric_->lnz;
        statistics_.values["u_nonzeros"] = numeric_->unz;
        statistics_.values["factor_nonzeros"] = factor_nonzeros;
        statistics_.values["factor_fill_ratio"] = fill_ratio;
        statistics_.values["off_diagonal_pivots"] = common_.noffdiag;
        statistics_.values["factor_reallocations"] = common_.nrealloc;
        statistics_.values["factor_memory_peak_bytes"] = common_.mempeak;
    }

    [[nodiscard]] std::vector<double> solve(
        const std::vector<double>& right_hand_side,
        BackendContext& context) override {
        if (symbolic_ == nullptr || numeric_ == nullptr) {
            throw std::runtime_error("KLU solve requires completed analysis and factorization");
        }
        if (right_hand_side.size() != static_cast<std::size_t>(symbolic_->n)) {
            throw std::runtime_error("right-hand-side dimension mismatch");
        }

        auto event = context.profiler.scoped("klu_solve", EventKind::event);
        context.profiler.add_attribute(
            "components", "permutation+off-diagonal update+forward solve+backward solve");
        const auto estimated_flops =
            2.0 * static_cast<double>(numeric_->lnz + numeric_->unz);
        context.profiler.add_estimated_flops(estimated_flops);
        auto solution = right_hand_side;
        const auto success = klu_solve(
            symbolic_, numeric_, symbolic_->n, 1, solution.data(), &common_);
        require_ok("klu_solve", success != 0);
        context.profiler.add_value("estimated_triangular_solve_flops", estimated_flops);
        statistics_.values["estimated_triangular_solve_flops"] = estimated_flops;
        return solution;
    }

    [[nodiscard]] BackendStatistics statistics() const override {
        return statistics_;
    }

private:
    void require_ok(const char* operation, bool object_valid) const {
        if (!object_valid || common_.status != KLU_OK) {
            throw std::runtime_error(
                std::string(operation) + " failed with KLU status " +
                std::to_string(common_.status));
        }
    }

    void release_numeric() noexcept {
        if (numeric_ != nullptr) klu_free_numeric(&numeric_, &common_);
    }

    void release_symbolic() noexcept {
        if (symbolic_ != nullptr) klu_free_symbolic(&symbolic_, &common_);
    }

    klu_common common_{};
    klu_symbolic* symbolic_{};
    klu_numeric* numeric_{};
    const CscMatrix* analyzed_matrix_{};
    BackendStatistics statistics_;
};

}  // namespace

std::unique_ptr<Task1Backend> make_cpu_klu_backend() {
    return std::make_unique<CpuKluBackend>();
}

}  // namespace eda_gpu
