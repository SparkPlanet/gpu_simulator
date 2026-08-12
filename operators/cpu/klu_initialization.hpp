#pragma once

#include "core/matrix.hpp"
#include "operators/sparse_operator.hpp"

#include <cstdint>
#include <memory>

namespace eda_gpu::operators::cpu {

struct KluInitializationOptions {
    bool enable_btf{true};
    std::int32_t scaling{2};
    // Build a fixed-pivot L/U pattern from P*A*Q using symbolic reach only.
    // This is the Task 1 path: CPU analysis performs no numerical LU.
    bool static_pivot_symbolic_pattern{};
};

// CPU symbolic analysis for a GPU sparse LU backend. The legacy/reference
// path can also extract a first KLU numerical factorization when requested.
class KluInitialization {
public:
    explicit KluInitialization(KluInitializationOptions options = {});
    ~KluInitialization();

    KluInitialization(const KluInitialization&) = delete;
    KluInitialization& operator=(const KluInitialization&) = delete;

    void analyze(const core::CscMatrix& matrix);
    [[nodiscard]] const InitialFactorization& factorize(
        const core::CscMatrix& matrix);
    void require_pattern(const core::CscMatrix& matrix) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eda_gpu::operators::cpu
