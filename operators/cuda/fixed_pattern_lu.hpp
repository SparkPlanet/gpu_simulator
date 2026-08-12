#pragma once

#include "operators/sparse_operator.hpp"

#include <memory>
#include <string>

namespace eda_gpu::operators::cuda {

// Competition-owned CUDA implementation. KLU supplies the one-time symbolic
// pattern and pivot permutations; all repeated numerical factorization and
// triangular solves below this boundary are implemented by project kernels.
[[nodiscard]] bool fixed_pattern_lu_available(std::string& detail);
[[nodiscard]] std::unique_ptr<SparseNumericOperator>
make_fixed_pattern_lu_operator();

}  // namespace eda_gpu::operators::cuda
