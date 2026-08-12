#pragma once

#include "operators/sparse_operator.hpp"

#include <memory>
#include <string>

namespace eda_gpu::operators::cuda {

// This backend is a development/reference baseline only. Competition rules do
// not permit submitting cuSolverRf as the team's sparse direct solver.
[[nodiscard]] bool cusolverrf_reference_available(std::string& detail);
[[nodiscard]] std::unique_ptr<SparseNumericOperator>
make_cusolverrf_reference_operator();

}  // namespace eda_gpu::operators::cuda
