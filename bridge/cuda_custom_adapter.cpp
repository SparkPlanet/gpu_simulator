#include "gpu_hybrid_solver.hpp"

#include "operators/cuda/fixed_pattern_lu.hpp"

#include <memory>
#include <string>

namespace eda_gpu::core {

bool cuda_custom_solver_runtime_available(std::string& detail) {
    return operators::cuda::fixed_pattern_lu_available(detail);
}

std::unique_ptr<LinearSolver> make_cuda_custom_solver() {
    // Task 1 uses CPU AMD + symbolic fill construction, followed by the first
    // numerical factorization on the GPU. No CPU numerical LU is performed.
    // BTF/scaling remain later symbolic-layer extensions.
    operators::cpu::KluInitializationOptions options;
    options.enable_btf = false;
    options.scaling = 0;
    options.static_pivot_symbolic_pattern = true;
    return bridge::make_gpu_hybrid_solver(
        "cuda", operators::cuda::make_fixed_pattern_lu_operator(), options);
}

}  // namespace eda_gpu::core
