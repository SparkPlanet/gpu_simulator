#include "gpu_hybrid_solver.hpp"

#include "operators/cuda/cusolverrf_reference.hpp"

#include <memory>
#include <string>

namespace eda_gpu::core {

bool cuda_reference_solver_runtime_available(std::string& detail) {
    return operators::cuda::cusolverrf_reference_available(detail);
}

std::unique_ptr<LinearSolver> make_cuda_reference_solver() {
    // cuSolverRf requires one global, unscaled P*A*Q=L*U input. A future custom
    // operator can choose BTF/scaling independently while reusing the bridge.
    operators::cpu::KluInitializationOptions options;
    options.enable_btf = false;
    options.scaling = 0;
    return bridge::make_gpu_hybrid_solver(
        "cuda-cusolverrf-reference",
        operators::cuda::make_cusolverrf_reference_operator(), options);
}

}  // namespace eda_gpu::core
