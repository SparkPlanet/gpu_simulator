#include "solver.hpp"

#include <stdexcept>

namespace eda_gpu::core {

std::unique_ptr<LinearSolver> make_klu_solver();
#ifdef EDA_GPU_HAS_CUDA
std::unique_ptr<LinearSolver> make_cuda_custom_solver();
bool cuda_custom_solver_runtime_available(std::string& detail);
std::unique_ptr<LinearSolver> make_cuda_reference_solver();
bool cuda_reference_solver_runtime_available(std::string& detail);
#endif
std::vector<SolverDescriptor> solver_descriptors() {
#ifdef EDA_GPU_HAS_CUDA
    std::string cuda_detail;
    const auto cuda_available = cuda_custom_solver_runtime_available(cuda_detail);
    if (cuda_available) {
        cuda_detail = "project-owned fixed-pattern CUDA LU on " + cuda_detail;
    }
    std::string cuda_reference_detail;
    const auto cuda_reference_available =
        cuda_reference_solver_runtime_available(cuda_reference_detail);
    if (cuda_reference_available) {
        cuda_reference_detail =
            "deprecated development baseline only; cuSolverRf on " +
            cuda_reference_detail;
    }
#else
    const bool cuda_available = false;
    const std::string cuda_detail =
        "not built; configure with -DEDA_GPU_ENABLE_CUDA=ON or use cuda-release preset";
    const bool cuda_reference_available = false;
    const std::string cuda_reference_detail =
        "not built; configure with -DEDA_GPU_ENABLE_CUDA=ON or use cuda-release preset";
#endif
    return {
        {"klu", true, "CPU sparse LU from the vendored ngspice KLU source"},
        {"cuda", cuda_available, cuda_detail},
        {"cuda-cusolverrf-reference", cuda_reference_available,
         cuda_reference_detail},
        {"metax", false, "adapter reserved; MetaX solver/operators are not implemented"},
    };
}

std::unique_ptr<LinearSolver> create_solver(std::string_view name) {
    if (name == "klu") {
        return make_klu_solver();
    }
#ifdef EDA_GPU_HAS_CUDA
    if (name == "cuda") {
        return make_cuda_custom_solver();
    }
    if (name == "cuda-cusolverrf-reference") {
        return make_cuda_reference_solver();
    }
#endif
    for (const auto& descriptor : solver_descriptors()) {
        if (descriptor.name == name) {
            throw std::runtime_error("solver '" + std::string(name) + "' is unavailable: " +
                                     descriptor.detail);
        }
    }
    throw std::runtime_error("unknown solver '" + std::string(name) + "'");
}

}  // namespace eda_gpu::core
