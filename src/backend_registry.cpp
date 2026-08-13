#include "backend.hpp"

#include <stdexcept>
#include <string>

namespace eda_gpu {

std::unique_ptr<Task1Backend> make_cpu_klu_backend();
std::unique_ptr<Task1Backend> make_reference_lu_backend();
#ifdef EDA_GPU_HAS_CUDA
std::unique_ptr<Task1Backend> make_cuda_lu_backend();
std::unique_ptr<Task1Backend> make_cuda_workspace_lu_backend();
std::unique_ptr<Task1Backend> make_cuda_right_looking_lu_backend();
bool cuda_runtime_available(std::string& detail) noexcept;
#endif

std::vector<BackendDescriptor> task1_backends() {
#ifdef EDA_GPU_HAS_CUDA
    std::string cuda_detail;
    const auto cuda_available = cuda_runtime_available(cuda_detail);
#else
    const bool cuda_available = false;
    const std::string cuda_detail = "not built; use the cuda-debug or cuda-release preset";
#endif
    return {
        {"cpu-klu", true, "CPU baseline using the vendored ngspice KLU"},
        {"reference-lu", true,
         "CPU reference for the fixed-pattern algorithm that CUDA will consume"},
        {"cuda-lu", cuda_available, cuda_detail},
        {"cuda-workspace-lu", cuda_available, cuda_detail},
        {"cuda-right-looking-lu", cuda_available, cuda_detail},
    };
}

std::unique_ptr<Task1Backend> create_task1_backend(std::string_view name) {
    if (name == "cpu-klu" || name == "klu") return make_cpu_klu_backend();
    if (name == "reference-lu" || name == "reference") {
        return make_reference_lu_backend();
    }
#ifdef EDA_GPU_HAS_CUDA
    if (name == "cuda-lu" || name == "cuda") return make_cuda_lu_backend();
    if (name == "cuda-workspace-lu" || name == "cuda-workspace") {
        return make_cuda_workspace_lu_backend();
    }
    if (name == "cuda-right-looking-lu" || name == "cuda-right-looking") {
        return make_cuda_right_looking_lu_backend();
    }
#endif
    if (name == "cuda-lu" || name == "cuda" || name == "cuda-workspace-lu" ||
        name == "cuda-workspace" || name == "cuda-right-looking-lu" ||
        name == "cuda-right-looking") {
        for (const auto& descriptor : task1_backends()) {
            const auto canonical_name =
                name == "cuda-workspace-lu" || name == "cuda-workspace"
                    ? "cuda-workspace-lu"
                    : name == "cuda-right-looking-lu" ||
                              name == "cuda-right-looking"
                          ? "cuda-right-looking-lu"
                          : "cuda-lu";
            if (descriptor.name == canonical_name) {
                throw std::runtime_error("CUDA Task 1 backend is unavailable: " +
                                         descriptor.description);
            }
        }
    }
    throw std::runtime_error("unknown Task 1 backend: " + std::string(name));
}

}  // namespace eda_gpu
