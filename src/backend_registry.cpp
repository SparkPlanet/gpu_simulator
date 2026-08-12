#include "backend.hpp"

#include <stdexcept>
#include <string>

namespace eda_gpu {

std::unique_ptr<Task1Backend> make_cpu_klu_backend();
std::unique_ptr<Task1Backend> make_reference_lu_backend();

std::vector<BackendDescriptor> task1_backends() {
    return {
        {"cpu-klu", true, "CPU baseline using the vendored ngspice KLU"},
        {"reference-lu", true,
         "CPU reference for the fixed-pattern algorithm that CUDA will consume"},
    };
}

std::unique_ptr<Task1Backend> create_task1_backend(std::string_view name) {
    if (name == "cpu-klu" || name == "klu") return make_cpu_klu_backend();
    if (name == "reference-lu" || name == "reference") {
        return make_reference_lu_backend();
    }
    throw std::runtime_error("unknown Task 1 backend: " + std::string(name));
}

}  // namespace eda_gpu
