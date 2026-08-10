#include "solver.hpp"

#include <stdexcept>

namespace eda_gpu::core {

std::unique_ptr<LinearSolver> make_klu_solver();

std::vector<SolverDescriptor> solver_descriptors() {
    return {
        {"klu", true, "CPU sparse LU from the vendored ngspice KLU source"},
        {"cuda", false, "adapter reserved; CUDA solver/operators are not implemented"},
        {"metax", false, "adapter reserved; MetaX solver/operators are not implemented"},
    };
}

std::unique_ptr<LinearSolver> create_solver(std::string_view name) {
    if (name == "klu") {
        return make_klu_solver();
    }
    for (const auto& descriptor : solver_descriptors()) {
        if (descriptor.name == name) {
            throw std::runtime_error("solver '" + std::string(name) + "' is unavailable: " +
                                     descriptor.detail);
        }
    }
    throw std::runtime_error("unknown solver '" + std::string(name) + "'");
}

}  // namespace eda_gpu::core
