#include "eda_gpu/task1.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace eda_gpu {

Task1Pipeline::Task1Pipeline(std::string backend_name)
    : backend_name_(std::move(backend_name)) {
    if (backend_name_.empty()) throw std::runtime_error("Task 1 backend name is empty");
}

Task1Result Task1Pipeline::run(const Task1Request& request) const {
    Profiler profiler;
    std::vector<double> solution;
    BackendStatistics backend_statistics;
    ExecutorStatistics host_statistics;
    std::string actual_backend;

    {
        auto task = profiler.scoped("task1", EventKind::stage);
        profiler.add_attribute("lifecycle", "one-shot");
        profiler.add_attribute("result_location", "host");

        {
            auto validation = profiler.scoped("input_validation", EventKind::event);
            request.matrix.validate();
            if (request.right_hand_side.size() !=
                static_cast<std::size_t>(request.matrix.dimension)) {
                throw std::runtime_error("right-hand-side dimension does not match the matrix");
            }
            for (const auto value : request.right_hand_side) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("right-hand side contains a non-finite value");
                }
            }
            profiler.add_value("dimension", request.matrix.dimension);
            profiler.add_value("matrix_nonzeros", request.matrix.nonzeros());
        }

        std::unique_ptr<Executor> host_executor;
        {
            auto setup = profiler.scoped("host_executor_create", EventKind::event);
            host_executor = make_host_executor(profiler);
            profiler.add_attribute("executor", std::string(host_executor->name()));
        }

        std::unique_ptr<Task1Backend> backend;
        {
            auto setup = profiler.scoped("backend_create", EventKind::event);
            backend = create_task1_backend(backend_name_);
            actual_backend = std::string(backend->name());
            profiler.add_attribute("backend", actual_backend);
        }

        BackendContext context{profiler, *host_executor};
        {
            auto analysis = profiler.scoped("analyze", EventKind::stage);
            backend->analyze(request.matrix, context);
        }
        {
            auto factorization = profiler.scoped("factorize", EventKind::stage);
            backend->factorize(request.matrix, context);
        }
        {
            auto triangular_solve = profiler.scoped("triangular_solve", EventKind::stage);
            solution = backend->solve(request.right_hand_side, context);
        }

        backend_statistics = backend->statistics();
        {
            auto teardown = profiler.scoped("backend_destroy", EventKind::event);
            backend.reset();
        }
        host_statistics = host_executor->statistics();
        {
            auto teardown = profiler.scoped("host_executor_destroy", EventKind::event);
            host_executor.reset();
        }
    }

    Task1Report report;
    report.backend = std::move(actual_backend);
    report.dimension = request.matrix.dimension;
    report.matrix_nonzeros = request.matrix.nonzeros();
    report.backend_statistics = std::move(backend_statistics);
    report.host_executor_statistics = host_statistics;
    report.events = profiler.records();
    return {std::move(solution), std::move(report)};
}

}  // namespace eda_gpu
