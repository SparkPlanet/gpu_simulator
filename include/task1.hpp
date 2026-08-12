#pragma once

#include "backend.hpp"
#include "matrix.hpp"
#include "profiler.hpp"

#include <string>
#include <vector>

namespace eda_gpu {

struct Task1Request {
    const CscMatrix& matrix;
    const std::vector<double>& right_hand_side;
};

struct Task1Report {
    std::string backend;
    SparseIndex dimension{};
    SparseIndex matrix_nonzeros{};
    BackendStatistics backend_statistics;
    ExecutorStatistics host_executor_statistics;
    std::vector<EventRecord> events;
};

struct Task1Result {
    std::vector<double> solution;
    Task1Report report;
};

// The only public solve path in this project. Each run starts with a fresh
// backend and ends with a host solution; all setup and teardown are measured.
class Task1Pipeline {
public:
    explicit Task1Pipeline(std::string backend_name);

    [[nodiscard]] Task1Result run(const Task1Request& request) const;

private:
    std::string backend_name_;
};

}  // namespace eda_gpu
