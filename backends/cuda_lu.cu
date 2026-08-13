#include "analysis.hpp"
#include "backend.hpp"
#include "cuda_executor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eda_gpu {
namespace {

constexpr int kFactorThreads = 128;
constexpr int kWorkspaceMediumThreads = 512;
constexpr int kWorkspaceHeavyThreads = 1024;
constexpr int kVectorThreads = 256;
constexpr int kWorkspaceBlocksPerMultiprocessor = 2;
constexpr int kPersistentSolveThreads = 256;
constexpr int kWarpsPerPersistentSolveBlock = kPersistentSolveThreads / 32;
constexpr int kPersistentSolveBlocksPerMultiprocessor = 2;

enum class FactorKernelKind {
    binary_search,
    dense_workspace,
    right_looking,
};

[[nodiscard]] constexpr std::string_view backend_name(FactorKernelKind kind) noexcept {
    switch (kind) {
        case FactorKernelKind::binary_search: return "cuda-lu";
        case FactorKernelKind::dense_workspace: return "cuda-workspace-lu";
        case FactorKernelKind::right_looking: return "cuda-right-looking-lu";
    }
    return "cuda-lu";
}

[[nodiscard]] constexpr std::string_view lookup_name(FactorKernelKind kind) noexcept {
    switch (kind) {
        case FactorKernelKind::binary_search: return "binary-search";
        case FactorKernelKind::dense_workspace: return "dense-workspace-direct-index";
        case FactorKernelKind::right_looking: return "binary-search+atomic-update";
    }
    return "unknown";
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename Value>
[[nodiscard]] Value* device_pointer(Allocation& allocation) noexcept {
    return static_cast<Value*>(allocation.data());
}

template <typename Value>
[[nodiscard]] const Value* device_pointer(const Allocation& allocation) noexcept {
    return static_cast<const Value*>(allocation.data());
}

class CudaEventTimer {
public:
    CudaEventTimer(Profiler& profiler, Executor& executor)
        : profiler_(profiler), executor_(executor) {
        check_cuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
        try {
            check_cuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
            check_cuda(cudaEventRecord(start_), "cudaEventRecord(start)");
        } catch (...) {
            static_cast<void>(cudaEventDestroy(start_));
            throw;
        }
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    ~CudaEventTimer() {
        if (start_ != nullptr) static_cast<void>(cudaEventDestroy(start_));
        if (stop_ != nullptr) static_cast<void>(cudaEventDestroy(stop_));
    }

    double finish() {
        if (finished_) throw std::logic_error("CUDA timer was already finished");
        check_cuda(cudaEventRecord(stop_), "cudaEventRecord(stop)");
        executor_.synchronize();
        float milliseconds{};
        check_cuda(cudaEventElapsedTime(&milliseconds, start_, stop_),
                   "cudaEventElapsedTime");
        profiler_.add_device_time(milliseconds);
        finished_ = true;
        return milliseconds;
    }

private:
    Profiler& profiler_;
    Executor& executor_;
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
    bool finished_{};
};

struct RightLookingTask {
    SparseIndex target_row{};
    SparseIndex lower_position{};
};

struct RightLookingPlan {
    // Empty when the ordinary L-dependency schedule in AnalysisPlan is exact
    // (all BTF diagonal blocks are structurally symmetric). Nonsymmetric
    // matrices own a replacement schedule with additional double-U hazards.
    LevelSchedule pivot_schedule;
    bool reuses_analysis_factor_schedule{};
    std::vector<SparseIndex> task_level_offsets;
    std::vector<RightLookingTask> tasks;
    std::uint64_t asymmetric_u_candidates{};
    std::uint64_t intersection_steps{};
    std::uint64_t dependency_edges{};
    std::uint64_t additional_double_u_edges{};

    [[nodiscard]] std::uint64_t storage_bytes() const noexcept {
        return static_cast<std::uint64_t>(pivot_schedule.level_offsets.size() +
                                          pivot_schedule.rows.size() +
                                          task_level_offsets.size()) *
                   sizeof(SparseIndex) +
               static_cast<std::uint64_t>(tasks.size()) * sizeof(RightLookingTask);
    }
};

[[nodiscard]] LevelSchedule make_level_schedule(
    const std::vector<SparseIndex>& levels) {
    LevelSchedule schedule;
    if (levels.empty()) {
        schedule.level_offsets.push_back(0);
        return schedule;
    }
    const auto maximum = *std::max_element(levels.begin(), levels.end());
    schedule.level_offsets.assign(static_cast<std::size_t>(maximum) + 2U, 0);
    for (const auto level : levels) {
        ++schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
    }
    std::partial_sum(
        schedule.level_offsets.begin(), schedule.level_offsets.end(),
        schedule.level_offsets.begin());
    schedule.rows.assign(levels.size(), 0);
    auto positions = schedule.level_offsets;
    for (SparseIndex row = 0; row < static_cast<SparseIndex>(levels.size()); ++row) {
        const auto level = levels[static_cast<std::size_t>(row)];
        schedule.rows[static_cast<std::size_t>(
            positions[static_cast<std::size_t>(level)]++)] = row;
    }
    return schedule;
}

// A right-looking pivot k writes A(i,j) whenever L(i,k) and U(k,j) exist.
// Pivot j later reads A(i,j) to form L(i,j). If k and j share such a target
// row, k -> j is an additional (double-U) dependency even when L(j,k) is zero.
// The fixed symbolic pattern lets us find these hazards once on the CPU.
[[nodiscard]] RightLookingPlan build_right_looking_plan(
    const AnalysisPlan& plan,
    Profiler& profiler) {
    auto event = profiler.scoped("right_looking_plan_build", EventKind::event);
    RightLookingPlan result;
    const auto dimension = plan.dimension;
    const auto lower_nonzeros = static_cast<std::size_t>(plan.lower_nonzeros());
    const auto all_blocks_symmetric = std::all_of(
        plan.structurally_symmetric_blocks.begin(),
        plan.structurally_symmetric_blocks.end(),
        [](std::uint8_t value) { return value != 0U; });

    if (all_blocks_symmetric) {
        result.reuses_analysis_factor_schedule = true;
        const auto& schedule = plan.factor_schedule;

        // Bucket each L(i,k) update directly by k's already-computed level.
        // This avoids rebuilding both the lower CSC transpose and the level
        // DAG for the overwhelmingly common circuit-matrix path.
        std::vector<SparseIndex> pivot_levels(
            static_cast<std::size_t>(dimension), 0);
        for (SparseIndex level = 0; level < schedule.levels(); ++level) {
            const auto begin = schedule.level_offsets[static_cast<std::size_t>(level)];
            const auto end =
                schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
            for (auto scheduled = begin; scheduled < end; ++scheduled) {
                const auto pivot = schedule.rows[static_cast<std::size_t>(scheduled)];
                pivot_levels[static_cast<std::size_t>(pivot)] = level;
            }
        }

        result.task_level_offsets.assign(
            static_cast<std::size_t>(schedule.levels()) + 1U, 0);
        for (SparseIndex row = 0; row < dimension; ++row) {
            const auto diagonal =
                plan.diagonal_positions[static_cast<std::size_t>(row)];
            for (auto position = plan.lu_row_offsets[static_cast<std::size_t>(row)];
                 position < diagonal; ++position) {
                const auto pivot =
                    plan.lu_column_indices[static_cast<std::size_t>(position)];
                const auto level = pivot_levels[static_cast<std::size_t>(pivot)];
                ++result.task_level_offsets[static_cast<std::size_t>(level) + 1U];
            }
        }
        std::partial_sum(
            result.task_level_offsets.begin(), result.task_level_offsets.end(),
            result.task_level_offsets.begin());
        result.tasks.resize(lower_nonzeros);
        auto task_write = result.task_level_offsets;
        for (SparseIndex row = 0; row < dimension; ++row) {
            const auto diagonal =
                plan.diagonal_positions[static_cast<std::size_t>(row)];
            for (auto position = plan.lu_row_offsets[static_cast<std::size_t>(row)];
                 position < diagonal; ++position) {
                const auto pivot =
                    plan.lu_column_indices[static_cast<std::size_t>(position)];
                const auto level = pivot_levels[static_cast<std::size_t>(pivot)];
                const auto destination =
                    task_write[static_cast<std::size_t>(level)]++;
                result.tasks[static_cast<std::size_t>(destination)] = {row, position};
            }
        }

        profiler.add_value("right_looking_levels", schedule.levels());
        profiler.add_value(
            "right_looking_widest_pivot_level", schedule.widest_level());
        profiler.add_value("right_looking_tasks", result.tasks.size());
        profiler.add_value("right_looking_asymmetric_u_candidates", 0);
        profiler.add_value("right_looking_intersection_steps", 0);
        profiler.add_value("right_looking_double_u_dependency_edges", 0);
        profiler.add_value("right_looking_additional_double_u_edges", 0);
        profiler.add_value("right_looking_plan_bytes", result.storage_bytes());
        profiler.add_attribute(
            "right_looking_dependency_path",
            "reuse analysis L schedule; direct task bucketing");
        return result;
    }

    std::vector<SparseIndex> lower_offsets(
        static_cast<std::size_t>(dimension) + 1U, 0);
    for (SparseIndex row = 0; row < dimension; ++row) {
        const auto diagonal = plan.diagonal_positions[static_cast<std::size_t>(row)];
        for (auto position = plan.lu_row_offsets[static_cast<std::size_t>(row)];
             position < diagonal; ++position) {
            const auto pivot =
                plan.lu_column_indices[static_cast<std::size_t>(position)];
            ++lower_offsets[static_cast<std::size_t>(pivot) + 1U];
        }
    }
    std::partial_sum(
        lower_offsets.begin(), lower_offsets.end(), lower_offsets.begin());
    std::vector<SparseIndex> lower_positions(lower_nonzeros);
    std::vector<SparseIndex> lower_rows(lower_nonzeros);
    auto lower_write = lower_offsets;
    for (SparseIndex row = 0; row < dimension; ++row) {
        const auto diagonal = plan.diagonal_positions[static_cast<std::size_t>(row)];
        for (auto position = plan.lu_row_offsets[static_cast<std::size_t>(row)];
             position < diagonal; ++position) {
            const auto pivot =
                plan.lu_column_indices[static_cast<std::size_t>(position)];
            const auto destination =
                lower_write[static_cast<std::size_t>(pivot)]++;
            lower_positions[static_cast<std::size_t>(destination)] = position;
            lower_rows[static_cast<std::size_t>(destination)] = row;
        }
    }

    // The ordinary L schedule already contains k -> j whenever L(j,k)
    // exists. For an asymmetric U(k,j), an additional right-looking hazard
    // exists exactly when columns k and j of strict L share a target row.
    // Both columns are sorted in lower_rows, so reciprocal filtering and the
    // exceptional intersections need no global marker or binary lookup.
    std::vector<SparseIndex> levels(static_cast<std::size_t>(dimension), 0);
    for (SparseIndex pivot = 0; pivot < dimension; ++pivot) {
        const auto next_level = static_cast<SparseIndex>(
            levels[static_cast<std::size_t>(pivot)] + 1);
        for (auto offset = lower_offsets[static_cast<std::size_t>(pivot)];
             offset < lower_offsets[static_cast<std::size_t>(pivot) + 1U]; ++offset) {
            const auto target = lower_rows[static_cast<std::size_t>(offset)];
            levels[static_cast<std::size_t>(target)] =
                std::max(levels[static_cast<std::size_t>(target)], next_level);
        }
        const auto upper_begin =
            plan.diagonal_positions[static_cast<std::size_t>(pivot)] + 1;
        const auto upper_end =
            plan.lu_row_offsets[static_cast<std::size_t>(pivot) + 1U];
        auto reciprocal = lower_offsets[static_cast<std::size_t>(pivot)];
        const auto reciprocal_end =
            lower_offsets[static_cast<std::size_t>(pivot) + 1U];
        for (auto upper = upper_begin; upper < upper_end; ++upper) {
            const auto dependent =
                plan.lu_column_indices[static_cast<std::size_t>(upper)];
            while (reciprocal < reciprocal_end &&
                   lower_rows[static_cast<std::size_t>(reciprocal)] < dependent) {
                ++reciprocal;
            }
            if (reciprocal < reciprocal_end &&
                lower_rows[static_cast<std::size_t>(reciprocal)] == dependent) {
                continue;
            }
            ++result.asymmetric_u_candidates;

            auto left = reciprocal;
            auto right = lower_offsets[static_cast<std::size_t>(dependent)];
            const auto right_end =
                lower_offsets[static_cast<std::size_t>(dependent) + 1U];
            bool has_shared_target = false;
            while (left < reciprocal_end && right < right_end) {
                ++result.intersection_steps;
                const auto left_row = lower_rows[static_cast<std::size_t>(left)];
                const auto right_row = lower_rows[static_cast<std::size_t>(right)];
                if (left_row == right_row) {
                    has_shared_target = true;
                    break;
                }
                if (left_row < right_row) ++left;
                else ++right;
            }
            if (!has_shared_target) continue;
            ++result.dependency_edges;
            ++result.additional_double_u_edges;
            levels[static_cast<std::size_t>(dependent)] =
                std::max(levels[static_cast<std::size_t>(dependent)], next_level);
        }
    }
    result.pivot_schedule = make_level_schedule(levels);

    result.tasks.reserve(lower_nonzeros);
    result.task_level_offsets.reserve(
        static_cast<std::size_t>(result.pivot_schedule.levels()) + 1U);
    result.task_level_offsets.push_back(0);
    for (SparseIndex level = 0; level < result.pivot_schedule.levels(); ++level) {
        const auto begin =
            result.pivot_schedule.level_offsets[static_cast<std::size_t>(level)];
        const auto end = result.pivot_schedule.level_offsets[
            static_cast<std::size_t>(level) + 1U];
        for (auto scheduled = begin; scheduled < end; ++scheduled) {
            const auto pivot =
                result.pivot_schedule.rows[static_cast<std::size_t>(scheduled)];
            for (auto offset = lower_offsets[static_cast<std::size_t>(pivot)];
                 offset < lower_offsets[static_cast<std::size_t>(pivot) + 1U];
                 ++offset) {
                result.tasks.push_back({
                    lower_rows[static_cast<std::size_t>(offset)],
                    lower_positions[static_cast<std::size_t>(offset)]});
            }
        }
        if (result.tasks.size() >
            static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max())) {
            throw std::runtime_error("right-looking task list exceeds 32-bit ABI");
        }
        result.task_level_offsets.push_back(
            static_cast<SparseIndex>(result.tasks.size()));
    }

    profiler.add_value("right_looking_levels", result.pivot_schedule.levels());
    profiler.add_value(
        "right_looking_widest_pivot_level",
        result.pivot_schedule.widest_level());
    profiler.add_value("right_looking_tasks", result.tasks.size());
    profiler.add_value(
        "right_looking_asymmetric_u_candidates",
        result.asymmetric_u_candidates);
    profiler.add_value(
        "right_looking_intersection_steps", result.intersection_steps);
    profiler.add_value(
        "right_looking_double_u_dependency_edges", result.dependency_edges);
    profiler.add_value(
        "right_looking_additional_double_u_edges",
        result.additional_double_u_edges);
    profiler.add_value("right_looking_plan_bytes", result.storage_bytes());
    profiler.add_attribute(
        "right_looking_dependency_path",
        "reciprocal-L filter plus sorted-column intersection");
    return result;
}

__device__ SparseIndex binary_lookup(
    const SparseIndex* columns,
    SparseIndex begin,
    SparseIndex end,
    SparseIndex target) {
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        if (columns[middle] < target) begin = middle + 1;
        else end = middle;
    }
    return begin;
}

__global__ void initialize_lu_values_kernel(
    SparseIndex nonzeros,
    const SparseIndex* original_row_indices,
    const SparseIndex* inverse_row_permutation,
    const SparseIndex* matrix_to_lu,
    const double* row_scale_factors,
    const double* matrix_values,
    double* factor_values) {
    const auto position = static_cast<SparseIndex>(blockIdx.x * blockDim.x + threadIdx.x);
    if (position >= nonzeros) return;
    const auto original_row = original_row_indices[position];
    const auto new_row = inverse_row_permutation[original_row];
    factor_values[matrix_to_lu[position]] =
        matrix_values[position] / row_scale_factors[new_row];
}

__global__ void numerical_lu_level_kernel(
    const SparseIndex* scheduled_rows,
    SparseIndex row_count,
    const SparseIndex* row_offsets,
    const SparseIndex* columns,
    const SparseIndex* diagonal_positions,
    double* factor_values,
    double pivot_threshold,
    int* first_bad_pivot) {
    const auto scheduled = static_cast<SparseIndex>(blockIdx.x);
    if (scheduled >= row_count) return;
    const auto row = scheduled_rows[scheduled];
    const auto row_begin = row_offsets[row];
    const auto row_end = row_offsets[row + 1];
    const auto diagonal = diagonal_positions[row];
    __shared__ double multiplier;
    __shared__ int pivot_valid;

    for (auto lower_position = row_begin; lower_position < diagonal; ++lower_position) {
        if (threadIdx.x == 0) {
            const auto pivot_row = columns[lower_position];
            const auto pivot = factor_values[diagonal_positions[pivot_row]];
            pivot_valid = isfinite(pivot) && fabs(pivot) > pivot_threshold;
            if (pivot_valid) {
                multiplier = factor_values[lower_position] / pivot;
                factor_values[lower_position] = multiplier;
            } else {
                multiplier = 0.0;
                atomicCAS(first_bad_pivot, -1, pivot_row);
            }
        }
        __syncthreads();
        if (!pivot_valid) return;

        const auto pivot_row = columns[lower_position];
        const auto upper_begin = diagonal_positions[pivot_row] + 1;
        const auto upper_end = row_offsets[pivot_row + 1];
        for (auto upper_position = upper_begin + static_cast<SparseIndex>(threadIdx.x);
             upper_position < upper_end;
             upper_position += static_cast<SparseIndex>(blockDim.x)) {
            const auto target_column = columns[upper_position];
            const auto destination =
                binary_lookup(columns, row_begin, row_end, target_column);
            if (destination >= row_end || columns[destination] != target_column) {
                atomicCAS(first_bad_pivot, -1, -2);
            } else {
                factor_values[destination] -=
                    multiplier * factor_values[upper_position];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const auto pivot = factor_values[diagonal];
        if (!isfinite(pivot) || fabs(pivot) <= pivot_threshold) {
            atomicCAS(first_bad_pivot, -1, row);
        }
    }
}

// Each physical block owns one dense workspace and processes multiple rows of
// a dependency level. The symbolic LU pattern guarantees that every target of
// an update is present in the current row. We therefore scatter only the known
// row pattern, index the workspace directly during elimination, and gather the
// completed row back to compressed factor storage. No O(n) clear is required.
__global__ void numerical_lu_workspace_level_kernel(
    const SparseIndex* scheduled_rows,
    SparseIndex row_count,
    SparseIndex dimension,
    const SparseIndex* row_offsets,
    const SparseIndex* columns,
    const SparseIndex* diagonal_positions,
    double* factor_values,
    double* workspaces,
    double pivot_threshold,
    int* first_bad_pivot) {
    auto* workspace =
        workspaces + static_cast<std::size_t>(blockIdx.x) *
                         static_cast<std::size_t>(dimension);
    __shared__ double multiplier;
    __shared__ int pivot_valid;

    for (auto scheduled = static_cast<SparseIndex>(blockIdx.x);
         scheduled < row_count;
         scheduled += static_cast<SparseIndex>(gridDim.x)) {
        const auto row = scheduled_rows[scheduled];
        const auto row_begin = row_offsets[row];
        const auto row_end = row_offsets[row + 1];
        const auto diagonal = diagonal_positions[row];

        // Every position used below belongs to this known factor row, so stale
        // values elsewhere in the workspace are never observed.
        for (auto position = row_begin + static_cast<SparseIndex>(threadIdx.x);
             position < row_end;
             position += static_cast<SparseIndex>(blockDim.x)) {
            workspace[columns[position]] = factor_values[position];
        }
        __syncthreads();

        for (auto lower_position = row_begin; lower_position < diagonal;
             ++lower_position) {
            const auto pivot_row = columns[lower_position];
            if (threadIdx.x == 0) {
                const auto pivot = factor_values[diagonal_positions[pivot_row]];
                pivot_valid = isfinite(pivot) && fabs(pivot) > pivot_threshold;
                if (pivot_valid) {
                    multiplier = workspace[pivot_row] / pivot;
                    workspace[pivot_row] = multiplier;
                } else {
                    multiplier = 0.0;
                    atomicCAS(first_bad_pivot, -1, pivot_row);
                }
            }
            __syncthreads();
            if (!pivot_valid) return;

            const auto upper_begin = diagonal_positions[pivot_row] + 1;
            const auto upper_end = row_offsets[pivot_row + 1];
            for (auto upper_position =
                     upper_begin + static_cast<SparseIndex>(threadIdx.x);
                 upper_position < upper_end;
                 upper_position += static_cast<SparseIndex>(blockDim.x)) {
                const auto target_column = columns[upper_position];
                workspace[target_column] -=
                    multiplier * factor_values[upper_position];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const auto pivot = workspace[row];
            pivot_valid = isfinite(pivot) && fabs(pivot) > pivot_threshold;
            if (!pivot_valid) atomicCAS(first_bad_pivot, -1, row);
        }
        __syncthreads();
        if (!pivot_valid) return;

        for (auto position = row_begin + static_cast<SparseIndex>(threadIdx.x);
             position < row_end;
             position += static_cast<SparseIndex>(blockDim.x)) {
            factor_values[position] = workspace[columns[position]];
        }
        // The next row assigned to this block can reuse the workspace only
        // after every thread has finished gathering the current row.
        __syncthreads();
    }
}

// GLU-style row-oriented right-looking update. One CUDA block owns one
// L(target,pivot) task and uses the completed pivot row to update the target
// row immediately. Independent pivots in a right-looking level may write the
// same future factor entry, so the compressed destination is updated atomically.
__global__ void numerical_lu_right_looking_level_kernel(
    const RightLookingTask* tasks,
    SparseIndex task_count,
    const SparseIndex* row_offsets,
    const SparseIndex* columns,
    const SparseIndex* diagonal_positions,
    double* factor_values,
    double pivot_threshold,
    int* first_bad_pivot) {
    const auto task_index = static_cast<SparseIndex>(blockIdx.x);
    if (task_index >= task_count) return;
    const auto task = tasks[task_index];
    const auto pivot = columns[task.lower_position];
    __shared__ double multiplier;
    __shared__ int pivot_valid;

    if (threadIdx.x == 0) {
        const auto diagonal = diagonal_positions[pivot];
        const auto pivot_value = factor_values[diagonal];
        pivot_valid = isfinite(pivot_value) && fabs(pivot_value) > pivot_threshold;
        if (pivot_valid) {
            multiplier = factor_values[task.lower_position] / pivot_value;
            factor_values[task.lower_position] = multiplier;
        } else {
            multiplier = 0.0;
            atomicCAS(first_bad_pivot, -1, pivot);
        }
    }
    __syncthreads();
    if (!pivot_valid) return;

    const auto target_begin = row_offsets[task.target_row];
    const auto target_end = row_offsets[task.target_row + 1];
    const auto upper_begin = diagonal_positions[pivot] + 1;
    const auto upper_end = row_offsets[pivot + 1];
    for (auto upper = upper_begin + static_cast<SparseIndex>(threadIdx.x);
         upper < upper_end;
         upper += static_cast<SparseIndex>(blockDim.x)) {
        const auto target_column = columns[upper];
        const auto destination =
            binary_lookup(columns, target_begin, target_end, target_column);
        if (destination >= target_end || columns[destination] != target_column) {
            atomicCAS(first_bad_pivot, -1, -2);
        } else {
            atomicAdd(
                factor_values + destination,
                -multiplier * factor_values[upper]);
        }
    }
}

__global__ void validate_diagonal_kernel(
    SparseIndex dimension,
    const SparseIndex* diagonal_positions,
    const double* factor_values,
    double pivot_threshold,
    int* first_bad_pivot) {
    const auto row = static_cast<SparseIndex>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= dimension) return;
    const auto pivot = factor_values[diagonal_positions[row]];
    if (!isfinite(pivot) || fabs(pivot) <= pivot_threshold) {
        atomicCAS(first_bad_pivot, -1, row);
    }
}

__global__ void permute_and_scale_rhs_kernel(
    SparseIndex dimension,
    const SparseIndex* row_permutation,
    const double* row_scale_factors,
    const double* right_hand_side,
    double* permuted_right_hand_side) {
    const auto row = static_cast<SparseIndex>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= dimension) return;
    permuted_right_hand_side[row] =
        right_hand_side[row_permutation[row]] / row_scale_factors[row];
}

template <bool Backward>
__global__ void triangular_solve_persistent_kernel(
    SparseIndex dimension,
    const SparseIndex* scheduled_rows,
    const SparseIndex* row_offsets,
    const SparseIndex* columns,
    const SparseIndex* diagonal_positions,
    const double* factor_values,
    const double* input,
    double* output,
    SparseIndex* next_task,
    int* row_ready) {
    const auto lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned int full_warp = 0xffffffffU;
    const volatile double* completed_values = output;

    while (true) {
        SparseIndex scheduled{};
        if (lane == 0) scheduled = atomicAdd(next_task, SparseIndex{1});
        scheduled = __shfl_sync(full_warp, scheduled, 0);
        if (scheduled >= dimension) return;

        const auto row = scheduled_rows[scheduled];
        const auto diagonal = diagonal_positions[row];
        const auto begin = Backward ? diagonal + 1 : row_offsets[row];
        const auto end = Backward ? row_offsets[row + 1] : diagonal;
        double partial{};
        for (auto position = begin + static_cast<SparseIndex>(lane);
             position < end; position += 32) {
            const auto dependency = columns[position];
            while (atomicAdd(row_ready + dependency, 0) == 0) {
                __nanosleep(64);
            }
            partial += factor_values[position] * completed_values[dependency];
        }
        for (int offset = 16; offset != 0; offset /= 2) {
            partial += __shfl_down_sync(full_warp, partial, offset);
        }
        if (lane == 0) {
            auto value = input[row] - partial;
            if (Backward) value /= factor_values[diagonal];
            output[row] = value;
            // Publish the value before releasing dependants that may be
            // spinning in a different SM.
            __threadfence();
            atomicExch(row_ready + row, 1);
        }
    }
}

__global__ void inverse_permute_solution_kernel(
    SparseIndex dimension,
    const SparseIndex* column_permutation,
    const double* permuted_solution,
    double* solution) {
    const auto column = static_cast<SparseIndex>(blockIdx.x * blockDim.x + threadIdx.x);
    if (column >= dimension) return;
    solution[column_permutation[column]] = permuted_solution[column];
}

struct DeviceBuffers {
    Allocation row_offsets;
    Allocation columns;
    Allocation diagonal_positions;
    Allocation matrix_to_lu;
    Allocation original_row_indices;
    Allocation inverse_row_permutation;
    Allocation row_permutation;
    Allocation column_permutation;
    Allocation row_scale_factors;
    Allocation factor_schedule_rows;
    Allocation forward_schedule_rows;
    Allocation backward_schedule_rows;
    Allocation matrix_values;
    Allocation factor_values;
    Allocation factor_workspaces;
    Allocation right_looking_tasks;
    Allocation right_hand_side;
    Allocation permuted_right_hand_side;
    Allocation intermediate;
    Allocation permuted_solution;
    Allocation solution;
    Allocation status;
    Allocation solve_next_task;
    Allocation solve_row_ready;
};

class CudaLuBackend final : public Task1Backend {
public:
    explicit CudaLuBackend(FactorKernelKind factor_kernel) noexcept
        : factor_kernel_(factor_kernel) {}

    [[nodiscard]] std::string_view name() const noexcept override {
        return backend_name(factor_kernel_);
    }

    void analyze(const CscMatrix& matrix, BackendContext& context) override {
        reset_device();
        matrix_ = nullptr;
        plan_ = analyzer_.analyze(matrix, context.profiler);
        right_looking_plan_ = {};
        if (factor_kernel_ == FactorKernelKind::right_looking) {
            right_looking_plan_ = build_right_looking_plan(plan_, context.profiler);
        }
        matrix_ = &matrix;
        populate_plan_statistics();
    }

    void factorize(const CscMatrix& matrix, BackendContext& context) override {
        if (matrix_ != &matrix || plan_.dimension != matrix.dimension) {
            throw std::runtime_error("CUDA LU requires analysis of the same matrix instance");
        }
        {
            auto event = context.profiler.scoped("cuda_executor_create", EventKind::event);
            cuda_executor_ = make_cuda_executor(context.profiler);
            context.profiler.add_attribute("executor", std::string(cuda_executor_->name()));
        }
        configure_persistent_solve(context);
        configure_factor_kernel(context);
        configure_kernel_occupancy(context);
        allocate_device_buffers(context);
        upload_structure(context);
        upload_matrix_values(matrix, context);
        initialize_factor_values(context);
        run_numerical_lu(context);
        check_factor_status(context);
        update_device_statistics();
    }

    [[nodiscard]] std::vector<double> solve(
        const std::vector<double>& right_hand_side,
        BackendContext& context) override {
        if (cuda_executor_ == nullptr) {
            throw std::runtime_error("CUDA solve requires a completed numerical factorization");
        }
        upload_right_hand_side(right_hand_side, context);
        permute_right_hand_side(context);
        run_forward_solve(context);
        run_backward_solve(context);
        inverse_permute_solution(context);

        std::vector<double> solution(static_cast<std::size_t>(plan_.dimension));
        {
            auto event = context.profiler.scoped("solution_d2h", EventKind::transfer);
            CudaEventTimer timer(context.profiler, *cuda_executor_);
            context.host_executor.copy_from(
                *cuda_executor_, solution.data(), buffers_.solution.data(),
                solution.size() * sizeof(double));
            timer.finish();
        }
        update_device_statistics();
        return solution;
    }

    [[nodiscard]] BackendStatistics statistics() const override {
        auto result = statistics_;
        if (cuda_executor_ != nullptr) {
            const auto& executor = cuda_executor_->statistics();
            result.values["device_live_bytes"] = executor.live_bytes;
            result.values["device_peak_bytes"] = executor.peak_bytes;
            result.values["device_allocation_calls"] = executor.allocation_calls;
            result.values["device_copy_calls"] = executor.copy_calls;
            result.values["device_copied_bytes"] = executor.copied_bytes;
            result.values["device_synchronization_calls"] = executor.synchronization_calls;
        }
        return result;
    }

private:
    struct OccupancyInfo {
        int threads_per_block{};
        int registers_per_thread{};
        std::size_t static_shared_memory_bytes{};
        int active_blocks_per_multiprocessor{};
        int active_warps_per_multiprocessor{};
        int maximum_warps_per_multiprocessor{};
        double theoretical_occupancy_percent{};
    };

    [[nodiscard]] const LevelSchedule& right_looking_schedule() const noexcept {
        return right_looking_plan_.reuses_analysis_factor_schedule
                   ? plan_.factor_schedule
                   : right_looking_plan_.pivot_schedule;
    }

    template <typename Value>
    void allocate(Allocation& allocation, std::size_t count) {
        allocation = cuda_executor_->allocate(count * sizeof(Value));
    }

    template <typename Value>
    void upload(
        Allocation& destination,
        const std::vector<Value>& source,
        BackendContext& context) {
        cuda_executor_->copy_from(
            context.host_executor,
            destination.data(),
            source.data(),
            source.size() * sizeof(Value));
    }

    void allocate_device_buffers(BackendContext& context) {
        auto event = context.profiler.scoped("device_allocation", EventKind::allocation);
        allocate<SparseIndex>(buffers_.row_offsets, plan_.lu_row_offsets.size());
        allocate<SparseIndex>(buffers_.columns, plan_.lu_column_indices.size());
        allocate<SparseIndex>(buffers_.diagonal_positions, plan_.diagonal_positions.size());
        allocate<SparseIndex>(buffers_.matrix_to_lu, plan_.matrix_to_lu.size());
        allocate<SparseIndex>(buffers_.original_row_indices, matrix_->row_indices.size());
        allocate<SparseIndex>(buffers_.inverse_row_permutation,
                              plan_.inverse_row_permutation.size());
        allocate<SparseIndex>(buffers_.row_permutation, plan_.row_permutation.size());
        allocate<SparseIndex>(buffers_.column_permutation, plan_.column_permutation.size());
        allocate<double>(buffers_.row_scale_factors, plan_.row_scale_factors.size());
        allocate<SparseIndex>(buffers_.factor_schedule_rows, plan_.factor_schedule.rows.size());
        allocate<SparseIndex>(buffers_.forward_schedule_rows, plan_.forward_schedule.rows.size());
        allocate<SparseIndex>(buffers_.backward_schedule_rows,
                              plan_.backward_schedule.rows.size());
        allocate<double>(buffers_.matrix_values, matrix_->values.size());
        allocate<double>(buffers_.factor_values, plan_.lu_column_indices.size());
        if (factor_kernel_ == FactorKernelKind::dense_workspace) {
            allocate<double>(
                buffers_.factor_workspaces,
                static_cast<std::size_t>(workspace_worker_blocks_) *
                    static_cast<std::size_t>(plan_.dimension));
            context.profiler.add_value(
                "factor_workspace_bytes", buffers_.factor_workspaces.size_bytes());
            context.profiler.add_value(
                "factor_workspace_worker_blocks", workspace_worker_blocks_);
        }
        if (factor_kernel_ == FactorKernelKind::right_looking) {
            allocate<RightLookingTask>(
                buffers_.right_looking_tasks, right_looking_plan_.tasks.size());
        }
        allocate<double>(buffers_.right_hand_side, static_cast<std::size_t>(plan_.dimension));
        allocate<double>(buffers_.permuted_right_hand_side,
                         static_cast<std::size_t>(plan_.dimension));
        allocate<double>(buffers_.intermediate, static_cast<std::size_t>(plan_.dimension));
        allocate<double>(buffers_.permuted_solution,
                         static_cast<std::size_t>(plan_.dimension));
        allocate<double>(buffers_.solution, static_cast<std::size_t>(plan_.dimension));
        allocate<int>(buffers_.status, 1U);
        allocate<SparseIndex>(buffers_.solve_next_task, 1U);
        allocate<int>(buffers_.solve_row_ready, static_cast<std::size_t>(plan_.dimension));
        context.profiler.add_value(
            "device_live_bytes", cuda_executor_->statistics().live_bytes);
    }

    void upload_structure(BackendContext& context) {
        auto event = context.profiler.scoped("analysis_plan_h2d", EventKind::transfer);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        upload(buffers_.row_offsets, plan_.lu_row_offsets, context);
        upload(buffers_.columns, plan_.lu_column_indices, context);
        upload(buffers_.diagonal_positions, plan_.diagonal_positions, context);
        upload(buffers_.matrix_to_lu, plan_.matrix_to_lu, context);
        upload(buffers_.original_row_indices, matrix_->row_indices, context);
        upload(buffers_.inverse_row_permutation, plan_.inverse_row_permutation, context);
        upload(buffers_.row_permutation, plan_.row_permutation, context);
        upload(buffers_.column_permutation, plan_.column_permutation, context);
        upload(buffers_.row_scale_factors, plan_.row_scale_factors, context);
        upload(buffers_.factor_schedule_rows, plan_.factor_schedule.rows, context);
        if (factor_kernel_ == FactorKernelKind::right_looking) {
            upload(
                buffers_.right_looking_tasks, right_looking_plan_.tasks, context);
        }
        upload(buffers_.forward_schedule_rows, plan_.forward_schedule.rows, context);
        upload(buffers_.backward_schedule_rows, plan_.backward_schedule.rows, context);
        timer.finish();
    }

    void upload_matrix_values(const CscMatrix& matrix, BackendContext& context) {
        auto event = context.profiler.scoped("matrix_values_h2d", EventKind::transfer);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        upload(buffers_.matrix_values, matrix.values, context);
        timer.finish();
    }

    void initialize_factor_values(BackendContext& context) {
        auto event = context.profiler.scoped("cuda_lu_value_initialize", EventKind::kernel);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        check_cuda(
            cudaMemset(buffers_.factor_values.data(), 0, buffers_.factor_values.size_bytes()),
            "cudaMemset(factor_values)");
        check_cuda(cudaMemset(buffers_.status.data(), 0xff, sizeof(int)),
                   "cudaMemset(status)");
        const auto blocks =
            (plan_.input_nonzeros + kVectorThreads - 1) / kVectorThreads;
        initialize_lu_values_kernel<<<blocks, kVectorThreads>>>(
            plan_.input_nonzeros,
            device_pointer<SparseIndex>(buffers_.original_row_indices),
            device_pointer<SparseIndex>(buffers_.inverse_row_permutation),
            device_pointer<SparseIndex>(buffers_.matrix_to_lu),
            device_pointer<double>(buffers_.row_scale_factors),
            device_pointer<double>(buffers_.matrix_values),
            device_pointer<double>(buffers_.factor_values));
        check_cuda(cudaGetLastError(), "initialize_lu_values_kernel launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", 1);
    }

    void run_numerical_lu(BackendContext& context) {
        auto event = context.profiler.scoped("cuda_numeric_lu", EventKind::kernel);
        context.profiler.add_attribute("lookup", std::string(lookup_name(factor_kernel_)));
        context.profiler.add_attribute("schedule", "one kernel launch per factor level");
        context.profiler.add_estimated_flops(plan_.estimated_factor_flops);
        CudaEventTimer timer(context.profiler, *cuda_executor_);

        if (factor_kernel_ == FactorKernelKind::right_looking) {
            const auto& schedule = right_looking_schedule();
            SparseIndex launches{};
            const auto* tasks =
                device_pointer<RightLookingTask>(buffers_.right_looking_tasks);
            for (SparseIndex level = 0;
                 level < schedule.levels(); ++level) {
                const auto begin = right_looking_plan_.task_level_offsets[
                    static_cast<std::size_t>(level)];
                const auto end = right_looking_plan_.task_level_offsets[
                    static_cast<std::size_t>(level) + 1U];
                const auto task_count = end - begin;
                if (task_count == 0) continue;
                numerical_lu_right_looking_level_kernel<<<task_count, kFactorThreads>>>(
                    tasks + begin,
                    task_count,
                    device_pointer<SparseIndex>(buffers_.row_offsets),
                    device_pointer<SparseIndex>(buffers_.columns),
                    device_pointer<SparseIndex>(buffers_.diagonal_positions),
                    device_pointer<double>(buffers_.factor_values),
                    64.0 * std::numeric_limits<double>::epsilon(),
                    device_pointer<int>(buffers_.status));
                ++launches;
            }
            const auto blocks =
                (plan_.dimension + kVectorThreads - 1) / kVectorThreads;
            validate_diagonal_kernel<<<blocks, kVectorThreads>>>(
                plan_.dimension,
                device_pointer<SparseIndex>(buffers_.diagonal_positions),
                device_pointer<double>(buffers_.factor_values),
                64.0 * std::numeric_limits<double>::epsilon(),
                device_pointer<int>(buffers_.status));
            ++launches;
            check_cuda(
                cudaGetLastError(),
                "numerical_lu_right_looking_level_kernel launch");
            timer.finish();
            context.profiler.add_value("kernel_launches", launches);
            context.profiler.add_value(
                "levels", schedule.levels());
            context.profiler.add_value(
                "widest_level", schedule.widest_level());
            context.profiler.add_value(
                "right_looking_tasks", right_looking_plan_.tasks.size());
            return;
        }

        const auto* scheduled_rows =
            device_pointer<SparseIndex>(buffers_.factor_schedule_rows);
        for (SparseIndex level = 0; level < plan_.factor_schedule.levels(); ++level) {
            const auto begin =
                plan_.factor_schedule.level_offsets[static_cast<std::size_t>(level)];
            const auto end =
                plan_.factor_schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
            const auto width = end - begin;
            if (factor_kernel_ == FactorKernelKind::binary_search) {
                numerical_lu_level_kernel<<<width, kFactorThreads>>>(
                    scheduled_rows + begin,
                    width,
                    device_pointer<SparseIndex>(buffers_.row_offsets),
                    device_pointer<SparseIndex>(buffers_.columns),
                    device_pointer<SparseIndex>(buffers_.diagonal_positions),
                    device_pointer<double>(buffers_.factor_values),
                    64.0 * std::numeric_limits<double>::epsilon(),
                    device_pointer<int>(buffers_.status));
            } else {
                const auto workers = std::min(width, workspace_worker_blocks_);
                const auto threads = workspace_threads_for_range(begin, end);
                numerical_lu_workspace_level_kernel<<<workers, threads>>>(
                    scheduled_rows + begin,
                    width,
                    plan_.dimension,
                    device_pointer<SparseIndex>(buffers_.row_offsets),
                    device_pointer<SparseIndex>(buffers_.columns),
                    device_pointer<SparseIndex>(buffers_.diagonal_positions),
                    device_pointer<double>(buffers_.factor_values),
                    device_pointer<double>(buffers_.factor_workspaces),
                    64.0 * std::numeric_limits<double>::epsilon(),
                    device_pointer<int>(buffers_.status));
            }
        }
        check_cuda(cudaGetLastError(), "numerical_lu_level_kernel launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", plan_.factor_schedule.levels());
        context.profiler.add_value("levels", plan_.factor_schedule.levels());
        context.profiler.add_value("widest_level", plan_.factor_schedule.widest_level());
        if (factor_kernel_ == FactorKernelKind::dense_workspace) {
            context.profiler.add_value("maximum_worker_blocks", workspace_worker_blocks_);
            context.profiler.add_value(
                "levels_128_threads", workspace_levels_128_threads_);
            context.profiler.add_value(
                "levels_512_threads", workspace_levels_512_threads_);
            context.profiler.add_value(
                "levels_1024_threads", workspace_levels_1024_threads_);
        }
    }

    void check_factor_status(BackendContext& context) {
        int status{-1};
        {
            auto event = context.profiler.scoped("factor_status_d2h", EventKind::transfer);
            CudaEventTimer timer(context.profiler, *cuda_executor_);
            context.host_executor.copy_from(
                *cuda_executor_, &status, buffers_.status.data(), sizeof(status));
            timer.finish();
            context.profiler.add_value("first_bad_pivot", status);
        }
        if (status == -2) {
            throw std::runtime_error("CUDA numerical LU found an incomplete symbolic pattern");
        }
        if (status >= 0) {
            throw std::runtime_error(
                "CUDA fixed-pivot LU encountered a zero/tiny pivot at permuted row " +
                std::to_string(status));
        }
    }

    void upload_right_hand_side(
        const std::vector<double>& right_hand_side,
        BackendContext& context) {
        auto event = context.profiler.scoped("rhs_h2d", EventKind::transfer);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        cuda_executor_->copy_from(
            context.host_executor,
            buffers_.right_hand_side.data(),
            right_hand_side.data(),
            right_hand_side.size() * sizeof(double));
        timer.finish();
    }

    void permute_right_hand_side(BackendContext& context) {
        auto event = context.profiler.scoped("cuda_rhs_permutation", EventKind::kernel);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        const auto blocks = (plan_.dimension + kVectorThreads - 1) / kVectorThreads;
        permute_and_scale_rhs_kernel<<<blocks, kVectorThreads>>>(
            plan_.dimension,
            device_pointer<SparseIndex>(buffers_.row_permutation),
            device_pointer<double>(buffers_.row_scale_factors),
            device_pointer<double>(buffers_.right_hand_side),
            device_pointer<double>(buffers_.permuted_right_hand_side));
        check_cuda(cudaGetLastError(), "permute_and_scale_rhs_kernel launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", 1);
    }

    void run_forward_solve(BackendContext& context) {
        auto event = context.profiler.scoped("cuda_forward_solve", EventKind::kernel);
        context.profiler.add_attribute(
            "schedule", "persistent warp tasks with dependency completion flags");
        context.profiler.add_estimated_flops(2.0 * plan_.lower_nonzeros());
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        check_cuda(
            cudaMemset(buffers_.solve_next_task.data(), 0, sizeof(SparseIndex)),
            "cudaMemset(forward solve queue)");
        check_cuda(
            cudaMemset(
                buffers_.solve_row_ready.data(), 0,
                buffers_.solve_row_ready.size_bytes()),
            "cudaMemset(forward solve ready flags)");
        triangular_solve_persistent_kernel<false>
            <<<persistent_solve_blocks_, kPersistentSolveThreads>>>(
                plan_.dimension,
                device_pointer<SparseIndex>(buffers_.forward_schedule_rows),
                device_pointer<SparseIndex>(buffers_.row_offsets),
                device_pointer<SparseIndex>(buffers_.columns),
                device_pointer<SparseIndex>(buffers_.diagonal_positions),
                device_pointer<double>(buffers_.factor_values),
                device_pointer<double>(buffers_.permuted_right_hand_side),
                device_pointer<double>(buffers_.intermediate),
                device_pointer<SparseIndex>(buffers_.solve_next_task),
                device_pointer<int>(buffers_.solve_row_ready));
        check_cuda(cudaGetLastError(), "triangular_solve_persistent_kernel<forward> launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", 1);
        context.profiler.add_value("persistent_blocks", persistent_solve_blocks_);
        context.profiler.add_value("warps_per_block", kWarpsPerPersistentSolveBlock);
    }

    void run_backward_solve(BackendContext& context) {
        auto event = context.profiler.scoped("cuda_backward_solve", EventKind::kernel);
        context.profiler.add_attribute(
            "schedule", "persistent warp tasks with dependency completion flags");
        context.profiler.add_estimated_flops(
            2.0 * plan_.upper_nonzeros() + plan_.dimension);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        check_cuda(
            cudaMemset(buffers_.solve_next_task.data(), 0, sizeof(SparseIndex)),
            "cudaMemset(backward solve queue)");
        check_cuda(
            cudaMemset(
                buffers_.solve_row_ready.data(), 0,
                buffers_.solve_row_ready.size_bytes()),
            "cudaMemset(backward solve ready flags)");
        triangular_solve_persistent_kernel<true>
            <<<persistent_solve_blocks_, kPersistentSolveThreads>>>(
                plan_.dimension,
                device_pointer<SparseIndex>(buffers_.backward_schedule_rows),
                device_pointer<SparseIndex>(buffers_.row_offsets),
                device_pointer<SparseIndex>(buffers_.columns),
                device_pointer<SparseIndex>(buffers_.diagonal_positions),
                device_pointer<double>(buffers_.factor_values),
                device_pointer<double>(buffers_.intermediate),
                device_pointer<double>(buffers_.permuted_solution),
                device_pointer<SparseIndex>(buffers_.solve_next_task),
                device_pointer<int>(buffers_.solve_row_ready));
        check_cuda(cudaGetLastError(), "triangular_solve_persistent_kernel<backward> launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", 1);
        context.profiler.add_value("persistent_blocks", persistent_solve_blocks_);
        context.profiler.add_value("warps_per_block", kWarpsPerPersistentSolveBlock);
    }

    void inverse_permute_solution(BackendContext& context) {
        auto event = context.profiler.scoped("cuda_solution_inverse_permutation",
                                             EventKind::kernel);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        const auto blocks = (plan_.dimension + kVectorThreads - 1) / kVectorThreads;
        inverse_permute_solution_kernel<<<blocks, kVectorThreads>>>(
            plan_.dimension,
            device_pointer<SparseIndex>(buffers_.column_permutation),
            device_pointer<double>(buffers_.permuted_solution),
            device_pointer<double>(buffers_.solution));
        check_cuda(cudaGetLastError(), "inverse_permute_solution_kernel launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", 1);
    }

    void populate_plan_statistics() {
        statistics_ = {};
        statistics_.attributes["algorithm"] =
            factor_kernel_ == FactorKernelKind::right_looking
                ? "CUDA fixed-pivot row-oriented right-looking sparse LU"
                : "CUDA fixed-pivot row-oriented up-looking sparse LU";
        statistics_.attributes["analysis"] =
            "CPU KLU BTF+AMD + blockwise elimination-tree/generic symbolic";
        statistics_.attributes["execution_space"] = "CUDA";
        statistics_.attributes["factor_storage"] = "combined CSR L/U";
        statistics_.attributes["lookup"] = std::string(lookup_name(factor_kernel_));
        statistics_.attributes["pivoting"] = "static diagonal pivot";
        statistics_.attributes["schedule"] = "level-scheduled kernels";
        statistics_.values["input_nonzeros"] = plan_.input_nonzeros;
        statistics_.values["factor_nonzeros_combined_diagonal"] = plan_.factor_nonzeros();
        statistics_.values["factor_fill_ratio"] =
            static_cast<double>(plan_.factor_nonzeros()) / plan_.input_nonzeros;
        statistics_.values["analysis_plan_bytes"] = plan_.storage_bytes();
        statistics_.values["btf_blocks"] = plan_.block_offsets.size() - 1U;
        statistics_.values["structurally_symmetric_btf_blocks"] = std::count(
            plan_.structurally_symmetric_blocks.begin(),
            plan_.structurally_symmetric_blocks.end(),
            std::uint8_t{1});
        const auto& factor_schedule = factor_kernel_ == FactorKernelKind::right_looking
                                          ? right_looking_schedule()
                                          : plan_.factor_schedule;
        statistics_.values["factor_levels"] = factor_schedule.levels();
        statistics_.values["factor_widest_level"] = factor_schedule.widest_level();
        statistics_.values["forward_levels"] = plan_.forward_schedule.levels();
        statistics_.values["backward_levels"] = plan_.backward_schedule.levels();
        statistics_.values["estimated_factor_flops"] = plan_.estimated_factor_flops;
        auto factor_kernel_launches = factor_schedule.levels();
        if (factor_kernel_ == FactorKernelKind::right_looking) {
            factor_kernel_launches = 1;
            for (SparseIndex level = 0; level < factor_schedule.levels(); ++level) {
                const auto begin = right_looking_plan_.task_level_offsets[
                    static_cast<std::size_t>(level)];
                const auto end = right_looking_plan_.task_level_offsets[
                    static_cast<std::size_t>(level) + 1U];
                factor_kernel_launches += begin != end;
            }
        }
        statistics_.values["factor_kernel_launches"] = factor_kernel_launches;
        statistics_.values["solve_kernel_launches"] = 4;
        statistics_.values["persistent_solve_blocks"] = persistent_solve_blocks_;
        statistics_.attributes["triangular_solve_schedule"] =
            "two persistent warp kernels with dependency completion flags";
        if (factor_kernel_ == FactorKernelKind::dense_workspace) {
            classify_workspace_levels();
            statistics_.values["factor_workspace_levels_128_threads"] =
                workspace_levels_128_threads_;
            statistics_.values["factor_workspace_levels_512_threads"] =
                workspace_levels_512_threads_;
            statistics_.values["factor_workspace_levels_1024_threads"] =
                workspace_levels_1024_threads_;
            statistics_.attributes["factor_thread_policy"] =
                "128/512/1024 threads selected by maximum row updates in each level";
        }
        if (factor_kernel_ == FactorKernelKind::right_looking) {
            statistics_.attributes["schedule"] =
                "exact right-looking dependency levels with atomic fan-out updates";
            statistics_.values["right_looking_tasks"] =
                right_looking_plan_.tasks.size();
            statistics_.values["right_looking_double_u_dependency_edges"] =
                right_looking_plan_.dependency_edges;
            statistics_.values["right_looking_additional_double_u_edges"] =
                right_looking_plan_.additional_double_u_edges;
            statistics_.values["right_looking_plan_bytes"] =
                right_looking_plan_.storage_bytes();
        }
    }

    [[nodiscard]] int workspace_threads_for_range(
        SparseIndex begin,
        SparseIndex end) const noexcept {
        std::uint64_t maximum_updates{};
        for (auto position = begin; position < end; ++position) {
            const auto row = plan_.factor_schedule.rows[static_cast<std::size_t>(position)];
            maximum_updates = std::max(
                maximum_updates,
                plan_.factor_row_updates[static_cast<std::size_t>(row)]);
        }
        if (maximum_updates < 128U) return kFactorThreads;
        if (maximum_updates < 65536U) return kWorkspaceMediumThreads;
        return kWorkspaceHeavyThreads;
    }

    void classify_workspace_levels() noexcept {
        workspace_levels_128_threads_ = 0;
        workspace_levels_512_threads_ = 0;
        workspace_levels_1024_threads_ = 0;
        for (SparseIndex level = 0; level < plan_.factor_schedule.levels(); ++level) {
            const auto begin =
                plan_.factor_schedule.level_offsets[static_cast<std::size_t>(level)];
            const auto end =
                plan_.factor_schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
            switch (workspace_threads_for_range(begin, end)) {
                case kFactorThreads: ++workspace_levels_128_threads_; break;
                case kWorkspaceMediumThreads: ++workspace_levels_512_threads_; break;
                default: ++workspace_levels_1024_threads_; break;
            }
        }
    }

    void configure_persistent_solve(BackendContext& context) {
        auto event = context.profiler.scoped(
            "cuda_persistent_solve_configuration", EventKind::event);
        int device{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties");
        const auto row_limited_blocks = static_cast<int>(std::max<SparseIndex>(
            1,
            (plan_.dimension + kWarpsPerPersistentSolveBlock - 1) /
                kWarpsPerPersistentSolveBlock));
        persistent_solve_blocks_ = std::max(
            1,
            std::min(
                row_limited_blocks,
                properties.multiProcessorCount *
                    kPersistentSolveBlocksPerMultiprocessor));
        context.profiler.add_value(
            "multiprocessor_count", properties.multiProcessorCount);
        context.profiler.add_value("persistent_blocks", persistent_solve_blocks_);
        context.profiler.add_value(
            "persistent_warps",
            persistent_solve_blocks_ * kWarpsPerPersistentSolveBlock);
        statistics_.values["persistent_solve_blocks"] = persistent_solve_blocks_;
        statistics_.values["persistent_solve_warps"] =
            persistent_solve_blocks_ * kWarpsPerPersistentSolveBlock;
    }

    void configure_factor_kernel(BackendContext& context) {
        if (factor_kernel_ != FactorKernelKind::dense_workspace) return;

        auto event = context.profiler.scoped(
            "cuda_workspace_configuration", EventKind::event);
        int device{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties");
        std::size_t free_bytes{};
        std::size_t total_bytes{};
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

        const auto bytes_per_worker =
            static_cast<std::size_t>(plan_.dimension) * sizeof(double);
        // Two logical worker blocks per SM keep the workspace pool bounded;
        // the per-level adaptive policy independently selects 128, 512, or
        // 1024 threads for each worker. Never reserve more than one quarter of
        // currently free memory for this experimental direct-index path.
        const auto target_workers = std::max(
            1,
            properties.multiProcessorCount * kWorkspaceBlocksPerMultiprocessor);
        const auto memory_limited_workers = std::max<std::size_t>(
            1U, (free_bytes / 4U) / bytes_per_worker);
        workspace_worker_blocks_ = static_cast<SparseIndex>(std::min<std::size_t>(
            static_cast<std::size_t>(target_workers), memory_limited_workers));
        workspace_worker_blocks_ = std::min(workspace_worker_blocks_, plan_.dimension);
        workspace_bytes_ =
            static_cast<std::uint64_t>(workspace_worker_blocks_) * bytes_per_worker;

        context.profiler.add_value("multiprocessor_count", properties.multiProcessorCount);
        context.profiler.add_value("workspace_worker_blocks", workspace_worker_blocks_);
        context.profiler.add_value("workspace_bytes", workspace_bytes_);
        context.profiler.add_value("device_free_bytes_before_allocation", free_bytes);
        context.profiler.add_value("device_total_bytes", total_bytes);
        statistics_.values["factor_workspace_worker_blocks"] = workspace_worker_blocks_;
        statistics_.values["factor_workspace_bytes"] = workspace_bytes_;
        statistics_.attributes["workspace_policy"] =
            "two worker blocks per SM, capped at 25% of free device memory";
    }

    template <typename Kernel>
    OccupancyInfo record_kernel_occupancy(
        BackendContext& context,
        std::string_view prefix,
        Kernel kernel,
        int threads_per_block,
        std::size_t dynamic_shared_memory_bytes = 0U) {
        cudaFuncAttributes attributes{};
        check_cuda(
            cudaFuncGetAttributes(&attributes, kernel),
            "cudaFuncGetAttributes(occupancy)");
        int active_blocks{};
        check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                kernel,
                threads_per_block,
                dynamic_shared_memory_bytes),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

        int device{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice(occupancy)");
        cudaDeviceProp properties{};
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties(occupancy)");
        const auto warps_per_block =
            (threads_per_block + properties.warpSize - 1) / properties.warpSize;
        const auto maximum_warps =
            properties.maxThreadsPerMultiProcessor / properties.warpSize;
        const auto active_warps = std::min(
            maximum_warps, active_blocks * warps_per_block);
        const auto occupancy = maximum_warps == 0
                                   ? 0.0
                                   : 100.0 * static_cast<double>(active_warps) /
                                         static_cast<double>(maximum_warps);

        const auto key = [&](std::string_view suffix) {
            return std::string(prefix) + std::string(suffix);
        };
        const std::pair<std::string_view, double> values[]{
            {"_threads_per_block", static_cast<double>(threads_per_block)},
            {"_registers_per_thread", static_cast<double>(attributes.numRegs)},
            {"_static_shared_memory_bytes",
             static_cast<double>(attributes.sharedSizeBytes)},
            {"_dynamic_shared_memory_bytes",
             static_cast<double>(dynamic_shared_memory_bytes)},
            {"_active_blocks_per_sm", static_cast<double>(active_blocks)},
            {"_active_warps_per_sm", static_cast<double>(active_warps)},
            {"_maximum_warps_per_sm", static_cast<double>(maximum_warps)},
            {"_theoretical_occupancy_percent", occupancy},
        };
        for (const auto& [suffix, value] : values) {
            context.profiler.add_value(key(suffix), value);
            statistics_.values[key(suffix)] = value;
        }
        return {
            threads_per_block,
            attributes.numRegs,
            attributes.sharedSizeBytes,
            active_blocks,
            active_warps,
            maximum_warps,
            occupancy,
        };
    }

    void record_factor_grid_saturation(
        BackendContext& context,
        const OccupancyInfo& occupancy,
        const cudaDeviceProp& properties) {
        const auto resident_block_capacity = static_cast<std::uint64_t>(
            occupancy.active_blocks_per_multiprocessor) *
            static_cast<std::uint64_t>(properties.multiProcessorCount);
        if (resident_block_capacity == 0U) return;

        std::uint64_t launches{};
        std::uint64_t underfilled_launches{};
        long double weighted_saturation{};
        std::uint64_t total_blocks{};
        std::uint64_t maximum_blocks{};
        std::uint64_t maximum_blocks_launch_index{};
        std::vector<std::uint64_t> launch_blocks;
        auto observe_launch = [&](std::uint64_t blocks) {
            if (blocks == 0U) return;
            if (blocks > maximum_blocks) {
                maximum_blocks = blocks;
                maximum_blocks_launch_index = launches;
            }
            launch_blocks.push_back(blocks);
            ++launches;
            underfilled_launches += blocks < resident_block_capacity;
            total_blocks += blocks;
            const auto saturation = std::min(
                1.0L,
                static_cast<long double>(blocks) /
                    static_cast<long double>(resident_block_capacity));
            weighted_saturation += static_cast<long double>(blocks) * saturation;
        };

        if (factor_kernel_ == FactorKernelKind::right_looking) {
            for (SparseIndex level = 0;
                 level < right_looking_schedule().levels(); ++level) {
                const auto begin = right_looking_plan_.task_level_offsets[
                    static_cast<std::size_t>(level)];
                const auto end = right_looking_plan_.task_level_offsets[
                    static_cast<std::size_t>(level) + 1U];
                observe_launch(static_cast<std::uint64_t>(end - begin));
            }
        } else if (factor_kernel_ == FactorKernelKind::binary_search) {
            for (SparseIndex level = 0; level < plan_.factor_schedule.levels(); ++level) {
                const auto begin = plan_.factor_schedule.level_offsets[
                    static_cast<std::size_t>(level)];
                const auto end = plan_.factor_schedule.level_offsets[
                    static_cast<std::size_t>(level) + 1U];
                observe_launch(static_cast<std::uint64_t>(end - begin));
            }
        } else {
            // The workspace backend changes block size by level, so a single
            // resident-capacity value would be misleading. Its three resource
            // occupancy configurations are still reported separately.
            return;
        }

        const auto saturated_launches = launches - underfilled_launches;
        const auto launch_saturation_percent = launches == 0U
            ? 0.0
            : 100.0 * static_cast<double>(saturated_launches) /
                  static_cast<double>(launches);
        const auto block_weighted_saturation_percent = total_blocks == 0U
            ? 0.0
            : 100.0 * static_cast<double>(
                  weighted_saturation / static_cast<long double>(total_blocks));
        std::uint64_t weighted_median_launch_index{};
        std::uint64_t weighted_median_launch_blocks{};
        if (total_blocks != 0U) {
            const auto middle = (total_blocks + 1U) / 2U;
            std::uint64_t cumulative{};
            for (std::size_t index = 0; index < launch_blocks.size(); ++index) {
                cumulative += launch_blocks[index];
                if (cumulative >= middle) {
                    weighted_median_launch_index = index;
                    weighted_median_launch_blocks = launch_blocks[index];
                    break;
                }
            }
        }
        const std::pair<std::string_view, double> values[]{
            {"factor_grid_resident_block_capacity",
             static_cast<double>(resident_block_capacity)},
            {"factor_grid_profiled_launches", static_cast<double>(launches)},
            {"factor_grid_underfilled_launches",
             static_cast<double>(underfilled_launches)},
            {"factor_grid_saturated_launches",
             static_cast<double>(saturated_launches)},
            {"factor_grid_saturated_launch_percent", launch_saturation_percent},
            {"factor_grid_block_weighted_saturation_percent",
             block_weighted_saturation_percent},
            {"factor_grid_maximum_blocks", static_cast<double>(maximum_blocks)},
            {"factor_grid_maximum_blocks_launch_index",
             static_cast<double>(maximum_blocks_launch_index)},
            {"factor_grid_block_weighted_median_launch_index",
             static_cast<double>(weighted_median_launch_index)},
            {"factor_grid_block_weighted_median_launch_blocks",
             static_cast<double>(weighted_median_launch_blocks)},
        };
        for (const auto& [name, value] : values) {
            context.profiler.add_value(std::string(name), value);
            statistics_.values[std::string(name)] = value;
        }
    }

    void configure_kernel_occupancy(BackendContext& context) {
        auto event = context.profiler.scoped(
            "cuda_kernel_occupancy", EventKind::event);
        context.profiler.add_attribute(
            "metric_kind",
            "CUDA runtime theoretical resource occupancy; not achieved occupancy");
        context.profiler.add_attribute(
            "grid_saturation_kind",
            "launch geometry ceiling; not a measured SM activity counter");
        statistics_.attributes["occupancy_metric_kind"] =
            "theoretical resource occupancy from CUDA runtime";
        statistics_.attributes["achieved_occupancy_collection"] =
            "Nsight Compute or CUPTI required";
        statistics_.attributes["vram_bandwidth_collection"] =
            "Nsight Compute or CUPTI required";
        statistics_.attributes["vram_bandwidth_metric"] =
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed";
        statistics_.attributes["memory_pipeline_metric"] =
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed";

        int device{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice(occupancy configuration)");
        cudaDeviceProp properties{};
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties(occupancy configuration)");
        statistics_.values["occupancy_multiprocessor_count"] =
            properties.multiProcessorCount;
        statistics_.values["occupancy_warp_size"] = properties.warpSize;
        statistics_.values["occupancy_max_threads_per_sm"] =
            properties.maxThreadsPerMultiProcessor;
        context.profiler.add_value(
            "multiprocessor_count", properties.multiProcessorCount);
        context.profiler.add_value("warp_size", properties.warpSize);
        context.profiler.add_value(
            "maximum_threads_per_sm", properties.maxThreadsPerMultiProcessor);

        record_kernel_occupancy(
            context,
            "value_initialization_kernel",
            initialize_lu_values_kernel,
            kVectorThreads);
        record_kernel_occupancy(
            context,
            "rhs_permutation_kernel",
            permute_and_scale_rhs_kernel,
            kVectorThreads);
        record_kernel_occupancy(
            context,
            "solution_inverse_permutation_kernel",
            inverse_permute_solution_kernel,
            kVectorThreads);
        const auto forward = record_kernel_occupancy(
            context,
            "forward_solve_kernel",
            triangular_solve_persistent_kernel<false>,
            kPersistentSolveThreads);
        static_cast<void>(forward);
        record_kernel_occupancy(
            context,
            "backward_solve_kernel",
            triangular_solve_persistent_kernel<true>,
            kPersistentSolveThreads);

        OccupancyInfo factor_occupancy{};
        if (factor_kernel_ == FactorKernelKind::right_looking) {
            factor_occupancy = record_kernel_occupancy(
                context,
                "factor_kernel",
                numerical_lu_right_looking_level_kernel,
                kFactorThreads);
            record_kernel_occupancy(
                context,
                "diagonal_validation_kernel",
                validate_diagonal_kernel,
                kVectorThreads);
            record_factor_grid_saturation(context, factor_occupancy, properties);
        } else if (factor_kernel_ == FactorKernelKind::binary_search) {
            factor_occupancy = record_kernel_occupancy(
                context,
                "factor_kernel",
                numerical_lu_level_kernel,
                kFactorThreads);
            record_factor_grid_saturation(context, factor_occupancy, properties);
        } else {
            record_kernel_occupancy(
                context,
                "factor_kernel_128",
                numerical_lu_workspace_level_kernel,
                kFactorThreads);
            record_kernel_occupancy(
                context,
                "factor_kernel_512",
                numerical_lu_workspace_level_kernel,
                kWorkspaceMediumThreads);
            record_kernel_occupancy(
                context,
                "factor_kernel_1024",
                numerical_lu_workspace_level_kernel,
                kWorkspaceHeavyThreads);
        }
    }

    void update_device_statistics() {
        if (cuda_executor_ == nullptr) return;
        const auto& executor = cuda_executor_->statistics();
        statistics_.values["device_live_bytes"] = executor.live_bytes;
        statistics_.values["device_peak_bytes"] = executor.peak_bytes;
    }

    void reset_device() {
        buffers_ = {};
        cuda_executor_.reset();
    }

    CpuSymbolicAnalyzer analyzer_;
    FactorKernelKind factor_kernel_;
    AnalysisPlan plan_;
    RightLookingPlan right_looking_plan_;
    const CscMatrix* matrix_{};
    BackendStatistics statistics_;
    std::unique_ptr<Executor> cuda_executor_;
    DeviceBuffers buffers_;
    SparseIndex workspace_worker_blocks_{};
    std::uint64_t workspace_bytes_{};
    SparseIndex workspace_levels_128_threads_{};
    SparseIndex workspace_levels_512_threads_{};
    SparseIndex workspace_levels_1024_threads_{};
    int persistent_solve_blocks_{1};
};

}  // namespace

std::unique_ptr<Task1Backend> make_cuda_lu_backend() {
    return std::make_unique<CudaLuBackend>(FactorKernelKind::binary_search);
}

std::unique_ptr<Task1Backend> make_cuda_workspace_lu_backend() {
    return std::make_unique<CudaLuBackend>(FactorKernelKind::dense_workspace);
}

std::unique_ptr<Task1Backend> make_cuda_right_looking_lu_backend() {
    return std::make_unique<CudaLuBackend>(FactorKernelKind::right_looking);
}

}  // namespace eda_gpu
