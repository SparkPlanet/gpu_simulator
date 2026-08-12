#include "analysis.hpp"
#include "backend.hpp"
#include "cuda_executor.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eda_gpu {
namespace {

constexpr int kFactorThreads = 128;
constexpr int kVectorThreads = 256;

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

__global__ void forward_solve_level_kernel(
    const SparseIndex* scheduled_rows,
    SparseIndex row_count,
    const SparseIndex* row_offsets,
    const SparseIndex* columns,
    const SparseIndex* diagonal_positions,
    const double* factor_values,
    const double* right_hand_side,
    double* intermediate) {
    const auto scheduled = static_cast<SparseIndex>(blockIdx.x * blockDim.x + threadIdx.x);
    if (scheduled >= row_count) return;
    const auto row = scheduled_rows[scheduled];
    auto value = right_hand_side[row];
    for (auto position = row_offsets[row]; position < diagonal_positions[row]; ++position) {
        value -= factor_values[position] * intermediate[columns[position]];
    }
    intermediate[row] = value;
}

__global__ void backward_solve_level_kernel(
    const SparseIndex* scheduled_rows,
    SparseIndex row_count,
    const SparseIndex* row_offsets,
    const SparseIndex* columns,
    const SparseIndex* diagonal_positions,
    const double* factor_values,
    const double* intermediate,
    double* permuted_solution) {
    const auto scheduled = static_cast<SparseIndex>(blockIdx.x * blockDim.x + threadIdx.x);
    if (scheduled >= row_count) return;
    const auto row = scheduled_rows[scheduled];
    const auto diagonal = diagonal_positions[row];
    auto value = intermediate[row];
    for (auto position = diagonal + 1; position < row_offsets[row + 1]; ++position) {
        value -= factor_values[position] * permuted_solution[columns[position]];
    }
    permuted_solution[row] = value / factor_values[diagonal];
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
    Allocation right_hand_side;
    Allocation permuted_right_hand_side;
    Allocation intermediate;
    Allocation permuted_solution;
    Allocation solution;
    Allocation status;
};

class CudaLuBackend final : public Task1Backend {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "cuda-lu"; }

    void analyze(const CscMatrix& matrix, BackendContext& context) override {
        reset_device();
        matrix_ = nullptr;
        plan_ = analyzer_.analyze(matrix, context.profiler);
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
        allocate<double>(buffers_.right_hand_side, static_cast<std::size_t>(plan_.dimension));
        allocate<double>(buffers_.permuted_right_hand_side,
                         static_cast<std::size_t>(plan_.dimension));
        allocate<double>(buffers_.intermediate, static_cast<std::size_t>(plan_.dimension));
        allocate<double>(buffers_.permuted_solution,
                         static_cast<std::size_t>(plan_.dimension));
        allocate<double>(buffers_.solution, static_cast<std::size_t>(plan_.dimension));
        allocate<int>(buffers_.status, 1U);
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
        context.profiler.add_attribute("lookup", "binary-search");
        context.profiler.add_attribute("schedule", "one kernel launch per factor level");
        context.profiler.add_estimated_flops(plan_.estimated_factor_flops);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        const auto* scheduled_rows =
            device_pointer<SparseIndex>(buffers_.factor_schedule_rows);
        for (SparseIndex level = 0; level < plan_.factor_schedule.levels(); ++level) {
            const auto begin =
                plan_.factor_schedule.level_offsets[static_cast<std::size_t>(level)];
            const auto end =
                plan_.factor_schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
            const auto width = end - begin;
            numerical_lu_level_kernel<<<width, kFactorThreads>>>(
                scheduled_rows + begin,
                width,
                device_pointer<SparseIndex>(buffers_.row_offsets),
                device_pointer<SparseIndex>(buffers_.columns),
                device_pointer<SparseIndex>(buffers_.diagonal_positions),
                device_pointer<double>(buffers_.factor_values),
                64.0 * std::numeric_limits<double>::epsilon(),
                device_pointer<int>(buffers_.status));
        }
        check_cuda(cudaGetLastError(), "numerical_lu_level_kernel launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", plan_.factor_schedule.levels());
        context.profiler.add_value("levels", plan_.factor_schedule.levels());
        context.profiler.add_value("widest_level", plan_.factor_schedule.widest_level());
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
        context.profiler.add_attribute("schedule", "one kernel launch per forward level");
        context.profiler.add_estimated_flops(2.0 * plan_.lower_nonzeros());
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        const auto* scheduled_rows =
            device_pointer<SparseIndex>(buffers_.forward_schedule_rows);
        for (SparseIndex level = 0; level < plan_.forward_schedule.levels(); ++level) {
            const auto begin =
                plan_.forward_schedule.level_offsets[static_cast<std::size_t>(level)];
            const auto end =
                plan_.forward_schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
            const auto width = end - begin;
            const auto blocks = (width + kVectorThreads - 1) / kVectorThreads;
            forward_solve_level_kernel<<<blocks, kVectorThreads>>>(
                scheduled_rows + begin,
                width,
                device_pointer<SparseIndex>(buffers_.row_offsets),
                device_pointer<SparseIndex>(buffers_.columns),
                device_pointer<SparseIndex>(buffers_.diagonal_positions),
                device_pointer<double>(buffers_.factor_values),
                device_pointer<double>(buffers_.permuted_right_hand_side),
                device_pointer<double>(buffers_.intermediate));
        }
        check_cuda(cudaGetLastError(), "forward_solve_level_kernel launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", plan_.forward_schedule.levels());
    }

    void run_backward_solve(BackendContext& context) {
        auto event = context.profiler.scoped("cuda_backward_solve", EventKind::kernel);
        context.profiler.add_attribute("schedule", "one kernel launch per backward level");
        context.profiler.add_estimated_flops(
            2.0 * plan_.upper_nonzeros() + plan_.dimension);
        CudaEventTimer timer(context.profiler, *cuda_executor_);
        const auto* scheduled_rows =
            device_pointer<SparseIndex>(buffers_.backward_schedule_rows);
        for (SparseIndex level = 0; level < plan_.backward_schedule.levels(); ++level) {
            const auto begin =
                plan_.backward_schedule.level_offsets[static_cast<std::size_t>(level)];
            const auto end =
                plan_.backward_schedule.level_offsets[static_cast<std::size_t>(level) + 1U];
            const auto width = end - begin;
            const auto blocks = (width + kVectorThreads - 1) / kVectorThreads;
            backward_solve_level_kernel<<<blocks, kVectorThreads>>>(
                scheduled_rows + begin,
                width,
                device_pointer<SparseIndex>(buffers_.row_offsets),
                device_pointer<SparseIndex>(buffers_.columns),
                device_pointer<SparseIndex>(buffers_.diagonal_positions),
                device_pointer<double>(buffers_.factor_values),
                device_pointer<double>(buffers_.intermediate),
                device_pointer<double>(buffers_.permuted_solution));
        }
        check_cuda(cudaGetLastError(), "backward_solve_level_kernel launch");
        timer.finish();
        context.profiler.add_value("kernel_launches", plan_.backward_schedule.levels());
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
        statistics_.attributes["algorithm"] = "CUDA fixed-pivot row-oriented sparse LU";
        statistics_.attributes["analysis"] =
            "CPU BTF+AMD ordering+self-owned symbolic LU";
        statistics_.attributes["execution_space"] = "CUDA";
        statistics_.attributes["factor_storage"] = "combined CSR L/U";
        statistics_.attributes["lookup"] = "binary-search";
        statistics_.attributes["pivoting"] = "static diagonal pivot";
        statistics_.attributes["schedule"] = "level-scheduled kernels";
        statistics_.values["input_nonzeros"] = plan_.input_nonzeros;
        statistics_.values["factor_nonzeros_combined_diagonal"] = plan_.factor_nonzeros();
        statistics_.values["factor_fill_ratio"] =
            static_cast<double>(plan_.factor_nonzeros()) / plan_.input_nonzeros;
        statistics_.values["analysis_plan_bytes"] = plan_.storage_bytes();
        statistics_.values["factor_levels"] = plan_.factor_schedule.levels();
        statistics_.values["factor_widest_level"] = plan_.factor_schedule.widest_level();
        statistics_.values["forward_levels"] = plan_.forward_schedule.levels();
        statistics_.values["backward_levels"] = plan_.backward_schedule.levels();
        statistics_.values["estimated_factor_flops"] = plan_.estimated_factor_flops;
        statistics_.values["factor_kernel_launches"] = plan_.factor_schedule.levels();
        statistics_.values["solve_kernel_launches"] =
            plan_.forward_schedule.levels() + plan_.backward_schedule.levels() + 2;
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
    AnalysisPlan plan_;
    const CscMatrix* matrix_{};
    BackendStatistics statistics_;
    std::unique_ptr<Executor> cuda_executor_;
    DeviceBuffers buffers_;
};

}  // namespace

std::unique_ptr<Task1Backend> make_cuda_lu_backend() {
    return std::make_unique<CudaLuBackend>();
}

}  // namespace eda_gpu
