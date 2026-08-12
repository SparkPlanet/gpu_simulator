#pragma once

#include "core/profile.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace eda_gpu::operators::cuda::detail {

// Kernel launch geometry is centralized here so the orchestration layer and
// occupancy calculations cannot silently diverge.
inline constexpr int kSolveThreads = 256;
inline constexpr int kRefactorThreads = 1024;
inline constexpr int kRightLookingThreads = 64;
inline constexpr int kPersistentBlocksPerMultiprocessor = 2;
inline constexpr int kRightLookingBlocksPerMultiprocessor = 12;
inline constexpr int kIndexBuildThreads = 256;
inline constexpr int kIndexBuildBlocks = 4096;
inline constexpr std::int32_t kRefactorFusedLevelWidth = 1;

[[noreturn]] inline void throw_cuda(
    cudaError_t status,
    const char* operation) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
}

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw_cuda(status, operation);
}

template <typename T>
[[nodiscard]] inline std::uint64_t byte_count(
    const std::vector<T>& values) {
    return static_cast<std::uint64_t>(values.size()) * sizeof(T);
}

// Owns every CUDA allocation, stream and timing event used by the custom LU
// backend. Keeping ownership in one RAII object makes initialize() rollback
// safe and keeps the public operator focused on algorithm orchestration.
struct DeviceState {
    // Matrix sizes and profiler metadata.
    std::int32_t dimension{};
    std::int32_t input_nonzeros{};
    std::int32_t lower_nonzeros{};
    std::int32_t upper_nonzeros{};
    std::uint64_t allocated_bytes{};
    std::int32_t schedule_operations{};
    std::int32_t widest_column{};
    std::int32_t narrow_columns{};
    std::int32_t forward_level_count{};
    std::int32_t backward_level_count{};
    std::int32_t forward_group_count{};
    std::int32_t backward_group_count{};
    std::int32_t refactor_level_count{};
    std::int32_t refactor_group_count{};
    std::int32_t widest_level{};
    std::int32_t narrow_levels{};
    std::int32_t cooperative_grid_blocks{};
    std::int32_t cooperative_refactor_grid_blocks{};
    std::uint64_t refactor_workspace_elements{};
    core::NumericFactorDagProfile refactor_dag;

    // Selected numerical-refactor mode and compact-index sizes.
    bool right_looking_refactor{};
    bool right_looking_compact_indices{};
    std::int32_t right_looking_level_count{};
    std::int32_t right_looking_task_count{};
    std::int32_t right_looking_u16_index_count{};
    std::int32_t right_looking_u32_index_count{};

    // Device-resident matrix, factor and permutation data.
    std::int32_t* input_column_offsets{};
    std::int32_t* input_row_indices{};
    double* input_values{};
    std::int32_t* lower_column_offsets{};
    std::int32_t* lower_row_indices{};
    double* lower_values{};
    std::int32_t* upper_column_offsets{};
    std::int32_t* upper_row_indices{};
    double* upper_values{};
    std::int32_t* row_permutation{};
    std::int32_t* inverse_row_permutation{};
    std::int32_t* column_permutation{};

    // Forward/backward solve schedules.
    std::int32_t* forward_level_offsets{};
    std::int32_t* forward_columns{};
    std::int32_t* forward_group_offsets{};
    std::int32_t* backward_level_offsets{};
    std::int32_t* backward_columns{};
    std::int32_t* backward_group_offsets{};

    // Left-looking refactor schedule and per-block dense workspaces.
    std::int32_t* refactor_level_offsets{};
    std::int32_t* refactor_columns{};
    std::int32_t* refactor_group_offsets{};
    double* refactor_workspaces{};

    // Right-looking GLU3 schedule and optional direct destination indices.
    std::int32_t* input_factor_destinations{};
    std::int32_t* right_looking_level_offsets{};
    std::int32_t* right_looking_level_columns{};
    std::int32_t* right_looking_task_offsets{};
    std::int32_t* right_looking_task_sources{};
    std::int32_t* right_looking_task_targets{};
    std::int32_t* right_looking_task_upper_positions{};
    std::int32_t* right_looking_task_index_offsets{};
    std::uint16_t* right_looking_indices_u16{};
    std::uint32_t* right_looking_indices_u32{};

    // Per-iteration vectors and singularity status.
    double* right_hand_side{};
    double* solution{};
    double* workspace{};
    std::int32_t* singular_column{};

    cudaStream_t stream{};
    cudaEvent_t start{};
    cudaEvent_t after_h2d{};
    cudaEvent_t after_kernel{};
    cudaEvent_t after_d2h{};

    DeviceState() {
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags(custom LU)");
        try {
            check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
            check_cuda(cudaEventCreate(&after_h2d),
                       "cudaEventCreate(after_h2d)");
            check_cuda(cudaEventCreate(&after_kernel),
                       "cudaEventCreate(after_kernel)");
            check_cuda(cudaEventCreate(&after_d2h),
                       "cudaEventCreate(after_d2h)");
        } catch (...) {
            release_events();
            if (stream != nullptr) cudaStreamDestroy(stream);
            stream = nullptr;
            throw;
        }
    }

    ~DeviceState() {
        release_buffers();
        release_events();
        if (stream != nullptr) cudaStreamDestroy(stream);
    }

    DeviceState(const DeviceState&) = delete;
    DeviceState& operator=(const DeviceState&) = delete;

    template <typename T>
    void allocate(T*& pointer, std::size_t count, const char* operation) {
        if (count == 0U) return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::runtime_error(std::string(operation) + " size overflow");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
                              count * sizeof(T)),
                   operation);
        allocated_bytes += static_cast<std::uint64_t>(count) * sizeof(T);
    }

    template <typename T>
    void copy_to_device(
        T* destination,
        const std::vector<T>& source,
        const char* operation) {
        if (source.empty()) return;
        check_cuda(cudaMemcpyAsync(destination, source.data(),
                                   static_cast<std::size_t>(byte_count(source)),
                                   cudaMemcpyHostToDevice, stream),
                   operation);
    }

    template <typename T>
    void release(T*& pointer) noexcept {
        if (pointer != nullptr) cudaFree(pointer);
        pointer = nullptr;
    }

    void release_buffers() noexcept {
        release(input_column_offsets);
        release(input_row_indices);
        release(input_values);
        release(lower_column_offsets);
        release(lower_row_indices);
        release(lower_values);
        release(upper_column_offsets);
        release(upper_row_indices);
        release(upper_values);
        release(row_permutation);
        release(inverse_row_permutation);
        release(column_permutation);
        release(forward_level_offsets);
        release(forward_columns);
        release(forward_group_offsets);
        release(backward_level_offsets);
        release(backward_columns);
        release(backward_group_offsets);
        release(refactor_level_offsets);
        release(refactor_columns);
        release(refactor_group_offsets);
        release(refactor_workspaces);
        release(input_factor_destinations);
        release(right_looking_level_offsets);
        release(right_looking_level_columns);
        release(right_looking_task_offsets);
        release(right_looking_task_sources);
        release(right_looking_task_targets);
        release(right_looking_task_upper_positions);
        release(right_looking_task_index_offsets);
        release(right_looking_indices_u16);
        release(right_looking_indices_u32);
        release(right_hand_side);
        release(solution);
        release(workspace);
        release(singular_column);

        dimension = 0;
        input_nonzeros = 0;
        lower_nonzeros = 0;
        upper_nonzeros = 0;
        allocated_bytes = 0;
        schedule_operations = 0;
        widest_column = 0;
        narrow_columns = 0;
        forward_level_count = 0;
        backward_level_count = 0;
        forward_group_count = 0;
        backward_group_count = 0;
        refactor_level_count = 0;
        refactor_group_count = 0;
        widest_level = 0;
        narrow_levels = 0;
        cooperative_grid_blocks = 0;
        cooperative_refactor_grid_blocks = 0;
        refactor_workspace_elements = 0;
        refactor_dag = {};
        right_looking_refactor = false;
        right_looking_compact_indices = false;
        right_looking_level_count = 0;
        right_looking_task_count = 0;
        right_looking_u16_index_count = 0;
        right_looking_u32_index_count = 0;
    }

    void release_events() noexcept {
        if (start != nullptr) cudaEventDestroy(start);
        if (after_h2d != nullptr) cudaEventDestroy(after_h2d);
        if (after_kernel != nullptr) cudaEventDestroy(after_kernel);
        if (after_d2h != nullptr) cudaEventDestroy(after_d2h);
        start = nullptr;
        after_h2d = nullptr;
        after_kernel = nullptr;
        after_d2h = nullptr;
    }

    [[nodiscard]] double elapsed(cudaEvent_t begin, cudaEvent_t end) const {
        float milliseconds{};
        check_cuda(cudaEventElapsedTime(&milliseconds, begin, end),
                   "cudaEventElapsedTime(custom LU)");
        return static_cast<double>(milliseconds);
    }
};

}  // namespace eda_gpu::operators::cuda::detail
