#pragma once

#include "operators/cuda/fixed_pattern_lu_runtime.cuh"

#include <cooperative_groups.h>

#include <algorithm>
#include <cstdint>

namespace eda_gpu::operators::cuda::detail {

namespace cg = cooperative_groups;

// --------------------------------------------------------------------------
// Numerical refactor layer: left-looking correctness/fallback implementations
// --------------------------------------------------------------------------

// Fixed-pattern left-looking LU. The symbolic L/U structure and pivot
// permutations are immutable. One block owns a column at a time and fuses
// scatter, sparse updates and gather without launching one kernel per column.
__global__ void refactor_fixed_pattern_kernel(
    std::int32_t dimension,
    const std::int32_t* input_column_offsets,
    const std::int32_t* input_row_indices,
    const double* input_values,
    const std::int32_t* lower_column_offsets,
    const std::int32_t* lower_row_indices,
    double* lower_values,
    const std::int32_t* upper_column_offsets,
    const std::int32_t* upper_row_indices,
    double* upper_values,
    const std::int32_t* inverse_row_permutation,
    const std::int32_t* column_permutation,
    double* workspace,
    std::int32_t* singular_column) {
    __shared__ std::int32_t update_column;
    __shared__ double update_value;
    __shared__ double pivot;

    const auto lane = static_cast<std::int32_t>(threadIdx.x);
    for (std::int32_t column = 0; column < dimension; ++column) {
        const auto input_column = column_permutation[column];
        const auto input_begin = input_column_offsets[input_column];
        const auto input_end = input_column_offsets[input_column + 1];
        for (auto position = input_begin + lane; position < input_end;
             position += static_cast<std::int32_t>(blockDim.x)) {
            const auto permuted_row =
                inverse_row_permutation[input_row_indices[position]];
            workspace[permuted_row] = input_values[position];
        }
        __syncthreads();

        const auto upper_begin = upper_column_offsets[column];
        const auto upper_end = upper_column_offsets[column + 1];
        for (auto position = upper_begin; position + 1 < upper_end; ++position) {
            if (lane == 0) {
                update_column = upper_row_indices[position];
                update_value = workspace[update_column];
                upper_values[position] = update_value;
                workspace[update_column] = 0.0;
            }
            __syncthreads();

            const auto lower_begin = lower_column_offsets[update_column];
            const auto lower_end = lower_column_offsets[update_column + 1];
            for (auto lower_position = lower_begin + 1 + lane;
                 lower_position < lower_end;
                 lower_position += static_cast<std::int32_t>(blockDim.x)) {
                const auto row = lower_row_indices[lower_position];
                workspace[row] -=
                    lower_values[lower_position] * update_value;
            }
            __syncthreads();
        }

        if (lane == 0) {
            const auto candidate = workspace[column];
            const bool invalid = candidate == 0.0 || !isfinite(candidate);
            if (invalid && *singular_column < 0) *singular_column = column;
            pivot = invalid ? 1.0 : candidate;
            upper_values[upper_end - 1] = candidate;
            workspace[column] = 0.0;
        }
        __syncthreads();

        const auto lower_begin = lower_column_offsets[column];
        const auto lower_end = lower_column_offsets[column + 1];
        for (auto position = lower_begin + 1 + lane; position < lower_end;
             position += static_cast<std::int32_t>(blockDim.x)) {
            const auto row = lower_row_indices[position];
            lower_values[position] = workspace[row] / pivot;
            workspace[row] = 0.0;
        }
        __syncthreads();
    }
}

// Factors one column using a dense row-addressable workspace. This helper is
// shared by the serial and level-persistent left-looking paths.
__device__ __forceinline__ void refactor_dense_column(
    std::int32_t column,
    const std::int32_t* input_column_offsets,
    const std::int32_t* input_row_indices,
    const double* input_values,
    const std::int32_t* lower_column_offsets,
    const std::int32_t* lower_row_indices,
    double* lower_values,
    const std::int32_t* upper_column_offsets,
    const std::int32_t* upper_row_indices,
    double* upper_values,
    const std::int32_t* inverse_row_permutation,
    const std::int32_t* column_permutation,
    double* workspace,
    std::int32_t* singular_column,
    std::int32_t* update_column,
    double* update_value,
    double* pivot) {
    const auto lane = static_cast<std::int32_t>(threadIdx.x);
    const auto group_threads = static_cast<std::int32_t>(blockDim.x);
    const auto input_column = column_permutation[column];
    const auto input_begin = input_column_offsets[input_column];
    const auto input_end = input_column_offsets[input_column + 1];
    for (auto position = input_begin + lane; position < input_end;
         position += group_threads) {
        const auto permuted_row =
            inverse_row_permutation[input_row_indices[position]];
        workspace[permuted_row] = input_values[position];
    }
    __syncthreads();

    const auto upper_begin = upper_column_offsets[column];
    const auto upper_end = upper_column_offsets[column + 1];
    for (auto position = upper_begin; position + 1 < upper_end; ++position) {
        if (lane == 0) {
            *update_column = upper_row_indices[position];
            *update_value = workspace[*update_column];
            upper_values[position] = *update_value;
            workspace[*update_column] = 0.0;
        }
        __syncthreads();

        const auto lower_begin = lower_column_offsets[*update_column];
        const auto lower_end = lower_column_offsets[*update_column + 1];
        for (auto lower_position = lower_begin + 1 + lane;
             lower_position < lower_end;
             lower_position += group_threads) {
            const auto row = lower_row_indices[lower_position];
            workspace[row] -=
                lower_values[lower_position] * *update_value;
        }
        __syncthreads();
    }

    if (lane == 0) {
        const auto candidate = workspace[column];
        const bool invalid = candidate == 0.0 || !isfinite(candidate);
        if (invalid) atomicCAS(singular_column, -1, column);
        *pivot = invalid ? 1.0 : candidate;
        upper_values[upper_end - 1] = candidate;
        workspace[column] = 0.0;
    }
    __syncthreads();

    const auto lower_begin = lower_column_offsets[column];
    const auto lower_end = lower_column_offsets[column + 1];
    for (auto position = lower_begin + 1 + lane; position < lower_end;
         position += group_threads) {
        const auto row = lower_row_indices[position];
        lower_values[position] = workspace[row] / *pivot;
        workspace[row] = 0.0;
    }
    __syncthreads();
}

// Each resident block owns a dense workspace and factors independent columns
// from the same U-dependency level. Narrow consecutive levels are fused into
// one block; wide levels are distributed over the cooperative grid.
__global__ void refactor_persistent_dense_kernel(
    std::int32_t dimension,
    const std::int32_t* input_column_offsets,
    const std::int32_t* input_row_indices,
    const double* input_values,
    const std::int32_t* lower_column_offsets,
    const std::int32_t* lower_row_indices,
    double* lower_values,
    const std::int32_t* upper_column_offsets,
    const std::int32_t* upper_row_indices,
    double* upper_values,
    const std::int32_t* inverse_row_permutation,
    const std::int32_t* column_permutation,
    const std::int32_t* level_offsets,
    const std::int32_t* level_columns,
    const std::int32_t* group_offsets,
    std::int32_t group_count,
    double* workspaces,
    std::int32_t* singular_column) {
    const auto grid = cg::this_grid();
    __shared__ std::int32_t update_column;
    __shared__ double update_value;
    __shared__ double pivot;

    for (std::int32_t group = 0; group < group_count; ++group) {
        const auto level_begin = group_offsets[group];
        const auto level_end = group_offsets[group + 1];
        const auto first_width = level_offsets[level_begin + 1] -
                                 level_offsets[level_begin];
        if (first_width <= kRefactorFusedLevelWidth) {
            if (blockIdx.x == 0) {
                for (auto level = level_begin; level < level_end; ++level) {
                    const auto begin = level_offsets[level];
                    const auto end = level_offsets[level + 1];
                    for (auto task = begin; task < end; ++task) {
                        refactor_dense_column(
                            level_columns[task], input_column_offsets,
                            input_row_indices, input_values,
                            lower_column_offsets, lower_row_indices,
                            lower_values, upper_column_offsets,
                            upper_row_indices, upper_values,
                            inverse_row_permutation, column_permutation,
                            workspaces, singular_column, &update_column,
                            &update_value, &pivot);
                    }
                }
            }
        } else {
            const auto begin = level_offsets[level_begin];
            const auto end = level_offsets[level_begin + 1];
            auto* workspace = workspaces +
                static_cast<std::uint64_t>(blockIdx.x) *
                    static_cast<std::uint64_t>(dimension);
            for (auto task = begin + static_cast<std::int32_t>(blockIdx.x);
                 task < end;
                 task += static_cast<std::int32_t>(gridDim.x)) {
                refactor_dense_column(
                    level_columns[task], input_column_offsets,
                    input_row_indices, input_values, lower_column_offsets,
                    lower_row_indices, lower_values, upper_column_offsets,
                    upper_row_indices, upper_values, inverse_row_permutation,
                    column_permutation, workspace, singular_column,
                    &update_column, &update_value, &pivot);
            }
        }
        grid.sync();
    }
}

// --------------------------------------------------------------------------
// Symbolic destination-index layer for the right-looking numerical kernel
// --------------------------------------------------------------------------

__device__ __forceinline__ std::int32_t find_sorted_row(
    const std::int32_t* rows,
    std::int32_t begin,
    std::int32_t end,
    std::int32_t row) {
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        const auto candidate = rows[middle];
        if (candidate < row) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin;
}

// Builds one reusable destination offset for every scalar subcolumn update.
// A non-negative task offset addresses the uint16 array; a negative encoded
// offset addresses the uint32 array. This setup runs once per fixed pattern.
__global__ void build_right_looking_indices_kernel(
    std::int32_t task_count,
    const std::int32_t* task_sources,
    const std::int32_t* task_targets,
    const std::int32_t* task_index_offsets,
    const std::int32_t* lower_column_offsets,
    const std::int32_t* lower_row_indices,
    const std::int32_t* upper_column_offsets,
    const std::int32_t* upper_row_indices,
    std::uint16_t* indices_u16,
    std::uint32_t* indices_u32,
    std::int32_t* failure_column) {
    const auto lane = static_cast<std::int32_t>(threadIdx.x);
    for (auto task = static_cast<std::int32_t>(blockIdx.x);
         task < task_count;
         task += static_cast<std::int32_t>(gridDim.x)) {
        const auto source = task_sources[task];
        const auto target = task_targets[task];
        const auto encoded_offset = task_index_offsets[task];
        const auto lower_begin = lower_column_offsets[source] + 1;
        const auto lower_end = lower_column_offsets[source + 1];
        const auto target_upper_begin = upper_column_offsets[target];
        const auto target_upper_end = upper_column_offsets[target + 1];
        const auto target_upper_nonzeros =
            target_upper_end - target_upper_begin;
        const auto target_lower_begin = lower_column_offsets[target];
        const auto target_lower_end = lower_column_offsets[target + 1];

        for (auto lower_position = lower_begin + lane;
             lower_position < lower_end;
             lower_position += static_cast<std::int32_t>(blockDim.x)) {
            const auto row = lower_row_indices[lower_position];
            std::int32_t relative_position{-1};
            if (row <= target) {
                const auto destination = find_sorted_row(
                    upper_row_indices, target_upper_begin,
                    target_upper_end, row);
                if (destination < target_upper_end &&
                    upper_row_indices[destination] == row) {
                    relative_position = destination - target_upper_begin;
                }
            } else {
                const auto destination = find_sorted_row(
                    lower_row_indices, target_lower_begin,
                    target_lower_end, row);
                if (destination < target_lower_end &&
                    lower_row_indices[destination] == row) {
                    relative_position = target_upper_nonzeros +
                        destination - target_lower_begin - 1;
                }
            }
            if (relative_position < 0) {
                atomicCAS(failure_column, -1, target);
                continue;
            }

            const auto local_position = lower_position - lower_begin;
            if (encoded_offset >= 0) {
                indices_u16[encoded_offset + local_position] =
                    static_cast<std::uint16_t>(relative_position);
            } else {
                indices_u32[-encoded_offset - 1 + local_position] =
                    static_cast<std::uint32_t>(relative_position);
            }
        }
    }
}

// --------------------------------------------------------------------------
// Numerical refactor layer: GLU3-style right-looking implementation
// --------------------------------------------------------------------------

// A cooperative grid processes relaxed dependency levels. First it normalizes
// every pivot column in the level, then distributes outgoing subcolumn updates
// over resident blocks. Compact offsets are the small-case fast path; large
// cases retain the same task graph and locate destinations by sorted search.
__global__ void refactor_right_looking_levels_kernel(
    std::int32_t dimension,
    std::int32_t input_nonzeros,
    const double* input_values,
    const std::int32_t* input_factor_destinations,
    std::int32_t lower_nonzeros,
    const std::int32_t* lower_column_offsets,
    const std::int32_t* lower_row_indices,
    double* lower_values,
    std::int32_t upper_nonzeros,
    const std::int32_t* upper_column_offsets,
    const std::int32_t* upper_row_indices,
    double* upper_values,
    const std::int32_t* level_offsets,
    const std::int32_t* level_columns,
    std::int32_t level_count,
    const std::int32_t* task_offsets,
    const std::int32_t* task_sources,
    const std::int32_t* task_targets,
    const std::int32_t* task_upper_positions,
    std::int32_t compact_indices,
    const std::int32_t* task_index_offsets,
    const std::uint16_t* indices_u16,
    const std::uint32_t* indices_u32,
    std::int32_t* singular_column) {
    const auto grid = cg::this_grid();
    const auto global_thread = static_cast<std::int32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const auto grid_threads =
        static_cast<std::int32_t>(gridDim.x * blockDim.x);
    const auto lane = static_cast<std::int32_t>(threadIdx.x);
    __shared__ double pivot;

    // Clear factor values before scattering the current input matrix into the
    // fixed L/U storage selected by the CPU symbolic stage.
    for (auto position = global_thread; position < lower_nonzeros;
         position += grid_threads) {
        lower_values[position] = 0.0;
    }
    for (auto position = global_thread; position < upper_nonzeros;
         position += grid_threads) {
        upper_values[position] = 0.0;
    }
    grid.sync();

    for (auto position = global_thread; position < input_nonzeros;
         position += grid_threads) {
        const auto destination = input_factor_destinations[position];
        if (destination >= 0) {
            upper_values[destination] = input_values[position];
        } else {
            lower_values[-destination - 1] = input_values[position];
        }
    }
    grid.sync();

    for (std::int32_t level = 0; level < level_count; ++level) {
        // Phase 1: finish pivots and normalize the L columns made ready by all
        // updates from earlier levels.
        const auto column_begin = level_offsets[level];
        const auto column_end = level_offsets[level + 1];
        for (auto position = column_begin +
                 static_cast<std::int32_t>(blockIdx.x);
             position < column_end;
             position += static_cast<std::int32_t>(gridDim.x)) {
            const auto column = level_columns[position];
            const auto upper_end = upper_column_offsets[column + 1];
            if (lane == 0) {
                const auto candidate = upper_values[upper_end - 1];
                const bool invalid = candidate == 0.0 || !isfinite(candidate);
                if (invalid) atomicCAS(singular_column, -1, column);
                pivot = invalid ? 1.0 : candidate;
            }
            __syncthreads();

            const auto lower_begin = lower_column_offsets[column] + 1;
            const auto lower_end = lower_column_offsets[column + 1];
            for (auto lower_position = lower_begin + lane;
                 lower_position < lower_end;
                 lower_position += static_cast<std::int32_t>(blockDim.x)) {
                lower_values[lower_position] /= pivot;
            }
            __syncthreads();
        }
        grid.sync();

        // Phase 2: apply every source-column outer-product update belonging to
        // this level. Atomics resolve different sources writing one target.
        const auto task_begin = task_offsets[level];
        const auto task_end = task_offsets[level + 1];
        for (auto task = task_begin + static_cast<std::int32_t>(blockIdx.x);
             task < task_end;
             task += static_cast<std::int32_t>(gridDim.x)) {
            const auto source = task_sources[task];
            const auto target = task_targets[task];
            const auto multiplier = upper_values[task_upper_positions[task]];
            const auto lower_begin = lower_column_offsets[source] + 1;
            const auto lower_end = lower_column_offsets[source + 1];
            const auto target_upper_begin = upper_column_offsets[target];
            const auto target_upper_end = upper_column_offsets[target + 1];
            const auto target_upper_nonzeros =
                target_upper_end - target_upper_begin;
            const auto target_lower_begin = lower_column_offsets[target];

            if (compact_indices != 0) {
                const auto encoded_offset = task_index_offsets[task];
                for (auto lower_position = lower_begin + lane;
                     lower_position < lower_end;
                     lower_position += static_cast<std::int32_t>(blockDim.x)) {
                    const auto update =
                        -lower_values[lower_position] * multiplier;
                    const auto local_position = lower_position - lower_begin;
                    const auto relative_position = encoded_offset >= 0
                        ? static_cast<std::int32_t>(indices_u16[
                              encoded_offset + local_position])
                        : static_cast<std::int32_t>(indices_u32[
                              -encoded_offset - 1 + local_position]);
                    if (relative_position < target_upper_nonzeros) {
                        atomicAdd(upper_values + target_upper_begin +
                                      relative_position,
                                  update);
                    } else {
                        atomicAdd(lower_values + target_lower_begin + 1 +
                                      relative_position -
                                      target_upper_nonzeros,
                                  update);
                    }
                }
                continue;
            }

            for (auto lower_position = lower_begin + lane;
                 lower_position < lower_end;
                 lower_position += static_cast<std::int32_t>(blockDim.x)) {
                const auto update =
                    -lower_values[lower_position] * multiplier;
                const auto row = lower_row_indices[lower_position];
                if (row <= target) {
                    const auto destination = find_sorted_row(
                        upper_row_indices, target_upper_begin,
                        target_upper_end, row);
                    if (destination < target_upper_end &&
                        upper_row_indices[destination] == row) {
                        atomicAdd(upper_values + destination, update);
                    } else {
                        atomicCAS(singular_column, -1, target);
                    }
                } else {
                    const auto end = lower_column_offsets[target + 1];
                    const auto destination = find_sorted_row(
                        lower_row_indices, target_lower_begin, end, row);
                    if (destination < end &&
                        lower_row_indices[destination] == row) {
                        atomicAdd(lower_values + destination, update);
                    } else {
                        atomicCAS(singular_column, -1, target);
                    }
                }
            }
        }
        grid.sync();
    }
}

// --------------------------------------------------------------------------
// Triangular-solve layer
// --------------------------------------------------------------------------

// Single-block correctness fallback used when cooperative launch is not
// available. L has an implicit unit diagonal; U stores its diagonal last.
__global__ void solve_fixed_pattern_kernel(
    std::int32_t dimension,
    const std::int32_t* lower_column_offsets,
    const std::int32_t* lower_row_indices,
    const double* lower_values,
    const std::int32_t* upper_column_offsets,
    const std::int32_t* upper_row_indices,
    const double* upper_values,
    const std::int32_t* row_permutation,
    const std::int32_t* column_permutation,
    const double* right_hand_side,
    double* workspace,
    double* solution) {
    __shared__ double column_value;
    const auto lane = static_cast<std::int32_t>(threadIdx.x);

    for (auto index = lane; index < dimension;
         index += static_cast<std::int32_t>(blockDim.x)) {
        workspace[index] = right_hand_side[row_permutation[index]];
    }
    __syncthreads();

    for (std::int32_t column = 0; column < dimension; ++column) {
        if (lane == 0) column_value = workspace[column];
        __syncthreads();
        const auto begin = lower_column_offsets[column];
        const auto end = lower_column_offsets[column + 1];
        for (auto position = begin + 1 + lane; position < end;
             position += static_cast<std::int32_t>(blockDim.x)) {
            workspace[lower_row_indices[position]] -=
                lower_values[position] * column_value;
        }
        __syncthreads();
    }

    for (std::int32_t column = dimension - 1; column >= 0; --column) {
        const auto begin = upper_column_offsets[column];
        const auto end = upper_column_offsets[column + 1];
        if (lane == 0) {
            workspace[column] /= upper_values[end - 1];
            column_value = workspace[column];
        }
        __syncthreads();
        for (auto position = begin + lane; position + 1 < end;
             position += static_cast<std::int32_t>(blockDim.x)) {
            workspace[upper_row_indices[position]] -=
                upper_values[position] * column_value;
        }
        __syncthreads();
    }

    for (auto index = lane; index < dimension;
         index += static_cast<std::int32_t>(blockDim.x)) {
        solution[column_permutation[index]] = workspace[index];
    }
}

// One warp owns one ready triangular column. Consecutive narrow levels are
// fused behind block barriers; wide levels use the full cooperative grid.
__global__ void solve_persistent_levels_kernel(
    std::int32_t dimension,
    const std::int32_t* lower_column_offsets,
    const std::int32_t* lower_row_indices,
    const double* lower_values,
    const std::int32_t* upper_column_offsets,
    const std::int32_t* upper_row_indices,
    const double* upper_values,
    const std::int32_t* row_permutation,
    const std::int32_t* column_permutation,
    const double* right_hand_side,
    const std::int32_t* forward_level_offsets,
    const std::int32_t* forward_columns,
    const std::int32_t* forward_group_offsets,
    std::int32_t forward_group_count,
    const std::int32_t* backward_level_offsets,
    const std::int32_t* backward_columns,
    const std::int32_t* backward_group_offsets,
    std::int32_t backward_group_count,
    double* workspace,
    double* solution) {
    const auto grid = cg::this_grid();
    const auto global_thread = static_cast<std::int32_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const auto grid_threads =
        static_cast<std::int32_t>(gridDim.x * blockDim.x);
    const auto lane = static_cast<std::int32_t>(threadIdx.x & 31);
    const auto block_warp = static_cast<std::int32_t>(threadIdx.x >> 5);
    const auto block_warps = static_cast<std::int32_t>(blockDim.x >> 5);
    const auto global_warp = global_thread >> 5;
    const auto grid_warps = grid_threads >> 5;

    for (auto index = global_thread; index < dimension;
         index += grid_threads) {
        workspace[index] = right_hand_side[row_permutation[index]];
    }
    grid.sync();

    // Unit-lower forward substitution.
    for (std::int32_t group = 0; group < forward_group_count; ++group) {
        const auto level_begin = forward_group_offsets[group];
        const auto level_end = forward_group_offsets[group + 1];
        const auto first_width = forward_level_offsets[level_begin + 1] -
                                 forward_level_offsets[level_begin];
        if (first_width <= block_warps) {
            if (blockIdx.x == 0) {
                for (auto level = level_begin; level < level_end; ++level) {
                    const auto begin = forward_level_offsets[level];
                    const auto end = forward_level_offsets[level + 1];
                    for (auto task = begin + block_warp; task < end;
                         task += block_warps) {
                        const auto column = forward_columns[task];
                        const auto column_value = workspace[column];
                        const auto lower_begin =
                            lower_column_offsets[column] + 1;
                        const auto lower_end =
                            lower_column_offsets[column + 1];
                        for (auto position = lower_begin + lane;
                             position < lower_end; position += 32) {
                            atomicAdd(workspace + lower_row_indices[position],
                                      -lower_values[position] * column_value);
                        }
                    }
                    __syncthreads();
                }
            }
        } else {
            const auto begin = forward_level_offsets[level_begin];
            const auto end = forward_level_offsets[level_begin + 1];
            for (auto task = begin + global_warp; task < end;
                 task += grid_warps) {
                const auto column = forward_columns[task];
                const auto column_value = workspace[column];
                const auto lower_begin = lower_column_offsets[column] + 1;
                const auto lower_end = lower_column_offsets[column + 1];
                for (auto position = lower_begin + lane;
                     position < lower_end; position += 32) {
                    atomicAdd(workspace + lower_row_indices[position],
                              -lower_values[position] * column_value);
                }
            }
        }
        grid.sync();
    }

    // Upper backward substitution.
    for (std::int32_t group = 0; group < backward_group_count; ++group) {
        const auto level_begin = backward_group_offsets[group];
        const auto level_end = backward_group_offsets[group + 1];
        const auto first_width = backward_level_offsets[level_begin + 1] -
                                 backward_level_offsets[level_begin];
        if (first_width <= block_warps) {
            if (blockIdx.x == 0) {
                for (auto level = level_begin; level < level_end; ++level) {
                    const auto begin = backward_level_offsets[level];
                    const auto end = backward_level_offsets[level + 1];
                    for (auto task = begin + block_warp; task < end;
                         task += block_warps) {
                        const auto column = backward_columns[task];
                        const auto upper_begin =
                            upper_column_offsets[column];
                        const auto upper_end =
                            upper_column_offsets[column + 1];
                        double column_value{};
                        if (lane == 0) {
                            column_value = workspace[column] /
                                upper_values[upper_end - 1];
                            workspace[column] = column_value;
                        }
                        column_value =
                            __shfl_sync(0xffffffffU, column_value, 0);
                        for (auto position = upper_begin + lane;
                             position + 1 < upper_end; position += 32) {
                            atomicAdd(workspace + upper_row_indices[position],
                                      -upper_values[position] * column_value);
                        }
                    }
                    __syncthreads();
                }
            }
        } else {
            const auto begin = backward_level_offsets[level_begin];
            const auto end = backward_level_offsets[level_begin + 1];
            for (auto task = begin + global_warp; task < end;
                 task += grid_warps) {
                const auto column = backward_columns[task];
                const auto upper_begin = upper_column_offsets[column];
                const auto upper_end = upper_column_offsets[column + 1];
                double column_value{};
                if (lane == 0) {
                    column_value = workspace[column] /
                        upper_values[upper_end - 1];
                    workspace[column] = column_value;
                }
                column_value =
                    __shfl_sync(0xffffffffU, column_value, 0);
                for (auto position = upper_begin + lane;
                     position + 1 < upper_end; position += 32) {
                    atomicAdd(workspace + upper_row_indices[position],
                              -upper_values[position] * column_value);
                }
            }
        }
        grid.sync();
    }

    for (auto index = global_thread; index < dimension;
         index += grid_threads) {
        solution[column_permutation[index]] = workspace[index];
    }
}

// --------------------------------------------------------------------------
// Host launch layer
// --------------------------------------------------------------------------

// These wrappers are the only place that knows raw kernel argument order.
// The operator lifecycle can therefore select an algorithm without duplicating
// CUDA launch plumbing or coupling itself to every kernel signature.
inline void launch_right_looking_index_build(DeviceState& state) {
    if (!state.right_looking_compact_indices ||
        state.right_looking_task_count <= 0) {
        return;
    }
    check_cuda(cudaMemsetAsync(state.singular_column, 0xff,
                               sizeof(std::int32_t), state.stream),
               "cudaMemsetAsync(custom index build status)");
    const auto blocks = std::min(kIndexBuildBlocks,
                                 state.right_looking_task_count);
    build_right_looking_indices_kernel
        <<<blocks, kIndexBuildThreads, 0, state.stream>>>(
            state.right_looking_task_count,
            state.right_looking_task_sources,
            state.right_looking_task_targets,
            state.right_looking_task_index_offsets,
            state.lower_column_offsets,
            state.lower_row_indices,
            state.upper_column_offsets,
            state.upper_row_indices,
            state.right_looking_indices_u16,
            state.right_looking_indices_u32,
            state.singular_column);
    check_cuda(cudaGetLastError(),
               "build_right_looking_indices_kernel launch");
}

inline void launch_numerical_refactor(DeviceState& state) {
    if (state.right_looking_refactor) {
        std::int32_t compact_indices =
            state.right_looking_compact_indices ? 1 : 0;
        void* arguments[] = {
            &state.dimension,
            &state.input_nonzeros,
            &state.input_values,
            &state.input_factor_destinations,
            &state.lower_nonzeros,
            &state.lower_column_offsets,
            &state.lower_row_indices,
            &state.lower_values,
            &state.upper_nonzeros,
            &state.upper_column_offsets,
            &state.upper_row_indices,
            &state.upper_values,
            &state.right_looking_level_offsets,
            &state.right_looking_level_columns,
            &state.right_looking_level_count,
            &state.right_looking_task_offsets,
            &state.right_looking_task_sources,
            &state.right_looking_task_targets,
            &state.right_looking_task_upper_positions,
            &compact_indices,
            &state.right_looking_task_index_offsets,
            &state.right_looking_indices_u16,
            &state.right_looking_indices_u32,
            &state.singular_column,
        };
        check_cuda(cudaLaunchCooperativeKernel(
                       reinterpret_cast<const void*>(
                           refactor_right_looking_levels_kernel),
                       dim3(state.cooperative_refactor_grid_blocks),
                       dim3(kRightLookingThreads), arguments, 0, state.stream),
                   "cudaLaunchCooperativeKernel(custom right-looking refactor)");
        return;
    }

    if (state.cooperative_refactor_grid_blocks > 0) {
        void* arguments[] = {
            &state.dimension,
            &state.input_column_offsets,
            &state.input_row_indices,
            &state.input_values,
            &state.lower_column_offsets,
            &state.lower_row_indices,
            &state.lower_values,
            &state.upper_column_offsets,
            &state.upper_row_indices,
            &state.upper_values,
            &state.inverse_row_permutation,
            &state.column_permutation,
            &state.refactor_level_offsets,
            &state.refactor_columns,
            &state.refactor_group_offsets,
            &state.refactor_group_count,
            &state.refactor_workspaces,
            &state.singular_column,
        };
        check_cuda(cudaLaunchCooperativeKernel(
                       reinterpret_cast<const void*>(
                           refactor_persistent_dense_kernel),
                       dim3(state.cooperative_refactor_grid_blocks),
                       dim3(kRefactorThreads), arguments, 0, state.stream),
                   "cudaLaunchCooperativeKernel(custom dense refactor)");
        return;
    }

    refactor_fixed_pattern_kernel<<<1, kRefactorThreads, 0, state.stream>>>(
        state.dimension, state.input_column_offsets, state.input_row_indices,
        state.input_values, state.lower_column_offsets,
        state.lower_row_indices, state.lower_values,
        state.upper_column_offsets, state.upper_row_indices,
        state.upper_values, state.inverse_row_permutation,
        state.column_permutation, state.workspace, state.singular_column);
    check_cuda(cudaGetLastError(), "refactor_fixed_pattern_kernel launch");
}

inline void launch_triangular_solve(DeviceState& state) {
    if (state.cooperative_grid_blocks > 0) {
        void* arguments[] = {
            &state.dimension,
            &state.lower_column_offsets,
            &state.lower_row_indices,
            &state.lower_values,
            &state.upper_column_offsets,
            &state.upper_row_indices,
            &state.upper_values,
            &state.row_permutation,
            &state.column_permutation,
            &state.right_hand_side,
            &state.forward_level_offsets,
            &state.forward_columns,
            &state.forward_group_offsets,
            &state.forward_group_count,
            &state.backward_level_offsets,
            &state.backward_columns,
            &state.backward_group_offsets,
            &state.backward_group_count,
            &state.workspace,
            &state.solution,
        };
        check_cuda(cudaLaunchCooperativeKernel(
                       reinterpret_cast<const void*>(
                           solve_persistent_levels_kernel),
                       dim3(state.cooperative_grid_blocks),
                       dim3(kSolveThreads), arguments, 0, state.stream),
                   "cudaLaunchCooperativeKernel(custom level solve)");
        return;
    }

    solve_fixed_pattern_kernel<<<1, kSolveThreads, 0, state.stream>>>(
        state.dimension, state.lower_column_offsets,
        state.lower_row_indices, state.lower_values,
        state.upper_column_offsets, state.upper_row_indices,
        state.upper_values, state.row_permutation, state.column_permutation,
        state.right_hand_side, state.workspace, state.solution);
    check_cuda(cudaGetLastError(), "solve_fixed_pattern_kernel launch");
}

}  // namespace eda_gpu::operators::cuda::detail
