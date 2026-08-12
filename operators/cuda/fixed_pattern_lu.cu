#include "fixed_pattern_lu.hpp"
#include "fixed_pattern_lu_kernels.cuh"
#include "fixed_pattern_lu_runtime.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace eda_gpu::operators::cuda {
namespace {

constexpr std::int32_t kParallelRefactorMinimumDimension = 16384;

// Import only the internal runtime and kernel symbols used by the host
// orchestration below. CUDA implementation details remain in detail::*.
using detail::DeviceState;
using detail::byte_count;
using detail::check_cuda;
using detail::kPersistentBlocksPerMultiprocessor;
using detail::kRefactorFusedLevelWidth;
using detail::kRefactorThreads;
using detail::kRightLookingBlocksPerMultiprocessor;
using detail::kRightLookingThreads;
using detail::kSolveThreads;
using detail::refactor_persistent_dense_kernel;
using detail::refactor_right_looking_levels_kernel;
using detail::solve_persistent_levels_kernel;

enum class RequestedRefactorMode {
    left_looking,
    right_looking_adaptive,
    right_looking_binary,
};

[[nodiscard]] RequestedRefactorMode requested_refactor_mode() {
    const char* value = std::getenv("EDA_GPU_REFACTOR_MODE");
    if (value == nullptr || std::string_view(value) == "right-looking") {
        return RequestedRefactorMode::right_looking_adaptive;
    }
    if (std::string_view(value) == "left-looking") {
        return RequestedRefactorMode::left_looking;
    }
    if (std::string_view(value) == "right-looking-binary") {
        return RequestedRefactorMode::right_looking_binary;
    }
    throw std::runtime_error(
        "EDA_GPU_REFACTOR_MODE must be left-looking, right-looking or "
        "right-looking-binary");
}

[[nodiscard]] bool query_device(std::string& detail) {
    int count{};
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        detail = std::string("CUDA runtime unavailable: ") + cudaGetErrorString(status);
        cudaGetLastError();
        return false;
    }
    if (count <= 0) {
        detail = "CUDA runtime is present but no device is visible";
        return false;
    }

    cudaDeviceProp properties{};
    const auto properties_status = cudaGetDeviceProperties(&properties, 0);
    if (properties_status != cudaSuccess) {
        detail = std::string("cannot query CUDA device 0: ") +
                 cudaGetErrorString(properties_status);
        cudaGetLastError();
        return false;
    }
    std::ostringstream description;
    description << properties.name << " (sm_" << properties.major << properties.minor
                << ", " << (properties.totalGlobalMem / (1024U * 1024U)) << " MiB)";
    detail = description.str();
    return true;
}

// ==========================================================================
// Static preparation layer
//
// Everything below this marker runs once for a fixed sparsity pattern. It
// converts KLU's L/U structure into dependency levels, fused level groups and
// right-looking subcolumn tasks. No numerical matrix values are changed here.
// ==========================================================================

struct LevelSchedule {
    std::vector<std::int32_t> offsets;
    std::vector<std::int32_t> columns;
    std::int32_t widest_level{};
    std::int32_t narrow_levels{};
};

[[nodiscard]] std::vector<std::int32_t> build_level_groups(
    const LevelSchedule& schedule,
    std::int32_t narrow_width) {
    const auto level_count = static_cast<std::int32_t>(
        schedule.offsets.size() - 1U);
    std::vector<std::int32_t> groups;
    groups.reserve(schedule.offsets.size());
    groups.push_back(0);
    std::int32_t level{};
    while (level < level_count) {
        const auto width = schedule.offsets[static_cast<std::size_t>(level) + 1U] -
                           schedule.offsets[static_cast<std::size_t>(level)];
        if (width <= narrow_width) {
            do {
                ++level;
                if (level >= level_count) break;
            } while (schedule.offsets[static_cast<std::size_t>(level) + 1U] -
                         schedule.offsets[static_cast<std::size_t>(level)] <=
                     narrow_width);
        } else {
            ++level;
        }
        groups.push_back(level);
    }
    return groups;
}

[[nodiscard]] LevelSchedule bucket_levels(
    const std::vector<std::int32_t>& column_levels) {
    const auto maximum = *std::max_element(column_levels.begin(), column_levels.end());
    LevelSchedule schedule;
    schedule.offsets.assign(static_cast<std::size_t>(maximum) + 2U, 0);
    for (const auto level : column_levels) {
        ++schedule.offsets[static_cast<std::size_t>(level) + 1U];
    }
    for (std::size_t level = 0; level + 1U < schedule.offsets.size(); ++level) {
        schedule.offsets[level + 1U] += schedule.offsets[level];
    }
    schedule.columns.resize(column_levels.size());
    auto cursor = schedule.offsets;
    for (std::int32_t column = 0;
         column < static_cast<std::int32_t>(column_levels.size()); ++column) {
        const auto level = column_levels[static_cast<std::size_t>(column)];
        schedule.columns[static_cast<std::size_t>(
            cursor[static_cast<std::size_t>(level)]++)] = column;
    }
    for (std::size_t level = 0; level + 1U < schedule.offsets.size(); ++level) {
        const auto width = schedule.offsets[level + 1U] - schedule.offsets[level];
        schedule.widest_level = std::max(schedule.widest_level, width);
        if (width <= 8) ++schedule.narrow_levels;
    }
    return schedule;
}

[[nodiscard]] LevelSchedule build_forward_levels(
    const SparseCscData& lower,
    std::int32_t dimension) {
    std::vector<std::int32_t> levels(static_cast<std::size_t>(dimension), 0);
    for (std::int32_t column = 0; column < dimension; ++column) {
        const auto next_level = levels[static_cast<std::size_t>(column)] + 1;
        const auto begin = lower.column_offsets[static_cast<std::size_t>(column)] + 1;
        const auto end = lower.column_offsets[static_cast<std::size_t>(column) + 1U];
        for (auto position = begin; position < end; ++position) {
            const auto row = lower.row_indices[static_cast<std::size_t>(position)];
            levels[static_cast<std::size_t>(row)] =
                std::max(levels[static_cast<std::size_t>(row)], next_level);
        }
    }
    return bucket_levels(levels);
}

[[nodiscard]] LevelSchedule build_backward_levels(
    const SparseCscData& upper,
    std::int32_t dimension) {
    std::vector<std::int32_t> levels(static_cast<std::size_t>(dimension), 0);
    for (std::int32_t column = dimension - 1; column >= 0; --column) {
        const auto next_level = levels[static_cast<std::size_t>(column)] + 1;
        const auto begin = upper.column_offsets[static_cast<std::size_t>(column)];
        const auto end = upper.column_offsets[static_cast<std::size_t>(column) + 1U] - 1;
        for (auto position = begin; position < end; ++position) {
            const auto row = upper.row_indices[static_cast<std::size_t>(position)];
            levels[static_cast<std::size_t>(row)] =
                std::max(levels[static_cast<std::size_t>(row)], next_level);
        }
    }
    return bucket_levels(levels);
}

[[nodiscard]] LevelSchedule build_refactor_levels(
    const SparseCscData& upper,
    std::int32_t dimension) {
    std::vector<std::int32_t> levels(static_cast<std::size_t>(dimension), 0);
    for (std::int32_t column = 0; column < dimension; ++column) {
        const auto begin = upper.column_offsets[static_cast<std::size_t>(column)];
        const auto end = upper.column_offsets[static_cast<std::size_t>(column) + 1U];
        auto level = 0;
        for (auto position = begin; position + 1 < end; ++position) {
            const auto dependency =
                upper.row_indices[static_cast<std::size_t>(position)];
            level = std::max(level,
                levels[static_cast<std::size_t>(dependency)] + 1);
        }
        levels[static_cast<std::size_t>(column)] = level;
    }
    return bucket_levels(levels);
}

// GLU3's relaxed dependency analysis augments the ordinary U-column
// dependencies with every structural L(k,i) edge to cover the read/write
// hazards introduced by right-looking subcolumn updates.
[[nodiscard]] LevelSchedule build_glu3_relaxed_refactor_levels(
    const SparseCscData& lower,
    const SparseCscData& upper,
    std::int32_t dimension) {
    std::vector<std::int32_t> levels(static_cast<std::size_t>(dimension), 0);
    for (std::int32_t column = 0; column < dimension; ++column) {
        auto level = levels[static_cast<std::size_t>(column)];
        const auto upper_begin =
            upper.column_offsets[static_cast<std::size_t>(column)];
        const auto upper_end =
            upper.column_offsets[static_cast<std::size_t>(column) + 1U];
        for (auto position = upper_begin; position + 1 < upper_end; ++position) {
            const auto dependency =
                upper.row_indices[static_cast<std::size_t>(position)];
            const auto lower_begin =
                lower.column_offsets[static_cast<std::size_t>(dependency)];
            const auto lower_end =
                lower.column_offsets[static_cast<std::size_t>(dependency) + 1U];
            if (lower_end - lower_begin > 1) {
                level = std::max(
                    level, levels[static_cast<std::size_t>(dependency)] + 1);
            }
        }
        levels[static_cast<std::size_t>(column)] = level;

        const auto next_level = level + 1;
        const auto lower_begin =
            lower.column_offsets[static_cast<std::size_t>(column)] + 1;
        const auto lower_end =
            lower.column_offsets[static_cast<std::size_t>(column) + 1U];
        for (auto position = lower_begin; position < lower_end; ++position) {
            const auto dependent =
                lower.row_indices[static_cast<std::size_t>(position)];
            levels[static_cast<std::size_t>(dependent)] = std::max(
                levels[static_cast<std::size_t>(dependent)], next_level);
        }
    }
    return bucket_levels(levels);
}

[[nodiscard]] core::LevelWidthProfile level_width_profile(
    const LevelSchedule& schedule) {
    core::LevelWidthProfile result;
    result.levels = static_cast<std::int32_t>(schedule.offsets.size() - 1U);
    result.widest = schedule.widest_level;
    for (std::int32_t level = 0; level < result.levels; ++level) {
        const auto width = schedule.offsets[static_cast<std::size_t>(level) + 1U] -
                           schedule.offsets[static_cast<std::size_t>(level)];
        if (width == 1) {
            ++result.width_1;
        } else if (width == 2) {
            ++result.width_2;
        } else if (width <= 8) {
            ++result.width_3_to_8;
        } else if (width <= 32) {
            ++result.width_9_to_32;
        } else {
            ++result.width_33_plus;
        }
    }
    return result;
}

[[nodiscard]] std::int32_t percentile(
    const std::vector<std::int32_t>& sorted_values,
    std::int32_t percentage) {
    if (sorted_values.empty()) return 0;
    const auto numerator = static_cast<std::uint64_t>(percentage) *
                           static_cast<std::uint64_t>(sorted_values.size());
    const auto rank = std::max<std::uint64_t>(1U, (numerator + 99U) / 100U);
    return sorted_values[static_cast<std::size_t>(rank - 1U)];
}

[[nodiscard]] core::NumericFactorDagProfile build_refactor_dag_profile(
    const SparseCscData& lower,
    const SparseCscData& upper,
    const LevelSchedule& u_schedule,
    const LevelSchedule& glu3_schedule,
    std::int32_t dimension) {
    core::NumericFactorDagProfile result;
    result.available = true;
    result.u_dependency_levels = level_width_profile(u_schedule);
    result.glu3_relaxed_levels = level_width_profile(glu3_schedule);
    result.dense_workspace_bytes_per_block =
        static_cast<std::uint64_t>(dimension) * sizeof(double);

    std::vector<std::int32_t> outgoing_offsets(
        static_cast<std::size_t>(dimension) + 1U, 0);
    for (std::int32_t target = 0; target < dimension; ++target) {
        const auto begin = upper.column_offsets[static_cast<std::size_t>(target)];
        const auto end = upper.column_offsets[static_cast<std::size_t>(target) + 1U];
        for (auto position = begin; position + 1 < end; ++position) {
            const auto source = upper.row_indices[static_cast<std::size_t>(position)];
            ++outgoing_offsets[static_cast<std::size_t>(source) + 1U];
        }
    }
    for (std::size_t source = 0; source + 1U < outgoing_offsets.size(); ++source) {
        outgoing_offsets[source + 1U] += outgoing_offsets[source];
    }
    std::vector<std::int32_t> outgoing_targets(
        static_cast<std::size_t>(outgoing_offsets.back()));
    auto cursor = outgoing_offsets;
    for (std::int32_t target = 0; target < dimension; ++target) {
        const auto begin = upper.column_offsets[static_cast<std::size_t>(target)];
        const auto end = upper.column_offsets[static_cast<std::size_t>(target) + 1U];
        for (auto position = begin; position + 1 < end; ++position) {
            const auto source = upper.row_indices[static_cast<std::size_t>(position)];
            outgoing_targets[static_cast<std::size_t>(
                cursor[static_cast<std::size_t>(source)]++)] = target;
        }
    }

    std::vector<bool> is_single_level_column(
        static_cast<std::size_t>(dimension), false);
    const auto glu3_level_count = static_cast<std::int32_t>(
        glu3_schedule.offsets.size() - 1U);
    for (std::int32_t level = 0; level < glu3_level_count; ++level) {
        const auto begin = glu3_schedule.offsets[static_cast<std::size_t>(level)];
        const auto end = glu3_schedule.offsets[static_cast<std::size_t>(level) + 1U];
        if (end - begin == 1) {
            is_single_level_column[static_cast<std::size_t>(
                glu3_schedule.columns[static_cast<std::size_t>(begin)])] = true;
        }
    }

    std::vector<std::int32_t> single_level_fanouts;
    single_level_fanouts.reserve(
        static_cast<std::size_t>(result.glu3_relaxed_levels.width_1));
    constexpr std::uint64_t kUint16Offsets =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint16_t>::max()) + 1U;
    for (std::int32_t source = 0; source < dimension; ++source) {
        const auto outgoing_begin =
            outgoing_offsets[static_cast<std::size_t>(source)];
        const auto outgoing_end =
            outgoing_offsets[static_cast<std::size_t>(source) + 1U];
        const auto fanout = outgoing_end - outgoing_begin;
        const auto lower_updates =
            lower.column_offsets[static_cast<std::size_t>(source) + 1U] -
            lower.column_offsets[static_cast<std::size_t>(source)] - 1;
        const auto source_updates = static_cast<std::uint64_t>(fanout) *
                                    static_cast<std::uint64_t>(lower_updates);
        result.subcolumn_tasks += static_cast<std::uint64_t>(fanout);
        result.scalar_updates += source_updates;
        result.full_index_bytes_u32 += source_updates * sizeof(std::uint32_t);

        const bool single =
            is_single_level_column[static_cast<std::size_t>(source)];
        if (single) {
            single_level_fanouts.push_back(fanout);
            result.single_level_subcolumn_tasks +=
                static_cast<std::uint64_t>(fanout);
            result.single_level_scalar_updates += source_updates;
        }

        for (auto task = outgoing_begin; task < outgoing_end; ++task) {
            const auto target =
                outgoing_targets[static_cast<std::size_t>(task)];
            const auto factor_column_nonzeros =
                lower.column_offsets[static_cast<std::size_t>(target) + 1U] -
                lower.column_offsets[static_cast<std::size_t>(target)] +
                upper.column_offsets[static_cast<std::size_t>(target) + 1U] -
                upper.column_offsets[static_cast<std::size_t>(target)] - 1;
            const auto bytes_per_index =
                static_cast<std::uint64_t>(factor_column_nonzeros) <= kUint16Offsets
                    ? sizeof(std::uint16_t)
                    : sizeof(std::uint32_t);
            const auto index_bytes =
                static_cast<std::uint64_t>(lower_updates) * bytes_per_index;
            result.full_index_bytes_mixed_u16_u32 += index_bytes;
            if (single) {
                result.single_level_index_bytes_mixed_u16_u32 += index_bytes;
            }
        }
    }
    if (result.scalar_updates != 0U) {
        result.single_level_scalar_update_fraction =
            static_cast<double>(result.single_level_scalar_updates) /
            static_cast<double>(result.scalar_updates);
    }
    std::sort(single_level_fanouts.begin(), single_level_fanouts.end());
    result.single_level_fanout_p50 = percentile(single_level_fanouts, 50);
    result.single_level_fanout_p90 = percentile(single_level_fanouts, 90);
    result.single_level_fanout_p99 = percentile(single_level_fanouts, 99);
    if (!single_level_fanouts.empty()) {
        result.single_level_fanout_max = single_level_fanouts.back();
    }

    // Count write conflicts for GLU3 columns that are released in the same
    // level. This is a lower bound for an asynchronous scheduler, but it is
    // enough to quantify the atomic-update pressure of a level-based prototype.
    std::vector<std::int32_t> target_level(
        static_cast<std::size_t>(dimension), -1);
    std::vector<std::int32_t> target_writers(
        static_cast<std::size_t>(dimension), 0);
    std::vector<std::int32_t> touched_targets;
    for (std::int32_t level = 0; level < glu3_level_count; ++level) {
        touched_targets.clear();
        const auto begin = glu3_schedule.offsets[static_cast<std::size_t>(level)];
        const auto end = glu3_schedule.offsets[static_cast<std::size_t>(level) + 1U];
        for (auto position = begin; position < end; ++position) {
            const auto source =
                glu3_schedule.columns[static_cast<std::size_t>(position)];
            const auto outgoing_begin =
                outgoing_offsets[static_cast<std::size_t>(source)];
            const auto outgoing_end =
                outgoing_offsets[static_cast<std::size_t>(source) + 1U];
            for (auto task = outgoing_begin; task < outgoing_end; ++task) {
                const auto target =
                    outgoing_targets[static_cast<std::size_t>(task)];
                if (target_level[static_cast<std::size_t>(target)] != level) {
                    target_level[static_cast<std::size_t>(target)] = level;
                    target_writers[static_cast<std::size_t>(target)] = 1;
                    touched_targets.push_back(target);
                } else {
                    ++target_writers[static_cast<std::size_t>(target)];
                }
            }
        }
        for (const auto target : touched_targets) {
            const auto writers =
                target_writers[static_cast<std::size_t>(target)];
            result.same_level_max_writers =
                std::max(result.same_level_max_writers, writers);
            if (writers > 1) {
                ++result.same_level_collision_groups;
                result.same_level_conflicting_tasks +=
                    static_cast<std::uint64_t>(writers);
                result.same_level_extra_writers +=
                    static_cast<std::uint64_t>(writers - 1);
            }
        }
    }
    return result;
}

struct RightLookingSchedule {
    std::vector<std::int32_t> level_offsets;
    std::vector<std::int32_t> level_columns;
    std::vector<std::int32_t> task_offsets;
    std::vector<std::int32_t> task_sources;
    std::vector<std::int32_t> task_targets;
    std::vector<std::int32_t> task_upper_positions;
};

struct CompactIndexLayout {
    std::vector<std::int32_t> task_offsets;
    std::int32_t u16_count{};
    std::int32_t u32_count{};

    [[nodiscard]] std::uint64_t bytes() const {
        return byte_count(task_offsets) +
               static_cast<std::uint64_t>(u16_count) * sizeof(std::uint16_t) +
               static_cast<std::uint64_t>(u32_count) * sizeof(std::uint32_t);
    }
};

[[nodiscard]] RightLookingSchedule build_right_looking_schedule(
    const SparseCscData& upper,
    const LevelSchedule& levels,
    std::int32_t dimension) {
    std::vector<std::int32_t> outgoing_offsets(
        static_cast<std::size_t>(dimension) + 1U, 0);
    for (std::int32_t target = 0; target < dimension; ++target) {
        const auto begin = upper.column_offsets[static_cast<std::size_t>(target)];
        const auto end = upper.column_offsets[static_cast<std::size_t>(target) + 1U];
        for (auto position = begin; position + 1 < end; ++position) {
            const auto source = upper.row_indices[static_cast<std::size_t>(position)];
            ++outgoing_offsets[static_cast<std::size_t>(source) + 1U];
        }
    }
    for (std::size_t source = 0; source + 1U < outgoing_offsets.size(); ++source) {
        outgoing_offsets[source + 1U] += outgoing_offsets[source];
    }
    std::vector<std::int32_t> outgoing_targets(
        static_cast<std::size_t>(outgoing_offsets.back()));
    std::vector<std::int32_t> outgoing_upper_positions(
        outgoing_targets.size());
    auto cursor = outgoing_offsets;
    for (std::int32_t target = 0; target < dimension; ++target) {
        const auto begin = upper.column_offsets[static_cast<std::size_t>(target)];
        const auto end = upper.column_offsets[static_cast<std::size_t>(target) + 1U];
        for (auto position = begin; position + 1 < end; ++position) {
            const auto source = upper.row_indices[static_cast<std::size_t>(position)];
            const auto destination =
                cursor[static_cast<std::size_t>(source)]++;
            outgoing_targets[static_cast<std::size_t>(destination)] = target;
            outgoing_upper_positions[static_cast<std::size_t>(destination)] = position;
        }
    }

    RightLookingSchedule result;
    result.level_offsets = levels.offsets;
    result.level_columns = levels.columns;
    const auto level_count = static_cast<std::int32_t>(levels.offsets.size() - 1U);
    result.task_offsets.reserve(static_cast<std::size_t>(level_count) + 1U);
    result.task_sources.reserve(outgoing_targets.size());
    result.task_targets.reserve(outgoing_targets.size());
    result.task_upper_positions.reserve(outgoing_targets.size());
    result.task_offsets.push_back(0);
    for (std::int32_t level = 0; level < level_count; ++level) {
        const auto begin = levels.offsets[static_cast<std::size_t>(level)];
        const auto end = levels.offsets[static_cast<std::size_t>(level) + 1U];
        for (auto position = begin; position < end; ++position) {
            const auto source =
                levels.columns[static_cast<std::size_t>(position)];
            const auto outgoing_begin =
                outgoing_offsets[static_cast<std::size_t>(source)];
            const auto outgoing_end =
                outgoing_offsets[static_cast<std::size_t>(source) + 1U];
            for (auto task = outgoing_begin; task < outgoing_end; ++task) {
                result.task_sources.push_back(source);
                result.task_targets.push_back(
                    outgoing_targets[static_cast<std::size_t>(task)]);
                result.task_upper_positions.push_back(
                    outgoing_upper_positions[static_cast<std::size_t>(task)]);
            }
        }
        result.task_offsets.push_back(
            static_cast<std::int32_t>(result.task_sources.size()));
    }
    return result;
}

[[nodiscard]] CompactIndexLayout build_compact_index_layout(
    const SparseCscData& lower,
    const SparseCscData& upper,
    const RightLookingSchedule& schedule) {
    CompactIndexLayout result;
    result.task_offsets.resize(schedule.task_sources.size());
    std::uint64_t u16_count{};
    std::uint64_t u32_count{};
    constexpr std::uint64_t kUint16Offsets =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint16_t>::max()) + 1U;
    constexpr std::uint64_t kMaximumEncodedOffset =
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) - 1U;
    for (std::size_t task = 0; task < schedule.task_sources.size(); ++task) {
        const auto source = schedule.task_sources[task];
        const auto target = schedule.task_targets[task];
        const auto updates =
            lower.column_offsets[static_cast<std::size_t>(source) + 1U] -
            lower.column_offsets[static_cast<std::size_t>(source)] - 1;
        const auto target_nonzeros =
            lower.column_offsets[static_cast<std::size_t>(target) + 1U] -
            lower.column_offsets[static_cast<std::size_t>(target)] +
            upper.column_offsets[static_cast<std::size_t>(target) + 1U] -
            upper.column_offsets[static_cast<std::size_t>(target)] - 1;
        if (static_cast<std::uint64_t>(target_nonzeros) <= kUint16Offsets) {
            if (u16_count + static_cast<std::uint64_t>(updates) >
                kMaximumEncodedOffset) {
                return {};
            }
            result.task_offsets[task] = static_cast<std::int32_t>(u16_count);
            u16_count += static_cast<std::uint64_t>(updates);
        } else {
            if (u32_count + static_cast<std::uint64_t>(updates) >
                kMaximumEncodedOffset) {
                return {};
            }
            result.task_offsets[task] =
                -static_cast<std::int32_t>(u32_count) - 1;
            u32_count += static_cast<std::uint64_t>(updates);
        }
    }
    result.u16_count = static_cast<std::int32_t>(u16_count);
    result.u32_count = static_cast<std::int32_t>(u32_count);
    return result;
}

[[nodiscard]] bool compact_index_layout_fits_device(
    const CompactIndexLayout& layout) {
    if (layout.task_offsets.empty()) return false;
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo(custom compact right-looking indices)");
    const auto reserve = std::max<std::uint64_t>(
        static_cast<std::uint64_t>(total_bytes) / 4U,
        512ULL * 1024ULL * 1024ULL);
    const auto required = layout.bytes();
    return static_cast<std::uint64_t>(free_bytes) > reserve &&
           required <= static_cast<std::uint64_t>(free_bytes) - reserve;
}

// Maps each input A(i,j) value directly into its fixed permuted L/U slot.
// This avoids repeating CPU-side sparse searches on every refactor call.
[[nodiscard]] std::vector<std::int32_t> build_input_factor_destinations(
    const core::CscMatrix& matrix,
    const InitialFactorization& factors,
    const std::vector<std::int32_t>& inverse_row_permutation) {
    std::vector<std::int32_t> destinations(matrix.values.size());
    for (std::int32_t factor_column = 0; factor_column < factors.dimension;
         ++factor_column) {
        const auto input_column = factors.column_permutation[
            static_cast<std::size_t>(factor_column)];
        const auto input_begin =
            matrix.column_offsets[static_cast<std::size_t>(input_column)];
        const auto input_end =
            matrix.column_offsets[static_cast<std::size_t>(input_column) + 1U];
        for (auto input_position = input_begin; input_position < input_end;
             ++input_position) {
            const auto input_row =
                matrix.row_indices[static_cast<std::size_t>(input_position)];
            const auto factor_row = inverse_row_permutation[
                static_cast<std::size_t>(input_row)];
            const auto& factor = factor_row <= factor_column
                                     ? factors.upper
                                     : factors.lower;
            const auto begin = factor.column_offsets[
                static_cast<std::size_t>(factor_column)];
            const auto end = factor.column_offsets[
                static_cast<std::size_t>(factor_column) + 1U];
            const auto first = factor.row_indices.begin() + begin;
            const auto last = factor.row_indices.begin() + end;
            const auto found = std::lower_bound(first, last, factor_row);
            if (found == last || *found != factor_row) {
                throw std::runtime_error(
                    "input nonzero is missing from the fixed L/U pattern");
            }
            const auto factor_position = static_cast<std::int32_t>(
                std::distance(factor.row_indices.begin(), found));
            destinations[static_cast<std::size_t>(input_position)] =
                factor_row <= factor_column ? factor_position
                                            : -factor_position - 1;
        }
    }
    return destinations;
}

[[nodiscard]] std::int32_t cooperative_right_refactor_grid_blocks() {
    int device{};
    check_cuda(cudaGetDevice(&device),
               "cudaGetDevice(custom right-looking schedule)");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(custom right-looking schedule)");
    if (properties.cooperativeLaunch == 0) return 0;

    int blocks_per_multiprocessor{};
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &blocks_per_multiprocessor,
                   refactor_right_looking_levels_kernel,
                   kRightLookingThreads, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(custom right-looking)");
    blocks_per_multiprocessor = std::min(
        blocks_per_multiprocessor, kRightLookingBlocksPerMultiprocessor);
    return static_cast<std::int32_t>(
        blocks_per_multiprocessor * properties.multiProcessorCount);
}

[[nodiscard]] std::int32_t cooperative_solve_grid_blocks() {
    int device{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(custom solve schedule)");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(custom solve schedule)");
    if (properties.cooperativeLaunch == 0) return 0;

    int blocks_per_multiprocessor{};
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &blocks_per_multiprocessor, solve_persistent_levels_kernel,
                   kSolveThreads, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(custom solve)");
    blocks_per_multiprocessor = std::min(
        blocks_per_multiprocessor, kPersistentBlocksPerMultiprocessor);
    return static_cast<std::int32_t>(blocks_per_multiprocessor *
                                     properties.multiProcessorCount);
}

[[nodiscard]] std::int32_t cooperative_refactor_grid_blocks(
    std::int32_t dimension) {
    int device{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice(custom refactor schedule)");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties(custom refactor schedule)");
    if (properties.cooperativeLaunch == 0) return 0;

    int blocks_per_multiprocessor{};
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &blocks_per_multiprocessor, refactor_persistent_dense_kernel,
                   kRefactorThreads, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor(custom refactor)");
    blocks_per_multiprocessor = std::min(
        blocks_per_multiprocessor, kPersistentBlocksPerMultiprocessor);
    auto blocks = static_cast<std::int32_t>(
        blocks_per_multiprocessor * properties.multiProcessorCount);
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo(custom refactor workspace)");
    const auto bytes_per_block =
        static_cast<std::uint64_t>(dimension) * sizeof(double);
    const auto workspace_budget = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(total_bytes) / 3U,
        static_cast<std::uint64_t>(free_bytes) / 2U);
    const auto memory_limited_blocks = static_cast<std::int32_t>(
        std::min<std::uint64_t>(workspace_budget / bytes_per_block,
                                static_cast<std::uint64_t>(blocks)));
    return memory_limited_blocks;
}

// ==========================================================================
// Validation and operator orchestration layer
// ==========================================================================

void validate_initialization(
    const core::CscMatrix& matrix,
    const InitialFactorization& factors) {
    const auto dimension = factors.dimension;
    const auto& lower = factors.lower;
    const auto& upper = factors.upper;
    if (dimension <= 0 || matrix.rows != dimension || matrix.columns != dimension ||
        matrix.row_indices.size() != matrix.values.size() ||
        lower.column_offsets.size() != static_cast<std::size_t>(dimension) + 1U ||
        lower.row_indices.size() != lower.values.size() ||
        upper.column_offsets.size() != static_cast<std::size_t>(dimension) + 1U ||
        upper.row_indices.size() != upper.values.size() ||
        factors.row_permutation.size() != static_cast<std::size_t>(dimension) ||
        factors.column_permutation.size() != static_cast<std::size_t>(dimension)) {
        throw std::runtime_error("invalid custom fixed-pattern LU initialization data");
    }
    if (!factors.off_diagonal.values.empty() || factors.block_boundaries.size() != 2U ||
        factors.block_boundaries.front() != 0 ||
        factors.block_boundaries.back() != dimension ||
        !std::all_of(factors.scale_factors.begin(), factors.scale_factors.end(),
                     [](double value) { return value == 1.0; })) {
        throw std::runtime_error(
            "custom fixed-pattern LU v0 requires one unscaled global factorization");
    }

    std::vector<bool> seen_rows(static_cast<std::size_t>(dimension), false);
    std::vector<bool> seen_columns(static_cast<std::size_t>(dimension), false);
    for (std::int32_t column = 0; column < dimension; ++column) {
        const auto lower_begin = lower.column_offsets[static_cast<std::size_t>(column)];
        const auto lower_end = lower.column_offsets[static_cast<std::size_t>(column) + 1U];
        const auto upper_begin = upper.column_offsets[static_cast<std::size_t>(column)];
        const auto upper_end = upper.column_offsets[static_cast<std::size_t>(column) + 1U];
        if (lower_begin >= lower_end || upper_begin >= upper_end ||
            lower.row_indices[static_cast<std::size_t>(lower_begin)] != column ||
            upper.row_indices[static_cast<std::size_t>(upper_end - 1)] != column) {
            throw std::runtime_error("custom LU requires explicit sorted L/U diagonals");
        }
        const auto row = factors.row_permutation[static_cast<std::size_t>(column)];
        const auto source_column =
            factors.column_permutation[static_cast<std::size_t>(column)];
        if (row < 0 || row >= dimension || source_column < 0 ||
            source_column >= dimension || seen_rows[static_cast<std::size_t>(row)] ||
            seen_columns[static_cast<std::size_t>(source_column)]) {
            throw std::runtime_error("custom LU received an invalid permutation");
        }
        seen_rows[static_cast<std::size_t>(row)] = true;
        seen_columns[static_cast<std::size_t>(source_column)] = true;
    }
}

class FixedPatternLuOperator final : public SparseNumericOperator {
public:
    FixedPatternLuOperator() {
        std::string detail;
        if (!query_device(detail)) {
            throw std::runtime_error("cannot create custom CUDA LU backend: " + detail);
        }
    }

    [[nodiscard]] std::string_view name() const noexcept override {
        return "custom-fixed-pattern-lu";
    }

    double initialize(
        const core::CscMatrix& matrix,
        const InitialFactorization& factors) override {
        // Phase 1: validate KLU output and build immutable host-side schedules.
        validate_initialization(matrix, factors);
        state_.release_buffers();
        state_.dimension = factors.dimension;
        state_.input_nonzeros = static_cast<std::int32_t>(matrix.values.size());
        state_.lower_nonzeros =
            static_cast<std::int32_t>(factors.lower.values.size());
        state_.upper_nonzeros =
            static_cast<std::int32_t>(factors.upper.values.size());
        const auto requested_mode = requested_refactor_mode();
        state_.right_looking_refactor =
            requested_mode != RequestedRefactorMode::left_looking;

        std::vector<std::int32_t> inverse_row_permutation(
            static_cast<std::size_t>(state_.dimension));
        for (std::int32_t index = 0; index < state_.dimension; ++index) {
            inverse_row_permutation[static_cast<std::size_t>(
                factors.row_permutation[static_cast<std::size_t>(index)])] = index;
        }
        const auto forward_schedule =
            build_forward_levels(factors.lower, state_.dimension);
        const auto backward_schedule =
            build_backward_levels(factors.upper, state_.dimension);
        const auto forward_groups = build_level_groups(
            forward_schedule, kSolveThreads / 32);
        const auto backward_groups = build_level_groups(
            backward_schedule, kSolveThreads / 32);
        const auto u_refactor_schedule =
            build_refactor_levels(factors.upper, state_.dimension);
        const auto glu3_refactor_schedule = build_glu3_relaxed_refactor_levels(
            factors.lower, factors.upper, state_.dimension);
        state_.refactor_dag = build_refactor_dag_profile(
            factors.lower, factors.upper, u_refactor_schedule,
            glu3_refactor_schedule, state_.dimension);
        LevelSchedule refactor_schedule;
        std::vector<std::int32_t> refactor_groups;
        RightLookingSchedule right_looking_schedule;
        CompactIndexLayout compact_index_layout;
        std::vector<std::int32_t> input_factor_destinations;

        // Phase 2: select the numerical refactor family. Right-looking GLU3
        // is the default; left-looking paths remain correctness fallbacks.
        if (state_.right_looking_refactor) {
            state_.cooperative_refactor_grid_blocks =
                cooperative_right_refactor_grid_blocks();
            if (state_.cooperative_refactor_grid_blocks <= 0) {
                throw std::runtime_error(
                    "right-looking refactor requires cooperative CUDA launch");
            }
            right_looking_schedule = build_right_looking_schedule(
                factors.upper, glu3_refactor_schedule, state_.dimension);
            input_factor_destinations = build_input_factor_destinations(
                matrix, factors, inverse_row_permutation);
            state_.right_looking_level_count = static_cast<std::int32_t>(
                right_looking_schedule.level_offsets.size() - 1U);
            state_.right_looking_task_count = static_cast<std::int32_t>(
                right_looking_schedule.task_sources.size());
            state_.refactor_level_count = state_.right_looking_level_count;
            state_.refactor_group_count = state_.right_looking_level_count;
        } else if (state_.dimension >= kParallelRefactorMinimumDimension) {
            state_.cooperative_refactor_grid_blocks =
                cooperative_refactor_grid_blocks(state_.dimension);
        }
        if (!state_.right_looking_refactor &&
            state_.cooperative_refactor_grid_blocks > 0) {
            refactor_schedule = u_refactor_schedule;
            refactor_groups = build_level_groups(
                refactor_schedule, kRefactorFusedLevelWidth);
            state_.refactor_level_count = static_cast<std::int32_t>(
                refactor_schedule.offsets.size() - 1U);
            state_.refactor_group_count = static_cast<std::int32_t>(
                refactor_groups.size() - 1U);
            state_.refactor_workspace_elements =
                static_cast<std::uint64_t>(state_.cooperative_refactor_grid_blocks) *
                static_cast<std::uint64_t>(state_.dimension);
        }
        state_.forward_level_count = static_cast<std::int32_t>(
            forward_schedule.offsets.size() - 1U);
        state_.backward_level_count = static_cast<std::int32_t>(
            backward_schedule.offsets.size() - 1U);
        state_.forward_group_count = static_cast<std::int32_t>(
            forward_groups.size() - 1U);
        state_.backward_group_count = static_cast<std::int32_t>(
            backward_groups.size() - 1U);
        state_.schedule_operations =
            state_.refactor_level_count + state_.forward_level_count +
            state_.backward_level_count;
        state_.widest_level = std::max(
            {refactor_schedule.widest_level, forward_schedule.widest_level,
             backward_schedule.widest_level});
        state_.narrow_levels =
            refactor_schedule.narrow_levels + forward_schedule.narrow_levels +
            backward_schedule.narrow_levels;
        state_.cooperative_grid_blocks = cooperative_solve_grid_blocks();
        for (std::int32_t column = 0; column < state_.dimension; ++column) {
            const auto width = factors.lower.column_offsets[
                                   static_cast<std::size_t>(column) + 1U] -
                               factors.lower.column_offsets[
                                   static_cast<std::size_t>(column)] -
                               1;
            state_.widest_column = std::max(state_.widest_column, width);
            if (width <= 8) ++state_.narrow_columns;
        }

        try {
            // Phase 3: allocate long-lived device storage. Static patterns,
            // permutations and schedules remain GPU-resident across calls.
            state_.allocate(state_.input_column_offsets, matrix.column_offsets.size(),
                            "cudaMalloc(custom Ap)");
            state_.allocate(state_.input_row_indices, matrix.row_indices.size(),
                            "cudaMalloc(custom Ai)");
            state_.allocate(state_.input_values, matrix.values.size(),
                            "cudaMalloc(custom Ax)");
            state_.allocate(state_.lower_column_offsets,
                            factors.lower.column_offsets.size(),
                            "cudaMalloc(custom Lp)");
            state_.allocate(state_.lower_row_indices, factors.lower.row_indices.size(),
                            "cudaMalloc(custom Li)");
            state_.allocate(state_.lower_values, factors.lower.values.size(),
                            "cudaMalloc(custom Lx)");
            state_.allocate(state_.upper_column_offsets,
                            factors.upper.column_offsets.size(),
                            "cudaMalloc(custom Up)");
            state_.allocate(state_.upper_row_indices, factors.upper.row_indices.size(),
                            "cudaMalloc(custom Ui)");
            state_.allocate(state_.upper_values, factors.upper.values.size(),
                            "cudaMalloc(custom Ux)");
            state_.allocate(state_.row_permutation, factors.row_permutation.size(),
                            "cudaMalloc(custom P)");
            state_.allocate(state_.inverse_row_permutation,
                            inverse_row_permutation.size(),
                            "cudaMalloc(custom Pinv)");
            state_.allocate(state_.column_permutation,
                            factors.column_permutation.size(),
                            "cudaMalloc(custom Q)");
            state_.allocate(state_.forward_level_offsets,
                            forward_schedule.offsets.size(),
                            "cudaMalloc(custom forward level offsets)");
            state_.allocate(state_.forward_columns,
                            forward_schedule.columns.size(),
                            "cudaMalloc(custom forward columns)");
            state_.allocate(state_.forward_group_offsets,
                            forward_groups.size(),
                            "cudaMalloc(custom forward group offsets)");
            state_.allocate(state_.backward_level_offsets,
                            backward_schedule.offsets.size(),
                            "cudaMalloc(custom backward level offsets)");
            state_.allocate(state_.backward_columns,
                            backward_schedule.columns.size(),
                            "cudaMalloc(custom backward columns)");
            state_.allocate(state_.backward_group_offsets,
                            backward_groups.size(),
                            "cudaMalloc(custom backward group offsets)");
            state_.allocate(state_.refactor_level_offsets,
                            refactor_schedule.offsets.size(),
                            "cudaMalloc(custom refactor level offsets)");
            state_.allocate(state_.refactor_columns,
                            refactor_schedule.columns.size(),
                            "cudaMalloc(custom refactor columns)");
            state_.allocate(state_.refactor_group_offsets,
                            refactor_groups.size(),
                            "cudaMalloc(custom refactor group offsets)");
            state_.allocate(state_.right_hand_side,
                            static_cast<std::size_t>(state_.dimension),
                            "cudaMalloc(custom RHS)");
            state_.allocate(state_.solution, static_cast<std::size_t>(state_.dimension),
                            "cudaMalloc(custom solution)");
            state_.allocate(state_.workspace, static_cast<std::size_t>(state_.dimension),
                            "cudaMalloc(custom workspace)");
            state_.allocate(state_.refactor_workspaces,
                            static_cast<std::size_t>(
                                state_.refactor_workspace_elements),
                            "cudaMalloc(custom dense refactor workspaces)");
            state_.allocate(state_.input_factor_destinations,
                            input_factor_destinations.size(),
                            "cudaMalloc(custom input factor destinations)");
            state_.allocate(state_.right_looking_level_offsets,
                            right_looking_schedule.level_offsets.size(),
                            "cudaMalloc(custom right-looking level offsets)");
            state_.allocate(state_.right_looking_level_columns,
                            right_looking_schedule.level_columns.size(),
                            "cudaMalloc(custom right-looking level columns)");
            state_.allocate(state_.right_looking_task_offsets,
                            right_looking_schedule.task_offsets.size(),
                            "cudaMalloc(custom right-looking task offsets)");
            state_.allocate(state_.right_looking_task_sources,
                            right_looking_schedule.task_sources.size(),
                            "cudaMalloc(custom right-looking task sources)");
            state_.allocate(state_.right_looking_task_targets,
                            right_looking_schedule.task_targets.size(),
                            "cudaMalloc(custom right-looking task targets)");
            state_.allocate(state_.right_looking_task_upper_positions,
                            right_looking_schedule.task_upper_positions.size(),
                            "cudaMalloc(custom right-looking task upper positions)");

            // The direct scalar-update index is selected only when it fits
            // with a safety reserve. Otherwise the same task graph uses the
            // stable sorted-row binary lookup path.
            if (requested_mode ==
                    RequestedRefactorMode::right_looking_adaptive &&
                state_.right_looking_task_count > 0) {
                auto direct_index_layout = build_compact_index_layout(
                    factors.lower, factors.upper,
                    right_looking_schedule);
                if (!direct_index_layout.task_offsets.empty() &&
                    compact_index_layout_fits_device(direct_index_layout)) {
                    compact_index_layout = std::move(direct_index_layout);
                    state_.right_looking_compact_indices = true;
                    state_.right_looking_u16_index_count =
                        compact_index_layout.u16_count;
                    state_.right_looking_u32_index_count =
                        compact_index_layout.u32_count;
                }
            } else if (requested_mode ==
                           RequestedRefactorMode::right_looking_adaptive) {
                state_.right_looking_compact_indices = true;
            }
            state_.allocate(state_.right_looking_task_index_offsets,
                            compact_index_layout.task_offsets.size(),
                            "cudaMalloc(custom right-looking task index offsets)");
            state_.allocate(state_.right_looking_indices_u16,
                            static_cast<std::size_t>(
                                state_.right_looking_u16_index_count),
                            "cudaMalloc(custom right-looking uint16 indices)");
            state_.allocate(state_.right_looking_indices_u32,
                            static_cast<std::size_t>(
                                state_.right_looking_u32_index_count),
                            "cudaMalloc(custom right-looking uint32 indices)");
            state_.allocate(state_.singular_column, 1U,
                            "cudaMalloc(custom singular status)");

            // Phase 4: transfer all immutable data and build optional reusable
            // destination indices before exposing the initialized operator.
            check_cuda(cudaEventRecord(state_.start, state_.stream),
                       "cudaEventRecord(custom initialize start)");
            state_.copy_to_device(state_.input_column_offsets, matrix.column_offsets,
                                  "cudaMemcpyAsync(custom Ap)");
            state_.copy_to_device(state_.input_row_indices, matrix.row_indices,
                                  "cudaMemcpyAsync(custom Ai)");
            state_.copy_to_device(state_.input_values, matrix.values,
                                  "cudaMemcpyAsync(custom Ax)");
            state_.copy_to_device(state_.lower_column_offsets,
                                  factors.lower.column_offsets,
                                  "cudaMemcpyAsync(custom Lp)");
            state_.copy_to_device(state_.lower_row_indices, factors.lower.row_indices,
                                  "cudaMemcpyAsync(custom Li)");
            state_.copy_to_device(state_.lower_values, factors.lower.values,
                                  "cudaMemcpyAsync(custom Lx)");
            state_.copy_to_device(state_.upper_column_offsets,
                                  factors.upper.column_offsets,
                                  "cudaMemcpyAsync(custom Up)");
            state_.copy_to_device(state_.upper_row_indices, factors.upper.row_indices,
                                  "cudaMemcpyAsync(custom Ui)");
            state_.copy_to_device(state_.upper_values, factors.upper.values,
                                  "cudaMemcpyAsync(custom Ux)");
            state_.copy_to_device(state_.row_permutation, factors.row_permutation,
                                  "cudaMemcpyAsync(custom P)");
            state_.copy_to_device(state_.inverse_row_permutation,
                                  inverse_row_permutation,
                                  "cudaMemcpyAsync(custom Pinv)");
            state_.copy_to_device(state_.column_permutation,
                                  factors.column_permutation,
                                  "cudaMemcpyAsync(custom Q)");
            state_.copy_to_device(state_.forward_level_offsets,
                                  forward_schedule.offsets,
                                  "cudaMemcpyAsync(custom forward level offsets)");
            state_.copy_to_device(state_.forward_columns,
                                  forward_schedule.columns,
                                  "cudaMemcpyAsync(custom forward columns)");
            state_.copy_to_device(state_.forward_group_offsets,
                                  forward_groups,
                                  "cudaMemcpyAsync(custom forward group offsets)");
            state_.copy_to_device(state_.backward_level_offsets,
                                  backward_schedule.offsets,
                                  "cudaMemcpyAsync(custom backward level offsets)");
            state_.copy_to_device(state_.backward_columns,
                                  backward_schedule.columns,
                                  "cudaMemcpyAsync(custom backward columns)");
            state_.copy_to_device(state_.backward_group_offsets,
                                  backward_groups,
                                  "cudaMemcpyAsync(custom backward group offsets)");
            state_.copy_to_device(state_.refactor_level_offsets,
                                  refactor_schedule.offsets,
                                  "cudaMemcpyAsync(custom refactor level offsets)");
            state_.copy_to_device(state_.refactor_columns,
                                  refactor_schedule.columns,
                                  "cudaMemcpyAsync(custom refactor columns)");
            state_.copy_to_device(state_.refactor_group_offsets,
                                  refactor_groups,
                                  "cudaMemcpyAsync(custom refactor group offsets)");
            state_.copy_to_device(state_.input_factor_destinations,
                                  input_factor_destinations,
                                  "cudaMemcpyAsync(custom input factor destinations)");
            state_.copy_to_device(state_.right_looking_level_offsets,
                                  right_looking_schedule.level_offsets,
                                  "cudaMemcpyAsync(custom right-looking level offsets)");
            state_.copy_to_device(state_.right_looking_level_columns,
                                  right_looking_schedule.level_columns,
                                  "cudaMemcpyAsync(custom right-looking level columns)");
            state_.copy_to_device(state_.right_looking_task_offsets,
                                  right_looking_schedule.task_offsets,
                                  "cudaMemcpyAsync(custom right-looking task offsets)");
            state_.copy_to_device(state_.right_looking_task_sources,
                                  right_looking_schedule.task_sources,
                                  "cudaMemcpyAsync(custom right-looking task sources)");
            state_.copy_to_device(state_.right_looking_task_targets,
                                  right_looking_schedule.task_targets,
                                  "cudaMemcpyAsync(custom right-looking task targets)");
            state_.copy_to_device(
                state_.right_looking_task_upper_positions,
                right_looking_schedule.task_upper_positions,
                "cudaMemcpyAsync(custom right-looking task upper positions)");
            state_.copy_to_device(state_.right_looking_task_index_offsets,
                                  compact_index_layout.task_offsets,
                                  "cudaMemcpyAsync(custom right-looking task index offsets)");
            check_cuda(cudaMemsetAsync(state_.workspace, 0,
                                       static_cast<std::size_t>(state_.dimension) *
                                           sizeof(double),
                                       state_.stream),
                       "cudaMemsetAsync(custom workspace)");
            if (state_.refactor_workspace_elements > 0) {
                check_cuda(cudaMemsetAsync(
                               state_.refactor_workspaces, 0,
                               static_cast<std::size_t>(
                                   state_.refactor_workspace_elements) * sizeof(double),
                               state_.stream),
                           "cudaMemsetAsync(custom dense refactor workspaces)");
            }
            // Symbolic destination offsets are built once and reused by every
            // numerical refactor. The launch layer owns raw kernel arguments.
            detail::launch_right_looking_index_build(state_);
            check_cuda(cudaEventRecord(state_.after_h2d, state_.stream),
                       "cudaEventRecord(custom initialize end)");
            check_cuda(cudaEventSynchronize(state_.after_h2d),
                       "cudaEventSynchronize(custom initialize)");
            if (state_.right_looking_compact_indices &&
                state_.right_looking_task_count > 0) {
                std::int32_t failed_column{-1};
                check_cuda(cudaMemcpy(&failed_column, state_.singular_column,
                                      sizeof(failed_column),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy(custom index build status D2H)");
                if (failed_column >= 0) {
                    throw std::runtime_error(
                        "right-looking index construction failed at "
                        "permuted column " + std::to_string(failed_column));
                }
            }
            return state_.elapsed(state_.start, state_.after_h2d);
        } catch (...) {
            state_.release_buffers();
            throw;
        }
    }

    [[nodiscard]] NumericFactorTimings factorize(
        const std::vector<double>& matrix_values) override {
        require_ready(matrix_values.size(), "refactor");

        // Only numerical A values cross PCIe here. L/U patterns, permutations,
        // dependency schedules and optional destination indices stay resident.
        check_cuda(cudaEventRecord(state_.start, state_.stream),
                   "cudaEventRecord(custom refactor start)");
        check_cuda(cudaMemcpyAsync(state_.input_values, matrix_values.data(),
                                   static_cast<std::size_t>(byte_count(matrix_values)),
                                   cudaMemcpyHostToDevice, state_.stream),
                   "cudaMemcpyAsync(custom refactor Ax)");
        check_cuda(cudaMemsetAsync(state_.singular_column, 0xff, sizeof(std::int32_t),
                                   state_.stream),
                   "cudaMemsetAsync(custom singular status)");
        check_cuda(cudaEventRecord(state_.after_h2d, state_.stream),
                   "cudaEventRecord(custom refactor H2D)");

        detail::launch_numerical_refactor(state_);
        check_cuda(cudaEventRecord(state_.after_kernel, state_.stream),
                   "cudaEventRecord(custom refactor kernel)");

        std::int32_t singular_column{-1};
        check_cuda(cudaMemcpyAsync(&singular_column, state_.singular_column,
                                   sizeof(singular_column), cudaMemcpyDeviceToHost,
                                   state_.stream),
                   "cudaMemcpyAsync(custom singular status D2H)");
        check_cuda(cudaEventRecord(state_.after_d2h, state_.stream),
                   "cudaEventRecord(custom refactor status)");
        check_cuda(cudaEventSynchronize(state_.after_d2h),
                   "cudaEventSynchronize(custom refactor)");
        if (singular_column >= 0) {
            throw std::runtime_error(
                "custom fixed-pattern LU encountered a zero/non-finite pivot at "
                "permuted column " + std::to_string(singular_column));
        }
        return {state_.elapsed(state_.start, state_.after_h2d),
                state_.elapsed(state_.after_h2d, state_.after_kernel),
                state_.elapsed(state_.after_kernel, state_.after_d2h)};
    }

    [[nodiscard]] SolveTimings solve(
        const std::vector<double>& right_hand_side,
        std::vector<double>& solution) override {
        if (state_.dimension <= 0 ||
            right_hand_side.size() != static_cast<std::size_t>(state_.dimension)) {
            throw std::runtime_error("custom CUDA LU solve dimension mismatch");
        }
        solution.resize(right_hand_side.size());

        // The launch layer chooses persistent level solve or its single-block
        // fallback; this lifecycle layer owns transfer and timing boundaries.
        check_cuda(cudaEventRecord(state_.start, state_.stream),
                   "cudaEventRecord(custom solve start)");
        check_cuda(cudaMemcpyAsync(state_.right_hand_side, right_hand_side.data(),
                                   static_cast<std::size_t>(byte_count(right_hand_side)),
                                   cudaMemcpyHostToDevice, state_.stream),
                   "cudaMemcpyAsync(custom RHS H2D)");
        check_cuda(cudaEventRecord(state_.after_h2d, state_.stream),
                   "cudaEventRecord(custom solve H2D)");

        detail::launch_triangular_solve(state_);
        check_cuda(cudaEventRecord(state_.after_kernel, state_.stream),
                   "cudaEventRecord(custom solve kernel)");
        check_cuda(cudaMemcpyAsync(solution.data(), state_.solution,
                                   static_cast<std::size_t>(byte_count(solution)),
                                   cudaMemcpyDeviceToHost, state_.stream),
                   "cudaMemcpyAsync(custom solution D2H)");
        check_cuda(cudaEventRecord(state_.after_d2h, state_.stream),
                   "cudaEventRecord(custom solve D2H)");
        check_cuda(cudaEventSynchronize(state_.after_d2h),
                   "cudaEventSynchronize(custom solve)");
        return {state_.elapsed(state_.start, state_.after_h2d),
                state_.elapsed(state_.after_h2d, state_.after_kernel),
                state_.elapsed(state_.after_kernel, state_.after_d2h), 1};
    }

    [[nodiscard]] OperatorProfile profile() const override {
        OperatorProfile result;
        if (state_.dimension <= 0) return result;
        if (state_.right_looking_refactor &&
            state_.cooperative_grid_blocks > 0) {
            result.algorithm_mode = state_.right_looking_compact_indices
                ? "custom-glu3-level-right-looking-compact-index-refactor+"
                  "persistent-fused-level-solve"
                : "custom-glu3-level-right-looking-binary-index-refactor+"
                  "persistent-fused-level-solve";
        } else if (state_.cooperative_refactor_grid_blocks > 0 &&
            state_.cooperative_grid_blocks > 0) {
            result.algorithm_mode =
                "custom-persistent-dense-refactor+persistent-fused-level-solve";
        } else if (state_.cooperative_grid_blocks > 0) {
            result.algorithm_mode =
                "custom-serial-refactor+persistent-fused-level-solve";
        } else {
            result.algorithm_mode =
                "custom-serial-refactor+single-block-solve-fallback";
        }
        result.device_memory_bytes = state_.allocated_bytes;
        const auto lower_off_diagonal = static_cast<std::uint64_t>(
            state_.lower_nonzeros - state_.dimension);
        const auto upper_off_diagonal = static_cast<std::uint64_t>(
            state_.upper_nonzeros - state_.dimension);
        // One division per strict-lower entry and one multiply-add (two
        // floating-point operations) per right-looking scalar update.
        result.estimated_numeric_factor_flops =
            lower_off_diagonal + 2U * state_.refactor_dag.scalar_updates;
        // Unit-lower forward solve and upper backward solve each perform a
        // multiply-subtract per strict off-diagonal plus one diagonal divide.
        result.estimated_triangular_solve_flops =
            2U * (lower_off_diagonal + upper_off_diagonal) +
            static_cast<std::uint64_t>(state_.dimension);
        result.schedule_operations = state_.schedule_operations;
        result.persistent_grid_blocks = state_.cooperative_grid_blocks;
        result.fused_operations_per_launch =
            state_.refactor_group_count + state_.forward_group_count +
            state_.backward_group_count;
        result.scheduled_columns =
            (state_.cooperative_refactor_grid_blocks > 0 ? 3 : 2) *
            state_.dimension;
        result.widest_operation_columns = state_.widest_level;
        result.narrow_operations = state_.narrow_levels;
        result.numeric_factor_blocks = state_.cooperative_refactor_grid_blocks > 0
                                     ? state_.cooperative_refactor_grid_blocks
                                     : 1;
        result.numeric_factor_dag = state_.refactor_dag;
        return result;
    }

private:
    void require_ready(std::size_t value_count, const char* operation) const {
        if (state_.dimension <= 0 ||
            value_count != static_cast<std::size_t>(state_.input_nonzeros)) {
            throw std::runtime_error(
                std::string("custom CUDA LU ") + operation +
                " requires the initialized matrix pattern");
        }
    }

    DeviceState state_;
};

}  // namespace

bool fixed_pattern_lu_available(std::string& detail) {
    return query_device(detail);
}

std::unique_ptr<SparseNumericOperator> make_fixed_pattern_lu_operator() {
    return std::make_unique<FixedPatternLuOperator>();
}

}  // namespace eda_gpu::operators::cuda
