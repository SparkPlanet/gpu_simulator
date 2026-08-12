#pragma once

#include <cstdint>

namespace eda_gpu::core {

struct LevelWidthProfile {
    std::int32_t levels{};
    std::int32_t width_1{};
    std::int32_t width_2{};
    std::int32_t width_3_to_8{};
    std::int32_t width_9_to_32{};
    std::int32_t width_33_plus{};
    std::int32_t widest{};
};

// Static fixed-pattern diagnostics used to choose the numerical refactor
// algorithm. All counts are derived once from the reusable L/U structure.
struct NumericFactorDagProfile {
    bool available{};
    LevelWidthProfile u_dependency_levels;
    LevelWidthProfile glu3_relaxed_levels;

    std::uint64_t subcolumn_tasks{};
    std::uint64_t scalar_updates{};
    std::uint64_t single_level_subcolumn_tasks{};
    std::uint64_t single_level_scalar_updates{};
    double single_level_scalar_update_fraction{};

    std::int32_t single_level_fanout_p50{};
    std::int32_t single_level_fanout_p90{};
    std::int32_t single_level_fanout_p99{};
    std::int32_t single_level_fanout_max{};

    std::uint64_t same_level_collision_groups{};
    std::uint64_t same_level_conflicting_tasks{};
    std::uint64_t same_level_extra_writers{};
    std::int32_t same_level_max_writers{};

    std::uint64_t full_index_bytes_u32{};
    std::uint64_t full_index_bytes_mixed_u16_u32{};
    std::uint64_t single_level_index_bytes_mixed_u16_u32{};
    std::uint64_t dense_workspace_bytes_per_block{};
};

}  // namespace eda_gpu::core
