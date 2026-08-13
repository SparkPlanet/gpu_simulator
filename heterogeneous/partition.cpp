#include "partition.hpp"

#include <algorithm>
#include <stdexcept>

namespace eda_gpu::heterogeneous {
namespace {

RegionWorkload measure_region(
    const AnalysisPlan& analysis,
    SparseIndex row_begin,
    SparseIndex row_end) {
    RegionWorkload workload;
    workload.rows = static_cast<std::uint64_t>(row_end - row_begin);

    for (auto row = row_begin; row < row_end; ++row) {
        const auto row_index = static_cast<std::size_t>(row);
        const auto factor_begin = analysis.lu_row_offsets[row_index];
        const auto factor_end = analysis.lu_row_offsets[row_index + 1U];
        workload.factor_nonzeros +=
            static_cast<std::uint64_t>(factor_end - factor_begin);

        const auto updates = analysis.factor_row_updates[row_index];
        workload.scalar_updates += updates;
        workload.maximum_row_updates =
            std::max(workload.maximum_row_updates, updates);
        if (updates < 32U) {
            ++workload.rows_updates_lt_32;
        } else if (updates < 128U) {
            ++workload.rows_updates_32_to_127;
        } else if (updates < 4096U) {
            ++workload.rows_updates_128_to_4095;
        } else {
            ++workload.rows_updates_ge_4096;
        }
    }
    return workload;
}

}  // namespace

void PartitionPlan::validate_candidate_cover(const AnalysisPlan& analysis) const {
    if (dimension != analysis.dimension || regions.empty()) {
        throw std::logic_error("heterogeneous candidate plan has invalid dimensions");
    }

    SparseIndex expected_begin{};
    std::uint64_t measured_rows{};
    for (const auto& region : regions) {
        if (region.row_begin != expected_begin || region.row_end <= region.row_begin ||
            region.row_end > dimension) {
            throw std::logic_error(
                "heterogeneous candidate regions do not form a contiguous cover");
        }
        if (region.workload.rows !=
            static_cast<std::uint64_t>(region.row_end - region.row_begin)) {
            throw std::logic_error("heterogeneous candidate workload has invalid row count");
        }
        measured_rows += region.workload.rows;
        expected_begin = region.row_end;
    }

    if (expected_begin != dimension || measured_rows != static_cast<std::uint64_t>(dimension)) {
        throw std::logic_error("heterogeneous candidate plan does not own every row");
    }
}

PartitionPlan CandidateBuilder::build_btf_candidates(
    const AnalysisPlan& analysis,
    Profiler& profiler) const {
    auto event = profiler.scoped("heterogeneous_btf_candidate_build", EventKind::event);
    if (analysis.block_offsets.size() < 2U ||
        analysis.factor_row_updates.size() !=
            static_cast<std::size_t>(analysis.dimension)) {
        throw std::logic_error("heterogeneous planning requires a complete analysis plan");
    }

    PartitionPlan result;
    result.dimension = analysis.dimension;
    result.regions.reserve(analysis.block_offsets.size() - 1U);

    std::uint64_t short_rows{};
    std::uint64_t scalar_updates{};
    for (std::size_t block = 0; block + 1U < analysis.block_offsets.size(); ++block) {
        Region region;
        region.row_begin = analysis.block_offsets[block];
        region.row_end = analysis.block_offsets[block + 1U];
        region.input_symmetry = analysis.block_input_symmetry[block];
        region.structurally_symmetric =
            analysis.structurally_symmetric_blocks[block] != 0U;
        region.workload = measure_region(analysis, region.row_begin, region.row_end);
        short_rows += region.workload.rows_updates_lt_32;
        scalar_updates += region.workload.scalar_updates;
        result.regions.push_back(region);
    }

    result.validate_candidate_cover(analysis);
    profiler.add_value("candidate_regions", result.regions.size());
    profiler.add_value("candidate_rows", result.dimension);
    profiler.add_value("candidate_short_rows", short_rows);
    profiler.add_value("candidate_scalar_updates", scalar_updates);
    profiler.add_attribute("candidate_kind", "BTF diagonal blocks");
    profiler.add_attribute("ownership", "unassigned; cost model required");
    return result;
}

}  // namespace eda_gpu::heterogeneous
