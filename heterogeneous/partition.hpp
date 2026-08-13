#pragma once

#include "analysis.hpp"
#include "profiler.hpp"

#include <cstdint>
#include <vector>

namespace eda_gpu::heterogeneous {

// Ownership is deliberately independent of the structural region type. The
// planner may assign a BTF block, an elimination subtree, or a frontal region
// to either processor without changing the numerical executor interface.
enum class RegionOwner {
    undecided,
    cpu,
    gpu,
    separator,
};

enum class RegionKind {
    btf_block,
    elimination_subtree,
    frontal_region,
};

// Structural measurements used by the future cost model. They describe work
// rather than prescribe a CPU/GPU threshold, so calibration remains portable
// between CUDA and MXMACA devices.
struct RegionWorkload {
    std::uint64_t rows{};
    std::uint64_t factor_nonzeros{};
    std::uint64_t scalar_updates{};
    std::uint64_t maximum_row_updates{};
    std::uint64_t rows_updates_lt_32{};
    std::uint64_t rows_updates_32_to_127{};
    std::uint64_t rows_updates_128_to_4095{};
    std::uint64_t rows_updates_ge_4096{};
};

struct Region {
    RegionKind kind{RegionKind::btf_block};
    RegionOwner owner{RegionOwner::undecided};
    SparseIndex row_begin{};
    SparseIndex row_end{};
    double input_symmetry{-1.0};
    bool structurally_symmetric{};
    RegionWorkload workload;
};

struct PartitionPlan {
    SparseIndex dimension{};
    std::vector<Region> regions;

    // The initial candidate builder emits a disjoint row cover. Later
    // separator-aware plans may add interface metadata, but must still own
    // every permuted row exactly once.
    void validate_candidate_cover(const AnalysisPlan& analysis) const;
};

// First-stage structural decomposition for the heterogeneous solver. It does
// not decide ownership yet: assigning regions requires measured CPU/GPU costs,
// separator traffic, achieved occupancy, and useful DRAM bandwidth.
class CandidateBuilder {
public:
    [[nodiscard]] PartitionPlan build_btf_candidates(
        const AnalysisPlan& analysis,
        Profiler& profiler) const;
};

}  // namespace eda_gpu::heterogeneous
