#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace eda_gpu {

using SparseIndex = std::int32_t;

// Canonical Task 1 host representation: a square, real CSC matrix.
// Indices in every column are sorted and unique. Backends may build their own
// device format, but that representation never leaks through the public API.
struct CscMatrix {
    SparseIndex dimension{};
    std::vector<SparseIndex> column_offsets;
    std::vector<SparseIndex> row_indices;
    std::vector<double> values;

    [[nodiscard]] SparseIndex nonzeros() const noexcept;
    void validate() const;
};

struct MatrixInput {
    CscMatrix matrix;
    std::string description;
};

[[nodiscard]] MatrixInput load_matrix_market(const std::filesystem::path& path);
[[nodiscard]] std::vector<double> load_matrix_market_vector(
    const std::filesystem::path& path,
    SparseIndex expected_size);
[[nodiscard]] MatrixInput make_poisson_2d(SparseIndex grid_size);
[[nodiscard]] std::vector<double> multiply(
    const CscMatrix& matrix,
    const std::vector<double>& vector);
[[nodiscard]] double relative_residual_l2(
    const CscMatrix& matrix,
    const std::vector<double>& solution,
    const std::vector<double>& right_hand_side);

}  // namespace eda_gpu
