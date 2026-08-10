#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace eda_gpu::core {

struct CscMatrix {
    std::int32_t rows{};
    std::int32_t columns{};
    std::vector<std::int32_t> column_offsets;
    std::vector<std::int32_t> row_indices;
    std::vector<double> values;

    [[nodiscard]] std::int32_t nonzeros() const;
    void validate() const;
};

struct MatrixInput {
    CscMatrix matrix;
    std::string description;
};

[[nodiscard]] MatrixInput load_matrix_market(const std::filesystem::path& path);
[[nodiscard]] std::vector<double> load_matrix_market_vector(
    const std::filesystem::path& path,
    std::int32_t expected_size);
[[nodiscard]] MatrixInput make_poisson_2d(std::int32_t grid_size);
[[nodiscard]] std::vector<double> multiply(
    const CscMatrix& matrix,
    const std::vector<double>& vector);

}  // namespace eda_gpu::core
