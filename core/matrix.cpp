#include "matrix.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace eda_gpu::core {
namespace {

struct Triplet {
    std::int32_t row{};
    std::int32_t column{};
    double value{};
};

[[nodiscard]] std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

[[nodiscard]] CscMatrix from_triplets(
    std::int32_t rows,
    std::int32_t columns,
    std::vector<Triplet> triplets) {
    std::sort(triplets.begin(), triplets.end(), [](const Triplet& lhs, const Triplet& rhs) {
        return std::tie(lhs.column, lhs.row) < std::tie(rhs.column, rhs.row);
    });

    std::vector<Triplet> merged;
    merged.reserve(triplets.size());
    for (const auto& item : triplets) {
        if (!merged.empty() && merged.back().row == item.row &&
            merged.back().column == item.column) {
            merged.back().value += item.value;
        } else {
            merged.push_back(item);
        }
    }
    if (merged.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("matrix has more nonzeros than the 32-bit solver ABI supports");
    }

    CscMatrix matrix;
    matrix.rows = rows;
    matrix.columns = columns;
    matrix.column_offsets.assign(static_cast<std::size_t>(columns) + 1U, 0);
    matrix.row_indices.reserve(merged.size());
    matrix.values.reserve(merged.size());

    std::size_t position = 0;
    for (std::int32_t column = 0; column < columns; ++column) {
        matrix.column_offsets[static_cast<std::size_t>(column)] =
            static_cast<std::int32_t>(position);
        while (position < merged.size() && merged[position].column == column) {
            matrix.row_indices.push_back(merged[position].row);
            matrix.values.push_back(merged[position].value);
            ++position;
        }
    }
    matrix.column_offsets.back() = static_cast<std::int32_t>(position);
    matrix.validate();
    return matrix;
}

}  // namespace

std::int32_t CscMatrix::nonzeros() const {
    return static_cast<std::int32_t>(values.size());
}

void CscMatrix::validate() const {
    if (rows <= 0 || columns <= 0 || rows != columns) {
        throw std::runtime_error("solver benchmark requires a non-empty square matrix");
    }
    if (column_offsets.size() != static_cast<std::size_t>(columns) + 1U ||
        column_offsets.front() != 0 ||
        column_offsets.back() != static_cast<std::int32_t>(values.size()) ||
        row_indices.size() != values.size()) {
        throw std::runtime_error("invalid CSC array sizes");
    }
    for (std::int32_t column = 0; column < columns; ++column) {
        const auto begin = column_offsets[static_cast<std::size_t>(column)];
        const auto end = column_offsets[static_cast<std::size_t>(column) + 1U];
        if (begin < 0 || begin > end) {
            throw std::runtime_error("CSC column offsets are not monotonic");
        }
        std::int32_t previous = -1;
        for (auto index = begin; index < end; ++index) {
            const auto row = row_indices[static_cast<std::size_t>(index)];
            if (row < 0 || row >= rows || row <= previous) {
                throw std::runtime_error("CSC row indices must be sorted and unique per column");
            }
            if (!std::isfinite(values[static_cast<std::size_t>(index)])) {
                throw std::runtime_error("matrix contains a non-finite value");
            }
            previous = row;
        }
    }
}

MatrixInput load_matrix_market(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open Matrix Market file: " + path.string());
    }

    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    if (!(input >> banner >> object >> format >> field >> symmetry)) {
        throw std::runtime_error("incomplete Matrix Market header");
    }
    banner = lowercase(banner);
    object = lowercase(object);
    format = lowercase(format);
    field = lowercase(field);
    symmetry = lowercase(symmetry);
    if (banner != "%%matrixmarket" || object != "matrix" || format != "coordinate") {
        throw std::runtime_error("only Matrix Market coordinate matrices are supported");
    }
    if (field != "real" && field != "integer" && field != "pattern") {
        throw std::runtime_error("only real, integer and pattern Matrix Market fields are supported");
    }
    if (symmetry != "general" && symmetry != "symmetric" && symmetry != "skew-symmetric") {
        throw std::runtime_error("unsupported Matrix Market symmetry: " + symmetry);
    }

    std::string line;
    std::getline(input, line);
    do {
        if (!std::getline(input, line)) {
            throw std::runtime_error("Matrix Market size line is missing");
        }
    } while (line.empty() || line.front() == '%');

    std::int64_t rows64{};
    std::int64_t columns64{};
    std::int64_t entries64{};
    std::istringstream dimensions(line);
    if (!(dimensions >> rows64 >> columns64 >> entries64) || rows64 <= 0 ||
        columns64 <= 0 || entries64 < 0 ||
        rows64 > std::numeric_limits<std::int32_t>::max() ||
        columns64 > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("invalid or unsupported Matrix Market dimensions");
    }
    if (rows64 != columns64) {
        throw std::runtime_error("solver benchmark requires a square Matrix Market matrix");
    }

    const bool mirrored = symmetry != "general";
    const auto reserve_count = static_cast<std::uint64_t>(entries64) * (mirrored ? 2U : 1U);
    if (reserve_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("expanded matrix exceeds the 32-bit solver ABI");
    }
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(reserve_count));

    for (std::int64_t entry = 0; entry < entries64; ++entry) {
        std::int64_t row64{};
        std::int64_t column64{};
        double value = 1.0;
        if (!(input >> row64 >> column64) || (field != "pattern" && !(input >> value))) {
            throw std::runtime_error("invalid Matrix Market entry at position " +
                                     std::to_string(entry + 1));
        }
        if (row64 <= 0 || row64 > rows64 || column64 <= 0 || column64 > columns64 ||
            !std::isfinite(value)) {
            throw std::runtime_error("out-of-range or non-finite Matrix Market entry");
        }
        const auto row = static_cast<std::int32_t>(row64 - 1);
        const auto column = static_cast<std::int32_t>(column64 - 1);
        triplets.push_back({row, column, value});
        if (mirrored && row != column) {
            triplets.push_back({column, row, symmetry == "skew-symmetric" ? -value : value});
        }
    }

    return {from_triplets(static_cast<std::int32_t>(rows64),
                          static_cast<std::int32_t>(columns64), std::move(triplets)),
            path.string()};
}

std::vector<double> load_matrix_market_vector(
    const std::filesystem::path& path,
    std::int32_t expected_size) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open Matrix Market vector: " + path.string());
    }

    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    if (!(input >> banner >> object >> format >> field >> symmetry)) {
        throw std::runtime_error("incomplete Matrix Market vector header");
    }
    banner = lowercase(banner);
    object = lowercase(object);
    format = lowercase(format);
    field = lowercase(field);
    symmetry = lowercase(symmetry);
    if (banner != "%%matrixmarket" || object != "matrix" ||
        (format != "array" && format != "coordinate")) {
        throw std::runtime_error("RHS must be a Matrix Market array or coordinate matrix");
    }
    if (field != "real" && field != "integer") {
        throw std::runtime_error("RHS must use a real or integer Matrix Market field");
    }
    if (symmetry != "general") {
        throw std::runtime_error("RHS Matrix Market symmetry must be general");
    }

    std::string line;
    std::getline(input, line);
    do {
        if (!std::getline(input, line)) {
            throw std::runtime_error("Matrix Market RHS size line is missing");
        }
    } while (line.empty() || line.front() == '%');

    std::int64_t rows{};
    std::int64_t columns{};
    std::int64_t entries{};
    std::istringstream dimensions(line);
    if (!(dimensions >> rows >> columns) || rows != expected_size || columns != 1) {
        throw std::runtime_error("RHS dimensions do not match the coefficient matrix");
    }

    std::vector<double> vector(static_cast<std::size_t>(expected_size), 0.0);
    if (format == "array") {
        for (std::int32_t row = 0; row < expected_size; ++row) {
            if (!(input >> vector[static_cast<std::size_t>(row)]) ||
                !std::isfinite(vector[static_cast<std::size_t>(row)])) {
                throw std::runtime_error("invalid or non-finite Matrix Market RHS value");
            }
        }
        return vector;
    }

    if (!(dimensions >> entries) || entries < 0) {
        throw std::runtime_error("invalid Matrix Market RHS entry count");
    }
    for (std::int64_t entry = 0; entry < entries; ++entry) {
        std::int64_t row{};
        std::int64_t column{};
        double value{};
        if (!(input >> row >> column >> value) || row <= 0 || row > rows || column != 1 ||
            !std::isfinite(value)) {
            throw std::runtime_error("invalid Matrix Market RHS coordinate entry");
        }
        vector[static_cast<std::size_t>(row - 1)] += value;
    }
    return vector;
}

MatrixInput make_poisson_2d(std::int32_t grid_size) {
    if (grid_size <= 1 ||
        static_cast<std::int64_t>(grid_size) * grid_size >
            std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("--grid must produce 2 to INT32_MAX unknowns");
    }
    const auto dimension = static_cast<std::int32_t>(grid_size * grid_size);
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(dimension) * 5U);
    for (std::int32_t row = 0; row < grid_size; ++row) {
        for (std::int32_t column = 0; column < grid_size; ++column) {
            const auto index = static_cast<std::int32_t>(row * grid_size + column);
            triplets.push_back({index, index, 4.0});
            if (row > 0) triplets.push_back({index, index - grid_size, -1.0});
            if (row + 1 < grid_size) triplets.push_back({index, index + grid_size, -1.0});
            if (column > 0) triplets.push_back({index, index - 1, -1.0});
            if (column + 1 < grid_size) triplets.push_back({index, index + 1, -1.0});
        }
    }
    return {from_triplets(dimension, dimension, std::move(triplets)),
            "generated:poisson2d:grid=" + std::to_string(grid_size)};
}

std::vector<double> multiply(const CscMatrix& matrix, const std::vector<double>& vector) {
    if (vector.size() != static_cast<std::size_t>(matrix.columns)) {
        throw std::runtime_error("matrix-vector dimension mismatch");
    }
    std::vector<double> result(static_cast<std::size_t>(matrix.rows), 0.0);
    for (std::int32_t column = 0; column < matrix.columns; ++column) {
        for (auto index = matrix.column_offsets[static_cast<std::size_t>(column)];
             index < matrix.column_offsets[static_cast<std::size_t>(column) + 1U]; ++index) {
            result[static_cast<std::size_t>(matrix.row_indices[static_cast<std::size_t>(index)])] +=
                matrix.values[static_cast<std::size_t>(index)] *
                vector[static_cast<std::size_t>(column)];
        }
    }
    return result;
}

}  // namespace eda_gpu::core
