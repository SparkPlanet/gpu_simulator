#include "matrix.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace eda_gpu {
namespace {

struct Entry {
    SparseIndex row{};
    SparseIndex column{};
    double value{};
};

[[nodiscard]] std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

[[nodiscard]] std::string next_data_line(std::istream& input) {
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first != std::string::npos && line[first] != '%') return line.substr(first);
    }
    throw std::runtime_error("Matrix Market data line is missing");
}

[[nodiscard]] CscMatrix make_csc(SparseIndex dimension, std::vector<Entry> entries) {
    std::sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
        return std::tie(left.column, left.row) < std::tie(right.column, right.row);
    });

    std::vector<Entry> canonical;
    canonical.reserve(entries.size());
    for (const auto& entry : entries) {
        if (!canonical.empty() && canonical.back().row == entry.row &&
            canonical.back().column == entry.column) {
            canonical.back().value += entry.value;
        } else {
            canonical.push_back(entry);
        }
    }
    if (canonical.size() > static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max())) {
        throw std::runtime_error("matrix exceeds the 32-bit Task 1 index ABI");
    }

    CscMatrix matrix;
    matrix.dimension = dimension;
    matrix.column_offsets.assign(static_cast<std::size_t>(dimension) + 1U, 0);
    matrix.row_indices.reserve(canonical.size());
    matrix.values.reserve(canonical.size());

    std::size_t position{};
    for (SparseIndex column = 0; column < dimension; ++column) {
        matrix.column_offsets[static_cast<std::size_t>(column)] =
            static_cast<SparseIndex>(position);
        while (position < canonical.size() && canonical[position].column == column) {
            matrix.row_indices.push_back(canonical[position].row);
            matrix.values.push_back(canonical[position].value);
            ++position;
        }
    }
    matrix.column_offsets.back() = static_cast<SparseIndex>(position);
    matrix.validate();
    return matrix;
}

}  // namespace

SparseIndex CscMatrix::nonzeros() const noexcept {
    return static_cast<SparseIndex>(values.size());
}

void CscMatrix::validate() const {
    if (dimension <= 0) throw std::runtime_error("Task 1 matrix must be non-empty");
    if (column_offsets.size() != static_cast<std::size_t>(dimension) + 1U ||
        column_offsets.front() != 0 || row_indices.size() != values.size() ||
        column_offsets.back() != static_cast<SparseIndex>(values.size())) {
        throw std::runtime_error("invalid CSC array sizes");
    }
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max())) {
        throw std::runtime_error("CSC nonzero count exceeds 32-bit indices");
    }
    for (SparseIndex column = 0; column < dimension; ++column) {
        const auto begin = column_offsets[static_cast<std::size_t>(column)];
        const auto end = column_offsets[static_cast<std::size_t>(column) + 1U];
        if (begin < 0 || end < begin) throw std::runtime_error("CSC offsets are not monotonic");
        SparseIndex previous = -1;
        for (SparseIndex position = begin; position < end; ++position) {
            const auto row = row_indices[static_cast<std::size_t>(position)];
            if (row < 0 || row >= dimension || row <= previous) {
                throw std::runtime_error("CSC row indices must be sorted and unique");
            }
            if (!std::isfinite(values[static_cast<std::size_t>(position)])) {
                throw std::runtime_error("matrix contains a non-finite value");
            }
            previous = row;
        }
    }
}

MatrixInput load_matrix_market(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open matrix: " + path.string());

    std::string banner, object, format, field, symmetry;
    if (!(input >> banner >> object >> format >> field >> symmetry)) {
        throw std::runtime_error("incomplete Matrix Market header");
    }
    banner = lowercase(banner);
    object = lowercase(object);
    format = lowercase(format);
    field = lowercase(field);
    symmetry = lowercase(symmetry);
    if (banner != "%%matrixmarket" || object != "matrix" || format != "coordinate") {
        throw std::runtime_error("matrix must use Matrix Market coordinate format");
    }
    if (field != "real" && field != "integer" && field != "pattern") {
        throw std::runtime_error("unsupported Matrix Market numeric field");
    }
    if (symmetry != "general" && symmetry != "symmetric" &&
        symmetry != "skew-symmetric") {
        throw std::runtime_error("unsupported Matrix Market symmetry");
    }

    std::getline(input, banner);
    std::istringstream dimensions(next_data_line(input));
    std::int64_t rows{}, columns{}, stored_entries{};
    if (!(dimensions >> rows >> columns >> stored_entries) || rows <= 0 || rows != columns ||
        stored_entries < 0 || rows > std::numeric_limits<SparseIndex>::max()) {
        throw std::runtime_error("Task 1 requires a square 32-bit-index matrix");
    }

    const bool mirrored = symmetry != "general";
    const auto maximum_entries = static_cast<std::uint64_t>(stored_entries) *
                                 static_cast<std::uint64_t>(mirrored ? 2U : 1U);
    if (maximum_entries > static_cast<std::uint64_t>(std::numeric_limits<SparseIndex>::max())) {
        throw std::runtime_error("expanded matrix exceeds the 32-bit Task 1 index ABI");
    }
    std::vector<Entry> entries;
    entries.reserve(static_cast<std::size_t>(maximum_entries));
    for (std::int64_t position = 0; position < stored_entries; ++position) {
        std::int64_t row{}, column{};
        double value = 1.0;
        if (!(input >> row >> column) || (field != "pattern" && !(input >> value)) ||
            row <= 0 || row > rows || column <= 0 || column > columns ||
            !std::isfinite(value)) {
            throw std::runtime_error("invalid Matrix Market entry");
        }
        const auto zero_row = static_cast<SparseIndex>(row - 1);
        const auto zero_column = static_cast<SparseIndex>(column - 1);
        entries.push_back({zero_row, zero_column, value});
        if (mirrored && zero_row != zero_column) {
            entries.push_back(
                {zero_column, zero_row, symmetry == "skew-symmetric" ? -value : value});
        }
    }
    return {make_csc(static_cast<SparseIndex>(rows), std::move(entries)), path.string()};
}

std::vector<double> load_matrix_market_vector(
    const std::filesystem::path& path,
    SparseIndex expected_size) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open right-hand side: " + path.string());

    std::string banner, object, format, field, symmetry;
    if (!(input >> banner >> object >> format >> field >> symmetry)) {
        throw std::runtime_error("incomplete Matrix Market vector header");
    }
    banner = lowercase(banner);
    object = lowercase(object);
    format = lowercase(format);
    field = lowercase(field);
    symmetry = lowercase(symmetry);
    if (banner != "%%matrixmarket" || object != "matrix" ||
        (format != "array" && format != "coordinate") ||
        (field != "real" && field != "integer") || symmetry != "general") {
        throw std::runtime_error("RHS must be a real general Matrix Market vector");
    }

    std::getline(input, banner);
    std::istringstream dimensions(next_data_line(input));
    std::int64_t rows{}, columns{}, entries{};
    if (!(dimensions >> rows >> columns) || rows != expected_size || columns != 1) {
        throw std::runtime_error("right-hand-side dimensions do not match the matrix");
    }

    std::vector<double> result(static_cast<std::size_t>(expected_size), 0.0);
    if (format == "array") {
        for (auto& value : result) {
            if (!(input >> value) || !std::isfinite(value)) {
                throw std::runtime_error("invalid Matrix Market RHS value");
            }
        }
        return result;
    }

    if (!(dimensions >> entries) || entries < 0) {
        throw std::runtime_error("invalid Matrix Market RHS entry count");
    }
    for (std::int64_t position = 0; position < entries; ++position) {
        std::int64_t row{}, column{};
        double value{};
        if (!(input >> row >> column >> value) || row <= 0 || row > rows || column != 1 ||
            !std::isfinite(value)) {
            throw std::runtime_error("invalid Matrix Market RHS coordinate");
        }
        result[static_cast<std::size_t>(row - 1)] += value;
    }
    return result;
}

MatrixInput make_poisson_2d(SparseIndex grid_size) {
    if (grid_size <= 1 || static_cast<std::int64_t>(grid_size) * grid_size >
                              std::numeric_limits<SparseIndex>::max()) {
        throw std::runtime_error("grid size is outside the Task 1 index range");
    }
    const auto dimension = static_cast<SparseIndex>(grid_size * grid_size);
    std::vector<Entry> entries;
    entries.reserve(static_cast<std::size_t>(dimension) * 5U);
    for (SparseIndex row = 0; row < grid_size; ++row) {
        for (SparseIndex column = 0; column < grid_size; ++column) {
            const auto index = static_cast<SparseIndex>(row * grid_size + column);
            entries.push_back({index, index, 4.0});
            if (row > 0) entries.push_back({index, index - grid_size, -1.0});
            if (row + 1 < grid_size) entries.push_back({index, index + grid_size, -1.0});
            if (column > 0) entries.push_back({index, index - 1, -1.0});
            if (column + 1 < grid_size) entries.push_back({index, index + 1, -1.0});
        }
    }
    return {make_csc(dimension, std::move(entries)),
            "generated:poisson2d:grid=" + std::to_string(grid_size)};
}

std::vector<double> multiply(const CscMatrix& matrix, const std::vector<double>& vector) {
    if (vector.size() != static_cast<std::size_t>(matrix.dimension)) {
        throw std::runtime_error("matrix-vector dimension mismatch");
    }
    std::vector<double> result(static_cast<std::size_t>(matrix.dimension), 0.0);
    for (SparseIndex column = 0; column < matrix.dimension; ++column) {
        for (auto position = matrix.column_offsets[static_cast<std::size_t>(column)];
             position < matrix.column_offsets[static_cast<std::size_t>(column) + 1U];
             ++position) {
            result[static_cast<std::size_t>(matrix.row_indices[static_cast<std::size_t>(position)])] +=
                matrix.values[static_cast<std::size_t>(position)] *
                vector[static_cast<std::size_t>(column)];
        }
    }
    return result;
}

double relative_residual_l2(
    const CscMatrix& matrix,
    const std::vector<double>& solution,
    const std::vector<double>& right_hand_side) {
    const auto product = multiply(matrix, solution);
    if (product.size() != right_hand_side.size()) {
        throw std::runtime_error("residual vector dimension mismatch");
    }
    double residual_squared{};
    double right_hand_side_squared{};
    for (std::size_t index = 0; index < product.size(); ++index) {
        const auto difference = product[index] - right_hand_side[index];
        residual_squared += difference * difference;
        right_hand_side_squared += right_hand_side[index] * right_hand_side[index];
    }
    return std::sqrt(residual_squared) /
           std::max(std::sqrt(right_hand_side_squared),
                    std::numeric_limits<double>::min());
}

}  // namespace eda_gpu
