#include "matrix.hpp"
#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef EDA_GPU_ROOT
#define EDA_GPU_ROOT "."
#endif

namespace {

using Clock = std::chrono::steady_clock;
using eda_gpu::core::CscMatrix;

struct Options {
    std::string solver{"klu"};
    std::optional<std::string> case_name;
    std::optional<std::filesystem::path> matrix;
    std::optional<std::filesystem::path> rhs;
    std::optional<std::int32_t> grid;
    std::int32_t warmup{1};
    std::int32_t iterations{5};
    double tolerance{1e-6};
    std::optional<std::filesystem::path> output;
    bool list_solvers{};
    bool list_cases{};
    bool help{};
};

struct CaseDescriptor {
    std::string_view name;
    std::int32_t rows;
    std::int32_t nonzeros;
    bool has_right_hand_side;
    std::string_view source;
};

constexpr CaseDescriptor kCases[]{
    {"asic_320k", 321821, 2635364, false, "SuiteSparse/Sandia"},
    {"asic_680k", 682862, 3871773, false, "SuiteSparse/Sandia"},
    {"circuit5m_dc", 3523317, 19194193, false, "SuiteSparse/Freescale"},
};

void resolve_case(Options& options) {
    if (!options.case_name) return;
    const auto found = std::find_if(std::begin(kCases), std::end(kCases), [&](const auto& item) {
        return item.name == *options.case_name;
    });
    if (found == std::end(kCases)) {
        throw std::runtime_error("unknown matrix case: " + *options.case_name);
    }
    const std::filesystem::path root = EDA_GPU_ROOT;
    options.matrix = root / "datasets/matrix" / (*options.case_name + ".mtx");
    if (found->has_right_hand_side) {
        options.rhs = root / "datasets/matrix" / (*options.case_name + "_b.mtx");
    }
}

struct Summary {
    double mean{};
    double median{};
    double minimum{};
    double maximum{};
};

[[nodiscard]] std::int32_t parse_integer(std::string_view text, std::string_view option) {
    std::size_t consumed{};
    const auto value = std::stoll(std::string(text), &consumed);
    if (consumed != text.size() || value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("invalid integer for " + std::string(option));
    }
    return static_cast<std::int32_t>(value);
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    auto value_after = [&](int& index, std::string_view option) -> std::string_view {
        if (++index >= argc) {
            throw std::runtime_error(std::string(option) + " requires a value");
        }
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--solver") {
            options.solver = value_after(index, argument);
        } else if (argument == "--case") {
            options.case_name = std::string(value_after(index, argument));
        } else if (argument == "--matrix") {
            options.matrix = value_after(index, argument);
        } else if (argument == "--rhs") {
            options.rhs = value_after(index, argument);
        } else if (argument == "--grid") {
            options.grid = parse_integer(value_after(index, argument), argument);
        } else if (argument == "--warmup") {
            options.warmup = parse_integer(value_after(index, argument), argument);
        } else if (argument == "--iterations") {
            options.iterations = parse_integer(value_after(index, argument), argument);
        } else if (argument == "--tolerance") {
            std::size_t consumed{};
            const auto text = value_after(index, argument);
            options.tolerance = std::stod(std::string(text), &consumed);
            if (consumed != text.size()) {
                throw std::runtime_error("invalid floating-point value for --tolerance");
            }
        } else if (argument == "--output") {
            options.output = value_after(index, argument);
        } else if (argument == "--list-solvers") {
            options.list_solvers = true;
        } else if (argument == "--list-cases") {
            options.list_cases = true;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }
    if (!options.help && !options.list_solvers && !options.list_cases) {
        const auto input_count = static_cast<int>(options.case_name.has_value()) +
                                 static_cast<int>(options.matrix.has_value()) +
                                 static_cast<int>(options.grid.has_value());
        if (input_count != 1) {
            throw std::runtime_error(
                "choose exactly one input: --case NAME, --matrix FILE or --grid N");
        }
        if (options.rhs && !options.matrix && !options.case_name) {
            throw std::runtime_error("--rhs requires --matrix");
        }
        if (options.case_name && options.rhs) {
            throw std::runtime_error("--case already selects its matching RHS");
        }
        if (options.warmup < 0 || options.iterations <= 0 ||
            !std::isfinite(options.tolerance) || options.tolerance <= 0.0) {
            throw std::runtime_error("warmup must be >= 0; iterations and tolerance must be > 0");
        }
    }
    return options;
}

void print_help(std::ostream& output) {
    output <<
        "Usage: solver_benchmark [options]\n"
        "  --solver NAME       klu, cuda or metax (default: klu)\n"
        "  --case NAME         bundled circuit matrix and RHS\n"
        "  --matrix FILE       Matrix Market coordinate matrix\n"
        "  --rhs FILE          Matrix Market right-hand side (array or coordinate)\n"
        "  --grid N            generate an N-by-N 2D Poisson grid\n"
        "  --warmup N          unreported refactor+solve runs (default: 1)\n"
        "  --iterations N      measured refactor+solve runs (default: 5)\n"
        "  --tolerance VALUE   relative solution error limit (default: 1e-6)\n"
        "  --output FILE       write JSON to a file instead of stdout\n"
        "  --list-solvers      show registered adapters\n"
        "  --list-cases        show bundled circuit matrices\n";
}

template <typename Callable>
[[nodiscard]] double timed_ms(Callable&& callable) {
    const auto start = Clock::now();
    std::forward<Callable>(callable)();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

[[nodiscard]] Summary summarize(const std::vector<double>& samples) {
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2U;
    const auto median = sorted.size() % 2U == 0U
                            ? (sorted[middle - 1U] + sorted[middle]) / 2.0
                            : sorted[middle];
    return {std::accumulate(samples.begin(), samples.end(), 0.0) /
                static_cast<double>(samples.size()),
            median, sorted.front(), sorted.back()};
}

[[nodiscard]] double relative_residual(
    const CscMatrix& matrix,
    const std::vector<double>& solution,
    const std::vector<double>& right_hand_side) {
    const auto product = eda_gpu::core::multiply(matrix, solution);
    double residual_squared = 0.0;
    double rhs_squared = 0.0;
    for (std::size_t index = 0; index < product.size(); ++index) {
        const auto difference = product[index] - right_hand_side[index];
        residual_squared += difference * difference;
        rhs_squared += right_hand_side[index] * right_hand_side[index];
    }
    return std::sqrt(residual_squared) /
           std::max(std::sqrt(rhs_squared), std::numeric_limits<double>::min());
}

[[nodiscard]] double relative_solution_error(
    const std::vector<double>& solution,
    const std::vector<double>& reference) {
    if (solution.size() != reference.size()) {
        throw std::runtime_error("solution/reference dimension mismatch");
    }
    double error_squared = 0.0;
    double reference_squared = 0.0;
    for (std::size_t index = 0; index < solution.size(); ++index) {
        const auto difference = solution[index] - reference[index];
        error_squared += difference * difference;
        reference_squared += reference[index] * reference[index];
    }
    return std::sqrt(error_squared) /
           std::max(std::sqrt(reference_squared), std::numeric_limits<double>::min());
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += character; break;
        }
    }
    return escaped;
}

void write_samples(std::ostream& output, const std::vector<double>& samples) {
    output << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0U) output << ", ";
        output << samples[index];
    }
    output << ']';
}

void write_timing(
    std::ostream& output,
    std::string_view name,
    const std::vector<double>& samples,
    bool trailing_comma) {
    const auto summary = summarize(samples);
    output << "    \"" << name << "\": {\n"
           << "      \"mean\": " << summary.mean << ",\n"
           << "      \"median\": " << summary.median << ",\n"
           << "      \"min\": " << summary.minimum << ",\n"
           << "      \"max\": " << summary.maximum << ",\n"
           << "      \"samples\": ";
    write_samples(output, samples);
    output << "\n    }" << (trailing_comma ? "," : "") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto options = parse_options(argc, argv);
        if (options.help) {
            print_help(std::cout);
            return 0;
        }
        if (options.list_solvers) {
            for (const auto& descriptor : eda_gpu::core::solver_descriptors()) {
                std::cout << descriptor.name << '\t'
                          << (descriptor.available ? "available" : "unavailable") << '\t'
                          << descriptor.detail << '\n';
            }
            return 0;
        }
        if (options.list_cases) {
            for (const auto& item : kCases) {
                std::cout << item.name << '\t' << item.rows << '\t' << item.nonzeros
                          << '\t' << item.source << '\t'
                          << (item.has_right_hand_side ? "provided RHS" : "generated RHS")
                          << '\n';
            }
            return 0;
        }
        resolve_case(options);

        eda_gpu::core::MatrixInput input;
        std::vector<double> right_hand_side;
        const auto load_ms = timed_ms([&] {
            input = options.matrix ? eda_gpu::core::load_matrix_market(*options.matrix)
                                   : eda_gpu::core::make_poisson_2d(*options.grid);
            if (options.rhs) {
                right_hand_side = eda_gpu::core::load_matrix_market_vector(
                    *options.rhs, input.matrix.rows);
            }
        });
        auto solver = eda_gpu::core::create_solver(options.solver);
        std::vector<double> reference_solution;
        if (!options.rhs) {
            reference_solution.assign(static_cast<std::size_t>(input.matrix.columns), 1.0);
            right_hand_side = eda_gpu::core::multiply(input.matrix, reference_solution);
        }

        const auto initial_start = Clock::now();
        const auto analyze_ms = timed_ms([&] { solver->analyze(input.matrix); });
        const auto initial_factor_ms = timed_ms([&] { solver->factorize(input.matrix); });
        std::vector<double> solution;
        const auto initial_solve_ms = timed_ms([&] { solution = solver->solve(right_hand_side); });
        const auto initial_total_ms = std::chrono::duration<double, std::milli>(
                                          Clock::now() - initial_start)
                                          .count();

        for (std::int32_t iteration = 0; iteration < options.warmup; ++iteration) {
            solver->refactorize(input.matrix);
            solution = solver->solve(right_hand_side);
        }

        std::vector<double> refactor_samples;
        std::vector<double> solve_samples;
        std::vector<double> total_samples;
        refactor_samples.reserve(static_cast<std::size_t>(options.iterations));
        solve_samples.reserve(static_cast<std::size_t>(options.iterations));
        total_samples.reserve(static_cast<std::size_t>(options.iterations));
        for (std::int32_t iteration = 0; iteration < options.iterations; ++iteration) {
            const auto total_start = Clock::now();
            refactor_samples.push_back(timed_ms([&] { solver->refactorize(input.matrix); }));
            solve_samples.push_back(timed_ms([&] { solution = solver->solve(right_hand_side); }));
            total_samples.push_back(std::chrono::duration<double, std::milli>(
                                        Clock::now() - total_start)
                                        .count());
        }

        // Keep the untimed CPU reference solve after the target measurements so it cannot
        // warm the target solver's input and bias cold-start timings.
        if (options.rhs) {
            auto reference_solver = eda_gpu::core::create_solver("klu");
            reference_solver->analyze(input.matrix);
            reference_solver->factorize(input.matrix);
            reference_solution = reference_solver->solve(right_hand_side);
        }

        const auto residual = relative_residual(input.matrix, solution, right_hand_side);
        const auto error = relative_solution_error(solution, reference_solution);
        if (!std::isfinite(error) || error > options.tolerance) {
            throw std::runtime_error("relative solution error " + std::to_string(error) +
                                     " exceeds tolerance " +
                                     std::to_string(options.tolerance));
        }

        std::ostringstream json;
        json << std::setprecision(17)
             << "{\n"
             << "  \"schema_version\": 2,\n"
             << "  \"status\": \"ok\",\n"
             << "  \"solver\": \"" << json_escape(solver->name()) << "\",\n"
             << "  \"case\": \""
             << json_escape(options.case_name ? *options.case_name : "custom") << "\",\n"
             << "  \"input\": \"" << json_escape(input.description) << "\",\n"
             << "  \"rhs\": \""
             << json_escape(options.rhs ? options.rhs->string() : "generated:A*ones")
             << "\",\n"
             << "  \"matrix\": {\"rows\": " << input.matrix.rows
             << ", \"columns\": " << input.matrix.columns
             << ", \"nnz\": " << input.matrix.nonzeros() << "},\n"
             << "  \"warmup\": " << options.warmup << ",\n"
             << "  \"iterations\": " << options.iterations << ",\n"
             << "  \"timing_scope\": \"host API; input generation/loading and validation excluded from solver phases\",\n"
             << "  \"timing_ms\": {\n"
             << "    \"load\": " << load_ms << ",\n"
             << "    \"analyze\": " << analyze_ms << ",\n"
             << "    \"initial_factor\": " << initial_factor_ms << ",\n"
             << "    \"initial_solve\": " << initial_solve_ms << ",\n";
        json << "    \"initial_total\": " << initial_total_ms << ",\n";
        write_timing(json, "refactor", refactor_samples, true);
        write_timing(json, "solve", solve_samples, true);
        write_timing(json, "refactor_solve_total", total_samples, false);
        json << "  },\n"
             << "  \"validation\": {\n"
             << "    \"relative_residual_l2\": " << residual << ",\n"
             << "    \"relative_solution_error_l2\": " << error << ",\n"
             << "    \"reference_solver\": \"klu\",\n"
             << "    \"tolerance\": " << options.tolerance << "\n"
             << "  }\n"
             << "}\n";

        if (options.output) {
            if (options.output->has_parent_path()) {
                std::filesystem::create_directories(options.output->parent_path());
            }
            std::ofstream output(*options.output);
            if (!output) {
                throw std::runtime_error("cannot write output file: " + options.output->string());
            }
            output << json.str();
        } else {
            std::cout << json.str();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "solver_benchmark: " << error.what() << '\n';
        return 1;
    }
}
