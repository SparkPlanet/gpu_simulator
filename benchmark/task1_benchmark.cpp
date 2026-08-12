#include "eda_gpu/matrix.hpp"
#include "eda_gpu/report.hpp"
#include "eda_gpu/task1.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef EDA_GPU_ROOT
#define EDA_GPU_ROOT "."
#endif

namespace {

struct Options {
    std::string backend{"cpu-klu"};
    std::optional<std::string> case_name;
    std::optional<std::filesystem::path> matrix_path;
    std::optional<std::filesystem::path> rhs_path;
    std::optional<eda_gpu::SparseIndex> grid_size;
    std::optional<std::filesystem::path> output_path;
    double tolerance{1e-9};
    bool list_backends{};
    bool list_cases{};
    bool help{};
};

struct CaseDescriptor {
    std::string_view name;
    eda_gpu::SparseIndex dimension;
    eda_gpu::SparseIndex nonzeros;
    std::string_view source;
};

constexpr CaseDescriptor kCases[]{
    {"asic_320k", 321821, 2635364, "SuiteSparse/Sandia"},
    {"asic_680k", 682862, 3871773, "SuiteSparse/Sandia"},
    {"circuit5m_dc", 3523317, 19194193, "SuiteSparse/Freescale"},
};

[[nodiscard]] std::string escape_json(std::string_view value) {
    std::string result;
    for (const auto character : value) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += character; break;
        }
    }
    return result;
}

[[nodiscard]] eda_gpu::SparseIndex parse_index(
    std::string_view text,
    std::string_view option) {
    std::size_t consumed{};
    const auto value = std::stoll(std::string(text), &consumed);
    if (consumed != text.size() || value <= 0 ||
        value > std::numeric_limits<eda_gpu::SparseIndex>::max()) {
        throw std::runtime_error("invalid positive integer for " + std::string(option));
    }
    return static_cast<eda_gpu::SparseIndex>(value);
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    auto value_after = [&](int& index, std::string_view option) -> std::string_view {
        if (++index >= argc) throw std::runtime_error(std::string(option) + " requires a value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--backend") {
            options.backend = value_after(index, argument);
        } else if (argument == "--case") {
            options.case_name = value_after(index, argument);
        } else if (argument == "--matrix") {
            options.matrix_path = value_after(index, argument);
        } else if (argument == "--rhs") {
            options.rhs_path = value_after(index, argument);
        } else if (argument == "--grid") {
            options.grid_size = parse_index(value_after(index, argument), argument);
        } else if (argument == "--tolerance") {
            std::size_t consumed{};
            const auto text = value_after(index, argument);
            options.tolerance = std::stod(std::string(text), &consumed);
            if (consumed != text.size() || !std::isfinite(options.tolerance) ||
                options.tolerance <= 0.0) {
                throw std::runtime_error("invalid tolerance");
            }
        } else if (argument == "--output") {
            options.output_path = value_after(index, argument);
        } else if (argument == "--list-backends") {
            options.list_backends = true;
        } else if (argument == "--list-cases") {
            options.list_cases = true;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }

    if (!options.help && !options.list_backends && !options.list_cases) {
        const auto inputs = static_cast<int>(options.case_name.has_value()) +
                            static_cast<int>(options.matrix_path.has_value()) +
                            static_cast<int>(options.grid_size.has_value());
        if (inputs != 1) {
            throw std::runtime_error("select exactly one of --case, --matrix or --grid");
        }
        if (options.rhs_path && !options.matrix_path) {
            throw std::runtime_error("--rhs is only valid with --matrix");
        }
    }
    return options;
}

void print_help() {
    std::cout
        << "Usage: task1_benchmark [options]\n"
        << "  --backend NAME     one-shot Task 1 backend (default: cpu-klu)\n"
        << "  --case NAME        bundled sparse circuit matrix\n"
        << "  --matrix FILE      Matrix Market coefficient matrix\n"
        << "  --rhs FILE         Matrix Market right-hand side\n"
        << "  --grid N           generated N-by-N Poisson grid\n"
        << "  --tolerance VALUE  relative residual limit (default: 1e-9)\n"
        << "  --output FILE      write JSON report to a file\n"
        << "  --list-backends    list Task 1 backends\n"
        << "  --list-cases       list bundled matrices\n";
}

[[nodiscard]] std::filesystem::path case_path(std::string_view name) {
    for (const auto& descriptor : kCases) {
        if (descriptor.name == name) {
            return std::filesystem::path(EDA_GPU_ROOT) / "datasets/matrix" /
                   (std::string(name) + ".mtx");
        }
    }
    throw std::runtime_error("unknown bundled case: " + std::string(name));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.help) {
            print_help();
            return 0;
        }
        if (options.list_backends) {
            for (const auto& backend : eda_gpu::task1_backends()) {
                std::cout << backend.name << '\t'
                          << (backend.available ? "available" : "unavailable") << '\t'
                          << backend.description << '\n';
            }
            return 0;
        }
        if (options.list_cases) {
            for (const auto& descriptor : kCases) {
                std::cout << descriptor.name << '\t' << descriptor.dimension << '\t'
                          << descriptor.nonzeros << '\t' << descriptor.source << '\n';
            }
            return 0;
        }

        const auto matrix_path = options.case_name
                                     ? std::optional<std::filesystem::path>(
                                           case_path(*options.case_name))
                                     : options.matrix_path;
        auto input = matrix_path ? eda_gpu::load_matrix_market(*matrix_path)
                                 : eda_gpu::make_poisson_2d(*options.grid_size);
        std::vector<double> right_hand_side;
        std::string rhs_description;
        if (options.rhs_path) {
            right_hand_side = eda_gpu::load_matrix_market_vector(
                *options.rhs_path, input.matrix.dimension);
            rhs_description = options.rhs_path->string();
        } else {
            const std::vector<double> reference(
                static_cast<std::size_t>(input.matrix.dimension), 1.0);
            right_hand_side = eda_gpu::multiply(input.matrix, reference);
            rhs_description = "generated:A*ones";
        }

        const eda_gpu::Task1Pipeline pipeline(options.backend);
        auto result = pipeline.run({input.matrix, right_hand_side});
        const auto residual = eda_gpu::relative_residual_l2(
            input.matrix, result.solution, right_hand_side);
        if (!std::isfinite(residual) || residual > options.tolerance) {
            throw std::runtime_error(
                "relative residual " + std::to_string(residual) +
                " exceeds tolerance " + std::to_string(options.tolerance));
        }

        std::ostringstream json;
        json << std::setprecision(17)
             << "{\n"
             << "  \"benchmark_schema_version\": 1,\n"
             << "  \"status\": \"ok\",\n"
             << "  \"input\": \"" << escape_json(input.description) << "\",\n"
             << "  \"rhs\": \"" << escape_json(rhs_description) << "\",\n"
             << "  \"validation\": {\"relative_residual_l2\": " << residual
             << ", \"tolerance\": " << options.tolerance << "},\n"
             << "  \"task1_report\": " << eda_gpu::task1_report_json(result.report)
             << "}\n";

        if (options.output_path) {
            if (options.output_path->has_parent_path()) {
                std::filesystem::create_directories(options.output_path->parent_path());
            }
            std::ofstream output(*options.output_path);
            if (!output) {
                throw std::runtime_error("cannot write report: " + options.output_path->string());
            }
            output << json.str();
        } else {
            std::cout << json.str();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "task1_benchmark: " << error.what() << '\n';
        return 1;
    }
}
