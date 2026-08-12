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
using eda_gpu::core::SolverMetrics;

struct Options {
    std::string solver{"klu"};
    std::optional<std::string> case_name;
    std::optional<std::filesystem::path> matrix;
    std::optional<std::filesystem::path> rhs;
    std::optional<std::int32_t> grid;
    std::int32_t iterations{1};
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

struct BackendSamples {
    std::vector<double> cpu_bootstrap_factor;
    std::vector<double> gpu_setup;
    std::vector<double> numeric_factor_cpu_prepare;
    std::vector<double> numeric_factor_h2d;
    std::vector<double> numeric_factor_kernel;
    std::vector<double> numeric_factor_status_d2h;
    std::vector<double> solve_cpu_prepare;
    std::vector<double> solve_h2d;
    std::vector<double> solve_kernel;
    std::vector<double> solve_d2h;
    std::vector<double> solve_cpu_finalize;

    void reserve(std::size_t count) {
        cpu_bootstrap_factor.reserve(count);
        gpu_setup.reserve(count);
        numeric_factor_cpu_prepare.reserve(count);
        numeric_factor_h2d.reserve(count);
        numeric_factor_kernel.reserve(count);
        numeric_factor_status_d2h.reserve(count);
        solve_cpu_prepare.reserve(count);
        solve_h2d.reserve(count);
        solve_kernel.reserve(count);
        solve_d2h.reserve(count);
        solve_cpu_finalize.reserve(count);
    }

    void push(const SolverMetrics& metrics) {
        cpu_bootstrap_factor.push_back(metrics.last_cpu_bootstrap_factor_ms);
        gpu_setup.push_back(metrics.last_gpu_setup_ms);
        numeric_factor_cpu_prepare.push_back(
            metrics.last_numeric_factor_cpu_prepare_ms);
        numeric_factor_h2d.push_back(metrics.last_numeric_factor_h2d_ms);
        numeric_factor_kernel.push_back(metrics.last_numeric_factor_kernel_ms);
        numeric_factor_status_d2h.push_back(
            metrics.last_numeric_factor_status_d2h_ms);
        solve_cpu_prepare.push_back(metrics.last_solve_cpu_prepare_ms);
        solve_h2d.push_back(metrics.last_solve_h2d_ms);
        solve_kernel.push_back(metrics.last_solve_kernel_ms);
        solve_d2h.push_back(metrics.last_solve_d2h_ms);
        solve_cpu_finalize.push_back(metrics.last_solve_cpu_finalize_ms);
    }
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
        if (options.iterations <= 0 ||
            !std::isfinite(options.tolerance) || options.tolerance <= 0.0) {
            throw std::runtime_error(
                "iterations and tolerance must be greater than zero");
        }
    }
    return options;
}

void print_help(std::ostream& output) {
    output <<
        "Usage: solver_benchmark [options]\n"
        "  --solver NAME       klu or a registered accelerator backend (default: klu)\n"
        "  --case NAME         bundled circuit matrix and RHS\n"
        "  --matrix FILE       Matrix Market coordinate matrix\n"
        "  --rhs FILE          Matrix Market right-hand side (array or coordinate)\n"
        "  --grid N            generate an N-by-N 2D Poisson grid\n"
        "  --iterations N      independent Task 1 cold solves (default: 1)\n"
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

[[nodiscard]] double effective_gflops(
    std::uint64_t estimated_flops,
    const std::vector<double>& kernel_ms_samples) {
    if (estimated_flops == 0U || kernel_ms_samples.empty()) return 0.0;
    const auto mean_ms = summarize(kernel_ms_samples).mean;
    if (!(mean_ms > 0.0)) return 0.0;
    return static_cast<double>(estimated_flops) / (mean_ms * 1.0e6);
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
    bool trailing_comma,
    std::size_t indentation = 4U) {
    const auto summary = summarize(samples);
    const std::string outer(indentation, ' ');
    const std::string inner(indentation + 2U, ' ');
    output << outer << "\"" << name << "\": {\n"
           << inner << "\"mean\": " << summary.mean << ",\n"
           << inner << "\"median\": " << summary.median << ",\n"
           << inner << "\"min\": " << summary.minimum << ",\n"
           << inner << "\"max\": " << summary.maximum << ",\n"
           << inner << "\"samples\": ";
    write_samples(output, samples);
    output << '\n' << outer << '}' << (trailing_comma ? "," : "") << '\n';
}

void write_level_width_profile(
    std::ostream& output,
    std::string_view name,
    const eda_gpu::core::LevelWidthProfile& profile,
    bool trailing_comma) {
    output << "      \"" << name << "\": {\n"
           << "        \"levels\": " << profile.levels << ",\n"
           << "        \"width_1\": " << profile.width_1 << ",\n"
           << "        \"width_2\": " << profile.width_2 << ",\n"
           << "        \"width_3_to_8\": " << profile.width_3_to_8 << ",\n"
           << "        \"width_9_to_32\": " << profile.width_9_to_32 << ",\n"
           << "        \"width_33_plus\": " << profile.width_33_plus << ",\n"
           << "        \"widest\": " << profile.widest << "\n"
           << "      }" << (trailing_comma ? "," : "") << "\n";
}

void write_refactor_dag_profile(
    std::ostream& output,
    const eda_gpu::core::NumericFactorDagProfile& profile) {
    output << "    \"numeric_factor_dag\": {\n"
           << "      \"available\": " << (profile.available ? "true" : "false");
    if (!profile.available) {
        output << "\n    },\n";
        return;
    }
    output << ",\n";
    write_level_width_profile(
        output, "u_dependency_levels", profile.u_dependency_levels, true);
    write_level_width_profile(
        output, "glu3_relaxed_levels", profile.glu3_relaxed_levels, true);
    output << "      \"right_looking\": {\n"
           << "        \"subcolumn_tasks\": " << profile.subcolumn_tasks << ",\n"
           << "        \"scalar_updates\": " << profile.scalar_updates << ",\n"
           << "        \"single_level_subcolumn_tasks\": "
           << profile.single_level_subcolumn_tasks << ",\n"
           << "        \"single_level_scalar_updates\": "
           << profile.single_level_scalar_updates << ",\n"
           << "        \"single_level_scalar_update_fraction\": "
           << profile.single_level_scalar_update_fraction << ",\n"
           << "        \"single_level_fanout\": {\"p50\": "
           << profile.single_level_fanout_p50 << ", \"p90\": "
           << profile.single_level_fanout_p90 << ", \"p99\": "
           << profile.single_level_fanout_p99 << ", \"max\": "
           << profile.single_level_fanout_max << "},\n"
           << "        \"same_level_collision_groups\": "
           << profile.same_level_collision_groups << ",\n"
           << "        \"same_level_conflicting_tasks\": "
           << profile.same_level_conflicting_tasks << ",\n"
           << "        \"same_level_extra_writers\": "
           << profile.same_level_extra_writers << ",\n"
           << "        \"same_level_max_writers\": "
           << profile.same_level_max_writers << ",\n"
           << "        \"full_index_bytes_u32\": "
           << profile.full_index_bytes_u32 << ",\n"
           << "        \"full_index_bytes_mixed_u16_u32\": "
           << profile.full_index_bytes_mixed_u16_u32 << ",\n"
           << "        \"single_level_index_bytes_mixed_u16_u32\": "
           << profile.single_level_index_bytes_mixed_u16_u32 << ",\n"
           << "        \"dense_workspace_bytes_per_block\": "
           << profile.dense_workspace_bytes_per_block << "\n"
           << "      }\n"
           << "    },\n";
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
        std::vector<double> reference_solution;
        if (!options.rhs) {
            reference_solution.assign(static_cast<std::size_t>(input.matrix.columns), 1.0);
            right_hand_side = eda_gpu::core::multiply(input.matrix, reference_solution);
        }

        std::unique_ptr<eda_gpu::core::LinearSolver> solver;
        std::vector<double> solution;
        SolverMetrics initial_backend_metrics;
        std::vector<double> create_samples;
        std::vector<double> analyze_samples;
        std::vector<double> factor_samples;
        std::vector<double> solve_samples;
        std::vector<double> total_samples;
        std::vector<double> process_cold_total_samples;
        BackendSamples backend_samples;

        const auto sample_count = static_cast<std::size_t>(options.iterations);
        create_samples.reserve(sample_count);
        analyze_samples.reserve(sample_count);
        factor_samples.reserve(sample_count);
        solve_samples.reserve(sample_count);
        total_samples.reserve(sample_count);
        process_cold_total_samples.reserve(sample_count);
        backend_samples.reserve(sample_count);

        for (std::int32_t iteration = 0; iteration < options.iterations; ++iteration) {
            // A fresh solver per sample enforces Task 1 semantics: no symbolic
            // pattern, device allocation or numerical factor is reused.
            solver.reset();
            const auto process_start = Clock::now();
            double create_ms{};
            auto current_solver = std::unique_ptr<eda_gpu::core::LinearSolver>{};
            create_ms = timed_ms([&] {
                current_solver = eda_gpu::core::create_solver(options.solver);
            });
            const auto task1_start = Clock::now();
            const auto analyze_ms = timed_ms(
                [&] { current_solver->analyze(input.matrix); });
            const auto factor_ms = timed_ms(
                [&] { current_solver->factorize(input.matrix); });
            const auto solve_ms = timed_ms(
                [&] { solution = current_solver->solve(right_hand_side); });
            const auto metrics = current_solver->metrics();
            const auto task1_total_ms = std::chrono::duration<double, std::milli>(
                                            Clock::now() - task1_start)
                                            .count();
            const auto process_cold_total_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - process_start)
                    .count();

            create_samples.push_back(create_ms);
            analyze_samples.push_back(analyze_ms);
            factor_samples.push_back(factor_ms);
            solve_samples.push_back(solve_ms);
            total_samples.push_back(task1_total_ms);
            process_cold_total_samples.push_back(process_cold_total_ms);
            backend_samples.push(metrics);
            solver = std::move(current_solver);
            initial_backend_metrics = metrics;
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
             << "  \"schema_version\": 9,\n"
             << "  \"status\": \"ok\",\n"
             << "  \"benchmark_mode\": \"task1\",\n"
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
             << "  \"iterations\": " << options.iterations << ",\n"
             << "  \"timing_scope\": \"task1_total is host wall-clock from analyze through returned solution and includes per-matrix CPU preparation, allocation, transfers, GPU work and synchronization; process_cold_total additionally includes solver/CUDA runtime creation; matrix file loading and post-solve validation are excluded\",\n"
             << "  \"timing_ms\": {\n"
             << "    \"load_excluded_from_solver_total\": " << load_ms << ",\n";
        write_timing(json, "solver_create", create_samples, true);
        write_timing(json, "symbolic_analysis", analyze_samples, true);
        write_timing(json, "numeric_factorization_and_setup", factor_samples, true);
        write_timing(json, "triangular_solve", solve_samples, true);
        write_timing(json, "task1_total", total_samples, true);
        write_timing(json, "process_cold_total_including_solver_create",
                     process_cold_total_samples, false);
        json << "  },\n"
             << "  \"backend\": {\n"
             << "    \"execution_model\": \""
             << json_escape(initial_backend_metrics.execution_model) << "\",\n"
             << "    \"schedule_mode\": \""
             << json_escape(initial_backend_metrics.schedule_mode) << "\",\n"
             << "    \"numeric_factor_mode\": \""
             << json_escape(initial_backend_metrics.numeric_factor_mode) << "\",\n"
             << "    \"gpu_compute\": "
             << (initial_backend_metrics.gpu_compute ? "true" : "false") << ",\n"
             << "    \"device_memory_bytes\": "
             << initial_backend_metrics.device_memory_bytes << ",\n"
             << "    \"device_memory_scope\": \"operator-owned live allocations after setup; exact for the project CUDA backend\",\n"
             << "    \"estimated_numeric_factor_flops\": "
             << initial_backend_metrics.estimated_numeric_factor_flops << ",\n"
             << "    \"estimated_triangular_solve_flops\": "
             << initial_backend_metrics.estimated_triangular_solve_flops << ",\n"
             << "    \"numeric_factor_effective_gflops\": "
             << effective_gflops(
                    initial_backend_metrics.estimated_numeric_factor_flops,
                    backend_samples.numeric_factor_kernel)
             << ",\n"
             << "    \"triangular_solve_effective_gflops\": "
             << effective_gflops(
                    initial_backend_metrics.estimated_triangular_solve_flops,
                    backend_samples.solve_kernel)
             << ",\n"
             << "    \"schedule_operations\": "
             << initial_backend_metrics.schedule_operations << ",\n"
             << "    \"backend_calls_per_solve\": "
             << initial_backend_metrics.backend_calls_per_solve << ",\n"
             << "    \"persistent_grid_blocks\": "
             << initial_backend_metrics.persistent_grid_blocks << ",\n"
             << "    \"fused_operations_per_launch\": "
             << initial_backend_metrics.fused_operations_per_launch << ",\n"
             << "    \"scheduled_columns\": "
             << initial_backend_metrics.scheduled_columns << ",\n"
             << "    \"widest_operation_columns\": "
             << initial_backend_metrics.widest_operation_columns << ",\n"
             << "    \"narrow_operations_le_8_columns\": "
             << initial_backend_metrics.narrow_operations << ",\n"
             << "    \"numeric_factor_blocks\": "
             << initial_backend_metrics.numeric_factor_blocks << ",\n";
        write_refactor_dag_profile(json, initial_backend_metrics.numeric_factor_dag);
        json << "    \"task1_phase_ms\": {\n";
        write_timing(json, "cpu_numeric_pattern_bootstrap",
                     backend_samples.cpu_bootstrap_factor, true, 6U);
        write_timing(json, "gpu_setup", backend_samples.gpu_setup, true, 6U);
        write_timing(json, "numeric_lu_cpu_prepare",
                     backend_samples.numeric_factor_cpu_prepare, true, 6U);
        write_timing(json, "numeric_lu_h2d", backend_samples.numeric_factor_h2d,
                     true, 6U);
        write_timing(json, "numeric_lu_kernel", backend_samples.numeric_factor_kernel,
                     true, 6U);
        write_timing(json, "numeric_lu_status_d2h",
                     backend_samples.numeric_factor_status_d2h, true, 6U);
        write_timing(json, "solve_cpu_prepare", backend_samples.solve_cpu_prepare, true, 6U);
        write_timing(json, "solve_h2d", backend_samples.solve_h2d, true, 6U);
        write_timing(json, "solve_kernel", backend_samples.solve_kernel, true, 6U);
        write_timing(json, "solve_d2h", backend_samples.solve_d2h, true, 6U);
        write_timing(json, "solve_cpu_finalize", backend_samples.solve_cpu_finalize, false, 6U);
        json << "    }\n";
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
