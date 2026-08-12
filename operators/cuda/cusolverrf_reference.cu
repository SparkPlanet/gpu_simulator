#include "cusolverrf_reference.hpp"

#include <cuda_runtime.h>
#include <cusolverRf.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace eda_gpu::operators::cuda {
namespace {

[[noreturn]] void throw_cuda(cudaError_t status, const char* operation) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw_cuda(status, operation);
}

void check_cusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuSolver status " +
            std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
[[nodiscard]] std::uint64_t byte_count(const std::vector<T>& values) {
    return static_cast<std::uint64_t>(values.size()) * sizeof(T);
}

[[nodiscard]] int environment_choice(
    const char* variable,
    int default_value,
    int minimum,
    int maximum) {
    const char* value = std::getenv(variable);
    if (value == nullptr) return default_value;
    char* end{};
    const auto parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::runtime_error(
            std::string(variable) + " must be an integer from " +
            std::to_string(minimum) + " to " + std::to_string(maximum));
    }
    return static_cast<int>(parsed);
}

[[nodiscard]] bool query_device(std::string& detail) {
    int count{};
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        detail = std::string("CUDA runtime unavailable: ") + cudaGetErrorString(status);
        cudaGetLastError();
        return false;
    }
    if (count <= 0) {
        detail = "CUDA runtime is present but no device is visible";
        return false;
    }

    cudaDeviceProp properties{};
    const auto properties_status = cudaGetDeviceProperties(&properties, 0);
    if (properties_status != cudaSuccess) {
        detail = std::string("cannot query CUDA device 0: ") +
                 cudaGetErrorString(properties_status);
        cudaGetLastError();
        return false;
    }
    std::ostringstream description;
    description << properties.name << " (sm_" << properties.major << properties.minor
                << ", " << (properties.totalGlobalMem / (1024U * 1024U)) << " MiB)";
    detail = description.str();
    return true;
}

struct ReferenceState {
    std::int32_t dimension{};
    std::int32_t input_nonzeros{};
    std::int32_t bundled_nonzeros{};
    std::uint64_t allocated_bytes{};
    int factor_algorithm{};
    int solve_algorithm{2};

    std::int32_t* input_column_offsets{};
    std::int32_t* input_row_indices{};
    double* input_values{};
    std::int32_t* row_permutation{};
    std::int32_t* column_permutation{};
    double* solution{};
    double* solve_workspace{};

    // Owned by cuSolverRf; queried only for visible-memory accounting.
    std::int32_t* bundled_column_offsets{};
    std::int32_t* bundled_row_indices{};
    double* bundled_values{};
    cusolverRfHandle_t rf_handle{};

    // cuSolverRf uses the legacy default stream, so all reference-backend
    // transfers and events deliberately use the same stream.
    cudaStream_t stream{};
    cudaEvent_t start{};
    cudaEvent_t after_h2d{};
    cudaEvent_t after_kernel{};
    cudaEvent_t after_d2h{};

    ReferenceState() {
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        try {
            check_cuda(cudaEventCreate(&after_h2d), "cudaEventCreate(after_h2d)");
            check_cuda(cudaEventCreate(&after_kernel), "cudaEventCreate(after_kernel)");
            check_cuda(cudaEventCreate(&after_d2h), "cudaEventCreate(after_d2h)");
        } catch (...) {
            release_events();
            throw;
        }
    }

    ~ReferenceState() {
        release_rf();
        release_buffers();
        release_events();
    }

    void release_events() noexcept {
        if (start != nullptr) cudaEventDestroy(start);
        if (after_h2d != nullptr) cudaEventDestroy(after_h2d);
        if (after_kernel != nullptr) cudaEventDestroy(after_kernel);
        if (after_d2h != nullptr) cudaEventDestroy(after_d2h);
        start = nullptr;
        after_h2d = nullptr;
        after_kernel = nullptr;
        after_d2h = nullptr;
    }

    void release_rf() noexcept {
        if (rf_handle != nullptr) cusolverRfDestroy(rf_handle);
        rf_handle = nullptr;
        bundled_column_offsets = nullptr;
        bundled_row_indices = nullptr;
        bundled_values = nullptr;
        bundled_nonzeros = 0;
    }

    template <typename T>
    void release(T*& pointer) noexcept {
        if (pointer != nullptr) cudaFree(pointer);
        pointer = nullptr;
    }

    void release_buffers() noexcept {
        release(input_column_offsets);
        release(input_row_indices);
        release(input_values);
        release(row_permutation);
        release(column_permutation);
        release(solution);
        release(solve_workspace);
        dimension = 0;
        input_nonzeros = 0;
        allocated_bytes = 0;
    }

    template <typename T>
    void allocate(T*& pointer, std::size_t count, const char* operation) {
        if (count == 0U) {
            pointer = nullptr;
            return;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::runtime_error(std::string(operation) + " size overflow");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), operation);
        allocated_bytes += static_cast<std::uint64_t>(count) * sizeof(T);
    }

    template <typename T>
    void copy_to_device(T* destination, const std::vector<T>& source, const char* operation) {
        if (source.empty()) return;
        check_cuda(cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream),
                   operation);
    }

    [[nodiscard]] double elapsed(cudaEvent_t begin, cudaEvent_t end) const {
        float milliseconds{};
        check_cuda(cudaEventElapsedTime(&milliseconds, begin, end),
                   "cudaEventElapsedTime");
        return static_cast<double>(milliseconds);
    }
};

class CuSolverRfReferenceOperator final : public SparseNumericOperator {
public:
    CuSolverRfReferenceOperator() {
        std::string detail;
        if (!query_device(detail)) {
            throw std::runtime_error("cannot create cuSolverRf reference backend: " + detail);
        }
    }

    [[nodiscard]] std::string_view name() const noexcept override {
        return "cusolverrf-reference";
    }

    double initialize(
        const core::CscMatrix& matrix,
        const InitialFactorization& factors) override {
        const auto dimension = factors.dimension;
        const auto& lower = factors.lower;
        const auto& upper = factors.upper;
        if (dimension <= 0 || matrix.rows != dimension || matrix.columns != dimension ||
            lower.column_offsets.size() != static_cast<std::size_t>(dimension) + 1U ||
            upper.column_offsets.size() != static_cast<std::size_t>(dimension) + 1U ||
            lower.row_indices.size() != lower.values.size() ||
            upper.row_indices.size() != upper.values.size() ||
            matrix.column_offsets.size() != static_cast<std::size_t>(dimension) + 1U ||
            matrix.row_indices.size() != matrix.values.size() ||
            factors.row_permutation.size() != static_cast<std::size_t>(dimension) ||
            factors.column_permutation.size() != static_cast<std::size_t>(dimension) ||
            matrix.values.size() >
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
            lower.values.size() >
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
            upper.values.size() >
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("invalid cuSolverRf reference initialization data");
        }
        if (!factors.off_diagonal.values.empty() ||
            factors.off_diagonal.column_offsets.size() !=
                static_cast<std::size_t>(dimension) + 1U ||
            factors.off_diagonal.column_offsets.back() != 0 ||
            factors.block_boundaries.size() != 2U ||
            factors.block_boundaries.front() != 0 ||
            factors.block_boundaries.back() != dimension ||
            !std::all_of(factors.scale_factors.begin(), factors.scale_factors.end(),
                         [](double value) { return value == 1.0; })) {
            throw std::runtime_error(
                "cuSolverRf reference requires one unscaled global KLU factorization");
        }

        state_.release_rf();
        state_.release_buffers();
        state_.dimension = dimension;
        state_.input_nonzeros = static_cast<std::int32_t>(matrix.values.size());
        state_.factor_algorithm = environment_choice("EDA_GPU_CUSOLVERRF_ALG", 0, 0, 2);
        state_.solve_algorithm =
            environment_choice("EDA_GPU_CUSOLVERRF_SOLVE_ALG", 2, 1, 3);

        try {
            state_.allocate(state_.input_column_offsets, matrix.column_offsets.size(),
                            "cudaMalloc(Ap)");
            state_.allocate(state_.input_row_indices, matrix.row_indices.size(),
                            "cudaMalloc(Ai)");
            state_.allocate(state_.input_values, matrix.values.size(), "cudaMalloc(Ax)");
            state_.allocate(state_.row_permutation, factors.row_permutation.size(),
                            "cudaMalloc(P)");
            state_.allocate(state_.column_permutation, factors.column_permutation.size(),
                            "cudaMalloc(Q)");
            state_.allocate(state_.solution, static_cast<std::size_t>(dimension),
                            "cudaMalloc(solution)");
            state_.allocate(state_.solve_workspace, static_cast<std::size_t>(dimension),
                            "cudaMalloc(cuSolverRf solve workspace)");

            check_cuda(cudaEventRecord(state_.start, state_.stream),
                       "cudaEventRecord(reference initialize start)");
            state_.copy_to_device(state_.input_column_offsets, matrix.column_offsets,
                                  "cudaMemcpyAsync(Ap)");
            state_.copy_to_device(state_.input_row_indices, matrix.row_indices,
                                  "cudaMemcpyAsync(Ai)");
            state_.copy_to_device(state_.input_values, matrix.values,
                                  "cudaMemcpyAsync(Ax)");
            state_.copy_to_device(state_.row_permutation, factors.row_permutation,
                                  "cudaMemcpyAsync(P)");
            state_.copy_to_device(state_.column_permutation, factors.column_permutation,
                                  "cudaMemcpyAsync(Q)");

            check_cusolver(cusolverRfCreate(&state_.rf_handle), "cusolverRfCreate");
            check_cusolver(cusolverRfSetMatrixFormat(
                               state_.rf_handle, CUSOLVERRF_MATRIX_FORMAT_CSC,
                               CUSOLVERRF_UNIT_DIAGONAL_STORED_L),
                           "cusolverRfSetMatrixFormat");
            check_cusolver(cusolverRfSetAlgs(
                               state_.rf_handle,
                               static_cast<cusolverRfFactorization_t>(
                                   state_.factor_algorithm),
                               static_cast<cusolverRfTriangularSolve_t>(
                                   state_.solve_algorithm)),
                           "cusolverRfSetAlgs");
            check_cusolver(cusolverRfSetResetValuesFastMode(
                               state_.rf_handle, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON),
                           "cusolverRfSetResetValuesFastMode");
            check_cusolver(cusolverRfSetupHost(
                               dimension, state_.input_nonzeros,
                               const_cast<std::int32_t*>(matrix.column_offsets.data()),
                               const_cast<std::int32_t*>(matrix.row_indices.data()),
                               const_cast<double*>(matrix.values.data()),
                               static_cast<std::int32_t>(lower.values.size()),
                               const_cast<std::int32_t*>(lower.column_offsets.data()),
                               const_cast<std::int32_t*>(lower.row_indices.data()),
                               const_cast<double*>(lower.values.data()),
                               static_cast<std::int32_t>(upper.values.size()),
                               const_cast<std::int32_t*>(upper.column_offsets.data()),
                               const_cast<std::int32_t*>(upper.row_indices.data()),
                               const_cast<double*>(upper.values.data()),
                               const_cast<std::int32_t*>(factors.row_permutation.data()),
                               const_cast<std::int32_t*>(factors.column_permutation.data()),
                               state_.rf_handle),
                           "cusolverRfSetupHost");
            check_cusolver(cusolverRfAnalyze(state_.rf_handle), "cusolverRfAnalyze");
            check_cusolver(cusolverRfAccessBundledFactorsDevice(
                               state_.rf_handle, &state_.bundled_nonzeros,
                               &state_.bundled_column_offsets, &state_.bundled_row_indices,
                               &state_.bundled_values),
                           "cusolverRfAccessBundledFactorsDevice");

            // Opaque cuSolverRf workspace is not exposed; this is a lower-bound
            // count of our buffers plus the accessible bundled factor storage.
            state_.allocated_bytes +=
                static_cast<std::uint64_t>(dimension + 1) * sizeof(std::int32_t) +
                static_cast<std::uint64_t>(state_.bundled_nonzeros) *
                    (sizeof(std::int32_t) + sizeof(double));

            check_cuda(cudaEventRecord(state_.after_h2d, state_.stream),
                       "cudaEventRecord(after reference initialize)");
            check_cuda(cudaEventSynchronize(state_.after_h2d),
                       "cudaEventSynchronize(reference initialize)");
            return state_.elapsed(state_.start, state_.after_h2d);
        } catch (...) {
            state_.release_rf();
            state_.release_buffers();
            throw;
        }
    }

    [[nodiscard]] NumericFactorTimings factorize(
        const std::vector<double>& matrix_values) override {
        if (state_.dimension <= 0 || state_.rf_handle == nullptr ||
            matrix_values.size() != static_cast<std::size_t>(state_.input_nonzeros)) {
            throw std::runtime_error(
                "cuSolverRf reference input does not preserve the matrix pattern");
        }

        check_cuda(cudaEventRecord(state_.start, state_.stream),
                   "cudaEventRecord(refactor start)");
        state_.copy_to_device(state_.input_values, matrix_values,
                              "cudaMemcpyAsync(refactor Ax)");
        check_cuda(cudaEventRecord(state_.after_h2d, state_.stream),
                   "cudaEventRecord(after refactor Ax H2D)");
        check_cusolver(cusolverRfResetValues(
                           state_.dimension, state_.input_nonzeros,
                           state_.input_column_offsets, state_.input_row_indices,
                           state_.input_values, state_.row_permutation,
                           state_.column_permutation, state_.rf_handle),
                       "cusolverRfResetValues");
        check_cusolver(cusolverRfRefactor(state_.rf_handle), "cusolverRfRefactor");
        check_cuda(cudaEventRecord(state_.after_kernel, state_.stream),
                   "cudaEventRecord(after cuSolverRf refactor)");
        check_cuda(cudaEventSynchronize(state_.after_kernel),
                   "cudaEventSynchronize(cuSolverRf refactor)");

        return {state_.elapsed(state_.start, state_.after_h2d),
                state_.elapsed(state_.after_h2d, state_.after_kernel), 0.0};
    }

    [[nodiscard]] SolveTimings solve(
        const std::vector<double>& right_hand_side,
        std::vector<double>& solution) override {
        if (state_.dimension <= 0 || state_.rf_handle == nullptr ||
            right_hand_side.size() != static_cast<std::size_t>(state_.dimension)) {
            throw std::runtime_error(
                "cuSolverRf reference solve dimension mismatch or missing factors");
        }
        solution.resize(right_hand_side.size());

        check_cuda(cudaEventRecord(state_.start, state_.stream),
                   "cudaEventRecord(solve start)");
        check_cuda(cudaMemcpyAsync(state_.solution, right_hand_side.data(),
                                   byte_count(right_hand_side), cudaMemcpyHostToDevice,
                                   state_.stream),
                   "cudaMemcpyAsync(RHS H2D)");
        check_cuda(cudaEventRecord(state_.after_h2d, state_.stream),
                   "cudaEventRecord(after RHS H2D)");
        check_cusolver(cusolverRfSolve(
                           state_.rf_handle, state_.row_permutation,
                           state_.column_permutation, 1, state_.solve_workspace,
                           state_.dimension, state_.solution, state_.dimension),
                       "cusolverRfSolve");
        check_cuda(cudaEventRecord(state_.after_kernel, state_.stream),
                   "cudaEventRecord(after solve)");
        check_cuda(cudaMemcpyAsync(solution.data(), state_.solution, byte_count(solution),
                                   cudaMemcpyDeviceToHost, state_.stream),
                   "cudaMemcpyAsync(solution D2H)");
        check_cuda(cudaEventRecord(state_.after_d2h, state_.stream),
                   "cudaEventRecord(after solution D2H)");
        check_cuda(cudaEventSynchronize(state_.after_d2h),
                   "cudaEventSynchronize(solve)");

        return {state_.elapsed(state_.start, state_.after_h2d),
                state_.elapsed(state_.after_h2d, state_.after_kernel),
                state_.elapsed(state_.after_kernel, state_.after_d2h), 1};
    }

    [[nodiscard]] OperatorProfile profile() const override {
        OperatorProfile result;
        if (state_.rf_handle == nullptr) return result;
        result.algorithm_mode =
            "cusolverrf-factor-alg" + std::to_string(state_.factor_algorithm) +
            "+solve-alg" + std::to_string(state_.solve_algorithm);
        result.device_memory_bytes = state_.allocated_bytes;
        result.numeric_factor_blocks = 1;
        return result;
    }

private:
    ReferenceState state_;
};

}  // namespace

bool cusolverrf_reference_available(std::string& detail) {
    return query_device(detail);
}

std::unique_ptr<SparseNumericOperator> make_cusolverrf_reference_operator() {
    return std::make_unique<CuSolverRfReferenceOperator>();
}

}  // namespace eda_gpu::operators::cuda
