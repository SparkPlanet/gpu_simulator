#include "cuda_executor.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace eda_gpu {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class CudaExecutor final : public Executor {
public:
    CudaExecutor(Profiler& profiler, int device_id)
        : Executor(profiler), device_id_(device_id) {
        check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "cuda"; }
    [[nodiscard]] MemorySpace memory_space() const noexcept override {
        return MemorySpace::device;
    }

protected:
    [[nodiscard]] void* raw_allocate(std::size_t bytes) override {
        activate();
        void* pointer{};
        check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
        return pointer;
    }

    void raw_free(void* pointer) noexcept override {
        if (cudaSetDevice(device_id_) == cudaSuccess) static_cast<void>(cudaFree(pointer));
    }

    bool raw_copy_from(
        const Executor& source,
        void* destination,
        const void* source_pointer,
        std::size_t bytes) override {
        activate();
        const auto kind = source.memory_space() == MemorySpace::host
                              ? cudaMemcpyHostToDevice
                              : cudaMemcpyDeviceToDevice;
        check_cuda(cudaMemcpy(destination, source_pointer, bytes, kind), "cudaMemcpy to device");
        return true;
    }

    bool raw_copy_to(
        const Executor& destination,
        void* destination_pointer,
        const void* source,
        std::size_t bytes) const override {
        activate();
        const auto kind = destination.memory_space() == MemorySpace::host
                              ? cudaMemcpyDeviceToHost
                              : cudaMemcpyDeviceToDevice;
        check_cuda(cudaMemcpy(destination_pointer, source, bytes, kind),
                   "cudaMemcpy from device");
        return true;
    }

    void raw_synchronize() override {
        activate();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

private:
    void activate() const {
        check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    }

    int device_id_{};
};

}  // namespace

bool cuda_runtime_available(std::string& detail) noexcept {
    int count{};
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        detail = cudaGetErrorString(status);
        static_cast<void>(cudaGetLastError());
        return false;
    }
    if (count <= 0) {
        detail = "no CUDA device found";
        return false;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        detail = "cannot query CUDA device 0";
        static_cast<void>(cudaGetLastError());
        return false;
    }
    detail = std::string(properties.name) + " (sm_" +
             std::to_string(properties.major) + std::to_string(properties.minor) + ")";
    return true;
}

std::unique_ptr<Executor> make_cuda_executor(Profiler& profiler, int device_id) {
    std::string detail;
    if (!cuda_runtime_available(detail)) {
        throw std::runtime_error("CUDA runtime is unavailable: " + detail);
    }
    return std::make_unique<CudaExecutor>(profiler, device_id);
}

}  // namespace eda_gpu
