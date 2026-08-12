#include "eda_gpu/executor.hpp"

#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace eda_gpu {

Allocation::Allocation(Executor* executor, void* pointer, std::size_t bytes) noexcept
    : executor_(executor), pointer_(pointer), bytes_(bytes) {}

Allocation::Allocation(Allocation&& other) noexcept
    : executor_(other.executor_), pointer_(other.pointer_), bytes_(other.bytes_) {
    other.executor_ = nullptr;
    other.pointer_ = nullptr;
    other.bytes_ = 0;
}

Allocation& Allocation::operator=(Allocation&& other) noexcept {
    if (this != &other) {
        reset();
        executor_ = other.executor_;
        pointer_ = other.pointer_;
        bytes_ = other.bytes_;
        other.executor_ = nullptr;
        other.pointer_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

Allocation::~Allocation() {
    reset();
}

void Allocation::reset() noexcept {
    if (executor_ != nullptr && pointer_ != nullptr) executor_->release(pointer_, bytes_);
    executor_ = nullptr;
    pointer_ = nullptr;
    bytes_ = 0;
}

Executor::Executor(Profiler& profiler) noexcept : profiler_(profiler) {}

Allocation Executor::allocate(std::size_t bytes) {
    if (bytes == 0) return {};
    auto* pointer = raw_allocate(bytes);
    if (pointer == nullptr) throw std::bad_alloc{};
    ++statistics_.allocation_calls;
    statistics_.live_bytes += bytes;
    if (statistics_.live_bytes > statistics_.peak_bytes) {
        statistics_.peak_bytes = statistics_.live_bytes;
    }
    profiler_.record_allocation(memory_space(), bytes);
    return Allocation(this, pointer, bytes);
}

void Executor::copy_from(
    const Executor& source,
    void* destination,
    const void* source_pointer,
    std::size_t bytes) {
    if (bytes == 0) return;
    if (destination == nullptr || source_pointer == nullptr) {
        throw std::runtime_error("executor copy received a null pointer");
    }
    if (!raw_copy_from(source, destination, source_pointer, bytes) &&
        !source.raw_copy_to(*this, destination, source_pointer, bytes)) {
        throw std::runtime_error(
            "unsupported executor copy from " + std::string(source.name()) +
            " to " + std::string(name()));
    }
    ++statistics_.copy_calls;
    statistics_.copied_bytes += bytes;
    profiler_.record_copy(source.memory_space(), memory_space(), bytes);
}

void Executor::synchronize() {
    raw_synchronize();
    ++statistics_.synchronization_calls;
    profiler_.record_synchronization();
}

const ExecutorStatistics& Executor::statistics() const noexcept {
    return statistics_;
}

void Executor::release(void* pointer, std::size_t bytes) noexcept {
    raw_free(pointer);
    ++statistics_.free_calls;
    statistics_.live_bytes = bytes > statistics_.live_bytes ? 0 : statistics_.live_bytes - bytes;
    profiler_.record_free(memory_space(), bytes);
}

namespace {

class HostExecutor final : public Executor {
public:
    using Executor::Executor;

    [[nodiscard]] std::string_view name() const noexcept override { return "host"; }
    [[nodiscard]] MemorySpace memory_space() const noexcept override {
        return MemorySpace::host;
    }

protected:
    [[nodiscard]] void* raw_allocate(std::size_t bytes) override {
        return ::operator new(bytes);
    }

    void raw_free(void* pointer) noexcept override {
        ::operator delete(pointer);
    }

    bool raw_copy_from(
        const Executor& source,
        void* destination,
        const void* source_pointer,
        std::size_t bytes) override {
        if (source.memory_space() != MemorySpace::host) return false;
        std::memmove(destination, source_pointer, bytes);
        return true;
    }

    bool raw_copy_to(
        const Executor& destination,
        void* destination_pointer,
        const void* source,
        std::size_t bytes) const override {
        if (destination.memory_space() != MemorySpace::host) return false;
        std::memmove(destination_pointer, source, bytes);
        return true;
    }

    void raw_synchronize() override {}
};

}  // namespace

std::unique_ptr<Executor> make_host_executor(Profiler& profiler) {
    return std::make_unique<HostExecutor>(profiler);
}

}  // namespace eda_gpu
