#pragma once

#include "profiler.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace eda_gpu {

struct ExecutorStatistics {
    std::uint64_t live_bytes{};
    std::uint64_t peak_bytes{};
    std::uint64_t allocation_calls{};
    std::uint64_t free_calls{};
    std::uint64_t copy_calls{};
    std::uint64_t copied_bytes{};
    std::uint64_t synchronization_calls{};
};

class Executor;

// Move-only ownership of memory allocated by an Executor.
class Allocation {
public:
    Allocation() = default;
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    Allocation(Allocation&& other) noexcept;
    Allocation& operator=(Allocation&& other) noexcept;
    ~Allocation();

    [[nodiscard]] void* data() noexcept { return pointer_; }
    [[nodiscard]] const void* data() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return bytes_; }
    [[nodiscard]] explicit operator bool() const noexcept { return pointer_ != nullptr; }
    void reset() noexcept;

private:
    friend class Executor;
    Allocation(Executor* executor, void* pointer, std::size_t bytes) noexcept;

    Executor* executor_{};
    void* pointer_{};
    std::size_t bytes_{};
};

// Ginkgo-style execution boundary. Allocation, transfer and synchronization
// are observable here, while algorithms remain backend-owned operations.
class Executor {
public:
    explicit Executor(Profiler& profiler) noexcept;
    virtual ~Executor() = default;

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual MemorySpace memory_space() const noexcept = 0;

    [[nodiscard]] Allocation allocate(std::size_t bytes);
    void copy_from(
        const Executor& source,
        void* destination,
        const void* source_pointer,
        std::size_t bytes);
    void synchronize();

    [[nodiscard]] const ExecutorStatistics& statistics() const noexcept;

protected:
    virtual void* raw_allocate(std::size_t bytes) = 0;
    virtual void raw_free(void* pointer) noexcept = 0;
    virtual bool raw_copy_from(
        const Executor& source,
        void* destination,
        const void* source_pointer,
        std::size_t bytes) = 0;
    virtual bool raw_copy_to(
        const Executor& destination,
        void* destination_pointer,
        const void* source,
        std::size_t bytes) const = 0;
    virtual void raw_synchronize() = 0;

private:
    friend class Allocation;
    void release(void* pointer, std::size_t bytes) noexcept;

    Profiler& profiler_;
    ExecutorStatistics statistics_;
};

[[nodiscard]] std::unique_ptr<Executor> make_host_executor(Profiler& profiler);

}  // namespace eda_gpu
