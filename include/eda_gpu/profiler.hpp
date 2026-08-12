#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace eda_gpu {

enum class EventKind {
    stage,
    event,
    kernel,
    transfer,
    allocation,
    synchronization,
};

enum class MemorySpace {
    host,
    device,
};

struct EventMetrics {
    std::uint64_t calls{1};
    double wall_time_ms{};
    double device_time_ms{};
    double estimated_flops{};
    std::uint64_t allocation_calls{};
    std::uint64_t free_calls{};
    std::uint64_t host_allocated_bytes{};
    std::uint64_t device_allocated_bytes{};
    std::uint64_t host_freed_bytes{};
    std::uint64_t device_freed_bytes{};
    std::uint64_t h2d_bytes{};
    std::uint64_t d2h_bytes{};
    std::uint64_t d2d_bytes{};
    std::uint64_t h2h_bytes{};
    std::uint64_t copy_calls{};
    std::uint64_t synchronization_calls{};
};

struct EventRecord {
    std::string name;
    EventKind kind{EventKind::event};
    EventMetrics metrics;
    std::map<std::string, double> values;
    std::map<std::string, std::string> attributes;
    std::vector<EventRecord> children;
};

// Aggregated by hierarchical path. The squared terms make mean/variance
// available when a kernel or operation is emitted more than once.
struct EventSummary {
    std::string path;
    EventKind kind{EventKind::event};
    std::uint64_t calls{};
    double wall_time_ms{};
    double wall_time_squared_ms2{};
    double device_time_ms{};
    double device_time_squared_ms2{};
    double estimated_flops{};
    double estimated_flops_squared{};
    std::uint64_t allocation_calls{};
    std::uint64_t free_calls{};
    std::uint64_t host_allocated_bytes{};
    std::uint64_t device_allocated_bytes{};
    std::uint64_t host_freed_bytes{};
    std::uint64_t device_freed_bytes{};
    std::uint64_t h2d_bytes{};
    std::uint64_t d2h_bytes{};
    std::uint64_t d2d_bytes{};
    std::uint64_t h2h_bytes{};
    std::uint64_t copy_calls{};
    std::uint64_t synchronization_calls{};
};

// A small PETSc-style hierarchical logger. Wall time is measured on the host;
// asynchronous device time must be supplied by a backend using add_device_time.
// Keeping the two clocks separate prevents GPU work from being double counted.
class Profiler {
public:
    class ScopedEvent {
    public:
        ScopedEvent() = default;
        ScopedEvent(const ScopedEvent&) = delete;
        ScopedEvent& operator=(const ScopedEvent&) = delete;
        ScopedEvent(ScopedEvent&& other) noexcept;
        ScopedEvent& operator=(ScopedEvent&& other) noexcept;
        ~ScopedEvent();

        void finish() noexcept;

    private:
        friend class Profiler;
        ScopedEvent(Profiler* profiler, EventRecord* record) noexcept;

        Profiler* profiler_{};
        EventRecord* record_{};
        std::chrono::steady_clock::time_point start_{};
    };

    [[nodiscard]] ScopedEvent scoped(
        std::string name,
        EventKind kind = EventKind::event);

    void add_device_time(double milliseconds);
    void add_estimated_flops(double flops);
    void add_value(std::string name, double value);
    void add_attribute(std::string name, std::string value);
    void record_allocation(MemorySpace space, std::uint64_t bytes);
    void record_free(MemorySpace space, std::uint64_t bytes);
    void record_copy(MemorySpace source, MemorySpace destination, std::uint64_t bytes);
    void record_synchronization();

    [[nodiscard]] const std::vector<EventRecord>& records() const noexcept;
    void reset();

private:
    friend class ScopedEvent;
    [[nodiscard]] EventRecord& append(std::string name, EventKind kind);
    [[nodiscard]] EventRecord& current();
    void finish(EventRecord* record, std::chrono::steady_clock::time_point start) noexcept;

    std::vector<EventRecord> roots_;
    std::vector<EventRecord*> stack_;
};

[[nodiscard]] std::string_view to_string(EventKind kind) noexcept;
[[nodiscard]] std::vector<EventSummary> summarize_events(
    const std::vector<EventRecord>& records);

}  // namespace eda_gpu
