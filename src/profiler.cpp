#include "eda_gpu/profiler.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace eda_gpu {

Profiler::ScopedEvent::ScopedEvent(Profiler* profiler, EventRecord* record) noexcept
    : profiler_(profiler), record_(record), start_(std::chrono::steady_clock::now()) {}

Profiler::ScopedEvent::ScopedEvent(ScopedEvent&& other) noexcept
    : profiler_(other.profiler_), record_(other.record_), start_(other.start_) {
    other.profiler_ = nullptr;
    other.record_ = nullptr;
}

Profiler::ScopedEvent& Profiler::ScopedEvent::operator=(ScopedEvent&& other) noexcept {
    if (this != &other) {
        finish();
        profiler_ = other.profiler_;
        record_ = other.record_;
        start_ = other.start_;
        other.profiler_ = nullptr;
        other.record_ = nullptr;
    }
    return *this;
}

Profiler::ScopedEvent::~ScopedEvent() {
    finish();
}

void Profiler::ScopedEvent::finish() noexcept {
    if (profiler_ != nullptr && record_ != nullptr) {
        profiler_->finish(record_, start_);
        profiler_ = nullptr;
        record_ = nullptr;
    }
}

Profiler::ScopedEvent Profiler::scoped(std::string name, EventKind kind) {
    auto& record = append(std::move(name), kind);
    stack_.push_back(&record);
    return ScopedEvent(this, &record);
}

void Profiler::add_device_time(double milliseconds) {
    current().metrics.device_time_ms += milliseconds;
}

void Profiler::add_estimated_flops(double flops) {
    current().metrics.estimated_flops += flops;
}

void Profiler::add_value(std::string name, double value) {
    current().values.insert_or_assign(std::move(name), value);
}

void Profiler::add_attribute(std::string name, std::string value) {
    current().attributes.insert_or_assign(std::move(name), std::move(value));
}

void Profiler::record_allocation(MemorySpace space, std::uint64_t bytes) {
    if (stack_.empty()) return;
    auto& metrics = stack_.back()->metrics;
    ++metrics.allocation_calls;
    if (space == MemorySpace::host) metrics.host_allocated_bytes += bytes;
    else metrics.device_allocated_bytes += bytes;
}

void Profiler::record_free(MemorySpace space, std::uint64_t bytes) {
    if (stack_.empty()) return;
    auto& metrics = stack_.back()->metrics;
    ++metrics.free_calls;
    if (space == MemorySpace::host) metrics.host_freed_bytes += bytes;
    else metrics.device_freed_bytes += bytes;
}

void Profiler::record_copy(
    MemorySpace source,
    MemorySpace destination,
    std::uint64_t bytes) {
    if (stack_.empty()) return;
    auto& metrics = stack_.back()->metrics;
    ++metrics.copy_calls;
    if (source == MemorySpace::host && destination == MemorySpace::device) {
        metrics.h2d_bytes += bytes;
    } else if (source == MemorySpace::device && destination == MemorySpace::host) {
        metrics.d2h_bytes += bytes;
    } else if (source == MemorySpace::device) {
        metrics.d2d_bytes += bytes;
    } else {
        metrics.h2h_bytes += bytes;
    }
}

void Profiler::record_synchronization() {
    if (!stack_.empty()) ++stack_.back()->metrics.synchronization_calls;
}

const std::vector<EventRecord>& Profiler::records() const noexcept {
    return roots_;
}

void Profiler::reset() {
    if (!stack_.empty()) throw std::logic_error("cannot reset an active profiler");
    roots_.clear();
}

EventRecord& Profiler::append(std::string name, EventKind kind) {
    EventRecord record;
    record.name = std::move(name);
    record.kind = kind;
    if (stack_.empty()) {
        roots_.push_back(std::move(record));
        return roots_.back();
    }
    stack_.back()->children.push_back(std::move(record));
    return stack_.back()->children.back();
}

EventRecord& Profiler::current() {
    if (stack_.empty()) throw std::logic_error("profiler metric requires an active event");
    return *stack_.back();
}

void Profiler::finish(
    EventRecord* record,
    std::chrono::steady_clock::time_point start) noexcept {
    record->metrics.wall_time_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    if (!stack_.empty() && stack_.back() == record) {
        stack_.pop_back();
        return;
    }
    const auto found = std::find(stack_.begin(), stack_.end(), record);
    if (found != stack_.end()) stack_.erase(found);
}

std::string_view to_string(EventKind kind) noexcept {
    switch (kind) {
        case EventKind::stage: return "stage";
        case EventKind::event: return "event";
        case EventKind::kernel: return "kernel";
        case EventKind::transfer: return "transfer";
        case EventKind::allocation: return "allocation";
        case EventKind::synchronization: return "synchronization";
    }
    return "unknown";
}

std::vector<EventSummary> summarize_events(const std::vector<EventRecord>& records) {
    std::vector<EventSummary> summaries;
    std::map<std::string, std::size_t> path_to_index;
    const auto visit = [&](const auto& self, const EventRecord& record,
                           const std::string& parent_path) -> void {
        const auto path = parent_path.empty() ? record.name : parent_path + "/" + record.name;
        auto found = path_to_index.find(path);
        if (found == path_to_index.end()) {
            found = path_to_index.emplace(path, summaries.size()).first;
            EventSummary summary;
            summary.path = path;
            summary.kind = record.kind;
            summaries.push_back(std::move(summary));
        }
        auto& summary = summaries[found->second];
        const auto& metrics = record.metrics;
        summary.calls += metrics.calls;
        summary.wall_time_ms += metrics.wall_time_ms;
        summary.wall_time_squared_ms2 += metrics.wall_time_ms * metrics.wall_time_ms;
        summary.device_time_ms += metrics.device_time_ms;
        summary.device_time_squared_ms2 += metrics.device_time_ms * metrics.device_time_ms;
        summary.estimated_flops += metrics.estimated_flops;
        summary.estimated_flops_squared +=
            metrics.estimated_flops * metrics.estimated_flops;
        summary.allocation_calls += metrics.allocation_calls;
        summary.free_calls += metrics.free_calls;
        summary.host_allocated_bytes += metrics.host_allocated_bytes;
        summary.device_allocated_bytes += metrics.device_allocated_bytes;
        summary.host_freed_bytes += metrics.host_freed_bytes;
        summary.device_freed_bytes += metrics.device_freed_bytes;
        summary.h2d_bytes += metrics.h2d_bytes;
        summary.d2h_bytes += metrics.d2h_bytes;
        summary.d2d_bytes += metrics.d2d_bytes;
        summary.h2h_bytes += metrics.h2h_bytes;
        summary.copy_calls += metrics.copy_calls;
        summary.synchronization_calls += metrics.synchronization_calls;
        for (const auto& child : record.children) self(self, child, path);
    };
    for (const auto& record : records) visit(visit, record, std::string{});
    return summaries;
}

}  // namespace eda_gpu
