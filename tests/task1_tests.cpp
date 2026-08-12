#include "executor.hpp"
#include "matrix.hpp"
#include "report.hpp"
#include "task1.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TestDeviceExecutor final : public eda_gpu::Executor {
public:
    using Executor::Executor;

    [[nodiscard]] std::string_view name() const noexcept override { return "test-device"; }
    [[nodiscard]] eda_gpu::MemorySpace memory_space() const noexcept override {
        return eda_gpu::MemorySpace::device;
    }

protected:
    [[nodiscard]] void* raw_allocate(std::size_t bytes) override {
        return ::operator new(bytes);
    }
    void raw_free(void* pointer) noexcept override { ::operator delete(pointer); }
    bool raw_copy_from(
        const eda_gpu::Executor&,
        void* destination,
        const void* source,
        std::size_t bytes) override {
        std::memmove(destination, source, bytes);
        return true;
    }
    bool raw_copy_to(
        const eda_gpu::Executor&,
        void* destination,
        const void* source,
        std::size_t bytes) const override {
        std::memmove(destination, source, bytes);
        return true;
    }
    void raw_synchronize() override {}
};

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

[[nodiscard]] const eda_gpu::EventRecord* find_event(
    const std::vector<eda_gpu::EventRecord>& events,
    std::string_view name) {
    for (const auto& event : events) {
        if (event.name == name) return &event;
        if (const auto* child = find_event(event.children, name)) return child;
    }
    return nullptr;
}

void test_profiler_and_executor() {
    eda_gpu::Profiler profiler;
    auto host = eda_gpu::make_host_executor(profiler);
    TestDeviceExecutor device(profiler);
    {
        auto stage = profiler.scoped("executor_test", eda_gpu::EventKind::stage);
        auto source = host->allocate(4U * sizeof(double));
        auto destination = host->allocate(4U * sizeof(double));
        const double values[]{1.0, 2.0, 3.0, 4.0};
        std::memcpy(source.data(), values, sizeof(values));
        host->copy_from(*host, destination.data(), source.data(), sizeof(values));
        auto device_values = device.allocate(sizeof(values));
        device.copy_from(*host, device_values.data(), source.data(), sizeof(values));
        host->copy_from(device, destination.data(), device_values.data(), sizeof(values));
        host->synchronize();
        require(std::memcmp(destination.data(), values, sizeof(values)) == 0,
                "host executor copy failed");
        source.reset();
        destination.reset();
        device_values.reset();
    }
    const auto& statistics = host->statistics();
    require(statistics.live_bytes == 0, "executor leaked tracked memory");
    require(statistics.peak_bytes == 8U * sizeof(double), "executor peak bytes are wrong");
    require(statistics.allocation_calls == 2 && statistics.free_calls == 2,
            "executor allocation counters are wrong");
    require(statistics.copy_calls == 2 && statistics.copied_bytes == sizeof(double) * 8U,
            "executor copy counters are wrong");
    const auto* event = find_event(profiler.records(), "executor_test");
    require(event != nullptr, "profiler lost the executor event");
    require(event->metrics.h2h_bytes == sizeof(double) * 4U,
            "profiler did not classify the host copy");
    require(event->metrics.h2d_bytes == sizeof(double) * 4U &&
                event->metrics.d2h_bytes == sizeof(double) * 4U,
            "profiler did not classify host/device transfers");
}

void test_task1_pipeline() {
    auto input = eda_gpu::make_poisson_2d(8);
    const std::vector<double> expected(
        static_cast<std::size_t>(input.matrix.dimension), 1.0);
    const auto right_hand_side = eda_gpu::multiply(input.matrix, expected);
    const eda_gpu::Task1Pipeline pipeline("cpu-klu");
    const auto result = pipeline.run({input.matrix, right_hand_side});

    require(result.solution.size() == expected.size(), "solution dimension is wrong");
    require(eda_gpu::relative_residual_l2(input.matrix, result.solution, right_hand_side) < 1e-12,
            "KLU Task 1 residual is too large");
    require(result.report.events.size() == 1 && result.report.events.front().name == "task1",
            "Task 1 root event is missing");
    require(find_event(result.report.events, "analyze") != nullptr,
            "analysis stage is missing");
    require(find_event(result.report.events, "klu_numeric_lu") != nullptr,
            "numeric LU event is missing");
    require(find_event(result.report.events, "klu_solve") != nullptr,
            "triangular solve event is missing");
    const auto summaries = eda_gpu::summarize_events(result.report.events);
    require(!summaries.empty() && summaries.front().path == "task1",
            "aggregated event summary is missing");
    require(result.report.backend_statistics.values.at("factor_fill_ratio") >= 1.0,
            "factor statistics are missing");
    const auto json = eda_gpu::task1_report_json(result.report);
    require(json.find("\"task\": \"task1\"") != std::string::npos,
            "Task 1 JSON schema is missing");
    require(json.find("\"exclusive_wall_time_ms\"") != std::string::npos,
            "exclusive timing is missing from JSON");
}

}  // namespace

int main() {
    try {
        test_profiler_and_executor();
        test_task1_pipeline();
        std::cout << "Task 1 infrastructure tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "task1_tests: " << error.what() << '\n';
        return 1;
    }
}
