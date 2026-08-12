#include "report.hpp"

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string_view>

namespace eda_gpu {
namespace {

[[nodiscard]] std::string escape_json(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (character < 0x20U) escaped += '?';
                else escaped += static_cast<char>(character);
        }
    }
    return escaped;
}

void indent(std::ostream& output, int spaces) {
    output << std::string(static_cast<std::size_t>(spaces), ' ');
}

[[nodiscard]] double exclusive_wall_time(const EventRecord& event) {
    double child_time{};
    for (const auto& child : event.children) child_time += child.metrics.wall_time_ms;
    return std::max(0.0, event.metrics.wall_time_ms - child_time);
}

void write_string_map(
    std::ostream& output,
    const std::map<std::string, std::string>& values,
    int spaces) {
    output << "{";
    if (!values.empty()) output << '\n';
    std::size_t position{};
    for (const auto& [name, value] : values) {
        indent(output, spaces + 2);
        output << '"' << escape_json(name) << "\": \"" << escape_json(value) << '"';
        output << (++position == values.size() ? "\n" : ",\n");
    }
    if (!values.empty()) indent(output, spaces);
    output << "}";
}

void write_number_map(
    std::ostream& output,
    const std::map<std::string, double>& values,
    int spaces) {
    output << "{";
    if (!values.empty()) output << '\n';
    std::size_t position{};
    for (const auto& [name, value] : values) {
        indent(output, spaces + 2);
        output << '"' << escape_json(name) << "\": " << value;
        output << (++position == values.size() ? "\n" : ",\n");
    }
    if (!values.empty()) indent(output, spaces);
    output << "}";
}

void write_event(std::ostream& output, const EventRecord& event, int spaces) {
    indent(output, spaces);
    output << "{\n";
    indent(output, spaces + 2);
    output << "\"name\": \"" << escape_json(event.name) << "\",\n";
    indent(output, spaces + 2);
    output << "\"kind\": \"" << to_string(event.kind) << "\",\n";
    indent(output, spaces + 2);
    output << "\"metrics\": {\n";
    const auto& metrics = event.metrics;
    const std::pair<std::string_view, double> floating_metrics[]{
        {"wall_time_ms", metrics.wall_time_ms},
        {"exclusive_wall_time_ms", exclusive_wall_time(event)},
        {"device_time_ms", metrics.device_time_ms},
        {"estimated_flops", metrics.estimated_flops},
    };
    for (const auto& [name, value] : floating_metrics) {
        indent(output, spaces + 4);
        output << '"' << name << "\": " << value << ",\n";
    }
    const std::pair<std::string_view, std::uint64_t> integer_metrics[]{
        {"calls", metrics.calls},
        {"allocation_calls", metrics.allocation_calls},
        {"free_calls", metrics.free_calls},
        {"host_allocated_bytes", metrics.host_allocated_bytes},
        {"device_allocated_bytes", metrics.device_allocated_bytes},
        {"host_freed_bytes", metrics.host_freed_bytes},
        {"device_freed_bytes", metrics.device_freed_bytes},
        {"h2d_bytes", metrics.h2d_bytes},
        {"d2h_bytes", metrics.d2h_bytes},
        {"d2d_bytes", metrics.d2d_bytes},
        {"h2h_bytes", metrics.h2h_bytes},
        {"copy_calls", metrics.copy_calls},
        {"synchronization_calls", metrics.synchronization_calls},
    };
    for (std::size_t index = 0; index < std::size(integer_metrics); ++index) {
        indent(output, spaces + 4);
        output << '"' << integer_metrics[index].first << "\": "
               << integer_metrics[index].second;
        output << (index + 1U == std::size(integer_metrics) ? "\n" : ",\n");
    }
    indent(output, spaces + 2);
    output << "},\n";
    indent(output, spaces + 2);
    output << "\"values\": ";
    write_number_map(output, event.values, spaces + 2);
    output << ",\n";
    indent(output, spaces + 2);
    output << "\"attributes\": ";
    write_string_map(output, event.attributes, spaces + 2);
    output << ",\n";
    indent(output, spaces + 2);
    output << "\"children\": [";
    if (!event.children.empty()) output << '\n';
    for (std::size_t index = 0; index < event.children.size(); ++index) {
        write_event(output, event.children[index], spaces + 4);
        output << (index + 1U == event.children.size() ? "\n" : ",\n");
    }
    if (!event.children.empty()) indent(output, spaces + 2);
    output << "]\n";
    indent(output, spaces);
    output << "}";
}

void write_event_summary(std::ostream& output, const EventSummary& summary, int spaces) {
    indent(output, spaces);
    output << "{\n";
    indent(output, spaces + 2);
    output << "\"path\": \"" << escape_json(summary.path) << "\",\n";
    indent(output, spaces + 2);
    output << "\"kind\": \"" << to_string(summary.kind) << "\",\n";
    const std::pair<std::string_view, double> floating[]{
        {"wall_time_ms", summary.wall_time_ms},
        {"wall_time_squared_ms2", summary.wall_time_squared_ms2},
        {"device_time_ms", summary.device_time_ms},
        {"device_time_squared_ms2", summary.device_time_squared_ms2},
        {"estimated_flops", summary.estimated_flops},
        {"estimated_flops_squared", summary.estimated_flops_squared},
    };
    for (const auto& [name, value] : floating) {
        indent(output, spaces + 2);
        output << '"' << name << "\": " << value << ",\n";
    }
    const std::pair<std::string_view, std::uint64_t> integers[]{
        {"calls", summary.calls},
        {"allocation_calls", summary.allocation_calls},
        {"free_calls", summary.free_calls},
        {"host_allocated_bytes", summary.host_allocated_bytes},
        {"device_allocated_bytes", summary.device_allocated_bytes},
        {"host_freed_bytes", summary.host_freed_bytes},
        {"device_freed_bytes", summary.device_freed_bytes},
        {"h2d_bytes", summary.h2d_bytes},
        {"d2h_bytes", summary.d2h_bytes},
        {"d2d_bytes", summary.d2d_bytes},
        {"h2h_bytes", summary.h2h_bytes},
        {"copy_calls", summary.copy_calls},
        {"synchronization_calls", summary.synchronization_calls},
    };
    for (std::size_t index = 0; index < std::size(integers); ++index) {
        indent(output, spaces + 2);
        output << '"' << integers[index].first << "\": " << integers[index].second;
        output << (index + 1U == std::size(integers) ? "\n" : ",\n");
    }
    indent(output, spaces);
    output << "}";
}

}  // namespace

std::string task1_report_json(const Task1Report& report) {
    const auto event_summaries = summarize_events(report.events);
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"task\": \"task1\",\n"
           << "  \"lifecycle\": \"one-shot\",\n"
           << "  \"backend\": \"" << escape_json(report.backend) << "\",\n"
           << "  \"matrix\": {\"dimension\": " << report.dimension
           << ", \"nonzeros\": " << report.matrix_nonzeros << "},\n"
           << "  \"backend_statistics\": {\n"
           << "    \"values\": ";
    write_number_map(output, report.backend_statistics.values, 4);
    output << ",\n    \"attributes\": ";
    write_string_map(output, report.backend_statistics.attributes, 4);
    output << "\n  },\n"
           << "  \"host_executor_statistics\": {\n"
           << "    \"live_bytes\": " << report.host_executor_statistics.live_bytes << ",\n"
           << "    \"peak_bytes\": " << report.host_executor_statistics.peak_bytes << ",\n"
           << "    \"allocation_calls\": "
           << report.host_executor_statistics.allocation_calls << ",\n"
           << "    \"free_calls\": " << report.host_executor_statistics.free_calls << ",\n"
           << "    \"copy_calls\": " << report.host_executor_statistics.copy_calls << ",\n"
           << "    \"copied_bytes\": " << report.host_executor_statistics.copied_bytes << ",\n"
           << "    \"synchronization_calls\": "
           << report.host_executor_statistics.synchronization_calls << "\n"
           << "  },\n"
           << "  \"event_summary\": [";
    if (!event_summaries.empty()) output << '\n';
    for (std::size_t index = 0; index < event_summaries.size(); ++index) {
        write_event_summary(output, event_summaries[index], 4);
        output << (index + 1U == event_summaries.size() ? "\n" : ",\n");
    }
    if (!event_summaries.empty()) output << "  ";
    output << "],\n"
           << "  \"event_tree\": [";
    if (!report.events.empty()) output << '\n';
    for (std::size_t index = 0; index < report.events.size(); ++index) {
        write_event(output, report.events[index], 4);
        output << (index + 1U == report.events.size() ? "\n" : ",\n");
    }
    if (!report.events.empty()) output << "  ";
    output << "]\n}\n";
    return output.str();
}

}  // namespace eda_gpu
