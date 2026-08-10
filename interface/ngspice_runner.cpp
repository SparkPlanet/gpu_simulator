#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#ifndef EDA_GPU_ROOT
#define EDA_GPU_ROOT "."
#endif

#ifndef EDA_GPU_NGSPICE
#define EDA_GPU_NGSPICE "build/tools/ngspice/bin/ngspice"
#endif

namespace {

struct Options {
    std::string case_name;
    std::filesystem::path ngspice{EDA_GPU_NGSPICE};
    std::filesystem::path log;
    std::filesystem::path output;
    std::int32_t timeout_seconds{600};
    bool help{};
};

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    auto value_after = [&](int& index, std::string_view option) -> std::string {
        if (++index >= argc) {
            throw std::runtime_error(std::string(option) + " requires a value");
        }
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--case") {
            options.case_name = value_after(index, argument);
        } else if (argument == "--ngspice") {
            options.ngspice = value_after(index, argument);
        } else if (argument == "--log") {
            options.log = value_after(index, argument);
        } else if (argument == "--output") {
            options.output = value_after(index, argument);
        } else if (argument == "--timeout") {
            std::size_t consumed{};
            const auto value = value_after(index, argument);
            const auto parsed = std::stoll(value, &consumed);
            if (consumed != value.size() || parsed <= 0 || parsed > INT32_MAX) {
                throw std::runtime_error("--timeout must be a positive 32-bit integer");
            }
            options.timeout_seconds = static_cast<std::int32_t>(parsed);
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }
    if (!options.help && options.case_name.empty()) {
        throw std::runtime_error("--case is required");
    }
    return options;
}

void print_help() {
    std::cout <<
        "Usage: ngspice_runner --case NAME|FILE [options]\n"
        "  --case VALUE       rc, c432, c1908, c3540, c7552 or a netlist path\n"
        "  --ngspice FILE     ngspice executable\n"
        "  --timeout SEC      process timeout (default: 600)\n"
        "  --log FILE         simulation log path\n"
        "  --output FILE      write the JSON result to a file\n";
}

[[nodiscard]] std::filesystem::path resolve_case(const std::string& name) {
    const std::filesystem::path root = EDA_GPU_ROOT;
    const std::map<std::string, std::filesystem::path> aliases{
        {"rc", root / "datasets/ngspice/rc_lowpass.cir"},
        {"c432", root / "datasets/ngspice/iscas85/Circuits/85/c432/c432.net"},
        {"c432_ann", root / "datasets/ngspice/iscas85/Circuits/85/c432/c432_ann.net"},
        {"c1908", root / "datasets/ngspice/iscas85/Circuits/85/c1908/c1908.net"},
        {"c1908_ann", root / "datasets/ngspice/iscas85/Circuits/85/c1908/c1908_ann.net"},
        {"c3540", root / "datasets/ngspice/iscas85/Circuits/85/c3540/c3540.net"},
        {"c3540_ann", root / "datasets/ngspice/iscas85/Circuits/85/c3540/c3540_ann.net"},
        {"c7552", root / "datasets/ngspice/iscas85/Circuits/85/c7552/c7552.net"},
    };
    const auto alias = aliases.find(name);
    const auto path = alias == aliases.end() ? std::filesystem::path(name) : alias->second;
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("case does not exist: " + path.string());
    }
    return std::filesystem::absolute(path);
}

[[nodiscard]] bool contains_text(
    const std::filesystem::path& path,
    std::string_view expected) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        if (line.find(expected) != std::string::npos) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::string escape_json(std::string_view value) {
    std::string escaped;
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped += '\\';
        escaped += character;
    }
    return escaped;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path wrapper;
    try {
        auto options = parse_options(argc, argv);
        if (options.help) {
            print_help();
            return 0;
        }

        const auto case_path = resolve_case(options.case_name);
        const auto ngspice = std::filesystem::absolute(options.ngspice);
        if (::access(ngspice.c_str(), X_OK) != 0) {
            throw std::runtime_error("ngspice is not executable: " + ngspice.string());
        }
        if (case_path.string().find('"') != std::string::npos) {
            throw std::runtime_error("case path cannot contain a double quote");
        }

        if (options.log.empty()) {
            options.log = std::filesystem::path(EDA_GPU_ROOT) / "build/ngspice" /
                          (case_path.stem().string() + ".log");
        }
        options.log = std::filesystem::absolute(options.log);
        std::filesystem::create_directories(options.log.parent_path());

        wrapper = std::filesystem::temp_directory_path() /
                  ("eda_gpu_" + std::to_string(::getpid()) + ".cir");
        {
            std::ofstream generated(wrapper);
            if (!generated) {
                throw std::runtime_error("cannot create temporary ngspice wrapper");
            }
            generated << "* generated by ngspice_runner\n"
                      << ".options klu\n"
                      << ".include \"" << case_path.string() << "\"\n"
                      << ".end\n";
        }

        const auto start = std::chrono::steady_clock::now();
        const auto child = ::fork();
        if (child < 0) {
            throw std::runtime_error("fork failed");
        }
        if (child == 0) {
            const auto log_fd = ::open(options.log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (log_fd < 0 || ::chdir(case_path.parent_path().c_str()) != 0 ||
                ::dup2(log_fd, STDOUT_FILENO) < 0 || ::dup2(log_fd, STDERR_FILENO) < 0) {
                _exit(126);
            }
            ::close(log_fd);
            ::setenv("LC_ALL", "C", 1);
            ::execl(ngspice.c_str(), ngspice.c_str(), "--no-spiceinit", "--batch",
                    wrapper.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }

        int status{};
        bool timed_out = false;
        pid_t wait_result{};
        while ((wait_result = ::waitpid(child, &status, WNOHANG)) == 0) {
            if (std::chrono::steady_clock::now() - start >
                std::chrono::seconds(options.timeout_seconds)) {
                ::kill(child, SIGKILL);
                wait_result = ::waitpid(child, &status, 0);
                timed_out = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
        std::filesystem::remove(wrapper);
        wrapper.clear();

        if (wait_result < 0) {
            throw std::runtime_error("waitpid failed");
        }
        if (timed_out) {
            throw std::runtime_error("ngspice timed out; log: " + options.log.string());
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw std::runtime_error("ngspice failed; log: " + options.log.string());
        }
        if (!contains_text(options.log, "Using KLU as Direct Linear Solver")) {
            throw std::runtime_error("ngspice did not use KLU; log: " + options.log.string());
        }

        std::ostringstream json;
        json << "{\n"
             << "  \"status\": \"ok\",\n"
             << "  \"case\": \"" << escape_json(case_path.string()) << "\",\n"
             << "  \"solver\": \"klu\",\n"
             << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
             << "  \"log\": \"" << escape_json(options.log.string()) << "\"\n"
             << "}\n";
        if (options.output.empty()) {
            std::cout << json.str();
        } else {
            if (options.output.has_parent_path()) {
                std::filesystem::create_directories(options.output.parent_path());
            }
            std::ofstream output(options.output);
            if (!output) {
                throw std::runtime_error("cannot write result: " + options.output.string());
            }
            output << json.str();
        }
        return 0;
    } catch (const std::exception& error) {
        if (!wrapper.empty()) std::filesystem::remove(wrapper);
        std::cerr << "ngspice_runner: " << error.what() << '\n';
        return 1;
    }
}
