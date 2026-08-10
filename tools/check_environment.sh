#!/usr/bin/env bash
set -u

status=0

check_command() {
    local command_name="$1"
    if command -v "${command_name}" >/dev/null 2>&1; then
        printf '%-12s %s\n' "${command_name}" "$(command -v "${command_name}")"
    else
        printf '%-12s MISSING\n' "${command_name}"
        status=1
    fi
}

echo "Required tools"
check_command cmake
check_command cc
check_command c++

echo
echo "ngspice CPU baseline"
project_ngspice="./build/tools/ngspice/bin/ngspice"
if [[ ! -x "${project_ngspice}" ]]; then
    echo "MISSING: ${project_ngspice}"
    echo "Run ./tools/build_ngspice.sh first."
    status=1
    ngspice_version=""
else
    ngspice_version="$("${project_ngspice}" --version 2>&1)"
fi
if printf '%s\n' "${ngspice_version}" | grep -q "KLU Direct Linear Solver"; then
    printf '%s\n' "${ngspice_version}" | grep -E 'ngspice-|KLU Direct Linear Solver'
else
    echo "ngspice is present but KLU support is missing."
    status=1
fi

echo
echo "Optional future GPU tools"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1
else
    echo "nvcc: not installed (not required in the current CPU phase)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
        --format=csv,noheader; then
        echo "NVIDIA GPU is not visible here (not required in the current CPU phase)."
    fi
fi

exit "${status}"
