#!/usr/bin/env bash
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

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
echo "Task 1 CPU baseline"
klu_header="${project_root}/third_party/ngspice/src/include/ngspice/klu.h"
klu_source="${project_root}/third_party/ngspice/src/maths/KLU/klu_factor.c"
if [[ -f "${klu_header}" && -f "${klu_source}" ]]; then
    echo "vendored KLU: available"
else
    echo "vendored KLU: MISSING"
    status=1
fi

echo
echo "Current presets"
cmake --list-presets 2>/dev/null || status=1

exit "${status}"
