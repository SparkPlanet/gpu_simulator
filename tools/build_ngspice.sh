#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
source_dir="${project_root}/third_party/ngspice"
build_dir="${project_root}/build/third_party/ngspice"
install_dir="${project_root}/build/tools/ngspice"

if [[ ! -x "${source_dir}/configure" ]]; then
    echo "Missing vendored ngspice source: ${source_dir}" >&2
    exit 1
fi

mkdir -p "${build_dir}" "${install_dir}"

cd "${build_dir}"
"${source_dir}/configure" \
    --prefix="${install_dir}" \
    --without-x \
    --with-readline=no \
    --with-fftw3=no \
    --disable-xspice \
    --disable-osdi \
    --disable-openmp \
    --disable-dependency-tracking \
    CFLAGS="-O2 -g0"

make -j"$(nproc)"
make install

"${install_dir}/bin/ngspice" --version
