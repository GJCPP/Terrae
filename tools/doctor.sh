#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="${repo_root}/data/datasets/raw"

status=0

check_command() {
  local name="$1"
  local hint="$2"
  if command -v "${name}" >/dev/null 2>&1; then
    printf 'ok\t%s\t%s\n' "${name}" "$(command -v "${name}")"
  else
    printf 'missing\t%s\t%s\n' "${name}" "${hint}"
    status=1
  fi
}

check_python() {
  local python_path
  local python_version
  if ! command -v python3 >/dev/null 2>&1; then
    printf 'missing\tpython3\tinstall Python 3.10 or newer\n'
    status=1
    return
  fi

  python_path="$(command -v python3)"
  python_version="$(python3 --version 2>&1)"
  if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    printf 'ok\tpython3\t%s\n' "${python_path}"
  else
    printf 'unsupported\tpython3\t%s; install Python 3.10 or newer\n' "${python_version}"
    status=1
  fi
  printf 'version\tpython3\t%s\n' "${python_version}"
}

printf 'Terrae environment check\n'
printf 'repo\t%s\n' "${repo_root}"

check_command cmake "install CMake 3.14 or newer"
if [[ -n "${TERRAE_MCL_SOURCE_DIR:-}" ]]; then
  if [[ -f "${TERRAE_MCL_SOURCE_DIR}/CMakeLists.txt" ]]; then
    printf 'ok\tmcl_source\t%s\n' "${TERRAE_MCL_SOURCE_DIR}"
  else
    printf 'missing\tmcl_source\t%s does not contain CMakeLists.txt\n' "${TERRAE_MCL_SOURCE_DIR}"
    status=1
  fi
else
  check_command git "install Git for the default mcl fetch, or set TERRAE_MCL_SOURCE_DIR"
fi
check_python
check_command c++ "install a C++17-capable compiler such as g++ or clang++"
check_command curl "install curl; make datasets uses it"
check_command unzip "install unzip; the Dry Bean dataset is distributed as a zip"

if command -v cmake >/dev/null 2>&1; then
  cmake_version="$(cmake --version | sed -n '1p')"
  printf 'version\tcmake\t%s\n' "${cmake_version}"
fi

if [[ -f "${raw_dir}/iris.data" ]]; then
  printf 'ok\tdataset\tiris.data\n'
else
  printf 'missing\tdataset\tiris.data; run make datasets before GBDT benchmarks\n'
fi

if [[ -f "${raw_dir}/Dry_Bean_Dataset.arff" ]]; then
  printf 'ok\tdataset\tDry_Bean_Dataset.arff\n'
else
  printf 'missing\tdataset\tDry_Bean_Dataset.arff; run make datasets before full GBDT benchmarks\n'
fi

if [[ "${status}" -eq 0 ]]; then
  printf 'doctor\tok\n'
else
  printf 'doctor\tmissing_required_tools\n'
fi

exit "${status}"
