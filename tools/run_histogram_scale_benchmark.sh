#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-release-proxy}"
JOBS="${JOBS:-1}"
ROUTE="${ROUTE:-both}"
REPEATS="${REPEATS:-1}"
# Minimum requested SRS size. The wrapper raises this to the exact
# route/case-dependent requirement below.
SRS_SIZE="${SRS_SIZE:-1}"
SRS_PATH="${SRS_PATH:-}"
SRS_CACHE="${SRS_CACHE:-}"
SRS_CACHE_DIR="${SRS_CACHE_DIR:-}"
SKIP_BUILD="${SKIP_BUILD:-0}"
PROOF_N="${PROOF_N:-}"
PROOF_M="${PROOF_M:-}"
PROOF_DIR="${PROOF_DIR:-}"
VERIFIER_ONLY="${VERIFIER_ONLY:-0}"

DEFAULT_CASES=(
  "32768:32768"
  "65536:32768"
  "131072:32768"
  "262144:32768"
  "524288:32768"
  "131072:16384"
  "131072:65536"
  "131072:131072"
  "131072:262144"
)

if [[ -n "${HISTOGRAM_CASES:-}" ]]; then
  read -r -a CASES <<< "${HISTOGRAM_CASES}"
else
  CASES=("${DEFAULT_CASES[@]}")
fi

if [[ -n "${SRS_PATH}" && -n "${SRS_CACHE}" ]]; then
  echo "error: SRS_PATH and SRS_CACHE are mutually exclusive" >&2
  exit 1
fi
if [[ -n "${SRS_PATH}" && -n "${SRS_CACHE_DIR}" ]]; then
  echo "error: SRS_PATH and SRS_CACHE_DIR are mutually exclusive" >&2
  exit 1
fi
if [[ -n "${SRS_CACHE}" && -n "${SRS_CACHE_DIR}" ]]; then
  echo "error: SRS_CACHE and SRS_CACHE_DIR are mutually exclusive" >&2
  exit 1
fi
if [[ -n "${PROOF_N}" || -n "${PROOF_M}" ]]; then
  if [[ -z "${PROOF_N}" || -z "${PROOF_M}" ]]; then
    echo "error: PROOF_N and PROOF_M must be set together" >&2
    exit 1
  fi
fi

next_power_of_two() {
  local value="$1" out=1
  while (( out < value )); do
    (( out *= 2 ))
  done
  printf '%s' "${out}"
}

runs_ram=0
case "${ROUTE}" in
  ram|both|all) runs_ram=1 ;;
  terrae|weighted-logup) ;;
  *) echo "error: ROUTE must be both, all, terrae, ram, or weighted-logup" >&2; exit 1 ;;
esac

max_required_domain=65536
for case_spec in "${CASES[@]}"; do
  IFS=':' read -r case_n case_m <<< "${case_spec}"
  if [[ -z "${case_n}" || -z "${case_m}" ]]; then
    echo "error: invalid histogram case ${case_spec}; expected n:m" >&2
    exit 1
  fi
  proof_n="${PROOF_N:-${case_n}}"
  proof_m="${PROOF_M:-${case_m}}"
  (( proof_n >= case_n && proof_m >= case_m )) || {
    echo "error: proof shape must cover case ${case_spec}" >&2
    exit 1
  }
  (( proof_n > max_required_domain )) && max_required_domain="${proof_n}"
  (( proof_m > max_required_domain )) && max_required_domain="${proof_m}"
  if (( runs_ram )); then
    ram_trace=$(next_power_of_two $((proof_n + 2 * proof_m)))
    (( ram_trace > max_required_domain )) && max_required_domain="${ram_trace}"
  fi
done
required_srs_size=$((max_required_domain * 4))
(( SRS_SIZE > required_srs_size )) && required_srs_size="${SRS_SIZE}"
SRS_SIZE="${required_srs_size}"
if [[ -n "${SRS_CACHE_DIR}" ]]; then
  SRS_CACHE="${SRS_CACHE_DIR%/}/kzg_srs_${SRS_SIZE}.params"
fi

{
  printf 'histogram_scale_config\tcases\t%s\troute\t%s\trepeats\t%s\tsrs_size\t%s\tbuild_dir\t%s\tverifier_only\t%s\n' \
    "${#CASES[@]}" "${ROUTE}" "${REPEATS}" "${SRS_SIZE}" "${BUILD_DIR}" "${VERIFIER_ONLY}"
  if [[ -n "${SRS_CACHE}" ]]; then
    printf 'histogram_scale_srs_cache\t%s\n' "${SRS_CACHE}"
  fi
  if [[ -n "${PROOF_DIR}" ]]; then
    printf 'histogram_scale_proof_dir\t%s\n' "${PROOF_DIR}"
  fi
  for i in "${!CASES[@]}"; do
    printf 'histogram_scale_case\t%s/%s\t%s\n' "$((i + 1))" "${#CASES[@]}" "${CASES[$i]}"
  done
} >&2

if [[ "${SKIP_BUILD}" != "1" ]]; then
  cmake -S . -B "${BUILD_DIR}" >&2
  cmake --build "${BUILD_DIR}" --target benchmark_histogram_primitives -- -j"${JOBS}" >&2
else
  printf 'histogram_scale_skip_build\tbuild_dir\t%s\n' "${BUILD_DIR}" >&2
fi

cmd=(
  "./${BUILD_DIR}/examples/benchmark_histogram_primitives"
  "--route" "${ROUTE}"
  "--repeats" "${REPEATS}"
  "--srs-size" "${SRS_SIZE}"
)

if [[ -n "${SRS_PATH}" && -n "${SRS_CACHE}" ]]; then
  echo "error: SRS_PATH and SRS_CACHE are mutually exclusive" >&2
  exit 1
fi

if [[ -n "${PROOF_N}" || -n "${PROOF_M}" ]]; then
  if [[ -z "${PROOF_N}" || -z "${PROOF_M}" ]]; then
    echo "error: PROOF_N and PROOF_M must be set together" >&2
    exit 1
  fi
  cmd+=("--proof-n" "${PROOF_N}" "--proof-m" "${PROOF_M}")
fi

if [[ -n "${SRS_PATH}" ]]; then
  cmd+=("--srs" "${SRS_PATH}")
fi

if [[ -n "${SRS_CACHE}" ]]; then
  cmd+=("--srs-cache" "${SRS_CACHE}")
fi

if [[ -n "${PROOF_DIR}" ]]; then
  cmd+=("--proof-dir" "${PROOF_DIR}")
fi

if [[ "${VERIFIER_ONLY}" == "1" ]]; then
  cmd+=("--verifier-only")
fi

for case_spec in "${CASES[@]}"; do
  cmd+=("--case" "${case_spec}")
done

{
  printf '==> running'
  printf ' %q' "${cmd[@]}"
  printf '\n'
} >&2
exec "${cmd[@]}"
