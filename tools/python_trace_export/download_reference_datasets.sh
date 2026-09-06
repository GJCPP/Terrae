#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
raw_dir="${repo_root}/data/datasets/raw"
mkdir -p "${raw_dir}"

force="${FORCE_DATASETS:-0}"
dataset_set="${DATASET_SET:-all}"

fetch() {
  local url="$1"
  local output="$2"
  if [[ "${force}" != "1" && -s "${output}" ]]; then
    printf 'dataset_exists\t%s\n' "${output}"
    return
  fi
  printf 'dataset_download\t%s\n' "${output}"
  curl -L "${url}" -o "${output}"
}

fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" \
  "${raw_dir}/iris.data"

if [[ "${dataset_set}" == "smoke" ]]; then
  exit 0
fi

if [[ "${dataset_set}" != "all" ]]; then
  echo "error: DATASET_SET must be 'smoke' or 'all'" >&2
  exit 1
fi

fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" \
  "${raw_dir}/wine.data"
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra" \
  "${raw_dir}/optdigits.tra"
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes" \
  "${raw_dir}/optdigits.tes"
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" \
  "${raw_dir}/wdbc.data"
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names" \
  "${raw_dir}/wdbc.names"

if [[ "${force}" == "1" || ! -s "${raw_dir}/Dry_Bean_Dataset.arff" ]]; then
  fetch "https://archive.ics.uci.edu/static/public/602/dry%2Bbean%2Bdataset.zip" \
    "${raw_dir}/dry_bean_dataset.zip"
  unzip -p "${raw_dir}/dry_bean_dataset.zip" "DryBeanDataset/Dry_Bean_Dataset.arff" \
    > "${raw_dir}/Dry_Bean_Dataset.arff"
  printf 'dataset_extract\t%s\n' "${raw_dir}/Dry_Bean_Dataset.arff"
else
  printf 'dataset_exists\t%s\n' "${raw_dir}/Dry_Bean_Dataset.arff"
fi
