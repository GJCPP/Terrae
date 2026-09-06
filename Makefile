.PHONY: help doctor configure build test datasets datasets-smoke srs benchmark benchmark-rerun-terrae-training benchmark-rerun-certxgb-training benchmark-rerun-certxgb benchmark-rerun-certxgb-inference benchmark-practical benchmark-paper benchmark-paper-summary benchmark-smoke benchmark-histogram-smoke benchmark-histogram benchmark-histogram-build benchmark-histogram-smoke-run benchmark-histogram-run artifact

BUILD_DIR ?= build
CMAKE ?= cmake
CTEST ?= ctest
CMAKE_CONFIGURE_ARGS ?=
TERRAE_MCL_SOURCE_DIR ?=
TERRAE_MCL_CMAKE_ARG = $(if $(strip $(TERRAE_MCL_SOURCE_DIR)),-DTERRAE_MCL_SOURCE_DIR="$(abspath $(TERRAE_MCL_SOURCE_DIR))",)
export TERRAE_MCL_SOURCE_DIR
CTEST_ARGS ?= --output-on-failure
SRS_CACHE_DIR ?= out/srs
SRS_PATH ?= $(BUILD_DIR)/kzg_srs.params
SRS_DEG1_PLUS_1 ?= 131072
SRS_DEG2_PLUS_1 ?= $(SRS_DEG1_PLUS_1)
GBDT_SRS_CACHE_DIR ?= $(SRS_CACHE_DIR)/gbdt
BENCHMARK_BUILD_DIR ?= build-release
BENCHMARK_OUTPUT_DIR ?= out/benchmark
BENCHMARK_CASES ?= tools/python_trace_export/configs/benchmark_rebuttal_full.json
BENCHMARK_LOG ?= $(BENCHMARK_OUTPUT_DIR)/benchmark.log
BENCHMARK_RERUN_STAGES ?=
BENCHMARK_RERUN_STAGE_ARGS = $(foreach stage,$(BENCHMARK_RERUN_STAGES),--rerun-stage $(stage))
BENCHMARK_CASE_IDS ?=
BENCHMARK_CASE_ID_ARGS = $(foreach case_id,$(BENCHMARK_CASE_IDS),--case-id $(case_id))
BENCHMARK_ARGS ?= --cases-json $(BENCHMARK_CASES) --output-dir $(BENCHMARK_OUTPUT_DIR) --build-dir $(BENCHMARK_BUILD_DIR) --srs-cache-dir $(GBDT_SRS_CACHE_DIR) --jobs 1 --no-timeout --reuse-existing-stages $(BENCHMARK_CASE_ID_ARGS) $(BENCHMARK_RERUN_STAGE_ARGS)
BENCHMARK_MEMORY_GUARD_ENV ?= BENCHMARK_ABORT_ON_SWAP=0 BENCHMARK_SWAP_LIMIT_BYTES= BENCHMARK_SWAP_LIMIT_DURATION_SECONDS= BENCHMARK_MIN_MEM_AVAILABLE_BYTES= BENCHMARK_MIN_MEM_AVAILABLE_DURATION_SECONDS= BENCHMARK_ABORT_ON_OOM=1 BENCHMARK_RSS_LIMIT_BYTES=193273528320
PRACTICAL_BENCHMARK_OUTPUT_DIR ?= out/benchmark_practical
PRACTICAL_BENCHMARK_CASES ?= tools/python_trace_export/configs/benchmark_practical_N1024_d3_T124.json
PRACTICAL_BENCHMARK_LOG ?= $(PRACTICAL_BENCHMARK_OUTPUT_DIR)/benchmark.log
PRACTICAL_BENCHMARK_ARGS ?= --cases-json $(PRACTICAL_BENCHMARK_CASES) --output-dir $(PRACTICAL_BENCHMARK_OUTPUT_DIR) --build-dir $(BENCHMARK_BUILD_DIR) --srs-cache-dir $(GBDT_SRS_CACHE_DIR) --jobs 1 --no-timeout --reuse-existing-stages --stop-subsequent-t-after-failure
PAPER_BENCHMARK_OUTPUT_DIR ?= out/benchmark_paper
HISTOGRAM_SMOKE_SRS_CACHE_DIR ?= $(SRS_CACHE_DIR)/histogram
HISTOGRAM_SRS_CACHE_DIR ?= $(SRS_CACHE_DIR)/histogram
HISTOGRAM_BUILD_DIR ?= $(BENCHMARK_BUILD_DIR)
GBDT_SMOKE_OUTPUT_DIR ?= out/benchmark_gbdt_smoke
HISTOGRAM_SMOKE_OUTPUT_DIR ?= out/benchmark_histogram_smoke
HISTOGRAM_OUTPUT_DIR ?= out/benchmark_histogram
HISTOGRAM_ROUTE ?= all
GBDT_SMOKE_ARGS ?= --cases-json tools/python_trace_export/configs/benchmark_smoke.json --output-dir $(GBDT_SMOKE_OUTPUT_DIR) --build-dir $(BUILD_DIR) --skip-build --srs-cache-dir $(GBDT_SRS_CACHE_DIR) --initial-srs-deg1-plus-1 65536 --initial-srs-deg2-plus-1 65536 --jobs 1
HISTOGRAM_SMOKE_ENV ?= BUILD_DIR=$(BUILD_DIR) JOBS=1 HISTOGRAM_CASES="128:64 256:128" ROUTE=$(HISTOGRAM_ROUTE) REPEATS=1 SRS_CACHE_DIR=$(HISTOGRAM_SMOKE_SRS_CACHE_DIR) SKIP_BUILD=1
HISTOGRAM_ENV ?= BUILD_DIR=$(HISTOGRAM_BUILD_DIR) JOBS=1 ROUTE=$(HISTOGRAM_ROUTE) SRS_CACHE_DIR=$(HISTOGRAM_SRS_CACHE_DIR) SKIP_BUILD=1
SOURCE_REVISION_COMMAND = python3 tools/python_trace_export/source_revision.py --repo-root .
ARTIFACT_OUTPUT_DIR ?= out/artifact
ARTIFACT_NAME ?= ZK-GBDT-Code
ARTIFACT_ZIP ?= $(ARTIFACT_OUTPUT_DIR)/$(ARTIFACT_NAME).zip
ARTIFACT_PREFIX ?= $(ARTIFACT_NAME)/

help:
	@echo "Usage:"
	@echo "  make doctor"
	@echo "  make configure"
	@echo "  make build"
	@echo "  make test"
	@echo "  make datasets"
	@echo "  make srs"
	@echo "  make benchmark"
	@echo "  make benchmark-practical"
	@echo "  make benchmark-histogram"
	@echo "  make benchmark-paper"
	@echo "  make benchmark-paper-summary"
	@echo "  make benchmark-smoke"
	@echo "  make artifact"
	@echo ""
	@echo "Benchmark entry points:"
	@echo "  make benchmark                  # paper scaling: Dry Bean + Optdigit*; resumable, no timeout"
	@echo "  make benchmark-rerun-terrae-training # rerun only Terrae training and reuse all other stages"
	@echo "  make benchmark-rerun-certxgb-training # rerun only CertXGB training and reuse all other stages"
	@echo "  make benchmark-rerun-certxgb  # rerun CertXGB training + inference and resume the full plan"
	@echo "  make benchmark-rerun-certxgb-inference # rerun only CertXGB inference and resume the full plan"
	@echo "  make benchmark-practical        # practical N=1024,d=3,T=1/2/4/8 plan"
	@echo "  make benchmark-histogram        # histogram microbenchmark"
	@echo "  make benchmark-paper            # run every paper benchmark and assemble outputs"
	@echo "  make benchmark-paper-summary    # assemble existing paper benchmark outputs only"
	@echo "  make benchmark-smoke            # small GBDT + histogram installation check"
	@echo "  make benchmark-histogram-smoke  # small histogram primitive smoke run"
	@echo ""
	@echo "Optional variables:"
	@echo "  CMAKE_CONFIGURE_ARGS=...   extra args for cmake configure"
	@echo "  TERRAE_MCL_SOURCE_DIR=...  local herumi/mcl source tree; bypasses the Git fetch"
	@echo "  CTEST_ARGS=...               extra args for ctest in make test"
	@echo "  SRS_CACHE_DIR=...           reusable benchmark SRS cache root"
	@echo "  SRS_PATH=...                SRS path for make srs"
	@echo "  SRS_DEG1_PLUS_1=...         SRS G1 size (default 131072)"
	@echo "  SRS_DEG2_PLUS_1=...         SRS G2 size (default 131072)"
	@echo "  BENCHMARK_BUILD_DIR=...     release build used by make benchmark"
	@echo "  BENCHMARK_OUTPUT_DIR=...    resumable full benchmark output directory"
	@echo "  BENCHMARK_CASES=...         full benchmark case-plan JSON"
	@echo "  BENCHMARK_ARGS=...          override complete full benchmark runner arguments"
	@echo "  BENCHMARK_RERUN_STAGES=...  selectively rerun checkpointed stages (space-separated)"
	@echo "  BENCHMARK_CASE_IDS=...      run only selected case ids (space-separated)"
	@echo "  BENCHMARK_MEMORY_GUARD_ENV=...  env configuring the 180-GiB RSS and OOM guards"
	@echo "  PRACTICAL_BENCHMARK_OUTPUT_DIR=...  resumable practical-setting output directory"
	@echo "  PRACTICAL_BENCHMARK_CASES=...       practical-setting case-plan JSON"
	@echo "  PRACTICAL_BENCHMARK_ARGS=...        override practical benchmark runner arguments"
	@echo "  PAPER_BENCHMARK_OUTPUT_DIR=...      assembled paper-summary directory"
	@echo "  GBDT_SMOKE_OUTPUT_DIR=...   smoke GBDT report directory"
	@echo "  GBDT_SMOKE_ARGS=...         args passed to run_benchmark_parameter_plan.py"
	@echo "  HISTOGRAM_SMOKE_OUTPUT_DIR=...  histogram smoke report directory"
	@echo "  HISTOGRAM_OUTPUT_DIR=...    histogram full report directory"
	@echo "  HISTOGRAM_BUILD_DIR=...     Release build used by the full histogram benchmark"
	@echo "  HISTOGRAM_SMOKE_ENV=...     env assignments for tools/run_histogram_scale_benchmark.sh"
	@echo "  HISTOGRAM_ENV=...           env assignments for tools/run_histogram_scale_benchmark.sh"
	@echo "  ARTIFACT_ZIP=...            tracked-files-only artifact archive output"

doctor:
	bash tools/doctor.sh

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DBUILD_TESTING=OFF $(TERRAE_MCL_CMAKE_ARG) $(CMAKE_CONFIGURE_ARGS)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -- -j4

test:
	$(CMAKE) -S . -B $(BUILD_DIR) -DBUILD_TESTING=ON $(TERRAE_MCL_CMAKE_ARG) $(CMAKE_CONFIGURE_ARGS)
	$(CMAKE) --build $(BUILD_DIR) --target test_all -- -j4
	$(CTEST) --test-dir $(BUILD_DIR) $(CTEST_ARGS)

datasets:
	DATASET_SET=all bash tools/python_trace_export/download_reference_datasets.sh

datasets-smoke:
	DATASET_SET=smoke bash tools/python_trace_export/download_reference_datasets.sh

srs: configure
	$(CMAKE) --build $(BUILD_DIR) --target kzg_setup_helper -- -j4
	@echo "==> generating SRS at $(SRS_PATH)"
	./$(BUILD_DIR)/examples/kzg_setup_helper $(SRS_PATH) $(SRS_DEG1_PLUS_1) $(SRS_DEG2_PLUS_1)

benchmark:
	@mkdir -p $(BENCHMARK_OUTPUT_DIR) $(GBDT_SRS_CACHE_DIR)
	@/bin/bash -o pipefail -c '{ \
		echo "[benchmark] started_at=$$(date --iso-8601=seconds)"; \
		echo "[benchmark] source_revision=$$($(SOURCE_REVISION_COMMAND))"; \
		echo "[benchmark] command=$(BENCHMARK_MEMORY_GUARD_ENV) python3 tools/python_trace_export/run_benchmark_parameter_plan.py $(BENCHMARK_ARGS)"; \
		status=0; \
		DATASET_SET=all bash tools/python_trace_export/download_reference_datasets.sh || status=$$?; \
		if [[ $$status -eq 0 ]]; then $(BENCHMARK_MEMORY_GUARD_ENV) python3 tools/python_trace_export/run_benchmark_parameter_plan.py $(BENCHMARK_ARGS) || status=$$?; fi; \
		echo "[benchmark] finished_at=$$(date --iso-8601=seconds) status=$$status"; \
		exit $$status; \
	} 2>&1 | tee -a "$(BENCHMARK_LOG)"'

benchmark-rerun-terrae-training:
	$(MAKE) benchmark BENCHMARK_OUTPUT_DIR="$(BENCHMARK_OUTPUT_DIR)" BENCHMARK_RERUN_STAGES=terrae-training

benchmark-rerun-certxgb-training:
	$(MAKE) benchmark BENCHMARK_OUTPUT_DIR="$(BENCHMARK_OUTPUT_DIR)" BENCHMARK_RERUN_STAGES=zkboost-full-training

benchmark-rerun-certxgb:
	$(MAKE) benchmark BENCHMARK_OUTPUT_DIR="$(BENCHMARK_OUTPUT_DIR)" BENCHMARK_RERUN_STAGES="zkboost-full-training zkboost-inference"

benchmark-rerun-certxgb-inference:
	$(MAKE) benchmark BENCHMARK_OUTPUT_DIR="$(BENCHMARK_OUTPUT_DIR)" BENCHMARK_RERUN_STAGES=zkboost-inference

benchmark-practical:
	@mkdir -p $(PRACTICAL_BENCHMARK_OUTPUT_DIR) $(GBDT_SRS_CACHE_DIR)
	@/bin/bash -o pipefail -c '{ \
		echo "[benchmark-practical] started_at=$$(date --iso-8601=seconds)"; \
		echo "[benchmark-practical] source_revision=$$($(SOURCE_REVISION_COMMAND))"; \
		echo "[benchmark-practical] trainer_parameters=learning_rate=4/1 tau_q=1"; \
		echo "[benchmark-practical] command=$(BENCHMARK_MEMORY_GUARD_ENV) python3 tools/python_trace_export/run_benchmark_parameter_plan.py $(PRACTICAL_BENCHMARK_ARGS)"; \
		status=0; \
		DATASET_SET=all bash tools/python_trace_export/download_reference_datasets.sh || status=$$?; \
		if [[ $$status -eq 0 ]]; then $(BENCHMARK_MEMORY_GUARD_ENV) python3 tools/python_trace_export/run_benchmark_parameter_plan.py $(PRACTICAL_BENCHMARK_ARGS) || status=$$?; fi; \
		echo "[benchmark-practical] finished_at=$$(date --iso-8601=seconds) status=$$status"; \
		exit $$status; \
	} 2>&1 | tee -a "$(PRACTICAL_BENCHMARK_LOG)"'

benchmark-paper:
	$(MAKE) benchmark
	$(MAKE) benchmark-practical
	$(MAKE) benchmark-histogram
	$(MAKE) benchmark-paper-summary

benchmark-paper-summary:
	python3 tools/python_trace_export/assemble_paper_benchmark_summary.py \
		--scaling-dir "$(BENCHMARK_OUTPUT_DIR)" \
		--practical-dir "$(PRACTICAL_BENCHMARK_OUTPUT_DIR)" \
		--histogram-dir "$(HISTOGRAM_OUTPUT_DIR)" \
		--output-dir "$(PAPER_BENCHMARK_OUTPUT_DIR)"

benchmark-smoke: build datasets-smoke
	@mkdir -p $(GBDT_SRS_CACHE_DIR)
	python3 tools/python_trace_export/run_benchmark_parameter_plan.py $(GBDT_SMOKE_ARGS)
	$(MAKE) benchmark-histogram-smoke-run
	@echo "Smoke benchmarks complete."

benchmark-histogram-smoke: build
	$(MAKE) benchmark-histogram-smoke-run

benchmark-histogram-smoke-run:
	@mkdir -p $(HISTOGRAM_SMOKE_OUTPUT_DIR) $(HISTOGRAM_SMOKE_SRS_CACHE_DIR)
	$(HISTOGRAM_SMOKE_ENV) bash tools/run_histogram_scale_benchmark.sh > $(HISTOGRAM_SMOKE_OUTPUT_DIR)/histogram.tsv 2> $(HISTOGRAM_SMOKE_OUTPUT_DIR)/run.log
	python3 tools/summarize_histogram_benchmark.py --input-tsv $(HISTOGRAM_SMOKE_OUTPUT_DIR)/histogram.tsv --log $(HISTOGRAM_SMOKE_OUTPUT_DIR)/run.log --output-dir $(HISTOGRAM_SMOKE_OUTPUT_DIR)
	@echo "Histogram smoke summary: $(HISTOGRAM_SMOKE_OUTPUT_DIR)/summary.csv"
	@echo "Histogram smoke summary: $(HISTOGRAM_SMOKE_OUTPUT_DIR)/summary.md"

benchmark-histogram: benchmark-histogram-build
	$(MAKE) benchmark-histogram-run

benchmark-histogram-build:
	$(CMAKE) -S . -B $(HISTOGRAM_BUILD_DIR) -DBUILD_TESTING=OFF $(TERRAE_MCL_CMAKE_ARG) $(CMAKE_CONFIGURE_ARGS) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(HISTOGRAM_BUILD_DIR) --target benchmark_histogram_primitives -- -j4

benchmark-histogram-run:
	@mkdir -p $(HISTOGRAM_OUTPUT_DIR) $(HISTOGRAM_SRS_CACHE_DIR)
	$(HISTOGRAM_ENV) bash tools/run_histogram_scale_benchmark.sh > $(HISTOGRAM_OUTPUT_DIR)/histogram.tsv 2> $(HISTOGRAM_OUTPUT_DIR)/run.log
	python3 tools/summarize_histogram_benchmark.py --input-tsv $(HISTOGRAM_OUTPUT_DIR)/histogram.tsv --log $(HISTOGRAM_OUTPUT_DIR)/run.log --output-dir $(HISTOGRAM_OUTPUT_DIR)
	@echo "Histogram summary: $(HISTOGRAM_OUTPUT_DIR)/summary.csv"
	@echo "Histogram summary: $(HISTOGRAM_OUTPUT_DIR)/summary.md"

artifact:
	@test -z "$$(git status --porcelain --untracked-files=all)" || { \
		echo "error: commit or remove non-ignored changes before packaging" >&2; \
		git status --short; \
		exit 1; \
	}
	@mkdir -p "$(ARTIFACT_OUTPUT_DIR)"
	git archive --format=zip --prefix="$(ARTIFACT_PREFIX)" --output="$(ARTIFACT_ZIP)" HEAD -- . ':(exclude).gitignore'
	python3 -c 'import sys, zipfile; path = sys.argv[1]; archive = zipfile.ZipFile(path, "a"); archive.comment = b""; archive.close()' "$(ARTIFACT_ZIP)"
	@sha256sum "$(ARTIFACT_ZIP)"
