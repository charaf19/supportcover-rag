# Commands to run the project

Verified on 2026-08-29 from PowerShell with Python 3.11. All commands use this working directory unless a section says otherwise:

```powershell
Set-Location 'C:\Users\pc\Desktop\PROJECTS\Research\supportcover-rag'
```

Status labels mean:

- **VERIFIED**: executed locally without an expensive model or dataset run.
- **SUPPORTED, NOT EXECUTED**: backed by the current CLI/config but intentionally not run because it downloads data/models or performs generation.
- **BLOCKED**: the command surface or required scientific assets do not yet exist; do not run it as a paper-grade study.
- **HISTORICAL ONLY**: supported for interpreting or recreating the old workflow, not valid for new final claims.

## Environment creation and dependency installation

### Create and activate a virtual environment

- Status: SUPPORTED, NOT EXECUTED.
- Working directory: repository root above.
- Inputs: Python 3.11 available as `py -3.11`.
- Outputs: `.venv/`.
- Cost: cheap; no model execution.
- Phase availability: any phase.

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install the package and test dependencies

- Status: SUPPORTED, NOT EXECUTED.
- Working directory: repository root.
- Inputs: `pyproject.toml`, network/package-index access.
- Outputs: editable package installation in the active environment.
- Cost: moderate download/install cost.
- Phase availability: any phase.

```powershell
python -m pip install -e '.[dev]'
```

`pyproject.toml` does not declare PyTorch. A hardware-appropriate PyTorch build is an additional prerequisite for Transformers generation. Do not guess a CUDA/XPU/CPU install command; select it for the target machine before model runs. `requirements.txt` and `pyproject.toml` are not currently identical dependency specifications; see the audit.

### Repository-local command fallback without installation

- Status: VERIFIED.
- Working directory: repository root.
- Inputs: current source tree and already installed runtime dependencies.
- Outputs: sets `PYTHONPATH` for the current PowerShell process.
- Cost: cheap.
- Phase availability: inspection and tests; editable installation remains preferred for recorded experiments.

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
```

## CLI help

- Status: VERIFIED with the repository-local fallback.
- Working directory: repository root.
- Inputs: installed package, or `PYTHONPATH` set as above.
- Outputs: command help on stdout; no files.
- Cost: cheap.
- Phase availability: any phase.

```powershell
python -m supportcover_rag --help
python -m supportcover_rag run --help
```

After editable installation, the equivalent console command is:

```powershell
supportcover-rag --help
```

## Tests

### Minimal end-to-end runner smoke test

- Status: VERIFIED as part of the full suite.
- Working directory: repository root.
- Inputs: synthetic in-test Hotpot-style record; echo generator; no dataset or model.
- Outputs: temporary run layout managed by pytest and automatically cleaned up.
- Cost: cheap.
- Phase availability: any phase.

```powershell
python -m pytest tests/test_experiment_outputs.py::test_runner_writes_new_output_layout_and_registry -q
```

### Targeted Phase 0 tests

- Status: VERIFIED.
- Working directory: repository root.
- Inputs: source tree and pytest.
- Outputs: test report only.
- Cost: cheap.
- Phase availability: Phase 0 and later.

```powershell
python -m pytest tests/test_experiment_outputs.py -q
```

### Full test suite

- Status: VERIFIED at the Phase 0 gate.
- Working directory: repository root.
- Inputs: source tree and pytest.
- Outputs: test report only.
- Cost: cheap; all tests are synthetic/unit-level in the current checkout.
- Phase availability: every phase gate.

```powershell
python -m pytest
```

### Targeted Phase 2 methods and contracts

- Status: VERIFIED.
- Working directory: repository root.
- Inputs: synthetic candidates, paragraphs, and external-compressor doubles; no dataset or model.
- Outputs: test report only.
- Cost: cheap.
- Phase availability: Phase 2 and later.

```powershell
python -m pytest tests/test_packing_baselines.py tests/test_paragraph_support.py tests/test_supportcover_ablation.py -q
```

## Data acquisition and preprocessing

### Acquire configured HotpotQA data

- Status: VERIFIED during Phase 1; do not repeat unless reconstructing the ignored local dataset.
- Working directory: repository root.
- Inputs: `configs/default.yaml`, network access, Hugging Face dataset availability.
- Outputs: `data/raw/train.jsonl` and `data/raw/validation.jsonl`.
- Cost: network-, disk-, and time-intensive; no model generation.
- Phase availability: Phase 1 preparation and later.

```powershell
python -m supportcover_rag acquire-data --config configs/default.yaml
```

### Preprocess acquired data

- Status: VERIFIED during Phase 1.
- Working directory: repository root.
- Inputs: `data/raw/*.jsonl` selected by the config.
- Outputs: `data/processed/train.jsonl` and `data/processed/validation.jsonl`.
- Cost: moderate CPU/disk cost; no model generation.
- Phase availability: after acquisition.

```powershell
python -m supportcover_rag preprocess --config configs/default.yaml
```

## Explicit scientific splits

### Frozen development split creation record

- Status: EXECUTED AND PERMANENTLY FROZEN. Do not rerun, resample, reshuffle, edit, or replace this manifest.
- Working directory: repository root.
- Inputs: `data/processed/train.jsonl` containing 90,447 HotpotQA training examples.
- Outputs: self-contained `data/splits/development_ids.json` containing ordered IDs, count, seed, role, stratification dimensions, and SHA-256.
- Cost: moderate JSONL read; no model generation.
- Phase availability: historical command record only; the resulting IDs are immutable.

```powershell
python -m supportcover_rag create-split --processed data/processed/train.jsonl --output data/splits/development_ids.json --role development --sample-size 2000 --seed 42 --stratify-by type,level
```

Frozen protocol: HotpotQA train, N=2,000, seed 42, stratified by `type,level`, ordered-ID SHA-256 `0e02afdcdff360d26725abe9c197a457dcbe76c92aa54338cdc146806b9ed7c6`.

### Frozen full final population creation record

- Status: EXECUTED AND PERMANENTLY FROZEN. Do not rerun, resample, reshuffle, edit, replace, or inspect this population for tuning.
- Working directory: repository root.
- Inputs: `data/processed/validation.jsonl` containing all 7,405 HotpotQA validation examples.
- Outputs: self-contained `data/splits/final_ids.json` with every validation ID in source order and its SHA-256.
- Cost: moderate JSONL read; no model generation.
- Phase availability: historical command record only; the resulting IDs are immutable.

```powershell
python -m supportcover_rag create-split --processed data/processed/validation.jsonl --output data/splits/final_ids.json --role final --seed 42
```

Frozen protocol: complete HotpotQA validation population, N=7,405, ordered-ID SHA-256 `fc5c4bbd3b2a0304803f118cc098eec9d78521ac7f769877774239f52a4ecf6c`.

### Validate isolation and hashes

- Status: VERIFIED on the real frozen manifests.
- Working directory: repository root.
- Inputs: development and final split manifests.
- Outputs: `data/splits/split_validation.json` and a PASS message; overlap, duplicates, missing/malformed IDs, or embedded hash mismatches fail the command.
- Cost: cheap.
- Phase availability: Phase 1 and every later protocol gate.

```powershell
python -m supportcover_rag validate-splits --development data/splits/development_ids.json --final data/splits/final_ids.json --output data/splits/split_validation.json
```

## Development experiments

### Historical small validation run

- Status: HISTORICAL ONLY; supported but not executed.
- Working directory: repository root.
- Inputs: `data/processed/validation.jsonl`, configured Qwen model, `configs/phase1_main.yaml`.
- Outputs: new run folders under `outputs/debug/` plus `outputs/registry/`.
- Cost: expensive model generation for the first 32 validation examples and three methods.
- Phase availability: debug only. It is not a development population and must not be used for tuning or final evidence.

```powershell
python -m supportcover_rag run --config configs/phase1_main.yaml --family debug
```

### Publication-grade development tuning

- Status: INFRASTRUCTURE READY; scientific execution remains pending the project-environment PyTorch install and non-final model smoke test.
- Working directory: repository root.
- Required inputs: frozen `data/splits/development_ids.json`, `data/processed/train.jsonl`, `configs/phase3_sensitivity.yaml`, and the configured development model.
- Expected outputs: generator-free packing rows and summary under `outputs/development/phase3/packing/`, shortlisted development generation runs under `outputs/development/phase3/runs/`, a completed decision record, `configs/final_frozen.yaml`, and `configs/frozen/final_manifest.json`.
- Cost: the packing screen is CPU/tokenizer-only; only shortlisted comparisons use model generation.
- Phase availability: Phase 3 only, after Phase 1 and Phase 2 gates.

The Phase-3 config is fail-closed on `role: development`, `experiments.split: train`, the full 2,000-ID count, the frozen development SHA-256, and `runtime.limit: null`. The packing command evaluates exactly 20 OFAT SupportCover settings, four MMR lambdas, and five clean component variants. It does not instantiate the answer generator. The legacy conflated `no_coverage` variant is excluded.

### Phase-3 execution order

#### A. Environment/GPU verification

```powershell
.\.venv\Scripts\python.exe -m supportcover_rag check-environment --output outputs/development/phase3/environment.preinstall.json
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
```

#### B. PyTorch installation if required

The repository `.venv` currently has Transformers but no PyTorch. This Windows machine has an NVIDIA GeForce RTX 5060 Laptop GPU and supports CUDA; the verified CUDA 13.0 wheel command is:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
.\.venv\Scripts\python.exe -m supportcover_rag check-environment --output outputs/development/phase3/environment.json
```

Do not continue unless `torch.cuda.is_available()` is `True` and the recorded backend is `cuda`.

#### C. Cheap Phase-3 unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase3.py tests/test_generation.py tests/test_packing_baselines.py tests/test_supportcover_ablation.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

#### D. Tiny non-scientific smoke test

This reads only two leading training examples, writes debug artifacts, and is not admissible tuning evidence.

```powershell
.\.venv\Scripts\python.exe -m supportcover_rag run --config configs/phase3_smoke.yaml --family debug --notes "NON-SCIENTIFIC CUDA/model smoke; never use for tuning"
```

#### E. Packing-level sensitivity on 2,000 development examples

```powershell
.\.venv\Scripts\python.exe -m supportcover_rag run-development-packing --config configs/phase3_sensitivity.yaml
```

The resulting `packing_summary.csv` is explicitly packing-only: it contains support EM/precision/recall/F1, coverage at budget, and evidence-token measurements, but no answer-generation metric.

#### F. Shortlist configurations

```powershell
Copy-Item configs/phase3_shortlist.template.json outputs/development/phase3/shortlist.json
```

Fill the shortlist only from the complete development packing summary. Retain the untuned base and non-dominated candidates; use support F1, support recall, coverage at budget, and evidence tokens, with lower numeric parameter value as the exact-tie breaker. Record the packing-manifest path and never combine independent OFAT winners into an untested Cartesian configuration.

#### G. Generation-based development validation where scientifically necessary

For each shortlisted SupportCover coefficient vector, run this command with the four values copied exactly from `shortlist.json`:

```powershell
$Beta = [double](Read-Host "beta_coverage from one recorded shortlist candidate")
$Title = [double](Read-Host "title_bonus from that same candidate")
$Delta = [double](Read-Host "delta_token_cost from that same candidate")
$Gamma = [double](Read-Host "gamma_redundancy from that same candidate")
.\.venv\Scripts\python.exe -m supportcover_rag run --config configs/phase3_sensitivity.yaml --family baseline --methods supportcover --beta-coverage $Beta --title-bonus $Title --delta-token-cost $Delta --gamma-redundancy $Gamma --notes "Phase 3 shortlisted SupportCover development validation"
```

`$Beta`, `$Title`, `$Delta`, and `$Gamma` must be assigned from a single recorded OFAT candidate, not independently optimized into a Cartesian combination.

#### H. MMR lambda selection

Packing evaluates all four preregistered lambdas. Run answer generation only for the MMR lambdas retained in `shortlist.json`:

```powershell
$MmrShortlist = (Read-Host "Comma-separated MMR lambdas retained in shortlist.json").Split(',') | ForEach-Object { [double]$_.Trim() }
$MmrShortlist | ForEach-Object { .\.venv\Scripts\python.exe -m supportcover_rag run --config configs/phase3_sensitivity.yaml --family baseline --methods mmr_sentence --mmr-lambda $_ --notes "Phase 3 shortlisted MMR development validation" }
```

#### I. Component ablation

After selecting the development-only full SupportCover coefficient vector, assign its four tuned values and execute the five clean variants. The checked-in config excludes legacy `no_coverage`.

```powershell
$Beta = [double](Read-Host "selected beta_coverage")
$Title = [double](Read-Host "selected title_bonus")
$Delta = [double](Read-Host "selected delta_token_cost")
$Gamma = [double](Read-Host "selected gamma_redundancy")
.\.venv\Scripts\python.exe -m supportcover_rag run-ablations --config configs/phase3_sensitivity.yaml --family ablation_component --beta-coverage $Beta --title-bonus $Title --delta-token-cost $Delta --gamma-redundancy $Gamma --notes "Phase 3 clean component ablation on development"
```

#### J. Select final configuration using development evidence only

```powershell
Copy-Item configs/phase3_decision.template.json outputs/development/phase3/decision.json
```

Complete every non-null field using only packing and generation artifacts under `outputs/development/phase3/`. The freeze command rejects out-of-grid selections, missing evidence classes, and evidence paths outside that development-only directory.

#### K. Produce `configs/final_frozen.yaml`

#### L. Produce frozen manifest and SHA256

```powershell
.\.venv\Scripts\python.exe -m supportcover_rag freeze-development --config configs/phase3_sensitivity.yaml --decision outputs/development/phase3/decision.json --final-ids data/splits/final_ids.json --final-config configs/final_frozen.yaml --manifest configs/frozen/final_manifest.json
```

The command creates both K and L in one validated operation only after all required development evidence exists. It records SHA-256 hashes for the decision and evidence artifacts, the selected coefficients, selected MMR lambda, the development split SHA, and the final split SHA.

#### M. Verify final population has never been used

```powershell
Get-ChildItem outputs/development/phase3 -Recurse -File -Include *.yaml,*.json | Select-String -Pattern "data/processed/validation.jsonl|data/splits/final_ids.json"
.\.venv\Scripts\python.exe -m supportcover_rag validate-splits --development data/splits/development_ids.json --final data/splits/final_ids.json --output data/splits/split_validation.json
git diff --check
git status --short
```

The first command should find no tuning input reference; the freeze manifest may contain the final split SHA and final manifest path, but no final prediction or metric. Do not execute the frozen final config during Phase 3.

## Final experiments

### Main unseen study

- Status: BLOCKED; MUST NOT RUN before the Phase 3 freeze and Phase 4 statistics gate.
- Working directory: repository root.
- Required inputs: `data/processed/validation.jsonl`, `data/splits/final_ids.json`, `configs/frozen/final_manifest.json`, resolved non-null frozen settings, the selected external compressor adapter, model files, and suitable hardware.
- Expected outputs: per-method raw predictions and resolved metadata under `outputs/main/`, followed by final aggregate/statistics artifacts.
- Cost: very expensive; target population is at least 1,000 generated examples and preferably 2,000+ or full validation.
- Phase availability: Phase 5 only.

The current CLI shape is:

```powershell
python -m supportcover_rag run --config configs/final_main.yaml
```

This command is documented for future verification only. The current config intentionally contains unresolved `null` frozen fields and references missing assets, so it is not executable as a valid final study.

### Robustness commands

- Status: BLOCKED until the main-study decision gate passes.
- Working directory: repository root.
- Inputs: the same frozen final assets plus each robustness config's model/dataset prerequisites.
- Outputs: robustness run directories and later aggregate CSVs.
- Cost: very expensive model generation.
- Phase availability: Phase 6 only.

```powershell
python -m supportcover_rag run-ablations --config configs/final_budget.yaml --family ablation_budget
python -m supportcover_rag run-robustness --config configs/final_models.yaml
python -m supportcover_rag run --config configs/final_cross_dataset.yaml
```

The budget command surface exists, but frozen `null` fields must be resolved before it is safe. The cross-dataset CLI does not yet route through the 2Wiki adapter and is therefore not a valid cross-dataset run in the current phase.

## Statistics

- Status: BLOCKED as an artifact-producing command.
- Working directory: repository root.
- Inputs when implemented: aligned per-example prediction JSONL files and resolved configs for every compared method.
- Expected outputs: `outputs/final/main_results.csv` and `outputs/final/main_statistics.csv`.
- Cost: moderate CPU cost; no model generation.
- Phase availability: implementation in Phase 4, execution after Phase 5 predictions.

The library implementation in `supportcover_rag.statistics` is tested, but there is no CLI that reads prediction artifacts and writes the standard result files. The only currently verified statistics command is its unit test:

```powershell
python -m pytest tests/test_statistics.py -q
```

## Systems benchmarking

- Status: BLOCKED as an end-to-end benchmark command.
- Working directory: repository root.
- Inputs when implemented: frozen methods, benchmark population, model/runtime, and environment descriptor.
- Expected outputs: raw latency, raw memory, summary, and environment artifacts under `outputs/final/`.
- Cost: expensive; at least 5 warmups and 30 measured repetitions per benchmarked stage.
- Phase availability: Phase 8 only.

`benchmark.py` and `resource_monitor.py` are unit-tested library components, but no CLI currently performs the authoritative benchmark. Verify the components with:

```powershell
python -m pytest tests/test_resource_monitor.py -q
```

The following existing command is historical summary generation, not an authoritative benchmark:

```powershell
python -m supportcover_rag run-systems-summary --config configs/phase8_efficiency_summary.yaml
```

It is currently blocked by absent EXP030/EXP031 source runs and must not be used for final P50/P95 or peak-memory claims.

## Error analysis

- Status: HISTORICAL ONLY and currently blocked by missing source runs.
- Working directory: repository root.
- Inputs: the EXP030/EXP031 directories named in `configs/phase7_error_analysis.yaml`.
- Outputs: CSV/Markdown files under `outputs/error_analysis/`.
- Cost: cheap once predictions exist.
- Phase availability: historical reproduction; the paired blinded Phase 9 workflow still needs canonical wiring.

```powershell
python -m supportcover_rag run-error-analysis --config configs/phase7_error_analysis.yaml
```

## Reproducibility verification

- Status: BLOCKED as a final-study command.
- Working directory: repository root.
- Inputs when implemented: split manifests, frozen hashes, resolved configs, predictions, metrics, and required artifact paths.
- Expected output: `outputs/final/reproducibility_check.json` with structured PASS/FAIL records.
- Cost: cheap to moderate; no model generation.
- Phase availability: Phase 9.

`supportcover_rag.reproducibility.verify_reproducibility()` currently validates already-loaded mappings but performs no file discovery and has no CLI. No final verification command should be claimed until Phase 9 completes.

## Phase-gate inspection

- Status: VERIFIED in Phase 0.
- Working directory: repository root.
- Inputs: Git worktree.
- Outputs: review information only; no files changed.
- Cost: cheap.
- Phase availability: every phase gate.

```powershell
git status --short
git diff --stat
git diff --check
git diff
```
