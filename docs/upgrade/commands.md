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

## Data acquisition and preprocessing

### Acquire configured HotpotQA data

- Status: SUPPORTED, NOT EXECUTED.
- Working directory: repository root.
- Inputs: `configs/default.yaml`, network access, Hugging Face dataset availability.
- Outputs: `data/raw/train.jsonl` and `data/raw/validation.jsonl`.
- Cost: network-, disk-, and time-intensive; no model generation.
- Phase availability: Phase 1 preparation and later.

```powershell
python -m supportcover_rag acquire-data --config configs/default.yaml
```

### Preprocess acquired data

- Status: SUPPORTED, NOT EXECUTED because raw data is absent.
- Working directory: repository root.
- Inputs: `data/raw/*.jsonl` selected by the config.
- Outputs: `data/processed/train.jsonl` and `data/processed/validation.jsonl`.
- Cost: moderate CPU/disk cost; no model generation.
- Phase availability: after acquisition.

```powershell
python -m supportcover_rag preprocess --config configs/default.yaml
```

## Explicit scientific splits

### Create a deterministic development split

- Status: VERIFIED on synthetic processed data; real population not generated because `data/processed/train.jsonl` is absent.
- Working directory: repository root.
- Inputs: processed HotpotQA training JSONL and a protocol-approved sample size.
- Outputs: self-contained `data/splits/development_ids.json` containing ordered IDs, count, seed, role, stratification dimensions, and SHA-256.
- Cost: moderate JSONL read; no model generation.
- Phase availability: Phase 1 after the development sample size is fixed.

```powershell
python -m supportcover_rag create-split --processed data/processed/train.jsonl --output data/splits/development_ids.json --role development --sample-size <APPROVED_DEV_SIZE> --seed 42 --stratify-by type,level
```

`<APPROVED_DEV_SIZE>` is intentionally unresolved: selecting it changes the experimental protocol and must not be guessed after results are observed.

### Create the full final validation population

- Status: VERIFIED on synthetic processed data; real population not generated because `data/processed/validation.jsonl` is absent.
- Working directory: repository root.
- Inputs: processed HotpotQA validation JSONL.
- Outputs: self-contained `data/splits/final_ids.json` with every validation ID in source order and its SHA-256.
- Cost: moderate JSONL read; no model generation.
- Phase availability: Phase 1. Omitting `--sample-size` selects the full source population.

```powershell
python -m supportcover_rag create-split --processed data/processed/validation.jsonl --output data/splits/final_ids.json --role final --seed 42
```

### Validate isolation and hashes

- Status: VERIFIED on synthetic manifests.
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

- Status: BLOCKED in Phase 0.
- Working directory: repository root.
- Required inputs: deterministic development IDs drawn from HotpotQA training data, processed training data, complete sensitivity runner, model, and frozen prompt/decoding candidates.
- Expected outputs: per-example development predictions, OFAT results, component ablation, MMR selection record, and a freeze manifest.
- Cost: expensive model generation.
- Phase availability: Phase 3 only, after Phase 1 and Phase 2 gates.

There is no verified publication-grade development command yet. `configs/phase3_sensitivity.yaml` declares a development ID file but points `experiments.split` at validation and no CLI command consumes its sensitivity grid. Running it now would not satisfy the protocol.

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
