# SupportCover-RAG

SupportCover-RAG is a compact, research-oriented codebase for budgeted support-coverage compression in on-device multi-hop question answering with small language models.

## Research Scope

This repository is organized around one narrow setting:

- task: multi-hop QA
- dataset: HotpotQA distractor
- method family: SupportCover-RAG
- main question: does SupportCover beat simpler packing baselines under the same frozen setup?

Keep the project lean. Do not add unrelated benchmark suites, extra tracking tools, prompt sweeps, model sweeps, or new methods unless a later phase explicitly requires them.

## Phase 1

Phase 1 is the core baseline comparison.

- goal: compare `paragraph_topk`, `relevance_only`, and `supportcover`
- canonical config: `configs/phase1_main.yaml`
- frozen split: `validation`
- frozen retrieval depth: `5`
- frozen token budget: `160`
- frozen generator setup: `transformers` with `Qwen/Qwen3-4B-Instruct-2507`
- frozen decoding: deterministic, `max_new_tokens=12`
- default small run: `32` examples
- follow-up run: `100` examples on the same setup

Phase 1 stays in scope only if all three methods use the same model, prompt, token budget, retrieval depth, and evaluation setup.

## Repository Layout

```text
supportcover-rag/
|-- configs/
|   |-- default.yaml
|   `-- phase1_main.yaml
|-- src/supportcover_rag/
|   |-- cli.py
|   |-- config.py
|   |-- experiment_outputs.py
|   |-- generation.py
|   |-- io_utils.py
|   |-- logging_utils.py
|   |-- packing.py
|   |-- pipeline.py
|   |-- retrieval.py
|   `-- types.py
`-- tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For tests:

```bash
pip install -e .[dev]
pytest
```

Install PyTorch separately for generation. For Intel Arc / XPU, use the official PyTorch XPU wheel for your OS and Python 3.11 environment.

## Data Prep

Acquire and preprocess HotpotQA distractor once:

```bash
supportcover-rag acquire-data --config configs/default.yaml
supportcover-rag preprocess --config configs/default.yaml
```

## Phase 1 Commands

Run the canonical 32-example Phase 1 comparison:

```bash
supportcover-rag run --config configs/phase1_main.yaml --family baseline --notes phase1_main_32
```

Run the 100-example follow-up on the exact same setup:

```bash
supportcover-rag run --config configs/phase1_main.yaml --family baseline --limit 100 --notes phase1_main_100
```

Do not use `run-ablations` for Phase 1. Budget sweeps, depth sweeps, component ablations, robustness runs, and prompt changes belong to later phases.

## Outputs

New runs use a family-first layout under `outputs/`:

```text
outputs/
  registry/
  main/
  baseline/
  ablation_budget/
  ablation_depth/
  ablation_component/
  robustness/
  debug/
```

Each run folder is named:

```text
{experiment_id}_{method}_{model}_{split}_b{budget}_d{depth}_{variant}
```

Examples:

- `EXP001_paragraph_topk_qwen_val_b160_d5_full`
- `EXP002_relevance_only_qwen_val_b160_d5_full`
- `EXP003_supportcover_qwen_val_b160_d5_full`

Paper-grade runs use `EXP` ids. Debug runs use `DBG` ids and always go under `outputs/debug/`.

Each run folder contains:

- `config.resolved.yaml`
- `metrics.json`
- `predictions.jsonl`
- `summary.csv`
- `run.log`

Multi-method `run` invocations also write a comparison CSV to the family directory, for example:

- `outputs/baseline/EXP001_EXP003_comparison.csv`

The central registry lives at `outputs/registry/experiments.csv`.

## Identifying Phase 1 Baseline Runs

Phase 1 baseline runs are the runs that satisfy all of the following:

- config: `configs/phase1_main.yaml`
- family: `baseline`
- methods: `paragraph_topk`, `relevance_only`, `supportcover`
- split: `validation`
- token budget: `160`
- retrieval depth: `5`
- notes: `phase1_main_32` or `phase1_main_100`

This is the intended guardrail against scope drift.

## Metrics To Read First

Look at these first in the comparison summary:

- `answer_f1`
- `support_f1`
- `coverage_at_budget`
- `total_latency_ms`

Use `answer_em` as a stricter correctness check, and inspect `retrieval_latency_ms`, `packing_latency_ms`, and `generation_latency_ms` when a method wins or loses on speed.

## Current Phase 1 Result

The completed 32-example Phase 1 run is recorded under `outputs/baseline/` with experiment ids `EXP001` through `EXP003`. The phase-specific comparison artifacts are:

- `outputs/baseline/phase1_main_32_comparison.csv`
- `outputs/baseline/phase1_main_32_comparison.md`

## Next Phase

After Phase 1, the next phase is controlled ablation work on the same frozen setup:

- budget ablation
- retrieval-depth ablation
- SupportCover component ablation

Do not start those until the Phase 1 baseline comparison is locked and reviewed.
