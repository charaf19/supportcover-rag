# SupportCover-RAG

SupportCover-RAG is a compact, research-oriented codebase for budgeted support-coverage compression in on-device multi-hop question answering with small language models.

## Research Scope

This repository stays inside one fixed setting:

- task: multi-hop QA
- dataset: HotpotQA distractor
- split: `validation`
- method family: SupportCover-RAG
- model/backend: `transformers` + `Qwen/Qwen3-4B-Instruct-2507`
- prompt: frozen from Phase 1
- decoding: frozen from Phase 1
- canonical token budget: `160`
- canonical retrieval depth: `5`

Keep the project lean. Do not add prompt sweeps, model sweeps, robustness studies, notebooks, tracking frameworks, or unrelated benchmarks unless a later phase explicitly requires them.

## Experiment Flow

- Phase 1: main baseline comparison on `32` examples
- Phase 2: stability check on `100` examples
- Phase 3: token budget ablation
- Phase 4: retrieval depth ablation
- Phase 5: component ablation
- Phase 6: cross-model robustness
- Phase 7: error analysis
- Phase 8: efficiency and final systems summary
- Phase 9: final paper package and write-up assets

## Canonical Configs

- Phase 1: `configs/phase1_main.yaml`
- Phase 2: `configs/phase2_stability_100.yaml`
- Phase 3: `configs/phase3_budget_ablation.yaml`
- Phase 4: `configs/phase4_depth_ablation.yaml`
- Phase 5: `configs/phase5_component_ablation.yaml`
- Phase 6: `configs/phase6_model_robustness.yaml`
- Phase 7: `configs/phase7_error_analysis.yaml`
- Phase 8: `configs/phase8_efficiency_summary.yaml`

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

## Phase 1

Phase 1 asks whether SupportCover beats the simpler baselines under one frozen setup on a 32-example subset.

- config: `configs/phase1_main.yaml`
- methods: `paragraph_topk`, `relevance_only`, `supportcover`
- output family: `baseline`

Run it with:

```bash
supportcover-rag run --config configs/phase1_main.yaml --family baseline --notes phase1_main_32
```

Artifacts:

- `outputs/baseline/phase1_main_32_comparison.csv`
- `outputs/baseline/phase1_main_32_comparison.md`

## Phase 2

Phase 2 asks whether the Phase 1 ranking stays stable when the subset grows from `32` to `100` examples.

- config: `configs/phase2_stability_100.yaml`
- methods: `paragraph_topk`, `relevance_only`, `supportcover`
- output family: `baseline`
- only difference from Phase 1: `runtime.limit=100`

Run it with:

```bash
supportcover-rag run --config configs/phase2_stability_100.yaml --family baseline --notes phase2_stability_100
```

Artifacts:

- `outputs/baseline/phase2_stability_100_comparison.csv`
- `outputs/baseline/phase2_stability_100_comparison.md`
- `outputs/baseline/phase1_vs_phase2_stability.md`

Inspect `answer_f1`, `support_f1`, and `coverage_at_budget` first. Phase 3 should start only after the 100-example ranking is acceptable.

## Phase 3

Phase 3 is the token budget ablation. This is the central budget-sensitivity check for the SupportCover-RAG contribution.

- config: `configs/phase3_budget_ablation.yaml`
- methods: `relevance_only`, `supportcover`
- budgets: `96`, `128`, `160`, `192`
- output family: `ablation_budget`
- frozen from Phase 2: model, backend, prompt, decoding, retrieval depth, split, subset policy, and evaluation procedure
- only experimental variable: token budget

Run the full ablation with:

```bash
supportcover-rag run-ablations --config configs/phase3_budget_ablation.yaml --family ablation_budget --notes phase3_budget_ablation
```

Main artifacts:

- `outputs/ablation_budget/phase3_budget_ablation_summary.csv`
- `outputs/ablation_budget/phase3_budget_ablation_summary.md`
- `outputs/ablation_budget/phase3_budget_analysis.md`

## Phase 4

Phase 4 is the retrieval depth ablation. It tests whether `relevance_only` and `supportcover` behave differently as upstream retrieval becomes weaker or stronger under the same frozen setup.

- config: `configs/phase4_depth_ablation.yaml`
- methods: `relevance_only`, `supportcover`
- retrieval depths: `5`, `10`, `15`
- token budget: fixed at `160`
- output family: `ablation_depth`
- frozen from Phase 3: model, backend, prompt, decoding, token budget, split, subset policy, and evaluation procedure
- only experimental variable: retrieval depth

Run the full ablation with:

```bash
supportcover-rag run-ablations --config configs/phase4_depth_ablation.yaml --family ablation_depth --notes phase4_depth_ablation
```

Main artifacts:

- `outputs/ablation_depth/phase4_depth_ablation_summary.csv`
- `outputs/ablation_depth/phase4_depth_ablation_summary.md`
- `outputs/ablation_depth/phase4_depth_analysis.md`

## Phase 5

Phase 5 is the component ablation. This is the direct method-validation phase for the SupportCover scoring design, especially the coverage-aware contribution.

- config: `configs/phase5_component_ablation.yaml`
- output family: `ablation_component`
- frozen from Phase 4: model, backend, prompt, decoding, token budget, retrieval depth, split, subset policy, and evaluation procedure
- only experimental variable: SupportCover scoring variant

Variants included:

- `relevance_only`: baseline comparator
- `full`: standard SupportCover
- `no_coverage`: coverage and title-gain removed
- `no_redundancy`: redundancy penalty removed
- `no_token_penalty`: token-cost term removed while still enforcing the hard token budget

Run the full ablation with:

```bash
supportcover-rag run-ablations --config configs/phase5_component_ablation.yaml --family ablation_component --notes phase5_component_ablation
```

Phase 5 outputs are stored under `outputs/ablation_component/`.

Main artifacts:

- `outputs/ablation_component/phase5_component_ablation_summary.csv`
- `outputs/ablation_component/phase5_component_ablation_summary.md`
- `outputs/ablation_component/phase5_component_analysis.md`

How to read Phase 5:

- compare `full` against `no_coverage` first to evaluate the novelty claim around coverage-aware scoring
- compare `full` against `no_redundancy` and `no_token_penalty` to see which other scoring terms matter on the frozen setup
- use `answer_f1`, `support_f1`, and `coverage_at_budget` as the first readout

## Phase 6

Phase 6 is the cross-model robustness study. It asks whether SupportCover stays useful when the small generator model changes under the same frozen QA setup.

- config: `configs/phase6_model_robustness.yaml`
- output family: `robustness`
- frozen from Phase 5: dataset, split, prompt, decoding, token budget `160`, retrieval depth `5`, subset policy, and evaluation procedure
- only experimental variable: generator model

Models included:

- `tinyllama`: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `qwen`: `Qwen/Qwen3-4B-Instruct-2507`

Methods included:

- `relevance_only`
- `supportcover_final`

Canonical variant mapping:

- `supportcover_final = no_redundancy`
- This mapping is frozen from Phase 5 because `no_redundancy` was the strongest SupportCover variant on the fixed component-ablation setup.

Run the full study with:

```bash
supportcover-rag run-robustness --config configs/phase6_model_robustness.yaml --notes phase6_model_robustness
```

Phase 6 outputs are stored under `outputs/robustness/`.

Main artifacts:

- `outputs/robustness/phase6_model_robustness_summary.csv`
- `outputs/robustness/phase6_model_robustness_summary.md`
- `outputs/robustness/phase6_model_robustness_analysis.md`

How to read Phase 6:

- compare `supportcover_final` against `relevance_only` within each model, not across unrelated models
- inspect `support_f1` and `coverage_at_budget` first to see whether the method keeps its evidence-selection advantage
- then inspect `answer_f1` to see whether better packed evidence transfers into answer quality for each generator
- treat this as a compact robustness check, not a model leaderboard

## Phase 7

Phase 7 is the structured error analysis. It does not run a new benchmark. Instead, it diagnoses the frozen Qwen setup from Phase 6 by comparing `relevance_only` against `supportcover_final`.

- config: `configs/phase7_error_analysis.yaml`
- frozen setup analyzed: Phase 6 Qwen comparison at token budget `160`, retrieval depth `5`, validation split, and the same 100-example subset policy
- methods compared: `relevance_only`, `supportcover_final`
- canonical mapping: `supportcover_final = no_redundancy`
- output area: `outputs/error_analysis/`

Phase 7 reuses the existing prediction artifacts:

- `outputs/robustness/EXP030_relevance_only_qwen_val_b160_d5_full`
- `outputs/robustness/EXP031_supportcover_final_qwen_val_b160_d5_final`

Run the analysis with:

```bash
supportcover-rag run-error-analysis --config configs/phase7_error_analysis.yaml
```

Main artifacts:

- `outputs/error_analysis/phase7_error_annotations.csv`
- `outputs/error_analysis/phase7_error_summary.csv`
- `outputs/error_analysis/phase7_error_analysis.md`

Phase 7 taxonomy:

- `support_missing`: the packed context misses too much gold support
- `support_present_answer_wrong`: the evidence is present, but generation still fails
- `formatting_mismatch`: the answer is close but fails normalization or exact matching
- `hallucination`: the answer is unsupported by the packed evidence
- `multi_hop_reasoning_failure`: the chain is partly present, but the model fails to connect it
- `insufficient_evidence_forced_answer`: the context is insufficient, but the model still commits to an answer
- `other`: rare fallback when the main labels do not fit cleanly

How to read Phase 7:

- inspect `support_missing` first to see whether SupportCover actually reduces evidence-loss failures
- inspect `support_present_answer_wrong` and `multi_hop_reasoning_failure` next to see whether the bottleneck shifts from packing to generation
- treat `formatting_mismatch` as a diagnostic cleanup issue, not the main method story
- use the representative examples in `phase7_error_analysis.md` to connect the counts to concrete cases

## Phase 8

Phase 8 is the final efficiency and systems summary. It does not start a new sweep. Instead, it summarizes the frozen Qwen main comparison and asks whether SupportCover improves evidence quality with only small pre-generation overhead.

- config: `configs/phase8_efficiency_summary.yaml`
- frozen setup summarized: Phase 6 Qwen comparison at token budget `160`, retrieval depth `5`, validation split, and the same 100-example subset policy
- methods compared: `relevance_only`, `supportcover_final`
- canonical mapping: `supportcover_final = no_redundancy`
- output area: `outputs/systems/`

Phase 8 reuses the existing metrics artifacts:

- `outputs/robustness/EXP030_relevance_only_qwen_val_b160_d5_full`
- `outputs/robustness/EXP031_supportcover_final_qwen_val_b160_d5_final`

Run the systems summary with:

```bash
supportcover-rag run-systems-summary --config configs/phase8_efficiency_summary.yaml
```

Main artifacts:

- `outputs/systems/phase8_systems_summary.csv`
- `outputs/systems/phase8_systems_summary.md`
- `outputs/systems/phase8_systems_analysis.md`
- `outputs/systems/phase8_latency_breakdown.csv`

How to read Phase 8:

- inspect `generation_pct_of_total` first, because the core claim is that generation still dominates end-to-end cost
- inspect `total_non_generation_latency_ms` and `supportcover_overhead_vs_relevance_only_ms` next to see whether SupportCover adds only small overhead before generation
- keep `answer_f1`, `support_f1`, and `coverage_at_budget` in the same table so the systems cost stays tied to quality
- use `phase8_latency_breakdown.csv` as the figure-ready source for a retrieval vs packing vs generation breakdown plot

## Outputs

New runs use a family-first layout:

```text
outputs/
  registry/
  main/
  baseline/
  ablation_budget/
  ablation_depth/
  ablation_component/
  robustness/
  error_analysis/
  systems/
  debug/
```

Each run folder is named:

```text
{experiment_id}_{method}_{model}_{split}_b{budget}_d{depth}_{variant}
```

Examples:

- `EXP021_relevance_only_qwen_val_b160_d5_relevance_only`
- `EXP022_supportcover_qwen_val_b160_d5_full`
- `EXP023_supportcover_qwen_val_b160_d5_no_coverage`
- `EXP024_supportcover_qwen_val_b160_d5_no_redundancy`
- `EXP025_supportcover_qwen_val_b160_d5_no_token_penalty`
- `EXP031_supportcover_final_qwen_val_b160_d5_final`

Paper-grade runs use `EXP` ids. Debug runs use `DBG` ids and go under `outputs/debug/`.

Each run folder contains:

- `config.resolved.yaml`
- `metrics.json`
- `predictions.jsonl`
- `summary.csv`
- `run.log`

The central registry lives at `outputs/registry/experiments.csv`.

## Current Read

- Phase 1 established the baseline ranking on 32 examples.
- Phase 2 showed that the sentence-level methods stay clearly ahead of `paragraph_topk` at 100 examples.
- Phase 3 showed that `supportcover` consistently improves support quality and coverage over `relevance_only` across the tested budgets.
- Phase 4 showed that `supportcover` remains competitive or better as retrieval depth changes.
- Phase 5 shows that removing coverage clearly hurts, while removing redundancy improved results on this fixed setup; the coverage-aware contribution is supported, but the redundancy term does not appear beneficial as currently tuned.
- Phase 6 shows that `supportcover_final` keeps higher `support_f1` and `coverage_at_budget` across both tested models, while answer-F1 gains are strong on Qwen but not on the weaker TinyLlama generator.
- Phase 7 shows that `supportcover_final` reduces `support_missing`, but many remaining failures shift to `support_present_answer_wrong` and `multi_hop_reasoning_failure`, which points to a generator-side bottleneck once evidence packing improves.
- Phase 8 shows that retrieval and packing remain a tiny share of end-to-end runtime under the frozen Qwen setup, so SupportCover's evidence-quality gains come with only small pre-generation systems overhead.

## Next Phase

Phase 9 is the final paper package and write-up asset pass on the same frozen setup. Start it only after the Phase 8 systems summary has been reviewed.
