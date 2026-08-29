# Baseline inventory

Inventory date: 2026-08-29  
Repository revision inspected: `79e2d4c1e3233c94a9a0faf80be770596a0bc72b`

## Result artifacts

No `data/`, `outputs/`, or `results/` directory exists in this checkout. No predictions, aggregate result files, experiment registry, resolved configuration, split file, environment manifest, or run log is available to hash or validate.

The paths embedded in `configs/phase7_error_analysis.yaml` and `configs/phase8_efficiency_summary.yaml` point to `EXP030` and `EXP031` runs that are absent. Those references are not evidence that the runs exist or are complete. The current checkout therefore contains no baseline result artifact that is safe to use as historical scientific evidence.

## Configuration files

All hashes are SHA-256 values computed from the files as present at the revision above.

| Path | SHA-256 | Status |
| --- | --- | --- |
| `configs/default.yaml` | `e60abfef02bcf54b1e61340c634dee2e6c646c37c42639ceb7969ed96ea3cd31` | Historical/default; contains the conflated `no_coverage` variant. |
| `configs/final_budget.yaml` | `f1141d9fbb765f685504e58a7db155c0b5e439819360b5663a0d7f42da103919` | Planned; unresolved frozen fields and missing split/manifest. |
| `configs/final_cross_dataset.yaml` | `f8a7843c102279cf9d17ee65e93f01198d3e34591012e91ee3efbba00ec04d73` | Planned; unresolved frozen fields and missing data/split/manifest. |
| `configs/final_main.yaml` | `bd47d757c18d648002fe4e09ac267a722c294b0a106e014c0fd68330abbe5980` | Planned; not executable as a final study yet. |
| `configs/final_models.yaml` | `65e4927f727ff2d90b29ba6a73dd14b32bc4d153da3951c15b4ff7b2fb315e7c` | Planned; unresolved frozen fields and missing split/manifest. |
| `configs/phase1_main.yaml` | `35ad22eec80c5f70f299dd9604e2707dbe876211f06421a92a4cf96b99d7aa60` | Historical first-32 validation protocol. |
| `configs/phase2_stability_100.yaml` | `c6ace3393950f9ed30c02144a39e1454c49cb1ed04e5c11370b41c6171e19675` | Historical first-100 validation protocol. |
| `configs/phase3_budget_ablation.yaml` | `a04d982e7d7b16a949dd5ceeffa4ea387ab9523f97e50427e5f706e9deeda323` | Historical first-100 validation protocol. |
| `configs/phase3_sensitivity.yaml` | `886f1fc5440b330a8dfe287c9976304b7d54764d8bcfcf0ab25c6642966cbeb2` | Planned development sensitivity descriptor; runner wiring is incomplete. |
| `configs/phase4_depth_ablation.yaml` | `0b3c04be26d9febf0beabf0227ada0a7ed45ae296c7eeb0dccb22f68a5541c3c` | Historical first-100 validation protocol. |
| `configs/phase5_component_ablation.yaml` | `4020d85a2efa657a0b6b996782193b12ef87bde8740f7470801815fa0ea64c7e` | Historical and scientifically conflated `no_coverage` protocol. |
| `configs/phase6_model_robustness.yaml` | `78cc9a5daa0e6a5e95c372ae39c7b012d785332b64e84fd0a37a44acbc963bcf` | Historical first-100 validation protocol with selected `no_redundancy` variant. |
| `configs/phase7_error_analysis.yaml` | `d9d7e07a3a3d2769722594a031f6d3778fbd2fd839be9e7a763cf4358432d3af` | Historical analysis config; referenced runs are absent. |
| `configs/phase8_efficiency_summary.yaml` | `5fd7e93d799e324691029788baf1ddc7f22386da6e9e03c39f3ccdd74050bae3` | Historical summary config; not an authoritative benchmark protocol. |

## Recoverable assumptions from configuration

- Dataset: HotpotQA `hotpotqa/hotpot_qa`, `distractor` configuration, normally validation in the historical configs.
- Models: Qwen `Qwen/Qwen3-4B-Instruct-2507`; TinyLlama `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in model robustness.
- Generator settings: deterministic decoding (`temperature: 0`, `do_sample: false`), 12 new tokens, titles included, abstention allowed.
- Controlled retrieval: per-example BM25, usually depth 5.
- Default packing budget: 160 tokens.

These are configuration declarations only. Without prediction files, resolved configs, exact ordered IDs, model revisions, environment records, and run logs, they cannot establish what was actually executed.

## Preservation decision

All existing configs and source files were left in place. No historical artifact was overwritten or deleted. Cleanup decisions are recorded in `docs/upgrade/codebase_audit.md`.
