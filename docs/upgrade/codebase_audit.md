# SupportCover-RAG codebase audit

Audit date: 2026-08-29  
Repository root: `C:\Users\pc\Desktop\PROJECTS\Research\supportcover-rag`  
Revision inspected before Phase 0 changes: `79e2d4c1e3233c94a9a0faf80be770596a0bc72b`

## Repository structure summary

- `src/supportcover_rag/`: the only Python package and canonical implementation location.
- `tests/`: 61 collected tests covering the runner, output layout, prompting, split utilities, packing baselines, ablations, statistics, global retrieval, resource monitoring, summaries, and error analysis.
- `configs/`: historical experiment configs plus incomplete planned final-study configs.
- `docs/upgrade/`: publication-upgrade audit, command, schema, and baseline-inventory documents.
- `src/supportcover_rag.egg-info/`: tracked generated packaging metadata.
- No notebooks, scripts directory, CI configuration, or local experiment outputs are present. Ignored raw and processed HotpotQA data are available locally; only the three frozen split artifacts are allowlisted for version control.

The active command path is `supportcover_rag.cli` → `ExperimentRunner` in `pipeline.py`. It loads YAML through `config.py`, processed JSONL through `data.py`, controlled per-example BM25 through `retrieval.py`, selectors through `packing.py`, generators through `generation.py`, metrics through `evaluation.py`, and run artifacts through `experiment_outputs.py` and `io_utils.py`.

## Findings

| Path | Category | Status | Duplicate or legacy relationship | Historical significance | Test coverage | Recommended action | Action taken |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/supportcover_rag/pipeline.py` | KEEP | Active CLI path | Sole active experiment runner | Produces the established run layout | Strong runner/output tests | Keep canonical; add later-phase wiring incrementally | Preserved; Phase 0 only extended optional provenance payloads |
| `src/supportcover_rag/config.py` | KEEP | Active | Sole config loader; accepts YAML values without semantic validation | Loads all historical configs | Indirect coverage | Keep and add validation at the relevant phase | Preserved |
| `src/supportcover_rag/data.py` | KEEP | Active | Sole Hotpot normalization/loading path | Defines canonical `HotpotExample` construction | Split tests cover explicit ID loading | Keep canonical | Preserved |
| `src/supportcover_rag/retrieval.py` | CONSOLIDATE | Active controlled retriever | BM25 scoring math is repeated in `retrieval_global.py` | Required for controlled distractor experiments | Indirect runner coverage | Retain both public retrievers but consolidate the scoring kernel after parity tests | Deferred; deletion would risk changing rankings |
| `src/supportcover_rag/retrieval_global.py` | CONSOLIDATE | Library-only, tested | Shares tokenization and BM25 formula with `retrieval.py` but operates on a global deterministic corpus | Intended corpus-level experiment, not interchangeable with controlled retrieval | Direct tests | Preserve distinct API and labels; share only proven common internals | Deferred |
| `src/supportcover_rag/packing.py` | KEEP | Active canonical selectors | Contains all active selectors; no competing implementation found | Historical and planned packing behavior | Direct baseline, paragraph, ablation, budget, determinism, and uniqueness tests | Keep canonical | Phase 2 audit retained the implementations and added duplicate-key protection to `relevance_only` |
| `src/supportcover_rag/types.py` | KEEP | Active | Sole canonical data model | Historical predictions depend on field names | Broad indirect coverage | Preserve public record compatibility | Preserved |
| `src/supportcover_rag/generation.py` | KEEP | Active | Sole generator abstraction; tokenizer counting and whitespace fallback have different purposes | Historical Transformers/Ollama execution | Prompt tests; model execution not locally exercised | Keep; document model-dependent prerequisites | Preserved |
| `src/supportcover_rag/evaluation.py` | KEEP | Active | Answer/support metrics are distinct from the runtime proxy diagnostics in `coverage_analysis.py` | Defines current aggregate metrics | Indirect runner coverage | Keep canonical metric definitions and names | Preserved |
| `src/supportcover_rag/coverage_analysis.py` | KEEP | Library-only, unexposed | Uses question-term coverage, intentionally different from gold support coverage | Planned mechanism diagnostic | No dedicated test module | Wire and test in Phase 9; never alias it to gold coverage | Deferred |
| `src/supportcover_rag/experiment_outputs.py` | KEEP | Active | Sole output manager | Defines historical folder/registry conventions | Direct tests | Preserve layout and add provenance compatibly | Optional hashes preserved in resolved config; Phase 0 payload storage extended |
| `src/supportcover_rag/statistics.py` | KEEP | Library-only, tested | Sole paired-inference implementation | Planned publication statistics | Direct tests | Add a canonical artifact-writing command in Phase 4 | Deferred |
| `src/supportcover_rag/main_analysis.py` | KEEP | Library-only, untested and unexposed | Composes evaluation, fairness validation, and statistics | Planned main-study aggregation | No direct tests | Wire only after Phase 4 gate | Deferred |
| `src/supportcover_rag/splits.py` | KEEP | Active through explicit-ID runner and CLI paths | Sole split/hash implementation | Critical leakage safeguard | Direct tests | Keep canonical | Added self-contained manifest and multi-field stratification support |
| `src/supportcover_rag/sensitivity.py` | KEEP | Library-only, tested only indirectly through no runner | Descriptor builder, not an executing sensitivity runner | Planned development-only tuning | Role behavior lacks a dedicated CLI test | Complete execution wiring in Phase 3 | Deferred |
| `src/supportcover_rag/freeze.py` | KEEP | Library-only, unexposed | Sole canonical hashing/freeze helper | Planned absolute freeze point | No dedicated tests | Add tests and a controlled freeze command in Phase 3 | Deferred |
| `src/supportcover_rag/final_validation.py` | KEEP | Library-only via `main_analysis.py` | Sole fair-comparison validator | Planned final gate | No direct tests | Test before final use | Deferred |
| `src/supportcover_rag/final_study.py` | KEEP | Library-only, unexposed | Declarative plan only; does not launch experiments | Planned dependency audit | No direct tests | Keep declarative and add verifier command in Phase 9 | Deferred |
| `src/supportcover_rag/reproducibility.py` | KEEP | Library-only, unexposed | Sole structured reproducibility checks | Planned final audit | No direct tests | Extend to file discovery/artifact checks in Phase 9 | Deferred |
| `src/supportcover_rag/corpus.py`, `retrieval_evaluation.py` | KEEP | Library-only, tested | Complement global retrieval; not duplicates of answer/support evaluation | Planned retrieval-realism evidence | Direct tests | Keep clearly labeled corpus-level outputs | Preserved |
| `src/supportcover_rag/datasets/` | KEEP | Library-only, unexposed | Adapter boundary for 2Wiki; Hotpot path remains in `data.py` | Planned cross-dataset robustness | No dedicated adapter tests | Preserve Hotpot semantics; test/wire in Phase 6 | Deferred |
| `src/supportcover_rag/external_baselines/` | KEEP | Interface active by dependency injection; no implementation installed | Boundary only, not a fabricated compressor | Required contemporary comparison | Synthetic runner contract tests cover inputs, missing adapter, wrong type, and budget violation | Select one reproducible method later or document external blocker | Interface verified; method selection deferred |
| `src/supportcover_rag/resource_monitor.py`, `benchmark.py`, `environment.py` | CONSOLIDATE | Tested library pieces, not CLI-wired | Compete in scientific authority with snapshot RSS and batch timing in `pipeline.py`/`systems_summary.py` | Planned authoritative systems path | Monitor/benchmark tests; environment untested | Make benchmark path authoritative in Phase 8; retain accuracy-run timings as diagnostic only | Deferred |
| `src/supportcover_rag/systems_summary.py` | DEPRECATE | Active CLI command | Summarizes ordinary run means and RSS snapshots, not true benchmark samples | May describe historical EXP030/EXP031 outputs | Direct tests | Preserve for historical summaries; do not cite as final systems evidence | No deletion; limitation documented |
| `src/supportcover_rag/error_analysis.py` | CONSOLIDATE | Active CLI plus unused blinded builder | Existing CLI produces historical taxonomy artifacts; blinded paired builder is not invoked | Historical analysis behavior and planned Phase 9 behavior coexist | Historical path tested; blinded path not directly tested | Add a canonical paired-blinded command without removing historical outputs | Deferred |
| `configs/phase1_main.yaml`–`phase8_efficiency_summary.yaml` | PRESERVE AS HISTORICAL ARTIFACT | Loadable; scientifically legacy | Numbering predates the new phase protocol; most select first N validation examples | May describe prior experiments even though outputs are absent | YAML loading verified only | Never use for final claims; retain paths for possible historical recovery | Preserved unchanged |
| `configs/default.yaml` | PRESERVE AS HISTORICAL ARTIFACT | Historical default example | `ablations.variants` uses conflated `no_coverage`; typed code defaults contain the canonical independent variants | Captures early behavior and may be needed for old runs | Config loads; legacy behavior has a focused compatibility test | Never use `no_coverage` in new publication ablations; use `no_query_coverage` and `no_title_gain` independently | Preserved and explicitly classified |
| `configs/phase3_sensitivity.yaml` | CONSOLIDATE | Loadable but not executable as declared | `split.ids_file` says development while `experiments.split` says validation; no sensitivity CLI consumes the grid | Planned, not historical evidence | Config-load check only | Resolve data source and runner semantics in Phase 1/3 | Deferred |
| `configs/final_*.yaml` | KEEP | Planned and blocked | Frozen numeric values are `null`; Hotpot final IDs now exist but freeze manifests do not | No result significance yet | Config-load check only | Do not run until dev tuning, freeze, and asset gates pass | Preserved unchanged |
| `src/supportcover_rag.egg-info/` | DELETE ONLY AFTER CONFIRMATION | Generated, tracked | Duplicates package metadata derived from `pyproject.toml` and can become stale | No scientific result significance found | None | Remove from version control and regenerate on install after confirmation | Not deleted |
| `requirements.txt` and `pyproject.toml` | CONSOLIDATE | Both active setup descriptions | Dependency sets differ: `numpy`/pytest extras versus model-runtime packages | Environment recreation depends on them | Install not exercised in Phase 0 | Choose one lock/source strategy and document optional model dependencies | Deferred |
| `README.md` | PRESERVE AS HISTORICAL ARTIFACT | Documentation, not source of truth | Describes an earlier phase sequence and results workflow | Important historical context | Not applicable | Do not edit under current instruction | Preserved unchanged |

## Phase 2 methods audit

The current repository has one canonical implementation of each Phase 2 method, all in `packing.py`; no competing MMR, query-cover, SupportCover, or paragraph packer was found.

| Requirement | Audit result |
| --- | --- |
| Independent SupportCover ablations | `full`, `no_query_coverage`, `no_title_gain`, `no_redundancy`, `no_token_penalty`, and `relevance_only` exist; tests prove each canonical ablation changes exactly one coefficient. |
| Legacy compatibility | `no_coverage` remains intentionally available and changes both `beta_coverage` and `title_bonus`; it is prohibited for new publication-grade ablation results. |
| MMR | Uses the shared `_base_relevance`, shared Jaccard function, shared candidates, configurable `retrieval.mmr_lambda_relevance`, hard budget, deterministic input-order tie breaking, and duplicate-key removal. |
| Greedy query cover | Uses newly covered query terms divided by token cost, without relevance, title gain, redundancy, or the SupportCover composite score. |
| Paragraph support | Renders packed paragraphs unchanged while exposing every real `(title, sentence_id)` through explicit support keys. |
| External compressor | Receives the same question, retrieved paragraphs, and budget; the runner rejects a missing adapter, wrong return type, and actual token-budget violation. No contemporary implementation has been selected or fabricated. |
| Runner registration | Includes `paragraph_topk`, `relevance_only`, `mmr_sentence`, `greedy_query_cover`, `external_compressor`, `supportcover`, and `supportcover_final`; historical debug methods remain separately available. |
| Duplication | No duplicate active implementation was added. |

The audit found and fixed one narrow invariant gap: `relevance_only` now skips duplicate support keys in a supplied candidate list. No scoring formula or candidate construction was changed.

## Duplicate and legacy conclusions

### Code safe to remove

No source file is approved for immediate deletion. The tracked `src/supportcover_rag.egg-info/` directory appears safely regenerable, but deletion is still classified as `DELETE ONLY AFTER CONFIRMATION` because it is tracked and the current task forbids uncertain cleanup.

### Code to consolidate

- Share BM25 internals only after controlled/global ranking parity tests establish that tie-breaking and corpus statistics remain appropriate.
- Route authoritative systems claims through `benchmark.py`, `resource_monitor.py`, and `environment.py`; retain ordinary-run timings for diagnostics.
- Expose the blinded paired error-analysis builder through the eventual Phase 9 path while preserving the historical CLI outputs.
- Reconcile dependency declarations before the reproducibility gate.

### Code and artifacts to preserve

- All older numbered configs and README descriptions remain available as historical protocol records.
- The controlled per-example retriever must remain alongside corpus-level retrieval.
- Existing run-folder and prediction schemas remain backward compatible.

### Items requiring user confirmation

- Removing tracked `src/supportcover_rag.egg-info/` files.
- Deleting or renaming any historical numbered config after external artifacts are recovered.
- Removing the legacy systems-summary command after downstream users, if any, are identified.

## Cleanup actually performed

- No file or artifact was deleted, renamed, or overwritten.
- Optional `config_sha256`, `code_revision`, and `split_sha256` values were made available in run payload artifacts while remaining absent for legacy invocations that do not supply them.
- Documentation now separates active, planned, and historical paths.
- Phase 1 added canonical `create-split` and `validate-splits` commands; explicit IDs were verified to override `runtime.limit` without truncation, while final roles reject a configured limit.
- Phase 1 froze 2,000 stratified HotpotQA training IDs (`0e02afdcdff360d26725abe9c197a457dcbe76c92aa54338cdc146806b9ed7c6`) and all 7,405 validation IDs (`fc5c4bbd3b2a0304803f118cc098eec9d78521ac7f769877774239f52a4ecf6c`), with zero overlap.
- `.gitignore` now keeps raw, processed, cache, corpus, and arbitrary split data ignored while allowing exactly the two frozen ID manifests and their validation report.

## Cleanup intentionally deferred

All consolidation and deletion work is deferred to its relevant phase gate. This avoids changing scientific behavior during baseline preservation and retains any path that could be needed to interpret external historical results not present in this checkout.

## Risks and rationale

The largest scientific risk is accidental reuse of first-N validation configurations as final populations. The largest reproducibility risk is that later-phase library modules look complete but lack canonical CLI wiring and artifact production. The largest systems risk is treating ordinary-run RSS snapshots or batch-amortized generation latency as authoritative benchmark measurements. These risks are documented rather than silently repaired across phase boundaries.
