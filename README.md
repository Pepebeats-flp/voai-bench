# VOAI-Bench

**An execution-grounded, bilingual benchmark of LLM systems operating IVOA Virtual Observatory services.**

Companion repository of: J. Pinto-Muñoz & M. Solar, *Single-shot, grounded or agentic? Three LLM architectures for natural-language access to the
Virtual Observatory, compared on an execution-grounded benchmark* (submitted to Astronomy and Computing). Repository: https://github.com/Pepebeats-flp/voai-bench. It extends the conference study Pinto-Muñoz & Solar (2026), Proc. SPIE,
doi:10.1117/12.3104988.

## Contents
| Path | What |
|---|---|
| `bench/queries_v1.yaml`, `bench/gold_v1.json` | **Test set**: 157 valid tasks (160 generated) in English and Spanish over SIA2, SCS, TAP/ADQL, SSA and out-of-scope requests, with frozen gold answers |
| `bench/queries_v0.yaml`, `bench/gold_v0.json` | Development set (50 tasks) used to build the harness |
| `bench/generate_v1.py`, `bench/build_gold.py` | Reproducible test-set generator (seed 2026) and gold builder (queries live services) |
| `harness/vo_tools.py` | Closed catalogue of VO tools (IRSA SIA2, SkyView, Gaia/IRSA TAP, ESA ISO/Herschel SSA, Sesame) with a record/replay cache |
| `harness/agents.py` | The three proposed architectures: A0 VOAI-Plan (single-shot plan), A1 VOAI-Ground (+ deterministic name resolution), A2 VOAI-Agent (tool-calling agent with execution feedback), and ablations |
| `harness/rules_baseline.py` | Rule-based baseline R0 (no LLM) |
| `harness/score.py` | Execution-grounded scorer (strict and target-tolerant success) |
| `harness/analysis.py` | Statistics: task-clustered bootstrap CIs, paired sign-flip tests with Holm, Friedman, GEE, Pareto, pass^k |
| `harness/run.py`, `harness/batch_runner.py` | Resumable experiment runner; batch runner with budget caps |
| `harness/dashboard.py` | Local live dashboard (`http://localhost:8765`) |
| `results/` | Analysis summaries (JSON) |
| `data/voai-bench-data-v1.0.0.tar.xz` | Full episode logs (test and development) and the recorded VO service responses (cache) |

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 1) Unpack the data package at the repository root (creates results/test.jsonl, results/pilot.jsonl and cache/):
tar -xJf data/voai-bench-data-v1.0.0.tar.xz
# 2) Reproduce all statistics offline (no API calls):
.venv/bin/python -I -c "import sys; sys.path.insert(0,'harness'); import analysis, json; \
  print(json.dumps(analysis.analyze('results/test.jsonl', reps=[0], archs_keep=['A0','A1','A2'])['by_arch'], indent=1))"
.venv/bin/python -I harness/make_figures.py results/test.jsonl
# 3) Re-run episodes in replay mode (VO responses from the cache; needs an LLM endpoint):
export VOAI_BASE_URL=https://api.openai.com/v1 VOAI_API_KEY=...   # any OpenAI-compatible endpoint
VOAI_CACHE_MODE=replay .venv/bin/python -I harness/run.py --name myrun --bench v1 \
  --models <model-id> --archs A0,A1,A2 --langs en,es
```
Set `VOAI_CACHE_MODE=record` (default) to query live services and extend the cache.

## Metrics (summary)
Primary: strict execution-grounded success (TAP: set equality of row identifiers, cf. Spider/BIRD execution
accuracy and ALeRCE perfect match; SCS: exact source set; SIA2/SSA: returned product must exist in the tool
trace, contain the target and overlap the requested band; out-of-scope: abstention). Sensitivity: target-tolerant
success accepting a different centre within an object-size radius. See the paper for details.

## Licence
Code: MIT (`LICENSE`). Benchmark tasks, gold answers and results: CC BY 4.0 (`LICENSE-data`).
Service responses in the cache remain subject to the terms of the originating archives (IRSA, NASA SkyView,
ESA Gaia/ISO/Herschel archives, CDS).

## Citation
See `CITATION.cff`.
