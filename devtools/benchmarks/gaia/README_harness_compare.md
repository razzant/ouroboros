# Multi-harness GAIA comparison

Measure different agent **harnesses** (scaffolds) on the same GAIA tasks with the same
official scorer, to isolate the *harness* contribution from the *model*. Each harness is
an inspect `@solver` over the official `inspect_evals/gaia` task; the official
`question_scorer` (quasi-exact-match) decides correctness. Report **pass@1 per level**.

## Harnesses

| Harness | Driver | Model control | Solver |
|---|---|---|---|
| Claude Code | `claude -p` (headless) | Claude-only (`--model claude-sonnet-4-6`) | `inspect_solver/claude_code_solver.py` |
| OpenAI Codex CLI | `codex exec` | OpenAI-only (`--model gpt-5.5`, default) | `inspect_solver/codex_solver.py` |
| Ouroboros | `ouroboros.cli run --start` (dedicated server) | any (`--solve-model openai/gpt-5.5`) | `inspect_solver/ouroboros_solver.py` via `run_gaia.py` |
| null (integrity probe) | none | n/a | `inspect_solver/null_solver.py` |

**Model-lock bridge (for a fair fixed-model delta):** Claude Code is Claude-locked and Codex
is OpenAI-locked, so a direct Codex-vs-Claude-Code comparison confounds model+scaffold.
Ouroboros (model-flexible) is the bridge: `Ouroboros@gpt-5.5` vs `Codex(gpt-5.5)` isolates the
OpenAI-lane scaffold delta; `Ouroboros@sonnet-4.6` vs `Claude Code(sonnet-4.6)` the Claude lane.

## Setup

- Python: `inspect_ai` + `inspect_evals` + the repo deps. (On this machine the working
  interpreter is `/opt/homebrew/bin/python3.11`; a bare `python3` may lack `inspect_evals`.)
- Docker running + the `aisiuk/inspect-tool-support` image (inspect_evals/gaia provisions a sandbox per sample).
- GAIA dataset (gated) cached: `~/.cache/huggingface/datasets/gaia` + `~/Library/Caches/inspect_evals/gaia_dataset`.
- **Codex CLI**: `brew install codex` (the npm path may be unavailable; `node`/`npm` can be broken). Verify `codex --version`.
- **Claude Code**: install the `claude` CLI. Headless `claude -p` needs auth — set `ANTHROPIC_API_KEY`
  (env, or `data/settings.json`, or `~/file1.txt` `anthropic: <key>`); the solver injects it (else "Not logged in").
- **Codex auth**: `OPENAI_API_KEY` (same resolution order). Keys are read at runtime, never committed.

## Run

One harness, all levels, parallel 3 (`GAIA_CODEX_EFFORT` / `GAIA_CLAUDE_EFFORT` pin reasoning
effort for an apples-to-apples cross-harness comparison; codex's own default is `xhigh`, our
solver default is `high`):
```bash
GAIA_CODEX_EFFORT=high \
python3.11 run_harness.py --harness codex      --model gpt-5.5         --subset 2023_all --limit 165 --max-samples 3 --out-dir <dir>
python3.11 run_harness.py --harness claude_code --model claude-sonnet-4-6 --subset 2023_all --limit 165 --max-samples 3 --out-dir <dir>
python3.11 run_harness.py --harness null                                 --subset 2023_all --limit 165 --out-dir <dir>   # must score ~0
```
Ouroboros bridge (dedicated server; `--max-samples 1` — concurrent samples can exhaust the
Docker subnet / contend on one server):
```bash
python3.11 run_gaia.py --solve-model openai/gpt-5.5            --subset 2023_all --limit 165 --websearch-backend ddgs --or-provider resilience --max-samples 1 --out-dir <dir>
python3.11 run_gaia.py --solve-model anthropic/claude-sonnet-4.6 --subset 2023_all --limit 165 --websearch-backend ddgs --or-provider resilience --max-samples 1 --out-dir <dir>
```
Detach a multi-hour run (survives shell exit, sleep, process-group teardown):
```bash
python3.11 daemonize.py <dir> -- python3.11 run_harness.py --harness codex --model gpt-5.5 --subset 2023_all --limit 165 --max-samples 3 --out-dir <dir>
# poll <dir>/.DONE
```

## Scoring

The inspect json log under `<dir>/inspect_logs/*.json` carries per-sample `scores` (`C`/`I`)
and `metadata.level`. Accuracy = correct / scored, reported per level (L1/L2/L3) + overall.

## Integrity caveats (must address before a *publishable* run)

- **Network isolation**: these runs allow web access NOT isolated from the public GAIA
  validation answers on HuggingFace. A web-capable agent can in principle fetch the answer
  (the Berkeley-RDI exploit hit ~98% by answer-lookup). The `null` probe (scores ~0) only
  proves the rig/scorer don't leak credit — it does NOT prove a real agent isn't fetching.
  For a publishable run, sandbox egress (block `huggingface.co`/the GAIA dataset; allowlist a
  proxied search) and inspect traces for answer-lookup behavior.
- **pass@1 only**: report `--epochs k` (pass@k / avg@k + CI) for variance in a final run.
- **Endpoint alignment**: a fair fixed-model delta needs both sides on the same endpoint
  (e.g. Codex via direct OpenAI vs Ouroboros@gpt-5.5 via OpenRouter routes the same model
  through different providers — align both for the final run).
- **Levels**: GAIA validation = 165 (L1=53/L2=86/L3=26); L1 is largely saturated, so report
  all levels and lead with L2/L3 where scaffolds differentiate.
