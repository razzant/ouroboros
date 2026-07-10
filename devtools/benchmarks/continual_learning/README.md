# CL-Bench (continual-learning-bench) launcher

Launcher wrapper for running Ouroboros on **CL-Bench**
(continual-learning-bench.com, arXiv 2606.05661) — a benchmark that feeds a
system a **strictly sequential** stream of task instances and measures
continual learning: a memoryless *stateless* baseline vs a *stateful* rollout
where one persistent system carries state across the whole ordered sequence.
For Ouroboros the carried state is its **native memory** (scratchpad /
knowledge), which is the point of running this bench: runs go against a live
agent with memory ON, evolution OFF by default.

See `METHODOLOGY.md` for what the benchmark measures, official scoring, our
scaffold disclosures, the known failure taxonomy, and honest limits.

## External runner REQUIRED (not vendored)

This directory contains only the launcher. The benchmark itself — tasks,
scoring, schedules, and the in-runner Ouroboros adapter — lives in the
external `continual-learning-bench` repository and must be obtained
separately:

- The runner repo provides `run_benchmark.py`, `src/`, `schedules/`, and the
  official analysis scripts (`scripts/analyze_final_results.py`,
  `scripts/generate_leaderboard.py`).
- The **Ouroboros adapter** is the `src/systems/ouroboros/` subtree of that
  repo (`system.py`, `_launcher.py`, `_docker_launcher.py`,
  `run_clbench_bridge_agent.py`, `clbench_step_shim.py`). The reference run
  pinned it at commit `56764d6`; a full copy ships in the run handoff bundle
  (`clbench-db-40q-2026-07-01.tar.gz`, under
  `bench-config/external-adapters/ouroboros/`) together with the adapter's own
  METHODOLOGY. If the adapter is missing from your checkout, restore it from
  that bundle.
- The adapter needs a **dedicated Ouroboros clone** (`--ouroboros-clone`,
  never the live repo) whose `devtools/benchmarks/common/server_runner.py` it
  imports, plus (docker path) the `clbench-ouroboros:dev` image and the
  runner-side `clbench_skill/clbench_remote` skill source.
- Run under a Python that has the runner's deps (litellm/pydantic/...) AND the
  Ouroboros deps; the launcher defaults to `<runner>/.venv/bin/python` when it
  exists.

The launcher fails loudly when any of these are absent; `--dry-run` works
without them.

## v6.56.0 operator patches for the pinned adapter

See `operator_patches/README.md` — three host/runtime incompatibilities of the
pinned external adapter with v6.56.0 (safety-lowering owner-guard, host-loopback
shim unreachability from Linux containers, rootful-daemon uid mismatch) and the
unified diffs that port it.

## Strictly sequential — parallelism is forbidden

Two separate knobs, deliberately kept apart:

1. **Cross-task order** is the benchmark. The stateful rollout MUST process
   instances strictly one-by-one (`--instance-workers 1`, the default; the
   launcher refuses more unless `--allow-parallel-baseline` is passed, and
   even then fan-out applies only to the per-instance-independent *stateless
   baseline* arm — each parallel worker boots its own container, so watch
   disk: the runner's default schedules carry `max_workers: 12` and that has
   filled a disk before).
2. **`OUROBOROS_MAX_WORKERS=4`** (settings template) is the agent's *internal*
   worker pool — subagent decomposition WITHIN one task. It does not (and must
   not) create cross-task parallelism; it is disclosed as a scaffold parameter
   in the manifest.

## Launch

```bash
cd repo
# Dry run: manifest + rendered settings + exact planned runner argv, no spend.
python devtools/benchmarks/continual_learning/run_clb.py \
  --runner-path ~/continual-learning-bench \
  --ouroboros-clone ~/ouroboros-bench-src \
  --dry-run

# Standard "clean path" (per-action, runner owns the loop) — DB domain, 1-seed smoke:
python devtools/benchmarks/continual_learning/run_clb.py \
  --runner-path ~/continual-learning-bench \
  --ouroboros-clone ~/ouroboros-bench-src \
  --path standard --domain database_exploration --runs 1

# Bridge path (whole-question; the shape of the reference full40 2026-07-01 run):
python devtools/benchmarks/continual_learning/run_clb.py \
  --runner-path ~/continual-learning-bench \
  --ouroboros-clone ~/ouroboros-bench-src \
  --path bridge --domain database_exploration \
  --phases stateless,stateful_noevo --num-instances 40

# Re-normalize an existing run dir:
python devtools/benchmarks/continual_learning/run_clb.py --collect-only bench_runs/continual_learning/<run>
```

Provider keys are taken from the environment first (`OPENROUTER_API_KEY`,
...), falling back to the live `data/settings.json`; they travel via the child
environment only and are never written into run artifacts (the rendered
`_run_settings.json` has all secret-shaped values blanked).

## Run layout (append-only, one fresh dir per launch)

```
bench_runs/continual_learning/continual_learning_<stamp>_<pid>/
  run_manifest.json     # provenance + template-fidelity report + disclosures
  _run_settings.json    # rendered settings template, secrets blanked
  traces/               # bridge path: <condition>/<domain>/q###/{prompt.txt,task_outcome.json,...}
  runner_state/         # adapter sidecar ledgers + isolated engine run roots
  results.json          # normalized per-condition means + memory/evolution effects
  result_index.jsonl    # denominator-preserving per-instance ledger
```

Official scoring authority stays with the external runner
(`normalized_reward_mean` via its analysis scripts); `results.json` is an
audit sidecar of raw per-instance rewards.

## Template fidelity (read the warnings)

The pinned external adapter enforces model slots, effort, worker pool,
evolution flag, budget, and OpenRouter provider routing through its own
interface; the launcher feeds those from `settings_base.json`. Three template
knobs are **declared but not forwarded** by the pinned adapter —
`OUROBOROS_SAFETY_MODE=light` and `OUROBOROS_REVIEW_ENFORCEMENT=blocking`
(docker path drops both) and
`CLBENCH_SOLVE_DISABLED_TOOLS` (incl. `claude_code_edit`) — the launcher
exports them, records the gap in the manifest, and prints loud warnings. See
METHODOLOGY.md "Template fidelity" for the two-line adapter patch that closes
this.
