# ProgramBench

Ouroboros adapter for official [ProgramBench](https://github.com/facebookresearch/programbench)
cleanroom execution. Scoring always goes through the official `programbench` CLI.

## Modes

### End-to-end (`run_programbench_e2e.py`)

Full harness: pull `task_cleanroom_v6` image, seed a host workspace, start a no-network
Docker backend, submit an Ouroboros task with `executor_ref` + `protected_artifacts`,
wait for completion, write `submission.tar.gz`, optionally run `programbench eval`.

```bash
cd <workspace>/repo
source .venv/bin/activate
export OUROBOROS_REPO_DIR=<workspace>/repo
export OUROBOROS_DATA_DIR=<workspace>/data
export OUROBOROS_BENCH_RUNS_ROOT=<workspace>/bench_runs

# Model slots: the server reads OUROBOROS_DATA_DIR/settings.json. For a pinned
# ProgramBench profile merge devtools/benchmarks/programbench/settings_base.json
# into data/settings.json before starting the server (secrets are blanked in the
# template; add the real provider key yourself, never commit it).
#
# settings_base decisions:
# - OUROBOROS_TASK_REVIEW_MODE=required so adaptive multi-pass improvement after
#   task_acceptance_review actually runs on headless bench tasks.
# - OUROBOROS_RUNTIME_MODE=pro (container bench), OUROBOROS_MAX_WORKERS=4,
#   OUROBOROS_SAFETY_MODE=light.
# - The solve task itself disables claude_code_edit (house rule: benches measure
#   the single-model Ouroboros harness, not an external coding agent).
# - Model slots pin openai/gpt-5.5: the id must exist in the OpenRouter catalog
#   (there is no openai/gpt-5.5-mini there — it 400s on every task).

# Terminal 1 — source server (not the desktop app; the sandboxed desktop cannot
# see bench_runs/ workspaces)
python server.py --host 127.0.0.1 --port 8770

# Terminal 2 — smoke (3 easy tasks)
python devtools/benchmarks/programbench/run_programbench_e2e.py \
  --difficulty easy \
  --slice 0:3 \
  --solve-model openai/gpt-5.5 \
  --eval \
  --ouroboros-url http://127.0.0.1:8770
```

Useful flags:

- `--instance-id <id>` — single task
- `--solve-model <id>` — expected OUROBOROS_MODEL; validated against `--settings-path`
- `--dry-run` — docker + `ouroboros_task_body.json` only (no solve)
- `--skip-pull` — reuse already-pulled images
- `--redo-existing` — rerun even when `submission.tar.gz` exists (clears the task checkpoint)
- `--timeout-sec 21600` — per-task wall clock (default; matches official 6h budget)
- `--cpus 4 --memory 16g` — container resources (lower than mini-swe baseline for Mac)

### Prepare-only (`run_programbench.py`)

Writes `ouroboros_task_body.json` and packages an existing workspace when you already
have a running cleanroom container. Does not submit to the gateway.

### Export (`export_programbench_submissions.py`)

Copies `submission.tar.gz` files into a minimal eval-compatible tree for targeted
`programbench eval` runs.

## Budget and pacing

The task body carries `metadata.budget_profile` (see `schemas.programbench_budget_profile`);
`POST /api/tasks` normalizes it additively into `task_contract.budget_profile`
(there is deliberately no top-level `task_contract` field on the gateway):

- `improvement_policy=adaptive`, `max_improvement_passes=3` (explicit cap — the adaptive multi-pass budget is real, not the config default of 1),
  `reserve_finalization_pct=15` (0–100 pct), `stall_rounds_threshold=12`.
- The 6h official budget flows through the body's `timeout_sec` (gateway →
  `deadline_at`); round caps come from settings (`OUROBOROS_MAX_ROUNDS`).

## Reliability (lessons from the 0/5 debug run)

- **Model-id preflight**: direct-provider routes reject legacy `provider/model`
  ids (they need `provider::model`). The runner validates every configured model
  slot via the runtime's `migrate_model_value` before any task burns and fails
  fast with the exact remediation. With an OpenRouter key, `provider/model` is
  canonical and left untouched.
- **Checkpoints**: `submit_and_wait` persists the latest task result to
  `<instance>/ouroboros_task_checkpoint.json` atomically on every poll; a
  restarted harness re-attaches to the recorded task_id instead of re-submitting,
  so a crash or client timeout does not discard hours of in-flight agent work.
- **Status inference**: terminal detection reads the payload's explicit `status`
  against the settled set (`completed/failed/cancelled/rejected_duplicate`) and
  infra failures are classified from `reason_code`/`outcome_axes.execution`
  (harbor-adapter pattern) — never heuristics over error text.

## Invariants

- Use official `programbench` CLI for evaluation and summaries.
- Use `task_cleanroom` task images; do not score locally.
- Tool execution for the benchmark workspace runs in a no-network Docker backend.
- Reference binaries are declared through `resource_policy.protected_artifacts`:
  execute is allowed; byte reads, copy/hash/static introspection/tracing/debugging are denied.
- Submission artifact is `<run>/<instance_id>/submission.tar.gz`.
- Sidecars (`run_manifest.json`, `result_index.jsonl`) are audit artifacts only.

## Mac notes

ProgramBench images are `linux/amd64` only. On Apple Silicon use Docker Desktop or
Colima (`colima start --arch x86_64 --cpu 4 --memory 16`) with
`DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` if needed. The e2e runner
auto-sets `DOCKER_HOST` when a Colima socket is present.

Use the **source** Ouroboros server on port **8770** for benchmarks. The packaged
desktop app on 8765 is sandboxed and cannot access `bench_runs/` workspaces.

Official `programbench eval` needs **Python 3.11+** (`typing.Self`). The adapter
auto-picks `python3.12`/`python3.11` with `programbench` installed, or set
`PROGRAMBENCH_PYTHON=/path/to/python3.11`. Eval defaults to `--docker-cpus 4`
(override with `PROGRAMBENCH_DOCKER_CPUS`).

Run dirs default under `bench_runs/`; set `OUROBOROS_BENCH_RUNS_ROOT` to redirect
(the test suite pins it to a temp dir so runs never leak into a developer's
`bench_runs/`).

## Full-run gate (gate-20, v6.56.0) — operator methodology

This is an **operator procedure**, not an automatic runner feature: the
ProgramBench runner has no built-in gate, and the steps below are enforced by
the operator (and the campaign watcher) around a normal run, not by
`run_programbench_e2e.py` itself.

Begin a full 200-task run with a 20-task calibration gate: the 10 smoke tasks
plus 10 unseen tasks. Score PAIRED per-task against the official leaderboard's
per-task results (matched by instance id), not aggregate-vs-aggregate. Pass
criteria: paired mean on these 20 >= the reference harness's paired mean on the
same 20, AND not worse than the Codex baseline on the shared subset. On PASS,
continue to the full 200; on FAIL, the operator stops the run and reports (on an
overloaded host, pause instead of failing — attach load data — and re-run when
contention clears). Record the gate result, task list, and per-task pairs
alongside the run so a published 200-task number can always be traced back
through its gate.
