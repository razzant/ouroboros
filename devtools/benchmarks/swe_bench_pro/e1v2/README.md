# SWE-bench Pro E1v2

> ⚠️ **The post-task EVOLUTION branch of this harness is DEPRECATED** (unmaintained,
> historically buggy). Do not rely on `--evolution` / `--self-improve` / the
> `prompt_e1v2.txt` evolution-mode prompt. The **supported path is the fixed-version
> baseline solve**: `run_pro.py` (default `--baseline`, post-task evolution off) with
> `prompt_baseline.txt`. The evolution code/prompt are kept for reference only, not deleted.

This directory is the cleaned repo-tracked port of the colleague `bench_kit`
E1v2 harness.

E1v2 measures a persistent Ouroboros agent across a SWE-bench Pro task order:

- the task repository lives at `/app` inside each Pro image;
- Ouroboros state and Ouroboros source are carried across tasks through
  `obo-data` and `obo-repo` volumes;
- each solve task is followed by native post-task evolution
  (`OUROBOROS_POST_TASK_EVOLUTION=true`, cadence `every_n:1`);
- patches are captured from `/app` with Method C (`git add -A` then
  `git diff --cached <base_commit>`, with validated junk/binary filters);
- grading remains offline through the official SWE-bench Pro evaluator.

Dangling evolution transactions between tasks are healed two ways, which
compose safely. (1) The current Ouroboros core handles them through boot
reconciliation + supervisor auto-restart. (2) The harness keeps the kit's
bash-level "Option A" heal at task start as a belt-and-braces for agents seeded
from an older core that lacks (1): it mirrors the verified path (marking the
committed transaction restart-verified at the container boundary) with a
`git merge-base --is-ancestor` guard that ABANDONS instead when the commit was
rolled back. With a reconciling core, Option A finds no `active_transaction` and
is a no-op. Without it, Option A prevents a poison-pill that would wedge
`enqueue_evolution_task_if_needed` for every later task (E1v2 → E1).

## Pre-task evolution (optional, default OFF)

`run_pro.py --pretask-evolution` inserts an OPTIONAL task-specific self-evolution
phase BEFORE each solve (ported from the colleague bench workspace where it was
developed against v6.45.5). It is disabled by default and fully flag-gated: with
the flag absent, the docker command, container env, mounts, prompts, and
predictions/timeline schemas are exactly the baseline ones.

- The evolution objective is built per task from public inputs only
  (`problem_statement` / `requirements` / `interface`); hidden tests, gold
  patches, and evaluator artifacts stay out of scope (see `METHODOLOGY.md` §6).
- Flow inside the container: `evolve start` with the mounted
  `/opt/pretask_evolution_prompt.txt` → wait for absorption / no-promotion /
  degraded (or `--pretask-evolution-wait-max` seconds; `0` = no cutoff) →
  `evolve stop` → import health-gate over `/obo-repo`; on failure the self-edit
  is rolled back to the pre-evolution HEAD (saved to
  `/out/pretask_rejected_self_edit.diff`) and the server is restarted.
- Extra artifacts (only when enabled): `/out/pretask_evolution_start.json`,
  `/out/pretask_evolution.json`, `/out/pretask_evolution_stop.json` (+ stderr
  sidecars), `/out/pretask_health.err`, `/out/pretask_rejected_self_edit.diff`;
  result/timeline rows gain a `pretask_evolution` object.
- `--solve-step-budget N` is descriptive text for the evolution prompt;
  `--no-hard-timeouts` removes the host-side subprocess timeout (use with
  `--solve-timeout 0` for step-budget-style experiments).
- This is DISTINCT from the deprecated post-task `--evolution`/`--self-improve`
  branch above. When experimenting with pre-task evolution, keep post-task
  evolution off (`--cadence off`) so the two signals do not mix.

Files:

- `run_pro.py` - one task in one Pro image (optional host image cache via `OBO_SWEPRO_IMG_CACHE`).
- `orchestrate_probe.py` - parallel fixed-version probe orchestrator: N workers over `--run-csv` with isolated `obo-repo-w{N}`/`obo-data-w{N}` volumes + per-task reset, inline official grading, image `rmi`, and a per-run `manifest.json`. Requires `OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1` (audited opt-in); `--out-dir` is forced outside `repo/`.
- `auto_run.py` - sequential orchestrator over `task_order_pro_70.csv`.
- `entrypoint_pro.sh` - in-container solve/evolution/capture flow.
- `prompt_baseline.txt` - solve prompt (CURRENT; clean fixed-version baseline, read by run_pro.py).
- `prompt_e1v2.txt` - DEPRECATED evolution-mode solve prompt (reference only; not read).
- `prompt_evolution_steer.txt` - benchmark-only evolution steer (DEPRECATED branch).
- `settings_base.json` / `_run_settings.example.json` - non-secret settings
  template and example materialized settings.
- `build_predictions.py` - build predictions from consolidated E1v2 patches.

Bench-only steer caveat: `prompt_evolution_steer.txt` limits churn (at most one
reviewed commit plus one restart per cycle) AND, in this benchmark environment
only, forbids release bookkeeping (no VERSION/CHANGELOG/README/ARCHITECTURE/
pyproject edits, no P9 version-bump rule). The standing evolution steer already
forbids touching those files, so advisory review will routinely flag a
`version_bump` / `forgotten_touchpoints` finding — that is expected here and is
left as advisory. The steer must NOT be "resolved" by hardcoding those findings
to block; the review enforcement mode is owner-controlled (BIBLE P3). This
mirrors the colleague kit and prevents the self-hardening deadlock where every
evolution commit becomes uncommittable under advisory mode.

## Diagnosing SIGKILL / OOM

A worker logged as `crashed with signal 9 — terminal (no retry)` (or a container
exiting 137) is almost always a kernel OOM kill, not a code crash. `run_pro.py`
sets `--memory`/`--memory-swap` to `--mem-limit` (default 8g) so a runaway
allocation OOMs the container cleanly (exit 137, noted in `container.log`)
instead of the host OOM-killer ambiguously killing the worker. The most common
trigger was an unbounded `search_code` whose root resolved to `/` under the
container's `HOME=/`; core now pre-enumerates a policy-gated file list and feeds
it to ripgrep in batches (no giant single argv → no `E2BIG`), caps the scan at a
file-count limit, and skips non-regular files (`/dev`, `/proc`). To
investigate a fresh OOM: check `container.log` for the 137 note, raise
`--mem-limit`, and read the host kernel log (`dmesg`) for the OOM line.
