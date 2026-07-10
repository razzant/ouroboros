# SWE-bench Pro Devtools

SWE-bench Pro is kept separate from standard SWE-bench because the colleague
materials target the `SWE-bench_Pro-os` evaluator and a Pro-specific patch JSON
handoff.

Files:

- `METHODOLOGY.md` documents the capture and grading assumptions.
- `capture_patch.sh` captures a task-repository patch with untracked text files,
  filters environment junk, drops binary blobs, and requires an explicit output
  path outside the Ouroboros repo.
- `pro_predictions.py` creates Ouroboros-style prediction JSONL by running
  `capture_patch.sh` for prepared task repositories.
- `grade_pro.py` invokes the official Pro evaluator when `--skip-run` is not
  supplied, then aggregates official per-instance outputs. It intentionally
  remains official-output-only; the Ouroboros denominator ledger is emitted by
  `pro_predictions.py` for the prediction/capture phase.
- `e1v2/` contains the persistent-agent evolutionary harness: sequential Pro
  tasks with carried Ouroboros data/source volumes and native post-task
  evolution between tasks. It also offers an OPTIONAL, default-OFF pre-task
  evolution phase (`run_pro.py --pretask-evolution`; see `e1v2/README.md`
  §Pre-task evolution) — the baseline path is unchanged when the flag is absent.
- `task_order_pro_70.csv` records the 70-task order plus the frozen E0 baseline
  verdicts used for E1v2 comparisons.

The aggregation in `grade_pro.py` is not replacement scoring. The official Pro
eval output remains the source of truth.

`pro_predictions.py` writes the official prediction JSONL plus sidecars:

- `<predictions>.ledger.jsonl` records every requested instance, including
  capture failures and empty patches.
- `<predictions>.errors.jsonl` records failed capture rows.
- `<predictions>.run_manifest.json` records source/model/output provenance.

`capture_patch.sh` deliberately keeps source/config fixes such as `setup.py`,
`pyproject.toml`, and lockfiles. It filters environment junk and binary blobs,
not broad config-like paths.

`evolve_pro.py` was removed after v6.26. It used external workspace tasks, which
structurally suppressed cross-task self-evolution in older Ouroboros releases.
Use `e1v2/run_pro.py` / `e1v2/auto_run.py` for evolutionary runs and
`pro_predictions.py` for frozen prepared-repo predictions.

## Building the `oboros-env` volume (self-contained prerequisite for solve runs)

`e1v2/run_pro.py` mounts a read-only Docker volume `oboros-env` into each task image
at `/opt/miniconda3/envs/oboros`; that volume supplies the Python interpreter and
Ouroboros's third-party dependencies (the agent SOURCE is seeded separately into
`/obo-repo` and imported via `PYTHONPATH`, so the volume holds DEPENDENCIES ONLY).
`build_env_volume.sh` builds it self-contained from this repo's `requirements.txt`:

```bash
devtools/benchmarks/swe_bench_pro/build_env_volume.sh            # idempotent; no-op if ready
devtools/benchmarks/swe_bench_pro/build_env_volume.sh --rebuild  # force a clean rebuild
```

It creates a conda env directly at the volume prefix (self-contained, mountable into
arbitrary glibc `jefzda/sweap-images` task images) and builds for `--platform linux/amd64`
to match those images. Run it from a clean checkout (a dirty tree would bake uncommitted
edits' deps into the measured env). On macOS/Colima the source path must be under a
host-mounted directory (e.g. `/Users/...`), not `/tmp`.

## Transport, resilience, and reward-hacking guards

- **musl/Alpine images (install-in-image fallback).** glibc images use the prebuilt
  `oboros-env` volume. For musl images (`oboros-env-musl` is unreliable — musllinux wheels
  for tree-sitter et al. are often missing), `run_pro.py` instead sets `OBO_INSTALL_IN_IMAGE=1`
  and `entrypoint_pro.sh` installs Ouroboros into the task image at container start (venv from
  the mounted clean source, `pip install -r requirements.txt` with a graceful tree-sitter
  fallback — code-intel degrades to string search, the solve still runs). This mirrors the
  Terminal-Bench installed-agent transport and removes the musl-skip class.
- **Crash/teardown resilience.** `run_pro.py` captures `patch.diff` and writes the
  `timeline.jsonl`/`predictions.jsonl` row **before** the post-solve teardown (volume dump +
  next image pull), and its docker cache-load/inspect ops are timed. `auto_run.py` has
  `--task-wall-timeout` (default 9000s): if one task overruns, it kills `run_pro` + the
  `obopro-*` container and continues — the captured patch is already on disk and recorded
  LEGIT, so a colima teardown stall no longer hangs the whole run or triggers a needless
  re-pull/re-solve. A completed task with an existing non-empty `patch.diff` is RESUME-skipped
  (unless `--reset-state`).
- **Gold-history strip (issue #93, OPEN/unpatched).** Public `jefzda` images carry future git
  history, so `git show <fix>` / `git log --all` / tags can leak the gold solution.
  `entrypoint_pro.sh` strips it before the agent starts (detach at `base_commit`, delete all
  other refs/tags/remotes, expire reflog, `gc --prune=now`); residual reachability is
  **warn-only** (`OBO_STRIP_GOLD_HISTORY=0` disables). Strip the history for any publishable
  comparison, or scores are inflated/incomparable.
- **Operator note:** do not run a solve (`run_pro`/`auto_run`) in parallel with `grade_pro.py`
  on the SAME colima/docker daemon — concurrent docker load is the contention that triggers
  teardown stalls. Use separate daemons or serialize solve and grade.

## Single-task smoke

- Eval pipeline (no env volume needed): build a gold prediction from the evaluator's
  `helper_code/sweap_eval_full_v2.jsonl` `patch` field and run `grade_pro.py`. A gold
  patch must resolve (all `FAIL_TO_PASS` + `PASS_TO_PASS` pass).
- Full solve: `build_env_volume.sh`, then `e1v2/run_pro.py --start 1 --limit 1 --baseline
  --reset-state --solve-model <model>`, then `grade_pro.py` on the produced patch.
- The Docker SDK used by the official evaluator ignores `docker context`; on Colima set
  `DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` for `grade_pro.py`.
