# Ouroboros Benchmark Devtools

This directory contains tracked operator tooling for reproducible benchmark
work. These files are reviewed when touched, but are not imported by the
runtime core and are not packaged as app runtime code.

## Integrations

- `terminal_bench/` — Harbor installed-agent adapter for Terminal-Bench 2.1.
  Use `run_tb.py` for leaderboard-shaped k-trial runs and submission layout;
  use `run_harbor_smoke.py` for small local smoke runs.
- `osworld/` — OSWorld 2.0-aligned step-loop adapter (pinned
  `xlang-ai/OSWorld-V2@c261cb57`, 500-step budget, official `show_result.py`
  result layout, VM-state-aware prompting + terminal `final_answer` audit
  capture) plus logs-only audit tooling; runnable against a local
  vmware/docker OSWorld checkout, cloud providers and checkpoint curves not
  implemented — see `osworld/METHODOLOGY.md`.
- `swe_bench_pro/` — SWE-bench Pro patch capture/grading. Frozen prepared
  repos use `pro_predictions.py`; persistent evolutionary runs use
  `e1v2/run_pro.py` / `e1v2/auto_run.py`.
- `swe_bench/` — standard SWE-bench prediction helpers.
- `programbench/` — ProgramBench cleanroom runner (`run_programbench_e2e.py`
  for end-to-end Ouroboros harness runs; `run_programbench.py`
  prepare/package-only).
- `continual_learning/` — launcher wrapper for the EXTERNAL CL-Bench
  (continual-learning-bench.com) runner: strictly sequential memory /
  continual-learning runs against a live Ouroboros agent (evolution off).
  Use `run_clb.py`; the runner repo + its `src/systems/ouroboros/` adapter
  are obtained separately (see its README).
- `cybergym/` — Level-1 CyberGym adapter for the pinned
  `sunblaze-ucb/cybergym` source.  Use `run_cybergym.py` for the
  binary-only, private-sidecar protocol; see its README and METHODOLOGY for
  the exact model, provider, final-PoC, and denominator contract.
- `harness_bench_fast/` — Ouroboros CLI wrapper and methodology notes for the
  public `ai-forever/harness-bench-fast` runner.
- `common/` — shared manifests, result ledgers, safe run roots, secret hygiene,
  and official command builders.

## Output Roots

Write generated run artifacts under an explicit benchmark output root outside
`repo/` and outside live runtime `data/`, typically
`/Users/anton/Ouroboros/bench_runs/...`. Tests must set
`OUROBOROS_BENCH_RUNS_ROOT` to a temporary directory so local test runs do not
pollute real benchmark bundles.

## CLEAN SEED IS MANDATORY FOR SUBMITTABLE RUNS

**BEFORE STARTING ANY SUBMITTABLE / LEADERBOARD RUN, THE SEED WORKTREE MUST BE
CLEAN: `git -C <seed> status --porcelain` MUST BE EMPTY AND
`git describe --dirty` MUST NOT CARRY A `-dirty` SUFFIX. A DIRTY SEED
(UNCOMMITTED ADAPTER EDITS, STRAY FILES) MAKES THE RUN MANIFEST RECORD
`...-dirty`, THE PROVENANCE BECOMES NON-REPRODUCIBLE, AND THE RUN CANNOT BE
SUBMITTED — THE MONEY IS BURNED. IF ADAPTER EDITS ARE NEEDED, COMMIT THEM
FIRST (A WIP COMMIT IN THE SEED IS ACCEPTABLE IF RECORDED IN THE RUN NOTES),
CLEAN UP STRAY FILES, THEN LAUNCH. IF A DIRTY SEED IS DISCOVERED AFTER
LAUNCH, ESCALATE TO THE OWNER IMMEDIATELY (STOP VS FINISH IS THE OWNER'S
CALL) — NEVER STAY SILENT.**

## Shared Sidecar Schemas

- Run manifests record non-secret provenance: requested task ids where the
  benchmark runner exposes them before execution, requested counts/selection
  slots for deterministic first-N runs such as Terminal-Bench, exact argv,
  official command shape, output paths, model slots, the canonical Available-subagents
  projection where the launcher owns it, source commit, dirty-state counts, and hashes.
  Defaults are adapter-specific (`run_manifest.json`,
  `<predictions>.run_manifest.json`, or `osworld_preflight.run_manifest.json`).
- Result ledgers are denominator-preserving Ouroboros JSONL files. They record
  every requested instance, including setup failures, timeouts, and empty
  patches, even when the official benchmark prediction/submission format only
  accepts successful rows. Defaults are adapter-specific (`result_index.jsonl`,
  `<predictions>.ledger.jsonl`, or `osworld_preflight.ledger.jsonl`).

These sidecars are audit artifacts, not replacement scoring. Official benchmark
harnesses and official result files remain the scoring authority.

## Bench-Template Scaffold Defaults (v6.55.0)

All committed bench settings templates share these disclosed defaults:

- Cost pacing (v6.56.0): tasks with a finite budget receive latched in-task
  COST milestones (50/25/10% remaining + ~80%-spent wrap-up note) from the
  `task_pacing` SSOT; `budget_profile.cost_hard_stop_pct=0` (SWE-Pro/PB
  profiles) disables the in-task hard stop AND the wrap-up affordability rail,
  so deadline/rounds own the bounds.

- `OUROBOROS_MAX_WORKERS=4` — same-model subagent slots for decomposition
  WITHIN one task (the root agent takes one lane). Never independent attempts
  with selection, so pass@1 claims hold. Fixed-model profiles serialize exactly
  one `api_model` row in `OUROBOROS_SUBAGENTS`; they cannot inherit a default
  Light scout, second family, or agent-session route. The core default (10) is untouched.
- `OUROBOROS_SAFETY_MODE=light` — bench containers/rendered data roots are
  disposable jails; deterministic guards stay, the LLM safety pass is kept for
  integration tools only. User defaults are untouched.
- `OUROBOROS_RUNTIME_MODE=pro` for CONTAINER benches (Terminal-Bench,
  SWE-bench Pro, ProgramBench, OSWorld). GAIA stays `light`: its solver runs
  without workspace isolation against a live repo, so pro would grant benchmark
  prompts write authority over the system body.
- `claude_code_edit` disabled in every bench solve task — benches measure the
  single-model Ouroboros harness; external agent-session delegates are a separate
  experiment. (D10 retired the tool itself; the legacy name in these configs
  stays meaningful because `disabled_tools=["claude_code_edit"]` also withholds
  the successor `delegate_start`.)

Per-bench METHODOLOGY files carry the full rationale.

## Methodology Rule

Benchmark changes must be general-purpose harness improvements first. Do not add
task-specific answers, hidden verifier knowledge, or resource/timeout overrides
that violate a benchmark's official submission rules.

## Upstream-Drift & Protocol-Fidelity Pre-Flight (MANDATORY before any expensive run)

Four checks, each of which failed at once in the CLB v6.81.0 campaign
(continual_learning/METHODOLOGY.md §10-§12) and would have cost the whole
budget if the run had been the submission:

1. **Upstream drift.** Check the external benchmark repo's commits/PRs/issues
   AFTER our pin (via GitHub API/web; do not fetch into the pinned clone). A
   metric/scale change can ship as **re-scored reference artifacts with no
   task-code commit** — when reference artifacts carry two reward copies
   (top-level vs nested), prove which one the local scorer reads before
   trusting any number it prints.
2. **Empirical protocol fidelity.** Verify every protocol parameter by its
   ARTIFACTS, not its flag. Multi-seed must mean different question orders:
   diff prompt hashes across seeds BEFORE mass spend (CLB: `--run-index` was
   accepted and silently dropped on 4/6 domains — five "seeds" were five
   replicates). This generalizes the "declared vs applied" settings rule to
   the harness itself.
3. **Read submission requirements BEFORE the run, not after.** Which arms are
   mandatory (stateless baseline!), how many seeds, which artifact layout,
   whether a public implementation link is required — and run in
   submission-shape via the official runner from the start. Money burned on a
   non-submittable path converts to a submission only by fabricating
   provenance, which is prohibited.
4. **A pinned local scoring script is not ground truth.** Reconcile local
   normalization against the PUBLIC leaderboard (top-1 value, last-updated
   date). A mismatch means a convention divergence to be root-caused — not
   "the other party's numbers are stale". Never compare two systems' raw
   scores without proving they are on the same scale.

## LifelongAgentBench Status

`lifelongagentbench` (arXiv 2508.19005) has NO adapter here: the external
runner is unavailable (only traces from a prior external run exist), and
the observed 100% run was a metric artifact — the runner lacked a gold oracle,
so e.g. a NULL-primary-key task graded as pass. Do not cite that number.
Status: blocked until a real runner with gold labels is obtained.
