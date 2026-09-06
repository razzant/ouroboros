# CL-Bench operations runbook — validated v6.71 recipe (one seed → 5-seed leaderboard submission)

Field-tested configuration and operational hazards from the 2026-07-20 full 1-seed campaign
(all 6 domains + own stateless baselines, run-all artifact, <1% instance loss). Complements
`README.md` (launcher surface) and `METHODOLOGY.md` (what the bench measures + results).

## Validated configuration

- Model `openrouter/anthropic/claude-sonnet-4.6`, **effort low pinned uniformly** across every
  engine task type (engine defaults drift between releases — v6.71 flipped review effort to
  high; pin them all).
- Review gate `OUROBOROS_TASK_REVIEW_MODE=required` + `OUROBOROS_REVIEW_ENFORCEMENT=blocking`
  with **`max_improvement_passes` pinned to 1** (adapter does this automatically; see the
  review-mode ablation in METHODOLOGY — unbounded convergence is 2.6× the cost for no measured
  gain on the toolchain-oracle domain, and burns the full per-issue step budget on doomed issues).
- **`max_workers: 3` in system-params — mandatory at scale.** Worker pool size drives engine
  container RSS: 10 workers ≈ 0.8GB/container, 3 workers ≈ 0.35GB. Capacity formula:
  `concurrent_containers × per-container-RSS ≤ 0.8 × Docker-VM memory limit`
  (`docker info | grep "Total Memory"`). Violation symptom: tasks die `terminal (failed)` with
  `worker process crashed (signal 9)` **visible only in the runroot's `task_results/*.json`,
  not in bridge logs** — the VM's OOM reaper kills workers silently.
- `OUROBOROS_TRANSIENT_RETRY_MAX=12` (engine-side provider-blip patience) and the bridge's
  network-outage hold `OUROBOROS_NET_OUTAGE_HOLD_SEC=21600` (submits wait for connectivity;
  in-flight questions pause their scope clock instead of forfeiting). Together they survive
  short blips and multi-hour outages.
  Disclosure (net-resilience sprint): `OUROBOROS_TRANSIENT_RETRY_MAX` no longer bounds a
  REMOTE pre-dispatch transport outage. That class (`transport_unavailable`, $0 released
  attempts) now waits and redials at the round level. CLB solve tasks carry no
  `deadline_at` and the waiting itself spends $0, so the binding rail here is the
  supervisor's absolute per-attempt ceiling (`OUROBOROS_TASK_ABS_CEILING_SEC`, default 6h),
  not a deadline or budget rail: a dead egress holds the task up to that ceiling instead of
  failing it after the burst. The wait is visible as durable `network_wait` events in the
  isolated server's `events.jsonl`. Note: idle-rail survival via waiting progress notes
  requires a real chat thread; headless tasks without a `chat_id` keep the idle rail
  (reaper) as an additional bound on the wait. A transport death AFTER dispatch (the
  socket dies mid-request) is repeated by the primary dispatch at most twice per round as
  new physical attempts. One call has one outer attempt budget
  (`OUROBOROS_TRANSIENT_RETRY_MAX` bounds every attempt of the call, repeats included),
  within which up to three `llm_api_error` rows can be typed transport-death failures (the
  first death plus at most two repeats), reserving up to three upper bounds against
  `OUROBOROS_TOTAL_BUDGET`.
- `OUROBOROS_TOTAL_BUDGET=200` per domain-seed (measured 1-seed domain costs:
  poker $117 · bsm $60 · cohort $58 · code $33 · sales $26 · db $25 — a $60 cap silently
  truncates poker mid-rollout).
- Runner parallelism `--task-parallelism 5 --per-task-parallelism 2`: rollout phases hold ONE
  container per domain, so all six domains can run concurrently; only stateless-baseline
  workers multiply containers. One full seed ≈ $530 / ~7h wall-clock.

## Mandatory companion daemons

1. **clone-sweeper** — every baseline instance boots a fresh container whose runroot carries a
   full git clone of the engine (~44MB). 300+ instances/seed ≈ 15GB of clones. Sweep `clone/`
   from runroots whose `data/` has been idle >15 min (live containers write continuously).
2. **container-reaper** — baseline pool workers can exit without stopping their container;
   zombies accumulate and exhaust the VM (the OOM path above). Every ~3 min stop any
   `clbench-obo-<pid>-*` container whose owner pid is dead.

Both ship in the run handoff bundle (`bench-config/`).

## One seed vs submission

- **Smoke first** (~$3–5): db domain, `--task.schedule smoke --task.num-questions 3`, baseline
  skipped. Confirms clone → image → container → bridge → typed final-answer delivery.
- **One seed**: `run-all --name <name> --runs 1` → `final_results/runs/<name>/` (the same
  artifact layout the leaderboard consumes; includes own baselines → mean gain).
- **Submission** (per continual-learning-bench.com/docs/submitting): `run-all --name <name>`
  WITHOUT `--runs` (default schedules = 5 rollouts/task), recovery via `--missing-only`
  (task-granular; keep `--system-params` byte-identical between resume invocations).
  PR checklist: system name + description, model (provider+version), mean gain + per-task
  scores from `final_results`, public implementation link, eval date, Discord pre-contact.
  Two gotchas: the leaderboard scripts hard-code known run names (`DEFAULT_RUN_NAMES` /
  `SYSTEM_DEFS`) — ask maintainers to add yours in the submission PR; and cohort_studies
  reference data mixed two reward metrics before pgasawa/continual-learning-bench#9
  (merged 2026-07-19) — score against a leaderboard checkout that includes it.

## Known loss classes and their closures

| Class | Signature | Closure |
|---|---|---|
| VM OOM worker kill | `signal 9` in runroot task_results | max_workers 3 + reaper + VM sizing |
| Host network outage | engine silent, DNS dead | bridge outage-hold + retry 12 |
| Prose final (payload left in agent workspace) | task `completed`, no parseable JSON | bridge format-repair round (same-container re-emission micro-task, review passes 0) |

## v6.81.0 campaign operational lessons (2026-07-26)

- **Scoring:** only with `analyze_final_results.py` at/after upstream
  `5f8c50eb` (cohort scale fix, merged 2026-07-19). The pinned checkout's
  script mixes cohort scales and fabricates a phantom top-1 (+0.2231 vs the
  real post-fix 0.196). See METHODOLOGY §10.
- **Multi-seed via bridge is broken for 4/6 domains** (run_index dropped —
  METHODOLOGY §11). Pre-flight: diff prompt MD5s across seeds before spending.
- **OpenRouter budget truth:** spendable = min(key limit − usage from
  `/api/v1/key`, account balance from `/api/v1/credits`). Key limits are not
  money (precedent: $12k limit on a $2k account nearly killed the campaign).
  Watchers must print both and alert on the min. Keys shared with parallel
  campaigns drain between checks.
- **Live key rotation without restart:** edit `OPENROUTER_API_KEY` in the
  isolated server's `runner_state/*/data/settings.json`; it is re-read at the
  next task start, rollout memory survives. After rotation, check the first
  completed questions for nonzero cost (reward=0 at cost=0 = fail-open
  poisoning; quarantine and re-run).
- **Secrets hygiene:** isolated servers snapshot LIVE owner keys into
  `runner_state/*/data/settings.json` (world-readable by default on a shared
  host). `chmod 600` after the campaign; never ship `runner_state/` or
  `clone/` in any bundle.
- **Interrupted stateful rollouts are not resumable as seeds** (memory state
  is the measured quantity): a partial domain arm is a write-off, not a
  top-up. Kill the least-complete arms first when money runs short.
