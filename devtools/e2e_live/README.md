# Live E2E stand (`devtools/e2e_live/`) — operator manual

Operator documentation for the opt-in live E2E stand. It lives beside the code it describes; the
engineering handbook keeps the rules that bind runtime changes and points here for the stand itself
(DEVELOPMENT "Rules by change class" → "Live E2E stand").


`python -m devtools.e2e_live.run_live_lanes` exercises owner-shaped work on
isolated real servers. `run_live_lanes.py` owns admission, seed/settings, the
lane pool, budget and reports; `scenarios.py::SCENARIOS` owns scenario prompts,
settings overrides and callable acceptance checks; `stub_lane.py` reuses the
loopback model and review answers in `tests/system_e2e/harness.py` for the
`--stub` $0 rehearsal; `ui_probe.py` owns the real-browser client;
`traces.py` owns each lane's key-redacted trace bundle.
Keep this opt-in stand outside runtime imports and default local evolution.

## Scenario acceptance

Judge durable artifacts and actual consumer observations, never model prose
or an HTTP 200 alone. `LaneContext.check` refuses duplicate keys so a later
task cannot overwrite an earlier verdict; multi-task scenarios use separate
terminal-check namespaces. `--attempts N --pass-of K` records every attempt
and requires K passes for EACH selected scenario.

| Scenario | Work and required evidence | Rationale / limits |
|---|---|---|
| SM1 | Change the shared brand accent consistently with DESIGN.md §3 in `web/ui.css`, exercise the app and setup wizard, then land a reviewed release through `preflight_review` → `commit_reviewed`. The full profile uses advanced runtime and blocking enforcement, with no landing skip flags. Acceptance retains the S2 checks: the commit exists and includes the changed shared palette with nonempty accent/focus roles, VERSION strictly increases, the landed carriers pass `commit_admission.release_metadata_preflight`, the worktree is clean, a real advisory ledger row and the commit gate's review-ledger record of the wave (seats dispatched, both questions answered) exist, usage is positive, and the browser reads the new accent and matching accent/focus roles on both `/` and `/onboarding` after restart. | One shared file does not prove both documents loaded it: the browser oracle detects a missing wizard link or divergent page override. The named accent roles and alpha ladder remain part of the palette. SM1's lane-local release/review/restart contract is separate from a version-neutral contributor PR; changing its source oracle must not remove those obligations. `vision_evidence_present` records browser/vision tool rows for reviewers to judge, not a host assertion that the image was inspected. `committed_companions` records paths beyond the palette, release carriers, DESIGN and comment-only CSS as facts, not an automatic scope failure: reviewers may identify another legitimate accent consumer. The clean-tree check discloses and tolerates only transient `.ouroboros/` scratch. |
| SW1 | The Swarm button arms `force_plan` on the ordinary chat send. Require the managed root and plan review, at least two completed children with causal parent/root/depth lineage, a `swarm_fanout` receipt covering them, absorbed-child finalization, the with-children cost rollup without retired aliases, positive usage and the `/proc` environment-based orphan check. | UI admission, child execution and root accounting are separate proofs. An API fallback can continue diagnostics when the browser is unavailable, but cannot pass `ui_swarm_path_exercised`. Children spend under their root's fence. |
| SK1 | The model authors `SKILL.md` + `plugin.py` and calls `skill_preflight`; the runner reviews, grants exactly the manifest's one privileged permission (`inject_chat`), enables, dispatches, disables and deletes. Require persisted findings plus HTTP 200 with `executable_review` in BOTH the review response and `/api/extensions`; retain separate `author_*` / `dispatch_*` terminals. Dispatch needs the generation-bearing durable row, typed `status=ok`, exact echo and one host-attributed owner-chat relay per successful call. | The product's executable-review gate decides eligibility under the applied enforcement; SK1 sets no enforcement override. Clean state, non-PASS items, status and blocking reason remain facts, because requiring all-PASS would measure author quality instead of the lifecycle. A generation digest alone also appears on failed dispatches. The fixture exercises its declared permission rather than requesting an unused grant. |

Preserve the typed SM1 refusal trail from `commit_refusal_facts`: advisory
`block_reason`s, every `commit_reviewed` / `preflight_review` result's
`⚠️ CODE:` (including `PREFLIGHT_BLOCKED`, `TESTS_PREFLIGHT_BLOCKED`,
`SCOPE_REVIEW_BLOCKED`), landing skip flags and the terminal `reason_code`
(for example `budget_exhausted` or `deadline_local`). The stub projects its
release bump offline through `release_sync.sync_release_metadata` plus the
README history row and exercises the same hermetic preflight. Its canned
review answers do not establish paid-model design judgment.

## Seed and settings

Run `admit_benchmark_run` before world-shaped work and finalize the manifest
on every exit (`launcher_audit.audit_source` pins this order). Materialize
`--seed` (default HEAD, resolved in `--source-repo`) once as a clean detached
clone under the run root; each lane clones it and checks both cleanliness and
the admitted SHA. Source-checkout dirt is disclosed, never included in the
seed. Unresolvable refs or a dirty seed produce a typed `run_manifest.json`
refusal. This prevents concurrent source edits from changing the tested tree.

Build settings from the tree's defaults and explicit stand knobs, never the
owner's live settings. Read the key only by the environment NAME in `--key-env`
(default `OUROBOROS_E2E_LIVE_OPENROUTER_KEY`), never a pool file. Keep the
run-root `effective_settings.json` redacted; only each lane's 0600 settings
file contains the key, with fingerprint disclosure. The model recorded in
the manifest comes from the applied settings file, not argv. Credit admission
uses the minimum of key-limit remaining and account credits, refusing below
`--min-credit-usd` (default the run cap).

Paid runs use `scenarios.STAND_PANEL_SETTINGS`: the review pool as catalog rows
marked Reviewer (`scenarios.STAND_REVIEW_PANEL`) — Gemini 3.8 Flash, GPT-5.6 Luna
and DeepSeek v4 Pro packing the brief, DeepSeek v4 Pro reading the work itself —
beside an unmarked Claude Sonnet 5 row an author may name for a preflight; every
row at low effort, task/evolution at medium. The retired lane and review-effort
keys are never written: the isolated settings builder drops them, so a panel
under them would vanish and the lane would run the factory rows. `--production-panel`
selects the tree's defaults instead (no catalog: the factory reviewer rows);
neither choice changes installed product defaults.
A scenario's overrides are applied over that template, and the lane reviews with
the document they produce: SW1 composes its catalog as the scout beside the
template's reviewers (the stand panel's marked rows, the factory rows under
`--production-panel`, the stub lane's keyless rows), so every lane has a pool.
The default `full` profile retains each scenario's enforcement; `wiring` sets
advisory enforcement and must be reported as such.

`--lanes` defaults to 4 (maximum 6); starts stagger 2–3 seconds. Cap nested
preflight load through the existing
`OUROBOROS_PREFLIGHT_TEST_WORKERS=max(2, 16 // lanes)` lever, set before lane
startup and recorded in `extra.preflight_test_workers` and every lane row.
`IsolatedServer` forwards that key through its settings-authoritative sweep;
`preflight_runner._preflight_env` scrubs it and all projected runtime settings
from the candidate suite. Otherwise each lane's `-n auto` multiplies the host
CPU count, while leaked loopback settings would change the suite under test.

## Budget admission and ordering

`RunBudget` owns the run-wide `--total-budget` cap (default $100). Each attempt
reserves `max(0.01, per_task_usd × (root_tasks + int(self_mod and expects_absorb)))`
and receives exactly that positive amount as its lane `TOTAL_BUDGET`, never
the whole run cap. SM1/SW1 have one root, SK1 has two; only SM1 adds an evolution
root under `--self-mod`. `--per-task-usd` (default $8) is the runtime per-root
fence, including children. Size it for the selected review panel's reservations,
not only settled spend; SK1's owner-side review is outside root-task accounting
but remains inside the lane cap.

Retain product-default in-task pacing without injecting a benchmark profile:
at task start the early stop is the minimum of `cost_hard_stop_pct` (50% of
global remaining) and the per-task cap minus its planning margin. A lane's
global remaining is its own budget; root reservations do not disable this
earlier stop. SW1's UI root and the evolution root also follow this path.

Read settled spend and unknown-cost counts from `state/usage_attempts.jsonl`;
`llm_usage` omits review/synthesis spend and cannot be the stand's money source.
Admit only while `spent + reserved(in flight) + reservation ≤ cap`. If only
in-flight reservations prevent admission, wait for settlement and recheck;
if `spent + reservation > cap`, record this attempt as `not_run` with
`reason_code=budget_cap` and keep later attempts eligible. The manifest retains
the rule, spend, refusals, `first_refused` and stop reason.

Before spending, `budget_preflight` records a per-scenario reservation table,
the sum across all attempts and `round_worst_case_usd` (the largest `--lanes`
reservations, ONE attempt per scenario). A reservation above the cap, or equal
to it with two or more attempts, refuses with `reservation_unreachable`,
stage `budget_preflight`, exit 3; change flags rather than bypassing the check.
The round estimate can understate concurrency when lanes exceed scenario
count; it is a planning projection, not the admission fence.

`dispatch_order` schedules a1 of every scenario before a2, largest reservation
first within a round and stable among equals. `RunBudget.admit` also enforces
FIFO among pending admissions by that dispatch index; a refused head leaves
the line. Ordering pool submissions alone lets a freed lane's next job beat
an already-waiting scenario. FIFO protects per-scenario attempt coverage at
the cost of head-of-line idle lanes when a smaller reservation could fit.
The initial concurrent arrivals still depend on thread scheduling; later
admission depends on actual spend and settlement. `requested_task_ids` retain
argument order for identity. Keep budget/credit/watch inputs finite and positive
and the watch interval at least five seconds.

## Self-modification and browser lifetime

`--self-mod` is opt-in: enable `OUROBOROS_POST_TASK_EVOLUTION` with cadence
`every_n:1` and seed only `owner_chat_id`, never an active campaign. The
scenario's own post-task promotion must create the one-shot campaign; a
pre-seeded campaign starts unrelated cycles and blocks that promotion.
Only `Scenario.expects_absorb` (SM1) owes the real re-exec/absorb proof.
SW1/SK1 pin promotion off and record `self_mod_absorb: {expected: false}`:
they commit nothing, and an unrelated cycle could restart the server during
the lifecycle being tested.

Capture clone HEAD, served SHA, uptime and absorbed-cycle count BEFORE the
task. `confirm_absorb` requires a later absorbed counter, changed served SHA,
uptime reset and readiness; mere liveness cannot pass
`self_mod_absorb_confirmed`. Each absorbing lane that ran must confirm, even
if the per-scenario pass count was already met. `IsolatedServer.wait_for_absorb`
may return early only after its idle grace and six consecutive ready, idle
polls with no promotion request and no campaign `active_transaction`.
`waiting_for_restart` is still pending work despite an idle queue.
Non-confirmation retains the durable reason (`no_promotion`, `no_decision`,
`cycle_no_op`, `cycle_not_absorbed`, `campaign_<status>`, `cycle_not_enqueued`,
or timeout).

`ui_probe.resolve_ui_client` prefers the suite's `PlaywrightUIClient` when it
supports the required interface, otherwise headless Chromium, otherwise a
typed `ui_unavailable` reason. Lane start probes availability and closes the
probe; `LaneContext.ui` opens the actual client on first use (SM1 after work
and absorb, SW1 throughout its UI-driven task). Restart closes it and the next
use opens against the current server. Open/use/close stay on the lane thread.
A mid-use browser failure records `ui_unavailable:<ExceptionType>` on UI
checks without discarding other evidence as a lane-wide `infra_error`.

## Reports, focused verification and CI

Keep run roots append-only outside `repo/` and live `data/`. Every attempt
writes `lanes/<id>_a<n>/result.json` and `result_index.jsonl`: checks, facts,
settings SHA and secret-free config digest, seed `git describe`, pre/post HEAD,
diff digest, grants by fingerprint, spend, runtime terminal disclosure, and
screenshots when available. Infrastructure failures retain typed
`refusal {type, code, message}` and `reason_code=infra_error:<code>` in both
result surfaces. Post-stop `/proc` survivors fail a passing lane and name up
to twenty PIDs/command heads with an omitted count; without `/proc` the scan
is explicitly unavailable, never passed.

After the lane server stops, on every outcome, `traces.py` copies its
journals into `lanes/<id>_a<n>/traces/` with the data-root layout kept:
`logs/*.jsonl`, `logs/*.log` with the rotated `server.log.<n>` backups, `task_results/*.json`,
`state/{advisory_review.json,usage_attempts.jsonl,queue_snapshot.json,evolution_campaign.json}`
and the observability call manifests `observability/calls/*/*.json`, for
the lane root and every `state/headless_tasks/<id>/data` fork. Never
`settings.json`, `memory/`, the gzip payload blobs or credential stores.
Every credential value the lane could have seen (the `--key-env` value and
each secret-shaped key of the lane settings file: `*_API_KEY`, `*_TOKEN`,
`*_CREDENTIALS`, `*_PASSWORD`, `*_SECRET`) is replaced by
`<redacted:NAME sha256:…>`, the same fingerprint the manifest discloses.
If a value still occurs in any bundle file afterwards, the bundle is
deleted and `result.json` records `traces: {published: false, reason:
"secret_residue"}`; otherwise `traces: {published: true, files, bytes,
redacted, limit_bytes, truncated}`. A bundle above 200 MiB
(`BUNDLE_LIMIT_BYTES`) keeps the newest tail of each journal, opened by a
`trace_truncated` line and listed in `truncated`; JSON files are never cut.
A lane that never started records `reason: "lane_not_started"`, one
without a data root `"no_data_root"`, and a failed copy `"collect_error"`
with its key-redacted error.

The watcher reports lane state, spend/cap and free disk on `/` and `/mnt/data`.
Key headroom is an informational probe on its own thread, with an eight-second
HTTP bound, at most once a minute and failure backoff. A failed probe is not
an alert or a delay of the watcher tick.

Focused contracts live in `tests/test_e2e_live_runner.py` (including exact FIFO
feasibility fixtures), `tests/test_e2e_live_sm1_checks.py`,
`tests/test_e2e_live_sk1_plugin.py`, `tests/test_e2e_live_panel.py`, `tests/test_e2e_live_traces.py`,
`tests/test_server_runner_absorb_wait.py` and `tests/test_e2e_live_ci_lane.py`;
`tests/test_web_typography_static.py` owns shared-source loading and variable
resolution; `tests/test_e2e_live_sm1_palette_browser.py` exercises the two-document
palette oracle, including missing-source and stale-focus-role failures. The real
SM1 stub rehearsal is separately gated by `integration`, `serial` and
`OUROBOROS_E2E_DEEP=mock`; it starts a real server and the hermetic suite, so
ordinary focused/default tests must not accidentally launch it.

The `.github/workflows/ci.yml` `e2e-live` job runs only on an explicit
`e2e_live=true` dispatch, never an ordinary dispatch, schedule, push, PR or
tag, and checks out the dispatched SHA (`gh workflow run CI --ref ouroboros
-f e2e_live=true` tests the development tip). It runs one SM1 attempt with
`--self-mod --total-budget 30 --per-task-usd 15`, reserving $30 for its two
roots, and `--task-timeout 4500` (75 minutes: at 2400 all four runs hit the
deadline, and the traced one spent ~21 minutes on edits and ~17 on the review
path before a review wave was cut off, #1501). The job's `timeout-minutes`
outlasts the stand's own worst-case waits, so the stand's verdict and traces,
not a job kill, end the run. The owner supplies
`OUROBOROS_E2E_LIVE_OPENROUTER_KEY`; its absence
produces the honest green summary `skipped: secret
OUROBOROS_E2E_LIVE_OPENROUTER_KEY not configured`, not a claimed run.
Upload the manifest, index, lane results, screenshots and each lane's
`traces/` bundle even on failure, never a lane's `data/` tree; read a run
with `gh run download <run-id> -n e2e-live-run`. The summary renders
verdicts or the typed refusal/error without changing the stand's exit
verdict. Browser PR proof and the keyless system-E2E lane (manual dispatch and
release tags) retain their
separate existing CI owners; the latter uploads its scenario servers'
`data/logs/`, `data/task_results/` and the journal segments rotated into
`data/archive/*.jsonl` as `system-e2e-traces`. Both uploads are
diagnostics: a failed upload never reddens its job.
