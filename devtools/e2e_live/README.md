# Live E2E stand (`devtools/e2e_live/`) — operator manual

Operator documentation for the opt-in live E2E stand. It lives beside the code it describes; the
engineering handbook keeps the rules that bind runtime changes and points here for the stand itself
(DEVELOPMENT "Rules by change class" → "Live E2E stand").


`python -m devtools.e2e_live.run_live_lanes` exercises owner-shaped work on
isolated real servers. `run_live_lanes.py` owns admission, seed/settings, the
lane pool and reports; `budget.py` owns the run-wide money ledger;
`scenarios.py::SCENARIOS` owns scenario prompts, settings overrides and callable
acceptance checks; `adoption.py` owns SM1's direct-adoption oracle; `stub_lane.py`
reuses the loopback model and review answers in `tests/system_e2e/harness.py` for
the `--stub` $0 rehearsal; `ui_probe.py` owns the real-browser client;
`traces.py` owns each lane's key-redacted trace bundle.
Keep this opt-in stand outside runtime imports and default local evolution.

Launch the keyless `--stub` rehearsal like any verification command: `python -I -S
scripts/safe_test.py -- <venv>/bin/python -m devtools.e2e_live.run_live_lanes --stub ...`
with a dependency-only venv (DEVELOPMENT "Build and CI"). The recipe covers that
rehearsal only: safe_test scrubs every `OUROBOROS_*` variable it does not keep, the
`--key-env` key (`OUROBOROS_E2E_LIVE_OPENROUTER_KEY`) included, so a paid run under it
refuses with `key_env_absent` before any spend, and how to launch a paid run is not
qualified here. Each lane's server is a real installation started on the
stand's interpreter, so it keeps the boot and adoption dependency sync. For a venv,
`pip_install_target_args` targets the venv and safe_test's `PIP_REQUIRE_VIRTUALENV`
admits it; for the bundled interpreter it adds `--user`, which that guard refuses
(SM1_a1 of 2026-11-02 logged `Could not find an activated virtualenv` at boot). When the bundled runtime's packages match the selected lock, a venv made with
`--system-site-packages --without-pip` can reuse them. Verify that exact runtime
and lock before the rehearsal; this does not qualify the packaged user-site path.
A lane's clone carries no `node-standalone/`, so on a venv outside the packaged
launcher the adoption's Node lane resolves `node` from `PATH` and a missing one
blocks it. Put a runnable Node >= 20.11 first on `PATH` (the bundled
`node-standalone/bin` qualified as 24.16.0 this way); the stand detects and
downloads nothing.

## Scenario acceptance

Judge durable artifacts and actual consumer observations, never model prose
or an HTTP 200 alone. `LaneContext.check` refuses duplicate keys so a later
task cannot overwrite an earlier verdict; multi-task scenarios use separate
terminal-check namespaces. `--attempts N --pass-of K` records every attempt
and requires K passes for EACH selected scenario.

Each lane's server mints its data root's own install identity at boot. The
stand process is another install, and the copied-folder rule refuses another
install even a read of that store. The stand launched the lane's install, so
`run_lane` runs the acceptance as it (`install_identity.acting_as`, a context
variable: that lane's thread only), and the store readers a scenario uses after
the terminal (the durable terminal row, the adoption's task record, SW1's
children and UI root) answer from the lane. Without that scope SM1_a1 of
2026-11-02 ended `infra_error:foreign_install_root` right after its HTTP terminal.

| Scenario | Work and required evidence | Rationale / limits |
|---|---|---|
| SM1 | Change the shared brand accent consistently with DESIGN.md §3 in `web/ui.css`, exercise the app and setup wizard, land a reviewed release through `commit_reviewed` on the task's body candidate, request its adoption with `request_restart(adopt_commit=<that SHA>)` and finish. The full profile uses advanced runtime and blocking enforcement, with no landing skip flags. Acceptance reads the commit the landing call's review record names, and retains the S2 checks there: it includes the changed shared palette with nonempty accent/focus roles, VERSION strictly increases, the landed carriers pass `commit_admission.release_metadata_preflight`, the author's candidate is clean, a real advisory ledger row and the commit gate's review-ledger record of the wave (seats dispatched, both questions answered) exist and usage is positive. The landed `--accent` is the target in both themes. After the task's terminal it follows that one commit through the adoption (`adoption_*` checks, next section), then the browser opens `/` and `/onboarding` fresh in Light and in Dark: that theme must paint, `--accent` must be the target, no accent role, focus role or control fill may keep that theme's seed value, and the documents must match per theme (`palette_computed`, `palette_problems`). | One shared file does not prove both documents loaded it: the browser oracle detects a missing wizard link or divergent page override. The named accent roles and alpha ladder remain part of the palette. `web/ui.css` declares `--accent` per theme and the probe opens Light where the suite's browser lanes open Dark, so one read proves only whichever theme the browser happened to open: a dark-only edit passed those lanes and failed SM1_a1 of 2026-11-03. Each theme is asked for through `prefers-color-scheme`, what System follows; a saved choice paints the same blocks. SM1's lane-local release/review/adoption contract is separate from a version-neutral contributor PR; changing its source oracle must not remove those obligations. `vision_evidence_present` records browser/vision tool rows for reviewers to judge, not a host assertion that the image was inspected. `committed_companions` records paths beyond the palette, release carriers, DESIGN and comment-only CSS as facts, not an automatic scope failure: reviewers may identify another legitimate accent consumer. The clean-tree check discloses and tolerates only transient `.ouroboros/` scratch. |
| SW1 | The room Subagents capsule selects Swarm before an ordinary send. The scenario requests two parallel helpers; verify at least two completed children with causal parent/root/depth lineage, a `swarm_fanout` receipt covering them, absorbed-child finalization, the with-children cost rollup without retired aliases, positive usage and the `/proc` environment-based orphan check. | Swarm itself guarantees neither promotion nor a plan; plan checks require an explicit planning request. UI choice, child execution and root accounting are separate proofs. An API fallback can continue diagnostics when the browser is unavailable, but cannot pass `ui_swarm_path_exercised`. Children spend under their root's fence. |
| SK1 | The model authors `SKILL.md` + `plugin.py` and calls `skill_preflight`; the runner reviews, grants exactly the manifest's one privileged permission (`inject_chat`), enables, dispatches, disables and deletes. Require persisted findings plus HTTP 200 with `executable_review` in BOTH the review response and `/api/extensions`; retain separate `author_*` / `dispatch_*` terminals. Dispatch needs the generation-bearing durable row, typed `status=ok`, exact echo and one host-attributed owner-chat relay per successful call. | The product's executable-review gate decides eligibility under the applied enforcement; SK1 sets no enforcement override. Clean state, non-PASS items, status and blocking reason remain facts, because requiring all-PASS would measure author quality instead of the lifecycle. A generation digest alone also appears on failed dispatches. The fixture exercises its declared permission rather than requesting an unused grant. |

Preserve the typed SM1 refusal trail from `commit_refusal_facts`: advisory
`block_reason`s, every `commit_reviewed` / `preflight_review` result's
`⚠️ CODE:` (including `PREFLIGHT_BLOCKED`, `TESTS_PREFLIGHT_BLOCKED`,
`REVIEW_BLOCKED`), landing skip flags and the terminal `reason_code`
(for example `budget_exhausted` or `deadline_local`). The stub projects its
release bump offline through `release_sync.sync_release_metadata` plus the
README history row and exercises the same hermetic preflight. The scripted author
makes the prompt's accent change mechanically through `scenarios.css_with_accent`:
both themes' `--accent`, the ladder at its alphas, and each theme's opaque roles and
crimson tones on the new hue at their own lightness, aliases untouched; the Owner,
Project, status and voice roles stay. Its canned review answers and scripted author
verify the machinery only, not independent semantic review or design quality: every
stub row carries `machinery_only` and the run's verdict output repeats it. A stub attempt also owes `stub_scripted_final_reached`,
because a script that ran out of steps ends its task `completed` as well.

### SM1's direct adoption

A self-change stays on the task's body candidate until an exact reviewed commit is
adopted. The restart request reaches the supervisor with the task's end, so the
author requests and finishes, and neither prompt nor script waits inside the task.
After the terminal `adoption.observe_direct_adoption` follows one commit:

| Check | The same commit, observed at |
|---|---|
| `adoption_requested_for_reviewed_commit` | the landing's review record (`commit_landings`; the tool text names the record and carries no SHA), the candidate row's newest reviewed commit, the accepted request whose `adopt_commit` spells that SHA (whole, or an abbreviation the tool resolves to it), the handoff authorized for it. The request is inferred from the newest attempt with a recorded restart call: its first `request_restart` the host answered, since a later call of that attempt only reads that request's receipt and its own arguments are ignored, while a call refused before anything was sent leaves the attempt free. A newer attempt with no restart rows or an unrecognisable first answer leaves this inference incomplete; the following durable and serving checks remain independently required |
| `adoption_qualified_by_real_tests` | the handoff's `tests`: `head` and `tree` bound to the commit. Tier (`selected` or `full`) and source (`proof` or `receipt`) are recorded as they happened; `adoption_test_evidence` already decided that they qualify, the stand adds no gate of its own and writes no receipt |
| `adoption_settled_and_restart_verified` | `body_adoption_closed` of that handoff id and SHA with outcome `adopted`, and `restart_verify` expecting and observing the SHA |
| `adoption_recorded_on_task` | the task result's `body_adoptions`, the one durable binding of the archived handoff to this task |
| `adoption_serves_commit` | health, `/api/state.sha` (its 8-character prefix), the runtime attestation against that commit's VERSION, a `worker_boot` at the SHA, and the serving checkout clean at the SHA on the handoff's branch (a `git status` that fails is not clean; `status_error` keeps why) |
| `adoption_serves_committed_stylesheets` | the bytes of `/static/ui.css` and `/onboarding/static/ui.css` against that commit's blob |

The wait reads durable files and tolerates refused connections through the re-exec
gap, bounded by `--ready-timeout`; a healthy adoption took 149–314 s from submit in
the S37 scenario on a loaded host, so size it for the host. A dead end stops the wait
at once, and `facts.direct_adoption.reason` names it: `no_reviewed_commit`,
`adoption_not_requested`, `restart_without_adoption`, `restart_for_other_commit` (a
ref or another commit: whatever settles next is not this request's),
`restart_refused:<code>` (`ADOPTION_TESTS_REQUIRED` for a missing or red run:
the commit stays on its candidate and the serving tree does not move),
`adoption_closed:<outcome>`, `restart_verify_failed`, `server_exited:<rc>` or
`timeout:<stage>`. An unconfirmed boot is no dead end: the product keeps that handoff
for the next start, so a later `adopted` of it still settles the wait, and only at the
deadline does the reason read `adoption_unconfirmed:<reason>`. Such a boot also leaves
an `unconfirmed` row of the adoption on the task until the adopting boot replaces it,
after archiving and verifying, so the wait ends only on the task's own `adopted` row
(`timeout:task_record` at the deadline). Refusals, blockers
and unconfirmed states stay separate facts of that record. The browser opens fresh on
the generation the adoption restarted into; the stand starts no second server for it.

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
advisory enforcement and must be reported as such. It is no qualified SM1 path:
the scripted author calls `commit_reviewed` once, and advisory review can ask the
author to continue (`advisory_replay`), which the script does not, so an SM1 result
under `wiring` qualifies nothing. The keyless SM1 rehearsal runs `full`.

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

Retain product-default in-task pacing without injecting a benchmark profile.
That default has no early cost stop for an ordinary root (owner decision of
2026-10-07, #1128; the former 50% share is gone): with no
`budget_profile.cost_hard_stop_pct` and no producer allowance
(`root_cost_ceiling_usd`, set only for consciousness wakes and Presence),
`task_pacing.resolve_cost_ceiling` answers `disabled` (`no_default_cost_stop`),
and new model calls are refused only when known spend reaches the per-root fence
(`OUROBOROS_PER_TASK_COST_USD`); the lane's `TOTAL_BUDGET`, its own
reservation, refuses new paid calls only. SW1's UI root and the evolution root are
ordinary roots too.

Read settled spend and unknown-cost counts through `budget.lane_spend` (also
reachable as `run_live_lanes.lane_spend`) from `state/usage.sqlite`, with journal
fallback for an older seed. The journal holds a row per transition, so it is reduced
as the store's import reduces it, to each attempt's latest row without the compaction
header, and only read, never truncated or quarantined. A row it cannot read as a JSON
object (torn at the tail or before it, undecodable, or not an object) makes the lane's
reading unknown instead of being skipped, because a torn one-shot row is its attempt's
only record; a journal that vanishes before it opens is unknown too, never an empty
read. Money is decoded as the core decodes it (`_usage_money.amount`): a numeric string
is money, as the core's own export writes a literal a float would change, and the store
keeps such a value in `extra` beside its typed column, so store rows go through the
store's own codec and both sources through one reduction. A journal row the core's row
rule refuses (`usage_ledger.validate_row_fields`: a negative, non-numeric or non-finite
cost) makes the reading unknown. A ledger the core calls degraded
(`usage_store.integrity_degraded`: a journal row it once quarantined into
`usage_attempts.quarantine.jsonl`, which its import keeps beside the store) is never
final, from either source: one more unknown-cost row, never a known $0.
`llm_usage` omits review/synthesis spend and cannot be
the stand's money source. Admit only while `spent + unknown liability +
reserved(in flight) + reservation ≤ cap`. If only in-flight reservations prevent
admission, wait for settlement and recheck; if `spent + unknown liability +
reservation > cap`, record this attempt as `not_run` with `reason_code=budget_cap`,
or `accounting_unavailable` when the known spend alone would have fitted, and keep
later attempts eligible. The manifest retains the rule, spend, unknown liability,
`spend_unknown` (attempt and why), refusals, `first_refused` and stop reason.

Known spend is what the core counts toward a limit (`_usage_money.exceeds_limit`,
#1487): every settled amount, a confirmed price or a disclosed estimate. An estimate
binds as spent and is never final. Unknown cost is never zero. It means a read that
failed, no usage record for a lane whose server process was spawned (zero only where
none ever ran), a degraded ledger, or attempts sent without a final price: settled at
an estimate (its amount already counted, as a floor), a partial receipt or nothing, or
`dispatched`/`unresolved` when the lane ended. A live lane in that state
is still working: its reservation stays counted, a failed read answers its last good
reading and is never raised into another attempt's admission. An ended lane always
leaves the flight and wakes the waiters, and stays charged `max(reservation, known
spend)`: the known part in `spent_usd` and the rest in `unknown_liability_usd`. A lane
that ended without proof its processes are gone (an unconfirmed server stop, a
failed orphan scan or a `/proc` survivor) stays charged the same way whatever its
prices say, because such a process may still spend under its own `TOTAL_BUDGET`. The
attempt row says the same: `budget.spent_usd` is `null` with `budget.spend_unknown`
when nothing could be read, and `budget.unknown_cost_rows` counts the attempts
without a final price.

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
the lifecycle being tested. This post-task evolution is a separate contract from
SM1's direct adoption, which runs with or without `--self-mod` and neither reads nor
moves the absorbed-cycle counter: under `--self-mod` SM1 owes both verdicts, each
from its own record, and the evolution cycle's own handoff never speaks for SM1's.

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
probe; `LaneContext.ui` opens the actual client on first use (SM1 after its
adoption, SW1 throughout its UI-driven task). `LaneContext.restart` closes it and
the next use opens against the current server. Open/use/close stay on the lane thread.
A mid-use browser failure records `ui_unavailable:<ExceptionType>` on UI
checks without discarding other evidence as a lane-wide `infra_error`.

## Reports, focused verification and CI

Keep run roots append-only outside `repo/` and live `data/`. Every attempt
writes `lanes/<id>_a<n>/result.json` and `result_index.jsonl`: checks, facts,
settings SHA and secret-free config digest, seed `git describe`, pre/post HEAD,
diff digest, grants by fingerprint, spend, runtime terminal disclosure, and
screenshots when available. An infrastructure exception retains a typed
`refusal {type, code, message}` and `reason_code=infra_error:<code>` in both
result surfaces. An exception keeps what the scenario had observed (checks, facts,
screenshots, the absorb record) in the row and stays `infra_error` whatever those
partial checks say. Each finalization step is contained, so a failed stop, trace
export or row write leaves the others done and names itself in `finalize_errors`; a
stop that raised counts as unconfirmed. Containment never turns a lost proof into a
pass, because the verdict counts `pass` rows: a passing attempt whose bundle was not
published (`infra_error:traces_<reason>`, `secret_residue` included) or whose row
write failed (`infra_error:record_row`) ends `infra_error`, while an earlier `fail` or
`infra_error` keeps its own reason. `result.json` is written before the index append.
When only the append failed, that file is written again, once, with the attempt's final
row; the append is never retried, because whether a failed append left a row, whole or
torn, is unknown. A failed write of `result.json` itself can leave it torn, and a
rewrite that fails too is named in the run log (`finalize record_row_correction
failed`): the file may then still say `pass` or be torn. The manifest's per-scenario
counts, the exit code and the run log count the row as `infra_error`. An
unknown spend is no product failure, and a failed browser or stub cleanup is
recorded without deciding the status. Post-stop `/proc` survivors fail a passing lane and name up
to twenty PIDs/command heads with an omitted count; without `/proc` the scan
is explicitly unavailable, never passed, and a scan that raised is `orphan_scan:
"failed"` and fails its check. An exit the server stop could not
prove fails a passing lane the same way, listed in `server_stop_unconfirmed`;
a restart after such a stop starts no second server on that data root, and
`--prune-clones` keeps the clone after such a stop, a failed scan or a `/proc` survivor.

After the lane server stops, on every outcome, `traces.py` copies its
journals into `lanes/<id>_a<n>/traces/` with the data-root layout kept:
`logs/*.jsonl`, `logs/*.log` with the rotated `server.log.<n>` backups,
`state/{advisory_review.json,usage_attempts.jsonl,queue_snapshot.json,evolution_campaign.json}`,
the review records, candidate records, qualification receipts and restart/adoption
receipts (`state/review_ledger/*.json`, `state/private_copies/*.json`,
`state/preflight_receipts/*.json`, `state/body_adoption/handoff.json`,
`state/pending_restart_verify*.json`) and the observability call manifests
`observability/calls/*/*.json`, for the lane root and every
`state/headless_tasks/<id>/data` fork, plus the server's own stdout and stderr as
`server_stdio.log`. That file (0600, in the lane directory, kept through the
server's in-place re-exec by `IsolatedServer.start`) is how a server that died
before its first import says why; only the redacted copy travels. Never
`settings.json`, `memory/`, the gzip payload blobs or credential stores.

`state/{current,usage,dialog}.sqlite` travel as data, as
`<store>.logical.jsonl`, never as files: their committed rows may still be in the WAL,
and replacing a value inside pages damages them. Each store is copied with its WAL and
rollback journal into scratch beside the data root (its source is never opened),
recovered and quick-checked there, then written as one header line (source sizes,
schema, column types, row counts) and one line per row of its user tables and virtual
tables, never `sqlite_*` internals or shadow pages. Every string it carries (table,
column and type names, schema SQL, TEXT values) is redacted before JSON encoding, whose
`\uXXXX` escapes would hide a non-ASCII secret from the byte scans after it, and a BLOB
on its raw bytes before base64, typed `{"$type": "blob", ...}`. Not covered: a value
kept compressed or otherwise encoded in any column (the writers of these stores in this
tree declare TEXT/INTEGER/REAL columns and encode nothing; a later writer may), and a
secret already stored JSON-escaped (a quote, backslash or control character inside JSON
text), which the byte redaction of the copied journals misses alike. An export is all or
nothing: a failed check or read, or residue in the finished file, withholds that store
alone, listed in `traces.gaps` with its reason and exported/total counts, while
`traces.sqlite_exports` lists what was published. Each export is consistent for its own
store, not across stores or journals, and is no runnable backup.
Every credential value the lane could have seen (the `--key-env` value and
each secret-shaped key of the lane settings file: `*_API_KEY`, `*_TOKEN`,
`*_CREDENTIALS`, `*_PASSWORD`, `*_SECRET`) is replaced by
`<redacted:NAME sha256:…>`, the same fingerprint the manifest discloses.
If a value still occurs in any bundle file afterwards, the bundle is
deleted and `result.json` records `traces: {published: false, reason:
"secret_residue"}`; otherwise `traces: {published: true, files, bytes,
redacted, limit_bytes, truncated, sqlite_exports, gaps}`. A bundle above 200 MiB
(`BUNDLE_LIMIT_BYTES`) keeps the newest tail of each journal, opened by a
`trace_truncated` line and listed in `truncated`; JSON files and store exports are
never cut.
A lane that never started records `reason: "lane_not_started"`, one
without a data root `"no_data_root"`, and a failed copy `"collect_error"`
with its key-redacted error.

The watcher reports lane state, spend/cap and free disk on `/` and `/mnt/data`.
Key headroom is an informational probe on its own thread, with an eight-second
HTTP bound, at most once a minute and failure backoff. A failed probe is not
an alert or a delay of the watcher tick.

Focused contracts live in `tests/test_e2e_live_runner.py` (including exact FIFO
feasibility fixtures), `tests/test_e2e_live_sm1_checks.py`, `tests/test_e2e_live_sm1_adoption.py`
(every identity of the adoption chain, without a server), `tests/test_e2e_live_attempt_faults.py`
(stdio, attempt records and read faults of the ledger),
`tests/test_e2e_live_sk1_plugin.py`, `tests/test_e2e_live_panel.py`, `tests/test_e2e_live_traces.py`,
`tests/test_server_runner_absorb_wait.py` and `tests/test_e2e_live_ci_lane.py`;
`tests/test_web_typography_static.py` owns shared-source loading and variable
resolution; `tests/test_e2e_live_sm1_palette_browser.py` exercises the two-document
palette oracle in both themes from either opening scheme, including one-theme edits,
roles left at the seed, missing-source and page-local-role failures. The real
SM1 stub rehearsal is separately gated by `integration`, `serial` and
`OUROBOROS_E2E_DEEP=mock`; it starts a real server and the hermetic suite, so
ordinary focused/default tests must not accidentally launch it.

The `.github/workflows/ci.yml` `e2e-live` job runs only on an explicit
`e2e_live=true` dispatch, never an ordinary dispatch, schedule, push, PR or
tag, and checks out the dispatched SHA (`gh workflow run CI --ref ouroboros
-f e2e_live=true` tests the development tip). It runs one SM1 attempt with
`--self-mod --total-budget 30 --per-task-usd 15`, reserving $30 for its two
roots. A keyless `--stub` SM1 rehearsal on the same seed must succeed first;
it exercises direct adoption, not the later evolution or live reasoning. It
gets no provider key, has a separate output directory and retains its own
results even on failure; a failed rehearsal admits no paid attempt. The live
step keeps `--task-timeout 4500` (75 minutes: at 2400 all four runs hit the
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
