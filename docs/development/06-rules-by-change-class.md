# Rules by change class

This chapter is the imperative body of the handbook: one section per change class, from tool registration and skill payload lanes through the live E2E stand, light mode and deliverables, retention, delegated subagents, cancellation, onboarding and settings surfaces, transport and late-result custody, LLM call rules, timeout and wait control, and the loop and acceptance state machines. Each section names the tests or gates that enforce it, or says that only review does, so a change is checked against the rules for the class it belongs to rather than against the whole book.

`docs/CHECKLISTS.md` remains the only reviewer scorer (its
`development_compliance` item points at this handbook as a whole). The
sections below are imperative rules per change class; each names its enforcing
surface or states that none exists.

### Tool registration and guard surfaces

- A new Tool: `get_tools()` exports it with the `ToolEntry` pattern from
  `registry.py`; an explicit entry goes in `ouroboros/safety.py::TOOL_POLICY`
  (`POLICY_SKIP` for trusted built-ins, `POLICY_CHECK` for opaque or
  outward-facing ones), and the capability class is declared in
  `ouroboros/tool_capabilities.py` (`CORE_TOOL_NAMES`, child profiles,
  parallel/truncation sets). Without the policy entry the tool falls through
  to `DEFAULT_POLICY = POLICY_CHECK` and pays a light-model LLM call per
  invocation. Add a tool to a child profile only when that narrower principal
  should receive it; test schema plus execution behavior rather than mirroring
  names into another catalog.
- A tool that WRITES the repo working tree needs the GUARD surfaces too, not
  only the visibility ones: add it to `_ROOT_ARG_REPO_WRITE_TOOLS` (the single
  set behind the acting-no-workspace fence, the protected-write gate, and the
  acting root-enum narrowing) and canonicalize its target paths — via
  `_PATH_NORMALIZED_TOOLS` for a top-level `path`, or
  `canonical_repo_relative_path` + `_payload_write_paths` for payload-borne
  paths. Visibility checks can all be green while these are missing, so tests
  must exercise the real guard chain, not only a mocked resolver.
- New memory/data files: decide whether they appear in LLM context
  (`context.py`) in the same change.

Enforcement: CHECKLISTS items 2(g) and 10 (`tool_registration`) in commit
review; the public schema/registry contract is pinned by
`tests/test_tool_api_v2_public_surface.py` and the safety-policy fallthrough
by `tests/test_local_routing_and_safety.py`; CHECKLISTS item 11 backstops the
memory/context decision.

### Skill repair and payload lanes

- Start Repair as ordinary managed development carrying the selected skill,
  source request and admitted revision. The UI's Repair and run request is a real
  owner message; its origin follows the ordinary task path. Resolve that source
  when the model enables the repaired skill, never trust a client allow_enable
  flag. Preserve a later direct owner disable; a load-error revert is not one.
  Read existing `skill_repair` records as
  selectors, never as a reduced profile. Preserve normal file, shell, browser
  and delegation tools, with existing readonly/acting-child ceilings.
- Keep installed payloads as ordinary directories; a delegated Git copy is an
  optional existing capability. Check the known revision before an operation;
  after opaque process work, record the observed revision without asserting
  exclusive authorship. No long shell lock or automatic rollback belongs here.
- Use the existing payload binding/policy owners for all path forms. Markerless
  native-directory payloads remain logical external; launcher seeds and
  provenance/review/grant/dependency control state keep their existing guards.
- Review, grants, dependency readiness, desired enablement and actual execution
  are independent facts. Resume an unchanged reviewed snapshot through its
  existing free replay; a dependency or load failure never rewrites the review
  verdict. Preserve explicit owner disable and the original automatic request.
- UI, existing CLI commands and task tools call the shared operation owners.
  Actor identity is host-derived; owner-only actions require a real member
  chat/quiz/mailbox source and exact skill, revision and requested items. The
  model interprets intent; no synthetic reference creates permission. Ordinary
  Repair implies neither grant-all, attestation nor deletion.
- Test the real installed script/tool/HTTP/widget/companion after review and
  prerequisites, repeat after corrections, and inspect a widget screenshot.
  Execution receipts name the actual dispatched revision; they are not PASS.

Enforcement: `tests/test_skill_development_revision.py`,
`tests/test_skill_lifecycle_actions.py`, `tests/test_skill_development_execution.py`
and the UI lane `tests/test_ui_smoke_skill_lifecycle.py`.

### Extension dispatch and isolated dependencies

- `type: extension` skills with reviewed isolated dependency envs must not
  import `plugin.py` or execute handlers inside `server.py`, even when the
  dependency tree looks pure-Python; payload-native marker files (`.so`,
  `.dylib`, `.dll`, `.pyd`) also force child dispatch — containment, not
  admission: a native payload still faces the skill-review checklist. Keep the split
  explicit: no-dependency pure-Python extensions may use `extension_loader`'s
  in-process PluginAPI; isolated-dep/native-marker extensions are cataloged
  and dispatched by `extension_process_runner` short-lived child processes.
- Proxies return normal tool errors / HTTP 502 / WS log messages on child
  crash, invalid JSON, timeout, or abort — a child `SIGABRT` is a handled
  extension failure, not a server crash. Children use scrubbed env, per-skill
  grants and isolated deps, process-group tracking, output caps, and timeout
  cleanup; do not add fallback code that imports native-risk plugin modules in
  the host process.

Enforcement: `tests/test_extension_dispatch_threaded.py`,
`tests/test_extension_isolated_deps.py`,
`tests/test_extension_process_runner.py`.

### Declared skill resources and builds

- Reuse the isolated dependency owner for exact downloads and literal build/check
  argv. New declarations must match a fresh executable review and hash-covered
  specs; revalidate the pinned payload before launching their processes.
- Keep verified resource/package caches outside the replaceable payload/env.
  Record actual resolved versions, resource/output hashes and diagnostics in
  the existing dependency records; package-manager success is not proof of the
  requested function. Only an explicitly declared check establishes that fact.
- Preserve wheel-only and npm ignore-scripts unless that entry opts into the
  corresponding build action. Manual dependencies keep their existing contract.
- Ordinary binary payload resources use the review classifier/descriptors during
  delegated capture too; large downloaded/build resources belong to the isolated
  dependency path, not a giant source patch.
- Go compilation/execution share the existing child owner and timeout. Deno
  flags express current script effects and task network constraints, not new
  grants or a stronger claimed OS sandbox than the other reviewed scripts.

Enforcement: `tests/test_skill_install_resources.py`,
`tests/test_skill_runtime_commands.py`, `tests/test_skill_runtime_lifetime.py`
and `tests/test_skill_payload_binary_transfer.py`.

### Task contract resource policy

- Outside Cyber Pro, `resource_policy.protected_artifacts` is a typed affordance
  policy: execute-only black-box references may run;
  byte reads, copy/hash/static introspection, tracing, and debugging of
  declared paths are blocked.
- Observable Acceptance Claims are bounded, advisory, task-general criteria
  (`id`, `claim`, `surface`, `support`, `priority`); `success_criteria` is an
  input alias, not a second persisted carrier, and
  `effective_acceptance_claims` is the only binder; its read-time semantics
  live in ARCHITECTURE §11.1. An OPEN wave binds nothing and is disclosed as
  `none_open_plan_wave` with the non-binding `plan_claims_exhibit` (declared
  intent, never in the resolvable vocabulary). A child receives only claims
  explicitly passed to its own `schedule_subagent` call. Reviewer
  `evidence_refs` resolve by exact membership in the already-built host
  packet — no fuzzy matching, filesystem reads, or re-execution — and
  resolution changes the clean bit and its disclosure, never actor parsing,
  quorum, or verdict. Do not turn claims into a hard acceptance gate or a
  surface-specific taxonomy.

Enforcement: `tests/test_protected_artifacts_policy.py` and
`tests/test_acceptance_claims_wiring.py`.

### Skill-defined Presence

- Keep behavior portable and authority installation-local: a reviewed
  `presence:` profile declares instructions, context topics, bounded runtime
  defaults, and conceptual tool/script/resource requests — never provider
  credentials, room ids, or one installed tool spelling.
  `presence_capabilities.py` stores the owner's exact selections outside the
  payload and fingerprints the request semantics that authorize them.
- Presence authority is a positive immutable ceiling, not a denylist or a
  prompt promise: admission requires the owner-created binding plus an
  installed, enabled, freshly executable behavior skill and every required
  selection, then freezes skill/profile/state/selection fingerprints, exact
  grants (the profile's selections plus the constant cognitive-memory
  baseline from `tool_capabilities.COGNITIVE_MEMORY_TOOL_NAMES`; a selected
  grant keeps its bindings), argument bindings, runtime slot, and round limit
  into `task_contract.capability_ceiling`; schema discovery and execution
  enforce the same ceiling for built-ins, extensions, MCP tools, scripts, and
  resource roots.
- `state/presence_bindings.json` is host-owned authority: a transport token
  resolves only bindings naming that exact transport skill, and the submitted
  provider/account/conversation/thread must match the binding origin — never
  recover those identities from message text. Staged files stay inside the
  calling skill's state root before entering the ordinary attachment store.
- Run each admitted event with a fresh agent, a deterministic
  binding-plus-source-event task id, the cross-process installation-wide
  concurrency gate, and per-conversation serialization; the transport's
  durable provider custody owns arrival FIFO before Host admission. Do not
  add a transport-specific task scheduler, memory silo, core terminal outbox,
  or resident cross-room agent.
- Completion is exactly `message`, `silent`, `tool_delivered`, or `deferred`
  (deferred requires a successfully promoted `work_ref`; correlated lookup
  stays behind the same transport token and binding, and
  `presence_cancel_work` additionally requires the current binding and
  conversation to match). Promotion and `schedule_followup` copy the Presence
  metadata and capability ceiling by value; any new descendant producer
  preserves this ceiling or refuses the transition — reconstructing authority
  from mutable current state is forbidden.
- Knowledge-topic and scratchpad mutation each use one stable lock, so
  concurrent owner and Presence turns cannot overwrite a newer projection
  with an older render. Test the boundary at both layers (strict
  profile/state/ceiling parsing, stale/missing review admission, schema and
  direct-execution filtering, argument binding, binding/token/origin checks,
  event idempotency and conversation ordering, typed outcomes, late-work
  correlation, promotion/follow-up inheritance); provider adapter E2E is
  separate evidence. Enforcement: `tests/test_presence_admission.py` plus the
  both-layer boundary tests this list requires.

### Devtools isolation

`devtools/` is tracked operator code outside runtime package discovery and the
runtime import graph; runtime modules, `server.py`, web modules, and build
scripts must not import it. Touched devtool files receive normal triad/scope
review; unrelated files may remain manifest-only in broad Atlas packs so
operator code does not drown core review. Generated outputs live in an
explicit external root, never in `repo/` or live `data/`; domain-specific
architecture and methodology live beside the devtool, not in core governance
docs. No automated import guard — review-only (triad/scope review of touched
devtool files).

### Live E2E stand (`devtools/e2e_live/`)

`python -m devtools.e2e_live.run_live_lanes` exercises owner-shaped work on
isolated real servers. `run_live_lanes.py` owns admission, seed/settings, the
lane pool, budget and reports; `scenarios.py::SCENARIOS` owns scenario prompts,
settings overrides and callable acceptance checks; `stub_lane.py` reuses the
loopback model and review answers in `tests/system_e2e/harness.py` for the
`--stub` $0 rehearsal; `ui_probe.py` owns the real-browser client.
Keep this opt-in stand outside runtime imports and default local evolution.

#### Scenario acceptance

Judge durable artifacts and actual consumer observations, never model prose
or an HTTP 200 alone. `LaneContext.check` refuses duplicate keys so a later
task cannot overwrite an earlier verdict; multi-task scenarios use separate
terminal-check namespaces. `--attempts N --pass-of K` records every attempt
and requires K passes for EACH selected scenario.

| Scenario | Work and required evidence | Rationale / limits |
|---|---|---|
| SM1 | Change the shared brand accent consistently with DESIGN.md §3 in `web/ui.css`, exercise the app and setup wizard, then land a reviewed release through `preflight_review` → `commit_reviewed`. The full profile uses advanced runtime and blocking enforcement, with no landing skip flags. Acceptance retains the S2 checks: the commit exists and includes the changed shared palette with nonempty accent/focus roles, VERSION strictly increases, the landed carriers pass `commit_admission.release_metadata_preflight`, the worktree is clean, a real advisory ledger row and `scope_review_complete` exist, usage is positive, and the browser reads the new accent and matching accent/focus roles on both `/` and `/onboarding` after restart. | One shared file does not prove both documents loaded it: the browser oracle detects a missing wizard link or divergent page override. The named accent roles and alpha ladder remain part of the palette. SM1's lane-local release/review/restart contract is separate from a version-neutral contributor PR; changing its source oracle must not remove those obligations. `vision_evidence_present` records browser/vision tool rows for reviewers to judge, not a host assertion that the image was inspected. `committed_companions` records paths beyond the palette, release carriers, DESIGN and comment-only CSS as facts, not an automatic scope failure: reviewers may identify another legitimate accent consumer. The clean-tree check discloses and tolerates only transient `.ouroboros/` scratch. |
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

#### Seed and settings

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

Paid runs use `scenarios.STAND_PANEL_SETTINGS`: Gemini 3.8 Flash / GPT-5.6
Luna / DeepSeek v4 Pro triad, DeepSeek v4 Pro scope, Claude Sonnet 5 advisory;
reviewers at low effort, task/evolution at medium. `--production-panel` selects
the tree's defaults instead; neither choice changes installed product defaults.
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

#### Budget admission and ordering

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

#### Self-modification and browser lifetime

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

#### Reports, focused verification and CI

Keep run roots append-only outside `repo/` and live `data/`. Every attempt
writes `lanes/<id>_a<n>/result.json` and `result_index.jsonl`: checks, facts,
settings SHA and secret-free config digest, seed `git describe`, pre/post HEAD,
diff digest, grants by fingerprint, spend, runtime terminal disclosure, and
screenshots when available. Infrastructure failures retain typed
`refusal {type, code, message}` and `reason_code=infra_error:<code>` in both
result surfaces. Post-stop `/proc` survivors fail a passing lane and name up
to twenty PIDs/command heads with an omitted count; without `/proc` the scan
is explicitly unavailable, never passed.

The watcher reports lane state, spend/cap and free disk on `/` and `/mnt/data`.
Key headroom is an informational probe on its own thread, with an eight-second
HTTP bound, at most once a minute and failure backoff. A failed probe is not
an alert or a delay of the watcher tick.

Focused contracts live in `tests/test_e2e_live_runner.py` (including exact FIFO
feasibility fixtures), `tests/test_e2e_live_sm1_checks.py`,
`tests/test_e2e_live_sk1_plugin.py`, `tests/test_e2e_live_panel.py`,
`tests/test_server_runner_absorb_wait.py` and `tests/test_e2e_live_ci_lane.py`;
`tests/test_web_typography_static.py` owns shared-source loading and variable
resolution; `tests/test_e2e_live_sm1_palette_browser.py` exercises the two-document
palette oracle, including missing-source and stale-focus-role failures. The real
SM1 stub rehearsal is separately gated by `integration`, `serial` and
`OUROBOROS_E2E_DEEP=mock`; it starts a real server and the hermetic suite, so
ordinary focused/default tests must not accidentally launch it.

The existing `.github/workflows/ci.yml` `e2e-live` job runs on its own nightly
03:17 UTC cron or explicit `e2e_live=true` dispatch, never an ordinary dispatch,
push, PR or tag. The cron fires on default-branch metadata but checks out
`ouroboros`; dispatch uses its selected SHA. It runs one SM1 attempt with
`--self-mod --total-budget 30 --per-task-usd 15`, reserving $30 for its two
roots. The owner supplies `OUROBOROS_E2E_LIVE_OPENROUTER_KEY`; its absence
produces the honest green summary `skipped: secret
OUROBOROS_E2E_LIVE_OPENROUTER_KEY not configured`, not a claimed run.
Upload the manifest, index, lane results and screenshots even on failure.
The summary renders verdicts or the typed refusal/error without changing
the stand's exit verdict. Browser PR proof and the keyless system-E2E
schedule retain their separate existing CI owners.

### Light mode and external deliverables

- `runtime_mode=light` is a self-modification boundary (`ouroboros/config.py`
  owns the semantics; ARCHITECTURE "Safety and runtime mode" states why). Pro
  permits protected rewrites; Cyber agency follows the same effective Access
  owner. Light still supports user deliverables outside the Ouroboros repo/control-plane.
- Preferred flow: `task_drive` for scratch, `artifact_store` for canonical
  deliverables, `user_files` for the owner's visible copy.
  `write_file(root=user_files)` and declared process `outputs` register/copy
  canonical task artifacts; rewrites keep the previous canonical artifact in
  non-manifest history with last-5 retention (history is for recovery, not a
  second deliverable list). The logical `root=deliverables` tool stays
  read/list/search-only and is not granted to children.
- Large task files use `artifacts.stream_artifact_file` and atomic
  `copy_artifact_file`; do not read complete datasets into a bytes object.
  HTTP admission and materialization run their complete blocking operation off
  the event loop through `gateway._helpers.run_sync_to_completion`. Cancellation
  waits for that operation before releasing its reservation, input or iterator;
  cancelling an HTTP waiter never means cancelling the admitted task. Multipart
  spool copy and close belong to the same worker so cleanup cannot be cancelled.
  Directory exports keep a complete relative member/size/SHA manifest and a
  streamed ZIP, including outputs above 50 MiB. File changes and missing members
  are explicit capture failures. Reject a read as soon as it exceeds the source's
  initial regular-file size; do not wait for a growing file to reach EOF. A borrowed
  completed multipart spool uses the same copy/hash/atomic owner and descriptor
  checks, without claiming verification of an original pathname; its caller closes
  it only after the copy worker settles, including cancellation. Host path uploads
  retain source confinement and the existing Path return. Automatic genesis listing
  is discovery: record unreadable/changing entries and incomplete coverage, while
  actual capture/copy remains strict. `send_file` uses immutable captured names so
  an earlier delivery URL never aliases a later rewrite. If capture is unavailable,
  existing small-file inline delivery stays available without a fabricated URL
  or reference; source-read refusals and the inline size boundary still apply.
  Registered immutable downloads verify their bytes once per request without
  materializing the whole task result; source and unregistered paths still use
  the existing effective-result owner. Browser URL downloads
  use the existing helper's streaming mode (HEAD then native browser download);
  the launcher URL backend already streams. `downloadBlobViaHostBridge` shares
  the existing bytes-save owner for an already-owned Blob or data/blob URL;
  it never turns an HTTP response/stream into a Blob. Native result/cancellation
  fields are preserved, and old launchers report unavailable saving explicitly.
- Input authority keeps at most 25 rows inline and, when needed, an additive
  `attachment_manifest_ref` in the existing source-handle store. Preserve its
  count/size/SHA and resolve the complete set before child materialization,
  mailbox inheritance, retry or copy-back; never fall back to the preview when
  the source fails. Re-publish a new manifest after rebasing paths. Input files
  remain inputs, outside deliverable inventories. Failed copy-back participates
  in the existing pending-ref retry/GC contract rather than losing child bytes.
  Accepted follow-up inputs retain their exact manifest sources separately from
  the initial task contract. Copy failure retains the owner mailbox as the retry
  source; successful retry releases it through normal terminal cleanup.
- Preserve exact process arguments and the prepared resource binding through
  admission and execution. Quoted examples and unknown interpreter effects are
  not proof of writes. ARCHITECTURE "Safety and runtime mode" owns the full
  source/Supervisor contract; reuse it without a second detector, consent store
  or automatic repetition of a denied operation. Post-execution observations
  annotate the original result and never roll back concurrent owner state.
- `scratch=[...]` is a DISTINCT channel from `outputs=[...]`: ephemeral
  in-cwd verification files, exempt from the undeclared-output guard, never
  registered as artifacts, adopted only with a declaration-time sha through
  the SSOT `artifacts.record_task_scratch`, and excluded from the workspace
  patch via `.scratch_manifest.json`. The guard verifies candidates post-exec
  by stat, so a mere path mention is not a write. Use `outputs` for
  deliverables, `scratch` for throwaway verification — never overload one for
  the other.
- cwd: omitted cwd selects `active_workspace`; a light direct task that needs
  writable scratch selects `task_drive` explicitly; long-running services in
  light use an explicit external/task/artifact cwd, and declared service
  `outputs` are copied when the service stops. Directory outputs become a
  complete manifest plus streamed zip. Policy-rejected members are skipped
  with explicit notes; missing or unreadable members fail the directory copy.
  `run_script` stages workspace scripts in a unique owned directory under `.ouroboros/tmp_scripts`, with a local `.gitignore` written before execution. Raw Git status and patches exclude that scratch without hiding neighbouring user files or changing Git configuration. Each call cleans only its own directory and then empty shared parents; cleanup failure preserves the process result with a warning. Non-workspace scripts keep the task-drive layout and garbage collection, so relative imports, generated files and toolchain
  discovery observe the requested cwd (`ouroboros/tools/shell.py`;
  `tests/test_shell_run_shell.py`).
- Policy denials stay separate from execution failures:
  `user_files_path_blocked`, `cwd_blocked`, and `artifact_output_undeclared`
  are non-failure outcomes; failing to register a declared output remains
  `artifact_output_error`.
- Outside Cyber Pro, the default shell lane carries target-aware git policy: mutating git is blocked only when it targets the Ouroboros
  runtime (bidirectional, casefold, symlink-resolved containment;
  `commit_reviewed` is the remedy for self-repo changes); read-only git works
  everywhere; the network fence still applies; acting `self_worktree`
  children keep the strict no-commit policy. `git init`/`commit`/`push` in an
  external project tree is legitimate task work, not a violation.
- In external workspace mode, light-mode self-repo dirty checks snapshot the
  system repo, not the active workspace; workspace patches are captured
  against the preflight git base. Project-room promotion provisions a
  standalone repo through `ensure_project_workspace` and fails loudly on a
  broken binding or unreadable registry.
- `claude_code_edit` is a retired tool name whose compatibility contract is
  one-way and permanent: a saved task contract carrying
  `disabled_tools=["claude_code_edit"]` also withholds the successor
  `delegate_start` (registry `_disabled_tools`). The successor path is the
  configured session actor — including the exact-payload class via
  `delegate_start(subagent_id=..., prompt=..., root="skill_payload",
  bucket=..., skill_name=...)` — and the api-route advisory successor is the
  bounded native inspection episode (`review_native_episode.py`). Do not
  resurrect the tool name.
- Successor parity rule: a tool may be called replaced, retired with a
  successor, or fully migrated only after a persistent golden test proves
  every user-visible target class the predecessor supported through the
  successor to the final outcome. Deleted-test tombstones prove intentional removal,
  not successor parity; dropping a target class requires an explicit
  owner-approved record naming the lost user outcome.
- Do not recommend `runtime_data/uploads`, skill payloads, or owner state
  directories as generic artifact transport.

Enforcement: `tests/test_v674_light_mode_cwd.py` (cwd selection and what light
refuses), `tests/test_deliverables_layout.py` (deliverable placement and the
output manifest), `tests/test_git_shell_policy.py` and
`tests/test_shell_redirect_guard.py` (the shell surfaces); the
successor-parity and artifact-transport rules are review-only.

### Runtime cleanup and retention

- Age-based GC of disposable runtime artifacts shares ONE owner knob,
  `OUROBOROS_GC_RETENTION_DAYS` (default 7, hard max 365), and the
  cutoff/clamp helpers in `ouroboros/retention.py` (`age_cutoff`,
  `clamp_retention_days`, `get_gc_retention_days`); do not hand-roll cutoff
  math in new prune code. Prune functions keep an explicit `retention_days=`
  parameter for tests; only the default (None) resolution reads the knob, and
  startup prunes are wired from one place (`server.py`).
- If a subsystem genuinely needs its own lifetime, name it
  `OUROBOROS_<SUBSYSTEM>_RETENTION_DAYS` and add it as a fallback in
  `retention.LEGACY_RETENTION_KEYS` — the migration-safe extension pattern —
  but prefer the unified knob. The three retired per-subsystem keys are
  migrated at `config.load_settings`; do not reintroduce them.
- Durable artifacts are NOT age-pruned and stay out of the GC sweep: genesis
  projects (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`) and forensic observability
  blobs (kept compressed indefinitely).
- Review continuations are recovery state, not disposable GC: archive a
  record (collision-safe move, never delete; the archive has no runtime
  reader) only when its owner task is settled, it stayed un-resumed past the
  seven-day threshold, and no recorded obligation remains open; any
  uncertainty or move error leaves the live record intact.

Enforcement: `tests/test_phase3c_observability_gc.py` (the unified knob and the cutoff math) and `tests/test_observability_retention.py` (blob pruning); the review-continuation archive rule has no automated surface — review-only.

### Live subagents

Mechanism — bootstrap branches, zero-run receipts, custody, work orders,
supervision, recovery — lives in ARCHITECTURE "Delegated subagents (Claudexor
transport + the nanny)" and the module docstrings it names. Review gate:
CHECKLISTS items 18 (`subagent_isolation`) and 23 (`delegated_transport`),
both critical. The imperatives:

- Schedule only through `schedule_subagent`; its public schema and the
  handler's closed keyword set are BOTH derived from
  `control.schedule_subagent_properties()` — a hand-maintained mirror is
  correct only until one side gains a parameter
  (`tests/test_tool_api_v2_public_surface.py`). No new
  `contracts/task_contract.py` fields for child needs: the closed capability
  enum declares them, never objective prose; for internal-only options the
  membership test is WHO DECIDES, not who currently calls. Delivery is
  at-least-once — an exact task id with live or durable custody is an
  idempotent no-op; never use semantic duplicate judgement as the physical
  identity fence.
- `subagent_id` selects one complete row from the canonical enabled
  `OUROBOROS_SUBAGENTS` list; freeze the normalized row at schedule time and
  dispatch/restart from that snapshot, never from mutable Settings. No
  second model/lane/executor selector, no host-side ranking, no substitute
  actor after a typed refusal; legacy selectors stay hidden handler-side
  compatibility.
- The typed parent-LLM substrate choice is the floor (truth, money, and
  authorship stay where the parent put them); topology, decomposition, and
  supervision judgment remain the model's ceiling (BIBLE P5/P13). Never
  reintroduce a host-side wait, poll, or supervised-wait in bootstrap:
  waiting is the model's own `delegate_wait` decision, which keeps owner
  messages, hurry controls, checkpoints, and parallel auxiliary children
  live for the whole run.
- Grow `subagent_bootstrap._DEFINITE_UNRUN_REASONS` only with reasons that
  PROVE no run can exist; everything ambiguous wakes the model (why: `docs/architecture/06-agent-core.md`
  § "Delegated subagents (Claudexor transport + the nanny)"). Zero-run receipts
  write only
  `incomplete | unknown` (a zero-run "complete" is unverifiable
  self-report); a substrate swap is a disclosed incomplete execution, never
  a silent vendor/API fallback
  (`tests/test_configured_session_prestart.py`).
- Work orders carry the complete chosen assignment and host authority without
  a compiler-size cutoff or compulsory question/file transport. Preserve the
  instruction roles and avoid duplicating objective/output inside the host
  authority. Real transport/provider refusals retain their cause, original
  input and execution custody; recovery is the model's choice. Legacy partial
  runs keep their exact renderer, source-range validation and stored-body retry.
  Operative plan text and canonical identity remain complete. Keep full specs
  in existing task source handles and only their references in the bounded
  review-state index; restore them for acceptance and plan comparisons.
  Redacted review evidence never substitutes for the original requirement text.
  Access is stated ONCE, by the host, from the typed run shape: `_host_instructions`
  renders `delegate_start_instructions.access_instruction(shape.access)` as one
  sentence that names the profile and says it governs. Assignment prose about
  access is CONTEXT, never authority, and is never parsed: a parent's prose ban
  that contradicted the derived profile once cost a run and a review cycle.
  `subagents.route_health`
  is the ONE route reader for every consumer; quota readers project one
  `ClaudexorGateway.quota_state()` envelope
  (`tests/test_available_subagents_runtime.py`). A fully-used ratio without a
  valid future reset must neither refuse dispatch nor certify available quota
  in the UI; retain explicit active cooldowns and leave final admission with
  the engine. Substrate facts and the
  packet's per-skill lifecycle facts are VISIBILITY ONLY — acceptance judges
  quality, never the execution route — and an unreadable custody log reads
  `evidence_read_failed`, never a proven-empty substrate.
- The coordination poll is READ-ONLY of task state: it observes the budget
  profile and snapshot rather than resolving and latching them, so a poll can
  never change its own next answer, and it writes nothing of its own beyond the
  canonical usage-ledger reader's bounded maintenance (what that maintenance
  is, and the torn-quarantine residual #586: the `delegate_supervision.py` row
  of `docs/architecture/01-high-level-architecture.md`; `usage_attempts.lock`
  recovery after confirmed owner death is immediate, unknown metadata retains
  the 90 s grace, and the caller's 45 s budget is unchanged). Every ledger
  state, including absence, uses the canonical reader.
- `task_constraint` boolean parsing is strict (`"false"` is false); deadlines
  only narrow, delegation budgets only reduce, absent depth requests stay
  unknown rather than inferred from prose; preserve the persisted
  requested/permitted/attempted/achieved depth facts and never recompute
  historical permission from current Settings.
- `active_tool_profile` fails closed to read-only, never to
  `self_modification`/`operator_control`. Ordinary external grants remain
  deny-by-default and ordinary acting-tool restrictions remain. Cyber acting
  children use the actual registered capability catalog; a static inherited
  name exclusion or default-empty grant list must not hide an existing tool.
  An explicit read-only assignment remains read-only. Only
  `schedule_subagent` may create subagents (forged `delegation_role`
  rejected at API/CLI ingress); live `memory_mode=shared` stays disabled
  (`tests/test_acting_subagents.py`). The subagent browser boundary refuses a
  target DNS cannot classify (a typed `BROWSER_POLICY_UNAVAILABLE` before
  navigation), allows loopback except actual or expected Ouroboros
  control-service endpoints by identity — an unverifiable expected endpoint is
  refused, never treated as foreign, while other ports and reused pathnames
  keep working — admits private origins only through host-established
  `resource_policy.allowed_origins`, and re-checks every redirect hop before
  returning page content (`tests/test_browser_url_policy.py`,
  `tests/test_browser_isolation.py`, `tests/test_browser_redirect_chain.py`;
  mechanism: ARCHITECTURE "Tool capability and execution"; full rules:
  CHECKLISTS item 18).
- The parent is the SOLE committer of the live body: acting children return
  a `workspace.patch`, the parent applies a chosen patch with
  `integrate_subagent_patch` and runs its own `commit_reviewed`. The shared
  `external_workspace` surface, including ordinary folders, verifies and records without re-applying; a
  genesis project is durable because the project directory IS the
  deliverable. A genesis project starts without a `.gitignore`, so its small
  text build output (`dist/`, `build/`) rides the `workspace.patch` record
  until the project declares one — a disclosed residual, bounded only by the
  per-file source-patch size boundary and git's binary verdict; otherwise
  eligible large/binary outputs are preserved as file artifacts with a full
  manifest, independently of the source patch. There is no total source-patch cap. The canonical/replica terminal field-custody projection is
  ONE pure reducer reused by copy-back and effective reads — every change
  adds a stale-replica regression at BOTH seams
  (`tests/test_available_subagents_runtime_review_fixes.py`). Do not broaden
  generic data-tool behavior while fixing subagent isolation
  (`forward_to_worker` writes only to validated running tasks in the
  current task/root lineage).
- The DELEGATED Git/payload lane is the other half of that rule: it edits a
  private execution snapshot and reaches a tree only through
  `integrate_delegated_patch` (the apply/reject authority and the orphan
  containment relaxation: `docs/architecture/06-agent-core.md` § "Delegated
  subagents (Claudexor transport + the nanny)"). One predicate
  (`delegate_shared.orphan_apply_target_ok`) serves the apply gate, the health
  invariant and the tool description; every other guard (owner terminality,
  top-level principal, proven drift, protected paths, staged-never-committed)
  is unchanged (`tests/test_delegated_run_isolation_orphans.py`).
  Snapshot tests compare raw LF/CRLF inputs under Git checkout filters, require
  zero patch before child edits, and retain normal Git apply semantics afterwards.
  Copy failures or a source change against the baseline leave no registered
  snapshot or pinned ref (`tests/test_snapshot_file_inputs.py`).
- Outcome honesty: a delegating parent must not produce a clean no-tool
  final answer while direct children run undecided — one bounded absorption
  reminder, then best-effort (`children_unabsorbed`); while that gate is
  open the delivery candidate is HELD, and the delivery-control instruction
  never rides the reminder round, which would contradict the required
  disposition tool call (`tests/test_v6570_swarm_honesty.py`). `wait_tasks`
  stays batch-compact; `control_task_results._wait_for_tasks` owns its
  projection, documented under ARCHITECTURE's "Waiting on children".
  The model result and optional `terminal_host_notice` remain separate;
  full untruncated handoff belongs to `get_task_result` and `wait_task`; no
  shared ledgers, automatic memory merges, or new settings/endpoints unless
  the accepted plan calls for them. Push/live events are wakeups, not
  terminal authority — lifecycle changes must exercise lost/reordered
  terminal frames and reversed snapshot completion.

For ordinary-directory sessions use the engine-owned file work product: preserve the parent-selected direct/copy strategy and footprint through actor-first startup, with stable source identity separate from actual execution root. Never initialize Git or change run mode as a workaround. Native writes preserve known successful file postimages through the existing artifact owner; do not replace this with a full-tree scan or success-text parser. Keep copied binary/large inputs outside the target Git object database with exact baseline identity. Capture and apply include both source patches and full file artifacts, including deletion; an empty text diff cannot certify no work. Apply selected results through the existing engine CAS and durable intent/key, retain unselected results, and never relabel direct effects or discard as an undo. Test actual file handlers through child copyback/reopen, mixed and file-only apply, binary input preservation, concurrent edits, and lost apply receipts.

### Cancellation and effective status

Mechanism — durable intents, the claim/generation fence, the one settle
owner, owed terminal delivery, cascade postconditions — lives in ARCHITECTURE
"5. Supervisor Loop". Enforcement: `tests/test_cancel_intents_phase_a.py` and
`tests/test_cancel_cascade_v664.py`. The imperatives:

- Effective task status belongs in `ouroboros/task_status.py`; never duplicate
  child-drive merge or terminality logic in gateways/tools. Task waits use
  `SETTLED_STATUSES` and structured facts plus queue-heartbeat freshness —
  never keyword matching.
- `wait_task` and `wait_tasks` also peek the waiting actor's own mailbox through
  the existing transport-wait reader. A pending message returns control without
  acknowledging it or stopping children; the ordinary round-top drain delivers
  and acknowledges it. Read the actor's execution drive, not its budget root.
  One wait/transport episode may retain only a successfully proved empty mailbox:
  compare both mailbox/ACK fingerprints before and after the existing full reader,
  plus execution root, task, attempt and seen ids. Read/parse/stat failure or torn
  data is not proof; never cache it. Check the in-memory incoming queue every tick.
  A changed source re-enters the full revocation-aware reader; no TTL or ACK in peek.
- Terminal quiz reconciliation closes the paired wait even if the answer arrived
  before worker capacity was granted; keep the answer and source unchanged. A
  failed loop without captured evidence reports unknown counts through the existing
  summary/outcome/metrics producers. Never infer zero work or read an unverified
  checkpoint to fill the gap (`tests/test_autonomy_review_fixes.py`).
- Cancellation observations use `task_status.observe_cancellation_target` before
  the existing intent write. They name the resolved physical target, separate
  task-result update/start facts from queue freshness, and optionally include
  recorded delegated execution. These are separate source observations, not an
  atomic snapshot. Caller reason and request origin are distinct; an HTTP client
  is not proof of personal owner intent. A later target mismatch is disclosed.
  Keep cancellation authority and completion-wins independent of these observations.
- Cancel INTENT is never a status value: every cancel ingress writes a durable
  intent through `ouroboros/cancel_intents.request_cancel`, fails closed when
  that write fails, checks live physical ownership (a settled RESULT does not
  mean a dead WORKER) and keeps the recorded scope widen-only — the skeleton is
  ARCHITECTURE "5. Supervisor Loop", the invariant "10. Key Invariants" 14.
- Natural completion WINS a late cancel — discarding is the parent's separate
  explicit `discard_child_result` — and timeout reaping is NOT a cancel ingress
  (both: the same section).
- The intent and delivery registries read STRICT to rows and `task_done`
  validates through the DURABLE result unconditionally (ARCHITECTURE "10. Key
  Invariants" 15 and "5. Supervisor Loop"); only `interrupted` keeps its
  restore-path exemption, and the legacy `cancel_requested` status survives on
  a read-path only.
- `stop_policy` is an axis on the durable intent and the owner hurry control is
  a typed TASK-LOCAL owner-mailbox control, never a chat message, a global
  settings mutation, or a review-gate weakening (semantics: ARCHITECTURE "5.
  Supervisor Loop"; the `owner_hurry.py` and `task_hurry.py` rows of its §1).
  Every same-id requeue producer calls the ONE shared
  `owner_hurry.retry_reset`; the durable hurry projection writes only through
  `update_json_locked` on the `owner_hurry` keys, never `write_task_result`; UI
  surfaces share `web/modules/task_control_menu.js`. Queue-owned hurry
  admission initializes only an absent pooled result through the task-result
  writer's atomic `create_only` branch, which preserves any racing stored row
  byte-for-byte; direct turns stay outside this initialization, and no
  model/start/cost/grant facts are inferred from the click.
- Code owners stay narrow behind one public queue/lifecycle surface:
  retry-aware target/subtree-liveness in `supervisor/queue_transitions.py`,
  capture-miss terminalization/publication in
  `supervisor/cancel_publication.py`, owner-stop delivery/validation in
  `supervisor/owner_stop.py`.
- Keep agent in-band cancel and the existing periodic cancel/delivery/ref sweep
  off supervisor drain. Reuse durable intent claims and generation checks; local
  in-flight keys only deduplicate dispatch and release on failure. Preserve the
  existing cadence and HTTP response contract. Do not queue unrelated Stop work
  behind a new general-purpose file executor.

### Onboarding and Settings surfaces

- Current tasks use the existing task-entry settings read view; next-task saves
  must not change an overlapping direct actor's Supervisor, Review, model or key.
  Owner writers and grant classification read current disk state. Keep document
  values and environment presence separate in the memory-only snapshot; preserve
  immediate effects, explicit task overrides and boot pins. The OOP extension
  payload carries only permitted typed values, never the whole snapshot. Test
  actual child dispatch, unchanged empty/absent values, and current grant checks.
- Cyber can select context, review scope/enforcement, models and Supervisor
  configuration through the existing settings writer. Use effective Access for
  that authority; retain task snapshots, restart-bound access, install-time
  provenance and honest write receipts. Permission is not a review verdict.


- One five-step wizard serves subscriptions, API keys and mixed installs (its
  steps and the completion transaction:
  `docs/architecture/02-startup-onboarding-flow.md`). Quick Review & start runs
  the same proposal compiler for skipped steps; Finish atomically commits the
  visible draft. Only declared raw-model sources can supply Main; an Agent-only connection cannot invent one.
  Subscription copy says "without an API key", never guaranteed free. Provider
  credits/spend settings are not enabled or changed by connecting an account.
- Settings validates the complete draft before any Save request; never omit an
  invalid custom-key row and save the remainder. Keep pure draft collection and
  dirty comparison separate from field-error painting. A failed save/refresh
  preserves edits; leaving or reloading a dirty draft asks before discarding.
  Preserve saved/unsaved/unknown write receipts and the independent owner-only
  endpoints. Tests: `web/tests/settings_validation.test.js` and the real
  `tests/test_ui_smoke_settings_drafts.py` consumer.
- Model-role and actor/reviewer adapters use `model_chooser.js` (its contract:
  ARCHITECTURE "Navigation and shared UI contracts"); the chooser owns
  suggestions/keyboard/position only, never route identity or entitlement.
  Update options in place and dispose bindings before replacing inputs. Tests: `web/tests/model_chooser.test.js`,
  `tests/test_model_chooser_browser.py`, `tests/test_subscription_role_routes_browser.py`.
- Models, actors and reviewers share source/model/account controls. Preserve
  exact pins on ordinary save/reload and catalog failure; a source's credential
  harness comes from its metadata, never an assumed equal name. A referenced
  reviewer stays a native-inspection actor, not an inline packed review.
- One capability, one section: the task-actor story lives in Agents →
  Available subagents (`web/modules/subagents_settings.js`), editing one
  canonical `OUROBOROS_SUBAGENTS` object (list-level Enabled, at most ten
  stable rows, one prose field `recommended_use`; id and compatibility name
  stay automatic and hidden). Never derive durable identity from the visual
  ordinal — removing a preceding row must not change task snapshots or
  receipts — and never render a second control over the same settings key
  (`OUROBOROS_MAX_WORKERS` stays in Advanced because it sizes the process
  pool). Share only neutral route/model/account/effort/status primitives with
  reviewer rows (`route_editor_primitives.js`); task routes serialize
  `api_model` + `credential_profile_id`, reviewer routes `api_chat` +
  `profile_id`; an empty managed-model/session pin means engine rotation;
  saved-but-undiscovered choices stay visible and editable; a compound effort
  slug plus a conflicting separate effort is a validation error, never two
  applied efforts.
  The Auto-lane account preference and its one-request suppression after a
  typed refusal are `docs/architecture/06-agent-core.md` § "Caller-owned
  subscription model calls"; a prospective pricing copy of that request reads
  the fact without spending it, pin never rotates, and preference suppression
  is never turned into a retry or cooldown (`OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL`
  and `OUROBOROS_FALLBACK_COOLDOWN_SEC` keep their existing escalation budget).
- Saved intent, generated drafts, and live status are different axes: a
  status/catalog failure annotates a loaded row and never erases it; GET may
  return an unsaved candidate but only explicit Save or onboarding completion
  materializes it; a late preview updates a clean generated baseline only,
  never absorbing owner edits or dropping focus/caret.
- Owner switches expose the semantic choices the owner can actually make: for
  `OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS`, Settings presents Off / Auto / On —
  Auto IS the unset, surface-aware runtime-mode default and saves the empty
  value (semantics: `config.get_allow_mutative_subagents`).
- Onboarding completes in ONE transaction and `GET /api/onboarding` must never
  persist (the transaction, the install-time proofs and the 503 unknown
  outcome: `docs/architecture/02-startup-onboarding-flow.md`). There is no
  second completion path on any host, and the client treats only the exact
  success envelope (`ok`/`runtime_mode`/`restart_required`) as completion: a
  2xx whose body will not parse is a failure the wizard shows, because a silent
  success discards the restart receipt.
- There is ONE wizard host, the served `GET /onboarding` page; do not
  reintroduce a pre-server or inlined copy. The frame is sandboxed WITH
  `allow-popups allow-popups-to-escape-sandbox` — a sandbox without those
  tokens blocks the sign-in click silently — and this is asserted
  behaviourally from the login card's own markup
  (`web/tests/onboarding_overlay_sandbox.test.js`). Onboarding and Settings
  share the setup contract; diagnostics must account for unsaved in-memory
  wizard values.
- Install-time defaults are compiled from LIVE discovery with typed refusals,
  never guessed, never half-applied, never re-derived after onboarding; install
  time is the conjunction of three proofs stated in
  `docs/architecture/02-startup-onboarding-flow.md`. A once-only decision is
  never taken on a moment-in-time reading (a spent subscription window stays in
  the preset), and the `next_up` verdict is read dual-wire (unified
  `accountPools` first, legacy per-harness second, never re-derived from the
  profile list; an unknown kind is a fail-safe refusal).
- Agent sign-in consumes the harness row's `setupLogin` field as four states
  (absent = legacy catalog; null = the pinned engine's typed
  `setup_job_admission`; a valid object selects `in_app` or
  `external_terminal`; malformed present data is a gap); never add a
  harness-name branch for this choice. External-terminal recovery binds its
  argv to the live handshake's exact engine identity and requires the fresh
  `--probe` to advertise `setup_attach`; render the argv through the owning
  `claudexor_daemon.py` consumer and never execute the text.
  Credential-profile DELETE remains a thin receipt-preserving proxy; mirror
  additive response fields in Python TypedDicts and
  `web/modules/api_types.js`, and extend field parity plus fixtures together
  (`tests/test_gateway_parity.py`).
- Owner settings writes go through `gateway/owner_settings.py` (the
  lock-as-precondition and `CommitBoundary` contract:
  `docs/architecture/01-high-level-architecture.md` § "Gateway Boundary v1");
  pre-commit refusals answer through `unsaved_error`. `owner_write_guard`
  belongs only on endpoints that call
  `_owner_update_settings` — directly with a transform, or through
  `_owner_write_settings` with a whole document; anywhere else it translates
  unraisable exceptions while advertising a lock the endpoint never takes.
- A setting only an ENDPOINT may author is disk-only in BOTH directions:
  `config.ENDPOINT_AUTHORED_SETTINGS` is consulted by the loader and by the
  environment projection, and the generic save's merge skip-list reads the
  same set — blocking only the request body is not enough, because an
  env-suppliable install-time fact closes its own window before the endpoint
  runs.
- A control the owner cannot use is worse than none: with no agent
  subscription the panel shows truthful configured/generated API or local
  actors and the session chooser points at Accounts instead of inventing a
  route; a saved unavailable session stays visible, and dispatch returns its
  typed refusal, never an API fallback. Harness lists come from one catalog
  path (`accountRows` over `/api/claudexor/status`; dual-shape
  unified/legacy; pins via `indexProfilesByHarness`, a projection of the same
  rows, so a pin option is called what Accounts calls it — `accountName`:
  display name, else the observed email, else the id — with the stored id
  appended only when it differs).
- Install compilation stays linear and split by semantic owner (the compiler's
  emission rules: `docs/architecture/02-startup-onboarding-flow.md` and
  `ouroboros/subscription_install_presets.py`); API-only/local-only compilation
  performs zero Claudexor reads; never fabricate diversity or build a
  harness/account/model powerset. `POST /api/onboarding/subagents/preview` is
  the read-only compiler surface; completion commits the visible owner-edited
  value.
- Owner-facing copy says "agent", never "coding agent" — the same
  subscriptions build presentations and run arbitrary tasks; product names
  (Claude Code, Codex, Cursor) are trademarks and stay as they are.

### Transport and late-result custody

- `LLMClient.chat` and `chat_async` accept optional `stream=False`, `caller_deadline_ts` (Unix seconds) and `caller_execution_deadline` (the existing quota-adjusted monotonic clock). Main opts into streaming. Subtract finalization reserve once at the caller; every physical recovery send re-checks the inherited bounds. Unset deadlines keep ordinary transport defaults. A socket-phase timeout is not an overall wall-clock promise, and late paid completion retains its original attempt.
- Stream consumption completes inside physical accounting; the assembler
  doctrine — strict about completeness, tolerant about form on the Chat path,
  post-terminal shape rejected on the native Messages path, unknown outcome
  reserved for a stream that never reached its terminal frame — is
  `docs/architecture/06-agent-core.md` § "Review delivery". Preserve indexed
  tools, native signatures, complete final framing and cumulative usage
  snapshots. An EOF/error/cancellation retains private wire evidence and cannot
  produce a usable partial answer. Only a structural parameter rejection uses
  the existing wire recovery; never infer a retry from missing stream text or
  ping cadence. Compatible async tool calls use the same normalizer/validation
  path; local, GigaChat and Claudexor retain their separate wire contracts.
- Late reviewer reuse resolves the exact operation's complete producer receipt
  from existing CAS (the binding it must carry: the same section); the current
  surface remains the sole wave writer and reducer, late plan feedback attaches
  as an exact-operation historical supplement with paid settlement recorded
  once, and no source file existence, preview or matching prompt prose alone
  grants authority; missing/partial/error/mismatched custody never buys another
  same-operation dispatch.
- Managed unknown-outcome recovery uses the existing network-wait owner with
  non-generating upstream observations (what proves recovery and what cannot:
  the same section). Keep old outcome/cost unknown and apply current
  budget/Stop/deadline before dispatch; a control-channel outage first rejoins
  the same accepted operation. No scheduler, provider/model table, paid
  readiness probe or automatic manual-restart recovery is introduced.
- `delegate_wait` supervision's observation beat is separate from its HTTP read
  allowance, and a typed read-only-retryable transport failure is a quiet
  observation hole, not a wake (the per-class reasons and the once-per-episode
  owner line: `docs/architecture/06-agent-core.md` § "Delegated subagents
  (Claudexor transport + the nanny)"); no durable counter or outage latch is
  kept. Received auth/protocol failures and owner controls remain meaningful. After terminal cleanup, use the current custody host notice alongside the original answer/narrative. Genuine builtin refusals publish typed non-success at their producer; successful warnings and existing review/Git warning buckets keep their semantics. Acceptance JSON validity and completion cleanliness remain separate decisions.

Focused regressions: `test_review_late_cas_recovery.py`, `test_delivery_control_lineage.py`, `test_terminal_custody_notice.py`, `test_delegate_observation_transport.py`, `test_delegate_hold.py`, `test_configured_session_wake_rail.py`, `test_health_invariants_ownership.py`, `test_transport_b_stream_deadlines.py`, `test_llm_wire_corpus.py`, `test_transport_unknown_continuation.py`, `test_builtin_refusal_results.py` and `test_v671_acceptance_convergence.py`. Use the ordinary isolated preflight runner; full provider/renderer smoke remains separate from local fake-provider evidence.

### LLM call rules

- Claudexor model calls are a transport, not delegated reasoning. Keep model
  content and native continuation byte-faithful through the purpose-bound engine
  operation; never inject its credentials, run its tools, compact inside the
  adapter or silently repeat a generation. Recover a local connection loss using
  the same operation ID; record unknown outcomes as unknown. ACK only after the
  existing private CAS owns the exact result. Failed-response capture uses the
  operation catalog's optional query, frozen before create and reused with the
  same idempotency key; absence preserves the strict legacy result shape. Keep
  full received bytes/exception chains private and compact diagnostics in the
  ordinary problem context. A known terminal with unusable output is a settled
  provider result plus local rejection, never unknown or not-dispatched: preserve
  both existing stream-rejection markers across sync, async and process boundaries.
  Local rejection must not rotate accounts. Optional host hints must be chosen
  by their caller according to transport capability; explicit unsupported options
  refuse, rather than being silently removed and retried. Record submitted model
  options beside the engine's applied options on the usage row; an absent report
  stays unknown. That recorded state covers every submitted option, while the owner line
  speaks only for the thinking horizon: the first changed reasoning effort of each model in a
  task emits one typed owner line (keyed by task and model, never per round) naming only the
  route that reported those applied options. A mismatch is disclosure, never a dispatch gate.
- The engine's active-turn token is one of those transport facts, so the CALLER
  owns its slot (`llm_claudexor.ModelTurnState` on the loop context, a wake-scoped
  one in Background Consciousness) and the engine boundary is its only writer.
  Give a new caller a fresh slot when its logical turn begins and clear it when
  its dispatch leaves this transport; never derive the turn from message roles,
  prose or the last stored assistant envelope (BIBLE P5), never persist the token
  into a checkpoint — a cold restart starts empty — and never let a reprepare,
  thread offload or kwargs copy fork the owner. Update it from a dispatched
  durable result of a request that actually carried the field — a legacy-shaped
  exchange is silence about the turn, not proof one ended — and never put it in
  usage, events, progress or task cards. Gate opting in on the version proven by
  the last SUCCESSFUL engine handshake, not on the next-spawn pin and not on a
  liveness projection a failed probe can blank, so concurrent status polling
  cannot change the shape a running caller sends (mechanism: the
  `llm_claudexor.py` docstring; ARCHITECTURE "Caller-owned subscription model
  calls").
- Pass model_role and the captured account explicitly at every helper/reviewer
  seam. Main and Light may have identical model names and different pins. Account
  context evidence stays source/profile/fingerprint-bound. Manual context sizing
  is not scope authority; an explicit scope ACK must bind the actual route.
  Never promote a changed model's token-density observation into the old model's
  evidence. Physical attempt limits may return a claim only after a successful,
  positive never-dispatched release; unknown or dispatched claims remain charged.
- Resource refusals wait inside the live call before helper catch-all blocks.
  Use the existing task owner, mailbox, clocks and settings writer; no parked
  rounds, compensation processes or replay of completed tools/reviews. Preserve
  typed errors through the tracked image child. The shared waiting card has
  revision fences and distinguishes accepted, applied and saved; a browser fixture
  must not invent a different acknowledgement protocol than the real ingress.


Accounting and transport mechanism — attempt lifecycle, pricing lookup, lock
discipline, snapshots, projections — lives in ARCHITECTURE "Budget tracking"
and "Usage ledger substrate vs. accounting policy"; route contracts (Chat
Completions dialect, request-wire driver, Anthropic native custody) are owned
by "Provider Independence" above. Call-site imperatives:

- New LLM calls go through the shared `LLMClient`/`llm.py` layer — no ad-hoc
  HTTP clients or direct provider SDKs outside it (review gate: CHECKLISTS
  item 2(e)). Exception: skill/extension `plugin.py` modules may call
  providers directly until a host-mediated bridge lands; runtime callers
  inside `ouroboros/` must use `LLMClient`.
- Keep canonical messages/tools provider-neutral and function-shaped; a
  provider dialect is an outbound projection plus inbound normalization only
  and must not mutate stored history or create a second compaction/replay
  contract. Custom-origin receipts stay private and catalog-bound
  (`ouroboros/request_wire_custom_validation.py`); wire-dialect recovery
  uses the one request-wire driver, ladder ordinals stay fixed at 1/2/3,
  custom→function is never persisted as learned dialect, there is no
  Responses migration, and owner `none` on direct Anthropic is
  `thinking.type=disabled` (`tests/test_request_wire_contract.py`,
  `tests/test_openai_chat_custom_contract.py`,
  `tests/test_anthropic_native_custody.py`). `usage.request_wire` describes
  one call's terminal candidate; nested aggregation preserves ordered
  `request_wire_history` with explicit omission accounting.
- Every core-mediated physical provider send goes through
  `usage_accounting.execute_physical_attempt[_async]`
  (`tests/test_usage_accounting.py`; the attempt lifecycle, what may release a
  dispatch, and the unknown/unmetered rule for external skills:
  `docs/architecture/06-agent-core.md` § "Budget tracking"). Custody
  classifiers use the
  explicit `__cause__` chain, never Python's implicit `__context__`; an
  ambiguous timeout remains unresolved (`tests/test_transport_custody.py`).
- Hold the usage-ledger cross-process lock only for budget check, validated
  append, and fsync — never over network I/O (the lock discipline and
  failed-settlement custody: the same section). Callers that own a finalization reserve pass it explicitly so
  admission and the transport bound cannot disagree.
- Keep root ceilings explicitly unreserved under the shared pool. Persist the
  actual applied global limit and its supplied source/revision on the same
  physical attempt through every transition; a missing revision is unknown,
  never the current settings file hash. Pacing facts reuse the existing note
  cadence and cached money projections. Count typed tool results incrementally
  on the loop usage carrier; reported durations are overlapping observations,
  not inferred sleeping/polling time or a new behavior gate. Tests:
  `tests/test_budget_resource_facts.py`.
- Tree-spend pacing decides on root-subtree ledger spend including in-flight
  holds, publishes the same `CostCeiling` object the loop decides on, and
  prices the wrap-up with the fence's own cache-aware reservation (the
  threshold resolution, the root carrier, the wallet read and why a check
  reserves no share: `docs/architecture/06-agent-core.md` § "Budget tracking";
  `tests/test_network_budget_wallet.py`). Keep explicit disabled profiles and
  real monetary fences independent, and read the configured global budget
  through the one resolver rather than an inline default, so the loop axis, the
  bound scope and the ledger fence cannot disagree about the same install.
  Post-task consolidation/synthesis reads one frozen `usage_breakdown` snapshot
  per root subtree (never `$0` on a read failure); no second ledger, no
  reconciliation LLM call.
- Runtime notices after the first user/assistant/tool turn are user notices
  (`[SYSTEM NOTICE]`), not new `role=system` messages; `LLMClient`
  defensively demotes non-leading system messages at the provider boundary.
- **Cache-friendliness invariant.** Byte-stable governance and task contracts
  precede mutable evidence; never place timestamps, hashes, counters, or
  task identity in a stable cached prefix — they fragment provider caches
  while conveying no stable policy. Builders declare bare breakpoints
  (`review_substrate.assert_cache_breakpoint_cap` keeps the count at four or
  fewer; `tests/test_review_prompt_caching.py`); only
  `LLMClient._normalize_payload_cache_ttl` finalizes the assembled wire
  payload. Prompt-cache support stays deliberately narrow — no provider
  hops, body rerouting, or a generic cache/retry framework. A tool-less
  variant of an otherwise identical request rebuilds the whole provider
  prefix, and a `tool_choice` change rebuilds the messages tier, so a
  wrap-up call keeps the schemas, the server-web flag and `tool_choice`
  identical to the working round and instructs in text instead.
  `context_fit.seal_task_transcript` owns the single message-side
  breakpoint -- the task message until the rolling tool-result seal
  qualifies, migrated in the same call -- preserved on the
  direct-Anthropic lane by `_anthropic_blocks_from_content` and on
  OpenRouter by `supports_message_cache_control`, and pinned by
  `tests/test_review_prompt_caching.py`. The main loop declares an
  execution-scoped cache affinity only for subscription transport; API-compatible
  lanes retain their prefix-derived session identity;
  `review_substrate.assert_cache_breakpoint_cap` covers only the review
  builders. Between the sends of ONE execution the transcript is append-only —
  compaction is the one sanctioned rewrite, and OpenAI-family caches discard
  the whole conversation on any other break (why, and the
  `prompt_prefix_break` record: the `transcript_prefix.py` row of
  `docs/architecture/01-high-level-architecture.md`, issue #906). The
  per-round acceptance observation is therefore an append-only row and
  `_append_or_merge_user_content` never merges into it;
  `tests/test_transcript_prefix.py` pins the loop-level invariant on the real
  `run_llm_loop`. Review gate: CHECKLISTS item 22
  (`cache_friendliness`).
- Provider fallback is disabled only when the transcript carries a SEALED
  reasoning artifact
  (`ouroboros/reasoning_artifacts.py::transcript_has_sealed_reasoning`),
  because only a sealed artifact is bound to the endpoint that minted it;
  readable reasoning stays failover-eligible for every family so one
  endpoint's outage does not strand valid work
  (`tests/test_llm_provider_routing.py`).
- Delegated agent sessions and the native review inspection episode preserve
  the full governance prompt; do not truncate
  BIBLE/ARCHITECTURE/DEVELOPMENT/CHECKLISTS to fit argv or transport limits.
- Delegated (subscription-harness) work is accounted on its OWN ledger row —
  `usage_accounting.record_subscription_session`, never
  `record_unmetered_external_dispatch` (it drops the sessions/quota axis); the
  four cash cases and the `input_token_usage` validation rule are
  `docs/architecture/06-agent-core.md` § "Delegated subagents (Claudexor
  transport + the nanny)" (`tests/test_gateway_usage_accounting.py`,
  `tests/test_delegated_run_custody.py`); token `None` means no harness
  reported it, not a run that used zero. Skill Review waves attribute every
  canonical usage row with the exact wave/slot identity; pre-marker waves stay
  "exact attribution unavailable" and are never reconstructed by time/model
  (`tests/test_skill_review_usage_accounting.py`).
- `cost_final` on a projection is a COUNT of open rows (`non_final_rows`),
  never a truthiness test on a dollar sum. A spent subscription window is
  `subscription_window_exhausted` — a TRANSIENT class carrying `reset_at` —
  never folded into `quota_exhausted`, which is correctly permanent for a
  billing refusal and wrong for a window whose only cure is waiting
  (`tests/test_reviewer_slot_config.py`).
- Classify provider failures before retrying the same request. The combined
  Anthropic input-plus-max_tokens rejection is a context-window overflow;
  preserve true output/body-size precedence in the shared context_budget
  classifiers, without requiring the input alone to exceed the window.
  quota/auth/billing, hard bad-request, and request-too-large failures are
  non-retryable as-is (record the exact category and surface a recovery
  hint); a typed 408/429/5xx or a failure proven pre-dispatch may retry; a dispatched request with no terminal provider outcome stops same-model and
  cross-model sends until reconciled. Which callers may repeat a request after
  a typed transport death, how many times, on whose ledger row, and what ends
  a round that holds such a record are stated once in
  `docs/architecture/06-agent-core.md` § "Context fitting, retry, and
  compaction" (`tests/test_transport_death_retry.py` pins the interactive
  primary rail). Call-site rules: decide the `retry_same_request` flag before
  the durable row is written — only a proven refusal (deadline admission
  `llm_not_dispatched`, deadline backoff `llm_retry_deadline_exhausted`, or a
  current typed finalization control during the paid-repeat wait,
  `finalize_control_pending`) takes a never-sent grant back off the round
  record; reuse the interruptible sleep and execution mailbox/current-intent
  readers, peek without delivery or ACK, and leave ordinary
  input/hurry/revoked controls alone; generic transient/empty-response
  backoffs retain their own contract; a budget refusal does NOT un-count (the
  budget rail cannot prove the repeat never left the host — `llm.chat` retries
  on the wire before a later reservation can be refused — so the record keeps
  the attempt booked and the budget terminal, not the provider terminal, ends
  the round); every caller outside the interactive primary rail keeps
  `transport_death_retries=0`; no consumed/terminal/patch-disposition
  predicate gates a session supervisor's cognition, and a successful
  live-leaf hold closes any prior transport episode so its acknowledged wake
  alone resumes the model.

#### Timeout & Wait Control

- Required owner waiting retains the original execution: pooled work stays
  RUNNING, the worker lends only active capacity, and waiting exempts only the
  idle timeout — Stop, deadline, absolute ceiling and monetary admission still
  bind (the capacity transfer, the warm/cold wake and what a grant consumes:
  ARCHITECTURE "5. Supervisor Loop"). Persist the completed-tool source, task
  wait and queue snapshot before lending active capacity; grant the original
  worker only after reserving active capacity, restoring both capacity marks on
  failure; keep attempt, start time, completed effects and usage unchanged
  across a warm wake. Cold recovery requires the acknowledged planned-restart
  handoff through every shutdown cleanup, and a direct-actor checkpoint alone
  grants no automatic cold-restart authority. After either kind of wait,
  control/deadline handling precedes the saved round's budget decision;
  preserve TaskModelWait role overrides, explicit Auto, auto-continue and
  completed quota union through that owner's continuation methods. Calendar
  deadlines and ordinary owner-wait time retain their meaning
  (`tests/test_owner_wait_pool.py`, `tests/test_owner_wait_restart.py`,
  `tests/test_owner_wait_cold_loop.py`, `tests/test_owner_wait_budget_tail.py`,
  `tests/test_owner_wait_model_context.py`).
- For a session nanny, `delegate_wait` is event-only at the model surface:
  host supervision renews bounded transport windows with zero LLM calls,
  journal progress streams to the owner without waking the model, and only
  terminal/interaction/fault, an addressed task/owner message, a direct-child
  signal, control/recovery judgment, or a model-requested one-shot
  checkpoint wakes it. Do not reintroduce caller-visible `wait_sec`,
  repeating timers, progress wakes, or a host semantic stall detector.
- The wait/continue/stop decision is a structured fact — terminal status plus
  heartbeat freshness from `queue_snapshot.json` via `task_status.py` — never
  a keyword or regex over content (BIBLE P5). Fixed kill-timeouts (hard
  task/tool ceilings, watchdog) remain the outer safety bound; progress-aware
  waiting tunes the passive wait only.
- Preserve raw terminal model/salvage bytes separately from the host-authored
  `terminal_provider_notice`. Existing receipts and secondary notices consume
  those same facts: attempted repeats, the last provider error, and an unknown
  dispatched outcome. A retained answer must not hide wait or unknown-attempt
  evidence or invite a blind rerun. Message/deferred Presence
  responses render one host-labelled status section; cached Presence output
  is already rendered. Preserve silent/tool-delivered authority and never
  change raw answer bytes in the Presence renderer. Carry the
  actual control reason through wait termination: owner Wrap up is distinct
  from deadline/budget finalization and still sends no new summary request.
- Transport-wait notes use the ordinary progress seam with `incident=None`.
  Explicit incidents from other producers keep their typed `task_incident`,
  `toast_once` and optional `toast_tone` presentation. Never infer urgency or
  valence from a task's prose (`tests/test_loop_transport_wait_interactive.py`).
  A cross-model lane switch carries the same incident pair; it names both models, the account the
  send's own binding selects when that route has accounts (a task-local wait
  override included, never the configured value alone), and the typed failure
  reason when the round record has one. The applied-option mismatch line also
  rides the same loop-level callable: the frozen `ToolContext.emit_progress_fn`
  takes one argument and never carries the pair.
- Timeout contract classes differ; keep the axes separate. A transport
  timeout only bounds a dead socket
  (`OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC`) — it is not a reasoning cutoff
  and never evidence that reasoning stalled. API review uses its transport
  bound as a settlement fallback because that request ends there; a delegated
  agent session inherits the task absolute ceiling because the paid engine
  run can outlive an HTTP read; the owner deadline always narrows either
  route, and provider transport defaults (Anthropic, VLM captioning) are
  ceilings, not promises to run past it. Default reviewer slots intentionally
  have no short cognition cap; the outer `plan_task` envelope covers the
  session lifetime; `web_search` sizes its outer envelope for the complete
  configured paid cascade, recomputed under an owner deadline.
- New numeric timeout constants are an SSOT in the owning settings leaf, not
  in the `config.py` facade: the key and its shipped default go into
  `settings_defaults.py` `SETTINGS_DEFAULTS`, the clamped getter into
  `runtime_limits.py`, both re-exported through `ouroboros.config`, which stays
  the one import surface. Register the env key; do not scatter magic wait
  numbers across call sites (`tests/test_timeout_policy.py`).
- Worker readiness has its own structural `WORKER_READY_WINDOW_SEC` and
  `WORKER_READY_MAX_ATTEMPTS` in `runtime_limits.py`, re-exported by config
  (the readiness/exhaustion lifecycle: ARCHITECTURE "5. Supervisor Loop").
  Reuse the lifecycle-owned execution-state reader (workers facade) at
  reserve/final enqueue and snapshot; retain the separate repository-writer
  policy at public admission and internal boot/update exceptions. Readiness,
  process liveness and task idle deadlines stay independent; failed writes
  retain terminalization retry, never false Done or an automatic fresh startup
  budget.
- Nested process wrappers are ordered, never tied: the provider bound settles
  before its killable child, the child before the generic ToolEntry envelope
  (fixed structural settlement margin from `config.py`), so a child or
  provider result cannot arrive after its owner has abandoned custody. The
  one deliberate early return is plan review's dispatch barrier
  (`ReviewRequest.drain_deadline`): the wrapper returns while its workers run,
  but custody is not abandoned — the workers settle into process-local custody
  and announce the wave through the task mailbox (`plan_review_collect`).
  Before its effective blocking verdict, `owner_hurry.force_plan_decision`
  collects once at zero wait and projects the returned state. Context health
  only reads the canonical wave; its pending count/time describe the recorded
  snapshot, not live worker progress. Neither path dispatches a second panel.
- Every physical LLM/review/VLM/tool operation that can outlive a logical
  wait emits typed `cognitive_operation` start and terminal facts; the
  supervisor uses the active-operation map only to spare the idle rail, and a
  terminal fact must match task-attempt plus execution/round/call identity
  before it clears the row. A logical timeout with a live paid worker is
  custody/reconciliation-pending, never permission for a blind paid retry;
  late results settle the original attempt and stay bound to its retry
  identity. Symmetrically, an owner terminal that is not a deliberate
  verdict is not permission to cancel the live paid run that owner held: the
  sweep spares it, discloses it, and lets its own bound limit the damage.
- Once the owner deadline minus finalization reserve is spent, an unstarted
  review row is a typed `$0 not_dispatched` actor — no worker, paid stamp, or
  active lease; an already-paid in-flight wave stays eligible for exact
  custody reconciliation without authorizing a new dispatch. An in-flight
  reviewer never counts as final quorum, under either enforcement mode; a
  `pending_dispatch` row (released at the dispatch barrier) is neither quorum
  nor a paid fact until its settled row proves the physical send.
- Every zero-physical acceptance refusal — an unresolvable partial source, an
  immutable-core overflow, a slot whose window cannot hold the rendered prompt
  — records a typed `$0 not_dispatched` row that carries its cause in `error`
  and folds the aggregate to `DEGRADED` (the one $0 exit shape: ARCHITECTURE
  "Review stack"; the acceptance instances: "Task lifecycle").
- A returned provider response (including an empty body) or typed terminal
  408/429/5xx is settled and may use the surface's bounded retry rail;
  `dispatched`/`unresolved` without a typed terminal status stays under the
  custody-lost/no-resend classification (the custody vocabulary, capture
  precedence and no-resend rules: ARCHITECTURE "Review stack"; the interactive
  repeat rail: "Context fitting, retry, and compaction"). A NEW logical request
  needs a unique host-attested input absent from the unknown one: the managed
  upstream-recovery notice or the separate nanny-leaf wake contract in
  `ouroboros/delegate_hold.py`.
- A custody retry key names semantic material and an admitted cycle, never its
  rendered prompt (the identity rule, the write-ahead paid stamp and Skill
  Review's wave reservation/`review_resume_of` rejoin: ARCHITECTURE "Review
  stack" and "Review delivery"); an unstarted chunk cannot authorize PASS, and
  a window with no dispatch capacity leaves an unpaid `$0` wave and no paid
  stamp.
- A reviewed mutative wrapper retains foreground custody until the workflow
  settles; never use the generic 600s tool default or a guessed hard ceiling
  to abandon a still-live reviewer or commit pipeline.
- Cooperative cancellation applies where the route supports it (delegated
  sessions); API/thread routes disclose an in-flight custody state until the
  physical result settles. A typed transport failure after a delegated run
  has an id is an unknown outcome — retain the durable invocation token and
  replay that started run on the permitted retry instead of posting a second
  paid run; a supplied retry token with no valid durable invocation is
  `review_custody_lost`, never permission for a fresh paid session. Elapsed TTL alone never authorizes a resend, and owner death is proven only by
  pid death (ARCHITECTURE "Review stack").

### Loop and acceptance state machines

#### Loop / State-Machine Changes

- Changes to `loop.py` or other task state-machine logic include adversarial
  tests for malformed output, false-completion prevention, replay/log
  durability, and failure modes — not just the happy path. Audit/checkpoint
  rounds must not silently reuse the normal final-answer path unless that
  invariant is explicitly tested and documented.
- Keep a complete loop-local `DeliveryCandidate` once a substantive answer
  exists, with host control exposure as sticky candidate provenance inherited
  through every replacement (mechanism and the disclosed test-pinned
  residuals: ARCHITECTURE "Task lifecycle" and
  `ouroboros/loop_delivery.py`). A FORCED finalization resolves an armed
  control purely and without retry: valid keep/replace is honored, anything
  malformed preserves the retained candidate with a typed degraded reason,
  and protocol JSON never reaches chat or the durable result. Main distinguishes
  consumed owner source from changed requirements. Effective criteria and
  material effects, including nominated read observations, define the reviewed
  subject; ingress generations preserve unread-message ordering. Status text,
  narration and a changed working view do not themselves buy another review;
  finalize task-scoped service outputs/errors before host acceptance. The
  control must not bypass verification, acceptance, safety, skill
  finalization, deadline, child handoff, the unconditional `FINAL ANSWER:`
  latch, or the task-level answer protocol.
  Host-authored notices remain outside model answer bytes and its acceptance
  identity; use the existing terminal record/outbox/System projection, preserving
  notice visibility on replay and single-body/headless transports. An unchanged
  answer never regains a verdict superseded by actual owner or evidence changes.
- Every direct child result needs an exact-hash disposition through the
  existing `tree_note(kind="decision")` tagged payload
  (`type=child_result_disposition`; the batch form validates entries
  individually by index). The typed task-tree row is the sole authority;
  task-result disposition fields are derived reads, never a mirrored write.
  Binding the complete-result SHA-256 means a parent cannot claim it
  integrated a result that later changed. `deferred` suppresses only the
  reminder and forces an honest degraded/best-effort terminal answer until
  resolved. That per-value consequence is carried by the
  `tree_note` payload schema itself, which is the SSOT for when to choose each
  value, and its enum reads the validator's own set. A child wedged in the legacy `cancel_requested` latch is intent,
  not outcome — it stays visible as cancel-pending until custody settles it.
- Host task acceptance is root-only; eligibility uses structured facts
  (`outcomes.turn_has_reviewable_effects` plus a typed
  deliverable/criterion), never keywords (BIBLE P3/P5). The agent-callable nomination is never authoritative (ARCHITECTURE "Task
  lifecycle"). Freeze its request and resolved roster; use existing review
  custody and mailbox continuation for pending work and free collection. The
  worker never writes Main's live candidate or author decision. Keep
  subtree/status facts
  separate from reviewer findings and Cyber's authority under BIBLE P0.
- Delivery-control JSON applies only to a final response with no tool calls.
  Retaining an answer leaves tools available for further work. Main may keep
  the complete answer while explicitly changing effective criteria or material
  evidence; that creates a new review subject, not fresh authority from an old
  verdict. Source acknowledgement never infers semantic change by generation.
  A requested file or diff
  does not imply a universal commit-or-revert rule. Self-modification keeps its
  reviewed-commit contract under BIBLE P0/P3.
- Post-task synthesis receives `completion_observations` from the existing
  terminal result writer (what the sealed package carries and what it cannot
  prove: ARCHITECTURE "Post-task reflection"). Counts come from
  `OWNER_DELIVERY_TOOL_NAMES`, not prose parsing. Recovery uses the stored
  snapshot; global skill state never attributes an owner click to the task;
  task-summary calls use the existing `chat_observed` custody seam.
- Promoted tasks carry their host-minted root id and role on the queue payload.
  RUNNING writes preserve the actual `_task_started_ts` as `started_at` and an
  existing `queued_at`; terminal `ts` remains its own field. Missing historical
  start facts stay missing. LLM usage carries the existing call/execution/round
  ids so it can join the worker's round without duplicating that round event.
  Delegated settlement/disposition/unread rows carry their existing root/parent
  ids. Intrinsic pacing exposes the same `cost_ceiling_disclosure` its text uses;
  tool error manifests carry their typed code and redacted reason preview.
  These are observation links, not new accounting or zero-price rules.
- Acceptance evidence identity hashes source facts before history-dependent
  budgeting; recording a review must not change the facts it reviewed. Complete
  applied host records are saved by
  `review_projection.publish_acceptance_checkpoint` through the existing
  write-once source handles before compact publication (the copy-back,
  CURRENT-basis and same-store rules: ARCHITECTURE "10. Key Invariants", the
  paragraphs after the continuity map; the completion-source reader: "Post-task
  reflection"). Ordinary artifact registration keeps its short locked manifest
  merge; copying/hashing finishes before that lock, which never acquires a
  task-result lock. Test delayed snapshots and child replicas through the same
  central merge, and verify that the full source downloads while the task is
  still running. Review/completion sources use
  `source_handles/context_checkpoints`, outside deliverables and the acceptance
  artifact manifest, and terminal references must carry the task's chat id,
  including zero. A missing source is disclosed, never reconstructed from a
  bounded preview. Source/capacity, publication order and paid identity are
  separate contracts; changing history or presentation must not mint work.
  Verify the persisted consumer after the actual snapshot merge and child
  cleanup, not just the copy result; one operation-scoped memo may reuse
  verified work but must not cache failure as success or survive as a second
  store.
- Pooled terminal file preparation belongs to
  `headless.prepare_terminal_task_files` at the worker's own task_done
  boundary, after blocking post-task work and before releasing the slot;
  earlier answer/metrics delivery stays early (the attempt/readiness split and
  the transient `terminal_source_present`: ARCHITECTURE "5. Supervisor Loop").
  Neither an I/O exception nor a lost event authorizes model replay; never
  persist `terminal_source_present` as a new anchor.
- Health owns captured terminal-file preparation/recovery; the existing reaper
  owns queue execution and deferred-job replay on the health cadence (the
  recovery flow and fault policy: ARCHITECTURE "5. Supervisor Loop"). Preserve
  worker/meta/task/attempt/root identity across each off-lock operation; keep
  the normal terminal event owner for queue release and project/evolution hooks
  — no separate crash executor; host crash terminals withdraw their captured
  RUNNING owner before emission, cancel checks file readiness before source
  removal, and deferred timeout jobs keep their original worker/attempt/root
  binding so old file recovery cannot kill, requeue or replace a newer
  execution.
- Same physical observability store means verified reuse of original manifest
  bytes and canonical path spelling, never a rewrite or native promotion
  marker; missing aliases resolve only through the exact verified canonical
  CAS/call readers (ARCHITECTURE "10. Key Invariants"). Do not add digest
  filenames, an initial-adoption anchor or a persistent transfer store.
- Pooled mailbox cleanup follows the file helper's settled-cleanup predicate,
  and startup recovers terminal child sources before the actual prune
  (ARCHITECTURE "5. Supervisor Loop"); direct canonical cleanup remains direct,
  never race unknown prior worker ownership, and no saved anchor is required.
- Acceptance payment follows the semantic subject and substantive disposition
  identity defined in ARCHITECTURE "Task lifecycle". Source generations,
  read repetition or narration alone do not create paid authority; changed
  effective criteria or material evidence can change it even with identical
  answer text. Reuse the existing subject and paid-identity owners, never
  introduce a second hash or rely on cosmetic answer edits.
- Task-acceptance actors are the configured triad rows (owner R0–R2,
  2026-09-01; `reviewer_slot_config.triad_delivery_slots`, malformed config
  refuses typed) and receive one substantive interaction on their own delivery;
  the retrieving work order, the `evidence_refs` resolution against the FULL
  packet, the money rule (one work-order send per paid row, no rounds
  multiplier, no second pricing pass), the once-per-panel launch floor (owner
  R52/R55, 2026-09-03; `task_pacing.review_launch_allowed`,
  `task_acceptance_paid_dispatch_stamp._claim`), the R23 clamps on a running
  panel and the disclosed deadline-cut residual are stated once in ARCHITECTURE
  "Task lifecycle". Format-repair resends are packet-row only; child/`off`
  acceptance runs packet rows only.
- The host acceptance decision is written ONLY by
  `loop_acceptance._set_acceptance_decision` (re-exported from `loop`), with exactly three
  owner-facing states, each with a typed reason from the closed set; an unknown
  status fails closed. When you add a writer, add its reason to the
  closed set AND check every value-keyed reader —
  `outcomes.derive_loop_outcome` keys degradations and blocked terminals on
  status+reason PAIRS, and breaking a pairing is a silent false green.
  Every forced rail closes a dangling `revision_requested` through
  `loop_acceptance.terminalize_dangling_revision` (what it stamps and never
  overwrites: ARCHITECTURE "Task lifecycle"). The reviewer verdict vocabulary `PASS|FAIL|DEGRADED` is NOT narrowable;
  `adaptive_quorum` applies, any contributing FAIL fails, DEGRADED abstains,
  and no quorum is a terminal HOST decision. Degraded review or a best-effort objective must never render as green solved
  on any surface (the shared phase projection: "Design System"; its host
  mirror: ARCHITECTURE "Chat and Projects"). Do not add task scope review or
  reuse the commit gate.
- The acceptance improvement loop is a reviewer-authored DIALOGUE: obligation
  identity comes from the reviewer's typed
  `disposition_kind`/`obligation_id` (an unknown re-raise id fails closed to
  `new`, disclosed); a re-raise reopens the row without wiping the agent's
  argument; termination beyond a clean PASS/accepted rebuttal happens ONLY
  via the reviewers' `dialogue_status` judgement or a real rail under Blocking.
  Advisory also permits explicit post-feedback author finish before another paid
  panel, including a revised answer. Keep critic and author hashes separate;
  bind controlling intent through the existing delivery-evidence fingerprint and
  consume it on owner/evidence supersession. No semantic host counters or
  keyword gates (P5). The vote reduction — one contributing reviewer holds the loop open only WITH
  MATERIAL, missing/invalid votes abstain and never default to continue, zero
  well-formed votes reduce to the typed `inconclusive` — is ARCHITECTURE "Task
  lifecycle". Changes here must cover malformed reviewer
  output, unknown/stale re-raise ids, partial panel failure, multi-slot
  status disagreement, replay/restart durability of obligation rows, false
  completion, and the backward-compatible default when new fields are
  absent.
- An explicit `max_improvement_passes` binds under every legacy policy;
  otherwise the shared review-cycle cap binds under EVERY policy, giving
  `improvement passes = cycles − 1` (the retired acceptance key is migrated
  into the shared key at settings load and never binds at runtime).

- A `PATCH_DISPOSED` row names its disposer (`disposed_by_task_id`),
  because a non-owner may write it once the owner task is terminal;
  wait/cancel/answer authority stays owner-only. A terminal custody
  obligation is disclosed additively (an objective warning plus the
  reason code, preserving a truncation rail code); converting a custody
  fact into a review, objective or execution verdict is the defect this
  rule prevents.

Enforcement: the adversarial tests the first bullet mandates, plus
`tests/test_child_result_disposition.py`, `tests/test_acceptance_fence.py`,
`tests/test_v674_acceptance_dialogue.py`, and `tests/test_review_cycles.py`
(cap migration).

#### Cognitive Artifact Integrity

- Cognitive artifacts (identity.md, scratchpad, task reflections, review
  outputs, pattern register) must NOT use hardcoded `[:N]` truncation. When
  content must be shortened, summarize explicitly — attempts, changes, and
  conclusions survive — and disclose the omission with a resolvable
  reference; an omission marker alone is disclosure, not sufficiency
  (BIBLE P1).
- All primary reasoning flows include the core governance artifacts as
  first-class sections — see "Core Governance Artifacts". A new reasoning
  flow MUST follow that contract, not rely on touched-file inclusions.

Enforcement: review-only — CHECKLISTS item 2(f) scores the no-`[:N]` rule in
commit review.

---

