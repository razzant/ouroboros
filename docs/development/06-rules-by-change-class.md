# Rules by change class

This chapter gives one rule section per change class: tool registration, skill payloads, the live E2E stand, light mode and deliverables, retention, delegated subagents, cancellation, onboarding and settings, transport and late-result custody, LLM calls, timeout/wait control, and loop/acceptance state machines. Each section names its enforcing tests or gates, or marks rules as review-only; check a change against its applicable classes.

`docs/CHECKLISTS.md` remains the only reviewer scorer; its `development_compliance` item points at this handbook as a whole.

### Tool registration and guard surfaces

- A new Tool: export it from `get_tools()` with the `ToolEntry` pattern in `registry.py`, give it an explicit `ouroboros/safety.py::TOOL_POLICY` entry (`POLICY_SKIP` for trusted built-ins, `POLICY_CHECK` for opaque or outward-facing ones) and declare its capability class in `ouroboros/tool_capabilities.py` (`CORE_TOOL_NAMES`, child profiles, parallel/truncation sets). Without the policy entry it falls through to `DEFAULT_POLICY = POLICY_CHECK` and pays a light-model LLM call per invocation. Add it to a child profile only when that narrower principal should receive it; test schema plus execution behavior rather than mirroring names into another catalog.
- A tool that WRITES the repo working tree needs the GUARD surfaces too, not only the visibility ones: add it to `_ROOT_ARG_REPO_WRITE_TOOLS` (the single set every repo-write fence keys on — the acting-no-workspace fence, the protected-write gate and the acting root-enum narrowing; ARCHITECTURE §6 "Tool capability and execution") and canonicalize its target paths — `_PATH_NORMALIZED_TOOLS` for a top-level `path`, `canonical_repo_relative_path` + `_payload_write_paths` for payload-borne paths. Visibility checks can all be green while these are missing, so tests must exercise the real guard chain, not only a mocked resolver.
- New memory/data files: decide in the same change whether they appear in LLM context (`context.py`).

Enforcement: CHECKLISTS items 2(g) and 10 (`tool_registration`) in commit review; `tests/test_tool_api_v2_public_surface.py` pins the public schema/registry contract and `tests/test_local_routing_and_safety.py` the safety-policy fallthrough; CHECKLISTS item 11 backstops the memory/context decision.

### Skill repair and payload lanes

- Start Repair as ordinary managed development carrying the selected skill, source request and admitted revision (admission and revision checks: ARCHITECTURE §13; §6 "Skills and extensions"). The UI's Repair-and-run request is a real owner message whose origin follows the ordinary task path; resolve that source when the model enables the repaired skill, never a client `allow_enable` flag. Preserve a later direct owner disable — a load-error revert is not one. Read existing `skill_repair` records as selectors, never as a reduced profile, and keep the normal file, shell, browser and delegation tools under the existing readonly/acting-child ceilings.
- Keep installed payloads as ordinary directories; a delegated Git copy is an optional existing capability. Check the known revision before an operation and, after opaque process work, record the observed revision without asserting exclusive authorship. No long shell lock or automatic rollback belongs here.
- Use the existing payload binding/policy owners for all path forms: a valid selected normal/repair `TaskConstraint` supplies omitted skill-name/bucket selectors, while explicit selectors must still match that physical payload. Markerless native-directory payloads remain logical external, and collision, native mutation, child-profile, launcher seed and provenance/review/grant/dependency controls keep their existing guards.
- Review, grants, dependency readiness, desired enablement and actual execution are independent facts. Resume an unchanged reviewed snapshot through its existing free replay; a dependency or load failure never rewrites the review verdict; preserve an explicit owner disable and the original automatic request.
- Base skill-review convergence and retry coaching on the lifecycle's group `review_round`, retaining `snapshot_attempt` for display. Changing only the coaching ordinal must preserve the prompt-builder and aggregation vocabulary's free-replay fingerprint; ceiling refusals name the exits the existing author-finish predicate permits. The rendering and rebuttal tests exercise revised snapshots across rounds, while `tests/test_review_cycles_gates.py` pins the replay fingerprint and enforcement-specific exit text.
- UI, existing CLI commands and task tools call the shared operation owners. Actor identity is host-derived: owner-only actions require a real member chat/quiz/mailbox source naming the exact skill, revision and requested items; the model interprets intent, no synthetic reference creates permission, and ordinary Repair implies neither grant-all, attestation nor deletion.
- Test the real installed script/tool/HTTP/widget/companion after review and prerequisites, repeat after corrections, and inspect a widget screenshot. Execution receipts name the actual dispatched revision; they are not PASS.

Enforcement: `tests/test_skill_payload_binding.py`, `tests/test_skill_development_revision.py`, `tests/test_skill_lifecycle_actions.py`, `tests/test_skill_development_execution.py` and the UI lane `tests/test_ui_smoke_skill_lifecycle.py`.

### Extension dispatch and isolated dependencies

- A `type: extension` skill with a reviewed isolated dependency env must not import `plugin.py` or execute handlers inside `server.py`, even when the dependency tree looks pure-Python; payload-native marker files (`.so`, `.dylib`, `.dll`, `.pyd`) also force child dispatch. This is containment, not admission: a native payload still faces the skill-review checklist. Keep the split explicit — no-dependency pure-Python extensions may use `extension_loader`'s in-process PluginAPI; isolated-dep/native-marker extensions are cataloged and dispatched by `extension_process_runner` short-lived child processes (staging and confinement: ARCHITECTURE §13).
- Proxies answer a child crash, invalid JSON, timeout or abort with normal tool errors / HTTP 502 / WS log messages — a child `SIGABRT` is a handled extension failure, not a server crash. Keep the child confinement (scrubbed env, per-skill grants and isolated deps, process-group tracking, output caps, timeout cleanup) and add no fallback code that imports native-risk plugin modules in the host process.

Enforcement: `tests/test_extension_dispatch_threaded.py`, `tests/test_extension_isolated_deps.py`, `tests/test_extension_process_runner.py`.

### Declared skill resources and builds

- Reuse the isolated dependency owner for exact downloads and literal build/check argv (records, caches and the delivery-versus-declared-check split: ARCHITECTURE §13). A new declaration must match a fresh executable review and hash-covered specs; revalidate the pinned payload before launching its processes.
- Keep verified resource/package caches outside the replaceable payload/env. Record actual resolved versions, resource/output hashes and diagnostics in the existing dependency records: package-manager success is not proof of the requested function — only an explicitly declared check establishes that fact.
- Preserve wheel-only and npm ignore-scripts unless that entry opts into the corresponding build action; manual dependencies keep their existing contract.
- Ordinary binary payload resources use the review classifier/descriptors during delegated capture too; large downloaded/build resources belong to the isolated dependency path, not a giant source patch.
- Go compilation/execution share the existing child owner and timeout. Deno flags express current script effects and task network constraints — not new grants, and not a stronger claimed OS sandbox than the other reviewed scripts have.

Enforcement: `tests/test_skill_install_resources.py`, `tests/test_skill_runtime_commands.py`, `tests/test_skill_runtime_lifetime.py` and `tests/test_skill_payload_binary_transfer.py`.

### Task contract resource policy

- Outside Cyber Pro, `resource_policy.protected_artifacts` is a typed affordance policy: execute-only black-box references may run, while byte reads, copy/hash/static introspection, tracing and debugging of the declared paths are blocked (the guards: ARCHITECTURE §6 "Safety and runtime mode").
- Acceptance claims (`id`, `claim`, `surface`, `support`, `priority`) are bounded advice, never gates/taxonomies. `success_criteria` aliases input only. `effective_acceptance_claims` binds frozen ingress over closed plan waves; after ingress, the evidence owner may select a valid current Advisory `author_plan`. OPEN critic waves bind nothing: expose `none_open_plan_wave` and non-binding `plan_claims_exhibit` (ARCHITECTURE §11.1, §6 "Task acceptance"). Children receive only explicitly passed `schedule_subagent` claims. Bind reviewer `evidence_refs` by exact host-packet membership, never fuzzy matching, file reads or re-execution; change only clean bit/disclosure, never actor parsing, quorum or verdict.

Enforcement: `tests/test_protected_artifacts_policy.py` and `tests/test_acceptance_claims_wiring.py`.

### Skill-defined Presence

- Keep behavior portable and authority installation-local: a reviewed `presence:` profile declares instructions, context topics, bounded runtime defaults and conceptual tool/script/resource requests — never provider credentials, room ids or one installed tool spelling; `presence_capabilities.py` stores the owner's exact selections outside the payload, fingerprinted by the request semantics that authorize them. Preserve its optional `workspace_root` (an owner-local external folder, validated through the existing workspace admission and copied into each task contract) when editing runtime/capability selections; unset profiles retain their prior serialized state and fingerprint. Presence keeps canonical shared memory without deriving a Project or creating a forked drive from that folder (ARCHITECTURE §6 "Skills and extensions").
- Presence authority is a positive immutable ceiling, not a denylist or a prompt promise: admission requires the owner-created binding plus an installed, enabled, freshly executable behavior skill and every required selection, then freezes skill/profile/state/selection fingerprints, exact grants (the profile's selections plus the constant cognitive-memory baseline `tool_capabilities.COGNITIVE_MEMORY_TOOL_NAMES`; a selected grant keeps its bindings), argument bindings, runtime slot and round limit into `task_contract.capability_ceiling`. Schema discovery and execution enforce that same ceiling for built-ins, extensions, MCP tools, scripts and resource roots.
- `state/presence_bindings.json` is host-owned authority: a transport token resolves only bindings naming that exact transport skill, and the submitted provider/account/conversation/thread must match the binding origin — never recover those identities from message text. Staged files stay inside the calling skill's state root before entering the ordinary attachment store (the turn flow: ARCHITECTURE §12).
- Run each admitted event with a fresh agent, a deterministic binding-plus-source-event task id, the cross-process installation-wide concurrency gate and per-conversation serialization; the transport's durable provider custody owns arrival FIFO before Host admission. Do not add a transport-specific task scheduler, memory silo, core terminal outbox or resident cross-room agent.
- Completion is exactly `message`, `silent`, `tool_delivered` or `deferred` (deferred requires a successfully promoted `work_ref`; correlated lookup stays behind the same transport token and binding, and `presence_cancel_work` additionally requires the current binding and conversation to match). Promotion and `schedule_followup` copy the Presence metadata, admitted workspace and capability ceiling by value; any new descendant producer preserves this ceiling or refuses the transition — reconstructing authority from mutable current state is forbidden.
- Knowledge-topic and scratchpad mutation each use one stable lock, so concurrent owner and Presence turns cannot overwrite a newer projection with an older render. Test the boundary at both layers — strict profile/state/ceiling parsing, stale/missing review admission, schema and direct-execution filtering, argument binding, binding/token/origin checks, event idempotency and conversation ordering, typed outcomes, late-work correlation, promotion/follow-up inheritance; provider adapter E2E is separate evidence. Enforcement: `tests/test_presence_admission.py` plus the both-layer boundary tests this list requires.

### Devtools isolation

`devtools/` is tracked operator code outside runtime package discovery and the runtime import graph (ARCHITECTURE §1 "Devtools boundary"): runtime modules, `server.py`, web modules and build scripts must not import it. Touched devtool files receive normal triad/scope review; unrelated files reach the scope reviewer as index rows it may open on demand, so operator code does not drown core review. Generated outputs live in an explicit external root, never in `repo/` or live `data/`; domain-specific architecture and methodology live beside the devtool, not in core governance docs. No automated import guard — review-only (triad/scope review of touched devtool files).

### Live E2E stand (`devtools/e2e_live/`)

`python -m devtools.e2e_live.run_live_lanes` exercises owner-shaped work (the SM1, SW1 and SK1 scenarios) on isolated real servers; `stub_lane.py` reuses the loopback model and review answers of `tests/system_e2e/harness.py` for the `--stub` $0 rehearsal. The one rule that binds runtime changes: keep this opt-in stand outside runtime imports and default local evolution. Its operation — scenarios, seed and settings, budget, self-modification proof, reports and its CI job — is the operator manual `devtools/e2e_live/README.md`, beside the code.

#### Scenario acceptance

Judge durable artifacts and actual consumer observations, never model prose or an HTTP 200 alone — see devtools/e2e_live/README.md#scenario-acceptance.

#### Seed and settings

Test one admitted, clean, detached seed with settings built from the tree's defaults and explicit stand knobs, never the owner's live settings, and read the provider key only by environment name, never from a pool file — see devtools/e2e_live/README.md#seed-and-settings.

#### Budget admission and ordering

Admit an attempt only while settled spend plus in-flight reservations plus its own reservation fit the run cap, FIFO by dispatch index (every scenario's first attempt before any second), with `state/usage_attempts.jsonl` as the only money source — see devtools/e2e_live/README.md#budget-admission-and-ordering.

#### Self-modification and browser lifetime

`--self-mod` is opt-in and only SM1 owes the real re-exec/absorb proof, which mere liveness never passes; the browser client opens on first use and is closed and reopened across a restart on the lane thread — see devtools/e2e_live/README.md#self-modification-and-browser-lifetime.

#### Reports, focused verification and CI

Run roots are append-only outside `repo/` and live `data/`; the focused contracts are the `tests/test_e2e_live_*` modules plus `tests/test_server_runner_absorb_wait.py`, and the `e2e-live` CI job runs only on its nightly cron or an explicit `e2e_live=true` dispatch — see devtools/e2e_live/README.md#reports-focused-verification-and-ci.

### Light mode and external deliverables

- `runtime_mode=light` is a self-modification boundary, not a deliverables
  boundary: Light refuses mutation of the Ouroboros repo/control-plane and
  still builds user deliverables outside them; Pro permits protected rewrites;
  Cyber agency follows the same effective Access owner (`ouroboros/config.py`
  owns the semantics; the ladder and its WHY: ARCHITECTURE §6 "Safety and
  runtime mode").
- Preferred flow: `task_drive` for scratch, `artifact_store` for canonical
  deliverables, `user_files` for the owner's visible copy (the roots:
  ARCHITECTURE §6 "Tool capability and execution"). `write_file(root=user_files)`
  and declared process `outputs` register/copy the canonical artifact; a
  rewrite keeps the previous copy in non-manifest history with last-5
  retention — recovery, never a second deliverable list. `root=deliverables`
  stays read/list/search-only and is never granted to children.
- Large task files stream (`artifacts.stream_artifact_file`, atomic
  `copy_artifact_file`): never a whole dataset in a bytes object, and a read is
  rejected as soon as it exceeds the source's initial regular-file size rather
  than waiting for a growing file to reach EOF. HTTP admission and
  materialization run their whole blocking operation off the event loop
  (`gateway._helpers.run_sync_to_completion`); cancellation waits for it before
  releasing anything, and cancelling an HTTP waiter never cancels the admitted
  task. Directory exports carry a complete relative member/size/SHA manifest
  plus a streamed ZIP (outputs above 50 MiB included); a changed file or
  missing member is an explicit capture failure, while genesis LISTING is
  discovery and only capture/copy is strict.
- `send_file` uses immutable captured names, so an earlier delivery URL never
  aliases a later rewrite; without capture, small-file inline delivery stays
  available with no fabricated URL or reference. Registered immutable
  downloads verify their bytes once per request without materializing the
  whole task result; `downloadBlobViaHostBridge` saves only an already-owned
  Blob or data/blob URL, never an HTTP response/stream turned into a Blob.
- Input authority keeps at most 25 rows inline and, beyond that, an additive
  `attachment_manifest_ref` in the source-handle store: preserve its
  count/size/SHA, resolve the complete set before child materialization,
  mailbox inheritance, retry or copy-back, and never fall back to the preview
  when the source fails. Inputs stay inputs, outside deliverable inventories;
  a failed copy-back rides the pending-ref retry/GC contract instead of losing
  child bytes.
- Preserve the exact process arguments and the prepared resource binding
  through admission and execution; quoted examples and unknown interpreter
  effects are not proof of writes. ARCHITECTURE §6 "Safety and runtime mode"
  owns the source/Supervisor contract — reuse it, with no second detector,
  consent store or automatic repetition of a denied operation.
- `scratch=[...]` is a DISTINCT channel from `outputs=[...]`: ephemeral in-cwd
  verification files, exempt from the undeclared-output guard, never
  registered as artifacts, adopted only with a declaration-time sha through the
  SSOT `artifacts.record_task_scratch`, and excluded from the workspace patch
  via `.scratch_manifest.json`; the guard verifies candidates post-exec by
  stat, so a path mention is not a write. Never overload one for the other.
- cwd: an omitted cwd selects `active_workspace`; a light direct task that
  needs writable scratch selects `task_drive` explicitly; a long-running
  service in light uses an explicit external/task/artifact cwd, and its
  declared `outputs` are copied when it stops. `run_script` stages workspace
  scripts in a unique owned directory under `.ouroboros/tmp_scripts`, which
  raw Git status and patches exclude without hiding neighbouring user files,
  and every script observes the requested cwd for relative imports, generated
  files and toolchain discovery (`ouroboros/tools/shell.py`;
  `tests/test_shell_run_shell.py`).
- Policy denials stay separate from execution failures:
  `user_files_path_blocked`, `cwd_blocked` and `artifact_output_undeclared`
  are non-failure outcomes; failing to register a declared output remains
  `artifact_output_error` (ARCHITECTURE §6 "Tool capability and execution").
- Outside Cyber Pro the default shell lane carries target-aware git policy:
  mutating git is blocked only when its symlink-resolved target lies in the
  Ouroboros runtime (`commit_reviewed` is the remedy for self-repo changes),
  read-only git works everywhere, the network fence still applies, and acting
  `self_worktree` children keep the strict no-commit policy. `git
  init`/`commit`/`push` in an external project tree is legitimate task work.
- In external workspace mode, light-mode self-repo dirty checks snapshot the
  system repo, not the active workspace, and workspace patches are captured
  against the preflight git base; project-room promotion provisions a
  standalone repo through `ensure_project_workspace` and fails loudly on a
  broken binding or unreadable registry.
- `claude_code_edit` is a retired tool name with a one-way, permanent
  compatibility contract: a saved contract carrying
  `disabled_tools=["claude_code_edit"]` also withholds the successor
  `delegate_start` (registry `_disabled_tools`). The successor is the
  configured session actor — the exact-payload class via
  `delegate_start(subagent_id=..., prompt=..., root="skill_payload",
  bucket=..., skill_name=...)` — and the api-route advisory successor is the
  native inspection episode (`review_native_episode.py`). Do not resurrect the
  name.
- Successor parity: a tool may be called replaced, retired or migrated only
  after a persistent golden test proves every user-visible target class the
  predecessor supported through the successor to the final outcome;
  deleted-test tombstones prove removal, not parity, and dropping a target
  class requires an explicit owner-approved record naming the lost outcome.
- Do not recommend `runtime_data/uploads`, skill payloads or owner state
  directories as generic artifact transport.

Enforcement: `tests/test_v674_light_mode_cwd.py` (cwd selection and what light
refuses), `tests/test_deliverables_layout.py` (deliverable placement and the
output manifest), `tests/test_git_shell_policy.py` and
`tests/test_shell_redirect_guard.py` (the shell surfaces); the
successor-parity and artifact-transport rules are review-only.

### Runtime cleanup and retention

- Age-based GC of disposable runtime artifacts shares ONE owner knob,
  `OUROBOROS_GC_RETENTION_DAYS` (default 7, hard max 365; declared in
  `SETTINGS_DEFAULTS`), and the cutoff/clamp helpers in
  `ouroboros/retention.py` (`age_cutoff`, `clamp_retention_days`,
  `get_gc_retention_days`); do not hand-roll cutoff math in new prune code.
  Prune functions keep an explicit `retention_days=` parameter; only the
  default (None) resolution reads the knob, and startup prunes are wired from
  one place (`server.py`).
- `retention.LEGACY_RETENTION_KEYS` is a migration seed, not an extension
  point: `tuple(LEGACY_RETENTION_DEFAULTS)`, the three RETIRED per-subsystem
  keys with their former defaults, which `config.normalize_settings_raw` folds
  into the unified key (customized value preserved) and drops on every
  settings read. Never add to it or reintroduce the retired keys. A subsystem
  that genuinely needs its own lifetime passes an explicit `retention_days=`;
  an owner-settable one is an ordinary `OUROBOROS_<SUBSYSTEM>_RETENTION_DAYS`
  declared in `SETTINGS_DEFAULTS` and clamped through `clamp_retention_days` —
  prefer the unified knob, and never carry a knob that deletes nothing
  (`OUROBOROS_OBSERVABILITY_RETENTION_DAYS` sits in `RETIRED_SETTING_KEYS` for
  that reason).
- Durable artifacts are NOT age-pruned: genesis projects
  (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`) and forensic observability blobs (kept
  compressed indefinitely by contract; startup runs a census, never a
  deletion).
- Review continuations are recovery state, not disposable GC: archive a record
  (collision-safe move, never delete) only when its owner task is settled, it
  stayed un-resumed past the seven-day threshold and no recorded obligation
  remains open; any uncertainty or move error leaves the live record intact.

Enforcement: `tests/test_phase3c_observability_gc.py` (the unified knob and the cutoff math) and `tests/test_observability_retention.py` (the census and preserve-indefinitely contract); the review-continuation archive rule has no automated surface — review-only.

### Live subagents

Mechanism — registry, scheduling, bootstrap, zero-run receipts, custody, work
orders, supervision, recovery, patch integration — lives in ARCHITECTURE §6
"Delegated subagents (Claudexor transport + the nanny)" and the module
docstrings it names. Review gate: CHECKLISTS items 18 (`subagent_isolation`)
and 23 (`delegated_transport`), both critical. The imperatives:

- Schedule only through `schedule_subagent`; its public schema and the
  handler's closed keyword set are BOTH derived from
  `control.schedule_subagent_properties()` — a hand-maintained mirror is
  correct only until one side gains a parameter
  (`tests/test_tool_api_v2_public_surface.py`). Child needs are declared by
  the closed capability enum, never by new `contracts/task_contract.py` fields
  or objective prose (membership test: WHO DECIDES, not who currently calls).
  Delivery is at-least-once: an exact task id with live or durable custody is
  an idempotent no-op, and semantic duplicate judgement is never the physical
  identity fence.
- `subagent_id` selects one complete row from the canonical enabled
  `OUROBOROS_SUBAGENTS` list; freeze the normalized row at schedule time and
  dispatch/restart from that snapshot, never from mutable Settings. No second
  model/lane/executor selector, no host-side ranking, no substitute actor
  after a typed refusal.
- The typed parent-LLM substrate choice is the floor (truth, money and
  authorship stay where the parent put them); topology, decomposition and
  supervision judgment are the model's ceiling (BIBLE P5/P13). Never
  reintroduce a host-side wait, poll or supervised-wait in bootstrap: waiting
  is the model's own `delegate_wait` decision, which keeps owner messages, hurry
  controls, checkpoints and parallel auxiliary children live for the whole run.
- Grow `subagent_bootstrap._DEFINITE_UNRUN_REASONS` only with reasons that
  PROVE no run can exist; everything ambiguous wakes the model. Zero-run
  receipts write only `incomplete | unknown` (a zero-run "complete" is
  unverifiable self-report); a substrate swap is a disclosed incomplete
  execution, never a silent vendor/API fallback
  (`tests/test_configured_session_prestart.py`).
- Work orders send the complete chosen assignment and host authority — no
  compiler-size cutoff, no compulsory question/file transport, no duplicated
  objective/output inside the host authority; real transport/provider refusals
  keep their cause, original input and execution custody (recovery is the
  model's choice). Full specs stay in the task source handles, with references
  only in the bounded review-state index; redacted review evidence never
  substitutes for the original requirement text. The host states native process
  access ONCE from the typed run shape (`delegate_start_instructions.access_instruction`);
  it governs that mechanism, while explicit task constraints and the assigned
  edit target still bind. Per-call access may only lower the captured profile,
  never mint task authority. Do not parse assignment prose to choose a profile
  or repeat competing native access instructions; preserve owner constraints
  in the complete work order.
- `subagents.route_health` is the ONE route reader for every consumer, and
  quota readers project one `ClaudexorGateway.quota_state()` envelope
  (`tests/test_available_subagents_runtime.py`): a fully-used ratio without a
  valid future reset neither refuses dispatch nor certifies available quota in
  the UI — final admission is the engine's. Substrate and per-skill lifecycle
  facts are VISIBILITY ONLY (acceptance judges quality, never the route), and
  an unreadable custody log reads `evidence_read_failed`, never proven-empty.
- The coordination poll is READ-ONLY of task state (it observes, never
  resolves or latches, so a poll cannot change its own next answer) and writes
  nothing beyond the canonical usage-ledger reader's bounded maintenance —
  that maintenance and its torn-quarantine residual: ARCHITECTURE §6
  "Delegated subagents (Claudexor transport + the nanny)"; owner-aware
  `usage_attempts.lock` recovery: ARCHITECTURE §1 "Platform substrate"; its
  45 s caller wait and 90 s stale grace (`ouroboros/usage_ledger.py`) are
  unchanged. Every ledger state, absence included, goes through that reader.
- `task_constraint` boolean parsing is strict (`"false"` is false); deadlines
  only narrow, delegation budgets only reduce, absent depth requests stay
  unknown rather than inferred from prose; preserve the persisted
  requested/permitted/attempted/achieved depth facts and never recompute
  historical permission from current Settings.
- `active_tool_profile` fails closed to read-only, never to
  `self_modification`/`operator_control`; ordinary external grants stay
  deny-by-default, and an explicit read-only assignment stays read-only. Only
  `schedule_subagent` may create subagents (a forged `delegation_role` is
  rejected at API/CLI ingress); live `memory_mode=shared` stays disabled
  (`tests/test_acting_subagents.py`). The Cyber acting-tool catalog and the
  subagent browser boundary (typed `BROWSER_POLICY_UNAVAILABLE`, loopback
  minus Ouroboros control-service endpoints by identity, private origins only
  via host-established `resource_policy.allowed_origins`, every redirect hop
  re-checked) are CHECKLISTS item 18 and ARCHITECTURE §6 "Tool capability and
  execution" (`tests/test_browser_url_policy.py`,
  `tests/test_browser_isolation.py`, `tests/test_browser_redirect_chain.py`).
- Acting children return `workspace.patch`; only the parent commits the live body,
  applying via `integrate_subagent_patch` then its own `commit_reviewed`.
  `external_workspace` verifies and records without re-applying. Edit/capture text preserves external Git authority and patch-only `self_worktree`. Capture bases prove no authorship; compare via explicit `vcs_diff`. A genesis project's directory is its durable deliverable
  (until it declares a `.gitignore`, small
  text build output rides `workspace.patch`, bounded by the
  per-file source-patch boundary and Git's binary verdict; no total
  source-patch cap). The canonical/replica terminal field-custody projection
  is ONE pure reducer for copy-back and effective reads — every change adds a
  stale-replica regression at BOTH seams
  (`tests/test_available_subagents_runtime_review_fixes.py`). Do not broaden
  generic data-tool behavior while fixing isolation (`forward_to_worker`
  writes only to validated running tasks in the current task/root lineage).
- A custody row carries its owner's kind; every sweep, audit and counter over
  custody rows states which kinds it covers. A review-owned run
  (`RunCustody.review_owned`) belongs to its panel — never the task's open
  delegation, substrate or replacement candidate, and never re-posted, on an
  owner cancellation too — so a task that consciously finishes under a running
  panel keeps its reviewer alive. A new delegation-domain reader joins the
  consumer matrix in `tests/test_custody_owner_kinds.py`; physical custody
  keeps seeing every run.
- The DELEGATED Git/payload lane edits a private execution snapshot and reaches
  a tree only through `integrate_delegated_patch`; one predicate,
  `delegate_shared.orphan_apply_target_ok`, serves the apply gate, the health
  invariant and the tool description, and every other guard (owner
  terminality, top-level principal, proven drift, protected paths,
  staged-never-committed) is unchanged
  (`tests/test_delegated_run_isolation_orphans.py`). A copy failure or a
  source change against the baseline leaves no registered snapshot or pinned
  ref (`tests/test_snapshot_file_inputs.py`).
- Outcome honesty: a delegating parent must not produce a clean no-tool final
  answer while direct children run undecided — one bounded absorption
  reminder, then best-effort (`children_unabsorbed`); the delivery candidate
  is HELD while that gate is open, and the delivery-control instruction never
  rides the reminder round (`tests/test_v6570_swarm_honesty.py`). `wait_tasks`
  stays batch-compact;
  `control_task_results._wait_for_tasks` owns its projection, documented under
  ARCHITECTURE's "Waiting on children"; full untruncated handoff belongs to `get_task_result` and `wait_task`, and the
  model result and the optional `terminal_host_notice` stay separate. No
  shared ledgers, automatic memory merges or new settings/endpoints unless the
  accepted plan calls for them. Push/live events are wakeups, not terminal
  authority — a lifecycle change must exercise lost/reordered terminal frames
  and reversed snapshot completion.
- Ordinary-directory sessions use the engine-owned file work product: keep the
  parent-selected direct/copy strategy, never initialize Git or change run
  mode as a workaround, keep copied binary/large inputs outside the target Git
  object database, and carry full file artifacts beside source patches
  (deletion included — an empty text diff cannot certify no work); apply
  through the existing engine CAS and durable intent/key, retain unselected
  results, and never relabel direct effects or a discard as an undo.
  Test the actual file handlers — child copyback/reopen, mixed and file-only apply,
  binary input preservation, concurrent edits and lost apply receipts
  (`tests/test_delegated_directory.py`, `tests/test_native_directory_writer_artifacts.py`,
  `tests/test_workspace_file_outputs.py`).

### Cancellation and effective status

Mechanism — durable intents, the claim/generation fence, the one settle owner,
owed terminal delivery, cascade postconditions, stop policy and hurry — lives
in ARCHITECTURE §5 "Supervisor Loop". Enforcement:
`tests/test_cancel_intents_phase_a.py`, `tests/test_cancel_cascade_v664.py` and
`tests/test_cancel_origin.py`.
The imperatives:

- Effective task status belongs in `ouroboros/task_status.py`; never duplicate
  child-drive merge or terminality logic in gateways/tools. Task waits use
  `SETTLED_STATUSES` and structured facts plus queue-heartbeat freshness —
  never keyword matching.
- `wait_task` and `wait_tasks` also peek the waiting actor's own mailbox (its
  execution drive, not its budget root) through the existing transport-wait
  reader: both waits disclose early return for pending mail without ACK or stopping
  children; the round-top drain delivers and acknowledges it. One
  episode may retain only a PROVED empty mailbox (fingerprints compared before
  and after the full reader); a read failure or torn data is never proof and
  is never cached; no TTL and no ACK in peek.
- Terminal quiz reconciliation closes the paired wait even if the answer
  arrived before worker capacity was granted; keep the answer and source
  unchanged. A failed loop without captured evidence reports unknown counts —
  never infer zero work or read an unverified checkpoint to fill the gap
  (`tests/test_autonomy_review_fixes.py`).
- Cancellation observations use `task_status.observe_cancellation_target`
  before the existing intent write: separate source observations (resolved
  physical target, task-result facts apart from queue freshness, recorded
  delegated execution), not an atomic snapshot; a later target mismatch is
  disclosed. Caller reason and request origin are distinct — an HTTP client is
  not proof of personal owner intent — and cancellation authority and
  completion-wins stay independent of these observations. Preserve the recorded
  `cancel_origin` through terminal publication, history, root/child card metadata and
  conditional result-tool reads after the active intent is removed; missing actors
  stay unknown, and exposure must not change `requested_by` parent-decision semantics.
- Cancel INTENT is never a status value: every cancel ingress writes a durable
  intent through `ouroboros/cancel_intents.request_cancel`, fails closed when
  that write fails, checks live physical ownership (a settled RESULT does not
  mean a dead WORKER) and keeps the recorded scope widen-only (ARCHITECTURE
  §10 "Key Invariants" 14). Natural completion WINS a late cancel —
  discarding is the parent's separate explicit `discard_child_result` — and
  timeout reaping is NOT a cancel ingress.
- The intent and delivery registries read STRICT to rows and `task_done`
  validates through the DURABLE result unconditionally (ARCHITECTURE §10 "Key
  Invariants" 15); only `interrupted` keeps its restore-path exemption, and
  the legacy `cancel_requested` status survives on a read path only.
- `stop_policy` is an axis on the durable intent, and the owner hurry control
  is a typed TASK-LOCAL owner-mailbox control — never a chat message, a global
  settings mutation or a review-gate weakening. Every same-id requeue producer
  calls the ONE shared `owner_hurry.retry_reset`; the durable hurry projection
  writes only through `update_json_locked` on the `owner_hurry` keys, never
  `write_task_result`; UI surfaces share `web/modules/task_control_menu.js`;
  queue-owned hurry admission initializes only an absent pooled result through
  the task-result writer's atomic `create_only` branch, and direct turns stay
  outside it.
- Code owners stay narrow behind one public queue/lifecycle surface:
  retry-aware target/subtree-liveness in `supervisor/queue_transitions.py`,
  capture-miss terminalization/publication in
  `supervisor/cancel_publication.py`, owner-stop delivery/validation in
  `supervisor/owner_stop.py`.
- Keep agent in-band cancel and the periodic cancel/delivery/ref sweep off
  supervisor drain; reuse durable intent claims and generation checks — local
  in-flight keys only deduplicate dispatch and release on failure — and
  preserve the existing cadence and HTTP response contract; do not queue
  unrelated Stop work behind a new general-purpose file executor.

### Onboarding and Settings surfaces

Mechanism — the wizard steps, the completion transaction and the install-time
proofs — lives in ARCHITECTURE §2; the Settings pages, agent accounts and the
shared chooser contract in ARCHITECTURE §3 "Settings and onboarding" and
"Navigation and shared UI contracts". Enforcement is the tests named inline plus
`tests/test_owner_settings_write_seam.py`, `tests/test_settings_env_on_disk.py` and
`web/tests/harness_setup_login_capabilities.test.js`; the copy and wizard-shape rules
are review-only. The imperatives:

- Current tasks read the existing task-entry settings view; a next-task save
  never changes an overlapping direct actor's Supervisor, Review, model or
  key; owner writers and grant classification read current disk state; the
  OOP extension payload carries only permitted typed values, never the whole
  snapshot.
- Cyber can select context, review scope/enforcement, models and Supervisor
  configuration through the existing settings writer, under effective Access;
  retain task snapshots, restart-bound access, install-time provenance and
  honest write receipts. Permission is not a review verdict.
- One five-step wizard serves subscriptions, API keys and mixed installs;
  Quick Review & start runs the same proposal compiler for skipped steps, and
  Finish atomically commits the visible draft. Only declared raw-model sources
  can supply Main — an Agent-only connection cannot invent one. Subscription
  copy says "without an API key", never guaranteed free, and connecting an
  account neither enables nor changes provider credits/spend settings.
- Settings validates the complete draft before any Save request — never omit
  an invalid custom-key row and save the remainder; a failed save/refresh
  preserves edits, leaving or reloading a dirty draft asks first, and
  saved/unsaved/unknown write receipts and the independent owner-only
  endpoints are preserved (`web/tests/settings_validation.test.js`; the real
  consumer `tests/test_ui_smoke_settings_drafts.py`).
- Model-role and actor/reviewer adapters use `model_chooser.js`; the chooser
  owns suggestions/keyboard/position only, never route identity or
  entitlement (`web/tests/model_chooser.test.js`,
  `tests/test_model_chooser_browser.py`,
  `tests/test_subscription_role_routes_browser.py`).
- Models, actors and reviewers share source/model/account controls: preserve
  exact pins on ordinary save/reload and on catalog failure; a source's
  credential harness comes from its metadata, never an assumed equal name.
  Delivery follows the row's surface and reference, not its model or account:
  every scope and deep-review row retrieves, a referenced API reviewer keeps
  native inspection, and model/account edits never silently turn it into a packet.
- One capability, one section: the task-actor story lives in Agents →
  Available subagents (`web/modules/subagents_settings.js`), editing one
  canonical `OUROBOROS_SUBAGENTS` object (list-level Enabled, at most ten
  stable rows, one prose field `recommended_use`; id and compatibility name
  automatic and hidden). Never derive durable identity from the visual
  ordinal, and never render a second control over the same settings key
  (`OUROBOROS_MAX_WORKERS` stays in Advanced because it sizes the process
  pool). Share only neutral route/model/account/effort/status primitives with
  reviewer rows (`route_editor_primitives.js`): task routes serialize
  `api_model` + `credential_profile_id`, reviewer routes `api_chat` +
  `profile_id`; an empty managed-model/session pin means engine rotation;
  saved-but-undiscovered choices stay visible and editable; a compound effort
  slug plus a conflicting separate effort is a validation error, never two
  applied efforts. The Auto-lane account preference and its one-request
  suppression after a typed refusal are ARCHITECTURE §6 "Caller-owned
  subscription model calls": pin never rotates, and suppression never becomes
  a retry or cooldown (`OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL` and
  `OUROBOROS_FALLBACK_COOLDOWN_SEC` keep their escalation budget).
- Saved intent, generated drafts and live status are different axes: a
  status/catalog failure annotates a loaded row and never erases it; only
  explicit Save or onboarding completion materializes a GET-returned
  candidate; a late preview never absorbs owner edits.
- Owner switches expose the semantic choices the owner can actually make: for
  `OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS`, Settings presents Off / Auto / On —
  Auto IS the unset, surface-aware runtime-mode default and saves the empty
  value (semantics: `config.get_allow_mutative_subagents`).
- Onboarding completes in ONE transaction and `GET /api/onboarding` must never
  persist (the transaction and the 503 unknown outcome: ARCHITECTURE §2).
  There is no second completion path on any host, and the client treats only
  the exact success envelope (`ok`/`runtime_mode`/`restart_required`) as
  completion — a 2xx whose body will not parse is a failure the wizard shows,
  because a silent success discards the restart receipt.
- There is ONE wizard host, the served `GET /onboarding` page; do not
  reintroduce a pre-server or inlined copy. The frame is sandboxed WITH
  `allow-popups allow-popups-to-escape-sandbox` (without those tokens the
  sign-in click is blocked silently), asserted from the login card's own
  markup (`web/tests/onboarding_overlay_sandbox.test.js`). Onboarding and
  Settings share the setup contract.
- Install-time defaults are compiled from LIVE discovery with typed refusals —
  never guessed, never half-applied, never re-derived after onboarding;
  install time is the conjunction of three proofs (ARCHITECTURE §2). A
  once-only decision is never taken on a moment-in-time reading (a spent
  subscription window stays in the preset), and the `next_up` verdict is read
  dual-wire — unified `accountPools` first, legacy per-harness second, never
  re-derived from the profile list; an unknown kind is a fail-safe refusal.
- Agent sign-in consumes the harness row's `setupLogin` field as four states
  (absent = legacy catalog; null = the pinned engine's typed
  `setup_job_admission`; a valid object selects `in_app` or
  `external_terminal`; malformed present data is a gap) — never a
  harness-name branch. External-terminal recovery binds its argv to the live
  handshake's exact engine identity, requires the fresh `--probe` to advertise
  `setup_attach`, and renders the argv through the owning `claudexor_daemon.py`
  consumer without executing the text. Credential-profile DELETE remains a
  thin receipt-preserving proxy; mirror additive response fields in Python
  TypedDicts and `web/modules/api_types.js` together
  (`tests/test_gateway_parity.py`).
- Owner settings writes go through `gateway/owner_settings.py` (the
  lock-as-precondition and `CommitBoundary` contract: ARCHITECTURE §1 "Gateway
  Boundary v1"); pre-commit refusals answer through `unsaved_error`.
  `owner_write_guard` belongs only on endpoints that call
  `_owner_update_settings` (directly, or through `_owner_write_settings`);
  anywhere else it advertises a lock the endpoint never takes.
- A setting only an ENDPOINT may author is disk-only in BOTH directions:
  `config.ENDPOINT_AUTHORED_SETTINGS` is consulted by the loader, the
  environment projection and the generic save's merge skip-list — blocking
  only the request body is not enough, because an env-suppliable install-time
  fact closes its own window before the endpoint runs.
- A control the owner cannot use is worse than none: with no agent
  subscription the panel shows truthful configured/generated API or local
  actors and the session chooser points at Accounts instead of inventing a
  route; a saved unavailable session stays visible, and dispatch returns its
  typed refusal, never an API fallback. Harness lists come from one catalog
  path (`accountRows` over `/api/claudexor/status`; pins via
  `indexProfilesByHarness`, a projection of the same rows), so a pin option is
  called exactly what Accounts calls it.
- Install compilation stays linear and split by semantic owner (the compiler's
  emission rules: ARCHITECTURE §2 and
  `ouroboros/subscription_install_presets.py`); API-only/local-only compilation
  performs zero Claudexor reads; never fabricate diversity or build a
  harness/account/model powerset. `POST /api/onboarding/subagents/preview` is
  the read-only compiler surface; completion commits the visible owner-edited
  value.
- Owner-facing copy says "agent", never "coding agent" — the same
  subscriptions build presentations and run arbitrary tasks; product names
  (Claude Code, Codex, Cursor) are trademarks and stay as they are.

### Transport and late-result custody

- `LLMClient.chat` and `chat_async` accept optional `stream=False`,
  `caller_deadline_ts` and `caller_execution_deadline`; Main opts into
  streaming. Subtract the finalization reserve once at the caller; every
  physical recovery send re-checks the inherited bounds. A socket-phase
  timeout is not an overall wall-clock promise, and a late paid completion
  retains its original attempt.
- Stream consumption completes inside physical accounting; the assembler
  doctrine (strict about completeness, tolerant about form; unknown outcome
  only for a stream that never reached its terminal frame) is ARCHITECTURE §6
  "Caller-owned subscription model calls". An EOF/error/cancellation retains
  private wire evidence and cannot produce a usable partial answer; only a
  structural parameter rejection uses the existing wire recovery — never infer
  a retry from missing stream text or ping cadence. Local, GigaChat and
  Claudexor retain their separate wire contracts.
- Late reviewer reuse resolves the exact operation's complete producer receipt
  from the existing CAS (the binding it must carry: ARCHITECTURE §6 "Late
  completion and typed refusals"); the current surface remains the sole wave
  writer and reducer, paid settlement is recorded once, and no source-file
  existence, preview or matching prompt prose alone grants authority —
  missing/partial/error/mismatched custody never buys another same-operation
  dispatch.
- Managed unknown-outcome recovery uses the existing transport-wait owner
  (`loop_transport.py`) with non-generating upstream observations (what proves
  recovery and what cannot: ARCHITECTURE §6 "Caller-owned subscription model
  calls"): keep the old outcome/cost unknown, apply current
  budget/Stop/deadline before dispatch, and let a control-channel outage first
  rejoin the same accepted operation. No scheduler, provider/model table, paid
  readiness probe or automatic manual-restart recovery is introduced.
- `delegate_wait` supervision's observation beat is separate from its HTTP
  read allowance, and a typed read-only-retryable transport failure is a quiet
  observation hole, not a wake (the per-class reasons and the once-per-episode
  owner line: ARCHITECTURE §6 "Delegated subagents (Claudexor transport + the
  nanny)"); no durable counter or outage latch is kept. After terminal
  cleanup, use the current custody host notice alongside the original
  answer/narrative. Genuine builtin refusals publish typed non-success at
  their producer; acceptance JSON validity and completion cleanliness remain
  separate decisions.

Focused regressions: `test_review_late_cas_recovery.py`, `test_delivery_control_lineage.py`, `test_terminal_custody_notice.py`, `test_delegate_observation_transport.py`, `test_delegate_hold.py`, `test_configured_session_wake_rail.py`, `test_health_invariants_ownership.py`, `test_transport_b_stream_deadlines.py`, `test_llm_wire_corpus.py`, `test_transport_unknown_continuation.py`, `test_builtin_refusal_results.py` and `test_v671_acceptance_convergence.py`. Use the ordinary isolated preflight runner; full provider/renderer smoke remains separate from local fake-provider evidence.

### LLM call rules

Mechanism is ARCHITECTURE's and is pointed to, not restated: the subscription transport
(§6 "Caller-owned subscription model calls"), accounting (§6 "Budget tracking", "Usage
ledger substrate vs. accounting policy"), route contracts (§6 "Context fitting, retry,
and compaction"; DEVELOPMENT §2 "Provider Independence"). Below: the call-site rules
and what enforces each.

#### Subscription transport

- Claudexor model calls are a transport, not delegated reasoning: model content and
  native continuation stay byte-faithful through the purpose-bound engine operation;
  never inject its credentials, run its tools, compact inside the adapter or silently
  repeat a generation. A lost local connection rejoins the same operation ID, unknown
  stays unknown, ACK only after the private CAS owns the exact result. Failed-response
  capture uses the catalog's optional query, frozen before create and reused under the
  same idempotency key (absence keeps the strict legacy result shape); full received
  bytes and exception chains stay private, and diagnostics stay compact in the ordinary
  problem context. A known terminal with unusable output is a settled provider result
  plus local rejection (`stream_rejected`), never unknown or not-dispatched — keep both
  stream-rejection markers across sync, async and process boundaries, and a local
  rejection never rotates accounts.
- Host hints are chosen by their caller from transport capability; an explicitly
  unsupported option refuses rather than being silently dropped and retried. Submitted
  options are recorded beside applied options on the usage row (absent report =
  unknown), and that record covers every submitted option while the owner line speaks
  only for the thinking horizon: the first changed reasoning effort of each model in a
  task emits one typed owner line (keyed by task and model, never per round, naming
  only the reporting route). A mismatch is disclosure, never a dispatch gate.
- The engine's active-turn token is a transport fact: the CALLER owns the slot
  (`llm_claudexor.ModelTurnState` on the loop context; a consciousness wake-up needs no
  slot of its own), the engine boundary is its only writer. Fresh slot per logical
  turn, cleared when dispatch leaves this transport; never derived from message roles,
  prose or the last stored assistant envelope (BIBLE P5); never checkpointed (a cold
  restart starts empty); never forked by a reprepare, thread offload or kwargs copy;
  updated only from a dispatched durable result of a request that carried the field (a
  legacy-shaped exchange is silence, not proof a turn ended); never in usage, events,
  progress or task cards. Opt-in is gated on the last SUCCESSFUL handshake's version —
  not the next-spawn pin, not a liveness projection a failed probe can blank (WHY:
  ARCHITECTURE §6 "The live turn slot"; the `llm_claudexor.py` docstring).
- Pass `model_role` and the captured account explicitly at every helper/reviewer seam
  (Main and Light may share a model name with different pins; account evidence stays
  source/profile/fingerprint-bound). Manual context sizing is not scope authority; a
  scope ACK binds the actual route; a changed model's token-density observation never
  becomes the old model's evidence. A physical attempt limit returns a claim only after
  a successful, positive never-dispatched release; unknown or dispatched claims stay
  charged.
- Resource refusals wait inside the live call, before helper catch-all blocks, on the
  existing task owner, mailbox, clocks and settings writer — no parked rounds,
  compensation processes or replay of completed tools/reviews (ARCHITECTURE §6 "Quota
  and auth waits"). Typed errors cross the tracked image child intact; the shared
  waiting card keeps its revision fences and accepted/applied/saved distinction, and a
  browser fixture never invents an acknowledgement protocol the real ingress lacks.

#### Call sites and accounting

- New LLM calls go through the shared `LLMClient`/`llm.py` layer — no ad-hoc HTTP
  clients or provider SDKs outside it (review gate: CHECKLISTS item 2(e)). Exception:
  skill/extension `plugin.py` modules may call providers directly until a host-mediated
  bridge lands; runtime callers inside `ouroboros/` must use `LLMClient`.
- Canonical messages/tools stay provider-neutral and function-shaped; a dialect is an
  outbound projection plus inbound normalization, never a mutation of stored history or
  a second compaction/replay contract. Custom-origin receipts stay private and
  catalog-bound (`ouroboros/request_wire_custom_validation.py`); one request-wire
  driver, ladder ordinals fixed at 1/2/3, custom→function never persisted as learned
  dialect, no Responses migration, owner `none` on direct Anthropic =
  `thinking.type=disabled` (`tests/test_request_wire_contract.py`,
  `tests/test_openai_chat_custom_contract.py`, `tests/test_anthropic_native_custody.py`).
  `usage.request_wire` is one call's terminal candidate; nested aggregation keeps the
  ordered `request_wire_history` with explicit omission accounting.
- Every core-mediated physical provider send goes through
  `usage_accounting.execute_physical_attempt[_async]` (`tests/test_usage_accounting.py`);
  custody classifiers read the explicit `__cause__` chain, never `__context__`, and an
  ambiguous timeout stays unresolved (`tests/test_transport_custody.py`).
- Administrative abandonment never turns a reservation bound into an actual price:
  use the existing unknown-price settlement and retain correction-eligible attempt
  chains across compaction. One real late receipt or positive never-started proof may
  correct that attempt; ordinary terminal rows stay immutable, and full/incremental
  validation must agree. Reconcile through existing custody maintenance only after
  physical ownership ends, preserve review owners, and read exact recorded model
  operations without creating new work. Retry existing cost projections independently
  of another ledger transition, including after compaction, using one indexed
  maintenance-drive view rather than filtering it for each owner. A different
  recorded budget root keeps its own accounting path; never fabricate completion
  (ARCHITECTURE §6 "Budget tracking"; storage rules and tests:
  `docs/USAGE_COMPACTION.md`, `tests/test_usage_abandoned_ledger.py`).
- Hold the usage-ledger cross-process lock only for budget check, validated append and
  fsync — never over network I/O; a caller that owns a finalization reserve passes it
  explicitly so admission and the transport bound cannot disagree.
- Keep root ceilings explicitly unreserved under the shared pool; persist the applied
  global limit and its source/revision on the physical attempt through every
  transition (a missing revision is unknown, never the settings-file hash). Pacing
  facts reuse the note cadence and cached money projections; typed tool results count
  incrementally on the loop usage carrier; durations are overlapping observations, not
  inferred sleep/poll time or a behavior gate (`tests/test_budget_resource_facts.py`).
- Tree-spend pacing decides on root-subtree spend including in-flight holds,
  publishes the same `CostCeiling` object the loop decides on, and prices the wrap-up
  with the fence's own cache-aware reservation (`tests/test_network_budget_wallet.py`).
  Explicitly disabled profiles and real monetary fences stay independent; the
  configured global budget is read through the one resolver, never an inline default.
  Post-task consolidation/synthesis reads one frozen `usage_breakdown` snapshot per
  root subtree (never `$0` on a read failure); no second ledger, no reconciliation LLM.
- Runtime notices after the first user/assistant/tool turn are `[SYSTEM NOTICE]` user
  notices, not new `role=system` messages; `LLMClient` demotes non-leading system
  messages at the provider boundary.
- **Cache-friendliness invariant.** Byte-stable governance and task contracts precede
  mutable evidence; never put timestamps, hashes, counters or task identity in a stable
  cached prefix — they fragment provider caches while conveying no stable policy.
  Builders declare bare breakpoints (`review_substrate.assert_cache_breakpoint_cap`
  keeps the review builders at four or fewer; `tests/test_review_prompt_caching.py`);
  only `LLMClient._normalize_payload_cache_ttl` finalizes the wire payload; no provider
  hops, body rerouting or generic cache/retry framework. A wrap-up call keeps schemas,
  the server-web flag and `tool_choice` identical to the working round and instructs in
  text (a tool-less variant rebuilds the whole prefix; a `tool_choice` change rebuilds
  the messages tier). `context_fit.seal_task_transcript` owns the single message-side
  breakpoint — the task message until the rolling tool-result seal qualifies, migrated
  in the same call — preserved on the direct-Anthropic lane by
  `_anthropic_blocks_from_content` and on OpenRouter by `supports_message_cache_control`,
  pinned by `tests/test_review_prompt_caching.py` (ARCHITECTURE §6 "Context fitting,
  retry, and compaction"). The subscription transport carries one install-scoped cache
  affinity (`llm_claudexor.cache_key_for_model`: one Codex `prompt_cache_key`, hence one
  `session_id`, per data root and model, shared by every task, child and consciousness
  cycle — Codex reuses a prefix across conversations only under the same session;
  ARCHITECTURE §6 "Caller-owned subscription model calls"); API-compatible lanes keep
  prefix-derived session identity. A consciousness wake-up shares an owner turn's
  byte-identical schema array and system prefix, so what the level or wake reason
  changes lives only in the wake's user message and the dynamic tail; its model slot
  (the owner's `consciousness` role, when set) decides which cache it lands in. Between
  sends of one execution, only compaction may rewrite the transcript; other breaks
  discard OpenAI-family caches (`prompt_prefix_break`; ARCHITECTURE §6 "Task lifecycle"). `_append_or_merge_user_content` never merges into acceptance
  observations. Other content merges only if `unsent_in_previous_send` proves the
  tail absent from the last observed send; without a slot or observation, append.
  Observation follows a usable ordinary response, not every physical send; image
  eviction is unchanged. Pin plain/multipart content and real local/GigaChat builders
  (`tests/test_transcript_prefix.py` on `run_llm_loop`,
  `tests/test_transcript_provider_shapes.py`); CHECKLISTS item 22 (`cache_friendliness`).
- Provider fallback is disabled only for a SEALED reasoning artifact
  (`ouroboros/reasoning_artifacts.py::transcript_has_sealed_reasoning`) — only a sealed
  artifact is bound to the endpoint that minted it; readable reasoning stays
  failover-eligible for every family so one outage does not strand valid work
  (`tests/test_llm_provider_routing.py`).
- Delegated agent sessions and the native review inspection episode get the full
  governance prompt; never truncate BIBLE/ARCHITECTURE/DEVELOPMENT/CHECKLISTS to fit
  argv or transport limits.
- Delegated (subscription-harness) work is accounted on its OWN ledger row —
  `usage_accounting.record_subscription_session`, never
  `record_unmetered_external_dispatch` (it drops the sessions/quota axis); token `None`
  means unreported, not zero (cash cases, `input_token_usage`: ARCHITECTURE §6
  "Delegated subagents (Claudexor transport + the nanny)";
  `tests/test_gateway_usage_accounting.py`, `tests/test_delegated_run_custody.py`). Skill
  Review waves attribute every usage row with the exact wave/slot identity; pre-marker
  waves stay "exact attribution unavailable", never reconstructed by time or model
  (`tests/test_skill_review_usage_accounting.py`).
- `cost_final` is a COUNT of open rows (`non_final_rows`), never a truthiness test on a
  dollar sum. A spent subscription window is `subscription_window_exhausted` (TRANSIENT,
  carries `reset_at`), never folded into `quota_exhausted`, which is permanent for a
  billing refusal and wrong for a window whose cure is waiting
  (`tests/test_reviewer_slot_config.py`).
- Classify a provider failure before repeating the request. The combined Anthropic
  input-plus-max_tokens rejection is a context-window overflow (keep output/body-size
  precedence in the shared context_budget classifiers, without requiring the input
  alone to exceed the window); quota/auth/billing, hard bad-request and
  request-too-large are non-retryable as-is (exact category, recovery hint); a typed
  408/429/5xx or a proven pre-dispatch failure may retry; a dispatched request with no
  terminal outcome stops same-model and cross-model sends until reconciled. Who may
  repeat after a typed transport death, how often, on whose row, what ends the round:
  ARCHITECTURE §6 "Context fitting, retry, and compaction"
  (`tests/test_transport_death_retry.py`). Call-site rules: decide `retry_same_request`
  before the durable row is written; only a proven refusal — `llm_not_dispatched`
  (deadline admission), `llm_retry_deadline_exhausted` (deadline backoff), or a typed
  finalization control during the paid-repeat wait (`finalize_control_pending`) — takes
  a never-sent grant back off the round record; reuse the interruptible sleep and the
  mailbox/current-intent readers (peek, no delivery or ACK; input/hurry/revoked controls
  untouched); generic transient/empty-response backoffs keep their contract. A budget
  refusal does NOT un-count: the budget rail cannot prove the repeat never left the
  host (`llm.chat` retries on the wire before a later reservation can refuse), so the
  attempt stays booked and the budget terminal ends the round. Every caller outside the
  interactive primary rail keeps `transport_death_retries=0`; no
  consumed/terminal/patch-disposition predicate gates a session supervisor's cognition,
  and a successful live-leaf hold closes any prior transport episode so its
  acknowledged wake alone resumes the model.

#### Timeout & Wait Control

- Required owner waiting keeps the original execution: RUNNING, the worker lends only
  active capacity, waiting exempts only the idle timeout — Stop, deadline, absolute
  ceiling and monetary admission still bind (ARCHITECTURE §5 "Supervisor Loop").
  Persist the completed-tool source, task wait and queue snapshot before lending; grant
  the original worker only after reserving active capacity (both marks restored on
  failure); attempt, start time, completed effects and usage are unchanged across a
  warm wake; cold recovery needs the acknowledged planned-restart handoff through every
  shutdown cleanup, and a direct-actor checkpoint alone grants none. After either wait,
  control/deadline handling precedes the saved round's budget decision, and
  TaskModelWait role overrides, explicit Auto, auto-continue and the completed quota
  union survive through that owner's continuation methods; calendar deadlines and
  owner-wait time keep their meaning (`tests/test_owner_wait_pool.py`,
  `tests/test_owner_wait_restart.py`, `tests/test_owner_wait_cold_loop.py`,
  `tests/test_owner_wait_budget_tail.py`, `tests/test_owner_wait_model_context.py`).
- For a session nanny, `delegate_wait` is event-only at the model surface: host
  supervision renews bounded transport windows at zero LLM calls, journal progress
  streams to the owner without waking the model, and only terminal/interaction/fault,
  an addressed task/owner message, a direct-child signal, control/recovery judgment or a
  model-requested one-shot checkpoint wakes it. No caller-visible `wait_sec`, repeating
  timers, progress wakes or host semantic stall detector.
- Wait/continue/stop is a structured fact — terminal status plus heartbeat freshness
  from `queue_snapshot.json` via `task_status.py` — never a keyword or regex over
  content (BIBLE P5). Fixed kill-timeouts (hard task/tool ceilings, watchdog) stay the
  outer safety bound; progress-aware waiting tunes only the passive wait.
- Raw terminal model/salvage bytes stay separate from the host-authored
  `terminal_provider_notice`; receipts and secondary notices consume the same facts
  (attempted repeats, last provider error, unknown dispatched outcome), and a retained
  answer must not hide wait or unknown-attempt evidence or invite a blind rerun.
  Message/deferred Presence responses render one host-labelled status section (cached
  output is already rendered); the Presence renderer preserves silent/tool-delivered
  authority and never changes raw answer bytes. Carry the actual control reason through
  wait termination: owner Wrap up is distinct from deadline/budget finalization and
  sends no new summary request.
- Transport-wait notes use the progress seam with `incident=None`; other producers'
  incidents keep their typed `task_incident`, `toast_once` and optional `toast_tone`;
  never infer urgency or valence from a task's prose
  (`tests/test_loop_transport_wait_interactive.py`). A cross-model lane switch carries
  the same incident pair naming both models, the account the send's own binding
  selects when the route has accounts (a task-local wait override included, never the
  configured value alone) and the typed failure reason when the round record has one;
  the applied-option mismatch line rides the same callable, and the frozen
  `ToolContext.emit_progress_fn` takes one argument, never the pair.
- Timeout classes are separate axes. A transport timeout
  (`OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC`) bounds only a dead socket — never a
  reasoning cutoff or evidence of a stall. API review uses it as a settlement fallback
  (that request ends there); a delegated agent session inherits the task absolute
  ceiling (the paid run can outlive an HTTP read); the owner deadline narrows either;
  provider transport defaults (Anthropic, VLM captioning) are ceilings, not promises.
  Default reviewer slots deliberately have no short cognition cap; the outer `plan_task` envelope
  covers the session lifetime; `web_search` sizes its envelope for the complete
  configured paid cascade, recomputed under an owner deadline.
- A new numeric timeout constant is an SSOT in the owning settings leaf, not the
  `config.py` facade: key and shipped default in `settings_defaults.py`
  `SETTINGS_DEFAULTS`, the clamped getter in `runtime_limits.py`, both re-exported
  through `ouroboros.config`, the one import surface; register the env key; no magic
  wait numbers at call sites (`tests/test_timeout_policy.py`).
- Worker readiness keeps `WORKER_READY_WINDOW_SEC`, `WORKER_READY_CEILING_SEC` and
  `WORKER_READY_MAX_ATTEMPTS` in `runtime_limits.py`, re-exported by config
  (ARCHITECTURE §5 "Supervisor Loop"). Reuse the lifecycle-owned execution-state
  reader (workers facade) at reserve, final enqueue and snapshot; keep the separate
  repository-writer policy at public admission and the boot/update exceptions;
  the child's own `worker_starting` row before extension loading permits one
  readiness extension to 300 seconds from birth, never a sliding deadline or a
  fresh window at observation. Readiness, process liveness and idle deadlines stay independent; a failed write keeps
  terminalization retry, never a false Done or a fresh startup budget.
- Nested process wrappers are ordered, never tied: provider bound before its killable
  child, child before the generic ToolEntry envelope (the settlement margin from
  `config.py`), so no result arrives after its owner abandoned custody. The two
  deliberate early returns — plan review's and task acceptance's dispatch barrier
  (`ReviewRequest.drain_deadline`) — keep custody: workers settle into process-local
  custody and announce the wave through the task mailbox (`plan_review_collect`;
  `acceptance_settlement.announce_acceptance_settlement`, at the wave's own quorum and
  at completion, each reviewer's own verdict, never an instruction to collect).
  `owner_hurry.force_plan_decision` collects once at zero wait before finalization in every enforcement mode, hurry
  included, and projects the returned state; task acceptance collects through the host
  reconcile `review_dispatch.reconcile_pending_acceptance_runs` (recorded request and
  roster replayed, nothing sent), never a model-callable verb; context health reads
  only the canonical wave, as a recorded snapshot. Neither path dispatches a second
  panel.
- Every physical LLM/review/VLM/tool operation that can outlive a logical wait emits
  typed `cognitive_operation` start/terminal facts; the supervisor uses the
  active-operation map only to spare the idle rail, and a terminal must match
  task-attempt plus execution/round/call identity to clear the row. A logical timeout
  over a live paid worker is custody/reconciliation-pending, never a blind paid retry;
  late results settle the original attempt under its retry identity. An owner terminal
  that is not a deliberate verdict is no permission to cancel the paid run it held: the
  sweep spares and discloses it, its own bound limits the damage.
- Once the owner deadline minus finalization reserve is spent, an unstarted review row
  is a typed `$0 not_dispatched` actor (no worker, paid stamp or lease); an in-flight
  paid wave stays eligible for exact custody reconciliation without a new dispatch. An
  in-flight reviewer never counts as final quorum under either enforcement mode; a
  `pending_dispatch` row (released at the barrier) is neither quorum nor a paid fact
  until its settled row proves the send.
- Every zero-physical acceptance refusal (unresolvable partial source, immutable-core
  overflow, a slot window too small for the rendered prompt) records a typed
  `$0 not_dispatched` row with its cause in `error` and folds the aggregate to
  `DEGRADED` (the one $0 shape: ARCHITECTURE §6 "Review stack"; instances: "Task
  acceptance").
- A returned provider response (empty body included) or typed terminal 408/429/5xx is
  settled and may use the surface's bounded retry rail; `dispatched`/`unresolved`
  without a typed terminal stays custody-lost/no-resend (ARCHITECTURE §6 "Physical
  custody"; the interactive repeat rail: "Context fitting, retry, and compaction"). A
  NEW logical request needs a unique host-attested input the unknown one lacked: the
  managed upstream-recovery notice or the nanny-leaf wake contract in
  `ouroboros/delegate_hold.py`.
- A custody retry key names semantic material and an admitted cycle, never its
  rendered prompt (ARCHITECTURE §6 "Physical custody"; Skill Review's
  `review_resume_of` rejoin: "Late completion and typed refusals"); an unstarted chunk
  cannot authorize PASS; a window with no dispatch capacity leaves an unpaid `$0` wave
  and no paid stamp.
- A reviewed mutative wrapper keeps foreground custody until the workflow settles;
  never abandon a live reviewer or commit pipeline on the generic 600s tool default or
  a guessed ceiling.
- Cooperative cancellation applies only where the route supports it (delegated
  sessions); API/thread routes disclose an in-flight custody state until the result
  settles. A typed transport failure after a delegated run has an id is unknown:
  retain the durable invocation token and replay that run on the permitted retry,
  never post a second paid run; a retry token without a valid durable invocation is
  `review_custody_lost`; elapsed TTL never authorizes a resend, and owner death is
  proven only by pid death (ARCHITECTURE §6 "Paid stamp and owner custody").

### Loop and acceptance state machines

#### Loop / State-Machine Changes

- Changes to `loop.py` or other task state-machine logic include adversarial tests —
  malformed output, false-completion prevention, replay/log durability, failure modes —
  not just the happy path. Audit/checkpoint rounds never silently reuse the normal
  final-answer path unless that invariant is explicitly tested and documented.
- Keep a complete loop-local `DeliveryCandidate`; sticky host-control provenance survives
  replacement (`ouroboros/loop_delivery.py`; ARCHITECTURE §6 "Task lifecycle").
  FORCED resolution: pure, no retry; honor valid keep/replace, preserve malformed controls'
  candidate with a typed degraded reason; no protocol JSON in chat/durable results.
  Distinguish consumed owner source from changed requirements; effective criteria and
  material effects (nominated reads included) define the subject; ingress generations
  preserve unread order. Status/narration/working-view changes buy no review. Finalize
  task-scoped service outputs/errors before acceptance. Controls never bypass
  verification, acceptance, safety, skill finalization, deadline, child handoff,
  unconditional `FINAL ANSWER:` or the task-level answer protocol. Host notices stay
  outside answer bytes/identity via terminal record/outbox/System on replay and
  single-body/headless transports; unchanged answers never revive superseded verdicts.
- Keep delivered result, unresolved tool-call evidence and host acceptance separate.
  Error count alone does not degrade execution or establish objective acceptance;
  retain `execution.unresolved_tool_errors` and the cosmetic bucket, and expose the
  existing no-review warning for either when the canonical objective is
  `not_evaluated`, including after delivery/child-state normalization. Test
  every verdict and the no-review case while preserving stronger typed terminal causes
  (ARCHITECTURE §6 "Task lifecycle"; `tests/test_outcome_tool_error_axes.py`).
- Every direct child result needs an exact-hash disposition through the existing
  `tree_note(kind="decision")` payload (`type=child_result_disposition`; the batch form
  validates entries by index). The typed task-tree row is the sole authority;
  task-result disposition fields are derived reads, never a mirrored write. The
  SHA-256 binding means a parent cannot claim it integrated a result that later
  changed; `deferred` suppresses only the reminder and forces an honest
  degraded/best-effort terminal until resolved; the payload schema is the SSOT for
  choosing a value and its enum reads the validator's own set. A child in the legacy
  `cancel_requested` latch is intent, not outcome — cancel-pending until custody
  settles it.
- Host acceptance: root-only, structured eligibility
  (`outcomes.turn_has_reviewable_effects` plus a typed deliverable/criterion), never
  keywords or authoritative agent nomination (BIBLE P3/P5; acceptance model,
  per-enforcement waiting, `previous_revision_accepted`, `late_settlement`:
  ARCHITECTURE §6 "Task acceptance"). Freeze request/roster; existing review custody/
  mailbox handles pending/free collection. Before new-panel evidence or
  `review_cycles_exhausted`, reconcile every paid panel still marked running for that
  root: $0, recorded request/roster; reauthoring loses no verdict. Settlement wakes bring
  verdicts whatever Main's draft. Re-offer only changed contract bytes; a spent repair
  stays spent. Settled panels/queued wakes
  skip parking, not retained-answer control preparation or typed provenance.
  Accept complete revised prose, never a status note; typed keep/replace/finish are
  optional. Prose resets pending-review choice to wait, never infers finish.
  Effect, owner-revision and
  child-action controls stay strict; owner-source acknowledgement and forced
  finalization retain their rules. Context-only mail wakes waits but does not block
  owner-source acknowledgement or imply an owner revision. Empty or recognizable malformed controls retain
  the answer (`tests/test_acceptance_optional_control.py`). Ready or pending feedback
  needs no extra panel or capacity refusal for delivery
  (`acceptance_settlement._deliver_under_running_panel`). Pending: default wait;
  Blocking waits; Cyber Pro never waits; Advisory finish needs explicit
  `"pending_review":"finish"` in delivery control. Keep the trace
  past exit (`remember_settlement_trace`). Late settlement: attach to the ended result,
  announce once on its task card (`card_row="reviews"`); no model turn or reviewer-as-
  open-delegation. Workers never write Main's candidate/author decision; subtree/status,
  findings and Cyber authority stay separate (BIBLE P0).
- Delivery-control JSON governs only tool-less final responses; retention leaves tools
  available. Changed criteria/material evidence mean a new subject even with kept text,
  never old verdict authority. Source acknowledgement infers no semantic change from
  generation. File/diff requests impose no commit-or-revert rule; self-modification
  keeps reviewed commits (BIBLE P0/P3).
- Before cleanup, freeze `review_evidence.task_inputs` and `completion_observations`
  for summary/reflection (ARCHITECTURE §6 "Post-task reflection"): whole owner Q/A,
  peer provenance and canonical split-root verification receipts. Zero exit is positive;
  absent is unknown; unrelated passes erase no failure. Send content, not pointers;
  recover the same snapshot. Count delivery via `OWNER_DELIVERY_TOOL_NAMES`, never
  global skill state. Summary uses `chat_observed` custody and the task-scoped,
  archive-aware trace reader.
- Promoted tasks carry their host-minted root id and role on the queue payload.
  RUNNING writes preserve the actual `_task_started_ts` as `started_at` and an existing
  `queued_at`; terminal `ts` stays its own field; missing historical start facts stay
  missing. LLM usage carries the existing call/execution/round ids (it joins the
  worker's round without duplicating it); delegated settlement/disposition/unread rows
  carry their root/parent ids; intrinsic pacing exposes the same
  `cost_ceiling_disclosure` its text uses; tool error manifests carry their typed code
  and redacted reason preview. Observation links, not new accounting or zero-price
  rules.
- Acceptance evidence identity hashes source facts before history-dependent budgeting;
  recording a review never changes the facts it reviewed. Complete applied host records
  go through `review_projection.publish_acceptance_checkpoint` via the write-once
  source handles before compact publication (copy-back, CURRENT-basis, same-store:
  ARCHITECTURE §10 "Key Invariants"; the reader: §6 "Post-task reflection"). Artifact
  registration keeps its short locked manifest merge — copy/hash before the lock, which
  never takes a task-result lock. Review/completion sources live in
  `source_handles/context_checkpoints`, outside deliverables and the acceptance
  manifest; terminal references carry the task's chat id, zero included; a missing
  source is disclosed, never rebuilt from a preview. Source/capacity, publication order
  and paid identity are separate contracts; history or presentation changes mint no
  work. Tests: delayed snapshots and child replicas through the central merge; the full
  source downloads while the task still runs; the persisted consumer verified after the
  real merge and child cleanup; an operation-scoped memo reuses verified work but never
  caches failure or becomes a second store.
- Mirror a split root's actual execution start and child-drive binding into its
  canonical result through the existing terminal-preserving writer.
  Recover a legacy missing binding only from positive known-child start evidence
  plus the existing fresh-queue/later-worker orphan proof, never while pending or
  actively cancelled and never as permission to resume execution.
- Pooled terminal file preparation belongs to `headless.prepare_terminal_task_files`
  at the worker's task_done boundary — after blocking post-task work, before the slot
  is released; earlier answer/metrics delivery stays early (ARCHITECTURE §5 "Supervisor
  Loop"). No I/O exception or lost event authorizes model replay; never persist
  `terminal_source_present` as an anchor.
- Health owns terminal-file preparation/recovery; the existing reaper owns queue
  execution and deferred-job replay on the health cadence (ARCHITECTURE §5 "Supervisor
  Loop"). Preserve worker/meta/task/attempt/root identity across off-lock operations;
  the normal terminal event owner keeps queue release and project/evolution hooks — no
  separate crash executor; crash terminals withdraw their captured RUNNING owner before
  emission; cancel checks file readiness before source removal; deferred timeout jobs
  keep their original binding so old recovery cannot kill, requeue or replace a newer
  execution.
- Same physical observability store means verified reuse of original manifest bytes
  and canonical path spelling — never a rewrite or native promotion marker; missing
  aliases resolve only through the exact verified CAS/call readers (ARCHITECTURE §10
  "Key Invariants"); no digest filenames, initial-adoption anchor or persistent
  transfer store.
- Pooled mailbox cleanup follows the file helper's settled-cleanup predicate; startup
  recovers terminal child sources before the prune (ARCHITECTURE §5 "Supervisor Loop");
  direct canonical cleanup stays direct, never races unknown prior ownership, needs no
  saved anchor.
- Acceptance payment follows the semantic subject and substantive disposition identity
  of ARCHITECTURE §6 "Task acceptance": source generations, read repetition or
  narration alone buy nothing; changed criteria or material evidence can, even with
  identical text. Reuse the existing subject and paid-identity owners — no second hash,
  no cosmetic edits.
- Task-acceptance actors are the configured triad rows
  (`reviewer_slot_config.triad_delivery_slots`; malformed config refuses typed), one
  substantive interaction each on its own delivery; the retrieving work order,
  `evidence_refs` against the FULL packet, the money rule (one work-order send per paid
  row, no rounds multiplier, no second pricing pass), the once-per-panel launch floor
  (`task_pacing.review_launch_allowed`, `task_acceptance_paid_dispatch_stamp._claim`),
  the clamps on a running panel and the deadline-cut residual are stated once in
  ARCHITECTURE §6 "Task acceptance". Format-repair resends are packet-row only; child
  and `off`-mode acceptance run packet rows only.
- The host acceptance decision is written ONLY by
  `loop_acceptance._set_acceptance_decision` (re-exported from `loop`): three
  owner-facing states, each with a typed reason from the closed set, unknown fails
  closed. A new writer adds its reason to the set AND checks every value-keyed reader —
  `outcomes.derive_loop_outcome` keys on status+reason PAIRS, and a broken pairing is a
  silent false green. Every forced rail closes a dangling `revision_requested` via
  `loop_acceptance.terminalize_dangling_revision` (ARCHITECTURE §6 "Task acceptance").
  `PASS|FAIL|DEGRADED` is NOT narrowable; `adaptive_quorum` applies, any contributing
  FAIL fails and DEGRADED abstains in the critic aggregate. Apply qualified Advisory
  author completion separately, never rewriting criticism or hiding independent failed
  effects, unaccepted review or unfinished stops (DEVELOPMENT §11; ARCHITECTURE §3).
  No task scope review or commit-gate reuse.
- Keep reviewer DIALOGUE evidence: typed `disposition_kind`/`obligation_id` identifies
  obligations; disclose unknown re-raise ids as `new`. Reopen rows with
  arguments intact. A terminal critic vote cannot deny author reaction or choose its stop. Blocking may save corrections and stop; advancement needs fresh
  reviewer authority. Advisory may explicitly finish revisions after exposed feedback
  or disclosed unavailability without another panel. Keep critic/author hashes separate;
  bind intent to delivery evidence; consume it on owner/evidence supersession.
  Queueing is not exposure; predeclared finish cannot authorize unseen feedback;
  `author_action=stop` grants neither completion nor permission. No semantic counters or
  keyword gates (P5). ARCHITECTURE §6 owns material-only continue, invalid-vote abstention
  and typed `inconclusive`. Test malformed output, unknown/stale ids, partial
  failures, disagreement, obligation replay/restart, false completion and missing fields.
- Explicit task-local `max_improvement_passes=p` retains p author responses and p+1 paid
  ceiling under all policies, including 0/1/6. Otherwise
  `OUROBOROS_REVIEW_MAX_CYCLES` limits paid panels, not author responses: last feedback
  permits work within ordinary task time/budget/cancel/round rails. The reviewer admission
  floor applies only to new critics. Keep settings-load migration of retired
  `OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES`; it never binds at runtime.
- A `PATCH_DISPOSED` row names its disposer (`disposed_by_task_id`) because a
  non-owner may write it once the owner task is terminal; wait/cancel/answer stay
  owner-only. A terminal custody obligation is disclosed additively (objective warning
  plus reason code, a truncation rail code preserved); turning a custody fact into a
  review, objective or execution verdict is the defect this rule prevents.

Enforcement: the adversarial tests the first bullet mandates, plus
`tests/test_child_result_disposition.py`, `tests/test_acceptance_fence.py`,
`tests/test_v674_acceptance_dialogue.py`, and `tests/test_review_cycles.py`
(cap migration).

#### Cognitive Artifact Integrity

- Cognitive artifacts (identity.md, scratchpad, task reflections, review outputs,
  pattern register) must NOT use hardcoded `[:N]` truncation. When content must be
  shortened, summarize explicitly — attempts, changes and conclusions survive — and
  disclose the omission with a resolvable reference; an omission marker alone is
  disclosure, not sufficiency (BIBLE P1).
- Governance residency is per mode and per actor, never universal (the per-flow
  context-delivery registry and its disclosed residuals: DEVELOPMENT §4 "Core
  Governance Artifacts"): `prompts/SYSTEM.md` and `BIBLE.md` are tier-0 and full in
  every projection; in Max, ARCHITECTURE is full-resident and DEVELOPMENT follows the
  active repository binding; in Low and Nano BOTH books are replaced by their book
  navigation (`context_layout.book_navigation`), and the explicit per-task
  `context_requires_self_body_docs` override is not honoured there (issue #1019); a
  delegated subagent child receives that navigation in every mode (issue #1026). A new
  reasoning flow MUST follow that contract, never rely on
  touched-file inclusions.

Enforcement: review-only — CHECKLISTS item 2(f) scores the no-`[:N]` rule in
commit review.

## Android platform development

Android keeps the common source/update/review authority. Its runtime layout,
launcher/seed distinction, native artifact receipt and lifecycle owners live in
ARCHITECTURE "Android host (experimental)"; the user procedure is
`docs/ANDROID_INSTALL.md`. Keep experiments described as rooted ARM64 Android,
physically tested on Pixel 10a only. Do not turn that tested model into a runtime
allowlist or treat the manifest's minimum SDK as a proven support matrix.

Native changes use ordinary Android source, reviewed commits and the persistent
per-install signing identity. Never copy the publisher private key, runtime
credentials, memory or a live rootfs into release assets. Verify official bytes
before trusting their source; distinguish a publisher-signed reference APK from
the locally built installed APK, and record each artifact's own identity. Restore
the original personal key after loss; do not silently generate another identity
for an installed package. Preserve source/data/key on installer retries and use
the existing managed Git merge to retain local evolution during official updates.

Dependency changes must reach the same source-selected hook: `ensure_platform`
serves first installation and later updates, with package, Java SDK, native AAPT,
common Node and browser input groups in the existing SDK receipt. Include tracked
recipes and patches; retain `platform_preparing` until completion so interruption
and Git rollback cannot reuse a partially changed SDK as current. Verify installed
output hashes and desired-source stability before native success. Use the existing
verified download cache, Node manager and Playwright installer; do not start a
second daemon. A legacy receipt is prepared once; Java SDK-only changes skip AAPT
compilation. The common `EXTERNAL_PLATFORM_UPDATE_TIMEOUT_SEC` constant is 3600
seconds in `runtime_limits.py`, re-exported by `config.py`, with the existing
ProcessContainer cleanup. Start the HTTP readiness clock only after server spawn, not while native preparation runs. Propagate launcher shutdown into the hook and await its owned cleanup before exiting, including a later generation. Preparation completion is not HTTP or native-success evidence. This bound is not an environment setting or a harness deadline.
The Ubuntu Base archive remains initial seed provenance. Same-Noble apt recipes
can evolve, with actual installed package versions recorded; Git rollback does
not promise package removal/downgrade or a major distribution migration.

Exercise actual source → dependency preparation → build → install → readback → restart behavior before
claiming native adoption. Cover failed native builds alongside a still-usable
core, older immutable seed with newer source, local APK modification followed by
an upstream source merge, bootstrap-only changes, rollback to older native source
with a newer versionCode, bridge loss/recovery, and Panic versus automatic entry.
Core HTTP health is separate from native artifact and bridge readiness; checks
must use their existing owners and preserve incomplete outcomes.

Keep delegated access with the common actor/delegation owners described in
ARCHITECTURE "Agent Core". Android must not acquire a second access default,
trust policy or retry implementation. Qualify the actual native route separately;
a sandbox failure is not a successful run or permission to disguise a retry.

On Android, `enter-linux` restores ordinary OOM selection for its own process and
descendants without removing root. The existing
`OUROBOROS_PREFLIGHT_TEST_WORKERS` operator lever defaults to 2 and
`OUROBOROS_PREFLIGHT_TIMEOUT_SEC` to 3600 seconds at this entry, preserving explicit
overrides. Other installs retain the upstream 1800-second total test budget.
Standalone preflight/advisory ToolEntry bounds add that resolved test total to the
existing plan-style task/transport settlement envelope and finalization grace;
they must not expire before tests and the critic can settle. This outer bound
creates no new cognitive deadline; inner critic/owner deadlines, test containment
and the reviewed commit's terminal wait remain unchanged. These settings change
test concurrency/time, not test content, review models or context. Measure memory/swap and confirm process cleanup before
running full preflight on a phone; do not deliberately reproduce a kernel panic.
Keep reusable large downloads in the installer's durable cache.

`android-test` explicitly collects `android/tests`; ordinary `pytest tests/` does
not cover that directory. Portable source/transport fixtures and host compilation
are separate from physical root, boot, permissions, hardware and battery evidence.
The same-key instrumentation under `android/tests/device` owns a temporary SDK
bridge and an uncommitted PackageInstaller session. The emulator job runs for Android source changes and tags, and executes
session readback on API 26/29/30/33/36 and accepts its explicit PASS only after
abandon and bridge cleanup. It requires neither root nor a provisioned Linux
core; it does not certify the phone bootstrap or owner consent UI.
Android release source/APK SBOMs describe those shipped bytes; installed dependency
pins/package inventories describe the provisioned phone. Neither invents the other.
The trusted tag-only `android-build` job reads `ANDROID_KEYSTORE_BASE64`,
`ANDROID_KEYSTORE_PASSWORD`, and `ANDROID_KEY_ALIAS` from repository secrets;
the branch/PR Android jobs use a disposable key and never publish it. A PR is
therefore source/build evidence, not a publisher-signed release claim.

---
