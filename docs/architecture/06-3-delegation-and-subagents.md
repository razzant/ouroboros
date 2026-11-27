# 6.3 Agent Core — Delegated Subagents & Vendor Maintenance

This part gives me helpers: API children and nannies scheduled with `schedule_subagent`, and external coding-agent sessions through the managed Claudexor transport (`delegate_start`, `delegate_wait`), plus the maintenance of those vendor programs. I read a result before any patch lands (`get_task_result` by a child's task id, `delegate_wait` by a run id) and integrate an eligible patch explicitly (`integrate_subagent_patch`, `integrate_delegated_patch`). A child inherits cancellation and receives my owner's later words as context, never a fresh grant, silent write authority or unreviewed promotion into my body; cancel intent precedes proof of exit.

### Delegated subagents (Claudexor transport + the nanny)

Delegation adds no engine review panel; the start's `engine_version` is the handshaken serving version. Engine review, execution, parent integration and Ouroboros gates are separate facts.

#### Child coordination, registry and scheduling

Tasks coordinate via `tree_note`/`tree_read`; a parent releases a child's constraint, a root its own. Read-only and acting children hold descendant-scoped `forward_to_worker`, `peek_task`, `cancel_task`, `discard_child_result`; recursion widens no filesystem, budget, depth, deadline, commit or owner authority. Task mail reaches same-tree peers as context-only `peer_task`: shared root is not ancestry (`owner_mailbox.py`). `delegation_budget` governs descendants; persisted admission outranks Settings; reduced depth reports `capability_reduced`, never a silent flat tree.

**Registry.** `configured_subagents.py` owns strict `OUROBOROS_SUBAGENTS {enabled, items}` (≤ `MAX_CONFIGURED_SUBAGENTS`): id/handle (§3), guidance, API/session route, effort, pin, access, enablement. Off keeps settings but removes catalog, alternative and reviewer eligibility; `subagent_disabled`, global `subagents_disabled` and live unavailability differ. Guidance never ranks: exact selection starts or refuses, without substitution. `inspect_subagent`, Accounts, quotas and health share passive `claudexor_observations` (no start, refresh or selection); `quota_block` alone is not admission (PASSIVE_QUOTA_READ).

**Scheduling.** `schedule_subagent` keeps blank roles empty (§3); `workspace_root` defaults to the parent's, acceptance claims never inherit, `access=inherit` keeps configured rights and `readonly|workspace_write` narrow them, while `write_surface` owns mutation authority. `subagent_intent_fields` carries intent through admission, worker and snapshot; `choose_effort` resolves `effort` (`auto` = omitted); `select_subagent_snapshot` freezes the enabled API child or session leaf. `main_context_authority` keeps briefs idempotent; lineage uses `resolve_task_lineage.is_root_task`, not roles; shared/declared helpers keep briefs and review authority; blank-role/retry roots and exact result reads keep predecessor history.

**Cognition.** `supervisor` selects enabled API cognition (subscriptions included); session, disabled, credential-less or API-child rows refuse, and omission follows the parent's role, not a fallback serving it. `supervisor_route.py` freezes `parent_cognitive_route` schema 2 (model, effort, locality, source role, account or empty Auto, processing, `chosen_by`, a delegator's row snapshot) through queue, worker, results and restart; a following parent's nanny follows its role (`follows_role`), and an own route is `parent_switch`. `model_slots` assigns `subagent:<supervisor row>` or `subagent:<leaf row>`; legacy bindings stay frozen; persistence cannot convert session rows. Receipts/cards show recorded provenance and billing (subscription by route, measured only from observed cost, else unknown) with no pricing lookup or JS inference; leaf billing proves no nanny cost.

**Selected first-input sources** (`context_input_selection.py`). Both modes carry the assignment, owner rules/words (`owner_words.py`: tree rows by value, room read at start, scheduled rows for sessions), marks and knowledge outline. Native children/nannies get SYSTEM/BIBLE, book maps and applicable DESIGN; sessions get work orders and host references. Only `shared` adds claimed/linked note heads, native identity and room page (life account except nannies), never whole dialogue. Signed drafts need acceptance. Selection changes no access or model/effort and cannot guarantee independence; tools/mail broaden evidence.

**Selected inputs.** Files or diffs (DELEGATED_ADMISSION §3b) replace folderless lineage; omission/null/[] keeps it. Readers bind separately; threads keep scope. Sessions get selected `knowledge_notes` whole, even if declared; only the first start defaults to the scheduled selection. API children get addresses.

**Engine inputs** (`delegate_documents.py`, `claudexor_uploads.py`; DELEGATED_ADMISSION §3a): keys survive copies and Continue; owed managed releases feed `runtime_retirement` under tree/Continue/Keep holds, without local files; Keep shares custody with release, late Keep reports uncertainty; open uploads cancel before settlement.

**Historical author inputs.** `context_input_selection.capture_historical_inputs` pins first usable input on response and latest on evaluation as immutable redacted JSON; logical input, verified projection/gap, Presence event and admission stay distinct, never rebuilt live. Helper carriage coalesces exact rows only at the pinned first system fragment, never drafts or printed stamps. `review_evidence.historical_author_inputs` shows them to reviewers beside owner/terminal inputs, changing no criteria, ACK or paid identity; native/session readers get exact sources, packet previews disclose omissions. Sources reach canonical/execution stores before publication; anchor refs alone govern retention and survive recovery. Missing or changed evidence stays unavailable.

#### Child handoffs and task waits

**Child reads and waits.** `get_task_result` and settled `wait_task` return full handoffs, `wait_tasks` `_compact_child_projection` (§11.1); `cost_final=false` until terminal. Short answer/summary/evidence views reread current facts and reject authority/source selectors. Matching `known_result_sha256[_by_task]` gives `result_unchanged`. `get_task_result(call_id, manifest_sha256)` serves authorized character ranges of verified redacted calls, Presence-scoped, never raw payloads or drafts. Parked children show `current_stop` (`work_stop.task_current_stop`; `resume_child_task` by parent/root). Unsettled `wait_task` lists open non-review `delegated_runs`; unavailable facts prove no liveness. Sibling advice carries parent facts, no bodies; a read failure drops only the advice. Waits use `SETTLED_STATUSES` and `sleep_mode`; cancel intent stays `cancel_state: pending`; `wait_expired_with_live_children` reports IDs. `_wait_window` reserves `NESTED_SETTLEMENT_MARGIN_SEC` so the executor outlasts the inner wait.

**Exact selections** (`task_result_views.py`): `authority_fields` reads top-level fields or their index, `source_only` source ranges; partial recipes aren't full authority; `authority_projection_sha256` detects drift, not `authority_sha256`. `delegate_output.project_task_output_readers` offers publisher readers first and `direct_read` rootless, without receipt mutation.

#### Delegated spend and token reads

**Delegated cash.** `delegate_custody_usage.disclosed_cost` validates `attemptExecution[].usageCost` before summary cash; known subtotals and unpriced remainders stay apart from estimates; replay keeps history. `authPreference: subscription` prevents invisible paid-key fallback:

| disclosure | ledger | finality |
|---|---|---|
| settled zero | `subscription_session`, `0.0` | final |
| settled charge | amount | final |
| estimated | amount | nonfinal, estimated zero too |
| partial | known cash; `cost_partial`, separate `cost_estimated`; no bound | nonfinal; an exact subtotal stays confirmed, an estimated one estimated |
| undisclosed | null; `unknown_unmetered` | nonfinal |

Undisclosed cash adds zero only to known totals; `TOTAL_BUDGET` cannot fence it, and no bound or charge is invented.

**Native input.** `_usage_response._normalized_input_token_usage` accepts exactly total/read/write nonnegative integers or null. `delegate_collection` appends `llm_usage` after accounting; reviewers observe attach-only. Read facts survive continuation/cancel failures; replay dedupes, and event loss never warrants repeat inference. Filters keep `input_token_usage`; `run_id`/`usage_scope=run` mark sessions (model unknown; final model and account in `observed_attempt`); legacy or tokenless rows stay unknown. `effort_evidence.session_prompt_cache_ttl_request` sends captured `promptCacheTtl` tiers; Default and continuations omit it.

#### Nanny, transport and custody

**Nannies.** `agent_session` leaves are unlimited, bounded and nonrecursive, host-verified, without a team cap; tasks own lineage, authority, deadline, money, review and cancellation. Root `requested_depth` narrows by `depth_remaining`. Login: `gateway/claudexor_accounts.py`. Engine-scoped leaves never see owner sessions or courier to them.

**Transport.** `gateways/claudexor.py` owns descriptor, handshake and `CLAUDEXOR_MIN_VERSION`; `trust_env=False` excludes proxies. `ensure_owned_gateway` or the run-bound engine (`execution_domain`) starts reviewed engine/Node pins, never PATH. Review seats stay `in_flight` until allowance ends, then `not_dispatched`, never waived. `gateways/claudexor_observation.py` observes runs (docstring). Unconfirmed acceptance has no stop-time guarantee. `OUROBOROS_CLAUDEXOR_BIN` overrides the binary; `OUROBOROS_CLAUDEXOR_ATTACH_HOME` attaches without spawn or operator-home fallback. Stops: §9.

**Durable custody.** `delegate_custody_current.py` reduces `custody_runs` in `state/current.sqlite`, importing the forensic outbox once (docstring); snapshots (`task_snapshot`/`payload_snapshot`) see STARTED replace request. `health_runs` reads own, held and possibly-unread rows in one read; OWNED/FOREIGN/UNKNOWN never reads unreadable storage as absence. UUID/Idempotency-Key names intent, not content. `delegate_accepted.py` binds accepted work once by noncreating lookup (a miss sends nothing); `delegate_pending.py` admits replay; a request ID proves no receiving store, and missing sources stay unknown without POST. Containment incidents persist.

`delegate_invocation.py` owns first-request/last-start and operation identity: unknown custody never authorizes a replacement send (docstring; `test_review_invocation_current.py`).

`delegate_collection` keeps facts, output and money (retry rules: `delegate_capture_attempt.py`); `delegate_capture_job` captures in the maintenance child before `dialogue_import`; its docstring owns revalidation by the lock holder. `delegate_named_evidence` publishes immutable receipts; failed copies await its staging remover. `collect_terminal` owns publication and disposition; reviewers keep their owner/modes. Startup, Stop and Restart never join capture; retention grants no EOF, ACK or verdict; billing and execution identities survive; observation starts no daemon or POST and answers or applies nothing.

Finish cancels ordinary orphans (late discoveries survive); interruption requests bound-run cancel; subtree Stop follows parents (`stop_tree_runs`). Without a verdict, crashes, transport/provider loss, unknown ownership and cold waits spare runs; panels keep critics. One-shot roots register below the engine floor; unused ones (proven no-send included) follow §10.18. Lost replies keep custody; exact lookups bind, misses prove nothing; auto-replay needs admitting-store proof, and accepted risk bars both replays. Cleanup retains causes and ledger debt. `started_uncustodied` needs absent/terminal proof.

**Cancel receipts.** `delegate_cancel` returns `confirmed` (verified terminal/absence), `requested`, `failed` or `containment_fault_run_may_still_be_live`; failures keep a critical incident until closure. Only a run-detail 404 naming the admitting `context.storeId` (legacy unbound excepted) proves absence, inventing no terminal, usage or spend. Recovered-run Stop needs the exact run/store, not a status write. Proven unsent retries; ambiguous or acknowledged sends are read back; after ambiguity or proven sender exit, a newer owner Stop reserves once after a fresh exact live read.

#### Nanny verbs and continuation

**Nanny verbs** (`tools/delegate.py`): start, wait, cancel, answer, message. A letter to a linked Ouroboros (§12) rides the same custody: `tools/peer_delegate.py` `delegate_to_ouroboros` starts a run whose engine is `peer` (`execution_domain.bound_gateway`), read, answered and stopped with wait/answer/cancel; `message_ouroboros` sends words that assign no work. `tools/peer_letter.py` `ask_sender`, shown to a letter's root alone, parks a `peer_question` wait (`owner_wait.py`), never an owner turn. Start takes `subagent_id` or recovery-only `retry_of`; saved supervision restores before admission; unobserved availability proves no readiness or capability loss. Messaging needs cataloged `POST /v2/runs/:id/messages` and `liveInput != none`; delivery proves consumption, not obedience, and `capability_read_failed` stays unknown. `not_active`/`interaction_pending` carries a question, never an auto-answer. A definite refusal needs a typed cause, no handle and `definitely_unrun`/`_DEFINITE_UNRUN_REASONS` (UNRUN, $0); ambiguity wakes judgment. Zero-run excludes `review_owned`; accounting, containment and registration count all runs; nothing caps admission (DELEGATED_ADMISSION).

At receipt, `owner_relay` mails every live tree member and `delegate_interactions` places rows no START body held into non-review live-input runs as `owner_words` frames, independent of turns/capacity (§10.26); a Reply the root answers is context, no frame. Root drain and starter round-top catch up; pending-frame replay belongs to `delegate_feed.py`.

**Addressed talk.** `delegate_shared.message_run` admits starters, ancestors and same-tree tasks; foreign roots refuse and `_owned_run` control stays. `delegate_message_store.py` commits the host frame and wire bytes before POST; replay binds them. `delegate_progress` shows the journal in the starter's next receipt. `delegate_reply.py` carries `reply_to={starter_task_id, run_id, interaction_id OR plan_run_id, question_ids}`: only the starter submits, stale or wrong ones refuse, absence selects nothing.

**Continuation** (`delegate_continuation.py`): `continue_from` opens new work/cap/key; work may repeat. Its docstring owns lineage, terminal/no successor, no ambiguous apply and access/target floors. `delegate_plan.promotion_permitted` excepts only access, for the starter under the current recorded ceiling; readonly: DELEGATED_ADMISSION §2. `continuity` names carrier; `packet` re-briefs. Facts enter `prompt`, not retained `instructions`; restate access/binding/inputs. `retry_of` replays request/receipt without recapture. A proven first pinned `POST /v2/runs` with `409 continuation_superseded` and a same-store miss settles sent/rejected, not unsent or zero spend. Accepted custody wins; same-store proof binds job aliases to runs. Unproved stays unknown.

**Plan leaves** (`delegate_plan.py`) use `review_thread_continuity` and need engine plan support. Custody pins mode, access, thread and ceiling; engine detail supplies questions and readiness, not live interactions. `delegate_answer` adds a turn; only the starter implements (agent), within its current ceiling, with recorded `override_plan_readiness` for open or unverified questions. The host checks retained bytes. Authorization precedes send and replays unchanged; legacy ceilings grant nothing; Git/payload stay private, folders implement in place, folder copies need a new executor (DELEGATED_ADMISSION §2). Ouroboros synthesizes N plans; engine council stays unwired.

**Execution evidence.** `delegate_evidence.task_execution_evidence` projects custody (§11.1); `delegate_start_attempted` counts blocked/uncustodied attempts, so refusal is not ignored delegation. `applied_access_profiles` uses SETTLED rows; an absent disposition is unrecorded, not "reviewed clean".

**Work orders** (`subagent_work_order.py`): assignment → `prompt`, normalized contract → `instructions`. `session_host_references` adds owner rules, body DESIGN bible/map/SHA and BIBLE, enabled skills (whole always-in-context; unless declared), writable Git environment/service paths and serving payload contracts (even when declared). `START_REQUESTED` pins bytes/ranges. Selection receipts grant no governance or access. Blocked checks return command/evidence to the integrator (DELEGATED_ADMISSION §2). Tool-less seats need excerpts; incomplete coverage bars PASS/apply (`delegate_source_coverage.py`).

#### Supervision, activity and recovery

**Supervision.** The nanny's `delegate_wait` sleeps via `run_wait` (§5); it and `await_messages(runs=…)` refuse reviewer runs before sleep, naming the original collector, reader or gap. A selected terminal, owned question, child attention, mail, owner input, cancel, deadline or checkpoint wakes it; progress does not; deadline, ceiling, budget and cancel still bind. Delivery, fallback, ACK and pacing: `delegate_supervision.py` docstring.

**Delegated activity.** `delegate_activity.py` emits message/thinking/problem parts in `progress_meta.delegated_activity`, apart from the answer; observation (`narration=False`) proves neither authorship nor execution. `source_handles/delegated_activity/` keeps redacted ranges with size/digest and loss disclosure; unnumbered SSE errors keep partial events; a permanent gap ends reads, not execution (docstring). Task-file `?source=`: §4; UI: §3.

**Questions.** `pendingInteractions` wake `waiting_on_user`; answers echo custody keys, escalating beyond authority to the nearest live ancestor. Terminal `input_required` needs `continue_from` + `prompt=<answers>`.

**Recovery** (`delegate_recovery.py`, holders `delegate_recovery_holders.py`) keeps independent successor proofs and precedes bootstrap; owner Restart, Panic, signals, deadline and cancellation never resume. Checkpoint schema and downgrade limits: PERSISTENCE.md; merges: `delegate_supervision_state.py`; old wake replay: `delegate_supervision.unseen_wake_view`. `delegate_bootstrap_identity.py` persists leaf-first identity under the actor lock: restart observes; at `subagent_runtime.delegate_start_entry` explicit starts take fresh rows, omission keeps the assignment, retry keeps row and body, and an empty-prompt replacement needs settlement; evidence separates initial from later work. Zero-run, unreadable custody/billing, starter-only control and single-writer floors stay ([details](../DELEGATED_ADMISSION.md)).

#### Live room Subagents mode

`settings_integrity.live_subagents_mode()` reads the room at admission and each round (§7). Scheduled work (`schedule_writer_kind`), Presence promises included, reads Swarm as Auto and keeps Off; unscheduled autonomous work keeps its controls. Off hides catalogs and refuses new runs (`schedule_subagent`, `delegate_start`, `continue_from`, plan answers; `subagents_off_in_room`); running helpers: CLIENT_CONTRACT §9.

`loop_messages` appends mode facts; owner rows keep their Swarm label; retries/switches keep the admission mode; failed reads leave labels absent. Receipts keep starts, `refused_starts`, reviews, models, cost and unfinished work; only proven zero attempts/starts hides the line, and gaps stay unknown (§3).

#### Dispatch, route health and history

**Configured dispatch is exact.** `subagent_runtime.resolve_configured_actor_dispatch` checks live availability for fresh work; a bad choice returns a typed cause, alternatives, reset time and `host_fallback:false`, never ranking, first-healthy selection or session→API conversion. The parent LLM chooses `agent_session` topology, decomposition and supervision; the host sets truth, money and authorship floors (BIBLE P13/P5). Fallbacks need explicit LLM-selected starts; pinned selections with invisible delegate tools are blocked, never substituted.

**Route health.** `subagents.route_health` serves dispatchers, nannies and reviewers; Doctor covers default credentials; engine pools own admission (`credential_pool_exhausted` at $0, healing only from reported `resetsAt`); checks cover disabled rows, capabilities, access, delegation markers and proven quota exhaustion; catalogs offer Main without rows. History grants no admission: fallback success never certifies a failed route, and code/test failures never condemn providers.

**Measurements.** `subagent_metrics` folds a bounded live journal suffix off-lock from its saved inode/offset and commits totals and cursor by CAS; cold history defers with unknown coverage. Wakes never block appends: a wake the held store refused leaves `OWED` (a pass takes it as `TAKEN`) and awaits the next publisher or explicit repair.

`swarm_receipt.py`/`swarm_receipt_sources.py` publish finalization receipts (intervals, accounts, context pressure); `swarm_wait_facts.py` keeps selections. Overlap is not productivity, missing causal edges stay unknown (never summed lifetimes), live actors keep receipts provisional, and `finish_task.swarm_comparison` is the mind's; no thresholds or automatic choice.

#### Run shape and mutation authority

**Run shape.** `subagents.delegated_run_shape` serves starts, dispatch and review seats; optional `mode=ask|plan|agent` (`delegate_start`, configured-session `schedule_subagent`) lowers ask/plan to readonly; API children keep ordinary execution and refuse explicit mode.

| task authority | access | mode | isolation | `execution.delegated` |
|---|---|---|---|---|
| acting subagent (valid write surface); ordinary root with a validated external workspace or selected Project room; top-level task selecting an exact skill payload (`root="skill_payload"`); research or reviewer seat without write authority (investigation) | captured `full` or `workspace_write` | `agent` | `live` | `true` |
| explicit `readonly`, Observe research, a Presence ceiling without delegated write, a fail-closed subagent | `readonly` | `ask` | envelope | omitted; false on writable→readonly continuation |

Host-derived **mutation authority** (`tools/delegate_integration._mutation_authority` / `_payload_mutation_authority`) names the destination, never model-supplied:

| source | `target_root` derivation | `capture_mode` |
|---|---|---|
| `acting_constraint` | the child's `task_constraint.write_root`, equal to the active workspace root | `delegated_snapshot` or `in_place` |
| `external_workspace_root` | the root's validated external workspace or Project room | `delegated_snapshot` or `in_place` |
| `skill_payload` | the payload the fresh `skill_payload.write` binding resolved | `delegated_snapshot` |
| `readonly` | active root (nothing to write) | `none` |
| `investigation` | active root, provenance only | `evidence` |

A genesis acting child's project (`ouroboros/genesis_project.py`) is a durable, standalone directory under `OUROBOROS_SUBAGENT_PROJECTS_ROOT`: not a checkout, not in the private-copy records and never pruned by age. `subagent_worktrees.provision_payload_snapshot` gives Git-free payloads private Git snapshots; undisposed captures never cap starts. `integrate_payload_patch` locks target CAS, apply and publication. Drift conflicts, equality is idempotent, `blocked_reserved_paths` refuses all, baseline-equal `apply_no_op` proves no write, other hashes keep ambiguity. Success calls `request_extension_reconcile` and stales review; mode-only changes refuse `unreviewable_metadata_change`.

#### Snapshots, capture and disposition

**Snapshots.** Native/API `external_workspace` reverse-checks only its captured Git delta, never whole work or authorship, under current rights, writing no tree or index; mismatches await `accept`/reject, and denied-hash, unstageable or unexaminable paths stay unknown. `subagent_worktrees.snapshot_commit` links writer, investigation and task copies; ignored files stay out, and nested repos or unreadable folders keep writer checkouts. Integration needs current write/Presence rights; body protected paths obey dispatch's rule; refusals name mode or grant. Seats and body candidates link frozen trees, candidates on branches. A pinned result maps saved → release checkout, empty → `no_changes`, continuation → restore, gone unsaved → `execution_snapshot_missing`; unreadable custody retains. `workspace: in_place` uses the starter's root; `delegate_directory` stages folderless inputs; `scope.root` grants reads, `execution.workspaceRoot` binds writes (DELEGATED_ADMISSION).

Collection and disposition share `delegate_integration.capture_terminal_patch_for_drive`; execution end, not money, admits capture, and Git readiness needs a receipt. Older bytes are proven in scratch; saved bytes are never overwritten. Changes and a moved HEAD are shown before apply, and reject can decide. Native disposition lends task custody to evidence publication; GC retires copies; unwritten rows or unretained evidence hold them. Eligible empty Git captures close as `no_changes`. Git checks baselines and stages, never commits. Git-only `target_files_changed` survives receipt or refresh failure; proof allows a receipt-only retry (`_admit_git_apply`), else ambiguity (`INTEGRATE_DELEGATED_APPLY_AMBIGUOUS`) is acknowledged after inspection.

Payload and seat capture share `subagent_worktrees.stage_raw_payload_inventory`/`_RawBlobWriter` (filter-free `fast-import`, then index path/mode/blob checks).

**Terminal products.** Revisions keep metadata, reader and hash across output retries; previews earn no EOF. Investigations self-dispose (`evidence_retained`) only after full output and required sources survive, never as acceptance. `undisposed_patches` keeps run and target until `PATCH_DISPOSED` names `disposed_by_task_id`. Only `result_loss` proof (no copy, usable capture, pin or Git-held work; own pin restored first) lets reject mark `result_lost`, keeping what a failed capture saved; a predecessor's pin keeps an uncaptured continuation owed. Confirmed automatic retries may read/dispose terminal-observed products, not live control; shared roots, manual Continue, review-owned runs and `CLOSED_ABSENT` grant none. Takeover supersedes the predecessor and cumulative work applies once; orphans need proven owner terminality. Off the active folder, apply targets the recorded path inside configured projects/worktrees roots under `copy_apply_refusal`'s current rights, with no rebind or grant; own/retry targets stay exact; reject needs no write. Missing, unreadable or unreaped ownership keeps custody; late sources re-owe a closing receipt (`delegate_terminal`/`context_health`).

**Custody disclosures heal from the write side.** `delegated_runs_unreconciled` warns while unreconciled own runs remain (`outcomes.objective_warned`); readers never replay custody, `delegate_terminal.py` writers refresh it, and unknown clears nothing. Stored `delegated_custody_unreconciled` stamps are read beneath (`recorded_cause`), warning until the list is known empty; clean debt is no evolution failure.

Disclosed isolation residuals: a non-owner orphan's verdict stays with its disposer. Task-drive Git locks let nanny integrations interleave in one external tree, so drift makes the losing apply a typed conflict; copies share refs, stash and hooks; in-place writers skip the delegated protected-path gate (the commit gate still checks them) and can mix.

**Native product succession.** `integrate_subagent_patch` admits finished native direct-child products (cancelled: ready capture) via hash-verified `owner_continue.continuation_chain` with target/CAS/resource checks; exact-result decisions stand, observations retry. Successor verdicts keep the original parent, child and hash; lineage, evaluation, controls, cancellation and missing marks stay unchanged.

#### Access disclosure

**Native access disclosure.** Full access has no filesystem sandbox; snapshots own delivery (an `in_place` writer delivers as it edits). `CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION` is a schema floor: `route_health` blocks incompatible mutating requests before spend; read-only delegation omits the marker. Attempt evidence alone determines isolation. Missing containment discloses `delegate_run_unconfined` without refusal, because the child already has native shell access. A recorded `harness_home_isolated:false` or a claimed isolated HOME equal to the operator's cancels as a breach; nested HOME is disclosed unconfined, missing HOME evidence stays unproven (DELEGATED_ADMISSION §8).

**Authority.** Start accepts session `subagent_id`, prompt, duration, lowering-only access, mode, `workspace` copy/in_place, retry/continuation and payload/document selectors (DELEGATED_ADMISSION §3a), never an isolation or scope override. Retry replays recorded rights and host instructions. Native access grants no task or edit authority. `_containment_breach` checks engine-applied access and HOME: wider cancels (`access_profile_widened`), narrower passes.

### Vendor program maintenance

`harness_maintenance.py` serves Accounts/Connect, `ouroboros/gateways/claudexor_maintenance.py` enforces the engine-version floor, and `ouroboros/tools/control_maintenance.py` exposes mind tools. Authority and Pause bind with no auto-update, version ordering or job ledger. Connect joins active installs under one deadline, cancels nothing, returns definite refusals and retries login once after selected-version evidence. An unknown maintenance POST keeps its key, body and context for a same-key retry.

`harness_program_facts.py` writes `state/harness_programs.json` under file-only locks: claims spend each generation/settlement once (even on failure), newer rows win and the latest full answer sets membership. Context reads only this file (dated installed/newest facts, unavailable/incomplete coverage or "not observed"; no timer; lifetime: PERSISTENCE).
