# 1. High-Level Architecture

Modules, devtools, gateway/CLI, host and data map. Packaged launcher recovers failed server edits. Server: direct turns/supervision; crash-isolated queued workers.

```
User → launcher.py (desktop/native) — the shared lifecycle; an immutable packaged shell, or source over an immutable seed on Android and the managed container (Runtime topology below; §2)
launcher.py spawns server.py (Starlette+uvicorn) — HTTP + WebSocket on configurable host:port; bind host via `OUROBOROS_SERVER_HOST` (§4; §7)
web/ — Web UI, a SPA of ES modules; chapter 3 owns the pages and the modules not listed here (§3)
  ui.css — the shared palette/control stylesheet of the SPA, onboarding and author pages (§3 Navigation)
  modules/ — browser modules (§3)
    ui_primitives.js — safe-field, escaping and tone/status helpers (§3 Navigation)
    i18n.js, settings_language.js, ui_i18n_types.js — interface language: a DOM overlay over rendered chrome from the install's translation memory, translating at the producer so chat prose, logs, code and owner names are never walked (§3 Settings; §7 `OUROBOROS_UI_LANGUAGE`)
    page_header.js, ui_interactions.js, scroll_fade.js — header/tab strip and the segmented-choice generator; dialog, menu and popup binders; scroll-edge fade (§3 Navigation)
    chat_decision.js, question_presentation.js, main_attention.js, chat_render_batch.js, task_phase_chip.js, task_activity_types.js, lifecycle_card.js — chat helpers: decision cards, the one Project-question form and its Main mirror (Python twin `project_dialogue.QUESTION_STATUS`), keyed timeline patches, shared status/motion and census schema, skill lifecycle card (§3 Chat and Projects)
    chat_history.js, chat_history_replay.js, chat_reading_position.js, chat_row_placement.js — bounded pages, replay, reading intent, child words versus notices (§3 Timeline ownership)
    delegated_activity.js — executor activity with preview gaps (§3); verification_presentation.js — recorded failed-check rows, live and replayed (§6 Tool capability and execution)
    welcome_preference.js, host_notice_text.js — Main's empty state from the hidden `welcome` preference; known host sentences painted without the leading pictograph (DESIGN "Chat authorship and System rows")
    project_answer.js — Main's Project lifecycle rows and folded mirrored answers (§3 Main rows)
    project_handoff.js — compact Main creation/transfer entries; receipt folding preserves work identity (§3 Project handoff receipts)
    project_reference.js — the one Project pointer control and the only raiser of `ouro:open-project` (DESIGN "References and actions")
    project_activity.js — `active_chat_activities` census projection for Project dots (§3 Liveness census)
    room_work.js, project_read_state.js — a room's header work list without execution authority; its read receipt (§3 Project rooms)
    model_wait.js — model-wait views and owner actions inside chat cards (§6 Quota and auth waits)
    task_checkpoints.js, cancel_presentation.js — typed checkpoints and cancellation-cause text shared by Chat and Logs (§3)
    dashboard.js, logs.js, worker_sha_presentation.js, costs.js, files.js — Dashboard, Logs, worker-SHA presentation, Costs (unknown is not free), Files (§3)
    skills.js, marketplace.js, skill_review_card.js, skill_publish_flow.js — Skills page, ClawHub marketplace, Skill Review cards, publish dialog (§3 Skills and Widgets)
    settings_ui.js, settings_catalog.js, settings_controls.js, settings_policy.js, settings_secrets.js, settings_local_model.js, settings_autostart.js, mcp_settings.js, mcp_header_fields.js, mcp_import_dialog.js — Settings page leaves: catalog refresh, control binders, policy-state lines, secret Show/Hide, local-model form, host sign-in toggle, MCP cards and import (§3 Settings)
    model_roles.js, model_chooser.js — the Models editor and editable chooser shared by Settings, onboarding and route editors; catalog arrival never assigns a value (§3 Navigation)
    subagents_settings.js, subagent_status_primitives.js, route_editor_primitives.js, harness_accounts.js, account_resources.js, harness_login_cards.js, claudexor_status_store.js — Agents editors; account/resource/login views share one status/action store (§3 Agent accounts)
    onboarding_agents_step.js, onboarding_overlay.js, project_create.js, utils.js — first-run accounts step; the wizard frame's sandbox policy, kept in one place as a security boundary; New Project dialog; shared utilities (§2; §3 Project rooms)
    review_command.js — `/review` chooser and send (§3 Chat and Projects)
    review_presentation.js, plan_record_presentation.js, review_record_card.js, review_dom_patch.js, harness_presentation.js — read-side Review Checkpoint grouping, one review record's panel and reviewer lines (`formatReviewProjection`), keyed DOM patching, the sole owner of harness identity markup (§3 Child cards and executor presentation)
    acceptance_incident_presentation.js — local evidence-preparation warning; no reviewer authority (§6 Task acceptance)
    widgets.js, widget_module.js, widget_frame.js, widget_card.js, widget_reorder.js, widget_list.js, widget_size.js, widget_chart.js, widget_job.js, masonry.js — Widgets page: host and card registry, framed mounts, card chrome, list-request policy, reorder/size/chart helpers and masonry (§3 Widgets page)
supervisor/ — background thread inside server.py (§5)
  active_activity.py — process-local direct actors and current call observations behind `/api/state`; typing receipts remain separate, with no queue records (§3 Direct turns and the activity block)
  message_bus.py — local message bus for the Web UI and transport skills; re-exports the ingress names
  message_ingress.py — one accepted row per message id before dispatch, through a process-local chat-chain index
  workers.py — worker pool: forkserver on Linux, spawn elsewhere, never fork: a child forked from the multi-threaded supervisor inherits a held import lock and wedges
  worker_assignment.py, worker_chat_lane.py, worker_health.py, worker_pool_lifecycle.py, worker_process.py, worker_promotion.py — pool leaves: assignment, chat-lane admission, crash recovery, lifecycle and tree kill, the worker child, promotion (§5; §6 Owner routing verbs)
  worker_owner_wait.py — a required owner wait lends its active slot; the task stays RUNNING, custodied (§5)
  state.py, state_initialization.py — `state/state.json` updates with typed read quality; init witness
  queue.py — task queue (PENDING/RUNNING) with activity-based timeouts; the one task-state authority (§5)
  queue_schedules.py, schedule_lifecycle.py, queue_snapshot.py, queue_timeouts.py — leaves re-exported through `supervisor.queue`: schedules, durable snapshot and restart restore, activity liveness (§5)
  followup_policy.py — related/independent relationship admission (`followup_relation`) (§5)
  cognitive_operations.py — in-memory LLM/review/VLM operation leases for the idle rail; no durable ledger (§6)
  task_model_wait.py — live model-wait projection with quota-clock reads for liveness (§6 Quota and auth waits)
  task_admission.py — token-owned reservations fence duplicate ingress ids before Project, workspace and attachment effects (§5)
  task_lifecycle.py — the one settle owner of durable cancel intents; `sweep_cancel_intents`; root-budget admission fence (§5; §10 invariant 14)
  cancel_publication.py — cancellation settlement publication: typed `CANCEL_*` outcomes, artifact-honest cancelled fields (§5)
  budget_resume.py — Resume grants, revocation and hold release (§6)
  owner_pause_control.py — owner Pause fence, member wake, tree settlement (§6 Owner Pause)
  continuation_admission.py — owner Continue admission: replay first, predecessor claim (§6 Owner Continue)
  sleep_wake.py — cold model-sleep wake with a typed `sleep_wake` grant, vetoed by holds (§6)
  restart_retention.py — stop retention and restart/update-abort return authority (§5, §9)
  queue_transitions.py, task_ownership.py — acceptance, Resume, evolution stop, fenced Project deletion; ownership reads precede the queue lock (§5, §6)
  terminal_delivery.py, terminal_cascade.py — delivery-id dedupe, exact emitted-byte receipts and the bounded pending outbox for every terminal answer; a settled cascade's one owner message (§5, §10)
  task_reaper.py, task_reaper_cancel_handoff.py — one off-loop owner of timeout, crash, idle and unready jobs, and its cancel-suppressed retry handoff; unconfirmed death keeps reaping (`task_reaper_wedged`); mints no cancel intents (§5)
  owner_stop.py — graceful stop: finalize first (`finalize_now`), then cancel on the same durable cancel intent; grace bounded by `OWNER_STOP_OUTER_CAP_SEC`
  schedule_time.py, schedule_occurrence.py — cron/timezone parsing; a due occurrence admits at most one root
  schedule_notes.py — a due `kind:"notify"` row becomes a signed `reminder` System row; no model, no retry
  evolution_lifecycle.py — evolution campaign state and cycle lifecycle (§6 Background consciousness)
  events.py, event_observation.py — worker→supervisor dispatch and pending idle-input index; `main_llm_call_state` binds the active attempt/round/call; unregistered types become `unknown_worker_event`
  subagent_task_truth.py — delegation-truth enrichment of the subagent `task_done` frame (§11.1)
  event_taxonomy.py, events_budget.py, events_chat_delivery.py, events_coop_checkpoint.py, events_evolution_done.py, events_project_routing.py, events_runtime_controls.py, events_schedule_task.py, events_subagent_admission.py, events_task_done.py, events_worker_reports.py — `EVENT_HANDLERS` leaves, one event family each; `event_taxonomy.py` declares each `EVENT_Q` disposition; `events_project_routing.py` is the single durable writer of promote/route refusals (§5)
  task_dispatch.py, workspace_preparation.py — admitted worker payload; off-loop preparation with durable phases (§5)
  log_addressing.py — audience of task-scoped log events: project binding wins, chat id 0 is `HIDDEN_CHAT_ID`, A2A frames are dropped at the `push_log` choke (§4 WebSocket protocol; §12)
  steering.py — steering into a running task's mailbox, keyed on the host-minted `issuer`; refused while a cancel intent is pending (§6 Owner routing verbs)
  plan_obligation.py — a promoting root hands an unmet `force_plan` to the new root inside the admission transaction (§6 Owner routing verbs)
  direct_roots.py — off-lock `state/direct_roots.json` roster of live direct-chat roots for workers
  telemetry_events.py — rare typed telemetry-only worker events: durable appends exclude high-rate narration
  git_ops.py — git operations (clone, checkout, rescue, rollback, push, credential helper) and the bounded local-Git runner
  git_ops_remotes.py, git_ops_rescue.py, git_ops_reset.py, git_ops_updates.py — leaves re-exported through `git_ops.py`: personal `origin` and push, the rescue snapshot before every destructive tree movement (§2), checkout/reset admission, managed-update status and tags
  update_source.py — official update-source selection and network policy
  update_recovery.py — exact owner Restore: pinned prior HEAD, rescue before reset
  update_merge.py — managed-update engine: exact-target 3-way plan, reviewed assisted merge, write-ahead transaction, verified rollback, boot recovery (DEVELOPMENT "Managed Update Rule")
  update_candidate.py — candidate/carrier primitives: rerere-neutral merges, failed-update preservation branches, write-ahead stash restore (§6 Hermetic preflight proof)
  update_carriers.py, update_merge_plan.py — carriers a merge may not resolve by hand; merge planning
  update_merge_policy.py — presentation-only conflict labels
ouroboros/ — agent core and shared runtime (§6)
  config.py — SSOT facade: paths, settings load/save, PID lock (§7)
  settings_defaults.py, settings_scales.py, model_slots.py, review_model_routes.py, runtime_limits.py — the settings vocabularies `config.py` re-exports: keys and defaults, closed scales, model-slot resolution, reviewer routes, clamped limits (§7; §10 invariant 3)
  version.py — version string from VERSION, importlib.metadata fallback
  secret_masking.py — Settings/MCP wire placeholders and secret repair before env overlay and persistence (§7)
  settings_integrity.py — task-local and skill settings views, strict snapshot pin (`OUROBOROS_SETTINGS_SHA256`), echo and digests of projections (`OUROBOROS_PROJECTED_*`) (§7)
  credential_shapes.py — credential leaf names and physical locations (§6 Credential mutation and diagnostic redaction)
  update_channels.py — Stable/QA/Development channel mapping (§8)
  update_letter.py — the update letter: `base..target` commit material from one accounted light-slot call, shared by the Updates payload and the Runtime-context `official_update` fact (§7; §3 Updates)
  colab_bootstrap.py — Google Colab source-mode bootstrap (official source, Drive-backed data, no-UI server)
  cli.py — source/headless CLI over gateway tasks, logs, settings, skills, local model and MCP (CLI / Headless Boundary below)
  packaged_cli.py, packaged_cli_install.py — packaged desktop CLI bridge and its command-shim installer
  agent.py — task orchestrator; a loop crash is projected from captured evidence or an explicit `loop_evidence_unavailable`, never invented counters (§6 Task lifecycle)
  startup_historical_audit.py — explicit seal-audit child (`python -m ouroboros.startup_historical_audit`, own process group); never automatic at boot (§2; docs/MODEL_SEND_OBSERVABILITY.md)
  focus.py — authored focus with typed source ref; no dialogue or path escapes
  agent_startup_checks.py — agent-boot verification (warning-only health) and native-host adoption for self-restart (§2)
  agent_task_pipeline.py — task execution pipeline: result, artifacts, frozen cost snapshot, the review lens, root-only post-task work (§6 Task lifecycle)
  agent_dispatch.py, post_task_synthesis.py — delegated-child dispatch seam; post-task workers (§6 Post-task work)
  task_finalization.py — early final-answer delivery under one `delivery_id`; the `swarm_efficiency` rollup reports `lanes_requested` and embeds the `depth` block from `depth_evidence.py` (§6 Post-task work; Budget tracking)
  mutation_attribution.py — root-task baseline capture, predecessor adoption, committed interval delta (§6 Git and commit review)
  process_interpreters.py — Python and Node resolvers for the process launch surfaces (§2; CLI / Headless Boundary below)
  post_task_checkpoint.py — root phase/cost and saved late-work Pause (§6)
  terminal_projection.py, terminal_time.py — cognition-free settled-result projection and durable Main delivery; per-attempt end time (§3 Main rows; §6 Post-task work)
  presence_profile.py, presence_runtime.py, presence_capabilities.py, presence_authority.py, presence_bindings.py, presence_admission.py, presence_context.py — Presence admission: reviewed `presence:` profile parser, symbolic defaults, host-owned capability selections, the immutable positive capability ceiling, revocable room → skill bindings, per-turn snapshot and context (§12)
  presence_runner.py, presence_continuation.py, presence_observations.py, presence_delivery.py, presence_recovery.py — Presence: capped, conversation-serialized fresh agents; same-author review continuation, transport observations, delivery receipts and retained room facts (§12)
  dialogue_provenance.py — exact transport-provenance rendering for history, memory and consolidation
  extension_companion.py, extension_reconcile_queue.py — host-supervised companion processes for transport skills; durable worker→server reconcile markers (§12; §13)
  event_bus.py — typed in-process event bus for skill subscriptions
  event_usage.py, event_token_totals.py — event token counters, offline totals (§11.4)
  evolution_checkpoints.py, evolution_fingerprint.py — append-only campaign checkpoint ledger; canonical objective fingerprint for repeat gating
  improvement_backlog.py — the authored backlog note's complete current text for the evolution chooser and context; no host writer (§6 Post-task work)
  loop.py — the LLM tool loop, finalization nudges and the FINAL ANSWER candidate (§6 Task lifecycle)
  acceptance_settlement.py, late_review_marks.py — quorum/final mail wake, durable post-terminal `late_settlement` and room mark (§6 Task acceptance)
  acceptance_history.py, owner_source.py, acceptance_late.py — frozen answer debt, cap and owner authority; historical review settlement (§6 Task acceptance)
  loop_acceptance.py, loop_acceptance_review.py, acceptance_preparation.py, acceptance_retrieving.py, acceptance_material.py — acceptance machinery re-exported from `loop`: fence and obligations with the sole decision writer, host packet/panel, retrieving-source fit, pre-binding incident identity, produced material (§6 Task acceptance)
  loop_llm_call.py — single-round LLM call, current-call event producer and usage accounting
  transcript_prefix.py — append-only transcript between the sends of one loop execution; a break is a recorded `prompt_prefix_break` fact, never a blocked send (§6 Task lifecycle)
  loop_transport.py — transport-outage wait episodes and provider-failure terminal text (§6 Context fitting)
  loop_delivery.py — delivery candidates and the delivery-control protocol (§6 Task lifecycle)
  loop_budget.py, loop_forced_finalization.py, loop_messages.py, loop_model_call.py, primary_route_observation.py, loop_nudges.py, loop_round_limits.py — leaves of `loop.py`, one rail each: budget, forced finalization (the one forced model call), owner-message plumbing, the per-round model call with context fit and fallback chain, a fallback's non-generating primary facts, nudges, round limits
  task_pacing.py — threshold facts, shared round-money reading, finalization reserve, `CostCeiling` and main-loop wire options (§6 Budget tracking)
  vision_routing.py, vision_image_limits.py, image_preparation.py — owner-mode image routing for Main, explicit VLM/caption sends, known route limits and shared byte preparation (§6 Vision and local image evidence)
  fallback_cooldown.py — per-process 429-aware cooldown for the `OUROBOROS_MODEL_FALLBACKS` chain; advisory
  model_concurrency.py — per-(model, use_local) semaphore (`OUROBOROS_MODEL_MAX_CONCURRENCY`), per-process only: one task's loop, children and pings cannot exhaust a model's rate limit
  project_naming.py — SSOT for LLM-first project naming with deterministic fallback: admission (no model call), card conversion, the lazy turn namer
  ui_translation.py — the translation generator: one worker thread per process fills the memory through accounted light-model batches; a free-text language resolves to a tag or `language_needs_model`
  ui_language.py, i18n_memory.py — the interface-language fact (an open BCP-47 tag; `""` is not chosen) and the per-language translation memory under `state/i18n/<tag>.json`, provenance owner > imported > generated (§7 `OUROBOROS_UI_LANGUAGE`)
  loop_tool_execution.py, tool_call_log.py — tool dispatch, results, per-invocation counting
  deadline_utils.py — deadline parsing and the transport-vs-logical wait seam
  observability.py, source_retention.py — private CAS: equal histories stored once, gzip-1, hardlinks (§10)
  evidence_packs.py, evidence_packer.py — task-local xz packs, refreshed census/pass; reads verify loose → index → pack; missing ≠ `evidence_corrupt` (PERSISTENCE.md)
  finalization_timing.py — final-event phase fields and `task_finalization_timing` (§6)
  process_logging.py — per-process logging bootstrap; the server is the sole `server.log` writer; the launcher copies the server pipe, byte-capped, into `agent_stdout.log`
  model_send_seal.py — the invariant `model-visible ⟺ logged` for `model_send`: a mismatch is a typed durable fact, not a blocked call (docs/MODEL_SEND_OBSERVABILITY.md)
  cancel_intents.py — durable cancel-intent projection with claim-generation fencing; the one ingress `request_cancel`; fail-closed reads (§5; §10 invariants 14–15)
  owner_hurry.py — owner "hurry": a typed task-local latch, never a chat message, written on its own keys, as `write_task_result`'s status-regression guard could drop concurrent terminal fields (§5)
  owner_quiz.py — canonical questions: answer/close first-winner, publication address, retained delivery debt, terminal reconcile of paired `owner_wait` (§3, §11.1)
  owner_wait.py — cognition serialization, native waits, acknowledged restart handoffs (§5, §6)
  model_sleep.py, sleep_custody.py — warm/cold sleep and body acceptance (§6 The model's own sleep)
  budget_pause.py, tool_custody.py — exact pause/Resume, invocation futures and durable residuals (§6)
  owner_pause.py — whole-tree Pause fence, launch gate, per-attempt review episode, retained-member selection (§6 Owner Pause)
  review_pause.py — review custody under author Pause or addressed mail (§6 Owner Pause)
  external_runs.py — a task's delegated runs under one stop policy (§6 Owner Pause)
  working_checkpoint.py — one rolling working state per attempt, delayed content ACK, frozen recovery (§6 Saved working state)
  local_custody_repair.py — an addressed action retires a positively ended local owner's claim; passive reads write nothing (§6 Owner Continue)
  owner_continue.py — owner Continue data half: identity, eligibility, exact owner sources (§6 Owner Continue)
  routing_wait.py — receipt waits and the one act projection that the picker, routing tools and observer share
  outcomes.py — typed task-outcome and acceptance-decision authority with separate lifecycle, execution, objective, review, artifact and verification axes; a policy denial never masquerades as a tool failure (§6 Task lifecycle; §10.1)
  outcome_receipt_store.py — verification-receipt append/read authority; a zero-run receipt writes `incomplete`/`unknown` only: a zero-run "complete" is unverifiable self-report
  depth_evidence.py — pure `requested_depth`/permitted/attempted/achieved projection; a missing admitted permission stays unknown, never reconstructed from live config
  _outcome_receipts.py, _outcome_tool_errors.py — receipt parsing and the one canonical receipt identity (`receipt_canonical_identity`; §10 invariant 16); tool-trace status vocabularies, re-exported by `outcomes.py`
  code_intelligence.py, code_intelligence_architecture.py — internal code inventory (file facts, outlines, bounded imports and calls) in a derived cache without source bodies; architecture facts over the pinned domain/contract carriers (`owner_of`) (Code navigation below)
  code_navigation.py, code_occurrences.py, code_import_candidates.py, code_search_rg.py — `query_code` views, request-local token/import evidence, import candidates with explicit ambiguity, optional ripgrep for `search_code` behind the protected/secret gates (Code navigation below)
  pricing.py — exact-route provider-catalog lookup with nullable estimates; no static tariffs (they go stale) (§6 Budget tracking)
  usage_accounting.py, usage_admission.py, _usage_wait.py — physical attempts (reserved → dispatched → settled|unresolved) under global/root/group and review-wave admission; group binding; pre-send lock slices (§6 Budget tracking)
  _usage_rows.py, _usage_money.py, _usage_response.py, usage_ledger.py, skill_review_usage.py — the one reducer, exact Decimal cash, the accounting usage normalizer, shared row rules and the money lock, the read-only Skill Review projection (§6 Usage ledger substrate vs. accounting policy)
  _usage_cache_splits.py — process-local cache split by task/provider/route/review; a missing entry prices a full cache write (§6)
  usage_store.py, usage_journal.py, openrouter_cost.py — `state/usage.sqlite`, its journal import and explicit OpenRouter price receipts (docs/USAGE_STORE.md)
  store.py, store_import.py, store_switch.py — `state/current.sqlite`: money's SQLite core, atomic `meta.commit_seq`, typed `StoreMigrating`/`StoreRetired`/`StoreLost`/`StoreBusy`. Boot import/reconcile (§2), export before storeless switches (§9); packer snapshots precede deletion locks, changed facts defer.
  cost_projection.py — the one task-cost projection for every producer: `accounted_upper_bound_usd`, null as None and never $0.00 (§6 Budget tracking)
  storage.py — Dashboard Storage: one bounded, link-safe metadata walk at a time, kept in memory (§3 Dashboard)
  delegate_custody.py, delegate_custody_reconcile.py, delegate_collection.py, delegate_state_sweep.py, delegate_custody_usage.py — run custody, observation, collection and usage; OWNED/FOREIGN/UNKNOWN (§6 Nanny, transport and custody)
  delegate_hold.py — unknown-provider nanny hold (§6 Context fitting)
  delegate_source_coverage.py — oversized work-order source custody; incomplete source cannot authorize a passing terminal verdict or an apply
  delegate_evidence.py — read-side execution evidence over custody rows; an unreadable log is `evidence_read_failed`, never clean (§6 Delegated subagents; §11.1)
  synthesis_cost_text.py — task_summary cost fields
  llm.py — multi-provider LLM routing (OpenRouter, OpenAI, compatible, Cloud.ru, MiniMax, DeepSeek, Z.ai, GigaChat, Anthropic); conversations stay function-shaped, exact-route adaptation lives in the request-wire leaves (§6 Context fitting; §7 Direct-provider routes)
  llm_routing.py, llm_attempt.py, llm_messages.py, llm_capability_policy.py, llm_fallback.py, llm_pricing.py, llm_openai_compatible.py, llm_anthropic.py, llm_gigachat.py, llm_local.py, llm_claudexor.py, llm_substitution.py — leaves of `llm.py`: routing and affinity, physical-attempt candidates and cache policy, wire transcript shaping, capability/effort policy, the recovery ladder, live price catalogs, one wire lane each (OpenAI-compatible, Anthropic, GigaChat, local llama.cpp, Claudexor), account preference and served-model redo (§6 Caller-owned subscription model calls)
  llm_stream.py — SSE assembly inside one physical attempt (§6 Streams and transport waits)
  send_clock.py — Main preparation clocks and physical-candidate binding
  net_transport.py — shared httpx transport: TCP keepalive and the owner's extra-CA bundle every first-party client trusts (§7 `OUROBOROS_EXTRA_CA_BUNDLE`)
  model_wait.py — quota/auth waits bound to the task or phase owner (§6 Quota and auth waits)
  transport_custody.py — typed transport facts for the physical-attempt custody seam
  openrouter_attribution.py — canonical OpenRouter application attribution, centralized so forks do not compete under one identity (§7)
  openai_chat_custom.py, openai_chat_dispatch.py — direct-OpenAI Chat function→custom codec and its dispatch policy: custom first, exact-dialect fallback, task-local `none` last
  request_wire_contract.py, request_wire_resolution.py, request_wire_receipts.py, request_wire_attempt.py, request_wire_custom_validation.py, request_wire_recovery.py, effort_evidence.py — Exact-route wire compatibility: success evidence, profiles, receipts, call validation, recovery and reported effort (§6 Context fitting, retry, and compaction; §10 invariant 13)
  anthropic_native_custody.py — whole-block replay custody for Anthropic native reasoning on the same route
  reasoning_artifacts.py — sealed-vs-portable reasoning-artifact classification, fail-closed; the `SIGNED_PORTABLE` roster is a decaying external fact, extended only by a fresh cross-provider replay probe (inventory: docs/DEVELOPMENT.md)
  llm_observability.py — public call projections persisted; private sidecars stripped from durable records
  llm_probe.py — oversized-context probe and Provider Test transport; no retry, fallback or learning
  merge_receipts.py — task-owned PR merge intent, GitHub readback, durable receipt (§6)
  upgrade_notices.py, notice_receipts.py — once-only owner notices with chat receipts (§7)
  mcp_client.py, mcp_task_sessions.py, mcp_chrome_protocol.py — configured MCP; task-attempt/process ownership; Chrome wire, ready-skill and lease admission (§6; docs/browser-task-bridge.md)
  mcp_headers.py, mcp_import.py — header validation/masks/restoration; draft-only import preview (§6 MCP)
  safety.py — Safety Supervisor call (typed outcomes: §6 Safety Supervisor outcomes)
  consciousness.py, consciousness_wake.py — the background alarm admitting an ordinary Main turn through the supervisor tick (`handle_wake_direct`); the complete wake user input and `wake_task_metadata` envelope (§6 Background consciousness)
  consciousness_authority.py, consciousness_allowance.py — Dispatch-bound autonomy ceilings (keeping the owner-turn prompt prefix) and the rolling 24-hour spend allowance (§6 Background consciousness and Evolution)
  chat_chain.py — the chat generation chain (archives, then live), index-free row addresses, `retain_memory_source` (§6)
  chronicle_store.py — append-only chronicle: pages/parts, version-bound accounts and separate common-view selections; disposable index (§6 Durable memory)
  chronicle_import.py, memory_nomination_receipts.py, memory_inventory.py — one model-free import of prior dialogue memory as `legacy` records; frozen-cursor nominations imported once as marks; what is open or folded (§6 Durable memory)
  consolidator.py — shared Light transport and scratchpad compression; writes no knowledge (§6)
  memory_fallback.py — one signed Light helper draft per root task while consciousness is off; a refusal on the same input and Light route is a receipt, never a repeat (§6)
  memory.py — scratchpad, identity, chat history
  knowledge.py — linked-Markdown notes with shelf indexes and revision-checked writes, so concurrent cognition cannot silently overwrite a newer note (§6 Durable memory)
  memory_journal_compaction.py — compatibility entry point that only measures journal sizes (`server_maintenance.py` emits them as `memory_journal_observation`); history stays complete
  project_facts.py — project_id resolution and per-project knowledge under `projects/<id>/knowledge`, isolated from `memory/knowledge`
  task_tree_ledger.py — append-only `data/task_trees/<root>/blackboard.jsonl`: ephemeral swarm coordination (`tree_note`/`tree_read`), mirrored into the project journal at root completion
  projects_registry.py, project_admission.py — the owner Project registry (active|deleting|tombstoned; delete keeps bindings, history, folder and memory) and strict admission claims (§6 Project registry and lease)
  project_handoff.py — the Main transfer receipt owed through the terminal outbox after a durable bind; a typed `RECEIPT_STATES` answer, never a boolean (§3 Project handoff receipts)
  project_dialogue.py — read-only chat lens and append-only `logs/chat_annotations.jsonl`; routes nothing but the `needs_manual_target` decision card; `routing_refusal_cause` (§3 Chat and Projects)
  owner_words.py — the owner's words that caused a work tree, rendered verbatim for children, sessions and reviewers
  project_lease.py — one-writer-per-project lease in `assign_tasks`; same-project swarms exempt (§6 Project registry and lease)
  context.py — Main context assembly and the Available-subagents catalog (§6)
  context_input_selection.py — declared-source composition and first/latest usable author-input exhibits, distinct from current criteria (§6 Selected first-input sources)
  main_context_authority.py — Main's authority view and helpers' idempotent predecessor briefs (§6)
  client_surface.py — bounded client-surface normalizer; identity excludes viewport/narrow_layout (§4 WebSocket protocol)
  context_fit.py — deterministic Max/Low/Nano projections from one immutable core with typed reclaim deficit; owns the message-side transcript cache seal; no routing or retry authority (§6 Context fitting)
  context_budget.py — budget vocabulary and typed reclaim SSOT; `estimate_message_chars` (§6 Context fitting)
  context_mode_compat.py — normalizes and persists the `OUROBOROS_CONTEXT_MODE`/`OUROBOROS_CONTEXT_MODE_AUTO_LOW` pair during `load_settings()` (§7 Default settings)
  memory_view.py, memory_view_legacy.py, memory_view_account.py — role-based story/room views, selected accounts, source changes and retained addresses (§6 Durable memory)
  memory_floor.py — the memory view's physical floor: what a window cannot hold becomes address lines, people's words last (§6 Context fitting)
  capability_evidence.py, response_limits.py — sourced capability, token-density and maximum-response evidence (`data/state/capability_evidence.json`); windows size sends and grant no review authority; also the `image_input` namespace (§6 Prompt size, density and windows)
  context_layout.py — doc-layout SSOT: `book_navigation` is a book's compact view; ARCHITECTURE is composed in Max and navigated in Low/Nano; reduction relocates behind a visible pointer, never truncates silently (§6 Context fitting)
  reference_books.py — the ordered Architecture/Development reader and validator; free worktree book balance against the cached official development merge-base (an unavailable base is unknown, never paid; no fetch or local block; DEVELOPMENT "Documentation contract")
  local_model_server.py — read-only local formatter measurement and serving-process probe
  context_compaction.py — atomic-unit compaction with provenance capsules and transactional apply (§6 Context fitting)
  context_health.py — health invariants; delegated-run obligations stay globally visible: a preserved-and-invisible result is how work rots on disk (§6 Context fitting)
  context_runtime_facts.py — the runtime section's fact builders
  headless.py, history_retention.py — child-drive isolation, answer/file adoption, workspace patches, memory export; typed `sensitive_blocked` exclusions (§6 Headless finalization; §10)
  task_custody.py — the one child-drive deletion owner (`settle_child_drive`) and per-task custody lock (§6 Headless finalization)
  headless_status.py — lifecycle vocabulary shared by the headless owners
  workspace_patch_rules.py, workspace_patch_capture.py — patch-exclusion rules (env/cache, junk, lockfiles, credential-shaped names); the patch artifact and its manifest
  workspace_copies.py — git-copy source/baseline identity and own-body policy (§6 Delegated subagents)
  coop_checkpoint.py — quiescent checkpoint commits of cooperative trees from a mutative child's `write_root`; a root mid merge/rebase/cherry-pick/revert is skipped: staging it would commit a half-resolved tree (§5)
  delegate_output.py, delegate_directory.py, workspace_file_outputs.py — retained terminal JSON with physical-task reads; artifact-owned directory apply/discard and ordinary-folder results with exact before/after identities (§6 Terminal products)
  delegate_activity.py — typed executor activity JSONL with an emission-committed cursor (§6 Delegated activity)
  delegate_containment.py — engine-derived isolation facts (§6 Access disclosure)
  delegate_progress.py — poll-bound transport retry and event-local executor observations (§6 Delegated subagents)
  nanny_pacing.py — round/cost anchors; no reset on coordination (§6 Supervision)
  delegate_interactions.py — nanny input into a live session: typed `waiting_on_user`, validated `_delegate_answer`, capability-gated `_delegate_message` (§6 Delegated subagents)
  delegate_shared.py — `_owned_run` keeps live control with the starter; confirmed successors can read and dispose of settled products (§6 Delegated subagents)
  route_spec.py — neutral route primitive: kind/target/pin normalization and effort validation
  configured_subagents.py, subagent_runtime.py — canonical `OUROBOROS_SUBAGENTS` parser/serializer (owner free text is never host-parsed); immutable task-start snapshots and exact `subagent_id` selection (§6 Delegated subagents)
  subagent_route_health.py — the one manifest reader behind every delegated dispatch (§6 Route health)
  subagent_work_order.py, subagent_bootstrap.py, delegate_start_instructions.py — Complete owner-word-preserving work orders, host pre-start and stable instructions with a separately hashed coordination appendix (§6 Delegated subagents)
  delegate_supervision.py — event-only sleeping-nanny loop (§6 Supervision, activity and recovery)
  delegate_target_drift.py — read-only authority-tree drift evidence, never attributed to the child (§6 Delegated subagents)
  delegate_recovery.py, delegate_continuation.py, delegate_pending.py — exact-leaf recovery; `continue_from` after any stop; pending-invocation replay with the original key (§6 Delegated subagents)
  delegate_registration_policy.py, delegate_readonly_inputs.py — `resolve_registration` and STARTED-row field tables; readonly lineage inputs
  delegate_terminal.py — terminal reconciliation and custody-audit persistence, audit-only in both directions; the typed `terminal_custody_notice` card row (§6 Delegated subagents)
  subagent_dispatch_notes.py — executor-note exports shared with `agent.py`; the note/blocked-outcome pair is implemented in `agent_dispatch.py`
  subagent_messages.py, subagents.py, subagent_history.py — durable child-message identity shared by frame, recovery and replay; subagent envelopes dispatching through `subagent_runtime`; the compact helper receipt for context and owner UI, never admission (§6 Delegated subagents; Route health)
  subagent_worktrees.py — `state/subagent_worktrees.json` registry, `refs/ouroboros/delegated/` pins, custody-checked GC (§6 Delegated subagents)
  body_candidate.py, body_adoption.py, body_switch.py — Own-body worktree authoring and restart-bound adoption of an exact reviewed commit before body imports (§6 Own-body candidates; §2)
  artifacts.py — attachment staging into `artifact_store/attachments/`, artifact records, scratch fingerprints (`.scratch_manifest.json`), the undeclared-output guard, partial tool-evidence reads
  chat_uploads.py, confined_files.py — the owner-attachment store (`data/uploads`) with byte-proven media kind; directory-confined regular-file open (DESIGN "Chat attachments")
  retention.py — GC retention SSOT: clamp and age cutoff
  workspace_preflight.py — read-only external-workspace snapshot for gateway task creation
  project_sources.py — folder attach validation (realpath, not the home root, no repo/data overlap); opt-in `init_git`; a server-side clone never prompts and reports a typed `auth_required`; attaching is the trust grant (`trusted_at`)
  promotion_source.py — promoted-task source admission after an executor/id reservation
  workspace_admission.py — Task/promotion workspace admission: disjoint roots and Project binding, never a system-repo fallback (§6 Owner routing verbs; CLI / Headless Boundary below)
  local_model.py, local_model_autostart.py — llama-cpp lifecycle over proxy-free loopback and its startup helper (§3 Settings)
  deep_self_review.py — Whole-system diagnostic review on a named enabled row, defaulting to the direct Main row (§6 Deep self-review)
  review.py — Size and complexity inventory behind the shrink-only ceilings and the diff pulse (§6 Structural gates, Commit preflight)
  size_ratchet_manifest.py — Generated data-only size-debt manifest (`scripts/regenerate_size_ratchet.py`)
  review_execution_projection.py — Reviewer-execution projection; kinds `api` | `harness` | `native`, so the owner can tell a retrieving review from a packet review on the same model
  preflight_runner.py, preflight_node.py, preflight_receipts.py — Hermetic test gate, Node lane, receipts (§6 Hermetic preflight proof; §8)
  review_substrate.py — Review seat coordinator: independent seats, per-actor records that keep transport, parse, verdict, coverage and quorum distinct (§6 Review stack, Task acceptance)
  review_custody.py, review_source_closure.py, review_operation.py, review_live_join.py — Reviewer workers: no-resend custody, retained owner-bound inputs, panel waits/checkpoints, existing-producer collection/live joins (§6 Task acceptance)
  review_owner_custody.py — Paid attempts record `(server session, pid)` (§6 Paid stamp and owner custody)
  review_execution.py, review_session_preparation.py — Review delivery (`delivery_retrieves(route, native_retrieval)`) and session preparation without transport fallback (§6 Review delivery)
  review_native_episode.py — Read-only native API reviewer episodes for pool seats and deep self-review (§6 Native tool-round episode)
  review_session_reads.py — Harness-journal reading diagnostics (`harness_observed`), never verdict, quorum or retry authority (§6 Review delivery)
  review_verdict_extraction.py — Verdict canonicalization by output shape: strict parse, then light-model extraction; `array` keeps the findings ladder, `object` the whole verdict, `report` passes through verbatim
  review_session_custody.py, review_checking.py — Delegated-review recovery validation and the pre-POST durable invocation checkpoint; writable seats' private checking copies (§6 Review delivery)
  review_slot_cancel.py — A cancel outcome reports only what it proved; a succeeded run whose result read fails is the typed `ReviewSessionSucceededResultUnavailable`, never "may still be live"
  review_actor_aggregation.py, review_session_usage.py, review_thread_continuity.py — Completed-actor contract aggregation; delegated-session usage attribution; Claudexor thread operations for delegated plan reviewers
  commit_admission.py — Deterministic commit-admission SSOT (release checks, staged-Python syntax, `run_tests_preflight_with_proof`); the commit gate delegates here (§6 Hermetic preflight proof)
  reviewer_slot_config.py — The one review-pool builder, `review_pool_slots`, over enabled `review_eligible` catalog rows (§7 Review pool)
  review_pool_migration.py, review_pool_receipts.py — The review-lane → review-pool migration at the settings read seam and its durable receipts (the `state/review_migrations/` snapshot is the rollback source) (§11.4)
  review_run_isolation.py — Contributor review isolation before `config`: private whole-data root, pinned host settings and panel, cumulative cap, attach-only engine (§6 Delegated subagents; Monetary authority and projections)
  review_state.py — Durable commit-review state (`state/advisory_review.json`; old advisory rows are read-only history); re-exports review_state_model.py, review_state_records.py, review_state_custody.py
  review_records.py, review_verdict.py, review_projection.py, review_evidence_sections.py, repo_diff_capture.py — Panel records, verdict reducers, redacted projections, acceptance evidence and the single tree capture shared by preview and exact source (§6 Task acceptance; Review ledger record)
  review_ledger.py — Durable per-wave review records (`state/review_ledger/<record_id>.json`; §6 Review ledger record)
  review_body_fact.py — Own-body identity and the checklist layer it selects (§6 Change review on any root)
  review_cycles.py — Shared paid-cycle cap SSOT (`OUROBOROS_REVIEW_MAX_CYCLES`); the three per-gate meanings: §6 Review stack (§10 invariant 17)
  review_dispatch.py — Review row identities and write-ahead paid stamps (§6 Paid stamp and owner custody)
  reviewer_window.py — Typed per-route window resolution and reserves, used only for sizing; an unknown route keeps a disclosed assumption, never a review-authority floor (§6 Prompt size, density and windows)
  triad_review.py — Shared review primitives: JSON-array extraction, per-actor records, quorum/degraded accounting; `REVIEW_JSON_ARRAY_CONTRACT` (clean is exactly `[]`: structure cannot distinguish refusal from preamble); `review_output_shape(surface)` is the one form fact (`array`/`object`/`report`/`two_part`/`plan_reply`)
  onboarding_wizard.py — Shared desktop/web onboarding bootstrap and validation (§2)
  subscription_install_presets.py — Pure sibling install compilers from one draft and one discovery snapshot; all-or-nothing (§2)
  settings_setup_contract.py — SSOT for the setup contract, derived bootstrap state, payload validation and the `TOTAL_BUDGET` resolver `resolve_total_budget_usd`
  owner_mailbox.py — Per-task user message mailbox: revocation-aware drain, proven-empty peek, the closed provenance set (`ancestor_task`, `peer_via_ancestor`, `system`, `descendant_task`, `independent_task`, `peer_task`)
  peer_roster.py — Host-listed roots from queue_snapshot/direct_roots, hidden included, with their recorded waits; also admits source-bound inline Presence mailboxes for `forward_to_worker` (§12; §6 Owner routing verbs)
  launcher_bootstrap.py, runtime_venv.py — Bundle-to-repo bootstrap, launch options, managed sync, native-host artifact synchronization and `--runtime-venv` for launcher.py (§2)
  launcher_onboarding.py — First-run onboarding as the desktop launcher presents it: the gateway /onboarding page (§2)
  launcher_server_reaper.py — POSIX same-install stray-server termination by the PID-lock-owning launcher (Runtime topology below)
  launcher_windows_runtime.py — Windows-only pythonnet/pywebview runtime preparation
  launcher_background.py — Desktop background mode: close vs quit, the one consent question, second-launch activation (§9); `DesktopApi`, the alert half the bridge's `MainApi` inherits (§3)
  launcher_appearance.py — A desktop window's native Windows caption takes the light or dark tint of its page's painted palette (§3)
  launcher_tray.py, launcher_tray_macos.py — Windows notification-area icon; macOS menu-bar item, Dock reopen, quit marking (§9)
  desktop_notifications.py — Desktop system notifications, one adapter per platform; submitted/unknown/typed refusal, click token back to the page (§3; DESIGN §9)
  desktop_autostart.py, windows_autostart.py — Host sign-in adapter table: Windows registry, macOS LaunchAgent, Linux systemd/XDG state (Runtime topology below)
  plan_intention_ledger.py — An ask's `intention` review-ledger record; never a verdict
  plan_record.py — The per-task plan record (schema 3): revisions, asks, typed replies, folds, legacy import (§6 Plan construction and review)
  plan_review_facts.py — Plan facts for the late-settlement row: counts and the objection fold, never a score (§6 Post-task work)
  provider_models.py — Model-ID helpers; `ACTIVE_MODEL_SETTING_KEYS` / `LEGACY_MODEL_SETTING_KEYS` keeps Heavy out of live route selection
  runtime_mode_policy.py — Protected-path policy (§6 Safety and runtime mode)
  schedule_contract.py — Schedule id, 5-field cron and IANA timezone validation SSOT
  reflection.py — The reflection log's routed store (history and late-settlement host rows) and the one failed-call reading (§6 Post-task work)
  post_task_evolution.py — The worker writes a durable promotion signal; only the supervisor idle tick applies it through the gated enqueuer (§6 Background consciousness and Evolution)
  repo_remotes.py — Role-based remotes: `managed` is the read/update-only official source, `origin` the personal target from the GitHub token
  review_evidence.py — Same-execution commit-review evidence (§6 Commit review evidence) and the bounded task-acceptance packet; execution facts are visibility only (§6 Task acceptance)
  review_evidence_refs.py — Leaf SSOT of the evidence-ref vocabulary; an unsupported claim cannot certify (`CLAIM_ID_UNSUPPORTED`)
  review_status_projection.py — Commit-review status projection over `review_state` records; re-exported by review_evidence.py
  semantic_dedup.py — LLM-first semantic dedup, fail-open None; consumed by review_state
  betterleaks_runtime.py — Pinned Betterleaks runtime resolver (six platform artifacts, packaged resource first)
  skill_loader.py — Skill discovery over `data/skills/{native,clawhub,ouroboroshub,external}` and `OUROBOROS_SKILLS_REPO_PATH`; `.self_authored.json` marker; per-skill state under `data/state/skills/<name>/` (§13)
  skill_catalogue.py — compact whole-record skill index pages and named full diagnostics with a physical manifest read address (§13)
  skill_readiness.py — Execution readiness and next actions from review, hash, enablement, grants, dependencies and peer conflicts
  skill_peer_inventory.py, skill_conflicts.py — Execution-time peers without payload reads: the non-executable `SkillPeer` projection and the one conflict verdict a loaded skill and a peer descriptor share (§13)
  skill_dependencies.py — Dependency-spec resolution and installed-readiness probe for skills
  skill_repair_admission.py — Selected-skill development admission: immutable `base_content_hash` against observed `expected_content_hash` before each payload operation; no long shell lock or rollback (§6 Skills and extensions)
  skill_owner_attestation.py — Owner attestation lane: the owner may skip the LLM skill review for skills they authored
  skill_publish_snapshot.py, skill_publish_scanner.py, skill_publish_result.py, skill_publish_github.py, skill_publish_eligibility.py — Publication leaves: captured-byte authority, exact-byte Betterleaks evidence, typed attempt/receipt, GitHub transport after the local gates, author publication authority (§6 Skill publication)
  skill_review_status.py — Verdict aggregation → `executable_review` (anchors the §13 readiness statuses); an Advisory author acceptance may outlive a stale critic hash, Blocking requires fresh critic authority
  skill_review_passes.py, skill_review_collect.py — One multi-model pass or chunked quorum, reserving the complete operation roster before dispatch; exact-wave collection without paid dispatch
  skill_review.py — Skill review orchestration: deterministic preflight, then the tri-model gate against the Skill Review Checklist (docs/CHECKLISTS.md) and docs/CREATING_SKILLS.md
  skill_review_prompt.py, skill_review_packs.py, skill_review_output.py, skill_review_rebuttals.py — The skill reviewer's leaves: prompt contract and waves, the reviewable payload, parsed findings and rendering, review-history evidence read before re-judging
  skill_review_history.py — Write-ahead review-history marker and the idempotent `state/skill_review_root_tasks.jsonl` projection; a failed append is the typed `skill_review_history_append_failed` event
  skill_review_cycles.py — Paid skill-review cycle counting, $0 replay and typed exhaustion; the cap SSOT is review_cycles.py
  extension_loader.py — Extension loading: in-process pure-Python via `PluginAPIImpl`, child-process proxies for isolated-dep and native extensions (§13)
  extension_process_runner.py — Extension child processes: scrubbed env, per-skill deps, timeouts, graceful host errors
  extension_route_stream.py — Portable stdio response frames and ASGI relay for out-of-process extension routes (§3 Out-of-process extension responses)
  extension_ui_validation.py — The host-owned declarative-schema-v1 widget validator
  extension_isolated_deps.py — Non-reentrant reader/writer leases for `sys.path`, polled asynchronously; dependency RLock also serializes owned importer-cache sweeps against double deletion, not plugin execution
  extension_health.py — Durable process-qualified per-skill health at `data/state/skills/<name>/health.json`; server observation is authoritative, worker observation a handoff-qualified view
  extension_plugin_api.py, extension_registry_state.py, extension_liveness.py, extension_child_catalog.py, extension_import_staging.py, extension_surface_names.py — The extension runtime's leaves: the `PluginAPI` handed to `register(api)`, live-surface registries, liveness, child-catalog validation, staged import trees, provider-safe surface naming
  skill_token.py — Opaque Host Service token minting/validation (§12)
  marketplace/ — ClawHub and OuroborosHub (§13)
    clawhub.py, ouroboroshub.py, fetcher.py, adapter.py, install.py, install_specs.py, isolated_deps.py, provenance.py — Registry clients (the hub update is an adopt transaction with verified rollback), fetch, manifest adaptation, install and dependency metadata, isolated deps, provenance and the publication receipt
  skill_lifecycle_queue.py — Single FIFO skill-mutation lane and event snapshot (§13)
  skill_lifecycle_actions.py — Shared grant/toggle effects and owner-action admission for UI, launcher, CLI and task adapters; no new permission store
  skill_uninstall_state.py — Marketplace-uninstall tombstones and explicitly authorized local payload/state deletion, with separate retention contracts
  skill_review_runner.py — Writes `review_job.json` and `skill_review_*` events; separate review, dependency and extension outcomes; a replayed verdict never overrides an owner disable
  server_auth.py — Non-localhost network gate via `OUROBOROS_NETWORK_PASSWORD`; warns when unset (§8)
  server_control.py, server_entrypoint.py, server_runtime.py, server_web.py — `restart_current_process` and `execute_panic_stop`; CLI parsing, data-root admission and port binding; startup wiring and WS liveness; `ServedUI` (§4 main interface) and `read_author_kit_assets(repo_dir)`
  server_process.py, server_liveness.py, server_maintenance.py, server_restart.py, server_owner_routing.py, server_routing_context.py — Server leaves the composition root calls: per-process facts, wedge detection, drive upkeep, restart operations, owner-message routing and its per-turn facts (§9)
  terminal_cost_reconciliation.py — Usage recovery/projection (§6 Budget tracking)
  task_continuation.py — Durable review continuation state
  task_results.py — Task-result merge-writes (one store transaction) and typed readers; the locked `task_acceptance_review_accounting` claim is unknown without a recoverable terminal host run, never permission to re-dispatch (§6 Task acceptance)
  task_store.py — Indexed facts/signatures, review/plan candidates and documents; implicit body+handoff snapshots decode after close, explicit read/modify stays (PERSISTENCE.md)
  pause_notices.py — Confirmed-pause System disclosures: pending episodes, saved-chat receipts, off-loop local replay
  task_result_schema.py — Task-result schema admission: the `_schema_version` stamp, the classifier, and the quarantine an unstamped, future or malformed row lands in
  task_status.py — Effective status, lineage and waits; `execution_owner` and dated `execution_observation` separate lifecycle from liveness (§5); `task_has_live_queue_ownership` supplies the worker-side cancel predicate and fails open toward liveness (§10 invariant 14)
  git_shell_policy.py — Shell Git argv checks
  protected_artifacts.py — Execute-only black-box policy for protected artifacts
  shell_parse.py — Shared command/argv normalization and POSIX wrapper grammar; observed targets, not permission judgments (§6 Safety and runtime mode)
  argv_budget.py — Argv admission counts encoded bytes of argv plus environment (ARG_MAX charges both); asked by skill_exec before exec
  workspace_executor.py, docker_process_observer.py — `local` and network-none `docker_exec`; Docker liveness needs `sh`, `cat`, readable `/proc` (§9)
  deliverables_paths.py — Lexical and case-folded deliverables path views
  tool_capabilities.py — SSOT for the core, parallel-safe, untruncated and stateful-browser tool sets and the cognitive-memory tool class every Presence ceiling carries
  tool_access.py — ToolProfile × ResourceRoot × Operation matrix, affordance map, closed-enum `required_capabilities` check
  tool_access_types.py, tool_access_roots.py, tool_access_paths.py, tool_access_user_files.py, tool_access_reads.py — Matrix types and physical roots; inherited reads, separate action authority (§6 Resource roots and physical file identity)
  tool_policy.py — Round-one tool visibility (the sets live in tool_capabilities.py)
  browser_policy.py — The browser tool's target and control-request policy: task-granted concrete origins, metadata/private/reserved refusals, the three-valued `runtime_service_kind` (§6 MCP and browser-facing external tools)
  skill_payload_binding.py — Skill payload targeting: `.seed-origin` distinguishes native from external; read/list/search only for read profiles
  utils.py — SSOT for atomic JSON, timestamps, hashes, sanitization, subprocess helpers, `truncate_review_artifact`
  jsonl_tail.py — `JsonlChainSnapshot` owns captured byte reads for history, wake and Presence reentry; bounded filtered tails serve history, logs, routing and task context
  markdown_source.py — Byte-preserving Markdown structure shared by books and knowledge notes; physical ranges tied to source SHA; `MarkdownSourceError` keeps a missing grammar or malformed YAML visible without replacing original bytes
  world_profiler.py — Generates WORLD.md
  contracts/ — Frozen ABI package (§11)
    tool_context.py — ToolContextProtocol
    tool_abi.py — ToolEntryProtocol + GetToolsProtocol
    chat_id_policy.py — SSOT for human-visible vs synthetic chat ids (§12)
    task_contract.py — Frozen task-contract normalization and effective acceptance-claim binding (semantics: §11.1)
    task_constraint.py — `VALID_WRITE_SURFACES` + surface/write_root validation, fail-closed
    skill_payload_policy.py — Payload path resolution/confinement/sidecar detection
    skill_manifest.py — Unified skill manifest parser (`VALID_SKILL_TYPES`: instruction|script|extension)
    schema_versions.py — Opt-in `_schema_version` stamping helpers (§11.2)
    record_contract.py — Record passport: the local ledger rows external observers may rely on (§11.1)
    plugin_api.py — PluginAPI, ExtensionRegistrationError and the closed vocabularies `FORBIDDEN_SKILL_SETTINGS`, `VALID_EXTENSION_PERMISSIONS`, `VALID_EXTENSION_ROUTE_METHODS` (§11.1)
  gateways/ — Thin outbound transport adapters; no business logic
    claudexor_images.py — requestKind=image upload and whole-result ACK (docs/IMAGE_GENERATION.md)
    image_api.py — Image API routes/credentials; one send, bounded results, cost evidence
    claudexor.py — Loopback descriptor/handshake/runs/quota; private token; owned daemon; `discover_daemon_at`: daemon/control-api.json (§6)
    claudexor_run_events.py — Catalog-negotiated, bounded run-journal SSE frames and durable seq cursors; no polling owner (§6 Delegated subagents)
  claudexor_runtime.py — Reviewed next-spawn pin in `data/state/cx`: seed/download, verify/probe, atomic promote; no mutable current; override `OUROBOROS_CLAUDEXOR_BIN` (§6)
  claudexor_exit_facts.py — Bounded saved exits; live memory/loop/store/admission facts and dated capacity snapshots, without lifecycle policy (§9)
  claudexor_daemon.py — Owned `data/claudexor`: lazy start, authenticated attach, `stop_outcome`, failed-start latch, `install_missing_harness_cli` (§9)
  claudexor_startup_failure.py — Stdlib `ExitFact`; `failed_without_control` drives spawn latch; diagnostic-only log labels and latch record (§9)
  gateway/ — Gateway Boundary v1: browser-facing route ownership and the frontend contract SSOT (Gateway Boundary v1 below)
    contracts.py — Active WS/HTTP envelope contract owner
    decision_contracts.py, history_contracts.py, attachment_contracts.py, schedule_contracts.py — Typed leaves re-exported by contracts.py: decisions, paged Chat, attachments, schedules; each ingress owns validation
    ui_i18n_contracts.py, model_route_contracts.py, mcp_import_contracts.py, client_contracts.py, management_contracts.py — interface-language envelopes (`/api/ui/i18n*`), model-route and MCP import previews, replaceable-client and product-management envelopes, separate for contracts.py's size cap; browser twins `web/modules/ui_i18n_types.js`, `model_route_types.js`, `mcp_import_types.js`, `client_contract_types.js`, `management_contract_types.js`
    endpoint_index.py — `HTTP_ENDPOINTS` index, re-exported by contracts.py, and its `HTTP_SHAPES`/`WS_SHAPES` bindings; routers own the Route objects
    schema.py — Executable gateway contract: JSON Schema derived from the TypedDicts, validating ingress; `contract_document`: `GET /api/contract`
    router.py — Starlette route collector for /api/* and /ws (§4)
    ws.py — WS manager, extension WS dispatch off the ASGI loop, broadcast (§4 WebSocket protocol)
    state.py — /api/health, /api/contract and /api/state
    tasks.py — Headless task create/list/get/cancel/events; cancel accepts `stop_policy` (`finalize_then_cancel` → supervisor/owner_stop.py) (CLI / Headless Boundary below)
    task_archive.py — Confined single-file and directory-ZIP reads of a task's own stores
    task_events.py — Task-event SSE endpoint: GET ranks plus read-only POST v2 physical-chain cursors (§3 History reads and the SSE v2 transport)
    task_hurry.py — POST hurry ingress: a one-field `{request_id}` body, extra fields refused: hurry carries no text by design and a smuggled field must not become a side channel (owner_hurry.py)
    task_pause.py — POST owner Pause `{request_id}`; answers after the durable root fence (owner_pause.py; §6 Owner Pause)
    task_continue.py — POST owner Continue `{action_nonce}`; one nonce answers one admission (owner_continue.py; §6 Owner Continue)
    main_decisions.py — Main's current questions and human-required controls, gaps explicit (§3)
    task_decision.py — The one `POST /api/decisions` ingress with family-parsed ids (`quiz:` here, `routing:` → routing_decision.py, `interaction:` reserved); writes `KIND_QUIZ_ANSWER` (owner_quiz.py; §11.1)
    task_model_wait.py — Shared model-wait decision effects over the existing mailbox, with live-owner/revision checks
    routing_decision.py — Validates a click against the durable `needs_manual_target` row, recovers the original text, dispatches `steer_task`/`promote_chat_to_task`, confirms through routing_wait receipts
    logs.py — Read-only runtime log tail; storage.py — Storage routes
    onboarding.py — `POST /api/onboarding/complete`: the one atomic onboarding write; a typed 503 persists nothing, except `settings_save_timeout`, the unknown outcome (order: §2)
    onboarding_host.py — GET /onboarding and /api/onboarding/bootstrap: one side-effect-free wizard bootstrap, as page or JSON
    owner_settings.py — Settings-lock-as-precondition and `CommitBoundary` (Gateway Boundary v1 below); owner_effort.py — the effort range
    settings_secrets.py — Explicit single-secret Settings reads; passive Settings responses stay masked (§3 Settings and onboarding)
    settings.py — /api/settings and /api/owner/*; `GET /api/review-pool` (`ReviewPoolResponse`); an unreadable catalog is a typed `config_error`, never a 500 (§7 Review pool)
    presence_settings.py — Owner-facing runtime overrides and working-folder selection for reviewed Presence behavior skills
    desktop_autostart.py — GET/POST /api/desktop/autostart and /api/desktop/background (§4)
    control.py — /api/reset, /api/command, /api/git/*, /api/update/*, /api/evolution-data
    update_progress.py — Process-local stages owned by the synchronous update executor; status projection and WS invalidation, never recovery authority
    schedules.py — Cron schedule HTTP surface
    files.py — File Browser and chat upload
    ui_preferences.py — `state/ui_preferences.json`: widget order and widths, per-card start modes (`extension_ui_validation.WIDGET_START_MODES`), nested subagent expansion, the empty-Main `welcome` copy
    ui_i18n.py — `GET/POST /api/ui/i18n*`: interface language and translation memory (i18n_memory.py); the language POST is a locked owner-settings write of `OUROBOROS_UI_LANGUAGE`
    models.py — Model catalog, provider probes, local-model lifecycle
    extensions.py — Extensions/skills HTTP surface (routes: §4)
    extension_receipts.py — Process-qualified extension index/toggle/reconcile receipt projection
    widgets.py — GET /api/widgets: the Widgets card list from the in-memory extension snapshot, with no discovery, reconcile, hashing or writes on the read path; homes the widget TypedDicts contracts.py re-exports
    skill_publish.py — Read-only publish preflight with scan cache; one five-state response; no task or GitHub effect (§6 Skill publication)
    marketplace.py — ClawHub and OuroborosHub HTTP surface
    mcp.py — MCP HTTP surface over the shared MCPManager
    claudexor_accounts.py — Thin owned-daemon status/login/account proxies; no auth logic or browser token. `reads` classifies catalog/accounts/quota, `resource_capabilities_read` operations; only a successful read proves absence (§3 Agent accounts; §4)
    claudexor_passive.py, claudexor_contracts.py — passive quota read and envelope types (docs/PASSIVE_QUOTA_READ.md)
    claudexor_quota.py — Owned-daemon refresh/reset/receipt proxies; catalog negotiation, exact key/body and typed errors, no start/retry/browser token (§3 Agent accounts; §4)
    harness_maintenance.py — Owner maintenance HTTP surface over the shared host service (§6 Vendor program maintenance; routes: §4)
    host_service.py — Loopback-only Host Service API (§12)
    host_notify.py — POST /notify beside the Host Service: a granted skill's sentence becomes one signed `skill_notice` System row in the owner's chat (§12)
    history.py — Shared Chat room/quiz/media/review/terminal projection and cost-breakdown factories
    history_paging.py — A room's own pages over retained chat/progress JSONL chains, frozen room-bound cursors, read gaps, a Project room's `latest_arrival`; no stored history copy (§3 History reads and the SSE v2 transport)
    history_segments.py — Process-local per-archive summaries that let a Project read skip archives holding none of its rows
    cost_breakdown.py — Ledger-derived dashboard buckets and root-task detail over the same physical-attempt authority
    projects.py — GET/POST /api/projects, /from-task, /update, /delete
    _helpers.py — Shared request-root/coercion/JSON error envelope and `run_sync_to_completion`, the settled worker wait for request-owned blocking work
  tools/ — Auto-discovered tool plugins
    registry.py — Tool registry SSOT: loads tool modules, exposes schemas, executes safely; owns the shell-guard/process-tool membership sets
    core.py — File/data tools (read_file, list_files, write_file, edit_text), delivery tools (send_photo, send_video, send_file, send_links), search_code, escalate, forward_to_worker
    core_file_tools.py, core_secret_paths.py, core_artifacts.py — File reads/lists; delegated action/runtime helpers; human artifact delivery (§6 Credential mutation and diagnostic redaction)
    shell.py — Process tools `run_command`/`run_script` (§9)
    shell_guards.py — Shared process-path inspection helpers and target extractors; process admission is owned by registry_guard_process.py (§6)
    registry_core.py, registry_guards.py, registry_guard_process.py, tool_context.py — Registry load/schema/dispatch, the capability/resource/update/skill guards, process admission and observations, `ToolContext`/`BrowserState` (protocol: contracts/tool_context.py) (§6 Safety and runtime mode)
    git.py — Git/write tools with the deterministic, test and one-wave review commit gates and the author's optional preflight (§6 Git and commit review)
    git_plumbing.py, git_repo_edit.py, git_vcs_ops.py, git_review_cycle.py, git_managed_postcommit.py, git_evolution.py — The git tool's leaves: plumbing, the uncommitted repo write and edit surface, VCS inspection and rollback, staging plus the optional preflight and the one-wave two-part review, managed-merge post-commit gates under Pause, evolution-campaign authority at the reviewed-commit and publication boundaries
    search.py — `web_search`, one leg per Source/Model of `ouroboros/search_routes.py` (§6 Web access mechanisms)
    browser.py — Playwright browser tools with per-ToolContext lifecycle and thread affinity (§6 MCP and browser-facing external tools)
    imagegen.py — list_image_models/generate_image/recover_image; explicit routes, artifacts first, separate view/send (docs/IMAGE_GENERATION.md)
    vision.py — Vision LLM tools for browser screenshots and uploaded images
    vision_process.py — Tracked vision-child IPC: validated receipts, result recovery, parent-owned cancellation
    knowledge.py — Persistent topic-based knowledge files with an auto-maintained index
    chronicle.py — `chronicle_write`/`memory_read`/`memory_mark`, exported by knowledge.py: host-expanded page row sets, task stamps, checked quotes; reads paged at the source (§6)
    memory_tools.py — Memory registry tools for data sources, gaps and trust
    health.py — Codebase health tool: complexity metrics and self-assessment
    compact_context.py — LLM-requested tool-history compaction trigger, applied on the next round
    control.py — Control tools: restart, timeout settings, scheduling, review, chat history and model switching; publishes the `schedule_subagent` contract and the compact `wait_task` projections (§6 Delegated subagents)
    control_delegation.py — Delegation-budget and in-task project-scoping affordances (`ensure_project_scope`; §6 In-task project scoping)
    control_events.py, control_routing.py, control_runtime.py, control_scheduling.py, control_subagent_spec.py, control_task_results.py — The control tools' leaves: control events and their durable outcomes, routing work into a supervised task, runtime self-control, scheduling one live subagent, the `schedule_subagent` parameter surface, absorbing a child's result
    tool_discovery.py — List/enable owner: callable catalog by namespace in every mode; grants nothing
    tool_result.py, tool_catalog.py, tool_resolution.py — Dispatch-side vocabulary: the typed internal tool result and its byte-compatible text adapter, intrinsic tool descriptors, argument normalization with physical target binding
    arg_feedback.py — What a tool says about an argument it did not obey: the one-line ignored disclosure and the typed refusal naming field, value and repair (DEVELOPMENT "LLM-first affordances")
    review_response.py — Response-envelope projection for multi-model review rows
    shell_process.py, shell_effects.py, shell_outputs.py — The command-running substrate: process execution, working-tree effects and their throwaway share, declared process outputs and per-path export eligibility
    plan_review_artifacts.py, plan_author_history.py — Earlier-schema waves and author lineage for history views and the importer
    evolution_stats.py — evolution.json metrics from sampled git history
    owner_delivery.py — Owner event delivery: live queue or the `pending_events` fallback with sticky deferral
    deliverables_shell.py — cp/mv/ln into deliverables with symlink checks
    shell_audit.py — Post-exec custody audit for process tools
    process_facts.py — Per-call selected environment, secret egress masking and typed process/runtime facts for loop_tool_execution.py
    write_shape.py — Interpreter/non-interpreter write-shape predicates behind shell_parse.py and shell_guards.py; process permission is owned by the task/resource and Supervisor contract (§6 Safety and runtime mode)
    extension_dispatch.py — Extension tool dispatch; discovery stays in registry.py
    release_sync.py — `sync_release_metadata` (version carriers) for commit-admission preflight; the carrier-span SSOT `VERSION_CARRIER_SPANS` shared by the managed-update resolver and the commit packet's carrier cut (§10 invariant 2)
    review_synthesis.py — Shared synthesis helpers
    preflight_review.py — `preflight_review` (`advisory_review` alias): one-row early worktree review; `deterministic_only=True, source=...` returns free release metadata plus separately labelled worktree `book_balance`; `review_status` reads attempts, obligations and readiness debt (§6 Commit preflight)
    recent_tasks.py — Read-only context recovery
    commit_gate.py — Commit gate: LLM claim synthesis, block classification, the free identical-verdict refusal, paid review-cycle counting and ceiling, the review-contract fingerprint
    git_rollback.py — Emits `rollback_request`; the supervisor runs the UI Rollback door's procedure (P4-S)
    git_pr.py — Five PR tools (non-core)
    github.py — Issue, PR and checks tools (frozen tool module): the shared process binding selects the active Project; discovery reads token sources or native CLI configuration without an authentication probe; a refusal before the first `gh` launch is `completed_no_effect`
    github_checks.py — The reader behind `get_github_checks`: one commit's workflow runs, jobs, failure annotations and pull request rollup, as facts and named unavailable sources under one deadline
    parallel_review.py — Prepares and admits every seat before the review wave sends (§6 Surfaces and money admission)
    plan_review_references.py — Reference projection with its own provenance rows (`logs/progress.jsonl`); no plan authority
    plan_review.py — `plan_task` entry: free write and read, the paid ask, typed `PLAN_*` refusals (§6 Plan construction and review)
    plan_ask.py — One ask: participants, body predicate, delivery, continuation, dispatch, settlement
    plan_review_runtime.py — Review-slot runtime of an ask (`run_plan_review_slots`, `plan_review_slots`, fit, deadline)
    plan_spec.py — `resolve_constitutional` and path normalization; earlier-schema spec reads; no I/O
    plan_evidence.py — Bounded plan-evidence manifest; the runtime data plane is denied
    plan_packet.py — Delivery builder and `PLAN_REPLY_CONTRACT`
    plan_delivery.py — A repeat participant's cross-talk and retained texts
    review.py — Acceptance review and multi-review adapters
    review_multi_model.py, review_file_pack.py, review_prompt_text.py — Commit-wave packet fan-out, working-tree file packs and prompt vocabulary; `span_only_release_carriers` is the packet's carrier cut, `triad_pack_exclusions` avoids inline governance duplicates (§6 Guaranteed-fit ladder)
    review_checklist.py — The layered change-review checklist as the review surfaces read it: one `## Header` section of docs/CHECKLISTS.md, the layers a subject is judged by, `checklist_fingerprint` on every review ledger record (§6 Change review on any root)
    review_context_atlas.py — `repository_index`: compact tracked-path map plus touched-file and direct-importer facts; no file bodies, no restriction on what a reviewer may read (§6 The coupling question)
    governance_context.py — Shared governance tiers for every change-review seat, the preflight and deep review: stable inline rules, bounded change-class rules, physical-source navigation; every omitted inline body has an explicit disposition (§6 Governance delivery)
    query_code.py — Read-only code navigation: scoped/paged outlines and digest, source occurrences and call evidence, candidate import impact; explicit `user_files` targets and permitted subagent reads (Code navigation below)
    edit_ops.py — `apply_patch` and `edit_batch` with the shared syntax check and unified diff backing write_file; the diff pulse in every system-repository edit result
    media.py — `ocr_pdf`, `youtube_transcript`, `extract_video_frames` (dependency-optional, typed capability envelopes; frames under `artifact_store/video_frames`)
    verify.py — Independent checks through the shared pre-exec guards, not a process-command tool by design; receipts at `task_results/artifacts/<task_id>/verification_receipts.jsonl` (§6 Tool capability and execution)
    review_helpers.py — Shared review helpers: governance-doc loading, checklist section slicing, the prompt-size SSOT, the density-calibrated input cap and probe sample `DENSITY_PROBE_SAMPLE_CHARS`
    review_binary_context.py — Staged/parent Git object metadata for review packs
    review_subject.py — Frozen review subjects, managed-resolution deltas and isolated reviewer checkouts; identities and reuse derive from the captured subject (§6 Subject operation)
    review_change.py — `review_change`: one review wave and ledger record on any registered root, without committing (§6 Change review on any root)
    review_change_custody.py — Paid-attempt custody and rejoining an open `review_change` wave without paying again (§6 Subject operation)
    review_admission.py — Review packet fit, retrieving briefs and whole-wave money admission (§6 Surfaces and money admission)
    review_revalidation.py — Review-contract fingerprint revalidation
    review_brief_coupling.py — Retrieving review brief: Part 1 the change, Part 2 the coupling questions, over one frozen subject (§6 The coupling question (Part 2 of the brief))
    scope_window.py — Scope window sizing and evidence provenance; fallbacks stay disclosed, never review authority
    scope_review_contract.py — Pure parser of the `coupling` answer (`normalize_scope_items`), also consumed by scripts/validate_scope_receipt.py
    scope_required_sources.py — Change-relative protected/prompt/contract sources, derived families and twins; the final manifest hash and policy version bind review replay
    services.py — Service mini-manager with process-group cleanup
    skill_exec.py — list_skills/skill_review/toggle_skill/skill_owner_action/skill_exec over a fixed interpreter allowlist; gated by enablement, fresh review and hash
    skill_publish.py — Thin publish transaction over the four leaves; success is PR-receipt-gated
    skill_preflight.py — Read-only skill preflight; a module widget's `render.entry` is containment-checked and parsed as a classic script: the widget frame runs it inline
    project_journal.py — journal_write/read, workpad_read/write, `update_focus`, journal_tail_digest; foreign project reads honoured for roots, foreign writes refused; owns `mirror_tree_coordination_to_journal`
    presence.py — configure_presence, initiate_presence, typed completion/cancel
    task_tree.py — tree_note/tree_read (storage SSOT: task_tree_ledger.py)
    followup.py — The agent's two schedule tools over `state/scheduled_tasks.json`: `manage_schedules` (list; root-only audited disable/delete/restore) and `schedule_followup` (`notify=true` leaves a note instead of a wake)
    join_ledger.py — Child-result absorption: validates lineage and exact hashes; dispositions integrated/irrelevant/deferred; `CHILD_RESULT_STALE`; keeps peek_task/discard_child_result
    delegate.py — Delegation facade verbs `delegate_start` (with `retry_of`), `delegate_wait`, `delegate_cancel`, `delegate_answer`, `delegate_message`; the host pre-start rides the same wrapper (§6 Delegated subagents)
    delegate_integration.py — Delegated mutation authority, snapshots and terminal capture; skill-payload CAS apply with extension reconciliation (§6 Snapshots, capture and disposition)
    delegate_payload_patch.py, delegate_terminal_evidence.py, subagent_integration_delegated.py — Delegation leaves: the skill-payload patch pipeline, the terminal story of one delegated run, `integrate_delegated_patch`
    subagent_integration.py — integrate_subagent_patch (sha256, 3-way --index, protected-path gated, genesis refused), external-workspace audited verdict, `coop_already_in_tree` no-op, compare_subagent_patches
    patch_verdict.py — The one verdict writer for both patch pipelines: subjects are minted by the writer (`run_<rid>`), never prefix-matched by readers; artifact plus typed `delegate_run_patch_verdict` custody row
  delegate_start_claims.py — Serializes custody recheck and `START_REQUESTED`; transport/run waits are outside locks. Unsent retries may hold actor/payload locks, never root/store locks
  process_containment.py — Env-token container membership (`OURO_PROC_CONTAINER_*`: Linux /proc, macOS `ps -E`, Windows Job Object); an alive-or-undeterminable member is an honest hard-block answer, never a kill guarantee (§6 Delegated subagents)
  process_scope.py, process_census.py — work scopes (membership, records, task-end/Panic reap, engine index); kernel-state readers (§9)
  process_custody.py — `spawn_supervised` and the durable process_ledger.jsonl; `reap_orphaned_processes` with strict identity and `retained_purposes`; the parent lifeline; `stop_ledgered_processes` requires measured identity and confirmed exit (Runtime topology below; §9)
  obligations.py — Atomic current sets (incl. `owner_questions`); owning transitions publish before work and retire after discharge
  startup_migrations.py, startup_task_files.py — Explicit inherited-state import/repair and addressed file recovery
  delegate_custody_current.py — Custody on the store: run rows, pending invocations, outbox, importer's fold
  owned_shutdown.py — The ownership set `state/owned_processes.json` (both custody funnels write it) and the one bounded exit stop `stop_owned_work`; `finish_unconfirmed_stops` retries leftovers at the next start (§9)
  platform_layer.py, platform_win32_abi.py — OS process/lock helpers, descendant seam, private Win32 ABI (Platform substrate below)
  verified_download.py — Shared exact-size/digest verification and atomic cached byte delivery
  node_runtime.py — Execution-probed Node runtime health (`node_runtime_health`; a missing binary is never cached), `select_skill_node_runtime`, `skill_node_emergency_path_dir`

```

### Code navigation

`query_code` exposes source evidence and local syntax facts through ten operations; occurrences keep source anchors and raw grammar slots, so the model inspects relationships without a prescribed traversal, and a syntactic call position does not establish binding. The cache (`code_intelligence.py`) holds only local facts and every query rechecks current bytes and paths, so adding or removing a target cannot leave an unchanged importer with stale cross-file joins. Replies disclose method, scope and incomplete coverage. Parameters and limits: [code navigation contract](../code-navigation.md); no compiler, binding engine or language server is implied.

### Devtools boundary

`devtools/` is excluded from runtime imports and package discovery; review applies and outputs stay external. A benchmark launcher stamps the isolated-benchmark sentinel into its throwaway data root, turning off JSONL log rotation and hot-store-growth warnings there (`supervisor/state.py`, `agent_startup_checks.py`). Each benchmark's own README keeps its module map and methodology: `devtools/benchmarks/cybergym/` pauses dispatch on three typed gates (dead gateway transport, refused budget claim, a sibling's unresolved workspace-custody gate) and keeps never-dispatched ids row-free; `devtools/benchmarks/cowork_bench/` keeps scoring authority in its audit (`METHODOLOGY.md`); `devtools/benchmarks/osworld/` splits runners from their owner leaves. `devtools/e2e_live/` is the live E2E stand: isolated real servers running the owner-shaped scenarios SM1, SW1 and SK1, judged on durable artifacts and a browser probe (manual `devtools/e2e_live/README.md`; the one rule binding runtime changes: DEVELOPMENT "Live E2E stand").

### Gateway Boundary v1

`ouroboros/gateway/` is the single inbound browser/CLI boundary; `ouroboros/gateways/` holds the thin outbound adapters. `files.py` and `host_service.py` stay separate trust boundaries. Handlers translate transport into calls on existing runtime owners and hold no second copy of queue, review, settings or lifecycle policy. The facade exists for dependency direction: the UI evolves without importing the agent body, the runtime without ad-hoc browser contracts. `GET /api/contract` publishes that boundary from the endpoint and shape indexes (meaning: `docs/CLIENT_CONTRACT.md`), so a replacement interface needs no core source. `gateway/owner_settings.py` is the single owner-scoped settings write seam (every owner write calls `_owner_update_settings`): the settings lock is a precondition, so a timed-out acquisition refuses before any write, and `CommitBoundary` marks the commit instant with `saved` present on both sides, so a later-step failure is reported as that step, not as an ambiguous omission (read-change-write and digest rule: §7 Reading and writing the settings document). Frontend calls go through `web/modules/api_client.js` (typed mirror: §11); extension HTTP lives under `/api/extensions/<skill>/…` with namespaced WS dispatch.

### CLI / Headless Boundary

`ouroboros.cli` is a client of the same gateway and queue; there is no second task engine. Its parser (`ouroboros/cli.py`) is the command-surface SSOT. Streaming commands reserve stdout for the final answer, patch, result or JSONL and send progress to stderr, so automation can pipe results.

`POST /api/tasks` creates an ordinary managed root (§4); the CLI refuses any `delegation_role` other than `root`, and only `schedule_subagent` creates children. Admission reserves the task id and a worker-pool slot under one queue lock, off the HTTP event loop, so a cancelled HTTP waiter never cancels an admitted task; unconfirmed persistence answers 503 `unconfirmed` and keeps the row. Attachments are copied into the effective task drive before enqueue; a stored artifact reference is no host-path authority.

Workspace tasks default to `memory_mode=forked`; `shared` is refused for an external workspace and runs on a forked child drive in project scope. The stored `memory_mode` says what was requested and `drive_root` where the task executes, so isolation never depends on relabelling the request. The mode isolates the execution drive and knowledge seed; identity and scratchpad writes still land on the canonical root. Forked and empty drives live under `data/state/headless_tasks/<task_id>/data`: a fork copies `identity.md`, `WORLD.md`, `registry.md` and global knowledge (only `knowledge/patterns.md` for project forks); dialogue, scratchpad, mailbox and history never cross, and the child's context still comes from repository governance and the canonical root's shared memory. `memory_export.json` is an explicit artifact, never auto-merged.

`--detach` returns after durable admission; `--no-stream` polls to completion. `ouroboros run` exits 0 only for a completed lifecycle with a clean execution axis, no failed or degraded objective and a finished artifact bundle (`_is_terminal_success`), so shell automation cannot read "the model answered" as "the deliverable exists"; `--patch`/`--patch-out` are stricter and trust `workspace_patch.json`, which distinguishes an omitted, no-op and failed patch (§6 Headless finalization and workspace patch capture). An explicitly partial cost gets a bounded finality wait (`_await_cost_finality`) before its partial flags stay visible. CLI and skill-manifest schedules enqueue ordinary supervisor tasks; no parallel scheduler exists. `resync_skill_schedules()` mirrors manifests into the same table (§5); a blank timezone means the DST-aware system zone (fixed offset only when that zone is unrecoverable), and the active schedule digest rides task and consciousness context.

The packaged CLI wrapper/installer reuses the embedded interpreter, not a second PyInstaller runtime: `packaged_cli` finds the seed and bootstraps `~/Ouroboros/repo`, then runs `ouroboros.cli`; path overrides are scrubbed, explicit URLs retained. Packaged `server` is refused: it would bypass launcher-owned bootstrap, identity, restart and cleanup. `run --start` is loopback-only and starts the desktop app when no ready gateway answers. Release builds carry Node and ripgrep. Skill-side Node resolves bundled-first (`node_runtime.select_skill_node_runtime()`), the generic process launch surfaces PATH-first (`process_interpreters.resolve_process_node`): a healthy PATH node stays byte-identical in argv and child env; the bundled runtime substitutes only when PATH is missing or probe-dead, attests the prepend, never inside a non-local executor backend. `search_code` pre-enumerates policy-approved files before invoking ripgrep; a faster binary widens no search authority. Every bundled consumer searches `bundled_resource_bases()` in one order (`OUROBOROS_BUNDLE_DIR`, frozen root, interpreter-ancestor roots, source checkout): server and CLI children run from the managed repo, never an in-bundle module path; the ancestor step lets an older launcher start a newer checkout. The embedded interpreter never writes into the signed application (codesign seal): `embedded_python_env()` redirects bytecode to `data/state/pycache` and user installs to `data/state/python-userbase`; `pip_install_target_args()` adds `--user` only for the embedded interpreter; the userbase outranks bundle site-packages and is never pruned, so recovery is deleting it and relaunching.

Workspace binding changes the contextual repo, never the system repo for BIBLE, prompts and review governance. `/api/tasks` and project-room promotion share `workspace_admission.validate_workspace_root()`. Ordinary folders support file/process work; Git operations need a Git worktree. Binding changes file, process and VCS defaults plus memory, lease, preflight and finalization, removes no top-level tool and keeps the Architecture context in Max mode. The workspace executor is a process-routing boundary, not a sandbox: `executor_ref` is host-owned, `network=none` holds only when the backend implements it; executor processes enter durable custody. Preflight writes the `workspace_preflight.json` artifact with a bounded summary in metadata; `tools_on_path`/`tools_missing_from_path` report `shutil.which` PATH presence, not executability, frozen in durable task metadata; a collection failure is a disclosed error summary, never a fictitious artifact.

Ordinary-directory delegated sessions submit `execution.workspaceKind=directory` on the `agent` mode after negotiating `mutability.workspaceKinds` from the engine catalog; the parent selects `directory_strategy=direct|copy` and optional `scope_paths`. The engine owns copy materialization, per-file CAS, apply and discard; the host keeps the complete before/after artifact closure on the existing custody path (no second file engine); file or GUI work is never judged from an empty text diff. Complete input sets live in the artifact store: `attachment_manifest` is a bounded preview, `attachment_manifest_ref` names the full immutable JSON; a preview never substitutes for an unreadable full reference. The router continuation tools require an explicit `predecessor_task_id` (`""` means fresh); the predecessor brief is §6 Delegated subagents. System self-modification, external workspace and genesis stay distinct task classes; a genesis child's `deliverable_manifest.json` listing is discovery (its gap rows never fail the task); a copy or ZIP capture requires stable source bytes and identity. Global and system installs stay runtime-policy reviewed; `sudo` is non-interactive (`sudo -n`).

### Runtime topology

Host sign-in startup is an owner opt-in (Behavior, §3) whose only authority is the OS registration. `desktop_autostart.py` picks the adapter (Windows HKCU `Run`; macOS LaunchAgent with `--launch-intent automatic` and no `KeepAlive`; Linux systemd unit for deb/rpm, XDG autostart for AppImage/tar), available only to a packaged host whose launcher exports a new enough `OUROBOROS_APP_VERSION`. States: `unavailable`, `off`, `on`, `other_copy`, `disabled_by_os`; the launcher Panic gate and saved-pause recovery keep authority; `context.py` records `runtime_env.autostart` and the effective `keep_running_after_close` at task start as host facts. Keep running after close is the second control: `OUROBOROS_DESKTOP_KEEP_RUNNING`, an owner choice stored in settings, offered where the launcher exports `OUROBOROS_DESKTOP_BACKGROUND=1` (Windows and macOS windows); with both on, a sign-in start begins hidden with an indicator. Close, quit and Panic in that mode: §9.

`launcher.py` owns the PID lock, bundle bootstrap, server process, presentation, restart and cleanup outside the managed repo; `server.py` is the self-editable inner runtime. The native packages' opt-in systemd unit passes `--launch-intent automatic` so Panic survives, with no restart policy: the launcher owns restarts and the crash fuse ([packaging/systemd/README.md](../../packaging/systemd/README.md)). Spawn custody: POSIX children start in a new session and process group; Windows creates the server suspended, assigns a kill-on-close Job, then resumes, refusing to run without Job custody; only the shared daemon requests breakaway. The launcher records `data/state/server_process.json` with the child's measured birth/command fingerprint and signals only that re-proved identity; an unmeasurable or unconfirmed predecessor keeps its record and blocks the next generation; forced cleanup spares the shared daemon subtree (§9).

Admission: every server entry (`server.main`, an imported app's lifespan) holds `<data>/state/server.lock` for life (`server_entrypoint.admit_data_root`): one server per data root, whatever its repository, PID file or port; records and bindings stay evidence; a running pre-lock server goes undetected.

Same-install reaper (`launcher_server_reaper.py`): under the PID lock, each generation first reaps strays proven by three live facts (the exact `<REPO_DIR>/server.py` argv token, `OUROBOROS_DATA_DIR`, `OUROBOROS_MANAGED_BY_LAUNCHER=1`) read byte-exact from /proc; `ps -E` mixes argv into the environment and authorizes no kill, so non-/proc hosts stay report-only. A missing custody row is the defect being repaired, not a reason to skip; never on panic or window close.

Durable process custody (`ouroboros/process_custody.py`): `spawn_supervised()` ledgers each long-lived child in `data/state/process_ledger.jsonl` (scope `task|session|daemon`). The reaper (startup, `server_maintenance` sweep) kills only entries whose generation or task owner died, by strict `(pid, start_time, cmd_sha256)` fingerprint, never command-line class, so dev and packaged installs coexist. Daemon rows stay; skill companions are reaped on owner uninstall or foreign generation (log-only default: `process_would_reap`). Worker tree-kills (`worker_pool_lifecycle.kill_worker_tree`) spare retained daemon subtrees. `start_parent_lifeline()` group-suicides a Python entrypoint on spawner death; multiprocessing children await the parent sentinel (a forkserver ppid outlives the supervisor). PID 1 adopts orphans too: ppid 1 at start is live only for a spawn/fork child of PID 1 or a plain child still under its `OUROBOROS_PARENT_PID`.

Launcher lifecycle: each generation resolves its predecessor, re-checks the automatic Panic gate and starts the server; ready is the endpoint this child published, proven against its handle, answering health. `RESTART_EXIT_CODE` requests a managed restart that refreshes bundle metadata and dependencies (§2); `MAX_CRASH_RESTARTS` crashes within `CRASH_WINDOW_SEC` stop automatic restart; a panic exit cleans up and ends the outer process. Presentation: the Linux browser fallback checks `DISPLAY`/`WAYLAND_DISPLAY` before touching pywebview; GTK needs a live default display, Qt is trusted on env alone: probing it constructs a `QGuiApplication` that can itself abort. On probe failure the same launcher supervises the same server, prints the authoritative URL and opens the system browser best-effort (the browser is the owner's application, outside process custody).

#### Android host (experimental)

`android/` is a platform of the same repository, Python core, SPA and update channels: the full core runs in an ARM64 GNU/Linux chroot on a Magisk-rooted device, with LLM APIs remote; physical qualification is limited to Pixel 10a. Installation, recovery and the bridge surface: `docs/ANDROID_INSTALL.md`.

`android/install.py` verifies the source-release archive, checks USB/root/ABI prerequisites and provisions `/data/local/ouroboros-phone/rootfs`; `android/provision/` pins upstream inputs and the installed dependency record. Inside Linux, `/opt/ouroboros/{repo,data,venv,tools,signing,launcher}` separates mutable source, personal state, tools, the personal APK key and the immutable seed. `bootstrap/enter-linux` creates private mounts while keeping Android's network namespace and sets its own `oom_score_adj` to 0, so root survives and the kernel can recover from OOM. `core-control` is a one-shot start/status entry, never another restart loop; it starts the common launcher with `--host-update` and `--seed-bundle` (seed and restart handoff: §2 "Packaged seed and managed checkout"); automatic entry preserves an existing Panic marker.

`bootstrap/update-host` is the native update hook: it builds the APK with `host/build.py` and the installation's persistent `signing/host.keystore`, records the `data/state/android_host.json` receipt, serializes under `android_host.lock` and retains candidates in `data/android-builds/`. Key creation belongs to provisioning only; loss means restoring that identity, never silent replacement. First the hook calls `provision/runtime.py::ensure_platform`, which marks dependency groups `platform_preparing` in `android-sdk/installation.json` before mutating them and clears the mark only after success, so an interrupted run or a rolled-back Git tree never reads as prepared. `launcher_bootstrap.update_external_host` bounds the hook with its own `runtime_limits.py` timeout, independent of tool and harness timeouts; the launcher's HTTP readiness window starts only after the core is spawned; core health alone claims no native success. Failed or stale native evidence leaves evolution adoption unresolved; ordinary startup keeps the available core and logs `native_update_failed`. Delegated access follows the common contracts (§6); a requested access profile does not prove the kernel can execute its sandbox, so the observed route is qualified.

The APK's `MainActivity` hosts the SPA and declares `ACTION_ASSIST` with an in-app `RoleManager` consent flow (an Activity entry, not a `VoiceInteractionService`). `CoreService` and `BootReceiver` live in the separate `:native` process at the same app UID, so a WebView renderer replacement does not own the SDK bridge; `BOOT_COMPLETED` uses automatic core entry after unlock; there is no second core supervisor. `AndroidBridge` exposes typed, bounded operations (packages, Intents, ContentResolver, location, opt-in accessibility and notifications) over a root/host-UID socket; `android-exec` keeps arbitrary owner-root argv capability. The rootfs under `/data/local` is readable by an authorized root before unlock, not credential-encrypted storage. The UI uses loopback HTTP; the network security config permits cleartext only for `localhost`/`127.0.0.1`. Distribution is GitHub sideloading with `QUERY_ALL_PACKAGES` retained; a locally built, personally signed APK is outside the publisher-artifact attestation.

#### Platform substrate

`platform_layer.py` owns OS observations and lock/process primitives; callers own policy. `kernel_file_locks_enforced` probes each real directory once, never from a failed live acquisition: unsupported-lock errors and `ENOLCK` select a recorded name-only tier, an unprobeable directory stays enforced for that call, other kernel errors fail closed (else: `platform_layer.py`). The name tier uses O_EXCL plus identity recheck without kernel exclusion; callers may refuse it (money does). A won lock must still have a readable descriptor and path inode identity; unreadable identity is no proof. Owner-aware recovery skips the stale-age grace after confirmed owner death and retains live writers; unknown metadata keeps the grace; permission denial proves no death. Refresh returns ownership, not a courtesy heartbeat: losing it means abandoning the protected work. Release unlinks only the held identity (ordering, the bounded unlink retry and the `LockFileEx` byte: `platform_layer.py`).

Birth identity is separate from PID presence. Linux mints boot-qualified ticks, then the ps wall-clock token, then bare ticks with a disclosed cross-boot collision limit; the custody ledger keeps the `start_time` spelling and an optional boot-qualified sibling: an older reader meeting an unknown token after rollback would prune every row without a kill. Windows uses the exact creation FILETIME, checks presence through non-signalling handles and treats access-denied or unknown presence as no licence to clean up; Win32 calls declare full-width HANDLE arguments (omitted declarations truncate 64-bit handles), and Job termination treats a false BOOL return as failure, so survivors stay disclosed. Bundled resources use the CLI / Headless Boundary lookup order; `node_runtime` owns Node health and selection (re-exported lazily by the platform layer). TCP keepalive: §6 Context fitting, retry, and compaction; kernel dead-peer detection never shortens a cognitive deadline.

### Data layout (`~/Ouroboros/`)

`~/Ouroboros/` is the default application root (overrides: §7). This tree is the orientation carrier; the complete per-entity registry of the data plane (writer, format, schema, retention and reset for every durable file) is `docs/PERSISTENCE.md`.

```
~/Ouroboros/
├── repo/                          ← the self-modifying git repository (§2)
│   ├── ouroboros/                 ← core package (module map above)
│   ├── supervisor/                ← supervisor package
│   ├── web/                       ← Web UI; ES modules under web/modules/
│   ├── docs/                      ← reference books, CHECKLISTS.md, subsystem contracts (PERSISTENCE.md, USAGE_STORE.md, …), install guides, inventories/
│   └── prompts/                   ← SYSTEM.md, SAFETY.md, CONSCIOUSNESS.md
├── data/                          ← runtime data root
│   ├── settings.json              ← owner settings, keys, models, budget (§7)
│   ├── task_results/              ← `<id>.json`: import input, export output (the store row is the authority); artifacts/<task_id>/ with .artifact_manifest.json
│   │   └── artifact_versions/<task_id>/ ← bounded artifact recovery history
│   ├── task_drives/<task_id>/     ← task-scoped scratch and per-call manifests
│   ├── task_trees/<root>/blackboard.jsonl ← swarm blackboard; pruned at root terminal
│   ├── state/
│   │   ├── state.json             ← runtime state and cost projection; never the monetary authority
│   │   ├── queue_snapshot.json    ← PENDING/RUNNING recovery projection, `worker_pool_disabled_reason` (§5)
│   │   ├── current.sqlite         ← the current-facts authority (task rows, obligations, custody, handoffs, outbox); `.exported-<UTC>` retired copies
│   │   ├── usage.sqlite           ← the monetary authority (docs/USAGE_STORE.md)
│   │   ├── usage_attempts.jsonl   ← imported journal; historical audit and older-seed input
│   │   ├── usage_attempts.quarantine.jsonl ← the journal's proven-corrupt rows
│   │   ├── usage_import_watermark.json ← import watermark
│   │   ├── skill_review_root_tasks.jsonl ← derived skill-review index; warns at `SKILL_REVIEW_ROOT_TASKS_WARN_BYTES`
│   │   ├── request_wire_compatibility.json ← exact-route wire evidence
│   │   ├── capability_evidence.json ← sourced model-capability evidence
│   │   ├── extra-ca-bundle/<digest>.pem ← content-addressed CA bundle every first-party HTTP client verifies against
│   │   ├── process_ledger.jsonl   ← process-custody ledger (Runtime topology)
│   │   ├── installation_scope.json ← scope marker
│   │   ├── task_scopes/ ← task scope records
│   │   ├── panic_sweep_pending.json ← pending Panic sweep
│   │   ├── obligations/           ← the old body's set files: import input, export output
│   │   ├── migrations.json        ← completed schema generations, dependency fingerprint
│   │   ├── body_adoption/         ← the one pending body adoption (handoff, captured switch.py, lock)
│   │   ├── owned_processes.json   ← ownership set with pending stops (§9)
│   │   ├── server.lock            ← server-lifetime DATA admission; never unlinked
│   │   ├── server_port            ← active HTTP port for launcher/browser handoff
│   │   ├── server_port.bindings.json ← informational endpoint snapshot, never a grant or custody ledger
│   │   ├── server_process.json    ← launcher-owned server identity for relaunch cleanup
│   │   ├── advisory_review.json   ← commit-review state: attempts, obligations, commit-readiness debts; old advisory runs read-only
│   │   ├── code_intel/<repo_key>/inventory.json ← code-inventory facts; no source cache
│   │   ├── evolution_metrics_cache.json ← per-tag metrics cache
│   │   ├── evolution_campaign.json ← campaign objective, progress, budget
│   │   ├── evolution_checkpoints.jsonl ← per-cycle checkpoints
│   │   ├── post_task_evolution_request.json ← one-shot promotion signal for the supervisor idle tick
│   │   ├── post_task_evolution_counter.json ← per-drive every_n counter
│   │   ├── scheduled_tasks.json   ← cron/one-shot schedules and `kind:"notify"` notes (§5)
│   │   ├── claudexor_rotation_provisioning.json ← last rotation-reconcile receipt
│   │   ├── subagent_last_delegation.json ← dated helper observations; never dispatch authority
│   │   ├── update_letter.json     ← update letter with `last_good`
│   │   ├── projects.json          ← Project registry; tombstones are durable
│   │   ├── projects.json.committed ← commit witness: beside it a missing registry is unavailable, not empty
│   │   ├── project_task_bindings.json ← root↔Project bindings with typed origin
│   │   ├── ui_preferences.json    ← owner-local layout prefs and seen-revision ACKs
│   │   ├── i18n/<tag>.json        ← translation memory per language, beside `<tag>.pending.json`
│   │   ├── cancel_intents.json    ← projection of active cancel intents
│   │   ├── terminal_deliveries.json ← delivery dedupe, byte receipts, pending outbox
│   │   ├── extension_companions.json ← live companion processes
│   │   ├── extension_reconcile/   ← worker-written markers for the server pickup task
│   │   ├── review_continuations/  ← blocked-review continuations (+ corrupt/, archived/)
│   │   ├── review_migrations/     ← `<ts>-slots-to-pool.json` per review-lane → review-pool migration; the rollback source, never deleted
│   │   ├── workspace_executor_processes/ ← executor cleanup records
│   │   ├── headless_tasks/<task_id>/data ← forked/empty child drives; per-call manifests promote at terminal
│   │   ├── custody_staging/       ← unserved copies prepared by a drive settlement
│   │   ├── custody_trash/         ← fully custodied drives awaiting deletion
│   │   ├── betterleaks/           ← scanner runtime, created only by the explicit installer
│   │   ├── cx/                    ← managed Claudexor store: immutable `<version>-<sha12>/` trees, install.lock
│   │   └── skills/<name>/         ← per-skill state (§13): review.json, owner_attestation.json, enabled.json, deps.json, health.json, extension_calls/, __extension_imports/
│   ├── claudexor/                 ← Ouroboros-owned Claudexor home (`CLAUDEXOR_CONFIG_DIR`)
│   ├── memory/
│   │   ├── identity.md            ← durable identity
│   │   ├── scratchpad.md          ← rendered from scratchpad_blocks.json, bounded
│   │   ├── dialogue_blocks.json   ← chronicle import source, with dialogue_summary.md
│   │   ├── dialogue_meta.json     ← chronicle import cursor and nomination source
│   │   ├── chronicle/             ← records.jsonl (append-only authority), index.sqlite3 (disposable)
│   │   ├── WORLD.md               ← host profile from first run
│   │   ├── knowledge/             ← topic files, patterns.md (Pattern Register), improvement-backlog.md
│   │   ├── deep_review.md         ← deep-self-review output
│   │   ├── registry.md            ← memory awareness map
│   │   └── owner_mailbox/         ← per-task owner messages
│   ├── projects/<id>/knowledge/   ← per-project facts with provenance sidecars
│   ├── observability/             ← private forensic ledger: blobs/<sha256>.json.gz, calls/<task_id>/<call_id>.json, packs/<task>.xz once settled
│   ├── services/<task_id>/<service>.log ← service runner logs
│   ├── logs/
│   │   ├── chat.jsonl             ← canonical chat, stored once, projected into Main/Project lenses
│   │   ├── chat_annotations.jsonl ← routing status by client_message_id
│   │   ├── progress.jsonl         ← progress/thinking stream
│   │   ├── events.jsonl           ← lifecycle, llm_round/llm_usage, errors
│   │   ├── tools.jsonl            ← tool calls
│   │   ├── supervisor.jsonl       ← workers/supervisor, cancel trail
│   │   └── task_reflections.jsonl ← reflection log: earlier automatic reflections, late-settlement rows
│   ├── archive/                   ← rotated logs, rescue snapshots, archived repos; never GC'd
│   └── uploads/                   ← chat attachments
├── projects/                      ← genesis subagent projects (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`), never GC-pruned
├── subagent_worktrees/            ← self-worktree checkouts (`OUROBOROS_SUBAGENT_WORKTREE_ROOT`), GC-pruned
├── Deliverables/                  ← bare user_files filenames (`OUROBOROS_DELIVERABLES_ROOT`), never GC-pruned
└── ouroboros.pid                  ← launcher PID lock; auto-released on crash
```

The generated `docs/inventories/DATA_LAYOUT_INVENTORY.md` probes each entry's last literal segment against tracked paths and runtime literals, so a durable file renamed in code turns its row red; it proves nothing stronger.

---
