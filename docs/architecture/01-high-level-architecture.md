# 1. High-Level Architecture

Modules, devtools, gateway/CLI, host and data. Launcher recovery needs an importable shell (§2). The server admits, routes, observes and kills, doing work bounded by logical change; every focus of the mind runs in a custodied worker (§5 Conversation lane).

```
User → launcher.py (desktop/native) — lifecycle over packaged seed or managed checkout (Runtime topology; §2)
launcher.py spawns server.py (Starlette+uvicorn) — HTTP/WS server; bind via `OUROBOROS_SERVER_HOST` (§4; §7)
web/ — build-free ES-module SPA (§3, plus unlisted modules/pages)
  ui.css — shared palette/control stylesheet (SPA, onboarding, author pages) (§3 Navigation)
  modules/ — browser modules (§3)
    media_capture.js — shared audio/visual leases, device replacement, release
    speech_providers.js, voice_recorder.js, dictation.js — speech hub, Opus/AAC/WAV recording, cancellable transcription, draft insert
    conversation_controls.js — mounts Live Conversation contribution at composer (§13)
    voice_controls.js, dictation_chooser.js, read_aloud.js — dictation mic, "+" → Add dictation…, Read aloud (DESIGN "Voice controls")
    ui_primitives.js — safe fields, escaping, tone/status helpers (§3 Navigation)
    page_header.js, ui_interactions.js, scroll_fade.js — header/tabs/segmented-choice; dialog/menu/popup binders; scroll-edge fade (§3 Navigation)
    utils.js — shared utilities
    i18n.js, settings_language.js, ui_i18n_types.js — chrome-only translation overlay; chat prose, logs, code and owner names never walked (§3 Settings)
    chat_decision.js, question_presentation.js, secret_hitl.js, main_waiting.js, panic_recovery.js, chat_render_batch.js, task_phase_chip.js, task_activity_types.js — decision/question cards, secret forms, Panic recovery, keyed timeline patches, census types (§3)
    chat_history.js, chat_history_reveal.js, chat_history_replay.js, chat_preview.js, chat_reading_position.js, layout_batch.js, chat_row_sleep.js, chat_row_placement.js, room_archive.js — paging/replay, preview, anchors, batched reads, row sleep, child words vs notices, read-only archive (§3 Timeline ownership)
    delegated_activity.js — executor activity with preview gaps (§3; §6 Delegated activity)
    verification_presentation.js — failed-check rows, live and replayed (§6 Tool capability and execution)
    welcome_preference.js, host_notice_text.js — Main empty state (`welcome`); notices (`host_notice.py`), legacy rows sans pictograph (DESIGN)
    project_answer.js — Main Project lifecycle rows and folded mirrored answers (§3 Main rows)
    work_portal.js, task_episodes.js, project_handoff.js — Main work card: portal/sources, Continue episodes, binding receipts (§3 Project handoff receipts)
    chat_reply.js, chat_reply_quote.js, chat_search*.js, chat_relay_deck.js, chat_row_menu.js, row_menu_gesture.js, message_actions.js, routing_receipt.js, text_fit.js — replies/quotes, search, note decks, row menus/actions, routing receipts, word fit (§3; DESIGN "Replies")
    composer.js, composer_model.js, chat_model_routes.js, composer_subagents.js, capsule_fit.js, swarm_presentation.js — composer capsules, Swarm labels/receipts (§3)
    chat_cards.js, card_components.js, card_frames.js — living cards, sandboxed app frames (CLIENT_CONTRACT §5)
    contributions.js, composer_plus.js, composer_chips.js, host_modules.js, holds.js — skill contributions, skills' "+" items and capsule-row chips, host modules, start ledger/holds (CLIENT_CONTRACT §11; DESIGN "Skills in the page")
    project_reference.js, task_reference.js — sole Project and Main task doors (DESIGN)
    project_activity.js, work_status.js, work_types.js, status_mark.js — Project marks/room words from server `rooms`/`works`; typedefs (§3 Liveness census)
    room_work.js, project_read_state.js, chat_header.js, consciousness_status.js — room bar: work list (no execution authority), read receipt, Panic/Restart, consciousness chip/Shade (§3 Project rooms)
    model_wait.js — model-wait actions (§6)
    task_checkpoints.js, cancel_presentation.js, dashboard.js, logs.js, worker_sha_presentation.js, costs.js, cost_project_days.js, costs_widgets.js, files.js — checkpoints/stop causes, Dashboard, Logs, worker SHA, Costs (spend, room days, widget placements), Files (§3)
    extensions.js, section_levels.js, skills.js, skill_*.js — Extensions, Skills views/cards/publish (§3 Extensions and Widgets)
    widgets.js, widget_*.js, masonry.js — Widgets host/renderer/framed mounts/chrome/masonry (§3)
    settings_ui.js, settings_home.js, settings_rows.js, settings_provider_cards.js, settings_catalog.js, settings_controls.js, settings_policy.js, settings_secrets.js, settings_connections.js, connection_labels.js, settings_local_model.js, settings_autostart.js, mcp_settings.js, mcp_header_fields.js, mcp_import_dialog.js — Settings: catalog, binders, policy, secrets, keys, local models, MCP (§3)
    model_roles.js, model_chooser.js — the Models editor and editable chooser shared by Settings, onboarding and route editors; catalog arrival never assigns a value (§3 Navigation)
    subagents_settings.js, subagent_status_primitives.js, route_editor_primitives.js, harness_accounts.js, harness_account_rows.js, account_resources.js, harness_login_cards.js, claudexor_status_store.js — Agents & harness accounts (§3 Agent accounts)
    onboarding_agents_step.js, onboarding_starter_skills.js, onboarding_summary.js, onboarding_connection_card.js, qr_svg.js — first-run steps, connection card (QR) (§2)
    onboarding_overlay.js — wizard frame sandbox policy, one security boundary (§2)
    project_create.js — New Project dialog (§3 Project rooms)
    review_command.js — `/review` chooser/send (§3)
    review_presentation.js, review_opinions.js, plan_record_presentation.js, review_record_card.js, review_dom_patch.js, harness_presentation.js, executor_card_facts.js — read-side Review/plan cards, keyed patches; harness_presentation.js owns sole harness identity markup (§3)
    acceptance_incident_presentation.js — stored acceptance incident, read-only; no reviewer authority (§6)
skills/browser_speech/, skills/device_voice/, skills/templates/ — built-ins (`skills/builtin.json`): speech (ui/stt.js, ui/tts.js), starters
skills/live_conversation/, skills/gemini_live/, skills/grok_realtime/, skills/openai_realtime/ — Live Conversation & native providers (§13)
supervisor/ — background thread inside server.py (§5)
  active_activity.py — inline-executor actors, start phases, turn receipts (§3)
  message_bus.py — Web/transport-skill bus; re-exports ingress names (§10.3)
  message_ingress.py — one accepted row per message id, durable dispatch (§4)
  workers.py — pool: forkserver Linux, spawn elsewhere, never fork (why: its comment)
  worker_assignment.py, dispatch_choice.py, worker_chat_lane.py, worker_health.py, worker_pool_lifecycle.py, worker_process.py, worker_promotion.py — pool leaves, one concern each as named (§5; §6 Owner routing verbs)
  worker_owner_wait.py — required owner wait lends active slot; task stays RUNNING, custodied (§5)
  conversation_lane.py — lane slots, admission, spare, census (§5 Conversation lane)
  inline_carrier.py — the two inline executors (§5 Conversation lane)
  state.py, state_initialization.py — `state/state.json` updates (typed read quality); init witness (§2)
  queue.py — `supervisor.queue` SSOT (PENDING/RUNNING, activity timeouts)
  loop_load.py — measures supervisor queue/loop load (§5)
  queue_publication.py — marks under Q; tick publishes outside Q by revision (§5)
  queue_snapshot.py, queue_timeouts.py — re-exported by `supervisor.queue`: snapshot/restore; activity timeouts (§5)
  queue_transitions.py, task_ownership.py — root seal (`acceptance_fences`), Resume, evolution stop, fenced Project delete; ownership reads precede queue lock (§6)
  queue_schedules.py, schedule_lifecycle.py, followup_policy.py — re-exported by `supervisor.queue`: schedules, lifecycle, `followup_relation` (§5)
  schedule_time.py, schedule_occurrence.py — cron/tz parsing; at most one root per occurrence
  schedule_notes.py — due `kind:"notify"` → signed `reminder` System row (no model, no retry)
  schedule_view.py — pure schedule facts and digest
  work_stop_migration.py — one-time boot conversion of legacy stops by `work_stop_format` (no execution authority) (§5)
  restart_retention.py — stop retention and restart/update-abort return authority (§9)
  cognitive_operations.py — operation leases/wait marks (`wait_for_tasks` picks responsible parent); no durable ledger (§6)
  silence_attention.py — silence episodes and mail to source and one waiting ancestor (§5)
  task_model_wait.py — live model-wait projection with quota-clock reads (§6)
  task_admission.py — duplicate-ingress reservation fences before effects (§5)
  task_dispatch.py, workspace_preparation.py — admitted worker payload; off-loop workspace prep with durable phases (§5)
  task_lifecycle.py — cancellation settlement, `sweep_cancel_intents`, root-budget fence (§10.14)
  cancel_publication.py, ended_work_helpers.py — `CANCEL_*` settlement (artifact-honest fields); ended work's parked helpers close (§5)
  owner_stop.py — graceful stop: `finalize_now`, then cancel on same cancel intent (`OWNER_STOP_OUTER_CAP_SEC`)
  task_reaper.py — off-loop reaper (timeout never retries, crash, idle, unready; unconfirmed death keeps reaping; mints no cancel intents) (§5)
  budget_resume.py, budget_resume_unstarted.py — Resume grants (queued root through Pause fence), revocation, hold release (§6)
  sleep_wake.py — exact wake grants for ready sleeps/technical waits (§6)
  consciousness_carriers.py — settles own-clock wake from carrier terminal (§6)
  owner_pause_control.py — owner Pause fence, member wake, tree settlement (§6)
  continuation_admission.py, tree_census.py — Continue admission (replay first, predecessor claim); writer census memo (§6)
  terminal_delivery.py, terminal_cascade.py, terminal_supersession.py — dedupe, byte receipts, bounded outbox (`ouroboros/terminal_outbox.py`); terminal_cascade.py emits one owner message per settled cascade (§10)
  evolution_lifecycle.py, evolution_cleanup.py — campaign lifecycle, Git cleanup (§6)
  events.py, event_observation.py — worker→supervisor dispatch; idle-input index, `main_llm_call_state`
  subagent_task_truth.py — subagent `task_done` delegation-truth enrichment (§11.1)
  card_delivery.py — card chat row, `card`/`card_state`, `chat.card`, `chat.outbound` fallback (§3)
  telemetry_events.py — rare typed telemetry-only events (excludes high-rate narration)
  event_taxonomy.py, events_budget.py, events_chat_delivery.py, events_coop_checkpoint.py, events_evolution_done.py, events_mind_controls.py, events_project_routing.py, events_runtime_controls.py, events_schedule_task.py, events_subagent_admission.py, events_task_done.py, events_worker_reports.py — `EVENT_HANDLERS` leaves; `event_taxonomy.py` declares `EVENT_Q` dispositions; `events_project_routing.py` alone writes routing refusals (§5)
  log_addressing.py — task-log audience (project binding wins, `HIDDEN_CHAT_ID` = 0, A2A dropped at `push_log`; §4; §12)
  steering.py — host-minted `issuer` mailbox steering (refused during pending cancel; §6)
  direct_roots.py — retired: old file folded/removed at boot (§5)
  git_ops.py — Git ops (clone/checkout/rescue/rollback/push/cred helper), bounded local-Git runner (§2; §8)
  git_ops_remotes.py, git_ops_rescue.py, git_ops_reset.py, git_ops_updates.py — `git_ops.py` leaves: remotes/push, rescue snapshot (§2), checkout/reset admission, update status (§8)
  update_source.py — official update-source selection/network policy (§8)
  update_recovery.py — owner Restore to pinned prior HEAD, rescue before reset (§8)
  update_merge.py — managed-update engine: 3-way/assisted merge, write-ahead tx, rollback, boot recovery (DEVELOPMENT "Managed Update Rule")
  update_candidate.py — rerere-neutral merges, preservation branches, stash restore (§6)
  update_carriers.py, update_merge_plan.py — protected carriers; merge planning (§8)
  update_merge_policy.py — presentation-only conflict labels (§3)
ouroboros/ — agent core and shared runtime (§6)
  config.py, launch_roots.py — SSOT facade: paths, settings I/O, PID lock (§7)
  settings_defaults.py, settings_scales.py, model_slots.py, review_model_routes.py, runtime_limits.py — settings vocabularies `config.py` re-exports (§7; §10.3)
  version.py — VERSION string with metadata fallback
  deployment_policy.py, deployment_extensions.py, execution_domain.py, work_domain.py, work_skill_payloads.py — deployment settings, Safety, body, extensions, work domains (§7; §13)
  runtime_mode_policy.py — protected-path policy (§6 Safety and runtime mode)
  secret_masking.py — Settings/MCP wire placeholders and secret repair (§7)
  settings_integrity.py — task/skill settings views, `OUROBOROS_SETTINGS_SHA256` pin, `OUROBOROS_PROJECTED_*` echo/digests (§7)
  route_choice.py — live route keys, turn route choice (§7)
  credential_shapes.py — credential leaf names and physical locations (§6 Credential mutation and diagnostic redaction)
  update_channels.py — Stable/QA/Development mapping (§8)
  update_letter.py — update description: check facts, owner request, writer (§3)
  update_qualification.py, update_generation_facts.py, update_impact.py — merge/check evidence; before/after facts; Apply effects (§3)
  engine_pin_policy.py — pin rule shared by Apply and restart (§3; §9)
  evolution_history.py — Evolution version history (§3)
  colab_bootstrap.py — Colab source-mode bootstrap (Drive data, no-UI)
  cli.py — source/headless CLI over gateway (CLI / Headless Boundary)
  packaged_cli.py, packaged_cli_install.py, git_selection.py — packaged CLI bridge, shim installer, Git selection (§2)
  process_interpreters.py — Python/Node launch resolvers (§2; CLI / Headless Boundary)
  agent.py — task orchestrator; crashes project captured evidence or `loop_evidence_unavailable`, never invented counters (§6 Task lifecycle)
  agent_startup_checks.py — boot health (warning-only) and self-restart host adoption (§2)
  agent_task_pipeline.py — pipeline: result, artifacts, frozen cost, review lens, root-only post-work (§6 Task lifecycle)
  agent_dispatch.py, post_task_synthesis.py — delegated-child dispatch seam; post-task workers (§6 Post-task work)
  focus.py — authored focus with typed source ref (no dialogue/path escapes)
  startup_historical_audit.py — seal-audit child (own process group; §2; docs/MODEL_SEND_OBSERVABILITY.md); background children stop list
  maintenance.py, maintenance_jobs.py, maintenance_retirement.py, plan_collection_job.py, record_compression.py, saved_state_retirement.py, temp_sweep.py — `job:<name>` maintenance children (§5)
  task_finalization.py, swarm_rollup.py, swarm_receipt.py, swarm_receipt_sources.py, swarm_wait_facts.py — delivery finalization, fan-out facts, tree receipts (§6)
  mutation_attribution.py — root baseline, predecessor adoption, committed delta (§6 Git and commit review)
  post_task_checkpoint.py — root phase/cost, saved late-work Pause (§6)
  terminal_projection.py, terminal_time.py — cognition-free settled-result/Main projection (§3); per-attempt end time
  dialogue_provenance.py — exact transport-provenance rendering (§6)
  dialogue_words.py, dialogue_recall.py, dialogue_word_index.py — word predicate, `chat_history(words=…)`, disposable postings (§6)
  event_bus.py — skill-subscription bus (§13)
  event_usage.py, event_token_totals.py — token counters, offline totals (§11.4)
  evolution_checkpoints.py, evolution_fingerprint.py — append-only campaign checkpoints; canonical objective repeat fingerprint
  improvement_backlog.py — authored backlog full text for evolution; no host writer (§6 Post-task work)
  post_task_evolution.py — worker's durable promotion signal for supervisor idle tick (§6 Background consciousness and Evolution)
  loop.py — LLM tool loop, nudges, FINAL ANSWER candidate (§6 Task lifecycle)
  loop_llm_call.py — single-round call, live event, usage accounting
  transcript_prefix.py — prefix digests, origin breaks, signed-reasoning retirement (facts cross boundary at call time) (§6 Task lifecycle)
  loop_transport.py — transport-outage waits, provider-failure text (§6 Context fitting)
  loop_output_exhaustion.py — typed reply recovery (§6)
  loop_delivery.py, delivery_facts.py, loop_budget.py, loop_completion_schema.py, loop_forced_finalization.py, loop_messages.py, loop_model_call.py, primary_route_observation.py, loop_nudges.py, loop_round_limits.py, loop_routes.py — `loop.py` rails; `loop_forced_finalization` owns the forced call (§6 Task lifecycle)
  late_review_marks.py — reviews settled after author left, kept in room view until released (§6 Review stack)
  result_material.py — exact copies of task products (§6 Result review)
  owner_source.py, answer_facts.py — owner chat/quiz/mailbox membership and answer frames; action meaning belongs to Main (§6 Owner routing verbs)
  delivery_seal.py — root stop seal (re-exported from `loop`; stored names keep `acceptance_fences`) (§6 Result review)
  review_price.py — measured per-row review price from usage ledger (§7)
  task_pacing.py — round-money thresholds, reserve, `CostCeiling`, wire options (§6 Budget tracking)
  vision_routing.py, vision_image_limits.py, image_preparation.py — Main/VLM image routing, route limits, byte prep (§6 Vision and local image evidence)
  fallback_cooldown.py — per-process advisory 429 cooldown (`OUROBOROS_MODEL_FALLBACKS`)
  model_concurrency.py — install-wide fair route queue (`OUROBOROS_MODEL_MAX_CONCURRENCY`; §7)
  project_naming.py — admission/Continue/Light/lazy turn naming (§5; §7)
  ui_translation.py — single translation worker, Light batches, `language_needs_model` (§7)
  ui_language.py, i18n_memory.py — interface-language fact; `state/i18n/<tag>.json` memory: owner > imported > generated (§7; §10.29)
  loop_tool_execution.py, tool_call_log.py, late_tool_results.py — tool dispatch, invocation logs, late settlements
  deadline_utils.py — deadline parsing, transport-vs-logical wait seam
  observability.py, redaction.py, source_retention.py, shared_values.py — private CAS (gzip-1, hardlinks), redaction, source retention, `$shared` dedupe (§10; PERSISTENCE)
  evidence_packs.py, evidence_packer.py — per-task xz packs (§5); missing ≠ `evidence_corrupt` (PERSISTENCE)
  finalization_timing.py — `task_finalization_timing` (§6)
  model_send_seal.py, context_assembly_trace.py — `model-visible ⟺ logged` seal; proven source contributions (docs/MODEL_SEND_OBSERVABILITY.md)
  process_logging.py, browser_output.py — bounded `server.log`/`agent_stdout.log`; Playwright stderr joins buffer (stdout stays protocol)
  cancel_intents.py — sole ingress `request_cancel`, durable projection, claim fencing, fail-closed reads (§5; §10.14–15)
  owner_hurry.py — task-local `kind=hurry` latch, one line for running work, never a chat message (§5)
  question_answers.py, room_reader.py, work_reply.py — `question_answers.py` delivers accepted answers (asker, admitted successor, fresh head); `room_reader.py` holds shared reader facts for the room's sole live reader; `work_reply.py` routes an accepted App Reply to its reading work or source room (§5 Bridge intake)
  owner_quiz.py — canonical questions: first-winner, publication address, delivery debt, `owner_wait` reconcile (§3; §11.1)
  secret_requests.py — skill secret fields, form outcome (§3; §11.1)
  card_store.py — living cards (`show_card`): `state/cards`, presses, content-addressed apps, Widgets pins (§4)
  owner_wait.py — cognition serialization, native waits, acknowledged restart handoffs (§5)
  model_sleep.py, model_detach.py, sleep_custody.py, native_task_wait.py — warm/cold sleep, same-id `continue_in_background`, cold-park custody, native task wait (body acceptance) (§6)
  task_carrier.py — attempt origin (`kind`), carrier (`lane`, `executor`) (§5)
  budget_pause.py, tool_custody.py — exact pause/Resume; invocation futures, durable residuals (§6)
  owner_pause.py, control_actor.py — whole-tree Pause fence/launch gate/review episode; mind-control attribution (§6 Owner Pause)
  external_runs.py — task's delegated runs under one stop policy (§6 Owner Pause)
  review_pause.py, review_waiting.py, ouroboros/tools/review_collection.py — review custody, wakes, cursors; collection by reference (§6)
  working_checkpoint.py — rolling per-attempt state, delayed ACK, frozen recovery (§6 Saved working state)
  local_custody_repair.py, custody_owed.py — ended claims retire on proof; unwritten facts stay owed (§6 Owner Continue)
  owner_continue.py, schedule_continuation.py — Continue eligibility and sources (§6)
  routing_wait.py — receipt waits, shared routing-act projection (§3)
  outcomes.py — typed multi-axis outcome/objective authority (§10.1)
  outcome_receipt_store.py — verification-receipt store; zero-run writes `incomplete`/`unknown` only (zero-run "complete" is unverifiable self-report)
  depth_evidence.py — pure `requested_depth`/permitted/attempted/achieved projection
  _outcome_receipts.py, _outcome_tool_errors.py — canonical receipt identity (§10.16); tool-error vocabularies (re-exported by `outcomes.py`)
  code_intelligence.py, code_intelligence_architecture.py — derived inventory cache (no bodies); pinned domain/contract facts (Code navigation)
  code_navigation.py, code_occurrences.py, code_import_candidates.py, code_search_rg.py — `query_code` views/evidence/imports; gated ripgrep for `search_code` (Code navigation)
  pricing.py — exact-route catalog lookup, nullable estimates (§6 Budget tracking)
  usage_accounting.py, usage_admission.py, _usage_wait.py — physical attempts; global/root/group and review-wave admission; pre-send waits (§6 Budget tracking)
  physical_attempt_context.py — frozen context-fit facts (incl. memory view kind) per attempt (§6 Context fitting)
  _usage_rows.py, _usage_money.py, _usage_response.py, usage_ledger.py — row reducer, Decimal cash, usage normalizer, row rules/money lock (§6 Usage ledger substrate vs. accounting policy)
  _usage_cache_splits.py — process-local cache split; missing entry prices full cache write (§6)
  _usage_settlement.py — usage/price evidence projected outside ledger tx (§6 Budget tracking)
  usage_store.py, usage_journal.py, openrouter_cost.py — `state/usage.sqlite`, journal import/export, OpenRouter receipts (USAGE_STORE.md)
  usage_origin.py — usage scopes, immutable billing/cause attribution (USAGE_STORE.md) (§6 Budget tracking; USAGE_STORE.md)
  cost_projection.py — sole task-cost projection (`accounted_upper_bound_usd`, null ≠ $0.00) (§6 Budget tracking)
  synthesis_cost_text.py — task_summary cost fields
  store.py, store_capture.py, store_empty.py, store_journal.py, store_task_cards.py, store_timing.py, store_import.py, store_switch.py, install_identity.py — `state/current.sqlite` core (atomic `meta.commit_seq`), import/export/cutover (§2; §9; PERSISTENCE.md; USAGE_STORE.md)
  dialogue_store.py, dialogue_history_evidence.py, dialogue_retry.py, dialogue_import.py, dialogue_positions.py — `state/dialog.sqlite` `append`, ordinals, cutover/export, first-Project evidence (§3; PERSISTENCE.md)
  storage.py, storage_deleted.py, storage_free.py, storage_free_view.py, storage_removers.py, payload_release.py, payload_release_state.py — Storage counts, owner Free, release receipts (§3; PERSISTENCE)
  llm.py — multi-provider LLM facade (function-shaped conversations; exact-route adaptation in request-wire leaves; §7)
  llm_routing.py, llm_attempt.py, llm_messages.py, llm_capability_policy.py, llm_fallback.py, llm_pricing.py, llm_openai_compatible.py, openai_responses.py, opencode_go.py, llm_anthropic.py, llm_gigachat.py, llm_local.py, llm_claudexor.py, llm_substitution.py — `llm.py` leaves: routes, attempt/cache, shaping, recovery, prices, wire formats (Go via compatible Chat), substitution (§6 Caller-owned subscription model calls)
  llm_helper_window.py — helper windows/replies (§6)
  server_window.py, size_refusal.py, window_learning.py — limits, refusals, retry schemas (§6)
  llm_stream.py — SSE assembly inside one attempt (§6 Streams and transport waits)
  send_clock.py — Main prep clocks, candidate binding
  transport_custody.py — physical-attempt transport facts
  openrouter_attribution.py — OpenRouter app identity (§7)
  llm_observability.py — public call projections (private sidecars stripped)
  llm_probe.py — non-generating Provider Test (no retry/fallback/learning; §3)
  net_transport.py — shared httpx transport: TCP keepalive, `OUROBOROS_EXTRA_CA_BUNDLE` every first-party client trusts (§7)
  model_wait.py — task/phase-bound quota/auth waits (§6 Quota and auth waits)
  owner_model.py, lead_fact.py — `owner_model.py` stores durable work-local primary/processing choices and sends a text-free wake to the focus; `lead_fact.py` builds and publishes factual acting `work_lead` across loop, census and routing manifest (§6)
  openai_chat_custom.py, openai_chat_dispatch.py — direct-OpenAI Chat codec and dispatch: custom first, exact-dialect fallback, task-local `none` last
  request_wire_contract.py, request_wire_resolution.py, request_wire_receipts.py, request_wire_attempt.py, request_wire_custom_validation.py, request_wire_recovery.py, request_wire_temperature.py, effort_evidence.py — exact-route wire compatibility, effort evidence (§10.13)
  anthropic_native_custody.py, openai_responses_custody.py — same-route replay custody (whole Anthropic reasoning blocks, whole Responses turns)
  reasoning_artifacts.py — fail-closed sealed-vs-portable reasoning (`SIGNED_PORTABLE` decays until re-probed)
  merge_receipts.py — task-owned PR merge intent, readback, receipt (§6)
  upgrade_notices.py, notice_receipts.py — once-only owner notices with chat receipts (§7)
  local_model.py, local_model_autostart.py — llama-cpp loopback lifecycle/autostart (§3)
  mcp_client.py, mcp_discovery.py — MCP identity/masks/launch admission; manager-owned joinable listing/settle (§6 MCP)
  mcp_schema.py, mcp_catalog.py — literal-safe metadata; catalogs
  mcp_oauth.py, oauth_engine.py — MCP sign-in; shared OAuth engine
  mcp_headers.py, mcp_import.py — origin-bound headers; draft import
  mcp_task_session.py, mcp_browser_guard.py — task stdio sessions; browser policy
  skill_mcp.py — skill-declared MCP binding
  safety.py — Safety Supervisor call/policy (§6 Safety Supervisor outcomes)
  safety_config.py — work/consciousness coverage, checker, migration (§7)
  safety_records.py, safety_evidence.py — verdict accounting, `tell_owner_over_flag`; evidence diet, row authorship
  ouroboros/gateway/safety_settings.py — owner Safety level/rules/checker endpoints (§4; §7)
  consciousness.py, consciousness_wake.py, consciousness_standing.py, consciousness_transitions.py, consciousness_view.py, consciousness_clock.py, own_time_seed.py — background alarm, wake input, what stands in it, accepted transition chain, `wake_task_metadata`, read-only view, named times and zones, `wake`-claiming seed (§6 Background consciousness)
  consciousness_authority.py, consciousness_allowance.py — autonomy ceilings (`observe`/`act`/`full`), rolling 24h allowance (§6 Background consciousness and Evolution)
  chat_chain.py, chat_references.py, chronicle_store.py, chronicle_quotes.py — addressed reads (legacy `(gen, line)`, Reply facts, `retain_memory_source`); append-only chronicle (§6)
  mark_anchors.py — tool/path-anchored marks (once per task) + live room/global updates (§6)
  chronicle_import.py, memory_nomination_receipts.py, memory_inventory.py — `legacy` model-free import; frozen cursor/nominations; open/folded inventory (§6)
  consolidator.py, memory_fallback.py — Light transport; one signed draft per root when wakes OFF (no knowledge writes) (§6)
  memory.py — scratchpad, identity, chat history
  knowledge.py, knowledge_edits.py, knowledge_versions.py, knowledge_rows.py, held_notes.py, text_change.py, memory_publication.py, memory_texts.py — Markdown notes, revision-checked writes (history row, then atomic replace); held_notes.py captures claimed/linked global heads once per work
  project_facts.py — `project_id` and `projects/<id>/knowledge` isolated from `memory/knowledge`
  task_tree_ledger.py — `data/task_trees/<root>/blackboard.jsonl` (`tree_note`/`tree_read`), mirrored at root end
  projects_registry.py, project_registry_recovery.py, project_admission.py, project_closure.py, project_folder.py — Project lifecycle, exact recovery, admission, reversible closure, folder door (§6 Project registry and lease)
  project_handoff.py — Main transfer receipt via terminal outbox (§3)
  project_dialogue.py, chat_annotations.py — chat lens/refusals; append-only routing receipts/picker claims (§3)
  room_membership.py, room_appearances.py, room_messages.py, room_archive.py — cross-room membership, derived facts, read-only history (§3)
  owner_words.py — dated verbatim owner words by value; schedule text as provenance (§10.26)
  project_lease.py — one-writer Project/folder lanes per tree (§6 Project registry and lease)
  context.py, context_timing.py — Main context, timing, subagent catalog (§6)
  context_input_selection.py — declared inputs, historical author views (not criteria), overflow rescue (§6 Selected first-input sources; Context fitting)
  main_context_authority.py — Main authority view, helpers' idempotent predecessor briefs (§6)
  client_surface.py — bounded client-surface normalizer; identity comparison (`client_surface_identity`) excludes viewport/`narrow_layout` (§4)
  context_fit.py — immutable Max/Low/Nano views, reclaim deficits, cache seal (no routing/retry authority) (§6 Context fitting)
  context_measurement.py — wire price/reply in route units (§6 Context fitting)
  context_budget.py — budget vocabulary, reclaim SSOT, `estimate_message_chars` (§6 Context fitting)
  context_mode_compat.py — normalizes/persists `OUROBOROS_CONTEXT_MODE`/`OUROBOROS_CONTEXT_MODE_AUTO_LOW` (§7 Default settings)
  context_layout.py — doc-layout SSOT; reduction relocates (§6 Context fitting)
  context_runtime_facts.py, engine_facts.py — runtime-section fact builders
  memory_view.py, memory_view_legacy.py, memory_view_account.py, memory_room_activity.py, memory_reading.py, memory_coverage.py, memory_waiting.py — story, legacy retellings, cross-room accounts, room considered points, reads, waiting acts (§6)
  memory_reader.py — extensions' `read_memory` (§6)
  memory_view_conversations.py, memory_view_conversation_forms.py — other conversations capture/forms (§6 Durable memory and project focus)
  memory_view_work.py — completion gists, work forms, source readers (§6)
  text_sentences.py — sentence boundaries (§6)
  ouroboros/gateway/completion_gist.py — short UI projections of complete completion gists (§6)
  memory_view_events.py — place event facts, room catalogue (§6)
  memory_floor.py, memory_forms.py, return_facts.py, memory_form_facts.py — distance-ordered fit, facts at human's word, form receipts/sources holding room (§6)
  memory_fold_receipt.py — fold receipts (§6)
  chronicle_lead.py — lead instruction and tool-description delivery evidence (§6)
  capability_evidence.py, response_limits.py, density_probe.py — sourced window/density/max-response evidence (`data/state/capability_evidence.json`; windows grant no review authority), cold-start density probe (§6 Prompt size, density and windows)
  reference_books.py — ordered book reader/validator; worktree balance vs cached merge-base (unknown if unavailable, nonblocking; DEVELOPMENT "Documentation contract")
  local_model_server.py — read-only local formatter/server probe
  context_compaction.py, context_source_view.py, context_renewal.py, context_health.py, arrival_notes.py — atomic compaction (transactional apply), source recovery, same-task renewal, health (own delegated obligations full, foreign work one exact-source pointer), dated arrival notes (§6 Context fitting)
  headless.py, history_retention.py — child-drive isolation, adoption, patches, memory export, typed `sensitive_blocked` (§10)
  task_custody.py — sole child-drive deletion owner (`settle_child_drive`), custody lock (§6 Headless finalization)
  owner_keep.py — task Keep (§6 Headless finalization)
  review_thread_retention.py — archived reviewer threads, gated release (§6 Headless finalization)
  headless_status.py — shared headless lifecycle states
  workspace_patch_rules.py, workspace_patch_capture.py, workspace_capture_result.py — patch exclusions, patch/manifest I/O
  workspace_copies.py — git-copy baseline identity, own-body policy (§6 Delegated subagents)
  coop_checkpoint.py — quiescent checkpoint commits from mutative child's `write_root` (root mid merge/rebase/cherry-pick/revert skipped; §5)
  delegate_custody.py, delegate_custody_reconcile.py, delegate_collection.py, delegate_state_sweep.py, delegate_custody_usage.py, delegate_capture_attempt.py, delegate_capture_job.py, delegate_named_evidence.py — run custody, collection, capture, usage (OWNED/FOREIGN/UNKNOWN) (§6 Nanny, transport and custody)
  delegate_custody_current.py — SQLite custody: run rows, pending invocations, outbox, importer fold
  delegate_start_claims.py — serializes custody recheck/`START_REQUESTED` outside waits (retries never hold root/store locks)
  delegate_hold.py — unknown-provider nanny hold (§6 Context fitting)
  delegate_source_coverage.py — oversized work-order custody; incomplete source blocks PASS/apply
  delegate_evidence.py — read-side execution evidence (§11.1)
  delegate_output.py — retained terminal JSON with physical-task reads (§6 Terminal products)
  delegate_directory.py, workspace_file_outputs.py — directory apply/discard, folder results with before/after identities (§6 Terminal products)
  delegate_activity.py, delegate_feed.py — typed activity/cursor; body feed (§6 Delegated activity)
  delegate_containment.py — engine-derived isolation facts (§6 Access disclosure)
  delegate_progress.py — current run views, streamed activity (§6 Delegated subagents)
  nanny_pacing.py — round/cost anchors (no reset on coordination) (§6 Supervision)
  delegate_interactions.py, delegate_shared.py — live nanny input (`waiting_on_user`, validated `_delegate_answer`, capability-gated `_delegate_message`); `_owned_run` keeps live control with starter, confirmed successors read/dispose settled products (§6 Delegated subagents)
  route_spec.py — neutral route/effort normalization
  configured_subagents.py, subagent_runtime.py — `OUROBOROS_SUBAGENTS` SSOT; task-start snapshots, exact `subagent_id` selection (§6 Delegated subagents)
  subagent_route_health.py, claudexor_observations.py, subagent_inspection.py, subscription_census.py — route health, passive observations, actor inspection, census (PASSIVE_QUOTA_READ)
  subagent_work_order.py, subagent_bootstrap.py, delegate_start_instructions.py — owner-word-preserving work orders, host pre-start, stable instructions (coordination appendix hashed apart) (§6 Delegated subagents)
  delegate_supervision.py — nanny sleep and owed wakes (§6 Supervision, activity and recovery)
  delegate_recovery.py, delegate_continuation.py, delegate_pending.py, delegate_accepted.py, delegate_resume.py — recovery, `continue_from`, replay/lookup Resume (§6 Delegated subagents)
  delegate_registration_policy.py, delegate_workspace_hold.py, delegate_result_taken.py, historical_retirement.py, delegate_readonly_inputs.py, delegate_git_range.py, delegate_documents.py, helper_receipt.py — holds, retirement, readonly inputs/ranges/docs, receipts (§6)
  delegate_terminal.py, owner_stop_receipt.py — terminal reconciliation, audit-only custody, custody cards; Wrap up receipt (§6 Delegated subagents)
  subagent_dispatch_notes.py — executor-note exports (`agent.py`/`agent_dispatch.py`)
  subagent_messages.py, subagents.py, subagent_history.py, subagent_metrics.py, subagent_metric_events.py — child messages, dispatch, dated measurements (no admission policy) (§6)
  subagent_worktrees.py, worktree_removal.py — private-copy primitive (`copies.noindex/` under worktree root), `state/private_copies/` custody records, `refs/ouroboros/delegated*/` pins, GC (PERSISTENCE), removal retries (§6 Delegated subagents)
  copy_capture_proof.py, private_copy_read.py, copy_environments.py — retirement proof; passive reads; `<copy>.env` folders (§6 Headless finalization)
  genesis_project.py — standalone genesis project (never age-pruned) (§6)
  force_remove.py, unindexed.py — permission-granting `force_rmtree`; `.noindex` names
  body_candidate.py, body_adoption.py, body_switch.py — own-body authoring and restart-bound adoption before body imports (§6 Own-body candidates; §2)
  artifacts.py, artifact_storage.py, artifact_directories.py — `artifact_store/`, clones, ZIPs, `.scratch_manifest.json` hashes, output audit (§6)
  chat_uploads.py, confined_files.py — owner uploads (`data/uploads`, byte-proven kind); confined handle walks/opens (§4; §12)
  retention.py — GC retention dates (§6)
  workspace_preflight.py, session_scratch.py, session_scratch_allocation.py — preflight session facts, locked cache prep, scratch allocation/custody (§6)
  project_sources.py — attach/`init_git`/clone/`apply_folder` (no temp host copy, recorded `no_folder`)
  promotion_source.py — promoted-task source admission post-reservation (§5)
  workspace_admission.py — disjoint roots, Project binding, never system-repo fallback (CLI / Headless Boundary) (§6 Owner routing verbs; CLI / Headless Boundary below)
  deep_self_review.py — whole-system diagnostic review (default Main; §6 Deep self-review)
  review.py, code_shape_facts.py — shrink-only size ceilings, diff pulse (§6 Structural gates, Commit preflight)
  size_ratchet_manifest.py — data-only size-debt manifest (`scripts/regenerate_size_ratchet.py`)
  review_execution_projection.py — kinds `api` | `harness` | `native` (so owner tells retrieving from packet review on same model)
  preflight_runner.py, preflight_probe.py, preflight_diagnosis.py, preflight_reading.py, preflight_case_facts.py, preflight_node.py, preflight_receipts.py, test_evidence.py — Gate, probe, failure reading, proof (§6)
  test_selection.py, preflight_inventory.py, preflight_qualification.py, test_parallelism.py, test_environment.py, test_bytecode.py, safe_test_supervisor.py — scope, inventory, adoption, slots, warm, teardown (§6; §8)
  review_substrate.py — independent-seat coordinator (transport/parse/verdict/coverage/quorum distinct) (§6 Review stack, Result review)
  review_custody.py, review_source_closure.py, review_operation.py, review_live_join.py — no-resend custody, retained inputs, waits, live joins (§6 Review stack; Paid stamp and owner custody)
  review_owner_custody.py — paid attempts record `(server session, pid)` (§6 Paid stamp and owner custody)
  review_execution.py, review_session_preparation.py — review delivery (`delivery_retrieves(route, native_retrieval)`), session prep without transport fallback (§6)
  review_native_episode.py, review_native_images.py — read-only native API episodes, image staging (§6 Native tool-round episode)
  review_session_reads.py — harness-journal diagnostics (`harness_observed`), never verdict/quorum/retry authority (§6 Review delivery)
  review_verdict_extraction.py — strict parse then light-model extraction (`array`, `object`, `report` pass-through)
  review_session_custody.py, review_checking.py, review_checking_prune.py — pre-POST checkpoints, writable copies, prune (held attempts repaired first) (§6 Review delivery)
  review_slot_cancel.py — typed cancel outcomes reporting only proof (`ReviewSessionSucceededResultUnavailable`, never "may still be live")
  review_seat_outcome.py, review_wave.py — seat outcomes; detach latch, worker permits (§6)
  review_session_usage.py, review_thread_continuity.py — session usage attribution; thread continuity
  commit_admission.py — release/syntax admission, bound test proof (§6)
  reviewer_slot_config.py — sole builder `review_pool_slots` over enabled `review_eligible` rows (§7)
  review_pool_migration.py, review_pool_receipts.py — lane→pool migration and rollback receipts (`state/review_migrations/`; §11.4)
  review_run_isolation.py — pre-`config` contributor-review isolation (private data, cap, attach-only) (§6 Delegated subagents; Monetary authority and projections)
  review_state.py — commit-review state (`state/advisory_review.json`; old rows verbatim; re-exports review_state_model.py, review_state_records.py, review_state_custody.py)
  review_records.py, review_doctrine.py, review_verdict.py, review_projection.py, review_evidence_sections.py — panel records, reducers, redacted projections (old acceptance rows read tolerantly), evidence sections (§6 Result review; Review ledger record)
  review_ledger.py, review_continuation.py — per-wave records, explicit continuations (§6 Review ledger record)
  review_body_fact.py — own-body predicate and checklist layer (§6 Change review on any root)
  review_cycles.py — cap SSOT (`OUROBOROS_REVIEW_MAX_CYCLES`; §10.17)
  review_dispatch.py — row identities, write-ahead paid stamps (§6 Paid stamp and owner custody)
  reviewer_window.py — sizing-only per-route windows (unknown keeps disclosed assumption) (§6 Prompt size, density and windows)
  triad_review.py — JSON-array extraction, quorum, `REVIEW_JSON_ARRAY_CONTRACT` (clean `[]`), `review_output_shape(surface)`
  review_evidence.py — same-execution evidence (visibility only), task/checkout review record
  review_status_projection.py, review_overrides.py — status projection over `review_state` (via review_evidence.py); owner-visible trace of FAIL/no verdict (skills left on included) (§6 Own-body candidates)
  onboarding_wizard.py, subscription_install_presets.py — desktop/web onboarding; all-or-nothing sibling install compilers (§2)
  settings_setup_contract.py — setup SSOT, `resolve_total_budget_usd` (`TOTAL_BUDGET`; §2; §7)
  owner_relay.py — `tell_task` relays (`relay_entry`), `carry_to_members` (§10.26)
  owner_mailbox.py — per-task mailbox: revocation-aware drain, proven-empty peek
  peer_roster.py — live roots/Presence (hidden included), waits, `forward_to_worker` mailboxes (§12)
  plan_intention_ledger.py — ask `intention` record; never a verdict
  plan_record.py — per-task plan record (schema 3) (§6 Plan construction and review)
  provider_models.py, provider_connections.py — kind table, route grammar, connection rows and revisions; `ACTIVE_MODEL_SETTING_KEYS` excludes Heavy (§7)
  schedule_contract.py — 5-field cron/IANA tz SSOT (§5)
  repo_remotes.py — Git fetch URLs (`managed` read/update-only official vs `origin` personal; §8)
  semantic_dedup.py — LLM-first semantic dedup (fail-open None; DEV 02; unused)
  betterleaks_runtime.py — pinned Betterleaks across six platforms (§8)
  server_auth.py — non-localhost `OUROBOROS_NETWORK_PASSWORD` gate (warns unset), recall peer (§4; §8)
  server_control.py — server restart and panic stop (§4)
  server_entrypoint.py — CLI parsing, data-root admission (`admit_data_root`), port binding (§2; §4)
  server_runtime.py — startup wiring, WS liveness (§2; §4)
  server_web.py — `ServedUI` static/SPA mounts (§4)
  server_process.py, server_liveness.py, server_boot.py, server_maintenance.py, server_restart.py — composition-root leaves: process facts, boot document/wedge detection, boot steps (§2), upkeep, shutdown restart (§9)
  server_owner_routing.py, server_routing_context.py — owner-message routing, bounded per-turn facts (§5)
  terminal_cost_reconciliation.py — late terminal usage recovery (§6 Budget tracking)
  task_continuation.py — durable review continuation state
  task_results.py — atomic result merge-writes, `resolve_ctx_lineage` (§6 Task lifecycle)
  task_store.py, task_reads.py — indexed SQLite task views; capture in tx, decode after close (PERSISTENCE.md)
  pause_notices.py — confirmed-pause System notices (§5)
  task_result_schema.py — `_schema_version` admission; unstamped/future/malformed quarantined (§11.4)
  queue_observation.py — off-lock queue reader expanding `$shared`/heartbeats (§5)
  task_status.py — `execution_owner`/`execution_observation` (lifecycle vs liveness); worker cancel predicate fails open toward liveness (§5; §10.14)
  git_shell_policy.py — shell Git argv checks
  protected_artifacts.py — execute-only black-box policy for protected artifacts
  shell_parse.py — POSIX wrapper parser (observation, not permission) (§6 Safety and runtime mode)
  argv_budget.py — ARG_MAX byte admission (argv + env; checked by `skill_exec`)
  workspace_executor.py, workspace_executor_paths.py, docker_process_observer.py — executor backends/output paths; docker_process_observer.py needs `sh`, `cat`, `/proc` (§9)
  deliverables_paths.py — lexical/case-folded Deliverables views
  tool_capabilities.py — SSOT for core, inert-read, untruncated, browser, Presence-memory tool sets
  tool_access.py — ToolProfile × ResourceRoot × Operation matrix, closed-enum `required_capabilities`
  tool_access_types.py, tool_access_roots.py, tool_access_paths.py, tool_access_user_files.py, tool_access_reads.py — matrix types/roots; inherited reads, separate action authority (§6 Resource roots and physical file identity)
  tool_policy.py — round-one visibility, Nano standing set (others: tool_capabilities.py)
  capability_breakage.py — dated breaks: notice once at start, one tail row on change (PERSISTENCE)
  icon_geometry.py — bounded `currentColor` connector icons
  browser_policy.py — browser grants and service identity (§6 MCP)
  chromium_runtime.py — pinned Chromium supply/sandbox (§6)
  skill_payload_binding.py — `.seed-origin` targeting (read/list/search only for read profiles)
  utils.py — atomic/locked JSON, hashes, sanitization, subprocess, `truncate_review_artifact`
  jsonl_tail.py — `JsonlChainSnapshot` (swarm rollup), rotated log tails
  markdown_source.py — byte-preserving Markdown/YAML (SHA ranges; `MarkdownSourceError` keeps grammar/YAML errors visible)
  world_profiler.py — startup `WORLD.md` block refresh (rest is mind's map, `update_world`) (§6)
  obligations.py — atomic current sets (incl. `owner_questions`; published before work, retired after discharge; §10.10)
  startup_migrations.py, startup_task_files.py — inherited-state import/repair, addressed file recovery (§2)
  verified_download.py — digest-verified cached downloads
  node_runtime.py — Node health probes, pinned Node (§6)
  platform_layer.py, platform_win32_abi.py, platform_darwin_abi.py, platform_file_rename.py, platform_process_target.py, process_identity.py, host_memory.py — OS locks/processes/memory, rename, native ABI, birth tokens (Platform substrate)
  presence_admission.py, presence_context.py, presence_message_facts.py, presence_authority.py, presence_capabilities.py, presence_bindings.py — event→place resolution, FOCUS context/facts; frozen readers of old ceilings/bindings (§12)
  place_intake.py, place_reply.py, place_projection.py, places_migration.py, presence_runner.py, presence_continuation.py, presence_observations.py, presence_delivery.py, presence_history.py, presence_recovery.py, presence_owner_contact.py, presence_carrier.py, presence_executions.py — model-free attention, automatic reply Silent/Safety boundary, views/import, focus turn execution/recovery/waiters (§12)
  places.py, place_words_history.py, conversation_lease.py, own_sends.py — place/circle store (removed words kept), conversation leases, own sends (§12)
  tool_effects.py, place_guard.py — connector effects (read · message · gesture · write), send refusals (Advises, Silent/Paused, one author), destination notes/circle words beside sends (§12)
  extension_companion.py, extension_reconcile_queue.py — transport companions; worker→server reconcile markers (§13)
  extension_health.py — process-qualified skill health/companion failures (§12)
  call_peers.py, call_protocol.py — meeting PCM mixing, companion codec (§12)
  presence_call.py, presence_events.py — Presence call admission, shared event identity (§12)
  skill_token.py — opaque Host Service tokens (§12)
  service_sign_in.py, service_sign_in_declaration.py — declared-service sign-ins (§13)
  skill_loader.py, skill_catalogue.py — discovery (`OUROBOROS_SKILLS_REPO_PATH`), `data/state/skills/<name>/`, paged catalogue, delegate index (§13)
  skill_context_snapshot.py — passive tail skill facts, lease renewal (§6)
  skill_readiness.py — execution readiness (review/hash/enablement/grants/deps/conflicts) (§13)
  skill_peer_inventory.py, skill_conflicts.py — `SkillPeer` (reads payload only for required package), sole conflict verdict (§13)
  skill_dependencies.py — dependency resolution, package-readying seam (§13)
  skill_repair_admission.py — repair admission vs immutable `base_content_hash` (no long shell lock/rollback; §6)
  skill_owner_attestation.py — Skip review/Turn on anyway (exact bytes accepted without LLM review) (§13)
  skill_identity.py, skill_install_record.py, skill_acceptance.py, skill_grant_follow.py, skill_builtin.py, skill_discovery_reuse.py — release digest, install/acceptance, grants, built-ins, discovery reuse (§13)
  skill_publish_snapshot.py, skill_publish_scanner.py, skill_publish_result.py, skill_publish_github.py, skill_publish_client.py, skill_publish_eligibility.py, skill_publish_destination.py — Betterleaks-scanned PR publish leaves (§6)
  skill_review_status.py — review gate → `executable_review` (§11.1)
  skill_review.py — lifecycle verb: `review_change(subject="skill")` → `review.json` (§13)
  skill_review_packs.py, skill_review_output.py — frozen payload pack; review block (§13)
  skill_review_history.py — legacy history/ledger reader, saved exports (§13)
  skill_review_runner.py — `review_job.json`/`skill_review_*` events (separate review/dep/ext outcomes) (§13)
  extension_loader.py, extension_process_runner.py — in-process `PluginAPIImpl`/child proxies; child processes (scrubbed env, per-skill deps, timeouts) (§13)
  extension_route_stream.py — stdio ASGI stream for out-of-process routes (§3)
  extension_ui_validation.py, extension_widget_assets.py, extension_contributions.py — widget/contribution schema, asset capture, manifest staging (§3)
  extension_isolated_deps.py — non-reentrant `sys.path` leases (RLock serializes cache sweeps, not plugin execution) (§13)
  extension_plugin_api.py, extension_registry_state.py, extension_liveness.py, extension_child_catalog.py, extension_import_staging.py, extension_surface_names.py — extension runtime leaves (`PluginAPI`, registries, liveness, child catalog, staged imports, surface names) (§13)
  live_conversation.py — `LiveHost` actor port (context, tools, history, money, Stop) (§13)
  live_timing.py — renewal timing journal (§13)
  live_tool_lane.py — serialized tool-effect lane (§13)
  realtime_session.py — joined native socket teardown, JSON transport (§13)
  realtime_item_echo.py — item ACK comparison (images by type, rest exact) (§13)
  realtime_turn.py, realtime_setup.py, realtime_credentials.py — call route evidence, provider input, session credential capture (§13)
  skill_sources.py, skill_source_git.py, skill_source_catalog.py, skill_source_install.py, skill_starter.py, onboarding_connections.py — sources, caches, catalogs, installs, starter, setup cards (§13)
  skill_lifecycle_queue.py — single FIFO skill-mutation lane (§13)
  skill_lifecycle_actions.py — shared grant/toggle/turn-on-when-ready effects, admission (§13)
  skill_uninstall_state.py — uninstall tombstones, sole removal rule both removers admit by (§13)
  marketplace/ — ClawHub substrate (§13)
    clawhub.py, fetcher.py, adapter.py, install.py, install_specs.py, isolated_deps.py, provenance.py — registry, fetch, adapt, specs, isolated deps, receipts
  launcher_bootstrap.py, runtime_venv.py, launcher_onboarding.py — bootstrap/source handoff, runtime venv, desktop `/onboarding` (§2)
  launcher_readiness.py — `/api/health` probe (§2 Boot order)
  launcher_windows_runtime.py — Windows runtime, pre-import legacy GUI handoff (§2)
  launcher_server_reaper.py — strays, close/Panic/Job cleanup, Quit stop (§9)
  launcher_background.py — desktop consent, close/quit, `DesktopApi` (MainApi methods: §3)
  panic_source.py, platform_peer.py — diagnostic Panic source, peer lookup (§9)
  launcher_tray.py, launcher_tray_macos.py — Windows tray; macOS menu bar, Dock reopen, quit marking (§9)
  launcher_appearance.py, launcher_window_chrome.py, launcher_page_zoom.py, launcher_context_menu.py, standalone_page.py, launcher_context_report.py — chrome/zoom/menus, inline `theme.js`/`ui.css`, offline Windows report window (§3)
  desktop_media.py — GTK display probe, own-loopback audio grant (other shells keep native prompts; §8)
  desktop_notifications.py — native notifications with click tokens (DESIGN §9)
  desktop_autostart.py, windows_autostart.py — host sign-in adapters (Runtime topology)
  claudexor_runtime.py, claudexor_runtime_retention.py, claudexor_node_seed.py — reviewed next-spawn `data/state/cx` pin (seed/fetch, verify/probe, atomic promote; no mutable current; `OUROBOROS_CLAUDEXOR_BIN` overrides the binary, §6); unpinned non-serving copy retirement (§5); Node generations apart from tree identity, `reviewed_node`/`planned_node` seeds (§6)
  claudexor_exit_facts.py — saved exits, live/capacity/route-health facts (no lifecycle policy) (§9)
  claudexor_daemon.py — owned `data/claudexor`: lazy start, auth attach, `stop_outcome`, latch (§9)
  claudexor_startup_failure.py — stdlib `ExitFact` (`failed_without_control` drives latch; labels diagnostic-only) (§9)
  claudexor_restart.py — owner Restart Claudexor (§3)
  harness_maintenance.py — vendor maintenance, Connect install (§6)
  process_containment.py, process_lifeline.py, process_overload.py, command_memory.py — token/Job custody, POSIX guardian, pressure stop (§9; DEV §9), Linux cgroup memory
  process_scope.py, process_census.py — work scopes; kernel-state readers (§9)
  process_custody.py, launch_custody.py — `spawn_supervised`, ledger, `retained_purposes`; kept-service list/Stop, measured identity/exit (§9)
  owned_shutdown.py — ownership set, bounded exit stop, next-start retry, `state/shutdown_entry.json` (§9)
  contracts/ — Frozen ABI package (§11.1)
    tool_context.py — ToolContextProtocol
    tool_abi.py — ToolEntryProtocol, GetToolsProtocol
    plugin_api.py — `PluginAPI`, ExtensionRegistrationError, `FORBIDDEN_SKILL_SETTINGS`, `VALID_EXTENSION_PERMISSIONS`, `VALID_EXTENSION_ROUTE_METHODS` (§11.1)
    chat_id_policy.py — human-visible vs synthetic chat IDs SSOT (§12)
    task_contract.py — task-contract normalization, acceptance-claim binding (semantics: §11.1)
    task_constraint.py — `VALID_WRITE_SURFACES`, fail-closed surface/write_root check
    skill_payload_policy.py — payload confinement/sidecars
    skill_manifest.py — unified manifest parser (`VALID_SKILL_TYPES`)
    live_conversation.py — native session events, media, host input (§13)
    schema_versions.py — opt-in `_schema_version` (§11.2)
    record_contract.py — JSONL ledger passport (§11.1)
  gateways/ — Thin outbound adapters; no business logic (§6)
    claudexor_images.py — `requestKind=image` upload/ACK (docs/IMAGE_GENERATION.md)
    image_api.py — image API (one send, bounded results, cost evidence)
    search_api.py — `web_search` wire
    claudexor.py, claudexor_uploads.py, claudexor_threads.py, claudexor_identity.py — `discover_daemon_at` (`daemon/control-api.json`)/handshake/runs/quota; uploads; threads/export/retirement; final-attempt proof (§6)
    claudexor_metadata.py — handshake/catalog cache (`storeId`)
    claudexor_run_events.py, claudexor_observation.py — run-journal SSE/durable seq cursors (no polling owner); run observation adapter (§6 Delegated subagents)
    claudexor_routing.py — passive account/model marks (§3)
    claudexor_route_health.py — passive route failure shares (§9)
    claudexor_live_runs.py — passive live runs (§3)
  gateway/ — Gateway Boundary v1 (below): inbound route ownership and frontend contract SSOT (§4)
    contracts.py, reference_contracts.py — WS/HTTP envelope owner; stored-row identity (`row_address`, `reply_to`)
    decision_contracts.py, history_contracts.py, attachment_contracts.py, schedule_contracts.py, health_contracts.py, card_contracts.py, update_contracts.py — ingress-validated leaves re-exported by contracts.py
    ui_i18n_contracts.py, model_route_contracts.py, mcp_import_contracts.py, skill_mcp_contracts.py, client_contracts.py, management_contracts.py — browser-twin envelopes (§11.1)
    endpoint_index.py — `HTTP_ENDPOINTS`/`HTTP_SHAPES`/`WS_SHAPES` (routers own Route objects)
    schema.py — JSON Schema ingress validation; `contract_document` (`GET /api/contract`; §11)
    router.py — Starlette `/api/*`+`/ws` collector (§4)
    ws.py — WS manager, extension WS dispatch, broadcast (§4 WebSocket protocol)
    call_ws.py, host_call.py, call_contracts.py — owner/companion calls, media contract (§12)
    state.py, state_publication.py — `/api/health`, `/api/contract`, versioned `/api/state` (§4)
    reply_preparation.py — active awaited replies (no store/scheduler/public census; §6)
    main_decisions.py — Main questions/controls, gaps explicit (§3)
    work_status.py, work_projection.py, work_contracts.py — room/work status in `/api/state` (§3)
    tasks.py — headless task API (`stop_policy`: §5; CLI / Headless Boundary)
    task_archive.py, task_reviews.py — confined ZIP/files; review pages/sources (§6)
    task_events.py — SSE: GET ranks, POST v2 cursors (§3)
    task_hurry.py, task_pause.py, task_continue.py, task_owner_keep.py — POST hurry, Pause, Continue, owner-keep (§6)
    task_decision.py, secret_decision.py — `POST /api/decisions` (`KIND_QUIZ_ANSWER`; §11.1); `/secret` form
    cards.py, card_decision.py — `/api/cards/*` reads (apps as inert text)/pins; presses mailed as `KIND_CARD_INPUT` or owner words
    task_model_wait.py — model-wait mailbox decisions
    routing_decision.py — legacy `needs_manual_target` clicks (none produced)
    logs.py, storage.py, kept_services.py — log reads; Storage; kept services/Stop (§3)
    schedules.py — schedule HTTP
    files.py — File Browser, chat upload
    control.py — `/api/reset`, `/api/command` (agent notes: `agent_note.py`), `/api/git/*`, `/api/update/*`, `/api/evolution-data`
    update_progress.py — last-attempt update observation, WS invalidation (§3)
    models.py — model catalog, probes, local-model HTTP
    cost_breakdown.py, cost_period.py — ledger projections (dashboard, UTC room days, cache periods, root detail; §3; §6)
    projects.py, project_folder_contracts.py — `/api/projects*`, `/folder` (§6)
    onboarding.py, onboarding_host.py — atomic `POST /api/onboarding/complete` (503 persists nothing except `settings_save_timeout`); passive reads (§2)
    owner_settings.py, owner_effort.py — lock-as-precondition, `CommitBoundary` (below); effort range
    settings_secrets.py, secret_rows.py — single-secret reveal; list-row masking/rehydration (§3)
    settings.py — `/api/settings`, `/api/owner/*`, `GET /api/review-pool` (`ReviewPoolResponse`; unreadable catalog → `config_error`, never 500; §7)
    desktop_autostart.py — `/api/desktop/autostart`, `/api/desktop/background` (§4)
    ui_preferences.py — `state/ui_preferences.json` (§3)
    ui_holds.py — UI holds as System rows (§4)
    ui_i18n.py — `GET/POST /api/ui/i18n*` (locked owner-settings write; §7)
    extensions.py — Extensions/skills HTTP surface (routes: §4)
    contributions.py — live contributions, voice availability (CLIENT_CONTRACT §11)
    extension_receipts.py — process-qualified extension receipts (§12)
    widgets.py, widget_assets.py — widgets, scoped browser-file leases (§3)
    skill_publish.py — read-only publish preflight (five-state, no task/GitHub effect; §6)
    marketplace.py, skill_sources.py, skill_my_skills.py, skill_source_contracts.py — ClawHub, sources, My skills, `skill_source_contract_types.js` (§13)
    mcp.py — `MCPManager` HTTP (§6)
    sign_in.py, sign_in_contracts.py — owner service sign-in HTTP (§13)
    claudexor_accounts.py — thin daemon status/login/account proxies; status reads never wake the daemon, the owner's Refresh or Agents activation does (no auth logic/browser token; only successful read proves absence; §3)
    claudexor_quota.py — daemon refresh/reset/receipt proxies (catalog negotiation, typed errors, no start/retry/browser token; §3)
    claudexor_passive.py, claudexor_contracts.py — passive quota HTTP (docs/PASSIVE_QUOTA_READ.md)
    harness_maintenance.py — maintenance HTTP (§6)
    host_service.py, host_notify.py, host_sign_in.py, host_late_result.py — loopback Host Service, `/notify` `skill_notice`, skill sign-in tokens, call-bound late returns (§12)
    places_api.py — owner place/circle choices (every write is my human's; §12)
    history.py, history_deliveries.py — Chat projection, `msg_type` (§3)
    history_paging.py, history_chronology.py, chat_search.py, chat_archive.py, recall.py — dialogue pages, Project `latest_arrival`, search, read-only archive, neighbour door (no history copy; §3)
    _helpers.py — `request_json_or`, error envelope, `run_sync_to_completion`
  tools/ — Auto-discovered tool plugins (§6)
    registry.py — registry SSOT; owns shell-guard/process-tool membership sets
    registry_core.py, registry_guards.py, registry_guard_process.py, tool_context.py — load/dispatch, capability/resource/update/skill guards, process admission, `ToolContext`/`BrowserState` (contracts/tool_context.py) (§6 Safety and runtime mode)
    core.py — `read_file`, `list_files`, `write_file`, `edit_text`, `send_photo`, `send_video`, `send_file`, `send_links`, `show_card`, `search_code`, `escalate`, `forward_to_worker`
    cards.py — `show_card` (stored before `send_card`; §4)
    core_file_tools.py — file reads/lists (opt-in bounded tree) (§6 Credential mutation and diagnostic redaction)
    core_secret_paths.py — delegated action/runtime-secret helpers (§6 Credential mutation and diagnostic redaction)
    core_artifacts.py — human artifact delivery (§6 Credential mutation and diagnostic redaction)
    secret_hitl.py — `showSecretHitl` (§11.1)
    shell.py — `run_command`/`run_script` (§9)
    shell_guards.py — process-path inspection/extractors (registry_guard_process.py admits) (§6)
    shell_process.py, shell_effects.py, shell_outputs.py — command execution, tree effects, declared outputs
    deliverables_shell.py — shell write-check/output-audit paths
    shell_audit.py — post-exec custody audit
    process_facts.py — per-call env, egress masking, `check_output_views`, `run_script` ref
    write_shape.py — Python AST, Node/Ruby write targets (observation, not permission) (§6 Safety and runtime mode)
    git.py, git_plumbing.py, git_repo_edit.py, git_vcs_ops.py, vcs_read.py, git_review_cycle.py, git_managed_postcommit.py, git_evolution.py — Git tools, reviewed commits, post-commit gates, evolution (§6 Git and commit review)
    git_rollback.py — emits `rollback_request` (§2; §8)
    release_sync.py — `sync_release_metadata`, `VERSION_CARRIER_SPANS` SSOT (§10.2)
    search.py — `web_search` (`search_routes.py`) (§6 Web access mechanisms)
    browser.py, browser_session.py — Playwright tools; loop per session (§6 MCP)
    mcp_sign_in.py — MCP OAuth sign-in (§6 MCP)
    imagegen.py — `list_image_models`/`generate_image`/`recover_image` (explicit routes, artifacts first, separate view/send; docs/IMAGE_GENERATION.md)
    vision.py, vision_process.py — VLM tools; tracked vision-child IPC (parent-owned cancel)
    media.py — `extract_pdf_text` (`ocr_pdf`; text layer, not OCR), `youtube_transcript`, `extract_video_frames`, `read_archive` (`archive_reader.py`; nothing extracted)
    task_retention.py — `set_task_retention` (Keep)
    knowledge.py — topic knowledge files/index
    chronicle.py — `chronicle_write`/`memory_read`/`memory_mark` (`memory_members.py`, via `knowledge.py`), source-paged (§6)
    memory_tools.py — `registry.md` tools
    health.py — codebase health tool
    compact_context.py — context compaction tool (§6 Context fitting)
    project_journal.py — journal/workpad (foreign writes refused), `update_focus`, `mirror_tree_coordination_to_journal`
    presence.py — `configure_presence`, `presence_cancel_work` (§12)
    task_tree.py — `tree_note`/`tree_read`
    followup.py — `manage_schedules` (list/get for any task; changes root-only, audited), `schedule_followup` (`notify=true` note; `state/scheduled_tasks.json`; §5)
    join_ledger.py — direct-child dispositions, `peek_task`, `discard_child_result` (§6)
    control.py — restart, timeout, scheduling, review, chat history, model switch, `schedule_subagent` (§6 Delegated subagents)
    control_delegation.py — delegation-budget affordances, `move_to_project` (§6 Project placement)
    control_events.py, control_mind.py, control_models.py, control_relay.py, control_routing.py, control_runtime.py, control_scheduling.py, control_storage.py, control_subagent_spec.py, control_task_results.py, control_ui_translation.py, control_update_description.py, task_result_views.py — control leaves: events, mind/models, `tell_task`, routing, runtime, scheduling, Storage Free, subagent spec, child results, UI translation, update text (§3)
    tool_discovery.py — `list_available_tools`/`enable_tools` (grants nothing)
    tool_result.py, tool_catalog.py, tool_resolution.py — `ToolResult`, descriptors, arg/target binding
    arg_feedback.py — ignored-arg disclosures, typed refusals (DEVELOPMENT "LLM-first affordances")
    owner_delivery.py — live/sticky `pending_events`, room-naming receipts
    recent_tasks.py — read-only context recovery
    query_code.py — `query_code` (Code navigation)
    edit_ops.py — `apply_patch`/`edit_batch` syntax/diff backing `write_file`, diff pulse
    verify.py — independent `verify` via pre-exec guards, not a process tool by design (§6 Tool capability and execution)
    services.py, kept_service_outputs.py — background services, outside-stop capture (§9)
    git_pr.py — five non-core PR tools
    github.py, github_read.py, github_write.py — `github_read`/`github_write`/`pr_merge` (Project-bound `gh` unless read names `repo`, unprobed credentials)
    github_checks.py — checks rollup, source gaps, one deadline
    github_workflows.py, github_pr_write.py — workflow run/log/artifact reads; PR create/update
    plan_review_references.py — read-only references (no plan authority)
    plan_review.py — `plan_task` (free write/read, paid ask, `PLAN_*` refusals; §6 Plan construction and review)
    plan_ask.py — ask participants, delivery, dispatch, settlement
    plan_review_runtime.py — `run_plan_review_slots`/`plan_review_slots`
    plan_spec.py — `resolve_constitutional`, earlier-schema reads (no I/O)
    plan_evidence.py — evidence manifest
    plan_packet.py — delivery builder, `PLAN_REPLY_CONTRACT`
    plan_delivery.py — repeat-participant cross-talk/retained texts
    plan_review_artifacts.py, plan_author_history.py — earlier-schema waves/author lineage
    review.py — multi-model review, pre-commit gate (§6 Review stack)
    review_response.py — review-row response envelopes
    review_synthesis.py — window caps, coupling brief part, plan messages
    preflight_review.py — `preflight_review` (`advisory_review`), free `deterministic_only`, `review_status` (§6 Commit preflight)
    commit_gate.py — block classification, paid cycle cap, contract fingerprint
    parallel_review.py — pre-wave seat prep/admission (§6 Surfaces and money admission)
    review_subject.py — frozen review subjects, managed deltas, isolated reviewer checkouts; identities/reuse derive from captured subject (§6 Subject operation)
    review_change.py — `review_change` on any root without committing (§6 Change review on any root)
    review_change_custody.py — $0 open-wave rejoin (§6 Subject operation)
    review_result.py — `review_change(subject="result")` (§6 Result review)
    review_skill.py — `review_change(subject="skill")`, manual-lane ceiling (§13)
    review_admission.py — packet fit, retrieving briefs, wave money admission (§6 Surfaces and money admission)
    review_revalidation.py — contract fingerprint revalidation
    review_brief_coupling.py — two-part retrieving brief (change + coupling) (§6 The coupling question (Part 2 of the brief))
    review_multi_model.py, review_file_pack.py, review_prompt_text.py — commit-wave packs, carrier cut, inline-governance exclusions (§6 Guaranteed-fit ladder)
    review_checklist.py — layered `docs/CHECKLISTS.md` (`## Header`), `checklist_fingerprint` (§6 Change review on any root)
    review_context_atlas.py — `repository_index` (no file bodies) (§6 The coupling question)
    governance_context.py — tiered governance (omitted inline bodies have explicit disposition) (§6 Governance delivery)
    review_helpers.py — governance loading, checklist slicing, prompt-size SSOT, density cap
    review_binary_context.py — staged/parent Git object metadata
    scope_window.py — scope sizing (fallbacks stay disclosed, never review authority)
    scope_review_contract.py — `coupling` parser (`normalize_scope_items`, `scripts/validate_scope_receipt.py`)
    scope_required_sources.py — required-source manifest (hash/policy version bind replay)
    extension_dispatch.py — extension tool dispatch (discovery stays in registry.py; §13)
    skill_exec.py — `list_skills`/`skill_review`/`toggle_skill`/`skill_owner_action`/`skill_exec` (interpreter allowlist; gated by enablement/review/hash; §13)
    skill_publish.py, skill_source_tools.py — PR-receipt-gated publish; source/install/update tools (§13)
    skill_preflight.py — read-only preflight (widget entries stay classic unless explicitly ESM; §3; §13)
    delegate.py — `delegate_start` (`retry_of`)/`delegate_wait`/`delegate_cancel`/`delegate_answer`/`delegate_message` (§6 Delegated subagents)
    delegate_integration.py — mutation authority, snapshots, terminal capture, skill-payload CAS apply (§6 Snapshots, capture and disposition)
    delegate_payload_patch.py — skill-payload patch pipeline
    delegate_terminal_evidence.py — terminal story of one delegated run
    subagent_integration_delegated.py — `integrate_delegated_patch`
    subagent_integration.py — `integrate_subagent_patch` (protected-path gated, genesis refused), `compare_subagent_patches`
    patch_verdict.py, patch_staging.py — verdict writer (`run_<rid>`, `delegate_run_patch_verdict`); Git index staging
```

### Code navigation

`query_code` returns source evidence and local syntax facts, never semantic binding. Operations, limits and rationale: [code navigation contract](../code-navigation.md).

### Devtools boundary

`devtools/` stays outside runtime imports and package discovery; review applies and outputs stay external. Throwaway benchmarks stamp `.ouroboros_isolated_benchmark` in their data root to disable JSONL rotation (`supervisor/state.py`). Each devtool's README keeps its module map and methodology: `devtools/benchmarks/cybergym/` pauses dispatch on typed gates (`cybergym_dispatch.py`); `devtools/benchmarks/cowork_bench/` keeps scoring authority in its audit (`METHODOLOGY.md`); `devtools/benchmarks/osworld/` splits runners from their owner leaves; `devtools/macos_wkwebview/` is the optional system-WKWebView composer check. `devtools/e2e_live/` is the live E2E stand (DEVELOPMENT "Live E2E stand").

### Gateway Boundary v1

`ouroboros/gateway/` is the sole inbound browser/CLI boundary and `ouroboros/gateways/` holds thin outbound adapters without business logic; `files.py` and `host_service.py` stay separate trust boundaries. Handlers call runtime owners and never duplicate queue/review/settings/lifecycle policy, keeping UI free of agent-body imports and runtime free of ad-hoc browser contracts. `GET /api/contract` publishes endpoint/shape indexes (`docs/CLIENT_CONTRACT.md`) so replacement UIs need no core source. Owner settings writes go through `gateway/owner_settings.py::_owner_update_settings` (lock acquisition is a precondition; timeout refuses before writing) and `CommitBoundary` marks `saved` on both sides so post-commit failures name their step (§7). Frontend: `web/modules/api_client.js` (§11); extensions: `/api/extensions/<skill>/…` and namespaced WS.

### CLI / Headless Boundary

`ouroboros.cli` (`ouroboros/cli.py` SSOT) is a gateway/queue client, not a second engine: streaming commands write answers, patches, results or JSONL to stdout and progress to stderr. `POST /api/tasks` admits managed roots (§4); CLI refuses non-`root` `delegation_role` (only `schedule_subagent` creates children). Admission reserves task id and slot under one queue lock off-loop, so a cancelled waiter never cancels an admitted task; unconfirmed persistence returns 503 `unconfirmed` and keeps the row. Attachments copy into the task drive before enqueue; stored refs grant no host-path authority.

Workspace tasks default to `memory_mode=forked`; `shared` refuses external workspaces and runs forked in project scope. Stored `memory_mode` records the request and `drive_root` where it runs, so isolation never relies on relabelling. Forked/empty drives live at `data/state/headless_tasks/<task_id>/data`: forks copy `identity.md`, `WORLD.md` and `registry.md`, plus global knowledge topics and index only for non-project forks; never dialogue, scratchpad, mailbox or history; identity/world/scratchpad writes still land on the canonical root, context keeps repo governance and canonical shared memory, and `memory_export.json` is explicit (never auto-merged).

`--detach` returns after admission; `--no-stream` polls. `ouroboros run` exits 0 only on `_is_terminal_success` (completed lifecycle, clean execution, no failed/degraded objective, finished artifacts—reply ≠ "deliverable exists"); `--patch`/`--patch-out` trust `workspace_patch.json` (omitted, no-op and failed patches stay distinct, §6). Partial cost awaits `_await_cost_finality`. Schedules enqueue ordinary tasks (§5): skill `scheduled_tasks` only suggest (`skill_schedule_suggestions.py`), blank timezones take the DST-aware system zone (fixed offset only if unrecoverable), and the schedule digest rides task and consciousness context.

`packaged_cli` bootstraps the managed repo via the embedded interpreter to run `ouroboros.cli` (explicit URLs win); packaged `server` refuses to bypass launcher lifecycle; loopback `run --start` opens the app unless its data-root server is ready. Builds may download Node, ripgrep and Git to bundle them; launch-time selection never downloads: skills use the reviewed Node, shipped or prepared, and wait if absent (§6), generic launches stay PATH-first (`process_interpreters.resolve_process_node`: healthy PATH keeps argv/env; missing/dead PATH permits attested bundled prepend, never in a non-local executor), and `search_code` enumerates policy-approved files before ripgrep so speed widens no authority. Bundled consumers share `bundled_resource_bases()` order (`OUROBOROS_BUNDLE_DIR`, frozen root, interpreter ancestors, source checkout); server/CLI children run the managed repo, never in-bundle modules, and ancestor lookup lets older launchers start newer checkouts. `embedded_python_env()` protects signed bundles: bytecode goes to `data/state/pycache.noindex` and user installs to `data/state/python-userbase` (`pip_install_target_args()` adds `--user` only there), outranking bundle `site-packages` and never pruned (recovery: delete and relaunch).

Workspace binding switches the contextual repo, never the system repo (BIBLE, prompts, review governance); `/api/tasks` and promotion share `workspace_admission.validate_workspace_root()`. Ordinary folders support file/process work; Git needs a worktree. Binding updates defaults, memory, lease, preflight and finalization without dropping top-level tools or Max Architecture. The workspace executor routes processes and is not a sandbox (`executor_ref` is host-owned, `network=none` holds only when implemented, processes enter custody). Preflight writes `workspace_preflight.json` (`tools_on_path`/`tools_missing_from_path` record `shutil.which` presence, not executability, frozen in durable task metadata; failure writes an error summary, never a fictitious artifact).

Directory delegated sessions submit `execution.workspaceKind=directory` on `agent` mode after negotiating `mutability.workspaceKinds` (`directory_strategy=direct|copy`, optional `scope_paths`): the engine owns copy/CAS/apply/discard while the host keeps before/after closure on the existing custody path (no second file engine; `delegate_directory.py`, §6), never judging file/GUI work from an empty text diff. Complete inputs live in the artifact store (`attachment_manifest` preview never substitutes for unreadable full immutable `attachment_manifest_ref`). Router continuations need explicit `predecessor_task_id` (`""` = fresh; §6). Self-modification, external workspace and genesis stay distinct; genesis `deliverable_manifest.json` gaps never fail a task; copy/ZIP captures need stable bytes; global/system installs stay policy-reviewed, and `sudo` runs non-interactively (`sudo -n`).

### Runtime topology

Host sign-in startup is an owner opt-in (This computer, §3) whose only authority is the OS registration (`desktop_autostart.py`: Windows HKCU `Run`; macOS LaunchAgent with `--launch-intent automatic` and no `KeepAlive`; Linux systemd unit for deb/rpm, XDG autostart for AppImage/tar; only packaged launchers exporting a new enough `OUROBOROS_APP_VERSION`; states `unavailable`, `off`, `on`, `other_copy`, `disabled_by_os`; Panic and saved-pause keep authority; `context.py` records `runtime_env.autostart` and `keep_running_after_close`). `OUROBOROS_DESKTOP_KEEP_RUNNING` is offered when `OUROBOROS_DESKTOP_BACKGROUND=1` (Windows/macOS); both on gives a hidden sign-in start with an indicator (§9).

`launcher.py` owns PID lock, bootstrap, server, presentation, restart and cleanup outside the managed repo; `server.py` is the self-editable inner runtime (readiness wait has no clock, §2). Opt-in systemd passes `--launch-intent automatic` so Panic survives, with no restart policy: the launcher owns restarts and the crash fuse ([packaging/systemd/README.md](../../packaging/systemd/README.md)). Spawn custody: POSIX starts a new session/group; Windows creates the server suspended in a kill-on-close Job and refuses without Job custody (only the shared daemon requests breakaway). `data/state/server_process.json` pins birth/command identity: signals re-prove identity, an unmeasurable or unconfirmed predecessor keeps its record and blocks the next generation, and forced cleanup/`kill_worker_tree` spare the daemon subtree (§9).

Every server entry (`server.main`, imported app lifespan) holds `<data>/state/server.lock` for life (`server_entrypoint.admit_data_root`), enforcing one server per data root regardless of repo, PID file or port (records and bindings stay evidence; pre-lock servers go undetected). Under the PID lock, each generation first runs `launcher_server_reaper.py` to reap this install's launcher-started strays on `/proc` hosts only (report-only elsewhere), never on panic or close.

Durable custody (`process_custody.py`; §9): `spawn_supervised()` ledgers `task|session|daemon` in `process_ledger.jsonl`; the reaper kills dead-owner entries by strict `(pid, start_time, cmd_sha256)` or launch identity, never command class, so dev and packaged installs coexist (daemons stay; companions reap on uninstall or foreign generation; log-only default: `process_would_reap`). `start_parent_lifeline()` group-suicides Python entrypoints on spawner death; multiprocessing children await the parent sentinel. At start (`_orphaned_at_start`), `ppid` 1 is live for a multiprocessing child only as a direct `spawn`/`fork` child of PID 1 (ignoring env, so forkserver adoption stays orphaned), and for a plain subprocess only when `OUROBOROS_PARENT_PID` names it.

Lifecycle: each generation resolves its predecessor, checks Panic and starts the server; ready means the child's published endpoint, proven against its handle, answers health. `RESTART_EXIT_CODE` refreshes metadata/deps (§2); `MAX_CRASH_RESTARTS` within `CRASH_WINDOW_SEC` stops auto-restart; Panic cleans up and exits. Linux checks `DISPLAY`/`WAYLAND_DISPLAY` before pywebview (GTK needs a live display; Qt trusts env alone because probing constructs `QGuiApplication`, which can abort); on failure the launcher still supervises the server, prints the authoritative URL and opens the system browser best-effort (outside custody).

#### Android host (experimental)

`android/` shares this repo, Python core, SPA and update channels: the full core runs in an ARM64 GNU/Linux chroot on a Magisk-rooted device with remote LLM APIs (physical qualification limited to Pixel 10a; install/recovery/bridge: `docs/ANDROID_INSTALL.md`).

`android/install.py` provisions `/data/local/ouroboros-phone/rootfs`; `android/provision/` pins upstream inputs and installed dependencies. Inside Linux, `/opt/ouroboros/{repo,data,venv,tools,signing,launcher}` separates mutable source, personal state, venv, tools, personal APK key and immutable seed. `bootstrap/enter-linux` creates private mounts in Android's network namespace and sets `oom_score_adj=0` so root survives and the kernel can recover from OOM. `core-control` is a one-shot start/status entry, never another restart loop: it starts the common launcher with `--host-update` and `--seed-bundle` (§2), keeping any Panic marker on automatic entry.

`bootstrap/update-host` builds the APK (`host/build.py`, persistent `signing/host.keystore`), writes `data/state/android_host.json`, serializes under `android_host.lock` and keeps candidates in `data/android-builds/`. First it calls `provision/runtime.py::ensure_platform`, marking groups `platform_preparing` in `android-sdk/installation.json` before mutating and clearing only after success so an interrupted run or rolled-back Git tree never reads as prepared. `launcher_bootstrap.update_external_host` applies its own `runtime_limits.py` timeout (independent of tool/harness timeouts); the launcher's HTTP readiness window starts only after core spawn, and core health alone claims no native success. Failed or stale native evidence leaves adoption unresolved; ordinary startup keeps the available core and logs `native_update_failed`. Delegated access follows §6: a requested access profile does not prove the kernel can execute its sandbox, so the observed route is qualified.

APK `MainActivity` hosts the SPA while `CoreService` and `BootReceiver` run in `:native` at the same app UID so a WebView renderer replacement never owns the SDK bridge; `BOOT_COMPLETED` uses automatic core entry after unlock (no second core supervisor). `AndroidBridge` exposes typed, bounded operations (packages, Intents, ContentResolver, location, opt-in accessibility, notifications) over a root/host-UID socket; `android-exec` keeps arbitrary owner-root argv capability. Locally built, personally signed APKs sit outside publisher-artifact attestation.

#### Platform substrate

`platform_layer.py` owns OS observations and lock/process primitives; callers own policy. Shared locks accept non-blocking probes on read-only descriptors. `kernel_file_locks_enforced` probes each directory once (never from failed live acquisition): unsupported locks/`ENOLCK` select a recorded name-only tier (`O_EXCL` + identity recheck, no kernel exclusion; callers like money may refuse); unprobeable dirs stay enforced for that call, other errors fail closed. Won locks require readable descriptor/path inode identities (unreadable is no proof); stale recovery skips age grace on confirmed owner death and keeps live writers (unknown keeps grace; permission denial proves no death). Refresh returns ownership (loss abandons work); release unlinks only held identity (POSIX under hold; Windows post-close with bounded retry; `LockFileEx` byte: `platform_layer.py`).

Birth ≠ PID presence: Linux uses boot-qualified ticks, ps time, then bare ticks (cross-boot collision limit; ledger keeps `start_time` + qualified sibling); macOS `process_identity` reads kernel `p_starttime`; Windows uses creation FILETIME and non-signalling handles. Unknown birth permits no kill. Win32 HANDLEs are full-width; false Job termination BOOL is failure. `shell_process.contained_popen` births Windows children suspended in non-breakaway Jobs; POSIX `WNOWAIT` retains leaders until `retire_tracked` measures no live members (Linux group probes use call-local `/proc` census). Root results return; owner/idle/stop retire; Linux `tracked_start` retries pending retirements; unknown retains custody and blocks idle-worker exit. Panic uses held groups/Jobs; group records transfer no Job handle. Bundles: CLI / Headless Boundary; Node: `node_runtime` (lazy export); TCP keepalive: §6 Context fitting.

### Data layout (`~/Ouroboros/`)

Default root (overrides: §7). This tree orients; `docs/PERSISTENCE.md` registers each durable entity's writer, format, schema, retention and reset.

```
~/Ouroboros/
├── .test-evidence/ ← taskless repo tests
├── repo/ ← self-modifying git repo (§2)
│   ├── ouroboros/
│   ├── supervisor/
│   ├── web/ ← SPA (`web/modules/`)
│   ├── docs/ ← books, CHECKLISTS.md, contracts, guides, inventories/
│   └── prompts/ ← SYSTEM.md, SAFETY.md, CONSCIOUSNESS.md
├── data/ ← runtime data root
│   ├── settings.json ← (§7)
│   ├── task_results/ ← `<id>.json` import/export (store-authoritative); artifacts/<task_id>/.artifact_manifest.json, source_handles/delegated_readonly_inputs/ (§6)
│   │   └── artifact_versions/<task_id>/ ← recovery history
│   ├── task_drives/<task_id>/ ← scratch/manifests
│   ├── task_trees/<root>/blackboard.jsonl ← pruned at root end
│   ├── state/
│   │   ├── state.json ← runtime state and cost projection (not monetary authority)
│   │   ├── queue_snapshot.json ← PENDING/RUNNING projection, `worker_pool_disabled_reason` (§5)
│   │   ├── current.sqlite ← current-facts authority (`.exported-<UTC>` retired)
│   │   ├── usage.sqlite ← monetary authority (docs/USAGE_STORE.md)
│   │   ├── usage_attempts.jsonl ← imported journal/downgrade export
│   │   ├── usage_attempts.quarantine.jsonl ← proven-corrupt rows
│   │   ├── usage_import_watermark.json
│   │   ├── storage_measurement.json ← last two counts (not deletion authority)
│   │   ├── test-durations.json
│   │   ├── request_wire_compatibility.json ← exact-route wire evidence
│   │   ├── capability_evidence.json
│   │   ├── openrouter_models.json
│   │   ├── extra-ca-bundle/<digest>.pem ← merged extra CA
│   │   ├── process_ledger.jsonl ← (Runtime topology)
│   │   ├── installation_scope.json
│   │   ├── task_scopes/
│   │   ├── panic_sweep_pending.json
│   │   ├── obligations/ ← set import/export
│   │   ├── migrations.json ← schema generations, dep fingerprint
│   │   ├── body_adoption/ ← handoff, switch.py, installation_stop.py, lock
│   │   ├── owned_processes.json ← ownership set, pending stops (§9)
│   │   ├── server.lock ← server-lifetime DATA lock; never unlinked
│   │   ├── server_port ← HTTP port handoff
│   │   ├── server_port.bindings.json ← endpoint snapshot (not grant/custody)
│   │   ├── server_process.json
│   │   ├── advisory_review.json ← retired rows kept as stored
│   │   ├── code_intel/<repo_key>/inventory.json ← no source
│   │   ├── evolution_metrics_cache.json
│   │   ├── evolution_campaign.json ← objective, progress, budget
│   │   ├── evolution_checkpoints.jsonl
│   │   ├── post_task_evolution_request.json ← idle-tick promotion signal
│   │   ├── post_task_evolution_counter.json ← per-drive `every_n` counter
│   │   ├── scheduled_tasks.json ← cron/one-shot, `kind:"notify"` (§5)
│   │   ├── claudexor_rotation_provisioning.json
│   │   ├── claudexor_route_health.json
│   │   ├── subscription_census.json ← `.request` triggers read
│   │   ├── subagent_last_delegation.json ← not dispatch authority
│   │   ├── update_letter.json
│   │   ├── update_material.json ← checked range for writer
│   │   ├── projects.json ← durable tombstones
│   │   ├── projects.json.committed ← witness/copy; freshness: `projects.json.lock` (§6)
│   │   ├── project_task_bindings.json
│   │   ├── ui_preferences.json ← layout prefs, seen-revision ACKs
│   │   ├── i18n/<tag>.json ← beside `<tag>.pending.json`
│   │   ├── cancel_intents.json
│   │   ├── terminal_deliveries.json ← dedupe, receipts, outbox
│   │   ├── extension_companions.json
│   │   ├── extension_reconcile/
│   │   ├── review_continuations/ ← + corrupt/, archived/
│   │   ├── review_migrations/ ← `<ts>-slots-to-pool.json` rollback snapshots
│   │   ├── workspace_executor_processes/
│   │   ├── headless_tasks/<task_id>/data ← forked/empty child drives
│   │   ├── custody_staging/ ← unserved settlement copies
│   │   ├── custody_trash/ ← settled drives, released files
│   │   ├── betterleaks/ ← explicit installer
│   │   ├── cx/ ← Claudexor: immutable `<version>-<sha12>/`, retained `node/<version>-<platform>.g-*`, install.lock
│   │   ├── skill_grant_receipts.json ← (§13)
│   │   └── skills/<name>/ ← per-skill state (PERSISTENCE.md)
│   ├── claudexor/ ← owned Claudexor home (`CLAUDEXOR_CONFIG_DIR`)
│   ├── memory/
│   │   ├── identity.md
│   │   ├── scratchpad.md ← rendered from `scratchpad_blocks.json`
│   │   ├── dialogue_blocks.json ← with `dialogue_summary.md`
│   │   ├── dialogue_meta.json ← chronicle import cursor, nominations
│   │   ├── chronicle/ ← append-only `records.jsonl` authority, disposable `index.sqlite3`
│   │   ├── WORLD.md
│   │   ├── knowledge/ ← `patterns.md`, `improvement-backlog.md`
│   │   ├── deep_review.md
│   │   ├── registry.md ← optional hand-written source registry (`memory_map`); absent means nothing was recorded
│   │   └── owner_mailbox/
│   ├── projects/<id>/knowledge/
│   ├── observability/ ← private CAS: `blobs/`, `calls/<task_id>/`, `packs/<task>.xz`
│   ├── services/<task_id>/ ← `<service>.log`, `kept_outputs/` (§9)
│   ├── logs/
│   │   ├── chat.jsonl ← frozen at cutover
│   │   ├── progress.jsonl ← frozen at cutover
│   │   ├── chat_annotations.jsonl
│   │   ├── events.jsonl ← lifecycle, `llm_round`/`llm_usage`, errors
│   │   ├── tools.jsonl
│   │   ├── supervisor.jsonl ← workers/supervisor, cancel trail
│   │   └── task_reflections.jsonl ← early reflections, late-settlement rows
│   ├── archive/ ← rotated logs, rescue snapshots, archived repos; never GC'd
│   └── uploads/
├── projects/ ← genesis (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`), never GC'd
├── subagent_worktrees/ ← (`OUROBOROS_SUBAGENT_WORKTREE_ROOT`), GC'd; `copies.noindex/`
├── Deliverables/ ← bare `user_files` (`OUROBOROS_DELIVERABLES_ROOT`), never GC'd
└── ouroboros.pid ← auto-released on crash
```

`scripts/regenerate_inventories.py --show data-layout` probes each entry's last literal segment against repository paths and runtime literals, not full path ownership.

---
