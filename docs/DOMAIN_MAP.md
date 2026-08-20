# Domain map — v7

The v7 census answers a bottom-up question: *which files got smaller, and by how much.* This
document answers the top-down one: **for each thing the system does, who owns it now, how do
you get in, and what proves it did not move silently.**

It is a map, not an essay. Every row cites; nothing here restates what another document already
owns:

- `docs/ARCHITECTURE.md` §1 is the directory walk — one annotated line per file, in tree order.
  This is its inverse index: domain first, files second, and it names the modules the tree
  never enumerated.
- `docs/FACADE_CONSUMERS.md` classifies who consumes each retained facade binding. The
  **Pins** rows here point at the identity suites; that document says what they are worth.
- `docs/PERSISTENCE_OWNERS.md` owns durable-file ownership. Where a domain writes state, the
  file is named there, not here.
- `MIGRATION_v7.md` is the migration SSOT. Every `path::symbol` move is a row there; the
  **v7 delta** lines below are summaries of those rows, never a second ledger.

## Method

- Population: every `.py` tracked under `ouroboros/` and `supervisor/`, plus `server.py` and
  `launcher.py`, at the frozen candidate `740357c8` — **411 files, every one assigned to exactly
  one domain** (the seven package `__init__.py` files go with their package). Verified
  mechanically: zero unassigned, zero assigned twice.
- **★** marks a module the campaign added against `353fd974`. There are **132** of them
  (103 in `ouroboros/`, 29 in `supervisor/`, 0 removed anywhere) and every one appears below.
- A domain is a *behaviour*, not a directory. `supervisor/evolution_lifecycle.py` sits in the
  self-evolution domain and `ouroboros/tools/followup.py` in the supervisor domain, because
  that is where each one's question is answered. Directory is in the path; domain is here.
- Line counts are from the same deterministic inventory the size ratchet uses.
- `web/` (66 ES modules, 50 node suites) is mapped at family granularity inside D11 rather than
  per file — the campaign's web work was one owner-extraction wave over `chat.js`, and
  `MIGRATION_v7.md` carries it row by row.

## Index

| # | domain | modules | ★ new | lines | scope in one line |
|---|---|---:|---:|---:|---|
| D01 | [Agent core & main loop](#d01--agent-core--main-loop) | 27 | 11 | 19 620 | one task's turn: rounds, acceptance, budget, nudges, finalization |
| D02 | [LLM client, routing & providers](#d02--llm-client-routing--providers) | 20 | 11 | 7 264 | where a model call goes and what comes back |
| D03 | [Context assembly, fit & compaction](#d03--context-assembly-fit--compaction) | 10 | 1 | 5 012 | what the model is shown, and what is reclaimed when it will not fit |
| D04 | [Tool execution: registry, access & typed results](#d04--tool-execution-registry-access--typed-results) | 20 | 12 | 9 108 | which tool exists, whether this actor may call it, and the typed answer |
| D05 | [Tool surfaces](#d05--tool-surfaces-files-code-shell-media-external) | 22 | 5 | 13 646 | the agent-callable tools themselves |
| D06 | [Review stack](#d06--review-stack) | 46 | 19 | 26 891 | task acceptance, plan review, triad/scope, advisory, contributor lanes |
| D07 | [Delegation, subagents & Claudexor](#d07--delegation-subagents--claudexor) | 28 | 8 | 17 315 | child cognition: scheduled subagents and delegated runs |
| D08 | [Supervisor: queue, workers, events & runtime control](#d08--supervisor-queue-workers-events--runtime-control) | 38 | 26 | 14 841 | the host side: what is queued, who runs it, what the worker reports |
| D09 | [Cancellation, owner control & process custody](#d09--cancellation-owner-control--process-custody) | 12 | 1 | 7 927 | stopping things, and proving they stopped |
| D10 | [Git, update & release machinery](#d10--git-update--release-machinery) | 27 | 11 | 10 940 | the repository, the managed update, the release carriers |
| D11 | [Gateway, server & Web UI](#d11--gateway-server--web-ui) | 38 | 7 | 20 335 | HTTP/WS boundary, the server process, the SPA |
| D12 | [Settings & configuration](#d12--settings--configuration) | 13 | 5 | 3 778 | the settings document, its vocabulary and its clamps |
| D13 | [Safety, guards & runtime mode](#d13--safety-guards--runtime-mode) | 6 | 0 | 3 969 | the LLM safety check and the structural argv/path guards |
| D14 | [Skills & extensions](#d14--skills--extensions) | 43 | 10 | 17 014 | skill lifecycle, skill review, extensions, marketplace |
| D15 | [Memory, knowledge, consciousness & self-evolution](#d15--memory-knowledge-consciousness--self-evolution) | 15 | 0 | 6 453 | what the system remembers and how it changes itself |
| D16 | [Observability, usage accounting & cost](#d16--observability-usage-accounting--cost) | 8 | 1 | 3 555 | the forensic ledger and the monetary one |
| D17 | [Projects, workspaces & task results](#d17--projects-workspaces--task-results) | 17 | 2 | 8 249 | where work happens on disk and what it leaves behind |
| D18 | [Launcher, packaging, platform & shared substrate](#d18--launcher-packaging-platform--shared-substrate) | 10 | 2 | 7 252 | starting the thing, and the cross-platform floor under it |
| D19 | [Frozen contracts (ABI)](#d19--frozen-contracts-abi) | 11 | 0 | 2 132 | the surfaces that may not change shape |
| | **total** | **411** | **132** | | |

---

## D01 — Agent core & main loop

**Owners (27, ★11).** Facade `ouroboros/loop.py` (629) → nine L-B leaves:
`loop_acceptance`★ `loop_acceptance_review`★ `loop_budget`★ `loop_delivery`★
`loop_forced_finalization`★ `loop_messages`★ `loop_model_call`★ `loop_nudges`★
`loop_round_limits`★, beside the pre-existing `loop_llm_call.py` (the single-round call) and
`loop_tool_execution.py` (tool dispatch and result handling). Facade `ouroboros/agent.py`
(1137) → `agent_dispatch`★; facade `agent_task_pipeline.py` (1111) → `post_task_synthesis`★,
with `task_finalization.py`, `post_task_checkpoint.py` and `synthesis_cost_text.py` beside
them. Outcome authority: `outcomes.py` with its private leaves `_outcome_receipts.py` and
`_outcome_tool_errors.py`. Pacing and bookkeeping: `task_pacing.py`, `deadline_utils.py`,
`mutation_attribution.py`, `owner_mailbox.py`, `agent_startup_checks.py`.

**Entry points.** `loop.run_llm_loop()` · `loop.seal_task_transcript()` ·
`agent.OuroborosAgent` / `agent.make_agent()` · the leaves reach rebindable parent state
through the call-time handle `_loop()`, never a from-import.

**Pins.** `tests/test_loop_owner_facades.py` (surviving identities **and** the asserted
absence of `RETIRED_FROM_LOOP`) · `tests/test_module_handle_extraction.py` (per-leaf D33/D38
handle sets) · `tests/test_lc2_owner_facades.py` · `tests/test_run_llm_loop.py` ·
`tests/test_loop_acceptance_gate.py` · `tests/test_v678_acceptance_state.py` ·
`tests/test_budget_limits.py` · inventories `lb_loop_*` / `lc2_*` in
`tests/_v7_ledger_inventories.py`.

**v7 delta.** `loop.py` 7102 → 629 lines: each phase of a round became an owner leaf reading
rebindable parent globals through `_loop()` (delta D33), and `agent.py` / `agent_task_pipeline.py`
shed their dispatch and post-task-synthesis clusters the same way (delta D38).

---

## D02 — LLM client, routing & providers

**Owners (20, ★11).** Facade `ouroboros/llm.py` (716) is the composition point: `LLMClient`
is assembled from ten mixins the leaves own — `llm_routing`★ (target resolution, clients,
cache/session affinity), `llm_messages`★ (send-copy shaping), `llm_capability_policy`★
(discovered route capability), `llm_fallback`★ (the recovery ladder and both send drivers),
`llm_attempt`★ (one physical attempt as a candidate), `llm_pricing`★, and the four native
lanes `llm_anthropic`★ `llm_gigachat`★ `llm_local`★ `llm_openai_compatible`★. Beside them:
`llm_probe`★ (bounded one-shot probes with no retry/fallback path), `llm_observability.py`,
`provider_models.py`, `pricing.py`, `model_concurrency.py`, `fallback_cooldown.py`,
`vision_routing.py`, `local_model.py`, `local_model_autostart.py`.

**Entry points.** `LLMClient.chat` / `.chat_async` · `llm_probe` for the Provider Test control
and oversized-context evidence · `openrouter_web_search_server_tool` /
`anthropic_web_search_server_tool`.

**Pins.** `tests/test_llm_extraction.py` (every `LLMClient` member resolves to its mixin owner,
and the member inventory is unchanged — stronger than binding identity) ·
`tests/test_llm_provider_golden.py` with nine replayable fixtures in
`tests/fixtures/llm_golden/` · `tests/test_llm_typed_policy_refusal.py` ·
`tests/test_multimodal_chat.py` · `tests/test_capability_probe_accounting_v664.py`.

**v7 delta.** `llm.py` 4370 → 716: one physical attempt became an owner (delta D09) and each
provider lane a module, with the leaves deliberately *not* reading back through the facade —
`llm_probe.py` states the rule in place: name the owner leaf, never `llm.py`.

**Not here.** Model *slot* resolution and the fallback chain are settings vocabulary → D12.

---

## D03 — Context assembly, fit & compaction

**Owners (10, ★1).** Facade `ouroboros/context.py` (1318) → `context_runtime_facts`★ (the
runtime section's fact builders) and `context_health.py` (health-invariant assembly).
Independent owners: `context_fit.py` (the task-local fit authority), `context_budget.py`
(window vocabulary and typed reclaim receipts), `context_layout.py` (reference-document form),
`context_compaction.py` (the reclaim materializer), `context_mode_compat.py`,
`capability_evidence.py` (sourced, route-fingerprinted window evidence). Agent surface:
`tools/compact_context.py`.

**Entry points.** `context.build_user_content()` · `context.build_runtime_section()` ·
`context.build_knowledge_sections()` · the `compact_context` tool · a `ContextFit` measurement
carrying a typed reclaim request/receipt.

**Pins.** `tests/test_context_fit_integration.py` · `tests/test_context_fit_v664.py` ·
`tests/test_context_budget_ssot.py` · `tests/test_context_reclaim_materializer.py` ·
`tests/test_compaction.py` · `tests/test_loop_compaction.py` · `tests/test_max_context_gate.py` ·
`tests/test_capability_evidence.py`; the test-side split of `tests/test_context.py` into
`test_context_runtime_section.py` / `_memory` / `_drive_state` / `_advisory_review` (S7a rows).

**v7 delta.** One extraction, and it was forced by the merge rather than by the campaign: upstream
growth pushed `context.py` past the branch's 1500-line ceiling during the v6.105.0 adoption, so
the runtime-section fact builders left behind a re-exporting facade.

---

## D04 — Tool execution: registry, access & typed results

**Owners (20, ★12).** Facade `ouroboros/tools/registry.py` — 39 lines, nothing but proven
compatibility re-exports — over `registry_core`★ (orchestration: catalog load, overlays, guard
order, builtin invocation), `registry_guards`★ (host-owned pre-dispatch capability/resource,
ephemeral, delegated-child, managed-update, skill-repair and process guards),
`registry_guard_process`★ (the process/shell coordinator and post-execution tripwires),
`tool_resolution`★ (argument normalization and physical binding), `tool_context`★ (the concrete
`ToolContext` and `BrowserState`), `tool_catalog`★ (immutable first-party catalog),
`tool_result`★ (the `ToolResult` / `ToolCodeSpec` owner and its finite code table). Access
decision: facade `ouroboros/tool_access.py` (734) → `tool_access_types`★ (the closed enums and
the profile × root × operation matrix), `tool_access_roots`★ (who is acting, where each root
lives), `tool_access_paths`★ (physical path primitives), `tool_access_user_files`★ (the
`user_files` confinement). Beside them: `tool_capabilities.py`, `tool_policy.py`,
`tool_module_inventory`★ (the AST membership authority the frozen manifest reads),
`protected_artifacts.py`, `tools/tool_discovery.py`, `tools/extension_dispatch.py`.

**Entry points.** `ToolRegistry` (through the facade, which 86 files still import) ·
`tool_access.decide_tool_access()` · `tool_access.filesystem_affordance_map()` ·
each tool module's `get_tools()`.

**Pins.** `tests/test_tool_owner_facades.py` · `tests/test_registry_core.py`
(`test_registry_core_extraction_preserves_only_proven_facades` — the enumerated string list is
itself the consumer for 17 private bindings) · `tests/test_tool_access_extraction.py` ·
`tests/test_tool_result.py`, `tests/test_tool_result_meta_boundaries.py`,
`tests/test_tool_result_t46.py`, `tests/test_process_guard_codes.py`,
`tests/test_process_result_corrections.py`, `tests/test_core_native_results.py`,
`tests/test_control_native_results.py` (the typed-result cutover, family by family) ·
`tests/test_tool_classification_differential.py` + `tests/tool_classification_corpus.py` +
`tests/fixtures/legacy_tool_classification_306f8827.json` (the golden of the *retired*
classifier) · `tests/test_frozen_tool_inventory.py` · `tests/test_policy_path_resolution.py`.

**v7 delta.** `registry.py` 3438 → 39 is the sharpest facade in the campaign, and the typed
`ToolResult`/`ToolCodeSpec` seam (delta D02) replaced re-read prose with a closed code table —
per-family approved deltas live in `tests/test_tool_classification_differential.py::APPROVED_DELTAS`,
not in this document.

---

## D05 — Tool surfaces: files, code, shell, media, external

**Owners (22, ★5).** File/edit/delivery: facade `tools/core.py` (1277) → `core_file_tools`★
(read/list handlers and their binding) and `core_artifacts`★ (photo/video/document delivery);
`tools/edit_ops.py` (`apply_patch`), `ouroboros/artifacts.py`. Process execution: facade
`tools/shell.py` (720) → `shell_process`★ (the execution substrate every command-running tool
shares), `shell_effects`★ (what a command did to the tree), `shell_outputs`★ (declared outputs,
fingerprints, artifact registration); `tools/verify.py` runs the agent's declared check through
the same machinery. Code intelligence: `tools/search.py`, `tools/query_code.py`,
`ouroboros/code_intelligence.py`, `ouroboros/code_search_rg.py`. Everything else the agent can
call directly: `tools/browser.py`, `tools/media.py`, `tools/vision.py`, `tools/services.py`,
`tools/health.py`, `tools/recent_tasks.py`, `ouroboros/mcp_client.py`,
`ouroboros/python_interpreter.py`.

**Entry points.** `get_tools()` per module (auto-discovered) · `read_file`/`list_files`/`write`/
`apply_patch` · `run_command`/`run_script`/`verify_and_record` · `search_code`/`query_code` ·
`send_file`/`send_photo`/`send_video` · `ocr_pdf`/`youtube_transcript` · `mcp_<server>__<tool>`.

**Pins.** `tests/test_shell_extraction.py` · `tests/test_core_extraction.py` ·
`tests/test_core_native_results.py` · `tests/test_smoke.py` ·
`tests/test_runtime_reliability_v655.py` · `tests/test_v678_receipt_reconciliation.py` ·
`tests/test_disabled_tools_policy.py` · `tests/test_workspace_authority_binding.py`.

**v7 delta.** Two facades: `tools/shell.py` split its process substrate, effect reading and
declared-output registration into three owners, and `tools/core.py` handed read/list and the
owner-chat artifact handlers to non-catalog implementation owners while keeping the schema and
membership itself.

---

## D06 — Review stack

The largest domain, and the one the constitution cares about most. Five review surfaces share
one substrate.

**Owners (46, ★19).**

- *Substrate and panel vocabulary* — `review_substrate.py` (the reviewer-slot coordinator and
  the single mint for slot identity) → `review_session_verdict`★; `review_records`★ (the typed
  panel records every surface shares), `review_verdict`★ (pure reducers over a completed run),
  `review_projection`★ (transport-failure classification and the outward view);
  `review_execution.py` → those three plus `review_slot_cancel.py`; `reviewer_slot_config.py`,
  `reviewer_window.py`, `review_cycles`★ (the one `OUROBOROS_REVIEW_MAX_CYCLES` cap with three
  documented per-gate meanings), `triad_review.py`, `task_continuation.py`.
- *Advisory ledger* — `review_state.py` (the durable store) → `review_state_model`★ (the
  in-memory ledger and its permitted transitions) and `review_state_records`★ (the record
  vocabulary, retention and identity).
- *Acceptance evidence* — `review_evidence.py` → `review_evidence_sections`★ (every typed
  section and the whole-packet budget) and `review_evidence_refs.py`.
- *Task acceptance* — `tools/review.py` → `review_multi_model`★ (delta D37).
- *Advisory gate* — `tools/claude_advisory_review.py` → `review_advisory_prompt`★ and
  `review_advisory_run`★ (delta D37); transport `gateways/claude_code.py`.
- *Scope review* — `tools/scope_review.py` → `scope_review_budget`★ and `scope_review_pack`★;
  beside them `scope_review_session.py`, `scope_window.py`, `scope_review_contract.py`.
- *Plan review* — `tools/plan_review.py` (the engine) with `plan_spec`★, `plan_evidence`★,
  `plan_packet`★, `plan_render`★ and `plan_review_runtime.py`.
- *Shared plumbing* — `tools/review_helpers.py` → `review_prompt_text`★ (the fixed reviewer
  vocabulary) and `review_file_pack`★ (reviewable-file classification); `tools/review_synthesis.py`,
  `tools/review_context_atlas.py`, `tools/review_binary_context.py`, `tools/parallel_review.py`,
  `ouroboros/review.py` (code collection and complexity metrics), `preflight_runner.py` (the
  hermetic pytest gate), `deep_self_review.py`.

**Entry points.** `review_substrate.run_review_request()` and `ReviewCoordinator` ·
`scope_review.run_scope_review()` · the `plan_task` tool (`plan_review.get_tools()`) · the task
acceptance review tool · `scripts/run_external_review.py` for the operator/contributor lane ·
`scripts/run_plan_review.py`.

**Pins.** `tests/test_review_owner_facades.py` · `tests/test_review_substrate_extraction.py` ·
`tests/test_review_state_extraction.py` · `tests/test_review_evidence_extraction.py` ·
`tests/test_review_helpers_extraction.py` · `tests/test_scope_review_extraction.py` ·
`tests/test_module_handle_extraction.py` (the `_rev()` / `_car()` D37 sets) · behaviour:
`tests/test_plan_review_engine.py`, `tests/test_plan_review_health.py`,
`tests/test_scope_review_wiring.py`, `tests/test_review_session_scope_wiring.py`,
`tests/test_review_anti_thrashing.py`, `tests/test_reviewer_slot_config.py`,
`tests/test_advisory_observability.py`, `tests/test_immune_hardening.py`, and the eight
`test_preflight_*` siblings of the W5 split.

**v7 delta.** The census's review-stack family went 13 modules → 27 (this domain is wider: it
also carries the plan-review engine, the hermetic preflight gate and the advisory transport).
The panel record/verdict/projection
vocabulary that four surfaces were each re-deriving now has one owner apiece, the advisory
ledger split store from model from records, and the D37 handle keeps every test that patches
`tools/review.py` or `tools/claude_advisory_review.py` intercepting the moved bodies. The
review *policy* — models, quorum, enforcement — is unchanged; only its ownership moved.

---

## D07 — Delegation, subagents & Claudexor

**Owners (28, ★8).** Axis vocabulary and dispatch: `subagents.py` → `subagent_route_health`★
(the one manifest reader behind every delegated dispatch, extracted after the v6.105.0
adoption); `subagent_worktrees.py` (acting `self_worktree` lifecycle); `task_tree_ledger.py`
(the swarm blackboard). Nanny verbs: facade `tools/delegate.py` (1270) → `delegate_terminal`★
and `delegate_payload_patch`★ (DEL1 leaves, delta D36) plus the earlier size-gate extractions
`delegate_output.py`, `delegate_containment.py`, `delegate_progress.py`,
`delegate_interactions.py`, `delegate_shared.py`; `tools/delegate_integration.py`;
`tools/subagent_integration.py` → `subagent_integration_delegated`★. Custody:
`delegate_custody.py` → `delegate_custody_reconcile`★; `delegate_evidence.py`. Scheduling
surface: `tools/control_scheduling`★, `tools/control_subagent_spec`★, `tools/control_task_results`★,
`tools/control_delegation.py`. Coordination tools: `tools/task_tree.py`, `tools/join_ledger.py`.
Claudexor: `gateways/claudexor.py` (the control-plane gateway; the daemon token stays inside
it), `claudexor_daemon.py` (the Ouroboros-owned `claudexord`), `claudexor_runtime.py` (the
exact managed engine pin).

**Entry points.** `schedule_subagent` · `wait_tasks` · `delegate_start` / `delegate_wait` ·
`integrate_subagent_patch` · `tree_note` / `tree_read` · `subagents.resolve_subagent_dispatch()`
→ `capability_delta`.

**Pins.** `tests/test_delegate_owner_facades.py` · `tests/test_module_handle_extraction.py`
(D36 per-leaf sets) · the eleven S7a siblings of `tests/test_delegated_subagent_transport.py`
(6178 → 366) · `tests/test_delegated_skill_payload.py` · `tests/test_delegate_answer.py` ·
`tests/test_claudexor_admission_wait.py` and the four S7a Claudexor siblings ·
`tests/test_acting_subagents.py`.

**v7 delta.** The delegate family went 9 modules → 12 with `delegate_custody.py` 1600 → 1275,
the terminal-payload and payload-patch clusters getting owners under the D36 handle; nothing
about the delegation *contract* moved, which is why the transport suite could be split eleven
ways without a behaviour row.

---

## D08 — Supervisor: queue, workers, events & runtime control

**Owners (38, ★26)** — the densest concentration of new modules in the tree.

- *Queue* — facade `supervisor/queue.py` (430: state, admission, fences, the re-entrant lock)
  → `queue_snapshot`★, `queue_timeouts`★, `queue_schedules`★, `queue_evolution`★, each reading
  rebound names through the call-time handle `_queue()` (delta D18); `queue_transitions.py`,
  `task_admission.py`.
- *Workers* — facade `supervisor/workers.py` (724: pool state and `init`) → `worker_promotion`★,
  `worker_chat_lane`★, `worker_health`★, `worker_pool_lifecycle`★, `worker_assignment`★ (all on
  the handle) and `worker_process`★ (what runs *inside* the child, reading no pool state at all).
- *Events* — dispatcher `supervisor/events.py` (270: the typed table and its loop, nothing else)
  → ten handler-family owners: `events_chat_delivery`★ `events_subagent_admission`★
  `events_schedule_task`★ `events_project_routing`★ `events_task_done`★ `events_evolution_done`★
  `events_coop_checkpoint`★ `events_budget`★ `events_worker_reports`★ `events_runtime_controls`★;
  plus `event_taxonomy`★ (data only: the declared disposition of every event kind, in four tiers).
- *Rest* — `supervisor/state.py`, `supervisor/message_bus.py`, `active_activity`★,
  `schedule_time.py`, `ouroboros/schedule_contract.py`, `ouroboros/promotion_source.py`.
- *Agent-side runtime control* — facade `tools/control.py` (411) → `control_events`★,
  `control_routing`★, `control_runtime`★; `tools/followup`★ (`schedule_followup` writes into the
  existing scheduled-task table rather than minting a second one).

**Entry points.** `queue.enqueue_task()` · `queue.init()` / `init_queue_refs()` ·
`workers.ensure_worker_pool_started()` / `workers.init()` · `events.dispatch_event()` ·
`worker_process.worker_main` (kept module-level so spawn platforms can re-import it by name) ·
the `restart`/`promote`/`toggle_evolution` control tools · `schedule_followup`.

**Pins.** `tests/test_events_extraction.py` · `tests/test_worker_process_extraction.py` ·
`tests/test_module_handle_extraction.py` (the D18 per-leaf sets for both queue and pool) ·
`tests/test_cancel_custody_extraction.py` · `tests/test_event_taxonomy.py` (a producer added
without an answer, and an answer left behind by its last producer, are both failures) ·
`tests/test_control_extraction.py` · `tests/test_promote_chat_flow.py` and its five S7a
siblings · `tests/test_task_status_flow.py`'s six S7b siblings ·
`tests/test_promote_event_transport.py`.

**v7 delta.** `events.py` 4288 → 270 and `queue.py` 1584 → 430 with `workers.py` 2894 → 724 —
and the mechanism that made it possible is the module handle (delta D18): `init` rebinds the
roots and dozens of tests monkeypatch them on the parent, so a leaf holding a from-import would
freeze the object it saw at import time. Note that `workers.py` grew no `workers_*` prefix
family; its responsibilities went to differently-named owners, which is why a prefix-only
census undercounts this split.

---

## D09 — Cancellation, owner control & process custody

**Owners (12, ★1).** Durable intent: `ouroboros/cancel_intents.py` (the one `request_cancel`
ingress and the four lifecycle mutators, every one fenced by the claim generation).
Custody and settlement: `supervisor/task_lifecycle.py` (the cascade protocol — fences, tokens,
subtree sweep) → `cancel_custody`★ (the ONE settle owner of a durable intent) and
`cancel_publication.py` (the typed `CANCEL_*` vocabulary and the owed-before-settle outbox).
Delivery and reaping: `supervisor/terminal_delivery.py`, `supervisor/task_reaper.py`.
Owner-initiated control: `supervisor/owner_stop.py` (graceful `finalize_then_cancel`),
`ouroboros/owner_hurry.py`, `supervisor/steering.py`, `ouroboros/server_control.py` (restart,
panic stop). Process truth: `ouroboros/process_custody.py` (the durable orphan ledger),
`ouroboros/process_containment.py` (env-token membership read from live kernel state).

**Entry points.** `cancel_intents.request_cancel()` · `POST /api/tasks/{id}/cancel` (with
`stop_policy`) · `POST /api/tasks/{id}/hurry` · `sweep_cancel_intents` (the watchdog) ·
`spawn_supervised()` / `start_parent_lifeline()`.

**Pins.** `tests/test_cancel_protocol_inventory_s6.py` (structural inventories of the
protocol's owners) · `tests/test_cancel_intent_corruption_s6.py`,
`tests/test_owner_stop_fences_s6.py`, `tests/test_cascade_chatless_residual_s6.py`,
`tests/test_daemon_token_containment_s6.py`, `tests/test_subagent_worktree_registry_s6.py` —
the S6 lane, which deliberately pins residuals **without** fixing them ·
`tests/test_e2e_cancellation_scenarios.py` (E1–E12 on an isolated server) · the eight S7b
siblings of `tests/test_cancel_intents_phase_a.py` (2400 → 241) ·
`tests/test_owner_hurry_s3.py` · `tests/test_process_custody.py`.

**v7 delta.** One extraction — the settle owner left `task_lifecycle.py` for `cancel_custody.py`
at the module-size boundary and is re-imported there, so `supervisor.queue`'s re-exports and
every caller keep one surface. The cascade protocol deliberately did **not** move: it is one
protocol over module-local state, and splitting it would have created a second answer.

---

## D10 — Git, update & release machinery

**Owners (27, ★11).** Agent-facing git: facade `tools/git.py` (991) → `git_plumbing`★ (the
low-level runner every git owner shares), `git_repo_edit`★ (uncommitted write and exact-match
edit), `git_vcs_ops`★ (inspection and rollback), `git_review_cycle`★ (staging, advisory/triad/
scope review, reviewed-material binding), `git_evolution`★ (campaign authority at the reviewed-
commit and publication boundaries); beside them `tools/commit_gate.py`,
`tools/review_revalidation.py`, `tools/git_rollback.py`, `tools/git_pr.py`, `tools/github.py`,
`tools/ci.py`, `tools/release_sync.py` (the span-descriptor SSOT). Host-side git: facade
`supervisor/git_ops.py` (605) → `git_ops_remotes`★, `git_ops_updates`★, `git_ops_reset`★,
`git_ops_rescue`★, all reading through the call-time handle `_go()` (delta D35). Managed update:
`supervisor/update_merge.py` → `update_merge_plan`★ (the three insertion points) and
`update_carriers`★ (carrier-aware span substitution, delta D34); `update_merge_policy.py`,
`update_source.py`, `update_recovery.py`, `ouroboros/repo_remotes.py`. Release carriers:
`ouroboros/version.py`, `ouroboros/size_ratchet_manifest.py`.

**Entry points.** `commit_reviewed` · `vcs_rollback` · `git_ops.init()` (which rebinds
`REPO_DIR` / `DRIVE_ROOT` / `BRANCH_*` — the reason the leaves must read through the facade) ·
the managed-update transaction · `scripts/carrier_rebase_helper.py`.

**Pins.** `tests/test_git_extraction.py` · `tests/test_git_ops_owner_facades.py` (the cleanest
facade in the tree: every retained binding has a live runtime consumer) ·
`tests/test_update_merge_owner_facade.py` · `tests/test_update_carriers.py` ·
`tests/test_carrier_rebase_helper.py` · `tests/test_git_ops_default_roots.py` (delta D13) ·
`tests/test_commit_gate.py` · the four W5 siblings of `tests/test_git_ops_recovery.py` and the
five T-wave siblings of `tests/test_git_review_pipeline.py`.

**v7 delta.** The git family went 4 modules → 13 (`tools/git.py` 2870 → 991,
`supervisor/git_ops.py` 1988 → 605) with its own handle delta D35, and the update engine gained
a carrier-aware resolver (D34) whose span descriptors are owned once, in `release_sync.py`.

---

## D11 — Gateway, server & Web UI

**Owners (38 Python, ★7; 66 ES modules, 21 of them new).** Server process: facade `server.py`
(1421) → `server_liveness`★ (wedge predicates and the watchdog), `server_maintenance`★ (the
upkeep a supervisor generation owes the drive), `server_owner_routing`★ (where one owner message
goes), `server_routing_context`★ (the bounded projections a turn may address), `server_restart`★
(the restart transaction) and `server_process`★ (the facts every leaf shares — drive root,
logger, restart signals); beside them `server_runtime.py`, `server_entrypoint.py`,
`server_auth.py`, `server_web.py`. Gateway Boundary v1 — 26 modules under `ouroboros/gateway/`,
unchanged in membership: `router.py`, `contracts.py` (the PRO-frozen envelope index), `ws.py`,
`state.py`, `tasks.py`, `task_hurry.py`, `task_events.py`, `control.py`, `settings.py`,
`owner_settings.py`, `onboarding.py`, `onboarding_host.py`, `schedules.py`, `files.py`,
`history.py`, `logs.py`, `models.py`, `projects.py`, `marketplace.py`, `extensions.py`,
`mcp.py`, `claudexor_accounts.py`, `host_service.py`, `ui_preferences.py`, `_helpers.py`.
SPA seam: `ouroboros/client_surface`★ (the Owner Surface Fact SSOT) with `web/modules/client_surface.js`.

**Web families** (`web/modules/`, `MIGRATION_v7.md` carries the rows): `chat.js` 4654 → 1477
after handing out **nineteen** new owners — `chat_card_state`, `chat_card_actions`,
`chat_live_cards`, `chat_live_card_view`, `chat_history_sync`, `chat_media_bubbles`,
`chat_document_bubble`, `chat_attachments`, `chat_composer`, `chat_controls`,
`chat_header_controls`, `chat_notices`, `chat_frame_routing`, `chat_task_frames`,
`chat_task_ui_state`, `chat_subagent_routing`, `chat_message_identity`,
`chat_message_annotations`, `chat_timeline_anchor` — plus rows into the pre-existing
`costs.js` and `utils.js`. The other two new ES modules, `chat_activity.js` and
`client_surface.js`, came with upstream, not with this wave.

**Entry points.** `server.py` composition root · `gateway/router.py` route collector for
`/api/*` and `/ws` · `gateway/contracts.py` envelope index · the SPA entry in `web/`.

**Pins.** `tests/test_server_extraction.py` · `tests/test_contracts.py` ·
`tests/test_client_surface.py` · `tests/test_onboarding_host.py` ·
`tests/test_restart_reconnect.py` · `tests/test_page_chrome_static.py` · the six W5 siblings of
`tests/test_ui_smoke_playwright.py` (3819 → 330, marker-gated browser lane) · node suites
`web/tests/chat_facade.test.js` (`assertChatFacadeOwnerIdentity`), `chat_live_cards.test.js`,
`chat_history_sync.test.js`, `timeline_anchor.test.js`, `composer.test.js` and sixteen siblings.

**v7 delta.** `server.py` 2986 → 1421 and the whole server+gateway family got *smaller*, not
just flatter — it lost 668 lines net, because the work went to the 29 new `supervisor/` modules
rather than to new server leaves. On the web side one wave gave `chat.js` twenty owners; five
anonymous bodies that had no name at the base could not carry a ledger row and are disclosed by
name in `MIGRATION_v7.md` instead.

---

## D12 — Settings & configuration

**Owners (13, ★5).** Facade `ouroboros/config.py` (900 — the SSOT import surface for paths, the
locked settings-file lifecycle, the owner-only mode ratchets and the PID lock) →
`settings_defaults`★ (shipped values, retired keys, the disk-only/never-exported classification),
`settings_scales`★ (the closed scales a value is clamped to), `model_slots`★ (slot resolution,
the ordered fallback chain, rename-alias migration), `review_model_routes`★ (reviewer model
lists per lane), `runtime_limits`★ (numeric knobs and their clamps). Beside them:
`secret_masking.py`, `update_channels.py`, `settings_setup_contract.py`, `onboarding_wizard.py`,
`subscription_install_presets.py`, `launcher_onboarding.py`, `colab_bootstrap.py`.

**Entry points.** `config.load_settings()` / `save_settings()` · the four `OUROBOROS_*` path
variables · `gateway/owner_settings.py`'s locked write seam (D11) · `POST /api/onboarding/complete`.

**Pins.** `tests/test_config_extraction.py` · `tests/test_settings_read_seam.py` (the read path
characterized *before* the normalization seam moved) · `tests/test_settings_env_on_disk.py`
(which environment values may become settings-file content) · `tests/test_onboarding_wizard.py` ·
`tests/test_onboarding_complete_endpoint.py` · `tests/test_colab_bootstrap.py`.

**v7 delta.** `config.py` kept ownership of the lifecycle and handed away the *vocabulary*: what
the values are (`settings_defaults`), what they are clamped to (`settings_scales`), which model
each slot resolves to (`model_slots`), which reviewer each lane gets (`review_model_routes`) and
what the numeric ceilings are (`runtime_limits`) — delta D03, with the retired-knob set as D04.

---

## D13 — Safety, guards & runtime mode

**Owners (6, ★0).** `ouroboros/safety.py` (the policy-based LLM safety check, taking its host
facts from the context because it runs inside every worker), `ouroboros/runtime_mode_policy.py`
(the protected-path policy shared by registry, git tools and the Claude gateway),
`ouroboros/git_shell_policy.py` (structural git argv classifiers), `ouroboros/shell_parse.py`
(the argv/inline-command parser guardrails use without importing the tools package),
`ouroboros/argv_budget.py` (the E2BIG admission SSOT), `ouroboros/tools/shell_guards.py`.

**Entry points.** the safety check invoked per tool call · `runtime_mode_policy` consulted by
`tools/registry_guards` (D04), `tools/git.py` (D10) and `gateways/claude_code.py` (D06).

**Pins.** `tests/test_interpreter_family_write_fence.py` · `tests/test_platform_guard.py` ·
`tests/test_process_guard_codes.py` · `tests/test_python_interpreter.py` ·
`tests/test_runtime_mode_registry_gating.py` and the ten S7a siblings of
`tests/test_runtime_mode_core.py` / `tests/test_runtime_mode_elevation.py` ·
`tests/test_v647_megacommit.py`.

**v7 delta.** Structural: none — not one module in this domain was split, merged, renamed or
retyped. Behavioral: real, small, and owner-approved. `safety.py` (+48/−6 against `353fd974`)
gained `_safety_drive_root` (the context-owned data-root resolver replacing the cwd-relative
`../data` guess), `_record_safety_usage` (an injected-or-fallback accounting sink so a safety
call with no event queue is still charged), and classifies `schedule_followup` as
`POLICY_SKIP`; `runtime_mode_policy.py` (+23/−2) widened the safety-critical and
release-protected file families. The test side moved as well: the two runtime-mode giants
became twelve themed suites, which is coverage rearranged, not policy changed.

---

## D14 — Skills & extensions

**Owners (43, ★10).**

- *Skill lifecycle* — `skill_loader.py` (discovery and durable state), `skill_readiness.py`,
  `skill_dependencies.py`, `skill_repair_admission.py`, `skill_publish_eligibility.py`,
  `skill_lifecycle_queue.py`, `skill_owner_attestation.py`, `skill_token.py`.
- *Skill review* — `skill_review.py` (the lifecycle driver) → `skill_review_packs`★ (the
  reviewable payload and its budget), `skill_review_prompt`★ (what the reviewer is asked),
  `skill_review_output`★ (what happens to the answers), `skill_review_rebuttals`★ (the durable
  anti-thrashing record); beside them `skill_review_status.py`, `skill_review_passes.py`,
  `skill_review_history.py`, `skill_review_runner.py`.
- *Extensions* — facade `extension_loader.py` (956, the lifecycle itself) → `extension_registry_state`★
  (the process-wide registries of live surfaces), `extension_surface_names`★ (the provider-safe
  namespace), `extension_child_catalog`★ (host-side re-validation at the trust boundary),
  `extension_import_staging`★ (staged import trees), `extension_liveness`★ (the liveness
  projection), `extension_plugin_api`★ (the `PluginAPI` object handed to `register(api)`);
  beside them `extension_process_runner.py`, `extension_ui_validation.py`,
  `extension_isolated_deps.py`, `extension_health.py`, `extension_companion.py`,
  `extension_reconcile_queue.py`, `event_bus.py`.
- *Marketplace* — `marketplace/` (8 modules: `clawhub`, `ouroboroshub`, `fetcher`, `adapter`,
  `install`, `install_specs`, `isolated_deps`, `provenance`).
- *Agent surface* — `tools/skill_exec.py`, `tools/skill_preflight.py`, `tools/skill_publish.py`.

**Entry points.** `extension_loader.reconcile_extension()` / `load_extension()` / `reload_all()` ·
`skill_loader.skill_state_dir()` and `review_status_allows_execution()` · the `list_skills`,
`review_skill`, `toggle_skill`, `skill_exec`, `submit_skill_to_hub` tools ·
`/api/extensions/*` and `/api/skills/*` (D11).

**Pins.** `tests/test_extension_loader_extraction.py` · `tests/test_skill_review_extraction.py` ·
`tests/test_extension_plugin_api_matrix.py` (the load/dispatch/unload characterization matrix) ·
the five TS1 siblings of `tests/test_extension_loader.py`, the five S7a siblings of
`tests/test_extensions_api.py`, the six S7b siblings each of `tests/test_skill_review.py`,
`tests/test_skill_loader.py` and `tests/test_skill_exec.py` ·
`tests/test_skill_smoke_official.py` · `tests/test_marketplace_api.py` ·
`tests/test_marketplace_provenance_contract.py` · `tests/test_owner_attestation_v639.py`.

**v7 delta.** Two facades: `extension_loader.py` handed out six owners covering registry state,
naming, the trust-boundary re-validation, import staging, liveness and the `PluginAPI` object,
and `skill_review.py` split into pack / prompt / output / rebuttals. The skill *review policy* —
which verdicts block, which are advisory — did not move; it still lives in
`skill_review_status.py`.

---

## D15 — Memory, knowledge, consciousness & self-evolution

**Owners (15, ★0).** Memory and dialogue: `memory.py` (scratchpad, identity, chat history),
`consolidator.py` (generation-aware block-wise consolidation), `project_facts.py`,
`tools/memory_tools.py`, `tools/knowledge.py`. Cognition: `consciousness.py` (the background
loop), `reflection.py`, `improvement_backlog.py`, `semantic_dedup.py`, `world_profiler.py`.
Self-evolution: `supervisor/evolution_lifecycle.py` (campaign state and transaction lifecycle),
`post_task_evolution.py` (the durable promotion signal a worker writes),
`evolution_checkpoints.py`, `evolution_fingerprint.py`, `tools/evolution_stats.py`.

**Entry points.** `Memory` · the consciousness loop · `/evolve start` / `/evolve off` and the
`toggle_evolution` tool (routed through D08) · the experience-review memory write-back
(`MEMORY_ACTIONS_JSON` → `apply_memory_actions`, never auto-written to `identity.md`).

**Pins.** `tests/test_consciousness.py` · `tests/test_context_memory.py` ·
`tests/test_post_task_evolution.py` · `tests/test_post_task_reflection.py` ·
`tests/test_root_post_task_synthesis.py` · `tests/test_semantic_dedup_v6370.py` ·
`tests/test_project_facts.py` · the six S7b siblings of
`tests/test_evolution_state_integrity_v3.py` (2386 → 201) · `tests/test_evolution_redesign.py`.

**v7 delta.** Structural: none — no module here was split, merged or retyped. Line-level: three
small touches totalling +25/−14 (`reflection.py`, 33 changed lines under the T1 owner-semantics
closure and the typed control-facts pass; `consolidator.py` and `consciousness.py` a few lines
each). The campaign's main contact stays the test suite, where the evolution-integrity giant
became six themed suites and the post-task synthesis workers moved to their own owner in D01.

---

## D16 — Observability, usage accounting & cost

**Owners (8, ★1).** `observability.py` (the private forensic execution ledger: redaction, gzip
CAS blobs, call manifests, trace refs). Monetary authority: `usage_accounting.py` (append-only
physical-model-attempt ledger, reserved → dispatched → settled) → `usage_legacy_import`★ (the
one-time resumable legacy import, delta D38) with the private row helpers `_usage_rows.py`,
`_usage_rows_memo.py`, `_usage_response.py`; the substrate below it, `usage_ledger.py` (locking,
atomic append+fsync, torn-tail quarantine — a one-way seam: accounting imports it, never the
reverse); and `cost_projection.py`, the one SSOT projection of task cost every producer reads.

**Entry points.** the accounting reservation/settlement cycle around each physical attempt ·
`cost_projection` (`accounted_upper_bound_usd`) · `data/logs/` (`events.jsonl` `tools.jsonl`
`progress.jsonl` `supervisor.jsonl` — owners in `docs/PERSISTENCE_OWNERS.md`) ·
`GET /api/logs` (D11).

**Pins.** `tests/test_usage_accounting.py` · `tests/test_usage_scope_transport_v664.py` ·
`tests/test_lc2_owner_facades.py` · `tests/test_advisory_observability.py` ·
`tests/test_physical_candidate_capture.py` · `tests/test_terminal_durability_v664.py` ·
`tests/test_owner_facing_honesty.py` · `tests/test_perf_budgets.py`.

**v7 delta.** One extraction: the legacy usage import left `usage_accounting.py` for its own
leaf under the L-C2 handle. The ledger substrate/accounting-policy seam was already one-way and
stayed that way.

---

## D17 — Projects, workspaces & task results

**Owners (17, ★2).** Projects: `projects_registry.py` (the durable registry with routing fence
and generation), `project_dialogue.py`, `project_lease.py`, `project_naming.py`,
`project_sources.py`, `tools/project_journal.py`. Workspaces: `workspace_admission.py`,
`workspace_preflight.py`, `workspace_executor.py` (the local/`docker_exec` process backend),
`workspace_patch_capture`★ (the streamed `workspace.patch` pair and the git plumbing under it),
`workspace_patch_rules.py`. Headless drive isolation: `headless.py` → `headless_status`★ (the
artifact-status vocabulary it shares with patch capture). Durable results: `task_results.py`,
`task_status.py` (the effective-status SSOT with its queue-snapshot twin), `retention.py`,
`coop_checkpoint.py`.

**Entry points.** `task_results.write_task_result()` / `load_task_result()` /
`resolve_task_lineage()` · `POST /api/tasks` with a `workspace` binding (D11) ·
`journal_write` / `journal_tail_digest` · the `workspace.patch` + `workspace_patch.json` pair.

**Pins.** `tests/test_headless_extraction.py` · the six T-wave siblings of
`tests/test_headless_cli.py` (2668 → 463) · the four S7a siblings of
`tests/test_workspace_executor.py` · `tests/test_v6580_projects_foundation.py` ·
`tests/test_project_routing_v664.py` · `tests/test_v6730_origin_invariant.py` ·
`tests/test_swarm_coordination_v639.py` · `tests/test_store_task_result.py`.

**v7 delta.** Two extractions at the module ceiling: `headless.py` gave its artifact/lifecycle
vocabulary to `headless_status.py` and the patch-capture plumbing to `workspace_patch_capture.py`,
both re-exported so `project_sources`, `coop_checkpoint` and the tests keep one name.

---

## D18 — Launcher, packaging, platform & shared substrate

**Owners (10, ★2).** `launcher.py` (1474 — the desktop shell) with `launcher_bootstrap.py`,
`launcher_server_reaper`★ (POSIX same-install server discovery and root-first termination before
every boot) and `launcher_windows_runtime`★ (Windows-only pythonnet/pywebview preparation).
Packaged CLI: `packaged_cli.py`, `packaged_cli_install.py`, and the source CLI `cli.py`.
Cross-platform floor: `platform_layer.py` (1498 — process/path/locking, the descendant-enumeration
seam, the Windows Job Object ABI). Shared substrate: `utils.py` (atomic JSON, UTC timestamps,
hashes, log sanitization, subprocess helpers).

**Entry points.** `launcher.py` (spawns `server.py`) · `ouroboros` console script → `cli.py` ·
`ouroboros server --no-ui` · `packaged_cli` bridge.

**Pins.** `tests/test_launcher_server_reaper.py` · `tests/test_launcher_sync.py` ·
`tests/test_packaged_cli.py` · `tests/test_packaged_runtime_and_lifecycle.py` ·
`tests/test_build_scripts.py` · `tests/test_platform_guard.py` · `tests/test_packaging_sync.py`.

**v7 delta.** Two extractions, one of them not a v7 decision at all: the final upstream cutoff
grew `launcher.py` to 1572 lines, past this branch's 1500-line ceiling, and because the >1500
debt layer is shrink-only the Windows-only runtime preparation was extracted verbatim to keep
the merge legal. `launcher_server_reaper.py` came in with upstream.

---

## D19 — Frozen contracts (ABI)

**Owners (11, ★0).** `contracts/tool_context.py` (`ToolContextProtocol`), `contracts/tool_abi.py`
(`ToolEntryProtocol` + `GetToolsProtocol`), `contracts/api_v1.py` (WS/HTTP envelope TypedDicts,
now a compatibility re-export over `gateway/contracts.py`), `contracts/chat_id_policy.py`,
`contracts/task_contract.py`, `contracts/task_constraint.py`, `contracts/skill_manifest.py`,
`contracts/skill_payload_policy.py`, `contracts/plugin_api.py`, `contracts/schema_versions.py`.

**Entry points.** imported by every domain that crosses a boundary — D04 (tool ABI), D07
(`task_constraint` write surfaces), D11 (`api_v1`), D14 (`plugin_api`, `skill_manifest`).

**Pins.** `tests/test_contracts.py` · `tests/test_task_constraint_tools.py` ·
`tests/test_delegated_skill_payload.py` · `tests/test_extension_process_runner.py` ·
`tests/test_marketplace_api.py` · ARCHITECTURE §11 is the prose authority on what is frozen.

**v7 delta.** Structural: none — 11 modules before, 11 after. Line-level: +6/−1 in
`task_contract.py`, the only change across the whole package. A
refactor campaign that touched 132 new modules left the frozen ABI byte-stable except for five
lines, which is the strongest single statement the census makes about blast radius.

---

## Coverage accounting

| claim | check |
|---|---|
| every tracked runtime module is in exactly one domain | 411 files assigned; 0 unassigned, 0 duplicated |
| every campaign-added module appears | 132 ★ marks, matching the census's 103 `ouroboros/` + 29 `supervisor/` |
| no module was removed | 0 deletions under `ouroboros/` or `supervisor/` between `353fd974` and `740357c8` |
| domains with no structural runtime delta | D13 safety, D15 memory/evolution, D19 contracts — no splits, merges or renames; each carries a small disclosed line-level delta, stated in its section |

**Two facts this map surfaces that the module tree does not.**

1. Eighteen pre-existing runtime modules are named *nowhere* in `docs/ARCHITECTURE.md` —
   `_usage_response.py`, `_usage_rows_memo.py`, `context_mode_compat.py`,
   `evolution_fingerprint.py`, `gateway/onboarding_host.py`, `gateway/task_events.py`,
   `llm_observability.py`, `marketplace/install_specs.py`, `skill_owner_attestation.py`,
   `tools/compact_context.py`, `tools/evolution_stats.py`, `tools/knowledge.py`,
   `tools/memory_tools.py`, `tools/search.py`, `tools/shell_guards.py`,
   `tools/tool_discovery.py`, `tools/vision.py`, `version.py`. That is a standing curation gap,
   not v7 staleness (`ouroboros/tools/` was never exhaustively enumerated), and all eighteen
   have an owner row above.
2. Fourteen campaign-added modules — the `control_*`, `git_*` and `shell_*` leaf families — are
   absent from the §1 module tree but described in ARCHITECTURE prose elsewhere. They are
   enumerated above: `control_*` across D07 (scheduling, subagent spec, task results) and D08
   (events, routing, runtime), `git_*` in D10, `shell_*` in D05.

**Fifteen modules have no test file naming them directly** (`_usage_response.py`,
`_usage_rows.py`, `_usage_rows_memo.py`, `contracts/schema_versions.py`, `contracts/tool_abi.py`,
`gateway/task_events.py`, `local_model_autostart.py`, `review_evidence_refs.py`,
`review_slot_cancel.py`, `synthesis_cost_text.py`, `tools/evolution_stats.py`,
`tools/memory_tools.py`, `tools/review_revalidation.py`, `workspace_patch_rules.py`,
`supervisor/task_admission.py`). Every one is a private leaf reached through a facade the suite
does name — which is the pattern the facade inventory calls parent-internal, not a coverage
hole. It is recorded here as a fact, not a recommendation: adding a suite would be a code change,
and this document is evidence.
