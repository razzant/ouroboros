# Domain map — v7next

Generated from `ouroboros/domains.toml` by `python scripts/check_domains.py --write`. Do not edit — edit the manifest and regenerate; `tests/test_domain_manifest.py` pins byte-identity.

The manifest is the SSOT of the module→domain assignment (1:1, complete over the tracked runtime population) and pins today's factual cross-domain dependency data as baseline; the gate contract is described in `scripts/check_domains.py`.

## Index

| domain | name | modules | proposed |
|---|---|---:|---:|
| D01 | Agent core & main loop | 29 | 0 |
| D02 | LLM client, routing & providers | 34 | 0 |
| D03 | Context assembly, fit & compaction | 11 | 0 |
| D04 | Tool execution: registry, access & typed results | 20 | 0 |
| D05 | Tool surfaces: files, code, shell, media, external | 27 | 0 |
| D06 | Review stack | 63 | 0 |
| D07 | Delegation, subagents & Claudexor | 48 | 0 |
| D08 | Supervisor: queue, workers, events & runtime control | 42 | 0 |
| D09 | Cancellation, owner control & process custody | 12 | 0 |
| D10 | Git, update & release machinery | 28 | 0 |
| D11 | Gateway, server & Web UI | 49 | 0 |
| D12 | Settings & configuration | 15 | 0 |
| D13 | Safety, guards & runtime mode | 9 | 0 |
| D14 | Skills & extensions | 52 | 0 |
| D15 | Memory, knowledge, consciousness & self-evolution | 17 | 0 |
| D16 | Observability, usage accounting & cost | 11 | 0 |
| D17 | Projects, workspaces & task results | 20 | 0 |
| D18 | Launcher, packaging, platform & shared substrate | 11 | 0 |
| D19 | Frozen contracts (ABI) | 10 | 0 |
| D20 | Presence | 9 | 0 |
| **total** | | **517** | **0** |

## Dependency direction matrix (strict, pinned)

Rows may import columns (`[graph].allowed`). `·` = forbidden direction.

| ↓ imports → | D01 | D02 | D03 | D04 | D05 | D06 | D07 | D08 | D09 | D10 | D11 | D12 | D13 | D14 | D15 | D16 | D17 | D18 | D19 | D20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **D01** | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **D02** | ✓ | · | ✓ | · | · | · | · | · | · | · | · | ✓ | · | · | · | ✓ | · | ✓ | · | · |
| **D03** | ✓ | ✓ | · | ✓ | · | · | · | · | · | · | · | ✓ | · | · | ✓ | · | · | ✓ | ✓ | · |
| **D04** | · | · | · | · | ✓ | · | · | · | · | · | · | · | ✓ | · | · | · | ✓ | ✓ | ✓ | · |
| **D05** | ✓ | ✓ | · | ✓ | · | · | · | · | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | · |
| **D06** | ✓ | ✓ | ✓ | ✓ | · | · | ✓ | · | · | ✓ | · | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · |
| **D07** | ✓ | ✓ | · | ✓ | ✓ | ✓ | · | ✓ | · | · | · | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | ✓ | · |
| **D08** | ✓ | · | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · |
| **D09** | ✓ | · | · | · | · | ✓ | · | ✓ | · | · | · | · | · | · | · | · | · | ✓ | · | · |
| **D10** | · | · | · | ✓ | ✓ | ✓ | · | ✓ | · | · | · | ✓ | ✓ | · | · | · | · | ✓ | ✓ | · |
| **D11** | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| **D12** | ✓ | ✓ | ✓ | · | · | · | ✓ | · | · | · | ✓ | · | · | · | · | · | · | ✓ | · | · |
| **D13** | ✓ | ✓ | · | ✓ | · | · | · | · | · | · | · | ✓ | · | · | · | · | ✓ | ✓ | · | · |
| **D14** | · | ✓ | · | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | ✓ | · | · | · | ✓ | · | ✓ | ✓ | · |
| **D15** | ✓ | ✓ | ✓ | ✓ | · | · | · | · | · | · | · | ✓ | · | · | · | ✓ | · | ✓ | ✓ | · |
| **D16** | · | ✓ | · | · | · | ✓ | · | · | · | · | · | ✓ | · | ✓ | · | · | · | ✓ | · | · |
| **D17** | ✓ | · | · | ✓ | · | · | · | · | · | · | · | · | · | · | ✓ | ✓ | · | ✓ | ✓ | · |
| **D18** | · | · | · | · | · | · | · | · | ✓ | ✓ | · | ✓ | · | · | · | · | · | · | · | · |
| **D19** | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | ✓ | · | · |
| **D20** | · | · | · | ✓ | ✓ | · | · | · | · | · | · | · | · | ✓ | ✓ | · | ✓ | ✓ | ✓ | · |

## Cycle status

1 pinned cycle group(s) — the SCC ceiling; the target is zero. Witness-level detail lives in `docs/v7next/DOMAIN_QUOTIENT_REPORT.md`.

- group 1 (20 domains): D01 ⇄ D02 ⇄ D03 ⇄ D04 ⇄ D05 ⇄ D06 ⇄ D07 ⇄ D08 ⇄ D09 ⇄ D10 ⇄ D11 ⇄ D12 ⇄ D13 ⇄ D14 ⇄ D15 ⇄ D16 ⇄ D17 ⇄ D18 ⇄ D19 ⇄ D20

## Hidden coupling (classified out of the strict graph)

- lazy-only cross-domain pairs: **96**
  - D01->D08
  - D01->D10
  - D01->D11
  - D02->D09
  - D02->D11
  - D03->D05
  - D03->D06
  - D03->D07
  - D03->D08
  - D03->D10
  - D03->D11
  - D03->D14
  - D03->D16
  - D03->D17
  - D03->D20
  - D04->D06
  - D04->D10
  - D04->D12
  - D04->D14
  - D04->D20
  - D05->D06
  - D05->D07
  - D05->D09
  - D05->D14
  - D06->D05
  - D06->D08
  - D06->D09
  - D06->D15
  - D07->D09
  - D07->D10
  - D07->D11
  - D07->D14
  - D07->D15
  - D07->D20
  - D08->D02
  - D08->D10
  - D08->D11
  - D09->D02
  - D09->D05
  - D09->D07
  - D09->D11
  - D09->D12
  - D09->D14
  - D09->D15
  - D09->D16
  - D09->D17
  - D10->D01
  - D10->D14
  - D10->D15
  - D10->D17
  - D11->D06
  - D11->D15
  - D12->D05
  - D12->D06
  - D12->D10
  - D12->D15
  - D12->D16
  - D12->D17
  - D13->D05
  - D13->D06
  - D13->D08
  - D13->D15
  - D14->D08
  - D14->D13
  - D14->D17
  - D14->D20
  - D15->D06
  - D15->D07
  - D15->D08
  - D15->D09
  - D15->D10
  - D15->D17
  - D16->D03
  - D16->D05
  - D16->D17
  - D17->D02
  - D17->D03
  - D17->D05
  - D17->D06
  - D17->D07
  - D17->D09
  - D17->D12
  - D18->D01
  - D18->D07
  - D18->D11
  - D18->D14
  - D18->D16
  - D18->D17
  - D18->D19
  - D19->D04
  - D19->D08
  - D19->D20
  - D20->D01
  - D20->D07
  - D20->D11
  - D20->D12
- dynamic-import cross-domain pairs: **0**

## Literal-copy baseline

No function body (≥ 10 normalized lines) is shared verbatim across domains. New occurrences turn the gate red.

## Modules by domain

`*` marks a `classification=proposed` row (owner review pending).

### D01 — Agent core & main loop

- `ouroboros/_outcome_receipts.py`
- `ouroboros/_outcome_tool_errors.py`
- `ouroboros/agent.py`
- `ouroboros/agent_dispatch.py`
- `ouroboros/agent_startup_checks.py`
- `ouroboros/agent_task_pipeline.py`
- `ouroboros/deadline_utils.py`
- `ouroboros/loop.py`
- `ouroboros/loop_acceptance.py`
- `ouroboros/loop_acceptance_review.py`
- `ouroboros/loop_budget.py`
- `ouroboros/loop_delivery.py`
- `ouroboros/loop_forced_finalization.py`
- `ouroboros/loop_llm_call.py`
- `ouroboros/loop_messages.py`
- `ouroboros/loop_model_call.py`
- `ouroboros/loop_nudges.py`
- `ouroboros/loop_round_limits.py`
- `ouroboros/loop_tool_execution.py`
- `ouroboros/loop_transport.py`
- `ouroboros/mutation_attribution.py`
- `ouroboros/outcome_receipt_store.py`
- `ouroboros/outcomes.py`
- `ouroboros/owner_mailbox.py`
- `ouroboros/post_task_checkpoint.py`
- `ouroboros/post_task_synthesis.py`
- `ouroboros/synthesis_cost_text.py`
- `ouroboros/task_finalization.py`
- `ouroboros/task_pacing.py`

### D02 — LLM client, routing & providers

- `ouroboros/anthropic_native_custody.py`
- `ouroboros/fallback_cooldown.py`
- `ouroboros/llm.py`
- `ouroboros/llm_anthropic.py`
- `ouroboros/llm_attempt.py`
- `ouroboros/llm_capability_policy.py`
- `ouroboros/llm_fallback.py`
- `ouroboros/llm_gigachat.py`
- `ouroboros/llm_local.py`
- `ouroboros/llm_messages.py`
- `ouroboros/llm_observability.py`
- `ouroboros/llm_openai_compatible.py`
- `ouroboros/llm_pricing.py`
- `ouroboros/llm_probe.py`
- `ouroboros/llm_routing.py`
- `ouroboros/local_model.py`
- `ouroboros/local_model_autostart.py`
- `ouroboros/model_concurrency.py`
- `ouroboros/net_transport.py`
- `ouroboros/openai_chat_custom.py`
- `ouroboros/openai_chat_dispatch.py`
- `ouroboros/openrouter_attribution.py`
- `ouroboros/pricing.py`
- `ouroboros/provider_models.py`
- `ouroboros/reasoning_artifacts.py`
- `ouroboros/request_wire_attempt.py`
- `ouroboros/request_wire_contract.py`
- `ouroboros/request_wire_custom_validation.py`
- `ouroboros/request_wire_receipts.py`
- `ouroboros/request_wire_recovery.py`
- `ouroboros/request_wire_resolution.py`
- `ouroboros/route_spec.py`
- `ouroboros/transport_custody.py`
- `ouroboros/vision_routing.py`

### D03 — Context assembly, fit & compaction

- `ouroboros/capability_evidence.py`
- `ouroboros/context.py`
- `ouroboros/context_budget.py`
- `ouroboros/context_compaction.py`
- `ouroboros/context_fit.py`
- `ouroboros/context_health.py`
- `ouroboros/context_layout.py`
- `ouroboros/context_mode_compat.py`
- `ouroboros/context_runtime_facts.py`
- `ouroboros/main_context_authority.py`
- `ouroboros/tools/compact_context.py`

### D04 — Tool execution: registry, access & typed results

- `ouroboros/protected_artifacts.py`
- `ouroboros/tool_access.py`
- `ouroboros/tool_access_paths.py`
- `ouroboros/tool_access_roots.py`
- `ouroboros/tool_access_types.py`
- `ouroboros/tool_access_user_files.py`
- `ouroboros/tool_capabilities.py`
- `ouroboros/tool_policy.py`
- `ouroboros/tools/__init__.py`
- `ouroboros/tools/extension_dispatch.py`
- `ouroboros/tools/process_facts.py`
- `ouroboros/tools/registry.py`
- `ouroboros/tools/registry_core.py`
- `ouroboros/tools/registry_guard_process.py`
- `ouroboros/tools/registry_guards.py`
- `ouroboros/tools/tool_catalog.py`
- `ouroboros/tools/tool_context.py`
- `ouroboros/tools/tool_discovery.py`
- `ouroboros/tools/tool_resolution.py`
- `ouroboros/tools/tool_result.py`

### D05 — Tool surfaces: files, code, shell, media, external

- `ouroboros/artifacts.py`
- `ouroboros/browser_policy.py`
- `ouroboros/code_intelligence.py`
- `ouroboros/code_intelligence_architecture.py`
- `ouroboros/code_search_rg.py`
- `ouroboros/mcp_client.py`
- `ouroboros/process_interpreters.py`
- `ouroboros/tools/browser.py`
- `ouroboros/tools/core.py`
- `ouroboros/tools/core_artifacts.py`
- `ouroboros/tools/core_file_tools.py`
- `ouroboros/tools/core_secret_paths.py`
- `ouroboros/tools/edit_ops.py`
- `ouroboros/tools/health.py`
- `ouroboros/tools/media.py`
- `ouroboros/tools/owner_delivery.py`
- `ouroboros/tools/query_code.py`
- `ouroboros/tools/recent_tasks.py`
- `ouroboros/tools/search.py`
- `ouroboros/tools/services.py`
- `ouroboros/tools/shell.py`
- `ouroboros/tools/shell_audit.py`
- `ouroboros/tools/shell_effects.py`
- `ouroboros/tools/shell_outputs.py`
- `ouroboros/tools/shell_process.py`
- `ouroboros/tools/verify.py`
- `ouroboros/tools/vision.py`

### D06 — Review stack

- `ouroboros/commit_admission.py`
- `ouroboros/deep_self_review.py`
- `ouroboros/preflight_node.py`
- `ouroboros/preflight_runner.py`
- `ouroboros/review.py`
- `ouroboros/review_actor_aggregation.py`
- `ouroboros/review_custody.py`
- `ouroboros/review_cycles.py`
- `ouroboros/review_dispatch.py`
- `ouroboros/review_evidence.py`
- `ouroboros/review_evidence_refs.py`
- `ouroboros/review_evidence_sections.py`
- `ouroboros/review_execution.py`
- `ouroboros/review_execution_projection.py`
- `ouroboros/review_native_episode.py`
- `ouroboros/review_owner_custody.py`
- `ouroboros/review_projection.py`
- `ouroboros/review_records.py`
- `ouroboros/review_session_custody.py`
- `ouroboros/review_session_usage.py`
- `ouroboros/review_slot_cancel.py`
- `ouroboros/review_state.py`
- `ouroboros/review_state_custody.py`
- `ouroboros/review_state_model.py`
- `ouroboros/review_state_records.py`
- `ouroboros/review_status_projection.py`
- `ouroboros/review_substrate.py`
- `ouroboros/review_thread_continuity.py`
- `ouroboros/review_verdict.py`
- `ouroboros/review_verdict_extraction.py`
- `ouroboros/reviewer_slot_config.py`
- `ouroboros/reviewer_window.py`
- `ouroboros/task_continuation.py`
- `ouroboros/tools/claude_advisory_review.py`
- `ouroboros/tools/parallel_review.py`
- `ouroboros/tools/plan_evidence.py`
- `ouroboros/tools/plan_packet.py`
- `ouroboros/tools/plan_render.py`
- `ouroboros/tools/plan_review.py`
- `ouroboros/tools/plan_review_artifacts.py`
- `ouroboros/tools/plan_review_references.py`
- `ouroboros/tools/plan_review_runtime.py`
- `ouroboros/tools/plan_spec.py`
- `ouroboros/tools/preflight_review_prompt.py`
- `ouroboros/tools/preflight_review_run.py`
- `ouroboros/tools/review.py`
- `ouroboros/tools/review_admission.py`
- `ouroboros/tools/review_binary_context.py`
- `ouroboros/tools/review_context_atlas.py`
- `ouroboros/tools/review_file_pack.py`
- `ouroboros/tools/review_helpers.py`
- `ouroboros/tools/review_multi_model.py`
- `ouroboros/tools/review_prompt_text.py`
- `ouroboros/tools/review_response.py`
- `ouroboros/tools/review_subject.py`
- `ouroboros/tools/review_synthesis.py`
- `ouroboros/tools/scope_review.py`
- `ouroboros/tools/scope_review_budget.py`
- `ouroboros/tools/scope_review_contract.py`
- `ouroboros/tools/scope_review_pack.py`
- `ouroboros/tools/scope_review_session.py`
- `ouroboros/tools/scope_window.py`
- `ouroboros/triad_review.py`

### D07 — Delegation, subagents & Claudexor

- `ouroboros/claudexor_daemon.py`
- `ouroboros/claudexor_runtime.py`
- `ouroboros/configured_subagents.py`
- `ouroboros/delegate_containment.py`
- `ouroboros/delegate_custody.py`
- `ouroboros/delegate_custody_reconcile.py`
- `ouroboros/delegate_custody_usage.py`
- `ouroboros/delegate_evidence.py`
- `ouroboros/delegate_hold.py`
- `ouroboros/delegate_interactions.py`
- `ouroboros/delegate_output.py`
- `ouroboros/delegate_pending.py`
- `ouroboros/delegate_progress.py`
- `ouroboros/delegate_recovery.py`
- `ouroboros/delegate_registration_policy.py`
- `ouroboros/delegate_shared.py`
- `ouroboros/delegate_source_coverage.py`
- `ouroboros/delegate_start_claims.py`
- `ouroboros/delegate_start_instructions.py`
- `ouroboros/delegate_state_sweep.py`
- `ouroboros/delegate_supervision.py`
- `ouroboros/delegate_terminal.py`
- `ouroboros/depth_evidence.py`
- `ouroboros/gateways/__init__.py`
- `ouroboros/gateways/claudexor.py`
- `ouroboros/nanny_pacing.py`
- `ouroboros/subagent_bootstrap.py`
- `ouroboros/subagent_dispatch_notes.py`
- `ouroboros/subagent_messages.py`
- `ouroboros/subagent_route_health.py`
- `ouroboros/subagent_runtime.py`
- `ouroboros/subagent_work_order.py`
- `ouroboros/subagent_worktrees.py`
- `ouroboros/subagents.py`
- `ouroboros/task_tree_ledger.py`
- `ouroboros/tools/control_delegation.py`
- `ouroboros/tools/control_scheduling.py`
- `ouroboros/tools/control_subagent_spec.py`
- `ouroboros/tools/control_task_results.py`
- `ouroboros/tools/delegate.py`
- `ouroboros/tools/delegate_integration.py`
- `ouroboros/tools/delegate_payload_patch.py`
- `ouroboros/tools/delegate_terminal_evidence.py`
- `ouroboros/tools/join_ledger.py`
- `ouroboros/tools/patch_verdict.py`
- `ouroboros/tools/subagent_integration.py`
- `ouroboros/tools/subagent_integration_delegated.py`
- `ouroboros/tools/task_tree.py`

### D08 — Supervisor: queue, workers, events & runtime control

- `ouroboros/promotion_source.py`
- `ouroboros/schedule_contract.py`
- `ouroboros/tools/control.py`
- `ouroboros/tools/control_events.py`
- `ouroboros/tools/control_routing.py`
- `ouroboros/tools/control_runtime.py`
- `ouroboros/tools/followup.py`
- `supervisor/__init__.py`
- `supervisor/active_activity.py`
- `supervisor/cognitive_operations.py`
- `supervisor/event_taxonomy.py`
- `supervisor/events.py`
- `supervisor/events_budget.py`
- `supervisor/events_chat_delivery.py`
- `supervisor/events_coop_checkpoint.py`
- `supervisor/events_evolution_done.py`
- `supervisor/events_project_routing.py`
- `supervisor/events_runtime_controls.py`
- `supervisor/events_schedule_task.py`
- `supervisor/events_subagent_admission.py`
- `supervisor/events_task_done.py`
- `supervisor/events_worker_reports.py`
- `supervisor/log_addressing.py`
- `supervisor/message_bus.py`
- `supervisor/queue.py`
- `supervisor/queue_schedules.py`
- `supervisor/queue_snapshot.py`
- `supervisor/queue_timeouts.py`
- `supervisor/queue_transitions.py`
- `supervisor/schedule_time.py`
- `supervisor/state.py`
- `supervisor/subagent_task_truth.py`
- `supervisor/task_admission.py`
- `supervisor/task_dispatch.py`
- `supervisor/telemetry_events.py`
- `supervisor/worker_assignment.py`
- `supervisor/worker_chat_lane.py`
- `supervisor/worker_health.py`
- `supervisor/worker_pool_lifecycle.py`
- `supervisor/worker_process.py`
- `supervisor/worker_promotion.py`
- `supervisor/workers.py`

### D09 — Cancellation, owner control & process custody

- `ouroboros/cancel_intents.py`
- `ouroboros/owner_hurry.py`
- `ouroboros/owner_quiz.py`
- `ouroboros/process_containment.py`
- `ouroboros/process_custody.py`
- `ouroboros/server_control.py`
- `supervisor/cancel_publication.py`
- `supervisor/owner_stop.py`
- `supervisor/steering.py`
- `supervisor/task_lifecycle.py`
- `supervisor/task_reaper.py`
- `supervisor/terminal_delivery.py`

### D10 — Git, update & release machinery

- `ouroboros/repo_remotes.py`
- `ouroboros/size_ratchet_manifest.py`
- `ouroboros/tools/ci.py`
- `ouroboros/tools/commit_gate.py`
- `ouroboros/tools/git.py`
- `ouroboros/tools/git_evolution.py`
- `ouroboros/tools/git_plumbing.py`
- `ouroboros/tools/git_pr.py`
- `ouroboros/tools/git_repo_edit.py`
- `ouroboros/tools/git_review_cycle.py`
- `ouroboros/tools/git_rollback.py`
- `ouroboros/tools/git_vcs_ops.py`
- `ouroboros/tools/github.py`
- `ouroboros/tools/release_sync.py`
- `ouroboros/tools/review_revalidation.py`
- `ouroboros/version.py`
- `supervisor/git_ops.py`
- `supervisor/git_ops_remotes.py`
- `supervisor/git_ops_rescue.py`
- `supervisor/git_ops_reset.py`
- `supervisor/git_ops_updates.py`
- `supervisor/update_candidate.py`
- `supervisor/update_carriers.py`
- `supervisor/update_merge.py`
- `supervisor/update_merge_plan.py`
- `supervisor/update_merge_policy.py`
- `supervisor/update_recovery.py`
- `supervisor/update_source.py`

### D11 — Gateway, server & Web UI

- `ouroboros/client_surface.py`
- `ouroboros/gateway/__init__.py`
- `ouroboros/gateway/_helpers.py`
- `ouroboros/gateway/claudexor_accounts.py`
- `ouroboros/gateway/claudexor_quota.py`
- `ouroboros/gateway/contracts.py`
- `ouroboros/gateway/control.py`
- `ouroboros/gateway/cost_breakdown.py`
- `ouroboros/gateway/endpoint_index.py`
- `ouroboros/gateway/extension_receipts.py`
- `ouroboros/gateway/extensions.py`
- `ouroboros/gateway/files.py`
- `ouroboros/gateway/history.py`
- `ouroboros/gateway/host_service.py`
- `ouroboros/gateway/logs.py`
- `ouroboros/gateway/marketplace.py`
- `ouroboros/gateway/mcp.py`
- `ouroboros/gateway/models.py`
- `ouroboros/gateway/onboarding.py`
- `ouroboros/gateway/onboarding_host.py`
- `ouroboros/gateway/owner_settings.py`
- `ouroboros/gateway/presence_settings.py`
- `ouroboros/gateway/projects.py`
- `ouroboros/gateway/router.py`
- `ouroboros/gateway/routing_decision.py`
- `ouroboros/gateway/schedules.py`
- `ouroboros/gateway/schema.py`
- `ouroboros/gateway/settings.py`
- `ouroboros/gateway/skill_publish.py`
- `ouroboros/gateway/state.py`
- `ouroboros/gateway/task_decision.py`
- `ouroboros/gateway/task_events.py`
- `ouroboros/gateway/task_hurry.py`
- `ouroboros/gateway/task_list_scan.py`
- `ouroboros/gateway/tasks.py`
- `ouroboros/gateway/ui_preferences.py`
- `ouroboros/gateway/widgets.py`
- `ouroboros/gateway/ws.py`
- `ouroboros/server_auth.py`
- `ouroboros/server_entrypoint.py`
- `ouroboros/server_liveness.py`
- `ouroboros/server_maintenance.py`
- `ouroboros/server_owner_routing.py`
- `ouroboros/server_process.py`
- `ouroboros/server_restart.py`
- `ouroboros/server_routing_context.py`
- `ouroboros/server_runtime.py`
- `ouroboros/server_web.py`
- `server.py`

### D12 — Settings & configuration

- `ouroboros/colab_bootstrap.py`
- `ouroboros/config.py`
- `ouroboros/launcher_onboarding.py`
- `ouroboros/model_slots.py`
- `ouroboros/onboarding_wizard.py`
- `ouroboros/review_model_routes.py`
- `ouroboros/runtime_limits.py`
- `ouroboros/secret_masking.py`
- `ouroboros/settings_defaults.py`
- `ouroboros/settings_integrity.py`
- `ouroboros/settings_scales.py`
- `ouroboros/settings_setup_contract.py`
- `ouroboros/subscription_install_presets.py`
- `ouroboros/update_channels.py`
- `ouroboros/update_letter.py`

### D13 — Safety, guards & runtime mode

- `ouroboros/argv_budget.py`
- `ouroboros/credential_shapes.py`
- `ouroboros/git_shell_policy.py`
- `ouroboros/runtime_mode_policy.py`
- `ouroboros/safety.py`
- `ouroboros/shell_parse.py`
- `ouroboros/tools/deliverables_shell.py`
- `ouroboros/tools/shell_guards.py`
- `ouroboros/tools/write_shape.py`

### D14 — Skills & extensions

- `ouroboros/betterleaks_runtime.py`
- `ouroboros/event_bus.py`
- `ouroboros/extension_child_catalog.py`
- `ouroboros/extension_companion.py`
- `ouroboros/extension_health.py`
- `ouroboros/extension_import_staging.py`
- `ouroboros/extension_isolated_deps.py`
- `ouroboros/extension_liveness.py`
- `ouroboros/extension_loader.py`
- `ouroboros/extension_plugin_api.py`
- `ouroboros/extension_process_runner.py`
- `ouroboros/extension_reconcile_queue.py`
- `ouroboros/extension_registry_state.py`
- `ouroboros/extension_surface_names.py`
- `ouroboros/extension_ui_validation.py`
- `ouroboros/marketplace/__init__.py`
- `ouroboros/marketplace/adapter.py`
- `ouroboros/marketplace/clawhub.py`
- `ouroboros/marketplace/fetcher.py`
- `ouroboros/marketplace/install.py`
- `ouroboros/marketplace/install_specs.py`
- `ouroboros/marketplace/isolated_deps.py`
- `ouroboros/marketplace/ouroboroshub.py`
- `ouroboros/marketplace/provenance.py`
- `ouroboros/skill_dependencies.py`
- `ouroboros/skill_lifecycle_queue.py`
- `ouroboros/skill_loader.py`
- `ouroboros/skill_owner_attestation.py`
- `ouroboros/skill_payload_binding.py`
- `ouroboros/skill_publish_eligibility.py`
- `ouroboros/skill_publish_github.py`
- `ouroboros/skill_publish_result.py`
- `ouroboros/skill_publish_scanner.py`
- `ouroboros/skill_publish_snapshot.py`
- `ouroboros/skill_readiness.py`
- `ouroboros/skill_repair_admission.py`
- `ouroboros/skill_review.py`
- `ouroboros/skill_review_cycles.py`
- `ouroboros/skill_review_history.py`
- `ouroboros/skill_review_output.py`
- `ouroboros/skill_review_packs.py`
- `ouroboros/skill_review_passes.py`
- `ouroboros/skill_review_prompt.py`
- `ouroboros/skill_review_rebuttals.py`
- `ouroboros/skill_review_runner.py`
- `ouroboros/skill_review_status.py`
- `ouroboros/skill_review_usage.py`
- `ouroboros/skill_token.py`
- `ouroboros/skill_uninstall_state.py`
- `ouroboros/tools/skill_exec.py`
- `ouroboros/tools/skill_preflight.py`
- `ouroboros/tools/skill_publish.py`

### D15 — Memory, knowledge, consciousness & self-evolution

- `ouroboros/consciousness.py`
- `ouroboros/consolidator.py`
- `ouroboros/dialogue_provenance.py`
- `ouroboros/evolution_checkpoints.py`
- `ouroboros/evolution_fingerprint.py`
- `ouroboros/improvement_backlog.py`
- `ouroboros/memory.py`
- `ouroboros/memory_journal_compaction.py`
- `ouroboros/post_task_evolution.py`
- `ouroboros/project_facts.py`
- `ouroboros/reflection.py`
- `ouroboros/semantic_dedup.py`
- `ouroboros/tools/evolution_stats.py`
- `ouroboros/tools/knowledge.py`
- `ouroboros/tools/memory_tools.py`
- `ouroboros/world_profiler.py`
- `supervisor/evolution_lifecycle.py`

### D16 — Observability, usage accounting & cost

- `ouroboros/_usage_cache_splits.py`
- `ouroboros/_usage_response.py`
- `ouroboros/_usage_rows.py`
- `ouroboros/_usage_rows_memo.py`
- `ouroboros/cost_projection.py`
- `ouroboros/model_send_seal.py`
- `ouroboros/observability.py`
- `ouroboros/usage_accounting.py`
- `ouroboros/usage_compaction.py`
- `ouroboros/usage_ledger.py`
- `ouroboros/usage_legacy_import.py`

### D17 — Projects, workspaces & task results

- `ouroboros/coop_checkpoint.py`
- `ouroboros/deliverables_paths.py`
- `ouroboros/headless.py`
- `ouroboros/headless_status.py`
- `ouroboros/project_dialogue.py`
- `ouroboros/project_lease.py`
- `ouroboros/project_naming.py`
- `ouroboros/project_sources.py`
- `ouroboros/projects_registry.py`
- `ouroboros/retention.py`
- `ouroboros/routing_wait.py`
- `ouroboros/task_result_schema.py`
- `ouroboros/task_results.py`
- `ouroboros/task_status.py`
- `ouroboros/tools/project_journal.py`
- `ouroboros/workspace_admission.py`
- `ouroboros/workspace_executor.py`
- `ouroboros/workspace_patch_capture.py`
- `ouroboros/workspace_patch_rules.py`
- `ouroboros/workspace_preflight.py`

### D18 — Launcher, packaging, platform & shared substrate

- `launcher.py`
- `ouroboros/__init__.py`
- `ouroboros/cli.py`
- `ouroboros/launcher_bootstrap.py`
- `ouroboros/launcher_server_reaper.py`
- `ouroboros/launcher_windows_runtime.py`
- `ouroboros/node_runtime.py`
- `ouroboros/packaged_cli.py`
- `ouroboros/packaged_cli_install.py`
- `ouroboros/platform_layer.py`
- `ouroboros/utils.py`

### D19 — Frozen contracts (ABI)

- `ouroboros/contracts/__init__.py`
- `ouroboros/contracts/chat_id_policy.py`
- `ouroboros/contracts/plugin_api.py`
- `ouroboros/contracts/schema_versions.py`
- `ouroboros/contracts/skill_manifest.py`
- `ouroboros/contracts/skill_payload_policy.py`
- `ouroboros/contracts/task_constraint.py`
- `ouroboros/contracts/task_contract.py`
- `ouroboros/contracts/tool_abi.py`
- `ouroboros/contracts/tool_context.py`

### D20 — Presence

- `ouroboros/presence_admission.py`
- `ouroboros/presence_authority.py`
- `ouroboros/presence_bindings.py`
- `ouroboros/presence_capabilities.py`
- `ouroboros/presence_context.py`
- `ouroboros/presence_profile.py`
- `ouroboros/presence_runner.py`
- `ouroboros/presence_runtime.py`
- `ouroboros/tools/presence.py`
