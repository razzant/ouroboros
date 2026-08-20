# Test split / delete disposition — v7

The v7 campaign took the test suite from 435 to 664 `tests/test_*.py` files. A count that
large invites one question and deserves a mechanical answer: **where did each new file come
from, and did anything get deleted quietly?** This document answers it per file.

It is a disposition table, not an argument. Every row is derived from Git and from the
migration ledger; nothing here is asserted from memory.

## Method

- Population: `git diff --name-status 353fd974..HEAD -- tests/ web/tests/` on the frozen
  candidate. `353fd974` is the pre-v7 reference the campaign's counts are taken against.
- Bucket **(a) split from a named giant** — the added path appears as a *destination* of a
  `MIGRATION_v7.md` row whose *source* is a test file. The ledger is the authority; the
  `tests/_v7_ledger_inventories.py` dicts (`s7a_`/`s7b_`/`w5_`/`ts1_`/`ts2_test_split_symbols_by_owner`)
  carry the same maps as data for the ledger-membership test.
- Bucket **(c) upstream-adopted** — the commit that added the file is reachable from
  `8028f1df` (the final upstream sync cutoff, `scripts/v7_migration.py::MERGE_BASE_SHA`) but
  not from `353fd974`, i.e. it is upstream development the branch merged in, not campaign work.
- Bucket **(b) new coverage for a campaign change** — everything else. The row names the
  commit whose change the file covers.
- Buckets are exclusive and (a) wins ties. No file fell into two buckets: the (a)∩(c)
  intersection is empty.

## 1. Headline

| file kind | `353fd974` | `740357c8` | added | deleted |
|---|---:|---:|---:|---:|
| `tests/test_*.py` | 435 | 664 | 230 | 1 |
| `tests/` helper modules (`_*.py`, corpora, fixtures modules) | 6 | 37 | 31 | 0 |
| `tests/` fixture data (`.json`) | 1 | 12 | 11 | 0 |
| `web/tests/*.test.js` | 26 | 50 | 24 | 0 |
| `web/tests/` helpers and fixtures | 3 | 5 | 2 | 0 |
| **total** | **471** | **768** | **298** | **1** |

| bucket | files |
|---|---:|
| (a) split from a named giant | 180 |
| (b) new coverage for a campaign change | 93 |
| (c) upstream-adopted | 25 |
| **total added** | **298** |

**The census's "deleted 0" is wrong by one.** `tests/test_planning_swarm_adaptive_wait.py`
was deleted; §5 gives its disposition. It is not a campaign deletion.

## 2. (a) Split from a named giant — 180 files

Thirty-seven test files gave symbols to a new sibling. Thirty-three of them are base-tree
giants the campaign broke up — thirty-two Python suites plus one web suite,
`web/tests/harness_accounts.test.js` (1882 lines at the base); three
(`test_plan_review_engine.py`, `test_plan_review_epoch.py`,
`test_delegation_account_pin.py`) were born upstream and split during adoption; one
(`test_advisory_delegated_route.py`) is not a giant at all — it contributed two symbols to a
shared helper another split created.

The table below is (source giant → added owners). It contains **185 source→destination pairs
over 180 distinct files**: five owners received symbols from two sources
(`tests/_delegated_transport_shared.py`, `tests/test_delegated_run_accounting.py`,
`tests/_review_session_route_shared.py`, `tests/_plan_review_engine_shared.py`,
`tests/test_plan_review_health.py`). Owners already present in the base tree
(`tests/_shared.py`, `tests/fixtures_mock_llm.py`, `tests/test_tool_execution_classification.py`)
received moved symbols too and are *not* counted here — they are not added files.


| source giant | lines base -> candidate | added owners | ledger rows | added owner files |
|---|---:|---:|---:|---|
| `tests/test_delegated_subagent_transport.py` | 6178 -> 366 | 11 | 173 | `_delegated_transport_shared.py` `test_delegated_cancellation_settlement.py` `test_delegated_executor_axis.py` `test_delegated_reconciliation.py` `test_delegated_result_delivery.py` `test_delegated_run_accounting.py` `test_delegated_run_containment.py` `test_delegated_run_custody.py` `test_delegated_run_profile.py` `test_delegated_wait_timeline.py` `test_delegated_wait_window.py` |
| `tests/test_devtools_benchmarks.py` | 6913 -> 536 | 10 | 217 | `_devtools_benchmarks_shared.py` `test_devtools_gaia.py` `test_devtools_harbor_jobs.py` `test_devtools_launcher_gate.py` `test_devtools_launcher_outcomes.py` `test_devtools_osworld.py` `test_devtools_programbench.py` `test_devtools_runtime_attestation.py` `test_devtools_swe_pro.py` `test_devtools_terminal_bench.py` |
| `tests/test_cancel_intents_phase_a.py` | 2400 -> 241 | 8 | 74 | `_cancel_intents_shared.py` `test_cancel_cascade_and_disclosure.py` `test_cancel_custody.py` `test_cancel_live_kill_path.py` `test_cancel_pending_outbox.py` `test_cancel_queue_integration.py` `test_cancel_task_done_validation.py` `test_cancel_terminal_delivery.py` |
| `tests/test_preflight_runner.py` | 4202 -> 451 | 8 | 113 | `_preflight_runner_shared.py` `test_preflight_candidate_capture.py` `test_preflight_commit_gate.py` `test_preflight_diagnosis.py` `test_preflight_hermetic_runs.py` `test_preflight_pass_orchestration.py` `test_preflight_process_containment.py` `test_preflight_process_reaping.py` |
| `tests/test_skill_exec.py` | 1899 -> 642 | 7 | 54 | `_skill_exec_shared.py` `test_registry_guard_process.py` `test_skill_exec_registry_surface.py` `test_skill_heal_context.py` `test_skill_preflight.py` `test_skill_review_lifecycle.py` `test_skill_toggle.py` |
| `tests/test_delivery_forced_finalization.py` | 1889 -> 483 | 6 | 39 | `_delivery_forced_shared.py` `test_delivery_control_latch.py` `test_delivery_forced_absorption_acceptance.py` `test_delivery_forced_acceptance_bypass.py` `test_delivery_forced_owner_refresh.py` `test_delivery_forced_suffix_binding.py` |
| `tests/test_evolution_state_integrity_v3.py` | 2386 -> 201 | 6 | 63 | `_evolution_state_shared.py` `test_evolution_commit_receipt.py` `test_evolution_publication.py` `test_evolution_restart_claims.py` `test_evolution_scheduler.py` `test_evolution_terminal_events.py` |
| `tests/test_headless_cli.py` | 2668 -> 463 | 6 | 70 | `_headless_cli_shared.py` `test_headless_task_api.py` `test_headless_task_artifacts.py` `test_headless_task_events.py` `test_headless_workspace_patch.py` `test_headless_workspace_shell.py` |
| `tests/test_runtime_mode_core.py` | 1647 -> 262 | 6 | 78 | `_runtime_mode_core_shared.py` `test_runtime_mode_registry_gating.py` `test_runtime_mode_repair_confinement.py` `test_runtime_mode_shell_gating.py` `test_runtime_mode_skill_payload.py` `test_runtime_mode_surfaces.py` |
| `tests/test_runtime_mode_elevation.py` | 2141 -> 350 | 6 | 71 | `_runtime_mode_elevation_shared.py` `test_runtime_mode_authorship.py` `test_runtime_mode_data_write.py` `test_runtime_mode_launcher_bridges.py` `test_runtime_mode_owner_endpoints.py` `test_runtime_mode_write_guards.py` |
| `tests/test_skill_review.py` | 1800 -> 391 | 6 | 65 | `_skill_review_shared.py` `test_skill_advisory_pre_review.py` `test_skill_review_aggregation.py` `test_skill_review_packs.py` `test_skill_review_rebuttals.py` `test_skill_review_rendering.py` |
| `tests/test_task_status_flow.py` | 2892 -> 354 | 6 | 62 | `test_task_status_duplicates.py` `test_task_status_results.py` `test_task_status_scheduling.py` `test_task_status_subagent_admission.py` `test_task_status_subagent_lifecycle.py` `test_task_status_wait_tools.py` |
| `tests/test_ui_smoke_playwright.py` | 3819 -> 330 | 6 | 32 | `_ui_smoke_shared.py` `test_ui_smoke_cards.py` `test_ui_smoke_chat.py` `test_ui_smoke_login.py` `test_ui_smoke_review_controls.py` `test_ui_smoke_widgets.py` |
| `tests/test_agent_task_pipeline.py` | 1641 -> 422 | 5 | 34 | `test_collect_review_evidence.py` `test_post_task_reflection.py` `test_root_post_task_synthesis.py` `test_store_task_result.py` `test_task_summary.py` |
| `tests/test_context.py` | 1625 -> 376 | 5 | 39 | `_context_shared.py` `test_context_advisory_review.py` `test_context_drive_state.py` `test_context_memory.py` `test_context_runtime_section.py` |
| `tests/test_extension_loader.py` | 1703 -> 359 | 5 | 45 | `_extension_loader_shared.py` `test_extension_plugin_api.py` `test_extension_reconcile.py` `test_extension_reconcile_queue.py` `test_extension_reload_all.py` |
| `tests/test_extensions_api.py` | 1667 -> 374 | 5 | 37 | `_extensions_api_shared.py` `test_extensions_dispatcher.py` `test_extensions_skill_grants.py` `test_extensions_skill_lifecycle.py` `test_extensions_websocket.py` |
| `tests/test_git_review_pipeline.py` | 2092 -> 335 | 5 | 19 | `_git_review_pipeline_shared.py` `test_git_review_advisory_skip_tests.py` `test_git_review_bypass_gate.py` `test_git_review_enforcement.py` `test_git_review_preflight_gate.py` |
| `tests/test_osworld_cu_bridge.py` | 2359 -> 444 | 5 | 74 | `_osworld_cu_bridge_shared.py` `test_osworld_cu_bridge_claims.py` `test_osworld_cu_bridge_gate.py` `test_osworld_cu_bridge_prompts.py` `test_osworld_cu_bridge_provenance.py` |
| `tests/test_promote_chat_flow.py` | 1811 -> 653 | 5 | 43 | `_promote_chat_shared.py` `test_chat_steering.py` `test_project_chat_routing.py` `test_project_task_binding.py` `test_promote_workspace_provisioning.py` |
| `tests/test_scope_review.py` | 3767 -> 969 | 5 | 62 | `_scope_review_shared.py` `test_scope_review_ladder.py` `test_scope_review_pack.py` `test_scope_review_slots.py` `test_scope_review_wiring.py` |
| `tests/test_skill_loader.py` | 1725 -> 520 | 5 | 42 | `_skill_loader_shared.py` `test_skill_availability.py` `test_skill_content_hash.py` `test_skill_grants.py` `test_skill_state_persistence.py` |
| `tests/test_claudexor_owned_daemon.py` | 2110 -> 1136 | 4 | 46 | `test_claudexor_executor_frame.py` `test_claudexor_login_accounts.py` `test_claudexor_login_jobs.py` `test_claudexor_status_payload.py` |
| `tests/test_delegated_run_isolation.py` | 1593 -> 503 | 4 | 17 | `_delegated_run_isolation_shared.py` `test_delegated_run_apply_intent.py` `test_delegated_run_capture_honesty.py` `test_delegated_run_reconciliation_capture.py` |
| `tests/test_git_ops_recovery.py` | 1878 -> 292 | 4 | 42 | `_git_ops_recovery_shared.py` `test_git_ops_checkout_reset.py` `test_git_ops_managed_update.py` `test_git_ops_rescue_snapshot.py` |
| `tests/test_loop_misc.py` | 1803 -> 264 | 4 | 30 | `test_loop_acceptance_gate.py` `test_loop_image_attach.py` `test_loop_skill_finalization.py` `test_run_llm_loop.py` |
| `tests/test_review_agent_session_route.py` | 2329 -> 692 | 4 | 72 | `_review_session_route_shared.py` `test_review_session_delivery.py` `test_review_session_poller.py` `test_review_session_scope_wiring.py` |
| `tests/test_review_substrate_v2.py` | 2014 -> 695 | 4 | 37 | `_review_substrate_shared.py` `test_review_substrate_acceptance.py` `test_review_substrate_actor_truth.py` `test_review_substrate_prompts.py` |
| `tests/test_tool_capabilities.py` | 1793 -> 442 | 4 | 42 | `test_tool_capabilities_black_box_policy.py` `test_tool_capabilities_readonly_subagent.py` `test_tool_capabilities_search_code.py` `test_tool_capabilities_subagent_scheduling.py` |
| `tests/test_workspace_executor.py` | 1641 -> 541 | 4 | 28 | `_workspace_executor_shared.py` `test_workspace_executor_admission.py` `test_workspace_executor_docker.py` `test_workspace_executor_services.py` |
| `web/tests/harness_accounts.test.js` | 1882 -> 695 | 4 | 10 | `harness_accounts_cards.test.js` `harness_accounts_custody.test.js` `harness_accounts_helpers.js` `harness_accounts_panel.test.js` |
| `tests/test_model_slot_role_model.py` | 1670 -> 318 | 3 | 46 | `_model_slot_role_shared.py` `test_model_slot_dispatch.py` `test_model_slot_scheduling.py` |
| `tests/test_delegation_account_pin.py` | (new upstream) -> 296 | 2 | 2 | `_delegated_transport_shared.py` `test_delegated_run_accounting.py` |
| `tests/test_plan_review_engine.py` | (new upstream) -> 1283 | 2 | 20 | `_plan_review_engine_shared.py` `test_plan_review_health.py` |
| `tests/test_plan_review_epoch.py` | (new upstream) -> 447 | 2 | 9 | `_plan_review_engine_shared.py` `test_plan_review_health.py` |
| `tests/test_review_prompt_caching.py` | 1627 -> 885 | 2 | 35 | `_review_prompt_caching_shared.py` `test_review_economics.py` |
| `tests/test_advisory_delegated_route.py` | 373 -> 373 | 1 | 2 | `_review_session_route_shared.py` |


No giant was deleted by its split. Every source in the table still exists and still holds
the residue the split did not move — the largest, `tests/test_delegated_subagent_transport.py`,
went 6178 → 366 lines and kept its name. That is the shape of the whole bucket: the split
made siblings, not replacements.

## 3. (b) New coverage for a campaign change — 93 files

These files did not come out of a giant. Each one covers a change the campaign made, named
by the commit that introduced both. Six sub-classes:

| sub-class | files | what it pins |
|---|---:|---|
| owner-split pin | 49 | a runtime module's split: the facade re-exports the same objects the leaves define, or the moved bytes are byte-identical |
| typed tool-result cutover | 11 | the §4.3.3 typed `ToolResult`/`ToolCodeSpec` conversion, per tool family |
| new campaign behaviour | 11 | behaviour the campaign shipped (carrier resolver, git_ops roots, settings seam, E2E cancellation, port sweep, event taxonomy, PluginAPI matrix) |
| provider-route golden | 10 | every `llm.py` provider route, replayable, so the ten-leaf split cannot change a wire payload |
| S6 characterization (no fix) | 6 | a cancellation/containment residual pinned as it is, deliberately without fixing it |
| ledger/evidence gate | 6 | the migration ledger's own membership, verbatim-byte and prologue-evidence gates |


| sub-class | campaign change (commit subject) | commit | added files |
|---|---|---|---|
| S6 characterization (no fix) | v7(S6): pin what a corrupt cancel-intent projection does to the claim fence | `50dda4ca` | `tests/test_cancel_intent_corruption_s6.py` |
| S6 characterization (no fix) | v7(S6): pin what an unreadable private-snapshot registry costs | `a7ec1aa7` | `tests/test_subagent_worktree_registry_s6.py` |
| S6 characterization (no fix) | v7(S6/C5): owner-stop fences survive a concurrent cascade's prune — no fix | `2d4e4267` | `tests/test_owner_stop_fences_s6.py` |
| S6 characterization (no fix) | v7(S6/C7-C10): structural inventories of the cancellation protocol's owners | `9140e995` | `tests/test_cancel_protocol_inventory_s6.py` |
| S6 characterization (no fix) | v7(S6/O5): make the daemon-token containment claim falsifiable | `a03786dd` | `tests/test_daemon_token_containment_s6.py` |
| S6 characterization (no fix) | v7(S6/R1): pin upstream's chat-less cascade residual without fixing it | `be78ecfb` | `tests/test_cascade_chatless_residual_s6.py` |
| ledger/evidence gate | test(v7): freeze prologue evidence contracts | `deb2617e` | `tests/fixtures/v7_prologue_baseline.json` `tests/test_v7_prologue_evidence.py` |
| ledger/evidence gate | test(architecture): gate top-level import cycles | `2f085a4b` | `tests/test_top_level_import_graph.py` |
| ledger/evidence gate | test(v7): move the ledger-membership test to its own module | `6a8e7a5d` | `tests/test_v7_migration_ledger.py` |
| ledger/evidence gate | v7(ledger): pin verbatim moves to the bytes they claim | `4784b83f` | `tests/test_v7_verbatim_moves.py` |
| ledger/evidence gate | v7: the ledger test's split inventories move to a data sibling | `c484c535` | `tests/_v7_ledger_inventories.py` |
| new campaign behaviour | v7 S6b: E2E cancellation/hurry scenarios E1-E12 on an isolated server | `2b49cbe1` | `tests/fixtures_e2e_cancellation.py` `tests/test_e2e_cancellation_scenarios.py` |
| new campaign behaviour | feat(scripts): carrier_rebase_helper — span-substitution 'ours' for carrier conflicts of tactical rebases | `fc5128ac` | `tests/test_carrier_rebase_helper.py` |
| new campaign behaviour | feat(update): carrier-aware update engine — span-substitution resolution at all three insertion points (D34) | `dd49ca8a` | `tests/test_update_carriers.py` |
| new campaign behaviour | fix(supervisor): git_ops pre-init roots follow the configured data drive | `42ebc8b1` | `tests/test_git_ops_default_roots.py` |
| new campaign behaviour | test(v7 S1): characterize the settings read path before the normalization seam | `aecf7cde` | `tests/test_settings_read_seam.py` |
| new campaign behaviour | test(v7 S1): pin which environment values may become settings-file content | `757ecbb3` | `tests/test_settings_env_on_disk.py` |
| new campaign behaviour | v7(L): the recovery ladder stops consuming a typed policy refusal | `ef3493f0` | `tests/test_llm_typed_policy_refusal.py` |
| new campaign behaviour | v7(S2): characterize which ports a panic sweeps before changing how it learns them | `c22e3ad9` | `tests/test_panic_stop_port_sweep.py` |
| new campaign behaviour | v7(S3): declare what answers every event, and prove both ends agree | `04cc28a5` | `tests/test_event_taxonomy.py` |
| new campaign behaviour | v7: characterization matrix for the PluginAPI load/dispatch/unload lifecycle | `af3e9428` | `tests/test_extension_plugin_api_matrix.py` |
| owner-split pin | refactor(agent): give the delegated-child dispatch seam its own owner (agent_dispatch) | `5ec912d6` | `tests/test_lc2_owner_facades.py` |
| owner-split pin | refactor(custody): give delegated-run reconciliation its own owner (delegate_custody_reconcile) | `effe0b5e` | `tests/test_delegate_owner_facades.py` |
| owner-split pin | refactor(git_ops): give the personal remote and push surface its own owner (git_ops_remotes) | `55e08f43` | `tests/test_git_ops_owner_facades.py` |
| owner-split pin | refactor(loop): give owner-message text plumbing its own owner (loop_messages) | `1d6c3173` | `tests/test_loop_owner_facades.py` |
| owner-split pin | refactor(review): give the multi-model review delivery its own owner (review_multi_model) | `492a9f56` | `tests/test_review_owner_facades.py` |
| owner-split pin | refactor(tools): enforce immutable catalog authority | `c1cb6dc4` | `tests/test_tool_catalog.py` |
| owner-split pin | refactor(tools): extract registry core | `cb2d1d3e` | `tests/test_registry_core.py` |
| owner-split pin | refactor(tools): extract tool context and catalog owners | `4a0f97eb` | `tests/test_tool_owner_facades.py` |
| owner-split pin | refactor(tools): split core resource owners | `24ae3c67` | `tests/test_core_extraction.py` |
| owner-split pin | refactor(update): extract the merge-planning/materialization cluster into supervisor/update_merge_plan.py | `3a517e88` | `tests/test_update_merge_owner_facade.py` |
| owner-split pin | refactor(web): extract chat primitives | `eed4cd0d` | `web/tests/chat_facade.test.js` |
| owner-split pin | refactor(web): give attachment staging its own owner | `8033511d` | `web/tests/chat_attachments.test.js` |
| owner-split pin | refactor(web): give delivered-document bubbles their own owner | `91dbd8c0` | `web/tests/document_bubble.test.js` |
| owner-split pin | refactor(web): give history hydration and the feed mount their own owner | `cda9ecc1` | `web/tests/chat_history_sync.test.js` |
| owner-split pin | refactor(web): give live-card presentation its own owner | `fe754dd4` | `web/tests/live_card_view.test.js` |
| owner-split pin | refactor(web): give message identity and presentation their own owner | `9e260bf6` | `web/tests/message_identity.test.js` |
| owner-split pin | refactor(web): give photo and video bubbles their own owner | `9ab805ef` | `web/tests/chat_media_bubbles.test.js` |
| owner-split pin | refactor(web): give routing acknowledgements their own owner | `e65bc678` | `web/tests/message_annotations.test.js` |
| owner-split pin | refactor(web): give subagent card routing its own owner | `26bd5b5b` | `web/tests/subagent_routing.test.js` |
| owner-split pin | refactor(web): give the composer row and its viewport reserve their own owner | `31bc2fcf` | `web/tests/composer.test.js` |
| owner-split pin | refactor(web): give the live-card owner actions their own owner | `0e6d0dd1` | `web/tests/card_actions.test.js` |
| owner-split pin | refactor(web): give the live-card store its own owner | `b47dbf07` | `web/tests/chat_live_cards.test.js` |
| owner-split pin | refactor(web): give the per-task UI ledger its own owner | `de3742ec` | `web/tests/task_ui_state.test.js` |
| owner-split pin | refactor(web): give the remaining chat primitives their domain owners | `5284d2e4` | `web/tests/chat_primitives.test.js` |
| owner-split pin | refactor(web): give the task-frame router its own owner | `c37b12e3` | `web/tests/chat_task_frames.test.js` |
| owner-split pin | refactor(web): give visible-timeline anchoring its own owner | `a0c57b11` | `web/tests/timeline_anchor.test.js` |
| owner-split pin | v7(L): split ouroboros/llm.py into ten owner leaves by verbatim extraction | `7a2af4dc` | `tests/test_llm_extraction.py` |
| owner-split pin | v7(L-A): split review_state.py into record, ledger and store owners | `adad51ea` | `tests/test_review_state_extraction.py` |
| owner-split pin | v7(L-A): split tools/review_helpers.py into vocabulary and file-pack owners | `9a203bb4` | `tests/test_review_helpers_extraction.py` |
| owner-split pin | v7(L-A): split tools/scope_review.py into budget and pack owners | `f744505c` | `tests/test_scope_review_extraction.py` |
| owner-split pin | v7(L2b): extract the acceptance evidence sections out of review_evidence.py | `c48d36a5` | `tests/test_review_evidence_extraction.py` |
| owner-split pin | v7(L2b): split review_substrate.py into three owner leaves by verbatim extraction | `2351eebc` | `tests/test_review_substrate_extraction.py` |
| owner-split pin | v7(L2b): split skill_review.py into four owner leaves by verbatim extraction | `d0702141` | `tests/test_skill_review_extraction.py` |
| owner-split pin | v7(S1): split config.py into five owner leaves by verbatim extraction | `3ff2d150` | `tests/test_config_extraction.py` |
| owner-split pin | v7(S2): split server.py into six owner leaves by verbatim extraction | `e139d59e` | `tests/test_server_extraction.py` |
| owner-split pin | v7(S3): give cancellation custody its own owner module | `e3c107bd` | `tests/test_cancel_custody_extraction.py` |
| owner-split pin | v7(S3): separate what runs inside a worker from the pool that spawns it | `7e846b7e` | `tests/test_worker_process_extraction.py` |
| owner-split pin | v7(S3): split supervisor/events.py into ten handler-family owners | `bb2eb9c3` | `tests/test_events_extraction.py` |
| owner-split pin | v7(S3b): split supervisor/queue.py by module handle (D10) | `738648cf` | `tests/test_module_handle_extraction.py` |
| owner-split pin | v7(T): split headless.py into two owner leaves by verbatim extraction | `359aaa0d` | `tests/test_headless_extraction.py` |
| owner-split pin | v7(T): split tool_access.py into four owner leaves by verbatim extraction | `c01edbe5` | `tests/test_tool_access_extraction.py` |
| owner-split pin | v7(T): split tools/git.py into five owner leaves by verbatim extraction | `306f8827` | `tests/test_git_extraction.py` |
| owner-split pin | v7(T): split tools/shell.py into three owner leaves by verbatim extraction | `7e9c70ed` | `tests/test_shell_extraction.py` |
| owner-split pin | v7(W): split OSWorld run_cu_bridge_agent.py into five owner leaves | `d60eead1` | `tests/test_osworld_cu_bridge_extraction.py` |
| owner-split pin | v7(W): split OSWorld run_step_agent.py into five owner leaves | `aa61afcf` | `tests/test_osworld_step_agent_extraction.py` |
| owner-split pin | v7(W): split skills/unix_computer_use/plugin.py into three skill leaves | `64b8cfa2` | `tests/test_unix_computer_use_extraction.py` |
| owner-split pin | v7(W): split web/tests/harness_accounts.test.js into four sibling suites | `f15c8922` | `tests/test_harness_accounts_test_split.py` |
| owner-split pin | v7: control tools split into owner leaves (verbatim extraction) | `5a78c471` | `tests/test_control_extraction.py` |
| owner-split pin | v7: extension_loader split into owner leaves (verbatim extraction) | `44482dec` | `tests/test_extension_loader_extraction.py` |
| provider-route golden | v7(L): pin every llm.py provider route with replayable golden fixtures | `ca065bdf` | `tests/fixtures/llm_golden/anthropic_native.json` `tests/fixtures/llm_golden/aux_routes.json` `tests/fixtures/llm_golden/compatible_lanes.json` `tests/fixtures/llm_golden/fallback_ladder.json` `tests/fixtures/llm_golden/gigachat_native.json` `tests/fixtures/llm_golden/local_lane.json` `tests/fixtures/llm_golden/openai_direct.json` `tests/fixtures/llm_golden/openrouter_payload.json` `tests/fixtures/llm_golden/target_resolution.json` `tests/test_llm_provider_golden.py` |
| typed tool-result cutover | v7(T1): pin the cutover against a golden of the retired classifier | `b5569add` | `tests/fixtures/legacy_tool_classification_306f8827.json` `tests/test_tool_classification_differential.py` `tests/tool_classification_corpus.py` |
| typed tool-result cutover | build(tools): derive frozen tool inventory | `4d93abb6` | `tests/test_frozen_tool_inventory.py` |
| typed tool-result cutover | refactor(tools): add typed result expansion seam | `84271726` | `tests/test_tool_result.py` |
| typed tool-result cutover | refactor(tools): name process guard denials | `e03d4ece` | `tests/test_process_guard_codes.py` |
| typed tool-result cutover | refactor(tools): type binding and route facts | `a0328d1c` | `tests/test_tool_result_meta_boundaries.py` |
| typed tool-result cutover | refactor(tools): type plan and git control facts | `a5e1cea3` | `tests/test_tool_result_t46.py` |
| typed tool-result cutover | refactor(tools): type process result facts | `784a2e2f` | `tests/test_process_result_corrections.py` |
| typed tool-result cutover | v7(A.21): the control tools' argument and access refusals name their own code | `91574b33` | `tests/test_control_native_results.py` |
| typed tool-result cutover | v7(T2): the read/list tools name their own refusal instead of leaving it to be re-read | `4b350a2d` | `tests/test_core_native_results.py` |


## 4. (c) Upstream-adopted — 25 files

These arrived with upstream development (v6.103.0 → v6.105.1 and the PR #257 sync). The
campaign wrote none of them; it merged them, and in three cases split what it merged (§2).


| added file | upstream commit | upstream change |
|---|---|---|
| `tests/test_claudexor_admission_wait.py` | `5eef9f27` | fix(delegation): claudexord recovery-only admission race — servingMode-aware spawn/attach |
| `tests/test_client_surface.py` | `d6bfb1f5` | feat(chat): per-message owner surface fact and process presentation posture |
| `tests/test_delegation_account_pin.py` | `c984dea9` | feat(accounts): unified account model — dual-engine wire, honest UI, delegation account pin |
| `tests/test_delegation_phase_b.py` | `ed625683` | fix(nanny): dispatch-time delegation mandate, typed capability_delta axes, nudge-ignored visibility |
| `tests/test_direct_activity_registry.py` | `b51b69f9` | Add server-authoritative in-flight indicator for direct and ephemeral chat turns |
| `tests/test_inflight_indicator_seams.py` | `b51b69f9` | Add server-authoritative in-flight indicator for direct and ephemeral chat turns |
| `tests/test_launcher_server_reaper.py` | `663d92e3` | feat: the launcher reaps leftover same-install server generations before every boot |
| `tests/test_plan_review_engine.py` | `d6210b1a` | plan review becomes a domain-neutral spec gate |
| `tests/test_plan_review_epoch.py` | `74c11db4` | Rotation visibility sprint (PR-B): typed lane facts, plan-review anti-loop, rotation reconcile, one-shot followups, lane history |
| `tests/test_plan_review_w3.py` | `bdbf062d` | plan review: the governance pack per the approved W3 wording — ARCHITECTURE.md inline for a self-modification plan, the host attaches reviewer-requested evidence |
| `tests/test_plan_spec.py` | `d6210b1a` | plan review becomes a domain-neutral spec gate |
| `tests/test_port_sweep_listener_scope.py` | `0c4acfd9` | fix: scope POSIX port sweeps to the listener, sparing connected clients |
| `tests/test_project_chat_continuity.py` | `53a2e439` | Fix project chat continuity: local-echo journal, queue-backed activity, honest finalizing |
| `tests/test_provider_key_test.py` | `95aa4850` | feat: a Test control on every provider card probes the entered credentials |
| `tests/test_review_cycles.py` | `d6210b1a` | plan review becomes a domain-neutral spec gate |
| `tests/test_route_health_pinned_profile.py` | `8acc6fc8` | fix(review routes): pass a pinned credential profile through route_health to the engine |
| `tests/test_schedule_followup.py` | `74c11db4` | Rotation visibility sprint (PR-B): typed lane facts, plan-review anti-loop, rotation reconcile, one-shot followups, lane history |
| `tests/test_ui_smoke_inflight_indicator.py` | `b51b69f9` | Add server-authoritative in-flight indicator for direct and ephemeral chat turns |
| `tests/test_ui_smoke_project_continuity.py` | `53a2e439` | Fix project chat continuity: local-echo journal, queue-backed activity, honest finalizing |
| `web/tests/chat_continuity.test.js` | `53a2e439` | Fix project chat continuity: local-echo journal, queue-backed activity, honest finalizing |
| `web/tests/chat_inflight_indicator.test.js` | `b51b69f9` | Add server-authoritative in-flight indicator for direct and ephemeral chat turns |
| `web/tests/client_surface.test.js` | `d6bfb1f5` | feat(chat): per-message owner surface fact and process presentation posture |
| `web/tests/fixtures/credential_profiles_response_unified.json` | `c984dea9` | feat(accounts): unified account model — dual-engine wire, honest UI, delegation account pin |
| `web/tests/provider_test.test.js` | `6ba18e82` | Refine PR #257 provider and integration contracts |
| `web/tests/windows_layout_switch.test.js` | `10779106` | fix(web): prevent Windows layout-switch Alt key from hijacking input focus |


## 5. Deletions — one, and it is upstream's

`git diff --name-status 353fd974..HEAD -- tests/ web/tests/` reports exactly one `D`:

| deleted file | deleted by | disposition |
|---|---|---|
| `tests/test_planning_swarm_adaptive_wait.py` | `d6210b1a` "plan review becomes a domain-neutral spec gate" | **upstream deletion, adopted.** `d6210b1a` is an ancestor of `e7c84240`, i.e. it is upstream's own v6.103.0 plan-review redesign, which retired planning scouts, the plan Atlas, `plan_class`, `context_level` and the hidden 32-wave limit. The suite characterized the adaptive wait of a mechanism that no longer exists. The campaign did not author, request or extend this deletion — it inherited it with the merge. |

No campaign wave deleted a test file. Where a v7 split *retired* a test-side symbol rather
than moving it, the ledger row carries a `retired:` destination and the file survives; those
rows are visible in `MIGRATION_v7.md` under `tests/test_commit_gate.py`,
`tests/test_external_review_script.py`, `tests/test_telegram_miniapp_companion.py` and
`tests/test_telegram_miniapp_lifecycle.py`.

## 6. Per-wave counts

The lane label is the commit-subject prefix of the commit that added the file — the
campaign's own labelling, not a reconstruction. The S7a lane's commits are labelled
`v7(S)`; the inventory dict that holds its maps is named `s7a_`.

| lane | added files |
|---|---:|
| `v7(S)` — S7a test-giant splits | 92 |
| `v7(S7b)` — S7b test-giant splits | 51 |
| upstream (adopted) | 25 |
| `v7(T)` — runtime and test splits, tool/headless/git/shell lane | 19 |
| `refactor(web)` — web owner extractions | 16 |
| `v7(TS2)` — review-family test splits | 15 |
| `v7(L)` — llm lane split and its goldens | 12 |
| `refactor(tools)` — tool registry/result owner extractions | 10 |
| `v7(W)` — devtools, OSWorld and skill-payload splits | 8 |
| `v7(S6)` — cancellation/containment characterization | 6 |
| `v7(S3)` — supervisor events/queue/worker splits | 4 |
| `v7:` — unlabelled owner splits (control, extension_loader, ledger data) | 4 |
| `v7(T1)` — classifier-cutover golden | 3 |
| `v7(L2b)` — review substrate/evidence/skill-review splits | 3 |
| `v7(L-A)` — review_state / scope_review / review_helpers splits | 3 |
| `test(v7)` — evidence freezes | 3 |
| `v7(S2)` — server.py split and the panic-sweep characterization | 2 |
| `v7 S6b` — E2E cancellation scenarios | 2 |
| `test(v7 S1)` — settings-seam characterization | 2 |
| adoption merge `734ac4fc` (upstream giant split during the merge) | 2 |
| sixteen single-file lanes (`v7(S1)`, `v7(S3b)`, `v7(T2)`, `v7(A.21)`, `v7(ledger)`, `feat(scripts)`, `feat(update)`, `fix(supervisor)`, `build(tools)`, `test(architecture)`, `refactor(agent)`, `refactor(custody)`, `refactor(git_ops)`, `refactor(loop)`, `refactor(review)`, `refactor(update)`) | 16 |
| **total** | **298** |

## 7. What this table does not claim

- It does not say the suite got *better*. It says where every file came from. A split moves
  assertions; it does not add them. The bucket that adds assertions is (b), 93 files, and
  the ledger/evidence and S6 sub-classes inside it are deliberately characterization —
  some of them pin a defect rather than a fix, and say so in their own docstrings.
- It does not measure collected items. The census does: 9 077 → 10 316 in the CI parallel
  lane, 409 → 443 serial, 360 → 584 node.
- "Split from a named giant" is a ledger fact, not a byte claim. The byte claim belongs to
  `tests/test_v7_verbatim_moves.py`, which pins the rows that assert verbatim movement.
- The commit named in a (b) row is the commit that *added* the file. A file later extended
  by another commit still shows its origin here, which is the question this table answers.
