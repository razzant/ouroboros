# Ouroboros Module Reference (Detailed)

This file contains the full verbose module listing for ARCHITECTURE.md.
The main ARCHITECTURE.md uses a condensed listing to reduce context window usage.
Read this file on demand via `read_file(root="system_repo", path="docs/ARCHITECTURE_REFERENCE.md")`.

---

## ouroboros/ — Full Module Descriptions

```
  └── ouroboros/               ← Agent core (runs inside worker processes)
      ├── config.py            ← SSOT: paths, settings defaults, load/save, PID lock
      ├── colab_bootstrap.py   ← Google Colab source-mode bootstrap helpers
      ├── cli.py               ← Source/headless CLI over gateway tasks, logs, settings
      ├── packaged_cli.py      ← Packaged desktop CLI bridge
      ├── packaged_cli_install.py ← Packaged CLI installer
      ├── agent.py             ← Task orchestrator
      ├── agent_startup_checks.py ← Startup verification and health checks
      ├── agent_task_pipeline.py  ← Task pipeline orchestration; swarm_efficiency rollup
      ├── post_task_checkpoint.py ← Durable root post-task phase/final-cost checkpoint
      ├── extension_companion.py ← Host-supervised companion processes
      ├── extension_reconcile_queue.py ← Worker→server extension reconcile markers
      ├── event_bus.py         ← Typed in-process event bus
      ├── evolution_checkpoints.py ← Campaign/eval checkpoint ledger
      ├── improvement_backlog.py ← Advisory improvement backlog with dedup + grooming
      ├── loop.py              ← High-level LLM tool loop + finalization nudges
      ├── loop_llm_call.py     ← Single-round LLM call + usage accounting
      ├── task_pacing.py       ← Task-pacing: deadline/cost milestones, BudgetSnapshot
      ├── vision_routing.py    ← Send-time image routing
      ├── fallback_cooldown.py ← 429-aware per-process cooldown for model fallbacks
      ├── model_concurrency.py ← Per-route concurrency cap (BoundedSemaphore)
      ├── project_naming.py    ← LLM-first project naming with heuristic fallback
      ├── loop_tool_execution.py ← Tool dispatch + tool-result handling
      ├── deadline_utils.py    ← Deadline parsing + remaining-time helpers
      ├── observability.py     ← Forensic execution ledger
      ├── outcomes.py          ← Typed loop/task outcome + verification ledger
      ├── code_intelligence.py ← Polyglot symbol/reference extraction (tree-sitter)
      ├── code_search_rg.py    ← Ripgrep-backed search helper
      ├── pricing.py           ← Provider-catalog lookup
      ├── usage_accounting.py  ← Append-only monetary authority
      ├── llm.py               ← Multi-provider LLM routing
      ├── mcp_client.py        ← HTTP/SSE MCP client manager
      ├── safety.py            ← Policy-based LLM safety check
      ├── consciousness.py     ← Background thinking loop
      ├── consolidator.py      ← Block-wise dialogue consolidation
      ├── memory.py            ← Scratchpad, identity, chat history
      ├── project_facts.py     ← Per-project facts store
      ├── task_tree_ledger.py  ← Task-tree coordination ledger (swarm blackboard)
      ├── projects_registry.py ← Multi-project registry
      ├── project_dialogue.py  ← Read-only Project dialogue lens
      ├── project_lease.py     ← One-writer-per-project lease
      ├── context.py           ← LLM context-source builder
      ├── context_fit.py       ← ContextFitPlan: Max/Low projections
      ├── context_budget.py    ← Context-window budget SSOT
      ├── capability_evidence.py ← Route-fingerprinted context-window evidence
      ├── context_layout.py    ← Reference-doc layout SSOT
      ├── context_compaction.py ← Context trimming + summarization
      ├── headless.py          ← Headless task isolation + memory export
      ├── coop_checkpoint.py   ← Genesis/coop tree checkpoint commits
      ├── subagents.py         ← Subagent model-lane resolution + lineage
      ├── subagent_worktrees.py ← Acting self_worktree lifecycle
      ├── artifacts.py         ← Task-scoped artifact helpers
      ├── retention.py         ← GC retention SSOT
      ├── workspace_preflight.py ← External-workspace snapshot
      ├── project_sources.py   ← Project working-folder sources
      ├── workspace_admission.py ← Workspace-task admission SSOT
      ├── local_model.py       ← Local LLM lifecycle
      ├── local_model_autostart.py ← Local model startup
      ├── deep_self_review.py  ← Deep self-review Atlas
      ├── review.py            ← Code collection + pre-commit review
      ├── preflight_runner.py  ← Hermetic pytest runner
      ├── review_substrate.py  ← Reviewer-slot coordinator
      ├── review_state.py      ← Advisory pre-review state
      ├── triad_review.py      ← Multi-model review primitives
      ├── onboarding_wizard.py ← Onboarding bootstrap
      ├── settings_setup_contract.py ← Settings/onboarding contract
      ├── owner_mailbox.py     ← Per-task user message mailbox
      ├── launcher_bootstrap.py ← Bundle-to-repo bootstrap
      ├── provider_models.py   ← Provider model ID helpers
      ├── runtime_mode_policy.py ← Runtime-mode protected-path policy
      ├── schedule_contract.py ← Cron + timezone validation SSOT
      ├── reflection.py        ← Execution reflection + pattern capture
      ├── post_task_evolution.py ← Post-task self-evolution
      ├── repo_remotes.py      ← GitHub remote provisioning
      ├── review_evidence.py   ← Structured review findings snapshot
      ├── semantic_dedup.py    ← LLM-first semantic-duplicate detector
      ├── skill_loader.py      ← Skill discovery + durable state
      ├── skill_readiness.py   ← Skill readiness helper
      ├── skill_dependencies.py ← Dependency-spec resolution
      ├── skill_publish_eligibility.py ← Publish eligibility predicate
      ├── skill_review_status.py ← Skill-review verdict aggregation
      ├── skill_review_passes.py ← Skill-review pass runner
      ├── skill_review.py      ← Skill review pipeline
      ├── extension_loader.py  ← Phase 4 extension loader
      ├── extension_process_runner.py ← Child-process runner for extensions
      ├── extension_ui_validation.py ← Widget/settings validation
      ├── extension_isolated_deps.py ← Isolated-dep bridge
      ├── extension_health.py  ← Per-extension health vector
      ├── skill_token.py       ← Host Service API token wrapper
      ├── marketplace/         ← ClawHub + OuroborosHub package
      ├── skill_lifecycle_queue.py ← Skill lifecycle action queue
      ├── skill_review_runner.py ← Lifecycle-backed skill review runner
      ├── server_auth.py       ← Non-localhost auth gate
      ├── server_control.py    ← Process-control helpers
      ├── server_entrypoint.py ← CLI + port-binding
      ├── server_runtime.py    ← Server startup + WebSocket helpers
      ├── server_web.py        ← Static web file helpers
      ├── task_continuation.py ← Per-task review continuation state
      ├── task_results.py      ← Durable task result files
      ├── task_status.py       ← Effective task-status SSOT
      ├── git_shell_policy.py  ← Git argv classifiers
      ├── protected_artifacts.py ← Protected artifact policy
      ├── shell_parse.py       ← Shell argv parser
      ├── workspace_executor.py ← Workspace process backend
      ├── tool_capabilities.py ← SSOT for tool sets
      ├── tool_access.py       ← Tool API v2 policy matrix
      ├── tool_policy.py       ← Round-one tool visibility policy
      ├── utils.py             ← Shared utilities
      ├── world_profiler.py    ← System profile generator
      ├── contracts/           ← Frozen ABI (Protocols, TypedDicts, SkillManifest)
      │   ├── tool_context.py  ← ToolContextProtocol
      │   ├── tool_abi.py      ← ToolEntryProtocol + GetToolsProtocol
      │   ├── api_v1.py        ← WS/HTTP envelope TypedDicts
      │   ├── chat_id_policy.py ← Chat id policy SSOT
      │   ├── task_contract.py ← Task contract draft/normalization
      │   ├── task_constraint.py ← Per-task execution constraints
      │   ├── skill_payload_policy.py ← Skill-payload path policy
      │   ├── skill_manifest.py ← SKILL.md / skill.json parser
      │   ├── schema_versions.py ← Schema version helpers
      │   └── plugin_api.py    ← PluginAPI Protocol + registration
      ├── gateways/            ← External API adapters
      │   └── claude_code.py   ← Claude Agent SDK gateway
      ├── gateway/             ← Gateway Boundary v1
      │   ├── contracts.py     ← HTTP/WS envelope + endpoint index
      │   ├── router.py        ← Starlette route collector
      │   ├── ws.py            ← WebSocket connection manager
      │   ├── state.py         ← Health + state handlers
      │   ├── tasks.py         ← Task CRUD + events
      │   ├── logs.py          ← Runtime log tail
      │   ├── settings.py      ← Settings + onboarding handlers
      │   ├── control.py       ← Reset, command, evolution handlers
      │   ├── schedules.py     ← Cron schedule HTTP surface
      │   ├── files.py         ← File browser + upload
      │   ├── ui_preferences.py ← UI preferences
      │   ├── models.py        ← Model catalog
      │   ├── extensions.py    ← Extensions HTTP surface
      │   ├── marketplace.py   ← Marketplace HTTP surface
      │   ├── mcp.py           ← MCP Settings API
      │   ├── host_service.py  ← Host Service API
      │   ├── history.py       ← Chat history + cost breakdown
      │   ├── projects.py      ← Multi-project CRUD
      │   └── _helpers.py      ← Shared HTTP helpers
      ├── tools/               ← Auto-discovered tool plugins
      │   ├── extension_dispatch.py ← Extension tool dispatch
      │   ├── release_sync.py  ← Release-metadata sync
      │   ├── review_synthesis.py ← LLM-based claim synthesis
      │   ├── ci.py            ← CI trigger + monitoring
      │   ├── claude_advisory_review.py ← Advisory pre-review
      │   ├── recent_tasks.py  ← Recent task context recovery
      │   ├── commit_gate.py   ← Commit gate + attempt recording
      │   ├── git_rollback.py  ← vcs_rollback tool
      │   ├── git_pr.py        ← PR integration tools
      │   ├── github.py        ← GitHub integration
      │   ├── parallel_review.py ← Parallel triad+scope orchestration
      │   ├── plan_review.py   ← Pre-implementation design review
      │   ├── review.py        ← Task acceptance review
      │   ├── review_context_atlas.py ← Bounded-context compiler
      │   ├── query_code.py    ← Structured code intelligence
      │   ├── media.py         ← Media tools (OCR, YouTube, video frames)
      │   ├── verify.py        ← verify_and_record core tool
      │   ├── review_helpers.py ← Shared review helpers
      │   ├── review_revalidation.py ← Reviewed-commit revalidation
      │   ├── scope_review.py  ← Scope reviewer
      │   ├── scope_review_contract.py ← Scope output parser
      │   ├── services.py      ← Long-running service manager
      │   ├── skill_exec.py    ← External-skill execution
      │   ├── skill_publish.py ← Hub publish tool
      │   ├── skill_preflight.py ← Skill payload preflight
      │   ├── project_journal.py ← Journal/workpad tools
      │   ├── task_tree.py     ← Task-tree coordination tools
      │   ├── join_ledger.py   ← Soft-join decision tools
      │   └── subagent_integration.py ← Acting subagent patch integration
      └── platform_layer.py    ← Cross-platform process/path/locking
```
