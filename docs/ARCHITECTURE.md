# Ouroboros v6.87.0 — Architecture & Reference

This file is NOT a changelog. Version history lives in README.md, git tags, and commit log.

This document is the current operational map of Ouroboros: structure, data flows, APIs, protected boundaries, and the rationale for non-obvious architectural choices. Rationale must be self-contained here; future maintainers should not need to open old commits to understand why a guard, review gate, or lifecycle exists.

---

## 1. High-Level Architecture

```
User
  │
  ▼
launcher.py (PyWebView)       ← desktop window, immutable outer shell (tracked in git; bundled as packaged entry point)
  │
  │  spawns subprocess
  ▼
server.py (Starlette+uvicorn) ← HTTP + WebSocket on configurable host:port (default localhost:8765; Docker/non-loopback supported via OUROBOROS_SERVER_HOST=0.0.0.0)
  │
  ├── web/                     ← Web UI (SPA with ES modules in web/modules/)
  │
  ├── supervisor/              ← Background thread inside server.py
  │   ├── message_bus.py       ← Queue-based local message bus (Web UI + reviewed transport skills)
  │   ├── workers.py           ← Multiprocessing worker pool (fork/spawn by platform)
  │   ├── state.py             ← Persistent state (state.json) with file locking
  │   ├── queue.py             ← Task queue management (PENDING/RUNNING lists) + activity-based timeout enforcement
  │   ├── task_lifecycle.py    ← Queue-owned atomic acceptance fences, one root-budget admission marker plus replay-safe task resume, cascade cancellation, and fenced Project deletion/quiescence; extends queue state without creating a second lifecycle authority
  │   ├── task_reaper.py       ← (v6.38.0) Variant A off-loop worker reaper (extracted from queue.py): kill/join/archive/respawn a timed-out worker on a single-owner background thread, off the loop critical path. (v6.38.1) STRICT fail-closed: if the worker will not confirm dead, it holds the slot `reaping` and leaves the task RUNNING (no terminal/task_done/retry/respawn while it may be alive), emits `task_reaper_wedged` + an owner /restart hint, and lets the custody reaper end the orphan on the next generation
  │   ├── schedule_time.py     ← Cron/timezone schedule time parsing helpers
  │   ├── evolution_lifecycle.py ← Evolution campaign state + transaction lifecycle (moved from queue.py in v6.30.0): campaign file IO, start/pause, begin/update transaction, cycle-outcome recording, deterministic no_op/abandoned worktree cleanup, owner cycle reports, supervisor auto-restart request
  │   ├── events.py            ← Event dispatcher (worker→supervisor events) + managed-update assisted-merge orphan watchdog hook
  │   ├── git_ops.py           ← Git operations (clone, checkout, rescue, rollback, push, credential helper)
  │   ├── update_merge.py      ← (v6.41.0) Managed-update merge engine: real 3-way merge plan, the AUTOMATED assisted live-worktree materialize + native-MERGE_HEAD commit helpers + 4-phase tx, fail-closed lock, apply/rollback/smoke, non-destructive boot recovery (RELEASE_INVARIANT-protected)
  │   └── update_merge_policy.py ← (v6.41.0) Per-path conflict classification (clean/doc_reconcile/conflicting; protected docs) for managed updates (RELEASE_INVARIANT-protected)
  │
  └── ouroboros/               ← Agent core (runs inside worker processes)
      ├── config.py            ← SSOT: paths, settings defaults, load/save, PID lock
      ├── colab_bootstrap.py   ← Google Colab source-mode bootstrap helpers, driven by the `notebooks/colab_quickstart.py` cell script: Drive-backed data/settings, fork-safe env, personal origin provisioning, no-UI server command assembly, and loopback discovery/grant/enable/configuration of the bundled native `telegram` skill
      ├── cli.py               ← Source/headless CLI over gateway tasks, logs, settings, skills, marketplace, local-model, and MCP wrappers
      ├── packaged_cli.py      ← Packaged desktop CLI bridge: resolves bundle roots, bootstraps the launcher-managed repo, and delegates to cli.py
      ├── packaged_cli_install.py ← Packaged CLI installer planning/execution for user-local command shims
      ├── agent.py             ← Task orchestrator
      ├── agent_startup_checks.py ← Startup verification and health checks
      ├── agent_task_pipeline.py  ← Task execution pipeline orchestration; emits a per-task `swarm_efficiency` rollup (subagent_count/wave_count/Σ inter-wave latency/lanes_used) for fan-out tasks only, and freezes one shared non-final subtree-cost snapshot for summary/reflection before the terminal checkpoint records final spend
      ├── mutation_attribution.py ← Root-task baseline capture in the existing task result and clean-at-baseline Git candidate projection; terminal projection includes the committed interval delta
      ├── python_interpreter.py ← One-time pre-guard unversioned-Python resolver for the four user process launch surfaces
      ├── post_task_checkpoint.py ← Durable root post-task phase/final-cost checkpoint shared by task finalization and Project naming recovery
      ├── extension_companion.py ← Host-supervised companion processes for transport skills
      ├── extension_reconcile_queue.py ← Durable worker→server extension reconcile markers and server pickup loop
      ├── event_bus.py         ← Typed in-process event bus for skill subscriptions
      ├── evolution_checkpoints.py ← Append-only campaign/eval checkpoint ledger for evolution progress
      ├── improvement_backlog.py ← Durable advisory improvement backlog: recurrence-counted dedup (bump count/last_seen, never drop), priority+recurrence+recency ranking, close-on-commit (`close_backlog_items`), and size-triggered non-error-gated LLM grooming (`groom_backlog`); parser-safe locked writer; entries carry priority/kind (bug/improvement/capability_idea)
      ├── loop.py              ← High-level LLM tool loop; one-shot no-op-attempt finalization nudge (declared expected_output + zero effects + no FINAL ANSWER); (v6.51.0) a one-shot ADVISORY red-verification finalization nudge (ordered before the receipt-absent nudge) when the latest host-attested verify receipt is unreconciled-RED (`outcomes.latest_unreconciled_failed_verification`) — re-check / explain / fix; (v6.52.2) a one-shot ADVISORY masked-verification nudge (ordered after the red nudge) when the latest PASSing verify check can launder its exit code (`outcomes.latest_unreconciled_masked_verification`) — re-ground without the masking pipe or explain; (v6.53.0) continuous explicit `FINAL ANSWER:` latching captures the latest typed candidate every round (tool-count-stamped, no prose mining) so review/nudge/forced-finalization paths do not erase a structured answer, and intrinsic no-deadline pacing asks for a salvageable current answer on long tasks; (v6.60.0) ALL marker prompting (P2 marker nudge, pacing salvage phrases, the context instruction) is gated on `task_contract.answer_protocol="final_answer_line"` via the `answer_protocol_active` SSOT — the latch/extractor stay unconditional; (v6.61.4) the protocol gate is SUFFICIENT for the P2 marker nudge — it no longer also requires a declared `expected_output` (GAIA-shaped contracts carry the question in `objective` with `expected_output` empty, which silently suppressed the one salvage surface and let a last-round refusal finalize with an empty typed answer), and `extract_final_answer` structurally rejects the snake_case outcome-tier ledger identifiers (`best_effort`/`blocked_with_evidence`) as answers — internal enum vocabulary is never a deliverable (a reviewed run shipped `FINAL ANSWER: blocked_with_evidence` verbatim); `solved` stays extractable as an ordinary English word
      ├── loop_llm_call.py     ← Single-round LLM call + usage accounting
      ├── task_pacing.py       ← Task-pacing SSOT: deadline/cost milestones, finalization reserve, BudgetSnapshot, and acceptance-review launch/improvement rails. v6.64 reserves at least 200s for the first review and then `max(configured_floor, 1.5×EWMA)` from existing timing events (`alpha=0.5`); an explicit `max_improvement_passes` always binds, while Required+Blocking without one has no local count cap (deadline/global rails still apply). Legacy `until_deadline` and `stall_rounds_threshold` are accepted for one compatibility window with a deprecation event. v6.74.4 (figlet incident mitigation): workspace deliveries (`_workspace_delivery`, canonical `is_workspace_mode()` with an attribute fallback) get one shared commit-neutral tree sentence (`_TREE_FLUSH_SENTENCE` — commit-neutral because acting self_worktree subagents cannot commit and a moved HEAD fails patch capture closed) on the 10% deadline flush, the ~80% cost wrap-up, and a late FIRST cost milestone that would otherwise suppress the wrap-up; non-workspace texts stay byte-identical. Disclosed residual (mitigation, not closure): a forced tool-less exit crossed inside one long round with no pacing note or acceptance capsule in the terminal stretch can still ship an unverified last edit — the structural verification-freshness seam is an owner-pending follow-up.
      ├── vision_routing.py    ← (v6.45) Send-time image routing SSOT: inline vision vs generic captions vs placeholders on a per-send message copy, controlled by `OUROBOROS_IMAGE_INPUT_MODE` and `OUROBOROS_MODEL_VISION`
      ├── fallback_cooldown.py ← (v6.39) Per-process 429-aware cooldown for the `OUROBOROS_MODEL_FALLBACKS` cross-model chain: a transiently-failed model (429/5xx/overloaded) is parked for a short window so a task's own fallback walk and repeated rounds skip it instead of re-hammering. PER-PROCESS only (not a swarm-wide governor — each worker has its own map; cross-worker coordination is Phase 3). Advisory, default-on, fail-soft, passive (timestamp) heal
      ├── model_concurrency.py ← (v6.40) Per-(model,use_local)-route `threading.BoundedSemaphore` capping CONCURRENT provider calls (`OUROBOROS_MODEL_MAX_CONCURRENCY`, default 3) so a task's main loop + its in-process subagent threads + status pings cannot self-DoS one model's rate limit — excess threads WAIT (deadline-bounded) instead of all firing 429s. PER-PROCESS only (like `fallback_cooldown`; heavy workers are separate processes, so this is not a swarm-wide governor — cross-worker admission is future work). Wraps ONLY the provider call in `loop_llm_call.call_llm_with_retry` (not the retry/backoff chain). Default-on, fail-soft
      ├── project_naming.py    ← (v6.40) SSOT for LLM-first project naming: a bounded LIGHT-model title with a deterministic heuristic fallback (P5, no keyword gates, fail-soft), shared by the proactive card namer (`supervisor/workers.py`), turn-into-project conversion (`gateway/projects.py`), and `ensure_project_scope`. The provider call goes through the `model_concurrency` slot
      ├── loop_tool_execution.py ← Tool dispatch and tool-result handling
      ├── deadline_utils.py    ← Shared deadline parsing/remaining-time helpers for loop milestones and process-tool timeouts
      ├── observability.py     ← Private forensic execution ledger: redaction, gzip CAS blobs, call manifests, trace refs
      ├── outcomes.py          ← Typed loop/task outcome, artifact bundle, verification ledger helpers; `children_unabsorbed` joins the honest best-effort shelf when a delegating parent ignores a bounded running-child absorption reminder; an unrecovered access-policy block (resource_policy_blocked/resource_constraint_blocked) on a READ-ONLY exploratory tool is demoted to an `ignored_tool_errors` axis (honest telemetry, never degraded); (v6.57.0) an unrecovered POLICY refusal (`*_blocked`) on ANY tool lands in a non-degrading `policy_denials` bucket and never headlines `tool_failure`; `compute_cost_with_children` rolls direct-child cost into an additive `cost_usd_with_children` (partial-marked); (v6.51.0) `latest_unreconciled_failed_receipt`/`latest_unreconciled_failed_verification` — the latest RED verify receipt with no later pass/observed reconciler FOR THE SAME VERIFICATION (v6.78.0: the same typed identity KEY — `criterion_id`, else canonical `check` text, else the observed `paths` set — matched on kind AND value, while a red carrying NO key keeps the older any-later-green rule; a `declared` escape hatch does NOT reconcile), feeding the finalize red-nudge; (v6.52.2) `latest_unreconciled_masked_pass`/`latest_unreconciled_masked_verification` — the latest PASSing receipt carrying the `check_exit_masking` flag with no later clean (non-masked) pass/observed (command text NEVER participates — the masked receipt's only text identity is its MASKED command, which the prescribed remediation necessarily changes — so a masked receipt that NAMES a `criterion_id` is reconciled only by a later clean receipt naming that SAME id, one that omits its id does NOT clear it, and the ANY-later-clean fallback applies only to a masked receipt naming no criterion), feeding the advisory masked-verification nudge + acceptance summary; (v6.53.0) answer-first headline precedence keeps lifecycle `completed` and execution-health details in `outcome_axes.execution`, but a valid structured answer no longer gets top-level `reason_code=tool_failure`; (v6.78.0) SSOT of the host acceptance-decision vocabulary — `accepted | revision_requested | finalized_unaccepted` (`ACCEPTANCE_DECISION_STATUSES`) plus the typed `reason` projected by `_acceptance_decision_projection`, and `derive_loop_outcome` keys the deadline-reserve degradation on that status+reason PAIR
      ├── _outcome_receipts.py ← Private pure helpers for parsing append-only verification receipts, finding the latest unreconciled failed/masked/agent-defined receipt, the ONE canonical receipt IDENTITY derivation everything else reads (`receipt_canonical_identity` → `ReceiptIdentity` — the three INDEPENDENT components `criterion_id` / STRUCTURALLY canonical `check` text PAIRED WITH ITS RENDERING (`shell_parse.canonical_command_text`, so whitespace between tokens folds but a quoted argument's contents, a quoted token that merely SPELLS like an operator, and the control operators do not — a lossy text identity let a green close an unrelated red; the `check_rendering` stamp — `shlex_join` / `declared_text` / absent = `unversioned` — is part of the check identity because the renderer CHANGED in v6.78.0 and the stored string alone cannot say which one wrote it, so an old space-joined `echo a b` and a `shlex.join` of a DIFFERENT argv reading the same were falsely equal: receipts from different renderings are never the same verification, unversioned↔unversioned still matches, and an unknown future stamp is automatically its own namespace) / canonical observed `paths` set (`canonical_path_set` — de-duplicated and sorted on the RAW values, whitespace never touched, since a leading or trailing space is a legal filename byte) of the command-less artifact-observation class, from which `ReceiptIdentity.key` selects ONE typed `(kind, value)` identity — the most specific component the receipt carries — and sameness is that key's equality, kind AND value, never a match across kinds; `receipt_identity` IS that key, `receipt_identity_parts`/`receipt_expected_whitespace_normalized` are DISCLOSURES of it and never the comparison (the parts are three plain texts; sameness reads one key and never falls back across components), the kind disclosed per row as `reconciliation_identity`; a single key replaced a per-component FALLBACK CHAIN, which was not transitive (`{c1,check}` matched `{check}` matched `{c2,check}`) and so let one check-only green clear two distinct criterion-keyed reds and made the outstanding set order-dependent — keying makes the relation the kernel of a function (an equivalence), makes an existing `criterion_id` authoritative structurally, and fails SAFE: strictly fewer reconciliations, so a red the chain used to clear may now stay open (a re-run that OMITS its id no longer clears its own red; the sound route to omission tolerance is carrying the id forward at receipt ingress, never inferring it from shared command text); `_reconciles` falls back to any-later-grounding when the EARLIER receipt has no key at all, and the masked path uses `_reconciles_masked` — the same rule on the `criterion_id` key alone, so an identified masked receipt is NOT cleared by a later clean receipt that omits its id and the any-clean fallback reaches only a masked receipt naming no criterion; both relations and both disclosures read ONE mode-aware projection, `receipt_reconciliation_key(receipt, masked=…)` (mode selected per receipt by `receipt_is_masked_pass` in `receipt_disclosed_reconciliation_key`), so `reconciliation_identity` and `expected_whitespace_normalized` report the authority that actually decided instead of re-deriving one beside it — round 6: an id-less masked pass was disclosed as `check`-governed with `expected_whitespace_normalized=true` while its reconciliation ignored check text entirely, host-attested evidence lying about its own basis; round 7 was the SAME class one kind over — the flag also read true for `artifact_paths`, whose set is compared byte-for-byte — so the per-kind answer now lives in the closed kind table `IDENTITY_KINDS`/`KIND_NORMALIZES_COMMAND_TEXT` beside the kinds themselves, `ReceiptIdentity.key` iterates that table and the flag is ONE lookup in it, total over every kind: true for `check`, false for `criterion_id`, `artifact_paths` and `none`, and a fourth kind must state its own answer in its own row rather than inherit a default), the OUTSTANDING SETS the advisory flags are projections of (`unreconciled_failed`/`unreconciled_masked` — each candidate scanned against ALL later reconcilers, so a newer failure can never erase an older still-unreconciled one the way a single latest-pointer did, then collapsed onto the IDENTITY it names via `_same_verification`/`_same_masked_verification` — reconciliation in BOTH directions, never on an identity-less receipt — so repeated failures of one check count as one red and are represented by their freshest receipt; `latest_unreconciled_*` return the newest element), the ONE shared disclosed identity projection both fixed reviewer surfaces render through (`receipt_identity_projection` — every participating component plus, whenever the path list is bounded, an explicit `paths_omitted` count and `paths_identity_sha256` over the injective serialization of the SAME canonical set the carried items come from), and the ONE shared disclosed-list projection every bounded list on these surfaces goes through (`disclosed_list_projection` — carried items plus an exact `<key>_omitted` count and, where the full set is not reachable from the store the row lives in, its hash, so a bound is never SILENT (BIBLE P1); string bounding is the SSOT `utils.truncate_review_artifact`, never a hand-rolled slice), the FIXED verification-ledger receipt row (`verification_receipt_ledger_row` — splats that projection; a new receipt key is dropped unless added there or to the projection), and reconciling current versus superseded acceptance-review runs; `outcomes.py` remains the public typed-outcome authority
      ├── code_intelligence.py ← Internal code inventory v2: derived-only file facts, hashes, polyglot symbol/import/call/reference extraction via tree-sitter for non-Python languages (Go/Rust/Java/Ruby/C/...) with Python on the stdlib `ast` path and a visible `structural_unavailable` fallback when a grammar is missing, plus an incremental JSON cache (no raw source)
      ├── code_search_rg.py    ← Optional ripgrep-backed search helper for search_code; every match is post-filtered through Ouroboros protected/secret gates
      ├── pricing.py           ← Exact-route best-effort provider-catalog lookup with nullable estimates; no static model tariffs and not the monetary ledger
      ├── usage_accounting.py  ← Append-only physical-model-attempt monetary authority: reserved→dispatched→settled|unresolved (or reserved→released), short cross-process check+append+fsync lock, conservative global/root admission, validated sequence replay/torn-tail quarantine, compatibility projections, and resumable legacy import
      ├── llm.py               ← Multi-provider LLM routing (OpenRouter/OpenAI/compatible/Cloud.ru/GigaChat/Anthropic) with adaptive request-parameter normalization for provider capabilities/rejections
      ├── mcp_client.py        ← HTTP/SSE MCP client manager: parses MCP_SERVERS, validates URLs/auth headers, masks tokens, normalizes external tool names as mcp_<server>__<tool>, refreshes tool lists, and dispatches calls through the guarded Python mcp SDK import
      ├── safety.py            ← Policy-based LLM safety check
      ├── consciousness.py     ← Background thinking loop (with progress emission)
      ├── consolidator.py      ← Block-wise dialogue consolidation (dialogue_blocks.json). (v6.73.0) The consolidation cursor is GENERATION-AWARE: on a chat.jsonl rotation the stored `chat_log_signature` locates its generation in the ordered `archive/chat_*.jsonl` chain and consolidation continues over `archives[i:]+live` (per-segment signature discipline), so the pre-rotation tail is never dropped; an unfindable generation (manual deletion/corruption) appends an explicit durable `[MEMORY GAP]` block instead of a silent offset reset
      ├── memory.py            ← Scratchpad, identity, chat history
      ├── project_facts.py     ← Thin per-project facts store (Phase 3b): project_id resolution (explicit `--project-id` or stable workspace-path hash) + a per-project knowledge dir under the canonical data dir (`projects/<id>/knowledge`), isolated from `memory/knowledge` and from the forked seed; v6.32.0 adds per-project journal/workpad path helpers
      ├── task_tree_ledger.py  ← (v6.38.0) Task-tree coordination ledger keyed by `root_task_id` — the domain-agnostic swarm blackboard + typed child→parent beacons. Append-only `data/task_trees/<root>/blackboard.jsonl` (size-capped, validated, GC-eligible with the tree); kinds: contract/decision/fact/note (coordination) + milestone/partial_finding/blocker/question/interface_contract/delegation_constraint (beacon). `delegation_constraint` rows carry a structured payload (`constraint_id`, closed-enum directive, scope, rationale) and are consumed at subagent admission unless a later decision row explicitly overrides them with a reason; prose is never authoritative. EPHEMERAL swarm coordination — distinct from the DURABLE project journal. Exposed via the `tree_note`/`tree_read` tools (`ouroboros/tools/task_tree.py`); the tail is injected into context each turn; a blocker/question/interface_contract/delegation_constraint beacon early-returns a parent's sliced `wait`; aged out by `headless.prune_task_trees` once the root task is terminal. (v6.39: on the swarm ROOT's terminal, the high-signal rows are mirrored into the DURABLE project journal — see "Letters home" — so they survive this tree's GC.)
      ├── projects_registry.py ← Multi-project registry: durable `data/state/projects.json` with immutable id/chat identity, 80-character display names, working-folder facts, `active|deleting|tombstoned` lifecycle, routing fence/generation, and visible revision; deletion preserves bindings/history/folder/memory and an id can never resurrect
      ├── project_dialogue.py  ← Read-only canonical Project dialogue lens plus append-only presentation annotations (`logs/chat_annotations.jsonl`); projects reference original chat rows instead of copying/mirroring them, and the sidecar never owns routing or Project state. (v6.73.0) `build_owner_message_ref` builds the origin identity AT INGRESS (identity by value — the content-hash lookup `find_owner_message_ref` was deleted as the anti-pattern instance); `project_origin_rows` returns the binding-held origin refs+texts the history lens synthesizes when a canonical row left the read window
      ├── project_lease.py     ← One-writer-per-project lease (v6.32.0): `assign_tasks` serializes top-level tasks of the same STORED `project_id`; same-project subagent swarm exempt; `project_id==""` is no lane
      ├── context.py           ← LLM context-source builder and public compatibility API for consciousness / ordinary-task message assembly
      ├── context_fit.py       ← Ordinary Main task-local ContextFitPlan: deterministic Max/Low projections from one immutable captured core, exact-route capability/calibration fit, no routing/retry/global-mode authority; commit/scope review stay outside this path
      ├── context_budget.py    ← Context-window budget SSOT for low/max profiles, raw-tail sizing, compaction thresholds, and static section limits
      ├── capability_evidence.py ← Sourced, route-fingerprinted context-window EVIDENCE (v6.33.0): provider `/models` metadata, local n_ctx, or owner-ack; each claim carries a status (`confirmed`/`asserted`/`unprobeable`/`failed`); `confirms_at_least` is fail-closed; a provider outage marks evidence stale and never erases a prior confirmed record; persisted to `data/state/capability_evidence.json`. The SSOT the ≥1M scope-reviewer floor and the max-mode gate consult. Also stores learned effort ceilings (v6.57.0), keyed by NORMALIZED MODEL IDENTITY (effort support is a model property — coarser than the per-route window records, disclosed r4): an effort-implicating provider rejection learns a ceiling one step below the requested effort (FLOORED at "low" — the lowest thinking tiers never poison a route to none, v6.61.1); later calls clamp down to it and the clamp is DISCLOSED in that call's usage event as `reasoning_effort_clamped={requested, applied, reason}` — never silent (BIBLE P1). v6.73.2 adds the symmetric learned effort FLOORS (`effort_floors` namespace): a MANDATORY-value rejection at a bottom-tier effort ("Reasoning is mandatory ... cannot be disabled" — the value-forbidden mirror of the too-high case) learns a floor of "low" for the model; later calls clamp UP into the `[floor, ceiling]` band with a direction-derived disclosure reason (`learned_floor`). WHY FLOOR, NOT DROP: dropping (and durably remembering) the reasoning carrier on such an error would strip effort control for EVERY lane of that model — including blocking reviewers at high — for the cache window; the floor preserves the carrier and raises only the forbidden bottom value. WHY THE LIFECYCLE ASYMMETRY: ceilings are sticky (a model's max supported effort is a stable model property) while floors EXPIRE in 14 days like `rejected_params` — whether reasoning can be disabled is provider POLICY that changes, and relearning costs one reactive 400; the llm.py floor cache re-syncs hourly (replace-not-union) so long-running processes heal like restarts
      ├── context_layout.py    ← Reference-doc layout SSOT: max/low doc tiers, ARCHITECTURE navigation map, DEVELOPMENT full/pointer policy, README/CHECKLISTS on-demand pointers
      ├── context_compaction.py ← Context trimming and summarization helpers
      ├── headless.py          ← Headless task child-drive isolation, workspace patch artifacts, and memory export helpers
      ├── coop_checkpoint.py   ← (v6.58.0) `checkpoint_commit_coop_roots` — at ROOT-task finalization, dirty host-minted genesis/coop trees get a local checkpoint commit (`user.name=Ouroboros`); credential-shaped files excluded (disclosed, headless sensitive-pattern SSOT), owner-attached folders NEVER auto-committed, skipped while tree tasks are live, fail-soft per root
      ├── subagents.py         ← Subagent model-lane resolution, task-group compaction, and structured lineage/usage envelopes
      ├── subagent_worktrees.py ← Acting self_worktree lifecycle: provision/remove/prune isolated git worktrees (outside repo/ and data/) + durable registry (state/subagent_worktrees.json) + cross-process ops lock; startup orphan reconciliation; also provisions durable from-scratch genesis projects (provision_genesis_project, never registry/GC)
      ├── artifacts.py         ← Task-scoped artifact helpers shared by user-file tools, process outputs, and outcome finalization. (v6.52.0, P1) `stage_task_attachments` stages every task's INPUT attachments (CLI/API, GAIA solver, desktop chat) into the agent-readable `artifact_store/attachments/` (skips secret SOURCES via the tool_access SSOT blocklist, bounded), returning a manifest of `read_file(root='artifact_store', path='attachments/<name>')` entries; `collect_task_artifact_records` EXCLUDES that subdir so staged inputs are never recorded as deliverables. (v6.52.2) `record_task_scratch`/`read_task_scratch_fingerprints` persist {abs_path: sha256} FINGERPRINTS of the run_command/run_script `scratch=[...]` ephemeral-verification files to `.scratch_manifest.json` (written to BOTH budget + live drive roots) so `headless.write_workspace_patch_artifacts` EXCLUDES a file from the workspace patch ONLY while its current content still matches (a later real file at the same path is never dropped). (v6.56.0) scratch declarations are IDEMPOTENT/ADOPTABLE: re-declaring a manifest path is ok, and an existing untracked in-cwd file may be adopted — its sha is recorded via the same SSOT writer at declaration time, so the sha-gate still excludes it only while unmodified (tracked / outside-cwd / outside-worktree declarations stay blocked); the undeclared-output guard stat-verifies candidates POST-exec (exists + mtime ≥ start−slack) for both run_command and run_script, so import strings/CLI flags/heredoc bodies no longer read as writes
      ├── retention.py         ← Unified GC retention SSOT: clamp/age-cutoff helpers + legacy-key seed picker used by worktree/task-drive/service-log startup pruning
      ├── workspace_preflight.py ← Read-only external-workspace git/manifest/toolchain snapshot used by gateway task creation
      ├── project_sources.py   ← (v6.59.0) Project working-folder sources: attach an existing owner folder (resolved-realpath validation — exists/dir/not-home-root/no repo-data overlap; opt-in `init_git` attach-snapshot commit, NEVER auto-init) and server-side `git clone` into the durable projects root (atomic tmp→rename, `GIT_TERMINAL_PROMPT=0` + BatchMode ssh, typed `auth_required`); provenance (attached|cloned|genesis|none) + `clone_url` are recorded on the registry as historical facts, `trusted_at` stamps automatically (notification trust model — attaching IS the owner's grant)
      ├── workspace_admission.py ← (v6.58.0) Workspace-task admission SSOT: the ONE workspace-root validator (git-worktree root, no repo/data overlap) shared by `/api/tasks` and the promote path; `resolve_room_workspace` defaults a project-room task to the room's registered `working_dir` (sentinel `workspace="none"` opts out) and LOUD-FAILS a set-but-broken working_dir (a room task must never silently degrade to a workspace-less self_modification-profile task); `compose_workspace_block` renders the shared [HEADLESS_WORKSPACE] guidance; `bounded_workspace_preflight` hard-caps the promote-path snapshot so the supervisor event-drain thread stays responsive
      ├── local_model.py       ← Local LLM lifecycle (llama-cpp-python)
      ├── local_model_autostart.py ← Local model startup helper
      ├── deep_self_review.py   ← Deep self-review: Generated Deep Self-Review Atlas repository context + full memory whitelist → 1M-context model. Guaranteed-fit assembly (v6.27.1): the in-prompt OMITTED-files section is bounded (counts per reason + capped sample; full coverage stays in the persisted atlas manifest) and reserved inside the atlas fixed budget; atlas budget_exceeded retries once with the compact manifest, and a final-shrink rebuild (tighter hard budget by the measured overage) replaces the historical fatal 'Review pack too large' error — the gate remains as the fail-closed last assertion. File selection is ranked by import-graph centrality (reverse-import in-degree from code_intelligence, additive bonus ≤600, deep-review-only)
      ├── review.py            ← Code collection, complexity metrics, pre-commit review
      ├── preflight_runner.py  ← Hermetic serial reviewed-change pytest gate: disposable git worktree, candidate diff replay, temp data/settings/pycache env, and live OUROBOROS_*/secret-class scrub so review tests cannot inherit operator behavior or mutate live repo/data
      ├── review_substrate.py  ← Reviewer-slot coordinator used by task acceptance and planning helpers; duplicate model ids remain independent slots. Actor records keep transport status, parse status, semantic verdict, model/provider, role, coverage, quorum contribution, reason, enforcement impact, and review-binding hashes distinct; only a compact projection reaches task/event/UI records. Task acceptance enforces adaptive quorum, one substantive call and no more than two physical attempts per actor, metric-grounded criterion evidence, provenance, and a public-info-only anti-cheat boundary. Commit/triad/scope P3 orchestration remains a separate one-pass contract.
      ├── review_state.py      ← Durable advisory pre-review state (advisory_review.json)
      ├── triad_review.py      ← Shared multi-model review primitives: JSON-array extraction is reused by repo + skill review; per-actor records, quorum/degraded accounting, and model-error events power the skill-review path
      ├── onboarding_wizard.py ← Shared desktop/web onboarding bootstrap + validation
      ├── settings_setup_contract.py ← SSOT for Settings/Onboarding setup contract, derived bootstrap state, and setup payload validation
      ├── owner_mailbox.py      ← Per-task user message mailbox (compat module name)
      ├── launcher_bootstrap.py ← Bundle-to-repo bootstrap and managed sync helpers (used by launcher.py)
      ├── provider_models.py   ← Provider-specific model ID helpers, direct-provider defaults (OpenAI, Anthropic, Cloud.ru, GigaChat)
      ├── runtime_mode_policy.py ← Runtime-mode protected-path policy (safety-critical files, frozen contracts, release/managed invariants) shared by registry, git tools, and Claude gateway guards
      ├── schedule_contract.py ← Schedule id, 5-field cron, and IANA timezone validation SSOT shared by gateway, manifests, and supervisor queue
      ├── reflection.py        ← Execution reflection and pattern capture
      ├── post_task_evolution.py ← Post-task self-evolution (V4 owner envelope + V5 LLM-first promotion): a worker writes a durable promotion signal; the supervisor idle tick applies it through the existing gated evolution enqueuer (one-shot autostop). Never enqueues from the worker; never fires from evolution/subagent tasks.
      ├── repo_remotes.py      ← Role-based GitHub remote provisioning: official update source (`managed`) stays read/update-only, personal persistence target (`origin`) can be auto-forked/configured from GitHub token
      ├── review_evidence.py   ← Structured review findings/obligations snapshot for summaries and reflections; (v6.51.0) `build_task_acceptance_evidence` — the process-aware acceptance packet (full task contract + first-class verification_summary + bounded/redacted tool-call trajectory + leak-safe artifact manifest + `__provenance__` tags) under a disclosed-truncation budget, shared by the agent-tool and host-forced acceptance paths; (v6.53.0) host-built `acceptance_support_refs` links Observable Acceptance Claims to actual verification receipts by `criterion_id` so expected support prose is never credited as evidence by itself; (v6.54.0) linked receipt refs carry host-attested artifact-lifecycle/missing-after facts when present, and the agent may add an advisory disposition/rationale under its agent-supplied evidence; (v6.78.0) the `verification_summary` renders receipt identity through the shared `_outcome_receipts.receipt_identity_projection` (SSOT with the fixed ledger row) and projects the UNRECONCILED RED's identity (`unreconciled_red_identity`) beside the latest receipt's — a later green of a DIFFERENT verification leaves an earlier red standing, so `unreconciled_red=true` beside a green `latest_*` would otherwise be a flag whose cause reaches no reviewer; (v6.71.1) a host-attested `acceptance_obligations` catalog (id/item/recommendation/status) rides alongside so the reviewer can adjudicate the agent's per-obligation dispositions as rebuttals (joinable by id; the disposition REASON stays under agent_supplied for clean provenance), and the bounded tool-call trajectory per-result cap follows the actor's own PER-TOOL window (SSOT `TOOL_RESULT_LIMITS`, fallback `DEFAULT_TOOL_RESULT_LIMIT`) — the ACCEPTANCE reviewer sees each tool result at the actor's own window (with the actor's own truncation marker preserved), never a narrower hidden trace copy; reflection's `_ERROR_MARKERS` trigger deliberately keeps scanning the historical 350-head+350-tail view (runtime markers are prefix/suffix-emitted; a wider scan false-positives on doc bodies that quote the marker strings), while its error snippets embed from the full redacted result
      ├── semantic_dedup.py    ← Shared LLM-first semantic-duplicate detector (C9.6) for free-text items (backlog nominations, review obligations): one light-model call after an exact-match MISS, biased to false-DUP / never false-MERGE, exact-id validation, fail-open (None on empty/no-candidates/transport/parse failure); consumed by improvement_backlog.py and review_state.py
      ├── skill_loader.py      ← Skill discovery + durable skill state (v5.8.2: walks data/skills/{native,clawhub,ouroboroshub,external}/ + optional OUROBOROS_SKILLS_REPO_PATH; persists to data/state/skills/<name>/; tags each LoadedSkill with `source` and `.self_authored.json` provenance; v5.19 computes review verdicts live from stored findings; v6.85 resolves manifest-declared enabled-skill conflicts symmetrically)
      ├── skill_readiness.py   ← Central skill readiness helper: combines review gate, stale hash, enablement, grants, and enabled-peer conflicts into a single finalization/execution verdict
      ├── skill_dependencies.py ← Shared dependency-spec resolution for skill payloads across manifests, sidecars, and provenance
      ├── skill_publish_eligibility.py ← (v6.47.0) SSOT predicate for skill→hub publish eligibility (`submit_hub:{visible,disabled,reason}`); imports only config-level review-status constants, consumed by the publish gate (`tools/skill_publish.py`) + the gateway serializer (`gateway/extensions.py`) + the Skills card, ending the clean-vs-advisory-warnings desync
      ├── skill_review_status.py ← Skill-review verdict aggregation SSOT (FAILs → clean/warnings/blockers/pending; hard trust-boundary items block on FAIL, bug_hunting + selected conditional safety items follow severity; enforcement maps verdicts to executable_review)
      ├── skill_review_passes.py ← (v6.41.0) Skill-review pass runner: one multi-model review pass, or a chunked per-pack pass (with per-chunk parseable quorum) when an over-budget skill is split — merged into one verdict (P5 token budget)
      ├── skill_review.py      ← Skill review pipeline: deterministic preflight + optional fail-open Claude Code advisory over the skill payload only (repo diff excluded, Skill Review Checklist coverage contract, scope-review effort, raw/session metadata plus parsed_items/contract_warning persisted as advisory_result) followed by the tri-model executable trust gate against the Skill Review Checklist section of docs/CHECKLISTS.md plus minimal host skill/widget context (CREATING_SKILLS.md, PluginAPI contract, extension UI validator); supports rebuttal/history/convergence evidence
      ├── skill_review_history.py ← Append-only Skill Review history helpers: group-wide rounds, per-snapshot attempts, legacy read-time ordinals, and job-idempotent terminal rows
      ├── extension_loader.py  ← Phase 4 loader for type: extension skills; imports no-dependency pure-Python extensions in-process with PluginAPIImpl, but catalogs isolated-dep/native-marker extensions through child-process proxies so plugin import cannot abort server.py; tracks registrations per-skill for atomic unload
      ├── extension_process_runner.py ← Short-lived child-process runner for isolated-dep/native-marker extension catalog/tool/route/WS dispatch; uses scrubbed env, per-skill deps, process-group tracking, timeout/output caps, and returns graceful host errors on child crash
      ├── extension_ui_validation.py ← One host-owned recursive declarative-schema-v1 validator shared by extension loader and skill preflight; exact tree paths, stable identity, depth/node budgets, passive-subscription enforcement
      ├── extension_isolated_deps.py ← Per-extension bridge for legacy/forced in-process isolated-dep tests; production reviewed isolated deps are exposed only inside extension_process_runner children
      ├── extension_health.py  ← Durable per-extension health vector (data/state/skills/<name>/health.json): live->broken regression memory across restarts, surfaced via health invariants + startup check + Installed UI
      ├── skill_token.py       ← Opaque Host Service API token wrapper used by reviewed skills/companions
      ├── marketplace/         ← ClawHub + OuroborosHub marketplace package (clawhub.py registry client, ouroboroshub.py static GitHub catalog client, fetcher.py staging, adapter.py OpenClaw->Ouroboros translation, install.py orchestration, isolated_deps.py per-skill dependency prefix, provenance.py durable provenance)
      ├── skill_lifecycle_queue.py ← single FIFO lane for mutating skill lifecycle actions (install/update/review/deps/enable/disable/uninstall) with recent event snapshot for Skills UI, chat live-card progress, dedupe keys, and sync tool wrapper
      ├── skill_review_runner.py ← shared lifecycle-backed skill review runner for API + agent tool paths; writes review_job.json + skill_review_* events and routes all executable skills (including self-authored provenance) through tri-model review
      ├── server_auth.py       ← Non-localhost auth gate (OUROBOROS_NETWORK_PASSWORD)
      ├── server_control.py    ← Process-control helpers: restart, panic stop
      ├── server_entrypoint.py ← CLI argument parsing, port-binding helpers
      ├── server_runtime.py    ← Server startup/onboarding and WebSocket liveness helpers
      ├── server_web.py        ← Static web file helpers (NoCacheStaticFiles, web dir resolver)
      ├── task_continuation.py ← Durable per-task review continuation state across restart/outage
      ├── task_results.py      ← Durable task result/status files (task_results/<id>.json)
      ├── task_status.py       ← Effective task-status SSOT: child-drive result merge, lineage lookup, bounded waits
      ├── git_shell_policy.py  ← Structural git argv classifiers for shell safety guards
      ├── protected_artifacts.py ← Task-contract protected artifact policy helpers for execute-only black-box references
      ├── shell_parse.py       ← Shared shell argv/inline-command parser helpers used by guardrails without importing the tools package; (v6.51.0) `recover_stringified_argv` (the SSOT JSON/AST stringified-argv recovery shared by run_command + verify_and_record) and `normalize_check_argv` (the verify check→argv SSOT that the shell guard AND execution both call, so the guard inspects exactly what runs; string → non-login `sh -c`); (v6.78.0) `shell_tokens_typed` (THE tokenizer of the module — tokens paired with whether each is real SYNTAX rather than a literal argument that spells like one, a distinction `shlex` destroys when it strips quotes; `_normalize_shell_source` marks quoted/escaped punctuation on the way in and the mark never leaves), its text view `shell_tokens` (what `shell_segments` and the guards read — a quoted `&&` still reads as a separator there, which over-splits and is the fail-safe direction) and `canonical_command_text` (the comparison-stable form of a command: one space BETWEEN tokens, token contents and control operators verbatim, nothing dropped and nothing re-classified — the seam `_outcome_receipts` derives a verification's check identity from, so neither collapsing whitespace inside a quoted argument nor stripping a literal `'&&'` argument as if it were syntax can make two different checks compare equal)
      ├── workspace_executor.py ← Host-owned local/docker_exec workspace process backend, path mapping, executor traces, and executor service lifecycle
      ├── tool_capabilities.py ← SSOT for tool sets (core, parallel-safe, truncation, browser)
      ├── tool_access.py       ← Tool API v2 policy matrix: ToolProfile × ResourceRoot × Operation; also projects the side-effect-free filesystem affordance map injected into runtime context and checks closed-enum subagent required_capabilities against the selected profile
      ├── tool_policy.py       ← Round-one tool visibility policy (tool sets live in tool_capabilities)
      ├── utils.py             ← Shared utilities; v5.8.3-rc.2 SSOT for JSON atomic writes/reads, UTC timestamps, hashes, log sanitization, and subprocess helpers
      ├── world_profiler.py    ← System profile generator (WORLD.md)
      ├── contracts/           ← Frozen ABI (Phase 1 Protocols + TypedDicts + SkillManifest; Phase 4 adds plugin_api.py with PluginAPI + ExtensionRegistrationError + permission/route-method/forbidden-settings tuples; v6.53.0 task_contract adds advisory Observable Acceptance Claims)
      │   ├── tool_context.py  ← ToolContextProtocol (minimum tool ABI, duck-typed)
      │   ├── tool_abi.py      ← ToolEntryProtocol + GetToolsProtocol
      │   ├── api_v1.py        ← WS/HTTP envelope TypedDicts
      │   ├── chat_id_policy.py ← SSOT for human-visible vs synthetic transport chat ids
      │   ├── task_contract.py ← Canonical per-task contract draft/resource normalization helpers, including advisory `acceptance_claims` (`claim`/`surface`/`support`/`priority`) for host-built support_refs
      │   ├── task_constraint.py ← Structured per-task execution constraints: skill-repair payload confinement AND live subagent authority — local-readonly and acting (mutative) envelopes (VALID_WRITE_SURFACES, surface/write_root/base_sha/protected_paths_grant/external_tool_grants, parent_only_commit), normalized + fail-closed
      │   ├── skill_payload_policy.py ← Shared skill-payload path resolution policy for data/skills buckets, path confinement, and control-plane sidecar detection
      │   ├── skill_manifest.py ← Unified SKILL.md / skill.json parser (instruction|script|extension)
      │   ├── schema_versions.py ← Opt-in _schema_version helpers
      │   └── plugin_api.py    ← Phase 4: PluginAPI Protocol + ExtensionRegistrationError + FORBIDDEN_EXTENSION_SETTINGS + VALID_EXTENSION_PERMISSIONS + VALID_EXTENSION_ROUTE_METHODS
      ├── gateways/            ← External API adapters (thin transport, no business logic)
      │   └── claude_code.py   ← Claude Agent SDK gateway (edit path via ClaudeSDKClient lifecycle; read-only advisory path isolated in a Python child process with structured signal/timeout errors and normalized SDK usage)
      ├── gateway/             ← Gateway Boundary v1: all browser-facing HTTP/WS route ownership and frontend contract SSOT
      │   ├── contracts.py     ← PRO-frozen HTTP/WS envelope and endpoint index (canonical replacement for the legacy contracts/api_v1.py surface)
      │   ├── router.py        ← Starlette route collector for /api/* and /ws
      │   ├── ws.py            ← WebSocket connection manager, extension WS dispatch, browser broadcast helpers
      │   ├── state.py         ← /api/health and /api/state handlers
      │   ├── tasks.py         ← Headless task create/list/get/cancel/events endpoints over the supervisor queue
      │   ├── logs.py          ← Read-only runtime log tail endpoint for CLI/headless clients
      │   ├── settings.py      ← /api/settings, /api/owner/*, onboarding, Claude runtime status/repair handlers
      │   ├── control.py       ← reset, command, git/update, and evolution-data handlers; schedule_subagent surfaces effective_lane(s), wait_task emits a burst/absorb advisory when other children are still in flight, and the descriptions steer burst+absorb and cooperative-multi-builder (external_workspace, omit write_root) vs genesis
      │   ├── schedules.py     ← queue-backed cron schedule HTTP surface (list/upsert/delete)
      │   ├── files.py         ← File Browser + chat upload endpoints
      │   ├── ui_preferences.py ← owner-local UI preferences (`state/ui_preferences.json`): widget order, nested subagent expansion, and UI defaults
      │   ├── models.py        ← model catalog + local-model lifecycle endpoints
      │   ├── extensions.py    ← extensions/skills HTTP surface (GET /api/extensions, GET /api/extensions/<skill>/manifest, ALL /api/extensions/<skill>/<rest:path>, POST /api/skills/<skill>/toggle, POST /api/skills/<skill>/delete, POST /api/skills/<skill>/review, POST /api/skills/<skill>/grants)
      │   ├── marketplace.py   ← ClawHub + OuroborosHub HTTP surface
      │   ├── mcp.py           ← MCP Settings API surface backed by the shared MCPManager
      │   ├── host_service.py  ← Loopback-only Host Service API for reviewed skill callbacks
      │   ├── history.py       ← Chat history + cost breakdown endpoint factories
      │   ├── projects.py      ← Multi-project CRUD surface (v6.32.0): GET /api/projects, POST /api/projects, POST /api/projects/from-task (bind an existing task to a new project). (v6.33.0 removed the /sleep + /wake status endpoints.)
      │   └── _helpers.py      ← shared HTTP request root helpers, coercion, and JSON error envelope
      ├── tools/               ← Auto-discovered tool plugins
      │   ├── extension_dispatch.py ← Extension tool dispatch helper extracted from registry.py; preserves liveness, safety, async, and out-of-process error contracts
      │   ├── release_sync.py    ← Release-metadata sync library; advisory_review uses sync_release_metadata before provider spend when VERSION is in scope; _preflight_check uses check_history_limit for P9 row caps; agents can also call it directly for version-carrier sync
      │   ├── review_synthesis.py ← LLM-based commit-finding synthesis (fail-open to the original findings on synthesis error) plus the strict plan-review parser/aggregator and fingerprint-bound `review_disposition` validator; plan outcomes are exactly `GREEN`, `REVIEW_REQUIRED`, or `REVISE_PLAN`
      │   ├── ci.py              ← CI trigger and monitoring (GitHub Actions API)
      │   ├── claude_advisory_review.py ← Advisory pre-review tool (read-only Claude Agent SDK)
      │   ├── recent_tasks.py    ← Read-only context recovery tool exposing recent task_results summaries/traces for LLM-first continuation recovery
      │   ├── commit_gate.py     ← Advisory freshness gate and commit-attempt recording (extracted from git.py); `_record_commit_attempt` runs LLM-based claim synthesis (via `review_synthesis.py`) on blocked attempts before durable obligations are created
      │   ├── git_rollback.py    ← vcs_rollback tool (wraps git_ops.rollback_to_version)
      │   ├── git_pr.py          ← PR integration tools: fetch_pr_ref, create_integration_branch, cherry_pick_pr_commits, stage_adaptations, stage_pr_merge (non-core, require enable_tools)
      │   ├── github.py          ← GitHub integration: issues (list/get/comment/close) + PR tools: list_github_prs, get_github_pr, comment_on_pr (non-core; github.py is in _FROZEN_TOOL_MODULES so PR inspection/comment tools work in packaged builds)
      │   ├── parallel_review.py ← Parallel triad+scope orchestration and verdict aggregation (extracted from git.py)
      │   ├── plan_review.py     ← Pre-implementation design review (adaptive context levels, shared ReviewCoordinator slots, duplicate model IDs allowed, `plan_task` tool); one scout wave per exact fingerprint waits to a shared boundary, sends every ready non-empty handoff plus explicit omissions to the panel, exact-hash binds included snapshots when still current, and keeps every post-review scout change audit-only. (v6.61.0) Agent-declared `plan_class` (self_mod|external|creative|research) structurally escalates to self_mod when files_to_touch resolve under the system repo (path fact, P5); non-self_mod reviewers get BIBLE+DEVELOPMENT full but ARCHITECTURE as the lossless nav map, context_level defaults to minimal, and planning scouts are framed to the plan's own domain instead of repo archaeology.
      │   ├── review.py          ← Task acceptance review tool plus multi-review adapters backed by the shared review substrate
      │   ├── review_context_atlas.py ← Deterministic bounded-context compiler for scope_review, plan_task, and deep_self_review; raw-inlines selected files and accounts for every tracked path in the manifest. Optional additive `centrality_scores` (rel_path→bonus) consumed in candidate scoring; empty default keeps scope/plan selection byte-identical (deep self-review is the only producer)
      │   ├── query_code.py     ← Read-only structured code intelligence tool (`query_code`) over the code inventory: symbols, definitions, references, callers/callees, impact, structural search, and relevant file ranking (v6.47.0: generalized `root=user_files` for read-only intelligence over an external target, e.g. a benchmark `/app`, with search_code-shape path guards + bounded symlink-safe structural walks)
      │   ├── media.py           ← (v6.52.0, P4b) Media tools: `ocr_pdf` (extract a PDF text layer; scanned/image-only PDFs return a typed `OCR_PDF_SCANNED_UNAVAILABLE` — true OCR is a deferred follow-up) and `youtube_transcript` (fetch a video's caption track over HTTP; web-gated via `_WEB_TOOLS`). Local-file tools reuse the view_image trust boundary; both are dependency-optional (graceful `*_UNAVAILABLE`). (v6.53.0) `extract_video_frames` optionally uses `ffmpeg` from PATH when available, writes bounded frames under `artifact_store/video_frames`, and returns typed `EXTRACT_VIDEO_FRAMES_UNAVAILABLE` when absent (no ffmpeg bundle added); (v6.54.0) it is wired into the same core/local-readonly/acting-subagent tool-capability envelopes as its sibling media tools
      │   ├── verify.py          ← (v6.47.0) `verify_and_record` core tool: the HOST runs the agent's declared verification `check` through the same PRE-EXECUTION machinery as run_command — the registry shell-guard (`_SHELL_GUARDED_TOOLS`: subagent-secret/protected-artifact/sudo, protected-root/workspace-state/light-mode writes — the security boundary that BLOCKS a forbidden mutation before the handler runs), `bootstrap_process_path`, the executor backend (`docker_exec` network=none routing) when the cwd is executor-mapped, else the tracked local subprocess — then writes a durable host-attested receipt (DISCLOSED truncation) to `<drive_root>/task_results/artifacts/<task_id>/verification_receipts.jsonl`. It is deliberately NOT in `_PROCESS_COMMAND_TOOLS`: those POST-execution checks (owner-restore, light-repo diff, git-ref tripwire) run AFTER the handler has already written the receipt, so they would not gate it — the pre-exec guards already do. Receipts feed the verification ledger and suppress the `receipt_absent` flag (verify-before-done flagship, FR3). (v6.50.2) An `expected_match` mode (substring default · exact · exact_line · json_equals) records how `expected` was matched into the receipt; anti-cheat: verify only against PUBLIC task info (no hidden /tests/, solution.sh, copied verifier, or online answer). (v6.51.0) Check normalization is the SSOT `shell_parse.normalize_check_argv` (the shell guard inspects EXACTLY the normalized argv that executes) — a stringified-argv `check` is recovered to argv (no more `sh -lc '["go","test"]'` exit-127), and a genuine string runs via a NON-login `sh -c` so it inherits the bootstrapped PATH (parity with run_command). (v6.52.0, C) After-only artifact-lifecycle FLAG: when the agent declares `artifact_paths` on a run-kind check, the host probes their existence AFTER the check via the SAME surface (executor when cwd-mapped, else host) and records `artifact_lifecycle`/`artifacts_missing_after` on the receipt — FLAG-ONLY (status stays `pass`), carried through the verification ledger's fixed key-set and surfaced to the ADVISORY acceptance reviewer, catching a check that built then DELETED the deliverable it just attested (e.g. compile+import+rm a `.so`). (v6.52.2) FLAG-ONLY exit-masking sensor (`_check_has_exit_masking`, shlex token-scan of a `["sh"/"bash",-c,text]` check): a pipeline that can launder the real exit code (`... | tail`/`grep`/`sed`, `|| true`, `>/dev/null`) records `check_exit_masking`/`check_exit_masking_reasons` on the receipt (status UNCHANGED) — projected into the verification ledger's fixed key-set, aggregated into the acceptance reviewer's `verification_summary`, and feeding a one-shot advisory masked-verification nudge — so a PASS over a possibly-laundered green is reconsidered (decides nothing; P5)
      │   ├── review_helpers.py  ← Shared review helpers (section loader, touched/head packs, intent, pytest preflight via agent interpreter)
      │   ├── review_revalidation.py ← Reviewed-commit fingerprint revalidation helpers (blocks when staged diff changes after review)
      │   ├── scope_review.py   ← Scope reviewer (enforcement-aware, budget-aware)
      │   ├── scope_review_contract.py ← Pure scope-output parser and one-pass validity contract; owns no routing, retries, or reviewer state
      │   ├── services.py        ← Task-scoped long-running service mini-manager: start/status/logs/stop with process-group cleanup and retained private log blobs
      │   ├── skill_exec.py      ← Phase 3 external-skill surface: list_skills, skill_review, toggle_skill, skill_exec (subprocess runner with cwd confinement, env scrubbing, timeout, runtime allowlist python/python3/bash/node/deno/ruby/go; gated by enabled + fresh executable review + fresh content hash — v5.1.2 Frame A: runtime_mode no longer blocks execution)
      │   ├── skill_publish.py   ← Agent-callable `submit_skill_to_hub` tool: validates a fresh no-blocker review — `clean` or advisory-only `warnings` (v6.27.1; advisory findings are disclosed in the PR body under `## Known advisory findings`; blockers/pending/stale still refuse) — for a local skill (sources `external`/`self_authored`/`user_repo`/`ouroboroshub`/`clawhub`; `native` only when no `.seed-origin` marker), infers OuroborosHub from `OUROBOROS_HUB_CATALOG_URL`, commits payload + catalog update to the user's fork via GitHub GraphQL, and opens a PR without mutating the local Ouroboros repo. For marketplace-managed sources the generated PR body is force-prefixed with a `## Provenance` block read from the local sidecar (`.ouroboroshub.json` slug / `.clawhub.json` clawhub_slug); when no sidecar exists the source is reclassified as `external` by skill_loader and submit proceeds without the block.
      │   ├── skill_preflight.py ← Heal-safe read-only payload preflight: manifest/syntax checks plus registration-aware literal UI-schema resolution; unresolved dynamic schemas are explicit degraded skips, with runtime validation still fail-closed
      │   ├── project_journal.py ← Thin per-project journal/workpad tools (v6.32.0): journal_write/read (durable milestone memory), workpad_read/write (scratch page), journal_tail_digest (context injection); over-limit writes are rejected, never silently sliced
      │   ├── task_tree.py     ← (v6.38.0) Task-tree coordination tools tree_note/tree_read (the swarm blackboard + child→parent beacons; storage/kind SSOT in ouroboros/task_tree_ledger.py)
      │   ├── join_ledger.py   ← Soft-join decision authority: validates direct lineage and exact current child-result hashes for tagged `tree_note(kind="decision")` dispositions (`integrated`, `irrelevant`, `deferred`), appends the sole authoritative task-tree row, rejects stale hashes as `CHILD_RESULT_STALE`, and keeps `peek_task`, `discard_child_result`, constraint override, cancellation, and shared child-decision helpers. The hash covers status, full result, trace summary, artifact status, and stable artifact identities, not cost/timestamps/queue diagnostics/parent decisions; task-result fields are derived read projections only.
      │   └── subagent_integration.py ← integrate_subagent_patch: parent's manifest-first integration of an acting subagent's workspace.patch. For self_worktree children it applies into ctx.active_repo_dir() (sha256-verified, 3-way --index, protected-path gated, top-only lineage check, genesis refused), stages but never commits. For external_workspace children it verifies the child wrote in the same active external workspace and records an audited verdict without re-applying the patch; (v6.58.0) a NON-workspace parent integrating a COOP child (write_root = a host-minted tree under the subagent-projects root) gets a read-only verification + a SUCCESSFUL `coop_already_in_tree` no-op verdict instead of a parent-missing error — the work is already in the shared tree, which `coop_checkpoint.checkpoint_commit_coop_roots` checkpoint-commits at root finalization. Also compare_subagent_patches: read-only best-of-N helper that shows several children's candidate patches side by side for LLM-first synthesis
      └── platform_layer.py    ← Cross-platform process/path/locking helpers

      ouroboros/process_custody.py ← Supervised spawning + durable orphan ledger
      (v6.26.0): `spawn_supervised()` records every long-lived child in
      `data/state/process_ledger.jsonl` ({pid, pgid, fingerprint{start_time,
      cmd_sha256}, purpose, scope task|session|daemon, owner_task, session_id});
      the reaper (server startup + 10-min supervisor tick) kills entries whose
      generation/task owner is gone, matching by STRICT fingerprint only —
      never by command-line class, so dev and packaged instances can coexist.
      Genuine `daemon` entries are kept; skill companions (daemon scope,
      `purpose companion:<skill>:<name>`) are the exception (v6.36.2) — reaped on
      owner-uninstall or a foreign generation, **log-only by default**
      (`enforce_companion_reap=False` → `process_would_reap`), fail-safe
      (unknown live-skill set ⇒ keep-all).
      `start_parent_lifeline()` gives our python entrypoints (workers,
      extension runner, claude readonly child) a ppid watchdog that
      group-suicides when the parent dies. Panic layers (`_active_subprocesses`,
      port sweeps, Windows Job Objects) are unchanged complements.

# Build & CI (not part of runtime)
.github/workflows/ci.yml     ← Five-tier CI (quick / full / integration / skill smoke / build+release)
build.sh                      ← macOS build (PyInstaller → .dmg)
build_linux.sh                ← Linux build (PyInstaller → .tar.gz)
build_windows.ps1             ← Windows build (PyInstaller → .zip)
scripts/build_repo_bundle.py  ← Builds `repo.bundle` + `repo_bundle_manifest.json` for packaged releases
scripts/run_external_review.py ← dual-lane non-committing review wrapper. The default operator lane reviews the staged tree through the production advisory→triad→scope cycle with resolved production policy. `--contributor` reviews an exact committed target-base..head proposal in a detached target-base checkout, forces shipped target-base triad/scope models and efforts through OpenRouter with blocking enforcement, excludes Claude advisory, forbids contributor VERSION allocation, and emits a redacted SHA-bound `review-evidence.json`/`review-packet.zip`. Contributor `READY_FOR_INTEGRATION` is triage evidence, never merge authority: final version carriers and production review belong to the maintainer squash landing. Both lanes use the production triad/scope substrate and fresh non-live observability roots.
scripts/run_plan_review.py ← v6.43.0 operator plan-review tool: invokes the reviewer-panel portion of `ouroboros.tools.plan_review` from outside the runtime, loading BIBLE/DEVELOPMENT/ARCHITECTURE/CHECKLISTS, the proposed plan, optional touched-file snapshots, and optional generated Atlas context. Inputs: `--plan`, explicit `--context-level`, optional `--files-to-touch`/`--extra-context`/`--drive-root`. Output: full raw reviewer responses plus coordinated plan-review output to stdout (and optional `--output PATH`), with no truncation. It deliberately skips the live planning-scout swarm because that requires a running worker/supervisor environment. Not part of the runtime gate; review-exempt dev tool.
scripts/cleanup_test_pollution.py ← Dry-run-first cleanup utility for local test-pollution artifacts: known test skill state dirs, stale `__extension_imports`, and accidental `MagicMock`-named repo-root files. Use `--apply` only after inspecting planned removals.
devtools/benchmarks/        ← Tracked operator benchmark tooling (ProgramBench, Terminal-Bench/Harbor, SWE-bench, SWE-bench Pro, OSWorld step-loop/log tools, harness_bench_fast wrapper). It is reviewed when touched, is manifest-accounted by Atlas, is not imported by runtime core, and is not packaged as runtime app code. Adapters write generated run sidecars (manifest/result-ledger schemas using adapter-specific default filenames such as `run_manifest.json`, `result_index.jsonl`, `<predictions>.run_manifest.json`, `<predictions>.ledger.jsonl`, `osworld_preflight.*`, `disclosure_ledger.json`, or E1v2 summaries) only under explicit benchmark output roots outside `repo/` and outside live runtime `data`. (v6.75.0) `devtools/benchmarks/common/manifests.py` is the single home for run provenance: `benchmark_run_manifest()` now carries the universal seed gate (`require_clean=True` by default, `expect=` pin, recorded `seed_gate` block; the launcher escape is `--allow-dirty-seed`). The migrated launchers — seven in v6.75.0 (both ProgramBench launchers, `harness_bench_fast/run_harness_bench_fast.py`, `swe_bench/swebench_predictions.py`, and SWE-Pro's `pro_predictions.py`, `e1v2/run_pro.py`, `e1v2/auto_run.py`) plus three in v6.79.0 (`gaia/run_gaia.py` and both Terminal-Bench launchers, `terminal_bench/run_tb.py` and `terminal_bench/run_harbor_smoke.py`) — build that manifest ONCE right after argument parsing/readiness, write it to disk immediately (a refusal after admission still leaves a durable record carrying a typed `refusal` block and the exit code), keep the dict, augment it, and rewrite it at the end with a final `outcome`, so the gate decides BEFORE the first paid task and the run's own record says how it ended; the earlier pattern of writing it after all the spend meant the gate could not stop an unreproducible run (SWE-Pro's `e1v2/run_pro.py`/`auto_run.py` wrote no manifest at all). That lifecycle is carried by two shared seams rather than per-launcher convention: `admit_benchmark_run()` builds the manifest, writes it and only then enforces (the refusal raises `BenchmarkAdmissionRefused`, a `RuntimeError` carrying the payload, so a refused run leaves the same durable record an admitted one does), and `finalize_run_manifest()` is the single finalization seam — a context manager that merges `outcome`, `exit_code` and a typed `refusal`/`error` into the retained manifest on every exit path, including an escaping exception, which previously left `outcome: started` behind. (v6.76.0) That contract is enforced by ONE shared structural gate, `devtools/benchmarks/common/launcher_audit.py`, rather than by per-launcher review: `audit_all_launchers()` names every migrated launcher — ELEVEN as of v6.76.0 (the seven of v6.75.0 plus `continual_learning/run_clb.py` and all three OSWorld launchers), FOURTEEN today after v6.79.0 migrated `run_gaia.py`, `run_tb.py` and `run_harbor_smoke.py`, with `PENDING_LAUNCHERS` now the empty tuple — and reports every violation of THREE invariants in one pass — (A) ADMISSION IS THE OUTER BOUNDARY, (B) CONFINEMENT IS COMPUTED FROM THE ACTIVE CHECKOUT, and (C) THE FINALIZATION SEAM'S EXIT IS THE ONLY PUBLISHER: since `finalize_run_manifest()` merges the terminal `outcome`/`exit_code`/`refusal` into the manifest only when its context EXITS, a manifest written from INSIDE that context publishes a pre-merge record (on a refusal, the admission seam's generic payload saying `exit_code` 1 while the process will exit 2) that a concurrent reader can observe and an interruption makes durable, and which the seam overwrites on exit anyway. Like (A), (C) judges by EFFECT rather than by callee name — a call is a publication when following its body, local or imported, reaches a write primitive whose DESTINATION names a `*run_manifest*.json`; the offenders were called `_write_task_records` and `_write_cu_outcome`, named for the records they keep. Each primitive's destination is derived from its REAL signature (`os.rename(src, dst)` publishes at argument 1, `pathlib.Path.write_text(self, data)` at its receiver or at argument 0 when called flat), never from a hand-written position table, and a write form no signature can place is REPORTED as unresolved rather than assumed harmless. The seam's own write is deliberately not caught: it writes the path it was handed and names no artefact, which is exactly the difference between publishing on exit and publishing early. Recording a manifest path in a payload is likewise not a publication — CL-Bench's `results.json` lists pointers to the runner's sidecar manifests — so only destinations are inspected — plus the seam shape itself (a launcher that pairs `benchmark_run_manifest()` with its own `write_json()` again, or skips finalization, or evaluates `runtime_attestation()` inside the admission argument list where Python's argument-before-call evaluation order defeats the durable refusal). It audits SOURCE TEXT through `audit_source()`, so the gate is pinned against a SYNTHETIC violating launcher and not merely against code that happens to be clean today; a separate property test drives each launcher's `main()` into a refusal path and asserts the recorded `exit_code` IS the status the process exits with. Invariant A is maintained by what a call DOES, not by its name: it resolves TWO hops of helper definitions, LOCAL AND IMPORTED (first-party modules are opened and read; stdlib/third-party callees stay unresolved and are covered by the name/prefix denylist), and reports `helper -> token`. The imported hop is the fix for a defect that survived six review rounds — `ensure_outside_repo` MKDIRS the directory it validates and is imported, so a resolver that followed only LOCAL definitions could never see it, and it was caught only because somebody had thought to name it in the denylist; both `ensure_*` names are now deliberately ABSENT from that denylist and are caught by their bodies instead, which is what makes the NEXT unenumerated imported mutator catchable. The same walk found work hiding one level down in helpers the denylist does not name (`_ensure_vmrun_on_path` probing for `vmrun` and mutating `$PATH`, `_install_optional_dependency_stubs` mutating `sys.modules`, `repo_provenance` shelling out to git, `harness_bench_fast`'s `_read_task_ids` running `uv run … list` with a 60s timeout — its task-id DISCOVERY now runs after admission, which records the ids the CLI declared and attaches the discovered set plus the derived official command to the retained manifest). Branches that always leave the function are excluded, because they are the deliberate step-aside paths (`--collect-only`, 'another lane owns this task', 'these output paths are not confined') that exist to leave NO footprint and have no run to record against; the branch's test expression is still walked. Invariant A enforces the WIDER class it states — nothing that can FAIL, not merely nothing that MUTATES — because a run that dies reading its dataset leaves no manifest at all and is invisible rather than footprint-free. So the effect vocabulary also covers content READS and parses (`read_text`/`open`/`load`/`glob`…), network reaches, and two effects that are not callee names at all and are read out of a resolved helper's BODY: a deferred non-stdlib import (`from datasets import load_dataset` inside `load_pro_rows`, whose ImportError or offline hub killed the process pre-manifest) and a refusal that depends on probed state. The walk also descends into the ADMISSION CALL'S OWN ARGUMENT LIST, since Python evaluates arguments before entering the callee — that is where `pro_predictions` read every `--attestation` file. The line is drawn once and stated in the module: ARGUMENT-shaped work (argv parsing, pure path arithmetic, the confinement primitives that compute the manifest's own path) may precede admission because its refusals are a deterministic function of argv; WORLD-shaped work may not. A bare existence probe is the permitted middle — it reads no content and cannot fail on malformed input, which is what makes `scored_claim_state` a legitimate footprint-free step-aside — and becomes a violation the moment the helper holding it can raise: probing may not refuse, refusing may not probe. The four launchers this caught (`run_programbench_e2e`'s model-slot preflight and instance load, `swebench_predictions`' `_records`, `pro_predictions`' `_rows` and attestation read, `e1v2/run_pro`'s order/dataset/settings reads) resolve the chicken-and-egg the same way `harness_bench_fast` already did: admission records the DECLARED selector or input path (pure argv), and `requested_task_ids`/`requested_count` are AMENDED on the retained manifest once discovery has run inside the admitted run. (v6.76.0) CL-Bench and all three OSWorld launchers now ADMIT AT THE GATE through the same two seams: each builds its manifest once from pure argument derivation, persists it, and only then does the seed gate enforce, with a recorded `--allow-dirty-seed` escape and a final `outcome`/`exit_code` on every exit path — so a dirty or unidentifiable seed is REFUSED there, not reported afterwards. `run_step_agent.py` no longer rebuilds a refused manifest with `require_clean=False` (that recorded a waived gate on the one path where the gate had refused), the CU bridge and the skeleton no longer refuse before the durable record exists, and CL-Bench binds the gate to the EXECUTION clone (`--ouroboros-clone`, the checkout the external adapter boots its agent servers from) with the launcher's own provenance recorded alongside it under `extra.launcher_provenance` — gating the launcher's tree let a dirty execution seed pass whenever this checkout happened to be clean. A meta-test in `tests/test_devtools_benchmarks.py` covers all FOURTEEN migrated launchers and fails if one pairs `benchmark_run_manifest()` with its own `write_json()` again; the count in this document is itself pinned to `len(MIGRATED_LAUNCHERS)` by a test, because two phases previously edited this sentence independently and left the paragraph asserting eleven and ten at once. GAIA and both Terminal-Bench launchers left that residual in v6.79.0 — they route through the two seams, default to `require_clean=True`, and each carry the `--allow-dirty-seed` escape. The residual is now EMPTY and `PENDING_LAUNCHERS` is the empty tuple: CL-Bench and all three OSWorld launchers migrated in v6.76.0 and GAIA and both Terminal-Bench launchers in v6.79.0, so `MIGRATED_LAUNCHERS` names all FOURTEEN and the gate audits every one of them; the pending tuple stays declared so a launcher added later that is not yet under the contract must be named there rather than be missing from both lists. `runtime_attestation()` records BOTH facts about a live server — the HTTP `runtime_version` from the frozen `/api/health` contract and the local HEAD/VERSION of the checkout it was started from — and fails closed on a skew unless the named `OBO_ALLOW_EVOLVED_VOLUME=1` override is set (which it records). ONLY the contracted `runtime_version` counts as an identity: its absence is the non-overridable `runtime_version_absent`, because reading a generic `version` key would let any server that returns one attest as Ouroboros. That override is narrow by construction — `manifests.OVERRIDABLE_ATTESTATION_REASONS` — and waives only the deliberately accepted skew: `runtime_unreachable` (any transport/parse failure, so no live identity at all) and `commit_unavailable` (no commit to attribute the numbers to) stay fail-closed with it set, because waiving them would let admission continue while attesting nothing. `commit_lineage_ok()` compares a line of descent (`merge-base --is-ancestor`), never equality, because an evolution run legitimately moves HEAD forward. A refusal raises `RuntimeAttestationRefused` (a `RuntimeError`) carrying the constructed record, so a launcher persists the exact typed reason and the runtime/commit identities rather than a generic message. Every attaching launcher catches that type explicitly, not just `RuntimeError`: ProgramBench's e2e launcher (v6.75.0) and, from v6.76.0, all three OSWorld entry points — `run_cu_bridge_agent.py`'s pre-claim attestation, `run_step_agent.py::_preflight` (whose details the run manifest is amended FROM, so the loss propagated into the record) and `osworld_adapter_skeleton.py::preflight`, whose whole job is to report that evidence. Each keeps the carried record under `extra.runtime_attestation` — at that top level in all three, not buried inside a nested preflight block — and names the EXACT attestation reason (`runtime_skew`, `runtime_unreachable`, …) in its typed `refusal` with stage `runtime_attestation`, rather than a generic `preflight_failed` that conflates a runtime/checkout disagreement with a missing task file; a refusal that carries no record at all falls back to `runtime_attestation_failed`. One refusal-path test per site pins it. Every refusal in this area is one shape — `BenchmarkAdmissionRefused`, `RuntimeAttestationRefused` and `SeedShapeRefused` are all `RuntimeError` subclasses carrying a typed `reason`, deliberately NOT `SystemExit` (a `BaseException`, which made the launchers' handlers inert), and a launcher records the refusal and returns a nonzero code rather than re-raising, so the manifest's `exit_code` matches the process status. `manifests.CAMPAIGN_FATAL_PROVENANCE_REASONS` is the single authority for the volume-wide refusals that must stop a whole schedule (`stamp_absent`, `seed_mismatch`, `lineage_broken`, `runtime_skew`, `runtime_unreachable`, `seed_head_unreadable`): BOTH SWE-Pro drivers consume it — `e1v2/run_pro.py` stops its schedule (exit 2, typed `volume_provenance` refusal) and `e1v2/auto_run.py` stops the shard — because a per-driver copy left a direct `run_pro` run refusing every task and still exiting 0. It rides inside readiness paths that cannot be skipped: `IsolatedServer._wait_ready` (evolve_smoke + the CLB host engine), ProgramBench's admission step, and — inside the container, where the evolved commits actually exist — one-shot steps in `e1v2/entrypoint_pro.sh` around the seed stamp `/obo-repo/.git/ouroboros_seed` written in the unchanged `[ -e /obo-repo/.git ] ||` seeding branch (a refusal must never live inside the polled `ready_probe`, which reads any non-zero rc as 'not ready yet'). `write_json` is atomic (`ouroboros.utils.atomic_write_json`, lazily imported so the module stays stdlib-only for the container-side harbor agent, `trailing_newline=True` to keep every sidecar byte-identical), and `openrouter_key_remaining()` reads the authoritative `limit_remaining` with `limit - usage` only as a fallback. (v6.76.0) `model_slot_snapshot()` takes `env_overrides`: a server started in THIS process's environment lets the environment win over settings.json, but a server started in a CONTAINER is handed the settings FILE and a fresh environment, so the launcher's own env is not part of its configuration and must not be reported as if it were. That distinction is why the manifest must name the DERIVED settings, not the template: a live SWE-Pro smoke found `run_manifest.json` reporting `anthropic/claude-sonnet-4.5` while `_run_settings.json`, the container environment and the in-container settings all agreed the run was on `openai/gpt-5.5`, because `e1v2/run_pro.py` passed `--settings` while `derive_run_settings()` applies `pin_single_model(--solve-model)` on top of it. Both container-seeded launchers (`e1v2/run_pro.py`, `continual_learning/run_clb.py`) now re-snapshot `model_slots` from the derived `_run_settings.json` once it exists and record `harness.settings_template`/`harness.settings_derived` alongside it. SWE-Pro solve containers run with an OPEN network, exactly as before v6.75.0 (no network flags at all): the structural egress-isolation subsystem prototyped during this phase is NOT part of this release and is deferred to a later one, so there is no `--network-mode` flag and no relay machinery in the tree. What actually keeps the solver off the upstream fix is the adapter's tool policy (`--disable-tools`), and the official harness does not regulate the solve container's network at all (`--block_network` in the official evaluator applies to the EVAL container) — see `swe_bench_pro/METHODOLOGY.md` §0(b)-(c). Plus tri-state grading (`grade_pro.py`: `pass|fail|ungraded` + reason, `grade_summary.json`, unchanged headline formula plus an explicitly non-leaderboard-valid diagnostic percentage). ProgramBench's `result_index.jsonl` is append-only per row at BOTH the run root and the instance dir, skip rows included (readers dedup by `instance_id`, last row wins). Terminal-Bench uses `terminal_bench/run_tb.py` / `harbor_installed_agent.py` for installed full-Ouroboros runs and leaderboard-shaped k-trial submission trees; `run_tb.py` also writes a post-run `disclosure_ledger.json` (schema `tb_disclosure_ledger.v1`) recording the reward distribution, `AgentTimeoutError`/rate-limit/provider-failure histograms, per-task pass rate, concurrency, and the multiplier/gating flags actually used, so each run's leaderboard-validity is auditable. OSWorld uses `osworld/run_step_agent.py` for official env.step trajectories with native screenshot attachments, and `osworld/run_cu_bridge_agent.py` for the persistent-agent shape: one Ouroboros task per OSWorld task drives the VM through the unix_computer_use skill's osworld_http backend (same guest `/execute` channel; official reset/evaluate; declared-infeasible final answers become the official FAIL action; ax_tree off by default with `--allow-a11y`; live-server/live-data-dir guards; dataset variant pin + budget counters in the outcome — protocol deltas disclosed in `osworld/METHODOLOGY.md` §7). `terminal_bench/METHODOLOGY.md` (v6.79.0) is that adapter's disclosure SSOT, including the fact that harbor PERSISTS every `--ae`/`--ve` env value into its own job config/lock/result artifacts: the launcher redacts only the command artifacts it owns, so a submission copy must be swept by VALUE with `terminal_bench/scrub_submission_secrets.py --env-passthrough NAME=VALUE` (fail-closed — a value it cannot sweep safely refuses the whole scrub before touching a file). `run_step_agent.py` is also the shared home for the OSWorld launcher helpers the other two import: the live-server/checkout probes, the admit-then-amend run manifest (`admit_step_loop_run`/`amend_task_manifest`, so the clean-seed gate runs — and is persisted — before the VM boots), `construct_desktop_env` (retries the VM-booting `DesktopEnv` constructor and tears down every failed attempt instead of leaking the emulator), and the lane claim helpers over `platform_layer.acquire_exclusive_file_lock` (stale bound = task_timeout + **2 ×** startup_timeout + margin, two independent startup windows because the holder gets one for the `DesktopEnv` constructor and a fresh one for the reset-to-usable-screenshot loop; a one-window bound expires while a holder is still legitimately working, which is how two attempts end up on one task; `claim_stale_sec` is the single implementation and this text follows it). The scored claim is a FAIL-CLOSED durable transition, not an optimisation: `mark_task_scored()` fsyncs the permanent marker immediately after `env.evaluate()` and BEFORE the score is projected into any artefact, raises `ClaimMarkerNotDurable` instead of swallowing a write failure, and `release_task_claim()` never releases a scored claim whose marker is not confirmed on disk — so the only crash orderings reachable are 'marker, no result' and 'no marker, no result', never the 'result without marker' that made another lane rerun an already-scored task. On the CU bridge's real control flow that refusal is caught SEPARATELY from the broad adapter-error handler, which would otherwise fall through to the `finally` and release the lock with `scored=False`. The DURABLE part of the protection is a second marker, not the retained lock: a lock is reclaimable by design once `stale_sec` elapses, so a lock-only protection merely DELAYED the rerun of an already-scored task. `mark_task_scored()` therefore records the scored-but-unmarked state at `<key>.scored_unconfirmed` (fsync'd, one further path, not a further layer of best-effort) and `scored_claim_state()` — consulted before the lock and again under it — refuses that task with its own typed reason `scored_unconfirmed` REGARDLESS of staleness, so the state is permanent and visible to an operator instead of silently becoming claimable. The bridge retains the lock as interim cover, reports the official reward with the bookkeeping failure disclosed, and exits 2. If even that marker cannot be written the refusal carries `unconfirmed_marker=None`: nothing on disk records the score, so the bridge refuses LOUDLY with the distinct `claim_state_unrecoverable` outcome and exit 3 rather than promising a protection that expires. The claim transition is guarded against `BaseException`, not just `Exception` — a `KeyboardInterrupt` derives from the former and used to unwind through the `finally` and release the claim of an already-scored task (the same trap that made a P1 refusal handler inert); the claim is retained and the interrupt re-raised so it still stops the run. (v6.76.0) Retaining the lock was NOT sufficient there either, for the same reason it is not sufficient anywhere: that lock EXPIRES, so an interrupt landing after `env.evaluate()` but before either marker was durable left a task whose score WAS recorded reclaimable once `stale_sec` elapsed — a genuine double count. The interrupt path therefore fsyncs `<key>.scored_unconfirmed` through the shared `record_unconfirmed_score()` BEFORE re-raising (that helper never raises, so a second failure cannot replace the operator's interrupt with a disk error), which CLOSES the interrupt window; only `SIGKILL` remains open. `--claim-dir` itself is resolved through the pure `assert_outside_repo` (split out of `ensure_outside_repo` so a boundary check no longer has to mkdir what it validates) before anything is created, so lock/marker files cannot land in a checkout or live data — and (v6.76.0) the authority is the EXECUTION checkout the run attests (`--repo-dir`) as well as this module's own location, BOTH and not either: `confined_claims_dir()` used to derive its authority solely from `repo_root_from_devtools()`, so `--repo-dir /other/bench-clone --claim-dir /other/bench-clone/.claims` wrote lock and marker state into the very seed whose cleanliness the gate was about to attest. Invariant B of the launcher gate refuses that shape generally: a launcher whose provenance is attested against a checkout it was HANDED may not take its confinement authority from module scope. A launcher that attests a statically derived root and confines against that same root (the in-repo prediction writers) AGREES with its own record and is not flagged. Invariant B also reports the same authority mistake in its REFUSAL shape — `if <path> == <a __file__-derived module root>: raise` — which a call-shaped detector cannot see: `run_clb.refuse_live_repo_clone` compared `--ouroboros-clone` against its own `REPO`, so handing a PINNED SEED's launcher that same seed (the recipe `continual_learning/METHODOLOGY.md` prescribes) was refused while the live repo the guard exists to protect went unmentioned; the two trees coincide only in the development workspace. The authority is now `run_roots.live_repo_roots()` — `$OUROBOROS_REPO_DIR` plus the `repo` sibling of each live data root, the same runtime-layout SSOT `live_data_roots()` uses. **One window stays open and is not described as closed:** a `SIGKILL` between `env.evaluate()` returning and `mark_task_scored()` completing runs no handler by definition. It is deliberately NOT covered by an intent marker written before `env.evaluate()`, because that would block staleness reclaim for the whole UNBOUNDED evaluation and leave every hard-killed-but-never-scored task needing manual clearing — a broad harmful window traded for a narrow benign one. Benign because the marker precedes `result.txt`, the outcome and the ledger row, so a kill there records the score NOWHERE: a later retry is correct rather than a rerun of a counted score, and the cost is one lost evaluation. After a hard kill, compare the claim dir with the results tree (`osworld/METHODOLOGY.md` §7.9 names the check). `acquire_task_claim()` reads the marker TWICE and the second read is the load-bearing one: checking it only before waiting for the lock is a live TOCTOU hole (two lanes see no marker, the first wins, scores and releases, the second then acquires the lock with the marker present and would still be told `claimed`), so it re-reads under the held lock and gives the lock back as `already_scored`. `task_already_scored()` is the read-only form, asked BEFORE admission so a lane arriving at an already-scored task leaves no footprint in the winner's shared per-task run directory. (v6.76.0) The claim alone was not enough while both attempts still wrote the same FILES: the per-task run directory is keyed by the task, so two overlapping lanes both wrote their admission manifest to `<run_dir>/task_run_manifest.json` before either had claimed anything, and the loser then finalized `skipped_in_flight` on top of the holder's still-running record. Every ADMITTED attempt now records into `<run_dir>/attempts/<attempt_id>/` (its own admission manifest, plus its outcome if it produces one) and only the attempt HOLDING the claim writes the canonical per-task artefacts (`task.json`, `result.txt`, `task_outcome.json`, `task_run_manifest.json`); the append-only `result_index.jsonl` is NOT a per-attempt log — an attempt enters it only when it produces an OUTCOME, so an attempt that steps aside on a held or already-scored claim (exit 4) writes no row at all, while one blocked before the claim (seed gate, runtime attestation) writes a row carrying `claim_owner: false`. Rows name their `attempt_dir` and `claim_owner` so a reader deduping by `instance_id` can tell the holder's row from a bystander's, and an auditor reconstructing a run reads the `attempts/` subtree for everything that was TRIED and `result_index.jsonl` for everything that produced an outcome and therefore counts in the denominator. Attempt ids come from the existing `run_roots.timestamp_run_id`, whose pid+counter suffix already distinguishes two attempts started in the same second. **MULTIPLE OSWORLD LANES ARE SUPPORTED; the lane-script GENERATOR is not in this release.** Overlapping runs are a supported configuration and the v6.76.0 smoke exercises them — several operator-written lane scripts, each invoking `run_cu_bridge_agent.py --claim-dir <shared>` against its own isolated bench server over a shared results tree. What was built during this phase and EXTRACTED before release is the CONVENIENCE GENERATOR for those scripts (`osworld/gen_lanes.py`, lane port binding, `lanes.json`), deferred to a later one: nothing in the tree generates lane scripts, allocates lane ports, writes a `lanes.json` or starts more than one bench server — the operator does that. It was extracted under the standing rule for a subsystem the release does not use (the smoke's lanes are hand-written, not generated) while it accumulated 6 of phase P2's 13 commit-gate findings, 3 of them still open (pre-admission work relocated into generated BASH where the `ast` guard cannot see it, a `server.py` started with the caller's inherited environment instead of the `IsolatedServer._env` sanitisation, and a lane-settings write depending on a mkdir side effect). The `--claim-dir` mechanism it drove is NOT lane-specific and stays, because append-only resumes and retry passes over a shared results tree need the same ownership answer; the pre-registered dedup rule for overlapping append-only runs is first-scored-attempt-wins (`osworld/METHODOLOGY.md` §7.9). What survives from that work is the general boundary split it needed: `assert_outside_repo` (pure resolve+refuse) is now separate from `ensure_outside_repo` (assert + mkdir), and every migrated launcher uses the PURE form before admission — creating an output directory to validate it left a filesystem footprint before the run manifest existed, which is the very invariant these seams establish; the atomic manifest write creates the tree, and the pre-admission gate catches an imported mutator by RESOLVING it across the module boundary rather than by naming it (see `launcher_audit.py` above), so the next one nobody enumerates is caught the same way. `osworld/operator_patches/` holds unified diffs for the third-party OSWorld checkout (never a fork, never tasks/evaluators/scoring) — currently the docker provider `LOCK_TIMEOUT` 10s→60s so concurrent lanes do not die on the global port-allocation lockfile. `continual_learning/operator_patches/` plays the same role for the external CL-Bench clone, including the whole-`instance_outcomes` bridge fix (co-resolved instances are recorded with `ouroboros_status="auto_resolved_no_agent_turn"`) and the docker-path runtime attestation the host path gets from `IsolatedServer._wait_ready()`. SWE-bench Pro frozen prepared-repo predictions use `pro_predictions.py`; evolutionary E1v2 runs live under `swe_bench_pro/e1v2/` and carry `obo-data` + `obo-repo` volumes across tasks. E1v2 settings are profile-driven through the shared `devtools/benchmarks/common/model_slots.py` single-model pinning helper plus a `swe_bench_pro/e1v2/profiles/*.json` profile; the adapter is crash-resilient (run_pro writes the timeline/predictions row BEFORE the post-solve teardown, times its docker cache-load/inspect ops, and RESUME-skips a task whose `patch.diff` already exists; auto_run kills a wall-timeout-exceeding run_pro process group plus its named `obopro-*` containers and continues), provides a musl/Alpine install-in-image transport fallback when no `oboros-env-musl` volume exists, and strips gold git-history from each task image before the agent starts (`swe_bench_pro/strip_gold_history.sh`, warn-only) to neutralize SWE-bench Pro issue #93. v6.44.0 makes fixed-model baseline the default Pro measurement mode (`--evolution` opts into native post-task evolution), sets `OUROBOROS_TASK_REVIEW_MODE=required` inside the Pro settings template only, passes `disabled_tools` through `ouroboros run --disable-tools`, and routes E1v2 patch capture through the shared `swe_bench_pro/capture_patch.sh` helper (lockfile-without-manifest churn is filtered there; pure lockfile patches are preserved). Post-task evolution can now receive GLOBAL improvement-backlog/promotion signals from project-scoped workspace tasks while project facts still stay isolated in the per-project store; this removes the earlier `no_promotion` limitation without weakening the project-fact leak guard. Between instances drivers reset only the PER-TASK cost tracking inside isolated benchmark roots that carry the explicit `.ouroboros_isolated_benchmark` sentinel; the CUMULATIVE shard ledger is deliberately NOT reset — `run_pro.py`'s budget rail compares that cumulative spend against `--total-budget`, so the shard total must be sized as `per_task_cost × scheduled tasks` (v6.74.0: `e1v2/auto_run.py` derives it by default and fails loudly when `total <= per-task cap` with more than one scheduled task, and `run_pro.py` reads the cumulative ledger spend on the FIRST task of an invocation too — the old `i > 1` fast-path made per-task auto_run invocations seed every container with `TOTAL_BUDGET = per_task_cost` regardless of the derived shard total; with parallel workers the ceiling is a bounded in-flight overshoot, not a strict per-task guarantee). Live data roots are never budget-reset. (v6.51.0) `swe_bench_pro/e1v2/orchestrate_probe.py` is the parallel fixed-version probe orchestrator: it fans `run_pro.py` across N workers with isolated `obo-repo-w{N}`/`obo-data-w{N}` volume suffixes + per-task reset, inline-grades each task and `docker rmi`s its image (disk-bounded), and writes a per-run `manifest.json`; like run_pro it routes `--out-dir` through `ensure_outside_repo` so nothing lands under `repo/`, and it REQUIRES the explicit `OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1` audited opt-in before forwarding the provider key into untrusted task containers (it never silently defaults it on). `run_pro.py` can populate a host image cache (`docker save | zstd`) under the configurable `OBO_SWEPRO_IMG_CACHE` dir (opt-in, atomic, fail-soft) so re-runs load images locally instead of re-pulling. (v6.55.0) All committed bench settings templates share disclosed scaffold defaults — `OUROBOROS_MAX_WORKERS=4`, `OUROBOROS_SAFETY_MODE=light`, `RUNTIME_MODE=pro` for container benches (GAIA deliberately stays `light`), `claude_code_edit` disabled (single-model harness measurement) — documented in `devtools/benchmarks/README.md`; Terminal-Bench raises the in-container finalization margin `_DEADLINE_SAFETY_SEC` 30→105 from measured overhead; `programbench/` gains a full e2e runner (gateway-driven cleanroom solve → submission export → official eval, with `task_contract.budget_profile` pacing, solve-model id normalization, resume-friendly per-instance checkpoints, result-payload status detection); v6.74.4: ProgramBench submission export (`create_submission_tarball`) reads the CURRENT live tree — not git, not a fresh checkout — excluding `.git`, the root `executable`/`reference_executable` binaries, `.ouroboros/` and named build/cache noise at any depth, and the instruction template now states that contract verbatim (uncommitted edits DO ship; run `./compile.sh` one final time), replacing the false fresh-checkout framing; `continual_learning/` wraps the external clbench runner (strictly sequential task stream); `osworld/` aligns to the official OSWorld 2.0 protocol (pinned upstream, 500-step default, submission-shaped results, env preflight, bridge-level `final_answer` population).
devtools/benchmarks/gaia/   ← v6.45.0 GAIA adapter (v6.79.0: alongside `GAIA_FORMAT_INSTRUCTION` and `GAIA_ANTI_LEAK_INSTRUCTION`, `inspect_solver/__init__.py` holds `GAIA_EPISTEMIC_INSTRUCTION` — an adapter-only DISCLOSURE rule, appended identically by all four solvers and stripped from traces before the leakage scan, that requires saying when a claim is unverified and explicitly does NOT ask for lookups of things the model already knows; owner Q20/Q22 kept it out of `prompts/SYSTEM.md` and out of the typed task contract, with no finalization gate, and `gaia/METHODOLOGY.md` discloses it): uses the official `inspect_evals/gaia` task/scorer, invokes Ouroboros through `ouroboros run --result-json-out` so answer extraction reads structured `final_answer`, writes run manifests under `bench_runs/gaia/`, and reports any local lenient-normalized score as diagnostic only. `settings_base.json` is the committed base template; `run_gaia.py` renders a per-run settings file that pins runtime/review/vision model slots, uses `OUROBOROS_TASK_REVIEW_MODE=required`, empty memory, and post-task evolution off. v6.53.0 adds explicit GAIA scaffold profiles: `web_off_baseline`, `strict_ddgs` (first-party `web_search` enabled with pure-retrieval `ddgs`), and `quality_openrouter_web` (main-model OpenRouter server-web, fail-fast if unsupported), plus a disclosed `--max-workers` worker-pool knob (v6.55.0 default 4 — same-model subagent decomposition slots, recorded per run as `worker_scaffold_disclosure`, never parallel sample best-of-N; pass 1 explicitly for the strict-baseline ablation). Attachment handling resolves real Inspect file paths first, then (v6.74.0) stages official `Sample.files` that Inspect placed INTO the sandbox via `sandbox().read_file` into the run-local attachment dir; the `GAIA_SHARED_FILES_ROOT` fallback is an EXACT relative lookup (the broad name-anywhere `rglob` fallback was removed — it could stage an unrelated same-named file), every staged file records a provenance row (`provenance.json`: host_path / sandbox_read), and a declared-but-unresolvable attachment raises the typed `GaiaAttachmentStagingError` (a harness infra error, never a silent no-attachment solve). Resolved files pass through `--attach`, stale `/shared_files` prompt text is rewritten toward the `[ATTACHMENTS]` manifest, and `user_files` stays jailed under the run root.
skills/telegram/            ← Bundled owner-only Telegram text/photo bridge plus optional Mini App gateway; seeded disabled until the bot-token and host-permission grants are approved, with bridge/Mini App readiness reported by its own bounded status route
skills/unix_computer_use/   ← Bundled extension skill payload for supervised desktop observation/input (screenshot with coordinate normalization, window_list, click/drag/type/key/move/scroll, mouse_down/up, hold_key, cursor_position, wait, best-effort AX set-of-marks). v6.63.0 adds explicitly configured REMOTE backends behind a connection registry persisted in skill state (`data/state/skills/unix_computer_use/{connections.json,active_connection.txt}`, atomic writes): `osworld_http` (OSWorld VM's in-guest server — `GET /screenshot`, pyautogui via `POST /execute`; success requires guest returncode 0, non-ASCII typing pastes via the in-VM clipboard, scroll is 1:1 wheel detents, screenshots size-capped + PNG-validated) and `ssh_macos` (screencapture/scp + cliclick over the owner's existing ssh config; no key material stored), plus `remote_exec` (shell on the active REMOTE only; refuses on local). A disabled or registry-missing active connection fails CLOSED — never silently falls back to the local desktop. The manifest declares `net` (needs no grant, but removes the skill from the native auto-enable class — the owner or a bench runner enables explicitly; zero-grant tool/subprocess skills still auto-enable under OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS); it reports missing platform backends instead of guessing. Windows support is a future separate skill (P7).
packaging/cli/                ← Packaged CLI shell/cmd wrappers and user-local installer launchers copied into desktop artifacts
Dockerfile                    ← Docker image (web UI runtime)
```

### Gateway Boundary v1

`ouroboros/gateway/` is the single browser-facing boundary between the
vanilla-JS frontend and the Python runtime. `server.py` owns process startup,
lifespan, supervisor hosting, and static-file mounting; `gateway/router.py`
owns every `/api/*` route and `/ws`; domain modules under `gateway/` own the
actual HTTP handlers. This keeps frontend work pointed at one explicit contract
surface instead of requiring contributors to understand supervisor, worker,
marketplace, extension, MCP, local-model, and settings internals at once.

The frozen contract is `ouroboros/gateway/contracts.py`. It carries the HTTP
endpoint index, WebSocket message discriminators, and TypedDict envelope shapes.
`runtime_mode='advanced'` may refactor gateway handlers and router plumbing, but
editing `gateway/contracts.py` is protected as a frozen contract and requires
`runtime_mode='pro'` plus the normal triad + scope review gate. The legacy
`ouroboros/contracts/api_v1.py` module remains as a compatibility import only.

Frontend modules call backend routes through `web/modules/api_client.js`, with
JSDoc mirrors in `web/modules/api_types.js`. `web/package.json` defines the UI
subpackage boundary without adding npm dependencies, TypeScript, codegen, or a
build step. `tests/test_gateway_parity.py` checks that the contract endpoint
index stays aligned with `gateway/router.py` and that the JSDoc mirror stays
present for the core browser-facing envelopes.

### CLI / Headless Boundary

`ouroboros.cli` is the second first-class interface to the same runtime. It is a
thin HTTP/SSE client over the gateway, not a benchmark-only harness and not a
parallel scheduler. `POST /api/tasks` creates managed queue tasks, `GET
/api/tasks/<id>` reads durable results, `GET /api/tasks/<id>/events`
replays task-scoped events from the existing logs before following live SSE
updates, and `GET /api/tasks/<id>/artifacts/<name>` serves declared task
artifacts from the task artifact directory only. For task streaming commands
such as `run` and `tasks watch`, stdout is
reserved for final machine-consumable output (or JSONL when requested) while
progress goes to stderr; status and admin wrappers may print human summaries.
`ouroboros schedule list|add|remove` is the CLI wrapper over `/api/schedules`;
it manages persisted 5-field cron schedules that enqueue ordinary tasks through
the supervisor queue rather than running a separate scheduler daemon.

Skill-manifest `scheduled_tasks` are mirrored into the same table by
`supervisor/queue.py::sync_skill_schedules`, whose enable gate is the
`skill_readiness_for_execution()` SSOT (review/grants/deps/enablement) plus a
`supervised_task` permission check. `resync_skill_schedules()` runs on every
skill lifecycle change (toggle, grants, reconcile, delete, review, and
marketplace install→review/uninstall) and on a 60 s scheduler tick; schedules
whose source skill or `scheduled_task` no longer exists are removed, not left as
disabled tombstones. A blank schedule timezone resolves the DST-aware system
local zone (`TZ`/`/etc/localtime`), falling back to a fixed current offset only
when no IANA name is found — set an explicit IANA timezone for DST-critical
schedules. A compact active-schedule digest (capped, with an omission note) is
injected into both normal task context and background consciousness context.

Packaged desktop artifacts ship a tiny `bin/ouroboros` wrapper and installer
instead of a second PyInstaller runtime. The wrapper runs the bundled
`python-standalone`, bootstraps the launcher-managed repo from the embedded
`repo.bundle` when needed, and then delegates to this same `ouroboros.cli`
module. In packaged mode, `run --start` launches the desktop app/launcher and
waits for `/api/health` plus `api_state.supervisor_ready`; it must not start
`server.py` directly through `sys.executable -m`, because that bypasses the
launcher-owned bootstrap, process record, and managed repo lifecycle.

Packaged artifacts also bundle an official, notarized **Node.js LTS** runtime
under `node-standalone/` (pruned to just `bin/node[.exe]`). The build scripts
fetch it via `scripts/download_node_standalone.sh`/`.ps1` (SHASUMS-verified)
before PyInstaller, and the macOS signing pass re-signs it under the hardened
runtime so it is not code-signing-killed (SIGKILL) when launched from the
packaged app. `platform_layer.resolve_bundled_node()` prefers this bundled node
over a PATH (e.g. Homebrew) node for `node`-runtime skills and the `node --check`
preflight; in dev builds without the bundle it falls back to PATH node.

Packaged artifacts also bundle **ripgrep** under `ripgrep-standalone/` (pruned
to `bin/rg` or `rg.exe`). The build scripts fetch it via
`scripts/download_ripgrep_standalone.sh`/`.ps1` before PyInstaller;
`search_code` resolves it through `platform_layer.resolve_bundled_ripgrep()`
before falling back to PATH `rg` and then the Python scanner. Unlike raw shell
`rg`, the first-class tool enumerates allowed files first and keeps the existing
protected/secret/subagent filters.

External workspace tasks keep `Env.repo_dir` pinned to the Ouroboros repo for
prompts, BIBLE, architecture/development docs, skills, and review policy.
`ToolContext` carries an optional `workspace_root`; contextual repo tools resolve
through `active_repo_dir()` when workspace mode is set. Workspace roots must be
separate git worktree roots and must not overlap the Ouroboros system repo or
data drive. Workspace mode uses an explicit allowlist for contextual repo/data,
search, shell, git status/diff, browser, log/history, planning, and parent-owned
delegation tools. Workspace children run as local-readonly subagents: local
writes, commits, review mutation, runtime control, tool expansion, shell, and
skill lifecycle stay blocked — except bounded task-tree coordination via
`tree_note`/`tree_read` and parent-only `override_delegation_constraint` (the
permitted local-write coordination paths: swarm beacons, shared-frame reads, and
reasoned override decisions; coordination, not state mutation). Nested readonly delegation is allowed only within
configured depth/cap limits, and descendants deeper than the configured capability
depth (`OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT`) are coerced to the light model
lane. Enabled/reviewed extension and
MCP tools remain callable by owner policy, subject to `task_contract`
resource constraints such as `web=false` or `network=false`. The target workspace
may be left dirty or may contain task-local git commits/branches/tags/pushes when
the task itself requires them; Ouroboros still blocks git operations that target
the Ouroboros repo/data roots. Workspace patch artifacts are captured against the
preflight git base, while acting self-worktree subagents remain strict patch-only
and still fail if their HEAD moves.
The CLI downloads patch artifacts through the task artifact endpoint, waits for
artifact finalization in `--patch` / `--patch-out` mode, and fails nonzero when
the patch is missing, empty, or failed. `--no-stream` suppresses live progress
but still waits; `--detach` is the explicit create-and-return mode.
Benchmark devtools under `devtools/benchmarks/` require clean per-instance local
checkouts or official benchmark containers; they do not commit target
repositories. Broad scope/plan/deep-review packs list unrelated `devtools/`
files in the Atlas manifest without inlining every benchmark harness, while
touched `devtools/` files are fully included in triad/scope review. This is a
context-management rule, not an immune-system escape hatch. SWE-bench Pro
install-in-image transport for Alpine/musl images is fail-fast and diagnostic:
it uses the task image's system Python without intentionally upgrading the
interpreter, checks pyexpat/pip/server imports before the solve, drops
unsupported Playwright wheels when browser tools are disabled, and records a
typed `infra_reason` so permanent non-runs are not retried like transient
provider failures. Patch capture records a base-untracked snapshot before the
solve and unstages those pre-existing files before emitting `model_patch`, so
task-image fixtures do not leak into official patches while new agent-created
files remain included.
Their generated audit sidecars are operator artifacts, not benchmark scoring
replacements: run-manifest files record requested task IDs/counts, exact
commands, model-slot settings, source provenance, output paths, and isolated
data roots; result-ledger JSONL files are denominator-preserving ledgers that
represent every requested task, including failures, timeouts, blocked
preflights, and empty patches. Adapter defaults may use fixed names
(`run_manifest.json`, `result_index.jsonl`) or prediction/preflight-derived
suffixes (`<predictions>.run_manifest.json`, `<predictions>.ledger.jsonl`,
`osworld_preflight.*`). Official benchmark predictions and scorers remain
benchmark-owned source of truth.

Workspace mode is a tool-routing and blast-radius guard, not an OS sandbox.
Like OpenClaw's host workspace mode, absolute host paths are not a hard security
boundary unless a Docker/SSH/remote backend is added around tool execution.
When task metadata contains a host-owned `executor_ref`, `run_command`,
`run_script`, and service tools route process execution through the declared
backend (`local` or `docker_exec`) only when the requested cwd is covered by an
executor path mapping. Unmapped task-drive, artifact-store, and user-files cwd
paths remain local host execution roots. File tools continue to operate on the
shared host workspace. `executor_ref.network=none` is enforced by the backend
transport for mapped backend executions, for example by requiring Docker
`NetworkMode=none`; LLM provider traffic remains outside the benchmark tool
environment. Executor-backed
foreground commands and services are also written to durable
`data/state/workspace_executor_processes/` records so server-side panic and
emergency cleanup can stop local process groups or Docker-side pidfile/service
processes even if the worker that started them has died. Do not grow ad-hoc
shell parsing to approximate that sandbox.
Project-local dependency installs are ordinary workspace work. In
`runtime_mode=pro`, system/global dependency installs may be attempted through
`run_command` and the safety supervisor when needed by the external workspace;
sudo must be noninteractive (`sudo -n`) and password-prompting sudo is blocked.

Headless memory isolation is implemented as a per-task child drive under
`data/state/headless_tasks/<task_id>/data`. `forked` mode copies stable memory
seed files (`identity.md`, `WORLD.md`, `registry.md`, and `knowledge/`) without
dialogue/task history; `empty` mode starts from a fresh child drive; live
`shared` mode is disabled for subagents and external workspace tasks until a
sanitized shared-context v2 exists. Ordinary local root tasks may still use the
parent drive directly when no external workspace isolation is requested. External
runs produce explicit artifacts under `data/task_results/artifacts/<task_id>/`:
`workspace_preflight.json`, `workspace_patch.json`, and `memory_export.json`;
patch finalization with changes also produces `workspace.patch`, while failed
patch finalization records `artifact_status=failed` and the manifest only.
`workspace_patch.json` records patch state, base git reference metadata
(`base_ref`, `base_head`, `base_is_empty_tree`, `current_head`), size, sha256,
diffstat, included/excluded untracked paths, git diagnostics, and artifact
errors. For acting-subagent tasks, `task_constraint.base_sha` is the authority
envelope: patch capture uses that commit as `base_ref`/`base_head` and fails
closed if final `HEAD` no longer matches it, so a child cannot hide commits or
return a patch against a shifted baseline. The parent result carries `artifact_status`
(`pending`/`finalizing`/`ready_with_changes`/`ready_no_changes`/`missing`/`failed`)
so headless clients cannot observe a terminal workspace result before artifacts
are ready, honestly no-op, missing, or explicitly failed.
Headless runs never auto-merge memory back into the parent drive. Queued
non-workspace tasks may also request `memory_mode=forked|empty`; in that case
the same child-drive mechanism is used for memory isolation while the active repo
remains the Ouroboros repo. Swarm
readiness in v1 is implemented as live child tasks over the existing queue:
`schedule_subagent` emits a normal `schedule_subagent` event, the supervisor enqueues it
as a child task, and an existing worker executes it. There is no separate
scheduler, dashboard, endpoint, or settings surface. Child lineage is inferred
from the active `ToolContext` and persisted as `parent_task_id`, `root_task_id`,
`session_id`, `actor_id`, `delegation_role`, `role`, `memory_mode`,
`drive_root`, `child_drive_root`, `budget_drive_root`, `task_contract`,
`task_metadata`, `task_constraint`, `requested_model_lane`,
`effective_model_lane`, `model`, `use_local_model`, `task_group_id`, and
`subagent_envelope`. For workspace/forked children,
`budget_drive_root` is also the canonical status/result root, so parent tools
read the same child lifecycle records that the supervisor writes.
Installed skill payloads exist only on the canonical data root, so the
`skill_payload` resource root resolves through `canonical_data_root()`
(`tool_access.py`: task_metadata `budget_drive_root` → ctx `budget_drive_root`
→ `drive_root`) rather than the child drive — a read-only scout on an isolated
child drive reads the real payload it was asked to audit, while the verb
matrix keeps skill-payload writes parent-only (v6.74.5).
`task_status.py` is the effective-status SSOT for gateway and tool reads: a
child terminal result overrides a stale parent `requested`/`scheduled`/`running`
result, while authoritative parent terminal failures/cancellations stay
authoritative. Workspace artifact tasks stay nonterminal while
`artifact_status` is `pending`/`finalizing`; only
`ready_with_changes`/`ready_no_changes`/`missing`/`failed` artifact states make
the effective workspace result terminal. `wait_task` performs a
bounded wait (default 180s) and returns the full untruncated child handoff.
`wait_tasks` performs batch waits (default 600s) and returns a compact
STRUCTURAL projection per child — task_id, status, cost_usd (on every status),
child_result_sha256 (the join-ledger hash), outcome_axes, result,
trace_summary, and duplicate_of when applicable — instead of the full
persisted envelope; forensics (trace_refs, loop_outcome, verification_ledger)
stay on disk in `task_results/<id>.json`, addressable by
`child_result_sha256` (a disclosed omission, not silent truncation);
`get_task_result` returns the full result text plus trace/outcome summaries. The wait envelope itself (all_terminal / timed_out /
elapsed_sec / live_child_status / early_return) is unchanged.

A burst of `schedule_subagent` calls emitted in ONE tool-call round runs in the
existing tool ThreadPool instead of sequentially (`schedule_subagent` is in
`tool_capabilities.PARALLEL_SAFE_ENQUEUE_TOOLS`); a process-local lock in
`tools/control.py` serializes the parent-side scheduling state so concurrent
emission cannot lose records, while the supervisor still drains its event queue
serially, keeping cap/dedup/enqueue single-threaded. Each spawn wave also writes
one durable `swarm_fanout` telemetry event to `events.jsonl` (requested count,
task group, role, requested/effective lanes, depth, inter-wave latency) for
fan-out observability; it carries no `delegation_role`/`subagent_task_id`, so the
Logs view renders it as a summary line, not a phantom child card. The supervisor
tags accepted subagent scheduling with `accepted`, `active_subagent_count`, and
`max_active_subagents`, and rejections with `accepted=false`; these markers are
declared on the `ChatOutbound` gateway contract and survive `/api/chat/history`
replay via `gateway.history._PROGRESS_META_FIELDS`.

Workspace tasks expose knowledge access (`knowledge_read`, `knowledge_list`, and —
since v6.23.3 — `knowledge_write`) because `workspace_task` permits runtime-data
reads and a workspace task is project-scoped, so `knowledge_write` is redirected to
that project's per-project facts store (`projects/<id>/knowledge`), never the global
`memory/knowledge`. Other mutating cognitive tools (`update_scratchpad`,
`update_identity`) stay out of the workspace allowlist, and acting subagents remain
blocked from all cognitive-memory writes by their authority envelope. Parent global
memory changes come from the post-task experience review/import path, not directly from the
workspace child.

Live subagents default to deterministic
`task_constraint.mode="local_readonly_subagent"`. The registry filters their
visible first-party tool schemas to repo/data/history reads plus web/browser
inspection and also blocks forbidden first-party calls at execute time,
including local writes, commits, review mutation, runtime control, tool
expansion, skills lifecycle, and shell — except bounded task-tree coordination
via `tree_note`/`tree_read`, parent-only `override_delegation_constraint`, and
bounded media projection such as `extract_video_frames` writing derived frames
only under `artifact_store/video_frames` through a host-owned command shape (the
permitted local coordination/projection paths; not arbitrary workspace/repo
mutation). Nested readonly `schedule_subagent`
recursion is visible only within configured depth/cap limits, and depth beyond the
configured capability depth (`OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT`, default 1)
is coerced to the light lane (an explicit capped main/heavy request surfaces a note).

v6.50.0 adds a reconciliation layer around this contract. `schedule_subagent`
may carry a closed-enum `required_capabilities` list (for example `shell` or
`vcs`); the parent-side tool path gives immediate feedback, while the supervisor
admission path is authoritative and rejects a child whose selected profile cannot
satisfy the declared needs. Non-advisory constraints discovered by scouts are
written as structured `delegation_constraint` rows on the task-tree ledger and
folded into the same admission reducer (`effective_delegation_budget`) unless
explicitly overridden with a reason. Scheduler back-pressure rows such as
`queued_behind_active_cap` are advisory telemetry: they explain why a child is
waiting but do not block later children from being queued below the hard
per-root ceiling.
When the active-subagent cap is full but the tree remains below the hard
per-root ceiling, admission leaves the child `PENDING`/`STATUS_SCHEDULED` with
`queued_behind_active_cap` metadata instead of failing it; `assign_tasks` then
serializes actual starts by checking the current RUNNING child count.
Delegating parents also receive a bounded absorption reminder before a clean
no-tool final answer while direct children are still running; ignoring the
reminder finalizes as honest `best_effort` (`children_unabsorbed`) rather than
silently orphaning paid child work.

Subagents may also be **mutative ("acting")** when the parent passes
`write_surface` to `schedule_subagent` and the master toggle
`OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS` allows it (default ON in advanced/pro, OFF
in light; owner-controlled). Acting children carry
`task_constraint.mode="acting_subagent"` with a machine-enforced authority
envelope (`surface`, `write_root`, `base_sha`, `protected_paths_grant`,
`external_tool_grants`, `parent_only_commit`, `return_kind`). They may write, run
shell, and run services inside ONE isolated write surface — `self_worktree` (a
`git worktree` of THIS repo checked out from the parent's base commit, under
`OUROBOROS_SUBAGENT_WORKTREE_ROOT`, outside `repo/` and `data/`),
`external_workspace` (an existing external project directory), or `genesis` (a
from-scratch project the supervisor provisions as a fresh empty git repo under the
durable `OUROBOROS_SUBAGENT_PROJECTS_ROOT`, outside `repo/` and `data/`) — but
still CANNOT commit the live body, run review / runtime /
skills lifecycle, enable tools, or write cognitive memory. `active_tool_profile`
resolves them to the `acting_subagent` profile only when the surface is valid and
fails closed to read-only otherwise; a delegated subagent never inherits
`self_modification` / `operator_control`. For `self_worktree` the registry keeps
protected-path write discipline and protected shell-write guards active (it is a
checkout of the system repo), allowing protected edits only in pro AND with
`protected_paths_grant`; extension/MCP tools are denied unless named in
`external_tool_grants`. Children produce a `workspace.patch`; the parent
integrates a chosen patch with `integrate_subagent_patch` (manifest-first,
sha256-verified, 3-way apply, advisory invalidation, `subagent_patch_verdict`
artifact) into `ctx.active_repo_dir()` and remains the **sole committer** of the
live body (enabling best-of-N: accept one, synthesize several, or reject).
Routing is top-only: a nested acting parent integrates a descendant's patch into
its own worktree, so patches bubble up one level at a time. `genesis` is the
exception to integration: the project directory itself is the deliverable (a new
game/site/app/Ouroboros), so it is durable, never GC-pruned, kept out of the
worktree registry, and never integrated into the live body. The supervisor
(`_resolve_subagent_constraint`) is the authoritative gate that validates the
toggle/surface and provisions `self_worktree`/`genesis`; startup
`subagent_worktrees.prune_orphans` reconciles leftover worktrees from a durable
registry at `data/state/subagent_worktrees.json`.
Enabled/reviewed extension tools and enabled MCP tools remain callable by owner
policy unless the inherited `task_contract.allowed_resources` forbids network
or web access; local-readonly means readonly against local Ouroboros/workspace
state, not a ban on owner-approved external capabilities. Generic
`read_file(root=runtime_data)` / `list_files(root=runtime_data)` behavior is
unchanged for normal tasks, but subagents additionally deny known
secret/control files such as `settings.json`, token/credential/key files, and
secret-like owner-state paths. Browser tools remain available for remote-page
inspection, but subagents fail closed instead of auto-installing browser
dependencies. Subagents MAY browse/act on external HTTP(S), on loopback
(localhost/127.0.0.1) EXCEPT the Ouroboros control-plane ports (agent API
8765, local-model 8766, host-service 8767, and any isolated-run server set via
`OUROBOROS_SERVER_PORT`/`OUROBOROS_HOST_SERVICE_PORT`), and on `file://` paths
scoped to the task's explicit workspace root — so they can visually verify their
own built apps (`browse_page`/`browser_action` screenshot + `analyze_screenshot`/
`vlm_query`). They still cannot browse private, link-local, reserved, unresolved,
or numeric-obfuscated hosts, cannot run `evaluate` JS, and `file://` outside the
workspace (e.g. the data root / `settings.json`) stays denied. The guard checks
literal IPs and DNS results before navigation, after redirects, and in route
handlers, so hostnames resolving to blocked addresses are denied. This is a URL/DNS-layer guard, not a
connect-time proxy; hostile DNS rebinding would need a future resolver-pinning
or proxy design if stronger network isolation is required. Subagents also
cannot run arbitrary browser JavaScript.

`memory_mode=forked` is the default and uses the same child-drive mechanism as
headless workspaces: copy stable memory seed files only (`identity.md`,
`WORLD.md`, `registry.md`, `knowledge/`) into
`data/state/headless_tasks/<task_id>/data`, without dialogue history, scratchpad
blocks, task history, or auto-merge. `empty` creates a blank child drive.
`shared` is rejected for live local subagents and external workspace tasks; a
future sanitized shared mode must be designed separately. On completion, only
the child task result is copied back to the parent drive; identity, scratchpad,
registry, knowledge, dialogue blocks, and `memory_export` are never merged or exported
automatically. The supervisor keeps a configurable structural depth cap
(`OUROBOROS_MAX_SUBAGENT_DEPTH`, default 2, hard max 10, **min 0**) and a configurable
active-child cap per root (`OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT`, default 6,
hard max 500, one shared constant `config.MAX_ACTIVE_SUBAGENTS_HARD_CAP` consumed by the
config clamp, both supervisor gates, and the `wait_tasks` id cap; v6.82.0 raised it from
50 with NO perf work — the O(n²) tree scans at 500 are an accepted owner trade-off). A configured depth of **0 disables delegation entirely** — both the tool-side
gate (`tools/control.py`) and the supervisor gate (`supervisor/events.py`) refuse every
child, while root tasks (depth 0 themselves) still run, and `plan_task`'s planning scouts,
which go through the same gate, are refused too so plan review completes on its
`degraded_evidence` path. Until v6.79.0 a configured 0 was silently rewritten to the
default 2 (`config._bounded_positive_int_setting` treated it as "unset"), so every
"no-swarm" run actually delegated two levels deep; `min_value=0` now applies to this
setting only, and every other bounded-int setting keeps its minimum of 1. Workspace
parents may schedule readonly or acting children; the child inherits
`workspace_root`, `workspace_mode`, task contract, deadline/resource metadata,
and lineage while the parent remains the only committer of the live body (acting
children return a `workspace.patch` for parent-side `integrate_subagent_patch`). External
`/api/tasks` and CLI `run` requests may not forge
`delegation_role=subagent` or parent/root lineage; only the internal
`schedule_subagent` event path can create live subagents. Startup performs a
best-effort prune of terminal copied
back child drives under `state/headless_tasks/` after the retention window
(default 7 days, env/settings override), and skips nonterminal or artifact
finalization states.

### Two-process model

1. **launcher.py** — immutable outer shell (tracked in the git repo; bundled as the packaged entry point via PyInstaller). Never self-modifies. Handles:
   - PID lock (single instance)
   - Bootstrap: initializes `~/Ouroboros/repo/` from the embedded `repo.bundle` +
     `repo_bundle_manifest.json` on the first launcher-managed run
   - Managed repo hand-off: after first bootstrap, keeps using the launcher-managed
     git checkout and normal managed-remote branch updates instead of per-launch
     file overwrites
   - Starts `server.py` as a subprocess via embedded Python
   - Shows PyWebView window pointed at the actual server port written to `data/state/server_port`
   - Monitors subprocess; restarts on exit code 42 (restart signal)
  - First-run wizard (shared desktop/web onboarding for multi-key and optional local setup)
   - **Graceful shutdown with orphan cleanup** (see Shutdown section below)

2. **server.py** — self-editable inner server. Can be modified by the agent.
   - Starlette app with HTTP API + WebSocket
   - Runs supervisor in a background thread
   - Supervisor manages worker pool, task queue, message routing
   - Local model lifecycle endpoints extracted to `ouroboros/gateway/models.py`

### Data layout (`~/Ouroboros/`)

```
~/Ouroboros/
├── repo/              ← Agent's self-modifying git repository
│   ├── server.py      ← The running server (kept in sync via the launcher-managed git clone, NOT copied from the workspace on each launch; see §2)
│   ├── ouroboros/      ← Agent core package
│   │   └── gateway/models.py  ← Local model API endpoints (extracted from server.py)
│   ├── supervisor/     ← Supervisor package
│   ├── web/            ← Web UI files
│   │   └── modules/    ← ES module pages (chat, logs, evolution, etc.)
│   ├── docs/           ← Project documentation
│   │   ├── ARCHITECTURE.md ← This document
│   │   ├── DEVELOPMENT.md  ← Engineering handbook (naming, entity types, review protocol)
│   │   ├── CHECKLISTS.md   ← Pre-commit review checklists (single source of truth)
│   │   ├── CREATING_SKILLS.md ← Skill author guide (manifest schema, PluginAPI, widgets, publishing)
│   │   └── DEPLOYMENT.md ← Deployment notes, including trusted Docker/Kubernetes non-local bind policy
│   └── prompts/        ← System prompts (SYSTEM.md, SAFETY.md, CONSCIOUSNESS.md)
	├── data/
	│   ├── settings.json   ← User settings (API keys, models, budget)
	│   ├── task_results/
	│   │   ├── artifacts/<task_id>/
	│   │   │   ├── .artifact_manifest.json ← Private task-artifact metadata for copied user/process outputs and provenance
│   │   │   ├── .scratch_manifest.json ← (v6.52.2) declared ephemeral `scratch=[...]` {abs_path: sha256} fingerprints; a matching untracked file is excluded from the workspace patch only while its content still matches (never a deliverable)
	│   │   │   └── <artifact files> ← Canonical task artifacts, including workspace patches, verification ledgers, and copied external deliverables
	│   │   └── artifact_versions/<task_id>/ ← Non-manifest recovery history for overwritten user-visible deliverables (last 5 versions per artifact name)
	│   ├── task_drives/<task_id>/ ← Task-scoped scratch for direct tasks and light-mode run_script defaults; startup prunes terminal tasks after the headless retention window
	│   ├── task_trees/<root_task_id>/blackboard.jsonl ← (v6.38.0) Task-tree coordination ledger: append-only swarm blackboard + child→parent beacons (tree_note/tree_read), scoped to the whole tree; EPHEMERAL coordination (distinct from the durable project journal)
	│   ├── state/
│   │   ├── state.json  ← Runtime state and compatibility cost projection (never the monetary authority)
│   │   ├── usage_attempts.jsonl ← Append-only monetary authority; every physical provider send has its own attempt id and state transition. A settled attempt with `cost=None` and a numeric reservation upper bound is counted at that bound as unresolved (protecting real spend of an unknown-price success from under-count); a zero-usage HTTP-200 body-error (429/5xx passed through the body) is instead settled at a confirmed $0 so its bound is released, not accumulated into phantom budget exhaustion under a provider storm (v6.65.4)
│   │   ├── usage_attempts.quarantine.jsonl ← Loud quarantine evidence for a proven corrupt final ledger row; the validated prefix remains readable
│   │   ├── usage_import_watermark.json ← Resumable/idempotent legacy-import watermark plus source hashes and archive reference
│   │   ├── server_port ← Active HTTP port used by the launcher/browser handoff
│   │   ├── server_process.json ← Launcher-owned server PID/process-group identity record for relaunch cleanup
│   │   ├── advisory_review.json ← Durable advisory/review ledger (runs, attempts, obligations, commit-readiness debts)
│   │   ├── deep_self_review_context.json ← Last deep self-review Generated Deep Self-Review Atlas manifest and model metadata
│   │   ├── code_intel/<repo_key>/inventory.json ← Internal Code Inventory v2 facts (file hashes, dispositions, symbols/imports/calls/references; no raw source cache)
│   │   ├── evolution_metrics_cache.json ← Cached per-tag Evolution metrics (schema 1; regenerated by `/api/evolution-data` / `collect_evolution_metrics`)
│   │   ├── evolution_campaign.json ← Active/paused Evolution Campaign objective, progress, cycle history, and budget counters
│   │   ├── evolution_checkpoints.jsonl ← Append-only per-evolution-cycle checkpoints with git/memory hashes and status/cost facts
│   │   ├── post_task_evolution_request.json ← Durable post-task self-evolution promotion signal (worker-written on the canonical drive; the supervisor idle tick consumes it to set the campaign objective + enable evolution, then deletes it; one-shot). When the durable owner-stop sentinel `state.evolution_owner_stopped` is set, `apply_pending_request` DROPS this request instead of consuming it, so an owner stop is never silently undone by a queued promotion.
│   │   ├── post_task_evolution_counter.json ← Per-drive task counter for the post-task evolution `every_n` cadence
│   │   ├── scheduled_tasks.json ← Queue-backed cron schedules (5-field cron, timezone, last/next run, task template)
│   │   ├── projects.json ← Project registry: immutable id/chat identity, optional working folder, lifecycle/routing fence, visible revision, and deletion error; tombstones are durable and never age-pruned
│   │   ├── project_task_bindings.json ← Task→project bindings (schema v1) with a REQUIRED typed origin: the ingress-captured source-row ref (+`source_text`, the retention-proof full copy, stored only for CROSS-thread origins — i.e. the message that started the project) or a closed-enum `origin_absent` reason. Immutable except ONE-WAY enrichment (a same-project re-bind may fill a missing ref; a valid ref is never changed); one root belongs to at most one Project and tombstoning never removes the binding. The retention-proof invariant is FORWARD-ONLY by owner decision: pre-v6.73.0 bindings (no `source_text`) are not migrated and their start messages remain rotation-vulnerable as before
│   │   ├── ui_preferences.json ← Owner-local layout preferences and monotonic `project_seen_revision` paint ACKs; legacy `project_last_viewed`/`project_hidden` are one-minor deprecated no-ops
│   │   ├── queue_snapshot.json
│   │   ├── extension_companions.json ← Runtime snapshot for live extension companion processes
│   │   ├── extension_reconcile/ ← Worker-written extension reconcile markers consumed by the server lifespan pickup task
│   │   ├── review_continuations/ ← Per-task blocked-review continuation payloads (+ quarantined corrupt files under `corrupt/`)
│   │   ├── workspace_executor_processes/ ← Durable local/docker executor foreground/service cleanup records for panic/shutdown recovery
│   │   └── skills/              ← Phase 3 external-skill state plane (sibling of advisory_review.json, not shared)
│   │       └── <skill_name>/
│   │           ├── enabled.json ← {"enabled": bool, "updated_at": iso_ts}
│   │           ├── review.json  ← {"content_hash": str, "findings": [...], "reviewer_models": [...], "timestamp": iso_ts, "raw_actor_records": [...], "advisory_result": {...}, ...}; `advisory_result` records optional fail-open Claude Code skill-advisory raw/session metadata, while tri-model findings remain authoritative. For full PASS/FAIL finding sets, status is computed live on load as `clean`/`warnings`/`blockers` from findings (`status` may remain only on legacy/pending infrastructure states; enforcement is applied later by `skill_review_gate`)
│   │           ├── owner_attestation.json ← (C1, v6.39; v6.43 official-hub extension) owner-issued marker: the owner skipped the EXPENSIVE LLM review for their own external/self-authored skill or for a freshly hash-verified official OuroborosHub payload. review.json then carries `review_profile="owner_attested"` + `reviewer_models=["owner_attestation"]`; the verdict is valid ONLY while this marker is present (removing it invalidates it, like native_seed provenance), the deterministic preflight floor still ran, and a content edit stales it via `content_hash`. An OWNER-STATE file: the agent can never forge it
│   │           ├── review_history.jsonl ← compact recent skill-review attempts (`status`, `content_hash`, failure signature) used for anti-thrashing/convergence context
│   │           ├── accepted_rebuttals.json ← accepted skill-review rebuttals injected into later review prompts
│   │           ├── deps.json    ← isolated dependency install fingerprint for skills with reviewed install specs
│   │           ├── auto_repair.json ← Marketplace auto-repair dedup marker; tracks attempted payload hashes so one broken payload cannot enqueue endless repair tasks
│   │           ├── health.json  ← durable per-extension health vector (v6.15: status + last_known_good vs last_observed); flags live->broken regressions across restarts for health invariants + startup check + Installed UI
│   │           ├── auth_token.json ← content-hash-bound Host Service token for reviewed live extensions
│   │           ├── extension_calls/ ← transient per-call child-process payload/result JSON files for isolated-dep extension catalog/tool/route/WS dispatch; files are private runtime transport state and are removed after each dispatch
│   │           └── __extension_imports/<pid>-<uuid>/skill/  ← Phase 4 staged import tree for type:extension skills (in-process host loads tag the leaf with the owner PID; created on load, removed on unload; see §13.1)
│   ├── memory/
│   │   ├── identity.md     ← Agent's self-description (persistent)
│   │   ├── scratchpad.md   ← Working memory (auto-generated from scratchpad_blocks.json)
│   │   ├── scratchpad_blocks.json ← Append-block scratchpad (FIFO, max 10)
│   │   ├── dialogue_blocks.json ← Block-wise consolidated chat history
│   │   ├── dialogue_summary.md ← Retired legacy flat dialogue summary (read-only historical fallback when present; not auto-migrated)
│   │   ├── dialogue_meta.json  ← Consolidation metadata (offsets, counts)
│   │   ├── WORLD.md        ← System profile (generated on first run)
│   │   ├── knowledge/      ← Structured knowledge base files
│   │   ├── identity_journal.jsonl    ← Identity update journal
│   │   ├── scratchpad_journal.jsonl  ← Scratchpad block eviction journal
│   │   ├── knowledge_journal.jsonl   ← Knowledge write journal
│   │   ├── knowledge_history.jsonl   ← Rollback-grade knowledge write history with old/new hashes and content refs
│   │   ├── knowledge/patterns_history.jsonl ← Append-only Pattern Register rewrite history for provenance/recovery
│   │   ├── deep_review.md            ← Last deep self-review report (written by deep_self_review task)
│   │   ├── registry.md              ← Source-of-truth awareness map (what data the agent has vs doesn't have)
│   │   ├── knowledge/improvement-backlog.md ← Durable advisory backlog of concrete post-task improvements
│   │   └── owner_mailbox/           ← Per-task user message files (compat path name)
│   ├── projects/<project_id>/knowledge/ ← Phase 3b per-project facts store (project-scoped knowledge; isolated from memory/knowledge and from the forked seed; no per-project identity). Provenance sidecars live alongside as projects/<project_id>/knowledge_history.jsonl and knowledge_journal.jsonl
│   ├── observability/
│   │   ├── blobs/<sha256>.json.gz ← Private compressed content-addressed forensic payloads (`0600` files under private dirs)
│   │   └── calls/<task_id>/<call_id>.json ← Private call manifests with blob refs, hashes, correlation ids, timing, usage, and redaction status
│   ├── services/
│   │   └── <task_id>/<service>.log ← Task-scoped long-running service logs; public tool output exposes bounded redacted tails plus private blob refs
│   ├── logs/
│   │   ├── chat.jsonl      ← Canonical chat message log (one logical message is stored once and projected into Main/Project lenses)
│   │   ├── chat_annotations.jsonl ← Compact presentation-only routing status by `client_message_id`; retention/compaction follows retained chat history and a torn final row is ignored
│   │   ├── progress.jsonl  ← Progress/thinking messages (BG consciousness, tasks)
│   │   ├── events.jsonl    ← LLM rounds, task lifecycle, errors
│   │   ├── tools.jsonl     ← Tool call log with args/results
│   │   ├── supervisor.jsonl ← Supervisor-level events
│   │   ├── task_reflections.jsonl ← Execution reflections (process memory)
│   │   └── skills/         ← Optional skill/companion runtime logs
│   ├── archive/            ← Rotated logs, rescue snapshots
│   └── uploads/            ← Chat file attachments (uploaded via paperclip button)
├── Deliverables/      ← (v6.38.0) Visible user-deliverables container: a BARE user_files filename (no directory) lands here instead of the home root (OUROBOROS_DELIVERABLES_ROOT; sibling of projects/, outside repo/ and data/, never GC-pruned)
└── ouroboros.pid           ← PID lock file (platform lock — auto-released on crash)
```

---

## 2. Startup / Onboarding Flow

```
launcher.py main()
  │
  ├── acquire_pid_lock()        → Show "already running" if locked
  ├── check_git()               → Show "install git" wizard if missing
  ├── bootstrap_repo()          → ensure_managed_repo(): first run clones from the embedded
  │                               repo.bundle + validates repo_bundle_manifest.json;
  │                               subsequent runs verify the managed clone's bootstrap pin
  │                               (source_sha + release_tag + bundle_sha256) and ensure the
  │                               managed remote metadata exists. Ordinary restart cleanup
  │                               lives in supervisor/git_ops.checkout_and_reset(), called by
  │                               server.py::_bootstrap_supervisor_repo(); it preserves the
  │                               local branch HEAD and cleans only the working tree. Explicit
  │                               Update Now uses a pinned update-intent SHA for official reset.
  ├── _run_first_run_wizard()   → Show shared setup wizard if no runnable config
  │                               (access entry → models → review mode → budget → summary)
  │                               Saves to ~/Ouroboros/data/settings.json
  ├── agent_lifecycle_loop()    → Background thread: start/monitor server.py
  └── webview.start()           → Open PyWebView window at the port from data/state/server_port
```

On macOS/Linux the launcher starts `server.py` in its own session/process
group and persists a verified `data/state/server_process.json` record
(pid, pgid, server path, repo path, port, timestamp). Startup preflight
verifies that the recorded PID still looks like this repo's `server.py` before
killing the recorded process group/tree, then runs the existing runtime-port
sweep as defense-in-depth. Windows keeps the Job Object kill-on-close path.

### First-run wizard

Shown when `settings.json` does not contain any supported remote provider key and has no
`LOCAL_MODEL_SOURCE`.

- Existing OpenRouter, OpenAI, OpenAI-compatible, Cloud.ru, GigaChat, Anthropic, or local-model-source settings skip the wizard automatically.
- The wizard is shared between desktop and web: one HTML/CSS/JS onboarding flow is rendered directly in pywebview for desktop and injected into a blocking web overlay for Docker/browser runs.
- The wizard is multi-step and provider-aware: it starts with a single access step that accepts multiple remote keys plus optional local-model setup, then shows visible model defaults, a dedicated review-mode step, a dedicated budget step, and the final summary before save.
- The wizard keeps the access step compact with responsive two-column field grids on normal desktop widths; mobile/narrow windows fall back to one column. Rarely used provider fields (Cloud.ru plus the OpenAI-compatible URL/key pair) sit in a collapsed "More options" group that auto-opens when any of its fields already carries a value; collapsed inputs stay mounted in the DOM so validation and payload build always see them.
- When an Anthropic key is present (including an unsaved key typed into the current wizard step), onboarding shows the Claude runtime status with `Repair Runtime` and `Skip for now` options without falsely warning that no Anthropic key exists.
- Desktop first-run uses the same onboarding bundle and talks to Claude SDK install/status through `pywebview` bridge methods.
  Web onboarding uses `/api/claude-code/status` and `/api/claude-code/install`.
- The wizard blocks progression if nothing runnable is configured.
- When OpenRouter is absent and official OpenAI is the only configured remote runtime, untouched default model values are auto-remapped to `openai::gpt-5.6-terra` / `openai::gpt-5.6-sol` / `openai::gpt-5.6-luna` so first-run startup does not strand the app on OpenRouter-only defaults.
- `web_search` uses the best configured backend in order: official OpenAI Responses, OpenRouter `openrouter:web_search` server tool, Anthropic `web_search_20250305`, then optional `ddgs`. Results are JSON with `answer`, `sources[]`, and `backend`; usage events include task/root/parent/delegation attribution and `source=web_search` when the backend reports usage. Missing credentials surface as an explicit unavailable-backends JSON error, not as repeated opaque tool failures. An empty OpenAI result (no answer text and no sources) falls through to the next backend rather than returning a fake `(no answer)` success, so a degenerate first leg cannot shadow a working one.
- v6.27.0 benchmark-harness hardening (rationale, so future maintainers need not dig commits): (1) **Service `keep_alive` / `service_teardown=keep`** lets a service deliberately outlive its task so an external verifier can connect; it stays custody-ledgered and dies on session change/panic, and cancel/hard-timeout worker kills now spare ledgered keep services (`kill_pid_tree(exclude_pids=...)`, POSIX-only — on Windows `kill_pid_tree` tree-kills via `taskkill /T` and does not honor exclusions, so `service_teardown=keep` is not preserved across Windows cancel/hard-timeout). (2) **Safety parse** does a robust bracket-scan + one same-slot repair retry, then fails closed — the worst-status object across candidates wins so an echoed `SAFE` cannot mask a `DANGEROUS` verdict. (3) **Deadline milestones** (50/25/10% remaining) and the deadline-derived `run_command` cap fire only when a task carries `deadline_at`; they are inert on Terminal-Bench leaderboard runs by design (Harbor owns task timeouts). (4) **External-workspace git policy** allows full local git in a task workspace while deterministically blocking any git that targets the Ouroboros self-repo/data via cwd, `-C`, `--git-dir`/`--work-tree`, `GIT_DIR`/`GIT_WORK_TREE` env, positional path, or glued/newline-separated segments. (5) **`search_code`** pre-enumerates a policy-gated file list (each path filtered through `path_allowed` before rg sees it — a security property), skips non-regular and oversized files, caps the scan at `MAX_SEARCH_FILES_SCANNED` with an explicit "scan stopped at N files" note, and hands the list to rg in batches (`batch_size=400`) to stay under `ARG_MAX`, so a search whose root resolves to `/` cannot OOM or `E2BIG` the worker. (6) **Review enforcement** (advisory vs blocking) is owner-only; the agent must not hardcode findings to always-block (BIBLE P3), pinned by item-agnostic invariant tests in the frozen contract suite.
- When Cloud.ru is the only configured remote runtime, first-run model defaults use explicit `cloudru::...` IDs from `provider_models.CLOUDRU_DIRECT_DEFAULTS`. OpenAI-compatible endpoints are first-run capable but never receive guessed defaults: the wizard asks for the base URL/key, can proxy `/models` through `/api/openai-compatible/models`, and requires explicit `openai-compatible::...` model slot values because arbitrary compatible endpoints have no universal safe model ID.
- Closing the wizard without saving is non-fatal: the main app still launches and the user can finish configuration in Settings.

### Launcher-managed bundle bootstrap

Packaged releases ship an embedded git bundle (`repo.bundle`) plus
`repo_bundle_manifest.json`. On the first launcher-managed run the launcher:

- verifies the manifest against the packaged app version,
- checks the bundle SHA-256,
- initializes `~/Ouroboros/repo/` as a managed git checkout,
- checks out the manifest-pinned `source_sha`,
- then keeps the managed checkout as the self-modifying local branch.
  Later launches clean the working tree without moving local commits;
  official branch following happens only through explicit managed updates.
  If a newer app bundle carries different embedded repo metadata, the
  launcher refreshes the managed remote/manifest metadata in place instead
  of archiving and replacing an existing git checkout.

Git remotes are role-based. `managed` is the official read/update source for
release provenance and explicit Updates-panel application. `origin` is the
personal persistence target for reviewed self-modification commits, tags, CI,
and optional metrics. `repo_remotes.py` provisions `origin` by reusing a
verified fork of `razzant/ouroboros` or creating one when a GitHub token is
configured; it never makes `origin` the official update source and never writes
to `managed`.

Self-modification success is local-first: `commit_reviewed` creating a reviewed
local commit is the durable boundary. `origin` push and CI are best-effort
follow-ups; missing `origin` is a local-only mode, not a broken evolution run.
Autonomous restart uses the local commit SHA or a clean no-op state as its
eligibility signal, never remote push success.

Reviewed-change pytest preflight is hermetic. `ouroboros/preflight_runner.py`
creates a disposable detached worktree, replays the candidate staged/unstaged
diff plus untracked candidate files, runs the default pytest suite serially with temporary
`OUROBOROS_DATA_DIR`, `OUROBOROS_SETTINGS_PATH`, and `PYTHONPYCACHEPREFIX`, and
scrubs inherited `OUROBOROS_*` behavior plus secret-class environment values.
This prevents tests launched by advisory/commit review from writing live
`data/settings.json`, inheriting owner runtime modes, or triggering
launcher-managed reset behavior against the live repo. CI alone owns its
parallel non-serial plus serial split.

Safety-critical protection is no longer implemented as "copy these files from the
bundle on every launch". The runtime guardrails are the runtime-mode protected-path
policy enforced by the `registry.py` dispatcher plus the launcher-managed repo
integrity checks.

### Single-source rescue on startup (v4.36.1+)

A dirty worktree (uncommitted changes inherited from the previous session) is
handled by exactly ONE supervisor-owned mechanism: `safe_restart(...)` from
`server.py::_bootstrap_supervisor_repo()` / agent restart handling. Ordinary
launcher bootstrap may still use `rescue_and_reset` after a complete rescue
snapshot, but an active evolution transaction switches to `rescue_and_block`:
the rescue ref/path is attached to the campaign transaction, the campaign is
paused, and the dirty worktree is left intact for deliberate recovery. This
keeps reset from erasing in-progress self-modification while preserving the
single supervisor-owned rescue path.

```mermaid
flowchart TD
    L[launcher.py] -->|spawn| S[server.py lifespan]
    S --> B["_bootstrap_supervisor_repo()"]
    B --> D{"active evolution transaction?"}
    D -->|no| SR["safe_restart(rescue_and_reset)"]
    D -->|yes| SB["safe_restart(rescue_and_block)"]
    SR --> RS["_create_rescue_snapshot() → rescue directory<br/>(only if dirty tree)"]
    SR --> RB[checkout local branch + reset --hard HEAD]
    SB --> TX["link rescue_ref/path to transaction<br/>pause campaign"]
    TX --> STOP["leave dirty tree for recovery"]
    RB --> SW["spawn_workers() → worker_main"]
    SW --> MA["make_agent() → OuroborosAgent.__init__"]
    MA --> LB["_log_worker_boot_once()"]
    LB --> CU["check_uncommitted_changes() DIAGNOSTIC-ONLY<br/>(never commits, never mutates git)"]
    LB --> VR["verify_restart + verify_system_state"]
```

Worker-side `ouroboros/agent_startup_checks.py::check_uncommitted_changes()` is
**warning-only**: it emits a `supervisor_side_rescue_owns_this` skip marker in
its result when the tree is dirty, and never runs `git add` or `git commit`.
**Why warning-only:** `OUROBOROS_MANAGED_BY_LAUNCHER=1` is inherited by every
subprocess (pytest runs, A2A agent-card builder via
`_build_skills_from_registry`, supervisor-side `_get_chat_agent`); a duplicate
worker-side `auto-rescue` commit would let any of those subprocess paths steal
the agent's in-progress edits into a commit on `ouroboros`. The single rescue
path lives in `safe_restart` so subprocesses cannot race it.

### Agent interpreter handle (`OUROBOROS_AGENT_PYTHON`)

`server.py` exposes the interpreter that launched Ouroboros as
`OUROBOROS_AGENT_PYTHON` early in startup, immediately after `REPO_DIR` is added
to `sys.path` and before workers or review subprocesses are spawned. Existing
operator/test overrides are respected; the assignment is guarded so exotic
embedded runtimes with `sys.executable is None` or `""` do not write an invalid
environment value.

The three pytest subprocess surfaces use the same fallback chain:
`sys.executable or os.environ["OUROBOROS_AGENT_PYTHON"] or "python3"`, then run
`-m pytest` through that interpreter:

- `ouroboros/tools/review_helpers.py::_run_review_preflight_tests`
- `ouroboros/tools/git.py::_run_pre_push_tests`
- `ouroboros/tools/shell.py::_run_validation`

This keeps packaged app bundles from depending on a `pytest` or `python`
executable on the user's PATH; the test runner comes from the same Python
environment that has Ouroboros dependencies installed.

For the four user launch surfaces (`run_command`, `run_script`, `start_service`,
and run-kind `verify_and_record`), `python_interpreter.py` resolves an unversioned
`python`/`python3` exactly once in registry pre-dispatch, before guard
normalization. Guard and handler receive the same argv bytes. Priority is a
reviewed skill environment, backend `python3` for Docker/executors, a valid
project `.venv` or target-environment PATH for external/user work, then the
verified `OUROBOROS_AGENT_PYTHON` for the system repo/task/artifact surfaces
(the validated current process executable is the direct/library fallback before
server bootstrap establishes that environment handle).
Absolute or versioned interpreters, non-Python commands, `sh -c` bodies, and
OSWorld `remote_exec` remain literal. Resolution provenance is traced without
secrets; an unproved system interpreter fails closed rather than falling through
to an ambiguous host PATH.

---

## 3. Web UI Pages & Buttons

The Web UI is a vanilla-JS SPA (`web/index.html`, `web/style.css`, `web/settings.css`, `web/modules/*`). There is intentionally no TypeScript/build step: the app must remain inspectable and editable by Ouroboros itself.

### Navigation

A left `#primary-sidebar` of ROWS (`.nav-row`, not an icon rail): Chat (Main), a compact data-driven Projects row, Files, Skills, Widgets, Dashboard, Settings. The Projects header keeps the shared layers icon, unread pill, correct chevron, and an always-visible `+`; each sibling row has an accessible Rename/Delete menu (pointer and keyboard focus, Enter/Space, Escape, click-outside, viewport-safe placement) and the backend-enforced name limit is 80 characters. `web/app.js::syncNavigationState` keeps the active row, Projects expansion, and any open project panel in sync. A project opens as a right split panel (desktop) / full-width overlay with backdrop (mobile) hosting a full chat instance over the ONE shared WebSocket (client-side fan-out by `chat_id`). On narrow screens the sidebar collapses behind an "Open navigation" drawer (not a fixed bottom bar). Mobile navigation and project overlays close through their explicit toggle, close button, or backdrop controls; there is no swipe-gesture layer competing with scroll, selection, or keyboard state. About is a Settings sub-tab, not a top-level page.

### Shared UI primitives

- `web/modules/page_header.js` renders common page headers/tab strips.
- `web/modules/page_icons.js` is the nav/header icon SSOT.
- `web/modules/api_client.js` is the frontend API boundary.
- `web/modules/api_types.js` mirrors browser-facing envelopes with JSDoc.
- `web/modules/ui_helpers.js` centralizes tone badges, age labels, inline status, host-bridge downloads, and the safe field renderer/value collector shared by Widgets and Settings (Settings retains its narrow route/component contract).
- `web/modules/skill_card_renderer.js` renders installed Skills cards from shared lifecycle/review/grant state.
- `web/modules/update_status.js` (v6.41.0) renders the main-screen Update pill + the staged auto/assisted/manual dialog; `web/modules/activity.js` (v6.41.0) renders the Dashboard Activity subtab (cron schedules, running/queued tasks, background consciousness) with direct mechanical controls. Both use the shared `.btn` button system.
- `web/modules/toast.js`, `masonry.js`, and CSS tokens in `style.css` keep cards/notifications/layout consistent without a build system.

Rationale: frontend work should not require understanding supervisor, worker, marketplace, extension, MCP, local-model, and settings internals at once. The Gateway Boundary and API client keep browser code pointed at one explicit contract.

### Chat

`web/modules/chat.js` owns the message timeline, input, attachment staging, input recall, budget pill, runtime controls, and live task cards. It loads persisted history from `/api/chat/history`, merges echoed local messages by `client_message_id`, and collapses task/progress/tool chatter into expandable cards rather than transcript spam. Every top-level bubble, media item, and task-card root carries numeric epoch `data-ts` derived from the raw source timestamp before display formatting. `insertMessageNode` inserts before the first sibling with a strictly greater timestamp, preserves arrival order for equal timestamps, leaves timestamp-free nodes on the append path, and always keeps the typing indicator last; an insertion above a scrolled-up viewport compensates `scrollTop` by the inserted `scrollHeight` delta, while a reconnect rebuild restores the first visible timestamped DOM anchor (so simultaneous older rows above and new rows below cannot move the reader), and near-bottom insertion retains normal autoscroll. A task card anchors its earliest raw timestamp on its DOM root, so ordinary card recreation naturally starts a fresh anchor without separate reset state. A logical chat message has one canonical `chat.jsonl` row; Project conversion stores a source ref and the history endpoint projects that row into the Project lens rather than creating a mirror bubble, duplicate unread, or duplicate cost. Owner-visible routing status is a compact `chat_annotations.jsonl` sidecar keyed by `client_message_id`; it is presentation-only. Chat attachments are staged client-side from paperclip, paste, and chat-wide file drag/drop, capped at 10 files, 50 MB per file, and 100 MB total per message; upload happens only immediately before send, attachment messages bypass the offline WebSocket queue, and partial upload/send failures best-effort DELETE already uploaded temporary files while preserving the staged batch for retry. The composer uses a responsive glass layout: desktop keeps Swarm, Low/Max, and Send inside the frosted text-entry surface; mobile lifts Swarm and Low/Max into a compact control row above the textarea while Send stays inside the field. Subagent progress uses separate child cards keyed by `subagent_task_id`/`task_id`; parent cards receive lineage references (`parent_task_id`, `root_task_id`, child id, role) without duplicating child bubbles on reload/reconnect. Ordinary nested child cards stay visible but collapsed by default with role-first headings so deep trees remain scannable; a child carrying review evidence expands by default so actor/model/verdict provenance is disclosed immediately. Each card is its own inline-size container: narrow summaries and timeline controls wrap against the card's actual width, and nesting indentation flattens after the first level so deep trees retain a usable text column. Mobile keyboard handling lives in `web/app.js` + CSS `keyboard-open` classes. Keyboard mode requires a focused editable plus a visual-viewport shrink from the last stable application viewport; opening the drawer blurs that editable and clears the keyboard state before navigation is shown, so sidebar state, backdrop, and rendered visibility cannot diverge.

A collapsed live card carries a dedicated activity line (`[data-live-activity]`, decided by the pure `projectCollapsedActivity`): the title keeps identity — the coined project name on root cards, role·model·id on subagent cards — while the line shows the latest meaningful action (the active headline for named root cards, the routed progress body for children). Unnamed root cards suppress the line because their title already shows the activity, and a finished card keeps its last activity. Card cost is a sticky per-record projection (`taskCostProjection`/`mergeStickyCostMeta`, `{meta, ts, final}`): only frames carrying task-scope accounting evidence (`cost_accounting_status`/`cost_final`/subtree/reserve fields — never a bare `llm_round_finished` per-round `cost_usd` delta) may update it; rank is unavailable &lt; pending &lt; final (an honest reading always outranks an unknown; a settled value outranks both), the newer raw source timestamp wins among equals, and costless frames re-render the stored value instead of erasing it. Reload replays the same truth: the flat cost fields (`TASK_COST_META_FIELDS`, `ouroboros/task_results.py`) are written onto `task_summary` chat rows from the pre-synthesis usage snapshot in `agent_task_pipeline`, passed through history's `_copy_task_summary_metadata`, and overridden by the persisted `task_results/<id>.json` values in `_annotate_terminal_task_truth`.

Live pooled ROOT cards carry a "Cancel run" action (v6.82): a confirm dialog, then `api_client.cancelTask(id, {cascade:true})` — the task plus its live subtree, resolved by the existing `task_done{status:"cancelled"}` events (the button disables on click; a 404 completion race withdraws the action from `cancelableTaskIds`, the eligibility authority). Eligibility needs the supervisor's host-attested `cancelable` marker on top of the structural checks (`cancelRunEligibility`: non-subagent, non-reusable slot, unfinished, unconverted), because card shape alone cannot tell a pooled root from an in-process direct-chat turn; the marker is emitted only for lineage-resolved non-subagent roots and carries the RUNNING row's lineage, which is what lets a timeout-retry root be cancelled. Cancellation renders honestly: `cancelled` is a first-class terminal severity/phase (`taskOutcomeSeverity`/`taskTerminalPhase`) consumed by both task_done summarizers, the live-card finishers and the history replay fallback — a cancelled root shows "Cancelled" on reload, never "Done". Teardown takes CUSTODY of each task, in three ordered phases: capture under the queue lock (a PENDING row leaves the queue; a RUNNING task deliberately KEEPS its row — authoritatively visible, lineage intact — until confirmed death and durable persistence), which also marks the worker slot `reaping` (the existing marker `assign_tasks` and `ensure_workers_healthy` already skip, so the off-lock kill cannot be seen as a crash and requeued, respawned twice, or counted toward crash-storm detection); kill and JOIN outside the lock with death CONFIRMED; then — and only then — persist the terminal result and publish `task_done`, respawn, drive cleanup and snapshot. Any failure in the last two phases restores custody, clears the marker and yields the typed `failed`. A task that already reached its own terminal result is never captured (a RUNNING row for one is left to its own finalizer), so completion always wins. Outcomes are TYPED per task, never OR-aggregated: only terminalized ids are marked done, a refused id is retried by the next sweep, and the cascade ends on an unconditional postcondition — nothing live, nothing refused — or the endpoint answers 503. Each sweep fences EVERY id it captures before releasing the lock (once a cancelled descendant leaves the live maps, a still-draining schedule event names a parent that no longer exists), the fence walks the surviving ancestry, and bounded re-sweeps are defence rather than authority. A promoted root still PENDING carries no card marker yet and is cancelled from the Dashboard Activity row until it starts.

Project unread is the durable comparison `visible_revision > project_seen_revision`. Only an owner-visible assistant/result row or a real incident advances `visible_revision`; ordinary progress/heartbeat telemetry does not. The browser posts a monotonic, server-clamped ACK only after the Project room has actually painted, so stale tabs and racing responses cannot move the cursor backwards or acknowledge content that was never shown.

History sync is intentionally two-pass: progress/system entries are replayed first to build timestamp-anchored live-card timelines, then cards and regular user/assistant messages are inserted chronologically before `finishLiveCard` seals them. This preserves progress-only, terminal-without-summary, nested-subagent, and disconnected-card replay while preventing `taskState.completed` from being set before progress events apply. Live-card and Logs timestamps include the date for non-today entries; that display string is deliberately separate from sortable `data-ts`.

`/api/chat/history` is rotation-aware (`gateway/history.py::make_chat_history_endpoint._read_chat_entries`, v6.58.5/6.58.6): the supervisor rotates `logs/chat.jsonl` to `archive/chat_<ts>.jsonl` once it crosses ~800KB (`supervisor/state.rotate_chat_log_if_needed`), so reading only the live file would erase the pre-rotation conversation — including delivered file/document bubbles — the instant a rotation happens. The endpoint therefore backfills from the most recent `archive/chat_*.jsonl` segments (newest-first, bounded to 3 files, until the requested thread's human-row quota is met) and reassembles them chronologically (oldest archive → live) off the event loop. The quota is counted with the SAME A2A + `chat_id`/project-thread filter used by the render loop, so a project-thread request backfills its OWN archived rows instead of being satisfied by unrelated main-chat rows still in the live file. A rotation changes granularity, never coverage (BIBLE P1: no silent loss).

### Files

`web/modules/files.js` is the browser file manager: directory tree, breadcrumbs, preview/editor, upload/download, copy/move, and write guards. Backend policy lives in `gateway/files.py`: root confinement is enforced on the RESOLVED path (v6.26.0) — symlinks whose target leaves the configured root are listed but not readable/writable/traversable; owner-only state and skill control-plane sidecars are protected.

### Skills, Marketplace, Widgets

`web/modules/skills.js` lists installed/bundled/user skills, review state, grants, enablement, repair affordances, and lifecycle progress. Marketplace panes (`marketplace.js`, `ouroboroshub.js`) install/update/uninstall skills through backend lifecycle jobs. Widgets are separate so extension UI surfaces are not buried in the skill list.

`web/modules/widgets.js` supports reviewed extension UI declarations: sandboxed `iframe`, additive declarative schema v1 (forms/actions/jobs/markdown/code/json/key-value/table/tabs/charts/stream/progress/subscription/poll/media/map/calendar/kanban plus `group`, `metric`, and `callout`), and sandboxed `kind: module` widgets served from reviewed skill payloads. The one recursive validator walks at most depth 8 / 256 nodes and reports exact tree paths. Groups and tabs may contain interactive children keyed by explicit id or stable tree path; `subscription.render` remains transitively passive. One widget disposer owns timers, streams, abort controllers, charts, and snapshots, and inactive tabs do not restart lifecycle work. Chart gaps remain `null` with `spanGaps=false` and share data with an ARIA-labelled semantic table fallback; native kanban `Move to` and drag/drop call the same route with `{card_id, column_id}`. Module widgets run in opaque `iframe srcdoc sandbox="allow-scripts"` with a parent-mediated fetch bridge restricted to `/api/extensions/<skill>/...`; they never execute in the SPA origin. Widget card order is an owner-local UI preference persisted through `/api/ui/preferences`; it does not change extension manifests or widget trust boundaries. v6.71.0 UI polish: rendered markdown/review disclosures share one `.ui-rich-content` contract (reserved list gutter, anywhere-wrap; applied to widget markdown and the Skills review history/findings — chat-bubble `.message` is deliberately untouched); widget charts render inside a bounded `.widget-chart-canvas` box (clamp 260–360px) and a data-only re-render adopts the live chart wrapper/canvas subtree (Chart.js watches the canvas PARENT for responsive resize, so the bare canvas must not be re-parented) and updates `chart.data` in place instead of destroy/recreate; poll refetches are SWR — `status` becomes `refreshing` (thin pulsing indicator) when data already exists, `loading` only on the first fetch. Live-line disclosures toggle from the whole non-interactive row surface (pure guard helper `liveLineRowToggleKey`: nested interactive elements and active text selection never toggle; focus returns to the real toggle button), and project/side-chat composers reuse the main chat's absolute dock + bottom-fade contract (`--chat-input-reserve` padding on the panel transcript, no second fade layer).

Rationale: useful extension UI should be possible, but the host must own rendering, sandboxing, and route confinement. Skills provide data and declarations; the browser host enforces the trust boundary.

Visible UI acceptance reuses the existing browser, vision, review-evidence, and
task-acceptance surfaces. The implementer opens at least one relevant real
consumer flow and actually inspects the rendered evidence with vision; a stored
screenshot alone is not inspection. The LLM selects states, viewports, and
additional engines from task risk. Mobile/WebKit are not a universal matrix, an
unavailable optional engine alone is not degradation, and unavailable evidence
the implementer judged necessary is disclosed as degraded/best-effort. No
visual-QA runner, endpoint, ledger, or browser auto-installer is introduced.

### Dashboard

Dashboard hosts Logs, Evolution, Costs, and Updates. Logs and Chat share event summarization (`log_events.js`) so task phases are described consistently. The same `taskOutcomeSeverity` reducer drives both surfaces: a degraded review or best-effort/degraded objective is warning-colored, never presented as a green solved result. Compact review projections disclose every panel and actor's model/provider, role, transport/parse/semantic state, coverage, quorum contribution, reason, and enforcement impact by default; since v6.70.0 the actor/panel `reason` is the COMPLETE redacted rationale (the former 500/800-char caps destroyed the only owner-reachable copy — reviewer rationale is a cognitive artifact, BIBLE P1) and each actor carries a forensic `response_ref` into the private observability store; full model prompts/responses themselves remain in that private audit storage. Evolution reads `/api/evolution-data`; Costs reads the physical-attempt ledger projection from `/api/cost-breakdown` and distinguishes confirmed/settled, reserved, unresolved upper bound, and unknown/unmetered work together with `cost_final`; it exposes budget controls from the shared setup contract via `/api/settings` GET/POST. Updates exposes official managed updates plus local recovery commits/tags.
The Logs page renders task/LLM/tool/progress events as grouped task cards, while Chat renders the same stream as a live task card so operational history and live dialogue stay visually consistent.
Worker tasks forward their `append_jsonl` log lines to the live dashboard over `EVENT_Q` via a per-worker log sink installed in `supervisor/workers.py::worker_main` (the WS log sink only exists in the main process), suppressing types that already arrive through a dedicated live sibling event — `tool_call`/`llm_round`/`task_checkpoint`/`task_done`/`llm_usage` — to avoid double broadcast and a double `task_checkpoint` file write. On load and on every reconnect the Logs page backfills recent history from `/api/logs/{events,tools,progress,supervisor}` (`web/modules/logs.js::backfillRecentLogs`) and dedupes the live-overlap window by event identity so the pre-connect window is neither dropped nor shown twice.
Chart.js is bundled locally as `web/chart.umd.min.js`; no CDN dependency by design.

### Forensic Observability and Typed Outcomes

`server.py` installs a `_SecretRedactingLogFilter` on its root log handlers (reusing the observability redaction SSOT) and quiets `httpx`/`httpcore` to WARNING, so third-party INFO lines can no longer print URL-embedded credentials (the Telegram `/bot<id>:<secret>/` polling line did exactly that; the token regex itself is fixed to match that URL form). `agent_startup_checks.check_stray_server_processes` is a report-only invariant naming ouroboros-server processes that belong to no current install (foreign pid, not in `server_process.json`, the custody ledger, or this process tree); it scans only this user's processes and runs at startup plus live through `build_health_invariants` behind a 15-minute TTL cache. The Skills card "Submit to OuroborosHub" button creates a real managed task via `POST /api/tasks` (the previous chat-command path printed "task queued" before any task existed, and the ephemeral decision turn cannot call `submit_skill_to_hub`); ephemeral decision turns carry a `decision_turn_rule` outcome contract — schedule real work via `promote_chat_to_task` or explicitly decline, never end on an unscheduled promise. After routing, their final no-tool response is self-contained because tool-round prose is transient progress rather than durable dialogue.

`ouroboros/observability.py` is the private replay layer for decision-affecting calls. It stores full payloads in `data/observability/blobs/<sha256>.json.gz`, manifests in `data/observability/calls/<task_id>/<call_id>.json`, and exposes only redacted previews plus blob refs through existing logs. LLM calls, review calls, supervisor/safety calls, and tool requests/results use correlated `execution_id`, `round_id`, `llm_call_id`, `tool_call_id`, and parent ids so a task can be reconstructed without trusting the truncated UI stream.

`ouroboros/llm_observability.py` is the LLM-side adapter: it persists provider request/response payloads before compaction can discard them and returns manifest refs for usage/outcome ledgers. `ouroboros/outcomes.py` is the typed result layer over the lifecycle record: `task_contract`, `outcome_axes`, `reason_code`, `loop_outcome`, `artifact_bundle`, and `verification_ledger` keep lifecycle, execution health, artifacts, objective evaluation, review status, and recovered tool failures separate. Objective success is filled only by the LLM-first host acceptance evaluator; if that evaluator did not run, objective is `not_evaluated`. Historical stored records with `result_status` are read through a compatibility normalizer, but new public task-result/API output uses `outcome_axes`; duplicate scheduling rejections are warning/degraded execution states, not red task failures. `task_results/<task_id>.json` remains the compatibility record; large verification details may spill to task-scoped artifacts.

The loop keeps one private `DeliveryCandidate` containing the complete answer text, content hash, monotonic revision, evidence fingerprint, acceptance binding, and finalization state. After a substantive candidate exists, a service round may return `keep` or `replace`; `replace` must include a complete replacement answer. One malformed control receives one repair round, then the prior complete candidate is preserved and finalization is marked degraded. A service notice alone does not invalidate evidence, while an owner message, tool effect, new child result, or verification receipt advances the revision and requires fresh delivery/acceptance binding. Task-scoped services finalize their declared outputs and teardown failures before the host acceptance panel; a changed service-evidence projection requires a complete replacement answer, while the ordinary `finally` cleanup remains an idempotent safety net. This is loop-local control, not a public tool or ledger, and it does not bypass the existing verification, acceptance, safety, skill-finalization, deadline, child-handoff, unconditional `FINAL ANSWER:` latch, or task-level answer-protocol gates.

Every direct child result must be dispositioned through the existing `tree_note(kind="decision")` surface with the tagged type `child_result_disposition`, the child id, `integrated | irrelevant | deferred`, and the expected SHA-256 of the complete result; the note text is the rationale. Only the join-ledger helper may validate direct lineage, recompute the current hash, and append the decision. That typed append-only task-tree row is the single durable disposition authority while the root task tree is active; `task_status` may expose its latest exact-hash decision through compatibility fields, but those fields are derived at read time and are never written to the child task result. Malformed input changes nothing, an exact retry is idempotent, and a later valid row deterministically supersedes an earlier decision for the same hash. `integrated` and `irrelevant` close only that exact content; `deferred` suppresses a repeated reminder for the unchanged result but cannot support clean `solved`, so deadline completion lists deferred work and remains degraded/best-effort. A changed child status, full result, trace summary, artifact status, or stable artifact identity makes the prior row audit-only and reopens absorption. Task-tree GC occurs only after root terminal state plus retention, when the decisions no longer participate in active finalization. Explicit cancellation wins a completion race: any late child result is ignored and removed with bounded child scratch, with no snapshot, copy, secondary authority, or restart recovery path.

Forced finalization is an honest positive shelf (v6.29.0). When a deadline grace window, budget stop, or round limit forces the final answer, the loop stamps the typed reason code (`finalization_grace` / `budget_exhausted` / `round_limit`) plus a typed `_best_effort_extracted` fact set ONLY when a real model answer came back, and `derive_loop_outcome` lands the result on `EXECUTION_BEST_EFFORT` when that fact is set and the final text is non-empty and not an error marker — a deterministic runtime-facts gate (P5-safe: no prose classification, no whitewash; host fallback strings such as budget rejection notices never set the fact and stay `failed`). `best_effort` is not "terminal success": CLI `_is_terminal_success` and the effective-status failure projection treat it as a non-failed, non-clean completion. The supervisor cooperates: when the grace window opens, `supervisor/queue.py` writes a typed `finalize_now` control into the task's owner mailbox (`ouroboros/owner_mailbox.py` entries carry a `kind`; control entries are routed structurally, never injected as owner prose), and the loop routes it to `_handle_forced_finalization` — one tool-less final answer inside the grace window, so a deadline never returns emptiness. On the hard-kill path the supervisor additionally salvages the last persisted assistant text from observability (`latest_llm_response_text`) into the terminal result. Budget exhaustion (`budget_remaining <= 0` past round 1) attempts one bounded tool-less best-effort extraction before rejecting. Provider-death joins the SAME shelf (v6.36.0): when the model returns no usable response after the transport same-model reroute + retries (+ the configured `OUROBOROS_MODEL_FALLBACKS` cross-model chain — walked deadline-aware, each link skipped while it sits on a short per-process 429-aware cooldown in `ouroboros/fallback_cooldown.py` so a task's own fallback walk / repeated rounds stop re-hammering a rate-limited model (per-process, not swarm-wide), with a small per-candidate total attempt cap), `_handle_provider_unavailable` (reason `provider_unavailable`) runs one tool-less final answer — which itself benefits from the reroute, so it often reaches a healthy provider — and otherwise salvages the last in-transcript assistant text, instead of discarding the workspace with a bare error string. If newer owner/tool/child/verification/service evidence has already made that retained candidate stale, provider or budget fallback preserves its old evidence fingerprint (`evidence_current=false`), clears acceptance authority, and appends a host-owned stale-evidence/resume disclosure; unchanged old text is never rebound as though it incorporated the newer evidence.

Task acceptance review is a completion coach (v6.29.0): with `classify_outcome_tier` policy, reviewer slots classify the deliverable tier — `solved` / `best_effort` / `blocked_with_evidence` — and name the single highest-value change to move one tier up, while the veto over FALSE `solved` claims is preserved (a `solved` tier with a FAIL verdict maps to objective `fail`). Blocking widening (v6.60.0, S1-lite quiz 18b): under `required`+`blocking`, `_collect_acceptance_obligations` turns contributing reviewers' findings into typed obligations at CRITICAL severity always, and ALSO at HIGH severity when the aggregate verdict itself is failing (signal FAIL or worst tier `blocked_with_evidence`) — the PB case where reviewers converged on a concrete "misses X" at high severity yet the task finalized clean; a PASS (incl. PASS-with-dissent) keeps the critical-only bar so clean runs are not taxed with hygiene items. The dead `verdict_is_advisory` request-policy key is gone — enforcement semantics live solely in `OUROBOROS_REVIEW_ENFORCEMENT`. The acceptance checklist now asks the SCOPE-CUT question explicitly (a silent/unjustified narrowing of the task is a high-severity finding, which under blocking becomes an obligation). Bench profiles (PB/TB defaults, SWE-Pro settings_base) run `REVIEW_ENFORCEMENT=blocking` + `TASK_REVIEW_MODE=required`. The objective axis consumes the aggregated worst-tier-wins classification: `solved`→`pass`, `best_effort`→`best_effort`, `blocked_with_evidence`→`fail`; without a tier the legacy verdict mapping applies. Final messages may carry a machine-readable `FINAL ANSWER: <answer>` line; `extract_final_answer` lifts it into the typed `final_answer` field of the loop outcome and task result record for exact-match deliverable consumers. **Answer protocol (v6.60.0, owner quiz 16b C+B):** the marker doctrine is a PER-TASK CONTRACT field, not a global prompt rule — `task_contract.answer_protocol` ("" | "final_answer_line", normalized by `normalize_answer_protocol`, propagated via `/api/tasks` `answer_protocol=` / CLI `--task-metadata-json` and inherited by subagents through the parent-contract spread). When declared, `context.py` injects the protocol instruction (with the opt-in `CANDIDATES:` ambiguity block) into the task's runtime context, and the marker NUDGES (`loop.py` P2 final-marker nudge) plus the pacing SALVAGE PHRASES (`task_pacing.py` wrap-up/10%-flush/intrinsic) activate — all through the ONE SSOT gate `answer_protocol_active`. Without it (ordinary chat/self tasks) no marker prompting ever appears; the LATCH + EXTRACTOR + typed `final_answer` stay UNCONDITIONAL (harmless without a marker, and still capturing a spontaneous one), `final_answer_missing_sentinel` keys on the typed payload (latch-recovered answers are not "missing"), and the no-op-attempt nudge keys on `expected_output` semantics with marker wording only under the protocol. GAIA declares the field in its solver; TB/SWE-Pro/PB deliberately do NOT (their deliverables are container state / patches / code, not an extracted line). The web UI presents a `FINAL ANSWER:` line as ordinary message text; the protocol-gated prompting and unconditional typed latch/extractor remain backend behavior, not a presentation capsule. Outcome honesty (v6.35.0): an unrecovered one-shot `run_command`/`run_script` non-zero exit (e.g. an X11-teardown `exit=1` after a green test, or an abandoned `find` probe) is demoted to a non-degrading `execution.cosmetic_tool_errors` bucket instead of forcing `EXECUTION_DEGRADED` — the execution axis means "harness/capability health", while "did it actually work?" lives on the objective/review axis (Bible P5). To keep that honest on the default `auto` path (where no review ran and the objective is `not_evaluated`), a structural `objective.warning = "residual_tool_errors_without_review"` is set when cosmetic errors exist with no judging review; the web UI escalates that warning to `warn` severity. `timeout` stays a blocking status. Policy-denial honesty (v6.57.0): an unrecovered POLICY refusal — a `*_blocked` status (`integration_blocked`, `workspace_blocked`, `light_mode_blocked`, `protected_blocked`, `resource_policy_blocked`, write/edit/shell `*_blocked`, …) on ANY tool — lands in a dedicated non-degrading `execution.policy_denials` bucket, distinct from the read-only `ignored_tool_errors` demotion: the runtime said "no" to an action, which is telemetry, not harness ill-health, and the agent's work is judged on the objective/review axis. So a headline `reason_code=tool_failure` is reserved for GENUINE unrecovered non-policy errors (`error`/`*_error`/`non_zero_exit`/`shell_error`/`timeout`/`unavailable`) without a valid deliverable — the site-presentation incident where `integration_blocked`+`LIST_FILES` reddened a shipped site is fixed at the classification, not by whitewashing prose (P5). `refused_out_of_scope` verify receipts are likewise non-failures in `has_failures`. `build_trace_summary` shows the honest bucket breakdown (`N errors, M policy-denied, …`) so post-task reflection/self-learning is not poisoned by counting policy refusals or intentional differential-probe exits as failures.

Verify-before-done flagship (v6.47.0, FR3). The core `verify_and_record` tool has the HOST run the agent's declared verification `check` (reusing run_command's safety/env/cwd machinery) and write a durable, host-attested receipt under `task_results/artifacts/<task_id>/verification_receipts.jsonl` (`validate_task_id`-guarded). A turn that produced real reviewable effects but recorded no grounding (no verify receipt, no trivial write/edit deliverable) triggers a ONE-SHOT verify-before-done nudge in `loop.py` (binary `_verify_nudged` latch, sibling BEFORE the acceptance-review gate so it reaches both `required` and `auto`; forced-finalization paths bypass it). `agent_task_pipeline.emit_task_results` applies this flag ONCE — after `derive_loop_outcome` and BEFORE the `task_eval`/`task_metrics`/`task_done` event stream — so the day-one monitoring metric reads it, not only the stored `task_result`: it injects the receipts into the trace for `build_verification_ledger` and, on a clean turn (execution ok), sets a BINARY objective warning — `receipt_absent` (effects but no grounding) or `expected_output_ungrounded` (M2 zero-grounding: a typed `expected_output` declared but no tool work and no structured answer). The single flagged `loop_outcome` is then threaded into `_store_task_result` (single source — no second derive), which persists it and builds the ledger. These warnings are transparency flags that KEEP the result solved and never downgrade it (anti-oscillation, BIBLE P5/P2); `_merge_objective_warning` accumulates them in `objective.warnings` so they co-exist with the cosmetic warning instead of clobbering it. `contract_kind` is agent-declared (the host never infers from prose whether a machine-checkable contract exists). The task-acceptance review's evidence dict carries the same receipts (R3), so a dig-direct (`/app`) verification the repo diff cannot capture is still judged. (v6.50.2) The receipt records an `expected_match` mode (substring default · exact · exact_line · json_equals · (v6.60.0) bytes_equal — after the check runs, `artifact_paths=[a, b]` are compared BYTE-FOR-BYTE on the same surface as the check (executor `cmp` in-container, host chunked read otherwise) with a bounded hexdump of the first divergence in the receipt: the golden-file/migration-parity shape a substring check silently under-verifies; (v6.61.1) BOTH operands are confined like every other artifact-path surface — `_confine_artifact_path` + the protected-artifacts `read_bytes` denial on each file, and in-executor operands must be workspace-RELATIVE (no absolute/`..`) — because the comparison is a byte-read oracle (sizes + divergence hexdump) that must not reach the control plane, black-box references, or hidden grader files; the mode is also rejected for non-run contract kinds instead of silently ignored) and a `matched` flag, and the acceptance reviewer now demands metric-grounded evidence — an existence-only or substring-only receipt is INSUFFICIENT when the task states a metric/worked example — under a public-info-only anti-cheat boundary (verify against instruction text / embedded examples / installed oracles / the agent's own independent checks, never a hidden /tests/, solution.sh, copied verifier, or an online answer); a one-shot no-op-attempt nudge (declared `expected_output` + zero tool calls + no reviewable effects + no FINAL ANSWER) fires alongside the verify nudge.

Rationale: logs are UI projections, not the source of truth. The private ledger preserves exact replay evidence locally while redacted projections keep operator-facing surfaces safe. Typed outcomes prevent benchmark adapters, CLI waiters, and the Web UI from treating non-empty error text as semantic success.

### Settings

Settings has Providers, Secrets, Models, Behavior, Advanced, and About. It handles provider keys, model routing, review settings, runtime mode, external skills repo, ClawHub registry URL, MCP servers, source control metadata, local model runtime, extension settings, timeouts, and reset. Rarely used provider cards (Cloud.ru, GigaChat) live under a collapsed "More providers" section that auto-opens when a usable provider credential is configured (a Cloud.ru key, a GigaChat OAuth credential, or a complete GigaChat basic-auth pair — base-URL/scope/TLS fields always carry shipped defaults and never count); the collapsed inputs stay mounted in the DOM so settings load/save applies to them unchanged. Hot-reload policy: total budget, timeouts, and GitHub metadata apply immediately; per-task cost threshold, models, API keys, effort, and review settings apply next task; local runtime, worker count, base URLs, provider runtime parameters, and runtime-mode changes require restart. Context mode hot-applies through the dedicated owner endpoint without a restart; lowering Max to Low is refused while queued, running, or direct-chat work exists. Runtime mode remains owner-controlled: ordinary `/api/settings` drops it, while `/api/owner/runtime-mode` persists the next-boot value without changing the current boot baseline.

## 4. Server API Endpoints

If `OUROBOROS_NETWORK_PASSWORD` is configured, non-loopback HTTP/WebSocket access requires authentication; `/api/health` stays public. With no password, non-loopback access remains open by explicit operator choice.

The executable route SSOT is `ouroboros/gateway/router.py`; file-browser routes come from `gateway/files.py::file_browser_routes()`, the contract index is `gateway/contracts.py::HTTP_ENDPOINTS`, and Host Service routes come from `gateway/host_service.py::create_host_service_app`.

File-browser symlink containment (v6.26.0): every `/api/files/*` endpoint
resolves the requested path and rejects it when the RESOLUTION leaves the
configured root (`Path escapes file browser root`). In-root symlinks keep
working; symlinks pointing outside the root are listed (with
`is_symlink: true`) but cannot be read, written, deleted, or traversed —
the old pass-through behavior was a root escape.

| Method | Path | Handler |
|---|---|---|
| GET | `/` | `server.index_page` |
| GET | `/api/health` | `gateway.state.api_health` |
| GET | `/api/state` | `gateway.state.api_state` |
| GET | `/api/extensions` | `gateway.extensions.api_extensions_index` |
| GET | `/api/extensions/{skill}/manifest` | `gateway.extensions.api_extension_manifest` |
| GET | `/api/extensions/{skill}/module/{entry}` | `gateway.extensions.api_extension_module` |
| GET | `/api/extensions/{skill}/settings_section` | `gateway.extensions.api_extension_settings_section` |
| ANY | `/api/extensions/{skill}/{rest:path}` | `gateway.extensions.api_extension_dispatch` |
| GET | `/api/skills/daemons` | `gateway.extensions.api_skill_daemons` |
| POST | `/api/skills/{skill}/toggle` | `gateway.extensions.api_skill_toggle` |
| POST | `/api/skills/{skill}/delete` | `gateway.extensions.api_skill_delete` |
| GET | `/api/skills/lifecycle-queue` | `gateway.extensions.api_skill_lifecycle_queue` |
| POST | `/api/skills/{skill}/review` | `gateway.extensions.api_skill_review` |
| POST | `/api/owner/skills/{skill}/attest-review` | `gateway.extensions.api_owner_skill_attest_review` (C1, v6.39; v6.43 official-hub extension: OWNER-ONLY — skip the expensive LLM review for the owner's own external/self-authored skill or for a freshly hash-verified official OuroborosHub payload; the deterministic preflight floor still runs, 409 on failure; routes through `run_skill_review_lifecycle` for the post-pass deps/extension reconcile) |
| POST | `/api/skills/{skill}/grants` | `gateway.extensions.api_skill_grants` |
| POST | `/api/skills/{skill}/reconcile` | `gateway.extensions.api_skill_reconcile` |
| GET | `/api/marketplace/clawhub/search` | `gateway.marketplace.api_marketplace_search` |
| GET | `/api/marketplace/clawhub/installed` | `gateway.marketplace.api_marketplace_installed` |
| GET | `/api/marketplace/clawhub/info/{slug:path}` | `gateway.marketplace.api_marketplace_info` |
| GET | `/api/marketplace/clawhub/preview/{slug:path}` | `gateway.marketplace.api_marketplace_preview` |
| POST | `/api/marketplace/clawhub/install` | `gateway.marketplace.api_marketplace_install` |
| POST | `/api/marketplace/clawhub/update/{name}` | `gateway.marketplace.api_marketplace_update` |
| POST | `/api/marketplace/clawhub/uninstall/{name}` | `gateway.marketplace.api_marketplace_uninstall` |
| GET | `/api/marketplace/ouroboroshub/catalog` | `gateway.marketplace.api_ouroboroshub_catalog` |
| GET | `/api/marketplace/ouroboroshub/installed` | `gateway.marketplace.api_ouroboroshub_installed` |
| GET | `/api/marketplace/ouroboroshub/preview/{slug:path}` | `gateway.marketplace.api_ouroboroshub_preview` |
| POST | `/api/marketplace/ouroboroshub/install` | `gateway.marketplace.api_ouroboroshub_install` |
| POST | `/api/marketplace/ouroboroshub/update/{name}` | `gateway.marketplace.api_ouroboroshub_update` |
| POST | `/api/marketplace/ouroboroshub/uninstall/{name}` | `gateway.marketplace.api_ouroboroshub_uninstall` |
| GET | `/api/files/list` | `gateway.files.api_files_list` |
| GET | `/api/files/read` | `gateway.files.api_files_read` |
| GET | `/api/files/content` | `gateway.files.api_files_content` |
| GET | `/api/files/download` | `gateway.files.api_files_download` |
| POST | `/api/files/upload` | `gateway.files.api_files_upload` |
| POST | `/api/files/mkdir` | `gateway.files.api_files_mkdir` |
| POST | `/api/files/write` | `gateway.files.api_files_write` |
| POST | `/api/files/delete` | `gateway.files.api_files_delete` |
| POST | `/api/files/transfer` | `gateway.files.api_files_transfer` |
| GET | `/api/onboarding` | `gateway.settings.api_onboarding` |
| GET | `/api/claude-code/status` | `gateway.settings.api_claude_code_status` |
| POST | `/api/claude-code/install` | `gateway.settings.api_claude_code_install` |
| GET | `/api/settings` | `gateway.settings.api_settings_get` |
| POST | `/api/settings` | `gateway.settings.api_settings_post` |
| POST | `/api/owner/runtime-mode` | `gateway.settings.api_owner_runtime_mode` |
| POST | `/api/owner/auto-grant` | `gateway.settings.api_owner_auto_grant` |
| POST | `/api/owner/context-mode` | `gateway.settings.api_owner_context_mode` |
| POST | `/api/owner/scope-review-floor` | `gateway.settings.api_owner_scope_review_floor` (DEPRECATED and ENFORCEMENT-INERT since v6.80.0; still mounted, still stores and audits — see below) |
| POST | `/api/owner/safety-mode` | `gateway.settings.api_owner_safety_mode` |
| POST | `/api/owner/capability-ack` | `gateway.settings.api_acknowledge_capability` |
| GET | `/api/ui/preferences` | `gateway.ui_preferences.api_ui_preferences_get` |
| POST | `/api/ui/preferences` | `gateway.ui_preferences.api_ui_preferences_post` |
| GET | `/api/model-catalog` | `gateway.models.api_model_catalog` |
| POST | `/api/openai-compatible/models` | `gateway.models.api_openai_compatible_models` |
| POST | `/api/tasks` | `gateway.tasks.api_tasks_create` |
| GET | `/api/tasks` | `gateway.tasks.api_tasks_list` |
| GET | `/api/tasks/{task_id}` | `gateway.tasks.api_task_get` |
| GET | `/api/tasks/{task_id}/events` | `gateway.tasks.api_task_events` |
| GET | `/api/tasks/{task_id}/artifacts/{name}` | `gateway.tasks.api_task_artifact` |
| POST | `/api/tasks/{task_id}/cancel` | `gateway.tasks.api_task_cancel` |
| POST | `/api/tasks/{task_id}/resume` | `gateway.tasks.api_task_resume` |
| GET | `/api/schedules` | `gateway.schedules.api_schedules_list` |
| POST | `/api/schedules` | `gateway.schedules.api_schedules_upsert` |
| DELETE | `/api/schedules/{schedule_id}` | `gateway.schedules.api_schedules_delete` |
| POST | `/api/command` | `gateway.control.api_command` |
| POST | `/api/reset` | `gateway.control.api_reset` |
| GET | `/api/git/log` | `gateway.control.api_git_log` |
| POST | `/api/git/rollback` | `gateway.control.api_git_rollback` |
| POST | `/api/git/promote` | `gateway.control.api_git_promote` |
| GET | `/api/update/status` | `gateway.control.api_update_status` |
| POST | `/api/update/check` | `gateway.control.api_update_check` |
| POST | `/api/update/preflight` | `gateway.control.api_update_preflight` |
| POST | `/api/update/apply` | `gateway.control.api_update_apply` |
| GET | `/api/cost-breakdown` | `gateway.history.make_cost_breakdown_endpoint` |
| GET | `/api/evolution-data` | `gateway.control.api_evolution_data` |
| GET | `/api/projects` | `gateway.projects.api_projects_list` |
| POST | `/api/projects` | `gateway.projects.api_projects_create` |
| POST | `/api/projects/from-task` | `gateway.projects.api_project_from_task` |
| POST | `/api/projects/{project_id}/update` | `gateway.projects.api_project_update` |
| POST | `/api/projects/{project_id}/delete` | `gateway.projects.api_project_delete` |
| GET | `/api/fs/dirs` | `gateway.projects.api_fs_dirs` |
| GET | `/api/chat/history` | `gateway.history.make_chat_history_endpoint` |
| GET | `/api/logs/{name}` | `gateway.logs.api_logs_tail` |
| POST | `/api/chat/upload` | `gateway.files.api_chat_upload` |
| DELETE | `/api/chat/upload` | `gateway.files.api_chat_upload_delete` |
| POST | `/api/local-model/start` | `gateway.models.api_local_model_start` |
| POST | `/api/local-model/stop` | `gateway.models.api_local_model_stop` |
| GET | `/api/local-model/status` | `gateway.models.api_local_model_status` |
| POST | `/api/local-model/test` | `gateway.models.api_local_model_test` |
| POST | `/api/local-model/install-runtime` | `gateway.models.api_local_model_install_runtime` |
| GET | `/api/mcp/status` | `gateway.mcp.api_mcp_status` |
| POST | `/api/mcp/refresh` | `gateway.mcp.api_mcp_refresh` |
| POST | `/api/mcp/test` | `gateway.mcp.api_mcp_test` |
| WS | `/ws` | `gateway.ws.ws_endpoint` |
| STATIC | `/static/*` | `server.NoCacheStaticFiles` |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/identity` | `gateway.host_service._api_identity` |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/tools/schemas` | `gateway.host_service._api_tool_schemas` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/allocate-internal` | `gateway.host_service._api_allocate_internal` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/inject` | `gateway.host_service._api_chat_inject` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/ui/ws-message` | `gateway.host_service._api_ws_message` |
| WS | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/events` | `gateway.host_service._ws_events` |

Rationale: `server.py` should own process startup/lifespan/static mounting, while `gateway/*` owns browser-facing HTTP/WS contracts. This keeps UI and runtime coupling explicit and testable.

### WebSocket protocol

Browser messages and backend broadcasts use typed envelopes from `gateway/contracts.py`. Extension WS messages are namespaced by `extension_loader.extension_surface_name()` so skills cannot shadow built-in message types. Reviewed transport skills can inject chat/photo/typing through the loopback Host Service rather than bypassing the browser protocol.

Server-side broadcast hardening (v6.34.0, WS4): `gateway/ws.py::broadcast_ws` fans out to all connected clients concurrently (`asyncio.gather(..., return_exceptions=True)`) so one slow or half-open client cannot head-of-line-block delivery to the others; the heartbeat path stays critical and is never dropped. `gateway/history.py::make_chat_history_endpoint` offloads the synchronous `iter_jsonl_objects` chat + progress parsing for `/api/chat/history` onto `asyncio.to_thread` so a large history read does not stall the event loop.

Security/behavioral endpoint contracts not obvious from route names:
- `POST /api/tasks/{task_id}/cancel` (v6.82) accepts an OPTIONAL JSON body `{"cascade": true}`. An ABSENT body (the CLI posts `{}`) keeps the pre-v6.82 single-task cancel, which preserves live children for headless callers and answers from the same TYPED custody outcome as the cascade — a worker that refuses to die is a 503, not the 404 the old boolean produced. A body that is PRESENT but unparseable or not a JSON object is a 400, as is a non-boolean `cascade`. With `cascade: true` the endpoint pre-checks subtree liveness (preserving the 404 contract) and then runs the WHOLE teardown before answering, off the event loop; the custody model, its ordering guarantees and the completion-wins rule are described once in the Chat section above. A teardown that refuses or raises while the tree is still live answers 503 and is recorded durably in `logs/supervisor.jsonl` (`task_cancel_cascade_noop` / `task_cancel_cascade_error`); a natural-completion race is a graceful no-op.
- `POST /api/owner/runtime-mode` persists the next-boot owner runtime mode and returns `restart_required=true`; it does not mutate the current boot baseline or process env.
- `POST /api/owner/auto-grant` persists the owner auto-grant toggle outside generic `/api/settings`.
- `POST /api/owner/context-mode` persists and hot-applies the owner-selected context horizon (`low` or `max`) outside generic `/api/settings`; it updates `OUROBOROS_CONTEXT_MODE` in settings/env and records an owner action. The chat composer uses this endpoint for the immediate Low/Max toggle, Settings/Behavior routes through it on Save Settings, and headless owners can call it via `ouroboros settings context-mode low|max`. Lowering `max -> low` is accepted only while Ouroboros is idle. Agent self-lowering attempts are blocked structurally: `config.save_settings` always rejects `max -> low`, `_owner_write_settings` keeps the same guard by default, `ToolRegistry` rejects process/CLI attempts early, and `browser.py` blocks internal Playwright POSTs to `/api/owner/context-mode` so UI clicks/evaluate payloads cannot lower the agent's horizon. `prompts/SAFETY.md` classifies any remaining attempt as dangerous if it reaches the safety supervisor. Since v6.80.0 the context mode also decides whether the BIBLE P3 blocking scope review applies, so two readers are deliberately DISTINCT: `config.get_context_mode()` is the EFFECTIVE mode used for context sizing, and `config.get_owner_context_mode()` is the OWNER-SELECTED one, which is what the scope gate reads. They diverge in exactly one case — the friction-free Max→Low AUTO-DOWNGRADE that `POST /api/settings` performs when a model change lands on a route whose ≥1M window cannot be confirmed. That narrowing is SYSTEM-initiated on an AGENT-REACHABLE path (a plain `{"OUROBOROS_MODEL": …}` POST names neither the context key nor `settings.json`, so none of the self-lowering guards above can see it), so it is recorded as DERIVED state in `OUROBOROS_CONTEXT_MODE_AUTO_LOW` — merge-skipped in both directions (setting it would fake an owner narrowing; clearing it would turn a system downgrade into an owner-declared scope-review skip) and cleared only by an explicit owner selection through this endpoint. The flag is TRI-STATE and resolved FAIL-CLOSED: an ABSENT or unrecognised value is UNKNOWN, never an owner declaration, so a `settings.json` written before v6.80.0 and an isolated server whose env allowlist forwards the context mode alone both keep the gate ON (the three benchmark allowlists forward the flag alongside the mode). `/api/state` publishes the derived flag as `context_mode_auto_low`, because an owner looking at an auto-downgraded `Low` needs their re-selection of `Low` to still POST this endpoint (both owner controls short-circuit on an unchanged value otherwise, and the endpoint is idempotent and clears the flag). Context sizing narrows exactly as it did before v6.80.0, and scope review stays ON. Without that split, picking a cheap small-window model would switch an immune gate off for the agent's own commits, which is precisely what BIBLE P1/P3 forbid.
- `POST /api/owner/scope-review-floor` (v6.34.0, CW1) is the dedicated owner endpoint for `OUROBOROS_SCOPE_REVIEW_FLOOR`. Since v6.80.0 the key is DEPRECATED and ENFORCEMENT-INERT: the write is accepted, the value is STORED (an owner customization is never destroyed) and audited, and the response carries an explicit `deprecation_notice` naming the control that actually decides — but NOTHING in the runtime consults the value, and there is no floor getter at all. Whether the BIBLE P3 blocking scope review applies is decided solely by the owner-only `OUROBOROS_CONTEXT_MODE` read as the OWNER-SELECTED value (`max`: blocking ≥1M scope gate; `low`: whole-repository scope review declaredly not performed, with a typed `scope_review status="skipped_low_context_mode"` evidence row on every skipped commit). The frozen contract surface is deliberately kept intact by owner decision — the typed `OwnerScopeReviewFloorResponse`, the `HTTP_ENDPOINTS` entry, the route, the generic-`/api/settings` merge-skip, the web client method and JSDoc — as are all three self-lowering guards (`ToolRegistry._detect_scope_review_floor_self_lowering`, the `browser.py` route + evaluate-JS guards, the `prompts/SAFETY.md` rule), because the key remains an owner-only stored setting the agent must not author. That shell guard is now PRECISE by INVERTED POLARITY: naming the endpoint or the key in a settings context is blocked UNLESS the whole command line is demonstrably read-only inspection (`_is_pure_read_inspection` splits the line with the shared `shell_parse.shell_segments` lexer and matches every segment HEAD against an allowlist of read commands), so `grep OUROBOROS_SCOPE_REVIEW_FLOOR data/settings.json` — a pure read that the old `key AND settings.json` substring test blocked — is allowed while `python -c "httpx.request('POST', '…/api/owner/scope-review-floor', …)"` is not. The first shape of the precision fix listed HTTP write MARKERS instead, which failed OPEN: any spelling outside the list (`request('POST', …)`, `--data-raw`, `fetch({method:'POST'})`) named the owner endpoint and mutated the setting. No enumeration of "what a write looks like" is ever complete (BIBLE P5), so the enumeration is of what a READ looks like, where an unrecognised command head is refused rather than admitted. What v6.80.0 DID remove as a capability is the audited owner-opt-in DEGRADED ADVISORY scope review together with `OUROBOROS_SCOPE_REVIEW_DEGRADED` (CHECKLISTS #21); a stale stored degraded key is inert (every consumer iterates `SETTINGS_DEFAULTS`) and never fails settings validation. The replacement path for an install with no ≥1M reviewer is `low` context mode.
- `POST /api/owner/safety-mode` (v6.54.3) is the dedicated owner endpoint for the LLM-safety-supervisor coverage mode (`OUROBOROS_SAFETY_MODE` = `full` default | `light` | `off`). The mode gates ONLY the LLM supervisor layer in `safety.py::check_safety` — the deterministic registry sandbox, protected-path policy, and light-mode guards run in every mode (BIBLE P3: the LLM supervisor is a configurable layer, not the immune floor). `light` keeps the LLM check for POLICY_CHECK integration tools and waves POLICY_CHECK_CONDITIONAL shell/verify through the deterministic guards; `off` makes no LLM safety calls. Every waved-through check emits a durable `safety_mode_skip` event. Lowering coverage is owner-only on every channel, mirroring the context-mode/scope-floor pattern: the key is merge-skipped from generic `/api/settings`, `config.save_settings` ratchets any downward step (`_guard_safety_mode_lowering`), `ToolRegistry` blocks shell/CLI attempts (`_detect_safety_mode_self_lowering`), `browser.py` blocks the route POST and evaluate-JS shapes, and `prompts/SAFETY.md` classifies remaining attempts as dangerous.
- `POST /api/settings` (v6.80.0) additionally PROBES the configured review and scope-review slots' own routes (never the main lane's `USE_LOCAL_MAIN` routing) and returns any scope slot that does not confirm ≥1M as `review_capability_notices[]` using the SAME `needs_ack:{route, route_fp, evidence}` payload shape the Max-context gate already returns, so the existing owner-ack UI flow applies unchanged. Before this, the Max gate probed only the MAIN route, so a PINNED scope reviewer had no path to `confirmed` evidence at all and silently ran in the conservative sub-floor window; `scope_review._scope_reviewer_window_evidence` additionally performs ONE lazy metadata-only (never generative, never billed) probe per process for an off-default pin. Both surfaces are ROUTE-aware, not model-aware: the lazy probe memoises by the full `capability_evidence.route_fingerprint` and the save-time notice fires on any route-affecting change (`_REVIEW_ROUTE_BASE_URL_KEYS` — `OPENAI_BASE_URL`, `OPENAI_COMPATIBLE_BASE_URL`, `CLOUDRU_FOUNDATION_MODELS_BASE_URL`, `GIGACHAT_BASE_URL`) as well as a slot change, and the slot itself is read from the CANDIDATE settings rather than process env. Keyed by model name alone, a hot base-URL change produced an unprobed route with no notice: the next scope review fell silently to the conservative sub-floor and the advertised owner-ack path was unreachable. Save also warns loudly about review model ids the provider catalog does not list (a truncated slot value such as `-5` used to surface only as three waves of `400 ... is not a valid model ID`, destroying the review quorum); nothing is rejected or rewritten, and fail-closed-on-absent-evidence is unchanged — a pin is routing intent, never an owner-ack.
- `POST /api/owner/capability-ack` (v6.33.0) records a route-fingerprinted owner acknowledgement that a model's route supports a given context window, stored as `asserted` Capability Evidence in `data/state/capability_evidence.json`. It is the owner escape hatch for the max-mode ≥1M gate when no provider metadata confirms the window; the ack is scoped to the exact route (provider/model/base_url/options) so it never leaks to a different route.
- `GET/POST /api/ui/preferences` stores owner-local, non-secret UI preferences in `data/state/ui_preferences.json`. It persists widget card order, nested-subagent expansion preference, sidebar/project-panel widths, and monotonic per-Project paint ACKs (`project_seen_revision`); the server clamps each ACK to the current durable visible revision and never moves it backwards. Legacy `project_last_viewed`/`project_hidden` inputs are accepted for one minor as loud no-ops. UI preferences remain separate from runtime settings and skill manifests.
- `POST /api/skills/{skill}/grants` is a dedicated owner grant path for manifest-declared keys and host permissions. It requires a fresh executable review under the current enforcement mode, content-hash-bound grant state, and script/extension skill type; desktop may still use the native bridge first, while web uses this endpoint after UI confirmation.
- `POST /api/skills/{skill}/delete` is limited to direct `data/skills/external/<name>` payloads. It unloads live extension surfaces, removes the local payload, and removes `data/state/skills/<name>`; marketplace skills keep using their hub-specific uninstall endpoints.
- `GET /api/update/status` is passive/read-only. It must not fetch, rewrite remotes, or mutate `.git`; explicit update checks/apply flows own network/git mutation.
- `POST /api/update/preflight` (v6.41.0) does NOT mutate the live worktree/branch/index, but it is NOT side-effect-free: `supervisor/update_merge.plan_managed_update_merge` runs `git fetch`, builds a local-snapshot commit (HEAD + tracked-dirty + untracked, `.gitignore`-respected) in a TEMP index, and runs a REAL 3-way merge against the managed target in an ISOLATED temp worktree (created + removed under `.git/worktrees`, leaving only dangling objects), then `supervisor/update_merge_policy.classify_conflicts` labels the result `clean` / `doc_reconcile` / `conflicting` (release docs auto-reconcilable EXCEPT `BIBLE.md` / `docs/CHECKLISTS.md` / `prompts/SAFETY.md`). `POST /api/update/apply {strategy}` stages it: `auto_merge` is taken ONLY when the working tree is clean (no uncommitted dirty/untracked work) and the merge is clean — it lands the merge behind a FAIL-CLOSED `data/locks/managed_update.lock` (re-plan after `kill_workers`, rescue snapshot, a transactional `.git/ouroboros-update-tx.json` marker, apply, a pre-restart smoke, then rollback-to-`pre_update_sha` on failure — never `origin`). A CLEAN managed-update auto_merge is an OWNER-GATED, review-exempt class (it fast-commits only already-reviewed committed history + the already-reviewed official release), parallel to `replace`/`checkout_and_reset`/rollback (P9) — NOT a weakened self-modification gate. Any UNCOMMITTED local work, code conflicts, or doc conflicts route to the AUTOMATED `assisted` flow: the supervisor stages a REAL `git merge --no-commit` into the LIVE worktree (`update_merge.materialize_assisted_merge_live` — MERGE_HEAD + conflict markers over the local-snapshot base) under the lock with a durable rescue-local ref, writes a 4-phase tx (`materializing_assisted` → `assisted_resolution` → `committing_assisted` → `pending_boot_smoke`), and enqueues ONE authorized resolution task. The agent resolves the markers with normal file tools (no blocked git) and the UNMODIFIED `commit_reviewed` lands a reviewed 2-parent merge commit through triad/scope (native MERGE_HEAD; a tx-keyed managed path in `_repo_commit_push` enforces exclusivity + a conflict-marker leakage gate + suppressed push/tag + an inline pre-restart smoke). A central write-exclusivity guard (`registry._managed_update_code_tool_block`) blocks every OTHER task's repo-mutating tools while the merge is staged; an orphaned-resolution watchdog (`abort_orphaned_assisted_tx`) and a non-destructive, merge-state-keyed boot recovery free or resume the update across restarts without ever resetting over a reviewed worker commit. Official changes to PROTECTED paths (BIBLE/CHECKLISTS/SAFETY + release invariants) route to `manual` (never the agent); `manual` returns the plan; `replace` is the legacy hard-reset escape hatch. `server.py` boot finalizes a pending update (`update_merge.finalize_managed_update_on_boot` — post-boot smoke + boot-loop-guarded rollback) and runs a one-shot update check (check-on-restart only; no periodic poll); the main-screen Update pill (`web/modules/update_status.js`) surfaces availability + the staged dialog. The merge engine and per-path policy live in `supervisor/update_merge.py` + `supervisor/update_merge_policy.py` (kept out of `git_ops.py` for module-size discipline; referenced via the `git_ops` module so monkeypatched globals follow).
- Dashboard **Activity** subtab (`web/modules/activity.js`, v6.41.0): cron schedules + the running/pending queue + background-consciousness status with direct cancel / enable-disable controls (skill-manifest schedules are read-only "managed by skill"). Since v6.82 the Activity cancel buttons use the shared `api_client.cancelTask(id, {cascade:true})` helper — the same declared subtree semantics as the chat card's "Cancel run", so cancelling an orchestrator never orphans its live subagents. Subagent/research live-card lines carry a `{truncated, full_ref}` contract (`supervisor/events.py`) and expand INLINE to the genuinely-full output fetched on demand via `GET /api/tasks/{id}` into a bounded-scroll box (P3). Skill review binds the payload to ONE pack-level token budget (`ouroboros/skill_review._skill_pack_token_budget`) instead of per-file byte caps, splitting an over-budget skill into chunked passes whose per-model results are merged (`ouroboros/skill_review_passes.run_skill_review_passes`); a read-only audit subagent can name `write_surface='read_only'` (P5).

## 5. Supervisor Loop

Runs in a background thread inside `server.py:_run_supervisor()`.

Each iteration (0.5s sleep):
1. `rotate_chat_log_if_needed()` — archive chat.jsonl if > 800KB
2. `ensure_workers_healthy()` — respawn dead workers, detect crash storms
3. Drain event queue (worker→supervisor events via multiprocessing.Queue)
4. `enforce_task_timeouts()` — activity-based stop (v6.38.0): a task is stopped only when it makes no REAL progress (`llm_usage`/progress events, NOT the 30s liveness heartbeat) AND has no progressing/queued subtree, beyond `OUROBOROS_TASK_IDLE_TIMEOUT_SEC` (floored to the per-call ceiling); the only HARD axes are an explicit `deadline_at`, `OUROBOROS_TASK_ABS_CEILING_SEC`, and budget. The heavy teardown (kill/join/archive/respawn) is handed OFF the loop to a single-owner background reaper (`supervisor/task_reaper.py`); the slot is marked `reaping` under `_queue_lock` (assign + crash-detector skip it) and the terminal write + retry + respawn happen only after the worker is PROVABLY dead (v6.38.1 strict fail-closed) — if it will not confirm dead, the reaper does nothing downstream, holds the slot `reaping`, leaves the task RUNNING (no terminal/respawn while it may be alive), and emits `task_reaper_wedged` + an owner /restart hint, with the orphan custody-reaped on the next generation — so a still-alive worker is never raced
5. Periodic custody reap (every 600s) and periodic zombie reconcile (every 300s,
   `server.py::_periodic_zombie_reconcile`): heals `review_job.json` files and
   `task_results/<id>.json` records stuck at `running` after a worker died
   mid-flight (crash/SIGKILL/manual stop). Both reconciles are liveness-gated
   (pid-dead / queue-snapshot-present + task-absent + worker-boot-after-task
   evidence + grace), so a live review or task is never touched; the same
   reconciles also run once at server startup (lifespan).
6. `enqueue_evolution_task_if_needed()` — auto-queue evolution if enabled
7. `assign_tasks()` — match pending tasks to free workers
8. `persist_queue_snapshot()` — save queue state for crash recovery
9. Poll `LocalChatBridge` inbox for user messages
10. Route messages: slash commands → supervisor handlers; text → agent

Task heartbeats remain internal liveness facts: workers update queue heartbeat, lag/idle tracking, deadlines, the absolute ceiling, reaper, finalization, planning waits, and Activity/live-card freshness. The former periodic owner-chat line (`Task … running for … heartbeat_lag … idle … Continuing`) is not logged or rendered as a user bubble and does not advance Project unread. Only a real incident—deadline/idle/ceiling action, lost/wedged worker, cancellation/finalization fault, or reaper action—enters owner presentation, as an Activity/live-card event plus one deduplicated toast. The historical Soft/Hard UI controls are removed; non-default legacy values for their environment keys are accepted for one minor as loud deprecated no-ops.

Chat-lane wedge resilience (v6.34.0, WS3): bridge intake (`_process_bridge_updates`) is hoisted EARLIER in the iteration so a slow later step (timeouts, reconcile, evolution) cannot starve new-message intake, and the per-iteration housekeeping that does not need to gate intake is grouped into `server.py::_periodic_supervisor_maintenance`. Each iteration stamps a `_loop_liveness` heartbeat.

### Mutation attribution (v6.66.0)

Physical exclusion is NOT claimed: parallel writers remain possible exactly as
before. What v6.66.0 adds is honest evidence and honest staging. When a queued
ROOT task starts, the host captures a `system_repo` baseline in the existing
task result (`mutation_evidence`): exact Git commit/tree plus the pre-existing
dirty paths with fingerprints. When that root task reaches outcome derivation,
a terminal candidate snapshot recomputes the observed-window delta, including
task commits already between the baseline commit and current HEAD. A candidate
is the clean-at-baseline delta of the observed window; pre-existing dirty paths
that changed, a stale/missing baseline, or a failed scan surface as typed
blockers.

Blockers are evidence, never a verdict: the projection
(`mutation_attribution.load_mutation_evidence_projection`) rides into
acceptance/review evidence and the loop-outcome failure evidence for the LLM
panels to weigh; nothing structurally downgrades an `ok` outcome.

`commit_reviewed(paths=None)` stages only the attributed candidate set when a
baseline exists for the task lineage; explicit paths must be a subset, and an
empty set returns `GIT_NO_ATTRIBUTED_CHANGES` — never whole-tree staging.
Contexts without a captured baseline (manual ToolContext, external dry-run
review) keep the legacy explicit/whole-tree staging contract, and managed
release/update transactions retain their separately typed `git add -A`
authority.

`scripts/run_external_review.py` has two explicit non-committing profiles. Its
default operator profile resolves live production policy, freezes the staged
tree in a detached checkout, and runs real advisory followed by triad/scope.
Its `external_pr_readiness` contributor profile binds the current target-base
SHA, proposal HEAD/tree and diff hash; extracts literal shipped review defaults
from the target checkout without executing PR code; forces those models through
OpenRouter at blocking enforcement; skips Claude advisory; and produces a
redacted shareable packet. Release metadata is a typed final-landing obligation,
so contributor evidence cannot authorize a canonical commit. If the proposal
changes the review substrate, the packet is diagnostic and requires a trusted
maintainer rerun. Supervisor-owned physical mutation
leases (surface holders, custody-confirmed release) were deliberately NOT
shipped in v6.66.0; if that concurrency layer is ever needed it is a separate
reviewed design on top of this evidence layer.

### Supervisor liveness watchdog (v6.34.0, WS3)

`server.py::_start_supervisor_liveness_watchdog` runs on a DEDICATED thread (not inside the serial supervisor loop, so it still fires if that loop stalls). It watches TWO silent-wedge classes against `OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC` (default 90s, comfortably above the ~0.5s healthy tick / 30s heartbeat cadence) and surfaces each to the owner via `get_bridge()`:
1. **Supervisor loop stall** — the loop's `_loop_liveness` tick goes silent (new-message intake starvation). WS10 ephemeral decision turns still answer meanwhile.
2. **In-process direct-chat turn wedge** — the chat agent is `_busy` but its `_last_activity_ts` (stamped at turn start + by the 30s heartbeat loop, INDEPENDENT of the event queue) goes silent. The direct turn is NOT a worker-queue `RUNNING` entry, so the watchdog reads the chat-agent state directly (`supervisor/workers.py::chat_turn_liveness`, which never takes `_chat_agent_lock`).

The watchdog deliberately does NOT kill the hung thread or free the chat-agent lock: a wedged direct turn holds `_chat_agent_lock` for its whole duration, and a `threading.Lock` released by a non-owner thread would crash when the zombie turn finally exits its `with` block — so true in-process admission-freeing is unsafe. Out-of-process direct chat for full kill-ability was deferred per owner; the safe full recovery is `/restart`, which the owner alert recommends. WS10 keeps the chat responsive (new messages run as ephemeral decision turns) while a turn is wedged, so the wedge is observable + recoverable rather than a silent outage.

### Worker crash handling and retry limits

When a worker process dies unexpectedly (e.g. SIGSEGV, signal -11) while
running a task, `ensure_workers_healthy()` in `supervisor/workers.py` performs
a three-way decision before requeueing:

1. **Already-completed check**: calls `load_task_result()` — if the task already
   reached a terminal state (e.g. completed via direct-chat inline path), the
   crash is silently skipped and the task is NOT requeued. Prevents duplicate execution.

2. **Retry limit exhausted** (`task["_attempt"] > QUEUE_MAX_RETRIES`): marks
   the task as `STATUS_FAILED`, emits a `task_done` event to close the chat UI
   live card, and sends an assistant message via `get_bridge()`. No requeue.

3. **Normal retry**: increments `task["_attempt"]` on a dict copy BEFORE requeue.
   The task is written with `STATUS_INTERRUPTED` and pushed to the front of the queue.

**Crash storm detection**: `respawn_worker()` no longer resets `_LAST_SPAWN_TIME`
(only `spawn_workers()` sets it at initial startup). This allows `CRASH_TS` to
accumulate 3 timestamps within 60 seconds during rapid crash loops, triggering
storm detection which kills all workers and switches to direct-chat mode.

**`deep_self_review` tasks** are exempt from the normal retry path — they fail
immediately on a crash signal (SIGSEGV) with a diagnostic message suggesting
`/restart` followed by `/review`.

### Slash command handling (server.py main loop)

| Command | Action |
|---------|--------|
| `/panic` | Kill workers (force), request restart exit |
| `/restart` | Run `safe_restart` (git/deps/import preflight); on success, write `owner_restart_no_resume.flag` plus a stable-compatible skip marker, cancel active worker tasks with owner-restart result text, tell the owner the active task is stopping, exit 42 |
| `/review` | Queue a deep self-review (1M-context single-pass Constitution review) |
| `/evolve on\|off` | Toggle evolution mode in state, prune evolution tasks if off |
| `/bg start\|stop\|status` | Control background consciousness |
| `/status` | Send status text with budget breakdown |
| (anything else) | Route to agent via `handle_chat_direct()` |

---

## 6. Agent Core

### Task lifecycle

A user message enters `server.py`, is routed through supervisor queue/workers, and runs inside `OuroborosAgent`. The task pipeline builds context, runs the LLM/tool loop, stores task results, emits progress/events, reflects, consolidates memory, and records review evidence.

Host-enforced task acceptance is root-owned and structurally gated, not turn-counted or keyword-gated. `off` disables it. In `auto` and `required`, queued/headless/scheduled substantive roots are reviewed. Direct chat is reviewed after an observable reviewable effect or a typed deliverable/criterion; ordinary read-only tool activity, pure conversation, and meta/routing-control turns are skipped. Reviewable effects are successful commits, non-scratch writes/edits, `claude_code_edit`, declared process outputs, integrated child patches, or a registered canonical artifact; cognitive-memory updates are not effects. Child tool reviews remain advisory evidence and are superseded by the root verdict rather than becoming a second authority.

Before the root panel runs, `supervisor/task_lifecycle.py` closes subtask admission atomically under the queue lock and `loop.py` proves the recursive subtree terminal/quiescent through `task_status.find_child_tasks`; revision reopens the fence, while terminal/degraded completion seals it. The immutable evidence core includes verbatim owner directives, the full task contract/criteria, canonical deliverable identity and aliases, terminal subtree statuses, verification receipts, artifact refs, provenance, and an explicit omissions manifest. If that core cannot fit, the affected actor returns `DEGRADED`; critical requirements are never silently truncated.

The configured task-acceptance panel uses independent reviewer slots and `adaptive_quorum` (1→1, 2→2, 3+→2). For an unchanged candidate hash, evidence revision, and fence token/state, the host runs that authoritative panel exactly once and reuses its recorded result. Each actor gets one substantive call and at most two physical sends total (same-route transport retry or extraction/format repair); a third send is blocked. Actor truth keeps transport (`success | timeout | provider/transport error`), parse (`valid | malformed`), and the semantic verdict of a valid response separate, alongside model/provider, role, coverage, quorum contribution, reason, enforcement impact, panel id, and binding hashes. A task-acceptance FAIL contributes only when it carries the required outcome tier plus a bounded correction rail; a bare veto stays auditable but abstains. DEGRADED abstains from quorum and never creates an authoritative obligation — the reviewer verdict vocabulary `PASS|FAIL|DEGRADED` is unchanged (v6.78.0 collapsed only the HOST decision statuses; the deliberate-DEGRADED capsule rail, the dialogue vote, and the host's own core-overflow DEGRADED all keep working). The HOST acceptance decision is written in exactly ONE place (`loop._set_acceptance_decision`) and has exactly three owner-facing states — `accepted | revision_requested | finalized_unaccepted` — each with a typed `reason` (`clean_pass`, `clean_pass_obligations_closed`, `no_actionable_changes`, `delivery_binding_superseded`, `owner_followup`, `evidence_refresh`, `improvement_capsule`, `dialogue_terminal`, `open_obligations`, `capsule_spent`, `improvement_window_closed`, `reviewer_fail_no_capsule`, `review_degraded`, `fence_reopen_failed`, `infra_failure`, `review_skipped_deadline_reserve`); an unknown status fails closed to `finalized_unaccepted` carrying its raw token as the reason. `accepted` is authorised by clean acceptance ALONE: every terminal branch reached after `review_substrate.task_acceptance_is_clean` has refused the panel writes `finalized_unaccepted` with its own typed reason (v6.78.0 — a non-clean PASS with nothing actionable left to feed back is `no_actionable_changes`, not `accepted`; the tier honesty keeps riding `outcome_tier`). The agent may record only its own `agent_disposition`/`agent_rationale` (merged, never overwriting the host verdict). The adjacent `acceptance_binding.acceptance_status`, `review_decision.eligibility`, obligation-row statuses and the audit-only `root_phase_checkpoint.status` are DIFFERENT vocabularies and did not collapse; historical task results keep the pre-v6.78.0 tokens (the projection is a passthrough, no normalizer). A deliberately parsed semantic DEGRADED with a concrete recommendation may still supply the advisory correction capsule for the Required+Blocking re-drive, while transport/unparseable no-quorum becomes a terminal host decision (v6.78.0: `finalized_unaccepted` with `reason=review_degraded`) after its bounded retry and never steers revision. Clean means quorum PASS, `solved`, and supported evidence for every contributing criterion. Actionable gaps are exact-deduplicated into the existing obligation/improvement path. (v6.71.1, convergence) An honestly-marked non-`solved` criterion (`partial`/`missing`/`rejected` at `best_effort`/`blocked_with_evidence`) is a VALID contributing vote — only a `solved` claim still requires every criterion `supported` with refs — so an honest partial no longer collapses to `malformed` and starves quorum; the reviewer receives the host-attested `acceptance_obligations` catalog and adjudicates the agent's per-obligation dispositions as rebuttals (a genuinely valid `rejected` argument retires the finding, an invalid one is re-raised with an explanation, and a rebuttal is NEVER itself evidence for a criterion — `solved` still requires an independent host/tool/artifact receipt); and an acceptance improvement pass is an ORDINARY substantive answer round (delivery-control is not armed on the acceptance path, so the finalize-JSON / open-obligations / periodic self-check directives no longer coexist and freeze the model into identical no-tool resubmits). Blocking stays blocking: the loop terminates only by reviewer agreement — a clean PASS, an accepted rebuttal, or (v6.74.0) a reviewer-quorum judgement that the dialogue is `unreachable_here`/`stable_disagreement` — or a real deadline/budget/lifecycle rail; never a unilateral agent give-up and never a host timer. (v6.74.0, acceptance dialogue) Each acceptance reviewer emits a typed `dialogue_status` (`continue_actionable` | `unreachable_here` | `stable_disagreement`); `review_substrate.aggregate_dialogue_status` is a pure reducer over ALL contract-valid actors (deliberately wider than the aggregate-filtered contributing set, so a DEGRADED slot's deliberate terminal vote counts) applying the panel's own quorum with explicit precedence — any contributing `continue_actionable` keeps the loop; a quorum of terminal votes finalizes through the existing honest path (v6.78.0: `finalized_unaccepted` with `reason=dialogue_terminal`, the open-obligation ids riding the decision) with both positions recorded and the full vote distribution persisted on the run record (`dialogue`) for audit; a missing/invalid vote defaults to `continue_actionable` (fail-safe, backward-compatible). Obligation identity is reviewer-authored: findings carry `disposition_kind` (`new`/`re_raise`); a `re_raise` MUST name an existing obligation id from the host-attested catalog and fails closed to `new` (disclosed note) on an unknown id, so a reworded re-raise cannot silently mint a fresh hash id; a re-raise REOPENS the row keeping `previous_disposition`/`previous_reason`/`reopened_count` plus the reviewer's stated reason, the obligations clause shows the agent its rebuttal was overruled, and the evidence catalog gives the next reviewer the prior argument to adjudicate (valid → retire the finding; invalid → maintain it and say why the argument fails). The improvement capsule leads with the actual verdict + tier + real blocker (one `panel_reason` reducer shared by capsule/projection/progress — the single-reviewer diversity note is an orthogonal label, not a degraded_reason), lists the open obligation ids, carries one pre-rendered rails line (money/time/rounds/review-passes headroom, assembled in `loop.py` from the ledger projection, `BudgetSnapshot`, the loop round counter, and the pacing pass cap; v6.74.4 adds a FINAL-pass marker when the launched pass is the last the pacing cap admits, and — for workspace deliveries via the canonical `is_workspace_mode()` authority — an always-on tree directive: keep the tree VERIFIED, rebuild/verify/commit if the task calls for a commit, revert unverified edits, because the workspace tree ships as-is on any forced end), and names the three real moves (fix / rebut via `obligation_dispositions` / declare unreachable) instead of the old do-nothing tail. Required+Advisory may finalize with an honest non-clean verdict; Required+Blocking keeps iterating until clean or a real deadline/budget/lifecycle rail, subject to any explicit `max_improvement_passes`. The first review reserves at least 200 seconds; later passes use `max(configured floor, 1.5×timing EWMA)` with `alpha=0.5` reconstructed from existing events. Lifecycle remains `completed` when an artifact exists; objective/review status and stop reason carry the warning. Only the root performs global post-task synthesis, once, and `root_phase_checkpoint` provides the minimal restart seam without a parallel admission journal: startup replays only `pending_once`; an indeterminate persisted `running` phase is terminally disclosed as degraded instead of risking a second paid sequence.

For an eligible root in `task_review_mode=auto|required`, agent-callable `task_acceptance_review` is evidence-only: it validates and stores claims, checklist items, evidence refs, and optional agent disposition, makes no reviewer-model call, and returns `status=deferred_to_host_acceptance`, `authoritative=false`, plus the evidence revision. Calling it does not widen structural eligibility. `off` mode and child-task review retain their existing behavior. Task acceptance and the P3 commit gate intentionally remain different systems.

In `runtime_mode=light`, generic writes to cognitive memory and absolute home paths are redirected, not just blocked: `tool_access.light_cognitive_or_root_redirect` returns `COGNITIVE_TOOL_REQUIRED` (use `update_identity`/`update_scratchpad`/`knowledge_write`) for `runtime_data` writes under `memory/{identity,scratchpad,knowledge}`, and `ROOT_REQUIRED_USER_FILES` for absolute home paths written with the default `active_workspace` root (an explicit non-`user_files` root still falls through to the generic block). The two statuses differ by intent: `COGNITIVE_TOOL_REQUIRED` is **advisory** — the agent sees the redirect and should use the cognitive tool, but a self-initiated cognitive write never fails the task (`outcomes._unresolved_tool_errors` skips it). `ROOT_REQUIRED_USER_FILES` is a real user deliverable and stays **blocking**, recovered only when every originally blocked filename (from `path` and `files[]`) is later written via `root=user_files`, so a corrected retry is not falsely failed while an ignored one still surfaces.

### Tool capability and execution

`tool_capabilities.py` is the SSOT for core tools, meta-tools, parallel-safe tools, stateful browser tools, untruncated tool results, per-tool result caps, and reviewed mutative tools. `tool_policy.py` decides round-one visibility. `loop_tool_execution.py` handles timeouts, thread pools, live logs, truncation, metadata, and reviewed mutative hard ceilings.

Rationale: tool classification drift caused subtle bugs; every hardcoded set now has one canonical home. Review outputs and cognitive artifacts are exempt from generic truncation because they are process memory, not transport noise.

#### Web access mechanisms (three distinct paths — do not conflate)

The agent can reach the web through three mechanisms with different model and observability consequences. Confusing them is a recurring source of methodology error (e.g. treating the weak `ddgs` backend as parity with a natively-searching harness):

1. **Main-loop native search** (`OUROBOROS_MAIN_WEB_SEARCH=openrouter`): injects OpenRouter's `openrouter:web_search` server tool directly into the **main solve-model** request (`llm.py` `_openrouter_main_web_search_tool`, attached only when `allow_server_web_search` is set — the main loop, never a reviewer). The SAME solve model decides when to search; **no second LLM enters the scaffold**. Fetched citations are harvested into `usage["web_search_sources"]` (`{url,title,content}`, capped 20) and persisted on the `llm_usage` row in `events.jsonl`; the search *query* is provider-side and not logged. (v6.78.0) A SECOND consumer now reads that key: `loop_llm_call.fold_retrieval_usage` folds it (with `usage["server_tool_use"].web_search_requests`) into `accumulated_usage["retrieval"]`, which `loop.py` mirrors into `llm_trace["retrieval"]` and `review_evidence.build_task_acceptance_evidence` exposes to the ACCEPTANCE REVIEWER as a host-attested `retrieval` fact — counts plus at most 20 URLs of at most 200 chars, no titles and no snippets. The reviewer rules frame it as factual context, never a criterion (an absent fact means only that no NATIVE search was recorded — main-loop native search is off by default and the `web_search`/browser tools issue their own calls, which never reach the answering call's usage — and must not be read as a gap), and the AGENT never sees it: it receives only the improvement capsule, which carries no evidence sections. Engine is `OUROBOROS_MAIN_WEB_SEARCH_ENGINE` (`auto`/`native`/`exa`/…).
2. **`web_search` function tool** (`ToolRegistry`): backends in order official OpenAI Responses → `openrouter:web_search` → Anthropic `web_search_20250305` → `ddgs`. The first three issue a **separate provider call whose model is `OUROBOROS_WEBSEARCH_MODEL`** (default `gpt-5.2`) — so this tool can introduce a second reasoning model unless pinned; `ddgs` is a keyless pure-retrieval scraper with no second LLM but markedly weaker results. Calls and results are logged to `tools.jsonl`.
3. **Browser tools** (`browse_page`/`browser_action`, Playwright): fetch arbitrary URLs locally; args and result previews logged to `tools.jsonl`.

Benchmark note: the GAIA harness's `quality_openrouter_web` profile uses mechanism (1) and disables mechanism (2), so retrieval flows through a single disclosed native path (see `devtools/benchmarks/gaia/METHODOLOGY.md`).

Context compaction policy remains owner-MODE-driven (v6.33.0; the static per-model window table was removed as a perpetual-staleness anti-pattern — 1M-beta models had been hard-coded to 200K). `context_budget.py` owns the thresholds and the owner `OUROBOROS_CONTEXT_MODE` is the global SSOT: **max** keeps remote models on emergency-only compaction at the ~1.2M-char ceiling (cache-friendly); **low** lowers the emergency threshold to ~400K chars and enables routine compaction after round 6 / >40 messages, matching the smaller ~200K/local horizon. For ordinary host-managed tasks, `context_fit.py` additionally projects that immutable core for the exact route using calibrated Atlas/Capability Evidence; an unknown route tries Max and is never silently assumed to be 200K. Its calibration baseline is DELIBERATELY narrower than the review-pack one (v6.80.0): only a MEASURED density for that exact model (plus exact-route `llm_round` observations) may raise it above 1.0, and the conservative cold-start density used for review packs is NEVER applied here — with an empty observation store (every fresh install and every isolated benchmark server) a cold cross-model density would silently demote `initial_mode` from Max to Low on the main loop. The cold baseline for an unmeasured route is therefore the neutral 1.0 — unknown routes try Max (BIBLE P1) — and the MEASURED density supersedes it from the first successful send of that model onward, so the old proactive protection returns as a measurement rather than a guess. Disclosed residual: the one exposed window is a FIRST round whose prompt already exceeds the calibrated window on a fresh evidence store; that round fails `infra_failed` with `context_overflow` (there is no rebuild inside `loop_llm_call`), and `loop.py` then reprojects the transcript into task-local Low exactly once — guarded by `_context_fit_low_retry_used`, after a forensic checkpoint — and retries the same model. The exposure does not recur for that model once its density is recorded. Manual pending compaction is always honored, and every manual/emergency/routine or fit-rebuild branch persists a forensic checkpoint before summarizing so a Low projection changes granularity without silent truncation.

`loop_llm_call.py` classifies provider context-window overflow separately from quota/auth/billing and other hard bad requests. A confirmed Max overflow on an ordinary task may rebuild the same immutable core once into a task-local Low projection and retry the same model; it does not change the owner's global mode. That transition records forensic evidence and an owner-visible Activity/card/toast incident. P3 commit/scope review is excluded from this task-local retry and retains its established one-pass fit/oversize policy; an already-Low ordinary task reports the failure without another downgrade. Other permanent classes are recorded in usage/error events and surfaced as recovery hints instead of consuming retries for an identical request.

LLM retry budgets are per failure class (v6.28.0). Transient provider failures — empty/incomplete responses (the `finish_reason=null` glitch and content-empty `llm_empty_response` shapes both retry under this budget) and `provider_transient` exceptions (429/5xx/overloaded) — retry the SAME model with a larger attempt budget (`transient_retry_max`, env `OUROBOROS_TRANSIENT_RETRY_MAX`, default 6, floored at the caller's budget) and exponential backoff capped at 60s, while permanent classes (auth, quota, bad request, request-too-large) keep failing fast at the base budget. Backoff sleeps are deadline-bounded: when the task deadline (`task_metadata.deadline_at`) cannot absorb the next sleep plus a useful follow-up attempt, the retry loop stops with a durable `llm_retry_deadline_exhausted` event — emitted by BOTH transient paths (the finish_reason=null/empty-response branch and the classified-exception branch) — instead of burning the remaining budget sleeping. No cross-model fallback is introduced by this policy — single-model setups (all slots one model, empty fallback) stay clean by design, and the final failure text reports the real attempts used. The OpenRouter strip-and-retry matcher also covers gpt-5-style "encrypted reasoning item"/"encrypted content for item rs_…" 400s, reusing the same one-shot reasoning-metadata strip as thought-signature errors.

Context compaction is failure-isolated (v6.28.0). `context_compaction.py` summarizes old rounds in batches with per-batch isolation: a failed batch leaves only its own rounds raw and its spend is still accounted (`_BatchSummaryError` carries usage); a round whose summary is missing degrades individually instead of failing the batch. The summarizer prefers a structured `emit_round_summaries` tool protocol (`tool_choice="required"`, reliable `round_id` keying) and falls back to the legacy `[round:N]` text protocol — automatically for local light models, or per-response when a model answers in prose. ⚠️-protection scans the first two non-empty lines of a tool result (shell autocorrect notes — with or without a blank separator line — can prefix the marker), and `⚠️ SHELL_EXIT_ERROR` rounds are deliberately compactable — failed-command trial-and-error history is exactly what must compact, with the summarizer instructed to keep the first error line verbatim. Emergency compaction in `loop.py` adapts `keep_recent` to `min(50, max(6, spans//2), max(1, spans-1))` — halve the history with floor 6, always clamped below the span count — so an oversized transcript with few huge rounds actually compacts instead of no-opping at the `len(spans) <= keep_recent` gate (a single round has nothing older to summarize).

Prompt caching and reasoning continuity are provider-gated in `llm.py`. Stable governance/policy is placed before dynamic evidence. Anthropic-compatible routes retain supported ephemeral message/tool cache markers (a caller-declared `ttl` survives normalization only as a provider-valid `"5m"`/`"1h"`, and the strongest TTL present is reported as `prompt_cache_ttl` in usage so extended-tier cache writes are priced correctly); direct OpenAI may additionally receive a stable-prefix `prompt_cache_key`, and OpenRouter a conversation-stable `session_id`. A caller may instead declare an explicit `cache_affinity` (`chat`/`chat_async` kwarg): review surfaces pass `{surface}:{task_id}` so repeat rounds with changing evidence keep one sticky-routing session (the default first-user-message key would fragment it every round); the key deliberately excludes reviewer slot ids so same-model slots keep the historical provider-concentration behavior, and Main's default identity is untouched. If the exact provider explicitly rejects that named cache parameter, the same request is retried once without it; other errors do not trigger this fallback. All four review prompt builders (triad, skill, scope, plan, plus the acceptance substrate default) assemble STABLE-FIRST and mark the byte-stable governance prefix with a block-level cache marker via `review_helpers.cached_prompt_blocks` (review TTL `1h` — rounds repeat past the 5-minute default); each builder reports its stable/dynamic boundary, and the skill-review output contract stays after the untrusted payload (anti-injection boundary). (v6.74.0) The acceptance substrate segments TWO cache-marked system blocks — the byte-stable governance instruction and the task-stable contract (goal/scope/checklist/policy, unchanged across a task's improvement passes) — followed by the unmarked mutable evidence tail; the `Slot: slot_N` label moved from byte 0 of the user message to its TAIL so concurrent same-model slots of one pass share a warm prefix, and `assert_cache_breakpoint_cap` (≤4) is asserted on the final serialized payload. The large evidence body changes every pass by design and is honestly not cached; the exact review binding is useless as a breakpoint (an unchanged binding reuses the recorded panel with no second call, a changed binding never prefix-matches), which is WHY only the governance+contract prefix is marked. Learned provider parameter rejections are durable: `capability_evidence.json` carries a `rejected_params` namespace (normalized model identity key, 14-day expiry) mirrored into the process cache, and `no_proxy` request builds consult an already-warm `supported_parameters` cache instead of ignoring it (skip means "skip the network fetch", not "forget what is known"). Direct/local payloads strip OpenRouter-only reasoning metadata, while OpenRouter retains its existing portable-family continuity and same-model provider-resilience rules — with one field-driven narrowing (v6.65.3): the transient body-error reroute preserves replayed reasoning only for Anthropic/Gemini; `openai/*` encrypted-reasoning items proved non-portable across OpenRouter sibling upstreams, so that reroute strips them as before v6.49.0, and an encrypted-reasoning 400 delivered as a body error inside HTTP-200 receives the same one-shot strip-and-retry as the exception path instead of a permanent bad_request. This narrow cache-affinity support does not introduce body rerouting, provider hopping, or an internal retry platform.

`LLMClient` enforces a system-message placement invariant at the provider boundary. One or more leading `system` messages remain authoritative system context; any later runtime notice is demoted to a `user` notice with a visible `[SYSTEM NOTICE]` marker. The normalizer buffers notices that appear between an assistant `tool_calls` message and its `tool` results, so tool-call adjacency is preserved. This makes strict OpenAI-compatible/local templates ("system message must be at the beginning") structurally unreachable without changing the authority or cache semantics of the leading system prompt.

`LLMClient` treats sampling controls such as `temperature`, `top_p`, and `top_k` as optional request intent, not as required semantic parameters. The request builder preserves required semantics (`reasoning`, prompt-cache markers, tools/tool choice, token budgets, and OpenRouter `provider.require_parameters`) while using OpenRouter `supported_parameters` when available and a one-shot parameter-rejection retry when providers reject optional sampling. This keeps review slots from disappearing on OpenRouter `404 No endpoints found...requested parameters` while preserving the quality guarantees that `require_parameters` was added to protect.

Vision tools (`analyze_screenshot`, `vlm_query`) route through `LLMClient.vision_query` using explicit `model` when supplied, otherwise the active model plus `OUROBOROS_MODEL_VISION` (empty→main), light/heavy/main, and fallback candidates, with `resolve_effort("task")`. They force a short no-proxy HTTP client timeout so a slow VLM provider cannot occupy the tool thread for the global 600s timeout, and image payloads are capped/downscaled before provider submission while non-image file inputs remain fail-closed. v6.45.0 adds send-time image routing in `ouroboros/vision_routing.py`: `OUROBOROS_IMAGE_INPUT_MODE=auto` leaves image blocks inline for vision-capable active models and converts them to generic captions for blind models on a per-send COPY of the transcript (canonical messages are not mutated); `caption` forces captions, `inline` refuses auto-caption fallback, and `off` emits placeholders. `OUROBOROS_MODEL_VISION` is the optional caption/VLM slot (empty→main, legacy `OUROBOROS_VISION_MODEL` migrates). A separate `view_image` tool (`ouroboros/tools/vision.py`) injects a LOCAL image file NATIVELY into the active model's conversation context (reusing the browser-screenshot native-injection path + `supports_vision` + the image-block eviction budget, `context_budget.MAX_LIVE_IMAGE_BLOCKS`, K=5 since v6.81.1) so a vision-capable model reasons over the image inline rather than via a second-model sub-call; it accepts LOCAL PATHS ONLY (no URL/base64, the same trust roots + protected-artifact read policy as `vlm_query` via the shared `_load_local_image_payload`). v6.81.1 factors that whole attach path into `vision.attach_local_image_to_context` — ONE body behind both the agent-called `view_image` tool and the host's same-round auto-attachment: a successful tool result whose JSON carries the typed, opt-in `auto_attach_image: <local path>` field (emitted by the `unix_computer_use` screenshot) has its image attached by `loop_tool_execution.process_tool_results` immediately after the round's tool messages, through that same shared seam — identical trust boundary, identical durable copy under `uploads/views`, identical message shape — so one observation costs one round instead of the previous screenshot→`view_image` pair (~21% of the round budget on computer-use benches). Attachment failure is strictly non-fatal: the result still carries the path and the agent can `view_image` it manually, which is exactly the pre-6.81.1 behavior. `view_image`, `vlm_query`, and `analyze_screenshot` are vision/local-media tools rather than `_WEB_TOOLS`; benchmark isolation withholds them by name through `disabled_tools`.

Background Consciousness is a high-horizon internal awareness loop, not a cheap helper lane. It may update memory and identity and proactively message the owner, but it does not directly execute powerful work such as subagent delegation, shell/code execution, reviews, commits, or evolution toggles. It grooms backlog and cognitive state; Evolution Campaigns execute targeted self-improvement work through the normal task/review path.

Evolution Campaigns replace the old empty `EVOLUTION #N` trigger text with a
goal-directed campaign prompt. The supervisor still schedules evolution from the
fast idle queue path, so consecutive campaign iterations can start as soon as the
queue is empty; only the task objective and persisted progress become explicit.
Campaign checkpoints are appended to `data/state/evolution_checkpoints.jsonl`
with git/memory hashes, `outcome_axes`, and per-cycle cost/rounds for future
eval curves. Task attempts/campaign tasks are counted separately from absorbed
evolution cycles: an absorbed cycle requires a reviewed self-mod commit plus
successful startup restart verification of that commit.
The evolution redesign (v6.30.0) closes the structural feedback and hygiene
gaps. Campaign state and the transaction lifecycle live in
`supervisor/evolution_lifecycle.py` (queue.py keeps queueing only).
Solve-capability ledger: because a commit-bearing cycle is recorded
`waiting_for_restart` at task-done, the later absorb/abandon resolution
(restart verification or boot reconcile, `agent_startup_checks.py`) appends a
`kind="cycle_outcome"` tag row to the checkpoints ledger (join key `task_id`),
and `build_solve_capability_digest` feeds the absorbed-vs-failed objective
history (explicit omission notes — never silent truncation) into the post-task
promotion prompt. The ledger is therefore schema-additive: classic absorb
checkpoints carry git/identity hashes while `cycle_outcome` tag rows do not,
and the `/api/evolution-data` checkpoints projection filters tag rows out so
the Dashboard view renders absorb checkpoints only. Deterministic cycle cleanup: a no_op/abandoned cycle restores
the worktree to the transaction's `base_head` —
dirty files are stashed (`evolution-cycle-cleanup-<tx>`), an ahead HEAD is
preserved as a local `evolution-leftover-*` branch, both refs are recorded on
the transaction; the reset is skipped (with a recorded reason) while other tasks
run in the shared worktree, under pytest against the live repo, or via the
`OUROBOROS_EVOLUTION_CYCLE_CLEANUP=false` kill-switch. `commit_reviewed`
refuses another triad+scope run after 3 genuine review-verdict blocks of a
byte-identical staged diff (`attempt_cap_reached`; diff-scoped so a new task
cannot reset the streak; preflight blocks neither count nor break; a changed
diff or a `review_rebuttal` lifts it). The hard-kill path also cleans the task
owner-mailbox so a stale `finalize_now` can never instantly force-finalize a
same-id subagent retry.
Post-task self-evolution (V4 envelope + V5 promotion, `post_task_evolution.py`) is
an owner-gated, default-OFF way to trigger a cycle BETWEEN tasks instead of only
when idle. After a qualifying task (never an evolution/`deep_self_review`/subagent
task, and only on the canonical dual-run pass), the worker makes an LLM-first
decision and, if it promotes one improvement, writes a durable
`state/post_task_evolution_request.json` signal — it never enqueues or enables
evolution itself. The supervisor idle tick reads that signal (`apply_pending_request`,
before `enqueue_evolution_task_if_needed`), sets the campaign objective via
`start_evolution_campaign`, enables evolution with a one-shot `post_task_autostop`
flag, and deletes the request; the normal gated enqueuer then runs exactly one
cycle through every safety gate, and the absorbed cycle clears the autostop flag so
evolution turns off again. A promoted backlog item that `requires_plan_review`
carries that obligation into the objective. The enable key is owner-only (not
settable via the generic `/api/settings` merge). An owner stop is AUTHORITATIVE
against this pipeline: `/evolve off`, the agent `toggle_evolution(False)` tool, and
panic all set the durable `evolution_owner_stopped` sentinel (`supervisor/state.py`,
cleared only by an owner-authorized start — the `/evolve start` slash command or the
owner-directed `toggle_evolution(True)` tool, mirroring how `toggle_evolution(False)`
is itself one of the three owner-stop sites; carrying explicit owner-authorized
provenance on evolution-start events is a deferred hardening), terminally close the
campaign via
`complete_evolution_campaign` (a non-`{active,paused}` `stopped` status, so a later
start mints a FRESH campaign rather than resurrecting the stopped one — distinct from
the resumable `pause_evolution_campaign` the system breakers use), and drop any queued
request. `apply_pending_request` re-checks `evolution_owner_stopped` before the sole
enable site and drops the request when set, so a boot tick can never autonomously
re-arm evolution after the owner stopped it (the one remaining autonomous enable path
flips `evolution_mode_enabled` only with the flag clear and is audit-logged). Evolution remains
self-modification work and is hard-blocked in `light` runtime mode
at every entry point (`/evolve`, the Evolution UI Start button, the agent
`toggle_evolution` tool, and the idle enqueue path), requiring `advanced`/`pro`.
Checkpoints are surfaced through `GET /api/evolution-data` (`checkpoints`) and
the JSONL ledger rather than the Evolution chart, which renders campaign
cycles/progress; the digest/ledger split keeps the chart focused on tag growth.

Loop checkpoints are plain user-message self-checks by design. A prior structured-reflection mechanism (four-field contract, tools disabled, `effort=xhigh`) produced 0 valid reflections and 37 anomaly records in production: system-role injection was absorbed into the top-level prompt, high effort with no tools invalidated cache every round, and the strict parser rejected natural model output. The minimal checkpoint is intentional; do not reintroduce structured reflection without new evidence.

Tool API v2 exposes neutral canonical names directly. Public schemas use
`read_file`, `list_files`, `search_code`, `write_file`, `edit_text`,
`run_command`, `run_script`, `claude_code_edit`, `verify_and_record`, service tools,
`commit_reviewed`, `vcs_*`, `schedule_subagent`, `wait_task`, and
`wait_tasks`. Legacy public tool names are a breaking rename in v6.3: they
are not exposed and are not translated at execute time.
The file tools share a path-based public ABI: `list_files` uses `path` like
`read_file`/`write_file`/`edit_text`/`search_code`, not a separate `dir`
parameter.

Filesystem tool output is self-locating: file/search/edit/write results use
canonical `root:path` labels, and `run_command` / `run_script` echo the
resolved `cwd` in command result headers. This makes root mismatches visible
without collapsing the storage or safety boundaries between resource roots.
`user_files` is the first-class root for user-visible files under the owner's
home directory. It accepts relative home paths such as `Desktop/report.html`,
`~` paths, and safe absolute home paths, but rejects the Ouroboros repo and
runtime control-plane. `task_drive` is task-scoped scratch and
`artifact_store` is task-scoped under `data/task_results/artifacts/<task_id>/`;
external deliverables written through `user_files` or declared process
`outputs` are copied into that canonical artifact store for audit. Declared
directory outputs are stored as bounded manifest+zip pairs so generated sites or
reports remain a single auditable artifact bundle without leaking hidden/control
files. When a
user-visible file is rewritten through the same source path, the previous
canonical copy is retained outside the manifest under
`task_results/artifact_versions/<task_id>/` with last-5 retention; old versions
are recoverable but are not advertised as deliverables or served as task
artifacts. Two READ-ONLY orchestrator roots complete the set: `subagent_projects`
and `deliverables` are granted `read`/`list`/`search` only to orchestrator
profiles (never write/shell/cwd, never handed to a subagent) so a parent can
inspect a child-task project tree or a finished deliverable when synthesizing its
work.

### Safety and runtime mode

Every tool call passes hardcoded registry sandbox first, then policy-based LLM safety when required. The LLM layer's coverage is owner-selected via `OUROBOROS_SAFETY_MODE` (see the `/api/owner/safety-mode` contract above); the deterministic layers run in every mode. `runtime_mode_policy.py` defines protected paths: safety-critical files, frozen contracts, release/build/managed-repo invariants. Light mode blocks Ouroboros self-repo/control-plane mutation, not ordinary user-file creation: `write_file(root=user_files|task_drive|artifact_store)`, process cwd under those roots, and `claude_code_edit` in external user/task/artifact directories remain valid. In external workspace mode, task-local git is allowed while runtime repo/data paths remain protected; patch artifacts are captured against the preflight git base. Light still blocks `runtime_data` as an artifact workaround, direct repo writes, native/control-plane skill paths, state/memory/settings, VCS mutation against the Ouroboros repo, and runtime-mode self-elevation. Advanced can evolve normal app code; pro can leave protected edits on disk but the commit still requires review.

Task contracts can declare `resource_policy.protected_artifacts[]` for
execute-only black-box reference artifacts. Declared paths may be executed, but
registry guards block byte reads, copy/hash/static-introspection tools, and
trace/debug wrappers against those paths; generated binaries and logs remain
unaffected unless separately declared.

Rationale: runtime mode is a self-modification boundary, not an OS sandbox. It prevents casual damage to core identity/safety/release surfaces while preserving self-creation through reviewed commits and preserving normal user deliverables in light mode. (v6.74.0, D1) The light-mode SHELL guard resolves the requested cwd through the shared `resolve_shell_cwd` resolver BEFORE judging repo targets: `repo_target_mentioned` previously joined the raw cwd STRING onto `repo_dir`, so a resource-root LABEL (`cwd="task_drive"`) read as a repo-internal path and a legitimate task-drive write was false-blocked with a message advising the very root that was used; a cwd resolution failure now fails closed with the standard cwd block. (D3) The post-task cost publish (`post_task_checkpoint.py`) pushes the `task_cost_finalized` event to the live UI only via `try_get_bridge` — the durable events.jsonl append is the record of truth, and headless/benchmark finalization without an initialized message bus no longer logs a spurious warning.

### Claude runtime

`gateways/claude_code.py` wraps `claude-agent-sdk` for edit and read-only
advisory paths. Edit-mode delegation runs in the worker process with
`ClaudeSDKClient` lifecycle hooks, SDK-level path/tool guards, stderr capture,
normalized usage, and the dispatcher's protected-path policy as defense in depth.

Read-only advisory review is a separate crash boundary: `run_readonly()` starts
the same module as a Python child (`--readonly-child`) over JSON stdin/stdout.
The child uses the SDK client lifecycle and read-only tool allowlist, but native
abort signals such as `SIGABRT` are converted into structured
`ClaudeCodeResult(success=False, error=..., stderr_tail=...)` in the parent
instead of killing the long-lived worker. The child is launched in its own
process group/session and timeout cleanup kills the process tree, matching the
extension/subprocess containment pattern.

### Git and commit review

`tools/git.py` owns repo writes, staging, commit, rollback/revert/restore, auto-tag, auto-push, and CI-status follow-up. `write_file` writes without committing; `edit_text` does exact one-occurrence edits; `commit_reviewed` stages, checks advisory freshness, runs deterministic preflight, runs triad + scope review, revalidates the exact Git binding, commits, verifies, tags, and pushes. The binding is the `git write-tree` SHA, ordered `HEAD`/`MERGE_HEAD` parents, indexed VERSION, expected `v{VERSION}` tag, any existing tag target, and binary staged-diff hash. A changed parent/tag is therefore as invalidating as a changed file; an existing release tag is never silently accepted or retargeted. After `git commit`, tree/parents/VERSION and tag target are re-read before a success attempt or push is recorded.

When a queued root task starts, `mutation_attribution.py` records the exact Git
commit/tree plus baseline dirty paths and fingerprints in the existing task
evidence (the capture supports bounded exact targets for non-Git surfaces, but
v6.66.0 wires only the `system_repo` surface; it never recursively scans a user
home). A candidate is the clean-at-baseline delta of the task's observed
window — no exclusivity is claimed. A pre-existing dirty path that changed, a
stale/missing baseline, or a failed scan surfaces as a typed blocker that
disables automatic staging and rides into review evidence.

`commit_reviewed(paths=None)` stages only that attributed candidate; explicit
paths must be a subset, and an empty set returns
`GIT_NO_ATTRIBUTED_CHANGES`—never whole-tree staging. Managed release/update
transactions retain their separately typed `git add -A` authority, and contexts
without a captured baseline keep the legacy staging contract.
`scripts/run_external_review.py` reviews the staged diff in a frozen detached
checkout through the production cycle, or uses the typed non-committing
`external_pr_readiness` profile described in §Mutation attribution. The latter
never replaces the final production review bound to the current landing parent,
VERSION and release tag.
Terminal/quiescent evidence recomputes the interval delta, including task
commits already between the baseline commit and current HEAD.

`review_state.py` persists advisory runs, reviewed attempts, obligations, commit-readiness debt, and stale markers. `commit_readiness_debts` must remain: it blocks repeated unresolved review friction and anchors retries to root causes.

Rationale: commit review is the immune system's blocking feedback loop. The staged snapshot, advisory coverage, triad evidence, scope evidence, parent vector, VERSION/tag expectation, and post-review fingerprint must describe the same commit material or the commit is not trustworthy.

### Review stack

- Advisory pre-review (`claude_advisory_review.py`) is mandatory freshness coverage before commit. It is staleness-aware and auditable; bypass is explicit and logged.
- Triad diff review (`tools/review.py`) asks configured reviewer slots to cover the Repo Commit Checklist with JSON findings. Quorum is adaptive to the configured reviewer count via `config.adaptive_quorum` (v6.36.0): 2-of-N for N≥3, both for N=2, and a single configured reviewer for N=1 — the latter runs as a loud `single_reviewer_no_diversity` degraded mode (owner's explicit small-config choice), while a configured-≥quorum-but-fewer-responded shortfall stays a loud infra quorum failure. The same SSOT governs scope/plan/skill/acceptance review.
- Scope review (`tools/scope_review.py`) sees touched context plus a Generated Scope Atlas and checks intent/scope/coupling. The Atlas target is an 850K estimated-token assembled prompt under the 920K hard review budget; it raw-inlines selected protected/central files and accounts for every tracked path as full, already included, manifest-only, excluded, sensitive, binary/media, vendored/minified, oversized, read-error, or budget-omitted. Scope review is fail-closed on unreadable touched files and budget-aware on oversized prompts; whether findings block or downgrade to advisory follows `OUROBOROS_REVIEW_ENFORCEMENT`.
- Parallel orchestration (`tools/parallel_review.py`) launches triad and scope concurrently so the agent receives all findings in one round.
- Shared helpers (`review_helpers.py`, `triad_review.py`) own pack building, checklist loading, JSON extraction, usage events, obligations/history prompt scaffolding, and reviewer actor records.

Task acceptance is a root-owned post-delivery system, separate from the P3 commit gate. `off` disables it; `auto` and `required` review queued/headless work plus direct work with effectful changes or an explicit typed deliverable/acceptance contract. Ordinary read-only research/tool use in direct conversation, pure conversation, and child authorities do not produce a competing root verdict. Before review, the supervisor closes subtree admission under the queue lock and requires recursive terminal quiescence. Split-drive fence acknowledgement, subtree lookup, and EWMA timing all use the canonical `budget_drive_root`; the one-shot `state/acceptance_fence_acks/` IPC sidecar is not a lifecycle authority, and each transition compacts rows older than one hour and bounds retained acknowledgements to 256. `_run_task_acceptance_review_once` then builds one immutable evidence core (verbatim owner directives and accepted decisions, deliverable and criteria, subtree statuses, verification/artifact references, canonical payload provenance, and explicit omissions) and gives it to the independently configured task-review panel. Each actor makes one substantive call and at most two physical attempts total (same-route transport retry or extraction-only repair); there is no acceptance scope actor. `adaptive_quorum` decides participation. A task-acceptance `FAIL` contributes only with the required outcome tier and a bounded correction rail; a bare veto abstains rather than terminalizing the task without an actionable path. `DEGRADED` abstains from quorum and obligations. A deliberate semantic DEGRADED with a concrete recommendation can still feed the advisory improvement capsule, while transport/unparseable no-quorum is recorded terminally (v6.78.0: `finalized_unaccepted` with `reason=review_degraded`), never as PASS and never as revision authority. A clean result requires quorum PASS, a `solved` tier, and supported evidence for every contributing criterion. Actionable gaps are exact-deduplicated and feed the existing improvement loop; an explicit `max_improvement_passes` binds every policy, while Required+Blocking without one has no local count cap and remains bounded by deadline/global lifecycle rails. The first review reserves at least 200 seconds; later passes reserve `max(configured floor, 1.5×EWMA)` using canonical existing timing events (`alpha=0.5`). The structured review axis is mirrored as top-level `review_status` for task-result/gateway/event compatibility. Post-task synthesis recovery runs only at startup and consults one checkpoint in the canonical `budget_drive_root` task result: it replays only `pending_once`, terminal-degrades indeterminate `running` without a second paid call, and ignores terminal markers. Normal supervisor child copy-back/artifact finalization remains responsible for materialization; a late copy-back may enrich the result but cannot overwrite a terminal canonical phase. Minority dissent and blocking-lane obligations remain typed, auditable inputs, but the root acceptance verdict and stop reason are stored separately from the terminal lifecycle/artifact result.

Rationale: diff reviewers catch line-level mistakes; scope reviewer catches cross-module contracts and forgotten touchpoints. Running both on the same staged snapshot prevents one reviewer result from hiding the other.

Structural smoke gates (deterministic, BIBLE P3 "codebase size" component): the
constants live in `ouroboros/review.py` (`MAX_TOTAL_FUNCTIONS`,
`MAX_MODULE_LINES`/`GRANDFATHERED_OVERSIZED_MODULES`, `MAX_FUNCTION_LINES`) and
are enforced by `tests/test_smoke.py` both in CI (quick-test on every push) and
in the hermetic pytest preflight that runs before every self-commit review.
`tests/`, `devtools/`, and the frozen `launcher.py` shell are excluded from the
function-count walk; grandfathered modules are an explicit debt register, not a
loophole. Growth must be acknowledged: raising a gate value requires a
deliberate edit of the constant with a one-line justification (never hardcode
the number elsewhere). These gates caught externally merged PRs that bypassed
the in-process review path — they are the last deterministic line of the immune
system, so weakening them requires the owner's explicit decision.

The shared hard prompt-size SSOT is `REVIEW_PROMPT_TOKEN_BUDGET = 920_000` in
`ouroboros/tools/review_helpers.py`. `review_context_atlas.py` targets 850K
estimated total prompt tokens for scope review, plan review, and deep
self-review, then leaves the final 920K gate in each caller as the hard stop so
oversized-context behavior cannot drift between review entry points.

Scope review additionally reserves output headroom inside the reviewer's 1M
window. The 920K SSOT governs INPUT, but the scope reviewer also reserves
`_SCOPE_MAX_TOKENS` (100K) for OUTPUT and a tokenizer headroom margin because
provider accounting can exceed the local estimator on atlas-heavy prompts. 920K
input + 100K output exceeds 1M, which the provider rejects with a hard 400.
Such a physical rejection is UNCONDITIONALLY fail-closed in `max` mode: there is
no authoritative verdict, and since v6.80.0 no setting can turn it into a
non-blocking `budget_exceeded` skip — `OUROBOROS_SCOPE_REVIEW_FLOOR` still exists as
a stored owner setting but is enforcement-inert and consulted by nothing. The only
owner control over scope review is the context mode: `low` means whole-repository
scope review is declaredly not performed (typed `skipped_low_context_mode` row), and
`max` means this fail-closed gate. So
`scope_review.py` gates the assembled INPUT prompt on
`_SCOPE_INPUT_TOKEN_LIMIT = min(920K, 1M − _SCOPE_MAX_TOKENS − margin)`, with a
substantial tokenizer headroom margin (currently 155K tokens) — the 920K
SSOT itself is left untouched. The cap is additionally DENSITY-CALIBRATED: the chars/4
estimator tracks GPT-style tokenizers within that 155K margin, but Claude-family
tokenizers cut code-heavy packs at ~2.5 chars/token — a real scope pack estimated at
739,508 tokens measured 1,166,914 REAL tokens (1.58x) and was rejected 400 `prompt is
too long` by every upstream. Since v6.80.0 the ratio is MEASURED, not a hand-set family
constant: the former `CLAUDE_REAL_TOKENS_PER_ESTIMATED = 1.65` and the
`is_claude_family_model` substring gate are DELETED. `usage_accounting.
execute_physical_attempt` records `(prompt_chars, real prompt_tokens)` after settlement
and OUTSIDE the ledger lock (fail-soft) into a separate `token_density` namespace in
`capability_evidence.json` — keyed by NORMALIZED MODEL IDENTITY like
`effort_ceilings`/`rejected_params`, bounded raw-pair retention, writes throttled to a
first observation or a >5% drift because that store shares one file and one lock with
the scope-review path, and cache-bearing usage SKIPPED because Anthropic excludes cache
reads/writes from `input_tokens` (an under-measured density would loosen the cap — the
one dangerous direction). The calibration SSOT
`review_helpers.calibrated_input_token_limit` then returns the STRICTEST of three
bounds — the 920K budget cap, the density form `(window − output_reserve) / density`,
and the historical `window − output_reserve − tokenizer_margin` — so it can never
exceed the previous cap; with the documented conservative cold-start density (1.65,
the measured 1.58 plus margin) a model with NO observation sizes DOWN rather than up
from an optimistic estimate, and provenance is reported as `measured` or
`cold_conservative`. That cold-start density bounds the COLD path ONLY:
`resolve_token_density` returns `measured × safety` once an observation exists, with no
cold floor on the measured path — the constant is Claude-derived, and flooring every
model with it would permanently charge a genuinely lighter tokenizer for Claude's
density with no way for measurement to correct the direction. "Measurement can only
ever TIGHTEN a cap" is supplied where it belongs, PER MODEL IDENTITY in the store:
`record_token_density` keeps the RUNNING MAXIMUM for a normalized model identity, and
one identity collects observations from EVERY surface that uses that model, so a run
of doc-only commits whose
prose-dominated packs measure ~1.1 cannot pull the stored density down and hand the
next code-heavy scope pack a bigger cap than today's — the same 400 this calibration
exists to prevent. The historical absolute-margin form still bounds every result, so no
cap can exceed its pre-measurement value. Provenance stays `measured` when an
observation exists. `scope_review._effective_scope_input_limit` computes it PER CALL
(an import-time constant froze the pre-measurement value for the whole process, so a
measurement could never reach it), and the triad (`tools/review.py`),
`plan_review.py`, and `deep_self_review.run_deep_self_review` consume the same helper. The scope cap is WINDOW-AWARE: a known reviewer window from
Capability Evidence (`_scope_reviewer_window` -> `ouroboros.capability_evidence`;
no static table, v6.33.0) replaces the assumed 1M when computing the effective
input cap. A known sub-1M reviewer remains advisory-only: in `max` its result is
preserved as evidence but cannot satisfy the gate, and the commit fails CLOSED —
the deprecated `OUROBOROS_SCOPE_REVIEW_FLOOR` no longer converts that into a
non-blocking `budget_exceeded` skip (the GigaChat-only / no-≥1M-reviewer case is answered by the
owner choosing `low`, where scope review is declaredly not performed and each
skipped commit records the typed `skipped_low_context_mode` row, not by a weaker
blocking gate). The same authority rule applies if the estimate-based gate passes but the
provider's REAL tokenizer rejects the prompt as oversized (`prompt is too long`,
`context_length_exceeded`, …). Every other provider or transport error remains
fail-closed. The calibration shrinks the PROMPT
for the same pinned reviewer — never the reviewer model or the ≥1M window floor
(P3). Plan review fans one shared prompt across mixed-family slots and (v6.80.0) now
sizes it PER SLOT from the same calibrated helper — closing the former "planned
follow-up work" gap that made a Claude plan slot 400 deterministically: a slot the
shared prompt cannot fit gets a FREE deterministic `preflight_oversize` record instead
of a guaranteed-400 call, and fewer callable slots than the review quorum is a loud
typed `PLAN_REVIEW_DEGRADED_PREFLIGHT_OVERSIZE` with NO reviewer called — never a
silent absence of review.
Non-responded scope actor records also surface the provider failure text
(`error` field in `build_scope_actor_record`) so a deterministic 400 is visible
in the verdict without observability digging. The scope coverage contract
requires explicit `severity` only on FAIL rows (it decides blocking and stays
fail-closed); PASS rows default to `advisory` like the triad parser.

Scope prompt assembly is GUARANTEED-FIT (v6.30.0): the owner directive is that
scope review must actually run, so the assembler walks a deterministic
degradation ladder instead of skipping. 1) full atlas; 2) compact atlas (the
durable `context_manifest` keeps full per-file coverage while the visible
prompt keeps a compact path/disposition coverage index); 3) inside the atlas a
required file that cannot fit degrades to an explicit `budget_omitted`
manifest entry instead of failing the whole pack (`budget_exceeded` survives
only when even the content-free manifest cannot fit); 4) the largest touched
files degrade to diff-only — their full post-change snapshots are replaced by
an explicit `TOUCHED FILE BUDGET DEGRADATION NOTE` while their complete
changes remain visible in the staged diff; 5) unchanged hunk context may be
removed with `-U0`, preserving every file/hunk identity and every `+`/`-` line.
Triad independently applies the same one-pass fit rule before dispatch: a
disclosed touched-path manifest can replace full snapshots duplicated by the
complete diff, followed by the same `-U0` fallback. Every step is a disclosed omission
(P1), never silent. Only the irreducible prompt (checklist + canonical docs +
staged diff) failing to fit remains, and that fails CLOSED
(`fixed_overflow` blocks the commit and tells the owner to split the diff or
configure a larger-window reviewer). The `budget_exceeded` and provider-oversize
outcomes are recorded as evidence but never satisfy the P3 gate, and since v6.80.0
no setting makes them non-blocking — in `max` they block. The P3-aligned remedy
for a structurally oversized repo stays shrinking/splitting the reviewed tree,
never lowering the reviewer below the 1M context floor.

In owner-selected `low` context mode (v6.80.0) `run_scope_review` returns before
assembling anything — the predicate reads `config.get_owner_context_mode()`, never the
effective mode, so the agent-reachable `/api/settings` auto-downgrade cannot switch the
gate off (see the `/api/owner/context-mode` contract above): no reviewer is called,
the commit is not gated on scope, and a
typed non-blocking `status="skipped_low_context_mode"` result is recorded through the
SAME `build_scope_actor_record` review-evidence surface that carries the fail-closed
results, so a low-mode commit is never forensically confusable with "scope review
silently failed to launch" (P1). This is the owner's policy coupling, not a coverage
claim; the removed opt-in degraded advisory builder (`OUROBOROS_SCOPE_REVIEW_DEGRADED`,
`_LOW_SCOPE_INPUT_TOKEN_LIMIT`) is gone with it, and the one-pass gate keeps returning
the normal actor's authoritative or fail-closed status in `max`.

### Planning, deep review, reflection, memory

`plan_review.py` runs the existing multi-model Atlas-backed panel before large implementation plans. Read-only planning scouts use the normal subagent pool and persist full raw handoff artifacts. (v6.79.0) A NEW wave is admitted before launch — and only a new one: worker capacity, the shared `review_helpers.review_wave_budget_gate` (the same admission the reviewer wave and skill review use, no second budget authority; it prices one opening round per scout, a deliberate lower bound), and a consumable window. Each scout's contract deadline is bound to that window (the wave's shared cutoff minus the finalization grace and a margin, the reserve capped at a fraction of a short window) instead of inheriting the parent deadline verbatim, so a scout cannot keep spending past the moment the parent stops reading; a wave whose window has already closed is refused with a typed reason instead of being launched and then omitted. The recovery/collection path is never admitted against — those handoffs are already paid for, and refusing them would abandon spend rather than save it. With delegation disabled (`OUROBOROS_MAX_SUBAGENT_DEPTH=0`) the same gate that refuses any child refuses the scouts, and plan review completes on the existing `degraded_evidence` path. Authoritative wave/review history is the bounded additive `plan_review_state` inside the existing `task_results/<id>.json`, updated under that result's per-file lock; the task-writable `plan_task_handoffs.json` projection is audit-only and is never read for closure or scout identity. The host persists every intended scout and one absolute cutoff before the first launch, then persists each issued task id or scheduling failure immediately; if launch completed before that second write, resume recovers the exact durable direct-child id instead of scheduling again. Fingerprint-keyed history therefore survives A→B→A without launching a duplicate wave; an older open `REVIEW_REQUIRED` result is re-presented from cache before it can accept a disposition, and every resume spends only the remainder of the original cutoff. Every successfully launched scout is awaited until terminal state or that shared cutoff; the panel receives every ready non-empty handoff plus exactly one typed omission per unfulfilled scout intent, including bounded redacted scheduling/terminal detail. A reviewer-included snapshot receives the common exact-hash `integrated` disposition when still current. If the child changes after the snapshot entered the prompt, the old hash remains non-authoritative, the paid review still persists once with a bounded `CHILD_RESULT_STALE` warning, and the newer result is audit-only. All task ids in a reviewed planning wave — included or omitted — are excluded from the generic child-absorption gate and root-acceptance quiescence once their review is authoritative, so late arrivals cannot reopen either boundary. Governance documents always come from the system repo, while planned file snapshots and Atlas inventory come from `active_repo_dir_for(ctx)`. A missing/mixed subject root or any planned path escaping that root fails loudly instead of silently reviewing Ouroboros in place of an external workspace.

The reviewer horizon carries the goal, mandatory invariants, scope boundaries, non-goals, chosen existing extension seam, and explicitly rejected expansions together with task aliases, forensic refs, handoffs, and omissions. Reviewers stay generative, but each finding must identify a concrete defect or a concrete smaller existing extension point; there is no numeric finding quota. The only public aggregate values are `GREEN`, `REVIEW_REQUIRED`, and `REVISE_PLAN`. `REVIEW_REQUIRED` may be closed on the same fingerprint without another reviewer call only by a valid fingerprint-bound `review_disposition` that addresses every finding exactly once with `accept | reject | defer`, evidence-based rationale, and a plan-revision reference for accepted findings. `REVISE_PLAN` requires changed plan text and therefore a changed fingerprint followed by a new `plan_task` call. Unknown, duplicate, contradictory, incomplete, or stale dispositions fail closed where a bindable review exists. A VACUOUS `review_disposition` (the empty object schema-filling models emit instead of omitting the optional field) still means "absent" and runs a real wave with a disclosed note. A NON-EMPTY disposition that can close nothing now FAILS FAST (v6.80.0) instead of being discarded before a paid wave: `PLAN_REVIEW_DISPOSITION_UNBINDABLE` reports the claimed fingerprint, the latest stored review fingerprint, the submitted request fingerprint, whether the plan text still matches, any `ENVELOPE_MISMATCH` components, and the two ways forward — the anti-wedge escape ("omit the field entirely") is stated EXPLICITLY in the error text, which is what the v6.65.0/1 silent discard was protecting. A disposition may also HONESTLY bind the wave it NAMES when all FOUR conditions hold: its `review_fingerprint` EQUALS the fingerprint of the envelope being submitted, that fingerprint is the state's `latest_review_fingerprint`, that review is an open `REVIEW_REQUIRED`, and the plan text still hashes to the stored `plan_text_hash`. The first condition is the P3 gate itself: `files_to_touch` is exported in the tool schema as part of the review IDENTITY, so binding a DRIFTED envelope with a prepended `ENVELOPE_MISMATCH` warning let a review of `[a.py]` close a submission for `[a.py, b.py, c.py]` — stale plan-review evidence authorising materially expanded scope. Agent-authored drift is therefore UNBINDABLE (fail-fast, no wave, no money; the escape is to omit the disposition and get a real review of the new envelope). The binding fingerprint is a pure function of the AGENT-PASSED envelope — host-resolved `plan_class` and `context_level` are EXCLUDED (they made the identity unreproducible by the agent), while per-field `component_hashes` stored at wave creation may include resolved values, so `ENVELOPE_MISMATCH` on a BOUND wave now reports exactly that host-resolved drift. Stored fingerprints from earlier releases are invalidated once by this change, harmlessly (the affected review simply re-runs). State-lookup failures stay loud `PLAN_REVIEW_STATE_INVALID` errors, never treated as absence. `scripts/run_plan_review.py` remains a thin operator wrapper over this production panel.

Ordinary Main calls capture one immutable context core and `context_fit.py` renders deterministic Max/Low projections from it. Fit uses exact family+route Capability Evidence plus the existing Atlas/family calibration; unknown routes attempt Max rather than receiving a silent 200K assumption. A confirmed real overflow may rebuild once into task-local Low for the same model, with a forensic checkpoint and visible Activity/card/toast event; global context mode and the P3 commit gate are unchanged. Stable policy/governance blocks precede dynamic evidence for cache reuse. Direct OpenAI may send `prompt_cache_key` and OpenRouter may send `session_id`; an explicit unsupported-parameter rejection gets one exact retry without that hint, while ordinary transport/deadline retry semantics remain unchanged.

`deep_self_review.py` runs a direct Atlas-backed self-review without the tool loop while keeping the memory whitelist full. `reflection.py` records process lessons; `consolidator.py` compacts dialogue/scratchpad through explicit summaries; `context.py` assembles static, semi-stable, and dynamic context sections. Summary/consolidation calls resolve through the existing Light lane at call time, including the lane resolver's empty-Light-to-Main and local-routing semantics; only remote routes pass through credential fallback. Post-task task summaries reuse that same route, so benchmark `--all-model` pinning covers late synthesis without a hidden model-specific override.

Experience Review closes the learning loop: the reflection LLM may append a `MEMORY_ACTIONS_JSON` block whose validated actions (`scratchpad_append`, `knowledge_write`, `identity_update_candidate`) are auto-applied via `reflection.apply_memory_actions` through the existing provenance-preserving memory/knowledge paths (`Memory.append_scratchpad_block`, `knowledge._knowledge_write`). Identity is deliberately conservative — an `identity_update_candidate` is only recorded in the scratchpad for review, never auto-written to `identity.md`, so autonomous learning cannot silently drift the personality. A split non-Project root runs the one full post-task synthesis on the canonical `budget_drive_root`; a Project-scoped root runs it once on the Project child drive and forwards only the sanitized improvement-backlog promotion to the canonical drive. There is no child+parent full dual-run. A root external/workspace or `--project-id` task derives a resolved `project_id` (explicit id, else a stable workspace-path hash); subagents INHERIT the parent's resolved scope and never derive their own (a subagent of an unscoped parent stays unscoped). Project facts therefore never contaminate global memory or another Project: `knowledge_write` and the context loader redirect to the per-project store (`projects/<id>/knowledge` under the canonical data dir via `ouroboros/project_facts.py`), which persists across forked/empty child drives. There is no per-project identity, and only the current project's facts are loaded at context build (red-team R3.1 leak guard).

Multi-project ("штаб и проекты", v6.32.0) builds the owner-facing layer on this substrate while identity, constitution, and evolution stay UNIFIED in the one agent (BIBLE P1):

- **Full project awareness (one mind, focused rooms).** Ouroboros is ONE awareness across direct chat, project rooms, and background consciousness (BIBLE P1), so its unified memory — the recent-dialogue tail, the consolidated `dialogue_blocks.json`, and `chat_history` recall — spans ALL threads (main + projects); only A2A virtual transport is excluded. A project is therefore a focused ROOM, not an isolated sub-mind: an individual project TASK gets a FOCUSED context (its own thread + its own journal/workpad/knowledge) to reduce cross-project interference while executing, but this is working focus, not memory isolation from the one identity. The UI organizes threads into panels (project raw chat in its panel; progress mirrored to the штаб), but that is presentation, not a cognition boundary.

- **Projects registry and lifecycle** (`ouroboros/projects_registry.py`, `data/state/projects.json`): immutable id + deterministic `chat_id`, name (80-character SSOT), optional working folder/provenance, `last_active_at`, `visible_revision`, routing generation/fence, and `active | deleting | tombstoned`. Boot reconcile registers pre-existing stores and never age-prunes or resurrects a reserved id. Creation still supports attach, clone, genesis, or file-less Projects and keeps `project_room_lens_dir` / `room_chat_lens_dir` behavior. Rename mutates only the display name. Delete first closes admission/routing and increments the fence generation, then reuses queue cascade cancellation/quiescence; only after the subtree settles does it atomically tombstone. ID, canonical chat/history, immutable bindings, folder, journal/workpad, and memory are preserved. An interrupted deletion remains recoverably `deleting` and resumes at startup; `project_hidden` is a one-minor deprecated no-op, not lifecycle state. `GET /api/fs/dirs` remains the owner-facing home-confined directory browser. **Room lens (v6.61.3):** direct-chat reads and default shell cwd use the registered working folder; default-root writes return `ROOM_WRITE_VIA_TASK`, and a broken folder fails loudly rather than falling back to the system repo.
- **Canonical chat and owner routing**: Web, CLI, and existing owner transports enter the same router. In Main, the ordinary decision turn gets a compact manifest of Projects, RUNNING/PENDING roots, recent final results/artifacts, and recent canonical dialogue; if there are no Projects/tasks it retains the prior direct flow. In a Project room, exactly one addressable RUNNING/PENDING root is a zero-call mailbox delivery. Zero or multiple candidates use one LLM decision scoped to that Project. Uncertainty, stale targets, and selection errors return typed `needs_manual_target` with concrete task options and `New task in Project`; they never pick or spawn randomly. `handle_chat_ephemeral`, `steer_task`, promote/bind/history/task-result lookup, attachment staging, and the existing serialized direct lane are reused—there is no parallel message lane. Each inbound row is stored once; Project history projects the binding-held source refs (immutable once valid; a ref-less binding may be one-way enriched), and when the canonical row has left the bounded read window the lens synthesizes the start message from the binding's own `source_text` (post-quota, identity-deduped, hard-capped — `origin_projected=true`); `chat_annotations.jsonl` carries only the latest owner-visible action/target/status. The typed annotation is presentation metadata attached to the owner's message, while the Agent normalizes blank model output to a visible warning and every finalized response remains a separate durable assistant reply across Web and non-Web transports. Inline Main replies create no routing annotation.
- **Multi-task chat steering (v6.34.0, WS1)**: when several tasks run in one chat, a new message must be able to STEER a chosen running task — "turn = decision" (WS10) alone could only answer or spawn. The agent makes that choice by JUDGMENT, not code: `build_runtime_section` (`context.py`) surfaces a STRUCTURAL `current_chat.running_tasks` fact (the running ROOT tasks in the current chat: `{task_id, title, objective, project_id, started_at, steerable}`, snapshotted from supervisor `RUNNING`) into the runtime JSON block, fed through `task_metadata` (`server.py::_decision_turn_metadata`) into both the direct and ephemeral decision turns. The new `steer_task(task_id, message)` capability (`tools/control.py`, beside `promote_chat_to_task`/`route_to_project`) enforces only TRANSPORT invariants — target still in `RUNNING`, same chat / project binding, not a subagent — and delivers via `write_owner_message` on the active task drive (`supervisor/events.py::_handle_steer_task`); the running task drains it at its next round. Idempotent via `client_message_id`-derived mailbox `msg_id` (no double-deliver on retry); a STALE target fails VISIBLY rather than auto-spawning (the agent's safe default when unsure is `promote_chat_to_task`). Code never decides "this message belongs to task A" — it only exposes chat state + enforces invariants (P5/BIBLE LLM-first). `forward_to_worker`'s parent→child descendant guard is unchanged. The project-room pre-LLM delivery (`_route_project_chat_to_running_task`) is narrowed to the UNAMBIGUOUS 1:1 case only — exactly one steerable pooled task in the room is a transport invariant; with zero or multiple candidates the message now flows to the decision turn so the agent picks via `steer_task` rather than code mechanically selecting the first of several.
- **In-task project scoping** (`ensure_project_scope`, v6.37.0 C4.1): once work is ALREADY running, the agent can name+create a project and bind THE CURRENT task to it in one structural move — the affordance for "make this a project named X" without falling back to a bare `mkdir`. `tools/control_delegation.py::_ensure_project_scope` (the delegation/scope affordances extracted from `tools/control.py`, wired into its `get_tools`) validates/derives the project id, sets `ctx.project_id` for the rest of the loop (so `journal_write`/per-project knowledge target it immediately), and emits an `ensure_project_scope` event; `supervisor/events.py::_handle_ensure_project_scope` → `supervisor/workers.py::ensure_project_scope` creates the registry project, durably binds the task (`bind_task_to_project`), updates the live `RUNNING[tid].task.project_id` (so the one-writer lease sees it), and broadcasts `projects_changed`. Idempotent for the same project; refuses to re-scope to a different one; subagents inherit the parent's scope and cannot change it. `promote_chat_to_task` remains the preferred FIRST move from chat; this is the mid-run complement. The whole subagent tree's live/history frames then route to the project thread by lineage (`project_chat_for_task_tree`).
- **Per-project memory**: beside `knowledge/`, each project store carries `journal.jsonl` (milestones via `journal_write`/`journal_read`: start/checkpoint/blocked/done/note), `workpad.md` (`workpad_read`/`workpad_write`), and a thread mirror. The project task's FOCUSED context injects these via `build_knowledge_sections` WITHOUT silent prefix-slicing (BIBLE P1): the workpad rides in full (a warning, not a clip, signals an oversized one), and the journal shows recent milestones in full with a VISIBLE `journal_read` index pointer for older entries. Generic data tools still cannot reach the store (`project_store_access_block`).
- **One writer per project** (`ouroboros/project_lease.py`): `assign_tasks` skips a PENDING top-level task whose `project_id` is already RUNNING (its own subagent swarm is exempt — the parent IS the writer); `project_id==""` means no lane. Parallelism happens BETWEEN projects and within a task's swarm. The lease reads the task's STORED `project_id` (never re-derives at assignment). v6.58.0 unified the storage discipline: `resolve_project_id` is REGISTRY-FIRST (a workspace path that equals a registered project's normalized `working_dir` resolves to THAT project's id; the stable `proj_<hash>` is minted only for unregistered folders), and EVERY admission surface — `/api/tasks` AND the promote path — stores the resolved id in the task dict, so one folder is one serialized writer lane on all entry paths (previously the promote path stored nothing and a derived-id task skipped the lane entirely). The lease surface is the supervisor's IN-MEMORY `task['project_id']` in `PENDING`/`RUNNING` (and its `state/queue_snapshot.json` persistence), NEVER the durable `project_task_bindings.json` (that file is for chat/history routing only — see "Letters home"). So BOTH post-hoc convert paths must mark that in-memory surface, not just bind durably: the in-task `ensure_project_scope` (above) and the UI "turn into project" endpoint (`POST /api/projects/from-task` → `gateway/projects.py::api_project_from_task`) both call the shared SSOT helper `project_lease.mark_task_project(RUNNING, PENDING, …)` under the `supervisor/queue.py::_queue_lock` — the gateway runs in the same process as a thread, so it shares that lock with `assign_tasks`. The UI path marks BEFORE its durable `bind_task_to_project` (the mark is the conversion's effective commit point for serialization; an assign pass and the mark are mutually exclusive on the lock) and then calls `persist_queue_snapshot` so a still-PENDING converted task restored after a restart comes back STILL scoped.
- **Letters home**: a project-scoped task's completion appends a journal milestone and emits a `project_digest` event (project_id + full objective + outcome statuses) that the supervisor injects into consciousness as a concise completion summary. **Durable swarm coordination (F2, v6.39):** when the SWARM ROOT (a top-level project task — no `parent_task_id`) finishes, `project_journal.mirror_tree_coordination_to_journal` mirrors the EPHEMERAL task-tree ledger's HIGH-SIGNAL rows (attention beacons `blocker`→`blocked`, `question`/`interface_contract`→`note`, plus `contract`→`note`) into the DURABLE project journal once, so a swarm's blockers/contracts survive the tree's GC; low-signal rows (`fact`/`note`/`decision`/`milestone`/`partial_finding`) are NOT mirrored (the journal stays curated). Subagents skip the mirror — the root absorbs the whole tree. This is a convenience digest, NOT an isolation boundary: under full project awareness the one identity already sees the project's chat thread in its unified dialogue memory; the digest just gives consciousness a crisp "task X finished" signal. A project's RAW per-cycle internal facts stay in the per-project knowledge/journal store (scoped tools, `project_store_access_block`), while the project conversation is part of the one mind's continuous memory.
- **Restart drain**: agent-requested restarts wait up to `OUROBOROS_RESTART_DRAIN_MAX_SEC` for heartbeat-fresh RUNNING tasks before proceeding fail-closed, so an evolution restart does not chop parallel project tasks mid-flight.
- **UI**: the compact Projects header exposes layers icon, label, unread pill, chevron, and permanent `+`; sibling menus provide accessible Rename/Delete. A Project opens as the existing desktop split/mobile overlay chat instance over the one shared WebSocket. `visible_revision > project_seen_revision` is the unread rule, and the browser acknowledges only after paint. `projects_changed` keeps tabs synchronized; global controls stay Main-only.

Rationale: Ouroboros learns from attempts, not just final answers. Compression must preserve what was tried, what changed, and why conclusions were reached.

### Skills and extensions

Core skill flow is discovery (`skill_loader.py`), review (`skill_review.py`/`skill_review_runner.py`), readiness (`skill_readiness.py`), execution (`tools/skill_exec.py`), extension loading (`extension_loader.py`), dependency reconciliation (`marketplace/isolated_deps.py`), and lifecycle queue (`skill_lifecycle_queue.py`). The loader separates payload plane (`data/skills/...`) from owner/review state (`data/state/skills/...`). Lifecycle snapshots include structured queued/running/succeeded/failed state plus stale metadata for long-running active work; stale state is observability and recovery guidance, not a fake unlock of a still-running Python worker thread.

Rationale: skills are capability growth, but execution must require fresh review, content hash match, enablement, grants, and dependency readiness. Native skills are bundled examples/core surfaces; editable marketplace/user skills live in the data plane.

### MCP and browser-facing external tools

`mcp_client.py` manages HTTP/SSE MCP servers configured in Settings. In v6.17,
discovered MCP tools are part of the selected initial capability envelope when MCP
is enabled; if discovery fails, `list_available_tools` reports an explicit
capability omission manifest instead of silently hiding the surface. MCP tools run
through safety and wrap untrusted descriptions/results. Browser tools are stateful
and thread-sticky because browser automation has session/greenlet affinity. They
support Chromium by default and WebKit plus Playwright device descriptors for
mobile/iOS-grade checks. When an actual Safari/iOS risk warrants that extra
coverage, request `engine=webkit` with an iPhone device descriptor rather than
treating a narrow Chromium viewport as Safari-equivalent; it is not a global
acceptance requirement. macOS packages ship bundled Chromium headless shell and
load WebKit through the managed Playwright cache on first `engine=webkit` use;
Linux, Docker, and Windows packages bundle both Chromium and WebKit. PR
helpers (`tools/git_pr.py`, `tools/github.py`) are first-party built-ins in the
normal parent envelope, but remain blocked by workspace/local-readonly/heal mode
guards when they would mutate local state outside their allowed scope.

### Budget tracking

`ouroboros/usage_accounting.py` is the single monetary authority for core-mediated model work. Every physical provider send receives a unique attempt id and append-only transitions `reserved → dispatched → settled | unresolved` (or `reserved → released` before dispatch); a retry is a new attempt. The same wrapper covers ordinary/direct calls, children, planning/scouts, task acceptance, triad/scope/advisory/safety/skill review, synthesis, background consciousness, transport/format retries, and opaque SDK calls. Opaque adapters reserve their available `max_budget_usd` and settle from provider `total_cost_usd`. A reviewed external script/extension is recorded as unknown/unmetered at each host-observed opaque execution boundary only when the declared allowlist, a fresh content-hash-bound grant, and a non-empty model-provider credential agree; PluginAPI settings access additionally requires `read_settings`. This covers script/OOP and in-process tool/route/WS dispatch, in-process register/event/supervised/unload callbacks, and each companion spawn/restart; ordinary or GitHub-only skills do not poison `cost_final`.

Before dispatching any post-task consolidation or synthesis worker, a root task reads the existing ledger once and freezes one shared non-final subtree snapshot for summary and reflection. It includes `cost_usd_with_children`, reserved spend, unresolved upper bound, unknown/unmetered count, ledger integrity, `cost_snapshot_at`, `cost_final=false`, and `cost_with_children_partial=true`; a read failure is explicit unavailable/null, never `$0`. Summary and reflection receive the same snapshot. Consolidation, summary, and reflection model calls are included only by the existing terminal checkpoint, which remains the sole final cost authority; there is no second accounting ledger or reconciliation LLM call.

Before dispatch, pricing performs a route-specific best-effort lookup bounded to
five seconds: OpenRouter's full `/models` catalog (all families) or cloud.ru's
catalog with an explicitly configured RUB/USD divisor. Only the exact normalized
model id and provider-supplied fields are used; there is no manual model table,
prefix inheritance, cache-price multiplier, or numeric fallback. Direct OpenAI,
OpenAI-compatible, Anthropic, and GigaChat routes without an automatic source are
honestly unknown.

Unknown price is fail-open for model admission under finite global/root budgets:
the attempt reserves `None`, dispatches while already-known accounted spend remains
below the limit, and settles from provider-reported cost or the fetched exact price.
If neither exists, `cost=None` and `cost_final=false` remain durable. This is not a
budget bypass: an already-exhausted known budget and any known reservation that
would exceed the remaining limit still block before dispatch.

A raised provider send is terminalized honestly: an OpenRouter router-side 404 ("No endpoints found …" — routing failed BEFORE any upstream generation, nothing billed) settles at a confirmed $0 with `settle_reason="pre_routing_rejection"`, releasing its conservative reservation instead of holding a phantom unresolved bound against the budget forever (the same blind-spot class as the v6.65.4 zero-usage body-error fix); an OpenRouter ToS-policy 403 ("prohibited due to a violation of provider Terms Of Service", HTTP 403, matched structurally-or-textually and raised BEFORE any generation; the audited CLB run showed 0 llm_usage events and 0 billed tokens while accumulating a ~$479 phantom bound) likewise settles at a confirmed $0 with `settle_reason="tos_rejection"`; every other raised failure — including generic 401/403/quota errors without these exact signatures — keeps today's unresolved upper bound. `review_wave_admission` is the read-only pre-flight for task-level review waves (skill review, plan review, task acceptance — never the P3 commit gate): it reuses the exact per-attempt reservation math plus the root-projection remainder, and a wave that cannot fit is declined BEFORE the first reviewer call with a typed `review_wave_budget_insufficient` event — the surface finalizes honestly (skill stays pending, plan returns `PLAN_REVIEW_SKIPPED_BUDGET`, acceptance records a terminal DEGRADED) instead of dying mid-wave with paid partial slots. It is fail-open on unknown pricing/limits, mirroring `reserve_attempt`'s unknown-price stance. Budget check, sequence validation, append, and fsync occur under one short cross-process lock; network I/O never holds it. The projection conservatively counts settled cost + live reservations + unresolved upper bounds before each dispatch, both globally and per root. Reservations use provider-declared piecewise-linear price tiers; for OpenAI-family sends through OpenAI/OpenRouter the chars-based prompt estimate gets a documented 1.10 empirical tokenizer envelope before tier selection, while settlement always uses actual provider usage/cost. Unknown price remains unknown. A corrupt final JSONL row is quarantined with a loud event while the validated prefix stays available; every affected projection becomes integrity-degraded and `cost_final=false` because the lost tail may have been paid. Earlier structural corruption fails closed. If settlement persistence fails after a paid response, the result is preserved and the attempt stays dispatched/unresolved rather than disappearing. New sends stop after the known limit, while already-dispatched work may finish and cause a small honest overrun. A refused root dispatch installs one durable admission marker; it does not scan, mirror, or reclassify the subtree, whose members independently meet the same ledger rail before another dispatch. Resume explicitly clears that marker only after proving the nominated task has not dispatched (or has a typed replay-safe checkpoint); otherwise cancel or start a new task.

`state.json`, `/api/state`, `/api/cost-breakdown`, task results, and `llm_usage` remain compatibility projections carrying ledger attempt ids; they never become a second charge source. Root totals aggregate the subtree and review/safety/post-task work exactly once. A late cosmetic project-naming settlement refreshes the terminal root projection without reviving a timed-out title/bubble; `task_cost_finalized` refreshes history/cards without creating chat or unread. Owner-visible chat budget lines and startup budget checks replay this ledger instead of trusting a stale state mirror, and fail loudly if replay is unavailable. Split-drive workers receive the canonical `budget_drive_root` before their startup check, so they never report a child-local `$0` projection. Owner surfaces distinguish Accounted/Limit, confirmed/settled, reserved, unresolved upper bound, unknown/unmetered, `cost_final`, and a loud ledger-integrity warning. Quarantine evidence marks the projection integrity-degraded, non-final, and non-replay-safe even when the validated prefix remains readable.

Startup runs one resumable importer with a source-hash watermark. It writes immutable copies of legacy `events.jsonl` and `state.json`, records their hashes plus a before hash of `settings.json`, imports only real usage rows as attempts, keeps ambiguous calls as `attempt_counts.metadata_only` without treating the call-count gap as monetary unknown work or making `cost_final` false, represents any state-vs-event monetary remainder as a disclosed legacy delta, never rewrites source logs, and does not fabricate attempts. Secret settings contents are neither archived nor modified. Any one-host forensic repair of an already completed watermark is an operator migration, not a second runtime reconciliation API.

## 7. Configuration (ouroboros/config.py)

Single source of truth for:
- **Paths**: HOME, APP_ROOT, REPO_DIR, DATA_DIR, SETTINGS_PATH, PID_FILE, PORT_FILE
- **Constants**: RESTART_EXIT_CODE (42), AGENT_SERVER_PORT (8765)
- **Settings defaults**: all model names, budget, timeouts, worker count
- **Functions**: `load_settings()`, `save_settings()`,
  `apply_settings_to_env()` (copies hot-reloadable/runtime keys — models, API keys,
  GitHub integration settings, review/effort settings, local-model config,
  and the Phase 2 three-layer-refactor axes
  `OUROBOROS_RUNTIME_MODE` + `OUROBOROS_SKILLS_REPO_PATH` — from the
  settings dict into `os.environ`),
  `normalize_runtime_mode()` (SSOT clamp for `OUROBOROS_RUNTIME_MODE`,
  shared by the save path in `server.py::api_settings_post`, the read
  path in `_coerce_setting_value`, and onboarding validation in
  `ouroboros/onboarding_wizard.py::prepare_onboarding_settings`),
  `get_runtime_mode()` / `get_skills_repo_path()` (read-side helpers
  used by `gateway/state.py::api_state`),
  `acquire_pid_lock()`, `release_pid_lock()`

Settings file: `~/Ouroboros/data/settings.json`. File-locked for concurrent access.

### LLM output token budgets

Ouroboros uses provider-specific names for the same output-token budget:
OpenRouter/Anthropic-compatible calls send `max_tokens`; direct OpenAI GPT-5
calls send `max_completion_tokens` through `LLMClient._build_remote_kwargs`.
Runtime floors:

| Surface | Output-token budget |
|---------|---------------------|
| `LLMClient.chat()` / `chat_async()` defaults | 65,536 |
| Main task loop (`loop_llm_call.MAIN_LOOP_MAX_TOKENS`) | 65,536 |
| `LLMClient.vision_query()` and VLM tools (`analyze_screenshot`, `vlm_query`) | 32,768 |
| Review synthesis dedup | 16,384 |
| Chat block consolidation, era compression, scratchpad consolidation | 16,384 |
| Execution reflection and pattern-register update | 16,384 |
| Improvement-backlog grooming (`improvement_backlog.groom_backlog`) | 8,192 |
| Post-task evolution promotion decision (`post_task_evolution`) | 8,192 |
| Task summary and chat/history summary tool | 16,384 |
| Context compaction round summaries | 32,768 |
| Skill publish PR body generation | 8,192 |
| Background consciousness loop | 65,536 |
| Project naming LIGHT one-shot (`project_naming.llm_project_name`) | 256 |

### Default settings

| Key | Default | Description |
|-----|---------|-------------|
| OPENROUTER_API_KEY | "" | Optional. Default multi-model router key |
| OPENAI_API_KEY | "" | Optional. Official OpenAI provider key (runtime + web search) |
| OPENAI_BASE_URL | "" | Optional custom/legacy OpenAI-compatible runtime base URL. Keep empty for official OpenAI `web_search`. |
| OPENAI_COMPATIBLE_API_KEY | "" | Optional. Dedicated OpenAI-compatible provider key |
| OPENAI_COMPATIBLE_BASE_URL | "" | Optional. Dedicated OpenAI-compatible provider base URL |
| CLOUDRU_FOUNDATION_MODELS_API_KEY | "" | Optional. Cloud.ru Foundation Models provider key |
| CLOUDRU_FOUNDATION_MODELS_BASE_URL | `https://foundation-models.api.cloud.ru/v1` | Cloud.ru provider base URL |
| GIGACHAT_CREDENTIALS | "" | Optional. Sber GigaChat authorization key (base64 `client_id:secret`, OAuth). Enables `gigachat::...` model values via the `gigachat` library |
| GIGACHAT_USER | "" | Optional. GigaChat basic-auth username (alternative to `GIGACHAT_CREDENTIALS`) |
| GIGACHAT_PASSWORD | "" | Optional. GigaChat basic-auth password (used with `GIGACHAT_USER`) |
| GIGACHAT_SCOPE | `GIGACHAT_API_PERS` | GigaChat API scope (`GIGACHAT_API_PERS` personal / `GIGACHAT_API_B2B` prepaid legal entity / `GIGACHAT_API_CORP` pay-as-you-go legal entity) |
| GIGACHAT_BASE_URL | `https://api.giga.chat/v1` | GigaChat API base URL for new connections (explicit legacy/internal overrides are preserved) |
| GIGACHAT_VERIFY_SSL_CERTS | `true` | Verify GigaChat TLS certs. Set `false` to skip (e.g. behind the Russian Trusted Root CA) |
| GIGACHAT_PROFANITY_CHECK | "" | Optional. `true`/`false` profanity filter; read directly by the `gigachat` library |
| ANTHROPIC_API_KEY | "" | Optional. Enables direct Anthropic runtime routing (`anthropic::...` model values) and Claude Agent SDK advisory/review internals |
| transport-skill requested bot token | "" | Optional stored secret used by the Telegram bridge skill after owner grant |
| transport-skill local chat id | "" | Optional stored setting used by the Telegram bridge skill |
| OUROBOROS_NETWORK_PASSWORD | "" | Optional. Enables the non-loopback auth gate when set; empty still allows open bind, but startup logs a warning |
| OUROBOROS_SERVER_HOST | 127.0.0.1 | Server bind host. Use `0.0.0.0` for LAN/Docker access; restart required. |
| OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD | unset | Env-only Docker/Kubernetes escape hatch. When set to `1`, Settings may save ordinary changes while a wildcard/non-localhost bind has no `OUROBOROS_NETWORK_PASSWORD`; use only behind ingress auth, VPN, private networking, or an auth proxy. |
| OUROBOROS_MODEL | x-ai/grok-4.5 | Main reasoning model (primary real default; worker slots below are empty→Main unless noted) |
| OUROBOROS_MODEL_HEAVY | "" | Strong acting/coding lane for mutative first-level subagents (`auto` routes a writing child here). Empty means use `OUROBOROS_MODEL`. (Renamed from `OUROBOROS_MODEL_CODE`; stored/legacy values migrate.) |
| OUROBOROS_MODEL_LIGHT | google/gemini-3.6-flash | Fast/cheap model for safety, compact routing, lightweight helper calls, and deep subagents. Empty means use `OUROBOROS_MODEL` (a real cheap default ships since v6.82.0) |
| OUROBOROS_MODEL_VISION | "" | Vision/caption model slot for send-time image captioning and VLM helpers. Empty means use `OUROBOROS_MODEL` for normal remote routes; local/blind routes require an explicit reachable vision slot for caption fallback. Legacy `OUROBOROS_VISION_MODEL` settings migrate here. |
| OUROBOROS_IMAGE_INPUT_MODE | auto | Image routing mode for model calls: `auto` keeps inline images for vision-capable active models and captions for blind models; `caption` always replaces image blocks with text captions; `inline` sends pixels only when supported; `off` replaces images with placeholders. |
| OUROBOROS_VISION_CAPTION_TIMEOUT_SEC | 90 | Provider timeout for send-time image caption sub-calls (`vision_routing.py`); keeps caption fallback from occupying the main loop indefinitely. |
| OUROBOROS_MODEL_CONSCIOUSNESS | "" | Background Consciousness model slot. Empty means use `OUROBOROS_MODEL`; do not silently downgrade this lane to the light model or a smaller context as a cost optimization |
| OUROBOROS_MODEL_FALLBACKS | openai/gpt-5.6-luna | Comma-separated cross-model fallback chain tried when the primary returns no usable response (429-aware cooldown, deduped, active model dropped; a benchmark setting all slots to one model dedupes to a no-op). (Renamed from `OUROBOROS_MODEL_FALLBACK`; stored/legacy values migrate.) |
| OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT | 1 | Capability/cost cap (distinct from the hard nesting cap `OUROBOROS_MAX_SUBAGENT_DEPTH`): the subagent depth at/below which an explicit `main`/`heavy` lane is honored; deeper descendants fall to Light. A capped explicit request surfaces a visible note |
| OUROBOROS_MODEL_MAX_CONCURRENCY | 3 | (v6.40) Max CONCURRENT provider calls per (model, use_local) route; excess worker threads wait (deadline-bounded) instead of storming one model's rate limit (self-DoS guard, `ouroboros/model_concurrency.py`). <=0 disables. Default-on, fail-soft |
| OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC | 180 | (v6.40) Hard ceiling (seconds) a provider call waits for a concurrency slot when the task has no deadline; past it the call proceeds without a slot (never blocks forever) |
| OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC | 60 | (v6.40) Transport timeout for the LIGHT project-naming provider call (`ouroboros/project_naming.py`) |
| OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC | 8 | (v6.40) Gateway HARD wait for the inline turn-into-project name before falling back to the heuristic (`ouroboros/project_naming.py::llm_project_name_async`) |
| OUROBOROS_FALLBACK_COOLDOWN_ENABLED | true | Default-on, fail-soft. Put a model that just failed transiently (429/5xx/overloaded) on a short process-local cooldown so the fallback chain / swarm skips it briefly |
| OUROBOROS_FALLBACK_COOLDOWN_SEC | 120 | Cooldown window length (seconds) for a transiently-failed model |
| OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL | 1 | Per-fallback-candidate transient-retry cap (1–2); does not touch the primary model's same-model transient-retry budget |
| CLAUDE_CODE_MODEL | opus[1m] | Anthropic model for Claude Agent SDK advisory/review internals (values: sonnet, opus, `opus[1m]`, or full model name; the `[1m]` suffix is a Claude Code selector that requests the 1M-context extended mode) |
| OUROBOROS_MODEL_DEEP_SELF_REVIEW | openai/gpt-5.6-sol-pro | Dedicated deep self-review model slot |
| OUROBOROS_MAX_WORKERS | 10 | Worker process pool size |
| OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT | 6 | Active subagent cap per root task — readonly or acting (hard max 500 = `config.MAX_ACTIVE_SUBAGENTS_HARD_CAP`, shared with the supervisor reject/reservation gates and the `wait_tasks` id cap; decided trade-off: at ~500 children the `wait_tasks` compact projection can hit the disclosed 15K tool-result truncation — chunked waits + `get_task_result` are the follow-up path, and the O(n²) active-tree scans are accepted with no perf work) |
| OUROBOROS_MAX_SUBAGENT_DEPTH | 2 | Nested subagent depth cap (hard max 10, min 0; **0 disables delegation entirely**, including plan_task's planning scouts — root tasks still run; descendants deeper than `OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT` use the light lane) |
| OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS | (empty) | Allow mutative (acting) subagents. Empty = follow runtime mode (ON in advanced/pro, OFF in light); explicit true/false overrides. Owner-controlled. Settings exposes explicit On/Off only; the empty runtime-default state is backend/default behavior, not a third owner-facing mode. |
| OUROBOROS_SUBAGENT_WORKTREE_ROOT | (empty) | Filesystem root for acting self_worktree checkouts; empty = ~/Ouroboros/subagent_worktrees (kept outside repo/ and data/) |
| OUROBOROS_SUBAGENT_PROJECTS_ROOT | (empty) | Durable root for genesis ("from scratch") subagent projects; empty = ~/Ouroboros/projects (outside repo/ and data/). Never age-pruned. |
| OUROBOROS_DELIVERABLES_ROOT | (empty) | (v6.38.0) Visible container for UNNAMED user deliverables; empty = ~/Ouroboros/Deliverables (sibling of the projects root, outside repo/ and data/, never GC-pruned). A BARE `user_files` filename (no directory) lands here instead of cluttering the home root; an explicit placement (`Desktop/…`, `Downloads/…`, any path WITH a directory) is honored under home as given. `user_files_path_block_reason` allows this container past the workspace-overlap guard only while it stays a genuine sibling of (never overlapping/containing) the hard data/repo/budget drives. |
| OUROBOROS_GC_RETENTION_DAYS | 7 | Unified age (days) for startup garbage collection of ALL disposable runtime artifacts: acting worktrees, terminal task drives, and leftover service logs (hard max 365; math SSOT in `ouroboros/retention.py`). Deprecated per-subsystem retention keys are migrated into this on settings load. |
| OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC | 120 | Poll slice for required `plan_task` planning subagents; every started scout remains eligible until terminal state or the shared max-wait cutoff. |
| OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC | 900 | Shared terminal-or-cutoff ceiling for one planning fingerprint. At the boundary every ready non-empty handoff and every precise omission go directly to the configured reviewer panel; capacity, scheduling failure, and a normal cutoff do not trigger an extra inline model call. Lower values apply as-is; values above the default are clamped to the `plan_task` tool/wrapper budget (raise those module constants to extend the real ceiling). |
| OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC | 120 | One-minor deprecated no-op retained for settings/env compatibility. The shared planning boundary never stops on heartbeat staleness; a non-default legacy value emits `deprecated_settings_ignored`. |
| OUROBOROS_RESTART_DRAIN_MAX_SEC | 120 | Agent-requested restarts drain first: while any RUNNING task still heartbeats, the restart waits up to this many seconds before proceeding fail-closed (0 = restart immediately). Owner restarts are not drained. |
| TOTAL_BUDGET | 10.0 | Total budget in USD |
| OUROBOROS_PER_TASK_COST_USD | 20.0 | Per-task soft threshold in USD |
| OUROBOROS_RUB_USD_RATE | (empty) | Explicit RUB→USD divisor for cloud.ru catalog token costs. Empty/invalid means cloud.ru cost is unknown; there is no implicit FX fallback. |
| OUROBOROS_PRICING_TTL_SEC | 21600 | Live-pricing (OpenRouter + cloud.ru catalog) refetch interval in seconds; prices/FX drift |
| OUROBOROS_TOOL_TIMEOUT_SEC | 600 | Global tool timeout override (read live from settings.json on each tool call) |
| OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC | 1800 | Upper bound (seconds) for an explicit per-call `run_command`/`run_script` `timeout_sec`/`timeout` override (v6.35.0). The handler clamps the requested value to this ceiling and to half the remaining task deadline; the matching outer tool-execution timeout rises to the same ceiling (plus a small margin) so a long approved command is not cut off by the static entry cap. |
| OUROBOROS_FINALIZATION_GRACE_SEC | 120 | Grace window before hard task termination becomes final. The supervisor clamps this setting to 0-300 seconds and uses it to let headless/workspace artifact finalization, verifier handoff, and honest terminal result writing complete before process teardown. |
| OUROBOROS_WEBSEARCH_MODEL | gpt-5.2 | Official OpenAI Responses model for `web_search` when `OPENAI_BASE_URL` is empty |
| OUROBOROS_WEBSEARCH_BACKEND | auto | Force a first-party `web_search` tool backend regardless of which keys are present: `auto` (default OpenAI-first cascade) \| `ddgs` (pure retrieval, no second LLM — for fixed-model runs) \| `openai` \| `openrouter` \| `anthropic`. A pin to a non-LLM backend keeps the active model the only reasoner. |
| OUROBOROS_MAIN_WEB_SEARCH | off | Opt-in main-loop provider server web search. `off` (default) preserves provider independence; `openrouter` injects OpenRouter's `openrouter:web_search` server tool into the **main** OpenRouter solve-model request, so the same model decides when/how to search. This is a transport setting, not a ToolRegistry function tool, and must be disclosed in fixed-model benchmarks. |
| OUROBOROS_MAIN_WEB_SEARCH_ENGINE | auto | OpenRouter server-web engine for `OUROBOROS_MAIN_WEB_SEARCH=openrouter` (`auto`, `native`, `exa`, `parallel`, etc.; OpenRouter support varies). |
| OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS | 10 | Total search-result cap across one main-model request for OpenRouter server-web. |
| OUROBOROS_OR_PROVIDER | "" | OpenRouter `provider` routing (v6.46.0): `resilience` (same-model failover on rate-limit/5xx, prompt-cache stays warm), `repro` (pin, no failover — fixed-model benchmark runs), or a raw JSON provider object. Gap-merged so it never overrides the anthropic `require_parameters` pin or the (unverified-family) reasoning-continuity `allow_fallbacks=false` pin; affects same-model provider routing only (never the model, so the P3 reviewer floor is untouched). |
| OUROBOROS_SEARCH_CODE_WALL_SEC | 45 | Total wall-clock budget for one `search_code` call, bounding BOTH the directory-walk enumeration and the batched rg loop so a search whose root resolves to a very large tree cannot run unbounded. |
| OUROBOROS_USER_FILES_ROOT | "" (home) | **Env-only operational override** (NOT a `settings.json`/UI carrier — like `OUROBOROS_DATA_DIR`; deliberately absent from `SETTINGS_DEFAULTS`/`apply_settings_to_env`, whose pop-on-absent would erase an injected value). Filesystem base for the `user_files` resource root, read directly by `tool_access._user_files_root`. Defaults to the owner's real home; a jailed/benchmark runtime sets a scratch dir so a task cannot read the owner's real home (e.g. secret files), and unnamed deliverables then derive under that jail (`tool_access._deliverables_root`). Any unusable value falls back to home (fail-safe). |
| OUROBOROS_OBSERVABILITY_KEEP_RAW | unset | **Env-only operator debug override** (NOT a `settings.json`/UI carrier — deliberately absent from `SETTINGS_DEFAULTS`/`apply_settings_to_env` so a self-change or non-owner save can NEVER enable secret logging). When set, persist the RAW LLM/tool payload as the authoritative observability blob. Default OFF: the authoritative blob is REDACTED (secret values masked, structure/route/non-secret text preserved per BIBLE P1) so no secret lands on disk; `full_payload_redacted` declares it honestly. |
| OUROBOROS_GENERATIVE_PROBE | 1 (on) | Enables the generative context-window probe (`capability_evidence`): on an explicit Max toggle/Save, when provider metadata gives no window, an over-window request empirically confirms ≥1M from a FREE pre-inference reject. `OUROBOROS_GENERATIVE_PROBE_CHARS` (default 5,000,000) sizes the padding. A 200 (possibly-paid accept) never auto-confirms — it routes to owner-ack. |
| OUROBOROS_REVIEW_MODELS | openai/gpt-5.6-luna,google/gemini-3.6-flash,anthropic/claude-sonnet-5 | Ordered reviewer slots shared by triad/plan/task/skill review; duplicate model IDs are independent slots |
| OUROBOROS_SCOPE_REVIEW_MODELS | openai/gpt-5.6-terra | Comma-separated scope reviewer slots; falls back from legacy `OUROBOROS_SCOPE_REVIEW_MODEL`. Designated-default window evidence: OpenRouter `/models` metadata reports gpt-5.6-terra context_length=1,050,000 (checked 2026-07-29) — satisfies the BIBLE P3 ≥1M floor |
| OUROBOROS_TASK_REVIEW_MODE | auto | Task acceptance mode: `off`, `auto`, or `required`. Only the root owns the verdict. `auto` and `required` host-review queued/headless/scheduled substantive roots and effectful direct turns; pure conversation and child authority are skipped. `required` combines with `OUROBOROS_REVIEW_ENFORCEMENT`: advisory records/finalizes honestly after available improvement, while blocking repeats until evidence-backed clean acceptance or a real rail, subject to any explicit task-local pass cap. |
| OUROBOROS_SAFETY_MODE | full | (v6.54.3) Owner-only LLM-safety-supervisor coverage: `full` (every guarded call checked) \| `light` (LLM check only for POLICY_CHECK integration tools; CONDITIONAL shell/verify fall to deterministic guards) \| `off` (no LLM safety calls). Deterministic sandbox/protected paths/light-mode guards stay ON in every mode; non-full modes emit durable `safety_mode_skip` audit events. Changed ONLY via the audited `/api/owner/safety-mode` endpoint; generic settings writes drop it and `save_settings` refuses lowering. (v6.82.0) `light` is the default only for a FRESH desktop-wizard setup whose settings file carries no prior choice; the shipped default and every fail-closed fallback stay `full`, so existing installs and headless/docker/web installs that skip the wizard-authored payload keep Full. |
| OUROBOROS_SAFETY_MAX_TOKENS | 2000 | (v6.54.3) Output-token budget for safety-supervisor LLM calls — the parse-bug fix: without it a reasoning light model could burn the whole budget on hidden reasoning and return an empty/1-token body that failed JSON parse and fail-closed blocked a benign command. |
| OUROBOROS_SAFETY_CALL_TIMEOUT_SEC | 60 | (v6.54.3) Transport timeout for safety-supervisor LLM calls. |
| OUROBOROS_WEBSEARCH_TIMEOUT_SEC | 480 | (v6.54.3) Explicit transport timeout for the `web_search` OpenAI streaming call (below the 540s ToolEntry outer cap so transport failures are cleanly messaged, not thread-killed). |
| OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC | 2700 | (v6.54.3) Default httpx read/write timeout for no_proxy LLM clients (was a hardcoded 3600). Deliberately generous — long silent reasoning (scope review, deep self-review) can take 20-40 min before the first byte; this is the dead-socket bound, and explicit per-call timeouts always win. |
| OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC | 300 | (v6.54.3) plan_task deadline scaling floor: with a task deadline the planning-swarm ceiling is min(configured ceiling, remaining/4); below this floor plan_task returns a typed `PLAN_TASK_SKIPPED_DEADLINE` + `plan_task_deadline_skip` telemetry instead of eating the budget tail. Without a deadline behavior is unchanged. |
| OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC | 200 | Initial acceptance-review estimate and floor. The first review reserves at least 200s; later passes use `max(floor, 1.5×EWMA)` with `alpha=0.5`, reconstructed from existing timing events. |
| OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES | 1 | Default local count cap outside Required+Blocking. An explicit `task_contract.budget_profile.max_improvement_passes` binds under every policy; Required+Blocking with no explicit cap has no local count cap, but deadline/budget/lifecycle rails still bind. `until_deadline` remains a one-minor compatibility alias. |
| OUROBOROS_ACCEPTANCE_RESERVE_PCT | 5 | (v6.54.4) Default finalization-reserve percentage; reserve = max(OUROBOROS_FINALIZATION_GRACE_SEC, pct×total budget). |
| OUROBOROS_OBSERVABILITY_RETENTION_DAYS | unset | Deprecated audit knob for private observability manifests/blobs; forensic replay blobs are kept compressed indefinitely |
| OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC | 600 | Env-only override read directly by `ouroboros.tools.review`. Per-reviewer model call timeout for multi-model review; timed-out reviewers become ERROR actors and quorum is adaptive to the configured reviewer count (`config.adaptive_quorum`). |
| OUROBOROS_REVIEW_MAX_TOKENS | 65536 | Env-only override read directly by `ouroboros.tools.review` (v6.61.1). Reviewer RESPONSE reservation for multi-model review; an operator may LOWER it (floor 8192, never above the default) when a mega-diff's input pack plus the default output reservation exceeds a reviewer endpoint's context cap — preserving full review input instead of trimming evidence. Reviewer models are never changed by this knob. |
| OUROBOROS_REVIEW_ENFORCEMENT | advisory | Review enforcement: `blocking` blocks commit critical findings, fresh-advisory open obligations/debts, and skill `blockers`; `advisory` downgrades those to warnings by operator choice. Fresh advisory with open obligations/debts writes `advisory_obligations_acknowledged`; stale advisory still blocks. Skill `warnings` do not block execution in either mode. |
| OUROBOROS_PREFLIGHT_TIMEOUT_SEC | 300 | Wall-clock timeout (seconds) for the hermetic reviewed-change pytest preflight (`preflight_runner.run_hermetic_pytest`), the single source shared by the review preflight (`review_helpers`) and the pre-push gate (`tools/git.py`). On timeout (or any crash/exception path) the runner guarantees full process-tree teardown — process group, recursive PID tree, captured escaped-session groups, and a temp-root command-line sweep — so no orphaned test processes survive. |
| OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS | true | Owner-confirmed setting; default-on as of v6.10.0 (installs without an explicit choice are enabled; existing explicit choices are preserved). When enabled, a fresh executable skill review grants only the manifest-declared settings keys and host permissions for that exact content hash so closed-loop skill development can run without repeated manual grants. Under `blocking`, blocker reviews are not executable and do not auto-grant; under `advisory`, blocker findings may auto-grant only because the current enforcement mode makes the review executable. Plain `/api/settings` POST drops this key; desktop uses the launcher confirmation bridge and web uses `/api/owner/auto-grant`. |
| OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS | true | Named, hash-pinned, audited exception to manual first review (v6.31.0, CHECKLISTS §Skills): when the LAUNCHER writes a bundled native skill payload (bootstrap seed, post-bootstrap new seed, version resync — all `.seed-origin`-marked), it stamps `review.json` `status=clean` with `reviewer_models=["repo_commit_gate"]` and `review_profile="native_seed"` because those exact bytes passed the repo triad+scope commit gate; zero-grant skills (no secret keys, no privileged permissions, only tool/subprocess surface) also auto-enable when no explicit owner enable/disable choice exists yet. Any later payload edit flips the verdict stale (only the TOP-LEVEL `.seed-origin` marker is hash-exempt), and a `native_seed` verdict whose marker is gone reads back as pending. The flag acts ONLY at launcher seed/resync moments over repo-reviewed bytes — flipping it at runtime grants nothing until the next launcher seeding event, which is why it ships without a dedicated owner-only settings gate (the generic settings self-change detectors still apply). Set to false to keep manual review for native seeds. |
| OUROBOROS_CONTEXT_MODE | max | Owner-selected context horizon: `max` or `low`. Ordinary Main calls build deterministic route-calibrated projections from one immutable core; an unknown route tries Max (never a silent 200K fallback), and a confirmed overflow may retry the same model once with task-local Low without mutating this global setting. Since v6.80.0 this key is ALSO the single control over BIBLE P3 scope-review applicability (`max`: blocking ≥1M scope gate; `low`: whole-repository scope review declaredly not performed, typed skip row recorded) — an owner policy coupling, not a structural limit; the triad's blocking staged-diff review is unaffected in both modes. Plain `/api/settings` POST drops this key; owner endpoints/CLI control it. |
| OUROBOROS_RUNTIME_MODE | advanced | Three-layer refactor axis: `light`, `advanced`, or `pro`. Orthogonal to `OUROBOROS_REVIEW_ENFORCEMENT`. Clamped via `normalize_runtime_mode` on both save and read paths. `light` is a compatibility/self-modification guard: it blocks repo-mutation tools at the `ToolRegistry.execute` gate, mutative direct git through `run_command`, shallow argv writer commands with explicit repo-local targets, and post-execution repo dirtiness from `run_command` (`LIGHT_MODE_REPO_WRITE_BLOCKED`, no automatic rollback). It also refuses runtime_mode self-elevation through the owner chokepoints (`save_settings`, `_data_write` settings.json block, `/api/settings` POST drop). Reviewed + enabled skills (script + extension) execute in light. `advanced` can evolve the application layer but blocks protected core/contract/release paths. `pro` may edit those protected surfaces directly, but committing them still requires the normal triad + scope review gate, whose blocking/advisory behavior follows `OUROBOROS_REVIEW_ENFORCEMENT`. Runtime mode is owner-only: desktop uses native confirmation, while web uses `/api/owner/runtime-mode` to persist the next-boot value; neither mutates the current boot baseline. |
| OUROBOROS_SKILLS_REPO_PATH | "" | Local checkout path for the external skills/extensions repo. Consumed by `ouroboros.skill_loader.discover_skills` (Phase 3); accepts absolute paths or `~`-prefixed paths; `get_skills_repo_path` expands `~` at read time. Ouroboros never clones/pulls this directory. |
| MCP_ENABLED | false | Optional. Enables the base-runtime HTTP/SSE MCP tool client. |
| MCP_SERVERS | [] | List of MCP server config dicts persisted in settings.json; not propagated through env. |
| MCP_TOOL_TIMEOUT_SEC | 60 | Per-tool timeout for MCP discovery and tool calls. |
| OUROBOROS_HUB_CATALOG_URL | `https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json` | Official static skill catalog. The client fetches only this JSON automatically; selected skill installs download the catalog-listed files and verify sha256. |
| OUROBOROS_SCOPE_REVIEW_MODEL | openai/gpt-5.6-terra | Legacy singular fallback for `OUROBOROS_SCOPE_REVIEW_MODELS`; kept for existing settings files |
| OUROBOROS_EFFORT_TASK | medium | Reasoning effort for task/chat. Full scale (config.EFFORT_SCALE, v6.57.0): none, minimal, low, medium, high, xhigh, max — xhigh/max clamp down to each model's learned ceiling, and (v6.73.2) none/minimal clamp UP to a learned floor on reasoning-mandatory endpoints, each with a disclosed `reasoning_effort_clamped` usage note (reason `learned_ceiling`/`learned_floor`); the Settings UI offers all tiers except `minimal` (a per-call tactical tier, not a standing default) |
| OUROBOROS_EFFORT_EVOLUTION | high | Reasoning effort for evolution tasks |
| OUROBOROS_EFFORT_REVIEW | high | Reasoning effort for review tasks |
| OUROBOROS_EFFORT_SCOPE_REVIEW | high | Reasoning effort for scope review |
| OUROBOROS_EFFORT_DEEP_SELF_REVIEW | high | Reasoning effort for deep self-review |
| OUROBOROS_EFFORT_CONSCIOUSNESS | high | Reasoning effort for background consciousness |
| OUROBOROS_RETURN_REASONING | true | OpenRouter reasoning continuity switch. Unset means return reasoning payloads by default; false-like values or an explicit empty string opt out. Direct/local routes strip OpenRouter-only reasoning fields on copied payloads. |
| OUROBOROS_REASONING_SUMMARY | auto | Narration display switch. `auto` (default) narrates an otherwise-empty tool-round bubble with readable reasoning the provider already returned (`LLMClient.extract_display_reasoning`, shape-based: flat `reasoning` / `reasoning_details` of readable types / Anthropic `thinking` / Gemini `part.thought`; opaque/encrypted skipped). `off` disables the fallback. DISPLAY-ONLY — never added to the transcript or sent back to a provider, so it cannot affect round-trip. Verified against live gpt-5.5, which returns a readable `reasoning.summary` alongside the encrypted block. |
| OUROBOROS_SOFT_TIMEOUT_SEC | 600 | One-minor deprecated no-op retained for settings/env compatibility; a non-default legacy value emits a deprecation event. No user heartbeat/status control is rendered from it. |
| OUROBOROS_HARD_TIMEOUT_SEC | 1800 | One-minor deprecated no-op retained for settings/env compatibility; a non-default legacy value emits a deprecation event. Task termination is governed by idle/absolute-ceiling/deadline/budget rails. |
| OUROBOROS_TASK_IDLE_TIMEOUT_SEC | 900 | (v6.38.0) Activity-based idle window: a task is stopped only after it has made NO real progress (`llm_usage`/progress events — NOT the unconditional 30s liveness heartbeat) AND has no progressing/queued subtree for this long. Effective value is floored to the per-call timeout ceiling (`max(idle, per_call_ceiling+120)`) so a single legitimate long tool/LLM call is never idle-killed mid-work. |
| OUROBOROS_TASK_ABS_CEILING_SEC | 21600 | (v6.38.0) Absolute per-task wall-clock backstop (6h), independent of activity — the unconditional safety ceiling. Together with an explicit `deadline_at` (a deliberate cap, honored promptly even while progressing) and the budget axis, these are the ONLY hard task-termination axes. |
| OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC | 90 | (v6.34.0, WS3) Dedicated-thread liveness watchdog deadline. If the supervisor loop tick OR an in-process direct-chat turn's heartbeat goes silent for longer than this, the watchdog surfaces the stall to the owner (detect + alert + `/restart` recommendation). It does NOT free the chat-agent lock / lane admission in-process (the wedged turn holds the lock; out-of-process kill deferred). Must exceed the ~0.5s tick / 30s healthy heartbeat cadence. |
| OUROBOROS_PACING_INTERVAL_SEC | 600 | (v6.34.0, CW9) Pacing interval (seconds) registered in the settings/env SSOT with the other numeric timeouts, per the DEVELOPMENT.md numeric-timeout-SSOT rule (no inline literals). |
| LOCAL_MODEL_SOURCE | "" | HuggingFace repo for local model |
| LOCAL_MODEL_FILENAME | "" | GGUF filename within repo. Accepts subfolder paths (`quant/model.gguf`) and split GGUF patterns (`quant/model-00001-of-00003.gguf`). All shards are downloaded automatically; specify the first shard. |
| LOCAL_MODEL_CONTEXT_LENGTH | 16384 | Context window for local model |
| LOCAL_MODEL_N_GPU_LAYERS | 0 | GPU layers (-1=all, 0=CPU/mmap) |
| USE_LOCAL_MAIN | false | Route main model to local server |
| USE_LOCAL_HEAVY | false | Route heavy model to local server |
| USE_LOCAL_LIGHT | false | Route light model to local server |
| USE_LOCAL_CONSCIOUSNESS | false | Route background consciousness model slot to local server |
| USE_LOCAL_FALLBACK | false | Route fallback model to local server |
| OUROBOROS_MAX_ROUNDS | 200 | Main-loop LLM round ceiling per task (hot-reloadable) |
| OUROBOROS_TRANSIENT_RETRY_MAX | 6 | Same-model attempt budget for transient provider failures (finish_reason=null, 429/5xx); floored at the base retry budget |
| OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC | 1800 | Skill lifecycle lane deadline before a wedged job fails loudly |
| OUROBOROS_BG_MAX_ROUNDS | 10 | Max LLM rounds per consciousness cycle |
| OUROBOROS_BG_WAKEUP_MIN | 30 | Min wakeup interval (seconds) |
| OUROBOROS_BG_WAKEUP_MAX | 7200 | Max wakeup interval (seconds) |
| OUROBOROS_POST_TASK_EVOLUTION | false | Owner-gated, default-OFF post-task self-evolution envelope (V4). The Settings UI presents this together with cadence as one Self-Improvement Trigger selector, but the persisted backend shape remains this boolean plus `OUROBOROS_POST_TASK_EVOLUTION_CADENCE`. When enabled, after an eligible task the worker may ask the main-model slot (medium effort — choosing the next evolution objective is a high-leverage decision, upgraded off the light lane in v6.30.0) whether to promote ONE improvement into the existing gated evolution campaign; it writes a durable request and the supervisor applies it later on an idle tick through the normal gates. Eligibility intentionally includes ordinary/trivial tasks; `every_n:1` means Ouroboros considers evolution after every eligible task. The agent's self-enable channels are blocked by shell/browser/settings/data-write guards plus SAFETY. |
| OUROBOROS_POST_TASK_EVOLUTION_CADENCE | llm | Post-task self-improvement trigger cadence: `llm` (after each eligible task, LLM decides whether to promote) or `every_n:<k>` (the counter is due every k eligible tasks, with `k=1` meaning every task). Unknown/malformed values normalize to `llm`; Off is represented by `OUROBOROS_POST_TASK_EVOLUTION=false`. |
| OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD | 0.0 | Optional start-floor for post-task cycles; if >0 a post-task cycle starts only when at least this much global budget remains. `0` means rely on the normal gates. Running evolution tasks still inherit the normal per-task soft cost note (`OUROBOROS_PER_TASK_COST_USD`) and global budget guards; there is no separate per-evolution-cycle cost cap. |
| OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE | "" | Optional owner standing steer appended (as a non-overriding bias) to EVERY evolution campaign's objective (`supervisor/evolution_lifecycle.py::build_evolution_task_text`), not only post-task ones; it never overrides the LLM-first promotion, and any biased cycle still passes full triad+scope review. Empty = pure LLM choice. Because it steers self-evolution, it is owner-only like `OUROBOROS_POST_TASK_EVOLUTION` — the same shell + browser-JS + POST-`/api/settings`-route self-change detectors and SAFETY.md cover it, so the agent cannot self-set it. |
| LOCAL_MODEL_PORT | 8766 | Port for local llama-cpp server |
| OUROBOROS_HOST_SERVICE_PORT | 8767 | Loopback-only Host Service API port used by reviewed skills/companions to call back into the host. Must not be exposed in Docker/LAN port mappings. |
| LOCAL_MODEL_CHAT_FORMAT | "" | Chat format for local model (`""` = auto-detect) |
| GITHUB_TOKEN | "" | Optional. GitHub PAT for remote sync |
| GITHUB_REPO | "" | Optional. GitHub repo (owner/name) for sync |
| OUROBOROS_FILE_BROWSER_DEFAULT | "" | Explicit Files tab root. Required for Docker/non-localhost Files access |

Direct-provider review fallback (formerly OpenAI-only review fallback): when exactly one official direct provider is configured, `config.get_review_models()` can fall back to `[main, light, light]` using provider-prefixed model IDs. Current scope covers official OpenAI, Anthropic, Cloud.ru, and GigaChat; `_exclusive_direct_remote_provider_env` returns empty when OpenRouter, legacy `OPENAI_BASE_URL`, OpenAI-compatible keys, or multiple official direct providers are present. The fallback also requires `provider_models.migrate_model_value` to make the main model already start with the exclusive provider prefix, preventing cross-provider free-text models from silently entering the direct-provider path. This direct-provider self-sufficiency is part of the single-provider independence invariant (see docs/DEVELOPMENT.md "Provider Independence").

GigaChat provider specifics (`gigachat::`): GigaChat is routed through the native `gigachat` library (NOT OpenAI-compatible) in `llm.py::_chat_gigachat`. OpenAI `tools` map to GigaChat `functions`; GigaChat returns at most ONE `function_call` per turn, so parallel OpenAI `tool_calls` collapse to the first. Role `tool` results become role `function` and must be valid JSON (plain text is wrapped as `{"result": ...}`); the `system` message must be first, so later system-reminders are demoted to `user`. `reasoning_effort` is intentionally omitted on the GigaChat path — GigaChat-3 can otherwise spend the whole `max_tokens` budget on hidden reasoning and return empty content/tool_calls. Fresh direct-only installs use `GigaChat-2-Max` for every ordinary/review slot: the newer `GigaChat-3-Ultra` is currently limited to personal Freemium, while Max is available across the supported personal and legal-entity tariff scopes. GigaChat exposes no automatic live cost source, so its cost remains nullable/unknown rather than coming from a hand-maintained tariff. GigaChat models are below the 1M scope-review context floor; a GigaChat-only setup fills the scope-reviewer slot with its GigaChat model exactly like the Cloud.ru direct-provider pattern. Since v6.80.0 the disclosed fallback where no ≥1M reviewer is configured is the owner-selected `low` context mode — whole-repository scope review is then declaredly not performed and every commit records a typed `skipped_low_context_mode` evidence row — replacing the removed owner-opt-in degraded advisory scope review; the blocking triad still reviews the full staged diff in both modes.

Claude Runtime Status appears when an Anthropic key exists or when backend/runtime checks or browser-side `refreshClaudeCodeStatus` transport failure paths set an error. This keeps Claude Code advisory/edit readiness visible even when the failure is UI transport rather than SDK installation.

---

## 8. Git Branching Model

- `main` is protected and not touched by Ouroboros self-modification.
- `ouroboros` is the working branch.
- `ouroboros-stable` is local recovery/fallback and is updated through `promote_to_stable`.
- Launcher-managed checkouts use the `managed` remote and update-intent markers; ordinary restarts preserve local commits and do not reset to a moving remote tip.
- External pull requests target `ouroboros`. Contributor commits are proposal
  transport and do not allocate release versions; maintainers squash-land one
  author-attributed canonical release commit on the current target, add all P9
  carriers, and run the final production review before commit/tag/push.

## 8.1 CI/CD Pipeline (`.github/workflows/ci.yml`)

CI has five tiers:

1. Quick tests on push to `ouroboros` for code/web/build paths and on fork-safe
   `pull_request` events targeting `ouroboros` (read-only permissions, no
   provider secrets, never `pull_request_target`).
2. Full matrix on stable/manual/tag.
3. Integration tests on main/ouroboros/stable/manual/tag when provider secrets exist.
4. Official-skill install smoke on stable/manual/tag: the `skill_smoke` lane installs the nine pinned official OuroborosHub skills (list in `tests/test_skill_smoke_official.py`) from the live catalog on all three OSes and validates payload/sha/provenance, manifest contract, offline preflight, real pip isolated deps, and keyless command probes; it gates release tags via the `release` job's `needs:`. An additional, ubuntu-only job step (ordered first — see below) runs the production install→review→auto-grant flow plus enable-persistence prerequisites (Tier 6, `test_review_grants_and_enable`) for a 4-skill subset through the real gateway wrapper, with Ouroboros's own skill review on one cheap stochastic reviewer slot (`google/gemini-3.5-flash`, low effort, `blocking` enforcement pinned by the test; production reviewer defaults untouched). Enable coverage is deliberately the persistence prerequisites (toggle-gate facts + `enabled.json` + `skill_readiness_for_execution`), NOT the lifecycle toggle with `reconcile_extension` — live extension loading is server runtime and would execute downloaded plugin code in the secret-bearing process. The step boundary is a security boundary and its ORDER is part of it: the secret-bearing review step runs first (it reads payload bytes but never executes them; the isolated-deps pip subprocess gets a scrubbed allowlist env), and only afterwards does the secret-free step import downloaded plugin code — so the runner never executes payload code while the secret is present. A missing key is a hard red, not a skip.
5. Build+release on `v*` tags: PyInstaller artifacts for macOS/Linux/Windows and GitHub Release.

Rationale: normal self-modification needs fast feedback, but release tags must prove cross-platform packaging AND that the live official-skill catalog still installs on this runtime (gating a release on live external services is a deliberate owner trade-off, documented in the `skill-smoke` job comment). Tag triggers are independent from branch path filters so release artifacts are always built.

### Build scripts

`build.sh`, `build_linux.sh`, `build_windows.ps1`, and `scripts/build_repo_bundle.py` are release-invariant surfaces. Changes to them must update README install/build notes and architecture rationale in the same commit.
Release tag prerequisite: platform build scripts delegate repo-bundle creation to `scripts/build_repo_bundle.py`; that Python bundler is the release-tag SSOT and verifies the annotated `v$(cat VERSION)` tag points at `HEAD` before packaged artifacts are produced. This catches untagged release builds locally instead of publishing artifacts whose version carriers disagree with git history.
Packaged Python bytecode policy (v6.36.0): platform build scripts PRECOMPILE the bundled `python-standalone` + `ouroboros` payload (`compileall -f --invalidation-mode unchecked-hash`, using the in-bundle interpreter) and SEAL the resulting `.pyc` inside the signature, replacing the old "delete all `.pyc` before signing" step — with the `.pyc` present and valid (and `unchecked-hash` skipping the source-mtime check on a read-only bundle), the runtime never writes new bytecode into the bundle, so the macOS codesign seal stays valid (a runtime `__pycache__` write previously broke the seal → AppTranslocation). `xattr -cr` + FinderInfo/detritus hygiene and `codesign --verify --strict` are preserved; hardened runtime/notarization untouched. As defense-in-depth the runtime launcher + packaged CLI set `PYTHONDONTWRITEBYTECODE=1` GLOBALLY at the earliest point and add the bytecode vars to the curated-env whitelists (`isolated_deps._SAFE_ENV_KEYS`, `extension_process_runner._child_env`) so embedded-python `pip`/`venv`/extension spawns also route caches to the external `data/state/pycache` prefix. Linux/Windows mirror the precompile for start-speed parity (they do not seal resources).

### Docker

Docker runs the web/server runtime without PyWebView. Non-loopback binding is allowed only by explicit network-gate policy (`OUROBOROS_NETWORK_PASSWORD` or trusted ingress override).

## 9. Shutdown & Process Cleanup

**Requirement: closing the window (X button or Cmd+Q) MUST leave zero orphan
processes. No zombies, no workers lingering in background.**

### 9.1 Normal Shutdown (window close)

```
1. _shutdown_event.set()           ← signal lifecycle loop to exit
2. stop_agent()
   a. SIGTERM → server.py process group on Unix / process on Windows
      │                             ← server runs its lifespan shutdown:
      │                                kill_workers(force=True) → SIGTERM+SIGKILL all workers
      │                                then server exits cleanly
   b. wait 10s for exit
   c. if still alive → SIGKILL process group/tree
3. _kill_orphaned_children()        ← SAFETY NET
   a. verify and clean data/state/server_process.json if present
   b. _kill_stale_on_port(active port + Host Service port)
   c. read data/state/extension_companions.json and kill listed companions/ports
   d. multiprocessing.active_children() → SIGKILL each
4. release_pid_lock()               ← delete ~/Ouroboros/ouroboros.pid
```

Inside `server.py` ordinary lifespan shutdown relies on graceful uvicorn
teardown for the Host Service listener; it does **not** blindly kill
`OUROBOROS_HOST_SERVICE_PORT` on normal non-restart exit. Blind port sweeping is
reserved for panic/emergency fallback and launcher-owned orphan cleanup.

### 9.2 Panic Stop (`/panic` command or Panic Stop button)

**Panic is a full emergency stop. Not a restart — a complete shutdown.**

The panic sequence (in `server.py:_execute_panic_stop()`):

```
1. consciousness.stop()             ← stop background consciousness thread
2. Save state: evolution_mode_enabled=False, bg_consciousness_enabled=False,
   evolution_owner_stopped=True, post_task_autostop=False ← owner-stop: authoritative
   vs the post-task pipeline (no autonomous re-arm next boot), then
   complete_evolution_campaign(stopped) + drop the queued promotion request
3. Write ~/Ouroboros/data/state/panic_stop.flag
4. LocalModelManager.stop_server()   ← kill local model server if running
5. kill_all_tracked_subprocesses()   ← os.killpg(SIGKILL) every tracked
   │                                    foreground subprocess process group
   │                                    (shell commands and ALL their children)
6. kill_all_foreground(data_dir)     ← stop durable executor-backed foreground
   │                                    local/docker processes
7. kill_all_services(data_dir)       ← stop service and executor-service groups
8. kill_workers(force=True)          ← SIGTERM+SIGKILL all multiprocessing workers
9. os._exit(99)                      ← immediate hard exit, kills daemon threads
```

Launcher handles exit code 99:

```
7. Launcher detects exit_code == PANIC_EXIT_CODE (99)
8. _shutdown_event.set()
9. Kill orphaned children (port sweep + multiprocessing sweep)
10. _webview_window.destroy()        ← closes PyWebView, app exits
```

On next manual launch:

```
11. auto_resume_after_restart() checks for panic_stop.flag or owner_restart_no_resume.flag
12. Flag found → skip auto-resume, delete flag
13. Agent waits for user interaction (no automatic work)
```

### 9.3 Subprocess Process Group Management

Subprocesses spawned by foreground agent tools (`run_command` and `run_script`)
use `start_new_session=True` via `_tracked_subprocess_run()` in
`ouroboros/tools/shell.py`. Task-scoped service tools use
`ouroboros/tools/services.py::_start_service`, which starts each service with
`subprocess_new_group_kwargs()` and records it in the `_SERVICES` registry.
Both paths create a separate process group for each subprocess and its children.
Executor-backed workspace processes additionally record local pids, Docker
pidfiles, and service pids under `data/state/workspace_executor_processes/` so
panic can clean them up from the server process after worker death.

On panic or timeout, the entire process tree is killed via
`os.killpg(pgid, SIGKILL)` — no orphans possible, even for deeply nested
foreground shell/script/service subprocess trees.
Panic/emergency paths call `kill_all_tracked_subprocesses()` and
`kill_all_foreground(data_dir)` plus `kill_all_services(data_dir)` without log
finalization so emergency stop remains fast; normal lifespan shutdown may pass a
drive root to `kill_all_services(drive_root)` to archive server-process service
logs before removing live log files. Services started inside worker tasks
normally finalize in `loop.py` task cleanup; forced worker termination kills the
worker process tree and archives remaining task service logs best-effort from
`data/services/<task_id>/`.

Active subprocesses are tracked in a thread-safe global set and cleaned up
automatically on completion or via `kill_all_tracked_subprocesses()` on panic.
`run_command` surfaces timeout-vs-signal distinctions in its result text so
`exit_code=-9` no longer looks like a silent success in summaries/reflections.
Claude Agent SDK gateways (`gateways/claude_code.py`) use the SDK client
lifecycle and SDK-level path/tool guards; they are not represented in
`_tracked_subprocess_run()` unless a future SDK transport exposes a first-class
child process handle. Edit-mode governance prompts are materialized into a
private task-scoped temp file before SDK option construction; SDK versions with
an explicit system-prompt-file surface receive the file path, while older
surfaces fall back to the existing string prompt without truncating BIBLE,
ARCHITECTURE, DEVELOPMENT, or checklist context. The temp file is removed on
success and failure.

---

## 10. Key Invariants

1. **Never delete BIBLE.md. Never physically delete `identity.md` file.**
   (`identity.md` content is intentionally mutable and may be radically rewritten.)
2. **Release carriers stay in sync**: `VERSION`, `web/package.json`, the README badge,
   the ARCHITECTURE header, and the latest release git tag use the same author-facing spelling
   (for example `4.50.0-rc.2` / `v4.50.0-rc.2`), while `pyproject.toml` stores the PEP 440-canonical form
   (for example `4.50.0rc2`). For packaged builds, `repo_bundle_manifest.json` pins that same
   release via `app_version`, `release_tag`, `source_sha`, and the embedded bundle hash for the
   first launcher-managed bootstrap before normal managed-remote updates resume.
3. **Config SSOT**: all settings defaults and paths live in `ouroboros/config.py`
4. **Message bus SSOT**: all messaging goes through `supervisor/message_bus.py`
5. **State locking**: `state.json` uses file locks for concurrent read-modify-write
6. **Budget authority**: `state/usage_attempts.jsonl` records every physical
   model attempt and is the only monetary authority. `llm_usage`, state, task,
   and UI totals are compatibility projections carrying attempt ids; unknown or
   unresolved spend is never represented as a false zero. Terminal-Bench replays
   this ledger for the selected root before emitting ATIF, its compatibility run
   summary, or Harbor context, so descendants and post-task attempts have the
   same token/cache/cost scope; pre-ledger artifacts retain the legacy fallback.
7. **Launcher-managed repo bootstrap**: packaged builds bootstrap from the manifest-pinned
   `repo.bundle` once, then continue from the managed git checkout. Ordinary
   restarts preserve the local branch tip; explicit Update Now is the only
   path that resets the active branch to a user-approved official SHA.
8. **Zero orphans on close**: shutdown MUST kill all child processes (see Section 9)
9. **Panic MUST kill everything**: all processes (workers, subprocesses, subprocess
   trees, consciousness, evolution) are killed and the application exits completely.
   No agent code may prevent or delay panic. See BIBLE.md Emergency Stop Invariant.
10. **Architecture documentation**: `docs/ARCHITECTURE.md` must be kept in sync with
    the codebase. Every structural change (new module, new API endpoint, new data file,
    new UI page) must be reflected here. This is the single source of truth for how
    the system works.
11. **External skills run only after a fresh executable tri-model review, and the review
    is the primary gate**: skills loaded from `OUROBOROS_SKILLS_REPO_PATH`
    may execute via the dedicated `skill_exec` substrate only when
    the skill is enabled + the live-computed tri-model review gate is
    executable (`clean`/`warnings`, plus `blockers` when enforcement is advisory) +
    the stored content hash matches the current skill payload hash
    (including the manifest-declared `entry` file for extensions).
    v5.1.2 Frame A: `OUROBOROS_RUNTIME_MODE` no longer gates skill execution
    — `light`/`advanced`/`pro` all let reviewed + enabled skills run.
    `skill_exec` additionally provides defense-in-depth via `cwd=skill_dir`,
    a scrubbed env (only `env_from_settings` allowlisted keys), a runtime
    allowlist (python/python3/bash/node/deno/ruby/go), a hard 300s timeout ceiling, output
    caps, and panic-kill tracking via `_tracked_subprocess_run` so
    `/panic` terminates the whole skill process tree. These runtime
    guards are NOT a filesystem sandbox: a malicious skill payload that
    slipped past review could still open absolute paths from inside its
    interpreter. The Skill Review Checklist items 3 (`no_repo_mutation`),
    4 (`path_confinement`), and 5 (`env_allowlist`) are therefore the
    actual authoritative checks, enforced by the multi-model reviewers
    before any `skill_exec` invocation. Skill review findings and full
    per-actor raw records live in `data/state/skills/<name>/review.json`;
    `skill_review_status.py` computes the current verdict from those
    findings as `clean`/`warnings`/`blockers`; `skill_review_gate`
    applies `OUROBOROS_REVIEW_ENFORCEMENT` when deciding executability. Skill state
    is deliberately siloed from the repo-review ledger
    (`data/state/advisory_review.json`) — a sticky skill finding cannot
    block repo commits and vice versa.
12. **Single-source startup rescue**: a dirty worktree inherited across sessions is
    rescued by exactly one mechanism — `safe_restart(..., "rescue_and_reset")` in
    `_bootstrap_supervisor_repo()`. The rescue path may clean dirty/untracked files
    back to `HEAD`, but it must not move the local `ouroboros` branch tip to the
    official managed remote unless a one-shot update-intent marker exists.
    `OuroborosAgent` construction and worker boot (including
    `_log_worker_boot_once → check_uncommitted_changes`) must never run
    `git add`/`git commit`. This keeps pytest subprocesses, A2A card builder, and
    supervisor-side `_get_chat_agent()` from stealing in-progress edits into the
    `ouroboros` branch.

---

## 11. Frozen Contracts v1 (`ouroboros/contracts/`)

Phase 1 of the three-layer refactor introduces a minimal, **frozen** ABI the
skill/extension layer will rely on. The package lives in
`ouroboros/contracts/` and is deliberately small — it declares structural
contracts only, not new runtime behaviour. Existing code is not required to
import from it; the protocols are verified against the real implementations
via `tests/test_contracts.py`.

### 11.1 What is frozen

| Contract | File | Anchored by |
|----------|------|-------------|
| `ToolContextProtocol` — workspace/task-aware minimum every tool handler relies on (attributes: `repo_dir`, `drive_root`, `budget_drive_root`, `pending_events`, `emit_progress_fn`, `current_chat_id`, `task_id`, `task_metadata`, `task_contract`, `workspace_root`, `workspace_mode`, `project_id`; methods: `repo_path`, `drive_path`, `drive_logs`, `active_repo_dir`, `is_workspace_mode`) | `ouroboros/contracts/tool_context.py` | `ouroboros.tools.registry.ToolContext` must satisfy it (duck-typed check + AST field/method parity) |
| `ToolEntryProtocol` + `GetToolsProtocol` — the tool-module ABI | `ouroboros/contracts/tool_abi.py` | Every entry returned by `ToolRegistry._entries` must satisfy `ToolEntryProtocol` |
| `api_v1` WS/HTTP envelopes — inbound: `ChatInbound`, `CommandInbound`; outbound WS: `ChatOutbound`, `PhotoOutbound`, `VideoOutbound`, `DocumentOutbound` (v6.57.0: `document` file-delivery frame emitted by `send_file`/`LocalChatBridge.send_document`, mirrored to the `chat.document` event topic; v6.57.2 adds optional `download_url` — a loopback `/api/files/download?path=` URL for the durable artifact copy so the desktop host-bridge download is WKWebView-safe and the bubble is rebuilt on reload; `send_document` also persists a compact base64-free `chat.jsonl` row via `log_chat(record_type="document", …)` that `/api/chat/history` replays as a `msg_type="document"` record), `TypingOutbound`, `LogOutbound`, `ExtensionLifecycleOutbound`, `TaskNamedOutbound` (v6.40.0: proactive `task_named` card-title push), `MessageAnnotationOutbound` (v6.64: canonical-message Project routing annotation); HTTP: `HealthResponse`, `StateResponse` (Phase 2 adds `runtime_mode: str` and `skills_repo_configured: bool`; v5.11.0 adds `github_token_configured: bool`; v6.13.0 adds `context_mode: str`; v6.57.0 adds `safety_mode: str` so the Settings Safety-Supervisor card reads the current mode; v6.64 makes ledger-backed `spent_usd`, `budget_pct`, and `spent_calls` nullable when accounting is unavailable), `TaskCreateRequest` (v6.23.3 adds optional `project_id: str` for the per-project facts scope; adds optional `disabled_tools: string[]` — the declarative tool-policy denylist, mirrored in `web/modules/api_types.js`; v6.60.0 adds optional `answer_protocol: str` — "" | "final_answer_line") + `ExecutorRef` for host-owned executor-backed external workspace tasks, `TaskCreateResponse`, `EvolutionStateSnapshot`, `SettingsNetworkMeta`, `SettingsMeta` (`custom_secret_keys` + setup contract metadata). v6.64 also makes `ChatOutbound` cost fields nullable so unavailable accounting is distinct from `$0`. | `ouroboros/gateway/contracts.py` | AST scans of `supervisor/message_bus.py` chat/media envelopes, `gateway/state.py::api_state`, `gateway/state.py::api_health`, `gateway/settings.py::_build_network_meta`, and `gateway/ws.py::ws_endpoint` inbound dispatch assert no un-declared keys leak out; `tests/test_contracts.py::test_state_response_declares_runtime_and_capability_keys` explicitly pins runtime/capability state keys, and `tests/test_contracts.py::test_task_create_request_declares_executor_ref_contract` pins the executor request surface |
| `ChatOutbound.cancelable` + `TaskCancelResponse.cascade` (v6.82.0) — additive-optional cancellation ABI: the host-attested `cancelable: true` progress-meta marker that gates the chat card's "Cancel run" action (a card's shape alone cannot distinguish a pooled root from an in-process direct-chat turn), plus the cancel endpoint's echoed `cascade` flag. Existing envelope semantics are unchanged; every field is optional. | `ouroboros/gateway/contracts.py`, `web/modules/api_types.js` | `tests/test_gateway_parity.py` pins both fields in both the Python and JavaScript mirrors; `tests/test_task_cancel_endpoint_v682.py` pins the response shapes; `tests/test_gateway_history.py` pins the marker's replay passthrough. |
| `ChatOutbound.review_projection` (v6.65.0) — optional compact panel/actor truth for Chat and Logs: transport status, parse status, semantic verdict, task-acceptance `outcome_tier`, model/provider/role, coverage, quorum/enforcement impact, the complete redacted reason, a forensic `response_ref` (flat content hashes, no host paths — v6.70.0), and exact candidate/evidence/fence binding hashes; v6.74.0 adds additive optional keys — per-actor `dialogue_status`, per-panel `dialogue` ({status, votes}) and the `single_reviewer_no_diversity` label; raw reviewer output remains in private audit storage. | `ouroboros/gateway/contracts.py`, `ouroboros/review_substrate.py` | `tests/test_contracts.py` pins the field as optional frozen ABI; `tests/test_gateway_parity.py` pins the field in both Python and JavaScript contracts; `tests/test_review_substrate_v2.py` pins the bounded actor projection including `outcome_tier`; `web/tests/review_truth.test.js` pins the shared renderer. |
| `chat_id_policy` — SSOT for A2A/synthetic chat-id filtering across message bus, history, memory, and consolidation | `ouroboros/contracts/chat_id_policy.py` | `tests/test_chat_id_policy.py` pins boundaries and human/transport positive ids |
| `task_contract` — canonical host-draft task objective/output/constraint/resource/disabled-tools/deadline/workspace/lineage/delegation-budget contract helpers (`build_task_contract`, `attach_task_contract`, `normalize_allowed_resources`, `normalize_resource_policy`, `normalize_disabled_tools`, `normalize_delegation_budget`). v6.37.0 adds the additive `delegation_budget` block (`may_delegate`/`may_mutate`/`may_fan_out`/`depth_remaining`/`max_children`/`intent_note`) so a parent's delegate/mutate/fan-out intent propagates structurally to children. The additive `disabled_tools` field (`normalize_disabled_tools`) is a declarative tool-policy: a list of tool names withheld from the agent (hidden from `schemas()`/`get_schema_by_name`/`available_tools` and blocked at `execute`), independent of `allowed_resources` — so a benchmark adapter can disable the agent's own web/search/VLM tools while leaving shell network egress (git/pip) intact, without tripping the web↔network cross-implication. v6.54.4 added the typed `budget_profile`; v6.64 makes `max_improvement_passes` authoritative under every policy and keeps `reserve_finalization_pct` as the review-reserve input. Legacy `improvement_policy=until_deadline` and `stall_rounds_threshold` are accepted for one compatibility window with a loud `deprecated_task_pacing_alias` event: `until_deadline` retains its historical no-implicit-count-cap behavior when a deadline exists, but an explicit pass cap still wins; `stall_rounds_threshold` is retained only as an ignored compatibility field and is not a pacing authority. v6.56.0 adds the additive `budget_profile.cost_hard_stop_pct` (0–100): the in-task cost hard-stop as a percentage of the budget remaining at task start — None → the historical 50%-of-remaining stop, 0 → NO in-task cost stop (never a $0 ceiling; deadline/rounds/global budget gate remain the bounds); resolved once per task by `task_pacing.resolve_cost_ceiling_usd` and consumed by the loop budget gate plus the latched 50/25/10%-remaining cost milestones + ~80%-spent wrap-up note (`task_pacing.build_cost_budget_note`), which replace the old round-gated `[INFO]` budget nudge. v6.44.0 also records built-in tool `missing_credential` capability omissions from lazy availability predicates (`claude_code_edit`→`ANTHROPIC_API_KEY`, `web_search`→available backend incl. no-key ddgs, GitHub tools→`GITHUB_TOKEN`) and blocks direct execution of unavailable built-ins with `CAPABILITY_UNAVAILABLE`. It propagates to subagents via the parent-contract spread. | `ouroboros/contracts/task_contract.py` | `tests/test_contracts.py::test_public_api_is_stable` pins the public helper names (subset); task/outcome tests pin resource and resource-policy normalization plus contract propagation; `tests/test_delegation_budget.py` pins the budget block; `tests/test_disabled_tools_policy.py` pins `disabled_tools` normalization, contract+gateway propagation, registry hiding/blocking, and missing-credential omissions |
| `PluginAPI` (Phase 4, v1.3) + `ExtensionRegistrationError` + `FORBIDDEN_EXTENSION_SETTINGS` + `VALID_EXTENSION_PERMISSIONS` + `VALID_EXTENSION_ROUTE_METHODS` — the surface every `type: extension` skill's `plugin.py::register(api)` binds against (`register_tool`, `register_route`, `register_ws_handler`, `register_ui_tab`, `register_settings_section`, `register_supervised_task`, `register_companion_process`, `subscribe_event`, `get_skill_token`, `send_ws_message`, `on_unload`, `log`, `get_settings`, `get_state_dir`, `skill_job_dir`, `get_runtime_info`). `skill_job_dir(job_id)` creates isolated `jobs/<sanitized_id>-<hash>/{assets,output,tmp}` state folders so generation skills do not overwrite their own assets across jobs. `VALID_EXTENSION_PERMISSIONS` includes host-mediated permissions (`companion_process`, `supervised_task`, `subscribe_event`, `inject_chat`) that require review/owner grants as documented in CHECKLISTS.md. The `ExecutionMode` capability matrix (`MATRIX_CAPABILITIES` / `OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES` / `capability_available` / `available_capabilities`) is the SSOT for which side-effect surfaces an out-of-process child may use and is pinned by the contract test. | `ouroboros/contracts/plugin_api.py` | `tests/test_contracts.py::test_plugin_api_surface_is_frozen` pins the frozen method set; `tests/test_contracts.py::test_extension_route_methods_contract_matches_server_dispatch` pins the route-methods tuple; `tests/test_extension_loader.py::test_plugin_api_impl_matches_protocol` asserts the concrete `PluginAPIImpl` structurally satisfies the runtime-checkable Protocol |
| `SkillManifest` — unified `SKILL.md` / `skill.json` format (`type: instruction \| script \| extension`; v6.9 adds reviewed `scheduled_tasks` cron metadata; v6.85 adds optional bounded canonical `conflicts` names) | `ouroboros/contracts/skill_manifest.py` | `parse_skill_manifest_text()` tolerates missing optional fields; `validate()` returns warnings without raising |
| `schema_versions` — opt-in `_schema_version` key + `with_schema_version`/`read_schema_version` helpers | `ouroboros/contracts/schema_versions.py` | First wired by the extension `health.json` vector (v6.15.0); other legacy state files still read as version 0 until migrated |

### 11.2 What is NOT frozen (intentionally)

- The full `ToolContext` dataclass (browser state, review history, model
  overrides, …) remains mutable implementation detail.
- `OUROBOROS_SCHEMA_VERSION` of `state.json` / `queue_snapshot.json` /
  `task_results/*.json` is treated as `0` (legacy) until Phase 2+ wires the
  helpers in.
- The raw WebSocket/HTTP *values* — only the *shape keys* are pinned.
- The `SKILL.md` body (human-readable markdown) — only the frontmatter
  schema is pinned.

### 11.3 What to do when extending

Any extension of the ABI MUST:

1. Add the new field/envelope key to the appropriate file under
   `ouroboros/contracts/`.
2. Mention the new frozen surface here (Section 11.1 table).
3. Update `tests/test_contracts.py` so the new surface is enforced.

Removing anything from Section 11.1 is a deliberate ABI break and requires
a version bump + a migration note in the release row.

### 11.3 Recent ABI Retirements

- `5.25.0-rc.4`: retired the native skill upgrade migration banner API
  (`GET /api/migrations`, `POST /api/migrations/{key}/dismiss`, and
  `MigrationsResponse`). The release row is the migration note: old
  dismissed banner state in `data/state/migrations.json` is intentionally
  ignored by current runtimes.

---

## 12. Host Service, Companion Processes, and Chat IDs

### Host Service API

The Host Service listens on loopback (`127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}`), separate from the public web app. Routes: `GET /identity`, `GET /tools/schemas`, `POST /chat/inject`, `POST /chat/allocate-internal`, `POST /ui/ws-message` (WS-out bridge for out-of-process extensions; gated on the manifest `ws_handler` permission), `WS /events`. Every route requires a content-bound `X-Skill-Token`; extension/companion skills receive an opaque `SkillToken` and must call `use_in_request()` at request construction sites.

Rationale: reviewed skills may need to report progress, inject transport messages, or observe lifecycle events, but they must not get the browser app's broad authority or raw credentials. Loopback + opaque token + review/grant gates keeps the trust boundary local and explicit.

`POST /chat/inject` is the reviewed transport path for chat skills. It accepts raw chat text, including owner slash commands, after the same fresh-review, enablement, content-hash token, `inject_chat` grant, rate-limit, and source-attribution gates as ordinary transport messages. The trust boundary is the reviewed transport skill plus grants; the host no longer treats slash-shaped text from that authenticated path as automatically less legitimate than the same text typed in the direct UI. A reviewed, owner-bound transport/control skill is a first-class control surface — a full replacement for the local UI for owners without a screen/notebook — and review judges its actual safety properties (binding, attribution, bounded polling, panic cleanup, token confinement, no exfiltration), not the breadth of control it exposes (see `docs/CHECKLISTS.md` → "Transport and control skills are first-class").

Owner slash commands arriving from external transports are additionally authorized in `server._process_bridge_updates` against a separate owner-external chat slot (`owner_external_id`/`owner_external_chat_id` in `state.json`), bound trust-on-first-use by the first external slash with positive identity (which registers the chat and asks for a resend instead of executing). The local web owner (`1/1`) and the external owner are tracked separately, so a desktop user who opened the web UI first never locks out a real Telegram owner, and an unidentified transport (`0/0`) can neither bind nor execute.

### Companion Process Supervisor

Companion processes are host-supervised subprocesses for transport/live skills. Their descriptors are registered through `PluginAPI.register_companion_process`, tracked in `extension_companion.py`, snapshotted under `state/extension_companions.json`, and stopped on unload/panic. For out-of-process (isolated-dep/native) extensions the companion is a cataloged surface: the per-call child records the manifest-declared name during catalog, and the host (which owns the supervisor) spawns it via the same descriptor build after catalog — so enable/disable/reload/panic coupling is identical to in-process companions. Only the server process owns the supervisor: worker-initiated lifecycle changes (agent `toggle_skill`, post-review auto-enable) write durable per-request markers under `state/extension_reconcile/` (safe skill prefix + request id), and the server lifespan pickup task consumes them to run server-side `reconcile_extension`, start registered-but-missing companions, or stop companions for disabled skills. The UI/HTTP enable still runs in the server and spawns immediately. The remaining deliberate tradeoff is companion `cwd`: it is the reviewed skill payload directory (like the OOP dispatch children), with the content-hash executable-review gate as the trust boundary; a post-review payload edit makes review stale and blocks reload. A staged OOP-companion snapshot remains a deferred follow-up.

### Chat IDs

`contracts/chat_id_policy.py` separates human-visible chat IDs from synthetic/internal transport IDs. Review-skill progress may route to `chat_id=0` (Skill Review panel), so absence from the main chat is not proof of inactivity.

### PluginAPI extension surface

`contracts/plugin_api.py` defines the frozen extension ABI: tools, routes, WS handlers, UI tabs, settings sections, supervised tasks, companion processes, event subscriptions, runtime info, state dirs, and skill tokens. Additive schema evolution is allowed; tightening permission allowlists or changing signatures is frozen-contract work.

### Settings secrets

Core settings and custom secrets remain owner state. Skills can request grants, but forwarded keys require fresh executable review plus owner approval.

## 13. External Skills Layer

### Topology

Payloads live under `data/skills/{native,clawhub,ouroboroshub,external}/<name>/` plus optional `OUROBOROS_SKILLS_REPO_PATH`. Review/enable/grant/deps/token state lives under `data/state/skills/<name>/`. Native skills are launcher-seeded examples/core payloads and not ordinary repair targets.

### Discovery and manifests

`skill_loader.py` parses `SKILL.md` / `skill.json` through `contracts/skill_manifest.py`, tags each skill source, computes content hashes over runtime-reachable text payloads, excludes secrets/caches, and detects provenance sidecars (`.clawhub.json`, `.ouroboroshub.json`, `.self_authored.json`, `.seed-origin`). The optional `conflicts` manifest list names skills that cannot share runtime ownership. A declaration on either side is enforced symmetrically against enabled installed peers; missing or disabled peers are inert. Conflict status is bounded and typed for API/UI consumers, and enable/reconcile/startup/dispatch fail closed without automatically disabling or deleting either skill.

### Lifecycle

Install/update/review/deps/enable/disable/uninstall run through `skill_lifecycle_queue.py`, one mutating FIFO lane with dedupe keys and UI/live-card progress. `skill_review_runner.py` writes `review_job.json`, emits skill_review events, and reconciles dependencies/extensions after review.

### Execution gates

A skill can execute only when all are true:

- manifest parsed and runtime type is executable;
- content hash matches the latest review;
- review verdict is executable under `skill_review_gate`;
- enabled state is true;
- no enabled installed peer conflicts with either manifest;
- required grants are present;
- isolated dependencies are installed when declared;
- extension/widget registration passes host validation.

Rationale: review PASS alone is not enough. Payload, grants, enablement, dependency state, and extension loading are separate trust boundaries.

### Skill review

Skill review uses deterministic preflight plus tri-model review against the Skill Review Checklist. Optional Claude Code advisory is fail-open and payload-scoped; tri-model findings remain authoritative. Findings are persisted with raw actor records for forensics, and current status is computed from findings (`clean` / `warnings` / `blockers` / `pending`). Accepted rebuttals/history prevent reviewer thrash without letting stale findings silently disappear.

`review_job.json` and append-only `review_history.jsonl` also carry exact
`task_id`, root task, chat, source, content hash, and terminal reason. A
`review_round` increases across one task-origin/skill group even when the hash
changes; `snapshot_attempt` counts only attempts for that group and hash.
Ordinals are allocated in `_on_started` under the existing cross-process
lifecycle lock; a started failure/cancel/timeout consumes its number and writes
one idempotent terminal row keyed by `job_id`, while pre-start dedupe consumes
none. Legacy rows receive computed ordinals only at read time. The UI shows the
current round and last ten rows from its group; full raw history remains private,
and the main chat receives one compact lifecycle/provenance record.

Official OuroborosHub payloads get a narrow `official_hub` review profile only when their `.ouroboroshub.json` sidecar, the live catalog file list, and the full local runtime-reachable file set match exactly by SHA-256. For such hash-verified official payloads the profile downgrades severity-driven hygiene/bug findings (`bug_hunting`, `companion_process_safety`, `extension_namespace_discipline`, `widget_module_safety`) to warnings — they already passed review at submission — so routine re-review no longer blocks on style nits. Hard trust-boundary checklist items (`manifest_schema`, `permissions_honesty`, `no_repo_mutation`, `path_confinement`, `env_allowlist`, `inject_chat_minimization`, `event_subscription_minimization`, `host_token_handling`) still aggregate to `blockers`. A deterministic `skill_preflight` FAIL aggregates to `pending` — non-executable under every enforcement mode (stronger than an advisory-overridable blocker) and surviving reload — and sensitive-file, binary-payload, path, dependency, grant, enablement, and hash-mismatch gates remain fail-closed. Because the profile relaxes findings, it requires the entire local runtime-reachable file set to match the catalog exactly: a local edit that changes a payload hash, adds a runtime-reachable file, or breaks the sidecar/catalog digest match drops the profile and returns to ordinary local skill-review semantics.

### Native Telegram

`skills/telegram/` combines the established owner-only text/photo bridge and Mini App PoC behind one native skill state boundary. The in-process extension owns polling, first-positive-private-chat owner binding, Host Service injection, outbound media/events, commands, cards, notifications, settings, and bounded bridge/Mini App status. Its host-supervised companion owns the authenticated loopback sidecar, process-memory Mini App sessions, pinned `cloudflared` Quick Tunnel, Telegram menu snapshot/restore, heartbeat, retry, singleton, and watchdog lifecycle. Mini App availability is independent from text-bridge loading: disabling it or running on an unsupported tunnel platform leaves the bridge available.

The private operational root is `data/state/skills/telegram/`:

| Path | Writer and authority | Durability / sensitive role |
|---|---|---|
| `settings.json`, `.settings.lock` | Extension settings route and `telegram_settings.py`; canonical owner binding plus bridge/Mini App preferences | Durable owner authority; lock is transient |
| `runtime_config.json`, `status.json`, `.companion.lock` | Mini App registration and supervised companion; loopback port/config plus bounded heartbeat/status | Config is regenerated on registration; status and singleton lock are runtime projections, not owner authority |
| `menu_button_snapshot.json` | `telegram_menu.py` before installing the Mini App button | Durable rollback authority until the prior Telegram menu is restored |
| `poll_offset.json`, `bridge_status.json`, `notif_state.json`, `silent_state.json`, `subagent_state.json` | In-process bridge and notification/card helpers | Durable cursors and bounded delivery/UI projections; no owner or rollback authority |
| `cloudflared/<version>/<platform>/`, `cloudflared/.install.lock`, `cloudflared/quick-tunnel-runs/` | Pinned installer and tunnel lifecycle | Verified binary cache is durable; install lock and per-generation homes are transient; no owner or rollback authority |

The manifest conflicts with `telegram-bridge` and `telegram-miniapp-poc`; no legacy state, settings, grants, or enablement are migrated. Launcher seeding installs the repo-reviewed bytes into the native data-plane slot and stamps the existing hash-bound `native_seed` verdict. Because the skill requests `TELEGRAM_BOT_TOKEN` and privileged host permissions, it remains disabled until the normal grant-then-enable owner flow completes. Google Colab waits for native discovery plus a fresh executable `native_seed` projection, grants only API-reported missing grantable items when the persisted auto-grant policy permits it, then enables and saves `full_access`, mirror-all, and Mini App-on settings. It never installs the Telegram skill from a marketplace or runs a separate skill review.

### Marketplace

ClawHub and OuroborosHub install into the data plane, trigger review, and persist provenance. ClawHub archives are host-allowlisted, size-capped, text-only, staged privately, translated from OpenClaw frontmatter, and landed atomically. OuroborosHub is a static GitHub catalog with per-file SHA checks. Install specs are normalized into bounded per-skill isolated dependency installs; global/manual specs remain guidance.

### Extension Loader

`extension_loader.py` has two reviewed `type: extension` paths. No-dependency pure-Python extensions import in-process through a staged import tree, namespace tools/routes/WS/UI/settings surfaces, validate declared permissions, track registrations per skill, and unload atomically. Extensions with reviewed isolated dependencies, plus any payload-native marker that somehow passes review, are cataloged in a short-lived `extension_process_runner` child; the host registers proxy descriptors and dispatches tool, HTTP route, and WS handler calls back through short-lived children.
`_stage_extension_import_tree` creates per-load `__extension_imports/<pid>-<uuid>/skill/` directories under `data/state/skills/<name>/` so in-process imports are isolated and can be removed on unload. The owner-PID prefix lets `_sweep_stale_extension_imports` distinguish a peer worker's still-loading tree (owner process alive, or fresh within a spawn grace) from a genuine orphan (owner dead and past grace) under `MAX_WORKERS>1`, where every worker stages into this shared dir concurrently — so a sibling sweep never `rmtree`s a live peer's tree mid-load (which would `FileNotFoundError` its `exec_module` and silently drop the skill in that worker). Legacy bare-`<uuid>` leaves keep the prior keep-set-only reap. Child-process dispatch uses the same staged loader inside the child (bare `<uuid>` under a private base, unswept), where isolated deps are allowed on `sys.path` without exposing native crashes to `server.py`.
Child processes set `OUROBOROS_EXTENSION_PROCESS_CHILD=1` as an internal runtime marker. In that mode the child skips cross-process staged-import sweeping and cleans only its own import root on exit. The `ExecutionMode` capability matrix in `contracts/plugin_api.py` is the single source of truth for what such a per-call child may use: `register_tool`/`register_route`/`register_ws_handler`/`register_ui_tab`/`register_settings_section` are proxied, `on_unload` runs at child teardown, `send_ws_message` relays through the Host Service `POST /ui/ws-message` bridge (identity from the skill token, host-side namespacing), and `register_companion_process` is recorded during catalog and then host-spawned/supervised. Only `subscribe_event` and `register_supervised_task` (which need a persistent in-process context) are unavailable in the per-call child — a manifest-declared `companion_process` is the supported alternative for long-running work and host-event subscription. `get_runtime_info()` returns `execution_mode`/`capabilities` so a skill negotiates instead of crashing mid-registration.

Rationale: in-process extensions are powerful and therefore stricter than subprocess skills. Namespacing, review, grants, dependency isolation, atomic unload, and child-process dispatch for isolated deps prevent one extension from shadowing core surfaces, leaking state after disable, or crashing the server with native `panic=abort`.

### Generic transport metadata and repair constraints

Transport skills annotate injected chat/photo messages with source/session metadata. Formal repair tasks carry `TaskConstraint(mode="skill_repair")`, and scoped editors enforce payload confinement with short relative paths only under the selected skill. At shared owner-message ingress, a valid repair constraint is promoted directly to a managed root task before Project/mailbox/ephemeral routing; `promote_chat_to_task` canonicalizes the named payload, forces `allow_enable=false` and `allow_review=true`, and attaches the constraint before the task contract. This avoids the former impossible intersection where an ephemeral turn hid repair mutators while heal mode blocked promotion, without weakening ordinary ephemeral default-deny behavior.
