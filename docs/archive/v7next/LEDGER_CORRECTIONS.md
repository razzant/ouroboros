# Ledger corrections discovered during v7next transplants (append-only)

Rows of the reference MIGRATION_v7.md / DOMAIN_MAP.md falsified by upstream drift,
with evidence, found lane by lane. Applied to the campaign's carried ledger at F5.

## From the D15 pilot (base b9f7597f, 2026-08-30)
1. MIGRATION row 351 (`tools/core.py::_filter_out_project_store` ->
   `project_facts.py::filter_out_project_store`, status "pending upstream
   transfer") — SUPERSEDED-BY-UPSTREAM: the tip already carries the extraction
   (project_facts.py byte-identical to the reference; core.py keeps only the
   import alias at :17 with two call sites).
2. DOMAIN_MAP §D15 "v7 delta" prose — remeasure from the new base: consolidator
   delta absorbed upstream (now 0); the true residue is +23/-12 in two files
   (consciousness.py, reflection.py), not +25/-14 in three.
3. RE-PROVE TRAP (D02 family, reflection.py): the reference's
   `_trace_call_errored` reads `_OK_TOOL_STATUSES` (with "untyped") from the v7
   leaf `_outcome_tool_errors`, which upstream does not have; a verbatim replay
   of the delta over upstream's own status handling would invert the fix. The
   D02 adoption must re-derive the delta against upstream bytes.
4. MIGRATION row 166 (retirement of 4 CLAUDE_CODE markers, id "none") — needs an
   explicit ADOPTION disposition (umbrella under D02 or its own row): zero
   production emitters of those markers exist at this tip (claim re-proven).

## From the D16 split pilot (base 5d3398c1, 2026-08-30)
5. MIGRATION row 3911 (`usage_accounting.py::_legacy_snapshot` ->
   `usage_legacy_import.py::_legacy_snapshot`, "verbatim extraction") —
   BYTE-FALSIFIED as a copy source, transform still valid: upstream e9bf6f14
   rewrote the settings-hash comment inside the span (two lines "... prove
   non-mutation by hash, but never copy / their contents into the usage
   archive." became one line "... never copy contents."). The tool's --check of
   the reference leaf against tip bytes fails token-lockstep on exactly this
   span (ast=True, tokens=False); re-emitting from tip bytes is proof-green on
   the first round with the reference declared set {_legacy_snapshot, _locked,
   _read_records_locked} unchanged. Copying the reference leaf verbatim would
   have silently reverted an upstream comment edit.
6. MIGRATION rows 3910-3914 status "pending upstream transfer" — RE-CONFIRMED
   at this tip (contrast with the D15 project_facts case, entry 1 above):
   upstream still carries the unsplit legacy import inside
   ouroboros/usage_accounting.py (1600 lines, exactly at the hard cap;
   IMPORT_REL at :60, the four defs at :1374-:1600). The extraction was
   performed by this lane from tip bytes.

## From the D03 lane (base f61ea3c2, 2026-08-30)
7. MIGRATION rows 3943-3946 (`ouroboros/context.py::{_project_room_fact,
   _runtime_budget_info,_promoted_task_toolset,_delegation_capability_fact}` ->
   `ouroboros/context_runtime_facts.py`, "pending upstream transfer") —
   RE-CONFIRMED pending at this tip (context.py 1590 lines, the four defs at
   :325-:544); the extraction was performed by this lane from tip bytes. The
   reference leaf is BYTE-FALSIFIED as a copy source for ONE of the four
   symbols: upstream b14ba397 ("expose available subagents in runtime
   context") rewrote `_delegation_capability_fact` (docstring collapsed to a
   one-line summary, `configured_route` dropped from the returned fact,
   requested/applied profile evidence and `selected_subagent_id` added, plus
   an all-absent -> None guard). Drift-probe `--check` of the reference leaf
   against tip bytes: 3/4 spans ast=tokens=bytes=True, this span
   ast=False/tokens=False; re-emitting from tip bytes was proof-green on the
   first round. Copying the reference leaf verbatim would have silently
   reverted the upstream subagent-profile feature.
8. MIGRATION row 3960 (`tests/test_context.py::
   test_delegation_fact_carries_configured_route_and_historical_rows` ->
   `tests/test_context_runtime_section.py::<same>`) — SOURCE SYMBOL FALSIFIED
   by the same upstream train: b14ba397 replaced the test with
   `test_delegation_fact_carries_historical_rows_and_profile_evidence`
   (asserts `"configured_route" not in delegation`). The upstream successor
   was moved to the row's destination as an identity continuation (tip
   bytes); the carried ledger must rename the row at F5.
9. MIGRATION row 1641 (`tests/test_context.py::
   test_runtime_section_includes_improvement_backlog_digest` ->
   `tests/test_context_runtime_section.py::<same>`) — SOURCE SYMBOL FALSIFIED:
   upstream 1b7f9497 replaced the test with
   `test_improvement_backlog_digest_is_actor_scoped` (the digest is now
   asserted ABSENT for ordinary/main/project/subagent tasks and present only
   for evolution/deep_self_review). Moved to the row's destination as an
   identity continuation (tip bytes); rename at F5.
10. S7a rows 1614-1640/1642-1648 — RE-CONFIRMED against tip bytes: every other
   moved symbol of the tests/test_context.py split is byte-identical between
   the tip monolith and the reference siblings (the D15-carried
   tests/test_context_memory.py re-derived from tip bytes came out identical
   — the carry was NOT stale), except row 1623's span
   (`test_force_plan_metadata_adds_structured_notice_without_rewriting_user_text`),
   which upstream drifted ADDITIVELY (rc-phaseC execution-shape assertions) —
   tip bytes transplanted. Note: between the D15 pilot and this lane the 15
   memory tests existed in BOTH tests/test_context.py and
   tests/test_context_memory.py on the integration branch (ran twice); this
   lane completed the split and deduplicated.
11. NO-ROW upstream additions (candidate rows for the carried ledger): 3459dd12
   added 8 recent-chat/archive-generation tests to tests/test_context.py
   (filters_archives_before_recent_bound, retention_proof_cross_thread,
   reads_only_bounded_generation_suffix, materializes_a_bounded_row_suffix,
   malformed_gap_even_when_search_matches_nothing,
   resumes_unconsolidated_archived_generation,
   archive_only_chat_chain_is_complete, missing_cursor_generation_hot_path).
   They have no MIGRATION rows, so this lane left them in the remainder
   tests/test_context.py (612 lines) rather than deciding their theme-home
   unilaterally; by the memory-file theme they are candidates for
   tests/test_context_memory.py at F5.
## From the D09 lane (base f61ea3c2, 2026-08-30)
7. MIGRATION rows 998-1013 (the 16-symbol task_lifecycle.py ->
   cancel_custody.py settle-owner extraction) — HOT-FALSIFIED as a transplant
   at this tip: upstream 65b5d19f ("Refactor cancellation ownership for size
   ratchet") re-decomposed the same ownership differently (task_lifecycle
   -408 lines into cancel_publication.py, owner_stop.py,
   queue_transitions.py, task_reaper.py, new evolution_lifecycle.py, new
   task_admission.py), then 3877e2ce/bea08137/21c59de2 reworked the
   survivors. Of the 16 declared symbols, _intent_outcome_fields now lives
   in cancel_publication.py:133 (task_lifecycle re-exports it at :26-35),
   _durable_settled_status no longer exists, and the remaining bodies were
   hardened by bea08137. Transplanting the reference cancel_custody.py would
   create a second ownership answer -> F2 (cancel/delegation organ, re-split
   from the upstream form).
8. MIGRATION rows 834-839 (cancel_intents.py D08 corrupt-projection rule) —
   PARTIALLY SUPERSEDED-BY-UPSTREAM: at this tip request_cancel and
   claim_intent already read strict and raise CancelIntentProjectionCorrupt
   (upstream custody train 34ca9b02/38196641/c8048f2c/bea08137 rewrote the
   module 888 -> 1281 lines), while release_claim, settle_intent,
   mark_intent_scope and mark_finalize_control_drained remain fail-open
   (AST probe over tip bytes; the reference pin
   test_cancel_intent_corruption_s6.py runs red on exactly those four).
   D08 must be re-derived against the rewritten bytes in F2 — same class as
   entry 3 (the re-prove trap).
9. MIGRATION rows 2152-2180 (the S7b split of
   tests/test_cancel_intents_phase_a.py) — falsified as a verbatim
   transplant: the giant drifted upstream since the merge-base, and the
   split's custody rows retarget monkeypatches to supervisor.cancel_custody,
   which this tip does not have (row 2171's own note binds the split to the
   extraction commit e3c107bd). Rides with entry 7 into F2.
10. DOMAIN_MAP §D09 pin test_subagent_worktree_registry_s6.py —
   cross-listed: the module it pins, ouroboros/subagent_worktrees.py, is a
   D07 owner, and the strict-registry behaviour the pin asserts lives in the
   reference's +104/-22 delta to that module (upstream never touched it:
   tip == merge-base). The pin transfers with D07's module delta, not with
   the D09 lane (11 of its tests are red without it).
11. DOMAIN_MAP §D09 pin test_daemon_token_containment_s6.py — HOT-DEFERRED
   with the delegation organ: its fixture's fresh delegate_start is refused
   at this tip with reason "subagent_selection_required" ("A fresh delegated
   start requires an explicit agent_session subagent_id. Only retry_of may
   replay a selectorless immutable invocation.") — the upstream
   delegation-by-construction train changed the entry contract the fixture
   drives.
12. Two reference pins byte-falsified by upstream drift, residual facts
   intact, re-pinned to tip bytes by this lane: (a)
   test_panic_stop_port_sweep.py — the panic's kill_workers call now carries
   reconcile_delegate_custody=False (dc4c0204), and this tree has 5
   ouroboros/server_*.py host leaves, not the reference's >= 11 (that floor
   returns with the D11 server split); (b) test_owner_stop_fences_s6.py C5 —
   _settle_descendants_hard now reuses the ordinary cascade's bounded
   re-sweep loop (65b5d19f), so one live child yields two token-less sweep
   calls instead of one; the pinned durable fact (the owner-stop sweep is
   token-less) is unchanged.
## From the D17 lane (base def681bd, 2026-08-30)
7. Runtime split rows 465-494 (`headless.py` -> `headless_status.py` (11) +
   `workspace_patch_capture.py` (19), "verbatim extraction") — RE-PROVEN at
   this tip: all 30 spans byte-identical between the reference leaves and
   `git show HEAD:ouroboros/headless.py` (hardened transplant --check, ast/
   tokens/bytes all green, both leaves, exit 0). The facade differs from the
   reference only by upstream residue drift (child_ref promotion machinery,
   `TASK_COST_META_FIELDS`/`replace_atomic` import changes) — replayed from
   tip bytes, 947 lines.
8. Test-split rows for `tests/test_workspace_executor.py` ->
   `test_workspace_executor_services.py` ("verbatim") — BYTE-FALSIFIED as a
   copy source for exactly two functions, transform still valid: upstream
   06339bb7 ("fix: preserve service readiness truth") rewrote
   `test_executor_local_service_lifecycle_hides_private_snapshot` (the READY
   marker is now planted before a 25k log suffix and asserted scanned) and
   upstream a849c9a6 ("fix: preserve executor probe uncertainty") extended
   `test_executor_service_status_and_durable_record_redact_secret_like_args`
   (adds the `'"readiness"' not in durable_text` clause). Both re-emitted
   from tip giant bytes; the other 26 moved wexec spans are byte-identical.
9. Reference residual `tests/test_headless_cli.py` and sibling
   `test_headless_workspace_shell.py` carry OTHER domains' v7 spellings
   inside 9 moved/kept spans (`_run_shell_safety_check(registry, ...)` typed
   result + `core_file_tools._repo_read` — D04/D05 split; `queue.init(path)`
   1-arg signature and `supervisor.state.QUEUE_SNAPSHOT_PATH` — D08/D33).
   On this tree those leaves/signatures do not exist; per §5.3-Δ item 2 every
   such span was reverse-mapped to the upstream spelling keyed to
   `git show HEAD:tests/test_headless_cli.py` (upstream: string-returning
   `registry._run_shell_safety_check`, module-binding `_repo_read`,
   `queue.init(path, 600, 1800)`, `queue.QUEUE_SNAPSHOT_PATH`). These
   adaptations return with their owning lanes, not with D17.
10. Thirteen upstream test functions written after the reference cutoff have
    NO ledger rows (hcli: 4 task-api + 1 artifact-endpoint; wexec: 6 docker
    stop/cleanup + 2 readiness). Placed by the split's own theme rule with
    imports satisfied by the target headers (task_api×4, task_artifacts×1,
    docker×6, services×2 + one `SimpleNamespace` header import); the carried
    ledger needs rows minted for them at F5. Placement is disclosed, not
    ledger-derived.
11. Row evidence `tests/test_headless_extraction.py` (rows 465-494): the
    reference pin imports `ouroboros.tool_module_inventory` (a D04-family v7
    leaf absent from this tree); the transplanted pin keeps every clause that
    types against THIS tree and replaces the frozen-tool-inventory clause
    with an oracle-SHA note — the clause returns with the tools lane.
12. `ouroboros/task_results.py` (upstream-hot, +555 lines drift): the ledger
    assigns NO D17 runtime split to it, and the reference copy is
    byte-identical to the merge base (zero v7 delta) — nothing to transplant,
    upstream bytes stand. Same zero-v7-delta fact re-proven for all 14
    non-split D17 runtime modules (task_status, retention, coop_checkpoint,
    projects_registry, project_dialogue, project_lease, project_naming,
    project_sources, tools/project_journal, workspace_admission,
    workspace_preflight, workspace_executor, workspace_patch_rules).
## From the D18 lane (base d830cdba, 2026-08-30)
13. MIGRATION rows 3998-4000 (`launcher.py::{_prepare_windows_webview_runtime,
    _show_windows_message,_windows_dll_dir_handles}` ->
    `ouroboros/launcher_windows_runtime.py`, "pending upstream transfer") —
    RE-CONFIRMED pending and transplanted by this lane. Drift-probe: hardened
    `--check` of the reference leaf against `git show d830cdba:launcher.py`
    is green on all three spans (ast=tokens=bytes=True, leaf invariants [],
    exit 0), so the reference leaf IS tip bytes; adopted verbatim. Facade =
    tip monolith minus the three spans plus the reference's re-export block;
    byte-diff against the reference facade is exactly upstream dc4c0204's
    +10 delegated-restart hunk (replayed from tip bytes). launcher.py
    1582 -> 1484 lines; band re-entry authorized via the official
    regenerator's --band-rationale.
14. MIGRATION row 917 (`ouroboros/packaged_cli.py::_save_settings`, semantic
    id D03: route the packaged bootstrap saver through the shared persistence
    prologue and serializer) — HOT-DEFERRED with the settings seam. At this
    tip `prepare_settings_for_persist` ALREADY EXISTS in ouroboros/config.py
    :1084 (upstream absorbed part of the seam with a different signature —
    an added `authored_keys` kwarg), while `serialize_settings` and the
    row's pin tests/test_settings_read_seam.py do not exist. A verbatim
    replay would bind a half-absorbed seam; the delta must be re-derived
    against the tip seam form when the D12 config/settings split lands.
    packaged_cli.py itself: tip == merge-base (zero upstream drift), so the
    module stays untouched by this lane.
15. Reference `ouroboros/utils.py` +9/-1 delta (O_BINARY flag inside
    `write_text_atomic`'s fsync path) — SUPERSEDED-BY-UPSTREAM as a class,
    solved differently: upstream c15389f4 added `write_bytes_atomic`
    (utils.py:276, fd opened with `getattr(os, "O_BINARY", 0)`) for
    byte-canonical consumers and pinned `write_text_atomic` to "platform
    newline semantics" in its docstring — a deliberate two-writer
    decomposition. Replaying the reference's O_BINARY into
    write_text_atomic would invert that upstream decision. No transplant;
    cross-OS class registry should record ONE decision for this class
    (upstream's).
16. Reference `tests/test_launcher_server_reaper.py` +8/-3 delta (normpath'd
    REPO/DATA/OURS literals + POSIX-only skipif on
    test_candidate_enumeration_uses_one_unbranded_full_width_ps_read) —
    SUPERSEDED-BY-UPSTREAM as the same cross-OS class: upstream 7de26338
    normpaths the same three literals (also the python binary path, which
    the reference did not) and, instead of skipping the enumeration test off
    POSIX, monkeypatches `reaper.os` with a getuid stub so it runs on every
    OS. Upstream form stands; nothing transplanted; the module itself is
    byte-identical across tip/reference/base.
17. Reference `tests/test_packaged_runtime_and_lifecycle.py` +7/-2 delta —
    DEFERRED WITH ITS OWNERS, not D18's to land: the `_enforce_harness`
    clock hunk patches `supervisor.events_budget/events_chat_delivery/
    events_task_done` (D33 events-split leaves absent from this tree) and
    `test_cancel_and_timeout_paths_share_one_salvage_helper` retargets to
    `supervisor/cancel_custody.py` (HOT-FALSIFIED per D09 lane entry 7;
    rides into F2). Tip bytes stand (upstream b3c9860e's -1 drift included).
18. Reference `tests/test_packaging_sync.py` +17/-7 delta
    (test_system_prompt_lists_bible_in_safety_critical_set strengthened to
    set-equality of BOTH prompts' inventories against
    `runtime_mode_policy.SAFETY_CRITICAL_PATHS`) — UNROWED in MIGRATION_v7;
    left at tip bytes per the wave-1 rule (unrowed test deltas are not
    resolved unilaterally); candidate row for the carried ledger at F5.
    Disjoint upstream drift a23e12b1 (push_to_remote test retargeted to
    `_git_network_bounded`) stands.
## From the D04 lane (base d830cdba, 2026-08-30)
1. Registry-split rows RE-PROVEN against tip bytes for the four landed tools/
   leaves (tool_context, tool_catalog, tool_resolution, registry_guards,
   registry_guard_process — 74 symbols): 61 spans byte-identical between the
   reference leaves and `git show HEAD:ouroboros/tools/registry.py`; 13 spans
   BYTE-FALSIFIED as copy sources by PURE UPSTREAM DRIFT (oracle==merge-base,
   tip moved): _prepare_public_builtin_args, _executor_backend_candidate_allowed,
   _authorized_managed_update_resolver (404B -> 1843B hardening), _disabled_tools,
   _detect_runtime_mode_elevation, _SUBAGENT_SHELL_SECRET_MARKERS,
   _detect_mutative_toggle_self_change, _detect_evolution_owner_control_self_change,
   _detect_context_mode_self_lowering, _DENIED_READ_OPTIONS,
   _is_pure_read_inspection, _detect_safety_mode_self_lowering,
   _detect_owner_skill_attest_self_call. All re-emitted from tip bytes,
   transplant proof green (ast=tokens=bytes on every symbol, exit 0).
2. Rows whose reference destination carries the TYPED-RESULT cutover semantics
   (PURE V7 DELTA; tip==merge-base): 144 (_normalize_dispatch_path_args reduced
   to a projection), 184 (_binding_error_text native codes), 185
   (_payload_dispatch_constraint typed second element), 226
   (_managed_update_code_tool_block thin wrapper), 138 (ToolEntry shallow-frozen
   — also upstream-drifted: tip added the alias_for field). This lane moved the
   TIP bodies verbatim; the typed deltas are deliberately NOT ported — they ride
   with the F2 typed-result organ, not with a byte-preserving relocation of a
   protected file.
3. HOT-DEFERRED: ouroboros/tools/registry_core.py (rows 156, 167, 170, 171,
   174, 175). Evidence: tip ToolRegistry is a 2252-line class (probe: tip span
   124364B vs reference 49860B, ast_equal=False); the reference slimmed it via
   17 method->function extractions (rows 189, 224, 225, 230, 235-242, 287,
   291-293) which change the receiver (self -> registry) and are NOT
   byte-preserving relocation — out of bounds for the protected
   tools/registry.py under this lane's mandate. ToolRegistry and the four
   process/mutation constants stay in the facade; the class also would put the
   new leaf straight into the >1500 band. Re-split from the upstream form in F2.
4. HOT-DEFERRED: ouroboros/tools/tool_result.py. 32 of the reference leaf's 33
   top-level symbols do not exist at tip (the ToolResult/ToolCodeSpec organ,
   D02-family approved deltas); the single registry-sourced verbatim row 139
   (_compose_execute_result) also drifted at tip (661B vs 671B). Creating a
   one-symbol leaf under the organ's name would falsely anchor the F2 re-split;
   _compose_execute_result stays in the facade.
5. HOT-DEFERRED: rows 187/188 (ToolRegistry._dispatch_mcp_tool /
   _dispatch_extension_tool -> extension_dispatch typed dispatchers).
   tip tools/extension_dispatch.py == merge-base (116 lines); the reference's
   +177 lines are the producer-boundary ToolResult typing plus method
   retirement. Upstream bytes stand; the methods stay on the class.
6. loop_tool_execution.py D04 rows (157, 159-164, 826-828) are ALL
   retire/rename/type rows of the classifier cutover — nothing is emittable as
   a byte-preserving span. Shared-monolith convention honored: this lane did
   not touch ouroboros/loop_tool_execution.py at all (D01 owns the rest).
7. tools/core.py shared-leaf note (row 353, core.py::active_repo_dir_for ->
   tool_resolution.py): already satisfied at tip by an import alias
   (core.py:20 imports it from the registry; the registry facade now re-exports
   it from tool_resolution — same object). core.py untouched by this lane.
8. tool_access split rows 495-535 RE-PROVEN against tip bytes: 39/41 spans
   byte-identical; 2 BYTE-FALSIFIED as copy sources by PURE UPSTREAM DRIFT:
   _skill_payload_base (upstream re-homed the body into
   skill_payload_binding.resolve_skill_payload_base — copying the reference
   leaf would have reverted that refactor) and ResolvedResourceBinding
   (upstream added the logical_base_path field). Both re-emitted from tip
   bytes, proof green. The D1 mirror-path defect (safe_relpath lstrip('/'),
   lying "caller rejects" docstrings) travels in the moved tip bytes UNFIXED,
   per the lane instruction — it remains an upstream issue-candidate.
9. Pins carried with disclosed adaptations (identity continuations to tip
   bytes): tests/test_tool_owner_facades.py (+ the alias_for row in the
   ToolEntry contract — upstream drift); tests/test_tool_access_extraction.py
   (4 adaptations, listed in its docstring: tool_module_inventory clause
   dropped until that leaf lands, backedge check narrowed to import-time
   imports because the D18/D33 call-time handle is deliberate, one-matrix
   clause asserts through the facade re-export, size bounds kept);
   tests/test_workspace_authority_binding.py gains the reference's
   tool_resolution identity test while its typed companion
   (_normalize_dispatch_path_args_result) is NOT carried — it pins deferred
   machinery. test_registry_core.py, test_tool_result*.py and the
   classification-differential suites are NOT carried for the same reason.
10. Test-split rows 784-825 (tests/test_tool_capabilities.py -> 4 siblings)
    RE-PROVEN against tip bytes: 34/42 moved spans byte-identical to the
    reference siblings, 8 re-emitted from tip (test_search_code_has_result_limit,
    test_local_readonly_subagent_execute_blocks_forbidden_tools,
    test_local_readonly_subagent_initial_schemas_are_allowlisted,
    test_schedule_subagent_in_initial_schemas,
    test_schedule_subagent_inherits_workspace_executor_ref, and the three
    test_schedule_subagent_required_*_for_readonly tests). Lossless: 61 == 61
    test functions, zero lost, zero added, no duplicate names introduced
    (tree-wide AST dup scan; the 10 pre-existing identical-body duplicates
    between test_review_cycles_dispatch.py and test_review_cycles_skill_dispatch.py
    plus the test_tool_registered same-name pair predate this lane — D06/D05
    territory, reported not touched). 21 unrowed/kept tip tests remain in the
    remainder; 3 header imports that lost their last reader were dropped there.
11. Protection-surface closure (code-side, protective-only): the reference
    extends ouroboros/runtime_mode_policy.py::SAFETY_CRITICAL_PATHS and
    supervisor/update_merge_policy.py::HOT_CODE_PATHS over the registry split
    leaves — without that, guard bodies moved out of the protected registry
    become writable in advanced mode and lose the hot-code label (this tree's
    own parity rule, tests/test_lc2_owner_facades.py, pins the inverse
    direction). This lane mirrored the closure for the five leaves that exist
    here (registry_core.py / tool_result.py rows return with their leaves) and
    pinned it (tests/test_tool_owner_facades.py::
    test_registry_split_leaves_keep_protected_label_parity). NOT mirrored —
    for the owner/F5: the reference's prose updates to prompts/SAFETY.md:10
    and prompts/SYSTEM.md "Immutable Safety Files" (operator-off-limits
    runtime prompts; enforcement is code-side, prose enumerates only the
    facade for now), and the reference's extra HOT_CODE_PATHS row for
    ouroboros/tools/extension_dispatch.py (nothing moved there on this tree —
    adding it is an oracle delta beyond relocation parity).
## From the D12 lane (base d830cdba, 2026-08-30)
13. Split rows 855-867 (settings_scales), 868-879 (model_slots), 880-886
   (review_model_routes) — RE-PROVEN against tip bytes: every span of the three
   reference leaves is ast=tokens=bytes=True against
   `git show HEAD:ouroboros/config.py` (drift-probe first, exit 0); the leaves
   landed from tip bytes and differ from the reference only in BETWEEN-SPAN
   comments upstream rewrote inside config.py (EFFORT_SCALE header now names
   exact-route request-wire recovery; the PROMPT_CACHE_TTL comment rewrapped) —
   carried from tip, since the span proof is blind to inter-span comment lines.
14. Shared-leaf rows 840-846/852-854 (config.py) + 3238-3241 (provider_models.py)
   -> settings_defaults.py — BYTE-FALSIFIED as a copy source on 4 of 12 spans,
   transform still valid: upstream rewrote SETTINGS_DEFAULTS (advisory slot is
   the routed id `anthropic/claude-sonnet-5`, `CLAUDE_CODE_MODEL` retired,
   MAX_SUBAGENT_DEPTH default 2->3, `OUROBOROS_SOFT/HARD_TIMEOUT_SEC` live
   again with a display-only note, plus new PRESENCE/SUBAGENTS/CLAUDEXOR/
   REVIEW_NATIVE_* keys), RETIRED_SETTING_KEYS (upstream itself retired only
   PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC and kept SOFT/HARD live — the
   reference's D04 retirement of those two knobs is DIVERGENT-SUPERSEDED and
   must be re-derived in its own return, not replayed), ENDPOINT_AUTHORED_
   SETTINGS (+OUROBOROS_SUBAGENT_PRESET_RECEIPT) and OPENROUTER_REVIEW_DEFAULTS
   (routed advisory id + comment). Leaf emitted FULL from BOTH parents (the
   shared-leaf convention: drift-probe per parent separately; the final
   transplant --check runs against the two parents concatenated into one
   upstream source so every span is verified in a single exit-0 report).
   provider_models.py was touched ONLY by span removal + the settings_defaults
   re-export import; its call-time `from ouroboros.config import ...` imports
   are tip truth (D02-owned) and stand.
15. Split rows 887-912 (runtime_limits) — 3 spans byte-falsified by upstream
   drift (get_websearch_timeout_sec docstring; get_search_code_wall_sec now
   routes through _clamped_number_setting; get_max_subagent_depth reads the
   named cap), all re-emitted from tip bytes. STRUCTURAL: upstream reshaped
   `MAX_ACTIVE_SUBAGENTS_HARD_CAP = 500` into the tuple statement
   `MAX_ACTIVE_SUBAGENTS_HARD_CAP, MAX_SUBAGENT_DEPTH_HARD_CAP = 500, 10`;
   the UNROWED twin (consumed by ouroboros/tools/control_delegation.py via
   config) rides the rowed statement into runtime_limits and the facade
   re-exports both — the carried ledger must mint its row at F5. Tool note:
   the hardened --check flags this one statement as `assignment to <complex
   target>` under undeclared_top_level even though BOTH bound names are
   requested symbols (Tuple-target blind spot; every span proof in the same
   report is green, leaf_invariants=[]) — the one lane gate that exits 2 with
   a proven false-positive cause; the tool wants Tuple support at F5.
16. Rows 918-920 (launcher_onboarding, semantic delta D03/settings seam,
   launcher half) — RE-PROVEN applicable and LANDED: the module is
   byte-identical between tip and merge-base (zero upstream drift), so the
   reference bytes apply verbatim; the pin renamed per row 920. The SERVER
   half of the same seam (rows 1080-1081, server.py lifespan) is NOT landed —
   server.py keeps the tip guarded write and its old pin; it returns with the
   D11 lane. Two unrowed oracle test adaptations were mirrored because they
   pin exactly this delta and go red without it: test_onboarding_wizard.py::
   test_the_launcher_onboarding_module_authors_no_onboarding_settings
   (reference bytes) and tests/test_server_runtime.py (launcher clause ->
   `"save_settings(" not in launcher_host`; the server clause KEEPS the tip
   guard-string assertion, diverging from the reference's both-sides form
   until D11 lands).
17. Rows 913-917 (the rest of the D03 settings seam: config.py
   normalize_settings_raw/serialize_settings, gateway/owner_settings digest +
   locked update, packaged_cli writer) — HOT-DEFERRED: upstream rewrote
   load_settings_lock_held's read path through the NEW post-cutoff
   settings_integrity module (read_settings_json_verified /
   SettingsIntegrityError raise-through), which the reference does not have;
   replaying the reference seam verbatim would revert the integrity feature
   (the re-prove-trap class, entry 3). The whole seam machinery re-derives
   against tip bytes in its own return; its pin tests/test_settings_read_seam.py
   (a DOMAIN_MAP D12 pin) defers WITH the machinery — not transplanted by this
   lane.
18. Pin adaptations recorded: test_settings_env_on_disk.py re-pinned one
   literal to tip bytes (ENDPOINT_AUTHORED_SETTINGS gains
   OUROBOROS_SUBAGENT_PRESET_RECEIPT — same upstream train as entry 14);
   test_config_extraction.py gains MAX_SUBAGENT_DEPTH_HARD_CAP in the owner
   inventory, Tuple-target parsing in its _top_level_names helper, and a
   narrowed provider_models clause (the reference's "no ouroboros.config
   import anywhere" + top-level model_slots import clauses type against the
   reference's D02 rework of provider_models and return with the D02 lane;
   the surviving clauses pin no IMPORT-TIME config read and leaf-object
   identity of both moved literals).
19. settings_integrity.py — NEW upstream module (post-cutoff, absent from the
   reference and the merge base), already D12 in scripts/v7next_domains.toml;
   no ledger rows; upstream bytes stand. Non-split D12 modules re-proven:
   colab_bootstrap.py / onboarding_wizard.py / secret_masking.py /
   update_channels.py byte-identical across tip==ref==merge-base;
   settings_setup_contract.py / subscription_install_presets.py pure upstream
   drift (ref==merge-base, zero v7 delta) — upstream bytes stand.

## From the integration seam (coordinator, base 0859b681, 2026-08-30)
1. Superseding note to D04 entry "four landed leaves": the lane landed FIVE
   registry leaves (tool_context, tool_catalog, tool_resolution,
   registry_guards, registry_guard_process) — the list in that entry is the
   authority, its count word is a typo (wave-2 conformance review item 6).
2. Superseding note to D12 entry on the tuple-target gate: the verifier fix
   landed in the wave-2 seam commit (unfold at any depth; non-Name leaves are
   complex targets; probes in tests/test_v7next_transplant.py) — the "future
   work / exits 2" claim in that entry is superseded.
3. Seam repair: [split_pending] registry row carries domain IDs again
   (["D04"]) and [split_pending_leaves] carries the two hot-deferred leaves —
   the first seam commit wrote the leaf list into the wrong section.
## From the D05 lane (base 0859b681, 2026-08-30)
1. Shell split rows 416-464 RE-PROVEN against tip bytes and landed. shell_process
   (11 spans, rows 416-426) and shell_effects (12 spans, rows 453-464): every
   reference-leaf span ast=tokens=bytes=True against
   `git show HEAD:ouroboros/tools/shell.py` (drift-probe first, exit 0) — the
   reference leaves ARE tip bytes, adopted verbatim. shell_outputs: only 16 of
   the row set remain in the tip monolith; 14 byte-identical, 2 re-emitted from
   tip: `_register_process_outputs` (ref moved, tip==merge-base — the
   reference's typed-cutover 3-tuple/artifact_registered plumbing is a PURE V7
   DELTA, deliberately NOT ported, rides with the F2 typed-result organ) and
   `_resolve_declared_output` (ref==merge-base, tip moved — PURE UPSTREAM DRIFT:
   the lexical deliverables/casefold machinery; tip bytes are the leaf).
2. Rows 429 (`_allowed_output_roots`), 439 (`_UNDECLARED_OUTPUTS_MARKER`), 445-451
   (the six output/user-file regexes + `_OUTPUT_STAT_SLACK_SEC`) and 452
   (`_mentioned_user_file_outputs_without_declaration`) — SUPERSEDED-BY-UPSTREAM
   as shell_outputs rows: upstream c7315c57 ("Relax scoped browser, native-read,
   and Deliverables false blocks") extracted those ten owners into its own NEW
   leaf `ouroboros/tools/shell_audit.py` (D05-owned, no ledger rows), and the tip
   facade already aliases/imports them from there. The carried ledger renames the
   destination of those ten rows at F5; the facade identity contract
   (tests/test_shell_extraction.py) covers them at their upstream owner.
3. Core split rows 311-349 RE-PROVEN against tip bytes and landed
   (core_file_tools 30 spans incl. row 311's tip alias form
   `_SKILL_OWNER_STATE_FILENAMES = SKILL_OWNER_STATE_FILENAMES`; core_artifacts
   9 spans): 29/39 byte-identical between the reference leaves and
   `git show HEAD:ouroboros/tools/core.py`; 10 BYTE-FALSIFIED as copy sources,
   ALL of the same class — tip==merge-base, reference moved (the typed-result
   cutover producers `_repo_read/_repo_list/_data_read/_data_list/_read_file/
   _list_files/_access_or_block/_send_photo/_send_video/_send_file`, i.e. the
   rows whose own notes disclose `_publish_tool_result`, including row 332's
   A.20 marker change). Tip bodies moved verbatim; the typed deltas ride with
   the F2 typed-result organ (same class as D04 entry 2). Both emitted leaves
   are proof-green (ast=tokens=bytes on every span, leaf_invariants=[], exit 0).
4. FACADE CONVENTION DIVERGENCE (disclosed): the reference cut core.py over with
   NO facade (rows 311-349 carry "-" in the re-export column; consumers rebound
   by rows 360-371 and unrowed edits to vision/query_code/edit_ops/
   delegate_output/shell_guards). This tree keeps a re-export facade on
   tools/core.py instead (the §5.3-Δ2 item-12 partial-split idiom, matching the
   shell facade): the tip consumer surface grew far beyond the reference's (6
   production modules + 20+ test files import the moved names from tools.core
   at this tip), and a no-facade cutover is a pure-hygiene consumer rebind that
   can land as its own wave at F5 without re-proving spans. Identity is pinned
   (`core.X is core_file_tools.X / core_artifacts.X`,
   tests/test_core_extraction.py::test_core_facade_reexports_every_moved_identity);
   the reference's `isdisjoint(vars(core))` clause is replaced by that pin.
5. Rowed TEST bindings landed: rows 363-371 (test_send_file/photo/video ->
   core_artifacts) and row 361 (test_filesystem_root_observability::_read_file ->
   core_file_tools). Row 362 (tests/test_headless_cli.py::_repo_read): the D17
   split moved that consumer into tests/test_headless_workspace_shell.py (D17
   lane entry 9 reverse-mapped it to the upstream spelling); this lane completed
   the row at its successor location (core_file_tools binding). Row 360
   (browser.py::_readonly_subagent) NOT landed: the reference's browser delta
   bundles a D01 rebinding (`loop_messages._append_or_merge_user_content`)
   absent from this tree; the facade preserves the exact object meanwhile —
   rides with the consumer-rebind wave.
6. Cross-domain core rows already satisfied at tip (SUPERSEDED-BY-UPSTREAM
   class, no action): the five tool_access rows (active_tool_profile,
   build_resolved_resource_binding, decide_tool_access, normalize_root,
   normalize_runtime_data_path — tip core.py imports them from
   ouroboros.tool_access), read_text -> utils, row 353 active_repo_dir_for ->
   tool_resolution (import alias, per D04 entry 7), _filter_out_project_store ->
   project_facts (per D15 entry 1), and the two contracts/skill_payload_policy
   rows (tip imports them as the `_policy_*` aliases). Registry rows 213/214
   (python_interpreter) and 246/247 (artifacts) ride with the HOT-DEFERRED
   registry_core leaf (D04 entry 3): tip registry.py still carries those import
   bindings (:59, :83); the protected file was not touched by this lane.
7. Unrowed reference deltas NOT replayed (candidate rows for the carried
   ledger): (a) code_intelligence.py `collect_top_level_python_imports` (+92) —
   its only consumer is the reference-only tests/test_top_level_import_graph.py
   (domain-graph tooling; F5/quotient territory); (b) mcp_client.py ToolResult
   cutover + `tool_name_collisions` field — F2 typed organ; (c) services.py
   `_publish_tool_result` cutover coupled to the 3-tuple
   `_register_process_outputs` — F2; (d) health.py module-debt band rendering —
   types against reference-only ratchet metrics keys (`module_debt_1500_active`
   etc.) that no producer on this tree emits, and the owner's Q11=B decision
   picked the upstream size law — DIVERGENT-SUPERSEDED, re-derive only if the
   debt-band UI returns; (e) vision/query_code/edit_ops/delegate_output/
   shell_guards import rebinds — pending with the consumer-rebind wave (all
   keep working through the facade).
8. Oracle test adaptations mirrored in this tree's equivalents (§5.3-Δ2 item
   10): load_settings monkeypatches retargeted to shell_process (its only
   reader moved there) in tests/test_shell_run_shell.py and
   tests/test_iteration2_fixes.py; module-object patch handles retargeted to
   core_file_tools in tests/test_repo_read_limits.py (read_text),
   tests/test_runtime_reliability_v655.py (_list_dir) and
   tests/test_workspace_authority_binding.py (build_resolved_resource_binding).
   Path-keyed mirror: tests/test_process_custody.py `_POPEN_ALLOWLIST` row
   "ouroboros/tools/shell.py" -> "ouroboros/tools/shell_process.py" (the
   facade's only Popen site moved with `_tracked_subprocess_run`; suite green).
9. Zero-v7-delta re-proofs for the rest of the domain: media.py /
   python_interpreter.py / code_search_rg.py byte-identical tip==ref==merge-base;
   artifacts.py / recent_tasks.py / search.py / verify.py pure upstream drift
   (ref==merge-base) — upstream bytes stand; shell_audit.py NEW upstream module
   (no rows, see entry 2). tools/core.py band re-entry (2283 -> 1373) recorded
   via the official regenerator's --band-rationale.
## From the D02 lane (base 0859b681, 2026-08-30)
1. llm.py split rows 1666-1793 + 4001-4003 (131 rows, ten leaves) RE-PROVEN
   against tip bytes: 100 spans byte-identical between the reference leaves and
   `git show HEAD:ouroboros/llm.py`; 28 spans BYTE-FALSIFIED as copy sources by
   PURE UPSTREAM DRIFT (oracle==merge-base for every non-D09 one) and re-emitted
   from tip bytes. The drift is the post-cutoff provider train: request-wire
   custody (041e6e39, issue-229 phase 2b — request_wire_scoped decorators and
   wire send/receipt hooks inside the send drivers and lanes), OpenRouter
   attribution rework (9a20df6a — OPENROUTER_APP_HEADERS), anthropic native
   custody (native_content_for_replay/retain_native_assistant_content),
   timeout/custody hardening (802f1056, f702439f). Transplant-tool verify:
   every module-level span ast=tokens=bytes=True, undeclared_top_level=[],
   leaf_invariants=[], plus a member-level byte proof for all 10 mixins
   (117 members byte-identical to the tip LLMClient members).
2. Row 1674 (`_applied_payload_cache_ttl`) — the ledger's own documented
   one-identifier requalification (LLMClient -> _PayloadCachePolicyMixin) kept
   from the reference; the only non-tip-byte span in the split besides row 1784.
3. Row 1784 (`_chat_local`, semantic id D09, the approved one-attempt delta) —
   CARRIED, but BYTE-FALSIFIED as a verbatim copy source: upstream 802f1056
   added the exception-owned capture custody clause INSIDE the retry loop the
   delta deletes; replaying the reference span verbatim would have silently
   reverted that upstream clause (the re-prove-trap class, D15 entry 3). The
   delta was re-derived on tip bytes: the `for attempt in range(3)` loop and its
   sleep/last_exc arms are gone (one physical attempt per call, transient
   failures surface to call_llm_with_retry), the custody clause and the
   warning/error identities are preserved. Pins: the reference's
   test_local_transport_makes_exactly_one_physical_attempt carried into
   tests/test_context_overflow_hint.py; the two sibling local-lane tests
   re-pinned per the reference (attempt count 3 -> 1, monkeypatches renamed to
   the owner leaf llm_local); upstream's post-cutoff
   test_local_retry_does_not_inherit_unrelated_physical_capture re-pinned the
   same way (its durable fact — exception-owned capture only, never the
   ContextVar — is unchanged; its `calls == 3` pinned the deleted loop).
4. D09 typed-policy-refusal subfamily (rows 1706, 1749, 1751, 1759, 1760 and
   the reference-only llm_attempt symbols PROVIDER_POLICY_REFUSAL /
   ProviderPolicyRefusal / _is_provider_policy_refusal) — HOT-DEFERRED with
   evidence: zero occurrences of `provider_policy_refusal` anywhere at this tip
   (no raiser, no classifier — loop_llm_call has no such code), and all five
   consuming ladder bodies drifted upstream (802f1056/f702439f hardened them);
   the refusal never surfaces without its D01-side classification, so carrying
   only the ladder half would ship dead semantics onto reworked bytes. The five
   bodies moved as TIP bytes; the reference pins
   tests/test_llm_typed_policy_refusal.py and the two
   `typed_policy_refusal` golden cases (fallback_ladder.json 17 -> 15) are NOT
   carried — they return with the delta's own re-derivation.
5. UNROWED tip symbols `_RESPONSE_METADATA_LABEL_MAX_CHARS` and
   `_bounded_response_metadata_label` (post-cutoff, llm.py top level) moved to
   ouroboros/llm_openai_compatible.py with their ONLY reader
   (`_normalize_remote_response`, row 1788); the facade re-exports both, so the
   tip import surface is unchanged. Candidate rows for the carried ledger at F5.
6. provider_models rows 840-886/3238-3241 note-contract COMPLETED: the rows'
   own notes say "provider_models now imports this leaf instead of lazily
   importing config"; D12 landed the leaves and left the consumption to D02.
   The two remaining call-time `from ouroboros.config import ...` reads
   (parse_fallback_chain at resolve_credentialed_model, SETTINGS_DEFAULTS at
   declared_model_settings) are now top-level leaf imports
   (model_slots/settings_defaults; cycle-free, verified at import). The
   reference pin test_provider_models_reads_the_shared_leaves_instead_of_
   importing_config is restored under its ledger name, superseding the D12
   lane's disclosed placeholder test_provider_models_reads_the_shared_defaults_
   leaf (its identity clauses are kept as a superset). Upstream's own
   provider_models evolution (ACTIVE/LEGACY_MODEL_SETTING_KEYS,
   *_in_settings twins, CLAUDE_CODE_MODEL retirement) is tip truth and stands.
7. ouroboros/llm_probe.py reference delta (+8/-6, tip==merge-base) ADOPTED
   verbatim: the lazy executor import redirects from the llm.py facade to the
   owner leaf llm_attempt (an llm_* leaf never imports its parent). Unrowed in
   MIGRATION; required by the leaf rule the carried pin
   tests/test_llm_extraction.py::test_llm_leaves_never_import_their_parent
   enforces. Candidate row at F5.
8. Provider-route goldens (tests/fixtures/llm_golden, 9 files) RE-BASELINED
   from tip behaviour via the suite's own `--write` entry: every diff class maps
   to a named upstream train — attribution headers (X-Title ->
   X-OpenRouter-Title + new referer, 9a20df6a), the `request_wire` disclosure
   block in usage (041e6e39), bounded `response_finish_reason` /
   `response_provider` labels, effort/dialect-ladder evolution, anthropic
   native-content retention. One suite adaptation: the per-process random
   `usage.request_wire.attempt_id` is projected to a presence flag (exactly the
   suite's existing ledger_attempt_ids treatment) — without it the recording is
   nondeterministic across processes.
9. Dead-patch class closed across tests: after the split,
   `execute_physical_attempt(_async)` is read in llm_attempt,
   `_execute_candidate`/`last_physical_attempt_capture` on the chat path in
   llm_fallback, and the local lane's executor in llm_local. Reference
   adaptations applied (test_capability_probe_accounting_v664,
   test_prompt_cache_v664, test_retry_bypass_response_cache verbatim —
   tip==base; test_effort_floor_v6732, test_usage_scope_transport_v664,
   test_provider_key_test re-derived on tip bytes); the same rule applied to
   two POST-CUTOFF upstream tests the reference never saw
   (test_openai_chat_dispatch, test_issue229_synthesis — llm ->
   llm_fallback, disclosed in-file); path-keyed mirror
   test_review_prompt_caching::test_global_ttl_docstrings_name_every_consumer
   re-pinned to `ouroboros/llm_attempt.py` (matches the reference's own bytes
   for that clause). Patches of names the facade still OWNS or that are read
   lazily through it (test_pricing fetch_openrouter_pricing, test_web_search
   server tools, all LLMClient-method patches) verified live and untouched.
10. Reference adaptations NOT carried (other domains' v7 spellings,
   reverse-mapped to tip per §5.3-Δ item 2): tests/test_multimodal_chat.py and
   tests/test_provider_failure_reporting.py retarget imports to
   loop_messages/loop_round_limits (D01 leaves absent here — tip bytes stand;
   tip already re-homed _provider_recovery_hint into loop_transport itself);
   the same import line in tests/test_context_overflow_hint.py keeps the tip
   spelling.

## From the D14 lane (base 92238298, 2026-08-30)
1. extension_loader.py split rows 2467-2519 (53 rows, six leaves) RE-PROVEN
   against tip bytes: 49 spans byte-identical between the reference leaves and
   `git show HEAD:ouroboros/extension_loader.py`; 4 spans BYTE-FALSIFIED as
   copy sources by PURE UPSTREAM DRIFT (oracle==merge-base 8028f1df for every
   one) and re-emitted from tip bytes: `_validate_child_ui_descriptor` and
   `PluginAPIImpl` (widget-geometry promotion `_widget_geometry_from_render`),
   `runtime_state_for_skill_name` / `runtime_state_for_loaded_skill` (durable
   companion-health overlay `_apply_durable_extension_health`). Transplant-tool
   verify per leaf: every span ast=tokens=bytes=True, undeclared_top_level=[],
   leaf_invariants=[], exit 0 (80 spans across the ten leaves of this lane).
2. UNROWED tip riders (candidate rows for the carried ledger):
   `_widget_geometry_from_render` -> ouroboros/extension_surface_names.py
   (readers live in two leaves — child_catalog and plugin_api — and it is the
   theme sibling of rowed `_widget_span_from_render`, which those same leaves
   already import); `_apply_durable_extension_health` ->
   ouroboros/extension_liveness.py (its only readers are the two moved
   runtime_state_* spans). The facade re-exports both; the carried identity
   suite pins both owners.
3. Row 2519-family `_ws_broadcaster`: moved to extension_plugin_api.py and
   deliberately NOT aliased on the facade (rebindable module global — a
   facade copy would freeze the value); RE-CONFIRMED as the reference
   contract, pinned by tests/test_extension_loader_extraction.py::
   test_the_broadcaster_slot_has_exactly_one_binding. server.py reaches it
   only through re-exported `set_ws_broadcaster`.
4. skill_review.py split rows (31 rows, four leaves): 25 executed from tip
   bytes. SIX rows SUPERSEDED by upstream's own re-decomposition (386e9417
   "Max Review Cycles" moved the accepted-rebuttal ledger and the wave-budget
   refusal whole into ouroboros/skill_review_cycles.py before this lane):
   `_accepted_rebuttals_path`, `_load_accepted_rebuttals`,
   `_persist_rebuttal_flips`, `_fail_items_from_history_entry`,
   `_record_accepted_rebuttal` (rebuttals-leaf rows) and
   `_review_wave_budget_block` (prompt-leaf row). Upstream ownership stands;
   the facade keeps the historical underscore aliases via tip's own cycles
   import; the carried identity suite pins that alias identity. The rebuttals
   leaf was emitted with its four remaining rows; the prompt leaf imports
   `load_accepted_rebuttals` from skill_review_cycles (tip truth), not from
   the rebuttals leaf as in the reference.
5. skill_review drifted spans re-emitted from tip bytes (5): `_read_skill_text`
   + `_build_skill_file_packs` (payload-snapshot digest gate,
   expected_content_hash), `_build_review_prompt` +
   `_run_skill_advisory_pre_review` (provider-neutral advisory critic rework
   f8d87c69 — "Optional Advisory Pre-Review", run_advisory_critic, hasattr
   no-op trap removed), `render_skill_review_block` (slot_id actor keys,
   distinct-item count, sanitize_tool_result_for_log).
6. Test rows tests/test_extension_loader.py (45, five siblings + shared):
   tip file has ZERO upstream drift since merge-base; 43 moved bodies
   byte-identical, 1 reference adaptation KEPT (dual supervisor patch in
   test_server_pickup_spawns_stops_and_redrives_missing_companion — PluginAPI
   owner reads the supervisor from its own leaf), 1 reference spelling
   REVERSE-MAPPED to tip (worker_main lives in supervisor/workers.py at this
   tip; the reference's supervisor/worker_process.py is the D08 split still
   pending here). Lossless: 52 test names before == 52 after, zero dup names.
7. Test rows tests/test_skill_review.py (65, five siblings + shared): 58
   moved bodies byte-identical; 3 re-emitted from tip bytes (pure test drift:
   advisory_model_credentials_missing label, provider-neutral advisory
   heading, review-delivery capture in
   test_review_skill_prompt_loads_core_governance_artifacts); 3 reference
   adaptations KEPT (patch retargets to leaf owners in
   test_review_skill_quorum_failure_on_one_responder and the two pack-budget
   tests). Row `test_skill_advisory_private_guards_precede_availability`
   SOURCE-FALSIFIED: upstream f8d87c69 deleted the test and replaced it with
   `test_skill_advisory_pytest_guard_precedes_availability` +
   `test_skill_advisory_missing_internal_symbol_is_loud_not_silent`; per the
   wave-2 rule the successors stay in the remainder with tip bytes (theme
   re-home is F5) and the reference copy of the deleted test was NOT carried.
   Lossless: 74 test names before == 74 after, zero dup names.
8. Identity suites carried: tests/test_extension_loader_extraction.py gains
   the two rider rows of entry 2; tests/test_skill_review_extraction.py
   adapted to tip — the reference's tool_module_inventory clauses dropped
   (v7-only mechanism, module absent at this tip; F5 restores it with its
   owner), a cycles-alias identity test added for the six superseded names,
   the facade size bound relaxed 800 -> 900 (tip retains the cycles gate,
   paid-fact stamping and _persist_reviewed_outcome the oracle-era monolith
   did not have), and the three tip-retained lifecycle members added to the
   patchable-seams pin.
9. Dead-patch class closed: the remainder's
   `patch("ouroboros.skill_review._run_skill_advisory_pre_review", ...)`
   retargeted to the prompt owner (mirrors the reference remainder :314);
   tests/test_extension_companion.py dual-patches get_global_supervisor on
   extension_plugin_api + extension_loader (2 tests, mirrors the reference
   adaptation; the single-module patch was proven dead by a red run). Every
   other facade-level patch site of moved names was verified LIVE: all
   production consumers of is_extension_live / runtime_state_for_* /
   `_lock`+`_tools` (skill_loader:1414) do call-time facade imports.
10. NOT carried, no ledger rows: the 8 post-cutoff D14 modules
   (betterleaks_runtime, skill_payload_binding, skill_publish_github/result/
   scanner/snapshot — secret-safe publishing train 8cc2ac69;
   skill_review_cycles — 386e9417; skill_review_usage — f18da8c3) stand on
   upstream bytes untouched. The reference's UNROWED `failure_kind` delta on
   ouroboros/extension_process_runner.py (typed timeout classification,
   consumed by the reference's tools/extension_dispatch.py:187) is NOT
   replayed — typed-dispatch family, Ф3 territory; tip bytes stand. The
   supervised-future leak (tip extension_plugin_api.py span of PluginAPIImpl)
   is preserved as-is per the plan (Ф3-acceptance carries the direct
   regression test). Pre-existing at base, untouched, for the record: 10
   ast-identical duplicate test bodies between
   tests/test_review_cycles_dispatch.py and
   tests/test_review_cycles_skill_dispatch.py.
## From the D08 lane (base 92238298, 2026-08-30)

1. Scope executed (the QUIET part): 16 leaves landed from tip bytes with the
   transplant tool (ast=tokens=byte-roundtrip=True on every span, exit 0,
   leaf_invariants=[], unread_declared=[]): control_events (rows 2520-2528),
   control_routing (2529-2536, 3954), control_runtime (2557-2568) — the D08
   half of the SHARED D07/D08 tools/control.py; queue_schedules (2029-2040 +
   alias rows 3950-3952); worker_promotion (2045-2054), worker_chat_lane
   (2055-2060), worker_pool_lifecycle (2065-2076), worker_process (1024-1029);
   events_chat_delivery (921, 923-929), events_budget (980-984),
   events_coop_checkpoint (964-969), events_project_routing (955-963),
   events_schedule_task (945-949, 951, 953-954), events_subagent_admission
   (930-944), events_worker_reports (985-991), events_runtime_controls
   (993-997). Facades = tip parent − moved spans + grouped re-export block
   (noqa discipline); facade audit green: every kept def/assign span
   byte-identical to `git show HEAD:<monolith>`, every moved name re-exported.
2. Drift-probe results (reference leaf --check against tip bytes, first step
   per leaf): whole-leaf byte-true — control_events 9/9, queue_schedules
   12/12, events_coop_checkpoint 6/6, events_subagent_admission 15/15;
   byte-falsified by pure upstream drift and re-emitted from tip bytes —
   control_routing 5/9 spans, control_runtime 7/12, worker_promotion 3/10,
   worker_chat_lane 2/6, worker_pool_lifecycle 2/12, worker_process 2/6,
   events_chat_delivery 4/8, events_budget 3/5, events_project_routing 6/9,
   events_schedule_task 2/9, events_worker_reports 4/7,
   events_runtime_controls 1/5. "Verbatim" in the ledger was re-proven by
   bytes in every case; no oracle semantics were replayed over tip drift.
3. SHARED-file convention (tools/control.py, D07/D08): this lane moved ONLY
   the D08 rows (control_events/routing/runtime per DOMAIN_MAP); the D07 rows
   (control_scheduling 2543-2556, control_subagent_spec 2537-2542,
   control_task_results 2569-2579) remain in the facade untouched for the D07
   lane. Unrowed post-cutoff predecessor-authority family
   (_MISSING_PREDECESSOR_SELECTOR, _predecessor_selector_error,
   _attach_predecessor_authority_from_metadata) rides with its only readers
   (_promote_chat_to_task/_route_to_project) into control_routing — a
   def-time default-argument read of the sentinel makes a facade-retained
   copy structurally impossible (F5 theme for the ledger's unrowed census).
4. HOT-DEFERRED, cancel/custody class (D09; upstream 65b5d19f re-decomposed
   this ownership — replaying the reference rows would be a second answer):
   - events_task_done rows 972-979: _resolve_lifecycle_fault reads
     cancel_intents, _maybe_notify_provider_death reads task_lifecycle,
     _task_done_durable_fault operates terminalization custody; the family is
     one dispatch cluster, deferred whole.
   - events_runtime_controls row 992 (_handle_cancel_task): the cancel ingress
     handler itself.
   - row 970 (_close_campaign_after_owner_stop -> queue_transitions.py) and
     row 971 (events_evolution_done): owner-stop family; 65b5d19f made
     queue_transitions.py its cancel-transition dumping ground, and the
     evolution-done handler calls the deferred campaign-closure symbol as a
     bare local name.
   - queue_snapshot rows 2017-2020: restore_pending_from_snapshot restores
     terminalization-retry rows and consults cancel_intents.has_active_intent
     (65b5d19f machinery); persist snapshots the same fences. Deferred whole
     (parse_iso_to_ts/_kept_service_pids ride only with their family).
   - queue_timeouts rows 2021-2028: _enforce_task_timeouts_locked drives
     cancel_intents/task_reaper/owner_stop.
   - queue_evolution rows 2041-2044: upstream itself moved
     _deliver_pending_owner_report/enqueue_evolution_task_if_needed into its
     own supervisor/evolution_lifecycle.py (65b5d19f); creating the reference
     leaf beside it would fork evolution-family ownership.
   - worker_assignment rows 2077-2079 (assign_tasks reshaped by 65b5d19f's
     600-line workers.py rework; _cancel_unauthorized_evolution) and
     worker_health rows 2061-2064 (_ensure_workers_healthy_locked writes
     STATUS_CANCELLED terminal outcomes and terminalizes admission-blocked
     retries). Both families stay on the facade.
5. Deferred SEMANTIC-DELTA rows (unsanctioned for this lane; tip bytes stand):
   1014-1015 (dispatch_event/EVENT_HANDLERS, delta D06 events taxonomy — the
   event_taxonomy.py leaf and tests/test_event_taxonomy.py are NOT created);
   1021/1022/2082 (queue.init/workers.init/refresh_timeouts_from_settings,
   delta D04 retired settings knobs — Q10/F3 territory); retired rows
   1017-1019, 1030, 2080-2081 (SOFT/HARD_TIMEOUT_SEC, TOTAL_BUDGET_LIMIT,
   QUEUE_SNAPSHOT_PATH — deletions are semantics, not relocation).
6. Row 2016 (_handle_schedule_task -> events_schedule_task.py) DEFERRED with
   a mechanism finding: the function carries the >300-line FUNCTION_DEBT entry
   keyed by (path, qualname), and THIS tree's transition validator
   (ouroboros/review.py::validate_manifest_transition) has no same-qualname
   relocation rule — that rule is reference delta D11, ratchet machinery out
   of this lane's bounds. The handler stays in the facade with its debt key;
   the eight quiet schedule-family rows moved. Every seam name it reads
   (_find_duplicate_task etc.) binds through the facade re-export, so existing
   facade-targeted test patches keep intercepting (verified green).
7. Reverse-mapped preamble spots (oracle spelling -> tip truth): queue_schedules
   `from supervisor.task_lifecycle import record_scheduled_admission` ->
   `from supervisor.task_admission import ...` (65b5d19f moved it); the two
   control leaves' `from ouroboros.tools.tool_result import ToolResult,
   _publish_tool_result` deleted — the module does not exist at tip (D04 lane
   hot-deferred that organ) and no tip span reads the names; alias mirrors
   from tip parents: _bound_project_chat_id (supervisor/log_addressing.py,
   upstream's own extraction), _build_scheduled_task_payload
   (supervisor/task_dispatch.py), _reject_if_no_chat_target
   (supervisor/task_admission.py), _once_due/_prune_consumed_once/
   _record_last_error (supervisor/schedule_time.py, rows 3950-3952 satisfied
   as leaf preamble imports exactly like the tip parent).
8. Handle idiom: queue_schedules/_queue, worker_promotion|chat_lane|
   pool_lifecycle/_pool declared sets re-derived on tip bytes (they grew past
   the reference table by the post-cutoff facade helpers:
   _announce_created_project, _apply_presence_promotion_authority,
   _promoted_scheduled_outcome, _reject_promoted_after_attachment_stage,
   _relocate_promoted_attachments, _stage_promoted_initial_attachments,
   _reconcile_confirmed_dead_review_owner); events_project_routing gained the
   D33-family handle `_events` for the single unrowed facade helper
   _routing_attachments. All sets pinned in
   tests/test_module_handle_extraction.py::LEAVES.
9. Path-keyed mirrors (Δ2 п.10): HOT_CODE_PATHS (supervisor/update_merge_policy.py)
   += the 12 carried hot leaves (D04-block precedent); FUNCTION_DEBT key NOT
   relocated (see 6); conftest _SERIAL_TEST_FILES needed no new rows (the new
   suites are structural). Dead-patch class re-pointed to owner leaves,
   mirroring the reference adaptations: test_coop_checkpoint_quiescence
   (events_coop_checkpoint, events_subagent_admission), test_evolution_redesign
   (queue_schedules._last_skill_schedule_sync), test_schedule_followup
   (queue_schedules._write_scheduled_tasks), test_worker_crash_retry
   (supervisor.worker_process trio), test_promote_chat_flow
   (control_events._wait_for_promotion_admission,
   control_routing._promotion_pool_disabled_from_snapshot),
   test_evolution_restart_claims (`control_runtime as control`, the reference's
   exact alias form), test_task_status_flow (control_runtime run_cmd/
   atomic_write_json), test_extension_loader (worker_main scan reads
   supervisor/worker_process.py), test_process_resource_leaks (reference
   bodies verbatim). All touched test files LOSSLESS (name multisets equal).
10. Pre-existing observation, NOT this lane's defect: tests/
   test_review_cycles_dispatch.py and tests/test_review_cycles_skill_dispatch.py
   share 10 ast-identical test bodies at the base SHA (D15-class dup, D06
   domain) — left for the D06 lane.
11. Unrowed tip top-level symbols stayed in their facades (F5 census):
   events.py _handle_main_llm_call_state/_parent_delegation_budget/
   _routing_attachments; queue.py 26 names (fences/admission/cancel seam);
   workers.py 88 names (65b5d19f terminalization-retry/custody machinery);
   control.py HIDDEN_LEGACY_SCHEDULE_PARAMS, _context_task_depth,
   _materialize_child_attachment_manifest, maybe_emit_delegated_run_fanout,
   get_tools + the predecessor family that rode into control_routing.
## From the integration seam (coordinator, D13 dispositions, 2026-08-30)
1. safety.py row 1016 (retire module-level supervisor import + _record_safety_usage,
   pin test_safety_module_has_no_import_time_dependency_on_the_supervisor) —
   LIVE, NOT landed on tip (import at :25, call at :1010). HOT-DEFERRED:
   protected file; rides the protected-surface wave (F2/F3) with owner-visible
   handling.
2. UNROWED live delta `_safety_drive_root` (fixes cwd-relative "../data" in
   safety.py, tip site :899; oracle had no ledger row, prose-only in
   DOMAIN_MAP). MUST gain a carried-ledger row before any replay; tip drift
   collapsed two mb sites into one — replay needs re-derivation. Candidate
   for the F5 carried-ledger mint. RISK: without this note the only useful
   unrowed safety delta would be silently lost.
3. shell_guards.py lazy-import rebind (tools.core → core_file_tools) —
   confirmed pending with the D05 consumer-rebind wave (D05 ledger §4(e));
   chain alive through the facade on tip.
4. runtime_mode_policy oracle delta remainder: registry_core.py +
   tool_result.py protection closure returns WITH those two hot-deferred
   leaves (D04 ledger §11); GIT_OPS_FAMILY_PATHS / RELEASE_INVARIANT_PATHS
   re-cut returns with the G1 git_ops split (D10 wave) — recording now would
   protect nonexistent files.
5. D13 census note: tip toml gives D13 eight owners vs oracle DOMAIN_MAP six —
   write_shape.py and deliverables_shell.py are new upstream surfaces
   post-freeze; not an oracle gap.
## From the D11 lane (base a56bb76a, 2026-08-30)
1. server.py split rows 1034-1078 + 3948-3949 (47 symbol rows): 43 landed into
   the six reference leaves (process 5, routing_context 13, owner_routing 5,
   liveness 4, maintenance 11, restart 5). Drift-probe FIRST per leaf: the
   reference leaves are byte-true against tip except 10 spans byte-falsified
   by upstream drift — _task_result_ground_truth (authority_source block),
   _stage_mailbox_attachments / _route_project_chat_to_running_task /
   _record_routing_receipt / _route_owner_message (attachment-report train),
   _start_supervisor_liveness_watchdog (OB-03 monotonic clock + pid-keyed
   toast), _periodic_supervisor_maintenance / _reconcile_delegated_runs
   (child-ref promotion replay + terminal-reconciliation refresh),
   _managed_update_pending_kwargs / _perform_supervisor_restart
   (planned-handoff train). All 43 landed spans emitted from tip bytes by the
   hardened transplant tool; --check green on every span (ast=tokens=bytes),
   leaf_invariants=[], no oracle semantics replayed over drift.
2. HOT-DEFERRED rows 1070/1072/1073/1074 (_pending_restart,
   _handle_restart_in_supervisor, _check_pending_restart_drain,
   _perform_supervisor_restart): the upstream delegation train re-decomposed
   restart ownership — _perform_supervisor_restart now WRITES the new module
   global _planned_delegate_restart_transaction_id that server.main() reads at
   the re-exec point; a byte-preserving relocation would fork that state (a
   leaf `global` write is invisible to the facade's from-import binding).
   D09-class "second answer about ownership" -> the four rows stay in the
   facade, the drain record stays beside its only two readers; the deferred
   inventory is pinned as the F2 work order in
   tests/test_server_extraction.py::_SERVER_OWNED.
3. Rows 1080-1081 (server.py::lifespan, semantic delta D03 settings-seam
   server half) HOT-DEFERRED: the reader-side halves of the same seam (rows
   913-917) are hot-deferred by the D12/D17 lanes (upstream rewrote the read
   path through post-cutoff settings_integrity); landing the boot half alone
   would leave provider normalization neither persisted nor re-derived.
   server.py keeps the tip guarded write and its old pin — exactly the state
   the D17 lane's note 16 anticipated.
4. Same-qualname ratchet delta (row 1033, semantic delta id D11) — LANDED.
   ouroboros/review.py verified NOT in the AGENTS.md protected list. The
   relocated_functions block replayed byte-identical from the oracle into the
   tip-shaped validate_manifest_transition (tip keeps its adjacent= interval
   form; the oracle-only MODULE_DEBT_1500 layer was NOT replayed — Q11=B keeps
   the upstream size law). The pin renamed per the row, oracle bytes
   (test_transition_rejects_function_swap_even_at_same_cardinality ->
   test_transition_allows_a_same_qualname_relocation_but_not_a_swap). This
   unblocks the D08 lane's row 2016 deferral (FUNCTION_DEBT relocation of
   _handle_schedule_task).
5. Rows 1192/1259 (theme split into tests/test_delegated_reconciliation.py):
   landed as the D11 SLICE only — the two tests that bind the
   server_maintenance owner. The in-place owner-retarget grew the shrink-only
   byte-debt giant test_delegated_subagent_transport.py by +40 bytes and the
   ratchet refused it; the re-home is the designed pressure valve (the giant
   shrinks 320340 -> 318310, the pin gains its family). The rest of the
   reference sibling (orphan-sweep predicate, absent-run closure, release
   points, _delegated_transport_shared helpers) arrives with the delegation
   organ's test split (F2). Row 1656 (TestStartupGCFailClosed): only the
   DATA_DIR owner-retarget mirrored; that file split also stays with F2.
6. Facade form: top from-import block (reference facade style), not an EOF
   re-export block — forced by module-level reads of moved state (PORT_FILE =
   DATA_DIR / ..., the logging bootstrap) before any def runs; base64 keeps a
   noqa: F401 exactly as the reference facade does (its only user moved).
   Facade audit green: every kept top-level span byte-identical to tip, no
   facade-new symbols, every moved name re-exported by identity. server.py
   3191 -> 1640 lines; it remains a GIANT_PATHS entry (>1600 upstream law) and
   only shrank, so the regenerated manifest changes one number (the transport
   giant's byte debt).
7. Leaf conventions: emitted leaves carry `from __future__ import annotations`
   (transplant-tool requirement; prior-lane convention) and tool span spacing.
   Zero declared names and NO module handles — the reference design homes the
   shared rebindable state in server_process (Events mutated in place, one
   DATA_DIR, one logger), so all six are projection-only leaves.
   Reverse-mapped preamble spots: server_liveness gains `import os` (drift:
   os.getpid() in the toast key); server_restart's preamble/docstring describe
   the landed five rows and name the deferral honestly.
8. Test adaptations mirrored path-keyed to THIS tree (Δ2 p.10): transport
   giant tests -> sm owner (see 5); test_delegated_run_isolation._server_gc ->
   server_maintenance.DATA_DIR (reference form); test_phase3c_observability_gc
   (two post-cutoff tests, no oracle counterpart) -> maintenance owner for
   DATA_DIR/_LAST_CANCEL_INTENT_SWEEP/time; test_project_routing_v664 ->
   server_routing_context patch, compressed to one line so the file stays at
   1000 lines (below the 1001 band); test_client_surface -> owner_routing text
   joined into the client_surface pin (reference form);
   test_ws3_wedge_resilience (post-cutoff OB-03 tests) -> fake clock retargets
   to server_liveness; test_panic_stop_port_sweep floor 5 -> 11 (the return
   the D09 lane's note 12(a) anticipated). Deliberately NOT retargeted:
   patches whose exercised readers stayed in the facade with the deferral
   (test_server_shutdown, test_evolution_restart_claims,
   test_restart_reconnect, test_promote_chat_flow, test_client_surface
   _process_bridge_updates block). All touched test files lossless (the one
   test rename is ledger row 1033; the two re-homed names moved whole).
9. Pre-existing base red, NOT this lane's defect:
   tests/test_smoke.py::test_size_ratchet_transition_against_explicit_base
   fails at pristine a56bb76a (probed in a throwaway worktree: 1 failed + 4
   passed) — parent 7d2dca49's manifest records
   tests/test_devtools_benchmarks.py at 328116 bytes while its own tree holds
   328195 (the +79-byte cherry-pick residue the seam commit message itself
   describes). The (a56bb76a -> this commit) pair is consistent: 327935 ==
   tree at the parent.
10. Module census, 34 D11 owners (tip vs merge-base 8028f1df vs oracle
   9f691656): 13 byte-identical in all three (client_surface,
   gateway/__init__, gateway/files, gateway/logs, gateway/mcp,
   gateway/onboarding_host, gateway/schedules, gateway/task_events,
   gateway/task_hurry, gateway/ui_preferences, server_auth, server_entrypoint,
   server_web); 17 pure upstream drift — tip bytes stand (gateway/_helpers,
   claudexor_accounts, contracts, control, extensions, history, host_service,
   marketplace, models, presence_settings, projects, router, skill_publish,
   state, tasks, ws, server_runtime); 3 carry ONLY D03 settings-seam /
   retired-knob (D04) oracle deltas -> HOT-DEFERRED with that seam
   (gateway/owner_settings, gateway/onboarding, gateway/settings; D12/D17
   precedent); server.py split per 1. Gateway ABI/alias retirements untouched
   (F3 territory); web/ untouched.
## From the D10 lane (base a56bb76a, 2026-08-30)
1. G1 split rows 3430-3457 (supervisor/git_ops.py -> 4 leaves, delta D35
   module-handle) executed for 26 of 28 rows from tip bytes with the transplant
   tool (ast=tokens=byte-roundtrip=True on every span, leaf_invariants=[],
   unread_declared=[], exit 0 per leaf). Drift-probe (reference leaf --check
   against `git show HEAD:supervisor/git_ops.py`, first step per leaf):
   git_ops_rescue 8/8 spans byte-true; git_ops_remotes 3/4 (push_to_remote
   BYTE-FALSIFIED by PURE UPSTREAM DRIFT — a23e12b1 routed the push through the
   bounded network runner; tip bytes emitted); git_ops_updates and
   git_ops_reset byte-true on every span except the two f-string rows below.
2. Rows 3439 (prepare_managed_update) and 3449 (safe_restart) DEFERRED — both
   spans stay facade DEFS: each reads the rebindable parent global BRANCH_DEV
   (safe_restart also BRANCH_STABLE) inside f-strings, and the hardened
   transplant gate fails closed on f-string reads of declared names ("the
   token proof cannot cover f-string internals"). The reference leaves carry a
   manual `_go().BRANCH_DEV` rewrite inside the f-strings, which this wave's
   ast=tokens=bytes gate cannot re-prove (tokens_equal=False on exactly those
   spans in the drift-probe). Their reads were dropped from the declared sets
   (tool-verified unread otherwise); tests/test_git_ops_owner_facades.py pins
   the two names as facade defs. Relocation returns if/when the tool grows
   f-string token support (D12 lane already noted the same gate wants Tuple
   support — same F5 tool-work theme).
3. git_ops rows 1031-1032 (DRIVE_ROOT/REPO_DIR config-aware pre-init defaults,
   semantic id D13) — HOT-DEFERRED: live semantic delta to a protected file
   (tip still binds `pathlib.Path.home()/"Ouroboros"` at :26-27, upstream did
   NOT absorb the hermetic-isolation fix). Not byte-preserving relocation, so
   out of this lane's mandate; rides the protected-surface wave with
   owner-visible handling (same class as the coordinator's safety.py row 1016
   disposition). Its pin tests/test_git_ops_default_roots.py is NOT carried.
4. update_merge split rows 3426-3429 (-> supervisor/update_merge_plan.py) —
   HOT-DEFERRED WHOLE with the update engine (F2 organ): rows 3427-3429 carry
   semantic id D34 (carrier engine insertion points, spans SSOT
   release_sync.py) and the single verbatim row 3426 (`_git_run`) is
   SOURCE-FALSIFIED — upstream's update-flow redesign DELETED _git_run from
   tip update_merge.py and rewrote the three D34 bodies (+517-line drift vs
   merge-base; post-cutoff supervisor/update_candidate.py exists at tip,
   absent from oracle AND merge-base). A one-symbol update_merge_plan.py would
   falsely anchor the F2 re-split (the D04 tool_result.py class). The oracle's
   +84 release_sync.py D34 delta (span-descriptor SSOT) defers with it; pins
   tests/test_update_merge_owner_facade.py / test_update_carriers.py /
   test_carrier_rebase_helper.py NOT carried.
5. tools/git.py split rows 374-415 executed for 41 of 42 rows from tip bytes
   (five leaves, proof green per leaf, exit 0). Drift-probe against tip bytes:
   git_plumbing 10/10 byte-true; git_evolution 3/5; git_repo_edit 2/4;
   git_vcs_ops 7/10; git_review_cycle 7/12. Falsified spans, two classes:
   (a) PURE UPSTREAM DRIFT (oracle==merge-base, tip moved):
   _finalize_blocked_review, _review_cycle_infra_failure,
   _check_evolution_commit_stage, _record_evolution_commit_receipt,
   _repo_write, _str_replace_editor, _ff_pull (+ the drifted halves of
   _run_reviewed_stage_cycle/_run_non_committing_review_cycle) — tip bytes
   emitted; (b) PURE V7 DELTA (tip==merge-base, reference typed by the
   git-control cutover a5e1cea3, oracle-only commit: _publish_git_error /
   _publish_review_blocked plumbing and the typed returns in _git_status,
   _git_diff, _stage_candidate_for_review and both stage cycles) — NOT
   replayed, rides with the F2 typed-result organ (same class as D04 entry 2
   / D05 entry 3). The reference-only plumbing symbols _publish_git_error and
   _publish_review_blocked were NOT created.
6. Row 392 (`_refuse_capped_attempt` -> git_review_cycle) — SOURCE-FALSIFIED:
   upstream 386e9417 ("Max Review Cycles") DELETED the symbol and re-derived
   the cap as the paid-cycle gate family (_free_cycle_gate,
   _install_paid_dispatch_stamp, _advisory_and_tests_gate,
   _repair_managed_merge_head, _finalize_pending_review,
   _review_custody_pending, _subject_binding_mismatch_outcome,
   _reconcile_and_clear_review_roster, _tests_preflight_block_message,
   _managed_candidate_needs_proof, _managed_committing_phase_error,
   _run_git_network_cmd — all unrowed post-cutoff facade symbols). The family
   STAYS in the facade (F5 unrowed census); moved spans read it through the
   call-time handle.
7. STRUCTURAL DIVERGENCE from the reference, disclosed: the reference's
   tools/git leaves bind cross-leaf/parent helpers with plain import-time
   from-imports; this tree's leaves declare EVERY parent-scope name their
   spans read and route it through the call-time `_git()` handle (the
   D18/D33/D35 mechanism, sets pinned in tests/test_module_handle_extraction.py).
   Reason, twice re-proven by red runs during the lane: the tip test surface
   monkeypatches those names on the PARENT facade
   (test_git_review_bypass_gate `_run_parallel_review`,
   test_update_status_cache `ensure_official_update_remote`), and an
   import-bound leaf copy makes every such patch silently dead — the
   monolith's module-global patchability is part of the moved behaviour. The
   only import-bound exceptions are the f-string reads the gate cannot
   rewrite (_sanitize_git_error in three leaves; format_protected_paths and
   utc_now_iso in one each), named in each leaf docstring; zero test patch
   surface exists for them today. The oracle's leaf-retarget test adaptations
   (rows 770/775 monkeypatch targets on git_review_cycle, test_commit_gate /
   test_vcs_target_binding / test_runtime_mode_registry_gating leaf imports)
   are therefore NOT mirrored — tip facade targets stay correct on this tree.
8. Test-split rows 3150-3191 (tests/test_git_ops_recovery.py -> 3 siblings +
   tests/_git_ops_recovery_shared.py) executed: 40/42 moved spans
   byte-identical to the reference siblings;
   test_official_fetch_timeout_kills_the_process_tree re-emitted from tip
   bytes (upstream communicate(input=...) drift, same train as the
   _run_git_process_bounded batch-stdin hunk); row 3179
   (test_dependency_sync_is_panic_tracked_and_killed_on_timeout) carried WITH
   the reference's hermetic root binding per its own row note (the tmp_path
   DRIVE_ROOT monkeypatch that keeps the mocked pip timeout from appending to
   the live supervisor log). Lossless: 48 == 48 test functions.
9. Test-split rows 765-783 (tests/test_git_review_pipeline.py -> 4 siblings +
   tests/_git_review_pipeline_shared.py) executed: 15/19 moved spans
   byte-identical; re-emitted from tip: _get_registry_module (reference
   imports the hot-deferred registry_core leaf — reverse-mapped to the tip
   registry spelling), TestAdvisorySkipTests (post-cutoff upstream autouse
   reviewer-slots fixture), TestBypassPathTestsRun / TestRouteSlotAwareBypassGate
   (reference monkeypatch retargets, entry 7). The reference shared module's
   unrowed `_get_git_review_cycle_module` accessor was NOT carried (nothing
   on this tree reads it; facade targets stay live through the handle).
   Lossless: 89 == 89 test callables. Path-keyed mirror: `_POPEN_ALLOWLIST`
   in tests/test_process_custody.py += supervisor/git_ops_reset.py
   (sync_runtime_dependencies moved with its waited+panic-tracked pip Popen —
   mirrors the reference's own allowlist row).
10. D13-remainder protective closure (coordinator LEDGER entry 4) landed by
   this lane per the D04 additive precedent: RELEASE_INVARIANT_PATHS
   (ouroboros/runtime_mode_policy.py, protected — strictly additive literal
   entries + comment) and scripts/run_external_review.py::
   _RELEASE_MACHINERY_PATHS += the four git_ops leaves; parity pinned by
   tests/test_git_ops_owner_facades.py (protection + hot-code-parity clauses).
   The reference's GIT_OPS_LEAF_MODULES/GIT_OPS_FAMILY_PATHS derived-set
   re-cut of the protected file is NOT replayed (a structural rewrite beyond
   additive closure — F5/owner decision); prompts-prose closure not touched
   (same as D04 entry 11). HOT_CODE_PATHS needs NO git rows (parent unlabeled
   at tip AND in the reference — parity, not blanket labelling).
11. Suite adaptation, disclosed: tests/test_module_handle_extraction.py
   `_module_bindings` gained Tuple-target unfolding (the git_ops facade binds
   its bounded-network aliases as `A, B = x, y` at :302-303) — same class as
   the D12 config-extraction Tuple fix. tests/test_git_extraction.py carried
   with adaptations named in its docstring (tool_module_inventory clauses
   dropped until that leaf lands; owner map minus the three reference-only /
   retired names; size bounds re-based on tip: facade <=1800 — it retains the
   paid-cycle gate family, the two deferred f-string spans and the catalog).
12. Zero-v7-delta re-proofs for the rest of the domain: repo_remotes.py,
   tools/git_rollback.py, version.py, update_recovery.py byte-identical
   tip==ref==merge-base; tools/ci.py, tools/commit_gate.py, tools/git_pr.py,
   tools/github.py, tools/review_revalidation.py, update_source.py pure
   upstream drift (ref==merge-base, zero v7 delta) — upstream bytes stand.
   update_candidate.py is a NEW post-cutoff upstream module (absent from
   reference and merge-base; no rows) — upstream bytes stand.
   update_merge_policy.py tri-divergence is fully owned by other lanes'
   landed HOT_CODE closures vs the reference's fuller loop/tool rows (their
   lanes) — no D10 action. size_ratchet_manifest.py regenerated with the
   official generator (git_ops.py and both split test giants left
   GIANT_PATHS; no new file enters any debt band).
13. Base-inherited, NOT this lane's defect:
   tests/test_smoke.py::test_size_ratchet_transition_against_explicit_base
   fails at the CLEAN base a56bb76a under its default HEAD-parent base
   (pre-proven in a detached worktree; BYTE_DEBT rows of four untouched files
   vs the wave-4 seam's parent); with the explicit base
   OURO_SIZE_RATCHET_BASE_REF=a56bb76a this lane's transition validates green
   (1 passed). Integration seam owns the default-base repair.
## From the D01 lane (base a56bb76a, 2026-08-30)
1. loop.py L-B split rows 3265-3425 (161 rows, nine leaves) executed against tip
   bytes: 150 spans landed with the transplant tool (drift-probe first per leaf;
   final --check per leaf: ast=tokens=bytes=True on every span,
   leaf_invariants=[], unread_declared=[], undeclared_top_level=[], exit 0).
   Drift-probe of the reference leaves against `git show HEAD:ouroboros/loop.py`:
   95/150 spans byte-identical, 55 BYTE-FALSIFIED as copy sources and re-emitted
   from tip bytes. Falsification class verified against the merge base
   (8028f1df): 52/55 pure upstream drift (oracle==merge-base, tip moved); the
   other 3 (_drain_incoming_messages, _check_budget_limits — oracle line-wraps
   around its own handle rewrites; _maybe_inject_finalization_nudges — oracle
   comment-prose rewording) carry NO code delta. Zero live v7 semantic deltas in
   the loop split; every span is tip truth.
2. ELEVEN loop rows SUPERSEDED-BY-UPSTREAM (upstream re-homed the symbol into
   its own leaf before this lane; tip ownership stands, no transplant):
   3273 (_last_assistant_text -> loop_transport.last_assistant_text),
   3312/3313 (_provider_failure_hint/_provider_recovery_hint ->
   loop_transport public pair; matches D02 lane entry 10), 3314
   (_task_deadline_epoch -> loop_transport.task_deadline_epoch), 3315/3316
   (_mark_owner_stop_control_drained/_owner_stop_window_elapsed ->
   supervisor/owner_stop.py, the 65b5d19f re-decomposition), 3340
   (_DELEGATE_ACTIVITY_TOOLS -> nanny_pacing.DELEGATE_ACTIVITY_TOOLS with a
   compat alias), 3341-3344 (the four _nanny_* helpers -> nanny_pacing.py
   public names; loop.py imports underscore aliases). The carried ledger
   renames those rows' destinations at F5.
3. Declared-set deltas against the reference LEAVES table, all tip truth,
   pinned in tests/test_module_handle_extraction.py: (a) same-leaf members tip
   tests monkeypatch on ouroboros.loop now read through _loop() even inside
   their own leaf (the reference instead re-pinned those tests to the leaf —
   its L3 wave; this tree keeps tip tests unchanged):
   _execute_task_acceptance_panel (acceptance_review);
   _compute_subagent_handoff, _resolve_delivery_control (delivery);
   _call_forced_model_once, _claimed_child_dispositions,
   _drain_forced_owner_directives (forced_finalization);
   _dispatch_round_model, _measure_round_main_fit, _run_main_reclaim
   (model_call); _skill_finalization_message (nudges);
   _mark_owner_stop_control_drained (round_limits; upstream re-homed the def
   into supervisor/owner_stop.py while tests still rebind it on the loop —
   proven by a red run of tests/test_owner_stop_s3.py before the declare).
   (b) round_limits gained _provider_unavailable_result and
   _append_or_merge_user_content as handle reads (tip drift); several oracle
   declared names dropped as unread on tip bytes (_last_assistant_text,
   _live_delivery_candidate in round_limits; _handle_forced_finalization in
   delivery) — the tool's unread_declared gate is the authority.
4. FACADE CONVENTION DIVERGENCE (disclosed, same class as D05 entry 4): the
   reference's L3 package trimmed the loop.py re-export surface
   (RETIRED_FROM_LOOP) after re-homing loop-private test imports to leaf
   owners. This tree keeps the FULL re-export surface (all 150 moved names,
   grouped per leaf at EOF) because the tip consumer set still addresses every
   moved name at ouroboros.loop; the L3 trimming is a consumer-rebind wave for
   F5, not part of the byte-preserving relocation.
   tests/test_loop_owner_facades.py is carried ADAPTED: the identity and
   hot-code-parity clauses survive over the full surface; the reference's
   RETIRED_FROM_LOOP absence clauses and the surviving-reason invariant are NOT
   carried (they pin the L3 state). HOT_CODE_PATHS closure mirrored for the
   nine loop leaves (D04/D08 precedent).
5. agent.py rows 3882-3897 -> agent_dispatch.py (D38 handle _agent, declared
   {write_task_result}): rows 3884-3897 executed from tip bytes (drift-probe:
   10/14 byte-identical, 4 pure upstream drift). Rows 3882-3883 SOURCE-
   FALSIFIED: upstream v6.105.0 moved dispatch_executor_note /
   executor_blocked_outcome into ouroboros/subagent_dispatch_notes.py; their
   live rows are 3935-3936 and the pair moved from THERE (shared-leaf
   convention: per-parent drift probes — the pair 0/2 byte-identical to the
   reference, both re-emitted from tip sdn bytes; final --check against the two
   parents concatenated, 16/16 green). subagent_dispatch_notes.py was touched
   ONLY by removing the pair spans + the re-export import (D01 part);
   its D07 rows 3937-3938 (SubagentExecutorResolution/SubagentLaneResolution
   bindings and the module-retirement question) stay untouched for the D07
   lane; the lost-reader imports keep the surface under noqa. agent_dispatch's
   tip spans additionally read _persist_early_origin_stub_impl (upstream
   re-homed the impl into agent_startup_checks.persist_early_origin_stub —
   tip-truth import, the D38 write_task_result handle read is intact).
6. agent_task_pipeline.py rows 3898-3909 -> post_task_synthesis.py: 11 rows
   executed from tip bytes (8/11 byte-identical, 3 pure upstream drift:
   _TASK_SUMMARY_PROMPT, _run_reflection, _run_task_summary). Row 3904
   (_summary_row_cost_fields) SUPERSEDED-BY-UPSTREAM: the symbol lives in
   ouroboros/synthesis_cost_text.py (public re-export list) — leaf imports it,
   ownership stands. The reference leaf has no handle; this tree's leaf is
   likewise projection-only (the auto-generated handle was stripped; zero
   declared). tests/test_lc2_owner_facades.py extended with the
   agent_dispatch/post_task_synthesis rows per the reference table, minus
   _summary_row_cost_fields (upstream home), with the sdn-facade note.
7. HOT-DEFERRED, typed-result/refusal class (tip bytes stand, nothing touched):
   loop_tool_execution.py rows 157-164/826-828 (classifier cutover; confirms
   D04 entry 6 from the D01 side — the shared monolith was not touched by
   either lane); loop_llm_call.py reference delta (+PROVIDER_POLICY_REFUSAL
   classification — imports llm_attempt symbols that do not exist at tip;
   rides with the D09 typed-refusal subfamily per D02 entry 4);
   _outcome_tool_errors.py reference delta (T1 status partitioning, D02
   family; re-prove trap per D15 entry 3); task_finalization.py reference
   delta (register-before-persist ordering — cancel/custody organ, 65b5d19f
   class, F2).
8. Test-split rows executed. tests/test_loop_misc.py (2037 lines, GIANT_PATHS)
   -> 4 siblings from tip bytes: test_loop_acceptance_gate.py (rows 3495-3505;
   6/11 spans byte-falsified by tip test drift, tip bytes moved;
   test_every_host_acceptance_writer_emits_a_canonical_status_and_typed_reason
   carried in the REFERENCE-ADAPTED form — the split spread the writers over
   loop.py + leaves and the reference's union-scan over loop_*.py is the
   identity continuation of the pin; the tip span byte-differs only by those
   two adaptation hunks), test_loop_image_attach.py (3506-3507),
   test_loop_skill_finalization.py (3508-3511), test_run_llm_loop.py
   (3512-3524; all byte-identical). UNROWED tip helper _seed_acceptance_root
   rode with its only readers into test_loop_acceptance_gate.py (F5 census).
   Rows 832-833 NOT executed: their destination suite pins the deferred typed
   cutover (D04 entry 9 class); the two tests stay in the remainder on tip
   bytes. Remainder 548 lines, left GIANT_PATHS; reader-less imports dropped.
   Lossless: 45 == 45 test names.
9. tests/test_agent_task_pipeline.py split rows: 22 of 34 rows
   SUPERSEDED-BY-UPSTREAM — upstream already extracted
   test_root_post_task_synthesis.py (3544-3556), test_post_task_reflection.py
   (3557-3560) and test_store_task_result.py (3561-3565) to the ledger's exact
   destinations. This lane executed the remaining two: test_task_summary.py
   (3535-3543) and test_collect_review_evidence.py (3566-3568), tip bytes,
   all byte-identical to the reference siblings. Lossless: 21 == 21.
10. Rowed import rebinds landed (identity continuations; the facade keeps both
   addresses live): 3915-3917/3932-3934 (test_v678_acceptance_state ->
   loop_acceptance / loop_acceptance_review), 3920-3921/3925-3928
   (test_loop_misc remainder -> nudges/round_limits/messages/acceptance), 3922
   (test_v6502_capability), 3923-3924 (test_budget_limits), 3929
   (test_nanny_finalization_nudge), 3930 (test_review_eligibility), 3931
   (test_transcript_seal), and the D02-deferred function-local retargets in
   tests/test_multimodal_chat.py (loop_messages; D02 entry 10 closure). Rows
   3918-3919 SUPERSEDED: tip already binds the provider hints from
   loop_transport (public names) — tip spelling stands.
11. Zero-v7-delta re-proofs for the rest of the domain (tip==ref==merge-base:
   _outcome_receipts.py, mutation_attribution.py; pure upstream drift,
   ref==merge-base: agent_startup_checks.py, deadline_utils.py, outcomes.py,
   owner_mailbox.py, post_task_checkpoint.py, synthesis_cost_text.py,
   task_pacing.py; NEW upstream modules, no rows: loop_transport.py,
   outcome_receipt_store.py) — upstream bytes stand. Ratchet: loop.py left
   GIANT_PATHS/BYTE_DEBT by extraction; agent_task_pipeline.py and
   loop_forced_finalization.py band entries recorded via the official
   regenerator's --band-rationale.
12. Post-battery closure per the D10 lane's lessons (superseding notes to
   entries 5-6): (a) MAXIMAL declared sets — a precise AST audit of every
   frozen leaf-preamble import against test patch surfaces (setattr on parent
   aliases + string-form patch targets) found three more dead facade patches
   and converted them to handle reads: agent_dispatch declared grew to
   {envelope_from_task, write_task_result} (test_available_subagents_runtime
   patches envelope_from_task on ouroboros.agent), and post_task_synthesis is
   NO LONGER projection-only — it carries the _atp() handle with declared
   {_is_root_post_task, load_task_result} (test_presence_post_task /
   test_agent_task_pipeline patch them on the pipeline), diverging from the
   reference's handle-less leaf, which froze _is_root_post_task by import.
   Zero f-string reads of rebindable globals were hit in any D01 emit (the
   tool's f-string gate never fired — no f-string HOT-DEFERRED spans in this
   lane). (b) tests/test_v7next_transplant.py loop probes re-pinned to the
   pre-split monolith bytes of the lane base (git show a56bb76a:ouroboros/
   loop.py, the D10 recipe) with a self-contained fallback that inverse-
   normalizes the landed loop_messages leaf — the suite is green either way.

## From the F2.1 D07-quiet lane (base 50377313, 2026-08-31)

1. Scope executed (the QUIET D07 part, F1 conveyor): 7 module leaves landed
   from tip bytes with the transplant tool (ast=tokens=byte-roundtrip=True on
   every span, exit 0, leaf_invariants=[], undeclared_top_level=[]):
   delegate_custody_reconcile (rows 3458-3466; D36 handle `_custody()`),
   delegate_payload_patch (3477-3483; `_di()`), subagent_integration_delegated
   (3484-3494; `_si()`), subagent_route_health (3939-3942; projection-only, no
   handle), and the D07 half of the SHARED tools/control.py —
   control_subagent_spec (2537-2542), control_scheduling (2543-2556),
   control_task_results (2569-2579). Facades = tip parent − moved spans +
   grouped EOF re-export block (noqa discipline for historical imports);
   facade audit green (every kept def/assign span byte-identical to `git show
   HEAD:<monolith>`, every moved name re-exported). Both 1600-hard-cap giants
   this lane was allowed to touch shrank: delegate_custody.py 1600→1305,
   control.py 2110→492 (control.py and the transport test giant LEAVE
   GIANT_PATHS/BYTE_DEBT); delegate_integration.py 1540→868,
   subagent_integration.py 1599→1027, subagents.py 1593→1370.
2. Drift-probe results (reference leaf `--check` against tip bytes, first
   step per leaf): delegate_custody_reconcile 2/9 spans byte-true (7
   re-emitted from tip); delegate_payload_patch 6/7 (integrate_payload_patch
   drifted); subagent_integration_delegated 10/11 (_integrate_delegated_patch
   drifted); subagent_route_health 2/4 (route_health, _exhausted_window
   drifted); control_subagent_spec 4/6 (schedule_subagent_properties,
   _validated_schedule_fields drifted); control_scheduling 9/14 rowed spans
   byte-true (5 drifted); control_task_results 7/11 (4 drifted). Every
   "verbatim" ledger claim was re-proven by bytes; no oracle semantics were
   replayed over tip drift (custody semantics: upstream is a strict superset
   — only the split FORM was taken from the reference).
3. Declared-set recalcs against the reference LEAVES table (tool
   unread/unresolved gates are the authority; new rows in
   tests/test_module_handle_extraction.py): `_custody()` dropped STARTED,
   START_REQUESTED, _CUSTODY, _iter_rows, event_log_path (tip drift stopped
   reading them) and gained retire_settled_registrations (upstream retirement
   decoupling 3226cc0c/8fe5a071); REVIEW_ATTRIBUTION_KEYS became a leaf
   preamble import (constant, never rebound in tests). `_di()`/`_si()` sets
   byte-matched the reference. control_scheduling declares exactly
   {load_settings} (tests rebind it on the facade); the reference control
   leaves were handle-free, the tip drift introduced that one read class.
4. Row 3467 (_capture_stranded_patch → delegate_custody_reconcile.py)
   SUPERSEDED-BY-UPSTREAM: 81194970 re-homed it as the public
   tools/delegate_integration.py::capture_stranded_patch and the body drifted
   further there; ownership stands with upstream, the row's destination needs
   an F5 rename (class: D01 lane entry 2).
5. Unrowed post-cutoff control.py neighbours ride with their only readers
   into control_scheduling: _context_task_depth (read only by
   _schedule_task), _materialize_child_attachment_manifest (same),
   maybe_emit_delegated_run_fanout (external reader tools/delegate.py:935
   does a call-time facade import — the facade re-export is load-bearing),
   HIDDEN_LEGACY_SCHEDULE_PARAMS. F2-matrix falsification: the matrix routed
   HIDDEN_LEGACY_SCHEDULE_PARAMS with the row-2541 reader
   (_validated_schedule_fields → control_subagent_spec); the tip readers are
   _schedule_task:865 and the module-level handler-attribute stamp
   `setattr(_schedule_task, "_hidden_legacy_params", …)`:1166 — probe beats
   matrix, the set moved with control_scheduling. The setattr Expr is the one
   facade statement RELOCATED below the re-export block (it reads two moved
   names at import time; consumer tools/tool_resolution.py:337 reads the
   attribute off the registered handler object, same object either way).
6. Transport test giant (6187 lines, 177 ledger rows): re-cut from tip bytes
   as the S7a theme split — 140 rowed tests moved to 10 destinations
   (cancellation_settlement 5, executor_axis 32, reconciliation 6 APPENDED to
   the file the D11 lane already created, result_delivery 12, run_accounting
   17, run_containment 11, run_custody 13, run_profile 15, wait_timeline 10,
   wait_window 19), 21 unrowed post-cutoff tests stayed in the remainder;
   lossless proven: 163 unique test names before == after (161 giant + 2
   pre-existing reconciliation), zero new duplicate names, every moved span
   byte-identical to the giant's bytes. Helper placement followed the rows
   (15 defs → tests/_delegated_transport_shared.py, private stubs → their
   sibling suites) with two documented lane placements: _plain_ctx went to
   the SHARED module instead of run_accounting (tip-only external consumer
   tests/test_delegation_account_pin.py imports it beside the autouse
   fixture; its import was re-pointed to the shared home), and the unrowed
   post-cutoff _transport_snapshot went to shared (the autouse fixture reads
   it). Four rows re-homing giant constants into runtime SSOTs
   (ACTING_SUBAGENT_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES →
   tool_capabilities; CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION → config;
   MODEL_SETTING_KEYS → provider_models) are SATISFIED BY UPSTREAM (the tip
   giant already imports them). 44 oracle-rowed test names are absent from
   the tip giant (upstream renamed/re-homed/retired them; the upstream
   test_delegated_run_isolation.py / test_delegated_skill_payload.py themes
   were built upstream as its own files) — no rows executed for absent
   names, F5 census item; tip bytes stand.
7. Dead-patch class (D08 lane entry 9 recipe; oracle adaptations mirrored
   into THIS tree's file names): tests/test_task_status_flow.py — 3
   wait-grace sites re-pointed to control_task_results (mirror of oracle
   test_task_status_wait_tools.py) and the queue-fallback test's
   write_task_result patch alias re-pointed to control_scheduling (mirror of
   oracle test_task_status_scheduling.py); tests/test_subagents_phase3.py —
   prepare_task_drive patch alias → control_scheduling (mirror of oracle:94);
   tests/test_external_workspace_access.py — system/active_repo_dir_for
   patch alias → control_scheduling (mirror of oracle:86);
   tests/test_cache_optimization.py — two wait aliases → control_task_results
   (mirror of oracle:435/480). A sweep of every other moved/frozen name found
   no further facade patches whose exercised path reads the leaf scope (the
   join_ledger _emit_control_event patches stay live: that path re-imports
   through the facade at call time; subagents.route_health and the custody
   sweep patches stay live: their callers stayed in the facades / read
   through `_custody()`).
8. HOT-DEFERRED with evidence (owner forks — nothing emitted):
   - Ф-2 (delegate_terminal name collision): rows 3468-3476 NOT emitted;
     tools/delegate.py stays exactly at the 1600 hard cap (at, not above —
     ratchet green). Probe evidence recorded: 7/9 reference spans byte-true,
     _terminal_payload + _delivered_terminal_payload upstream-drifted; the
     reference facade-identity rows for this family are also held back.
   - Ф-1 (subagent_worktrees.py strict-registry, rows 1083-1092 + the
     280-line pin suite): in-place semantic delta, not transplanted without
     owner sanction; tip==merge-base for the module, so the delta stays
     cleanly appliable.
   - Ф-3 (subagent_dispatch_notes retirement): rows 3937-3938 verified
     SATISFIED as identity on tip (sdn:17 imports the pair from
     ouroboros.subagents under the D01-lane noqa marker); the 71-line facade
     stays; retirement is an F5 consumer-rebind item (agent.py at its size
     ceiling + 3 test files + 2 unrowed helpers).
   - Six D02 rows (2548 _build_acting_constraint, 2549
     _select_subagent_constraint, 2556 _schedule_task, 2571 _get_task_result,
     2574 _wait_for_task, 2579 _wait_for_tasks) were cut in TIP form —
     ouroboros/tools/tool_result.py does not exist on tip; the D02 delta
     returns as a package with the typed-result organ (the plan's mandatory
     "D02 loop" return).
9. Hot-code label parity: the three control leaves joined HOT_CODE_PATHS
   beside the D08 trio (control.py stays labeled); the delegate/subagents
   families are unlabeled and their leaves keep parity — pinned in the
   adapted tests/test_delegate_owner_facades.py (reference file minus the
   deferred delegate_terminal group and the superseded stranded-patch row).
10. Out-of-scope defect FOUND (not fixed here, D06/review-organ material,
    D15 class): tests/test_review_cycles_dispatch.py and
    tests/test_review_cycles_skill_dispatch.py carry 10 AST-identical
    duplicate test functions (pre-existing at this lane's base).
11. For the integration seam: scripts/v7next_domains.toml rows for the seven
    new runtime leaves (D07) and the ten new/regrown test siblings follow the
    established seam convention (lanes do not edit the map); quotient report
    regeneration likewise.

## From the F2 addendum (coordinator, base 3c425206, 2026-08-31)
1. MIGRATION row 2016 (_handle_schedule_task -> events_schedule_task.py) —
   EXECUTED: the D08 deferral was unblocked by the D11 same-qualname
   relocation rule. Proof: span emit via the tool (ast=tokens=bytes=True,
   one handle read `_parent_delegation_budget`); whole-leaf verify with
   leaf_owned = the leaf's prior residents (the emit-time top-level gate is
   structurally blind to append-into-existing-leaf — assembly ran with that
   gate bypassed and the hardened verify as the actual authority; a
   `--leaf-owned` CLI flag is an F5 tool candidate). FUNCTION_DEBT key
   relocated with the function; events.py 1947->1406 entered the 1001-1500
   band by extraction with rationale. D08's work-order pins flipped as
   designed (dispatch owners + facade census).
2. Stale docstring of tests/test_delegated_reconciliation.py refreshed
   (F2.1 conformance item 7): the file owns the full reconciliation theme.
3. Addendum round 2 (battery findings): (a) tests patching
   `events._find_duplicate_task` retargeted to the leaf module — the ORACLE's
   own adaptation shape (its tests patch `schedule_module`), 25 sites across
   three files; the declared-through-handle alternative is structurally
   refused by the tool for leaf-resident names (ambiguous ownership, by
   design). (b) `_build_scheduled_task_payload` restored as a noqa facade
   import — tests import it from supervisor.events.
4. SUPERSEDING correction to entry 1 of this section (audit 31.08 07:27, F5):
   the FINAL landed leaf carries TWO handle reads (_parent_delegation_budget
   AND get_max_subagent_depth - the latter added when the patch-surface scan
   found tests monkeypatching it on the facade), and the facade size after
   the final import restores is 1392 lines, not 1406/1389 as the earlier
   prose said. The proof chain (span emit + hardened verify with leaf_owned)
   was re-run at each state; entry 1's figures describe an intermediate
   state and are superseded by these.
5. Audit 31.08 F1 second name: tests patching `_resolve_subagent_constraint`
   on the facade (one negative sentinel in test_nested_rights_depth) were
   retargeted to the reading leaf module - same oracle retarget-to-owner
   shape as the _find_duplicate_task sites; the sentinel's teeth are
   restored (the leaf's import-bound name is the one the handler reads).

## From the F2 D07-finisher lane (base 2878560e, 2026-08-31)

1. Scope executed (the three D07 owner forks, decided 31.08 batch 5: 5.9A,
   5.10A, 5.11A): the deferred terminal leaf of tools/delegate.py, the
   subagent_worktrees.py strict-registry delta with its pin suite, and NO
   sdn retirement (the facade stays).
2. F5-RENAME record (owner fork F-2=A, ledger rows 3468-3476): the reference
   leaf destination `ouroboros/tools/delegate_terminal.py` is renamed at
   landing to `ouroboros/tools/delegate_terminal_evidence.py`. Rationale:
   upstream already owns `ouroboros/delegate_terminal.py` ("terminal
   reconciliation boundary", 189 lines) and the ledger name would put two
   different delegate_terminal modules in neighbouring packages — a
   permanent grep/reading trap. Same class as the D01/D03 F5 destination
   renames. Rows 3468-3476 read onto the renamed file unchanged otherwise.
3. Terminal leaf landed from tip bytes (rows 3468-3476, D36 handle
   `_delegate()`): drift-probe first (reference leaf `--check` against
   `git show HEAD:ouroboros/tools/delegate.py`): 7/9 spans byte-true,
   _terminal_payload and _delivered_terminal_payload upstream-drifted —
   matching the quiet lane's held-back probe evidence — so the leaf was
   EMITTED from tip bytes, no oracle semantics replayed. Final proof:
   ast=tokens=byte-roundtrip=True on all 9 symbols, leaf_invariants=[],
   undeclared_top_level=[], unread_declared=[], exit 0 (re-run after the
   manual TYPE_CHECKING preamble addition, the D07-quiet
   reconcile-leaf precedent).
4. Declared-set recalc, MAXIMAL form (D10 tools/git precedent, finisher
   work-order): the reference cut this leaf with plain preamble imports and
   declared only {_emit}; the landed leaf declares EVERY parent-scope name
   the moved spans read at call time — 12 names: _Breach,
   _PAYLOAD_ENVELOPE_HEADROOM, _emit, _home_isolation_breach,
   _preview_payload, _resolve_full_primary_output, _stage_full_output,
   _widened_access, add_terminal_source_verification, custody,
   home_nested_under_operator_home, tool_result_limit — so patches on the
   historical `ouroboros.tools.delegate` surface keep their teeth. Only
   stdlib (json) and typing stay preamble imports; annotation-only names
   (_Breach for its `-> Optional[_Breach]` use, _RunCustody, ToolContext,
   DelegatedRunShape) ride an `if TYPE_CHECKING:` block, inert under future
   annotations. New LEAVES row pinned in
   tests/test_module_handle_extraction.py.
5. Facade: tools/delegate.py = tip parent - the 9 moved spans (lines
   225-576 of the HEAD file) + the grouped EOF re-export block + noqa
   discipline: exactly four `# noqa: F401` markers on the import lines of
   parent members now read only through `_delegate()` at call time
   (_home_isolation_breach, _widened_access, home_nested_under_operator_home,
   add_terminal_source_verification — the bindings are load-bearing for the
   leaf and must survive ruff F). Every kept def/assign span proven
   byte-identical to `git show HEAD:ouroboros/tools/delegate.py` (the diff
   of the kept region is exactly those four marker lines); re-exports
   proven same-object by import smoke. tools/delegate.py 1600 -> 1263: the
   LAST 1600-hard-cap giant of the D07 organ leaves the cap and enters the
   1001-1500 band with a rationale. The reference facade-identity rows for this family (held
   back by the quiet lane) landed in tests/test_delegate_owner_facades.py
   under the renamed leaf.
6. Ф-1 strict-registry delta (rows 1083-1092, owner sanction 5.10A —
   SANCTIONED SEMANTIC DELTA in an otherwise byte-preserving lane):
   drift-probe first — tip blob of ouroboros/subagent_worktrees.py ==
   merge-base 8028f1df blob (fd2db424, upstream never touched the module),
   so the reference diff (+104/-22) applied clean; the landed module is
   byte-identical to the reference module (blob ee694e4d on both sides).
   Semantics: absent registry stays an ordinary empty registry; malformed
   registry raises typed SubagentWorktreeRegistryCorrupt for every author/
   destructor (provision_worktree, provision_execution_snapshot,
   provision_payload_snapshot, find_execution_snapshot,
   remove_execution_snapshot, prune_execution_snapshots, remove_worktree,
   prune_orphans) instead of silently collapsing to empty; bytes are kept;
   one durable subagent_worktree_registry_corrupt event; inspection reads
   stay soft; registration moves INSIDE the cleanup scope on all three
   provisioning branches. Pin suite
   tests/test_subagent_worktree_registry_s6.py copied verbatim from the
   oracle (281 lines, 11 tests, red without the delta per D09 entry 10):
   imports only stdlib + the module itself, zero v7-only names to reverse-
   map; its docstring's sibling reference
   (test_delegated_skill_payload.py::test_registry_save_failure_leaves_no_orphan_snapshot_dir)
   exists on tip; the oracle registered it in no conftest path-keyed table.
   The one pre-existing tip test touching the registry
   (tests/test_acting_subagents.py:1298) uses the soft read, whose
   signature and behavior are unchanged.
7. Ф-3 (sdn): no action, per owner 5.11A — the quiet lane's entry 8 stands
   (rows 3937-3938 satisfied as identity; retirement stays an F5
   consumer-rebind item).
8. Ratchet (official regenerator): ouroboros/tools/delegate.py enters the
   band by extraction (1600->1263, rationale recorded);
   ouroboros/subagent_worktrees.py enters the band by the sanctioned delta
   (1000->1082, rationale recorded). domains.toml untouched (coordinator
   seam owns the map).
## From the F2.4 update-engine lane (base 2878560e, 2026-08-31)
D34 return + 1A re-split, per owner answers Ф-1=A / Ф-2=A / Ф-3=A
(= plan rows 5.12-5.14A). Every re-derived body below is justified as
reference-fact ↔ tip-fact ↔ result.
1. Span-SSOT re-cut (ouroboros/tools/release_sync.py, merged ATOP the tip
   file, not a replacement). Reference: 8 descriptors (v7_wip
   release_sync.py:65-148). Tip inventory is WIDER: sync_release_metadata
   writes the two public install pages (tip :423-434) and the README
   direct-download reference block (:100-113); version_carrier_desyncs /
   update_candidate.py:697-698 check them. Result: 25 descriptors = the 8
   reference spans + readme_download_refs (the contiguous
   `[download-<id>]:` block) + 8 anchors per install page, derived from
   RELEASE_ASSET_TEMPLATES (a new installer automatically gets a span);
   macos-arm64 appears twice per page and is disambiguated by the
   quick-start step's literal "Click " prefix (lookaround pair) — a page
   restructure degrades to malformed/duplicate-anchor, never a guess.
   Latent-trap fix proven by span inspection: proof ids carry `x86_64`, so
   a `[a-z0-9-]` class matched the tip block ONCE but covered only its
   first 3 lines (wrong-coverage, silent partial substitution) — the class
   is `[a-z0-9_-]`, and the live-tree pin asserts full-block coverage
   indirectly through exactly-once anchoring of every descriptor.
2. supervisor/update_carriers.py returned WHOLE (no upstream analog); two
   bodies re-derived against the redesign train's bounded-plumbing rule
   (4795a810/c404c056 class): _run_git and the merge-file runner now start
   the child in its own process group and kill the WHOLE TREE on a 300s
   timeout (constant mirrors update_candidate._GIT_RUN_TIMEOUT_SEC) —
   insertion point 3 runs while the update lock is held. Byte-exact capture
   (text=False semantics) preserved from the reference. Deliberately NOT
   routed through git_ops._run_git_process_bounded: that helper imports
   ouroboros.tools.shell at call time (tool-registry package init), which
   would break the standalone operator rebase helper; the
   _active_subprocesses shutdown-tracking nicety is therefore not carried
   (short-lived waited children — disclosed residual). Docstring
   re-derived: insertion host is the re-cut update_merge_plan.py; the
   resolver never runs `git merge` (explicit index stages + `git
   merge-file`), so it is rerere-neutral by construction, in line with the
   train's _MERGE_NEUTRAL_FLAGS discipline; M0 note per Ф-2=A.
3. Three insertion points re-derived against the REWRITTEN tip bodies
   (reference bodies were pre-redesign; matrix rows MIGRATION:3427-3429):
   (a) point 1 (row 3428): reference update_merge_plan.py:334-344 ↔ tip
   plan_managed_update_merge (stash-first; snapshot via
   worktree_snapshot_tree instead of the temp-index) → resolution after
   the merge/inventory consistency check, BEFORE classify_conflicts; the
   single body serves both the preview plan and the authoritative
   build=True replan (control.py replans on the clean tree through the
   same function). `carrier_resolved_paths` restored to the ff-clean,
   base-conflict and main returns (reference shape).
   (b) point 2 (row 3427): reference :88-97 ↔ tip _build_clean_merge_commit
   (fast_forwardable early return, Q8 projection before write-tree) →
   resolution inside the rc_bm==1 branch BEFORE write-tree; the tip's
   `if base_conflicts: return` inverted to the reference's
   no-inventory-error + resolve + `if remaining: return` shape; the Q8
   projection now runs AFTER span resolution, so its postcondition also
   verifies the just-resolved carriers.
   (c) point 3 (row 3429): reference :454-469 ↔ tip materializer
   (rerere-off flags, mandatory Q8 projection, CAS re-parent, M0 pin) →
   resolution after MERGE_HEAD validation and BEFORE the projection and
   the M0 pin (Ф-2=A: span policy is part of the mechanical baseline;
   reviewers diff an M0 already free of carrier markers); the tip 3-tuple
   return (ok, message, m0_tree) preserved.
   Handle idiom: the reference `_um()` handle is retained ONLY for
   managed_update_constitution_present (monkeypatched on the parent facade
   — test_update_merge_assisted.py:973); update_candidate members are read
   through the `_uc` module object (test_update_hardening.py:99/125
   patches update_candidate.worktree_snapshot_tree) — the D10-lane entry-7
   patch-surface rule. The row-3426 verbatim `_git_run` relocation stays
   SUPERSEDED (upstream re-homed it to update_candidate; the leaf reads
   `_uc._git_run`).
4. Boot-recovery backfill window NOT extended (upstream recovery semantics
   = floor): _recover_assisted_on_boot's M0 backfill re-runs only the Q8
   projection; a carrier still conflicted through that crash window
   degrades to the assisted lane (fail-safe, never fail-wrong). Disclosed
   in the wiring pin's docstring; keeps the "3 resolver calls in the leaf,
   0 in the parent" invariant intact.
5. 1A re-split executed per Ф-3=A from the two-module tip form:
   update_merge.py 1593 → 1193 (tx/lock/rollback/boot-recovery facade,
   re-exports both leaves), new supervisor/update_merge_plan.py (490 =
   three tip bodies + the documented deltas). The reference leaf is the
   THEME (same three owners), not bytes. Ratchet: update_merge.py entered
   the 1001-1500 band by extraction with a rationale via the official
   generator; `-m size_ratchet` = 5 passed.
6. update_merge_policy.py coordination (matrix row "согласовать"):
   carrier_guidance's hand-list VERSION_CARRIER_PATHS (6 paths, already
   narrower than the tip's own carrier inventory) replaced by a call-time
   read of the span SSOT (CARRIER_SPAN_PATHS); prose re-derived — spans
   resolved mechanically never reach the resolver's list (verified:
   control.py:820 refreshes tx.conflict_paths from live_unmerged_paths
   after materialization), so the guidance now describes exactly the
   DEGRADED remainder and what degradation means.
7. Protection closure (the G1/D10 additive-literal precedent, coordinator
   LEDGER entry 4 class): RELEASE_INVARIANT_PATHS +=
   supervisor/update_merge_plan.py, supervisor/update_carriers.py —
   the split moved planner/materializer bodies out of a release-invariant
   file and the resolver rewrites worktree files under the update lock;
   parity pinned in tests/test_update_merge_owner_facade.py. DISCLOSED
   upstream inventory gap, NOT repaired (Q4=A, upstream owns protected
   surfaces): supervisor/update_candidate.py carries bodies upstream's own
   redesign moved out of the same protected parent, yet is absent from
   RELEASE_INVARIANT_PATHS — owner/Ф3 material.
8. Tests: test_update_carriers.py ported with re-derivations (leaf import
   path unchanged; materializer test unpacks the tip 3-tuple and pins that
   M0 names the official VERSION blob; corpus README fixture extended with
   the FULL 7-id download-refs block — the Q8 postcondition checks every
   RELEASE_ASSET_TEMPLATES member once a README opts into the projection,
   and the new span must anchor; SSOT pin re-cut to 25; an explicit
   "conflicted carrier never routes to assisted" strategy pin added per
   the work order). test_carrier_rebase_helper.py + the operator helper
   returned (helper docstring's carrier list re-cut).
   test_update_merge_owner_facade.py re-derived: owners = update_merge_plan
   (3 bodies) + update_candidate (the redesign's own boundary, identity
   now pinned); hot-code and release-invariant parity clauses.
   _POPEN_ALLOWLIST (tests/test_process_custody.py) +=
   supervisor/update_carriers.py (path-keyed mirror, D10 git_ops_reset row
   class).
9. NAME COLLISION tests/test_update_merge_plan.py resolved as SUPERSEDED,
   not transplanted: the oracle file's 13 test functions are
   name-set-identical to the tip file and the tip bodies are the
   upstream-evolved forms of the same assertions (stash status tuple
   "ok"/sha, failed-update-<target12> forensics naming) — zero unique
   oracle content; a rename-transplant would mint 13 AST-near-duplicates
   (the D15 class the wave mandate bans). Tip bytes stand.
10. Upstream test re-derived (falsified-by-D34 fixture, the "test pinning
   the gap" class): test_update_merge_assisted.py::
   test_materialize_projects_version_to_target_and_pins_m0 used a clean
   1.5.0-vs-2.0.0 VERSION token conflict, which the D34 planner now
   resolves (plan turns clean — the scenario could no longer reach the
   materializer's projection). The local token becomes a malformed anchor
   ("not-a-version"), so the span resolver degrades honestly and the Q8
   projection clause the test pins stays reachable; docstring says why.
11. Ф3 joints named, untouched (report-only): the future N−1 shim surface
   (finalize_managed_update_on_boot / _recover_assisted_on_boot /
   _recover_replace_on_boot / _finalize_pending_boot_smoke /
   apply_managed_merge_update / rollback_managed_update) stays WHOLE in
   the parent — the re-split does not dissect ABI-7/F14 material; the RC
   auditor's evidence surface (record_managed_tests_evidence /
   managed_tests_evidence_covers) untouched in update_candidate;
   git_ops.py:1031-1032 (D13) untouched — protected wave; Ф-4 derived
   FAMILY_PATHS not executed (coordinator's tail item — the additive
   entries in item 7 keep that door open).
12. Pre-existing at base, NOT this lane's defects (dup-scan receipts):
   10 AST-identical test pairs across test_review_cycles_dispatch.py /
   test_review_cycles_skill_dispatch.py (already named by the Ф2-plan) and
   an in-file duplicate def test_ripgrep_download_script_verifies_checksum
   in tests/test_build_scripts.py (the later def shadows the earlier —
   D15-class latent, review-organ/F5 material).
## From the F2.3a review-mechanics lane (base 2878560e, 2026-08-31)
1. FALSIFIED row: `tests/test_review_substrate_v2.py::_render_prompt ->
   review_substrate` (repoint to the "canonical substrate owner"). Upstream
   moved `_render_prompt`/`_render_prompt_parts` into review_execution
   (substrate back-imports them as compat re-exports), so the row's target is
   stale. Executed as re-derive: the split's prompts suite
   (tests/test_review_substrate_prompts.py) imports `_render_prompt` from
   ouroboros.review_execution.
2. ROW CORRECTION: `ouroboros/tools/scope_review.py::_load_canonical_context_docs
   -> scope_review_pack.py` was NOT executed as written — the symbol stays a
   facade def. Its body reads `load_governance_doc` inside an f-string (the
   byte gate refuses f-string reads of rebindable globals) and tests rebind
   that name on the parent (test_review_convergence_rule.py:122 et al.), so a
   leaf copy would go dead-patch. Same class as the D10 lane's
   safe_restart/prepare_managed_update facade retention. The pack leaf reads
   it through the `_sr()` handle; 19/20 pack rows moved.
3. NEW-OWNER leaf (owner decision 5.3=B, one-cut): ouroboros/review_state_custody.py
   carries nine post-cutoff upstream symbols no MIGRATION row names —
   unrowed F5 candidates, recorded here as adoption rows:
   review_state.py::{_ACTIVE_REVIEW_OPERATION_STATES, _attempt_review_roster_rows,
   _review_roster_row_is_pending, _attempt_has_active_review_custody,
   checkpoint_pending_review_invocation, _attempt_history_evictable,
   _STRIPPED_DETAILS_LIMIT, _STRIPPED_MESSAGE_LIMIT, _strip_attempt_heavy_payload}
   -> review_state_custody.py::<same name> (adaptive-timeout/custody train;
   tool-proof ast=tokens=bytes on every span). The four authority-shape
   deserialization symbols of the same train (_malformed_roster_row,
   _ATTEMPT_AUTHORITY_STRING_FIELDS, _ATTEMPT_AUTHORITY_BOOL_FIELDS,
   _validate_attempt_authority_shape) stay with the parent STORE by design.
4. SUPERSEDED rows honored (upstream home wins, Q4=A; leaves/tests do not
   replay them): review_evidence.py::{_ACCEPT_DELTA_CHILD_CAP,
   _accept_capability_deltas} -> delegate_evidence (facade reads
   acceptance_capability_deltas back at call time);
   tools/review.py::_parse_model_response -> tools/review_response.py
   (facade re-import is the single alias, pinned by
   test_review_owner_facades.py). RETIRED rows honored:
   tools/review.py::{DEFAULT_REVIEW_MODEL_TIMEOUT_SEC, _review_model_timeout_sec}
   died with the adaptive-timeout contract and are not restored.
5. Import-bound exceptions (f-string/import-time gate; named in each leaf
   docstring): review_multi_model: SLOT_ID_PREFIX (default argument);
   review_file_pack: format_prompt_code_block (f-string; unpatched in tests);
   scope_review_pack: format_review_history_entry,
   _HISTORY_VERIFICATION_ONLY_RULE, _ANTI_THRASHING_RULE_VERDICT,
   _CONVERGENCE_RULE_TEXT (f-strings; owner review_prompt_text);
   review_evidence_sections: DEFAULT_TOOL_RESULT_LIMIT (default argument);
   review_state_model: _STATE_SCHEMA_VERSION, _DEFAULT_ADVISORY_TOOL_NAME,
   _REVIEW_ATTEMPT_TTL_SEC, _REVIEW_ATTEMPT_GRACE_SEC (class-level defaults),
   _stable_digest (f-strings) — owner review_state_records;
   review_records/review_verdict: ReviewRouteKind / OUTCOME_TIER_* (class-level
   and module-level constants). None of these names is monkeypatched on the
   parents anywhere in tests/ (verified by grep before binding).
6. TEST DELETION disclosure (owner decision 5.2=A): ten AST-identical test
   functions plus seven byte-identical orphan helpers were deleted from
   tests/test_review_cycles_dispatch.py; the owner of those tests is
   tests/test_review_cycles_skill_dispatch.py (D14 family — they exercise
   skill_review_* modules only). Verified byte-level: ast.dump-identical in
   both files before deletion, zero shared-but-different defs. −510 lines of
   double-executed runtime; the dispatch file remains the D06 commit-gate
   paid-accounting suite.
7. Session-route split: three reference-authored tests absent from the tip
   giant were NOT replayed (skipped, F5 material):
   test_unhealthy_route_refuses_typed_never_falls_back,
   test_route_status_refusal_carries_its_typed_code,
   test_retry_of_a_pinned_session_health_checks_the_stored_account. Thirteen
   tip-only (post-cutoff) tests were placed with the sibling that owns their
   helpers (2 -> scope_wiring, 1 -> poller, 2 -> delivery, 8 stay in the
   remainder with FakeGateway/_run_session_directly imported from the shared
   module). Lossless: 102 == 102 test names across the five files, zero
   duplicate names.
8. Substrate split lossless: 71 == 71 test names across six files — the
   reference's five plus tests/test_review_substrate_custody.py, a NEW
   sibling created by this lane for the eighteen post-cutoff upstream tests
   (the adaptive-timeout/custody train theme, 907 lines of tip bytes); the
   remainder would otherwise have stayed a >1600 giant. Both re-derived
   extraction suites drop the reference's tool_module_inventory clauses
   (that module exists only on the reference).
9. Path-keyed mirrors (D10 additive-closure precedent):
   review_context_atlas._REVIEW_STACK_PATHS += the eight state/evidence/
   helpers/scope leaves (oracle placements) + the new custody leaf;
   scripts/run_external_review.py::_REVIEW_SUBSTRATE_PATHS += all eleven
   leaves beside their parents. The hand-list's structural rot (28/48 D06
   modules absent before this lane) is the Р1/D31 fork — Ф2.3b territory,
   not repaired here beyond the additive closure for our own leaves.
10. review_records is a projection-only leaf (zero handle reads, zero
   declared) and stays off the LEAVES table per the D07/D08 precedent; the
   other ten leaves carry tool-derived exact declared sets there.
## From the f22 lane (base 2878560e, 2026-08-31)
1. Drift-probes (recipe §5.3-Δ2 step 9) of every oracle leaf against this
   base's monolith bytes, before any emit. Byte-identical tip↔oracle:
   _task_done_review_projection, _PROVIDER_DEATH_NOTIFIED,
   _task_done_durable_fault, _handle_task_done, _handle_evolution_task_done,
   _close_campaign_after_owner_stop, _kept_service_pids, parse_iso_to_ts,
   all queue_timeouts symbols except _enforce_task_timeouts_locked,
   _evolution_assignment_error, _cancel_unauthorized_evolution,
   terminal_task_metadata, _emit_task_done_terminal, ensure_workers_healthy.
   Byte-FALSIFIED as copy-source (upstream drift, re-emitted from tip bytes):
   _authoritative_terminal_cost, _maybe_notify_provider_death,
   _finish_task_done_dispatch, _resolve_lifecycle_fault, _handle_cancel_task,
   persist_queue_snapshot, restore_pending_from_snapshot,
   _enforce_task_timeouts_locked, assign_tasks,
   _ensure_workers_healthy_locked. ALL families were emitted from tip bytes
   regardless (proof: ast=tokens=True per symbol, leaf_invariants=[]).
2. Q-a=A (owner, 2026-08-31): the sixteen settle-owner rows 998-1013
   (task_lifecycle -> supervisor/cancel_custody.py) are SUPERSEDED — the
   settle owner STAYS in task_lifecycle.py; the upstream custody cut
   (65b5d19f/bea08137) is the authoritative floor and cancel_custody.py is
   never created. tests/test_cancel_custody_extraction.py is NOT replayed
   (its identity/size clauses are form-dependent on the extraction; matrix
   §3.8). Row 1000 (_durable_settled_status) is doubly retired: upstream
   removed the symbol (fail-soft equivalent lives as
   cancel_intents.settled_status).
3. Q-b=A: rows 2041-2044 (queue.py -> supervisor/queue_evolution.py) are
   RESOLVED WITHOUT the reference leaf. Upstream itself moved
   _deliver_pending_owner_report and enqueue_evolution_task_if_needed into
   supervisor/evolution_lifecycle.py; get_evolution_status_snapshot and
   queue_deep_self_review_task stay on the queue facade by owner decision
   (do not fork the evolution-family ownership a second time).
4. Q-c=A: row 970 EXECUTED — _close_campaign_after_owner_stop moved to
   supervisor/queue_transitions.py (byte-identical span; the drift probe
   proved tip==oracle here), events.py re-exports it, and
   events_evolution_done reads it through the _events() handle (no bare
   local name survives the split). queue_transitions.py entered the
   1001-1500 band with a rationale and joined HOT_CODE_PATHS (parity: the
   span moved out of the hot events monolith).
5. Rows 971-979 (events_task_done + events_evolution_done), the cancel
   ingress row (file row 994 / D08-ledger row 992), rows 2017-2028
   (queue_snapshot + queue_timeouts) and rows 2061-2064/2077-2079
   (worker_health + worker_assignment) EXECUTED as reference-named leaves
   from tip bytes. Declared sets are MAXIMAL (wave-2 dead-patch lesson),
   larger than the oracle's: every parent global the spans read at call time
   routes through the handle, including same-leaf reads
   (_PROVIDER_DEATH_NOTIFIED — tests rebind it on the facade), the
   cross-family coop hooks (_checkpoint_coop_roots_on_root_done and
   _maybe_checkpoint_coop_on_tree_quiescence: the GR4-3 probes patch them on
   supervisor.events — a module-scope import here is the dead-patch class the
   first emit reproduced and the re-emit fixed), `time` (the
   enforce-harness in test_packaged_runtime_and_lifecycle rebinds
   events.time), `_bound_project_chat_id` (the terminal-frame delivery tests
   rebind it on supervisor.events — a MULTI-LINE setattr the first
   single-line patch-surface grep missed; the closing sweep is an ast.walk
   over every tests/*.py catching setattr in any form through module
   aliases) and `BUDGET_ROOT_FENCES` in queue_snapshot (tests rebind it on
   the queue facade while persist_queue_snapshot reads it at call time; the
   pre-split span read queue's own re-export binding). Facade imports that
   now serve ONLY leaf handle reads carry per-line noqa markers naming the
   leaf.
6. Delta-D08 RE-DERIVED on tip bytes (Q-d=A): mark_finalize_control_drained,
   mark_intent_scope, release_claim and settle_intent now read the projection
   strict (_load_intents(strict=True) + strict_existing_dict=True) and turn
   the typed ValueError into CancelIntentProjectionCorrupt via the
   _refuse_corrupt helper (oracle shape); the tip GR5-6 docstring that
   RATIONALIZED fail-open ("non-minting mutators find no row in {}") is
   deliberately rewritten — that was the semantic delta, not a drift.
   Upstream's own strict sites (request_cancel, claim_intent, active_intents)
   keep their tip bytes. Caller audit (every tip call site, what happens on
   raise): (a) task_lifecycle._settle_intent/_release_intent_claim wrappers —
   except Exception, log.debug: the intent stays OPEN/CLAIMED for the
   watchdog; (b) task_lifecycle cancel_task_by_id cascade postcondition —
   outer except, cascade intent stays open, watchdog re-runs the cascade;
   (c) task_lifecycle record-cascade-scope site — except Exception with
   log.warning + typed cascade_scope_record_failed forensic row (loud, second
   line of defense); (d) ouroboros/task_results.fail_tasks budget drain —
   both settle and release wrapped, log.debug, intent stays for the watchdog;
   its claim path already maps a raise to claim_refused and skips the task;
   (e) workers pending-drop lanes (_settle_cancelled_pending_row,
   _release_pending_claim, terminalization retry) — except Exception ->
   claim_unresolved -> the row is RETAINED in the terminalization-retry lane,
   nothing silently dropped; (f) owner_stop._mark_owner_stop_control_drained
   — outer except returns False: no drain stamp, the finalization episode
   stays bounded by the unstamped request anchor (a corrupt projection can
   not buy an unlimited final turn). No caller needed a code change; the pin
   is tests/test_cancel_intent_corruption_s6.py (C1/C2), re-keyed to the tip
   bool contract of release_claim (upstream fence-proof return; the oracle's
   `is None` clauses would pin a retired signature).
7. S7b split RE-DERIVED from tip bytes (rows 2152-2223): lossless — 107
   test functions / 112 expanded items before == after, zero duplicate
   names, all green. The oracle partition is honored row-by-row for every
   surviving name; tip-new (bea08137-class) objects were placed by theme and
   these MINTED rows are: retry-race custody family
   (_patch_retry_input_handoff, _root_retry_task,
   test_retry_cancel_before_admission_publishes_no_successor,
   test_retry_admission_before_cancel_canonicalizes_and_stops_leaf,
   test_retry_leaf_cannot_escape_a_logical_root_cascade_at_final_boundary,
   test_cancel_suppressed_retry_task_done_waits_for_summary_obligation,
   test_timeout_precheck_yields_retry_leaf_to_logical_root_cascade,
   test_retry_boundary_refuses_missing_physical_leaf_authority,
   test_terminal_retry_leaf_wins_even_when_predecessor_lineage_is_corrupt,
   test_terminal_before_retry_boundary_creates_no_scheduled_ghost,
   test_same_id_timeout_retry_cancels_exactly,
   test_retry_leaf_completion_between_request_and_custody_wins,
   test_graceful_single_retry_targets_leaf_and_stop_now_hardens_same_intent,
   test_task_lifecycle_keeps_scheduled_admission_import_surface,
   test_task_lifecycle_keeps_capture_miss_calling_convention)
   -> tests/test_cancel_custody.py; dispatch-authority family
   (test_assignment_blocks_when_cancel_intent_projection_is_unreadable,
   test_assignment_retains_pending_when_claim_authority_raises,
   test_timeout_reaper_does_not_clone_over_unreadable_cancel_authority,
   test_snapshot_restore_blocks_when_cancel_intent_projection_is_unreadable,
   test_cancel_authority_hold_never_releases_a_terminal_row_to_dispatch,
   test_preserve_pending_shutdown_keeps_cancel_authority_hold_nonterminal,
   test_drop_cancelled_pending_retains_custody_until_task_done_is_published,
   test_drop_cancelled_pending_releases_a_failed_intent_claim,
   test_drop_cancelled_pending_does_not_assume_settled_when_settle_helper_missing,
   test_drop_cancelled_pending_defers_when_intent_vanishes_before_settle)
   -> tests/test_cancel_queue_integration.py; durable-gate additions
   (test_blank_status_task_done_over_a_running_row_is_a_durable_fault,
   test_blank_status_task_done_over_a_settled_row_is_admitted,
   test_copy_back_exception_never_synthesizes_a_completed_row)
   -> tests/test_cancel_task_done_validation.py; projection-primitive
   additions (retry-lineage mint family rows 82-238 of the monolith,
   test_claim_intent_refuses_an_existing_corrupt_projection,
   test_claim_intent_absent_projection_is_a_read_only_miss)
   -> residual tests/test_cancel_intents_phase_a.py; and
   _write_root_retry_pair joined tests/_cancel_intents_shared.py (read by
   both the mint suite and the custody retry suite — a tip extension of the
   shared set, rows 2152-2155 class). The monolith's section-banner comments
   are not carried (the same inter-span-comment loss the D14 lane recorded
   for the emitter). tests/test_cancel_cascade_v664.py's source-scan clause
   retargeted to the owner leaf (events_task_done) and its now-unused facade
   import dropped.
8. Durable pins landed with tip re-keys: tests/test_e2e_cancellation_scenarios.py
   + tests/fixtures_e2e_cancellation.py (E-suite; the driver extensions —
   typed cancel_task with cascade/stop_policy, hurry_task, _api_status —
   ported into devtools/benchmarks/common/server_runner.py, options-free
   cancel keeps the legacy empty-body wire shape for the existing benchmark
   callers); E8 is RETIRED and superseded by E13 (F6 disposition, owner
   Q9=A/Q10=A: a budget-drained queued task PAUSES — durable scheduled
   result with reason_code=budget_exhausted plus the typed
   budget_scope_paused event — it is not failed); C5/R1 were already on tip
   (D09 quiet edge); tests/test_cancel_protocol_inventory_s6.py (C7-C10)
   re-keyed by symbol to the tip owners (settle-owner cluster in
   task_lifecycle, miss lane in cancel_publication, admission in
   task_admission, the F2.2 leaves) with the upstream retry/depth terminal
   lanes ADDED to both the C7 manifest and the no-deliverable enumeration;
   C9's task_finalization docstring (row 1093) corrected to the VERIFIED tip
   call order (emit_task_results registers the owed row, then stores).
9. Mock-lane execution proof (post-commit verification, then amended in):
   the eight mock scenarios (E4-E7, E9-E12) ran GREEN against a real isolated
   server on this exact tree — after ONE harness adaptation of the class the
   suite's own docstring predicts: upstream delegation-by-construction makes
   subagent selection explicit (`subagent_configuration_unsaved` /
   `subagent_selection_required`), so isolated_settings() now pins a saved
   one-row Available-subagents roster (api_model on the lane's own slug) and
   the stub's spawn turn passes subagent_id="mock-scout". Scenario semantics
   untouched; the same class the deferred
   test_daemon_token_containment_s6.py note in the matrix recorded.
10. tests/test_v7next_transplant.py queue probes re-pinned to the PRE-SPLIT
   monolith bytes of this lane's base (git show 2878560e:supervisor/queue.py)
   with the landed-leaf inverse-normalization fallback — the D01/D10 probe
   recipe.
11. Path-keyed mirrors updated in the same commit: HOT_CODE_PATHS gained the
    four F2.2 leaves + queue_transitions; test_contracts' literal
    progress_meta scan gained events_runtime_controls + events_task_done;
    test_heartbeat_presentation's message-seam scan gained the two worker
    leaves; tests/test_events_extraction.py flipped from the pinned
    partial-split work order to the completed shape; the five satisfied
    [split_pending]/[split_pending_leaves] rows left scripts/v7next_domains.toml
    with the two owner-retired leaves recorded in a comment.

## From the F2.3b review-semantics lane (base dcf8dd4b, 2026-08-31)

1. F5 leaf-name mint (advisory split): the reference leaves
   `ouroboros/tools/review_advisory_prompt.py` / `review_advisory_run.py` are
   NOT the landed names. The drift probe (`--check` of both reference leaves
   against `git show dcf8dd4b:ouroboros/tools/claude_advisory_review.py`)
   byte-falsified the organ's semantics: prompt leaf 4/5 rows byte-true with
   `_build_advisory_prompt` falsified (governance_by_retrieval pointer form);
   run leaf 10/18 byte-true with `_run_claude_advisory`,
   `_run_advisory_delegated`, `_llm_extract_advisory_items`,
   `advisory_review_route`, `advisory_slot_enabled`-adjacent route/gate
   projections falsified (native episode + reviewer-slot SSOT replaced the
   Claude-SDK transport). Landed as `preflight_review_prompt.py` /
   `preflight_review_run.py` — the organ's public rename vocabulary (Q1) —
   cut from tip bytes, tool proof green on every symbol.
2. Advisory row dispositions against the 30 ledger rows (3852-3881):
   23 transplanted-from-tip (5 prompt + 18 run); 3 SUPERSEDED —
   `_release_metadata_preflight`, `_auto_sync_release_metadata_if_needed`,
   `_syntax_preflight_staged_py_files` live with upstream's
   `ouroboros/commit_admission.py` (Q3=A SSOT; the parent keeps the alias
   monkeypatch seams, pinned by
   test_review_owner_facades.test_the_deterministic_preflights_live_with_commit_admission);
   4 RETIRED with the SDK transport — `_changed_paths` (upstream's
   `review_helpers.parse_changed_paths_from_porcelain` class),
   `advisory_route_requires_api_key`, `_advisory_session_deltas`,
   `_advisory_sdk_budget` (no tip bodies exist; not replayed).
3. Scope budget probe corrections (mandate: re-verify the matrix's
   6-superseded/1-retired): the reference leaf probed 7/8 byte-true against
   tip; ONE row falsified — `_SCOPE_REVIEW_SLOT_TIMEOUT_SEC` (reference `900`,
   tip `None`: the adaptive-timeout contract retired the constant; tip byte
   kept). The matrix called `_SCOPE_BUDGET_TOKEN_LIMIT` retired (#383) — the
   probe shows the NAME alive on tip as a private alias of
   `review_helpers.REVIEW_PROMPT_TOKEN_BUDGET` (the reference-era standalone
   constant is what died); it moved with the other five owner aliases
   (`_SCOPE_MODEL_DEFAULT`, `_SCOPE_FAILCLOSED_WINDOW`,
   `_SCOPE_MODEL_CONTEXT_WINDOW`, `_shared_window_scaled_reserves`,
   `_calibrated_input_token_limit`) plus `_is_provider_oversize_error` into
   the budget leaf per the ledger's rebind disposition, parent re-exports —
   import-frozen on both sides exactly as before the split, so no
   patch-visibility change.
4. D31 port (owner decision 5.1=A): `_run_on_trusted_base` re-derived from
   the reference (scripts/run_external_review.py:539 @ 9f691656) onto the tip
   script — the contributor lane now ALWAYS executes the target base's own
   review machinery (self-re-run from a detached base worktree with pinned
   base/head SHAs). `_REVIEW_SUBSTRATE_PATHS` demoted to EVIDENCE ONLY
   (`review_substrate_changed` packet diagnostic), never a gate:
   `_contributor_result` is exit-code-only (reference form), and
   `finalize_contributor_outcome` dropped both the `snapshot` parameter and
   the `trusted_base_rerun_required` downgrade (reference form; its one
   script call site and two test call sites re-derived). The fail-closed
   `INCOMPLETE_MAINTAINER_TRUSTED_BASE_RERUN_REQUIRED` vocabulary survives on
   the ONE non-portable path — a target base whose tree carries no review
   wrapper (new guard; the reference would have misfiled python's exit 2
   there as "empty diff").
5. D31 pin suite ported from the reference (probe script, handoff/forwarding/
   in-place/dirty/e2e pins) with tip reverse-mapping: the seeded repo stubs
   `ouroboros/openrouter_attribution.py::OPENROUTER_APP_HEADERS` (the tip
   wrapper's module-level import) where the reference seeded
   `runtime_mode_policy::GIT_OPS_FAMILY_PATHS` (its wrapper's import). NEW pin
   beyond the reference: the always-runs-on-base parametrization includes
   `ouroboros/review_native_episode.py` — a review-machinery module ABSENT
   from the evidence hand-list — plus a fail-closed pin for the wrapperless
   base. Disclosed test replacement: the old gate clause
   (`test_contributor_outcome_fails_closed_on_receipt_or_trust_drift`'s
   substrate-downgrade half) asserted the hand-list AS a gate — exactly the
   semantics the owner retired — and is replaced by the reference's
   receipt-drift-only pin plus `test_contributor_result_is_decided_by_the_exit_code_alone`.
6. Path-keyed mirrors in the same commit: `_REVIEW_SUBSTRATE_PATHS` (evidence
   list) and `review_context_atlas._REVIEW_STACK_PATHS` gained the three new
   leaves beside their parents; domains.toml gained the three D06 leaf rows
   and cleared both satisfied [split_pending]/[split_pending_leaves] entries
   (review_execution's row left untouched — matrix A2 marks it superseded by
   upstream's review_verdict_extraction, an integrator decision, and this
   lane's mandate excludes review_execution).

## From the integration seam (coordinator, F2 close-out, 2026-08-31)
1. split_pending row `review_execution.py -> review_session_verdict.py`
   retired as SUPERSEDED-BY-UPSTREAM (matrix D06 verdict A2, confirmed by the
   F2.3a lane): upstream performed the same extraction itself as
   review_verdict_extraction.py; the reference leaf name never materializes.
   The F2.3b lane left this disposition to the integrator - recorded here.
2. SUPERSEDING note to the F2.3b lane's atlas claim: the two advisory leaves
   entered _REVIEW_STACK_PATHS with the F2 close-out conformance fix, not
   with the lane commit (the lane's ledger entry overstated); a membership
   pin now accompanies them.
3. Cross-test fragility class (found by loadscope redistribution after the
   close-out fixes): supervisor.queue globals (PENDING/RUNNING/...) are
   rebound by init_queue_refs across ~35 upstream test sites with no restore
   - an upstream-wide convention, not to be mass-rewritten. READER-SIDE RULE
   for campaign pins: never assume those globals are empty; REPLACE the dict
   for the test's scope (monkeypatch.setattr), never append into the live
   one. Applied to test_both_custody_surfaces_see_the_same_live_task_set.

## From the f30 lane (F3.0 opening train, base db944347, 2026-08-31)
1. ABI-6 re-location on tip (the roast-session scratchpad that minted the P1
   inventory did not survive; the surviving primary source is
   V7NEXT_SYNTHESIS_DRAFT.md, which names items without addresses):
   (a) `_call_llm_with_retry` alias re-located at ouroboros/loop.py:74 -
   ZERO code readers on db944347 (every monkeypatch targets the public
   name); removed. (e) `compute_cost_with_children` (task_status.py:1001) +
   `format_handoff_message` (:1054) - zero production callers; the canonical
   with-children rollup lives in agent_task_pipeline/post_task_synthesis
   with cost_projection.py as projection SSOT; removed with their private
   helper and tests. (zh) "CHECKLISTS:507" re-located: the line number
   drifted on both b9f7597f and db944347; the actual finding (archive,
   sol audit 30.08) is the env_allowlist checklist row claiming
   TELEGRAM_BOT_TOKEN is in FORBIDDEN_SKILL_SETTINGS while
   contracts/plugin_api.py:23 does not contain it - doc aligned to code
   (10 keys), code deliberately unchanged.
2. ABI-6 items NOT re-locatable on tip - recorded, NOT replaced by
   invention (f3 plan instruction): "failure-detector compat wrapper" and
   "3 underscore renames". Evidence of the sweep: compat/alias comment grep
   across ouroboros/ (12 candidates read - none is a failure-detector
   wrapper); AST scan for one-line delegating wrappers with
   fail/retry/error/detect/classify names (single hit:
   git_review_cycle._handle_revalidation_failure, which is the D18/D33
   module-handle idiom, not a compat shim); targeted reads of
   llm*/loop*/transport modules. Disposition: superseded-by-upstream inside
   the ABI-6 row; a future lane finding the real item re-opens it with
   bytes, not memory.
3. ABI-5 execution corrections against the f3 plan text:
   - The two ws5 "read exemption" tests are NOT floor tests but family
     read-carve mechanism tests that used the floor detector as vehicle;
     deleted only the detector's own test, RETARGETED the two mechanism
     tests to the surviving `_detect_safety_mode_self_lowering` (same
     composition through `_owner_control_mention_blocks`).
   - `effective_max_improvement_passes(has_deadline=)` existed solely for
     the until_deadline count-axis branch; the parameter was removed with
     the alias (callers: task_results wrapper + wallet cap + rails line;
     BudgetSnapshot.has_deadline and every TIME rail untouched).
   - The wallet-authority test derives its uncapped lane from
     OUROBOROS_REVIEW_MAX_CYCLES=unlimited now (the alias lane is gone);
     v664's deprecation-noise test now pins that resolve_budget_profile
     emits NO deprecation events at all.
   - Bench adapters (programbench, swe_bench_pro) switch to
     improvement_policy=fixed: behavior-identical because their explicit
     max_improvement_passes=6 was always the binding count axis.
   - Ratchet: tests/test_v664_acceptance_planning.py briefly crossed the
     1001 band (1005 lines) after a test rewrite - shrunk back to 996
     instead of minting a band rationale; BYTE_DEBT for
     tests/test_devtools_benchmarks.py regenerated 327935->327888
     (reduction) by the official generator.
4. Disclosed consequence (rides ABI-2/Q8=B): a pre-7.0 stored root contract
   whose normalized profile says until_deadline is judged malformed by the
   acceptance-wallet authority (pre-existing unknown-policy behavior);
   pre-7.0 task-result history is quarantined wholesale by ABI-2 in the
   same release and the ABI-7 RC auditor names the migration.

## From the f31b lane (extensions, base 29e2b045, 2026-08-31)

1. Plan line-ref drift, re-verified on base bytes: the supervised-future
   leak pinned as "extension_plugin_api.py:459-466" lives at :460-466 on
   29e2b045 (future minted at :460, the second `_require_open_locked`
   re-check at :461-462). The leak itself is REAL and was reproduced red
   by the direct regression test
   (tests/test_extension_registration_atomicity.py::
   test_supervised_future_never_leaks_when_unload_wins_the_registration_race)
   before the ABI-9 fix: the factory ran despite the refusal.
2. ABI-9 semantic tightening, disclosed: `on_unload` callbacks registered
   during a FAILED registration are no longer executed on abort (on the
   base they ran via unload_extension because the bundle pre-existed the
   register() call). Staged side effects (event-bus subscriptions,
   supervised runners, companion spawns) are disposed/never-started
   instead; on_unload fires only for a published extension. No test on
   the base pinned the old failed-register callback behavior.
3. FORBIDDEN_EXTENSION_SETTINGS reader refs from the f3 plan
   ("extension_plugin_api.py:513/:664") re-located on base bytes to
   :513 (companion env filter) and :664 (get_settings protected set) —
   both verified before the ABI-1 alias collapse.
4. ABI-1 execution notes (owner-ratified design + batch №6 answers):
   - Admission timing: the ratified text anchors the predicate "at NEW-PASS
     issuance"; the lane evaluates it EAGERLY, after the $0 free-replay gate
     and BEFORE the paid panel dispatch — no outcome of a dispatched panel
     could mint a PASS for an inadmissible payload, so dispatching would only
     burn reviewer money. Byte-identical re-review of grandfathered bytes
     still free-replays the recorded PASS first.
   - Reload-aggregation hole found and closed: a persisted
     plugin_api_admission FAIL finding re-aggregated to WARNINGS (executable!)
     on load_review_state; aggregate_skill_review_status now treats it as a
     structural gate like skill_preflight (PENDING under every enforcement).
   - Preflight infra failures now fail closed WITHOUT persisting (a transient
     breakage must not clobber live review state); genuine payload gate
     failures keep persisting PENDING as before.
   - 6.2=A scope note: the declarative dependency fingerprint is enforced on
     the extension liveness path (deps_declaration_desync). Script-skill deps
     flow through the same read_deps_state/specs-hash gates but their
     readiness callers are outside this lane's files — residual disclosed for
     the RC auditor (ABI-7) inventory.
   - launcher_bootstrap plan ref ":565-579 resync grants" re-located: the
     grant-carry seam landed as _carry_grants_across_reseed called from
     _reseed_native_skill_in_place (the :565-579 span on 29e2b045 is
     _stamp_native_seed_trust's docstring).
5. Test pinned to the pre-2.0 contract, updated with disclosure:
   tests/test_native_seed_trust.py seed fixtures wrote type=extension seeds
   WITHOUT the plugin_api field and asserted the native-trust stamp — under
   ABI-1 that stamp is correctly refused. The fixtures now declare
   plugin_api: "2.0" (matching real bundled seeds); the field-less refusal
   itself is pinned in tests/test_plugin_api_admission.py::
   test_native_seed_trust_is_closed_to_fieldless_extensions.
## From the f31c lane (F3.1-C schema/updater, base 29e2b045, 2026-08-31)
1. ABI-2 reader seam widened beyond the plan's single address: the plan named
   `load_task_result` (:665) as THE reader, but the sibling
   `list_task_results` feeds UI/recent (gateway/tasks.py:796) from the same
   rows - quarantine is implemented at BOTH, batched per scan. Direct
   observational globs (server_routing_context.py:207, gateway/tasks.py:803)
   are deliberately untouched: after the first swept read they see nothing,
   and touching them would be compat machinery Q8=B forbids. The quarantine
   subdirectory is invisible to every `*.json` glob (non-recursive).
2. Plan section 7 item (3) "ONE durable event / chat notice per batch" is
   superseded by the batch-6 answer 6.3=B: visibility is the durable events
   log ONLY - one `task_results_quarantined` row per read/scan batch, no UI
   counter, no chat notice (pinned by a no-chat-jsonl test). The move itself
   is the dedupe: a row can appear in exactly one batch ever.
3. Writer-inventory correction against "writers stamp": five writer sites
   exist on tip, four stamp (write_task_result, the acceptance-state and
   plan-review merge-writers in task_results.py, the owner_hurry projection
   writer). The cancel-receipt amend-writer
   (supervisor/terminal_delivery.py:1284) deliberately does NOT stamp: it
   never creates a row, its dict-copy merge preserves whatever stamp the row
   carries (so no downgrade path exists), and the module sits exactly at the
   1500-line band edge - a stamp there is correctness-redundant. Disclosed
   residual: a pre-7.0 row whose ONLY post-upgrade write is a cancel receipt
   stays unstamped and is later quarantined with its receipt - consistent
   with wholesale pre-7.0 quarantine (f30 entry 4).
4. Module-size: the ABI-2 machinery lives in a new leaf
   `ouroboros/task_result_schema.py` (task_results.py re-exports; callers
   and tests import through the facade). Inlining it drove task_results.py
   to 1592/1600 against the hard cap; after the split the ratchet manifest
   is byte-identical to the base (task_results.py 1465, band entry kept).
5. Disclosed interaction: `restore_pending_from_snapshot` probes each
   snapshot-pending task's result with `load_task_result(strict=True)`
   (queue_snapshot.py:305). A pre-7.0 unstamped row now raises there, so the
   task is terminalized through the existing result-authority custody path
   instead of being revived - the N-1-snapshot restore of pre-7.0 tasks
   degrades fail-closed, consistent with Q8=B wholesale quarantine.
6. Strict-path contract stability: for MALFORMED rows the pre-ABI-2 strict
   messages are kept byte-stable ("task result authority is unreadable or
   invalid" / "task result is unreadable or invalid" - test_review_cycles
   pins the former); schema refusals raise the new typed message with
   reason=quarantined_schema. Strict reads never mutate storage: an
   authority probe is not allowed to be the mover.
7. ABI-7a: `read_update_tx_strict` grew the fourth status `"future"`
   (integer stamp above ours; raw tx returned as evidence). Full strict
   caller sweep on the base: update_merge internal consumers fail closed via
   existing `!= "valid"` branches; `update_tx_phase` raises the typed
   refusal without writing; `_safe_restart_serialized` defers the restart;
   git_ops_reset.py:326 keeps `tx_matches` false and clears the orphan
   intent (fail-closed). A NON-integer stamp reads `corrupt` (evidence kept
   on disk, `{}` returned) - only a genuine newer-release stamp is `future`.
   An unstamped marker stays `valid`: that IS the N-1 transition contract.
8. F2.4 boot-finalize family untouched byte-wise except the dispatch
   docstrings and the future branch in `finalize_managed_update_on_boot`;
   the carrier-conflict crash floor (F2.4 ledger entry 4) keeps its existing
   pins - the shim suite adds the N-1 byte-form fixtures for every phase
   seam plus the marker upgrade-on-first-rewrite assertion.
9. Fixture sweep: 9 test files hand-writing task-result rows as
   current-version writers now stamp them (acting_subagents, presence_tools,
   tasks_list_slice, headless_task_events, context_drive_state,
   gateway_history, host_service_api, plan_review_public_projection - plus
   the new F12 suite writes both forms deliberately).
## From the f31d lane (F3.1 polosa D: ABI-3 -> ABI-10, base 29e2b045, 2026-08-31)

1. ABI-3 F11 inventory frozen BEFORE the first removal in
   docs/v7next/ABI3_GATEWAY_ALIAS_INVENTORY.md (the RC-auditor feeder). Key
   falsification against the plan text: the plan named api_types.js as a
   removal surface, but the lane constraint (web/ untouchable; chat.js at its
   BYTE_DEBT ceiling) plus the JS evidence made it a NON-surface: no alias
   has a functional JS reader (`resolveCostPair` falls back to the honest
   name; telegram/prefs aliases are JSDoc-only), so NO alias was HOT-DEFERRED
   - only the stale JSDoc typedef lines and the GATEWAY_CONTRACT_VERSION
   carrier switch are deferred, and tests/test_gateway_parity.py excuses
   exactly that frozen extra-set and nothing else.
2. Cost-alias removal is CLASS-level at the SSOT seams (cost_projection.py
   emitters strip the retired spellings; read tolerance + deprecated-wins
   precedence for stored pairs kept verbatim). Consumer fixes rode along in
   files outside the polosa-D list (disclosed cross-lane touches):
   agent_task_pipeline.py, supervisor/events_task_done.py (polosa C
   neighborhood - different hunks from C's ABI-2 seam at state.py:198/
   task_results.py:665/723; task_results.py itself was NOT touched),
   post_task_synthesis.py + synthesis_cost_text.py (pre-synthesis snapshot
   and prompt renderer moved to the honest with_children name - the snapshot
   feeds task_summary chat rows, i.e. gateway egress), tools/recent_tasks.py,
   tools/control_task_results.py.
3. Cross-lane invariant relied upon (for the coordinator's integration
   check): the "stale stored cost_usd beside a fresh honest name after a
   post-upgrade merge-write" class is consumed by polosa C's ABI-2 Q8=B
   quarantine (pre-7.0 unstamped records never reach the merge path). Within
   this lane's own tree the class is test-visible only via records written by
   current code, which now write honest names.
4. Ingress validation (Q7=A) is inbound-only by test-pinned design: history
   replay is egress and gateway/history.py must never import validate_ingress
   (pinned). PEP 563 falsified `__required_keys__` on Python 3.10 (string
   annotations make every total-class key read required, ExecutorRef lost its
   Required["type"]) - requiredness is re-derived from resolved hints +
   per-class totality in gateway/schema.py.
5. UpdateApplyRequest.strategy: the runtime silently defaulted a missing
   strategy to auto_merge while the contract declares it REQUIRED; the
   executable schema now enforces the contract as written (web client always
   sends it; no test posted a bare body expecting the default).
6. ABI-10 default panel deliberately resolves through
   get_review_models()/get_scope_review_models() (derived env plane), NOT a
   static list: preserves the identical review models for every config class
   (shipped defaults, single-direct-provider adaptation, bench env overrides)
   - "review models change nowhere". The SETTINGS-plane comma keys die
   (RETIRED_SETTING_KEYS, ghost purge); a comma-only-settings install gets
   the default panel exactly as ratified (5.4=A).
7. server_runtime.apply_runtime_provider_defaults was a settings-plane WRITER
   of the retired comma keys (direct-provider path INTRODUCED them; the
   prior-scope-default migration would KeyError post-retirement): it now
   normalizes only values it is fed and never introduces a retired key; the
   read-time getters own the direct-provider review adaptation (pinned per
   provider by new read-time tests). model_slots' singular->plural promotion
   removed (dead after purge; both files outside the polosa-D list -
   disclosed).
8. Bench templates (continual_learning, gaia, swe_bench_pro x4) migrated to
   structured OUROBOROS_REVIEWER_SLOTS with byte-identical model sets; the
   comma keys were dropped from all bench settings JSONs. NOT touched:
   operator_patches/*.patch (append-only artifacts) and the env-plane
   forwarding lists (server_runner/manifests/cybergym_lifecycle) - the env
   spellings remain the legitimate derived/operational plane.
9. Removed-with-evidence test clauses (the "test pinning the bug" class):
   test_gateway_parity's ChatOutbound cost_usd JSDoc pin; the legacy
   migration tests of reviewer_slot_config (phase-5 route envs, session-row
   migration, legacy advisory materialization) replaced by
   retired-envs-are-ignored pins; test_git_review_bypass_gate's
   "unroutable enabled session advisory" state is UNREACHABLE by construction
   post-ABI-10 (the parser refuses an enabled session advisory without a
   concrete target at save AND load) - the defensive fail-open branch stays
   covered via a synthesized config in test_skill_advisory_pre_review.
10. Residuals disclosed (NOT executed, outside the named scope): the
    phase-5 env reads that survive in review_substrate.scope_reviewer_slots
    (route_env_key plumbing for explicit-models callers) and the
    OUROBOROS_ADVISORY_REVIEW_ROUTE mentions in
    preflight_review_run/claude_advisory_review prose/vocabulary - env-plane
    remnants, candidates for the F3.3 sweep extension; the JS-side typedef
    cleanup + GATEWAY_CONTRACT_VERSION carrier switch (web lane).
## From the F3.1 lane A (typed organ, base 29e2b045, 2026-08-31)
1. extension_dispatch.py typed dispatchers (D04 entry 5, rows 187/188) ADOPTED
   WHOLE from the reference WITH BYTE PROOF: the tip file, the merge-base
   (8028f1df) file and the v6.64.0 file are md5-identical (4e9ad3ba…), so
   reference == tip + delta exactly (the same adoption class the lane used for
   mcp_client.py). The ToolRegistry methods `_dispatch_extension_tool` /
   `_dispatch_mcp_tool` and the hoisted `_extension_dispatch_candidate` retire
   from registry_core; call sites read the module handle. The unknown-name
   answer for a registered-but-not-live extension is typed
   EXTENSION_UNAVAILABLE (the D02 liveness bit); the truly-unknown name keeps
   the tip's alias-filtered legacy text (tip drift the oracle lacks). The
   `failure_kind` delta on extension_process_runner (unrowed in MIGRATION,
   named by D14 entry 10 as Ф3 territory) lands here: `ExtensionProcessError`
   gains the kwarg, only the deadline kill raises `failure_kind="timeout"`.
2. D09 typed-policy-refusal subfamily (D02 entry 4): the five ladder bodies
   drifted upstream after the fork (`plan_next_wire_retry` state machine,
   request-wire custody, effort-clamp discard rules), so the reference deltas
   were RE-DERIVED onto the tip structure: the three planning rungs decline a
   refusal; the retries twins raise it out of the bounded state machine before
   any wire-retry planning and out of the reroute/strip body-error arms instead
   of absorbing it into the first errored response, discarding the pending
   effort-clamp note on each raise path (tip custody rule the reference
   predates). classify_llm_exception branch inserted before the prose
   heuristics, after the tip's provider_code read (the tip computes
   provider_message/classification_text the oracle lacks — refusal outranks
   them). tests/test_llm_typed_policy_refusal.py carried whole: 25 passed with
   zero adaptation. Goldens: the two typed_policy_refusal cases returned to
   fallback_ladder.json (15 -> 17) with `expected` RE-RECORDED from this
   tree's live code via the suite's own --write entry; the write left all 15
   existing cases untouched (append-only diff = no accidental drift), and both
   recorded blocks carry the oracle-intended semantics (refusal raises; the
   exception case spends exactly one physical send).
3. Reference test adaptations, each disclosed in-file at the non-verbatim
   spot (reverse-mapping rule §5.3-Δ item 2):
   - registry facade: the reference pins an exact-32-name minimal facade; this
     tree deliberately keeps the broad historical import surface, so the pin
     is re-derived as an AST "the facade module DEFINES nothing but the
     disclosed read-carve helper" plus owner-leaf homing/retirement asserts
     (test_registry_core, test_registry_guard_process ×2).
   - guard collaborator patch points follow this tree's `_registry()`
     call-time-handle idiom (protected_artifact_shell_block_reason,
     workspace_executor_state_write_block, build_resolved_resource_binding,
     resolve_shell_cwd, shell_cwd_block_message, system/active_repo_dir_for,
     light_shell_repo_mutation, runtime_data_guard_targets,
     workspace_git_safety_violation, run_shell_git_block_reason, run_cmd for
     git_vcs_ops); the reference's `shell_has_write_indicator` seam does not
     exist here — the tip write-shape seam is `non_interpreter_write_shape`.
   - the managed-update resolver pin re-targets the tip's TYPED
     `authorized_assisted_task_strict` (adds the corrupt-marker A4-channel
     clause the reference could not know).
   - SCOPE_REVIEW_FLOOR rows (detector signature, denial text, code-contract
     row, precede-safety parametrization) removed: the setting, guard and code
     were retired by ABI-5 (owner Q10=A) in F3.0.
   - detector-family signatures pin the tip's whole-family `writeish`
     read-carve; three constant cardinalities follow upstream drift
     (secret markers 17->18, denied read options 11->12, owner-state stems
     12->14); `_workspace_shell_write_block` pins the upstream
     `write_target_argvs` parameter.
   - plan-review pins: the reference's sync-side `_record_raw_plan_request_attempt`
     and the vacuous-note wrapper (`_reuse_or_disposition_plan_review`,
     `_VACUOUS_*_NOTE`) do not exist on tip — the parametrized wrapper test is
     re-derived over the tip's three projection paths (review mode, vacuous
     disposition fall-through, `_apply_disposition`).
   - `_parse_plan_review_control` readers (test_plan_spec, test_plan_review,
     plan_spec docstring) re-point to its new home tools/plan_render.
   - two loop fakes (test_openai_chat_dispatch._FakeTools,
     test_owner_hurry_s3._ProbeTools) gain `execute_result` adapting their
     text exactly as the registry adapts a legacy handler — the loop now reads
     the typed seam.
4. Reference DELTA re-applied, not replayed: the plan handler's pool hop wraps
   `asyncio.run` in `contextvars.copy_context().run` so the sidecar
   publication reaches the dispatching thread's slot; the reference's
   surrounding `asyncio.wait_for` wrapper-timeout machinery is NOT reproduced —
   the tip deliberately retired the nested wait (its comment explains the
   cancel-then-block hazard), and replaying the span verbatim would have
   reverted that decision (re-prove-trap class, D15 entry 3).
5. Protection closure: registry_core.py + tool_result.py membership in
   SAFETY_CRITICAL_PATHS/HOT_CODE_PATHS verified landed with the re-split
   commit (parity pin green); extension_dispatch.py — already safety-critical —
   JOINS HOT_CODE_PATHS here because the dispatch bodies moved onto it from
   the hot ToolRegistry class (the same parity rule; oracle carries the same
   membership), and the parity pin now lists it.
6. Owner 6.1=A edge checked and NOT implicated: lane A changes no
   admission/review/PASS semantics — the extension liveness refusal existed on
   tip and is only retyped (EXTENSION_UNAVAILABLE), so the auto_review=false
   contract (no PASS issued, nothing blocked) is untouched by the typed organ.
7. Function-size law: the EXTENSION_UNAVAILABLE branch pushed
   `_execute_legacy_text` to 303 lines; resolved by extracting the
   module-level `_unknown_tool_result` helper (behavior identical), not by a
   band exception. tests/test_tool_result.py enters the 1001-1500 band with a
   rationale via the official regenerator.
## From the F3.1 conformance fix-round (base 9edb9199, 2026-08-31)

Dispositions for the six blocking findings of the Ф3.1 conformance review
(GPT-5.6 Sol, read-only, range 29e2b045..9edb9199). One finding per entry;
every closure carries its pin.

1. ABI-9 ordering (finding 1) — FIXED. `_publish_registrations` is now
   validate -> effects -> swap under ONE `_lock` hold: the definitive
   unload/conflict validation runs BEFORE any deferred side effect
   (supervised runners, companion spawns, bus subscriptions), and the swap
   follows in the same critical section, so no concurrent unload/conflicting
   publication can interleave anywhere between the three steps (both mutate
   only under `_lock`; CompanionSupervisor uses its own lock and never takes
   the registry lock — checked). Event subscriptions are STAGED
   (`_StagedEventSubscription`; `EventBus.subscribe` accepts a pre-minted
   sub_id so the id returned by `subscribe_event` is the id the bus attaches
   at publication). Pins: tests/test_extension_registration_atomicity.py::
   test_conflict_refused_publication_has_zero_external_effects (conflict
   arising between staging and publication -> refusal with factory never
   started, bus untouched, bundle empty) and ::
   test_event_published_before_publication_never_invokes_the_handler
   (pre-publication invisibility, not eventual cleanup; sub_id fidelity).
   The pre-existing leak regression and disposer-ABI pins stay green.
2. ABI-2 readers (finding 2) — FIXED. POST /api/tasks identity-collision
   probe switched from the fail-soft loader (which QUARANTINED the probed row
   and then read "no result", freeing the id) to `load_task_result(strict=
   True)`: any stored row — admissible or not — keeps its identity occupied
   (409) and the probe never mutates storage. The unfiltered GET /api/tasks
   slice-before-projection path now runs the same fail-soft admission as
   `list_task_results` (quarantine + ONE batched `task_results_quarantined`
   event per scan, 6.3=B) via the schema primitives imported from
   `ouroboros.task_result_schema`. Pins: tests/test_headless_task_api.py::
   test_task_api_identity_collision_check_is_strict_not_fail_soft (both an
   unstamped and a torn row: 409, bytes unchanged, no quarantine dir) and
   tests/test_tasks_list_slice.py::
   test_unfiltered_list_slice_is_admission_aware_with_one_batched_event.
3. ABI-3 producers (finding 3) — FIXED as a CLASS, read tolerance untouched.
   Honest-name cutover of every remaining task-result/task-done producer:
   the four named files (supervisor/task_admission.py,
   events_schedule_task.py, workers.py x2 fallbacks, events_task_done.py
   root/subtree/unavailable branches) PLUS the same class found by the
   sweep in supervisor/queue.py, cancel_publication.py, task_lifecycle.py
   (x2 fallbacks), supervisor/state.py `reconstruct_task_cost` internals
   (fields seam kept as idempotent guard; the tuple path reads the honest
   key), ouroboros/post_task_checkpoint.py, and
   ouroboros/post_task_synthesis.py child-evidence rows (resolve stored pair
   deprecated-wins, emit honest name) — cross-file touches beyond the
   review's four examples are this disclosure. Fan-out pin:
   tests/test_gateway_abi3_removals.py::TestAliasProducerFanOutSweep — an
   AST sweep of EVERY ouroboros/ + supervisor/ module: (a) no
   `write_task_result` call passes a retired alias (kwarg or dict-literal
   arg; NO allowlist), (b) every dict-key/subscript emission of a retired
   spelling must be an allowlisted INTERNAL non-gateway plane (physical
   ledger rows, llm/usage observability events, review/evidence receipts,
   subagent envelope, evolution state, custody settlement events, reflection
   records — 20 rows, each with its plane named), (c) stale allowlist rows
   FAIL the test, (d) the three non-cost aliases have zero emissions with no
   allowlist at all. Explicitly NOT cut over (they are not the gateway
   alias): the internal planes above keep their own `cost_usd` field
   spellings — renaming ledger/receipt/envelope schemas is outside ABI-3's
   inventory and would be an unsanctioned break of their own producer/reader
   pairs.
4. ADOPTION (finding 4) — ABI-2/ABI-3/ABI-9 remain `done` LAWFULLY after the
   fixes above; each row's what/hook columns now name the fix-round closure
   and the new pins (`scripts/v7next_adoption.py` green). No residual was
   left open, so no row moved to in-progress.
5. Domain manifest (finding 5) — the stale `ouroboros/contracts/api_v1.py`
   row (module removed by ABI-3 lane D3) dropped from
   scripts/v7next_domains.toml [modules]; DOMAIN_QUOTIENT_REPORT.md
   regenerated by the official scripts/v7next_domain_report.py: 487 modules,
   "manifest drift: none (manifest == tracked population)".
6. Whitespace (finding 6) — `git diff --check 29e2b045..HEAD` is now clean
   (rc=0). Provenance checked BEFORE fixing, per the campaign's verbatim-
   bytes rule: all 50 trailing-whitespace lines are campaign-authored (the
   `_shell_guard_text` rewrite introduced by lane A commit 2e575b82; the
   frozen oracle v7_wip @ 9f691656 contains no `_shell_guard_text` call at
   all), so no byte-proved span was touched and nothing had to be declined.
   Blank-EOF fixes: tests/test_core_native_results.py, and
   ouroboros/tools/registry.py — a PROTECTED file; the delta is exactly the
   two trailing blank lines left by the lane-A facade assembly, zero code
   bytes (disclosed in the commit).
7. Function-size law (found by the fix-round's own gate run, not by the
   review): the ABI-2 strict-probe block pushed `api_tasks_create` to 310
   lines and `scripts/regenerate_size_ratchet.py --check` refused new
   function debt. Resolved per the lane-A entry-7 precedent — module-level
   helper extraction (`_task_identity_occupied`, behavior identical), no
   band exception, no manifest change. Enforcement note for auditors: this
   line's ratchet is pairwise base-vs-tip with NO committed-history replay
   (`ouroboros/review.py::validate_size_ratchet` docstring), so the four
   fix-round commits between the ABI-2 landing and the extraction carry the
   over-limit function in their trees without being audited surfaces; the
   final tree and the CI base (the pushed 9edb9199) are both clean, and the
   local degraded parent-tree check is green from this commit's parent on.
   DOMAIN_QUOTIENT_REPORT.md regenerated once more so its analyzed-inputs
   fingerprint matches the final runtime tree.

## From the F3.1 conformance fix-round-2 (base aae647fb, 2026-08-31)

Round-2 verdict (GPT-5.6 Sol): three findings NOT-CLOSED (ABI-9, ABI-2,
ABI-3 + the dependent ADOPTION claims); Domains manifest, git hygiene,
helper extraction and new-defects CLOSED with no action owed. Dispositions:

1. ABI-9 (finding 1) — FIXED. The round-1 order (validate -> effects ->
   swap) attached the bus subscription and started the supervised runner
   BEFORE the registry swap; EventBus.publish() takes only the bus's own
   lock, so a concurrent publish could invoke a handler of a
   not-yet-published extension, and the round-1 pins never exercised the
   window. `_publish_registrations` is now validate -> SWAP -> attach under
   the SAME single registry-lock hold: the validated snapshot becomes the
   authoritative bundle (digest minted, every attachable effect recorded on
   the bundle) BEFORE any effect attaches, so a handler is visible to the
   bus only for an already-published extension. A post-swap attach failure
   is disclosed (log.warning) and raised into the callers' standard
   dispose+unload path — load_extension/unload_extension reap everything the
   bundle recorded (surfaces, sub_ids, futures, companion names, on_unload).
   Pins (tests/test_extension_registration_atomicity.py): a REAL
   barrier-sequenced race — a publish interleaved between validation and
   attach never invokes the handler while the bundle is provably already
   published at attach time; the supervised effect observes a published
   bundle at its start; a post-swap attach failure ends with empty
   registries, an empty bus and the extension's on_unload having run.
   ARCHITECTURE rows (extension_loader, extension_plugin_api) and the
   registry-state staging docstrings restated to the true order.
2. ABI-2 (finding 2) — FIXED. `_raw_sorted_result_names()` no longer
   silently drops a file whose bytes fail to parse: malformed candidates are
   returned separately and `_tasks_list_payload` routes EVERY one through
   the same admission reader — quarantine plus a contribution to the SINGLE
   batched `task_results_quarantined` event of the scan — even when the
   candidate would have sorted beyond the slice window (the sort had read
   its bytes anyway). Torn-concurrent-write safety moved to where it truly
   lives: the quarantine primitive re-checks under the row's own write lock
   (kept_admissible), and a malformed name is never memoized. Disclosed
   residual (documented in the payload docstring): a PARSEABLE inadmissible
   row beyond the window is not classified by the sliced request — the next
   full/filtered scan quarantines it. TEST-CONTRACT DISCLOSURE: the pre-fix
   clause test_torn_result_file_is_skipped_then_recovered_without_
   poisoning_memo pinned the silent drop as "torn-write tolerance" — a test
   asserting the defect; replaced by
   test_malformed_result_file_is_quarantined_not_silently_dropped and the
   REAL slice-boundary pin
   test_malformed_candidate_beyond_the_slice_window_is_still_quarantined
   (rows > limit; one batch event spans both sides of the boundary).
3. ABI-3 (finding 3) — FIXED as the projection-boundary semantics. The ABI
   carries no alias: outbound surfaces (public_task_result, task detail,
   history frames, the cancel path through queue) emit ONLY honest names;
   stored legacy resolves deprecated-wins and NORMALIZES at projection and
   at re-write. Landed: TASK_COST_META_FIELDS honest-only;
   write_task_result merges over with_cost_aliases(existing) and normalizes
   the merged row (a legacy mutator's edit still wins its pair, then is
   stripped); public_task_result normalizes the top level + subagent
   envelope + loop-outcome usage; history mapper converts via
   carry_cost_meta at all three copy seams (task-summary replay,
   progress-meta replay, terminal-truth annotate); task_lifecycle
   stored/child cancel costs via carry_cost_meta; post_task_checkpoint
   task_cost_finalized event via carry_cost_meta, with an explicit SCRUB set
   that still pops the retired spellings (a stale legacy replica cannot
   smuggle an amount past deprecated-wins at the write seam). Producers
   whose data reaches the public projection are cut over and REMOVED from
   the sweep allowlist: build_subagent_envelope/envelope_from_task (key and
   kwarg now accounted_upper_bound_usd), the pipeline unavailable-patch, the
   loop-outcome usage sub-dict. Fan-out pin upgraded per the round-2
   mandate: (a) runtime projection-boundary pins
   (TestProjectionBoundaryNormalization — stored legacy row -> outbound
   payload deep-scanned for alias keys; rewrite normalization;
   legacy-mutator honor-then-strip) catch generic passthrough no AST scan
   can see; (b) the static sweep now also treats keyword args on ANY call
   as emission-shaped, and its allowlist is PER-SITE (file, alias,
   enclosing scope) — a new emission in an allowlisted file fails, any
   stale row fails; (c) a dedicated pin bans allowlisting outcomes.py /
   subagents.py / agent_task_pipeline.py. TEST-CONTRACT DISCLOSURE (the
   round-2 mandate names these as OLD-ABI contract tests): converted to
   honest-name assertions — tests/test_gateway_history.py (windowed anchor
   cost, terminal cost truth, override precedence, nullable bounds,
   task-summary flat-field passthrough), tests/test_tasks_list_slice.py
   compact-row cost clause, tests/test_cost_projection.py meta-field
   derivation + stored-legacy-tolerance clauses,
   tests/test_task_summary.py snapshot fixture (its legacy key was stale —
   the real _pre_synthesis_usage_snapshot emits the honest name),
   tests/test_headless_task_artifacts.py mirror-cost merge and
   finalized-accounting clauses, tests/test_task_result_monotonic.py
   kept-cost clause. NOT changed: the JS read seam keeps pair tolerance
   (web mirror switch stays deferred per the ABI-3 row), and the internal
   non-gateway planes (ledger rows, review/evidence receipts, evolution
   state, custody settlement events, reflection/consciousness records) keep
   their own spellings under anchored per-site allowlist rows.
4. ADOPTION claims (finding 4) — ABI-2/ABI-3/ABI-9 stay `done` LAWFULLY
   after the fixes above; each row now names the fix-round-2 closure and the
   new pins (scripts/v7next_adoption.py OK, 36 rows). The absolute ABI-9
   claim in docs/ARCHITECTURE.md:291 is restated to the true order
   (validate -> swap -> attach) including the post-swap-failure disclosure,
   so the "refused registration publishes nothing" clause is now exactly
   true (refusal = validation failure; a post-swap attach failure is a
   published-then-disposed bundle, said in the same sentence).
5. Findings 5-8 (domains manifest, git hygiene, helper extraction, new
   defects) — CLOSED by the verdict itself; no action owed, nothing
   changed there in this round.
6. Module-size law (found by the fix-round-2's own gate run, not by the
   review; the round-1 entry-7 precedent): the ABI-2 admission routing
   pushed ouroboros/gateway/tasks.py past the 1600-line hard cap, and the
   ABI-3 honest-name comment pushed outcomes.py::derive_loop_outcome past
   300 lines. Resolved by extraction, no band exception, no debt entry:
   the raw creation-ts sort scan + malformed-candidate admission moved to
   the new ouroboros/gateway/task_list_scan.py (tasks.py 1615 -> 1563;
   same objects imported back; ARCHITECTURE row added; module mapped to
   D11 in scripts/v7next_domains.toml, DOMAIN_QUOTIENT_REPORT regenerated
   by the official script — 488 mapped, drift none), and the loop-outcome
   usage snapshot became module-level `_loop_usage_snapshot`
   (derive_loop_outcome 304 -> ~292). Each extraction lands INSIDE the
   commit whose growth caused it, so every first-parent tree of this
   round satisfies its own manifest (no condemned intermediate commits —
   the local unpushed round-2 series was arranged for this before any
   push; the pushed tip aae647fb was not rewritten).
7. Serial-battery addendum (found by this round's full CI-shape battery,
   serial pass): tests/test_cancel_live_kill_path.py
   ::test_e2e_child_finishing_before_the_kill_keeps_its_completed_result
   asserted the OLD-ABI alias on the kept row and the task_done relay —
   converted to the honest name (same class as the entry-3 disclosure
   list; the child writer's legacy kwarg is honored deprecated-wins, then
   stripped by the write seam).

## From the F3.1 conformance fix-round-3 (base f8e579de, 2026-08-31)

Round-3 verdict (GPT-5.6 Sol, read-only @ f8e579de): ABI-9 and ABI-3
NOT-CLOSED on the enumerated tails, ADOPTION/ARCHITECTURE claims dependent on
them, one NEW UI defect; ABI-2, ratchet extractions and manifest CLOSED with
no action owed. Dispositions:

1. ABI-9 (finding 1) — FIXED as the disclosed STAGED PROTOCOL, not a false
   "one atomic publication" absolute. (a) The OOP load published surfaces
   and companions as TWO transactions (extension_loader
   _register_out_of_process_surfaces + _spawn_out_of_process_companions),
   with the second re-minting bundle.generation_digest without re-stamping
   published descriptors. Both are replaced by ONE staged publication:
   _stage_out_of_process_surfaces validates catalog descriptors through the
   same _stage_surface_locked seam the in-process register() window uses,
   and _publish_out_of_process_registration stages surfaces AND companion
   spawns on one PluginAPI snapshot — one validate -> SWAP -> attach
   transaction. The one structurally LATER publication that remains —
   server-side companion recovery (reconcile_server_companions) onto a live
   bundle — mints a fresh digest and _publish_registrations RE-STAMPS every
   already-published descriptor the bundle owns in the same lock hold, so
   per-surface provenance never diverges from bundle.generation_digest.
   (b) The recovery failure path was a silent _abort_registration leaving
   the extension half-alive; ANY failure of the shared seam now routes
   through the standard dispose+unload path (unload_extension). (c) Unload
   visibility: _unload_extension_locked popped the bundle and surfaces
   BEFORE the bus unsubscribe and runtime-API close; the order is now
   outside-in — subscription ids + the _unloading latch snapshot in ONE
   registry-lock hold (no publication can interleave), bus unsubscribe,
   runtime-API close, THEN bundle/surface removal, future cancel, companion
   stop, module purge. RESIDUAL BY DESIGN (pinned + disclosed in
   EventBus.publish's docstring): the bus COPIES subscribers under its own
   lock before invoking handlers, so a publisher that copied a handler
   before the unsubscribe may still invoke it after surfaces are gone; the
   supported guarantee is "a publish STARTED after unsubscribe never
   delivers", and the closed runtime API + _unloading latch make the late
   call a host no-op. Pins: test_out_of_process_surfaces_and_companions_
   publish_as_one_transaction, test_companion_recovery_failure_unloads_
   instead_of_silent_abort, test_late_publication_restamps_already_
   published_descriptors, test_unload_closes_bus_and_runtime_visibility_
   before_surfaces_leave, test_publish_started_after_unload_never_delivers.
   TEST-CONTRACT DISCLOSURE: test_spawn_out_of_process_companions_host_
   spawns_declared_name renamed to test_publish_out_of_process_registration_
   host_spawns_declared_name (the unified seam it exercises); the two
   catalog-revalidation suites and the loader-extraction _STAYED list now
   name the unified surface (_publish_out_of_process_registration /
   _stage_out_of_process_surfaces).
2. ABI-3 (finding 3) — FIXED in depth. (a) build_subagent_envelope
   normalizes the stored usage snapshot BEFORE embedding (deprecated-wins
   kept; the amount fallback reads the resolved honest name). (b+c) ONE
   shared normalizer cost_projection.normalize_task_result_cost_planes (top
   level + subagent envelope + envelope.usage + loop_outcome.usage) serves
   BOTH public_task_result and write_task_result (both merge passes) — the
   sanctioned known-paths + deep-test-scan variant of the round-3 mandate:
   internal evidence planes (review receipts, ledger rows) stay their own
   schemas per the round-2 disposition, named per-site in the sweep
   allowlist. (d) Evolution history: the update_evolution_campaign_after_
   task row now stamps accounted_upper_bound_usd (allowlist row REMOVED);
   the one internal reader (Recent Campaign Cycles prompt block) resolves
   the pair deprecated-wins; gateway/state._evolution_state_public converts
   stored legacy rows at the /api/state projection boundary (copy-on-write
   over the shared snapshot). (e) The deep-scan fixture now places the
   legacy alias on the ACTUALLY SUPPORTED producer path
   subagent_envelope.usage.cost_usd beside the envelope-root spelling, with
   resolved-amount assertions on the public projection, the task-detail
   endpoint and a new rewrite pin. (f) The sweep allowlist is COUNT-
   ANCHORED per site — (file, alias, scope) -> (reason, exact count); a new
   emission inside an allowlisted function breaks the anchor and fails.
   (g) Own AST sweep re-run: 55 emission-shaped sites dispositioned — 1
   honest cutover (the evolution history row), 3 sites under the
   events_evolution_done anchor re-classified honestly (2 internal
   lifecycle/checkpoint call kwargs + 1 supervisor.jsonl observability row,
   converted at the /api/logs boundary on replay), remaining 51 = internal
   planes kept per-site with exact counts.
3. New defect (finding 6) — FIXED. /api/logs emits the honest name but
   web/modules/log_events.js read only `cost_usd ?? cost`, so the LLM-round
   money column was empty after a reload. All three read sites now resolve
   through the existing SSOT JS helper accountedUpperBound() (the
   resolve_cost_pair mirror, deprecated-wins) with the live-frame `cost`
   spelling last — deliberately the shared precedence rule rather than a
   hand-ordered honest-first list, so a diverged stored pair tells the same
   story on every surface. Pinned in web/tests/cost_presentation.test.js
   (backfill name, live frame, diverged pair). chat.js (BYTE_DEBT ceiling)
   untouched; log_events.js stays in its band.
4. ADOPTION/ARCHITECTURE (finding 4) — restated to the post-fix truth:
   ABI-9 row + docs/ARCHITECTURE.md extension_loader/extension_plugin_api
   rows describe the staged protocol (single OOP transaction, recovery
   restamp + unload-on-failure, outside-in unload visibility, EventBus copy
   residual); ABI-3 row + the cost_projection ARCHITECTURE row describe the
   shared nested-plane normalizer and the boundary conversions
   (/api/logs, /api/state evolution history) with the internal-plane
   residual named. scripts/v7next_adoption.py OK (36 rows).
5. Findings 2 and 5 (ABI-2, ratchet extractions/manifest) — CLOSED by the
   verdict itself; nothing changed there in this round.

## From the F3.1 conformance fix-round-4 (base 163c2765, 2026-08-31)

Round-4 verdict (GPT-5.6 Sol, read-only @ 163c2765): ONE blocker — the ABI-9
companion-recovery lifecycle TOCTOU (finding 1); ABI-3 and Logs UI CLOSED with
no action owed; ADOPTION/ARCHITECTURE NOT-CLOSED only as a dependent of the
race. Dispositions:

1. ABI-9 recovery TOCTOU — FIXED as a GENERATION-BOUND protocol. The race:
   ensure_companions_running snapshotted liveness/bundle, then published
   WITHOUT the lifecycle lock (production re-invokes it from
   extension_reconcile_queue after locked reconciliation returns); a
   concurrent unload could complete in the window, after which the stale
   recovery re-created an empty bundle inside _publish_registrations
   (bundle-if-None branch) and started its companion — resurrecting a
   companion-only bundle after disable/unload. The fix, per the pinned
   protocol: (a) the recovery snapshot carries the observed
   bundle.generation_digest (read in the same registry-lock hold as the
   companion names); (b) the recovery publication runs UNDER the lifecycle
   lock and _publish_registrations(require_live_generation=...) re-validates
   under the registry lock that the observed publication is STILL live — a
   vanished or reloaded bundle raises the typed ExtensionStaleRecoveryError
   BEFORE any mutation (zero effects; ensure_companions_running surfaces it
   as the typed action "stale_recovery_refused"), and the recovery form of
   _publish_out_of_process_registration structurally REQUIRES a pre-existing
   live bundle (exactly-one-of form gate: current_hash XOR
   expected_generation), so recovery can never create a bundle; (c) the
   failure-disposal is generation-bound: unload_extension gained
   expected_generation and no-ops WITH DISCLOSURE (warning log) when the
   live generation is not the one this recovery observed or itself swapped
   in (_published_generation), so a failed recovery can never unload a newer
   publication. Pins: test_unload_completing_between_snapshot_and_
   publication_refuses_recovery (deterministic same-thread barrier — the
   unload completes between the snapshot and the publication), test_
   recovery_publication_refuses_on_generation_mismatch_without_effects,
   test_generation_bound_disposal_skips_a_newer_publication, plus the two
   recovery-form atomicity tests updated to the generation-bound call form.
   TEST-CONTRACT DISCLOSURE (test-that-pinned-the-bug): the clause of
   test_publish_out_of_process_registration_host_spawns_declared_name that
   asserted the recovery-form helper accepts NO pre-existing live bundle and
   spawns anyway pinned the resurrection defect itself; it is REPLACED by
   test_recovery_publication_requires_a_pre_existing_live_bundle (opposite
   pin: typed refusal, zero effects, nothing created), and the surviving
   host-spawn/trust-boundary clauses now use the initial-load form.
2. LOW — the stale reference to a nonexistent reconcile_server_companions in
   _publish_out_of_process_registration's docstring is gone; the docstring
   names the real recovery caller (ensure_companions_running) and the
   generation-bound contract. (The same stale name inside the round-3 ledger
   section above is historical record and stays as written.)
3. ADOPTION/ARCHITECTURE — restated to the post-fix truth: the ABI-9 row and
   the ARCHITECTURE extension_loader/extension_plugin_api rows now describe
   recovery as generation-bound onto the still-live observed publication
   (typed zero-effect refusal otherwise; generation-bound disposal), not as
   an unconditional "re-publishes onto the already live bundle".
4. RATCHET-DRIVEN MOVE (disclosed): extension_loader.py sits at the pinned
   <=1000-line extraction bound with 2 lines of headroom, so the fix is
   funded by moving _stage_out_of_process_surfaces whole into
   extension_child_catalog.py — its natural owner (it composes ONLY the
   child-catalog validators + registry maps + the PluginAPI staging seam and
   needs nothing from the loader); the loader re-exports it, the extraction
   contract (_MOVED_OWNERS/_STAYED) and the ARCHITECTURE child-catalog row
   are updated, and the leaf never imports the loader (DAG preserved).

## From the F3.1 conformance fix-round-5 (base c26c89a3, 2026-08-31)

Round-5 verdict (GPT-5.6 Sol, read-only @ c26c89a3): NEEDS FIXES — ONE HIGH
(the round-4 "zero effects" claim was false on the filesystem: recovery
mutated authorization state BEFORE the generation fence) plus one MEDIUM on
the strength of the round-4 test pins; verification points 2 and 5 CLOSED
(generation-bound disposal; extraction/sizes). Dispositions:

1. HIGH — stale recovery mutated `auth_token.json` before the generation
   fence: FIXED by post-fence token materialization. The defect:
   `register_companion_process` called `get_skill_token()` during descriptor
   build (before publication), and `mint_skill_token` WRITES
   `auth_token.json` whenever the stored token is missing or its bound
   content hash mismatches the live recompute; the fence sits in
   `_publish_registrations`, so a recovery that lost the race to an
   unload/reload holding a stale payload snapshot (an old skill root still
   on disk with the pre-update content) rotated the G2-bound token file and
   only THEN raised `ExtensionStaleRecoveryError` — the live G2 companion,
   spawned with the current token in its env while the Host Service rereads
   the file on every request (`host_service.authenticate_token_payload`),
   was left permanently unauthorized. The fix separates descriptor build
   (pure computation — env carries no HOST_SERVICE_TOKEN) from token
   materialization: `_publish_registrations` mints the token and injects it
   into every staged companion descriptor's env only inside the post-swap
   attach, AFTER `require_live_generation` admitted the publication, in the
   same registry-lock hold (a mint failure there routes through the
   standard dispose+unload path like any attach failure). The initial-load
   path is unchanged in effect — the token is still legitimately minted at
   its publication and the spawned descriptors reference it — and the
   runtime `get_skill_token()` API still mints on demand for a live
   extension. Pin (red pre-fix on both):
   test_stale_recovery_does_not_break_live_publication_authorization (the
   verdict's exact repro through the REAL entry — G1 loaded from an old
   root, in-window unload+reload of v2 content from a new root, stale
   refusal, byte-identical token, end-to-end
   HostServiceContext.authenticate_token_payload success for the G2
   spawn-env token) and the token-absence clause of
   test_recovery_publication_requires_a_pre_existing_live_bundle (a
   no-live-bundle refusal must not CREATE the token file).
2. MEDIUM — round-4 pin strength: (а) the unload-window interleaving test
   now also asserts `auth_token.json` is byte-untouched after the stale
   refusal, and the direct-call fixtures were moved off `drive_root/state`
   onto the production per-skill directory
   (`skill_state_dir(drive_root, name)` — the directory the Host Service
   actually scans); (б) the generation-mismatch test is rebuilt through the
   REAL recovery entry: `ensure_companions_running` with a deterministic
   in-window unload+reload (same payload), asserting the typed refusal, the
   preserved NEW generation, no recovery spawn, and the untouched token
   file — no directly supplied digest remains in that test; (в) the new
   live-G2-authorization test above. DISCLOSED: the byte-equality clauses
   in (а) and (б) are belt-and-suspenders rather than red-pre-fix pins —
   with an unchanged payload the pre-fix mint was a read (hash match, no
   rotation); the red-pre-fix coverage of the HIGH lives in the two pins
   named in item 1.
3. Docs — the ABI-9 row (ADOPTION_v7next.md), the ARCHITECTURE
   extension_loader/extension_plugin_api rows, and the
   `ExtensionStaleRecoveryError`/`mint_skill_token`/`_publish_registrations`
   docstrings now state the post-fence materialization explicitly: the
   round-4 "zero effects / before any mutation" wording is true only as of
   this round, and the round-4 ledger section above stays as written
   (append-only historical record).
4. Size pins: extension_loader.py untouched (998); extension_plugin_api.py
   998 after comment condensation (both within the <=1000 extraction pin,
   600 <= plugin API respected).

## From the F3.1 conformance fix-round-6 (base 1aae9868, 2026-08-31)

Round-6 verdict (GPT-5.6 Sol, read-only @ 1aae9868): NEEDS FIXES — ONE MEDIUM
(pre-fence filesystem side writes remained: the round-5 "pure computation"
claim was still false for `env_from_settings` manifests and the recovery
entry's state-dir mkdir) plus ONE LOW (a token rotation can strand an
already-running companion on its old spawn-env token); no HIGH; verification
points 3/4/5 CLOSED. Dispositions:

1. MEDIUM — pre-fence side writes: FIXED by extending the round-5 post-fence
   materialization to the WHOLE companion env. The defect:
   `register_companion_process` called `_scrub_env` during descriptor build,
   and for a manifest with `env_from_settings` that invokes
   `load_settings()` — which creates/unlinks the settings lock file
   (`config._acquire_settings_lock`) and can PERSIST a context-mode
   settings migration (`config.load_settings_lock_held` →
   `normalize_and_persist_context_mode_compat`) — before the generation
   fence in `_publish_registrations`; additionally `ensure_companions_running`
   resolved the state dir via the creating `skill_state_dir()` before the
   fence. The fix: the staged spawn now carries the manifest companion spec
   (`_StagedCompanionSpawn.spec`) and the descriptor is built with an EMPTY
   env; `extension_child_catalog.materialize_companion_env` fills it —
   settings-derived values, manifest env overlay, host bridge URL,
   isolated-dep PYTHONPATH and the auth token — only inside the post-swap
   attach, after `require_live_generation` admitted the publication, in the
   same registry-lock hold where the state dir is now created
   (`mkdir(parents=True, exist_ok=True)`) and the token is minted; the
   recovery entry resolves its path via the new non-creating
   `skill_state_path`. Env precedence is preserved (settings-derived base,
   then manifest overlay, then reserved bridge keys). Pin:
   test_stale_recovery_with_env_from_settings_has_zero_filesystem_effects —
   an `env_from_settings` manifest recovery losing the race to an unload is
   a typed refusal with ZERO `load_settings` calls from `_scrub_env` (the
   lock-file/migration hazard) and an unchanged data-root file tree; the
   same test proves the post-fence path still delivers the materialized env
   (skill name, token, bridge URL) to the publication's spawns.
   DISCLOSURE: the pin's semantic red-pre-fix content is the
   `load_settings` tripwire (pre-fix, descriptor build called `_scrub_env`);
   the test as written cannot execute verbatim on the pre-fix tree because
   its interleave seam (`skill_state_path`) is introduced by the fix itself
   — the round-4/5 interleave tests were migrated to the same seam.
2. LOW — stale spawn-env token after an accepted rotation: RESIDUAL BY
   DESIGN, no code change (proportionality: no degradation case exists).
   Preconditions analysis: `mint_skill_token` rotates ONLY when the stored
   token file is missing, carries no token (corrupt), or its bound
   content_hash mismatches the live recompute. The Host Service
   authenticates EVERY request against the file (token equality) AND
   against a freshly computed on-disk content hash
   (`authenticate_token_payload` + `_assert_active_token` →
   `find_skill().content_hash`), so in every rotation precondition the
   already-running companion's spawn-env token was ALREADY non-authorizing
   BEFORE the mint: missing/corrupt file fails the token compare;
   hash-stale file fails the "token is stale" check. Rotation therefore
   restores authorization for the publication's own spawns and can never
   revoke a still-valid token. Residual: recovery stages only MISSING
   companion names, and a supervisor auto-restart can re-spawn an old
   descriptor whose publication's start then reports success without
   replacing it — such a companion, de-authorized by a content change, is
   not healed by companion recovery (same dead-token state as before the
   recovery); the heal path is the ordinary unload/reload, which stops and
   re-spawns every companion with the fresh env.
3. Docs truth: with item 1 landed, the absolute claims are now factual and
   were tightened rather than weakened —
   `ExtensionStaleRecoveryError`'s docstring adds "no settings read/lock or
   state directory is materialized", `_publish_registrations` describes the
   whole-env post-fence materialization, the ARCHITECTURE
   extension_plugin_api row and the ADOPTION ABI-9 row carry the
   fix-round-6 clause plus the item-2 residual disclosure. The round-5
   ledger section above stays as written (append-only historical record);
   its "descriptor build stays pure" wording described the token plane only
   and is superseded by this section for the settings/state-dir planes.
4. Size pins: extension_plugin_api.py 982 and extension_loader.py 1000
   (<=1000 extraction pin, 600 <= plugin API respected); the env
   materialization helper and the non-creating path resolver live in
   extension_child_catalog.py (222), `_StagedCompanionSpawn.spec` in
   extension_registry_state.py (182).

## From the F3.1 conformance fix-round-7 (base 267b71bf, 2026-09-01) — FINAL

Round-7 verdict (GPT-5.6 Sol, read-only @ 267b71bf): NEEDS FIXES — ONE MEDIUM
(the round-6 zero-filesystem-effects claim is still absolute while the
liveness/grant projection legitimately reads settings/state pre-fence), ONE
LOW (a transient hash error rotates a live valid token), ONE LOW (pin blind
zones); verification points 2/3/6/8 CLOSED. This is the FINAL micro-round of
the Ф3.1 conformance cycle; the cycle is declared converged after it.
Dispositions:

1. MEDIUM — pre-fence filesystem effects: SCOPED AS CLAIMS, deliberately NOT
   fixed by rewriting the read layer. The pre-fence reads the verdict names —
   `health_path` → creating `skill_state_dir` (extension_health.py /
   skill_loader.py), the creating `skill_state_dir` inside
   `load_skill_grants`, and `config.load_settings` (settings lock, possible
   context-mode migration persist) inside `requested_core_setting_keys` — are
   the RUNTIME-WIDE settings/grant read idiom, used identically by status
   projections, the loader, skill exec and the UI; recovery merely calls the
   same projections every other caller uses. Carving a non-creating,
   non-locking read path through `skill_loader`/`config` for one caller would
   fork the SSOT read layer (over-engineering for a refusal path whose reads
   are idempotent infrastructure). Instead the false absolutes were removed:
   `ExtensionStaleRecoveryError`'s docstring and the ADOPTION ABI-9 row now
   state the exact contract — before the fence there are NO effects on
   authorization/token/registries/bundles/companion-env; infrastructure
   reads (settings lock inside `load_settings`, state-dir mkdir via the
   grant/health projection) MAY occur, as anywhere in the runtime. The
   round-6 sections above stay as written (append-only historical record);
   their "zero effects hold on every filesystem plane" wording is superseded
   by this scoped contract.
2. LOW — transient hash error rotates a valid token: FIXED in
   `mint_skill_token`. "Could not read/compute the hash" (transient) is now
   DISTINCT from "read fine and mismatched / token file missing or corrupt"
   (legitimate mint/rotate): on a transient `compute_content_hash` failure
   the stored token, when it parses, is returned byte-for-byte unrotated
   (`auth_token.json` untouched — the running companion whose spawn env
   holds it stays authorized against the file the Host Service rereads per
   request); with no reusable stored token the mint fails closed with the
   typed `SkillTokenHashUnavailableError` instead of minting a token bound
   to an empty hash. Pins:
   test_transient_hash_error_never_rotates_a_valid_token (red pre-fix: the
   old code collapsed the error to content_hash="" and rotated) and
   test_transient_hash_error_without_reusable_token_fails_closed (red
   pre-fix structurally — the typed error class arrives with the fix).
   RESIDUAL DISCLOSED (pre-existing, NOT introduced by this cycle, not fixed
   here): concurrent mints (publication attach / `get_skill_token` /
   process-runner child env) are read-decide-write over `auth_token.json`
   with no shared lock or CAS — the last writer can supersede a token just
   returned to another caller.
3. LOW — pin blind zones: FIXED by hardening
   test_stale_recovery_with_env_from_settings_has_zero_filesystem_effects:
   (a) the tripwire now covers BOTH seams — `skill_exec.load_settings`
   (`_scrub_env`) and the direct `config.load_settings`
   (`requested_core_setting_keys` and any other caller) — and measures
   exactly the window "after the recovery's state-dir resolution, up to the
   fence" (the counter clears where the interleave snapshot is taken), so
   the legitimate pre-fence grant/liveness projection calls are not broken;
   (b) `_data_root_tree` snapshots name + size + mtime_ns (an in-place
   rewrite is now caught) and the settings lock file is asserted absent in
   the window, with `SETTINGS_PATH` repointed test-locally so the assertion
   cannot race a concurrent test process on the shared run-wide path;
   (c) `EXT_DEMO_VALUE` is actually SET in the settings fixture, granted
   (custom-secret key) and asserted DELIVERED in the materialized spawn env,
   alongside the manifest companion env overlay (`EXT_OVERLAY`) and the
   isolated-dep `PYTHONPATH` (a real `.ouroboros_env` site dir) — the
   "settings-derived value silently lost" hole is closed.
4. Size pins: extension_loader.py untouched at 1000/1000;
   extension_plugin_api.py 1000 (<=1000 extraction pin, 600 <= plugin API
   respected); extension_child_catalog.py untouched (222).

## From the F3.2 lane A (ResolvedModelTarget, base 3ba9f452)

ABI-4 consumer sweep per docs/v7next/DESIGN_RESOLVED_MODEL_TARGET.md
(greenfield §6-design; zero occurrences on base — NOT a transplant).

Seam inventory (rg over comma/at model-string parsing beside resolution
seams) and dispositions:

| seam | prior output | consumers | migration |
| --- | --- | --- | --- |
| `model_slots.get_fallback_models` (cross-model ladder) | `list[str]` | `loop_model_call._run_cross_model_fallback_chain`; `tools/control_runtime` membership check | `provider_models.fallback_candidate_targets()` → `tuple[ResolvedModelTarget, ...]` (typed view over the ONE chain SSOT); the loop chain iterates typed candidates, `.model_id` crosses to a string only at the chat-API transport boundary |
| `review_model_routes.get_review_models` / `get_scope_review_models` | `list[str]` | reviewer slot builders (`reviewer_slots`, `structured_scope_review_slots`), review surfaces | typed views `get_review_targets` / `get_scope_review_targets` / `resolved_review_model_target` at the SAME seam; `review_model_uses_local` is applied ONCE at construction (`provider_route == "local"` ⇔ the predicate), and the slot builders read that fact off the dataclass instead of re-asking per string. MODELS UNCHANGED: purely typization, byte-identical per configuration class (structured, default panel, local-only route, exclusive-direct-provider rewrite) |
| `subagents.parse_subagent_harness` → `DelegationRoute` | typed route (already constructed at the parse seam) | `tools/delegate` run-request assembly | `DelegationRoute.resolved_model_target()` bridge; `_build_delegated_run_request` assembles harness pin, model, effort and credential pin from ONE typed target read |

Contract facts: frozen+slots, value equality/hash, ""/0 sentinels (no
Optional/None-vs-missing), NO pricing fields; `context_window` stays 0 at
these seams (windows remain Capability Evidence's fact, fail-open).
Home: `model_slots.py` (dataclass) + `provider_models.resolve_model_target`
(constructor) — the D02-owner seam; `config.py` facade re-exports every new
name (`test_config_extraction` owner inventory extended accordingly).

Verification hook: tests/test_resolved_model_target.py (name fixed by the
design note) — frozen-ness, value identity, construction at each seam,
consumer-sweep grep pins (no comma/at parsing in the swept consumers).
ADOPTION ABI-4 row: hook updated, status done.

DISCLOSED RESIDUAL (typed up to the transport boundary, per the lane note —
transports NOT rebuilt):

1. The Claudexor wire body serializes the typed target back to strings
   (`model`/`effort`/`credentialProfileId`/`harnesses`) — the engine's JSON
   contract; the adapter no longer re-parses a `harness[=model]` slug, the
   parse seam (`parse_subagent_harness`) remains the one string reader.
2. Reviewer agent_session rows keep their OPAQUE `harness[=model]` spec
   (RouteSpec/ReviewSlot vocabulary): a session spec is not an API model
   destination, so it is not forced into `ResolvedModelTarget`; the shared
   dataclass covers API-routed model destinations plus the delegated bridge.
3. `tools/control_runtime.py` still consumes `get_fallback_models()` as an id
   membership check (no parsing, no route facts) — left on the string list
   deliberately; `resolve_credentialed_model`/`vision` candidate walks are
   internal to the provider seam itself.
4. `ReviewSlot`/`commit_triad_delivery` keep their existing parallel-vector
   ABI (models/routes/efforts) — the sweep types the route-fact derivation,
   not the review delivery contract (review models/behaviour byte-identical).

## From the F3.2 lane B (dispatch digest read, base 3ba9f452)

1. The digest-API inventory the seam relied on is EXACTLY as the F3.1-B
   ledger promised — nothing new was built (reuse-first proven): the
   per-surface `extension_generation` stamp minted at publication
   (extension_plugin_api.py `_publish_registrations`) already reaches the
   dispatcher through `extension_loader.get_tool()`'s descriptor copy, and
   `extension_registry_state.extension_generation_digest` (re-exported by
   extension_loader) already serves the registry read. The Ф3.2 delta is
   confined to ouroboros/tools/extension_dispatch.py:
   `_dispatch_extension_tool_result` became a stamping wrapper over the
   verbatim inner dispatcher (`_dispatch_extension_tool_untagged`) plus the
   `_generation_digest_for` reader (descriptor stamp first, registry reader
   fallback for a descriptor predating the stamp).
2. Provenance seam REUSED, no new ledger: the digest rides the typed
   `ToolResult.meta` (`extension_generation` key), which the loop already
   projects into the tools.jsonl record via `_tool_result_fields` →
   `tool_result_meta` (loop_tool_execution.py). `ToolResult` is frozen with
   MappingProxyType meta, so the wrapper REBUILDS the result with the same
   status/code/text — the model-facing projection is byte-identical.
3. Scope of the stamp, disclosed: only outcomes of a PHYSICAL dispatch
   attempt carry the digest. The two pre-dispatch typed refusals —
   EXTENSION_UNAVAILABLE (liveness) and SAFETY_VIOLATION (safety block) —
   are excluded by `_UNDISPATCHED_CODES` and keep their exact pre-seam
   shape (pin: test_unavailable_refusal_keeps_the_pre_seam_typed_shape
   asserts meta == {"dynamic_provider": True}). No validation, no gate, no
   behavior branch reads the digest (P5: a provenance fact, not a gate).
4. Pins (tests/test_extension_registration_atomicity.py), red-proof done by
   running them against the base dispatcher:
   test_dispatch_provenance_carries_the_published_generation_digest (red
   pre-fix; also proves a reload's NEW publication puts the NEW digest on
   the next call) and
   test_dispatch_provenance_falls_back_to_the_registry_reader (red
   pre-fix); the unavailable-shape pin is an invariance pin (green on both
   sides by design).
5. Size pins: extension_loader.py and extension_plugin_api.py untouched at
   1000/1000; extension_child_catalog.py untouched (222);
   extension_registry_state.py untouched (182).

## From the F3.3 RC auditor (base 4fa2f01a)

Sources of each check class of the machine-readable scope
(scripts/rc_audit.py, ABI-7b/F13), file:line of the feeder inventory at this
base, consumed READ-ONLY (reuse-first — no parallel parsers or lists):

1. gateway-alias (5 checks) — the frozen F11 per-alias inventory
   docs/v7next/ABI3_GATEWAY_ALIAS_INVENTORY.md (cost_usd/cost_usd_with_children
   §1–2 at :16, telegram_chat_id §3 at :53, project_last_viewed/project_hidden
   §4–5 at :76). Stored-axis tolerance kept per the inventory, so on-disk hits
   (task_results alias keys, state/ui_preferences.json legacy keys) render as
   NOTES, never blocking findings; the live-client half is owner attestation.
2. retired-setting — ouroboros/settings_defaults.py::RETIRED_SETTING_KEYS
   (:314), imported at execution time; the ABI-5/Q10 semantics per
   tests/test_abi5_q10_removals.py. fail_tasks (no install-visible key) and
   until_deadline/stall_rounds_threshold (pacing knobs, not settings keys)
   live in the report PROSE plane exactly as the design note requires.
3. comma-list — ouroboros/settings_defaults.py::RETIRED_COMMA_LIST_SETTING_KEYS
   (:350, NEW in this commit): the ABI-10 classification INSIDE
   RETIRED_SETTING_KEYS, placed at the retirement SSOT so the auditor snaps
   the exact list at execution time instead of hardcoding it; subset
   membership is fail-closed in build_scope() and pinned by the suite.
4. plugin-api — ouroboros/contracts/plugin_api.py: PLUGIN_API_VERSION="2.0"
   (:29), LEGACY_PLUGIN_API_GENERATION="1.3" (:32, absent ≡ 1.3 by
   construction), extension_new_pass_admission_error (:285) reused as THE
   admission predicate; hash-bound grandfather adjudicated via
   skill_review_status.skill_review_gate over the install's
   state/skills/<name>/review.json (read without the creating
   skill_state_dir helper — read-only guarantee).
5. schema-stamp — ouroboros/task_result_schema.py:
   TASK_RESULT_SCHEMA_VERSION=1 (:34) and the PURE classifier
   task_result_schema_refusal (:39) reused directly (never
   load_task_result, which quarantines on read — a mutation). The Q8=B
   consequence is named verbatim in the scope check, in every schema-stamp
   finding, and in the owner-attestation list.

N−1 fixture catalog (F14, real bytes): tests/fixtures/nminus1/ —
settings_v6.113.4.json and task_result_v6.113.4.json were produced by RUNNING
the v6.113.4 code itself (git archive of the tag; config.save_settings /
task_results.write_task_result in an isolated mktemp root; all secret fields
empty), telegram_SKILL_v6.113.4.md is `git show f0313064:skills/telegram/SKILL.md`
(the commit before ABI-1 added the plugin_api field). The inline N−1 byte
forms of the ABI-2 quarantine suite and the ABI-7a updater shim remain where
they are; this catalog is the one FILE-shaped N−1 store (no second catalog).

Remaining owner attestation (F13 — printed by the auditor, never pretended
machine-checked): live custom gateway clients (send/read of the five removed
aliases), external automation treating the retired comma-list env spellings
as a settings surface, out-of-tree extension authors declaring plugin_api
"2.0" before new PASSes, reliance on fail_tasks / the removed pacing knobs,
and owner acceptance of the Q8=B quarantine consequence.

## From the F3.3 comma-sweep (base 1bd342b1, 2026-09-01)

1. The phase CI gate landed as tests/test_comma_list_remnant_sweep.py
   (named in the ABI-10 hook column beside the F3.1 sweep): retired-key
   mentions over ouroboros/ + web/ + supervisor/ snapped DYNAMICALLY from
   RETIRED_COMMA_LIST_SETTING_KEYS, comma-split parsing in model/review
   modules, phase-5 plumbing absence, and a retired-envs-are-ignored
   runtime pin - all count-anchored per site with reasons (the
   test_gateway_abi3_removals.py allowlist discipline).
2. Residual inventory matrix (remnant | live/dead | action | reason):
   - review_substrate.scope_reviewer_slots route_env_key plumbing
     (configured_review_routes + TRIAD/SCOPE_REVIEW_ROUTES_ENV in
     review_execution.py) | DEAD post-ABI-10 | REMOVED | the only
     production explicit-models caller (tools/scope_review.py:345)
     overrides the route itself ("the caller's fanned-out route is
     authoritative"); the models=None path reaches the env read only on a
     structured-config-less install exporting a RETIRED spelling - exactly
     the class ABI-10 retired. Rows built from plain model lists are now
     pinned api_chat; retired-envs-are-ignored pinned in the sweep and in
     test_review_agent_session_route.py (the phase-5 env-parsing clauses
     were replaced, not deleted silently: test_configured_review_routes_
     parsing / test_scope_rows_carry_their_configured_routes asserted the
     retired behavior). test_review_session_scope_wiring's mixed fan-out
     now builds its mixed panel from a structured OUROBOROS_REVIEWER_SLOTS
     payload.
   - OUROBOROS_ADVISORY_REVIEW_ROUTE (preflight_review_run.py constant +
     prose, claude_advisory_review.py messages) | DEAD (no os.environ read
     anywhere; advisory_review_route() reads only the structured SSOT) |
     prose/constant REMOVED | operator guidance now names the
     OUROBOROS_REVIEWER_SLOTS advisory row; vestigial setenv/delenv lines
     dropped from 4 test files (they were no-ops - nothing read the env).
   - JS typedef cleanup (web/modules/api_types.js) | stale | REMOVED | the
     8 HOT-DEFERRED JSDoc lines from the ABI-3 inventory (ChatOutbound
     cost_usd/cost_usd_with_children/telegram_chat_id, Photo/Video/
     DocumentOutbound telegram_chat_id, UiPreferencesResponse
     project_last_viewed/project_hidden); the _abi3_deferred_js_extras
     excuse set in tests/test_gateway_parity.py deleted - the browser
     mirror is exact again. api_types.js only shrank (ratchet-safe);
     chat.js untouched (BYTE_DEBT ceiling). node --test: 695/695 pass.
   - GATEWAY_CONTRACT_VERSION carrier switch (api_types.js '6.113.4' ->
     mirror of gateway.schema.GATEWAY_ABI_VERSION '7.0') | DEFERRED to the
     release tact | NOT executed | it rewrites a release version carrier
     and the test_gateway_parity pin that ties the JS constant to the
     VERSION file - version carriers move synchronously in release
     mechanics only.
   - Allowlisted-with-reason remnants (live, NOT removed): settings_
     defaults retirement SSOT; the derived env plane (reviewer_slot_config
     projection + review_model_routes/scope_review_budget readers);
     server_runtime raw-dict retired-model default refresh; provider_
     models declared-model surface; gateway/settings changed-key warning
     triggers; retirement prose in review_execution/preflight_review_run/
     web settings.js.
3. Same-commit collateral: ARCHITECTURE.md review-substrate paragraph now
   states per-row delivery comes from the structured OUROBOROS_REVIEWER_
   SLOTS rows (retired phase-5 envs ignored); plan_review's
   "Set OUROBOROS_REVIEW_MODELS in settings" operator error - a retired
   settings key - now points at Review lanes / OUROBOROS_REVIEWER_SLOTS;
   test_review_owner_facades' facade roster dropped the removed
   ADVISORY_REVIEW_ROUTE_ENV re-export.

## From the F3 adversarial fix-round (base 71e1f13f, 2026-09-01)

Disposition of the 14 findings of the F3.2+F3.3 adversarial wave (sol),
verified against the code before fixing; every fix landed in the four
single-intent commits of this round.

1. FIXED (HIGH, rollback fail-open). `rollback_managed_update` read the
   marker with the permissive `read_update_tx()`, so a FUTURE-schema tx was
   interpreted, re-phased and destructively reset. It now reads
   `read_update_tx_strict()` and refuses typed on `future` BEFORE any marker
   write or reset/checkout/clean (marker byte-identical, worktree and dirty
   local work untouched); a corrupt-stamp marker now refuses on the empty tx
   instead of being interpreted (the permissive reader returned the raw dict
   for a non-integer stamp). RELEASE_INVARIANT surface
   supervisor/update_merge.py: the delta is sanctioned by this fix-round and
   minimal (one strict read + one typed refusal). Pin: the future-schema
   suite now drives the direct rollback entry point (red pre-fix).
2. FIXED (HIGH, false grandfather PASS). rc_audit now looks review state up
   by the skill DIRECTORY basename (the runtime identity,
   skill_loader.load_skill) and verifies the stored PASS hash against the
   runtime's own `compute_content_hash` over the CURRENT payload bytes
   (read-only reuse); a mismatch is an INCOMPATIBLE stale review. Fixtures
   store the real computed hash; the "a"*64 form is now the stale-hash red
   fixture; a basename-vs-manifest.name divergence pin proves the lookup key.
3. FIXED (MEDIUM, exit contract). Chosen and documented in the module
   docstring: an unreadable/unparseable MANDATORY source (manifest, payload
   under a hash-bound PASS) is a BLOCKING `unauditable-source` finding
   (exit 1, its own check id, outside the five scope classes — an
   audit-integrity plane, not an ABI class); traversal OSError and
   report-write OSError map to exit 2, so a bare Python exit 1 can no longer
   read as "incompatibilities found". Pins on both planes.
4. FIXED WITH AN ADAPTED PIN (MEDIUM, bytecode). `sys.dont_write_bytecode =
   True` is set before any runtime import. The requested "prefix in audited
   root -> tree untouched" pin is PHYSICALLY unreachable for the naked
   launcher mode: the interpreter writes ~40 stdlib .pyc files under
   PYTHONPYCACHEPREFIX during startup, BEFORE the script's first line
   (measured on this host). The landed contract, both sides pinned: a prefix
   inside the audited root without startup bytecode suppression is REFUSED
   loudly (exit 2, the guarantee was already violated by the invoking
   environment); with PYTHONDONTWRITEBYTECODE=1 (or -B) the audit runs and
   the audited tree stays byte-for-byte identical.
5. FIXED (MEDIUM, inventory parity). `_iter_skill_dirs` now yields
   `skill_loader._walk_skill_packages(data_root/"skills")` — the runtime's
   own discovery (hidden excluded, `.replaced-`/`.staging-`/`.tmp-` orphans
   excluded, descent stops at a found package), read-only reuse instead of a
   parallel-rules mirror. Pin: orphan/hidden dirs with broken manifests in an
   otherwise clean install stay exit 0.
6. FIXED (MEDIUM, provenance). `sources.tree` appends `-dirty` when
   `git status --porcelain --untracked-files=no` is non-empty (tracked
   scope: untracked files supply no resolved classifier bytes), and
   REPO_ROOT is moved to the FRONT of sys.path (an earlier checkout later in
   PYTHONPATH could otherwise supply the classifiers). Pins: monkeypatched
   dirty/clean `_tree_sha`, sys.path[0] identity.
7. FIXED AS DISCLOSED HONESTY, BEHAVIOR BYTE-IDENTICAL (MEDIUM, ignored
   route). PROVEN OLD BEHAVIOR FIRST: on base 3ba9f452 the fallback loop
   iterated `get_fallback_models` strings and the dispatch lane was the one
   global USE_LOCAL_FALLBACK env flag — so per-candidate dispatch would be a
   BEHAVIOR CHANGE the sweep's byte-identical contract forbids (consuming
   `provider_route == "local"` per candidate would flip dispatch for a
   `"(local)"-suffixed model with the flag unset). Disposition branch taken:
   the ladder no longer fabricates the unconsumed fact —
   `fallback_candidate_targets` leaves `provider_route` the "" sentinel,
   docstrings on both sides and the loop comment state the lane contract,
   and an equivalence pin fixes the loop's global-flag read. The finding's
   mixed-ladder scenarios are therefore the PRE-EXISTING chain semantics,
   disclosed rather than silently re-engineered inside a typing sweep.
8. FIXED (MEDIUM, ABI-4 over-claim). ADOPTION row truth-scoped: typed
   consumers are the fallback ladder, the reviewer slot builders
   (`resolved_review_model_target` — production consumers in
   reviewer_slot_config) and the delegated lane; `get_review_targets`/
   `get_scope_review_targets` are marked typed views WITHOUT production
   consumers in their docstrings (wiring a whole-list consumer is
   review-surface work, not byte-identical); NAMED RESIDUAL:
   plan_review_runtime, review_multi_model and the reviewer parallel vectors
   keep their string ABI — their migration was NOT performed and review
   surfaces were not touched.
9. FIXED (MEDIUM, digest not in tools.jsonl). The DIRECT tools.jsonl record
   now carries `tool_result_meta` (bounded by the ToolResult contract:
   <=32 producer keys, <=8KB, JSON-safe — no secret plane), so the ABI-9
   generation digest survives a failed `persist_call`, exactly as the
   ADOPTION/ledger claims read. Consumers (memory summarizer, /api/logs
   tail) read named fields from JSON lines — the key is additive. Pin: the
   tools.jsonl row of a physical extension call carries
   `extension_generation` with persist_call forced to fail.
10. FIXED (MEDIUM, pre-handler stamp). The stamp now keys on a POSITIVE
    `physical_dispatch` meta fact set only when the handler / child process
    is actually invoked, replacing the `_UNDISPATCHED_CODES` exclusion list;
    the calling-convention resolution moved into its own pre-handler try.
    Pins on all three pre-handler EXTENSION_ERROR paths (runner import,
    disclosure gate, calling convention) plus a contrast pin that a genuine
    handler exception with the same code IS stamped.
11. FIXED (LOW, fallback-reader race). The registry digest is snapshotted
    BEFORE the handler call. Deterministic barrier pin: the handler itself
    republishes the extension mid-call; the result carries the pre-call
    digest while the live digest has moved on.
12. FIXED (MEDIUM, JS-parity hole). DocumentOutbound and
    UiPreferencesResponse joined the exact field loop in
    test_gateway_parity, plus explicit resurrection pins on
    cost_usd/cost_usd_with_children/telegram_chat_id/project_last_viewed/
    project_hidden in BOTH mirrors; the stale ABI-3 "JS mirror switch
    deferred / frozen excused set" ADOPTION claim replaced with the done
    state (the cleanup landed in the F3.3 comma-sweep tact).
13. FIXED (LOW, comma-gate evasion). The model/review comma-split scan is
    AST-level for Python (Attribute call `split`/`rsplit`, first positional
    or `sep=` keyword constant ",", any spacing/quotes/maxsplit) with a
    detector self-test on the evasion spellings; non-Python mirrors keep a
    hardened textual scan (no Python AST exists for them).
14. FIXED (LOW, shrink-only). The DelegationRoute typed-target bridge
    relocated to `provider_models.delegated_route_target` (the resolution-
    seam owner, headroom); ouroboros/subagents.py is back at 1380 lines and
    ouroboros/tools/delegate.py at 1263 — their 3ba9f452 base sizes.
    Monotonicity ENFORCEMENT deliberately not built (out of scope per the
    fix-round brief).

## From the F3 adversarial fix-round 2 (base d1d131df, 2026-09-01)

Disposition of the 8 defects the second adversarial wave (sol) left OPEN
against the round-1 dispositions above; every fix landed in the
single-intent commits of this round.

1. FIXED (HIGH, null stamp read as legacy). `read_update_tx_strict` used a
   plain `.get()`, so an explicit `_schema_version: null` was
   indistinguishable from the accepted pre-7.0 UNSTAMPED form and read
   `valid` — rollback would interpret and destructively act on a damaged
   stamp. A dict-get sentinel now distinguishes key ABSENCE (legacy, valid)
   from a stored `null` (corrupt, like every other non-integer stamp; no
   writer ever stamps null). Pin: null-stamped marker reads `("corrupt",
   {})` and the direct rollback entry point refuses typed — marker
   byte-identical, HEAD unmoved, dirty owner work untouched; `None` joined
   the invalid-stamp loop. Protected update_merge.py delta sanctioned by
   this round and minimal (sentinel + docstring).
2. FIXED (HIGH, admission-state divergence). The auditor's grandfather
   judgment trusted raw stored status/content_hash while the runtime admits
   through `load_review_state` (findings re-aggregation + provenance
   preconditions: official_hub sidecar, native_seed `.seed-origin`,
   owner_attested marker). `_review_gate_for` now calls `load_review_state`
   itself, with the runtime's identity inputs (directory basename, manifest
   type, module-widget shape, skill_dir). Mutation-free reuse:
   `load_review_state` now resolves state paths through the new
   NON-CREATING `skill_state_dir_path` (writers keep the created-on-demand
   `skill_state_dir`). Pin: native_seed PASS without `.seed-origin` →
   INCOMPATIBLE, with the marker (native bucket, hash-exempt) →
   grandfather note; both audits leave the install byte-identical.
3. FIXED (MEDIUM, both planes). (a) The audit walk no longer stands on the
   runtime's fail-soft `_safe_listdir`: `_walk_skill_packages` accepts an
   optional traversal reader (runtime default unchanged) and the auditor
   passes a strict lister whose OSError propagates to the existing exit-2
   traversal handler — an unreadable skills subtree can no longer audit
   clean. (b) `args.json.resolve()` moved under the OSError handler → exit
   2 (REPORT UNWRITABLE), never Python's bare exit 1. Pins on both.
4. FIXED (MEDIUM, fail-open provenance). `_tree_sha` with rev-parse OK but
   `git status` failing/erroring returned the bare SHA as if proven clean.
   Chosen and documented: the suffix `-unknown-dirty-state` (over the
   conservative bare `-dirty`, so an auditor can tell proven-dirty from
   could-not-check); only a zero-exit empty porcelain yields the bare SHA.
   Pins: status exit 128 and status OSError.
5. FIXED (MEDIUM, pre-Popen stamp). The OOP branch stamped
   `physical_dispatch` on EVERY exception from
   `dispatch_extension_tool_subprocess`, though resolve/load/env/staging and
   Popen itself fail BEFORE any child exists. Typed mechanism (not
   text-guessing): `_run_child` stamps a positive child-spawned marker onto
   every exception crossing the spawn boundary (both the on_spawn
   disclosure path and the drain/poll/result protocol path);
   `extension_child_was_spawned(exc)` reads it and the dispatcher's OOP
   error arm keys `dispatched` on it. Pins: pre-spawn failure → no stamp;
   marked post-spawn failure → stamp + digest; unit seam — Popen OSError
   raises unmarked, post-spawn protocol failure raises marked.
6. FIXED (LOW, pre-snapshot gap). Descriptor and legacy-fallback digest are
   now read under ONE lock hold: new combined reader
   `get_tool_with_generation` in `extension_registry_state` (the lock's
   home — extension_loader/plugin_api stay at their size caps); the
   dispatch candidate pre-stamps the snapshot digest onto the detached
   copy, so the separate registry fallback no longer runs on the live path.
   Pin: republish AFTER the descriptor is taken moves the live digest on
   while the dispatch result still names the snapshot generation.
7. FIXED (LOW, detector evasion). The comma-split AST detector now seeks
   the separator in the first TWO positionals and every `sep=` keyword, so
   the unbound forms `str.split(raw, sep=",")` / `str.split(raw, ",")` no
   longer evade; self-test extended with the unbound evasions and the
   `str.split(raw)` negative. Live allowlist counts unchanged.
8. FIXED (LOW, stale references). docs/ARCHITECTURE.md line 80 now names
   the landed bridge `provider_models.delegated_route_target` instead of
   the removed `DelegationRoute.resolved_model_target`. CORRECTION to the
   ABI-4 seam-inventory table above (the `subagents.parse_subagent_harness
   → DelegationRoute` row, written before the round-1 relocation): its
   migration cell reads `DelegationRoute.resolved_model_target()` — the
   landed form is the `provider_models.delegated_route_target(route)`
   bridge (round-1 disposition 14). The table itself stays as written:
   this ledger is append-only, corrections supersede in place of edits.

## From the F3 adversarial fix-round 3 (base 4f894191, 2026-09-01)

Disposition of the 5 defects the third adversarial wave (sol) left OPEN
against the round-2 dispositions above; every fix landed in the
single-intent commits of this round, and every pin was proven RED against
the base implementation before the fix.

1. FIXED (HIGH, grandfather-predicate divergence). The auditor keyed the
   grandfather on `skill_review_gate`'s `executable_review`, which under
   the DEFAULT advisory enforcement admits a BLOCKERS verdict — while the
   real PluginAPI grandfather (`plugin_api_admission_refusal_outcome`)
   accepts only clean|warnings under every enforcement mode. The predicate
   is now literally shared: new `review_status_grandfatherable` in
   `skill_review_status` (clean|warnings only, enforcement-independent) is
   consumed by BOTH the refusal path and the auditor's
   `_admission_state_for`; `skill_review_gate` left the auditor entirely.
   Pin: a hash-matching PASS carrying a critical blocker finding, audited
   with `OUROBOROS_REVIEW_ENFORCEMENT=advisory`, reports plugin-api
   INCOMPATIBLE and never the grandfather note.
2. FIXED (MEDIUM, audit identity ≠ runtime identity). The auditor bound
   review state to the LEXICAL walk name while the runtime resolves the
   directory first and derives state/tool identity from the sanitized
   RESOLVED basename (`load_skill`), refusing identity collisions before
   any review-state read. `_resolved_skill_identities` now mirrors that:
   `skill_dir.resolve()` (failure → blocking unauditable-source finding),
   dedup on the resolved path like the runtime inventory, identity =
   `_sanitize_skill_name(resolved.name)`; two directories sanitising to
   one identity emit a blocking collision finding and never reach
   `load_review_state`. Pins: a symlinked skill grandfathers only on the
   TARGET-basename state (link-name state → INCOMPATIBLE); a collision
   pair yields the blocking finding and no plugin-api judgment.
3. FIXED (MEDIUM, mandatory-source traversal class). `task_results`
   listing stood on fail-soft `Path.glob`, which on supported Python 3.10
   suppresses PermissionError — an unreadable directory audited clean.
   New `_strict_json_files` (same direct-child `*.json` selection, OSError
   raises to the exit-2 handler); a per-file read OSError now also
   propagates (exit 2) instead of masquerading as a "malformed →
   quarantine" verdict, and the `ui_preferences` reader keeps tolerating
   content damage but no longer swallows read OSError. The class sweep:
   settings already raised `InstallUnreadable` (exit 2), skills already
   used the strict lister — task_results and ui_preferences were the
   remaining fail-soft members. Pin: chmod-0 `task_results` → exit 2
   ("audit traversal failed"), never exit 0.
4. FIXED (MEDIUM, resolve-error exits). `data_root.resolve()` and the
   pycache-prefix resolve ran outside any handler, and the report-path
   handler caught only OSError — a 3.10 pathlib symlink loop raises
   RuntimeError, giving Python's bare exit 1 ("incompatibilities found" to
   automation). All three resolve points now catch (OSError, RuntimeError)
   and map to exit 2 (INSTALL UNREADABLE / READ-ONLY GUARANTEE UNPROVABLE
   / REPORT UNWRITABLE). Pins: RuntimeError from the data-root resolve →
   exit 2; RuntimeError from the report-path resolve → exit 2.
5. FIXED (MEDIUM, spawn-marker gaps). (a) The process REGISTRATION between
   Popen and the protected block could raise unmarked: the whole post-Popen
   span (registration, on_spawn disclosure, protocol body) now lives in ONE
   try whose handler stamps every BaseException — the separate on_spawn
   cleanup arm collapsed into the shared finally (same kill/reap/unlink
   semantics). (b) A cleanup failure in that finally could REPLACE a marked
   in-flight exception with an unmarked one: the finally's own guard now
   stamps the replacing exception too (original stays chained as context).
   (c) `_mark_child_spawned` silently tolerated an unattachable marker: a
   weak side-table (`_spawned_marker_fallback`, consulted by
   `extension_child_was_spawned`) now records the fact for exceptions that
   refuse setattr; the only unmarkable residue (no attributes AND no
   weakref support) is logged, never dropped. Pins: registration exception
   → stamped; cleanup exception over a marked one → stamp preserved on
   both; a setattr-refusing exception → stamped via the side-table.

## From the F3 adversarial fix-round 4 (final, base 5187fcdc)

Disposition of the 3 defects the fourth adversarial wave (sol) raised
against the round-3 dispositions above — all MEDIUM, all fixed in this
round's single-intent commits, every pin proven RED against the base
implementation before the fix.

1. FIXED (MEDIUM, spawn-marker fallback not identity-safe). The round-3
   WeakSet side-table depended on the exception being HASHABLE: an
   unhashable exception raised TypeError on `add` (marker silently lost)
   AND on the membership check — and the check runs inside the
   dispatcher's except handler, so the secondary TypeError REPLACED the
   original tool error; equal-but-distinct exceptions could also borrow
   the marker through `__eq__`/`__hash__` (false `physical_dispatch`).
   The side-table is now keyed by `id()` with a weakref finalizer purging
   the entry (identity-safe, leak-free, no hashability requirement), and
   `extension_child_was_spawned` is FAIL-CLOSED: any failure of the
   marker read answers False — a physical call is never claimed on a
   broken check and the in-flight exception is never masked (a hostile
   `__getattr__` probe falls through to the side-table, which is exactly
   where the marker would live for an attribute-refusing object). Pins:
   unhashable post-spawn → stamped; unmarked unhashable → unstamped
   without raising; equal-but-distinct twin → no false positive;
   side-table entry dies with the exception; hostile exception at
   dispatch level → ORIGINAL error reported unstamped, no masking
   TypeError.
2. FIXED (MEDIUM, mandatory-source pre-checks fail-soft). The round-3
   strict listing/read only helped once the source was ENTERED:
   `Path.is_dir()`/`is_file()` fold ELOOP and dangling symlinks into
   plain False, so a symlink loop standing where `task_results` lives (or
   a broken `state/ui_preferences.json` link) skipped the whole source
   before the strict reader ran — a false-clean audit. New
   `_stat_mandatory_source` probes with strict `os.stat`: only TRUE
   absence (lstat agrees) is a legitimate skip; a loop or dangling link
   raises to the exit-2 handler. Pins: task_results symlink loop →
   exit 2; dangling ui_preferences symlink → exit 2; genuinely absent
   sources still audit clean (contrast).
3. FIXED (MEDIUM, resolve-error class incompletely closed). Round 3
   covered the auditor's OWN three resolve points, but
   `compute_content_hash` resolves manifest-DECLARED entry/script paths
   unguarded (`skill_loader._add_if_confined`): a symlink loop there
   raises RuntimeError on supported 3.10 — past the
   `SkillPayloadUnreadable` clause and the OSError-only top handler into
   Python's bare exit 1 with no report. Fixed on the AUDITOR side only
   (skill_loader untouched — runtime semantics unchanged): the
   compute_content_hash wrapper maps (OSError, RuntimeError) to the same
   blocking unauditable-source finding as an unreadable payload
   (per-skill scope, the rest of the install still gets audited —
   consistent with the neighbouring SkillPayloadUnreadable and
   skill-dir-resolve dispositions), and the top-level handler extends to
   (OSError, RuntimeError) → exit 2 as the class backstop for every
   other resolve the audit or its read-only runtime classifiers perform.
   Pins: loop in a declared entry → blocking finding, report written, no
   crash; RuntimeError from the audit walk → exit 2.

CONVERGENCE. Four adversarial waves over the F3 surface: finding profile
14 → 8 → 5 → 3, severity ceiling HIGH → HIGH → MEDIUM → MEDIUM, with
every HIGH exhausted by round 3 and round 4 consisting solely of
narrowing residues of already-dispositioned classes (side-table
completeness, pre-check strictness, one more resolve seam). The wave-4
verdict itself confirms fix claims 1–2 CLOSED and 3–5 OPEN only through
the three findings above — now fixed and pinned. Remaining review
surface is hygiene-grade; per the bounded-wave contract the adversarial
cycle is declared CONVERGED at this base.

## From the F5 lane A (CPL-1/2, base 5187fcdc)

1. CPL-1 LANDED. The Ф0 evidence manifest scripts/v7next_domains.toml is
   promoted to the production manifest `ouroboros/domains.toml` (ships as
   package data): module→domain 1:1 over all 488 tracked runtime modules,
   D01–D20, [classification].proposed carried unchanged (80 rows, still
   owner-review pending). New generated sections pin today's FACTUAL
   dependency data as baseline: `[graph].allowed` (164 strict cross-domain
   directions), `[graph].cycle_groups` (the single 20-domain strict-quotient
   SCC as the ceiling; target `[]`), `[graph].lazy_only` (92 hidden-coupling
   pairs), `[graph].dynamic_pairs` (empty), `[duplicates].allowed` (EMPTY —
   zero cross-domain literal-copy bodies ≥10 normalized lines exist today, so
   the literal-copy ban starts strict). Gate: scripts/check_domains.py
   (--write regenerates the generated sections + docs/DOMAIN_MAP.md); verify:
   tests/test_domain_manifest.py (completeness = red on drift, baseline
   exactness, DOMAIN_MAP byte-identity, synthetic red-branch pins for every
   detector). Shared core scripts/domain_graph.py extracted from the Ф0
   report generator; scripts/v7next_domain_report.py now consumes it and
   stays report-only.
2. Plan §7.1 "циклы=0 на domain-нодах" honesty note: the live strict quotient
   is ONE 20-domain SCC (921+ module-edge witnesses at Ф0, 164 domain pairs
   now), so a flat cycles=0 assert would be red on the campaign's own tree.
   Per the Ф5 baseline discipline (current reality = baseline, tightening =
   separate owner decisions) the gate pins the SCC as a data ceiling: growth
   (a new cycle group, a domain joining the SCC, a new direction) is red;
   shrinkage must be banked by regeneration; `cycle_groups = []` is the
   terminal state at which the gate becomes the literal cycles=0 check.
3. Ф0 report drift note: the Ф0 quotient report (generated at 1633b54f)
   counted 163 strict domain pairs; the tree at 5187fcdc has 164 — ordinary
   inter-phase drift, no population change (488 modules unchanged). The
   report was regenerated on the shared core alongside the manifest move.
4. CPL-2 LANDED as three gen/verify pairs (generator
   scripts/regenerate_inventories.py with --check; verify
   tests/test_generated_inventories.py; staleness = red):
   docs/v7next/FROZEN_CONTRACTS_INVENTORY.md (ARCHITECTURE §11.1 machine
   extraction — 20 rows, all owner/anchor paths resolve),
   docs/v7next/DATA_LAYOUT_INVENTORY.md (111 entries of the §1 Data-layout
   tree, all probed: 91 code-ref, 17 repo paths/dirs, 3 placeholders, 0
   unresolved), docs/v7next/FACADE_INVENTORY.md (49 facades, 2175 marked
   noqa:F401 re-export bindings, 121 cross-domain facade→leaf pairs).
5. PERSISTENCE_OWNERS carrier finding (CPL-2 spec said "найди фактический
   носитель"): the reference tree ouroboros_v7_wip @ 9f691656 carries
   docs/PERSISTENCE_OWNERS.md (hand-derived writer/reader/lifecycle rows) and
   docs/FACADE_CONSUMERS.md; NEITHER exists in this tree. The factual
   data-layout carrier here is the ARCHITECTURE §1 "Data layout
   (`~/Ouroboros/`)" tree, and the inventory generator binds to it. Porting
   the reference's full per-row writer/reader/lifecycle derivation is the
   CPL-4 persistence pass, not this lane.
6. §11.1 package-coverage finding, now pinned as data: two
   `ouroboros/contracts/` modules are documented in the §1 tree but have no
   §11.1 frozen-table row — contracts/task_constraint.py and
   contracts/skill_payload_policy.py. The inventory lists them as the exact
   known gap and the verify test pins the set, so a THIRD uncovered frozen-
   package module turns red even after regeneration; retiring the two-row gap
   itself (writing their §11.1 rows) is an owner-visible follow-up, not
   silently done here.

## From the F4 lane 1 (system_e2e skeleton, base 5187fcdc)

1. FIXED (HIGH, ABI-10 drift in the Ф0 harness). The keyless lane pinned the
   RETIRED comma-list reviewer keys (`OUROBOROS_REVIEW_MODELS` /
   `OUROBOROS_SCOPE_REVIEW_MODELS` / `OUROBOROS_SCOPE_REVIEW_MODEL`) in the
   isolated settings.json; `load_settings` drops retired keys, so the review
   organ fell back to the shipped OpenRouter default panel. Observed live on
   the base SHA: S2's triad ledger named gemini-3.7-flash/gpt-5.6-terra/
   claude-opus-5, keyless, and commit_reviewed blocked deterministically at
   scope-pack assembly ($0 spent; the keyless design failed CLOSED, which is
   why Ф0's smoke run never caught it — the failure only shows in the mock
   lane). `keyless_settings` now pins the STRUCTURED `OUROBOROS_REVIEWER_SLOTS`
   (three api_chat triad rows + one scope row onto the stub slug), and a
   default-lane pin feeds that value to the tree's own `parse_reviewer_slots`.
   S2 green again (67s solo).
2. EGRESS HARDENING FIRST (plan §8: the ANTHROPIC_API_KEY hole). Two layers on
   top of the Ф0 strip list: (a) a default-lane CLASS pin scans the runtime
   tree (`ouroboros/**`, `supervisor/**`, `server.py`) for every
   credential-shaped env key it actually reads (`os.environ[...]`/`.get`/
   `os.getenv`) and requires each to be covered by
   `STRIPPED_PROVIDER_ENV_KEYS` ∪ secret-shape sanitizer ∪
   `STALE_INHERITED_ENV_KEYS` — a provider credential added upstream fails by
   name; (b) scenario S3 boots a REAL server with poisoned fake credentials in
   the parent env, completes a scripted task keyless, and probes
   `/proc/<pid>/environ` of the WHOLE live process tree (server + workers):
   no planted or real credential VALUE (values, not names — a rename cannot
   hide one) and no stripped provider key NAME (stub pair exempt: the server
   legitimately projects the loopback-only pair from its settings) is visible
   to any child. Linux-only probe by construction, skipif elsewhere.
3. NEW SURFACES on the skeleton (plan §8 first wave): S1 extended with the
   WS-chat answer over the real `/ws` ingress (assistant reply frame + durable
   chat.jsonl row) and port-file honesty (`state/server_port` == the port the
   driver talks to); S4 — typed tools + safety: `write_file` onto
   `prompts/SAFETY.md` under runtime_mode=advanced answers the typed
   `CORE_PROTECTION_BLOCKED` refusal, the task still completes, and the clone
   fingerprint (HEAD + porcelain + exact bytes of SAFETY.md/BIBLE.md) is
   IDENTICAL before/after — zero side effects, not "still clean"; S5 —
   cost-truth (ABI-3) on a live server: task detail and list projections are
   deep-scanned for the retired `cost_usd[_with_children]` spellings (keys
   derived from `COST_ALIAS_PAIRS`, never literals) and must be clean, the
   honest `accounted_upper_bound_usd` is present at the detail top level, and
   the durable stored row is honest-only at the top level (internal evidence
   planes keep their own schemas by ABI-3 design, so the stored-row pin stays
   top-level).
4. ReplayModel landed per the plan matrix: deterministic fixtures bound by
   `(lineage, slot, attempt)` — slot = the wire's `model` field (scenarios pin
   distinct stub slugs per model slot), lineage = the LAST `[E2E-LINEAGE:*]`
   tag in the prompt text (default `root`), attempt = 1-based per-(lineage,
   slot) ordinal of fixture-consulted calls. Review-organ/safety calls are
   answered canned BEFORE the finalization check (shared `canned_review_answer`
   with the scripted stub — one HTTP base class, so wire shape and window
   evidence cannot drift between the two models) and never touch the fixture.
   A miss answers loudly (`REPLAY_MISS`, the server cannot hang) and
   `assert_consumed()` is red on ANY miss and ANY unconsumed row
   (недоеденная фикстура = красный). Default-lane pins cover binding, ordinal
   attempts, last-tag-wins, review-no-consume, and both red paths.
5. DEFERRED BY DESIGN (disclosed, plan §8 says later lanes): FakeClaudexorDaemon
   (delegated-transport wave) and PlaywrightUIClient (gateway/UI-truth wave)
   are interface STUBS in `tests/system_e2e/interfaces.py` that raise
   NotImplementedError naming their lane — a default-lane pin asserts they
   refuse instantiation, so nothing can silently pretend they exist.
6. LANE/MARKER DECISION: every scenario test carries `integration` AND `serial`
   markers PLUS the `OUROBOROS_E2E_DEEP=mock` env gate. Both CI pytest passes
   AND-exclude `integration`, the default local addopts excludes it, and the
   CI-shape battery's serial pass (`-m "serial and not integration ..."`)
   excludes it too — the suite cannot slow any existing lane. The manifest is
   data with a two-direction gen/verify pin (manifest row without a test = red;
   `test_s<N>_*` without a manifest row = red) plus a marker-discipline pin
   (scenario test without both markers = red). pyproject's `integration`
   marker description and DEVELOPMENT's marker-lanes section now name the
   keyless system_e2e lane; ARCHITECTURE gained the "System E2E suite"
   subsection (same commit as the structural surface).
7. Suite time (this host, mock lane, serial): 23 tests in ~142s wall — inside
   the plan's 10-25min PR keyless budget with room for the next scenario
   waves; the default-lane pins add ~2.4s to the ordinary non-serial battery.

## From the F5 lane B (CPL-4/7 + CPL-5 note, base 5187fcdc, 2026-09-01)

Lane deliverables (single-intent commits on this lane):

- CPL-7 (feat commit): skill-manifest `model_experience` prose section
  (`what_model_sees` / `token_effect`; bare string = shorthand), rendered on
  BOTH model-visible skill surfaces (`summarize_skills` rows → `list_skills`
  JSON; installed-skills context section, bounded), preserved across clawhub
  adaptation; teaching refusals — `SkillManifestError` gained a typed
  `fix_hint` rendered into the message and EVERY refusal site in
  `contracts/skill_manifest.py` now explains the repair. Bundled telegram +
  unix_computer_use manifests carry the section (their content hash moves —
  same owner-ratified class as the ABI-1 `plugin_api` field rollout).
  Pins in `tests/test_skill_model_experience.py`: with-section parses and
  reaches the surfaces; without-section behavior byte-identical; refusals
  teach. Disclosed residual: `ExtensionRegistrationError` (registration
  layer below the manifest) already carries prose guidance from ABI-1
  negotiation but was not converted to the typed fix_hint shape — separate
  seam, untouched here.
- CPL-5 (docs commit): `docs/v7next/DESIGN_MODEL_VISIBLE_LOGGED.md` — the
  F15-narrowed model-visible⟺logged contract (sealed model_send records at
  `llm_attempt._candidate_before_dispatch`, reconstruction + byte compare on
  call, typed durable mismatch facts, reverse-⟺ for model_send only, closed
  exclusion enum, divergence-class canonicalization contracts incl. the
  single-assembly rule for streaming and per-rung retry seals). Design only;
  implementation sketch names the next lane's seams. ADOPTION row CPL-5
  stays in-progress until the reconstruction suite lands.
- CPL-4 (docs+test commit): `docs/PERSISTENCE.md` — full durable data-plane
  inventory (≈60 entities; per-entity schema_version / migration / retention
  / reset decisions, all local, no framework) + verify pair
  `tests/test_persistence_inventory.py` (AST scan of every runtime
  data-path writer; 118 distinct normalized paths, count-anchored both
  directions, sentinel-guarded).

§16 disclosure: the plan's "§16 findings" (undocumented planes, unbounded
ledgers, mismatched temp) could not be recovered from the plan, the v7 spec,
the campaign scratchpads, or the roast archives — no §16 exists in any of
them. Per the lane instruction the inventory was built from scratch by
factual scan; the classes §16 named were independently re-derived and are
covered: undocumented planes (e.g. `state/consciousness_observations.jsonl`,
`state/betterleaks/`, reader-only `state/crash_report.json`, orphan
`state/project_source_locks/`), unbounded ledgers (table below), and
temp/cache planes (`state/pycache`, `state/python-userbase`, `.staging`,
`tmp_scripts` fallback).

### CPL-4 candidate code fixes (NOT touched in this lane — plan rule)

Real gaps where a decision is recorded in PERSISTENCE.md but the closing code
change belongs to a later lane / post-release backlog. Each is local; none
proposes a generic framework.

| id | entity | gap | proposed local fix |
|---|---|---|---|
| CPL4-C1 | logs/events.jsonl | unbounded, no rotation (100 MB WARN says "tracked as issue"); delegate custody replays it and the fault tail-scan reads last 4 MB | rotation with archive chain ONLY after custody readers (delegate_custody.replay, fault scan, legacy-usage import) become chain-aware; alternative: move custody rows to their own bounded store first |
| CPL4-C2 | logs/tools.jsonl | unbounded, no rotation (100 MB WARN) | reuse rotate_jsonl_log_if_needed on the supervisor tick; readers (recent-tools tail, ATIF auditors) are tail/chain-tolerant |
| CPL4-C3 | logs/supervisor.jsonl | unbounded AND no size tripwire at all | add hot-store tripwire row + same rotation |
| CPL4-C4 | logs/task_reflections.jsonl | unbounded, no tripwire; read is tail-20 | same rotation; project-scoped copies follow project retention |
| CPL4-C5 | logs/agent_stdout.log | launcher pipe-copy, unbounded, no cap | size-capped rotation in the launcher copy thread (keep N segments) |
| CPL4-C6 | state/usage_attempts.jsonl | unbounded monetary ledger; 20 MB WARN; full re-read under the monetary lock | seq-preserving compaction snapshot (settled rows folded into a stamped baseline row + archive of the raw segment) — needs its own reviewed design, monetary authority |
| CPL4-C7 | state/scheduled_tasks.json | consumed `once` receipts kept forever; `schema_version` defaulted on read, never authored on write | author the stamp in _write_scheduled_tasks; prune consumed-once receipts older than GC retention |
| CPL4-C8 | state/capability_evidence.json | TTL-expired entries never deleted; route-key growth unbounded | drop expired keys on write (same TTLs the reader applies) |
| CPL4-C9 | state/crash_report.json | reader-only orphan — the rollback writer no longer exists in this tree | either restore the writer on the rollback path or retire the reader + health line (owner decision: crash-rollback visibility) |
| CPL4-C10 | state/skills/<name>/ core files | review.json/grants.json/enabled.json/review_job.json/owner_attestation.json/accepted_rebuttals.json carry no version key (ABI-2 idiom exists) | stamp `_schema_version: 1` on write; readers keep legacy-0 tolerance |
| CPL4-C11 | state/skills/<name>/ lifetime | uninstall removes only deps.json; state dir + stale grants outlive the payload forever | tombstone-on-uninstall (keep grants as owner authority, mark payload-gone; GC only what the owner's uninstall names) |
| CPL4-C12 | state/skills/<name>/review_history.jsonl | unbounded per skill; load_history whole-file reads | bounded segment reads everywhere (find_history_job_bounded idiom), optional archive rotation per skill |
| CPL4-C13 | state/delegate_recovery*, delegate_supervision | one file per crashed/restarted task, never unlinked | terminal+age local sweep beside the existing startup custody sweep (fail-closed on unreadable custody, like _prune_delegated_snapshots) |
| CPL4-C14 | state/code_intel/<root-sha>/ | root-dir count unbounded; stale workspace roots never expire | age-prune roots whose inventory.json mtime exceeds GC retention (pure cache) |
| CPL4-C15 | state/extension_reconcile/failed/ | kept forever, never retried | age-prune failed markers past GC retention (events.jsonl already carries the failure) |
| CPL4-C16 | memory journals (identity_journal, knowledge_history, patterns_history) | full old+new document text per write → O(doc×edits) growth | keep full-text only for the newest N entries per journal, older entries digest-only — needs owner sign-off (cognitive provenance) |
| CPL4-C17 | knowledge_history.jsonl / knowledge_journal.jsonl appends | raw open("a") without the append sidecar lock → torn-line hazard | route through append_jsonl (same seam every other journal uses) |
| CPL4-C18 | memory/owner_mailbox/ | a task that dies off the terminal paths leaks its mailbox permanently | startup sweep: unlink mailboxes whose task is terminal per task_results (fail-closed when no result) |
| CPL4-C19 | task_results/<id>.json | one file per task forever (lifecycle authority) | accepted-unbounded for 7.0; any prune needs an owner decision first (authority precedent: archive/ never GC'd) — named here so the decision is visible, not silently open |
| CPL4-C20 | data/tmp_scripts fallback | never swept (task-drive copies are swept transitively) | include the fallback dir in sweep_stale_temp_files scope |
| CPL4-C21 | uploads/ (+screenshots, views) | no retention of any kind; owner-explicit delete only | accepted for owner attachments; screenshots/views (agent-generated) could follow GC retention — owner decision |
| CPL4-C22 | observability retention knob | OUROBOROS_OBSERVABILITY_RETENTION_DAYS is parsed, clamped, reported — and deletes nothing (preserve-indefinitely contract) | retire the knob or make it honest (documented no-op is misleading operator surface); the preserve contract itself is accepted |
| CPL4-C23 | state/consciousness_observations.jsonl | unbounded append (render bounded last-10) | ACKed rows older than GC retention fold into an archive segment; unacknowledged rows never pruned (contract: survive restart/overflow) |

Cross-lane note: CPL4-C1..C5 and C12 are one mechanism family (the existing
rotator + chain-aware readers) — a later lane should land them as one train,
not six bespoke rotators. C6, C9, C11, C16, C19, C21, C22 carry owner
decisions and must go to a batch before code.
## From the F6 rolling-upstream sync (upstream 8d13373b, base 5187fcdc, 2026-09-01)

One merge (`git merge 8d13373b`, merge-base b9f7597f: 121 upstream commits,
319 files, 93 overlapping) under the standing principle **upstream = semantic
truth, campaign = structural truth**. Decision map by conflict class:

- **Campaign-split monoliths** (loop.py, supervisor/events.py, tools/registry.py,
  tools/control.py, tools/core.py, tools/shell.py, extension_loader.py, config.py,
  llm.py, usage_accounting.py, delegate_custody.py, review_evidence.py,
  review_substrate.py, supervisor/queue.py, supervisor/git_ops.py, server.py,
  agent_task_pipeline.py, skill_review_status.py, subagent_dispatch_notes.py,
  scope_review.py, claude_advisory_review.py): OUR facade form kept; every
  upstream semantic delta re-seated in its owner leaf (retarget-to-owner).
  Notable seats: DESIGN.md governance doc -> scope_review_pack +
  preflight_review_prompt; EFFORT_SCALE `ultra` -> settings_scales +
  llm_capability_policy + control_runtime (whole-call effort validation);
  bounded prompt-token estimate -> llm_attempt + usage_accounting dataclass;
  cached ledger read -> usage_legacy_import; sweep_orphaned_budget_fences ->
  queue_snapshot restore seam; managed-update truthfulness (checked_at
  discipline, availability recompute, `_public_repo_url` credential strip,
  list_versions sha) -> git_ops_updates (+ facade re-export); custody
  cursor/backfill passes -> server_maintenance; R5 delegated-custody
  projection -> control_task_results; routing candidates reorder + durable
  attachment carrier -> control_routing; owner_delivery live-first send family
  -> control_runtime + core_artifacts; child-absorption hold + fresh-listing
  prompts -> loop_delivery/loop_forced_finalization/loop_budget; quiz-answer
  drain + extracted handle_finalize_now_entry -> loop_round_limits.
- **Double extractions — the upstream twin never lives**:
  - `acceptance_dialogue.py` (853 L) -> folded into `loop_acceptance.py`
    (REASON_* closed-set rows) and `loop_acceptance_review.py` (A-material
    paid identity family: `acceptance_paid_identity`,
    `bind_acceptance_paid_identity`, `_refuse_identical_acceptance`,
    `acceptance_dialogue_history`, superseded-replay `_prior_acceptance_run`,
    inconclusive/DEGRADED dialogue semantics in `_apply_task_acceptance_result`,
    paid-cycles timing disclosure, UNHASHED history key). Importers (loop.py
    re-export list, tests/test_acceptance_a_material.py) retargeted; file removed.
  - `delivery_protocol.py` (170 L) -> folded into `loop_delivery.py`
    (hold-control literals, RecursionError-degraded object parser,
    `_parse_delivery_control_body` over `strip_protocol_fence` +
    `extract_trailing_json_object`); file removed.
  - `supervisor/chat_delivery_events.py` (150 L) -> folded into
    `events_chat_delivery.py` (unified `_delivery_chat_id` incl. chat-0 media,
    `send_links`/`send_quiz`, EVENT_HANDLERS merged as `**_CDE`); test importers
    retargeted; file removed. `telemetry_events.py` has no campaign twin and
    lands as-is; the campaign copy of `_handle_review_wave_budget_insufficient`
    (events_budget) is retired for the typed telemetry registry, its
    extraction-test pin removed with it.
- **Rename classes accepted** (same behavior, public name): `_deadline_expired`
  -> `deadline_utils.deadline_expired` (tests/test_delegated_run_profile pins
  moved); `_contract_valid_actors` -> `review_actor_aggregation
  .contract_valid_actors` (review_verdict local copy deleted, re-export list +
  extraction pin updated); `record_python_resolution`/`python_interpreter` ->
  `process_interpreters.record_interpreter_resolution` + scoped
  `interpreter_attestation` (registry_core save/restore replaced; node
  post-gates seated as a registry_core helper under the 300-line function gate);
  `_describe_returncode` -> `process_facts.describe_returncode` (shell_process
  aliases it); `_write_verdict` -> `tools/patch_verdict.py` (upstream leaf
  as-is, subagent_integration re-exports).
- **Protected surfaces as-is**: BIBLE.md (+P7 DESIGN.md), safety.py (chat.links,
  node-runtime, escalate carve), docs/CHECKLISTS.md, gateway/contracts.py incl.
  the `endpoint_index.py` extraction (campaign's scope-review-floor endpoint
  removal re-applied there; upstream's `POST /api/decisions` kept), registry.py
  deltas seated into the campaign leaf structure.
- **size_ratchet_manifest**: union-resolved, then regenerated twice via
  `scripts/regenerate_size_ratchet.py` (band rationales recorded for
  loop_acceptance_review, loop_delivery, tests/test_delegate_answer;
  extension_plugin_api held at 1000 by moving the node argv policy beside its
  PATH-prepend half in extension_child_catalog); `--check` green. To keep
  tools/core.py under the 1600 giant gate the upstream-new link/quiz/escalate
  spans live in `core_artifacts.py` (the D05 owner-chat delivery leaf) with
  re-exports; `_run_shell` and the registry dispatch were shaved under the
  300-line function gate via shell_process/registry_core helpers.
- **Web/VERSION**: upstream web wave, vendor assets, VERSION 6.113.5 and
  package data landed as-is (node --test: 812/812).

Disclosed dispositions (contract-affecting):

- **Upstream R5 regex fallback vs the ratified D02 typed organ**: upstream kept
  `_EXIT_CODE_RE`/`_SIGNAL_RE` prose harvest as a fallback for untyped records
  and pinned it in tests/test_process_signal_observability. The campaign organ
  (owner-ratified D02) retired prose classification: process facts flow ONLY
  from typed publications (ToolResult meta + the new thread-local
  `process_facts` channel, which the loop merges with whole-family precedence).
  The fallback pin was REPLACED by a campaign-contract pin (untyped legacy
  prose forges no process facts); stale fallback comments corrected. This
  replaces a clause that would have re-opened stdout forgery of
  exit_code/signal.
- **Classification deltas A.23** (tests/test_tool_classification_differential):
  BROWSER_SESSION_RETIRED ok->timeout/error, BROWSER_BACKLOG_RETIRED_SESSIONS
  ok->unavailable/error (upstream #440 failure prefixes, typed here as exact
  identifier codes), ESCALATE_UNAVAILABLE error->unavailable (generic marker
  chain). Golden regenerated by the corpus recipe at 0f715831. The `A.23`
  numbering continues the existing owner-item sequence but has NOT been through
  an owner batch — see open fork Q-F6-2.
- **Cost-projection doc row**: upstream's ARCHITECTURE text still describes the
  deprecated `cost_usd[_with_children]` outbound aliases; the campaign's ABI 7.0
  (ABI-3) removal is owner-ratified and its text was kept. No code conflict —
  upstream never re-added the aliases.
- **test_v678_acceptance_state / test_owner_facing_honesty / test_loop_misc
  writer inventory**: upstream counts assumed acceptance_dialogue.py; pins now
  count the campaign leaves (19 writers incl. the A-material refusal;
  expression-valued reasons resolved through loop_acceptance/_review/outcomes).
- **ABI 7.0 test fixtures**: upstream-new tests (routing_decision,
  find_child_prefilter) wrote unstamped task-result rows, which campaign
  readers QUARANTINE; fixtures now stamp `_schema_version` like real writers.

Domain map (`scripts/v7next_domains.toml`): python_interpreter.py row replaced
by process_interpreters.py (D05); new rows D04 process_facts, D05
owner_delivery, D07 delegate_registration_policy + patch_verdict, D08
telemetry_events, D09 owner_quiz, D11 endpoint_index + routing_decision +
task_decision, D17 routing_wait, D18 node_runtime.

Open fork questions for the owner:

- **Q-F6-1 (D02 vs R5 fallback)**: the regex-fallback retirement above follows
  the ratified D02 organ; if the owner wants upstream's fallback-for-legacy
  reading preserved on THIS branch, the typed-absence contract pin and the two
  comment corrections are the exact surface to revisit.
- **Q-F6-2 (A.23 numbering)**: three approved classification deltas are recorded
  under a self-assigned `A.23` owner-item id (the table validator requires the
  `A.` prefix); ratify or renumber in the next owner batch.
- **Q-F6-3 (owner-stop drain monkeypatch surface)**: the FINALIZE_NOW drain now
  calls the upstream-extracted `supervisor.owner_stop.handle_finalize_now_entry`
  directly; `loop._mark_owner_stop_control_drained` remains re-exported but the
  drain path no longer reads through the loop facade. No test relied on that
  monkeypatch point; flagged in case an external harness did.

## From the F6 rolling-upstream sync #2 (upstream f3fbfdbb, base 3e4a6181)

One merge (`git merge f3fbfdbb`, merge-base 8d13373b: 101 upstream commits,
180 files, 20 of them new, 13 overlapping campaign-touched paths) under the
standing principle **upstream = semantic truth, campaign = structural truth**.
Merge commit b9ceed6e; every conflict resolved by keeping OUR facade/leaf form
and re-seating the upstream semantic delta in the leaf that owns its span.

Decision map by conflict class:

- **Campaign-split monoliths** (registry.py, tools/core.py, tools/shell.py,
  tools/git.py, llm.py, loop.py, loop_tool_execution.py, tool_access.py,
  review_state.py, review_helpers.py, skill_review.py, headless.py,
  supervisor/events.py, supervisor/workers.py, config.py, delegate_integration.py,
  subagent_integration.py, and the four split test monoliths): OUR facade kept;
  each upstream hunk re-seated in its owner. Notable seats:
  - registry post-exec organ: the owner-state snapshot/restore was DELETED
    upstream (it reverted ANY post-command difference without proving cause, and
    read an OSError as "file absent" so a transient read error could unlink the
    live settings.json). Its replacement — the `_owner_settings_snapshot`
    baseline plus the OWNER_SETTINGS_CHANGED tripwire, which ANNOTATES and never
    rolls back — lives in `registry_guard_process`, with the dispatch half
    (`settings_before`, tripwires on the TOOL_ERROR path, #447 B2) in
    `registry_core`. `_binding_state_drive_root` went with it as dead code.
  - the read-carve's git classification now consumes the `git_shell_policy`
    SSOT (`_git_subcommand_is_readonly` + `_git_output_file_args`, #447 A7); the
    divergent `_READ_ONLY_GIT_SUBCOMMANDS` table is retired.
  - `gh repo create/delete/auth` moved from substring matching to the
    argv-positional `gh_shell_block_reason` resolver (A7).
  - #447 H1 note ordering: ALL host notes (auto-route, safety warning,
    light-repo, workspace-ref and settings tripwires) now TRAIL the payload, in
    `tool_result._compose_execute_result`, `extension_dispatch` and the
    post-exec guard. `_wrap_run_script_process_result` stops REPLACING a
    successful script's payload with the undeclared-outputs nudge.
  - В23=A owner-home read carve: `user_files_path_block_reason(operation=)` in
    `tool_access_user_files` applies the credential-shape gate to MUTATIONS
    only, delegating to the new upstream `credential_shapes` leaf; the read
    egress is masked instead, in `core_file_tools` (read) and `tools/core`
    (search). `_WRITE_LIKE_OPS` (H2) seats in `tool_access_types`.
  - #468 shape-first reasoning pin: `transcript_has_sealed_reasoning` replaces
    the model-family portability predicate in BOTH directions —
    `llm_openai_compatible` (proactive dispatch pin) and `llm_fallback`
    (reactive reroute) — with the pin fact staged on send success
    (`llm_attempt._stage_reasoning_pin_disclosure`), carried on a ContextVar in
    `llm_messages`, read into usage in `llm_openai_compatible`, and rendered by
    `supervisor/events_budget`. `_pop_thread_disclosure` generalizes the
    thread-local popper in `llm_capability_policy`.
  - delivery control: `control_episode_seen` provenance, the
    `_classify_parsed_delivery_control` classifier and
    `_resolve_forced_delivery_control_body` seat in `loop_delivery`; the forced
    flow (control resolved BEFORE the incomplete branch, `candidate_reason`
    plumbed through `_forced_fallback_result`) in `loop_forced_finalization`.
  - D4 export policy (per-member skip receipts, workspace-patch SSOT for
    credential shapes) -> `shell_outputs`; A5 literal-argv DISCLOSURE (shell
    operators/redirects/env refs in direct argv are notes, no longer refusals)
    -> `tools/shell._literal_argv_notes`.
  - patch capture: sensitive untracked files became a per-file exclusion rather
    than a whole-manifest error, and PEM private-key material is detected by
    CONTENT (`workspace_patch_capture`); the exclusion RENDERER seats in
    `workspace_patch_rules` (the SSOT for the pure manifest rules it renders,
    and the natural owner of a formatter two integration tools display).
  - partial attachment staging (В25c) -> `supervisor/worker_chat_lane` beside
    the already-merged `workers.py` half; `reason_kind` (H4) ->
    `review_state_records`; `--untracked-files=all` -> `review_file_pack`;
    the H3 module-load omission ledger -> `registry_core`.
- **Upstream twins of campaign organs — the twin never lives**:
  - `tools/read_inspection.py` (verbatim extraction of the read-carve at
    upstream's byte gate) folds into `registry_guard_process`, which already
    owned it; the facade re-export serves every test.
  - `tools/result_envelope.py` (the В12=A *minimal* typed-result variant) folds
    into `tools/tool_result.py`, which is the same contract in stronger form
    (status/code/meta published objects the loop reads directly). Its test was
    retargeted onto the campaign organ. ONE product gap the fold surfaced is
    closed with it: with a note appended, the whole text is no longer parseable
    JSON, so the legacy TEXT adapter lost structured-failure detection —
    `_structured_failure` now also tries the pre-note payload.
  - `tools/output_export_policy.py` (extraction of the export-eligibility rules)
    folds into `shell_outputs`, which already owned them.
  - `delivery_protocol.py` stays folded in `loop_delivery` (sync #1 decision);
    upstream's further edits to it were re-seated there.
- **Protected surfaces**: docs/CHECKLISTS.md took upstream's restructured item
  21 (executable guard-change trigger, owner-acceptance rule, standing
  disclosures moved to the new `docs/CHECKLISTS_ARCHIVE.md`).
  `docs/CHECKLISTS_ARCHIVE.md` is taken from upstream BYTE-IDENTICAL and the
  campaign's floor correction lands as an APPENDED superseding entry — owner
  batch #9 item 6=A, and the archive's own stated rule ("append-only:
  corrections land as new superseding entries, not edits to old ones"). The
  first pass of this sync edited the v6.80.0 entry in place; that was reverted
  and re-done as the append. The entry states the ABI-5 train's own wording —
  the `OUROBOROS_SCOPE_REVIEW_FLOOR` gateway surface (key, endpoint, contract
  field, route, merge-skip, web client, self-lowering guards) IS removed in 7.0
  by owner Q10=A, the key retired via `RETIRED_SETTING_KEYS` — and names the one
  clause of the superseded entry that narrows with it: the inverted-polarity
  read-carve survives family-wide as `_owner_control_mention_blocks`, while the
  floor-SPECIFIC detector went with its setting. gateway/contracts.py,
  runtime_mode_policy.py and registry.py deltas landed by the leaf-seating rule
  above.
- **size_ratchet_manifest**: union-resolved (mcp_client's rationale merges the
  campaign typed-organ and upstream E5 reasons), then regenerated by
  `scripts/regenerate_size_ratchet.py`; `--check` green.
- **Web/vendored assets**: the upstream web wave (chat.js +1379, style.css,
  chat_decision.js) landed as-is; `node --test` 838/838.

Disclosed dispositions (contract-affecting):

- **`ambiguous_safety_wrapper` retired**: the meta recorded ambiguity about a
  `---` separator the host itself inserted around the payload. With H1 the host
  inserts no wrapper, so a separator in the composed text is the producer's own
  markdown rule and carries no ambiguity. Which notes rode along stays disclosed
  by the existing `route_note`/`safety_warning` host facts. Upstream's `notes`
  LIST is deliberately NOT adopted: it carries full note text, and the campaign's
  host-meta reserve is a bounded 256 bytes — a real safety warning would raise
  ValueError inside composition. The note text is in the result itself.
- **`OWNER_STATE_RESTORED` kept as a legacy-only code**: no producer emits it
  any more, but persisted traces from ≤6.113 still carry the marker text and
  must keep classifying through `LegacyTextResultAdapter` rather than degrading
  to `LEGACY_TOOL_ERROR`. Documented, not resurrected.
- **Two preflight heuristics removed (В27)** — the commit-message
  version-reference guess and the tests-required predicate. Six enforcement
  pins that asserted the block were inverted to assert the pass, with the
  reason named in each; `test_copy_source_not_treated_as_deletion` was removed
  because it pinned an effect that no longer exists.
- **Skill review judges binaries by CONTENT** (X4/В21): a renamed ELF still
  hard-blocks, a text file with a scary extension no longer does, and a
  non-executable non-UTF-8 file enters the pack as a typed descriptor. The
  campaign's `test_skill_review_packs` pins were rewritten to the new contract
  rather than kept asserting the retired "any non-UTF-8 blocks review" rule.
- **Function/band gates paid down where the change lives**, not by raising a
  ceiling: `api_tasks_create` (dead pre-assignment + one duplicated statement of
  the same fail-closed rationale, 299), `_execute_legacy_text` (one elif ladder
  instead of a second early_error branch, 296), `subagent_integration` (the
  exclusion renderer moved to its natural owner, 998).
  `tests/test_services_tool_v2.py` took a band rationale.

Domain map (`ouroboros/domains.toml`): new rows `credential_shapes.py` (D13),
`reasoning_artifacts.py` (D02), `gateway/claudexor_quota.py` (D11).

What the merged tree's own battery then found (29 reds, all dispositioned):

- **Three product gaps the sync opened, closed at the cause**: (a) the skill
  owner-state read-carve was taught to the predicate but never passed at its
  call site, so `rg review.json` stayed refused with a WRITE-named marker;
  (b) a post-exec tripwire lost its typed fact whenever the producer returned
  plain text — with the notes now TRAILING, the marker no longer owns line 1,
  so a text-only reader could not re-derive the classification; the guard
  adapts once through the ONE legacy adapter instead; (c) the reasoning-pin
  ContextVar first landed in `llm_messages`, which imports `llm_attempt`,
  closing an import cycle the leaf-graph test caught — it moved to
  `reasoning_artifacts`, beside the fact it carries.
- **One golden case repaired rather than re-baselined**:
  `openrouter.payload.or_provider_never_unpins_reasoning` held a READABLE
  `reasoning: "t"`, which the shape-first classifier never pins, so accepting
  the recorded flip would have left a case that no longer tested its own
  contract (the owner preset "cannot lift the pin" with no pin to lift). Its
  transcript now carries a sealed artifact and the pin holds.
- **One real classification delta recorded as A.24**: a structured
  `{"ok": false}` answer behind an appended host note read as SUCCESS under the
  retired pair, which json.loads()-ed the whole composed text. The same defect
  class the 329 OSWorld rows measured, on the composition seam.
- **`echo <owner-state path>` stopped being a denial** in three ordering pins:
  it is a pure inspection under A2, so those pins are spelled with a real write
  (`cp payload.json …`) and still assert the ordering they exist for.

Open fork questions for the owner:

- **Q-F6b-1 (upstream `notes` meta vs the bounded host reserve)**: upstream
  records every host note's TEXT in result_meta. This branch records the typed
  booleans instead, because the campaign's `_MAX_HOST_META_BYTES` is 256 and a
  single safety warning exceeds it — adopting the list verbatim would make
  composition raise on ordinary traffic. If the owner wants the note text in
  result_meta, the decision is whether to widen the host reserve (a numeric
  contract change) or to store bounded note KINDS.
- **Q-F6b-2 (В23=A read carve scope)**: upstream's owner-home read carve lifts
  the credential-NAME gate for the root principal on read/list/search and
  relies on egress masking. The campaign inherits it unchanged, including the
  disclosed residual that masking is shape-based. Re-affirm or narrow.
- **Q-F6b-3 (A5 literal-argv disclosure)**: shell operators, redirects and env
  references in a direct argv array are now DISCLOSED notes rather than
  refusals. This is a capability widening on the model-facing surface (commands
  that used to be refused now run). Ratify, or keep the refusal for the
  operator subset.

## From the F5 lane C (CPL-3/6, base a12c873c)

Lane deliverables (single-intent commits on this lane):

- CPL-3 (feat commit): architecture facts with Ouroboros self-evolution as
  consumer #1 — `ouroboros/code_intelligence_architecture.py`, a new D05 leaf
  beside `code_intelligence.py` (the reuse-first survey found the existing
  module at 801 lines against the 1000-line band floor, so the organ grew a
  leaf instead of pushing the parent into the band; manifest row added,
  DOMAIN_MAP regenerated, no new cross-domain direction — D05->D13 was
  already in the pinned matrix). Five pure queries over data the repo
  already pins, no LLM / caches / ledgers: `owner_of(path|symbol)` and
  `domain_dependencies(d)` over `ouroboros/domains.toml` (symbol resolution
  through the existing code inventory), `facade_consumers(sym)` over the
  same noqa-F401 top-level re-export convention the generated facade
  inventory pins (consumers = import statements across the manifest
  population; attribute access on a plain module import is disclosed
  out-of-scope), `persistence_entities_written_by(sym)` over the
  `docs/PERSISTENCE.md` Path|Writer tables (module path, dotted module, or
  bare writer-function name), `protected_contracts_affected(diff)` over the
  `runtime_mode_policy` protected inventories plus the generated
  `docs/v7next/FROZEN_CONTRACTS_INVENTORY.md` rows (unified-diff text or a
  changed-path list). Suite `tests/test_architecture_facts.py`: every query
  on real examples (protected diff → the contract is NAMED; facade → its
  consumers; writer → its entities) plus completeness against each carrier
  in both directions — every manifest module answers with exactly its
  pinned domain, the per-domain edges reproduce `[graph].allowed` and
  `[graph].lazy_only` exactly, the runtime facade scan equals the pinned
  facade-inventory row set, the persistence parser yields one row per table
  line with every exact writer span resolving, and every frozen-contract
  row is reachable from its own owner file.
- CPL-3 tool-seam DECISION (lane decision, per the lane instruction): the
  model consumes architecture facts through the EXISTING `query_code` tool —
  a new `op=architecture` with `query='<fact> <argument>'` — and NO new tool
  enters the registry. Justification: `query_code` is the established
  read-only code-intelligence seam, already carried by the main loop and
  both subagent profiles and already op-vocabulary-shaped; architecture
  facts are exactly its kind of answer (compact rows over repo structure).
  The op serves code roots only (`active_workspace`/`system_repo`); any
  other root is refused typed. Registry/tool_capabilities untouched.
- CPL-6 (test commit): `tests/test_multiprovider_conformance.py` — the
  normative shared suite of the two multi-provider seams. Provider half:
  parametrization DERIVES from the factual registry
  (`provider_models.PROVIDER_PREFIXES` + the local lane), so a newly
  registered provider without a conformance driver is structurally red; the
  shared contract every lane passes: route-resolution form (required target
  keys, unambiguous per-provider usage_model attribution), `(message,
  usage)` shape, honest-only cost planes (`cost` always present, None when
  unknown, `cost_final` never true over an estimate; the local lane's 0.0
  is its honest free contract), transport failure raises instead of
  fabricating an answer, typed policy refusal is permanent by class (exactly
  one physical send on EVERY lane — probed and true uniformly, incl.
  anthropic/gigachat), HTTP-200 body 429 is a typed `provider_error`
  rate-limit marker (lanes derived from the route's own
  `supports_openrouter_extensions` fact, not a hardcoded list),
  `finish_reason: null` is surfaced observably (key present, null marker)
  on the choice-shaped family, caller timeout reaches the transport on
  every lane (payload / request-row / client slot per transport), and one
  successful send settles exactly one physical-attempt ledger row
  (reserved→dispatched→settled) attributed to the right provider.
  Transports are the recording fakes REUSED from
  `tests/test_llm_provider_golden.py` — the golden suite characterizes each
  route byte-level; this suite pins the cross-provider norm. Executor half:
  parametrization derives from `subagents.SUBAGENT_EXECUTORS` (a new axis
  point without an outcome row = red); the closed rule-table matrix
  (requested × route state → executor/reason/blocked, reset instant riding
  along whenever exhaustion is the surfaced fact), typed refusals at the
  schema seam (`normalize_subagent_executor`) and the tool seam
  (`delegate_start` → `subagent_selection_required`), stale stored executor
  degrades to auto, a plain task is exempt from the axis, the native point
  never contacts the daemon, a started harness run carries run identity
  with the configured route on the wire, and the durable last-delegation
  projection keeps requested vs applied facts separate. Native-side task
  artifacts stay pinned by their own suites — disclosed scope, not a gap.

### CPL-6 findings (observed, NOT fixed in this lane — plan rule)

| id | seam | finding | proposed local fix |
|---|---|---|---|
| CPL6-F1 | local provider lane | the local lane stamps the physical-attempt LEDGER with provider=local but leaves the returned usage dict without the `provider`/`resolved_model` provenance keys every remote lane carries — downstream consumers reading usage alone cannot attribute the call; the conformance suite pins today's asymmetry via the driver's `usage_stamped=False` flag instead of hiding it | stamp `usage["provider"]="local"` / `usage["resolved_model"]="local-model"` in the local normalization path (one seam in `llm_local.py`), then drop the driver flag |
| CPL6-F2 | provider goldens | `tests/test_llm_provider_golden.py::test_golden_covers_every_declared_provider_lane` floors coverage with a HARDCODED lane set, so a new registry provider never turns the golden suite red | now structurally closed by the conformance registry pin; optionally re-derive the golden floor from `PROVIDER_PREFIXES` the same way |

## From the F4 wave 2 (subagent/cancel/update, base a12c873c)

Scenario wave on the lane-1 skeleton — three plan-§8 surfaces (subagent tree,
cancellation, managed-update core), five manifest rows S6-S10, all keyless
mock-lane, every scenario green on this host.

1. SCENARIOS LANDED (tests/system_e2e/test_system_scenarios_w2.py):
   - S6 subagent tree (~28s solo): a ReplayModel parent drives
     schedule_subagent → wait_tasks → exact-hash tree_note
     child_result_disposition → final. The child runs on its OWN stub slot
     (roster row routes to `openai-compatible::mock-child`), the slot binder is
     `model id × tool-bearing shape`, so the supervisor's tool-less semantic
     duplicate probe (light slot) gets its own deterministic fixture ordinal
     and parent/child concurrency can never mis-consume a step. Pinned: child
     row lineage (parent_task_id / root_task_id / delegation_role=subagent /
     depth_provenance.achieved_depth=1 / configured_subagent snapshot /
     task_contract.lineage), swarm_fanout receipt in the PARENT's forked
     drive, child's marker text reaching the parent verbatim through the
     durable wait_tasks tool row, quiescence (child task_done strictly
     precedes parent task_done; parent terminal clean, NOT
     children_unabsorbed), the authoritative disposition row on
     task_trees/<root>/blackboard.jsonl, root rollup keys
     (accounted_upper_bound_usd_with_children on the parent's terminal event,
     honest-only names both rows), and fixture integrity via assert_consumed
     (the observed call pattern is EXACTLY the scripted tree — the fixture
     model matched the live server first try).
   - S7 cancellation single (~11s solo): typed `cancelled` durable terminal
     with non-empty owed answer, honest cost plane
     (accounted_upper_bound_usd + cost_accounting_status enum, no retired
     aliases at top level), self-draining cancel_intents projection, forensic
     requested→claimed→settled trail (source=http_single, outcome=cancelled),
     terminal task_done event, and the /proc no-orphans oracle (below).
   - S8 cancellation cascade (~20s solo): live child (schedule_subagent roster
     row), cascade=true over the UI's endpoint; both rows cancelled, root
     intent scope=cascade minted at the ingress and settled only on the
     "cascade postcondition" detail, descendant carries its own
     source=cascade_descendant intent naming the root, the durable
     task_cancel_subtree_snapshot lists the child, both intents drained, both
     task_done events written, no orphan processes.
   - S9 managed update ff core (~90s solo, the expensive one — deliberately a
     single test): a REAL managed install (`.git/ouroboros-managed.json` +
     `managed` remote onto a LOCAL upstream one ff-commit ahead;
     OUROBOROS_UPDATE_CHANNEL=development), DIRTY tree (tracked edit +
     untracked file), preflight → apply(auto_merge) over the live HTTP
     surface: stash-first insurance (Q1=C) carries the work, the server
     re-execs, boot-finalize consumes tx + intent markers, writes
     `managed_update_finalized` with head == target and
     `managed_update_stash_restored` (context=boot_finalize), the worktree
     lands exactly on target with the dirty work back as uncommitted content
     and the durable `rescue-local-<stash12>` pin present.
   - S10 rollback contracts (~2s solo, subprocess driver on a second real
     isolated install — the real supervisor code, no live server needed):
     absent marker → typed "no pre_update_sha" refusal; explicit null stamp →
     strict `corrupt`, same typed refusal, marker bytes byte-identical;
     FUTURE-schema stamp → the "newer version" refusal from BOTH
     rollback_managed_update and finalize_managed_update_on_boot, marker
     byte-identical (left for the owner); and the restore contract — a
     half-applied update (target commit + extra dirt + valid pending tx)
     rolls back to a worktree whose FULL file fingerprint (sha256 of every
     non-.git file) equals the pre-update snapshot, porcelain empty, failed
     candidate preserved on `failed-update-<target12>`, tx cleared, durable
     `managed_update_rolled_back` receipt with the exact pre_update_sha.
2. HARNESS DELTAS (tests/system_e2e/harness.py): (a) callable steps — a
   ScriptedStubModel script step or ReplayModel fixture row may be
   `callable(body) -> step`, deriving arguments the scenario cannot know
   statically (server-minted child ids, exact result hashes) from the
   transcript the model was actually shown; default-lane pins cover both.
   (b) ReplayModel `model_ids=` override for compound slot binders (the
   default /models advertisement derives from fixture slot names, which are
   not wire model ids under a compound binder). (c) `pids_with_env_value` —
   the /proc environ scan behind the no-orphans oracle: every pid carrying
   the scenario's unique data root must sit INSIDE the live server tree
   (`process_tree_pids`), re-polled via wait_until so a transiently exiting
   worker cannot false-positive. (d) ArtifactOracle readers:
   terminal_deliveries (owed-answer outbox), child_task_ids (lineage
   enumeration — the parent row deliberately lists no children),
   tree_blackboard. (e) The manifest gen/verify + marker pins now scan EVERY
   test module of the package (wave modules stay visible to the discipline);
   the shared session clone fixture moved to the package conftest.py —
   scenarios that move HEAD or add remotes (S9/S10) build private clones.
3. LOCAL MANAGED REPO WITHOUT PATCHING (S9 design decision): the update path
   hard-pins the managed remote's URL to the OFFICIAL github URL on every
   fetch (`ensure_official_update_remote`, supervisor/git_ops_updates.py:80,
   called from plan_managed_update_merge fetch=True), so a live-server local
   managed repo is reached by redirecting that exact URL through standard git
   `url.<mirror>.insteadOf` config in the ISOLATED clone — install
   configuration, byte-identical runtime path (the set-url still happens, the
   fetch resolves to the local mirror). The pip step of update_restart_smoke
   was verified a no-op against this venv (`pip install --dry-run -r
   requirements-runtime.lock` → nothing to install) and belted with
   PIP_NO_INDEX=1 in the scenario env so an unexpected resolution attempt
   fails loudly instead of reaching the network.
4. E2E-находки wave-2 (runtime defects/observations — NOT fixed in this lane,
   per the lane rule):

   | id | surface | observation | evidence |
   |---|---|---|---|
   | W2-F1 | cancel owed-answer / details panel | The `cancel_receipt` block (Q5=A details-panel facts) and the outbox registration exist ONLY for tasks with chat lineage: `build_unreviewed_salvage_event` returns None for a chatless task and the owed answer degrades to the typed `terminal_delivery_handoff` row (reason=no_lineage_chat). Every API-submitted (headless) task therefore never gets the details-panel stop receipt. Contract-conformant per GR2-4, but the panel-facts gap for headless tasks may deserve an owner decision. | supervisor/cancel_publication.py:212-239; supervisor/terminal_delivery.py:1326-1327; observed live in S7 (receipt absent after 60s poll, handoff row present) |
   | W2-F2 | managed update source | `managed_remote_url` from `.git/ouroboros-managed.json` is honored ONLY by launcher/colab bootstrap (launcher_bootstrap.py:280,323); every update fetch unconditionally retargets the remote to the hardcoded `OFFICIAL_UPDATE_REMOTE_URL` (git_ops_updates.py:86-90). An install bootstrapped from a fork/mirror (colab writes `managed_remote_url=source_url`) is silently retargeted to razzant/ouroboros on its first update check. Air-gapped/fork installs need git insteadOf config (as S9 does) or an owner decision to honor the meta URL. | git_ops_updates.py:80-90; colab_bootstrap.py:369; update_merge_plan.py:161-169 |
   | W2-F3 | /api/state identity | `/api/state` answers `"sha": ""` and `"branch": null` on a source-mode isolated server even after a completed managed update — the state surface does not carry runtime repo identity here; identity lives in /api/health runtime_version + boot attestation. S9's restart proof therefore rests on the boot-finalize receipt (which only the restarted process can write), not on /api/state. | observed live in S9; devtools/.../server_runner.py current_sha() reads this field |
   | W2-F4 | "managed" is two different predicates | server.py:144 gates the bootstrap safe_restart on the ENV flag only (`OUROBOROS_MANAGED_BY_LAUNCHER=1`), while the update surface gates on the meta file (`git_ops._is_launcher_managed_repo` accepts either). A meta-managed source install (S9's shape) gets managed UPDATES but no managed bootstrap reset — apparently intentional (source checkouts keep their tree), named here so the asymmetry is a decision, not an accident. | server.py:144,610; supervisor/git_ops.py:110-113 |

5. LANE BUDGET: full mock lane (S1-S10 + default pins, serial) — 31 passed in
   ~219s (3:38) on this host; the wave added ~78s over lane 1's ~142s, well
   inside the plan's 10-25 min PR keyless budget and the wave's own 6-8 min
   target. Solo timings: S6 ~28s, S7 ~11s, S8 ~20s, S9 ~21s, S10 ~2s. The
   three new default-lane pins add well under 1s to the ordinary battery. Deferred to wave 3 (disclosed): carrier-conflict /
   assisted-merge / crash-mid-phase update variations, chat-lineage cancel
   receipt path (the outbox+receipt form of W2-F1), delegated-transport
   (FakeClaudexorDaemon) and gateway/UI-truth (Playwright) waves.

## From the external-audit correction lane (base 8827fd2c)

Five externally-audited items, re-verified by the coordinator, fixed as five
single-intent commits on `ouroboros_v7next` (no push — P-LANE pause). Item 4
was amended mid-lane by owner answers №8=A / №9=A (2026-09-01, relayed by the
coordinator); the amended form is what landed.

1. **wait_tasks description vs producer (ABI-3 honesty)** — FIXED (d3db6424).
   `tools/control.py` promised the model a `cost_usd` projection key while the
   producer (`control_task_results` batch projection) emits
   `accounted_upper_bound_usd` + `cost_final`. Description now names the
   actual keys. Class sweep: no other builtin tool description and no
   prompts/ text mentions the removed alias; the `_children_roster_projection`
   docstring lied with the same legacy name and was aligned to the fields the
   roster actually emits (`accounted_upper_bound_usd` only — no `cost_final`
   there). Pins: `tests/test_cost_projection.py`
   (`TestModelVisibleToolSurfaces`) — wait_tasks description carries the
   actual keys and no legacy spelling, plus a class-wide registry sweep over
   every builtin tool schema.
2. **write_text_atomic(fsync=True) short write** — FIXED (f772717c). The
   fsync lane issued one bare `os.write` and trusted its return; a POSIX
   partial write published a truncated file behind the successful atomic
   rename. Both fsync lanes now share `_write_fd_fully` (the loop
   `write_bytes_atomic` already had), each keeping its own open flags (no
   `O_BINARY` on the text lane — historical platform newline semantics
   unchanged). Pins: `tests/test_atomic_write_v639.py` — one-byte-at-a-time
   `os.write` mock, both lanes, full content on disk.
3. **rc_audit: present-but-unparseable ui_preferences.json audited clean** —
   FIXED (b0960407). The bare `except JSONDecodeError: return` violated the
   fix-round-1 contract («a malformed mandatory source is never a clean
   exit 0»); it is now a blocking `unauditable-source` finding (exit 1), same
   class as an unparseable skill manifest; a read OSError still propagates to
   exit 2. Class sweep of every `_audit_*` source: settings raises
   InstallUnreadable (exit 2), skills raise `unauditable-source`,
   task_results map parse damage to the blocking schema-stamp quarantine
   finding — ui_preferences was the one surviving instance. A parsed
   non-object still audits clean (it holds no keys; the legacy-key audit has
   a truthful answer). Pins both ways in
   `tests/test_rc_audit_fixture_suite.py`.
4. **ADOPTION_v7next.md status truth** — FIXED (1e9915ac), amended per owner:
   CPL-1 stays `done` WITH SANCTION — owner №8=A (2026-09-01): the
   all-20-domain strict-quotient SCC ceiling is accepted for v7.0
   (shrink-only gate, target empty; true cycles=0 = post-release campaign);
   open residual disclosed: 80 `[classification].proposed` new-upstream
   placements await owner review. CPL-4 `done` → `in-progress`: inventory +
   verify hook done, 23 candidate code fixes (CPL4-C1..C23) deliberately not
   touched; owner №9=A routes the mechanical fixes (rotation train +
   retention knob + orphans) into v7.0 as a separate lane, the 7
   owner-decision rows into one pre-release batch. TRAIN-F6-8d13373b row
   added (the header's promised train row for the post-cutoff upstream lane):
   121 upstream commits b9f7597f..8d13373b, merge 0aa74e9f, done, phase F6,
   hook = this ledger's F6 sync section + the merge + the full batteries.
   Validator: `ID_RE` already admits TRAIN- ids and phase F6; plain run OK
   (37 rows); `--release` red AS EXPECTED — current count: rc=1,
   31 findings over 21 rows (18 pending + 3 in-progress status rows, 4
   pending-decision dispositions, 4 missing-hook-file + 3 prose-only-hook
   findings among them).
5. **ARCHITECTURE.md stale after the F6 sync** — FIXED (b652bd15). Removed
   the `/api/owner/scope-review-floor` endpoint-table row and corrected the
   owner-endpoint count to four (the gateway mounts exactly
   runtime-mode/auto-grant/context-mode/safety-mode); retargeted
   `delivery_protocol.parse_delivery_control_body` →
   `loop_delivery._parse_delivery_control_body` and
   `acceptance_dialogue._set_acceptance_decision` →
   `loop_acceptance._set_acceptance_decision` (both verified still
   re-exported from `loop`); same-class fix in DEVELOPMENT.md's acceptance
   checklist row. No `chat_delivery_events` mention existed in
   ARCHITECTURE.md. `regenerate_inventories.py` reruns byte-identical (§11.1
   untouched); `check_domains` green.

## Owner closures for the F6-sync forks (2026-09-01, batch 7 + re-ask)

- **Q-F6-1 CLOSED (owner №1=A)**: typed process facts come only from
  structured records; the upstream prose-regex fallback stays out. The owner
  additionally commissioned a provenance investigation (how the fallback was
  born and passed review, and whether the underlying gap — delegated runs
  lacking typed exit facts — is deeper); its findings land as a separate
  ledger section when verified.
- **Q-F6-2 CLOSED (owner «ок, A» on the re-asked plain-language question)**:
  the `A.23` classification row is RATIFIED as-is (the three reclassifications
  BROWSER_SESSION_RETIRED ok→timeout/error, BROWSER_BACKLOG_RETIRED_SESSIONS
  ok→unavailable/error, ESCALATE_UNAVAILABLE error→unavailable under
  upstream #440 semantics).
- **Q-F6-3 acknowledged (owner «ок»)**: the FINALIZE_NOW drain calling
  `owner_stop.handle_finalize_now_entry` directly is accepted; the loop
  re-export stays.
- Related batch-7 outcomes recorded for the campaign: №4=A (update remote
  honors the configured source — F6-tail fix), №5=A (headless cancel
  receipts — post-release), №6 confirmed (ABI-8 stays post-release backlog).
## From the F5 lane D (CPL-5 impl + small fixes, base 8827fd2c, 2026-09-01)

1. CPL-5 LANDED (ratified note `docs/v7next/DESIGN_MODEL_VISIBLE_LOGGED.md`,
   F15-narrowed to `model_send` at `llm_attempt._candidate_before_dispatch`).
   Mechanics, all reuse-first (no parallel serializer, plane or scheduler):
   - **Seal**: `persist_physical_candidate` now finalizes the
     EXISTING manifest with a `model_send_seal` block (v1; basis
     `canonical_json_v1`; `pre_redaction_sha256`/`size_bytes` = the existing
     candidate digests; per-instance exclusion rows). `persist_call` gained the
     one seam this needs: an optional `finalize_manifest(RedactionResult)`
     callable, so the seal discloses the exact redaction instances of the CAS
     write without re-running redaction. The returned `manifest_ref` is stamped
     `model_send_seal_version`, which lands on the ledger row and is the
     reverse-sweep join marker (legacy pre-seal rows are structurally excluded
     — zero false positives by construction). `persist_physical_candidate`
     itself moved whole into `model_send_seal.py` (its seal-bearing home)
     because `observability.py` stood 29 lines under the 1500 band ceiling;
     `observability.persist_physical_candidate` stays as the thin historical
     compatibility name (llm_attempt's lazy import and existing monkeypatch
     surfaces unchanged).
   - **Forward, on the call**: after persist, the seam
     (`model_send_seal.verify_sealed_candidate`) re-reads the manifest + CAS
     blob FROM DISK, applies the same exclusion map to the wire-bound candidate
     (`physical_custody_projection` → `observability_custody_projection` →
     `redact_projection` → CAS-basis serialize) and compares byte-for-byte,
     plus digest-compares the seal against the fresh seam identity (reused, not
     recomputed). Mismatch = typed durable
     `model_send_invariant_violation` fact written beside the seal
     (`<attempt>.model_send_violation.json`, write-once dedup latch) AND into
     `logs/events.jsonl` — per the ratified narrowing this is an OBSERVABILITY
     invariant: it never blocks dispatch; the pre-existing in-memory identity
     refusal stays the only blocking authority. Kinds/classes:
     `content_divergence/sdk_mutation` (durable claim vs wire bytes),
     `reconstruction_divergence/{seal_unreadable, serializer_basis,
     undisclosed_exclusion, record_unreadable, record_corrupt, redaction}`.
     Facts carry digests and the first divergent offset only, never payload
     bytes.
   - **Closed exclusion enum** (§4): `secret_redaction` (per-instance rows from
     the CAS write's RedactionRecords), `provider_native_custody` (per-item
     rows named by a structural diff against the custody projector's own
     output — the projector stays SSOT of what leaves the byte domain, with
     `opaque_sha256` where the projection minted digests), `transport_envelope`
     and `provider_side_transform` (class-level rows, always disclosed: the
     SDK adds envelope below the seam on every lane and provider-side
     transforms are unobservable, so an empty exclusions list would be a false
     exact-reconstruction claim). An exclusion row with an unknown class is
     itself a violation (`undisclosed_exclusion`). Delegated harness sessions
     (`record_subscription_session`) now carry `model_send_seal: "unobserved"`
     on their ledger rows; opaque SDK attempts remain disclosed by their
     existing `candidate_measurement_kind: "opaque"`.
   - **Note boundaries kept literally**: per-rung seals (each ladder rung =
     new candidate + own seal/attempt id — pinned), versioned serializer basis
     (a foreign basis is reported, never re-read), single-assembly rule stays a
     design-level constraint on the response side (non-goal here; the next
     round's send record covers assembly transitively).
   - **Reverse sweep** (§3.3): `model_send_seal.reconcile_model_send_seals`
     rides `server_maintenance._startup_custody_sweep` (no new scheduler);
     bounded (2000 manifests / 50 facts per pass), fail-soft (unreadable
     ledger = UNKNOWN state = whole pass skipped; the sweep only writes facts,
     so there is no destructive conclusion to skip), facts `orphan_seal` /
     `unlogged_attempt`, never repairs. Manifests promoted from child drives
     are excluded via a new honest provenance marker
     (`promoted_call_manifest: true`, stamped by `promote_call_manifest_ref`)
     — their attempt rows legitimately live in the child ledger.
   - **Cost, measured** (this host, isolated env, fake transport): on a
     ~112 KiB canonical candidate the verification adds ~29 ms/call
     (persist+dispatch path 46 → 75 ms; one read-back + one
     projection+serialization, dominated by the redaction regex pass — the
     same order as the persist itself, i.e. the note's budgeted "one
     read-back and one projection"; sub-1% against a real multi-second
     provider round-trip). Measured linear in candidate size
     (~0.3 ms/KiB/projection: 111 KiB → 34 ms, 437 KiB → 152 ms,
     1.7 MiB → 523 ms), so a full-window candidate pays ~0.5 s — still small
     against the provider latency of a request that size. Shape pinned by test:
     verification adds ZERO raw `canonical_json_v1` serializations (seam
     digests reused; 4 stays 4) and exactly ONE `redact_projection` pass
     (persist 1 + verify 1 = 2 total per dispatch).
   - Pins: `tests/test_model_send_seal.py` (19 tests — clean round-trip on an
     ordinary call; deliberately damaged durable records: corrupted CAS blob,
     tampered seal digest, dropped seal, smuggled exclusion class, foreign
     basis → each a typed fact AND dispatch still settles; per-rung seals;
     every enum class covered incl. custody per-item disclosure and the
     delegated `unobserved` row; cost-shape; sweep: clean join, synthetic
     orphan both ways, idempotent dedup, promoted-manifest skip, corrupt-ledger
     skip, startup-family wiring). ADOPTION CPL-5 → done with that hook.
2. CPL6-F1 CLOSED (small fix, no owner decision): `llm_local._chat_local` now
   stamps `usage["provider"]="local"` / `usage["resolved_model"]="local-model"`
   symmetrically with every remote lane; the conformance driver's
   `usage_stamped` escape flag is REMOVED (the provenance assertions now run
   for every lane) and the golden fixture `local.dispatch.tool_call_from_text`
   re-recorded via the suite's own `--write` (delta = exactly the two new
   provenance keys).
3. W2-F3 CLOSED (small fix): `/api/state` no longer answers `sha:""` /
   `branch:null` on source-mode installs. `gateway/state.py` resolves runtime
   repo identity as: supervisor-stamped state values win (managed update/reset
   provenance), else the ACTUAL git checkout at `config.REPO_DIR` read by pure
   stdlib file reads on the snapshot thread (`.git` dir or worktree pointer,
   `commondir`, loose then packed refs; detached HEAD = sha with honestly no
   branch; unreadable layout degrades to unknown, never invented). No
   subprocess on the poll path, no new dependencies, no contract change
   (`branch`/`sha` fields keep their shape). Removed alongside: the dead
   `st.get("current_branch", "ouroboros")` default — `load_state` always seeds
   the key with None, so the "ouroboros" guess never fired and would have been
   a lie where it could. Pins: `tests/test_api_state_runtime_identity.py`
   (checkout reader layouts incl. linked worktree + packed refs; state-wins vs
   source-mode-gap endpoint parametrization).
4. NOT touched, per the lane scope: W2-F1 (panel stop-facts for headless tasks
   — product decision), W2-F2 (managed-remote repin — fork policy, owner
   batch), Q-F6-* (sync forks).
## From the F4 wave 3b (claudexor transport + skills, base 8827fd2c)

Scenario wave on the lane-1/2 skeleton — the plan-§8 delegated-transport
(+restart recovery, no-orphans) and skills-lifecycle surfaces, manifest rows
S11-S13, all keyless mock-lane, every scenario green on this host.

1. FakeClaudexorDaemon LANDED (tests/system_e2e/interfaces.py — the wave the
   lane-1 stub named): a loopback claudexord imitation serving the EXACT
   client contract of ouroboros/gateways/claudexor.py, derived from the code,
   not from memory: authenticated protocol-3 handshake (bearer token from the
   installed descriptor; 401 otherwise), capability catalog
   (`/v2/agent-capabilities` rows with accessProfilesSupported), harness rows,
   fail-open empty quota envelope, Idempotency-Key'd project registry
   (register idempotent per root / find / typed-404 delete), POST /v2/runs
   with the ENGINE REPLAY CHECK (same key + byte-identical digest → the
   ORIGINAL handle; same key + different digest → 409 idempotency_conflict;
   missing key → typed 400), run detail with the summary facts the custody
   settler consumes (state/model/spendUsd/spendEstimated/tokens/
   authRoute.profileId/effectiveAccess, untruncated primaryOutput), and the
   cancel control verb (accepted → terminal `cancelled` on the next read).
   Per-run behavior is scripted by prompt markers ([FAKE:HANG] never-terminal,
   [FAKE:REFUSE] typed 400) plus the pinned ghost-profile 409; every request
   is recorded (method/path/Idempotency-Key/protocol header/body) for wire
   assertions. TWO load-bearing identity choices: (a) the fake reports the
   TREE'S OWN claudexor_runtime_pin.json version+sha — `ensure_running`
   attaches fast-path on exact pin identity, and ANY other version walks into
   `runtime_manager.ensure()`, whose repair path downloads the pinned archive
   (network egress the keyless lane must never take); (b) the descriptor is
   installed at `<data_root>/claudexor/daemon/control-api.json` — the owned
   (D30) layout — so the server's default discovery prefers it with zero
   monkeypatching. Default-lane contract pins drive the fake with the REAL
   ClaudexorGateway (handshake, route_health == ("", ""), registry, replay,
   both refusal codes, cancel), so fake↔client drift is a named failure.
2. SCENARIOS LANDED (tests/system_e2e/test_system_scenarios_w3b.py):
   - S11 delegated transport (~27s solo): a scripted top-level nanny drives
     delegate_start(subagent_id=cx-scout) → delegate_wait (run id parsed from
     the transcript by a callable step) → two more intended starts that refuse
     typed → final. Pinned: the durable custody chain in the canonical
     events.jsonl — 3× START_REQUESTED (one pre-wire row per POST), exactly
     one STARTED (route/model/access=readonly/mode=ask/
     selected_subagent_id=cx-scout), LEDGER_RECORDED, SETTLED
     (state=succeeded, engine-reported model, cost_usd=0.0 + cost_final=true,
     applied credential_profile_id) — wire Idempotency-Key == the STARTED
     row's invocation_id, three DISTINCT invocation ids across the three
     intended starts (fresh logical invocation per intention); the wire body
     the derived SHAPE authored (authPreference=subscription, mode=ask,
     access=readonly, harnesses==[route]==primaryHarness, model pin,
     maxSeconds>0, non-empty host instructions, project scope; NO execution
     block on a readonly run); project registration idempotency + the settled
     run's owned registration retired (DELETE observed); the honest
     requested-vs-applied last-delegation receipt
     (state/subagent_last_delegation.json: requested_model=mock-model vs
     applied_model=mock-model-echo, requested_profile="" vs applied profile);
     and the transcript truth — FAKE_RUN_RESULT, cost_final, the
     session_route_resolves_its_own_model capability_delta and the typed
     refusal code all reached the model. The two refusals land as
     delegate_run_start_failed rows with definite=true (invocation retired:
     scripted 400 fake_route_refused + pinned-profile 409
     credential_profile_unknown — the strict D-U6 pin refused by the ENGINE,
     $0, after route_health correctly fail-opened on the empty quota).
   - S12 no-orphans restart recovery (~24s solo, Linux-only): the nanny holds
     delegate_wait on a [FAKE:HANG] run (durable STARTED row + observed wire
     GETs), the WHOLE server tree is SIGKILLed (hard crash, no cleanup;
     /proc scan proves nothing carrying the data root survives), and a new
     generation on the same clone+data root recovers custody at BOOT
     (_startup_custody_sweep → reconcile_orphaned_runs with running=∅):
     cancel control delivered (observed POST /v2/runs/<id>/control),
     CANCEL_OUTCOME outcome=confirmed (verified terminal read-back), SETTLED
     state=cancelled, RECONCILED action=cancelled — and EXACTLY ONE physical
     POST /v2/runs across both generations (recovery adopts custody from the
     durable rows; it never re-POSTs a run that has a STARTED row — the
     late-result/custody semantics of delegate_custody_reconcile pinned as a
     contract), with every pid carrying the data root inside the live gen-B
     tree.
   - S13 skills lifecycle E2E (~33s solo, two phases): local extension payload
     (SKILL.md + plugin.py, plugin_api: "2.0", permissions tool+inject_chat,
     model_experience prose) written into data/skills/external/ on a LIVE
     server → POST /api/skills/<s>/review runs the REAL triad panel against
     the loopback stub (≥3 skill_review-classified packets answered with the
     canned all-PASS verdict over the tree's own _SKILL_REVIEW_ITEMS) →
     status=clean, durable review.json carries the all-PASS findings plane
     (and NO status key — this tree's derived-status contract for
     verdict-bearing reviews) → POST grants {items:[inject_chat]} →
     all_granted, durable grants.json → POST toggle enabled=true →
     live_loaded+dispatch_live TRUE in the server process → RESTART (the
     product's worker pickup point — finding W3B-F1) → a scripted task
     dispatches the extension tool: durable tools.jsonl row with the result
     text and the ABI-9 tool_result_meta.extension_generation digest (smoke
     over the unit-pinned provenance), CPL-7 Model Experience prose observed
     in a recorded agent body ("Model experience: E2E-MX-MARKER…" in the
     Installed Skills context section) → disable (live_loaded/dispatch_live
     FALSE) → delete (payload dir AND state/skills/<s>/ both removed; listing
     empty).
3. HARNESS DELTAS (tests/system_e2e/harness.py): manifest rows S11-S13; the
   skill-review branch in the stub classifier — SKILL_REVIEW_MARKER ("You are
   performing a SKILL review, …", pinned to ouroboros/skill_review_prompt.py
   through the existing MARKER_SOURCES drift pin) classified FIRST among the
   review branches (its pack embeds whole governance docs that can quote the
   other markers) and answered with `skill_review_clean_text()` — an all-PASS
   array derived from the tree's own `_SKILL_REVIEW_ITEMS`, verified by a
   default-lane pin to aggregate to "clean" under
   `aggregate_skill_review_status`. The lane-1 interface-stub pin now asserts
   FakeClaudexorDaemon constructs (and releases its bound socket) while
   PlaywrightUIClient still refuses.
4. E2E-находки w3b (runtime defects/observations — NOT fixed in this lane,
   per the lane rule):

   | id | surface | observation | evidence |
   |---|---|---|---|
   | W3B-F1 | skills enable → running workers | Enable-time reconcile loads an extension ONLY in the SERVER process; task WORKERS are separate multiprocessing processes that load extensions once, at worker spawn (supervisor/worker_process.py:161 `reload_all`), and the reconcile queue (`extension_reconcile_queue`) is worker→server only — no server→worker channel exists. A skill enabled through the UI/API after boot is therefore INVISIBLE to every task until the workers respawn: the model calling the freshly enabled tool gets "Unknown tool: …" while /api/extensions truthfully reports live_loaded/dispatch_live TRUE (observed live: the S13 draft's dispatch task; generalizes the runbook's OSWorld lesson «сервер грузит расширения только на reload_all»). The UI likely masks this for chat turns served by the server process, but queued tasks run in workers. Candidate fix classes: a server→worker reconcile signal on the existing queue idiom, or a worker-side staleness probe at toolset materialization. | worker_process.py:161; extension_loader._request_server_reconcile_if_worker (worker→server only); registry_core extension discovery reads the process-local `_ext_tools`; S13 first-draft trace: tools.jsonl "Unknown tool: ext_11_r_e2e_probe_echo" with the base-tool Available list |
   | W3B-F2 | skill review durable verdict | `SkillReviewState.to_dict` persists a `status` key ONLY for pending/no-verdict reviews; a verdict-bearing review persists the findings plane alone and status re-derives on load. Contract-conformant (the load side re-aggregates), named here because any external reader of `state/skills/<name>/review.json` that greps `status` will misread a CLEAN review as absent. | ouroboros/skill_loader.py `SkillReviewState.to_dict` (has_review_verdicts branch); observed live in S13 |
   | W3B-F3 | new-extension review admission | A `type: extension` payload WITHOUT `plugin_api` is refused a NEW review PASS by the ABI-1 admission gate (typed plugin_api_admission FAIL, $0, review stays pending) — working as designed (grandfather covers only existing hash-bound PASSes), noted as the one non-obvious install-time requirement for local skill authors: a fresh extension must declare `plugin_api: "2.0"`. | ouroboros/contracts/plugin_api.py:285 extension_new_pass_admission_error; observed live in S13's first draft |

5. LANE BUDGET: full mock lane (S1-S13 + default pins, serial) — 37 passed in
   ~320s (5:20) on this host; the wave added ~100s over wave 2's ~219s, inside
   the plan's 10-25 min PR keyless budget. Solo timings: S11 ~27s, S12 ~24s,
   S13 ~33s. The new default-lane pins (fake-daemon contract ×2 + skill-review
   verdict) add ~2s to the ordinary battery (loopback HTTP, no server).
   Deferred (disclosed): the gateway/UI-truth (Playwright) wave; delegated
   MUTATING-run scenarios (snapshot provisioning + integrate_delegated_patch
   + containment evidence — needs the fake to serve attempt.yaml applied-fact
   artifacts); delegate_answer/waiting_on_user interactive flow; the
   carrier-conflict/assisted-merge/crash-mid-phase update variations carried
   from wave 2. S-id note for the integrator: a parallel Ф4 wave (gateway/UI
   truth) may also claim S11+; renumber ONE side's manifest rows at
   integration if both landed — the gen/verify pins make a collision loud.
## From the F4 wave 3a (review/acceptance, base 8827fd2c)

Scenario wave on the lane-1/2 skeleton — the plan-§8 review surfaces (plan
review; commit triad+scope in BOTH enforcement classes with stale-rejection;
acceptance loop), four manifest rows S11-S14, all keyless mock-lane, every
scenario green on this host.

1. SCENARIOS LANDED (tests/system_e2e/test_system_scenarios_w3a.py):
   - S11 plan review (~22s solo + ~21s cap variant): a scripted task drives
     `plan_task` through a real REVISE→ACCEPT cycle — cycle 1: every triad
     slot (plan review rides the configured triad rows) returns one blocking
     finding with `breaks: goal` → aggregate REVISE_PLAN, open; cycle 2: a
     CHANGED spec (new fingerprint, new paid cycle) → all-clean → GREEN,
     closed. Durable chronicle asserted honest: `plan_review_state` on the
     stored task row (`cycles_paid == 2`, last wave GREEN/closed) plus the
     immutable per-wave artifacts (`task_results/artifacts/<task>/`
     `plan-review-wave-*.json` — aggregates exactly [REVISE_PLAN, GREEN], the
     scripted finding text preserved verbatim). The cap variant proves the
     shared owner ceiling (`OUROBOROS_REVIEW_MAX_CYCLES`, shipped default 2)
     end-to-end: two paid waves (6 slot calls total), then a THIRD changed
     envelope answered with the typed `PLAN_REVIEW_CYCLES_EXHAUSTED` refusal
     at $0 (no reviewer called, stub count still 6),
     `current_attempt.status == "cycles_exhausted"` stamped, and the durable
     `review_cycles_exhausted` escalation event (surface=plan_review,
     cycles_paid=2, cap=2) — which lands in the SERVER-level events.jsonl,
     not the task drive (see W3A-F4).
   - S12 commit triad+scope, ADVISORY class (~25s in-lane): the SAME red
     triad verdict that blocks S13 is waved through — the doc-only commit
     LANDS, and the wave-through is loud and durable (BIBLE P3): typed
     `review_advisory_override` event (block_reason=critical_findings), the
     persistent `state/advisory_overrides.json` counter, and the verbatim
     verdicts on the commit-attempt ledger row. Both organs dispatched
     (3 triad + 1 scope stub calls).
   - S13 commit triad+scope, BLOCKING class (~30s in-lane): red critical
     triad FAIL → `REVIEW_BLOCKED`, repo HEAD does not move; a BYTE-IDENTICAL
     resubmission → the typed `IDENTICAL_DIFF_REFUSED` free refusal (verdict
     streak, $0 — no triad calls consumed); a FIXED diff → clean verdicts →
     the commit lands exactly once (HEAD parent == pre-task HEAD). Durable:
     the verdict-blocked attempt row (block_reason=critical_findings), 6
     triad + 2 scope calls total (none for the identical resubmit).
   - S13 freshness stale-rejection (~50s in-lane, private clone): pins the
     tree's ACTUAL freshness mechanics, both layers, live:
     (a) advisory freshness — a REAL fresh `preflight_review` verdict (the
     advisory row of `OUROBOROS_REVIEWER_SLOTS` pinned onto the stub; the
     native inspection episode answers `[]`+NO_FINDINGS → status "fresh"),
     then a worktree edit (`invalidate_advisory_after_mutation` via
     write_file), then `commit_reviewed` WITHOUT the audited skip →
     `ADVISORY_PRE_REVIEW_REQUIRED` naming the worktree edit, $0 (no
     triad/scope dispatched), the fresh run demoted to status="stale" on the
     durable ledger;
     (b) post-verdict revalidation — with the advisory bypassed, a
     `ReviewScript` CALLABLE hooked on the scope call stages NEW bytes into
     the clone's index mid-wave (after the pre-dispatch
     `_fingerprint_staged_diff`, before settlement) and returns an ALL-CLEAN
     verdict; the commit is still refused: `REVIEW_REVALIDATION_FAILED`,
     attempt row block_reason=revalidation_failed with
     fingerprint_status="mismatch", typed
     `reviewed_attempt_revalidation_failed` event — a verdict for other
     bytes is never carried forward. HEAD never moves in this scenario.
   - S14 acceptance loop, required+blocking (~25s + ~26s variants): the
     terminal runs the REAL acceptance dialogue keyless (acceptance panels
     ride the triad rows; the clean accept cites `verification_summary` — a
     top-level packet section `build_task_acceptance_evidence` writes
     host-attested unconditionally, so the supported criterion's evidence_ref
     resolves under the exact-match vocabulary). Reject→rework→accept: panel
     1 FAIL (well-formed: tier best_effort + coach + typed finding +
     criteria), improvement note fed back, the agent reworks, panel 2 accepts
     clean — `acceptance_decision.status == "accepted"`, aggregate signals
     PASS-only, and the durable wallet
     (`task_acceptance_review_accounting.claims_by_binding`) carries TWO
     claims: both candidate identities were separately paid. Free-replay
     variant: a rework that changes NOTHING is refused for FREE — one paid
     panel only (3 acceptance stub calls total), ONE wallet claim,
     `acceptance_decision {status: finalized_unaccepted, reason:
     identical_acceptance_refused}` — the keyless instance of the $0-refusal
     class.
2. HARNESS DELTAS (tests/system_e2e/harness.py):
   (a) two new prompt markers with source pins — `PLAN_REVIEW_MARKER`
   (tools/plan_packet.py system-prompt head) and `NATIVE_EPISODE_MARKER`
   (review_native_episode.py; deliberately the FIRST source line of the
   two-literal sentence, because the marker-pin test greps the source file
   where the concatenation does not exist); `classify_call` now returns
   `plan_review` / `advisory_review` / `native_episode` (the native episode
   is a TOOL-BEARING review call — unclassified it would eat agent script
   steps; the surface is parsed off the episode's own `Surface:` line);
   `REVIEW_KINDS` is the closed kind set.
   (b) canned clean answers for the new kinds are the shared
   `[]`+NO_FINDINGS shape — verified-clean under BOTH real parsers
   (plan_spec.parse_findings; the advisory clean predicate shared with
   triad_review.empty_array_is_verified_clean), pinned in the default lane.
   (c) `ReviewScript` — ordered per-kind verdict queues for review-organ
   calls: each call of a kind consumes one entry (str content / dict message
   / CALLABLE over the request body — the S13 freshness hook both mutates
   the staged tree and returns the verdict), an exhausted or absent queue
   falls back to canned all-clean, `assert_consumed()` is the integrity
   gate, and review calls still never consume agent script steps (default-
   lane pins in both directions). `ScriptedStubModel(review_script=...)` →
   `scripted_completion(..., review_next=...)`, consulted ONLY for
   `REVIEW_KINDS`.
   (d) `keyless_reviewer_slots(advisory=True)` — the ONE optional advisory
   row pinned onto the stub slug, so the advisory pre-review runs its real
   bounded native inspection episode keyless instead of degrading to the
   audited-bypass compensation; the default form stays byte-identical.
3. E2E-находки wave-3a (runtime defects/observations — NOT fixed in this
   lane, per the lane rule):

   | id | surface | observation | evidence |
   |---|---|---|---|
   | W3A-F1 | advisory admission vs commit gate (P9 scope) | `release_metadata_preflight` hard-blocks ANY changed set without VERSION in scope — including doc-only diffs, which the commit gate deliberately exempts (`_diff_is_doc_only` carve, applied only to the tests preflight; the commit path runs no release-metadata preflight). Net effect on every install: a doc-only change can NEVER obtain a fresh advisory verdict — the standard preflight_review→commit_reviewed flow structurally degrades to the audited bypass for doc-only work. S13b reaches a real verdict by naming the UNCHANGED `VERSION` in `paths` (satisfies the scope check; carriers already consistent). Owner decision candidates: a doc-only carve in the admission, or an explicit typed "advisory not applicable to doc-only" outcome instead of PREFLIGHT_BLOCKED. | ouroboros/commit_admission.py:90-99 vs ouroboros/tools/git.py:614; observed live (doc-only diff → PREFLIGHT_BLOCKED naming BIBLE P9) |
   | W3A-F2 | tree state: README Version History over its own P9 limit | The checked-in README on this base carries 6 patch rows against the limit of 5 (`check_history_limit`), so EVERY VERSION-in-scope advisory admission on this tree is deterministically blocked before any review. The next release touching VERSION must trim the oldest patch entry anyway; until then advisory verdicts are unreachable on this checkout without the S13b fixture repair (the scenario trims the oldest patch row in its private clone and commits it). | ouroboros/tools/release_sync.py:576-605; README.md Version History (6.113.5..6.113.1 + 6.110.1); observed live (PREFLIGHT_BLOCKED "6 patch rows (limit 5)") |
   | W3A-F3 | advisory stale mark transience | `last_stale_from_edit_ts` is a transient repo-scoped mark: it is cleared once a LATER advisory run row lands (including the audited-bypass row a subsequent `skip_advisory_review=True` commit records), so a post-hoc reader cannot rely on it. The durable evidence of edit-staleness is the demoted run row (status="stale") plus the refusal text naming the edit; the scenario pins those. | ouroboros/review_state_model.py add_run/mark_all_stale_except; observed live in S13b (mark set at refusal time, empty in the final state) |
   | W3A-F4 | plan-review escalation event routing | `emit_review_cycles_exhausted` for surface=plan_review lands in the SERVER-level `logs/events.jsonl`, while the rest of the task's plan-review evidence (tool rows, wave artifacts via the task artifact store) is task-scoped — a reader scanning only the task drive misses the escalation row. Cheap disclosure, not necessarily a defect. | ouroboros/review_cycles.py:171-199; observed live in S11-cap |

4. DEFERRED (disclosed, wave-4 remainder): self-evolution absorb with
   kill-mid-absorb restart recovery (the instruction's optional item 4 — the
   evolution-campaign fixture is a full wave of its own), plus the carried
   wave-2 remainder (carrier-conflict / assisted-merge / crash-mid-phase
   update variations, chat-lineage cancel receipt, delegated-transport
   FakeClaudexorDaemon, gateway/UI-truth Playwright, skills lifecycle).
5. LANE BUDGET: full mock lane (S1-S14 + default pins, serial) — 43 passed
   in ~610s (10:09) on this host, inside the plan's 10-25 min PR keyless
   budget; wave 3a added 12 tests / ~6.5 min over wave 2's ~219s. The five
   new default-lane pins add well under 1s to the ordinary battery.

## Integration note for the F4 wave-3a section above (2026-09-01)

The wave-3a scenarios were RENUMBERED at integration: the parallel wave-3b
lane claimed S11-S13 first, so wave-3a's rows landed as S14 (plan review),
S15 (advisory class), S16 (blocking class + freshness), S17 (acceptance
loop). The wave-3a section text above says "S11-S14" — read it through this
mapping; test names and the scenario manifest carry the final numbers.
## From the hermetic class-fix lane (base 8827fd2c, 2026-09-01)

Issue #455 (confirmed by a LIVE repeat 2026-09-01 14:55 UTC): with all four
OUROBOROS_* env variables pointing at a temp root, update-merge tests still
wrote `managed_update_stash_restored (context=test)` /
`managed_update_wave_floor_refused` into the LIVE
`~/ouro/data/logs/supervisor.jsonl` (455 `context=test` lines accumulated, 56
on 2026-09-01 alone). Root: `supervisor/git_ops.py` bound
`REPO_DIR`/`DRIVE_ROOT` at import to hardcoded `Path.home()/"Ouroboros"/...`
— the one process-global root pair that NEVER read the env — and
`update_merge._log_supervisor` (update_merge.py:742), the update_candidate
stash writers (update_candidate.py:554-576) and the git_ops_reset `_go()`
writers all resolve supervisor.jsonl through it; `tests/conftest.py
_bind_pytest_runtime_roots` rebound config/state/queue/workers but not
`git_ops.DRIVE_ROOT`.

1. WRITER MAP (process-global data/repo roots, classified):

   | global | resolution | class |
   |---|---|---|
   | supervisor/git_ops.py REPO_DIR/DRIVE_ROOT | hardcoded home default, env never read, rebound only by init()/worker bind/monkeypatch | DEFECT (the proven live leak; fixed) |
   | ouroboros/config.py APP_ROOT/REPO_DIR/DATA_DIR/SETTINGS_PATH/... | env at import (documented four-env recipe) | import-cache; healthy under the whole-process recipe, stale under late isolation — mitigated by the conftest rebind list + the new fail-closed guard |
   | supervisor/state.py DRIVE_ROOT/STATE_PATH/... , queue.py DRIVE_ROOT, workers.py REPO_DIR/DRIVE_ROOT | Path(config.DATA_DIR) at import + init() rebind; conftest rebinds all three | import-cache; same mitigation, guard-covered |
   | ouroboros/server_process.py DATA_DIR | env at import (own copy) | import-cache; read for child-env assembly, guard-covered downstream |
   | ouroboros/tools/evolution_stats.py _REPO_DIR | env at import | import-cache (repo dir, read-only stats) |
   | ouroboros/tools/browser.py:336 | env per call | healthy |
   | hardcoded home paths in state.py:46/360, worker_process.py:166, server.py:117/1211, gateway/files.py:802, gateway/settings.py:1023 | comparison targets ("is this the live root?"), not writers | healthy by design |
   | ouroboros/packaged_cli.py PYTHONPYCACHEPREFIX | known sibling class, packaged launcher only | out of this lane's scope (already documented in AGENTS) |

2. CLASS FIX (both sides of the shared helper):
   - Resolution side: `config.resolve_app_root()/resolve_repo_dir()/
     resolve_data_dir()` — the per-call form of the SSOT constants (same
     precedence, two env reads + Path construction, no cache needed — file
     I/O dominates every consumer). `git_ops.REPO_DIR`/`DRIVE_ROOT` are now
     UNBOUND until init()/worker rebind/monkeypatch pins them; module
     `__getattr__` resolves un-pinned attribute access per call via the
     resolvers, and `current_repo_dir()/current_drive_root()` serve git_ops'
     internal uses. This fixes every `_g.DRIVE_ROOT`/`_go().DRIVE_ROOT`
     consumer at once (update_merge, update_candidate, update_merge_plan,
     update_recovery, update_source, git_ops_reset, evolution_lifecycle,
     gateway/control) with zero call-site edits outside git_ops, and covers
     consumers nobody thought of (AGENTS class-fix rule). Live server
     behavior unchanged: server.py:569 bootstrap still calls git_ops.init()
     which pins real attributes; worker `_bind_worker_repo_root` unchanged.
   - Guard side (the structural invariant): `assert_test_data_path` moved to
     the ouroboros/utils leaf (state.py re-exports it) and `append_jsonl` now
     calls it — the jsonl half of the durable-write plane was the unguarded
     half that let the leak land silently while `state.atomic_write_text` was
     already guarded. Outside pytest this is one env read per append.
   - conftest `_bind_pytest_runtime_roots` now also pins
     `git_ops.DRIVE_ROOT` (the one root the rebind list missed).
   - Leaking fixture fixed: `tests/test_update_merge_plan.py _point_at` now
     patches `DRIVE_ROOT` like its siblings in test_update_dirty_stash.py.

3. PINS (all six RED on base 8827fd2c, verified on a git-archive copy of the
   base tree — the E2E pin fails there with exactly
   `home/Ouroboros/data/logs/supervisor.jsonl` leaked):
   - tests/test_git_ops_default_roots.py (the D13 hook, previously named in
     the ledger but nonexistent): unpinned roots follow env PER CALL, pinned
     roots win, `_log_supervisor` lands in the env root, and the exact issue
     #455 subprocess repro (all four env set + throwaway HOME → the
     live-shaped root stays empty). D13 flipped pending→done.
   - tests/test_hermetic_data_root.py: append_jsonl fails closed on a
     live-root write (PYTEST_LIVE_DATA_WRITE_BLOCKED) + the serial E2E class
     pin — a real subprocess pytest run of test_update_dirty_stash.py +
     test_update_merge_plan.py under throwaway HOME + full env isolation with
     an empty pre/post inventory delta of `$HOME/Ouroboros/data` (the AGENTS
     `find -newermt` etalon as a regression test).

4. ISSUE #455 CLOSURE: the fix closes exactly the issue's class — "a
   process-global root that resolves before the isolation applies" — for the
   supervisor writer plane (the proven leak), and adds the fail-closed guard
   that turns ANY residual member (late in-process isolation of the
   import-cached config-derived globals, future writers) into a loud pytest
   failure instead of a silent live write. DISCLOSED RESIDUAL:
   `utils.atomic_write_json` and other non-jsonl writers outside
   state.atomic_write_text are not guard-wrapped (no observed leak path;
   candidates for the same one-line guard if one appears), and non-pytest
   drivers (bench harnesses) still rely on explicit rebinds per AGENTS.
5. LIVE PROOF + GATE COUNTERS (host 0897-oma, 2026-09-01, all pytest via
   ~/ouro/venv with isolated OUROBOROS_APP_ROOT/DATA_DIR mktemp roots;
   `git rev-parse HEAD` = 8827fd2c re-checked after EVERY pytest;
   `git diff --check` clean; `ruff check . --select F` clean):
   - Red repro pre-fix (throwaway HOME + all four OUROBOROS_* env): the write
     landed in the fake home root, not the isolated one. Post-fix: isolated
     root; a LATE env change is followed per call; init() pin honored; unpin
     returns to lazy.
   - All 6 new pins RED on a git-archive copy of base 8827fd2c (the E2E pin
     fails there with exactly home/Ouroboros/data/logs/supervisor.jsonl
     leaked); all GREEN on the fixed tree.
   - Gates: targeted update-merge/git_ops suites 161 passed (-n 8, not
     serial) + 35 passed (serial), both EXIT=0; full CI-shape battery
     (-n 16, 'not serial' composed WITH the default addopts exclusions) →
     13351 passed, 3 skipped, EXIT=0; full serial pass → 622 passed,
     19 skipped, EXIT=0; size_ratchet lane 5 passed EXIT=0. (A first battery
     attempt passed `-m 'not serial'` ALONE, which OVERRODE the default
     addopts exclusions and pulled the skill_smoke/browser lanes in — its 4
     fails/4 errors were lane-selection noise plus two real same-class test
     instances, both fixed below.)
   - LIVE class check: tight-window run of the exact leak reproducers
     (update_merge_plan / dirty_stash / merge_assisted / git_ops_managed)
     15:26:57-15:27:05 UTC → live supervisor.jsonl delta ZERO lines
     (context=test 456→456, total 899→899); the green serial rerun window
     (~15:33-15:38) likewise added ZERO context=test lines. The live log's
     context=test stream CONTINUES from other writers exactly as before this
     lane's gates (15:06/15:12/15:16 pre-gate; 15:25:30 and the 15:28
     cluster in-gate): concurrent NEIGHBOR pytest batteries on unfixed trees
     (~/ouro/subagent_worktrees/v7next and ~/ouro/worktrees/
     frontend-sprint-20260901 were running `pytest tests/` at those moments,
     recorded by PID+cwd) plus a foreign preflight runtime
     (/tmp/ouro-issue449-formal-r1-runtime, the `wave_floor_refused
     estimated_wave_usd=6.42` writer). A silent live write from THIS lane is
     structurally excluded: its runs had the append_jsonl guard armed, under
     which a live-root write is a LOUD red test, and the runs were green.
   - Collateral same-class instances the new guard exposed (fixed here):
     tests/test_v678_receipt_reconciliation.py and
     tests/test_v652_scratch_and_masking.py shaped their tmp harness as
     home/Ouroboros/data under a patched Path.home — moved to neutral tmp
     layouts (their registries already take explicit roots).
   - OBSERVED SIBLING-CLASS RESIDUAL (not fixed here, disclosed): fresh pyc
     mirrors landed in the LIVE ~/ouro/data/state/pycache/tmp/
     ouroboros-pytest-data-*/skills/ouroboroshub/{a2a,duckduckgo}/
     .ouroboros_env/... at 15:28:39-54 — a skill-host/exec child resolved
     PYTHONPYCACHEPREFIX (or its data dir) to the live root while importing
     isolated skill deps from a pytest session dir. A twin directory from
     2026-07-18 shows the mechanism predates this lane entirely (the known
     packaged_cli/PYTHONPYCACHEPREFIX sibling class from AGENTS). The
     skill_smoke lane that triggers it runs only outside the default CI
     shape. Follow-up candidate: the same lazy-resolution treatment for the
     skill-host spawn env.

## From the persistence mechanical train (№9=A, base d1276c5f, 2026-09-01)

The CPL4-C1..C23 candidate table (F5 lane B) executed as one lane. Scope =
the owner-sanctioned mechanical set (№9=A) PLUS the owner batch №8 rulings
delivered mid-lane (all A: 1A..7A). Single-intent commits, author Ouroboros;
docs/PERSISTENCE.md rows and the verify pin moved in the same commit as each
fix. Disposition per row:

| id | disposition | mechanism |
|---|---|---|
| CPL4-C1 | fixed (rotation train) | events.jsonl rotates on the supervisor tick AFTER its custody readers went chain-aware: `utils.jsonl_chain_handles` (open-live-first + inode dedup, rotation-race-safe) feeds `delegate_custody._iter_rows` (full replay + chain-windowed fault tail), strict `complete_custody_rows` (unopenable segment = incomplete view), `custody_log_unreadable` probes the chain, the settled-terminal cursor became a monotonic CHAIN offset (immutable archive ⇒ monotonic; torn archive line consumed, torn live line waits), legacy usage import snapshots the chain, the swarm-fanout rollup reads `iter_jsonl_chain_objects`, worker boot verify falls back to the newest segment on a rotated offset. 8 MB live tripwire (rotation-regression) + 100 MB chain watch inherit the old signals |
| CPL4-C2 | fixed (rotation train) | tools.jsonl rotates on the same tick; api_logs_tail/task_events were already archive-chain-aware; tripwire re-texted to rotation-regression |
| CPL4-C3 | fixed (rotation train) | supervisor.jsonl rotates + gains its tripwire row (`SUPERVISOR_LOG_WARN_BYTES`); `memory.read_jsonl_tail` (the tail-200 context read) backfills from newest archive segments |
| CPL4-C4 | fixed (rotation train) | task_reflections.jsonl rotates + gains its tripwire; the tail-20 read backfills; project-scoped copies stay under project retention (never age-pruned) |
| CPL4-C5 | fixed | launcher pipe-copy thread rotates agent_stdout.log at 2 MB × `.1..3` backups (the server.log RotatingFileHandler bound), rotation failure never kills the copy thread |
| CPL4-C6 | NOT touched (owner 1A) | monetary usage-ledger compaction is excised to its own reviewed lane — monetary authority; the 20 MB WARN and quarantine semantics stand |
| CPL4-C7 | fixed (half pre-landed) | the consumed-once receipt prune already landed with the scheduler tick (`prune_consumed_once_records` + GC retention, found in-tree at base); this lane authored the missing `schema_version` stamp at `_write_scheduled_tasks` and trued up the tripwire text |
| CPL4-C8 | fixed | `_store_evidence` drops expired probe keys on write: failed/unprobeable past `_FAILED_TTL_SEC`, confirmed past GC retention; owner acks never expire, in-retention stale confirmed survives (blip-keep invariant pinned), unreadable ts/unknown status kept fail-closed |
| CPL4-C9 | fixed (owner 2A) | reader surface retired: `inject_crash_report`, the CRITICAL crash-rollback health line and their stickiness tests removed; no writer existed in this tree, stale files inert; PERSISTENCE row retired, scan pin moved |
| CPL4-C10 | fixed | `_schema_version: 1` (ABI-2 `with_schema_version`) on all six owner-state writes; review_job gained the one `_write_review_job` seam so merge writers stamp; grants re-stamp at write (its read normalizes keys away); readers keep legacy-0 tolerance, no read-time retrofits; content_hash never covers state files so no verdict/grant staled |
| CPL4-C11 | fixed (owner 3A) | hub uninstalls write a stamped `uninstalled.json` tombstone; the startup sweep clears dead state BY the mark keeping `grants.json` (owner authority) + the tombstone; reinstall self-heals; tombstone filename joined the owner-state forgery allowlist; gateway local delete (whole-dir removal) untouched |
| CPL4-C12 | fixed (bounded reads; rotation declined) | every review_history reader windows the 4 MB tail (`find_history_job_bounded` idiom incl. `load_history` and the idempotent terminal dedup scan); counters stay exact inside the window because lifecycle terminal rows persist their ordinals and `normalize_history` takes max(stored, derived) — a group aged past the window under-counts, never over-blocks (disclosed). Per-skill archive rotation NOT taken: no per-skill archive plane exists and the bounded reads land the read-cost half; retention stays unbounded-accepted with disclosure |
| CPL4-C13 | fixed | terminal+age startup sweep beside the custody sweep: terminal-status recovery rows (vetoed/adopted) past GC retention, transactions no surviving row references (skipped wholesale if ANY recovery row unreadable), supervision files of SETTLED tasks; `active.json` never; unreadable custody log skips the sweep (the `_prune_delegated_snapshots` idiom) |
| CPL4-C14 | fixed | code_intel roots age-prune by inventory.json mtime past GC retention (pure cache; the root key is a one-way hash so mtime IS the liveness signal) |
| CPL4-C15 | fixed | `failed/` reconcile markers age-prune past GC retention (failure fact already durable in events.jsonl) |
| CPL4-C16 | fixed (owner 4A) | identity/knowledge/patterns journal entries older than GC retention become digest-only (sha256+len, existing hashes never overwritten, `content_digested` mark) under the append lock; unparseable lines and ts-less rows byte-preserved; scratchpad journal and the observation inbox out of scope |
| CPL4-C17 | fixed | both knowledge_history writers + knowledge_journal now append through the sidecar-locked `append_jsonl` seam (last raw `open("a")` journals) |
| CPL4-C18 | fixed | startup sweep unlinks mailboxes (+acks) of tasks with a SETTLED durable result; no result / non-terminal / unclassifiable name keeps them fail-closed; reuses `cleanup_task_mailbox` |
| CPL4-C19 | recorded (owner 5A) | task_results stay eternal for 7.0 deliberately — ratified in PERSISTENCE + ADOPTION; any future prune needs a fresh owner decision |
| CPL4-C20 | fixed | the data-root `tmp_scripts` fallback's `script_*` hard-kill orphans joined `sweep_stale_temp_files` scope (top-level dir only, same age guard, startup-only); task-drive copies stay owned by the drive prune |
| CPL4-C21 | fixed (owner 6A) | `uploads/screenshots` + `uploads/views` (agent-generated) age out past GC retention at startup; owner attachments in the uploads/ root untouched (owner-explicit delete only); readers already skip missing |
| CPL4-C22 | fixed (owner 7A) | `OUROBOROS_OBSERVABILITY_RETENTION_DAYS` removed entirely (parse/clamp/report); `prune_observability_blobs` is now the honest startup census; key added to `RETIRED_SETTING_KEYS` (ghost drop on load); ARCHITECTURE env row removed; preserve-indefinitely contract stands |
| CPL4-C23 | fixed | acknowledged observations older than GC retention fold — with their ack rows — into `archive/consciousness_observations_<ts>.jsonl` at startup under the store's writer lock; unACKed rows NEVER pruned; any malformed line / ghost ack skips the whole fold; archive written before the live rewrite (crash duplicates into forensic history, never loses) |

Cross-cutting disclosures:

- **Verify pin**: `tests/test_persistence_inventory.py` EXPECTED_SCAN_PATHS
  118 → 123 across the lane (C13 +2 dir enumerations, C14/C15 +2, C16 +2
  canonical journal paths, C9 −1 retired plane), each move in the same commit
  as its rows.
- **Size ratchet**: utils.py (1451→1550), launcher.py (1500→1536) and
  agent_startup_checks.py (1487→1515) left the tracked 1001–1500 band into
  the untracked 1501–1600 zone; the regenerator retired their band
  rationales and `--check` is green. The chain helpers stay in utils beside
  the jsonl family (splitting the jsonl seam would scatter it); memory.py
  was shaved back under 1000 instead of banding.
- **C12 residual**: direct (non-lifecycle) `append_history` rows still do
  not persist ordinals; only a group whose newest ordinal-bearing row aged
  past the 4 MB window under-counts. Named, not hidden.
- **Worker-boot verify residual (C1)**: on the rare rotated-offset fallback
  the newest-segment scan can surface a PREVIOUS generation's boot event
  (worst case: a spurious sha-verify supervisor row); the pre-train behavior
  on that race was a spurious timeout instead.
- **C13/C18 residual**: state files of tasks that never wrote a durable
  result are kept forever by design (fail-closed beats leak-free).
- **Owner batch №8** (2026-09-01, all A): 1A C6 excised to its own lane;
  2A C9 retire reader; 3A C11 tombstone with grants preserved; 4A C16
  digest-only past GC retention; 5A C19 recorded eternal; 6A C21 agent
  media only; 7A C22 knob removed with `RETIRED_SETTING_KEYS` idiom.
  prompts/SYSTEM.md's combined "CRASH ROLLBACK / RESCUE SNAPSHOT" line was
  left untouched (rescue-snapshot half still live; prompts are outside this
  lane's mechanical scope).

## From the F4 wave 4 (update variants + interactive + W2-F2, base 74a03082)

Tail scenario wave on the lane-1/2/3 skeleton — the carried update variations,
the chat-lineage cancel receipt (the W2-F1 counterpart), the absorb
kill-recovery remainder and the delegated interactive answer — plus the ONE
sanctioned runtime fix of the wave, W2-F2. Manifest rows S18-S23, all keyless
mock-lane, every scenario green on this host. Commits: 68b19a61 (W2-F2),
4e68526c (scenarios), 0196b3d7 (facade-inventory regen tail).

1. W2-F2 FIXED (owner sanction 2026-09-01 batch 7, №4=A; commit 68b19a61).
   The update-fetch path unconditionally repinned the managed remote to the
   hardcoded official URL (`ensure_official_update_remote`), silently
   retargeting fork/mirror/air-gap installs whose `managed_remote_url` both
   bootstraps write into `.git/ouroboros-managed.json`. Class fix at the one
   shared helper: NEW `git_ops_updates.managed_update_remote_url(meta=None)`
   resolves the configured source (else the official default) and every repin
   — set-url, add, and `compute_managed_update_status`'s `official_repo_url`
   fallback — goes through it. No configured source = byte-identical former
   default. Pins: unit (the configured source survives N repin cycles; blank/
   absent config keeps the official URL), the module-handle extraction table
   and the owner-facade identity list extended, and S9 RESHAPED to the
   fork-install form the fix serves: the local upstream is now CONFIGURED as
   `managed_remote_url` (the former `url.<mirror>.insteadOf` workaround is
   gone), the insteadOf config now redirects the OFFICIAL URL to a
   non-existent path (a REGRESSED repin fails loudly with zero network
   egress — proven red pre-fix on the base tree), and the scenario asserts
   `git remote get-url managed` == the configured source AFTER the full
   update cycle. ARCHITECTURE §8 updated same-commit; the generated
   FACADE_INVENTORY row followed in the 0196b3d7 tail (caught by the
   byte-identity pin in the CI battery — the first battery run failed exactly
   there, 13430/13431 otherwise green).
2. SCENARIOS LANDED (tests/system_e2e/test_system_scenarios_w4.py):
   - S18 update carrier path (~22s solo): a diverged fork (local commit
     pinning VERSION=0.0.1) takes an official VERSION=9.9.9 bump through a
     configured-source managed install — the merge conflicts EXACTLY inside
     the declared version-carrier span. Pinned: plan kind=clean with
     `carrier_resolved_paths == ["VERSION"]` and empty conflict inventories
     (the carrier engine resolved the span to the official side in the
     ISOLATED planner), the apply lands a REAL 2-parent merge commit
     (`HEAD^1` == fork base, `HEAD^2` == official target), boot-finalize is
     honest (`managed_update_finalized.head` == merge commit, tx consumed),
     and THE CARRIER TRANSFER: VERSION, pyproject, the README badge and the
     ARCHITECTURE header all name 9.9.9 and the tree's own
     `check_worktree_version_sync` reports ZERO desyncs — the Q8 projection
     moved every mechanical carrier token, none drifted. The W2-F2 configured
     source survives the cycle.
   - S19 update conflicting → typed refusal (~16s solo): a genuine code
     conflict (both sides rewrite Makefile) routes the smart apply to the
     assisted lane, whose admission refuses TYPED on an exhausted budget —
     seeded HONESTLY through the tree's own monetary writers
     (reserve→dispatch→settle at exactly the configured TOTAL_BUDGET=10.0;
     no forged ledger rows). Pinned: preflight kind=conflicting with
     code_conflict_paths=["Makefile"] and recommended_strategy=assisted; the
     409 names the budget and "nothing was changed"; and TREE INTACT as byte
     truth — the FULL worktree fingerprint (dirty tracked edit + untracked
     file included) is IDENTICAL before/after, HEAD/branch unmoved, no
     MERGE_HEAD, no tx marker, no assisted resolver task anywhere in the
     durable queue, server healthy after the writer fence reopened.
   - S20 crash mid-apply (~4s solo, subprocess driver on a second real
     install — the S10 idiom): three honest boot-finalize outcomes. (a) crash
     between the durable `stashing_local_work` marker and the `stash_sha`
     write → boot finds the stash BY ATTEMPT ID (`lookup_update_stash`),
     restores the owner's work UNCOMMITTED, clears the marker, HEAD untouched,
     one `managed_update_stash_recovered_on_boot` receipt; (b) a half-written
     `pending_boot_smoke` tx whose merge commit never reached HEAD → honest
     typed rollback (`managed_update_rollback_after_failed_boot` +
     `managed_update_rolled_back` naming the exact pre_update_sha, tree clean
     at pre) — and NO junk `failed-update-*` ref is minted for a non-attempt
     (the guard that keeps a REPLAYED rollback from clobbering a real
     preserved attempt, pinned green); (c) merge applied + crash before the
     restart smoke → boot RUNS the smoke, finalizes
     (`managed_update_finalized.head` == the applied merge), restores the
     stashed dirty work, clears the tx — the applied update survives.
   - S21 chat-lineage cancel (~12s solo): the ONLY input delta vs S7 is
     `chat_id: 1` on the POST /api/tasks body, and it flips the owed-answer
     accounting from the `no_lineage_chat` handoff row (asserted ABSENT here)
     to the real thing: the outbox delivery `cancel:<task>:<request_id>`
     (the id derived from the intent forensics' own requested row) registered
     AND drained to `delivered` with NO WS client ever connected, the
     `cancel_receipt` chat row durably in logs/chat.jsonl
     (direction=system, chat_id=1, the cancellation text naming the task),
     and the `cancel_receipt` block on the stored result
     (settled_status/outcome=cancelled, the exact delivery_id, the salvage
     disclosure) — same requested→claimed→settled forensic trail as S7.
   - S22 absorb kill-recovery (~95s solo, Linux-only, three generations): a
     REAL evolution cycle on a private clone — campaign seeded, the
     supervisor mints the evolution task itself, the scripted agent lands a
     reviewed commit through the BLOCKING triad+scope organ (stub panel), the
     campaign records the receipt and settles into `waiting_for_restart` —
     then the whole tree is SIGKILLed in that crash window.
     `OUROBOROS_EVOLUTION_AUTO_RESTART=false` makes the window STABLE and
     the crash state exact (a real crash between the campaign write and the
     restart-verify marker leaves precisely this durable shape: commit on
     HEAD, open transaction, NO marker). Generation B (same clone+data root;
     the owner's evolution toggle flipped off so no cycle-2 races the
     assertions — the boot reconcile deliberately ignores that flag) absorbs
     the cycle through the MARKERLESS dangling-transaction reconciliation:
     absorbed_cycles_done == 1, the history row carries
     verified_by=boot_reconciliation and the exact commit_sha, the commit is
     still on HEAD (no loss), `evolution_tx_reconciled` in events.jsonl,
     EXACTLY ONE cycle_outcome=absorbed checkpoint row. Generation C boots
     and re-proves NOTHING absorbs twice: count still 1, single history row
     for the transaction id, HEAD unmoved, no marker files ever created.
   - S23 delegated interactive answer (~25s solo): a `[FAKE:ASK]` run pauses
     on a pending interaction; `delegate_wait` returns IMMEDIATELY as
     `waiting_on_user` with the full question set; the scripted nanny answers
     through `delegate_answer` (run/interaction/question ids parsed from the
     transcript the model actually saw); the run resumes and settles. Pinned:
     the ORDERED custody chain (STARTED < `delegate_interaction_answered`
     status=delivered/questions_answered=1 < SETTLED state=succeeded, one run
     id), NO cancel row (a run that merely asked is never torn down), ONE
     physical POST /v2/runs across the whole episode, the exact camelCase
     wire on the exact path
     (`/v2/runs/<id>/interactions/<iid>/answer`,
     `{"answers":[{"questionId","selectedLabels","freeText"}]}`), the
     transcript truth (the pause, the question text, the delivered relay,
     the terminal result all reached the model), and the honest
     last-delegation receipt settled exactly once.
3. HARNESS DELTAS: FakeClaudexorDaemon (tests/system_e2e/interfaces.py) gained
   the MINIMAL interactive surface — a `[FAKE:ASK]` prompt marker seeds one
   `ControlPendingInteraction`-shaped row; the run detail surfaces
   `pendingInteractions` + `summary.waitingOnUser` and STAYS running until the
   answer verb clears the row (the next poll then flips terminal exactly as
   before); `POST /v2/runs/:id/interactions/:iid/answer` answers the typed
   statuses at their real HTTP codes (200 delivered / 409 already_resolved /
   400 rejected — free contract fidelity: the client accepts a typed status
   at ANY code). A default-lane pin drives the surface with the REAL
   ClaudexorGateway (`pending_interactions` normalization reads the row
   whole; refusals never consume the question; the delivered answer resumes
   the run; a late duplicate is already_resolved, never a re-run). Manifest
   rows S18-S23; the new module carries its size-ratchet band rationale.
4. E2E-находки w4 (runtime defects/observations — NOT fixed in this lane,
   per the lane rule; W2-F2 was the one sanctioned fix):

   | id | surface | observation | evidence |
   |---|---|---|---|
   | W4-F1 | evolution absorb, commit-vs-receipt crash window | A crash between the reviewed `git commit` and `record_evolution_commit` leaves a landed reviewed commit that NO boot path will ever attribute: the markerless reconcile short-circuits on an empty `commit_sha` (it only stamps the generation), and `_preserve_evolution_orphan` runs only on the AUTHORITY-REFUSAL path, never on a crash. The commit sits on HEAD forever — not lost as code, but the cycle never resolves and is never counted. | ouroboros/tools/git.py:1411-1436 (commit → receipt order); ouroboros/tools/git_evolution.py:272-330; ouroboros/agent_startup_checks.py:1130-1142 (`if not commit_sha … return`) |
   | W4-F2 | absorb outcome ledger atomicity | The campaign absorb write and the `cycle_outcome` checkpoint append are NOT one transaction: the reconcile writes the campaign under `update_json_locked` and appends the checkpoint row AFTER the lock (same shape on the claim path). A crash in between yields a campaign that says `absorbed` with no `cycle_outcome` row — `build_solve_capability_digest` then under-reports the cycle forever, and nothing re-derives the row. | ouroboros/agent_startup_checks.py:1227-1243 (locked `update_json_locked` → post-lock `_append_cycle_outcome_tag`); S22 asserts the row IS written on the clean path |
   | W4-F3 | evolution restart marker vs manual restarts | `request_evolution_restart` returns BEFORE writing `pending_restart_verify.json` when `OUROBOROS_EVOLUTION_AUTO_RESTART` is off — the exact-claim verify path (campaign∧transaction∧task∧commit authority, `require_claim=True`) is structurally unreachable for installs that restart manually; absorb attribution then rests wholly on the weaker markerless reconciliation. Deliberate-looking (the knob predates the claim machinery), named so the asymmetry is a decision, not an accident. S22 exploits exactly this to make its crash window deterministic. | supervisor/evolution_lifecycle.py:1437-1438 (early return before the marker write at :1457; re-anchored on rc.9 c1a4b2bc by the F3-C lane — the F2 relocation moved it from :1362-1368) |
   | W4-F4 | rescue-local ref accumulation | `create_rescue_local_ref` pins every update stash to a durable `rescue-local-<stash12>` branch and NOTHING ever deletes them — a refused/unwound attempt (S19's shape) leaves its ref behind exactly like a successful one. Deliberate durability ("git-gc can never lose the owner's work"); the unbounded per-distinct-stash accumulation is the disclosed cost. | supervisor/update_merge.py:294-305 (re-anchored on rc.9 c1a4b2bc by the F3-C lane; was :293-304); no deletion call site in the update flow (`rg rescue-local`) |

5. LANE BUDGET + GATE COUNTERS (host 0897-oma, 2026-09-01; every pytest via
   ~/ouro/venv with isolated OUROBOROS_APP_ROOT/DATA_DIR mktemp roots;
   `git rev-parse HEAD` re-checked after EVERY pytest; `git diff --check`
   clean; `ruff check . --select F` clean): full mock lane (S1-S23 + default
   pins, serial) — 56 passed in ~833s (13:52), inside the wave's ≤18 min
   budget (wave 4 added ~175s of scenarios over the integrated S1-S17 lane);
   solo timings S18 ~22s, S19 ~16s, S20 ~4s, S21 ~12s, S22 ~95s, S23 ~25s;
   the new default-lane pin adds ~1s to the ordinary battery. CI-shape
   non-serial battery (-n 16 loadscope): first run 13430 passed / 1 failed —
   the facade-inventory byte-identity pin catching W2-F2's new re-export
   (fixed by regeneration, 0196b3d7); clean rerun 13431 passed, 3 skipped,
   EXIT=0. Serial pass: 622 passed, 19 skipped, EXIT=0. size_ratchet lane:
   5 passed, EXIT=0.
6. DEFERRED (disclosed): the gateway/UI-truth (Playwright) wave — a separate
   browser-bearing lane by instruction; the claim-file absorb crash windows
   (dead-claim reclaim / stale-claim-after-absorb) stay unit-covered by
   tests/test_evolution_restart_claims.py — the E2E covers the markerless
   window honestly instead of forging claim files on the real drive;
   delegated MUTATING-run scenarios (snapshot provisioning +
   integrate_delegated_patch + containment evidence) carried from wave 3b.

## Q-F6-1 provenance investigation — findings (owner batch 7 №1, 2026-09-01)

Commissioned by the owner («изучи подробнее по чатам тебя и кодекса, отправь
скаут субагента opus-ов и перепроверь их результаты»). Two opus scouts
(upstream provenance; transcript provenance) — every load-bearing claim below
was re-verified by the coordinator against git objects, the GitHub API and the
plan files.

1. **Origin.** `_EXIT_CODE_RE` / `_SIGNAL_RE` are as old as the repository:
   present in the initial app-bundle commit `6700358a` (2026-04-22,
   loop_tool_execution.py:42-43). For four months the regex harvest over the
   rendered result text was the ONLY source of `exit_code`/`signal` facts.
   What appeared in the drift window was not the fallback but the TYPED
   channel (`ouroboros/tools/process_facts.py`, commit `55af051e`, PR #404
   «node-runtime sprint», merged 2026-08-30 20:01Z), which demoted the regex
   to a documented fallback («read-fallback for records that lack typed meta»).
2. **How it passed review — it was BORN of review.** The sprint's plan
   decision D6 (owner «6. A») said «regex-механика остаётся»; the plan roast
   reversed it: fable-5 finding #5 MAJOR («D7 classification must not hang on
   string matching — BIBLE P5») + grok-4.6 #6/#7 MAJOR. Disposition R5 in
   `~/.claude/plans/node-runtime-sprint/PLAN.md:199`: typed channel for
   run_command/run_script; «regex остаётся только read-легаси для старых
   записей». Five later internal waves (triad+scope 12 runs, two delta rounds,
   full-scope sol) raised no objection to the retained fallback; the
   adversarial reviewer of stream B caught «false red from prose» → typed facts
   given precedence over the whole key family (`705ffc51`), fallback kept.
   GitHub PR #404: 0 reviews, 0 comments; heavy CI jobs skipped — all review
   authority was internal.
3. **Two lines that did not know each other.** The v7 lane retired both
   regexes on 2026-08-19 (`5440e407`, delta id D02: «the exact scrape a
   producer-controlled stdout line could forge»), eleven days BEFORE PR #404
   resurrected them as a fallback on mainline. The F6 sync collided the two
   decisions; owner №1=A keeps the campaign contract (typed facts only).
4. **Deeper gap the fallback masked.** On the upstream tip the regex ran
   unconditionally for EVERY tool (a `signal=SIGKILL` inside a read_file
   result forged a fact), and `run_command` status was recomputed on the
   regex value BEFORE the typed merge (loop_tool_execution.py:629 vs :771).
   Typed facts existed for exactly two tools. On this tree (regex gone) the
   honest remaining «no fact» cases are: extension-child death signals
   (extension_dispatch stamps typed codes but no exit_code/signal — closed by
   deletion, not by typing), skill_exec/skill_preflight (preflight synthesizes
   `-9` on timeout), verify_and_record (renders `exit=`, never stamped into
   result_meta), timeout/pre-exec failures of run_command (by construction),
   and Windows kills (structurally invisible to the POSIX signal partition).
   The node-runtime sprint's unpublished issue drafts (`issue_drafts.md`
   Issue 1-3) name the same stragglers.
5. **Disposition.** №1=A stands. The typed-producer gaps in item 4 are a
   candidate post-release train («typed process facts for extension children,
   preflight and verify»), to be batched to the owner; the unpublished issue
   drafts of the node-runtime sprint are surfaced to the owner separately.
   Lesson recorded for AGENTS (proposal pending owner «ok»): before keeping a
   legacy mechanism as a fallback, `git log --all -S<identifier>` — a parallel
   line may already have retired it with a stated reason.
## From the persistence corrective lane (audits #14/#15, base 3e4a6181)

The daemon audits found the mechanical persistence train (base d1276c5f)
landed with real defects around it. This lane is the corrective pass over
audit #15 findings 7 and 11-14 and audit #14 findings 5-6 — no new
persistence policy, only the guards the landed policy assumed it had. Every
fix is red-first pinned; each is one commit, author Ouroboros.

| finding | disposition | mechanism |
|---|---|---|
| #15-11 (CRITICAL, journal compactor) | ACCEPTED, fixed | Three guards on the one sweep that destroys CONTENT. (a) `_digest_row` used `setdefault`, so a stored `*_sha256`/`*_len` that CONTRADICTED the text was kept while the text was deleted — the lie became the whole record. A row whose stored fact disagrees with its text now keeps its FULL content and is reported as a typed `digest_mismatch` on the existing `memory_journal_compaction` event. (b) The append lock is taken `owner_aware_stale=True`, so elapsed time alone can never hand a live journal to a second writer. (c) The rewrite streams line by line into the temp sibling, and the source is re-identified (bytes consumed + device + inode) immediately before `os.replace`; any delta abandons the rewrite and leaves the appender's file. Pins: false digest (both `_sha256` and `_len` shapes) → text untouched + typed fact; a concurrent unlocked append → the row survives whichever branch runs; a source swapped for a different inode → nothing published, `source_changed` reported, no temp left behind; whole-file readers poisoned → compaction still works; source pin on the owner-aware lock |
| #15-12a (append short-write) | ACCEPTED, fixed | `append_jsonl` issued one bare `os.write` and returned success without checking the byte count — the class the atomic writers closed in f772717c, left in the append SSOT every authority JSONL stream goes through, where a torn line is a LOST RECORD (not merely a truncated file). Both lanes now share `_write_fd_fully`. A failure MID-record returns False instead of replaying the whole line over the prefix already on disk (only the open is retried); the next append's `ensure_record_boundary` starts a clean record. The SAME commit removes the second duplicate in that function: the non-required lane hand-rolled its own `O_CREAT\|O_EXCL` + age-reclaim loop (the duplicate DEVELOPMENT.md tells feature code not to write), and that copy was NOT owner-aware — a high-volume appender could delete the lockfile of a LIVE holder, and the memory-journal compactor rewrites a journal under exactly this lock. Both lanes take the shared owner-aware primitive and release through it; a live holder is waited out and the non-required lane then appends unlocked, as before. Pins: one-byte-at-a-time `os.write` → whole record lands; die-after-4-bytes → `False`, exactly two write attempts, no whole-record replay |
| #15-12b / #14-6a (chain enumeration) | ACCEPTED, fixed — and the audit's evidence line was WEAKER than the defect | The audit quoted `jsonl_archive_segments`' `except OSError: return []`. That except was in fact nearly dead: `Path.glob` SWALLOWS a `PermissionError` on the archive directory and yields nothing, so an unreadable archive reached every reader as "this store never rotated" without any exception at all. Enumeration is now an explicit `scandir`, and `strict=True` raises the typed `JsonlChainUnreadable` from `jsonl_archive_segments`/`jsonl_chain_handles`. Strict callers = the authority readers: the legacy usage import (money — its `except OSError` → `UsageAccountingError` was written expecting exactly this and never received it), `complete_custody_rows` and `custody_log_unreadable`. Fail-soft (documented, unchanged): memory tail backfill, the settled-terminal cursor, worker-boot probe, the swarm-fanout rollup, `delegate_custody._iter_rows` (whose strict sibling IS `custody_log_unreadable`). `complete_custody_rows` also drops its hand-rolled duplicate of the chain traversal for the shared helper: one chain SSOT, −30 lines |
| #15-12c (ATIF) | ACCEPTED, fixed | `build_trajectory` read `events.jsonl`/`tools.jsonl` as single from-birth files while chat/progress already used `_read_jsonl_chain`; after the C1/C2 train those two rotate too, so a rotated trial published a trajectory missing its early tool calls and its usage/startup events — a FALSE trajectory. Both go through the chain reader. Pin: rotated tools+events → both calls present in order, tokens summed across the chain, the agent version read from the archived startup row |
| #15-13a (media prune symlinks) | ACCEPTED, fixed — defect reproduced | `is_file()`/`stat()` follow symlinks, so the C21 age sweep would unlink old files wherever a link pointed. The pre-fix pin proves it: the sweep deleted a file OUTSIDE the drive root. Only REGULAR files inside the real family directory are unlinked now (`lstat` + `S_ISREG`, one stat per entry); a symlinked `uploads/screenshots`/`uploads/views` and any non-regular entry are counted `skipped` and surfaced on the event |
| #15-13b (reconcile GC premise) | ACCEPTED, fixed | The C15 GC was chosen on the stated premise that "the failure fact is already durable in events.jsonl". It was not — `_mark_failed` wrote the marker and nothing else, so the age prune destroyed the only record of why an extension gave up. The terminal failure (skill, request id, reason, source, attempts, last error) is appended to the event log BEFORE the `failed/` marker is written, so the marker is genuinely a cache of a fact that outlives it. Pin: five failing attempts → exactly one `extension_reconcile_failed` row; the marker aged out and pruned; the fact still readable |
| #14-5 (Windows O_BINARY) | ACCEPTED, fixed — and BROADER than the audit or ledger item 15 said | Both parties were half right. Ledger item 15 rejected replaying the frozen reference's `O_BINARY` into `write_text_atomic`'s fsync path, correctly: that fixes ONE lane. What neither the reference, the audit, nor item 15 noticed is that the OTHER lane translated too — `Path.write_text` opens in text mode, so the non-fsync path rewrote newlines on Windows exactly the same way. "Platform newline semantics" was therefore not a decomposition anyone had chosen for a caller; it was an unexamined default on both halves. Caller survey: `atomic_write_json` (all durable JSON state), `write_text` , outcome receipts, reviewer-slot projections, benchmark `run_manifest.json` (byte-compared by `tests/test_devtools_benchmarks.py`), and `tools/core.py`'s `write_file`/`edit_file`, which round-trip source Python read back with universal newlines and would have re-saved LF files as CRLF. NOT ONE of them wants translation. The complete class fix is the decomposition upstream already built: `write_text_atomic` is now `write_bytes_atomic` plus a UTF-8 encode — one full-write loop, one flag set, byte-exact everywhere. DEVELOPMENT.md's shared-helper paragraph updated in the same commit. Pins: exact bytes on both lanes; the fsync lane's `os.open` flags carry `O_BINARY` under a simulated Windows `os` (POSIX has no translation to observe); `atomic_write_json` byte pin |
| #14-6b (unbounded history scan) | ACCEPTED, fixed | `load_history` documents "every reader is byte-bounded" (CPL4-C12) while `count_paid_skill_review_cycles` still scanned EVERY installed skill's `review_history.jsonl` whole — the read where the cost multiplies by the number of skills, and the one that made the claim false. The bound stays where it lives: `skill_review_history.iter_history_rows_bounded` (raw rows, same tail window) and the cycles module carries no window size. Same disclosed residual as the family — a group whose newest ordinal-bearing row aged past the window under-counts, never over-blocks — and the docstring now NAMES this reader so the claim is checkable rather than merely asserted. Pin: with the window shrunk, ancient paid rows fall out of the cross-skill count |
| #14-6c (rotation by size) | ACCEPTED, fixed — defect reproduced | Worker-boot verification decided "rotated" from `old_offset > new_size`. Under a busy supervisor the fresh live file has usually already grown past the old offset by the time the verify reads, so the rotation went unnoticed and the read seeked into a DIFFERENT file at a meaningless offset. Reproduced on the pre-fix tree: the boot lookup returns `None`. The cursor is now `(size, device, inode)` from `events_log_cursor()`; a moved identity reads the MATCHING archive segment from the same offset (exact continuation), a cursor whose file is gone falls back to the newest segment whole, and a truncated same-inode file still resets. Both capture sites (pool spawn, assisted-resolver boot wait) pass the cursor; the new re-export regenerated `FACADE_INVENTORY.md` in the same commit |
| #15-14 (ARCHITECTURE same-commit) | ACCEPTED, fixed for this lane's scope | Ownership/data-flow rows added for `delegate_state_sweep.py`, `memory_journal_compaction.py` and `skill_uninstall_state.py`: what each OWNS (which durable paths), what it removes and on what proof, where it fails closed. No absolutes. `scripts/regenerate_inventories.py --check` green. NOT covered here: `usage_compaction.py` belongs to the C6 lane and must carry its own row with it |
| #15-7 (CPL-4 status) | ACCEPTED, fixed | `CPL-4` `done` → `in-progress`, with honest text: the mechanical train landed, this corrective lane landed the defects it left, and C6 runs in a separate reviewed lane and is NOT integrated. The row becomes `done` only after C6 integrates with a green hook — status must not run ahead of the work. `scripts/v7next_adoption.py` OK |

Cross-cutting notes:

- **One bug-cementing test removed.** `tests/test_memory_journal_compaction.py`
  asserted `digested["old_sha256"] == "pinned-old-hash"` — i.e. it pinned
  that a digest known to contradict its own text survives the deletion of
  that text. That is the #15-11 defect stated as intent, with a convincing
  comment ("existing hash never overwritten") for the next reader. The row
  is reshaped to a truthful stored digest and the false-fact case is now its
  own pin.
- **`append_jsonl` failure semantics changed, narrowly.** A mid-record write
  failure now returns `False` immediately instead of retrying the record and
  then falling through to the text-mode fallback. Retrying a partially
  landed record duplicates its prefix; open failures still retry three times
  and still fall through. Callers already had to handle `False` (that is why
  the helper returns a bool).
- **Not taken, deliberately.** `rotate_jsonl_log_if_needed` takes the same
  sidecar lock without `owner_aware_stale`. Left as is: rotation is an
  atomic rename, so a stolen lock cannot LOSE a row (a racing appender's
  bytes land in the renamed inode, which the chain readers read). The
  journal compactor is different in kind — it rewrites the file — which is
  why the owner-aware lock went there.
- **Residual named.** `iter_jsonl_chain_objects` has no `strict` parameter:
  its only runtime caller is the swarm-fanout rollup (observability). Adding
  an unused knob would be surface without a caller; the strict path exists
  where an authority reader asks for it.
- **Size ratchet, and why the lane's local history was rebuilt.** The chain
  work took `ouroboros/utils.py` from 1560 to 1629 lines — over the 1600 hard
  cap. The paydown is the append_jsonl lock dedup above: a real removal of a
  duplicated primitive in the same function the lane was already fixing, not
  a helper split and not comment golf. Because a commit that exceeds the cap
  is condemned on the first-parent line forever (a linear repair descendant
  does not heal it), the lock dedup was folded into the FIRST commit and the
  lane's LOCAL, unpushed chain was rebuilt so no commit ever crossed the cap;
  the byte-exact commit was ordered before the chain commit for the same
  reason. Per-commit sizes on the rebuilt chain: 1546, 1545, then 1599 to the
  tip. `regenerate_size_ratchet.py --check` green, `-m size_ratchet` green.
  **Disclosed residual: utils.py sits ONE line under the cap.** The next lane
  that touches it must reduce before it adds; there was no further honest
  simplification available inside this lane's scope, and buying the room by
  deleting contract-bearing docstrings or by exporting a fragment to a
  neighbour module would have been the forbidden kind of paydown.

## From the ui-smoke seed-stamp lane (base bef13f5e, 2026-09-01)

The CI `ui-smoke` job (`pytest tests/ -m ui_browser`, gated to tag and
workflow_dispatch, so it never ran on the campaign's commit tier) carried 7
campaign-born failures with ONE cause, proven by an HTTP probe against a live
seeded server plus a control run on this base: ABI-2 admission
(`ouroboros/task_result_schema.py::task_result_schema_refusal`) QUARANTINES a
durable task-result row that carries no `_schema_version`, reason
`unstamped_pre_7_0`. The browser lane's hand-written `task_results/<id>.json`
seeds are exactly such rows, so every seeded card was read as "no result": the
cards never reached finished state, and the assertions on finished/cancelled
state, cost, chronology and the review checkpoint all timed out waiting for a
card that could never close. Commit 9c4bf0b5 (the ABI-2 stamp — see "From the
f31c lane") had already stamped the hand-written fixtures of 9 test files, but
it did not reach the `ui_browser` lane, whose gate its commit tier does not
run.

Fix: the six hand-written seeds carry the same single additive key the writers
emit (`stamp_task_result_schema`). No production code, no assertion and no
test contract changed — the seeds now represent what a CURRENT-version writer
produces, which is what they always meant to represent.

| file | seeded `task_results` row | before | after |
|---|---|---|---|
| `tests/test_subagent_final_lineage_ui.py` :32 | `child-final-only.json` | no stamp -> quarantined | `"_schema_version": 1` |
| `tests/test_ui_smoke_liveness.py` :67 | `swarm-root.json` | no stamp -> quarantined | `"_schema_version": 1` |
| `tests/test_ui_smoke_playwright.py` :999 | `named-act.json` | no stamp -> quarantined | `"_schema_version": 1` |
| `tests/test_ui_smoke_playwright.py` :1460 | `chronology-progress-only.json` | no stamp -> quarantined | `"_schema_version": 1` |
| `tests/test_ui_smoke_playwright.py` :3662 | `gone-root.json` | no stamp -> quarantined | `"_schema_version": 1` |
| `tests/test_ui_smoke_review_checkpoint.py` :158 | `review-no-summary.json` | no stamp -> quarantined | `"_schema_version": 1` |

CLASS SWEEP, not instance. Every hand-written write into a `task_results` path
across the whole of `tests/` was enumerated by source scan (97 write sites),
then filtered to the lanes whose fixtures the classifier can bite
(`ui_browser`, `integration`, `serial`):

- `ui_browser` — 14 marked files, 58 collected tests: exactly the six rows
  above. The other twelve marked files seed no task-result row at all
  (`tests/ui_media_delivery_smoke.py` writes only media blobs under
  `task_results/artifacts/`, which no admission classifier reads).
- `serial` — no unstamped current-writer seed exists. Every serial file that
  stores a row goes through `write_task_result`, which stamps. The two
  hand-written exceptions are deliberately unstamped and must STAY that way:
  `test_e2e_cancellation_scenarios.py:508` seeds a pre-redesign
  `cancel_requested` latch whose whole point is the legacy shape, and
  `test_promote_event_transport.py:757/784/801` seed malformed/empty bytes for
  the fail-closed lookup pins.
- `integration` — `tests/test_provider_integration.py` touches no task result.
- Also left alone on purpose: the quarantine suite's own fixtures
  (`test_tasks_list_slice.py` `unstamped.json`/`future.json`,
  `test_task_result_schema_quarantine.py`). Those rows ARE the contract.

DISCLOSED, NOT FIXED HERE:
`tests/test_ui_browser_smoke.py::test_gateway_frontend_uses_api_client_boundary`
stays red in the same CI job. It is upstream-born, not campaign-born:
`web/modules/chat_media.js:185` calls raw `fetch(` outside the api_client
boundary. Upstream already fixed it in 619d6177, which arrives with sync #2;
editing the file here would collide with that sync for no gain.

GATES (host 0897-oma, base bef13f5e plus this commit; every pytest run with
all four `OUROBOROS_*` env vars on a fresh mktemp root, chromium+webkit from
`~/.cache/ms-playwright`; live `~/ouro/data` untouched —
`find ~/ouro/data -newermt <start>` empty):

- before (control on a `git archive` copy of bef13f5e, the three cheap files):
  `-m ui_browser test_subagent_final_lineage_ui.py test_ui_smoke_liveness.py
  test_ui_smoke_review_checkpoint.py` -> **3 failed, 1 passed in 96.51s**,
  each failure a Playwright timeout waiting for the card the quarantine
  prevented from ever finishing.
- after (the lane gate, all four seed-bearing files):
  `-m ui_browser test_subagent_final_lineage_ui.py test_ui_smoke_liveness.py
  test_ui_smoke_review_checkpoint.py test_ui_smoke_playwright.py -q` ->
  **35 passed, 1 deselected in 282.80s**, exit 0.
- rest of the lane (the ten remaining `ui_browser` files, so that the two runs
  together cover the CI job's full 58-test collection): **1 failed, 22 passed
  in 187.38s** — the single failure is the upstream-born boundary test above
  (`assert ['chat_media.js'] == []`). Whole job after this commit: 57 passed,
  1 failed, and that one arrives fixed with sync #2.
- `ruff check . --select F` -> All checks passed. `git diff --check` clean.
  `scripts/regenerate_size_ratchet.py --check` green (manifest byte-identical;
  `test_ui_smoke_playwright.py` is a GIANT_PATHS entry and grew by 3 lines,
  which the path-set manifest does not measure). `git rev-parse HEAD`
  unchanged (bef13f5e) after every pytest.

## Owner closures for the F6-sync #2 forks and A.24 (2026-09-01, batch 10)

- **Q-F6b-1 CLOSED (owner 1A)**: host notes stay TYPED flags in result_meta
  (note kinds, no text); the 256-byte host reserve is not widened — the note
  text remains visible to the model in the composed result.
- **Q-F6b-2 CLOSED (owner 2A)**: the owner-home read carve (В23=A) is
  re-affirmed as inherited from upstream — the credential-NAME gate applies to
  mutations only; reads/list/search rely on shape-based egress masking, with
  that residual disclosed.
- **Q-F6b-3 CLOSED (owner 3A)**: A5 literal-argv disclosure is ratified —
  shell operators, redirects and env references inside a direct argv array are
  DISCLOSED notes, not refusals (model-facing capability widening accepted).
- **A.24 RATIFIED (owner 4A)**: the composition-seam classification delta (a
  structured `{"ok": false}` behind an appended host note is a typed refusal,
  never a success) carries the owner-item id A.24.
- **Hub wave (owner 5A of batch 9) DELIVERED**: razzant/OuroborosHub PR #52
  merged by razzant (merge commit 67dd5ca7): all 22 extension manifests declare
  `plugin_api: "2.0"`, catalog.json regenerated by the hub's own
  `scripts/build_catalog.py`. Backward compatibility verified on the live
  6.113.5 code (f3fbfdbb): `parse_skill_manifest_text` parses the new
  manifests, the field is preserved as unknown (`plugin_api=None`), `validate()`
  is empty, and 6.113.5 has no PluginAPI negotiation/admission consumer — old
  installs are unaffected beyond the ordinary content-hash bump.

## From the Windows-lane green + daemon-audit #16 fixes (base 196438c9)

CI runs 33555971481 (9a28e58f) and 33563498919 (196438c9) were the first
3-OS runs on the campaign branch: ubuntu and macos full-test green, the
Windows full-test lane red on sixteen tests. Dispositions, one per class:

| class | tests | disposition | commit |
|---|---|---|---|
| chmod(0) unreadable probes (POSIX-only) | update-tx marker; rc_audit skills/task_results | skip on Windows; `os.geteuid` guarded | tests commit |
| open file cannot be unlinked (no FILE_SHARE_DELETE) | memory-journal replaced-source | skip on Windows | tests commit |
| `signal.alarm` POSIX-only | telegram chunker ×2 (upstream-identical file) | pytest-timeout is the Windows guard | tests commit |
| native separators | abi5 remnant scans ×1, golden credential listing | compare POSIX-relative paths | tests commit |
| shlex eats backslashes | glued `git -C<path>` predicate pin | POSIX spellings; residual is upstream-owned (`git_shell_policy`), disclosed in the test | tests commit |
| cp1252 decode/encode | message-bus chat.jsonl reads (upstream-identical file); `rc_audit --scope-only` (U+2261 in the scope text) | explicit utf-8 read; ASCII-escaped scope JSON | tests commit |
| simulated `O_BINARY` stripped the real bit | byte-exact atomic-write pin `[fsync=True]` | pin the real bit where it exists, simulate only on POSIX | tests commit |
| byte-exact writer vs `os.linesep` expectation | cybergym applied-settings verification (upstream-identical test; v7next writer contract) | expected bytes = the serialization | devtools commit |
| `signal.Signals` knows only host signals | process-signal observability ×3 (`SIG9` ≠ `SIGKILL`) | one `platform_layer.posix_signal_name` SSOT; the runner's duplicate table removed | platform commit |

Daemon audit #16 (sol, 22:11Z) — accepted findings landed here:
- finding 5: `owner_quiz._mutate_projection` lacked the ABI-2 write guard
  that `owner_hurry` carries → guard + stamp-on-write, red-first pin
  (`tests/test_quiz_answer.py`);
- finding 6 (partial): `supervisor.state.atomic_write_text` single
  `os.write` → delegates to `utils.write_bytes_atomic` (write loop, fsync,
  guard kept); the `view = view[os.write(fd, view):]` loops in
  `skill_review_history` / `project_dialogue` were REJECTED — POSIX
  `write()` returns 0 only for an empty buffer, the loop terminates.
- finding 4 (C6 after three rounds): owner checkpoint raised in the same
  batch; round 4 runs as a non-integrating lane until an independent PASS
  and the owner's choice.
- process findings 2/3/8/9 (uid-scoped pgrep, env on every python call,
  separate gate calls, lane pools ≤ healthy profiles, per-delta code check
  before merge) adopted as operator rules; full text in the coordinator
  disposition file.
## From the safety-port + preflight carve-out lane (owner 2B/11A, base bef13f5e)

Two owner decisions from batch №9, landed as two single-intent commits. Both
are gate-shaped: one moves a host fact into a PROTECTED file, the other
narrows an admission block that was structurally degrading a whole class of
commits to the audited bypass.

**2B — ADOPTION D05 ported into the protected `ouroboros/safety.py`.** This
lane edits a protected surface (AGENTS.md protected-paths list) under an
EXPLICIT owner sanction («2. B», 2026-09-01); the delta is exactly the two
facts named there and nothing beside them.

| fact | what the tip did | what it does now |
|---|---|---|
| observability root of a safety call | `chat_observed(drive_root=pathlib.Path(getattr(ctx, "drive_root", "../data")) if ctx is not None else pathlib.Path("../data"))` — a CWD-RELATIVE guess. It names whatever directory happens to sit beside the process's current working directory and only resolves to the real data root by coincidence of the dev layout; the `ctx=None` call shape (extension/MCP dispatch, and every direct `check_safety` call) took it unconditionally | `_safety_drive_root(ctx)`: the context when it has one, otherwise `config.DATA_DIR`, read LATE off the module so test isolation and runtime rebinding are honored — the same resolution order the review surfaces already use |
| who charges a safety call with no event queue | module-top-level `from supervisor.state import update_budget_from_usage` — an IMPORT-TIME edge from the module every worker imports, and which runs on every guarded tool call, into the supervisor package | `_record_safety_usage(ctx, payload)`: the context's own ledger writer when it injects one, else a CALL-TIME import of `supervisor.state` — the idiom the six sibling call sites of this writer in `ouroboros/` (reflection, post_task_synthesis ×3, post_task_evolution, improvement_backlog, semantic_dedup) already use. `ouroboros.safety` was the only module of that family importing it eagerly |

Not a byte copy of the frozen reference: the reference's `_safety_model_call`
does not exist there in this shape, so both facts were re-seated on tip bytes
at their tip call sites (`safety.py:942` and the no-queue branch of
`_emit_safety_usage`). The third fact of the D05 row, `schedule_followup =
POLICY_SKIP`, needed no port at all — the tip already carries it byte-identical
(`ouroboros/safety.py:111`); the ADOPTION row now says so instead of claiming
the tip "does NOT carry these facts".

Pins (`tests/test_safety_policy.py`, the D05 hook): `ctx=None` and a
context without `drive_root` both resolve to a REBOUND `config.DATA_DIR`
(late read, never `../data`); a context with `drive_root` wins; a CLEAN
interpreter importing `ouroboros.safety` leaves `supervisor.state` out of
`sys.modules`; the injected sink beats supervisor state and its absence falls
back to it.

**Two monkeypatch seams moved, and why that is not a weakened test.** Three
tests patched `ouroboros.safety.update_budget_from_usage` — a module global
that no longer exists once the import is call-time. They now patch
`supervisor.state.update_budget_from_usage`, which is the SAME function object
the code reaches, resolved at call time; every assertion they make about the
fallback is unchanged. Nothing was deleted to make a test pass.

**Residual, disclosed.** `_record_safety_usage`'s injected-sink branch is
ported with the function, but nothing on this tip provides it on the safety
path: `ToolContext` has no `update_budget_from_usage` field, and the object
that does carry that attribute is the supervisor's event context
(`server.py:781`, consumed at `supervisor/events_budget.py:122`), which never
reaches `check_safety`. So the branch's only current exerciser is its pin. It
is kept because it is half of the owner-adopted fact and because the attribute
name is an EXISTING convention in this codebase rather than an invented
protocol — not because a caller was found.

**11A — the doc-only carve in `release_metadata_preflight` (finding W3A-F1).**
Two gates classified the same diff two different ways. `ouroboros/tools/git.py`
exempts a doc-only diff from the compensating tests preflight
(`_doc_only = _diff_aware and _diff_is_doc_only(classification_paths)`), while
`ouroboros/commit_admission.py::release_metadata_preflight` returned
PREFLIGHT_BLOCKED for ANY changed set without `VERSION` in scope — the doc-only
diff included. The consequence was not a nuisance refusal: it was a structural
one. `preflight_review` is the only producer of a FRESH advisory verdict, so a
doc-only change on any install could not obtain one at all, and the standard
`preflight_review` → `commit_reviewed` flow degraded to the AUDITED BYPASS for
the whole class. It bites hardest on the two commit classes BIBLE P9 itself
exempts from the bump (`BIBLE.md:717-725`): a version-neutral external
contribution and a forensic recovery snapshot have no `VERSION` to name by
construction, so for them the bypass was the ONLY door.

The carve is three statements and reuses the commit gate's own detector,
imported from its owner module (`ouroboros.tools.git_review_cycle._diff_is_doc_only`,
which `git.py` re-exports at line 1635). SSOT on purpose: a second doc-only
predicate written here would be a detector the two gates could drift apart on,
and the whole finding is that they had already drifted. The import is
call-time and sits INSIDE the no-VERSION branch — the local idiom in this
module — so the ordinary VERSION-in-scope preflight pulls in no new module and
the admission layer keeps no import-time edge into `ouroboros.tools`.

| scope | before | now |
|---|---|---|
| `docs/NOTES.md` | PREFLIGHT_BLOCKED (never reaches a critic) | admitted — the advisory actually runs |
| `ouroboros/feature.py` | PREFLIGHT_BLOCKED | PREFLIGHT_BLOCKED (unchanged) |
| `docs/NOTES.md` + `ouroboros/feature.py` | PREFLIGHT_BLOCKED | PREFLIGHT_BLOCKED (mixed is not doc-only) |
| `tests/NOTES.md` | PREFLIGHT_BLOCKED | PREFLIGHT_BLOCKED (`_diff_is_doc_only` excludes `tests/`) |
| `VERSION` + a desynced carrier | PREFLIGHT_BLOCKED | PREFLIGHT_BLOCKED (the carve never runs; carrier coherence owns the answer) |

**Deliberately NARROWER than the two BIBLE classes it is motivated by.** A
version-neutral external contribution or a rescue snapshot that carries code
still blocks here, exactly as before. The carve keys on the diff's shape, not
on a claimed provenance, because provenance is caller-asserted and would be a
new trust surface on an admission gate; widening it is an owner decision this
lane did not take. Recorded as the residual of the finding rather than left
implicit.

Pins (`tests/test_advisory_preflight.py`, the class that already owned the
block): `test_doc_only_diff_without_version_reaches_the_critic` drives the real
`_handle_advisory_pre_review` and asserts the critic is DISPATCHED (not merely
"not blocked") for a doc-only scope with no `VERSION`;
`test_doc_only_carve_is_the_commit_gate_classifier` walks the table above and
asserts the classifier verdict beside every admission verdict, so a divergence
between the two gates fails as a classifier mismatch rather than as a mystery
block. The pre-existing
`test_changed_diff_without_version_blocks_before_sdk` is untouched — the
ordinary code diff still blocks, and it still proves the SDK is never reached.

**E2E S16 pinned the bypass-consequence, and is updated honestly.** In
`tests/system_e2e/test_system_scenarios_w3a.py`, S13B's `preflight_review` step
used to name the UNCHANGED `VERSION` alongside the doc — a workaround written
INTO the scenario precisely because the admission blocked the doc-only scope
(the finding says so in its own row). The step now names the doc-only scope
ALONE, which makes S16 a live end-to-end proof of the carve instead of a
scenario routing around it. Two consequences are disclosed rather than
silently absorbed:

- Naming `VERSION` put the run on the VERSION-in-scope branch, where
  `check_history_limit` runs. This checkout's README carries more patch rows
  than the P9 limit (finding W3A-F2, a PRE-EXISTING tree-state violation), so
  the scenario needed a `_trim_readme_history_to_p9_limit` fixture repair to
  reach a verdict at all. With no `VERSION` in scope that branch is never
  entered, so the fixture is deleted — it was compensation for the workaround,
  not for the scenario. W3A-F2 itself is NOT fixed by this lane and remains
  open against the tree.
- S16's contracts are unchanged: the fresh advisory, the post-verdict
  stale-from-edit refusal, and the revalidation refusal are all still asserted
  on the same steps. What changed is which door the first step walks through.

**GATE EVIDENCE: NOT PRODUCED IN THIS SESSION — disclosed, not implied.** The
session that wrote this entry had no permission to execute Python: every form
of `python3 -m pytest`, `python3 <script>`, `python3 -c`, and
`scripts/v7next_adoption.py` was refused by the host policy, and `ruff` is not
installed on it. So NONE of the owner-required gates ran here — not the
targeted suites, not the CI-shape non-serial battery, not `-m serial`, not
`-m size_ratchet`, not `ruff check . --select F`, not the adoption validator.
The change was verified by reading only: the classifier's contract and its
`tests/` exclusion, the carve's placement strictly inside the
`touched and not version_in_scope` branch (so no carrier-coherence check moves),
the call-time import (no new import-time edge into `ouroboros.tools`), the new
pins' fixtures against their proven siblings in the same file, and the size
lanes (`ouroboros/commit_admission.py` 246 lines, `tests/test_advisory_preflight.py`
709, `tests/system_e2e/test_system_scenarios_w3a.py` 791 — all under
TARGET_MODULE_LINES, so no `BAND_PATHS`/`GIANT_PATHS` membership changes and the
manifest stays exact). `git diff --check` is clean. The two commits of this lane
MUST NOT be treated as gate-proven; whoever integrates them owes the full
battery on an integrated tree.

Follow-up from CI run 33567328254 (9754cc95): Windows down to one red —
the credential-listing pin's own fold (JSON-escaped `\\` became `//`),
fixed; macOS red on two scheduler-sensitive pins (timed-out-tool lease drain,
SSE incremental follow at the 8 s stream deadline) — bounded waits, test-only;
neither touches the state-write delegation (in-memory queue / progress.jsonl).
## From the D04/D06 closure lane (owner 1B/3A, base bef13f5e)

### D04 — the flat wall-clock timeout pair, retired (owner 1B)

| Question | Answer |
| --- | --- |
| What the owner chose | 1B: DELETE both settings. Not "keep them one more minor as documented no-ops" |
| Idiom used | The existing one — membership in `RETIRED_SETTING_KEYS`, which `load_settings` strips off disk. No new migration machinery, no converter, no seed-into-a-successor-knob (there is no successor: the activity model already governs) |
| Why it was worth doing | The pair stopped terminating anything when the activity model (idle window + subtree liveness + absolute ceiling) replaced it. What survived was five surfaces discussing a value none of them obeyed: `SETTINGS_DEFAULTS` offered it, the Settings UI accepted a number, the save response apologised for it in a warning, `queue.init` compared the caller's value against the constant it then wrote anyway and logged a `deprecated_settings_ignored` row, and `/status` printed `legacy_timeouts_ignored: soft=600s, hard=1800s` on every request. A knob that is discussed everywhere and obeyed nowhere is worse than a deleted one — it reads as a live tunable to anyone grepping the setting name |

What went, by surface:

- `ouroboros/settings_defaults.py` — the two `SETTINGS_DEFAULTS` entries and
  the NOTE explaining that one of them no longer terminates tasks; both keys
  added to `RETIRED_SETTING_KEYS` with the rationale.
- `ouroboros/gateway/settings.py` — `_RETIRED_NO_EFFECT_KEYS` and the save
  warning built from it. A retired key cannot reach an effect bucket at all
  (the merge walks only `SETTINGS_DEFAULTS`; `load_settings` strips the key
  off disk), so `_effect_buckets` no longer needs the `warnings` parameter or
  the third exclusion, and both buckets are now truthful by construction
  rather than by subtraction.
- `supervisor/queue.py` — the two module globals, the `soft_timeout`/
  `hard_timeout` parameters of `init` (taken only to be compared and
  discarded), the legacy-key detection in `refresh_timeouts_from_settings`,
  and `_emit_timeout_deprecation_once` with its `_timeout_deprecation_emitted`
  latch. Keeping the emitter would have left a branch only a future
  non-default value could reach, and there can be no future value.
- `supervisor/workers.py` — the two globals and the two `init` parameters that
  existed only to forward them to `queue.init`.
- `supervisor/state.py` — the two `status_text` parameters and the
  `legacy_timeouts_ignored` line. The surviving line states the active model
  (`active_liveness: idle+deadline+absolute_ceiling+reaper`) without pretending
  two constants are configuration.
- `server.py` — the two `settings.get` reads, the `workers_init` arguments and
  the two supervisor-ctx fields (nothing read them).
- Bench carriers: the four SWE-Pro `e1v2` settings documents and the TB
  harbor agent's forwarded-env allowlist. A stripped key forwarded into a
  container is a value that silently does nothing on the far side too.
- `docs/ARCHITECTURE.md` — the two settings-table rows deleted; the settings-save
  paragraph now states the general rule (retired keys never reach a bucket; the
  RC auditor is the migration surface) instead of the special case.

Pins (`tests/test_legacy_timeout_retirement.py`, ten cases): keys in
`RETIRED_SETTING_KEYS` and absent from `SETTINGS_DEFAULTS`; a stored document
carrying both loads without them while the rest of the round trip survives, and
the env plane cannot smuggle them back; neither supervisor module still carries
a global; neither `init` still asks a caller for a value it discards; the
deprecation-event path is gone with its keys; `status_text` renders no legacy
line and grew no replacement knob; a grep-class sweep over every `.py/.js/.json/
.md/.html` in the tree with a named allowlist; and the RC auditor reports both
keys as `since: 7.0`, `behavior: stripped-on-load`.

Two existing tests asserted the OLD semantics and were reshaped rather than
deleted, because both still have a contract to state:

- `tests/test_settings_honesty.py` — was "a retired key is reported retired";
  now "a retired key is absorbed silently and claimed by no bucket", naming
  the RC auditor as where an upgrading install actually learns.
- `tests/test_heartbeat_presentation.py` — the two adjacent tests (one asserting
  the deprecation row fires on a non-default value, one asserting the retired
  planning-heartbeat key stays silent) collapse into one class-level pin: all
  three retired liveness knobs leave no deprecation chatter.

`tests/test_rc_audit_fixture_suite.py` gains the migration-visibility pin: the
frozen N−1 document (`settings_v6.113.4.json`) carries the pair at its DEFAULT
values, and a default-valued ghost is exactly the one nobody would think to look
for — so the retired-setting finding must appear for both keys on that fixture.

Cross-cutting notes:

- **`rc_audit.py`'s `since` field stopped being a one-key special case.** It read
  `"7.0" if key == "OUROBOROS_SCOPE_REVIEW_FLOOR" else "pre-7.0"`. Rather than
  grow an `if`-chain, the distinction it encodes is now named data —
  `RETIRED_IN_THIS_ABI` — so the auditor keeps telling an upgrading install the
  difference between "your stored value stopped working in THIS upgrade" and
  "it was already inert".
- **One deliberate small loss, disclosed.** Saving `OUROBOROS_SOFT_TIMEOUT_SEC`
  through `POST /api/settings` used to return an explicit "Retired setting(s)
  saved" warning; now the key is merged away silently, exactly like every other
  retired key. Restoring the warning would have meant reading the raw request
  body for keys the merge deliberately never looks at — new plumbing for a
  surface the RC auditor already owns. Consistency with the other seven retired
  keys won.
- **`queue.append_jsonl` kept its import.** With the deprecation emitter gone the
  name has no in-module caller, but `queue_snapshot.py` reads it through the
  `_queue()` module handle (four call sites) and three tests monkeypatch it. It
  now carries the same `noqa` marker as its two sibling facade names instead of
  looking like a leftover.

### D06 — the event taxonomy, four tiers with pairing enforced (owner 3A)

| Question | Answer |
| --- | --- |
| What the table had to be | DERIVED, not restated. The tree already holds the truth in two places — `EVENT_HANDLERS` (who answers) and the emitter call sites (who asks). A taxonomy that re-types either one is a third copy that drifts. So `supervisor/event_taxonomy.py` declares only what neither place records: the TIER, and the producing files |
| Home | `supervisor/event_taxonomy.py` — data plus one lookup. It imports `__future__`/`dataclasses`/`typing` and nothing from the runtime, and a test pins that import set: a table that can call nothing decides nothing, so it cannot quietly become a second dispatcher beside `EVENT_HANDLERS` |
| Reuse over a new framework | The producer→answer direction is ALREADY scanned by `tests/test_worker_event_registry.py`. The new suite imports its `_emitted_types`/`ALLOWLIST` rather than growing a second AST walker, so the two tables cannot describe different runtimes |

The four tiers, and why the split is real rather than decorative:

- **`worker_handler` (34)** — a dispatch entry whose handler takes a runtime
  action.
- **`telemetry_only` (7)** — also a dispatch entry, but the answer is the
  durable append in `telemetry_events.py` and nothing more. The tier is not a
  prose claim: the test asserts it equals `TELEMETRY_EVENT_HANDLERS` exactly, so
  a passthrough handler that grows an action has to change tier to stay green.
- **`server_intercept` (1)** — `restart_request`, consumed by `server.py`'s drain
  loop before dispatch because the supervisor thread cannot restart itself.
- **`nested_log_event` (3)** — `task_checkpoint`,
  `task_start_settings_reload_failed`, `review_reference` travel inside
  `log_event.data` and are answered by the nested branch, so they need no
  dispatch key. Without a row these read as producers nobody answers, which is
  precisely the shape the taxonomy exists to distinguish from a real hole.

`worker_handler ∪ telemetry_only == EVENT_HANDLERS` is asserted in BOTH
directions, and the two undispatched tiers are asserted ABSENT from it — so a
new key with no row, and a row claiming a dispatched tier with no key, both fail.

**The audit finding, retired rather than excused.** `events.py:272` advertised a
`schedule_task` dispatch key with no emitter anywhere in the tree — a capability
nothing could reach, indistinguishable by grep from a live one. The key is gone;
the FUNCTION keeps its name (`_handle_schedule_task`, whose family placement and
FUNCTION_DEBT key are pinned to it) and serves `schedule_subagent`, its only real
producer. Two tests followed the vocabulary rather than the behaviour and were
corrected with it: `test_events_extraction`'s owner assertion for the dead key is
dropped, and `test_nested_rights_depth` — which calls the handler DIRECTLY, so
the type string was incidental — now names the key that exists.

Cross-cutting notes:

- **No producer-less allowlist, deliberately.** The brief allowed known
  exceptions with a stated reason. After the retirement there were none, and
  adding an empty allowlist would have built the door through which the next
  dead key walks. A row with no producer is a hard failure; the remedy is to
  retire the key, as this lane did. The one exception that DOES exist —
  `test_worker_event_registry`'s `restart_request` — is not re-excused here:
  the taxonomy requires it to name the tier that answers it instead, so being
  "allowlisted" over there means "declared as `server_intercept`" over here.
- **The declared-producer check is the half no scan can do.** The scan only
  proves an emitted type is answered; nothing in it notices when the LAST
  emitter of an answered type disappears. Each row therefore names its producing
  files, and the test fails when a named file no longer contains the event
  string — a producer that moves or stops producing surfaces as a failure
  instead of a key that still reads as live.
- **One documented claim went stale and was fixed in the same commit.**
  `docs/ARCHITECTURE.md` said an `emit_log_event`-enveloped type was outside the
  scan's reach and that "for those the discipline is code review, not this
  test". That is no longer true — `review_reference` and the two nested
  siblings are exactly such types and now carry declared rows — so the sentence
  now points at the taxonomy.
## From the ADOPTION truth wave (base bef13f5e, 2026-09-01)

`ADOPTION_v7next.md` is the release inventory, and until this wave it was the
one campaign artifact nobody re-read against the tree. Its rows were written at
F0 and left alone: eleven deltas that LANDED months of lanes ago still read
`pending`, four rows still carried `pending-decision` after the owner had
answered them, and several hook cells named suites from the frozen reference
that do not exist here — a hook that cannot resolve proves nothing, and a
`pending` row whose work is done hides completed work exactly as badly as a
`done` row whose work is not.

This wave changes NO code. It is a bookkeeping pass in which every row was
re-read against the tree, every flip proven by running the suite named in its
new hook cell, and every remainder written down rather than rounded away. The
rule applied throughout: a row goes `done` only when a resolvable file hook
exists AND is green here; a row that owes work keeps its honest status and
describes the owed suite in `what` instead of spelling a non-existent path in
the hook cell.

### The flips, one row per line

| row | was | became | hook | evidence |
|---|---|---|---|---|
| D07 | `pending-decision` / `pending`, hook = the cancellation E2E suite (E1–E12) | `re-prove` / `done` | `tests/test_panic_stop_port_sweep.py` (both bound-port pins) + `tests/test_server_control_panic_daemon.py` + `tests/test_post_task_evolution.py::test_execute_panic_stop_wires_owner_stop` | Owner batch №5 5.5-5.8=A (2026-08-31) took the §5.4 three-column pass, so the disposition had no reason left to be `pending-decision`; the code landed at `88479fa7`. The F0 hook was REPLACED, not extended — the cancellation E2E suite pins nothing about the panic sweep, so it could never have proven this row |
| D08 | `pending-decision` / `pending`, hook = `tests/test_cancel_protocol_inventory_s6.py`, `what` = "same three-column pass as D07" | `re-prove` / `done` | + `tests/test_cancel_intent_corruption_s6.py` + `tests/test_subagent_worktree_registry_s6.py` | Re-proven from the owner batch verbatim rather than by cross-reference: 5.6=A (the four fail-open cancel-intent mutators re-derived on the upstream custody floor; `request_cancel`/`claim_intent` already strict = the one superseded sub-row) and 5.10=A (the `subagent_worktrees` strict registry applied byte-identical). Landed by `1b4a8da9` and `4fffefb1`. The borrowed-resolution cross-reference is dropped — both rows now carry their own |
| D09 | `re-prove` / `pending`, hook = `tests/test_llm_extraction.py` + "Ф4 D09-invariant scenario (plan §8)" | `re-prove` / `done` | `tests/test_context_overflow_hint.py::test_local_transport_makes_exactly_one_physical_attempt` + `tests/test_llm_typed_policy_refusal.py` + `tests/test_llm_provider_golden.py` + `tests/test_multiprovider_conformance.py::test_typed_policy_refusal_is_permanent_one_send_only` + `tests/test_llm_extraction.py` | Both halves landed (`86244943` deleted the `for attempt in range(3)` loop, `b94a6d1d` brought the typed-refusal subfamily with goldens 15→17). Two corrections travel with the flip. ADDRESS: the F0 row's `llm.py:2487` is the pre-split base address — the lane lives in `ouroboros/llm_local.py` here. PROMISE NOT KEPT, now disclosed instead of dropped: the Ф4 "D09-scenario" was never built. `tests/system_e2e/` runs S1–S23 and none of them is an LLM-routing row (checked by enumeration, not by memory); the invariant is carried by the unit, golden and CPL-6 conformance pins. Building an S24 belongs to the scenario lane |
| D11 | `retain` / `pending`, hook = "`scripts/v7_migration.py`-style ledger checker (reference: v7_wip)" | `retain` / `done` | `tests/test_repo_health_smoke.py::test_transition_allows_a_same_qualname_relocation_but_not_a_swap` + `tests/test_smoke.py::test_size_ratchet_transition_against_explicit_base` | Landed by `d1c8fca4` and exercised on real history by `14567df5`. HOOK REPLACED: the F0 cell named a checker from the frozen reference that does not exist in this tree — an unresolvable hook on a row whose work was finished |
| D31 | `re-prove` / `pending` | `re-prove` / `done` | `tests/test_external_review_script.py` (D31 pins named individually) | Landed by `3b62c1d6` with the marker teeth of `db944347` and preserved through the F6 sync; upstream still carries the classifier gate instead, so the delta is live, not superseded |
| D33 | `re-prove` / `pending` | `re-prove` / `done` | `tests/test_module_handle_extraction.py` (the nine `_loop` LEAVES rows) + `tests/test_loop_owner_facades.py` | Landed by `f0d8b147`, declared sets re-derived after the F6 sync |
| D34 | `re-prove` / `pending` | `re-prove` / `done` | + `tests/test_update_merge_owner_facade.py` + `tests/test_update_merge_assisted.py::test_materialize_projects_version_to_target_and_pins_m0` | Landed by `7f0a1124` under owner 5.12-5.14=A. Two residuals disclosed in the row rather than left implicit by `done`: the engine governs steady state only (the first pre-v7 upgrade still runs the OLD updater) and the boot-recovery M0 backfill degrades to assisted |
| D36 | `re-prove` / `pending` | `re-prove` / `done` | `tests/test_module_handle_extraction.py` (the four delegate-family rows) + `tests/test_delegate_owner_facades.py` | Landed by `782b5fe3` and `1b4a8da9`; the leaf rename `delegate_terminal` → `delegate_terminal_evidence` is owner 5.9=A |
| D37 | `re-prove` / `pending` | `re-prove` / `done` | `tests/test_module_handle_extraction.py` (three review-stack rows) + `tests/test_review_owner_facades.py` | Landed by `04b1de9c` and `3b62c1d6`. The reference leaf names `review_advisory_prompt`/`run` were byte-falsified by the drift probe (the SDK transport had been retired) and were re-minted under the organ's public rename |
| D38 | `re-prove` / `pending` | `re-prove` / `done` | + `tests/test_lc2_owner_facades.py` + `tests/test_generated_inventories.py::test_facade_inventory_is_byte_identical` | Landed by `1a17218d` and `f0d8b147`. Disclosed: the usage declared set and the `post_task_synthesis` handle diverge from the frozen reference table — tip truth |
| ABI-6 | `retain` / `in-progress`, hook = "ruff F + targeted per-item suites + grep-level absence checks" | `retain` / `done` | `tests/test_contracts.py::test_api_v1_shim_removed_and_gateway_declares_core_ws_message_types` + `tests/test_gateway_abi3_removals.py` (`TestApiV1ShimRemoval`) + `tests/test_tool_classification_differential.py` + the per-item consumer suites | Both ROUTED items closed since the F0 text was written: the `_typed_or_adapted` branch was not reproduced by the F3.1 lane A re-derivation (`ccbb933a`, zero tip hits) and the `contracts/api_v1` shim was removed with negative pins by lane D3 (`33ba6e83`). HOOK REPLACED: the F0 cell was prose plus grep verbs and could never resolve. RESIDUAL kept in the row: the three F3.0 removals are proven by surviving positive suites plus grep-level absence, not by dedicated negative pins |

### The rows that did NOT flip, and why

- **D03 — partial, and the hook the F0 row promised does not exist.**
  `tests/test_settings_read_seam.py` is a reference name with no file behind
  it. The vocabulary split (oracle rows 840-912) and the launcher half (rows
  918-920) landed at `a4481521`; the semantic delta proper — rows 913-917 and
  1080-1081 — is hot-deferred and no later train re-derived it. The hook cell
  now names the two landed halves (`tests/test_config_extraction.py`,
  `tests/test_onboarding_host.py`), both green here, and the owed pin is
  described in `what` instead of being spelled as a path. The phase moved
  F1→F6 because F1 closed without the remainder, so the cell named a dead
  phase; `scripts/v7next_adoption.py`'s `REQUIRED_PHASE` moved in the SAME
  commit, which is the point of that pin. Recorded plainly: this is an
  OPERATOR scheduling correction, not an owner decision about where the
  remainder lands, and the owner may overturn it.
- **D18 — the four F2.2 leaves were proof-green and never pinned.** The f22
  ledger entry claimed `queue_snapshot`, `queue_timeouts`, `worker_health` and
  `worker_assignment` had their declared sets in `LEAVES`; the tree did not
  bear it out, so none of the three parametrized invariants was running on
  them. `fc1528d5` adds the four rows with tool-derived exact read sets — 147
  passed, up from 135. DISPOSITION OF MIGRATION ROW 1030, which is why the row
  stays `pending`: the `QUEUE_SNAPSHOT_PATH` single-authority collapse is
  neither re-applied nor dispositioned. Verified on the tree, not assumed —
  `supervisor/queue.py:17` imports the name from `supervisor/state.py` and
  `supervisor/queue.py:71-73` rebinds its own copy in `init()`, beside
  `supervisor/state.py:31,41-46` doing the same. Two module globals answer one
  question and agree only because both inits are handed the same drive root.
  That is a live single-authority defect, so the row cannot go `done`, and
  choosing which global wins is a change to durable-state addressing — a lane,
  not a bookkeeping edit.
- **R-WINWAVE — the registry now exists; the matrix that would prove it does
  not.** `docs/v7next/WINWAVE_CLASS_REGISTRY.md` records one decision per
  class: 7 re-applied in reference form, 3 superseded-by-upstream, 6
  not-applicable because the carriers the reference patched are absent here.
  The count is SIXTEEN, not the fifteen the campaign audit's summary line
  said: the recount separates the `PermissionError`-beside-`IsADirectoryError`
  clause from the `tools.jsonl` utf-8 read, which that line had merged. A
  seventeenth class was found by the matrix itself and is recorded the same
  way — source-text regex pins in the JS suite cannot match a CRLF checkout —
  decided narrowly (normalize at the two `readFileSync` reads) and fixed by
  `a0b35fcd`, which is NOT an ancestor of this worktree. Run log: run
  33555971481 on `9a28e58f` is green on ubuntu and macos and RED on windows on
  exactly that class; run 33563498919 on `196438c9` carries the fix
  (`a0b35fcd` verified as an ancestor of `196438c9`) and is recorded as
  **pending** because nobody here read its result. The hook cell now points at
  the registry plus the per-class Linux pins — a real file that a reader can
  open — instead of the F0 cell's prose `gh workflow run` invocation.
- **D04, D05, D06, D35 — owner-decided, lanes owed.** These are not operator
  judgements; owner batch №9 (2026-09-01) answered all four, and three of the
  four answers went AGAINST the operator's recommendation. №1=B: retire
  `OUROBOROS_SOFT/HARD_TIMEOUT_SEC` in 7.0 via `RETIRED_SETTING_KEYS` rather
  than accept upstream's keep-and-warn form — lane d04d06. №2=B: port
  `_safety_drive_root` and the function-scope budget import INTO the protected
  `ouroboros/safety.py` rather than keep the upstream form and disclose the
  residual — lane gates; the B answer is itself the explicit owner sanction
  the protected-file rule requires. №3=A: build the four-tier event taxonomy
  in 7.0, absorbing (not forking) the upstream `telemetry_events.py` registry
  that arrived with the F6 sync — lane d04d06. №13=B: do the transport work
  (f-string support in the transplant tool) and move `prepare_managed_update`
  and `safe_restart` rather than ratify their current facade shape — lane d35.
  All four keep `status = pending` with those texts in `what`; their
  dispositions leave `pending-decision` for `re-prove`, because a decision the
  owner has taken is no longer a decision that is owed. Their phase cells
  still read F1 although F1 is closed: re-phasing a required row is an owner
  decision and this wave does not invent one.

### Prose corrections in the same manifest

1. **"17 delta families" → 18.** The header sentence had read 17 since the F0
   skeleton. The list it introduces (`D02–D09, D11, D13, D18, D31, D33–D38`)
   and the validator's `REQUIRED_DELTAS` tuple both hold 18 and always did.
   The word was wrong, not the inventory — a documentation error, not a
   missing row, and it is stated that way so no one goes looking for a
   nineteenth family.
2. **CPL-1 "488 tracked runtime modules" → 504.** Re-counted from the manifest
   rather than carried forward: `docs/DOMAIN_MAP.md` totals 504 across the
   twenty domains, and `ouroboros/domains.toml` has 504 module entries. The
   488 was true when the row was written and stopped being true when the F6
   sync and the accepted proposed placements grew the population. The same row
   loses its stale open residual: the 80 `[classification].proposed`
   placements were accepted as the 7.0 base by owner batch №9 №10=A and the
   map regenerated with zero starred rows (`8e885412`).
3. **TRAIN-F6-8d13373b "(owner signal, 2026-09-01)" → the real provenance.**
   The parenthetical claimed the «иди забирай» signal that the Q1 adoption
   contract names. That signal was not given for this train. The train was
   adopted on the OPERATOR's inference from the owner's «И надо как-то
   ускоряться … а то уроборос двигается вперёд быстрее чем ты работаешь»
   (10:10Z), and the owner sanctioned taking upstream POST HOC — after the
   merge — with «уроборос далеко уехал в ветке ouroboros … можешь
   ребейзнуться, забрать оттуда вещи новые» (19:12Z). The sanction is real and
   the train stands; the ORDER is what was misstated, and a row that reads
   "owner signal" where the truth is "operator inference, sanctioned
   afterwards" is precisely the kind of drift this ledger exists to catch.
   CPL-4 and CPL-5 were left untouched by this wave.

### What `--release` still refuses, verbatim

The validator is green in normal mode and red at the release bar. The eight
refusals are the honest remainder of the campaign, and they are listed here so
that nobody has to run the tool to learn the size of what is left:

```
release: D03 status 'pending' != done
release: D04 status 'pending' != done
release: D05 status 'pending' != done
release: D06 status 'pending' != done
release: D18 status 'pending' != done
release: D35 status 'pending' != done
release: R-WINWAVE status 'pending' != done
release: CPL-4 status 'in-progress' != done
```

Every one is a status refusal. NOT ONE is a hook-resolution refusal, and that
is the load-bearing result of this wave: `--release` resolves every `tests/`,
`scripts/` and `docs/` token in the hook cell of every `done` row and would
name any that pointed at a missing file. Before the wave, four rows (D07, D09,
D11, ABI-6) carried hooks that could not resolve at all; after it, the whole
inventory's hook surface is real files. The suites behind them were RUN, not
merely resolved — 74 hook files, green.

### Notes rewritten under the rows

- The F0 note "the `pending-decision` rows (D04, D05, D07, D08) are exactly the
  deltas that need the plan §5.4 three-column matrix" is now false and was
  replaced rather than left to age: all four have their resolutions (D07/D08
  from batch №5, D04/D05 from batch №9). The enum value stays in the schema
  for the next genuine fork.
- The F0 note "hook suites named from the frozen reference do not exist on this
  tree yet; the validator checks the manifest contract, not hook existence" was
  written before the release-bar hook contract existed. The replacement states
  the contract that is actually enforced, and states the corollary that governs
  the non-`done` rows: an owed suite is described in `what`, never spelled as a
  path in the hook cell, because a path nobody wrote reads as a hook and proves
  nothing.
## From the W3B-F1 lane (owner 8A, base bef13f5e)

**STATUS: LANDED AND GATED** by the continuation session (base bef13f5e,
unchanged). The first attempt at this lane ran under a harness policy that
permitted only a read-only command allowlist — `python`, `python -m pytest`,
`mktemp`, `ruff`, `bash -c`, `git status` and `git diff --check` were all
refused — so it could run NOT ONE gate and committed nothing; its findings are
kept below verbatim because the first row invalidates part of the lane's own
written instruction. The continuation session had full command access, ran
every gate for real, and observed the pins red before they were green. The
"work this lane did NOT complete" list at the end of the section is answered
row by row in the closing table.

| finding | disposition | mechanism |
|---|---|---|
| The lane instruction's prescribed digest CANNOT carry this fix | CORRECTION to the instruction, disclosed | W3B-F1 was specified as "compare the ABI-9 per-publication digest (`extension_registry_state.extension_generation_digest`)" between worker and server. That digest is `uuid.uuid4().hex`, minted fresh at every publication (`extension_plugin_api.py:801`) and re-stamped on every already-published descriptor in the same lock hold. Two processes importing the BYTE-IDENTICAL payload therefore mint different digests, and a worker comparing its own against the server's would see permanent divergence — a reload before every task, which is exactly the unbounded behaviour the lane forbids. The digest is a within-process dispatch-provenance fact and is correct at that job; it is not a cross-process identity. The inherited WIP had already reached this independently and substituted `live_extension_fingerprint()` — a sha256 over the sorted `(skill name, content hash, skill dir)` triple, i.e. the SAME identity `live_loaded` already compares — which is genuinely equal across processes that loaded the same payload. That substitution is endorsed; the instruction's pointer is wrong |
| Durable carrier: none existed, one was added | Inherited WIP, endorsed by inspection | The lane said "find the durable carrier". There is none: `state/skills/<name>/enabled.json` and `.../health.json` are per-skill owner/health state, and `state/extension_companions.json` is a companion snapshot — none publishes the server's live extension SET. The WIP adds `state/extension_generation.json` (schema 1) written by the server and read by workers. Deriving the desired set from durable sources instead would have meant a `discover_skills` sweep per task, defeating the O(stat/read of a small file) budget the lane fixes as a requirement |
| No publish/adopt feedback loop — and the guard is load-bearing | Verified by inspection, NOT by test | A worker that adopts runs `reload_all`, whose per-skill `reconcile_extension` calls announce worker→server reconcile requests; the server then reconciles and re-announces. The cycle terminates only because `publish_extension_generation` is write-if-changed: the server's live fingerprint is unchanged by reconciling an already-live extension, so no new generation is published and the worker's next probe reads in-sync. This is the property most worth a red-first pin and it does not have one |
| A worker-side adopt cannot globally disable a skill | Verified by inspection, NOT by test | The risk is real in shape: adopting calls `reload_all`, and a skill that loads in the server but not in a worker takes the load-error branch, which can call `_revert_enabled_after_load_error` → `save_enabled(..., False)` — a worker's local failure demoting a globally healthy skill. It does not fire: `revert_enabled_on_error` defaults `False` (`extension_loader.py:273`) and `reload_all` never passes it, so the revert is reachable only from the enable paths that own it. Structural divergence instead degrades to the pre-fix behaviour, bounded by the one-reload-per-distinct-generation guard |
| Spawned workers correctly report non-server | Verified by inspection, NOT by test | The whole direction switch is `is_server_process()`. Had it been `os.getpid()` captured at import, a SPAWNED worker (macOS/Windows) would re-import the module, adopt its own pid as the server pid, and publish its own generation over the server's — inverting the fix. It reads `OUROBOROS_SERVER_PROCESS_PID` from the inherited environment (`extension_companion.py:23`), which the spawned child inherits, so the worker resolves to non-server on both start methods |
| `extension_loader.py` line pin | Inherited WIP, respected | The lane pinned the file at 1000/1000 with nothing to be added inside. The WIP took it to 992 by REMOVING `_request_server_reconcile_if_worker` and rerouting its call sites to `announce_extension_state_change` in the module that owns the durable carriers — a relocation to the owning leaf, not a helper split to buy room. `TARGET_MODULE_LINES` is 1000 and the ratchet transition is shrink-only, so the direction is safe; unverified, because `-m size_ratchet` could not be run |

Work the FIRST session did not complete (every item closed by the
continuation; the closing table below records how):

- **The third natural point is missing.** The WIP adopts only at task start
  (`worker_process.worker_main`, before `handle_task`), which does cover the
  tool-catalog build because the catalog materializes inside `handle_task`.
  The lane also required the first `Unknown tool` for a name in the durable
  registry. The clean seam is `_extension_dispatch_candidate`
  (`ouroboros/tools/extension_dispatch.py:20`), where
  `parse_extension_surface_name(name)` succeeds — the name IS structurally an
  extension surface — but `get_tool(name)` returns `None`, meaning this
  process has no such surface. A probe there is already bounded by the
  one-reload-per-distinct-generation guard, so a bogus `ext_`-prefixed name
  cannot buy repeated reloads.
- **Every pin is missing.** No unit pin (a worker with a stale digest loading
  a skill enabled after its spawn, before dispatch); no disable/uninstall
  symmetry pin; no pin on the write-if-changed loop guard above. Nothing was
  ever observed red, which is the point of red-first.
- **The S13 E2E variant is missing.** `tests/system_e2e/test_system_scenarios_w3b.py`
  currently encodes the DEFECT as intended behaviour: its phase 2 restarts the
  server and comments "the product's worker pickup point — task workers load
  extensions only at worker spawn". The variant must enable after boot and
  dispatch into an EXISTING worker with no restart, and the scenario manifest
  (`tests/system_e2e/harness.py`, `SCENARIOS`) needs the matching row — the
  harness pins that a new `test_s<N>_*` without a manifest row is red.
- **The overhead was never measured**, as the lane required.
- **ARCHITECTURE same-commit** rows for the new durable carrier were not written.

### Continuation session — dispositions

| item | disposition | mechanism |
|---|---|---|
| The inherited WIP adopted against the WRONG root | DEFECT IN THE WIP, fixed | `worker_main` called `_adopt_published_extensions(task_drive_root)`, i.e. `task["drive_root"] or drive_root`. A subagent or headless task carries its OWN forked root, which has no extension registry and no published marker, so the probe read "nothing published" and returned without doing anything. This was not a theory: the S13 hot-adoption E2E written for this lane went RED against the WIP with `"result_preview": "⚠️ Unknown tool: ext_15_r_e2e_hot_probe_echo"`, `"status": "unknown_tool"`, and ZERO `extension_generation_adopted` rows in the events log — the original defect, reproduced end to end through the WIP's own mechanism. The worker now adopts against the POOL root it loaded from at spawn, which is the only root the two generations are comparable in, and the same E2E is green |
| The second natural point (the dispatch miss) | ADDED | `_extension_dispatch_candidate` (`ouroboros/tools/extension_dispatch.py`) now runs the same bounded adopt when `parse_extension_surface_name` succeeds and `get_tool` returns `None`, then re-reads. That is the enable landing MID-task, which the task-start point structurally cannot see. No new predicate was needed to keep a hallucinated `ext_`-prefixed name from buying reloads: the one-reload-per-distinct-generation guard already bounds it, so such a name costs one 131-byte read. (The lane's third listed point, the tool-catalog build, is the same event as task start — the catalog materializes inside `handle_task`, after the probe.) The pin is behavioural and was observed red with the carrier PRESENT and only the two seams at base: `AssertionError: the dispatch miss answered Unknown tool without probing` |
| The pins | WRITTEN, observed red first | `tests/test_extension_generation_adoption.py` (10 tests): the defect at both natural points, the disable direction, write-if-changed publication, one-reload-per-generation with `load_enabled` still True after a worker-local load failure, fail-closed on absent/unparseable/empty markers, the steady-state probe reloading nothing, and the server never adopting its own publication. Red was observed twice: once with the whole carrier reverted to base (import error — honest for a new mechanism, but weak) and once with the carrier present and only the consumption seams at base, which is the behavioural red quoted above. Both generation values the disable pin replays are genuine `publish_extension_generation` outputs, not literals |
| The S13 E2E variant | WRITTEN | `test_s13_hot_enable_reaches_an_already_spawned_worker_without_a_restart`. It keeps the scenario id (so the lane's own target command, `-k s13`, selects both tests) and the manifest row was rewritten to declare both. "Already spawned" is READ, not assumed: the premise waits for the supervisor's durable roster (`state/worker_pids.json`) and for every pid on it to have announced `worker_ready`, all before the payload exists; the no-restart claim is that same roster unchanged across the enable and the dispatch, because a respawned pool would satisfy the dispatch assertion while proving nothing. The old phase-2 comment, which recorded the DEFECT as the product's contract ("task workers load extensions only at worker spawn"), was rewritten to say what that restart actually pins — that the enabled state survives a reboot |
| Overhead | MEASURED | Steady state (published == local, the case every task pays): **27.6 µs at 1 live extension, 51.7 µs at 5, 62.9 µs at 20**, against a `reload_all` of 5.9 ms / 17.2 ms / 42.5 ms — **212× to 676× cheaper**, and the marker is a fixed **131 bytes** at every registry size. Method: 2000 in-sync probes per point, one temp data root per point, `n=20` for the reload baseline. The growth with registry size is the in-memory fingerprint, not I/O; the file read is constant |
| ARCHITECTURE | WRITTEN same-commit | The `extension_reconcile_queue.py` module row now carries both directions and all three bounding properties, `extension_registry_state.py` gains `live_extension_fingerprint` beside the digest it is deliberately not, `state/extension_generation.json` is in the data-layout inventory, and the companion-catalog prose names the reverse direction |
| Disable/uninstall symmetry | PINNED | Unit: `test_a_worker_adopts_the_disable_direction_too` — the surface LEAVES a worker that already had it live, converging on the real empty-set generation. E2E: the hot variant asserts the owner's disable republishes a DIFFERENT generation at once (the carrier evidence that running workers will retract), and the pre-existing S13 keeps the uninstall contract (payload + state dir removed, skill delisted) |
| `extension_loader.py` line pin | HELD at 992/1000 | Unchanged from the WIP's relocation; `-m size_ratchet` was run for real this time and is green, so the shrink-only transition is verified rather than assumed. Nothing was added inside the file: the dispatch-miss probe went to `ouroboros/tools/extension_dispatch.py`, which owns dispatch, and no helper was created to buy room anywhere |
## From the D35 f-string transplant lane (owner 13B, base bef13f5e)

1. **F-string reads are ordinary declared reads.** The transplant walk now
   records `ast.Name` loads under `ast.JoinedStr` / `ast.FormattedValue` and
   rewrites them to the same call-time `HANDLE().NAME` form as every other
   function-body load. The token proof accounts for pre-3.12's single STRING
   token without weakening literal-byte comparison. A fourth, parse-tree
   inverse proof catches the one semantic exception: debug expressions such
   as `f"{X=}"`, where CPython derives a displayed Constant from the expression
   spelling, fail closed after a handle rewrite.
2. **The last G1 spans reached their semantic owners.** The stock tool
   re-derived and proved `safe_restart` into `supervisor/git_ops_reset.py`
   (8 declared parent names, 24 rewrite sites) and
   `prepare_managed_update` into `supervisor/git_ops_updates.py` (14 names,
   21 sites). `supervisor/git_ops.py` re-exports both objects. Every mutable
   branch/helper/root read remains behind `_go()` at call time; the pre-init
   data root continues through `current_drive_root()` and the parent's
   per-call config resolution, so importing a leaf cannot capture a live root.
3. **Inventories and disposition.** The module-handle exact-read sets and G1
   facade-owner map include the two spans. No path was added, so the derived
   `GIT_OPS_FAMILY_PATHS` set is unchanged and remains wholly absorbed by
   `RELEASE_INVARIANT_PATHS` and the contributor release inventory. D35 is
   `done`; its adoption hooks are the module-handle and owner-facade suites.
## From the E9 carve-out + scheduled E2E CI lane (owner 4A/9A, base bef13f5e)

1. ABI-2 CARVE-OUT (owner 4A, one commit). Q8=B quarantines every unstamped
   `task_results` row on the first ordinary read — including the rows the
   cancellation redesign exists to rescue. A pre-redesign task wedged in the
   `cancel_requested` latch is unstamped BY DEFINITION, so the first reader
   moved it into `task_results/quarantine/`; `migrate_legacy_cancel_latches`
   then found nothing to migrate, and custody's own fail-soft read (which also
   quarantines) saw no durable result and settled `not_found`. The wedged task
   disappeared without ever reaching a terminal — E9 red. The migration now
   OPENS with a pre-pass that re-writes exactly those rows through the ordinary
   writer: same status, same fields, no conversion. That is the identical
   stamp-on-write `require_writable_task_result_schema` already admits as
   lawful for a live pre-upgrade task on its next lifecycle write; the only
   thing special about a wedged task is that it has no worker left to perform
   it. The row is then an ordinary latch, the existing scan adopts it, and the
   existing intent -> custody path drives it to the `cancelled` terminal — the
   carve-out buys admission, not an outcome, and no terminal-writing code was
   added. Scope is deliberately narrow on four axes at once: refusal reason
   must be `unstamped_pre_7_0` (future/malformed still quarantine), status must
   be the latch, `task_id` must equal the filename stem (a mismatch would make
   this write a DIFFERENT file), and everything else quarantines on the very
   next read — including the migration's own scan one line below.
2. THE CARVE-OUT IS AN ORDER, NOT ONLY A WRITE — and this is what kept E9 red
   after the write was correct and its unit pins were green. Whichever durable
   read reaches the latch first quarantines it, so the exemption is worthless
   unless it runs before all of them. The migration lived in
   `_startup_custody_sweep`; the orphan reconcile in
   `_run_startup_task_recovery` is an earlier read and won the race every boot.
   Moved to the head of `_run_startup_task_recovery`, ahead of the reconcile,
   and pinned there by source order (`test_boot_migrates_the_latch_before_any_
   quarantining_read`) — including the negative half, that it must not ALSO
   still run from the later sweep. Worth recording how the race was found: the
   unit pins call the function directly, so they were green while E9 was red,
   and E9 runs against a `git clone` of the tree, so uncommitted work is
   invisible to it. Both facts hide an ordering bug; instrumenting
   `quarantine_task_result` with a stack dump was what actually located it.
3. VISIBILITY: applying the carve-out is ONE typed durable
   `task_result_cancel_latch_admitted` event per boot (count + task_ids +
   reason), never one per file — the same log-only shape owner decision 6.3=B
   gives the quarantine itself. A second boot admits nothing and writes no
   second row, which is pinned.
4. SCHEDULED SYSTEM-E2E LANE (owner 9A, one commit, protected `ci.yml`).
   `tests/system_e2e/` is gated three ways — `integration` + `serial` markers
   plus `OUROBOROS_E2E_DEEP` — precisely so no existing pass can trigger it,
   which left a suite nobody executes. The plan's §8 pull-request lane is
   REPLACED by a daily `system-e2e-mock` job (cron `37 4 * * *`, off the hour
   because GitHub's cron queue is deepest at :00, plus `workflow_dispatch`),
   ubuntu-latest, `timeout-minutes: 40`, the standard `setup-python-env`, and
   `python -m pytest tests/system_e2e/ -o addopts="" -q` with the four
   `OUROBOROS_*` roots under `runner.temp`. Owner rationale for schedule over
   PR: each scenario spawns a real isolated server, so the cost belongs on a
   nightly rather than on every review. Keyless by construction — the job
   names no secret, which is the only way a job gets one.
5. THE COSTLY PART OF 9A WAS THE TRIGGER, NOT THE JOB. Three of
   `integration-test`'s conditions match a branch ref, and a scheduled run
   carries the DEFAULT BRANCH in `github.ref` — so merely adding `schedule:`
   to this workflow would have woken the PAID provider lane every night. It
   now leads with `github.event_name != 'schedule' && (...)`, and the added
   parenthesis matters: without it the guard would only have bound the first
   alternative. Pinned in `tests/test_system_e2e_ci_lane.py` alongside the
   job's own shape (schedule present and off-peak, `if` admits only schedule
   and dispatch, the four temp roots, no `secrets.` anywhere in the block).
6. AN EXISTING PIN HAD TO CHANGE, AND ONLY ONE.
   `test_pull_request_ci_is_fork_safe_and_does_not_enable_provider_jobs`
   asserted `"\n  schedule:" not in workflow` — a blanket ban standing in for
   the real invariant, that no unattended trigger reaches a paid job. With the
   owner sanctioning a schedule, the ban was replaced by that invariant stated
   directly (the `!= 'schedule'` guard is present in the `integration-test`
   block), which is strictly stronger than what the ban bought: the ban would
   have passed a workflow whose schedule existed but whose guard did not.

## Integration evidence for the gates lane (2B/11A) and the finale bookkeeping (base d21806d8)

The 2B safety port (0bf723cc) and the 11A doc-only carve (0ebbfaeb) landed
without their own gate run (the lane's Bash was read-only); the debt the lane
disclosed is discharged on the integrated tree: CI-shape non-serial and
serial batteries green on 0ebbfaeb, E2E W3A 12/12 (S16 re-run on the
committed carve), and the full 3-OS matrix green on 8b27b507 (run
33569841899) and on every later tip. Scope-review lane 1 (claude opus-5,
run-852bb8facb34) also surfaced: the RC auditor's attestation now names the
removed ``ouroboros.contracts.api_v1`` module; ``supervisor.state`` no longer
carries a second ``QUEUE_SNAPSHOT_PATH`` (MIGRATION row 1030, D18 -> done);
the W3A scenario prose is renumbered to the manifest (S14-S17); the WINWAVE
registry run table carries every matrix run through 33574822693 and R-WINWAVE
-> done; the ADOPTION notes name the rows that still owe work (D03, CPL-4).
Left for the owner: ``docs/CHECKLISTS.md`` still cites the removed
``contracts/api_v1.py`` (protected file) and ``review_model_routes``' typed
views without production consumers (ABI-4 status).

## Superseding note on «What `--release` still refuses» (as of c7992acf)

The list above was written on the adoption lane's worktree and names eight
rows; on the integrated branch the validator refuses exactly two: D03
(`pending` — the settings read seam pin is still owed, owner fork open) and
CPL-4 (`in-progress` — C6 waits on the owner checkpoint after review round
4). D18 closed with 091ee3b3 (one `QUEUE_SNAPSHOT_PATH` authority) and
R-WINWAVE with 9509d493 (green matrices 33569841899..33572515529). Scope
review №10 (three read-only lanes on d21806d8: claude opus-5
run-852bb8facb34, codex gpt-5.6-sol run-97075f418c64, cursor grok-4.6
run-6e1963c63371) returned NOT READY on this release bar and on the same
bookkeeping; every non-owner item is closed in 091ee3b3..c7992acf, the owner
items (CHECKLISTS.md api_v1 clause, ABI-4 typed views, D03, D09/S24, web
vision evidence) are in the owner batch. `git diff --check b9f7597f..HEAD`
stays red on the vendored `web/mermaid.min.js` (upstream minified asset) and
the generator-owned `DOMAIN_QUOTIENT_REPORT.md` EOF — disclosed, not edited.

## Integration corrections — the SHA a lane reported vs the SHA that landed (RES-27)

Every section above records the SHA its lane produced. Integration rebases, so
several of those SHAs name commits that are not on this branch: the work landed
under a different id. Reading a lane section literally therefore sends you to a
commit `git show` cannot resolve on the integration line, and in one case the
lane's result was briefly LOST and had to land twice.

This section is the reconciliation. It is derived, not remembered: every row
below was produced by taking each 8-hex token in this file, keeping the ones
that are commits, dropping the ones reachable from the integration HEAD, and
matching the remainder to the integration commit with the identical subject.

### Lane SHA -> landed SHA

| lane section | lane SHA | landed as | subject |
|---|---|---|---|
| F4 wave 4 (W2-F2) | `68b19a61` | `2e93906f` | update: honor the configured managed update source on every fetch (W2-F2) |
| F4 wave 4 (scenarios) | `4e68526c` | `592434d0` | tests: system_e2e wave 4 — update variants, chat-lineage cancel, absorb kill-recovery, interactive delegate_answer (S18-S23) |
| F4 wave 4 (tail) | `0196b3d7` | `fc12f7cc` | docs: regenerate the facade inventory for managed_update_remote_url (W2-F2 tail) |
| F5 lane D (CPL-5) | `50d68aaa` | `45904d87` | CPL-5: model-visible <=> logged invariant for model_send at the dispatch seam |
| F3.1 lane A | `0f715831` | `fa2f6fc5` | v7next F3.1 lane A: producer cutovers — core/shell/services/mcp/git and the six control leaves |
| ADOPTION truth wave | `fc1528d5` | `497333de` | test: pin the four F2.2 queue/worker handle leaves that never entered LEAVES |
| domain-placement base | `8e885412` | `93065f77` | docs: the 80 proposed domain placements are the accepted 7.0 base |

### Waves 3a, 3b and lane D named no SHAs of their own

Their sections cite only their base (`8827fd2c`), so there is nothing to
reconcile for them and this map must not pretend otherwise. What landed:

- **wave 3a** — `676ebe8c` (plan review, commit enforcement classes, acceptance
  loop) and its section `2021fba1`. Its scenarios were RENUMBERED at
  integration; the note at "Integration note for the F4 wave-3a section above"
  (`f71f4e42`) is the mapping and stays the authority.
- **wave 3b** — `13c93a2a` (delegated transport + skills lifecycle), plus
  `02feb1a3`, a size-ratchet band entry for `tests/system_e2e/harness.py` that
  integration added and the lane did not produce.
- **lane D** — `45904d87` (the CPL-5 row above) and its section `e26cfb61`.

### CPL-5 landed twice

`50d68aaa` was the lane's commit. It did not survive integration, and the
ADOPTION row that recorded CPL-5 as done was then OVERWRITTEN by the
persistence-fix union — so for a stretch the tree neither carried the lane's
SHA nor claimed the work. `45904d87` is the implementation as it actually
landed, and `bef13f5e` ("integration: restore the CPL-5 done row overwritten
by the persfix union (45904d87 truth)") is the repair of the claim. Both
halves are named here because the loss is the interesting fact: an ADOPTION
row can be un-done by a neighbouring merge without anything failing.

### SHAs that are correctly NOT on this branch

Not every unresolvable SHA is an integration defect. `9f691656` is the frozen
`ouroboros_v7_wip` reference oracle the transplant lanes were cut against, and
`5440e407`, `705ffc51`, `a5e1cea3` and `e3c107bd` are upstream/oracle-only
commits cited as provenance for a decision. They were never meant to land here
and are listed so a future reader stops looking for them.

## RES-15a — retraction: `GATEWAY_CONTRACT_VERSION` is not a browser ABI mirror

`ouroboros/gateway/schema.py` and `docs/ARCHITECTURE.md` both said that the
browser mirror's `GATEWAY_CONTRACT_VERSION` "still carries the pre-7.0
product-version spelling — its switch to this carrier is deferred with the rest
of the web mirror". Two things in that sentence are false on this tree.

1. **There is no "rest of the web mirror" left deferred.** The F3.3 comma-sweep
   section above records that the eight HOT-DEFERRED JSDoc lines were removed
   and the `_abi3_deferred_js_extras` excuse set in `tests/test_gateway_parity.py`
   was deleted — "the browser mirror is exact again". The prose kept pointing at
   a backlog that had already been paid.
2. **The switch is not deferred; it is refused.** `GATEWAY_CONTRACT_VERSION` is
   a RELEASE VERSION carrier, not an ABI mirror: `tests/test_gateway_parity.py`
   asserts it equals the `VERSION` file byte for byte, and
   `ouroboros/tools/release_sync.py` rewrites it beside `pyproject.toml`,
   `uv.lock`, `web/package.json` and the README badge on every release. Pointing
   it at `GATEWAY_ABI_VERSION` would break both. The name is the whole
   confusion — "CONTRACT" in a constant that carries a product version.

Corrected, not rewritten: the two prose sites now say what the carrier is and
why it will not move, and the F3.3 row above stays as written, because it was
right — it recorded the switch as deferred "to the release tact ... version
carriers move synchronously in release mechanics only", which is the same fact
this retraction generalises. The claim being retracted is the ARCHITECTURE /
schema.py framing of it as an outstanding web-mirror cleanup item.

## From the simplification lane (owner 16A, base 9238cc2d)

Nine accepted audit items, one single-intent commit each. The governing rule
throughout is docs/DEVELOPMENT.md "Paying down a size cap": when an item's fix
ran into a cap, the cap was paid down by SIMPLIFYING the module the change
belongs to — never by a new helper, wrapper or neighbour module, and never by
loosening a pin. Two items are reported as dispositions rather than done,
with numbers, because no honest simplification supported them.

| item | what was done | lines before -> after | pins |
|---|---|---|---|
| **CH-4** (MEDIUM) | `extension_child_catalog.skill_state_path` was a byte-for-byte re-derivation of `skill_loader.skill_state_dir_path` reached through a private back-import of `_skills_state_root` / `_sanitize_skill_name`. Deleted; `extension_loader` resolves through the SSOT, and the fix-round-6 pre-fence contract the duplicate's docstring carried moved onto `skill_state_dir_path` | child_catalog 271 -> 258 | 11 companion-race monkeypatch sites in `tests/test_extension_companion.py` follow the name; no pin weakened |
| **CH-1 part 1** (HIGH, paydown) | `extension_plugin_api.py` sat at 999/1000, which is why ~100 lines of companion policy were pushed out in the first place. Paid down by simplification: the route-method vocabulary had two implementations (`register_route` and the child-catalog re-check) and became one `contracts.plugin_api.normalize_extension_route_methods`; the `_StagedRegistrations.disposers` list had NO producer anywhere (no caller appended, no caller passed `extra=`) so the empty-loop disposer machinery is gone; `get_runtime_info` lost three aliased-import blocks and a `getattr` for a field `__init__` always sets; `log` and `_wrap_runtime_handler` lost dead statements | plugin_api 999 -> 962 | both route refusal messages byte-identical; `test_disposers_stay_out_of_the_plugin_api_surface` still green |
| **CH-1 part 2** (HIGH) | `materialize_companion_env(api, ...)`, which read five `PluginAPIImpl` privates from outside the class, is now the method `PluginAPIImpl._companion_env`. `companion_node_argv` / `companion_manifest_path_override` are `node_runtime.skill_node_argv` / `skill_manifest_owns_path`, beside `select_skill_node_runtime` and `skill_node_emergency_path_dir` — the module that already documents the npm-shebang contract they restated. The emergency PATH prepend had two copies (companion env, isolated-dep installer env) and became `node_runtime.prepend_skill_node_emergency_path`. The five surface kinds were enumerated three times and became `extension_registry_state.SURFACE_KINDS` | child_catalog 258 -> 154; plugin_api 962 -> 999; node_runtime 226 -> 271 | node-policy tests moved from the `platform_layer` re-export facade to `node_runtime`, where the policy lives; the `600 <= plugin_api <= 1000` band pin held, not raised |
| **CH-3** (MEDIUM) | 17 module-handle helpers with zero call sites deleted, together with docstrings promising call-time rebinding the leaf never performed | 13 lines each, 221 total | none loosened: `tests/test_module_handle_extraction.py::LEAVES` declares only leaves that DO read through a handle, and none of the 17 were in it (the table already records that precedent) |
| **CH-6** (LOW) | `provider_models.delegated_route_target(route)` — four `getattr` defaults over a frozen `str`-field dataclass, in another module — is `DelegationRoute.resolved_target()`. `delegate._start_request` now declares `route: "DelegationRoute"` instead of `Any` | provider_models 580 -> 562; subagents 1381 -> 1399 | the source-string pin `"delegated_route_target(route)" in source` is rewritten as behaviour: build a real run request and assert the wire body carries exactly the typed target's fields, plus the case the grep never covered (a bare route sends no empty `model`/`effort`/`credentialProfileId` keys) |
| **CH-7** (LOW) | `registry_core._resolve_node_postgates_predispatch` — a six-argument forward whose docstring said "off the hot dispatch body for the function-size gate" — inlined at its one caller, funded by simplifying `_execute_legacy_text`: a thrice-asked `_binding_set_targets_system_repo(...) or acting_self_worktree` computed once, an `implicit_skill_cwd_allowed` that recomputed the existing `heal_no_enable`, a double-negated branch order, and four one-argument-per-line guard calls | `_execute_legacy_text` 296 -> 263 (37 under the 300 gate) | function gate not raised |
| **CH-12 / CH-2 docs** (INFO/MEDIUM) | five `__import__("base64")` / `__import__("mimetypes")` calls became two module-level imports; the `core_artifacts` docstring and its ARCHITECTURE row now name the `escalate` verb and the validators they omitted | core_artifacts 522 -> 517 | none |
| **RES-14b** (class fix) | `append_jsonl` and `state.atomic_write_text` guarded their paths; every OTHER full-file writer went through `_atomic_overwrite` unguarded. The guard moves onto that seam, so `write_bytes_atomic`, `write_text_atomic`, `atomic_write_json` and `state.atomic_write_text` are all covered and the next writer is covered by construction | utils 1599 -> 1596 | red-first: `test_every_atomic_writer_fails_closed_on_live_root_write` observed failing (`DID NOT RAISE`) before the fix |
| **RES-27 / RES-15a** (docs) | the integration-corrections SHA map and the `GATEWAY_CONTRACT_VERSION` retraction, both in the sections immediately above | — | — |

### Two dispositions, not done

- **CH-7, `shell_process._publish_unfinished_process_facts`** — the audit called
  it a two-line passthrough cut for the 300-line gate. On these bytes it has
  TWO callers (the timeout and the pre-exec failure paths), it is one half of a
  pair with `_publish_finished_process_facts` in the module that owns process
  facts, and its docstring is the only statement that a child with no
  returncode publishes duration plus the attested substituted runtime and
  nothing else. The audit's premise is also stale: `_run_shell` is **265**
  lines, not 296, so it sits 35 under the gate and no size pressure argues for
  the change. Inlining would duplicate a lazy import across two except branches
  and delete a contract. NOT DONE, reported instead.
- **CH-2, moving the validators to "their owners"** — `validate_link_actions`
  and `validate_quiz_payload` define the wire shapes the verbs beside them
  emit, and `supervisor.message_bus` re-validates through the same functions
  rather than owning a copy. There is no more natural owner, so only the
  docs half of CH-2 landed. This is the "otherwise report" branch the item
  itself allows.

### Owner item 16 (A) — the pins that held a cap-driven placement

Only ONE existed. CH-6's `"delegated_route_target(route)" in source` named the
cross-module function and so pinned the PLACEMENT; it is rewritten as a
behaviour pin. Nothing pinned the companion env/argv placement — those names
were never in `test_extension_loader_extraction.py::_MOVED_OWNERS` — and
nothing pinned `_resolve_node_postgates_predispatch`. The extraction suite's
`600 <= extension_plugin_api <= 1000` band pins SIZE, not placement, and is
untouched: it held at 999 across the whole lane, which is the point.

### Residual disclosed

`extension_plugin_api.py` lands at **999/1000** — the same number it started
at, with ~85 lines of returned policy inside it and the paydown funding the
difference. That is one line of headroom, and it is honest to say so: the
next change there has to pay its own way in, exactly as this one did. The
alternative was to leave the companion env in a module that validates child
catalogs, which is the defect the item exists to close.

## Vision evidence for CHECKLISTS 2(i) on a70747cf (owner batch №11, 6=A)

28 headless-Chromium screenshots (14 consumer states × 1440x900 and 390x844)
captured on an isolated root from `git archive` of a70747cf through the repo's
own `direct_server_with_data` fixture and MockLLMServer (no provider keys):
chat rich markdown, document/audio cards, photo gallery + links (javascript:
action correctly not rendered), collapsed task cards, the acceptance
review-findings panel, plan-review findings, settings model slots, the Phase 3
extension settings section, Updates/controls + recovery, the skill card with
**Grant access as a primary card button**, review findings on the card, the
ClawHub marketplace, and the chat header agent controls. Vision inspection
of every PNG: PASS with minor caveats — no raw JSON, error banners,
overlapping text or horizontal overflow; cosmetic items for the backlog: a
~60-80 px empty body inside collapsed task cards, the `Online` badge wrapping
to a third header row at 390 px, and `Reset All Data` possibly sitting under
the sticky Reload/Save footer at 390 px (unverified whether at max scroll).
Capture gaps (script scroll offset, not rendering): the audio card at mobile
and its filename at desktop; not exercised: the post-click Grant access
face, the managed-update "apply" face (isolated checkout has no managed
remote), the OuroborosHub tab and the Widgets page. Files and manifest:
operator scratchpad `ui_evidence/a70747cf/` (session 3ab25cbc); the owner's
manual test on the STOP tree is the release-grade 2(i) evidence.
## From the D03 settings-read-seam lane (owner batch №11 2=A, base 1072a317)

The remainder of semantic delta D03 (§4.3.5): MIGRATION rows 913-917 and
1080-1081, hot-deferred since the D11/D12/D17/D18 lanes. Landed as the tip's own
code, re-derived on tip bytes — the delta is NOT span-shaped for
`scripts/v7next_transplant.py` (four of the six rows are rewritten bodies
against a read path, `settings_integrity`, that the oracle does not have; the one
pure relocation, row 913, is byte-identical and needed no tool proof).

1. Row 914 (`config.load_settings_lock_held` -> `config.normalize_settings_raw`)
   — LANDED. The loader's inline raw-stage block (coerce, retention fold,
   review-cycle seed, retired purge, slot rename, secret repair) collapsed onto
   the one pure seam; the loader keeps the tip's
   `settings_integrity.read_settings_json_verified` / `SettingsIntegrityError`
   raise-through UNDER the seam (the D12/D17 re-prove trap: replaying the oracle
   verbatim would have reverted the integrity feature). Docstring re-derived, not
   copied: the oracle's "singular scope pin promoted before the plural"
   ordering clause is DEAD on this tree (both comma spellings are
   `RETIRED_SETTING_KEYS`, ABI-10); the load-bearing order here is pass-count-
   before-purge (`OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES` is itself a
   retired key and must seed `OUROBOROS_REVIEW_MAX_CYCLES` before the purge
   drops it) and purge-before-rename. `_owner_read_settings_raw` applies the
   same seam before its defaults merge; the context-fit route resolver and the
   generic save inherit the fix through it.
2. Row 916 (`config.serialize_settings`) — LANDED. `config.save_settings` (both
   the atomic path and its OSError fallback) and `packaged_cli._save_settings`
   emit through it; `utils.atomic_write_json` already produced that text on the
   owner writer, so the function names bytes one writer already had. Visible
   change: a non-ASCII owner value is written as UTF-8 by all three rather than
   `\uXXXX`-escaped by two.
3. Row 913 (`gateway/onboarding._settings_fingerprint` ->
   `gateway/owner_settings.settings_document_digest`) — relocated byte-identical
   (same absent / never-equal-unreadable sentinels). DEVIATION from the row
   note: the onboarding one-line delegating wrapper was NOT kept — the tip's
   no-passthrough rule; both onboarding call sites (the pre-derive read and the
   locked precondition) call the digest directly, and the unreadable-sentinel
   pin moved with the function (`tests/test_onboarding_complete_endpoint.py::
   test_an_unreadable_settings_file_can_never_compare_equal` ->
   `tests/test_settings_read_seam.py::test_an_unreadable_settings_file_never_compares_equal`).
4. Row 915 (`_owner_write_settings` -> `_owner_update_settings`) — LANDED.
   `_owner_update_settings(transform, expected_digest, ...)` reads (through the
   normalized owner reader), transforms and persists inside ONE settings lock,
   with the persistence prologue now genuinely under the lock; a transform
   returning `None` writes nothing; a digest mismatch refuses with the typed
   `SettingsPreconditionFailed` before the transform runs (`STALE_SETTINGS_READ_REFUSAL`).
   `_owner_write_settings` keeps its name, signature and refusal types as the
   whole-document caller (generic save, onboarding, tests). The tip has FOUR
   single-decision endpoints, not the oracle's five (the scope-review floor is
   retired, ABI 7.0 Q10=A): runtime mode (digest before the deciding read),
   context mode (digest before the idle check, idle re-proved under the lock —
   the digest cannot see the queue), safety mode (digest under the document
   lock beside the audit read), auto-grant (no digest: the body carries the
   whole decision). The tip's `settings_document_mutation()` + `asyncio.to_thread`
   `_sync` structure is kept; the closed reader inventory is pinned at FIVE names
   (`_api_settings_post_locked` plus the four `_sync` bodies).
5. Row 917 (`packaged_cli._save_settings`) — LANDED through the shared prologue
   and serializer; the prologue tripwire now sees the parameter-path writer
   shape (`tests/test_runtime_mode_elevation.py::test_every_settings_writer_routes_through_the_shared_prologue`
   widened to the oracle's form: a function name carrying "settings" plus the
   `settings.json` literal trigger; routed owner = `_owner_update_settings`;
   the packaged saver asserted routed; the Colab generator exempted with the
   oracle's reason). The oracle's extra exemption for
   `tools/registry_guard_process._restore_owner_files` was not needed here (the
   widened scan does not flag it on this tree) and was not copied. DISCLOSED:
   `BootstrapContext.save_settings` has NO caller in `bootstrap_repo` on this
   tree (nor on the oracle) — the callback is bound by both `launcher.py` and
   `packaged_cli.py` and never invoked. Routing it keeps the writer inventory
   honest for the day it is called; retiring the dead field is a
   `launcher_bootstrap` contract change shared with the launcher, outside this
   lane — candidate cleanup row.
6. Rows 1080-1081 (`server.py::lifespan`) — LANDED. The guarded boot write
   (`if provider_defaults_changed and _settings_path.exists(): save_settings(...)`)
   and the lifespan's `SETTINGS_PATH` import are gone; the apply -> env-apply
   -> `initialize_runtime_mode_baseline()` order is unchanged. Pins flipped:
   `tests/test_onboarding_host.py::test_server_boot_normalization_carries_the_same_guard`
   -> `test_server_boot_never_writes_the_settings_file` (oracle bytes, asserted
   on the lifespan's syntax, not its text); `tests/test_server_runtime.py`'s
   server clause -> `"save_settings(" not in server_src` (the oracle's
   both-sides form the D17 lane deferred). `save_settings` stays imported in
   `server.py` for the gateway compatibility shim
   (`_gateway_settings.save_settings = save_settings`), which tests monkeypatch.
7. Pin port notes (`tests/test_settings_read_seam.py`, 18 tests): the oracle's
   `test_the_three_retired_timeout_knobs_are_gone_from_every_owner_surface`
   was NOT ported — D04's `tests/test_legacy_timeout_retirement.py` already
   carries that pin on this tree and a second copy would be a parallel pin;
   `test_a_live_setting_cannot_be_retired_by_accident` re-derived without the
   oracle's inert `ouroboros.contracts` import; `test_the_packaged_bootstrap_writes_the_path_the_prologue_reads`
   re-pinned on the tip's per-call `resolve_app_root` / `resolve_data_dir`
   (the D13 form) instead of the oracle's module-constant strings; the golden
   document carries the retired acceptance-pass count instead of the dead
   singular scope pin; `test_every_owner_endpoint_reaches_the_same_normalized_read`
   pins five readers. The write-seam suite's future-writer scan now also
   catches `_owner_update_settings(` call sites outside the locked writers.
8. Red-first table (isolated root, `1072a317` pre-fix tree, command:
   `pytest tests/test_settings_read_seam.py -rA`; output kept in the lane's
   scratch as `d03_red_first_prefix.txt`):

   | test | pre-fix | post-fix |
   |---|---|---|
   | test_owner_read_settings_raw_applies_the_same_normalization_as_load_settings | FAILED (owner reader served shipped defaults over legacy keys) | PASSED |
   | test_one_owner_endpoint_write_preserves_every_owner_customization | FAILED (auto-grant POST changed 6 keys, ghost survived) | PASSED |
   | test_every_owner_endpoint_reaches_the_same_normalized_read | FAILED (no locked primitive; reader inventory open) | PASSED |
   | test_normalize_settings_raw_is_idempotent | FAILED (AttributeError: no normalizer) | PASSED |
   | test_a_stale_owner_read_cannot_overwrite_a_change_it_never_saw | FAILED (ImportError: no digest / no primitive) | PASSED |
   | test_a_transform_that_returns_nothing_writes_nothing | FAILED (ImportError) | PASSED |
   | test_an_unreadable_settings_file_never_compares_equal | FAILED (ImportError: digest lived in onboarding) | PASSED |
   | test_all_three_writers_serialize_a_document_to_the_same_bytes | FAILED (config/packaged escaped non-ASCII; packaged skipped the prologue) | PASSED |
   | test_the_three_settings_writers_are_exactly_these_three | FAILED (owner write was `_owner_write_settings`, prologue outside the lock) | PASSED |
   | test_a_retired_key_is_dropped_by_every_reader | FAILED (owner reader served every retired key) | PASSED |
   | test_a_retired_key_leaves_the_file_on_the_next_owner_write | FAILED (ghosts written straight back) | PASSED |
   | test_load_settings_migrates_every_renamed_key_and_drops_the_retired_one | PASSED (golden: the loader already did this) | PASSED |
   | test_load_settings_coerces_declared_types_before_the_defaults_merge | PASSED (golden) | PASSED |
   | test_reading_settings_writes_nothing_to_disk | PASSED (golden) | PASSED |
   | test_the_one_read_that_writes_is_the_context_compatibility_migration | PASSED (golden) | PASSED |
   | test_the_packaged_bootstrap_writes_the_path_the_prologue_reads | PASSED (golden: path identity the routing relies on) | PASSED |
   | test_a_retired_key_is_absent_from_the_defaults_that_offer_it | PASSED (golden) | PASSED |
   | test_a_live_setting_cannot_be_retired_by_accident | PASSED (golden) | PASSED |

   11 of 18 observed RED before the fix; the 7 goldens characterize behaviour the
   seam must keep producing and cannot be red on a tree whose loader already
   migrates. CORRECTED in round 3: this table describes the round-1 FILE. The file
   as delivered carries 22 pins and 15 of them are red on a `1072a317` export — the
   11 here, the three round-2 pins (item 20) and
   `test_the_packaged_bootstrap_writes_the_path_the_prologue_reads`, which round 2
   turned behavioural and which this table still lists as a green golden. The scout's two repro scripts confirmed the three defect shapes on
   the pre-fix tree (6 keys changed by one auto-grant POST; ghost and legacy key
   both persisted; `config == owner` bytes False) and both come out clean on the
   fixed tree.
9. Adapted pins with no weakening: `tests/test_onboarding_host.py` (docstring
   bullet: never WRITES, not merely never CREATES), `tests/test_server_runtime.py`,
   `tests/test_runtime_mode_elevation.py` (tripwire), `tests/test_owner_settings_write_seam.py`
   (membership wording; the future-writer scan), `tests/test_onboarding_complete_endpoint.py`
   (the moved sentinel pin). ARCHITECTURE: rows for `server.py`, `config.py`,
   `packaged_cli.py`, `gateway/onboarding.py`, `gateway/owner_settings.py`,
   `gateway/settings.py`, the owner-settings prose paragraph, the §7 function
   list, and a new "Reading and writing the settings document" section (oracle
   text re-derived for four endpoints and the tip's ordering claim).
10. Residuals, disclosed: (a) the digest deliberately over-refuses (a write
    landing between digest and lock, or a formatting-only rewrite, answers 409
    `settings_precondition_failed` — one retry; the oracle's accepted D03
    trade-off, previously the endpoint re-read under the document lock and
    silently won); (b) `launcher.py::_save_settings` and
    `tools/control_runtime._set_tool_timeout` persist through
    `config.save_settings` (routed, file lock only) and stay outside the
    in-process document lock — pre-existing, unchanged by this lane; (c) the
    dead `BootstrapContext.save_settings` callback (item 5).
11. Gates (each a separate command, isolated root): targeted suites green
    (`tests/test_settings_read_seam.py` 18 passed; the settings/owner/onboarding
    family incl. `test_config_extraction.py`, `test_onboarding_host.py`,
    `test_owner_settings_write_seam.py`, `test_runtime_mode_elevation.py`,
    `test_runtime_mode_core.py`, `test_onboarding_complete_endpoint.py`,
    `test_cybergym_server.py` (the strict-snapshot refusal on the writer),
    `test_settings_budget_hotreload.py`, `test_settings_env_on_disk.py`,
    `test_ws5_carryover.py`, `test_mcp_api.py`, `test_scope_review.py`,
    `test_domain_manifest.py`, `test_generated_inventories.py`,
    `test_repo_health_smoke.py`, `test_smoke.py`); `ruff check . --select F`
    rc=0; `scripts/check_domains.py` rc=0 (no module row moved, no `--write`);
    `scripts/regenerate_size_ratchet.py --check` rc=0 (config.py 945/1000,
    owner_settings.py 364, gateway/settings.py 1429 inside its band, server.py
    1647 -> 1639 giant shrink-only); `scripts/regenerate_inventories.py --check`
    rc=0 (facade inventory unchanged); `scripts/v7next_adoption.py` rc=0;
    `--release` refuses exactly one row now (CPL-4 `in-progress`, the owner
    checkpoint — not this lane's); `git diff --check` rc=0.

### Fix round 2 (review findings on the lane, same base 1072a317)

12. Colab reader (MEDIUM) — LANDED. `colab_bootstrap.build_colab_settings` folds
    the Drive document it re-reads through `config.normalize_settings_raw` BEFORE
    the defaults merge (it applied `migrate_legacy_slot_keys` alone: the retired
    ghost survived, the folded retention and review-cycle customizations were
    replaced by their defaults and written back to Drive as owner choices — the
    §4.3.5 defect, one reader over); `write_colab_settings` emits
    `serialize_settings` bytes through `write_text_atomic` (it wrote
    `atomic_write_json(..., trailing_newline=True)`, a second spelling). The
    prologue exemption stays (foreign root: the prologue proves ratchets against
    THIS process's `SETTINGS_PATH`); its docstring no longer names the prologue
    function literally — the first draft did, and the tripwire read the
    docstring as routing, which made the exemption dead. Pin: spec §4.3.5-7's
    Colab fixture, `test_the_colab_re_run_reads_the_drive_document_through_the_same_normalization`.
    The oracle carries the same gap; parity is exceeded here, disclosed.
13. Boot-write claim corrected (MEDIUM) — LANDED at the route consumers. Item 6
    ("every reader re-derives it from the same seam") was false for the PROVIDER
    normalization: `normalize_settings_raw` carries the vocabulary normalization
    only and `_owner_read_settings_raw` never applied
    `apply_runtime_provider_defaults`, so `context_fit.resolve_context_fit_route`
    and `_failed_route_evidence` (through `_active_main_route`) probed the
    owner-raw route — on a direct-provider install with no explicit model the
    OpenRouter-form shipped default instead of the `anthropic::…` / `openai::…`
    route the loop runs; the base tree agreed from the first boot on only because
    the retired boot write had persisted the normalization, the tip never agreed
    until a generic POST or onboarding rewrote the file. Both resolvers now read
    `apply_runtime_provider_defaults(load_settings())[0]` — the derivation the
    task-start projection (`subagent_runtime.apply_task_start_settings`), the
    settings GET and the onboarding reads already make; nothing new. NOT changed:
    `_owner_read_settings_raw` (carrying provider defaults there would change
    what the four single-decision endpoints write back — an owner fork, see
    residual (a)). server.py's boot comment, ARCHITECTURE (the server row, the
    context_fit row, the seam section and its closing sentence) now say which
    normalization the read seam carries and who re-derives the other. Item 1's
    clause "the context-fit route resolver … inherit[s] the fix through it" is
    superseded: the resolver no longer reads through the owner reader at all.
    Pin: `test_the_context_fit_route_is_the_provider_normalized_effective_route`.
14. Owner reader through the verified primitive (LOW) — LANDED.
    `_owner_read_settings_raw` reads via
    `settings_integrity.read_settings_json_verified` and re-raises
    `SettingsIntegrityError` past its defaults fallback (it read
    `json.loads(SETTINGS_PATH.read_text())` under a broad except, so a mismatched
    `OUROBOROS_SETTINGS_SHA256` pin served the unverified file while
    `load_settings` refused). Pin:
    `test_a_pinned_snapshot_that_changed_refuses_every_reader` (both readers and
    the context-fit route refuse; the matching digest serves through all three).
15. Writer tripwires (LOW) — LANDED. Both scans see `write_text_atomic(` and
    `.write_bytes(`; the prologue tripwire scans `launcher.py` and
    `supervisor/**` too, the seam inventory scans `launcher.py` (its
    `_save_settings` delegates to `config.save_settings`; no direct writer
    exists in either root today). Red-first by injection: a synthetic
    `write_text_atomic(SETTINGS_PATH, serialize_settings(...))` function appended
    to an export's `config.py` passed BOTH old scans (2 passed) and fails BOTH new
    scans naming it.
16. Pins made behavioural (LOW) — LANDED. (1) the three-writer bytes test drives
    `_owner_update_settings(lambda _c: dict(document))` rather than calling the
    writer's helper itself; (2) the packaged path identity is the computed
    property (`resolve_packaged_runtime().data_dir / settings.json ==
    config.resolve_data_dir() / settings.json` with the four path variables
    deleted and the bundle finders stubbed) plus the bootstrap wiring proven by
    capturing the `BootstrapContext` and driving its saver; (3)
    `tests/test_onboarding_host.py::test_server_boot_leaves_the_settings_bytes_alone`
    boots the REAL `server.lifespan` (TestClient; the extension-suite stub set —
    no supervisor, no host-service uvicorn, no skills seeding; the settings
    read, the provider normalization and every writer stay real) over a document
    carrying a retired model default and asserts bytes and mtime unchanged — red
    on base 1072a317 ("boot rewrote the settings document"), green on the tip.
    The syntactic pin stays as the fast tripwire.
17. DEVELOPMENT.md (LOW) — membership sentence synced with the module docstring
    and ARCHITECTURE: `_owner_update_settings`, directly or through
    `_owner_write_settings`.
18. Packaged saver path identity (LOW) — DISCLOSED, not re-derived. The
    `packaged_cli._save_settings` docstring now states the identity holds for an
    outer packaged process carrying no `OUROBOROS_*` path override: the packaged
    runtime ignores the environment by design (the inner CLI child is handed the
    packaged paths explicitly) while the prologue proves against
    `config.SETTINGS_PATH`, which honours one. Deriving the saver's target from
    `config.SETTINGS_PATH` was rejected: the OUTER process would then write the
    env-overridden path while the inner child reads the packaged one — a worse
    split than the disclosed one. Dormant (item 5: the callback has no caller).
19. ARCHITECTURE "three surfaces" (LOW) — qualified: three writers of THIS
    process's document through prologue and serializer; the two exempt writers
    named (the raw context-pair migration under the load lock; the Colab
    generator for the Drive root, serializer bytes, no prologue); the scanned
    roots stated.
20. Red-first table, round 2 (counts CORRECTED in round 3: the table below is the
    round-2 DELTA, not the file total — on a `1072a317` export the delivered 22-pin
    file is red on 15, the three rows here included). Isolated roots; "pre-fix" = the lane HEAD 1b80a38a
    exported with the new pins overlaid, i.e. before the round-2 code; the boot
    pin additionally against the base export; outputs kept in the lane scratch as
    `d03_r2_*.txt`):

    | test | pre-fix | post-fix |
    |---|---|---|
    | test_the_colab_re_run_reads_the_drive_document_through_the_same_normalization | FAILED (OUROBOROS_GC_RETENTION_DAYS 7 != 30: the folded retention customization replaced by its default) | PASSED |
    | test_the_context_fit_route_is_the_provider_normalized_effective_route | FAILED (route == the owner-raw openrouter route, not the anthropic one) | PASSED |
    | test_a_pinned_snapshot_that_changed_refuses_every_reader | FAILED (DID NOT RAISE SettingsIntegrityError: the owner reader served the unverified file) | PASSED |
    | test_server_boot_leaves_the_settings_bytes_alone | FAILED on base 1072a317 ("boot rewrote the settings document"); PASSED on the lane HEAD | PASSED |
    | test_every_settings_writer_routes_through_the_shared_prologue / test_the_three_settings_writers_are_exactly_these_three, injected `write_text_atomic` writer | old scans: 2 PASSED (blind); new scans: 2 FAILED naming `_fourth_settings_writer_probe` | (injection removed) PASSED |
    | test_all_three_writers_serialize_a_document_to_the_same_bytes (real locked writer) | PASSED (golden, strengthened) | PASSED |
    | test_the_packaged_bootstrap_writes_the_path_the_prologue_reads (property) | PASSED (golden, strengthened) | PASSED |

21. Gates, round 2 (each a separate command, isolated root):
    `tests/test_settings_read_seam.py` (21) + `test_onboarding_host.py` +
    `test_colab_bootstrap.py` + `test_model_slot_role_model.py` +
    `test_context_fit_v664.py` + `test_owner_settings_write_seam.py` = 174
    passed; the settings/owner/onboarding/server family of item 11 plus
    `test_startup_hygiene.py`, `test_extensions_api.py`, `test_launcher_sync.py`,
    `test_launcher_headless_fallback.py`, `test_settings_secret_mask.py`,
    `test_max_context_gate.py`, `test_settings_honesty.py`,
    `test_onboarding_wizard.py`, `test_packaged_runtime_and_lifecycle.py`,
    `test_legacy_timeout_retirement.py`, `test_server_extraction.py` = 1174
    passed, 1 skipped; `ruff check . --select F` rc=0; `scripts/check_domains.py`
    rc=0; `scripts/regenerate_size_ratchet.py --check` rc=0 (the giant
    `tests/test_runtime_mode_elevation.py` stays at 2222 lines);
    `scripts/regenerate_inventories.py --check` rc=0; `scripts/v7next_adoption.py`
    rc=0 (`--release` still refuses exactly CPL-4); `git diff --check` rc=0.
22. Residuals, disclosed: (a) whether `_owner_read_settings_raw` should ALSO
    carry the provider normalization (which would persist it as owner choices on
    every owner-endpoint write) is an owner fork, not taken — the fix stands at
    the consumers; (b) under a mismatched benchmark pin `_failed_route_evidence`
    now raises `SettingsIntegrityError` instead of answering a route from the
    unverified document, so a pinned child whose snapshot changed fails its task
    loudly (fail-closed under the trust root; unreachable outside a strict pin);
    (c) the behavioural boot pin stubs the supervisor, host-service and
    skills-seeding halves of the lifespan — the segment under test is the settings
    read, the normalization and the absence of a write, which is what the pin is
    about; (d) the Colab quickstart's own pre-import read of the Drive file
    (`notebooks/colab_quickstart.py`, the update-channel bootstrap) stays raw by
    necessity — it runs before the runtime is importable and only feeds the
    document to `build_colab_settings`, which now normalizes it.

### Fix round 3 (review findings on the lane, same base 1072a317)

23. Prologue tripwire reads CALLS, not prose (MEDIUM) — LANDED. The whole-tree scan
    classified a function as routed when the string `prepare_settings_for_persist`
    appeared anywhere in its SOURCE SEGMENT — docstring included — so a writer that
    merely NAMED the prologue was vouched for. Proven by injection on an export of
    `2593a248`: an unrouted `write_text_atomic(SETTINGS_PATH, serialize_settings(...))`
    whose docstring names the prologue passes BOTH scans when it lives in a file the
    six-file seam inventory does not enumerate (`ouroboros/gateway/settings.py`); in
    `config.py` only the enumerating seam test caught it. Routing is now
    `any(ast.Call whose func id/attr == "prepare_settings_for_persist")` over the
    function. The fail-open direction is the reason this mattered: item 12 met the
    INVERSE of the same defect (a docstring made an exemption dead) and answered it by
    rewording the docstring, which left the fail-open half standing. The round-2
    constraint on `colab_bootstrap.write_colab_settings`'s docstring is therefore
    LIFTED — it may name the prologue function again; its current wording is kept only
    because "deliberately NOT routed through the persistence prologue" reads better.
24. Writer-scan triggers (LOW) — LANDED. A function entered the scan only through
    `SETTINGS_PATH` or a "settings" in its own NAME, so a writer taking its path as a
    parameter under a neutral name was invisible: an injected
    `_seed_child_document(data_dir)` in `ouroboros/headless.py` writing
    `serialize_settings(...)` to `data_dir / "settings.json"` passed both scans on the
    same export (item 15's injection named `SETTINGS_PATH`, i.e. the trigger itself, so
    it never exercised this class). Calling the serializer, and naming the settings
    file, are now triggers in their own right. On this tree the widening flags nothing
    new — the same 7 functions, 3 routed, 4 exempt, no dead exemption — so the
    ARCHITECTURE sentence about the scanned roots is now delivered rather than claimed,
    and it states the triggers.
25. "One spelling on disk" made TRUE, and pinned as what is actually guaranteed
    (MEDIUM) — LANDED. Two of the three writers still committed through
    `Path.write_text` (`config.save_settings` and its OSError fallback,
    `packaged_cli._save_settings`), which translates `\n` to `\r\n` on Windows, while
    the owner writer committed byte-exactly through `atomic_write_json` ->
    `write_text_atomic`. On the 3-OS matrix the same document therefore had ONE
    spelling on the Linux legs and TWO on the Windows one, and `serialize_settings`'s
    docstring, the ARCHITECTURE seam section and item 2 all asserted otherwise. All
    three writers now commit `serialize_settings()` output through
    `utils.write_text_atomic`; the owner writer calls the serializer itself instead of
    re-deriving the same JSON text inside `atomic_write_json`, so "one serializer" is
    the code rather than two spellings pinned equal (which was the parallel-authority
    shape). The config saver keeps its `except OSError` in-place fallback for a
    filesystem that cannot rename a sibling — pre-existing, and now byte-exact too.
    Three properties come WITH the shared helper rather than being added: the temp
    sibling carries the atomic signature `sweep_stale_temp_files` recognises (the old
    `settings.tmp` never did), the existing permission bits survive the replace (a
    0600 settings.json previously reset to the umask default on every save), and the
    `_atomic_overwrite` pytest live-data guard now covers this surface too. The PIN is
    split to match what each half can prove: the byte comparison (now `read_bytes`,
    and against the serializer's own output, so a trailing newline or a translation
    between serializer and disk fails it) proves the platform it runs on; the
    cross-platform half is pinned on the MECHANISM — no settings writer may commit
    through a text-mode write — because no run on a Linux leg can observe a Windows
    translation. Adapted patch site: `tests/test_cybergym_server.py` patches
    `owner_settings.write_text_atomic` as its write sentinel (same assertion).
26. Packaged saver takes the write guards (LOW) — LANDED.
    `packaged_cli._save_settings` called neither `config._guard_live_settings_write()`
    nor `settings_integrity.guard_live_settings_write`, so under a strict
    `OUROBOROS_SETTINGS_SHA256` pin, or from pytest against the live install path it
    resolves by construction, a bootstrap save was the one settings write that could
    land. It now guards the path it ACTUALLY writes (not `config.SETTINGS_PATH`),
    which also keeps the disclosed override split of item 18 honest. Pin:
    `test_the_packaged_bootstrap_writes_the_path_the_prologue_reads` drives the
    captured `BootstrapContext` saver under a mismatched pin and requires
    `SettingsIntegrityError` with the bytes unchanged.
27. Context-fit pin taken from the LOOP side (LOW) — LANDED. The round-2 pin computed
    its expectation as `_active_main_route(apply_runtime_provider_defaults(load_settings())[0])`
    — the resolver's own expression — so implementation and expectation could drift
    together and stay green on the stated contract ("the route the loop runs"). The pin
    now calls `subagent_runtime.apply_task_start_settings()`, which is what a task start
    projects into the environment, takes `OUROBOROS_MODEL` from there, ROLLS THAT
    PROJECTION BACK, and requires the resolver to reach the same model on its own. Name
    kept (`test_the_context_fit_route_is_the_provider_normalized_effective_route`) so
    items 13/20 and the ADOPTION hook keep resolving.
28. Retired-key classification, class-shaped (LOW) — LANDED, and the oracle's rule is
    CORRECTED rather than ported. Item 7 declined the oracle's
    `test_the_three_retired_timeout_knobs_are_gone_from_every_owner_surface` because
    D04 carries that pin; D04 covers only the SOFT/HARD pair, so the classification
    surfaces were unpinned for the other thirteen retired keys. The new
    `test_a_retired_key_is_absent_from_every_surface_that_would_react_to_it` walks all
    fifteen: absent from `gateway/settings._IMMEDIATE_KEYS` and
    `_RESTART_REQUIRED_KEYS`. The oracle's docs clause ("not the docs") is REJECTED
    with evidence: three retired keys carry honest ARCHITECTURE rows today
    (`OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES` as `(retired)`,
    `OUROBOROS_REVIEW_MODELS` and `OUROBOROS_SCOPE_REVIEW_MODELS` as `(env-only)` with
    "RETIRED settings key" in the body), and deleting them would remove the only place
    an owner learns where the value they wrote went. The defect class is a retired key
    still being OFFERED, so the pin is on the DEFAULT column: it must be a status
    marker, never a value.
29. Two residuals of this lane's own pins closed (LOW). (a) The behavioural boot pin
    left `server._boot_managed_update_tasks` real — a daemon thread that finalizes a
    pending managed update and refreshes its feed, i.e. runs git against whatever
    `REPO_DIR` resolves to in the process running pytest, and whose work races the
    assertion anyway. Stubbed, with the reason in the docstring. (b) "The closed
    inventory of readers" over-claimed: the scan closes `gateway/settings.py` and
    `gateway/owner_settings.py`, which is where an owner endpoint could grow a second
    reader, while `gateway/onboarding.py:658` calls the same reader for its subagent
    preview — the seam used as intended. Both docstrings now say what is closed.
30. ARCHITECTURE and stale wording (LOW) — LANDED. The seam section states the
    two-part guarantee (one serializer AND one byte-exact commit) instead of "serialize
    through `serialize_settings()`", which was false for the owner writer, and names
    the scan's triggers plus the calls-not-prose routing rule. The `packaged_cli.py`
    module row says "owns its own path but takes the write guards on it" instead of
    "owns its own path and atomic rename". The startup rule at the Onboarding Flow
    section says the boot normalization may not WRITE `settings.json` AT ALL (it still
    said "may CREATE", the weaker pre-lane rule, while this lane's pins and comments
    say never write). `owner_settings._owner_write_settings`'s docstring says "every
    persisting writer" (it said "both", while the tripwire asserts three) and the
    module docstring's commit boundary names `write_text_atomic`.
31. Red-first counts corrected (LOW) — item 8, item 20 and the ADOPTION D03 row now
    state the delivered file's number (15 of 22 red on a `1072a317` export) beside the
    round-1 draft's (11 of 18), and name the packaged-path pin as red on base since its
    round-2 strengthening. The seven that stay green on base are goldens, the new
    classification pin among them — the tip's classification was already clean, and the
    pin exists so the next retirement stays that way.
32. Red-first table, round 3 (isolated roots; exports of `2593a248` for the code fixes,
    of `1b80a38a` for the round-2-era context-fit shape, of `1072a317` for the base):

    | pin / probe | pre-fix | post-fix |
    |---|---|---|
    | injected unrouted writer whose DOCSTRING names the prologue (`ouroboros/gateway/settings.py`) | old scans: 2 PASSED (blind — the substring vouched for it, and the six-file inventory does not enumerate that module); new tripwire: FAILED naming `_docstring_vouched_settings_writer` | (injection removed) PASSED |
    | injected neutral-named serializer writer (`ouroboros/headless.py`, path from a parameter) | old scans: 2 PASSED (blind); new tripwire: FAILED naming `_seed_child_document` | (injection removed) PASSED |
    | test_all_three_writers_serialize_a_document_to_the_same_bytes (mechanism half) | FAILED on the `2593a248` export: "ouroboros/config.py::save_settings commits a settings document through a text-mode write" | PASSED |
    | test_the_packaged_bootstrap_writes_the_path_the_prologue_reads (pinned-snapshot refusal) | FAILED on the `2593a248` export: "DID NOT RAISE SettingsIntegrityError" | PASSED |
    | test_the_context_fit_route_is_the_provider_normalized_effective_route (loop-side expectation) | FAILED on the `1b80a38a` export: `'google/gemini-3.7-flash' != 'anthropic::claude-opus-5'` — "the probe sized a model the loop never runs" | PASSED |
    | test_a_retired_key_is_absent_from_every_surface_that_would_react_to_it | mutation A (a retired key put back into `_IMMEDIATE_KEYS`): FAILED; mutation B (the `(retired)` default column replaced by `1`): FAILED; unmutated `2593a248` and `1072a317` exports: PASSED (golden) | PASSED |
    | test_server_boot_leaves_the_settings_bytes_alone / test_server_boot_never_writes_the_settings_file (re-verified with the update-thread stub) | FAILED on the `1072a317` export ("boot rewrote the settings document"; `save_settings` still named) | PASSED |
    | tests/test_settings_read_seam.py, delivered 22-pin file | 15 FAILED on the `1072a317` export | 22 PASSED |

33. Gates, round 3 (each a separate command, isolated root, `LC_ALL=C`): the targeted
    family `tests/test_settings_read_seam.py` (22) + `test_config_extraction.py` +
    `test_onboarding_host.py` + `test_runtime_mode_elevation.py` +
    `test_owner_settings_write_seam.py` + `test_colab_bootstrap.py` +
    `test_context_fit_v664.py` rc=0; the wider settings/owner/onboarding/server family
    (that set plus `test_runtime_mode_core.py`, `test_onboarding_complete_endpoint.py`,
    `test_cybergym_server.py`, `test_settings_budget_hotreload.py`,
    `test_settings_env_on_disk.py`, `test_model_slot_role_model.py`,
    `test_startup_hygiene.py`, `test_launcher_sync.py`,
    `test_launcher_headless_fallback.py`, `test_settings_secret_mask.py`,
    `test_max_context_gate.py`, `test_settings_honesty.py`,
    `test_onboarding_wizard.py`, `test_packaged_runtime_and_lifecycle.py`,
    `test_legacy_timeout_retirement.py`, `test_server_extraction.py`,
    `test_server_runtime.py`, `test_atomic_write_v639.py`) rc=0;
    the FULL default suite 13903 passed / 3 skipped rc=0 and the serial lane 623 passed
    / 46 skipped rc=0 (a core writer changed, so the narrow families are not the
    evidence); `ruff check . --select F` rc=0; `scripts/check_domains.py` rc=0;
    `scripts/regenerate_size_ratchet.py --check` rc=0 at the tip and on each of the
    round's four commit TREES (config.py 945 -> 946 of 1000; owner_settings.py 368;
    packaged_cli.py 485; gateway/settings.py 1429 unchanged; server.py 1643 unchanged;
    the giant `tests/test_runtime_mode_elevation.py` 2222 -> 2232), the CI-blocking
    `-m size_ratchet` lane rc=0, and
    `review.validate_size_ratchet_transition_against_base(repo, "1072a317")` == [] —
    this branch's contract is tip exactness plus that pairwise transition, with no
    committed-history replay (`review.validate_size_ratchet` docstring), so "every
    commit" here means every commit's tree passes the tip check, not a replayed audit;
    `scripts/regenerate_inventories.py --check` rc=0; `scripts/v7next_adoption.py` rc=0;
    `git diff --check` rc=0. Hermeticity: `find ~/ouro/data -newermt <start>` lists only
    the two `data/claudexor/daemon/` files the running daemon writes; the live repo
    stayed clean.
34. Residuals after round 3, disclosed. (a) Everything in items 10 and 22 stands: the
    digest's deliberate over-refusal, the two `config.save_settings` callers outside the
    in-process document lock, the dead `BootstrapContext.save_settings` callback,
    `_owner_read_settings_raw` not carrying the provider normalization (an owner fork),
    `_failed_route_evidence` raising under a mismatched pin, the boot pin's stubs, and
    the Colab quickstart's pre-import raw read. (b) NOT taken: the onboarding
    transaction still asks the staleness question with its own compare inside
    `_write_precondition` instead of handing `expected_digest` to the primitive — one
    digest function, two compares, oracle parity; collapsing them changes an
    install-time transaction's refusal shape, which is not a fix-round change. (c) NOT
    taken: `config.save_settings`'s `except OSError` fallback is still a non-atomic
    in-place write on a filesystem that cannot rename a sibling — pre-existing
    behaviour, kept deliberately and now byte-exact. (d) The stub in 29(a) is this
    lane's pin only; the same unstubbed boot thread in the pre-existing extension
    lifespan tests is not this lane's to change.

### Fix round 4 (review findings on the lane, same base 1072a317)

35. The context-pair migration commits the one serializer's bytes (MEDIUM) — LANDED.
    `context_mode_compat.normalize_and_persist_context_mode_compat` wrote
    `atomic_write_json(settings_path, normalized, trailing_newline=False)`, whose text is
    `utils.atomic_write_json`'s own `json.dumps` — the parallel-authority shape item 25 removed
    from the owner writer, one file over. Item 25's "'one serializer' is the code rather than
    two spellings pinned equal", the seam file's "the byte/mechanism pin proves all five put the
    same spelling on disk" and config.py's "every one of them commits through
    `write_text_atomic`" were therefore true only while nobody changed the serializer, and
    unpinned: on a `410613f7` export the serializer mutated to append `\n` left the whole seam
    family green while `config.save_settings` and the migration wrote two spellings of the same
    file (the round-3 verdict's proof, reproduced here). It now writes
    `write_text_atomic(settings_path, serialize_settings(normalized))`, the serializer bound
    late inside the function because `config` imports this module. Bytes are unchanged today
    (both spellings were `json.dumps(..., ensure_ascii=False, indent=2)` with no trailing
    newline). Adapted patch sites: the two write-failure injectors in
    tests/test_settings_budget_hotreload.py patch `context_mode_compat.write_text_atomic`.
36. One writer predicate, closed over the tree, seeing a plain-handle write (one MEDIUM per
    lens) — LANDED. The two scans had drifted into two predicates: the tripwire's regex (five
    spellings, `json.dump`, no `os.replace`, evaluated per function) and the seam inventory's
    tuple (five spellings, `os.replace`, no `json.dump`, evaluated per CALL text, six
    enumerated files). Neither saw `open(path, "w")` / `handle.write(...)`, so an unrouted
    saver writing `SETTINGS_PATH` through a text-mode handle passed every pin — even appended to
    `config.py`, the file both scans enumerate — and a ROUTED sixth writer outside the six files
    passed both as well. Both scans now read `tests._shared.settings_writers`, one predicate in
    the suite's existing shared-helper module (whose stated purpose this is; the seam file and
    the giant tripwire file both SHRANK): a function is a writer when it CALLS
    `serialize_settings` (a settings document by definition), or when it names the settings path
    or file — `SETTINGS_PATH`, a `settings_path` parameter (the case-insensitive form retires the
    tripwire's `atomic_write_json(settings_path` special case), the `settings.json` literal — or
    carries "settings" in its own name AND does a write-shaped thing (`.write_text`,
    `.write_bytes`, `.write`, `json.dump`, or an atomic helper). The tripwire additionally
    asserts the flagged set EQUALS `tests._shared.SETTINGS_WRITERS` plus its declared non-writer
    matches, so the inventory is closed over `ouroboros/**`, `supervisor/**`, `server.py` and
    `launcher.py`, routed or not; the seam file's six-file inventory
    (`test_the_three_settings_writers_are_exactly_these_three`) is deleted as the weaker copy.
    On this tree the predicate flags exactly the same 7 functions (5 writers, 2 non-writer
    matches) before and after item 35.
37. The file-level prefilter (MEDIUM) — REMOVED rather than widened. The tripwire skipped any
    module carrying none of `SETTINGS_PATH` / `atomic_write_json` / `settings.json`, so the
    round-3 `serialize_settings(` trigger could never fire in such a module: the round-3
    red-first row used `headless.py`, which carries a `settings.json` literal — the token that
    admitted it — so the trigger it claimed to add was never exercised (reproduced: `cli.py`,
    carrying none of the three, with an injected `_persist_owner_document` was blind). Both
    verdicts' fix (one more token in the conjunction) would have been the instance; the class is
    "a prefilter narrows the predicate", so the scan now parses every file. Affordable because
    the function source is sliced by line numbers (`node.lineno..end_lineno`) instead of
    `ast.get_source_segment`, which re-split the whole module per function: the tripwire runs
    in 2.5s against the tree where the prefiltered scan took 13s.
38. The mechanism half is a positive rule (one LOW per lens) — LANDED. `".write_text(" not in
    body` denied one spelling; a packaged saver rewritten to commit through `open(tmp, "w")` +
    `handle.write(...)` + `replace_atomic` passed it. The byte pin now `findall`s every
    commit-shaped token in each writer's source (`write_text_atomic(`, `write_bytes_atomic(`,
    `atomic_write_json(`, `.write_bytes(`, `.write_text(`, `.write(`, `json.dump(`, `open(`)
    and requires the set to be non-empty and a subset of the byte-exact four.
39. The byte pin drives all five writers, requires the serializer CALL, and pins the spelling
    — LANDED. `test_all_three_writers_serialize_a_document_to_the_same_bytes` is now
    `test_every_settings_writer_puts_the_one_serializers_bytes_on_disk`: it drives the config
    saver, the locked owner writer, the packaged saver, the migration (a `load_settings` over
    an ambiguous Low — the one read that writes) and the Colab generator through their real
    entry points, compares each file to `serialize_settings(json.loads(file))` bytes, asserts
    by AST that each entry of `SETTINGS_WRITERS` calls `serialize_settings`, applies item 38's
    rule, and pins the spelling as a golden. The golden is deliberate and disclosed: once every
    writer calls the serializer, the agreement assertions cannot see the serializer change at
    all (after item 35 the round-3 mutation is green on every writer at once), and the spelling
    — UTF-8, two-space indent, no trailing newline — is the choice the three spelling defects of
    rounds 1-3 converged on and what the digest precondition, a pinned benchmark snapshot and
    the migration's convergence observe. It is the one half of the pin that still reads the
    serializer, and the one that makes the mutation red on the fixed tree.
40. Wording matched to the code (LOWs) — LANDED. config.py's `serialize_settings` docstring:
    "each of the five calls this function and commits through a byte-exact helper
    (`write_text_atomic`, or `Path.write_bytes` on `save_settings`'s rename-less fallback)"
    instead of "every one of them commits through `write_text_atomic`" (false for the config
    saver's own `OSError` branch three lines below it). ARCHITECTURE's seam paragraph: five
    writers, the three routed and the two exempt named, the byte-exact commit with its
    fallback, the one predicate with BOTH halves of its AND stated, the closed-set rule. The seam
    file's "Three write THIS process's document; the other two ..." comment and the deleted
    inventory docstring ("No fourth writer ... all three") are gone with what they annotated; the
    tripwire's exempt reason for the migration says "the raw document with only its context
    compatibility pair changed, in serializer bytes". Round-1/2/3 table rows keep the old test
    names as history.
41. Red-first table, round 4 (isolated roots, `LC_ALL=C`; "old pins" = the `410613f7` export as
    delivered, "new pins" = the round-4 test files; probe trees are `git archive` exports and
    rsync copies under the lane scratch, never the worktree):

    | pin / probe | pre-fix | post-fix |
    |---|---|---|
    | test_every_settings_writer_puts_the_one_serializers_bytes_on_disk, calls-the-serializer half | FAILED on the `410613f7` export with the new pins overlaid: "ouroboros/context_mode_compat.py::normalize_and_persist_context_mode_compat persists a settings document without calling the one serializer" | PASSED |
    | the same pin, `serialize_settings` mutated to append `\n` | FAILED on the `410613f7` export at the migration's bytes (`b'...}' != b'...}\n'` — the round-3 verdict's two spellings, now caught); on the FIXED tree the same mutation fails at the golden ALONE ("the one on-disk spelling changed") with all five writers still agreeing | unmutated: PASSED |
    | test_context_mode_compat_migration_write_failure_is_honest_and_nonfatal / ..._does_not_rewrite_unchanged_raw_pair (injector on `write_text_atomic`) | FAILED on the `410613f7` export: `AttributeError: ... has no attribute 'write_text_atomic'` — the module's write went through `atomic_write_json` | PASSED |
    | P1 `config.py::save_settings_document` — unrouted, `SETTINGS_PATH.open("w")` + `fh.write(serialize_settings(...))` | old pins: 3 PASSED (blind); new tripwire: FAILED naming `('ouroboros/config.py', 'save_settings_document')` as unrouted | (injection removed) PASSED |
    | P2 `cli.py::_persist_owner_document` — `write_text_atomic` + serializer in a module carrying none of the old prefilter tokens | old pins: 3 PASSED (blind); new tripwire: FAILED naming it | PASSED |
    | P3 `headless.py::_seed_child_document` — neutral name, `open(path, "w")` + `handle.write(serialize_settings(...))` | old pins: 3 PASSED (blind); new tripwire: FAILED naming it | PASSED |
    | P4 `packaged_cli._save_settings` committing through `open(tmp, "w")` + `handle.write` + `replace_atomic` | old mechanism pin: PASSED (blind); the two old scans went red only because the text-mode saver VANISHED from their inventory (`assert None is True`; the packaged writer missing from the six-file set) — the fail-open shape itself; new byte pin: FAILED "commits a settings document through ['.write(', 'open(']" | PASSED |
    | P5 `headless.py::_seed_child_settings` — a ROUTED sixth writer (prologue + serializer + `write_text_atomic`) | old pins: 3 PASSED (blind, both scans); new tripwire: FAILED "the settings-writer inventory drifted: [('ouroboros/headless.py', '_seed_child_settings')]" | PASSED |
    | tests/test_settings_read_seam.py, delivered 21-pin file | 14 FAILED on the `1072a317` export (the round-3 count of 15 minus the deleted six-file inventory pin; the renamed byte pin is among the 14) | 21 PASSED |

42. Gates, round 4 (each a separate command, isolated root, `LC_ALL=C`): the targeted family
    `tests/test_settings_read_seam.py` (21) + `test_runtime_mode_elevation.py` +
    `test_settings_budget_hotreload.py` + `test_config_extraction.py` + `test_onboarding_host.py`
    + `test_owner_settings_write_seam.py` + `test_colab_bootstrap.py` + `test_context_fit_v664.py`
    + `test_cybergym_server.py` rc=0 (112 passed); the wider settings/owner/onboarding/server
    family of item 33 (25 files) rc=0; the FULL default suite on a shared clone of `f0de05e5`
    rc=0 — 14525 passed / 22 skipped, counted from the progress markers because this harness prints no summary line under `-q` (the count exceeds round 3's 13903 for the same reason the round-3 verdict's 14174 did: the clone environment, not this lane) and the serial lane rc=0 (623 passed / 46 skipped, same marker count); `ruff check . --select F`
    rc=0; `scripts/check_domains.py` rc=0; `scripts/regenerate_size_ratchet.py --check` rc=0 at
    the tip and on each of the round's commit trees (config.py 946 -> 949 of 1000;
    context_mode_compat.py 90 -> 96; tests/_shared.py 83 -> 150; tests/test_settings_read_seam.py
    739 -> 728; the giant tests/test_runtime_mode_elevation.py 2232 -> 2217, shrink);
    `scripts/regenerate_inventories.py --check` rc=0; `scripts/v7next_adoption.py` rc=0
    (`--release` still refuses exactly CPL-4); `git diff --check` rc=0; the tripwire's own
    duration 13s -> 2.5s. Hermeticity: `find ~/ouro/data -newermt <session start>` lists only `data/claudexor/daemon/` and `data/claudexor/profiles/` entries written by the running claudexord and a Codex session — nothing under settings, state, logs or task_results; the live repo stayed clean; the probe and mutation trees were exports and rsync copies under the lane scratch, never the worktree.
43. Rejected or not taken, with evidence. (a) Widening the prefilter by one token (both
    verdicts' fix) — the instance, not the class; removed instead (item 37). (b) Verdict 1's
    "walk every call that reaches the filesystem" AST rule for the mechanism half — "reaches
    the filesystem" has no finite definition; the token rule over the shapes both lenses probed
    is what P4 proves (item 38). (c) Keeping two predicates and merely aligning their marker
    sets — the round-3 evidence is that two copies drift (`json.dump` vs `os.replace`,
    per-function vs per-call); one predicate with two readers instead (item 36). (d) Sharing the
    scan through a cross-test-module import (`from tests.test_runtime_mode_elevation import
    ...`) — no precedent in the suite; `tests/_shared.py` is the precedent (`_shared`,
    `_typed_guard_shared`). (e) Not claimed: the trigger half remains a heuristic — a writer that
    neither calls the serializer nor names the settings path, the settings file or "settings"
    anywhere in its source is invisible by construction; the structural trigger is the
    serializer call, which any writer of the one spelling must make, and a writer that
    re-derives the JSON text under a neutral name and a neutral path is outside every
    scan-based pin. Disclosed, not fixable by scanning. (f) `atomic_write_json(` stays in the
    mechanism half's byte-exact set (it is byte-exact); a settings writer using it would fail
    the calls-the-serializer assertion instead, which is the assertion that owns "one
    serializer". (g) The round-3 table (item 32) is left as written: verdict 2's remark that it
    recorded only shapes the old markers already saw is answered by item 41, not by rewriting
    history. (h) Everything in items 10, 22 and 34 stands.

### D03 fix round 5 (operator, after the round-4 lenses)

44. The unified writer predicate had lost the ``os.replace`` commit shape and never
    carried ``replace_atomic`` / ``shutil.copy*``: an unrouted writer naming
    ``SETTINGS_PATH`` and committing through a rename or a copy passed both pins
    (lens probes P7/P8 on an export of b3ec2f1f). The shape alternation now names
    them; on the real tree the widened predicate flags the same functions (no new
    exemption). The shape half is disclosed as a finite list in the predicate
    docstring and in ARCHITECTURE, replacing the absolute «a sixth writer anywhere in
    those roots fails the tripwire». ``serialize_settings``'s docstring scopes «every
    writer» to the scanned roots (the benchmark harness under devtools/ derives its
    own JSON text for an isolated install and is outside the inventory).
45. Ledger item 42's full-suite count is the default lane plus the serial lane as the
    fixer ran them; the per-lane counts are the gate manifests of the operator's
    script run on b3ec2f1f (non-serial 0 FAILED/ERROR, serial 0 FAILED/ERROR).
## From the C6 usage-ledger compaction lane (1A, base 74a03082)

1. CPL4-C6 LANDED — the excised monetary row of the CPL-4 persistence train
   (owner batch №8 item 1A: own reviewed lane, monetary authority). Design
   note ratified BEFORE code: `docs/v7next/DESIGN_USAGE_COMPACTION.md`.
   Mechanics, one new leaf `ouroboros/usage_compaction.py` (D16):
   - **Fold scope**: terminal (`settled`/`unresolved`/`released`),
     review-attribution-free `kind="attempt"` chains only. In-flight chains
     never fold; idempotency-bearing kinds (`subscription_session`,
     `external_unmetered`, `legacy_*`) never fold — their replay dedup and
     conflict checks read the LIVE replay, so folding them would convert an
     idempotent replay into a silent double charge; review-attributed rows
     never fold — `skill_review_usage` projects historical waves per-attempt.
     Unknown future kinds are retained (fail-safe default). All four are the
     disclosed residual growth terms.
   - **Baseline block**: one stamped `usage_baseline` header (archive ref,
     source sha256/size/row-count/seq-range, epoch, counts) + one
     `usage_baseline_group` row per attribution tuple (state, model, provider,
     category, source, task/root/parent ids, ttl, cost/bound/pricing-known
     flags, cost_final) — per-group, not the sketch's single row, because
     budget enforcement is per-root and `usage_breakdown` groups per axis: a
     single global row would zero per-root accounting (a budget bypass).
   - **Decimal exactness rule (fixed)**: group sums are computed as exact
     `Decimal`s of the literals in the file and carried as exact-decimal JSON
     strings (`_number` already accepts them at every reader); retained rows
     are verified Decimal-identical across re-serialization, and a foreign
     non-round-trippable literal ABORTS the pass instead of approximating.
   - **Prove-then-swap**: the pass commits only after the candidate bytes
     re-validate (`_validate_records`) AND the production aggregation
     (`_summary`, per-root summaries + min limits, `_breakdown_bucket` on all
     axes) renders EQUAL dicts on before/after finals AND decimal money
     totals match — any inequality aborts, leaving the ledger byte-identical.
   - **seq policy**: live file keeps the validator's dense-seq integrity
     authority by starting a fresh epoch (retained rows renumbered, each
     carrying `pre_compaction_seq`; originals live forever in the archive
     segment). Monotonicity/density preserved; nothing durable references
     rows by seq (cross-refs are attempt ids). `_append_rows_locked`,
     resume fingerprints and both read caches keep their arithmetic unchanged;
     cache coherence is structural (atomic swap = new inode = refold).
   - **Crash-safety**: archive segment (exact source bytes) is written
     O_EXCL + fsync (file AND directory, best-effort on Windows) BEFORE the
     atomic ledger swap; a crash at any point leaves a valid ledger; orphan
     segments are harmless. Quarantine semantics untouched.
   - **Trigger (config SSOT, no env knob)**: `USAGE_LEDGER_COMPACT_BYTES`
     (8 MB) + `USAGE_LEDGER_COMPACT_RETRY_GROWTH_BYTES` (1 MB) in
     `config.py`; `reserve_attempt` calls the guard at the top of its locked
     section (exactly the path whose lock hold degrades with size), contained
     — a compaction defect never fails a reservation. The 20 MB
     `USAGE_LEDGER_WARN_BYTES` stays as the broken-compaction tripwire
     (warn texts updated in `agent_startup_checks`/`context_budget`).
   - **CPL-5 join surface**: `archived_attempt_ids` /
     `usage_attempt_recorded` walk the hash-chained archive (live header
     pins the newest segment's sha256; each segment's own leading header pins
     the previous one) with per-process immutable-segment caching. Contract
     recorded for the CPL-5 lane (not on this base yet): the reverse sweep's
     `orphan_seal` verdict must consult live ∪ archive, and a corrupt chain
     is its UNKNOWN/skip-pass state. `legacy-import` needs no lookup — its
     rows never leave the live file.
   - **Aggregation**: `_summary`/`_physical_call_count` became baseline-aware
     in the narrowest way (skip header; group rows: counts × folded weight,
     sums added once); weight-1 paths byte-equivalent to before.
   - Measured on this host (isolated env): 24,000-row / 11.9 MB synthetic
     ledger → 183 KB (65×), 280 groups, 1.16 s single pass under the 45 s
     monetary lock; global+per-root projections byte-equal, budget refusal
     thresholds identical.
   - Pins: `tests/test_usage_compaction.py` (16 tests — exact-money +
     whole-projection equality incl. skill-review waves; budget/root-budget
     refusal equality; in-flight survival + post-compaction lifecycle; crash
     injection between archive and swap; chained-compaction id resolution +
     tampered-segment detection; subscription/external replay dedup +
     conflict; legacy import with and without watermark; threshold/throttle
     policy; verify-abort on a foreign literal; head-only baseline
     validation; quarantine on a compacted file; archive byte-exactness).
     Key pins mutation-probed red (weight math, decimal rounding, in-flight
     folding). ABI-3 alias sweep: one per-site allowlist row for the
     internal ledger-plane `cost_usd` emission in `_build_candidate`.
     Persistence inventory: `archive/usage_ledger/*` plane row + scan pin
     123→124; ADOPTION CPL-4 row updated with the C6 hook.
2. NOT touched, per the lane scope: folding of
   subscription/external/legacy/review-attributed rows (future lane behind an
   archived-identity membership check if their residual growth ever matters),
   any archive GC (never), CPL-5 sweep implementation, warn threshold values.

## From the C6 fix-round (base `e2801c52`)

1. The C6 lane's external adversarial wave returned NEEDS FIXES with nine
   findings against the landed compaction. All nine were ACCEPTED and fixed —
   this is the monetary authority, so none was dispositioned as theoretical —
   in three single-intent commits authored by Ouroboros: `9e99eb55`,
   `0ed2dc2c`, `6b03212e`. Full disposition table:
   `docs/v7next/C6_REVIEW_PACKET.md` §6.
   - **Lock ownership (finding 1, HIGH)**: the pass ran under a lock that was
     evictable on elapsed time alone, while a pass over a multi-megabyte
     ledger legitimately outlives the 90 s window — so a second writer could
     take the lock, fsync a settled row, and have the finishing compactor
     replace the file with bytes that never contained it. Fixed twice over:
     `_named_lock` now acquires `owner_aware_stale=True` (a live owner is
     never evicted by age; this also covers the long `usage_import.lock`
     hold), the hold yields a heartbeat
     (`platform_layer.refresh_exclusive_file_lock`, descriptor-targeted so a
     stolen lock is never refreshed for the thief) that the pass beats at
     each checkpoint, AND the swap is refused unless the live bytes are still
     byte-identical to the snapshot, re-read under the same held lock
     immediately before the rename. A lost race costs one skipped pass.
   - **Archive durability (finding 2, HIGH)**: `mkdir -p` fsync'd only the
     segment's own parent and `_fsync_dir` swallowed every error, POSIX
     included — so a crash could lose the `archive/usage_ledger` directory
     entry while the swap survived. Now every created directory entry
     (segment parent, `archive/`, data root) is fsync'd before the swap, and
     a POSIX fsync failure ABORTS the pass; Windows stays a no-op by the
     platform predicate, never by a bare `except`.
   - **Provenance validation (findings 3 + 6, HIGH/MEDIUM)**: the substrate
     now validates the stamp it stores — positive epoch, bounded
     `archive_rel` (`valid_archive_rel`: relative, exactly
     `archive/usage_ledger/<name>`, no traversal/absolute/drive), 64-hex
     sha, counts that close, and block↔header agreement on `group_count` and
     summed `folded_attempt_count` — plus `pre_compaction_seq` as a checked
     claim (only under a stamp, strictly increasing, closed by the first row
     that lacks it). The archive reader bounds the resolved path to the
     archive directory, cross-checks size and row count against the naming
     header, runs each segment through `_validate_records`, and requires the
     chain's epochs to step down by one to a header-less epoch 1 — which is
     what makes re-pointing a header at an older GENUINE segment corruption
     rather than a shorter history.
   - **Warm-cache integrity (finding 4, MEDIUM)**: the per-segment cache hit
     requires the file's `(ino, dev, size, mtime_ns)` fingerprint, so a
     deleted or rewritten segment raises `UsageLedgerCorrupt` even in a
     process that already read it.
   - **Decimal precision (finding 5, MEDIUM)**: sums ran in the ambient
     28-digit context and the self-check rounded the same way, so a rounded
     total verified against itself (10²⁸ + 1 folded to 10²⁸). Money is now
     summed under `_exact_money` (`prec=60`, `Inexact` trapped → abort, never
     approximate); the pin's oracle sums in its own wider context.
   - **Typed corruption (finding 7, MEDIUM)**: an unreadable live leading row
     raised nothing and read as "never compacted"; it now raises, so the
     CPL-5 join reaches UNKNOWN/skip instead of reporting a folded attempt as
     an orphan seal.
   - **Join cost (finding 8, LOW)**: the chain union is cached by chain
     identity, so a reverse sweep of H seals costs H stat-checked walks and
     ONE union instead of H unions over the whole archived id set. The
     per-segment stat check still runs, so finding 4 is not traded away.
   - **Pin quality (finding 9, MEDIUM)**: the three pins the wave called weak
     were rebuilt — the crash pin injects at `os.replace` itself and asserts
     the segment is already durable with the exact source bytes (a
     swap-before-archive reorder now fails it); the threshold pin proves the
     lock is HELD at the call; the head-only pin contrasts one unmodified
     baseline block validating at the head against the same rows rejected
     purely for position. Every new pin in this round was verified RED against
     the exact mutation it claims to catch before being accepted.
2. NOT changed by this round, deliberately: the fold scope (design note §3),
   the per-group baseline shape (the wave independently confirmed it preserves
   per-root enforcement and all five breakdown axes, and that a single global
   row would have been a budget bypass), the trigger thresholds, the ABI-3
   allowlist row, and the disclosed residuals for
   subscription/external/legacy/review-attributed rows.
3. NEW residual, disclosed: a pass that loses the snapshot race leaves an
   orphan archive segment behind (written, never referenced). Orphan segments
   were already disclosed as harmless and the archive is append-only by
   design; the alternative — swapping anyway — is the defect being fixed.

## From the C6 fix-round 3 (base `830aa35a`)

1. A second external adversarial wave re-read the round-2 fixes and returned
   NEEDS FIXES: five of the nine findings were judged still OPEN (1, 2, 3, 4,
   6) and four confirmed closed (5, 7, 8, 9). All five were ACCEPTED and fixed
   in single-intent commits authored by Ouroboros; each fix carries a pin that
   was verified RED on this base against the exact mutation it claims to
   catch. Full disposition table: `docs/v7next/C6_REVIEW_PACKET.md` §7.
   - **Lock ownership (finding 1, HIGH)**: round 2 made the lock owner-aware
     and heartbeaten, but never proved OWNERSHIP anywhere. Stale inspection
     judged the path and then unlinked the path, so a release plus a second
     acquirer landing in that window lost their lock to the evictor; the
     release unlinked whatever occupied the name; and the POSIX heartbeat
     renewed the descriptor and answered success even after that descriptor
     had been unlinked, while `_beat` ignored the answer and nothing beat at
     all during the long build/verification span. `ouroboros/platform_layer.py`
     now compares descriptor identity against path identity in all three
     places — the eviction unlinks only the exact file it judged abandoned
     (re-checked immediately before the unlink), the release unlinks only the
     file it still holds, and `refresh_exclusive_file_lock` returns an
     ownership verdict. The pass consumes that verdict: a `False` or
     unanswerable heartbeat aborts, and beats now run inside both candidate
     row walks and between every verification stage. The compare→replace
     TOCTOU is closed structurally rather than narrowed — every writer of this
     ledger appends under the same owner-aware lock and there is NO unlocked
     fallback (a timed-out acquisition raises `UsageAccountingError` and writes
     nothing), which is now pinned rather than assumed — and after the rename
     the pass re-reads the path, so a receipt only ever describes bytes that
     landed.
   - **Retry durability (finding 2, HIGH)**: round 2 fsync'd the directory
     levels a pass CREATED. A pass that died on that fsync left the levels
     present, so the retry skipped them and could complete its swap with the
     archive it depends on still unnamed on disk. `_mkdir_fsync_chain` now
     takes the data root and fsyncs the whole chain unconditionally on every
     pass.
   - **Chain anchor and provenance range (finding 3, HIGH)**: the epoch
     step-down rule only proves the chain BELOW the live header, and
     `compaction_epoch` is as mutable as the rest of that row — so repointing
     the header at an older genuine segment while lowering its epoch produced
     a chain that verified and simply ended early. The generations such a
     forgery orphans are still on disk, so the archive now anchors the stamp:
     no segment may carry a generation newer than the live one, derived from
     each segment's own embedded header rather than from its name. The one
     legal newer case — an uncommitted orphan of the CURRENT generation, whose
     embedded leading row IS the live header — stays legal and is pinned so
     the anchor cannot over-reach into the disclosed orphan residual.
     Separately, `pre_compaction_seq` was only required to increase; it must
     now fall inside the source range the header declares, so a retained row
     cannot claim an origin the archived source never held.
   - **Segment-cache shelf life (finding 4, MEDIUM)**: the
     `(ino, dev, size, mtime_ns)` fingerprint is not proof of identity — an
     in-place same-size rewrite within the filesystem's timestamp granularity,
     or one that restores the mtime, keeps it whole. A hit now also requires an
     mtime settled for more than 2 s and an entry younger than 60 s. The
     remaining window (a rewrite that restores `mtime_ns` exactly, answered
     from a warm entry for up to a minute) is DISCLOSED as a residual with its
     cost: closing it means re-hashing every segment on every question, which
     is the quadratic cost the cache exists to remove.
   - **Archive symlink bound (finding 6, MEDIUM)**: comparing a resolved
     segment against a resolved archive directory is satisfied for free by a
     symlink AT `archive/` or `archive/usage_ledger`, because both sides
     resolve through the same link. Neither level may be a link now, the
     resolved directory must be exactly the resolved data root's archive path,
     and no segment may be a link; the reader raises `UsageLedgerCorrupt` and
     the writer aborts its pass with the ledger byte-identical.
2. Absolutes removed. `docs/ARCHITECTURE.md` claimed a long pass is "never
   robbed" of its lock and described the archive references as simply
   "bounded"; the packet repeated both. The contract is now stated with its
   limits: ownership is defended and its LOSS is survivable (a robbed pass
   cannot finish, it aborts), the archive bound is explicit about symlinks and
   the epoch anchor, and the cache window plus the orphan segments are named
   where the mechanism is described. Residuals §5.6–5.9 of the packet carry
   them: orphan segments (LOW availability/forensic clutter, no GC by design),
   the warm-cache window, "ownership defended, not guaranteed", and the
   deliberate choice to treat an unreadable segment first row as no evidence
   of a generation rather than as corruption — so a garbage file dropped into
   the archive cannot deny service to the whole history, while every segment
   an answer depends on is still fully verified by the walk.
3. `tests/test_usage_compaction.py` crossed 1000 lines and entered the
   1001-1500 size band with a recorded rationale in the same commit; the five
   lock-primitive pins live in `tests/test_lockfile_helpers.py`, where those
   primitives live, rather than in the compaction suite.

## From the C6 fix-round 4 (base d7b487ab)

1. A third external adversarial wave returned NEEDS FIXES against `d7b487ab`:
   the round-3 ownership and bound work was judged short of the contract in
   four ways, all ACCEPTED and fixed. Full disposition table:
   `docs/v7next/C6_REVIEW_PACKET.md` §8.
   - **Kernel-enforced exclusion**: the lock's exclusion still rested on the
     O_EXCL name protocol; the stale eviction re-checked the inode and then
     unlinked the PATH, so two reclaimers racing over one abandoned lock
     could both evict — the second removing the first one's freshly won lock
     and putting two writers on the monetary authority. The lock fd now HOLDS
     `fcntl.flock` (`LockFileEx` on Windows) from acquisition; eviction
     unlinks only while flock-holding the very fd it judged, with the path
     re-checked under that hold, and a release unlinks BEFORE its close,
     under the still-held flock. Pinned in `tests/test_lockfile_helpers.py`:
     two reclaimers herded into the check-to-unlink window yield at most one
     holder (red on the pre-fix shape: both returned descriptors), and a
     heartbeat after an ATOMIC replacement of the lock file — the path never
     absent — answers False (red against the utime-only mutation).
   - **Ownership adjacent to the decisions**: the pass beat through the long
     span but nothing proved the hold immediately before the snapshot
     re-checks, and nothing at all between the final re-check and the swap.
     `beat()` now runs immediately before EACH snapshot look and before the
     swap: a hold lost at the archive write aborts before the post-archive
     re-check is even asked (pinned by counting exactly one `_snapshot_intact`
     call — the "remove that beat" mutation makes it two), and a hold lost
     after the re-check aborts before the replace (red on `d7b487ab`: the
     swap ran).
   - **The recheck→replace gap**: a row appended after the pre-swap re-check
     and before the rename was erased by the swap, receipt and all.
     `usage_ledger._write_bytes_atomic_fsync` takes a `precondition`
     evaluated once the temp bytes are durable, immediately before
     `os.replace` — the last instant the replace can still be refused — and
     the compactor passes `_snapshot_intact`. Red on `d7b487ab`: the injected
     row vanished; now the pass returns None, the row survives byte-for-byte,
     money equals before-plus-that-row, no temp residue.
   - **Symlink bound at the open, not before it**: `_archive_dir_bounded` and
     `_segment_path` judged paths and the write/read then re-resolved those
     paths — a link planted in between received the segment (writer) or
     served a byte-identical foreign copy (reader; the hash cannot object,
     only refusing the traversal defends). POSIX now opens
     root→`archive/`→`usage_ledger` `O_DIRECTORY|O_NOFOLLOW` handle-to-handle
     and creates/opens segments `O_NOFOLLOW` via `dir_fd`, fingerprinting and
     reading from the open fd and fsyncing durability through the same held
     handles. Both planted-link pins red on `d7b487ab`. Windows (no
     `dir_fd`/`O_DIRECTORY`, no unlink of an open file) and kernel-lockless
     filesystems keep the round-3 identity-re-check shapes as a best effort
     chosen by the platform predicate — disclosed, never swallowed.
2. Disclosed trade recorded with the fix: a live-but-WEDGED holder can no
   longer be evicted by age on POSIX — the flock outlives the staleness clock
   until the process dies. Age-evicting a live monetary writer was the
   two-writers defect; a wedged writer is an availability incident, not a
   correctness one. `ouroboros/usage_compaction.py` entered the 1001-1500
   size band with a recorded rationale; `ouroboros/platform_layer.py` stayed
   inside the band (1498) paid for by prose compression in the same module.
3. Evidence status, stated plainly: this round was authored under a run
   policy that denied every `python`/`pytest`/`ruff` invocation and every git
   write, so NO gate ran here and the round-4 changes reached the tree
   uncommitted. The red-first claims are per-pin, against the named base or
   mutation, by construction. The integrating coordinator must run the full
   battery (targeted suites, CI-shape non-serial, serial, size_ratchet,
   `ruff check . --select F`, `scripts/v7next_adoption.py`,
   `git diff --check`) and record the result in the packet §8 before round 4
   is called green; the packet carries the same MUST-RUN notice.

## From the C6 fix-round 4 verification (base d7b487ab)

1. Round 4 shipped unexecuted (its run policy denied every interpreter
   invocation), so this verification pass ran every claim for real: each
   round-4 pin was shown RED under the exact mutation or base it names, then
   green with the fix restored. Observed, not argued:

   | pin | mutation | red observed | green |
   |---|---|---|---|
   | `test_two_racing_reclaimers_never_yield_two_holders` | `platform_layer.py` → `d7b487ab` | 2 holders (both fds returned) | pass |
   | `test_heartbeat_after_an_atomic_swap_of_the_lock_reports_false` | utime-only refresh (identity compare removed) | heartbeat True for a replaced lock | pass |
   | `test_an_append_between_the_recheck_and_the_replace_aborts_without_loss` | swap precondition removed | receipt returned; injected row erased | pass |
   | `test_a_hold_lost_at_the_archive_is_seen_before_the_snapshot_is_trusted` | post-archive `beat()` removed | 2 `_snapshot_intact` calls, not 1 | pass |
   | `test_a_hold_lost_after_the_recheck_aborts_before_the_swap` | no ownership proof between re-check and rename | the swap ran; a baseline landed | pass |
   | `test_a_hold_lost_while_the_temp_is_written_refuses_the_replace` (NEW) | round-4-as-authored shape (outer `beat()` + snapshot-only precondition) | receipt returned while robbed | pass |
   | `test_a_link_planted_after_the_writer_bound_check_cannot_receive_history` | `usage_compaction.py` → `d7b487ab` | segment crossed the link; swap completed | pass |
   | `test_a_link_planted_after_the_reader_bound_check_is_refused` | `usage_compaction.py` → `d7b487ab` | identical copy read through the link, no raise | pass |

2. One finding of the round-4 review panel (codex, FIX_FIRST, accepted by
   the coordinator) was fixed here, red-first, without changing the accepted
   architecture: the ownership proof stood immediately BEFORE
   `_swap_ledger_fsync`, but the atomic writer may spend unbounded time
   writing and fsyncing the candidate temp before its snapshot look and
   `os.replace` — a hold lost in that window let a new holder's charge
   (landing after the in-swap snapshot answer and before the rename) be
   erased by the swap. The proof now lives in the PRECONDITION of the atomic
   replace: `_swap_ledger_fsync` passes `beat` into
   `_write_bytes_atomic_fsync`, which evaluates — once the temp bytes are
   durable, immediately before the rename — ownership FIRST, then
   `_snapshot_intact`. The pin above is red on the round-4-as-authored shape
   and green with the fix; the outer pre-swap `beat()` moved inside (not
   duplicated), so the proof adjacent to the irreversible step is the one
   that licenses it.
3. Windows tier verified at the predicate level: `fcntl` is imported only
   inside `not IS_WINDOWS` branches (`file_lock_exclusive_nb` and friends;
   Windows takes `LockFileEx`), `_dir_fd_capable()` gates every dir-fd path,
   and the two new lockfile pins — whose MECHANICS are POSIX (flock-held
   eviction; replacing an open kernel-locked file) — now carry
   `skipif(IS_WINDOWS)` with the disclosed-best-effort reason. One latent
   gap disclosed rather than fixed: `_win32_lock` raises `OSError` without
   an errno, so on Windows a refused kernel lock degrades to the disclosed
   name-protocol tier instead of standing down — same tier the packet §5.10
   already discloses for that platform.
4. `ouroboros/size_ratchet_manifest.py` was hand-edited in round 4 and the
   generator check failed on it (`regenerate_size_ratchet.py --check`:
   stale — the entry's em-dashes must be the generator's `\u2014` escape
   serialization); regenerated, gate green. This is exactly the class of
   defect an unexecuted round cannot see.
5. Gate evidence for this verification (isolated env roots per invocation,
   venv python 3.10.12 / pytest 9.1.1, `git rev-parse HEAD` verified
   unchanged after every pytest run):
   - targeted `test_usage_compaction.py` (49) + `test_lockfile_helpers.py`
     (10) + `test_usage_accounting.py`: 122 passed, exit 0;
   - CI-shape non-serial battery (`-m "not serial and not integration and
     not browser and not ui_browser and not ui_browser_docker and not
     portable_detail and not skill_smoke and not size_ratchet" -n 16 --dist
     loadscope --max-worker-restart=0 --timeout=300
     --timeout-method=thread`): EXIT=0, ~13.5k outcomes, zero FAILED/ERROR;
   - `-m serial`: EXIT=0, 661 outcomes (39 skipped); `-m size_ratchet`:
     5 green, exit 0; `ruff check . --select F` clean;
     `scripts/check_domains.py` OK; `scripts/regenerate_inventories.py
     --check` OK; `git diff --check` clean.
   With this recorded, round 4 is verified; the round-4 MUST-RUN notice in
   the packet §8 is discharged by its verification block.

## From the C6 fix-round 5 (base 13af62c5)

1. A fourth external verdict (gpt-5.6-sol, read-only review of `13af62c5`)
   returned NEEDS FIXES with four open items; the round-3 split-brain class,
   the single lock over all monetary writers and the dir-fd anchoring of the
   archive writer/reader it confirmed CLOSED. All four ACCEPTED and fixed,
   each red-first. Full disposition: `docs/v7next/C6_REVIEW_PACKET.md` §9.
   - **The lock tier is a capability predicate, and a refused kernel lock
     fails closed** (HIGH): on any `OSError` from `fcntl.flock` the
     acquisition silently degraded to the pathname/inode name tier, where the
     round-3 race returns; on Windows the errno-less `LockFileEx` failure
     fell into the same degrade. `platform_layer.kernel_file_locks_enforced`
     now decides the tier once per lock directory by kernel-locking a scratch
     file there — only ENOLCK/EOPNOTSUPP/ENOSYS select the name tier — and on
     the enforced tier contention (EAGAIN/EACCES/EWOULDBLOCK) stands down and
     re-contends while every other refusal fails closed: no descriptor, our
     own file removed, a stale lock never evicted without the held flock.
     `_win32_lock` raises `OSError(0, msg, None, winerror)` so
     `ERROR_LOCK_VIOLATION` classifies as contention. The name tier makes no
     kernel call and `compact_usage_ledger_locked` refuses it with the typed
     `NAME_TIER_REFUSAL` (logged, growth-guard throttled) while appends
     continue under the name protocol — disclosed in the design note, the
     packet and ARCHITECTURE. `usage_ledger.LOCK_REL` is the lock-path SSOT.
     Commit `f5eb969f`.
   - **Ownership and snapshot re-proven before EVERY rename attempt** (HIGH):
     the precondition ran once before `utils.replace_atomic`, which retries
     `os.replace` up to ten times with pauses on a Windows sharing violation —
     a charge appended or a hold lost between attempts was erased by the
     retry that landed. `replace_atomic(src, dst, *, precondition=None)`
     asks the precondition immediately before every attempt and returns
     False without replacing when refused; the ledger writer routes its
     ownership-first, snapshot-second proof through it. POSIX unchanged (one
     syscall). Commit `8ed4f11b`.
   - **The epoch anchor scans through the handle the walk held, fail-closed**
     (MEDIUM): `_no_newer_archived_epoch` re-resolved the archive path and
     turned `OSError` into "no evidence", so a directory swapped after the
     safe chain walk hid a newer generation and admitted a forged rollback —
     the packet's §5.10 claim that path-based anchor reads "can only ever ADD
     a corruption verdict" was false and is corrected. `archived_attempt_ids`
     opens the `O_DIRECTORY|O_NOFOLLOW` handle chain once for the whole
     question; segment loads and the anchor scan open entries relative to it
     (one `_open_archive_entry` rule; path-based only without `dir_fd`); an
     entry the scan cannot list, open or read is `UsageLedgerCorrupt`, while
     a first row that reads but does not parse stays the disclosed
     torn-segment case. Commit `a3d4d51d`.
   - **Surviving mutations pinned; absolutes stated per tier** (LOW): the
     first commit-section beat (a hold lost as the commit section is entered
     must abort before the pre-archive look and write no orphan) — commit
     `4b872c22`; the hold lost between rename retries — the `[hold_lost]`
     variant of the finding-2 pin. DESIGN §8/§10/§12, ARCHITECTURE and the
     packet now state «cannot finish while robbed», «a hold lost anyway
     abandons» and «kernel-held» for the enforced tier, and say plainly that
     on the name tier the pass does not run.
2. Red observed, not argued (pin run against the exact pre-fix shape or
   mutation, then green with the fix):

   | pin | mutation / base | red observed |
   |---|---|---|
   | `test_a_kernel_refusal_that_is_not_contention_fails_closed` | `13af62c5` | a descriptor returned for an ENOLCK-refused lock |
   | `test_a_stale_lock_is_never_evicted_without_the_kernel_hold` | `13af62c5` | stale file evicted by name, descriptor returned |
   | `test_the_name_tier_is_chosen_by_the_predicate_not_by_a_refusal` | `13af62c5` | a kernel call on the name tier (`[16] == []`) |
   | `test_the_capability_probe_decides_once_and_leaves_no_residue` | `13af62c5` | no predicate (`AttributeError: _KERNEL_LOCK_TIER`) |
   | `test_the_pass_refuses_on_the_name_tier_while_appends_continue` | `13af62c5` | receipt returned on the name tier |
   | `test_a_refused_rename_re_proves_…[append]` | `13af62c5` | retried rename landed: receipt, appended row erased |
   | `test_a_refused_rename_re_proves_…[hold_lost]` | `13af62c5` | retried rename landed while robbed: receipt |
   | `test_the_epoch_anchor_scans_the_directory_the_chain_was_walked_in` | `13af62c5` | DID NOT RAISE: look-alike directory hid epoch 3 |
   | `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption` | `13af62c5` | DID NOT RAISE: dangling entry swallowed |
   | `test_a_hold_lost_before_the_first_commit_look_writes_no_orphan` | first commit `beat()` deleted | `[1] == []`: look asked, orphan written |

3. Disclosed, not fixed: (a) Windows already held `LockFileEx` on the lock
   fd; `msvcrt.locking` was not adopted (a thinner CRT wrapper over the same
   kernel lock with an EACCES/EDEADLOCK ambiguity). The Windows-only pin
   `test_windows_lockfileex_contention_reads_as_busy` and the
   `OSError(0, msg, None, winerror)` mapping were NOT executed on this host;
   they follow the documented CPython constructor contract and are owed to
   the 3-OS CI matrix. (b) The tier is decided per process per directory: a
   lockd dying mid-run can leave one process on each tier until restart —
   the name-tier process still never compacts. (c) A directory an attacker
   swaps BEFORE the question starts is out of scope for the anchor (it is
   the same power as deleting the newer segments); round 5 removes only the
   window between the walk and the anchor and the swallowed `OSError`.
   (d) `ouroboros/usage_compaction.py` grew 1094→1124 inside the band; its
   band rationale could not be extended because the ratchet's own transition
   rule makes a surviving rationale immutable between adjacent manifests
   (`validate_manifest_transition`), so the growth is recorded here.
   `platform_layer.py` stays at 1498 by prose compression and by the pid
   lock / port sweep reusing the module's own primitives, not by a helper.
   `tests/test_usage_compaction.py` sits at 1492 (the raced-charge literal
   folded into `_raced_row`).
4. Gate evidence (this host, isolated env roots per invocation, venv python
   3.10.12; `git rev-parse HEAD` verified after every pytest run):
   - targeted `test_usage_compaction.py` (55) + `test_lockfile_helpers.py`
     (15, one Windows-only skip) + the four other `test_usage_*.py` suites:
     exit 0 at `4b872c22`;
   - neighbouring lock consumers (`test_bughunt_fixes.py`,
     `test_task_status_flow.py`, `test_evolution_commit_receipt.py`,
     `test_skill_lifecycle_queue.py`, `test_osworld_cu_bridge.py` claim/lock
     cases, `test_atomic_write_v639.py`): exit 0;
   - `ruff check . --select F` clean; `scripts/check_domains.py` OK;
     `scripts/regenerate_size_ratchet.py --check` exit 0;
     `scripts/regenerate_inventories.py --check` exit 0; `git diff --check`
     clean;
   - CI-shape non-serial battery at `4b872c22` (`-m "not serial and not
     integration and not browser and not ui_browser and not
     ui_browser_docker and not portable_detail and not skill_smoke and not
     size_ratchet" -n 16 --dist loadscope --max-worker-restart=0
     --timeout=300 --timeout-method=thread`): EXIT=0, 13498 outcomes
     (13494 passed, 4 skipped, zero FAILED/ERROR — counted from the
     progress markers, the `-q` summary line was not emitted in this run);
   - `-m serial` pass at `4b872c22` (`--timeout=600 --timeout-method=thread`):
     EXIT=0, 661 outcomes (622 passed, 39 skipped, zero FAILED/ERROR).
   With this recorded, round 5 is executed and verified on this host; the
   Windows-only pin remains owed to the 3-OS CI matrix (item 3a).

## From the C6 fix-round 5.2 (base 2dd3e017)

1. An independent adversarial pass over the round-5 tree (`2dd3e017`; PoCs
   executed on scratch copies) left five HIGH/MEDIUM findings open and six
   LOW. All eleven ACCEPTED: nine fixed in code, each red-first; two closed
   by the disclosure the finding asked for. Full disposition:
   `docs/v7next/C6_REVIEW_PACKET.md` §9, "Round 5.2".
   - **A creator evicted while still lock-less never returns a descriptor**
     (HIGH): between its O_EXCL create and its kernel lock the creator's file
     was EMPTY and held nothing an evictor had to respect; stalled there past
     `stale_sec` (SIGSTOP, a suspend, a debugger, NFS clock skew) it was
     evicted and its flock then SUCCEEDED on the unlinked inode — two
     descriptors believed to be one monetary lock (PoC `HOLDERS: 2`), and the
     ordinary append transaction never heartbeats. The owner pid is now
     written BEFORE the kernel lock and a won lock is returned only while the
     path still names the descriptor — both tiers, every caller of the
     primitive (the age-only non-monetary locks included). Commit `7923e624`.
   - **LockFileEx refusals classify by their Win32 error** (LOW): winerror
     5/32/33 all map onto EACCES, which sat in the busy set, so on Windows a
     genuine access-denied re-contended until the 45 s timeout. Only
     ERROR_LOCK_VIOLATION reads as EAGAIN; the busy set is {EAGAIN,
     EWOULDBLOCK} on both platforms. Commit `f2b118a4` (the errno arithmetic
     is pinned and executed here; the LockFileEx call itself stays owed to
     the 3-OS matrix).
   - **Ownership proven on both sides of the in-swap look** (MEDIUM): the
     last proof of ownership preceded a full-file read (≈1.8 ms on an 8 MB
     ledger) and an fsync'd append (≈0.2 ms) by an out-of-protocol holder
     landed after the look answered True and before `os.replace` (PROBE-1:
     receipt returned, the row erased); the documented ownership-FIRST order
     was load-bearing for no pin. `owned_and_intact` now beats, looks, and
     beats again, so the only interval between the last proof and the rename
     is the syscall. Commits `ff6bb399` (one snapshot-look recorder for the
     hold/append pins, no pin weakened) and `79a1b9fb`.
   - **The suite fold** (repair, not a finding): `79a1b9fb` left
     `tests/test_usage_compaction.py` at 1512 lines — past the 1500-line band
     ceiling, the manifest stale (`regenerate_size_ratchet.py --check` exit 1
     at that tree) — the session limit hit before the check ran. There is no
     committed-history replay on this line (`ouroboros/review.py`: the local
     surface warns), so the linear repair stands: three verbatim scaffolding
     duplicates folded in place (the raced-charge-survived assertion block,
     five copies → `_charge_survived`; the retry-durability pin re-running
     the first-pass proof instead of carrying a second fsync recorder; the
     single-caller `_lock_is_held` inlined) and PEP 8 spacing, 1512 → 1461,
     no new abstraction, every folded pin still red under the
     swap-precondition-removed mutation. Commit `6ad110e9`; the round's pins
     bring the suite to 1499 and `24b410d8` removes three no-op cache clears
     and hoists an import for headroom: 1492 at the tip.
   - **Non-regular archive entries are classified, a FIFO cannot hang the
     question, a named segment that is not a regular file is typed**
     (MEDIUM + two LOW): round 5's fail-closed rule made a stray `backup/`
     directory typed-corrupt for every history question (`os.read` → EISDIR;
     `13af62c5` answered) — an availability regression with no correctness
     gain; a FIFO blocked the open indefinitely (pre-existing: the path-based
     open blocked the same way); a directory standing where the header names
     a segment escaped as a bare `IsADirectoryError`. `_open_archive_entry`
     opens `O_NONBLOCK` through the held dir-fd, `_no_newer_archived_epoch`
     fstat-classifies (non-regular → no segment, skipped; cannot
     list/open/read → corruption), `_load_segment` raises typed on a
     non-regular named segment or any `OSError` of its fstat/read. Commit
     `503a0dd6`.
   - **The anchor-swap pin covers the open-through-fd half** (MEDIUM): under
     "list via the held fd, OPEN BY PATH" the round-5 pin stayed green for the
     wrong reason (`FileNotFoundError` → "could not complete") while a
     look-alike carrying the epoch-3 NAME with the forged live header as its
     leading row was admitted by the orphan exemption. The look-alike now
     carries that segment; the pin requires `generation newer`. Commit
     `95a53ad2`.
   - **The name-tier refusal is a typed, durable event; the 20 MB tripwire
     names the tier** (MEDIUM): the refusal was a throttled log line folded
     into the same `False` as "nothing foldable", and the round-5 claim that
     "the 20 MB tripwire names the case" was false (the text named only a
     broken compaction or a large residue). One `usage_ledger_compaction_refused`
     row per process per data root in `logs/events.jsonl` (existing
     `append_jsonl`, contained); the tripwire text and the threshold comment
     name the third cause and the event; no return-type change. Commit
     `208fe5ac`.
   - **Doc absolutes bounded; residuals named** (LOW ×2, no code): «cannot
     finish while robbed» and «a hold lost anywhere abandons» become the
     bounded contract (a concurrent holder only after an out-of-protocol
     removal of the lock file; caught at the next proof; the irreducible
     residual is the one syscall between the last proof and the rename); the
     dir-fd is held from the handle open AFTER the live-header read, not
     "for the whole question"; the orphan exemption admits a verbatim restore
     of the previous generation; the mixed-tier residual names by-name
     eviction of the enforced-tier process's appends; packet §5.9 rewritten
     (a garbage REGULAR file cannot deny service; non-regular entries are
     skipped; cannot list/open/read = corruption); packet §8 row 1 and the
     round-4 residual paragraph carry the correction marker. DESIGN §8/§10/
     §12, ARCHITECTURE, PERSISTENCE, the packet and this section: the docs
     commit that lands this section.
2. Red observed, not argued (each pin run on a scratch copy against the exact
   pre-fix shape or mutation it names, then green with the fix):

   | pin | mutation / base | red observed |
   |---|---|---|
   | `test_a_creator_evicted_while_lock_less_never_returns_a_descriptor` | `platform_layer.py` @ `2dd3e017` | `assert 2 == 1`: two descriptors believed to be one lock |
   | `test_lockfileex_refusals_classify_by_the_win32_error` | same | `assert 13 not in frozenset({11, 13})` |
   | `test_a_hold_lost_after_the_last_snapshot_look_refuses_the_rename` | `owned_and_intact` in the `2dd3e017` order (beat → look → replace) | a receipt returned while robbed (`… is None` failed) |
   | `test_a_hold_lost_after_the_recheck_aborts_before_the_swap` (look-count clause) | snapshot-first / beat-second (mutation M3) | `assert 3 == 2` |
   | `test_the_epoch_anchor_scans_the_directory_the_chain_was_walked_in` | anchor opens entries by path (listing through the held fd kept) | `DID NOT RAISE UsageLedgerCorrupt`; the previous pin shape passed under the same mutation |
   | `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption` (subdirectory) | `79a1b9fb` | `anchor scan could not complete: [Errno 21] Is a directory` |
   | same, FIFO half alone | `79a1b9fb` | `TimeoutError: FIFO open blocked` — the open hung until the 5 s alarm |
   | `test_warm_segment_cache_revalidates_the_file_it_cached` (directory shape) | `79a1b9fb` | bare `IsADirectoryError: [Errno 21] Is a directory` from `os.read` |
   | `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (event + tripwire clauses) | `79a1b9fb` | no `events.jsonl` row at all (`FileNotFoundError`); the tripwire note named no tier |
   | the five folded raced-charge pins (control for the fold) | swap precondition removed, on `6ad110e9`'s suite | all five red |

3. Disclosed, not fixed: (a) the LockFileEx call and the Windows-only
   contention pin stay unexecuted on this host (owed to the 3-OS matrix); the
   path-based Windows anchor scan keeps round 5's fail-closed rule without
   the `S_ISREG`/`O_NONBLOCK` classification — a directory in the archive is
   corruption there — and the FIFO/dangling-link pin is POSIX-only
   (`skipif(IS_WINDOWS)`). (b) The bounded ownership residual: a charge landed
   by an out-of-protocol holder inside the one syscall between the last proof
   and the rename is erased, then surfaced as `UsageLedgerCorrupt` by the
   post-swap re-read or quarantined seq-misnumbered at the next read —
   reachable only after a hand removal of the lock file or from a name-tier
   process of a mixed-tier install. (c) In that mixed mode the name-tier
   process also evicts by name with no kernel hold, so the two-writer class
   returns for the enforced-tier process's heartbeat-less appends, not only
   for compaction. (d) The orphan exemption admits a verbatim restore of the
   previous generation (the same power as truncating the live tail; it hides
   no id from the join). (e) A UNIX socket in the archive directory cannot be
   opened at all (ENXIO) and therefore reads as corruption, not as a skipped
   special file. (f) `ouroboros/usage_compaction.py` grew 1124 → 1158 inside
   its band (rationale immutable between adjacent manifests; growth recorded
   here); `platform_layer.py` stays at 1499, `agent_startup_checks.py` at
   1490.
4. Gate evidence (this host, isolated env roots per invocation, venv python
   3.10.12; `git rev-parse HEAD` verified unmoved after every pytest; the
   battery ran on a scratch clone checked out at the code tip `24b410d8` with
   these docs copied in, so the git-recovery tests could not move the lane's
   HEAD):
   - targeted suites at `24b410d8` (`test_usage_compaction.py` 56,
     `test_lockfile_helpers.py` 17 with one Windows-only skip, the other four
     `test_usage_*.py` suites, `test_context.py`, `test_startup_hygiene.py`,
     `test_persistence_inventory.py`, `test_atomic_write_v639.py`, and the
     lock consumers `test_bughunt_fixes.py`, `test_task_status_flow.py`,
     `test_evolution_commit_receipt.py`, `test_skill_lifecycle_queue.py`):
     348 passed, 1 skipped, exit 0;
   - `ruff check . --select F` exit 0; `scripts/check_domains.py` exit 0;
     `scripts/regenerate_size_ratchet.py --check` exit 0;
     `scripts/regenerate_inventories.py --check` exit 0; `git diff --check`
     exit 0; the CI `size_ratchet` pytest lane 5 passed, exit 0;
   - CI-shape non-serial battery (round 5's marker expression and flags,
     `-n 16`): EXIT=0, 13501 outcomes (13497 passed, 4 skipped, 28 warnings, 113 s) on the scratch clone at `24b410d8` with these docs copied in;
   - `-m serial` pass (`--timeout=600 --timeout-method=thread`): three runs at `24b410d8` and one control, all on scratch clones: run 1 — 1 failed / 621 passed / 19 skipped, the failure `tests/test_workspace_executor_services.py::test_executor_services_participate_in_task_and_global_cleanup` (a service-start/cleanup timing test that iterates the executor's in-memory registry; outside this lane's files; 3/3 green when rerun in isolation on the same tree); run 2 — 125 setup ERRORS / 497 passed, the host's per-user inotify-instance ceiling (128/128 held, 124 of them by containerd shims of this user's rootless TB/cybergym docker daemons — the same exhaustion that made `tail -f` fall back to polling during the battery), disclosed as environment, not tree; control on the pre-round base `2dd3e017` in the next window — 622 passed / 19 skipped, EXIT=0 (320 s); run 3 at `24b410d8` — 622 passed / 19 skipped, EXIT=0 (288 s), the same outcome set as the control;
   - hermeticity of the battery against the live data root: `find ~/ouro/data -newermt <battery start>` names only `claudexor/daemon/journal/global-…/journal.bin` — the live Claudexor daemons' own journal, held open by six daemon processes with 3–13 day uptimes — no pytest write; every pytest data root was an isolated mktemp under /tmp/claude-1006, and the lane's HEAD (`24b410d8`) and each clone's HEAD were unmoved after every run.
   With this recorded, round 5.2 is executed and verified on this host; the
   Windows LockFileEx mechanics remain owed to the 3-OS CI matrix (item 3a).

## From the C6 fix-round 5.3 (base 5e4829e3)

1. Adversarial lenses over the round-5.2 tree (`5e4829e3`; PoCs executed
   against this lane's own code) left six HIGH/MEDIUM findings open and
   seven LOW. All thirteen ACCEPTED: twelve fixed in code, each red-first;
   one closed by the disclosure the finding itself offered. The round ran in
   two halves — the first fix agent hit its session limit after `cbfd23ce`
   with the docs staged and this section unwritten; the resumed half
   re-observed every red below in a scratch copy of the worktree before
   changing anything, kept all ten commits, and closed two residues of the
   round's own fixes (3b, 4b). Full disposition:
   `docs/v7next/C6_REVIEW_PACKET.md` §9, "Round 5.3".
   - **A lock whose own identity cannot be read is never a hold** (HIGH):
     `_lock_identity` answers `()` for a descriptor it cannot `fstat`
     (ESTALE/EIO on the network filesystems the enforced tier exists for)
     and the acquisition compared the two identities raw, so with the path
     momentarily absent — a reclaimer's own unlink→re-create window —
     `() == ()` returned a descriptor for an unlinked inode as the monetary
     lock (PoC: two holders; the append transaction never heartbeats). The
     won lock is returned only when its own identity READS and matches; an
     unreadable one fails closed and removes the file we stamped with our
     LIVE pid when its bytes are still exactly ours — left behind, no
     owner-aware reclaimer could ever evict it. Commit `7e6b935e`, after
     `7d134fd8` (both kill-tree sweeps on the module's own `force_kill_pid`,
     no behaviour change: the bytes that keep `platform_layer.py` at 1500).
   - **The design note names the exact refusal sets** (MEDIUM): the ratified
     note still called EACCES a contention code — the negation of the code,
     of round 5.2's own pin and of packet §5.10 — and the unsupported set
     was spelled three-of-four in three places. DESIGN §8 now states both
     sets exactly (`EAGAIN`/`EWOULDBLOCK`; `ENOLCK`/`EOPNOTSUPP`/`ENOTSUP`/
     `ENOSYS`) with the Win32 answers mapped onto them, and
     `test_the_design_note_names_the_exact_kernel_refusal_sets` compares the
     note's spelled sets with the code's by number. SUPERSEDES the round-5
     entry above where it says "only ENOLCK/EOPNOTSUPP/ENOSYS" and
     "contention (EAGAIN/EACCES/EWOULDBLOCK)"; the packet's round-5 row
     carries the correction marker. Docs commit (this section).
   - **The compaction pass requires the lock's heartbeat** (MEDIUM):
     `heartbeat` defaulted to `None` and `_beat` returned at once on it, so
     a pass entered without one swapped the authority with no ownership
     proof at all, and deleting the single production wire (MUT-U) left the
     whole battery green. Both entry points take `heartbeat` as a required
     keyword, `_beat` has no `None` case, and the reserve-path pin asserts
     the callable it is handed. Commit `c71a36ea`. Resumed half (3b): a
     caller passing `heartbeat=None` is refused too — `_beat(None)` fails at
     the call and that failure is the existing abort — pinned beside the
     False and the raising heartbeats. Commit `48f7b115`.
   - **The orphan exemption is a prefix proof, and the anchor runs without
     a stamp** (HIGH + MEDIUM): the exemption decided on ONE row, so a
     ledger restored from a backup taken just after a compaction — its
     leading row equal to the newest segment's — was admitted as an orphan
     while being a strict SUBSET of that segment, and the attempts the pass
     folded (nowhere else on disk) were reported absent (PoC: 4 of 8 hidden,
     no corruption raised); with the stamp itself gone the anchor never ran
     and the whole archive was invisible. An orphan is the pre-swap COPY of
     the live file, which only grows behind it, so its bytes are still a
     PREFIX of it — that is the test — and the anchor runs on a stamp-less
     file with the floor at epoch zero (a root with no archive directory and
     no stamp still answers empty at once). Commit `e08a0392`. Resumed half
     (4b): the proof re-opened the entry by name after classifying it, so an
     entry swapped between the two opens read as zero bytes and passed; one
     open per entry now, the comparison from the descriptor that was
     classified. Commit `72d17f51`.
   - **The archive root open is typed corruption** (MEDIUM): the ROOT's
     `O_DIRECTORY` open sat outside `_archive_dir_fds`' `try`, so an
     unreadable data root (permissions, EMFILE) left a bare `OSError` on the
     join surface — the class round 5.2 closed one function away. Commit
     `82250a45`.
   - **A name-tier refusal event that did not land is not "already told"**
     (LOW): the root was marked before the append, so one transient failure
     downgraded the durable typed event to a log line for the process's
     life; the mark now follows the row that landed and both per-root maps
     key on the resolved root. Commit `023b2e84`.
   - **Non-regular entries are skipped on the path shape too** (LOW):
     without a dir-fd the entry was opened first — a directory refuses that
     on Windows and a writer-less FIFO blocks on it; the classification now
     happens BEFORE the open there, and the path-based open carries
     `O_NONBLOCK` where the platform has one. Commit `232500f4`.
   - **The swap's crash durability and the byte half of its snapshot proof
     are pinned** (LOW ×2): deleting the candidate temp's fsync (MUT-H), the
     ledger directory's fsync after the rename (MUT-L) or the byte compare
     of `_snapshot_intact` (MUT-E) had left the battery green. Commit
     `c5fa1ac7`.
   - **The name tier is reachable on Windows** (LOW): no LockFileEx refusal
     could select it (CPython lands ERROR_INVALID_FUNCTION and
     ERROR_NOT_SUPPORTED on EINVAL), so a lock-less Windows volume failed
     every monetary append closed instead of degrading as disclosed.
     `_win32_lock_error` classifies by one table — 33 busy, 1/50
     unsupported, anything else winerror-derived and fail-closed. Live
     evidence for exactly those two codes: LockFileEx on `\\wsl$` answers
     ERROR_INVALID_FUNCTION (microsoft/WSL#5762) and on a Samba share
     ERROR_NOT_SUPPORTED (error 50, samba list thread "FileLockEx Problem").
     Commit `f7b8a578`.
   - **The chain-union cache is bounded; DESIGN §10 states the anchor's
     per-question cost** (LOW): the cache is keyed by the chain identity,
     which changes at every compaction, and was never evicted. Commit
     `cbfd23ce`.
   - **Recycled pid: disclosed, not changed** (LOW): a lock whose owner
     died and whose pid was reused reads as alive (POSIX `kill(0)` answers
     EPERM for another user's process on this shared host) and is never
     reclaimed by age; the alternative — the probe flock on any aged file —
     would also evict a LIVE name-tier holder of a mixed-tier install.
     DESIGN §8, the ARCHITECTURE row and the PERSISTENCE row state the
     wedge and the hand repair.
2. Red observed, not argued (each pin run against the exact pre-fix shape
   or mutation it names; every row re-observed by the resumed half in a
   scratch copy, then green together as one control):

   | pin | mutation / base | red observed |
   |---|---|---|
   | `test_a_lock_whose_identity_cannot_be_read_is_never_a_hold` | `platform_layer.py` @ `7e6b935e^` | `assert 14 is None`: a descriptor for an unlinked inode returned |
   | same, second half | the truthiness guard kept, the stamp cleanup removed | `a live pid was stamped on a lock nobody may reclaim` |
   | `test_lockfileex_refusals_classify_by_the_win32_error` (unsupported clause) | `platform_layer.py` @ `f7b8a578^` | `assert (0 in frozenset({37, 38, 95}))` for winerror 1 |
   | `test_reserve_path_compacts_only_past_config_threshold` | `usage_compaction.py` @ `c71a36ea^` + MUT-U | `the pass was entered without the lock's heartbeat`; at the tip the same mutation is a `TypeError` at the call |
   | `test_a_lost_lock_aborts_the_pass_instead_of_swapping` (`None` clause) | `usage_compaction.py` @ `c71a36ea^` | a receipt returned: `None` skipped every proof |
   | `test_a_restored_previous_generation_…[stamped]` / `[unstamped]` | `usage_compaction.py` @ `e08a0392^` | `DID NOT RAISE UsageLedgerCorrupt`, both |
   | same, second-open clause | `cbfd23ce` (the two-open anchor) | `DID NOT RAISE UsageLedgerCorrupt`, both |
   | `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption[True]` (chmod half) | `usage_compaction.py` @ `82250a45^` | bare `PermissionError` from the root open |
   | same, `[False]` | `usage_compaction.py` @ `232500f4^` | `TimeoutError: FIFO open blocked` after the 5 s alarm |
   | `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (event clauses) | `usage_compaction.py` @ `023b2e84^` | `FileNotFoundError: …/logs/events.jsonl` — one failed append suppressed the event for good |
   | `test_the_swap_fsyncs_the_candidate_before_the_rename_and_its_directory_after` | MUT-H / MUT-L | the candidate's inode absent from the fsyncs before the rename / the directory absent after it |
   | `test_a_same_size_rewrite_between_the_recheck_and_the_replace_also_refuses` | MUT-E | a receipt returned over the rewritten row |
   | `test_archived_id_union_is_built_once_per_chain` (bound clause) | `usage_compaction.py` @ `cbfd23ce^` | `AttributeError: … '_CHAIN_UNION_CACHE_MAX'` |
   | `test_the_design_note_names_the_exact_kernel_refusal_sets` | the note as committed at `5e4829e3` | `AssertionError: []` — no set spelled at all |

   The three PoCs of the lenses (a stamped restore, an unstamped restore,
   the unreadable root) all reach typed `UsageLedgerCorrupt` on the tip.
3. Disclosed, not fixed: (a) the LockFileEx call and the Windows-only
   contention pin stay unexecuted on this host (owed to the 3-OS matrix);
   the two mapped codes rest on the cited live reports, not on a run here.
   (b) Identity residuals of finding 1: a descriptor that READS while the
   path's stat does not (ESTALE on the directory) still takes the
   re-contend branch and leaves its stamped file behind; a release whose
   own descriptor identity cannot be read unlinks nothing (it cannot prove
   the file is ours), which wedges that lock for the life of the process;
   and the byte test identifies our file by the stamp, so a caller-supplied
   `metadata` without a pid (`git_plumbing`'s `locked_at=`, an age-only
   non-monetary lock) could match another creator's stamp only if it wrote
   the identical string in the same window. (c) The recycled-pid wedge
   (item 1, last bullet). (d) The epoch anchor lists the archive and reads
   one leading row per unwalked entry on every question, ahead of the
   union cache (DESIGN §10 states it). (e) A UNIX socket in the archive
   directory reads as corruption (ENXIO), carried from round 5.2. (f) Size:
   `tests/test_usage_compaction.py` LEFT the 1001-1500 band at `e08a0392`
   and sits at 1597 in the ungated 1501-1600 zone, three lines under the
   hard cap; the manifest is regenerated with it and `--check` replays green
   at every commit of the round; the next pin on this surface needs an owner
   decision (packet §9 "Round 5.3"). `ouroboros/usage_compaction.py`
   1158 → 1197 inside its band (rationale immutable between adjacent
   manifests); `platform_layer.py` at exactly 1500.
4. Gate evidence (this host, isolated env roots per invocation, venv python
   3.10.12; `git rev-parse HEAD` verified unmoved after every pytest run):
   - targeted suites at `72d17f51`: `tests/test_usage_compaction.py` 61
     passed; `tests/test_lockfile_helpers.py` 18 passed, 1 skipped
     (Windows-only); the five `tests/test_usage_*.py` suites 152 passed —
     each as its own command, exit 0;
   - `ruff check . --select F` exit 0; `scripts/check_domains.py` exit 0;
     `scripts/regenerate_size_ratchet.py --check` exit 0 (and replayed exit
     0 at each of the twelve commits `5e4829e3..72d17f51` in a scratch
     clone); `scripts/regenerate_inventories.py --check` exit 0;
     `git diff --check` exit 0;
   - CI-shape non-serial battery (round 5's marker expression and flags,
     `-n 16`) on a scratch clone at `72d17f51`: EXIT=0, 13507 outcomes (13503 passed, 4 skipped, zero FAILED/ERROR — counted from the progress markers: the doubled `-q` suppressed the summary line, as in round 5.2); no write under the live `~/ouro/data`
     during the run (`find -newermt` of its start time answered nothing);
   - `-m serial` pass (`--timeout=600 --timeout-method=thread`) on a scratch
     clone at `72d17f51`: EXIT=0, 661 outcomes (622 passed, 39 skipped, zero FAILED/ERROR — counted from the progress markers: the doubled `-q` suppressed the summary line, as in round 5.2); the only paths under the live `~/ouro/data`
     newer than its start (26) are the live Claudexor daemon's journal and the
     `codex-ouro-reviewer` profile's own files under `data/claudexor/` —
     sqlite side files, plugin cache, a codex `tmp/arg0` sandbox helper: a
     live delegated run in that profile, not a pytest write (every pytest
     data root was an isolated mktemp); the lane's HEAD unmoved after both.
   With this recorded, round 5.3 is executed and verified on this host; the
   Windows LockFileEx mechanics remain owed to the 3-OS CI matrix (item 3a).

## From the C6 micro-round 5.4 (owner batch №12 A, base 096437c2)

1. Owner-bounded micro-round over the round-5.3 candidate `096437c2`: the
   residual list left by the Fable lenses and the independent gpt-5.6-sol
   read-only review — eight items, no new exploration. All eight disposed:
   seven changed in code with red-first pins, one (R8 c–f) closed by the
   disclosures it asked for. Full disposition and the red-first table:
   `docs/v7next/C6_REVIEW_PACKET.md` §9, "Round 5.4".
   - **ENOLCK fails closed; the tier is decided once under a lock** (R1,
     HIGH): "no locks available" is a missing lock daemon OR an exhausted
     kernel lock table, not a capability answer, yet it selected the name
     tier. The unsupported set is exactly `EOPNOTSUPP`/`ENOTSUP`/`ENOSYS`
     (winerror 1 → ENOSYS, 50 → EOPNOTSUPP); ENOLCK keeps the enforced tier
     and every live acquisition the kernel refuses with it fails closed, so
     a lockd-less NFS refuses every monetary write typed instead of running
     the name protocol; `_KERNEL_LOCK_TIER_LOCK` makes the probe
     single-flight. SUPERSEDES the round-5 and round-5.3 entries above where
     they spell the unsupported set with ENOLCK. Commit `7ce7e83d`.
   - **A pid that refuses our signal is alive** (R7, MEDIUM): `pid_is_alive`
     folded EPERM into "dead", so another user's recycled pid was reclaimed
     through the age path and only a same-uid recycle wedged — the OPPOSITE
     of the mechanism the round-5.3 L2 bullet above, DESIGN §8 and
     PERSISTENCE stated. EPERM reads alive (the process exists); only ESRCH
     is dead; `pid_provably_gone` is the exact negation and became one line.
     The real residual: any live impostor wedges the lock from the 90 s
     staleness window (`usage_ledger._locked`, `stale_sec=90.0`) until it
     exits. SUPERSEDES the L2 bullet's mechanism. Commit `01c89685`.
   - **The name-tier refusal is "told" only once its row landed** (R2,
     MEDIUM): `append_jsonl` reports exhausted retries as `False`; the mark
     now follows a `True` answer only. Commit `12558046`.
   - **The history reader types every path inspection, the stamp-less
     question included** (R3 + R4, MEDIUM): `pathlib` re-raises every
     `OSError` but ENOENT/ENOTDIR/EBADF/ELOOP; the symlink bounds on both
     archive levels and on the named segment, and the stamp-less fast path
     (`is_dir()`: a silent `frozenset()` for a regular file where the
     directory belongs, a bare `OSError` for an uninspectable one), are
     `UsageLedgerCorrupt` now; before any compaction the question ends early
     only on an exact ENOENT. Commit `d99ff6a9`.
   - **A charge erased inside the swap's one syscall is quarantined and
     surfaced, never receipted** (R5, MEDIUM): the round-5.2 §3(b) line above
     and the round-5.3 §3(b) line — "erased, then surfaced as
     `UsageLedgerCorrupt` by the post-swap re-read or quarantined
     seq-misnumbered at the next read" — were FALSE: neither could see it,
     the loss was silent and a success receipt was returned. On POSIX the
     swap holds the old inode open across the rename and reads back what
     landed beyond the proven snapshot: quarantined (`raw_base64`,
     `integrity_degraded`), typed raise, no receipt, never re-appended;
     Windows stays a disclosed silent loss. SUPERSEDES both lines. Commit
     `ea4d4337`.
   - **The reserve path is pinned to the held lock's OWN heartbeat** (R6,
     MEDIUM): `assert callable(...)` let a constant-True stub survive the
     battery; the pin now requires the aged lock renewed. Commit `9306f962`.
   - **`_build_candidate`'s beat is required; the heartbeat's blind spots
     are pinned** (R8a, R8b, LOW). Commit `02338c9b`.
   - **Disclosed** (R8c–f): the transient `generation newer` UNKNOWN when a
     compaction commits mid-question; a ledger reset beside a surviving
     archive as a permanent verdict (PERSISTENCE both rows); "is skipped"
     qualified to entries that OPEN (a UNIX socket is corruption); the
     POSIX-only release-under-flock and the contention-branch orphan. Docs
     commit (this section).
   Two no-behaviour commits paid for the pins inside the 1600 hard cap
   without a neighbour suite: `bd9e99a4` (a `compacted` fixture folding
   twenty-one seed-then-compact preambles, −39) and `b9c43911`
   (argument-list/data-literal reflows, −28); every claim kept.
2. Red observed, not argued (each pin run against the exact pre-fix shape or
   mutation it names, in this worktree; pin red, fix applied or mutation
   reverted, pin green):

   | pin | mutation / base | red observed |
   |---|---|---|
   | `test_a_pid_that_refuses_our_signal_is_alive_and_its_lock_is_not_reclaimed` | `platform_layer.py` @ `bd9e99a4` | `assert (False is True)`; the aged lock evicted, fd 3 returned |
   | `test_the_capability_probe_decides_once_and_leaves_no_residue` (ENOLCK clause) | `platform_layer.py` @ `01c89685` | `AssertionError: 37` — errno 37 selected the name tier |
   | `test_enolck_keeps_the_enforced_tier_and_the_acquisition_fails_closed` | same | `enforced = False`, fd 3 returned on the name tier |
   | `test_two_threads_racing_the_first_probe_run_one_probe_and_read_one_tier` | same | `2 == 1`: two probes |
   | `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (False append) | `usage_compaction.py` @ `7ce7e83d` | `FileNotFoundError: …/logs/events.jsonl` |
   | `test_a_stamp_less_ledger_still_inspects_its_archive_fail_closed` | `usage_compaction.py` @ `b9c43911` | `DID NOT RAISE` (answered `frozenset()`); bare `PermissionError` |
   | `test_a_path_inspection_the_reader_cannot_make_is_typed_corruption` | same | bare `PermissionError` from the segment's lstat; bare `PermissionError` from `is_symlink` |
   | `test_a_swap_that_did_not_land_is_a_typed_failure_not_a_receipt[erased]` | `usage_compaction.py` @ `d99ff6a9` | `DID NOT RAISE`: receipt returned, the charge gone, no quarantine, `integrity_degraded` False |
   | `test_reserve_path_compacts_only_past_config_threshold` (heartbeat clause) | M9 (`heartbeat=lambda: True`) | `a stub, not the held lock's heartbeat`; 84 other items green under M9 |
   | `test_a_lock_whose_identity_cannot_be_read_is_never_a_hold` (refresh clauses) | M15 / M16 | `assert True is False` under each |
   | `test_the_long_build_and_verification_section_beats_the_lock` (control) | M10 (`beat` dropped at the call) | `TypeError … missing 1 required positional argument: 'beat'` → `assert None is not None` |
3. Disclosed, not fixed: (a) a lockd-less NFS `state/` now fails every
   monetary write closed (the owner's decision; the repair is a filesystem
   that locks); (b) no `events.jsonl` row is fsync'd — the per-process mark
   dies with the crash that could lose the row, a delayed writeback error
   with the process alive is the residual; (c) R5 detects by size (a
   same-size in-place rewrite inside the syscall is not a landed charge) and
   Windows stays silent; (d) the recycled-pid wedge now covers another-uid
   recycles too, the flock alternative would evict a live mixed-tier holder;
   (e) R8c–f as disclosures; no socket pin (the 108-byte `AF_UNIX` cap, a
   `chdir` hazard); (f) size: `platform_layer.py` 1500 → 1497 and
   `usage_compaction.py` 1197 → 1245 inside their bands (the band rationale
   is immutable between adjacent manifests by the ratchet's own rule, so the
   owner's "extend it in the same commit" is recorded here instead),
   `tests/test_usage_compaction.py` 1597 → 1597; (g) Windows unexecuted here;
   (h) sol's tier-attestation-on-the-hold suggestion not taken (the module
   lock closes the intra-process disagreement; cross-process mixed tiers
   remain the round-5.2 disclosure).
4. Gate evidence (this host, isolated env roots per invocation, venv python
   3.10.12; `git rev-parse HEAD` verified unmoved after every pytest run;
   author and committer `Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>`
   on every commit, no push):
   - targeted at `02338c9b` (docs in the tree, uncommitted):
     `tests/test_usage_compaction.py` + `tests/test_lockfile_helpers.py`
     85 passed, 1 skipped (Windows-only), exit 0; the four other
     `tests/test_usage_*.py` suites + `tests/test_persistence_inventory.py`
     94 passed, exit 0; the `pid_is_alive`/lock consumers
     (`test_process_custody`, `test_services_tool_v2`, `test_launcher_sync`,
     `test_extension_reload_all`, `test_review_reconciliation_custody`,
     `test_evolution_restart_claims`,
     `test_available_subagents_runtime_review_fixes`,
     `test_launcher_server_reaper`, `test_bughunt_fixes`,
     `test_task_status_flow`, `test_evolution_commit_receipt`,
     `test_skill_lifecycle_queue`, `test_context`, `test_startup_hygiene`,
     `test_atomic_write_v639`) 382 passed, exit 0; no
     `tests/test_platform_layer*.py` exists in this tree;
   - `ruff check . --select F` exit 0; `scripts/check_domains.py` exit 0;
     `scripts/regenerate_size_ratchet.py --check` exit 0 (at every commit of
     the round the module stayed inside its band and the suite under the
     hard cap); `scripts/regenerate_inventories.py --check` exit 0;
     `git diff --check` exit 0 — each as its own command, rc printed.
   With this recorded, round 5.4 is executed on this host; the CI-shape
   battery and the serial pass were not re-run for this micro-round (its
   surfaces are the two pinned suites and the lock/liveness consumers above,
   all run), and the Windows LockFileEx mechanics remain owed to the 3-OS CI
   matrix.

## From the C6 micro-round 5.4 close-out (operator, base b4938c31; owner batch №12 A)

Three read-only lenses on the 5.4 delta: NEEDS_FIXES × 3, no HIGH (3 MEDIUM, 7 LOW). The operator closed them without a further agent round (the owner bound was one micro-round):

1. **R1 blast radius (MEDIUM, two lenses) — fixed.** Round 5.4 had moved `ENOLCK` out of the unsupported set so the probe kept the enforced tier and EVERY live acquisition failed closed. The primitive is shared (state singletons, task results, custody, claims, …), so a lockd-less NFS `state/` would have refused every locked write in the product, where the name protocol had always worked. Now `ENOLCK` selects the name tier like a filesystem that cannot, the probe records the errno beside the verdict, and `acquire_exclusive_file_lock(refuse_name_tier_errnos=…)` lets a caller fail closed on it — only `usage_ledger._named_lock` names `ENOLCK`. Red-first: `platform_layer.py`/`usage_ledger.py` @ `b4938c31` → `assert True == (True, 5)`, `assert True is False`.
2. **R4 typed resolution (MEDIUM) — fixed.** `_segment_path` resolves with the non-strict `os.path.realpath` inside the typed `try` (`except (OSError, RuntimeError)`). Red-first: a self-loop segment link → `RuntimeError: Symlink loop` / `OSError [Errno 40]` escaped @ `b4938c31`.
3. **R3 stamp-less symlink levels (LOW) — fixed**, pin deferred (suite at its 1600-line cap; mutation-verified by hand). **R5 witness inode tie + `_Abort` on a vanished ledger (LOW) — fixed**, no pin (disclosed). **R6 heartbeat-ownership pin (LOW) — disclosed, not fixed.** **R1 uncached unprobeable directory, R7 shared liveness primitive, R8d epoch-floor duration, R8e socket shape (LOW) — docs corrected** in DESIGN §8/§10/§12.5, packet §5.9/§5.10/§9/§10, ARCHITECTURE row.

Gates at the close-out tip: `tests/test_lockfile_helpers.py` + `tests/test_usage_compaction.py` + `tests/test_usage_*` + `tests/test_persistence_inventory.py` rc 0 (107 passed, 1 skipped Windows-only); `ruff --select F` rc 0; `scripts/check_domains.py` rc 0; `scripts/regenerate_size_ratchet.py --check` rc 0 (`platform_layer.py` 1500, `tests/test_usage_compaction.py` 1600 — both AT their ceilings); `git diff --check` rc 0. The CI-shape battery and `-m serial` run on the integration tree after the lane merges.

## From the C6 integration battery (serial lane on 9faccf31)

The serial lane found the one consumer of `pid_is_alive` the round-5.4 R7 flip (EPERM → alive) turned from "defer" into "signal": `ouroboros/workspace_executor._host_pid_matches_record` fell back to liveness whenever a foreground record had no `host_command_sha256` — a fallback written for Windows (no command line there) but reachable on POSIX by an owner-shaped FORGED record — so `kill_all_foreground` signalled pid 1 (`tests/test_workspace_executor_cleanup.py::test_executor_cleanup_ignores_owner_shaped_forged_host_pid_records`, red at 9faccf31: «owner-shaped forged record should be ignored»). The fallback is now Windows-only: on POSIX a registered process always carries its command hash, so an empty hash is not ours to kill — liveness is not ownership. *(correction, abe93702 → the platform primitive commit: both claims fell on macOS — `ps` can return no command line right after the spawn, so the fallback runs on POSIX too and asks «signalable by us»; see «From the macOS full-test 33658408570 and the platform guard» below)* Other consumers re-checked: process_custody, launcher_server_reaper, delegate_recovery, cancel_intents, extension_import_staging, review_owner_custody, skill_review_runner, agent_startup_checks all DEFER (keep/skip) on an alive pid; none signals on liveness alone. Executor suites (serial + non-serial), process custody and zombie prevention green after the fix; the packet §9 R7 disclosure stands with this consumer named.
## From the stage-2 fix wave, lane code-smalls (base 9faccf31)

Nine independent small fixes, one single-intent commit each, every behaviour
change pinned red-first. The lane touched no protected file except the
owner-sanctioned ABI-1 edit named in item 4.

| # | item | red-first evidence (pre-fix shape) | green |
|---|---|---|---|
| 1 | dead `TOTAL_BUDGET_LIMIT` copy in `supervisor/workers.py` (owner batch №6 item 3=A, second SSOT fix) | new pin in `tests/test_settings_budget_hotreload.py` fails on `assert not hasattr(workers, "TOTAL_BUDGET_LIMIT")` | 74 passed (`test_settings_budget_hotreload` + `test_legacy_timeout_retirement` + the three `workers.init` callers + `test_packaging_sync`) |
| 2 | no negative contract for the three ABI-6 P1 removals | a resurrection mutant re-adding all five retired names (`_call_llm_with_retry`, `compute_cost_with_children`, `format_handoff_message`, `_handoff_snippet`, `HANDOFF_SNIPPET_CHARS`) to `ouroboros/loop.py` + `ouroboros/task_status.py` → 5 failed, 2 passed | 7 passed; tree restored byte-identical after the demonstration |
| 3 | `timeout-minutes > 0` on the `system-e2e-mock` job | the new `>= 30` floor fails under a `40 -> 1` mutant, run against a TEMP COPY of `.github/workflows/ci.yml` (the real file is protected and untouched) | 5 passed |
| 4 | `plugin_api_surface_fingerprint` hashed only method NAMES and RuntimeInfo KEYS | with the pre-fix payload restored (`git stash`), both mutants — a parameter annotation `Sequence[str] -> List[str]`, and `server_port: int -> str` — produce the IDENTICAL digest `4b391ba5…` and negotiate happily; the two parametrized cases fail on that equality | 102 passed (`test_extension_plugin_api_matrix` + `test_contracts` + `test_plugin_api_admission` + `test_oop_extension_parity`), then 108 more across the extension loader/API suites |
| 5 | whitespace-only `model_experience` string accepted | the two new parametrize cases (`"   "`, `"\n\t \n"`) report `DID NOT RAISE SkillManifestError` | 147 passed (`test_skill_model_experience` + `test_skill_manifest_v11` + `test_contracts` + `test_skill_loader`) |
| 6 | `broadcast_ws` coroutine leaked when `run_coroutine_threadsafe` raised | the new pin drives a CLOSED loop through `broadcast_ws_sync` and records exactly one `coroutine 'broadcast_ws' was never awaited` | 82 passed (`test_broadcast_ws` + `test_extensions_api` + three ws suites) |
| 7 | route pin read `os.name == "nt"` (R-WINWAVE open item 1) | with the target binding forced to raise — the 8.3 short-path miss reproduced on this host — the scenario still passes its refusal contract on the native route with 0 adapter calls: the NEW pin is green and the OLD `os.name` expression fails `assert 0 == 3` | 20 passed |
| 8 | pinned transplant corpora silently reconstructed from the landed leaf | with `git` forced to fail, the pre-fix probes RUN on reconstructed bytes (the git_ops case does not even reconstruct faithfully: `unresolvable names: {BRANCH_DEV, BRANCH_STABLE, append_jsonl, current_drive_root, utc_now_iso}` — a meaningless red, not merely a vacuous green); post-fix the same run reports 7 honest SKIPs naming the file and base SHA | 50 passed, 0 skipped (the real corpus is present on this host) |
| 9 | retired comma-list keys dropped with no owner-visible word (review M2) | both new pins fail: no notice is emitted at all | 23 passed (`test_settings_read_seam`), then 123 across `test_settings_honesty`, `test_comma_list_sweep`, `test_legacy_timeout_retirement`, `test_abi5_q10_removals`, `test_config_extraction`, `test_review_cycles`, `test_rc_audit_fixture_suite` |

Dispositions beyond the plain fix:

1. **Item 1 — the accepted-and-ignored ARGUMENT went with the global (fixed,
   scope named).** The brief named `:34/:71/:75`. Keeping the
   `total_budget_limit` parameter while deleting the global it wrote would
   have left the worse half of the defect — an input accepted from
   `server.py` on every boot and honored by nobody — so `workers.init` no
   longer asks for it and the three test call sites drop the argument. This
   is the same idiom `tests/test_legacy_timeout_retirement.py` already pins
   for `soft_timeout`/`hard_timeout` (the FIRST fix of that owner batch). The
   two copies that ARE read (`supervisor.state`, the authority
   `budget_remaining` reads; `supervisor.message_bus`, the reporting plane)
   are untouched.
2. **Item 2 — the ADOPTION row text is NOT edited (disclosed).** Row ABI-6
   carries "DISCLOSED RESIDUAL: the three F3.0 removals are proven by
   surviving positive suites plus grep-level absence, not by dedicated
   negative-pin tests". That residual is now closed by
   `tests/test_abi6_removals.py`, but ADOPTION_v7next.md belongs to the
   ledger lane in this wave, so the row is left for it to update rather than
   risking a cross-lane conflict. `scripts/v7next_adoption.py` and
   `--release` are rc 0 as-is (the row was already `done`).
3. **Item 4 — protected-file edit, owner-sanctioned ABI-1 package.** The
   `2.0` fingerprint is re-recorded for the wider payload
   (`03fabdf4334e6b2bde217b4cb83a80faaebc773ddce870546dd2108b75de17ca`); a
   repo-wide grep confirmed no fixture, doc or receipt stored the previous
   digest, so nothing else needed re-recording. Both halves of the payload
   are the annotation SOURCE text, because `from __future__ import
   annotations` is active in the module.
4. **Item 5 — fixed by subtraction, not by a second error site.** The string
   form IS the one-key mapping form, so it routes through the mapping branch
   instead of carrying its own `return`. The two shapes now refuse
   identically BY CONSTRUCTION rather than through a copied message, and the
   function is one line shorter.
5. **Item 6 — the leak is in the code path; no test double is involved.**
   The brief offered both hypotheses. `broadcast_ws_sync` built the coroutine
   before the call and dropped it on the `RuntimeError` arm; the warning
   surfaces inside `test_api_extension*` only because an earlier test leaves
   a finished loop in the module global, so the collection point — not the
   author — got the blame. The pin records with `simplefilter("always")`, not
   `"error"`: raised from `__del__`, an escalated warning becomes UNRAISABLE
   (a pytest side note) and the pin would have stayed green. That was
   observed on the first draft of this pin.
6. **Item 7 — the predicate is the BINDING, not the access-layer detector.**
   A first draft pinned on `light_cognitive_or_root_redirect` returning text;
   that is wrong, because BOTH routes take their text from that same
   function. `registry_core` reaches the light repo-mutation block only when
   `_build_builtin_target_binding` RETURNED a binding (legacy text → adapter,
   3 calls); when it RAISES, the except arm answers from the resolution layer
   with a native `ToolResult` (0 calls). The test now reads that predicate.
   Only the test landed — the R-WINWAVE row text belongs to the ledger lane.
7. **Item 8 — fixed as a CLASS, one file wider than the brief.** The brief
   named `_QUEUE_BASE_SHA` and `_GO_BASE_SHA`; `LOOP_UPSTREAM` (line 211 of
   the pre-fix file) carried the identical fallback and is included, and the
   three copies of the fetch collapse into one reader that returns EMPTY on
   an unreachable object. The pre-existing `needs_go_corpus` marker could
   never fire because `GO_UPSTREAM` was never empty; each real case now
   carries a skipif naming its file and base SHA.
8. **Item 9 — the notice is a LOG line, not a typed event (disclosed
   residual).** `normalize_settings_raw` is THE raw-stage seam every settings
   reader applies, and it exists precisely so that a read stays a read;
   routing the notice into `logs/events.jsonl` would make every settings read
   a write. So the first read that finds a retired key warns on the module
   logger (`server.log` + stdout), once per process per dropped set, naming
   the exact keys and — for the comma-list family — `OUROBOROS_REVIEWER_SLOTS`
   plus the fact that the SHIPPED default panel is what runs until it is
   authored. The docstring's purity claim is amended to name this one effect
   instead of continuing to claim unqualified purity; the properties the pin
   actually asserts (no file read, no document persisted, no environment, no
   mutation of the caller's mapping, idempotent output) all still hold.
   An owner who watches only the Logs panel will not see it there.

Gates, each its own command with rc printed at the lane tip `fd0fa676`:
`ruff check . --select F` rc 0; `scripts/check_domains.py` rc 0 (no module row
moved, nothing to commit); `scripts/regenerate_inventories.py --check` rc 0
(no inventory changed); `scripts/regenerate_size_ratchet.py --check` rc 0
(`supervisor/workers.py` only SHRANK; the three files at their exact caps —
`ouroboros/utils.py`, `ouroboros/platform_layer.py`,
`tests/test_usage_compaction.py` — were not touched);
`scripts/v7next_adoption.py` rc 0 and `--release` rc 0 (37 rows, 36 done, 1
deferred); `git diff --check` rc 0; `git diff --check 9faccf31..HEAD` rc 0.
Suites: the 17 directly touched files 380 passed; the wider related sweep
(every `test_*settings*`, `*config*`, `*skill*`, `*extension*`, `*worker*`,
`*supervisor*`, `*ws*`, `*plugin*`) 1651 passed, 7 skipped.
`git rev-parse HEAD` was re-read after every pytest invocation and never
moved off the lane branch.
## From the stage-2 fix wave, lane ui-cost-regression (base 9faccf31)

One user-visible regression: a child subagent card showed `cost pending`
forever. The root cause was not the alias removal itself but what the browser
does with a name it no longer receives. Since ABI-3 `cost_projection.py:141`
(`out.pop(old, None)`, 33ba6e83) strips the retired `cost_usd` /
`cost_usd_with_children` aliases from every frame, while `chat.js` `costMetaKeys`
copies all twelve cost names unconditionally — so an alias-free frame reaches the
readers with the retired names as own properties valued `undefined`. Nothing
serializes them away on that path: `costMetaKeys` output goes straight into
`summarizeSubagentCardFrame` -> `withTaskCostMeta` -> `taskCostProjection`, with
no JSON round-trip in between. Both readers tested presence with bare
`hasOwnProperty`, so a key with no value counted as a value.

Live path (`task_done` on the log channel ->
`chat.js::routeSubagentTerminalToCard`, chat.js:2399-2430 on this base ->
`summarizeSubagentCardFrame` -> `withTaskCostMeta` -> `taskCostProjection`):
the terminal frame arrives AFTER the honest progress frame, projects
`final: true`, and therefore outranks it in `mergeStickyCostMeta` for the rest
of the run. The reload path (chat.js:2992) calls the same
`routeSubagentTerminalToCard`, so it froze identically.

Class fix, both readers, presence now means a DEFINED value:

- `web/modules/utils.js` `resolveCostPair` — the ONE pair resolver shared by
  every reader. An own property valued `undefined` is absence; an explicit
  `null` is still present (Python parity with `resolve_cost_pair`'s `old in
  src`), and a mirrored legacy amount still wins its pair, so the
  alias-carrying upstream frames (f3fbfdbb / a76961de) are unaffected.
- `web/modules/chat_activity.js` `taskCostMeta` — the accounting-evidence test.
  The same materialized keys made EVERY whitelisted frame look like evidence, so
  an evidence-free terminal projected `cost pending` at pending rank and
  overwrote a live ceiling on recency alone — the exact thing
  `taskCostProjection` documents it never does.

Rejected: dropping the two retired names from `costMetaKeys`. It fixes this
frame and silently breaks the other direction — an upstream producer that still
mirrors only `cost_usd_with_children` would lose its amount entirely.

Red-first evidence (pins shown failing on the pre-fix modules, then green):

| # | Pin | Pre-fix (red) | Post-fix (green) |
| - | --- | --- | --- |
| 1 | `web/tests/chat_instance_dom.test.js` "an alias-free subagent terminal keeps the honest amount, live and on reload" — real `createChatInstance`, honest progress frame then an alias-free `task_done` on the log channel | `error: 'the alias-free terminal settles the amount'`, actual `<span class="chat-live-meta-text">cost pending</span>` | `$0.99` on the card; `$1.25` when a frame mirrors the legacy alias; `$0.50` on the history-replay (reload) route |
| 2 | same test, the costless-terminal step (`task_done` with no cost fields after the live ceiling) | `error: 'a costless terminal keeps the ceiling'`, actual `cost pending` | `up to $0.99` kept |
| 3 | `web/tests/cost_presentation.test.js` resolver + evidence assertions (`{cost_usd: undefined, accounted_upper_bound_usd: 9}` -> 9; `{cost_usd: null, …}` -> null; an all-`undefined` frame -> no meta) | resolver answered `null` for the honest 9 | as asserted |

Dispositions:

1. **The `cost pending` regression — FIXED** (both readers, pins 1-3).
2. **`ABI3_GATEWAY_ALIAS_INVENTORY.md:32` — FIXED.** It claimed the JS side
   needed no edit and that "undefined drops out of JSON". False for this
   in-memory path — nothing serializes between the whitelist and the reader.
   The bullet now states what actually happens, that the removal DID need a web
   edit, and that the `api_types.js` typedefs already name the honest fields
   (its "JSDoc goes stale — HOT-DEFERRED" note was itself stale).
3. **Python twin — checked, none demonstrable; DISCLOSED.** `resolve_cost_pair`
   has the same shape of rule (`old in src` wins with a `None` value), but no
   live Python producer materializes a retired key beside a set honest name:
   `with_cost_aliases` pops the retired names on every write seam, and the
   remaining `"cost_usd"` literals in `ouroboros/` (usage rows, delegate
   evidence, triad/skill receipts, consciousness and reflection rows) are their
   own schemas that carry no honest-name pair to shadow. Python left untouched.
4. **`web/tests/chat_instance_dom.test.js` entered the 1001-1500 size band
   (1000 -> 1107) — DISCLOSED, band rationale in the same commit.** The
   regression only reproduces through the real `createChatInstance` card path,
   whose DOM harness lives in this file; a new neighbour file would have paid
   the cap with a third copy of that ~186-line harness, which the owner rules
   forbid as a design reason.

Gate evidence (this host, isolated env roots per invocation, venv python;
`git rev-parse HEAD` verified unmoved after every pytest; author and committer
`Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>`, no push) is
recorded in the lane report: `cd web && node --test tests/` 848 passed / 0
failed rc 0 (847 before the new pin); `tests/test_web_utils_ssot.py`,
`test_cost_projection.py`, `test_gateway_abi3_removals.py`,
`test_widgets_ui_static.py`, `test_qa_fixes_6_12.py`, `test_restart_reconnect.py`,
`test_projects_v6640.py` rc 0; `ruff check . --select F` rc 0;
`scripts/check_domains.py` rc 0; `scripts/regenerate_inventories.py --check`
rc 0; `scripts/regenerate_size_ratchet.py --check` rc 0 after regeneration with
the band rationale; `scripts/v7next_adoption.py` and `--release` rc 0;
`git diff --check` and `git diff --check 9faccf31..HEAD` rc 0.
## From the stage-2 fix wave, lane ledger-validator (base 9faccf31)

Scope: `ADOPTION_v7next.md`, `scripts/v7next_adoption.py`,
`docs/v7next/WINWAVE_CLASS_REGISTRY.md` and this file. The lane's finding is
one class, not six: **the manifest was checked for shape, never for truth**, so
every claim it made about the tree was load-bearing and unpinned. A whole-file
overwrite could delete a row, a hook could name a pin nobody wrote, and a row
could say in prose the opposite of its own status cell — all at rc 0 in both
validator modes.

### 1. The deleted train row (restored)

`TRAIN-F6b-f3fbfdbb` — upstream sync #2, 101 commits `8d13373b..f3fbfdbb`,
campaign merge `b9ceed6e` — was written in `8ac736a8` and disappeared in
`285ab66d` ("adoption: the ledger says what the tree proves"), whose whole-file
rewrite came from a lane based before the row existed. Nothing announced the
loss: `git show 285ab66d -- ADOPTION_v7next.md` shows the row deleted in the
same diff that added 47 other lines.

Restored verbatim (commit `1525fff5`). Its hook names no test that moved, and
the ledger section it cites («From the F6 rolling-upstream sync #2», line 3839
of this file) is still here, so no adjustment was owed.

The deletion is the lane's red-first evidence, measured rather than argued: on
the pre-fix validator, the live manifest with **every** `TRAIN-` row removed
returned rc 0 in the default mode **and** rc 0 under `--release`.

### 2. The missing train row (added)

Upstream sync #3 had no row at all: 40 upstream commits
`f3fbfdbb..a76961de` (upstream release 6.114.0, the live chat card, the Agents
panel, the SYSTEM.md rewrite, governance schemas) merged whole as `f4abe0a5`
(parents `43dcc1d2`, `a76961de`) on 2026-09-02, under the owner signal recorded
as `[A-BATCH-12-C6-SYNC]` in `~/.claude/ouroboros_requirements_archive.md`,
owner verbatim: «+не забудь подтянуть все свежие изменения, там многое
изменилось вреоятно уже в ветке ouroboros. Или это и так в плане в конце?».
`TRAIN-F6c-a76961de` (commit `4b004a2e`) cites the merge commit message for the
by-class resolutions rather than restating them, and hooks the five suites the
merge actually moved bytes in: `tests/test_packaging_sync.py` (the SAFETY and
SYSTEM mirrors), `tests/test_golden_capabilities.py`,
`tests/test_capability_effect_predicates.py`,
`tests/test_terminal_delegation_receipt.py`, `tests/test_git_extraction.py`.

### 3. Three enforcement rules and the pytest wrapper (commit `19a4ff7d`)

**Train coverage, both modes.** `REQUIRED_TRAINS` maps each absorbed upstream
train to `(upstream tip, campaign merge)`; a missing row, a row of the wrong
kind, or a row whose text no longer names both SHAs is an error in the default
mode as well as under `--release`.

Frozen inventory rather than a git derivation, and the reason is this history.
Each sync's absorb merge *does* take its upstream tip as the literal second
parent — `20850191` (parent `8d13373b`), `b9ceed6e` (parent `f3fbfdbb`),
`f4abe0a5` (parent `a76961de`) — but only `f4abe0a5` sits on this branch's
first-parent line. The other two were made on lane lines and reached mainline
as the second parent of a lane-integration merge over a *campaign* commit:
`0aa74e9f` over `816e7b82` carried `20850191`, and `0f9a8daf` over `4c32691e`
carried `b9ceed6e`. So a rule walking `--first-parent` merges and reading second
parents would police one train of three and stay blind to exactly the two whose
row was lost; widened to «second parent descends from a recorded upstream tip»
it would demand a train row for every lane merge made after a sync — 35, 15 and
6 merges on this tree for the three tips, the C6 lane merge `9faccf31` (second
parent `8fb08d44`) included. Neither form is honest, and both need a subprocess
in a checker that today reads one file. Adding a train to the inventory is the
same edit as merging one.

**Hook `::nodeid` tokens read by AST.** The path half of a hook was resolved;
the `::name` half was free text, so a hook could point at a suite that exists
and a pin that does not. `_defined_names()` collects functions and classes at
any depth (`path::Class::method`) plus module-level bindings — the latter
because a hook may legitimately name the closed inventory a pin drives
(`tests/_shared.py::SETTINGS_WRITERS`), not only the pin. An unparseable hook
file is an error, not a pass.

**A `done` row may not say the work is open.** `not done`, `open residual`,
`not integrated yet`, `still owed`, `read pending` (case-insensitive) in a row
with `status=done` is an error unless the row declares what stays open in an
explicit `residual:` clause. This is a text-vs-cell consistency lint on an
operator manifest — not a gate on any runtime decision of Ouroboros (BIBLE P5
is about the latter) — and the escape is explicit so a genuine disclosure on a
shipped row stays sayable. `CPL-1` carried a *negated* «no longer an open
residual», which the rule would have read as a marker; the phrase was reworded
to «no longer open» rather than escaped, because it is not a residual.

Hook resolution also moved off the `--release` invocation onto the property it
actually is — something true of a **shipped** row — which is what the
manifest's own Notes already claimed the validator did.

`tests/test_v7next_adoption.py` (new, 144 lines) runs `validate()` in both
modes on the live manifest and drives one mutant per rule. This closes «the
release bar is executed by nothing automatic» at the test level only:
`.github/workflows/ci.yml` is a protected file and was NOT touched, so the bar
runs wherever the default pytest lane runs, not as a separate CI job.

### Red-first table

| # | pin | red on the pre-fix shape | green after |
|---|---|---|---|
| 1 | train coverage — every `TRAIN-` row deleted from the live manifest | pre-fix `scripts/v7next_adoption.py` @ `9faccf31`: rc 0 default **and** rc 0 `--release` (the deletion of `TRAIN-F6b` in `285ab66d` passed for exactly this reason) | `test_deleting_an_upstream_train_row_turns_the_bar_red` — 6 cases (3 trains × 2 modes) |
| 2 | train row re-pointed at another merge | same shape: rc 0 both modes | `test_repointing_a_train_row_at_another_merge_turns_the_bar_red` — 2 cases |
| 3 | hook naming a pin that does not exist (`tests/test_smoke.py::test_no_such_pin_was_ever_written`) | pre-fix: accepted, the path resolved and the nodeid was prose | `test_a_bogus_hook_nodeid_turns_the_bar_red` |
| 4 | `done` row whose text says the work is open | pre-fix: accepted for all five markers | `test_a_done_row_that_says_it_is_not_done_turns_the_bar_red` — 5 cases |
| 5 | goldens: live manifest green both modes; a real nodeid (incl. a module-level inventory) accepted; a `residual:` clause is the escape | green before and after — characterizing, by design | 3 cases |
| 6 | post-release without a recorded deferral; a required row parked post-release by operator authority | green on this lane's own base **only because the vocabulary landed one commit earlier** (`b89b9bd2`) — the pre-`b89b9bd2` script has no `DEFERRED_OUT_OF_V70` symbol at all, so no red on the old shape is expressible; recorded as goldens for that commit, not as red-first | `test_a_post_release_row_needs_a_recorded_deferral`, `test_a_required_row_cannot_be_parked_post_release_by_the_operator` |

Step-A evidence for rows 1-4 (the honest form of «red on the pre-fix shape»,
since the suite imports symbols the pre-fix script does not export): the
constant `REQUIRED_TRAINS` was added with **no enforcement wired**, and
`tests/test_v7next_adoption.py` ran 14 failed / 5 passed. With the three rules
wired: 19 passed.

### Prose and status corrections (commit `f2dea89b`)

| row | claim | disposition |
|---|---|---|
| D06 | «all 44 kinds the tree actually has» | **fixed** → 45. `supervisor/event_taxonomy.py::EVENT_DISPOSITIONS` has 45 keys: 34 `worker_handler` + 7 `telemetry_only` + 1 `server_intercept` + 3 `nested_log_event`. The row's own arithmetic (34+7, plus the two tiers it asserts absent from `EVENT_HANDLERS`) already summed to 45 |
| D07 | «so the disposition leaves pending-decision» beside `status=done` | **fixed**. «Leaves» was meant as *departs from*; next to a done row it reads as *keeps it pending*. Now: «is no longer pending-decision: it reads re-prove and the row ships» |
| D08 | «Disposition therefore leaves pending-decision for re-prove» | **fixed**, same ambiguity → «moves off pending-decision to re-prove» |
| D18 | «OPEN RESIDUAL, and the reason the row is not done: MIGRATION row 1030 … is neither re-applied nor dispositioned» while the row reads `done` and its own hook names `test_queue_snapshot_path_has_a_single_authority` | **fixed, and the inversion stated.** The residual was closed by `091ee3b3` — in the direction **opposite** to the oracle. MIGRATION row 1030 reads «retired: supervisor.state owns the queue snapshot path; the queue reads it through the module at use time»; what landed is `supervisor.queue` as the sole authority with no copy in `supervisor/state.py`. The defect cured is the same (two module globals answering one question, agreeing only because both `init()`s got the same drive root); the harness consequence is mirrored — the oracle noted that an isolation harness must then call `state.init` because `queue.init` alone would no longer redirect, and here it is `queue.init` that redirects and `state.init` alone that does not (production binds both either way). The oracle's hook name never came across either (`test_the_queue_snapshot_path_has_one_owner`, which does not exist in `tests/test_heartbeat_presentation.py`). Choosing the direction was the **operator's** call, not an owner decision: disclosed here and in the row for the owner to overturn |
| R-WINWAVE | «NOT DONE, and the reason is the re-prove itself» + «its result is recorded as PENDING because nobody has read it» beside `status=done` and a hook listing four green 3-OS runs | **fixed.** The re-prove is run 33569841899 (`8b27b507`, full 3-OS green) and it held on 33570328266, 33571681398, 33572515529 and on the later tips 33579445704 (`1072a317`), 33624546416 (`ac17fa03`), 33626834806 (`43dcc1d2`). The «unread» run 33563498919 has been read (it cleared class 17 and surfaced nine further Windows classes, fixed in `20afdbb7..e0aee1ac`). What remains is **freshness, not colour**: run 33644668074 on the sync #3 merge `f4abe0a5` is dispatched and unread. The second item — the accepted 2026-08-30 audit item to re-pin the registry route test on the actual alias condition — is **still open on this tree** (`tests/test_registry_core.py:813` reads `os.name`) and is named with the lane landing it: the smalls lane of this same stage-2 wave |
| ABI-4 | «`provider_models.delegated_route_target(route)` feeds the run-request assembly — relocated from a `DelegationRoute` method» | **fixed.** No such symbol exists on this tree (`rg delegated_route_target` — zero hits). The landed shape is the method: `DelegationRoute.resolved_target()` at `ouroboros/subagents.py:183`, read once by `ouroboros/tools/delegate.py:245`. The relocation an earlier draft described did not happen, and the row now says so |
| CPL-1 | «are no longer an open residual» | **fixed** (reworded to «no longer open») — a negated marker would have read as a real one to the new honesty rule |
| Notes | «D04/D05 are decided but their lanes are still owed, which is why they read `pending` and not `done`» | **fixed.** Both landed (D04 `5b1767fa`, D05 `0bf723cc`) and both rows read `done`; the sentence outlived the lanes |
| Notes | «The row that still owes work is CPL-4 (C6, on the owner checkpoint)» | **fixed.** CPL-4 landed with the C6 usage-ledger compaction lane (merge `9faccf31`, this lane's own base). Found while fixing the neighbouring sentence and corrected in the same commit rather than left as a known-stale line |

### Rows added (commit `b89b9bd2`)

| row | provenance | note |
|---|---|---|
| `DEFER-BROWSER` | owner batch №9 №14, 2026-09-01, archive `[A-BATCH-9-ANSWERS]`; owner verbatim «14. A» on the option recorded as «браузерная волна пост-релиз, смоук зелёным до тега» | The gateway/UI-truth E2E actor is deferred out of 7.0 by the owner, and the condition on the tag is a green smoke rather than a green browser lane. Hook: `tests/system_e2e/interfaces.py`'s refusing `PlaywrightUIClient` stub and its pin `tests/system_e2e/test_system_scenarios.py::test_interface_stubs_refuse_instantiation_until_their_lanes_land`. `residual:` clause names what nobody covers in 7.0 |
| `W4-F1` | `docs/v7next/LEDGER_CORRECTIONS.md` «From the F4 wave 4» findings table (this file, the W4-F1 row) | Crash window between the reviewed `git commit` and `record_evolution_commit`: a landed reviewed commit no boot path will attribute. `post-release` |
| `W4-F2` | same table, W4-F2 row | Absorb write and `cycle_outcome` append are not one transaction; the digest can under-report a cycle forever. `post-release` |
| — | same table, W4-F3 and W4-F4 | **SUPERSEDED by d348ea46 (2026-09-02): both ARE rows, `operator-disclosed`, no owner quote — corrected by the F3-C lane, see «From the F3-C lane» at the end of this file.** Original note: **disclosed observations, no row.** Both are named asymmetries of decisions that already exist (the restart-marker knob predates the claim machinery; rescue-local refs are deliberately durable), not work owed. Recorded as such in the manifest Notes |

The validator's single post-release allowlist (`OWNER_DEFERRED = {"ABI-8"}`)
became `DEFERRED_OUT_OF_V70`, an id → authority record. Every post-release row
must be listed, and a row of the owner-approved required inventory
(`REQUIRED_PHASE`) may only be parked there with `OWNER` authority — so the
anti-bypass property the frozenset carried is unchanged, while an operator
disclosure can be stated as what it is instead of borrowing an owner decision
it does not have. `DEFER-BROWSER` is also pinned to phase `POST` in
`REQUIRED_PHASE` so it cannot be silently re-phased.

### WINWAVE registry reconciliation (commit `fada56d6`)

Open item 2 read «No green windows leg exists yet on any frozen SHA» while the
run table three sections below it logged a green Windows leg (33568728122,
`f5a94675`) and four consecutive green 3-OS matrices from 33569841899 onward —
the same table the ADOPTION row cites as its re-prove. The item now states what
is actually open: **freshness on the newest tip**, not colour. Open item 1 gains
the lane landing the alias-condition repin.

Later CI facts appended to the run table: 33579445704 (`1072a317`) green,
33624546416 (`ac17fa03`) green, 33626834806 (`43dcc1d2`) green — full-test 3-OS
green on the **first** attempt — and 33644668074 (`f4abe0a5`, the sync #3 merge)
**pending**, written as pending because a verdict nobody read is not evidence.

The two red `system-e2e-mock` subtests on attempt 1 of 33626834806 are written
up in the registry as what they were: races in the mock lane's own scaffolding
— the `/proc`-environ scan of `pids_with_env_value` (a process can exit between
the listing and the read) and an S22 wait that assumed its window was wide
enough under CI load — green on rerun of the same job on the same SHA, and
57/57 twice locally. Recorded so the attempt-1 red is not later misread as a
cross-OS class; it belongs to the E2E lane's ledger, not to this row.

### Not done, and why

- **The pending 3-OS matrix on `f4abe0a5` (run 33644668074) was not read.** No
  verdict was fetched in this lane, so both the registry and the R-WINWAVE row
  say «pending» rather than guessing. The re-prove itself stands on the seven
  green runs already logged; what is missing is coverage of the newest bytes.
- **Run 33574822693 (`d21806d8`) is left as it was recorded** («pending at the
  time of writing»). It is the first dispatch of the scheduled
  `system-e2e-mock` job, its outcome was not verified in this lane, and
  overwriting an unread verdict with a guess is the failure mode the row's own
  wording warns about.
- **`ci.yml` was not touched.** The brief forbids it and it is a protected
  file, so «the release bar is executed by nothing automatic» is closed at the
  test level only: `scripts/v7next_adoption.py` now has a pytest wrapper, and
  it runs wherever the default lane runs.
- **The alias-condition repin (`tests/test_registry_core.py:813`) was not
  applied here** — it is the smalls lane's item in this same wave. This lane
  only names it and its owner.
- **The D18 direction was not sent back to the owner.** The inversion is
  disclosed in the row and here; deciding it is not this lane's authority, and
  it changes no behaviour (production binds both modules).

### Gates (host 0897-oma, 2026-09-02; every python/pytest under a fresh mktemp `OUROBOROS_APP_ROOT`/`REPO_DIR`/`DATA_DIR`/`SETTINGS_PATH` plus a private `TMPDIR`, `-p no:cacheprovider`, `git rev-parse HEAD` re-checked after every pytest)

Each as its own command, rc printed: `tests/test_v7next_adoption.py` +
`tests/test_legacy_timeout_retirement.py` + `tests/test_event_taxonomy.py` +
`tests/test_module_handle_extraction.py` — 197 passed, rc 0;
`scripts/v7next_adoption.py` rc 0 and `--release` rc 0 (42 rows);
`ruff check . --select F` rc 0; `scripts/check_domains.py` rc 0;
`scripts/regenerate_inventories.py --check` rc 0;
`scripts/regenerate_size_ratchet.py --check` rc 0 (no path of this lane is in
`GIANT_PATHS`, `BAND_BASELINE_PATHS` or `BYTE_DEBT`; the new suite is 144
lines, `scripts/v7next_adoption.py` 418); `git diff --check` and
`git diff --check 9faccf31..HEAD` rc 0. No helper module, wrapper or
neighbour file was added to pay a size cap; the only new file is the pin the
brief asked for.
## From the stage-2 fix wave, lane docs-truth (base 9faccf31)

Nine documentation and generated-report defects the stage-2 review verified on
the integration tree. Every one is pinned red-first: the pin was run against
the pre-fix shape of its own subject (working tree reverted per file, or the
pre-fix commit range for the whitespace gate) and shown red before the fix.

| # | defect (pre-fix state) | red-first witness | disposition |
|---|---|---|---|
| 1 | `docs/ARCHITECTURE.md` — README calls it the full component map, but 24 tracked runtime modules had no row: `_usage_response`, `context_mode_compat`, `credential_shapes`, `delegate_custody_usage`, `evolution_fingerprint`, `gateway/onboarding_host`, `gateway/task_events`, `marketplace/install_specs`, `review_actor_aggregation`, `review_session_usage`, `review_thread_continuity`, `settings_integrity`, `skill_owner_attestation`, `tools/{compact_context,control_delegation,evolution_stats,knowledge,memory_tools,plan_review_artifacts,search,tool_discovery}`, `transport_custody`, `version`, `supervisor/subagent_task_truth` | `test_docs_sync.py::test_architecture_component_map_covers_every_live_runtime_module` on the pre-row document: AssertionError naming exactly those 24 paths | **fixed** — a row each, written from the module docstring and its callers (purpose, data flow, facade relation), placed under its real package next to its domain neighbours (`ouroboros/domains.toml` used for the owner). No module was judged deliberately private, so the pin was NOT weakened |
| 2 | `ARCHITECTURE.md` deep-review row promised an atlas retry "once with the compact manifest", which `deep_self_review.py` explicitly removed (compact IS the default; there is no fuller form to fall back from) | `…::test_architecture_deep_review_has_no_compact_manifest_retry_rung`: red on `retries once with the compact manifest` | **fixed** — the row now states the no-retry rule and the P3 no-pack outcome; the distinct final-shrink rebuild (`hard_budget_reduction`, still live) stays described |
| 3 | Default-settings table skipped two live `SETTINGS_DEFAULTS` keys: `OUROBOROS_CONTEXT_MODE_AUTO_LOW`, `OUROBOROS_CLAWHUB_REGISTRY_URL` | `…::test_settings_docs_name_every_key_owner_and_what_startup_persists`: red at `OUROBOROS_CONTEXT_MODE_AUTO_LOW is missing from the settings table` | **fixed** — a row each, keyed to their real consumers (`config.get_owner_context_mode` provenance rule; `config.get_clawhub_registry_url` normalization, with the callers' host allowlists named so the row is not read as an authorization) |
| 4 | Two places said server startup "persists nothing", while the lifespan's `load_settings()` runs `context_mode_compat.normalize_and_persist_context_mode_compat`, which rewrites the compat pair under a held lock (topology line 19 and the settings-write section) | same pin: red on `boot provider normalization in-process and persists nothing` | **fixed** — both places now say that no provider decision is persisted AND that the one write startup can make is the compat-pair migration inside the read seam (raw mapping plus the pair, lock-held, warn-and-retry otherwise). The document's two other "persists nothing" statements (onboarding failure path, no-change owner transform) are true and were left alone — the pin is phrase-scoped for that reason |
| 5 | `README.md`, the `DEVELOPMENT.md` numeric-timeout checklist item and ARCHITECTURE invariant 3 assigned settings/defaults ownership to the `config.py` facade after the v7next split moved the vocabularies into leaves | same pin: red on the README `exact settings and defaults live in` clause, on the DEVELOPMENT `an SSOT in \`config.py\` \`SETTINGS_DEFAULTS\`` clause, and on invariant 3 not naming the owners | **fixed** — all three now point at `settings_defaults` / `settings_scales` / `model_slots` / `review_model_routes` / `runtime_limits` / `settings_integrity` with `config.py` kept as the one import surface. Note on the review's line numbers: `DEVELOPMENT.md:2074-2076` is the provider-failure checklist on this base; the actual carrier is the numeric-timeout SSOT item (`:2103`), and `ARCHITECTURE.md:2956` already described the five-leaf family correctly |
| 6 | `docs/v7next/DESIGN_MODEL_VISIBLE_LOGGED.md` still opened "DESIGN ONLY … code lands in a later lane" and its §3.2 still demanded a fail-closed dispatch refusal on a reconstruction mismatch, against the landed observability-only contract | `…::test_model_send_design_note_matches_the_landed_observability_contract`: red on `Status: DESIGN ONLY` (and on the `PhysicalAttemptPreparationFailed` clause) | **fixed** — status names the landed module, its wiring and its pins; §3.2 states the observability rule and why fail-closed was rejected on its merits (a record defect must not stop a paid, otherwise-correct call, and the candidate fact keeps its unchanged fail-closed refusal above the call); the stale "gap the implementation must close" paragraph is retired |
| 7 | `ARCHITECTURE.md` §11.4 "Recent ABI Retirements" ended at `5.25.0-rc.4` — no ABI 7.0 entry at all | `…::test_recent_abi_retirements_section_carries_the_abi_70_window`: red on `**ABI 7.0**` absent from the section | **fixed** — one entry summarising the window from the ADOPTION rows ABI-1..ABI-10, including the pre-upgrade migration note for the reviewer comma-list keys, `scripts/rc_audit.py` as the executable note, and the explicit statement that the handler ABI (ABI-8) is NOT in it. Membership of `RETIRED_SETTING_KEYS` is deliberately delegated to the tuple instead of claimed exhaustively in prose |
| 8 | `ouroboros/domains.toml` and `docs/DOMAIN_MAP.md` were reachable from neither `DEVELOPMENT.md` nor `README.md`, so a contributor met the domain gate only as a red run | `…::test_the_domain_manifest_is_reachable_from_the_handbook`: red on `DEVELOPMENT.md never mentions ouroboros/domains.toml` | **fixed** — the handbook's Role-and-authority section names the manifest as the module→domain SSOT, the map as its generated projection, `scripts/check_domains.py --write` as the one regeneration for both, the witness report, and that a new cross-domain direction is an owner decision rather than a manifest edit |
| 9 | `scripts/v7next_domain_report.py` wrote `"\n".join(L) + "\n"` over a list whose last element is a section separator `""`, so the report ended with a blank line and the whitespace gate was red; the committed report was also bound to HEAD `5187fcdc` / 488 modules / 80 `proposed` / a domains.toml sha that no longer exists | `git diff --check f3fbfdbb..HEAD` → `docs/v7next/DOMAIN_QUOTIENT_REPORT.md:1966: new blank line at EOF`, plus `…::test_the_domain_quotient_report_ends_without_a_blank_line` red on the committed artifact | **fixed** — the generator drops trailing separators and writes exactly one newline; the report is regenerated as the current witness (509 modules, 0 proposed, manifest drift none, 1614 module edges / 164 domain edges / 1 cycle group). `git diff --check f3fbfdbb..HEAD` is now rc 0 |

Disclosed, not fixed:

1. `docs/v7next/DESIGN_USAGE_COMPACTION.md` §10 still calls CPL-5's
   implementation "not yet landed on this base". That was true on the C6 lane
   base and is stale on this tip (`ouroboros/model_send_seal.py` is wired and
   swept). It is the same class as item 6 but was not in this lane's verified
   item list, so it is reported rather than edited: a one-line parenthetical an
   owner can sanction.
2. `docs/CHECKLISTS.md` is untouched by instruction (protected; its two stale
   sentences are an owner question).

Rejected: none. Every item reproduced on the tree exactly as the review
described it, apart from the two line-number drifts noted in row 5.

Gate evidence (this host, a fresh isolated env root plus private `TMPDIR` per
invocation, `~/ouro/venv` python; `git rev-parse HEAD` verified unmoved after
every pytest run; author and committer
`Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>` on all seven
commits; no push), each as its own command with its rc printed:

- `tests/test_docs_sync.py` rc 0 (15 passed) — run after every item, and per
  item against the reverted subject to show the red first;
- `tests/test_architecture_facts.py tests/test_docs_sync.py
  tests/test_doc_context.py tests/test_packaging_sync.py
  tests/test_domain_manifest.py tests/test_model_send_seal.py
  tests/test_legacy_timeout_retirement.py tests/test_skip_tests_doc_only.py
  tests/test_public_site_metadata.py` rc 0 (160 passed, `-n 6`);
- `tests/test_smoke.py tests/test_context_layout.py tests/test_context.py
  tests/test_repo_read_limits.py tests/test_settings_read_seam.py
  tests/test_release_proof.py tests/test_contracts.py` rc 0 (`-n 6`);
- `ruff check . --select F` rc 0; `scripts/check_domains.py` rc 0 (no manifest
  row moved, so no `--write` commit was needed);
  `scripts/regenerate_inventories.py --check` rc 0 (no inventory changed);
  `scripts/regenerate_size_ratchet.py --check` rc 0
  (`ouroboros/utils.py`, `ouroboros/platform_layer.py` and
  `tests/test_usage_compaction.py` untouched at their caps);
  `scripts/v7next_adoption.py` rc 0 and `--release` rc 0 (37 rows, unchanged);
  `git diff --check` rc 0, `git diff --check 9faccf31..HEAD` rc 0 and
  `git diff --check f3fbfdbb..HEAD` rc 0.

Not run for this lane: the CI-shape battery and the `-m serial` pass. The
delta is documentation, one generated report, one report-only generator write
and one test module; no runtime module changed.
## From the stage-2 fix wave, lane persistence-inventory (base 9faccf31)

CPL-4's claim — «every durable entity under the data root is inventoried»
(docs/PERSISTENCE.md header, ARCHITECTURE §10 prose) — was false and its own
verify pair could not see it. Two independent holes:

1. **The scanner collapsed every non-literal segment to `*`.** A durable file
   whose name lives in a module constant, an imported constant, a literal
   `Path` chain, an f-string or a helper's returned prefix never entered the
   scan population under its own name, so the forward check never asked for
   its row. Twelve live entities were inventoried nowhere.
2. **A wildcard matched anything in both directions.** The resulting `state/*`
   was "covered" by the exact row `state/state.json`, and backwards every
   exact row stayed "real" with no writer behind it; a dotted token such as
   `*.lock` answered by basename for any path whose leaf was unresolved. Both
   directions of a count-anchored contract were vacuous for anything the
   scanner could not name.

### Red-first

| # | Pin | Red on the pre-fix shape | Green |
|---|---|---|---|
| 1 | `test_constant_named_durable_files_are_visible_to_the_scan` | base scanner (9faccf31 `tests/test_persistence_inventory.py` + the pin appended): `constant-named durable files invisible to the scan: ['logs/chat_annotations.jsonl', 'state/claudexor_rotation_provisioning.json', 'state/delegate_terminal_refresh_cursor.json', 'state/extension_generation.json', 'state/post_task_evolution_counter.json', 'state/post_task_evolution_request.json', 'state/presence_bindings.json', 'state/request_wire_compatibility.json', 'state/subagent_last_delegation.json']` | all nine resolve; they and three more (`state/review_continuations/*.json`, `state/skills/*/repair_admission.json`, `state/skills/*/auth_token.json`) are pinned as SENTINELS |
| 2 | `test_unresolved_wildcard_never_certifies_an_exact_row` | base matcher: `assert not _covers("state/*", "state/state.json")` → `assert not True` | `_seg_match` requires a wildcard-bearing row segment to certify a wildcard scan segment (and the basename fallback requires a named leaf) |
| 3 | `test_every_scanned_path_has_an_inventory_row` (the doc gap) | fixed verifier against the base `docs/PERSISTENCE.md`: 17 undocumented paths — `.ouroboros_isolated_benchmark`, `logs/chat_annotations.jsonl`, `state/claudexor_rotation_provisioning.json`, `state/delegate_terminal_refresh_cursor.json`, `state/extension_generation.json`, `state/post_task_evolution_{counter,request}.json`, `state/presence_bindings.json`, `state/request_wire_compatibility.json`, `state/review_continuations` (+ `*.json`, `archived`, `archived/*`, `corrupt`, `corrupt/*.*.json`), `state/subagent_last_delegation.json`, `state/usage_attempts.quarantine.jsonl` | every one carries a row with its four decisions |

### What the fix is

Resolution before wildcards, each plane a fact the source states: module and
imported string constants (per-file first, then repo-wide when the name binds
to exactly one value), literal `Path` chains (`ARTIFACTS_DIR = Path("task_results") / "artifacts"`),
f-string text (`f"{pid}-{uuid}.json"` → `*-*.json`), the data-relative prefix
a helper returns (fixed point, file-local first then repo-wide-if-unambiguous,
so `skill_state_dir` → `skill_state_dir_path` resolves and the two same-named
`managed_runtime_root`s each resolve to their own runtime root), and one-scope
local flow including `for x in <resolved>.iterdir()/.glob(...)`. Population
124 → 271: the old anchor counted SPELLINGS, most of them family wildcards
hiding exact files, not entities.

Two audited tables carry what resolution cannot reach, both asserted rather
than assumed:

- `SUBROOT_ALIASES` (3 entries, was 4 — the observability `blobs/*`,
  `calls/*`, `calls/*/*` guesses and the claudexor `cache/*` guess all became
  dead and were removed, `auth_token.json`, `jobs/*`, `jobs/*/*` added): the
  callee receives its parent directory as a
  PARAMETER, so a call-site audit is the only evidence. Both new aliases were
  audited to every caller: `mint_skill_token` is called with
  `skill_state_dir(...)` at `extension_process_runner.py:291` and through
  `PluginAPI._state_dir`, and every `_PluginAPIConfig` in
  `extension_loader.py` (:477, :581) is built with
  `skill_state_dir[_path](drive_root, skill.name)`.
- `UNRESOLVED_SPELLINGS` (4 entries, asserted by EQUALITY): `state/*`,
  `logs/*`, `skills/*`, `skills/*/*` — parameterized readers/writers over
  planes that DO carry rows (the two stop flags, the usage ledger's own file
  names, the bounded log tail and rotation helper, skill payload roots from a
  validated relpath). A new unresolvable spelling now fails the suite until it
  is named in a constant or audited in; a spelling that becomes resolvable
  fails until it is removed from the table.

Resolution also retired `ROW_SCAN_EXEMPT_PRIMARY`: both rows that needed it
(the external claudexord daemon's directory, the benchmark sentinel) are
reached by in-tree path expressions now — Ouroboros creates and appends the
daemon directory and reads the sentinel — so the list was dead, and a dead
exemption is the same vacuous-contract defect in miniature.

### Dispositions

**Fixed — inventory rows added, each derived by reading the owner (not the
plan):**

| Entity | Owner read | The four decisions in one line |
|---|---|---|
| `state/presence_bindings.json` | `presence_bindings.py` | own `schema_version: 1` typed-refusal on mismatch; `update_json_locked`; overwrite bounded by owner links; reset loses every transport→behavior link |
| `state/request_wire_compatibility.json` | `request_wire_contract.py` | own `schema_version: 1`, other versions read as empty; TTL 14 days on read; reset re-learns from one fresh provider refusal |
| `state/claudexor_rotation_provisioning.json` | `claudexor_daemon.py` | no stamp (disclosure receipt, never read back); last patching reconcile wins; reset loses the receipt only, provisioning is idempotent |
| `state/delegate_terminal_refresh_cursor.json` | `delegate_terminal.py` | no stamp, self-grounding offset; `deferred` capped at 500; reset re-grounds at 0 and replays paced by the 5 MB per-tick cap |
| `state/extension_generation.json` | `extension_reconcile_queue.py` | own `schema_version: 1`; write-if-changed; absent marker is fail-closed no-evidence, workers keep their spawn-time set |
| `state/post_task_evolution_{request,counter}.json` | `post_task_evolution.py` | request own `schema_version: 1` + one-shot consume, counter unstamped single `n`; reset drops one promotion signal and restarts the cadence — the durable backstop stays the `evolution_owner_stopped` flag |
| `state/subagent_last_delegation.json` | `subagents.py` | no stamp (disclosure projection); last run wins; reset shows absence |
| `state/review_continuations/<task>.json` + `archived/`, `corrupt/` | `task_continuation.py` | typed dataclass with an ownership check, corrupt files quarantined not migrated; retire at 7 days settled-and-unresumed, quarantines unbounded-accepted; reset forces review to be re-run |
| `state/skills/<name>/auth_token.json` | `extension_plugin_api.py`, `gateway/host_service.py` | no stamp — `content_hash` IS the staleness contract; 0600, rotate on hash change, transient hash failure never rotates; reset de-authorizes live companions until respawn |
| `state/skills/<name>/repair_admission.json` | `skill_repair_admission.py` | own `schema_version: 1`; newest admission owns the record; reset refuses repair writes (fail-closed by design) |
| `logs/chat_annotations.jsonl` | `project_dialogue.py` | `type: chat_annotation` keyed by `client_message_id`; self-compacting at 800 KB to messages still in the chat chain (live + 3 newest archives), NOT rotated into `archive/`; reset falls back to plain chat rows and loses one pending picker token (#198) |
| `state/usage_attempts.quarantine.jsonl`, `state/usage_attempts.lock` | `usage_ledger.py`, `usage_compaction.py` | already governed by the ledger row — the row named them as bare `.quarantine.jsonl`/`lock` labels, which matched nothing; now spelled in full |

**Fixed — entities the tightened verifier and the owner sweep surfaced that
were in NO plan or review note (item d of the lane):**

| Entity | Owner | Why it matters |
|---|---|---|
| `.ouroboros_isolated_benchmark` | written by `devtools/benchmarks/**` launchers, read by `supervisor/state.py` and `agent_startup_checks.py` | the only marker that makes a data root declare itself synthetic; deleting it turns rotation and the benchmark carve-outs back on inside a throwaway root |
| `logs/tasks/task_<id>.txt` | `ouroboros/utils.py` log sanitization | spilled full text of oversized task prompts; **unbounded → candidate**: no retention plane names `logs/tasks/` |
| `state/skills/<name>/chat_id_counter.json` | `gateway/host_service.py` | A2A chat-id allocation descending from `A2A_CHAT_ID_MAX`; deleting it restarts allocation at the top and a fresh room can reuse an id already in history |
| `state/skills/<name>/jobs/<job>/` (`assets/`, `output/`, `tmp/`) | `extension_plugin_api.py::skill_job_dir` | extension job workspaces; **unbounded → candidate**: nothing sweeps them, only the gateway's local skill delete removes them with the state dir |

**Disclosed, not fixed (residuals of this lane):**

1. **A directory row still covers undocumented children.** A pattern that is a
   strict prefix of a scanned path covers it without spelling `**`, so a NEW
   file under an inventoried family directory (e.g. `state/skills/<name>/`)
   would not be demanded by the forward check. Tightening it is a bigger
   change than this lane's defect: it also requires resolving the `TOP_LEVEL`
   leading-literal heuristic's non-data-relative hits (`cache`, `cache/pip`,
   `cache/npm`, `tmp` reached through a `.ouroboros_env` root passed as a
   parameter), which are nominally covered by the `.ouroboros_env` row today.
   The three per-skill files this lane added were therefore found by READING
   the owners, not demanded by the verifier.
2. **Two entities are inventoried as `unbounded → candidate`** (`logs/tasks/`,
   `state/skills/<name>/jobs/`) with no fix and no CPL4-Cn number: adding a
   retention plane is a behaviour change with owner-visible effects (deleting
   an extension's assets, deleting spilled owner text) and belongs to a
   decision, not to a verifier lane.
3. **`state_dir` joined `DATA_ROOT_MARKERS`.** It admits any chain rooted at a
   parameter or attribute whose name contains `state_dir`; that is how the
   per-skill token and jobs planes became visible, and the equality-asserted
   residual table is what keeps a false positive from passing silently.
4. **The scan is ~13 s** (constant maps, a prefix fixed point of at most six
   rounds, one memoized scope walk per resolver) against ~2 s before — the
   same order as `tests/test_architecture_facts.py`'s heaviest case in this
   tree, and paid once per session by `functools.lru_cache`.

**Rejected with evidence:**

1. **Resolving a bare name or attribute against a repo-wide helper of the same
   name — rejected as unsound, after it produced a false plane.** An early
   revision resolved `self._state_dir / "jobs" / <job>` through
   `workspace_executor._state_dir()` and claimed
   `state/workspace_executor_processes/jobs/*`, a directory nothing writes.
   Name and attribute prefixes are now file-local facts; only CALLS resolve
   repo-wide, and only when the name binds to one path tree-wide. The false
   pair is gone from the population.
2. **Keeping the observability `blobs/*` / `calls/*` aliases — rejected.**
   Resolution reaches `observability/{blobs,calls,salvaged}/…` directly;
   removal was verified by deleting each alias and diffing the population
   (no change). An alias that changes nothing is a human promise about a fact
   the scan already has.
3. **Inventing a `state/*`-shaped family row so the residuals would "match" —
   rejected.** `state/*` is not an entity; a row for it would restore exactly
   the vacuous coverage this lane removed. The audited equality table says the
   same thing honestly and fails on the next new spelling.

### Gate evidence

This host, isolated env roots per invocation (`OUROBOROS_APP_ROOT`/`_REPO_DIR`/
`_DATA_DIR`/`_SETTINGS_PATH` + private `TMPDIR` under a fresh `mktemp -d`),
venv python 3.10.12, `-p no:cacheprovider`; `git rev-parse HEAD` verified
unmoved after every pytest run; author and committer
`Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>`; no push.
Each gate its own command with its rc printed:

- `tests/test_persistence_inventory.py` 5 passed, rc 0 (2 of the 5 are this
  lane's new pins);
- `tests/test_persistence_inventory.py tests/test_architecture_facts.py
  tests/test_docs_sync.py` 34 passed, rc 0 (the architecture-facts suite owns
  the PERSISTENCE.md row parser, its one-row-per-table-line completeness check
  and the writer-span resolution the new rows had to satisfy);
- `tests/test_doc_context.py tests/test_skip_tests_doc_only.py` included in an
  earlier 71-passed run, rc 0;
- `ruff check . --select F` rc 0; `scripts/check_domains.py` rc 0 (no module
  row moved); `scripts/regenerate_inventories.py --check` rc 0;
  `scripts/regenerate_size_ratchet.py --check` rc 0 (the suite grew 303 → 701
  lines, inside its band with no manifest change);
  `scripts/v7next_adoption.py` rc 0 and `--release` rc 0;
  `git diff --check` rc 0 and `git diff --check 9faccf31..HEAD` rc 0.

The CI-shape battery and the `-m serial` pass are not re-run here: the lane's
surfaces are one test module and two documents, and no runtime code changed.

## From the stage-2 close-out, lane ledger2 (base bf8b6549)

Four lens findings on the ledger-validator lane, plus one inherited citation
from the ui lane. Base `bf8b6549`, the v7next integration tip with all five
stage-2 fix lanes merged. Nothing runtime changed: the surfaces are
`scripts/v7next_adoption.py`, its test module, `ADOPTION_v7next.md` and three
documents under `docs/v7next/`.

### Red-first table

| # | item | red-first evidence (pre-fix shape) | green |
|---|---|---|---|
| 3 | five hook-resolution messages prefixed `release:` although hook resolution runs for every `done` row in BOTH modes, and a docstring claiming «Outside --release hooks stay free prose» | the new parametrized pin drives four hook shapes (prose-only, missing file, `tests/../` escape, bogus `::nodeid`) through the DEFAULT mode and asserts no message claims the release bar → **4 failed, 19 passed**, the failure text being `release: D03 hook names tests/test_smoke.py::…` | 23 passed (`tests/test_v7next_adoption.py`), validator rc 0 in both modes |
| 1 | the frozen-inventory rationale stated false git facts in two places | **not pinnable, by the design the finding asks to keep**: the checker deliberately reads one file and spawns no subprocess, so no pytest assertion can hold the comment to `git log`. Re-derived instead with read-only `git log -1 --format='%h %p'`, `git rev-list --first-parent` and `git merge-base --is-ancestor`, and the derivation is written into the commit message | validator rc 0 in both modes; `scripts/v7next_adoption.py` unchanged in behaviour |
| 2 | two WINWAVE run rows read «green \| green \| green \| re-prove holds» for reruns, and one row still read «pending» | **not pinnable in pytest**: the facts live in GitHub Actions, not in the tree. Re-read read-only against the public API (`runs/<id>` → `run_attempt`, `conclusion`; `.../attempts/<n>/jobs` → per-job conclusions) | the run table now carries each row's attempt count, and the attempt-1 outcome of every rerun |
| 4 | three citations naming lines they inherited from the review | **deliberately not pinned by a line assert** — a test that pins `chat.js` line numbers is the same rot class with a red light on it. Re-derived with `grep -n` and the live-path citation now leads with the symbol, which is what survives the next edit | `grep -n` agrees on all three: `chat.js:2399-2430`, `chat.js:2992`, `api_types.js:287/:290` |

Only item 3 changes program text, and it is the only one with a pin. Items 1,
2 and 4 are factual corrections to prose; each says above why a pytest pin
would be either impossible or actively harmful, rather than claiming a pin it
does not have.

### Item 1 — the train inventory's rationale (fixed, and the reason replaced)

`scripts/v7next_adoption.py` and the sync #3 ledger section both said «of the
three recorded sync merges only `f4abe0a5` has an upstream commit as its
literal second parent». Derived facts:

| merge | parents | first-parent line of this branch |
|---|---|---|
| `20850191` (absorb, sync #1) | `5187fcdc` `8d13373b` | no |
| `b9ceed6e` (absorb, sync #2) | `3e4a6181` `f3fbfdbb` | no |
| `f4abe0a5` (absorb, sync #3) | `43dcc1d2` `a76961de` | yes |
| `0aa74e9f` (lane integration) | `a12c873c` `816e7b82` | yes — carries `20850191` |
| `0f9a8daf` (lane integration) | `ad8506ef` `4c32691e` | yes — carries `b9ceed6e` |

So **all three** absorb merges take the upstream tip as their literal second
parent; the false half was «only `f4abe0a5`». What holds is the first-parent
shape: two of the three absorb merges were made on lane lines and reached
mainline as the second-parent side of a lane-integration merge over a campaign
commit. Three further corrections in the same paragraph:

1. **The re-tie `f61ea3c2` is not the cause and cannot be.** It is an ancestor
   of all three syncs (`git merge-base --is-ancestor f61ea3c2 20850191` and the
   same for `b9ceed6e`, `0aa74e9f`, `0f9a8daf` all true; the reverse false), so
   it predates them. The clause is removed rather than reworded — no causal
   claim replaces it, because none is derivable from the graph alone.
2. **The widened rule is now quantified.** «Second parent descends from a
   recorded upstream tip» matches 35 merges for `8d13373b`, 15 for `f3fbfdbb`
   and 6 for `a76961de` on this tree, which is the honest form of «every lane
   merge».
3. **The example was not a merge.** `8fb08d44` has one parent; the C6 lane
   merge is `9faccf31`, whose second parent it is.

Disposition on the inventory itself: `REQUIRED_TRAINS` is **unchanged**. Sync
#1 is still recorded by its mainline carrier `0aa74e9f` while syncs #2 and #3
are recorded by their absorb merge — an asymmetry that is now stated in the
comment instead of hidden, and the `TRAIN-F6` manifest row names both
`20850191` and `0aa74e9f` so the absorb merge is not lost. Repointing the tuple
at `20850191` was rejected: it would move an enforced SHA for no gain, and the
row records both facts as text either way.

### Item 2 — rerun-greens are not first-attempt greens (fixed)

Read read-only from the public GitHub API, unauthenticated (`gh` is installed
on this host but not logged in, and nobody logged in for this read):

| run | SHA | run_attempt | conclusion | attempt-1 failures |
|---|---|---|---|---|
| 33569841899 | `8b27b507` | 1 | success | — |
| 33570328266 | `285ab66d` | 1 | success | — |
| 33571681398 | `9238cc2d` | 1 | success | — |
| 33572515529 | `c0029d45` | 1 | success | — |
| 33574822693 | `d21806d8` | 1 | **failure** | `full-test (windows-latest)`, never rerun |
| 33579445704 | `1072a317` | 2 | success | `full-test (windows-latest)` |
| 33624546416 | `ac17fa03` | 2 | success | `full-test (windows-latest)` |
| 33626834806 | `43dcc1d2` | 2 | success | `system-e2e-mock` only |
| 33644668074 | `f4abe0a5` | 1 | success | — |

Per row, as the finding asks:

- **33579445704 (`1072a317`)** — attempt 1 red on `full-test
  (windows-latest)`, everything else green; attempt 2 green. Named cause,
  **operator-read**: `tests/test_phase3c_observability_gc` on its copy-back
  step (intermittent) and `tests/test_preflight_runner` on an xdist worker
  timeout. **No code fix followed**, so the class stays *intermittent,
  unrooted* and is carried as the **O3** question to the owner.
- **33624546416 (`ac17fa03`)** — attempt 1 red on `full-test
  (windows-latest)`, everything else green; attempt 2 green. Cause: the
  session-engine horizon read 301 for a deadline 300 s away on the coarse
  Windows clock. **A code fix followed**: `43dcc1d2`, whose own message names
  this run id and this cause, in `tests/test_review_agent_session_route.py`.
  Rooted and closed — this is the one cause corroborated inside the repo
  rather than only operator-read.
- **33574822693 (`d21806d8`)** — the row said «pending at the time of
  writing». Verdict: red on the Windows `full-test` leg and never rerun, so it
  is **not a re-prove**. No class is claimed for it: the failing subtests are
  not attributable from here.
- **33626834806 (`43dcc1d2`)** and **33644668074 (`f4abe0a5`)** are the two
  later first-attempt greens (the former on its three `full-test` legs, its
  separate `system-e2e-mock` job having been rerun; the latter on every job),
  and R-WINWAVE's hook now names them beside the four early ones.

**Honesty boundary on this item.** Job LOGS are not readable this way — the
API answers 403 «Must have admin rights to Repository» — and the check-run
annotations carry only «Process completed with exit code 1». So the *attempt
structure* is re-derived fact, while the *failing test names* for 1072a317
remain operator-read facts and are labelled as such in the registry. The
d21806d8 cause is recorded as unattributable rather than guessed.

Also corrected while here: `ADOPTION_v7next.md` R-WINWAVE still said run
33644668074 «is dispatched and its verdict has not arrived». It had been read
green on 2026-09-02 15:00Z and `7b6d3613` recorded that in the registry without
updating the manifest row.

### Item 3 — `hook:` instead of `release:` (fixed, pinned)

Hook resolution moved to «every `done` row, both modes» in an earlier round,
but its five messages kept the `release:` prefix and the docstring kept the
old sentence. Both now match the code: the prefix is `hook:`, the two genuine
release-bar messages (`still pending-decision`, `status != done`) keep
`release:`, and the docstring says what gates the resolution — the `done` cell,
not the mode. The comment at the call site cited the manifest's Notes for the
rule; the Notes did not actually state it, so the sentence was added there
instead of dropping the citation. The fifth message (an unparseable hook file)
is renamed with the others but not driven by the pin — it needs a planted
syntax-error file, and the four drivable shapes already fix the prefix class.

### Item 4 — inherited line citations (fixed)

`chat.js:2412-2443` → **`chat.js:2399-2430`** (the function is 32 lines,
`routeSubagentTerminalToCard` at `:2399`, closing brace at `:2430`);
`chat.js:3005` → **`chat.js:2992`** (the reload path's call);
`api_types.js:288/:291` → **`:287/:290`** (the `@property` lines for
`accounted_upper_bound_usd[_with_children]`; `:288/:291` are the description
lines beneath them, and were wrong against the frozen base as well — there the
honest names sat at `:284`/`:287`).

The live-path citation now leads with `chat.js::routeSubagentTerminalToCard`,
so the symbol survives the next line shift. No line-number pin was added: a
test asserting `chat.js:2399` would go red on the next unrelated edit to that
file, which is the defect this finding is about, not a fix for it.

**Checked and NOT changed:** the remaining `file:line` references in
`docs/v7next/ABI3_GATEWAY_ALIAS_INVENTORY.md` (`gateway/tasks.py:227`,
`supervisor/events_task_done.py:256`, `agent_task_pipeline.py:842/:853`,
`gateway/history.py:849/:381/:401`, `api_types.js:333/:370/:389/:410`,
`api_types.js:1017-1018`) no longer point at their subjects on this tree, but
that document declares itself «Frozen BEFORE the first removal (lane D3, base
`29e2b045`)», so those citations are as-of that base by design and rewriting
them would destroy the evidence they exist to carry. Two of them are worth a
reader's note all the same: the `telegram_chat_id` and ui-preferences JSDoc
typedefs the document lists as HOT-DEFERRED were removed by `71e1f13f` (the
F3.3 residual sweep) later in this campaign, so the deferral has since been
discharged. Updating that disposition belongs to the ABI-3 row, not to this
close-out, and is disclosed here rather than done.

### Gates

Run as separate commands with the return code printed:

- targeted suites `tests/test_v7next_adoption.py` +
  `tests/test_legacy_timeout_retirement.py` (it scans `ADOPTION_v7next.md`) +
  `tests/test_gateway_abi3_removals.py` + `tests/test_comma_list_remnant_sweep.py`
  — **63 passed, rc 0**;
## From the stage-2 close-out, lane persistdocs (base bf8b6549)

Six lens findings against the two verify pairs the stage-2 wave had just
rewritten: three on the persistence inventory's scanner and its two matching
directions, three on the architecture document's own truth. Every behaviour
change below is pinned RED on the pre-fix shape first, then green.

### Red-first table

| # | Sev | Pin | Red on the pre-fix shape | Green after |
| --- | --- | --- | --- | --- |
| 1 | HIGH | `test_persistence_inventory.py::test_parent_of_a_helper_returned_path_is_a_named_root` | `AssertionError: assert 'state/reviewer_slot_api_fallback.json' in frozenset({...})` — the path is in the population under no spelling at all | resolved via `_base_prefix`, and `test_every_scanned_path_has_an_inventory_row` then went red on the missing row (`['state/reviewer_slot_api_fallback.json']`) until the PERSISTENCE.md §2 row landed |
| 2 | MEDIUM | `test_persistence_inventory.py::test_a_fabricated_inventory_row_is_caught_as_stale` | `TypeError: _covers() got an unexpected keyword argument 'scan_prefix_ok'` — the directional matcher the backward check needs did not exist; with the old matcher the fabricated row `state/does_not_exist.json` was silently absorbed by the bare `state` token | directional `_match`/`_covers`; the mutant row is reported and the honest document adds nothing beyond its one audited exemption |
| 3 | LOW | `test_persistence_inventory.py::test_no_parameter_rooted_spelling_lands_at_the_data_ROOT` | `AssertionError: parameter-rooted spellings left at the data root: ['__extension_imports/*-*', '__extension_imports/*-*/skill', '__extension_imports/*-*/skill/*', 'uninstalled.json']` | four `SUBROOT_ALIASES` entries with their call-site audits |
| 4 | HIGH | `test_docs_sync.py::test_architecture_component_map_covers_every_live_runtime_module` | with the token matcher in place and the doc unchanged: `names no owner for these live modules: ['ouroboros/tools/browser.py', 'ouroboros/tools/health.py', 'ouroboros/tools/vision.py']` | three component-map rows written from each module's own source |
| 5 | MEDIUM | `test_docs_sync.py::test_architecture_does_not_claim_usage_response_is_the_only_usage_reader` | all three predicates red against `HEAD:docs/ARCHITECTURE.md` (absolute claim present, normalizer claim absent, residual undisclosed) | reworded row + pin |
| 6 | LOW | `test_docs_sync.py::test_settings_docs_name_every_key_owner_and_what_startup_persists` (clauses 4) | both predicates red against `HEAD:docs/ARCHITECTURE.md` (`retired \`OUROBOROS_CONTEXT_MODE\`` present, mechanism phrasing absent) | reworded startup sentence + pin |

Findings 5 and 6 are prose-only, so their red evidence is the pin predicates
evaluated against `git show HEAD:docs/ARCHITECTURE.md` rather than a pytest
run: reverting the file in the worktree to drive pytest twice produced a run
that did not finish in the time allowed, both times — the cause (found by the
close-out lens) is pytest's assertion introspection over the ~810 KB flattened
document on a failing `not in`: `--assert=plain` fails the same pin in 0.13 s,
so the pins are not flaky, only slow to EXPLAIN a red. The predicates are the
exact expressions the committed pins assert.

### Dispositions

1. **`state/reviewer_slot_api_fallback.json` — fixed (HIGH).**
   `reviewer_slot_config._record_api_fallback_substitution` (:833) writes
   `_last_execution_path().parent / "reviewer_slot_api_fallback.json"`. The
   chain base was an unresolved `.parent` attribute, so the file entered the
   population under no name: a live durable disclosure record invisible to the
   forward check and absent from the inventory. `_base_prefix` now reads
   `.parent` as an operation on an already-named path (path minus its leaf; a
   single-segment path has no named parent — that is the data root — and still
   yields nothing), which is a fact the source states, not an alias. Chosen
   over an audited alias precisely because the fact is derivable: an alias
   would have been a human promise where the AST already has the answer, and
   it would not have caught the next writer of this shape. The scan gained ten
   spellings, nine of them already documented (`state/cx/*`, `projects/*`).
   The new PERSISTENCE.md §2 row names the writer, the atomic-text write with
   its swallowed `OSError`, `schema_version: none` with the reason (never read
   back — verified by grep: the only other mentions of the name are the
   writer, its `__all__`, and the save-time warning
   `reviewer_slot_api_fallback_warning`, which re-derives its text from the
   config and not from this file), retention (one object, last substitution
   wins) and reset (delete with `state/`).
2. **The backward check — fixed (MEDIUM), and it cost three rows.**
   `_match` accepted an exhausted SCAN path as "named by a deeper row" in
   BOTH directions. Since the population also holds bare top-level tokens
   (`state`, `logs`, `memory`, `archive`, `skills`, `uploads` …), `state`
   certified any invented `state/<name>.json` row: the no-stale-rows direction
   proved nothing. The prefix rule is now directional — the forward check
   keeps it (`state` genuinely IS answered by the deeper rows under it), the
   backward check drops it, and a trailing `**` still matches zero segments so
   a `dir/**` row needs the row's own depth, not one more. Three rows fell out
   of the stricter direction and each was disposed of separately, NOT by one
   exemption list:
   - `logs/server.log` (+`.1..3`) and `logs/launcher.log` — real writers the
     scan could not see: both stdlib `RotatingFileHandler`s are built at
     module import time from `_log_dir = DATA_DIR / "logs"` (server.py:122,
     launcher.py:123), and module scope was the one scope the resolver ran
     with an EMPTY locals map. Module scope now gets its own local flow, the
     same fact function scope already had. +2 spellings, both matching the
     existing row.
   - `logs/tasks/task_<id>.txt` — a real writer whose root arrives as a
     parameter: `sanitize_task_for_event(task, drive_logs, …)` writes
     `drive_logs / "tasks" / f"task_{id}.txt"` (utils.py:1160). Adding
     `drive_logs` to `DATA_ROOT_MARKERS` was rejected: it would have rooted
     `drive_logs / "events.jsonl"` and its siblings AT the data root — the
     exact mis-rooting defect finding 3 is about — and then needed three
     relocating aliases to undo. Instead one audited `PARAM_SUBROOTS` entry
     names the plane up front (`drive_logs` -> `logs`); every binding in the
     tree is `env.drive_path("logs")` (agent.py:442/795,
     agent_startup_checks.py:860) or `ctx.drive_logs`/`drive_root / "logs"`
     (commit_gate.py:907) threaded down unchanged, and nothing else binds the
     name. +1 spelling.
   - `state/project_source_locks/` — the only genuine orphan: the row itself
     reads "none in this tree — orphan plane seen in live layouts (removed
     feature leftover)", and `rg project_source_locks` finds the string in no
     `.py` file. Exempted BY NAME in `STALE_ROW_EXEMPTIONS`, asserted by
     equality, so a row that acquires a writer surfaces too. The
     `test_every_inventory_row_is_still_real` docstring claimed resolution had
     retired the exemption list; that claim is replaced by what is now true.
3. **Mis-rooted per-skill spellings — fixed (LOW).** `state_dir` matches
   `DATA_ROOT_MARKERS`, so a chain hanging off that parameter was correctly
   seen as data-relative and wrongly placed at the data ROOT.
   `uninstalled.json` (skill_uninstall_state.py:65, `state_dir` bound by the
   sweep's own `drive_root / "state" / "skills"` listing) and the
   `__extension_imports/*-*` family (extension_import_staging.py:84-88, whose
   ONE caller extension_loader.py:649 passes the
   `state_dir = skill_state_dir(drive_root, skill.name)` bound at :581) passed
   only through the basename and bare-token fallbacks — covered by accident,
   documented nowhere they actually live. Four `SUBROOT_ALIASES` entries with
   the call-site audit written beside them, in the shape
   `auth_token.json`/`jobs/*` already use. Both families have a second,
   already-resolved spelling through `skill_state_dir(...)`, which is what
   makes the aliases checkable rather than asserted: the tombstone collapses
   onto that spelling (pin 284 -> 283) and the staged-import paths sit under
   the `state/skills/*/__extension_imports` the sweep resolves.
4. **The component-map substring hole — fixed (HIGH).** The pin tested
   `PurePosixPath(path).name in arch`, so a basename buried inside a longer
   file name counted as a row: `browser.py` was "documented" by
   `test_s3_task_control_browser.py` (line 3534), `health.py` by
   `extension_health.py`/`worker_health.py`/`context_health.py`, `vision.py`
   by `delegate_supervision.py`. All three modules had no mention of any kind.
   The replacement states the boundary as "not a file-name character"
   (`(?<![\w.\-])name(?!\w)`) rather than an allow-list of delimiters: the
   first draft allowed only start/space/backtick/slash and reported
   `ouroboros/marketplace/clawhub.py` as missing, because the map introduces
   it after an opening parenthesis (`(clawhub.py registry client`). A lexical
   boundary on a file name in prose is not a semantic gate (BIBLE P5): it
   replaces a weaker lexical test in a documentation pin, and no runtime
   decision reads it. The three rows were written from each module's own
   source, not from the finding text: the Playwright per-ToolContext session
   with thread affinity, its engine/bundle resolution and out-of-band cleanup
   of retired generations, plus the browser-side trust boundary
   (`_is_subagent_blocked_browser_url`, the control-plane loopback ports, the
   private/link-local/metadata address and resolving-hostname denials, the
   workspace-only `file://` carve, and the route/in-page-script guards against
   self-lowering context or safety mode, mutative and post-task-evolution
   toggles, owner settings self-elevation and owner skill self-attestation);
   the three VLM tools plus `attach_local_image_to_context` with the vision
   slot resolution, the deadline-aware wait, the local-file trust boundary its
   `media.py` siblings reuse and the 20 MB / 6 MB / 1600 px / magic-byte
   bounds; and the read-only P7 report whose findings — including the
   size-ratchet validator's — are warnings against the enforcing CI lane.
5. **"The only reader of a provider's usage block" — fixed (MEDIUM).** False:
   `llm_openai_compatible.py:285`, `llm_anthropic.py:318`,
   `llm_local.py:271`, `local_model.py:678` and `tools/vision.py:192` all read
   a raw `usage` dict for their own envelopes. The replacement was derived
   from an exhaustive scan of the class (`rg 'get\("usage"\)|\["usage"\]'` over
   `ouroboros/` and `supervisor/`), not from the fixed sentence, so one false
   absolute is not swapped for another: the row now claims the one
   NORMALIZER of that block into reported token counts and a provider-declared
   cost, names its exactly two importers (`usage_accounting.py`,
   `loop_llm_call.py` — verified by `rg 'from ouroboros._usage_response'`),
   and DISCLOSES the residual in the same sentence rather than leaving a
   consolidation invitation for the next author.
6. **`OUROBOROS_CONTEXT_MODE` called retired — fixed (LOW).** The startup
   sentence (:19) called the key retired while the settings table (:3212)
   documents it as the live owner-selected context horizon; the module row
   (:87) already had it right ("the RETIRED persistent auto-Low context
   state"). The sentence now says what retired is the mechanism, names the
   pair as its compat residue and `OUROBOROS_CONTEXT_MODE_AUTO_LOW` as its
   provenance tombstone. Pinned as a fourth clause of the existing
   settings-docs test rather than as a new test, because that test already
   owns the startup-persistence claim about the same seam.

### Rejected

1. **An audited alias for `reviewer_slot_api_fallback.json` instead of
   teaching the resolver `.parent` — rejected.** The finding offered either.
   The parent of a resolved path is derivable from the AST, and an alias would
   have bought one entity while leaving every future `<helper>().parent /
   "name"` writer invisible. Aliases stay for roots that arrive as parameters,
   where only a call-site audit can answer.
2. **One exemption list for all three rows the strict backward check exposed —
   rejected.** Two of the three had real writers; excusing them would have
   re-created, in the exemption table, exactly the vacuity the fix removed. A
   writer must be SEEN, not excused, and only the documented orphan is
   exempted.
3. **Adding `drive_logs` to `DATA_ROOT_MARKERS` — rejected.** It resolves
   `logs/tasks/*` at the price of mis-rooting `events.jsonl`, `tools.jsonl`
   and their siblings at the data root, i.e. manufacturing three new instances
   of finding 3 and then aliasing them back.
4. **An allow-list of delimiters for the basename boundary — rejected after
   measurement.** start/space/backtick/slash reported
   `ouroboros/marketplace/clawhub.py` as undocumented although the map names
   it; the negative-lookbehind form is the same intent stated over the right
   alphabet.
5. **Rewriting the `logs/tasks/` retention disclosure — out of scope.** The
   row already discloses that no retention sweep names the directory
   (recorded by the stage-2 wave). This lane made the writer visible; whether
   that plane gets a sweep is a separate owner decision.

### Gate evidence

This host, isolated env roots per invocation (`OUROBOROS_APP_ROOT`/`_REPO_DIR`/
`_DATA_DIR`/`_SETTINGS_PATH` + a private `TMPDIR` under a fresh `mktemp -d`),
venv python, `-p no:cacheprovider -o addopts=""`, at most `-n 6`;
`git rev-parse HEAD` verified unmoved after every pytest run; author and
committer `Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>`;
five single-intent commits; no push. Each gate its own command with its rc
printed:

- `tests/test_persistence_inventory.py tests/test_docs_sync.py
  tests/test_architecture_facts.py tests/test_reviewer_slot_config.py
  tests/test_size_ratchet_ci_shape.py tests/test_domain_manifest.py`
  128 passed, rc 0;
- `ruff check . --select F` rc 0;
- `scripts/check_domains.py` rc 0;
- `scripts/regenerate_inventories.py --check` rc 0;
- `scripts/regenerate_size_ratchet.py --check` rc 0;
- `scripts/v7next_adoption.py` rc 0 and `--release` rc 0;
- `git diff --check` rc 0 and `git diff --check f3fbfdbb..HEAD` rc 0.

`git rev-parse HEAD` was read after every pytest invocation and never moved.
The full non-serial battery, the `-m serial` pass and the CI-shape checks are
NOT re-run here, and the omission is deliberate rather than forgotten: the only
program text this lane changes is five f-string prefixes and two docstrings in
`scripts/v7next_adoption.py`, a checker no runtime path imports, plus one test
module. Everything else is prose. A release candidate still owes its own full
battery and its own 3-OS matrix — and, per item 2, that matrix is now the place
where the 1072a317 intermittent class either recurs or does not.
- `scripts/v7next_adoption.py` rc 0 and `scripts/v7next_adoption.py --release`
  rc 0;
- `git diff --check` rc 0 and `git diff --check f3fbfdbb..HEAD` rc 0.

No runtime module changed in this lane: the diff is one test module, one
architecture document, one persistence inventory and this ledger. The
CI-shape battery and the `-m serial` pass are therefore not re-run here.
## From the stage-2 close-out, lane smalls2 (base bf8b6549)

Four read-only lens findings on the stage-2 code-smalls lane (2 MEDIUM, 2 LOW),
one single-intent commit each, every behaviour change pinned red-first against
the pre-fix shape. No protected file is touched. `ouroboros/config.py` ends at
1000 lines — AT the band floor, not over it, so no band rationale is due.

| # | item | red-first evidence (pre-fix shape) | green |
|---|---|---|---|
| 1 | item 7's route pin keyed the adapter count on `binding_resolves`, a predicate that is FALSE for the binding-RAISES case of the cognitive scenario | with `_build_builtin_target_binding` forced to raise (the Windows outcome the retired `os.name` stand-in encoded), the previous pin fails `assert 3 == 0` on the cognitive scenario | 2 passed on the real tree AND on that same mutant; still fails `assert 0 == 3` when the resolving-binding branch is mutated to answer natively, so the derived pin is not vacuous |
| 2 | item 9's notice told every non-comma dropped set "no replacement setting: what they used to configure is fixed behavior in this release" — false for a retired key the table gives a successor | both new pins fail against the VERBATIM `bf8b6549` `ouroboros/config.py`: the wall-clock pair is answered with the "fixed behavior" sentence (captured log quoted in the run), and the neutral shape has no wording of its own | 25 passed (`test_settings_read_seam`), 41 passed with `test_legacy_timeout_retirement` + `test_config_extraction`, 129 passed across the seven retirement-adjacent suites |
| 3 | item 8's `_pinned_upstream` caught only a nonzero exit code, so a host without `git` failed COLLECTION instead of the promised skip | on a PATH with no `git`, the pre-fix file errors out with `FileNotFoundError: [Errno 2] ... 'git'` and runs ZERO tests | same no-git PATH: 43 passed + exactly 7 honest corpus skips naming file and base SHA; the tree with `git`: 50 passed |
| 4 | item 6's docstring/ledger claim about WHERE the RuntimeWarning surfaced was unreproduced | n/a — this is a truth claim about an observation, not a behaviour change; no pin was weakened | 48 passed (`test_broadcast_ws` + `test_extensions_api`) |

Dispositions beyond the plain fix:

1. **Item 1 — the route follows the returned SHAPE, not the binding predicate
   (supersedes the code-smalls disposition 6).** That entry reads
   "`registry_core` reaches the light repo-mutation block only when
   `_build_builtin_target_binding` RETURNED a binding (legacy text → adapter, 3
   calls); when it RAISES, the except arm answers from the resolution layer
   with a native `ToolResult` (0 calls)". The first half is right; the second is
   only half the except arm. `tool_resolution._light_binding_failure_result`
   types the ROOT redirect as a native `ToolResult` (0 adapter calls) and hands
   the COGNITIVE redirect back as legacy TEXT — 3 adapter calls, exactly like
   the resolving-binding branch. So `binding_resolves` alone inherited the
   retired `os.name` stand-in's second, unstated assumption: that the cognitive
   scenario can never take the except arm. Forced onto it, that scenario still
   costs 3 calls while the old predicate demanded 0. The test now calls the
   except arm itself and keys on `isinstance(..., ToolResult)`. The earlier
   entry is left in place as the record of what that lane concluded; this row
   supersedes it.
2. **Item 2 — a second classification INSIDE the retirement SSOT, not a new
   module.** `RETIRED_SETTING_SUCCESSORS` lives in
   `ouroboros/settings_defaults.py` next to `RETIRED_SETTING_KEYS` and
   `RETIRED_COMMA_LIST_SETTING_KEYS`, which is the same shape the ABI-10 comma
   classification already has; the notice composes one clause per
   classification, so a MIXED dropped set no longer gets a single sentence that
   is wrong for half of it. Two entries only, each stated twice over (the
   comment above the key in the retirement tuple, and the ABI-5/D04 rows in
   `docs/ARCHITECTURE.md`): the flat wall-clock pair → the activity model. Keys
   the table does not give a successor stay neutral ("removed, not honored —
   see the release notes for the surface that replaced them") rather than
   getting either claim: the observability retention knob genuinely has no
   successor setting, and the three plan-task swarm timeouts have none the
   table states PER KEY, so naming one would have been an invention.
3. **Item 2 — a MIGRATED key is not in the map, and that is pinned.**
   `OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES` has a documented successor, but
   `config._seed_review_cycles_from_legacy_passes` consumes it into
   `OUROBOROS_REVIEW_MAX_CYCLES` BEFORE the purge computes the dropped set, so
   it can never reach this notice: there is no loss to report, and an entry for
   it would promise a line nothing emits. The pin asserts its absence from the
   map, that the notice stays silent about it, and that cycles = passes + 1
   still happens.
4. **Item 2 — the successor pin derives its document from the table.** Spelling
   the retired wall-clock keys in `tests/test_settings_read_seam.py` would have
   made that file an offender of the D04 grep gate
   (`tests/test_legacy_timeout_retirement.py::test_no_runtime_or_settings_surface_still_names_either_key`,
   which allows the pair only in the retirement SSOT and its own audits). The
   pin reads `RETIRED_SETTING_SUCCESSORS` instead, which is also the right
   altitude: it is about the CLASS "a retired key whose successor the table
   states", not about one key's spelling. The gate stayed rc 0 without being
   widened. `tests/test_config_extraction.py`'s owner inventory gained the new
   symbol's row, and `docs/v7next/FACADE_INVENTORY.md` was REGENERATED for the
   one added `config.py` re-export (2222 → 2223 marked bindings) — the
   inventory `--check` was red until then, and that red was ours, not
   pre-existing.
5. **Item 3 — the class is the OSError, and only that.** `subprocess.run`
   raises `FileNotFoundError` (an `OSError`) before it can return a code, so
   the "corpus unreachable" marker the item promised was unreachable itself on
   a no-git host. The three call sites already share one reader, so the fix is
   one `except OSError: return ""`; nothing else in the file shells out.
6. **Item 4 — the mechanism REPRODUCED, so the pin and the claim both stand;
   what changed is the honesty about its determinism.** Under a CI-shaped full
   run of the pre-fix `ouroboros/gateway/ws.py` (`pytest tests/ -n 6 -W always`
   with the local marker set, instrumented so the swallowed `RuntimeError` arm
   logs the running test), the leak path was taken as
   `tests/test_extensions_api.py::test_api_extensions_index_lists_extension_skills
   (setup) | closed=True | RuntimeError: Event loop is closed`, and the warnings
   summary of that same run carried
   `tests/test_extensions_api.py::test_api_extensions_index_lists_extension_skills
   / ws.py:184: RuntimeWarning: coroutine 'broadcast_ws' was never awaited` (the
   line is in the INSTRUMENTED copy; the committed pre-fix line is the `pass`
   at f52fcec7^:ouroboros/gateway/ws.py:183) —
   the innocent-bystander attribution, exactly as claimed. It is NOT
   deterministic: a second full run under the same markers did not take that
   path, and a single-file run of `tests/test_extensions_api.py` never does.
   What IS deterministic is who leaves the finished loop in the module global —
   tracing every `ws.set_event_loop` writer across a full run gives exactly the
   three `with TestClient(server.app)` lifespan tests
   (`test_extensions_api::test_testclient_lifespan_reload_all_uses_app_state_drive_root`,
   `test_extensions_api::test_testclient_settings_hot_reload_uses_app_state_drive_root`,
   `test_onboarding_host::test_server_boot_leaves_the_settings_bytes_alone`),
   because `server.lifespan` is the only writer in the product. The docstring
   now separates the deterministic half from the flaky half, which is also the
   argument for pinning the leaking line rather than the bystander. Both
   full-run logs are operator artifacts under `/tmp` and are NOT committed; the
   copies used for them were tar copies outside this worktree, whose missing
   `.git` accounts for the 5 failed / 7 errored git-dependent tests in those
   logs (inventories, domain manifest, docs sync).

Gates at the lane tip: targeted suites rc 0 (`test_registry_core` 20 passed,
`test_settings_read_seam` 25, `test_v7next_transplant` 50, `test_broadcast_ws` +
`test_extensions_api` 48, `test_generated_inventories` 13, plus 129 across the
retirement-adjacent suites); the CI-shape battery rc 0 — 14040 passed / 4
skipped parallel (`-n 6`) and 624 passed / 19 skipped on `-m serial`, `HEAD`
unmoved before and after both; `ruff check . --select F` rc 0;
`scripts/check_domains.py` rc 0; `scripts/regenerate_inventories.py --check`
rc 0 (after the regeneration named in disposition 4);
`scripts/regenerate_size_ratchet.py --check` rc 0;
`scripts/v7next_adoption.py` rc 0 and `--release` rc 0; `git diff --check` rc 0,
`git diff --check f3fbfdbb..HEAD` rc 0 and `git diff --check bf8b6549..HEAD`
rc 0.

## From the Windows CI matrix on bf8b6549 (run 33654743857; C6 lane's first Windows execution)

windows-latest full-test red, 12 tests, three classes (packet §10 addendum): the `LockFileEx`
kernel tier is unusable as written (a mandatory byte-range lock refuses contenders the read
of the owner stamp → every wait times out) — **7.0 ships Windows on the name tier**
(`kernel_file_locks_enforced` → False on Windows; compaction refuses there, disclosed); seven
lane pins encode the POSIX eviction/fsync protocol and skip on Windows with the reason; one
regex pin accepts the path shape's typed text; `force_kill_pid` spells SIGKILL portably. The
re-enable of the Windows tier (stamp-safe byte range + a Windows-executed pin) is a
post-release item (DEFER-C6-RESIDUALS names it). NOT verified by the next matrices: the two
concurrency tests stayed red on every name-tier leg — see «From the Windows CI matrix on 35b82db0».

## From the macOS full-test 33658408570 and the platform guard (candidate follow-ups after 5ae7f357)

1. **macOS caught the executor rule of 504bb20c.** «no command hash → not ours to kill» leaked a genuine child on macOS (`ps -ww -o command=` can return nothing right after the spawn, so the record was hash-less): `test_executor_panic_cleanup_kills_durable_foreground_and_service_processes`. The kill decision now asks the question it needs — **signalable by us** — through the new platform primitive `platform_layer.pid_is_signalable` (POSIX: signal 0; EPERM and ESRCH are «not ours»; Windows: liveness, no signal probe). Red-first: `test_executor_cleanup_kills_a_hash_less_record_of_our_own_live_child` fails on the 504bb20c rule and passes now; the forged pid-1 record stays ignored. `pid_is_alive` is deliberately not used for kills: since C6 round 5.4 it reads EPERM as alive, the right answer for a lock owner and the wrong one for a kill.
2. **The platform guard refused the first shape** (`os.kill` directly in workspace_executor, abe93702 → CI 33661022574 red on every OS at `test_no_platform_specific_apis_outside_platform_layer`): the primitive moved into the platform layer, which is where the question belongs.
3. **Residual (root).** Under root every live pid is signalable, so the forged-record pin is a non-root property (`skipif(geteuid()==0)`) and a hash-less stale record naming a recycled pid is killable — exactly the v6-line exposure, which C6 never narrowed and 504bb20c narrowed for two hours. Forging a record needs write access to `data/state/`; disclosed, no product gate.
4. **Windows and the compaction suite.** On the name tier the pass refuses by design, so `tests/test_usage_compaction.py` cannot land a compaction on Windows: under an emulated name tier 32 tests fail and 23 error (not only the 12–15 fixture errors CI showed). The suite's `data_root` fixture skips on Windows; the four tests whose subject is tier-agnostic (`test_the_pass_refuses_on_the_name_tier_while_appends_continue` — the Windows production path itself, `test_every_ledger_writer_refuses_when_the_lock_cannot_be_taken`, the two structural validator pins) run everywhere through `data_root_any_tier`. Three probe pins in `tests/test_lockfile_helpers.py` whose mechanics the Windows short-circuit never reaches are `skipif(IS_WINDOWS)` with the reason; the Windows-executed lock coverage that remains: name protocol, non-contention fail-closed, Win32 error classification, `pid_is_signalable`.

## From the Windows CI matrix on 35b82db0 (run 33663258606; the stage-2 delta review's HIGH)

1. **The name-tier disposition of abea91ec did not close the class.** windows-latest full-test on
   `35b82db0` was red on exactly two tests, both concurrency shapes, both green on the pre-C6 tip
   43dcc1d2; on the five C6-bearing Windows legs (33654743857, 33658408570, 33658966160,
   33661022574, 33663258606) the chat-append test was red on all five and the 16-writer test on
   four (it passed once, on 33661022574 — a race, not a fix): `test_concurrent_writers_keep_monotonic_sequence` («usage accounting
   lock unavailable» at `usage_import.lock`) and
   `test_terminal_projection_dedup_does_not_lose_concurrent_chat_append` (8 of 24 appended rows
   missing). The stage-2 verifier (lens e2e-and-ci, `ADVERSARIAL_windows_name_tier_finding.md`)
   named the mechanism and the operator confirmed it from the code: the name tier since round 3
   opens the lock file on EVERY poll (`os.open` → `fstat` → 512-byte read → close) to judge identity
   and owner pid; pre-C6 a contender only `stat`ed it. CPython's `os.open` on Windows shares
   read/write but not delete, so the owner's release `os.unlink` fails with ERROR_SHARING_VIOLATION
   whenever a contender's probe is open — swallowed at debug level, the lock is orphaned with the
   owner's LIVE pid: `_named_lock` (owner-aware) refuses every later monetary writer for the life
   of the process; `append_jsonl` waits its 2 s timeout on every call and appends UNLOCKED, which on
   Windows (no atomic O_APPEND) interleaves and loses rows. One mechanism, both symptoms, only under
   contention — which is why two legs happened to pass one of the two tests.
2. **Reproduced without a Windows host.** The verifier's simulator (Linux shim of the delete
   semantics, `mega_review/stage2/verify_win_orphan/sim.py`, 12 writers × 20 s of
   `append_jsonl`-shaped acquire/release): candidate — 1 refusal → orphan → 19 acquisitions, 120
   timeouts; fixed — 288 refusals absorbed, 70 238 acquisitions, no orphan.
3. **Fix (class, not instance):** `platform_layer._unlink_lock_path` — the unlink of OUR lock (release,
   both orders) and of a path-only lock (`unlink_lockfile`) retries a `PermissionError` on Windows for
   a bounded 2 s window (5 ms poll) while the path still names the held inode; POSIX does not retry
   (a PermissionError there is the directory's mode, permanent). The contender's probe is unchanged:
   the refusal is transient by construction (every handle this protocol opens on a lock lives
   microseconds), so the release is where the class is closed for every caller at once.
4. **Red-first pins** (`tests/test_lockfile_helpers.py`): `test_windows_release_retries_a_contenders_transient_sharing_refusal`
   (fails on 35b82db0: the lock survives the release; passes now — the next acquirer wins at once),
   `test_windows_release_gives_up_a_refusal_that_never_clears` (bounded window, file left, no hang),
   `test_posix_release_does_not_retry_a_permission_refusal` (one attempt). The Windows-executed proof
   is the matrix on the SHA carrying the fix; recorded in WINWAVE_CLASS_REGISTRY «3-OS matrix runs».
5. **Residual (LOW, delta lens 35b82db0..d0bb839e):** the «identity unreadable» branch of
   `acquire_exclusive_file_lock` unlinks its own file BEFORE closing the descriptor — on Windows
   always a sharing violation, so that (practically unreachable on NTFS, pre-existing) branch
   would leave a file stamped with our live pid; the stale-eviction unlink is deliberately not
   retried — the poll loop re-judges and retries it. The delta lens also measured a contender
   probe under `owner_aware_stale` at milliseconds (OpenProcess), not microseconds — inside the
   2 s window either way.
6. **Windows proof.** Runs 33668287491 (d0bb839e) and 33669250620 (4c7c5aed): both concurrency tests
   green on both legs. Red there: the POSIX release pin (`IS_WINDOWS` patched False makes the
   enforced-tier probe import `fcntl`) — `skipif(IS_WINDOWS)`; and once, on d0bb839e only,
   `test_proactive_namer_late_settlement_refreshes_cost_without_late_name` (0.0 == 0.25 inside its
   2 s poll window) — green on the next leg with byte-identical runtime code; intermittent, unrooted,
   recorded in the registry table (the retry cannot lengthen an uncontended release, and before this
   commit any refused unlink orphaned the lock and failed that test the same way).
7. **Prose corrected:** packet §10 addendum and DESIGN §8 no longer claim the Windows name tier is
   «the protocol it always ran»; the earlier ledger line «Verified by the next dispatched matrix»
   is withdrawn above.

## From the owner's batch №13 answers (2026-09-02 ~20:30Z; requirements archive [A-BATCH-13-ANSWERS])

1. **Release bar in `ci.yml` (item 4 = A, item 16 = A; protected file, owner-sanctioned).** `system-e2e-mock`
   also runs on a release tag and is a dependency of `release-preflight`; `release-preflight` runs
   `scripts/v7next_adoption.py --release` on the tag path itself; the cancellation E-suite mock lane
   (`OUROBOROS_E2E_CANCEL=mock`, 18 + 4 scenarios, ~2 min, keyless) runs inside `system-e2e-mock` on the
   same isolated roots. Pins updated: `tests/test_system_e2e_ci_lane.py` (condition),
   `tests/test_contributor_flow.py` (needs). The «never on push/PR» contract of the lane is unchanged.
2. **Pre-release for the owner's packaged manual test (item 14, owner's own words: «драфт тега … pre-release,
   чтобы всё собралось и я мог установить»).** Carriers moved to `7.0.0-rc.1` (VERSION, pyproject `7.0.0rc1`,
   uv.lock, web/package.json, GATEWAY_CONTRACT_VERSION, README badge/history row/download refs, both install
   pages, ARCHITECTURE header — `check_worktree_version_sync` clean); tag `v7.0.0-rc.1` on that commit. The
   runtime is byte-identical to `c9f5c1a4`. Item 5 = B: the FINAL delivery (ff of `ouroboros`, `v7.0.0`,
   promotion) is the owner's own act; a pre-release is not that delivery.
3. **Decisions recorded for the work that follows (each gets its own ledger section when it lands):**
   1 = B — 7.0 is not released until the Windows kernel tier works (LockFileEx on a byte range beyond the
   stamp, Windows-executed pin); 2 = A — the paid E-lane runs once on the candidate; 3 — the owner's `data/`
   is disposable («только для тестов»); 6 = B — manual test on `c9f5c1a4`, sync №4 after it, before the tag;
   7 = A — F23 closes as covered-by-gates with its gaps recorded; 8 = A — spec §6.4 stays post-release
   (DEFER-PATHS-6.4 confirmed); 9 = B — W4-F1/W4-F2 fixed now; 10 = B — the five typed process-fact surfaces
   land in 7.0 (DEFER-TYPED-PROC-5 becomes a done row when they do); 11 = A — `tests/test_usage_compaction.py`
   is split at its natural boundary, `platform_layer.py` pay-down ≤ 1500 is post-release with an issue on
   the regenerator gap; 12 = A — family manifest + a one-off projection report «oracle rows → families»;
   15 = B — mutating delegation E2E scenarios and the copy-back race (O3) are fixed now (upstream
   `a76961de..cc2eac50` carries no fix for either); 17 = B — TEST_DISPOSITION/nav20 is withdrawn from the
   spec by this record. Item 13 was not read by the owner (too long) and is re-asked in a shorter form.
4. **Records landed with this section (items 7, 8, 17).** F23: row DEFER-F23-ACCEPTANCE is `done`
   with the release bar as its hook and the gaps in its `residual:` clause. §6.4 paths/roots: row
   DEFER-SPEC64-PATHS keeps `post-release`, authority now OWNER (quote in the row). Item 17 = B: the
   spec's «test split/delete disposition» artifact and the «навигационное упражнение 20 вопросов»
   (OUROBOROS_V7_SPEC_v72.md, «Human artifacts», ~:947-950) are WITHDRAWN from the 7.0 acceptance by
   the owner's decision; what stands in for them: the hook inventory (`docs/v7next/*INVENTORY*`, the
   validator's AST-resolved nodeids) and the owner's manual test of the packaged pre-release.

## From the compaction test split lane (owner 11 = A)

`tests/test_usage_compaction.py` stood at exactly 1600 lines — the hard cap —
so the next monetary-accounting pin had nowhere to land. Owner batch №13 item
11 = A: split it at its natural boundary rather than raise the cap.

1. **The boundary: archive reader ↔ compaction pass.** The pass side stays in
   `tests/test_usage_compaction.py` — invariants 1+4 (monetary exactness and
   budget equality), 2 (in-flight rows never fold), 3 (crash-safety), 1b (the
   lock the pass runs under, including the swap-window pins), 6 (idempotent
   kinds) and 7 (trigger policy). The reader side moved to the new
   `tests/test_usage_compaction_archive.py` — invariant 5 (the CPL-5 join
   surface: chained compactions, segment hashes and the cache window, epoch
   anchors, orphan-vs-rollback, the archive symlink bounds, typed corruption
   for what the reader cannot inspect) and invariant 8 (structural validation
   of the compacted ledger, judged by what the tests exercise — every one of
   them drives `_validate_records` over baseline blocks, i.e. reads a
   compacted ledger rather than running a pass).
2. **Shared fixtures, conftest-free.** `tests/fixtures_usage_compaction.py`
   follows the existing `tests/fixtures_*.py` convention
   (`tests/fixtures_e2e_cancellation.py`): it holds `data_root_any_tier`,
   `data_root` (which skipped on Windows at the time of the split; the Windows kernel-tier lane removed that skip in the shared module), `compacted`,
   and the helpers both modules need (`_request`, `_ledger_lines`,
   `_ledger_rows`, `_settle`, `_seed_mixed_ledger`, `_compact`,
   `_append_raw_row`, `_raced_row`). Module-local helpers stayed local:
   `_decimal_money`, `_lock_path`, `_charge_survived`, `_snapshot_looks`,
   `_projection_snapshot` on the pass side; `_rewrite_header`,
   `_rewrite_segment_in_place`, `_SOURCE_PROVENANCE_KEYS`, `_embedded_header`
   on the reader side. The three fixtures are re-exported through the module
   object (`data_root = _fixtures.data_root`) rather than imported by name: a
   test's `data_root` PARAMETER shadows a bare import of that name and F811
   fires on every such test, and `pytest_plugins` would have cost the fixture
   module its assertion rewriting (`PytestAssertRewriteWarning`).
3. **No behaviour change.** No test dropped, no test edited: the two files
   collect the SAME 64 node ids as the old single file at `72bb4949` (diffed
   name-by-name, not just counted). The four tier-agnostic tests keep running
   on Windows through `data_root_any_tier` —
   `test_the_pass_refuses_on_the_name_tier_while_appends_continue` and
   `test_every_ledger_writer_refuses_when_the_lock_cannot_be_taken` on the
   pass side, `test_baseline_rows_are_rejected_outside_the_leading_block` and
   `test_group_rows_require_a_leading_header` on the reader side.
4. **Counts.** Before: `tests/test_usage_compaction.py` 1600/1600, 64 items.
   After: `tests/test_usage_compaction.py` 900, `tests/test_usage_compaction_archive.py`
   660, `tests/fixtures_usage_compaction.py` 123 — 64 items, ~700 lines of cap
   headroom for the next pin, and no new size-ratchet band entry needed (the
   manifest names none of these files; `regenerate_size_ratchet.py --check`
   is green unregenerated).
5. **Hooks repointed.** `ADOPTION_v7next.md` CPL-4 verification column now
   names all three files with what each holds (no reference in the tree was a
   `::<nodeid>` — the AST resolver in `scripts/v7next_adoption.py` sees only
   file paths here, and both it and `--release` stay green).
   `docs/v7next/DESIGN_USAGE_COMPACTION.md` §12 heading names both suites and
   which invariants each pins. The `C6_REVIEW_PACKET.md` close-out size line
   said the owner decision was "still owed" — it records the answer and the
   post-split sizes instead. The append-only round records in this ledger and
   in the packet describe the file as it was and are left as written.

Gates: `tests/test_usage_compaction.py` + `tests/test_usage_compaction_archive.py`
+ `tests/test_v7next_adoption.py` + `tests/test_docs_sync.py` rc 0 (103 passed);
`--collect-only` node-id sets identical before/after (64 = 64);
`ruff check . --select F` rc 0; `scripts/v7next_adoption.py` rc 0 and
`--release` rc 0; `scripts/regenerate_size_ratchet.py --check` rc 0.

## From the migration projection lane (owner 12 = A)

1. **The one-off projection report exists: `docs/v7next/MIGRATION_PROJECTION.md`.**
   Owner batch №13 item 12 = A kept the family-level `ADOPTION_v7next.md` as the
   proof of the v7 transplant and asked for a report projecting the oracle's
   row-level ledger onto it. The report quotes the spec clauses that demanded the
   row-level ledger (§3.2 «единственная parseable migration-таблица … CI проверяет
   уникального owner, полноту moved symbols, валидность facade/test references»,
   §8.0-2, §8.4, §8.5) and the reverse checker they imply
   (`scripts/v7_migration.py::validate_migration`, reverse arm, driven by
   `tests/test_v7_migration_ledger.py`), then accounts for every oracle row.
2. **Row-count correction.** `MIGRATION_v7.md` on `ouroboros_v7_wip @ 9f691656` has
   **3901** rows by its own parser (table lines minus header minus separator), not
   3902. The `ADOPTION_v7next.md` Sources bullet carried the header-inclusive figure
   from the F0 skeleton; it is corrected in place with the cause named. No test or
   script read the number.
3. **Grouping key.** The oracle groups its own rows by the semantic-delta id in
   column 4; the 18 non-`none` ids are exactly this manifest's 18
   `kind=semantic-delta` rows (290 oracle rows). The 3611 `{"id":"none"}` verbatim
   moves have no id, so the report groups them by source stream: ouroboros 599,
   ouroboros/tools 491, supervisor 106, server.py 47, launcher.py 3, tests 2070,
   web 170, devtools 84, skills 41. 290 + 3611 = 3901; `pending` 3876 + `retired`
   25 = 3901.
4. **Residuals the projection surfaces, none of them new defects but none of them
   previously stated in one place.** 1461 oracle rows have no destination module of
   the declared name on this tip. 1282 of those sit on files this tree still carries
   as `GIANT_PATHS` entries (23 giants; the oracle reached 0, and spec §8.5 makes an
   empty `GIANT_PATHS` part of formal completion): the 15 test giants (1006 rows),
   `web/modules/chat.js` + `web/tests/harness_accounts.test.js` (148), the two
   OSWorld bench runners (84), `skills/unix_computer_use/plugin.py` (41),
   `supervisor/workers.py` (3). The remaining 179 are covered work that landed under
   different leaf names (D36/D37 renames, `claude_advisory_review.py`,
   `review_execution.py`, `task_lifecycle.py`, `chat_activity.js`) and are already
   recorded lane by lane above.
5. **The W (web) stream has no family row at all.** 157 of its 170 rows have no
   destination here; the oracle's `chat.js` decomposition into ~20 modules did not
   come across, and its hook `web/tests/chat_facade.test.js` does not exist on this
   tree. Stated as an untransplanted stream, not as covered. The oracle's own
   disclosed anonymous-source residual (12 unnamed handler/IIFE bodies that
   structurally cannot carry a `path::symbol` row) carries over unproven with it.
6. **What the family form does not prove, named in the report §4.** Nothing on this
   tree walks `merge-base..HEAD` and demands a row for each moved symbol: the
   manifest is a manifest of decisions, not of motions. A verbatim extraction with
   no family row and no facade change passes every current gate
   (`scripts/v7next_adoption.py --release`, the owner-facade suites,
   `tests/test_generated_inventories.py::test_facade_inventory_is_byte_identical`,
   `ouroboros/domains.toml` completeness, the size ratchet). The report says so
   instead of implying coverage from the word "family".
7. **No new gate, no new script.** The table is re-derivable from two frozen inputs
   by the commands printed in report §5 (a `git show` of the oracle ledger and a
   ~25-line inline reader); adding a checker would contradict the owner's own
   decision to stop at a one-off report.

## From the Windows kernel-tier lane (owner batch №13 item 1 = B)

Owner decision, 2026-09-02: 7.0 is not released until the Windows KERNEL lock tier works.
This lane implements the post-release recipe the C6 packet wrote for itself and brings it
into 7.0. Base `72bb4949`.

1. **What was wrong, and what changed.** The `LockFileEx` tier of `bf8b6549` locked the
   whole file (offset 0, length `0xFFFFFFFFFFFFFFFF`). A Win32 byte-range lock is MANDATORY:
   the bytes inside it cannot be READ by another handle, so a contender that opened the held
   lock file to read the owner stamp — the read this protocol's every poll makes to judge a
   hold — was refused, judged nothing and waited out its timeout (run 33654743857: eight
   monetary writers answered «lock unavailable», four `update_json_locked` timeouts, a lost
   chat append). The fix is the range, not the tier: `platform_layer._WIN32_LOCK_OFFSET`
   = `0x7FFFFFFF00000000`, `_WIN32_LOCK_LENGTH` = 1 — one byte at an offset no lock file can
   reach (a lock beyond end-of-file is legal on Windows), leaving [0, 512) readable by
   everyone. `kernel_file_locks_enforced` lost its `IS_WINDOWS → False` short-circuit and
   probes there exactly as on POSIX; the errno classification (`_WIN32_LOCK_ERRNOS`: 33 held,
   1/50 unsupported, anything else fail-closed) is untouched.
2. **Eviction on Windows, and the guarantee it does NOT give.** The enforced tier's stale
   eviction now takes the same non-blocking kernel lock on the judged descriptor on Windows
   as on POSIX (a creator stalled between its O_EXCL create and its lock is judged by the
   kernel, not by age alone; a live holder's lock refuses the probe and nothing is evicted).
   But Windows deletes no open file, so it releases that hold, closes the probe and only then
   re-checks the identity and unlinks. POSIX's «of two racing reclaimers at most one can
   evict» is a KERNEL guarantee across judge → re-check → unlink; Windows does not have it.
   What it has instead, stated in the docstring, DESIGN §8, the packet §10 addendum and the
   ARCHITECTURE row: two reclaimers may both re-check, and the loser's unlink is refused by
   the winner's own open handle on its freshly won lock — a sharing violation the eviction
   path deliberately does NOT retry (the poll loop re-judges; only the RELEASE retries, which
   is the d0bb839e class). Release order on Windows is unlock → close → unlink: a handle
   closed with an outstanding lock leaves the release undefined per Win32.
3. **`_win32_overlapped` is gone.** The per-fd OVERLAPPED map existed only to remember the
   range for the unlock; the range is now a constant, so `_win32_unlock` rebuilds it. That
   removes state that could answer for a descriptor number the process had recycled onto
   another file — a live hazard the moment any caller closes a locked fd, which the acquire
   loop does on every stand-down.
4. **The compaction pass runs on Windows.** `tests/test_usage_compaction.py`'s `data_root`
   fixture no longer skips there (the four tier-agnostic pins keep `data_root_any_tier`; the
   POSIX fsync/dir-fd/inode pins keep their `skipif`). Four probe pins in
   `tests/test_lockfile_helpers.py` skipped «because the Windows short-circuit never reaches
   the probe» run again, as does `test_a_stale_lock_is_never_evicted_without_the_kernel_hold`
   (Windows takes that hold now). The pins that stay Windows-skipped are the ones whose TEST
   BODY unlinks or rewrites a lock file its owner holds open, plus the POSIX eviction ORDER
   pin and the POSIX release pin — their reasons were rewritten to say that instead of «7.0
   ships Windows on the name tier».
5. **Pins, and exactly what they prove.** Red-first on `72bb4949`, all four:
   `test_the_windows_lock_range_lies_beyond_every_owner_stamp` (the constants, and that
   neither wrapper still spells the whole-file range),
   `test_windows_contenders_read_the_owner_stamp_while_the_kernel_hold_stands` (an emulated
   LockFileEx — `ctypes.wintypes` is Windows-only — grants exactly the range the real wrapper
   asks for and refuses it to a second descriptor with ERROR_LOCK_VIOLATION → EAGAIN; the
   predicate answers enforced, a second acquirer stands down, and the stamp is still readable
   through the hold), `test_windows_evicts_a_stale_lock_only_under_the_probe_hold`, and
   `test_windows_release_unlocks_before_the_close_and_unlinks_after_it` (order read off the
   fd's own liveness). The two existing Windows release pins now install the same emulation.
   Plus the verifier's delete-semantics simulator, re-run against this platform_layer with
   `IS_WINDOWS = True` (12 writers × 20 s): base `72bb4949` (name tier) 45 826 acquisitions,
   0 timeouts, 1 248 sharing violations absorbed, no orphan; this lane (kernel tier) 43 317
   acquisitions, 0 timeouts, 1 281 absorbed, no orphan — against the 35b82db0 shape's 19
   acquisitions and 120 timeouts. The ~5% is the extra probe lock/unlock per contended poll.
   None of these is a Windows execution: they pin the mechanism on Linux under an emulation
   of the two Win32 calls and of the delete semantics.
6. **The Windows-EXECUTED proof is the next CI matrix**, dispatched by the operator after
   integration. Only a Windows leg can prove: that `LockFileEx` accepts a lock at that offset
   past EOF and refuses the same range to a second handle with ERROR_LOCK_VIOLATION; that the
   capability probe answers enforced on the runner's volume; that a contender's stamp read is
   no longer refused; that `tests/test_usage_compaction.py` — 130-odd pins that never once
   executed a compaction on Windows — passes with the pass actually landing there; and that
   the two concurrency shapes (`test_concurrent_writers_keep_monotonic_sequence`,
   `test_terminal_projection_dedup_does_not_lose_concurrent_chat_append`) stay green on the
   kernel tier as they now are on the name tier (runs 33668287491, 33669250620).
7. **Not touched, disclosed.** The pre-existing LOW residual of item 5 in «From the Windows
   CI matrix on 35b82db0» — the «identity unreadable» branch unlinks its own file BEFORE
   closing the descriptor, which on Windows is always a sharing violation — is unchanged and
   is not made more reachable by this lane (it needs an `fstat` of our own fresh descriptor to
   fail). The Windows no-ops of the pass itself are unchanged: no directory fsync, and no
   old-inode witness across `os.replace`, so a charge landed in the swap's last syscall is
   still lost silently there rather than quarantined.
### Pre-release delivery, continued (batch №13 item 14; sits after the Windows kernel-tier lane's items by union-merge order)

- **rc.1 never built; rc.2 replaces it.** The tag run 33678261200 (`v7.0.0-rc.1` @ 72bb4949) failed at
   the embedded-bundle step on macOS and Linux: `scripts/build_repo_bundle.py` refuses a HEAD the
   configured managed source branch does not contain, and `ci.yml` hardcoded that branch to
   `ouroboros` — a pre-release cut from `ouroboros_v7next` could never build. The build job now
   RESOLVES the branch per tag (`ouroboros` when HEAD is an ancestor of `origin/ouroboros`, else the
   one remote branch containing HEAD; none/ambiguous fails loudly); pin in
   `tests/test_build_scripts.py`. Consequence for the pre-release install: its update source is
   `ouroboros_v7next` until 7.0 lands on `ouroboros` (then a reinstall or a fast-forward of that
   branch is the operator's step; disclosed). The published `v7.0.0-rc.1` tag is left as is (no
   rewrite of a published ref); `v7.0.0-rc.2` carries the fix plus the lanes merged so far
   (compaction split, migration projection, Windows kernel tier — the tag's own 3-OS matrix is that
   tier's first Windows execution).

## From the W4 crash-window lane (owner 9 = B)

Owner batch №13 item 9 = B pulled the two evolution crash windows the F4 wave-4 lane
disclosed (W4-F1, W4-F2) INTO 7.0. Both were fixed at the depth of the fact, not at the
symptom: the durable record each recovery needs is now written, or derivable, so the next
boot heals the window instead of a new gate guarding it.

**W4-F1 — the reviewed commit is two-phase.** `git commit` and `record_evolution_commit`
are two durable writes; a crash between them left a reviewed commit on HEAD that no boot
path attributed (the markerless reconcile short-circuits on an empty `commit_sha`, and
`_preserve_evolution_orphan` runs only on the authority-refusal path). The
`pre_commit_authority` boundary — the last gate before the commit, which already re-checks
the exact claim — now also records a `commit_intent` on the active transaction: the
`tree_sha` and `parents` of the post-review binding, i.e. exactly the material
`_verify_reviewed_commit_binding` verifies after the commit. Recovery
(`adopt_evolution_commit_intent`) adopts the commit at HEAD only when its tree AND its full
parent list are identical to that intent, and writes the `commit_receipt` the crash never
wrote (`reason: recovered_from_commit_intent`), so the existing restart-authority check
passes on a receipt that is as exact as the one the tool path writes. Attribution is
structural, never a guess: a failed commit, a contained orphan (the branch is rewound to
the parent) or any later HEAD movement fails the match and the transaction stays open, as
today. Two readers consume the intent — the boot reconcile, and the task-done classifier in
`update_evolution_campaign_after_task`, which would otherwise close a crashed
commit-bearing cycle as `no_op` before boot ever ran. The one writer that DISOWNS a commit —
`_preserve_evolution_orphan`, on every authority-refusal and binding-failure path — clears
the intent in the same act, so recovery can never adopt a commit an authority check refused
(cleared even when the ref surgery itself fails: the refusal is the durable fact). No new ledger: the intent lives on the
transaction that already carries the receipt, written through the existing
`update_evolution_transaction` seam.

**W4-F2 — the outcome row is re-derived, not made atomic.** The campaign write that resolves
a cycle and the `cycle_outcome` append cannot be one transaction (the append is deliberately
outside the lock so a ledger failure cannot break the restart path). They do not need to be:
the resolved transaction carries every field the row holds, so the row is DERIVABLE.
`backfill_missing_cycle_outcomes` replays, at boot, every commit-bearing resolved
transaction in `transaction_history` that has no `cycle_outcome` row
(`source: boot_backfill`), and is idempotent — a task that already has a row is skipped, so
repeated boots write nothing. Disclosed fidelity loss: `backlog_id` is not recoverable after
the fact and is left empty, exactly as the abandoned path already writes it. The
swallow-exceptions wrapper both append sites shared moved from `agent_startup_checks.py` to
`evolution_checkpoints.py` (`append_cycle_outcome_tag`), where the ledger it writes lives;
that removal is also what paid for the new boot call within the module's size band.

Red-first pins (base `72bb4949`, `tests/test_evolution_restart_claims.py`):

| pin | asserts | failure on `72bb4949` |
|---|---|---|
| `test_boot_attributes_the_commit_a_crash_left_without_a_receipt` | the intent is durable at the moment `git commit` runs, and the next boot absorbs the commit with a recovered receipt | `KeyError: 'tree_sha'` on `assert committed["intent_at_commit_time"]["tree_sha"] == binding["tree_sha"]` — nothing is written before the commit, and the boot leaves the transaction unattributed |
| `test_boot_backfills_the_cycle_outcome_row_a_crash_lost` | after a crash between the absorb write and the append, the next boot re-derives the row and the digest reports `absorbed=1`; a third boot writes nothing | `FileNotFoundError … state/evolution_checkpoints.jsonl` — no row is ever re-derived |
| `test_boot_refuses_to_attribute_a_head_that_is_not_the_reviewed_material` | a HEAD whose tree differs from the intent is never adopted (fail-closed guard, green on both trees) | — |
| `test_containment_disowns_the_commit_intent_so_boot_cannot_adopt_it` | containment clears the intent, and the following boot leaves the commit unattributed (fail-closed guard on the new mechanism; red on 72bb4949 too — `KeyError: 'commit_intent'`, the stage-3 delta lens re-derived it) | — |

Gates on the fix commit: the affected suites (evolution restart/receipt/publication/redesign/
state-integrity/scheduler, startup hygiene, module-handle extraction, commit gate, persistence
inventory, docs sync, `tests/system_e2e/`) green; `ruff check . --select F` clean;
`scripts/regenerate_size_ratchet.py --check` green (`_repo_commit_push` stays at exactly its
300-line function cap — the intent write is a keyword argument on the authority call that was
already there, not a new statement); `scripts/v7next_adoption.py --release` green with both
rows `done` and their entries removed from `DEFERRED_OUT_OF_V70`.

## From the typed process-facts lane (owner 10 = B)

Owner batch №13 item 10 = B: «the five typed process-fact surfaces land in 7.0».
The regex harvest stays retired (batch №7 item 1 = A); nothing here reads prose.
Every fact is stamped where the truth is known, and where the platform gives no
fact none is synthesized — the two `skill_preflight` fakes (`-9`, `-1`) are
retired for that reason, not merely supplemented.

Structural change under all five: the thread-local channel is now
**publisher-scoped instead of tool-name-scoped**. `_PROCESS_META_TOOLS` (a
frozenset of two names) could never list a dynamic `ext_*` surface, and the loop
now clears the slot before EVERY dispatch and merges whatever the call itself
published. That is a STRONGER no-contamination contract than the name gate:
under the gate a stale publication survived on the thread and was merely ignored
by non-process tools; now it is dropped. `tests/..::test_non_process_tools_do_not_consume_or_merge_facts`
pinned the old mechanism (the slot staying full) and is re-stated as
`test_tools_that_run_no_process_merge_no_facts`, which pins the contract.

The family grew three members that exist exactly where an exit code does not:
`timed_out`, `killed_by_host`, `pre_exec_failure` (the platform's exception
class). One projection (`loop_tool_execution._process_fact_fields`) carries the
family into the UI live-log card, the tools.jsonl row and the durable trace.

| surface | producer (fact stamped) | consumer | pin | red assertion on 72bb4949 |
|---|---|---|---|---|
| extension child-process death | `extension_process_runner._run_child` → `_publish_child_facts`: `exit_code` + POSIX `signal` on a clean/abnormal exit, `timed_out`+`killed_by_host` on the deadline kill, `killed_by_host` alone on the output-cap kill | result_meta → live card / tools.jsonl / trace | `tests/test_process_signal_observability.py::test_extension_child_death_publishes_typed_exit_and_signal`, `::test_extension_child_clean_exit_publishes_zero`, `::test_extension_child_timeout_publishes_host_kill` | `facts["exit_code"] == -9` — the channel was never published to at all (the dispatcher stamped EXTENSION_ERROR; the code lived only in the error prose) |
| `skill_exec` | `_run_skill_subprocess`: `exit_code`/`signal` for a returned child (including the NEGATIVE code its `returncode or 0` return flattens), `timed_out`+`killed_by_host` on the deadline kill, `killed_by_host` on the output-cap kill, `pre_exec_failure` on the spawn `OSError` | same | `::test_skill_exec_signal_death_survives_the_or_zero_flattening`, `::test_skill_exec_timeout_publishes_host_kill_without_exit_code`, `::test_skill_exec_output_cap_kill_is_a_host_kill_not_a_timeout`, `::test_skill_exec_spawn_failure_publishes_typed_pre_exec_cause` | `consume_last_process_facts()` returns `None` → `TypeError` on subscript: no publication existed |
| `skill_preflight` | `_run_check`: synthesized `-9` (timeout) and `-1` (missing runtime) RETIRED for `returncode=None` + `timeout`/`pre_exec_failure`; publishes `timed_out`+`killed_by_host`, `pre_exec_failure`, and a real signal death (`rc < 0`, the macOS code-signing SIGKILL this skip path exists for). The finding's skip reason gains `validator_not_started`, and its prose names the signal through the one host table instead of `-rc` | finding + result_meta → same | `::test_skill_preflight_timeout_reports_no_returncode_instead_of_fake_minus_nine`, `::test_skill_preflight_missing_runtime_reports_typed_pre_exec_cause` | `result["returncode"] is None` fails: the base answers `-9` / `-1`, which every reader downstream reads as a real POSIX signal death |
| `verify_and_record` | the check publishes `exit_code`/`signal` (completed) or `timed_out`+`killed_by_host` (timeout). The receipt now COPIES that publication (`_RECEIPT_PROCESS_KEYS`) instead of deriving duration/signal a second time — one derivation, two disclosures; the receipt's stored shape is unchanged | result_meta → same, plus the unchanged receipt | `::test_verify_check_publishes_typed_exit_and_signal`, `::test_verify_killed_check_publishes_signal_fact`, `::test_verify_timeout_publishes_typed_kill_facts` | `facts["exit_code"] == 3` → `TypeError`: the base rendered `exit=3` for the agent and stamped nothing |
| `run_command` timeouts / pre-exec | `_publish_unfinished_process_facts` gains `timed_out` (with `killed_by_host`, because the deadline IS enforced by killing the tree) and takes the spawn exception, deriving `pre_exec_failure` from an `OSError`'s class (the derivation lives in the publisher helper, which already owns the fact shaping — `shell.py` is at its 800-line extraction bound and buys no lines with a wrapper) | same | `::test_run_shell_timeout_publishes_typed_timeout_facts`, `::test_run_shell_missing_binary_publishes_duration_only` | `facts["timed_out"] is True` / `facts["pre_exec_failure"] == "FileNotFoundError"` — the base published duration only, so a timeout and a dead binary were indistinguishable in the record |
| Windows kills | no POSIX partition is faked: `signal_name_for_returncode` returns `""` for the large POSITIVE status `TerminateProcess` leaves, and `killed_by_host` beside that `exit_code` is the whole fact the platform gives | same | `::test_windows_host_kill_carries_no_forged_signal` | `facts["killed_by_host"]` → `KeyError`: the fact did not exist |

**Windows-executed proof pending the matrix.** The Windows pin runs the
platform-independent derivation (a positive exit status names no signal; the
kill fact rides beside it) on the host that is available; no Windows-executed
claim is made here.

`ouroboros/contracts/` is untouched — the typed meta shape grew only in
`ouroboros/tools/process_facts.py`, additively (three new optional members;
nothing removed or renamed). `publish_process_facts` now RETURNS what it
published, which is what lets the verify receipt disclose the published facts
rather than re-derive them.

Two function bodies were at the 300-line ratchet cap and the additions were paid
inside them rather than with a helper: `_verify_and_record` folded its
five-line duration/signal/runtime derivation into the copy of the one
publication (and moved the key-by-key rationale to `_RECEIPT_PROCESS_KEYS`,
where the contract now lives), and `_handle_skill_exec` gave its two pre-exec
publications back to `_run_skill_subprocess`, which is where the spawn — and
therefore the truth about it — actually is.

## From the paid E-lane (owner batch №13 item 2 = A; first executions 2026-09-02/03)

Receipts (operator host, isolated roots under /tmp/claude-1006, keys passed BY NAME, values never printed):

| run | tree | model / key name | result |
|---|---|---|---|
| 20:28–20:50Z 02.09 | ad39ec54 | `anthropic::claude-haiku-4-5` / `anthropic` | 4/4 red — E1–E3: `delegate_start` refused («api_actor_requires_schedule_subagent», then «subagent_selection_required»): the lane's roster had only an `api_model` row; E13: `spent_usd 0.0` after four completed tasks — the direct `anthropic::` route has no tariff, the drain never fires |
| 07:40–07:47Z 03.09 | 6bd799fb | `openrouter::anthropic/claude-haiku-4.5` / `anton_new_nsfw_key_openrouter` | E13 GREEN (budget_scope_paused recorded, no intents left); E1–E3: owned claudexord failed to start — `listen EINVAL` on its unix socket: the isolated root under the operator's private TMPDIR made `data/claudexor/daemon/claudexord.sock` 115 bytes, beyond AF_UNIX's 108 |
| 07:48–07:55Z 03.09 | ca1b38df | same, `--basetemp=/tmp/claude-1006/pb` | E1: the four verb families landed, the assertion read the faults LOG as a fault (it held only `delegate_run_containment_resolved` rows); E2/E3: the run reaches the real lane and fails at routing — `claude is unavailable: Claude subscription route is not ready` |
| 07:56–07:58Z 03.09 | 9c04ff47 | same | E1 GREEN with the assertion on `open_containment_faults(data_root) == []` |

Adjustments to the lane (test-shape, committed with this section): the paid roster gets a real
delegated leaf (`agent_session`, `claude=claude-haiku-4-5`) and E1–E3 name it in `subagent_id`;
E1 asserts OPEN faults. Not adjusted: E2/E3 — `ouroboros/tools/delegate.py` asks the engine for
`authPreference: subscription` on purpose (an invisible API-key fallback would settle a run at a
confident $0.00), and the owned daemon of an isolated install has no subscription login; logging an
account into it is an interactive owner act the operator may not perform. E2/E3 therefore remain
unexecuted (row DEFER-E2E-PAID-LANE, OWNER authority) until the owner's own logged-in install runs
`OUROBOROS_E2E_CANCEL=paid`.

Product finding from the lane (disclosed, post-release issue draft): the owned claudexord listens on
`<data>/claudexor/daemon/claudexord.sock`; a data root deeper than ~70 characters puts that path over
the AF_UNIX limit (108 on Linux, 104 on macOS) and every delegated start is refused with
`daemon_spawn_failed` whose log tail reads `listen EINVAL` — the refusal is typed but the cause is not
named. Ordinary installs (`~/Ouroboros/data`) are far below the limit; deep roots (nested temp dirs,
long usernames under /Users) are not.

## From the delegation-mutation and races lane (owner 15 = B)

Owner batch №13 item 15 = B («если это не исправлено в основной ветке ouroboros уже») —
verified before starting: upstream `a76961de..cc2eac50` carries no fix for any of the three
items below. Branch `lane/deleg-mut-and-races` off `72bb4949`; commits `626b48b7` (the two
races) and this section's scenario commit.

### 1. MUTATING delegated runs now have system-E2E cover (DEFER-E2E-DELEG-MUT closed)

Every F4 wave delegated only READ-ONLY runs and disclosed the same gap: the ONE delegation
branch that changes the owner's tree on behalf of an external harness had no system-level
cover. What was missing was not scenario prose but an ACTOR — the fake daemon had no mutating
half, so no run could produce the applied facts an integration reads.

`tests/system_e2e/interfaces.py::FakeClaudexorDaemon` gained exactly that half, and nothing
else: on `[FAKE:MUTATE]` a run (a) edits the workspace its own start body named — read from
`execution.workspaceRoot`, the PRIVATE snapshot, never from `scope.root`, so the fake cannot
break the isolation the scenarios exist to prove — and (b) writes
`<runDir>/attempts/a01/attempt.yaml` in Claudexor's applied-facts shape, the only evidence
`gateways/claudexor.py::attempt_containment` has that the harness HOME was scoped and an OS
boundary applied (the mechanism is written WITH its proven denied path, because a mechanism
without one is read as no boundary at all).

- **S24 — clean pull-in.** An external-workspace task delegates `access=workspace_write`; the
  host provisions the private Git snapshot and sends it as `execution.workspaceRoot` while
  `scope.root` stays the live tree; the harness edits the snapshot; `delegate_wait` captures;
  `integrate_delegated_patch(apply)` stages into the live workspace. Pinned: the durable
  custody chain (STARTED with access/mode/snapshot_id/execution_root/baseline_sha →
  PATCH_CAPTURED → APPLY_STARTED → DISPOSED(applied) → the `delegate_run_patch_verdict` row,
  whose `patch_sha256` equals the capture manifest's), the capture artifacts on disk, the
  ISOLATION proof, the containment read (no `delegate_run_unconfined` row for this run), the
  staged-not-committed contract (`rev-list --count HEAD` unchanged), and the released snapshot.
- **The isolation proof is causal, not timed.** The script step that runs AFTER the wait
  returned executes in the test process, so it reads the live workspace at a point the
  server's own ordering guarantees is between the terminal capture and the decision: the
  run's file is absent and `tracked.txt` is still the owner's. No sleep is a proof.
- **S25 — conflicting pull-in.** The same script step writes the drift into the live tree
  before answering with `integrate_delegated_patch(apply)`. The apply is refused typed
  (`INTEGRATE_CONFLICT` … «YOU own this conflict»; custody row
  `delegate_run_patch_apply_resolved` reason `baseline_drift`), nothing is disposed, the live
  file keeps the OWNER's content, and the snapshot, its registry row and the patch all survive.
- **Finding, recorded rather than fixed (no defect):** a task that ENDS holding an undisposed
  captured patch is not a success, and the tree says so — S25's terminal is `failed` with
  `reason_code=delegated_custody_unreconciled`,
  `delegated_runs_unreconciled=["patch:<run_id>"]` and `outcome_axes.execution=infra_failed`,
  while the model's own answer is kept verbatim. S25 pins that vocabulary, so a future change
  that quietly promoted such a task to `completed` is red.

Red-first for a coverage lane is the absence of the capability, so it is recorded as a
CONTROL run instead: with the fake's mutating half disabled (`_perform_run_work` short-
circuited) S24 fails at `assert 'ready_no_changes' == 'ready_with_changes'` — the scenario
tests the mutating path rather than passing vacuously.

### 2. O3 — the copy-back source-handle race (CI 33579445704, windows-latest attempt 1)

Two concurrent copy-backs of one task promote the SAME content-addressed source handle. Both
miss the destination, both write; on Windows the loser's `os.replace` over a destination the
winner (or a verifying reader) holds open is a sharing violation — CPython opens files without
`FILE_SHARE_DELETE`. The loser then published an INCOMPLETE promotion (`pending_refs` non-empty,
`promoted_source_handle_count=0`) while the winner published complete/1: two different custody
projections for one settled fact, which is the `child_ref_promotion` mismatch the Windows leg
read. Fixed as a class at both seams, with no sleeps and no test-side retry:

- `artifacts.py::store_actor_source_bytes` is WRITE-ONCE — the digest is in the file name, so a
  destination already holding exactly these bytes IS this handle, and the contended replace is
  removed outright for the ordinary second copier;
- `observability.py::_promote_task_source_ref` judges by the POSTCONDITION, not by authorship
  of the write: when the store raises it asks the destination once more through
  `read_actor_source_bytes`, and a verified handle there counts as promoted. Only a
  still-unreadable destination stays a pending ref.

The mechanism is Windows-only (POSIX `rename(2)` never refuses an open destination); the fix is
platform-neutral, because the postcondition is the same everywhere and both pins run on every OS.

### 3. The `/proc` env-marker race of the system-E2E lane (CI 33671108287, attempt 1)

`assert 3898 in []`: `Popen` returns as soon as the exec SUCCEEDED — the CLOEXEC error pipe
closes inside `execve` — but the kernel publishes the new image's `env_start`/`env_end` later in
that same path, so a read landing in that window sees an EMPTY environ for a live, correctly
marked child. The harness now separates the two oracles it had conflated:
`wait_pid_env_value` polls THE ONE pid for a bounded window (the POSITIVE claim), while
`pids_with_env_value` keeps its single scan (the no-orphans postcondition — an orphan was execed
long before the scan, so the window cannot hide it, and a wait there would slow every clean
teardown). Both read through one seam (`_read_proc_environ_bytes`), which is what makes the
window pinnable at all.

### Red-first table

| pin | file | red on `72bb4949` |
| --- | --- | --- |
| copy-back postcondition | `tests/test_phase3c_observability_gc.py::test_copyback_source_handle_promotion_survives_a_lost_write_race` | `AssertionError: assert 'incomplete' == 'complete'` (loser publishes a pending ref, count 0) |
| content-addressed write-once | `tests/test_phase3c_observability_gc.py::test_store_actor_source_bytes_does_not_rewrite_an_identical_handle` | `assert ['…/tool-<sha>.txt'] == []` — the identical re-store replaced the file |
| post-exec environ window | `tests/system_e2e/test_system_scenarios_w2.py::test_pid_env_wait_rides_out_the_post_exec_empty_environ_window` | `ImportError: cannot import name 'wait_pid_env_value'`; the mechanism itself demonstrated on the base harness: a single scan returns False for a live, correctly marked pid inside the window |
| mutating delegated run (S24) | `tests/system_e2e/test_system_scenarios_w5.py` | control run with the fake's mutating half disabled: `assert 'ready_no_changes' == 'ready_with_changes'` |

### Verification of this lane

`OUROBOROS_E2E_DEEP=mock … pytest tests/system_e2e/ -q` on isolated roots: **61 passed in
16:47**, rc 0 (57 scenario tests + the default-lane contract pins, now including S24/S25 and
the fake's mutating-half pin). Also green on the candidate tree: `ruff check . --select F`,
`scripts/regenerate_size_ratchet.py --check`, `scripts/v7next_adoption.py --release`
(53 rows, 39 done / 14 deferred at the time of the lane; 43 / 10 after the six lanes merged), `tests/test_docs_sync.py`,
`tests/test_system_e2e_ci_lane.py`, `tests/test_generated_inventories.py`,
`tests/test_phase3c_observability_gc.py` (16) and the 12 neighbouring suites that read
`child_ref_promotion` / `copy_child_task_result` (229 passed).

Lane hazard worth carrying: the mock lane needs a SHORT `TMPDIR`. The suite's isolated
servers spawn a `multiprocessing.Manager`, whose AF_UNIX listener path is built under
`TMPDIR`, and the 108-byte sockaddr limit turns a deep temp root into
`EOFError` from `Manager().start()` — the server then reports `supervisor_ready=True
workers=0` and every scenario dies in `_wait_ready` after its full timeout, with the real
cause only in `data/logs/server.log`.
- **rc.2 → rc.3 → rc.4 → rc.5.** rc.2 (6bd799fb, run 33680647341): the kernel tier's first Windows
  execution — one text pin red (`segment unreadable`), build skipped. rc.3 (5fbdabd3, run
  33729370259): the late-settlement timing pin red a second time (2 of 5 legs) → its poll window
  widened, build skipped. rc.4 (79597f9d, run 33730918681): the matrix green, the macOS BUILD red on
  the branch-resolution step itself — `mapfile` is a bash-4 builtin and macOS runners execute `shell:
  bash` under bash 3.2; ubuntu/windows passed the same step; rc.4 also predates the sixth lane
  (626b48b7). rc.5 is cut from the integrated tip with the portable form and a pin against bash-4
  builtins in the workflow (stage-3 delta lens H1/H2, `mega_review/stage3/DELTA_72bb4949_099dab0d.md`).
- **rc.5 → rc.6 → rc.7.** rc.5 (55d347f9) was pushed with a red test: the new no-bash-4-builtins pin
  matched its own explanatory comment in `ci.yml` — the operator's chain did not gate the push on the
  pin run (the d348ea46 lesson repeated; run cancelled). rc.6 (2eca3f4d) carried the fix but the
  branch matrix on 099dab0d (run 33732134815) showed the sixth lane's new pin
  `test_pid_env_wait_rides_out_the_post_exec_empty_environ_window` reading `/proc` on macOS and
  Windows — the sibling pin already carried `skipif(sys.platform != "linux")`, this one did not (run
  cancelled). rc.7 carries the skip; every step of its chain is gated on the pins' rc.


---

# F2 absorption (upstream 23ab428f into 18b9832e): per-symbol relocation ledger

> Appended here per roast correction R6 (no new campaign file: S1 rows summarize the transplant, S2/S3 carry their proofs). Body follows verbatim.


## What this records

Phase 2 of the v7 follow-up sprint absorbs the frozen upstream `ouroboros` branch into the
v7 line. The v7 line had already split most large modules into leaves that reach parent-owned
names through a call-time handle (`_loop()`, `_registry()`, `_rev()`, `_sub()`, `_car()`,
`_queue()`, `_events()`), so an upstream change to a function almost never lands in the file
the conflict marker points at. This ledger names, per symbol or per hunk, where each upstream
change actually went and on what authority.

It is the per-symbol relocation ledger the sprint plan requires (§5.2 item 2). It is a record
of one merge, not a live inventory: the authorities for the merged tree itself are
`ouroboros/size_ratchet_manifest.py`, `ouroboros/domains.toml`, `docs/ARCHITECTURE.md` and the
generated inventories under `docs/v7next/`.

## The three trees

| role | commit | what it is |
|---|---|---|
| merge base | `a76961de` | the last commit both lines share |
| upstream | `23ab428f` | frozen tip of `ouroboros` (`6.114.0` line) — the side being absorbed |
| v7 | `18b9832e` | `ouroboros_v7next` at `v7.0.0-rc.8` — the side being merged into |

The merge is `git merge --no-commit --no-ff 23ab428f` on top of `18b9832e`: 51 conflicted files,
370 auto-merged, one modify/delete (`ouroboros/acceptance_dialogue.py`).

## Class vocabulary

| class | meaning |
|---|---|
| **S1** | Upstream changed the function and v7 did not. The result is upstream's body — comments and docstrings verbatim, since they carry the owner's numbered decisions — placed in v7's owning leaf with v7's mechanical rewrites (handle prefixes, local imports) re-applied. Never a duplicate definition beside the leaf copy. |
| **S2** | Both sides changed the same function. Hand-merged so both intents survive; every S2 row carries its proof on the same line. |
| **S3** | v7 supersedes. Admitted only with proof: v7's mechanism already covers upstream's, or a v7 ABI 7.0 decision retired the surface, or upstream's change targets code v7 deleted. Retired 7.0 surfaces do not come back. |

Where a v7 design decision (D-nn) collided with a *later* upstream owner decision (R-nn, the
September 2026 commits), upstream was implemented provisionally and the collision is listed in
§7 as an owner fork rather than being settled silently.

## How to read the tables

`upstream symbol or hunk` names the thing that changed on the upstream side. `upstream
commit(s)` are short SHAs in `a76961de..23ab428f` (all validated in range). `v7 home (file)`
is where it lives in the merged tree; every path and symbol in this ledger was checked against
the worktree (see §9). Rows marked **UNVERIFIED** could not be proved and are called out as
such rather than asserted.

---

## 1. G1 — loop / llm / context / pipeline

### 1.1 `context.py` → `context_runtime_facts.py`

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_runtime_budget_info` (`+ctx=None`, `TOTAL_BUDGET` env → `resolve_total_budget_usd()`, `status="no_global_limit"`, `remaining_usd=None`, `+in_task_cost_ceiling`) | 8bbdac50, d4fd933c | S1 | `ouroboros/context_runtime_facts.py` | v7's copy was byte-identical to base, so upstream's body was transplanted whole |
| `+from ouroboros.task_pacing import in_task_cost_ceiling_disclosure as _in_task_cost_ceiling` | 8bbdac50, d4fd933c | S1 | `ouroboros/context_runtime_facts.py` | moved to the leaf that uses it; left in `context.py` it would be F401 |
| call site `_runtime_budget_info(env, task, ctx)` | 8bbdac50 | S1 (auto-merged, broken, fixed) | `ouroboros/context.py` | 3-arg call auto-merged against the 2-arg relocated callee — `TypeError` at every task start; fixed by the transplant above, not by reverting the call |
| `_build_installed_skills_section` `+live_loaded`/`live_reason` | 9d2fcdc0, 01a9df8e | S1 | `ouroboros/context.py` | symbol stayed in `context.py`; auto-merged and verified |
| `_project_room_fact` | — | S3 | `ouroboros/context_runtime_facts.py` | no upstream delta; the conflict was only v7's extraction |
| `_promoted_task_toolset` | — | S3 | `ouroboros/context_runtime_facts.py` | no upstream delta |
| `_delegation_capability_fact` | — | S3 | `ouroboros/context_runtime_facts.py` | no upstream delta |

### 1.2 `agent_task_pipeline.py` → `post_task_synthesis.py`

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `outcomes` import swap (`+BEST_EFFORT_REASON_CODES`, `+custody_debt_axes`, `−infra_failed_axes`) | 2421e7e5, 65e6614b | S1 | `ouroboros/agent_task_pipeline.py` | auto-merged; both new names exist in the merged `outcomes.py` |
| `_apply_terminal_custody_outcome` custody-debt overlay; best-effort rail keeps its own reason | 2421e7e5 | S1 | `ouroboros/agent_task_pipeline.py` | auto-merged, verified |
| `_store_task_result` axes ordering | 2421e7e5 | S1 | `ouroboros/agent_task_pipeline.py` | auto-merged, verified |
| `_run_task_summary` `+"outcome_phase": outcome_phase(stored_result, usage)` | 00ec9fd3 | S1 relocated | `ouroboros/post_task_synthesis.py` | `project_dialogue.outcome_phase` exists; imported function-locally in the leaf |
| `_run_task_summary` `acceptance_panels=review_projection.get("panels")` | 96cb95a3 | S1 relocated | `ouroboros/post_task_synthesis.py` | `review_evidence.format_review_evidence_for_prompt` accepts `acceptance_panels`; `review_projection` is in scope in the same function |
| `_TASK_SUMMARY_PROMPT`, `_run_chat_consolidation`, `_run_scratchpad_consolidation`, `_run_reflection` | — | S3 | `ouroboros/post_task_synthesis.py` | conflict hunk resolved to HEAD; these bodies carry no upstream delta |

### 1.3 `llm.py` and the ten lane mixins

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `+DEEPSEEK_BASE_URL`, `+normalize_deepseek_reasoning_effort` | 75c78ca2, 080be086, f109af3f | S1 | `ouroboros/llm.py` (prior-import surface) + `llm_routing.py`, `llm_openai_compatible.py` | the real consumers are the two lane leaves; the `llm.py` names are import-surface parity only |
| `+_EFFORT_CLAMP_CVAR` (ContextVar) | 1a525dbd | S1 | `ouroboros/llm_capability_policy.py` | both readers live there; `import threading` became dead and was removed |
| `_clamp_effort_for_model` thread-local → ContextVar | 1a525dbd | S1 | `ouroboros/llm_capability_policy.py` | |
| `_pop_thread_disclosure` docstring | 1a525dbd | S1 | `ouroboros/llm_capability_policy.py` | |
| `_pop_effort_clamp_disclosure` | 1a525dbd | S1 | `ouroboros/llm_capability_policy.py` | |
| `+_finalized_physical_candidate` (new) | ca775882, f9e0d840, 5dc32a8a | S1 | `ouroboros/llm_attempt.py`, re-exported from `llm.py` | `task_pacing.py` and `tests/test_tree_cost_ceiling.py` import it from `ouroboros.llm`; upstream's function-local `prepare_wire_payload_for_send` hoisted to module scope so the body stays upstream-verbatim |
| `−fetch_openrouter_pricing`, `−fetch_cloudru_pricing` → new `provider_catalogs.py` | 1a525dbd | **S2 → D-18(c)** | `ouroboros/llm_pricing.py`; `provider_catalogs.py` **deleted** | two destinations for one extraction; bodies identical apart from the logger, so v7's leaf keeps them and `llm.py` re-exports, as v7 already did. Two pricing SSOTs would otherwise ship |
| `_qualified_model_name` `+deepseek` | 080be086, f109af3f | S1 | `ouroboros/llm_routing.py` | |
| `_resolve_remote_target` `+deepseek` target (incl. `requires_reasoning_echo`) | 080be086, f109af3f | S1 | `ouroboros/llm_routing.py` | `+DEEPSEEK_BASE_URL` import added to the leaf |
| `_copy_messages_with_cache_policy` `+flatten_non_user_content_blocks` | c174522c | S1 | `ouroboros/llm_messages.py` | |
| `_strip_openrouter_roundtrip_metadata` `+keep_reasoning_content` | f109af3f | S1 | `ouroboros/llm_messages.py` | |
| `_payload_cache_breakpoints` docstring `loop.seal_task_transcript` → `context_fit.seal_task_transcript` | 743597ee | S1 (coupled to D-18a) | `ouroboros/llm_attempt.py` | |
| `_normalize_anthropic_response` (two comment trims) | b7a73355 | S1 | `ouroboros/llm_anthropic.py` | |
| **new** `_build_remote_candidate` | b7a73355 | S1, **placement decision** | `ouroboros/llm_anthropic.py::_AnthropicLaneMixin` | one provider-aware builder serves the Anthropic send and `task_pacing`'s wrap-up estimate; `llm.py` cannot host it under the 750-line pin. Owner fork §7 |
| `_chat_anthropic` payload build → `_build_remote_candidate`; `_send` → `_finalized_physical_candidate`; 3 comment trims | b7a73355, ca775882 | S1 | `ouroboros/llm_anthropic.py` | unused `_physical_candidate` / `prepare_wire_payload_for_send` imports removed |
| `_build_remote_kwargs` (+95 lines: dual-identity vision judgment, `flatten_non_user_content_blocks`, DeepSeek effort dialect, thinking-disabled forced tool choice, OR `reasoning_content` strip) | c174522c, 080be086, f109af3f, 1a525dbd | S1, byte-identical | `ouroboros/llm_openai_compatible.py` | v7's copy was verbatim to base, so upstream's body replaced it wholesale (proved by diff against upstream) |
| `_normalize_remote_response` (`reasoning_effort_clamped` pop, DeepSeek `reasoning_content` retention, `prompt_cache_hit_tokens` fallback) | ca775882, f9e0d840, f109af3f | **S2** | `ouroboros/llm_openai_compatible.py` | after the three upstream hunks the function differs from upstream by exactly v7's rename `_pop_reasoning_pin_note()` → `reasoning_artifacts.pop_reasoning_pin_note()` — nothing else |
| `_create_chat_completion_with_retries` `_send` prologue → `_finalized_physical_candidate(target, candidate, "chat.completions")` | ca775882, 5dc32a8a | **S2** | `ouroboros/llm_fallback.py` | after the edit the ladder differs from upstream by exactly v7's D09 typed-refusal rail (3 × `_is_provider_policy_refusal`) plus the `pop_reasoning_pin_note` rename; both intents kept |
| `_create_chat_completion_with_retries_async` same prologue | ca775882, 5dc32a8a | **S2** | `ouroboros/llm_fallback.py` | same proof as the sync ladder |
| `_openrouter_main_web_search_tool`, `_chat_gigachat`, lane bodies (hunks 4/5) | — | S3 | `llm_openai_compatible.py`, `llm_gigachat.py`, `llm_local.py` | unchanged upstream; the conflict was only v7's mixin split |

### 1.4 `acceptance_dialogue.py` — D-20: stays deleted, no shim

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_apply_task_acceptance_result` (drops `estimated_sec=` from `improvement_pass_allowed`) | 93c7523a (R52) | S1 | `ouroboros/loop_acceptance_review.py` | applied to v7's body so v7's handle rewrites survive |
| `_build_host_acceptance_evidence` (history/undispositioned/`budget_chars` passed into `build_task_acceptance_evidence`; `UNHASHED_ACCEPTANCE_DIALOGUE_HISTORY_KEY` gone) | b1b2f2c5, 73fb6d1d | S1 | `ouroboros/loop_acceptance_review.py` | upstream body taken whole |
| `_total_paid_acceptance_cycles` | 93c7523a | S1 | `ouroboros/loop_acceptance_review.py` | upstream's only change was `ctx: _TaskAcceptanceContext` → `ctx: Any`; see the annotation row below |
| `_execute_task_acceptance_panel` (`triad_delivery_slots`, typed `ValueError` refusal, `_refused()`, `deadline_at=_owner_deadline_at(...)`, retrieving work order, route-aware floor-priced admission, panel/delivery/native-round telemetry) | 6a5c9b82, a3599ecd, 73fb6d1d, 6ff83c60, 93c7523a | S1 | `ouroboros/loop_acceptance_review.py` | full upstream body; uses `_loop()._extract_plain_text_from_content`, v7's existing idiom in that leaf |
| **new** `terminalize_dangling_revision` | e10b3cf3 | S1 (homeless) | `ouroboros/loop_acceptance.py`, re-exported from `loop.py` | placed beside `_set_acceptance_decision` (decision writer); `loop_forced_finalization` calls it as `_loop().terminalize_dangling_revision(...)` |
| **new** `_RETRIEVING_ACCESS_DISCLOSURE` | 6a5c9b82 | S1 (homeless) | `ouroboros/loop_acceptance_review.py`, re-exported from `loop.py` | placed immediately before its sole caller |
| **new** `_retrieving_packet_projection` | 2ac3ffbe, 5a6c7724 | S1 (homeless) | `ouroboros/loop_acceptance_review.py`, re-exported from `loop.py` | `import dataclasses` added to the leaf for `dataclasses.replace` |
| **new** `acceptance_retrieving_work_order` | 6a5c9b82 | S1 (homeless) | `ouroboros/loop_acceptance_review.py`, re-exported from `loop.py` | sole caller is `_execute_task_acceptance_panel` in the same leaf |
| the other 20 top-level names of upstream's `acceptance_dialogue.py` (obligations, decision writers, quorum, dialogue history, paid identity, checklist, `_prior_acceptance_run`, `_refuse_identical_acceptance`, …) | — | S3 | `ouroboros/loop_acceptance.py` / `ouroboros/loop_acceptance_review.py` | unchanged upstream and already homed in v7; keeping upstream's file would have duplicated all 20 |
| `ctx: Any` widening on `_build_host_acceptance_evidence`, `_total_paid_acceptance_cycles`, `_execute_task_acceptance_panel` | 93c7523a | S3 (deliberate deviation, non-semantic) | `ouroboros/loop_acceptance_review.py` | upstream widened only because its split moved them away from `_TaskAcceptanceContext`; in v7 the dataclass is in the same leaf, so the typed annotation is kept |

### 1.5 R52 (upstream 93c7523a) — the host stops predicting review duration

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_run_task_acceptance_review_once` + `_TaskAcceptanceContext` R52 shape (`review_launch_allowed(budget_snapshot)` single-arg, no `launch_decision`, no `launch_at_floor_payload`) | 93c7523a | S1 | `ouroboros/loop_acceptance_review.py` | upstream tip bodies replaced v7's older shape |
| `acceptance_review_estimate_sec`, `launch_at_floor_payload`, `acceptance_native_rounds_estimate`, `DISCLOSURE_DISPATCHED_AT_FLOOR`, `launch_decision`, `wave_at_floor`, `acceptance_admission_projection`, `REASON_LAUNCHED_AT_FLOOR`, `acceptance_panel_delivery` | 93c7523a | S3 (retired by the same upstream commit) | — | tree-wide grep in `ouroboros/` returns zero hits for every one of the nine names |
| `task_pacing` deprecation-alias writer (`append_jsonl`, `utc_now_iso` imports) | 93c7523a | S3 | — | the reader of that timing record was deleted upstream; v7's Q10=A retirement already removed the alias surface |
| `project_task_acceptance_review_capacity` loses the estimate block (wallet + cancellation only) | 93c7523a, 4d35f521 | S1 (auto-merged) | `ouroboros/task_results.py` | this deletion is what makes the row above safe |

### 1.6 `loop.py` — the facade and the barrel

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `run_llm_loop` (`+invalidate_task_cache_splits`, `tool_schemas=` on `_RoundLimitContext`, comment rewraps) | 5dc32a8a, b7a73355, d472167f | S1 | `ouroboros/loop.py` | landed byte-identical to `23ab428f` from the auto-merge; verified by diff, no edit needed |
| `_setup_dynamic_tools` | 5dc32a8a | S1 | `ouroboros/loop.py` | byte-identical to upstream from the auto-merge |
| `_provider_unavailable_result` (86→83 L) | d472167f, dcea27d6 | S1 | `ouroboros/loop.py` | byte-identical to upstream from the auto-merge |
| `+invalidate_task_cache_splits` on the `usage_accounting` import | b7a73355 | S1 | `ouroboros/loop.py` | |
| `seal_task_transcript` removed from `loop.py`, now owned by `context_fit` | 743597ee | **S2 → D-18(a)** | `ouroboros/context_fit.py`; `loop.py` re-exports | two destinations; v7's 45-line copy deleted, upstream's import is the only binding. `from ouroboros.loop import seal_task_transcript` and `_loop().seal_task_transcript(...)` both still resolve |
| `_skill_names_touched_by_trace` removed from `loop.py`, now `skill_readiness.skill_names_touched_by_trace` | 0463c6bb, f8d4408c | **S2 → D-18(b)** | `ouroboros/skill_readiness.py`; `loop_nudges.py` aliases it | two destinations; v7's `loop_nudges` copy deleted. The `skill_readiness` implementation is a strict superset (adds lifecycle-tool identity keys and `skill_payload` selector keys); `write_file`/`edit_text` carry no selector key, so the existing `["alpha","beta","delta"]` assertion is unaffected |
| `TERMINAL_ORIGIN_HOST_NOTICE` import | 60c10bf4 | S1 | `ouroboros/loop.py` (`# noqa: F401 -- historical import surface`) | the leaves now import it directly from `task_finalization` |
| barrel additions: `terminalize_dangling_revision`, `_RETRIEVING_ACCESS_DISCLOSURE`, `_acceptance_delivery_slots`, `_retrieving_packet_projection`, `_skip_task_acceptance_for_launch_reason`, `acceptance_retrieving_work_order`, `_FORCED_BEST_EFFORT_TAIL`, `_prepare_forced_prompt` | e10b3cf3, 6a5c9b82, a3599ecd, f54a7cf6, f8d4408c | S1 | `ouroboros/loop.py` | all eight verified bound in `loop.py` |
| `handle_finalize_now_entry` not bound in `loop.py` | — | S3 | `ouroboros/loop_round_limits.py` | pre-existing v7 relocation untouched by this delta; nothing imports it from `loop` |
| dead alias `_call_llm_with_retry = call_llm_with_retry` | — | S3 | — | v7 dropped it; zero references tree-wide |

### 1.7 The 36 upstream `loop.py` symbol deltas, one row per leaf home

Verification method for this block: a positional AST rewriter that prefixes only `Name`-Load
nodes outside annotations. It was proved before use — applied to the BASE body of all 134 v7
leaf symbols that came from `loop.py`, it reproduces the v7 leaf byte-for-byte for 127; the 7
that differ are v7 semantic/comment edits. Inverting `_loop().` back out of every transplanted
body reproduces upstream byte-for-byte for 33 of 35 loop symbols and 4 of 4 new acceptance
symbols; the two exceptions are the S2 rows at the end of this table.

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_RoundLimitContext` (+4 fields) | d472167f, 5dc32a8a | S1 | `ouroboros/loop_round_limits.py` | handle-inversion diff = upstream |
| `_TaskAcceptanceContext` (+2 fields) | 93c7523a | S1 | `ouroboros/loop_acceptance_review.py` | handle-inversion diff = upstream |
| `_call_forced_model_once` (22→36 L, `initial_messages`/`admitted_request` + candidate predicate) | f54a7cf6, fb95e6f2, b54a2cdd | S1 | `ouroboros/loop_forced_finalization.py` | handle-inversion diff = upstream |
| `_check_budget_limits` | 8bbdac50 | S1 | `ouroboros/loop_budget.py` | handle-inversion diff = upstream |
| `_child_disposition_state` | 2421e7e5 | S1 | `ouroboros/loop_forced_finalization.py` | handle-inversion diff = upstream |
| `_force_plan_decision` (docstring trim) | f8d4408c | S1 | `ouroboros/loop_nudges.py` | handle-inversion diff = upstream |
| `_forced_delegation_note` | f54a7cf6 | S1 | `ouroboros/loop_nudges.py` | verified at `loop_nudges.py:404`; called from `loop_forced_finalization.py:497` as `_loop()._forced_delegation_note(...)`. (The lane report listed this symbol without a leaf; the home was resolved here.) |
| `_forced_fallback_result` | f54a7cf6 | S1 | `ouroboros/loop_forced_finalization.py` | handle-inversion diff = upstream |
| `_forced_orphan_note` | f54a7cf6 | S1 | `ouroboros/loop_forced_finalization.py` | handle-inversion diff = upstream |
| `_handle_provider_unavailable` (−1 L) | d472167f | S1 | `ouroboros/loop_round_limits.py` | handle-inversion diff = upstream |
| `_inject_round_checkpoints` (+1 L) | dc1487fb | S1 | `ouroboros/loop_nudges.py` | handle-inversion diff = upstream |
| `_loop_tree_accounting` | d4fd933c | S1 | `ouroboros/loop_budget.py` | handle-inversion diff = upstream |
| `_maybe_inject_finalization_nudges` (248→216 L, 25 diff hunks — the largest single delta in the organ) | 8bbdac50, d4fd933c, 18f78bcb | S1 | `ouroboros/loop_nudges.py` | differed from base only by 9 handle prefixes, so the upstream rewrite landed clean |
| `_maybe_inject_self_check` | d4fd933c | S1 | `ouroboros/loop_nudges.py` | handle-inversion diff = upstream |
| `_maybe_inject_cost_budget_milestone` | d4fd933c | S1 | `ouroboros/loop_nudges.py` | handle-inversion diff = upstream |
| `_nanny_finalization_message` | b54a2cdd | S1 | `ouroboros/loop_nudges.py` | handle-inversion diff = upstream |
| `_no_tool_final_answer` | 60c10bf4 | S1 | `ouroboros/loop_delivery.py` | handle-inversion diff = upstream |
| `_rebind_context_fit_plan` (104→99 L) | b54a2cdd, 18f78bcb | S1 | `ouroboros/loop_model_call.py` | handle-inversion diff = upstream |
| `_record_forced_acceptance_bypass` | 60c10bf4 | S1 | `ouroboros/loop_acceptance.py` | handle-inversion diff = upstream |
| `_record_forced_finalization` | e10b3cf3 | S1 | `ouroboros/loop_forced_finalization.py` | handle-inversion diff = upstream; the inline dangling-revision write is now `terminalize_dangling_revision` |
| `_reproject_actual_overflow_low` | b54a2cdd | S1 | `ouroboros/loop_model_call.py` | handle-inversion diff = upstream |
| `_resolve_task_cost_ceiling` (22→10 L, delegates to `task_pacing.resolve_task_cost_ceiling`) | 8bbdac50 | S1 | `ouroboros/loop_budget.py` | handle-inversion diff = upstream |
| `_run_forced_children_acceptance` | e10b3cf3 | S1 | `ouroboros/loop_forced_finalization.py` | handle-inversion diff = upstream |
| `_run_main_reclaim` | b54a2cdd | S1 | `ouroboros/loop_model_call.py` | handle-inversion diff = upstream |
| `_run_round_compaction` (+1 L) | d472167f | S1 | `ouroboros/loop_round_limits.py` | handle-inversion diff = upstream |
| `_run_task_acceptance_review_once` (290→274 L) | 93c7523a, a3599ecd | S1 | `ouroboros/loop_acceptance_review.py` | handle-inversion diff = upstream |
| `_soft_land_exhausted_ceiling` (35→52 L, priced-candidate path) | d4fd933c, 155ec7c5 | S1 | `ouroboros/loop_budget.py` | now reaches `_loop()._prepare_forced_prompt`; handle-inversion diff = upstream |
| `_maybe_deadline_local_finalize` | f54a7cf6 | S1 | `ouroboros/loop_round_limits.py` | handle-inversion diff = upstream |
| `_server_web_allowed_by_task` | d4fd933c | S1 | `ouroboros/loop_acceptance.py` | handle-inversion diff = upstream |
| `_direct_context_fence_state` | a3599ecd | S1 | `ouroboros/loop_acceptance_review.py` | handle-inversion diff = upstream |
| **new** `_prepare_forced_prompt` | f54a7cf6 | S1 (homeless) | `ouroboros/loop_forced_finalization.py` | placed beside the one forced model call it prices; 5 upstream tests patch it as `ouroboros.loop._prepare_forced_prompt`, which the barrel + handle preserve |
| **new** `_FORCED_BEST_EFFORT_TAIL` | f8d4408c | S1 (homeless) | `ouroboros/loop_forced_finalization.py` | used by `_soft_land_exhausted_ceiling` through the handle |
| **new** `_acceptance_delivery_slots` | a3599ecd | S1 (homeless) | `ouroboros/loop_acceptance_review.py` | sole caller lives in the same leaf |
| **new** `_skip_task_acceptance_for_launch_reason` | a58d6afd, c6d78403 | S1 (homeless) | `ouroboros/loop_acceptance_review.py` | sole caller lives in the same leaf |
| `_forced_final_answer` (122→131 L: `_prompt_prepared`/`_initial_messages`/`_admitted_request`, prompt prep moved out, first-attempt admitted candidate, `incomplete` no longer gated on `provider_terminal`, tool-asking replies degrade on every rail, unconditional `terminal_plan_review_open`/`terminal_origin`) | f54a7cf6, fb95e6f2, 60c10bf4 | **S2** | `ouroboros/loop_forced_finalization.py` | upstream's body taken whole, then v7's two edits re-applied: the `degraded` → `control_degraded` rename at 4 sites and the "Control resolution runs BEFORE the incomplete branch (#447/issue-449)" comment. Both intents present |
| `_resolve_delivery_control` (shortened hold-gate comment) | 60c10bf4 | **S2** | `ouroboros/loop_delivery.py` | `git merge-file --diff3` merged cleanly: upstream's shortened comment plus v7's three explanatory comments; no marker, no behaviour line lost |
| `_drain_incoming_messages` | — | S3 | `ouroboros/loop_round_limits.py` | unchanged upstream |
| `_maybe_inject_nanny_economics_reminder` | — | S3 | `ouroboros/loop_nudges.py` | unchanged upstream |
| `_swarm_handoff_attempt` | — | S3 | `ouroboros/loop_delivery.py` | unchanged upstream |

### 1.8 Cross-leaf imports added while transplanting

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `messages_carry_native_images` | 743597ee | S1 | `ouroboros/loop_budget.py` | no cycle: `context_fit` reaches `loop_llm_call` only lazily inside a function |
| `TERMINAL_ORIGIN_HOST_NOTICE`; `−ACCEPTANCE_FINALIZED_UNACCEPTED`/`−ACCEPTANCE_REVISION_REQUESTED` | 60c10bf4, e10b3cf3 | S1 | `ouroboros/loop_forced_finalization.py` | upstream replaced that inline write with `terminalize_dangling_revision` |
| `TERMINAL_ORIGIN_HOST_NOTICE`, `invalidate_task_cache_splits` | 60c10bf4, b7a73355 | S1 | `ouroboros/loop_round_limits.py` | |
| `invalidate_task_cache_splits` on the existing `usage_accounting` import | b7a73355 | S1 | `ouroboros/loop_model_call.py` | |
| `import dataclasses`; `−resolve_effort` from the `config` import | 6a5c9b82, a3599ecd | S1 | `ouroboros/loop_acceptance_review.py` | `resolve_effort`'s only user was the `reviewer_slots` call upstream replaced |

### 1.9 G1 tests

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `tests/test_context.py` conflict | 9d2fcdc0 | S3 | `tests/test_context.py` | resolved to v7's split shape; byte-identical to `18b9832e` |
| new `test_installed_skills_section_qualifies_extension_liveness_process` | 9d2fcdc0 | S1 | `tests/test_context_memory.py` | appended verbatim beside the sibling family v7 moved there |
| `tests/test_loop_misc.py` conflict | 4a7fa18b, a3599ecd, dc1487fb, d4fd933c | S3 | `tests/test_loop_misc.py` | resolved to v7's split shape; byte-identical to `18b9832e`; the 5 changed tests were redistributed (rows below) |
| 4 × `reviewer_slots` → `triad_delivery_slots` | a3599ecd | S1 | `tests/test_loop_acceptance_gate.py` | v7's home for the three acceptance-gate tests |
| new `test_skill_finalization_message_sees_real_skill_payload_selectors` | 0463c6bb | S1 | `tests/test_loop_skill_finalization.py` | added verbatim |
| `assert seen["tools"] is None` → `is not None` + upstream's two-line comment | dc1487fb | S1 | `tests/test_run_llm_loop.py` | |
| `LEAVES` declared sets for `loop_budget` and `loop_forced_finalization` | — | S2 | `tests/test_module_handle_extraction.py` | rows re-derived and re-implemented offline; every loop leaf passes. The `seal_task_transcript` pin needed no edit — `_module_bindings()` counts `ImportFrom` names, so the row already asserts the re-export |
| 5 × `from ouroboros.acceptance_dialogue import …` → `ouroboros.loop_acceptance_review` | D-20 | S2 | `tests/test_acceptance_delivery.py` | |
| 2 × `from ouroboros.acceptance_dialogue import terminalize_dangling_revision` → `ouroboros.loop_acceptance` | D-20 | S2 | `tests/test_v671_acceptance_convergence.py` | |
| `from ouroboros import acceptance_dialogue, task_pacing` → `loop_acceptance_review, task_pacing`; `monkeypatch.setattr(acceptance_dialogue, "_build_host_acceptance_evidence", …)` retargeted | D-20 | S2 | `tests/test_acceptance_floor_admission.py` | the patch bites: both callers reach it as a bare local name in that leaf |
| `tests/test_transcript_seal.py` | 743597ee | S1 (verify only) | `tests/test_transcript_seal.py` | already at upstream content; `from ouroboros.loop import seal_task_transcript` resolves through the D-18(a) re-export |
| `tests/test_acceptance_packet_sizing.py` | b1b2f2c5 | S1 (no change) | `tests/test_acceptance_packet_sizing.py` | never imported `acceptance_dialogue`; its two string hits are dict keys |

### 1.10 Monkeypatch reachability (the silent-green hazard)

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| the 34 names tests patch on `ouroboros.loop` | — | invariant, not a relocation | all 13 `ouroboros/loop_*.py` leaves | AST scan for bare `Name`-Load nodes with those ids across all 13 leaves: **0 hits**. Every one is reached through `_loop().` at every call site, so a patch on `ouroboros.loop` binds. Without this, a transplanted sibling call would make a test pass while patching nothing |

---

## 2. G2 — review family

### 2.1 `tools/review.py` and `tools/review_multi_model.py`

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| upstream deletes the `emit_review_usage` import | 761cf1b9 | S3 | `ouroboros/tools/review.py` (kept, `# noqa: F401`) | v7's leaves read and patch it at `tools.review.emit_review_usage` — `review_multi_model` via `_rev()`, `preflight_review_run.py` via `_car()`; both handles resolve |
| **new** `_owner_deadline_at(ctx)` | 6ff83c60 (R23) | S1 (mandatory add) | `ouroboros/tools/review.py` | three live importers reach it *from this module*: `tools/scope_review.py`, `loop_acceptance_review.py`, `tools/review_multi_model.py`; upstream's new `tests/test_delivery_retrieves.py` also imports it from here. Taking HEAD alone = two `ImportError`s |
| the rest of upstream's `:330-589` block | — | S3 | `ouroboros/tools/review_multi_model.py` | the whole block is v7's leaf; keeping upstream's copy would duplicate `_query_model` |
| `_query_model`: `retrieves = delivery_retrieves(slot_route, subagent_id)` replaces the inline `delegated or (bool(subagent_id) and route is API_CHAT)` | eb3a9b14 | **S2** | `ouroboros/tools/review_multi_model.py` | one predicate now owns delivery; upstream's own `tests/test_delivery_retrieves.py` exercises the wire-string route and the whitespace id |
| `_query_model`: `deadline_at=_rev()._owner_deadline_at(ctx)` on `ReviewRequest` | 6ff83c60 (R23) | **S2** | `ouroboros/tools/review_multi_model.py` | upstream's two-line comment verbatim; handle idiom matches the neighbouring `_rev()._cfg` / `_rev().slot_id_for_row` |
| `_multi_model_review_async`: `any_api_rows = any(not delivery_retrieves(route, row_actors[idx]) …)` | eb3a9b14 | **S2** | `ouroboros/tools/review_multi_model.py` | |
| deletion of the per-result `emit_review_usage` block | 761cf1b9 | **S2** | `ouroboros/tools/review_multi_model.py` | the producer half had auto-merged into the substrate observer while this v7-only leaf still emitted a second row — one physical send, two `llm_usage` rows, straight into the budget rail. Verified live in the isolated env: 1 physical `chat()` → exactly 1 `llm_usage` row; a control run with 2 slots emitted exactly 2. The other two `emit_review_usage` sites (`review_synthesis.py`, `preflight_review_run.py`) wrap direct `LLMClient.chat` calls that never go through the substrate |
| `_review_output_budget` | — | S3 | `ouroboros/tools/review_multi_model.py` | no upstream delta |
| `_handle_task_acceptance_review` → `triad_delivery_slots` + typed refusals | a3599ecd | S1 (auto-merged) | `ouroboros/tools/review.py` | |

### 2.2 `review_substrate.py` and its leaves

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `ReviewSlot.native_retrieval`: identity `is` → `.value` compare | 34fa7931 | **S2** | `ouroboros/review_records.py` | upstream's dataclass copies dropped (they would duplicate v7's); the delta moved into the leaf. Verified against upstream's `test_slot_properties_and_plan_review_facade_share_the_predicate` |
| `ReviewSlot.retrieves` → `delivery_retrieves(self.route, self.subagent_id)` | eb3a9b14 | **S2** | `ouroboros/review_records.py` | same proof |
| **new** field `ReviewRequest.slot_session_tasks: Dict[str, str]` | 6a5c9b82 | **S2** | `ouroboros/review_records.py` | closes a silent degradation: the writers (`loop_acceptance_review.py`) now reach the readers (`review_execution.py`, `review_native_episode.py`), which read it defensively via `getattr(...) or {}`. Un-relocated the per-slot work order collapses to the shared `session_task` with no error anywhere |
| `_review_actor_projection`: typed `not_dispatched` transport branch | 32e4d847 | S1 | `ouroboros/review_projection.py` | absent from `18b9832e:review_projection.py` before the edit; the merged coordinator emits `operation_state="not_dispatched"` from three sites |
| `compact_review_projection`: `not_dispatched` panel aggregate | 32e4d847 | S1 | `ouroboros/review_projection.py` | same |
| `_transport_error_status`, `_public_review_reason`, `_response_ref_projection`, `_review_enforcement_impact`, `_review_panel_id`, `build_review_binding`, `_criteria_*`, `aggregate_outcome_tier`, `task_acceptance_is_clean`, `build_improvement_capsule` | — | S3 | `ouroboros/review_projection.py`, `ouroboros/review_verdict.py` | verified against `git diff a76961de 23ab428f` — no upstream delta on any of them |
| `import json` in `review_substrate.py` | 32e4d847 | S3 (removed) | — | v7 used it only in the `free_refusal` `json.dumps` block that upstream deleted (replaced by `_error_actor(..., operation_state="not_dispatched")`); leaving it is `ruff F401` |
| `ReviewCoordinator` budget scope → `resolve_total_budget_usd()` | 2ed94f78 | S1 (auto-merged) | `ouroboros/review_substrate.py` | |
| `_error_actor(prompt_ref=…)` | 32e4d847 | S1 (auto-merged) | `ouroboros/review_substrate.py` | |
| `_run_slot` `usage_observer` plumbing | 761cf1b9 | S1 (auto-merged) | `ouroboros/review_substrate.py` | this is the producer half whose consumer half is the `review_multi_model` row above |
| `zero_physical_refusal(retrieving=)`, `acceptance_slot_fit` backstop, `_emit_usage(provider=…, model=resolved_model)`, the `delivery_retrieves`/`acceptance_slot_fit`/`triad_delivery_slots` re-exports | 60221fa7, b1b2f2c5, 73fb6d1d, a3599ecd | S1 (auto-merged) | `ouroboros/review_substrate.py` | verified present in the merged file |

### 2.3 `review_evidence.py` → `review_evidence_sections.py`

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| **new** `ACCEPTANCE_PROMPT_OVERHEAD_CHARS` | b1b2f2c5, 73fb6d1d | S1 | `ouroboros/review_evidence_sections.py` | placed beside its sibling caps; body upstream-verbatim |
| **new** `_ACCEPT_DENSE_CHARS_PER_TOKEN` | b1b2f2c5, 73fb6d1d | S1 | `ouroboros/review_evidence_sections.py` | |
| **new** `class AcceptancePacketBudget(int)` | b1b2f2c5, 73fb6d1d | S1 | `ouroboros/review_evidence_sections.py` | |
| **new** `acceptance_packet_budget_chars(slots)` | b1b2f2c5, 73fb6d1d | S1 | `ouroboros/review_evidence_sections.py` | needs no handle: `log` is already pinned to `"ouroboros.review_evidence"` in the leaf and the `tools.review_synthesis` import is call-time, so upstream's monkeypatch seam survives |
| `_accept_effective_claims` → 3-tuple `(claims, source, open-wave exhibit)` + `none_open_plan_wave` disclosure | 021cbdb1 | S1 | `ouroboros/review_evidence_sections.py` | leaf had no rewrites, so upstream verbatim; caller `review_evidence.py` (auto-merged) already unpacks 3 and consumes `plan_exhibit`; `current_plan_review_wave` exists in `task_results.py` |
| `_accept_artifact_manifest` (drops the redundant local `import hashlib`) | 09ac51b2 | S1 | `ouroboros/review_evidence_sections.py` | the leaf's `_ev().truncate_review_artifact` rewrite re-applied |
| `_accept_enforce_budget(ev)` → `(ev, *, budget: int = 0)` plus the whole new shed ladder (predecessor-authority envelope first, `tool_trajectory_complete` flags, repo-diff preview rung, largest-sections overflow reason, unresolved-partial trajectory rows) | 73fb6d1d | **S2** | `ouroboros/review_evidence_sections.py` | both `truncate_review_artifact` sites carry `_ev().`; caller passes `budget=budget_chars`, so the sizing work is live, not inert. Smoke: `_accept_enforce_budget(ev, budget=20_000)` sheds and writes the omission note |
| `_accept_receipt_exhibits`, `_accept_verification_summary`, `_accept_claim_support_refs`, `_accept_obligation_row`, `_accept_trajectory`, `_owner_content_projection`, `_accept_owner_directives`, `obligation_is_pending`, `task_acceptance_evidence_revision`, `_accept_redact_cap`, `_accept_task_contract`, `_accept_protected_set`, `_ACCEPT_RETRIEVAL_URLS_MAX` | — | S3 | `ouroboros/review_evidence_sections.py` | unchanged upstream; the leaf as-is is the winner. `_ACCEPT_RETRIEVAL_URLS_MAX` was scouted as new — it is in the base and already in the leaf |
| facade re-exports of the four new packet-budget names | b1b2f2c5, 73fb6d1d | S1 | `ouroboros/review_evidence.py` | required by `loop.py`, `test_acceptance_packet_sizing.py`, `test_acceptance_fixround.py`, `test_v671_acceptance_convergence.py` |
| upstream's new leaf `review_status_projection.py` (`build_review_projection`, `build_review_status_payload`, `_run_failure_reason`, 8 × `_review_status_*`) | 852ce967 | S1 (auto-merged `A`) | `ouroboros/review_status_projection.py` | kept as staged; ten facade re-exports remain in `review_evidence.py` |

### 2.4 `reviewer_slot_config.py`

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| module docstring | a3599ecd, e2945da3 | **S2** | `ouroboros/reviewer_slot_config.py` | kept v7's ABI-10 paragraph, spliced upstream's `triad_delivery_slots`/R2 sentence, dropped upstream's MIGRATION paragraph (the code it describes is deleted in v7) and v7's D15 sentence (see §7) |
| `ReviewerSlotConfig.deep_review` field | e2945da3 | S1 | `ouroboros/reviewer_slot_config.py` | mandatory: auto-merged `_parse_deep_review`, `parse_reviewer_slots`, `deep_review_slot` and `gateway/settings.py` all read it. Kept v7's `source: str  # "structured" \| "default"` line; upstream's "legacy model key" reworded to "deep-review model key" since v7 has no legacy read |
| **new** `_delivery_slot` | a3599ecd | **S2** | `ouroboros/reviewer_slot_config.py` | upstream's body with v7's typed local-route predicate `resolved_review_model_target(row.target_id).provider_route == "local"` in place of `review_model_uses_local(row.target_id)` (ABI-4); v7's ABI-4 comment carried over |
| **new** `triad_delivery_slots` | a3599ecd | **S2** | `ouroboros/reviewer_slot_config.py` | same substitution. One truthfulness edit inside upstream's docstring: "``slot_N`` from the one mint on legacy" → "owner-assigned on the structured config (ABI-10 retired the legacy comma-list read)" — the original sentence is false on this tree |
| `structured_scope_review_slots` delegates to `_delivery_slot(effort_surface="scope_review")` | a3599ecd | S1 | `ouroboros/reviewer_slot_config.py` | exactly as upstream |
| `reviewer_slots.__doc__` | a3599ecd | **S2** | `ouroboros/reviewer_slot_config.py` | kept v7's text, appended upstream's closing sentence, dropped upstream's `route_env_key` clause (the parameter does not exist in v7) and v7's D15 clause |
| `commit_triad_delivery` body | a3599ecd | **S2** | `ouroboros/reviewer_slot_config.py` | HEAD alone was non-viable — the auto-merged preamble binds `slots`, not `rows` → 8 × `NameError`. Took upstream's slot-based projection, kept v7's fingerprint predicate expressed over `slots` (upstream's `config.source == "legacy"` can never be true; ABI-10 retired that source). Verified live: default panel → `legacy_skill_fingerprint=True`, `session_targets=["","",""]`; structured mixed panel → `False` |
| `api_fallback_disclosure` deleted; `_ACCEPTANCE_API_PANEL_MEASURED` added | a3599ecd, daf37e99, fb8073b6 | **S3 / owner fork D-19** | `ouroboros/reviewer_slot_config.py` | see §7. Post-merge its three v7 collaborators were gone, `__all__` no longer exported it, and the auto-merged `test_the_retired_acceptance_api_pin_apparatus_is_gone` asserts the name absent |
| `_parse_deep_review`, `parse_reviewer_slots`, `deep_review_slot`, `acceptance_delivery_disclosure`, `reviewer_slot_save_check(previous_raw=)`, `project_reviewer_slots_into_env`, `__all__` | a3599ecd, e2945da3, daf37e99 | S1 (auto-merged) | `ouroboros/reviewer_slot_config.py` | |

### 2.5 `skill_review.py` → its four leaves

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `WASM_MAGIC` import | 4bd148a3 | S1 relocated | `ouroboros/skill_review_packs.py` | facade resolved to HEAD and is byte-identical to `18b9832e` |
| `_SKILL_PACK_TOKEN_HEADROOM` comment reword | 4bd148a3 | S1 relocated | `ouroboros/skill_review_packs.py` | value identical |
| `_LOADABLE_BINARY_EXTENSIONS` loses `".wasm"` + reworded comment | 1e406093, 4bd148a3 | S1 relocated (semantic) | `ouroboros/skill_review_packs.py` | |
| `_read_skill_file` docstring + `if text is not None and not data.startswith(WASM_MAGIC):` | 4bd148a3 | S1 relocated (semantic) | `ouroboros/skill_review_packs.py` | closes a live split-brain: `skill_review_passes.py` auto-merged and dropped `(b"\x00asm", …)` from `_EXECUTABLE_MAGICS` while the compensating guard lived in the untouched leaf. Verified end-to-end in the isolated env: the canonical 8-byte module and an all-ASCII UTF-8-decodable module are both descriptor-ised, never inlined; an ELF renamed `core.wasm` still raises `_SkillBinaryPayload` |
| `_build_skill_file_packs` `"non-UTF-8 file"` → `"binary file"` | 4bd148a3 | S1 relocated | `ouroboros/skill_review_packs.py` | |
| `_SKILL_REVIEW_ITEMS` widget-sandbox (opaque-origin) comment | 382e754f | S1 relocated | `ouroboros/skill_review_prompt.py` | coherent with the auto-merged `docs/CHECKLISTS.md` items 7/8 |
| `render_skill_review_block`: `"Claude advisory"` → `"Advisory pre-review"` ×3 | 01a9df8e | S1 relocated | `ouroboros/skill_review_output.py` | user-visible label |
| `skill_review_passes.py` | 1e406093 | S1 (auto-merged) | `ouroboros/skill_review_passes.py` | byte-identical to upstream; not edited |

### 2.6 plan review, advisory pre-review, native episode

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| removal of the `if cannot_verify:` early-return block | 68337105 | S1 (forced) | `ouroboros/tools/plan_review.py` | the HEAD side is structurally dead after the auto-merge: `_build_packet` already unpacks a 4-tuple with no `cannot_verify`, and the `_record_cannot_verify_attempt` / `_plan_reviewer_config_fingerprint` imports were auto-removed. Keeping HEAD = 3 × `NameError`; tree swept, zero remaining references |
| `_SPEC_SCHEMA` prose; `_apply_disposition` closure notes | 8be9725e, 046c4572 | S1 (auto-merged) | `ouroboros/tools/plan_review.py` | |
| import block: keep `ToolResult`/`_publish_tool_result`, drop `review_model_uses_local` | a3599ecd via 77b9df3e | **S2** | `ouroboros/tools/plan_review_runtime.py` | the only caller of `review_model_uses_local` was replaced by the auto-merged `triad_delivery_slots`; a naive union is `ruff F401` and CI-red |
| `usage_attribution={"review_wave_id": …}` | 797055ad | S1 (auto-merged) | `ouroboros/tools/plan_review_runtime.py` | |
| `_build_advisory_prompt` comment `~830KB governance bodies` → `governance bodies (hundreds of KB)` | 840167b4 | S1 relocated | `ouroboros/tools/preflight_review_prompt.py` | the whole-file upstream diff on `claude_advisory_review.py` is this one line; the facade resolved to HEAD, byte-identical to `18b9832e`. The `_mandatory_read_pointer` hunk header was a misleading diff anchor — that function is unchanged at `ouroboros/tools/claude_advisory_review.py:101` |
| `_ADVISORY_EXTRACT_CONTRACT`, `_resolve_fallback_model`, `_llm_extract_advisory_items`, `advisory_review_route`, … | — | S3 | `ouroboros/tools/preflight_review_run.py` | zero upstream delta |
| stale "SDK budget kill is replaced by the executor's config-owned round/transcript caps" comment | 60221fa7 | S1 | `ouroboros/tools/preflight_review_run.py` | rewritten to the episode's transcript bound derived from the reviewer's own window, with no round cap; grounded in the auto-merged `review_native_episode.review_native_transcript_bound` and in `OUROBOROS_REVIEW_NATIVE_MAX_ROUNDS` now being a retired key |
| `emit_review_usage` `+attribution` from `REVIEW_ATTRIBUTION_KEYS` | 761cf1b9 | S1 (auto-merged) | `ouroboros/tools/review_helpers.py` | landed in a live function; `REVIEW_ATTRIBUTION_KEYS` exists in `ouroboros/_usage_rows.py` |
| `_call_scope_llm`: `delivery_retrieves` + `deadline_at=_owner_deadline_at(ctx)` | eb3a9b14, 6ff83c60 | S1 (auto-merged) | `ouroboros/tools/scope_review.py` | resolves only because of the `tools/review.py` addition above |

### 2.7 G2 tests

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| 9 new usage-row tests (`test_one_llm_usage_row_per_physical_reviewer_call`, `…format_repair…`, `…failed_physical_reviewer_send…`, `…terminal_failed_reviewer_retry…`, `…terminal_budget_refusal…`, `…terminal_attempt_limit…`, `…internal_reviewer_transport_attempts…`, `…single_usage_row_carries_the_reviewer_attribution`, `…session_row_reports_its_own_route_provider_and_model`) | 761cf1b9, 2ed94f78, d48904a5, 18f78bcb | S1 | `tests/test_review_substrate_v2.py` | added verbatim after `test_review_usage_preserves_unknown_cost_as_null`; the file's own docstring already claims usage emission and they have no other v7 home. Zero duplicate defs |
| the 6 prompt-golden tests and `_seam_prompt_cases` in the same upstream block | — | S3 | `tests/test_review_substrate_prompts.py` | unchanged upstream and already redistributed by v7 |
| new `test_positive_custody_session_failure_emits_one_unknown_cost_usage_row` | d48904a5 | S1 | `tests/test_review_agent_session_route.py` | genuinely new, no v7 home |
| `test_applied_access_is_the_receipt_alone_never_the_request_echoed_back` | — | S3 | `tests/test_review_session_delivery.py` | v7 relocated it; the copy there is byte-identical to base (no upstream delta), so keeping upstream's would duplicate it |
| upstream's `:1276-2658` test block | — | S3 | `test_review_session_poller.py`, `test_review_session_scope_wiring.py`, `test_review_session_delivery.py` | base→upstream symbol diff proves none of them changed upstream |
| `test_acceptance_rows_stay_api_even_when_triad_routes_delegate` → `test_acceptance_rows_follow_the_configured_triad_delivery` | a3599ecd, daf37e99 | S1, **adapted setup** | `tests/test_review_agent_session_route.py` | upstream configures the panel through the legacy comma-list + `TRIAD_REVIEW_ROUTES_ENV`, both retired by ABI-10. Setup rewritten to the structured SSOT; every upstream assertion kept, including the stale-route-env negative control. Verified live: rows come back `[('slot_1', AGENT_SESSION, 'codex', 'task acceptance', retrieves=True), ('slot_2', API_CHAT, …, False)]` |
| `test_reviewer_slot_config.py` module docstring | a3599ecd, 2f61e9be | **S2** | `tests/test_reviewer_slot_config.py` | v7's ABI-10 half + upstream's R2/legacy-consumers half |
| `test_all_delegated_triad_projects_no_api_model_and_acceptance_follows_the_rows` | a3599ecd | **S2** | `tests/test_reviewer_slot_config.py` | upstream's `SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"]` KeyErrors on v7 (retired key); kept v7's `OPENROUTER_REVIEW_DEFAULTS["triad"]` form and added upstream's `triad_delivery_slots` block, which is the half that actually pins R2 |
| the two `_legacy_config` tests | a3599ecd | S3 | — | dropped, not ported: they target the comma-list migration read v7 deleted. Replacement coverage: `test_retired_phase5_route_envs_are_ignored` in the same file and `tests/test_settings_effort.py::test_review_models_default_in_config` |
| `test_all_delegated_triad_writes_no_fallback_record_and_reaches_acceptance` | a3599ecd, daf37e99 | S1 | `tests/test_reviewer_slot_config.py` | taken verbatim, replacing v7's `test_all_delegated_commit_surface_discloses_the_api_fallback` |
| `tests/test_skill_review.py` conflict | 1e406093, 4bd148a3 | S3 | `tests/test_skill_review.py` | resolved to HEAD, byte-identical to `18b9832e` |
| `test_skill_review_blocks_executables_by_magic_not_filename`: docstring `(ELF/PE/Mach-O/WASM/.pyc)` → `(ELF/PE/Mach-O/.pyc)`, `"module.blob": b"\x00asm…"` sample removed | 1e406093 | S1 relocated (semantic) | `tests/test_skill_review_packs.py` | the sample pinned behaviour upstream deliberately retired |
| new `test_skill_review_admits_wasm_as_content_hash_bound_descriptor`, `test_skill_review_routes_utf8_decodable_wasm_to_descriptor` | 1e406093, 4bd148a3 | S1 | `tests/test_skill_review_packs.py` | v7 moved all 17 sibling tests there; their imports resolve through `skill_review.py`'s re-exports |
| `test_review_evidence_keeps_the_packet_assembler_and_its_patchable_seams` `__module__` pins | 852ce967 | **S2** | `tests/test_review_evidence_extraction.py` | after upstream's leaf split two of the six names are re-exports, so the pin was guaranteed red. The pin now states the truth: four names keep the `review_evidence` pin; `build_review_projection` / `build_review_status_payload` are pinned to `"ouroboros.review_status_projection"` **plus** `name in vars(review_evidence)` so the historical read is still guarded. `_MOVED_NAMES` extended with the four new section-leaf names |
| 8 × `monkeypatch.setattr(rs, "reviewer_slots")` → `"triad_delivery_slots"` | a3599ecd | S1 | `tests/test_review_substrate_acceptance.py` | v7 redistributed the 8 acceptance tests out of `test_review_substrate_v2.py`; all 8 copies are byte-identical to base, so upstream's one uniform token change applies mechanically. Applied by the root operator on G2's request |

### 2.8 auto-merged sites G2 checked and did not need to fix

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `reviewer_slot_save_check(previous_raw=…)` at its callers | daf37e99 | S1 (no patch) | `ouroboros/gateway/settings.py`, `ouroboros/subscription_install_presets.py` | the gateway caller is byte-identical to upstream and does pass `previous_raw=stored`; the presets caller does not, and neither does upstream's own copy (that call only validates a compiled preset). The scout's "R12 fires on every save" hazard was stale |
| signature sweep: `task_acceptance_zero_physical_refusal(retrieving=)`, `acceptance_slot_fit(slot_input_caps=)`, `_error_actor(prompt_ref=)`, `emit_review_usage(provider=)`, `row_effort(default=)`, `delivery_retrieves(route, subagent_id)`, `_accept_enforce_budget(budget=)`, `_accept_effective_claims` 3-tuple, `reviewer_slot_save_check(previous_raw=)` | mixed | invariant check | organ-wide | every call site agrees with the merged signature — the "auto-merged caller + relocated callee" hazard class |

---

## 3. G3 — tools / extensions / registry

### 3.1 `ouroboros/tools/registry.py` — protected (D-21)

The merged file is byte-identical to `18b9832e:ouroboros/tools/registry.py` except for six
lines (verified by `diff -u`: exactly 6 changed lines, 313 lines total vs 309). No function
body, no `__all__`, no comment was touched. The three pure-upstream conflict regions were
resolved to the v7 side — they were the pre-split bodies, and their substance landed in the
unprotected leaves below.

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `+directory_destination_child_name` in the `shell_parse` import block | a7f7ce9a, 2a431b82, e4c7bdc0 | S1 (mandatory) | `ouroboros/tools/registry.py` line 31 | the symbol moved `shell_guards` → `shell_parse` upstream; without this the whole tool layer fails at import |
| `−directory_destination_child_name` from the `shell_guards` import block | a7f7ce9a | S1 (mandatory) | `ouroboros/tools/registry.py` | merged `shell_guards.py` no longer defines it |
| `+sequential_effective_cwds` | 2a431b82 | S1 | `ouroboros/tools/registry.py` line 34 | |
| `+interpreter_inline_code` | e4c7bdc0 | S1 | `ouroboros/tools/registry.py` line 45 | |
| `+writer_target_rows` (beside the retained `writer_target_tokens`) | a7f7ce9a, 2a431b82 | S1 | `ouroboros/tools/registry.py` line 55 | |
| `+from ouroboros.tools.write_shape import _workspace_write_candidates  # noqa: F401` | a7f7ce9a, fcf0bc10, 872a9c1b | S1, **operator addition** | `ouroboros/tools/registry.py` | the sixth line, added by the root operator after the lane flagged it: upstream's `tests/test_workspace_write_shape.py` imports the name from the facade, and it is the only such name that did not resolve. Disclosed in §7 |

### 3.2 `extension_loader.py` and the extension leaves

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_iter_payload_files` import | afe1fc4f | S1 relocated | `ouroboros/extension_plugin_api.py` | added to the module-level `skill_loader` import where `_read_module_sources` now lives, not to the loader facade |
| `_ExtensionRegistrations.module_sources` | afe1fc4f | S1 | `ouroboros/extension_registry_state.py` | field + comment verbatim |
| `_request_server_reconcile_if_worker` → `_finalize_extension_reconcile` | 2f1e2c23, db9ad562 | **S2** | `ouroboros/extension_loader.py` | v7 had generalised the seam to bidirectional and moved the transport to `extension_reconcile_queue.announce_extension_state_change`; upstream added two orthogonal jobs (a `state["server_reconcile"]` receipt and `record_health_for_runtime_state` on every reconcile exit). The result is a thin loader function that calls the queue's announcer, stamps `state["process"]`, projects the announcement onto the receipt and records health — the `is_server_process()` direction branch stays in the queue module |
| `_validate_runtime_ui_render` | b2bcd659 | S1 relocated, renamed public | `ouroboros/extension_ui_validation.py` as `validate_runtime_ui_render` (+`__all__`) | two importers (`extension_child_catalog.py`, `extension_plugin_api.py`) import it `as _validate_runtime_ui_render`; the single natural owner already holds `validate_ui_render`. This is the O3 row in §8 |
| out-of-process module-source capture | afe1fc4f | **S2** | `ouroboros/extension_registry_state.py::_StagedRegistrations.module_sources` + `extension_child_catalog._stage_out_of_process_surfaces` + `extension_plugin_api._publish_registrations` | upstream wrote `bundle.module_sources.update(...)` directly inside the registration lock; in v7 that window is validate → SWAP → attach, and a direct `_extensions[…]` write would publish a partially-populated bundle (ABI-9 atomicity + generation-bound recovery). Staged instead and copied onto the bundle at the swap, so a refused publication leaves zero servable bytes — upstream's observable contract |
| `_read_module_sources` | afe1fc4f | S1 relocated | `ouroboros/extension_plugin_api.py` | body verbatim; smoke returns the reviewed-payload walk including siblings |
| `PluginAPIImpl.register_ui_tab` (`_validate_runtime_ui_render`, module-source capture, `−"ui_host_pending": True`) | afe1fc4f, b2bcd659 | **S2** | `ouroboros/extension_plugin_api.py` | the dead write is a silent survivor of v7's split: the auto-merge had already dropped `ui_tabs_pending` from `gateway/extensions.py` and the tests, so nothing read it. Capture stages rather than mutates |
| `_extension_runtime_state` `+"process"` | 01a9df8e | S1 | `ouroboros/extension_liveness.py` | resolved via the new `_process_role()` (§7) |
| `runtime_state_for_skill_name` if/else restructure, `process` on both branches | 01a9df8e | S1 | `ouroboros/extension_liveness.py` | reproduced exactly; a missing skill is still an observation and the receipt must name the observer |
| `save_enabled(..., actor="load_error_revert")` | 01a9df8e | S1 | `ouroboros/extension_liveness.py` | `actor=` kwarg present in the merged `skill_loader.py` |
| `reconcile_extension(health_stamp=…)` + 7 call sites → `_finalize_extension_reconcile` | 2f1e2c23, db9ad562 | S2 (mechanical) | `ouroboros/extension_loader.py` | upstream's compact one-line-per-4-params signature reflow (its own size gate) was NOT carried; v7's expanded signature kept and `health_stamp` added |
| `reload_all`: `fresh_code_stamp()` + `record_health_for_runtime_state` batch; drop `hv_version`/`hv_sha`/`regressions` | 2f1e2c23 | **S2** | `ouroboros/extension_loader.py` | the mechanical merge had deleted v7's whole-set announcement `_announce_extension_state_change(drive_root, "", reason="reload_all")` together with the `regressions` block. That line has no upstream counterpart and is the only signal covering skills unloaded by the "gone" sweep — restored with its comment. Upstream's bookkeeping removal stands: its substance is inside `extension_health.record_health_for_runtime_state`, including the `extension_regression` event append |
| drop `"ui_tabs_pending": []` from `snapshot()` | b2bcd659 | S1 (auto-merged) | `ouroboros/extension_loader.py` | |
| `live_widget_projection`, `live_module_sources`, `__all__` additions | fa397986, afe1fc4f | S1 (auto-merged) | `ouroboros/extension_loader.py` | must be reachable as `extension_loader.<name>` for `gateway/widgets.py`, `gateway/extensions.py` and `tests/test_gateway_widgets.py` |
| `announce_extension_state_change` return vocabulary `{"", "requested", "request_failed"}` | 2f1e2c23, db9ad562 | **S2, widened v7 seam** | `ouroboros/extension_reconcile_queue.py` | upstream's contract is pinned by the auto-merged `tests/test_skill_widget_surface.py` and `tests/test_extensions_api.py`; v7's seam could only return `""`/`published:<gen>`/`no_skill`/`requested`, so `request_failed` was unreachable and a swallowed exception was indistinguishable from "nothing to do". The blanket `try` was split into publish / request halves. Disclosed in §7 |
| `_process_role()` | 01a9df8e | **S2, new v7 seam** | `ouroboros/extension_liveness.py` | upstream's receipt tests monkeypatch `extension_loader.is_server_process`; under the v7 split the state functions live in a leaf with no such binding. `_process_role()` resolves `is_server_process` at call time through `ouroboros.extension_loader` (the v7 leaf idiom). Disclosed in §7 |
| `control_scheduling` `parent_cognitive_route` / `status_drive_root` reflows | 2f1e2c23 | S3 (shape only) | `ouroboros/extension_loader.py` | v7's shape kept, only the semantics taken — same de-reflow rule as `reconcile_extension` |

### 3.3 `tools/core.py` → `tools/core_file_tools.py` (native read-receipt extent)

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `import bisect`, `Dict`, `Optional` header | d7f977b9, 8c7f0833 | S1 relocated | `ouroboros/tools/core_file_tools.py` | the mechanical merge had leaked `bisect`/`Optional` into the facade where nothing used them (both F401); reverted, so `tools/core.py` is byte-identical to `18b9832e` |
| `_render_line_slice(extent=…)` with the `ends`/`line_ends` arithmetic | d7f977b9, 89b7dedf | S1 relocated | `ouroboros/tools/core_file_tools.py` | body verbatim |
| `_repo_read(extent=…)` | d7f977b9, 24272e32 | S1 relocated | `ouroboros/tools/core_file_tools.py` | |
| `_data_read(extent=…)` | d7f977b9, 24272e32 | S1 relocated | `ouroboros/tools/core_file_tools.py` | |
| **new** `_stamp_read_view` | a6def41f | S1 relocated | `ouroboros/tools/core_file_tools.py` | placed beside `_annotate_reread`; deliberately NOT added to `core.py`'s re-export block — a new symbol with no historical importer |
| `_read_file` entry reset + `opened`/`opened_root` derivation + three stamp wrappings | a6def41f, 24272e32 | S1 relocated | `ouroboros/tools/core_file_tools.py` | `ctx.last_read_view` needs no `ToolContext` field: the dataclass is unslotted |

### 3.4 `tools/control.py` → the control leaves

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `schedule_subagent` BURST+ABSORB cash clause; `request_deep_self_review` reviewer-row text | 592e8be3 | S1 (auto-merged) | `ouroboros/tools/control.py` | the only upstream control content that stays in `control.py`. Verified live: `initial_tool_schemas` yields a description containing "BURST + ABSORB", "prefix write" and "burst buys latency and spacing buys cash" — the three assertions of upstream's new `test_burst_absorb_clause_states_the_prefix_write_cost` |
| **new** `_schedule_parent_chat`; `_populate_subagent_event_extras` `is not None` | 68eab3ea | S1 relocated | `ouroboros/tools/control_scheduling.py` | |
| `_schedule_task`: `_schedule_parent_chat`, `root_cost_ceiling_usd`, `chat_id=current_chat_id`, `requested_depth` | 68eab3ea, d48904a5, 558a5c65 | S1 relocated | `ouroboros/tools/control_scheduling.py` | |
| `schedule_subagent_properties["requested_depth"]` schema property | 558a5c65 | S1 relocated | `ouroboros/tools/control_subagent_spec.py` | the `requested_depth` plumbing predates the split; only the schema property and `child_budget_for_schedule(requested_depth=…)` are new |
| `_subtask_outcome_summary` `ledger_summary` `entry_count` authority | 6fdfdd75 | S1 relocated | `ouroboros/tools/control_task_results.py` | |
| `_request_deep_self_review` → `deep_review_route` / `deep_review_unavailable_text` | 592e8be3 | S1 relocated | `ouroboros/tools/control_runtime.py` | both helpers present in the merged `deep_self_review.py` |

### 3.5 `tools/shell.py` and the typed result envelope

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `from ...result_envelope import annotate, append_note` | 32628385 | S3 | — | `tools/result_envelope.py` is deleted in v7; resurrecting the import would reintroduce the untyped envelope path v7 removed |
| `from ...verify import check_exit_masking` | 32628385 | S1 | `ouroboros/tools/shell.py` | module-level exactly as upstream; no cycle (`verify.py` imports `shell` only lazily inside a function) |
| `_masked_green_disclosure` | 32628385 | **S2, re-expressed** | `ouroboros/tools/shell.py`, signature `(ctx, result, cmd)` | reads the same sensor and, when masked, republishes the already-published `ToolResult` via `_published_tool_result` / `_replace_tool_result(meta_updates=…)` / `_publish_tool_result`. Status, code and the trusted process facts (`exit_code`, `signal`) carry through untouched; the note TRAILS the payload so line 1 still belongs to the producer's typed marker; reasons ride `meta["exit_masking_reasons"]`. Applied at all three green `_publish_process_result` sites in `_run_shell` and at the `sh`/`bash` site in `_run_script`. Live smoke: masked → note present, first line unchanged, `meta={'exit_code': 0, 'exit_masking_reasons': ['\|\| true']}`; unmasked → no note, no key |
| `_preserve_result_meta` | 32628385 | **S3, superseded** | — | `tools/tool_result.py::_wrap_run_script_process_result` performs the identical job: it rebuilds the exact three run_script text framings and republishes the inner typed base via `_replace_tool_result`, promoting `code` to `ARTIFACT_OUTPUT_UNDECLARED` when an audit note is present and the base was `ok`. Upstream's helper existed only to copy `result_meta` off the retired `ToolResultText`. `tools/tool_result.py` needed no edit. This is the O3 row in §8 |
| two stranded `, cmd,` arguments inside v7's `text = ( … )` expressions | 32628385 | auto-merge defect, fixed | `ouroboros/tools/shell.py` | the merge had silently turned `text` into a tuple at both sites |
| removal of the `timeout` alias property from both schemas | d0caa69b | S1 | `ouroboros/tools/shell.py::get_tools` | paired with the `tool_resolution` row below — land both or the alias silently disappears. Smoked live |

### 3.6 shell-guard rows, per-segment write shape

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| **new** `_no_deliverables_decision` | a7f7ce9a, fcf0bc10, 872a9c1b | S1 relocated | `ouroboros/tools/write_shape.py` (landed in `registry_guards.py`, moved by the root operator — §6) | body verbatim |
| **new** `_directory_change_argv` | a7f7ce9a, fcf0bc10 | S1 relocated | `ouroboros/tools/write_shape.py` (same move) | body verbatim |
| **new** `_workspace_write_candidates` | a7f7ce9a, fcf0bc10, 872a9c1b | S1 relocated | `ouroboros/tools/write_shape.py` (same move) | body verbatim apart from two handle-prefixed `shell_argv_with_path_tokens` calls |
| `_workspace_shell_write_block`: `write_target_argvs: list[list[str]]` → `target_rows: list`, per-row cwd via `sequential_effective_cwds`, write-vs-mention split, `is_write and not pro_workspace_passthrough` at all three outside-root gates, `candidate_cwd / candidate` for the relative branch | a7f7ce9a, 2a431b82, e4c7bdc0, a2af2123 | **S2** (semantics S1, shape not) | `ouroboros/tools/registry_guards.py` | upstream's semantics landed whole; the shape was rewritten to v7's `_registry()` call-time handle and v7's typed denial carriers (`_workspace_write_block_runtime_result` / `_workspace_write_block_outside_root_result`) kept instead of upstream's message functions. Behavioural smoke: `cd /tmp && echo hi > out.txt` yields the two expected rows and row cwds `['/base','/tmp']`; `cp ../data/settings.json ./x` treats the source as mention-only and `./x` as the write |
| `_run_shell_safety_check`: `target_rows = _registry().writer_target_rows(raw_cmd)` as the one per-segment SSOT, derived `write_target_argvs`, `cd`/`pushd` rows filtered out of `explicit_write_targets`, `cp DIR/` derived children folded back into the rows, `writeish |= any(row[3] for row in target_rows)` | a7f7ce9a, 2a431b82 | **S2** | `ouroboros/tools/registry_guard_process.py` | same handle rewrite; v7's local `_interp_inline_code` alias dropped for the module-level `interpreter_inline_code` |
| `_TOOL_ARG_ALIASES["*"]["timeout"] = "timeout_sec"`; `_prepare_public_builtin_args(rejected=…)`; `_format_tool_arg_error(*, rejected=())` | d0caa69b | S1 relocated | `ouroboros/tools/tool_resolution.py` | upstream's comment verbatim; the other caller (`registry_core.py`) uses the default and is unaffected |
| `segment_write_shape()` call-time import of `_is_pure_read_inspection` | 2a431b82, fcf0bc10 | S1, **hard runtime break fixed** | `ouroboros/tools/write_shape.py` | upstream's `+72` lines auto-merged with no conflict but imported `ouroboros.tools.read_inspection`, a module v7 deleted — `ModuleNotFoundError` inside `shell_guards.writer_target_rows` on any write-shaped segment, taking down the whole run_command safety lane. Re-pointed to `ouroboros.tools.registry_guard_process`. (G3's report attributed this to `49e6359c`; that SHA is a v7-line commit — the upstream provenance is `2a431b82`/`fcf0bc10`, corrected here.) |

### 3.7 delegation / integration / preflight

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_patch_touched_paths(env=…)` | f7910f04 | S1 (auto-merged) | `ouroboros/tools/subagent_integration.py` | |
| `_target_mismatch_verdict` dedup | b7f1bdb0 | S1 (auto-merged) | `ouroboros/tools/subagent_integration.py` (nested in `_handle_external_workspace_integration`) | verified at line 495 |
| `integrate_delegated_patch` tool description | 9f692ddb | S1 (auto-merged) | `ouroboros/tools/subagent_integration.py::get_tools` | |
| `_delegated_disposition_refusal` orphan text | 9f692ddb | S1 relocated | `ouroboros/tools/subagent_integration_delegated.py` | |
| `_dispose_delegated(disposed_by=…)` | 9f692ddb | S1 relocated | `ouroboros/tools/subagent_integration_delegated.py` | `delegate_custody.record_patch_disposed(**payload)` accepts it |
| `_integrate_delegated_patch`: `orphan_disposition_status` + `orphan_note` on all three exits, `_dispose(disposed_by=…)` | 9f692ddb, dffcc89a | S1 relocated | `ouroboros/tools/subagent_integration_delegated.py` | |
| `_PAYLOAD_PRINCIPAL_PROFILES ← tool_access._TOP_LEVEL_PRINCIPAL_PROFILES` | f7910f04 | S1 (auto-merged) | `ouroboros/tools/delegate_integration.py` | |
| `_payload_delegation_busy` terminal-owner filter | fcecf13c | S1 (auto-merged) | `ouroboros/tools/delegate_integration.py` | |
| `_payload_mutation_authority` `holder_owner_task_id` | 0be2daed | S1 (auto-merged) | `ouroboros/tools/delegate_integration.py` | |
| `integrate_payload_patch`: `GIT_CEILING_DIRECTORIES` no-repo env on both git calls, `INTEGRATE_APPLY_NO_OP` branch, `disposed_by`, reworded ambiguity text | f7910f04 | S1 relocated | `ouroboros/tools/delegate_payload_patch.py` | `_finalize_payload_apply` is in the same leaf |
| `_run_check` validator loop: `**grammar` merged into v7's typed finding dict | 60a47952 | **S2** | `ouroboros/tools/skill_preflight.py` | both sides are additive to the same literal: v7's `pre_exec_failure` / `killed_by_host` / typed `skip_reason` kept, upstream's `**grammar` merged in |
| `_CLASSIC_SCRIPT_VALIDATOR`, `_widget_entry_exists_finding`, `_validate_widget_render` returning `entry`, `module_entries`, `_PREFLIGHT_SCHEMA` description | 60a47952 | S1 (auto-merged) | `ouroboros/tools/skill_preflight.py` | |
| `project_retirement_lock`, `_EMITTED_SESSION_USAGE`, `session_usage_once`, `observe_review_usage`, `observe_failed_review_send` | 09ac51b2, e0cf7910 | S1 (additive) | `ouroboros/delegate_custody_usage.py` | appended after v7's chain-aware `complete_custody_rows`, which is kept |

### 3.8 G3 tests

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `test_runtime_state_for_skill_name_reports_missing_skill` `+assert state["process"]` | 01a9df8e | S1 | `tests/test_extension_reconcile.py` | v7's home for that test |
| new `test_reload_all_reads_git_info_once_for_the_health_batch` | 2f1e2c23 | S1 | `tests/test_extension_reload_all.py` | uses the file's existing `_write_ext_skill` import from `tests/_extension_loader_shared.py` |
| new `test_standalone_reconcile_reads_a_fresh_health_stamp_each_time` | db9ad562 | S1 | `tests/test_extension_reconcile.py` | `_write_ext_skill` added to that file's shared import |
| writer-set scanner `_CORE` re-pointed `tools/core.py` → `tools/core_file_tools.py` | a6def41f | S1 | `tests/test_native_tool_round_executor.py` | verified by running the scanner's own AST logic: exactly three sites (`core_file_tools._read_file` reset, `core_file_tools._stamp_read_view` stamp, `review_native_episode._execute_inspection_call` clear); `_RESET` still matches exactly once |
| `tests/test_extension_loader.py` conflict | afe1fc4f, 01a9df8e | S3 | `tests/test_extension_loader.py` | resolved to v7's split shape; still re-exports `_prepare_extension` for `tests/test_gateway_widgets.py` |
| `tests/test_tool_capabilities.py` | 592e8be3 | S1 (no change needed) | `tests/test_tool_capabilities.py` | v7's file is a real split, not an emptied shell; the new burst test passes against the real schema |
| `tests/test_gateway_widgets.py` ×2 tests calling the v7-deleted `_register_out_of_process_surfaces` | afe1fc4f | **S2** | `tests/test_gateway_widgets.py` | v7 replaced that entry point with `_publish_out_of_process_registration`, whose signature additionally requires `state_dir`, `settings_reader`, `granted_keys`, `dependency_site_dirs_enabled` and which reads `skill.manifest`. Re-expressed by the root operator through staged publication with a real reviewed skill rather than resurrecting the old entry point (which would be a second publication path beside the ABI-9 staged one) |
| `_workspace_shell_write_block` signature pin | a7f7ce9a, 2a431b82 | S2 | `tests/test_registry_guard_process.py` | the pinned string re-stated as `target_rows: 'list'` (which replaced `write_target_argvs`), header comment updated |
| `TestMaskedGreenDisclosure` ×4-5 assertions on `result.result_meta[...]` | 32628385 | **S2** | `tests/test_shell_run_shell.py` | that attribute belonged to the retired `result_envelope.ToolResultText`; re-expressed on typed publication (`_published_tool_result(ctx, None)` → `.status` / `.code` / `.meta`) by the root operator. `test_shell_and_verify_share_one_exit_masking_sensor` passes unchanged |
| `tests/test_workspace_write_shape.py` ×21 call sites + textual `ToolResult` projection | a7f7ce9a, 2a431b82, fcf0bc10 | **S2** | `tests/test_workspace_write_shape.py` | adapted to v7's form by the root operator |

---

## 4. G4 — supervisor / server / startup / gateway

### 4.1 `server.py`

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_latest_project_task_result` (total mtime ordering, equal-mtime group read to its end across the 64-entry window, `_stamp`/`_order` helpers, `uncertain` gate on the pointer write-back) | 3b90af8d, 4cf13aa0, 5624c77f | S1 | `ouroboros/server_routing_context.py` | `git diff a76961de 23ab428f -- server.py` is 53+/20−, one function only. v7's copy was byte-identical to base, so upstream's patch applies with no mechanical rewrite. `tests/test_server_extraction.py` pins that owner |
| the other 26 functions inside the same conflict hunk | — | S3 | `server_routing_context.py`, `server_owner_routing.py`, `server_liveness.py`, `server_maintenance.py`, `server_restart.py`, `server_process.py` | unchanged upstream; conflict resolved to HEAD, leaving `server.py` byte-identical to `18b9832e` (1642 lines). Completeness check: all 57 top-level `def`s of `23ab428f:server.py` exist somewhere in the merged tree |

### 4.2 `supervisor/events.py` → the events leaves

The facade resolved to HEAD and is byte-identical to `18b9832e` (351 lines); every upstream
function delta went to its leaf. Proof for the whole block: each of the ten functions was
extracted from `23ab428f:supervisor/events.py` and from the merged leaf and diffed — the only
residual differences are v7's handle prefixes and v7's ABI-3 cost naming. All ten names stay
re-exported by the facade, so the ~25 test modules that patch `supervisor.events.<name>` still
bind. All 65 top-level `def`s of upstream's `events.py` exist in the merged tree.

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_send_subagent_rejection` (lineage routed through `notification_chat_route`, hidden-partition toast dropped) | 68eab3ea | S1 | `supervisor/events_subagent_admission.py` | per-function diff vs upstream: **0 lines** |
| `_handle_llm_usage` `+reasoning_effort_clamped` passthrough | f9e0d840 | S1 | `supervisor/events_budget.py` | per-function diff vs upstream: **0 lines** |
| `_handle_typing_start` (`notification_chat_route`, `if chat_id is not None`) | 68eab3ea | S1 | `supervisor/events_chat_delivery.py` | per-function diff: 0 (comment only) |
| `_maybe_notify_provider_death` (`notification_chat_route`, `is None`) | 68eab3ea | S1 | `supervisor/events_task_done.py` | residual = the handle prefix on `_PROVIDER_DEATH_NOTIFIED` |
| `_finish_task_done_dispatch` (local `notification_chat_route` import hoisted to module level) | 68eab3ea | S1 | `supervisor/events_task_done.py` | residual = handle prefixes + v7's ABI-3 `cost_usd` → `accounted_upper_bound_usd` seed |
| `_resolve_lifecycle_fault` (`row_chat_identity(..., default=HIDDEN_CHAT_ID)`) | 68eab3ea, 5147c851 | S1 | `supervisor/events_task_done.py` | residual = handle prefixes + ABI-3 `cost=` key |
| `_handle_task_done` (`row_chat_identity(..., default=HIDDEN_CHAT_ID)`) | 68eab3ea, 5147c851 | S1 | `supervisor/events_task_done.py` | residual = handle prefixes + ABI-3 `eff_cost` key |
| `_find_duplicate_task` (`resolve_total_budget_usd()` replaces the raw `TOTAL_BUDGET` float parse; `global_limit_usd=global_limit` with no `> 0` clamp) | 2ed94f78 | S1 | `supervisor/events_schedule_task.py` | per-function diff vs upstream: **0 lines**; the resolver returns `None` for "unconfigured" itself |
| `_reject_schedule_task` (`if chat_id:` → `if chat_id is not None:`) | 68eab3ea | S1 | `supervisor/events_schedule_task.py` | residual = v7 ABI-3 `accounted_upper_bound_usd=0.0` (upstream's `cost_usd=0.0` is the retired alias) |
| `_handle_schedule_task` (`coerce_chat_identity` with the "membership, not truthiness" comment; `"chat_id": chat_id` no longer `or None`; `root_cost_ceiling_usd` captured and put on both payloads; scheduled-toast routing via `notification_chat_route` with the hidden-partition suppression) | 68eab3ea, d48904a5 | S1 | `supervisor/events_schedule_task.py` | residual = `get_max_subagent_depth` / `_parent_delegation_budget` via `_events()`. The ceiling chain was verified end-to-end: producer `tools/control.py` → this leaf → `supervisor/task_dispatch.py`, pinned by `tests/test_tree_cost_ceiling.py`. Without the capture the field was silently dropped in transit |
| `−reject_if_no_chat_target` import and call site | 68eab3ea | S1, **silent ImportError fixed** | `supervisor/events_schedule_task.py` | upstream deleted the function from `supervisor/task_admission.py` and the deletion auto-merged, leaving v7's leaf importing a name that no longer exists — `supervisor.events_schedule_task`, hence `supervisor.events`, hence the whole supervisor, would raise `ImportError` at import time. `ruff --select F` cannot see this (cross-module). Verified absent tree-wide |
| `"chat_id": chat_id or None` → `"chat_id": chat_id`; `reject_if_no_chat_target` deletion | 68eab3ea | S1 (auto-merged) | `supervisor/task_admission.py` | no edit needed |

### 4.3 `supervisor/queue.py` → `supervisor/queue_timeouts.py`

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `+coerce_chat_identity, notification_chat_route` on the `message_bus` import | 68eab3ea | S1 | `supervisor/queue.py` | kept in v7's noqa form: `coerce_chat_identity` is read only by the timeouts leaf through the `_queue()` handle, which is what makes the noqa comment true |
| `_enforce_task_timeouts_locked` ×2 sites → `chat_id=_queue().coerce_chat_identity(task.get("chat_id"), int(owner_chat_id or 0))` | 68eab3ea | S1 | `supervisor/queue_timeouts.py` | the `_queue().` prefix is the leaf's established idiom (it already reads `get_task_idle_timeout_sec`, `FINALIZATION_GRACE_SEC` that way) |
| `queue_deep_self_review_task` (`notification_chat_route` route, typed `role="system", system_type="deep_self_review_queued"` ack) | 68eab3ea | S1 (auto-merged) | `supervisor/queue.py` | verified byte-for-byte against `23ab428f` |
| `_emit_timeout_deprecation_once` | — | **S3** | — | the record it writes declares `"remove_in": "7.0.0"`, upstream never touched it in this delta, and `tests/test_legacy_timeout_retirement.py` asserts `not hasattr(queue, "_emit_timeout_deprecation_once")` / `"_timeout_deprecation_emitted"`. Restoring it would resurrect a retired 7.0 ABI surface. 35 of 36 upstream `queue.py` defs exist in the merged tree; this is the one absentee |

### 4.4 startup, config, providers

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `SETTINGS_DEFAULTS` `+DEEPSEEK_API_KEY`, `−OUROBOROS_REVIEW_NATIVE_MAX_ROUNDS`, acceptance-floor comment | 60221fa7, 75c78ca2 | S1 relocated | `ouroboros/settings_defaults.py` | nothing may return to `config.py`: `tests/test_config_extraction.py` pins it ≤1000 lines and pins each symbol's leaf |
| `RETIRED_SETTING_KEYS += "OUROBOROS_REVIEW_NATIVE_MAX_ROUNDS"` | 60221fa7 | S1 relocated | `ouroboros/settings_defaults.py` | appended in the plain-retirement bucket, not the ABI-10 sub-block; landed atomically with the review organ's getter removal |
| `_exclusive_direct_remote_provider_env` `+deepseek`; `direct_provider_review_models_fallback` `+"deepseek"` | 75c78ca2 | S1 relocated | `ouroboros/review_model_routes.py` | |
| `get_acceptance_review_est_sec` docstring | 60221fa7 | S1 relocated | `ouroboros/runtime_limits.py` | |
| `get_finalization_grace_sec` drops the `load_settings()` fallback | — | S1 (auto-merged) | `ouroboros/config.py` | stayed in the facade |
| `OPENROUTER_DEFAULTS` `main` 3.7-flash → **gemini-3.8-flash**, `deep_self_review` `sol-pro` → **`sol`** with upstream's pro-mode billing rationale; `OPENROUTER_REVIEW_DEFAULTS.triad[0]` → **gemini-3.8-flash** | d8300b30 | S1 relocated | `ouroboros/settings_defaults.py` | v7 moved both dicts out of `provider_models.py`, which re-imports them; `tests/test_config_extraction.py::_MOVED_OWNERS` names `settings_defaults` as the destination |
| DeepSeek onboarding (`DEEPSEEK_BASE_URL`, `DEEPSEEK_REASONING_EFFORT_ALIASES`, `normalize_deepseek_reasoning_effort`, `deepseek::` in the prefix/env/credential tables, `DEEPSEEK_DIRECT_DEFAULTS`, `DIRECT_PROVIDER_DEFAULTS`, `DIRECT_PROVIDER_REVIEW_ROLES`, `_VISION_MODEL_PREFIXES`) | 75c78ca2, 080be086, f109af3f | S1 (auto-merged) | `ouroboros/provider_models.py` | live owner; landed verbatim |
| `_exclusive_direct_remote_provider(settings)` deepseek branch | 75c78ca2 | S1 (auto-merged, **no edit needed**) | `ouroboros/server_runtime.py` | the scout predicted a missing mirror; both resolvers were printed side by side and agree — same disqualifiers, same five+deepseek direct list, same `len == 1` rule |
| `plan_review_authority_core` projection; `resolve_total_budget_usd` in `check_budget`; the `state/skill_review_root_tasks.jsonl` threshold row and its `SKILL_REVIEW_ROOT_TASKS_WARN_BYTES` import | d4fd933c, 4a7fa18b, 6a6db620, 2ed94f78 | S1 (auto-merged) | `ouroboros/agent_startup_checks.py` | all cross-organ deps present; the import was moved to upstream's own placement in the list |
| `hot_store_growth_notes` docstring: v7 "four", upstream "seven" | 4a7fa18b | **S2** | `ouroboros/agent_startup_checks.py` | neither side's number is right on the merged tree. The `_hot_store_thresholds()` rows were counted — consciousness_observations, usage_attempts, events, tools, supervisor, task_reflections, progress, scheduled_tasks, skill_review_root_tasks = **nine** `os.stat` calls; written as "nine" |
| `_subject_too_large_blocked` (oversized-subject fail-closed refusal) | — | S1, adjacency only | `ouroboros/safety.py` (protected) | strictly additive in both directions: v7's `_safety_drive_root` and `_record_safety_usage` kept, upstream's `_render_subject_json` + `_subject_too_large_blocked` added |
| `_SAFETY_SUBJECT_CHAR_BUDGET`, `_render_subject_json`, `_build_check_prompt(args_json=…)`, `_run_llm_check` pre-check, `DEEPSEEK_API_KEY` in `_REMOTE_PROVIDER_KEYS`/`_PROVIDER_KEY_ENV` | 75c78ca2 | S1 (auto-merged) | `ouroboros/safety.py` | verified alongside v7's own additions |

### 4.5 gateway

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| **new** `_broadcast_task_named` | 1bd71856 | S1, adjacency | `ouroboros/gateway/tasks.py` | pure adjacency against v7's new `_task_identity_occupied` — both kept |
| **new** `_admission_names` | 1bd71856 | S1 relocated | `ouroboros/project_naming.py` as `admission_names` | moved by the root operator to the title-derivation owner (§6); the O3 row in §8 |
| `api_tasks_create` project-id guard restructure | — | **S2** | `ouroboros/gateway/tasks.py` | upstream's structure taken (it hoists `explicit_project_id_ok` into the one `project_facts` import upstream also uses for `resolve_project_id`); v7's fuller comment kept |
| `ingress_chat_id` / `ProjectThreadConflict`; `chat_id`/`title`/`suggested_name` on the admission row; `title` in the top-level-only metadata guard; the `_broadcast_task_named` call site; `timeout_sec` pre-init removed | 1bd71856, 68eab3ea | S1 (auto-merged) | `ouroboros/gateway/tasks.py` | `supervisor/log_addressing.py` supplies both new names |
| `_copy_task_summary_metadata` replays `outcome_phase` / `outcome_final` from the summary row | dcea27d6, ab0112a6 | **S2** | `ouroboros/gateway/history.py` | v7 had replaced the field loop with `carry_cost_meta(entry)` (ABI-3: cost pair converted, deprecated-wins). Hand-merged as `carry_cost_meta` **plus** an explicit copy of the two outcome keys — they are not cost fields and must not go through the cost converter |
| anchored-lineage fixed point (`active_children` → `anchored_children`, `_represents`, `_alive`), `HIDDEN_CHAT_ID` in `_make_thread_filter`, `limit` → `n_human` default, budget limit via `resolve_total_budget_usd`, `outcome_phase(result, {})` in `_annotate_terminal_task_truth` | dcea27d6, ab0112a6, 2ed94f78 | S1 (auto-merged) | `ouroboros/gateway/history.py` | v7's `terminal_truth.update(carry_cost_meta(result))` and upstream's `"outcome_phase"/"outcome_final"` coexist in one dict literal |
| `web/modules/api_types.js` +64 typedefs (history finality metadata, live-card frames) | ab0112a6, dcea27d6 | S1 (auto-merged) | `web/modules/api_types.js` | `GATEWAY_CONTRACT_VERSION` stays v7's; `node --test tests/*.test.js` → 938/938 pass |

### 4.6 build carriers and benchmark launcher

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `VERSION`, `pyproject.toml` version, `web/package.json`, README badge, ARCHITECTURE header | — | S3 | in place | upstream touched no version carrier; v7 owns the number (`7.0.0-rc.8`). `release_sync.version_carrier_desyncs()` returns `[]` |
| `pyproject.toml` `markers.integration` string | 75c78ca2 | **S2** | `pyproject.toml` | v7's two-lane sentence (CI provider-contract + the keyless `tests/system_e2e/` lane) kept, upstream's expanded key list (`+MINIMAX_API_KEY`, `DEEPSEEK_API_KEY`, `GIGACHAT_CREDENTIALS`) spliced into its parenthetical |
| `_container_triad`, `triad_rows_not_executable_in_container`, `_host_settings`, `_effective_helper_models(settings=)`, `leaderboard_metadata(settings=)`, metadata render moved into `finalize_run_manifest` | — | **S2** | `devtools/benchmarks/terminal_bench/run_tb.py` | upstream's structure with v7's source for the legacy fallback: upstream keeps `SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"]`, a `KeyError` on v7 (ABI-10 retired the key), so `",".join(OPENROUTER_REVIEW_DEFAULTS["triad"])` is used. The failure would have surfaced only at manifest render — after admission, i.e. a burned run |
| `_PROVIDER_ROUTE_ENV_KEYS` derived from `PROVIDER_CREDENTIAL_GROUPS`; `assert "scope_review" not in meta` | 75c78ca2 | **S2** | `tests/test_devtools_benchmarks.py` | upstream's registry-derived tuple taken (strictly better; picks up deepseek automatically) combined with v7's `OPENROUTER_REVIEW_DEFAULTS` loop |
| `test_apply_runtime_provider_defaults_keeps_new_triad_on_openrouter` 3.7 → 3.8 | d8300b30 | **S2** | `tests/test_server_runtime.py` | only the `OUROBOROS_MODEL` line taken; upstream's `OUROBOROS_REVIEW_MODELS == …` assert is dead on v7, where the tail asserts the retired keys are absent |
| `ouroboros/size_ratchet_manifest.py` (7 hunks) | — | **S2 (regenerate)** | `ouroboros/size_ratchet_manifest.py` | generated file; hand-merging it is meaningless. Regenerated by the root operator after all lanes landed (§6) |

### 4.7 G4 tests

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `tests/test_headless_cli.py` conflict (2453-line upstream body) | — | S3 | `tests/test_headless_cli.py` | v7's 3-line autouse-fixture import from `tests/_headless_cli_shared.py` kept; result byte-identical to `18b9832e` (462 lines) |
| `test_workspace_patch_preserves_lockfile_when_other_changes_are_junk` (`.gitignore` with `dist/`, `git add README.md .gitignore`, `untracked_excluded` 1 → **0**) | — | S1 | `tests/test_headless_workspace_patch.py` | v7's relocation target. Production side verified: `workspace_patch_rules._PATCH_EXCLUDE_RULES_VERSION = 3` and `workspace_patch_capture` enumerates untracked files with `git ls-files --others --exclude-standard`, so a git-ignored file never enters the capture universe. Both tests were red before this edit |
| `test_workspace_patch_excludes_binary_junk_and_oversize` (`exclude_rules_version` 2 → **3**) | — | S1 | `tests/test_headless_workspace_patch.py` | same |
| `test_gaia_events_serializer_carries_web_search_sources` | — | S3 | — | v7 deleted it with the events monolith; the pinned string now lives in `supervisor/events_budget.py` |

---

## 5. G5 — docs

Method (owner decision D-02): both resident docs were replaced wholesale with upstream's bytes
and the v7 payload re-applied as compact deltas. That is correct rather than hunk-merging
because upstream rewrote ARCHITECTURE from 3,679 to 1,635 lines, so essentially all 23
ARCHITECTURE conflicts were artifacts of that rewrite.

Final numbers: `docs/ARCHITECTURE.md` 105,152 tokens (budget 113,989) / 1,720 lines (cap 2,000);
`docs/DEVELOPMENT.md` 36,594 tokens (budget 41,192); residue counter CLEAN on both;
80/80 positive string pins present, 14/14 negative pins absent; 0 conflict markers.

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| upstream's consolidated ARCHITECTURE (3,679 → 1,635 lines) as the base | — | S1 (base) | `docs/ARCHITECTURE.md` | `git show 23ab428f:docs/ARCHITECTURE.md` taken verbatim, then the v7 delta re-applied |
| upstream's consolidated DEVELOPMENT (2,922 → 2,491 lines) as the base | — | S1 (base) | `docs/DEVELOPMENT.md` | same method |
| §1 component map: 127 v7 modules absent from upstream's map | — | S2 | `docs/ARCHITECTURE.md` | recomputed against the live tree with the exact function the test uses (`scripts.domain_graph.tracked_population`): 508 non-`__init__` modules, 127 missing. Result 490 rows (+39 grouped, −8), +2,810 tokens against a ≤7,000 target |
| 8 stale upstream map rows (`supervisor/chat_delivery_events.py`, `ouroboros/acceptance_dialogue.py`, `ouroboros/delivery_protocol.py`, `ouroboros/provider_catalogs.py`, `ouroboros/contracts/api_v1.py`, `ouroboros/tools/read_inspection.py`, `ouroboros/tools/result_envelope.py`, `ouroboros/tools/output_export_policy.py`) | — | S3 | `docs/ARCHITECTURE.md` | each named a module v7 deleted; each row was repurposed or dropped with the replacing module named on disk |
| §4 endpoint table | — | S1 regenerated from code | `docs/ARCHITECTURE.md` | mirrored against `gateway.endpoint_index.HTTP_ENDPOINTS` + `gateway.files.file_browser_routes()` + `{GET /, WS /ws, STATIC /static/*}` + `create_host_service_app().routes`: public missing [] / stale [], host missing [] / stale [], no duplicates. Exactly one row removed (`POST /api/owner/scope-review-floor`, retired with `OUROBOROS_SCOPE_REVIEW_FLOOR`) |
| §7 Default settings table | — | S1 regenerated from code | `docs/ARCHITECTURE.md` | mirrored against `config.SETTINGS_DEFAULTS` (140 keys): dupes [], missing [], undeclared [], mismatched defaults [], 152 rows. 6 retired rows removed; 3 added (`OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC`, `OUROBOROS_ONBOARDING_SNAPSHOT_TIMEOUT_SEC`, `OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC`). Not 4: `OUROBOROS_REVIEW_NATIVE_MAX_ROUNDS` is in `RETIRED_SETTING_KEYS`, not `SETTINGS_DEFAULTS`, so it must have no row |
| §1 `server.py` row: the startup read seam | — | S2 | `docs/ARCHITECTURE.md` | required by `test_settings_docs_name_every_key_owner_and_what_startup_persists` (3 positive pins + 2 negatives); v7's `(D03)` label dropped per D-02 |
| §1 `_usage_response.py` row | — | S2 | `docs/ARCHITECTURE.md` | required by `test_architecture_does_not_claim_usage_response_is_the_only_usage_reader` |
| §1 `deep_self_review.py` row: "the compact manifest is the atlas default" | — | S2 | `docs/ARCHITECTURE.md` | required by `test_architecture_deep_review_has_no_compact_manifest_retry_rung` |
| §1 Gateway Boundary v1: `contracts/api_v1.py` compatibility re-export gone; `gateway/contracts.py` is the one envelope owner | — | S2 | `docs/ARCHITECTURE.md` | v7's ABI-labelled narrative dropped |
| §1 Data layout: `task_results/*.json` stamped `_schema_version: 1`, one line per file, closing pointer to `DATA_LAYOUT_INVENTORY.md` | — | S2 | `docs/ARCHITECTURE.md` | v7's `(ABI 7.0, Q8=B)` label dropped |
| §5 Supervisor loop: "the **one** secondary settle site"; budget-exhausted queued task PAUSES; the direct-chat turn stop paragraph | — | S2 | `docs/ARCHITECTURE.md` | `fail_tasks` appears nowhere in the runtime (grep: only `scripts/rc_audit.py` and tests asserting its absence). Stop mechanism verified in `supervisor/worker_chat_lane.py` and `supervisor/cancel_publication.py` |
| §6 Review stack, §6 Budget tracking, §2 Startup | a3599ecd, 2ed94f78 | S3 (nothing to add) | `docs/ARCHITECTURE.md` | upstream already carries them, including "Every triad row reaches the panel as configured (`reviewer_slot_config.triad_delivery_slots`)". v7's earlier "acceptance stays API-only" wording appears nowhere in either merged doc |
| §7 new subsection "Reading and writing the settings document" | — | S2 compressed | `docs/ARCHITECTURE.md` | 83 → 48 lines / ~1,180 tok. Kept verbatim: the load-bearing ORDER argument, the five/three/two writer split with all five names, the `_owner_update_settings(transform, expected_digest)` contract, the pinned literal "Startup is a read, with one exception". Dropped the write-shape enumeration (its closed list lives in `tests/_shared/settings_writers.py`; copying it into prose is a second SSOT) |
| §8 new subsection "System E2E suite" | — | S2 compressed | `docs/ARCHITECTURE.md` | 86 → 31 lines / ~700 tok. The per-scenario narratives were replaced by one truthful coverage sentence over all 25 rows of `tests/system_e2e/harness.py::SCENARIOS`, which the doc names as the authority |
| §10 invariant 3 | — | S2 | `docs/ARCHITECTURE.md` | rewritten to name all six settings leaves, keeping the pinned literal prefix |
| §11.1 `PluginAPI` 2.0 with manifest negotiation; §11.2 `task_results/*.json` carve-out | — | S2 | `docs/ARCHITECTURE.md` | `PLUGIN_API_VERSION = "2.0"`, `LEGACY_PLUGIN_API_GENERATION = "1.3"`, 16 public methods — verified |
| §11.4 new subsection "Recent ABI Retirements" | — | S2 compressed | `docs/ARCHITECTURE.md` | 65 → 33 lines / ~830 tok. All four v7 pins hold (`**ABI 7.0**`, `5.25.0-rc.4`, `OUROBOROS_REVIEWER_SLOTS`, all 6 `RETIRED_COMMA_LIST_SETTING_KEYS`). Versions written bare, never parenthesised, which is what lets residue-zero and the §11.4 pin coexist. Added beyond v7: `OUROBOROS_REVIEW_NATIVE_MAX_ROUNDS` with its replacement bound stated |
| §6 the three gen/verify inventories paragraph; §12 `state/extension_generation.json`; §13 `model_experience` + `fix_hint` | 60221fa7, 2f1e2c23 | S2 | `docs/ARCHITECTURE.md` | each verified against the named constant or module |
| header line `# Ouroboros v7.0.0-rc.8 — Architecture & Reference` | — | S3 (v7 carrier) | `docs/ARCHITECTURE.md` | matches `VERSION`; release-sync carrier |
| DEVELOPMENT: domain-manifest pointer paragraph | — | S2 | `docs/DEVELOPMENT.md` | required by `test_the_domain_manifest_is_reachable_from_the_handbook`; names `ouroboros/domains.toml`, `docs/DOMAIN_MAP.md`, `scripts/check_domains.py --write` |
| DEVELOPMENT: "Paying down a size cap" bullet | — | S2 | `docs/DEVELOPMENT.md` | the exact phrase is kept because CHECKLISTS item 31 cites it by name; the date and the numbered advisory label were reworded away |
| DEVELOPMENT: contributor lane re-runs the target base's own review machinery from a detached worktree | — | S2 | `docs/DEVELOPMENT.md` | verified in `scripts/run_external_review.py` |
| DEVELOPMENT: "Cancellation and effective status" correction | — | S3 (dropped here, applied elsewhere) | `docs/ARCHITECTURE.md` §5 | upstream deleted the sentence v7 was correcting from DEVELOPMENT and points at ARCHITECTURE §5, where the stale claim actually survived |
| DEVELOPMENT: `wait_tasks` batch projection names `accounted_upper_bound_usd` first | — | S2 | `docs/DEVELOPMENT.md` | verified against `ouroboros/cost_projection.py` and `tools/control_task_results.py` |
| DEVELOPMENT: `owner_write_guard` membership; timeout SSOT in `settings_defaults.py` + `runtime_limits.py`; byte-exact atomic writers; the `system_e2e` marker lane | — | S2 | `docs/DEVELOPMENT.md` | the timeout row is required: the merged test asserts `settings_defaults.py` appears in DEVELOPMENT (upstream had 0 occurrences) and that the old `config.py` phrasing does not |
| DEVELOPMENT: `acceptance_dialogue.` module prefix removed from three sentences | D-20 | S2 | `docs/DEVELOPMENT.md` | `loop_acceptance._set_acceptance_decision` is verified; `acceptance_retrieving_work_order` and `terminalize_dangling_revision` are written without a module prefix (see §7's UNVERIFIED note, now resolved — both homes are recorded in §1.4) |
| `docs/CHECKLISTS.md` item 5 `env_allowlist` (the one conflict hunk) | 75c78ca2 | **S2** | `docs/CHECKLISTS.md` | the two sides differed by exactly two tokens: upstream ADDS `DEEPSEEK_API_KEY`, v7 REMOVES `TELEGRAM_BOT_TOKEN`. Both intents kept and verified against the merged code constant `ouroboros/contracts/plugin_api.py::FORBIDDEN_SKILL_SETTINGS`, not against either diff — DEEPSEEK present, TELEGRAM absent. Verified in the merged file: 1 occurrence of `DEEPSEEK_API_KEY`, 0 of `TELEGRAM_BOT_TOKEN` |
| `docs/CHECKLISTS.md` items 7 (`extension_namespace_discipline`) and 8 (`widget_module_safety`) | 382e754f | S1 (auto-merged) | `docs/CHECKLISTS.md` | upstream's rewritten launch-policy text with `WIDGET_START_MODES` / `__ouroWidgetOnDispose`, untouched. v7's item 31 `size_cap_paydown` and the `api_v1.py` note survive in non-conflicting hunks |
| 7 campaign-bookkeeping files → `docs/archive/v7next/` | D-02 | archive move | `docs/archive/v7next/` | `C6_REVIEW_PACKET.md`, `DESIGN_RC_AUDIT_SCOPE.md`, `DESIGN_RESOLVED_MODEL_TARGET.md`, `DESIGN_TYPED_ORGAN.md`, `LEDGER_CORRECTIONS.md`, `MIGRATION_PROJECTION.md`, `WINWAVE_CLASS_REGISTRY.md`. Verified present in the archive directory |
| `tests/test_legacy_timeout_retirement.py` skip-prefix `+"docs/archive/"` | D-02 | S2 (required) | `tests/test_legacy_timeout_retirement.py` | `LEDGER_CORRECTIONS.md` contains a retired timeout key and no longer matches the `docs/v7next/` prefix after the move. Verified on line 130 |
| two upstream-base `D-nn` labels de-labelled | — | S2 | `docs/ARCHITECTURE.md` | `the durable D22 projection` → `the durable execution projection`; `the former D15 api pin` → `the earlier api-row-only pin`. Upstream's own `(owner R2, …)` / `(owner R52)` attributions were LEFT in place — they are upstream's bytes, the operative gate scores them 0, and stripping ~20 of them is an unrequested rewrite of the base D-02 says to take verbatim. Disclosed rather than decided silently |

---

## 6. Root operator, post-lane

These are the relocations and repairs the root operator made after the five lanes finished:
ratchet paydown, cross-lane requests, and test adaptations the lanes did not own.

| upstream symbol or hunk | upstream commit(s) | class | v7 home (file) | proof / note |
|---|---|---|---|---|
| `_workspace_write_candidates` walker + 2 predicates (93 lines) | a7f7ce9a, fcf0bc10, 872a9c1b | S1 relocation (ratchet paydown) | `ouroboros/tools/write_shape.py` | `registry_guards.py` had reached 1652 lines (>1600 hard cap) purely from this absorption. `write_shape.py` is the per-segment write-shape owner, so the walker has a reason-to-change there independent of the cap. `registry_guards.py` now 1558; `registry.py` re-exports from the new owner |
| `_no_deliverables_decision` | a7f7ce9a, fcf0bc10 | S1 relocation | `ouroboros/tools/write_shape.py` | same move; imported back by `registry_guards.py` |
| `_directory_change_argv` | a7f7ce9a, fcf0bc10 | S1 relocation | `ouroboros/tools/write_shape.py` | same move |
| `_lane_writer_targets(raw_cmd)` named out of `_run_shell_safety_check` | a7f7ce9a, 2a431b82 | S2 (function-size paydown) | `ouroboros/tools/registry_guard_process.py` | the function had grown to 310 lines. The extracted 50 lines are upstream's own "ONE per-segment writer-target SSOT for this lane" — a named contract, not a passthrough — and use the same handle idiom. `_run_shell_safety_check` now 269 lines; `_lane_writer_targets` 50 |
| `_admission_names` → `admission_names` | 1bd71856 | S1 relocation (ratchet paydown) | `ouroboros/project_naming.py` | `gateway/tasks.py` had crossed 1600 by the growth of BOTH sides. `project_naming.py` is the title-derivation owner and the D11 → D17 direction was already resolved; `gateway/tasks.py` now 1593. Upstream's `tests/test_headless_task_title.py` re-pointed. This is the O3 row in §8 |
| cost-breakdown endpoint family (compat buckets/groups + `/api/cost-breakdown`) | d4fd933c, dcea27d6, b7a73355 | S1 relocation (ratchet paydown) | `ouroboros/gateway/cost_breakdown.py` (new, 162 lines) | `gateway/history.py` had crossed 1600 by the growth of both sides; now 1465. `history.py` re-exports `make_cost_breakdown_endpoint` with a `# noqa: F401 — historical import path (router)` so `gateway/router.py` still resolves. Component-map row and `domains.toml` D11 entry added |
| 12 one-line re-export statements → one explicit statement | — | S2 (ratchet paydown) | `ouroboros/outcomes.py` | `_outcome_tool_errors` is a leaf module; the twelve separate `from ouroboros._outcome_tool_errors import X as X` lines became one explicit statement. Not one name was removed. `outcomes.py` 1612 → 1590 |
| `_REVIEW_SUBSTRATE_PATHS` literal compaction | — | S2 (ratchet paydown) | `scripts/run_external_review.py` | three entries per line; not one entry removed. 1606 → 1558 |
| `api_tasks_create` project_id block | 1bd71856 | S1 (upstream verbatim at the cap) | `ouroboros/gateway/tasks.py` | 302 → **300** lines exactly at the function cap, using upstream's own one-line forms: the ingress gate expressed through a walrus, the comment preserved. Verified: 300 lines |
| `_schedule_task` | 68eab3ea, d48904a5, 558a5c65 | S1 (upstream verbatim at the cap) | `ouroboros/tools/control_scheduling.py` | 303 → **300** lines using two upstream one-line forms the lane had not carried across. Verified: 300 lines |
| `usage_accounting.py` dead `import os` | 2ed94f78 | S1 (auto-merge artifact) | `ouroboros/usage_accounting.py` | both `os.environ.get("TOTAL_BUDGET", "200")` reads were replaced by `resolve_total_budget_usd()`, leaving the import dead and `ruff --select F` red. Deleted on G1's and G2's request |
| `tests/test_review_substrate_acceptance.py` 8 patch targets | a3599ecd | S1 | `tests/test_review_substrate_acceptance.py` | applied on G2's request (§2.7) |
| `tests/test_persistence_inventory.py` + `docs/PERSISTENCE.md` retired-file row | a3599ecd | S2 | `tests/test_persistence_inventory.py`, `docs/PERSISTENCE.md` | the writer of `state/reviewer_slot_api_fallback.json` was deleted by the auto-merge, so a v7-only test and the persistence doc both lied. PERSISTENCE gained 2 rows for upstream's new durable plans and lost 1 retired row; the pinned scan population moved 283 → 285 |
| `differential-golden` classification corpus | — | S1 (regenerated) | `tests/fixtures/legacy_tool_classification_0f715831.json` | upstream added two identifiers (`INTEGRATE_APPLY_NO_OP`, `SAFETY_SUBJECT_TOO_LARGE_BLOCKED`); the golden was regenerated by the standard recipe (the old tree at `0f715831` answering the new corpus) |
| history finality test `_schema_version: 1` | dcea27d6, ab0112a6 | S2 | upstream's history finality test | v7's ABI 7.0 loader is strict (Q8=B), so the stamped row is required |
| `tests/test_llm_extraction.py`: `_build_remote_candidate` in `_MIXIN_OWNERS`, recomputed member digest + provenance note | b7a73355 | S2 | `tests/test_llm_extraction.py` | applied on G1's request; the suite is red without it |
| `tests/test_module_handle_extraction.py` row for `tools/review_multi_model.py` (`+_owner_deadline_at`, `−emit_review_usage`) | 6ff83c60, 761cf1b9 | S2 | `tests/test_module_handle_extraction.py` | applied on G1's request after the review organ settled |
| `queue_timeouts` module-handle pin | 68eab3ea | S2 | `tests/test_module_handle_extraction.py` | follows the `_queue().coerce_chat_identity` reads |
| `docs_sync` hot-store count seven → nine | 4a7fa18b | S2 | `tests/test_docs_sync.py` | follows the `hot_store_growth_notes` docstring correction (§4.4) |
| `tests/test_review_prompt_caching.py` docstring `loop.seal_task_transcript` → `context_fit.seal_task_transcript` | 743597ee | S2 | `tests/test_review_prompt_caching.py` | cosmetic, follows D-18(a) |
| `review_native_episode.py` writer-set prose path `tools/core.py` → `tools/core_file_tools.py` | a6def41f | S2 | `ouroboros/review_native_episode.py` | docstring only; the code was already correct |
| `tests/test_gateway_widgets.py` ×2 | afe1fc4f | S2 | `tests/test_gateway_widgets.py` | re-expressed through staged publication with a real reviewed skill (§3.8) |
| masked-green assertions ×4 | 32628385 | S2 | `tests/test_shell_run_shell.py` | re-expressed on typed publication (§3.8) |
| transport-free contracts test | — | S2 | upstream's contracts test | re-pointed to `ouroboros/gateway/contracts.py`, v7's one envelope owner |
| `ouroboros/domains.toml` + 4 module rows (D16, D11, D11, D06) and `--write` | — | manifest regeneration | `ouroboros/domains.toml` | the four new/relocated modules registered; see §7 for the D06 → D07 direction |
| `docs/v7next/FROZEN_CONTRACTS_INVENTORY.md`, `DATA_LAYOUT_INVENTORY.md`, `FACADE_INVENTORY.md` | — | generated | `docs/v7next/` | regenerated after all lanes; the §11.1 table header was brought to the extractor's form (`Contract \| File \| Anchored by`) |
| `ouroboros/size_ratchet_manifest.py` | — | generated | `ouroboros/size_ratchet_manifest.py` | regenerated. 16 band rationales, of which 5 are upstream's verbatim; the rest are v7's own. `acceptance_dialogue.py` and `provider_catalogs.py` rows are gone (verified absent from `BAND_PATHS`); `loop_forced_finalization.py` (1041 lines) carries a band rationale |

---

## 7. Owner forks and disclosures

Everything in this section is a decision the owner may want to reverse. Each one was
implemented provisionally in upstream's direction (or in the direction the plan recorded) and
is flagged here rather than settled silently.

### D-18 — three symbols with two destinations

Upstream and v7 independently relocated the same three surfaces to different owners. One
destination had to win for each; no duplicate definition survives, and every historical import
path stays alive.

| symbol | upstream destination | v7 destination | decided | how the other path stays alive |
|---|---|---|---|---|
| `seal_task_transcript` (upstream 743597ee) | `ouroboros/context_fit.py` | `ouroboros/loop.py` | **`context_fit`** | v7's 45-line copy deleted; `loop.py` re-exports, so `from ouroboros.loop import seal_task_transcript` and `_loop().seal_task_transcript(...)` both resolve |
| `skill_names_touched_by_trace` (upstream 0463c6bb/f8d4408c) | `ouroboros/skill_readiness.py` | `ouroboros/loop_nudges.py` | **`skill_readiness`** | `loop_nudges.py` aliases it as `_skill_names_touched_by_trace`; `loop.py`'s barrel entry now resolves to the one `skill_readiness` object. No test patches the name |
| `fetch_openrouter_pricing` / `fetch_cloudru_pricing` (upstream 1a525dbd) | new `ouroboros/provider_catalogs.py` | `ouroboros/llm_pricing.py` | **`llm_pricing`**; `provider_catalogs.py` **deleted** | `llm.py` re-exports both, as v7 already did. The bodies are identical apart from a module-level vs local logger; shipping both files would be two pricing SSOTs |

`ouroboros/provider_catalogs.py` is verified absent from the worktree, from `BAND_PATHS` and
from `BYTE_DEBT`.

### D-19 — v7's D15 ("task acceptance stays API-only") retired in favour of upstream's later R2

Upstream's R2 (`a3599ecd` / `daf37e99`, 2026-09-01) says task acceptance reads the configured
triad rows through one builder. That is a *later* owner decision than v7's D15, so it was
implemented and D15's apparatus removed. **This is an owner decision, not a merge mechanic.**

Exact v7 surfaces removed:

| surface | v7 location | what it did |
|---|---|---|
| `api_fallback_disclosure(config)` | `ouroboros/reviewer_slot_config.py` | returned `{"triad": OPENROUTER_REVIEW_DEFAULTS["triad"]}` when no triad row was api-deliverable, so API-pinned acceptance could substitute the shipped default models |
| the D15 paragraph of the module docstring | `ouroboros/reviewer_slot_config.py` | "Task acceptance stays API-only by owner decision (D15)" |
| the D15 clause in `reviewer_slots.__doc__` | `ouroboros/reviewer_slot_config.py` | the same claim on the model-list builder |
| the D15 paragraph in the test module docstring | `tests/test_reviewer_slot_config.py` | "…runtime projection for the API-pinned surfaces (D15)" |
| `test_all_delegated_commit_surface_discloses_the_api_fallback` | `tests/test_reviewer_slot_config.py` | pinned the fallback disclosure text and the durable `reviewer_slot_api_fallback.json` record |
| `test_acceptance_rows_stay_api_even_when_triad_routes_delegate` | `tests/test_review_agent_session_route.py` | pinned "acceptance rows are always API_CHAT" |
| the D15 comment in `test_all_delegated_triad_projects_no_api_model…` | `tests/test_reviewer_slot_config.py` | "The API-only task-acceptance surface falls back to shipped defaults" |

Proof that keeping HEAD was not viable, independent of the decision: after the auto-merge,
`api_fallback_disclosure`'s three collaborators (`_fallback_warning_text`,
`reviewer_slot_api_fallback_warning`, `_record_api_fallback_substitution`) were gone, its only
reference in the tree was its own `def`, `__all__` no longer exported it, and the auto-merged
`test_the_retired_acceptance_api_pin_apparatus_is_gone` (`fb8073b6`) asserts the name absent.
Restoring D15 would require un-doing four already-auto-merged production surfaces that call
`triad_delivery_slots(role_hint="task acceptance")`.

**Live behavioural consequence for an install:** a triad whose rows are all delegated
(agent_session) or all native-retrieving now runs task acceptance *on those rows*, spending
subscription minutes or native-episode API rounds per substantive task, instead of silently
substituting three shipped default API models. The one-time R12 save-time disclosure
(`acceptance_delivery_disclosure` + `_ACCEPTANCE_API_PANEL_MEASURED`) is what the owner now
sees. Root-agent-called and `off`-mode acceptance stay packet-only — `tools/review.py` filters
on `if not getattr(slot, "retrieves", False)`, upstream's own R2 carve-out.

`api_fallback_disclosure` is verified absent from the tree.

### D-20 — `ouroboros/acceptance_dialogue.py` stays deleted, with no shim

Upstream grew the file from 853 to 1253 lines over 20 commits while v7 had split it into
`loop_acceptance.py` and `loop_acceptance_review.py`. The file was left deleted: 20 of its 28
top-level names were unchanged upstream and already had a v7 home (S3), 4 changed (S1,
transplanted), and 4 were new and homeless (S1, placed — §1.4). No compatibility shim module
was created; the seven upstream tests that imported from it were adapted to v7's leaves
(§1.9). Verified: the file is absent, and there are zero references to the
`ouroboros.acceptance_dialogue` module anywhere in the tree.

### D-21 — `ouroboros/tools/registry.py` is protected: exactly six facade lines changed

`diff -u` against `git show 18b9832e:ouroboros/tools/registry.py` reports exactly 6 changed
lines (313 lines vs 309). No function body, no `__all__`, no comment was touched. The six:

1. `+    directory_destination_child_name,  # noqa: F401 — historical facade surface` — added to the `from ouroboros.shell_parse import (…)` block
2. `+    sequential_effective_cwds,  # noqa: F401 — historical facade surface` — same block
3. `+    interpreter_inline_code,  # noqa: F401 — historical facade surface` — added to the `from ouroboros.tools.shell_guards import (…)` block
4. `−    directory_destination_child_name,  # noqa: F401 — historical facade surface` — removed from the `shell_guards` block (the symbol moved to `shell_parse` upstream; the merged `shell_guards.py` no longer defines it)
5. `+    writer_target_rows,  # noqa: F401 — historical facade surface` — added to the `shell_guards` block beside the retained `writer_target_tokens`
6. `+from ouroboros.tools.write_shape import _workspace_write_candidates  # noqa: F401 — historical facade surface`

Lines 1–5 are the lane's; line 4 combined with line 1 was the import-time break for the whole
tool layer. **Line 6 is the root operator's addition and the disclosure**: D-21 enumerated
exactly what may change in this protected file, and this line goes past that enumeration. It
was added because upstream's auto-merged `tests/test_workspace_write_shape.py` does
`from ouroboros.tools.registry import _workspace_write_candidates`, and after the ratchet
relocation (§6) that was the only name any test imports from the facade that did not resolve.
The alternative — re-pointing that one test line at the owning module — was available and was
not taken. The owner may prefer it.

### `_build_remote_candidate`'s mixin placement (G1)

Upstream's new `_build_remote_candidate` (b7a73355) is one provider-aware candidate builder
that serves both the direct-Anthropic send and `task_pacing`'s prospective wrap-up estimate.
It was placed in `ouroboros/llm_anthropic.py::_AnthropicLaneMixin`, not on `LLMClient` itself.
Reason: 90% of the body is Anthropic payload construction, upstream defines it immediately
before `_chat_anthropic`, and `llm.py` cannot host it — 33 lines would push 732 → 765, over the
750-line pin in `tests/test_llm_extraction.py`. Callers are `_chat_anthropic` and
`task_pacing`, which calls `llm._build_remote_candidate(...)` on the instance and resolves
through the MRO. **This is a lane judgement call, not a briefed decision**: it changes which
mixin owns a cross-lane dispatcher. If the owner would rather have it as a parent member, the
750-line pin has to move first.

### `_process_role` via the loader facade, and `announce_extension_state_change` gaining `"request_failed"` (G3)

Two widenings of v7 seams, both driven by upstream's receipt contract rather than by a merge
mechanic:

- **`extension_liveness._process_role()`** resolves `is_server_process` **at call time through
  `ouroboros.extension_loader`** rather than binding `extension_companion.is_server_process` at
  import. Upstream's receipt tests (`tests/test_skill_widget_surface.py`,
  `tests/test_extensions_api.py`) monkeypatch `extension_loader.is_server_process` and assert
  `state["process"]`; under the v7 split the state functions live in a leaf that had no such
  binding. The rationale in the docstring is that the loader is the seam the whole organ
  answers process identity at — its reconcile decides the announcement direction — and a
  receipt naming a different process than its own announcement would be worse than no receipt.
  The alternative is binding `extension_companion` directly and adapting the two upstream
  tests.
- **`extension_reconcile_queue.announce_extension_state_change` gained `"request_failed"`.**
  Upstream's receipt vocabulary is exactly `{"", "requested", "request_failed"}`; v7's seam
  could only return `""` / `"published:<gen>"` / `"no_skill"` / `"requested"`, so
  `"request_failed"` was unreachable and a swallowed exception was indistinguishable from
  "nothing to do". The single blanket `try` was split into publish and request halves and the
  return vocabulary documented. No caller outside `extension_loader` reads the value (verified
  by grep). Smoked: with `request_extension_reconcile` patched to raise, the announcer returns
  `request_failed`.

### The D06 → D07 direction banked in `ouroboros/domains.toml`

Upstream `761cf1b9` ("Emit one llm_usage row per physical reviewer call") moved reviewer usage
emission out of the review-execution domain into the custody-usage domain. In the merged tree
that is a cross-domain edge: `ouroboros/review_execution.py` is **D06** and
`ouroboros/delegate_custody_usage.py` is **D07** (both verified in `domains.toml`), and
`review_execution.py` now calls `observe_review_usage(self.usage_observer, usage)`, whose
definition lives in `delegate_custody_usage.py`. The direction is recorded in the manifest
rather than being refactored away: the observer is installed by the substrate and fired by the
executor, so the usage row has exactly one producer. This is the same commit whose consumer
half required deleting the second emission in `tools/review_multi_model.py` (§2.1).

### `ouroboros/agent_startup_checks.py` is 1503 lines — out of the band, under the cap

`BAND_MODULE_MAX_LINES = 1500` and `MAX_MODULE_LINES = 1600` (`ouroboros/review.py`). At 1503
lines the file leaves `BAND_PATHS` on regeneration (verified: it is no longer in `BAND_PATHS`,
and it remains in the immutable `BAND_BASELINE_PATHS`). It stays under the 1600 hard cap, so no
"new module debt above 1600 lines" error fires and leaving the band is not itself refused. The
+11 lines are **entirely upstream's auto-merged delta** — the `plan_review_authority_core`
block (+4), the skill-review threshold row (+6) and the import (+1); the lane's own net
contribution to this file is 0 lines. Disclosed because the rationale text this file used to
carry asserted band membership that is no longer true. Its `FUNCTION_DEBT` row is still true:
`verify_restart` is 505 lines (verified).

### D-02 archive-scope correction — 8 test-bound docs stay, 7 bookkeeping files archived

D-02 sends `ADOPTION_v7next.md` and `docs/v7next/*` to `docs/archive/v7next/`. Only 7 of those
files are true campaign bookkeeping; **8 are load-bearing** and stayed in place:

| stays | bound by |
|---|---|
| `docs/v7next/FROZEN_CONTRACTS_INVENTORY.md` | `ouroboros/code_intelligence_architecture.py` `FROZEN_INVENTORY_RELPATH` (a runtime constant), `tests/test_architecture_facts.py`, `scripts/regenerate_inventories.py` |
| `docs/v7next/FACADE_INVENTORY.md` | `ouroboros/code_intelligence_architecture.py`, `tests/test_architecture_facts.py` |
| `docs/v7next/DATA_LAYOUT_INVENTORY.md` | `scripts/regenerate_inventories.py`, `tests/test_generated_inventories.py` |
| `docs/v7next/DOMAIN_QUOTIENT_REPORT.md` | `scripts/v7next_domain_report.py`, `tests/test_docs_sync.py`, DEVELOPMENT prose |
| `docs/v7next/DESIGN_MODEL_VISIBLE_LOGGED.md` | `tests/test_docs_sync.py`, `tests/test_model_send_seal.py` |
| `docs/v7next/DESIGN_USAGE_COMPACTION.md` | `tests/test_lockfile_helpers.py`, `tests/test_usage_compaction.py` |
| `docs/v7next/ABI3_GATEWAY_ALIAS_INVENTORY.md` | `tests/test_gateway_abi3_removals.py` |
| `ADOPTION_v7next.md` (repo root, not `docs/v7next/`) | `scripts/v7next_adoption.py` `MANIFEST = REPO_ROOT / "ADOPTION_v7next.md"`, `tests/test_v7next_adoption.py`, and the `allowed` set of `tests/test_legacy_timeout_retirement.py` |

Archiving `ADOPTION_v7next.md` would be a code change across a script and two tests — outside a
docs lane — so it was **not** decided silently. Either it stays at the root (current state) or
the owner authorizes moving it together with those three call sites.

Verified on disk: `docs/v7next/` holds exactly the 7 bound files listed above,
`docs/archive/v7next/` holds exactly the 7 archived files, and `ADOPTION_v7next.md` is at the
repo root.

---

## 8. O3 — upstream-added symbols absent from the merged tree

The O3 oracle enumerates every top-level `def`/`class` added between `a76961de` and `23ab428f`
and requires each to exist somewhere in the merged tree. It checked 176 symbols; 4 are absent.
Each is S3 with proof — none is a lost upstream change. (O4: 0 conflict markers. O5: 158/158
upstream test files present, 0 absent. O6: 13 net-removed upstream symbols, 0 surviving.)

| absent symbol | upstream commit(s) | class | what replaced it | proof |
|---|---|---|---|---|
| `_deprecated_pacing_aliases` | a58d6afd, 4d35f521, de71a16a, 9117efc9 | **S3 — retired ABI** | nothing; the alias surface itself is gone | upstream rebuilt the deprecation machinery around `until_deadline` / `stall_rounds_threshold`; v7 deleted both aliases under the 7.0 ABI window (Q10=A), and `tests/test_abi5_q10_removals.py` covers the removal. Taking upstream's helper would re-introduce a retired 7.0 ABI surface. The sibling names it served survive: `_supplied_budget_profile`, `observe_budget_profile` and `resolve_budget_profile` all exist in `ouroboros/task_pacing.py`, so `delegate_supervision.py`'s call still resolves. Verified: no `def _deprecated_pacing_aliases` anywhere |
| `_preserve_result_meta` | 32628385 | **S3 — superseded** | `ouroboros/tools/tool_result.py::_wrap_run_script_process_result` | that function performs the identical job: it rebuilds the exact three run_script text framings and republishes the inner typed base via `_replace_tool_result`, promoting `code` to `ARTIFACT_OUTPUT_UNDECLARED` when an audit note is present and the base was `ok`. Upstream's helper existed only to copy `result_meta` off the retired `ToolResultText` in `tools/result_envelope.py`, which v7 deleted; resurrecting it would reintroduce the untyped envelope path. `tools/tool_result.py` needed no edit. Verified: `_wrap_run_script_process_result` exists, `_preserve_result_meta` does not |
| `_validate_runtime_ui_render` | b2bcd659 | **S3 — renamed public in its owner leaf** | `ouroboros/extension_ui_validation.py::validate_runtime_ui_render` | the private name had two importers (`extension_child_catalog.py`, `extension_plugin_api.py`), so it belongs in the single natural owner beside `validate_ui_render`, not in the loader facade. It is public there and in `__all__`; both importers import it `as _validate_runtime_ui_render`, so the historical local spelling is unchanged. Verified: `validate_runtime_ui_render` exists in `extension_ui_validation.py`; no `def _validate_runtime_ui_render` anywhere |
| `_admission_names` | 1bd71856 | **S3 — renamed public in its owner module** | `ouroboros/project_naming.py::admission_names` | the helper derives a run's title and suggested name at admission; `project_naming.py` is the title-derivation owner and the D11 → D17 direction was already resolved. Moved there by the root operator to bring `gateway/tasks.py` back under 1600 lines (1614 → 1593); `gateway/tasks.py` imports it by the public name and calls it at the admission site; upstream's `tests/test_headless_task_title.py` was re-pointed. Verified: `def admission_names` in `project_naming.py`, imported and called in `gateway/tasks.py`; no `def _admission_names` anywhere |

---

## 9. How this ledger was verified

Every path in the tables was checked to exist in the worktree, and every symbol was resolved
by parsing the file's AST and looking for the name among its top-level definitions, class
members, assignments and imported bindings. Four symbols did not resolve at top level and were
each located by hand and recorded at their true site: `outcome_phase` (a function-local import
in `post_task_synthesis.py`), `_forced_delegation_note` (`loop_nudges.py:404`, not the leaf the
lane report implied), `_outcome_tool_errors` (a leaf *module*, `ouroboros/_outcome_tool_errors.py`,
re-exported from `outcomes.py`) and `_target_mismatch_verdict` (nested inside
`_handle_external_workspace_integration`). Every short SHA cited was resolved and confirmed to
lie in `a76961de..23ab428f` (407 commits); one attribution in a lane report — `49e6359c` for
`segment_write_shape` — is a v7-line commit and was corrected to `2a431b82` / `fcf0bc10` in §3.6.
Line counts, function lengths, the six-line `registry.py` diff, band membership, the absence
of the retired names and the docs directory layout were all read from the worktree.

Nothing in this ledger is asserted from a lane report alone. Statements that could not be
proved are marked **UNVERIFIED** in place; there are none remaining at the time of writing.

### F2 absorption — corrections after the review wave (9698e2e0)

| # | row | correction | proof |
|---|---|---|---|
| 1 | §3.5 `_masked_green_disclosure` home | lives in `ouroboros/tools/shell_audit.py` (imported into `tools/shell.py`), not `tools/shell.py` | `grep -n "def _masked_green_disclosure" ouroboros/tools/shell_audit.py` |
| 2 | §3.2 `_read_module_sources` home | verbatim body is `extension_ui_validation.read_module_sources` (exported), imported by `extension_plugin_api.py` and `extension_child_catalog.py` | `grep -n "def read_module_sources" ouroboros/extension_ui_validation.py` |
| 3 | §1.9 `seal_task_transcript` "import resolves" | it resolved only through the shared venv's editable install of the live repo; `context_fit.py` imported the retired `ouroboros.delivery_protocol` — fixed to `loop_messages._extract_plain_text_from_content`; hermetic runs now strip that finder | review wave (grok lane, lens D) + hermetic import smoke |
| 4 | web tests absent (`web/tests/chat_ledgers.test.js`, `web/tests/skill_review_detail_store.test.js`) | deleted by upstream itself before 23ab428f (their subjects were dropped); not a merge loss | `git cat-file -e 23ab428f:<path>` fails for both |
| 5 | masked-green gate | the disclosure gates on the trusted process fact `exit_code == 0`, not on the typed status, so the undeclared-output and artifact-error publications (exit-0, non-ok codes) keep upstream's disclosure | lens B M-1; `tests/test_shell_run_shell.py::TestMaskedGreenDisclosure::test_masked_green_with_undeclared_output_still_discloses` |
| 6 | §2.7 "8 × `setattr(rs, "reviewer_slots")` → `triad_delivery_slots`" | only the four single-line sites were converted by the merge; the four multi-line sites (tests/test_review_substrate_acceptance.py ~:181,242,315,523) stayed on the dead name until the review wave (lens A F1) converted them | `grep -c '"reviewer_slots"' tests/test_review_substrate_acceptance.py` → 0 after the fix |
| 7 | §2.1 `emit_review_usage` S3 proof | the cited reader (`review_multi_model`) no longer reads it; the live reader is `tools/preflight_review_run.py` through the `_car()` handle (`claude_advisory_review.py`) | `grep -n emit_review_usage ouroboros/tools/preflight_review_run.py` |
| 8 | §4.5 `api_tasks_create` "v7's fuller comment kept" | false — the block is upstream's bytes; one upstream three-line comment was folded to one line during the 300-line paydown | `git diff 23ab428f -- ouroboros/gateway/tasks.py` |
| 9 | §1.9 `test_acceptance_floor_admission.py` | undisclosed: upstream's `until_deadline` count-axis test was dropped (retired alias, Q10=A pin in tests/test_abi5_q10_removals.py); the surviving `adaptive` test carries the Required+Blocking control | tests/test_acceptance_floor_admission.py NOTE above the surviving test |
| 10 | §3.2 vs §6 `control_scheduling` reflows | the code carries upstream's one-line forms (§6 is right, §3.2's "not carried" is stale) | `grep -n "status_drive_root, root_cost_ceiling_usd" ouroboros/tools/control_scheduling.py` |
| 11 | O3 absent-by-rename | `_read_module_sources` → `extension_ui_validation.read_module_sources` (fifth rename the O3 list omitted) | `grep -n "def read_module_sources" ouroboros/extension_ui_validation.py` |
| 12 | `complete_custody_rows` | an unreachable trailing `return rows` (auto-merge residue after v7's try/with return) was removed | lens A F2 |
| 13 | §7 `_process_role` | reads `extension_companion.is_server_process` at call time (the owner), not the loader facade; the loader still stamps `state["process"]` on its own receipts from the same owner, so the two agree by construction | `ouroboros/extension_liveness.py:113-123` (lens C MINOR-1) |
| 14 | test patch reachability (upstream-new tests) | `tests/test_tree_cost_ceiling.py::test_wire_recovery_matches_physical_candidate` must patch `ouroboros.llm_attempt.prepare_wire_payload_for_send` (the send binds it there), not only the `ouroboros.llm` re-export | lens C MAJOR-1; fixed in the review-wave batch |
| 15 | S25 (`tests/system_e2e/test_system_scenarios_w5.py`) pinned the v7 terminal for an undisposed captured patch (`infra_failed` → `failed`) | upstream 09ac51b2 (P3: custody-only settlement receipts) made that terminal a DISCLOSED custody debt — `completed`, execution axis = the model's own work, reason `delegated_custody_unreconciled`, `objective.warnings` + `delegated_runs_unreconciled` on the row; the conflict itself is still refused typed (`INTEGRATE_CONFLICT`, `baseline_drift`, material kept). S1: the scenario follows upstream; the v7 unit pins were already upstream's (battery #3 green) | reproduced on iso root (`status=completed`, `objective.warnings=["delegated_custody_unreconciled"]`), then S25 green |
| 16 | S21 (`tests/system_e2e/test_system_scenarios_w4.py`) created an API task with `chat_id=1` (Main) as its lineage | upstream's ingress capture rule (68eab3ea + fed16935, `supervisor/log_addressing.ingress_chat_id`) refuses any chat but the hidden partition for a task with no registered ACTIVE project, and chat 0 is "no lineage" for terminal delivery (`lineage_chat_id` → the typed handoff row, as S7 pins). S1: the scenario now registers a file-less project and is admitted into its thread; the receipt lands in that chat | probe: `ProjectThreadConflict("chat_id is not available to a task with no registered active project…")`; S21 green after the change |
| 17 | `process_custody.spawn_supervised` docstring + ARCHITECTURE component row said the ledger record lands "BEFORE" the child can be orphaned | the write follows `Popen`; a hard kill of the spawner inside that window leaves an unledgered child — now disclosed as the residual in both places (delta review, codex MAJOR-2) | `ouroboros/process_custody.py:168-171`, `docs/ARCHITECTURE.md` process_custody row |
| 18 | `tests/test_contracts.py` progress_meta scan claimed to see `cancel_physical_task_id` | the collector only followed nodes literally named `progress_meta`; the emitter hands over `progress_meta=incident_meta`. The scan now resolves such aliases and pins the key explicitly (delta review, codex MAJOR-1 / fable N1) | `tests/test_contracts.py::_collect_literal_progress_meta_keys` |
| 19 | `tests/system_e2e/harness.ModelGate` expiry | `Event.wait()` result was ignored: an expired hold silently resumed the round. Expiry now sets `timed_out` and raises `TimeoutError` in the request thread; S26 asserts the flag; a default-lane test pins the path; the real-port gate test carries `serial` (delta review, codex MAJOR-3/MINOR-6, fable N2) | `tests/system_e2e/harness.py`, `tests/system_e2e/test_system_scenarios_w6.py` |

## From the F3-C lane (base c1a4b2bc = rc.9, 2026-09-04) — OPERATOR deferrals of 7.0 (owner decision D-14)

Bookkeeping over the ten post-release rows: no runtime semantics, no protected file. The three
rows that need an owner word — W4-F3, W4-F4, DEFER-SPEC64-PATHS — were left exactly as they are;
the F3 owner batch asks. Every correction below was read from the rc.9 tree.

| # | where | correction | proof |
|---|---|---|---|
| 1 | ADOPTION Notes + this file, «Rows added (commit b89b9bd2)» | «W4-F3/W4-F4 get no row» → both ARE rows since d348ea46 (2026-09-02); the sentence outlived that commit by two days past a green bar | `git log -S'\| W4-F3 \| plan-item \|' -- ADOPTION_v7next.md` = d348ea46; `scripts/v7next_adoption.py --release` rc 0 on both days |
| 2 | scripts/v7next_adoption.py (class fix) | the validator read rows only, never the prose: `manifest_prose` + `_prose_id_errors` now resolve every id-shaped token outside the table against the table by the table's own id grammar (the declared form for a rowless id is a `No-row ids: …` line) — a phrasing-independent check, not a keyword gate on «gets no row» | red-first pins in tests/test_v7next_adoption.py: `- No-row ids: <any row>` and an undeclared ghost id both turn the bar red |
| 3 | scripts/v7next_adoption.py (record comment + authority lint) | the comment over `DEFERRED_OUT_OF_V70` called E2/E3 and spec §6.4 «operator disclosures without an owner decision yet» beside OWNER values (batch №13 items 2 = A / 8 = A); rewritten, and the validator now requires the `owner verbatim «…»` quote on OWNER rows and refuses it on operator-disclosed rows, so the record and the row cannot drift apart again | red-first pins on both directions |
| 4 | ADOPTION row ABI-8 | «(owner one-line confirm queued)» → the confirm arrived on 2026-09-01: owner verbatim «6. ок» (batch №7, [A-BATCH-7-ANSWERS], «№6=ок: ABI-8 подтверждён в пост-релизный бэклог»); the no-«7.1» frame is Q16=A + поправка ([A-V7NEXT-BATCH-2], 2026-08-30) | requirements archive |
| 5 | ADOPTION row DEFER-C6-RESIDUALS | «the test-suite cap is an owner question (C6-TESTCAP)» → closed by batch №13 item 11 = A; the split landed (tests/test_usage_compaction.py 900 + tests/test_usage_compaction_archive.py 660 lines, was 1600/1600); open post-release: `ouroboros/platform_layer.py` 1587 → ≤1500 lines + issue | `wc -l` on rc.9; ouroboros/size_ratchet_manifest.py:53 (BAND_BASELINE_PATHS) |
| 6 | ADOPTION row DEFER-E2E-PAID-LANE | «has never been executed … the four scenarios are unverified» stood glued to «EXECUTED BY OWNER DECISION … E13 GREEN … E1 GREEN» with two `residual:` clauses; the stale opening is gone and the row says what «2. A» is authority for (the run) and what the E2/E3 remainder is (a structural block — a logged-in Claude account is the owner's act, which the operator may not perform); the OWNER value in `DEFERRED_OUT_OF_V70` is unchanged | [A-BATCH-13-ANSWERS] «2=A: прогнать платную E-полосу (E1–E3, E13) один раз» |
| 7 | this file, w4 findings table, W4-F3/W4-F4 evidence cells | file:line anchors moved by the F2 relocation: supervisor/evolution_lifecycle.py :1362-1368 → :1437-1438 (marker write at :1457); supervisor/update_merge.py :293-304 → :294-305 | `grep -n OUROBOROS_EVOLUTION_AUTO_RESTART supervisor/evolution_lifecycle.py`; `grep -n "def create_rescue_local_ref" supervisor/update_merge.py` |

Verified on rc.9, not changed — the batch-13 lanes the plan named all landed: DEFER-TYPED-PROC-5
(row done/F6; tests/test_process_signal_observability.py), W4-F1 and W4-F2 (rows done/F6),
DEFER-E2E-DELEG-MUT (row done/F4; tests/system_e2e/test_system_scenarios_w5.py), the C6-TESTCAP
split (item 5 above).

Held for the owner batch, with the facts that batch must carry:

- W4-F3: the obvious fix (write `pending_restart_verify.json` before the auto-restart check) breaks
  the determinism lever S22 depends on — tests/system_e2e/test_system_scenarios_w4.py:822-826 sets
  `OUROBOROS_EVOLUTION_AUTO_RESTART=false` precisely to make the crash window stable. A second
  writer of the same marker exists (ouroboros/tools/control_runtime.py:69-87, the agent-callable
  `restart` with an `evolution_claim`), so the row's «structurally unreachable» is scoped to the
  supervisor auto-restart path, not to every install.
- W4-F4: the disclosure sentence belongs in docs/ARCHITECTURE.md, not docs/PERSISTENCE.md —
  tests/test_persistence_inventory.py:764 anchors that inventory both ways to the DATA root, and a
  git ref under repo/.git matches no scanned writer path.
- DEFER-SPEC64-PATHS: the row's «seven domain modules» are FOUR silent fallbacks
  (ouroboros/tools/browser.py:336, ouroboros/tools/evolution_stats.py:17,
  ouroboros/gateway/files.py:802, ouroboros/server_process.py:18) plus TWO live-repo pytest fuses
  that must never be removed — supervisor/git_ops.py:69 (protected file) and
  supervisor/evolution_lifecycle.py:912 — plus the packaged composition root
  ouroboros/packaged_cli.py:116; the `SimpleNamespace` anchors :241/:681 are :242/:685 on rc.9.
  A post-release lane sized from the row's current text would walk into the fuses.

Cross-lane hand-over: DEFER-BROWSER's hook is
tests/system_e2e/test_system_scenarios.py::test_interface_stubs_refuse_instantiation_until_their_lanes_land,
which pins that `PlaywrightUIClient()` REFUSES. The lane that lands the client (plan Ф4-B) must
flip that row to `done` and replace the pin in the same commit — the validator resolves hooks for
`done` rows only and cannot see the collision.

### Second absorption (62f87cc9, upstream db6d7cf8) — corrections after the review wave

| # | claim | correction | evidence |
|---|---|---|---|
| 20 | commit message of 62f87cc9: `loop_llm_call.py` lands **verbatim** | not byte-verbatim: v7's typed `ProviderPolicyRefusal` branch (the provider-policy refusal classified as its own kind, `loop_llm_call.py` ~:31 and ~:735-746) is PRESERVED over upstream's body — an S3 preservation with proof: `tests/test_llm_typed_policy_refusal.py:270` pins it; upstream has no equivalent and the branch is unreachable for upstream's inputs. Removing it needs an owner decision. Upstream's own ± lines in the range all landed. | codex delta review finding #3; `git diff db6d7cf8..62f87cc9 -- ouroboros/loop_llm_call.py` |
| 21 | `ouroboros/update_letter.py` joined the tree without a domain row | assigned to D12 "Settings & configuration" beside `update_channels.py` (a human assignment; `scripts/check_domains.py --write` refuses to invent one); `context.py:55` imports the projection module-level, which the regenerated `[graph]` records | grok finding #1; commit 591e80f5 |

## From the W4-F3 lane (base 2a6bdb22 = F3-C tip, 2026-09-04) — owner decision «5. A»

The F3 owner batch item 5 («W4-F3 fix now (always write the marker)») answered «5. A». Runtime
delta, no protected file:

| # | where | change | proof |
|---|---|---|---|
| 1 | supervisor/evolution_lifecycle.py `request_evolution_restart` | the `OUROBOROS_EVOLUTION_AUTO_RESTART` gate moved from the function entry to just before the `restart_request` event: the exact claim marker is written whenever the commit's authority still holds, and the knob skips only the restart (an info line names the manual restart the marker now awaits) | tests/test_evolution_terminal_events.py::test_auto_restart_off_still_writes_the_exact_restart_marker — red on the F3-C tip (the marker file does not exist there), green here |
| 2 | one marker schema | `write_pending_restart_marker` is the single writer of `state/pending_restart_verify.json`; ouroboros/tools/control_runtime.py `_request_restart` (the second writer the F3-C scout named) calls it instead of its own `atomic_write_json` dict. The claim key is present only for a non-empty claim (an empty dict would read as `restart_claim_mismatch` at boot); git-newline stripping is now the helper's, not each caller's. docs/PERSISTENCE.md names the one writer | tests/test_evolution_terminal_events.py::test_both_restart_marker_writers_share_one_schema; the two suites that injected a disk failure / captured the write at `control_runtime.atomic_write_json` now do so at the helper's seam (tests/test_evolution_restart_claims.py, tests/test_task_status_flow.py) — same contract, moved seam |
| 3 | S22 disposition | the scenario's determinism lever (auto-restart off ⇒ no marker) is obsolete by the decision; the ordering it relies on (campaign `waiting_for_restart` write, THEN the marker write) is unchanged, so the markerless crash window is reachable exactly when the crash precedes the marker write. S22 now pins the new contract in generation A (marker written with the exact claim, tree not restarted — the harness runs the server without a launcher, so a restart exit would end the process) and shapes the crash-window durable state after the SIGKILL by removing the marker: the campaign write and the marker write are two separate atomic files, so the resulting state is byte-identical to a crash between them; generations B/C are unchanged (markerless reconcile absorbs exactly once, nothing absorbs twice) | tests/system_e2e/test_system_scenarios_w4.py::test_s22_absorb_kill_recovery_absorbs_once_and_never_twice under OUROBOROS_E2E_DEEP=mock: the E2E harness clones committed HEAD, so a first run over the dirty tree exercised the F3-C tip and failed at the new generation-A pin (red-first for the scenario); the run on the lane commit is the receipt below |
| 4 | ADOPTION row W4-F3 → re-prove/done/F6 with the owner quote; `DEFERRED_OUT_OF_V70` drops W4-F3 (W4-F4 stays operator-disclosed pending its own owner-batch item); docs/ARCHITECTURE.md §evolution names the always-written claim | — | scripts/v7next_adoption.py --release rc 0; tests/test_v7next_adoption.py green |

Receipts on the lane commit: S22 mock lane `OUROBOROS_E2E_DEEP=mock -m integration -k s22` → 1 passed in 190 s (rc 0; generation A logged «Automatic evolution restart is off; the restart-verify marker for bb46116d awaits a manual restart», generation B absorbed once with `verified_by=boot_reconciliation`, one `evolution_tx_reconciled` event, no marker left after C); the same run over the dirty tree before the commit (the harness clones committed HEAD = the F3-C tip) failed at the new generation-A pin — red-first for the scenario. Unit: the two new pins are red on the F3-C tip (2 failed) and green here; touched suites 312 passed (evolution lifecycle/restart claims/redesign, startup hygiene, persistence inventory, task-status flow, control extraction, server shutdown, adoption); ruff F clean on the whole tree; size ratchet --check rc 0 (supervisor/evolution_lifecycle.py 1486 → 1494 lines, inside its band); `scripts/v7next_adoption.py --release` rc 0.

## From the rc.11 delta-review fix lane (base 49dede67 = candidate 20d99508 + the facade-inventory regeneration, 2026-09-04)

| # | claim | correction | evidence |
|---|---|---|---|
| 1 | `ouroboros/size_ratchet_manifest.py` ~:149, band rationale for `ouroboros/tools/claude_advisory_review.py`: «re-enters the band at 1434» | a historical figure, not the live one: the file stands at 1492 lines on this base (the F3 Q6 mandatory-read lane and its siblings grew it inside the 1001-1500 band). Band rationales are immutable between manifests, so the string stays as written and the number is read as the re-entry point it names, not the current size; the live fact is the band check (`scripts/regenerate_size_ratchet.py --check`), and this lane's fix of the advisory preview slot kept the file at exactly 1492 (fable minor, rc.11 delta review) | `wc -l ouroboros/tools/claude_advisory_review.py` = 1492 before and after the lane; `scripts/regenerate_size_ratchet.py --check` rc 0 |
