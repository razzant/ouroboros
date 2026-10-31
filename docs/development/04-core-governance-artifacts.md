# Core Governance Artifacts

BIBLE/Architecture/Development ground identity/structure/procedure; DESIGN names product/interface rules. This chapter governs delivery, exact premises and disclosed views; availability, delivered content and reading/application stay distinct.

### Invariant: Full availability in reasoning flows

Architecture, constitution and procedure flows receive applicable governance as first-class context: inline or complete navigable sources named by their registry row, never silently omitted or dependent on touched files.

**Review surfaces.** `tools/governance_context.py` supplies CODE/coupling briefs for change-review seats, preflight and deep self-review, by layer (`review_body_fact.layer_for`); the three body tiers and the universal core/foreign section: ARCHITECTURE §6 “Governance delivery”.

**Plan review.** The body pack follows declared `affected_paths`, never prose or plan kind (P5): a body plan reaches the installed repository or a recognized copy, or is folderless research with explicit `context_requires_self_body_docs=true` plus `task_design_source` acceptance; Main/cwd or `[]` alone cannot qualify, and foreign or unknown roots get no Ouroboros governance and no maps. Required assembly failures are `PlanPacketError`; unattached evidence stays named; a participant continues only from its own retained delivery and reply, an unreadable source meaning a fresh send disclosed as `continued`, never a borrowed or invented transcript (ARCHITECTURE §6 “Plan construction and review”, `tools/plan_packet.py`).

**DESIGN navigation.** Own-body Main/API planners/consilia and declared children get canonical `docs/DESIGN.md` headings/ranges in `dynamic_head` after the stable prefix. Workspace provenance/`body_fact` select sources; the flag covers folderless self work. Foreign work is excluded. Sessions append prepared map/path/SHA to host instructions after root preparation; retained starts pin retries; canonical work orders unchanged. Plan/change CODE/coupling/preflight use installed governance. Web review keeps full DESIGN within its share, otherwise map/pointer. Missing sources are named. Select originals: maps add no gate, whole-book duty, rule text or reading proof. Tool-less packets need attached excerpts, naming omissions/uncertainty.

The context-delivery registry:

| Flow | BIBLE.md | ARCHITECTURE.md | DEVELOPMENT.md |
|------|----------|-----------------|----------------|
| Main task context (`context.py`) | full tier-0 in every mode | full in Max; book navigation in Low/Nano and for a subagent child | Max: a stable block after the governance prefix for body work (`development_resident`; unknown keeps it), else the chapter map; book navigation in Low/Nano and for a subagent child |
| Change review, Part 1 (`tools/review.py`) | full (API preamble or retrieving task) | tier 3 navigation; packet rows also get sections naming touched files within the inline share | tier 2 within the share: the review protocol and chapters naming touched files |
| ↳ `review_change` by layer (`review_body_fact.layer_for`) | body: as Part 1; core: `not_applicable`, the preamble names no constitution (`review_prompt_text.review_preamble("core")`) | body: as Part 1; core: `not_applicable`, the navigation indexes the subject's documents (`subject_root`) | body: as Part 1; core: the universal `Change Review Checklist` alone, with an empty-by-rule required-source manifest |
| Background consciousness wake-up (`consciousness.py` → `handle_wake_direct`) | = Main task context | = Main task context | = Main task context |
| Preflight (`review_change(surface=preflight)`: one named row over the system repository's worktree) | the `review_change` body row | as BIBLE | as BIBLE |
| The coupling question (Part 2 of the retrieving seat's brief, `tools/review_brief_coupling.py`) | full tier 1 beside the `Coupling questions` section, in every context mode | tier 3 navigation | tier 2 within the usable-window share |
| Skill review (`review_change(subject="skill")`) | full inline (packet) / declared mandatory full read (retrieving) | tier 3 navigation | tier 2: owed chapters + selected contracts |
| Plan review (`tools/plan_review.py`) | full for a body plan (`api_chat` inline, `agent_session` mandatory full read); otherwise none, no map | as BIBLE | not delivered or pointed at; the mind may attach it |
| Deep self-review (`deep_self_review.py`) | full tier 1 on both deliveries | tier 3 navigation | tier 2 within the row's share (ARCHITECTURE §6 "Deep self-review") |

`tools/scope_required_sources.py` sets minimum body reading, never its ceiling. Core manifests stay empty and staged diffs complete. Coverage changes no findings, quorum or permission and buys no repeat; the mind decides further reads. The window sizes delivery, never authority.

Planning resolves targets against `active_repo_dir_for(ctx)`, attachments against the task's readable roots, governance against the system repository; external plans never fall back to reviewing Ouroboros. Exact user-managed installed-skill paths are a classification-only exception: never self-modification or attachable evidence (`denied_path`).

Review history keeps the operative contract, current author/critic statuses and open/deferred/pending facts explicit. Actor `review_notes` may shorten individual reasons or group exact historical decisions, retaining membership and lineage in the existing selected view. No grouping by age or terminal status. Producers project decisions once with mirror references; unknown legacy content stays full. Recheck versions through publication, cold context, new packets and Continue; source gaps preserve recoverable bodies. Invalid optional transfers leave their own attachments full. Unchanged views retain their prefix; all surviving accounts cross continuations. Notes grant no verdict and rewrite no paid replay. Enforcement: `tests/test_review_local_accounts.py`, `tests/test_review_cold_history.py`, `tests/test_review_view_integration.py`. Plan objections: ARCHITECTURE §6 "Plan construction and review".

**Context mode (Nano / Low / Max).** Use the Main row. Max binds Development to the active repository, keeping books until recovery's earlier steps fall short. Known-window no-result-room predictions and provider refusals share recovery; unknown windows send full. Preserve owner Low/Nano sizing, lower starts, tier-0 and P3. Targets cut no chosen read or owner words; rebinds re-cut no fitting sent view (`tests/test_context_economy_window.py`). Mechanism: ARCHITECTURE §6 “Context fitting, retry, and compaction”.

### Invariant: Exact premises with explicit source ownership

Plan from the complete retained room — both speakers, options and answers exact (`dialogue_evidence.py`, `plan_dialogue.py`) — with no independent dialogue byte cap, never from the bounded post-consolidation reader or the owner-directive ledger (`review_evidence._accept_owner_directives`). JSONL records and chat line selectors split on physical LF only, never inside valid Unicode of a message. Enforcement: `tests/test_plan_dialogue_review_regressions.py`.

### Invariant: Compaction must earn its rewrite

Treat the task's dated focus as data, not owner instructions. Strictly read its canonical result, distinguishing absence/failure; update same-round facts and remeasure (`tests/test_self_focus_context.py`).

Known-window no-result-room predictions and typed size refusals use two model-free recovery passes (ARCHITECTURE §6). No helper summarizer: `compact_context` without `working_note` is `authored_note_required`. Actor notes may fold completed working prose/tools without shrinking, preserving governing human words, recorded outward speech, newer arrivals and opaque protocol. Local notes keep positions unless selected for merging; exposure uses physical evidence, not call IDs. Test final sealed requests, cold restoration and both sides of dialogue: `tests/test_compaction.py`, `tests/test_main_authored_context.py`, `tests/test_context_source_view.py`, `tests/test_delivery_dialogue_context.py`.

### Invariant: No silent truncation

When a core governance artifact does not fit the available budget:

- Where the flow requires inline delivery, inability to fit is an assembly failure, not a smaller pack (BIBLE P3): a typed entry names the artifact and the reason, the review does not proceed on the remainder, and the disclosure accompanies the refusal rather than replacing it. Elsewhere the omission is named where the reader sees it (`Reference book source unavailable: …`, `⚠️ OMISSION NOTE`), never silent; a reviewer without ARCHITECTURE.md is not operating with full context.
- Retain exact tool sources before projecting Main batches, without exemptions. Measure host cut reasons inside partial envelopes; preserve outcomes, ranges and source gaps. Producer pages stay outside this frame; read receipts count the consumer projection.
- Book navigation (`context_layout.book_navigation`), heading maps and named pointers index available sources; they do not deliver source bodies. Low/Nano use them, never `[:N]` on a document.
- Bound strings through `utils.truncate_review_artifact` (display previews) or `utils.truncate_within_limit` (a strict wire/prompt bound), never a hand-rolled `text[:cap] + marker`, which loses the anti-waste floor and can return a value longer than its input.
- A list obeys the same rule: a `[:N]` slice carries the omitted count and, where downstream compares an identity, a durable hash or reference for the full set (`_outcome_receipts.receipt_identity_projection`). Bounding a set is allowed; hiding that it was bounded is the P1 violation.

Enforcement: `tests/test_tool_result_delivery.py`, `tests/test_delegated_result_delivery.py` and `tests/test_owner_facing_honesty.py`.

### Invariant: Owner-facing surfaces show the full text

Disclosed truncation (the `⚠️ OMISSION NOTE` marker) protects **LLM context budgets**; it never licenses shortening what the owner reads.

- **Owner/UI-bound surfaces** (chat, task_results projections, review verdicts shown to a person) present the complete text or a reference to a durable full copy (an observability `response_ref`): reviewer rationale is a cognitive artifact (BIBLE P1), and truncating it beside an unreferenced full copy is partial memory loss. Terminal text asserts only recovery facts the round record carries, never a route mechanism the selected transport cannot perform.
- **Model-bound projections** (review packs, context sections, tool-result transport) keep their disclosed-truncation budgets.
- **A cut cheaper than its own marker is forbidden everywhere** (the shared primitive's floor).

Enforcement: `tests/test_owner_facing_honesty.py`.

### Invariant: No "only if touched" gate for core artifacts

Applicable body governance reaches review/reasoning independently of `touched_paths`; `build_touched_file_pack` serves changed files, not eligibility. Presence tests cover delivery; otherwise review-only.

### When adding a new reasoning flow

A new flow that reasons about code structure, system architecture or engineering standards:

1. Explicitly loads `ARCHITECTURE.md` (and BIBLE.md when constitutional reasoning applies).
2. Logs a warning when the file is missing or unavailable — never skips silently; a required artifact that cannot fit fails assembly ("No silent truncation").
3. Adds a test asserting the file is present in the assembled context/prompt. That test is the enforcing surface; CHECKLISTS item 22 (`context_building`, advisory) backstops the review.

---
