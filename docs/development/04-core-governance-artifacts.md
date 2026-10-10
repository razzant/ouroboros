# Core Governance Artifacts

BIBLE, Architecture and Development ground identity, structure and procedure. This chapter governs their delivery, exact premises, earned compaction and disclosed views. Full availability is inline text or complete navigable sources; absence is never full context.

### Invariant: Full availability in reasoning flows

A flow that reasons about architecture, constitution or procedure receives these artifacts as **first-class context** — inline, or through the complete navigable on-demand sources its registry row names — never as opportunistic touched-file inclusions and never silently omitted.

**Review surfaces.** Every change-review seat (both parts of its brief, the preflight's one seat included) and deep self-review take their governance from `ouroboros/tools/governance_context.py`, switched by one fact: the checklist `layer` that `ouroboros/review_body_fact.py` decides (`body_fact` → `layer_for`). The governance root is always the installed system repository, because the installed body's rules execute, never a proposal's copy. The body layer (the subject is Ouroboros's body) inlines tier 1 — BIBLE, the applicable CHECKLISTS sections (`review_helpers.load_checklist_layers("body")`) and the standing disclosures — bounds tier 2 (DEVELOPMENT, DESIGN) by `runtime_limits.REVIEW_GOVERNANCE_INLINE_SHARE`, and delivers tier 3 (ARCHITECTURE) as navigation. The core layer (the subject is another repository) runs the surface's own universal section alone and records BIBLE, the archive, the shared-contract section, DEVELOPMENT, DESIGN and ARCHITECTURE `not_applicable`, so Ouroboros's constitution stays off repositories it does not govern. A pointer gives a packet row no tools or evidence it did not receive. Tiers and bounds: ARCHITECTURE §6 "Governance delivery".

**Plan review.** Plan governance is tiered by one structural fact — a declared `affected_paths` locator resolving under the system repository — never by prose or a plan-kind taxonomy (BIBLE P5). Required governance that cannot be assembled is a typed `PlanPacketError`; unattached evidence stays a named absence. Exact-wave custody is fail-closed: each packet slot continues its recorded transcript, the panel goes out fresh only when no exact artifact reference exists or the roster changed, and an unreadable referenced artifact returns `plan_review_exact_artifact_unavailable` without minting replacement authority. Mechanism: ARCHITECTURE §6 "Plan construction and review", `ouroboros/tools/plan_packet.py`.

The context-delivery registry:

| Flow | BIBLE.md | ARCHITECTURE.md | DEVELOPMENT.md |
|------|----------|-----------------|----------------|
| Main task context (`context.py`) | full tier-0 in every mode | full in Max; book navigation in Low/Nano and for a subagent child | Max: a stable block after the common governance prefix when the active binding targets the system repository, else a visible on-demand pointer; book navigation in Low/Nano and for a subagent child |
| Change review, Part 1 (`tools/review.py`) | full (API preamble or retrieving task) | tier 3 navigation; packet rows also get sections naming touched files within the inline share | tier 2 within the share: the review protocol and chapters naming touched files |
| ↳ `review_change` by layer (`review_body_fact.layer_for`) | body: as Part 1; core: `not_applicable`, the preamble names no constitution (`review_prompt_text.review_preamble("core")`) | body: as Part 1; core: `not_applicable`, the navigation indexes the subject's documents (`subject_root`) | body: as Part 1; core: the universal `Change Review Checklist` alone, with an empty-by-rule required-source manifest |
| Background consciousness wake-up (`consciousness.py` → `handle_wake_direct`) | = Main task context | = Main task context | = Main task context |
| Preflight (`review_change(surface=preflight)`: one named row over the system repository's worktree) | the `review_change` body row | as BIBLE | as BIBLE |
| The coupling question (Part 2 of the retrieving seat's brief, `tools/review_brief_coupling.py`) | full tier 1 beside the `Coupling questions` section, in every context mode | tier 3 navigation | tier 2 within the usable-window share |
| Skill review (`skill_review.py`) | full inline (`api_chat`) / mandatory full source-root read (`agent_session`) | same two classes | same two classes |
| Plan review (`tools/plan_review.py`) | full for a self-modification plan (`api_chat` inline, `agent_session` mandatory full read); otherwise a navigation map plus a named on-demand pointer, never a copy | as BIBLE | not delivered or pointed at; reachable only through a generic `need_evidence` locator |
| Deep self-review (`deep_self_review.py`) | full tier 1 on both deliveries | tier 3 navigation | tier 2 within the row's share (ARCHITECTURE §6 "Deep self-review") |

`tools/scope_required_sources.py` sets minimum body reading; reviewers may read more. Body-only rules leave the core manifest empty and the staged diff complete. Reading coverage is diagnostic: it changes no findings, quorum or permission and buys no repeat. The author decides further reads. The window sizes delivery, never its authority.

Planning resolves targets and evidence against `active_repo_dir_for(ctx)` and governance against the system repository; never fall back to reviewing the Ouroboros repository for an external plan. Exact user-managed installed-skill payload paths are the one data-plane exception, for classification only: never a self-modification, never attachable evidence (`denied_path`).

Review history keeps the operative contract, current author/critic statuses and open/deferred/pending facts explicit. Actor `review_notes` may shorten individual reasons or group exact historical decisions, retaining membership and lineage in the existing selected view. No grouping by age or terminal status. Producers project decisions once with mirror references; unknown legacy content stays full. Recheck versions through publication, cold context, new packets and Continue; source gaps preserve recoverable bodies. Invalid optional transfers leave their own attachments full. Unchanged views retain their prefix; all surviving accounts cross continuations. Notes grant no verdict and rewrite no paid replay. Enforcement: `tests/test_review_local_accounts.py`, `tests/test_review_cold_history.py`, `tests/test_review_view_integration.py`. Closure/cycles: ARCHITECTURE §6 "Plan construction and review".

**Context mode (Nano / Low / Max).** Follow the Main registry row. Max binds Development to the active repository, never inferred intent, and retains its books at startup and rebind despite predicted pressure. Only actual size-refusal recovery may lower that document projection, after earlier recovery steps. Preserve owner Low/Nano sizing, current lower starts, tier-0 and P3; an estimate is neither a refusal nor permission to reduce quality. Mechanism: ARCHITECTURE §6 "Context fitting, retry, and compaction".

### Invariant: Exact premises with explicit source ownership

Plan from the complete retained room — both speakers, options and answers exact (`dialogue_evidence.py`, `plan_dialogue.py`) — with no independent dialogue byte cap, never from the bounded post-consolidation reader or the acceptance directive ledger (`review_evidence._accept_owner_directives`). JSONL records and chat line selectors split on physical LF only, never inside valid Unicode of a message. A replay or an addressed re-ask of the same author request keeps its recorded snapshot and discloses later messages as unreviewed. Enforcement: `tests/test_plan_review_w3.py::test_packet_uses_full_dialogue_and_keeps_acceptance_directives`, `tests/test_plan_dialogue_review_regressions.py`.

### Invariant: Compaction must earn its rewrite

Treat the task's dated focus as data, not owner instructions. Strictly read its canonical result, distinguishing absence/failure; update same-round facts and remeasure (`tests/test_self_focus_context.py`).

Predicted pressure buys no helper. Actual refusal follows ARCHITECTURE §6’s recovery order; unseen bodies have a separate late source-only rescue. Checkpoint originals first; helpers replace only fully covered units with smaller signed views. Actor notes may fold completed working prose/tools without shrinking, preserving governing human words, recorded outward speech, newer arrivals and opaque protocol. Local notes keep positions unless selected for merging; exposure uses physical evidence, not call IDs. Test final sealed requests, cold restoration and both sides of dialogue: `tests/test_compaction.py`, `tests/test_main_authored_context.py`, `tests/test_context_source_view.py`, `tests/test_delivery_dialogue_context.py`.

### Invariant: No silent truncation

When a core governance artifact does not fit the available budget:

- Where the flow requires inline delivery, inability to fit is an assembly failure, not a smaller pack (BIBLE P3): a typed entry names the artifact and the reason, the review does not proceed on the remainder, and the disclosure accompanies the refusal rather than replacing it. Elsewhere the omission is named where the reader sees it (`Reference book source unavailable: …`, `⚠️ OMISSION NOTE`), never silent; a reviewer without ARCHITECTURE.md is not operating with full context.
- Retain exact tool sources before projecting the complete Main batch; preserve outcomes, requested ranges and source gaps. Producer pages are separate from this frame, and no tool/path exemption applies. Read receipts count the actual consumer projection, not a predicted head cut.
- Book navigation (`context_layout.book_navigation`), a single-document navigation map and a named on-demand pointer are lossless representations, not truncation; Low and Nano use them and never apply `[:N]` to a document.
- Bound strings through `utils.truncate_review_artifact` (display previews) or `utils.truncate_within_limit` (a strict wire/prompt bound), never a hand-rolled `text[:cap] + marker`, which loses the anti-waste floor and can return a value longer than its input.
- A list obeys the same rule: a `[:N]` slice carries the omitted count and, where downstream compares an identity, a durable hash or reference for the full set (`_outcome_receipts.receipt_identity_projection`). Bounding a set is allowed; hiding that it was bounded is the P1 violation.

Enforcement: `tests/test_tool_result_delivery.py`, `tests/test_delegated_result_delivery.py` and `tests/test_owner_facing_honesty.py`.

### Invariant: Owner-facing surfaces show the full text

Disclosed truncation (the `⚠️ OMISSION NOTE` marker) protects **LLM context budgets**; it never licenses shortening what the owner reads.

- **Owner/UI-bound surfaces** (chat, task_results projections, review verdicts shown to a person) present the complete text or a reference to a durable full copy (an observability `response_ref`): reviewer rationale is a cognitive artifact (BIBLE P1), and truncating it beside an unreferenced full copy is partial memory loss. Terminal text asserts only recovery facts the round record carries, never a route mechanism the selected transport cannot perform.
- **Model-bound projections** (review packs, context sections, tool-result transport) keep their disclosed-truncation budgets.
- **A cut cheaper than its own marker is forbidden everywhere** (the shared primitive's floor); the one exception, an identifier-shaped single-line field under a tiny limit, is `reflection._truncate_with_notice`.

Enforcement: `tests/test_owner_facing_honesty.py`.

### Invariant: No "only if touched" gate for core artifacts

Core governance artifacts reach review and reasoning flows unconditionally, not only when they appear in `touched_paths`: `build_touched_file_pack` serves changed files; core artifacts load independently. The per-flow presence tests below are the mechanical cover; otherwise review-only.

### When adding a new reasoning flow

A new flow that reasons about code structure, system architecture or engineering standards:

1. Explicitly loads `ARCHITECTURE.md` (and BIBLE.md when constitutional reasoning applies).
2. Logs a warning when the file is missing or unavailable — never skips silently; a required artifact that cannot fit fails assembly ("No silent truncation").
3. Adds a test asserting the file is present in the assembled context/prompt. That test is the enforcing surface; CHECKLISTS item 25 (`context_building`, advisory) backstops the review.

---
