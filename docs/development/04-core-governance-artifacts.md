# Core Governance Artifacts

BIBLE/Architecture/Development ground identity/structure/procedure; DESIGN names product/interface rules. This chapter governs delivery, exact premises and disclosed views; availability, delivered content and reading/application stay distinct.

### Invariant: Full availability in reasoning flows

Architecture, constitution and procedure flows receive applicable governance as first-class context — inline or complete navigable sources per the registry, never silently omitted.

**Review surfaces.** Briefs per layer (`tools/governance_context.py`, `review_body_fact.layer_for`), body tiers and the universal core/foreign section: ARCHITECTURE §6 "Governance delivery".

**Plan review.** Body governance follows declared `affected_paths`, never prose or plan kind (P5): a path into the installed repo or recognized copy, or folderless research with explicit `context_requires_self_body_docs=true` and accepted `task_design_source` — never Main/cwd or `[]` alone. Reviewers of a body plan must read BIBLE and ARCHITECTURE in full, regardless of the planner's context mode. Required assembly failures raise `PlanPacketError`; unattached evidence stays named; a repeat seat continues only from its own retained thread or delivery and reply (`continued=true`); an unreadable source means a fresh send recorded as `continued=false`, never a borrowed or invented transcript (ARCHITECTURE §6 "Plan construction and review", `tools/plan_packet.py`).

**DESIGN delivery.** Authors get the bible before mutation in every mode; ARCHITECTURE §6 owns selection and fitting. Sessions on the work's folder use the same scoped reader after root preparation, pinning view/path/SHA from one read in start instructions, not the work order. Plans get headings; plan/CODE/coupling/preflight use installed governance. Web review fits DESIGN whole, else its bible or principles and a map; changed words elsewhere fit only those excerpts. Gaps and actual inline coverage are named. A map gives a packet no reader and proves no reading or application. No new gate or whole-book duty.

The context-delivery registry:

| Flow | BIBLE.md | ARCHITECTURE.md | DEVELOPMENT.md |
|---|---|---|---|
| Main task context (`context.py`) | full tier-0 in every mode | the resident Body Map in every mode and focus (`reference_books.body_map`), its parts and addresses only where a physical window cannot hold it; chapters are read on demand | read on demand at the Map's addresses; a task's requirement (`development_required`) adds it whole, as its own block, in any mode |
| Change review, Part 1 (`tools/review.py`) | full (API preamble or retrieving task) | tier 3 navigation; packet rows also get sections naming touched files within the inline share | tier 2 within the share: the review protocol and chapters naming touched files |
| ↳ `review_change` by layer (`review_body_fact.layer_for`) | body: as Part 1; core: `not_applicable`, the preamble names no constitution (`review_prompt_text.review_preamble("core")`) | body: as Part 1; core: `not_applicable`, the navigation indexes the subject's documents (`subject_root`) | body: as Part 1; core: the universal `Change Review Checklist` alone, with an empty-by-rule required-source manifest |
| Background consciousness wake-up (`consciousness.py` → `handle_wake_direct`) | = Main task context | = Main task context | = Main task context |
| Preflight (`review_change(surface=preflight)`: one named row over the system repository's worktree) | the `review_change` body row | as BIBLE | as BIBLE |
| The coupling question (Part 2 of the retrieving seat's brief, `tools/review_brief_coupling.py`) | full tier 1 beside the `Coupling questions` section, in every context mode | tier 3 navigation | tier 2 within the usable-window share |
| Skill review (`review_change(subject="skill")`) | full inline (packet) / declared mandatory full read (retrieving) | tier 3 navigation | tier 2: owed chapters + selected contracts |
| Plan review (`tools/plan_review.py`) | full for a body plan (`api_chat` inline, `agent_session` mandatory full read); otherwise none, no map | as BIBLE | not delivered or pointed at; the mind may attach it |
| Deep self-review (`deep_self_review.py`) | full tier 1 on both deliveries | tier 3 navigation | tier 2 within the row's share (ARCHITECTURE §6 "Deep self-review") |

`tools/scope_required_sources.py` sets a body reading floor, not a ceiling; core manifests stay empty and staged diffs complete. Reading coverage and window size never change review authority or buy another review: coverage is diagnostic, window size only shapes delivery, and the mind decides further reads (DEVELOPMENT §5 "Review pool and paid cycles"). `plan_ask.planning_roots` and `read_admission` bind planning roots and attachments (data roots: own stores only), so external plans never fall back to Ouroboros and user-managed skill paths only classify (`denied_path`, never self-modification or attachable evidence).

Review history keeps the operative contract, current author/critic statuses and open/deferred/pending facts explicit. Actor `review_notes` may shorten reasons or group exact historical decisions, keeping membership and lineage in the existing selected view (never grouping by age or terminal status), without granting verdicts or rewriting paid replays; producers project once with mirror refs, keep unknown legacy bodies (and recoverable ones at source gaps), recheck versions through publication, cold context, new packets and Continue, keep unchanged prefixes intact, carry all surviving accounts across continuations, and leave invalid optional transfers unshortened (`ouroboros/review_history_view.py`; ARCHITECTURE §6 "Review ledger record", "Plan construction and review").

**Context mode (Nano / Low / Max).** The Main row describes the acting agent's own context: no mode or binding preloads a book; Max holds while the Body Map, my shortest tellings and the work's room fit, else that work runs in Low, temporarily, for the window, the Map at its parts before any memory goes below a telling; unknown windows send full. Preserve owner Low/Nano sizing, lower starts, tier-0 and P3; targets cut no chosen read or owner words, and rebinds re-cut no fitting sent view (`tests/test_context_economy_window.py`; ARCHITECTURE §6 "Context fitting, retry, and compaction").

### Invariant: Exact premises with explicit source ownership

Plan from the complete retained room — both speakers, options and answers exact (`dialogue_evidence.py`, `ouroboros/tools/plan_dialogue.py`) — with no independent dialogue byte cap, never from the bounded post-consolidation reader or the owner-directive ledger (`review_evidence._accept_owner_directives`). JSONL records and chat line selectors split on physical LF only, never inside valid Unicode.

### Invariant: Compaction must earn its rewrite

Treat dated focus as data, never owner instructions; read canonical results strictly, distinguishing absence from failure; update same-round facts and remeasure (`tests/test_self_focus_context.py`).

Known-window no-result-room predictions and typed size refusals run model-free recovery passes (ARCHITECTURE §6), with no helper summarizer: `compact_context` requires `working_note` (else `authored_note_required`). Actor notes may fold completed working prose/tools without shrinking while keeping governing human words, recorded outward speech, newer arrivals, opaque protocol and local note positions (unless merged); prove exposure from physical evidence, not call IDs. Test final sealed requests, cold restoration and both sides of dialogue: `tests/test_compaction.py`, `tests/test_main_authored_context.py`, `tests/test_context_source_view.py`, `tests/test_delivery_dialogue_context.py`.

### Invariant: No silent truncation

When a core governance artifact does not fit the available budget:

- Required inline delivery fails assembly rather than shipping a smaller pack (BIBLE P3): emit a typed refusal naming artifact and cause, and halt review; the disclosure accompanies the refusal rather than replacing it. Elsewhere disclose omissions in-band (`Reference book source unavailable: …`, `⚠️ OMISSION NOTE`), a missing ARCHITECTURE.md included.
- Persist exact tool sources before projecting Main batches with no exemptions; record host cut reasons, outcomes, ranges and gaps inside partial envelopes; producer pages stay outside this frame and read receipts count consumer projections.
- The Body Map's parts, book navigation (`context_layout.book_navigation`, reviewers' chapter-addressed view), heading maps and pointers index sources without delivering bodies; never slice documents with `[:N]`.
- Bound strings via `utils.truncate_review_artifact` (previews) or `utils.truncate_within_limit` (wire/prompt bounds), never hand-rolled `text[:cap] + marker`, which bypasses the anti-waste floor and can lengthen input.
- Any list `[:N]` slice states its omitted count and, where identity is compared downstream, a durable full-set hash/ref (`_outcome_receipts.receipt_identity_projection`): bounding is allowed, hiding it violates P1.

Enforcement: `tests/test_tool_result_delivery.py`, `tests/test_owner_facing_honesty.py`.

### Invariant: What a person reads whole stays whole

Disclosed truncation (`⚠️ OMISSION NOTE`) protects LLM context budgets, never human-read artifacts.

- **Human-read artifacts** (reviewer rationale, answers) stay reachable in full, in details or via durable `response_ref` (BIBLE P1: unreferenced cuts are partial memory loss). Status/label/card lines follow DESIGN principle 7; terminal text asserts only recorded round recovery facts, never unsupported transport mechanisms.
- **Model-bound projections** (review packs, context sections, tool-result transport) keep disclosed-truncation budgets.
- **A cut cheaper than its own marker is forbidden everywhere** (shared primitive floor).

Enforcement: `tests/test_owner_facing_honesty.py`.

### Invariant: No "only if touched" gate for core artifacts

Applicable body governance reaches review/reasoning whatever the `touched_paths`; `build_touched_file_pack` serves changed files, not eligibility. Presence tests enforce delivery; else review-only.

### When adding a new reasoning flow

A new flow reasoning about structure, architecture or standards must:

1. Explicitly deliver Architecture as a row of the registry above: `ARCHITECTURE.md` whole, or the Body Map with its chapters readable at their addresses (plus BIBLE.md for constitutional reasoning).
2. Log a warning when it is missing/unavailable — never skip silently; an unfit required artifact fails assembly ("No silent truncation").
3. Add a presence test on the assembled prompt/context (enforcing surface; CHECKLISTS item 22 `context_building`, advisory, backstops review).

---
