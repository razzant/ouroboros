# Core Governance Artifacts

The **core governance artifacts** — `BIBLE.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md` — ground constitution, architecture and procedure. This chapter owns delivery, structural plan tiering, exact premises, earned compaction and disclosed model-only truncation. Full availability means inline text or complete navigable sources; absence never proves full context.

### Invariant: Full availability in reasoning flows

Any flow that requires architectural, constitutional, or procedural reasoning
MUST include these artifacts as **first-class context sections** — not as
optional or opportunistic inclusions via touched-file packs. Each registry row
names its inline and on-demand delivery; neither permits silent truncation.

Commit triad, scope, advisory and deep self-review share
`ouroboros/tools/governance_context.py`. Tier 1 always delivers BIBLE.md, the
applicable CHECKLISTS section and CHECKLISTS_ARCHIVE standing disclosures in
full. Tier 2 selects the review-protocol chapter, DEVELOPMENT chapters naming
touched files, and DESIGN for `web/` changes within
`runtime_limits.REVIEW_GOVERNANCE_INLINE_SHARE` of the usable window; overflow
stays named in navigation. Tier 3 delivers the ARCHITECTURE book navigation,
never the whole map; tool-free triad packet rows also receive relevant
sections within that same share. This keeps shared rules consistent without
letting reference books crowd out the change. A pointer gives a packet row
no tools or evidence it did not receive.

Plan governance tiering uses only declared `affected_paths` resolving under the system repository, never prose or plan-kind taxonomy. The subject is an intention; tiering permits no silent omission. Unassemblable required governance raises typed `PlanPacketError`; unattached evidence remains a named absence (`[reviewer-requested]`, head cut `truncated_to_<N>`). Repeated locators stay `need_evidence`/`need_evidence_repeat` without new memory or paid cycles. Classification, bounds and wave/replay owners: ARCHITECTURE §6 "Plan construction and review", `ouroboros/tools/plan_packet.py`, `plan_spec.py`.

Exact-wave custody is fail-closed: each packet slot continues its recorded
transcript; the panel goes out fresh only when no exact artifact reference exists
or the roster changed, each packet slot disclosing its cause; an unreadable
referenced artifact returns `plan_review_exact_artifact_unavailable` and never
mints replacement authority.

The context-delivery registry:

| Flow | BIBLE.md | ARCHITECTURE.md | DEVELOPMENT.md |
|------|----------|-----------------|----------------|
| Main task context (`context.py`) | full tier-0 | full composition in Max, a subagent child excepted (issue #1026); book navigation in Low/Nano and for every subagent child | book navigation in Low/Nano and for a subagent child; in Max full when the active binding targets the system repo (evolution/self-body work, `workspace="none"`, a project-room turn with no external binding), else a visible on-demand pointer (external workspace, API/CLI/scheduled surface) |
| Triad review (`tools/review.py`) | full via API preamble or retrieving task | Tier 3: book navigation; packet rows also receive sections naming touched files within the inline share | Tier 2: review protocol and chapters naming touched files within the share; the rest remains navigable |
| ↳ Cold-start density rung | — | — | Triad packets only: an oversized packet without fresh exact-model density evidence gets one bounded probe of its own 80,000-char slice and one rebuild; a budget refusal stays disclosed (`review_admission.density_probe_before_size_refusal`). Retrieving surfaces have no packet-fit rung. |
| ↳ Anti-thrashing | — | — | Open obligations from `review_state` (`load_state(drive_root)` + `make_repo_key(repo_dir)`) enter `_build_review_history_section`; the scope brief does the same when `drive_root` is available (`scope_review_session.build_scope_session_task`). |
| Background consciousness wake-up (`consciousness.py` → `handle_wake_direct`) | = Main task context | = Main task context | = Main task context |
| Advisory pre-review (`tools/claude_advisory_review.py`) | full, shared tier 1 on both retrieving deliveries | Tier 3: book navigation and on-demand reading | Tier 2 within this row’s transcript-bound share; touched files arrive as a size/disposition manifest with the span-only carrier cut disclosed, while changed lines are in the diff |
| Scope review (`tools/scope_review.py`) | full, shared tier 1 beside the Intent / Scope checklist, in every context mode | Tier 3: physical chapter navigation and on-demand reading | Tier 2 within the usable-window share; the brief carries the complete staged change inline or as an exact paged source |
| Skill review (`skill_review.py`) | full inline (`api_chat`) / mandatory full source-root read (`agent_session`) | same two classes | same two classes |
| Plan review (`tools/plan_review.py`) | full for a SELF-MODIFICATION plan; otherwise a runtime heading-derived navigation map, never a copy | full for a self-modification plan (`api_chat` inline, `agent_session` mandatory full read); otherwise book navigation + a resolvable pointer | not resident: a named on-demand pointer; a reviewer needing it returns `need_evidence` with an exact `::lines=A-B` range |
| Deep self-review (`deep_self_review.py`) | full inline through shared tier 1 on native and session deliveries, without a duplicate-read demand; the seven-file memory whitelist stays byte-exact inline with per-entry dispositions | Tier 3: book navigation and chapters on demand | Tier 2 within this row’s transcript-bound share; deep keeps its own report criteria and CHECKLISTS navigation (ARCHITECTURE §6 "Deep self-review") |

`input_sources="declared"` selects authored inputs; `memory_mode=empty` only seeds a drive. Shared views and native work orders retain distinct transport evidence. Qualify actual sends; tools/mail may broaden inputs, exchange remains the mind's choice (ARCHITECTURE §6 "Selected first-input sources").

Scope's change-relative source manifest (`tools/scope_required_sources.py`)
names touched protected runtime, frozen contracts and prompts, their declared
families and cross-language twins. It is a minimum, not a sufficiency claim;
reviewers may read any part of the body. Native delivered-range receipts are
host-observed; session journals yield weaker harness-observed facts or
unobserved extents. Complete, incomplete, declared-empty and unobserved
coverage, including unavailable sources, stays diagnostic on every route: it
changes neither findings,
quorum nor commit permission and trigger no automatic paid repeat. The author
judges whether a concrete gap warrants more reading. The window sizes delivery,
never its authority.

Skill review keeps the full stable governance/host prefix cache-friendly on API
rows; a retrieving session reads those canonical files from its
source-repository root and takes only the byte-exact dynamic tail inline, so
payload snapshot and per-chunk quorum stay identical without rebilling or
crowding its window.

Planning resolves targets and evidence against `active_repo_dir_for(ctx)` and
governance against the system repository; never fall back to reviewing the
Ouroboros repo for an external plan. Exact user-managed installed-skill payload
paths are the one data-plane exception, for CLASSIFICATION only: never a
self-modification, never attachable evidence (`denied_path`).

Specs, findings and closure: ARCHITECTURE §6 "Plan construction and review".
Accept, reject or defer findings. Disposition-only
`plan_task(review_disposition={review_fingerprint, items:[{finding_id, decision, rationale}]})`
closes `need_evidence` at $0, one item per required finding; under advisory a
reasoned reject also closes a below-quorum blocking finding.
Answers merge by `finding_id` across calls: a later answer supersedes only its
own id, and two entries for one id in ONE call stay contradictory and open.
Duplicate, conflicting, unknown, stale, incomplete or vacuous calls
return typed argument errors before recording; `plan_review._handle_plan_task` ignores
default-empty optional fields. An envelope sent with items records them first,
then reviews.

Only explicit `review_disposition.author_action`, author disposition and critic
fingerprint select corrected goal/plan/spec. Exact `current_attempt.author_subject`
is separate from the critic. Advisory finish buys no panel; Blocking stop saves
without implementation approval. `closed_plan_review_wave` means critic-closed;
acceptance reads Advisory author claims as `author_plan`.

Force-plan is an LLM-first pre-implementation obligation on the admitted managed
root, not a mechanical permission check. `plan_review_state` owns durable review
facts, `config.get_review_enforcement()` the blocking/advisory value; effective
Cyber authority is separate (BIBLE P0/P3) and never rewrites an old wave to
GREEN. Every submitted envelope reaching `plan_task` supersedes prior authority,
so no newer attempt falls back to an older GREEN. Paid cycles are bounded by the
shared `OUROBOROS_REVIEW_MAX_CYCLES` (`ouroboros/review_cycles.py`).

**Context mode (Nano / Low / Max).** The Main row above defines book projections
(ARCHITECTURE §6 "Context fitting, retry, and compaction"); Max binds
`DEVELOPMENT.md` to the active repository by path, not message wording. Tier-0 identity and constitutional context
stays full in every mode. Predicted Max pressure never swaps in Low documents:
only actual provider overflow, after useful working-history and narrative-memory
reduction, may use task-local Low book navigation. Recovery sends must be strictly
smaller; none changes owner mode or P3 commit/scope review. Disclosed residual: an explicit per-task handbook override
(`context_requires_self_body_docs`) wins in Max only: Low and Nano ignore it
(issue #1019), as does a delegated subagent child in every mode (issue #1026); the
sibling `context_requires_development` flag is ignored on the same paths.

### Invariant: Exact premises with explicit source ownership

Plan from the complete retained room — both speakers, options and answers
exact — (`dialogue_evidence.py`, `plan_dialogue.py`; ARCHITECTURE §6 "Plan construction and
review"), with no independent dialogue byte cap, never from the bounded
post-consolidation reader or the acceptance directive ledger that task
acceptance keeps (`review_evidence._accept_owner_directives`). JSONL records
and chat line selectors split on physical LF only, never on valid Unicode
inside a message. Each consumer redacts at its boundary and discloses missing
source or ranges; a replay or an addressed re-ask of the same author request
keeps its recorded snapshot (complete and uncapped; the inline view is the
conversation only, with a pointer naming exact omitted line ranges),
disclosing later messages as unreviewed. Follow
ARCHITECTURE's per-delivery context/source contract for delivery, coverage and sizing. Enforcement:
`test_packet_uses_full_dialogue_and_keeps_acceptance_directives`,
`tests/test_plan_dialogue_review_regressions.py`, the acceptance ledger tests
in `tests/test_loop_misc.py`.

### Invariant: Compaction must earn its rewrite

Helper compaction is deficit-triggered and low-water-sized: one pass per route/round requests the deficit plus an eighth of the binding target/capacity boundary (`RECLAIM_LOW_WATER_DIVISOR`, `tests/test_context_budget_ssot.py`). Requested margin and achieved headroom are distinct facts, never a judgment of thinking. Main may replace only exact units whose call arguments, complete results and text were present in a usable physical attempt. Syntax changes across provider formats do not imply missing exposure, but an unsent or omitted unit stays raw. Checkpoint the complete source before map/fold, publish only bound, smaller units, and split only on typed summarizer overflow. Helper text is an attributed host record; actor-authored notes remain the actor's. Preserve older capsule/source readers.

Authored views reuse source custody and need not shrink. Preserve complete owner/new tails and schema residency, measure through existing fit arithmetic, and expose directly usable inspect/restore handles on canonical and live child roots. Same-ID continuation retains the frozen memory view and exposure, never recaptures it as today's history. Sources being stored is not proof they were delivered. Test real loop wiring and both preservation/replacement paths (`tests/test_main_authored_context.py`, `test_context_physical_exposure.py`); ARCHITECTURE §6 owns mode/route, custody and sizing mechanisms. No second trigger or retry policy belongs here.

### Invariant: No silent truncation

If a core governance artifact cannot fit in the available context budget:

- Where the flow requires inline delivery, inability to fit is an assembly
  FAILURE, not a smaller pack
  (BIBLE P3): a typed entry names the artifact and reason, the review does not
  proceed on the remainder, and disclosure accompanies the refusal, never
  replaces it; adjust the budget/flow or refactor. Elsewhere an omission or cut is
  NAMED where the reader sees it (`Reference book source unavailable: …`,
  `⚠️ OMISSION NOTE`), never silent, so operator and model both know the
  context is incomplete.
- A reviewer or agent operating without ARCHITECTURE.md MUST NOT be treated as
  operating with full context — findings may be incomplete.
- Tools returning multi-model review findings (`commit_reviewed`,
  `skill_review`, scope/advisory review helpers) MUST be in
  `UNTRUNCATED_TOOL_RESULTS` or carry an explicit per-tool limit; the default
  15,000-char `DEFAULT_TOOL_RESULT_LIMIT` is not acceptable for review verdicts.
- Book **navigation** (`context_layout.book_navigation`: per chapter the
  authored introduction, physical path and H2-H4 inclusive complete-subtree
  ranges), a single-doc **navigation map** and a named on-demand pointer are
  visible, lossless representations, NOT silent truncation; Low and Nano use
  them and never apply `[:N]` to a doc.
- Bound strings through the SSOT `utils.truncate_review_artifact` (DISPLAY
  previews) or `utils.truncate_within_limit` (a STRICT wire/prompt bound that
  never exceeds its limit), never a hand-rolled `text[:cap] + marker`, which
  loses the anti-waste floor and can return a value LONGER than its input.
- A LIST obeys the same rule: a `[:N]` slice carries an explicit omitted COUNT
  and, where it touches an identity something downstream compares, a durable
  hash or reference for the full set
  (`_outcome_receipts.receipt_identity_projection`); bounding a set is allowed,
  hiding that you bounded it is the P1 violation.

Disclosed source-read gap: governance loaders may continue with a named omission
when a book cannot be loaded. This is not full context; loader gaps and failures
to fit required inline material are separate facts. Retrieving source coverage
remains diagnostic, not another assembly or commit gate.

Enforcement: `tests/test_tool_capabilities.py` (the `UNTRUNCATED_TOOL_RESULTS`
roster) and the truncation-floor coverage in
`tests/test_owner_facing_honesty.py`.

### Invariant: Owner-facing surfaces show the full text

Disclosed truncation (the `⚠️ OMISSION NOTE` marker) protects **LLM context
budgets**; it never licenses shortening what the owner reads:

- **Owner/UI-bound surfaces** (chat panels, task_results projections, review
  verdicts shown to a person) present the COMPLETE text or a reference to a
  durable full copy (e.g. an observability `response_ref`): reviewer rationale
  is a cognitive artifact (BIBLE P1), so truncating it beside an unreferenced
  full copy in private blobs is partial memory loss. Terminal text asserts only
  recovery facts the round record carries, never a route mechanism the selected
  transport cannot perform.
- **Model-bound projections** (review packs, context sections, tool-result
  transport) keep their disclosed-truncation budgets — real context economics.
- **A cut cheaper than its own marker is forbidden everywhere** (the shared
  primitive's floor). One named exception: single-line identifier fields under
  a 100-char limit (a reflection backlog `kind`) take a plain hard slice, as a
  multi-line marker in a one-line value does worse damage than the cut.

Enforcement: `tests/test_owner_facing_honesty.py`.

### Invariant: No "only if touched" gate for core artifacts

Core governance artifacts reach review/reasoning flows unconditionally, NOT only
when in `touched_paths`: `build_touched_file_pack` serves _changed_ files; core
artifacts load independently. No surface of its own — the per-flow presence
tests below are the mechanical cover; otherwise review-only.

### When adding a new reasoning flow

A new flow that reasons about code structure, system architecture, or
engineering standards MUST:

1. Explicitly load `ARCHITECTURE.md` (and BIBLE.md if constitutional reasoning
   applies).
2. Log a warning if the file is missing or unavailable — never skip silently
   (a REQUIRED artifact that cannot FIT fails assembly — "No silent truncation").
3. Add a test asserting the file is present in the assembled context/prompt.
   That test is the enforcing surface; CHECKLISTS item 11 (`context_building`,
   advisory) backstops the review.

---
