# Core Governance Artifacts

This chapter owns the availability contract for BIBLE, ARCHITECTURE and DEVELOPMENT in every reasoning flow: the per-flow delivery registry, the one structural fact by which plan review tiers its governance pack, and the truncation invariants that make an omission visible instead of silent. It exists because a reviewer operating without the architecture map is not operating with full context, and the only honest response is to disclose that rather than to shrink the pack quietly.

`BIBLE.md`, `docs/ARCHITECTURE.md`, and `docs/DEVELOPMENT.md` are **core
governance artifacts** — the constitutional, architectural, and procedural
ground truth of the system.

### Invariant: Full availability in reasoning flows

Any flow that requires architectural, constitutional, or procedural reasoning
MUST include these artifacts as **first-class context sections** — not as
optional or opportunistic inclusions via touched-file packs.

Plan review is the one flow whose governance pack is tiered, and by ONE
structural fact — whether a declared `affected_paths` target resolves under the
Ouroboros system repository (the classification, and what never counts as a
change: `docs/architecture/06-agent-core.md` § "Plan construction and review")
— never by prose and never by a plan-kind taxonomy, which is what keeps
classification un-gameable. This is a tiering,
not an omission: before any work exists the reviewer's subject is the
INTENTION, and every absence is a named pointer or a typed `need_evidence`
finding the host attaches on the next cycle under the same evidence policy —
the locator enters the manifest hash, so the next envelope is a new
fingerprint, never an idempotent replay; nothing is silently omitted (P1).
Two branches follow, and only one of them stops a review. A REQUIRED
governance pack that cannot be assembled is a typed assembly failure and the
review does not run (`PlanPacketError`). Declared or reviewer-requested
evidence the policy cannot attach is a named absence instead — a
`[reviewer-requested]` omission row, or the head attached with the cut named
`truncated_to_<N>` — and the panel still runs and judges with it; a re-asked
locator stays a `need_evidence` request with a `need_evidence_repeat` disclosure,
requiring the same free disposition without expanding request memory or paid cycles.
DEVELOPMENT.md is not resident in a plan-review packet; it is one such request
away. Packet composition, bounds, and wave/replay mechanics: ARCHITECTURE
"Plan construction and review" and `ouroboros/tools/plan_packet.py` /
`plan_spec.py`. Planning room evidence comes from `dialogue_evidence.py` over
`Memory.read_chat_generations` and the shared `project_dialogue.room_membership`;
`plan_dialogue.py` binds its redacted source to author-request identity. Do not
substitute the bounded post-consolidation reader or acceptance directive ledger.
An identical replay keeps the recorded source; real plan/evidence changes capture
current discussion. Follow the per-delivery context/source contract in ARCHITECTURE:
API window fit, native mandatory-read bound, and delegated harness-owned reading
with full immutable files and honest coverage. No independent dialogue byte cap.

Exact-wave custody is fail-closed: the evidence continuation uses a fresh
full-packet dispatch only when no exact artifact reference exists. An unreadable
referenced artifact returns `plan_review_exact_artifact_unavailable`; it never
mints replacement authority.

The context-delivery registry:

| Flow | BIBLE.md | ARCHITECTURE.md | DEVELOPMENT.md |
|------|----------|-----------------|----------------|
| Main task context (`context.py`) | full tier-0 | composed in full in Max for every task class; in Low/Nano the book navigation — the authored chapter introductions with each chapter's own heading index and line ranges into that physical chapter file (`context_layout.book_navigation`) | mode-independent: full when the active binding targets Ouroboros's system repo, including evolution/self-body work and a project-room turn without an external binding; visible on-demand pointer for a bound external workspace, subagent, or API/CLI/scheduled external surface. `workspace="none"` and explicit self-body overrides retain full Development. |
| Triad review (`tools/review.py`) | ✅ via preamble | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` |
| ↳ Cold-start density rung | — | — | Shared with scope review and the packed deep self-review (`capability_evidence.cold_start_density_probe` on a `review_helpers.density_probe_sample` slice, wired at the admission seam `review_admission.density_probe_before_size_refusal`): a packet that would be refused or degraded for size while its route has no fresh exact-model density witness gets one bounded probe send on the exact model, then one re-size/rebuild; a budget-refused probe is a typed disclosure and the existing refusal stands. |
| ↳ Anti-thrashing | — | — | Open obligations loaded from `review_state` via `load_state(drive_root)` + `make_repo_key(repo_dir)`, injected unconditionally into `_build_review_history_section` prompt context. Same mechanism in `scope_review.py::_build_scope_prompt` (best-effort when `drive_root` available). |
| Background consciousness (`consciousness.py`) | ✅ full | ✅ composed in full (max) / book navigation (low) | — (not yet required) |
| Advisory pre-review (`tools/claude_advisory_review.py`) | Two delivery classes: an `api_chat` row runs the bounded NATIVE inspection episode (governance docs reached through its read-only tools); an `agent_session` row receives a resolvable pointer marked MANDATORY FULL READ and the session reads the full doc itself — retrieval is disclosed (native reads are host-observed; vendor-session reads are not) | same two delivery classes | same two delivery classes |
| Scope review (`tools/scope_review.py`) | full canonical doc + Atlas accounting; a size terminal or a degradation rung under a cold density cap takes the shared cold-start density rung above (one probe, one rebuild, `density_probe` ladder step) | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting |
| Skill review (`skill_review.py`) | full inline (`api_chat`) / mandatory full source-root read (`agent_session`) | full inline (`api_chat`) / mandatory full source-root read (`agent_session`) | full inline (`api_chat`) / mandatory full source-root read (`agent_session`) |
| Plan review (`tools/plan_review.py`) | full for a SELF-MODIFICATION plan (the structural path fact above); otherwise a heading-derived navigation map of BIBLE.md generated at runtime (never a copy) | inline, in full, for a self-modification plan; otherwise the book navigation (per chapter: introduction, physical path, heading index) + a resolvable pointer | named on-demand pointer; a reviewer that needs it returns `need_evidence` (an exact `::lines=A-B` range for one section) and the host attaches what the evidence policy allows on the next cycle, naming every absence |
| Deep self-review (`deep_self_review.py`) | Three deliveries on the `deep_review` row (`docs/architecture/06-agent-core.md` § "Deep self-review"). Packed api row: full canonical doc + Atlas accounting, with the shared cold-start density rung and the typed `deep_self_review_pack_unfit` refusal (no automatic fallback). Native inspection episode / agent session: MANDATORY full read at the repository root — host-observed and coverage-derived on the native episode, `unobserved` for a session — disclosed in the report header and as a typed `capability_delta`. Memory (up to seven whitelisted files) is INLINED byte-exact on every delivery with each entry's disposition disclosed, never receipt-checked | Packed: composed in full (max) / the authored chapter introductions with their physical paths (low) + Atlas accounting. Retrieving rows: those same introductions, chapters read on demand by path | Packed: full composed book + Atlas accounting. Retrieving rows: the authored chapter introductions, chapters read on demand (CHECKLISTS.md keeps its single-doc `generate_doc_nav_map`) |

Skill Review keeps the full stable governance/host prefix for cache-friendly
API rows; a retrieving session reads those same canonical files from its
source-repository root and receives the byte-exact dynamic tail inline, so the
payload snapshot and per-chunk quorum stay identical without rebilling or
crowding the session window.

Planning has two distinct roots — governance documents from the system
repository; declared targets and evidence against `active_repo_dir_for(ctx)` —
and a path escaping the active subject or an unavailable root is a named
omission (the same ARCHITECTURE section). Exact user-managed installed-skill
payload paths are the one data-plane exception for CLASSIFICATION only — they
never make a plan a self-modification — and are not attachable as evidence (a
payload locator comes back as a named `denied_path` omission). Do not fall back
to reviewing the Ouroboros repo for an external plan.

The SPEC shape, the finding vocabulary (`blocking` with a `breaks` id, `note`,
`need_evidence`), the verdicts (`GREEN`, `REVIEW_REQUIRED`, `REVISE_PLAN`, the
honest `DEGRADED`) and the closure rules per finding class and enforcement mode
are stated once in `docs/architecture/06-agent-core.md` § "Plan construction
and review"; findings are inputs the main agent may accept, reject, or defer.
Outstanding `need_evidence` closes without a second LLM call through a separate
`plan_task` call containing `review_disposition` only — `{review_fingerprint,
items: [{finding_id, decision, rationale}]}` — covering each required finding
exactly once; duplicates, contradictions, unknown, stale, or incomplete
required dispositions fail closed, and mixed or vacuous calls fail before an
attempt is recorded as typed argument errors — an optional field that carries
no meaning beside a disposition (a blank `goal`/`plan`; a `spec` holding only
declared keys whose values are `None`, `""` or `[]`) is ignored, never mistaken
for a second operation, while any non-empty list, undeclared key or non-blank
string is meaning and makes the call mixed. The same call may voluntarily
annotate a current closed note-only wave without reopening the review or buying
another panel. Never replay the plan envelope with the disposition.

Force-plan is an LLM-first pre-implementation obligation on the admitted
managed root, not a mechanical permission check. `plan_review_state` owns
durable review facts and `config.get_review_enforcement()` owns the configured
blocking/advisory value. Effective Cyber authority is separate and follows
BIBLE P0/P3; it never rewrites an old wave to GREEN. Every submitted envelope that reaches `plan_task`
supersedes prior authority, so a newer attempt cannot fall back to an older
GREEN. Wave recording, free replays, DEGRADED semantics, structurally dead
slots, and `quorum_unreachable` release live in ARCHITECTURE "Plan
construction and review"; paid cycles are bounded by the shared
`OUROBOROS_REVIEW_MAX_CYCLES`.

**Context mode (Nano / Low / Max).** The projection each mode gives the two
books — Max composes `ARCHITECTURE.md` in full for every task class; Low and
Nano supply the book navigation addressed to the physical chapter files;
`DEVELOPMENT.md` is mode-independent and follows the active repository binding,
a path fact, never a guess from message text — is the Main task context row
above and `docs/architecture/06-agent-core.md` § "Context fitting, retry, and
compaction". The rule: tier-0 identity and constitutional context stays full in
every mode; predicted Max pressure never swaps in Low documents — only actual
provider overflow may use a task-local Low projection, followed by at most one
same-route strictly-smaller call — and this never changes owner mode or P3
commit/scope review.

### Invariant: Exact premises with explicit source ownership

Planning reads the complete retained room through `dialogue_evidence` and
`Memory.read_chat_generations`, with the shared Project membership predicate,
progress and addressed mailbox provenance. Both speakers, options and answers
remain exact. JSONL records and chat line selectors use physical LF boundaries;
valid Unicode inside a message is never a record delimiter. Task acceptance
keeps `review_evidence._accept_owner_directives` over the task-local ledger;
planning does not recreate its retired bounded directive section.

Each consumer redacts at its boundary and discloses missing source or ranges.
A replay or an already-earned paid retry of the same author request keeps its
recorded snapshot and discloses that later messages were not reviewed. A changed
plan or explicit evidence request captures current discussion. Native sizing
measures the complete first request, including its schemas and instruction
wrapper; existing task-local model choices apply before fresh source fitting.
Enforcement: `test_packet_uses_full_dialogue_and_keeps_acceptance_directives`,
`tests/test_plan_dialogue_review_regressions.py`, and the acceptance ledger tests
in `tests/test_loop_misc.py`.

### Invariant: Compaction must earn its rewrite

Context compaction is a deficit-requested materializer, not an independent
threshold, timer, route, or retry policy (selection over completed atomic
units, hard user-turn boundaries, the no-reclaim-no-mutation rule and the
route+round latch: `docs/architecture/06-agent-core.md` § "Context fitting,
retry, and compaction"). For a non-empty selection, persist the exact actor-visible checkpoint before
calling the summarizer. Summary input covers complete stable hashed chunks
with gap-free offsets; only typed summarizer context overflow may split a
source recursively. A replacement publishes only after transcript/unit
binding, complete coverage, checkpoint provenance, and a strictly smaller
representation on the caller's ContextFit measurement basis are all proved
(the bounded image proxy and density must match; raw base64 byte count is not
token reclaim). Capsules carry host-only generation, source-hash, part,
checkpoint, and CAS-ref metadata so a later pass can recompact them without
losing the original provenance union. Enforcement: `tests/test_compaction.py`,
`tests/test_loop_compaction.py`, `tests/test_loop_compaction_policy.py`.

### Invariant: No silent truncation

If a core governance artifact cannot fit in the available context budget:

- Do **not** silently omit it or truncate it without a visible marker. Either
  adjust the budget/flow to accommodate it, or emit an explicit warning
  (`⚠️ OMISSION NOTE: ARCHITECTURE.md omitted due to budget constraints`) so
  the operator and the model both know the context is incomplete.
- A reviewer or agent operating without ARCHITECTURE.md MUST NOT be treated as
  operating with full context — findings may be incomplete.
- Tools that return multi-model review findings (`commit_reviewed`,
  `skill_review`, scope/advisory review helpers) MUST be listed in
  `UNTRUNCATED_TOOL_RESULTS` or have an explicit per-tool limit; the default
  15KB transport cap is not acceptable for review verdicts.
- A reference-book **navigation** view (per chapter: the authored introduction,
  the chapter's physical path, and its H2-H4 inclusive complete-subtree ranges)
  and a single-doc **navigation map** (H2-H4 inclusive complete-subtree ranges,
  with parent rows overlapping descendants and full sections one `read_file`
  away) and a named on-demand pointer are visible, lossless representations —
  NOT silent truncation. The low context mode uses these; it never applies
  `[:N]` to a doc.
- String bounding goes through the SSOT `utils.truncate_review_artifact`,
  never a hand-rolled `text[:cap] + marker`. Besides the marker, that helper
  carries an anti-waste FLOOR: a cut saving fewer characters than its own
  omission note is pure damage, so below it the text passes through whole. A
  local re-implementation loses the floor and can return a value LONGER than
  the input it "shortened". The two bounded-string primitives serve different
  contracts: `truncate_review_artifact` produces DISPLAY previews (its floor
  may return the text whole), while `truncate_within_limit` enforces a STRICT
  wire/prompt bound — the omission marker lands INSIDE the limit and the
  result never exceeds it.
- Bounding a LIST is subject to the same rule: a `[:N]` slice must be
  accompanied by an explicit omitted COUNT, and — where the slice touches an
  identity that something downstream compares — a durable hash or reference
  for the full set (see `_outcome_receipts.receipt_identity_projection`).
  Bounding a set is allowed; hiding that you bounded it is the P1 violation.

Enforcement: `tests/test_tool_capabilities.py` (the `UNTRUNCATED_TOOL_RESULTS`
roster) and the truncation-floor coverage in
`tests/test_owner_facing_honesty.py`.

### Invariant: Owner-facing surfaces show the full text

Disclosed truncation (the `⚠️ OMISSION NOTE` marker) exists to protect **LLM
context budgets** — it is a model-bound mechanism, not a licence to shorten
what the owner reads:

- **Owner/UI-bound surfaces** (chat panels, task_results projections, review
  verdicts shown to a person) present the COMPLETE text, or carry a reference
  to a durable full copy (e.g. an observability `response_ref`). Reviewer
  rationale is a cognitive artifact (BIBLE P1): projecting it truncated while
  the full copy sits unreferenced in private blobs is partial memory loss.
  Terminal text asserts only recovery facts carried by the round record, never
  a route mechanism that the selected transport cannot perform.
- **Model-bound projections** (review packs, context sections, tool-result
  transport) keep their disclosed-truncation budgets — those are real context
  economics.
- **A cut cheaper than its own marker is forbidden everywhere** (the shared
  primitive enforces the floor — see "No silent truncation"). One named
  exception: tiny single-line identifier fields (limit < 100, e.g. a
  reflection backlog `kind`) keep a plain hard slice — a multi-line omission
  marker inside a one-line value is worse damage than the cut it discloses.

Enforcement: `tests/test_owner_facing_honesty.py`.

### Invariant: No "only if touched" gate for core artifacts

Core governance artifacts reach review/reasoning flows unconditionally — NOT
only when they appear in `touched_paths`. `build_touched_file_pack` is for
_changed_ files; core artifacts are a separate concern loaded independently.
No surface of its own — the per-flow presence tests required below are the
mechanical cover; otherwise review-only.

### When adding a new reasoning flow

If you add a new flow that reasons about code structure, system architecture,
or engineering standards, you MUST:

1. Explicitly load `ARCHITECTURE.md` (and BIBLE.md if constitutional reasoning
   applies).
2. Log a warning if the file is missing or unavailable — do not silently skip.
3. Add a test asserting the file is present in the assembled context/prompt.

That required presence test is the enforcing surface; CHECKLISTS item 11
(`context_building`, advisory) backstops the review.

---

