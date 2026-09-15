# Module Size & Complexity

This chapter owns the size discipline — the deterministic line, function and byte gates, the debt manifest they read, and the rule that a cap is paid down by simplifying where the change lives rather than by extracting a passthrough — together with the invariants that keep a growing system readable: projection over replay for hot readers, the source-complete decision pipeline, continuation authority, disposers for UI resources, and the geometry and refresh contract for embedded surfaces. It exists because every rule here is about bounded reading cost, and each names its enforcing surface or discloses that it has none.

P7 makes context fit a maintenance constraint, not a line-count aesthetic.

- Python modules everywhere (including `tests/` and `devtools/`) and
  first-party `web/**/*.js` modules (including `web/tests/`) target roughly
  1000 lines. The deterministic hard gates: 1600 lines per module (exact-path
  debt in `ouroboros/size_ratchet_manifest.py::GIANT_PATHS`; stale or newly
  oversized entries fail), 300 lines per non-grandfathered Python function
  (`FUNCTION_DEBT`, exact `(path, qualname)` keys), 200,000 UTF-8 bytes per
  module (`BYTE_DEBT`, shrink-only), and the exact-current 1001–1500 band in
  `BAND_PATHS` (a new or re-entered path requires a nonblank rationale).
  Regenerate the manifest with `scripts/regenerate_size_ratchet.py`; it
  validates the rendered candidate before writing and refuses an unmerged
  index with a typed error. Sources decode as strict UTF-8 and normalize to
  POSIX LF before counting, so checkout policy cannot change the inventory;
  vendored/minified assets are excluded, and the same production iterator
  drives the gates, smoke, health, and census.
- Paying down a size cap: when a change runs into a band, the hard cap, the
  function-size gate or the byte debt, pay it down by SIMPLIFYING where the
  change lives — simpler control/data flow and interfaces, dead code and
  duplicates removed, an existing SSOT reused, prose made compact and legible,
  so the module reads BETTER after the change. Extracting a helper, a
  passthrough wrapper or a neighbour module is the LAST resort, and a paydown
  only when the new unit is a natural boundary that would stay correct with the
  parent far below the cap: its own reason-to-change, an explicit contract, a
  real caller. A cap-driven bucket, a one-caller passthrough, and bytes bought
  by deleting contract-bearing comments, docstrings, messages or tests are
  defects, not paydown — report the conflict instead (BIBLE P7 «When adding a
  major feature — first simplify what exists»; the scope-review advisory item).
- Methods above 150 lines and more than eight parameters are decomposition
  signals (BIBLE P7, CHECKLISTS item 2(c)), not deterministic gates. Existing
  baseline debt is not retroactively a failing tree. JavaScript currently has
  only the module line-count gate.
- Runtime Python function/method count stays under
  `ouroboros/review.py::MAX_TOTAL_FUNCTIONS` (that iterator excludes
  tests/devtools; the module gates include them). The ceiling is a high-water
  alarm with ample headroom; raising it requires a one-line campaign rationale
  in the same commit.
- Enforcement: the OFFICIAL repository's CI runs the dedicated `size_ratchet`
  pytest lane as a blocking step (`OURO_SIZE_RATCHET_BASE_REF` names the event
  base; lane placement and base fallback:
  `docs/architecture/08-git-branching-ci-and-build.md` § "CI topology"). Local
  surfaces never block on size: the default pytest lanes exclude the marker,
  and `check_worktree_readiness` plus `codebase_health` report the same
  `validate_size_ratchet` findings as "official CI will enforce" warnings. Why
  a locally evolved fork is never trapped by inherited debt (no
  committed-history replay): `docs/architecture/06-agent-core.md` § "Review
  stack".

### Pragmatic SOLID

SOLID is a direction for making changes legible to future agents, not a demand
for classes or extra framework surface:

- **SRP — Single Responsibility Principle:** keep one coherent reason and one
  clear authority for a unit to change.
- **OCP — Open/Closed Principle:** extend an existing stable seam when it
  preserves the contract instead of rewriting unrelated callers.
- **LSP — Liskov Substitution Principle:** an implementation or backend must
  preserve the caller-visible behavior of the contract it implements.
- **ISP — Interface Segregation Principle:** consumers should depend only on
  the capabilities they actually use, not a broad convenience interface.
- **DIP — Dependency Inversion Principle:** policy should depend on small,
  host-owned contracts rather than provider-specific or concrete details.

Apply these principles pragmatically. They do not require a class hierarchy,
DI container, numeric score, AST analyzer, or a new review pass. A SOLID or
minimalism finding must name the exact symbol or authority, the concrete
duplication or coupling, and a smaller alternative that still satisfies the
contract. Diff size, line count, and file count alone are not findings.
Enforcement: review-only — CHECKLISTS item 2(d) scores these rules in commit
review.

### Shared behavior and data-flow changes

When changing shared behavior or data flow, identify the owning authority,
the identity and scope of its facts, and the affected producers and consumers.
Include unchanged consumers whose inputs or assumptions the change alters.
Preserve the promised semantics across relevant live and recovery paths at
comparable freshness; make legitimate scope or freshness differences explicit.
Verify preservation with falsifiable checks at real consumer boundaries, not
only helper outputs or matching field names. Select the paths from the change
and its dependencies rather than a fixed inventory of surfaces or events.

Enforcement: review-only through the existing scope-review `cross_module_bugs`
and `implicit_contracts` items; no separate gate.

### Invariant: Projection over replay (hot readers of growing stores)

A reader that runs per INTERACTION — an HTTP request, a WS/SSE message, a poll
tick, a task turn — must not replay a growing store to produce its answer.
Interactive read cost must be O(response), achieved through a maintained
projection, a cursor, rotation, or a bounded tail — never a full-history scan
filtered down to the answer.

- **Evaluate the whole operation as the project grows.** For changed data
  readers, consider growth in history, object count and project size, including
  nested repetition, cold caches and concurrent users of shared resources.
  A once-per-boot or explicit-owner scan is still allowed; its cost belongs to
  the whole operation, not separately to every child, file or lookup it visits.
  Where growth can materially hurt responsiveness, show evidence at a
  representative scale on the affected path. First remove redundant work or
  reuse a validated view within one operation; add a projection, cache or other
  mechanism only when that simpler change is insufficient. A batch names its
  observation boundary; the next batch refreshes it, and unknown evidence
  never becomes an empty answer. This is advisory reasoning, not a universal
  time limit, mandatory heavy benchmark for every PR, or a new approval gate.
- **Storage-agnostic.** A full-table read filtered in code IS a replay (a
  `SELECT *` narrowed in Python is the same failure as parsing a whole JSONL
  file for its tail), including unbounded collections INSIDE snapshot/state
  files.
- **Passive GET.** Read handlers perform no NEW steady-state durable writes.
  Exactly two named exceptions exist: (1) substrate-owned integrity repair
  under the substrate's own lock (the usage-ledger torn-tail quarantine in
  `ouroboros/usage_ledger.py`), and (2) one-time idempotent migrations guarded
  by a durable watermark (the legacy usage import). Anything else that "just
  materializes a bit of state" on a GET is a mutation hiding on a read path.
- **House precedents — reuse these shapes:** chat log rotation with
  archive-aware readers (`supervisor/state.py::rotate_chat_log_if_needed`);
  the compact `containment_faults.jsonl` projection maintained beside an
  unbounded event log (`ouroboros/delegate_custody.py`); one shared custody replay
  per context build and per terminal audit (`delegate_terminal.custody_audit_snapshot`,
  consumed by `context_health.build_health_invariants` and `_audit_task_custody`): several
  projections of the same growing store share ONE traversal instead of replaying per reader,
  which bounds the multiplier, not the scan: the read stays O(history) until a compact
  projection replaces it; the fingerprint-keyed
  render cache in `ouroboros/_usage_rows_memo.py` — a projection cached while
  its input is unchanged, invalidated only by advance/refold, never by TTL.
  The `gateway/task_list_scan.py` stat-invalidated result memo and the
  task-event SSE v2 cursor discipline are further precedents; their rules
  (never cache failed or torn reads, physical byte positions through rotation,
  per-source logical EOF per pass, buffers closed before delivery, creation
  stamped only at fresh id allocation, nonfatal `history_gap` disclosure) are
  stated once in ARCHITECTURE "Chat and Projects".

Enforcement: Repo Commit Checklist item 24 (advisory) triggers on diffs that
change data readers, startup/shutdown or other batch operations, or an
endpoint/poller/subscription/timer;
the hot-store growth health invariant
(`agent_startup_checks.py::hot_store_growth_notes`, surfaced by
`context_health.py::build_health_invariants`, thresholds justified in
`ouroboros/context_budget.py`) is the deterministic runtime tripwire. A change
that introduces a new append-only store read on an interactive path must
enroll that store in the `ouroboros/context_budget.py` threshold table (with a
justified constant) in the same commit — an unenrolled hot store is invisible
to the tripwire. Retained execution drives under both `state/headless_tasks`
and `task_drives` are enrolled by direct-child count at
`context_budget.RETAINED_EXECUTION_DRIVES_WARN_COUNT`; startup never recursively
sizes those trees.

### Invariant: Source-complete decision pipeline

Every new or changed continuity surface is reviewed as one narrow chain:

`producer → canonical full source → bounded projection → consumer → decision → retention/GC`.

- The producer records the complete event or artifact before it publishes a
  projection or wakeup. The canonical source owns identity, order, bytes, and
  integrity state; a cache or hot index is never a second authority.
- A bounded projection names what it omitted and carries a source reference
  that the *same actor* can resolve through an existing reader.
  `source_complete` is a coverage fact, not a permission to infer missing
  material.
- Every over-limit tool result persists its exact source; there is no per-tool
  exemption, because the DECIDER must be able to resolve what the actor could
  page. A bounded row whose exact source is durable and referenced is an
  omission for the acceptance panel, and only `source_unavailable` — no
  actor-resolvable source at all — is an unresolved partial that withholds
  dispatch. An `api_chat` acceptance reviewer has no tools and cannot resolve
  `repo_diff_source_ref`, so a criterion that depends on the unseen part of the
  diff is at most `partial`.
- A consumer that can authorize PASS, a destructive rewrite, or replacement of
  a full contract must materialize the named source first. A known `partial`
  marker and an unverified claim that some host might retrieve more are not
  equivalent: the latter is not actor-attested coverage and cannot release the
  decision.
- Retention and GC are part of the chain. Anything referenced by a canonical
  result, review, identity decision, or project summary is retained or promoted
  before its execution root can be collected; an unavailable legacy source is
  represented as an explicit gap, never silently treated as complete.

**Control-plane distrust is metadata, not a data-plane operation.** Paid model
output is evidence until a typed validity predicate fails. Control-plane
distrust — profile, route, parser, window — may lower authority to
DEGRADED/SKIPPED/NOT_RUN, but it must not blank, rewrite, or relabel the
artifact or its original cause.

Enforcement: CHECKLISTS item 25 `source_completeness` (critical when
applicable) scores the chain in commit review; the presentation-adapter
contracts below are pinned by the named web tests.

#### Review presentation adapters

`web/modules/review_presentation.js` (grouping, identity, ordering, typed
presentation state), `ouroboros/review_execution_projection.py` (the bounded
cross-domain `executions[]` wire), `web/modules/review_dom_patch.js` (keyed
in-place DOM reconciliation) and `web/modules/harness_presentation.js`
(harness identity marks and labels) are pure read-side presentation — they
never author, mutate, or feed back canonical verdict, lifecycle, routing,
attention, or enforcement authority, and never infer that a requested route
executed. Admission is source-complete: an incomplete row is omitted, never
guessed from chat, repository, timestamps, model, tool name, or activity.
Reuse the existing chat-history, task-detail, and canonical physical-attempt
readers; do not add a review ledger, endpoint, persisted UI state, cost copy,
or enforcement layer. Money presentation and the folded-group bounds follow ARCHITECTURE "Chat and
Projects" (card cost is sticky task-scope evidence; compact review rows copy or
sum no money). Pin these contracts in
`web/tests/review_presentation.test.js` and
`web/tests/harness_presentation.test.js`; module headers carry the per-module
contracts.

#### Context and growth matrix

| Store / surface | Complete producer and source | Interactive projection / consumer | Growth and retention proof |
|---|---|---|---|
| Background observations | `BackgroundConsciousness.inject_observation` → `state/consciousness_observations.jsonl` enqueue rows | Cached pending/oldest status and bounded `_render_observations` view; identity-update consumer reads the gap marker and source ref | `BG_OBSERVATIONS_WARN_BYTES` in `context_budget.py` / `agent_startup_checks.py`; append-only rows, including unacknowledged rows, are not GC-pruned by the hot-store warning |
| Chat and biography | Canonical `logs/chat.jsonl`, rotated generations, and dialogue blocks | Main/Project context and archive-aware `chat_history` | Rotation/archive readers carry generation/gap coverage; blocks are the compression path, not a deletion of the horizon |
| Plan/review evidence | Exact task-artifact/observability bodies and reviewer route/thread receipts | Bounded review hot index, obligations, and latest-wave status | Exact artifact refs and candidate SHA bind the decision; index rotation cannot certify a missing or partial wave |
| Skill-review root tasks | Per-skill `state/skills/<name>/review_history.jsonl`; `skill_review_runner._append_terminal_history` projects terminal identities to `state/skill_review_root_tasks.jsonl` | `skill_readiness._skill_names_from_review_history` reads a bounded newest-first suffix for acceptance | Derived index is append-only and idempotent by root/task/outcome identity; `SKILL_REVIEW_ROOT_TASKS_WARN_BYTES` warns at 20 MB |
| Task/project execution | Canonical task result plus promoted child artifacts and summaries | Status cards, terminal rows, and Main/Project summary projections | Canonical promotion precedes child-drive GC; disposable task scratch follows the unified retention owner |

### Invariant: Continuation authority and bounded Main projection

Continuation is an explicit relation, not an inferred chat-memory feature: the
router contract requires `predecessor_task_id` (`""` = fresh; omission or
`null` is a typed refusal before any lookup, enqueue, or provider spend), and
Main receives only a defensive provider projection of the predecessor authority
— never a raw head/tail slice, an invented summary, or a mutation of the
canonical result (the contract and the projection rules:
`docs/architecture/01-high-level-architecture.md` § "CLI / Headless Boundary").
The authored continuation narrative is written at the result owner together
with its exact `get_task_result(include_authority=True)` source; the projection
deep-copies the authority, removes only the current task's duplicate nested
predecessor, and thresholds only the closed raw keys `result` and
`final_answer` using `context_budget.PREDECESSOR_RESULT_INLINE_CHARS`.

The startup injection is a bounded continuation ENVELOPE, not a body copy,
minted by the one producer `contracts.task_contract.bounded_continuation_envelope`:
the predecessor's contract core inherits without its nested
`predecessor_authority`, and every field is whole-or-pointer against one strict
serialized budget (previews carry `full_chars` plus a named `source_ref`;
`previous_task_id` keeps the chain walkable). Durable `task_results` bodies are
the untouched SSOT. The bound is per-field, so a pathological row can still
exceed the wire budget — the refusal is typed and loud rather than a silent
$0, and no hop cap exists anywhere: depth belongs to the mind, the floor only
keeps bodies off the wire.

Provider context overflow is a typed recovery fact: after the useful reclaim
and one strictly-smaller same-route retry, a final `context_overflow` skips the
provider-unavailable/forced-provider path, keeps
`execution_status=infra_failed` and `reason_code=llm_api_error`, and records
the typed acceptance bypass and `failure.error_kind`. Ordinary provider
outages keep their existing recovery behavior. Enforcement:
`tests/test_continuation_context_authority.py`.

### Invariant: UI resources carry a disposer

Every long-lived acquisition in `web/` returns or records a disposer, and a UI
instance owns a `destroy()` that releases everything the instance acquired.
The resource kinds this covers: WS subscriptions (`ws.on(...)`),
`document`/`window` event listeners, observers (`ResizeObserver`,
`MutationObserver`, `IntersectionObserver`), timers, `requestAnimationFrame`
loops, and `EventSource`/streaming connections.

An instance that can be closed, hidden, or replaced (project chat panels are
the canonical case) must be destroyable without leaving any acquisition
behind. A UI instance may survive being hidden only under an explicit,
owner-visible retention reason — a project chat with pending work, a widget
card the owner set to Keep running — and even then it owns its disposer, and
Stop / unload / reload / shutdown remain force-destroy boundaries; the reason
is re-evaluated at the instance's next lifecycle point, not continuously. The
untyped shape "hide the DOM node, keep the handlers" remains the leak this
invariant forbids. Late async continuations check a `destroyed` flag before
touching state or re-arming loops. A module widget's disposer is the ordered dispose with acknowledgement
(ARCHITECTURE "Skills and Widgets" owns the sequence and
`WIDGET_DISPOSE_ACK_TIMEOUT_MS`); the masonry's `applyMasonry` returns an
idempotent disposer for its observers and pending frame. That bounded wait is not the forbidden shape: the handlers live
only until the settle promise the page tracks per card key resolves.

Enforcement (honest disclosure): the deterministic leak test runs in the
release-tier `ui_browser` lane, not at commit tier; commit-tier coverage is
the advisory Repo Commit Checklist item 24. The class is closed
deterministically for the instrumented surfaces and advisorily for future
ones.

### Invariant: Embedded surfaces declare geometry and refresh semantics

Every owner-visible embedded or framed surface has an explicit host-owned
geometry/overflow contract, a paired disposer for every long-lived resource,
declared refresh/stream/error semantics, and a named real-consumer visual
verification path. Intentional omissions record why they are safe to defer.
For Widgets, framed `height` values are bounded and module auto-height is
host-controlled. Below its finite ceiling, applying a reported block size must
not change the child's inline-size basis; the host owns vertical scrollbar
mode without disabling the orthogonal horizontal overflow capability, and
content measurement includes the measured document's bottom padding and border
(three separate feedback-loop bugs encoded as one rule).
Feedback-sensitive verification is event-driven on the relevant engine: it
proves temporal convergence to a quiet fixed point with a real consumer or
production-derived fixture that crosses the known wrapping threshold, rather
than comparing two snapshots. A module widget's own faults are declared error semantics, not silence (the
`ouro-widget-error` channel: ARCHITECTURE "Skills and Widgets"). There is no
server-side widget fault ledger, so those faults are visible only while the
Widgets page is open; that is safe to defer because the browser is the
verification path for a widget in the first place.
Module source loading and declarative requests
have a bounded host timeout; declarative job widgets keep their `job_id` and
bounded retry/timeout behavior visible in the refresh contract. Missing or
malformed job status is an immediate protocol error, while unknown non-empty
in-progress labels remain bounded pending states for producer compatibility.
Repo Commit Checklist item 24 points lifecycle changes here instead of
re-deriving a second domain-specific rule; the widget geometry/refresh
contracts are pinned in `tests/test_widgets_ui_static.py` and
`tests/test_extension_surfaces.py`.

---

