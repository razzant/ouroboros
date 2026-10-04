# Module Size & Complexity

This chapter owns deterministic line/function/byte gates, their debt manifest, and paydown by simplifying in place rather than extracting passthroughs. Its readability invariants cover hot-reader projections, the source-complete decision pipeline, continuation authority, UI disposers, and embedded-surface geometry/refresh. Each rule bounds reading cost and names its enforcement or absence.

P7 makes context fit a maintenance constraint, not a line-count aesthetic.

- Python modules everywhere (including `tests/` and `devtools/`) and
  first-party `web/**/*.js` modules (including `web/tests/`; vendored/minified
  excluded) target roughly 1000 lines. The deterministic hard gates read
  exact-path debt from the manifest: 1600 lines per module
  (`ouroboros/size_ratchet_manifest.py::GIANT_PATHS`), 200,000 UTF-8 bytes per
  module (`BYTE_DEBT`, shrink-only) and the 1001–1500 band (`BAND_PATHS`; an
  entry may stay at 1501–1600 lines; a new entry needs a nonblank rationale) —
  all three apply to Python and JavaScript alike — and 300 lines per
  non-grandfathered function (`FUNCTION_DEBT`, exact `(path, qualname)` keys),
  which sees the runtime-Python function inventory only (the iterator skips
  `tests/`, `devtools/`, JavaScript and `FUNCTION_COUNT_EXCLUDED_FILES`). A
  stale or newly oversized entry fails. Regenerate with
  `scripts/regenerate_size_ratchet.py`; it validates the rendered candidate
  before writing and refuses an unmerged index with a typed error. Counts are
  taken on strict-UTF-8, POSIX-LF-normalized text, so checkout policy cannot
  change the inventory; the one shared iterator and the manifest mechanics:
  ARCHITECTURE §6 "Review stack".
- Paying down a size cap: a change that runs into a band, the hard cap, the
  function-size gate or the byte debt pays it down by SIMPLIFYING where the
  change lives — simpler control/data flow and interfaces, dead code and
  duplicates removed, an existing SSOT reused, prose made compact and legible —
  so the module reads BETTER afterwards. Extracting a helper, a passthrough
  wrapper or a neighbour module is the LAST resort, and a paydown only when the
  new unit is a natural boundary that would stay correct with the parent far
  below the cap: its own reason-to-change, an explicit contract, a real caller.
  A cap-driven bucket, a one-caller passthrough, or bytes bought by deleting
  contract-bearing comments, docstrings, messages or tests is a defect, not
  paydown — report the conflict instead (BIBLE P7 «first simplify what
  exists»). Enforcement: Repo Commit Checklist item 31 `size_cap_paydown`,
  advisory when applicable.
- Methods above 150 lines and more than eight parameters are decomposition
  signals (BIBLE P7, CHECKLISTS item 2(c)), not deterministic gates; existing
  baseline debt is not retroactively a failing tree.
- Runtime Python function/method totals are descriptive inventory, with no
  aggregate ceiling or remaining quota. A repository-wide count does not
  measure per-unit complexity and can penalize useful decomposition. Module
  line/UTF-8 byte limits, function-length limits, exact debt manifests and
  shrink-only debt transitions retain their authority; their inventories and
  scope remain unchanged.
- Enforcement: the OFFICIAL repository's CI runs the dedicated `size_ratchet`
  pytest lane as a blocking step (`OURO_SIZE_RATCHET_BASE_REF` names the event
  base; lane placement and base fallback: ARCHITECTURE §8 "CI topology").
  Local surfaces never block on size: the default pytest lanes exclude the
  marker, and `check_worktree_readiness` and `codebase_health` report the same
  `validate_size_ratchet` findings as "official CI will enforce" warnings.
  Both readouts also show descriptive totals and per-unit capacity from the
  same inventory and current limits; readiness passes this information
  separately from warnings and focuses on touched paths.
  Registered debt and omitted rows are labelled; a nearly full valid module
  remains admissible. Why a locally evolved fork is never trapped by inherited
  debt (no committed-history replay): ARCHITECTURE §6 "Review stack".

### Pragmatic SOLID

SOLID is a direction for making changes legible to future agents, not a demand
for classes or extra framework surface:

- **SRP — Single Responsibility Principle:** one coherent reason and one clear
  authority for a unit to change.
- **OCP — Open/Closed Principle:** extend an existing stable seam that
  preserves the contract instead of rewriting unrelated callers.
- **LSP — Liskov Substitution Principle:** an implementation or backend
  preserves the caller-visible behavior of the contract it implements.
- **ISP — Interface Segregation Principle:** consumers depend only on the
  capabilities they actually use, not a broad convenience interface.
- **DIP — Dependency Inversion Principle:** policy depends on small,
  host-owned contracts, not provider-specific or concrete details.

Apply them pragmatically: they require no class hierarchy,
DI container, numeric score, AST analyzer, or new review pass. A SOLID or
minimalism finding must name the exact symbol or authority, the concrete
duplication or coupling, and a smaller alternative that still satisfies the
contract.
Diff size, line count, and file count alone are not findings.
Enforcement: review-only — CHECKLISTS item 2(d) scores these rules in commit
review.

### Shared behavior and data-flow changes

When changing shared behavior or data flow, identify the owning authority, the
identity and scope of its facts, and the affected producers and consumers,
including unchanged consumers whose inputs or assumptions the change alters.
Preserve the promised semantics across the relevant live and recovery paths at
comparable freshness, and make legitimate scope or freshness differences
explicit. Verify preservation with falsifiable checks at real consumer
boundaries, not only helper outputs or matching field names, selecting the
paths from the change and its dependencies rather than a fixed inventory of
surfaces or events. Enforcement: review-only through the existing scope-review
`cross_module_bugs` and `implicit_contracts` items; no separate gate.

### Invariant: Projection over replay (hot readers of growing stores)

A reader that runs per INTERACTION — an HTTP request, a WS/SSE message, a poll
tick, a task turn — must not replay a growing store to produce its answer.
Interactive read cost is O(response), through a maintained projection, a
cursor, rotation or a bounded tail — never a full-history scan filtered down to
the answer.

- **Evaluate the whole operation as the project grows.** For a changed data
  reader, weigh growth in history, object count and project size, including
  nested repetition, cold caches and concurrent users of shared resources. A
  once-per-boot or explicit-owner scan is still allowed; its cost belongs to
  the whole operation, not separately to every child, file or lookup it
  visits. Where growth can materially hurt responsiveness, show evidence at a
  representative scale on the affected path. First remove redundant work or
  reuse a validated view within one operation; add a projection, cache or
  other mechanism only when that simpler change is insufficient. A batch names
  its observation boundary, the next batch refreshes it, and unknown evidence
  never becomes an empty answer. This is advisory reasoning, not a universal
  time limit, mandatory heavy benchmark for every PR, or a new approval gate.
- **Storage-agnostic.** A full-table read filtered in code IS a replay (a
  `SELECT *` narrowed in Python, a whole JSONL file parsed for its tail),
  including unbounded collections INSIDE snapshot/state files.
- **Passive GET.** Read handlers perform no NEW steady-state durable writes.
  Exactly two named exceptions exist: (1) substrate-owned integrity repair
  under the substrate's own lock (the usage-ledger torn-tail quarantine in
  `ouroboros/usage_ledger.py`), and (2) one-time idempotent migrations guarded
  by a durable watermark (the legacy usage import). Anything else that
  materializes state on a GET is a mutation hiding on a read path.
- **House precedents — reuse these shapes:** archive-aware chat log rotation
  (`supervisor/state.py::rotate_chat_log_if_needed`); the compact
  `containment_faults.jsonl` projection maintained beside an unbounded event
  log (`ouroboros/delegate_custody.py`); the process-local custody row memo
  behind `delegate_custody.custody_rows` (`ouroboros/delegate_custody_memo.py`:
  an ordered inode/size/mtime fingerprint of the rotated chain prefix, only
  appended bytes folded, a refold on any doubt, a bypass while unreadable — it
  bounds the warm read, not the cold fold, so a durable compact projection
  stays the next step); the bounded filtered tail reader
  `ouroboros/jsonl_tail.py` (doubling live tail, three newest archives,
  coverage facts) for history endpoints and the per-task recent-activity
  sections alike; the
  fingerprint-keyed render cache in `ouroboros/_usage_rows_memo.py`, held while
  its input is unchanged and invalidated only by advance/refold, never by TTL;
  the `gateway/task_list_scan.py` stat-invalidated result memo and the
  task-event SSE v2 cursor discipline, whose rules are stated once in
  ARCHITECTURE §3 "Chat and Projects".

Enforcement: Repo Commit Checklist item 24 (advisory) triggers on diffs that
change data readers, startup/shutdown or other batch operations, or an
endpoint/poller/subscription/timer; the deterministic runtime tripwire is
`agent_startup_checks.py::hot_store_growth_notes`, surfaced by
`context_health.py::build_health_invariants`, with thresholds justified in
`ouroboros/context_budget.py`. A change that introduces a new append-only store
read on an interactive path enrolls that store in that threshold table, with a
justified constant, in the same commit — an unenrolled hot store is invisible
to the tripwire. Retained execution drives under both
`state/headless_tasks` and `task_drives` are enrolled by direct-child count at
`context_budget.RETAINED_EXECUTION_DRIVES_WARN_COUNT`; startup never
recursively sizes those trees.

### Invariant: Source-complete decision pipeline

Every new or changed continuity surface is reviewed as one narrow chain:

`producer → canonical full source → bounded projection → consumer → decision → retention/GC`.

- The producer records the complete event or artifact before it publishes a
  projection or wakeup. The canonical source owns identity, order, bytes and
  integrity state; a cache or hot index is never a second authority.
- A bounded projection names what it omitted and carries a source reference
  that the *same actor* can resolve through an existing reader.
  `source_complete` is a coverage fact, not a permission to infer missing
  material.
- Every over-limit tool result persists its exact source; there is no per-tool
  exemption, because the DECIDER must be able to resolve what the actor could
  page. A bounded row whose exact source is durable and referenced is an
  omission for the acceptance panel. If its primary handle fails, a verified
  matching redacted tool projection may recover the recorded pre-truncation
  result through the existing source writer. Preserve complete inline evidence
  on publication failure; a later cut or whole-row omission must not reuse the failed handle or the
  original partial corpus. Only `source_unavailable` — no
  actor-resolvable source at all — is an unresolved partial that withholds
  dispatch. An `api_chat` acceptance reviewer has no tools and cannot resolve
  `repo_diff_source_ref`, so a criterion that depends on the unseen part of the
  diff is at most `partial`.
- A consumer that can authorize PASS, a destructive rewrite, or replacement of
  a full contract materializes the named source first. A known `partial`
  marker and an unverified claim that some host might retrieve more are not
  equivalent: the latter is not actor-attested coverage and cannot release the
  decision.
- Retention and GC are part of the chain. Anything referenced by a canonical
  result, review, identity decision, or project summary is retained or promoted
  before its execution root can be collected; an unavailable legacy source is
  an explicit gap, never silently treated as complete.

**Control-plane distrust is metadata, not a data-plane operation.** Paid model
output is evidence until a typed validity predicate fails. Actual profile,
route or subject mismatches, invalid output contracts and delivery failures may
affect review authority; window sizing and reading diagnostics alone may not
(BIBLE P3). Neither case blanks, rewrites or relabels the artifact or its
original cause.

Enforcement: CHECKLISTS item 25 `source_completeness` (critical when
applicable) scores the chain in commit review; the presentation-adapter
contracts below are pinned by the named web tests.

#### Review presentation adapters

`web/modules/review_presentation.js` (grouping, identity, ordering, typed
presentation state), `ouroboros/review_execution_projection.py` (the bounded
cross-domain `executions[]` wire), `web/modules/review_dom_patch.js` (keyed
in-place DOM reconciliation) and `web/modules/harness_presentation.js`
(harness identity marks and labels) are pure read-side presentation: they
never author, mutate or feed back canonical verdict, lifecycle, routing,
attention or enforcement authority, and never infer that a requested route
executed. Admission is source-complete — an incomplete row is omitted, never
guessed from chat, repository, timestamps, model, tool name or activity. Reuse
the existing chat-history, task-detail and canonical physical-attempt readers;
add no review ledger, endpoint, persisted UI state, cost copy or enforcement
layer. Money presentation and the folded-group bounds follow ARCHITECTURE §3
"Chat and Projects"; compact review rows copy or sum no money. Enforcement:
`web/tests/review_presentation.test.js` and
`web/tests/harness_presentation.test.js` pin these contracts; module headers
carry the per-module contracts.

#### Context and growth matrix

| Store / surface | Complete producer and source | Interactive projection / consumer | Growth and retention proof |
|---|---|---|---|
| Chat and biography | Canonical `logs/chat.jsonl`, rotated generations, and `memory/chronicle/records.jsonl` (dialogue blocks stay read-only legacy) | the memory view (`memory_view.py`) and archive-aware `chat_history` | Rotation/archive readers carry generation/gap coverage; pages and parts are the compression path, not a deletion of the horizon; the journal warns at 64 MB (`CHRONICLE_JOURNAL_WARN_BYTES`) |
| Plan/review evidence | Exact task-artifact/observability bodies and reviewer route/thread receipts | Bounded review hot index, obligations, and latest-wave status | Exact artifact refs and candidate SHA bind the decision; index rotation cannot certify a missing or partial wave |
| Skill-review root tasks | Per-skill `state/skills/<name>/review_history.jsonl`; `skill_review_runner._append_terminal_history` projects terminal identities to `state/skill_review_root_tasks.jsonl` | `skill_readiness._skill_names_from_review_history` reads a bounded newest-first suffix for acceptance | Derived index is append-only and idempotent by root/task/outcome identity; `SKILL_REVIEW_ROOT_TASKS_WARN_BYTES` warns at 20 MB |
| Task/project execution | Canonical task result plus promoted child artifacts and summaries | Status cards, terminal rows, and Main/Project summary projections | Canonical promotion precedes child-drive GC; disposable task scratch follows the unified retention owner |

### Invariant: Continuation authority and bounded Main projection

Continuation is an explicit relation, not an inferred chat-memory feature: the
router contract requires `predecessor_task_id` (`""` = fresh; omission or
`null` is a typed refusal before any lookup, enqueue, or provider spend), and
Main receives only a defensive provider projection of the predecessor authority
— never a raw head/tail slice, an invented summary, or a mutation of the
canonical result (the contract and the projection rules: ARCHITECTURE §1
"CLI / Headless Boundary"). A legacy authored continuation narrative is read at
the result owner with its exact `get_task_result(include_authority=True)`
source; the projection thresholds only the closed raw keys `result` and
`final_answer`, at `context_budget.PREDECESSOR_RESULT_INLINE_CHARS`.

The startup injection is a bounded continuation ENVELOPE, not a body copy,
minted by the one producer `contracts.task_contract.bounded_continuation_envelope`:
the predecessor's contract core inherits without its nested
`predecessor_authority`, every field is whole-or-pointer against one strict
serialized budget (a preview carries `full_chars` plus a named `source_ref`;
`previous_task_id` keeps the chain walkable), and durable `task_results` bodies
stay the untouched SSOT. Disclosed: the bound is per-field, so a pathological
row can still exceed the wire budget — the refusal is typed and loud rather
than a silent $0. No hop cap exists anywhere: depth belongs to the mind, the
floor only keeps bodies off the wire.

Provider context overflow is a typed recovery fact: after the useful reclaim
and one strictly-smaller same-route retry, a final `context_overflow` skips the
provider-unavailable/forced-provider path, keeps
`execution_status=infra_failed` and `reason_code=llm_api_error`, and records
the typed acceptance bypass and `failure.error_kind`; ordinary provider outages
keep their existing recovery behavior. Enforcement:
`tests/test_continuation_context_authority.py`.

### Invariant: notifications ring for live events only

Owner-facing notification POLICY is `docs/DESIGN.md` §9 — one canonical
section, never re-derived here. The engineering rules are:

The subscription is CLIENT-level and must never move into a chat instance. An
instance dies with its room — closing a Project panel disposes its `ws.on`
handlers — so a notifier wired inside one is silent in exactly the case
notifications exist for: the owner left and the room is closed.
`notifications.js::attach()` takes one subscription on the shared socket in
`app.js`, and `chat.js` holds no notification code.

Only live frames reach it; history and reconnect backfill run through the
instances' own readers, which never call it. That boundary — not a persisted
ledger — is what makes replay safe, so no notification state survives a reload.
The room gate is the client's owner-visible chat set: the hidden partition, A2A
ids and unknown chats are refused.

One ending is one key per task: the `task_done` log frame, the authored summary
and the turn's ordinary reply all collapse together, and a direct turn's ending
is the ordinary-reply category rather than a finished task. Lineage comes from
the delegation facts frames carry, because the terminal frame has none — a child
must not reach the owner's banner.

Classification and delivery gating are pure over one frame and stored preferences,
testable without a DOM or socket. Client-local preferences have no `s-` field,
are excluded from the settings-dirty tracker, never reach `/api/settings` or
prompt to discard unsaved settings (`tests/test_notifications_static.py` asserts
these causes, not just effects). Delivery degrades instead of disappearing;
the status line identifies this client's surface. Feature-detect the optional
desktop bridge per call at delivery: its result is capability evidence, not a
banner/delivery claim. It may raise the existing window and request one system
sound; no scheduler, persistence or background process. Importance adds no host
field, text heuristic or second model call.

### Invariant: UI resources carry a disposer

Every long-lived acquisition in `web/` returns or records a disposer, and a UI
instance owns a `destroy()` that releases everything the instance acquired: WS
subscriptions (`ws.on(...)`), `document`/`window` event listeners, observers
(`ResizeObserver`, `MutationObserver`, `IntersectionObserver`), timers,
`requestAnimationFrame` loops, and `EventSource`/streaming connections.

An instance that can be closed, hidden, or replaced (project chat panels are
the canonical case) must be destroyable without leaving any acquisition
behind. It may survive being hidden only under an explicit, owner-visible
retention reason — a project chat with pending work, a widget card the owner
set to Keep running — and even then it owns its disposer, Stop / unload /
reload / shutdown remain force-destroy boundaries, and the reason is
re-evaluated at the instance's next lifecycle point, not continuously. The
untyped shape "hide the DOM node, keep the handlers" remains the leak this
invariant forbids; late async continuations check a `destroyed` flag before
touching state or re-arming loops. A module widget's disposer is the ordered
dispose with acknowledgement (the sequence and `WIDGET_DISPOSE_ACK_TIMEOUT_MS`:
ARCHITECTURE §3 "Skills and Widgets") — that bounded wait is not the forbidden
shape, because its handlers live only until the settle promise the page tracks
per card key resolves; the masonry's `applyMasonry` returns an idempotent
disposer for its observers and pending frame.

Enforcement (honest disclosure): the deterministic leak test runs in the
release-tier `ui_browser` lane, not at commit tier; commit-tier coverage is
the advisory Repo Commit Checklist item 24. The class is closed
deterministically for the instrumented surfaces and advisorily for future
ones.

### Invariant: Embedded surfaces declare geometry and refresh semantics

Every owner-visible embedded or framed surface has an explicit host-owned
geometry/overflow contract, a paired disposer for every long-lived resource,
declared refresh/stream/error semantics, and a named real-consumer visual
verification path; intentional omissions record why they are safe to defer.

For Widgets, framed `height` values are bounded and module auto-height is
host-controlled. Below its finite ceiling, applying a reported block size must
not change the child's inline-size basis; the host owns vertical scrollbar
mode without disabling the orthogonal horizontal overflow capability; and
content measurement includes the measured document's bottom padding and
border. Feedback-sensitive verification is event-driven on the relevant
engine: it proves temporal convergence to a quiet fixed point with a real
consumer or production-derived fixture that crosses the known wrapping
threshold, rather than comparing two snapshots.

A module widget's own faults are declared error semantics, not silence (the
`ouro-widget-error` channel: ARCHITECTURE §3 "Skills and Widgets"). There is
no server-side widget fault ledger, so those faults are visible only while the
Widgets page is open — safe to defer because the browser is the verification
path for a widget in the first place. Module source loading and declarative
requests have a bounded host timeout; declarative job widgets keep their
`job_id` and bounded retry/timeout behavior visible in the refresh contract,
where missing or malformed job status is an immediate protocol error while an
unknown non-empty in-progress label remains a bounded pending state for
producer compatibility.

Enforcement: Repo Commit Checklist item 24 points lifecycle changes here
instead of re-deriving a second domain-specific rule; the widget
geometry/refresh contracts are pinned in `tests/test_widgets_ui_static.py` and
`tests/test_extension_surfaces.py`.

---
