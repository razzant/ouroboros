# Module Size & Complexity

This chapter owns the deterministic size gates and their debt manifest, paydown by
simplifying in place rather than extracting passthroughs, and the reading-cost
invariants reviewers check against: hot-reader projections, the source-complete
decision pipeline, continuation authority, live-only notifications, UI disposers and
embedded-surface geometry. Each rule bounds reading cost and names its enforcement or
its absence.

P7 makes context fit a maintenance constraint, not a line-count aesthetic. Python
modules everywhere (`tests/` and `devtools/` included) and first-party `web/**/*.js`
(`web/tests/` included; vendored and minified excluded) target roughly 1000 lines. The
gates read exact-path debt from `ouroboros/size_ratchet_manifest.py`, the same for
Python and JavaScript: 1600 lines per module (`GIANT_PATHS`), 200,000 canonical
UTF-8/LF bytes (`BYTE_DEBT`, shrink-only), the 1001–1500 band (`BAND_PATHS`; a new
entry needs a nonblank rationale), and 300 lines per non-grandfathered runtime-Python
function (`FUNCTION_DEBT`; `FUNCTION_COUNT_EXCLUDED_FILES` names the skipped files).
A stale or newly oversized entry fails; regenerate with
`scripts/regenerate_size_ratchet.py` (mechanics: ARCHITECTURE §6 "Review stack").
Methods over 150 lines or eight parameters are decomposition signals, not gates
(CHECKLISTS item 11(c); BIBLE P7). Runtime function totals are descriptive, with no
aggregate ceiling, because a repository-wide count penalizes useful decomposition.

Enforcement: the official repository's CI runs the `size_ratchet` pytest lane as a
blocking step (`OURO_SIZE_RATCHET_BASE_REF` names the event base; ARCHITECTURE §8 "CI
topology"). Local surfaces never block on size: default lanes exclude the marker, and
`check_worktree_readiness` and `codebase_health` report `validate_size_ratchet`
findings as "official CI will enforce" warnings, so a locally evolved fork is never
trapped by inherited debt.

### Paying down a size cap

A change that runs into a band, the hard cap, the function gate or the byte debt pays it
down by simplifying where the change lives (simpler control and data flow, dead code and
duplicates removed, an existing SSOT reused, prose made compact), so the module reads
better afterwards. Extraction is the last resort and counts only for a natural boundary
that would stay correct with the parent far below the cap: its own reason to change, an
explicit contract, a real caller. A cap-driven bucket, a one-caller passthrough, or
bytes bought by deleting contract-bearing comments, docstrings, messages or tests is a
defect to report, not paydown (BIBLE P7 «first simplify what exists»). Enforcement:
CHECKLISTS item 31 `size_cap_paydown`, advisory when applicable.

### Pragmatic SOLID

SOLID is a direction for making changes legible to future agents, not a demand for
classes or framework surface:

- **SRP — Single Responsibility Principle:** one coherent reason and one authority for
  a unit to change.
- **OCP — Open/Closed Principle:** extend an existing stable seam that preserves the
  contract instead of rewriting unrelated callers.
- **LSP — Liskov Substitution Principle:** an implementation or backend preserves the
  caller-visible behavior of the contract it implements.
- **ISP — Interface Segregation Principle:** consumers depend only on the capabilities
  they use, not a broad convenience interface.
- **DIP — Dependency Inversion Principle:** policy depends on small host-owned
  contracts, not provider-specific or concrete details.

No class hierarchy, DI container, numeric score, AST analyzer or new review pass
follows. A SOLID or minimalism finding names the exact symbol or authority, the
concrete duplication or coupling, and a smaller alternative that still satisfies the
contract. Diff size, line count, and file count alone are not findings. Enforcement:
review-only; CHECKLISTS item 11(d).

### Shared behavior and data-flow changes

When changing shared behavior or data flow, identify the owning authority, the identity
and scope of its facts, and the affected producers and consumers, including unchanged
consumers whose inputs or assumptions the change alters. Preserve the promised
semantics across live and recovery paths at comparable freshness, and make legitimate
differences explicit. Verify with falsifiable checks at real consumer boundaries, not
helper outputs or matching field names, choosing the paths from the change and its
dependencies rather than a fixed inventory. Enforcement: review-only, through the
scope-review `cross_module_bugs` and `implicit_contracts` items.

### Invariant: Projection over replay (hot readers of growing stores)

Apply ARCHITECTURE §10 invariant 10 to current-state readers; review and growth checks follow.

- **Evaluate the whole operation as the project grows:** history, object count, nested
  repetition, cold caches, shared-resource contention. Where growth can hurt
  responsiveness, show representative-scale evidence from monotonic spans and counts
  in the event log, never new history reads for telemetry. Remove redundant work or
  reuse a validated view before adding a
  projection or cache. Unknown evidence never means empty. This advisory reasoning sets
  no universal time limit, mandatory benchmark or extra approval gate.
- **Passive GET.** Read handlers perform no new steady-state durable writes.
  `ouroboros/usage_store.py::read` may initialize an empty store when nothing awaits
  import; display reads leave pending imports to startup and report unavailable data.
  Enforced by `tests/test_usage_store.py`; the read path never starts a history import.
- **House precedents, reuse these shapes:** archive-aware chat log rotation
  (`supervisor/state.py::rotate_chat_log_if_needed`); a compact projection beside an
  unbounded event log with a warm-read memo (`ouroboros/delegate_custody.py`,
  `ouroboros/delegate_custody_memo.py`); the bounded filtered tail
  (`ouroboros/jsonl_tail.py`); per-archive room summaries
  (`ouroboros/gateway/history_segments.py`); summary rows kept by the writing
  transaction (`ouroboros/usage_store.py`); the stat-invalidated memo
  (`task_result_facts.py`); the task-event SSE v2 cursor (ARCHITECTURE §3 "Chat and
  Projects").

Enforcement: CHECKLISTS item 9 `perf_lifecycle` (advisory) fires on diffs that change
data readers, startup/shutdown or batch operations, or an endpoint, poller,
subscription or timer. The runtime tripwire is
`agent_startup_checks.py::hot_store_growth_notes` (thresholds in
`ouroboros/context_budget.py`); a change adding an append-only store read on an
interactive path enrolls that store there, with a justified constant, in the same
commit, because an unenrolled hot store is invisible to the tripwire.

### Invariant: Source-complete decision pipeline

Every new or changed continuity surface is reviewed as one narrow chain:

`producer → canonical full source → bounded projection → consumer → decision → retention/GC`.

- The producer records the complete event or artifact before publishing a projection
  or wakeup; the canonical source owns identity, order, bytes and integrity state, and
  a cache or hot index is never a second authority.
- A bounded projection names what it omitted and carries a source reference the *same
  actor* can resolve through an existing reader; `source_complete` is a coverage fact,
  not permission to infer missing material. Every over-limit tool result persists its
  exact source, with no per-tool exemption; only `source_unavailable` (no
  actor-resolvable source at all) is an unresolved partial that withholds dispatch.
- An `api_chat` acceptance reviewer has no tools to resolve `repo_diff_source_ref`;
  a criterion depending on the unseen diff remains at most `partial`.
- A consumer that can authorize PASS, a destructive rewrite or replacement of a full
  contract materializes the named source first; a known `partial` marker and an
  unverified claim that some host might retrieve more are not equivalent.
- Anything referenced by a canonical result, review, identity decision or project
  summary is retained or promoted before its execution root is collected; an
  unavailable legacy source is an explicit gap, never silently complete.

Control-plane distrust is metadata, not a data-plane operation: paid model output is
evidence until a typed validity predicate fails. Profile, route or subject mismatches,
invalid output contracts and delivery failures may affect review authority; window
sizing and reading diagnostics alone may not (BIBLE P3). Neither case blanks, rewrites
or relabels the artifact.

Enforcement: CHECKLISTS item 21 `source_completeness` (critical when applicable).

#### Review presentation adapters

`web/modules/review_presentation.js`, `ouroboros/review_execution_projection.py` (the
bounded cross-domain `executions[]` wire), `web/modules/review_dom_patch.js` and
`web/modules/harness_presentation.js` are pure read-side presentation: they never
author, mutate or feed back canonical verdict, lifecycle, routing, attention or
enforcement authority, never infer that a requested route executed, and omit an
incomplete row rather than guess it. They reuse the existing chat-history, task-detail
and physical-attempt readers and add no review ledger, endpoint, persisted UI state or
cost copy (ARCHITECTURE §3 "Chat and Projects"). Enforcement:
`web/tests/review_presentation.test.js`, `web/tests/harness_presentation.test.js`.

#### Context and growth matrix

The canonical continuity map is ARCHITECTURE §10 "Continuity data-flow map", which
carries the source, projection and retention rule of chat and biography, plan/review
authority and execution evidence. Record a new hot store there, add its growth proof
here, and enroll it in `hot_store_growth_notes` in the same change.

| Store / surface | Complete source → interactive projection | Growth and retention proof |
|---|---|---|
| Chat and biography | `logs/chat.jsonl` with its rotated generations and `memory/chronicle/records.jsonl` → the memory view (`memory_view.py`) and archive-aware `chat_history` | Rotation and archive readers carry generation/gap coverage; pages and parts compress the horizon, never delete it; the journal warns at `CHRONICLE_JOURNAL_WARN_BYTES` |
| Plan/review evidence | ARCHITECTURE §10.1, "Plan/review authority" | Source and retention rules live in that row |
| Skill-review root tasks | Per-skill `state/skills/<name>/review_history.jsonl`; `skill_review_runner._append_terminal_history` projects terminal identities to `state/skill_review_root_tasks.jsonl` → `skill_readiness._skill_names_from_review_history` reads a bounded newest-first suffix | Append-only and idempotent by root/task/outcome identity; warns at `SKILL_REVIEW_ROOT_TASKS_WARN_BYTES` |
| Task/project execution | ARCHITECTURE §10.1, "Execution evidence" | Promotion and retention rules live in that row |

### Invariant: Continuation authority and bounded role projections

Continuation is an explicit relation, not an inferred chat-memory feature: the router
contract requires `predecessor_task_id` (`""` = fresh; omission or `null` is a typed
refusal before any lookup, enqueue or provider spend), and Main receives only a
defensive projection of the predecessor authority, never a raw head/tail slice, an
invented summary or a mutation of the canonical result (ARCHITECTURE §1 "CLI /
Headless Boundary"); an exact read goes to `get_task_result(include_authority=True)`.

The root startup injection is a bounded envelope, not a body copy, minted by the one
producer `contracts.task_contract.bounded_continuation_envelope`: every field is
whole-or-pointer against a per-field serialized budget (a preview carries `full_chars` and a
`source_ref`; `previous_task_id` keeps the chain walkable), durable `task_results`
bodies stay the untouched SSOT, and no hop cap exists: depth belongs to the mind, the
floor only keeps bodies off the wire. Children and external work orders receive the
pure, idempotent brief `main_context_authority.project_helper_predecessor_authority`;
a brief never stands in for the omitted evidence. Enforcement:
`tests/test_continuation_context_authority.py`.

### Invariant: notifications ring for live events only

Owner-facing notification policy is `docs/DESIGN.md` §9, one canonical section, never
re-derived here. The engineering rules:

The subscription is client-level and never moves into a chat instance: an instance dies
with its room (closing a Project panel disposes its `ws.on` handlers), so a notifier
inside one is silent in exactly the case notifications exist for, the owner gone and
the room closed. `notifications.js::attach()` takes one subscription on the shared
socket in `app.js`; `chat.js` holds no notification code.

Only live frames reach it; history and reconnect backfill run through the instances'
own readers, which never call it. That boundary, not a persisted ledger, makes replay
safe, so no notification state survives a reload. The room gate is the client's
owner-visible chat set (hidden partition, A2A ids and unknown chats refused); one
ending is one key per task (a direct turn's ending is an ordinary reply, not a
finished task); lineage comes from the delegation facts frames carry, so a child never
reaches the owner's banner.

Classification and gating are pure over one frame and stored preferences, testable
without a DOM or socket. Preferences are client-local: no `s-` field, excluded from the
settings-dirty tracker, never sent to `/api/settings`. Delivery degrades instead of
disappearing and never asks for OS permission itself; importance adds no host field,
text heuristic or second model call (mechanism: the `web/modules/notifications.js`
header). Enforcement: `tests/test_notifications_static.py` asserts these causes, not
just effects.

### Invariant: UI resources carry a disposer

Every long-lived acquisition in `web/` returns or records a disposer, and a UI instance
owns a `destroy()` that releases everything it acquired: WS subscriptions
(`ws.on(...)`), `document`/`window` listeners, observers (`ResizeObserver`,
`MutationObserver`, `IntersectionObserver`), timers, `requestAnimationFrame` loops and
`EventSource`/streaming connections.

An instance that can be closed, hidden or replaced (project chat panels are the
canonical case) is destroyable without leaving an acquisition behind. It may survive
being hidden only under an explicit, owner-visible retention reason (a project chat
with pending work, a widget card set to Keep running); even then it owns its disposer,
and Stop, unload, reload and shutdown stay force-destroy boundaries. "Hide the DOM
node, keep the handlers" is the leak this invariant forbids; late async continuations
check a `destroyed` flag before touching state or re-arming loops. A module widget's
disposer is the ordered dispose with acknowledgement (`WIDGET_DISPOSE_ACK_TIMEOUT_MS`;
ARCHITECTURE §3 "Skills and Widgets").

Enforcement: the deterministic leak test runs in the release-tier `ui_browser` lane;
commit-tier coverage is the advisory CHECKLISTS item 9 `perf_lifecycle`. The class is
closed deterministically for the instrumented surfaces and advisorily for future ones.

### Invariant: Embedded surfaces declare geometry and refresh semantics

Every owner-visible embedded or framed surface has an explicit host-owned
geometry/overflow contract, a paired disposer for every long-lived resource, declared
refresh/stream/error semantics, and a named real-consumer visual verification path; an
intentional omission records why deferring it is safe.

For Widgets, framed `height` values are bounded and module auto-height is
host-controlled below a finite ceiling. Below that ceiling, updates preserve the
child's inline-size basis; the host controls vertical scrollbar mode while preserving
horizontal overflow. Feedback-sensitive verification is event-driven on the relevant engine and proves
convergence to a quiet fixed point with a real consumer or production-derived fixture
that crosses the known wrapping threshold, not two snapshots. A module widget's own
faults are declared error semantics, not silence (the `ouro-widget-error` channel:
ARCHITECTURE §3 "Skills and Widgets"); with no server-side widget fault ledger they are
visible only while the Widgets page is open, safe to defer because the browser is a
widget's verification path. Module source loading and declarative requests have a
bounded host timeout; declarative job widgets keep their `job_id` and bounded
retry/timeout behavior in the refresh contract, where missing or malformed job status
is an immediate protocol error and an unknown in-progress label is a bounded pending
state.

Enforcement: CHECKLISTS item 9 `perf_lifecycle` points lifecycle changes here; the
widget geometry/refresh contracts are pinned in `tests/test_widgets_ui_static.py` and
`tests/test_extension_surfaces.py`.
