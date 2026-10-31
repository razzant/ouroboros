# Module Size & Complexity

This chapter owns the deterministic size gates and their debt manifest, paydown by
simplifying in place, and the reading-cost invariants reviewers check against: hot-reader
projections, the source-complete decision pipeline, continuation authority, live-only
notifications, UI disposers and embedded-surface geometry. Each rule bounds reading cost
and names its enforcement or its absence.

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
aggregate ceiling: a repository-wide count penalizes useful decomposition.

Enforcement: manual/tag CI runs `size_ratchet` (ARCHITECTURE §8). Without an event
base its transition check compares only HEAD's parent, not all intervening commits;
local qualification must name the contribution base. Local default lanes exclude
this marker; readiness/health warn and the diff pulse reports growth (BIBLE P3;
ARCHITECTURE §6 "Commit preflight").

### Paying down a size cap

A change that runs into a band, the hard cap, the function gate or the byte debt pays it
down by simplifying where the change lives (simpler control and data flow, dead code and
duplicates removed, an existing SSOT reused, prose made compact), so the module reads
better. Extraction is the last resort and counts only for a natural boundary that would
stay correct with the parent far below the cap: its own reason to change, an explicit
contract, a real caller. A cap-driven bucket, a one-caller passthrough, or bytes bought by
deleting contract-bearing comments, docstrings, messages or tests is a defect to report,
not paydown (BIBLE P7 «first simplify what exists»). Enforcement: CHECKLISTS item 27
`size_cap_paydown`, advisory when applicable.

### Pragmatic SOLID

SOLID serves the body's health, not minimalism for its own sake: fewer moving parts and
one owner per fact leave fewer places for bugs and a body Ouroboros can still
understand. It is a direction, not a demand for classes or framework surface:

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
follows. Diff size, line count and file count alone are not findings; CHECKLISTS item
11(d) says what a finding names (review-only).

### Shared behavior and data-flow changes

For shared behavior/data changes, name the authority, fact identity/scope and affected
producers/consumers, including unchanged consumers whose inputs or assumptions change.
Preserve live/recovery semantics at comparable freshness; explain legitimate differences.
Verify falsifiable behavior at real consumer boundaries, not helper outputs or matching
field names; select paths from dependencies, not a fixed inventory. Enforcement:
review-only, coupling `implicit_contracts`.

### Invariant: Projection over replay (hot readers of growing stores)

Apply ARCHITECTURE §10 invariant 10 to current-state readers. A new fact belongs in its
transition's transaction; legacy derivation is a named durable job under sequence/source CAS, a
process-local flag never proves an installation migration, and passive control readers fall
back only for underived rows. The census snapshots controls/phases in one short transaction per
addressed store, outside queue locks, released before display consumers; unavailable stays
unknown for that request (`tests/test_hot_facts*.py` fix Stop/custody results while settled
history grows 20→2,000 rows). Main batches question display views and warms lineage by
generation, keeping strict per-member gaps if a batch fails (`tests/test_main_decisions_reads.py`);
`tests/test_cost_contract.py` fixes every boundary and recurring pass as history grows.

- **Measure growth:** history, object count, nesting, cold caches and contention, using
  existing spans and real-reader spies when responsiveness suffers. Telemetry adds no
  steady-state writes/history reads. Remove repetition or reuse a validated view before
  adding a projection/cache. Unknown is never empty; no universal time limit, benchmark or approval gate.
- **Passive GET:** rendering/stored display adds no steady-state writes. Already-performed
  external discovery may store typed provider facts through their existing evidence owner;
  this permits no new network or control-state writes. `usage_store.py::read` may initialize
  an empty store without pending import; otherwise display reports unavailable data and
  leaves import to startup (`tests/test_usage_store.py`).
- **Reuse precedents:** a writer seam whose transaction also writes the facts readers
  address (`dialogue_store.py::append`) or reduces current rows (`delegate_custody_current.py`);
  bounded filtered tail (`jsonl_tail.py`); transactional summaries (`usage_store.py`, `store.py`);
  one-step indexed selects (`store_capture.py`); task-event SSE v2 cursor (ARCHITECTURE §3).

CHECKLISTS item 9 `perf_lifecycle` covers changed readers, startup/shutdown, batches, endpoints, pollers, subscriptions and timers. Runtime `storage` supplies the last count; `query_data(kind="storage")` its detail. The mind judges growth (P5). New growing stores name their `storage.PLACES` entry and PERSISTENCE retirer.

`tests/test_store_boot.py` fixes `prepare_startup_state` at 50/500 frozen results and 20/200
child stores (a clean boot is one read transaction, no history access or writes);
`tests/test_store_legacy_reconcile.py` covers both detector outcomes, cursor resume and the
foreground CLI; the import/switch suites keep downgrade semantics.

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
- An `api_chat` result reviewer cannot open the produced files; a judgement on their
  content stays at most `partial`.
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

Enforcement: CHECKLISTS item 20 `source_completeness` (critical when applicable).

#### Review presentation adapters

`web/modules/review_presentation.js`, `review_opinions.js`, `review_dom_patch.js`,
`harness_presentation.js` and `ouroboros/review_execution_projection.py` (bounded
`executions[]`) only present producer facts: no verdict, lifecycle, routing,
attention or enforcement changes, no requested-as-observed execution. Incomplete
rows remain gaps. Existing chat/task/attempt readers supply the view;
`gateway/task_reviews.py` adds passive paged history and owned saved-text exports
through ledger/Plan/skill readers. No second ledger, persisted UI state or cost
copy (ARCHITECTURE §3). Enforcement: `web/tests/review_presentation.test.js`,
`web/tests/harness_presentation.test.js`, `tests/test_task_review_sources.py`.

#### Context and growth matrix

ARCHITECTURE §10 "Continuity data-flow map" owns sources, projections and retention for chat/biography, plan/review and execution. Add new hot stores there and their growth proof here together.

| Store / surface | Complete source → interactive projection | Growth and retention proof |
|---|---|---|
| Chat and biography | `state/dialog.sqlite` (`dialogue_store.py`; the frozen archive imported once) and `memory/chronicle/records.jsonl` → the memory view (`memory_view.py`) and `chat_history` | Reads cost the page, never another room's history; import coverage is disclosed; pages and parts compress the horizon, never delete it; size appears in the storage count |
| Plan/review evidence | ARCHITECTURE §10.1, "Plan/review authority" | Source and retention rules live in that row |
| Skill-review root tasks | The review ledger's hot index (`state/review_ledger/index.jsonl`; a `surface="skill"` row names its skill) → `skill_readiness._skill_names_from_review_history`, bounded; a legacy `review_history.jsonl` is only read | Append-only; archived segments are disclosed as truncation; size under App state |
| Task/project execution | ARCHITECTURE §10.1, "Execution evidence" | Promotion and retention rules live in that row |

### Invariant: Continuation authority and bounded role projections

Continuation is an explicit relation, not inferred chat memory: only the Continue CAS
(`continued_by`) links a successor, and new work names no predecessor. Main gets only a
defensive projection of a predecessor's authority, never a raw head/tail slice, an invented
summary or a canonical-result mutation (ARCHITECTURE §1 "CLI / Headless Boundary"); the
optional exact read is `get_task_result(include_authority=True)`.

Every predecessor consumer (the successor's start, children, external work orders) gets
the same pure, idempotent brief
`main_context_authority.project_helper_predecessor_authority` of the one envelope producer
`contracts.task_contract.bounded_continuation_envelope` (fields whole-or-pointer under the
serialized budget; previews carry `full_chars` and a `source_ref`). Durable `task_results`
bodies stay the SSOT; the brief names each omitted field and size and never stands in for
it. No hop cap, no chain walk: a successor reads its predecessor through the exact read
above, and the predecessor mind's notes follow the assignment as one foldable row.
Enforcement: `tests/test_continuation_context_authority.py`,
`tests/test_helper_predecessor_history.py`.

### Invariant: notifications ring for live events only

`docs/DESIGN.md` §9 owns notification policy. `notifications.js::attach()` subscribes
once to the shared socket in `app.js`; `chat.js` holds no notification code, since a
closed Project disposes its handlers. Only live frames reach the notifier; that
boundary, not a persisted ledger, makes replay safe. History and reconnect readers
never call it; no notification state survives reload.

The room gate refuses hidden, A2A and unknown chats. Each task ending has one key;
a direct ending is an ordinary reply. Frame lineage excludes child banners.
Activation resolves exact Project identity through `GET /api/projects?chat_id=…`,
independent of the capped sidebar. Missing, inactive or unreadable destinations show
unavailable without navigating. Established Main notices, including Telegram, open Main.
A later click supersedes an unfinished lookup; hydration stays silent.

Classification and gating are pure over a frame and stored preferences, testable
without DOM or socket. Preferences stay client-local, outside `s-`, settings-dirty
tracking and `/api/settings`. Delivery degrades rather than disappearing and never
requests OS permission. Importance adds no host field, text heuristic or model call
(mechanism: `web/modules/notifications.js` header). Enforcement:
`tests/test_notifications_static.py`, `tests/test_notification_destination.py`,
`web/tests/notifications.test.js`.

### Invariant: UI resources carry a disposer

Every long-lived acquisition in `web/` returns or records a disposer, and a UI instance
owns a `destroy()` that releases everything it acquired: WS subscriptions
(`ws.on(...)`), `document`/`window` listeners, observers (`ResizeObserver`,
`MutationObserver`, `IntersectionObserver`), timers, `requestAnimationFrame` loops and
`EventSource`/streaming connections.

An instance that can be closed, hidden or replaced (canonically, project chat
panels) is destroyable without leaving an acquisition behind. It may survive
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

For Widgets, framed `height` values are bounded and module auto-height is host-controlled
below a finite ceiling, where updates preserve the child's inline-size basis; the host
controls vertical scrollbar mode while preserving horizontal overflow. Feedback-sensitive
verification is event-driven on the relevant engine and proves convergence to a quiet
fixed point with a real consumer or production-derived fixture that crosses the known
wrapping threshold, not two snapshots. A module widget's own faults use
`ouro-widget-error`; the builtin page sends bounded, best-effort reports to the existing
event log (ARCHITECTURE §3 "Skills and Widgets"). These page-reported diagnostics are not
task-execution evidence. Module source loading and declarative requests have a bounded
host timeout; declarative job widgets keep their `job_id` and bounded retry/timeout
behavior in the refresh contract, where missing or malformed job status is an immediate
protocol error and an unknown in-progress label is a bounded pending state.

Enforcement: CHECKLISTS item 9 `perf_lifecycle` points lifecycle changes here; the
widget geometry/refresh contracts are pinned in `tests/test_widgets_ui_static.py` and
`tests/test_extension_surfaces.py`.
