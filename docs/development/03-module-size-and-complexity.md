# Module Size & Complexity

This chapter owns the deterministic size gates and their debt manifest, paydown by
simplifying in place, and the reading-cost invariants reviewers check against: hot-reader
projections, the source-complete decision pipeline, continuation authority, live-only
notifications, UI disposers and embedded-surface geometry. Each rule bounds reading cost
and names its enforcement or its absence.

P7 makes context fit a maintenance constraint, not an aesthetic rule. Runtime Python (`devtools/` included) and first-party `web/**/*.js` (vendored/minified excluded) target ~1000 lines. These module and function size limits exclude `tests/`; `devtools/` and the entrypoints in `FUNCTION_COUNT_EXCLUDED_FILES` also sit outside the function cap. `ouroboros/size_ratchet_manifest.py` holds Python and JS debt: 1600 lines per module (`GIANT_PATHS`), 200,000 UTF-8/LF bytes (`BYTE_DEBT`, shrink-only), 1001–1500 lines (`BAND_PATHS`; a touched 1001–1600-line module owes a reason, replaceable, never dropped) and 300 lines per non-grandfathered runtime Python function (`FUNCTION_DEBT`). Stale or newly oversized entries fail; regenerate with `scripts/regenerate_size_ratchet.py` (ARCHITECTURE §6 "Review stack"). Methods over 150 lines or eight parameters signal decomposition without gating (CHECKLISTS item 11(c); BIBLE P7); repo-wide function totals have no ceiling, so decomposition is never penalized.

Enforcement: CI's `size_ratchet` lane (ARCHITECTURE §8) checks a post-merge window's base; manual/tag runs read HEAD's parent (for a merge, only its own changes), so local qualification names the contribution base. Default local lanes skip it, readiness/health warn, and the diff pulse reports growth (tests' too) and split shape (BIBLE P3; ARCHITECTURE §6 "Commit preflight").

### Paying down a size cap

When a change hits a band, hard cap, function gate, byte debt, the size rule of `BIBLE.md`, the prompts or the Body Map, or a chapter's one-read band, pay where the change lives by removing what readers no longer need. In code, leave fewer concepts, branches and state: simplify control/data flow, delete dead code and unused re-exports (a presence-only test is not a consumer), reuse an existing SSOT. In books, `BIBLE.md` or prompts, remove stale descriptions, history, facts owned elsewhere or mechanism detail moved to module docstrings/tests; a chapter past one read splits by subject with its reasons kept. Judge rewording by clarity with every fact, reason and qualifier preserved, never by saved bytes, and never cut what a chapter's H1 introduction tells of its part (the Body Map shows it to every mind) to pay. Extraction is the last resort, valid only for a natural boundary that stays correct far below the cap (own reason to change, explicit contract, real caller, no back-import). Tighter formatting, cap-driven buckets, one-caller passthroughs, or bytes bought by dropping contract-bearing comments, docstrings, messages, tests, reasons or qualifiers are defects (BIBLE P7). When nothing honest fits, record a band reason or request `book-growth`; hard caps, function gates and byte debt admit no new entries. Enforcement: CHECKLISTS item 27 `size_cap_paydown`.

### Pragmatic SOLID

Apply SOLID for the body's health (one owner per fact, fewer moving parts):

- **SRP — Single Responsibility Principle:** one coherent reason and authority to change.
- **OCP — Open/Closed Principle:** extend stable, contract-preserving seams without rewriting unrelated callers.
- **LSP — Liskov Substitution Principle:** preserve caller-visible contract behavior across implementations.
- **ISP — Interface Segregation Principle:** depend only on needed capabilities.
- **DIP — Dependency Inversion Principle:** policy depends on small host-owned contracts, never concrete provider details.

No class hierarchy, DI container, framework surface, numeric score or extra review pass follows; diff size, line count and file count alone are never findings (CHECKLISTS item 11(d), review-only).

### Shared behavior and data-flow changes

Name the authority, fact identity/scope and affected producers/consumers (unchanged consumers whose inputs or assumptions shift included; orphaned producers when removing a reader). Preserve live/recovery semantics at comparable freshness, explaining legitimate differences. Verify falsifiable behavior at real consumer boundaries, not helper outputs or matching field names, and select paths from dependencies, not fixed inventories. Enforcement: review-only (`implicit_contracts`).

### Invariant: Projection over replay (hot readers of growing stores)

Current-state readers project transition facts instead of replaying history (ARCHITECTURE §10 invariant 10). Write new facts in the transition transaction. Before display, the census snapshots controls/phases in one short transaction per store, outside queue locks; unavailable stays unknown (`tests/test_hot_facts*.py`). Main batches question views and warms lineage by generation, keeping strict per-member gaps if a batch fails (`tests/test_main_decisions_reads.py`). Cost stays flat at rest, one axis at a time: `tests/test_cost_contract.py` pins every boundary and recurring pass as history grows, `tests/test_cost_live_set.py` as the live set grows.

- **One stored format:** only imports and history readers decode old shapes; working code reads the current one. Startup imports run once before dependent work, preserve original bytes and reopen only on restored/older writes (never process flags); older imports keep their recorded contracts. History-sized backfills are durable CAS maintenance jobs; passive control readers fall back only for their underived rows, and displays never import (unknown while owed). Unrewritable input adapts at one ingress seam. Retire an import only on owner acceptance naming the lost upgrade, never by age, and changed double readers convert, proven both ways on old data.
- **Store transactions:** lease the shared core's per-thread connection, capture rows in one aggregate step (`store_capture.py`) and end before decoding, external work or forking. Lifecycle callers quiesce users and call `store.prepare_handoff` before database renames or older-runtime handoffs (name-tier callers hold their name lock) (`tests/test_store_connections.py`, `tests/test_store.py`, WAL/FULL `tests/test_store_journal.py`).
- **Measure growth:** measure history, object count, nesting, cold caches and contention via `work` receipts (L7) and reader spies, adding no steady-state telemetry writes or history reads. Finished history grows repeating work only through what it reads, and a computation with no reader left on that path leaves it. Unknown is never empty; no universal time limit, benchmark or approval gate applies.
- **Passive GET:** display/status reads make no steady-state writes, except recall counts (PERSISTENCE) and typed provider facts already fetched, stored through their existing evidence owner; this permits no new network or control-state writes (`usage_store.py::read` may initialize an empty store without pending import; `tests/test_usage_store.py`).
- **Reuse precedents:** writer seams that also write reader-addressed facts (`dialogue_store.py::append`) or reduce current rows (`delegate_custody_current.py`), bounded tails (`jsonl_tail.py`), transactional summaries (`usage_store.py`, `store.py`), single-step captures (`store_capture.py`), SSE v2 cursors (ARCHITECTURE §3), single tree census (`tree_census.writer_facts`), reuse by source identity, never time (`memory_view.capture_story`, `skill_discovery_reuse.py`), reader-verdict/owing sets (`delegate_custody_current.held_runs`, `task_store.owing_documents`), batched results (`obligations.result_rows`).

CHECKLISTS item 9 `perf_lifecycle` governs review. Runtime `storage` and `query_data(kind="storage")` report counts and the mind judges growth (P5); new growing stores declare `storage.PLACES` and PERSISTENCE retirement. `tests/test_store_boot.py` pins `prepare_startup_state` (ARCHITECTURE §2); `tests/test_store_legacy_reconcile.py` and import suites pin legacy/downgrade behavior.

### Invariant: Source-complete decision pipeline

Every continuity surface obeys `producer → canonical full source → bounded projection → consumer → decision → retention/GC`:

- Producers persist the full event/artifact before publishing projections or wakeups; canonical sources, not caches, own identity, order, bytes and integrity.
- Bounded projections name omissions and supply an actor-resolvable source ref; `source_complete` is coverage only, never permission to infer. Over-limit tool outputs persist exact sources with no exemption; only `source_unavailable` withholds dispatch.
- `api_chat` result reviewers cannot open produced files, so content judgment stays at most `partial`.
- Consumers authorizing PASS, destructive rewrites or contract replacements materialize the named source first; an unverified claim that a host could retrieve more is not a known `partial` marker.
- Canonical results, reviews, identity decisions and summaries retain or promote referenced sources before execution-root GC; unavailable legacy sources stay explicit gaps.

Control-plane distrust is metadata, not a data-plane operation: paid model output stays evidence unless a typed validity predicate fails. Profile, route or subject mismatches, invalid output contracts and delivery failures may affect review authority; window sizing and reading diagnostics alone never do (BIBLE P3). Neither case blanks, rewrites or relabels artifacts. Enforcement: CHECKLISTS item 20 `source_completeness`.

#### Review presentation adapters

`web/modules/review_presentation.js`, `review_opinions.js`, `review_dom_patch.js`, `harness_presentation.js` and `ouroboros/review_execution_projection.py` (`executions[]`) only present producer facts: they change no verdict, lifecycle, routing, attention or enforcement and show no requested call as observed; incomplete rows stay gaps. `gateway/task_reviews.py` adds passive paged history and safe text exports over existing readers, with no second ledger, UI state or cost copy (ARCHITECTURE §3). Enforcement: `web/tests/review_presentation.test.js`.

#### Context and growth matrix

Register new hot stores in ARCHITECTURE §10 "Continuity data-flow map" and add their growth proofs here together:

| Store / surface | Complete source → interactive projection | Growth and retention proof |
|---|---|---|
| Chat and biography | `state/dialog.sqlite` (`dialogue_store.py`; one-time archive import) and `memory/chronicle/records.jsonl` → `memory_view.py` and `chat_history` | Reads and host work cost the page, never another room's history (tail: named debt); disclosed import coverage, lossless horizon folding |
| Skill-review root tasks | Hot index `state/review_ledger/index.jsonl` (`surface="skill"`) → bounded `skill_readiness._skill_names_from_review_history` (`review_history.jsonl` read-only) | Append-only, archived segments disclosed, size under App state |
| Plan/review evidence & Task/project execution | ARCHITECTURE §10.1, "Plan/review authority" & "Execution evidence" | Owned by those §10.1 rows |

### Invariant: Continuation authority and bounded role projections

Continuation is explicit, not inferred chat memory: only the Continue CAS (`continued_by`) links a successor; new tasks name no predecessor. Main receives only a defensive authority projection, never a raw head/tail slice, invented summary or canonical-result mutation (`get_task_result(include_authority=True)` for exact reads; ARCHITECTURE §1 "CLI / Headless Boundary"). Every consumer gets the idempotent brief `main_context_authority.project_helper_predecessor_authority` from `contracts.task_contract.bounded_continuation_envelope`: fields whole or as pointers within the serialized budget, previews carrying `full_chars` and `source_ref`, omitted fields named with size, `task_results` staying SSOT. Successors read the direct predecessor (no hop cap or chain walk), whose notes follow the assignment as one foldable row. Enforcement: `tests/test_continuation_context_authority.py`.

### Invariant: notifications ring for live events only

`docs/DESIGN.md` §9 owns notification policy. `notifications.js::attach()` subscribes once on `app.js`'s shared socket, never in `chat.js`, which dies on panel close. Only live frames notify; history/reconnect readers never call it, so replay is safe without persisted state. Gate out hidden, A2A and unknown rooms and child banners, and give each task ending one key (a direct ending is an ordinary reply). Resolve clicks through ARCHITECTURE §3's full registry, never capped sidebars: an unresolved target shows unavailable and keeps the view, Main notices open Main, later clicks supersede unfinished lookups, hydration stays silent. Keep classification pure (testable without DOM or socket) and preferences client-local, outside `/api/settings`. Degrade rather than disappear, never request OS permission, and never add importance host fields, text heuristics or model calls (mechanics: `web/modules/notifications.js` header). Enforcement: `web/tests/notifications.test.js`.

### Invariant: UI resources carry a disposer

Every long-lived `web/` acquisition returns or records a disposer, and each UI instance's `destroy()` releases all `ws.on(...)`, `document`/`window` listeners, observers (`ResizeObserver`, `MutationObserver`, `IntersectionObserver`), timers, `requestAnimationFrame` loops and `EventSource` streams. Closed, hidden or replaced instances release them unless retained for a visible purpose: immediate room return with reading, forms and unsent work, or a Keep-running widget. Retained rooms suspend polling/rendering but keep send custody and `destroy()`; Stop, unload, reload and shutdown force-destroy. Late async results defer hidden painting and check `destroyed` before mutating state or re-arming loops; hiding active handlers is a leak. Module widgets dispose in order with ack (`WIDGET_DISPOSE_ACK_TIMEOUT_MS`; ARCHITECTURE §3 "Extensions and Widgets"). Enforcement: release-tier `ui_browser` deterministic leak lane; commit-tier, advisory CHECKLISTS item 9 `perf_lifecycle`, so future surfaces are not deterministically closed.

### Invariant: Embedded surfaces declare geometry and refresh semantics

Every owner-visible embedded or framed surface declares host-owned geometry/overflow bounds, paired disposers, refresh/stream/error semantics and real-consumer visual verification, or documents why deferral is safe. Widgets bound framed `height` and host-cap auto-height, preserving inline-size basis and horizontal overflow; the host controls vertical scrollbar mode. Feedback-sensitive checks are event-driven on the relevant engine to a quiet fixed point, using a real consumer or production-derived fixture past its wrapping threshold, not two snapshots. Module faults use `ouro-widget-error`; builtin diagnostics are bounded best-effort logs, never execution evidence (ARCHITECTURE §3 "Extensions and Widgets"). Module source loading and declarative requests have bounded host timeouts, and job widgets keep `job_id` retry/timeout refresh contracts (malformed or missing status fails immediately; unknown in-progress stays bounded pending). Enforcement: CHECKLISTS item 9 `perf_lifecycle`, `tests/test_widgets_ui_static.py`.
