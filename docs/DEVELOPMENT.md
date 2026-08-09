# DEVELOPMENT.md — Development Principles & Module Guide

## Role and authority

This is Ouroboros's engineering handbook: how to name, design, implement, and
verify changes. `BIBLE.md` owns constitutional principles;
`docs/ARCHITECTURE.md` owns the current structure, data flow, and rationale map;
`docs/CHECKLISTS.md` owns reviewer items, severity, and output contracts. This
file does not duplicate their inventories or serve as a changelog.

Rules here describe current practice or a deliberately enforced standard. When
code and prose disagree, inspect the implementation and history, repair the
authoritative surfaces together, and retain the failure a non-obvious rule
prevents.

---

## Naming and boundaries

- Code identifiers, comments, docstrings, and commit messages are English.
- Follow PEP 8: modules and variables use `snake_case`, classes use
  `PascalCase`, and constants use `UPPER_SNAKE_CASE`.
- Name the observable responsibility and authority, not the implementation
  fashion. Prefer a clear function module over a class with no lifecycle.

The repository has two deliberately different gateway directions:

- `ouroboros/gateway/` is the inbound browser/CLI HTTP and WebSocket facade.
  `contracts.py` owns typed shapes, `router.py` owns route collection, and
  domain handlers translate transport into calls on existing runtime owners.
- `ouroboros/gateways/` contains thin outbound adapters to external runtimes.
  They translate calls and errors but do not choose product policy, fallback,
  custody, or authorization.

Contracts are typed shapes, not service objects. A manager is justified when it
owns lifecycle or mutable state. LLM-callable Tools remain thin `{verb}_{noun}`
functions that validate their public input, call the owning subsystem, and
format a result. No universal `{Domain}Service`, `{Platform}Gateway`, or class
layer is required.

Dependency direction is the test: UI/CLI → inbound gateway → domain owner;
runtime policy → small host-owned contract → outbound adapter. Provider- or
transport-specific decisions do not flow back into core policy.

### CLI and headless work

- CLI commands parse, call the existing gateway/scheduler, and render
  text or typed JSON/JSONL/SSE. They do not create a second task state machine.
- External workspace tasks keep governance bound to the system repository while
  contextual tools resolve through `ToolContext.active_repo_dir()`. Admission
  rejects overlap with the system repo/data and records a read-only preflight.
- Workspace execution returns durable artifacts and patch diagnostics; it does
  not grant commit, restart, runtime-control, or review-state authority over
  Ouroboros. `executor_ref` selects a process backend, not an implicit sandbox.
- Project-local installs may run within the workspace policy. Global/system
  installs remain safety-reviewed, and `sudo` is non-interactive (`sudo -n`).
- Do not add a second scheduler for operator tooling or a generic CLI file
  manager. Use the task queue, attachments, logs, and artifact endpoints.

### Cognitive quality

Do not lower model quality, reasoning effort, output budget, or context breadth
for consciousness, review, or self-evolution as an incidental latency/cost
optimization. An intentional narrowing is an owner decision reflected in the
plan, docs, tests, and evidence.

### LLM-first affordances

Do not repair a semantic tool-choice failure by adding one more keyword hint to
`prompts/SYSTEM.md`. Put stable discoverability in the tool schema or add a
typed affordance at the point of need. SYSTEM accretion trains around one
incident, bloats the resident prefix, and forks the authority.

### Pricing and admission

Never add hand-maintained model-price tables, inherited prefix tariffs, or
numeric fallback prices. Query the exact route when a live source exists,
prefer provider-settled usage, and otherwise preserve `cost=None` and
`cost_final=false`. Unknown price is neither free nor a model-admission veto;
known exhausted budget remains enforceable.

### Anti-pattern: content-derived identity for host-minted records (v6.73.0)

If the host itself created a record — a chat message, a task, a binding — its
identity is CAPTURED AT INGRESS and passed downstream BY VALUE as a typed
reference (e.g. `origin_message_ref = {chat_id, client_message_id, ts,
text_sha256}` built where `log_chat("in", …)` writes the canonical row). Do NOT
re-derive that identity later by searching logs/state for a row whose text
hash/equality/prefix matches whatever text the caller happens to hold: in an
LLM-first system the text is routinely rewritten between ingress and use, so
content-derived lookup fails exactly on the normal path (the four
start-message-loss incidents fixed serially before v6.73.0 were all this class).
Content hashes remain legitimate in two roles only: (a) an INTEGRITY CHECK on an
already-known identity (`text_sha256` inside a ref verifies the row wasn't
swapped — it is never the lookup key), and (b) content-ADDRESSING where the
content IS the identity (artifact stores, observability blobs, join-ledger
result hashes, staged-diff review bindings). One NAMED exception inside role (b),
v6.78.0 / owner Q28=B: a verification RECEIPT is reconciled by ONE TYPED
IDENTITY KEY — its `criterion_id` when it has one, else its canonical `check` text, else
(for the artifact-observation class, which runs no command) its observed `paths` SET —
because with no criterion id the check text / observed path set IS that verification's
identity (there is no earlier ingress point to capture, and a class whose identity is
dropped can never reconcile itself: a red "report.md missing" must be clearable by the
byte-identical green observation once the file exists). Two receipts name the same
verification when the key's KIND and VALUE both match, never across kinds. The key
replaced a per-component FALLBACK CHAIN, and the reason is worth stating as a rule: a
chain is not an equivalence relation. It was not transitive — `{c1, check}` matched
`{check}`, `{check}` matched `{c2, check}`, while `c1` and `c2` are explicitly different
criteria — so one check-only green reconciled two distinct reds, "collapse the candidates
onto the identity they name" was not even well defined, and the outstanding set came out
order-dependent. No care at the call sites can repair a relation that is not an
equivalence; keying makes sameness the KERNEL of a function (reflexive, symmetric,
transitive by construction) and makes an existing `criterion_id` authoritative
STRUCTURALLY rather than by a rule someone must remember. It is a SIMPLIFICATION —
strictly fewer branches — and it fails in the SAFE direction: strictly fewer
reconciliations, so a red the chain used to clear may now stay open. Concretely, a re-run
that OMITS the `criterion_id` it carried before no longer clears its own red; the cost is
one advisory nudge and one advisory reviewer flag, never a false green. Read that as the
design, not a regression. If omission tolerance is ever genuinely needed, the only sound
route is to carry the id forward STRUCTURALLY at receipt ingress (`tools/verify.py`) —
never to infer it back from shared command text under another name. Two paths deliberately
keep the older any-later-grounding rule, because there the key cannot do its job: a
receipt with NO key at all (nothing to protect — a malformed `artifact_observation` with
no paths would otherwise mint an unclearable red), and the MASKED-pass path, where the
only text identity is the masked command itself and the prescribed remediation ("drop the
masking pipe") necessarily changes it, so a byte-identical clean re-run cannot exist.
On that masked path the `criterion_id` key alone binds, by the same equality: a masked
receipt that NAMES a criterion is cleared only by a later clean receipt naming that SAME
criterion — one that merely omits its id does NOT clear it — and the any-later-clean
fallback reaches only a masked receipt that names no criterion. Command text never
participates there, in either direction. **And whatever decides must be what is
reported.** Both relations and both disclosures read ONE mode-aware projection —
`receipt_reconciliation_key(receipt, masked=…)`, with the mode read off the receipt by
`receipt_is_masked_pass` for the per-row question (`receipt_disclosed_reconciliation_key`)
— so `reconciliation_identity` and `expected_whitespace_normalized` name the authority
that actually cleared the receipt instead of re-deriving one beside it. Re-deriving is how
round 6 arrived: an id-less masked pass, reconciled by ANY later clean grounding, was
disclosed to the acceptance reviewer as `check`-governed with
`expected_whitespace_normalized=true`, while `_reconciles_masked` never looked at check
text at all. That flag is now FALSE across the whole masked path. A host-attested artifact
that misstates its own basis is worse than one that says nothing: the reviewer cannot
discount evidence whose provenance it has been told wrongly, and "the reporting path reads
the deciding path" is the general fix — the same move that collapsed the projection/
comparison split in round 4 and the fallback chain in round 5.

Round 7 was that same class one kind over: `expected_whitespace_normalized` also read true
for `artifact_paths`, whose observed set is compared BYTE-FOR-BYTE (every whitespace byte of
a filename counts), so the fix for one kind had simply never been asked of the others. The
durable form is to answer the question ONCE PER KIND, next to the kinds: `IDENTITY_KINDS` is
the closed, ordered table of reconciliation identity kinds, each row carrying its name, how
to read its value off a `ReceiptIdentity`, and whether that identity is canonical command
text; `ReceiptIdentity.key` iterates the table and `KIND_NORMALIZES_COMMAND_TEXT` is the
total lookup the flag performs — true for `check`, false for `criterion_id`,
`artifact_paths` and `none`, with a `KeyError` rather than a default for a kind that skipped
the table. The general rule: when a disclosure describes a property of a closed set of
kinds, put the property IN the set, so a new kind cannot be added without answering, and a
per-kind fix cannot be mistaken for a fix of the class. Every component that
participates carries into BOTH fixed reviewer projections
(`verification_receipt_ledger_row`, `_accept_verification_summary`), and it does so
through ONE shared renderer — `_outcome_receipts.receipt_identity_projection` — rather
than two independently maintained key lists, so the two surfaces cannot drift apart
about the identity the reconciliation used. Three rules make that claim literal rather
than aspirational. First, a projection that BOUNDS a list discloses the bounding, and it
does so through ONE shared helper — `_outcome_receipts.disclosed_list_projection` — not a
hand-rolled `[:N]` per call site: every bounded list on a cognitive-review surface emits
`<key>_omitted` (the exact count, 0 included), bounds each string through the SSOT
`utils.truncate_review_artifact` so a clipped value carries its own omission note, and
adds a hash of the FULL set (`paths_identity_sha256` for the path identity `_reconciles`
compares; `urls_identity_sha256` for the native-retrieval URL set) wherever the complete
evidence is not reachable from the store the bounded row lives in — the receipt lists need
no hash because the whole receipt is durable in the per-task `verification_receipts.jsonl`.
Bounding a set is allowed; hiding that you bounded it is the P1 violation, and a review
round that finds one surviving `[:N]` on such a surface should sweep the phase for the
rest rather than patch the instance. Second, "is anything still outstanding?" is a
question about a SET of identities, so it is answered by an outstanding SET
(`unreconciled_failed` / `unreconciled_masked`, each candidate scanned against ALL later
reconcilers) and never by a single latest-POINTER, which a newer candidate silently
overwrites: fail A, fail B, pass B once reported no red at all, and masked c1, masked c2,
clean c2 lost c1 the same way. The `latest_*` helpers are projections of that set, and the
acceptance summary carries its SIZE (`unreconciled_red_count`,
`check_exit_masking_unreconciled_count`) so a second outstanding item cannot hide behind a
flag that reads as if it described exactly one. Third, the acceptance summary projects the
identity of the UNRECONCILED RED (`unreconciled_red_identity`) and not only of the latest
receipt: a later green of a DIFFERENT verification leaves an earlier red standing, so a
reviewer shown `unreconciled_red=true` beside a green `latest_*` would see a flag whose
cause is nowhere in the packet. A flag without its cause is not reconstructible. Fourth —
the rule the first three kept re-learning instance by instance — there is exactly ONE
canonical identity derivation (`_outcome_receipts.receipt_canonical_identity`, built on
the shared `shell_parse.canonical_command_text` seam and `canonical_path_set`), and
comparison, hashing, counting and projection all read THAT object. A phase that carries a
normalized/set-shaped identity for comparison and a raw/ordered one for display will keep
producing findings where the two disagree: a lossy comparison form (`" ".join(x.split())`
collapses whitespace inside quoted arguments, so two checks asserting different things
compare equal and a green closes an unrelated red), rows counted where identities were
promised, and a hash describing a set the carried items do not. The derivation implies an
ORDER, and the order is always canonicalize the RAW values → render → bound: rendering
is lossy (redaction, truncation), so de-duplicating after it drops distinct values while
the omitted count still reports zero.

Two consequences of that fourth rule are worth writing down rather than rediscovering.
**The receipt store's `check` rendering changed in v6.78.0**: `tools/verify.py` now writes
`shlex.join(argv)` where it wrote `" ".join(argv)`. A space-join is not injective — argv
`["echo","a b"]` and `["echo","a","b"]` rendered to the same text, and since that text IS
the verification's identity downstream, a green on one could clear a red on the other.
`shlex.join` is the exact inverse of the lexer `canonical_command_text` re-tokenizes with,
so the stored text round-trips back to the argv that ran — which only holds because that
lexer now preserves QUOTING: `shlex` strips quotes before yielding tokens, so `echo '&&' x`
and `echo && x` arrived identical, and the canonicalizer's leading/trailing separator strip
then dropped a literal final `&&` argument as if it were syntax. Quoted and escaped
punctuation is marked on the way into the lexer and the mark is read back off the tokens,
so a literal argument that merely spells like an operator can never be mistaken for one;
nothing is stripped afterwards. Path values are likewise left byte-for-byte alone — a
leading or trailing space is a legal filename byte, and trimming it let observations of two
different files reconcile. The rule behind all three: a normalization that discards
information the identity depends on is not a normalization.

**Changing a stored rendering means versioning it.** The cross-version cost of that switch
was first written down here as safe in ONE direction — an old receipt carries `echo a b`
where a new one for the same argv carries `echo 'a b'`, so they fail to reconcile, and a
non-reconciling red STAYS RED. True, and not the whole picture: an old red and a new green
from DIFFERENT argvs can render IDENTICALLY (`["echo","a b"]` and `["echo","a","b"]` were
both space-joined to `echo a b`), and there the new green cleared the old red — a false
green produced by the change made to remove one. Reasoning about a format migration one
direction at a time is how that got missed; ask both, always. The root was that the receipt
did not record WHICH renderer produced its text, so the comparator could not tell the two
formats apart. It records it now: `check_rendering` is stamped beside the text by every
writer that renders one (`shlex_join` for a rendered argv, `declared_text` for the agent's
own verbatim text), an absent stamp reads `unversioned`, and the check identity is the
RENDERING PAIRED WITH THE TEXT. Receipts from different renderings are therefore never the
same verification — an `unversioned` receipt is not known-equal to a versioned one, it is
UNKNOWN, and unknown must not clear a red. Two `unversioned` receipts still match each
other, which is both the behaviour they had before the upgrade and the most that can be
recovered from strings already on disk. An unrecognised future rendering is automatically
its own namespace, so the next renderer change is safe without a code change — but it must
still take a new stamp value, and a writer that stores a `check` without one is a bug (a
test walks `verify.py`'s receipt writers and asserts every check-writing one stamps).
The direction is now honestly stated: cross-version reconciliation is strictly LESS likely,
so an upgrade may leave standing a red that was really fixed. A false red costs a human a
second look; a false green costs the thing this whole surface exists for. The other two
identities were asked the same question and neither has the hole: the MASKED path keys on
the agent-authored `criterion_id`, which no writer ever re-rendered, and the observed
`paths` set is stored RAW and canonicalized by the READER, so both eras are compared on
today's terms — the phase's path change (dropping `.strip()`) was a comparator change, not
a stored-format one. **And one KNOWN, deliberately
deferred limit:** `tools/verify.py` bounds an `artifact_observation` receipt's observed
path set at twenty (`paths[:20]`) with no omission count, so two observations differing
only past the twentieth path are indistinguishable IN THE STORE. Unlike every bound above
it, this one is on what gets written DURABLY, not on what gets projected — there is no
complete set behind it for a downstream projection to recover, and `paths_identity_sha256`
can only ever hash what the writer wrote. Fixing it means changing what the durable store
holds and deserves its own scope. It is recorded here so it is a known limit rather than a
silent one. It is ADVISORY only — it shapes a nudge and a
disclosed reviewer flag (`expected_whitespace_normalized`), never a gate — so a
mismatch costs at most one advisory nudge and can never lose a record. For semantic matching of fuzzy
entities, use the LLM-first pattern (`semantic_dedup`: exact fingerprint as a
cheap first pass, an LLM as the authority, fail-open) — never string equality.
The enforcement shape is a REQUIRED typed argument at the consuming seam
(`bind_task_to_project(..., *, origin)`: a valid ref or a closed-enum absence
reason; omission raises), so a future call site cannot silently skip the
invariant.

### Mutable external-fact inventory

This table is a maintenance inventory, not a second runtime authority. External
facts change independently of Ouroboros releases; prefer live metadata or a
bounded probe where that can answer the exact question, and otherwise keep the
current conservative behavior visible. v6.67.0 documents these facts but does
not migrate their runtime representations.

| Location | Fact | Mutability | Current authority | Live/probe option | Risk | Recommendation |
|----------|------|------------|-------------------|-------------------|------|----------------|
| `ouroboros/provider_models.py::_VISION_MODEL_PREFIXES` / `_VISION_OVERLAY` | Which model families accept native image input | High as model families and route capabilities change | Conservative shipped prefixes, overridden by parsed OpenRouter `/models` `architecture.input_modalities` for exact model ids | Exact provider metadata when available; otherwise a bounded image-input capability probe | A stale positive sends unsupported image blocks; a stale negative needlessly captions them | Keep the conservative fallback and exact-model overlay; consider broader provider metadata only in a separately reviewed migration |
| `ouroboros/llm.py::supports_message_cache_control` and `_reasoning_signature_portable_across_or_providers` | Which families support message cache controls and portable replayed reasoning | Medium/high as provider routing contracts change | Explicit family rules backed by provider behavior and dated live probes | Provider documentation plus a same-model cross-provider replay probe | A false positive can invalidate a request; a false negative loses cache or reasoning continuity | Retain the small explicit rules and re-probe when provider behavior changes; do not generalize by model-name resemblance |
| `ouroboros/provider_models.py::_ANTHROPIC_MODEL_ALIASES` / `migrate_model_value` | Direct-provider id spelling compatibility | Medium as providers rename ids and prefixes | Shipped compatibility mapping and current direct-provider id contract | Exact provider catalog/documentation can confirm a current id, but cannot establish whether a saved spelling was intentional | Removing an alias breaks upgrades; guessing aliases can silently reroute | Keep explicit compatibility aliases until a separately documented retirement window closes |
| `ouroboros/server_runtime.py::_RETIRED_MODEL_DEFAULT_REPLACEMENTS` and scope prior/legacy defaults | Which formerly shipped defaults are upgraded automatically | Release-dependent | Release history plus current `SETTINGS_DEFAULTS`; only known former defaults are migrated | A live catalog can show availability, but cannot infer user intent or whether a saved value was a default | Over-broad migration overwrites an explicit owner choice | Keep release-scoped exact replacements and regression tests; review retirement separately |
| `ouroboros/pricing.py::get_pricing` and `ouroboros/llm.py::fetch_openrouter_pricing` / `fetch_cloudru_pricing` | Exact-route model tariffs | High; pricing and FX drift independently | Exact provider catalog with nullable unknowns; provider-settled usage wins | Bounded live catalog fetch and provider-reported settled cost | Static prices look authoritative after becoming wrong and can corrupt admission | Preserve the live nullable design and cover it by regression; do not restore runtime tariff tables |

### Provider Independence

One configured provider must be sufficient for the agent loop, commit review,
scope policy, safety, and context/memory flows. Core capability must not acquire
a hidden OpenRouter or second-provider dependency.

When adding or changing a provider, update one coherent route contract:

1. credential/readiness detection and exact model-id migration;
2. Main/Light/Fallback and reviewer-slot defaults without overwriting explicit
   owner choices;
3. tool, reasoning, image, and cache-control translation at `llm.py`;
4. nullable pricing/settlement and truthful capability omissions;
5. review and scope routing, including sourced context-window evidence;
6. direct-provider and single-provider regression tests.

Local-only installs keep their local route. Unreachable shipped remote defaults
may be cleared, but explicit owner values are not. Scope authority follows the
BIBLE P3 policy: in owner-selected Max it requires the applicable sourced window
evidence; owner-selected Low records the declared skip rather than pretending a
partial review occurred. Current model ids and defaults belong in code/config,
not in this handbook.

The `-pro` suffix is an OpenRouter routing slug, not an official OpenAI model id.
Until a Responses-API lane exists, a direct OpenAI slot uses the plain Sol id;
projecting the slug into chat completions would turn an owner route choice into
a guaranteed 404. This is a compatibility constraint, not a mutable capability
table.

Provider-specific optional features may be unavailable on another single
provider, but the core loop must degrade explicitly rather than crash or silently
reroute.

## Module Size & Complexity

P7 makes context fit a maintenance constraint, not a line-count aesthetic.

- Python and first-party `web/**/*.js` modules target roughly 1000 lines. The
  deterministic hard gate is 1600 lines for paths not listed in
  `ouroboros/review.py::GRANDFATHERED_OVERSIZED_MODULES`; that code-owned set is
  the debt SSOT. Vendored/minified assets and `web/tests/` are excluded.
- Every non-grandfathered Python function or method fails the deterministic gate
  above 300 lines; exceptions live in
  `ouroboros/review.py::GRANDFATHERED_OVERSIZED_FUNCTIONS`. Methods above 150
  lines are a decomposition signal. JavaScript currently has only the module
  line-count gate.
- Runtime Python function/method count is checked against
  `ouroboros/review.py::MAX_TOTAL_FUNCTIONS`; tracked `devtools/` is outside
  that runtime-health count but remains reviewable when touched.
- More than eight parameters is a decomposition signal applied by BIBLE and
  reviewer checklist 2(c), not a deterministic size-test gate. Existing
  baseline debt is not retroactively a failing tree. Any advisory ratchet must
  publish its AST counting scope and bind its baseline to the final SHA.
- Prefer deleting dead/duplicate authority before raising a cap. Add an
  abstraction only when it removes concrete coupling or preserves a stable
  extension seam.

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

### Invariant: Projection over replay (hot readers of growing stores)

A reader that runs per INTERACTION — an HTTP request, a WS/SSE message, a poll
tick, a task turn — must not replay a growing store to produce its answer.
Interactive read cost must be O(response), achieved through a maintained
projection, a cursor, rotation, or a bounded tail — never a full-history scan
filtered down to the answer.

- **Per interaction is the unit.** Work that runs once per boot or per explicit
  owner action may scan history; work on a request/message/poll-tick/task-turn
  path may not. A scan that is cheap today is not the point — every growing
  store crosses the threshold eventually, and the reader degrades exactly when
  the system is most used.
- **Storage-agnostic.** JSONL logs, SQLite tables, JSON snapshots — any store:
  a full-table read filtered in code IS a replay (a `SELECT *` narrowed in
  Python is the same failure as parsing a whole JSONL file for its tail). This
  includes unbounded collections INSIDE snapshot/state files: a "snapshot" that
  accretes an unbounded list re-reads its entire history on every load.
- **Passive GET.** Read handlers perform no NEW steady-state durable writes.
  Exactly two named exceptions exist: (1) substrate-owned integrity repair
  performed under the substrate's own lock (the usage-ledger torn-tail
  quarantine in `ouroboros/usage_ledger.py`), and (2) one-time idempotent
  migrations guarded by a durable watermark (the legacy usage import). Anything
  else that "just materializes a bit of state" on a GET is a mutation hiding on
  a read path.
- **House precedents — reuse these shapes instead of inventing new ones:**
  chat log rotation with archive-aware readers
  (`supervisor/state.py::rotate_chat_log_if_needed`); the compact
  `containment_faults.jsonl` projection maintained beside an unbounded event
  log (`ouroboros/delegate_custody.py`); dialogue-block consolidation — the P1
  "infinite horizon, variable granularity" reader; the passive-GET contract
  of `gateway/control.py::api_update_status`; and the fingerprint-keyed render
  cache inside the usage-ledger rows memo (implemented in
  `ouroboros/_usage_rows_memo.py`, consumed through `usage_accounting`) —
  a projection cached while its input is unchanged, invalidated only by
  advance/refold, never by TTL.

Enforcement: Repo Commit Checklist item 24 (advisory) triggers on diffs that
add or change an endpoint/poller/subscription/timer or read a growing store;
the hot-store growth health invariant
(`agent_startup_checks.py::hot_store_growth_notes`, surfaced by
`context.py::build_health_invariants`, thresholds justified in
`ouroboros/context_budget.py`) is the deterministic runtime tripwire. A change
that introduces a new append-only store read on an interactive path must
enroll that store in the `ouroboros/context_budget.py` threshold table (with a
justified constant) in the same commit — an unenrolled hot store is invisible
to the tripwire.

### Invariant: UI resources carry a disposer

Every long-lived acquisition in `web/` returns or records a disposer, and a UI
instance owns a `destroy()` that releases everything the instance acquired.
The resource kinds this covers:

- WS subscriptions (`ws.on(...)`)
- `document`/`window` event listeners
- observers (`ResizeObserver`, `MutationObserver`, `IntersectionObserver`)
- timers (`setInterval`, long-lived `setTimeout` chains)
- `requestAnimationFrame` loops
- `EventSource` / streaming connections

An instance that can be closed, hidden, or replaced (project chat panels are
the canonical case) must be destroyable without leaving any acquisition
behind; "hide the DOM node, keep the handlers" is the leak shape this
invariant forbids. Late async continuations check a `destroyed` flag before
touching state or re-arming loops.

Enforcement (honest disclosure): the deterministic leak test runs in the
release-tier `ui_browser` lane, not at commit tier; commit-tier coverage is
the advisory Repo Commit Checklist item 24. The class is closed
deterministically for the instrumented surfaces and advisorily for future
ones.

---

## Core Governance Artifacts

`BIBLE.md`, `docs/ARCHITECTURE.md`, and `docs/DEVELOPMENT.md` are **core governance artifacts**.
They are the constitutional, architectural, and procedural ground truth of the system.

### Invariant: Full availability in reasoning flows

Any flow that requires architectural, constitutional, or procedural reasoning MUST include
these artifacts as **first-class context sections** — not as optional or opportunistic
inclusions via touched-file packs.

Concrete requirements:

| Flow | BIBLE.md | ARCHITECTURE.md | DEVELOPMENT.md |
|------|----------|-----------------|----------------|
| Main task context (`context.py`) | full tier-0 | full in Max for every task class; lossless navigation map in Low | mode-independent: full when the active binding targets Ouroboros's system repo, including evolution/self-body work and a project-room turn without an external binding; visible on-demand pointer for a bound external workspace, subagent, or API/CLI/scheduled external surface. `workspace="none"` and explicit self-body overrides retain full Development. |
| Triad review (`tools/review.py`) | ✅ via preamble | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` |
| ↳ Anti-thrashing (v4.35.1) | — | — | Open obligations loaded from `review_state` via `load_state(drive_root)` + `make_repo_key(repo_dir)`, injected unconditionally into `_build_review_history_section` prompt context. Same mechanism in `scope_review.py::_build_scope_prompt` (best-effort when `drive_root` available). |
| Background consciousness (`consciousness.py`) | ✅ full | ✅ full (max) / navigation map (low) | — (not yet required) |
| Advisory pre-review (`tools/claude_advisory_review.py`) | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` |
| Scope review (`tools/scope_review.py`) | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting |
| Plan review (`tools/plan_review.py`) | ✅ full (every plan class) | full for `plan_class=self_mod`; lossless **navigation map** (sections + line ranges, full sections on demand) for external/creative/research plans (v6.61.0, owner-approved governance change) | ✅ full (every plan class) |
| Deep self-review (`deep_self_review.py`) | full canonical doc + Atlas accounting | full (max) / navigation map (low) + Atlas accounting | full canonical doc + Atlas accounting |

Plan review always keeps BIBLE.md, DEVELOPMENT.md, the proposed plan,
touched-file snapshots, and reviewer-slot framing as first-class context.
ARCHITECTURE.md is CLASS-TIERED (v6.61.0, an owner-approved governance
evolution — quiz 19): the agent declares `plan_class`
(`self_mod | external | creative | research`), and the host STRUCTURALLY
escalates to `self_mod` whenever `files_to_touch` resolve under the system repo
(a path fact, never keyword matching). `self_mod` plans keep the full inline
ARCHITECTURE.md — unchanged from the historical contract. Non-self_mod plans
(an external codebase, a creative deliverable, a research question) receive
ARCHITECTURE.md as the LOSSLESS navigation map (`context_layout.
generate_doc_nav_map`: every section + line range, full sections readable on
demand) — their reviewers judge the plan against its own domain, not ~45K
tokens of self-body detail. Rationale: the full-pack requirement existed to
protect SELF-modification reasoning; for non-self plans it actively hurt
review quality (reviewers anchored on runtime internals irrelevant to the
deliverable) while tripling cost. The agent must choose `context_level`
explicitly for `self_mod` plans; non-self_mod plans may omit it (defaults to
`minimal`). That field controls only the generated repository Atlas: `minimal`
omits Atlas accounting for bounded/local plans, while `localized`, `broad`, and
`constitutional` add progressively larger Atlas packs. A typed non-minimal
Atlas assembly failure or final quorum-fit failure rebuilds the same requested
fingerprint/scout wave once at loud `minimal`; compiler exceptions, monetary
budget refusal, and a minimal prompt that still cannot fit do not. Planning
scouts are likewise class-framed: `self_mod` scouts keep the repo-archaeology emphasis;
external/creative/research scouts are steered to the plan's own domain
(requirements, verification, sources, design) and never default to Ouroboros
internals.

Planning has two distinct roots. Governance documents are always loaded from
the system repository; planned snapshots and Atlas inventory always use
`active_repo_dir_for(ctx)`. A workspace/subject mismatch, an unavailable root,
or a `files_to_touch` path escaping that subject must fail loudly. Do not fall
back to reviewing the Ouroboros repo for an external plan. Read-only scouts use
the existing worker pool with its generic `executor=auto` route (selected
healthy harness first, existing loud native fallback) and persist full raw
handoffs. Wait for every launched
scout until it is terminal or the shared swarm ceiling is reached; give the
panel every ready non-empty handoff and an explicit reason for every omission.
Launch only one scout wave per exact plan fingerprint. A handoff is marked
consumed only after it was actually included in the reviewer request; a late
terminal handoff is audit-only and never reopens an already considered plan.
Canonical intent, task aliases, forensic refs, and omissions belong in one
shared evidence horizon—not copied corpora or a second planning engine.

The planning horizon must state the goal, mandatory invariants, scope
boundaries, non-goals, chosen existing extension seam, and explicitly rejected
expansions. Plan review publishes exactly `GREEN`, `REVIEW_REQUIRED`, or
`REVISE_PLAN`. `REVIEW_REQUIRED` findings are inputs: the main agent may accept,
reject, or defer any/all of them. Blocking closes the latest still-current,
reviewed, integrated, non-degraded result without a second LLM call through a
separate `plan_task` call containing `review_disposition` only: every finding
appears exactly once with evidence-based rationale, and each acceptance names
the matching plan revision. Never replay plan/goal/scope/files/context with the
disposition. Mixed calls and vacuous disposition-only calls fail before a new
attempt is recorded; exact replay is idempotent. Blocking `REVISE_PLAN` requires
changed plan text/fingerprint and another panel, while advisory may proceed only
under loud host disclosure and the main agent's rationale. Unknown, stale,
duplicate, contradictory, or incomplete dispositions fail closed. Reviewers
remain generative, but a finding must name
a concrete defect or a concrete smaller existing extension seam; never require
a fixed number of findings.

Force-plan is an LLM-first pre-implementation obligation on the admitted managed
root, not a mechanical permission check around implementation tools. The existing
`plan_review_state` owns durable review authority and
`config.get_review_enforcement()` owns blocking/advisory policy. Every submitted
envelope that reaches `plan_task` supersedes prior authority: invalid plan/goal/scope
input stores a domain-separated open attempt, while a valid envelope stores its
canonical fingerprint before repository/path validation. A newer attempt therefore
cannot fall back to an older GREEN. Immediately before first panel dispatch, the exact planning-scout
handoff component is frozen in a fingerprint-keyed host write-once continuity artifact;
the remaining live reviewer context is rebuilt. An unavailable reviewer
never becomes a disposition-able verdict; a repeat call reuses that handoff snapshot and
retries the panel, including after A→B→A. Blocking stays in
analysis and non-mutating preparation until closure or a real task-wide rail;
advisory may proceed by agent judgment with a host-owned disclosure, including
an explicitly rejected `REVISE_PLAN`. A planning
deadline skip records a typed rail attempt before returning so the reducer cannot
misread it as an absent `plan_task` call.
The short-lived Swarm router admits one new root and transfers the intent; it
never runs `plan_task`, steers an existing task, or publishes the work inline.

**Context mode (Low / Max).** `OUROBOROS_CONTEXT_MODE` controls the Architecture projection in the agent's own context: Max keeps `ARCHITECTURE.md` full for every task class, while Low supplies its lossless navigation map. `DEVELOPMENT.md` is mode-independent and follows the active repository binding. It is full when the task targets Ouroboros's system repository, including self-body and evolution work and a project room with no external binding; a bound external workspace, auto-provisioned external project tree, subagent, or API/CLI/scheduled external surface receives a visible on-demand pointer. Explicit structured overrides remain authoritative. Tier-0 identity and constitutional context stays full in every mode.
For ordinary Main calls, `context_fit.py` may render Max and Low from one
immutable captured core and apply exact family+route calibration. Unknown
routes try Max; there is no silent 200K assumption. Only a confirmed physical
overflow may retry the same model once with a task-local Low projection, with
forensic and owner-visible disclosure. This never changes the global context
mode and never applies to P3 commit/scope review.

### Invariant: Compaction must earn its rewrite

Emergency compaction separates necessity from utility. Total calibrated wire
pressure, including system/context blocks and tool schemas, decides whether
relief is needed; only the compactable transcript and its best reachable
post-pass floor decide whether a pass can help. When too few eligible spans
exist or that floor remains over the trigger, record durable disclosed
hysteresis instead of repeatedly paying a summarizer and rewriting the prompt.
The pass's own rewrite cannot satisfy the growth condition: genuine
compactable-region growth or the bounded round interval must rearm it. Preserve
the independent reactive provider-overflow retry. This prevents an unchanged
irreducible frame from destroying cache reuse while still allowing later useful
compaction.

### Invariant: No silent truncation

If a core governance artifact cannot fit in the available context budget:
- Do **not** silently omit it or truncate it without a visible marker.
- Either adjust the budget/flow to accommodate it, or emit an explicit warning
  (`⚠️ OMISSION NOTE: ARCHITECTURE.md omitted due to budget constraints`) so the
  operator and the model both know the context is incomplete.
- A reviewer or agent operating without ARCHITECTURE.md MUST NOT be treated as
  operating with full context — findings may be incomplete.
- Tools that return multi-model review findings (`commit_reviewed`, `skill_review`,
  scope/advisory review helpers) MUST be listed in
  `UNTRUNCATED_TOOL_RESULTS` or have an explicit per-tool limit; the default
  15KB transport cap is not acceptable for review verdicts.
- A reference-doc **navigation map** (full sections one `read_file` away) and a
  named on-demand pointer are visible, lossless representations — NOT silent
  truncation. The low context mode uses these; it never applies `[:N]` to a doc.
- String bounding goes through the SSOT `utils.truncate_review_artifact`, never a
  hand-rolled `text[:cap] + marker`. Besides the marker, that helper carries an
  anti-waste FLOOR: a cut saving fewer characters than its own omission note is pure
  damage, so below it the text passes through whole. A local re-implementation loses
  the floor and can return a value LONGER than the input it "shortened" (a `…[+N
  chars]` marker is 11 characters, so any overflow under that grew the field).
- Bounding a LIST is subject to the same rule as bounding a string: a `[:N]` slice
  must be accompanied by an explicit omitted COUNT, and — where the slice touches an
  identity that something downstream compares — a durable hash or reference for the
  full set (see `_outcome_receipts.receipt_identity_projection`).

### Invariant: Owner-facing surfaces show the full text (v6.70.0)

Disclosed truncation (the `⚠️ OMISSION NOTE` marker) exists to protect **LLM
context budgets** — it is a model-bound mechanism, not a licence to shorten
what the owner reads:

- **Owner/UI-bound surfaces** (chat panels, task_results projections, review
  verdicts shown to a person) present the COMPLETE text, or carry a reference
  to a durable full copy (e.g. an observability `response_ref`). Reviewer
  rationale is a cognitive artifact (BIBLE P1): projecting it truncated while
  the full copy sits unreferenced in private blobs is partial memory loss.
- **Model-bound projections** (review packs, context sections, tool-result
  transport) keep their disclosed-truncation budgets — those are real context
  economics.
- **A cut cheaper than its own marker is forbidden everywhere.** Truncation
  that saves fewer characters than the omission note it appends is pure
  damage; the shared primitive (`utils.truncate_review_artifact`) enforces
  this floor, and new truncation sites must reuse it rather than hand-rolling
  `[:N]` + marker. One named exception: tiny single-line identifier fields
  (limit < 100, e.g. a reflection backlog `kind`) keep a plain hard slice —
  a multi-line omission marker inside a one-line value is worse damage than
  the cut it discloses.

### Invariant: No "only if touched" gate for core artifacts

Core governance artifacts reach review/reasoning flows unconditionally — NOT only
when they appear in `touched_paths`. The `build_touched_file_pack` function is for
_changed_ files; core artifacts are a separate concern and are loaded independently.

### When adding a new reasoning flow

If you add a new flow that reasons about code structure, system architecture, or
engineering standards, you MUST:
1. Explicitly load `ARCHITECTURE.md` (and BIBLE.md if constitutional reasoning applies).
2. Log a warning if the file is missing or unavailable — do not silently skip.
3. Add a test asserting the file is present in the assembled context/prompt.

---

## Review & Commit Protocol

Reviewed commits separate cheap improvement evidence from authoritative
candidate-bound authority.

1. **Cheap advisory preflight.** After edits, `advisory_review` may find
   omissions before the expensive gate. Without an explicit skip,
   `commit_reviewed` requires fresh advisory coverage and no open advisory
   obligations or commit-readiness debt; any edit makes that coverage stale.
   `skip_advisory_review=True` bypasses only these advisory admission checks.
   Ouroboros chooses the skip by LLM judgment when the pass is slow, unhealthy,
   unavailable, or unlikely to add value, and states why in durable task/commit
   evidence. The skip itself and available gate reason are durably audited.
2. **Authoritative gate.** Fresh advisory coverage or an audited skip is
   followed by independently configured deterministic test policy, staged
   fingerprinting, triad review, applicable scope review, aggregation, and
   pre/post revalidation. The fingerprint binds `git write-tree`, ordered
   `HEAD`/`MERGE_HEAD` parents, indexed VERSION, expected `v{VERSION}` tag and
   existing target, plus the binary staged-diff hash. Advisory skip does not
   change those policies.
3. **Publication binding.** The created commit/tag is checked against the same
   tree, parents, VERSION, tag, and reviewed fingerprint before push. Any
   mutation, rebase, conflict resolution, or changed landing parent invalidates
   exact-candidate authority and requires the applicable final gate again.

Triad slots review the staged diff against `docs/CHECKLISTS.md`; duplicate model
ids remain independent slots and `config.adaptive_quorum` owns quorum. Scope
slots inspect touched context plus the repository Atlas. Required artifacts may
never disappear silently: the assembler reduces optional context and unchanged
diff context, records every degradation, and fails closed if its irreducible
pack cannot fit. Freely degradable touched snapshots move to diff-only first,
largest-first within that tier; an artifact owed in full is reached only after
the `-U0` rung and cannot buy fit by degrading into an invalid review.
Owner-selected Low records the distinct BIBLE P3 scope skip; other route or
assembly failure is not a clean verdict.

The gate is one logical reviewer interaction per API slot. A same-route
transport or empty-response rail may make one bounded second physical send. A
hosted agent-session slot is one multistep execution; local extraction reuses
its collected transcript rather than launching another session. Actor transport,
parse status, semantic verdict, model and route, coverage, cost, and capability
delta remain distinct durable facts.

`docs/CHECKLISTS.md` is the only reviewer-question, severity, and output SSOT.
Architecture owns the dataflow; this section owns operator sequence. Finish all
edits, run focused tests, run the advisory when useful, then freeze and review
the exact candidate. Do not interleave edits with repeated review calls.

### External PR readiness is not commit authorization

`scripts/run_external_review.py --contributor` reviews a clean committed
base-to-head proposal with target-base reviewer defaults and emits shareable
evidence. It establishes readiness and exact base/head/diff facts; it does not
authorize an operator to commit, push, merge, or publish. A proposal changing
the review substrate cannot self-attest its own fast path. Maintainers choose
the landing parent and release version, preserve authorship, and run the normal
final exact-candidate gate. `CONTRIBUTING.md` owns the contribution procedure.
Accordingly, a pull request into `ouroboros` leaves `VERSION`,
`pyproject.toml`, `web/package.json`,
`web/modules/api_types.js::GATEWAY_CONTRACT_VERSION`, the README badge, and
the Architecture header byte-identical to its target. At integration,
`ouroboros/tools/release_sync.py::sync_release_metadata()` projects the chosen
version and `version_carrier_desyncs()` verifies those carriers; changelog prose
remains a deliberate maintainer edit.

Hermetic preflight uses a disposable worktree, temporary data/settings/pycache,
and scrubbed runtime/secret-class environment. Tests must rebind imported
process-global roots and fail closed on the live data root; setting only
`OUROBOROS_DATA_DIR` is insufficient. A reviewed local commit is the durability
boundary; an `origin` push and CI are follow-up signals, not prerequisites for
local self-modification survival.

### DEVELOPMENT.md Compliance Checklist

Before every commit, verify the following:

#### Naming Conventions
- [ ] Modules and variables use `snake_case`
- [ ] Classes use `PascalCase`
- [ ] Constants use `UPPER_SNAKE_CASE`
- [ ] Names are self-explanatory

#### Entity Type Rules
- [ ] **Gateway** (if present): contains ONLY transport. No business logic, no routing.
- [ ] **Tool** (`{verb}_{noun}`): thin LLM-callable wrapper. Validates input, formats output.

#### Module Size & Complexity
- [ ] Module stays near one context window (~1000 lines target; 1600 hard gate unless explicitly grandfathered debt)
- [ ] No non-grandfathered Python function or method exceeds the 300-line hard gate (`GRANDFATHERED_OVERSIZED_FUNCTIONS` is the exception SSOT); methods above 150 lines trigger decomposition review
- [ ] Total Python function count stays under the current smoke hard gate (consult `ouroboros/review.py::MAX_TOTAL_FUNCTIONS` for the active value; bump with a comment if a feature requires more headroom)
- [ ] More than eight parameters is a decomposition signal; consider a typed context object, but do not claim a hard gate or mark existing baseline debt noncompliant
- [ ] No gratuitous abstract layers (Bible P7)

#### Structural Rules
- [ ] New Tool? `get_tools()` exports it using the `ToolEntry` pattern from `registry.py`, an explicit entry is added to `ouroboros/safety.py::TOOL_POLICY` (`POLICY_SKIP` for trusted built-ins, `POLICY_CHECK` for opaque or outward-facing ones), AND the intended visibility is declared in `ouroboros/tool_capabilities.py` (`CORE_TOOL_NAMES`, local-readonly/acting subagent allowlists, parallel/truncation sets as appropriate). If workspace tasks should see the tool, update the workspace allowlist in `tools/registry.py` too. Without the policy entry the tool falls through to `DEFAULT_POLICY = POLICY_CHECK` and pays a light-model LLM call per invocation, and without the capability/allowlist wiring a packaged/visible tool can still be unreachable to subagents or workspace tasks. **A tool that WRITES the repo working tree needs the GUARD surfaces too, not only the visibility ones:** add it to `_ROOT_ARG_REPO_WRITE_TOOLS` (the single set behind the acting-no-workspace fence, the protected-write gate and the acting root-enum narrowing) and make sure its target paths are canonicalized — via `_PATH_NORMALIZED_TOOLS` if it takes a top-level `path`, or via `canonical_repo_relative_path` + `_payload_write_paths` if its paths ride inside the payload. Visibility lists are all green while these are missing, so the gap does not surface as a failing test: `apply_patch`/`edit_batch` shipped a protected-path bypass that way (a guard reading `repo/BIBLE.md` while the write landed on `BIBLE.md`). Tests must exercise the REAL guard chain — a test that monkeypatches the resolver proves the mechanics, not the fence.
- [ ] New Gateway (if extracted)? Contains no business logic, only transport.
- [ ] New memory/data files? Should they appear in LLM context (`context.py`)?

#### Skill Repair Task Constraints
- Skill repair tasks use structured `task_constraint.mode="skill_repair"`, not prompt markers.
- In repair mode, edit paths are payload-relative: `plugin.py` means the selected `data/skills/{external,clawhub,ouroboroshub}/<skill>/plugin.py`.
- Use `edit_text` for one exact replacement and `write_file` only for new files or intentional full rewrites with `root=skill_payload`. (`edit_batch`/`apply_patch` are repo-lane tools and do not accept `root=skill_payload`.)
- Finish repair with `skill_preflight` and `skill_review`; grants and enablement stay owner-controlled.
- Repair mode is a stricter UI lane, not the only path for skill authoring. In `runtime_mode=light`, ordinary chat tasks may edit explicit `data/skills/{external,clawhub,ouroboroshub}/<skill>/...` payloads via `write_file`/`edit_text` with `root=skill_payload`, `bucket`, and `skill_name`. Explicit repo/data paths keep their own address space and ignore stale short-form args. Core/repo paths, `data/skills/native/*`, `data/state/skills/*`, marketplace/provenance sidecars, and direct `run_command` writes to repo targets remain blocked.
- New path checks for skill edits must use `ouroboros.contracts.skill_payload_policy` rather than reimplementing bucket/path traversal logic in each tool.

#### Native-Risk Extension Dispatch
- `type: extension` skills with reviewed isolated dependency envs must not import `plugin.py` or execute handlers inside `server.py`, even when the dependency tree looks pure-Python. Payload-native marker files (`.so`, `.dylib`, `.dll`, `.pyd`) also force child dispatch as defense in depth, but opaque native payloads remain subject to the skill-review checklist and are not newly allowed by this runtime fallback.
- Keep the split explicit: no-dependency pure-Python extensions may use `extension_loader`'s in-process PluginAPI path; isolated-dep/native-marker extensions are cataloged and dispatched by `extension_process_runner` short-lived child processes.
- Tool, HTTP route, and WebSocket handler proxies must return normal tool errors / HTTP 502 / WS log messages on child crash, invalid JSON, timeout, or abort. A child `SIGABRT` is a handled extension failure, not a server crash.
- Child processes must use scrubbed env, per-skill grants, per-skill isolated deps, process-group tracking, output caps, and timeout cleanup. Do not add fallback code that imports native-risk plugin modules in the host process.

#### Task Contract Resource Policy
- When a task contract declares `resource_policy.protected_artifacts`, enforce it as a typed affordance policy in every runtime mode: execute-only black-box references may be run, but byte reads, copy/hash/static introspection, tracing, and debugging against declared paths are blocked.
- Observable Acceptance Claims are bounded, advisory, task-general criteria (`id`, `claim`, `surface`, `support`, `priority`). `success_criteria` is an input alias, not a second persisted carrier. `effective_acceptance_claims` is the only binder: ingress-contract claims win, otherwise the current closed plan wave's frozen claims apply at read time; neither path mutates the live contract. A child receives only claims explicitly passed to its own `schedule_subagent` call. Reviewer `evidence_refs` resolve by exact membership in the already-built host packet, without fuzzy matching, filesystem reads, or re-execution; a claim reference certifies clean only through a passing host-attested support row. Non-passing receipts, agent prose, expected-support text, and unattested references remain named but non-resolving. Resolution changes the clean bit and its disclosure, not actor parsing, quorum, or verdict. Do not turn claims into a hard acceptance gate or surface-specific taxonomy.

#### Devtools isolation

- `devtools/` is tracked operator code outside runtime package discovery and
  the runtime import graph. Runtime modules, `server.py`, web modules, and build
  scripts must not import it.
- Touched devtool files receive normal triad/scope review. Unrelated files may
  remain manifest-only in broad Atlas packs so operator code does not drown
  core review.
- Generated outputs live in an explicit external root, never in `repo/` or live
  `data/`. Domain-specific architecture, launch procedure, and methodology live
  beside the relevant devtool rather than in core governance docs.

#### Light Mode External Deliverables
- `runtime_mode=light` is a self-modification boundary, not an OS sandbox. User-visible deliverables are allowed when they are outside the Ouroboros repo/control-plane.
- Preferred flow: `task_drive` for scratch, `artifact_store` for canonical deliverables, and `user_files` for the owner's visible copy (for example `Desktop/report.html`). `write_file(root=user_files)` and declared process `outputs` must register/copy canonical task artifacts. Rewrites of the same user-visible source keep the previous canonical artifact in non-manifest history with last-5 retention; history is for recovery, not a second deliverable list.
- `run_command`/`run_script` `scratch=[...]` (v6.52.2) is a DISTINCT channel from `outputs=[...]`: it declares EPHEMERAL in-workspace verification files (a throwaway test the agent writes, runs, and deletes — e.g. an in-package test that must live in the repo to compile). Scratch is exempt from the undeclared-output guard, never registered as an artifact, confined to the cwd, honored for NEW files and (v6.56.0) for ADOPTED existing untracked in-cwd files — adoption records the file's sha at declaration time through the SSOT `artifacts.record_task_scratch`, so the patch exclusion applies only while the content still matches (tracked files, paths outside the cwd, and paths outside a git worktree stay blocked; a real edit can never hide behind a scratch declaration) — and excluded from the workspace patch via `.scratch_manifest.json` (`headless.write_workspace_patch_artifacts`). Re-declaring a manifest path is idempotent. The undeclared-output guard verifies candidates POST-exec by stat (exists + mtime ≥ start−slack), so a mere path MENTION (import strings, CLI flags, heredoc bodies) is not a write. Use `outputs` for deliverables, `scratch` for throwaway verification — never overload one for the other.
- `run_command`/`run_script`/`start_service` may use cwd under `active_workspace`, task-scoped `task_drive`, task-scoped `artifact_store`, and external `user_files` where the active profile permits it. In light direct tasks, omitted `run_script.cwd` defaults to task scratch instead of the Ouroboros repo; long-running services in light must use an explicit external/task/artifact cwd. Declared service `outputs` are copied into the task artifact store when the service stops.
- `run_script` temporary files are created under the active workspace when the task is workspace/executor-backed, then removed after execution. Do not run workspace scripts from the system repo temp path; relative imports, generated files, and toolchain discovery must observe the same cwd the user requested.
- Declared process outputs may be files or directories. Directory outputs are copied to the canonical artifact store as a bounded manifest plus zip archive; hidden/control/credential-shaped files, excessive file counts, and excessive byte sizes fail closed instead of leaking through artifact registration.
- In external workspace mode, light-mode self-repo dirty checks snapshot the system repo, not the active workspace. Task-local git operations inside the external workspace are allowed when the task requires them; Ouroboros repo/data paths remain structurally protected, and workspace patch artifacts are captured against the preflight git base.
- Project-room promotion with no working folder and no `workspace="none"` opt-out idempotently provisions a standalone git repo through `ensure_project_workspace`, then runs the ordinary workspace admission checks. Never provision over a non-empty broken binding or an unreadable registry; those cases fail loudly. Binding affects tool profile, memory, lease, and preflight, not the Max-mode Architecture projection.
- Keep policy denials separate from execution failures: `user_files_path_blocked`, `cwd_blocked`, and `artifact_output_undeclared` are non-failure outcomes, while failure to register an explicitly declared output remains `artifact_output_error`.
- The DEFAULT (non-workspace) shell lane carries the SAME target-aware git policy in every runtime mode including light (Q4=A sandbox unwind): mutating git is blocked only when it targets the Ouroboros runtime (system repo / any data drive — bidirectional, casefold, symlink-resolved containment; `commit_reviewed` is the remedy for self-repo changes), read-only git works everywhere including at the system repo, `allowed_resources.network=false` still fences network git subcommands, and acting `self_worktree` children keep the strict no-commit policy. `git init`/`commit`/`push` in `~/projects`, `/tmp`, an attached project folder, or a host-minted coop tree is legitimate task work, not a violation.
- `claude_code_edit` is RETIRED (D10, owner-approved migration, phase 6.4): the SDK edit gateway's job moved to the delegated coding path — a mutating subagent (`schedule_subagent`) whose nanny drives the session with `delegate_start`/`delegate_wait`/`delegate_cancel`, on the owner's subscription when a harness route is configured. Compatibility is one-way and permanent: a saved task contract carrying `disabled_tools=["claude_code_edit"]` also withholds the successor `delegate_start` (registry `_disabled_tools`), and the frozen `GET /api/claude-code/status` + `POST /api/claude-code/install` endpoints stay — the Claude runtime still powers the api-route advisory review. Do not resurrect the tool name.
- Do not recommend `runtime_data/uploads`, skill payloads, or owner state directories as generic artifact transport.

#### Runtime Cleanup / Retention
- All age-based garbage collection of disposable runtime artifacts shares ONE
  owner knob, `OUROBOROS_GC_RETENTION_DAYS` (default 7, hard max 365), and the
  cutoff/clamp math in `ouroboros/retention.py` (`age_cutoff`,
  `clamp_retention_days`, `get_gc_retention_days`). Do not hand-roll
  `now - days * 86400` or `max(1, min(days, 365))` in new prune code; reuse the
  helpers.
- The three former per-subsystem keys
  (`OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS`,
  `OUROBOROS_SERVICE_LOG_RETENTION_DAYS`,
  `OUROBOROS_HEADLESS_TASK_RETENTION_DAYS`) are deprecated and migrated into the
  unified key on settings load (`config.load_settings`). Do not reintroduce them.
  If a subsystem ever genuinely needs its own lifetime, name it
  `OUROBOROS_<SUBSYSTEM>_RETENTION_DAYS` and add it as a fallback in
  `retention.LEGACY_RETENTION_KEYS`, but prefer the unified knob.
- Prune functions keep an explicit `retention_days=` parameter for tests/special
  cases; only the default (None) resolution reads the owner knob. Startup prunes
  are wired from one place (`server.py`).
- Durable artifacts are NOT age-pruned and must stay out of the GC sweep: genesis
  projects (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`) and forensic observability blobs
  (kept compressed indefinitely).
- Review continuations are recovery state, not disposable GC. At Review Continuity
  context build, move a record to `state/review_continuations/archived/` only when
  its owner task is settled, it has remained un-resumed for at least the configured
  seven-day threshold, and none of its recorded obligations remains open. Fresh or
  actionable records stay live; retirement is a collision-safe move, never delete,
  the archive has no runtime reader, and any uncertainty or move error leaves the
  live record intact. This removes closed history from later prompts without losing
  the recovery trail.

#### Live Subagent Task Constraints
- Live subagents are scheduled only through the existing `schedule_subagent` tool.
  Its public schema is strict: `objective` and `expected_output` are required;
  optional public fields are `role`, `context`, `constraints`, `memory_mode`,
  `model_lane`, `executor`, `deadline_at`, `acceptance_claims`, `write_surface`,
  `write_root`, `protected_paths_grant`, `external_tool_grants`,
  `delegation_intent`, `may_mutate`, `may_fan_out`, `max_children`, and
  `required_capabilities`. `schedule_subagent_properties()` owns both this schema
  and the handler's closed keyword set. `effort`, `parent_task_id`, and
  `description` are not public inputs: effort is dispatch-derived, lineage comes
  from `ToolContext`, and capability requirements are admission data rather than a
  frozen contract field. Claims must be plain strings; malformed shapes fail with
  a typed argument error, blanks normalize away, and omission means none rather
  than parent inheritance. Child builders re-state claims and the emptied
  `success_criteria` alias after the parent-contract spread. Boolean grants use
  strict parsing, deadlines can only narrow, and `_narrow_child_delegation_budget`
  can only reduce a subagent parent's authority; a root's explicit mutation grant
  still passes through the ordinary runtime checks.
- Live `memory_mode=shared` is disabled. Keep `forked` and `empty` as the only
  live subagent modes unless a later design adds sanitized shared-context v2.
- External `/api/tasks` and CLI requests must reject forged
  `delegation_role=subagent`; only `schedule_subagent` may create subagents.
- `task_constraint.mode="local_readonly_subagent"` must be enforced twice:
  schema discovery exposes only the local-readonly allowlist, and registry
  execution rejects forbidden calls even when invoked manually.
- Mutative ("acting") subagents (`task_constraint.mode="acting_subagent"`) are
  opt-in via `schedule_subagent(write_surface=...)` plus the master toggle
  `OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS` (explicit owner value applies to every
  surface; unset default is SURFACE-AWARE: every surface ON in advanced/pro,
  light allows `external_workspace`/`genesis` — they build outside the Ouroboros
  runtime — and keeps `self_worktree` OFF).
  `active_tool_profile` must fail closed: an invalid/missing surface, or a
  delegated subagent with a broken constraint, resolves to read-only — never to
  `self_modification`/`operator_control`. Acting children write only inside their
  surface (`self_worktree`/`external_workspace`/`genesis`) and keep commit,
  review, runtime control, tool-enable, skills lifecycle, and cognitive-memory
  writes blocked; `external_tool_grants` is deny-by-default for extension/MCP.
- `genesis` is a from-scratch deliverable surface: the supervisor provisions a
  fresh EMPTY git repo under the durable projects root
  (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`, outside `repo/`/`data/`) via
  `subagent_worktrees.provision_genesis_project`. It is NOT the system repo, so
  protected-path discipline does not apply; it is durable (never GC-pruned and not
  in the worktree registry) because the project directory IS the deliverable. The
  parent does NOT `integrate_subagent_patch` a genesis project into the live body;
  the returned `workspace.patch` (diff from the empty seed commit) is only a record.
- `self_worktree` is a checkout of the system repo: keep protected-path write
  discipline AND protected shell-write guards active for it (no workspace bypass),
  permitting protected edits only in pro AND with `protected_paths_grant`. The
  worktree root must stay outside `repo/` and `data/` (guarded in
  `subagent_worktrees.provision_worktree`).
- `external_workspace` acting children write in the SAME active external workspace
  as the parent. `integrate_subagent_patch` verifies the child's declared
  write/root lineage and that reported files are present, then records a verdict;
  it must not re-apply the patch into the shared workspace because the edits are
  already there.
- The parent is the SOLE committer of the live body. Acting children return a
  `workspace.patch`; the parent applies a chosen patch with
  `integrate_subagent_patch` (manifest-first, sha256-verified, 3-way apply, writes
  a `subagent_patch_verdict` artifact, invalidates advisory) and then runs its own
  `commit_reviewed`. Routing is top-only — never integrate a descendant directly
  into the live repo; bubble patches up one parent at a time
  (`ctx.active_repo_dir()`).
- The supervisor (`_resolve_subagent_constraint`) is the authoritative gate that
  validates the toggle/surface and provisions/validates `self_worktree`;
  `server.py` startup calls `subagent_worktrees.prune_orphans()` (git has no
  worktree GC). Worktree mutations use a dedicated cross-process ops lock, not the
  drive-scoped repo git lock.
- `task_constraint` boolean parsing must be strict; strings such as `"false"`
  are false, never truthy through Python's `bool("false")`.
- The effective delegation budget is a pure admission reducer: declared
  `delegation_budget`, explicit `required_capabilities`, and unresolved
  structured non-advisory `delegation_constraint` rows are reconciled before a
  child runs. Scheduler back-pressure rows may be advisory telemetry (for
  example `queued_behind_active_cap`) and must not block later queued children
  below the hard ceiling.
  Do not infer child needs from objective prose; the LLM declares them via the
  closed enum. Do not add fields to `contracts/task_contract.py` for this.
- Treat a delegated 404 as scoped evidence. Project 404 discharges registration;
  run 404 after owned-daemon reprovisioning closes custody as unreachable, not
  settled, only after registration retirement and without invented usage/spend.
  For a shared project, the lowest run id owns retirement retries and siblings
  defer quietly until removal or a project 404 completes the obligation.
- `delegation_constraint` is a typed task-tree beacon with a structured payload
  (`constraint_id`, directive, scope, rationale). Consumers must read the payload,
  never parse the text. Overrides require an explicit reason and are recorded as
  decision rows.
- Subagent changes must keep writes, commits, review mutation, runtime control,
  tool expansion, skills lifecycle, and shell blocked — except bounded task-tree
  coordination via `tree_note`/`tree_read`, parent-only
  `override_delegation_constraint`, and bounded media projection such as
  `extract_video_frames` writing derived frames only under the task artifact store
  (`artifact_store/video_frames`) through a host-owned command shape (the permitted
  local coordination/projection paths; not arbitrary workspace or repo mutation).
  Nested readonly
  `schedule_subagent` recursion is allowed only within configured depth/cap
  limits; depth bounds nesting only and never rewrites a
  descendant's lane. Enabled/reviewed extension tools and enabled MCP tools may remain
  callable by owner policy, subject to inherited `task_contract.allowed_resources`
  such as no-network/no-web.
- A NEW `plan_task` scout wave is admitted before launch, and only a NEW one: worker capacity,
  the shared review-wave budget gate (`review_helpers.review_wave_budget_gate` — no second budget
  authority), and a consumable window. Each scout's deadline is bound to that window (the wave's
  shared cutoff minus the finalization grace and a margin, the reserve capped at a fraction of a
  short window) instead of inheriting the parent deadline verbatim, and a wave whose window has
  already closed is refused with a typed reason rather than started and then omitted. The
  recovery/collection path is NEVER gated: those handoffs are already paid for, so declining them
  would abandon spend. With `OUROBOROS_MAX_SUBAGENT_DEPTH=0` scouts are refused by the same
  delegation gate as any other child, and `plan_task` then completes on its existing
  `degraded_evidence` path — no wedge, no second wave.
- Runtime-internal scheduling knobs do NOT become `schedule_subagent` parameters.
  `control._schedule_task` is `(ctx, internal, /, **params)`: `params` is validated against
  `control.schedule_subagent_param_names()`, which is DERIVED from
  `control.schedule_subagent_properties()` — the single source the published JSON schema in
  `get_tools` is also built from (unknown keys get the strict v6 refusal). There is no separate
  mirror of the schema to keep in sync, and none may be reintroduced: a hand-maintained copy is
  correct only until one side gains a parameter, at which point the handler refuses something the
  model can see or accepts something it cannot (BIBLE P7). Internal-only options instead travel in
  the POSITIONAL-ONLY `internal` mapping keyed by `_INTERNAL_SCHEDULE_OPTIONS` — structurally
  unreachable from tool-call JSON, which is keyword-only. That is what keeps the handler inside the
  <8-parameter contract. Add an internal knob to that closed set, never to the signature and never
  to the public schema. The test of membership is WHO DECIDES, not who currently calls: an option
  the runtime decides belongs in the closed set, and an option the parent LLM is the right judge of
  belongs in the public schema. `deadline_at` was the set's only member until v6.87.7, when it moved
  to the schema on exactly that test — the parent is what knows when a child's handoff stops being
  useful, and a scout deadline was only ever runtime-internal because `plan_task` happened to be its
  first caller. The set is empty today; it stays because it is closed and an unknown key in it still
  fails loudly.
- `plan_task` planning scouts use the same live-subagent worker pool and one
  shared terminal-or-cutoff wait boundary. Poll in
  `OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC` slices, but wait for every started scout
  until it becomes terminal or the existing
  `OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC` ceiling. At that
  boundary, send every ready non-empty handoff to the reviewer and include every
  omission with its precise terminal/wait reason; missing evidence must never be
  silently presented as complete. Capacity, scheduling failure, or a normal
  cutoff does not trigger an extra inline model call: the omissions manifest goes
  directly to the configured reviewer panel. Repeated calls with the same plan fingerprint
  reuse the existing durable `plan_review_state` wave and never schedule a second wave, including
  when the first wave ended without a usable handoff. Only reviewer-included
  handoffs become consumed. Late terminal results are retained as audit evidence
  with `affects_review=false` and do not reopen the plan. If an included child
  changes after its exact snapshot enters the reviewer prompt, keep the old hash
  non-authoritative, persist the review once with a bounded stale-binding warning,
  and treat the newer child result as audit-only rather than paying for replay.
- `read_file(root=runtime_data)` and `list_files(root=runtime_data)` secret/control-file denials are subagent-scoped.
- Browser isolation for local-readonly/acting subagents (DNS fail-closed): block
  non-HTTP(S) schemes, private/link-local/reserved/unspecified and numeric-obfuscated
  literal IPs, unresolved hostnames, and hostnames resolving to any blocked IP — before
  goto, after redirects, and in route handlers. Loopback HTTP(S) is ALLOWED EXCEPT the
  Ouroboros control-plane ports (agent API / local-model / host-service, the configured
  `LOCAL_MODEL_PORT`, and the actual bound `state/server_port`); `file://` is ALLOWED
  only under the task's explicit `workspace_root` (symlink/traversal-safe), denied
  otherwise. `evaluate` JS stays unavailable to subagents; `vlm_query` /
  `analyze_screenshot` are available. (Relaxed in v6.24.0 for local UI/build inspection;
  control-plane, private-range, and DNS-rebind denial preserved. See ARCHITECTURE.md.)
- Effective task status belongs in `ouroboros/task_status.py`. Do not duplicate
  child-drive merge or terminality in gateways/tools. Task waits use
  `SETTLED_STATUSES`; `cancel_requested` is not settled. `wait_task` and
  `get_task_result` keep the full handoff plus a bounded verification-receipt
  projection: every outstanding red/masked receipt first, then newest rows, with
  an exact omitted count; read the canonical store and fall back to the recorded
  child drive before copy-back. `wait_tasks` stays batch-compact:
  `task_id, status, cost_usd, child_result_sha256, outcome_axes, result,
  trace_summary, capability_delta when disclosable, duplicate_of`; it points to
  the hash-addressed full result rather than re-inlining trace/ledger forensics.
  Unknown ids are probed across result, queue, and tree-ledger authorities and
  return typed rows plus a bounded actual-child roster and its exact omitted count;
  an all-unminted set returns after the 30-second registration grace unless an id
  becomes real. `wait_task`, `wait_tasks`, and `delegate_wait` may disclose an
  expired cache horizon only from the latest recorded applied `5m`/`1h` TTL;
  absent, bare `default`, or unknown TTL evidence stays silent, and no surface
  predicts the next send's token rewrite.
- `forward_to_worker` may write only to validated running tasks whose lineage
  belongs to the current task/root, and must route forked/empty child subagents
  to the child-drive mailbox.
  Do not broaden generic data-tool behavior for normal tasks while fixing
  subagent isolation.
- The pre-final handoff reminder is a compact effective-status snapshot. Full
  untruncated child handoff belongs to `get_task_result` and `wait_task`
  (`wait_tasks` is a compact batch projection — see above). Do not add shared
  ledgers, automatic memory merges, or new settings/endpoints unless the
  accepted plan explicitly calls for them.
- A delegating parent must not produce a clean no-tool final answer while direct
  children are still running and undecided. One bounded absorption reminder is
  allowed; after that, finalization is best-effort (`children_unabsorbed`) rather
  than clean. This is an outcome-honesty rule, not a new wait loop.

#### Page Header Layout
- Top-level page chrome (`renderPageHeader`, tab strips, primary actions) must sit outside the scrolling content region.
- Pages use an outer flex column plus an inner `<page>-scroll` body with `overflow-y:auto`. Skills, Widgets, Settings, and Chat follow this pattern.
- Page icons come from `web/modules/page_icons.js`; do not paste divergent SVGs into individual page modules or the navigation rail.
- Primary page actions, including Refresh, live in the `renderPageHeader({ actionsHtml })` slot on the right. Do not add ad-hoc refresh rows inside scroll bodies.
- Non-chat top-level pages use `.app-page-glass` for the shared dim/brand backdrop. Header padding should stay compact; if a page needs more space, simplify its copy rather than growing the chrome.
- A new top-level page that scrolls its header together with content violates the architecture mirror: fix the layout, not the symptom.
- Top-level page tabs are a single design-system control: `renderTabStrip` + `.app-tab-strip` + `.app-tab` in `web/style.css`, consumed by Dashboard, Skills and Settings. They are **underline tabs, not pills** — a flat label row over one `--divider` rule, with a 2px `--accent` bottom border marking the active tab (`.active`, or the Skills strip's `is-active`). Dimensions come from `--pill-padding-y` and `--pill-font-size` (legacy names from the retired pill strip; `.app-tab` is their only consumer). Do not redeclare per-page tab padding, font size, border radius, or active styling in page CSS files — a page stylesheet may add extra class names to the shared strip, but the geometry, including any mobile variant, is restyled on the shared classes. Pinned by `tests/test_page_chrome_static.py::test_page_tabs_are_underline_tabs_not_pills`.
- Scrollable page bodies use the shared `.scroll-fade-y` mask when content can pass under fixed page chrome. Do not copy/paste custom gradient masks into page modules; extend the shared class if the fade rhythm changes.
- Masonry-style widget packing uses `web/modules/masonry.js::applyMasonry`. Do not reintroduce CSS Grid row packing (`align-items: start`) for unequal-height widget cards; it leaves row gaps under shorter cards.
- Widget card ordering is a host UI preference. Persist it through `/api/ui/preferences` and `data/state/ui_preferences.json`; never rewrite extension manifests or widget declarations to store owner layout.
- New visual dimensions should become CSS variables first (`--button-*`, `--page-header-*`, etc.) and then be consumed by shared classes. Hardcoded page-local dimensions are review debt unless the component is genuinely unique.

#### Setup / Onboarding Layout
- The first-run wizard is a compact multi-step flow. At the default desktop
  window size it should not force scrolling merely because the access step has
  several provider fields; use responsive two-column field grids where width
  allows and keep step copy short.
- Onboarding and Settings share the setup contract. If a key is typed in the
  current unsaved wizard payload, UI diagnostics must account for that in-memory
  value instead of warning from stale saved settings alone.
- Owner switches should expose the semantic choices the owner can actually make.
  For `OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS`, Settings presents Off / Auto / On
  (SC-4): Auto IS the unset, mode-derived state — it saves the empty value so
  the surface-aware runtime-mode default keeps deciding. Unset displays as the
  effective state where a binary label is truthful (advanced/pro = every surface
  on = "On") and as "Auto" in light, where the surface-aware default
  (`external_workspace`/`genesis` on, `self_worktree` off) makes both "On" and
  "Off" false claims. The v6.22.1-era rule that the empty runtime-default must
  not become a third owner-facing button was valid only while the unset default
  was binary per mode; the v6.91 surface-aware light default obsoleted it.
- One capability, one section. Delegation (`OUROBOROS_SUBAGENT_HARNESS`) and the
  write permission (`OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS`) share Models → Subagents
  (`web/modules/subagents_settings.js`), beside Reviewer Slots: both answer "where
  and how far do subagents run". Never render a second control over the same
  settings key — two controls carry two drafts, and the last one collected wins.
  The delegated-run MODEL is the owner's default, authored here as the `=model`
  tail of the same key from engine discovery ("Engine default model" = empty
  tail); reasoning effort stays derived per call, and a hand-written `:effort`
  remainder rides through verbatim with no control over it.
- A control the owner cannot use is worse than none. With no coding-agent
  subscription connected the Subagents section says so and points at Providers →
  Harness Accounts instead of rendering a delegation toggle whose every dispatch
  would silently fall back to an API child. Harness lists come from the accounts
  panel's own source (`accountRows` over `/api/claudexor/status`) — one catalog
  path, one login-capable discriminator.

#### LLM Call Rules
- [ ] New LLM calls go through the shared `LLMClient` / `llm.py` layer — no ad-hoc HTTP clients or direct provider SDKs outside that layer. **Exception (v5.7.0+):** skill / extension `plugin.py` modules may call providers directly because they have not yet been migrated to a host-mediated `api.invoke_llm(...)` bridge. When that bridge lands, the exception goes away. Runtime callers (anything inside `ouroboros/`) must still use `LLMClient`.
- [ ] Every core-mediated physical provider send goes through `usage_accounting.execute_physical_attempt[_async]`: reserve, mark dispatched, then settle/unresolve. A transport retry is a new attempt. `llm_usage`, state, and UI counters are projections carrying attempt ids, never a second monetary authority. Provider tier pricing and any empirical tokenizer margin affect only a known reservation; settlement prefers actual provider usage/cost. Unknown price reserves `None`, remains nullable in usage events, and never blocks a model merely because its tariff is unavailable. An external skill with granted model-provider credentials is explicitly unknown/unmetered when it bypasses core transport—not `$0`; an ordinary spawned process must not be mislabeled as monetary work.
- [ ] Hold the usage-ledger cross-process lock only for budget check, validated append, and fsync. Never hold it over network I/O. Preserve a paid response if settlement persistence fails and leave an honest dispatched/unresolved bound.
- [ ] **Tree-spend visibility.** Under a root cap, pacing and stop text use root-subtree ledger spend including in-flight holds; own cost is diagnostic, and unavailable remains unknown rather than `$0`. Reuse `usage_accounting.last_root_accounting` and refresh only at rare cache-breaking/explicitly stale decision surfaces, never by an unconditional per-round ledger scan or inside a stable cached prefix. `task_pacing.resolve_cost_ceiling` returns `disabled|active|exhausted_soft_land|unknown` from the independent global-percentage and root-cap-minus-absolute-margin axes; graceful finalization precedes, but cannot bypass, the ledger fence. `resolve_deciding_spend` is the sole fallback seam and must label own-cost-under-root-cap as a lower bound.
- [ ] Before dispatching any post-task consolidation or synthesis worker, read `usage_breakdown` once for the whole root subtree and pass the same loop-local snapshot to summary and reflection. It is explicitly non-final (`cost_final=false`, `cost_with_children_partial=true`) and carries child-inclusive accounted cost, reservations, unresolved upper bound, unknown/unmetered count, ledger integrity, and capture time. A read failure is unavailable/null, never `$0`. Consolidation, summary, and reflection model spend belongs only to the existing terminal checkpoint; do not add another ledger or reconciliation LLM call.
- [ ] Runtime notices after the first user/assistant/tool turn are user notices, not new `role=system` messages. `LLMClient` defensively demotes non-leading system messages at the provider boundary; source call-sites should still append `[SYSTEM NOTICE]` user turns so provider payloads, local templates, and prompt authority stay consistent.
- [ ] Keep stable policy/governance first and dynamic evidence last. Prompt-cache support is deliberately narrow: direct OpenAI `prompt_cache_key`, OpenRouter `session_id` (or a caller-declared `cache_affinity` for surfaces whose rounds repeat with changing evidence, e.g. review), and one exact retry without the named parameter only when the provider explicitly rejects that parameter. Do not add provider hops, body rerouting, or a generic cache/retry framework.
- [ ] **Cache-friendliness invariant.** Keep byte-stable governance and task contracts before mutable evidence; never place timestamps, hashes, counters, or task identity in a stable cached prefix. Builders declare bare breakpoints and `review_substrate.assert_cache_breakpoint_cap` keeps the declared count at four or fewer. Only `LLMClient._normalize_payload_cache_ttl` finalizes the assembled wire payload: it supplies a missing tools marker where supported, legalizes TTL order, and discloses any reduction. The owner setting `OUROBOROS_PROMPT_CACHE_TTL=default|5m|1h` stamps existing Anthropic-family markers at that send boundary and never creates new ones; non-Anthropic wire stays unchanged. Preserve cache-affinity keys and exact review bindings.
- [ ] OpenRouter reasoning continuity belongs to OpenRouter conversations only. Direct/local payloads strip OpenRouter round-trip metadata; OpenRouter payloads with `reasoning_details` disable provider fallback to avoid endpoint-bound thought-signature corruption.
- [ ] Claude Agent SDK sessions (the api-route advisory since D10 retired the edit gateway — the edit path's system-prompt file handoff died with it) must preserve the full governance prompt; do not truncate BIBLE/ARCHITECTURE/DEVELOPMENT/CHECKLISTS to avoid argv or transport limits.
- [ ] Delegated (subscription-harness) work is accounted on its OWN ledger row:
  `usage_accounting.record_subscription_session`, which feeds the separate sessions/quota
  axis (`subscription_sessions` / `subscription_windows`). Its cash has THREE states and
  only the first is final: a DISCLOSED ZERO (`spend_usd=0.0`) settles at `cost_usd=0.0,
  cost_final=True` and leaves the projection final — the case this row kind exists for;
  an ESTIMATED amount (`spend_estimated=True`, the engine's `spendEstimated`) rides as
  money but never as finality; an UNDISCLOSED spend (`spend_usd=None`) is `cost_usd: None`
  and counted unknown/unmetered, never a confident `0.0`. Token counts are the same rule
  on the usage axis: `None` means no harness reported it, which is not a run that used
  zero. Do NOT reuse `record_unmetered_external_dispatch` for any of them — it also drops
  the sessions/quota axis. The nanny's own model calls remain ordinary metered attempts
  and keep rolling into the task projection; a subscription session is not counted as a
  physical provider call. D29: the APPLIED `credential_profile_id` and `access_profile`
  the engine's `authRoute` receipt / `effectiveAccess` disclosed ride the durable row (and
  the settled event) BY DEFAULT — empty when telemetry predates the receipt, never
  invented — so "which account paid, under which access" is answerable from the ledger
  row, the settled event, and (for reviewer slots) the last-execution file. Those are
  three separate stores, deliberately not joined into one applied receipt.
- [ ] `cost_final` on a projection is a COUNT of open rows (`non_final_rows`), never a
  truthiness test on a dollar sum: a reserved/dispatched/unresolved row, a settled row
  with an unknown price, and a settled row its writer marked non-final are each open
  however little they cost — `$0.00` is a real reservation for `provider="local"` and a
  real estimate for a delegated run. `non_final_rows` is returned beside the flag because
  a projection can be non-final with every dollar bucket at zero, and a flag without its
  cause is not reconstructible.
- [ ] A spent SUBSCRIPTION WINDOW is `subscription_window_exhausted`, a TRANSIENT class
  carrying `reset_at`, scheduled against that instant rather than through the
  60-second-capped exponential backoff. Do not fold it into `quota_exhausted`, which is
  classified permanent — correctly so for a billing refusal (402, no credits) and wrong
  for a window whose only cure is waiting.
- [ ] Provider failures must be classified before retrying the same request.
  Quota/auth/billing, hard bad-request, and request-too-large/context failures
  are non-retryable as-is: record the exact category and surface a recovery hint
  instead of burning rounds on identical calls. Transient rate limits/timeouts may
  still use the normal retry path.

#### Timeout & Wait Control
- [ ] For cognitive/long-horizon work (subagent waits and review),
  prefer **progress-aware / re-decidable waits** over a single fixed cutoff that
  discards in-flight work. A passive wait that does not kill should stay in its
  window while the observed task is non-terminal **and** progressing, up to a
  generous ceiling, then fail closed with a precise structured reason. Progress
  ADMITS the wait to keep waiting; it does not hand control back per event —
  returning on each advance woke a full-context nanny round every poll interval
  (measured: 18 rounds, 861k prompt tokens, for a run that was doing fine), so
  the observations are carried back once, at the window's expiry.
- [ ] Planning-scout collection is deliberately different: every started scout
  shares one terminal-or-cutoff boundary, and the reviewer receives explicit
  omissions at that boundary without a heartbeat-based early stop or inline
  fallback model.
- [ ] The wait/continue/stop decision must be a **structured fact** — terminal
  status plus heartbeat freshness from `queue_snapshot.json` — not a keyword or
  regex over content (Bible P5). Use `task_status.py` terminal-status helpers and
  the supervisor heartbeat, not string matching.
- [ ] Fixed **kill**-timeouts (hard task/tool ceilings, watchdog) still exist as
  the outer safety bound and get sensible ceilings under high-reasoning models;
  progress-aware waiting tunes the *passive* wait, it does not remove the watchdog.
- [ ] New numeric timeout constants are an SSOT in `config.py` `SETTINGS_DEFAULTS`
  with a getter and env registration; do not scatter magic wait numbers across
  call sites.

#### Loop / State-Machine Changes
- [ ] Changes to `loop.py` or other task state-machine logic include adversarial tests for malformed output, false-completion prevention, replay/log durability, and failure modes — not just the happy path.
- [ ] Audit/checkpoint rounds must not silently reuse the normal final-answer path unless that invariant is explicitly tested and documented.
- [ ] Keep a complete loop-local `DeliveryCandidate` once a substantive answer exists. A service round may return `keep`, or `replace` plus the complete replacement answer; allow one repair for malformed control, then preserve the prior complete answer and mark finalization degraded. A service notice alone does not change evidence. Owner messages, tool effects, child results, and verification receipts advance the evidence revision and require fresh delivery/acceptance binding. Finalize task-scoped service outputs/errors before host acceptance and require a complete replacement when that evidence changes; keep the `finally` path as idempotent cleanup only. This control must not bypass verification, acceptance, safety, skill-finalization, deadline, child-handoff, the unconditional `FINAL ANSWER:` latch, or the task-level answer protocol.
- [ ] Every direct child result needs an exact-hash disposition through the existing `tree_note(kind="decision")` tagged payload (`type=child_result_disposition`, child id, `integrated | irrelevant | deferred`, complete-result SHA-256; note text is rationale). The typed task-tree row is the sole authority; task-result disposition fields are derived reads, never a mirrored write. The join-ledger helper alone validates lineage and current content. Stale or malformed payloads change nothing. `deferred` suppresses only the unchanged reminder and forces an honest degraded/best-effort terminal answer until the item is resolved. Explicit cancellation wins a late-completion race and bounded child scratch is removed without preserving another copy.
- [ ] Host task acceptance is root-only. Queued/headless/scheduled roots are reviewed in `auto` and `required`; direct eligibility is the union of `outcomes.turn_has_reviewable_effects` and a typed deliverable/criterion. Ordinary read-only tool activity, pure conversation, and meta/routing controls are not reviewed, and child reviews remain advisory. Eligibility must use structured facts, never keywords (Bible P3/P5). For an eligible root under `auto|required`, agent-callable `task_acceptance_review` validates/stores evidence and optional agent disposition but makes zero reviewer calls; it returns `deferred_to_host_acceptance`, `authoritative=false`, and the evidence revision. The call itself never widens eligibility; child and `off` behavior remain unchanged.
- [ ] Before root acceptance, atomically fence new descendants under the queue lock and prove recursive subtree quiescence from the existing task-status SSOT. Split-drive ACK, subtree, and acceptance-timing reads/writes use canonical `budget_drive_root`. Preserve the prior verdict until the replacement is recorded. A revision must explicitly reopen the fence; terminal/degraded outcomes seal it.
- [ ] The host runs the authoritative acceptance panel once per unchanged candidate-hash/evidence-revision/fence binding. Task-acceptance actors receive one substantive call and at most two physical attempts total. Record transport status, parse status, and valid-response semantic verdict separately, with actor model/provider, role, coverage, panel id, quorum contribution, reason, enforcement impact, and binding hashes. Public task/event/UI records receive only the compact projection; full model payloads remain in private audit storage. `adaptive_quorum` applies; any contributing FAIL fails, DEGRADED abstains (the reviewer verdict vocabulary `PASS|FAIL|DEGRADED` is NOT narrowable — `_contract_valid_actors`, the deliberate-DEGRADED capsule rail and the host's core-overflow DEGRADED all depend on it), and no quorum is a terminal HOST decision. The host acceptance decision itself is written ONLY by `loop._set_acceptance_decision` and has exactly three owner-facing states — `accepted | revision_requested | finalized_unaccepted` — each with a typed `reason` from an existing structured fact; an unknown status fails closed to `finalized_unaccepted` keeping its raw token as the reason. When you add a writer, add its reason to the closed set AND check every value-keyed reader: `outcomes.derive_loop_outcome` keys the eligible-but-skipped degradation on the status+reason PAIR (`review_skipped_deadline_reserve` plus the closed forced-rail `ACCEPTANCE_BYPASS_REASONS`), and breaking that pairing is a silent false green. Forced exits stamp their typed bypass record in the common terminal recorder (`_record_forced_acceptance_bypass`) as a pure ledger write — never a fence, panel, extra round, or prompt text on a forced path, and never overwriting an existing host decision. The agent may write only `agent_disposition`/`agent_rationale`, merged into the host decision, never replacing it. Clean requires PASS + solved + supported criterion evidence. Chat and Logs must use the same severity reducer, and degraded review or best-effort/degraded objective must never render as green solved. Do not add task scope review or reuse the commit gate.
- [ ] The acceptance improvement loop is a reviewer-authored DIALOGUE (v6.74.0): obligation identity comes from the reviewer's typed `disposition_kind`/`obligation_id` (an unknown re-raise id fails closed to `new`, disclosed — never a silent fresh hash id); a re-raise reopens the row WITHOUT wiping the agent's argument (`previous_disposition`/`previous_reason`/`reopened_count` survive into the evidence catalog and the obligations clause); termination beyond a clean PASS/accepted rebuttal happens ONLY via the reviewers' quorum `dialogue_status` judgement reduced over ALL contract-valid actors (`aggregate_dialogue_status` — never `_contributing_actors`, which drops a DEGRADED slot's vote) or a real rail — no host counters, no answer/verdict hashes, no keyword gates (P5). Changes here must cover: malformed reviewer output, unknown/stale `obligation_id` on a re_raise, partial panel failure, multi-slot dialogue-status disagreement (the reducer's precedence), replay/restart durability of obligation rows, false completion, and the backward-compatible default when the new fields are absent.
- [ ] An explicit `max_improvement_passes` binds under every legacy policy. Required+Blocking without one has no local count cap, but real deadline/budget/lifecycle rails remain. The first acceptance review reserves at least 200s; later passes use the canonical event-derived `max(floor, 1.5×EWMA)` (`alpha=0.5`). Only the root runs global post-task synthesis once and persists one phase checkpoint in the canonical `budget_drive_root`. Recovery is startup-only: replay `pending_once`, degrade indeterminate `running` without another paid call, and let the normal supervisor copy-back/artifact path materialize child results without overwriting a terminal canonical phase.

#### Cognitive Artifact Integrity
- [ ] Cognitive artifacts (identity.md, scratchpad, task reflections, review outputs, pattern register) must NOT use hardcoded `[:N]` truncation. If content must be shortened, include an explicit omission note (e.g. `⚠️ OMISSION NOTE: truncated at N chars`).
- [ ] `BIBLE.md`, `docs/ARCHITECTURE.md`, and `docs/DEVELOPMENT.md` are **core governance artifacts**. All primary reasoning flows (triad review, consciousness, advisory pre-review, deep review) include them as first-class sections — see the "Core Governance Artifacts" table. If you add a new reasoning flow, it MUST follow this contract, not rely on touched-file inclusions.

---

*This section is the authoritative definition of "DEVELOPMENT.md compliance" referenced in the `development_compliance` item in `docs/CHECKLISTS.md`.*

---

## Managed Update Rule

- Keep the local work branch and the official update feed separate. The local
  branch is `ouroboros`; `OUROBOROS_UPDATE_CHANNEL` maps Stable to `main`, QA to
  `ouroboros-stable`, and Development to `ouroboros`. QA and Development follow
  their branch tips. Stable resolves the newest plain `vX.Y.Z` tag whose commit
  is present in both `main` and `ouroboros-stable`.
- A preflight chooses one exact official target SHA. Apply must bind to the
  disclosed base/target, close new writers, drain existing direct/ephemeral
  turns, stop workers and tracked services, then re-plan before mutation.
- Clean fast-forwards land the official SHA directly. Git also builds clean
  merges for divergent local history, with parents = reviewed HEAD + official
  target. Dirty local work never enters that history: the apply stashes it and
  restores it as uncommitted content (boot finalize on success, the pre-update
  tree on rollback; a conflicting restore keeps the stash and discloses the
  recovery command). The reviewed assisted resolver runs only when Git reports
  a real conflict; filenames do not create a second update policy. Hard reset
  is an explicitly confirmed recovery only.
- The authorized resolver stages the complete merge, including tracked binary
  files. Review receives their exact staged mode/blob/size plus the HEAD and
  official MERGE_HEAD object ids; deletions carry an explicit absent stage and
  exact parent identities. Missing exact metadata still blocks. This exception
  does not weaken the ordinary commit pipeline's binary policy.
- Write the update transaction before mutation. Reopen writers only after a
  verified abort/rollback or a healthy restart. An unverified rollback keeps
  its retryable phase plus the full failure evidence; a legacy `gate_blocked`
  marker retries rollback on boot. Delayed evolution cleanup also acquires the
  same update lock and honors this admission owner; it must not stash/reset
  behind the fence. Managed merge tests pass before restart; the ordinary
  self-modification commit/tag/test/push ordering remains unchanged.
- Manual Restore reuses the same writer fence and pins the previous HEAD on a
  local recovery branch before reset. Promotion resolves the development SHA
  once and uses that exact SHA for both the local QA ref and any remote push.

---

## Mutation Attribution Rule

- Attribution is evidence, not exclusion. The host captures a `system_repo`
  baseline in the existing task result when a queued root task starts and a
  terminal candidate snapshot at outcome derivation; blockers (pre-existing
  dirty changed, stale/missing baseline, failed scan) ride into review and
  acceptance evidence for the LLM panels to weigh. Do not turn them into
  structural outcome vetoes, and do not add a lease/holder service, a second
  ledger, or runtime writer keyword scanners.
- Git staging is attribution-based. `paths=None` means the clean-at-baseline
  candidate set, an explicit list must be its subset, and empty never means
  `git add -A`. Preserve pre-existing user dirt as excluded evidence. Whole-tree
  staging belongs only to already-typed managed update/release transactions and
  a dedicated typed external patch-capture transaction. Contexts without a captured baseline
  (manual ToolContext, external dry-run review) keep the legacy staging
  contract.
- Resolve unversioned Python only for `run_command`, `run_script`,
  `start_service`, and run-kind `verify_and_record`, once before the shell guard.
  Guard and handler must receive identical argv. Do not rewrite explicit paths,
  versioned interpreters, shell bodies, or remote execution, and never install a
  dependency in response to `ModuleNotFoundError`.
- Skill Review ordinals and provenance stay in `review_job.json` and the
  append-only `review_history.jsonl`: allocate under the lifecycle lock, consume
  a round only after actual start, write one terminal row per `job_id`, and
  compute legacy ordinals at read time without rewriting history.

## Process Custody Rule

Long-lived OS processes (anything `subprocess.Popen`-ed or `mp.Process`-ed
without a bounded wait in the same call) **MUST** be spawned through
`ouroboros.process_custody.spawn_supervised(cmd, drive_root=..., purpose=...,
scope=...)` — or, when an existing manager owns the Popen call, registered via
`record_process(...)` write-through immediately after spawn. The custody
ledger (`data/state/process_ledger.jsonl`) is what lets the orphan reaper find
children after an abrupt worker/server death; an unledgered process orphans
invisibly and forever. Scopes: `task` (dies with its task), `session` (dies
with the server generation), `daemon` (genuine launcher-managed lifecycles,
e.g. `server_restart_fallback` — reaper keeps them, only pruning dead entries).
Skill **companions** also record `daemon` scope but are the documented
exception: `reap_orphaned_processes` reaps a companion (`purpose
companion:<skill>:<name>`) when its owning skill is **uninstalled** OR the entry
is from a **foreign (dead) server generation** (`CompanionSupervisor.start()`
always re-spawns a fresh pid, so a generation-crossing match is a stale
duplicate). This is **log-only by default** (`enforce_companion_reap=False`
emits a `process_would_reap` event instead of killing) and **fail-safe**:
`live_owner_skills=None` (unknown install set — incl. a momentarily empty skills
dir, coalesced to `None`) means keep-all, never a mass-kill, and same-session
companions of installed skills are always kept so the live `CompanionSupervisor`
stays their sole owner. The reaper kills strictly by (pid, start_time,
cmd_sha256) fingerprint — never add command-line-class matching, which would let
a dev instance reap a packaged instance's processes.
`tests/test_process_custody.py` enforces the chokepoint with an explicit
allowlist for bounded synchronous helpers.

## Platform Abstraction Rule

All platform-specific code **MUST** go through `ouroboros/platform_layer.py`.

### Shared State-File Helpers

Durable JSON state files should use the SSOT helpers in `ouroboros/utils.py`:
`atomic_write_json(path, payload, trailing_newline=False, fsync=False)` for
write-then-rename persistence and `read_json_dict(path)` for dict-shaped JSON
reads. `write_text_atomic(path, content, fsync=False)` is the underlying shared
atomic FULL-OVERWRITE primitive (temp-sibling + `os.replace`, existing permission
bits preserved, crash leaves the old file intact); `atomic_write_json` layers JSON
serialization on it, and `write_text` (the plain text overwrite helper) routes
through it, so every overwrite routed through these helpers is crash-safe — prefer
them over a bare `Path.write_text` for any full-file overwrite. Appends are
intentionally NOT atomic (they extend in place). Lockfile acquisition should go through
`platform_layer.acquire_exclusive_file_lock` /
`release_exclusive_file_lock` rather than reimplementing `O_CREAT|O_EXCL`
loops in feature modules.

Narrow exceptions are allowed only when the file's contract is not JSON-object
state or intentionally has extra durability semantics: `supervisor/state.py`
keeps `atomic_write_text` for mirrored `state.json` / `state.last_good.json`
text writes, and `ouroboros/config.py` keeps its settings-file lock because the
settings path is bootstrapped before broader runtime helpers should depend on
settings state.

### What counts as platform-specific

- Direct use of: `os.kill`, `os.setsid`, `os.killpg`, `os.getpgid`, `signal.SIGKILL`, `signal.SIGTERM`
- Unix-only modules: `fcntl`, `resource`, `grp`, `pwd`
- Windows-only modules: `msvcrt`, `winreg`, `ctypes.windll`
- `subprocess` with platform-conditional flags: `start_new_session`, `creationflags`
- Hardcoded path separators (`/` or `\\`) in filesystem logic (use `pathlib` instead)

### Rules

1. **All platform-specific calls live in `platform_layer.py`** — the rest of the codebase imports cross-platform wrappers from there.
2. **Platform-specific modules are imported inside `platform_layer.py` only**, guarded by `IS_WINDOWS` / `IS_MACOS` / `IS_LINUX` checks.
3. **No top-level imports of Unix-only or Windows-only modules** outside `platform_layer.py`. If you need `fcntl` — you're in the wrong file.
4. **Use `pathlib.Path`** for filesystem paths. Never construct paths with string concatenation using `/` or `\\`.

### Enforcement

- **AST-based test** (`tests/test_platform_guard.py`): scans `.py` files under `ouroboros/`, `supervisor/`, and `server.py` for:
  - Top-level imports of platform-specific modules (`fcntl`, `msvcrt`, `winreg`, `resource`)
  - Direct `os.kill`, `os.killpg`, `os.setsid`, `os.getpgid` attribute access
  - Direct `signal.SIGKILL`, `signal.SIGTERM` attribute access
  
  Not scanned by the AST guard: `launcher.py` (release-reviewed outer launcher,
  intentionally excluded) and subprocess flag patterns (`creationflags`,
  `start_new_session`). For subprocess isolation, use
  `subprocess_new_group_kwargs()` and `subprocess_hidden_kwargs()` from
  `platform_layer.py` — enforced by code review and the `cross_platform`
  checklist item.
- **Pre-commit review**: checklist item `cross_platform` (#15) catches violations during code review.
- **CI matrix**: tests run on Ubuntu, Windows, and macOS to catch runtime failures.

### Adding new platform-specific code

1. Add the cross-platform wrapper to `platform_layer.py`.
2. Import and use the wrapper in callers.
3. Add platform-conditional tests if behavior differs across OSes.

---

## Design System

`web/style.css` custom properties and shared component classes are the value
SSOT. Documentation keeps semantic roles and failure-prevention rules, not a
copied color/radius/dimension inventory.

Ouroboros uses a **flat neutral-gray + red developer-tool** visual language.
Every surface is a SOLID background token plus a 1px border. There is no
glassmorphism and **no `backdrop-filter` anywhere in the main SPA** — depth
comes from the background ladder, borders, and (sparingly) shadow:

```css
background: var(--bg-elevated);           /* cards, menus, composer, controls */
border: 1px solid var(--surface-border);
```

`grep -n "backdrop-filter" web/style.css` must return NO declaration — the only
match is the token banner at the top of the file stating that there is none. If a component
appears to need a blur to separate itself from what is behind it, it needs a
higher background token or a border — not a filter (removing a
`backdrop-filter` also dissolves the stacking context it silently created, so
components that relied on that get an explicit `isolation` / positioned
`z-index` instead).

**Onboarding is the one deliberate exception.** `web/onboarding.css` styles the
standalone first-run wizard, which is served as its own page and cannot import
`web/style.css`. It keeps its own translucency and blur formulas; only its
`:root` COLOR VALUES are kept in sync with the palette below. Do not "fix" its
blur, and do not copy it back into the SPA.

### Token families

All of these live in ONE `:root` block in `web/style.css`. A component rule
references a token for every COLOR it paints; it never hardcodes a literal.
Adding a visual dimension means adding a token first.

**Two deliberate exceptions, both `rgba(0, 0, 0, …)`.** Neither is a palette
color, and neither can be expressed as one:

* **Drop shadows** (`box-shadow`) are a depth cue, not a hue. They must darken
  whatever is behind them on any surface in the ladder, so they take true black at
  a per-elevation alpha. `--bg-primary-rgb` would tint them toward the page
  background and flatten the elevation they exist to signal.
* **Overlay scrims** — the `position: fixed; inset: 0` backdrops behind modals,
  the mobile nav drawer, the reconnect overlay and the panel dismiss layers —
  darken the whole app so the layer above reads as separate. Routing them through
  `rgba(var(--bg-primary-rgb), α)` does not work *arithmetically*: over
  `--bg-primary` that composites to exactly `#131315` at **every** alpha, so the
  scrim becomes invisible, and over `--bg-panel`/`--bg-elevated` it lands within a
  few units of the page background. The dim is the whole point, so these stay
  black.

A new `rgba()` literal anywhere else — a text alpha, a surface tint, a border, an
accent wash — is token debt. Take the nearest rung of the existing ladder and
record the delta in a comment (see `.nav-budget-label` and `.app-tab`, both of
which land a `0.45` handoff spec on `--text-secondary` at `0.55`) rather than
introducing an off-ladder value in shared chrome.

| Family | Tokens | Role |
|---|---|---|
| Backgrounds | `--bg-primary` `#131315`, `--bg-sidebar` `#0f0f11`, `--bg-panel` `#151517`, `--bg-elevated` `#1a1a1d` | page → sidebar → rails/panels → cards/controls |
| Background channel | `--bg-primary-rgb` | the one hue every gradient/edge FADE interpolates (not the scrims — see above) |
| Text | `--text-primary` `#e7e7ea`, `--text-secondary`, `--text-muted` | body / secondary / hint |
| Lines | `--divider`, `--surface-border`, `--surface-border-soft` | separators and surface edges |
| Accent | `--accent` `#c93545`, `--accent-hover` `#d4485a`, `--accent-light`, `--accent-chip-text`, `--accent-hover-rgb`, `--accent-04…55` alpha ladder | brand red: primary actions, active nav, task identity |
| Roles | `--accent-task`, `--accent-system`, `--accent-user`, `--accent-project`, `--tone-ok/warn/danger` | name the SIGNAL, not the hue |
| Voices | `--user` `#6e96d2` (owner), `--project` `#2dd4bf` (project identity), `--system-*` amber | who is speaking |
| Status | `--green` `--amber` `--red` `--blue` (+ `-dim`, `--blue-08`) | outcome hues |
| Diff | `--diff-add-bg` `--diff-del-bg` `--diff-add-num` `--diff-del-num` `--diff-add-text` `--diff-del-text` `--diff-ctx-*` `--diff-hunk-bg` | unified AND split render from the same pair |
| Syntax | `--code-keyword` `--code-self` `--code-string` `--code-number` `--code-comment` `--code-default` | the in-house highlighter names lexeme roles |
| Type | `--font-mono` = `ui-monospace, Menlo, monospace` | every code/path/number/amount surface |
| Shell | `--sidebar-width`, `--project-panel-width`, `--inspector-width`, `--nav-item-radius` | app shell geometry |
| Spacing | `--space-1…6` (8pt scale) | never ad-hoc pixels |

The project identity hue moved from fuchsia to teal in the flat redesign: the
role (and every consumer) is unchanged, only the value moved, because that is
what a token layer is for.

Mono text uses `var(--font-mono)`. Do not paste a new font stack
(`'SF Mono', 'JetBrains Mono', …`) into a component rule.

### Layout and controls

- Top-level pages use a fixed `renderPageHeader` outside an independently
  scrolling body. Shared tab strips, buttons, scroll fades, page glass, icons,
  and masonry helpers replace page-local copies.
- Static visual properties belong in CSS classes/tokens. New inline
  `style=""` markup and `.style.<property>` assignments are review debt.
  Dynamic measured values may update a narrowly named CSS custom property when
  that is the actual runtime data flow.
- One semantic button variant expresses one action role. Notifications use the
  shared toast host unless status belongs to a permanently reserved control
  row. Working, warning, error, and destructive states keep consistent meaning
  across Chat, Logs, Settings, and Skills.
- Floating chrome is a flat darkening gradient with **no** blur band, and the
  scrolling surface reserves measured padding around it rather than hiding
  content underneath — see *Docks and page-chrome overlays* below.

### Docks and page-chrome overlays

Chrome that floats over scrolling content (the chat header, the chat input
dock, sticky tab strips) is a **flat darkening gradient with no blur band**:

1. The element is `position: absolute` on its edge (`top: 0` for headers,
   `bottom: 0` for docks) and spans the horizontal axis.
2. Its background is a single multi-stop `linear-gradient` from
   `rgba(var(--bg-primary-rgb), …)` at the anchor edge to fully transparent at
   the far edge, so the transcript fades out instead of hitting a hard line.
   The intermediate stops are anchored to the FAR edge as a fixed length
   (`calc(100% - 30px)`), never as percentages. This chrome changes height — the
   chat header wraps from one row (56px) to three (~129px at 375px) — and
   percentage stops stretch with the box, which is what left the whole wrapped
   control row sitting over effectively unmasked transcript text. A fixed tail
   keeps the controls on a solid backdrop at any height and always spends the
   same distance softening the edge.
3. **No `backdrop-filter`, and therefore no companion `mask-image`.** The mask
   only ever existed to fade a blur in lockstep; with the blur gone the mask is
   dead weight (`.chat-page-header` pins `mask-image: none` for exactly this
   reason).
4. The control surface INSIDE the dock is a normal flat surface
   (`--bg-elevated` + 1px border + accent focus ring), not a frosted pane.
5. The scrollable surface reserves space for the chrome instead of hiding
   content under it: `#chat-messages` reserves bottom padding through
   `--chat-input-reserve` and top padding through `--chat-header-reserve`, both
   set from the REAL measured height by the `updateMessagesPadding()` /
   `ResizeObserver` pair (mobile adds safe-area on top). The header reserve reads
   `getBoundingClientRect().height`, not `offsetHeight`: the latter is already
   rounded down to an integer, so a fractional wrapped header (129.39px at 375px)
   would reserve one pixel too little. `updateMessagesPadding()` preserves scroll
   stickiness only; it must not mutate DOM padding.

Do NOT introduce a separate `.chat-bottom-fade` (or analogous overlay) layer.
A second fade layer compounds the gradient and can produce a visible "double
dim", especially over short messages. One gradient, on the chrome element.

### Control rules

- Composer, toolbar, segmented, and widget-reorder controls share one flat
  grammar: a solid background token, a subtle 1px border, and a bounded radius
  from the scale. Do not add transparent text-only pills for primary actions.
- Top-level page tabs are **underline tabs**, not pills — see "Page Header
  Layout" above for the full rule.
- Chat-header controls are **ghost buttons** (`.chat-header-btn`: transparent
  background, `rgba(255,255,255,0.10)` border, `--radius-7`, hover
  `rgba(255,255,255,0.06)`) with exactly one danger variant
  (`.chat-header-btn.danger`, accent-tinted) for Panic. Overflow lives in the
  slim `.chat-header-more` `<details>` menu, which must auto-dismiss on an
  outside click and on Escape.
- The chat composer is one flat 12px box: a control row (Swarm, Low/Max, and the
  READ-ONLY model chip pushed to the end) sits above a text row holding the attach
  button, the `composer_parts` mount, and Send. Send is the composer's one PRIMARY
  action (solid `--accent`), not a ghost pill. The model chip rides the control row
  rather than the field because at 390px a mono model id inside the text row eats
  the usable input width; it is hidden entirely until the Settings snapshot names a
  model (never a guessed default).
- Button and segmented-control labels use `letter-spacing: 0` and stable
  dimensions. If a label does not fit on mobile, shrink the control group or
  move it to another row; do not reserve a large textarea padding gutter.
- Drag/drop affordances are stateful CSS classes (`drag-active`, `drag-over`,
  etc.) on the host control/card. Do not use inline styles for visual feedback.
- Context-capture chips (⌘L) are one shared DOM contract in
  `web/modules/composer_parts.js` + the `.composer-part-*` classes. Every
  composer (chat, Changes dock, Files dock) mounts that module rather than
  restyling its own chip. The same chip appears again in the sent message as
  `.chat-context-chip` — same mono/accent grammar, no second look for the same
  thing.

### Flat chat transcript

The transcript is one centred reading column (max 760px, gutters via
`padding-inline: max(…, calc((100% - 760px) / 2))` — no wrapper element, so every
insertion path stays untouched). The owner's message is the ONLY bubble left; agent
prose and task cards are plain rows with a sender line. Task activity is mono rows
behind a 2px left rule, and the `✓`/`●`/`✕` status glyph is a `::before` keyed off
the row's phase class — add a status by adding a phase rule, never by adding markup
to the row (the `data-live-line-*` disclosure contract is pinned). Row titles wrap;
do not ellipsize owner-facing narration. Only transient status events (awakened /
reconnected, marked `.chat-status-event`) collapse into a centred pill — a rich
system renderer keeps its full disclosure.

### CSS ownership anchors

`web/style.css` carries `/* ===== [stream …] ===== */` banner comments marking
which screen owns which region (app shell, chat, changes/inspector, files,
dashboard/skills/settings/widgets, plus the shared composer-parts contract).
Add a rule inside the region that owns it. `web/app.js` carries the matching
append-only `/* [anchor:phase-…] */` regions for cross-screen registrations
(right-panel kinds, the global capture hotkey).

A rule that two screens genuinely share belongs in a `[shared: …]` region, NOT in
whichever screen's region it was written in first. `[shared: status tone ladder]`
is the worked example: the done / warn / error tones for small status pills are
read by the chat's phase pills, the Logs grid's phase column and the Evolution
runtime pills, so one screen's region deciding another's colours is the thing the
banners exist to prevent. Every rule there is at least two classes, which is what
lets it outrank the per-surface pill bases on SPECIFICITY rather than on where the
region happens to sit in the file — a shared region must never depend on file
order, because the regions around it are independently owned and get reordered.

### Responsive and accessible behavior

Navigation, headers, controls, and dialogs must stay operable by pointer and
keyboard, preserve focus order, and fit the relevant narrow viewport without
stealing usable text space. Use the shared responsive component before adding a
page-specific layout. A visible change is inspected with vision in at least one
relevant real consumer flow. A stored screenshot alone is not verification;
mobile or WebKit is not a universal requirement and is selected from risk.

### Browser dialogs

`window.prompt`, `window.confirm`, and `window.alert` are forbidden in `web/modules`. PyWebView shells implement them inconsistently, native dialogs bypass the design system and browser tests, and the macOS shell has no prompt delegate, so `window.prompt` silently returns `null`. Use `confirm_dialog.js::openConfirmDialog`: confirm mode returns a strict boolean, input mode returns `{confirmed, value}`, and alert mode renders one acknowledgement action. Close, Cancel, backdrop, Escape, and supersession are always non-confirming. Critical actions must test the exact confirmed result and keep the confirmation plus side effect in one injectable flow. `tests/test_web_dialogs_static.py` keeps the native-dialog class closed; host-side PyWebView bridge dialogs are a separate platform surface.

### Declarative widgets

`web/modules/widgets.js` is the host for reviewed widget declarations.
Declarative v1 includes forms/actions, text/data/media, tabs/charts, async jobs,
files, map/calendar/kanban, and composition through `group`, `metric`, and
`callout`. Nested interactive components use stable identity and one disposer;
`subscription.render` is transitively passive. Escape text and attributes for
their actual HTML contexts, constrain media to extension routes or safe data
URLs, and keep charts accessible through a semantic table.

Rare `kind: "module"` UI runs only in a sandboxed opaque-origin iframe, with no
`allow-same-origin`; its parent bridge proxies only the owning extension route.
Never load skill JavaScript into the SPA origin. Long-running actions use a
durable job id and resumable status polling rather than a foreground request
lost on remount.

Every timer, listener, observer, stream, abort controller, chart, and mounted
widget has a paired disposer. UI preferences such as widget order belong in
host state, never in extension manifests.

### Navigation sidebar (v6.32.0 redesign; sections + brand + budget in the flat-UI redesign)

The desktop navigation is a left `#primary-sidebar` of ROWS (not an icon
rail): each destination is a `.nav-row` (16px icon + 13px label, 34px tall) and
the Projects group is a `.nav-section-toggle` that expands a data-driven list of
project rows (`renderProjectsNav` in `web/app.js`, fed by `/api/state`).
`syncNavigationState` keeps the active row, the Projects expand/collapse, and the
open right panel in sync. A project opens as a right split panel on desktop and a
full-width overlay with backdrop on mobile, hosting a full chat instance over the
ONE shared WebSocket (client-side fan-out by `chat_id`). On mobile the sidebar
collapses behind an "Open navigation" toggle (drawer), NOT a horizontal bottom
bar. Spacing/typography come from the shared design tokens in `web/style.css`
(no per-screen hardcoding); global agent controls (Evolve/Review/Restart ghost
buttons + Panic, and the slim "More" menu holding Consciousness) live in the chat
header, not the sidebar.

The sidebar has three fixed parts:

- **Brand row** (`.nav-brand`, outside the scroll region): the 26px app mark
  (`/static/favicon.png`, `--radius-7`), the product name, and one sub line
  carrying `#nav-version` (filled by `loadVersion()` — the ONE version span) plus
  the live socket state and a `.nav-status-dot` driven from the shared WS
  `open`/`close` events. There is no second version label anywhere.
- **Scrolling rows** (`.sidebar-scroll`): Main Chat, the Projects section
  (mechanics unchanged), then `.nav-section-label` groups — **Workspace**
  (Changes, Files) and **System** (Dashboard, Skills, Widgets, Settings). Section
  labels are 10px/600 uppercase; page glyphs come from
  `web/modules/page_icons.js` and are hydrated by `hydrateNavIcons()`. A new
  destination needs a `PAGE_ICONS` entry, not a pasted SVG.
- **Budget meter** (`.nav-budget`, pinned to the bottom): label row + mono amount
  + a 3px bar. It renders `chat.js::headerBudgetPresentation` — the ONE budget
  formatting projection, which fails closed to "Unavailable" and never shows a
  fabricated `$0`. The bar fill is written as the `--budget-fill` CSS custom
  property (the accepted dynamic-value exception below), never `style.width`.
  There is exactly one budget meter in the app; the old chat-header pill is gone.

Every consumer of `/api/state` (sidebar budget, chat header controls, projects
nav, task bindings) reads ONE app-owned snapshot: `refreshState()` in
`web/app.js` publishes to `subscribeState` handlers on a single self-scheduling
timer (~3s while Chat is visible, ~20s elsewhere, paused while
`document.hidden`), forced on WS `open`, `projects_changed`, and owner-control
writes. Do not add a module-local `/api/state` poll or a second timer.

The right panel is ONE slot with mutually exclusive kinds (`project` today, plus
whatever registers via `registerRightPanel(kind, {mount, unmount})`). Opening one
kind closes the other, and navigating away closes the panel. The project kind
keeps its persisted drag width; the task inspector is fixed at
`--inspector-width`.

The compact Projects header keeps the shared layers icon, label, unread pill,
chevron, and an always-visible `+`. Project rows expose one sibling Rename/Delete
menu, reachable by pointer and keyboard; Enter/Space open, Escape closes, focus
order stays logical, click-outside closes, and placement is viewport-safe. Name
validation uses the backend `PROJECT_NAME_MAX` SSOT (80), never a divergent UI
constant. Unread is `visible_revision > project_seen_revision`; acknowledge only
after the room has painted, and make cursor writes monotonic/server-clamped.
Routine task heartbeat telemetry must never create a bubble or unread revision.
Only typed real incidents may enter the live card/Activity plus one deduplicated
toast.

Project history is a projection of canonical chat rows, not a mirror log. A
presentation annotation sidecar may store the latest routing action/target/status
for a `client_message_id`, but it must never become routing or Project-state
authority. Deletion is fenced `active → deleting → tombstoned`; preserve id,
bindings, chat/history, folder, and memory, and never permit resurrection.

<!-- Historical (pre-v6.32.0 icon rail; superseded by the sidebar above):
The desktop `#nav-rail` used Material 3 / Apple HIG navigation-rail
spacing norms: `padding: 28px 0 16px; gap: 10px;`. The previous
`12px / 4px` was visibly cramped (the first button hugged the top edge
of the viewport). Bump these values together when adding new nav
buttons; resist tightening them.

On mobile (`@media (max-width: 640px)`) the rail flips to a horizontal
bottom bar with `justify-content: safe center`. The `safe` keyword
keeps the row centered when content fits and gracefully degrades to
flex-start when content overflows on very narrow phones. `min-width:
60px` per `.nav-btn` keeps labels like "Dashboard" from truncating in
space-evenly mode.
-->

The mobile `.scroll-tabs` pattern (settings/dashboard/skills) uses
horizontal-scroll pills with `scrollIntoView({ inline: 'center' })`
on activation so the active pill is always visible. Do not reintroduce
the v5.6.0 drill-down accordion (`settings-subtab-open` /
`settings-mobile-back`) — it traded one tap for two.

### Notifications

Transient status must use `web/modules/toast.js::showToast()`, which renders
fixed-position notifications in `#toast-stack`, top-right but below page chrome.
The offset is intentional: toasts must never cover the Chat composer or primary
page actions. Toasts must not be inserted into page content or headers, because
that shifts the interface while the person is reading or clicking. Use reserved
inline status rows only when the status belongs to a specific control group and
that row is always present (for example marketplace search status). Do not
create page-prepended banners or local wrapper aliases such as `showBanner` for
short-lived events such as review started, install queued, or grant saved.

### Accent colors

| Role | Token | Value | Usage |
|------|-------|-------|-------|
| Primary | `--accent` | `#c93545` | Primary buttons, active nav, task identity, borders |
| Hover | `--accent-hover` | `#d4485a` | Primary-button hover |
| Light | `--accent-light` | `#f07a86` | Agent sender name, "Working" phase, keywords |
| Chip text | `--accent-chip-text` | `#f0a3ab` | Text on accent-tinted chips |
| Alpha ladder | `--accent-04` … `--accent-55` | `rgba(201,53,69,α)` | tinted fills and borders |
| Focus | `--focus-accent-border` / `--focus-accent-ring` | derived from `--accent-hover-rgb` | focus border + ring |

Use these for new features. Do not introduce additional red/crimson shades, and
do not write a raw `rgba(232, 93, 111, …)` / `rgba(201, 53, 69, …)` literal — the
hue lives once, in `--accent-hover-rgb` and the alpha ladder.

### Border radius scale

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-xs` | `3px` | Micro accents (progress bars) |
| `--radius-4` | `4px` | Inline code, tightest chips |
| `--radius-5` | `5px` | Micro selects, dense controls |
| `--radius-6` | `6px` | Context chips, activity rows |
| `--radius-7` | `7px` | Ghost buttons, brand mark, diff file rows |
| `--radius-sm` | `8px` | Nav rows, small controls, filter chips |
| `--radius-9` | `9px` | Send button, mid controls |
| `--radius-md` | `10px` | Chips, log-counter pills, page-fade rules |
| `--radius` | `12px` | Inputs, composer box, inner cards |
| `--radius-lg` | `16px` | Live cards, large cards |
| `--radius-xl` | `20px` | Logo images, large media |
| *(no token)* | `18px` | Section cards (settings, form panels) |
| *(no token)* | `24px` | Modal/wizard shells |

Use CSS variables where possible. Do not introduce new hardcoded radius values.
When a new radius value is needed, add it to `:root` in `web/style.css` first.
`999px` (pill) stays a literal — it is a shape, not a step on the scale.

### Interactive states

```css
hover:  border-color +1 step + background rgba(255,255,255,0.06)
active: background var(--accent-12) + color var(--text-primary)
focus:  border-color var(--focus-accent-border)
        + box-shadow 0 0 0 3px var(--accent-10)
```

Flat surfaces do not scale on hover; they change border/background. A hover
`transform: scale(...)` on a card or row is legacy and should not spread.

### Button conventions

All normal application buttons use the shared `.btn` base class plus exactly
one semantic variant:

| Variant | Purpose |
|---------|---------|
| `.btn-primary` | Primary action in the current surface: enable, install, update, start |
| `.btn-secondary` | Neutral secondary action next to a primary action: reload, cancel, install runtime |
| `.btn-default` | Low-emphasis utility action: refresh, details, open related view |
| `.btn-ghost` | Very quiet action on an already-strong surface |
| `.btn-save` | Persist settings or budget changes |
| `.btn-danger` | Destructive or emergency action |

Size modifiers are `.btn-xs` and `.btn-sm`; omit a size modifier for the
default medium size. Do not combine semantic variants (for
example, `.btn-default.btn-primary` is invalid), and do not invent one-off
button schemes in feature modules. Onboarding and modal buttons use the same
`.btn` variants as the main SPA.

Buttons are horizontally centered by default. If a control intentionally uses a
menu-row layout, use a named menu-item class (for example `.skills-menu-item`)
rather than overloading `.btn`.

### "Working" phase color

Use **crimson** — `var(--accent-light)` / `rgba(248, 130, 140, ...)` — for
active/working states everywhere, not blue. The Logs page phase badges match the
Chat live card colors, and the pulsing "Working" pill uses the same hue over an
`--accent-12` fill.

### No inline styles in JS

JS modules that generate HTML must use CSS class names, not `style=""` attributes.
This is enforced by reviewer policy — `.style.*` assignments on DOM elements (e.g.
`element.style.display`, `element.style.color`) will produce a REVIEW_BLOCKED finding.
**Accepted exception — dynamic CSS custom properties.** Setting a CSS variable for a
genuinely DYNAMIC value (`root.style.setProperty('--sidebar-width', w + 'px')` for a
live drag) is the idiomatic CSS-variable theming API, not a static inline style — it
feeds a stylesheet rule rather than hard-coding a visual property on the element, and
routing it through a managed `<style>` rule re-parsed each frame would be strictly
worse. CSS-variable mutation via `setProperty('--x', …)` is therefore allowed; static
visual properties (`display`/`color`/`width`/…) remain blocked. (v6.34.0, CW10)
The sidebar budget bar is the second sanctioned instance: `app.js` writes
`setProperty('--budget-fill', pct)` and `.nav-budget-bar-fill` consumes it as
`width: var(--budget-fill, 0%)`. Progress/meter fills follow that shape — never a
`.style.width` assignment.
Existing classes (`.stat-card`, `.page-header`, `.app-page-*`, `.app-tab-*`, `.about-*`, `.costs-*`) cover common layouts.
For new top-level pages, prefer `web/modules/page_header.js` over bespoke header/tab markup.
Add new classes to `web/style.css` when needed.
Before staging any `web/modules/*.js` file: `grep -n "\.style\." web/modules/*.js`
and fix any hits.
Legacy inline assignments that already existed before a scoped change are tracked
debt, not an automatic release blocker, when the diff does not add or worsen that
style usage. Prefer paying them down opportunistically instead of expanding the
scope of unrelated UI work.

### Declarative widget UI

Extension widgets should prefer host-owned declarative render schemas.
`web/modules/widgets.js` is the single host for `register_ui_tab`
declarations: `iframe` remains sandboxed with no relaxed tokens, and
`kind: "declarative"` / `schema_version: 1` covers forms, actions, markdown,
JSON, key/value summaries, tables, progress, files, galleries,
image/audio/video media, map/calendar/kanban, and the additive `group`,
`metric`, and `callout` composition components. New common widget capabilities
should extend that declarative schema and its tests, not introduce arbitrary
skill HTML, CSS, JavaScript, chart options, or cross-widget bindings.

v5.7.0 adds one deliberate exception for rare custom UI: `kind: "module"`
loads reviewed skill-provided `widget.js` into a sandboxed `srcdoc` iframe
(`sandbox="allow-scripts"`, **no** `allow-same-origin`). The parent host
fetches the reviewed JS from `/api/extensions/<skill>/module/<entry>` and
injects a constrained `fetch` bridge that only proxies
`/api/extensions/<skill>/...` routes. This is not same-origin SPA execution;
the module cannot access app cookies or `localStorage`.

Rules for widget changes:

- `group.components` and `tabs[].components` may contain interactive
  components. Give every mounted component an explicit `id` or let the host use
  its stable tree path; never key lifecycle state by a top-level array index.
  `subscription.render` is transitively passive: forms, actions, pollers,
  streams, subscriptions, and mutating kanban remain forbidden anywhere below
  it. One widget-level disposer owns timers, streams, abort controllers, charts,
  and snapshots, and inactive tabs do not restart lifecycle work.
- Escape by HTML context: use `escapeHtmlText()` for text-node content and
  markdown fallbacks, `escapeHtmlAttr()` for interpolated attribute values
  (`data-*`, `src`, `alt`, `title`, `href`, `value`) and mixed template
  snippets, and DOMPurify only for markdown blocks.
- Media sources must be extension routes under `/api/extensions/<skill>/...`
  or explicitly safe `data:` URLs for image/audio/video MIME types.
- Long-running user actions (image/music/research generation) must use the
  declarative async job contract: start route returns `job_id`, status route
  returns `queued|running|done|error`, and the widget host resumes polling by
  `job_id` after tab switches. Do not implement long generation as a single
  foreground HTTP request that can be lost when the widget remounts.
- Download controls must use the host download helper (`data-widget-download-url`
  / desktop bridge / fetch-blob fallback). Raw in-app navigation links are not
  acceptable for downloads because desktop WebView may replace the Ouroboros UI
  with the media file.
- Forms and Settings reuse the safe field renderer/value collector in
  `web/modules/ui_helpers.js`; Settings keeps its narrow route/component
  contract. Password values never persist across renders, duplicate submit is
  blocked, and busy/error cleanup must restore the control state.
- Charts preserve unknown/non-finite values as `null`, keep `spanGaps=false`,
  expose an ARIA label and an expandable semantic table built from the same
  data, and fall back to that table when Chart.js is unavailable. Kanban drag
  and native `Move to` use the same route and `{card_id, column_id}` payload.
- Do not load arbitrary JS modules from skill directories into the SPA origin.
  `kind: "module"` is allowed only through the sandboxed iframe + parent fetch
  bridge above, and must be covered by the `widget_module_safety` review item.
- Add/update `tests/test_widgets_ui_static.py` for every new component kind or
  media policy.

---

## MCP Client Integration

The base runtime is an optional client for trusted HTTP/SSE and local stdio MCP
servers; it is not an MCP server. `ouroboros/mcp_client.py` owns server parsing,
transport-specific validation, auth masking, provider-safe tool names,
discovery, timeout, and result normalization. Settings carry only `MCP_ENABLED`,
`MCP_TOOL_TIMEOUT_SEC`, and structured `MCP_SERVERS`; tokens never appear in
status responses.

MCP descriptions/results are untrusted data, not policy. Enabled tools join the
initial capability envelope, still pass runtime safety, and remain unavailable
in repair/heal contexts. Discovery failure becomes a visible capability
omission. Stdio accepts one executable command and an exact string argument
list, uses no shell, custom environment, or custom working directory, and
relies on the SDK context for teardown. Resources, prompts, and MCP server
behavior remain separate architecture changes.

## Gateway Boundary Pattern

Browser-facing backend work enters through `ouroboros/gateway/`.
`gateway/contracts.py` owns frozen envelopes/indexes and `gateway/router.py`
owns route mounting; domain handlers remain thin. Frontend calls go through
`web/modules/api_client.js`, with `api_types.js` and parity/smoke tests mirroring
the contract. Outbound provider/harness adapters belong in
`ouroboros/gateways/`. Do not require a class when established function owners
already preserve the boundary.

## Build & CI

### Pytest marker lanes

Default local pytest excludes costly or environment-dependent lanes:
`integration`, `browser`, `ui_browser`, `ui_browser_docker`,
`portable_detail`, and `skill_smoke`. CI opts into them explicitly:

- `integration` runs real provider checks, including Cloud.ru when
  `CLOUDRU_FOUNDATION_MODELS_API_KEY` is configured and GigaChat when
  `GIGACHAT_CREDENTIALS` is configured.
- `browser` launches real Playwright Chromium/WebKit for agent browser tools.
- `ui_browser` launches the host-side web UI under Playwright.
- `ui_browser_docker` talks to an `ouroboros-web:test` container and must
  skip cleanly when Docker is unavailable locally.
- `portable_detail` covers build/portable artifact invariants and also runs
  inside Docker in the manual/tag CI tier.
- `skill_smoke` installs the nine pinned official OuroborosHub skills
  (list in `tests/test_skill_smoke_official.py`) from the LIVE catalog and
  validates payload/sha/provenance, manifest contract, offline
  `skill_preflight`, real pip isolated deps, and keyless command probes. It
  runs as the dedicated 3-OS `skill-smoke` CI job (stable promote / manual /
  `v*` tags) in serial pytest invocations with real network + real pip:
  red means investigate (our runtime or the published catalog broke) — there
  is deliberately no fallback-skip. Its tests must NOT carry the `serial`
  marker or join `_SERIAL_TEST_FILES`: the `and not skill_smoke` markexprs
  in quick/full-test are the barrier that keeps the lane out of those
  passes, and the no-serial rule keeps each test's lane assignment single
  and unambiguous (defense-in-depth on top of that barrier).
  The lane's Tier 6 (`test_review_grants_and_enable`) additionally exercises
  the production install→review→auto-grant flow plus enable-persistence
  prerequisites for a 4-skill subset through the real gateway wrapper:
  Ouroboros's own skill review on ONE cheap stochastic reviewer slot
  (`google/gemini-3.5-flash`, low effort, `blocking` enforcement — pinned by
  the test's env; production reviewer defaults stay untouched), with
  auto-grant inside `review_skill`, then post-review dependency reconcile,
  enabled persistence (`save_enabled` + the toggle-gate facts — deliberately
  NOT the lifecycle toggle with `reconcile_extension`, which is server
  runtime and would execute downloaded plugin code in the secret-bearing
  process), and `skill_readiness_for_execution`. The
  CI job runs Tier 6 as a SEPARATE pytest step (fresh process) that alone
  carries `OPENROUTER_API_KEY`, ORDERED FIRST — the other tiers import
  downloaded plugin code in-process and must never share a process with the
  secret, and running the secret step first means the runner has never
  executed payload code while the secret was present — and only on the
  ubuntu shard (an LLM verdict is OS-independent). Paid step (~$1.2/run,
  ~$2.4 with the single fresh verdict retry); a missing key is a hard red,
  not a skip.

When adding a new opt-in lane, register the marker in `pyproject.toml`, add
a collect-only zero-test guard in CI, and keep the default local addopts
token-safe and Docker-safe.

### Parallel CI and the `serial` marker

CI runs the full default suite **in parallel** — `pytest -m "not serial" -n auto --dist loadscope
--max-worker-restart=0` (~5× faster than serial) — followed by a short serial pass for `-m serial`
(`.github/workflows/ci.yml`, jobs `quick-test` / `full-test`). Two rules keep new tests from breaking
that:

- **Mark real-process / real-port / process-global tests `@pytest.mark.serial`.** A test that spawns
  a real OS process, binds a real port, or mutates a module-level registry is not parallel-safe:
  under `-n` it flakes on kill/reap or port-reclaim timing, or it crashes its worker — which (with
  `--max-worker-restart=0`) fails that worker's WHOLE co-located batch and shows up as spurious
  failures in unrelated files. Mark such a test `@pytest.mark.serial` (or add its file to
  `_SERIAL_TEST_FILES` in `tests/conftest.py`) so it runs in the serial pass instead.
- **Keep every other test parallel-safe** so it stays in the fast pass: use `tmp_path` (never a fixed
  path like `/tmp/foo.pid`); use `monkeypatch.setenv` / `monkeypatch.setattr` (never a bare
  `os.environ[...] = ...`, which leaks to other tests on the same worker); never assume execution
  order; and if you must mutate a module global, reset it around the test (pattern:
  `tests/conftest.py::_isolate_workspace_executor_globals`).

### The commit gate mirrors the CI split

`ouroboros/preflight_runner.py::run_hermetic_pytest` runs the same two logical
passes as CI in one disposable checkout and scrubbed temporary data root:

1. parallel `not serial` with xdist, loadscope distribution, no worker restart,
   and the configured per-test timeout;
2. flag-free `serial` for tests whose real process/port/global-state behavior
   cannot be parallel-safe.

Both passes share one total timeout. `LANE_EXCLUSION_EXPR` and
`PARALLEL_PASS_FLAGS` are executable SSOTs pinned against both CI jobs. The
selected interpreter must provide `pytest-xdist` and `pytest-timeout`; plugin
presence is probed outside candidate control, forced on for the parallel pass,
and proven by host-owned worker markers. `OUROBOROS_PREFLIGHT_SERIAL=1` is the
explicit temporary rollback lever, never a silent fallback. Evidence runs set
`OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS=1`.

The candidate environment cannot weaken the pass with inherited `PYTEST_*`
values, delete an inherited suite and earn green, or replace required plugins
with fake command-line options. Pre-commit checks bind to `HEAD`; post-commit
checks also inspect `HEAD~1` so deletion of the suite cannot hide after the
commit exists. Exit status owns the verdict; rendered/truncated diagnostics do
not.

A red post-commit gate preserves the local commit for forensics but blocks push.
Inside a managed update it also blocks boot promotion and routes through
rollback; an incomplete rollback leaves `gate_blocked` so boot retries recovery
instead of promoting the rejected merge. Review-binding and tag-binding
mismatches use the same failure route.

Process containment is unconditional, including a green pass. Windows uses a
kill-on-close Job Object. POSIX uses an environment membership token plus a
process-group enumeration backstop; signaling is best effort, while a live or
unreadable member fails closed. POSIX promises detection and an honest block,
not guaranteed teardown of an arbitrary detached process. A crashed xdist
worker, a per-test timeout that killed a worker, a missing plugin, containment
failure, and ordinary test failure keep distinct diagnostics and remediation.

Exit 5 is green for an individually empty pass but the overall suite may not be
empty. The first red pass returns immediately. Pass 2 receives only the
remaining total budget. Mark process/port/global-state tests `serial`; make a
merely slow test faster or split it, because the serial lane has no per-test
timeout.

### GitHub Actions: secrets in step-level `if:` conditions

GitHub Actions rejects `secrets.*` inside step-level `if:` expressions, and a
step's own `env:` block is not visible to that same step's `if:`. Derive a
non-secret boolean in the job-level `env:` block, gate steps with that boolean,
and map the actual credentials only inside the first-party steps that need them.

```yaml
jobs:
  build:
    runs-on: macos-latest
    env:
      HAS_APPLE_SIGNING: ${{ secrets.BUILD_CERTIFICATE_BASE64 != '' && secrets.P12_PASSWORD != '' && 'true' || 'false' }}
    steps:
      - name: Import Apple signing certificate
        if: env.HAS_APPLE_SIGNING == 'true'
        env:
          BUILD_CERTIFICATE_BASE64: ${{ secrets.BUILD_CERTIFICATE_BASE64 }}
          P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
        run: |
          echo "${BUILD_CERTIFICATE_BASE64}" | base64 -d > cert.p12
          security import cert.p12 -P "${P12_PASSWORD}" ...
      - name: Cleanup keychain
        if: always() && env.HAS_APPLE_SIGNING == 'true'
        run: security delete-keychain ...
```

```yaml
# ❌ WRONG — workflow fails to parse
- name: Bad
  if: secrets.BUILD_CERTIFICATE_BASE64 != ''   # parse error
  env:                                          # not visible to this step's if:
    P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
```

`tests/test_build_scripts.py::TestMacOSSigning::test_ci_uses_env_context_for_condition`
enforces this across every workflow `if:` block.

### Apple signing & notarization (macOS Build job)

Prerelease artifacts may intentionally be unsigned and must report that state;
stable publication continues to apply the configured signing and notarization
policy rather than implying credentials or success that were absent.

When Apple signing secrets are configured, the macOS shard imports the Developer
ID certificate into a temporary keychain and `build.sh` signs the `.app` and
`.dmg` via `SIGN_IDENTITY`. Only a non-secret `HAS_APPLE_SIGNING` gate is
job-wide. Certificate and keychain values exist only in the import step, while
Apple ID notarization values exist only in the first-party build step. Later
SBOM and attestation steps inherit none of them. If `APPLE_ID` and
`APPLE_APP_SPECIFIC_PASSWORD` are present, notarization runs; otherwise the DMG
ships signed but not notarized. Notary/stapler failures are soft warnings,
recorded through `NOTARIZE_OUTCOME`, so transient Apple issues do not silently
drop the macOS artifact. Cleanup uses `always()` plus macOS/env guards, and
signing material never persists across runs.

### Release proof capsule

The tagged build binds public release assets to their source and verification
record. Each platform shard locates the final DMG, tarball, or ZIP after all
packaging steps, then performs a smoke test against that final archive. The
smoke checks require the embedded repository bundle, run the packaged CLI with
`--help` in an isolated home directory, then use the embedded Claudexor seed and
Node from that extracted final artifact to perform install, extraction, exact
identity probe, owned-daemon handshake, one fake task, and an identity-bound
graceful stop of the serving closure. The separate
Claudexor platform gate repeats that fixture path on ordinary branch changes and
adds the explicit-key live compatibility matrix; neither path installs a
floating Claudexor npm package. The macOS check also requires the
`Applications -> /Applications` drag target, the separate `Install CLI.command`
payload, and an arm64 app executable.

Each shard also generates a CycloneDX SBOM from the payload extracted from the
final archive. The macOS smoke proves the Applications link, then removes only
that link from the SBOM staging copy so Syft cannot follow it into the runner's
host `/Applications`; the app and CLI launcher remain in the scan. The workflow
downloads a fixed Syft release asset and checks its platform-specific SHA-256
before execution. GitHub artifact attestations bind both build provenance and
the SBOM to the final archive digest. The release job downloads the three
archives and their proof files, checks the exact platform allowlist,
recalculates every digest, and verifies both predicates against the exact source
SHA, tag ref, repository, and signer workflow before it writes:

- `SHA256SUMS` for archives, SBOMs, and smoke receipts;
- `release-evidence.json` with tag, commit, workflow, checks, and artifact
  bindings;
- release notes from the matching README Version History row.

Publication uses a draft release. A per-tag concurrency group serializes release
jobs, and a fail-closed preflight allows only an absent release or an existing
draft; a published release is never overwritten by a rerun. The workflow
uploads only the explicit allowlist and compares GitHub's stored sizes and
SHA-256 digests with the local files. Immediately before draft creation and
again before publication, it requires the remote tag to exist as an annotated
tag whose peeled commit is the workflow event SHA. It publishes only after all
of those checks pass. A release from an
older workflow may receive a clearly labelled post-publication checksum
inventory, but it must never claim build-time provenance, an SBOM, or packaged
smoke evidence that the original build did not create.
