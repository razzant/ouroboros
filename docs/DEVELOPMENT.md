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
- User-facing product UI strings (web UI labels, toasts, chat/receipt copy) are English as well.
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
  contextual tools default through `ToolContext.active_repo_dir()`. Admission
  rejects overlap with the system repo/data and records a read-only preflight.
- Project focus changes the default target, not the ordinary top-level tool
  surface. Generic VCS selects active/system explicitly; advisory, reviewed
  commit, rollback, promotion, restart, and runtime control keep their intrinsic
  system-repository contracts and existing gates. Workspace finalization still
  returns durable patch artifacts. `executor_ref` selects a process backend, not
  an implicit sandbox.
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

Each route's wire projection is pinned by the golden fixtures in
`tests/fixtures/llm_golden/`: per route they record the resolved target, the
client (base url, header set, retry policy, proxy trust), every request payload
with its canonical digest, the physical-attempt ledger rows, and the returned
`(message, usage)`. They replay against recording fakes — never the network, and
never a real credential — so a changed payload byte, header, model-slot
resolution or fallback order fails `tests/test_llm_provider_golden.py` instead of
reaching a provider. A deliberate route change re-records them with
`python tests/test_llm_provider_golden.py --write` and explains every diff.

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

- Python modules everywhere (including `tests/` and `devtools/`) and first-party
  `web/**/*.js` modules (including `web/tests/`) target roughly 1000 lines. The
  deterministic hard gate is 1600 lines for exact repo-relative paths not listed
  in `ouroboros/size_ratchet_manifest.py::GIANT_PATHS`; stale or newly oversized
  entries fail. Vendored/minified assets are excluded. The same production
  iterator drives smoke, health, census, and the 200,000-byte ratchet. Sources
  decode as strict UTF-8 and normalize line endings to canonical POSIX LF before
  line and UTF-8-byte counts, so checkout policy cannot change the inventory.
- The second module layer is the optional `MODULE_DEBT_1500` manifest field:
  absent until activated (absence means "not activated"; presence, even as
  an empty tuple, means active). Once active, every exact path above 1500 lines
  must be listed there, so new/non-debt paths are capped at 1500 while every
  path above 1600 additionally stays in legacy `GIANT_PATHS`; both layers are
  enforced independently for the live tree, the staged index, bootstrap, and
  every audited first-parent transition. Activation happens exactly once via
  `scripts/regenerate_size_ratchet.py --activate-1500-layer`; its only admission
  authority is the activation commit's exact first-parent >1500 inventory, which
  permits same-commit paydown and rejects same-commit self-authorization of a
  fresh 1501-line path. Afterwards the set is shrink-only and irrevocable:
  ordinary regeneration and `--check` preserve and enforce it without the flag,
  and additions, retired-path re-entry, or deactivation fail validation.
- The exact-current 1001-1500-line band lives in `BAND_PATHS`. A new or
  re-entered path requires a nonblank rationale. `BYTE_DEBT` stores exact counts
  above 200,000 UTF-8 bytes and is shrink-only; regenerate both with
  `scripts/regenerate_size_ratchet.py`.
- Every non-grandfathered Python function or method fails the deterministic gate
  above 300 lines; exceptions live in
  exact `(repo-relative path, lexical qualname)` keys in
  `ouroboros/size_ratchet_manifest.py::FUNCTION_DEBT`. The set is shrink-only,
  with one non-growing move allowed: a debt function whose exact qualname leaves
  one path and appears at exactly one other path in the same transition keeps
  its row (an extraction may carry it into a leaf); a fresh oversized function,
  a swap onto another qualname, or an ambiguous many-to-one move is refused.
  Methods above 150 lines are a decomposition signal. JavaScript currently has
  only the module line-count gate.
- Runtime Python function/method count is checked against
  `ouroboros/review.py::MAX_TOTAL_FUNCTIONS`; the function iterator preserves
  the runtime scope it always had (tests/devtools excluded) while module gates include
  those trees.
- More than eight parameters is a decomposition signal applied by BIBLE and
  reviewer checklist 2(c), not a deterministic size-test gate. Existing
  baseline debt is not retroactively a failing tree. Any advisory ratchet must
  publish its AST counting scope and bind its baseline to the final SHA.
- The committed first-parent history is audited with the same exact inventory
  as the live tree: every commit's manifest must match its own tree, so a
  giant that appears and disappears inside history is still caught. The walk
  reuses one cache keyed by Git blob id — content-addressed, therefore a hit is
  the same bytes by construction — so a multi-commit audit costs only the blobs
  that changed rather than a full census per commit. Sampling commits instead
  would be cheaper and wrong: it would retire that transient-giant guard.
- Splitting a module records every moved or retired symbol in `MIGRATION_v7.md`,
  one row per identity, and the row's semantic-delta note is a claim under test.
  `tests/test_v7_verbatim_moves.py` compares the declaration at the old identity
  in the ledger's recorded merge base against the declaration at the new identity
  in the working tree, so a note that calls its move verbatim must hold
  byte-for-byte — modulo the one indentation level a method legitimately loses
  when it becomes a module-level function. Text that changes after a move (a
  widened signature, a typed return, a retargeted call site, an edited docstring)
  belongs in the note, which names what changed and why; leaving the word on a
  row whose text has since moved on is the failure this gate exists to catch.
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

Plan review is the one flow whose governance pack is tiered by the self-modification path fact
(see the table below). Before any work exists, the reviewer's subject is the INTENTION, not the
engineering handbook; what it does not receive resident it may request as typed evidence. This is
a tiering, not an omission: every absence is a named pointer or a `need_evidence` finding.

Concrete requirements:

| Flow | BIBLE.md | ARCHITECTURE.md | DEVELOPMENT.md |
|------|----------|-----------------|----------------|
| Main task context (`context.py`) | full tier-0 | full in Max for every task class; lossless navigation map in Low | mode-independent: full when the active binding targets Ouroboros's system repo, including evolution/self-body work and a project-room turn without an external binding; visible on-demand pointer for a bound external workspace, subagent, or API/CLI/scheduled external surface. `workspace="none"` and explicit self-body overrides retain full Development. |
| Triad review (`tools/review.py`) | ✅ via preamble | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` |
| ↳ Anti-thrashing (v4.35.1) | — | — | Open obligations loaded from `review_state` via `load_state(drive_root)` + `make_repo_key(repo_dir)`, injected unconditionally into `_build_review_history_section` prompt context. Same mechanism in `scope_review.py::_build_scope_prompt` (best-effort when `drive_root` available). |
| Background consciousness (`consciousness.py`) | ✅ full | ✅ full (max) / navigation map (low) | — (not yet required) |
| Advisory pre-review (`tools/claude_advisory_review.py`) | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` |
| Scope review (`tools/scope_review.py`) | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting |
| Plan review (`tools/plan_review.py`) | full for a SELF-MODIFICATION plan (structural path fact: a declared target resolves under the system repo); otherwise a heading-derived navigation map of BIBLE.md generated at runtime (never a copy) | inline, in full, for a self-modification plan; otherwise the lossless navigation map + a resolvable pointer (W3) | named on-demand pointer; a reviewer that needs it returns `need_evidence` and the host attaches it on the next cycle |
| Deep self-review (`deep_self_review.py`) | full canonical doc + Atlas accounting | full (max) / navigation map (low) + Atlas accounting | full canonical doc + Atlas accounting |

Plan review keeps the reviewed SPEC, the task objective, the agent-declared evidence, and
reviewer-slot framing as first-class context. Governance packs are tiered by ONE structural
fact — whether the plan's declared targets resolve under the Ouroboros system repository
(self-modification) — never by prose and never by a plan-kind taxonomy. A self-modification plan
carries BIBLE.md in full and ARCHITECTURE.md inline; every other plan carries the constitutional
excerpt (the heading-derived navigation map of BIBLE.md, `context_layout.generate_doc_nav_map`,
never a copy), the ARCHITECTURE navigation map, and named on-demand pointers. A reviewer that needs
more returns a typed `need_evidence` finding naming exactly what is missing, and the host
attaches it on the next cycle: nothing is silently omitted (P1). A host-attached locator goes
through exactly the same allowed-root, deny-path, sensitivity and redaction policy as declared
evidence (a refusal is a named `[reviewer-requested]` omission row), and it enters the manifest
hash — so the agent's next envelope is a new fingerprint carrying the evidence, never an
idempotent replay. DEVELOPMENT.md is not resident in a plan-review packet; it is one such
request away. Delivery form: an `api_chat` row receives the constitutional pack inline; a
retrieving (`agent_session`) row receives the executor's compact form of the same pack —
BIBLE.md and ARCHITECTURE.md as mandatory full reads at their resolvable locators
(`governance_by_retrieval`), the only evidence locators a session may read raw. Bounds: the
per-task request memory (`need_evidence_seen`) holds at most `MAX_NEED_EVIDENCE_MEMORY` locators
(a request past it is demoted, disclosed `need_evidence_memory_full`), each at most
`MAX_ITEM_CHARS`; the host honours at most `MAX_LIST_ITEMS` of them per wave (the rest are named
`reviewer_request_cap` omissions, still tagged as reviewer requests).

The skill-payload exemption is unchanged: an exact path inside an installed skill payload under
the canonical data root is data-plane work and does not make a plan a self-modification, even
when the active workspace is the system repository itself.

Paid review cycles per task are bounded by the owner's shared `OUROBOROS_REVIEW_MAX_CYCLES`
(default 2, `unlimited` available). Under blocking enforcement an exhausted cap holds
implementation and escalates with the typed `review_cycles_exhausted` reason; under advisory the
agent may proceed with the wave open under the host's loud disclosure. An idempotent replay of
the same fingerprint — a recorded DEGRADED wave included — consumes no cycle; a wave pays iff
at least one reviewer slot was physically dispatched, so a dispatched DEGRADED panel pays its
cycle while a nothing-dispatched wave of typed $0 skip rows stays unpaid.

Planning has two distinct roots. Governance documents are always loaded from
the system repository; declared targets and evidence locators resolve against
`active_repo_dir_for(ctx)`. Exact non-native installed-skill payload paths are the one data-plane exception for
CLASSIFICATION: they never make a plan a self-modification, even when the active workspace is
the system repository. They are not attachable as evidence: the evidence resolver allows only the
active workspace and the system repository and refuses the runtime data plane outright, so a
payload locator comes back as a named `denied_path` omission. Any declared path escaping the active subject, a
workspace/subject mismatch, or an unavailable root must fail loudly with a named
omission. Do not fall back to reviewing the Ouroboros repo for an external plan.
Evidence the host cannot attach is a typed omission row, never a silent gap, and
a reviewer that needs more asks for it with `need_evidence`.

The SPEC must state the goal, acceptance claims, invariants, in-scope and
non-goals, the load-bearing decisions with their rejected alternatives, and what
is consciously deferred. Plan review publishes exactly `GREEN`, `REVIEW_REQUIRED`,
or `REVISE_PLAN`. `REVIEW_REQUIRED` findings are inputs: the main agent may
accept, reject, or defer any/all of them. Closure happens without a second LLM
call through a separate `plan_task` call containing `review_disposition` only —
`{review_fingerprint, items: [{finding_id, decision, rationale}]}` — covering
every finding exactly once with an evidence-based rationale; duplicate or
contradictory entries for one finding are refused. Never replay the plan
envelope with the disposition. Mixed calls and vacuous disposition-only calls fail before a new
attempt is recorded; exact replay is idempotent. Blocking `REVISE_PLAN` requires
changed plan text/fingerprint and another panel, while advisory may proceed only
under loud host disclosure and the main agent's rationale. Unknown, stale,
duplicate, contradictory, or incomplete dispositions fail closed. Reviewers are
findings-only — they judge the intention and never author a competing plan — and
a blocking finding must name the spec id it breaks; there is never a required
number of findings.

Force-plan is an LLM-first pre-implementation obligation on the admitted managed
root, not a mechanical permission check around implementation tools. The existing
`plan_review_state` owns durable review authority and
`config.get_review_enforcement()` owns blocking/advisory policy. Every submitted
envelope that reaches `plan_task` supersedes prior authority: invalid plan/goal/scope
input stores a domain-separated open attempt, while a valid envelope stores its
canonical fingerprint before repository/path validation. A newer attempt therefore
cannot fall back to an older GREEN. The wave records the frozen SPEC, its hash, the evidence
manifest (attached hashes plus every omission) and the composed fingerprint before dispatch, so a
repeat call with the identical envelope replays that recorded wave for free instead of buying a
second panel, including after A→B→A. An unavailable reviewer never becomes a disposition-able
verdict; a DEGRADED wave (no parseable quorum) records OPEN with per-slot typed failure facts,
pays the cycle its dispatch cost, reaches the agent as an honest DEGRADED control outcome, and
replays free ONLY under all three conditions: an identical envelope, a NON-EMPTY recorded
structural lane-health epoch that a fresh pre-fan-out snapshot still matches, and an unchanged
reviewer roster (slot ids, targets, routes, pinned profiles and efforts — an effort change is a
roster change). An empty-epoch DEGRADED wave re-dispatches a PAID panel on the identical
envelope; so does a lane the snapshot proves healed or newly dead, or a changed roster; a
failed snapshot keeps the free replay — the next step (change the spec, wait, escalate, or
proceed where enforcement permits) is the agent's judgment, never a host-authored re-call
imperative. Structurally dead slots (dated window exhaustion with a future reset, or a typed
dead-pool code) are skipped before dispatch as $0 typed rows that stay in the quorum
denominator; unknown health dispatches. A wave whose typed rows prove the quorum structurally
unreachable carries `quorum_unreachable` + the earliest reset; under blocking the finalization
gate releases for an agent-chosen honest `blocked_with_evidence` terminal (review open,
implementation held), and a one-shot deferred follow-up can be registered through
`schedule_followup` on the existing supervisor scheduler. An open wave recorded under advisory
enforcement emits one typed owner-visible
`plan_review_advisory_open` event at record time. Blocking stays in
analysis and non-mutating preparation until closure or a real task-wide rail;
advisory may proceed by agent judgment with a host-owned disclosure, including
an explicitly rejected `REVISE_PLAN`. A planning
deadline skip records a typed rail attempt before returning so the reducer cannot
misread it as an absent `plan_task` call.
The short-lived Swarm router admits one new root and transfers the intent; it
never runs `plan_task`, steers an existing task, or publishes the work inline.

**Context mode (Low / Max).** `OUROBOROS_CONTEXT_MODE` controls the Architecture projection in the agent's own context: Max keeps `ARCHITECTURE.md` full for every task class, while Low supplies its lossless navigation map. `DEVELOPMENT.md` is mode-independent and follows the active repository binding. It is full when the task targets Ouroboros's system repository, including self-body and evolution work and a project room with no external binding; a bound external workspace, auto-provisioned external project tree, subagent, or API/CLI/scheduled external surface receives a visible on-demand pointer. Explicit structured overrides remain authoritative. Tier-0 identity and constitutional context stays full in every mode.
For ordinary Main calls, `context_fit.py` renders Max and Low from one immutable
captured core and measures the sealed transcript plus live schemas on one
labelled density basis. Owner Low has an elastic 200K total-context economy
target; the actual route window remains the physical capacity, so a target miss
is non-terminal and may send best effort after one useful pass. Predicted Max
pressure never swaps in Low documents. Only actual provider overflow may use a
task-local Low projection, followed by at most one same-route call whose final
context-bearing candidate is strictly smaller with the same response reserve.
This never changes owner mode or P3 commit/scope review.

### Invariant: Compaction must earn its rewrite

Context compaction is a deficit-requested materializer, not an independent
threshold, timer, route, or retry policy. It first performs pure selection over
completed atomic units: one assistant tool-call message plus all and only its
contiguous matching results. User turns are hard boundaries; malformed,
missing, delayed, duplicated, visually opaque, or corrupt-capsule units remain
byte-identical. No eligible positive reclaim means no checkpoint, summarizer
call, or transcript mutation.

For a non-empty selection, persist the exact actor-visible checkpoint before
calling the summarizer. Summary input covers complete stable hashed chunks with
gap-free offsets; it never uses head excerpts, long-string markers, or hidden
argument/result omissions. Only typed summarizer context overflow may split a
source recursively. Missing leaf coverage keeps that whole atomic unit raw,
while independently covered units may apply. A replacement publishes only
after transcript/unit binding, complete coverage, checkpoint provenance, and a
strictly smaller representation on the caller's ContextFit measurement basis
are all proved. The bounded image proxy and density must match the requesting
fit calculation; raw base64 byte count is not token reclaim. Capsules carry
host-only generation, source-hash, part, checkpoint, and CAS-ref metadata so a
later pass can recompact them without losing the original provenance union.

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
- A reference-doc **navigation map** (H2-H4 inclusive complete-subtree ranges,
  with parent rows overlapping descendants and full sections one `read_file` away) and a
  named on-demand pointer are visible, lossless representations — NOT silent
  truncation. The low context mode uses these; it never applies `[:N]` to a doc.
- String bounding goes through the SSOT `utils.truncate_review_artifact`, never a
  hand-rolled `text[:cap] + marker`. Besides the marker, that helper carries an
  anti-waste FLOOR: a cut saving fewer characters than its own omission note is pure
  damage, so below it the text passes through whole. A local re-implementation loses
  the floor and can return a value LONGER than the input it "shortened" (a `…[+N
  chars]` marker is 11 characters, so any overflow under that grew the field).
  The two bounded-string primitives serve different contracts:
  `truncate_review_artifact` produces DISPLAY previews of review artifacts (its
  anti-waste floor may return the text whole), while `truncate_within_limit`
  enforces a STRICT wire/prompt bound — the omission marker lands INSIDE the
  limit and the result never exceeds it.
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
assembly failure is not a clean verdict. An agent-session scope slot delivers by
retrieval: its verdict is authoritative once its window is sourced at ≥200K, and
"the host did not observe which files it read" is a provenance disclosure, never
a missing-authority finding.

The gate is one logical reviewer interaction per API slot. A same-route
transport or empty-response rail may make one bounded second physical send. A
hosted agent-session slot is one multistep execution; local extraction reuses
its collected transcript rather than launching another session. Actor transport,
parse status, semantic verdict, model and route, coverage, cost, and capability
delta remain distinct durable facts.

One shared owner knob bounds PAID review cycles across the gates:
`OUROBOROS_REVIEW_MAX_CYCLES` (SSOT `ouroboros/review_cycles.py`; a STRING —
a positive integer or `unlimited`; default `"2"`; Settings → Behavior → "Max
Review Cycles" 1 / 2 / 3 / 5 / ∞). Its per-gate meaning is documented in that
module and is literally: plan review — paid reviewer-panel cycles per task;
task acceptance — paid panel runs per task, `improvement passes = cycles − 1`
(the retired `OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES` is migrated into the shared key at
settings load — cycles = passes + 1 — and never binds at runtime); commit gate — consecutive review-blocks of a
BYTE-IDENTICAL staged diff before the identical-diff attempt cap refuses another
triad+scope run (changing the diff starts a fresh streak; a rebuttal lifts the
cap for that attempt only; the default moved from a hardcoded 3 to the shared 2). `unlimited` removes the
local count everywhere; deadline, budget and lifecycle rails still bind. A
malformed value fails closed to the bounded default and is logged once.

`docs/CHECKLISTS.md` is the only reviewer-question, severity, and output SSOT.
Architecture owns the dataflow; this section owns operator sequence. Finish all
edits, run focused tests, run the advisory when useful, then freeze and review
the exact candidate. Do not interleave edits with repeated review calls.

### External PR review is not commit authorization

The authoring agent freezes the final committed base-to-head range and gives it
to a separate agent context for read-only review. Any coding harness or provider
may supply that independent context; same-conversation self-review does not.
Unavailable review is recorded as `NOT_RUN`, never silently presented as clean.
`CONTRIBUTING.md` owns the public procedure and evidence fields.

`scripts/run_external_review.py --contributor` is an optional structured
producer for the same evidence. It preserves and freezes the machine's
configured `api_chat` and `agent_session` triad/scope rows, then binds each row
to its dispatched prompt receipt and observed response receipt. The shareable
packet records exact base/head/tree/diff hashes, route/model/profile facts,
terminal settlement and capability-delta facts, telemetry limitations, and
full redacted agent-session transcripts. Missing, tampered, drifted,
unprovable, or contradictory identity/terminal receipts make the packet
`INCOMPLETE`. Non-identity capability deltas remain explicit degradation
evidence and do not override the production actor-status/quorum result. The
review machinery is always the target base's own (owner decision, 2026-08-19):
unless the invoking checkout already is the target base, the lane materializes
that commit in a detached worktree and re-runs the review from it, so a proposal
is never trusted to review itself and no per-proposal trust classification
remains. The proposal stays the reviewed subject in the frozen checkout. The
guarantee is scoped: the wrapper deciding to hand off is itself read from the
invoking checkout, so run it from a trusted one. That trust root is the same as
before the change; it is now stated instead of assumed.

This evidence establishes readiness; it does not authorize commit, push, merge,
or publication. Maintainers choose the landing parent and release version,
preserve authorship, and run the normal final exact-candidate gate. Accordingly,
a pull request into `ouroboros` leaves `VERSION`, `pyproject.toml`, the editable
root version in `uv.lock`, `web/package.json`,
`web/modules/api_types.js::GATEWAY_CONTRACT_VERSION`, the README badge and
latest Version History row, the named direct-download links in README and both
install pages, and the Architecture header byte-identical to its target. At
integration,
`ouroboros/tools/release_sync.py::sync_release_metadata()` projects the chosen
version and `version_carrier_desyncs()` verifies those carriers; changelog prose
remains a deliberate maintainer edit. The same projection owns the seven public
installer filename templates and rewrites the named direct-download links in
README, the source install page, and its generated Pages copy. Those links use
the immutable exact tag (`/releases/download/v{VERSION}/...`), not
`/releases/latest/download/...`: prereleases are excluded from GitHub's latest
release and a versioned latest-link would therefore fail during an RC. Release
notes are generated from the same templates only after the seven proof-bound
assets have been assembled.

The integration branch may therefore name installers that are not published
yet. Public onboarding does not use that branch: the default README and legacy
GitHub Pages source are `main` and `main:/docs`. Stable promotion advances
`main` only after the release and all seven installers are public; if promotion
does not happen, users stay on the previous working release.

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
- [ ] Module stays near one context window (~1000 lines target; exact-path 1600 hard-gate debt is checked in, stale entries fail, and new/re-entered 1001-1500 paths carry a rationale; with the v7 `MODULE_DEBT_1500` layer active, non-debt paths are additionally capped at 1500 lines and the active set is shrink-only)
- [ ] No non-grandfathered Python function or method exceeds the 300-line hard gate (`FUNCTION_DEBT` exact `(path, qualname)` keys are the exception SSOT); methods above 150 lines trigger decomposition review
- [ ] Total Python function count stays under the current smoke hard gate (consult `ouroboros/review.py::MAX_TOTAL_FUNCTIONS` for the active value; bump with a comment if a feature requires more headroom)
- [ ] More than eight parameters is a decomposition signal; consider a typed context object, but do not claim a hard gate or mark existing baseline debt noncompliant
- [ ] No gratuitous abstract layers (Bible P7)

#### Structural Rules
- [ ] New Tool? `get_tools()` exports it using the shallow-frozen `ToolEntry` descriptor owned by `ouroboros/tools/tool_catalog.py` and re-exported by `registry.py`, while `ouroboros/tools/registry_core.py` owns ordered `ToolRegistry` orchestration and builtin invocation; `tool_resolution.py` owns public-argument/target preparation, `registry_guards.py` owns payload and access policy, `extension_dispatch.py` owns dynamic extension/MCP dispatch, and `registry.py` remains compatibility-only. Private tests and patches bind canonical owners; ordinary imports are not promoted into facade ABI. An explicit entry is added to `ouroboros/safety.py::TOOL_POLICY` (`POLICY_SKIP` for trusted built-ins, `POLICY_CHECK` for opaque or outward-facing ones), and the intended capability class is declared in `ouroboros/tool_capabilities.py` (`CORE_TOOL_NAMES`, local-readonly/acting child profiles, parallel/truncation sets as appropriate). First-party and task-scoped duplicate names fail closed with both registration origins. Extension/MCP collisions do not replace catalog entries or break installation/refresh: the dynamic projection is omitted with a loud log and visible capability omission. Ordinary top-level tasks share the registered built-in surface; add a tool to a child profile only when that narrower principal should receive it, and test schema plus execution behavior rather than mirroring names into another catalog. Without the policy entry the tool falls through to `DEFAULT_POLICY = POLICY_CHECK` and pays a light-model LLM call per invocation. **A tool that WRITES the repo working tree needs the GUARD surfaces too, not only the visibility ones:** add it to `tool_resolution._ROOT_ARG_REPO_WRITE_TOOLS` (the single set behind the acting-no-workspace fence, the protected-write gate and the acting root-enum narrowing) and make sure its target paths are canonicalized — via `_PATH_NORMALIZED_TOOLS` if it takes a top-level `path`, or via `canonical_repo_relative_path` + `tool_resolution._payload_write_paths` if its paths ride inside the payload. Visibility checks can all be green while these are missing, so tests must exercise the real guard chain, not only a mocked resolver.
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
- `run_command`/`run_script`/`start_service` may use cwd under `active_workspace`, explicit `system_repo`, task-scoped `task_drive`, task-scoped `artifact_store`, and external `user_files` where the active profile permits it. Omitted cwd consistently selects `active_workspace`; a light direct task that needs writable scratch must therefore select `task_drive` explicitly. Long-running services in light must use an explicit external/task/artifact cwd. Declared service `outputs` are copied into the task artifact store when the service stops.
- `run_script` temporary files are created under the active workspace when the task is workspace/executor-backed, then removed after execution. Do not run workspace scripts from the system repo temp path; relative imports, generated files, and toolchain discovery must observe the same cwd the user requested.
- Declared process outputs may be files or directories. Directory outputs are copied to the canonical artifact store as a bounded manifest plus zip archive; hidden/control/credential-shaped files, excessive file counts, and excessive byte sizes fail closed instead of leaking through artifact registration.
- In external workspace mode, light-mode self-repo dirty checks snapshot the system repo, not the active workspace. Task-local git operations inside the external workspace are allowed when the task requires them; Ouroboros repo/data paths remain structurally protected, and workspace patch artifacts are captured against the preflight git base.
- Project-room promotion with no working folder and no `workspace="none"` opt-out idempotently provisions a standalone git repo through `ensure_project_workspace`, then runs the ordinary workspace admission checks. Never provision over a non-empty broken binding or an unreadable registry; those cases fail loudly. Binding affects tool profile, memory, lease, and preflight, not the Max-mode Architecture projection.
- Keep policy denials separate from execution failures: `user_files_path_blocked`, `cwd_blocked`, and `artifact_output_undeclared` are non-failure outcomes, while failure to register an explicitly declared output remains `artifact_output_error`.
- The DEFAULT (non-workspace) shell lane carries the SAME target-aware git policy in every runtime mode including light (Q4=A sandbox unwind): mutating git is blocked only when it targets the Ouroboros runtime (system repo / any data drive — bidirectional, casefold, symlink-resolved containment; `commit_reviewed` is the remedy for self-repo changes), read-only git works everywhere including at the system repo, `allowed_resources.network=false` still fences network git subcommands, and acting `self_worktree` children keep the strict no-commit policy. `git init`/`commit`/`push` in `~/projects`, `/tmp`, an attached project folder, or a host-minted coop tree is legitimate task work, not a violation.
- `claude_code_edit` is RETIRED (D10, owner-approved migration, phase 6.4): the SDK edit gateway's job moved to the delegated coding path — a mutating subagent (`schedule_subagent`) whose nanny drives the session with `delegate_start`/`delegate_wait`/`delegate_answer`/`delegate_cancel`, on the owner's subscription when a harness route is configured. The D10 migration shipped INCOMPLETE for one supported target class — the old gateway could edit an exact non-Git skill payload directly, while the successor knew only Git workspaces — and that class was RESTORED (owner option A, 2026-08-14): a top-level task selects the exact payload with `delegate_start(root="skill_payload", bucket=..., skill_name=...)`, the harness edits a private standalone Git snapshot, and the parent applies the captured diff explicitly under a whole-payload content-hash CAS, after which the existing skill review is stale. Compatibility is one-way and permanent: a saved task contract carrying `disabled_tools=["claude_code_edit"]` also withholds the successor `delegate_start` (registry `_disabled_tools`), and the frozen `GET /api/claude-code/status` + `POST /api/claude-code/install` endpoints stay — the Claude runtime still powers the api-route advisory review. Do not resurrect the tool name.
- Successor parity rule (from the D10 postmortem): a tool may be called replaced, retired with a successor, or fully migrated only after a persistent golden test proves every previously supported user-visible target class through the successor to the final outcome. Deleted-test tombstones and disclosure prove intentional code removal, not successor parity. Dropping a target class requires an explicit owner decision naming the lost user outcome; approval to remove the old tool name or implementation is not that approval.
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
- `plan_task` runs no scouts and no Atlas (plan-review redesign 2026-08-15): the engine
  in `ouroboros/tools/plan_review.py` reviews the agent's SPEC through the configured
  reviewer rows in-process (api_chat) or as retrieving sessions (agent_session), pays at
  most `OUROBOROS_REVIEW_MAX_CYCLES` panels per task, replays an identical envelope for
  free, and records every wave in `plan_review_state` v2 (bounded history). The
  planning-scout wait knobs are retired (`config.RETIRED_SETTING_KEYS`).
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
  `SETTLED_STATUSES`. Cancel INTENT is never a status value (Poltergeist phase A):
  every cancel ingress — tool, HTTP single/cascade, evolution stop (pending AND
  running evolution tasks; never an in-place queue prune), project
  delete, cascade descendants, boot migration — writes a durable intent through
  `ouroboros/cancel_intents.request_cancel` (nothing is minted for an
  already-settled task WITH NO LIVE OWNERSHIP — a settled RESULT does not mean
  a dead WORKER (GR6-1): the terminal result is persisted before post-task
  cognition ends, so every ingress checks live physical ownership
  (`supervisor.queue.task_has_live_ownership` in-process; the queue-snapshot
  twin `task_status.task_has_live_queue_ownership` worker-side) and passes
  `allow_settled_target` while a RUNNING row / busy worker remains, letting
  custody kill the still-spending process while completion-wins preserves the
  stored result; the cascade-coordination shape does the same for a settled
  root with live descendants: `scope=cascade` with `allow_settled_target` —
  the watchdog's replay trigger for the subtree, settled only by the cascade's
  no-live postcondition; the recorded scope is WIDEN-ONLY, cascade never
  narrows back to single) and FAILS CLOSED when that write fails: a cancel
  without a durable, watchdog-replayable intent is refused with a typed error,
  never run unfenced — an evolution-stop task whose intent write fails is KEPT
  and the stop reported INCOMPLETE.
  Timeout reaping is deliberately NOT a cancel ingress (owner decision: 1=A
  covers explicit cancellation; the reaper keeps its own custody protocol over
  the shared `reaping` slot marker and mints no intents). Effective reads
  project the intent as
  `cancel_state: "pending"` (plus `cancel_reason` when the intent carries one),
  the supervisor's `cancel_task_custody` is the one
  settle owner — it claims the intent BEFORE any custody mutation (a refused
  claim is `failed` with zero mutation, so racing custodies can never
  double-settle through the capture-miss lane), its claim is EXCLUSIVE while
  alive (a claimant the pid probe proves alive is NEVER abandoned by age; only
  a provably dead pid, or age-stale with liveness unknown, is recoverable),
  every intent mutation is
  fenced by the claim generation, so a second ingress cannot double-settle and a
  taken-over attempt cannot revert the new owner, and the claim is re-verified
  (pid + generation) immediately before the durable terminal write — a claim
  lost across the kill/join window aborts the publication (deliberately ONE
  re-read at the one write that matters, not a renewable-lease subsystem —
  rejected by owner scope); the secondary settle
  sites (pre-assignment pending drop, budget-drain `fail_tasks` — whose intent
  reads resolve at the CANONICAL supervisor root, never a child's
  `budget_drive_root`; note `fail_tasks` has no production caller today —
  budget exhaustion pauses tasks before dispatch rather than draining them,
  and the fence there is pinned by tests against future wiring) hold the SAME
  claim/generation fence and yield to a live claim owner. A `scope=cascade`
  intent is settled EXCLUSIVELY by the cascade's no-live postcondition: every
  other settle site is refused atomically against the CURRENT durable scope
  (a mid-flight widen beats a stale claim snapshot; the refused claimant's
  claim is auto-released for the watchdog), and the postcondition always owes
  the tree's one summary BEFORE it settles — including the replay/already-down
  path — while re-judging stale sweep failures against the current durable
  status. Natural
  completion WINS a
  late cancel (a completed result is
  never overwritten or stripped — discarding is the parent's separate explicit
  `discard_child_result`). The owner's terminal answer is registered as OWED in
  the durable outbox BEFORE the intent settles (a crash between settle and send
  replays instead of losing both the watchdog trigger and the answer); a
  registration that could not be made durable leaves the intent OPEN on the
  cancel path and is a typed `terminal_delivery_unregistered` disclosure on the
  normal path — never a silent gap. On the natural path the owed row is
  registered immediately BEFORE the durable result write (projection-over-
  replay: a crash in the window leaves an owed row boot replay delivers; no
  boot scan over task_results).   The intent and delivery registries read STRICT
  in every mutator, not only at the ingress — claim, release, settle and scope
  fail closed on the same typed error the mint raises, because a mutator that
  read softly would answer "no active intent" over an unreadable file and drop
  the claim-first fence:
  a corrupt projection refuses the mutation loudly instead of collapsing to
  `{}` and overwriting every active row — and strictness reaches ROWS, not
  just containers (GR6-3): a malformed pending/intent row or `delivered`
  entry refuses the mutation (bytes kept) and the enforcement reads
  (watchdog sweep, outbox replay) disclose loudly once, then quarantine the
  row. The unreconciled-delegated-runs disclosure is outcome-INDEPENDENT
  (GR6-5a) and rides completed/failed deliveries too; an EXISTING-but-
  unreadable custody log audits as the typed
  `delegated_run_state_unknown:custody_log_unreadable` marker, never as
  cleanly reconciled (GR6-4); and the cascade digest enumerates descendants
  by ancestry rooted at the cancelled node, so a mid-tree cascade lists its
  grandchildren and non-subagent descendants (GR6-2).
  `task_done` is validated through the DURABLE result
  UNCONDITIONALLY for every non-ephemeral event — a blank event status (the
  primary producer's shape) validates exactly like a settled claim, a settled
  claim over a non-settled row is the same lifecycle fault as a non-settled
  claim, and the copy-back exception path neither skips the validation nor
  synthesizes a `completed` row for a task that never wrote one (only
  `interrupted` keeps its restore-path exemption). The legacy
  `cancel_requested` status survives on a
  read-path only. `wait_task` and
  `get_task_result` keep the full handoff plus a bounded verification-receipt
  projection: every outstanding red/masked receipt first, then newest rows, with
  an exact omitted count; read the canonical store and fall back to the recorded
  child drive before copy-back. `wait_tasks` stays batch-compact:
  `task_id, status, cost_usd (+ its honest alias accounted_upper_bound_usd and
  cost_final — C2), child_result_sha256, outcome_axes, result,
  trace_summary, capability_delta when disclosable, duplicate_of`; it points to
  the hash-addressed full result rather than re-inlining trace/ledger forensics.
  Unknown ids are probed across result, queue, and tree-ledger authorities and
  return typed rows plus a bounded actual-child roster and its exact omitted count;
  an all-unminted set returns after the 30-second registration grace unless an id
  becomes real. `wait_task`, `wait_tasks`, and `delegate_wait` may disclose an
  expired cache horizon only from the latest recorded applied `5m`/`1h` TTL;
  absent, bare `default`, or unknown TTL evidence stays silent, and no surface
  predicts the next send's token rewrite.
- Stop policy and owner hurry (S3). `stop_policy` is an axis on the durable
  cancel intent, independent of cascade scope: absence means IMMEDIATE (frozen
  programmatic compatibility), `finalize_then_cancel` is 202-pending plus one
  bounded owner-stop episode owned by `supervisor/owner_stop.py`, transitions
  are monotonic (immediate hardens, graceful never softens). The owner hurry
  control is typed and TASK-LOCAL: `kind=hurry` through the owner mailbox only,
  never a chat message, never owner prose in `_drain_incoming_messages`, never
  a global settings mutation, never a P3/commit/review-gate weakening — these
  hold for every install configuration class. Its durable projection writes
  ONLY through `update_json_locked` on the `owner_hurry`/`owner_hurry_history`
  keys (never `write_task_result`), is keyed by `task["_attempt"]`, and every
  same-id requeue producer (reaper timeout AND crash requeue) must call the
  ONE shared `owner_hurry.retry_reset`. UI surfaces share
  `web/modules/task_control_menu.js`; the `owner_hurry` event family is
  non-chat (`log_events.js` hides it with `visible=false`).
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
- Top-level tab/pill buttons are a single design-system control: `renderTabStrip` + `.app-tab-strip` + `.app-tab` + the `--pill-*` CSS variables in `web/style.css`. Do not redeclare per-page tab padding, font size, border radius, or active styling in page CSS files.
- Scrollable page bodies use the shared `.scroll-fade-y` mask when content can pass under fixed page chrome. Do not copy/paste custom gradient masks into page modules; extend the shared class if the fade rhythm changes.
- Masonry-style widget packing uses `web/modules/masonry.js::applyMasonry`. Do not reintroduce CSS Grid row packing (`align-items: start`) for unequal-height widget cards; it leaves row gaps under shorter cards.
- Widget card ordering is a host UI preference. Persist it through `/api/ui/preferences` and `data/state/ui_preferences.json`; never rewrite extension manifests or widget declarations to store owner layout.
- New visual dimensions should become CSS variables first (`--pill-*`, `--button-*`, `--page-header-*`, etc.) and then be consumed by shared classes. Hardcoded page-local dimensions are review debt unless the component is genuinely unique.

#### Setup / Onboarding Layout
- The first-run wizard is a compact multi-step flow. At the default desktop
  window size it should not force scrolling merely because the access step has
  several provider fields; use responsive two-column field grids where width
  allows and keep step copy short.
- There is ONE wizard host: the `GET /onboarding` page, served by the gateway
  and loaded as an ES module from `/static`. The desktop setup window opens that
  URL after the managed server is healthy, and the blocking overlay frames it.
  Do not reintroduce a pre-server or inlined copy: a step that cannot call
  `/api/*` or import `web/modules/*` is the defect this host removed.
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
- One capability, one section. The whole subagent story lives in Agents →
  Delegation (`web/modules/subagents_settings.js`), beside Review lanes: the
  route (`OUROBOROS_SUBAGENT_HARNESS`), the optional account pin
  (`OUROBOROS_SUBAGENT_PROFILE`), the write permission
  (`OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS`), and the two counts that bound it
  (`OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT`, `OUROBOROS_MAX_SUBAGENT_DEPTH`),
  with the two path roots behind a collapsed Advanced disclosure. They all answer
  "where and how far do subagents run"; `OUROBOROS_MAX_WORKERS` stays in Advanced
  because it sizes the process pool, not the agents. Never render a second control
  over the same settings key — two controls carry two drafts, and the last one
  collected wins, and a MOVE that leaves the old markup behind is exactly how a
  duplicate appears (`tests/test_agents_tab_static.py` pins each moved id to one
  occurrence).
  The delegated-run MODEL is the owner's default, authored here as the `=model`
  tail of the same key from engine discovery ("Engine default model" = empty
  tail); reasoning effort stays derived per call, and a hand-written `:effort`
  remainder rides through verbatim with no control over it.
  The ACCOUNT pin is a sibling settings key, never a fourth grammar position:
  its selector reuses the reviewer rows' `profileOptionsFor` ('' = automatic
  rotation first; an undiscovered saved pin keeps its option so a daemon-down
  save cannot erase it), a harness switch visibly drops the pin, and turning
  delegation off authors the pin away with the route.
- Onboarding completes in ONE transaction, and install-time defaults belong in
  it. `POST /api/onboarding/complete` persists settings, the next-boot runtime
  mode, the fresh-install safety default and the agent-subscription preset in a
  single write; `GET /api/onboarding` normalizes for display but must never
  persist, because a read that authors `settings.json` destroys the
  fresh-install latch both install-time behaviours depend on. There is no second
  completion path on any host — no generic-settings pair, no desktop save bridge
  — and the client treats only the exact success envelope as a completion: a 2xx
  whose body will not parse, or lacks `ok`/`runtime_mode`/`restart_required`, is
  a failure the wizard shows, never a silent success that discards the restart
  receipt. A failure after the bytes reach disk says so rather than claiming
  nothing was saved.
- The onboarding frame is sandboxed WITH popup permission
  (`allow-popups allow-popups-to-escape-sandbox`). The Agents step's primary
  action is the agent's own sign-in link in a new tab; a sandbox without those
  tokens blocks that click silently, and a popup that inherits the sandbox
  reaches the vendor's OAuth page with neither same-origin nor scripts. Assert
  this behaviourally, from the login card's own markup — an attribute-string
  test passes while the click stays broken. Install-time
  defaults are compiled from LIVE discovery and refuse typed when a model
  cannot be resolved — never guessed, never half-applied, and never
  re-derived after onboarding (that would be a second, continuous authority
  over settings the owner has since edited). Install time is a conjunction of
  three proofs — no recorded completion (`OUROBOROS_ONBOARDING_COMPLETED_AT`,
  written by every completion), no preset generation, no `settings.json` yet —
  because "no working provider" is a state an old install reaches too. A
  once-only decision is never taken on a moment-in-time reading: the daemon's
  `next_up` is quota-derived, so a subscription whose window is spent during
  onboarding must stay in the preset (D-3), and the seat is resolved from the
  durable facts — credential kind, enabled, present, verified. The verdict is
  read DUAL-WIRE: a unified engine carries it in the additive `accountPools`
  key beside an empty `harnessAccounts` compatibility list, a legacy engine on
  the per-harness accounts row — pool first, legacy second, never re-derived
  from the profile list, and an unknown `next_up.kind` on either wire is a
  fail-safe refusal that still lets the configured-seat scan answer.
- Owner settings writes go through `gateway/owner_settings.py`. The settings
  lock is a PRECONDITION of the write, not a hint: `_acquire_settings_lock`
  answers `None` on timeout and a writer that proceeds anyway is unlocked while
  claiming to be atomic. Once the bytes land, the response must say so — carry a
  `CommitBoundary` through the write and report a later failure as that step
  failing, never as a failed save (BIBLE P1). `saved` is a FIELD on both sides:
  pre-commit refusals answer through `unsaved_error`, or a client cannot tell
  "nothing was written" from an envelope that simply predates the field.
  `owner_write_guard` belongs only on an endpoint that calls
  `_owner_write_settings`; on any other it translates unraisable exceptions
  while advertising a lock the endpoint never takes.
- A setting only an ENDPOINT may author is disk-only in BOTH directions.
  `config.ENDPOINT_AUTHORED_SETTINGS` is consulted by the loader and by the
  environment projection, so the value is never read from `os.environ` and never
  exported back to it; the generic save's merge skip-list reads the same set.
  Blocking only the request body is not enough — an install-time fact that the
  environment can supply closes its own window before the endpoint runs.
- A control the owner cannot use is worse than none. With no agent subscription
  connected the Delegation section says so and points at Accounts in the same tab
  instead of rendering a delegation toggle whose every dispatch would silently
  fall back to an API child. Harness lists come from the accounts panel's own
  source (`accountRows` over `/api/claudexor/status`) — one catalog path, one
  login-capable discriminator — and `accountRows` is dual-shape: a unified
  payload (server-stamped `unified_accounts`) serves every account as a named
  profile wrapper and synthesizes no native pseudo-row, while a legacy payload
  keeps the pseudo-row behavior-identical, plus the additive fail-open
  `enabled` projection every row now carries; the account-pin options ride the
  same payload's named profiles through `indexProfilesByHarness`.
- Owner-facing copy says "agent", never "coding agent" (D-10, owner verbatim:
  the same subscriptions build presentations and run arbitrary tasks). Product
  names — Claude Code, Codex, Cursor — are trademarks and stay as they are.

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
- [ ] Keep a complete loop-local `DeliveryCandidate` once a substantive answer exists. A service round may return `keep`, or `replace` plus the complete replacement answer; allow one repair for malformed control, then preserve the prior complete answer and mark finalization degraded. A FORCED finalization (budget/round/deadline/provider/children rails) resolves an armed control purely and without retry instead: valid keep/replace is honored, anything malformed preserves the retained candidate with a typed degraded reason, and the protocol JSON itself never reaches chat or the durable result. A service notice alone does not change evidence. Owner messages, tool effects, child results, and verification receipts advance the evidence revision and require fresh delivery/acceptance binding. Finalize task-scoped service outputs/errors before host acceptance and require a complete replacement when that evidence changes; keep the `finally` path as idempotent cleanup only. This control must not bypass verification, acceptance, safety, skill-finalization, deadline, child-handoff, the unconditional `FINAL ANSWER:` latch, or the task-level answer protocol.
- [ ] Every direct child result needs an exact-hash disposition through the existing `tree_note(kind="decision")` tagged payload (`type=child_result_disposition`, child id, `integrated | irrelevant | deferred`, complete-result SHA-256; note text is rationale). One call may instead carry a `children` array of such entries (batch form): each entry is validated exactly like the single form, invalid entries are rejected individually by index while valid ones record. The typed task-tree row is the sole authority; task-result disposition fields are derived reads, never a mirrored write. The join-ledger helper alone validates lineage and current content. Stale or malformed payloads change nothing. `deferred` suppresses only the unchanged reminder and forces an honest degraded/best-effort terminal answer until the item is resolved. Natural completion WINS a late cancellation (owner decision 4=A, 2026-08-11): a child that settled its own completed result keeps it — payload, artifacts, and cost — and the cancel settles as already-settled; discarding a kept result is the parent's separate explicit `discard_child_result`. A cancelled (not completed) child still has its salvageable output preserved on the canonical drive before its bounded scratch is removed. Only a SETTLED `cancelled` status counts as a handled cancellation disposition: a child wedged in the legacy `cancel_requested` STATUS latch is intent, not outcome — it stays visible in the parent's handoff reminder as cancel-pending until custody settles it.
- [ ] Host task acceptance is root-only. Queued/headless/scheduled roots are reviewed in `auto` and `required`; direct eligibility is the union of `outcomes.turn_has_reviewable_effects` and a typed deliverable/criterion. Ordinary read-only tool activity, pure conversation, and meta/routing controls are not reviewed, and child reviews remain advisory. Eligibility must use structured facts, never keywords (Bible P3/P5). For an eligible root under `auto|required`, agent-callable `task_acceptance_review` validates/stores evidence and optional agent disposition but makes zero reviewer calls; it returns `deferred_to_host_acceptance`, `authoritative=false`, and the evidence revision. The call itself never widens eligibility; child and `off` behavior remain unchanged.
- [ ] Before root acceptance, atomically fence new descendants under the queue lock and prove recursive subtree quiescence from the existing task-status SSOT. Split-drive ACK, subtree, and acceptance-timing reads/writes use canonical `budget_drive_root`. Preserve the prior verdict until the replacement is recorded. A revision must explicitly reopen the fence; terminal/degraded outcomes seal it.
- [ ] The host runs the authoritative acceptance panel once per unchanged candidate-hash/evidence-revision/fence binding. Task-acceptance actors receive one substantive call and at most two physical attempts total. Record transport status, parse status, and valid-response semantic verdict separately, with actor model/provider, role, coverage, panel id, quorum contribution, reason, enforcement impact, and binding hashes. Public task/event/UI records receive only the compact projection; full model payloads remain in private audit storage. `adaptive_quorum` applies; any contributing FAIL fails, DEGRADED abstains (the reviewer verdict vocabulary `PASS|FAIL|DEGRADED` is NOT narrowable — `_contract_valid_actors`, the deliberate-DEGRADED capsule rail and the host's core-overflow DEGRADED all depend on it), and no quorum is a terminal HOST decision. The host acceptance decision itself is written ONLY by `loop._set_acceptance_decision` and has exactly three owner-facing states — `accepted | revision_requested | finalized_unaccepted` — each with a typed `reason` from an existing structured fact; an unknown status fails closed to `finalized_unaccepted` keeping its raw token as the reason. When you add a writer, add its reason to the closed set AND check every value-keyed reader: `outcomes.derive_loop_outcome` keys the eligible-but-skipped degradation on the status+reason PAIR (`review_skipped_deadline_reserve` plus the closed forced-rail `ACCEPTANCE_BYPASS_REASONS`), and breaking that pairing is a silent false green. Forced exits stamp their typed bypass record in the common terminal recorder (`_record_forced_acceptance_bypass`) as a pure ledger write — never a fence, panel, extra round, or prompt text on a forced path, and never overwriting an existing host decision — with ONE exception (owner decision Q2A, 2026-08-10): the forced `children_unabsorbed` rail still runs the acceptance panel for an acceptance-eligible root when the subtree is quiescent, with the undispositioned-children debt included in the evidence packet; because that rail cannot take another round, a requested revision terminalizes as `finalized_unaccepted` with the typed `revision_unavailable_on_forced_rail` reason, while the process outcome stays best-effort `children_unabsorbed`. The agent may write only `agent_disposition`/`agent_rationale`, merged into the host decision, never replacing it. Clean requires PASS + solved + supported criterion evidence. Chat and Logs must use the same severity reducer, and degraded review or best-effort/degraded objective must never render as green solved. Do not add task scope review or reuse the commit gate.
- [ ] The acceptance improvement loop is a reviewer-authored DIALOGUE (v6.74.0): obligation identity comes from the reviewer's typed `disposition_kind`/`obligation_id` (an unknown re-raise id fails closed to `new`, disclosed — never a silent fresh hash id); a re-raise reopens the row WITHOUT wiping the agent's argument (`previous_disposition`/`previous_reason`/`reopened_count` survive into the evidence catalog and the obligations clause); termination beyond a clean PASS/accepted rebuttal happens ONLY via the reviewers' quorum `dialogue_status` judgement reduced over ALL contract-valid actors (`aggregate_dialogue_status` — never `_contributing_actors`, which drops a DEGRADED slot's vote) or a real rail — no host counters, no answer/verdict hashes, no keyword gates (P5). Changes here must cover: malformed reviewer output, unknown/stale `obligation_id` on a re_raise, partial panel failure, multi-slot dialogue-status disagreement (the reducer's precedence), replay/restart durability of obligation rows, false completion, and the backward-compatible default when the new fields are absent.
- [ ] An explicit `max_improvement_passes` binds under every legacy policy. Without one, the shared review-cycle cap binds under EVERY policy, Required+Blocking included: `OUROBOROS_REVIEW_MAX_CYCLES` (SSOT `ouroboros/review_cycles.py`; string, default `"2"`, `"unlimited"` = no local count cap) gives `improvement passes = cycles − 1`; the retired `OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES` is migrated into the shared key at settings load and never binds at runtime.

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
- Take a fresh rescue before every destructive rollback and before boot-resume
  re-materialization: the pre-update snapshot predates the merge and holds none
  of the resolution. The hook is fail-open — never block a rollback on it — but
  its outcome, captured or failed, is disclosed durably at capture time, before
  the destruction. Record the pointer in the update transaction so a replayed
  rollback does not re-snapshot and a retry rescues what appeared since; keep
  that pointer until the transaction ends, because a re-materialized merge looks
  identical to restored work and is not evidence the rescue was applied.
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

> **Authority split — read this first.** `docs/DESIGN.md` defines the **visual
> and interaction semantics**: the type scale and hierarchy rule, foreground and
> state colour meaning, status/chip conventions, card and row anatomy, onboarding
> density, dark-theme contrast. **This section defines the engineering rules that
> preserve them**: where values may live, which component is the SSOT, what
> counts as review debt, how a visual change is verified. Changing what a size or
> colour *means* is a `DESIGN.md` edit; changing how the code is allowed to
> *express* it is an edit here. Neither file restates the other.

`web/style.css` custom properties and shared component classes are the value
SSOT. Documentation keeps semantic roles and failure-prevention rules, not a
copied color/radius/dimension inventory.

Typography and hierarchy are `docs/DESIGN.md`'s call, and the engineering
consequence is narrow: a text declaration on a migrated surface names a
`--type-*` size token and a named foreground token. A rule that declares a size
and no colour is the specific defect that made secondary text inherit
near-white primary ink across the settings panels;
`tests/test_web_typography_static.py` keeps that class closed on the migrated
files. Migrating a new surface and extending that guard to it are the same
commit.

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
- Floating chrome combines gradient and masked backdrop blur so the blur edge
  does not become a visible seam. The chat composer intentionally keeps blur on
  the input surface and reserves measured message padding around the dock.

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

## MCP Client Integration

The base runtime is an optional client for trusted HTTP/SSE and local stdio MCP
servers; it is not an MCP server. `ouroboros/mcp_client.py` owns server parsing,
transport-specific validation, provider-safe tool names, discovery, timeout,
and result normalization. `ouroboros/secret_masking.py` owns the exact shared
Settings/MCP auth-placeholder emitters and recognizers. Settings carry only
`MCP_ENABLED`, `MCP_TOOL_TIMEOUT_SEC`, and structured `MCP_SERVERS`; tokens never
appear in status responses.

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

### Python dependency locks

`pyproject.toml` is the direct-dependency SSOT and `uv.lock` is the reviewed
cross-platform resolution. Local and CI commands use `uv sync --locked`; do not
add an independent hand-written requirements list. Release packaging preserves
the separate build and embedded interpreters by exporting build requirements
ephemerally and committing `requirements-runtime.lock` for embedded pip. The
legacy `requirements.txt` is a generated pointer to that export for N-1 managed
updaters, never a dependency authority. A dependency change updates the project
metadata, runs `uv lock`, regenerates the runtime export with the exact command
in README, and leaves its CI clean-diff check green.
The pinned `tool.uv.required-version` and the digest-pinned `setup-uv` action
make resolver changes deliberate rather than an ambient CI upgrade.

`uv tool install "git+https://github.com/razzant/ouroboros.git@ouroboros"`
is the documented checkout-free CLI/server path. It resolves the project
metadata into an isolated tool environment but does not read this repository's
`uv.lock`; branch installs therefore follow dependency ranges as well as source
HEAD. Documentation may pair that convenient form with a full commit SHA to pin
the source revision, but must not claim that it locks dependencies or describe
it as a release-artifact install or contributor development environment.

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

An opt-in lane may also be gated by an environment variable instead of a
marker of its own, for a suite whose cost is a real server rather than a
provider key. `tests/test_e2e_cancellation_scenarios.py` (E1-E12: cancel,
cascade, graceful stop, hurry) is gated by `OUROBOROS_E2E_CANCEL`:
`mock` spawns a real isolated `server.py` against a LOCAL stub model
(`tests/fixtures_e2e_cancellation.py`) and contacts no external host, while
`paid` adds the scenarios whose subject is a real delegated-run transport or
real cost accounting and needs one provider credential, named through
`OUROBOROS_E2E_PAID_KEY_ENV` and a slug in `OUROBOROS_E2E_PAID_MODEL`. Unset,
the whole server-driven part skips and only the driver/gateway contract tests
run. Every scenario asserts the durable artifacts — `state/cancel_intents.json`,
the `cancel_intent` forensics in `logs/supervisor.jsonl`, `task_results/<id>.json`,
`state/terminal_deliveries.json`, the `owner_hurry` projection — never an HTTP
status alone. The driver is `devtools/benchmarks/common/server_runner.py`
(`IsolatedServer.cancel_task` / `hurry_task`), which posts the same bodies
`web/modules/api_client.js` does.

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
passes as CI in one disposable checkout and scrubbed temporary data root. The
candidate is captured universally — one hardened worktree-vs-`HEAD` binary diff
applied as raw bytes, assembled identically whether the source index is clean,
dirty, or mid-merge — and a capture or apply failure is the typed
`PREFLIGHT_CANDIDATE_ASSEMBLY` hard block with its own remediation, never a
test failure:

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
archive packaging steps, then performs a smoke test against that final archive.
The smoke checks require the embedded repository bundle, run the packaged CLI
with `--help` in an isolated home directory, then use the embedded Claudexor
seed and Node from that extracted final artifact to perform install, extraction,
exact identity probe, owned-daemon handshake, one fake task, and an
identity-bound graceful stop of the serving closure. The Linux shard builds
PyInstaller under the pinned portable interpreter so the runner's glibc cannot
leak into the desktop launcher's libpython, then wraps that proven x86_64
payload into `.deb`, generic `.rpm`, and RED OS 8 `.rpm` assets. Their metadata
declares Git, which packaged bootstrap requires; the gating smoke installs
through `apt` or `dnf` in Ubuntu 22.04/Fedora 42 and proves dependency
resolution, desktop integration, the installed opt-in systemd user unit and
its launcher/cgroup/no-restart contract, the real packaged CLI, and a bounded
desktop-launcher start. The unit never activates during package installation;
the launcher remains the sole restart and panic-policy owner. Vendor
image smokes for Astra Linux and RED OS are non-blocking evidence, and their
outcome is reported without becoming release authority. The separate
Claudexor platform gate repeats that fixture path on ordinary branch changes and
adds the explicit-key live compatibility matrix; neither path installs a
floating Claudexor npm package. The macOS check also requires the
`Applications -> /Applications` drag target, the separate `Install CLI.command`
payload, and an arm64 app executable.

Linux additionally emits an AppImage built by a version- and digest-pinned
`appimagetool` with a separately SHA-pinned embedded type-2 runtime. CI extracts
it for metadata and SBOM inspection, then uses real extract-and-run invocations
to verify product version, CLI dispatch, the browser-fallback launcher, gateway
readiness, payload lifetime after `run --start`, shared libraries, and graceful
shutdown. A nested extract-and-run relaunch receives a private temporary base;
its marker-gated `AppRun` waits as the payload custodian, restores the caller's
`TMPDIR` before launch, and removes the verified extraction plus the empty private
base after the launcher exits. The release smoke proves the resulting type-2
runtime → custodian → launcher process chain and waits on the runtime before it
requires both paths to be absent. Ordinary FUSE launches retain direct `exec`.
This smoke deliberately makes no native GTK/Qt claim: packaged native
webview coverage remains a separate Linux distribution contract.
`OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1` is only a local-builder escape hatch
for hosts whose system packages are managed separately: it skips Playwright's
interactive host-library installation, not browser-binary bundling, and a build
using it must disclose that browser host compatibility was not locally proven.
Each shard also generates a CycloneDX SBOM from the payload extracted from the
final archive. The Linux payload inventory is reused for its three native
wrappers because their `/opt/ouroboros` bytes come from that same payload; each
wrapper still has its own digest-bound smoke receipt, provenance attestation,
and SBOM attestation. The macOS smoke proves the Applications link, then removes only
that link from the SBOM staging copy so Syft cannot follow it into the runner's
host `/Applications`; the app and CLI launcher remain in the scan. The workflow
downloads a fixed Syft release asset and checks its platform-specific SHA-256
before execution. GitHub artifact attestations bind both build provenance and
the SBOM to each final asset digest. The release job downloads the three
archives, the AppImage, three native Linux packages, and their proof files,
checks the exact seven-asset allowlist,
recalculates every digest, and verifies both predicates against the exact source
SHA, tag ref, repository, and signer workflow before it writes:

- `SHA256SUMS` for release assets, SBOMs, and smoke receipts;
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
