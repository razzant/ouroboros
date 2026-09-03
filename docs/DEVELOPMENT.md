# DEVELOPMENT.md — Development Principles & Module Guide

## Role and authority

This is Ouroboros's engineering handbook: how to name, design, implement, and
verify changes. `BIBLE.md` owns constitutional principles;
`docs/ARCHITECTURE.md` owns the current structure, data flow, and rationale map;
`docs/DESIGN.md` owns visual and interaction semantics;
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

Preserve user-visible progress and reasoning. Chat authorship is stamped by the
producer and preserved through persistence and replay; never infer it from text
or promote host-selected intermediate output to a model final.

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

What belongs in `prompts/SYSTEM.md` (tier-0 for every Main/task profile in both
context modes — Background Consciousness and the safety supervisor carry their
own prompts — and competing with the task for context): identity and tone, the decision
loop (answer / promote / route / delegate / do it myself), cross-tool policy
(which class of tool or lane for which situation, root semantics, memory only
through its own tools, untrusted external data), prohibitions and safety
invariants stated once, and the memory contract. What does NOT belong there:
how a tool or mechanism works. A tool's parameters, signatures, recipes,
typed outcomes, and "when to choose it" live in its `get_tools()` schema — each
profile receives its own visible schema set on every round (delegated, repair,
ephemeral, credential and contract filters narrow it), so the schema is the SSOT
of the per-tool contract and a prompt sentence about it is a second copy that
drifts, while SYSTEM.md stays the cross-tool selection policy; mechanism
documentation lives in ARCHITECTURE or here;
runtime facts (capabilities, queue, catalog, receipts, health) are injected per
turn. A new tool therefore requires NO SYSTEM.md mention. Before adding a
sentence to a prompt, check that the schema or runtime block does not already
carry it; before removing one, check that they do (or add the missing fact to
the schema without growing it into a paragraph). Local-model compaction keeps
only the text before the first `## ` heading (plus the BIBLE section), so the
load-bearing floor rules stay in that preamble. Every prompt change reports the before/after byte size
in the commit or PR.

Recoverable tool failures are evidence for the next LLM turn, not triggers for
a host-authored recovery workflow. Return a typed, redacted result naming the
failed stage, already-completed external effects, and an actionable repair
hint; the LLM decides whether to inspect, repair, retry, use another capability,
clean up, or stop. Host code remains responsible only for deterministic
integrity and authority boundaries plus truthful receipts; do not add
task-specific auto-retry, fallback, cleanup, resume, or terminal-flow state
machines.

### Generality and emergence (P13)

Every non-trivial change picks a level: patch the case in front of you,
solve the class it belongs to, or build a framework for cases that do
not exist yet. The first fossilizes, the third speculates; aim for the
second. Two questions find it:

- **The invariant question.** What must stay true here for every
  install, provider, model, and consumer this seam serves — and what is
  merely a feature of the case at hand? Mechanism goes where the
  invariant lives, at its existing SSOT owner; the case's accidents —
  today's provider quirk, config, workflow, team shape, benchmark —
  stay out of shared surfaces.
- **The stronger-mind question.** If tomorrow's model were sharply more
  capable, would this change let it do more through the same seam — or
  would the change itself have to be torn out first? Strategies the
  current model exhibits (orderings, roles, decomposition habits) are
  examples worth recording as hints, never contracts to enforce.

The proof burden is symmetric: promoting a case detail into shared
structure requires showing it is an invariant (several real variants,
or one already-stable boundary); adding an abstraction requires a
demonstrated class — an imagined consumer is not one. In doubt,
generalize the meaning and the authority, keep the mechanism minimal
and local, and let the next real case pay for the next step. Reviewer
findings are evidence for this judgment, never policy that overrides it.

### Pricing and admission

Never add hand-maintained model-price tables, inherited prefix tariffs, or
numeric fallback prices. Query the exact route when a live source exists,
prefer provider-settled usage, and otherwise preserve `cost=None` and
`cost_final=false`. Unknown price is neither free nor a model-admission veto;
known exhausted budget remains enforceable.

### Anti-pattern: a chat id tested for truth (v6.115.0)

A chat id is a VALUE, not a boolean. `HIDDEN_CHAT_ID` (0) is the hidden
partition — the Skill Review panel plus every headless task admitted without a
registered project — and it is a REAL destination that no browser surface reads.
Absence is `None`, and a negative id is synthetic A2A traffic. `if chat_id:`
therefore does two wrong things at once: it drops a partition-bound notice AND
re-routes hidden work to the owner's main chat. That single habit is what made a
whole `ouroboros run` invisible while its children surfaced in Main as a nameless
card, and it recurred across two dozen sites because each one re-invented the
test.

Use the two normalizers instead of a third: `message_bus.notification_chat_route`
answers "where does this notice go" (first DELIVERABLE candidate, `None` when
none is), and `message_bus.coerce_chat_identity` answers "what is this row's
address" (explicit value kept, absence defaulted). Address a headless task once,
at admission (`log_addressing.ingress_chat_id`), and pass it downstream by value
— the ingress-capture rule below, applied to routing. `tests/test_chat_id_truthiness_guard.py`
is the source lint that keeps the class closed; its allowlist is where a
deliberate exception states its reason.

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
| `ouroboros/llm.py::supports_message_cache_control` | Which families support message cache controls | Medium/high as provider routing contracts change | Explicit family rules backed by provider behavior and dated live probes | Provider documentation plus a bounded cache-control send | A false positive can invalidate a request; a false negative loses the prompt cache | Retain the small explicit rules and re-probe when provider behavior changes; do not generalize by model-name resemblance |
| `ouroboros/reasoning_artifacts.py::SIGNED_PORTABLE` and its sealed classifier | Which families' SEALED reasoning artifacts (signed, encrypted, redacted, unrecognized) survive a same-model cross-provider replay; readable artifacts are portable by shape for every family | High; an upstream can bind a reasoning artifact to its endpoint without a routing-contract change | A short vouched family roster plus a shape-first classifier that fails closed on artifacts it cannot read | A same-model cross-provider replay probe of the exact family | A false positive 400s the replayed turn (the reactive strip-and-retry is the net); a false negative pins a portable transcript to one endpoint and forfeits same-model failover | Extend the roster only by a fresh cross-provider replay probe of the exact family, never by model-name resemblance; `openai/` was removed on 2026-07 field evidence despite an earlier passing probe |
| `ouroboros/provider_models.py::_ANTHROPIC_MODEL_ALIASES` / `migrate_model_value` | Direct-provider id spelling compatibility | Medium as providers rename ids and prefixes | Shipped compatibility mapping and current direct-provider id contract | Exact provider catalog/documentation can confirm a current id, but cannot establish whether a saved spelling was intentional | Removing an alias breaks upgrades; guessing aliases can silently reroute | Keep explicit compatibility aliases until a separately documented retirement window closes |
| `ouroboros/server_runtime.py::_RETIRED_MODEL_DEFAULT_REPLACEMENTS` and scope prior/legacy defaults | Which formerly shipped defaults are upgraded automatically | Release-dependent | Release history plus current `SETTINGS_DEFAULTS`; only known former defaults are migrated | A live catalog can show availability, but cannot infer user intent or whether a saved value was a default | Over-broad migration overwrites an explicit owner choice | Keep release-scoped exact replacements and regression tests; review retirement separately |
| `ouroboros/pricing.py::get_pricing` and `ouroboros/llm.py::fetch_openrouter_pricing` / `fetch_cloudru_pricing` | Exact-route model tariffs | High; pricing and FX drift independently | Exact provider catalog with nullable unknowns; provider-settled usage wins | Bounded live catalog fetch and provider-reported settled cost | Static prices look authoritative after becoming wrong and can corrupt admission | Preserve the live nullable design and cover it by regression; do not restore runtime tariff tables |

### Provider Independence

One configured provider must be sufficient for the agent loop, commit review,
scope policy, safety, and context/memory flows. Core capability must not acquire
a hidden OpenRouter or second-provider dependency.

Tool-schema changes are provider-contract changes. Every shipped built-in schema
must pass general JSON Schema and the known cross-provider subset over the
complete registry; trusted integration CI sends that same registry in one bounded
tool canary per supported provider family/API surface, while pull-request CI
remains secretless.

The trusted canary keeps the response choice's outer `finish_reason` as the
bounded per-call usage fact `response_finish_reason`; it is observational and
the reserved usage keys are host-owned, so provider-supplied extensions are
discarded unless the designated outer response supplies the value. It must not
be copied into canonical assistant history or used to change retry
classification. A schema-valid native call remains usable when a provider also
returns assistant text, with only a length/hash warning emitted on the trusted
integration test's warning stream; the warning list is host-owned and provider
extension fields are discarded before emission. Malformed native
arguments and invalid schemas stay red, with diagnostics limited to structural
facts, hashes, and parse position. Do not add a prose parser, provider hop, or
unbounded retry to make that contract green.

When adding or changing a provider, update one coherent route contract:

1. credential/readiness detection and exact model-id migration;
2. Main/Light/Fallback and reviewer-slot defaults without overwriting explicit
   owner choices;
3. canonical tool/reasoning/image/cache intent at `llm.py`, with provider wire
   projection and exact-route recovery delegated to the small transport leaves;
4. nullable pricing/settlement and truthful capability omissions;
5. review and scope routing, including sourced context-window evidence;
6. direct-provider and single-provider regression tests.

Local-only installs keep their local route. Unreachable shipped remote defaults
may be cleared, but explicit owner values are not. Scope authority follows the
BIBLE P3 policy: in owner-selected Max it requires the applicable sourced window
evidence; owner-selected Low records the declared skip rather than pretending a
partial review occurred. Current model ids and defaults belong in code/config,
not in this handbook.

Use `provider_models.ACTIVE_MODEL_SETTING_KEYS` for any new active consumer such
as provider detection, model catalog/provenance, credential planning, or Provider
Test. `LEGACY_MODEL_SETTING_KEYS` exists only for migration/history. In particular,
`OUROBOROS_MODEL_HEAVY` and `USE_LOCAL_HEAVY` may seed an explicit configured API
actor while the canonical list is absent, but must never become an active slot,
startup-readiness signal, test probe, or fallback. Do not patch each consumer with
its own Heavy exclusion; preserve the shared split.

The `-pro` suffix is an OpenRouter routing slug, not an official OpenAI model id.
A direct OpenAI Chat slot uses the plain Sol id; projecting the slug into Chat
Completions would turn an owner route choice into a guaranteed 404. This is a
compatibility constraint, not a mutable capability table or a reason to migrate
the conversation to Responses.

Provider-specific optional features may be unavailable on another single
provider, but the core loop must degrade explicitly rather than crash or silently
reroute.

Canonical assistant history and tool schemas are function-shaped across
providers. Do not add a second stored transcript for a provider dialect.
Direct OpenAI tool conversations stay on Chat Completions: the physical copy is
custom-first when non-`none` reasoning is requested, an exact custom rejection
may fall back to function with the same effort, and explicit `none` is a
task-local last resort only after both cognition-preserving forms fail. Direct
OpenAI sends `reasoning_effort` and `max_completion_tokens` provider-wide;
model-name prefixes are not admission authority.

All learned request-shape adaptation goes through the one provider-neutral
request-wire driver. Its identity is the exact provider/endpoint/API/model and
request shape, its action vocabulary is closed (`set_value`, `drop_field`, and
a registered `replace_dialect`), and it may never execute provider prose or
switch route. Reactive evidence becomes durable only after semantic success on
the exact settled physical candidate, expires after the shared TTL, and is
applied under the caller's existing attempt rail. Explicit `none` is never
durable. Legacy model-global effort/rejected-parameter stores are diagnostics,
not scheduling or normal dispatch authority.

Direct Anthropic is the deliberate exception to a purely reconstructed
provider transcript: while a native tool turn is unfinished, keep one private,
route-bound receipt of the whole assistant `content` list and replay it
byte-for-byte before the matching tool results. Scrub it on any
provider/endpoint/API/model change, fence that active unit from compaction, and
exclude thinking text, signatures, and redacted data from summarizer/public
observability. Do not synthesize an effort-to-`budget_tokens` policy.

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
- The exact-current 1001-1500-line band lives in `BAND_PATHS`. A new or
  re-entered path requires a nonblank rationale. `BYTE_DEBT` stores exact counts
  above 200,000 UTF-8 bytes and is shrink-only; regenerate both with
  `scripts/regenerate_size_ratchet.py`.
- Every non-grandfathered Python function or method fails the deterministic gate
  above 300 lines; exceptions live in
  exact `(repo-relative path, lexical qualname)` keys in
  `ouroboros/size_ratchet_manifest.py::FUNCTION_DEBT`. Methods above 150
  lines are a decomposition signal. JavaScript currently has only the module
  line-count gate.
- Runtime Python function/method count is checked against
  `ouroboros/review.py::MAX_TOTAL_FUNCTIONS`; the function iterator preserves
  the pre-v7 runtime scope (tests/devtools excluded) while module gates include
  those trees.
- More than eight parameters is a decomposition signal applied by BIBLE and
  reviewer checklist 2(c), not a deterministic size-test gate. Existing
  baseline debt is not retroactively a failing tree. Any advisory ratchet must
  publish its AST counting scope and bind its baseline to the final SHA.
- Enforcement surfaces: the OFFICIAL repository's CI runs the dedicated
  `size_ratchet` pytest lane as a blocking third step in quick-test and
  full-test — manifest exactness against the tip tree plus the pairwise
  shrink-only transition against the event base
  (`OURO_SIZE_RATCHET_BASE_REF`). Local surfaces never block on size: the
  default pytest lanes exclude the marker, and `check_worktree_readiness`
  (advisory preflight) and `codebase_health` report the same
  `validate_size_ratchet` findings as "official CI will enforce" warnings.
  The lane blocks only in post-push/PR CI, so its authority presupposes
  branch protection / required status checks on the official `ouroboros`
  branch (a repository-settings prerequisite outside this codebase).
  There is no committed-history replay: the previous manifest resolves
  merge-aware from `HEAD` or any of its parents, and a checkout with no
  committed manifest anywhere bootstraps from its own tree — so a locally
  evolved fork can always take an official update without being trapped by
  structural debt it inherited, while the official line keeps ratcheting.
  `scripts/regenerate_size_ratchet.py` validates its rendered candidate
  BEFORE writing it and refuses an unmerged index with a typed
  "merge in progress" error.
- Treat a size gate as pressure to reduce total complexity, not as a design
  reason for a helper or sibling module. First simplify where the change
  belongs: reduce control/data flow, delete dead, duplicate, or trivial-wrapper
  code, reuse an existing SSOT, and compact only redundant non-contract prose.
  Extract only when the new unit would still be the right boundary with the
  parent well under the cap: it owns a cohesive responsibility and explicit
  boundary, and is not a passthrough. Relocating the same complexity, or
  stripping contract-bearing comments, diagnostics, or tests to buy bytes, is
  not paydown. If neither a safe simplification nor a natural boundary exists,
  report the ratchet conflict instead of gaming or silently raising the cap.

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

### Invariant: Source-complete decision pipeline

Every new or changed continuity surface is reviewed as one narrow chain:

`producer → canonical full source → bounded projection → consumer → decision → retention/GC`.

- The producer records the complete event or artifact before it publishes a
  projection or wakeup. The canonical source owns identity, order, bytes, and
  integrity state; a cache or hot index is never a second authority.
- A bounded projection names what it omitted and carries a source reference
  that the *same actor* can resolve through an existing reader. `source_complete`
  is a coverage fact, not a permission to infer missing material.
- A consumer that can authorize PASS, a destructive rewrite, or replacement of
  a full contract must materialize the named source first. A known `partial`
  marker and an unverified claim that some host might retrieve more are not
  equivalent: the latter is not actor-attested coverage and cannot release the
  decision.
- Retention and GC are part of the chain. Anything referenced by a canonical
  result, review, identity decision, or project summary is retained or promoted
  before its execution root can be collected. Unreferenced scratch remains
  disposable under the unified retention owner; an unavailable legacy source is
  represented as an explicit gap, never silently treated as complete.

**Control-plane distrust is metadata, not a data-plane operation.** Paid model
output is evidence until a typed validity predicate fails. Control-plane
distrust — profile, route, parser, window — is metadata on that evidence: it
may lower authority to DEGRADED/SKIPPED/NOT_RUN, but it must not blank,
rewrite, or relabel the artifact or its original cause.

#### Review presentation adapters

`web/modules/review_presentation.js` is the frontend SSOT for adapting bounded
domain projections into review groups and attempts. It owns validation, stable
identity, ordering, labels, summary fallback, grouping, and typed presentation
state/tone. It does not author, mutate, or feed back canonical domain verdict,
lifecycle, routing, attention, or enforcement authority. Admission is
source-complete: require a stable group/attempt identity, an exact real
presentation-owner task, typed state/verdict (or explicit unavailable), an
exact detail reference when detail is offered, and exact task/candidate binding
for repository review. Omit an incomplete row; never guess from current chat,
repository, timestamps, model, tool name, or activity.

Legacy Plan Review is normalized once by `task_results.legacy_plan_review_projection`:
`public_task_result` adds that derived field only to its copied v1 state, and the
frontend adapter never reparses the nested legacy schema. A task-bound review may
create an inert owner anchor when no task activity survived the history window;
the anchor renders `Reviews` but contributes no task phase or liveness until a
typed task status, cancel state, progress/typing frame, or terminal detail proves it.

`ouroboros/review_execution_projection.py` owns the tiny cross-domain
`executions[]` wire. It admits only returned API usage or an actually resolved
delegated harness route plus model, and strips money, profile, raw output, and
requested-only intent. Skill history, Plan waves, and task-acceptance actors
reuse that projection. `web/modules/review_dom_patch.js` is the keyed DOM leaf:
routine review updates reconcile stable group/attempt/detail nodes in place so
lazy state, focused descendants, and detail scroll survive without adding a UI
state authority.

`web/modules/harness_presentation.js` is the vector-and-label SSOT for harness
identity. It owns the known labels, monochrome SVG geometry, generic unknown and
neutral direct-API fallbacks, escaping, and compact mark-plus-text helpers.
Marks use `currentColor`, visible text is always retained, and the surrounding
component owns status tone and execution wording. The helper must never infer
that a requested route executed or turn native selects into custom controls.

Both modules are pure read-side presentation. Reuse the existing Chat-history,
task-detail, exact domain-detail, and canonical physical-attempt readers; do not
add a review ledger, endpoint, persisted UI state, cost copy, or enforcement
layer. Compact review rows carry no dollars. Exact Skill attempt money appears
only inside the existing lazy detail when the history row declares
`physical_attempt_v1`; that detail joins the canonical ledger by exact wave and
slot, persists no totals, and leaves legacy attribution unavailable. A Plan
state write also appends one empty typed `review_reference` to the existing
bounded progress-history rail before publishing its live invalidation; the task
result remains the only Plan authority. Reconnect retains the latest reference
per owner, then independently limits references to the requested progress
window without consuming visible telemetry quota. Folded Skill groups likewise
retain every group and attempt for only the newest distinct owners within that
window, so one history rebuild cannot fan out unbounded task-detail reads.
Omitted review overlays use the existing `quota` truncation reason and Load
older expansion. Duplicate Skill
lifecycle acknowledgements remain typed `lifecycle_pointer` rows with no task
id, so they can never become lineage. They
enrich an existing exact owner card, or render once as subdued non-task progress
in the duplicate caller's chat when that owner card is absent.
Pin these contracts in `web/tests/review_presentation.test.js` and
`web/tests/harness_presentation.test.js`; keep grouping/reconnect/disclosure and
requested/effective/executed truth covered by the existing Skill Review,
render-batch, and review-truth suites.

#### Context and growth matrix

| Store / surface | Complete producer and source | Interactive projection / consumer | Growth and retention proof |
|---|---|---|---|
| Background observations | `BackgroundConsciousness.inject_observation` → `state/consciousness_observations.jsonl` enqueue rows | Cached pending/oldest status and bounded `_render_observations` view; identity-update consumer reads the gap marker and source ref | `BG_OBSERVATIONS_WARN_BYTES` in `context_budget.py` / `agent_startup_checks.py`; append-only rows, including unacknowledged rows, are not GC-pruned by the hot-store warning |
| Chat and biography | Canonical `logs/chat.jsonl`, rotated generations, and dialogue blocks | Main/Project context and archive-aware `chat_history` | Rotation/archive readers carry generation/gap coverage; blocks are the compression path, not a deletion of the horizon |
| Plan/review evidence | Exact task-artifact/observability bodies and reviewer route/thread receipts | Bounded review hot index, obligations, and latest-wave status | Exact artifact refs and candidate SHA bind the decision; index rotation cannot certify a missing or partial wave |
| Task/project execution | Canonical task result plus promoted child artifacts and summaries | Status cards, terminal rows, and Main/Project summary projections | Canonical promotion precedes child-drive GC; disposable task scratch follows the unified retention owner |

### Invariant: Continuation authority and bounded Main projection

Continuation is an explicit relation, not an inferred chat-memory feature. The
router contract requires `predecessor_task_id`: an empty string means a fresh
task, a non-empty value means continuation, and omission or `null` is a typed
refusal before any lookup, enqueue, or provider spend. The existing predecessor
source is retained by queue snapshot/restore, so a restart cannot silently turn
the selected task into a fresh one.

The authored continuation narrative is written at the result owner together
with its exact `get_task_result(include_authority=True)` source. A Project child
must promote its child-born narrative before child-drive collection; for a
non-Project split the parent-born narrative owns the field when a stale child
replica is copied back. Main's provider projection is defensive and recursive:
it deep-copies the authority, removes only the current task's duplicate nested
predecessor, and thresholds only the closed raw keys `result` and
`final_answer` using `context_budget.PREDECESSOR_RESULT_INLINE_CHARS`.
Oversized values resolve as persisted narrative, bounded exact-key legacy
authored lookup, or an explicit source-resolvable gap. They never use a raw
head/tail slice, invent a summary, or mutate the canonical result. Exact task
reads retain the full source.

The automatic startup injection (2026-08-30) is a bounded continuation
ENVELOPE, not a body copy - minted by ONE producer
(`contracts.task_contract.bounded_continuation_envelope`) for both the
startup binding and the legacy collapse on contract rebuilds. Every compact
terminal fact inherits by copy; the predecessor's operative contract core
inherits without its nested `predecessor_authority` (the recursion that
compiled 300K+ work orders); every field is whole-or-pointer against one
strict tool-result budget measured on its serialized form (lists and dicts
count, previews carry `full_chars` plus a named `source_ref`);
sha256/chars ride with their observation moment, and `previous_task_id`
keeps the chain walkable. A legacy body already free of growth carriers
passes rebuilds byte-identical - exact strings are authority. Durable
`task_results` bodies are the untouched SSOT, pulled whole through the
named `get_task_result(include_authority=True)` source (exact ranges apply
to the canonical work-order source, not to authority). The bound is
per-field: a pathological row of many near-limit fields can still exceed
the wire budget, where the refusal is typed and loud rather than a silent
$0 - no aggregate cap is imposed. No hop cap exists anywhere - depth
belongs to the mind, the floor only keeps bodies off the wire.

Provider context overflow is a typed recovery fact. The existing useful reclaim
and one strictly-smaller same-route retry retain their route order; a final
`context_overflow` skips the provider-unavailable/forced-provider path, keeps
`execution_status=infra_failed` and `reason_code=llm_api_error`, and records the
typed acceptance bypass and `failure.error_kind`. Ordinary provider outages keep
their existing recovery behavior.

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
behind. A UI instance may survive being hidden only under an explicit,
owner-visible retention reason — a project chat with pending work (staged
attachments, an upload in flight), a widget card the owner set to Keep running
— and even then it still owns its disposer, and Stop / unload / reload /
shutdown remain force-destroy boundaries. The reason is re-evaluated at the
instance's next lifecycle point, not continuously — a project chat's at the next
navigation, a kept widget card's at the next Widgets entry or lifecycle event —
so a hidden instance whose reason lapsed is released then, and this rule
promises no earlier release. For a kept widget card the force-destroy
boundaries are the owner's Stop, its skill leaving the live list (also while
hidden), a window reload and closing Ouroboros; a server reconnect
with the same served SHA keeps the frame when its skill is live again with the
same `revision` (a changed revision stops it in order and re-mounts it — at
once while Widgets is visible, at the next Widgets entry while the page is
hidden, where the pass compares presence only). The untyped shape "hide the
DOM node, keep the handlers" remains the leak this invariant forbids. Late
async continuations check a `destroyed` flag before touching state or
re-arming loops.

A module widget's disposer (`kind: module`) is the ordered dispose with
acknowledgement (ARCHITECTURE "Skills and Widgets"): it posts the dispose
message, keeps the bridge answering the child's hooks, and finishes — abort,
unlisten, remove the iframe — on the child's acknowledgement or after
`WIDGET_DISPOSE_ACK_TIMEOUT_MS`; a route iframe (`kind: iframe`) has no bridge
and its disposer removes the frame synchronously. The Widgets masonry (`applyMasonry`) returns an idempotent disposer for its two
`ResizeObserver`s, its `MutationObserver` and its pending animation frame.
That bounded wait is not the forbidden shape: the handlers live only until a
settle promise the page tracks per card key resolves, and a remount of the same
key waits on that promise instead of racing it.

Enforcement (honest disclosure): the deterministic leak test runs in the
release-tier `ui_browser` lane, not at commit tier; commit-tier coverage is
the advisory Repo Commit Checklist item 24. The class is closed
deterministically for the instrumented surfaces and advisorily for future
ones.

---

### Invariant: Embedded surfaces declare geometry and refresh semantics

Every owner-visible embedded or framed surface has an explicit host-owned
geometry/overflow contract, a paired disposer for every long-lived resource,
declared refresh/stream/error semantics, and a named real-consumer visual
verification path. Intentional omissions record why they are safe to defer.
For Widgets, framed `height` values are bounded and module auto-height is
host-controlled. Below its finite ceiling, applying a reported block size must
not change the child's inline-size basis; the host owns vertical scrollbar mode
without disabling the orthogonal horizontal overflow capability, and content
measurement includes the measured document's bottom padding and border.
Feedback-sensitive verification is event-driven on the relevant engine: it
proves temporal convergence to a quiet fixed point with a real consumer or
production-derived fixture that crosses the known wrapping threshold, rather
than comparing two snapshots. Module source loading and declarative requests
have a bounded host timeout; declarative job widgets keep their `job_id` and
bounded retry/timeout behavior visible in the refresh contract rather than
hiding them in an author script.
Missing or malformed job status is an immediate protocol error, while unknown
non-empty in-progress labels remain bounded pending states for producer
compatibility.
Repo Commit Checklist item 24 points lifecycle changes here instead of
re-deriving a second domain-specific rule.

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
| Advisory pre-review (`tools/claude_advisory_review.py`) | Two delivery classes: an `api_chat` row runs the bounded NATIVE inspection episode (governance docs reached through its read-only tools); an `agent_session` row receives a resolvable pointer marked MANDATORY FULL READ and the session reads the full doc itself — retrieval is disclosed (native reads are host-observed; vendor-session reads are not) | same two delivery classes | same two delivery classes |
| Scope review (`tools/scope_review.py`) | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting |
| Skill review (`skill_review.py`) | full inline (`api_chat`) / mandatory full source-root read (`agent_session`) | full inline (`api_chat`) / mandatory full source-root read (`agent_session`) | full inline (`api_chat`) / mandatory full source-root read (`agent_session`) |
| Plan review (`tools/plan_review.py`) | full for a SELF-MODIFICATION plan (structural path fact: a declared target resolves under the system repo); otherwise a heading-derived navigation map of BIBLE.md generated at runtime (never a copy) | inline, in full, for a self-modification plan; otherwise the lossless navigation map + a resolvable pointer (W3) | named on-demand pointer; a reviewer that needs it returns `need_evidence` and the host attaches it on the next cycle |
| Deep self-review (`deep_self_review.py`) | full canonical doc + Atlas accounting | full (max) / navigation map (low) + Atlas accounting | full canonical doc + Atlas accounting |

Skill Review keeps the full stable governance/host prefix for cache-friendly API rows. A
retrieving session reads those same canonical files from its source-repository root and receives
the byte-exact dynamic tail inline: manifest, frozen skill chunk, history and output contract.
The payload snapshot and per-chunk quorum therefore stay identical without rebilling or crowding
the session window with source text it can inspect sequentially.

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
`active_repo_dir_for(ctx)`. Exact user-managed installed-skill payload paths are the one data-plane exception for
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

1. **Cheap advisory preflight.** After edits, `preflight_review` may find
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
ids remain independent slots and `config.adaptive_quorum` owns quorum. Managed
exception: a managed-update resolution commit reviews the declared M0→S
resolution delta (`tools/review_subject.py`), and the commit gate binds S to the
index write-tree the review-binding fingerprint pins. Scope
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
settings load — cycles = passes + 1 — and never binds at runtime); commit gate
— paid triad+scope cycles per ROOT task (the whole task tree shares one
ceiling; a manual session is its own task; a follow-up task is a fresh root;
the paid fact is recorded on the attempt row at dispatch and the count is
derived from the attempt ledger); skill review — paid reviewer-panel
dispatches per ceiling key (the root task for task-driven groups; the manual
lane is scoped per content snapshot, so revised content restarts its count;
one chunked wave = ONE cycle). `unlimited` removes the local count
everywhere; deadline, budget and lifecycle rails still bind. A malformed
value fails closed to the bounded default and is logged once.

For task acceptance, the exact-binding tree-wallet claim is a strict write-ahead
stamp immediately before the API usage ledger crosses into physical dispatch.
Panel assembly, an unavailable route, or another pre-transport refusal consumes
no claim and leaves the binding retryable. Deadline and cancellation are
rechecked by that stamp; an unavailable claim releases the usage reservation and
blocks every parallel panel slot before reviewer transport rather than degrading
hard authority into fail-open cost telemetry. Task acceptance remains API-only;
the session-route stamp ordering continues to serve its separate replayable
custody contract.

Anti-pattern: paying for byte-identical review material. Never dispatch a paid
reviewer wave for material a gate has already reviewed under the same review
contract — the commit gate refuses a byte-identical staged diff for free from
the FIRST verdict-block (`identical_diff_refused`, quoting the recorded
verdict), and skill review replays a recorded substantive verdict for an
identical snapshot at $0 (only while the persisted review state still covers
it). A rebuttal is identified by CONTENT (sha256): a hash new to the streak
buys exactly ONE paid re-review; a repeated hash is refused free. The exact
rule keeps two axes distinct. Refusal-streak eligibility is about VERDICTS:
only substantive reviewer verdicts build (or end) the identical-bytes refusal
streak, and a rebuttal is "spent" only by the substantive verdict it bought —
never when it was refused undispatched or when its wave died on infra. Money
accounting is about DISPATCH: the limit counts PAID cycles, and every
physically dispatched wave counts whatever its terminal — so a dispatched
infra terminal (quorum failure, transport death, timeout) consumes money but
not the rebuttal, while infra facts refused at assembly (fit overflow,
sub-floor window) never dispatched and stay outside the count; the paid fact
is recorded write-ahead at first dispatch. Byte-identical resubmissions are
refused before any spend. Exhaustion is always the typed
`review_cycles_exhausted` event with honest exits, never a silent grind or
another paid dispatch. The paid-cycle ceiling counts dispatched waves under
BOTH enforcement modes; under advisory enforcement a commit after exhaustion
proceeds as a free replay with a loud typed disclosure (no new review spend),
while blocking enforcement refuses it. Note the honest scope of each
guarantee: the identical-diff refusal replays only recorded VERDICT blocks,
and a pure advisory line never mints one (advisory criticals disclose, they
do not block) — under advisory the no-new-spend guarantee therefore comes
from the exhaustion free replay, not from the refusal streak.

Scope of the review-contract fingerprint (deliberate): it covers the reviewer
roster, routes, enforcement, resolved efforts, and the prompt constants —
including the retrieval/task/prompt-wrapper serialization when Skill Review
actually contains an agent-session row, without repricing API-only panels —
governance-document CONTENTS (BIBLE.md, CHECKLISTS.md, ARCHITECTURE.md) are
deliberately outside it, so editing those documents neither lapses recorded
verdicts nor frees replays. The accepted trade-off is that an old verdict can
replay under amended governance text; this keeps routine documentation
maintenance from repricing every recorded review.

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

`scripts/run_external_review.py --contributor` is maintainer-grade
large-window tooling that produces structured review evidence (its scope
reviewer's required-artifact pack is independent of diff size and can exceed
a default install's scope window — see CONTRIBUTING for the budget shape). It preserves and freezes the machine's
configured `api_chat` and `agent_session` triad/scope rows, then binds each row
to its dispatched prompt receipt and observed response receipt. The shareable
packet records exact base/head/tree/diff hashes, route/model/profile facts,
terminal settlement and capability-delta facts, telemetry limitations, and
full redacted agent-session transcripts. Missing, tampered, drifted,
unprovable, or contradictory identity/terminal receipts make the packet
`INCOMPLETE`. Non-identity capability deltas remain explicit degradation
evidence and do not override the production actor-status/quorum result. A
proposal changing this review substrate still requires a trusted-target rerun.

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
version and `version_carrier_desyncs()` verifies the file carriers (the history row is pinned by the packaging-sync test); changelog prose
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
- [ ] Module stays near one context window (~1000 lines target; exact-path 1600 hard-gate debt is checked in, stale entries fail the official-CI `size_ratchet` lane and warn locally, and new/re-entered 1001-1500 paths carry a rationale)
- [ ] No non-grandfathered Python function or method exceeds the 300-line hard gate (`FUNCTION_DEBT` exact `(path, qualname)` keys are the exception SSOT); methods above 150 lines trigger decomposition review
- [ ] Total Python function count stays under `ouroboros/review.py::MAX_TOTAL_FUNCTIONS` (enforced by the official-CI `size_ratchet` lane, warned locally like the other size gates; bump with a comment if a feature requires more headroom)
- [ ] More than eight parameters is a decomposition signal; consider a typed context object, but do not claim a hard gate or mark existing baseline debt noncompliant
- [ ] No gratuitous abstract layers (Bible P7)

#### Structural Rules
- [ ] New Tool? `get_tools()` exports it using the `ToolEntry` pattern from `registry.py`, an explicit entry is added to `ouroboros/safety.py::TOOL_POLICY` (`POLICY_SKIP` for trusted built-ins, `POLICY_CHECK` for opaque or outward-facing ones), and the intended capability class is declared in `ouroboros/tool_capabilities.py` (`CORE_TOOL_NAMES`, local-readonly/acting child profiles, parallel/truncation sets as appropriate). Ordinary top-level tasks share the registered built-in surface; add a tool to a child profile only when that narrower principal should receive it, and test schema plus execution behavior rather than mirroring names into another catalog. Without the policy entry the tool falls through to `DEFAULT_POLICY = POLICY_CHECK` and pays a light-model LLM call per invocation. **A tool that WRITES the repo working tree needs the GUARD surfaces too, not only the visibility ones:** add it to `_ROOT_ARG_REPO_WRITE_TOOLS` (the single set behind the acting-no-workspace fence, the protected-write gate and the acting root-enum narrowing) and make sure its target paths are canonicalized — via `_PATH_NORMALIZED_TOOLS` if it takes a top-level `path`, or via `canonical_repo_relative_path` + `_payload_write_paths` if its paths ride inside the payload. Visibility checks can all be green while these are missing, so tests must exercise the real guard chain, not only a mocked resolver.
- [ ] New Gateway (if extracted)? Contains no business logic, only transport.
- [ ] New memory/data files? Should they appear in LLM context (`context.py`)?

#### Skill Repair Task Constraints
- Skill repair tasks use structured `task_constraint.mode="skill_repair"`, not prompt markers.
- In repair mode, edit paths are payload-relative: `plugin.py` means the selected `data/skills/{external,clawhub,ouroboroshub}/<skill>/plugin.py`.
- Use `edit_text` for one exact replacement and `write_file` only for new files or intentional full rewrites with `root=skill_payload`. (`edit_batch`/`apply_patch` are repo-lane tools and do not accept `root=skill_payload`.)
- Finish repair with `skill_preflight` and `skill_review`; grants and enablement stay owner-controlled.
- Repair mode is a stricter UI lane, not the only path for skill authoring. In every runtime mode, ordinary top-level tasks may mutate an exact user-managed payload via `root=skill_payload`, `bucket`, and `skill_name`; `skill_payload_binding.py` projects an existing physical `data/skills/native/<skill>` without `.seed-origin` as logical `external` while retaining its physical confinement. Marker-present launcher seeds, `data/state/skills/*`, marketplace/provenance/dependency sidecars, and direct `run_command` writes to repo targets remain blocked. The legacy constrained `skill_repair` selector stays limited to `{external,clawhub,ouroboroshub}`.
- The direct `operator_control` and `local_readonly_subagent` profiles may inspect a selected native payload with `read`/`list`/`search`, including ordinary payload markers and dependency directories. This is a read-only binding exception: native mutation, owner state, grants/review/enablement, acting-child selection, and the constrained `skill_repair` lane remain closed.
- New path checks for skill edits must use `ouroboros.contracts.skill_payload_policy` rather than reimplementing bucket/path traversal logic in each tool.

#### Native-Risk Extension Dispatch
- `type: extension` skills with reviewed isolated dependency envs must not import `plugin.py` or execute handlers inside `server.py`, even when the dependency tree looks pure-Python. Payload-native marker files (`.so`, `.dylib`, `.dll`, `.pyd`) also force child dispatch as defense in depth, but opaque native payloads remain subject to the skill-review checklist and are not newly allowed by this runtime fallback.
- Keep the split explicit: no-dependency pure-Python extensions may use `extension_loader`'s in-process PluginAPI path; isolated-dep/native-marker extensions are cataloged and dispatched by `extension_process_runner` short-lived child processes.
- Tool, HTTP route, and WebSocket handler proxies must return normal tool errors / HTTP 502 / WS log messages on child crash, invalid JSON, timeout, or abort. A child `SIGABRT` is a handled extension failure, not a server crash.
- Child processes must use scrubbed env, per-skill grants, per-skill isolated deps, process-group tracking, output caps, and timeout cleanup. Do not add fallback code that imports native-risk plugin modules in the host process.

#### Task Contract Resource Policy
- When a task contract declares `resource_policy.protected_artifacts`, enforce it as a typed affordance policy in every runtime mode: execute-only black-box references may be run, but byte reads, copy/hash/static introspection, tracing, and debugging against declared paths are blocked.
- Observable Acceptance Claims are bounded, advisory, task-general criteria (`id`, `claim`, `surface`, `support`, `priority`). `success_criteria` is an input alias, not a second persisted carrier. `effective_acceptance_claims` is the only binder: ingress-contract claims win, otherwise the current closed plan wave's frozen claims apply at read time; neither path mutates the live contract. A child receives only claims explicitly passed to its own `schedule_subagent` call. Reviewer `evidence_refs` resolve by exact membership in the already-built host packet, without fuzzy matching, filesystem reads, or re-execution; a claim reference certifies clean only through a passing host-attested support row. Non-passing receipts, agent prose, expected-support text, and unattested references remain named but non-resolving. Resolution changes the clean bit and its disclosure, not actor parsing, quorum, or verdict. Do not turn claims into a hard acceptance gate or surface-specific taxonomy.

#### Skill-defined Presence
- Keep behavior portable and authority installation-local. A reviewed `presence:` profile may declare instructions, context topics, bounded `main`/`light` runtime defaults, and conceptual tool/script/resource requests. It must not name provider credentials, room ids, or assume one installed tool spelling. `presence_capabilities.py` stores the owner's exact selections outside the payload and fingerprints the request semantics that authorize each selection.
- Presence authority is a positive immutable ceiling, not a denylist or a prompt promise. Admission requires the owner-created binding plus an installed, enabled, freshly executable behavior skill and every required selection; it then freezes skill/profile/state/selection fingerprints, exact tool and resource grants, argument bindings, runtime slot, and round limit into `task_contract.capability_ceiling`. Registry schema discovery and execution must enforce the same ceiling for built-ins, extensions, MCP tools, scripts, and resource roots.
- `state/presence_bindings.json` is host-owned authority. A transport token may resolve only bindings naming that exact transport skill, and the submitted provider/account/conversation/thread must match the binding origin. Transport payloads carry structured actor, conversation, message, and source-event facts; never recover those identities from message text. Staged files stay inside the calling skill's state root before entering the ordinary attachment store.
- Run each admitted event with a fresh agent, a deterministic binding-plus-source-event task id, a cross-process installation-wide concurrency gate, and per-conversation serialization. The transport's durable provider custody owns arrival FIFO before Host admission. Dialogue uses the ordinary history, memory, consolidation, and task-result owners with exact transport provenance; `chat_history` may narrow the live-plus-archive timeline by exact provider/account/conversation/thread/actor/date facts. Do not add a transport-specific task scheduler, memory silo, core terminal outbox, or resident cross-room agent.
- Presence completion is exactly `message`, `silent`, `tool_delivered`, or `deferred`. A deferred result requires a successfully promoted `work_ref`; correlated lookup stays behind the same transport token and binding instead of exposing the general task API, and `presence_cancel_work` additionally requires the current binding and conversation to match. Owner chat and Background Consciousness may initiate only an existing reviewed binding, and an initiation is delivered only through a selected transport tool. Promotion must clear Project/workspace/source widening and copy the same Presence metadata and capability ceiling by value. `schedule_followup` does the same for one-shot and recurring work. Any new descendant producer must either preserve this ceiling or refuse the transition; reconstructing authority from mutable current state is forbidden.
- Shared autobiography does not mean unlocked shared files. Knowledge topic mutation and index regeneration use one stable lock, and scratchpad block mutation and markdown regeneration use one stable lock, so concurrent owner and Presence turns cannot overwrite a newer projection with an older render. An exhausted companion persists its terminal failure in existing skill health before leaving the live process snapshot; a successful new start clears the matching failure.
- Test the boundary at both layers: strict profile/state/ceiling parsing, stale/missing review admission, schema and direct-execution filtering, argument binding, binding/token/origin checks, event idempotency and conversation ordering, typed outcomes, late-work correlation, and promotion/follow-up inheritance. Provider adapter E2E is separate evidence and must not be inferred from core tests.

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
- The logical `root=deliverables` tool remains read/list/search-only for orchestrator inspection. A configured physical Deliverables destination remains reachable through existing top-level `user_files`-authorized paths; the new shell seam is narrow, does not grant that logical root to a child, and the normal declared-output/undeclared-output custody flow still applies.
- For argv-visible targets, the shell guard checks lexical Deliverables origin before generic workspace or executor roots, then checks the symlink-resolved destination. Hidden, credential-like, protected, and symlink-escaping descendants therefore do not inherit a broader root's admission; direct `cp`/`mv`/`ln` directory destinations derive their immediate child target (including attached `-tDIR`/`-Ssuffix` forms, `cp --parents`, and `cp -s`/`--symbolic-link` symlink creation). `ln --relative` resolves its source from the command cwd before checking the resulting payload. Recursive directory/archive copies are not walked for nested symlinks or hidden descendants. Declared-output custody and Presence ceilings use the same target-first rule, and Presence keeps its logical `user_files` prefix across the physical Deliverables remap. The undeclared-output audit remains best-effort rather than a full shell parser; relative writes after an in-command `cd`, shell-variable/indirect destinations, and arbitrary inline-code path construction remain deferred parser residuals. Declared outputs still use normal custody, while a successful dynamic undeclared write may lack the nudge. Inode aliases (hardlinks) remain a disclosed filesystem residual rather than a new Deliverables authority check.
- `run_command`/`run_script` `scratch=[...]` (v6.52.2) is a DISTINCT channel from `outputs=[...]`: it declares EPHEMERAL in-workspace verification files (a throwaway test the agent writes, runs, and deletes — e.g. an in-package test that must live in the repo to compile). Scratch is exempt from the undeclared-output guard, never registered as an artifact, confined to the cwd, honored for NEW files and (v6.56.0) for ADOPTED existing untracked in-cwd files — adoption records the file's sha at declaration time through the SSOT `artifacts.record_task_scratch`, so the patch exclusion applies only while the content still matches (tracked files, paths outside the cwd, and paths outside a git worktree stay blocked; a real edit can never hide behind a scratch declaration) — and excluded from the workspace patch via `.scratch_manifest.json` (`headless.write_workspace_patch_artifacts`). Re-declaring a manifest path is idempotent. The undeclared-output guard verifies candidates POST-exec by stat (exists + mtime ≥ start−slack), so a mere path MENTION (import strings, CLI flags, heredoc bodies) is not a write. Use `outputs` for deliverables, `scratch` for throwaway verification — never overload one for the other.
- `run_command`/`run_script`/`start_service` may use cwd under `active_workspace`, explicit `system_repo`, task-scoped `task_drive`, task-scoped `artifact_store`, and external `user_files` where the active profile permits it. Omitted cwd consistently selects `active_workspace`; a light direct task that needs writable scratch must therefore select `task_drive` explicitly. Long-running services in light must use an explicit external/task/artifact cwd. Declared service `outputs` are copied into the task artifact store when the service stops.
- `run_script` temporary files are created under the active workspace when the task is workspace/executor-backed, then removed after execution. Do not run workspace scripts from the system repo temp path; relative imports, generated files, and toolchain discovery must observe the same cwd the user requested.
- Declared process outputs may be files or directories. Directory outputs are copied to the canonical artifact store as a bounded manifest plus zip archive; hidden/control/credential-shaped files, excessive file counts, and excessive byte sizes fail closed instead of leaking through artifact registration.
- In external workspace mode, light-mode self-repo dirty checks snapshot the system repo, not the active workspace. Task-local git operations inside the external workspace are allowed when the task requires them; Ouroboros repo/data paths remain structurally protected, and workspace patch artifacts are captured against the preflight git base.
- Project-room promotion with no working folder and no `workspace="none"` opt-out idempotently provisions a standalone git repo through `ensure_project_workspace`, then runs the ordinary workspace admission checks. Never provision over a non-empty broken binding or an unreadable registry; those cases fail loudly. Binding affects tool profile, memory, lease, and preflight, not the Max-mode Architecture projection.
- Keep policy denials separate from execution failures: `user_files_path_blocked`, `cwd_blocked`, and `artifact_output_undeclared` are non-failure outcomes, while failure to register an explicitly declared output remains `artifact_output_error`.
- The DEFAULT (non-workspace) shell lane carries the SAME target-aware git policy in every runtime mode including light (Q4=A sandbox unwind): mutating git is blocked only when it targets the Ouroboros runtime (system repo / any data drive — bidirectional, casefold, symlink-resolved containment; `commit_reviewed` is the remedy for self-repo changes), read-only git works everywhere including at the system repo, `allowed_resources.network=false` still fences network git subcommands, and acting `self_worktree` children keep the strict no-commit policy. `git init`/`commit`/`push` in `~/projects`, `/tmp`, an attached project folder, or a host-minted coop tree is legitimate task work, not a violation.
- `claude_code_edit` is RETIRED (D10, owner-approved migration, phase 6.4): the SDK edit gateway's job moved to the configured session-actor path — `schedule_subagent(subagent_id=...)` freezes a mutating nanny's selected row, and the host pre-starts the exact subscription leaf through the configured `delegate_start` bridge before the nanny's first round. `delegate_wait`/`delegate_answer`/`delegate_cancel` supervise it, and explicit `delegate_start(subagent_id=..., prompt=...)` handles bounded direct or replacement starts. The D10 migration shipped INCOMPLETE for one supported target class — the old gateway could edit an exact non-Git skill payload directly, while the successor knew only Git workspaces — and that class was RESTORED (owner option A, 2026-08-14): a top-level task selects the session transport and exact user-managed payload with `delegate_start(subagent_id=..., prompt=..., root="skill_payload", bucket=..., skill_name=...)`, including a markerless physical native payload through logical `external`; the harness edits a private standalone Git snapshot, and the parent applies the captured diff explicitly under a whole-payload content-hash CAS, after which the existing skill review is stale. The resource fields select authority and never select transport. Compatibility is one-way and permanent: a saved task contract carrying `disabled_tools=["claude_code_edit"]` also withholds the successor `delegate_start` (registry `_disabled_tools`). The Claude runtime itself was later FULLY retired with explicit owner consent (2026-08-29): the `/api/claude-code/*` endpoints, the SDK gateway, the launcher verify step, and the required `claude-agent-sdk` dependency are gone — the api-route advisory's successor is the bounded native inspection episode on the review substrate (`review_native_episode.py`). Do not resurrect the tool name.
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
  Its public schema is strict: `subagent_id`, `objective`, and `expected_output` are required;
  optional public fields are `role`, `context`, `constraints`, `memory_mode`,
  `deadline_at`, `acceptance_claims`, `write_surface`,
  `write_root`, `protected_paths_grant`, `external_tool_grants`,
  `delegation_intent`, `may_mutate`, `may_fan_out`, `max_children`, and
  `required_capabilities`. `schedule_subagent_properties()` owns both this schema
  and the handler's closed keyword set. `effort`, `parent_task_id`, and
  `description` are not public inputs: effort belongs to the selected configured
  row, lineage comes from `ToolContext`, and capability requirements are admission data rather than a
  frozen contract field. Claims must be plain strings; malformed shapes fail with
  a typed argument error, blanks normalize away, and omission means none rather
  than parent inheritance. Child builders re-state claims and the emptied
  `success_criteria` alias after the parent-contract spread. Boolean grants use
  strict parsing, deadlines can only narrow, and `_narrow_child_delegation_budget`
  can only reduce a subagent parent's authority; a root's explicit mutation grant
  still passes through the ordinary runtime checks.
- `delegation_budget.may_delegate=false` is an admission refusal for every new
  descendant. `may_fan_out=false` still permits one direct child, but refuses a
  second and later direct child using the existing parent/child status records;
  it is not a topology mode. If that authority scan is incomplete, its count is
  unknown and only an explicit fan-out/child cap refuses; omitted legacy budget
  flags remain permissive. The existing budget carries additive
  `depth_provenance` facts (`requested_depth`,
  `permitted_depth`, `attempted_depth`, and host-visible `achieved_depth`), where
  an absent explicit root request remains unknown rather than being inferred from
  prose or a vendor's internal children. Persisted permission remains monotonic
  across ordinary Settings changes, while the explicit global depth value `0`
  still refuses every new descendant and the immutable hard ceiling bounds any
  malformed persisted projection. External task ingress and supervisor queue
  admission accept only non-negative typed depths; malformed or negative persisted
  rows are terminalized before assignment rather than clamped.
- `subagent_id` selects one complete row from the canonical enabled
  `OUROBOROS_SUBAGENTS` list. At schedule time, freeze the normalized row and list
  fingerprint into the task; dispatch/restart must use that snapshot rather than
  mutable Settings. An `api_model` row is the recursive API child. An
  `agent_session` row is the recursive nanny bound to one exact external session
  route; selecting the row IS the parent's substrate decision — the host starts
  that leaf before the nanny's first round — and its model/account
  facts come from requested→effective custody evidence. Do not add a
  second model/lane/executor selector to the public schema, parse `recommended_use`,
  rank rows in host code, or substitute another actor after a typed refusal.
- Legacy `model_lane`/`executor` parameters are handler-side compatibility only,
  hidden from schemas. Accept them only when they deterministically map to exactly
  one migrated configured row; new+legacy is a conflict and omitted/ambiguous
  `auto`, zero matches, or multiple matches returns `subagent_selection_required`.
  Historical task/result fields remain readable; do not make them active defaults.
- A configured session child means the work EXECUTES ON THE HARNESS by
  construction: the substrate choice is the PARENT's, made by selecting the row,
  and the host executes that choice — the nanny never re-decides it. The typed
  parent-LLM choice is the floor (truth, money, and authorship stay where the
  parent put them); topology, decomposition, and supervision judgment remain the
  model's ceiling (BIBLE P5/P13 — code executes a typed LLM decision, it does
  not choreograph cognition). `subagent_bootstrap.bootstrap_before_context`
  starts the exact snapshotted leaf BEFORE the first model round through the
  SAME wrapper the model's `delegate_start(prompt="")` call uses
  (`delegate_start_entry`) — one start path, one set of refusal shapes. Branch
  order: recovery adoption first; durable zero-run / unknown-evidence fences
  second (a fence may hide a live prior run, so a fence-wake outranks every
  terminal); dispatch-blocked third; otherwise pre-start. The host NEVER waits
  inside bootstrap: a live run — fresh start or adopted recovery — hands the
  model its first round immediately with a `configured_session_started` receipt
  carrying the run id, and waiting is the model's own `delegate_wait` decision,
  which keeps owner messages, hurry controls, loop checkpoints, and PARALLEL
  auxiliary children (critics, follow-ups) live for the whole run. Do not
  reintroduce a host-side wait, poll, or supervised-wait call on this seam.
  A blocked dispatch (unless fenced) or a DEFINITE start refusal — a typed
  `refused` payload with no custody handle and a reason inside the closed
  `subagent_bootstrap._DEFINITE_UNRUN_REASONS` set or in the
  `access_profile_unsupported` prefix family beside it — ends the child UNRUN
  and typed at $0 through the existing
  `executor_blocked_outcome` (`agent.py` fills `cap_info` from
  `ctx._configured_startup_refusal`); there is never a silent vendor/API
  fallback. Everything ambiguous — any custody handle, `started_uncustodied`,
  an unknown reason code, unparseable output — wakes the model instead: a false
  "spent nothing" terminal over a possibly-live run is the one direction this
  classification must never fail toward. Grow the definite set only with
  reasons that PROVE no run can exist.
  The zero-run receipt remains
  `verify_and_record(contract_kind="delegation_zero_run", zero_run_decision,
  zero_run_basis)`, with the WRITE enum `incomplete | unknown`
  (`ZERO_RUN_WRITE_DECISIONS` in `outcome_receipt_store.py`): a zero-run
  "complete" is unverifiable self-report and stopped being writable. The READ
  enum additionally keeps historical `complete` receipts valid — an old receipt
  still fences a second physical start — but the terminal projection degrades
  them to `unknown` plus disclosure (reason `historical_zero_run_complete`),
  never clean. Prose alone is not a zero-run receipt. Before writing one, the
  host must prove from the canonical custody root that no open run, ambiguous
  start invocation, or undisposed physical result remains. Once durably
  recorded, it is terminal for that actor; a later physical start is refused
  rather than contradicting the receipt. A malformed or unreadable receipt
  store with no still-parseable terminal row is typed unknown and also blocks a
  physical start; the narrow zero-run form remains available to re-ground the
  decision, and child copy-back must preserve rather than rewrite away the
  corrupt evidence. A valid terminal row still wins over an unrelated malformed
  row.
  A session actor's terminal is CLEAN only through a SUCCEEDED delegated run
  (or adoption) on its own physical leaf, or a durable typed zero-run receipt —
  a start merely ACCEPTED is not clean: all-failed runs project an incomplete
  execution axis, unsettled/uncustodied runs project unknown. "Completed direct
  child ⇒ clean" is DELETED: host children are auxiliary evidence — the
  unresolved fact carries `reason=physical_leaf_not_started` plus
  `direct_child_statuses`, and the `CONFIGURED_ACTOR_INCOMPLETE`/
  `CONFIGURED_ACTOR_UNKNOWN` finalization fact fires no matter how much
  coordination activity or how many children the round had. A substrate swap
  onto host API children is a disclosed incomplete execution, never a clean
  one.
  Metered pacing (`nanny_pacing.py`): the burn baseline resets ONLY on real
  acts of delegation (`delegate_start`/`schedule_subagent`); supervision verbs
  (`delegate_wait`/`delegate_answer`/`delegate_cancel`) advance the round
  baseline while dollars keep accumulating; coordination verbs are untracked —
  no meter reset, no separate observation; the unified reminder wording counts
  supervision/coordination rounds toward the burn. `_nanny_route_dispatched`
  covers every configured `agent_session` row as well as `executor="harness"`,
  so the reminders stay armed across mid-run failures.
  Supervision is not a topology state machine: host code must not infer a
  required number or order of descendants. The canonical brief and its hash
  remain unchanged; any coordination appendix is additive and separately
  disclosed. When a physical start or recovery actually occurs, inject the
  existing custody-durable startup/wake receipt. `started_uncustodied` is a
  fault with a possibly live run: do not enter quiet sleep or start a
  replacement until the invocation is proven absent or terminal and any
  captured physical result is explicitly disposed; replay the original pending
  invocation/idempotency key after worker loss.
  A fresh physical start and `delegation_zero_run` are mutually exclusive actor
  decisions. Rebuild all run/start/patch blockers from one custody-log snapshot and
  hold the existing short per-task file-lock seam only across the final recheck plus
  START_REQUESTED/zero-run append (`delegate_start_claims` owns the start side);
  never hold it across transport or waiting.
  Treat supervisor delivery of one `schedule_subagent` event as at-least-once:
  an exact task id with live or durable custody is an idempotent no-op before
  write-surface provisioning, and the same identity check runs again under the
  queue lock immediately before enqueue. Never use semantic duplicate judgement
  as the physical identity fence.
  The complete external work-order wire budget is one total 250,000-character
  limit, not a model-context claim and not a per-field prefix rule. A brief that
  fits is sent byte-complete. A brief above that limit is never silently prefixed:
  the exact-start path may send only a compact `coverage=partial`
  source-request lens when the selected route's live manifest positively
  declares an interactive question channel. The lens carries the full brief SHA/size and an
  actor-resolvable `get_task_result` canonical-work-order selector; the child must
  request exact character ranges through the existing interaction seam before
  substantive work. The reader and validator share one renderer, so the bytes and
  offsets the actor sees are exactly the bytes the host verifies.
  A route whose channel is unavailable or unknown receives a typed source-channel
  refusal before any external start. Pending recovery replays the stored compact
  request body and the full canonical fingerprint, never a fresh reconstruction
  from the oversized task. The source request and its host-verified character
  intervals are part of durable delegate custody and replay. `delegate_answer`
  accepts a typed `source_response`; the host re-renders the canonical complete
  brief and compares the selector, digest, bounds, and exact bytes before recording
  an interval. If the engine answers `already_resolved`, the host records an interval
  only when a durable prior delivery receipt binds that same interaction and exact
  source selector; timeout or another resolution remains incomplete. Until the union covers the whole brief, terminal delivery carries
  `work_order_verification.status=cannot_verify`, and `integrate_delegated_patch`
  may reject the captured result but may not apply it.
  The manifest observation is a point-in-time preflight, not a lease: capability
  may change before the later POST. Never call the probe delivery evidence or add
  a second probe/lease to pretend the race vanished. Durable verified range coverage
  remains the authority; a raced run stays `cannot_verify` and its patch stays
  unapplied until coverage is complete.
- `subagents.route_health` is the ONE route reader for every consumer —
  dispatcher, the nanny's own `delegate_start`, and review slots alike: a
  degraded-status reviewer slot now reaches the engine and receives its typed
  refusal, never a silent api fallback. The harness row's aggregate doctor
  `status` is NOT a refusal: it describes the default credential store while
  real accounts live in the engine's credential-profile pool, so admission
  belongs to the engine — a genuinely empty or exhausted pool answers the start
  POST with its own typed refusal (INV-135 `credential_pool_exhausted` plus the
  earliest reset), which under pre-start costs $0 and zero model rounds. The
  row's `enabled` field IS honored for unpinned routes as `route_disabled`: the
  engine schema defines it as the OWNER's settings toggle, not an observation
  (a pinned profile keeps its historical skip — the pin is itself an explicit
  owner row). The engine's belt capability row (`delegation.available`) is not
  consulted: Ouroboros runs never request the belt. The remaining typed
  refusals are `route_not_in_capability_catalog`, `route_disabled` (unpinned),
  access-profile mismatch, `engine_rejects_delegated_marker`, and positive
  quota exhaustion for the route's own model.
- Any reader that needs quota snapshots plus typed absences must call
  `ClaudexorGateway.quota_state()` once and project both from that envelope.
  The list helpers are compatibility projections, not permission to perform
  two `/v2/quota` reads and mix evidence epochs. Optional absence metadata
  from older engines fails to an empty neutral value.
- The acceptance packet carries a host-attested `substrate_execution` section —
  `actual_substrate`, `delegated_runs_*` counters, zero-run facts — read from
  durable custody rows at packet-build time
  (`delegate_evidence.acceptance_substrate_facts`). VISIBILITY ONLY: zero typed
  rules tie substrate to the verdict — acceptance judges quality, never the
  execution route. An unreadable custody log reads `evidence_read_failed`,
  never a proven-empty substrate.
- `delegate_wait` is an event-only model sleep. Renew bounded transport windows in
  `delegate_supervision` with zero LLM calls; journal progress may stream to the
  owner but is not a wake. Wake only for terminal/interaction/fault, an addressed
  owner/task message, a direct-child attention beacon or terminal transition,
  cancellation/deadline control, recovery judgment, or one explicitly requested
  `checkpoint_after_sec` + `checkpoint_reason`. A real event
  consumes the one-shot checkpoint. Keep durable pending/ack/replay semantics for
  wakes and message/interaction ids. An oversized combined wake must remain valid
  bounded JSON with a hash-verified actor-readable source for the exact full payload;
  if source staging or delivered-payload acknowledgement fails, leave the wake pending
  and replay it rather than advancing the coordination cursor. On wake the nanny retains its full ordinary
  tool surface and inherited parent cognitive route; no-co-building is a
  prompt/review/receipt role contract, not a host allowlist.
  Session-child startup receipts and every newly minted meaningful wake carry one fresh
  `coordination_context`: full parent-authored advisory `intent_note`, explicit
  deadline time remaining, known/partial/unknown tree spend, active host-visible
  descendants and root acceptance capacity. Vendor-internal descendants stay opaque.
  Persist this context inside the pending wake so replay is identical; recompute only
  after acknowledgement on a later wake. When the combined wake spills, preserve the
  complete context in the exact source and keep only a typed bounded projection in the
  envelope. These facts inform the LLM and never become an automatic fan-out, hurry,
  review or stop state machine. Treat active descendants as known only from a fresh
  queue snapshot and targeted live-row ancestry; stale/missing lineage is unknown.
- Recovery is cause-specific. A proven non-signal worker crash and an explicit
  planned-self-restart transaction may adopt the same exact run before orphan
  cleanup/LLM/start. Owner restart, signals, panic, timeout/deadline, explicit
  cancellation and abrupt whole-app loss are no-resume causes. Never generalize this
  into arbitrary `RUNNING` resurrection or a public parked-task state.
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
- Preserve requested/permitted/attempted/achieved depth on every admitted child.
  Root acceptance must summarize those persisted host-visible facts; do not
  recompute historical permission from current Settings when child provenance exists,
  do not fill missing historical permission from mutable live Settings, and do not
  count opaque vendor-internal descendants as host depth. A tool-level over-cap
  attempt must leave a typed durable rejected child result rather than only prose.
- Treat a delegated 404 as scoped evidence. Project 404 discharges registration;
  run 404 after owned-daemon reprovisioning closes custody as unreachable, not
  settled, only after registration retirement and without invented usage/spend.
  For a shared project, the lowest run id owns retirement retries and siblings
  defer quietly until removal or a project 404 completes the obligation.
- `delegation_constraint` is a typed task-tree beacon with a structured payload
  (`constraint_id`, directive, scope, rationale). Consumers must read the payload,
  never parse the text. Overrides require an explicit reason and are recorded as
  decision rows.
- `review_requested` is a typed, advisory task-tree beacon with the closed payload
  `{evidence_ref, evidence_sha256}`. It wakes the waited/direct parent and preserves
  separate typed concerns even when they reference the same bytes, but never starts
  or waits for a reviewer. The full hash remains visible through `tree_read`/`peek_task`.
  The parent/root decides whether to inspect, spawn an ordinary critic, or use the
  root-owned acceptance path. The beacon itself spends no cycle, and its hash remains
  caller-authored until host bytes are actually read and verified. Immediately before
  a real root acceptance transport, strictly claim the complete candidate/evidence/fence
  binding in canonical `task_acceptance_review_accounting` under the root task-result
  lock. Cap check and exact-binding dedupe are one mutation; missing/malformed authority,
  lock failure, or a prior claim without a recoverable terminal run is typed unknown and
  starts no reviewer. Ordinary critic children remain ordinary budgeted tasks, not a role-
  parsed hidden review flow.
- Subagent changes must keep writes, commits, review mutation, runtime control,
  tool expansion, skills lifecycle, and shell blocked — except bounded task-tree
  coordination via `tree_note`/`tree_read`, parent-only
  `override_delegation_constraint`, existing lineage-gated `peek_task`/
  `cancel_task`/`discard_child_result` over the caller's own direct children, and
  bounded media projection such as
  `extract_video_frames` writing derived frames only under the task artifact store
  (`artifact_store/video_frames`) through a host-owned command shape (the permitted
  local coordination/projection paths; not arbitrary workspace or repo mutation).
  Nested
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
  otherwise. `evaluate` JS stays unavailable to `local_readonly_subagent`; a valid
  `acting_subagent` may evaluate JavaScript on its current page while the
  owner/self-lowering guards remain active. `vlm_query` /
  `analyze_screenshot` are available. (Relaxed in v6.24.0 for local UI/build inspection;
  control-plane, private-range, and DNS-rebind denial preserved. Acting children retain
  the pre-existing shell-to-loopback `/ws` route; this change does not add WebSocket
  authentication or broaden that route to local-readonly children. See ARCHITECTURE.md.)
- The canonical/replica terminal post-task/accounting field-custody projection
  must live in one pure reducer reused by both physical copy-back and effective
  reads; never blanket-overlay the replica over canonical truth. Every change
  to that projection must add a stale-replica regression at both seams. The
  same reducer owns the reconciliation-disclosure pair and protects a canonical
  terminal delegation receipt only when its durable `started`/`settled` counts
  do not regress the replica; equal counts may enrich cost/access/substrate,
  while a dispatch-only canonical envelope still accepts the first child
  receipt. Historical top-level `delegated_runs_*` counters are not rewritten.
- Push/live events are wakeups and a fast path, not terminal authority. Durable
  task detail/history and authoritative snapshots must converge terminal UI
  state through the existing refresh/reconnect seams. Shared snapshot consumers
  mutate projections only for a request generation newer than the last applied,
  while the request-start barrier protects later live frames; lifecycle changes must
  exercise lost/reordered terminal frames and reversed snapshot completion.
  History replay projects a durable delegation receipt onto the latest emitted
  terminal progress row even when a separate task summary survives, because
  that progress row is the executor-chip consumer; absent durable evidence
  never erases a receipt already present on the row.
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
  `budget_drive_root`) hold the SAME
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
  boot scan over task_results).   The intent and delivery registries read STRICT:
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
  an exact omitted count; union the canonical and recorded child-drive replicas
  with exact-row de-duplication and stable receipt chronology before and during
  copy-back, so neither a canonical zero-run fact nor a child-local ordinary check
  can hide or erase the other and an older PASS cannot reconcile a newer FAIL.
  `wait_tasks` stays batch-compact:
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
- Cancellation keeps one public queue/lifecycle surface while its code owners
  stay narrow: retry-aware physical-target and subtree-liveness resolution live
  in `supervisor/queue_transitions.py`, capture-miss terminalization/publication
  lives in `supervisor/cancel_publication.py`, and owner-stop control delivery,
  stale-control validation, and deadline narrowing live in
  `supervisor/owner_stop.py` with compatibility re-exports from the loop.
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
  than clean. This is an outcome-honesty rule, not a new wait loop. While the
  gate is open the delivery candidate is HELD
  (`child_absorption_or_revision_required`), never armed: the JSON-only
  delivery-control instruction must not ride the same round as the absorption
  reminder — nor a post-tool evidence change while that hold is active —
  because it would contradict the required disposition tool call. Before the
  first reminder places the hold, no disposition instruction exists yet, so
  the ordinary evidence-change arm still applies there.

#### Page Header Layout
- Top-level page chrome (`renderPageHeader`, tab strips, primary actions) must sit outside the scrolling content region.
- Pages use an outer flex column plus an inner `<page>-scroll` body with `overflow-y:auto`. Skills, Widgets, Settings, and Chat follow this pattern.
- Page icons come from `web/modules/page_icons.js`; do not paste divergent SVGs into individual page modules or the navigation rail.
- Primary page actions, including Refresh, live in the `renderPageHeader({ actionsHtml })` slot on the right. Do not add ad-hoc refresh rows inside scroll bodies.
- Non-chat top-level pages use `.app-page-glass` for the shared dim/brand backdrop. Header padding should stay compact; if a page needs more space, simplify its copy rather than growing the chrome.
- A new top-level page that scrolls its header together with content violates the architecture mirror: fix the layout, not the symptom.
- Top-level tab/pill buttons are a single design-system control: `renderTabStrip` + `.app-tab-strip` + `.app-tab` + the `--pill-*` CSS variables in `web/style.css`. Do not redeclare per-page tab padding, font size, border radius, or active styling in page CSS files.
- Scrollable page bodies use the shared `.scroll-fade-y` mask when content can pass under fixed page chrome. Do not copy/paste custom gradient masks into page modules; extend the shared class if the fade rhythm changes.
- Masonry-style widget packing uses `web/modules/masonry.js::applyMasonry`. Do not reintroduce CSS Grid row packing (`align-items: start`) for unequal-height widget cards; it leaves row gaps under shorter cards. It packs in the page's explicit key order and writes only `--masonry-*` custom properties (the static rules in `web/style.css` apply them); never move `<article>` nodes to reorder — a moved `<iframe>` reloads.
- Widget card ordering and the owner's per-card launch-policy override (`widget_start_mode`, values from `extension_ui_validation.WIDGET_START_MODES`) are host UI preferences. Persist them through `/api/ui/preferences` and `data/state/ui_preferences.json`; never rewrite extension manifests or widget declarations to store owner layout or owner overrides.
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
- One capability, one section. The whole task-actor story lives in Agents →
  **Available subagents** (`web/modules/subagents_settings.js`), beside but not
  inside Review lanes. It edits one canonical `OUROBOROS_SUBAGENTS` object with a
  list-level Enabled flag and at most ten stable rows. The UI numbers rows and asks
  the owner for one prose field, Description (`recommended_use`); id and compatibility
  name stay automatic and hidden, while API-model or Agent-session route, optional
  effort and optional session account pin remain structured controls. Never derive
  durable identity from the visual ordinal: removing a preceding row must not change
  task snapshots or receipts. The write permission, active/depth counts and path roots
  remain orthogonal controls in that section; `OUROBOROS_MAX_WORKERS` stays in Advanced
  because it sizes the process pool. Never render a second control over the same
  settings key.
  Share only neutral route/model/account/effort/status primitives with reviewer
  rows (`route_editor_primitives.js`). Preserve the two public schemas: task API
  routes serialize `api_model`, reviewer routes `api_chat`; task pins serialize
  `credential_profile_id`, reviewer pins `profile_id`. Empty session pin means the
  engine's compatible-account rotation. Saved choices absent from discovery remain
  visible and editable, marked unavailable/not checked according to the exact
  catalog/accounts facet. A Cursor/Agy compound effort slug plus a conflicting
  separate effort is a validation error, never two applied efforts.
- Saved intent, generated drafts and live status are different axes. A status or
  catalog failure annotates a loaded row and must never erase it. GET may return an
  unsaved migration/default candidate; only explicit Settings Save or onboarding
  completion materializes it. A late preview/status result may update a clean
  generated baseline, not absorb owner edits or drop focus/caret. Generic Settings
  validates/canonicalizes the draft before the existing serialized off-loop owner
  transaction and reports that active tasks keep their starting snapshot.
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
- Agent sign-in consumes the exact harness row's optional-without-default
  `setupLogin` field as four states: absent is legacy, null delegates support
  to the exact pinned engine's typed setup/profile admission, a valid object
  selects `in_app` or `external_terminal`, and malformed present data is a gap.
  Only absence may consult the old global operations catalog. The null path is
  stamped `setup_job_admission`: an omitted transport stays omitted and an
  explicit `client_pty` stays exact. Never add a harness-name branch for this
  choice. Typed required-profile and duplicate codes decide their respective
  flows; only an old generic 409 (`internal_error` on 3.6.0, or `http_409` when
  no body code survived) may use an exact same-harness/same-profile read-back.
  Project external-terminal recovery only from the exact required action or
  durable native-command error code, and start its new job through the existing
  custody-release guard. Before profile registration or job creation, bind the
  recovery argv to the live handshake's exact engine version, build SHA and
  absolute entry, locate its preserved exact-Node runtime without consulting
  the staged next-spawn pin, and require that same entry's fresh `--probe` to
  advertise the additive `setup_attach` role. A missing role on an old probe is
  a typed 409 with no job; a failed/identity-mismatched probe is a retryable
  typed 503. Retain that argv through job creation, then render it in full and
  compact through the owning `claudexor_daemon.py` consumer for POSIX or PowerShell
  target, with owned `CLAUDEXOR_CONFIG_DIR` and an empty inherited
  `CLAUDEXOR_DAEMON_SOCK`; label the shell, and do not execute the text.
- Credential-profile DELETE remains a thin receipt-preserving proxy. Frontend
  contracts require the daemon's `profile`, `removed` and exact
  `credentialCleanup` (`config_dir_removed | secret_deleted | none`), with
  `cleanupWarning` and vendor disposition optional. Frontend
  code may show the retained-vendor warning only for the exact
  `vendorCredentialDisposition` tuple `vendor/left_unchanged/os_user`;
  `verification=not_run` is neutral unknown while `failed` is an error. Mirror
  additive response fields in Python TypedDicts and `web/modules/api_types.js`,
  and extend the field-parity plus focused Python/Node fixtures together.
- Keep install compilation linear and split by semantic owner. Available
  subagents include every supported connected task harness (Claude, Codex,
  Cursor, Agy) plus truthful API/local Main/Light actors. Agy's generated row is
  unpinned `gemini-3.7-flash-high`. Reviewer defaults independently consume only
  ratified Claude/Codex/Cursor policies; Agy-only emits no structured reviewer
  override, and mixed reviewer bytes must equal the core subset alone. On the
  fresh-install path reviewer slots are `subagent_id` references into the
  roster the preset ships (unmatched seats mint `review-<harness>` rows);
  an owner-configured roster is validate-only and its seats stay inline. API-only
  and local-only actor compilation performs zero Claudexor reads. With one session
  and credentials, emit the Light-derived Fast scout and a distinct Main
  Independent perspective when real; never fabricate diversity or build a
  harness/account/model powerset. `POST /api/onboarding/subagents/preview` is the
  read-only compiler surface for the open wizard draft, while completion commits
  its visible owner-edited value and independent reviewer disposition together.
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
- A control the owner cannot use is worse than none. With no agent subscription,
  Available subagents still shows truthful configured/generated API or local actors
  and the session chooser points at Accounts instead of inventing a route. A saved
  unavailable session remains visible; dispatch returns its typed refusal and never
  falls back to an API child. Harness lists come from the accounts panel's own
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
- [ ] Keep canonical messages/tools provider-neutral and function-shaped. A provider dialect is an outbound physical projection and inbound normalization only; it must not mutate stored history or create a second compaction/replay contract. Direct OpenAI Chat custom-origin arguments carry private parser-issued receipts bound to the exact physical catalog/full schema. Main, Background Consciousness, and structured compaction must consume those receipts before execution; never persist them in public usage or observability.
- [ ] Use the one same-route request-wire driver for optional-parameter, reasoning-carrier, and registered tool-dialect recovery. Key evidence by the exact credential-free route and relevant request predicate; compose only typed actions; require semantic success bound to the settled physical attempt before a write; keep the shared TTL and malformed-state fail-open behavior. Never turn provider prose into a model/provider/API switch, never raise a caller's attempt rail, and never persist a task-local explicit-`none` fallback.
- [ ] `usage.request_wire` describes the terminal candidate returned by one LLM call. Nested aggregation preserves those terminal disclosures in ordered `request_wire_history` with explicit omission accounting; it does not copy failed physical sends or replace `state/usage_attempts.jsonl`. Keep these evidence domains distinct in names, docs, and tests.
- [ ] Official direct OpenAI Chat sends `max_completion_tokens` and the requested `reasoning_effort` for the provider route as a whole, not for a hand-maintained model-name allowlist. Every eligible function-tool/non-`none` call begins custom+same effort; generic repairs stay on that rung; exact custom rejection creates fresh function+the original effort; exact function rejection may create task-local function+`none`. Keep ordinals fixed at 1/2/3 and never persist custom→function as learned dialect evidence. The last rung exists only within remaining physical-attempt authority. There is no Responses migration in this contract.
- [ ] Preserve exact direct-Anthropic native assistant content only as private unfinished-turn custody: same-route replay must include the complete original block list/order and every opaque member; cross-route send, summarizer input, and public observability must scrub values. The active assistant/tool-result unit cannot compact until a later successful assistant response consumes it. Owner `none` is `thinking.type=disabled`; do not guess legacy manual-thinking budgets.
- [ ] Every core-mediated physical provider send goes through `usage_accounting.execute_physical_attempt[_async]`: reserve, mark dispatched, then settle/unresolve. A marked dispatch may be released only through a typed pre-dispatch connection/pool failure that proves no request bytes were sent; an ordinary timeout or unknown error remains unresolved. A transport retry is a new attempt. `llm_usage`, state, and UI counters are projections carrying attempt ids, never a second monetary authority. Provider tier pricing and any empirical tokenizer margin affect only a known reservation; settlement prefers actual provider usage/cost. Unknown price reserves `None`, remains nullable in usage events, and never blocks a model merely because its tariff is unavailable. An external skill with granted model-provider credentials is explicitly unknown/unmetered when it bypasses core transport—not `$0`; an ordinary spawned process must not be mislabeled as monetary work.
- [ ] Custody classifiers must not treat Python's implicit exception `__context__` as transport provenance: a fallback raised inside a prior provider's `except` block inherits that prior attempt even after its own request was dispatched. Use the explicit `__cause__` chain or typed transport metadata only; an ambiguous timeout remains unresolved.
- [ ] The low-level `call_llm_with_retry` seam treats an omitted transport reserve as the raw owner-deadline window; callers that own a finalization reserve must pass it explicitly so admission and the transport bound cannot disagree.
- [ ] Hold the usage-ledger cross-process lock only for budget check, validated append, and fsync. Never hold it over network I/O. Preserve a paid response if settlement persistence fails and leave an honest dispatched/unresolved bound.
- [ ] **Tree-spend visibility.** Under a root cap, pacing and stop text use root-subtree ledger spend including in-flight holds; own cost is diagnostic, and unavailable remains unknown rather than `$0`. Reuse `usage_accounting.last_root_accounting` and refresh only at rare cache-breaking/explicitly stale decision surfaces, never by an unconditional per-round ledger scan or inside a stable cached prefix. `task_pacing.resolve_cost_ceiling` returns `disabled|active|exhausted_soft_land|unknown` from the independent global-percentage and root-cap-minus-absolute-margin axes; graceful finalization precedes, but cannot bypass, the ledger fence. `resolve_deciding_spend` is the sole fallback seam and must label own-cost-under-root-cap as a lower bound.
- [ ] Before dispatching any post-task consolidation or synthesis worker, read `usage_breakdown` once for the whole root subtree and pass the same loop-local snapshot to summary and reflection. It is explicitly non-final (`cost_final=false`, `cost_with_children_partial=true`) and carries child-inclusive accounted cost, reservations, unresolved upper bound, unknown/unmetered count, ledger integrity, and capture time. A read failure is unavailable/null, never `$0`. Consolidation, summary, and reflection model spend belongs only to the existing terminal checkpoint; do not add another ledger or reconciliation LLM call.
- [ ] Runtime notices after the first user/assistant/tool turn are user notices, not new `role=system` messages. `LLMClient` defensively demotes non-leading system messages at the provider boundary; source call-sites should still append `[SYSTEM NOTICE]` user turns so provider payloads, local templates, and prompt authority stay consistent.
- [ ] Keep stable policy/governance first and dynamic evidence last. Prompt-cache support is deliberately narrow: direct OpenAI `prompt_cache_key`, OpenRouter `session_id` (or a caller-declared `cache_affinity` for surfaces whose rounds repeat with changing evidence, e.g. review), and one exact retry without the named parameter only when the provider explicitly rejects that parameter. Do not add provider hops, body rerouting, or a generic cache/retry framework.
- [ ] **Cache-friendliness invariant.** Keep byte-stable governance and task contracts before mutable evidence; never place timestamps, hashes, counters, or task identity in a stable cached prefix. Builders declare bare breakpoints and `review_substrate.assert_cache_breakpoint_cap` keeps the declared count at four or fewer. Only `LLMClient._normalize_payload_cache_ttl` finalizes the assembled wire payload: it supplies a missing tools marker where supported, legalizes TTL order, and discloses any reduction. The owner setting `OUROBOROS_PROMPT_CACHE_TTL=default|5m|1h` stamps existing Anthropic-family markers at that send boundary and never creates new ones; non-Anthropic wire stays unchanged. Preserve cache-affinity keys and exact review bindings.
- [ ] OpenRouter reasoning continuity belongs to OpenRouter conversations only. Direct/local payloads strip OpenRouter round-trip metadata. Provider fallback is disabled only when the transcript carries a SEALED reasoning artifact (`ouroboros/reasoning_artifacts.py::transcript_has_sealed_reasoning`: an opaque or signed shape not vouched by the signed-portable roster), because only a sealed artifact is bound to the endpoint that minted it; readable reasoning artifacts stay failover-eligible for every family, and the same predicate governs preserve-vs-strip on a same-model reroute.
- [ ] Delegated agent sessions and the native review inspection episode must preserve the full governance prompt; do not truncate BIBLE/ARCHITECTURE/DEVELOPMENT/CHECKLISTS to avoid argv or transport limits.
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
- [ ] New Skill Review waves attribute every canonical usage row with the exact
  `review_skill`, `review_wave_id`, and stable `review_slot_id`. API attempts inherit
  these fields through `UsageScope`; agent sessions persist them, plus review
  category/source, in the existing `RunCustody` start/replay facts so restart and late
  settlement pass them explicitly to `record_subscription_session`. The history row's
  `physical_attempt_v1` marker only declares that this join key exists; lazy detail
  projects the same `usage_attempts.jsonl` rows and persists no totals. Pre-marker waves
  stay “exact attribution unavailable” and must never be reconstructed by time/model.
  If a terminal row lands before a late worker dispatches, readers overlay only the exact
  same-wave write-ahead marker; an idempotent retry must not clear that marker until its
  facts are already durable in the raw append-only row.
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
  instead of burning rounds on identical calls. A typed 408/429/5xx or a failure
  proven pre-dispatch may retry; a dispatched request with no terminal provider
  outcome must stop same-model and cross-model sends until it is reconciled.

#### Timeout & Wait Control
- [ ] For a session nanny, `delegate_wait` is **event-only** at the model surface.
  Host supervision renews bounded transport windows while the run is non-terminal;
  ordinary journal progress streams to the owner and advances the cursor but neither
  returns to the model nor ends sleep. Only terminal/interaction/fault, addressed
  task/owner message, direct-child attention/terminal state, control/recovery
  judgment, or a model-requested one-shot checkpoint wakes it. A quiet
  transport-window expiry is an internal renewal with
  zero LLM calls. Do not reintroduce caller-visible `wait_sec`, repeating timers,
  progress wakes, or a host semantic stall detector. Reviews and ordinary task waits
  keep their own existing progress-aware/re-decidable contracts.
- [ ] The wait/continue/stop decision must be a **structured fact** — terminal
  status plus heartbeat freshness from `queue_snapshot.json` — not a keyword or
  regex over content (Bible P5). Use `task_status.py` terminal-status helpers and
  the supervisor heartbeat, not string matching.
- [ ] Fixed **kill**-timeouts (hard task/tool ceilings, watchdog) still exist as
  the outer safety bound and get sensible ceilings under high-reasoning models;
  progress-aware waiting tunes the *passive* wait, it does not remove the watchdog.
- [ ] Keep transport dead-socket bounds, task deadlines, and cognitive in-flight
  state separate. Each main-loop LLM call emits exact-attempt `started` and
  `finished | failed` facts; while that row is active it spares only the task idle
  rail. Silent elapsed reasoning is not semantic stall evidence. Explicit deadline,
  budget, cancellation, and the absolute task ceiling still cut through, and a
  terminal fact must match task-attempt plus execution/round/call identity before
  it clears the row. Do not reuse `external_wait_lease`, a transport timeout, or a
  heartbeat as this authority, and do not add an elapsed-time stall detector.
- [ ] New numeric timeout constants are an SSOT in `config.py` `SETTINGS_DEFAULTS`
  with a getter and env registration; do not scatter magic wait numbers across
  call sites.
- [ ] Cognitive waits use separate axes. A transport timeout only bounds a dead
  socket; it is not a reasoning cutoff. The shared no-proxy LLM transport bound
  is `OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC` (2700s), while review slots,
  plan/acceptance wrappers, web-search attempts, delegated polling, and VLM
  calls use their own logical deadline or provider-specific transport setting.
  An explicit slot deadline narrows the route-owned bound. API review uses its
  transport bound as a settlement fallback because that request ends there;
  a delegated agent session instead inherits the existing task absolute ceiling
  because the paid engine run can outlive an HTTP read. The owner deadline always
  narrows either route. A caller/task deadline always narrows nested waits, and Anthropic (120s) plus
  VLM captioning (90s) retain their separate provider transport defaults as
  ceilings, not promises to run past the owner deadline. Delegated review uses
  an opt-in strict poll bound for the remaining logical window; the general
  delegate-wait floor remains unchanged for its existing transport contract.
  Default reviewer slots intentionally have no short cognition cap. The outer
  `plan_task` ToolEntry envelope covers `max(transport + 2 × grace,
  task-absolute-ceiling + grace)`, so an `agent_session` worker can settle
  inside the same existing task lifetime instead of being abandoned by an
  API-only wrapper. The non-delegated Claude advisory child follows the same
  owner-narrowed process bound; with no owner deadline it keeps its 900-second
  child-safety ceiling.
- [ ] A paid process-local review belongs to the exact existing process-custody
  identity stamped in its commit-attempt row: server session plus pid. A new
  Agent, a sibling worker boot, task return, heartbeat silence, or elapsed time
  is not owner death. Settle tokenless rows only after an existing supervisor
  teardown seam confirms that pid is dead, or when a later server generation
  observes the prior-session pid already dead; retain delegated rows carrying
  their durable invocation token, and leave legacy ownerless rows fail-closed.
  Do not add a second process ledger or use TTL as resend authority.
- [ ] Without an owner deadline, `web_search` sizes its outer ToolEntry
  envelope for the complete configured paid cascade: two safe OpenAI attempts,
  one OpenRouter attempt, one Anthropic attempt, and finalization grace. With
  all three provider routes available at the defaults, that is
  `4 × 480s + 120s = 2040s` (about 34 minutes). An owner deadline is narrower
  authority: the outer wait and every provider leg recompute against the
  remaining window minus finalization reserve, so the no-deadline envelope is
  not promised under a shorter deadline.
- [ ] Nested process wrappers are ordered, never tied: the provider bound must
  settle before its killable child, and the child must settle before the generic
  ToolEntry envelope. Explicit VLM helpers reuse the fixed structural settlement
  margin from `config.py` (`provider`, `provider + margin`, and a ToolEntry
  minimum of `provider + 2 × margin`) and subtract the whole hierarchy from the
  owner window before dispatch. The global owner tool-timeout setting may widen
  that outer envelope; the margin is not a user-facing cognition timeout.
  Send-time captions have no child wrapper and keep the direct 90-second provider
  contract.
- [ ] Every physical LLM/review/VLM/tool operation that can outlive a logical
  wait emits a typed `cognitive_operation` start and terminal fact. The
  supervisor uses the active-operation map only to spare the idle rail; the
  task deadline, budget, cancellation and absolute ceiling still cut through.
  A logical timeout with a live worker is custody/reconciliation-pending, never
  permission for a blind paid retry. Late review results settle the original
  attempt and remain bound to its retry identity. Once the owner deadline minus
  finalization reserve is spent, an unstarted review row is a typed `$0
  not_dispatched` actor: no worker, paid stamp, or active lease is created.
  An already-paid in-flight review wave remains eligible for exact custody
  reconciliation after that deadline; this settlement path does not authorize
  a new dispatch or extend the task's cognition window.
  A commit attempt cannot treat an in-flight reviewer as a final quorum verdict,
  including under advisory enforcement. Plan review applies the same rule: if a
  paid actor remains in flight, the wave is projected as the existing open
  `DEGRADED` state with `review_late_result_pending`, even when the settled
  responses already meet the arithmetic quorum; the counts remain factual, but
  the wave cannot close until custody settles.
- [ ] A returned provider response (including an empty/incomplete body) or typed
  terminal 408/429/5xx is settled and may use the surface's bounded retry/repair
  rail. During a mixed plan/commit cycle, the typed settled terminal API actor
  remains in the exact cycle's replay roster even when optional physical-attempt
  capture metadata is absent; when that metadata is present it must say `settled`,
  while explicit `reserved` or `released` states remain eligible for a real retry
  rather than becoming sticky replay rows. `dispatched` or `unresolved` states
  without a typed terminal HTTP status stay under the custody-lost/no-resend
  classification; with such a status they are retained as terminal actors for
  same-cycle replay, never as a second physical send. An identical envelope
  never buys a settled actor twice, and a new retry cycle uses a new key. A
  pre-write-ahead route/configuration/admission refusal is a retryable
  `$0 not_dispatched` actor even though the host minted a synthetic operation id;
  a later checkpoint failure cannot erase an already-fired paid stamp.
  Positive `settled`/`dispatched`/`unresolved` capture evidence outranks a
  contradictory synthetic `not_dispatched` label; only `reserved`/`released`
  is pre-dispatch. Reuse the physical-state vocabulary exported by
  `usage_accounting`, never a surface-local copy.
  Carry `physical_attempt_state` and `provider_status_code` through durable plan
  rows and frozen actors. Across one bounded retry rail, retain the strongest
  earlier capture: a later released reservation or budget refusal cannot erase a
  prior dispatch, and any unknown prior outcome monotonically forces no-resend.
  For non-Skill-Review delegated surfaces, a supplied retry token with no valid
  durable invocation is `review_custody_lost` before route/project/POST work; it
  is never reinterpreted as permission for a fresh paid session. Recovery also
  binds the token to the recorded delegated surface, slot, and operation; an API
  row has no durable-token recovery authority without process-local custody.
  A dispatched request whose socket or stream ends without terminal
  provider evidence is `provider_outcome_unknown`: THAT request is never
  resent by any route (same-model, fallback, provider, local-server, or
  forced-final) — its `unresolved` ledger row is terminal and never settles.
  A NEW logical request is legal only when it carries a unique host-attested
  input absent from the unknown request (e.g. a delegated-leaf wake receipt,
  the nanny-leaf hold contract in `ouroboros/delegate_hold.py`).
- [ ] A reviewed mutative wrapper must retain foreground custody until the
  workflow settles. Inner phase bounds and the task/supervisor absolute deadline
  are the stop axes; never use the global 600s tool default or a separately
  guessed hard ceiling to abandon a still-live reviewer or commit pipeline.
- [ ] A custody retry key names semantic material and an admitted cycle, not its
  rendered prompt. Prior-round/history scaffolding may change while the same
  physical operation is settling and must still join that operation; changed
  snapshots, owner intent, route/model rows, or a genuinely new review cycle
  must mint a new key. Use the canonical staged tree/parent binding for commit
  review, pass the key immutably to every row, and do not admit the next paid
  plan-review cycle while the previous cycle is still in flight. Plan-cycle
  reconciliation freezes its originally dispatched rows and `$0` skip rows instead
  of re-running live health/fit admission; reviewer-requested evidence advances
  the next envelope only after every actor in the current cycle is terminal.
  Skill Review keys additionally bind the exact skill, lifecycle wave, content,
  panel/rebuttal contract, and frozen chunk digest/index so concurrent waves or
  oversized chunks cannot join one another; they remain process-local custody,
  not a promise of restart recovery without a durable invocation token.
- [ ] Commit review writes and rereads `paid=True`, the exact nonempty retry
  key, and both complete slot rosters with reserved operation ids in one locked
  write before either parallel surface starts, except when the owner window
  already has no dispatch capacity after finalization reserve: that prepared
  roster remains an unpaid `$0` wave and no paid stamp is fired. A delegated row must add its
  `pending_invocation_id` to that exact reservation after `START_REQUESTED` and
  before POST. Exact resume keeps the durable actor roster. Deterministic packet reassembly may reconstruct unchanged executor
  inputs, but cannot erase a frozen row, admit an unmatched paid row, or replace
  a pending row with a different `slot_id`/`operation_id`. Delegated pending rows
  carry the durable invocation token; API rows without process custody stay unresolved.
- [ ] Cooperative cancellation is used where the existing route supports it
  (delegated sessions); API/thread routes disclose an in-flight custody state
  until their physical result settles. Do not replace this with a keyword or
  model-name heuristic, a second scheduler, or a new global timing ledger.
  A typed transport failure after a delegated run has an id is an unknown
  outcome: retain the durable invocation token and replay that started run on
  the permitted retry instead of posting a second paid run. A late tool worker
  closes its own cognitive lease through its completion callback, and a partial
  terminal event that lacks the stored correlation identity cannot close it.
  While the process lives, unknown local custody remains a no-resend tombstone.
  A later process startup is separate causal evidence that a tokenless local
  waiter cannot return: settle that row as a typed paid infrastructure failure,
  while rows with durable delegated tokens stay pending for exact rejoin. This
  allows a new explicit attempt after restart; elapsed TTL alone never does.

#### Loop / State-Machine Changes
- [ ] Changes to `loop.py` or other task state-machine logic include adversarial tests for malformed output, false-completion prevention, replay/log durability, and failure modes — not just the happy path.
- [ ] Audit/checkpoint rounds must not silently reuse the normal final-answer path unless that invariant is explicitly tested and documented.
- [ ] Keep a complete loop-local `DeliveryCandidate` once a substantive answer exists. Record host control exposure as sticky loop-local candidate provenance and inherit it through every replacement. After such an episode, a cleared transient latch must not make a recognizable whole-body control envelope ordinary owner-facing text: valid `keep`/`replace` is decoded by both ordinary and forced resolvers, while recognizable invalid or unknown whole-body control preserves the prior candidate without repair. Do not broaden this lineage rule to no-episode JSON, mixed prose with a trailing object, or truncated fragments. A service round may return `keep`, or `replace` plus the complete replacement answer; allow one repair for malformed control, then preserve the prior complete answer and mark finalization degraded. The "malformed" class includes a protocol object embedded as a balanced trailing JSON object at the end of prose (and any fenced variant after the shared fence-strip) — never honored, never published raw while the latch is ARMED; three disclosed, test-pinned residuals: a mid-prose quotation of the literal stays prose; with the latch OFF prose with a trailing protocol object passes through the ordinary resolver as ordinary text; and a TRUNCATED (unbalanced) trailing protocol fragment passes through the forced resolver as prose even under an armed latch — a fragment is not a parseable object, and containing it would need the substring scanning the rule rejects. A FORCED finalization (budget/round/deadline/provider/children rails) resolves an armed control purely and without retry instead: valid keep/replace is honored, anything malformed preserves the retained candidate with a typed degraded reason, and the protocol JSON itself never reaches chat or the durable result. A service notice alone does not change evidence. Owner messages, tool effects, child results, and verification receipts advance the evidence revision and require fresh delivery/acceptance binding. Finalize task-scoped service outputs/errors before host acceptance and require a complete replacement when that evidence changes; keep the `finally` path as idempotent cleanup only. This control must not bypass verification, acceptance, safety, skill-finalization, deadline, child-handoff, the unconditional `FINAL ANSWER:` latch, or the task-level answer protocol.
- [ ] Every direct child result needs an exact-hash disposition through the existing `tree_note(kind="decision")` tagged payload (`type=child_result_disposition`, child id, `integrated | irrelevant | deferred`, complete-result SHA-256; note text is rationale). One call may instead carry a `children` array of such entries (batch form): each entry is validated exactly like the single form, invalid entries are rejected individually by index while valid ones record. The typed task-tree row is the sole authority; task-result disposition fields are derived reads, never a mirrored write. The join-ledger helper alone validates lineage and current content. Stale or malformed payloads change nothing. `deferred` suppresses only the unchanged reminder and forces an honest degraded/best-effort terminal answer until the item is resolved. Natural completion WINS a late cancellation (owner decision 4=A, 2026-08-11): a child that settled its own completed result keeps it — payload, artifacts, and cost — and the cancel settles as already-settled; discarding a kept result is the parent's separate explicit `discard_child_result`. A cancelled (not completed) child still has its salvageable output preserved on the canonical drive before its bounded scratch is removed. Only a SETTLED `cancelled` status counts as a handled cancellation disposition: a child wedged in the legacy `cancel_requested` STATUS latch is intent, not outcome — it stays visible in the parent's handoff reminder as cancel-pending until custody settles it.
- [ ] Host task acceptance is root-only. Queued/headless/scheduled roots are reviewed in `auto` and `required`; direct eligibility is the union of `outcomes.turn_has_reviewable_effects` and a typed deliverable/criterion. Ordinary read-only tool activity, pure conversation, and meta/routing controls are not reviewed, and child reviews remain advisory. Eligibility must use structured facts, never keywords (Bible P3/P5). For an eligible root under `auto|required`, agent-callable `task_acceptance_review` validates/stores evidence and optional agent disposition but makes zero reviewer calls; it returns `deferred_to_host_acceptance`, `authoritative=false`, and the evidence revision. The call itself never widens eligibility; child and `off` behavior remain unchanged.
- [ ] Before root acceptance, atomically fence new descendants under the queue lock and prove recursive subtree quiescence from the existing task-status SSOT. Split-drive ACK, subtree, and acceptance-timing reads/writes use canonical `budget_drive_root`. Preserve the prior verdict until the replacement is recorded. A revision must explicitly reopen the fence; terminal/degraded outcomes seal it.
- [ ] The host buys one authoritative acceptance panel per PAID IDENTITY: `sha256(candidate_hash + the sorted set of nonempty (obligation_id, disposition, sha256(reason)) tuples)` (owner ratification 2026-08-30, replacing the earlier candidate-hash/evidence-revision/fence binding rule). Only two things mint a new paid panel — a changed candidate answer, or a new nonempty obligation disposition; an empty disposition reason hashes to `""` and buys nothing (mirroring `commit_gate.compute_rebuttal_sha256`). The evidence revision must NOT mint a paid cycle — every cosmetic tool call moves it — and remains stale-packet detection for the supersede paths. A resubmit with an unchanged paid identity replays the recorded verdict for FREE, must NOT re-enter the improvement capsule, and terminalizes with the typed `identical_acceptance_refused` reason. Task-acceptance actors receive one substantive call and at most two physical attempts total. Record transport status, parse status, and valid-response semantic verdict separately, with actor model/provider, role, coverage, panel id, quorum contribution, reason, enforcement impact, and binding hashes. Public task/event/UI records receive only the compact projection; full model payloads remain in private audit storage. `adaptive_quorum` applies; any contributing FAIL fails, DEGRADED abstains (the reviewer verdict vocabulary `PASS|FAIL|DEGRADED` is NOT narrowable — `_contract_valid_actors`, the deliberate-DEGRADED capsule rail and the host's core-overflow DEGRADED all depend on it), and no quorum is a terminal HOST decision. The host acceptance decision itself is written ONLY by `acceptance_dialogue._set_acceptance_decision` (re-exported from `loop`) and has exactly three owner-facing states — `accepted | revision_requested | finalized_unaccepted` — each with a typed `reason` from an existing structured fact; an unknown status fails closed to `finalized_unaccepted` keeping its raw token as the reason. When you add a writer, add its reason to the closed set AND check every value-keyed reader: `outcomes.derive_loop_outcome` keys the eligible-but-skipped degradation on the status+reason PAIR (`review_skipped_deadline_reserve` plus the closed forced-rail `ACCEPTANCE_BYPASS_REASONS`), and the BLOCKED objective terminal on `_ACCEPTANCE_BLOCKED_TERMINAL_REASONS` (`review_cycles_exhausted`, `identical_acceptance_refused`); breaking either pairing is a silent false green. Forced exits stamp their typed bypass record in the common terminal recorder (`_record_forced_acceptance_bypass`) as a pure ledger write — never a fence, panel, extra round, or prompt text on a forced path, and never overwriting an existing host decision — with ONE exception (owner decision Q2A, 2026-08-10): the forced `children_unabsorbed` rail still runs the acceptance panel for an acceptance-eligible root when the subtree is quiescent, with the undispositioned-children debt included in the evidence packet; because that rail cannot take another round, a requested revision terminalizes as `finalized_unaccepted` with the typed `revision_unavailable_on_forced_rail` reason, while the process outcome stays best-effort `children_unabsorbed`. The agent may write only `agent_disposition`/`agent_rationale`, merged into the host decision, never replacing it. Clean requires PASS + solved + supported criterion evidence. Chat and Logs must use the same severity reducer, and degraded review or best-effort/degraded objective must never render as green solved. Do not add task scope review or reuse the commit gate.
- [ ] The acceptance improvement loop is a reviewer-authored DIALOGUE (v6.74.0): obligation identity comes from the reviewer's typed `disposition_kind`/`obligation_id` (an unknown re-raise id fails closed to `new`, disclosed — never a silent fresh hash id); a re-raise reopens the row WITHOUT wiping the agent's argument (`previous_disposition`/`previous_reason`/`reopened_count` survive into the evidence catalog and the obligations clause); termination beyond a clean PASS/accepted rebuttal happens ONLY via the reviewers' `dialogue_status` judgement (`aggregate_dialogue_status`) or a real rail — no host counters, no keyword gates (P5). That reducer is now read over `_contributing_actors` (owner ratification 2026-08-30, replacing the earlier widening to all contract-valid actors: a slot whose verdict did not reach the aggregate does not steer the loop either), and majority voting stays REJECTED — ONE contributing reviewer may hold the loop open, but only WITH MATERIAL: a `continue_actionable` vote counts only when the same response carries a concrete finding or a completion_coach, otherwise it is disclosed as `continue_without_findings` and abstains. Missing/invalid votes abstain too (`abstain_invalid`) and NEVER default to continue; a single well-formed terminal vote ends the dialogue; ZERO well-formed votes reduce to the typed `inconclusive`, which is not reviewer vocabulary (`DIALOGUE_STATUS_VALUES` is unchanged), grants the dialogue no authority, and falls through to the existing DEGRADED / no-capsule / exhaustion terminals — never a host-minted `stable_disagreement` and never another paid round. The reviewer verdict vocabulary `PASS|FAIL|DEGRADED` remains NOT narrowable. Changes here must cover: malformed reviewer output, unknown/stale `obligation_id` on a re_raise, partial panel failure, multi-slot dialogue-status disagreement (the reducer's precedence), replay/restart durability of obligation rows, false completion, and the backward-compatible default when the new fields are absent.
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
  marker retries rollback on boot, while the `marker_cleanup_retry` phase only
  retries the tx-marker unlink of an already-final repository — never a
  rollback. Delayed evolution cleanup also acquires the
  same update lock and honors this admission owner; it must not stash/reset
  behind the fence. A managed merge commits only with proof that the full
  suite ran green on the exact candidate tree: the resolver's single
  pre-commit hermetic run is recorded as `tests_evidence` and reused by the
  commit gate instead of a duplicate post-commit run; the ordinary
  self-modification commit/tag/test/push ordering remains unchanged. Any
  non-commit terminal of the assisted resolver rolls the live tree back and
  best-effort preserves the attempt — committed or as a synthetic commit of
  the uncommitted resolution — on the deterministic `failed-update-<target12>`
  branch (a retry of the same target through the ordinary apply supersedes
  it); the fresh rescue snapshot, not the branch, is the carrier rollback
  itself depends on.
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
- Resolve bare Node (`node`/`nodejs`; Windows launcher suffixes normalize for
  matching) for the same four surfaces, but once AFTER the dispatch gates: the
  node health check is an execution probe of an argv[0]-steered candidate, and
  pre-guard it would run a planted `PATH` shim before the fences refuse the
  call. The ladder is PATH-first with the probe — a healthy `PATH` node is a
  byte-identical no-op in argv and child env — and the bundled runtime
  substitutes only when the `PATH` candidate is missing or probe-dead: rewrite
  only `node`/`nodejs` argv[0]; npm/npx/pnpm/yarn/corepack and `sh`/`bash`
  bodies naming a family token get only the attested child-env `PATH` prepend
  (a formula with a rewritten absolute shebang is a disclosed residual). Guards
  inspect the original argv; the substitution stays in the same interpreter
  family and reaches the handler through the per-call attestation, which
  `verify_and_record` uses to execute the resolved argv while `check` keeps the
  original receipt-identity text. A non-local executor backend is never touched
  (no host path leaks into a container; a local executor continues the ladder),
  explicit paths and versioned names are never rewritten, and node never
  pre-blocks: with no usable runtime the argv runs as written and fails
  honestly with the probe facts disclosed.
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

Durable state files should use the SSOT helpers in `ouroboros/utils.py`:
`atomic_write_json(path, payload, trailing_newline=False, fsync=False)` for
write-then-rename persistence and `read_json_dict(path)` for dict-shaped JSON
reads. `write_text_atomic(path, content, fsync=False)` and
`write_bytes_atomic(path, content, fsync=False)` share one atomic FULL-OVERWRITE
seam (temp-sibling + `os.replace`, existing permission bits preserved, crash leaves
the old file intact); the text variant preserves normal platform newline handling
and the bytes variant uses binary mode for exact bytes. `atomic_write_json` and
`write_text` route through the text variant. Prefer these helpers over bare
`Path.write_text` / `Path.write_bytes` for full-file overwrites. Appends are
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

The variable contract is checked in **both** directions by the same file, and
unlike the typography guard it covers the whole stylesheet, migrated or not.
A `var(--x)` must resolve — declared in `web/style.css` / `web/settings.css`,
or written at runtime by a module — because an undeclared one is silent: the
rule renders its hardcoded fallback and the fallback becomes the real value
nobody can find. And a `:root` token must have a reader, with no allowlist for
reserves: a documented token that resolves nowhere is precisely what makes
surfaces reach for literals. Fixing a dangling name means pointing it at an
existing token, not declaring a new one.

### Layout and controls

- Top-level pages use a fixed `renderPageHeader` outside an independently
  scrolling body. Shared tab strips, buttons, scroll fades, page glass, icons,
  and masonry helpers replace page-local copies.
- Static visual properties belong in CSS classes/tokens. New inline
  `style=""` markup and `.style.<property>` assignments are review debt.
  Dynamic measured values may update a narrowly named CSS custom property when
  that is the actual runtime data flow.
- One semantic button variant expresses one action role. Neutral Settings and
  onboarding controls use the existing `.btn.btn-default` role; do not add a
  parallel `settings-ghost-btn` variant. A one-action result row uses the
  named `.settings-action-row` contract (status first, action docked right),
  while multi-action toolbars retain their own grouping. Notifications use the
  shared toast host unless status belongs to a permanently reserved control
  row. Working, warning, error, and destructive states keep consistent meaning
  across Chat, Logs, Settings, and Skills.
- A list editor reveals the entry it just added through
  `ui_helpers.revealNewRow(row, field)` — the one seam for "scrolled into
  view, caret in the first field" (`docs/DESIGN.md` "List editors"). A local
  `scrollIntoView`/`focus` pair in a list editor's add path is review debt, and a freshly
  added entry shows no error before the owner tries to save — an attempt
  judges the entries that existed then, never one added afterwards.
  `tests/test_available_subagents_ui_static.py` pins the seam; the
  `ui_browser` acceptance in `tests/test_ui_smoke_agents_panel.py` pins the
  behaviour.
- Task outcome truth stays in `log_events.js::taskOutcomeSeverity` and
  `taskTerminalPhase`; `taskPresentation` is the one compact factual projection
  consumed by task chips, live completion, history replay, and child terminal
  presentation. Keep raw reason codes in details/Logs. A non-terminal
  LLM/tool/checkpoint diagnostic may add a timeline fact, but must not promote
  the whole task, including when no earlier human headline exists. Unknown
  event names never acquire Chat severity from `error`/`crash`/`fail` keyword
  matching; Logs retains its independent diagnostic categorization.
- The Chat header reports connection and server-authoritative activity only.
  Failed task status does not synthesize header attention, a toast, unread
  state, or an owner action; those remain explicit domain/incident contracts.
- Floating chrome combines gradient and masked backdrop blur so the blur edge
  does not become a visible seam. The chat composer intentionally keeps blur on
  the input surface and reserves measured message padding around the dock.
- **Chat viewport invariant:** sample live-edge intent before an ordinary
  transcript mutation; follow only inside the 48 CSS-pixel zone, otherwise
  preserve the visible keyed message, nested-card, or Reviews anchor. Route late
  application-controlled DOM writes through the existing stable-viewport seam,
  but keep awaited Load-older, reconnect reconciliation, browser-visibility
  return, and cross-instance restoration as explicit lifecycle transactions. Native scroll
  anchoring is not proof of this contract: focused regressions disable it. Keep
  local viewport preservation separate from the remote activity marker, and do
  not add generic observers or competing `scrollTop` writers without a
  reproduced gap. Browser coverage is chosen by risk; this WebKit-sensitive
  contract requires the engines exercised by its marker-gated UI smoke.

### Responsive and accessible behavior

Navigation, headers, controls, and dialogs must stay operable by pointer and
keyboard, preserve focus order, and fit the relevant narrow viewport without
stealing usable text space. Use the shared responsive component before adding a
page-specific layout. A visible change is inspected with vision in at least one
relevant real consumer flow. A stored screenshot alone is not verification;
mobile or WebKit is not a universal requirement and is selected from risk.
Disclosed residual: a Widgets reorder changes the visible order without moving
DOM nodes (a moved frame would reload), so after a reorder the Tab/focus order
may differ from the visible order until a window reload rebuilds the cards;
keyboard reorder through the card handle follows the key order.

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
`allow-same-origin`; its parent bridge proxies only the owning extension route,
and its document policy admits scripts, images, media and fonts only from that
skill's own prefix (plus `data:`/`blob:`; `connect-src` closed). The route
iframe (`kind: iframe`) shares the sandbox and permissions set and has no bridge.
Both framed mounts (the extension-route iframe and the module `srcdoc` iframe
with its CSP/sandbox constants and parent bridge) live in
`web/modules/widget_module.js` (the child-side bootstrap that runs inside the
module frame — bridge grammar, `Response` rebuilt over a stream, resize reports,
dispose acknowledgement — is `web/modules/widget_frame.js`) and return their
disposer to the `mountTab` dispatcher in `widgets.js`, which keeps the card registry and the declarative
renderer. The framed card's chrome — the effective launch policy (owner override
> author `render.start` > kind default), whether it keeps the card running while
Widgets is hidden (`retain`, framed cards only), the one primary Start / Stop
control, the launch-policy menu and the stopped card's facade — lives in
`web/modules/widget_card.js`; the card reorder handles live in
`web/modules/widget_reorder.js` (a reorder is a pure move in the key order handed
back to the page; no node moves); the declarative `chart` helpers, the table
cell renderer and the shared dotted-path reader live in
`web/modules/widget_chart.js`; the masonry
(`web/modules/masonry.js`) packs the cards in that key order through
`--masonry-*` custom properties and returns a disposer. The pure list helpers —
per-card and order-independent list change signatures plus the keyed patch plan
— live in `web/modules/widget_list.js`; the page compares the signature after
every `GET /api/widgets`; an unchanged signature adds, removes, replaces or
moves no card node, and only card controls and the masonry properties are
reconciled.
Never load skill JavaScript into the SPA origin. Long-running actions use a
durable job id and resumable status polling rather than a foreground request
lost on remount.

Every timer, listener, observer, stream, abort controller, chart, and mounted
widget has a paired disposer. UI preferences such as widget order and the
per-card start-mode override belong in host state, never in extension manifests.

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

Reviewed skill callbacks use the separate loopback
`gateway/host_service.py` boundary. Its Presence routes accept authenticated
transport facts and return typed delivery receipts; they delegate binding,
admission, authority compilation, and execution to the Presence domain modules.
Do not copy that policy into an adapter or promote the callback surface into a
general owner/task API.

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
`portable_detail`, `skill_smoke`, and `size_ratchet`. CI opts into them
explicitly:

- `integration` runs real provider checks, including Cloud.ru when
  `CLOUDRU_FOUNDATION_MODELS_API_KEY` is configured and GigaChat when
  `GIGACHAT_CREDENTIALS` is configured. Its trusted target-push/manual/tag
  direct-OpenAI row derives unique shipped models from
  `OPENAI_DIRECT_DEFAULTS`, uses only public `LLMClient.chat`, and requires
  custom+`medium` plus a real registry-tool call. The shipped Main model also
  consumes a nonce-bearing tool result in a second turn. Missing credentials
  are red in the official repository job; explicit quota/429/5xx/timeout may
  be typed inconclusive, while contract/auth/model/reasoning/tool 4xx stay red.
  Secretless deterministic request-wire and Anthropic-custody contracts remain
  in ordinary pull-request tests; do not move provider secrets into PR jobs or
  edit the workflow merely to duplicate this existing trusted lane.
- `browser` launches real Playwright Chromium/WebKit for agent browser tools.
- `ui_browser` launches the host-side web UI under Playwright. The marker is
  the source of truth for what the lane collects; the Widgets lifecycle suites
  in it are `tests/test_widgets_ui_browser.py` (geometry, job retry),
  `tests/test_widgets_ui_browser_lifecycle.py` (launch policy, ordered stop,
  `retain`, the streaming bridge), `tests/test_widgets_ui_browser_patch.py`
  (keyed patch of a running card, reconnect reconcile, serialized policy
  writes) and `tests/test_widgets_ui_browser_capabilities.py` (the frame CSP,
  sandbox and permissions boundary: Wasm, blob workers, media and fonts,
  negative origins, on Chromium and WebKit) — run all of them before a release that touched Widgets, e.g.
  `OUROBOROS_RUN_UI_SMOKE=1 OUROBOROS_DATA_DIR=$(mktemp -d) python -m pytest -o addopts="" -m ui_browser tests/test_widgets_ui_browser*.py`.
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

- `size_ratchet` carries the live-repo size gates: exact-manifest smoke and
  the module/function hard gates (`tests/test_smoke.py`), the generator
  `--check` exactness half (`tests/test_repo_health_smoke.py`), and the
  pairwise base-vs-tip transition check. It is the ONLY blocking surface for
  repository size: official-repository CI runs `python -m pytest tests/ -m
  size_ratchet` as a dedicated third step in quick-test AND full-test, with
  `OURO_SIZE_RATCHET_BASE_REF` naming the event base (PR base SHA / push
  `event.before`; an all-zeros or unresolvable base degrades to the tip's
  parent manifest, verified against the parent's own tree — never a skip —
  while a resolvable base without the manifest fails closed).
  Default local runs exclude the marker and surface the same
  `validate_size_ratchet` findings as warnings through
  `check_worktree_readiness` and `codebase_health`; fixture-based ratchet
  unit tests stay in the default lanes — only checks against the live repo
  carry the marker. Its tests must not carry the `serial` marker (same
  single-lane rule as `skill_smoke`).

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
  path like `/tmp/foo.pid`); use `monkeypatch.setenv` / `monkeypatch.delenv` for environment
  changes — the autouse conftest snapshot (`_os_environ_isolation`) restores `os.environ` at every
  test boundary, so a bare `os.environ[...] = ...` no longer leaks, but monkeypatch stays the rule
  because it reverses exactly the named change inside the test (its undo runs last, after the
  snapshot); never assume execution order; and if you must mutate a module global, reset it around
  the test (pattern: `tests/conftest.py::_isolate_workspace_executor_globals`).

### The commit gate mirrors the CI split

`ouroboros/preflight_runner.py::run_hermetic_pytest` runs the node test lane
(`NODE_OPTIONS` scrubbed, as `PYTEST_*` is for the pytest passes)
plus the same two logical pytest passes as CI in one disposable checkout and
scrubbed temporary data root. The
candidate is captured universally — one hardened worktree-vs-`HEAD` binary diff
applied as raw bytes, assembled identically whether the source index is clean,
dirty, or mid-merge — and a capture or apply failure is the typed
`PREFLIGHT_CANDIDATE_ASSEMBLY` hard block with its own remediation, never a
test failure:

1. parallel `not serial` with xdist, loadscope distribution, no worker restart,
   and the configured per-test timeout;
2. flag-free `serial` for tests whose real process/port/global-state behavior
   cannot be parallel-safe.

Before those two pytest passes, one contained `node` subprocess runs the
browser-module suite (`ouroboros/preflight_node.py`): when the assembled
candidate carries `web/tests/*.test.js` files, the gate runs `node --test`
over them from the candidate's `web/` directory — the same suite
`web/package.json` scripts and both CI jobs run — resolving the bundled signed
node first and PATH second, with a 20.11 version floor. The lane is
content-keyed: a candidate without web tests never requires node, but while
the lane is active a missing or unusable runtime is the typed
`PREFLIGHT_NODE_MISSING`/`PREFLIGHT_NODE_TOO_OLD` hard block and a red suite
is `NODE_TESTS_FAILED` — never a silent skip.

All passes share one total timeout. `LANE_EXCLUSION_EXPR` and
`PARALLEL_PASS_FLAGS` are executable SSOTs pinned against both CI jobs, and so
is the node step (`cd web && node --test tests/*.test.js`, derived from the
lane's own glob constants). The selected interpreter must provide
`pytest-xdist` and `pytest-timeout`; plugin presence is probed outside
candidate control, forced on for the parallel pass, and proven by host-owned
worker markers. `OUROBOROS_PREFLIGHT_SERIAL=1` is the explicit temporary
rollback lever, never a silent fallback. Evidence runs set
`OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS=1` (and `OUROBOROS_PREFLIGHT_REQUIRE_NODE=1`
for the node lane's real-spawn tests).

The candidate environment cannot weaken the pass with inherited `PYTEST_*`
values, delete an inherited suite and earn green, or replace required plugins
with fake command-line options. Pre-commit checks bind to `HEAD`; post-commit
checks also inspect `HEAD~1` so deletion of the suite cannot hide after the
commit exists. Exit status owns the verdict; rendered/truncated diagnostics do
not.

A red post-commit gate preserves the local commit for forensics but blocks push.
Inside a managed update it also blocks boot promotion and routes through
rollback; an incomplete rollback leaves `gate_blocked` so boot retries recovery
instead of promoting the rejected merge. The managed gate's mandate is "the
full suite provably ran green on the exact committed tree", not "run it
twice": recorded `tests_evidence` from the resolver's green pre-commit
hermetic run is reused when it covers that exact tree, and the post-commit run
happens only when no such proof exists. Review-binding and tag-binding
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
