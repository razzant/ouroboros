# Naming and boundaries

This chapter owns the rules that keep the body legible from outside: naming and entity-type conventions, dependency direction, the CLI and headless contract, cognitive quality, LLM-first affordances and the prompt-edit discipline, the documentation contract, the generality question, pricing and admission, and the anti-patterns that have actually cost this system incidents. It exists because each rule here names the test, gate or CI lane that enforces it, or states honestly that only review does.

- Code identifiers, comments, docstrings, commit messages, and user-facing
  product UI strings are English.
- Follow PEP 8: modules and variables use `snake_case`, classes use
  `PascalCase`, constants use `UPPER_SNAKE_CASE`. Name the observable
  responsibility and authority, not the implementation fashion; prefer a clear
  function module over a class with no lifecycle.
- Contracts are typed shapes, not service objects. A manager is justified when
  it owns lifecycle or mutable state. LLM-callable Tools remain thin
  `{verb}_{noun}` functions that validate their public input, call the owning
  subsystem, and format a result. No universal `{Domain}Service`,
  `{Platform}Gateway`, or class layer is required.
- Dependency direction is the test: UI/CLI → inbound gateway → domain owner;
  runtime policy → small host-owned contract → outbound adapter (the
  `ouroboros/gateway/` inbound vs `ouroboros/gateways/` outbound map is
  ARCHITECTURE "Gateway Boundary v1"). Provider- or transport-specific
  decisions do not flow back into core policy.
- Chat authorship is stamped by the producer and preserved through persistence
  and replay; never infer it from text or promote host-selected intermediate
  output to a model final (`tests/test_terminal_provenance.py`).

Enforcement: naming and entity-type rules are scored in commit review by
CHECKLISTS item 2 `development_compliance` (a)/(b); the transport/core
dependency direction has one CI guard (the "Guard extracted transport imports
stay out of core" step in `.github/workflows/ci.yml`); the remaining boundary
rules have no automated surface — review-only.

### CLI and headless work

- CLI commands parse, call the existing gateway/scheduler, and render text or
  typed JSON/JSONL/SSE. They do not create a second task state machine.
- External workspace tasks keep governance bound to the system repository while
  contextual tools default through `ToolContext.active_repo_dir()`. Admission
  rejects overlap with the system repo/data and records a read-only preflight.
- Physical file resolution must preserve the requested address. Normalize
  in-root absolute paths for every resource; refuse an outside address before
  relative-path sanitization. Keep repo-prefix and canonical artifact redirects
  on the same resolver used by guards and handlers.
- Project focus changes the default target, not the top-level tool surface.
  Generic VCS selects active/system explicitly; advisory, reviewed commit,
  rollback, promotion, restart, and runtime control keep their intrinsic
  system-repository contracts. `executor_ref` selects a process backend, not an
  implicit sandbox.
- Project-local installs may run within the workspace policy. Global/system
  installs remain safety-reviewed, and `sudo` is non-interactive (`sudo -n`).
- GitHub issue/PR tools resolve the same process binding as shell commands and
  carry an explicit `repo` into every dependent call. Project focus overrides
  ambient `GH_REPO`; broken room bindings refuse and file-less Projects need an
  explicit repository. Presence may override its default repository only through
  a host-selected argument binding. Native CLI configuration proves configuration,
  not active authentication; discovery never logs in or probes the network.
- Do not add a second scheduler for operator tooling or a generic CLI file
  manager. Use the task queue, attachments, logs, and artifact endpoints.
- A process cwd determines relative paths, not the root task's entire write
  authority. Reuse the resource binding for other authorized destinations;
  preserve child write confinement and actual runtime/credential boundaries.
  Process admission preserves original argv and the prepared physical target.
  Do not reconstruct semantic permission from command words or guessed effects;
  use the Supervisor source and mode contract in ARCHITECTURE "Safety and runtime
  mode". Explicit task resources and actual Git targets remain distinct from
  semantic judgment; this does not promise an SSH or OS sandbox.
- Do not infer credential authority from ordinary source/config directory
  names, and never refuse owner input or owner output on a SUFFIX or a WORD
  inside a file name — on attachment ingest, on export or Deliverables, on a
  `user_files` mutation, or in the git lanes — with one surviving tail rule:
  dotenv spellings (`.env`, `.env.local`, `prod.env`) still refuse on ingest,
  on export, in the git lanes and in the skill/review packs. Which locations
  the owner credential fence covers, how the one byte masker and the PEM
  content evidence work, and why unlisted stores such as `.cargo`,
  `.terraform.d` and `.kaggle` get no dotted-directory default-deny:
  `docs/architecture/06-agent-core.md` § "Tool capability and execution".
  Prepare invariant credential/root locations once per list/search/query call
  through `make_subagent_secret_target_check`; never retain that predicate
  across calls — target resolution and owner-state/file-identity checks remain
  per target. The capture, snapshot and cooperative checkpoint consumers use
  `pem_capture_refusal`: effective Cyber preserves the original finding as
  advisory and includes the requested bytes; ordinary modes retain the existing
  exclusion. The exact SSH config exception does not permit key writes under
  `.ssh`.
- An unlaunchable sole cmd element gets an actionable argv/shell hint, never
  automatic splitting or an implicit shell. Repo-only edit tools reject their
  unsupported roots through their existing argument categories.

Enforcement: `tests/test_headless_cli.py` (task-API admission, typed refusal
terminality, attachment admission), `tests/test_cli_entrypoint.py` (the CLI
surface), and `tests/test_external_workspace_access.py` plus
`tests/test_workspace_authority_binding.py` (workspace policy and system-repo
collision blocking). The no-second-scheduler and no-generic-file-manager
rules have no automated surface — review-only.

### Cognitive quality

Do not lower model quality, reasoning effort, output budget, or context breadth
as an incidental latency/cost optimization (BIBLE P1 owns the principle). An
intentional narrowing is an explicit recorded decision reflected in the plan,
docs, tests, and evidence; outside Cyber Pro it belongs to the owner. Cyber
configuration authority follows BIBLE P0/P3 without rewriting earlier call facts.
No automated surface — review-only: commit review carries
it through CHECKLISTS items 1 (`bible_compliance`) and 21
(`capability_regression`: an accidental narrowing is the named failure class).

### LLM-first affordances

Do not repair a semantic tool-choice failure by adding one more keyword hint to
`prompts/SYSTEM.md`. A model mistake alone does not justify a new host contract:
first establish a real missing capability or information, or an independently
valid requirement. When discoverability is genuinely missing, repair the existing
tool schema or affordance at the point of need. Do not freeze the model's reasoning,
dialogue representation or collaboration strategy to make one incident testable.
SYSTEM accretion trains around one incident, bloats the resident prefix, and forks
the authority. These choices are reviewed through CHECKLISTS' development and
capability-regression items, not a new semantic gate.

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
credential and contract filters narrow it), so the schema is the SSOT
of the per-tool contract and a prompt sentence about it is a second copy that
drifts, while SYSTEM.md stays the cross-tool selection policy; mechanism
documentation lives in ARCHITECTURE or here;
runtime facts (capabilities, queue, catalog, receipts, health invariants,
memory sections, registry digest, installed skills, review section) are
assembled ONCE per task attempt in `build_llm_messages`, so the Health
Invariants block lists custody obligations as of task start and does not
refresh mid-task (a deliberate frozen-ContextCore / prompt-cache choice; the
integrate schema, the apply receipts and the absorption digest carry the
mid-task fact instead). Tool schemas are re-sent every round.
A new tool therefore requires NO SYSTEM.md mention. Before adding a
sentence to a prompt, check that the schema or runtime block does not already
carry it; before removing one, check that they do (or add the missing fact to
the schema without growing it into a paragraph). Local-model compaction keeps
only the text before the first `## ` heading (plus the BIBLE section), so the
load-bearing floor rules stay in that preamble. Every prompt change reports the before/after byte size
in the commit or PR. The memory contract's resident rule: a note's authored
summary is its resident face in the knowledge index, and a resident memory
carrier that is absent renders as a visible gap, never as silence.

Recoverable tool failures are evidence for the next LLM turn, not triggers for
a host-authored recovery workflow. Return a typed, redacted result naming the
failed stage, already-completed external effects, and an actionable repair
hint; the LLM decides whether to inspect, repair, retry, use another capability,
clean up, or stop. Host code remains responsible only for deterministic
integrity and authority boundaries plus truthful receipts; do not add
task-specific auto-retry, fallback, cleanup, resume, or terminal-flow state
machines.

Explicitly naming a documented default is never a different request. An argument
whose value is what omitting it already means (`directory_strategy="direct"` with
no `scope_paths`) takes the omitted path on a shape that cannot serve the argument
at all; only values that genuinely ask for something are refused there, typed, at
the earliest layer holding the authority to judge them, with the repair named.

A producer that already knows its call failed publishes that fact typed: a
`ToolResult` through `tool_result._publish_tool_result`, or a first-line
`⚠️ IDENTIFIER` marker the legacy adapter maps to a status. Identifier-less
`⚠️ prose` and a bare `ERROR: ...` string are recorded by the registry — in
`tools.jsonl`, the outcome classifier and the acceptance packet — as a successful
call, so the failure the producer saw is lost exactly where the next decision
reads it. A wrapper over an inner producer (`view_image` over the local image
loader, the publication transaction over the GitHub transport) carries the inner
failure and its safe cause forward instead of a stage-only word.

Inside the external-executor family (`delegate_start`, `delegate_wait`,
`delegate_cancel`, `delegate_answer`) the result IS the `ToolResult`: one refusal
author (`delegate_shared._fail`) writes `ok: false` plus `host_code` beside the
domain payload, the producers, decorators and host consumers (bootstrap, recovery,
the unknown-provider hold, the pending-wake replay) pass that value around, and the
four registered entries publish it once, after every decoration, immediately before
returning its text. Publishing earlier is silently discarded: the registry accepts a
published result only when its text IS the string the handler returned. The
acknowledgement of a supervising wake is keyed on the `supervision_wake_id` that
result publishes, not on the tool's name.

Enforcement: the prompt-edit discipline is scored by CHECKLISTS item 13(b)
(a prompt edit never restates a tool schema and is never an incident patch);
the recoverable-failure boundary has no automated surface — review-only; the
typed-refusal rule is ratcheted by `tests/test_typed_tool_refusals.py`, a
source lint over returned literals in `ouroboros/tools/` that flags only
identifier-less heads (`⚠️ prose`, bare `ERROR:`); its per-file allowlist
discloses that identifier-less residual — a file whose count grows fails, a
shrink must be recorded, and a same-file swap of one old site for a new one is
invisible to the count. A marker-shaped refusal the adapter still buckets as a
warning (`⚠️ SOME_IDENTIFIER: …` recorded `ok`) is a separate, larger residual
owned by the adapter vocabulary, not by this lint; a failure text that travels
through a variable, a tuple or a helper is outside its reach and is pinned by
the producer's own tests.

### Documentation contract

`docs/ARCHITECTURE.md` is the map of the body (BIBLE P6), written in the present
tense: what exists, where it lives and how it flows (structure and operation),
and WHY it is so. Every important WHY — cross-module, gate-level or module-local
— stays in the map, at least briefly (a second line under a module row or one
sentence in the owning section); mechanism detail beyond that lives in the
module's docstring, which the map points to by name and never copies.
`docs/DEVELOPMENT.md` is how the body is changed (its authority and shape:
"Role and authority"); mechanisms, reviewer criteria, constitutional text and
defaults belong to their owners (ARCHITECTURE, CHECKLISTS, BIBLE,
`config.py`) and are pointed to, not restated — the ARCHITECTURE settings and
endpoint tables are test-checked registries of those owners, not second
authorities. A change REPLACES the description of the node it touched; release
history lives in git and the README history table, leftovers go to issues.
Residue — parenthesized version stamps, decision codenames, "used to /
previously" narrative — is caught by the shrink-only residue check in
`tests/test_docs_sync.py`, which enforces only the explicit, case-sensitive
matches in `DOC_RESIDUE_PATTERNS`, outside its declared skipped subsections and
language-tagged fences (the untagged module-tree fence in ARCHITECTURE §1 IS
scanned — an owner decision); semantically equivalent historical prose stays
review-only under CHECKLISTS item 7.

Both of those documents are reference BOOKS: an entrypoint carrying its H1, one
authored introductory paragraph and an ordered `## Chapters` membership list,
plus one chapter file per subject under `docs/architecture/` or
`docs/development/`. Four rules keep that shape honest, and
`tests/test_reference_book_validation.py` runs the validator over the tracked
tree so a breach is red rather than discovered by the next review:

- **One membership list.** The entrypoint's `## Chapters` list IS the book.
  Adding a chapter file without listing it, or listing one that is not tracked,
  fails the validator; there is no second manifest, registry or JSON index to
  keep in step.
- **Every source carries one authored introduction.** The first paragraph under
  a chapter's H1 says what that chapter owns and why it exists. It is the
  compact view — there is no second editable summary corpus — so a chapter whose
  H1 is followed straight by a subsection is refused rather than having body
  prose quoted as if someone had written it for that purpose.
- **WHY stays where the reader is.** A chapter keeps the rationale for what it
  owns; moving prose between chapters is a documentation change like any other
  and REPLACES the description at its destination.
- **One line ending.** `.gitattributes` pins `docs/**/*.md` to LF, so the
  physical line ranges and SHA-256 digests the generated inventories and the
  transfer table carry mean the same bytes on every platform; a CRLF checkout
  would move every cited line.
- **Readers ask for a view, not for a file.** `load_governance_doc` composes a
  book for a surface that owes it in full, `context_layout.book_navigation`
  renders the compact chapter-addressed view, and
  `reference_books.book_path_role` answers whether a path is part of a book.
  Never read an entrypoint with `read_text()` and treat the result as the book
  — that is a membership list, and a substring pin over it passes while testing
  nothing (`tests/_governance_docs_shared.py` is the one reader tests use).
  `docs/reference-books-migration.md` is the operator record of the original
  split and is deliberately not a member of either book.

### Generality and emergence (P13)

Every non-trivial change picks a level: patch the case in front of you, solve
the class it belongs to, or build a framework for cases that do not exist yet.
The first fossilizes, the third speculates; aim for the second — BIBLE P13's
invariant question and stronger-mind question find it. The proof burden is
symmetric: promoting a case detail into shared structure requires showing it is
an invariant (several real variants, or one already-stable boundary); adding an
abstraction requires a demonstrated class — an imagined consumer is not one. In
doubt, generalize the meaning and the authority, keep the mechanism minimal and
local, and let the next real case pay for the next step. Reviewer findings are
evidence for this judgment, never policy that overrides it. No automated
surface — review-only.

### Pricing and admission

Never add hand-maintained model-price tables, inherited prefix tariffs, or
numeric fallback prices; preserve `cost=None` and `cost_final=false` when no
live source answers the exact route. Unknown price is neither free nor a
model-admission veto; known exhausted budget remains enforceable. Enforced by
`tests/test_pricing.py` (exact-route lookup, no prefix inheritance, unknown
cost stays `None`) and `tests/test_budget_limits.py` (a known exhausted budget
still fences); mechanism: ARCHITECTURE "Budget tracking".

### Anti-pattern: content-derived identity for host-minted records

If the host itself created a record — a chat message, a task, a binding — its
identity is captured at ingress and passed downstream BY VALUE as a typed
reference (e.g. `origin_message_ref` built where `log_chat("in", …)` writes
the canonical row). Never re-derive it later by searching logs/state for a
row whose text hash/equality/prefix matches: in an LLM-first system the text
is routinely rewritten between ingress and use, so content-derived lookup
fails exactly on the normal path. Content hashes are legitimate only as (a)
an INTEGRITY CHECK on an already-known identity, and (b) content-ADDRESSING
where the content IS the identity (artifact stores, observability blobs,
staged-diff review bindings). The enforcement shape is a REQUIRED typed
argument at the consuming seam (`bind_task_to_project(..., *, origin)`: a
valid ref or a closed-enum absence reason; omission raises), so a future call
site cannot silently skip the invariant — `tests/test_projects_v6640.py`
exercises that seam. For fuzzy entities use the LLM-first pattern
(`semantic_dedup`), never string equality.

The same captured reference is also the IDENTITY OF THE WORK, not just its
provenance: a new task id minted from the same owner message (a promoted root,
a mid-run scope call, the timeout retry that replaces a dead attempt) must
INHERIT that origin's project binding (`projects_registry.project_id_for_origin`,
keyed by value on chat id + client message id), never re-derive project
membership from its own id — one convertible unit per message, not one per task
id. A timeout retry is bound at RETRY ADMISSION, inside the transaction that admits
it, and only once cancellation can no longer win the boundary (the mechanism
and its WHY: `docs/architecture/06-agent-core.md` § "Durable memory and project
focus"). Enforced by `tests/test_retry_project_binding.py`.

One named exception inside role (b): a verification RECEIPT with no earlier
ingress point is reconciled by ONE TYPED IDENTITY KEY, matching on the key's
kind AND value, never across kinds — a per-component fallback chain is not an
equivalence relation (the outstanding set came out order-dependent), while
keying makes sameness the kernel of a function and fails safe: a false red
costs a human a second look, a false green costs the thing this surface
exists for. The full mechanism (masked-path rules, `IDENTITY_KINDS`,
projections, bounds, rendering stamps) lives in
`ouroboros/_outcome_receipts.py`, enforced by
`tests/test_v678_receipt_reconciliation.py`. The process-tool lane discloses a
masked exit code in its result envelope only and writes no receipt, so nothing
there participates in masked-pass reconciliation or the masked-verification
nudge. Four rules generalize:

- **Whatever decides must be what is reported**: the reporting path reads the
  deciding path through one shared projection, never a re-derivation beside
  it — a host-attested artifact that misstates its own basis is worse than
  one that says nothing, because a reviewer cannot discount evidence whose
  provenance it was told wrongly.
- **A property of a closed set of kinds lives IN the set** (a table row per
  kind plus a total lookup that raises on a kind that skipped the table), so
  a new kind cannot be added without answering.
- **One canonical identity derivation** — comparison, hashing, counting, and
  projection read the same derived object, in the order canonicalize the RAW
  values → render → bound; a normalization that discards information the
  identity depends on is not a normalization.
- **Changing a stored rendering means versioning it**; reason about a format
  migration in BOTH directions — the false-green direction is the one that
  gets missed, and unknown must not clear a red.

Disclosed deferred limit: `tools/verify.py` bounds the DURABLE
`artifact_observation` path set at twenty with no omission count — advisory
only (a nudge and a disclosed reviewer flag, never a gate); fixing it means
changing the durable store and deserves its own scope. Beyond the typed seams
and tests above, this section is review-only.

### Anti-pattern: an open default behind a closed exception list

A behaviour that is "on by default, except for these names" keeps its real
rule in a list that can only go stale. The retired `ADDRESSING_ONLY_TOOLS`
(`web/modules/chat_activity.js`) decided whether a chat turn had a card by
subtracting three tool names from the tool count: every new addressing tool
would have minted an empty card, every renamed one would have silently left
the list, and the list restated a fact the host already carried as the typed
routing annotation. Derive presence from the facts the record already holds
(`web/modules/chat.js::blockVisible`) and let the host name the special case
from the one table it owns (`ouroboros/tool_capabilities.py::ROUTING_VERBS`:
`routing_action` on the live tool-call frames, `routing_tool_calls` in the task
metrics, `typed_routing_action` on the terminal event), never a client-side
exception list. The same shape hides in "hide unless kind ∈ {…}" and "count
unless name ∈ {…}": when the list is the rule, the rule is missing.

### Task-authored messages are never owner text

Who is speaking through a routing act is ONE fact the host mints by value
(`control_routing._routing_issuer`; the owner-turn/task-turn distinction and
what each may carry: `docs/architecture/06-agent-core.md` § "Durable memory and
project focus"). Never derive it again from a proxy — a routing contract only
chat turns carry, an empty client id, the chat id of the event — and never give
the model an argument for it. A task's own words are delivered as a task
message (`KIND_TASK_MESSAGE`, provenance `independent_task`), never as
`KIND_OWNER_TEXT`, and the provenance value lands at three seams in one change:
the writer's closed set (`owner_mailbox.TASK_MESSAGE_PROVENANCES`), the render
ladder (`deliver_task_message`: `[Message from independent task <id>]`, never
the ancestor fallback) and the drain mapping (`loop_round_limits`): an
independent task's words are context the receiving model judges, so they enter
no owner corpus — `owner_source_sha256`, the post-drain growth check that
supersedes a paid acceptance panel and the acceptance premises stay the
owner's (owner 4=A). Enforcement: `tests/test_task_authored_messages.py`.

### Anti-pattern: a chat id tested for truth

A chat id is a VALUE, not a boolean. `HIDDEN_CHAT_ID` (0) is a REAL
destination, the hidden partition
(`docs/architecture/12-host-service-companions-and-chat-ids.md`); absence is
`None`, and a negative id is synthetic A2A traffic. `if chat_id:` therefore
does two wrong things at once: it drops a partition-bound notice AND re-routes
hidden work to the owner's main chat, which is how a whole `ouroboros run`
went invisible while its children surfaced in Main as a nameless card. Use the
two normalizers instead of a third: `message_bus.notification_chat_route`
answers "where does this notice go" (first DELIVERABLE candidate, `None` when
none is) and `message_bus.coerce_chat_identity` answers "what is this row's
address" (explicit value kept, absence defaulted). Address a task once at admission (`log_addressing.ingress_chat_id`) and pass
the value downstream; a producer that sends to the owner DIRECTLY (nothing
re-addresses it later) resolves the task's durable project binding AT EMISSION
through `log_addressing.resolve_project_chat` and puts it ahead of the row's
chat, because a task bound to a project after admission still carries the chat
it was born in. The admission contract — one destination per registered project, the hidden
default for ordinary API tasks, browser-declared Main — is
`docs/architecture/05-supervisor-loop.md`; task type is never source
provenance. Enforcement: `tests/test_chat_id_truthiness_guard.py` is the
source lint that keeps the class closed; it also sees the id read straight off a
mapping inside a condition (`if row.get("chat_id") and ...`), the form where no
local exists for the other alternatives to match; its allowlist is where a
deliberate exception states its reason.

### Mutable external-fact inventory

This table is a maintenance inventory, not a second runtime authority. External
facts change independently of Ouroboros releases; prefer live metadata or a
bounded probe where that can answer the exact question, and otherwise keep the
current conservative behavior visible. This inventory documents these facts but does not migrate their runtime representations. No automated surface checks these
rows — review-only maintenance.

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
scope policy, safety, and context/memory flows. Core capability must not
acquire a hidden OpenRouter or second-provider dependency. (CHECKLISTS item
2(h) and ARCHITECTURE both point here; this is the SSOT sentence.)

Tool-schema changes are provider-contract changes. Every shipped built-in
schema must pass general JSON Schema and the known cross-provider subset over
the complete registry; trusted integration CI sends that same registry in one
bounded tool canary per supported provider family/API surface, in the transport
Main uses (streaming for compatible routes), while pull-request CI remains
secretless. Malformed native arguments and invalid schemas stay red, with
diagnostics limited to structural facts, hashes, and parse position. Do not add
a prose parser, provider hop, or unbounded retry to make that contract green
(canary anatomy: ARCHITECTURE "CI topology").

When adding or changing a provider, update one coherent route contract:

1. credential/readiness detection and exact model-id migration;
2. Main/Light/Fallback and reviewer-slot defaults without overwriting explicit
   owner choices;
3. canonical tool/reasoning/image/cache intent at `llm.py`, with provider wire
   projection and exact-route recovery delegated to the small transport leaves;
4. nullable pricing/settlement and truthful capability omissions;
5. review and scope routing, including sourced context-window evidence;
6. direct-provider and single-provider regression tests;
7. record the route's real streamed wire into `tests/fixtures/llm_wire/`
   (redacted: opaque reasoning payloads and signatures truncated) and replay it;
   a hand-written stream fixture is not evidence of a provider dialect. Record
   with `curl -N` against the route, or export the private `physical_stream`
   blob the runtime already retains for every stream.

Local-only installs keep their local route. Unreachable shipped remote defaults
may be cleared, but explicit owner values are not. Scope authority follows
BIBLE P3: owner-selected Max requires the applicable sourced window evidence;
owner-selected Low records the declared skip rather than pretending a partial
review occurred. Current model ids and defaults belong in code/config, not in
this handbook.

Use `provider_models.ACTIVE_MODEL_SETTING_KEYS` for any new active consumer
(provider detection, model catalog/provenance, credential planning, Provider
Test); `LEGACY_MODEL_SETTING_KEYS` exists only for migration/history. In
particular, `OUROBOROS_MODEL_HEAVY` and the paired `USE_LOCAL_HEAVY` may seed
an explicit configured API actor while the canonical list is absent, but must
never become an active slot, startup-readiness signal, test probe, or fallback.
Do not patch each consumer with its own Heavy exclusion; preserve the shared
split.

The `-pro` suffix is an OpenRouter routing slug, not an official OpenAI model
id; a direct OpenAI Chat slot uses the plain Sol id, because projecting the
slug into Chat Completions would turn an owner route choice into a guaranteed
404. This is a compatibility constraint, not a mutable capability table.

Provider-specific optional features may be unavailable on another single
provider, but the core loop must degrade explicitly rather than crash or
silently reroute.

Canonical assistant history and tool schemas are function-shaped across
providers; do not add a second stored transcript for a provider dialect. Direct OpenAI tool conversations stay on Chat Completions (the
custom/function/`none` ladder and the provider-wide
`reasoning_effort`/`max_completion_tokens` carriage:
`docs/architecture/06-agent-core.md` § "Context fitting, retry, and compaction"
and `docs/architecture/07-configuration.md` § "LLM output token budgets");
model-name prefixes are not admission authority. DeepSeek is the second effort-carrying route (`reasoning_effort` beside the
compatible-lane `max_tokens` carrier; the tier projection, the
forced-tool-choice thinking rule and the `reasoning_effort_clamped` disclosure:
`docs/architecture/07-configuration.md`, DeepSeek provider specifics). The
carriage is keyed on the provider id, never a model-name prefix or a target capability field, so a
hand-built target cannot silently drop it.

Apply static, semantics-preserving wire normalization before request-wire
binding; keep learned recovery and its evidence store separate.

All learned request-shape adaptation goes through the one provider-neutral
request-wire driver (`ouroboros/request_wire_contract.py`: exact-route
identity, closed action vocabulary, shared TTL, never executes provider prose
or switches route); do not add a second driver, and explicit `none` is never
durable. Direct Anthropic is the deliberate exception to a purely reconstructed provider
transcript (its route-bound replay receipt:
`docs/architecture/06-agent-core.md` § "Context fitting, retry, and
compaction"); do not synthesize an effort-to-`budget_tokens` policy.

