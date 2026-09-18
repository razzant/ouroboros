# Naming and boundaries

This chapter owns the rules that keep the body legible from outside: naming and entity types, dependency direction, the CLI and headless contract, cognitive quality, LLM-first affordances, the documentation contract, generality, pricing, and the anti-patterns that have cost this system real incidents. Each rule names the test, gate or CI lane that enforces it, or says that only review does.

- Code identifiers, comments, docstrings, commit messages and user-facing
  product UI strings are English.
- Fork divergence (`fork/custom`), recorded so the rule above is not read as
  broken: product source strings stay English, and a localization never edits a
  call site. A language ships as `web/i18n/<lang>.js` — a data file keyed by the
  English source string, applied to the rendered DOM at runtime — so the Russian
  text in `web/i18n/ru.js` is translation data, not a product UI string.
  Separately, commit messages on fork branches are Russian by the repository
  owner's standing policy; a change proposed upstream must have its commit
  messages re-worded to English before the pull request is opened.
- Follow PEP 8 (`snake_case`, `PascalCase`, `UPPER_SNAKE_CASE`); name the
  observable responsibility and authority, not the implementation fashion, and
  prefer a function module over a class with no lifecycle.
- Contracts are typed shapes, not service objects; a manager is justified by
  lifecycle or mutable state. Tools stay thin `{verb}_{noun}` functions that
  validate public input, call the owning subsystem and format a result — no
  universal `{Domain}Service`, `{Platform}Gateway` or class layer.
- Dependency direction is the test: UI/CLI → inbound gateway → domain owner;
  runtime policy → host-owned contract → outbound adapter (`ouroboros/gateway/`
  in, `ouroboros/gateways/` out: ARCHITECTURE §1 "Gateway Boundary v1"). Provider- and
  transport-specific decisions never flow back into core policy.
- Chat authorship is stamped by the producer and survives persistence and
  replay; never infer it from text, never promote host-selected intermediate
  output to a model final (`tests/test_terminal_provenance.py`).

Enforcement: CHECKLISTS item 2 `development_compliance` (a)/(b); one CI guard
for the transport/core direction (the "Guard extracted transport imports stay out
of core" step in `.github/workflows/ci.yml`); the rest is review-only.

### CLI and headless work

- CLI commands parse and call the existing gateway/scheduler, rendering text or
  typed JSON/JSONL/SSE. Never add a second task state machine, a second
  scheduler for operator tooling or a generic CLI file manager: the task queue,
  attachments, logs and artifact endpoints are it (ARCHITECTURE §1 "CLI / Headless Boundary").
- External workspace tasks keep governance bound to the system repository while
  contextual tools default through `ToolContext.active_repo_dir()`; admission
  refuses overlap with the system repo/data and records a read-only preflight.
- Resolve every resource address through the one resolver guards and handlers
  use, and never sanitize a relative path before refusing an outside address.
- Project focus changes the default target, not the top-level tool surface:
  generic VCS selects active/system explicitly, advisory, reviewed commit,
  rollback, promotion, restart and runtime control keep their system-repository
  contracts, and `executor_ref` selects a process backend, not a sandbox.
  Project-local installs may run within the workspace policy; global/system
  installs stay safety-reviewed and `sudo` is non-interactive (`sudo -n`).
- GitHub issue/PR tools take the same process binding as shell commands and an
  explicit `repo` in every dependent call; project focus overrides ambient
  `GH_REPO`; broken room bindings refuse, and file-less Projects need an
  explicit repository. Native CLI configuration proves configuration rather than
  authentication, and discovery never logs in or probes the network. Presence may
  override its default repository only through a host-selected argument binding.
- A process cwd decides relative paths, not the root task's write authority:
  preserve child write confinement and the real runtime/credential boundaries,
  and never reconstruct semantic permission from command words or guessed
  effects (ARCHITECTURE §6 "Safety and runtime mode"). This promises no SSH or OS sandbox.
- Never infer credential authority from an ordinary directory name, and never
  refuse owner input or output on a SUFFIX or WORD inside a file name; dotenv
  spellings are the one surviving tail rule. Owner locations are a physical list
  (`credential_shapes.owner_credential_locations`), so an unlisted store keeps
  ordinary access; `make_subagent_secret_target_check` is prepared once per call
  and never retained; `pem_capture_refusal` keeps effective Cyber's finding
  advisory with the requested bytes while ordinary modes keep the exclusion; the
  SSH config exception permits no key writes under `.ssh`.
- An unlaunchable sole cmd element gets an actionable argv/shell hint, never
  automatic splitting or an implicit shell; repo-only edit tools reject
  unsupported roots through their existing argument categories.

Resolver, fence and admission mechanics: ARCHITECTURE §6 "Tool capability and execution".
Enforcement: `tests/test_headless_cli.py`, `tests/test_cli_entrypoint.py`,
`tests/test_external_workspace_access.py`,
`tests/test_workspace_authority_binding.py`; the no-second-scheduler and
no-generic-file-manager rules are review-only.

### Cognitive quality

Never lower model quality, reasoning effort, output budget or context breadth as
an incidental latency/cost optimization (BIBLE P1). An intentional narrowing is a
recorded decision carried in plan, docs, tests and evidence; outside Cyber Pro it
belongs to the owner, and Cyber's own configuration authority follows BIBLE P0/P3
without rewriting earlier call facts. Review-only: CHECKLISTS items 1 and 21,
whose named failure class is an accidental narrowing.

### LLM-first affordances

Never repair a semantic tool-choice failure with one more keyword hint in
`prompts/SYSTEM.md`. A model mistake alone does not justify a new host contract:
establish a real missing capability or information, or an independently valid
requirement, then repair the tool schema or affordance at the point of need.
Never freeze the model's reasoning, dialogue representation or collaboration
strategy to make one incident testable — SYSTEM accretion trains around that
incident, bloats the resident prefix and forks the authority.

`prompts/SYSTEM.md` is tier-0 for every Main/task profile in both context modes
and competes with the task for context; the safety supervisor is the one caller
with a prompt of its own, and Background Consciousness uses SYSTEM.md like any
other turn, with `prompts/CONSCIOUSNESS.md` as the USER message a wake-up
receives, never a second system prompt. It carries identity and tone, the
decision loop, cross-tool policy, prohibitions and safety invariants stated once,
and the memory contract — whose resident rule is that a note's authored summary
is its resident face in the knowledge index and an absent carrier renders as a
visible gap, never as silence. It never carries how a tool works: parameters,
recipes, typed outcomes and "when to choose it" belong to the `get_tools()`
schema every profile receives in full each round (delegated, repair, credential
and contract filters narrow it), so a prompt sentence about a schema is a
drifting second copy and a new tool needs NO SYSTEM.md mention. Runtime facts are
assembled ONCE per task attempt, so the Health Invariants block states custody
obligations as of task start and never refreshes mid-task — a deliberate
frozen-ContextCore / prompt-cache choice (ARCHITECTURE §6 "Context fitting, retry, and
compaction"). Check the schema and runtime block before adding a prompt sentence,
and before removing one. Keep SYSTEM's load-bearing floor rules in its preamble,
because local-model overflow compaction keeps only the text before a block's
first `## ` heading (`ouroboros/llm_local.py`); that splitter also leaves BIBLE's
principle bodies as compaction markers, a disclosed defect (issue #1018), never
an intended reduction of the constitution (BIBLE P1). Every prompt change reports
its before/after byte size in the commit or PR.

Recoverable tool failures are evidence for the next LLM turn, not triggers for a
host-authored recovery workflow: return a typed, redacted result naming the
failed stage, the completed external effects and an actionable repair hint, and
let the LLM decide. Host code owns deterministic integrity, authority boundaries
and truthful receipts only — no task-specific auto-retry, fallback, cleanup,
resume or terminal-flow state machines. Explicitly naming a documented default is
never a different request: an argument whose value is what omitting it already
means — `directory_strategy="direct"` with no `scope_paths` on a shape that cannot
serve the argument, or `workspace_root` naming the Ouroboros repository itself —
takes the omitted path, disclosed in the result; only a value that genuinely asks
for something is refused there, typed, at the earliest layer holding the authority
to judge it, with the repair named.

A producer that knows its call failed publishes that fact typed
(`tool_result._publish_tool_result`, or a first-line `⚠️ IDENTIFIER` the legacy
adapter types): identifier-less `⚠️ prose` and bare `ERROR: ...` are recorded as
a SUCCESSFUL call in `tools.jsonl`, the outcome classifier and the acceptance
packet, losing the failure where the next decision reads it, and a wrapper
carries its inner producer's failure forward instead of a stage-only word. In the
external-executor family the result IS the `ToolResult`, published once, after
all decoration, immediately before its text is returned — the registry accepts it
only when its text IS the string the handler returned — and a supervising wake is
acknowledged on the `supervision_wake_id` that result publishes, never on the
tool's name (ARCHITECTURE §6 "Delegated subagents").

Enforcement: CHECKLISTS item 13(b) scores the prompt-edit discipline; the
recoverable-failure boundary is review-only; `tests/test_typed_tool_refusals.py`
is the shrink-only source lint over returned literals in `ouroboros/tools/`,
flagging identifier-less heads, and its per-file allowlist IS that residual's
disclosure. A same-file swap is invisible to the count; a marker-shaped refusal
the adapter buckets as a warning is a separate, larger residual owned by the
adapter vocabulary; a failure text travelling through a variable, tuple or helper
is outside the lint's reach and pinned by the producer's own tests.

### Documentation contract

`docs/ARCHITECTURE.md` is the present-tense map of the body in BIBLE P6's three
layers: what exists and where it lives, how it flows, and WHY.
`docs/DEVELOPMENT.md` is how the body is changed (authority and shape: "Role and
authority"): one rule per change class, each naming its enforcing test, gate or
CI lane or saying review-only, while ARCHITECTURE, CHECKLISTS, BIBLE, DESIGN and `config.py`
are pointed to, never restated, and the ARCHITECTURE settings and endpoint
tables are test-checked registries of those owners, not second authorities.

A change REPLACES the description of the node it touched; release history lives
in git and the README history table. A `§1` module-tree row
is an address plus the capability and the typed codes, events and state files it
emits as grep targets, plus a `(§N)` pointer. The owning subsection keeps the
complete rationale and trade-offs; internal mechanism may live in the named
module and its docstrings. A disclosed limitation belongs in a book when it
affects authority, money, source completeness, recovery, security, performance,
observability, platform or UI behaviour, or freshness; other follow-ups belong
in issues. Decision codes,
chat-batch answers and reviewer-round labels belong to the commit or PR; an open
limitation may cite its tracker once, as `(issue #NNN)`; probe dates and
measured numbers live only in the inventory below. The authored introduction
under a chapter's H1 IS its compact Low/Nano view: re-read and correct it
whenever the chapter changes.

Track assets with a continuing purpose for the product, contributors, verification,
legal requirements or evidence for public claims, beyond the work that introduced them. Plans, review packets, run receipts
and campaign bookkeeping belong in the external work area or durable task evidence,
not the tracked source tree; a test preserving their presence or wording does not
give them a permanent product role. Retire temporary campaign tooling when its
purpose ends. Keep current behavior and its rationale in their existing owners;
future-work lists and campaign backlog stay outside the tracked product tree.
Generated snapshots with real product, verification or publication consumers remain
valid; optional reports use stdout or an explicit output destination. Existing
review enforces this contract, without automatic deletion or filename matching.

Both are reference BOOKS: an entrypoint plus one chapter file per subject under
`docs/architecture/` or `docs/development/`. Five rules keep the shape honest:

- **One membership list.** The entrypoint's ordered `## Chapters` list IS the
  book; an unlisted chapter file or a listed untracked path fails the validator,
  and there is no second manifest or index.
- **Every source carries one authored introduction.** One paragraph under each
  H1, so an H1 followed straight by a subsection is refused; it is the compact
  view, with no second editable summary corpus.
- **WHY stays where the reader is.** A chapter keeps the rationale for what it
  owns; moved prose REPLACES the description at its destination.
- **One line ending.** `.gitattributes` pins `docs/**/*.md` to LF, so the digests
  and byte offsets carried by source refs and inventories
  stay stable everywhere.
- **Readers ask for a view, not for a file.** `load_governance_doc` composes,
  `context_layout.book_navigation` navigates, `reference_books.book_path_role`
  classifies; `tests/_governance_docs_shared.py` is the one reader tests use: a
  substring pin over an entrypoint `read_text()` passes while testing nothing,
  since that page is only a membership list.

Enforcement: `tests/test_reference_book_validation.py` validates book shape.
Residue — version stamps, decision codenames, owner codes, Cyrillic, bare issue
numbers, "used to / previously" narrative — is caught by the shrink-only check
in `tests/test_docs_sync.py`, which enforces only the case-sensitive matches in
`DOC_RESIDUE_PATTERNS`, outside language-tagged fences and its declared skipped
subsections ("Mutable external-fact inventory" and this one); the untagged
module-tree fence in ARCHITECTURE §1 IS scanned, an owner decision. Each chapter
also carries a byte budget in the official-CI `size_ratchet` lane, raised only
in the diff that needs it, with a reason; local surfaces never block on it.
Equivalent historical prose stays review-only under CHECKLISTS item 7.

### Generality and emergence (P13)

Every non-trivial change picks a level: patch the case in front of you, solve its
class, or build a framework for cases that do not exist yet. The first
fossilizes, the third speculates; aim for the second, which BIBLE P13's invariant
and stronger-mind questions find. The burden is symmetric: shared structure needs
a demonstrated invariant (several real variants, or one already-stable boundary),
an abstraction needs a real class, and an imagined consumer is not one. In doubt
generalize the meaning and the authority, keep the mechanism minimal and local,
and let the next real case pay. Reviewer findings are evidence here, never policy
that overrides it. Review-only.

### Pricing and admission

Never add hand-maintained model-price tables, inherited prefix tariffs or numeric
fallback prices; preserve `cost=None` and `cost_final=false` when no live source
answers the exact route. Unknown price is neither free nor a model-admission
veto, and a known exhausted budget stays enforceable (`tests/test_pricing.py`,
`tests/test_budget_limits.py`; ARCHITECTURE §6 "Budget tracking").

### Anti-pattern: content-derived identity for host-minted records

A host-minted record — a chat message, a task, a binding — has its identity
captured at ingress and passed downstream BY VALUE as a typed reference
(`origin_message_ref`). Never re-derive it by searching logs or state for a row
whose text hash, equality or prefix matches: an LLM-first system rewrites the
text between ingress and use, so content-derived lookup fails on the normal path.
Content hashes stay legitimate only as an INTEGRITY CHECK on an already-known
identity and as content-ADDRESSING where the content IS the identity. Enforce it
as a REQUIRED typed argument at the consuming seam — a valid ref or a closed-enum
absence reason, omission raises — so no future call site can skip it
(`bind_task_to_project(..., *, origin)`, `tests/test_projects_v6640.py`); for
fuzzy entities use `semantic_dedup`, never string equality.

That reference is also the IDENTITY OF THE WORK: a new task id minted from the
same owner message — a promoted root, a mid-run scope call, the timeout retry
replacing a dead attempt — INHERITS the origin's project binding
(`projects_registry.project_id_for_origin`) instead of re-deriving membership
from its own id, because a message is one convertible unit, not one per task id.
A timeout retry binds at RETRY ADMISSION, inside the admitting transaction and
only once cancellation can no longer win the boundary (ARCHITECTURE §6 "Project binding by task
and by origin"; `tests/test_retry_project_binding.py`).

One named exception: a verification RECEIPT with no ingress point reconciles by
ONE TYPED IDENTITY KEY, on the key's kind AND value, never across kinds — a
per-component fallback chain is not an equivalence relation and comes out
order-dependent, while keying fails safe and a false green costs the thing this
surface exists for (`ouroboros/_outcome_receipts.py`,
`tests/test_v678_receipt_reconciliation.py`). The process-tool lane writes no
receipt, so nothing there joins masked-pass reconciliation or the
masked-verification nudge. Four rules generalize:

- **Whatever decides must be what is reported**: reporting reads the deciding
  path through one shared projection, because a reviewer cannot discount evidence
  whose provenance it was told wrongly.
- **A property of a closed set of kinds lives IN the set** (a row per kind plus a
  total lookup that raises on a kind that skipped it).
- **One canonical identity derivation** — comparison, hashing, counting and
  projection read one derived object (canonicalize RAW values → render → bound);
  a normalization that discards what the identity depends on is not one.
- **Changing a stored rendering means versioning it**; reason about the migration
  in BOTH directions, since the false-green direction gets missed and unknown
  must not clear a red.

Disclosed deferred limit: `ouroboros/tools/verify.py` bounds the DURABLE
`artifact_observation` path set at twenty with no omission count — advisory only,
never a gate. Otherwise this section is review-only.

### Anti-pattern: an open default behind a closed exception list

Never reconstruct a producer-owned classification inside a consumer through a
second exception list: "on by default, except for these names" keeps the real
rule in a list that can only go stale, so subtracting three addressing tool names
from a chat turn's tool count mints an empty card for every new addressing tool
and silently loses every renamed one, while the host already carries that fact as
a typed routing annotation. Derive presence from what the record already holds
(`web/modules/chat.js::blockVisible`, `web/modules/chat_activity.js`), let the
host name the special case from the one table it owns
(`tool_capabilities.ROUTING_VERBS`, stamping `routing_action`,
`routing_tool_calls` and `typed_routing_action`), and let it compose an owner
sentence from its typed reason (`project_dialogue.routing_refusal_cause`) that
the browser prints verbatim, keeping no reason→sentence map (ARCHITECTURE §3 "Chat and
Projects"). When the list is the rule, the rule is missing.

A closed vocabulary that DEFINES a rule at its owner is a different thing and is
correct: `tool_capabilities.OBSERVE_WORLD_MUTATION_TOOLS`, compiled by
`ouroboros/consciousness_authority.py`, names the verbs that START work or CHANGE
the world, so a new READ tool reaches Observe by default, pinned against the
catalog's own `mutates_worktree` marker. That module owns a level's two
consequences — `disabled_tools` and the per-task `runtime_mode_cap`, the stricter
of install mode and cap binding even on an advanced/pro/cyber_pro install — and
for a consciousness-origin task `disabled_tools` binds at DISPATCH ONLY, so the
wake keeps an owner turn's cached prefix while every other contract keeps both
enforcement halves (ARCHITECTURE §6 "Background consciousness and Evolution").

### Task-authored messages are never owner text

Who is speaking through a routing act is ONE fact the host mints by value
(`control_routing._routing_issuer`). Never derive it again from a proxy (a
routing contract only chat turns carry, an empty client id, the event's chat id)
and never give the model an argument for it. A consciousness wake-up runs on the
direct lane but nobody typed it, so `is_direct_chat` does not make it an owner
turn (`metadata.initiator == "consciousness"`: it speaks as a task). Draining an
owner message keys the visible receipt without changing the issuer; genuine
owner ingress retains its provenance. A task's own words travel as `KIND_TASK_MESSAGE` with provenance
`independent_task`, never as `KIND_OWNER_TEXT`, and that value lands at three
seams in one change: `owner_mailbox.TASK_MESSAGE_PROVENANCES`,
`deliver_task_message` and `loop_round_limits`. Such words are context the
receiving model judges, so they enter no owner corpus: `owner_source_sha256`, the
post-drain growth check that supersedes a paid acceptance panel, and the
acceptance premises stay the owner's (`tests/test_task_authored_messages.py`;
ARCHITECTURE §6 "Owner routing verbs").

### Anti-pattern: a chat id tested for truth

A chat id is a VALUE, not a boolean: `HIDDEN_CHAT_ID` (0) is a REAL destination,
the hidden partition, absence is `None`, and a negative id is synthetic A2A
traffic. `if chat_id:` does two wrong things at once — it drops a partition-bound
notice AND re-routes hidden work to the owner's main chat, so a whole
`ouroboros run` goes invisible while its children surface in Main as a nameless
card. Use the two normalizers, never a third:
`message_bus.notification_chat_route` for where a notice goes,
`message_bus.coerce_chat_identity` for a row's address. Address a task once at
admission (`log_addressing.ingress_chat_id`) and pass the value downstream; a
producer sending to the owner DIRECTLY resolves the task's durable project
binding AT EMISSION through `log_addressing.resolve_project_chat` and puts it ahead
of the row's chat, because a task bound to
a project after admission still carries the chat it was born in. Task type is
never source provenance. The admission contract and the partition are §5 and §12;
enforcement is `tests/test_chat_id_truthiness_guard.py`, the source lint that
keeps the class closed, whose allowlist is where a deliberate exception states
its reason.

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
| `ouroboros/reviewer_slot_config.py::_ACCEPTANCE_API_PANEL_MEASURED` | Historical API-panel comparison: approximately 12 s / $0.07 per model row per task (median of the 2026-09-01 OSWorld traces); 75 s / $0.82 for a three-row panel on ProgramBench | Workload and route dependent | The named measurement constant used by the one-time delivery disclosure | Repeat the same workload with recorded model, route and usage | An old comparison can be mistaken for a current tariff or a subscription-cost estimate | Keep the date and workload visible; current usage owns money, and session delivery spends subscription time |
| `ouroboros/llm_claudexor.py::cache_key_for_model` | The 2026-09-17 measurement found Codex prefix reuse across conversations requires one `prompt_cache_key` + `session_id`, while per-conversation turn states remain valid under that shared session | Provider dependent | Dated measurement beside the key derivation | Re-measure cache reads and turn state across two conversations | A stale positive pays cold prefixes or breaks turn state | Re-measure before changing the key scope |
| `ouroboros/llm_openai_compatible.py` DeepSeek send projection | The 2026-09-03 probe found thinking accepts only `auto`/`none` tool choice; required/named calls returned 400 on both probed v4 models | Provider dependent | Dated probe recorded beside the send projection and its transport tests | Re-probe the exact endpoint/model when that dialect changes | Removing the projection too early breaks forced calls; keeping it after a provider change may suppress supported thinking | Revalidate the wire contract before changing the projection; keep its effect disclosed |

### Provider Independence

One configured provider must be sufficient for the agent loop, commit review,
scope policy, safety, and context/memory flows; core capability must not acquire
a hidden OpenRouter or second-provider dependency. (CHECKLISTS item 2(h) and
ARCHITECTURE both point here; this is the SSOT sentence.)

Tool-schema changes are provider-contract changes: every shipped built-in schema
must pass general JSON Schema and the known cross-provider subset over the
complete registry, trusted integration CI sends that registry in one bounded tool
canary per supported provider family/API surface in the transport Main uses, and
pull-request CI stays secretless. Malformed native arguments and invalid schemas
stay red, diagnostics limited to structural facts, hashes and parse position;
never add a prose parser, provider hop or unbounded retry to make that contract
green (ARCHITECTURE §8 "CI topology").

Adding or changing a provider updates one coherent route contract:

1. credential/readiness detection and exact model-id migration;
2. Main/Light/Fallback and reviewer-slot defaults, never overwriting explicit
   owner choices;
3. canonical tool/reasoning/image/cache intent at `llm.py`, wire projection and
   exact-route recovery in the small transport leaves;
4. nullable pricing/settlement and truthful capability omissions;
5. review and scope routing with sourced context-window evidence for send sizing;
6. direct-provider and single-provider regression tests;
7. the route's real streamed wire recorded (redacted) into
   `tests/fixtures/llm_wire/` and replayed — a hand-written fixture is not
   evidence of a dialect; record with `curl -N` or export the private
   `physical_stream` blob the runtime retains.

Local-only installs keep their local route; unreachable shipped remote defaults
may be cleared, explicit owner values may not. Scope runs in every context-size
mode; window evidence governs send sizing, not review authority. Reading coverage
is diagnostic under BIBLE P3: missing observations do not discard a received
verdict or remove a responding reviewer from quorum.
Current model ids and defaults belong in code/config, not here. Use
`provider_models.ACTIVE_MODEL_SETTING_KEYS` for any new active consumer;
`LEGACY_MODEL_SETTING_KEYS` is migration/history only, and `OUROBOROS_MODEL_HEAVY`
with its paired `USE_LOCAL_HEAVY` may seed an explicit configured API actor while
the canonical list is absent but must never become an active slot,
startup-readiness signal, test probe or fallback — no consumer gets its own Heavy
exclusion.

Provider facts that bind routing: the `-pro` suffix is an OpenRouter routing
slug, not an official OpenAI model id, so a direct OpenAI Chat slot uses the plain
Sol id (the slug in Chat Completions is a guaranteed 404) — a compatibility
constraint, not a mutable capability table; direct OpenAI tool conversations stay
on Chat Completions and a model-name prefix is never admission authority;
DeepSeek is the second effort-carrying route, its `reasoning_effort` keyed on the
provider id rather than a name prefix or capability field, so a hand-built target
cannot silently drop it; direct Anthropic is the deliberate exception to a purely
reconstructed provider transcript, and no effort-to-`budget_tokens` policy is
synthesized (ARCHITECTURE §6 "Context fitting, retry, and compaction", ARCHITECTURE §7 "LLM output token
budgets"). A provider-specific optional feature may be unavailable elsewhere, but
the core loop degrades explicitly rather than crashing or rerouting.

Canonical assistant history and tool schemas are function-shaped across
providers: never a second stored transcript for a provider dialect. Apply static,
semantics-preserving wire normalization before request-wire binding and keep
learned recovery separate — all learned request-shape adaptation goes through the
one provider-neutral driver (`ouroboros/request_wire_contract.py`: exact-route
identity, closed action vocabulary, shared TTL, never executes provider prose or
switches route), never a second driver, and explicit `none` is never durable.
