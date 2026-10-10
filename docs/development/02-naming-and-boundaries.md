# Naming and boundaries

This chapter owns the rules that keep the body legible from outside: naming and
dependency direction, the CLI contract, cognitive quality and LLM-first affordances,
where a fact may live in the two books, the anti-patterns behind real incidents,
external facts and provider independence. Each rule names what enforces it or says
that only review does, and points to ARCHITECTURE for mechanism.

- Code, comments, docstrings, commit messages and product UI strings are English source
  copy; a non-English owner reads the install's translation memory
  (`state/i18n/<tag>.json`; ARCHITECTURE §1 `i18n_memory.py`), never a repository
  locale file: no language is enumerated in code, durable rows keep English plus typed
  codes, surfaces translate at render.
- PEP 8 names (`snake_case`, `PascalCase`, `UPPER_SNAKE_CASE`) that say the observable
  responsibility and authority, not the implementation fashion; a function module over
  a class with no lifecycle.
- Contracts are typed shapes, not service objects; a manager needs lifecycle or mutable
  state. Tools stay thin `{verb}_{noun}` functions (validate input, call the owner,
  format a result); no universal `{Domain}Service`, `{Platform}Gateway` or class layer.
- Dependency direction: UI/CLI → inbound gateway → domain owner; runtime policy →
  host-owned contract → outbound adapter (`ouroboros/gateway/` in,
  `ouroboros/gateways/` out; ARCHITECTURE §1 "Gateway Boundary v1"). Provider- and
  transport-specific decisions never flow back into core policy.
- Chat authorship is stamped by the producer and survives persistence and replay; never
  infer it from text or promote host-selected intermediate output to a model final,
  because text inference lets a model or a task speak as the owner
  (`tests/test_terminal_provenance.py`).

Enforcement: CHECKLISTS item 11 `development_compliance` (a)/(b); the "Guard extracted
transport imports stay out of core" step in `.github/workflows/ci.yml`; the rest is
review-only.

### CLI and headless work

- CLI commands parse, call the existing gateway/scheduler and render text or typed
  JSON/JSONL/SSE; no second task state machine, scheduler or generic file manager
  (ARCHITECTURE §1 "CLI / Headless Boundary").
- An external workspace binding changes the contextual repository
  (`ToolContext.active_repo_dir()`), never the system repository governance binds to;
  every resource address goes through the one resolver guards and handlers use, and a
  relative path is never sanitized before an outside address is refused.
- Project focus changes the default target, not the tool surface: advisory, reviewed
  commit, rollback, promotion, restart and runtime control keep their
  system-repository contracts; `executor_ref` selects a process backend, not a sandbox;
  GitHub tools take the shell binding, and project focus overrides ambient `GH_REPO`
  (`ouroboros/tools/github.py`).
- A process cwd decides relative paths, not the root task's write authority; semantic
  permission is never reconstructed from command words or guessed effects
  (ARCHITECTURE §6 "Safety and runtime mode"). No SSH or OS sandbox is promised.
- Credential authority is never inferred from a directory name, nor owner input refused
  on a suffix or word inside a file name (dotenv spellings are the one tail rule):
  owner credential locations are the physical list
  `credential_shapes.owner_credential_locations`, so an unlisted store keeps ordinary
  access (ARCHITECTURE §6 "Credential mutation and diagnostic redaction").

Enforcement: `tests/test_headless_cli.py`, `tests/test_external_workspace_access.py`;
the no-second-scheduler rule is review-only. Mechanics: ARCHITECTURE §6 "Tool
capability and execution".

### Cognitive quality

Never lower model quality, reasoning effort, output budget or context breadth as an
incidental latency or cost optimization (BIBLE P1). An intentional narrowing is a
recorded decision carried in plan, docs, tests and evidence; outside Cyber Pro it
belongs to the owner, and Cyber's own configuration authority follows BIBLE P0/P3
without rewriting earlier call facts. Review-only: CHECKLISTS items 10 and 7, whose
named failure class is an accidental narrowing.

### LLM-first affordances

Never repair a semantic tool-choice failure with one more keyword hint in
`prompts/SYSTEM.md`. A model mistake alone does not justify a new host contract:
establish a missing capability or piece of information, or an independently valid
requirement, then repair the schema or affordance at the point of need. Never freeze
the model's reasoning, dialogue representation or collaboration strategy to make one
incident testable: SYSTEM accretion trains around that incident, bloats the resident
prefix and forks the authority.

Whether, when and how often to tell the owner, retry, wait or substitute is the prompt's
and model's judgment; a threshold, timer or counter standing in for it is the if-else
selection BIBLE P5 forbids, whoever proposes it, while numbers still bound physical,
safety, budget, transport and evidence expiry. A missing channel is a capability gap:
`send_user_message(destination="main")` gives a registered Project root or a
host-attested owner-origin root a Main voice; no chat number, Main's included, is owner
proof.

`prompts/SYSTEM.md` is tier-0 for every Main and task profile in both context modes and
competes with the task for context. It carries identity and tone, the decision loop,
cross-tool policy, prohibitions and safety invariants stated once, and the memory
contract's resident rule: active summaries stay in the index, archived sources remain
addressable and explicitly listable (ARCHITECTURE §6 "Durable memory and project focus").
A missing carrier stays a visible gap. Load-bearing floor rules stay in its preamble, the part
overflow compaction keeps (ARCHITECTURE §6 "Context fitting, retry, and compaction").
It never carries how a tool works: parameters, recipes, typed outcomes and "when to
choose it" belong to the `get_tools()` schemas selected for the acting profile, so a prompt sentence
about a schema is a drifting second copy and a new tool
needs no SYSTEM.md mention. Every prompt change reports its before/after byte size in
the commit or PR.

Recoverable tool failures are evidence for the next model turn, not triggers for a
host-authored recovery workflow: return a typed, redacted result naming the failed
stage, the completed external effects and a repair hint, and let the model decide; host
code owns deterministic integrity, authority boundaries and truthful receipts only.
Naming a documented default is never a different request: a value that means what
omission means takes the omitted path, disclosed in the result; a value that genuinely
asks for something is refused typed, at the earliest layer with authority to judge it,
naming field, value and repair in one reply, because a refusal that only restates its
rule is retried unchanged (`tools/arg_feedback`).

A producer that knows its call failed publishes that fact typed
(`tool_result._publish_tool_result`, or a first-line `⚠️ IDENTIFIER`): identifier-less
`⚠️ prose` and bare `ERROR: ...` are recorded as a successful call in `tools.jsonl`, the
outcome classifier and the acceptance packet, so the failure is lost where the next
decision reads it. A wrapper carries its inner producer's failure forward, and a
supervising wake is acknowledged on the published `supervision_wake_id`, never on the
tool's name (ARCHITECTURE §6 "Delegated subagents").

Enforcement: CHECKLISTS item 15(b) scores the prompt-edit discipline;
`tests/test_typed_tool_refusals.py` is the shrink-only source lint over returned
literals in `ouroboros/tools/`, its per-file allowlist the disclosure of the
identifier-less residual; the owner-judgment and recoverable-failure boundaries are
review-only.

### Documentation contract

`docs/ARCHITECTURE.md` is the present-tense map of the body in BIBLE P6's three layers:
what exists and where, how it flows, why. `docs/DEVELOPMENT.md` is how the body is
changed: one rule per change class, each naming its enforcing test, gate or CI lane or
saying review-only. ARCHITECTURE, CHECKLISTS, BIBLE, DESIGN and `config.py` are pointed
to, never restated; the ARCHITECTURE settings and endpoint tables are test-checked
registries of those owners, not second authorities.

A sentence belongs in a book when Ouroboros, reading the map before touching the code,
needs it to find the right place, to keep an invariant other code relies on, or to keep
a decision whose WHY would otherwise be lost. A `§1` module-tree row is one line
(address, capability, the grep targets of a cross-module contract, a `(§N)` pointer);
the owning subsection keeps the WHY at its node. How a module works inside (call order,
locks, field lists, guard order, enumerated edge cases and typed refusals) belongs in
the module, its docstring or its test. A change replaces the description of the node it
touched; release history lives in git and the README history table. A limitation is
stated once, where an owner or maintainer would otherwise act wrongly; other follow-ups
are issues. Decision codes, chat-batch answers and reviewer-round labels belong to the
commit or PR; an open limitation may cite its tracker once, as `(issue #NNN)`; probe
dates and measured numbers live only in "External facts: unknown is not no". The
introduction under a chapter's H1 is its compact Low/Nano view: correct it whenever the
chapter changes.

Track assets with a continuing purpose (product, contributors, verification, legal
requirements, evidence for public claims). Plans, campaign backlog, review packets and
run receipts belong in the external work area or durable task evidence, not the
tracked tree; a test preserving their presence gives them no product role. Retire
campaign tooling when its purpose ends. Review enforces this, without automatic
deletion or filename matching.

Both are reference books: an entrypoint plus one chapter file per subject under
`docs/architecture/` or `docs/development/`. The entrypoint's ordered `## Chapters`
list is the one membership list; every chapter carries one authored introduction under
its H1 (an H1 followed straight by a subsection is refused); moved prose replaces the
description at its destination; `.gitattributes` pins `docs/**/*.md` to LF so the
digests and byte offsets in source refs and inventories stay stable; readers ask for a
view, not a file (`load_governance_doc` composes, `context_layout.book_navigation`
navigates, `reference_books.book_path_role` classifies,
`tests/_governance_docs_shared.py` is the one reader tests use, because a substring
pin over an entrypoint `read_text()` passes while testing nothing).

Enforcement: `tests/test_reference_book_validation.py` validates book shape. Residue
(version stamps, decision codenames, owner codes, Cyrillic, bare issue numbers, "used
to / previously" narrative) is caught by the shrink-only check in
`tests/test_docs_sync.py`: the case-sensitive `DOC_RESIDUE_PATTERNS`, outside
language-tagged fences and the declared skipped subsections ("External facts: unknown
is not no" and this one); the untagged module-tree fence in ARCHITECTURE §1 is
scanned.

Size is pairwise: a change ends each composed book, entrypoint included, no
larger than at its event base, so text it adds is paid by shortening the same
book. A stored budget only moved up and per-change grants made every growth a
cheap self-raise; comparing tip with base leaves no number to raise. The
official-CI `size_ratchet` lane blocks growth on owner, member and collaborator
pull requests and on pushes to `ouroboros`; outside contributors and forks receive
warnings. The repository owner's `book-growth` label exempts its PR and the exact
commit that lands it. Local surfaces never block: the edit
tools, `plan_task`, the free `preflight_review(deterministic_only=True)` diagnostic
and `codebase_health` report the running balance as a fact. Contribution is measured
against the displayed merge-base of the cached official development ref; an unknown
base stays unknown, and no network refresh is implied. Free diagnostics label book
bytes as worktree separately from the selected release-metadata source.
Equivalent historical prose stays review-only under CHECKLISTS item 5.

### Current state first (ARCHITECTURE invariant 10)

Apply ARCHITECTURE §10 invariant 10; enforcement: "Invariant: Projection over replay (hot readers of growing stores)" below.

### Generality and emergence (P13)

Every non-trivial change picks a level: patch the case in front of you, solve its
class, or build a framework for cases that do not exist yet. The first fossilizes, the
third speculates; aim for the second. The burden is symmetric: shared structure needs a
demonstrated invariant (several real variants, or one already-stable boundary), an
abstraction needs a real class, and an imagined consumer is not one. In doubt,
generalize the meaning and the authority, keep the mechanism minimal and local, and let
the next real case pay. Reviewer findings are evidence here, never policy. Review-only.

### Pricing and admission

Never add hand-maintained model-price tables, inherited prefix tariffs or numeric
fallback prices: they go stale silently and look authoritative. Preserve `cost=None`
and `cost_final=false` when no live source answers the exact route; unknown price is
neither free nor a model-admission veto, and a known exhausted budget stays enforceable
(`tests/test_pricing.py`, `tests/test_budget_limits.py`; ARCHITECTURE §6 "Budget
tracking"; the pricing row of "External facts: unknown is not no").

### Anti-pattern: content-derived identity for host-minted records

A host-minted record (a chat message, a task, a binding) has its identity captured at
ingress and passed downstream by value as a typed reference (`origin_message_ref`).
Never re-derive it by searching logs or state for a row whose text hash, equality or
prefix matches: an LLM-first system rewrites text between ingress and use, so
content-derived lookup fails on the normal path. Content hashes stay legitimate as an
integrity check on a known identity and as content-addressing where the content is the
identity. Enforce the reference as a required typed argument at the consuming seam (a
valid ref or a closed-enum absence reason; omission raises), so no call site can skip it
(`bind_task_to_project(..., *, origin)`, `tests/test_projects_v6640.py`); for fuzzy
entities use `semantic_dedup`, never string equality. That reference is also the
identity of the work: every task id minted from the same owner message inherits the
origin's project binding, because a message is one convertible unit (ARCHITECTURE §6
"Project binding by task and by origin"; `tests/test_retry_project_binding.py`).

One named exception: a verification receipt with no ingress point reconciles by one
typed identity key, never across kinds, because a per-component fallback chain is not
an equivalence relation and comes out order-dependent, while keying fails safe
(`ouroboros/_outcome_receipts.py`, `tests/test_v678_receipt_reconciliation.py`). Four
rules generalize: whatever decides is what is reported, through one shared projection;
a property of a closed set of kinds lives in the set, with a total lookup that raises
on a kind that skipped it; one canonical identity derivation (canonicalize → render →
bound) feeds comparison, hashing, counting and projection alike; a stored rendering
changes only with a version, reasoned in both directions, because unknown must not
clear a red. Review-only.

### Anti-pattern: an open default behind a closed exception list

Never reconstruct a producer-owned classification inside a consumer through a second
exception list: "on by default, except for these names" keeps the real rule in a list
that can only go stale (subtracting three addressing tool names from a chat turn's tool
count mints an empty card for every new addressing tool), while the host already
carries that fact as a typed routing annotation. Derive presence from what the record
holds (`web/modules/chat.js::blockVisible`), let the host name the special case from
the one table it owns (`tool_capabilities.ROUTING_VERBS`, stamping `routing_action`,
`routing_tool_calls` and `typed_routing_action`) and compose the owner sentence from
its typed reason (`project_dialogue.routing_refusal_cause`), printed verbatim by the
browser (ARCHITECTURE §3 "Chat and Projects"). When the list is the rule, the rule is
missing. A closed vocabulary that defines a rule at its owner is correct:
`tool_capabilities.OBSERVE_WORLD_MUTATION_TOOLS` names the verbs that start work or
change the world, so a new read tool reaches Observe by default (ARCHITECTURE §6
"Background consciousness and Evolution").

### Task-authored messages are never owner text

Who is speaking through a routing act is one fact the host mints by value
(`control_routing._routing_issuer`); never derive it again from a proxy (a routing
contract only chat turns carry, an empty client id, the event's chat id, the direct
lane a consciousness wake-up also runs on) and never give the model an argument for it.
A task's own words travel as `KIND_TASK_MESSAGE` with provenance `independent_task`,
never as `KIND_OWNER_TEXT`; the value lands at three seams in one change
(`owner_mailbox.TASK_MESSAGE_PROVENANCES`, `deliver_task_message`,
`loop_round_limits`) and enters no owner corpus (`owner_source_sha256`, the acceptance
premises), because a task-minted message there would let a task authorize its own
acceptance (`tests/test_task_authored_messages.py`; ARCHITECTURE §6 "Owner routing
verbs").

### The owner corpus archives inputs; the owner door's stamp is the only authority

A task-authored objective copies only its retained owner corpus, never the draft. Other
runs record their first user turn for acceptance, Safety and reflection, labelled by
what the host knows: `initial_user` when owner routing stamped the run
(`metadata.origin_message_ref` or `origin_suppressed`, inherited by value by a promoted
root), `initial_text` otherwise. `dialogue_provenance.run_origin` mints that fact once
from typed fields (`owner_ingress`): an owner turn is a direct turn the door stamped,
never a lane, a client id or an inherited stamp, and the stamp is reserved on
`/api/tasks` and schedule templates. Neither label decides what work was accepted; the
task contract and the owner's recorded answers do.

### Anti-pattern: a chat id tested for truth

A chat id is a value, not a boolean: `HIDDEN_CHAT_ID` (0) is a real destination, the
hidden partition; absence is `None`; a negative id is synthetic A2A traffic.
`if chat_id:` drops a partition-bound notice and re-routes hidden work to the owner's
main chat at once. Use the two normalizers, never a third:
`message_bus.notification_chat_route` for where a notice goes,
`message_bus.coerce_chat_identity` for a row's address. Address a task once at
admission (`log_addressing.ingress_chat_id`) and pass the value downstream; task type
is never source provenance. Admission contract and partition: ARCHITECTURE §5 and §12.
Enforcement: `tests/test_chat_id_truthiness_guard.py`, the source lint whose allowlist
is where a deliberate exception states its reason.

### External facts: unknown is not no

A fact about a model or provider that Ouroboros does not own (what it accepts, holds,
costs, how it is spelled, whether it is still served) changes without a release.

1. Evidence of the exact route, not lists: the route's own metadata, a dated probe
   recorded beside a provider-keyed wire projection with a recovery path, or the
   owner's setting. A model name, prefix or family is never that evidence, nor is
   another route's catalog.
2. Unknown is not no. Without evidence the owner's input, the tool and the explicit
   request go through unchanged and the route answers; no shipped literal may drop
   input, refuse a tool, rewrite a saved choice or narrow a request.
3. A refusal is an observation about one request. It repairs that attempt and is
   disclosed as what happened; a durable "no" needs evidence that states the absence
   for that scope.
4. A limit of our own transport is a fact about Ouroboros, declared at the lane and
   named as such in what the model reads.
5. A retained wire fact (the table) keeps source and date, exact scope and what happens
   when it goes stale either way; requested, sent and provider-reported values stay
   distinct.

Guards: `tests/test_image_capability_contract.py` (an unseen model id keeps its image on
every image-capable transport), `tests/test_model_name_invariance.py` (with empty
evidence the physical payload does not depend on the model name; its declared
exceptions are rows below), `tests/test_tristate_truthiness_guard.py` (a yes/no/unknown
answer is never tested for truth). The table is maintenance provenance, grants no
runtime authority, and probe detail lives in the docstring at each location.

| Location | Fact | Evidence & re-probe | If stale |
|---|---|---|---|
| `ouroboros/llm_attempt.py::supports_message_cache_control` | Families whose message cache controls OpenRouter accepts or translates (`anthropic/`, `google/gemini-`, `openai/`) | `openai/` rests on the OpenRouter caching guide cited in the docstring, not a live probe; re-probe before widening, never by name resemblance | False positive invalidates a request; false negative loses the prompt cache |
| `ouroboros/llm_attempt.py::openai_family_model` | OpenAI's public API looks a cache up only at message ends unless a breakpoint is explicit | Probe in the docstring; re-probe `cached_tokens` across two conversations before widening; never match on a substring such as `gpt` | Stale positive moves the cache boundary for a prefix-caching family; stale negative pays cold prefixes |
| `ouroboros/llm_openai_compatible.py` OpenRouter `extra_body.provider.require_parameters` for `anthropic/` | Anthropic models on OpenRouter route only to endpoints accepting every sent parameter | Undated; OpenRouter per-endpoint `supported_parameters` (`/api/v1/models/{author}/{slug}/endpoints`); a declared exception of `tests/test_model_name_invariance.py` | Stale positive narrows failover; stale negative routes to an endpoint that drops or refuses a parameter |
| `ouroboros/reasoning_artifacts.py::SIGNED_PORTABLE` and its sealed classifier | Which families' sealed reasoning artifacts survive a same-model cross-provider replay; readable artifacts are portable by shape | Short vouched roster plus a shape-first classifier that fails closed on unreadable artifacts; extend only by a fresh cross-provider replay probe of the exact family | False positive 400s the replayed turn (strip-and-retry is the net); false negative pins a portable transcript to one endpoint |
| `ouroboros/provider_models.py::_ANTHROPIC_MODEL_ALIASES` / `migrate_model_value` | Direct-provider id spelling compatibility | Shipped mapping; a catalog confirms a current id, not whether a saved spelling was intentional; retire an alias only through a documented window | Removing an alias breaks upgrades; guessing one silently reroutes |
| `ouroboros/server_runtime.py::_PRIOR_SHIPPED_SLOT_DEFAULTS`, `_SCOPE_REVIEW_PRIOR_DEFAULTS`, `_normalize_direct_scope_review_model` | Only exact former Ouroboros defaults migrate; no list of retired external models exists | Release history plus current `SETTINGS_DEFAULTS`, under regression tests; a catalog shows availability, never intent | Over-broad migration overwrites an explicit owner choice; a retired model fails loudly at first call |
| `ouroboros/pricing.py::get_pricing`; `ouroboros/llm_pricing.py::fetch_openrouter_pricing` / `fetch_cloudru_pricing` | Exact-route model tariffs | Live catalog fetch with nullable unknowns; provider-settled usage wins; no runtime tariff table | A static price looks authoritative after becoming wrong and corrupts admission |
| `ouroboros/llm_claudexor.py::cache_key_for_model` | Codex prefix reuse across conversations needs one `prompt_cache_key` + `session_id`; per-conversation turn states stay valid under it | Measurement in the docstring; re-measure cache reads and turn state across two conversations before changing the scope | Stale positive pays cold prefixes or breaks turn state |
| `ouroboros/llm_openai_compatible.py` DeepSeek send projection | Thinking accepts only `auto`/`none` tool choice; required and named calls return 400 | Probe beside the projection and its transport tests; re-probe the exact endpoint when the dialect changes | Removed early, forced calls break; kept after a change, supported thinking is suppressed |
| `ouroboros/provider_models.py::ZAI_REASONING_EFFORT_ALIASES` (Z.ai send projection) | GLM-5.3 accepts only `low`/`high`/`max`, serves an absent tier at max, cannot disable thinking, takes forced tool_choice with thinking on; GLM-5.2 accepts the wider scale | Probe beside the projection and its tests; re-probe when the enum or a GLM release changes | Dropping the projection bills every call at max; a stale one rewrites tiers the provider accepts |
| `ouroboros/provider_models.py::PROVIDER_TOOL_SCHEMA_LIMITS` | Tool-schema ceilings per execution provider; an absent provider declares no ceiling | One bounded send of ceiling + 1 schemas to the exact route (the provider canaries); change a number only on a documented limit or such a send | Too high, requests above the ceiling are refused; too low or invented, overflow schemas leave the request needlessly (still loadable) |

### Provider Independence

One configured provider must be sufficient for the agent loop, commit review, scope
policy, safety, and context/memory flows; core capability must not acquire a hidden
OpenRouter or second-provider dependency. (CHECKLISTS item 11(h) and ARCHITECTURE both
point here; this is the SSOT sentence.)

Tool-schema changes are provider-contract changes: validate the full shipped registry
against JSON Schema and the cross-provider subset, then run the bounded canaries
through Main's transports; PR jobs stay secretless, malformed arguments and invalid
schemas stay red, and no prose parser, provider hop or unbounded retry is added to make
the contract green (ARCHITECTURE §8 "CI topology").

Adding or changing a provider updates one coherent route contract: credential and
readiness detection with exact model-id migration; Main/Light/Fallback and
factory review-pool defaults that never overwrite explicit owner choices; canonical
tool/reasoning/image/cache intent at `llm.py`, with wire projection and exact-route
recovery in the small transport leaves; nullable pricing and truthful capability
omissions; review and scope routing with sourced context-window evidence for send
sizing; direct-provider and single-provider regression tests; and the route's real
streamed wire recorded (redacted) into `tests/fixtures/llm_wire/` and replayed, because
a hand-written fixture is not evidence of a dialect.

Local-only installs keep their local route; unreachable shipped remote defaults may be
cleared, explicit owner values may not. Scope runs in every context-size mode; window
evidence governs send sizing, not review authority. Reading coverage is diagnostic
under BIBLE P3: missing observations do not discard a received verdict or remove a
responding reviewer from quorum. Current model ids and defaults belong in code and
config, not here: a new active consumer reads
`provider_models.ACTIVE_MODEL_SETTING_KEYS`; `LEGACY_MODEL_SETTING_KEYS` is migration
only, and `OUROBOROS_MODEL_HEAVY` with its paired `USE_LOCAL_HEAVY` only seeds a
configured API actor while the canonical list is absent, never an active slot,
readiness signal, test probe or fallback.

Provider facts that bind routing: the `-pro` suffix is an OpenRouter routing slug, not
an OpenAI model id, so a direct OpenAI Chat slot uses the plain Sol id; direct OpenAI
tool conversations stay on Chat Completions, and a model-name prefix is never admission
authority; DeepSeek and Z.ai carry `reasoning_effort` through projections keyed on the
provider id; direct Anthropic is the deliberate exception to a purely reconstructed
provider transcript, and no effort-to-`budget_tokens` policy is synthesized
(ARCHITECTURE §7 "LLM output token budgets"). A provider-specific optional feature may
be unavailable elsewhere; the core loop degrades explicitly rather than crashing or
rerouting.

Canonical assistant history and tool schemas are function-shaped across providers,
never a second stored transcript for a provider dialect. Static, semantics-preserving
wire normalization runs before request-wire binding, and every learned request-shape
adaptation goes through the one provider-neutral driver
`ouroboros/request_wire_contract.py` (exact-route identity, closed action vocabulary,
shared TTL; it never executes provider prose or switches route); explicit `none` is
never durable.
