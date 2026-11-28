# Naming and boundaries

Rules keeping the body legible from outside: naming, dependency direction, CLI contract, cognitive quality, LLM-first affordances, book placement, incident anti-patterns, external wire facts and provider independence. Each rule names its enforcement or marks it review-only and cites ARCHITECTURE for mechanism.

- English only in code, comments, docstrings, commits and UI strings; non-English installs use translation memory, never repo locale files (ARCHITECTURE §10 invariant 29).
- PEP 8 names (`snake_case`, `PascalCase`, `UPPER_SNAKE_CASE`) matching observable authority. Prefer function modules over stateless classes: contracts are typed shapes, managers require lifecycle/mutable state, and tools stay thin `{verb}_{noun}` wrappers (validate input, call owner, format result)—no universal `{Domain}Service`, `{Platform}Gateway` or class layer.
- Dependencies run UI/CLI → inbound gateway → domain owner, and runtime policy → host-owned contract → outbound adapter (ARCHITECTURE §1 "Gateway Boundary v1"); provider/transport details never enter core policy.
- Producers stamp chat authorship before persistence and replay; never infer it from text or promote host-selected intermediate output to a model final (`tests/test_terminal_provenance.py`).

Enforcement: CHECKLISTS item 11 `development_compliance` (a)/(b); `.github/workflows/ci.yml` "Guard extracted transport imports stay out of core"; rest review-only.

### CLI and headless work

- CLI commands parse input, call the shared gateway/scheduler and emit text or typed JSON/JSONL/SSE—no second state machine, scheduler or file manager (ARCHITECTURE §1 "CLI / Headless Boundary").
- External workspaces change `ToolContext.active_repo_dir()`, never governance's system repo; every path crosses the single guard/handler resolver, refusing outside relative paths before sanitizing.
- Project focus sets only the default target (`ouroboros/tools/github.py` prefers it over ambient `GH_REPO`); advisory, reviewed commit, rollback, promotion, restart and runtime control keep system-repo contracts.
- Process cwd resolves relative paths, never write authority; never infer permission from command words or guessed effects (ARCHITECTURE §6 "Safety and runtime mode"), nor promise an SSH/OS sandbox.
- Credential authority comes solely from `credential_shapes.owner_credential_locations` (ARCHITECTURE §6 "Credential mutation and diagnostic redaction"), never directory names; never refuse owner input on filename suffixes or words (dotenv spellings are the sole tail rule).

Enforcement: `tests/test_headless_cli.py`, `tests/test_external_workspace_access.py`; no-second-scheduler review-only; mechanics: ARCHITECTURE §6 "Tool capability and execution".

### Cognitive quality

Never lower model quality, reasoning effort, output budget or context breadth for latency or cost; intentional narrowing requires recorded plan, docs, tests and evidence under BIBLE P1. Review-only: CHECKLISTS items 10 and 7 (accidental narrowing).

### LLM-first affordances

Never fix semantic tool-choice mistakes with `prompts/SYSTEM.md` keyword hints or freeze reasoning, dialogue or collaboration for testability—SYSTEM accretion overfits incidents, bloats the prefix and forks authority. Prove a missing capability, info gap or independent requirement, then fix the schema or affordance at point of need.

Notifying the owner, retrying, waiting or substituting is model/prompt judgment, never thresholds, timers or counters, whoever proposes them (BIBLE P5); numbers bound physical, safety, budget, transport and expiry limits only. `send_user_message(destination="main")` gives registered Project or host-attested owner-origin roots a Main channel; no chat id, Main's included, proves owner identity.

`prompts/SYSTEM.md` (tier-0 across profiles/modes) holds how Ouroboros decides: identity/tone, decision loop, cross-capability judgment, once-stated prohibitions/safety invariants and the memory contract (ARCHITECTURE §6 "Durable memory and project focus"):

- Put capability mechanics (parameters, recipes, statuses, costs, recovery, quirks) at the choice site: surviving schema leads (the only host text that may recommend its own tool), typed results/refusals, or conditional host text (code-rendered prose included, stating facts and role duties only, never which choice to make).
- Keep a capability clause in `SYSTEM.md` only as last home while the thinnest supported request (Nano, parameter-prose-free projection, bookless child) cannot choose without it, leaving missing carriers visible and naming the gap in the commit. New tools need no mention; removals name the line readers still read. Every prompt change reports before/after byte size and behaviour-check outcome (or that none ran) in the commit or PR.

Recoverable failures are next-turn evidence, never host recovery workflows: return a typed redacted failure naming stage, completed effects and repair hint (host owns deterministic integrity, authority boundaries and truthful receipts only). Values meaning omission, or single-meaning extra keys, take that path and disclose; values asking for more or ambiguous keys refuse typed at the earliest entitled layer, naming field, value and repair in one reply—restated rules invite unchanged retries (`tools/arg_feedback` docstring).

Producers publish known failures typed (`tool_result._publish_tool_result` or leading `⚠️ IDENTIFIER`); identifier-less `⚠️ prose` or `ERROR: ...` records as success in `tools.jsonl`, outcome classifier and result subject, hiding failures. Wrappers preserve inner failures; supervising wakes ACK `supervision_wake_id`, never tool name (ARCHITECTURE §6 "Delegated subagents").

Enforcement: CHECKLISTS item 14(a); `tests/test_typed_tool_refusals.py` (shrink-only lint over `ouroboros/tools/` literals; per-file allowlist discloses the identifier-less residual); owner-judgment and recoverable-failure rules review-only.

### Documentation contract

`docs/ARCHITECTURE.md` is the present-tense BIBLE P6 map (what/where, flows, why); `docs/DEVELOPMENT.md` states one rule per change class, naming its invariant home (home test or owner, never one test file per rule), gate/CI lane or review-only. Cite ARCHITECTURE, CHECKLISTS, BIBLE, DESIGN and `config.py` instead of restating them; ARCHITECTURE's settings and endpoint tables are test-checked registries, not second authorities.

Keep prose that locates mechanisms, preserves cross-code invariants or gives WHY: `§1.1` rows stay one line (address, capability, cross-module grep targets, `(§N)` pointer) while subsections hold rationales. Replace touched descriptions, keep chapter H1 intros accurate (an Architecture chapter's rides every request in the Body Map), and place other material by kind:

- Code, docstrings, tests: internal call/guard order, locks, fields, edge cases, typed refusals.
- Git/README: release history; commits/PRs: decision codes, batch answers, reviewer-round labels.
- Issues: follow-ups; state a limitation once where it prevents wrong action (open ones may cite `(issue #NNN)`).
- Probe dates and measurements: only where "External facts: unknown is not no" puts them.

Track only continuing-purpose assets (product, contributors, verification, legal, public-claim evidence); keep plans, backlogs, review packets and run receipts in the external work area or durable task evidence, not git (presence tests confer no product role) and retire finished campaign tooling. Name new test modules by behaviour or owner, never round, batch, issue, PR or version (review-enforced; no auto-delete or filename matching). Never cross-import test modules; share fixtures in `tests/_*.py`, which select importers in the commit gate.

Each book pairs an entrypoint (H1, intro, flat ordered `## Chapters` list, Architecture's Reference Atlas; nothing beyond chapters/specs, since Architecture's intro and Atlas ride every request in the Body Map) with one chapter per subject under `docs/architecture/` or `docs/development/` (authored H1 intro required—H1 directly before a subsection fails; an Architecture chapter's intro is what the Map shows of that part: what it gives me, how I reach it, which limit changes a choice). `.gitattributes` pins `docs/**/*.md` to LF for stable digests/offsets. Readers use `load_governance_doc`, `context_layout.book_navigation`, `reference_books.book_path_role` and `tests/_governance_docs_shared.py`, never `read_text()` (entrypoint substring pins pass vacuously). Enforcement: `tests/test_reference_book_validation.py`; review judges entrypoint facts (`architecture_doc`); `tests/test_docs_sync.py::test_resident_docs_residue_only_shrinks` (Cyrillic included).

Size rule, by what a mind carries: what rides every request ends no larger than at the event base (`BIBLE.md`, `prompts/*.md` as one text, and the rendered Body Map: Architecture's intro, chapter intros and Atlas, and Development's chapter titles and paths), paying additions in the same text by cutting obsolete or duplicated prose, never losing a fact (DEVELOPMENT 03 "Paying down a size cap"); the lane renders the base's Map with the base revision's own code, so renderer text counts, and a base without a renderer had no Map and is not compared; a text there but unreadable is unmeasured, never absent or smaller. A book chapter is read on demand, so its limit is physical, one default `read_file` (2,000 lines, 80,000 characters): past it the chapter splits by subject with its reasons kept, and an unreadable chapter is unmeasured, never within the band. No other doc is under a size rule: DESIGN, PERSISTENCE, CLIENT_CONTRACT, CREATING_SKILLS and CHECKLISTS show their delta as a fact in the balance (`reference_books.book_balances`). The balance an edit shows gives each earlier side (HEAD, the contribution base) as one of three: a measured delta, absent there, or unknown and said so (unreadable there, or a revision that rendered no Map: `map_renderer_at`, the lane's own answer), an unknown side pricing nothing, per file or in total. With push/PR CI disabled, manual/tag runs lack a per-change base and skip comparison (ARCHITECTURE §8); local checks pass `OURO_SIZE_RATCHET_BASE_REF`, review checks contribution, owner `book-growth` exempts, and edit tools, `plan_task`, `preflight_review(deterministic_only=True)` and `codebase_health` report balance without blocking.

### Generality and emergence (P13)

Non-trivial changes patch cases, solve their class or build speculative frameworks: avoid the first (fossilizes) and third (invents cases). Shared structure needs a demonstrated invariant or stable boundary; imagined consumers are not a class. In doubt, generalize meaning and authority while keeping mechanism minimal and local until a second real consumer arrives. Reviewer findings are evidence, never policy. Review-only.

### Pricing and admission

Never add static model-price tables, prefix tariffs or numeric fallbacks—they go stale silently. Keep `cost=None` and `cost_final=false` until a live exact-route source answers; unknown price is neither free nor an admission veto, while known budget limits bind (`pricing.py`, `llm_pricing.py`, `tests/test_pricing.py`, `tests/test_budget_limits.py`; ARCHITECTURE §6 "Budget tracking").

### Anti-pattern: content-derived identity for host-minted records

Capture host-minted record identity at ingress and pass it by value (`origin_message_ref`), never re-querying text hashes, equality or prefixes, which post-ingress edits break (content hashes remain valid for integrity checks or content-addressing only). Consuming seams require the typed ref or closed absence enum, raising on omission (`bind_task_to_project(..., *, origin)`, `tests/test_projects_v6640.py`); use `semantic_dedup` over string equality for fuzzy entities, but distinguish shared work identity from provenance: moving one task moves no siblings from its message (ARCHITECTURE §6 "Project placement"; `tests/test_one_work_placement.py`).

Exception: ingress-less verification receipts reconcile by one typed `(kind, value)` key, never cross-kind, since per-component fallbacks are order-dependent, not an equivalence relation; keying fails safe (`ouroboros/_outcome_receipts.py`, `tests/test_v678_receipt_reconciliation.py`). Generalizing rules (review-only): report via the deciding projection; keep closed-kind properties in total lookups raising on gaps; drive comparison, hashing, counting and projection from one canonical derivation; version stored rendering changes and reason both ways, since unknown must not clear a red.

### Anti-pattern: an open default behind a closed exception list

Never rebuild producer classification in consumers with "on by default except these names" lists, which go stale silently. Derive presence from what the record holds (`web/modules/chat.js::blockVisible`), stamp special verbs from the owner table (`tool_capabilities.ROUTING_VERBS` → `routing_action`, `routing_tool_calls`, `typed_routing_action`) and render typed reasons verbatim (`project_dialogue.routing_refusal_cause`; ARCHITECTURE §3 "Chat and Projects"). Owner-side closed vocabularies (`tool_capabilities.OBSERVE_WORLD_MUTATION_TOOLS`) remain valid because new read tools default cleanly to Observe (ARCHITECTURE §6 "Background consciousness and Evolution").

### Task-authored messages are never owner text

Host code mints routing speaker identity by value (`control_routing._routing_issuer`); never re-derive it from chat-only contracts, empty client ids, chat ids or lanes, or expose it as a tool argument.

- Task speech uses `KIND_TASK_MESSAGE` (`independent_task`), never `KIND_OWNER_TEXT`, across `owner_mailbox.TASK_MESSAGE_PROVENANCES`, `deliver_task_message` and `loop_round_limits` and outside owner corpora (`owner_source_sha256`), blocking self-authored asks (`tests/test_task_authored_messages.py`; ARCHITECTURE §6 "Owner routing verbs").
- Host-held owner rows stay my human's across the tree by ingress possession, never task text's say-so: context/precedence for non-addressed targets (never acceptance criteria or reply debt), delivered verbatim to addressed live tasks at their next input boundary (`tests/test_helper_owner_words_flow.py`).
- Card questions, options and details belong to the asking task; wordless presses are acts, not owner words, while typed comments stay verbatim (`tests/test_press_is_an_act.py`). Accounts (notes, plans, briefs, labels) carry author and timestamp (or `"not recorded"`; `context_compaction.note_written`), never posing as a person's words; host frames state who/when/route without interpreting intent—what a person meant is theirs to settle. Rest review-only.

### The owner corpus archives inputs; the owner door's stamp is the only authority

Task-authored objectives copy only the retained owner corpus, never drafts; other runs log their first user turn for the corpus and Safety, labelled by host facts (`loop_messages._initialize_owner_directives`). `dialogue_provenance.run_origin` stamps `owner_ingress` once from door-stamped turns—never lanes, client ids or inherited stamps—and that stamp is reserved on `/api/tasks` and schedule templates; accepted work follows the contract and owner answers, never labels.

### Anti-pattern: a chat id tested for truth

Chat ids are typed values, not booleans: `HIDDEN_CHAT_ID` (`0`) is the hidden partition, `None` is absence, negative ids are synthetic A2A traffic, so `if chat_id:` misroutes hidden notices to Main. Use only `message_bus.notification_chat_route` (routing) and `message_bus.coerce_chat_identity` (row addresses), stamping `log_addressing.ingress_chat_id` once at admission (never inferring provenance from task type; ARCHITECTURE §5, §12). Enforcement: `tests/test_chat_id_truthiness_guard.py` (allowlist states each exception's reason).

### External facts: unknown is not no

Inputs, capacity, prices, IDs and availability change independently. Use exact-route metadata, refusals, dated wire probes or explicit owner settings; never infer from names, prefixes, families or other routes. Unknown leaves inputs, tools and explicit requests intact. Disclose wire changes and preserve saved choices. Repair the refused attempt; durable absence requires scoped evidence. Identify Ouroboros's own transport limits to the model at the enforcing lane.

Keep a wire exception's source, date if known, exact scope, evidence kind (live or docs-only), recheck and risks of stale support and stale absence at the code that enforces it. Read that record before editing. Distinguish requested, sent and provider-reported values. Provenance grants no runtime authority.

Remember a cache-hint refusal only after the repaired call succeeds, for that route profile until expiry (`request_wire_recovery.py`). Direct Z.ai sends requested `medium` as `high` (`reasoning_effort_clamped`); success tests `high`, not `medium`. Omitting the tier on GLM-5.3 can bill at max.

Guards: `tests/test_image_capability_contract.py` keeps images for unseen models; `tests/test_model_name_invariance.py` checks name-independent payloads absent evidence; `tests/test_tristate_truthiness_guard.py` bans implicit truth tests on tri-state facts; withhold input only on confirmed absence (`is False`).

Where to look under `ouroboros/`:

- Requests: [image contracts](../IMAGE_GENERATION.md#direct-api-contracts) (`tools/imagegen.py`, `gateways/image_api.py`), DeepSeek Search query limits (`gateways/search_api.py`), schema ceilings (`provider_models.py`). A shared Go session ID mixes conversations (`opencode_go.py`).
- Cost and turn state: cache boundaries (`llm_attempt.py`), shared Codex prefixes with per-conversation turn state (`llm_claudexor.py`), live tariffs for cost and budget accounting (`pricing.py`, `llm_pricing.py`).
- Effort and history: tier maps (`provider_models.py`), forced calls that fail with thinking (`llm_openai_compatible.py`), prefix changes that retire signed reasoning (`transcript_prefix.py`), sealed cross-provider replay (`reasoning_artifacts.py`).
- Routing and choices: filtering can narrow failover (`llm_routing.py`); stale Ultra support can hide or send a tier (`transport_custody.py`). ID aliases and migration of retired defaults must preserve saved choices (`provider_models.py`, `server_runtime.py`).

### Provider Independence

One configured provider suffices for loop, commit review, scope, safety and context/memory—no hidden OpenRouter or second provider (SSOT for ARCHITECTURE and CHECKLISTS item 11(h)). Tool schemas are provider contracts: validate the shipped registry against JSON Schema and its cross-provider subset, and run bounded Main canaries; bad arguments/schemas stay red (no prose parser, provider hop or unbounded retry); PR jobs stay secretless (ARCHITECTURE §8 "CI topology").

A provider change covers credentials, readiness, exact model-id migration, Main/Light/Fallback and review-pool defaults (keeping explicit choices), canonical tool/reasoning/image/cache intent in `llm.py` with leaf wire projection/recovery, nullable pricing, honest capability gaps, review/scope routing, sourced context windows and direct/single-provider tests using real redacted `tests/fixtures/llm_wire/` streams (handmade fixtures prove no dialect). Local-only installs keep local routes; unauthored slots are proposed blank. Window size and reading coverage never change review authority (BIBLE P3; DEVELOPMENT §5).

Keep current model ids and defaults in code/config, not here: active consumers read `provider_models.ACTIVE_MODEL_SETTING_KEYS`; `LEGACY_MODEL_SETTING_KEYS` is migration-only; `OUROBOROS_MODEL_HEAVY` + `USE_LOCAL_HEAVY` only seed a configured API actor while the canonical list is absent, never an active slot, readiness signal, test probe or fallback. `-pro` is an OpenRouter slug, not an OpenAI model id (direct OpenAI uses plain Sol and its connection's API; model prefixes never authorize admission). Non-OpenRouter Chat routes send `reasoning_effort` (DeepSeek/Z.ai by provider-id projection); direct Anthropic and OpenAI Responses are the deliberate non-reconstructed transcript exceptions, with no synthesized effort-to-`budget_tokens` policy (ARCHITECTURE §7 "LLM output token budgets"). Missing optional features degrade explicitly without crashing or rerouting.

Canonical history and tool schemas stay function-shaped, never a second stored transcript per dialect; static normalization precedes request-wire binding, and learned request-shape adaptation uses `ouroboros/request_wire_contract.py` (exact route, closed actions, shared TTL, no provider prose or route switches, explicit `none` never durable).
