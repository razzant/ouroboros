# Reference books: chapter migration transfer table

This is the operator record of the physical split of the two reference books
(`docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`) at base commit `5585133db86419c1a28673e498de4fb13c6b2d1e`.
It is a docs file and deliberately NOT a book member: it lives directly under
`docs/`, so `validate_reference_books` never sees it in either book's chapter
population.

The split happened in two phases, both recorded here. Phase one (the tables
under each book heading below) was a **verbatim move**: the moved bytes were
every byte of the old `##` section AFTER its heading line, including its
`###`/`####` sub-headings at the levels they already had; nothing was merged,
rewritten, summarized, reordered or deleted, and no section title was renamed
— the rename column is empty for every row. Phase two, the semantic
subtraction, changed chapter bodies on purpose and is recorded in the
"Semantic subtraction" section at the end: every merged duplicate, rewritten
explanation and removed piece of narrated history, with the chapter and
heading that keeps the fact.

## What changed, exactly

Per chapter file, the only new bytes are a two-line prologue:

1. `# <the old section title>` — the old `## N. Title` text at H1, numbering
   text kept so every cross-reference in the corpus still reads;
2. one authored introductory paragraph (2–4 sentences: what the chapter owns
   and why it exists), the only new prose in this migration.

Everything after that prologue is the old section body, byte for byte.

Per entrypoint, the body is replaced by an ordered `## Chapters` membership
list; the H1 line stays byte-identical in both books.

## Byte proof

`tests/test_reference_book_migration.py` reverses the prologue of every chapter
AS THE MIGRATION COMMIT WROTE IT (drop the H1 line, drop the one introductory
paragraph, re-prefix `## ` to the H1 text), concatenates the results in
membership order, and requires the SHA-256 below against `git show
<base>:<path>`. The proof is bound to history — the chapters at the commit that
added this file against the monolith at the base commit — so the semantic
subtraction below could not and did not turn it red; it stays as the record
that phase one moved bytes, not meaning. The current tree is proven by the
structural and composition tests instead: `test_every_base_section_is_still_exactly_one_chapter`,
`tests/test_reference_book_validation.py` (the validator over the tracked
tree) and `tests/test_reference_book_pack_assembly.py` (the packs assembled
from the real chapters).

| Book | Old file | Old bytes | Old SHA-256 | Preamble bytes (replaced) | Moved-body bytes | Moved-body SHA-256 |
|---|---|---|---|---|---|---|
| architecture | `docs/ARCHITECTURE.md` | 724691 | `5db278f8ef5060c4aff5ee1e8743c279661ddd975a311a9bafb5858f32b080de` | 610 | 724081 | `f1f054c700a15533687e0cf81cf19ac53ffcf022eb179f84c0cbccacbb1d305e` |
| development | `docs/DEVELOPMENT.md` | 275551 | `50eb460602f1501915195e3ad1918366312e6292b0dccec6f7ef53f5a5302f3b` | 60 | 275491 | `bffc00227bc5e91f054b38eaed63acd256c2ddf111e231754bfbe92c7dd122e3` |

## `docs/ARCHITECTURE.md`

Entrypoint preamble before: `# Ouroboros v7.0.0 — Architecture & Reference` (byte-identical, the release version carrier `release_sync.VERSION_CARRIER_SPANS` writes), then two paragraphs (`This file is NOT a changelog…` and `This is the present-tense operational map…`) and a `---` rule.

Entrypoint preamble after: the same H1, then ONE merged paragraph carrying both original paragraphs' claims (present-tense map in three layers, not a changelog, WHY stays in the book, rationale self-contained), then `## Chapters`. The `---` rule is dropped.

| Old `##` section | Lines at base | Section bytes | Destination chapter | Disposition | Title rename |
|---|---|---|---|---|---|
| `1. High-Level Architecture` | 9–659 | 188782 | `docs/architecture/01-high-level-architecture.md` | verbatim move | — |
| `2. Startup / Onboarding Flow` | 660–695 | 14555 | `docs/architecture/02-startup-onboarding-flow.md` | verbatim move | — |
| `3. Web UI Pages & Buttons` | 696–868 | 88631 | `docs/architecture/03-web-ui-pages-and-buttons.md` | verbatim move | — |
| `4. Server API Endpoints` | 869–1031 | 23636 | `docs/architecture/04-server-api-endpoints.md` | verbatim move | — |
| `5. Supervisor Loop` | 1032–1082 | 32383 | `docs/architecture/05-supervisor-loop.md` | verbatim move | — |
| `6. Agent Core` | 1083–1871 | 258915 | `docs/architecture/06-agent-core.md` | verbatim move | — |
| `7. Configuration (ouroboros/config.py)` | 1872–2084 | 33538 | `docs/architecture/07-configuration.md` | verbatim move | — |
| `8. Git Branching, CI, and Build` | 2085–2140 | 17850 | `docs/architecture/08-git-branching-ci-and-build.md` | verbatim move | — |
| `9. Shutdown & Process Cleanup` | 2141–2156 | 12956 | `docs/architecture/09-shutdown-and-process-cleanup.md` | verbatim move | — |
| `10. Key Invariants` | 2157–2235 | 15834 | `docs/architecture/10-key-invariants.md` | verbatim move | — |
| `11. Frozen Contracts v1 (`ouroboros/contracts/`)` | 2236–2326 | 19130 | `docs/architecture/11-frozen-contracts-v1.md` | verbatim move | — |
| `12. Host Service, Companion Processes, and Chat IDs` | 2327–2371 | 10594 | `docs/architecture/12-host-service-companions-and-chat-ids.md` | verbatim move | — |
| `13. External Skills Layer` | 2372–2386 | 7277 | `docs/architecture/13-external-skills-layer.md` | verbatim move | — |

## `docs/DEVELOPMENT.md`

Entrypoint preamble before: `# DEVELOPMENT.md — Development Principles & Module Guide` (byte-identical), then `## Role and authority` directly, with no introductory paragraph of its own.

Entrypoint preamble after: the same H1, then ONE newly authored orientation paragraph (what the handbook is, how the chapters are ordered, read the chapter for the class of change in hand), then `## Chapters`. No moved prose.

| Old `##` section | Lines at base | Section bytes | Destination chapter | Disposition | Title rename |
|---|---|---|---|---|---|
| `Role and authority` | 3–32 | 1707 | `docs/development/01-role-and-authority.md` | verbatim move | — |
| `Naming and boundaries` | 33–526 | 35291 | `docs/development/02-naming-and-boundaries.md` | verbatim move | — |
| `Module Size & Complexity` | 527–849 | 21621 | `docs/development/03-module-size-and-complexity.md` | verbatim move | — |
| `Core Governance Artifacts` | 850–1121 | 20584 | `docs/development/04-core-governance-artifacts.md` | verbatim move | — |
| `Review & Commit Protocol` | 1122–1351 | 15585 | `docs/development/05-review-and-commit-protocol.md` | verbatim move | — |
| `Rules by change class` | 1352–2963 | 118624 | `docs/development/06-rules-by-change-class.md` | verbatim move | — |
| `Managed Update Rule` | 2964–3022 | 3732 | `docs/development/07-managed-update-rule.md` | verbatim move | — |
| `Mutation Attribution Rule` | 3023–3059 | 2289 | `docs/development/08-mutation-attribution-rule.md` | verbatim move | — |
| `Process Custody Rule` | 3060–3203 | 10776 | `docs/development/09-process-custody-rule.md` | verbatim move | — |
| `Platform Abstraction Rule` | 3204–3247 | 2514 | `docs/development/10-platform-abstraction-rule.md` | verbatim move | — |
| `Design System` | 3248–3557 | 22543 | `docs/development/11-design-system.md` | verbatim move | — |
| `MCP Client Integration` | 3558–3605 | 3454 | `docs/development/12-mcp-client-integration.md` | verbatim move | — |
| `Gateway Boundary Pattern` | 3606–3635 | 1986 | `docs/development/13-gateway-boundary-pattern.md` | verbatim move | — |
| `Build & CI` | 3636–3886 | 14785 | `docs/development/14-build-and-ci.md` | verbatim move | — |

## Chapter granularity

One chapter per old `##` section, with no merges. Two adjacent pairs were
under the ~60-line merge threshold on both sides — Architecture §12/§13 and
Development "MCP Client Integration"/"Gateway Boundary Pattern" — but neither
pair shares a subject (a host callback boundary is not the external skills
plane; an outbound MCP client is not the inbound browser boundary), so the
default 1:1 mapping was kept. It also keeps every existing cross-reference of
the form `ARCHITECTURE "8. Git Branching, CI, and Build"` resolving to exactly
one chapter.

## Semantic subtraction

Phase two walked both books paragraph by paragraph and gave every fact stated
in more than one place ONE owner: structure, mechanism and WHY to an
Architecture chapter; process, gate and how-to-change to a Development
chapter. The other statement became a pointer to the owner's chapter and
heading, or nothing when the surrounding paragraph already implies it. Every
imperative, every enforcing test name, every owner citation and every disclosed
residual survived; where the Development copy carried a fact the Architecture
owner lacked, the owner received it (listed under "Facts whose owner moved").
Base commit of the subtraction: `4128b2048` (the integration branch after phase one).

### Statistics

| Book | Paragraphs merged (duplicate) | Paragraphs rewritten | Paragraphs removed (obsolete history) | Bytes before | Bytes after | Delta |
|---|---:|---:|---:|---:|---:|---:|
| architecture | 7 | 9 | 15 | 731409 | 730965 | -444 |
| development | 93 | 6 | 1 | 284650 | 252872 | -31778 |

Paragraph counts are disposition rows: one row per paragraph, bullet or table
row whose text changed. Byte counts are the sum of the book's chapter files
(entrypoints unchanged).

### Facts whose owner moved between the books

Development → Architecture (the Development copy was the only complete statement; the owner now carries it):

- the shared read/search byte masker — `docs/architecture/06-agent-core.md` § "Tool capability and execution"
- the path-selected attachment-ingest checks (exact credential leaves, credential/control directory components; enumerated owner stores as mutation-fence authority) — the same section
- the size ratchet's merge-aware manifest resolution and own-tree bootstrap — `docs/architecture/06-agent-core.md` § "Review stack"
- the `reasoning_effort_clamped` usage disclosure of the DeepSeek tier projection — `docs/architecture/07-configuration.md` (DeepSeek provider specifics)
- the date of the transcript-cache byte-prefix measurement (2026-09-14) — `docs/architecture/01-high-level-architecture.md`, the `transcript_prefix.py` row
- why a resolvable CI base without a manifest fails closed (copied manifests would launder debt) — `docs/architecture/08-git-branching-ci-and-build.md` § "CI topology"
- the README history row and the named direct-download links as release carriers — `docs/architecture/10-key-invariants.md`, invariant 2

Architecture → Development (the Architecture copy is now a pointer):

- what belongs in a prompt versus a tool schema (a prompt sentence restating a schema is a second copy that drifts) — `docs/development/02-naming-and-boundaries.md` § "LLM-first affordances"
- what each review-enforcement mode permits after a technical review failure — `docs/development/05-review-and-commit-protocol.md`

### Byte proof after phase two

`test_the_migration_commit_reconstructs_the_base_monolith_byte_for_byte` is kept:
it proves the phase-one commit against the base monolith and never reads the
working tree, so it stays green after this subtraction and records what phase
one was. No byte-equality promise is made for the current tree.

### Dispositions — `docs/architecture/`

| Chapter | Heading | Disposition | Retained owner | What changed | Chars before → after |
|---|---|---|---|---|---:|
| `docs/architecture/06-agent-core.md` | Tool capability and execution | merged duplicate | docs/architecture/06-agent-core.md § "Tool capability and execution" (first statement, same paragraph) | the three discovery outcomes were stated twice in one paragraph | 463 → 55 |
| `docs/architecture/06-agent-core.md` | Tool capability and execution | rewritten explanation | docs/architecture/06-agent-core.md § "Tool capability and execution" | owner now states the shared read/search masker fact that Development carried | 197 → 308 |
| `docs/architecture/06-agent-core.md` | Tool capability and execution | rewritten explanation | docs/architecture/06-agent-core.md § "Tool capability and execution" | owner now states the attachment-ingest leaf/directory checks that Development carried | 175 → 354 |
| `docs/architecture/06-agent-core.md` | Context fitting, retry, and compaction | merged duplicate | docs/development/02-naming-and-boundaries.md § "LLM-first affordances" | schema-is-the-SSOT sentence; Development owns the prompt-edit rule | 180 → 127 |
| `docs/architecture/06-agent-core.md` | Task lifecycle | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | the nomination facts (deferred_to_host_acceptance, authoritative=false) now stated once | 214 → 306 |
| `docs/architecture/06-agent-core.md` | Task lifecycle | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | second statement of the nomination | 256 → 57 |
| `docs/architecture/06-agent-core.md` | Review delivery | merged duplicate | docs/architecture/06-agent-core.md § "Caller-owned subscription model calls" | "the raw-model adapter is Codex" stated once | 120 → 81 |
| `docs/architecture/06-agent-core.md` | Review delivery | merged duplicate | docs/development/05-review-and-commit-protocol.md | advisory-continue / blocking-refuse / never-PASS rule; Development owns the gate rule | 184 → 149 |
| `docs/architecture/06-agent-core.md` | Review stack | rewritten explanation | docs/architecture/06-agent-core.md § "Review stack" | owner now states the merge-aware resolution and own-tree bootstrap that Development carried | 271 → 430 |
| `docs/architecture/07-configuration.md` | Default settings (OUROBOROS_REVIEWER_SLOTS) | rewritten explanation | docs/architecture/07-configuration.md § "Default settings" | the row said an empty value reads the retired comma keys; §11.4 and reviewer_slot_config.py say those keys are stripped at load and the shipped default panel serves | 80 → 134 |
| `docs/architecture/07-configuration.md` | Default settings (DeepSeek provider specifics) | rewritten explanation | docs/architecture/07-configuration.md (DeepSeek provider specifics) | owner now names the usage disclosure field Development carried | 83 → 113 |
| `docs/architecture/07-configuration.md` | Default settings (OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC) | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | no-prediction / R23 clamp restated in the settings row | 343 → 196 |
| `docs/architecture/07-configuration.md` | Default settings (OUROBOROS_REVIEW_MAX_CYCLES) | rewritten explanation | docs/architecture/06-agent-core.md § "Review stack" | §10 invariant 17 itself points to §6 Review stack for the four meanings | 56 → 68 |
| `docs/architecture/08-git-branching-ci-and-build.md` | CI topology | rewritten explanation | docs/architecture/08-git-branching-ci-and-build.md § "CI topology" | owner now states the debt-laundering WHY Development carried | 56 → 114 |
| `docs/architecture/10-key-invariants.md` | Key Invariants (2) | rewritten explanation | docs/architecture/10-key-invariants.md invariant 2 | owner now lists the history-row and download-link carriers Development carried | 99 → 195 |
| `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (transcript_prefix.py row) | rewritten explanation | docs/architecture/01-high-level-architecture.md (transcript_prefix.py row) | owner now carries the measurement date Development carried | 44 → 65 |
| `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (review_execution.py row) | obsolete history | — | 'out of this change s scope' narrates one past commit; the residual and its issue stay | 128 → 100 |
| `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (plan_packet.py row) | obsolete history | — | design-wave codename with no current effect | 59 → 56 |
| `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (deep_self_review.py row) | obsolete history | — | narrated the retired error it replaced; the current rung stays | 187 → 137 |
| `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (reviewer_slot_config.py row) | obsolete history | — | "is gone" narration; owner citation kept | 184 → 130 |
| `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (review_execution.py row) | obsolete history | — | "is gone" narration; owner citation kept | 110 → 74 |
| `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (review_substrate.py row) | obsolete history | — | "are gone" narration; owner citation kept | 120 → 48 |
| `docs/architecture/03-web-ui-pages-and-buttons.md` | Settings and onboarding | obsolete history | docs/architecture/07-configuration.md (OUROBOROS_MODEL_DEEP_SELF_REVIEW row states where the row lives now) | narrated a removed UI field; the settings row states the current home | 180 → 138 |
| `docs/architecture/03-web-ui-pages-and-buttons.md` | Chat and Projects | obsolete history | — | "former … rejection" narration | 112 → 102 |
| `docs/architecture/03-web-ui-pages-and-buttons.md` | Chat and Projects | obsolete history | — | "former"/"remain" narration; the boundary itself stays | 65 → 55 |
| `docs/architecture/06-agent-core.md` | Delegated subagents (Claudexor transport + the nanny) | obsolete history | — | plan-item codename with no current effect | 87 → 69 |
| `docs/architecture/06-agent-core.md` | Caller-owned subscription model calls | obsolete history | — | "initial" narrates the adoption order | 63 → 55 |
| `docs/architecture/06-agent-core.md` | Review delivery | obsolete history | docs/architecture/11-frozen-contracts-v1.md § "11.4 Recent ABI Retirements" | retirement narrated beside the rule; §11.4 is the retirement ledger | 83 → 68 |
| `docs/architecture/07-configuration.md` | Default settings (OUROBOROS_REVIEWER_SLOTS) | obsolete history | — | plan-step codename | 74 → 68 |
| `docs/architecture/07-configuration.md` | Default settings (OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS) | obsolete history | docs/architecture/11-frozen-contracts-v1.md § "11.4 Recent ABI Retirements" | retirement narrated beside the rule | 81 → 66 |
| `docs/architecture/12-host-service-companions-and-chat-ids.md` | Host Service, Companion Processes, and Chat IDs | obsolete history | — | "old … is removed" narration | 104 → 94 |

### Dispositions — `docs/development/`

| Chapter | Heading | Disposition | Retained owner | What changed | Chars before → after |
|---|---|---|---|---|---:|
| `docs/development/02-naming-and-boundaries.md` | CLI and headless work | merged duplicate | docs/architecture/06-agent-core.md § "Tool capability and execution" | credential fence locations, byte masker, PEM content evidence, restricted-read masking, unlisted-store rule | 2091 → 1292 |
| `docs/development/02-naming-and-boundaries.md` | Documentation contract | merged duplicate | docs/development/01-role-and-authority.md | handbook definition (rules by change class naming the enforcing surface) | 277 → 184 |
| `docs/development/02-naming-and-boundaries.md` | Task-authored messages are never owner text | merged duplicate | docs/architecture/06-agent-core.md § "Durable memory and project focus" | routing issuer definition (owner turn vs task speaking for itself) | 352 → 282 |
| `docs/development/02-naming-and-boundaries.md` | Task-authored messages are never owner text | merged duplicate | docs/architecture/06-agent-core.md § "Durable memory and project focus" | written/refused receipt, task_message_routed row, no chat told | 286 → 73 |
| `docs/development/02-naming-and-boundaries.md` | Anti-pattern: a chat id tested for truth | merged duplicate | docs/architecture/12-host-service-companions-and-chat-ids.md | HIDDEN_CHAT_ID partition definition | 284 → 230 |
| `docs/development/02-naming-and-boundaries.md` | Anti-pattern: a chat id tested for truth | merged duplicate | docs/architecture/05-supervisor-loop.md | browser-source Main addressing vs hidden API default | 140 → 228 |
| `docs/development/02-naming-and-boundaries.md` | Provider Independence | merged duplicate | docs/architecture/07-configuration.md (DeepSeek provider specifics) | DeepSeek tier projection map, forced tool choice thinking rule, live probe date; `reasoning_effort_clamped` name moves to the owner | 609 → 330 |
| `docs/development/02-naming-and-boundaries.md` | Provider Independence | merged duplicate | docs/architecture/06-agent-core.md § "Context fitting, retry, and compaction" | direct-OpenAI dialect ladder | 356 → 371 |
| `docs/development/02-naming-and-boundaries.md` | Provider Independence | merged duplicate | docs/architecture/06-agent-core.md § "Context fitting, retry, and compaction" | Anthropic native custody receipt | 357 → 260 |
| `docs/development/02-naming-and-boundaries.md` | Anti-pattern: content-derived identity for host-minted records | merged duplicate | docs/architecture/06-agent-core.md § "Durable memory and project focus" | retry binding transaction (bind_retry_to_origin_project) and immutability WHY | 787 → 296 |
| `docs/development/03-module-size-and-complexity.md` | Module Size & Complexity | merged duplicate | docs/architecture/06-agent-core.md § "Review stack" | size-ratchet CI lane mechanics and no-history-replay WHY; the own-tree bootstrap fact moves to the owner | 860 → 655 |
| `docs/development/03-module-size-and-complexity.md` | Invariant: Projection over replay (hot readers of growing stores) | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Chat and Projects" | task_list_scan memo rules and SSE v2 cursor/rotation/handle discipline | 1486 → 426 |
| `docs/development/03-module-size-and-complexity.md` | Review presentation adapters | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Chat and Projects" | compact review rows carry no dollars; Skill attempt money only in lazy detail via physical_attempt_v1 | 308 → 171 |
| `docs/development/03-module-size-and-complexity.md` | Invariant: Continuation authority and bounded Main projection | merged duplicate | docs/architecture/01-high-level-architecture.md § "CLI / Headless Boundary" | predecessor_task_id contract, snapshot/restore retention, defensive projection outcomes | 1023 → 889 |
| `docs/development/03-module-size-and-complexity.md` | Invariant: UI resources carry a disposer | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Skills and Widgets" | ordered dispose-with-acknowledgement sequence | 422 → 258 |
| `docs/development/03-module-size-and-complexity.md` | Invariant: Embedded surfaces declare geometry and refresh semantics | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Skills and Widgets" | widget fault channel | 286 → 140 |
| `docs/development/04-core-governance-artifacts.md` | Invariant: Full availability in reasoning flows | merged duplicate | docs/architecture/06-agent-core.md § "Plan construction and review" | constitutional classification rule (affected_paths vs affected_resources vs evidence) | 412 → 406 |
| `docs/development/04-core-governance-artifacts.md` | Invariant: Full availability in reasoning flows (context-delivery registry) | merged duplicate | docs/architecture/06-agent-core.md § "Plan construction and review" | registry row restated the classification; "(W3)" wave codename dropped | 503 → 361 |
| `docs/development/04-core-governance-artifacts.md` | Invariant: Full availability in reasoning flows (context-delivery registry) | merged duplicate | docs/architecture/06-agent-core.md § "Deep self-review" | deep self-review delivery mechanics (probe/rebuild, receipt extents, coverage states, memory header) | 1027 → 704 |
| `docs/development/04-core-governance-artifacts.md` | Invariant: Full availability in reasoning flows | merged duplicate | docs/architecture/06-agent-core.md § "Plan construction and review" | two planning roots and named-omission rule | 724 → 589 |
| `docs/development/04-core-governance-artifacts.md` | Invariant: Full availability in reasoning flows | merged duplicate | docs/architecture/06-agent-core.md § "Plan construction and review" | SPEC field list, verdict/finding vocabulary, note-only and blocking-finding closure rules, no competing-plan quota | 2513 → 1313 |
| `docs/development/04-core-governance-artifacts.md` | Invariant: Full availability in reasoning flows | merged duplicate | docs/architecture/06-agent-core.md § "Context fitting, retry, and compaction" | Max/Low/Nano book projections and context_fit rendering | 1289 → 792 |
| `docs/development/04-core-governance-artifacts.md` | Invariant: Compaction must earn its rewrite | merged duplicate | docs/architecture/06-agent-core.md § "Context fitting, retry, and compaction" | compaction selection rules and no-reclaim rule | 527 → 358 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Review delivery" | AdvisoryRunRecord.execution binding, audited-skip custody, late results, native preflight custody | 1498 → 821 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Git and commit review" | staged fingerprint composition | 174 → 85 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Git and commit review" | post-commit re-read of the binding | 117 → 72 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Git and commit review" | managed-update resolution subject binding | 158 → 166 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Review stack" | scope assembler degradation ladder summary | 322 → 229 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Review delivery" | native episode bounds (transcript bound measurement, window clamp, ledger, typed exhaustion) | 794 → 207 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Review delivery" | advisory row enum parsing, final/telemetry.yaml identity, no extra engine panel | 890 → 371 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Review stack" | per-gate meanings of the shared review-cycle cap | 141 → 196 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | acceptance paid-stamp points, R52/R55/R23 pacing, deadline-cut residual, issue #588 residual | 1872 → 980 |
| `docs/development/05-review-and-commit-protocol.md` | Review & Commit Protocol | merged duplicate | docs/architecture/06-agent-core.md § "Review stack" | the $0 exit shape and its four instances | 553 → 167 |
| `docs/development/05-review-and-commit-protocol.md` | Release sync | merged duplicate | docs/architecture/10-key-invariants.md (invariant 2) and docs/architecture/08-git-branching-ci-and-build.md § "Build scripts" | version-carrier list (history row + download links move to invariant 2), exact-tag download links, main promotion | 1248 → 660 |
| `docs/development/06-rules-by-change-class.md` | Live subagents | merged duplicate | docs/architecture/06-agent-core.md § "Delegated subagents (Claudexor transport + the nanny)" | WHY a false "spent nothing" terminal is the one forbidden direction | 190 → 172 |
| `docs/development/06-rules-by-change-class.md` | Live subagents | merged duplicate | docs/architecture/01-high-level-architecture.md (delegate_supervision.py row) | read-only poll anchor semantics, torn-tail quarantine, lock recovery mechanism | 1144 → 674 |
| `docs/development/06-rules-by-change-class.md` | Live subagents | merged duplicate | docs/architecture/06-agent-core.md § "Delegated subagents (Claudexor transport + the nanny)" | orphan apply containment relaxation | 522 → 348 |
| `docs/development/06-rules-by-change-class.md` | Cancellation and effective status | merged duplicate | docs/architecture/05-supervisor-loop.md (cancellation skeleton) and docs/architecture/10-key-invariants.md invariant 14 | fail-closed intent write, allow_settled_target, widen-only scope | 549 → 390 |
| `docs/development/06-rules-by-change-class.md` | Cancellation and effective status | merged duplicate | docs/architecture/05-supervisor-loop.md (cancellation skeleton) | completion-wins and reaper-is-not-a-cancel-ingress mechanism | 310 → 185 |
| `docs/development/06-rules-by-change-class.md` | Cancellation and effective status | merged duplicate | docs/architecture/10-key-invariants.md invariant 15 and docs/architecture/05-supervisor-loop.md | strict registry reads and durable task_done validation | 366 → 322 |
| `docs/development/06-rules-by-change-class.md` | Cancellation and effective status | merged duplicate | docs/architecture/05-supervisor-loop.md (stop policy / hurry) and docs/architecture/01-high-level-architecture.md (owner_hurry.py, task_hurry.py rows) | stop_policy axis semantics, hurry projection mechanics, non-chat event family | 1164 → 878 |
| `docs/development/06-rules-by-change-class.md` | Onboarding and Settings surfaces | merged duplicate | docs/architecture/02-startup-onboarding-flow.md | wizard step list; Accounts shared / Models and Agents edit roles | 325 → 324 |
| `docs/development/06-rules-by-change-class.md` | Onboarding and Settings surfaces | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Navigation and shared UI contracts" | model chooser native-select / arbitrary-id behaviour | 330 → 303 |
| `docs/development/06-rules-by-change-class.md` | Onboarding and Settings surfaces | merged duplicate | docs/architecture/06-agent-core.md § "Caller-owned subscription model calls" | Auto-lane account preference and suppression | 625 → 465 |
| `docs/development/06-rules-by-change-class.md` | Onboarding and Settings surfaces | merged duplicate | docs/architecture/02-startup-onboarding-flow.md | completion transaction contents, GET-never-persists WHY, post-disk failure reporting | 703 → 497 |
| `docs/development/06-rules-by-change-class.md` | Onboarding and Settings surfaces | merged duplicate | docs/architecture/02-startup-onboarding-flow.md | served wizard page as ES module | 168 → 113 |
| `docs/development/06-rules-by-change-class.md` | Onboarding and Settings surfaces | merged duplicate | docs/architecture/02-startup-onboarding-flow.md | three install-time proofs and their WHY | 554 → 425 |
| `docs/development/06-rules-by-change-class.md` | Onboarding and Settings surfaces | merged duplicate | docs/architecture/01-high-level-architecture.md § "Gateway Boundary v1" | settings lock precondition, CommitBoundary, saved on both sides | 541 → 283 |
| `docs/development/06-rules-by-change-class.md` | Onboarding and Settings surfaces | merged duplicate | docs/architecture/02-startup-onboarding-flow.md | preset compiler roster rules (review-<harness> rows, validate-only roster) | 533 → 334 |
| `docs/development/06-rules-by-change-class.md` | Transport and late-result custody | merged duplicate | docs/architecture/06-agent-core.md § "Review delivery" | stream assembler completeness/form doctrine and SSE error-frame classification | 1504 → 833 |
| `docs/development/06-rules-by-change-class.md` | Transport and late-result custody | merged duplicate | docs/architecture/06-agent-core.md § "Review delivery" | late-completion receipt binding fields | 632 → 497 |
| `docs/development/06-rules-by-change-class.md` | Transport and late-result custody | merged duplicate | docs/architecture/06-agent-core.md § "Review delivery" | subscription-catalog reachability proof, HEAD connection allowance | 826 → 444 |
| `docs/development/06-rules-by-change-class.md` | Transport and late-result custody | merged duplicate | docs/architecture/06-agent-core.md § "Delegated subagents (Claudexor transport + the nanny)" | observation beat, per-class quiet reasons, once-per-episode disclosure | 680 → 423 |
| `docs/development/06-rules-by-change-class.md` | LLM call rules | merged duplicate | docs/architecture/06-agent-core.md § "Budget tracking" | attempt lifecycle, release-only-on-typed-pre-dispatch-failure, projections never a second authority | 551 → 337 |
| `docs/development/06-rules-by-change-class.md` | LLM call rules | merged duplicate | docs/architecture/06-agent-core.md § "Budget tracking" | ledger lock discipline and failed-settlement bound | 235 → 201 |
| `docs/development/06-rules-by-change-class.md` | LLM call rules | merged duplicate | docs/architecture/06-agent-core.md § "Budget tracking" | tree-spend pacing mechanics, wallet read, post-task frozen snapshot, acceptance-panel projection in synthesis prompts | 1761 → 860 |
| `docs/development/06-rules-by-change-class.md` | LLM call rules | merged duplicate | docs/architecture/06-agent-core.md § "Delegated subagents (Claudexor transport + the nanny)" | delegated cash cases and input_token_usage validation | 1261 → 773 |
| `docs/development/06-rules-by-change-class.md` | LLM call rules (cache-friendliness) | merged duplicate | docs/architecture/01-high-level-architecture.md (transcript_prefix.py row) | append-only transcript mechanism, prompt_prefix_break kinds, byte-prefix cache measurement (date moves to the owner) | 1189 → 560 |
| `docs/development/06-rules-by-change-class.md` | LLM call rules | merged duplicate | docs/architecture/06-agent-core.md § "Context fitting, retry, and compaction" | transport-death repeat rail (who, how often, ledger rows, round-record terminal) and upstream-observed continuation | 2808 → 1633 |
| `docs/development/06-rules-by-change-class.md` | Timeout & Wait Control | merged duplicate | docs/architecture/05-supervisor-loop.md (owner-wait paragraphs) | owner-wait capacity transfer, cold preparation (CostCeiling/ContextFit rebinding), resume locator consumption, direct-actor wait | 2410 → 1329 |
| `docs/development/06-rules-by-change-class.md` | Timeout & Wait Control | merged duplicate | docs/architecture/05-supervisor-loop.md (worker readiness paragraph) | readiness/exhaustion lifecycle, temporary slots, watcher release | 1186 → 610 |
| `docs/development/06-rules-by-change-class.md` | Timeout & Wait Control | merged duplicate | docs/architecture/06-agent-core.md § "Review stack" | $0 exit shape restated for acceptance | 468 → 374 |
| `docs/development/06-rules-by-change-class.md` | Timeout & Wait Control | merged duplicate | docs/architecture/06-agent-core.md § "Review stack" and § "Context fitting, retry, and compaction" | custody precedence, provider_outcome_unknown, interactive repeat rail restatement | 1369 → 634 |
| `docs/development/06-rules-by-change-class.md` | Timeout & Wait Control | merged duplicate | docs/architecture/06-agent-core.md § "Review stack" and § "Review delivery" | retry-key identity, Skill Review wave reservation and review_resume_of, locked paid=True write | 1173 → 393 |
| `docs/development/06-rules-by-change-class.md` | Timeout & Wait Control | merged duplicate | docs/architecture/06-agent-core.md § "Review stack" | process-local custody tombstone, tokenless waiter settlement, pid-death owner-loss | 506 → 121 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | deferred_to_host_acceptance / authoritative=false nomination, host advance after the tool-result block, early settlement does not seal ingress | 528 → 310 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/06-agent-core.md § "Post-task reflection" | sealed final package contents and what tool success does not prove | 698 → 434 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/10-key-invariants.md (post-map paragraphs) and docs/architecture/06-agent-core.md § "Post-task reflection" | canonical-first persistence, native reader selector, CURRENT-basis publication, relocation rules | 2318 → 1465 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/05-supervisor-loop.md (pooled completion paragraph) | _files_prepared_attempt, terminal_task_files_ready, split-drive adoption | 783 → 475 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/05-supervisor-loop.md (terminal-file recovery paragraphs) | recovery ownership split, infra_failed fault policy, deferred replay | 1068 → 703 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/10-key-invariants.md (post-map paragraphs) | same-store copy rules and alias resolution | 459 → 372 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/05-supervisor-loop.md (pooled completion and startup recovery paragraphs) | mailbox cleanup predicate, startup source recovery and prune deferral | 545 → 305 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | acceptance delivery work orders, money rule, R52/R55/R23 pacing, deadline-cut residual (third copy) | 2277 → 777 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | the three acceptance decision states | 174 → 121 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | forced-rail terminalization and never-overwritten pair | 364 → 207 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/development/11-design-system.md (task outcome truth) and docs/architecture/03-web-ui-pages-and-buttons.md § "Chat and Projects" | shared outcome phase across Chat/Logs/Project rows/Telegram | 329 → 251 |
| `docs/development/06-rules-by-change-class.md` | Loop / State-Machine Changes | merged duplicate | docs/architecture/06-agent-core.md § "Task lifecycle" | dialogue_status vote reduction | 369 → 260 |
| `docs/development/07-managed-update-rule.md` | Managed Update Rule | merged duplicate | docs/architecture/02-startup-onboarding-flow.md | fresh-rescue WHY and replayed-rollback pointer semantics | 438 → 323 |
| `docs/development/09-process-custody-rule.md` | Process Custody Rule | merged duplicate | docs/architecture/01-high-level-architecture.md § "Runtime topology" | ledger path/role, strict fingerprint, skill-companion daemon-scope exception | 861 → 381 |
| `docs/development/09-process-custody-rule.md` | Process Custody Rule | merged duplicate | docs/architecture/09-shutdown-and-process-cleanup.md | daemon stop protocol (CLI stop receipt, forced fallback custody, HTTP refusal semantics, Windows empty-birth rows) and latch classification | 2938 → 1850 |
| `docs/development/09-process-custody-rule.md` | Process Custody Rule | merged duplicate | docs/architecture/09-shutdown-and-process-cleanup.md | latch release list | 447 → 272 |
| `docs/development/11-design-system.md` | Design System | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Chat and Projects" | taskReasonDetail precedence order | 246 → 155 |
| `docs/development/11-design-system.md` | Design System | rewritten explanation | docs/architecture/03-web-ui-pages-and-buttons.md § "Chat and Projects" | pointer re-pointed from the monolith to the chapter section | 122 → 139 |
| `docs/development/11-design-system.md` | Design System | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Chat and Projects" | 48 CSS-pixel live-edge zone | 119 → 189 |
| `docs/development/11-design-system.md` | Browser dialogs | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Navigation and shared UI contracts" | native-dialog WHY and openConfirmDialog mode contract | 546 → 226 |
| `docs/development/11-design-system.md` | Browser dialogs | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Navigation and shared UI contracts" | bindDialogFocus/bindMenu/bindPopoverPosition structure | 686 → 403 |
| `docs/development/11-design-system.md` | Declarative widgets | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Skills and Widgets" and docs/architecture/01-high-level-architecture.md (module tree) | module-frame CSP, Widgets module inventory, list-signature reconciliation | 988 → 415 |
| `docs/development/11-design-system.md` | Optional author controls | merged duplicate | docs/architecture/03-web-ui-pages-and-buttons.md § "Skills and Widgets" | author-kit recipe mechanics | 563 → 317 |
| `docs/development/12-mcp-client-integration.md` | MCP Client Integration | merged duplicate | docs/architecture/06-agent-core.md § "MCP and browser-facing external tools" | stdio contract, reference precedence, unknown-field warning, MCP_CONFIG_ERROR, auth_configured, secret masking | 1688 → 899 |
| `docs/development/13-gateway-boundary-pattern.md` | Gateway Boundary Pattern | merged duplicate | docs/architecture/12-host-service-companions-and-chat-ids.md | host attachment copy custody and operation correlation | 1004 → 658 |
| `docs/development/14-build-and-ci.md` | Python dependency locks | merged duplicate | docs/architecture/08-git-branching-ci-and-build.md § "Build scripts" | dependency authority structure | 425 → 240 |
| `docs/development/14-build-and-ci.md` | Pytest marker lanes | merged duplicate | docs/architecture/08-git-branching-ci-and-build.md § "CI topology" | secret-step ordering WHY | 347 → 195 |
| `docs/development/14-build-and-ci.md` | Pytest marker lanes | merged duplicate | docs/architecture/08-git-branching-ci-and-build.md § "CI topology" | base fallback rule; the debt-laundering WHY moves to the owner | 388 → 307 |
| `docs/development/14-build-and-ci.md` | The commit gate mirrors the CI split | merged duplicate | docs/architecture/06-agent-core.md § "Git and commit review" | preflight test proof binding and reuse rules | 1192 → 619 |
| `docs/development/02-naming-and-boundaries.md` | Mutable external-fact inventory | obsolete history | — | release stamp inside prose; the sentence is about the table, not a release | 81 → 88 |
| `docs/development/05-review-and-commit-protocol.md` | (introduction) | rewritten explanation | — | intro re-read: the carrier list now lives with invariant 2 | 104 → 89 |
| `docs/development/07-managed-update-rule.md` | (introduction) | rewritten explanation | — | intro re-read: the fresh-rescue WHY now lives in the startup chapter | 262 → 227 |
| `docs/development/09-process-custody-rule.md` | (introduction) | rewritten explanation | — | intro re-read: ledger/fingerprint mechanism and the stop protocol now live in Architecture | 259 → 261 |
| `docs/development/12-mcp-client-integration.md` | (introduction) | rewritten explanation | — | intro re-read: the stdio contract now lives in Architecture | 147 → 139 |
| `docs/development/14-build-and-ci.md` | (introduction) | rewritten explanation | — | intro re-read: the lock authority structure now lives in Architecture "Build scripts" | 149 → 122 |

### References re-pointed

| File | Heading | Disposition | Retained owner | What changed |
|---|---|---|---|---|
| `tests/test_v652_scratch_and_masking.py` | (test docstring) | reference re-pointed | docs/development/06-rules-by-change-class.md § "Loop / State-Machine Changes" | line-number reference replaced by the heading |

### Ported upstream edits (b6a7702b0)

Not part of the split or the subtraction: this records ordinary target drift.
Merging `managed/ouroboros` at `b6a7702b0` brought ten hunks that upstream wrote
into the two monoliths, which are entrypoints here. The entrypoints were kept
byte-for-byte and each hunk was applied once, at the chapter that owns the
section its surrounding text belongs to. None of the ten landed on a sentence
the subtraction had merged, so no hunk had to choose between two copies.

| Upstream hunk | Destination chapter | Heading |
|---|---|---|
| `review_owner_custody.py` module-tree row: owner-death batching, one off-lock event read, locked reconcile | `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (module tree, `ouroboros/`) |
| new `ouroboros/gateway/update_progress.py` module-tree row | `docs/architecture/01-high-level-architecture.md` | High-Level Architecture (module tree, `ouroboros/gateway/`) |
| Updates page: the executor's process-local stage observations, `update_progress_changed` invalidation, lock-free passive Git reads | `docs/architecture/03-web-ui-pages-and-buttons.md` | Dashboard |
| outbound envelope list gains `update_progress_changed` and its non-authority sentence | `docs/architecture/04-server-api-endpoints.md` | WebSocket protocol |
| new paragraph: the startup seal audit's bounded manifest set and shared archive-ID observation | `docs/architecture/06-agent-core.md` | Usage ledger substrate vs. accounting policy |
| paid-attempt locked write: the confirmed-dead PID batch, the empty-custody skip, one event snapshot then a locked reread | `docs/architecture/06-agent-core.md` | Review stack |
| Managed update ABI row gains `update_progress` and `update_progress_changed` | `docs/architecture/11-frozen-contracts-v1.md` | 11.1 What is frozen |
| new subsection "Shared behavior and data-flow changes" with its review-only enforcement | `docs/development/03-module-size-and-complexity.md` | Shared behavior and data-flow changes |
| projection-over-replay first bullet rewritten as whole-operation growth reasoning | `docs/development/03-module-size-and-complexity.md` | Invariant: Projection over replay (hot readers of growing stores) |
| item-24 trigger wording: data readers, startup/shutdown and other batch operations | `docs/development/03-module-size-and-complexity.md` | Invariant: Projection over replay (hot readers of growing stores) |
