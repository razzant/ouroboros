# DEVELOPMENT.md — Development Principles & Module Guide

## What This File Is

This is Ouroboros's **engineering handbook** — the bridge between philosophy (BIBLE.md) and architecture (ARCHITECTURE.md).

**BIBLE.md** answers *why* and *what matters*.
**ARCHITECTURE.md** describes *what exists right now*.
**DEVELOPMENT.md** answers *how to build* — the concrete principles, patterns, and checklists for writing, modifying, and reviewing code in this project.

## Scope

- **Code style & structure:** naming, file layout, module boundaries, error handling patterns.
- **Module lifecycle:** how to create a new module, what it must include, how it integrates.
- **Review & commit protocol:** what happens before code lands — gates, checks, invariants.
- **Testing standards:** what gets tested, how, minimum expectations.
- **Prompt engineering:** standards for writing and modifying LLM prompts (SYSTEM.md, CONSCIOUSNESS.md, etc.).
- **Integration patterns:** how modules communicate, data flows, shared state.

## What It Is NOT

- Not philosophy — that's BIBLE.md.
- Not an architecture map — that's ARCHITECTURE.md.
- Not a changelog — that's README.md + git log.
- Not aspirational — every rule here must reflect current practice or an immediately enforced standard.

## Relationship to Other Documents

```
BIBLE.md (soul — principles, constraints, identity)
    ↓ informs
DEVELOPMENT.md (hands — how to build, concretely)
    ↓ produces
ARCHITECTURE.md (mirror — what currently exists)
```

Rules in this file must not contradict BIBLE.md.

---

## Naming Convention

### General Rules

- **Language:** All code identifiers, comments, docstrings, and commit messages are in English.
- **Style:** Python PEP 8. Modules and variables — `snake_case`. Classes — `PascalCase`. Constants — `UPPER_SNAKE_CASE`.
- **Self-explanatory names** over abbreviations. A name should tell you what the thing *does*, not just what it *is*. Derived from P6 (Authenticity & Reality Discipline).

### Entity Types

| Entity Type | Purpose | Naming Pattern | Contains Business Logic? | Example |
|-------------|---------|----------------|--------------------------|---------|
| **Gateway** | Thin adapter to an external API. Wraps third-party SDK/HTTP calls into clean Python functions. | `{Platform}Gateway` | No. Pure I/O — translate calls in, translate responses out. | `BrowserGateway` |
| **Service** | Orchestrates a domain concern. May use one or more Gateways, manage state, apply business rules. | `{Domain}Service` | Yes. Coordinates, decides, transforms. | — |
| **Tool** | An LLM-callable function exposed to the agent. Thin wrapper that connects the agent to a Gateway or Service. | `{verb}_{noun}` (snake_case function) | Minimal. Validates input, calls Gateway/Service, formats output. | `read_file`, `browse_page`, `web_search` |

### Gateway Rules (recommended pattern, not enforced)

When adding a new external API integration, the recommended pattern is a **Gateway** class that isolates transport from business logic. The `ouroboros/gateways/` directory houses external API adapters. As the codebase grows, extract Gateways as needed.

When a Gateway exists, it should follow these guidelines:
- No business logic: no routing, no decisions. Just transport.
- Input/output: takes Python primitives, returns Python primitives.
- Error handling: translates platform-specific errors into consistent return values.
- Stateless where possible.

**Existing Gateways:**
- `ouroboros/gateways/claude_code.py` — Claude Agent SDK gateway. Two paths: `run_edit`
  (edit mode with PreToolUse safety hooks) and `run_readonly` (advisory review, no
  mutating tools). Structured `ClaudeCodeResult` output.

### Relationship Between Entities

```
LLM Agent
    |  calls
Tool (read_file, web_search, browse_page)
    |  delegates to
Gateway or direct implementation
    |  calls
External API / filesystem / subprocess
```

Not every layer is required for every operation. Simple cases (e.g., `read_file`) go Tool → filesystem directly.

### CLI / Headless Additions

- CLI commands should stay thin: parse flags, call gateway HTTP/SSE endpoints,
  render stdout/stderr/JSONL, and avoid duplicating runtime business logic.
- Headless task features belong behind gateway task APIs and the existing
  supervisor queue. Do not add a second scheduler for benchmarks.
- External workspace support must keep Ouroboros governance context pinned to
  the system repo while contextual repo tools resolve against the active
  workspace through `ToolContext.active_repo_dir()`.
- Workspace-mode tasks must use an explicit allowlist, reject system-repo/data
  overlap, require a git worktree root, and return patch artifacts instead of
  committing in the target repository.
- External workspace completion must be gated on explicit artifact finalization:
  `workspace.patch` is served through the task artifact endpoint, strict patch
  CLI modes fail on missing/empty/failed artifacts, and `workspace_patch.json`
  records diagnostics instead of silently treating patch-builder failures as
  empty diffs.
- Workspace preflight belongs in the headless task create flow as read-only
  facts: compact summary in task metadata/prompt, full
  `workspace_preflight.json` as an artifact. Do not dump full manifests into
  the prompt and do not add benchmark-specific instructions.
- Dependency installation guidance is workspace-scoped: project-local installs
  are allowed for external tasks, while system/global installs are pro-mode
  safety-reviewed attempts. `sudo` must be noninteractive (`sudo -n`).
- Treat workspace mode as routing plus guardrails, not an OS sandbox. If stronger
  isolation becomes required, add a real Docker/SSH/remote tool-execution layer
  instead of expanding shell-command heuristics.
- Do not add a CLI file-manager surface. Attachments, task artifacts, and logs
  are the allowed v1 file-adjacent surfaces.

### Cognitive Quality

- Do not lower model quality, reasoning effort, max-token budgets, or context
  breadth for core cognitive loops (especially Background Consciousness, review,
  and self-evolution) as an incidental cost or latency optimization.
- If a change intentionally narrows cognitive horizon, make that owner decision
  explicit in the plan, docs, tests, and review packet. Silent quality downgrades
  are continuity regressions, not refactors.

---

## Module Size & Complexity

Derived from P7 (Minimalism): entire codebase fits in one context window.

- Module target: ~1000 lines. Crossing that line is P7 pressure and should trigger extraction or an explicit justification.
- Module hard gate: 1600 lines for non-grandfathered modules in `tests/test_smoke.py`. Grandfathered (`GRANDFATHERED_OVERSIZED_MODULES` in `ouroboros/review.py`): `llm.py`, `claude_advisory_review.py`, `review_state.py`, `server.py`, and temporary v5.7.1 debt `git.py` — split deferred until each surface stabilises, with `git.py` expected to pay down in the next tools pass.
- Method target: <150 lines. Crossing that line is a decomposition signal, not an automatic failure by itself.
- Method hard gate: 300 lines in `tests/test_smoke.py`.
- Codebase-wide function-count hard gate: enforced by `tests/test_smoke.py` against the value defined in `ouroboros/review.py::MAX_TOTAL_FUNCTIONS` (single source of truth — bump the constant when adding a feature with an explicit comment justifying the increase).
- Function parameters: <8.
- Net complexity growth per cycle approaches zero.
- If a feature is not used in the current cycle — it is premature.

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
| Main task context (`context.py`) | ✅ full | ✅ full | ✅ full |
| Triad review (`tools/review.py`) | ✅ via preamble | ✅ via `_load_architecture_text` | ✅ via `_load_dev_guide_text` |
| ↳ Anti-thrashing (v4.35.1) | — | — | Open obligations loaded from `review_state` via `load_state(drive_root)` + `make_repo_key(repo_dir)`, injected unconditionally into `_build_review_history_section` prompt context. Same mechanism in `scope_review.py::_build_scope_prompt` (best-effort when `drive_root` available). |
| Background consciousness (`consciousness.py`) | ✅ full | ✅ full | — (not yet required) |
| Advisory pre-review (`tools/claude_advisory_review.py`) | ✅ via `_load_doc` | ✅ via `_load_doc` | ✅ via `_load_doc` |
| Scope review (`tools/scope_review.py`) | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting |
| Plan review (`tools/plan_review.py`) | full canonical doc + adaptive context level | full canonical doc + adaptive context level | full canonical doc + adaptive context level |
| Deep self-review (`deep_self_review.py`) | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting |

Plan review always keeps BIBLE.md, ARCHITECTURE.md, DEVELOPMENT.md, the proposed
plan, touched-file snapshots, and reviewer-slot framing as first-class context.
The agent must choose `context_level` explicitly; there is no host-side `auto`
heuristic. That field controls only the generated repository Atlas: `minimal`
omits Atlas accounting for bounded/local plans, while `localized`, `broad`, and
`constitutional` add progressively larger Atlas packs.

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

Reviewed commits now have an explicit **two-step gate**:

1. **Advisory freshness gate**: finish all edits, then run `advisory_review`.
   Without a bypass, `commit_reviewed` requires a fresh matching
   advisory run, no open obligations from earlier blocked rounds, and no open
   commit-readiness debt. Any edit after advisory makes it stale and requires a
   re-run. When debt remains, `review_status` reports `repo_commit_ready=false`
   plus `retry_anchor=commit_readiness_debt` so the next retry starts from the
   repeated root cause rather than one obligation at a time. `skip_advisory_review=True`
   is an **absolute** escape hatch: it short-circuits the entire commit gate
   after writing an audit entry to `events.jsonl`. Open obligations and open
   commit-readiness debt stay visible in `review_status` (`repo_commit_ready`
   stays `false`) but do NOT block the bypassed commit. Use bypass when advisory
   cannot run (provider outage, rate limit) or when the stale signals are known
   to be obsolete; in both cases subsequent `on_successful_commit()` clears
   them automatically.
2. **Unified pre-commit review**: once advisory is fresh, the reviewed commit path
   runs reviewer slots in parallel on the exact staged snapshot:
   - **Triad review** (`ouroboros/tools/review.py` + `ouroboros/triad_review.py`,
     orchestrated by `ouroboros/tools/parallel_review.py`): at least 2 reviewer
     slots (as configured in `OUROBOROS_REVIEW_MODELS`; duplicate model ids
     are valid independent slots) review the staged diff against
     `docs/CHECKLISTS.md`.
   - **Scope review** (`ouroboros/tools/scope_review.py`): one or more scope slots review
     completeness and cross-module consistency with touched context plus a
     generated repository Atlas (`review_context_atlas.compile_review_context_atlas`).

Triad and scope reviewers run concurrently via `concurrent.futures.ThreadPoolExecutor`
(orchestrated in `ouroboros/tools/parallel_review.py`). The caller receives one
combined verdict with all findings in a single round. Scope review findings block
only when `OUROBOROS_REVIEW_ENFORCEMENT=blocking`; advisory mode downgrades them
to warnings by operator policy. Scope review still runs even when triad blocks,
**except** when the fully assembled scope-review prompt exceeds the scope-review
input cap. The shared `REVIEW_PROMPT_TOKEN_BUDGET` / `_SCOPE_BUDGET_TOKEN_LIMIT`
(920K estimated tokens) is the INPUT-size SSOT, but scope review also reserves
`_SCOPE_MAX_TOKENS` for OUTPUT inside the reviewer's 1M context window, so it gates
the assembled input on the lower `_SCOPE_INPUT_TOKEN_LIMIT = min(920K, 1M −
_SCOPE_MAX_TOKENS − margin)` and can skip below 920K. In that case scope review is
skipped with a non-blocking advisory warning (never a hard provider 400). `docs/CHECKLISTS.md` remains the
single source of truth for review items; do not duplicate or fork checklist policy here.

Preferred workflow for non-trivial edits: choose the right edit tool first —
`edit_text` for one exact replacement and `write_file` for new files or
intentional full rewrites — then `advisory_review`, then `commit_reviewed`
immediately on the final diff.

Review preflight tests are hermetic. Any pytest run launched by
`advisory_review` or `commit_reviewed` must execute in a disposable git worktree
with a temporary `OUROBOROS_DATA_DIR` / `OUROBOROS_SETTINGS_PATH`, a temp
`PYTHONPYCACHEPREFIX`, and no inherited `OUROBOROS_MANAGED_BY_LAUNCHER`. Tests
may read the live source checkout as the candidate snapshot, but they must not
write the live repo or live `data/`.

Self-modification durability is local-first. A successful reviewed local commit
is the persistence boundary; `origin` push and CI are optional follow-up signals.
Missing `origin` is not a failed evolution. `managed` is the official
update/provenance remote and must not become the personal self-modification push
target.

Autonomous restarts must not erase active self-evolution. If an evolution task
requests restart while work is dirty and not yet represented by a reviewed local
commit, the runtime must preserve a rescue/transaction pointer and pause or stop
the campaign rather than `rescue_and_reset` and continue as if nothing happened.
Explicit owner restart remains an operator action with broader authority.

The full pre-commit review checklists live in **`docs/CHECKLISTS.md`** —
the single source of truth (Bible P7: DRY).

This section defines what "DEVELOPMENT.md compliance" means in practice — it is the
detailed expansion of the `development_compliance` item in `docs/CHECKLISTS.md`.

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
- [ ] No method exceeds the practical target (150 lines) or the hard gate (300 lines)
- [ ] Total Python function count stays under the current smoke hard gate (consult `ouroboros/review.py::MAX_TOTAL_FUNCTIONS` for the active value; bump with a comment if a feature requires more headroom)
- [ ] No function has more than 8 parameters
- [ ] No gratuitous abstract layers (Bible P7)

#### Structural Rules
- [ ] New Tool? `get_tools()` exports it using the `ToolEntry` pattern from `registry.py`, AND an explicit entry is added to `ouroboros/safety.py::TOOL_POLICY` (`POLICY_SKIP` for trusted built-ins, `POLICY_CHECK` for opaque or outward-facing ones). Without the policy entry the tool falls through to `DEFAULT_POLICY = POLICY_CHECK` and pays a light-model LLM call per invocation, and the `test_tool_policy_covers_all_builtin_tools` invariant will fail.
- [ ] New Gateway (if extracted)? Contains no business logic, only transport.
- [ ] New memory/data files? Should they appear in LLM context (`context.py`)?

#### Skill Repair Task Constraints
- Skill repair tasks use structured `task_constraint.mode="skill_repair"`, not prompt markers.
- In repair mode, edit paths are payload-relative: `plugin.py` means the selected `data/skills/{external,clawhub,ouroboroshub}/<skill>/plugin.py`.
- Use `edit_text` for one exact replacement and `write_file` only for new files or intentional full rewrites with `root=skill_payload`.
- Finish repair with `skill_preflight` and `skill_review`; grants and enablement stay owner-controlled.
- Repair mode is a stricter UI lane, not the only path for skill authoring. In `runtime_mode=light`, ordinary chat tasks may edit explicit `data/skills/{external,clawhub,ouroboroshub}/<skill>/...` payloads via `write_file`/`edit_text` with `root=skill_payload`, `bucket`, and `skill_name`. Explicit repo/data paths keep their own address space and ignore stale short-form args. Core/repo paths, `data/skills/native/*`, `data/state/skills/*`, marketplace/provenance sidecars, and direct `run_command` writes to repo targets remain blocked.
- New path checks for skill edits must use `ouroboros.contracts.skill_payload_policy` rather than reimplementing bucket/path traversal logic in each tool.

#### Native-Risk Extension Dispatch
- `type: extension` skills with reviewed isolated dependency envs must not import `plugin.py` or execute handlers inside `server.py`, even when the dependency tree looks pure-Python. Payload-native marker files (`.so`, `.dylib`, `.dll`, `.pyd`) also force child dispatch as defense in depth, but opaque native payloads remain subject to the skill-review checklist and are not newly allowed by this runtime fallback.
- Keep the split explicit: no-dependency pure-Python extensions may use `extension_loader`'s in-process PluginAPI path; isolated-dep/native-marker extensions are cataloged and dispatched by `extension_process_runner` short-lived child processes.
- Tool, HTTP route, and WebSocket handler proxies must return normal tool errors / HTTP 502 / WS log messages on child crash, invalid JSON, timeout, or abort. A child `SIGABRT` is a handled extension failure, not a server crash.
- Child processes must use scrubbed env, per-skill grants, per-skill isolated deps, process-group tracking, output caps, and timeout cleanup. Do not add fallback code that imports native-risk plugin modules in the host process.

#### Light Mode External Deliverables
- `runtime_mode=light` is a self-modification boundary, not an OS sandbox. User-visible deliverables are allowed when they are outside the Ouroboros repo/control-plane.
- Preferred flow: `task_drive` for scratch, `artifact_store` for canonical deliverables, and `user_files` for the owner's visible copy (for example `Desktop/report.html`). `write_file(root=user_files)` and declared process `outputs` must register/copy canonical task artifacts.
- `run_command`/`run_script`/`start_service` may use cwd under `active_workspace`, task-scoped `task_drive`, task-scoped `artifact_store`, and external `user_files` where the active profile permits it. In light direct tasks, omitted `run_script.cwd` defaults to task scratch instead of the Ouroboros repo; long-running services in light must use an explicit external/task/artifact cwd. Declared service `outputs` are copied into the task artifact store when the service stops.
- `claude_code_edit` remains a first-class high-capability coding tool for substantial external artifacts; do not remove, hide, or downgrade it when refactoring Tool API names. It may run under external user/task/artifact cwd in direct light tasks, and under active workspace/task/artifact cwd in workspace tasks, while Ouroboros repo/control-plane cwd stays on the reviewed self-modification path. Use `outputs=[...]` when it creates deliverables that must be audited.
- Do not recommend `runtime_data/uploads`, skill payloads, or owner state directories as generic artifact transport.

#### Live Subagent Task Constraints
- Live subagents are scheduled only through the existing `schedule_subagent` tool.
  Its public schema is strict: `objective` and `expected_output` are required;
  `role`, `context`, `constraints`, and `memory_mode` are optional. Do not
  reintroduce public `parent_task_id` or `description` arguments; lineage comes
  from `ToolContext`.
- Live `memory_mode=shared` is disabled. Keep `forked` and `empty` as the only
  live subagent modes unless a later design adds sanitized shared-context v2.
- External `/api/tasks` and CLI requests must reject forged
  `delegation_role=subagent`; only `schedule_subagent` may create subagents.
- `task_constraint.mode="local_readonly_subagent"` must be enforced twice:
  schema discovery exposes only the local-readonly allowlist, and registry
  execution rejects forbidden calls even when invoked manually.
- `task_constraint` boolean parsing must be strict; strings such as `"false"`
  are false, never truthy through Python's `bool("false")`.
- Subagent changes must keep writes, commits, review mutation, runtime control,
  tool expansion, skills, MCP/extensions, shell, and further `schedule_subagent`
  recursion blocked unless a later accepted design explicitly changes the
  permission model.
- `read_file(root=runtime_data)` and `list_files(root=runtime_data)` secret/control-file denials are subagent-scoped.
- Browser isolation for local-readonly subagents is DNS fail-closed: block
  non-HTTP(S), loopback/private/link-local/reserved/unspecified literal IPs,
  unresolved hostnames, and hostnames resolving to any blocked IP before goto,
  after redirects, and in route handlers.
- Effective task status belongs in `ouroboros/task_status.py`. Do not duplicate
  child-drive result merge or terminal-status logic in gateways/tools; use
  `load_effective_task_result`, `effective_task_result`, and bounded wait
  helpers. `wait_task` and `wait_tasks` results must remain untruncated.
- `forward_to_worker` may write only to validated running tasks whose lineage
  belongs to the current task/root, and must route forked/empty child subagents
  to the child-drive mailbox.
  Do not broaden generic data-tool behavior for normal tasks while fixing
  subagent isolation.
- The pre-final handoff reminder is a compact effective-status snapshot. Full
  untruncated child handoff belongs to `get_task_result`, `wait_task`, and
  `wait_tasks`. Do not add shared ledgers, automatic memory merges, or new
  settings/endpoints unless the accepted plan explicitly calls for them.

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
- New visual dimensions should become CSS variables first (`--pill-*`, `--button-*`, `--page-header-*`, etc.) and then be consumed by shared classes. Hardcoded page-local dimensions are review debt unless the component is genuinely unique.

#### LLM Call Rules
- [ ] New LLM calls go through the shared `LLMClient` / `llm.py` layer — no ad-hoc HTTP clients or direct provider SDKs outside that layer. **Exception (v5.7.0+):** skill / extension `plugin.py` modules may call providers directly because they have not yet been migrated to a host-mediated `api.invoke_llm(...)` bridge. When that bridge lands, the exception goes away. Runtime callers (anything inside `ouroboros/`) must still use `LLMClient`.

#### Loop / State-Machine Changes
- [ ] Changes to `loop.py` or other task state-machine logic include adversarial tests for malformed output, false-completion prevention, replay/log durability, and failure modes — not just the happy path.
- [ ] Audit/checkpoint rounds must not silently reuse the normal final-answer path unless that invariant is explicitly tested and documented.
- [ ] Host-enforced task-acceptance review eligibility is derived from observable effects, not message content. `outcomes.turn_has_reviewable_effects` is the SSOT for "did this turn do reviewable work"; `_task_acceptance_eligible` consumes it for `required` mode. `auto` stays LLM-first (`return False`). Cognitive-memory updates are not reviewable effects. Keep this an effect/structured-fact gate (Bible P3), never a keyword/heuristic over the user's message (Bible P5). The `required` direct-chat exemption depends on `ToolContext.is_direct_chat` being set (from `task._is_direct_chat`); any new direct-chat entry point must set it, or greetings will be reviewed.

#### Cognitive Artifact Integrity
- [ ] Cognitive artifacts (identity.md, scratchpad, task reflections, review outputs, pattern register) must NOT use hardcoded `[:N]` truncation. If content must be shortened, include an explicit omission note (e.g. `⚠️ OMISSION NOTE: truncated at N chars`).
- [ ] `BIBLE.md`, `docs/ARCHITECTURE.md`, and `docs/DEVELOPMENT.md` are **core governance artifacts**. All primary reasoning flows (triad review, consciousness, advisory pre-review, deep review) include them as first-class sections — see the "Core Governance Artifacts" table. If you add a new reasoning flow, it MUST follow this contract, not rely on touched-file inclusions.

---

*This section is the authoritative definition of "DEVELOPMENT.md compliance" referenced in the `development_compliance` item in `docs/CHECKLISTS.md`.*

---

## Platform Abstraction Rule

All platform-specific code **MUST** go through `ouroboros/platform_layer.py`.

### Shared State-File Helpers

Durable JSON state files should use the SSOT helpers in `ouroboros/utils.py`:
`atomic_write_json(path, payload, trailing_newline=False, fsync=False)` for
write-then-rename persistence and `read_json_dict(path)` for dict-shaped JSON
reads. Lockfile acquisition should go through
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
  
  Not scanned by the AST guard: `launcher.py` (immutable outer shell, intentionally excluded) and subprocess flag patterns (`creationflags`, `start_new_session`). For subprocess isolation, use `subprocess_new_group_kwargs()` and `subprocess_hidden_kwargs()` from `platform_layer.py` — enforced by code review and the `cross_platform` checklist item.
- **Pre-commit review**: checklist item `cross_platform` (#15) catches violations during code review.
- **CI matrix**: tests run on Ubuntu, Windows, and macOS to catch runtime failures.

### Adding new platform-specific code

1. Add the cross-platform wrapper to `platform_layer.py`.
2. Import and use the wrapper in callers.
3. Add platform-conditional tests if behavior differs across OSes.

---

## Design System

Ouroboros uses **glassmorphism** as its visual language. All interactive surfaces follow this pattern:

```css
background: rgba(26, 21, 32, 0.62–0.88);
backdrop-filter: blur(8–16px);
border: 1px solid rgba(255, 255, 255, 0.06–0.12);
```

### Floating overlay transparency (v5.7.0+)

Floating chrome that overlays scrolling content (chat header, sticky tab
strips inside Settings/Dashboard/Skills, files preview gradient) follows ONE
shared formula and never relies on a separate fade-overlay element:

1. The chrome element is `position: absolute` with the appropriate edge
   (`top: 0` for headers, `bottom: 0` for bottom overlays, etc.) and
   covers the whole horizontal axis.
2. Its background is a **single 4-stop linear gradient** that fades from
   the dense brand background at the chrome's anchor edge to fully
   transparent at the opposite edge.
3. `backdrop-filter: blur(10–14px)` is applied on the same element
   (the host always supplies `-webkit-` prefix in lockstep).
4. **A CSS `mask-image` matching the gradient direction fades the blur
   in lockstep**: `mask-image: linear-gradient(0deg, black 0%, black 70%, transparent 100%)`.
   This is the rule that prevents the visible "glass edge" the v5.6.x
   chat dock had — without the mask the blur creates its own hard
   horizontal line at the gradient's transparent stop.
5. The scrollable surface reserves enough top/bottom padding so content is
   reachable outside the overlay's dense zone.

**Chat input dock exception:** the bottom composer intentionally splits the
formula. `#chat-input-area` is a compact absolute bottom overlay with a
darkening gradient only (no wrapper `backdrop-filter`), so message text fades
under the dock without a tall smeared blur band. The active textarea itself
is the frosted surface (`background: rgba(26,21,32,0.55);
backdrop-filter: blur(20px)`). `#chat-messages` reserves bottom padding
through `--chat-input-reserve`, which JS sets from the actual dock height
plus a small buffer; mobile adds safe-area on top of that.
`updateMessagesPadding()`
preserves scroll stickiness only; it must not mutate DOM padding.

Do NOT introduce a separate `.chat-bottom-fade` (or analogous overlay)
layer. A second fade layer compounds the gradient and can produce a visible
"double dim" especially over short messages.

### Navigation rail spacing (v5.7.0+)

The desktop `#nav-rail` uses Material 3 / Apple HIG navigation-rail
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

| Role | Value | Usage |
|------|-------|-------|
| Primary | `rgba(201, 53, 69, ...)` = `#c93545` | Nav buttons, chat cards, borders |
| Hover/focus | `rgba(232, 93, 111, ...)` = `#e85d6f` | Focus glow, settings hover |

Use the primary accent for new features. Avoid introducing additional red/crimson shades.

### Border radius scale

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-xs` | `3px` | Micro accents (progress bars) |
| `--radius-sm` | `8px` | Small controls, filter chips |
| `--radius-md` | `10px` | Chips, log-counter pills, page-fade rules |
| `--radius` | `12px` | Inputs, inner cards |
| `--radius-lg` | `16px` | Nav buttons, chat/live cards |
| `--radius-xl` | `20px` | Logo images, large media |
| *(no token)* | `18px` | Section cards (settings, form panels) |
| *(no token)* | `24px` | Modal/wizard shells, chat input |

Use CSS variables where possible. Do not introduce new hardcoded radius values.
When a new radius value is needed, add it to `:root` in `web/style.css` first.

### Interactive states

```css
hover:  transform: scale(1.02–1.04) + border-color +1 step brightness
active: background rgba(201,53,69, 0.12) + crimson glow
focus:  border-color rgba(232,93,111,0.4) + box-shadow 0 0 0 3px rgba(201,53,69,0.10)
```

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

Use **crimson** (`rgba(248, 130, 140, ...)`) for active/working states everywhere — not blue.
The Logs page phase badges now match Chat live card colors.

### No inline styles in JS

JS modules that generate HTML must use CSS class names, not `style=""` attributes.
This is enforced by reviewer policy — `.style.*` assignments on DOM elements (e.g.
`element.style.display`, `element.style.color`) will produce a REVIEW_BLOCKED finding.
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
`kind: "declarative"` / `schema_version: 1` covers forms, actions, markdown, JSON, key/value
summaries, tables, progress, files, galleries, image/audio/video media, and
v5.7.0 map/calendar/kanban components. New common widget capabilities should
extend that declarative schema and its tests.

v5.7.0 adds one deliberate exception for rare custom UI: `kind: "module"`
loads reviewed skill-provided `widget.js` into a sandboxed `srcdoc` iframe
(`sandbox="allow-scripts"`, **no** `allow-same-origin`). The parent host
fetches the reviewed JS from `/api/extensions/<skill>/module/<entry>` and
injects a constrained `fetch` bridge that only proxies
`/api/extensions/<skill>/...` routes. This is not same-origin SPA execution;
the module cannot access app cookies or `localStorage`.

Rules for widget changes:

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
- Do not load arbitrary JS modules from skill directories into the SPA origin.
  `kind: "module"` is allowed only through the sandboxed iframe + parent fetch
  bridge above, and must be covered by the `widget_module_safety` review item.
- Add/update `tests/test_widgets_ui_static.py` for every new component kind or
  media policy.

---

## MCP Client Integration

The base-runtime MCP surface is a **client only** for trusted HTTP/SSE MCP
servers. It borrows external tools and exposes them through `ToolRegistry`;
it does not expose Ouroboros as an MCP server.

Rules for MCP changes:

- Keep MCP disabled by default. `MCP_ENABLED`, `MCP_TOOL_TIMEOUT_SEC`, and
  `MCP_SERVERS` are the only base settings. `MCP_SERVERS` stays in
  `settings.json` as a list of dicts; do not serialize it into env vars.
- Support only `streamable_http` and `sse` in the base runtime. Stdio MCP,
  resources, and prompts are separate architectural changes.
- All MCP tool names must be produced by
  `ouroboros.mcp_client.make_tool_name()` and must remain provider-safe
  (`mcp_<server>__<tool>`, max 64 chars).
- All URL and header validation lives in `ouroboros/mcp_client.py`.
  Do not duplicate scheme, metadata-host, link-local, auth-header, or
  control-character checks in UI/API modules.
- `auth_token` values flow only through `settings.json` and in-process
  manager state. `/api/settings` masks them, `/api/settings` POST rehydrates
  masked values from old settings, and `/api/mcp/status` exposes only
  `auth_configured`.
- MCP descriptions and tool results are server-supplied untrusted data.
  Descriptions must be wrapped before reaching the LLM, UI strings must be
  escaped, and MCP text must never be treated as policy.
- MCP tools are non-core. They must require `list_available_tools` /
  `enable_tools`, be blocked in skill-repair/heal contexts, and run through
  `safety.check_safety` before dispatch.
- When changing MCP behavior, update the focused MCP tests:
  `tests/test_mcp_client.py`, `tests/test_mcp_api.py`,
  `tests/test_mcp_registry_integration.py`,
  `tests/test_mcp_settings_roundtrip.py`, and
  `tests/test_mcp_ui_static.py`.

---

## Gateway Boundary Pattern

Browser-facing backend work goes through `ouroboros/gateway/`.

- `gateway/router.py` is the single place that mounts Starlette routes for
  `/api/*` and `/ws`. Do not add new browser routes directly in `server.py`.
- `gateway/contracts.py` is the frozen frontend/backend contract. It contains
  endpoint tokens, WebSocket discriminators, and TypedDict envelope shapes.
  This file is protected by `runtime_mode_policy.py` and may be edited only in
  `runtime_mode='pro'`.
- Domain handlers live in sibling modules: `settings.py`, `control.py`,
  `files.py`, `models.py`, `extensions.py`, `marketplace.py`, `mcp.py`,
  `host_service.py`, `history.py`, `tasks.py`, `schedules.py`, `logs.py`,
  and `state.py`.
- Frontend code calls backend APIs through `web/modules/api_client.js`.
  `web/modules/api_types.js` mirrors core contracts via JSDoc so frontend
  contributors have a visible surface without TypeScript, codegen, or a build
  step.
- Any new browser endpoint must update `gateway/contracts.py`,
  `gateway/router.py`, `web/modules/api_client.js` when the UI consumes it, and
  the parity/smoke tests in `tests/test_gateway_parity.py` /
  `tests/test_gateway_smoke.py`.

---

## Build & CI

### Pytest marker lanes

Default local pytest excludes costly or environment-dependent lanes:
`integration`, `browser`, `ui_browser`, `ui_browser_docker`, and
`portable_detail`. CI opts into them explicitly:

- `integration` runs real provider checks, including Cloud.ru when
  `CLOUDRU_FOUNDATION_MODELS_API_KEY` is configured and GigaChat when
  `GIGACHAT_CREDENTIALS` is configured.
- `browser` launches real Playwright Chromium for agent browser tools.
- `ui_browser` launches the host-side web UI under Playwright.
- `ui_browser_docker` talks to an `ouroboros-web:test` container and must
  skip cleanly when Docker is unavailable locally.
- `portable_detail` covers build/portable artifact invariants and also runs
  inside Docker in the manual/tag CI tier.

When adding a new opt-in lane, register the marker in `pyproject.toml`, add
a collect-only zero-test guard in CI, and keep the default local addopts
token-safe and Docker-safe.

### GitHub Actions: secrets in step-level `if:` conditions

GitHub Actions rejects `secrets.*` inside step-level `if:` expressions, and a
step's own `env:` block is not visible to that same step's `if:`. Map secrets at
the job-level `env:` block, then gate steps with `env.*`.

```yaml
jobs:
  build:
    runs-on: macos-latest
    env:
      # job-level: visible to step-level `if:` via env.*
      BUILD_CERTIFICATE_BASE64: ${{ secrets.BUILD_CERTIFICATE_BASE64 }}
      P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
    steps:
      - name: Import Apple signing certificate
        # ✅ env.* — visible inside step-level if
        if: env.BUILD_CERTIFICATE_BASE64 != '' && env.P12_PASSWORD != ''
        run: |
          echo "${BUILD_CERTIFICATE_BASE64}" | base64 -d > cert.p12
          security import cert.p12 -P "${P12_PASSWORD}" ...
      - name: Cleanup keychain
        if: always() && env.BUILD_CERTIFICATE_BASE64 != ''
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

When Apple signing secrets are configured, the macOS shard imports the Developer
ID certificate into a temporary keychain and `build.sh` signs the `.app` and
`.dmg` via `SIGN_IDENTITY`. Apple secrets are job-level env values guarded by
`matrix.os == 'macos-latest'`, so Linux/Windows shards receive empty strings.
If `APPLE_ID` and `APPLE_APP_SPECIFIC_PASSWORD` are present, notarization runs;
otherwise the DMG ships signed but not notarized. Notary/stapler failures are
soft warnings, recorded through `NOTARIZE_OUTCOME`, so transient Apple issues do
not silently drop the macOS artifact. Cleanup uses `always()` plus macOS/env
guards, and signing material never persists across runs.
