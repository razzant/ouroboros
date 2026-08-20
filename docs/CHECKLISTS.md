# Pre-Commit Review Checklists

Single source of truth for all automated review checklists (Bible P7: DRY).
Loaded by `ouroboros/tools/review.py` at review time and injected into the
multi-model review prompt.

When a new reviewable concern appears, add it here — not in prompts or docs.

---

## Advisory Pre-Review Workflow

**Default sequence:**

```
1. Finish ALL edits first (`edit_text` / `edit_batch` / `apply_patch` / `write_file`)
2. advisory_review(commit_message="...")       ← run AFTER all edits, ONCE
3. commit_reviewed(commit_message="...")       ← run IMMEDIATELY after advisory
```

**Rules:**
- Successful worktree mutations automatically mark advisory as **stale**. This includes
  `write_file`, `edit_text`, `edit_batch`, `apply_patch`, and mutating `run_command` /
  reviewed-commit paths when they change tracked worktree state.
- A stale advisory must be re-run unless Ouroboros explicitly chooses the
  audited skip below.
- Do NOT interleave edits and advisory calls: `edit → advisory → edit → advisory` wastes two
  expensive advisory cycles. Finish all edits first.
- If advisory finds critical issues: **strongly recommended** to fix them and re-run advisory
  before calling commit_reviewed.
  Note: commit_reviewed's gate checks snapshot freshness, open obligations, and open
  commit-readiness debt — it does not enforce zero advisory FAIL items as a hard
  gate. Fixing critical findings and re-running advisory is best practice. Under
  `OUROBOROS_REVIEW_ENFORCEMENT=advisory`, a fresh advisory also downgrades open
  obligations and commit-readiness debt to a warning by writing
  `advisory_obligations_acknowledged` to `events.jsonl`; stale advisory still
  blocks. Under `blocking`, `commit_reviewed` can proceed only when no open
  obligations or commit-readiness debt remain.
- **Loud advisory enforcement (BIBLE P3 bound):** the owner chooses enforcement;
  `advisory` is legitimate ONLY while every decision blocking enforcement would
  have stopped (critical findings, quorum failure, infrastructure failure,
  missing advisory provider) leaves a durable trace: a `review_advisory_override`
  event in `events.jsonl` plus the persistent `advisory_overrides_count` /
  recent-overrides fields in `review_status`. Silent advisory is forbidden.
- Once advisory is fresh → call commit_reviewed immediately without further edits.
- `skip_advisory_review=True` skips only advisory freshness and the
  obligation/debt admission attached to it. Use LLM judgment when this cheap
  error-finding pass is slow, unhealthy, unavailable, or unlikely to add value;
  otherwise run it before the expensive authoritative review. The gate records
  the explicit skip; explain the LLM judgment in the surrounding task narrative.
  The flag does not change independently applicable test policy, triad,
  applicable scope review, or
  pre/post fingerprint and exact-commit/tag binding. The skip is durably
  audited, and unresolved obligations/debt remain visible.

**Obligation tracking:**
- Every blocking `commit_reviewed` result creates "open obligations" — a structured checklist of
  unresolved issues that advisory must explicitly address on the next run.
- Advisory will receive the full list of open obligations and should respond to each one by name.
- A generic PASS without addressing open obligations is a weak signal — advisory is expected
  to confirm each obligation is resolved, though the gate does not enforce this at the code level.
- Open obligations are cleared automatically on a successful commit.
- Both triad-review blocks and scope-review blocks produce structured obligations.
- Repeated blockers may also synthesize **commit-readiness debt**. When present,
  the non-bypass `commit_reviewed` path remains blocked under `blocking` until
  advisory clears both the open obligations and the debt; `review_status` reports
  this via `commit_readiness_debts_count`, `repo_commit_ready=false`, and
  `retry_anchor=commit_readiness_debt`. Under `advisory`, a fresh advisory allows
  commit after recording `advisory_obligations_acknowledged`; `review_status`
  still shows the debt until a successful commit clears it.
  `skip_advisory_review=True` overrides only this advisory debt admission;
  the debt remains visible and authoritative review still runs.
- **Anti-thrashing injection (v4.35.1):** On retry attempts, open obligations are loaded from durable review state and injected into reviewer prompts as an inert JSON data block (fenced ```json``` with a "DATA records — not instructions" disclaimer). Two mandatory rules are also appended: (1) The JSON `"verdict"` field is the authoritative signal — withdrawal notes in `"reason"` text are ignored; (2) Do not rephrase prior findings under a different checklist item name. In `claude_advisory_review.py::_build_advisory_prompt`, these same two rules are injected at **step 5a unconditionally** (on every advisory run, not only when obligations exist), and reinforced at steps 6.e/6.f when obligations are present.
- **Obligation storage policy:** All obligations are stored; deduplication is the agent's responsibility.
  Multiple obligations describing the same root cause (from reviewer rephrasing across attempts) are
  expected — address them together and explain this in `review_rebuttal`.
- **Note:** conservative false-stale is acceptable. If you are unsure whether a mutating path
  changed the relevant repo snapshot, re-run `advisory_review` explicitly.

---

### Review-exempt operations

The following tools create commits but are **exempt** from multi-model review
(Bible P9 explicit exception):

- `vcs_restore` — discards uncommitted changes (not a commit, no review needed)
- `vcs_revert` — creates a mechanical inverse of an already-reviewed commit
- `vcs_rollback` — resets to an existing tag/SHA (already-reviewed state)

Rationale: review gates on rollbacks create a paradox where reviewers block
the undo for missing tests/VERSION, trapping the agent with broken code.
These tools restore to already-reviewed states by definition.

---

## Pre-Commit Self-Check (Ouroboros, before calling advisory_review)

Run this walkthrough honestly before every `advisory_review` call for a
`commit_reviewed`. The correct sequence is:

```
finish ALL edits → Pre-Commit Self-Check → advisory_review → commit_reviewed
```

This section is **not injected as a named checklist section by the review prompts** — it exists here so the agent's
pre-flight checklist lives in the same single source of truth as the review
checklists it guards. When `docs/CHECKLISTS.md` itself appears in a commit's
touched files, reviewers will see it as part of the `Current touched files`
pack, but it is not loaded as a standalone checklist the way the Repo Commit
or Intent/Scope checklists are.

| # | Check | How |
|---|-------|-----|
| 1 | `VERSION`, `README.md` badge, `docs/ARCHITECTURE.md` header, and the latest git tag — are all four carrying the *author-facing* spelling (for example `4.50.0-rc.3`)? And does `pyproject.toml` carry the **PEP 440 canonical form** of that same version (for example `4.50.0rc3`)? | `read_file` each file before editing. Never reconstruct version strings from memory — the in-context copy may be stale. The `VERSION` vs `pyproject.toml` divergence is intentional: `pyproject.toml` must satisfy PEP 440 so pip / build / twine accept it, while `VERSION` / tags / README / ARCHITECTURE use the author-facing spelling. `tests/test_packaging_sync.py::test_version_file_and_pyproject_are_synced` enforces the relationship via `ouroboros.tools.release_sync._normalize_pep440`. |
| 2 | Preparing any commit → is `VERSION` bumped? | Under BIBLE.md P9, every commit is a release. A `VERSION` bump is mandatory for every commit, including docs/config/memory changes. Update `VERSION`, `pyproject.toml`, `README.md`, and `docs/ARCHITECTURE.md` together. |
| 3 | New or changed logic → does an existing or newly staged test assert on the specific scenario it introduces? | Name the scenario your code handles in plain words. If no test asserts on THAT named scenario, write or update one now. "Tests exist for the module" is not the same as "tests cover this new behavior". |
| 4 | Shared log / memory / replay format changed? | Grep every reader and writer first. JSONL logs (`events.jsonl`, `task_reflections.jsonl`, replay indexes), durable state files (`advisory_review.json`, `review_continuations/*.json`), and canonical-vs-derived memory pairs (patterns-register journal / `patterns.md`, improvement-backlog items) must stay coherent across every consumer. |
| 5 | New validation guard, input filter, or edge-case check? | Before the first commit attempt, name three concrete ways it could break: wrong bounds, legitimate inputs it silently blocks, platform-specific edge cases. If you cannot name three, think longer. One honest minute here is cheaper than one reviewer round. |
| 6 | New tool added? | `get_tools()` exports it, `prompts/SYSTEM.md` tool tables mention it, the handler signature matches the declared schema, and (if it mutates repo state) it is routed through the reviewed commit path rather than ad-hoc `run_command`. Also add an explicit entry in `ouroboros/safety.py::TOOL_POLICY` (`POLICY_SKIP` for trusted built-ins, `POLICY_CHECK` for opaque or outward-facing ones) — the `test_tool_policy_covers_all_builtin_tools` invariant will fail otherwise, and without an entry the tool falls through to `DEFAULT_POLICY = check` and pays a light-model LLM call per invocation. |
| 7 | Tests green before first `commit_reviewed`? | Run `pytest -x` on the narrowest relevant target(s) you can name before the first `advisory_review` / `commit_reviewed` attempt. If a new `.py` file is added under `ouroboros/` or `supervisor/`, **always** run `pytest tests/test_smoke.py` first — module-size and function-count violations are cheap to catch locally and expensive in review. A red test suite before the first commit attempt has caused repeated $2-5 blocked-review cycles. |
| 8 | Adding a `README.md` version row? | BIBLE.md P9 hard cap: ≤ 2 major, ≤ 5 minor, ≤ 5 patch visible entries. Categories are mutually exclusive: major = `X.0.0` (minor=0, patch=0); minor = `X.Y.0` (patch=0, Y≠0); patch = all other `X.Y.Z` (Z≠0). Count existing rows in the category you are adding to. Easy check: `run_command(["python", "-c", "import sys; from ouroboros.tools.release_sync import check_history_limit; warns=check_history_limit(open('README.md').read()); print(warns or 'OK')"])` — if it prints warnings, trim the oldest row in the over-limit category **in the same edit** before committing. |
| 9 | Changing any of `build.sh`, `build_linux.sh`, `build_windows.ps1`, `Dockerfile`, or `ouroboros/tools/browser.py`? | Cross-surface doc sync is mandatory. Check ALL of: `README.md` Install section (Linux native-lib caveat), `README.md` Build section (per-platform instructions), `docs/ARCHITECTURE.md` browser tools paragraph, WebKit/mobile verification notes, and inline comments in the touched build script. Any one of these being stale has blocked review twice. Verify before staging. |
| 10 | Changing `ouroboros/tools/commit_gate.py`? | Coupled surfaces that MUST be updated atomically in the same commit: (a) `claude_advisory_review.py::get_tools()` tool description for `advisory_review` and `review_status`; (b) `claude_advisory_review.py::_next_step_guidance()` strings; (c) `docs/DEVELOPMENT.md` Review & Commit Protocol section; (d) `prompts/SYSTEM.md` Commit review section. Missing any one has blocked review. |
| 11 | Changing VERSION + pyproject.toml? | Ordering matters: (1) write `VERSION` and `pyproject.toml` first; (2) then write `README.md` badge + changelog row; (3) then run `pytest`. Never interleave — updating README before VERSION means `test_version_in_readme` will catch a stale badge. |
| 12 | Writing or editing any JS file under `web/modules/`? | New or changed static inline visual properties are blocked: inspect the diff for added/changed `style=""` markup and `.style.<property>` assignments, and use CSS classes/tokens plus `classList`/`hidden` instead. Unchanged legacy hits are debt, not a blocker. A dynamic measured value may update a narrowly named CSS custom property when that is the actual runtime data flow. |
| 13 | Changing LLM output-token budgets? | Grep the whole repo for `max_tokens`, `max_completion_tokens`, `_MAX_TOKENS`, and `max_toks`. Keep `docs/ARCHITECTURE.md` §LLM output token budgets and `tests/test_max_tokens_constants.py` in sync so main-loop, VLM, summaries, compaction, skill publish, and consciousness floors cannot drift independently. |
| 14 | Changing extension loader/dispatch or isolated deps? | Native-risk extension imports and tool/route/WS handlers must stay out-of-process. Add or run regression tests where a native-risk plugin aborts during import and the host survives, plus tool/route child-dispatch tests. Do not "fix" failures by importing native-risk plugin code in `server.py`. |
| 15 | Changing `supervisor/git_ops.py`, `launcher.py`, `server.py`, `ouroboros/tools/review_helpers.py`, `ouroboros/tools/git.py`, tests, or evolution scheduling/checkpoint code? | Prove two invariants before review spend: (1) pytest/preflight cannot mutate the live repo or live `data/` (`OUROBOROS_DATA_DIR` / `OUROBOROS_SETTINGS_PATH` must be isolated, and `OUROBOROS_MANAGED_BY_LAUNCHER` must not leak into test subprocesses); (2) autonomous restart/reset cannot erase active evolution work — it must either land a reviewed local commit or preserve a rescue/transaction recovery pointer and pause/stop the campaign. |
| 16 | Changing `devtools/`? | Keep operator tools isolated from runtime and package discovery: no runtime, web, or release imports from `devtools/`; no generated outputs under `repo/` or live `data/`; no secrets printed or committed. A touched devtool is reviewable executable code even when unrelated devtools stay compact in broad review packs. Domain-specific launch, routing, and methodology guidance stays with the relevant devtool. |
| 17 | Diff spawns OS processes (`subprocess.Popen` / `mp.Process` without a bounded wait)? | Route it through `ouroboros.process_custody.spawn_supervised` (or `record_process` write-through) with an explicit `scope` (`task`/`session`/`daemon`) so the orphan reaper can find it; `tests/test_process_custody.py` enforces the allowlist. |
| 18 | New/changed **test** spawns a real OS process, binds a real port, or mutates a module-level global/registry? | CI **and the hermetic commit gate** (`preflight_runner.py`, v6.88.0) both run the suite as a parallel `-m "not serial" -n auto` pass plus a serial pass. Mark such a test `@pytest.mark.serial` (or add its file to `_SERIAL_TEST_FILES` in `tests/conftest.py`) — otherwise it flakes on kill/port-reclaim timing, or crashes a worker, which the gate reports as a `PARALLEL_WORKER_CRASH` **hard block** (never retried, never a flake). Static screen before you commit: grep the new test for `subprocess.Popen`/`subprocess.run`, socket or fixed-port binding, `start_new_session=True`, and un-monkeypatched module globals — but syntax cannot prove mocking, so still classify it semantically. The `serial` marker is the fix for *crashes and interference only*: the parallel pass also carries a 300s per-test timeout, and the serial pass carries none, so marking a merely-slow test `serial` moves the hang into the one lane that cannot bound it — make it faster or split it instead. Every other test must stay parallel-safe: `tmp_path` (not fixed `/tmp/...` paths), `monkeypatch.setenv`/`setattr` (not bare `os.environ[...] =`), no execution-order assumptions, reset any mutated module global. See `docs/DEVELOPMENT.md` "The commit gate mirrors the CI split". |

Rule: read before write. Never reconstruct `VERSION`, `pyproject.toml`
`version`, or the README badge from memory — one stale reconstruction creates
a `self_consistency` FAIL that an entire advisory cycle is then spent on.

**After a blocked reviewed commit (`commit_reviewed`) — mandatory regrouping before the next attempt:**
When a reviewed commit returns critical findings, the reflex is to patch the single
flagged finding and retry. That pattern reliably produces 5-10 blocked rounds.
The correct procedure before **every** retry:
1. List all open obligations and commit-readiness debt (`review_status` tool or the Review Continuity context section).
2. Group them by root cause — one underlying problem often generates 2-4 separately-named obligations from reviewer rephrasing.
3. Write a short plan in a progress message: one paragraph naming each root-cause group and the single code/doc change that resolves it.
4. Only then open any file and edit.

This step takes 2-3 minutes and has saved $20-50 in blocked-review cycles in practice.
The rule is already nominally in `prompts/SYSTEM.md` and `review.py::_build_critical_block_message`,
but without it appearing here as a procedural step it stays theoretical rather than reflexive.

---

## Repo Commit Checklist

Used by `commit_reviewed` for all changes to the Ouroboros repository.

| # | item | what to check | severity when FAIL |
|---|------|---------------|--------------------|
| 1 | bible_compliance | Does the diff violate any BIBLE.md principle? | critical |
| 2 | development_compliance | Does it follow DEVELOPMENT.md patterns? Check explicitly: (a) naming conventions (snake_case modules/vars, PascalCase classes, UPPER_SNAKE_CASE constants); (b) entity type rules — Gateway classes contain ONLY transport, no business logic; Tool functions are thin wrappers; (c) Python everywhere (including `tests/`/`devtools/`) and first-party `web/**/*.js` (including `web/tests/`) target ~1000 lines; exact repo-relative module debt above the 1600-line hard gate, the active `MODULE_DEBT_1500` layer (every path above 1500 lines is listed shrink-only after its one-time first-parent-authorized activation, so new/non-debt paths are capped at 1500 and >1600 additionally requires legacy `GIANT_PATHS`), exact `(path, qualname)` Python-function debt above 300 lines, the exact-current 1001-1500 band (new/re-entered paths need a nonblank rationale), and exact byte debt above 200,000 canonical UTF-8/LF bytes are checked in to `ouroboros/size_ratchet_manifest.py`, stale entries fail, methods above 150 lines are a decomposition signal, runtime-code total Python function/method count stays under `ouroboros/review.py::MAX_TOTAL_FUNCTIONS`, and more than eight parameters is a decomposition signal, not a hard gate; (d) no gratuitous abstract layers, and any SOLID/minimalism finding names an exact symbol/authority, concrete duplication or coupling, and a smaller contract-preserving alternative rather than citing diff size (P7 Minimalism) — and when the diff ADDS a surface (a new module, state file, ledger, resolver, cache, retry path, tool, endpoint, or background loop), the reviewer consults the docs/ARCHITECTURE.md map and NAMES the existing mechanism that already covers the need when one exists (name it exactly — the reuse-first duty this checklist carries for a CHANGE; the plan-review checklist judges an intention and has no such generative duty); absence of a covering mechanism may be stated in one line; (e) new LLM calls go through the shared `LLMClient`/`llm.py` layer, not ad-hoc HTTP clients; (f) cognitive artifacts (identity.md, scratchpad, task reflections, review outputs) must NOT use hardcoded `[:N]` truncation — explicit omission notes required; (g) new `get_tools()` exports use the shallow-frozen `ToolEntry` descriptor owned by `ouroboros/tools/tool_catalog.py` and re-exported by `registry.py`; first-party/scoped duplicate names fail with both origins, while extension/MCP collisions preserve the authoritative entry and emit a loud log plus visible capability omission; (h) provider independence — no change may make a core capability (agent loop, multi-model commit review, scope review, or memory/context flows) silently require a second provider or OpenRouter specifically, and every supported single direct provider (local, OpenAI, Anthropic, MiniMax, Cloud.ru, GigaChat) must keep its model AND review/scope slots self-fillable (see DEVELOPMENT.md "Provider Independence"); (i) a claimed-complete visible UI change includes vision-inspected evidence from at least one relevant real consumer flow. A screenshot file without inspection is insufficient; states/viewports/additional engines are risk-selected, mobile/WebKit are not universal, and an unavailable optional engine alone is not degradation. | critical |
| 3 | secrets_check | Are secrets, API keys, .env files, credentials present in the diff? | critical |
| 4 | code_quality | Careful code review: bugs, logic errors, crashes, regressions, race conditions, resource leaks? | critical |
| 5 | security_issues | Security vulnerabilities: injection, path traversal, secret leakage, unsafe operations? | critical |
| 6 | tests_affected | Did code logic change without corresponding test changes? (PASS if only docs/config/memory changed, or if tests already cover the new behavior.) **Critical FAIL requires all three:** (a) name a specific behavior, code path, symbol, or failure scenario that THIS diff introduces or changes; (b) explain why existing or newly staged tests do NOT catch that specific scenario; (c) the gap is concrete, not speculative. Adjacent tests in the same module or for the same feature count as coverage. Requiring an additional overlapping selector/unit/e2e test is only justified when a second distinct failure mode is named explicitly. If the only concern is "I'd feel better with one more test," that is advisory, not critical. | critical |
| 7 | architecture_doc | New module, endpoint, or data flow added but ARCHITECTURE.md not updated? (Write "Not applicable" with PASS if no architectural change.) | critical |
| 8 | version_bump | Does this commit leave VERSION unchanged, or leave release artifacts out of sync? | critical |
| 9 | changelog_and_badge | VERSION bumped but README.md badge or changelog not updated? (PASS if VERSION not bumped.) | critical |
| 10 | tool_registration | New tool function added but not exported in `get_tools()` OR missing explicit entry in `ouroboros/safety.py::TOOL_POLICY`? (PASS if no new tool.) Both surfaces are required: `get_tools()` makes the tool visible; `TOOL_POLICY` makes the per-call safety routing explicit and is guarded by the `test_tool_policy_covers_all_builtin_tools` invariant. | critical |
| 11 | context_building | New data/memory files that should appear in LLM context (context.py) but don't? | advisory |
| 12 | knowledge_index | Knowledge base topics changed but memory/knowledge/index-full.md not updated? | advisory |
| 13 | self_consistency | Does this change affect behavior described in `BIBLE.md`, `prompts/`, `docs/`, or this checklist itself? Check explicitly: (a) version in `ARCHITECTURE.md` header matches `VERSION` file; (b) tool names/descriptions in `prompts/SYSTEM.md` match tools actually exported by `get_tools()`; (c) JSONL log/memory file formats described in `ARCHITECTURE.md` match all readers/writers; (d) any behavioral change reflected in `prompts/CONSCIOUSNESS.md` if it affects background loop behavior; (e) DEVELOPMENT.md rules still accurate after the change. Severity must follow the shared `Critical surface whitelist` below — release metadata, tool schema, module map, behavioural documentation, or safety contracts are critical; commentary/prose/stylistic mismatches are advisory. | critical |
| 14 | light_external_artifacts | If tool/runtime policy changed, does light mode still allow external user deliverables via `user_files`, task-scoped `task_drive`/`artifact_store`, and process `outputs` while blocking Ouroboros repo/control-plane mutation? (The external `claude_code_edit` cwd lane retired with the tool — D10.) Do review prompts avoid recommending `runtime_data/uploads` or skill payloads as generic artifact transport? | critical |
| 15 | cross_platform | Does the diff use platform-specific APIs (`os.kill`, `os.setsid`, `os.killpg`, `os.getpgid`, `fcntl`, `msvcrt`, `signal.SIGKILL`, `signal.SIGTERM`, `subprocess` with `start_new_session`/`creationflags`, hardcoded `/` or `\\` in filesystem paths) outside of `ouroboros/platform_layer.py`? Does it import Unix-only or Windows-only modules (`fcntl`, `msvcrt`, `winreg`, `resource`) at any level without a platform guard (`sys.platform`/`IS_WINDOWS` check)? | critical |
| 16 | changelog_accuracy | Do the exact wording, test counts, and minor description details in the README Version History row match what the diff actually does? Wording drift, off-by-one test counts, minor inaccuracies in descriptive prose — these belong here, NOT in `self_consistency` or `changelog_and_badge`. This item exists so reviewers have a dedicated advisory bucket for prose-level changelog imprecision that does not affect release metadata, runtime behavior, or safety contracts. | advisory |
| 17 | gateway_parity | If the diff changes any browser-facing endpoint, WebSocket message, or frontend API call, are `ouroboros/gateway/contracts.py`, `ouroboros/gateway/router.py`, `web/modules/api_client.js`, `web/modules/api_types.js`, and `tests/test_gateway_parity.py` still aligned? Missing alignment is advisory unless it also breaks a frozen contract, safety guard, release metadata, or runtime behavior. | advisory |
| 18 | subagent_isolation | If the diff changes `schedule_subagent`, child-task queueing, task constraints, tool discovery/execution, data reads, or memory handoff, does it preserve the accepted live-subagent contract: strict `objective` + `expected_output` schema, inferred lineage/workspace/contract/deadline/resource inheritance, `local_readonly_subagent` schema and execute-time allowlist, subagent-scoped secret/control-file denial for data tools, nested readonly delegation only within configured depth/cap limits (since v6.87.7 depth bounds how deep delegation NESTS and never how strong a descendant is — since v6.87.26 an omitted lane INHERITS the parent's effective lane at every depth, an explicit lane is honored at every depth, and any child landing below what was asked for discloses it through `capability_delta` — since v6.87.28 the parent declares only `write_surface`/`model_lane`/`executor`, and model/effort/route/profile are derived by the ONE dispatch-time resolution `subagents.resolve_subagent_dispatch`, so a schedule-time surface must never mint a second answer), no arbitrary local writes/commits/review/runtime/tool-expansion/skills-lifecycle/shell (bounded media projection such as `extract_video_frames` may write derived outputs only under `artifact_store/video_frames` through a host-owned command shape), enabled external tools allowed only by owner policy and inherited resources, the subagent browser boundary (external HTTP(S) + `file://` scoped to `workspace_root` + loopback EXCEPT Ouroboros control-plane ports; private/link-local/reserved/DNS-rebind still blocked; `evaluate` JS unavailable; `vlm_query`/`analyze_screenshot` available), full task-result handoff, new/changed wait/timeout paths for cognitive work using progress-aware/re-decidable waiting rather than a fixed cutoff that discards in-flight work (P5), and tests for both allowed and blocked paths? | critical |
| 19 | evolution_durability | If the diff touches `supervisor/git_ops.py`, `launcher.py`, `server.py`, `ouroboros/preflight_runner.py`, `ouroboros/tools/review_helpers.py`, `ouroboros/tools/git.py`, tests, review gates, or evolution code, does it preserve hermetic preflight, live repo/data mutation fuses, remote-optional local commit success, and transaction/rescue evidence for interrupted self-modification? | critical |
| 20 | context_budget_ssot | If the diff changes context-size budgets/constants (`ouroboros/context_budget.py`), the context layout/manifest, a section's tier/policy, or the typed ContextFit deficit/reclaim contract: does it keep the low/max context split coherent (single SSOT + both profiles + docs + drift-guard tests in sync), preserve the tier-0 always-full core (BIBLE/SYSTEM/identity/scratchpad/knowledge-index/recent-dialogue) in EVERY mode, use a visible on-demand pointer instead of silent truncation (P1), and leave the blocking scope-reviewer >=1M floor untouched wherever scope review applies? Since v6.80.0 the owner-only `OUROBOROS_CONTEXT_MODE` ALSO decides scope-review applicability (`max`: blocking ≥1M gate; `low`: declaredly not performed with a typed skip row), so any change that widens what `low` mode implies, or that lets the AGENT reach that setting, is an immune-system change under P3 — not a context-budget tweak. (PASS with "Not applicable" if no context-budget/layout change.) | critical |
| 21 | capability_regression | Does the diff REMOVE or NARROW a previously-supported user-facing behavior or capability — a tool/flag/mode/path that worked before now errors or is gated tighter (e.g. a new `is_dir`/existence guard that blocks a legitimate create, a tightened allowlist that drops a real path, a removed fallback)? If so, is it INTENTIONAL and disclosed as a breaking/capability change in the commit message + changelog? Accidental capability removal is the failure class this item names. Ask whether a golden "from zero" test would have caught it. Severity follows the `Critical surface whitelist` below — silently removing a documented capability or a safety/release contract is critical; a deliberate, disclosed narrowing or an internal-only refactor is advisory. **Standing disclosure (v6.80.0), do not re-raise as an undisclosed removal:** the audited owner-opt-in DEGRADED ADVISORY scope review was REMOVED as a capability, together with `OUROBOROS_SCOPE_REVIEW_DEGRADED`. By owner decision the frozen gateway surface of `OUROBOROS_SCOPE_REVIEW_FLOOR` is NOT removed: the endpoint, contract field, route, merge-skip, web client and all three self-lowering guards stay, the stored value is preserved, and every write is answered with an explicit deprecation notice — the key is simply ENFORCEMENT-INERT (nothing consults it, no getter exists). Whether the P3 blocking scope review applies is decided solely by the owner-only `OUROBOROS_CONTEXT_MODE`, read as the owner-selected value through `config.get_owner_context_mode`; persistent system auto-Low and settings-time model downgrades are retired, so only the owner endpoint authors stored Low. The REPLACEMENT PATH for an install with no ≥1M-context reviewer (fully local, GigaChat-only, single small provider) is the `low` context mode: whole-repository scope review is then declaredly NOT performed, every skipped commit records a typed `scope_review status="skipped_low_context_mode"` evidence row, and the diff-reviewer triad still blocks in every mode. Since the BIBLE P3 retrieving-scope amendment, `low` is no longer the only ADMITTED replacement path: the owner may instead declare a RETRIEVING scope reviewer (reads the surface itself through read-only tools; owner-selected per scope slot; context window established by sourced Capability Evidence at ≥200K tokens; every such review records a typed durable row naming the mode; opened artifacts recorded as forensic, non-certifying evidence) as a second, owner-declared AGENTIC DELIVERY of the same authoritative review — not a degraded fallback, in BIBLE P3's own words, once its four bounds hold. That mode is IMPLEMENTED — with its fourth BIBLE bound DISCLOSED AS UNIMPLEMENTED rather than claimed: the artifacts a session opened are not recorded, because the host cannot see them and no upstream Claudexor read-event capability exists yet, so the coverage manifest states that coverage is the session's own retrieval and is not host-attested. What IS implemented: the scope fan-out delivers an `agent_session` row through the review-execution seam, `scope_review_session` gates its blocking authority on SOURCED ≥200K window evidence (stale or unsourced evidence downgrades the row to a typed `session_advisory` whose findings are advisory AND which BLOCKS the commit for want of an authoritative verdict — the same shape as a sub-floor api row; returning a non-blocking row there made the P3 gate fail OPEN on an all-retrieving panel), the ≥200K floor is REACHABLE through the same owner-capability-ack the api slot uses (the scope-slot save offers session rows their ack against the session floor, fingerprinted under the `agent_session` provider rather than a guessed one), and every such review records a typed durable row naming the mode. Two costs are accepted explicitly by the owner: in `low` the whole-repository architectural review is lost even when a genuine ≥1M reviewer IS configured (the coupling is a POLICY choice, not a technical limit), and no partial-coverage reviewer remains that could be mistaken for the gate. The replacement path is REACHABLE: during the one compatibility window `OUROBOROS_CONTEXT_MODE_AUTO_LOW=false` is an inert owner-provenance tombstone, bare env Low remains owner Max, `/api/state.context_mode_auto_low` is the frozen literal `false`, and scope-slot changes still surface the pinned reviewer's `needs_ack` through the existing owner-capability-ack flow. Also disclosed: review-pack sizing is deliberately conservative — the review reducer takes the DENSEST still-fresh witnessed density times its safety factor and never sizes below the Claude-derived 1.65 cold floor (or the independent absolute-margin bound), so a lighter measured tokenizer does NOT earn a larger review pack; the cap may loosen back toward that floor only after the supporting dense witness expires, never because lighter traffic refreshed or evicted an unsupported maximum. Main-context density is the separate liberal reducer (newest fresh exact-route witness, then newest exact-model witness, then neutral 1.0) and is not floored. A stale stored degraded key is inert (every consumer iterates `SETTINGS_DEFAULTS`) and never fails settings validation. Since v6.87.44 blocking scope authority is a property of sourced, non-stale ≥1M Capability Evidence for the slot's route — the designated default carries no name-based authority and is probed (metadata-only, rate-limited by the evidence TTL, v6.87.45) like any pin; the no-evidence 1M sentinel only SIZES prompts and never signs a blocking verdict. The floor guard also became PRECISE rather than weaker, by INVERTED polarity: reaching the key or the owner endpoint is blocked unless the whole command line is demonstrably read-only inspection (per-segment command-head allowlist), so a pure `grep OUROBOROS_SCOPE_REVIEW_FLOOR data/settings.json` read is no longer blocked while an interpreter or HTTP client naming the endpoint is refused whatever verb spelling it carries. **Standing disclosure (v6.89.0), do not re-raise as an undisclosed narrowing:** the light-mode interpreter write fence is INVERTED — an inline invocation is refused unless python's AST proves it cannot write into the repo. The narrowing is bounded to what the owner approved: a non-python inline invocation that names the repo by an ABSOLUTE path, or by a `./`- or `../`-prefixed relative one, is refused even for reading. One that does not name the repo in one of those spellings still runs, whatever it writes elsewhere — the resolved-cwd test is python-only, because the DEFAULT shell cwd is the repository and applying it to every family refused ordinary `node -e` / `ruby -e` work outright. Measured, a PLAIN relative spelling does not count as naming the repo: `EMBEDDED_RELATIVE_PATH_RE` anchors on `./`/`../`, so `node -e "…('ouroboros/safety.py')"` names the repo to a reader but not to the scan and RUNS — for a write as much as for a read. Disclosed, not closed: widening the regex would be a strengthening. `drive`, `user_files` and advanced/pro are untouched: the non-python read-vs-write refinement was rolled back to the public head, which leaves a KNOWN write hole there (`node -e "require('fs').writeFileSync('<drive>/logs/chat.jsonl','')"` runs) — disclosed, not closed, under the owner's standing direction that protection may be weakened but not strengthened. Measured, that drive hole covers the NODE and PHP families in EVERY spelling: `node`, `nodejs`, `node18`, `php` and `php8.3` all reach another task's drive path unblocked (the versioned spellings blocked at the public head and no longer do; the unversioned ones were already open). It does NOT extend to ruby or perl — both are `LIGHT_SHELL_WRITER_COMMANDS` members, so `ruby`, `ruby3.2`, `perl` and `perl5.38` are still caught by the conservative full-mention scan on the earlier writer-command branch. One narrowing does ride with the inversion and is deliberate: python inline code is now inspected on `run_command`, not only on `run_script`. Three gaps are DISCLOSED rather than detected, and must not be re-raised as findings to fix. (a) No invariant ties the AST walker to the write-indicator regex: `<mod>.open(p,"w")` (io/codecs/gzip/bz2/lzma) counts as a write to the regex while the walker reads argument 0 as the MODE and can answer "no targets", which the fence reads as a proven read; measured, that truncates a repo source file. (b) Aliased writers escape the walker's vocabulary (`import shutil as sh`, `from shutil import copy`, `getattr(os,"replace")`). (c) The fence is a light-mode convenience boundary against the agent's own mistakes, not a containment boundary against a determined writer. **Standing disclosure (Q4=A sandbox unwind, owner decision 2026-08-08), do not re-raise as an undisclosed widening or narrowing:** the argv-text blanket `GIT_VIA_SHELL_BLOCKED` default lane was deliberately replaced by the target-aware `git_shell_policy.external_workspace_git_violation` resolver for ALL non-workspace shell lanes. Mutating git is free outside the Ouroboros runtime in every runtime mode including light and direct chat; read-only git is allowed even at runtime targets (the v4.5.1/f14baf8f false-block line); only mutating git that targets the system repo / data drives (bidirectional, casefold, symlink-resolved containment) is refused, with `commit_reviewed` as the remedy; acting `self_worktree` children keep the strict read-only git policy, and the `allowed_resources.network` fence and `gh repo create/delete`/`gh auth` blocks are unchanged. Disclosed residuals, not defects to fix: git launched through a transparent wrapper (`nice`/`xargs`) or from interpreter code is not classified by the deterministic guard (the pre-flip text classifier never saw the interpreter form either; the LLM safety layer and the light post-exec system-repo dirtiness tripwire remain), and `-c` alias/config indirection is not parsed. Completing that unwind, two composition rules are also standing and disclosed: `git init <dir>` / `git clone <url> <dir>` are judged by their DESTINATION rather than the cwd (the default lane's cwd IS the system repo, so the cwd verdict was a blanket false block; the destination is the last path-shaped operand with init/clone's own value-taking flags consumed; with no destination the cwd remains the target; `--flag=<path>` retargets stay checked by the flag's documented type while non-path flag values like `-b feature/x` are never resolved as paths; relative candidates are canonicalized like absolute ones), and in external-workspace mode the runtime/secret READ guard exempts commands proven read-only git in EVERY segment (`is_readonly_git_command`) — all-or-nothing, so a compound that merely starts with git still meets the full guard, and WRITE-aware, so a read-only subcommand carrying the diff `--output=<file>` option (which truncates the file) or `--no-index` (which reads arbitrary host files) does not ride the exemption; `--output` targets are containment-judged at the FILE, and only with those rules does "the credential surface is unchanged" hold. | advisory |
| 22 | cache_friendliness | If the diff builds or reorders LLM prompt/context content (context builders, review prompt assembly, message construction in `llm.py` callers): does it keep prompt caching intact — stable governance/policy content BEFORE dynamic evidence, no dynamic values (timestamps, hashes, round counters, task ids) injected into a stable cached prefix, and no removal/breakage of existing `cache_control` markers or session/cache affinity keys? A change that silently fragments an existing cached prefix re-bills the full prompt on every repeat call. (PASS with "Not applicable" if no prompt/context assembly changed.) | advisory |
| 23 | delegated_transport | If the diff touches the delegated execution/review transport (`ouroboros/subagents.py` dispatch/route-health, `ouroboros/tools/delegate.py`, `ouroboros/delegate_custody.py`, `ouroboros/delegate_progress.py`, `ouroboros/review_execution.py` session executors, `ouroboros/gateways/claudexor.py`, `ouroboros/claudexor_daemon.py`, `ouroboros/claudexor_runtime.py`), does it preserve the delegation invariants: capability reductions reach all three destinations (durable envelope, child prompt, parent result — D4); a result counts as received only after a hash-bound read to EOF and retries replay the recorded byte-identical body (D7); an unavailable pinned route is a TYPED blocker and `auto` falls back loudly, never a silent re-route onto another model/route (D28); quota exhaustion needs POSITIVE evidence judged against the route's own model (`applies_to_models` scoping, absence = unknown = usable); delegated spend settles through custody with unknown-never-rendered-as-zero and root/parent lineage; and no vendor/harness name is ever branched on in core? (PASS with "Not applicable" if no delegated-transport surface changed.) | critical |
| 24 | perf_lifecycle | If the diff adds or changes an endpoint, poller, subscription, or timer, or reads a growing store (JSONL log, ledger, event table): does any interaction-path read scan an unbounded store per request/message/tick (a full-table read filtered in code is such a scan)? Is a subscription/observer/interval/listener added without a paired disposer? Does O(history) work run on a poll/stream path? Does a GET handler perform new steady-state durable writes outside the two named exceptions? The authoritative definitions are DEVELOPMENT.md "Invariant: Projection over replay (hot readers of growing stores)" and "Invariant: UI resources carry a disposer" — check against those, do not re-derive them here. (PASS with "Not applicable" if the diff touches none of these surfaces.) | advisory |

**Standing note for item 21 (2026-08-15), do not re-raise as a missing successor class:** the delegated-coding target class lost in the D10 migration — editing one exact non-Git installed skill payload — is restored through `delegate_start(root="skill_payload", bucket, skill_name)` (private standalone snapshot, explicit parent CAS apply, review goes stale), with a golden registry-level test pinning it.

### Severity rules

- Items 1-5 are always critical.
- Items 6-10, 14-15, 18, 19, 20, and 23 are conditionally critical: FAIL only when the condition applies.
  If the condition does not apply, write verdict PASS with a short reason
  (e.g. "Not applicable — no code logic change").
- Items 11-12, 16-17, 22, and 24 are advisory: FAIL produces a warning but does not
  block. Item 22 (`cache_friendliness`) passes with "Not applicable" when the
  diff touches no prompt/context assembly. Item 24 (`perf_lifecycle`) passes with
  "Not applicable" when the diff adds/changes no endpoint, poller, subscription,
  or timer and reads no growing store.
- Item 13 (self_consistency) is conditionally critical: FAIL only when the
  mismatch falls in the `Critical surface whitelist` below AND a concrete
  stale artifact is named (specific file, line, or symbol). If no whitelisted
  surface is affected, the finding is advisory. If no concrete staleness is
  found at all, write verdict PASS with a short reason.
- Item 16 (`changelog_accuracy`) is advisory by design: prose-level wording
  drift, off-by-one test counts, and minor descriptive inaccuracies in the
  README changelog row MUST NOT be raised as critical under `self_consistency`
  or `changelog_and_badge`. They surface here and do not block.
- Item 21 (`capability_regression`) is advisory by default but escalates to
  critical under the `Critical surface whitelist` below: a SILENT removal/narrowing
  of a documented capability or a safety/release contract is critical; a
  deliberate, disclosed narrowing or an internal-only refactor stays advisory.

### Retry convergence for tests_affected

When the previous blocker was *only* `tests_affected` and the new diff changes
*only* files under `tests/` plus release/version touchpoints (`VERSION`,
`pyproject.toml`, `README.md`, `docs/ARCHITECTURE.md`), reviewers must focus
on verifying whether the newly staged tests address the named gap — not search
for fresh gaps in unchanged code. A new critical finding on this retry round
requires a new concrete artifact, consistent with the Critical threshold rule
below: a reformulation of an earlier concern is not a new finding.

### Critical threshold rule (applies to ALL items)

Before marking any item CRITICAL you MUST be able to answer YES to ALL of:
1. I can name the **exact file, symbol, function, test, or config path** in this
   repository that makes this problem live RIGHT NOW.
2. That artifact actually appears in the diff or touched-file context I have been given
   (not just in a hypothetical future scenario or external environment).
3. The fix requires a **change to this diff** — not a follow-up task or speculative guard.

If you cannot satisfy all three, use **advisory**, not critical.

For any finding about narrative, prose, or cross-surface consistency, also apply
the `Critical surface whitelist` below (same rules for every reviewer — triad,
scope, and advisory). A mismatch outside the whitelist is advisory.

One root cause = one FAIL entry. Do NOT split one underlying problem into multiple
FAIL items that all require the same change. Do NOT hold an obligation open by
reformulating a fixed concrete issue into a broader future-risk variant — if the
named artifact is fixed, mark PASS; raise a new advisory if a broader concern remains.
Coverage is semantic, not numerical: zero or one FAIL is valid, and reviewers
must never invent findings to reach a count.

### Critical surface whitelist (binding for ALL reviewers — triad, scope, advisory)

When marking a cross-surface / self-consistency / narrative / "prose-vs-code"
mismatch as **critical**, the mismatch MUST live in one of these categories:

1. **Release metadata** — `VERSION` vs `pyproject.toml` vs README badge vs
   `docs/ARCHITECTURE.md` header vs latest git tag. Also: `VERSION` bumped
   but no README changelog row for the new version.
2. **Tool schema** — tool names, parameters, or descriptions in
   `prompts/SYSTEM.md`'s command tables that disagree with what each tool's
   `get_tools()` actually exports. Applies to user-facing CLI/tool contracts.
3. **Module map** — `docs/ARCHITECTURE.md` naming a module / endpoint /
   data file / UI page that does not exist (or the reverse: a new one was
   added and the map was not updated). This is a hard P6 (Architecture
   mirror) contract.
4. **Behavioural documentation** — a docstring, README description, or
   ARCHITECTURE section explaining what a changed tool/command actually
   **does at runtime**, where the description is factually wrong after the
   change (e.g. "sends files X, Y" when the code sends X, Y, Z). This
   matters because operators and future reviewers rely on it to use and
   audit the feature.
5. **Safety guarding** — a documented safety / permission / authorization
   contract vs. the actual guard in code (e.g. ARCHITECTURE says "panic
   kills all subprocess trees" but the implementation misses process groups).
6. **Frozen contracts (v1)** — the ABI under `ouroboros/contracts/`
   (`ToolContextProtocol`, `ToolEntryProtocol`, `SkillManifest`,
   `schema_versions`) plus the browser gateway contract in
   `ouroboros/gateway/contracts.py` (canonical HTTP/WS envelope and
   endpoint index; `ouroboros/contracts/api_v1.py` is compatibility only).
   Removing a field, renaming a TypedDict key that the runtime already
   emits, removing an endpoint token that the router still mounts, or
   breaking the `parse_skill_manifest_text` tolerance contract is critical,
   because external skills/extensions and the frontend boundary are expected
   to pin against this surface. Non-breaking *additions* are not critical.
   The regression suites are `tests/test_contracts.py` and
   `tests/test_gateway_parity.py`.

**All OTHER mismatches are advisory, not critical.** Including:

- Wording of explanatory comments that is imprecise but does not misstate
  runtime behaviour of the feature (e.g. comment says "Claude Opus 4.6"
  when the resolved model is `openai/gpt-5.5-pro`; the comment is stale but
  the runtime is fine — advisory).
- Stylistic inconsistency between changelog entries, commit-message wording
  that doesn't literally match the code in every respect, descriptive prose
  in README intro sections, "N fixes" narrative summaries, formatting of
  bullet points.
- Documentation that is merely verbose or redundant rather than wrong.

Reviewers MUST apply this whitelist before escalating any prose-level
mismatch to critical. If in doubt, advisory.

### Loop / state-machine changes

When the diff changes `ouroboros/loop.py`, task finalization semantics, checkpoint/audit rounds,
or other state-machine behavior, reviewers MUST verify adversarial paths — not only the happy path.
At minimum, check for:
- malformed or empty model output
- false task completion / premature finalization
- replay durability in logs/history
- visible anomaly path when structured output is missing or broken

A state-machine change that only passes the success-path test is incomplete.

---

## Skill Review Checklist

Used by `skill_review` to vet a single
external skill before it is allowed to execute via `skill_exec`. This uses the
shared reviewer-slot configuration (`OUROBOROS_REVIEW_MODELS`) but preserves the
existing skill-review gate semantics against a skill package in the local
checkout of `OUROBOROS_SKILLS_REPO_PATH`, not against a staged git diff.

### Transport and control skills are first-class (binding for ALL reviewers)

A reviewed, token-bound, enabled, owner-granted, owner-bound transport/control
skill — a chat bridge like Telegram today, or any future remote-control skill —
is a **first-class control surface: a legitimate full replacement for the local
UI**, not a demo or a toy. Owners may have no screen, no notebook, and no
terminal; the control skill is how they run Ouroboros. Such a skill is therefore
**expected** to:

- carry every owner command the local UI accepts, including `/panic`,
  `/restart`, `/evolve`, `/bg`, `/review`, `/status`, and free-form owner text;
- run a long-lived poller / `supervised_task` to receive inbound traffic;
- subscribe to the owner-conversation events it needs to mirror.

**Power is not a defect.** Reviewers MUST NOT raise a finding because a control
skill *can* issue owner commands, runs a long-lived poller, or observes
owner-conversation events — those are its purpose. Reviewers score the skill's
**actual safety properties** instead:

- owner/chat binding (or an equivalent access rule) so only the owner drives it;
- trustworthy source attribution (the host stamps `source=skill:<name>`);
- bounded polling / backpressure and no unbounded restart loop;
- cleanup on unload/panic so the Emergency Stop invariant always holds;
- host-token confinement (loopback only, `use_in_request()`, never logged);
- no exfiltration of secrets or owner-conversation content to unrelated parties.

A control skill FAILS only for being **unsafe** on one of those properties,
never for being **powerful**. The capability itself is already gated by the
host (token auth, fresh executable review, enablement, content-hash-bound
grants) and by core owner/chat binding (`server._process_bridge_updates`) — not
by withholding control from the skill. Items 9–12 below are scored against these
safety properties, not against the breadth of control the skill exposes.

Scope of a skill review pack:

- The skill's `SKILL.md` / `skill.json` manifest (parsed by
  `ouroboros.contracts.skill_manifest.parse_skill_manifest_text`).
- The body of the `SKILL.md` (human-readable instructions).
- **Every regular file under `<skill_dir>/`** that the subprocess could
  ``import`` / ``source`` / ``read`` at runtime (the skill runs with
  ``cwd=skill_dir`` so the reviewed/hashed surface must equal the
  runtime-reachable surface). This includes top-level helpers like
  `helper.py`, manifest-declared scripts outside `scripts/` (e.g.
  `bin/run.sh`), and manifest-declared extension entry modules (e.g.
  `plugin.py`). Hidden files that are NOT VCS/cache metadata (e.g.
  `.hidden_helper.py`) are hashed + reviewed for the same reason — a
  skill could still ``import`` them.
- The manifest's declared `permissions` list, for comparison against
  what the code actually does.
- Any declared `scheduled_tasks` entries, for comparison against the code they
  trigger and the `supervised_task` permission they require.

What is **deliberately excluded** from both the content hash and the
review pack:

- VCS / package-manager / editor scratch: `.git`, `.hg`, `.svn`,
  `.idea`, `.vscode`, `.tox`, `__pycache__`, `node_modules`, `.DS_Store`
  (silently excluded — a byte-flip in a cache file does not
  invalidate a PASS review).
- **Sensitive file shapes HARD-BLOCK the skill**: `.env*`, `.pem`,
  `.key`, `.p12`, `.pfx`, `.jks`, `.keystore`, `credentials.json`,
  `service-account.json`, `secrets.yaml`, `secrets.json`,
  `.git-credentials`, `.netrc`, `.npmrc`, `.pypirc`. (Allowlist
  reused from `ouroboros.tools.review_helpers._SENSITIVE_EXTENSIONS`
  + `_SENSITIVE_NAMES`.) The loader raises `SkillPayloadUnreadable`
  on first discovery and the skill shows up in `list_skills` with a
  non-empty `load_error` — neither reviewable nor executable until
  the operator renames or relocates the file outside the skill
  tree. Rationale: silently excluding the file would leave it
  runtime-reachable via `open('.env').read()`, so a reviewed skill
  could still exfiltrate credentials the reviewer never saw. If
  your skill legitimately ships an example config, rename it so it
  does not match this allowlist (e.g. `env.sample.txt` instead of
  `.env.example`).
- Symlinks whose targets resolve outside `skill_dir` (confinement
  guard — otherwise a symlink to `/etc/passwd` would leak into the
  review pack sent to external reviewer models).

Skill review is **text-only**: any non-UTF-8 file in the runtime-
reachable skill surface (whether a recognised loadable-binary extension
like `.so`/`.dylib`/`.pyc`/`.node`/`.wasm` or an extensionless
disguised blob) is a hard review blocker. ``_read_capped_text`` raises
``_SkillBinaryPayload`` for any such file and ``skill_review`` converts
that into ``status="pending"`` with an actionable error — never a
filename+size note that would let bytes the reviewer could not inspect
slip past the gate. The subprocess runs with ``cwd=skill_dir`` so it
could otherwise ``ctypes.CDLL('./payload')`` / ``import`` / ``require``
opaque bytes, which breaks the "review is the primary gate" invariant.
Media-carrying skills that need binary assets must fetch them on
demand from a reviewable HTTPS source rather than vendoring opaque
bytes inside the skill checkout. The text-only invariant is permanent
in v5.7.0 — there is no follow-up "binary-asset allowlist" phase
planned; a future sandbox project (out-of-process / WASM) is the
prerequisite for trusting any opaque bytes inside the skill tree.

Skills default to **disabled** and cannot be executed by `skill_exec`
until review produces a fresh executable verdict. Skill review output is persisted
to `~/Ouroboros/data/state/skills/<name>/review.json` with a content
hash so an edit to the skill invalidates the previous verdict.
`review.json`, `enabled.json`, `grants.json`, and marketplace/self-authored
provenance are skill trust/control-plane state: they are mutated only
through the review, toggle, launcher-grant, self-authored finalize,
native launcher-seed trust, and marketplace paths, not through generic
agent/browser file writes.

Native launcher-seed trust (v6.31.0) is a named, hash-pinned, audited
exception to manual first review: when the LAUNCHER itself writes a
bundled native skill payload (bootstrap seed, post-bootstrap new seed,
or version resync — all marked by `.seed-origin`), it stamps
`review.json` with `status=clean`, `reviewer_models=["repo_commit_gate"]`,
and `review_profile="native_seed"`, because those exact payload bytes
already passed the repo triad+scope commit gate. The verdict is bound to
the post-seed content hash (lifecycle control files excluded), so ANY
later edit flips it stale and non-executable exactly like an ordinary
review; removing `.seed-origin` reclassifies the skill as user-managed.
Zero-grant native seeds (no secret keys, no privileged permissions, only
tool/subprocess surface) also auto-enable — but only when no explicit
owner enable/disable choice exists yet; a version resync never overrides
an owner's disable. The verdict is additionally bound to the marker at
LOAD time: a `native_seed` review whose `.seed-origin` is gone reads back
as pending (non-executable). The owner opt-out is
`OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS=false`; the trust never extends to
clawhub/external/self-authored skills. Packaged installs bind native-seed
trust to the SHA-pinned `repo.bundle`; source-mode installs copy current
worktree bytes and therefore lack that packaged-byte provenance.
`skip_advisory_review` changes only advisory coverage: repo triad and
applicable scope review still run.

Self-authored skills carry payload-local `.self_authored.json` and
owner-state `data/state/skills/<skill>/self_authored.json` provenance,
but they do not bypass review on the agent's own initiative. `skill_review` routes them
through the same tri-model skill review as marketplace and user-managed
skills; no deterministic PASS or enablement is written automatically.
EXCEPTION (C1, v6.39; narrowed hub extension in v6.43 — owner attestation): the OWNER
may explicitly skip the EXPENSIVE LLM review for their OWN skill (`source=external` or
self-authored) or for a hash-verified official OuroborosHub payload (fresh sidecar +
live-catalog hash match, no extra runtime-reachable files) via the owner-only
`POST /api/owner/skills/<skill>/attest-review`. Native, ClawHub, and unverified
OuroborosHub payloads are not attestable. The DETERMINISTIC preflight floor still runs
(409 on failure); only the LLM phase is skipped. The result is a durable `clean` verdict
with `review_profile=owner_attested`, `reviewer_models=[owner_attestation]`, bound to
`content_hash` (a content edit stales it) and valid only while the owner-issued
`owner_attestation.json` marker is present. The AGENT can NEVER trigger this — the
marker is an owner-state file (agent-write-blocked) and the endpoint is blocked from
agent self-call on shell/CLI/browser channels (`prompts/SAFETY.md` DANGEROUS rule).
This is the only owner-issued review-bypass.
`OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS` is default-on as of v6.10.0 (the owner
may disable it), in which case a fresh executable review grants only
manifest-declared settings keys and host permissions for that exact content
hash; when disabled, key and permission grants remain explicit.

The Skills UI Repair affordance is only a task starter: it asks Ouroboros
to edit payload files and rerun `skill_review`. It must not write
trust/control-plane state directly, auto-enable a repaired skill, or
grant keys. Repair tasks carry the legacy `task_constraint.mode="skill_repair"` marker so deterministic
tool guards allow `list_skills`, payload-oriented `read_file`/`list_files`/
`write_file`/`edit_text` with `root=skill_payload`, `skill_review`, and
`skill_preflight` for cheap offline syntax/manifest/widget
validation, and block `toggle_skill`, `skill_exec`, shell/browser
indirection, extension tools, broad repo mutation, and subtask delegation
while the repair task is active.
Payload data access is scoped to the selected non-native skill under
`data/skills/external/<skill>/`, `data/skills/clawhub/<skill>/`, or
`data/skills/ouroboroshub/<skill>/`. Marketplace/official provenance
sidecars inside those payload roots (`.clawhub.json`, `.ouroboroshub.json`)
remain control-plane state and are not writable from Repair mode. User-managed
payloads accidentally left under `data/skills/native/` are blocked with an
actionable manual-upgrade error; move them to `data/skills/external/` or
reinstall them. The Repair guard still does not grant write access to true
native launcher-seeded skills.

### Output contract

Reviewers return a JSON array covering every item below (16 items total).
Each entry carries `item`, `verdict` (`PASS`/`FAIL`), `severity`
(`critical`/`advisory`), and `reason`. If one item has multiple distinct
concrete problems, reviewers may return multiple `FAIL` entries for that
same item; each distinct root cause must stay visible. If an item has no
problems, return one `PASS` entry. Do not return duplicate `PASS` entries,
and do not return `PASS` for an item that also has a `FAIL` — the concrete
`FAIL` is authoritative.

### Checklist items

| # | item | what to check | severity when FAIL |
|---|------|---------------|--------------------|
| 1 | manifest_schema | Does the manifest parse cleanly? Does `type` match the actual payload (`instruction` = no scripts/entry; `script` = at least one entry in `scripts`; `extension` = non-empty `entry`)? Is `runtime` one of `python`/`python3`/`node`/`bash`/`deno`/`ruby`/`go` for `type: script` (empty `""` is allowed ONLY for `type: instruction` since instruction skills never execute; extension entries are Python `plugin.py` modules)? Is `timeout_sec` > 0? | critical |
| 2 | permissions_honesty | Do the declared `permissions` match what the scripts actually do? Missing permission declaration for an effect the code performs is a concrete FAIL. Examples: `net` must be declared if any script uses `httpx`/`requests`/`socket`/`urllib`; `fs` must be declared if a script writes outside the skill state dir; `subprocess` must be declared if a script spawns another process. | critical |
| 3 | no_repo_mutation | Does any script attempt to write to the self-modifying Ouroboros repo (`~/Ouroboros/repo/`)? Import of `write_file`/`commit_reviewed` against the system repo, `git add`/`git commit`, or any path that starts with `OUROBOROS_REPO_DIR` / `~/Ouroboros/repo` is a concrete FAIL. Skills may only propose patches by returning artifact bundles; commits go through the first-party reviewed path. | critical |
| 4 | path_confinement | Do scripts stay inside the skill directory and the dedicated state dir (`~/Ouroboros/data/state/skills/<name>/`)? Absolute paths, `..` traversal, and writes to arbitrary user home subdirs are concrete FAIL. Reading from outside the skill dir is OK for read-only lookups (e.g. system info), write-path confinement is the strict rule. | critical |
| 5 | env_allowlist | Is `env_from_settings` a short, justified list of settings keys? Core keys in `FORBIDDEN_SKILL_SETTINGS` (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`, `CLOUDRU_FOUNDATION_MODELS_API_KEY`, `GIGACHAT_CREDENTIALS`, `GIGACHAT_PASSWORD`, `ANTHROPIC_API_KEY`, `MINIMAX_API_KEY`, `TELEGRAM_BOT_TOKEN`, `GITHUB_TOKEN`, `OUROBOROS_NETWORK_PASSWORD`) may be declared only when the skill genuinely needs that provider/token for its stated purpose; runtime forwards them only after a fresh executable review and a content-bound desktop-launcher owner grant. v5.2.2 dual-track grants: both `type: script` skills (forwarded by `_scrub_env`) and `type: extension` skills (forwarded by `PluginAPIImpl.get_settings`) are eligible; `type: instruction` skills cannot receive core keys. Mark unjustified core-key requests or non-forbidden secrets unrelated to the purpose as FAIL. An empty list is the default and always fine. | critical |
| 6 | timeout_and_output_discipline | Is `timeout_sec` reasonable for the stated workload (default 60, hard cap 300)? Do scripts print to stdout in chunks that the runtime can cap, rather than streaming unbounded output? Unbounded loops without a `break`/timeout path are a concrete FAIL. | advisory |
| 7 | extension_namespace_discipline | `type: extension` only: does the extension register its tool/route/ws-handler/ui-tab under the namespace derived from its `name` (e.g. provider-safe tool/ws names like `ext_<len>_<token>_<surface>`, route `/api/extensions/<name>/…`)? Tool and WS short names must be alphanumeric/underscore and at most 24 characters. Namespace collisions with built-in surfaces are a concrete FAIL. If the extension uses `api.send_ws_message`, are emitted event names short/provider-safe and paired with reviewed host-owned widget `subscription` components rather than arbitrary same-origin JavaScript? If the extension declares streaming UI, is it a reviewed extension route consumed by a host-owned `stream` component? If the extension owns background resources (threads, sockets, EventSource clients, subprocesses), does it register cleanup with `api.on_unload(callback)`? If the extension declares a widget render block, is it one of the host-owned schemas (`iframe`, `module`, or declarative v1: forms/actions, markdown/code, JSON/kv/table, tabs/chart, stream/subscription, progress/poll, file/gallery/media, map/calendar/kanban, group/metric/callout), with media sourced from extension routes or safe data URLs and no arbitrary same-origin JavaScript? Nested interactive group/tab children must use stable identity and one host-owned lifecycle, while `subscription.render` stays transitively passive. For non-extension skills, verdict PASS with reason "Not applicable — type != extension." | severity-driven for applicable extensions |
| 8 | widget_module_safety | **v5.7.0+. ``kind: "module"`` widgets only.** Does the extension-supplied ``widget.js`` avoid touching ``document.cookie``, ``localStorage``, ``sessionStorage``, ``window.parent`` data, or ``fetch``/``XMLHttpRequest`` URLs OUTSIDE ``/api/extensions/<skill>/``? The host fetches reviewed ``widget.js`` through ``GET /api/extensions/<skill>/module/<entry>``, embeds the source into a sandboxed ``<iframe srcdoc sandbox="allow-scripts">`` with no ``allow-same-origin``, and injects a parent-mediated ``fetch`` bridge that rejects paths outside the owning skill route prefix. Reviewers must still confirm at the source level that the script is NOT trying to escape the sandbox via arbitrary ``postMessage`` protocols, opaque-origin storage probes, or unauthorised cross-origin fetches. Acceptable interactions: ``fetch('/api/extensions/<skill>/...')`` (through the host bridge), ``window.OuroborosWidget.fetch('/api/extensions/<skill>/...')``, and host-supplied data attributes. Mark non-module widgets and non-extension skills PASS with reason "Not applicable". | severity-driven when kind=module |
| 9 | inject_chat_minimization | Does any use of the `inject_chat` permission have a narrow, user-facing transport purpose? The Host Service enforces token auth, skill-source attribution, rate limits, in-flight limits, fresh executable review, enablement, and explicit content-hash-bound grants. Reviewed chat transports may carry the same raw owner text as direct chat, including slash commands such as `/panic`, `/restart`, `/review`, `/evolve`, `/bg`, and `/status`; reviewers must evaluate whether the transport itself is authorized, attributable, bounded, and user-facing rather than treating slash-shaped text as automatically forbidden. A skill that accepts external inbound traffic must still show local defense-in-depth appropriate to its transport: owner/chat binding or an equivalent access rule, bounded polling/backpressure, and no unaudited broadcast to unrelated parties. Missing local defense-in-depth is a concrete FAIL for network transports. Mark PASS with reason "Not applicable" when `inject_chat` is not declared. | critical |
| 10 | event_subscription_minimization | Are `subscribe_event` and `subscribe_events` limited to the minimum host event topics required by the skill? `chat.outbound`, `chat.typing`, `chat.photo`, `chat.video`, and `chat.document` expose owner/agent conversation data (including delivered file bytes) and require explicit justification. Wildcards, undeclared topics, or forwarding subscribed chat content to unrelated external services are concrete FAILs. Mark PASS with reason "Not applicable" when `subscribe_event` is not declared. | critical |
| 11 | companion_process_safety | For `companion_process` / `supervised_task` skills: is every command declared as an argument list (not shell string), using an allowlisted runtime, with no writes outside `skill_dir` / `state_dir`, no unbounded restart loop, and cleanup on unload/panic? Does the process avoid inheriting secrets except through reviewed `env_from_settings` grants? Mark PASS with reason "Not applicable" when no long-lived process/task is declared — a transient `subprocess.run`/`subprocess.Popen` invocation of a build tool like `ffmpeg`, `ImageMagick`, or `git` inside a normal request handler is NOT a long-lived companion process and does not trigger this item (its safety belongs under items 4 / 6 / 13). | severity-driven when applicable |
| 12 | host_token_handling | If the skill calls the Host Service API, does it use the provided `SkillToken.use_in_request()` only at request construction sites, avoid logging/serializing tokens, and keep all host-service calls on the loopback endpoint? Printing, persisting, exfiltrating, or embedding the token into user-visible output is a concrete FAIL. Mark PASS with reason "Not applicable" when the skill does not access the Host Service API. | critical |
| 13 | error_handling | Does the skill surface actionable errors instead of swallowing exceptions, returning success on partial failure, or leaving users to inspect raw logs manually? Are retry/backoff paths bounded and purpose-specific? | advisory |
| 14 | integration_preflight | Does the skill include cheap local preflight checks for the APIs/files/runtimes it depends on before spending provider budget or starting long work? Missing preflight for fragile external integrations is an advisory FAIL. | advisory |
| 15 | bug_hunting | Are there obvious runtime bugs in reviewed code: wrong filenames, mismatched manifest script names, missing imports, impossible arguments, JSON/schema mismatches, blocking calls in async handlers, or untested happy-path assumptions? For every FAIL, cite the concrete runtime bug and state how you propose to fix it (file/symbol/change), so the author can apply the correction instead of guessing. Concrete likely runtime breakage should use `severity=critical`; latent issues, provider fragility, minor cleanup, or dead-code concerns should use `severity=advisory`. | severity-driven |
| 16 | completion_notification | For long-running or user-visible work, does the skill emit or document a completion/failure notification path (for example a host event, `events.jsonl` append, or clear stdout marker consumed by Ouroboros)? Mark PASS with reason "Not applicable" for tiny synchronous utilities. | advisory |

### Severity rules

- Skill review verdicts are enforcement-independent:
  - `clean` — no FAIL findings.
  - `warnings` — one or more advisory FAIL findings, no blocker findings.
  - `blockers` — one or more critical/blocker FAIL findings.
  - `pending` — no reliable completed review verdict.
- Enforcement maps verdicts to execution:
  - `OUROBOROS_REVIEW_ENFORCEMENT=blocking`: `clean` and `warnings` are
    executable; `blockers` are not.
  - `OUROBOROS_REVIEW_ENFORCEMENT=advisory`: `clean`, `warnings`, and
    `blockers` are executable by operator choice. This changes
    `executable_review` only; it does not rewrite the verdict, suppress
    findings, or change `skill_review_status` semantics.
  - `pending` and stale reviews are never executable.
- Review state stores findings and computes the verdict at load time. Agents
  and UI callers must use `review_gate.executable_review` / `executable_review`,
  not the raw status string, when deciding whether the skill is runnable.
- A deterministic `skill_preflight` FAIL is a structural gate failure, not an LLM
  verdict: it persists and aggregates to `pending`, which is non-executable under
  EVERY enforcement mode (advisory included) and in every readiness/execution
  caller — the strongest fail-closed outcome, stronger than an overridable blocker.
- Hard trust-boundary items are blocker findings on any FAIL regardless of
  reviewer-supplied severity: `manifest_schema`,
  `permissions_honesty`, `no_repo_mutation`, `path_confinement`,
  `env_allowlist`, `inject_chat_minimization`,
  `event_subscription_minimization`, and `host_token_handling`.
- Items 7, 8, 11, and 15 are severity-driven. A `FAIL` with
  `severity=critical` produces `blockers`; a `FAIL` with
  `severity=advisory` produces `warnings`. Reviewers MUST reserve
  critical severity for concrete dangerous or runtime-breaking cases
  that meet the Critical threshold rule below.
- Item 8 (`widget_module_safety`) applies to module widgets. Reviewers MUST
  mark it PASS with reason "Not applicable" when the extension does not
  use a module widget. This runtime rule deliberately does not rely only
  on manifest `ui_tab` detection because extensions can register module
  widgets dynamically from `plugin.py` via `PluginAPI.register_ui_tab`.
- Items 9, 10, and 12 are critical only when their corresponding capability is
  declared or used. Reviewers MUST mark them PASS with reason "Not
  applicable" for skills outside that surface.

### Critical threshold rule (applies to ALL items)

Before marking any skill item CRITICAL you MUST be able to answer YES to ALL of:
1. I can name the **exact file, symbol, function, or manifest field** inside
   the reviewed skill package that makes this problem live RIGHT NOW.
2. That artifact actually appears in the file pack or manifest I have been
   given (not a hypothetical future use the skill *might* grow into).
3. The fix requires a **change to the skill payload or manifest** — not a
   follow-up task on the host or a speculative "the author might one day
   add X" guard.

If you cannot satisfy all three, use **advisory**, not critical.

One root cause = one FAIL entry. Do NOT split one underlying problem into
multiple FAIL items that all require the same change. If the same finding
already has a documented accepted rebuttal in the prompt (see "Previously
accepted rebuttals"), do NOT re-raise it without new evidence — the rebuttal
section is binding guidance, not background reading.

### Marketplace-installed skill review (ClawHub provenance)

When a skill's directory carries a `.clawhub.json` provenance sidecar,
its source is the ClawHub marketplace (v4.50). The review pack will
also contain a `SKILL.openclaw.md` file — that is the **original**
publisher-authored manifest, preserved by the marketplace adapter
(`ouroboros/marketplace/adapter.py`) before it wrote the translated
`SKILL.md` that the runtime executes. Reviewers MUST cross-check the
two manifests as part of items 2 (`permissions_honesty`) and 5
(`env_allowlist`) without adding extra JSON checklist entries:

1. **Permissions parity** — confirm the translated `permissions` list
   captures every effect the original `metadata.openclaw.requires.bins`
   / `allowed-tools` / scripts imply. A subprocess-spawning publisher
   that translates to an empty `permissions: []` is a concrete FAIL of
   item 2 (`permissions_honesty`).
2. **Env key honesty** — denylisted/core keys from
   `metadata.openclaw.requires.env` become explicit key-grant
   requirements, not automatic environment access. If
   `env_from_settings` is non-empty and any listed key does not appear
   in the original `metadata.openclaw.requires.env`, that is a concrete
   FAIL of item 5 (`env_allowlist`) — the adapter is fabricating a
   permission the publisher never asked for.
3. **Install spec policy (v5.7.0+)** — the adapter NORMALISES
   `metadata.openclaw.install` specs into Ouroboros's isolated
   per-skill dependency lane. ``pip``/``pipx``/``uv``/``npm``/``node``
   specs land in `data/skills/<bucket>/<skill>/.ouroboros_env/` and
   are invoked with `--ignore-scripts` for npm + `--only-binary=:all:`
   for pip. Specs with global side effects (``brew``, ``apt``,
   ``cargo``, ``go``, ``download``) are translated into manual setup
   warnings instead. Reviewers should confirm the auto-installed
   packages match the skill's stated purpose; an unjustified `pip
   install <package>` for a skill that doesn't import it is a FAIL of
   item 2 (`permissions_honesty`). The adapter still rejects Node/TS
   plugin packages outright at the staging step; seeing
   ``openclaw.plugin.json`` in the file pack means the install
   pipeline should have aborted, which FAILs item 1
   (`manifest_schema`) because the skill should not have landed.
   v5.8 generalises the same readiness contract to official and local
   manifests that declare reviewed `install` / `dependencies` metadata:
   A fresh executable review installs auto specs into `.ouroboros_env`, and enable/load/exec
   paths refuse missing, failed, or stale dependency fingerprints.
4. **Plugin packages** — `openclaw.plugin.json` in the file pack means
   the publisher shipped a Node/TS plugin. The adapter refuses these,
   so seeing one in a successfully-installed skill is a contradiction
   and FAILs item 1 (`manifest_schema`).

The marketplace pipeline writes the provenance audit trail to
`data/state/skills/<name>/clawhub.json` (slug, version, sha256,
original_manifest_sha256, translated_manifest_sha256, adapter_warnings).
This file is **not** part of the review pack (it lives outside the
skill directory) but reviewers may reference its existence as
context — its absence on a `data/skills/clawhub/...` skill would be a
concrete FAIL of item 1 (the skill claims marketplace provenance
without the audit record).

### Skill review vs. repo review

These are **separate surfaces** with separate models, prompts, and state:

- Repo review (triad + scope + advisory) protects the self-modifying
  `~/Ouroboros/repo/`. Its state lives in `data/state/advisory_review.json`
  and is keyed by staged diff snapshot.
- Skill review protects the external skills repo. Its state lives in
  `data/state/skills/<name>/review.json` and is keyed by a content hash
  of the skill's manifest + payload files.

A blocked skill review must NOT create obligations, commit-readiness
debt, or any artefact visible to the repo-review pipeline — the two
surfaces are deliberately siloed so a sticky skill finding cannot
block repo commits and vice versa.

---

## Plan Review Checklist

Used by `plan_task` to review an INTENTION before the work starts — the same organ whether the
work is code, research, a deliverable, or an action in the world. Reviewers see the agent's typed
SPEC, the task objective, the evidence the agent declared (attached bounded, with every absence
named), and — for a self-modification plan — BIBLE.md and ARCHITECTURE.md in full (inline for
an api reviewer; a retrieving reviewer reads both in full with its own tools, the pack names them
as mandatory reads); every other plan gets the heading-derived navigation maps of BIBLE.md and
ARCHITECTURE.md and may request more with `need_evidence` (the host attaches it on the next cycle).
Judge only the evidence actually present; nothing missing is ever silent.

**One question: is this SPEC sufficient to START the work safely?** Not "is everything
specified" — details may be worked out while doing. The reviewer does not write a competing
plan, does not propose alternatives for their own sake, and has no finding quota. The agent
authors the plan (P0); reviewers attack it; the host counts, aggregates and enforces.

### The spec you are reviewing

```
goal · in_scope[] · non_goals[] · acceptance_claims[] (checkable "done" statements)
invariants[] (budget, deadline, safety, irreversibility, external commitments)
decisions[] {choice, rejected[], why}  ·  deferred[] {what, why_safe_to_defer}
affected_resources[] (what the work will CHANGE)  ·  evidence[] (what to look at)
```
Every element has a host-minted id (`goal`, `claim_N`, `invariant_N`, `decision_N`, `deferred_N`).
Those ids are the only valid `breaks` targets. Ids may shift between cycles when the agent
rewrites the spec — re-target `breaks` against the CURRENT ids using the Spec delta.

### The rubric (five domain-free questions + one for self-modification)

| # | item | what to check |
|---|------|---------------|
| 1 | success conditions | Are the acceptance claims checkable — could a third party tell whether each one holds? |
| 2 | load-bearing decisions | Are the decisions that are expensive to reverse explicit, with their rejected alternatives and why? |
| 3 | constraints and invariants | Are the real constraints named — budget, deadline, safety, irreversibility, commitments to others? |
| 4 | deferrals | Is anything deferred that will be expensive to change once the work has started? |
| 5 | evidence sufficiency | Is the evidence enough to judge? If not, ask for exactly what is missing (`need_evidence` with a locator) instead of inventing a gap. |
| 6 | governance (self-modification plans only) | Does the intention contradict BIBLE.md or a frozen contract? Name the principle or contract. |

### Height rule — what may block

A finding is **blocking** only if being wrong about it AFTER the work starts would invalidate work
already done, violate a declared commitment, or make an acceptance claim unverifiable — and it
MUST name the spec id it breaks. Everything else is a **note**. If missing evidence makes a claim
structurally unverifiable, that is blocking against the claim, not a `need_evidence` request.

- `blocking` — requires `breaks: <spec id>`. Without a valid id the host demotes it to a note and
  discloses the demotion.
- `note` — the agent disposes of it (accept / reject with rationale / defer) at no cost.
- `need_evidence` — a typed request `{locator, why}`. It never blocks by itself and the same
  locator cannot be asked twice on one task; the host attaches the locator on the next cycle
  (through the same evidence policy), so the agent's next envelope carries it — a new
  fingerprint, i.e. a paid cycle that actually has the evidence.

### Output

Return ONLY a JSON array of findings (optionally one code fence); for nothing to report return the
empty array followed by `NO_FINDINGS`. Each element:

```
{"id": "f1", "class": "blocking|note|need_evidence", "breaks": "<spec id>",
 "locator": "<path|url|task:id>", "summary": "...", "recommendation": "..."}
```

There is no reviewer-authored aggregate line: the HOST computes the aggregate from the findings
across all configured slots using `config.adaptive_quorum(N)`. A reviewer never emits GREEN as
authority, and prose outside the array is not parsed.

### Cycles and closure

- **GREEN** — no findings. Proceed.
- **REVIEW_REQUIRED** — notes / `need_evidence`, or a blocking finding BELOW quorum. The agent
  closes notes and `need_evidence` with one disposition call covering every finding id
  (accept / reject with rationale / defer) — no new panel, no cost. A below-quorum blocking
  finding stays OPEN whatever the disposition says: it closes only through a changed spec
  (a new fingerprint, the next paid cycle) or a reject the next paid delta cycle judges.
- **REVISE_PLAN** — blocking findings at quorum. A disposition can never close it: the agent
  either changes the spec (a new fingerprint, the next paid cycle) or rejects a blocking finding
  with a rationale that rides into that next cycle, where reviewers mark it resolved or still open.
- **DEGRADED** — no parseable quorum. Not a verdict, but the dispatched panel PAID its cycle:
  the wave records OPEN with each slot's typed failure state (code and reset time when known),
  the control line reports DEGRADED honestly, and the recorded result replays for free ONLY
  under all three conditions — an identical envelope, a NON-EMPTY recorded structural
  lane-health epoch that a fresh snapshot still matches, and an unchanged reviewer roster
  (slot ids, targets, routes, pinned profiles and EFFORTS). An empty-epoch DEGRADED wave
  (slots died at dispatch time, no structural snapshot evidence) re-dispatches a PAID panel
  on the identical envelope; so does a healed or newly dead lane or a changed roster. Only a
  wave in which no reviewer slot was physically dispatched (typed $0 skip rows only —
  pre-fan-out health skips included) stays unpaid.
  When the wave's typed rows prove the quorum STRUCTURALLY unreachable, the wave carries
  `quorum_unreachable` + the earliest reset: under blocking the finalization gate releases for
  an agent-chosen honest `blocked_with_evidence` terminal (review stays open, implementation
  stays held), waiting via a one-shot `schedule_followup` and asking the owner stay open too.

Paid cycles per task are bounded by the owner's `OUROBOROS_REVIEW_MAX_CYCLES` (default 2,
`unlimited` available). Replaying an identical envelope is free — identical including the
evidence the host attaches for reviewers' `need_evidence` requests, so a request received in the
last cycle makes the next envelope a new one. On cycle 2+ every reviewer sees
all reviewers' findings from the previous cycle, the agent's dispositions and the spec delta:
a reformulation of an earlier finding is not a new finding, and a new blocking finding must say
why it was invisible before. When the cap is spent under blocking enforcement the host holds
implementation and escalates with the typed `review_cycles_exhausted` reason; under advisory the
agent may proceed with the wave open under a loud host disclosure.

### Rules for reviewers

- Do not name files, functions or modules as a finding unless naming them is what breaks a spec id;
  a file-level observation without a `breaks` id is a note.
- Do not penalise missing tests, version bumps, changelog rows or doc updates — there is no code
  yet, and the commit gate reviews those at commit time.
- Do not require the plan to specify what can safely be decided while doing the work. "Unspecified"
  is only a finding when leaving it open is expensive to reverse.
- A plan with no file paths at all (a trip, a deck, a research question) is a first-class subject:
  the same questions apply, and repository conventions are irrelevant to it.

---

## Intent / Scope Review Checklist

Used by the Atlas-backed scope reviewer, which runs IN PARALLEL with the triad diff review.
Unlike triad reviewers who see only the diff, the scope reviewer sees touched files plus
a Generated Scope Atlas that accounts for the ENTIRE repository. Its unique advantage
is finding cross-module bugs, broken implicit contracts, and hidden regressions that
diff-only reviewers cannot see.

**Output contract (v4.34.0):** the scope reviewer returns a JSON array that covers every
item below (8 items total). PASS entries are mandatory for items with no problems and must
carry 1–2 sentences of justification naming a concrete artifact or code path that was
actually checked — a bare "PASS" or single-word reason is treated as a reviewer failure.
Multiple FAIL entries for the same item are valid when they describe distinct concrete
root causes; do not merge unrelated scope bugs into one summary. Do not emit duplicate
PASS entries, and do not emit PASS for an item that also has a FAIL. See the
`Anti pattern-lock guard` section of the scope prompt in `ouroboros/tools/scope_review.py`
for the second-pass requirement when a single FAIL is surfaced. The commit gate still
forwards only `verdict == "FAIL"` entries; the PASS rows exist so that coverage and the
reviewer's actual reasoning are auditable in `scope_raw_result`. The scope
pipeline validates this coverage contract before classifying findings: missing
required items, unexpected items, duplicate PASS rows, or PASS+FAIL for the same
item fail closed as reviewer output failures rather than being treated as a
clean response.

| # | item | what to check | severity when FAIL |
|---|------|---------------|--------------------|
| 1 | intent_alignment | Does the staged change actually fulfill the intended transformation, not merely touch related files? | critical if the incompleteness is concrete and evidenced; otherwise advisory |
| 2 | forgotten_touchpoints | Are there specific coupled files, tests, prompts, docs, configs, or sibling paths that must also change? Name the exact file(s) or symbol(s). | critical if a required touchpoint is concretely omitted; otherwise advisory |
| 3 | cross_surface_consistency | If behavior changed, are adjacent surfaces still consistent: prompts, docs, comments, tool descriptions, automation, or user-visible workflow? Apply the shared `Critical surface whitelist` — only release metadata, tool schema, module map, behavioural documentation, or safety contracts count as critical; commentary and prose mismatches are advisory. | critical if the mismatch is in a whitelisted surface AND concrete; otherwise advisory |
| 4 | regression_surface | Does wider repository context show a concrete sibling path, migration edge, or parallel flow that remains broken or incomplete after this change? | critical if it leaves a concrete broken/incomplete path; otherwise advisory |
| 5 | prompt_doc_sync | If prompts or docs are relevant to the changed behavior, are they still accurate and mutually consistent? Apply the shared `Critical surface whitelist` — behavioural documentation describing what a tool/command DOES at runtime is critical; wording/style of comments is advisory. | critical if a whitelisted prompt/doc artifact becomes false; otherwise advisory |
| 6 | architecture_fit | Does the change solve the class of problem, or is it a narrow patch that leaves the underlying pattern unresolved? | advisory |
| 7 | cross_module_bugs | Does this change break something in a different module through implicit coupling, shared state, or assumed call/return patterns? Name the exact module, symbol, or call site. | critical if a concrete cross-module breakage can be cited; otherwise advisory |
| 8 | implicit_contracts | Are there constants, data format assumptions, expected function signatures, or protocol invariants relied upon by OTHER modules that this change violates without updating those callers? Name the exact symbol or file. | critical if a concrete violated contract can be cited; otherwise advisory |

### Severity rules

- Any critical FAIL must cite a concrete file, symbol, prompt, doc, test, config, or sibling flow.
- If the reviewer cannot point to an exact touchpoint, the FAIL must be advisory, not critical.
- Scope affects only unchanged code outside the diff. The diff itself remains fully reviewable.
- For narrative / prose / cross-surface findings, apply the shared `Critical surface whitelist`
  defined in the Repo Commit Checklist section above. Only release metadata, tool schema,
  module map, behavioural documentation, and safety contracts qualify as critical. Wording
  of explanatory comments, stylistic mismatches in changelogs, and non-contractual prose
  are advisory regardless of how concrete the citation is.
