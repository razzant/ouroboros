# Pre-Commit Review Checklists

Single source of truth for all automated review checklists (Bible P7: DRY).
Loaded by `ouroboros/tools/review.py` at review time and injected into the
multi-model review prompt.

When a new reviewable concern appears, add it here — not in prompts or docs.

---

## Advisory Pre-Review Workflow

**Correct sequence (mandatory):**

```
1. Finish ALL edits first (`edit_text` / `write_file`)
2. advisory_review(commit_message="...")       ← run AFTER all edits, ONCE
3. commit_reviewed(commit_message="...")       ← run IMMEDIATELY after advisory
```

**Rules:**
- Successful worktree mutations automatically mark advisory as **stale**. This includes
  `write_file`, `edit_text`, and mutating `run_command` /
  reviewed-commit paths when they change tracked worktree state.
- Any stale advisory → must re-run advisory before commit_reviewed.
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
- Bypass (`skip_advisory_review=True`) is an **absolute** escape hatch: it short-circuits the entire commit gate (freshness + open obligations + open commit-readiness debt). Every bypass is durably audited in events.jsonl. Open obligations/debt stay visible in `review_status` (`repo_commit_ready=false`) but do NOT block the bypassed commit. Reach for it when advisory cannot run (provider outage, rate limit) or when the stale signals are known to be obsolete.

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
  still shows the debt until a successful commit clears it. `skip_advisory_review=True`
  overrides this — bypass is absolute and does not require clearing obligations/debt first.
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
| 12 | Writing or editing any JS file under `web/modules/`? | Inline styles are banned. Before staging: `grep -n "\.style\." web/modules/*.js` — any hit on `.style.display`, `.style.color`, `.style.visibility`, etc. is a REVIEW_BLOCKED waiting to happen. Use CSS classes and `classList`/`hidden` attribute instead. |
| 13 | Changing LLM output-token budgets? | Grep the whole repo for `max_tokens`, `max_completion_tokens`, `_MAX_TOKENS`, and `max_toks`. Keep `docs/ARCHITECTURE.md` §LLM output token budgets and `tests/test_max_tokens_constants.py` in sync so main-loop, VLM, summaries, compaction, skill publish, and consciousness floors cannot drift independently. |
| 14 | Changing extension loader/dispatch or isolated deps? | Native-risk extension imports and tool/route/WS handlers must stay out-of-process. Add or run regression tests where a native-risk plugin aborts during import and the host survives, plus tool/route child-dispatch tests. Do not "fix" failures by importing native-risk plugin code in `server.py`. |
| 15 | Changing `supervisor/git_ops.py`, `launcher.py`, `server.py`, `ouroboros/tools/review_helpers.py`, `ouroboros/tools/git.py`, tests, or evolution scheduling/checkpoint code? | Prove two invariants before review spend: (1) pytest/preflight cannot mutate the live repo or live `data/` (`OUROBOROS_DATA_DIR` / `OUROBOROS_SETTINGS_PATH` must be isolated, and `OUROBOROS_MANAGED_BY_LAUNCHER` must not leak into test subprocesses); (2) autonomous restart/reset cannot erase active evolution work — it must either land a reviewed local commit or preserve a rescue/transaction recovery pointer and pause/stop the campaign. |
| 16 | Changing `devtools/benchmarks/`? | Confirm it preserves official benchmark boundaries: no replacement scoring, no benchmark-specific prompt/routing hacks, no generated benchmark outputs under `repo/`, no secrets printed or committed, and no runtime-core imports from `devtools/`. Touched `devtools` files are reviewable executable operator code, even though unrelated `devtools` files use Atlas `excluded_dir` coverage-manifest entries and stay compact in broad packs. |
| 17 | Diff spawns OS processes (`subprocess.Popen` / `mp.Process` without a bounded wait)? | Route it through `ouroboros.process_custody.spawn_supervised` (or `record_process` write-through) with an explicit `scope` (`task`/`session`/`daemon`) so the orphan reaper can find it; `tests/test_process_custody.py` enforces the allowlist. |
| 18 | New/changed **test** spawns a real OS process, binds a real port, or mutates a module-level global/registry? | CI runs the suite under `pytest -n auto` (parallel). Mark such a test `@pytest.mark.serial` (or add its file to `_SERIAL_TEST_FILES` in `tests/conftest.py`) — otherwise it flakes on kill/port-reclaim timing, or crashes a worker and fails that worker's whole co-located batch as spurious failures in unrelated files. Every other test must stay parallel-safe: `tmp_path` (not fixed `/tmp/...` paths), `monkeypatch.setenv`/`setattr` (not bare `os.environ[...] =`), no execution-order assumptions, reset any mutated module global. See `docs/DEVELOPMENT.md` "Parallel CI and the `serial` marker". |

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
| 2 | development_compliance | Does it follow DEVELOPMENT.md patterns? Check explicitly: (a) naming conventions (snake_case modules/vars, PascalCase classes, UPPER_SNAKE_CASE constants); (b) entity type rules — Gateway classes contain ONLY transport, no business logic; Tool functions are thin wrappers; (c) module-size target stays near one context window (~1000 lines) with a hard fail above 1600 lines for non-grandfathered modules, method-size target stays under 150 lines with a hard fail above 300 lines, runtime-code total Python function/method count stays under the smoke hard gate defined by `ouroboros/review.py::MAX_TOTAL_FUNCTIONS` (the literal value evolves with the codebase — consult the constant rather than hardcoding the number; `devtools/` is excluded from this health gate but reviewed when touched), and functions keep `<= 8` params; (d) no gratuitous abstract layers (P7 Minimalism); (e) new LLM calls go through the shared `LLMClient`/`llm.py` layer, not ad-hoc HTTP clients; (f) cognitive artifacts (identity.md, scratchpad, task reflections, review outputs) must NOT use hardcoded `[:N]` truncation — explicit omission notes required; (g) new `get_tools()` exports follow the ToolEntry pattern in registry.py; (h) provider independence — no change may make a core capability (agent loop, multi-model commit review, scope review, or memory/context flows) silently require a second provider or OpenRouter specifically, and every supported single direct provider (local, OpenAI, Anthropic, Cloud.ru, GigaChat) must keep its model AND review/scope slots self-fillable (see DEVELOPMENT.md "Provider Independence"). | critical |
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
| 14 | light_external_artifacts | If tool/runtime policy changed, does light mode still allow external user deliverables via `user_files`, task-scoped `task_drive`/`artifact_store`, process `outputs`, and external `claude_code_edit` cwd while blocking Ouroboros repo/control-plane mutation? Do review prompts avoid recommending `runtime_data/uploads` or skill payloads as generic artifact transport? | critical |
| 15 | cross_platform | Does the diff use platform-specific APIs (`os.kill`, `os.setsid`, `os.killpg`, `os.getpgid`, `fcntl`, `msvcrt`, `signal.SIGKILL`, `signal.SIGTERM`, `subprocess` with `start_new_session`/`creationflags`, hardcoded `/` or `\\` in filesystem paths) outside of `ouroboros/platform_layer.py`? Does it import Unix-only or Windows-only modules (`fcntl`, `msvcrt`, `winreg`, `resource`) at any level without a platform guard (`sys.platform`/`IS_WINDOWS` check)? | critical |
| 16 | changelog_accuracy | Do the exact wording, test counts, and minor description details in the README Version History row match what the diff actually does? Wording drift, off-by-one test counts, minor inaccuracies in descriptive prose — these belong here, NOT in `self_consistency` or `changelog_and_badge`. This item exists so reviewers have a dedicated advisory bucket for prose-level changelog imprecision that does not affect release metadata, runtime behavior, or safety contracts. | advisory |
| 17 | gateway_parity | If the diff changes any browser-facing endpoint, WebSocket message, or frontend API call, are `ouroboros/gateway/contracts.py`, `ouroboros/gateway/router.py`, `web/modules/api_client.js`, `web/modules/api_types.js`, and `tests/test_gateway_parity.py` still aligned? Missing alignment is advisory unless it also breaks a frozen contract, safety guard, release metadata, or runtime behavior. | advisory |
| 18 | subagent_isolation | If the diff changes `schedule_subagent`, child-task queueing, task constraints, tool discovery/execution, data reads, or memory handoff, does it preserve the accepted live-subagent contract: strict `objective` + `expected_output` schema, inferred lineage/workspace/contract/deadline/resource inheritance, `local_readonly_subagent` schema and execute-time allowlist, subagent-scoped secret/control-file denial for data tools, nested readonly delegation only within configured depth/cap limits with depth>1 coerced to light, no local writes/commits/review/runtime/tool-expansion/skills-lifecycle/shell, enabled external tools allowed only by owner policy and inherited resources, the subagent browser boundary (external HTTP(S) + `file://` scoped to `workspace_root` + loopback EXCEPT Ouroboros control-plane ports; private/link-local/reserved/DNS-rebind still blocked; `evaluate` JS unavailable; `vlm_query`/`analyze_screenshot` available), full task-result handoff, new/changed wait/timeout paths for cognitive work using progress-aware/re-decidable waiting rather than a fixed cutoff that discards in-flight work (P5), and tests for both allowed and blocked paths? | critical |
| 19 | evolution_durability | If the diff touches `supervisor/git_ops.py`, `launcher.py`, `server.py`, `ouroboros/preflight_runner.py`, `ouroboros/tools/review_helpers.py`, `ouroboros/tools/git.py`, tests, review gates, or evolution code, does it preserve hermetic preflight, live repo/data mutation fuses, remote-optional local commit success, and transaction/rescue evidence for interrupted self-modification? | critical |
| 20 | context_budget_ssot | If the diff changes context-size budgets/constants (`ouroboros/context_budget.py`), the context layout/manifest, a section's tier/policy, or compaction thresholds: does it keep the low/max context split coherent (single SSOT + both profiles + docs + drift-guard tests in sync), preserve the tier-0 always-full core (BIBLE/SYSTEM/identity/scratchpad/knowledge-index/recent-dialogue) in EVERY mode, use a visible on-demand pointer instead of silent truncation (P1), and leave the blocking scope-reviewer >=1M floor untouched? (PASS with "Not applicable" if no context-budget/layout change.) | critical |
| 21 | capability_regression | Does the diff REMOVE or NARROW a previously-supported user-facing behavior or capability — a tool/flag/mode/path that worked before now errors or is gated tighter (e.g. a new `is_dir`/existence guard that blocks a legitimate create, a tightened allowlist that drops a real path, a removed fallback)? If so, is it INTENTIONAL and disclosed as a breaking/capability change in the commit message + changelog? Accidental capability removal is the failure class this item names. Ask whether a golden "from zero" test would have caught it. Severity follows the `Critical surface whitelist` below — silently removing a documented capability or a safety/release contract is critical; a deliberate, disclosed narrowing or an internal-only refactor is advisory. | advisory |

### Severity rules

- Items 1-5 are always critical.
- Items 6-10, 14-15, 18, 19, and 20 are conditionally critical: FAIL only when the condition applies.
  If the condition does not apply, write verdict PASS with a short reason
  (e.g. "Not applicable — no code logic change").
- Items 11-12 and 16-17 are advisory: FAIL produces a warning but does not block.
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
clawhub/external/self-authored skills. Honest caveat: "passed the repo
commit gate" is hash-exact for packaged installs (sha-pinned
`repo.bundle`); on a source-mode install the seed copies the current
worktree bytes, and an owner-audited `skip_advisory_review` bypass commit
could land seed bytes that skipped triad+scope — the bypass itself
remains durably audited.

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
| 5 | env_allowlist | Is `env_from_settings` a short, justified list of settings keys? Core keys in `FORBIDDEN_SKILL_SETTINGS` (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`, `CLOUDRU_FOUNDATION_MODELS_API_KEY`, `GIGACHAT_CREDENTIALS`, `GIGACHAT_PASSWORD`, `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `GITHUB_TOKEN`, `OUROBOROS_NETWORK_PASSWORD`) may be declared only when the skill genuinely needs that provider/token for its stated purpose; runtime forwards them only after a fresh executable review and a content-bound desktop-launcher owner grant. v5.2.2 dual-track grants: both `type: script` skills (forwarded by `_scrub_env`) and `type: extension` skills (forwarded by `PluginAPIImpl.get_settings`) are eligible; `type: instruction` skills cannot receive core keys. Mark unjustified core-key requests or non-forbidden secrets unrelated to the purpose as FAIL. An empty list is the default and always fine. | critical |
| 6 | timeout_and_output_discipline | Is `timeout_sec` reasonable for the stated workload (default 60, hard cap 300)? Do scripts print to stdout in chunks that the runtime can cap, rather than streaming unbounded output? Unbounded loops without a `break`/timeout path are a concrete FAIL. | advisory |
| 7 | extension_namespace_discipline | `type: extension` only: does the extension register its tool/route/ws-handler/ui-tab under the namespace derived from its `name` (e.g. provider-safe tool/ws names like `ext_<len>_<token>_<surface>`, route `/api/extensions/<name>/…`)? Tool and WS short names must be alphanumeric/underscore and at most 24 characters. Namespace collisions with built-in surfaces are a concrete FAIL. If the extension uses `api.send_ws_message`, are emitted event names short/provider-safe and paired with reviewed host-owned widget `subscription` components rather than arbitrary same-origin JavaScript? If the extension declares streaming UI, is it a reviewed extension route consumed by a host-owned `stream` component? If the extension owns background resources (threads, sockets, EventSource clients, subprocesses), does it register cleanup with `api.on_unload(callback)`? If the extension declares a widget render block, is it one of the host-owned schemas (`iframe`, `module`, or declarative v1: forms/actions, markdown/code, JSON/kv/table, tabs/chart, stream/subscription, progress/poll, file/gallery/media, **map/calendar/kanban (v5.7.0)**), with media sourced from extension routes or safe data URLs and no arbitrary same-origin JavaScript? For non-extension skills, verdict PASS with reason "Not applicable — type != extension." | severity-driven for applicable extensions |
| 8 | widget_module_safety | **v5.7.0+. ``kind: "module"`` widgets only.** Does the extension-supplied ``widget.js`` avoid touching ``document.cookie``, ``localStorage``, ``sessionStorage``, ``window.parent`` data, or ``fetch``/``XMLHttpRequest`` URLs OUTSIDE ``/api/extensions/<skill>/``? The host fetches reviewed ``widget.js`` through ``GET /api/extensions/<skill>/module/<entry>``, embeds the source into a sandboxed ``<iframe srcdoc sandbox="allow-scripts">`` with no ``allow-same-origin``, and injects a parent-mediated ``fetch`` bridge that rejects paths outside the owning skill route prefix. Reviewers must still confirm at the source level that the script is NOT trying to escape the sandbox via arbitrary ``postMessage`` protocols, opaque-origin storage probes, or unauthorised cross-origin fetches. Acceptable interactions: ``fetch('/api/extensions/<skill>/...')`` (through the host bridge), ``window.OuroborosWidget.fetch('/api/extensions/<skill>/...')``, and host-supplied data attributes. Mark non-module widgets and non-extension skills PASS with reason "Not applicable". | severity-driven when kind=module |
| 9 | inject_chat_minimization | Does any use of the `inject_chat` permission have a narrow, user-facing transport purpose? The Host Service enforces token auth, skill-source attribution, rate limits, in-flight limits, fresh executable review, enablement, and explicit content-hash-bound grants. Reviewed chat transports may carry the same raw owner text as direct chat, including slash commands such as `/panic`, `/restart`, `/review`, `/evolve`, `/bg`, and `/status`; reviewers must evaluate whether the transport itself is authorized, attributable, bounded, and user-facing rather than treating slash-shaped text as automatically forbidden. A skill that accepts external inbound traffic must still show local defense-in-depth appropriate to its transport: owner/chat binding or an equivalent access rule, bounded polling/backpressure, and no unaudited broadcast to unrelated parties. Missing local defense-in-depth is a concrete FAIL for network transports. Mark PASS with reason "Not applicable" when `inject_chat` is not declared. | critical |
| 10 | event_subscription_minimization | Are `subscribe_event` and `subscribe_events` limited to the minimum host event topics required by the skill? `chat.outbound`, `chat.typing`, `chat.photo`, and `chat.video` expose owner/agent conversation data and require explicit justification. Wildcards, undeclared topics, or forwarding subscribed chat content to unrelated external services are concrete FAILs. Mark PASS with reason "Not applicable" when `subscribe_event` is not declared. | critical |
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

Used by `plan_task` for pre-implementation design reviews, BEFORE any code is written.
Reviewers see the proposed plan, HEAD snapshots of files planned to be touched,
and an agent-selected context level: `minimal`, `localized`, `broad`, or
`constitutional`. `minimal` keeps governance docs and touched-file snapshots
but omits the generated Atlas; `localized` adds a bounded neighborhood around
planned files, `broad` is for shared contracts, and `constitutional` is
reserved for self-evolution / immune-system surfaces.

**Reviewer role is GENERATIVE, not audit.** The primary job is to contribute
ideas the implementer may not see, using the repository evidence available for
the selected context level. Finding defects in
the plan is secondary; proposing concrete alternatives, surfacing existing
surfaces that already solve the goal, and flagging subtle contract breaks the
implementer missed is primary.

### Required output structure

Reviewers must structure their response in this order:

1. **Your own approach** (1-2 sentences). State what YOU would do if this goal
   came to you with the available repository evidence: the concrete alternative path, the
   existing file/function you would reuse, or the simpler route. If after real
   effort you genuinely see no better approach, say so explicitly.
2. **`## PROPOSALS` section** (top 1-2 contributions). The highest-value thing
   you add. Each proposal should be one of:
   - An existing function/module that already solves this (named exactly).
   - A subtle contract break or shared-state interaction the plan likely missed.
   - A simpler path with less surface area that still preserves the goal.
   - A risk pattern visible from codebase history in your context.
   - A BIBLE.md alignment issue with a specific principle cited.
3. **Per-item verdicts** (PASS / RISK / FAIL), each with a detailed explanation
   and — when RISK or FAIL — a concrete fix naming the exact file/function/symbol.
4. **Final line** (exactly one of):
   - `AGGREGATE: GREEN`
   - `AGGREGATE: REVIEW_REQUIRED`
   - `AGGREGATE: REVISE_PLAN`

### Checklist items

| # | item | what to check | severity |
|---|------|---------------|----------|
| 1 | completeness | Are there files, tests, docs, prompts, configs, or sibling paths that must also change but are NOT mentioned in the plan? Name each one specifically. | FAIL if a required touchpoint is concretely missing; RISK if uncertain |
| 2 | correctness | Given the existing code, will the proposed approach actually work? Are there hidden dependencies, wrong assumptions about how existing code works, or API mismatches? Name exact functions/constants/modules at risk. | FAIL if a concrete breakage can be identified; RISK if uncertain |
| 3 | minimalism | Is there a simpler solution to the same problem with less surface area? If yes, describe the concrete alternative with the files/approach it would use. | RISK (advisory — help the implementer, not block them) |
| 4 | bible_alignment | Does the proposed approach violate any BIBLE.md principle? Check especially P5 (LLM-First — no hardcoded behavior logic), P7 (Minimalism — no gratuitous abstraction), and P2 (Meta-over-Patch — fix the class, not the instance). | FAIL if a concrete principle violation is identifiable |
| 5 | implicit_contracts | Does the plan touch a module that other modules depend on through implicit contracts — format assumptions, expected function signatures, shared constants, protocol invariants? Name the callers/dependents that would break. | FAIL if a concrete broken caller can be named; RISK if uncertain |
| 6 | testability | Is the plan testable? Are there obvious edge cases not covered by the stated test approach? Are there integration boundaries that require mocking or fixtures not mentioned? | RISK (advisory) |
| 7 | architecture_fit | Does the plan solve the class of problem or is it a narrow patch leaving the root cause unresolved? If the latter, describe what architectural change would address the root cause. | RISK (advisory) |
| 8 | forgotten_docs | If the change affects behavior described in ARCHITECTURE.md, SYSTEM.md, README.md, DEVELOPMENT.md, or BIBLE.md, is that update included in the plan? Name the specific stale artifact. | FAIL if a concrete doc/prompt becomes stale and is not mentioned |

### Aggregate signal levels (adaptive quorum)

The coordinator aggregates the configured reviewer slots (an arbitrary N,
duplicates allowed) via `config.adaptive_quorum(N)` — the same reviewer-slot
SSOT used by commit/scope/skill review: `2` for `N ≥ 3`, `N` for `N` in
`{1, 2}` (i.e. `min(2, N)`).

- **GREEN** — all reviewers PASS. Read every reviewer's `## PROPOSALS` section
  (they are the point of this call), then proceed with implementation.
- **REVIEW_REQUIRED** — one or more of: (a) a `REVISE_PLAN` count BELOW the
  adaptive quorum (minority dissent — e.g. exactly one dissent in a 2+-slot
  setup); (b) one or more RISK items were raised; (c) non-substantive
  degradation occurred (a reviewer errored, timed out, or returned an
  unparseable response, so `GREEN` cannot be confirmed). Read every reviewer's
  full response and all PROPOSALS before deciding: a single dissenting reviewer
  often sees the structural issue the others missed.
- **REVISE_PLAN** — a `REVISE_PLAN` count **at or above `adaptive_quorum(N)`**
  (2-of-N for 3+ slots, both in a 2-slot setup, and the lone reviewer in a
  1-slot setup). Quorum confirms a structural problem with the plan. Redesign
  before writing code. A single dissent in a multi-reviewer setup surfaces as
  `REVIEW_REQUIRED`, not `REVISE_PLAN`.

### Rules for reviewers

- `plan_review` does NOT block the agent — the implementer decides what to do
  with the feedback. Aggregate levels are advisory coordination, not
  enforcement.
- Name exact files, functions, symbols, or line numbers when raising FAIL/RISK.
  Generic concerns without a concrete pointer are advisory only.
- Do NOT mark RISK on `minimalism` just because you would have done it
  differently. Flag RISK only when you can name (a) fewer files touched,
  (b) fewer lines changed, or (c) reuse of a specific existing surface —
  concrete alternative, not taste.
- Do NOT penalise missing tests, `VERSION` bumps, `README.md` changelog rows,
  or `docs/ARCHITECTURE.md` updates — the plan has no code yet. Focus on design
  correctness and elegance, not commit hygiene. Commit-gate reviewers handle
  those at commit time.

Reviewers must end with exactly one of `AGGREGATE: GREEN`,
`AGGREGATE: REVIEW_REQUIRED`, or `AGGREGATE: REVISE_PLAN`.

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
