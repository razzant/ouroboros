# Review Checklists

The questions independent reviewers answer about each subject of my immune
system: an intention before work starts (Plan Review), a delivered result
(Result Review), a staged change to any repository (Change Review, with the
Ouroboros Body Layer on my own body) and a skill (Skill Review). Loaded by
`ouroboros/tools/review.py`, `plan_task`, `review_change` and `skill_review` at
review time and delivered to reviewers verbatim.

A new reviewable concern belongs here, not in prompts or docs. A change to a
governance document replaces or merges the description of the node it touches
rather than adding beside it.

**Application follows BIBLE P0/P3.** Review findings and failures are independent
facts in every mode. In Cyber Pro they inform Ouroboros and never prohibit an
action or require permission; the agent may configure its own subsequent work.
Configured enforcement remains recorded as selected, and an author decision does
not rewrite FAIL, pending, missing source or an unperformed effect as PASS.
Enforcement requirements below describe ordinary modes; Cyber applies the same
checks as advice. This product rule does not replace an external developer's
explicit work-order review obligations.

---

## Advisory Pre-Review Workflow

**Default sequence:**

```
1. Finish ALL edits first (`edit_text` / `edit_batch` / `apply_patch` / `write_file`)
2. commit_reviewed(commit_message="...", preflight_reviewer="<enabled row>")   ← the optional early look rides the commit
   or: preflight_review(reviewer="<enabled row>", commit_message="...") standalone, then commit_reviewed(commit_message="...")
```

**What runs:**
- `preflight_review` is `review_change(subject=worktree, surface=preflight,
  reviewers=[reviewer])` over the system repository: ONE enabled catalog row the
  author names — a review-pool member or not — reads the live worktree and
  informs. It never gates, its `surface=preflight` record never answers the
  commit panel, and the same unchanged worktree returns the settled record free.
  No tests run there. The row is the author's LLM judgment: name one when an
  early look is likely to add value.
- `commit_reviewed` runs, ahead of the panel: the free deterministic checks
  (size headroom as information and warnings, release metadata of the staged
  index, syntax of staged `.py` files), the tests preflight under its own policy
  (`skip_tests` is the one exemption, a documentation-only diff included; a
  managed resolution always runs it; a matching unsigned receipt of a complete run
  counts), then the named row's look, then ONE review wave in which every seat on the panel (every pool
  seat under Blocking; the pool seats Ouroboros named with a reason under
  Advisory), shown the same test execution fact, answers both parts of the brief. The commit
  records its preflight fact: `performed` with the record id, `skipped` for an
  explicit `skip_advisory_review=True`, `not_performed` when no row was named.
  `preflight_reviewer` and `skip_advisory_review=True` are opposite choices
  (`TOOL_ARG_ERROR`); an author continuation dispatches no critic and names no row.
- Finish all edits first: a worktree mutation after the look marks the recorded
  preflight stale (`review_status` reports `stale_from_edit` and its editor);
  staleness informs and gates nothing, and a look at a changed worktree is a new
  record.
- `skip_advisory_review=True` records only `preflight: skipped`; it does not
  waive obligations or debt. Independently applicable checks and tests, the panel,
  the coupling question, pre/post fingerprint and exact commit/tag binding still
  run; unresolved obligations/debt remain visible. Explain the judgment in the
  surrounding task narrative.
- **Loud advisory enforcement (BIBLE P3 bound):** the owner chooses enforcement
  outside Cyber Pro; Cyber may configure its own review and continue under P0;
  `advisory` is legitimate ONLY while every decision blocking enforcement would
  have stopped (critical findings, quorum failure, infrastructure failure,
  missing advisory provider) leaves a durable trace: a `review_advisory_override`
  event in `events.jsonl` plus the persistent `advisory_overrides_count` /
  recent-overrides fields in `review_status`. Silent advisory is forbidden.
- **Author finality remains evidence, not reviewer PASS:** the authors of a task
  result, a skill and a commit may record an explicit author disposition against the
  exact current subject after received feedback or a disclosed unavailable review under Advisory.
  Advisory and Cyber commit outcomes return before Git; explicit continuation uses
  the returned `review_reference` and author disposition, with independent current
  preflight/test/Git checks and no new critic.
  Cyber may decide to continue without prior feedback, keeping the actual missing,
  failed or pending review separate from its own decision.
  A revised author subject keeps the original critic hash/findings separately;
  author finish precedes any repeat panel. The last paid panel still permits ordinary
  author correction; N limits paid waves, not default author responses. Blocking may
  save corrections and stop, but still needs fresh reviewer authority to advance.
  An explicit unfinished stop creates neither approval nor an automatic follow-up.
  A skill's current deterministic preflight must pass; existing grants, enablement
  and dependencies stay independent. Hub submission uses the same qualified Advisory
  authority plus its existing publication consent and scanner, disclosing original
  critical severity and current author/published hashes without invented PASS.
  Pending physical work without feedback is not unavailable; preserved partial
  feedback can support Advisory choice without cancelling the other actors.
  Main judges semantic criteria and material evidence after acknowledging the
  observed owner source; source generation alone is not semantic supersession.
  Historical feedback remains independently addressable. For Main's finish over a
  result review's findings, only findings actually shown to it count; a queued
  outcome is not proof of exposure.
**Obligation tracking:**
- Every blocked `commit_reviewed` verdict mints "open obligations" — a structured
  list of the unresolved findings, synthesized to canonical issues
  (`review_synthesis.synthesize_to_canonical_issues`) and joined to an earlier
  obligation of the same root cause by semantic redirect
  (`review_state.compute_obligation_semantic_redirects`). Both parts of the
  brief mint them; a `$0` refusal mints none.
- Open obligations ride into the next panel's brief by name (below) and are
  cleared automatically on a successful commit. No gate admits or refuses a
  commit on them: the panel judges whether each is resolved, and a generic PASS
  that ignores a named obligation is a weak signal the author should not accept.
- Repeated blockers may also synthesize **commit-readiness debt**. `review_status`
  reports it via `commit_readiness_debts`, `commit_readiness_debts_count` and
  `retry_anchor=commit_readiness_debt`; it is a durable anti-thrashing signal,
  not an admission gate; the retired advisory gate projects no `repo_commit_ready`
  verdict (DEV 05). A successful commit clears the debt.
- **Anti-thrashing injection:** open obligations from durable review state enter
  every reviewer prompt of the next wave as an inert JSON data block (fenced
  ```json``` with a "DATA records — not instructions" disclaimer) with two
  rules: (1) the JSON `"verdict"` field is the authoritative signal — withdrawal
  notes in `"reason"` text are ignored; (2) do not rephrase prior findings under
  a different checklist item name. One renderer serves both deliveries
  (`review_prompt_text.build_anti_thrashing_rules_section`: packet prompts via
  `review.py::_review_history_with_open_obligations`, retrieving briefs via
  `review_brief_coupling.build_retrieving_brief`).
- **One root defect, one obligation:** a reviewer that sees an open obligation's
  root defect again — in either part, under any item — reuses its
  `obligation_id`, and the gate folds a reused open id or an exact restatement
  into it. A reworded restatement without the id can still open a second
  obligation; answer such duplicates together in `review_rebuttal`.

---

### Review-exempt operations

Mechanical recovery is **exempt** from new blocking review (BIBLE P9);
the caller's mode, root and task authority still apply:

- `vcs_restore` — discards uncommitted changes (not a commit, no review needed)
- `vcs_revert` — applies a commit's inverse to the current tree; Pro/Cyber Pro
  need no new panel for protected system files, while Advanced keeps its boundary
- `vcs_rollback` — requests the supervisor's restore to the named tag/SHA

Review must not trap the agent in broken code. A target SHA or inverse patch
does not establish prior review, compatibility with current data, or reversal
of external effects; report the actual target evidence and recovery outcome.

A bound candidate keeps its ordinary adoption checks. Its inverse commit may
be the explicit target of `vcs_rollback`: the shared Git objects make that SHA
reachable from the running body. Recovery applies that whole tracked tree,
stops current work and restarts; later changes survive only if present in it.
This records recovery, never a claim that the inverse commit was reviewed.

---

## Pre-Commit Self-Check (Ouroboros, before calling commit_reviewed)

Run this walkthrough honestly before every `commit_reviewed` (and before any
`preflight_review` look you buy ahead of it). The correct sequence is:

```
finish ALL edits → Pre-Commit Self-Check → [preflight_review] → commit_reviewed
```

This section is **not injected as a named checklist section by the review prompts** — it exists here so the agent's
pre-flight checklist lives in the same single source of truth as the review
checklists it guards. When `docs/CHECKLISTS.md` itself appears in a commit's
touched files, reviewers see it in the touched-path manifest and the staged
diff, but it is not loaded as a standalone checklist the way the Change Review
(with its Ouroboros Body Layer) or Coupling questions sections are.

| # | Check | How |
|---|-------|-----|
| 1 | Do the numbered release carriers agree with `VERSION`, using the **PEP 440 canonical form** where required? | The inventory is `ouroboros/tools/release_sync.py::VERSION_CARRIER_SPANS`; `version_carrier_desyncs()` checks it. Read current files before editing, never reconstruct versions from memory. `_normalize_pep440` maps the author-facing spelling (e.g. `4.50.0-rc.3`) to `4.50.0rc3` for `pyproject.toml` and the `uv.lock` root. `tests/test_packaging_sync.py::test_version_file_and_pyproject_are_synced` checks this relationship. |
| 2 | Which BIBLE P9 commit form is being prepared? | An ordinary commit on the serving checkout is a numbered release, including documentation or prompt changes. Bump `VERSION` and synchronize every carrier from `release_sync.VERSION_CARRIER_SPANS`; the annotated tag follows on commit. An external contribution or body-candidate commit may instead be version-neutral: every carrier stays byte-identical and no tag is created. Choose and state the form; a partial carrier change is neither. Managed-update resolution keeps the official target version; forensic snapshots and mechanical recovery retain P9 exemptions. |
| 3 | New or changed logic → does an existing or newly staged test assert on the specific scenario it introduces? | Name the scenario in plain words and the test that distinguishes it; extend that test, write a new one only when none does, and retire assertions the change supersedes in the same diff. A pin of wording, source text, call presence or AST shape does not cover behaviour: review owns meaning. "Tests exist for the module" is not "tests cover this behaviour". |
| 4 | Shared log / memory / replay format changed? | Grep every reader and writer first. JSONL logs (`events.jsonl`, `task_reflections.jsonl`, replay indexes), durable state files (`advisory_review.json`, `review_continuations/*.json`), and memory note/history pairs (a knowledge note / its `knowledge_history.jsonl` records) must stay coherent across every consumer. |
| 5 | New validation guard, input filter, or edge-case check? | Before the first commit attempt, name three concrete ways it could break: wrong bounds, legitimate inputs it silently blocks, platform-specific edge cases. If you cannot name three, think longer. One honest minute here is cheaper than one reviewer round. |
| 6 | New tool added? | `get_tools()` exports it, its schema description says WHEN to choose it (each profile receives its visible schema set every round, so the schema is the SSOT of the per-tool contract; `prompts/SYSTEM.md` is the cross-tool selection policy and mentions a tool only when the change alters that policy, never as a catalog entry; mechanism documentation lives in ARCHITECTURE/DEVELOPMENT), the handler signature matches the declared schema, and (if it mutates repo state) it is routed through the reviewed commit path rather than ad-hoc `run_command`. Also give it one explicit safety policy, its `ToolEntry.safety_policy` or an `ouroboros/safety.py::TOOL_POLICY` entry (`skip` for trusted built-ins, `check` for opaque or outward-facing ones) — `test_tool_policy_covers_all_builtin_tools` fails otherwise, and a tool without one pays a light-model LLM call per invocation. |
| 7 | Tests green before first `commit_reviewed`? | Run `pytest -x` on the narrowest relevant target(s) you can name before the first `preflight_review` / `commit_reviewed` attempt. Size gates live in the opt-in `size_ratchet` lane; ordinary local surfaces only warn. Manual/tag CI checks the manifest and parent transition, but cannot certify per-change book growth without a contribution base. Supply `OURO_SIZE_RATCHET_BASE_REF` for that comparison. When a size warning appears — or a new `.py` file lands under `ouroboros/` or `supervisor/` — run `pytest tests/ -m size_ratchet` and `scripts/regenerate_size_ratchet.py` locally to preview and fix what official CI would reject. A red test suite before the first commit attempt has caused repeated $2-5 blocked-review cycles. |
| 8 | Adding a `README.md` version row? | BIBLE.md P9 hard cap: ≤ 2 major, ≤ 5 minor, ≤ 5 patch visible entries. Categories are mutually exclusive: major = `X.0.0` (minor=0, patch=0); minor = `X.Y.0` (patch=0, Y≠0); patch = all other `X.Y.Z` (Z≠0). Count existing rows in the category you are adding to. Easy check: `run_command(["python", "-c", "import sys; from ouroboros.tools.release_sync import check_history_limit; warns=check_history_limit(open('README.md').read()); print(warns or 'OK')"])` — if it prints warnings, trim the oldest row in the over-limit category **in the same edit** before committing. |
| 9 | Changing any of `build.sh`, `build_linux.sh`, `build_windows.ps1`, `Dockerfile`, or `ouroboros/tools/browser.py`? | Cross-surface doc sync is mandatory. Check ALL of: `README.md` Install section (Linux native-lib caveat), `README.md` Build section (per-platform instructions), `docs/ARCHITECTURE.md` browser tools paragraph, WebKit/mobile verification notes, and inline comments in the touched build script. Any one of these being stale has blocked review twice. Verify before staging. |
| 10 | Changing `ouroboros/tools/commit_gate.py`? | Coupled surfaces that MUST be updated atomically in the same commit: (a) `tools/preflight_review.py` tool descriptions for `preflight_review` and `review_status` and its `_next_step` strings; (b) the `commit_reviewed` schema in `tools/git.py` (`preflight_reviewer`, `skip_advisory_review`, `reviewers`/`reason` through `compose_commit_panel`) and the preflight-and-tests gate it drives; (c) `docs/DEVELOPMENT.md` Review & Commit Protocol section; (d) the `prompts/SYSTEM.md` Self-Modification section IF the commit-gate rule it states changed. Missing any one has blocked review. |
| 11 | Changing VERSION + pyproject.toml? | Ordering matters: (1) write `VERSION`, `pyproject.toml`, `uv.lock`, `web/package.json` and `web/modules/api_types.js` first; (2) then write the `README.md` badge + changelog row + download links, both install pages' download links, and the `docs/ARCHITECTURE.md` header; (3) then run `pytest`. Never interleave — updating README before VERSION means `test_version_in_readme` will catch a stale badge. |
| 12 | Writing or editing any JS file under `web/modules/`? | New or changed static inline visual properties are blocked: inspect the diff for added/changed `style=""` markup and `.style.<property>` assignments, and use CSS classes/tokens plus `classList`/`hidden` instead. Unchanged legacy hits are debt, not a blocker. A dynamic measured value, or a renderer's count of what it rendered, may update a narrowly named CSS custom property when that is the actual runtime data flow. |
| 13 | Changing LLM output-token budgets? | Grep the whole repo for `max_tokens`, `max_completion_tokens`, `_MAX_TOKENS`, and `max_toks`. Keep [LLM output token budgets](architecture/07-configuration.md#llm-output-token-budgets) and `tests/test_max_tokens_constants.py` in sync so main-loop, VLM, summaries, compaction, skill publish, and consciousness floors cannot drift independently. |
| 14 | Changing extension loader/dispatch or isolated deps? | Native-risk extension imports and tool/route/WS handlers must stay out-of-process. Add or run regression tests where a native-risk plugin aborts during import and the host survives, plus tool/route child-dispatch tests. Do not "fix" failures by importing native-risk plugin code in `server.py`. |
| 15 | Changing `supervisor/git_ops.py`, `launcher.py`, `server.py`, `ouroboros/tools/review_helpers.py`, `ouroboros/tools/git.py`, tests, or evolution scheduling/checkpoint code? | Prove two invariants before review spend: (1) pytest/preflight cannot mutate the live repo or live `data/` (`OUROBOROS_DATA_DIR` / `OUROBOROS_SETTINGS_PATH` must be isolated, and `OUROBOROS_MANAGED_BY_LAUNCHER` must not leak into test subprocesses); (2) autonomous restart/reset cannot erase active evolution work — it must either land a reviewed local commit or preserve a rescue/transaction recovery pointer and pause/stop the campaign. |
| 16 | Changing `devtools/`? | Keep operator tools isolated from runtime and package discovery: no runtime, web, or release imports from `devtools/`; no generated outputs under `repo/` or live `data/`; no secrets printed or committed. A touched devtool is reviewable executable code even when unrelated devtools stay compact in broad review packs. Domain-specific launch, routing, and methodology guidance stays with the relevant devtool. |
| 17 | Diff spawns OS processes (`subprocess.Popen` / `mp.Process` without a bounded wait)? | Route it through `ouroboros.process_custody.spawn_supervised` (or `record_process` write-through) with an explicit `scope` (`task`/`session`/`daemon`) so the orphan reaper can find it; `tests/test_process_custody.py` enforces the allowlist. |
| 18 | New/changed **test** spawns a real OS process, binds a real port, or mutates a module-level global/registry? | CI **and the hermetic commit gate** (`preflight_runner.py`) both use a parallel `-m "not serial" -n auto` pass plus a serial pass. Mark it `@pytest.mark.serial` (or add its file to `_SERIAL_TEST_FILES` in `tests/conftest.py`) only for a shared resource it cannot isolate — a fixed port, a real daemon, a process-global its fixture cannot restore; an owned child, a private root or port 0 is parallel-safe. Otherwise it flakes on kill/port-reclaim timing, or crashes a worker, which the gate reports as a `PARALLEL_WORKER_CRASH` **hard block** (never retried, never a flake). Syntax (`subprocess`, sockets, `start_new_session=True`, module globals) is only a screen; classify semantically. `serial` fixes crashes and interference only: the serial pass has the same 300 s per-test timeout, so a merely slow test is made faster or split instead. Every other test stays parallel-safe (`docs/DEVELOPMENT.md` "Parallel CI and the `serial` marker"). |

Rule: read before write. Never reconstruct `VERSION`, `pyproject.toml`
`version`, or the README badge from memory — one stale reconstruction creates
a `self_consistency` FAIL that an entire review cycle is then spent on.

**Before retrying after review findings (`commit_reviewed`, `skill_review`):**
The recorded review state — the verdict when one was reached and the individual
findings, including the partial findings of a pending review; for commits also
the open obligations and commit-readiness debt shown by `review_status` and the
Review Continuity context section — stays recorded until a later review or
successful commit resolves it; the author's response rewrites neither it nor the
selected enforcement. Before the next attempt the author owes an outcome, not a
procedure: consider the open findings together against the evidence and accepted
goal rather than patching one visible symptom. Repair what the requirement still
needs, or replace or remove a chosen means within the applicable task authority,
with evidence that the requirement still holds. This does not make a true
historical finding false or permit dropping an owner- or parent-fixed requirement
or method. Rebut unsupported findings or their current applicability with reasons
(`review_rebuttal`), preserving the original record and anything unresolved.
How to inspect, group, order and explain that work, and whether to seek
more feedback first, is the author's judgment for the case (BIBLE P13). Whether
the next attempt is replayed or refused for free, rejoins unresolved review
work, or dispatches a paid review is decided by the gate's recorded replay
eligibility, custody, budget and configured cycle limit (`docs/DEVELOPMENT.md`
Review & Commit Protocol), not by this checklist: an eligible verdict on
unchanged material under the same review contract is not re-reviewed without a
genuinely new rebuttal, while an infrastructure outcome such as a missed
reviewer quorum neither replays nor lapses a verdict. If attempts stop
converging, reconsider the approach instead of repeating it. The shared retry
note (`review_prompt_text.py::build_self_verification_template`, in the commit
block message and the skill review block) and the open-obligation/debt branches
of `review_status` `next_step` state this same duty.

---

## Change Review Checklist

Used by `commit_reviewed` and by the `review_change` capability for every
staged change Ouroboros reviews — in its own body or in any other repository
it works on. This section is the universal core: its items hold for any
codebase. When the subject is the Ouroboros body — the system repository, a
worktree or copy of it, or a candidate whose upstream is this install's
configured update source or the install's own fork, established by
`ouroboros/review_body_fact.py` from git facts and the install manifest — the
`Ouroboros Body Layer` section below is appended and its items continue this
numbering. For any other subject this section is the whole rule set: the
subject's own documents are evidence, not law, and the constitution of
Ouroboros binds Ouroboros as the author, never the subject repository.

| # | item | what to check | severity when FAIL |
|---|------|---------------|--------------------|
| 1 | secrets_check | Are secrets, API keys, .env files, credentials present in the diff? | critical |
| 2 | code_quality | Careful code review: bugs, logic errors, crashes, regressions, race conditions, resource leaks? | critical |
| 3 | security_issues | Security vulnerabilities an outside input can reach: injection from model, tool, web or skill input; path traversal out of a confinement the code promises; secret leakage to a third party? Defenses against the owner or the subject's own process are not security findings. | critical |
| 4 | tests_affected | Did code logic change without corresponding test changes? (PASS if only docs/config/memory changed, or if tests already cover the new behavior.) **Critical FAIL requires all three:** (a) name a specific behavior, code path, symbol, or failure scenario that THIS diff introduces or changes; (b) explain why existing or newly staged tests do NOT catch that specific scenario; (c) the gap is concrete, not speculative. Existing tests in any module or layer count when they exercise the changed behavior and would catch the identified failure; module or feature proximity alone is not coverage. Requiring an additional overlapping selector/unit/e2e test is only justified when a second distinct failure mode is named explicitly. Wanting one more test is not a finding. Surplus direction, advisory: name a newly staged test that only pins wording, source text or shape, or duplicates an existing oracle, with the test to extend; retiring such a test is not a weakening. Public contracts, machine-read book shape and architecture boundaries stay pinned. | critical |
| 5 | architecture_doc | Did the change REPLACE the current description of every node it touched in the subject's architecture / engineering documents (map rows, flows, invariants)? PASS when a new module, endpoint or data flow gains its one-line row and touched descriptions read as current truth; PASS with "Not applicable" when nothing architectural changed or the subject keeps no such document. A removed or relocated claim names its surviving owner or evidence of retirement; registry tests prove membership, not preservation of meaning. FAIL when the edit only appended a version stamp, a decision codename, a "used to / previously" narrative, or a second statement of a fact already owned elsewhere in the docs (a present-tense map, one home per fact). | critical |
| 6 | cross_platform | Does the diff use platform-specific APIs (`os.kill`, `os.setsid`, `os.killpg`, `os.getpgid`, `fcntl`, `msvcrt`, `signal.SIGKILL`, `signal.SIGTERM`, `subprocess` with `start_new_session`/`creationflags`, hardcoded `/` or `\\` in filesystem paths) outside the subject's platform abstraction layer, when it has one? Does it import Unix-only or Windows-only modules (`fcntl`, `msvcrt`, `winreg`, `resource`) at any level without a platform guard (a `sys.platform` check or the subject's equivalent)? | critical |
| 7 | capability_regression | Does the diff REMOVE or NARROW a previously-supported user-facing behavior or capability — a tool/flag/mode/path that worked before now errors or is gated tighter (e.g. a new `is_dir`/existence guard that blocks a legitimate create, a tightened allowlist that drops a real path, a removed fallback)? If so, is it INTENTIONAL and disclosed as a breaking/capability change in the commit message + changelog? Accidental capability removal is the failure class this item names. Ask whether a golden "from zero" test would have caught it. **Guard-change trigger (executable requirements, not an essay):** ADDING or TIGHTENING a guard, filter, allowlist, or deny rule IS a capability change and fires this item. For such a diff the reviewer must verify two things: (a) the diff names and runs a positive test, existing or newly staged, that walks a legitimate flow THROUGH the new guard and proves it still succeeds — a negative "it blocks X" test alone is insufficient (a gate can pass its own probe while breaking every real run); (b) the diff or its disclosure NAMES THE SURVIVING POSITIVE PATH — the concrete actor and flow that still work after the change. A guard change without passing surviving-path evidence is a capability-regression finding, not a safety improvement; a missing *new* test is not. A guard keyed on model or provider identity (id, prefix, family, provider name) always fires this item; its positive test uses an identity that the diff's own literals do not contain, and a test that proves only refusal is insufficient. **Owner acceptance:** a narrowing counts as OWNER-ACCEPTED only when the subject's owner actually decided it (the owner's words or answer for this work, or a standing disclosure the subject keeps for this purpose) or it lies within authority the owner had already delegated for this work; the plan council's answers are the reviewers' view, not the owner's consent. Owner acceptance makes the finding advisory (disclosed, non-blocking). Intent wording, a commit-message disclosure, or a changelog row alone is disclosure, NOT acceptance. This names who accepts a narrowing; it adds no approval step. Severity follows the `Critical surface whitelist` below — silently removing a documented user-facing capability or a release contract is critical; removing or loosening an internal guard, proof or re-check is advisory and needs one line naming the residual and its cost measured on a real install at today's history size; an owner-accepted narrowing or an internal-only refactor is advisory. | advisory |
| 8 | changelog_accuracy | Do the exact wording, test counts, and minor description details in the subject's changelog or release-notes entry match what the diff actually does? Wording drift, off-by-one test counts, minor inaccuracies in descriptive prose — these belong here, NOT in a release-metadata or consistency item. This item exists so reviewers have a dedicated advisory bucket for prose-level changelog imprecision that does not affect release metadata, runtime behavior, or safety contracts. (PASS with "Not applicable" when the subject keeps no changelog.) | advisory |
| 9 | perf_lifecycle | If the diff changes data readers or an endpoint, poller, subscription, timer, startup, shutdown or other batch operation: evaluate the total read cost as history, object count and project size grow, including nested repeats, cold caches and contention for shared resources. Where that growth can materially hurt responsiveness, is there proportionate evidence at representative scale, and was redundant work or within-operation reuse considered before adding infrastructure? This remains advisory, with no universal time limit, mandatory heavy benchmark or new approval gate. Does any interaction-path read scan an unbounded store per request/message/tick (a full-table read filtered in code is such a scan)? Is a subscription/observer/interval/listener added without a paired disposer? Does each observed or non-terminal state name which live producer can still change it and which fact ends the observation (an unknown past is a fact on a terminal, not an open state)? Does O(history) work run on a poll/stream path? Does a GET handler perform new steady-state durable writes? Where the subject documents these invariants, check against its definitions rather than re-deriving them here. (PASS with "Not applicable" if the diff touches none of these surfaces.) | advisory |

### Severity rules

- Items 1-3 are always critical.
- Items 4-6 are conditionally critical: FAIL only when the condition applies.
  If the condition does not apply, write verdict PASS with a short reason
  (e.g. "Not applicable — no code logic change").
- Items 7-9 are advisory: FAIL produces a warning but does not block. Item 9
  (`perf_lifecycle`) passes with "Not applicable" when the diff changes no data
  reader, startup/shutdown or other batch operation, endpoint, poller,
  subscription or timer.
- Item 8 (`changelog_accuracy`) is advisory by design: prose-level wording
  drift, off-by-one test counts, and minor descriptive inaccuracies in a
  changelog entry MUST NOT be raised as critical under a release-metadata or
  consistency item. They surface here and do not block.
- Item 7 (`capability_regression`) is advisory by default but escalates to
  critical under the `Critical surface whitelist` below: a SILENT removal/narrowing
  of a documented user-facing capability or a release contract is critical;
  removing or loosening an internal guard, proof or re-check stays advisory with
  one line naming the residual and its measured cost (item 7); an owner-accepted narrowing (an actual owner
  decision or authority the owner already delegated — see item 7; plan council
  answers are not one) or an internal-only refactor stays advisory. Disclosure
  alone (commit message, changelog) is not acceptance.
- The `Ouroboros Body Layer` sets the severity of its own items and names the
  body's addresses for items 5-9. Its editorial calibration of item 5
  applies only to the body; other core severities remain unchanged.

### Retry convergence for tests_affected

When the previous blocker was *only* `tests_affected` and the new diff changes
*only* test files plus the subject's release/version touchpoints (the
`Ouroboros Body Layer` names the body's), reviewers must focus on verifying
whether the newly staged tests address the named gap — not search for fresh
gaps in unchanged code. A new critical finding on this retry round requires a
new concrete artifact, consistent with the Critical threshold rule below: a
reformulation of an earlier concern is not a new finding.

### Critical threshold rule (applies to ALL items)

Before marking any item CRITICAL you MUST be able to answer YES to ALL of:
1. I can name the **exact file, symbol, function, test, or config path** in the
   subject repository that makes this problem live RIGHT NOW.
2. That artifact actually appears in the diff or touched-file context I have been given
   (not just in a hypothetical future scenario or external environment).
3. The fix requires a **change to this diff** — not a follow-up task or speculative guard.
4. If my proposed fix ADDS or TIGHTENS a guard, restriction, or removal: I can
   **name the surviving positive path** — the concrete legitimate flow that still
   works after that fix. (A fix that narrows nothing answers YES automatically.)
5. I can name **who or what triggers it on a supported install** — a user flow, a
   crash or restart, a concurrent writer the subject itself runs, a platform it
   ships on. A fault that needs someone who already holds the power (the owner,
   or the subject's own process editing, planting or restoring files under its
   own data root) is advisory, and disclosure is a valid fix. Ordinary use of a
   supported capability — installing an extension through the subject's public
   extension API, choosing a supported provider or setting — is a supported-install
   trigger, not such a power: the extension need not ship in the subject repository.

If you cannot satisfy all five, use **advisory**, not critical.

The threshold binds proposed REMEDIES as much as findings: a recommended fix
that deletes or narrows a capability is itself a capability change and must meet
the same concreteness bar as the finding it answers — exact artifact, live
problem, named surviving positive path; a remedy that adds a check, re-read,
proof or refusal states what it costs on the path it sits on at today's history
size, or what work it refuses. A remedy that cannot meet the bar of its
own finding is advisory, and repeating it across rounds or reviewers confers no
additional authority.

For any finding about narrative, prose, or cross-surface consistency, also apply
the `Critical surface whitelist` below (same rules for every reviewer — both
parts of the brief, and a preflight). A mismatch outside the whitelist is advisory.

One root cause = one FAIL entry, across both parts of the brief. Do NOT split one
underlying problem into multiple FAIL items that all require the same change. Do NOT
hold an obligation open by reformulating a fixed concrete issue into a broader
future-risk variant — if the named artifact is fixed, mark PASS; raise a new
advisory if a broader concern remains. Coverage is semantic, not numerical: zero or
one FAIL is valid, and reviewers must never invent findings to reach a count.

### Critical surface whitelist (binding for ALL reviewers — every pool seat, both parts, and a preflight)

When marking a cross-surface / self-consistency / narrative / "prose-vs-code"
mismatch as **critical**, the mismatch MUST live in one of these categories:

1. **Release metadata** — the subject's version carriers disagreeing with each
   other (version file vs package metadata vs lockfile vs badge vs documented
   header vs the latest tag once it exists), or a version bumped without its
   changelog entry.
2. **Public API / schema** — a published interface's declared shape (a tool
   schema, an HTTP/WS envelope, an exported type, a CLI contract) that
   disagrees with its implementation, or a name the subject's own operating
   instructions use that the implementation does not export. The declared
   schema is the SSOT of a contract; a document that omits a surface is not a
   mismatch.
3. **Behavioural documentation** — a docstring, README description, or
   architecture section explaining what a changed tool/command actually
   **does at runtime**, where the description is factually wrong after the
   change (e.g. "sends files X, Y" when the code sends X, Y, Z). This
   matters because operators and future reviewers rely on it to use and
   audit the feature.
4. **Safety contracts of the project** — a documented emergency-stop, secret
   or permission contract vs. the actual guard in code (e.g. the docs say
   "stop kills all subprocess trees" but the implementation misses process
   groups), and a frozen or pinned compatibility contract the subject promises
   to external consumers: removing a field, renaming a key the runtime already
   emits, or removing an endpoint that is still mounted is critical;
   non-breaking *additions* are not.

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
mismatch to critical. If in doubt, advisory. The `Ouroboros Body Layer` names
the body's concrete carriers for each category.

---

## Ouroboros Body Layer

Appended to the `Change Review Checklist` when the subject is the Ouroboros
body — the system repository, a worktree or copy of it, or a candidate whose
upstream is this install's configured update source or the install's own fork
(`ouroboros/review_body_fact.py`; the fact and the way it was established are
recorded with every review). `commit_reviewed` on the body always carries both
sections; the items continue the core numbering, and the `Shared Contract
Ownership` section is delivered beside them.

### Body addresses for core items

- Item 5 (`architecture_doc`): the documents are `docs/ARCHITECTURE.md` /
  `docs/DEVELOPMENT.md` and their owning physical chapters; the rule is P6
  (present-tense map) and P7 (one home per fact) — see DEVELOPMENT.md
  "Documentation contract". Missing map nodes and false runtime descriptions keep
  their existing critical criteria. Omitting internal mechanism is not a mismatch:
  the books are a map, with detail in the named module. For the body, editorial
  excess, duplication and historical prose under item 5 are advisory. Correct the
  owning text when fixing a body documentation or prompt finding; consolidate
  necessary limitations there rather than accumulating repeated caveats.
- Item 6 (`cross_platform`): the platform abstraction layer is
  `ouroboros/platform_layer.py`; the guard is `sys.platform` / `IS_WINDOWS`.
- Item 7 (`capability_regression`): **standing disclosures for this item live
  in `docs/CHECKLISTS_ARCHIVE.md`** (owner-accepted removals/narrowings and
  standing notes); they remain binding on every reviewer — consult that file
  before raising a removal/narrowing finding on a surface it covers, and do not
  re-raise anything recorded there. Owner acceptance is the owner's actual
  decision for this work or authority already delegated for it; plan council
  answers are not one.
- Item 8 (`changelog_accuracy`): the entry is the README Version History row;
  the buckets it must NOT leak into are `self_consistency` (item 14) and
  `version_bump` (item 12).
- Item 9 (`perf_lifecycle`): the authoritative definitions are DEVELOPMENT.md
  "Invariant: Projection over replay (hot readers of growing stores)" and
  "Invariant: UI resources carry a disposer" — check against those, do not
  re-derive them here; a GET handler performing new steady-state durable writes
  outside that passive-read contract is a finding. For an embedded or
  framed UI surface, also use DEVELOPMENT.md "Invariant: Embedded surfaces
  declare geometry and refresh semantics" for host geometry/overflow, teardown,
  retry/error, and real-consumer visual evidence.
- Retry convergence for `tests_affected`: the body's release/version
  touchpoints are `VERSION`, `pyproject.toml`, `README.md`,
  `docs/ARCHITECTURE.md`.

| # | item | what to check | severity when FAIL |
|---|------|---------------|--------------------|
| 10 | bible_compliance | Does the diff violate any BIBLE.md principle? | critical |
| 11 | development_compliance | Does it follow DEVELOPMENT.md patterns? Check explicitly: (a) naming conventions (snake_case modules/vars, PascalCase classes, UPPER_SNAKE_CASE constants); (b) entity type rules and dependency direction — Gateway classes contain ONLY transport, no business logic; Tool functions are thin wrappers; provider- and transport-specific decisions never flow back into core policy (judged by the `Shared Contract Ownership` section delivered with this checklist); (c) module and function size follow DEVELOPMENT.md "Module Size & Complexity" (official CI enforces the gates, so a stale or growing size-debt entry is review debt to flag, not a local commit block); (d) one owner per decision — a local test of the diff that packet seats apply too: nothing the diff adds (state, branch, layer or wrapper) decides again what an existing owner decides; a derived view stays a view, never extra truth; new durable state needs an invariant that existing facts cannot reconstruct. A finding names the exact symbol, the owner or fact it duplicates and the smaller contract-preserving alternative; diff size, line count and file count alone are not findings (P7). For an added surface (module, state file, ledger, resolver, cache, retry path, tool, endpoint or background loop) name the existing mechanism `docs/ARCHITECTURE.md` maps for the need, or state its absence in one line; this judges a CHANGE, not a plan. A second home in another module is one finding: a seat asked Part 2 reports it under coupling `architecture_fit`, a packet seat here. Tracked material needs a continuing purpose (DEVELOPMENT.md "Documentation contract"): retire finished campaign machinery instead of keeping it with presence-only tests; tests retire under DEVELOPMENT.md "Retiring tests" (a verdict line each; wording pins by class, a structural scan of a live rule rewritten or kept, merges with a named survivor); keep book-shape, public-contract and architectural-boundary checks; (e) new LLM calls go through the shared `LLMClient`/`llm.py` layer, not ad-hoc HTTP clients; (f) cognitive artifacts follow BIBLE.md P1's record/view contract: retain exact sources before replacing a body, identify partial views and their delivered ranges with resolvable sources, and preserve operative decisions, provenance and the required resident core; a partial view is disclosure, not evidence of complete review or authority; (g) new `get_tools()` exports follow the ToolEntry pattern in registry.py; (h) provider independence — no core capability (agent loop, multi-model commit review, the coupling question, memory/context flows) silently requires a second provider or OpenRouter, and every supported single direct provider keeps its model AND review-pool rows self-fillable (DEVELOPMENT.md "Provider Independence"); (i) a claimed-complete visible UI change carries vision-inspected evidence from a relevant real consumer flow opened in the changed state with realistic data; an uninspected screenshot, or one whose data hides the change, is insufficient; a change to an owner-chosen look is shown beside that mockup without awaiting a second owner approval, and a reviewer unable to open the images says so; states, viewports and engines are risk-selected, mobile/WebKit are not universal, and an unavailable optional engine alone is not degradation. | critical |
| 12 | version_bump | Release carriers: an ordinary commit is a numbered release (P9) — is `VERSION` bumped, does every other carrier agree with it (the file carriers `release_sync.version_carrier_desyncs()` checks, the `docs/ARCHITECTURE.md` header included) and does README carry the badge and the changelog row for the new version? A version-neutral contribution — an external contribution commit (P9 / CONTRIBUTING.md; the maintainer's integration commit bumps) or a commit prepared in a body candidate that P9 lets the installation adopt as it is — instead keeps EVERY carrier byte-identical and takes no tag. Authorized managed-update resolution keeps the selected official release metadata without assigning a new local version. A partial carrier change fails; these exceptions waive no other item. | critical |
| 13 | tool_registration | New tool not exported in `get_tools()`, OR without exactly one explicit safety policy (`ToolEntry.safety_policy` or `ouroboros/safety.py::TOOL_POLICY`, guarded by `test_tool_policy_covers_all_builtin_tools`), OR a new or lowered `skip` it does not earn? (PASS if no new tool or policy change.) `skip` fits a trusted built-in whose effects host guards already fence; opaque or outward-facing tools keep `check`. | critical |
| 14 | self_consistency | Does this change affect behavior described in `BIBLE.md`, `prompts/`, `docs/`, or this checklist itself? Check explicitly: (a) every tool name a prompt (`prompts/*.md`) mentions exists in `get_tools()` and means the same thing there — completeness is NOT required, the schemas are the catalog; a concrete prompt contradiction with BIBLE, another prompt or a book follows the shared critical threshold and surface whitelist; a prompt that restates mechanism, a tool's own description, a structurally enforced gate or a rule stated elsewhere is advisory, and a prompt that gains a sentence loses an equivalent one; a keyword, regex, branch, threshold or counter that stands in for judgment in a prompt violates P5 (DEVELOPMENT.md "LLM-first affordances"); and when the diff touches the code that assembles a prompt (`context.py`, review packet/template builders, advisory, consciousness, plan packet, loop nudges) the injected prompt text and that code still agree; (b) JSONL log/memory file formats described in `ARCHITECTURE.md` match all readers/writers; (c) any behavioral change reflected in the wake template `prompts/CONSCIOUSNESS.md` if it affects wakes; (d) DEVELOPMENT.md rules still accurate after the change. Severity must follow the shared `Critical surface whitelist` with the body carriers below — release metadata, tool schema, module map, behavioural documentation, safety guarding or frozen contracts are critical; commentary/prose/stylistic mismatches are advisory. | critical |
| 15 | light_external_artifacts | If tool/runtime policy changed, does light mode still allow external user deliverables via `user_files`, task-scoped `task_drive`/`artifact_store`, and process `outputs` while blocking Ouroboros repo/control-plane mutation? Do review prompts avoid recommending `runtime_data/uploads` or skill payloads as generic artifact transport? | critical |
| 16 | subagent_isolation | If the diff changes delegation — `schedule_subagent`, child queueing or constraints, tool discovery/execution, reads, memory or chronicle handoff, waits — does every child still hold exactly the authority its assignment grants, on every surface? Check against ARCHITECTURE §6 "Delegated subagents" and "Tool capability and execution" rather than re-deriving them: (a) the model-visible schema stays `subagent_id` + `objective` + `expected_output` with no route selector; lineage, workspace, contract, deadline and resources are inherited, never widened; the selected row's snapshotted route is the one that starts, or a definite typed refusal ends the child at $0 (ambiguous start evidence wakes the model instead) — never a substitute; (b) a read-only child reads what its parent can read on every surface (files, queries, media, local browser) and writes only host-owned derived outputs (e.g. `extract_video_frames` under `artifact_store/video_frames`) — no commits, review, runtime or skill-lifecycle changes, shell or tool expansion; nested delegation stays within the configured depth and cap, and depth never strengthens an actor; an investigation session works in an evidence-only copy without apply authority, and readonly, Observe and Presence ceilings still narrow it; ordinary external tools follow owner policy, and an acting child gets the real registered catalog under its constraint; (c) the browser boundary keeps external HTTP(S), parent-equivalent `file://` reads and loopback except Ouroboros control endpoints, admits private origins only through host-established `resource_policy.allowed_origins` (an unidentified matching endpoint is never a foreign service), checks navigation, actions, subresources and every redirect hop, still refuses metadata, link-local and reserved targets, keeps `evaluate` off for a read-only child and on an acting child's current page, and keeps vision tools available; (d) start inputs match the child's role with every omission named (§6 "Selected first-input sources"); a child or nanny writes only its own signed chronicle drafts, and accepting them stays with the integrating mind (§6 chronicle "Writers"); the full task result returns to the parent; (e) waits on cognitive work stay progress-aware and re-decidable, never a fixed cutoff that discards in-flight work (P5); (f) tests cover an allowed and a blocked path. | critical |
| 17 | evolution_durability | If the diff can change how self-modification is tested, committed or interrupted (preflight, commit, restart and reset paths, evolution scheduling, tests): do tests and preflight never mutate the live repository or live `data/`, does a local commit succeed without a remote, and does interrupted self-modification end in a reviewed commit or a preserved rescue/transaction pointer? | critical |
| 18 | context_budget_ssot | If the diff changes context budgets, the context layout or a section's tier, or the ContextFit deficit/reclaim contract: is there still one budget SSOT that both context profiles, the docs and the drift-guard tests read; does the always-full core (BIBLE, SYSTEM, identity, scratchpad, marks) stay in view in every mode, with the knowledge index resident whole or as its outline (every active note named, people's summaries kept) and the sealed story resident; does only the physical window turn content into addresses — people's words last (ARCHITECTURE `memory_floor.py`, §6 "Context fitting") — with every degraded span named by a visible on-demand pointer (P1), and does an owner-selected smaller target bound host material only; and does the coupling question stay independent of the context mode (coupling the two is an immune-system change under P3)? Outside Cyber Pro the owner selects the context mode. (PASS with "Not applicable" if no context-budget/layout change.) | critical |
| 19 | delegated_transport | If the diff touches delegated execution or review transport (ARCHITECTURE §6 "Delegated subagents" and the session executors of `review_execution.py`, matched by role, not by a file list): (a) does the exact selected `subagent_id` route start with custody-durable requested→effective evidence, or return a typed refusal — never a host-chosen substitute, any fallback being a new explicit LLM choice? (b) does a capability reduction reach the durable envelope, the child prompt and the parent result? (c) is a result received only after a hash-bound read to EOF, and does a retry replay the recorded bytes? (d) is quota exhaustion judged only on positive evidence for the route's own model (unknown stays usable), and does spend settle through custody with root/parent lineage, unknown never shown as zero? (e) does the work order carry the complete assignment and host authority once, with no arbitrary size cap, compulsory file/question transport or duplicate objective, do plan normalization and reviewer inputs keep full content and tail-sensitive identity, and do real route limits keep the original input, cause and execution state? (f) does a legacy partial request stay recoverable byte-for-byte (exact selector, digest and range validation, durable source coverage) and refuse apply until complete, with reject still available? No vendor or harness name is branched on in core; source availability never proves reading. (PASS with "Not applicable" if no delegated-transport surface changed.) | critical |
| 20 | source_completeness | Partial views and source truth: if the diff adds or changes a bounded view (projection, summary, omission marker, status count) or review evidence (plan, change review, preflight, result review): (a) does every cut say so in band — a marker naming what was cut and a reference to the existing file, blob, record or range the deciding actor can still open after the real merge, promotion and cleanup? A new store, copy or sidecar is not required when such a reference exists; (b) does a consumer that can authorize PASS, a destructive rewrite or replacement of a full contract read the named sources first, or abstain with the existing typed incomplete/degraded outcome? A marker or host claim alone is never sufficient; (c) do review records keep exact artifact bodies and source selectors, candidate SHA, reviewer route continuity and every omission, with continuity, transport and coverage discrepancies as typed facts beside them, so a failed or truncated delivery stays DEGRADED/NOT_RUN rather than PASS? File-read coverage is diagnostic (BIBLE P3), never a reason to discard a received verdict or drop its reviewer from quorum; no disclosure discards, blanks or relabels a body or its cause. | critical when applicable |
| 21 | canonical_memory_fork | If the diff touches Project/fork/execution roots, summaries, memory, or GC, does it preserve one canonical identity and distinguish authority/biography from execution-local state? Are referenced canonical artifacts promoted or retained before a child/root is collected, with missing legacy bytes represented as gaps? | critical when applicable |
| 22 | context_building | Context inputs: does a new data/memory file that should reach LLM context appear there (`context.py`), and does a changed knowledge-base topic update `memory/knowledge/index-full.md`? | advisory |
| 23 | gateway_parity | If the diff changes any browser-facing endpoint, WebSocket message, or frontend API call, are `ouroboros/gateway/contracts.py`, `ouroboros/gateway/router.py`, `web/modules/api_client.js`, `web/modules/api_types.js`, and `tests/test_gateway_parity.py` still aligned? Missing alignment is advisory unless it also breaks a frozen contract, safety guard, release metadata, or runtime behavior. | advisory |
| 24 | cache_friendliness | If the diff builds or reorders LLM prompt/context content (context builders, review prompt assembly, message construction in `llm.py` callers): does it keep prompt caching intact — stable governance/policy content BEFORE dynamic evidence, no dynamic values (timestamps, hashes, round counters, task ids) injected into a stable cached prefix, no ordinary step rewriting sent history (DEVELOPMENT 06 "Cache-friendliness invariant"), and no removal/breakage of existing `cache_control` markers or session/cache affinity keys? A change that fragments or rewrites an existing cached prefix loses the reuse of everything after the provider's last stored boundary before it, on every repeat call; measurements of the route, not the size or position of the edit, set that cost. (PASS with "Not applicable" if no prompt/context assembly changed.) | advisory |
| 25 | display_identity_replay | If the diff changes routing, steering, task cards, or history replay, does it preserve the event-time human `Project › Task` presentation snapshot in both live and replay paths while keeping opaque IDs as internal/debug facts? | advisory when applicable |
| 26 | web_design_system | If the diff touches `web/` (modules, stylesheets, `index.html`, onboarding assets) or backend producers/delivery of owner-facing UI messages: when the subject carries frames of the change, take the Result Review's first look at them; then, does it conform to `docs/DESIGN.md`, first to its design bible (does each new or changed element take one of its roles: the plain room, an object or tray, a capsule tone, one height per role with states in colour; is a deliberate departure named in the commit message, not introduced silently?), then to its type scale, foreground roles, status pairs and spacing tokens, and to the engineering rules in DEVELOPMENT.md "Design System" (no new inline visual styles, values live in `web/ui.css` tokens, shared components over page-local copies)? When the diff ADDS a UI control, chip, card, dialog, or visual pattern, does it reuse an existing shared frontend primitive (name it — the registry is ARCHITECTURE.md §3 "Navigation and shared UI contracts") or state in one line why none covers the need — and, because reuse alone does not show that two controls agree, does a control for an owner intent that already has a door come from that door (DESIGN "References and actions"), with the neighbouring rows of `docs/inventories/UI_CONTROL_TEXT_INVENTORY.md` named when control text is added or changed? If the diff edits a DESIGN rule, does the new wording cite the owner's decision it rests on, and does the PR show the old and new sentence with a shot of the state it describes? Wording that goes beyond what the owner chose is a question for the owner, not a rule. A status or notice it adds names what ends it (DESIGN §4); a floating surface carries its own paint, and a shell change opens every surface that borrowed its backing in both themes (DESIGN principle 2). Check host/model authorship, explicit role/type and their preservation through live delivery and history against DESIGN's "Chat authorship and System rows"; Python producers are part of this surface. (PASS with "Not applicable" if neither frontend nor backend UI-message behavior changes.) | advisory |
| 27 | size_cap_paydown | If the diff was shaped by a size limit (module cap, function gate, byte debt, or a reference book's size — added book text is paid by shortening the same book, the pairwise contribution-base rule): was it paid by making the code simpler where it lives? A reduction is honest when the reader holds fewer concepts, branches and state afterwards — dead code and duplicates removed, an existing owner reused; the same logic in fewer lines or tighter formatting is not one. An extraction is honest only when each resulting file has one responsibility you can state in one sentence without naming the other file, and the files have no mutual imports. A cap-driven bucket, a one-caller passthrough, or bytes bought by deleting contract-bearing comments, docstrings, messages or tests is a defect to report, not a paydown (BIBLE P7 «first simplify what exists»; DEVELOPMENT.md "Paying down a size cap"). | advisory when applicable |
| 28 | capability_placement | Does the change add to the body a capability or a control that a setting, a declarative contribution or a skill could carry? If it opens a seam, does the seam ship with its first consumer, and does core consume the contract without naming a skill or vendor (DEVELOPMENT "Capability placement")? An honest answer is advice for the author, never a block. | advisory |
| 29 | copy_mechanism | Does the change add or change a mechanism that copies a project, workspace or store (a checkout, clone, snapshot, retained view or scratch tree)? Does it name its job, the fault it prevents, its size on a small install and its remover (DEVELOPMENT "Live subagents")? A new kind of copy is a plan-review question. An honest answer is advice for the author, never a block. | advisory |

### Body severity rules

- Items 10-11 are always critical.
- Items 12-13 and 15-21 are conditionally critical: FAIL only when the condition
  applies. If the condition does not apply, write verdict PASS with a short
  reason (e.g. "Not applicable — no new tool").
- Items 22-27 are advisory: FAIL produces a warning but does not block. Item 24
  (`cache_friendliness`) passes with "Not applicable" when the diff touches no
  prompt/context assembly.
- Item 14 (self_consistency) is conditionally critical: FAIL only when the
  mismatch falls in the `Critical surface whitelist` (with the body carriers
  below), AND a
  concrete stale artifact is named (specific file, line, or symbol). If no
  whitelisted surface is affected, the finding is advisory. If no concrete
  staleness is found at all, write verdict PASS with a short reason.

### Critical surface whitelist — the body's carriers

The four core categories, with the concrete surfaces of the Ouroboros body:

1. **Release metadata** — `VERSION` vs `pyproject.toml` vs the root version in
   `uv.lock` vs `web/package.json` vs `web/modules/api_types.js`
   `GATEWAY_CONTRACT_VERSION` vs README badge vs the named direct-download links
   (README and both install pages) vs `docs/ARCHITECTURE.md` header — the file
   carriers `release_sync.version_carrier_desyncs()` checks — and vs the latest
   git tag once it exists. Also: `VERSION` bumped but no README changelog row
   for the new version.
2. **Tool schema** (public API / schema) — a tool's `get_tools()` schema (name,
   parameters, description) that disagrees with its handler, or a tool
   name/argument that `prompts/SYSTEM.md` or the wake template
   `prompts/CONSCIOUSNESS.md` names but `get_tools()` does not export. The
   schema is the SSOT of a tool's contract; a prompt that omits a tool is not a
   mismatch. Applies to user-facing CLI/tool contracts.
3. **Module map** (behavioural documentation) — `docs/ARCHITECTURE.md` naming a
   module / endpoint / data file / UI page that does not exist (or the reverse:
   a new one was added and the map was not updated). This is a hard P6
   (Architecture mirror) contract. Behavioural documentation elsewhere
   (docstrings, README, ARCHITECTURE sections) is critical on the core terms.
4. **Safety guarding** (safety contracts) — a documented Emergency Stop, secret
   or owner-granted permission contract vs. the actual guard in code (e.g.
   ARCHITECTURE says "panic kills all subprocess trees" but the implementation
   misses process groups).
5. **Frozen contracts (v1)** (safety contracts) — the ABI under
   `ouroboros/contracts/` (`ToolContextProtocol`, `ToolEntryProtocol`,
   `SkillManifest`, `schema_versions`) plus the browser gateway contract in
   `ouroboros/gateway/contracts.py` (canonical HTTP/WS envelope and endpoint
   index; the `ouroboros/contracts/api_v1.py` compatibility re-export was
   removed in ABI 7.0 — an import of the old name fails at load time). Removing
   a field, renaming a TypedDict key that the runtime already emits, removing
   an endpoint token that the router still mounts, or breaking the
   `parse_skill_manifest_text` tolerance contract is critical, because
   external skills/extensions and the frontend boundary are expected to pin
   against this surface. Non-breaking *additions* are not critical. The
   regression suites are `tests/test_contracts.py` and
   `tests/test_gateway_parity.py`.

### Loop / state-machine changes

When the diff changes `ouroboros/loop.py`, task finalization semantics, checkpoint/audit rounds,
or other state-machine behavior, reviewers MUST verify failure paths — not only the happy path.
At minimum, check for:
- malformed or empty model output
- false task completion / premature finalization
- replay durability in logs/history
- visible anomaly path when structured output is missing or broken

A state-machine change that only passes the success-path test is incomplete.

---

## Shared Contract Ownership

Applies to every review of this repository's own code — the commit panel's
change part (Ouroboros Body Layer item 11(b)), its coupling part (item 5
`architecture_fit`), a preflight and deep self-review — and is delivered to
each of them beside its own
checklist. Plan and skill review judge other subjects and do not apply it, and
a change review of another repository (the core layer alone) does not carry
it. It is the review
form of the dependency-direction rule in DEVELOPMENT.md "Naming and
boundaries": provider- and transport-specific decisions never flow back into
core policy.

**The defect.** A shared module makes one producer's private grammar — its
argument names, keys inside an untyped payload, its id syntax, a branch on its
name — the condition for, or the meaning of, a fact that an existing contract
of this repository treats as shared across producers. For example, a shared
delivery collector that confirms a delivery only for a tool carrying one
transport's addressing argument, while its contract serves every transport;
that transport's own adapter knowing its argument is correct.

**Severity.** Critical only when the finding cites that existing contract or
ownership boundary — a schema, an interface, a document addressed to
producers, or the producer/consumer code that already carries the fact as
shared — shows how this change violates it, and meets the ordinary critical
threshold; otherwise advisory. Disclosing the narrow choice (a docstring, the
commit message, a note that the join is one transport's shape) does not lower
a shown violation. A preference for a more general design, a producer that
does not exist yet, or a vendor name appearing in shared code is advisory at
most.

**Outside this rule, not exempt from others.** A producer's own leaf, a closed
vocabulary that defines a rule at its owner, a field a schema declares, and a
dated wire-fact row in DEVELOPMENT.md "External facts: unknown is not no" do
not violate it by themselves, and they remain subject to every other item.

**Judgment, not a scan.** No keyword, name list or lint decides this finding
or its severity; the cited contract and the reviewer's reading do.

---

## Skill Review Checklist

Used by `skill_review` to vet a single
external skill before it is allowed to execute via `skill_exec`. This uses the
shared review pool (the `review_eligible` rows of `OUROBOROS_SUBAGENTS`; the
retired comma-list model/route settings are not read) and follows every
row's configured delivery, while preserving the existing gate semantics against a skill package in the local
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
- Any declared `scheduled_tasks` suggestions, for comparison against the
  behaviour they describe; a declaration installs nothing and needs no
  `supervised_task` permission.

What is **deliberately excluded** from both the content hash and the
review pack:

- VCS / package-manager / editor scratch: `.git`, `.hg`, `.svn`,
  `.idea`, `.vscode`, `.tox`, `__pycache__`, `node_modules`, `.DS_Store`
  (silently excluded — a byte-flip in a cache file does not
  invalidate a PASS review).
- **Exact credential file names HARD-BLOCK the skill**: the shared
  `_SENSITIVE_NAMES` policy (`.env` and its explicitly listed runtime
  variants, `credentials.json`, `service-account.json`, `secrets.yaml`,
  `secrets.json`, `id_rsa` and the other SSH private-key names,
  `.git-credentials`, `.netrc`, `.npmrc`, `.pypirc`) plus the `.env` tail in
  `_SENSITIVE_EXTENSIONS`, which covers spellings such as `prod.env`.
  (Reused from `ouroboros.tools.review_helpers._SENSITIVE_EXTENSIONS`
  + `_SENSITIVE_NAMES`.) The loader raises `SkillPayloadUnreadable`
  on first discovery and the skill shows up in `list_skills` with a
  non-empty `load_error` — neither reviewable nor executable until
  the operator renames or relocates the file outside the skill
  tree. Rationale: silently excluding the file would leave it
  runtime-reachable via `open('.env').read()`, so a reviewed skill
  could still exfiltrate credentials the reviewer never saw.
  `.pem`, `.key`, `.p12`, `.pfx`, `.jks`, `.keystore`, `.kdbx`, `.gpg` and
  `.asc` are NOT credential shapes: they are ordinary reviewed payload and are
  SHOWN to the reviewer. A name rule never bought what the rationale above
  promises anyway, since a real `.env` renamed to `config.txt` walks straight
  through it, while a public certificate or a release signature used to break
  the whole skill (owner decision, 2026-09-11).
  `.env.example` is ordinary reviewed payload: its bytes remain in the
  freshness hash and the existing publication scan.
- Symlinks whose targets resolve outside `skill_dir` (confinement
  guard — otherwise a symlink to `/etc/passwd` would leak into the
  review pack sent to external reviewer models).

Binary payloads are judged by CONTENT, not by filename (#447 X4). A
file whose bytes start with a loader magic (ELF, PE, Mach-O incl. the
fat/byte-swapped variants, the host interpreter's exact `.pyc`
magic) is a hard review blocker regardless of its name or extension:
``executable_magic_kind`` raises ``SkillBinaryPayload`` and
``skill_review`` converts that into ``status="pending"`` with an
actionable error. WebAssembly (`\x00asm`) is no longer a loader-magic
hard blocker: a `.wasm` file is admitted as a content-hash-bound binary
descriptor (below) because it executes only inside the sandboxed widget
frame, never natively in the host process — the reviewer does not read
the WebAssembly bytes (the review pack never inlines them), so judge the
widget JavaScript that instantiates
it and the module's provenance instead. Any OTHER non-UTF-8 file is
neither blocked nor inlined: the review pack carries a typed
``{path,size,mime_from_name,sha256}`` descriptor instead of raw bytes.
Judge such a descriptor on the merits: `mime_from_name` is guessed
from the FILENAME and is not a content attestation; `size`/`sha256`
are exact. An opaque blob is only dangerous through loader code the
review CAN see (``ctypes.CDLL`` / ``zipimport`` / ``require`` /
``sys.path`` insertion) — flag any code path that loads or executes a
descriptor-only file, and treat an unexplained opaque blob in an
executable position as a finding (the deliberate exception: a `.wasm`
module that widget JavaScript instantiates inside the browser-sandboxed
widget frame, which never runs in the host process). ``skill_exec``
independently refuses to execute a declared script that is not valid
UTF-8 text, so a blob renamed into the scripts list cannot run.
Media-carrying skills should still prefer fetching binary assets on
demand from a reviewable HTTPS source over vendoring opaque bytes. This
content-judged posture is the current owner decision, revisitable
explicitly; a future sandbox project (out-of-process / WASM) remains
the prerequisite for trusting opaque bytes as EXECUTABLE payload inside
the skill tree — the WASM there is a host-side WebAssembly runtime, a
separate future topic from the browser-sandboxed `.wasm` above.

A skill runs through `skill_exec` only when its exact current bytes are accepted
(by a fresh executable review verdict or by one of the exact-byte bases below) and
it is enabled (by the owner, or by an install's turn-on-when-ready request, never
over an explicit Off). Skill review output is persisted
to `~/Ouroboros/data/state/skills/<name>/review.json` with a content
hash so an edit to the skill invalidates the previous verdict.
`review.json`, `enabled.json`, `grants.json`, `install.json`, `owner_edit.json`,
`update_backup.json` and marketplace/self-authored provenance are skill
trust/control-plane state:
they are mutated only through the review, toggle, launcher-grant, self-authored
finalize, installer and owner-edit paths, not through generic agent/browser
file writes.

A source the owner trusts is the named, digest-pinned exception to a first
review: the installer (built-ins included) writes the host-owned
`install.json` with basis `trusted` and the exact release digest, and the
gate admits those bytes only (`trusted_source`); any later edit leaves the
release and is unreviewed again. No synthetic `clean` verdict is written;
the deterministic preflight still gates the bytes an install or update lands (a
built-in re-seeded with an app update carries the app's own release); a source's
trust never rewrites earlier installs. A skill installed from a source that
reviews each skill keeps blocker findings non-executable even under Advisory
until the owner skips the review. Enablement of such installs is the
owner's choice or the install's turn-on-when-ready request, never over an
explicit Off. The Built in row of the source list sets built-in trust;
`OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS=false` only seeds that row while no list is saved.
A built-in whose manifest `os` excludes this host is neither offered nor newly
installed; an installed copy is still re-seeded and discovery hides it, except a
Windows Unix-remote choice: the re-seed records `legacy_remote_platform=win32` in
`.seed-origin` for an enabled, formerly unrestricted Unix copy whose ON was not
recorded as automatic. Any executable acceptance admits its declared packages:
Enable installs or repairs them; startup retries installed on/requested skills.
Packages alone never require a paid review. Packaged installs take built-ins from the SHA-pinned
`repo.bundle`; source mode copies worktree bytes, without that provenance.
`skip_advisory_review` records only the commit's skipped preflight: the repo
review wave, both parts, still runs.

Self-authored skills carry payload-local `.self_authored.json` and
owner-state `data/state/skills/<skill>/self_authored.json` provenance,
but outside Cyber Pro they do not bypass review on the agent's own initiative. `skill_review` routes them
through the same review-pool skill review as marketplace and user-managed
skills; no deterministic PASS or enablement is written automatically.
EXCEPTION (owner attestation, "Skip review" / "Turn on anyway"): the OWNER may skip the
LLM review for the exact bytes of a skill from any origin via the owner-only
`POST /api/owner/skills/<skill>/attest-review`. The DETERMINISTIC preflight floor still runs
(409 on failure); only the LLM phase is skipped. The result is a durable `clean` verdict
with `review_profile=owner_attested`, `reviewer_models=[owner_attestation]`, bound to
`content_hash` (a content edit stales it) and valid only while the owner-issued
`owner_attestation.json` marker is present. The marker records owner attestation; outside Cyber Pro it is protected from
generic agent writes and raw shell/CLI/browser endpoint self-calls. Cyber
configuration does not turn a self-authored decision into an owner attestation. An ordinary task may carry an already
expressed owner instruction through `skill_owner_action`; the shared host owner
checks the actual caller, resolved member source, selected revision and source
eligibility before invoking this same attestation. The model interprets the
owner's words; a formatted reference alone grants nothing. Without a review
panel, exact bytes are accepted only by: a trusted source (`trusted`); Ouroboros's
own recorded check of that release, only in Cyber Pro (`self_check`, its notes in
`install.json`; outside Cyber Pro it admits nothing); the owner's Skip review or "Update without
review" of that release (owner attestation, `owner_skip`); and an owner edit: the
admitted selected-skill task the owner's own door started records `owner_edit.json`
for each revision it writes. Each binds to exact bytes; any later edit leaves it.
`OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS` is default-on as of v6.10.0 (the owner
may disable it), in which case a fresh executable review grants only
manifest-declared settings keys and host permissions for that exact content
hash; when disabled, key and permission grants remain explicit.

The Skills UI Repair affordance starts an ordinary managed development task; no
automatic repair exists. Keep the selected skill, source request and admitted payload
revision; a legacy `task_constraint.mode="skill_repair"` record remains readable
as a selected-resource binding, never a reduced execution profile. Ordinary file,
shell, browser, delegation and functional skill operations remain available
through their existing guards. Review must not force-unload a skill because the
caller is repairing it. Preserve readonly and acting-child ceilings independently.

Before a selected payload operation, verify the task's known revision. File-tool
effects retain their existing attribution; after opaque process work, persist the
observed payload revision without claiming exclusive authorship. A known foreign
change refuses overwrite. No long shell lock, mandatory private Git copy, new
patch pipeline or automatic rollback is implied. Native seeds with `.seed-origin`
retain their existing protections. Ordinary user-managed native-directory payloads
without that marker retain the existing logical-external binding. Outside Cyber Pro, generic tools cannot write provenance, review, grants,
dependencies or other control state. Cyber changes retain their actual source;
they do not fabricate an independent review or owner action.

UI, CLI and task calls share each existing lifecycle effect owner. An owner-only
action names its exact action, selected skill and revision, requested grant items,
and source of expressed owner intent. The host validates actual caller identity,
source membership/provenance and current manifest/review/dependency restrictions;
the acting model interprets the source's meaning. A formatted reference or an
automatically generated repair request is not owner permission. Ordinary repair
does not grant all permissions, self-attest or authorize deletion. Preserve the
configured auto-grant policy, ClawHub no-Skip rule, and explicit owner disable/Stop.
Do not require repeated owner clicks for operations already covered by the task.

Keep review, grants, desired enablement, dependency readiness and loaded/tested
revision as separate facts. A dependency or load failure does not rewrite a review
verdict, and a functional test is not independent review PASS. Reconcile supported
owner attestation through the same post-review lifecycle. Project the actual
blocking phase and available next action; repeat review only when it addresses
that blocker. Verify the real installed script/tool/HTTP/widget/companion flow
and preserve an honest partial result if a required permission or runtime is
unavailable.

The direct/read-only inspection profiles retain their existing native read/list/
search contract; ordinary Repair does not grant owner-state access or widen
unrelated child/Presence authority.

A selected `skill_publish` task may recover one exact known manifestless
`user_repo` leaf through the existing selected-candidate binding: omitted
`bucket` permits inspect operations and creation only of root `SKILL.md` or
`skill.json`, after which ordinary discovery and fresh review resume. Reviewers
must verify that grouping roots, unknown/colliding identities, explicit wrong
buckets, nested manifests, and path escapes remain refused.

### Output contract

Reviewers answer in the common findings-only form of every review
(`triad_review.REVIEW_JSON_ARRAY_CONTRACT`): a JSON array of PROBLEMS ONLY.
Each entry carries `item` (exactly one of the names in the table below),
`verdict` `FAIL`, `severity` (`critical`/`advisory`) and a concrete `reason`
with the fix. If one item has multiple distinct concrete problems, return one
entry per root cause; each must stay visible. Return nothing for an item that
passes or does not apply — no `PASS` entries, no "Not applicable" rows. A
reviewer that found nothing returns the empty array followed by `NO_FINDINGS`;
a reviewer that said nothing about an item still counts for what it said. The
severity guidance below is applied to the findings before the verdict is read:
a hard trust-boundary item blocks whatever severity the reviewer wrote, an
advisory-only or unknown item never blocks, and a severity-driven item follows
the reviewer's word.

### Checklist items

| # | item | what to check | severity when FAIL |
|---|------|---------------|--------------------|
| 1 | manifest_schema | Does the manifest parse with matching `type` (`instruction`: no scripts/entry; `script`: non-empty `scripts`; `extension`: non-empty `entry`)? Is script `runtime` (manifest-level) one of `python`/`python3`/`node`/`bash`/`deno`/`ruby`/`go` (no per-script override)? Instructions never execute; Python extensions import `entry` with `register(api)` and may omit `runtime`. Each companion still needs its own runtime. Is `timeout_sec` > 0? | critical |
| 2 | permissions_honesty | Do the declared `permissions` match what the scripts actually do? Missing permission declaration for an effect the code performs is a concrete FAIL. Examples: `net` must be declared if any script uses `httpx`/`requests`/`socket`/`urllib`; `fs` must be declared if a script writes outside the skill state dir; `subprocess` must be declared if a skill spawns another process. A transport using the `presence` permission must submit only authenticated provider event facts plus an opaque owner-created binding; supplying prompt text, a profile, tools, roots, destinations, authority hashes, or owner commands as trusted fields is a concrete FAIL. | critical |
| 3 | no_repo_mutation | Does any script attempt to write to the self-modifying Ouroboros repo (`~/Ouroboros/repo/`)? Import of `write_file`/`commit_reviewed` against the system repo, `git add`/`git commit`, or any path that starts with `OUROBOROS_REPO_DIR` / `~/Ouroboros/repo` is a concrete FAIL. Skills may only propose patches by returning artifact bundles; commits go through the first-party reviewed path. | critical |
| 4 | path_confinement | Do scripts stay inside the skill directory and the dedicated state dir (`~/Ouroboros/data/state/skills/<name>/`)? Absolute paths, `..` traversal, and writes to arbitrary user home subdirs are concrete FAIL. Reading from outside the skill dir is OK for read-only lookups (e.g. system info), write-path confinement is the strict rule. | critical |
| 5 | env_allowlist | Is `env_from_settings` a short, justified list of settings keys? Core keys in `FORBIDDEN_SKILL_SETTINGS` (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`, `CLOUDRU_FOUNDATION_MODELS_API_KEY`, `GIGACHAT_CREDENTIALS`, `GIGACHAT_PASSWORD`, `ANTHROPIC_API_KEY`, `MINIMAX_API_KEY`, `DEEPSEEK_API_KEY`, `GITHUB_TOKEN`, `OUROBOROS_NETWORK_PASSWORD`) may be declared only when the skill genuinely needs that provider/token for its stated purpose; runtime forwards them only after a fresh executable review and a content-bound desktop-launcher owner grant. v5.2.2 dual-track grants: both `type: script` skills (forwarded by `_scrub_env`) and `type: extension` skills (forwarded by `PluginAPIImpl.get_settings`) are eligible; `type: instruction` skills cannot receive core keys. Mark unjustified core-key requests or non-forbidden secrets unrelated to the purpose as FAIL. An empty list is the default and always fine. | critical |
| 6 | timeout_and_output_discipline | Is `timeout_sec` reasonable for the stated workload (default 60, hard cap 300)? Do scripts print to stdout in chunks that the runtime can cap, rather than streaming unbounded output? Unbounded loops without a `break`/timeout path are a concrete FAIL. | advisory |
| 7 | extension_namespace_discipline | `type: extension` only: does the extension register its tool/route/ws-handler/ui-tab under the namespace derived from its `name` (e.g. provider-safe tool/ws names like `ext_<len>_<token>_<surface>`, route `/api/extensions/<name>/…`)? Tool and WS short names must be alphanumeric/underscore and at most 24 characters. Namespace collisions with built-in surfaces are a concrete FAIL. If the extension uses `api.send_ws_message`, are emitted event names short/provider-safe and paired with reviewed host-owned widget `subscription` components rather than arbitrary same-origin JavaScript? If the extension declares streaming UI, is it a reviewed extension route consumed by a host-owned `stream` component? A reviewed `module` widget may also consume the skill's own routes (including streaming responses) and the skill's namespaced WebSocket events through the host-mediated bridge (`OuroborosWidget.fetch` / `OuroborosWidget.onEvent`), which is not arbitrary same-origin JavaScript. If the extension owns background resources (threads, sockets, EventSource clients, subprocesses), does it register cleanup with `api.on_unload(callback)`? If the extension declares a widget render block, is it one of the host-owned schemas (`iframe`, `module`, or declarative v1: forms/actions, markdown/code, JSON/kv/table, tabs/chart, stream/subscription, progress/poll, file/gallery/media, map/calendar/kanban, group/metric/callout), with media sourced from extension routes, safe data URLs or an ESM module's own declared files and no arbitrary same-origin JavaScript? Nested interactive group/tab children must use stable identity and one host-owned lifecycle, while `subscription.render` stays transitively passive. A `page_module` permission (contributions with `render.kind: host`; CREATING_SKILLS "Page modules") is the deliberate exception to "no arbitrary same-origin JavaScript": its `ui/` modules run as trusted code in the owner's pages, so confirm at the source level that each reaches the host only through its `ctx` and the skill's own `/api/extensions/<skill>/` routes (no other `/api/*`, WebSocket, feed or composer DOM, keystroke or clipboard capture, nor network beyond its declared `destination`; browser APIs its engine needs, such as speech, are fine) and that `dispose()` stops what it started; a module that reaches the host past its `ctx` is a concrete FAIL. Not applicable for non-extension skills — return nothing for it. | severity-driven for applicable extensions |
| 8 | widget_module_safety | **v5.7.0+. ``kind: "module"`` widgets only.** A classic entry (the default, even for ``.mjs``) is fetched through ``GET /api/extensions/<skill>/module/<entry>`` and inlined; only an explicit ``render.entry_type: "module"`` loads ``entry`` as a native module from a per-mount lease URL, the frame's document base, that serves just ``entry`` and the declared ``render.assets`` as captured at publication (CREATING_SKILLS "Native modules and declared resources"). Either way the code runs in a sandboxed opaque-origin ``<iframe srcdoc sandbox="allow-scripts allow-pointer-lock allow-downloads" allow="autoplay; fullscreen; clipboard-write">`` with no ``allow-same-origin`` — ``document.cookie``, ``localStorage``, and ``sessionStorage`` throw ``SecurityError`` there by construction and need no source review — behind a parent-mediated ``fetch`` bridge that rejects paths outside the owning skill route prefix, except GET/HEAD reads of that mount's own files. Reviewers confirm at the source level what the sandbox cannot: (a) no ``fetch``/``XMLHttpRequest`` URL outside ``/api/extensions/<skill>/`` and the widget's own declared files, and no bespoke ``postMessage`` protocol to ``window.parent`` beyond the host bridge; (b) the declared launch policy ``render.start`` (SSOT ``ouroboros/extension_ui_validation.py::WIDGET_START_MODES``; see CREATING_SKILLS "Launch policy") fits the widget's weight — ``auto`` only for a cheap instrument, ``manual`` for a program that should not run all the time, ``retain`` only for a program that genuinely must keep running while the owner is elsewhere and stays cheap while hidden; (c) a widget with state worth keeping registers ``window.__ouroWidgetOnDispose(fn)`` (never assigns over it) and saves that state through the skill's own routes, because the frame is disposable; (d) a module that declares ``render.appearance: host`` uses the optional ``OuroborosWidget.onTheme(callback)`` bridge; a real consumer should prove both resolved palettes, while ``independent``/``fixed`` modules remain author-owned. The declaration is author/reviewer intent, not a source-level proof or runtime gate for legacy payloads; a source mismatch or missing browser evidence is advisory unless it exposes a concrete runtime failure. Acceptable interactions: ``fetch('/api/extensions/<skill>/...')`` (through the host bridge), ``window.OuroborosWidget.fetch('/api/extensions/<skill>/...')``, an ESM entry's imports, stylesheets, media and GET/HEAD fetches of its declared files by relative or ``import.meta.url`` address, ``window.OuroborosWidget.onTheme(callback)``, and host-supplied data attributes. Not applicable for non-module widgets and non-extension skills — return nothing for it. | severity-driven when kind=module |
| 9 | inject_chat_minimization | Does any use of the `inject_chat` permission have a narrow, user-facing transport purpose? The Host Service enforces token auth, skill-source attribution, rate limits, in-flight limits, fresh executable review, enablement, and explicit content-hash-bound grants. Reviewed chat transports may carry the same raw owner text as direct chat, including slash commands such as `/panic`, `/restart`, `/review`, `/evolve`, `/bg`, and `/status`; reviewers must evaluate whether the transport itself is authorized, attributable, bounded, and user-facing rather than treating slash-shaped text as automatically forbidden. A skill that accepts external inbound traffic must still show local defense-in-depth appropriate to its transport: owner/chat binding or an equivalent access rule, bounded polling/backpressure, and no unaudited broadcast to unrelated parties. Missing local defense-in-depth is a concrete FAIL for network transports. The same criteria and severity govern `notify_owner` (`POST /notify`): one bounded, attributable, user-facing sentence the host shows in the owner's chat as a System row signed with the skill's name, never a way to inject conversation or wake the agent. Not applicable when neither `inject_chat` nor `notify_owner` is declared — return nothing for it. | critical |
| 10 | event_subscription_minimization | Are `subscribe_event` and `subscribe_events` limited to the minimum host event topics required by the skill? `chat.outbound`, `chat.typing`, `chat.photo`, `chat.video`, `chat.document`, `chat.links`, `chat.quiz`, `chat.quiz_state` and `chat.card` expose owner/agent conversation data (including delivered file bytes, outbound link actions and the owner's verbatim quiz answers) and require explicit justification. `chat.outbound` also carries the owner's notes and skill notices (`reminder`/`skill_notice` System rows), which may name a private event; their transport destination and handling need the same justification. So does any `api.read_memory` use: it needs no permission yet answers with the installation's whole memory (the owner's quoted words, identity, scratchpad, knowledge, Project journals) under host authority, never the calling task's view. Wildcards, undeclared topics, forwarding subscribed chat content (notes and notices included) or read memory to unrelated external services, or serving read memory through a tool or route to an actor not entitled to all of it, are concrete FAILs. Not applicable when `subscribe_event` is not declared and `read_memory` is not called — return nothing for it. | critical |
| 11 | companion_process_safety | For `companion_process` / `supervised_task` skills: is every command declared as an argument list (not shell string), using an allowlisted runtime, with no writes outside `skill_dir` / `state_dir`, no unbounded restart loop, and cleanup on unload/panic? Does the process avoid inheriting secrets except through reviewed `env_from_settings` grants? Not applicable when no long-lived process/task is declared — return nothing for it; a transient `subprocess.run`/`subprocess.Popen` invocation of a build tool like `ffmpeg`, `ImageMagick`, or `git` inside a normal request handler is NOT a long-lived companion process and does not trigger this item (its safety belongs under items 4 / 6 / 13). | severity-driven when applicable |
| 12 | host_token_handling | If the skill calls the Host Service API, does it use the provided `SkillToken.use_in_request()` only at request construction sites, avoid logging/serializing tokens, and keep all host-service calls on the loopback endpoint? Printing, persisting, exfiltrating, or embedding the token into user-visible output is a concrete FAIL. Not applicable when the skill does not access the Host Service API — return nothing for it. | critical |
| 13 | error_handling | Does the skill surface actionable errors instead of swallowing exceptions, returning success on partial failure, or leaving users to inspect raw logs manually? Are retry/backoff paths bounded and purpose-specific? | advisory |
| 14 | integration_preflight | Does the skill include cheap local preflight checks for the APIs/files/runtimes it depends on before spending provider budget or starting long work? Missing preflight for fragile external integrations is an advisory FAIL. | advisory |
| 15 | bug_hunting | Are there obvious runtime bugs in reviewed code: wrong filenames, mismatched manifest script names, missing imports, impossible arguments, JSON/schema mismatches, blocking calls in async handlers, or untested happy-path assumptions? For every FAIL, cite the concrete runtime bug and state how you propose to fix it (file/symbol/change), so the author can apply the correction instead of guessing. Concrete likely runtime breakage should use `severity=critical`; latent issues, provider fragility, minor cleanup, or dead-code concerns should use `severity=advisory`. | severity-driven |
| 16 | completion_notification | For long-running or user-visible work, does the skill emit or document a completion/failure notification path (for example a host event, `events.jsonl` append, or clear stdout marker consumed by Ouroboros)? A skill that must reach the owner directly (a reminder, a finished external job) declares `notify_owner` and hands the host one bounded sentence through `POST /notify`, shown as a signed System row in the owner's chat — judged as under item 9: attributable, bounded, user-facing, and never a way to inject conversation. Not applicable for tiny synchronous utilities — return nothing for it. | advisory |
| 17 | context_snapshot_discipline | For `context_snapshot`: are the declared local snapshots compact deterministic facts, with honest dates/units/unknowns, no instructions, secrets or relative-age counters? Check atomic publication, the complete UTF-8 file bound and withdrawal on expiry/revocation (CREATING_SKILLS “Passive context snapshots”); no network, model or refresh belongs on prompt preparation. Not applicable for other skills — return nothing for it. | severity-driven when applicable |

### Severity rules

- Skill review verdicts are enforcement-independent:
  - `clean` — no FAIL findings.
  - `warnings` — one or more advisory FAIL findings, no blocker findings.
  - `blockers` — one or more critical/blocker FAIL findings.
  - `pending` — no reliable completed review verdict.
- Outside Cyber Pro, enforcement maps verdicts to execution:
  - `OUROBOROS_REVIEW_ENFORCEMENT=blocking`: `clean` and `warnings` are
    executable; `blockers` are not.
  - `OUROBOROS_REVIEW_ENFORCEMENT=advisory`: `clean`, `warnings`, and
    `blockers` are executable by operator choice. This changes
    `executable_review` only; it does not rewrite the verdict, suppress
    findings, or change `skill_review_status` semantics.
  - Outside Cyber Pro, `pending` is not executable. A stale critic verdict does not authorize bytes;
    under Advisory a separate current author acceptance may admit the payload
    after deterministic preflight. Blocking still requires fresh critic evidence.
- Review state stores findings and computes the verdict at load time. Agents
  and UI callers must use `review_gate.executable_review` / `executable_review`,
  not the raw status string, when deciding whether the skill is runnable.
- A deterministic `skill_preflight` FAIL is a structural gate failure, not an LLM
  verdict: it persists and aggregates to `pending`, which is non-executable under
  ordinary enforcement mode (advisory included). Cyber keeps the failed check
  and pending verdict visible while leaving the execution decision to Ouroboros.
- Hard trust-boundary items are blocker findings on any FAIL regardless of
  reviewer-supplied severity: `manifest_schema`,
  `permissions_honesty`, `no_repo_mutation`, `path_confinement`,
  `env_allowlist`, `inject_chat_minimization`,
  `event_subscription_minimization`, and `host_token_handling`.
- Items 7, 8, 11, 15 and 17 are severity-driven. A `FAIL` with
  `severity=critical` produces `blockers`; a `FAIL` with
  `severity=advisory` produces `warnings`. Reviewers MUST reserve
  critical severity for concrete dangerous or runtime-breaking cases
  that meet the Critical threshold rule below.
- Item 8 (`widget_module_safety`) applies to module widgets. It is not
  applicable when the extension does not use a module widget — return
  nothing for it. This runtime rule deliberately does not rely only
  on manifest `ui_tab` detection because extensions can register module
  widgets dynamically from `plugin.py` via `PluginAPI.register_ui_tab`.
- Items 9, 10, and 12 are critical only when their corresponding capability is
  declared or used. They are not applicable for skills outside that surface —
  return nothing for them.

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
multiple FAIL items that all require the same change. If the author's answer
to the previous review (the brief's "The author's answer to the previous review
of this skill" section) addresses the same finding, judge it against the current
bytes: do not re-raise a finding the bytes no longer earn; keep one they still
earn and say why the answer does not resolve it.

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
   specs land in `data/skills/<bucket>/<skill>/.ouroboros_env/`; npm
   defaults to `--ignore-scripts` and pip to `--only-binary=:all:`.
   A reviewed entry may explicitly opt into source builds/install scripts
   with a concrete executable check. Exact ``download`` entries bind URL,
   digest, size and relative target; any literal build steps also declare
   outputs and a check. These declarations must match the fresh reviewed
   payload at process launch; package-manager exit zero alone is not a
   functional verdict. Verified caches stay outside replaceable payload/env,
   and actual resource/package/output facts and diagnostics stay in deps.json.
   Global-manager specs (``brew``, ``apt``, ``cargo``, ``go``) remain manual
   setup warnings. Reviewers should confirm the auto-installed
   packages match the skill's stated purpose; an unjustified `pip
   install <package>` for a skill that doesn't import it is a FAIL of
   item 2 (`permissions_honesty`). The adapter still rejects Node/TS
   plugin packages outright at the staging step; seeing
   ``openclaw.plugin.json`` in the file pack means the install
   pipeline should have aborted, which FAILs item 1
   (`manifest_schema`) because the skill should not have landed.
   v5.8 generalises the same readiness contract to official and local
   manifests that declare reviewed `install` / `dependencies` metadata:
   Executable acceptance admits auto specs into `.ouroboros_env`; Enable prepares them,
   while load/exec refuse missing, failed, or stale dependency fingerprints.
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

These are **separate surfaces** with separate prompts and state, drawn from the same review pool:

- Repo review (the review pool's one wave, both parts, plus any preflight)
  protects the self-modifying `~/Ouroboros/repo/`. Its attempt state lives in
  `data/state/advisory_review.json`, keyed by staged diff snapshot; its records
  in `data/state/review_ledger/`.
- Skill review protects the external skills repo. Its record is a
  `surface="skill"` entry of the same `data/state/review_ledger/`, keyed by a
  content hash of the skill's manifest + payload files;
  `data/state/skills/<name>/review.json` is the run gate's projection of that
  record.

A blocked skill review must NOT create obligations, commit-readiness
debt, or any artefact visible to the repo-review pipeline — the two
surfaces are deliberately siloed so a sticky skill finding cannot
block repo commits and vice versa.

---

## Plan Review Checklist

Used by `plan_task` when the mind asks the council about an INTENTION — code, research, a
deliverable or an action in the world. The council is one organ: independent reviewers (by default
the review pool; the Main model when no reviewer is marked) weigh the plan and each other's answers.
The host builds what each of you receives: this section; the task objective with who wrote it, and
the owner's words as the host recorded them, with the own-room conversation inline as numbered lines
(both speakers, options, quiz cards with the chosen answer, attachment names, addressed mailbox
provenance; progress rows and host notices stay
in the exact snapshot those numbers address; a route that cannot hold the whole
conversation keeps the newest rows and names the cut as an exact line range; related rooms are
pointers); the plan when one exists, always whole, with the change since the revision you last
answered about when you answered before; everything said in the council since your last answer,
peers attributed by seat and model; your own open objections; the mind's message; and attached
evidence (bounded, every absence named). A body intention — its declared `affected_paths` reach
Ouroboros's own body, or the task declares body work with a recognized design source — carries
BIBLE.md and ARCHITECTURE.md in full (inline for an api reviewer; mandatory full reads with its own
tools for a retrieving reviewer). Any other plan carries no Ouroboros governance and no navigation
maps: judge it in its own domain. Judge only the evidence actually present; nothing missing is ever
silent.

**One question: is it right to START the work as planned?** Not "is everything specified": details
may be worked out while doing. The council is also where directions are compared before one is
chosen: challenge the premise and suggest simpler, more general alternatives when useful.
Suggestions, alternatives and questions are prose; the mind decides whether to adopt them. A premise
challenge, preference or repetition alone is not an objection; a demonstrated failure follows the
height rule below. No compulsory competing plan or finding quota. The mind authors the plan and is
the addressee of everything the council produces (P0).

**Before any plan, the question is the reading:** is this what the owner asked, and which direction
would you take? Compare their words with each account the work rests on — the objective, an issue, a
brief, an earlier plan, each labelled with its author — and say where one narrows, widens or replaces
what they asked. Nothing raised before a plan holds, and the plan that follows is judged as below.

### The plan you are reviewing

```
goal · done_when[] {id, claim, support} · affected_paths[] (files the work will CHANGE; [] = none)
plan (prose: the reading of the ask and its bar, approach, constraints, decisions that are
      expensive to reverse with their alternatives, risks)
```
`done_when` is what the task's result review will judge: each `claim` an observable outcome, its `support` how it
will be shown (run a check, inspect the artifact, a judgment and by whom). Ids are the mind's and
stay stable across revisions.

### The rubric (six domain-free questions + one for body plans)

| # | item | what to check |
|---|------|---------------|
| 1 | the bar | Does `done_when` capture what the ask needs? For a vague ask ("make it good"), is it the bar a demanding person who made the request would mean — nothing obviously missing, nothing met trivially? Is each claim an observable outcome a third party could check, or a method the owner fixed for this work — not a method of the mind's own choosing? When work changes what a person sees, does the plan apply the product's information-placement rules to each fact, including text added within existing components, and does a claim reach that changed view rather than only its data or API? |
| 2 | load-bearing decisions | Are the decisions that are expensive to reverse explicit, with their rejected alternatives and why? |
| 3 | constraints and invariants | Are the real constraints named — budget, deadline, safety, irreversibility, commitments to others? |
| 4 | deferrals | Is anything deferred that will be expensive to change once the work has started? |
| 5 | evidence sufficiency | Is the evidence enough to judge? If not, read what is missing with your own tools when you have them, or name exactly what is missing — a locator, or a question for the mind — instead of inventing a gap. |
| 6 | direction and health | Is this the right way to the outcome? Does an existing owner, primitive or contract in the subject already give the same owner-chosen result? Would the plan add a second home for a fact, a duplicate mechanism, state or a background loop nothing needs, or a concept a reader must trace across files (why it matters: DEVELOPMENT "Pragmatic SOLID")? Such an objection names all three: the plan element that adds the cost, the existing simpler path to the same owner-chosen outcome, and the concrete cost. Without all three it is advice, and a preference is advice. A method the owner fixed is not objectionable this way. For a new capability or control: is it on the lowest rung that delivers it, and does a new seam ship with its first consumer (DEVELOPMENT "Capability placement")? Offering a lower rung is advice, also where the owner fixed the method. |
| 7 | governance (body plans only) | Does the intention contradict BIBLE.md or a frozen contract? Name the principle or contract. A plan that narrows what an integrating focus holds — other rooms, time, people's words — to save room or money is a P1 horizon change, not a saving: removing what carries no meaning and the mind's own folding come first, the option that keeps the horizon leads, and the narrowing needs a recorded horizon decision; asking the owner does not make it the recommendation. |

### Height rule — what is an objection

An **objection** says that starting or finishing as planned would be wrong: being wrong about it
after the work starts would invalidate work already done, violate a declared commitment (the
owner's ask among them), or make a `done_when` claim unverifiable. It names what in the plan causes
the failure — a `done_when` id, a path, a passage, or the requirement it omits — and what happens.
Everything else is prose and never holds. A claim you cannot check as written is a question, not
an objection. A row 6 objection reaches this height only when starting as planned would itself
build the cost: a durable mechanism, state or boundary that should not exist.

An objection belongs to the defect, never to its remedy: the mind may answer with any change that
removes the defect, and you judge the changed plan, not whether your remedy was adopted. Narrowing
the plan so it no longer delivers part of what the owner asked needs the owner's realignment; an
unobjected plan is not the owner's consent. An objection that demands a new guard, check, binding
or restriction stands only when it names the fault it prevents, what in the plan's own world can
actually produce that fault (for a body plan: a trigger a real actor can produce on a supported
install, BIBLE P7), the legitimate work that still succeeds with it in place, and what it costs or
forecloses; otherwise it is advice, and an unverified premise is a question. A past incident is
not required. That an actor holding the same authority could already cause the same effect answers
only deliberate misuse; it never excuses an accidental fault such as a correctness error or a race.

### Your answer

Answer in the reply contract delivered beside this section (`PLAN_REPLY_CONTRACT`), the one home of
the reply's form and of who withdraws what. Adjudicate your own open objections first: withdraw each
with why (the change or the answer removed the defect, or you were wrong), or restate what is still
missing. A reformulation of an earlier objection is not a new one, and a new objection on a revision
you saw before says why it was invisible then. An answer with no objection is legitimate.
Nobody votes, and whether an open objection holds is the host's fact, not part of your judgment.

### Rules for reviewers

- Name a file, function or module in an objection only when it is what fails; a file-level
  observation is prose.
- Do not penalise missing tests, version bumps, changelog rows or doc updates — there is no code
  yet, and the commit gate reviews those at commit time.
- Do not require the plan to specify what can safely be decided while doing the work. "Unspecified"
  is an objection only when leaving it open is expensive to reverse.
- A plan with no file paths at all (a trip, a deck, a research question) is a first-class subject:
  the same questions apply, and repository conventions are irrelevant to it.

---

## Result Review Checklist

Used by `review_change(subject="result")`: the mind asks for it, and under task review Required the host
asks for it at a root delivery whose result carries produced work; task review Off refuses it. You review
the RESULT of one task, not a code change. The host builds the subject, frozen at the call, in four parts:
what was asked (the task, its owner's words and the claims it was held to, the latest accepted plan's
`done_when` among them), the reply the task gives, what the task produced (files and effects, named by path
and content hash) and the host-attested verification receipts.

**One question: does this result do what was asked?**
- Judge the result, not the method — unless the owner's own words or explicit choice in what was asked
  fix a method for this work: following it is then part of the ask. A fixed method skipped without
  saying so is a silent narrowing; one the subject cannot show is a gap to name, not by itself a
  failure. A method that only an account names (the objective, the plan) is not fixed by the owner.
- A partial result that says plainly what it did not do is honest, not a defect. A result that silently
  narrows what was asked, claims what it did not do, or contradicts its own material is a defect.
- Read the produced files you can reach; a claim about a file must match its content.
- When what was produced is something a person will look at, look at it (the frames the subject carries,
  at the viewer's size) and say first what a person sees there and whether it does its job; its geometry
  and its code come after.
- A finding names the concrete gap, where it is, and what would close it.

### Severity rules

- **critical** — the result fails what was asked or misstates what it delivers, so it should not be
  delivered as it is.
- **advisory** — a real improvement that does not block delivery.

---

## Coupling questions

Part 2 of the brief every retrieving reviewer of a change receives (`## Part 2 —
Coupling questions`, `ouroboros/tools/review_synthesis.py`): the seat that answers
the Change Review Checklist in Part 1 answers these questions in the same
brief and reaches the repository with its own tools (configured writable sessions
use private checking copies without candidate-apply authority) — the intent, the
complete change, the touched-path manifest, a compact repository index, governance
navigation and the per-change required-source manifest (a minimum owed to the seat,
never a claim that everything else is dispensable) are in the brief. On the
Ouroboros body the questions apply with the `Critical surface whitelist` of the
Ouroboros Body Layer; on another root (`review_change`, core layer) with the
universal whitelist of the Change Review Checklist, against the subject's own
documents. A packet seat (a model without tools) is not asked this part, and the
record says so. This part follows the assumptions and consequences of the change across
relevant producers and consumers, including connections between changed
modules. Review the behavior the change creates, alters or relies on.
Reading coverage (host-observed native
receipts; a session's `harness_observed` journal; unobserved where neither is
available) is diagnostic: preserve the received verdict and quorum participation,
disclose gaps, and leave further reading to the agent's judgment (BIBLE P3).

**Output contract:** the seat returns one JSON object whose `coupling` block has one
entry for every item below (`REVIEW_TWO_PART_OBJECT_CONTRACT` in
`ouroboros/triad_review.py`; a seat asked this part alone returns the bare array).
PASS is one line: a pointer to what was checked, `see Part 1 <item>` for a defect
already reported there, or `N/A — <reason>` when the change does not reach the
question. FAIL names the file:line (or exact artifact) and the violated invariant.
Several FAIL entries for one item are valid for distinct root causes; never PASS an
item that also has a FAIL. The gate forwards only FAIL entries; PASS rows stay in
the record's per-seat answers. The matrix is validated before findings are
classified (`ouroboros/tools/scope_review_contract.py`): a missing or unexpected
item, a duplicate PASS, or PASS+FAIL for one item leaves this part unanswered as a
reviewer output failure; a seat that answered Part 1 without it is recorded
`coupling_block_missing`, and a wave in which no responding seat answered it
aggregates `NOT_PERFORMED`, never PASS.

| # | item | what to check | severity when FAIL |
|---|------|---------------|--------------------|
| 1 | intent_alignment | Does the staged change actually fulfill the intended transformation, not merely touch related files? | critical if the incompleteness is concrete and evidenced; otherwise advisory |
| 2 | forgotten_touchpoints | Are there specific coupled files, tests, prompts, docs, configs, or sibling paths that must also change? Name the exact file(s) or symbol(s). | critical if a required touchpoint is concretely omitted; otherwise advisory |
| 3 | cross_surface_consistency | If behavior changed, do the surfaces that describe it still say what it does — prompts, docs, comments, tool descriptions, automation, user-visible workflow — and do they agree with each other? Severity follows the shared `Critical surface whitelist`: a whitelisted surface (release metadata, tool schema, module map, behavioural documentation of what a tool or command does at runtime, safety contracts) that is now false is critical; commentary, wording and style are advisory. | critical if a whitelisted surface is concretely false; otherwise advisory |
| 4 | regression_surface | Does wider repository context show a concrete sibling path, migration edge, or parallel flow that remains broken or incomplete after this change? | critical if it leaves a concrete broken/incomplete path; otherwise advisory |
| 5 | architecture_fit | Does the change fix the owner of the behaviour it touches? Critical in exactly two cases: (1) a violation the `Shared Contract Ownership` section beside this checklist makes critical; (2) the diff creates a second home for a fact or mechanism that already has an owner module — a duplicate store, resolver, state machine, cache or wrapper — or works around a neighbour instead of fixing that owner. A case (2) finding names the owner, the duplicate or work-around (file:symbol) and the concrete path on which they diverge — a write through one home and a read through the other, or a caller still meeting the unfixed owner; that path is the trigger the critical threshold asks for. A narrow fix where the class of problem is wider is advisory. | critical in the two cases above; otherwise advisory |
| 6 | implicit_contracts | Follow each fact the change creates, alters or relies on through creation → use → eviction/retirement: does every constant, data format, signature, shared state, call/return assumption or protocol invariant another module relies on still hold, with every caller updated? Name the exact module, symbol or call site; DEVELOPMENT.md "Shared behavior and data-flow changes" defines authority, scope, freshness and preservation evidence. | critical if a concrete broken contract or cross-module breakage is cited; otherwise advisory |

### Severity rules

- Any critical FAIL must cite a concrete file, symbol, prompt, doc, test, config, or sibling flow.
- If the reviewer cannot point to an exact touchpoint, the FAIL must be advisory, not critical.
- For narrative / prose / cross-surface findings, apply the shared `Critical surface whitelist`
  defined in the Change Review Checklist section above, with the body's carriers named in the
  Ouroboros Body Layer when the subject is the body. Only release metadata, tool schema,
  module map, behavioural documentation, and safety contracts qualify as critical. Wording
  of explanatory comments, stylistic mismatches in changelogs, and non-contractual prose
  are advisory regardless of how concrete the citation is.
