# Contributing to Ouroboros

This guide is written for coding agents and people. Before editing, read it in
full and follow it as the repository's pull-request workflow. If an agent is
implementing an issue, the agent itself must perform the preparation,
verification, separate-context review, and PR evidence steps below.

The short version:

1. Read the project before changing it.
2. Branch from and open the pull request against lowercase `ouroboros`.
3. Keep one coherent scope, test it, and do not bump the version.
4. Freeze the final committed diff and review it in a separate agent context.
5. Record exact verification and review evidence in the pull request.

The pull request template in
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) mirrors
this flow.

## 1. Read the Project Before Editing

For a substantive change, ground yourself in the project documents before
designing or editing. Read [`docs/CHECKLISTS.md`](docs/CHECKLISTS.md) — the
review checklist single source of truth — **in full**: your change will be
judged against it. For the other four, build a navigation map from their
headings first, then read every section relevant to your change **in full**
(skimming a relevant section does not count):

- [`BIBLE.md`](BIBLE.md) — constitutional principles and design priorities.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — structure, data flows, and
  rationale for non-obvious decisions.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — engineering, testing, and
  review conventions.
- [`docs/DESIGN.md`](docs/DESIGN.md) — visual and interaction semantics.

When in doubt whether a section is relevant, read it. Reading everything in
full remains the strongest preparation for a large or cross-cutting change.

Reuse the modules, contracts, and authorities those documents name. Do not
invent a parallel mechanism when the repository already has one. A useful
first instruction for a coding agent is:

> Read CONTRIBUTING.md and docs/CHECKLISTS.md in full. Map BIBLE.md,
> docs/ARCHITECTURE.md, docs/DEVELOPMENT.md, and docs/DESIGN.md by their
> headings and read every section relevant to the requested change in full.
> Follow their current architecture and keep the requested change focused.

These documents may themselves be improved, but constitutional changes must
follow the semantic change process in `BIBLE.md`, and behavior, tests, and
documentation must remain consistent.

## 2. Prepare One Focused Change

Keep one coherent purpose per pull request. Make the description state:

- the problem or opportunity;
- what changed and why it fits the existing architecture;
- what is deliberately out of scope;
- important compatibility, migration, safety, or operational risks;
- exact verification and its result.

Open an issue or discussion first for a broad, ambiguous, constitutional, or
direction-changing proposal. Small, well-understood fixes do not need
ceremonial design work.

Never commit local settings, credentials, runtime state, logs, caches,
benchmark runs, generated review runs, or build artifacts.

## 3. Branch from `ouroboros` and Do Not Bump the Version

Keep an `upstream` remote pointed at the official repository and start from the
latest contribution branch:

```bash
git remote add upstream https://github.com/razzant/ouroboros.git  # once
git fetch upstream
git switch -c your-focused-branch upstream/ouroboros
```

If `upstream` exists, verify it instead of adding it again. Open the pull
request against lowercase `ouroboros`, not `main` or `ouroboros-stable`.

Before the first push, update the branch and resolve drift:

```bash
git fetch upstream
git rebase upstream/ouroboros
```

After publishing the branch, do not rewrite its remote history. Bring later
target changes in with a normal merge, or coordinate a replacement with a
maintainer. Any update invalidates earlier review evidence.

External contributors do not allocate release versions. Leave `VERSION`, the
project version in `pyproject.toml`, the editable root in `uv.lock`, the version
in `web/package.json`, `GATEWAY_CONTRACT_VERSION`, the README badge and Version
History, the named installer links in README and both install pages, the
Architecture version header, and release tags unchanged.
Maintainers assign collision-free release metadata on the final landing tree.

Commit the intended change before producing final evidence, then require
`git status --short` to be empty. Review evidence covers a committed
base-to-head range, never uncommitted edits.

## 4. Verify the Change

Use [`README.md` → Run from Source](README.md#run-from-source) for setup. Run
focused tests while developing, then the default local suite when practical:

```bash
make test
```

Record exact commands, outcomes, and producer exit codes. If a check could not
run, record `NOT_RUN` and the reason instead of claiming it passed.

Tests that spawn a real process, bind a real port, or mutate module-level
global state must be marked `@pytest.mark.serial`. A merely slow test must be
made faster or split, not moved to the unbounded serial pass. See
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the CI split and isolation
rules.

For a visible UI change, inspect at least one relevant rendered user flow and
attach before/after screenshots or equivalent evidence. A saved screenshot
that was not inspected is not visual verification.

## 5. Review the Frozen Diff in a Separate Agent Context

Before opening a substantive PR, the authoring agent must hand the final
committed diff to a **separate agent context**. Use a subagent, new task, or
fresh agent session. Reviewing in the authoring conversation does not count.

The main review path is an **agentic checklist review**: the reviewer reads
the repository with its own tools and covers the "Intent / Scope Review
Checklist" from [`docs/CHECKLISTS.md`](docs/CHECKLISTS.md) — every one of its
eight items (`intent_alignment`, `forgotten_touchpoints`,
`cross_surface_consistency`, `regression_surface`, `prompt_doc_sync`,
`architecture_fit`, `cross_module_bugs`, `implicit_contracts`) — following
that checklist's output contract: one JSON array covering all eight items,
PASS rows mandatory and justified with a concrete artifact, FAIL rows with
severity (a critical FAIL names an exact file/symbol).

Give the reviewer the issue or goal, non-goals, exact base and head SHAs, and
repository access. The reviewer must not edit the candidate. Use this compact
instruction:

```text
Review the final pull request diff from <base SHA> to <head SHA>. Do not edit.

Read CONTRIBUTING.md and docs/CHECKLISTS.md in full. Map BIBLE.md,
docs/ARCHITECTURE.md, docs/DEVELOPMENT.md, and docs/DESIGN.md by their
headings and read every section relevant to this change in full.
Inspect the complete diff, touched files, relevant callers, tests, and docs.

Cover the Intent / Scope Review Checklist from docs/CHECKLISTS.md exactly:
output a JSON array of objects with the keys "item", "verdict" (PASS/FAIL),
"severity" (critical/advisory), and "reason", covering all eight checklist
items per its output contract — PASS rows are mandatory and justified with a
concrete artifact; a critical FAIL names an exact file/symbol. Then report
any further findings with severity and file/line references, checks
performed, coverage limitations, and one verdict: PASS, NEEDS_CHANGES, or
INCOMPLETE. Do not report PASS when required material was unavailable.
```

Paste the reviewer's checklist JSON into the PR's review-evidence block and
fill the checklist table from it. Validate the JSON locally before opening
the PR — the schema validator reuses the runtime contract and accepts the
array bare, inside a fenced `json` code block, or embedded in the
reviewer's prose. It
validates SHAPE, not truth: a passing receipt is well-formed coverage, not a
review verdict.

```bash
python scripts/validate_scope_receipt.py path/to/receipt.json
```

Reproduce material findings when possible. Fix confirmed problems, and briefly
record why any finding was rejected or deferred. Any code change, rebase, or
conflict resolution makes the old review stale; review the new final range.
Stop when no material finding remains rather than chasing an unbounded review
loop.

If no separate agent context is available, do not substitute same-context
self-review. Mark the review `NOT_RUN` and explain why in the PR.

### Maintainer-grade project-native review command

Ouroboros can produce review evidence in a structured SHA-bound packet. Its
contributor mode uses the reviewer slots actually configured on the machine:
`api_chat`, `agent_session`, or a mixture.

Treat this command as **maintainer / large-window tooling**, not the default
contributor path. The scope reviewer's required-artifact pack (protected
runtime paths, prompts, contracts, canonical docs, the review stack) is
required regardless of how small the diff is, and on a default install it
can exceed the configured scope slot's context window even after every
degradation step — the run then fails closed with `SCOPE_REVIEW_BLOCKED`
and still preserves the evidence packet (marked incomplete). The documented
routes past that pack budget are: configure the scope row as an
`agent_session` reviewer (a different delivery class — it reads the
repository with its own tools instead of being handed one assembled pack,
and needs its own confirmed 200K+ window), or configure an API scope slot
whose confirmed context window fits the pack. The agentic checklist review
above needs neither.

Configured API slots need their provider credentials and a positive finite
`TOTAL_BUDGET`. Agent-session slots need their configured agent route and
account to be available. The wrapper checks route-specific readiness where it
has a reliable probe; the selected route reports other failures explicitly.

From a clean committed branch:

```bash
python scripts/run_external_review.py \
  --contributor \
  --base-ref upstream/ouroboros \
  --head-ref HEAD \
  "<PR title>" \
  --goal "<goal>" \
  --scope "<scope>"
```

The command creates `review-evidence.json`, `full-output.txt`, and
`review-packet.zip`. The packet records the configured slots, observed
route/model/profile facts, absent telemetry, base/head/tree/diff hashes,
verdicts, and incomplete or degraded actors. It fails closed when the declared
slot route and observable execution receipt disagree or cannot be correlated.

Applied reasoning effort is not currently exposed by every route. The packet
records configured effort as requested and leaves effective effort absent
rather than presenting the request as observed fact.

The lane always executes the target base's own review machinery: run from any
checkout that is not the base, it re-runs itself from a detached worktree of
the base commit. Your proposal is therefore never reviewed by its own copy of
the review flow, whatever it touches, and no extra step is needed when a PR
changes the review script or review substrate.

## 6. Open the Pull Request

Complete the PR template with:

- summary, scope, and non-goals;
- exact verification commands and outcomes;
- authoring agent/context;
- separate review agent/context and model when exposed;
- reviewed base SHA and head SHA;
- verdict, findings, and their disposition;
- checks, coverage limitations, and full output or artifact link;
- the scope-checklist coverage table and the reviewer's checklist JSON;
- `NOT_RUN` plus a reason for unavailable verification or review.

Review output is public evidence. Inspect attachments for credentials, private
paths, or unrelated local data. Attach generated packets; never commit them.
Do not cherry-pick only favorable reviewer output or hide failed actors.

A clean review is evidence, not a promise to merge. It does not authorize a
commit, push, merge, release, or publication.

## Final Checklist

- [ ] The PR targets `ouroboros` and is current with its recorded base.
- [ ] The PR has one coherent purpose and explicit non-goals.
- [ ] `docs/CHECKLISTS.md` was read in full and every relevant section of the
      other project documents was read in full.
- [ ] Relevant tests and UI evidence are recorded honestly.
- [ ] No release-version carrier was changed.
- [ ] No secret, runtime state, generated run, or build artifact is in the diff.
- [ ] The final range was reviewed in a separate agent context, or `NOT_RUN` is
      recorded with a reason.
- [ ] Findings, limitations, and follow-up work are disclosed.

## Maintainer Boundary

Maintainers may reproduce tests, request a smaller diff, add integration fixes,
and rerun the project's final review on the exact landing tree. They choose the
landing parent and release version while preserving contributor authorship.

Contributions are licensed under the repository's
[`LICENSE`](LICENSE). By submitting a contribution, you confirm that you have
the right to provide it under those terms.
