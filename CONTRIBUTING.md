# Contributing to Ouroboros

Thank you for helping Ouroboros evolve. This guide explains how to prepare a
pull request that is easy to understand, reproduce, review, and integrate.

The short version:

1. Base your work on the lowercase `ouroboros` branch and open the pull request
   against `ouroboros`, not `main` or `ouroboros-stable`.
2. Read the project governance and engineering documents before making a
   substantive change.
3. Keep the pull request focused, test the changed behavior, and show rendered
   evidence for visible UI changes.
4. Do not bump the project version. Maintainers assign the final release
   version when integrating the pull request.
5. Optionally attach a current triad + scope review packet to enter the faster
   review path.

The pull request template in
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) mirrors
this flow.

## Read the Project Before Changing It

For substantive changes, read these files **in full** before designing or
editing:

- [`BIBLE.md`](BIBLE.md) — constitutional principles and design priorities.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the current structural and
  operational map, including the rationale for non-obvious decisions.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — engineering conventions,
  module boundaries, testing guidance, and the commit/review protocol.
- [`docs/CHECKLISTS.md`](docs/CHECKLISTS.md) — the review checklist single
  source of truth.

Use the existing modules, contracts, and single sources of truth described in
those documents before introducing a new mechanism. If a coding agent is doing
the work, give it the same instruction explicitly. A useful starting prompt is:

> Before editing, read BIBLE.md, docs/ARCHITECTURE.md,
> docs/DEVELOPMENT.md, and docs/CHECKLISTS.md in full. Follow their current
> principles and reuse the architecture and existing extension seams instead
> of inventing parallel mechanisms. Keep the change focused and verify it
> against the repository's tests and review checklist.

These documents are authoritative descriptions of the current project, not
immutable external rules. A pull request may improve `BIBLE.md`, the
architecture, development practices, or review checklists. Explain why the
change is needed, what invariant it preserves or intentionally evolves, and
keep related code, tests, and documentation consistent. Constitutional changes
must also respect the change process and semantic protections stated in
`BIBLE.md` itself.

## Choose a Focused Change

Prefer one coherent purpose per pull request. The description should make the
following clear:

- the problem or opportunity;
- what changed and why this approach fits the existing architecture;
- what is deliberately out of scope;
- compatibility, migration, safety, or operational risks;
- the exact verification performed and its result.

For a broad, ambiguous, security-sensitive, constitutional, or
direction-changing proposal, opening an issue or discussion first can avoid
expensive rework. Small, well-understood fixes do not need ceremonial design
work.

Do not commit local settings, API keys, runtime state, logs, caches, benchmark
runs, generated review runs, or build artifacts. Runtime data belongs outside
the git repository.

## Branch and Pull Request Flow

Keep an `upstream` remote pointed at the official repository and start from the
latest working branch:

```bash
git remote add upstream https://github.com/razzant/ouroboros.git  # once
git fetch upstream
git switch -c your-focused-branch upstream/ouroboros
```

If `upstream` already exists, verify it instead of adding it again. Before final
verification and review evidence, update your branch and resolve any drift:

```bash
git fetch upstream
git rebase upstream/ouroboros
```

Open the pull request with the base branch set to lowercase `ouroboros`.
`main` is the default public branch, and `ouroboros-stable` is a recovery
branch. Neither is the contribution target.

Keep the worktree clean when producing final evidence: commit the intended
changes on your pull request branch, then confirm that `git status --short` is
empty. The contributor review command examines a committed base-to-head range;
it does not approve uncommitted edits.

### Do not bump the version

External contributors should not edit release-only version carriers for a
normal pull request: `VERSION`, the package versions, README badge/history,
the architecture version header, or release tags. Ouroboros treats an
integrated commit as a release, but maintainers assign the collision-free
version during final integration, normally while squash-integrating the pull
request. This avoids unrelated version conflicts between concurrent
contributions.

You may still update ordinary README or architecture content when the behavior
or structure changes; leave only the release number to the maintainer.

## Verification

Install the source environment using
[`README.md` → Run from Source](README.md#run-from-source). The default local
suite is documented in [`README.md` → Run Tests](README.md#run-tests):

```bash
make test
```

Run the narrowest relevant tests while developing, then the repository's
default local test suite when practical. Record the exact commands and
outcomes in the pull request; “tests pass” without a command is not
reproducible evidence. Follow the marker and environment guidance in
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for integration, browser,
portable, and skill-smoke lanes.

Add or update tests for changed behavior. If a test cannot be run in your
environment, say which one and why rather than marking it as passed.

For a visible UI change:

1. Open at least one relevant real user flow in an available browser.
2. Inspect the rendered result, including the states and viewport sizes that
   matter for the change.
3. Attach before/after screenshots or equivalent evidence and state what was
   actually inspected.

A saved screenshot that nobody inspected is not visual verification. Mobile
or WebKit evidence is risk-driven rather than mandatory for every UI change.

## Optional Fast Path: Triad + Scope Review

A contributor-supplied triad + scope packet is optional, but it materially
reduces the work needed to evaluate a pull request. A current, complete packet
places the pull request on the faster review path. A pull request without one
may still be considered, but Ouroboros and the maintainers must reproduce the
review and may need to revise the change manually; that path is slower and the
pull request is more likely to be deferred when maintainer capacity is limited.

Neither a clean packet nor its absence decides acceptance. A passing review is
evidence, not a promise to merge, and maintainers may rerun it or request
additional changes.

If the pull request changes the review substrate itself — for example the
review script, reviewer configuration, review checklists, or production review
code — its contributor packet is diagnostic only. A maintainer must rerun the
review from a trusted target-base implementation before integration.

### What the contributor review runs

The checked-in
[`scripts/run_external_review.py`](scripts/run_external_review.py) contributor
mode runs the production triad and scope review over the committed range from
the target base to your branch head. It deliberately:

- resolves the reviewer models and reasoning efforts from the shipped defaults
  in the target `ouroboros` revision, not from values modified by the pull
  request;
- routes every triad and scope reviewer through OpenRouter;
- applies the clean blocking contract for the contributor packet;
- does **not** run the Claude advisory pre-review and does not require an
  Anthropic key;
- records the exact base/head binding and full reviewer evidence without
  committing or modifying the reviewed branch.

Triad completeness follows Ouroboros's production quorum contract together
with an authoritative scope verdict; it does not require every configured
actor to respond. The packet preserves actor statuses and degraded reasons, so
any failed, partial, or unparsable actor must still be disclosed even when the
remaining actors satisfy quorum.

Do not override the reviewer models, scope-review model, reasoning efforts, or
provider. The value of this packet is that every contributor uses the review
configuration selected by the current target version of Ouroboros.

### Produce a packet

First fetch and rebase on the current target, commit the final change, and
confirm the worktree is clean. Configure your own `OPENROUTER_API_KEY` without
placing it in a command argument, tracked file, shell history, or attachment.
Also set `TOTAL_BUDGET` to a positive finite USD ceiling that you explicitly
authorize for this run; contributor mode fails before model calls when that
ceiling is missing or invalid. The review calls may incur OpenRouter charges,
so use a key with sufficient remaining provider balance and check current
pricing before choosing the ceiling. For example:

```bash
export TOTAL_BUDGET="<authorized USD ceiling>"
```

Then run from the repository root:

```bash
python scripts/run_external_review.py \
  --contributor \
  --base-ref upstream/ouroboros \
  --head-ref HEAD \
  "<PR title>" \
  --goal "<goal>" \
  --scope "<scope>"
```

The script creates:

- `review-evidence.json` — machine-readable base/head, configuration, actor,
  verdict, and outcome evidence;
- `full-output.txt` — full human-readable triad and scope output;
- `review-packet.zip` — the attachment-ready packet.

Use the fresh output directory printed by the command. Review packets are
operator artifacts: attach them to the pull request, but do not add them to the
git diff.

### Attach and summarize the evidence

In the pull request's **Review evidence** section:

1. Attach `review-packet.zip` and, when useful for quick inspection,
   `review-evidence.json` and `full-output.txt`.
2. Record the reviewed base SHA and head SHA.
3. Record the exact command/profile and the resolved reviewer model IDs.
4. Summarize every triad actor verdict, the aggregate triad result, and the
   scope verdict.
5. Disclose known advisory findings, accepted trade-offs, infrastructure
   failures, skipped coverage, or other non-clean conditions. Do not present a
   failed or incomplete run as a pass.

The evidence is valid only for the recorded head and base. Rerun it after any
code change, rebase, conflict resolution, or other change to the reviewed
range. Do not cherry-pick only favorable actor outputs or substitute a
screenshot for the raw records.

### Protect secrets and private data

Treat review output as public before uploading it. Attach only the generated
packet, not the review drive, local observability store, `settings.json`, or an
environment dump. Inspect the packet for API keys, authorization headers,
credentials, private paths, or unrelated local data even when automatic
redaction reports success.

If redaction is necessary, preserve all verdicts and finding structure and
state exactly what was removed. Never upload a secret and never commit a
credential in order to make review reproducible; revoke any credential that
was exposed.

## Pull Request Checklist

Before marking the pull request ready for review, confirm that:

- the base branch is `ouroboros` and the branch is current with its intended
  base revision;
- the pull request has one coherent purpose with explicit non-goals;
- the description explains the architectural fit and important trade-offs;
- relevant tests pass, with exact commands and outcomes recorded;
- behavior and architecture documentation are updated where necessary;
- visible UI changes include inspected rendered-flow evidence;
- no version or release-only carrier was bumped by the contributor;
- no secrets, local state, logs, caches, generated runs, or build artifacts are
  present in the diff;
- triad + scope evidence is attached and current, or the reason it was not run
  is stated honestly;
- limitations, unresolved findings, and follow-up work are disclosed.

Disclosure of coding-agent assistance is optional. If you include it, name the
agent/model, what it changed, and what a human or independent process verified.
Agent use is context, not a positive or negative quality signal.

## What Happens After Submission

Maintainers may reproduce tests, rerun triad + scope review, ask for a smaller
diff, request changes, or integrate the contribution with additional fixes and
the final release-version update. Review evidence accelerates that process but
does not replace maintainer judgment or Ouroboros's own governance gates.

Contributions are licensed under the repository's
[`LICENSE`](LICENSE). By submitting a contribution, you confirm that you have
the right to provide it under those terms.
