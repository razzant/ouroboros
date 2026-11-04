<!--
Thank you for contributing to Ouroboros.

Open this PR against the `ouroboros` branch, not `main` or
`ouroboros-stable`. Keep review artifacts out of the git diff: attach or link
them in the Review evidence section instead.
-->

## Summary

<!-- What changes, and why is it needed? Link an issue or discussion when relevant. -->

## Scope

In scope:

-

Non-goals:

-

## Verification

<!-- List exact commands and outcomes. Do not write only "tests pass". -->

- [ ] Focused tests for the changed behavior pass.
- [ ] The default local test suite passes, or the reason it was not run is below.
- [ ] Lint/static checks relevant to this change pass.

Commands and results:

```text

```

## Visual evidence

<!-- Check one. Visible UI changes need evidence from a real rendered flow. -->

- [ ] Not applicable; this PR has no visible UI change.
- [ ] Before/after screenshots or other rendered-flow evidence are attached below.

Evidence:

## Governance and documentation

- [ ] I read `CONTRIBUTING.md` and `docs/CHECKLISTS.md` in full; for a
      substantive change I mapped `BIBLE.md`, `docs/ARCHITECTURE.md`,
      `docs/DEVELOPMENT.md`, and `docs/DESIGN.md` by their headings and read
      every section relevant to this change in full.
- [ ] I updated tests and documentation where behavior or architecture changed.
- [ ] I did not include secrets, local settings, runtime state, logs, caches, or
      generated build/review artifacts in the commit; tracked material follows
      DEVELOPMENT.md "Documentation contract" (including plans and optional reports).
- [ ] I did **not** bump `VERSION` or release-only version carriers; maintainers
      open each development cycle and cut each release.

## Review evidence

<!--
Have a separate agent context review the final committed diff without editing
it. A subagent, new task, or fresh agent session counts; self-review in the
authoring conversation does not. Evidence must match the current base/head
range, so rerun after code changes or a rebase. If no separate agent is
available, use NOT_RUN and say why.
-->

- Review status (`PASS`, `NEEDS_CHANGES`, `INCOMPLETE`, or `NOT_RUN`):
- Authoring agent/context:
- Separate review agent/context:
- Reviewer model and effort (when exposed):
- Reviewed base SHA:
- Reviewed head SHA:
- Findings and disposition:
- Checks performed and coverage limitations:
- Full review output or artifact link:
- If not run, reason:

Scope checklist coverage (from the reviewer's JSON; one row per item, extra
rows for additional FAIL findings on the same item):

| Item | Verdict | Evidence |
| --- | --- | --- |
| intent_alignment | | |
| forgotten_touchpoints | | |
| cross_surface_consistency | | |
| regression_surface | | |
| architecture_fit | | |
| implicit_contracts | | |

<details>
<summary>Reviewer checklist JSON (validate with scripts/validate_scope_receipt.py)</summary>

```json

```

</details>

## Final checklist

- [ ] The PR base branch is `ouroboros`.
- [ ] The branch is based on a current `ouroboros` revision.
- [ ] The PR is one coherent change and is ready for maintainer integration.
- [ ] The description explains any limitations, follow-up work, or compatibility impact.
