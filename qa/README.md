# Room-history reader QA (#4287)

Temporary harness for https://github.com/razzant/ouroboros/issues/4287. The QA
commit is not the product PR. It must have exactly one parent (the intended
product commit) and add exactly these four regular files:

- `.github/workflows/qa-room-history-reader.yml`
- `qa/README.md`
- `qa/witness.py`
- `qa/windows_reader_close_rename.py`

## One push, two GitHub-hosted jobs

The publisher verifies `HEAD^` against the intended product SHA, then pushes
that one QA commit to this exact branch (no tag and no PR):

```sh
git push origin HEAD:refs/heads/qa/4287-room-history-reader
```

Use the publisher's authorised GitHub credential, not the Actions
`GITHUB_TOKEN` (its pushes do not trigger ordinary workflows). The trigger is
`push.branches: [qa/4287-room-history-reader]`. No `workflow_dispatch`, and no
need to add the workflow to the default branch. On the checked product
parent, `ci.yml` and `desktop-window.yml` push only on `v*` tags;
`post-merge.yml` pushes only on `ouroboros`; the other push-triggered
workflows (`claudexor-platform-gate.yml`, `dependency-graph.yml`,
`scorecard.yml`, `sync-legacy-mirrors.yml`) accept only `main`, `ouroboros`
or `ouroboros-stable` — the first two behind path filters; every remaining
workflow is `workflow_dispatch`, `workflow_call`, `schedule` or
`pull_request` only. No existing `workflow_run` consumes this result.
Recheck the workflow inventory if moving to a different parent; do not
dispatch or tag the full suite.

Both hosted jobs independently verify event/ref, clean checkout, `GITHUB_SHA`
equals `HEAD`, the single explicit parent in the raw commit header, and the
exact four additions relative to that parent. `QA_HEAD_BOUND` logs both SHAs
and the ref. Linux executes a `git archive` of the parent in its own scratch;
Windows runs the checkout whose product tree is identical to the parent,
using the ordinary Actions checkout policy. Use the Actions run with the
exact QA SHA, branch and `push` event, not the most recent green for the name.
A missing, skipped, cancelled or timed-out run/job is not verification.

## Evidence expected from the jobs

- **`linux-panic`:** `ubuntu-latest` only. The witness checks Linux,
  `GITHUB_ACTIONS=true`, the GitHub-managed `RUNNER_ENVIRONMENT=github-hosted`,
  the exact push ref and its product parent, and a fresh directory beneath
  `RUNNER_TEMP`, before starting an isolated providerless app. The serving
  reader is the per-generation one (`history_reader.begin_generation` at
  lifespan startup); the witness marks each generation's read through that
  seam, so a pre-boot singleton can never be mistaken for the serving child.
  After `tasks_ready`, it makes a real loopback HTTP history GET and checks
  that the reply came from the observed live reader PID. Only then does it
  send one
  real `/panic` POST to the observed bound port. Required: HTTP 200 `status:ok`,
  server exit 99, present `panic_stop.flag`, the observed reader no longer
  executing, and read-only `PRAGMA integrity_check` returning `ok` on both
  **current.sqlite and usage.sqlite**. If dialog.sqlite is present, it too must
  pass; an untouched profile can legitimately report
  `not_created_in_fresh_profile` for dialogue alone. `guest-api-panic` records
  each fact. A zombie is shown as such (exited, not proof of OS reaping); the
  test does not attribute the reader's exit to a particular kill syscall.
- **`windows-reader`:** `windows-latest`, shipped Python inside the locked uv
  venv (not the packaged launcher). Five named product cases and one QA-only
  case, selected by exact IDs. The QA case creates three real fixture stores,
  releases its own SQLite handles, and renames each file out and back before
  spawning a child. It verifies that the live child runs the direct shipped
  interpreter and returns an actual history row over the reader IPC. It then
  releases serving-side handles in the test process, leaving the child live.
  `reader.stop()` must call the real Win32 `TerminateProcess` on that exact
  live child handle from the calling thread, not just observe an exit or
  close a pipe. The same child must exit with code 1, its stdin/stdout close,
  and the same three stores must rename out and back again. The test does not
  assert that the reader touched all three database files. The Windows job
  independently reads the six JUnit results: missing, extra, duplicate,
  skipped, failed or errored entries cannot produce its success marker. It
  prints `WINDOWS_OWN_HANDLE_STOP_AND_RENAME_VERIFIED` (QA test captured output)
  and `WINDOWS_SIX_CASES_VERIFIED` (job readback).

`*_UNVERIFIED`, `*_UNCONFIRMED`, `*_MISSING`, `*_FAILED`, wrong path/PID, or
nonzero exit are **not passes**. On Windows separate an out/back failure at
`BASELINE_BEFORE_SPAWN` (test root/setup handles) from `AFTER_CHILD_STOP`
(native stop/store boundary), from missing JUnit or a skipped/unexecuted ID.
Linux `LINUX_REQUIRED_STORE_MISSING` cannot be excused as optional dialogue.
A missing expected final marker is a gap, even if a workflow was scheduled.

These are synthetic lifecycle witnesses. They do not certify real data,
latency, the parallel after-cursor change, Android, native Windows Panic or
its tray/bridge, the frozen launcher, or full CI. Product anchors, all six
exact test IDs and every kit import were statically checked at
`a29f16109cbc3e2cc7410d551e9c9b1d0e7f44c1` (verified integration commit,
parent `9545e039b1db561d7f0377d453f359ca5078be7b`): the public endpoint
factory is `ouroboros.gateway.history.make_chat_history_endpoint` (the
internal `history_response.make_history_endpoint` now requires a builder and
quotas), and the serving reader is created per server generation, so the
witness marks each generation's read through
`history_reader.begin_generation`. Moving any anchor requires a fresh
source check. This local kit preparation did not execute either hosted job.
