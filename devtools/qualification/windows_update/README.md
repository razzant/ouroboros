# Windows Packaged First-Run & Managed Update Qualification Overlay

Separate disposable-runner QA workflow (`.github/workflows/windows-update-qualification.yml`,
triggered only on push to `qualification/windows-update-bundled-git` on `windows-2025`)
for qualifying packaged Windows desktop builds (`Ouroboros-<VERSION>-windows-x64.zip`)
with no reachable system Git on the host. QA only: never a merge or release gate.

## Modules

- `candidate.json`: 40-hex commit pointer naming candidate A (`candidate_sha`).
- `prepare_build.py`: stdlib-only helper run on base Python before any venv exists
  (`pointer`, `checkout`, `bind`, `verify-package`, `record-failure`, `summary`).
  Copies the harness to runner scratch and moves (never deletes) its own checkout out of the
  workspace, so A is checked out into an empty one; enforces A's own `VERSION` guard without
  editing product files, creates the isolated local QA test tag `v<VERSION>` at A, verifies the
  packaged ZIP and `_internal/git-standalone/` digests, strips host paths from published metadata,
  and always leaves `verdict`, `phase` and `code` in `result.json` + `$GITHUB_STEP_SUMMARY`, a
  harness exception included (`error` / `harness_error`).
- `fixture_model.py`: loopback (`127.0.0.1`) OpenAI-compatible server answering only
  `qa-fixture-model` and the two run-tagged synthetic prompts, each at most once. It validates
  Ouroboros's real host-authored segments, permits the exact Turn 1 exchange on Turn 2, refuses
  everything else, and attributes a refusal to slot 1 or 2 only when that request's newest user text
  is exactly that prompt.
- `windows_host.py`: removes the hosted runner's Git for Windows with its own uninstaller and
  verifies zero reachable `git` without running a bare-name program; finds the launched program's
  own top-level windows by its PID (no process census, no other window's title read); sends native
  input only on a disposable hosted runner, to an owned foreground window under the pointer; reads
  the Git trace sinks and runs the instrument's positive control.
- `git_traces.py`: stdlib-only, platform-free interpretation of those sinks (below), so it can be
  checked anywhere on synthetic text.
- `uia_reading.py`: stdlib-only, platform-free reading of a UIA tree (the visibility rule and the chat
  messages' screen witness), checked on synthetic trees by `tests/test_windows_update_chat_witness.py`.
- `scenario.py`: unpacks the verified ZIP, creates synthetic fast-forward commit B (one doc file)
  on `ouroboros` in the local bare mirror, launches `Ouroboros.exe` on a fresh
  `%USERPROFILE%\Ouroboros` profile and drives its windows with candidate A's own
  `WindowsComputerUse` UIA skill (`comtypes==1.4.17` in `$RUNNER_TEMP\qa-venv`) plus owned native
  input: no WebView2 CDP/Playwright, `auditpol`/`wevtutil`, screenshots or product edits.

## Consumer Surfaces & Verdict Discipline

The first settled outcome keeps its phase and code, except that a positive `failed` outranks a missing
screen witness the run went on past (2. below); Git traces read after a failure are facts only.

1. **Setup wizard -> fresh Main window**: clears the sticky `.wizard-footer` before acting; exact
   field readback; selects `OpenAI-compatible endpoint` from the `Main Model source` option row;
   keeps every other default, Safety included; the Setup HWND closes and a different Main HWND opens.
2. **Chat Turn 1 & Turn 2**: `model.cursor()` is taken before `#chat-send`. Only a refusal of that
   turn's own slot ends it early; another request's refusal (the first meeting, a wake) does not.
   A turn the model never received is a product `failed` only after the owner's row is seen in
   `GET /api/chat/history`; otherwise the Send did not take (`unconfirmed`). Words and order are
   the durable history's; the screen witnesses each message of the turn on its own, in whichever
   `Main` reading shows it: a `text` leaf under the exposed Main `chat-messages` feed, outside every `edit`/`combo_box` whose whole name is the
   message or holds its run-unique marker, UIA `IsOffscreen` read `false`, a non-empty rect. A
   message on the page but off screen gets one `scroll_into_view`. A witness still missing after an
   answered, exactly durable turn fails its check with allowlisted counts (never text) and the run
   goes on to the Update and Turn 2, `unconfirmed` at best; an omitted feed ancestor leaves visibility unconfirmed. Continuing requires a current successful
   read and a still-owned Main window; a lost window or a failed final read stops the step. Turn 1 in the window after the restart is judged the same way.
3. **Plain update check & Update A->B**: first waits until the passive `GET /api/update/status` shows
   the boot's own Stable check of A (`checked_target_sha`, `checked_at`), then selects `Development`
   in `Dashboard › Updates`. A pill already showing before the plain check is `unconfirmed`. Invokes the
   sidebar `Check for updates` (never `Check and describe`, never a POST from QA), waits for the
   Development check of B with a `checked_at` after that click, then the `#update-pill`, `Update to ...`
   and one `Update now`; after `concluded_by == "boot"` and the second chat it requires B checked out and
   running, the same `initialization_id`, and one attempt: `state.json` and `GET /api/update/status`
   both equal, on `SAME_ATTEMPT_KEYS`, the record as concluded before Turn 2.
4. **Which Git ran (`GIT_TRACE` + `GIT_TRACE2_EVENT`, inherited only by the app)**. The pinned Git
   never writes Trace2 `cmd_path` (it is computed before Trace2 starts). Each Git process instead
   writes one Trace1 line `trace: resolved executable dir:`, the folder of its own image from
   `_wpgmptr`, and one Trace2 session. Identity is a run-wide, bounded claim, never one session's
   path: as many identity lines as started sessions, none from the `argv[0]` fallback, every line
   inside `_internal/git-standalone/`, every `.exe` in those folders byte-equal to the ZIP. A
   foreign or altered folder is `failed`; anything else short of that, an empty set included, is
   `unconfirmed`. Before the app starts, the package's own `git version` must leave exactly one
   bundled line and one complete `version` session (positive control on this build); if it does not,
   or its sinks cannot be read, the run goes on and identity settles `unconfirmed`. A command counts
   in a window when its session starts there and every terminal event (`exit`, `atexit`) carries
   code 0 and its own time inside the same window. Windows: a successful `clone` from launch to the
   Setup window; a successful `fetch` from the plain check's click to the `Update to` click (its
   preflight comes later); and, within the carried attempt's own apply (`started_at`, after the
   recorded `Update now` click, to the `stage_started_at` of `restart_requested`, recorded before
   the restart), exactly `checkout -B ouroboros <B>` and `reset --hard <B>` by the argv tail after
   argv[0] (the `apply_managed_merge_update` steps). A missing or zone-less bound leaves its window
   not evaluated, never widened to the boot. Sinks are re-read until the counts settle, so a Git
   starting during the read cannot pass as a match, and a run whose other steps passed is judged
   even when both sinks are gone; after a failure the traces are diagnostics only.

## Published Evidence Scope & Explicit Non-Claims

- **Allowlisted synthetic JSON only**: `result.json`, `build-binding.json`, `package.json`,
  `host-git.json`, `git-traces.json` and `fixture-model.json`: no screenshots, UIA dumps, raw trace
  lines, `argv` or host paths (`assert_privacy_clean`).
- **Git scope**: only Git processes that inherited the app's environment;
  `system_wide_process_exclusion_claimed` is `false`. A build or environment step that stops
  (downloads, network, packages) is `unconfirmed`, not a product failure.
- **Local bare mirror transport**: `https_git_transport_tested` is `false`; internet HTTPS/TLS and
  the Windows certificate store in bundled MinGit are not exercised.
