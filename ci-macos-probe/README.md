# ci-macos-probe: one bounded b1s probe on a GitHub-hosted macOS arm64 runner

Test equipment for the qualification of the unified Update, **not product**. One job runs the stock official
Ouroboros 7.8.0 app and lets the app's own stock 7.8.0 updater deliver product C:
1. The app is copied from the pinned public 7.8.0 DMG.
2. It is checked as the release smoke checks it.
3. It is opened through LaunchServices and quit through its own Apple Event.
4. The stock updater delivers C from a run-local mirror; then the app is quit, reopened cold, and checked for its
   roots, profile and the owner's local function.

Nothing here changes C or the meaning of any test. The probe's own exit status and its uploads never stand in for a
verdict: `RESULT.txt` and `claims.tsv` carry the verdicts, and every claim is PASS, FAIL, UNKNOWN, REFUSED or NOT_RUN,
each with a short name.

## Files (the public inventory)
| file | role |
|---|---|
| `workflow/qualification-update-macos.yml` | the workflow; placed byte-identical at `.github/workflows/qualification-update-macos.yml` |
| `probe/run-probe.sh` | the runner: setup gates, the b1s step sequence through `capture.py`, the overlay claims, finish, result |
| `probe/equipment/guest/qb1.sh` | the qb1 guest steps (changes below) |
| `probe/equipment/guest/procwatch.py` | process and exit observer; docstring hygiene, and `PYTHONPYCACHEPREFIX` recorded in plain |
| `probe/equipment/host/mkbundle.sh` | C's bundle and bind file; comment-only hygiene |
| `probe/equipment/scripts/capture.py` | bounded, create-only receipts of each step, unchanged |
| `probe/equipment.patch` | the behaviour changes of the equipment copies (qb1.sh and procwatch.py), relative to their originals with the comment-only hygiene applied |
| `probe/OVERLAY-SHA256` | the four equipment digests, checked by the probe before anything else |
| `OVERLAY-MANIFEST.json` | provenance: original digest, overlay digest and change of each equipment file |
| `README.md` | this file |
| `SHA256SUMS` | every file above except itself, checked by the probe |

Only these files, and the workflow copy at its path, are published. Anything else in a working copy of this folder is
local and is not part of the execution commit.

**Equipment changes** (`probe/equipment.patch`; behaviour changes in `qb1.sh` and one name in `procwatch.py`):
- procwatch's plain allowlist gains exactly one name, `PYTHONPYCACHEPREFIX` (the bytecode-cache path the app's own
  launcher sets). The first hosted attempt recorded it as a hash only, so `localfn` was rightly UNKNOWN. Every other
  `PYTHON*` name not on the list is still a hash only and still makes `localfn` UNKNOWN.
- `facts` and `obs` record TCP listeners with `lsof -u "$(/usr/bin/id -u)"` (the actual user), not `-u anka`. These are
  recorders, never verdicts.
- `localfn` for the `site` kind (b1s) runs the helper with the serving process's own observed package-loading
  environment (HOME, VIRTUAL_ENV, `__PYVENV_LAUNCHER__` and every `PYTHON*` name, from its procwatch snapshot). It used
  to construct the expected one. If that environment is not recorded in plain, the step is UNKNOWN.
- `localfn`'s yaml containment also counts the macOS app's own data-file links. PyInstaller keeps the data files of
  `<X>.app/Contents/Frameworks/<P>` at the same paths under `Contents/Resources/<P>` and links them (the stock
  `ouroboros/launcher_bootstrap.py` documents this linking). This change handles only that same-path twin geometry:
  only that twin counts, and only if it is an existing file named under the prefix. A resolved target anywhere else
  outside the prefix and user base still FAILs, and the verdict now prints it. In the second hosted attempt, all three
  `localfn` checks recorded yaml named under the serving prefix and resolved outside both allowed roots (the resolved
  prefix and user base). They did not record the resolved target, so they do not show that it was this twin. Those
  verdicts remain FAIL. This change is not a historical PASS, not proof of native function containment and not an
  acceptance.
- Internal identifiers of earlier lab work are removed from comments and docstrings. Lab names that are also code
  (the `q7b4/wheelhouse` path, the `FB1_`/`Q7B4_` prefixes, mkbundle's lab-mode commit message) are unchanged. This
  hygiene is not reproduced in the patch; `OVERLAY-MANIFEST.json` pins every original and every copy by sha256.

## Pins
- **Product C (C2):** `553c8f892b10ed7b278b0f87c0e48adb82b9a4a5`.
  - Tree `50ab1b2d1e7a5f0e3b935f103339dc509e6ca95c`.
  - Its sole parent is the earlier C `08bfafd851a139bade14fb75cec005ee9824b32f`. The only differences are
    `ouroboros/app_install.py` (an optional `O_BINARY`, 0 on macOS) and three test files.
  - The first hosted attempt (execution commit `aab34cf26f565ce13c89a589b2e05aa54a9c2f20`) ran against that earlier C
    and stays bound to it.
  - Runtime lock sha256 `68e739bde5b7fbf90d2663eb01939f52c5c4b0b087097390e05807e143e8673d`, the same as 7.8.0's.
  - Kept apart from the execution commit E (`GITHUB_SHA`). The probe requires:
    - E has exactly one parent, C, and is the checkout's HEAD;
    - `git diff --no-renames --name-status C E` holds only additions, and its paths are exactly the public inventory:
      `ci-macos-probe/SHA256SUMS`, every file it lists, and the workflow path;
    - C's tree is the pinned one.
  - The bundle is built from C, never from E. After delivery the probe requires E to be absent from the run's checkout
    and mirror.
- **Base:** v7.8.0 = `6f3ddff05ff3e6c6f8d007ff96358df6af872e83`. `mkbundle.sh` also refuses unless v7.7.0, v7.7.1
  and v7.8.0 name their official commits.
- **Public inputs** (exact bytes and sha256, checked before the final file name exists):
  - `https://github.com/razzant/ouroboros/releases/download/v7.8.0/Ouroboros-7.8.0.dmg`: 754,985,809 B, sha256
    `41b9a8901d29c9a9b5dfdda8780d9317c441ba8c2941d949041c15abae0eff57`.
  - `https://github.com/razzant/ouroboros/releases/download/v7.8.0/SHA256SUMS`: 2,642 B, sha256
    `c3c1c197760efa4a4a6f6b791f097c5fc288343f2ca708137180401d84dec049`. It must contain the exact line
    `41b9a890…57  Ouroboros-7.8.0.dmg`.
- **Actions, by the commit C's own workflows pin:**
  - `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`;
  - `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02` (C's own workflows also use its
    `include-hidden-files`).
- **Job:**
  - triggered only by a push to `qualification/update-macos-*`;
  - `permissions: contents: read`; checkout with `persist-credentials: false`;
  - no secrets, no logins, no signing, no provider keys.

## Runner fixtures (create-only)
- **The qb1 geometry is kept.** `/Users/anka/oulab/<namespace>` with namespace `qbci`; run `b1s` (kind `site`,
  port 28823); tmp root `/private/tmp/qbci-b1s`.
- **`/Users/anka` must not exist.**
  - It is made once with `sudo -n mkdir -m 0700` and handed to the runner user.
  - It then holds an empty `.bash_profile`, `oulab/packages` (the DMG) and `oulab/pkg/7.8.0` (the observer: a whole
    copy of the pinned app, checked with `codesign --verify --deep --strict`).
  - It also holds `oulab/tools/…/CommandLineTools`, a link to the runner's own Command Line Tools (else
    `xcode-select -p`), and the namespace's `bin` and `pkg`.
  - Last, it holds `oulab/qbci/evidence`, a link into the run's own `out/guest-evidence` (see Upload).
- **Ownership is this run's own successful `mkdir`** of `/Users/anka`, recorded at that moment, never inferred later
  from a pathname. If `/Users/anka` exists, the fixture is refused. Nothing under it is then written to, listed,
  sized, quit into or uploaded; that also holds for the finish and the stop handler.
- **Nothing existing is overwritten or deleted.** The `bin`/`pkg` digests are re-checked before every step.
- **qb1 steps run under `env -i`** with `HOME=/Users/anka` and the runner's real user name. The app receives the run's
  own roots through qb1's launch environment (custom app root, repo, data, settings, pid and port files, home, ports).

## Sequence
Windows are the outer capture bounds, in seconds; `gate` stops the chain on anything but PASS.

| # | step | kind | window |
|---|---|---|---|
| S | runner facts; `runner-arm64` (`uname -m`, `hw.optional.arm64`, not translated; the image recorded) | gate | – |
| S | `disk-start` (≥ 10 GiB free), `overlay-integrity`, `clt-git`, `execution-pin`, `fixture-root` | gate | – |
| S | `input-sums`, `sums-names-dmg`, `input-dmg`, `observer`, `product-bundle` (`mkbundle.sh commit C`), `equipment` | gate | – |
| 1 | `facts x` | recorder | 300 |
| 2 | `selftest x` (procwatch's exit-status method) | gate | 120 |
| 3 | `install b1s` (official 7.8.0 app: ditto, codesign, `spctl`, version) | gate | 1800 |
| 4 | `gatekeeper`: the release smoke's own checks on the DMG and this app copy (below) | gate | – |
| 5 | `mirror b1s` (main/ouroboros/ouroboros-stable = C, official tags; the install's `insteadOf` points here) | gate | 1800 |
| 6 | `openw b1s open-1 300` (LaunchServices open, loopback HTTP 200), then `identity-open-1` | gate, claim | 600 |
| 7 | `quit b1s quit-1 240` (Apple Event) with an extra procwatch, then `quit-1-request` and `quit-1-exits` | gate, gate, claim | 480 |
| 8 | `lscanary b1s canary` (file and localStorage canaries while quit) | claim | 300 |
| 9 | `compose b1s` (owner H, I, W and U on v7.8.0's stock checkout) | gate | 600 |
| 10 | `openw b1s open-2 300`, then `identity-open-2` | gate, claim | 600 |
| 11 | `obs b1s pre`, `localfn b1s fn-pre` | recorder, claim | 600, 300 |
| 12 | `settings b1s channel OUROBOROS_UPDATE_CHANNEL qa` (the owner's ordinary Settings save of the QA channel, read back) | gate | 300 |
| 13 | `legacy b1s legacy 1800` (stock check, preflight, apply auto_merge), then `qa-target` | gate, claim | 2400 |
| 14 | `c-delivered`, then `identity-post-legacy` (the server the stock updater restarted, against open-2's tuple) | claim, claim | – |
| 15 | `obs b1s post`, `localfn b1s fn-post` | recorder, claim | 600, 300 |
| 16 | `quit b1s quit-2 240` with `quit-2-request` and `quit-2-exits` | gate, gate, claim | 480 |
| 17 | `openw b1s reopen 300` (cold reopen), then `identity-reopen` | gate, claim | 600 |
| 18 | `obs b1s cold`, `profile-survives`, `localfn b1s fn-cold`, `effects b1s fx` | recorder, claims | 600, –, 300, 600 |
| F | finish: leftover watchers get their stop file; if a process of the run is seen, one ordinary `quit b1s quit-final 240` with its request and exits; then `owned-processes-after` and `guest-evidence` (a digest list of the live guest evidence) | always | 480 |

## Overlay claims
- **`gatekeeper`.** The release smoke's checks, unchanged:
  - `xcrun stapler validate` on the DMG and the app copy;
  - `spctl --assess --type open --context context:primary-signature -vv` on the DMG and `--type exec -vv` on the app,
    each showing `source=Notarized Developer ID` (install's own assessment too);
  - `codesign --verify --deep --strict`.
  - If `spctl --status` is not "assessments enabled", the claim is UNKNOWN `gatekeeper_assessments_not_enabled` and
    nothing is opened. Because a copy downloaded by `curl` carries no quarantine, LaunchServices would not evaluate it.
- **`identity-*`.** After each open that served (and after the stock updater's restart), the one `server.py` process
  is identified:
  - **Runtime:** its interpreter (kernel image path) is inside the app (`open-1`, `open-2`) or inside the app or run
    root (after delivery).
  - **Source:**
    - `open-1`: HEAD is v7.8.0's commit, with no tracked change;
    - `open-2`: HEAD is the composed owner commit on v7.8.0;
    - after delivery: HEAD contains C and differs from C only in the owner's paths.
    - In every case, tracked changes stay in the owner's paths.
  - **Loopback:** `127.0.0.1:<port>` answers 200.
  - **Roots:** the server's own recorded environment must hold the exact `--env` tuple qb1 passed to that open
    (`argv.txt`): C's launch tuple (`OUROBOROS_APP_ROOT`, `_REPO_DIR`, `_DATA_DIR`, `_SETTINGS_PATH`, `_PID_FILE`,
    `_PORT_FILE`), the server port, host-service port and host, `HOME` and `TMPDIR`. It must also hold the run's user
    base (`PYTHONUSERBASE` = the run's `data/state/python-userbase`). Paths compare by real path.
    - A value that differs is FAIL; a requested name absent from the server's environment is UNKNOWN.
    - qb1's default-location witnesses (`data/settings.json`, `app/ouroboros.pid`, `data/state/server_port`,
      `home/Ouroboros`, `/Users/anka/Ouroboros/data`) must be absent.
  - Untracked files outside the owner's paths are recorded, not judged.
- **Quit.** A normal Quit needs three claims:
  - `quit-N`: qb1's cold state;
  - `quit-N-request`: the Apple Event request itself succeeded (osascript exit 0). A denial (-1743) or a timeout
    (-1712) stays its own non-PASS;
  - `quit-N-exits`: an extra procwatch (`--need exit+status`, qb1's own marks) armed before the request observes the
    launcher's and the server's exits and every registered exit status. Exit codes are recorded, not judged.
- **`qa-target`.** Stock Stable follows only the newest `vX.Y.Z` release tag, and QA follows the `ouroboros-stable`
  branch tip.
  - So, after the owner's QA save, the stock updater's own check and plan from `legacy` must both name:
    - channel `qa`;
    - `<remote>/ouroboros-stable`;
    - exactly C, as latest, checked and plan target.
  - The claim is FAIL only when every one of those fields was observed and one names another target.
  - A response that is unreadable or not an object, or that leaves a field missing, empty or not a string, is UNKNOWN,
    with the endpoints' own warnings and errors kept. The stock check's and plan's early and error responses do leave
    fields out.
  - Whether the owner's dirty merge comes out clean is `legacy`'s own trial outcome. It is recorded here, never
    promised.
- **`c-delivered`.** After the stock updater's PASS:
  - C is an ancestor of HEAD;
  - `git diff --name-only C HEAD` lists only the owner's paths (`OWNER_NOTES.md`, `owner-scratch.txt`,
    `owner-untracked/`);
  - C's tree is the pinned one;
  - E is absent from the run's checkout and mirror.
- **`profile-survives`.** The file canary's digest, and the `b1s-ls-canary` localStorage key with its value, are on
  disk in the run's private WebKit store after delivery and the cold reopen. This is a file and SQLite witness only. It
  does not show that the live WebView used that store.
- **`owned-processes-after`.** From one `ps` read, no process whose command line names the run root or the app. A `ps`
  or filter failure is UNKNOWN. The claim does not cover a descendant whose command line names neither; launcher and
  server exits are the `quit-*-exits` claims.

## Named outcomes (honest failures)
- **Setup:**
  - `not_arm64`, `low_disk`/`disk_unread`;
  - `procedure_error` (overlay or equipment changed, `GITHUB_SHA` unset);
  - `clt_git_missing`, `not_c_plus_inventory`;
  - `fixture_root_exists`/`cannot_create_fixture_root`;
  - `download_failed`, `digest_mismatch`, `sums_line_missing`;
  - `dmg_attach_failed`, `observer_invalid`;
  - `mkbundle_failed`, `bundle_bind_mismatch`, `equipment_copy_failed`.
- **Steps:** each takes qb1's own verdict.
  - Open: `launchservices_open_failed`, `app_not_running_after_open`, `not_serving`, `exit_observer_incomplete`.
  - Quit: `quit_not_cold`.
  - Canary: `no_private_webkit_store`, `no_localstorage_db`.
  - Legacy: `stock_plan_not_auto_merge`, `apply_rejected`, `no_new_server`, `head_not_moved`, `cand_not_delivered`.
  - Local function: `package_env_not_established`, `server_count_not_one`.
  - Generic: `capture_timeout` (the client's window ended; outcome unknown), `refused_before_effect`,
    `verdict_missing`, `verdict_exit_mismatch`, `time_budget`.
  - Open and quit failures also record the session type (`launchctl managername`).
- **Overlay claims:**
  - `gatekeeper_assessments_not_enabled`, `release_checks_failed`;
  - `identity_mismatch`, `identity_not_established`, `server_count_N`;
  - `apple_event_request_ok`, `tcc_automation_refused` (-1743), `apple_event_timed_out` (-1712, possibly an unanswered
    consent), `apple_event_request_failed`, `apple_event_request_unread`;
  - `settings_fail`/`settings_unknown` (the channel save or its read-back);
  - `qa_target_selected`, `qa_target_not_selected`, `qa_target_unread`;
  - `exit_watch_not_armed`, `exit_unobserved`;
  - `c_not_delivered`;
  - `canary_not_planted`, `cold_store_unread`, `canary_lost`;
  - `no_matching_rows`, `process_table_unread`, `processes_remain`;
  - `job_stop_signal` (`INTERRUPTED.txt`).
- **Overall:** `ERROR` (2) > `FAIL` (1) > `UNKNOWN` (3, including REFUSED and NOT_RUN) > `PASS` (0).

## Not qualified by this probe
- **The unified Update itself:** C's own preparation (`apply` with `prepare`), activation, restart, native app
  replacement, or the chosen-runtime and package-root changes it makes. Here only the **stock 7.8.0 updater**
  delivers C.
- **Upgrades from 7.7.0** (chain A, `b1a`/`b1e`), `b1p`, `b1r`, the published-range readiness roots, C's update checks
  and the full test suite. Upstream #2517 is not exercised, and no `.noindex` root is changed.
- **A browser download's first launch.** The DMG is fetched by `curl` (no quarantine), so no first-launch Gatekeeper
  evaluation or prompt occurs. Gatekeeper acceptance is established only by the release smoke's own assessments.
- **The live profile and the live server's own imports.** The profile claim is an on-disk witness. The local-function
  helper runs as its own process with the server's interpreter and observed package environment, not inside the
  serving process.
- **Anything of a real owner:** installation, home, Settings, profile, data or credentials. The run's roots are
  fixtures on a disposable runner. The runner user's real home is only listed before and after
  (`runner/real-home-*.txt`).
- **Equivalence with the lab VM.** The machine, macOS image, Command Line Tools git version, `.bash_profile` (empty
  here), GUI and Automation (TCC) state, and network all differ. Here the network is open internet; only the official
  repository URL is served by the mirror, through the run home's `insteadOf`.
- **Dependency changes.** C's lock equals 7.8.0's, so no dependency sync is exercised.
- **The stock Stable channel.** It never targets an untagged C, by its own release-tag rule. The first hosted
  attempt's `legacy` FAIL (strategy none, nothing sent) is that stock rule, not a product fault. This probe measures the
  owner's QA choice instead.
- **A clean merge of the owner's dirty work.** Whether it comes out clean is the trial's outcome, not part of any
  claim here.
- **Other platforms:** Windows, Linux, Intel Macs and other macOS versions.
- **Any measured stop or upload.** Cancelling a run, a job timeout, or the provider discarding the runner is not a
  measured process or VM exit, and provider teardown is not claimed. Processes still running after the finish's
  ordinary quit are reported, never signalled. A runner lost before its upload step uploads nothing.
- **Beyond one sample:** exit codes of the quit, untracked non-owner files, typical durations, and coverage of all
  tracked work. Only compose's witnesses are recorded.

## Bounds and estimates
- **Time:**
  - The job runs at most 180 min and the probe step at most 160 min; only these bound everything.
  - Inside the script, the 130-minute point (from the script's start) bounds the qb1 steps only: no step starts with
    less than 60 s before it, and no step window passes it. The finish keeps a 20 min reserve.
  - The setup before the first step is **not** under the 130-minute point. It covers two downloads, each bounded only
    by `curl --max-time 1800`, the observer copy, verification and `mkbundle.sh`.
  - Durations are **UNKNOWN** until measured; the windows are bounds, not estimates.
- **Stop:**
  - The workflow `exec`s the probe, so the runner's cancel or timeout signal reaches it directly.
  - The probe waits on downloads and qb1 steps in the background, so it handles the signal at once. It then gives the
    watchers their stop file, writes `INTERRUPTED.txt` and exits, with no collection or digests.
  - **A stop after the overall result was written** keeps that result in `RESULT.txt` as `completed_*` lines, under
    the final overall:
    - a measured FAIL or ERROR stays the overall verdict; anything else (PASS, UNKNOWN) becomes UNKNOWN;
    - the claim table's `interrupted` line says whether the stop came before or after the result;
    - a step summary that already shows the result gets an "Interrupted" amendment;
    - the exit code follows the final overall;
    - a `SHA256SUMS` already written predates the stop.
  - A signal during another setup command is handled when that command returns.
  - The runner's own escalation after its grace period is neither delayed nor measured here.
- **Disk:**
  - 10 GiB free is required at start and 3 GiB before each step.
  - **Estimated** use, unmeasured, about 4.5–5.5 GB:
    - the DMG, 0.755 GB;
    - two app copies (observer and run), about 1.5 GiB each;
    - the mirror, about 0.15 GB;
    - the run checkout, about 0.3 GB;
    - the bundle, about 0.03 GB.
  - Plus the full checkout: **UNKNOWN**.
  - Peak use: **UNKNOWN**. Measured values: `space.log`, `checks/du.txt`.
- **Upload:** one artifact, `qb-ci-probe-<run id>-<attempt>`, `if: always()`, hidden files included, retention 30 days.
  - **What it holds:** exactly the output root this run created,
    `$RUNNER_TEMP/qb-ci-probe-<run id>-<attempt>/out`. That root's name is unique to the job attempt, `RUNNER_TEMP`
    is emptied at the start of each job, and the probe makes the root create-only.
  - **Contents:** `RESULT.txt`, `claims.tsv`, `probe.log`, `space.log`, `runner/`, `inputs/`, `product/`, `checks/`,
    `watch/`, `receipts/` (capture receipts), `guest-evidence/`, `guest-evidence-SHA256SUMS`, and `SHA256SUMS` (written
    at a normal end only).
  - **`guest-evidence/` is qb1's live evidence.** `/Users/anka/oulab/qbci/evidence` is a link this run makes into it.
    So every step's evidence is under the upload root from the start, also after a cancel or a stop signal. It is
    complete only as far as `guest-evidence-SHA256SUMS` lists it.
  - **Nothing outside that root is ever selected.** A pre-existing `/Users/anka` is refused, and the probe then never
    writes to, lists, sizes, quits into or uploads anything under it.
  - On a public repository, logs and artifacts are public. Only named variables are recorded, never the environment.

## Before dispatch (owner of the run)
1. **The execution branch publishes C.** Pushing `qualification/update-macos-*` makes C and its history public on
   the repository. That is a publication decision.
2. **E must be one commit whose only parent is C.** It adds exactly the files listed above at `ci-macos-probe/`
   (SHA256SUMS, plus every file it lists) and the workflow copy at its path, nothing else. Before the push, check
   these three facts:
   - `git rev-list --parents -n 1 E` prints `E C`;
   - every line of `git diff --no-renames --name-status C E` is an addition;
   - its sorted paths equal the inventory.
   The probe repeats these checks on the commit that ran.
3. **No other workflow of C triggers** on this branch pattern (by static reading of C's workflows).
4. **Actions must be enabled for the repository,** with hosted macOS runners available.
