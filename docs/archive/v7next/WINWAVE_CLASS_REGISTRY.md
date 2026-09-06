# R-WINWAVE — one decision per cross-OS class

The v7 campaign and upstream fixed the same Windows/macOS territory
independently. Plan §2 mandates the return of the v7 cross-OS wave and plan §11
names the hazard: re-applying it commit-by-commit would fight the upstream
fixes and duplicate them. So the return is **by class**, and this file is the
registry the ADOPTION row points at — one recorded decision per class, and no
class silently re-applied twice.

Reference wave (frozen, `ouroboros_v7_wip @ 9f691656`): commits `18048261`
(14 files), `073c610d`, `640a249a`, `711c982b`, `6cdc8e16`, `b20c94a1`. None is
an ancestor of this branch or of `managed/ouroboros`. The upstream fixes for the
same territory ARE ancestors here: `7de26338`, `c15389f4`, `18a4e17d`,
`78d168b8`, `d2f6701a`, `0c4acfd9`, `10779106`, `d428f125`, `4a550589`.

Decision vocabulary, one per row:

- **re-applied** — the class exists only on our side; the campaign landed the
  reference form (the carrier files do not exist upstream at all).
- **superseded-by-upstream** — upstream solved the same class, differently or
  more broadly; the reference form is NOT transplanted, upstream's stands.
- **not-applicable** — the carrier the reference patched does not exist in this
  tree, so there is nothing to decide until a carrier arrives.

Sixteen classes from the reference wave. (The campaign audit's summary line
said "15"; the recount below separates the
`PermissionError`-beside-`IsADirectoryError` clause from the `tools.jsonl`
utf-8 read — they were one line there.) A seventeenth class was found by the
matrix itself and is recorded the same way, in its own section: a class this
registry learns about from a red run still owes exactly one decision.

## Re-applied (7)

| # | class | decision | where it lives now |
|---|---|---|---|
| 1 | `fchmod` POSIX guard + paid-lane fail-closed off POSIX (a non-POSIX host must refuse the paid lane rather than write a live key before chmod; reference `b20c94a1`) | re-applied | `tests/fixtures_e2e_cancellation.py` (guard + refusal), landed by `4fffefb1` |
| 2 | 0600-mode assertions skipped where the OS has no POSIX mode bits | re-applied | `tests/test_e2e_cancellation_scenarios.py`, landed by `4fffefb1` |
| 3 | Canonical-path compares instead of string equality (Windows short paths, drive-letter case) | re-applied | `tests/test_external_review_script.py` (`3b62c1d6`) and `tests/test_cancel_protocol_inventory_s6.py` (`4fffefb1`) |
| 4 | `os.sep` / `!r`-mirror expectations instead of hardcoded `/` in message pins | re-applied | `tests/test_core_native_results.py`, landed by `fa2f6fc5` |
| 5 | Per-scenario registry route pin (the `640a249a` lesson: `user_files` is path-dependent so Windows takes the native route, `cognitive` is arg-dependent so the adapter runs everywhere — one blanket platform pin was the over-generalisation) | re-applied | `tests/test_registry_core.py`, landed by `ccbb933a` |
| 6 | Planted-stamp comparison tolerant of the 15 ms clock tick | re-applied | `tests/test_owner_stop_fences_s6.py`, landed by `88479fa7` |
| 7 | utf-8 ARCHITECTURE fixture (explicit encoding on read/write, not the cp1252 default) | re-applied | `tests/test_update_carriers.py`, landed by `7f0a1124` |

## Superseded by upstream (3)

| # | class | decision | evidence |
|---|---|---|---|
| 8 | Text-mode CRLF translation in the atomic writer — the root cause of the original 61-red Windows run | superseded-by-upstream | Upstream `c15389f4` decomposed it into `write_bytes_atomic` (+ UTF-8 encode) instead of adding `O_BINARY` inside `write_text_atomic`; the corrective lane then proved upstream's form is the BROADER fix (the non-fsync `Path.write_text` lane translated too). Replaying the reference would invert that decision. Ledger row 15; pinned by `tests/test_atomic_write_v639.py` |
| 9 | Launcher reaper: normpath'd path literals + POSIX-only enumeration test | superseded-by-upstream | Upstream `7de26338` normpaths the same literals (and the python binary path the reference missed) and, instead of skipping the enumeration test off POSIX, stubs `getuid` so it runs on every OS. Ledger row 16; pinned by `tests/test_launcher_server_reaper.py` |
| 10 | Child-process environment forwarding (`SystemRoot`, `TEMP`, `TMP`, `USERPROFILE`) | superseded-by-upstream | Upstream `78d168b8`, pre-cutoff; pinned by `tests/test_evolution_state_integrity_v3.py` |

## Not applicable in this tree (6)

| # | class | decision | why |
|---|---|---|---|
| 11 | Evidence/migration script hardening: JSON through stdin instead of argv (the Windows 32767-byte **command-line** cap, not a path-length cap), env forwarding, `PYTHONIOENCODING=utf-8`, stderr tail in the failure message | not-applicable | Carriers `scripts/v7_evidence.py` and `scripts/v7_migration.py` do not exist here |
| 12 | POSIX-scoped exactness pin of the prologue evidence | not-applicable | `tests/test_v7_prologue_evidence.py` does not exist here |
| 13 | Ledger test moved to the serial lane (its 72 MB RSS was a timeout symptom, not a memory limit) | not-applicable | The `test_v7_migration_ledger` carrier does not exist here |
| 14 | SIG9 exit-facts parity between the executor and the local shell | not-applicable | The asserting test is absent from `tests/test_shell_run_shell.py` and from everywhere else; the producer-side fallback exists (`ouroboros/tools/process_facts.py`, `tool_result.py`) |
| 15 | `tools.jsonl` read with an explicit utf-8 encoding | not-applicable | The reference test was re-derived as a `tests/test_tool_result.py` case, and the only reader of `tools.jsonl` content on this tree already passes `encoding="utf-8"` |
| 16 | `PermissionError` accepted beside `IsADirectoryError` on a directory write | not-applicable | The asserting directory-write test no longer exists in `tests/test_core_native_results.py` |

## Found by the matrix, not by the reference wave (1)

| # | class | decision | where it lives now |
|---|---|---|---|
| 17 | Source-text regex pins in the JS suite: a test that reads a source file off disk and matches a `\n`-bearing regex against it cannot pass on a CRLF checkout | re-applied | `web/tests/chat_plain_system_rows.test.js` normalizes CRLF to LF at both `readFileSync` reads, landed by `a0b35fcd` on the campaign integration branch — NOT on this worktree, which is why the row below still needs its own matrix leg |

Class 17 is upstream-born territory (`817a834f`, `dbd500cc`, both ancestors of
`managed/ouroboros`), so neither the reference wave nor the upstream fixes had
decided it; the red windows leg of run 33555971481 is what surfaced it. The
decision is the narrow one — normalize at the read, not a `.gitattributes`
checkout policy — because the assertion is about source TEXT, and only the two
reads that feed source-text regexes are touched.

## Open items on this row

1. **`tests/test_registry_core.py` still pins `os.name == "nt"`.** The accepted
   2026-08-30 audit item (`~/.claude/plans/v7next/V7NEXT_PLAN.md`; owner
   requirements archive, «os.name→alias-условие в route-пине») asks for the pin
   to name the actual alias condition instead of the platform.
   Class 5 above is landed; this is a follow-up on its expression, not on its
   decision. Still open on the integration tip (`tests/test_registry_core.py`
   line 813 reads `os.name`); the repin is being landed by the smalls lane of
   the stage-2 fix wave, so this item is assigned, not merely listed.
2. **Green windows legs exist, and the whole matrix is green.** This item used to
   read «no green windows leg exists yet on any frozen SHA», which the run table
   below has contradicted since run 33568728122 (`f5a94675`, first green Windows
   leg) and, for the full matrix, since run 33569841899 (`8b27b507`) — four
   consecutive first-attempt-green 3-OS matrices are logged there (33569841899,
   33570328266, 33571681398, 33572515529), and on the later branch tips two more
   whose Windows `full-test` leg went green only on a RERUN (33579445704 on
   `1072a317`, 33624546416 on `ac17fa03` — attempt-1 detail under the run table)
   plus two more first-attempt greens: 33626834806 on `43dcc1d2`, where all three
   `full-test` legs passed on attempt 1, and 33644668074 on the sync #3 merge
   `f4abe0a5`, green on every job on attempt 1. Class 17 is decided, fixed and
   proved. What is genuinely open is FRESHNESS, not colour: 33644668074 is the
   newest verdict (read 2026-09-02 15:00Z); the tips after it (C6 merge, the
   stage-2 fix lanes) await their own dispatch on the release candidate. Second,
   narrower open point: the attempt-1 Windows failure of 33579445704 was two
   failures, and only one is now rooted — the observability copy-back race is
   fixed by `626b48b7` (owner 15 = B answered O3 with «fix now»), while the
   `tests/test_preflight_runner` xdist worker timeout still has no landed
   root-cause fix and is carried below as an intermittent class.

## 3-OS matrix runs

The row's re-prove needs a green full-test 3-OS matrix on a frozen branch SHA
(`gh workflow run CI --ref <branch>` → `ci.yml` full-test). Runs so far:

| run id | SHA | ubuntu | macos | windows | verdict |
|---|---|---|---|---|---|
| [33555971481](https://github.com/razzant/ouroboros/actions/runs/33555971481) | `9a28e58f` (branch `ouroboros_v7next`, 2026-09-01 20:33Z) | green | green | **RED** (class 17) | not a re-prove |
| [33563498919](https://github.com/razzant/ouroboros/actions/runs/33563498919) | `196438c9` (carries the class-17 fix `a0b35fcd`) | green | green | **RED** — 16 tests, nine platform classes (chmod probes, open-file unlink, signal.alarm, separators, shlex backslashes, cp1252, simulated O_BINARY, byte-exact writer vs os.linesep, host-only signal names) | not a re-prove; classes fixed in 20afdbb7..e0aee1ac |
| [33567328254](https://github.com/razzant/ouroboros/actions/runs/33567328254) | `9754cc95` | green | **RED** (two scheduler-sensitive pins, hardened in 455c9a1e) | **RED** (one: the listing pin's own fold of the JSON-escaped separator, fixed d0d52677) | not a re-prove |
| [33568284121](https://github.com/razzant/ouroboros/actions/runs/33568284121) | `7c93e8b7` | green | green | **RED** (the same single pin) | not a re-prove |
| [33568728122](https://github.com/razzant/ouroboros/actions/runs/33568728122) | `f5a94675` | green | **RED** (custody pin, gated in 8b27b507) | **green** — first green Windows leg | not a re-prove (macOS) |
| [33569841899](https://github.com/razzant/ouroboros/actions/runs/33569841899) | `8b27b507` | green | green | green | **RE-PROVE** — full matrix green |
| [33570328266](https://github.com/razzant/ouroboros/actions/runs/33570328266) | `285ab66d` | green | green | green | re-prove holds |
| [33571681398](https://github.com/razzant/ouroboros/actions/runs/33571681398) | `9238cc2d` | green | green | green | re-prove holds |
| [33572515529](https://github.com/razzant/ouroboros/actions/runs/33572515529) | `c0029d45` | green | green | green | re-prove holds |
| [33574822693](https://github.com/razzant/ouroboros/actions/runs/33574822693) | `d21806d8` (first run of the scheduled `system-e2e-mock` job on dispatch) | success | success | **failure** | **not a re-prove** — verdict read 2026-09-02: `full-test (windows-latest)` red, the run never rerun (`run_attempt` 1, run conclusion `failure`); every other job green, the new `system-e2e-mock` included. The failing subtests are not attributable from here — job logs need repo-admin rights — so no class is claimed for it; the two tips after it (1072a317, ac17fa03) went green on the Windows leg only on rerun; the first later first-attempt-green Windows leg is 43dcc1d2 (33626834806) |
| [33579445704](https://github.com/razzant/ouroboros/actions/runs/33579445704) | `1072a317` | success | success | success **on rerun** | re-prove holds **on attempt 2, not on attempt 1** — `full-test (windows-latest)` failed first and was rerun green; every other job green on attempt 1. Named cause and its open residual below |
| [33624546416](https://github.com/razzant/ouroboros/actions/runs/33624546416) | `ac17fa03` | success | success | success **on rerun** | re-prove holds **on attempt 2, not on attempt 1** — `full-test (windows-latest)` failed first and was rerun green; every other job green on attempt 1. A code fix followed (`43dcc1d2`), so this one is rooted, not intermittent |
| [33626834806](https://github.com/razzant/ouroboros/actions/runs/33626834806) | `43dcc1d2` | green | green | green | re-prove holds — full-test 3-OS green on the FIRST attempt; the separate `system-e2e-mock` job was red 2/57 on attempt 1 and green on rerun (see below) |
| [33644668074](https://github.com/razzant/ouroboros/actions/runs/33644668074) | `f4abe0a5` (the sync #3 merge) | success | success | success | **green** — full-test on all three OS, `system-e2e-mock` green on the first attempt, integration-test green (verdict read 2026-09-02 15:00Z) |
| [33654743857](https://github.com/razzant/ouroboros/actions/runs/33654743857) | `bf8b6549` (C6 lane merged; first Windows execution of its code) | success | success | **failure** — 12 tests: LockFileEx mandatory-lock class, five POSIX-protocol pins, `signal.SIGKILL` spelling (packet §10 addendum; abea91ec) | not a re-prove |
| [33658408570](https://github.com/razzant/ouroboros/actions/runs/33658408570) | `f2f014bc` | success | **failure** (executor rule of 504bb20c: hash-less child leaked on macOS `ps`; fixed 35b82db0) | **failure** — compaction fixture class (name tier cannot compact; suite skipped at abe93702) + the two concurrency tests below | not a re-prove |
| [33658966160](https://github.com/razzant/ouroboros/actions/runs/33658966160) | `5ae7f357` (7.0.0 carriers) | success | success | **failure** — same compaction fixture class + the two concurrency tests + three probe pins | not a re-prove |
| [33661022574](https://github.com/razzant/ouroboros/actions/runs/33661022574) | `abe93702` | **failure** (platform guard: `os.kill` outside platform_layer) | failure (same) | **failure** — chat-append concurrency test, three probe pins, platform guard | not a re-prove |
| [33663258606](https://github.com/razzant/ouroboros/actions/runs/33663258606) | `35b82db0` | success | success | **failure** — exactly the two concurrency tests: the name-tier release orphan (LEDGER «From the Windows CI matrix on 35b82db0»; class fix `_unlink_lock_path`) | not a re-prove — `system-e2e-mock`, integration, all smokes green |
| [33668287491](https://github.com/razzant/ouroboros/actions/runs/33668287491) | `d0bb839e` (name-tier release retry) | success | success | **failure** — the two concurrency tests now GREEN (class closed); red = the new POSIX release pin importing `fcntl` on Windows (skipif, next row) and `test_task_result_monotonic::test_proactive_namer_late_settlement_refreshes_cost_without_late_name` (0.0 == 0.25 inside its 2 s poll window; first time on any Windows leg, green on the next run with identical runtime code — intermittent, unrooted: the retry cannot lengthen an uncontended release, and any unlink refusal orphaned the lock before this commit, failing the same test the same way) | not a re-prove |
| [33669250620](https://github.com/razzant/ouroboros/actions/runs/33669250620) | `4c7c5aed` (3.11+ pin, ledger precision) | success | success | **failure** — exactly one: the POSIX release pin's `fcntl` import (test shape; skipif in the next SHA); concurrency tests green, timing test green | not a re-prove — `system-e2e-mock`, integration, all smokes green |

The windows failure in run 33555971481, read from the run's own log rather than
from its exit code: two subtests of `web/tests/chat_plain_system_rows.test.js`
— «render arm order and enhancement guard are pinned in source» and «chat
bubble heading clamp is scoped in style.css». That is class 17 above, not a
regression of anything this campaign re-applied, and the same run's `full-test
(ubuntu-latest)` and `full-test (macos-latest)` were green. Also red in that
run and outside this row's scope: `ui-smoke` and `skill-smoke (ubuntu-latest)`.

Run 33563498919 (`196438c9`, `a0b35fcd` an ancestor — both are ancestors of
every later tip of this branch) cleared class 17 and surfaced nine further
platform classes on Windows (see the run table); they were fixed in
20afdbb7..e0aee1ac and the first green Windows leg is run 33568728122
(`f5a94675`). The full 3-OS matrix went green on run 33569841899
(`8b27b507`) and held on every later run in the table, which is the
re-prove the ADOPTION row R-WINWAVE cites; the per-class decisions above
stand as recorded.

The two red `system-e2e-mock` subtests on attempt 1 of run 33626834806
(`43dcc1d2`) were not platform classes and are not registry rows: both were
races inside the mock lane's own scaffolding — the `/proc`-environ scan of
`pids_with_env_value` and an S22 wait that assumed its window was wide enough
under CI load. Neither is a cross-OS class; both belong to the E2E lane's own
ledger rather than to this row.

The `/proc`-environ half is no longer «lane flakiness … disclosed»: it is
**rooted and fixed by `626b48b7`** (owner batch №13 item 15 = B). The cause is
not «a process can exit between the listing and the read» but the post-exec
window — `Popen` returns once the exec SUCCEEDED (the CLOEXEC error pipe closes
inside `execve`), while the kernel publishes the new image's
`env_start`/`env_end` later in that same path, so a read landing there sees an
EMPTY environ for a live, correctly marked child (the same shape failed again as
`assert 3898 in []` on run 33671108287). The harness now separates the positive
oracle (`wait_pid_env_value`, a bounded poll of THE ONE pid) from the no-orphans
postcondition (`pids_with_env_value`, still a single scan), and the window is
pinned deterministically through their shared read seam. The S22 wait remains a
disclosed lane-timing observation.

### The two rerun-greens, per row

Two «re-prove holds» rows above were reruns, not first-attempt greens, and each
is recorded with its attempt-1 outcome, its attempt-2 outcome, the named cause,
and whether a code fix followed.

- **33579445704 (`1072a317`).** Attempt 1: `full-test (windows-latest)`
  **failure**, every other job green. Attempt 2 (rerun of the failed job):
  green. Named cause, **operator-read**: two Windows failures —
  `tests/test_phase3c_observability_gc` on its copy-back step (intermittent) and
  `tests/test_preflight_runner` on an xdist worker timeout. The copy-back half is
  now **rooted and fixed by `626b48b7`** (owner batch №13 item 15 = B, which
  answered the open **O3** question with «fix now»): two concurrent copy-backs
  promote the same content-addressed source handle, and on Windows the loser's
  `os.replace` over a destination the winner or a verifying reader holds open is a
  sharing violation, so the loser published an INCOMPLETE promotion while the
  winner published a complete one. The store is now write-once and the promotion
  judges by its postcondition rather than by authorship of the write (diagnosis
  and red-first pins: docs/v7next/LEDGER_CORRECTIONS.md, «From the
  delegation-mutation and races lane»). The `tests/test_preflight_runner` xdist
  timeout had **no code fix** and stands as **intermittent, unrooted**. Either
  way this SHA's Windows leg is a rerun-green and must not be cited as a
  first-attempt green.
- **33624546416 (`ac17fa03`).** Attempt 1: `full-test (windows-latest)`
  **failure**, every other job green. Attempt 2: green. Named cause,
  operator-read and independently corroborated by the fix commit's own message:
  the session-engine horizon is the ceiling of the seconds left to the deadline,
  and a sub-second remainder on the coarse Windows clock made the pin read 301
  for a deadline 300 s away. **A code fix followed** — `43dcc1d2` («tests: the
  session-engine horizon pin tolerates the coarse-clock ceiling (Windows
  full-test)», `tests/test_review_agent_session_route.py`, naming this run id) —
  so this class is **rooted and closed**, and the next run on `43dcc1d2`
  (33626834806) has all three `full-test` legs green on attempt 1.

How the attempt structure was read: read-only against the public GitHub API —
`GET /repos/razzant/ouroboros/actions/runs/<id>` for `run_attempt` and
`conclusion`, and `.../attempts/<n>/jobs` for the per-job conclusions. `gh` is
installed on this host but not authenticated and nobody logged in for this read;
unauthenticated requests answered both endpoints because the repository is
public. Job **logs** are not readable that way (403, «Must have admin rights to
Repository»), and the check-run annotations carry only «Process completed with
exit code 1» — which is why the failing test names above are recorded as
operator-read facts rather than as facts re-derived here. Attempt counts as
read: 33569841899, 33570328266, 33571681398, 33572515529 and 33644668074 are
`run_attempt` 1 with conclusion `success`; 33579445704, 33624546416 and
33626834806 are `run_attempt` 2; 33574822693 is `run_attempt` 1 with conclusion
`failure`.

(Written on the adoption lane's own worktree, where `a0b35fcd` and `196438c9`
were not yet ancestors; on the integrated branch both are, and the run table
above carries the outcomes.)
