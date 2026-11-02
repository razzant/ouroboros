# ci-macos-native: two bounded native cases of the unified Update on GitHub-hosted macOS arm64 runners

Test equipment for the qualification of the unified Update, **not product**. Each case runs on its own fresh runner:
- **A (`b1a`)**: the owner's runtime venv was made from the 7.7.0 app's own Python;
- **b1e**: the owner's runtime venv was made from an external, relocated copy of that Python, the owner's chosen
  interpreter.

In both cases:
1. The official 7.7.0 app, copied from its pinned public DMG, is the owner's existing installation, opened through
   LaunchServices and quit through its own Apple Event.
2. It serves product C from a checkout composed with the owner's local work (H, I, W and U), on that venv.
3. C's own Update prepares the change (its real checks, unstubbed) and activates it: C stays the body, and the official
   7.8.0 app takes the place of 7.7.0 at the installed path.
4. The app is quit, reopened cold, and checked for its launch roots, runtime, profile and the owner's local function.

Nothing here changes C or the meaning of any test. A case's exit status and its uploads never stand in for a verdict:
`RESULT.txt` and `claims.tsv` carry the verdicts, and every claim is PASS, FAIL, UNKNOWN, REFUSED, NOT_RUN or RECORDED,
each with a short name.

## Files (the public inventory)
| file | role |
|---|---|
| `workflow/qualification-update-native-macos.yml` | the workflow; placed byte-identical at `.github/workflows/qualification-update-native-macos.yml` |
| `native/run-native.sh` | the runner of one case: setup gates, the case's qb1 steps through `capture.py`, the overlay claims, finish, result |
| `native/wheelhouse.py` | derives and checks the wheel pins; reports a staged wheel folder (standard library only) |
| `native/wheelhouse.pins` | the declared wheel set: one PyPI wheel per applicable lock line, pinned by C's uv.lock |
| `native/equipment/guest/qb1.sh` | the qb1 guest steps: the hosted b1s probe's copy with three repairs, its readiness reader adapted, the attempt capture, the test-evidence copy and the lab budget proof added (below) |
| `native/equipment/guest/procwatch.py` | process and exit observer, byte-identical to the b1s probe's copy |
| `native/equipment/guest/venv_lab.py` | the owner's earlier venv preparation through C's own functions; docstring-only change |
| `native/equipment/host/mkbundle.sh` | C's bundle and bind file, byte-identical to the b1s probe's copy |
| `native/equipment/scripts/capture.py` | bounded, create-only receipts of each step, unchanged |
| `native/OVERLAY-SHA256` | the equipment and wheel-pin digests, checked before anything else |
| `OVERLAY-MANIFEST.json` | provenance: each file's original, its digest and its change |
| `README.md` | this file |
| `SHA256SUMS` | every file above except itself, checked first |

Only these files, and the workflow copy at its path, are published. Anything else in a working copy of this folder is
local and is not part of the execution commit.

`run-native.sh` is the b1s probe's `run-probe.sh` adapted to these two cases. Its changes:
- the case's own steps (`install`, `candsrc`, `extpy` for b1e, `venvprep`, `prepare`, `activate`) instead of the stock
  updater's;
- the 7.7.0 inputs, the public wheel folder and the existing-copy rule (below);
- new claims `existing-copy`, `release-capsule-7.7.0`, `wheel-pins`, `wheelhouse` and `native-target`;
- `identity-*` checks the runtime venv: argv[0] and the interpreter's location. A venv run's user base is recorded,
  not judged, because a venv without system site packages turns the user site off;
- a step whose whole window does not fit the remaining budget is not started: UNKNOWN `time_budget`, and no window
  is ever shortened;
- new claims `work-pre` and `work-cold`: compose's own record of the owner's work, judged after the first composed
  launch and after the cold reopen;
- `profile-survives` compares the canary key and its whole value exactly, and an unread observation stays UNKNOWN;
- an exit 2 without a verdict is REFUSED only with qb1's own pre-effect refusal line; otherwise it is UNKNOWN
  `step_aborted`.

`qb1.sh` is the b1s probe's copy (`aa2f7a97…`) with three repairs and two adaptations (the readiness reader, then the
attempt capture and the end of a preparation), and the chain's other three waits bounded in elapsed time, last:
- **Exit 2.** `die` refuses before its step's effect, and its line on stderr (`qb1: refused before effect: …`) is
  the explicit receipt of that. `abort` (`qb1: aborted after its effects began: …`) takes the six failures that come
  after effects began: `install`'s four after it made the run root, and the two mount failures after a mount folder
  was made. Neither writes a verdict.
- **The profile record.** `profile_store` keeps each read's own status, and its data and diagnostics never share a
  line:
  - `file_canary=` is the sha256, `absent` or `unread`;
  - `databases=` is the discovery's count, or `unread`;
  - each database, still queried on a copy, records `copy=` and `query=`;
  - each row is `item|typeof(key)|<key is this run's canary key, compared by SQLite: 1 or 0>|hex(key)|byte
    length|hex(value)`;
  - then `localstorage_reads=` and, last, `PROFILE_READS=complete` or `incomplete failed=…`.
  A missing file or store is a completed read; any other error leaves that part unread.
- **The byte checks of `prepare` and `activate`.** A completed change of the owner's work in the step's own two
  records is never hidden: not by an incomplete record, and not by the step's other outcome.
  - Complete records keep their own rules: `prepare` the whole record, `activate` the selected work (its HEAD's
    ancestry stays a later check).
  - In an incomplete record, a witness (H, I, W or U) that both records read completely and that differs counts. The
    detail names the unread rest by the record's own `READS` line.
  - In the record, W is `absent` only when its lookup answered that nothing is there. When the shell's tests find
    no `OWNER_NOTES.md`, one `lstat` by the observer interpreter decides: `ENOENT` (a missing file or checkout) or
    `ENOTDIR` (a component on the way is not a directory) is a completed read. Any other error (no search
    permission, an I/O error, …), an entry found after all, or a reader that did not run or answer leaves W
    `unread`, with a `W_lookup=` line saying why, named in `READS` like H, I and U, and never compared. `obs`'s
    records for `work-pre` and `work-cold` follow the same rule.
  - Such a change is FAIL `local_work_changed`. When the step's status, adoption, server or process check came out
    otherwise (not settled, not closed, unreadable, refused, …), that outcome's own explanation is kept after
    "beside it:".
  - With no such change, every outcome is as before: incomplete records stay UNKNOWN `local_work_unread`, and an
    unfinished step is not read as a loss. Nothing is inferred from a failed read, and no snapshot of another step is
    compared.

- **The readiness reader (`prepare`'s receipt) reads the one-budget record contract** of the Unified Update candidate's
  `supervisor/update_checks.py` at C13 `cbf3e1fa…` (sha256 `6fbd74f0…`, tree `39be0f44…`). C12's `214a9ccb…` (tree
  `ae7a27e1…`, C13's sole parent) wrote the same record shapes ("Bound to C13's producer" below); C10's `456fb394…`
  wrote the same records without the node lane (below); it wrote the records of C7's `34aa0f51…`, tree `20789d15…`, the same file
  from C6 `015ce40b…` through C9, and differed from it only in its process root's name prefix and a docstring; the
  reader before `34aa0f51…` followed `d5220c74…`, staged index tree `39267fcf…`, and before that `48f93667…`, index tree
  `5814671d…`:
  - **What ran** is the JUnit reports' own membership: unique (file, node id) pairs, so a node pytest records twice
    (its call and its teardown both failed) counts once. It is never compared with the number of target lines. The full
    tier gives the lanes one suite directory (`tests/`), the selected tier its modules, a comparison on W or T complete
    node ids. Every executed case must lie inside them; a given module or node without a case, and the executed file
    count beside `file_count`, are recorded, not judged.
  - **Deadlines are the records' own.** The checks' `budget_seconds` is the one budget of W′, W and T. Each run's
    `run.json` `deadline_seconds` (what remained of it when the run began) may not exceed what the run before it left.
    A lane's exit 124 is its deadline only when the run reached its recorded deadline or its materialization ran out
    of time; otherwise the receipt names it. No constant stands in for the product's deadline.
  - Each run's `materialization` output, `installation_user_base`, and the new run's `node` and `user_site` are read
    against the checks; T's presence read (`target_run.presence`) is copied with its input. A ready claim also needs no
    missing `prerequisite`, and already failing cases need `baseline.excuses`.
  - `budget.txt` states the budget and each run's end on one line, and the step's detail carries it beside the checks'
    own `unverified` text. A missing Node or user site is FAIL `prerequisite_unverified` (the claim stays FAIL, as
    before; only its name is new).
  - `prepare` also records which bundled or checkout Node files exist (`node-candidates.txt`), not judged; nothing is
    supplied.
  - **The node lane (C12).** C12's checks run the browser-module suite (`web/tests/*.test.js`, `preflight_node.node_suite`:
    `node --test --test-concurrency=N <web-relative files>` from `web/`, the runs' environment without `NODE_OPTIONS`)
    as lane `node`, first, in the new run alone, within the same one budget, and only with the Node `checks.tests.node`
    names (`_node`, now in either tier). The comparison run on W and the target run on T have no node lane. The lane
    records `files`, not `cases`, and writes no JUnit report, so none is read or required for it: what ran in it and
    which files failed are only in its own output (`node.stdout.txt`, `node.stderr.txt`), copied and not parsed. It is
    complete when it exited 0 or 1 with its tree proven gone; exit 1 is its measured failure: `run.json` `node_failed`,
    which the reader recomputes from the lane and compares, and `web/tests/*.test.js` in `blocking`, never excused,
    compared or attributed. The reader requires the lanes `node`, `parallel`, `serial` exactly when the checks name a
    Node (else `parallel`, `serial`), the node lane's argv to be that Node over `tests/*.test.js` files with its `files`
    count, and the checks' lane summary to name its `files`; its `cwd` is recorded, not judged. A ready claim beside a
    failed suite is missing evidence; a failed suite is named in `failures` (`new_run_node_suite_failed`), so the
    receipt is never green with it. A red suite stays a FAIL or UNKNOWN as C's own outcome says: it is never waived,
    and no JUnit is fabricated for it. `budget.txt` names the lane's exit, files and seconds.
  - **Lane containment.** A lane whose process tree the producer could not prove gone carries `containment` (in the
    checks' `lanes` and in `run.json`, only then). The reader counts such a run incomplete, as the producer does, and
    compares that field between the checks and `run.json`. `budget.txt` names the lanes (`not proven gone: …`).
    `process_root` and `process_root_release` are kept in the checks as found, not judged; the release is absent when
    the checks raised.
  - **A lane that exited 1 though none of its cases failed.** C6's producer marks such a lane `no_failing_case`
    (its session failed outside its tests; in the checks' `lanes` and in `run.json`, only then) and counts the run
    incomplete, even when every JUnit case passed. The reader recomputes that flag from the lane's own report, counts
    the run incomplete the same way, compares the flag between the checks and `run.json`, and names such lanes in
    `budget.txt` (`exit 1 though none of its cases failed: …`). The lane's cases still count as what ran.
  - **Bound to C13's producer.** C4's own `update_checks.py` wrote the earlier shape (no `budget_seconds`,
    `deadline_seconds` or `materialization`), so on C4 a ready candidate's evidence was incomplete and `prepare` gave
    UNKNOWN. C and CT are now C13, and `PRODUCER` is C13's `update_checks.py` (`6fbd74f0…`). C12's `214a9ccb…` wrote the
    same record shapes; C10's `456fb394…` wrote no
    node lane and no `node_failed`, so this reader counts a C10-shaped new run incomplete; C6 through C9 held
    `34aa0f51…`, which differed from C10's only in the process root's name prefix and a docstring.
    Any other producer bytes need another rebind.
    C13 changes how each pytest lane is started, not what `run.json` records: each lane's argv gains `--maxfail=0` and
    `-p ouroboros_update_lane_<this run folder's name>`, and its environment's `PYTHONPATH` (also what `run.json`
    `pythonpath` records) begins with the run folder, where the checks write that plugin file (`<name>.py`) beside the
    reports. The plugin makes a lane whose configuration selects `--setup-only`/`--setup-plan`, or sets `maxfail` after the
    lane's own `--maxfail=0`, exit 4 (the xdist lane 2), measuring nothing; that is a lane exit this reader already counts
    incomplete (only 0 or 1 with cases, or 5 with none, is complete) and names, never a pass. The reader judges, of a
    pytest lane's argv, only that its `@FILE` argument is the one `run.json` names, and it reads no `pythonpath`; the
    plugin file is not one of the files it copies (`run.json`, `targets.txt`, the lanes' reports and outputs, the
    materialization's output), so it stays in the candidate's scratch outside this capture. No record the reader reads
    changed shape; this was read from C13's source and has no actual sample.
  - **The tests' own incremental records (`test_evidence`).** The checks name one durable folder for the tests'
    records (`checks.tests.test_evidence`, `test-evidence-*` beside the run folders in the candidate's `update-checks`
    scratch). Each pytest controller started there writes one `pytest-<session root>` folder through
    `tests/ci_evidence.py`: `events.jsonl`, its unbuffered journal (start, report and finish rows, so it names what was
    in flight and which reports failed when a session is killed before its export), and `results.json`, its final
    export, absent for a killed session. The reader copies each session folder's two files into
    `readiness/test_evidence/<name>/`, regular files only, never following a link, to the length each had when opened,
    within caps (64 MiB a file, 384 MiB in all, 200 sessions); every other entry (nested gate `run-*` folders, `ui`,
    ...) is listed by name and type, not copied. `readiness.json` `test_evidence` keeps the outer identity (candidate,
    process root, each run's folder and its started lanes) and, per session, the rows read, what was in flight at the
    end of its journal, its failed reports, the export's exit and counts.
    - **Outer or nested is not recorded by the producer.** It links no lane to its session folder, and nested pytest
      sessions started by the tests inherit the same folder. A session whose own node ids include one outside every
      run's targets is `not_an_outer_lane_session` (a lane collects only its `@targets.txt`); any other session with
      cases is `unattributed`, with the runs whose targets contain all its node ids. Neither label proves that a
      session is outer or nested. No session is chosen by its name, order or time.
    - A folder that does not lie beside the run folders the checks name is not this check's folder: it is not read,
      and its gap names it and those run folders. The run folders themselves are still copied.
    - A missing reference, a folder retired with the candidate's scratch, a link, an absent, unread or vanished
      selected file (`events.jsonl` or `results.json`; a killed session's absent export is one), a file or session
      past a cap, an unparsable row, an unsupported `results.json` shape, or a report row that the producer's own
      reader (`tests/ci_verdicts.py` `valid`, `1dc74a65…`, byte-identical at C10, C12 and C13, as are `tests/ci_evidence.py`
      and `tests/conftest.py`) does not accept is a named gap in `test-evidence.txt`, never a result.
      Such rows stay in the copied bytes and are counted and named. The export's summary counts only the rows that
      reader accepts. `read=complete` covers the two selected files of each session only, not the folder's whole
      state or what the records mean. That line goes into the step's detail; it decides nothing, and the readiness
      evidence rules above are unchanged.
    - What a journal shows is what that session recorded, not why. The earlier run 41310222116's failed case is not
      recovered by this: its records were never copied.
  - **The lab's readiness budget (13000 s, this lab only).** `run-native.sh` hands
    `OUROBOROS_PREFLIGHT_TIMEOUT_SEC=13000` to open-2 alone, through qb1's existing `ENV=VALUE` argument (not open-1,
    not reopen; no Settings write; C's default 7200 s is unchanged). C reads it in the serving process
    (`update_checks._budget`, `preflight_runner._resolve_preflight_timeout`). Before `prepare` sends anything, the
    one serving server, by its whole `ps` row, is read with `procwatch.py snap` (its exec environment,
    `KERN_PROCARGS2`): the pair must be there with exactly that value, in plain or by its recorded sha256 prefix,
    and the serving rows must be the same after the read (`served-env*.txt`). An absent, different or unread value,
    no single server, or rows that changed refuse the step before its effect (REFUSED, nothing sent), which stops the
    chain. A later change inside the process is not visible to that read; the checks' own record is: the receipt
    needs `checks.tests.budget_seconds` exactly 13000. Another value, none, or one that is not a number is missing
    evidence, named in `evidence.txt` and `budget.txt`: a PASS becomes UNKNOWN, and a FAIL stays FAIL. Without the
    pair, the budget rules are unchanged. Each run's own `deadline_seconds` may be smaller (what remained). The test lanes' scrubbed
    environment is C's own and unchanged (C12's node lane gets the same environment without `NODE_OPTIONS`). `identity-*` records the name where a server holds it, judging nothing.

- **The attempt capture and the end of a preparation.** Runs 41268055234 (b1a, b1e) answered 202, read `preparing` at
  +1 s and `none` from +6 s or +7 s, then waited out the whole 9600 s. The server log after the request was never kept,
  because the only later copy was `obs-post`, which a stopped chain does not reach. That traceback cannot be recovered.
  - **`attempt_capture`** copies this attempt's server logs and raw records into `<step>/attempt/`: `logs/`, `records/`
    (`state/private_copies/*.json`), `body_adoption/` and `state/` (`update_activation.json`, `process_ledger.jsonl`,
    `server_process.json`, `owned_processes.json`). It reads only this run's own data root and copies regular files
    only, naming a link without following it. `capture.txt` gives each file's digest, `records_count=N` (an upload drops
    an empty folder, so zero is written down) and the top-level names in the declared work root. Its last line is
    `CAPTURE=complete` or `CAPTURE=incomplete failed=…`. A source is absent only when its lookup answered that nothing
    is there. The live status is saved beside it, unjudged.
  - **When it runs:** at every outcome of `prepare`, after its last byte check; and in the finish of a stopped chain,
    before the quit and after it (`attempt-stopped`, `attempt-after-quit`). Those two claims say only whether the reads
    completed. A step stopped by a signal or lost with its runner gets no finish, as before.
  - **The following is bounded in elapsed time.** `prepare`'s SECS is measured from the step's own start (its clock is
    read first, before any read or request): a status read begins only while the step is younger than SECS, and the
    reads are 5 s apart (`sleep 5` after each). So the following ends at most one read (each bounded: `curl -m 60` and
    `-m 2`, two `ps`, one reader) and its 5 s pause after SECS, and what the client's window leaves beyond SECS stays
    for the reads and copies after it. Before this, SECS/5 reads were counted; each read also took its own time, so the
    following outlived SECS (C10 run 41310222116's `prepare` timelines: 1247 and 1234 reads over 7618 s and 7686 s,
    6.11 and 6.23 s a read, so 3080 reads would take about 18,830 and 19,200 s, past the 17,500 s window, and the final
    capture would be lost). `result.txt` records `reads=` (what fit) and
    `follow_limit_s=`; the count is never the bound.
  - **`prep_ended`** ends the following once a preparation that was witnessed reads as gone: 12 status reads in a row,
    at least 5 s apart, with HTTP 200, state `none`, no candidate, no refusal of this request, no `pip`, `pytest` or
    `git` child of the run, and the same serving server. Witnessed means its slot read `preparing`, or C answered 202
    (`update_prepare.admit` holds the slot before that answer).
  - **Its verdict is UNKNOWN `ended_without_candidate`:** the current status names no candidate. C's status stops
    naming a discarded one, so a candidate of this request that the status named earlier in the following is named in
    the detail, with the read and state it was last seen at; whether it was discarded, and why, stays unestablished.
    A status read counts only when it answers 200 and gives a state C's projection declares. Any other read (another
    answer, a body that is no status, C's own `unknown`) establishes no presence, absence or settlement, whatever its
    file holds: its code stays in `timeline.txt` (`status_http`) and `result.txt`, its bytes in `kept/` and
    `status-final.json`. Wherever such a candidate is named (beside `serving_changed` after a following that did not
    settle, and in `not_settled`), the detail gives the last read that named it and says whether the last read still
    names it or names it no more; a last read that does not count establishes neither, and the detail says so and
    gives the last read that did count, with its state.
    Why it ended and what it did first stay unestablished, and nothing is sent again. A lost admission answer with
    nothing witnessed, an unread status, a moved server, a running child or a candidate named now keeps the old
    following and `not_settled`.
  - **A whole serving-server mismatch is FAIL `serving_changed` however the following stopped.** The quiet reads
    compare PID lists only. When the server rows before the request and the process table after it were both read and
    the rows differ (`lstart` included), the step is FAIL, also after an early end or with no settled candidate. An
    unidentified server before or an unread process table after keeps the UNKNOWN and is named in its detail. A change
    of the owner's H/I/W/U stays first, as before.
- **The chain's other waits are bounded in elapsed time too** (`openw`, `quit`, `activate`; `wait_for` and `restart`
  are not in the chain and keep their loops). Each counted its SECS as reads with a 1 s pause, and each read takes its
  own time (`curl -m 2`, several `ps`, the observer interpreter's readers), so a wait could outlive its SECS and the
  window and lose its receipt. Now SECS is measured from the request (the clock read just before the open, the quit
  or the activation POST): a read begins only while the request is younger than SECS, so a wait ends at most one read
  and its pause after SECS (`activate` after an `adopted` closure: below); `quit` always makes its first read.
  `result.txt` records the reads that fit and when the outcome was read (`looks=`/`serving_seen_s=`,
  `reads=`/`no_process_read_s=`/`cold_read_s=`, `reads=`/`closed_read_s=`/`late_closure=`, and `limit_s=`); the count
  is never the bound. A success counts only when the read that saw it ended within SECS (whole seconds): serving first
  seen later is UNKNOWN `not_serving`; a cold state whose whole read (`cold_state`: its `ps` and every `lsof`) ended
  only after SECS is UNKNOWN (`quit_absence_unknown`, unless the Apple Event's own error code names it), also when the
  loop's own read found no process in time, while a process or listener it finds stays FAIL and a read it could not
  complete stays UNKNOWN, whenever it ended; an `adopted` closure first read after SECS is named in the detail and not
  counted (UNKNOWN `adoption_not_closed`). A closure otherwise is FAIL whenever it is read, and a change of the
  owner's H/I/W/U stays first. After an `adopted` closure the settling is elapsed time from the read that found it:
  serving is accepted only from reads begun at least 20 s after it, it ends at 300 s after it, and it runs past SECS
  only to complete those 20 s. SECS, the windows, every other PASS rule and reason name are unchanged. A read's own
  commands (`ps`, `lsof`, file reads) have no timeout of their own here; the window stays their outer bound.

## Pins
- **Product C (C13):** `cbf3e1fa619d42474ebd3145cc57c812c5c280ff`, tree `39be0f449b5afd29d9d0761760403243513287cb`.
  - C13's sole parent is C12 `a2424ed19abc3886132315889277910ed3472b60` (tree `ae7a27e1…`); C12 to C13 changes three paths (read-only Git readback):
    `supervisor/update_checks.py` (`6fbd74f0…`, from `214a9ccb…`: the per-run pytest mode plugin above, `--maxfail=0`,
    the exit 4 and 2 meanings and their `_incomplete` text), `tests/test_update_checks.py` and `docs/CLIENT_CONTRACT.md`
    (neither is read here). Every other path is byte-identical to C12's, so the C10-to-C12 account below, the tags and
    the 348 `web/tests/*.test.js` files hold at C13; both locks (`68e739bd…`, `8c2b87781…`) and `VERSION` (7.9.0) are
    unchanged.
  - (Earlier bound, C12.) C12's sole parent is C11 `4328d1c23dd212686ba4eaf4386bf906b0672497` (tree `8d9dcd88…`), which composes C10 with
    the upstream `fee36369ba460d572a58861766d676128534f23c` (parents C10 and `fee36369…`). C10 is an ancestor of C12.
    C10 to C12 changes 126 paths (111 modified, 15 added); for this equipment:
    - `supervisor/update_checks.py` (`214a9ccb…`): the node lane above; `ouroboros/preflight_node.py` (`4be09350…`): its
      `node_suite`; `ouroboros/test_selection.py` (`9857d74c…`): the Update's own files join the apparatus and cutover
      sets, so a change to them keeps the full tier (here the tier is full already); `launcher.py`
      (`00d7a601…`) and `ouroboros/body_switch.py` (`31d04a52…`) change (C12: the Panic handover repair).
    - Byte-identical to C10's (read-only Git readback): `supervisor/update_prepare.py` (`78d39f97…`),
      `supervisor/update_native.py`, `ouroboros/gateway/update_activation.py`, `ouroboros/body_adoption.py`,
      `supervisor/update_recovery.py`, `ouroboros/runtime_venv.py`, `ouroboros/native_package.py`,
      `ouroboros/app_install.py`, `ouroboros/launch_roots.py`, `ouroboros/config.py`, `ouroboros/test_environment.py`,
      `ouroboros/test_parallelism.py`, `ouroboros/test_evidence.py`, `ouroboros/node_runtime.py`, `server.py`,
      `pyproject.toml`, `tests/conftest.py`, `tests/ci_evidence.py`, `tests/ci_verdicts.py`, both locks and `VERSION`
      (7.9.0). The official tags below are ancestors of C12 and of C13; it holds 348 `web/tests/*.test.js` files, as C10 did.
    - Read from source, unverified natively: how the changed `launcher.py` and `body_switch.py` behave in the
      activation and the cold reopen; no reader here depends on their record shapes beyond what C10's did.
  - The earlier pin, C10 `25fc93a8e3a406c5d79c5ab8004b6dda7f36b19f`, tree `f7d60f815b8b21ade58411a9cc334bfc80af4639`.
    C10 is C9 plus one commit, sole parent C9. It changes 4 paths: `supervisor/update_checks.py` (`456fb394…`),
    `tests/test_update_checks.py`, `tests/test_storage_final_boundaries.py` (a failure message only) and
    `docs/PERSISTENCE.md`. In `update_checks.py` only the process root's `mkdtemp` prefix (`ob-`, before it
    `ouroboros-update-checks-`) and that function's docstring change, so its record format and its one readiness-check
    budget are C9's. `supervisor/update_prepare.py`, `ouroboros/launch_roots.py`, `ouroboros/config.py`, the selector,
    the test environment, both locks and `VERSION` (7.9.0) are byte-identical to C9's.
  - Read from source, unverified natively: the reader keeps `process_root` as found, not judged, and no step here
    selects by that name, so the new prefix changes no verdict rule.
  - The earlier pin, C9 `10d51f8508e1dbecc9d01d0be088916ce7f8cc1f`, tree `e93feefeb0b9f9576d14ca85eefcdcbf1d5d0c86`.
    C9 is C8 plus one commit, sole parent C8. It changes 13 paths: `supervisor/update_prepare.py` (`78d39f97…`),
    `ouroboros/gateway/update_activation.py`, three documents, four Python tests, three web modules and their test.
    `supervisor/update_checks.py` (`34aa0f51…`), `ouroboros/launch_roots.py`, `ouroboros/config.py`, the selector, the
    test environment, both locks and `VERSION` (7.9.0) are byte-identical to C8's.
  - Read from source, unverified natively: C9 keeps how an early preparation ended as `candidate.refusal`, typed
    (an undecided root `UPDATE_WORK_ROOT_UNDECIDED`, an untyped error `UPDATE_PREPARE_CRASHED`), before it frees the
    slot, and shows it beside a live candidate too. `prepare` ends on a new refusal only at state `none`; a refusal
    beside the candidate from before the request stays in the raw status and does not end the following.
  - The earlier pin, C8 `79360ab837c2565a41310cd9a300e450eb96c640`, tree `604c2574512622ecd43f5bb93576e923624f7440`.
    C8 is C7 plus one commit, sole parent C7. It changes three Python tests (`tests/test_git_extraction.py`,
    `tests/test_host_notice_presentation.py`, `tests/test_tool_classification_differential.py`) and no other path, so
    what this file says of C7's readiness producer, `update_prepare.py`, runtime and locks holds at C8 byte for byte.
  - The earlier pin, C7 `b87e41411d9172846b09570d46aaa67539355e1a`, tree `20789d1554e5be6c61a9ffd4c99846b96aff9a6c`.
    C7 is C6 plus one commit, sole parent C6. It changes seven paths: `prompts/SYSTEM.md`, the import-failure
    text of `supervisor/git_ops_reset.py`, three Python tests, `web/modules/update_candidate.js` and its test. The
    readiness producer, `update_prepare.py`, the selector, the runtime and both locks are byte-identical to C6's.
  - The earlier pin, C6 `015ce40bb7f13ce1bd82bb67947f1eff85a8d057` (tree `cf86b0c1d9465926f5ad379ab092e58a0cf841e8`):
    C6 is the composed Unified Update: parents C5 `378b4abcc6a054ab363c8eb1dbb4ef0e03dae480` and the official
    `389783d41e41a8d3fad68c4be6f545ba2d0ba330`. Its `supervisor/update_checks.py` (`34aa0f51…`) and
    `supervisor/update_prepare.py` (`9e300fa2…`) differ from C4's; both locks and `VERSION` (7.9.0) are C4's.
  - The earlier pin, C4 `9aa7bab70b34c067226c62246f345828b73cea7d` (tree `8cbe97254b73d32bff6f26069fd551f5827ed01d`),
    was C2 (`553c8f892b10ed7b278b0f87c0e48adb82b9a4a5`, tree `50ab1b2d1e7a5f0e3b935f103339dc509e6ca95c`, the
    earlier bound subject) plus two commits that change only test fixtures:
    - C3 `a37ef20bdcc6c0b145b2fecc91550c368461b971` changes `tests/test_context_economy_window.py`;
    - C4 changes `tests/_budget_pause_exact_helpers.py` and `tests/test_acceptance_fence.py`.
  - In C4 the runtime, the preparation checks (`supervisor/update_checks.py`), the test selector
    (`ouroboros/test_selection.py`) and both locks were byte-identical to C2's. C6 keeps C2's test selector and locks.
  - Runtime lock sha256 `68e739bde5b7fbf90d2663eb01939f52c5c4b0b087097390e05807e143e8673d`, the same as 7.7.0's and
    7.8.0's; `VERSION` 7.9.0.
  - Kept apart from the execution commit E (`GITHUB_SHA`). Each case requires:
    - E has exactly one parent, C, and is the checkout's HEAD;
    - `git diff --no-renames --name-status C E` holds only additions, and its paths are exactly the public inventory:
      `ci-macos-native/SHA256SUMS`, every file it lists, and the workflow path;
    - C's tree is the pinned one.
  - The bundle is built from C, never from E.
- **Official tags:** v7.7.0 `e41596c389c3c33c3155637aaf67f8bebbf4934d`, v7.7.1 `fc8cd8146671fdcb71d336b9b49970b6f9893a0a`,
  v7.8.0 `6f3ddff05ff3e6c6f8d007ff96358df6af872e83`. `mkbundle.sh` refuses unless they name these commits.
- **Public inputs.** Each URL follows C's own release rule (`release_sync.release_asset_download_url`). Each pinned
  input's exact size and sha256 hold before its final file name exists.
  - `https://github.com/razzant/ouroboros/releases/download/v7.8.0/SHA256SUMS`: 2,642 B, sha256
    `c3c1c197760efa4a4a6f6b791f097c5fc288343f2ca708137180401d84dec049`. It must contain the exact line
    `41b9a890…57  Ouroboros-7.8.0.dmg`.
  - `https://github.com/razzant/ouroboros/releases/download/v7.8.0/Ouroboros-7.8.0.dmg`: 754,985,809 B, sha256
    `41b9a8901d29c9a9b5dfdda8780d9317c441ba8c2941d949041c15abae0eff57`.
  - `https://github.com/razzant/ouroboros/releases/download/v7.7.0/Ouroboros-7.7.0.dmg`: 751,126,575 B, sha256
    `de1387593eda0f5ea5a162e23002f116eabdd7ea974406946d9d4623a5d9bdef`. These are the digest the lab equipment pins
    and the size of the same bytes, retained from 7.7.0's release build. Whether the published asset has exactly these
    bytes is checked by the run (`input-dmg-7.7.0`), not assumed.
  - `https://github.com/razzant/ouroboros/releases/download/v7.7.0/SHA256SUMS`: not pinned. Its line for the DMG is
    the claim `release-capsule-7.7.0`.
  - 83 wheels from `https://files.pythonhosted.org/`, 119,148,291 B in all: `native/wheelhouse.pins`.
- **Actions, by the commit C's own workflows pin:**
  - `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`;
  - `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02`.
- **Jobs:**
  - triggered only by a push to `qualification/update-native-macos-cbf3e1fa619d-r1`; no workflow of C triggers on that
    branch (C13's twelve workflows, the same files as C12's, read statically: tags `v*`, the branches `main`, `ouroboros`, `ouroboros-stable`,
    pull requests to `ouroboros`, schedules, dispatch and `workflow_call` only);
  - one job per case (matrix `b1a`, `b1e`, `fail-fast: false`), each on a fresh `macos-latest` runner;
  - `permissions: contents: read`; checkout with `persist-credentials: false`;
  - no secrets, no logins, no signing, no provider keys.

## The public wheel folder
qb1's `venvprep` installs C's runtime lock into the owner's venv with pip from a local folder (`PIP_NO_INDEX=1`,
`PIP_FIND_LINKS`), as C's own `install_deps` runs pip against the index. On the lab VM that folder held wheels repacked
from an installed app, which no one else can reproduce. Here it holds PyPI's own wheels, fixed in advance by C's files:
- **The requirement set** is C's `requirements-runtime.lock`: every line whose marker holds for CPython 3.10 on macOS
  arm64. That gives 83 lines; 6 are skipped (`brotlicffi`, `colorama`, `psutil`, `pywin32`, `rpds-py==2026.6.3`,
  `websockets==17.0.1`). A marker of another shape is refused, never guessed.
- **Each line's file** is one wheel of that exact version from C's `uv.lock` (registry pypi.org), chosen by pip's own
  tag preference (`packaging.tags`, CPython 3.10, macOS arm64) and pinned by `uv.lock`'s sha256 and size. The choice
  must be the same for macOS 11.0, 16.0 and 26.0, or the line is refused. Nothing is built from source.
- **Against the lab's folder:** the same 83 names and versions. One file differs: playwright's `py3-none-any`
  repack there, PyPI's `py3-none-macosx_11_0_arm64` wheel here, which is the one pip picks on arm64.
- **At run time** the `wheel-pins` gate derives the pins again from this checkout's lock and `uv.lock` and requires
  the same bytes. `wheelhouse` then downloads each file create-only, checks its size and sha256 before its final name
  exists, and `wheelhouse.py report` requires exactly the pinned files. It writes the two files qb1 reads:
  `wheelhouse-report.json` (`lock_sha256`, which `wheels_match` compares with C's lock) and `SHA256SUMS`.

## Runner fixtures (create-only)
- **The qb1 geometry is kept.** `/Users/anka/oulab/<namespace>` with namespace `qbci`; run `b1a` (kind `app`, port
  28821) or `b1e` (kind `ext`, port 28822); tmp root `/private/tmp/qbci-<case>`.
- **`/Users/anka` must not exist.**
  - It is made once with `sudo -n mkdir -m 0700` and handed to the runner user.
  - It then holds an empty `.bash_profile`, `oulab/packages` (both DMGs), `oulab/pkg/7.8.0` (the observer: a whole copy
    of the pinned 7.8.0 app, checked with `codesign --verify --deep --strict`) and `oulab/q7b4/wheelhouse` (the public
    wheel folder, at the path qb1 reads).
  - It also holds `oulab/tools/…/CommandLineTools`, a link to the runner's own Command Line Tools (else
    `xcode-select -p`), and the namespace's `bin` (`qb1.sh`, `procwatch.py`, `venv_lab.py`) and `pkg`.
  - Last, it holds `oulab/qbci/evidence`, a link into the run's own `out/guest-evidence` (see Upload).
- **Ownership is this run's own successful `mkdir`** of `/Users/anka`, recorded at that moment, never inferred later
  from a pathname. If `/Users/anka` exists, the fixture is refused. Nothing under it is then written to, listed,
  sized, quit into or uploaded; that also holds for the finish and the stop handler.
- **Nothing existing is overwritten or deleted.** The `bin`/`pkg` digests are re-checked before every step.
- **qb1 steps run under `env -i`** with `HOME=/Users/anka` and the runner's real user name. The app receives the run's
  own roots through qb1's launch environment (custom app root, repo, data, settings, pid and port files, home, ports)
  and the owner's runtime venv (`--runtime-venv`).
- **The declared work root.** Every open of these positive cases also passes
  `OUROBOROS_SUBAGENT_WORKTREE_ROOT=<run>/app/subagent_worktrees`, through qb1's existing `ENV=VALUE` argument.
  - The key is C's own (`launch_roots.WORK_ROOT_KEYS`), read from C by the `work-root-declared` gate and recorded in
    `checks/work-root.txt`. An explicit value is what C asks for when its work-root basis is undecided; `packaged_cli`
    hands this key to an app opened through macOS the same way. The root policy is unchanged.
  - The location is the `OUROBOROS_APP_ROOT` candidate C names. The other candidate, `<run>/home/Ouroboros`, is a default
    location `identity-*` requires absent, so this run could never have meant it.
  - The other two work-root keys stay undeclared. `update_prepare.py`, `update_native.py`, `update_checks.py`,
    `body_adoption.py` and `gateway/update_activation.py` call neither getter directly (indirect calls were not traced).
    If one is reached, it shows in the kept logs.

## The existing copy and its known negative
Gatekeeper rejects a fresh copy of the official 7.7.0 app: the lab measured `spctl` exit 3, `source=Unnotarized
Developer ID`, and 7.7.0's public release material discloses it. That stays measured here:
- `install` keeps qb1's own verdict. In exactly that shape (`ditto=0 codesign=0 spctl=3 version=7.7.0`, `rejected`,
  `source=Unnotarized Developer ID`) the claim is `FAIL known_negative_unnotarized`. It is the one declared known
  negative: `RESULT.txt` lists it apart (`known_negative=`), and it decides nothing else. Any other install FAIL is an
  ordinary FAIL.
- `existing-copy` (a gate) then allows the copy to go on only as the owner's existing installation. It requires the
  pinned DMG, a whole copy (`ditto` 0), `codesign --verify --deep --strict` 0 and version 7.7.0. Gatekeeper's answer
  must be that known negative or an acceptance; any other answer is FAIL.
- The copy is opened only through ordinary LaunchServices (`/usr/bin/open`). Nothing removes a quarantine, signs,
  changes a trust or system policy, grants a permission or starts the app's binary directly. If LaunchServices does
  not run the app, `open-1` fails and the chain stops there.

**This is no fresh-install PASS.** A browser download's first launch of 7.7.0 is not qualified (below).

## Sequence
Windows are the outer capture bounds, in seconds; qb1's own seconds follow the step. `gate` stops the chain on
anything but PASS.

| # | step | kind | window |
|---|---|---|---|
| S | runner facts; `runner-arm64` (`uname -m`, `hw.optional.arm64`, not translated; image, CPU count and memory recorded) | gate | – |
| S | `disk-start` (≥ 25 GiB free), `overlay-integrity`, `clt-git`, `execution-pin`, `fixture-root` | gate | – |
| S | `input-sums-7.8.0`, `sums-names-dmg-7.8.0`, `input-dmg-7.8.0`, `input-dmg-7.7.0` | gate | – |
| S | `release-capsule-7.7.0` | claim | – |
| S | `observer`, `wheel-pins`, `wheelhouse`, `product-bundle` (`mkbundle.sh commit C`), `equipment`, `work-root-declared` | gate | – |
| 1 | `facts x` | recorder | 300 |
| 2 | `selftest x` (procwatch's exit-status method) | gate | 120 |
| 3 | `install CASE` (official 7.7.0 app: ditto, codesign, `spctl`, version), then `existing-copy` | claim, gate | 1800 |
| 4 | `mirror CASE` (main/ouroboros/ouroboros-stable = C, official tags; the install's `insteadOf` points here) | gate | 1800 |
| 5 | `candsrc CASE` (a clean checkout of C, venvprep's source) | gate | 900 |
| 6 | b1e only: `extpy b1e` (a complete relocated copy of the installed app's Python) | gate | 1800 |
| 7 | `venvprep CASE` (the owner's earlier venv, C's own functions, the public wheel folder) | gate | 3600 |
| 8 | `openw CASE open-1 300` (LaunchServices open with `--runtime-venv`, loopback HTTP 200), then `identity-open-1` | gate, claim | 600 |
| 9 | `quit CASE quit-1 240` (Apple Event) with `quit-1-request` and `quit-1-exits` | gate, gate, claim | 480 |
| 10 | `lscanary CASE canary` (file and localStorage canaries while quit) | claim | 300 |
| 11 | `compose CASE` (the owner's H, I, W and U on C; a disclosed manual checkout of C) | gate | 600 |
| 12 | `openw CASE open-2 300` (also `OUROBOROS_PREFLIGHT_TIMEOUT_SEC=13000`), then `identity-open-2` | gate, claim | 600 |
| 13 | `obs CASE pre`, `work-pre`, `localfn CASE fn-pre 6.0.3 venv` | recorder, claim, claim | 600, –, 300 |
| 14 | `prepare CASE prep 15400 ready full OUROBOROS_PREFLIGHT_TIMEOUT_SEC=13000` (C's own preparation through the live server, its real checks; the pair proven in the serving server first) | gate | 17500 |
| 15 | `activate CASE act 900` (the activation request, until that adoption closes) | gate | 1200 |
| 16 | `native-target`, then `identity-post-activate` (the server the activation's own launch started, against open-2's tuple) | claim, claim | – |
| 17 | `obs CASE post`, `localfn CASE fn-post 6.0.3 venv` | recorder, claim | 600, 300 |
| 18 | `quit CASE quit-2 240` with `quit-2-request` and `quit-2-exits` | gate, gate, claim | 480 |
| 19 | `openw CASE reopen 300` (cold reopen), then `identity-reopen` | gate, claim | 600 |
| 20 | `obs CASE cold`, `work-cold`, `profile-survives`, `localfn CASE fn-cold 6.0.3 venv`, `effects CASE fx` | recorder, claims | 600, –, –, 300, 600 |
| F | finish: leftover watchers get their stop file; on a stopped chain `attempt CASE attempt-stopped`; if a process of the run is seen, one ordinary `quit CASE quit-final 240` with its request and exits; on a stopped chain `attempt CASE attempt-after-quit`; then `owned-processes-after` and `guest-evidence` (a digest list of the live guest evidence) | always (`attempt`: stopped chain only) | 180, 480, 180 |

Each `openw` in the table also passes the declared work root (Runner fixtures).

What each qb1 step requires for PASS is qb1's own rule, which the repairs above leave unchanged (save that a wait's
success read after its SECS no longer counts). In particular:
- `prepare` sends nothing unless the status before it is readable and settled. It settles as `ready` with checks ok on
  the `full` tier, the serving server and the owner's H/I/W/U unchanged, and complete evidence: the runs the checks
  name, copied with their reports, full outputs and materialization output, inside the checks' one budget as recorded.
- `activate` requires that adoption's own `body_adoption_closed` row `adopted`, `update_activation.json` retaining it,
  a new launcher and server serving, the H/I/W/U bytes kept and the previous HEAD contained.
- `localfn` runs the owner's untracked helper on the live server's own interpreter (kernel argv[0]). It requires yaml
  6.0.3 loaded from inside that interpreter's prefix, and that prefix to be the runtime venv.

## Expected outcomes, read from C's own code
These are expectations for the verdicts to test, read from C's sources. They are no results.
- **Target.** The default channel is stable: the newest plain `vX.Y.Z` tag reachable from both `main` and
  `ouroboros-stable`. The run's mirror is the 7.7.0 app's own seed plus C's bundle, whose official v7.7.0, v7.7.1 and
  v7.8.0 tags qb1's mirror check pins; that check also requires no v7.9.0 other than C. So the target is v7.8.0.
- **An app-only Update.** v7.8.0 is an ancestor of the owner's HEAD (H on C), so C's preparation goes on only because
  the app cannot host the source: H, I and W stay, and no commit is made.
- **The native step.** 7.7.0 has no signed source entry. C and v7.8.0 both declare `MACOS_SOURCE_ENTRY_PROTOCOL = 1`,
  and v7.8.0's release lists its DMG, so the package is 7.8.0.
  - C proves the installed app's origin against 7.7.0's own release image and checksum capsule, which it fetches
    itself.
  - It then downloads, checks and stages the 7.8.0 app beside the installed one.
- **The tier.** The footprint against v7.8.0 includes cutover and apparatus paths (`server.py`, `launcher.py`,
  `ouroboros/test_selection.py`, `pyproject.toml`, `uv.lock` and others), so the selector's own answer is `full`.
  `prepare … ready full` reads the tier from the receipt, and a different tier is FAIL `tier_mismatch`. It is never
  forced.
- **The Node prerequisite (the candidate's checks, read from source; unverified natively).** From C6 on, a full tier
  needed a Node at v20.11 or later for its browser-module tests (`web/tests`); from C12 either tier does, and the checks
  run that suite as their node lane (above). Without it the checks stay unverified before any test runs. The serving server resolves it as the candidate's `resolve_node` does: `$OUROBOROS_BUNDLE_DIR/node-standalone/bin/node`, then the bundle of its own
  interpreter, then the serving checkout's `node-standalone/`, then `PATH`.
  - v7.7.0's frozen launcher stamps `OUROBOROS_BUNDLE_DIR` (its `_MEIPASS`) into the server it starts.
  - v7.7.0's `build.sh` and `Ouroboros.spec` are set to bundle Node v24.16.0 as `node-standalone/`
    (`download_node_standalone.sh`). That the published 7.7.0 DMG holds it is not checked here.
  - So the 7.7.0 app's own Node is expected to answer. launchd's default `PATH` (`/usr/bin:/bin:/usr/sbin:/sbin`) has
    no `node`, and whether this runner's launchd adds one is unverified. Without the bundled one, the checks would
    probably name `prerequisite node absent` (`prerequisite_unverified`).
  - Nothing supplies a Node here. A known Node may only come through a separate staged directory or an explicit `PATH`,
    never by writing into the tested checkout.
  - `prepare`'s `node-candidates.txt` records which of those files exist; the checks record the Node they used.
- **Already-failing cases.** A ready candidate may name already-failing cases: failures that also fail the same way on
  the serving composition. qb1's receipt then requires the baseline run behind each one.
- **The venv.**
  - b1a: the venv stays on the installed path's Python, whose bytes the exchange replaces with 7.8.0's.
  - b1e: the venv keeps the external Python.

## Overlay claims
- **`existing-copy`**: the gate above.
- **`release-capsule-7.7.0`**: 7.7.0's published `SHA256SUMS` names the pinned DMG (PASS); names another digest (FAIL);
  is unreadable or names none (UNKNOWN). It is not a gate: C's own preparation shows what it does with that capsule.
- **`wheel-pins`, `wheelhouse`**: the public wheel folder above.
- **`identity-*`.** After each open that served, and after the activation's own launch, the one `server.py` process
  is identified:
  - **Runtime:** argv[0] is the runtime venv's interpreter (`<run>/venv/bin/…`), and the interpreter itself (its
    kernel image) is inside the app for b1a, or inside the external root for b1e.
  - **App:** version 7.7.0 at `open-1` and `open-2`, 7.8.0 after the activation.
  - **Source:**
    - `open-1`: HEAD is v7.7.0's commit, with no tracked change;
    - `open-2`: HEAD is the composed owner commit on C;
    - after the activation: HEAD contains C and the composed owner commit, and differs from C only in the owner's
      paths. C's tree there is the pinned one, and the execution commit E is in neither the run's checkout nor its
      mirror. A HEAD that moved is named.
    - Tracked changes stay in the owner's paths.
  - **Loopback:** `127.0.0.1:<port>` answers 200.
  - **Roots:** the server's own recorded environment holds the exact `--env` tuple qb1 passed to that open: C's launch
    tuple (`OUROBOROS_APP_ROOT`, `_REPO_DIR`, `_DATA_DIR`, `_SETTINGS_PATH`, `_PID_FILE`, `_PORT_FILE`), the server
    port, host-service port and host, the declared `OUROBOROS_SUBAGENT_WORKTREE_ROOT`, `HOME` and `TMPDIR`. Paths
    compare by real path.
    - After the activation it is checked against open-2's tuple. C carries the launch configuration (roots, settings,
      pid and port files, server port, `HOME`, `CFFIXED_USER_HOME`) and keeps the inherited operator inputs.
    - A value that differs is FAIL; a requested name absent from the server's environment is UNKNOWN.
    - qb1's default-location witnesses must be absent: `<run>/app/data/settings.json`, `<run>/app/ouroboros.pid`,
      `<run>/app/data/state/server_port`, `<run>/home/Ouroboros` and `/Users/anka/Ouroboros/data`.
      - One that is found is FAIL. Found means by `test -e`, or else by the lookup below; both follow links.
      - Absent only when its lookup answered that nothing is there. `test -e` carries no errno, so a location it does
        not find gets one `stat` by the observer interpreter. `ENOENT` (a dangling link too) or `ENOTDIR` is a
        completed absence.
      - Any other error (no search permission, an I/O error, a link loop, …), or a reader that did not run or answer,
        leaves the location unread, never absent. That is UNKNOWN, or FAIL when another location is in use or another
        rule fails.
      - `checks/identity-*/default-roots.txt` keeps each answer. `obs`'s `config.txt` lists the same five by the same
        rule, recorded and not judged.
- **Quit.** A normal Quit needs three claims, as in the b1s probe:
  - `quit-N`: qb1's cold state;
  - `quit-N-request`: the Apple Event request succeeded (osascript exit 0);
  - `quit-N-exits`: an extra procwatch armed before the request observes the launcher's and the server's exits and
    every registered exit status. Exit codes are recorded, not judged.
- **`native-target`.** After the activation, the app at the installed path must be the official 7.8.0:
  - it has the same code directory hash and team as the whole copy of the pinned 7.8.0 DMG's app;
  - `codesign --verify --deep --strict` passes;
  - Gatekeeper accepts it as Notarized Developer ID;
  - before the preparation, that path held 7.7.0.

  Folder identities, the previous app's backup and the staging folders are recorded, not judged.
- **`work-pre`, `work-cold`.** The owner's selected work as `compose` recorded it (qb1's `work_lines`: `OWNER_NOTES.md`
  in HEAD, the index and the worktree, and every untracked file with their count) has the same bytes in `obs`'s
  record after the first composed launch (`work-pre`) and after the whole transition and the cold reopen
  (`work-cold`).
  - Each witness counts only when both records read it completely (checkout_bytes' own `READS` line).
  - A witness that was read completely and differs is FAIL, also when another witness is unread. Otherwise an unread
    witness is UNKNOWN.
  - HEAD itself is `identity-*`'s (its ancestry). `prepare` and `activate` keep their own comparisons in between.
- **`profile-survives`.** The file canary and the `<case>-ls-canary` localStorage key with its value are on disk in the
  run's private WebKit store after the activation and the cold reopen. This is an on-disk witness only.
  - The file canary's sha256 is the planted one.
  - One row's key is exactly the canary key (SQLite compares it), with exactly the planted byte length and whole hex
    value. A longer, shorter or otherwise changed value is not the planted one. Its bytes are compared; its SQLite
    type is not.
  - Each canary is judged only when its planting is established. The file canary needs its own planted sha256. The
    localStorage canary needs lscanary's PASS and the planted value. A planting not established is never read as a
    loss.
  - A completed read that misses a canary is FAIL `canary_lost`; one that finds it changed is FAIL `canary_changed`.
    Either stays FAIL when another part was unread or not established, and its detail still names that part.
  - Otherwise such a part is UNKNOWN: `canary_not_planted` when lscanary did not PASS, else `canary_unread` (or
    `cold_store_unread` with no cold record). PASS needs every read complete (`PROFILE_READS=complete`).
- **`owned-processes-after`.** From one `ps` read, no process whose command line names the run root or the app. The
  claim does not cover a descendant whose command line names neither.

## Named outcomes (honest failures)
- **Setup:** `not_arm64`, `low_disk`/`disk_unread`, `procedure_error`, `clt_git_missing`, `not_c_plus_inventory`,
  `fixture_root_exists`/`cannot_create_fixture_root`, `download_failed`, `digest_mismatch`, `sums_line_missing`,
  `dmg_attach_failed`, `observer_invalid`, `pins_not_reproduced`, `wheelhouse_exists`, `wheelhouse_invalid`,
  `mkbundle_failed`, `bundle_bind_mismatch`, `equipment_copy_failed`, `work_root_key_not_in_c`.
- **Steps:** each takes qb1's own verdict.
  - Install: `known_negative_unnotarized` (the declared one), `install_fail`.
  - Open: `launchservices_open_failed`, `app_not_running_after_open`, `not_serving`, `exit_observer_incomplete`.
  - Quit: `quit_not_cold`. Canary: `no_private_webkit_store`, `no_localstorage_db`.
  - Prepare: `prepare_not_admitted`, `ended_without_candidate`, `not_settled`, `serving_not_identified`, `serving_changed`, `local_work_unread`,
    `local_work_changed`, `tier_mismatch`, `evidence_incomplete`, `prerequisite_unverified`, `not_ready_fail`.
  - Activate: `activation_refused`, `no_adoption_id`, `adoption_not_closed`, `adoption_closed_otherwise`,
    `answer_not_activating`, `activation_record_missing`, `activation_record_differs`, `not_serving_after_adoption`,
    `same_server`, `local_work_unread`, `local_work_changed`, `head_not_contained`.
  - Local function: `package_env_not_established`, `server_count_not_one`.
  - Generic: `capture_timeout` (the client's window ended; outcome unknown), `refused_before_effect` (REFUSED: exit 2
    with qb1's own refusal line last on stderr), `step_aborted` (UNKNOWN: any other exit 2 without a verdict, so
    effects are not excluded), `verdict_missing`, `verdict_exit_mismatch`, `time_budget`.
  - Open and quit failures also record the session type (`launchctl managername`).
- **Overlay claims:** `official_existing_copy`/`not_the_official_copy`/`install_unread`;
  `capsule_names_pinned_dmg`/`capsule_names_other_dmg`/`capsule_lacks_dmg`/`capsule_unread`; `identity_mismatch`,
  `identity_not_established`, `server_count_N`; the Apple Event names of the b1s probe; `exit_watch_not_armed`,
  `exit_unobserved`; `official_7_8_0`/`native_target_mismatch`/`native_target_unread`;
  `local_work_kept`/`local_work_changed`/`local_work_unread`; `canary_not_planted`, `cold_store_unread`,
  `canary_unread`, `canary_lost`, `canary_changed`; `no_matching_rows`, `process_table_unread`, `processes_remain`;
  `attempt_unknown` (a read of the attempt capture did not complete);
  `job_stop_signal` (`INTERRUPTED.txt`).
- **Overall:** `ERROR` (2) > `FAIL` (1) > `UNKNOWN` (3, including REFUSED and NOT_RUN) > `PASS` (0). The declared
  known negative counts in none of them; `RESULT.txt` names it on its own line.

## Not qualified by these cases
- **A custom app root whose work root is undecided.** On this fixture, a custom `OUROBOROS_APP_ROOT`, no work-root value
  and a providerless 7.7.0 history (`queue_snapshot.json` with no state or witness) leave C's basis unknown, and C's root
  resolver raises `WorkRootUnavailable` (from C9 on the preparation names it, `UPDATE_WORK_ROOT_UNDECIDED`, read from
  source). That stays a negative product case, which these positive cases step around by
  declaring the root. An `openw` without the pair still reaches it. It is not waived here, and its repair is not claimed.
- **What a declared root proves.** It says where the workspaces go. It does not imply a fresh install of 7.7.0, and it
  does not explain the earlier runs: their post-request server log is lost, so `WorkRootUnavailable` was never observed
  as their exception.
- **A fresh install of 7.7.0** and a browser download's first launch: Gatekeeper rejects the fresh copy (the known
  negative), and a copy fetched by `curl` carries no quarantine, so no first-launch evaluation occurs.
- **The Updates panel.** `activate` posts the panel's body directly; the panel's read-only effects preflight and its
  disclosure dialog are not exercised.
- **The stock updater** (the b1s probe), `b1p`, `b1r`, the published-range readiness roots, and Windows and Linux.
- **The live profile and the live server's own imports.** The profile claim is an on-disk witness. The local-function
  helper runs as its own process with the server's interpreter, not inside the serving process.
- **Anything of a real owner:** installation, home, Settings, profile, data or credentials. The run's roots are
  fixtures on a disposable runner. The runner user's real home is only listed before and after
  (`runner/real-home-*.txt`).
- **Equivalence with the lab VM.** The machine, macOS image, Command Line Tools git version, `.bash_profile` (empty
  here), GUI and Automation (TCC) state, and network all differ. Here the network is open internet. Only the official
  repository URL is served by the mirror, through the run home's `insteadOf`. C's own preparation fetches 7.7.0's and
  7.8.0's release assets from GitHub itself.
- **Dependency changes.** C's lock equals 7.7.0's and 7.8.0's, so no dependency change is exercised.
- **Other Macs:** Intel Macs and other macOS versions.
- **Any measured stop or upload.** Cancelling a run, a job timeout, or the provider discarding the runner is not a
  measured process or VM exit, and provider teardown is not claimed. Processes still running after the finish's
  ordinary quit are reported, never signalled. A runner lost before its upload step uploads nothing.
- **Beyond one sample:** durations, exit codes of the quit, untracked non-owner files, and coverage of all tracked
  work. Only compose's selected witnesses are judged (`work-pre`, `prepare`, `activate`, `work-cold`):
  `OWNER_NOTES.md` in HEAD, the index and the worktree, and every untracked file. Other tracked files are not.

## Bounds and estimates
- **Time:**
  - GitHub ends a hosted job at 360 minutes; the job states that limit. The case step ends at 345 minutes, so the
    upload still has 15. Only these bound everything.
  - Inside the script, a step starts only if its whole window fits before the 315-minute point after the script's
    start (`BUDGET_S` 20100 s less the finish's `RESERVE_S` 1200 s). Otherwise it is UNKNOWN `time_budget` and the chain
    stops; no window is shortened. The finish may use the last 20 minutes of `BUDGET_S`: on a stopped chain 180 + 480 +
    180 s at most, so 840 of its 1200 s.
  - C's checks have one monotonic budget, `UPDATE_CHECKS_TIMEOUT_SEC` (7200 s by default; an explicit
    `OUROBOROS_PREFLIGHT_TIMEOUT_SEC` replaces it). This lab sets 13000 s in open-2's launch environment only and
    proves it in the serving server before `prepare` sends anything (above). The budget begins just before the new run
    W′ is materialized and bounds every process after it: W′, a comparison run on W of W′'s failures, a target run on T
    of blocking ones, their materializations and both lanes of each. Each run gets what remains.
  - `prepare`'s allowances are sized from that budget (`run-native.sh`: `PREP_GUEST_S`, `PREP_WINDOW_S`):
    - qb1 follows the status until 15400 s after the step's start (elapsed time, above) = 13000 s (the budget) + 1800 s before it (the native app's downloads and
      staging, the generation's clone and pip, W′ materialized once outside the budget, the prerequisite probes) +
      600 s after it (W′ restored under Git's own 300 s bound, the record settled);
    - the client's window is 17500 s = 15400 s + 2100 s for qb1's reads and copies after the settle.
    These are observation allowances, not a product deadline. The verdict reads the deadlines the checks recorded; a
    client timeout leaves the outcome unknown and proves no deadline and no stop.
  - So `prepare` starts only if no more than 1400 s have passed since the script's start (18900 − 17500). Whether
    the setup and steps 1–13 finish by then on a hosted runner is not measured; if not, `prepare` is UNKNOWN
    `time_budget`, not started, and the chain stops. The steps after it hold windows of 4680 s (activate 1200,
    obs-post 600, fn-post 300, quit-2 480, reopen 600, obs-cold 600, fn-cold 300, effects 600) that are **not**
    reserved: each starts only if its own window fits what is left, as every step does, so a `prepare` that uses
    much of its window leaves later steps UNKNOWN `time_budget`. A preparation seen to end without a candidate frees
    its window at once (`prep_ended`). The job, step and script bounds (360 min, 345 min, 20100 s) are unchanged and
    stay within GitHub's hosted limit.
  - The setup before the first step is **not** under the budget. It covers four downloads and the wheels (each bounded
    only by `curl --max-time`), the observer copy, verification and `mkbundle.sh`.
  - Durations are **UNKNOWN** until measured. That includes whether C's full tier (the suite directory) completes
    within the checks' one budget on this runner, together with any comparison runs; a run that does not complete
    leaves the candidate unverified, never ready.
- **Stop:**
  - The workflow `exec`s the runner, so the runner's cancel or timeout signal reaches it directly.
  - Downloads and qb1 steps are waited on in the background, so the signal is handled at once. The handler gives the
    watchers their stop file, writes `INTERRUPTED.txt` and exits, with no collection and no digests.
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
  - 25 GiB free is required at start and 8 GiB before each step.
  - **Estimated** use, unmeasured, about 10–12 GB:
    - the two DMGs, about 1.5 GB;
    - two app copies (observer and run), about 1.5 GiB each;
    - the wheels, 0.12 GB, and the venv;
    - the mirror, the candidate source and the run checkout, under 1 GB;
    - during `prepare`: C's candidate workspace, its downloads of the 7.7.0 and 7.8.0 DMGs, the staged 7.8.0 app and
      the generation's venv clone.
  - Plus the full checkout: **UNKNOWN**.
  - Peak use: **UNKNOWN**. Measured values: `space.log`, `checks/du.txt`.
- **Upload:** one artifact per case, `qb-ci-native-<case>-<run id>-<attempt>`, `if: always()`, hidden files included,
  retention 30 days.
  - **What it holds:** exactly the output root that case created,
    `$RUNNER_TEMP/qb-ci-native-<case>-<run id>-<attempt>/out`. That root's name is unique to the case and the job
    attempt, `RUNNER_TEMP` is emptied at the start of each job, and the runner makes the root create-only.
  - **Contents:** `RESULT.txt`, `claims.tsv`, `probe.log`, `space.log`, `runner/`, `inputs/`, `product/`, `checks/`,
    `watch/`, `receipts/` (capture receipts), `guest-evidence/` (with each attempt capture's server logs and raw
    records, as `obs` always copied them), `guest-evidence-SHA256SUMS`, and `SHA256SUMS` (written at a normal end only).
  - **`guest-evidence/` is qb1's live evidence.** `/Users/anka/oulab/qbci/evidence` is a link this run makes into it.
    So every step's evidence is under the upload root from the start, also after a cancel or a stop signal. It is
    complete only as far as `guest-evidence-SHA256SUMS` lists it.
  - **Nothing outside that root is ever selected.** A pre-existing `/Users/anka` is refused, and the runner then never
    writes to, lists, sizes, quits into or uploads anything under it.
  - On a public repository, logs and artifacts are public. Only named variables are recorded, never the environment.

## Before dispatch (owner of the run)
1. **The execution branch publishes C.** Pushing `qualification/update-native-macos-cbf3e1fa619d-r1` makes C and its history public
   on the repository. That is a publication decision.
2. **E must be one commit whose only parent is C.** It adds exactly the files listed above at `ci-macos-native/`
   (SHA256SUMS, plus every file it lists) and the workflow copy at its path, nothing else. Before the push, check
   these three facts:
   - `git rev-list --parents -n 1 E` prints `E C`;
   - every line of `git diff --no-renames --name-status C E` is an addition;
   - its sorted paths equal the inventory.
   Each case repeats these checks on the commit that ran.
3. **No other workflow of C triggers** on this branch (by static reading of C's workflows).
4. **Actions must be enabled for the repository,** with hosted macOS runners available for two jobs at once.
