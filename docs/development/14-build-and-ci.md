# Build & CI

The commit gate stays reproducible and candidate-independent; ARCHITECTURE §8 owns the job graph, triggers and release chain.

### Python dependency locks

`pyproject.toml` + `uv.lock` are sole dependency authority; `requirements-runtime.lock` is the generated runtime projection, `requirements.txt` a static `-r requirements-runtime.lock` pointer (`tests/test_packaging_sync.py`; ARCHITECTURE §8 "Build scripts"). Dependency edits update metadata, run `uv lock`, regenerate `requirements-runtime.lock` via the README command and keep CI's clean-diff green; pinned `tool.uv.required-version` and digest-pinned `setup-uv` keep resolver updates deliberate. Docs may pair checkout-free `uv tool install` with a full commit SHA, never claiming locked deps, a release-artifact install or a dev environment.

### Pytest marker lanes

Default local pytest excludes seven costly/environment-dependent lanes in `pyproject.toml` `addopts` (mirrored by `LANE_EXCLUSION_EXPR`); CI opts in explicitly (job graph/matrix: ARCHITECTURE §8 "CI topology"):

- `integration`: live provider checks; split manual/tag jobs by `credential_required`, never waive mixed exits, require `provider_release_floor.py` canaries before official tags. Keyless `tests/system_e2e/` lanes: ARCHITECTURE §8.
- `browser` / `ui_browser` / `ui_browser_docker`: real Playwright engines (agent browser / host UI / `ouroboros-web:test` container, skipping without Docker), marker owning collection; `PageErrors` (`tests/ui_console_errors.py`) strict unless declared; `portable_detail` covers build/portable invariants; manual/tag `ui-smoke` runs full `ui_browser` ("Safe local verification" below); none needs paid keys.
- `skill_smoke` (`tests/test_skill_smoke_official.py`): pinned official catalog on three OSes with real network/pip—investigate RED, never skip. Only Ubuntu review legs hold keys (missing fails); one author task uses `skill_review` without payload edits (at most two paid cycles including infra, a second only for a grounded rebuttal or settled verdictless review), keep pending RED, accept only persisted independent PASS and reap the key-bearing tree before enable.
- `size_ratchet`: live-repo size gates and sole blocking repo-size surface (rules: "Module Size & Complexity"; base fallback and fail-closed rule: ARCHITECTURE §8).

Never combine `skill_smoke` or `size_ratchet` with `serial` (`and not <lane>` markexprs enforce the barrier; single-lane assignment keeps placement unambiguous). New opt-in lanes register in `pyproject.toml`, add a collect-only zero-test CI guard and keep default `addopts` free of network/Docker.

### Retiring and merging tests

For any test module/class, answer: what stays true; who can break it on a supported install and see what; where its home lives; cost vs protection. Verdicts: `keep`, `extend <home>`, `merge into <home>` (named survivor, one falsifying negative per failure mode), `delete: obsolete (<what ended it>)`, `delete: review owns (<item>)`, `rewrite onto <seam>`. Retire classes via re-runnable `devtools/test_census.py` rules + sampled independent reads, logging rule and manifest in PR evidence (never in-tree ledgers). Never delete or trim tests in current-cycle post-merge-red, `FLAKY` or `ACCEPTED` records (folding keeps every case and real process); real-child tests never move onto in-process doubles. Review: CHECKLISTS Body item 11(d).

### Parallel CI and the `serial` marker

CI (`run_tests.py --lane-pass`, wrapped by `ci_retry.py`) and commit gate share exclusions and flags. xdist groups named resources longest-first by duration; only `serial("exclusive")` uses the one-process pass. Both enforce a 300 s timeout (POSIX fails the test, Windows stops its worker); empty passes are green (`DEFAULT_PASS_EMPTY`). CI passes paths or retry `@file` plus reporting/shard options, never pytest policy flags. `tests/default_lane.py` packs whole files by duration, else count. Shard 1's collect-only manifests are independent witnesses; benchmarks use `pyproject.toml` addopts without serial split.

- `@pytest.mark.serial("<resource>")` names the unisolatable host resource (fixed port, real daemon, unrestorable process global), never an owned child, private root or port 0; bare `serial` fails collection, unlabelled sharers fail workers under `--max-worker-restart=0`, and slow tests are sped up or split instead.
- All other tests stay parallel-safe via `tmp_path`, `monkeypatch.setenv`/`delenv`/`setattr`, no order assumptions and fixture-restored globals (`tests/conftest.py::_isolate_workspace_executor_globals`).

### Test design

Start real processes only for process-boundary claims (signals, exit/teardown order, exec env/flags, stdio encoding, cross-process locks, sockets/pipes, containment/reaping, external binaries), keeping one real-process witness per distinct boundary claim at each entry point and running other cases in-process (`main(argv)`, `runpy.run_path(..., run_name="__main__")`) or via public seams without patching internals. Build one immutable template per world per run (per process without a shared owner root) and clone it privately for each test (`tests/_worlds.py`); delete test folders once processes end.

### The local battery entry point

Context PR checks: `python scripts/context_gate.py`.

`python scripts/run_tests.py` (`make test`) runs Node and default Python lanes (forwarding pytest args); `--selected [--base REF]` (default `HEAD`) runs commit-selected passes, printing tier/rule/module count via inventory plugin (`preflight_inventory.launch_args` avoids Windows' 32,767-char argv cap); `--extra-test PATH`/`--full-tests` widen with `--test-reason`. Missing Node fails `NOT_RUN`, never green. Workers come from `test_parallelism.SlotLease` (floor 1, waiting only on critical kernel memory pressure; ledger/counting: `ouroboros/test_parallelism.py` docstring). The runner first warms the bytecode prefix (`ouroboros/test_bytecode.py`) that later processes only read.

`--evidence-dir DIR` sets per-pass `--ci-evidence-dir` and keeps prior exports/journals (`selection.json` + inventory JSONL for `--selected`); never set via `PYTEST_ADDOPTS` (nested pytest could overwrite).

### Safe local verification and dirty browser candidates

Use the stdlib launcher (audited helper imports, scrubbed owner config/creds, printed isolated roots) with a dependency-only venv so editable installs never import deleted modules (`scripts/safe_test.py` docstring):

```bash
uv sync --locked --extra browser --group dev --no-install-project
python -I -S scripts/safe_test.py -- .venv/bin/python -m pytest tests/test_test_environment.py
export PLAYWRIGHT_BROWSERS_PATH="$(pwd)/.tmp-data-browsers"
python -I -S scripts/safe_test.py -- .venv/bin/python -m playwright install chromium webkit
OUROBOROS_RUN_UI_SMOKE=1 OUROBOROS_EXPECT_BROWSER_ENGINES=chromium,webkit \
  python -I -S scripts/safe_test.py -- .venv/bin/python -m pytest tests/ -m ui_browser --require-ui-browser
```

`--require-ui-browser` fails narrow/empty collection, missing engines, unsettled cases or unapproved skips (`tests/browser_lane.py`); `--temp-parent` rejects Git ancestors; custom paths may exceed socket limits. Owners export streams, source and declared diagnostics (not runtime archives) to `SAFE_TEST_EVIDENCE` (`--evidence-dir`, `OUROBOROS_TEST_EVIDENCE_ROOT` or docstring defaults), reap and release declared scratch on green and red (`SAFE_TEST_RETAINED`/`RELEASED`; signals/`--timeout`: docstring). Unknown inputs stay; `--keep`, root markers or export/teardown failures keep the root, sub-markers their test folder (ARCHITECTURE §9 "Test runs"; bare/hard-killed pytest promises no cleanup).

Launcher and preflight scrub inherited provider/lane/run controls; only the launcher preserves `OUROBOROS_UI_BROWSER_ENGINE` and `OUROBOROS_PREFLIGHT_REQUIRE_{PLUGINS,NODE}`. Bare pytest keeps CI controls; `OUROBOROS_PYTEST_ACTIVE=1` blocks installs, `MAC_CHROMIUM_TMPDIR` confines Chromium staging.

`tests/candidate_checkout.py` owns byte-faithful dirty capture, refusals and boot proof (`origin_proof=True` binds served static/Python by sentinel/VERSION suffix; installed distributions refuse; docstring); `ui_candidate_checkout` owns one copy/session, `direct_server_with_data` (`owned_process_temp`) isolates data/processes/ports/sockets, and `case_checkout` verifies source/copy and restores overlays after proven teardown.

`scripts/claudexor_lifecycle_smoke.py` and `scripts/claudexor_quit_witness.py --evidence-dir D [--handle-exe H]` (only Quit frees home and project; base `--expect hold`) run a real credential-free engine under `python -I -S scripts/safe_test.py -- <python> <script>`, stopping daemons and removing `cleanup.temp_root` only on clean pass (substituting only fake harnesses, witness stubs and Restart seams).

### Reading CI failure evidence

`tests/ci_evidence.py` and `.github/actions/test-evidence` report producer exits separately from test counts (passing cases never mask nonzero exits; skips/missing creds stay explicit; never upload raw JUnit; diagnostic failures report `diagnostics_incomplete` without changing release eligibility). UI `safe_test` runs upload only `test_evidence.run_diagnostics` under `ci-raw/ui/<producer>` (`ci_retry` rounds in `retry-<k>`). `reconcile-shards` rejects missing/unreadable/red projections, misassignments, duplicates/unexecuted nodes and nonzero sessions using highest attempts (red sessions pass only via their pass's `FLAKY`/`ACCEPTED` `verdict.json`, whose rounds count as executed). Failed proof upload/export fails the attempt; step ceilings fail without cancelling the job; dead xdist workers report `crash` (timeouts come from callback records since thread banners die with stdout); UI shards stop cooperatively on `--session-timeout`.

### Flaky and accepted tests

`scripts/ci_retry.py` retries failed tests once and continues unrun ones (retry pass → `FLAKY`; fail/unrun → red). `ACCEPTED` requires `tests/accepted_failures.json` and an open `not-a-product-bug` issue. New scale shapes must fit post-merge's eight-minute contract (`tests/test_post_merge_workflow.py`).

### The tiered commit and CI gates

`commit_reviewed` selects `test_selection.FLOOR_TESTS`, changed tests, direct AST importers, `test_*_static.py` for `web/` and authored widening (missing/empty floor blocks), honoring `LANE_EXCLUSION_EXPR` and dirty/untracked files vs `HEAD` pre-commit / `HEAD~1` post-commit (body candidates: serving fork). Imports are minimum: `extra_tests` covers indirect/dynamic, prompt, skill and cross-language consumers, `full_tests` unresolved impact (both need `test_reason`, kept until adoption). Edits to runner/selector, floor, pytest config/plugins, fixtures, non-version deps, `_CUTOVER` or `tests/` helpers imported by config/nothing force full; other shared helpers select importers transitively (no product fan-out threshold). Records bind tier, candidate/base/policy, reasons and counts (missing/duplicate execution blocks).

Full is required for managed updates (forced), non-candidate evolution publication, post-adoption boot push and tag/dispatch CI; local adoption also accepts an exact serving-`HEAD` selected run under loaded policy covering the reviewed switch delta and declared widening (withholding branch/tag); ordinary serving commits and `promote_to_stable` have no tier gate (a later push can publish a withheld adoption). New `HEAD` requires new evidence (skip/`None` proves nothing). Post-commit failures keep ordinary commits with warnings, withhold evolution publication, and roll back or hold managed updates at `gate_blocked`.

`preflight_runner.run_hermetic_pytest` runs Node (`--test-timeout` 300 s), bytecode warm and both pytest passes in one disposable checkout at background priority with leased workers. Node and pytest share one timeout (`OUROBOROS_PREFLIGHT_TIMEOUT_SEC`; no default total for full). Warming is bounded by the remaining time at its start but does not consume the passes’ budget. Unbounded collection or warm can block until Stop. `result.json` records `workers`, budget, `elapsed_seconds` and cleanup/retention facts, including evidence errors. `LANE_EXCLUSION_EXPR`/`*_PASS_FLAGS` govern passes; missing Node/plugins block; `preflight_node.node_environment` fingerprints Node on gate/CLI PATH only; Acorn (`web/tests/no_undef.test.js`) and lockfile-frozen ESLint stay independent; temp defaults match `safe_test` without `--temp-parent` (small/noexec `/tmp` can fail).

Capture uses one hardened worktree-vs-`HEAD` binary diff (else `PREFLIGHT_CANDIDATE_ASSEMBLY`), scrubs `PYTEST_*`, `NODE_OPTIONS`, settings and creds, probes plugins outside candidate control, adds `HEAD~1` post-commit to catch suite deletions, and allows explicit `OUROBOROS_PREFLIGHT_SERIAL=1` rollback without silent fallback. Lane exits decide; unconfirmed children block green and retain the tree (ARCHITECTURE §9). Pre-commit `reading` names one blocked attempt in CI terms (no retries/waivers), not another verdict. Proof binding: ARCHITECTURE §6.

### GitHub Actions: secrets in step-level `if:` conditions

GitHub rejects `secrets.*` in step-level `if:` (step `env:` is unavailable too): derive a non-secret boolean in job `env:`, gate steps on it, and pass creds only to first-party steps needing them—never SBOM/attestation steps. Enforced for `ci.yml` only by `tests/test_build_scripts.py::TestMacOSSigning::test_ci_uses_env_context_for_condition`.

### Apple signing & notarization (macOS Build job)

Prereleases may ship unsigned if disclosed (ARCHITECTURE §8 "Build scripts"). Only non-secret `HAS_APPLE_SIGNING` is job-wide: cert/keychain secrets stay in import, Apple ID secrets in build, with `always()` cleanup preventing persistence. When configured, `build.sh` notarizes fail-closed—app and DMG both require `Accepted` plus a validated staple, and signed stable DMG smoke requires both tickets plus `spctl` `Notarized Developer ID` on the copied-out app—so signing without notarization stops release (receipts/notes record `notarized`/`not_notarized`).

### Windows Authenticode release signing (SSL.com eSigner)

Non-secret repo variable `ESIGNER_CERT_SHA1` selects signing on `v*` tags (prereleases included): official `razzant/ouroboros` fails closed if unset (primary Setup must be signed); forks without it bind no signing Environment/secrets, build unsigned Setup in `windows-proof` and disclose both unsigned; when set, any config, tool, signature, timestamp or digest error stops release without fallback.

Pipeline: `build` packs unsigned ZIP (`pack_windows_archive.ps1`) by digest; `windows-sign` (`windows-release-signing`) stages Inno secret-free, runs `sign_windows_release.ps1` (scrubs `ESIGNER_*` from child envs) over `windows_signing_catalog.json` (`Ouroboros.exe`, `python.exe`, `pythonw.exe`; sync pins with downloader), checks payload/unsigned/pins/CodeSignTool before request 1, signs those 3 files, packs ZIP + Setup `.exe` (`pack_windows_setup.ps1`, `ouroboros.iss`) from that tree, signs outer Setup; secret-free `windows-proof` runs `smoke_windows_setup.ps1` (disposable-runner gated: outer+installed signatures, foreign/busy refusal, ZIP parity, Start AUMID, CLI/Betterleaks, data-preserving uninstall).

Setup: create `windows-release-signing` (`v*` tag rule, no reviewers), set `ESIGNER_{USERNAME,PASSWORD,CREDENTIAL_ID,TOTP_SECRET}` as Environment secrets (`gh secret set <NAME> --env windows-release-signing`; never repo secrets or pasted values—rotate exposed TOTP seeds at SSL.com), then set `ESIGNER_CERT_SHA1`. Each attempt spends up to four billed eSigner signings even on failure; overages are invoiced and balance is invisible to CI.

Verify extracted ZIPs/installs (`-ExtractedRoot`) or outer Setup (`-Executable`) via `scripts/verify_windows_signature.ps1` (`-ExpectedThumbprint` = `signerThumbprint`, SDK `signtool`). Mocked tests prove no credential, timestamp or publisher display (live signing waits for Windows release launch); other bundled files, venv templates, later venvs and Git updates stay unsigned, and SmartScreen/AV weigh reputation.

### Release proof capsule

ARCHITECTURE §8 and `.github/workflows/ci.yml` own the artifact pipeline; keep these invariants:

- Valid tags start builds and signing beside tests; red tests still spend notarization/eSigner signings and record attestations, publishing no Release.
- Proof/notes log every job conclusion plus `FLAKY` and `ACCEPTED` tests; no tag waives a lane.
- Releases are draft-first with per-tag concurrency, revalidating the remote annotated tag against the event SHA before drafting/publishing; reruns never overwrite published releases.
- Vendor smokes report separately (only the RED OS CI image fetch uses its registry CA) and never gate release.
- Native Linux builds need Jammy x86_64 and authenticated source indices (missing exact sources fail capture; other builders keep host backend); CI keeps complete/partial sources 30 days, and recipient delivery stays a release obligation.
- Release AppImage smoke stays headless ([Native GTK qualification](../../packaging/appimage/README.md) is separate); local `OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1` skips host-lib installs only, not browser bundling, disclosing unproven host compatibility.
- Later checksums are not build provenance, an SBOM or packaged smoke evidence.
