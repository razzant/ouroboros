# Build & CI

Rules for dependency locks, pytest lanes, CI and commit-gate tests, and Actions secrets. The gate stays reproducible and candidate-independent; ARCHITECTURE §8 owns the job graph, triggers and release chain.

### Python dependency locks

`pyproject.toml` + `uv.lock` are the one dependency authority; `requirements-runtime.lock` and `requirements.txt` are generated projections (ARCHITECTURE §8 "Build scripts"). A dependency change updates the metadata, runs `uv lock`, regenerates the runtime export with the exact README command and leaves the CI clean-diff check green; the pinned `tool.uv.required-version` and the digest-pinned `setup-uv` action make a resolver change deliberate. Docs may pair the checkout-free `uv tool install` form with a full commit SHA, never claiming it locks dependencies or calling it a release-artifact install or a development environment.

### Pytest marker lanes

Default local pytest excludes seven costly or environment-dependent lanes — `integration`, `browser`, `ui_browser`, `ui_browser_docker`, `portable_detail`, `skill_smoke` and `size_ratchet` (`pyproject.toml` `addopts`, mirrored by `LANE_EXCLUSION_EXPR`) — and CI opts into each explicitly (job topology and provider matrix: ARCHITECTURE §8 "CI topology"):

- `integration` runs real provider checks. Split manual/tag jobs by `credential_required`, never waive a mixed pytest exit: optional failures stay red outside release dependencies; required missing credentials fail; quota/429/5xx/timeout stay inconclusive, contract/auth/model/reasoning/tool 4xx red. `provider_release_floor.py` blocks official tags unless every required canary passed. No provider secrets in PRs. Keyless `tests/system_e2e/` lanes and markers: ARCHITECTURE §8.
- `browser` / `ui_browser` / `ui_browser_docker` launch real Playwright engines (agent browser tools / the host UI / the `ouroboros-web:test` container; the docker lane skips cleanly when Docker is unavailable locally). The marker is the source of truth for what a lane collects. `portable_detail` covers build and portable artifact invariants. Manual/tag `ui-smoke` runs the complete `ui_browser` lane (required invocation: "Safe local verification and dirty browser candidates" below). None needs a paid provider.
- `skill_smoke` (`tests/test_skill_smoke_official.py`): pinned live official catalog, three OSes, real network/pip; investigate RED, never skip. Ubuntu review legs alone hold keys; missing fails. One author calls `skill_review`; no payload edits; two paid cycles count infra, the second one grounded rebuttal or one retry of a verdictless review. Pending stays RED. Only persisted independent acceptance passes; reap the key-bearing tree before enable (ARCHITECTURE §8).
- `size_ratchet` carries the live-repo size gates and is the only blocking surface for repository size (rules under "Module Size & Complexity"; base fallback and the fail-closed rule: ARCHITECTURE §8 "CI topology").

`skill_smoke` and `size_ratchet` tests never also carry the `serial` marker or join `_SERIAL_TEST_FILES`: the `and not <lane>` markexprs are the lane barrier, and single-lane assignment keeps each test's placement unambiguous. A new opt-in lane registers its marker in `pyproject.toml`, adds a collect-only zero-test guard in CI, and keeps the default local addopts free of network and Docker requirements.

### Retiring tests

Each removed or rewritten test gets a verdict line in PR evidence, not a ledger:
obsolete (what ended it), wording owned by review (which item), rewritten, or merged
(the survivor). Wording pins retire by class: a re-runnable rule and a sampled
independent read; a structural scan of a live rule is rewritten into behaviour or
kept. A merge runs its survivor and one falsifying negative per failure mode.
CHECKLISTS Body item 11(d) owns the review.

### Parallel CI and the `serial` marker

CI runs the default suite in two passes, each owned by `scripts/run_tests.py` behind `scripts/ci_retry.py` (jobs: ARCHITECTURE §8): parallel, `-m "not serial and <the seven lane exclusions>"` with `PARALLEL_PASS_FLAGS` (`-n auto --dist loadscope --max-worker-restart=0 --timeout=300`), then serial, `-m "serial and <the same exclusions>"` with `SERIAL_PASS_FLAGS` (`--timeout=300`: SIGALRM fails one hung test on POSIX, Windows' thread method stops the worker). Every OS adds `--default-suite-shard=K/N` (`tests/default_lane.py` packs whole files by recorded durations, else by count; shard 1 adds unsharded manifests). Two rules preserve the split:

- Mark `@pytest.mark.serial` (or list the file in `_SERIAL_TEST_FILES` in `tests/conftest.py`) only for a shared resource the test cannot isolate: a fixed port, a real daemon, a process-global its fixture cannot restore. An owned child, a private root or port 0 is not one. Under `-n` such a test flakes on kill/reap or port-reclaim timing or crashes its worker, and with `--max-worker-restart=0` a dead worker fails its co-located batch. A merely slow test is made faster or split, never `serial`.
- Keep every other test parallel-safe: `tmp_path` (never a fixed path), `monkeypatch.setenv`/`delenv`/`setattr`, no execution-order assumptions; a fixture-restored global stays parallel (`tests/conftest.py::_isolate_workspace_executor_globals`).

### The local battery entry point

`python scripts/run_tests.py` (`make test`) always runs the full Node and default Python battery; pytest arguments select focused work. `--selected [--base REF]` uses the commit selector (default base `HEAD`), prints tier/rule/module count and runs both marker passes on them. `--extra-test PATH` or `--full-tests` widens it with `--test-reason`. Missing required Node is red `NOT_RUN`. No hermetic proof is minted.

Full default overlaps duration-sized `--serial-shards` groups; `--sequential` restores the gate split; recheck overlap-only failures with it. `--evidence-dir DIR` gives each pass its own explicit `--ci-evidence-dir`; selected runs add `selection.json` and full inventory JSONL. Never export evidence via `PYTEST_ADDOPTS`: nested pytest could overwrite its parent's results. The CLI's `safe_test` (below) keeps `OUROBOROS_PREFLIGHT_REQUIRE_*`.

### Safe local verification and dirty browser candidates

Before imports/tests, use the stdlib launcher: it checks helper imports, scrubs owner config/credentials and prints isolated roots. Use a dependency-only venv: an editable install can import deleted modules from the checkout:

```bash
uv sync --locked --extra browser --group dev --no-install-project
python -I -S scripts/safe_test.py -- .venv/bin/python -m pytest tests/test_test_environment.py
export PLAYWRIGHT_BROWSERS_PATH="$(pwd)/.tmp-data-browsers"
python -I -S scripts/safe_test.py -- .venv/bin/python -m playwright install chromium webkit
OUROBOROS_RUN_UI_SMOKE=1 OUROBOROS_EXPECT_BROWSER_ENGINES=chromium,webkit \
  python -I -S scripts/safe_test.py -- .venv/bin/python -m pytest tests/ -m ui_browser --require-ui-browser
```

`--require-ui-browser` rejects narrow/empty collection, missing engines, unsettled cases and unapproved skips (`tests/browser_lane.py`). Temps: macOS/Linux `/tmp`, platform defaults elsewhere. `--temp-parent` rejects Git ancestors; custom paths may exceed socket limits. Owners export full streams, source (PERSISTENCE) and declared diagnostics (`SAFE_TEST_EVIDENCE`), reap, then release declared scratch on green AND red (failed launch: exit 2, `launch_error`); `SAFE_TEST_RETAINED`/`RELEASED` names the outcome. POSIX Ctrl-C gives the measured root 10 s to export, then TERM/10 s/forced reap; timeout/other signals start at TERM (124/128+N). Unknown inputs stay; non-Git has no recipe; untracked links record text, not targets. Marked run outputs inside a checkout are listed, not re-copied. Input gaps allow commands, not gate proof; unretainable disposable inputs refuse execution. `--keep`, retention, unreadable trees and export/teardown failure retain; bare pytest/hard-killed owners promise no cleanup. Only an unwritable default falls back into retained scratch. Declare diagnostics, not runtime archives. Launcher/preflight scrub provider/lane controls and run IDs; bare pytest keeps CI controls. `OUROBOROS_PYTEST_ACTIVE=1` prevents dependency installs; `MAC_CHROMIUM_TMPDIR` confines Chromium staging.

`tests/candidate_checkout.py` documents byte-faithful dirty capture, refusals and boot verification. `origin_proof=True` binds served static/Python by sentinel/VERSION suffix; installed distributions refuse. `ui_candidate_checkout` owns one copy/session; `direct_server_with_data` isolates case data/processes/ports; `case_checkout` verifies source/copy and restores overlays after proven teardown. Keep source stable: edits are detected, not locked. Older cancellation/system clones use independent full-history packs, not shared objects or shallow history.

`scripts/claudexor_lifecycle_smoke.py` witnesses real-engine custody under `python -I -S scripts/safe_test.py --temp-parent /tmp -- <python> scripts/claudexor_lifecycle_smoke.py`. No credentials or vendor task are needed; it stops its daemon children and keeps temporary roots. Only its fake-harness catalog is substituted (production excludes fakes); admission and custody are production.

### Reading CI failure evidence

`tests/ci_evidence.py` and `.github/actions/test-evidence` report Actions producer outcomes separately from testcase counts: passing cases never override a nonzero exit, skips and absent credentials stay explicit, raw JUnit is never uploaded, and diagnostic failures report `diagnostics_incomplete` without changing release eligibility. Shard proof is authoritative: `reconcile-shards` rejects missing, unreadable or red projections, wrong assignments, duplicate or unexecuted nodes and nonzero sessions, using each producer's highest attempt; a red session passes only via its pass's FLAKY/ACCEPTED `verdict.json`, whose rounds count as executed. Nested-pytest stack sampling and timeout attribution: ARCHITECTURE §8. A failed proof export or upload fails its attempt (no old-evidence fallback); step ceilings fail without cancelling the job, dead xdist workers report phase `crash` (the gate reads a timeout only from their callback's record: the thread banner dies with their stdout), and UI shards stop cooperatively between tests via `--session-timeout`. The `viewport`/`inflight` dispatches in `ui-browser-push.yml` are partial diagnostics, never full proof.

### Flaky and accepted tests

`scripts/ci_retry.py` retries a failed test once and continues unrun tests: retry PASS is listed as FLAKY; failed again or unrun is red. ACCEPTED needs `tests/accepted_failures.json` and an open issue labelled `not-a-product-bug`. Skill/provider runs are never retried. New scale shapes must fit post-merge's eight-minute cost contract (`tests/test_post_merge_workflow.py`).

### The tiered commit and CI gates

`commit_reviewed` runs `test_selection.FLOOR_TESTS`, changed tests, direct AST importers and authored widening. Missing/empty floor modules block. Selection includes dirty/untracked candidate files against `HEAD` before commit and `HEAD~1` after (a body candidate: its serving fork). Automatic CI follows ARCHITECTURE §8; local checks and review remain.

Imports are a minimum: `extra_tests` covers indirect/dynamic, prompt, skill and cross-language consumers; `full_tests`, unresolved impact. Both need `test_reason`, retained until adoption. Verifier/cutover changes force full: runner/selector, floor, shared helpers/data, pytest configuration, non-version dependency edits and `_CUTOVER`. No product fan-out threshold. Records bind tier, candidate/base/policy, reasons and execution counts; missing/duplicate execution blocks.

Full is required for managed updates (their forced run), non-candidate evolution publication, post-adoption boot push and tag/dispatch CI. Local adoption also accepts an exact selected run against the serving HEAD, under the loaded policy, covering the reviewed switch delta and declared widening; it withholds both branch and tag. Ordinary serving-checkout commits and `promote_to_stable` have no tier gate: a later push can publish a withheld adoption. New HEAD needs new evidence; skip/`None` proves nothing. Post-commit failures retain ordinary commits with warnings; evolution withholds publication, managed updates roll back or retain `gate_blocked`.

`preflight_runner.run_hermetic_pytest` runs Node, parallel `not serial`, then timed `serial` in one disposable checkout; `LANE_EXCLUSION_EXPR`/`*_PASS_FLAGS` own exclusions and flags. Missing Node/plugins blocks. `preflight_node.node_environment` fingerprints bundled-first Node on gate/CLI child PATH; other environments retain their policy. The acorn `web/tests/no_undef.test.js` and lockfile-frozen CI ESLint remain independent. The gate shares `safe_test`'s temp default without `--temp-parent`; small/noexec `/tmp` can fail.

Candidate capture uses one hardened worktree-vs-`HEAD` binary diff; assembly failure is `PREFLIGHT_CANDIDATE_ASSEMBLY`. It scrubs `PYTEST_*`, `NODE_OPTIONS`, settings and credentials; plugin probes run outside candidate control. Post-commit baselines include `HEAD~1` to catch suite deletion; exit status owns the verdict. `OUROBOROS_PREFLIGHT_SERIAL=1` is an explicit rollback lever, never silent fallback.

Containment is checked even after green (§9 "Test runs"); unknown or surviving members block and retain the tree. Worker crash, timeout, missing plugin, containment and test failures keep distinct diagnoses. Proof binding: ARCHITECTURE §6.

### GitHub Actions: secrets in step-level `if:` conditions

GitHub Actions rejects `secrets.*` inside a step-level `if:` expression, and a step's own `env:` block is not visible to that step's `if:`. Derive a non-secret boolean in the job-level `env:` block, gate steps with that boolean, and map the actual credentials only inside the first-party steps that need them, so later SBOM and attestation steps inherit none of them:

```yaml
jobs:
  build:
    env:
      HAS_APPLE_SIGNING: ${{ matrix.os == 'macos-latest' && secrets.BUILD_CERTIFICATE_BASE64 != '' && secrets.P12_PASSWORD != '' && secrets.KEYCHAIN_PASSWORD != '' && secrets.APPLE_TEAM_ID != '' && 'true' || 'false' }}
    steps:
      - name: Import Apple signing certificate
        if: env.HAS_APPLE_SIGNING == 'true'
        env:
          BUILD_CERTIFICATE_BASE64: ${{ secrets.BUILD_CERTIFICATE_BASE64 }}
      - name: Cleanup keychain
        if: always() && matrix.os == 'macos-latest' && env.HAS_APPLE_SIGNING == 'true'
```

```yaml
# ❌ WRONG — workflow fails to parse
- name: Bad
  if: secrets.BUILD_CERTIFICATE_BASE64 != ''   # parse error
  env:   # not visible to this step's if:
    P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
```

`tests/test_build_scripts.py::TestMacOSSigning::test_ci_uses_env_context_for_condition` enforces this for `ci.yml` only.

### Apple signing & notarization (macOS Build job)

Prereleases may be unsigned and say so (ARCHITECTURE §8 "Build scripts"). Only the non-secret `HAS_APPLE_SIGNING` gate is job-wide: certificate and keychain values exist only in the import step, Apple ID values only in the build step, and an `always()` cleanup keeps signing material from persisting across runs. With the Apple ID values present, `build.sh` notarizes fail-closed: the app, then the DMG, must each read `Accepted` in Apple's JSON and get a validated staple. A signed stable DMG's smoke requires both tickets and `spctl` accepting the copied-out app as `Notarized Developer ID`, so signing without notarization stops the release; receipt and notes carry the `notarized`/`not_notarized` state.

### Windows Authenticode release signing (SSL.com eSigner)

The nonsecret repository variable `ESIGNER_CERT_SHA1` (certificate SHA-1 thumbprint) selects the mode for every `v*` tag, prereleases included. Unset (forks, repositories without a certificate), no job binds a signing Environment or sees a signing secret, and receipt and notes disclose an unsigned Windows ZIP. Set, any configuration, tool, signature, timestamp or digest failure stops the release, never falling back to unsigned.

The `build` shard always packs the unsigned ZIP (`scripts/pack_windows_archive.ps1`) and hands it over by digest. When configured, `windows-sign` (Environment `windows-release-signing`) runs `scripts/sign_windows_release.ps1` on exactly the files `scripts/windows_signing_catalog.json` names: `Ouroboros.exe` and the embedded `python.exe` and `pythonw.exe`, pinned to upstream hashes beside the downloader's pin (a Python upgrade restates both). It checks every file before the first billed request and never replaces an existing signature. Secret-free `windows-proof` re-verifies all three and records the signer and each file's pre- and post-signing hashes, or `unsigned`; assembly requires exactly the catalogued files and `ESIGNER_CERT_SHA1`'s signer.

Setup: the repository owner creates the Environment `windows-release-signing` with a `v*` tag rule and no required reviewer, so a tag signs without manual approval. The certificate holder stores `ESIGNER_USERNAME`, `ESIGNER_PASSWORD`, `ESIGNER_CREDENTIAL_ID` and `ESIGNER_TOTP_SECRET` as environment-scoped secrets (`gh secret set <NAME> --env windows-release-signing`), because repository secrets are readable from any branch push, then sets `ESIGNER_CERT_SHA1`, which turns signing on. Never paste values into an issue, chat, workflow or log; an exposed TOTP seed is regenerated at SSL.com. Each `windows-sign` run or re-run spends up to three eSigner signings, kept if it fails; beyond the subscription tier they are invoiced, and the repository cannot see the holder's balance.

Check an extracted ZIP with `scripts/verify_windows_signature.ps1 -ExtractedRoot <folder holding Ouroboros\> -ExpectedThumbprint <signerThumbprint from release-evidence.json>` (SDK `signtool`). Mocked tests prove no credential, timestamp acceptance or shown publisher; live signing stays unverified until a published release is checked and launched on Windows. Other bundled files carry at most a vendor's signature; venv launcher templates, environments made from them and later Git updates stay unsigned; SmartScreen and antivirus still weigh reputation.

### Release proof capsule

ARCHITECTURE §8 and `.github/workflows/ci.yml` own the artifact pipeline. Preserve these invariants:

- Valid tags start builds and configured signing beside tests. Red tests still spend notarization and eSigner signings and record attestations, but publish no Release.
- Proof/notes carry each job's actual conclusion and the run's FLAKY and ACCEPTED tests; no tag waives a lane.
- Publication is draft-first with per-tag concurrency. Revalidate the remote annotated tag against the event SHA before drafting and publishing; reruns never overwrite published releases.
- Astra/RED OS smokes are evidence, not release authority: third-party registry reachability is outside pipeline control.
- Native Linux requires Jammy x86_64 and authenticated source indices; missing exact sources fail capture. Other builders keep the host backend. CI retains complete/partial sources for 30 days; recipient delivery remains a release obligation.
- Release AppImage smoke stays headless. [Native GTK qualification](../../packaging/appimage/README.md) is separate.
- Local-only `OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1` skips host-library installation, not browser bundling; disclose unproven host compatibility.
- Later checksum inventories are not build-time provenance, an SBOM, or original packaged smoke evidence.
