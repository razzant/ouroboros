# Build & CI

Rules for dependency locks, pytest lanes, CI and commit-gate tests, and Actions secrets. The gate stays reproducible and candidate-independent; ARCHITECTURE §8 owns the job graph, triggers and release chain.

### Python dependency locks

`pyproject.toml` + `uv.lock` are the one dependency authority; `requirements-runtime.lock` and `requirements.txt` are generated projections (ARCHITECTURE §8 "Build scripts"). A dependency change updates the metadata, runs `uv lock`, regenerates the runtime export with the exact README command, and leaves the CI clean-diff check green; the pinned `tool.uv.required-version` and the digest-pinned `setup-uv` action make a resolver change deliberate rather than an ambient CI upgrade. Documentation may pair the checkout-free `uv tool install` form with a full commit SHA to pin the source revision, but must not claim it locks dependencies or call it a release-artifact install or a contributor development environment.

### Pytest marker lanes

Default local pytest excludes seven costly or environment-dependent lanes — `integration`, `browser`, `ui_browser`, `ui_browser_docker`, `portable_detail`, `skill_smoke` and `size_ratchet` (`pyproject.toml` `addopts`, mirrored by `LANE_EXCLUSION_EXPR`) — and CI opts into each explicitly (job topology and provider matrix: ARCHITECTURE §8 "CI topology"):

- `integration` runs real provider checks. Split manual/tag jobs by `credential_required`, never waive a mixed pytest exit: optional failures stay red outside release dependencies. Required missing credentials fail; quota/429/5xx/timeout stay inconclusive, contract/auth/model/reasoning/tool 4xx red. `provider_release_floor.py` blocks official tags unless every required canary passed; pre-tag qualification must check those rows too, not aggregate workflow green. No provider secrets in PRs. Keyless `tests/system_e2e/` adds `serial` and `OUROBOROS_E2E_DEEP`: only `system-e2e-mock` runs `mock`; S39 `container` is operator-run (ARCHITECTURE §8).
- `browser` / `ui_browser` / `ui_browser_docker` launch real Playwright engines (agent browser tools / the host UI / the `ouroboros-web:test` container; the docker lane skips cleanly when Docker is unavailable locally). The marker is the source of truth for what a lane collects. `portable_detail` covers build and portable artifact invariants. Manual/tag `ui-smoke` runs the complete `ui_browser` lane (required invocation: "Safe local verification and dirty browser candidates" below). None of these joins the default local run or needs a paid provider.
- `skill_smoke` installs pinned official skills (`tests/test_skill_smoke_official.py`) from the live catalog on three OSes with real network/pip; red means investigate, never fallback-skip. Paid review runs first on Ubuntu and alone holds the provider key (ARCHITECTURE §8); a missing key fails. Its single retry requires settled reviewer work and keeps state: production replay/cycle rules still apply, pending/unknown never passes.
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

CI runs the default suite in two passes, each owned by `scripts/run_tests.py` (`.github/workflows/ci.yml`, runners `quick-test` / `full-test-runs`): a parallel pass, `-m "not serial and <the seven lane exclusions>"` with `PARALLEL_PASS_FLAGS` (`-n auto --dist loadscope --max-worker-restart=0 --timeout=300 --timeout-method=thread`), then a flag-free serial pass, `-m "serial and <the same exclusions>"`. Windows alone adds `--default-suite-shard=K/4` (`tests/default_lane.py` packs whole files by collected count; shard 1 adds the unsharded manifest witnesses). Two rules preserve the split:

- Mark `@pytest.mark.serial` (or list the file in `_SERIAL_TEST_FILES` in `tests/conftest.py`) only for a shared resource the test cannot isolate: a fixed port, a real daemon, a process-global its fixture cannot restore. An owned child, a private root or port 0 is not one. Under `-n` such a test flakes on kill/reap or port-reclaim timing or crashes its worker, and with `--max-worker-restart=0` a dead worker fails its co-located batch. A merely slow test is made faster or split, never `serial`: that pass has no per-test timeout.
- Keep every other test parallel-safe: `tmp_path` (never a fixed path), `monkeypatch.setenv`/`delenv`/`setattr`, no execution-order assumptions; a fixture-restored global stays parallel (`tests/conftest.py::_isolate_workspace_executor_globals`).

### The local battery entry point

`python scripts/run_tests.py` (`make test`) always runs the full Node and default Python battery; pytest arguments select focused work. `--selected [--base REF]` uses the commit selector (default base `HEAD`), prints tier/rule/module count and runs both marker passes with paths. `--extra-test PATH` or `--full-tests` widens it with `--test-reason`. Missing required Node is red `NOT_RUN`. No hermetic proof is minted.

Full default overlaps bounded `--serial-shards` groups; `--sequential` restores the gate split. Recheck overlap-only failures sequentially. `--evidence-dir DIR` gives each pass its own explicit `--ci-evidence-dir`; selected runs also write `selection.json` with scope and reconciled inventory. Never export evidence via `PYTEST_ADDOPTS`: nested pytest could overwrite its parent's results.

### Safe local verification and dirty browser candidates

Before application imports or local tests, use the stdlib boundary launcher. It checks the helper's imports, scrubs owner configuration and credentials, and prints isolated roots before Python starts. Use a dependency-only venv: an editable project could import deleted modules from the source checkout:

```bash
uv sync --locked --extra browser --group dev --no-install-project
python -I -S scripts/safe_test.py -- .venv/bin/python -m pytest tests/test_test_environment.py
export PLAYWRIGHT_BROWSERS_PATH="$(pwd)/.tmp-data-browsers"
python -I -S scripts/safe_test.py -- .venv/bin/python -m playwright install chromium webkit
OUROBOROS_RUN_UI_SMOKE=1 OUROBOROS_EXPECT_BROWSER_ENGINES=chromium,webkit \
  python -I -S scripts/safe_test.py -- .venv/bin/python -m pytest tests/ -m ui_browser --require-ui-browser
```

`--require-ui-browser` fails for narrowed/empty collection, missing engines, unsettled cases or unregistered skips (`tests/browser_lane.py`). `test_environment.default_temp_parent` gives `safe_test` and hermetic preflight `/tmp` on macOS for short Unix-socket paths; other platforms keep their system temp default. Explicit `--temp-parent` wins, rejects Git checkouts and may exceed socket-path limits. Launcher/pytest trees remain (`SAFE_TEST_RETAINED <path>`): exit alone proves no descendant teardown. `test_environment.py` owns isolated data/settings/app/HOME/workspace roots, not an OS sandbox. Bare pytest keeps explicit provider/lane controls for integration CI; launcher/preflight scrub them and run identities, so tests cannot attach a host engine. With `OUROBOROS_PYTEST_ACTIVE=1`, `supervisor/git_ops_reset.py` installs no dependencies; production is unchanged. `MAC_CHROMIUM_TMPDIR` shares the isolated temp root because macOS Chromium ignores `TMPDIR` for download staging.

`tests/candidate_checkout.py` runs the shared UI and keyless wait/repair fixtures against a byte-faithful copy of the dirty candidate, tracked worktree bytes plus non-ignored new files, with source and copy identities verified around capture and before each server incarnation (inputs and the explicit-failure list: the module docstring; summary: ARCHITECTURE §8 "CI topology"). `origin_proof=True` adds a static sentinel under `web/` and a `+candidate.<token>` VERSION suffix, so startup and restart prove that the served static tree and the Python behind `/api/health`'s `runtime_version` came from this copy; an interpreter carrying an installed `ouroboros` distribution is refused. Concurrent edits are detected, not locked out: keep the source stable for the whole run.

`scripts/claudexor_lifecycle_smoke.py` is the real-engine custody witness: run it under `python -I -S scripts/safe_test.py --temp-parent /tmp -- <python> scripts/claudexor_lifecycle_smoke.py`. It needs no credentials or vendor task, always stops its own daemon children and keeps its temporary roots. Only the fake-harness catalog projection is a test substitute (the real catalog excludes fakes); every delegated admission and custody effect is production. CI runs it in `claudexor-platform-gate.yml` on all three desktop OSes, apart from the gateway-only fixture lane.

### Reading CI failure evidence

`tests/ci_evidence.py` and `.github/actions/test-evidence` report Actions producer outcomes separately from testcase counts: passing cases never override a nonzero exit, skips and absent credentials remain explicit, and raw JUnit is never uploaded. Diagnostic failures report `diagnostics_incomplete` without changing release eligibility. Shard proof is authoritative: `reconcile-shards` rejects missing/unreadable/red projections, wrong assignments, duplicate or unexecuted nodes, and nonzero sessions, using each producer's highest attempt. UI uses four `ui-shard` jobs plus `ui-manifest`; Windows default passes use independent manifests on shard 1 and `full-test` reconciliation. Proof export/upload failures also fail the producing attempt, preventing fallback to old evidence. Step ceilings fail without cancelling the job, leaving evidence steps runnable; dead xdist workers report phase `crash`. UI shards additionally stop cooperatively between tests via `--session-timeout`. The `viewport`/`inflight` dispatches in `ui-browser-push.yml` are partial diagnostics, never full proof.

### The tiered commit and CI gates

Ordinary `commit_reviewed` runs the floor in `test_selection.FLOOR_TESTS`, changed test modules, direct AST importers of changed product modules, and authored widening. Each floor entry names its invariant; missing or empty modules block. Selection reads the assembled candidate, dirty/untracked files included, against `HEAD` before commit and `HEAD~1` afterwards. Ordinary PR/branch cloud CI is disabled; local checks and independent review are not.

Imports are a minimum: `extra_tests` covers indirect/dynamic consumers, prompts, skills and cross-language effects; `full_tests` covers unresolved impact. Both require `test_reason`, retained in the commit record. Verifier changes force full: runner/selector, floor, shared test helpers or data, pytest configuration, and dependency locks beyond a release bump. Product fan-out alone never triggers a threshold. Records name tier, candidate/base/policy and reasons, with execution counts; missing or duplicate execution blocks.

Full qualification remains mandatory for managed updates, evolution publication and exact-commit adoption (existing full evidence only), and CI on release tags and dispatch. A selected proof never mints `FullGateReceipt` or satisfies these consumers. Full receipts may reuse a complete run of the same HEAD/tree/index/workload and installation; a new HEAD needs a new run. Skip/no-suite `None` grants no proof. Ordinary post-commit failures preserve the local commit with a warning; evolution refuses publication, and managed updates roll back or retain `gate_blocked` until recovery.

`preflight_runner.run_hermetic_pytest` runs the entire Node lane, parallel `not serial` and flag-free `serial` in one disposable checkout. `LANE_EXCLUSION_EXPR`/`PARALLEL_PASS_FLAGS` own exclusions and flags. Missing required Node/plugins blocks. `preflight_node.node_environment` puts bundled-first Node on the gate's and CLI's child PATH, which is fingerprinted; other pytest and child environments keep their policy. The `web/tests/no_undef.test.js` acorn walker and CI's lockfile-frozen ESLint (`web/eslint.config.js`, `npm ci`) remain independent checks.

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
  env:                                          # not visible to this step's if:
    P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
```

`tests/test_build_scripts.py::TestMacOSSigning::test_ci_uses_env_context_for_condition` enforces this for `.github/workflows/ci.yml` only.

### Apple signing & notarization (macOS Build job)

Prerelease artifacts may be unsigned and must report that state; stable publication applies the configured signing and notarization policy rather than implying credentials or success that were absent (ARCHITECTURE §8 "Build scripts"). Only the non-secret `HAS_APPLE_SIGNING` gate is job-wide: certificate and keychain values exist only in the import step, Apple ID notarization values only in the first-party build step, and cleanup runs under `always()` plus the matrix/env guards, so signing material never persists across runs. Notary and stapler failures are soft outcomes recorded through `NOTARIZE_OUTCOME` (`build.sh`), so a transient Apple service problem does not silently drop an otherwise valid signed artifact.

### Windows Authenticode release signing (SSL.com eSigner)

The nonsecret repository variable `ESIGNER_CERT_SHA1` (the certificate's SHA-1 thumbprint) selects the mode for every `v*` tag, prereleases included. Unset (forks, repositories without a certificate): no job binds a signing Environment or sees a signing secret, and the release publishes an explicitly unsigned Windows ZIP that its receipt and notes disclose. Set: signing is required, and any configuration, tool, signature, timestamp or digest failure stops the whole release rather than falling back to an unsigned ZIP.

Secrets stay out of the build: the `build` matrix's Windows shard always packs the unsigned ZIP with `scripts/pack_windows_archive.ps1`, the one packer for local and both release modes, and hands it over by digest. When configured, a separate `windows-sign` job binds the `windows-release-signing` Environment and runs `scripts/sign_windows_release.ps1` (signs only `Ouroboros.exe`, verifies signer and timestamp, repacks; vendor output is never shown). The secret-free `windows-proof` job re-verifies the signature, runs the packaged smoke and records the signer or `unsigned`; assembly requires a receipt matching `ESIGNER_CERT_SHA1`. Tool pin, Java runtime and exit-code handling live in the scripts.

Setup: the repository owner creates the Environment `windows-release-signing` with a `v*` tag rule and no required reviewer, so a tag signs without manual approval. The certificate holder stores `ESIGNER_USERNAME`, `ESIGNER_PASSWORD`, `ESIGNER_CREDENTIAL_ID` and `ESIGNER_TOTP_SECRET` as environment-scoped secrets (`gh secret set <NAME> --env windows-release-signing`), because repository secrets are readable from any branch push, then sets `ESIGNER_CERT_SHA1`. Never paste values into an issue, chat, workflow or log; an exposed TOTP seed is regenerated at SSL.com. Each `windows-sign` run, re-runs included, spends one eSigner signing, and signings beyond the subscription tier are invoiced; the repository cannot see the holder's balance.

Anyone can check a downloaded ZIP with `scripts/verify_windows_signature.ps1 -Executable Ouroboros\Ouroboros.exe -ExpectedThumbprint <signerThumbprint from release-evidence.json>` (needs the Windows SDK `signtool`). The tests do not prove that the credential signs, that the runner accepts the timestamp, or which publisher a downloaded app shows; until a published release is checked and launched on Windows, live signing is unverified. Bundled runtimes and later Git code updates are not signed, and SmartScreen still weighs reputation.

### Release proof capsule

The artifact pipeline (per-platform archive smokes, native Linux packages, the AppImage custody chain, SBOM and attestation binding, the seven-required-desktop plus optional-Android release job, and `release-preflight`'s tag behaviour) lives in ARCHITECTURE §8 and `.github/workflows/ci.yml`. The honesty invariants a change must preserve:

- A failed test prerequisite makes `release-preflight` red but lets the desktop build run as a diagnostic rehearsal, which may consume configured macOS signing/notarization and records attestations in the repository and the public transparency log (configured Windows signing requires a green preflight, so a red run has no signed Windows ZIP); artifacts stay downloadable from the run, no GitHub Release is published, and `release` still requires a successful preflight.
- Publication is draft-first under a per-tag concurrency group; the remote annotated tag is revalidated against the event SHA before draft creation and again before publication, and a published release is never overwritten by a rerun.
- Vendor-distribution smokes (Astra, RED OS) are reported evidence, never release authority: third-party registry reachability is outside the publication pipeline's control.
- The AppImage smoke makes no native GTK/Qt claim; packaged native webview coverage is a separate Linux distribution contract.
- `OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1` is a local-builder escape hatch only: it skips Playwright's host-library installation, not browser-binary bundling, and a build using it must disclose that browser host compatibility was not locally proven.
- Never represent a later checksum inventory as build-time provenance, an SBOM, or packaged smoke evidence that the original build did not create.
