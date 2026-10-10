# Build & CI

This chapter maps dependency locks, opt-in pytest lanes, CI's parallel/serial split, its hermetic commit-gate mirror, and Actions secret gating. The gate must be reproducible and independent of the candidate; the CI job graph, its triggers and the release chain live in ARCHITECTURE §8.

### Python dependency locks

`pyproject.toml` + `uv.lock` are the one dependency authority; `requirements-runtime.lock` and `requirements.txt` are generated projections, never authorities (ARCHITECTURE §8 "Build scripts"). A dependency change updates the metadata, runs `uv lock`, regenerates the runtime export with the exact README command, and leaves the CI clean-diff check green; the pinned `tool.uv.required-version` and the digest-pinned `setup-uv` action make a resolver change deliberate rather than an ambient CI upgrade. Documentation may pair the checkout-free `uv tool install` form with a full commit SHA to pin the source revision, but must not claim it locks dependencies or call it a release-artifact install or a contributor development environment.

### Pytest marker lanes

Default local pytest excludes seven costly or environment-dependent lanes — `integration`, `browser`, `ui_browser`, `ui_browser_docker`, `portable_detail`, `skill_smoke` and `size_ratchet` (`pyproject.toml` `addopts`, mirrored by `LANE_EXCLUSION_EXPR`) — and CI opts into each explicitly (job topology and provider matrix: ARCHITECTURE §8 "CI topology"):

- `integration` runs real provider checks. Missing core credentials are red in the official repository job; quota/429/5xx/timeout may be typed inconclusive, while contract/auth/model/reasoning/tool 4xx stay red; on an official-repository release tag `tests/provider_release_floor.py` also fails unless every required canary passed. Secretless wire contracts stay in ordinary pull-request tests; provider secrets never move into PR jobs. The keyless `tests/system_e2e/` scenario lane rides the same marker plus `serial` and the `OUROBOROS_E2E_DEEP=mock` env gate, so only the dedicated `system-e2e-mock` job runs it (ARCHITECTURE §8 "System E2E suite").
- `browser` / `ui_browser` / `ui_browser_docker` launch real Playwright engines (agent browser tools / the host UI / the `ouroboros-web:test` container; the docker lane skips cleanly when Docker is unavailable locally). The marker is the source of truth for what a lane collects. `portable_detail` covers build and portable artifact invariants. `ui-smoke` runs the complete `ui_browser` lane on non-documentation PRs and on every push to `ouroboros` (required invocation: "Safe local verification and dirty browser candidates" below). None of these joins the default local run or needs a paid provider.
- `skill_smoke` installs the pinned official skills from the live catalog (list in `tests/test_skill_smoke_official.py`) as the dedicated 3-OS CI job, with real network and real pip; red means investigate, and there is deliberately no fallback-skip. Its paid review tier is a separate ubuntu-only step, ordered first and alone carrying the provider key (why: ARCHITECTURE §8 "CI topology"); a missing key is a hard red, not a skip.
- `size_ratchet` carries the live-repo size gates and is the only blocking surface for repository size (rules under "Module Size & Complexity"; base fallback and the fail-closed rule: ARCHITECTURE §8 "CI topology").

`skill_smoke` and `size_ratchet` tests never also carry the `serial` marker or join `_SERIAL_TEST_FILES`: the `and not <lane>` markexprs are the lane barrier, and single-lane assignment keeps each test's placement unambiguous. A new opt-in lane registers its marker in `pyproject.toml`, adds a collect-only zero-test guard in CI, and keeps the default local addopts free of network and Docker requirements.

### Parallel CI and the `serial` marker

CI runs the default suite in two passes (`.github/workflows/ci.yml`, jobs `quick-test` / `full-test`): a parallel pass, `-m "not serial and <the seven lane exclusions>"` with `PARALLEL_PASS_FLAGS` (`-n auto --dist loadscope --max-worker-restart=0 --timeout=300 --timeout-method=thread`), then a flag-free serial pass, `-m "serial and <the same exclusions>"`. Two rules keep new tests from breaking that:

- Mark real-process and real-port tests, and tests that mutate process-global state without reliable fixture isolation, `@pytest.mark.serial` (or add the file to `_SERIAL_TEST_FILES` in `tests/conftest.py`): under `-n` such a test flakes on kill/reap or port-reclaim timing or crashes its worker, and with `--max-worker-restart=0` a dead worker fails its whole co-located batch as spurious failures in unrelated files.
- Keep every other test parallel-safe so it stays in the fast pass: `tmp_path` (never a fixed path), `monkeypatch.setenv`/`delenv`/`setattr` for environment and attribute changes, no execution-order assumptions. A module-global mutation reliably snapshot-and-restored by a fixture may stay in the parallel pass (pattern: `tests/conftest.py::_isolate_workspace_executor_globals`).

### The local battery entry point

`python scripts/run_tests.py` (also `make test`) is the documented local run: the node lane (a missing node is a red `NOT_RUN`), then every default-lane test, never skipping a test on the strength of an earlier run; extra arguments are forwarded to pytest as a focused run. It imports `LANE_EXCLUSION_EXPR` and `PARALLEL_PASS_FLAGS` from the gate rather than restating them. Its default mode is one xdist run in which every serial file is pinned to one of a few file-sharded groups (`--serial-shards`, `tests/conftest.py::_pin_lane_groups`), so serial tests overlap the parallel ones and also get the parallel flags' per-test timeout; `--sequential` keeps the gate's two-pass split. CI and the commit gate keep that split and stay the authority: a failure seen only in overlapped mode is re-checked with `--sequential` before it is believed. Enforcement: `tests/test_run_tests_script.py`.

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

`--require-ui-browser` fails on narrowed or empty collection, missing engines, unsettled cases, or skips outside `tests/browser_lane.py`'s reviewed platform registry. `--temp-parent` (`/tmp` on macOS: short socket paths) refuses Git checkouts. Launcher and pytest trees persist (`SAFE_TEST_RETAINED <path>`), because an exit status does not prove a command's descendants gone. `ouroboros/test_environment.py` owns the disposable data, settings, app, HOME and workspace defaults for pytest, preflight and their server children; it is not an OS sandbox. Bare pytest keeps explicitly supplied provider and lane controls for integration CI; the launcher and preflight scrub them, and no run identity passes through, so no test reaches a host engine. Under an active test boundary (`OUROBOROS_PYTEST_ACTIVE=1`) `supervisor/git_ops_reset.py` installs no dependencies for any caller; production is unchanged. `MAC_CHROMIUM_TMPDIR` shares the disposable temp root, because macOS Chromium ignores `TMPDIR` for its initial download staging.

`tests/candidate_checkout.py` runs the shared UI and keyless wait/repair fixtures against a byte-faithful copy of the dirty candidate, tracked worktree bytes plus non-ignored new files, with source and copy identities verified around capture and before each server incarnation (inputs and the explicit-failure list: the module docstring; summary: ARCHITECTURE §8 "CI topology"). `origin_proof=True` adds a static sentinel under `web/` and a `+candidate.<token>` VERSION suffix, so startup and restart prove that the served static tree and the Python behind `/api/health`'s `runtime_version` came from this copy; an interpreter carrying an installed `ouroboros` distribution is refused. Concurrent edits are detected, not locked out: keep the source stable for the whole run.

`scripts/claudexor_lifecycle_smoke.py` is the real-engine custody witness: run it under `python -I -S scripts/safe_test.py --temp-parent /tmp -- <python> scripts/claudexor_lifecycle_smoke.py`. It needs no credentials or vendor task, always stops its own daemon children and keeps its temporary roots. Only the fake-harness catalog projection is a test substitute (the real catalog excludes fakes); every delegated admission and custody effect is production. CI runs it in `claudexor-platform-gate.yml` on all three desktop OSes, apart from the gateway-only fixture lane.

### Reading CI failure evidence

Test reports are informational projections (`tests/ci_evidence.py`, published through `.github/actions/test-evidence`), never the verdict: a report states the Actions producer outcome separately from testcase counts, so a passing case never overrides a nonzero session exit; skips and unavailable credentials stay explicit; raw JUnit stays out of uploads. An upload or report failure reads `diagnostics_incomplete` and does not alter release eligibility, with one exception: the sharded UI lane (`ui-browser.yml`, four `ui-shard` jobs under a cooperative `--session-timeout` plus one unsharded `ui-manifest` collection) is proven only by its shards' and manifest's projections, so `ui-smoke` (`tests/ci_evidence.py reconcile-shards`) is red, naming each gap, when a projection is missing, unreadable or red, when a shard differs from its assigned manifest slice or fails to execute it, or when any judged session ended nonzero; after a re-run the proof is the highest attempt. Each producer step has its own wall-clock ceiling that fails the step without cancelling the job, so the evidence step still runs after a hang; a dead xdist worker is phase `crash`. `ui-browser-push.yml` dispatched with `viewport` or `inflight` runs one partial diagnostic scenario, and partial success is never full proof.

### The commit gate mirrors the CI split

`ouroboros/preflight_runner.py::run_hermetic_pytest` mirrors CI in one disposable checkout and scrubbed temporary data root: the node test lane (`cd web && node --test tests/*.test.js`; a candidate without web tests never requires node, while a present suite cannot silently disappear when node is missing), then the same two pytest passes (parallel `not serial`, then flag-free `serial`). `LANE_EXCLUSION_EXPR` and `PARALLEL_PASS_FLAGS` are the executable SSOT, pinned against both CI jobs. The candidate is one hardened worktree-vs-`HEAD` binary diff, and a capture or apply failure is the typed `PREFLIGHT_CANDIDATE_ASSEMBLY` hard block, never a test failure (what the proof binds and why every workload binds HEAD: ARCHITECTURE §6 "Hermetic preflight proof"). The browser no-undef check has two layers: the dependency-free acorn walker `web/tests/no_undef.test.js` is the hermetic gate's, and both CI jobs additionally run ESLint's `no-undef` (`web/eslint.config.js`, installed with `npm ci`) as an independent second opinion, CI-only. Contributor rules:

- The candidate cannot weaken the pass: `PYTEST_*`, `NODE_OPTIONS` and owner runtime state (`OUROBOROS_*`, secret-suffixed keys and every settings key `config.apply_settings_to_env` projects) are scrubbed, so the verdict cannot depend on the operator's install profile; required plugins are probed outside candidate control; post-commit checks also inspect `HEAD~1`, so suite deletion cannot hide after the commit exists; exit status owns the verdict, rendered diagnostics do not. `OUROBOROS_PREFLIGHT_SERIAL=1` is the explicit temporary rollback lever, never a silent fallback.
- A red post-commit gate is warning-only for an ordinary commit (the local commit is preserved for forensics); evolution publication refuses to auto-push while the warning stands; inside a managed update the gate blocks boot promotion and routes through rollback, and an incomplete rollback leaves `gate_blocked` so boot retries recovery instead of promoting the rejected merge.
- The managed mandate is "the full suite provably ran green on the exact committed tree", not "run it twice": the reuse authority is the process-held `ctx._preflight_test_proof`, never the durable `tests_evidence` record or the event log. A new commit or a restart needs a new run, and a skip, no applicable suite or a mocked `None` return mints no proof.
- Process containment is unconditional, after a green pass too: Windows uses a kill-on-close Job Object; POSIX uses an environment membership token plus a process-group backstop, and promises honest detection with a fail-closed verdict for attributed members, not guaranteed teardown of an arbitrary detached process (`tests/test_preflight_process_containment.py`). A crashed worker, a timeout-killed worker, a missing plugin, containment failure and ordinary test failure keep distinct diagnostics.
- Mark process, port and global-state tests `serial`; make a merely slow test faster or split it, because the gate's and CI's serial pass has no per-test timeout and a serial test consumes the remaining total gate budget.

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

`tests/test_build_scripts.py::TestMacOSSigning::test_ci_uses_env_context_for_condition` enforces this for `.github/workflows/ci.yml` only; other workflow files are not scanned by it.

### Apple signing & notarization (macOS Build job)

Prerelease artifacts may be unsigned and must report that state; stable publication applies the configured signing and notarization policy rather than implying credentials or success that were absent (ARCHITECTURE §8 "Build scripts"). Only the non-secret `HAS_APPLE_SIGNING` gate is job-wide: certificate and keychain values exist only in the import step, Apple ID notarization values only in the first-party build step, and cleanup runs under `always()` plus the matrix/env guards, so signing material never persists across runs. Notary and stapler failures are soft outcomes recorded through `NOTARIZE_OUTCOME` (`build.sh`), so a transient Apple service problem does not silently drop an otherwise valid signed artifact.

### Windows Authenticode release signing (SSL.com eSigner)

The nonsecret repository variable `ESIGNER_CERT_SHA1` (the certificate's SHA-1 thumbprint) selects the mode for every `v*` tag, prereleases included. Unset (forks, repositories without a certificate): no job binds a signing Environment or sees a signing secret, and the release publishes an explicitly unsigned Windows ZIP that its receipt and notes disclose. Set: signing is required, and any configuration, tool, signature, timestamp or digest failure stops the whole release rather than falling back to an unsigned ZIP.

Secrets stay out of the build: the `build` matrix's Windows shard always packs the unsigned ZIP with `scripts/pack_windows_archive.ps1`, the one packer for local and both release modes, and hands it over by digest. When configured, a separate `windows-sign` job binds the `windows-release-signing` Environment and runs `scripts/sign_windows_release.ps1` (signs only `Ouroboros.exe`, verifies signer and timestamp, repacks; vendor output is never shown). The secret-free `windows-proof` job re-verifies the signature, runs the packaged smoke and records the signer or `unsigned`; assembly requires a receipt matching `ESIGNER_CERT_SHA1`. Tool pin, Java runtime and exit-code handling live in the scripts.

Setup: the repository owner creates the Environment `windows-release-signing` with a `v*` tag rule and no required reviewer, so a tag signs without manual approval. The certificate holder stores `ESIGNER_USERNAME`, `ESIGNER_PASSWORD`, `ESIGNER_CREDENTIAL_ID` and `ESIGNER_TOTP_SECRET` as environment-scoped secrets (`gh secret set <NAME> --env windows-release-signing`), because repository secrets are readable from any branch push, then sets `ESIGNER_CERT_SHA1`, which turns signing on. Never paste values into an issue, chat, workflow or log; an exposed TOTP seed is regenerated at SSL.com. Each `windows-sign` run, re-runs included, spends one eSigner signing, and signings beyond the subscription tier are invoiced; the repository cannot see the holder's balance.

Anyone can check a downloaded ZIP with `scripts/verify_windows_signature.ps1 -Executable Ouroboros\Ouroboros.exe -ExpectedThumbprint <signerThumbprint from release-evidence.json>` (needs the Windows SDK `signtool`). The tests do not prove that the credential signs, that the runner accepts the timestamp, or which publisher a downloaded app shows; until a published release is checked and launched on Windows, live signing is unverified. Bundled runtimes and later Git code updates are not signed, and SmartScreen still weighs reputation.

### Release proof capsule

The artifact pipeline (per-platform archive smokes, native Linux packages, the AppImage custody chain, SBOM and attestation binding, the seven-required-desktop plus optional-Android release job, and `release-preflight`'s tag behaviour) lives in ARCHITECTURE §8 and `.github/workflows/ci.yml`. The honesty invariants a change must preserve:

- A failed test prerequisite makes `release-preflight` red but lets the desktop build run as a diagnostic rehearsal, which may consume configured macOS signing/notarization and records attestations in the repository and the public transparency log (configured Windows signing requires a green preflight, so a red run has no signed Windows ZIP); artifacts stay downloadable from the run, no GitHub Release is published, and `release` still requires a successful preflight.
- Publication is draft-first under a per-tag concurrency group; the remote annotated tag is revalidated against the event SHA before draft creation and again before publication, and a published release is never overwritten by a rerun.
- Vendor-distribution smokes (Astra, RED OS) are reported evidence, never release authority: third-party registry reachability is outside the publication pipeline's control.
- The AppImage smoke makes no native GTK/Qt claim; packaged native webview coverage is a separate Linux distribution contract.
- `OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1` is a local-builder escape hatch only: it skips Playwright's host-library installation, not browser-binary bundling, and a build using it must disclose that browser host compatibility was not locally proven.
- Never represent a later checksum inventory as build-time provenance, an SBOM, or packaged smoke evidence that the original build did not create.
