# Build & CI

This chapter maps dependency locks, opt-in pytest lanes, CI's parallel/serial split, its hermetic commit-gate mirror, and Actions secret gating. The gate must be reproducible and independent of the candidate.

### Python dependency locks

The one dependency authority and its packaging projections are ARCHITECTURE §8 "Build scripts" (`pyproject.toml` + `uv.lock`; `requirements-runtime.lock` and `requirements.txt` are generated projections, never authorities). A dependency change updates the metadata, runs `uv lock`, regenerates the runtime export with the exact README command, and leaves the CI clean-diff check green; the pinned `tool.uv.required-version` and digest-pinned `setup-uv` action make resolver changes deliberate rather than an ambient CI upgrade. Documentation may pair the checkout-free `uv tool install` form with a full commit SHA to pin the source revision, but must not claim it locks dependencies or call it a release-artifact install or a contributor development environment.

### Pytest marker lanes

Default local pytest excludes seven costly or environment-dependent lanes — `integration`, `browser`, `ui_browser`, `ui_browser_docker`, `portable_detail`, `skill_smoke` and `size_ratchet` — and CI opts into them explicitly (job topology and provider matrix: ARCHITECTURE §8 "CI topology"):

- `integration` runs real provider checks, including the trusted direct-OpenAI canary rows derived from `OPENAI_DIRECT_DEFAULTS`. Missing core credentials are red in the official repository job; quota/429/5xx/timeout may be typed inconclusive, while contract/auth/model/reasoning/tool 4xx stay red; on an official-repository release tag the job's last step (`tests/provider_release_floor.py`) also fails unless every required canary passed, and elsewhere it only reports. Secretless request-wire and Anthropic-custody contracts remain in ordinary pull-request tests: do not move provider secrets into PR jobs or duplicate the trusted lane. The KEYLESS `tests/system_e2e/` scenario lane rides the same marker plus `serial` and the `OUROBOROS_E2E_DEEP=mock` env gate (real isolated servers, no provider keys: ARCHITECTURE §8 "System E2E suite"); those three gates shut it out of every other pass, so only the dedicated `system-e2e-mock` job runs it — manual dispatch or release tag, never an ordinary push or pull request, carrying no secret.
- `browser` / `ui_browser` / `ui_browser_docker` launch real Playwright engines (agent browser tools / the host UI / the `ouroboros-web:test` container; the docker lane skips cleanly when Docker is unavailable locally). The marker is the source of truth for what the lane collects; the four Widgets lifecycle suites listed under "Declarative widgets" run in it. `portable_detail` covers build/portable artifact invariants. `ui-smoke` runs the complete existing `ui_browser` lane on non-documentation PRs and on EVERY push to `ouroboros`; manual/tag runs also retain Chromium/WebKit browser-tool coverage (triggers: ARCHITECTURE §8; required invocation and its guards: "Safe local verification" below). The one `ui_browser` module outside that lane is the owner-attachment module's Chromium cases on `full-test`'s Windows leg, which tolerates no skip. None of these joins the default local run or needs a paid provider.
- `skill_smoke` installs the nine pinned official skills from the LIVE catalog (list in `tests/test_skill_smoke_official.py`) as the dedicated 3-OS CI job, in serial pytest invocations with real network and real pip; red means investigate — there is deliberately no fallback-skip. Its paid review tier runs as a SEPARATE pytest step, ORDERED FIRST and ubuntu-only, alone carrying the provider key (why: ARCHITECTURE §8 "CI topology"); a missing key is a hard red, not a skip.
- `size_ratchet` carries the live-repo size gates and is the ONLY blocking surface for repository size (rules under "Module Size & Complexity"; base fallback and the fail-closed rule: ARCHITECTURE §8 "CI topology"); only checks against the live repo carry the marker.

`skill_smoke` and `size_ratchet` tests must NOT also carry the `serial` marker or join `_SERIAL_TEST_FILES`: the `and not <lane>` markexprs in quick/full-test are the lane barrier, and single-lane assignment keeps each test's placement unambiguous. A new opt-in lane registers its marker in `pyproject.toml`, adds a collect-only zero-test guard in CI, and keeps the default local addopts free of network and Docker requirements.

### Parallel CI and the `serial` marker

CI runs the default suite in parallel — `python -m pytest tests/` with `-m "not serial and <the seven lane exclusions>"`, `-n auto --dist loadscope --max-worker-restart=0 --timeout=300 --timeout-method=thread` — followed by a serial pass for `-m "serial and <the same exclusions>"` (`.github/workflows/ci.yml`, jobs `quick-test` / `full-test`). Two rules keep new tests from breaking that:

- Mark real-process / real-port tests, and tests that mutate process-global state WITHOUT reliable fixture isolation, `@pytest.mark.serial` (or add the file to `_SERIAL_TEST_FILES` in `tests/conftest.py`): under `-n` such a test flakes on kill/reap or port-reclaim timing or crashes its worker, and with `--max-worker-restart=0` a dead worker fails its WHOLE co-located batch as spurious failures in unrelated files.
- Keep every other test parallel-safe so it stays in the fast pass: `tmp_path` (never a fixed path), `monkeypatch.setenv`/`delenv`/`setattr` for environment and attribute changes, no execution-order assumptions. The autouse `tests/conftest.py::_os_environ_isolation` snapshot restores `os.environ` at every test boundary, but monkeypatch stays the rule because it reverses exactly the named change inside the test. A module-global mutation reliably snapshot-and-restored by a fixture may stay in the parallel pass (pattern: `tests/conftest.py::_isolate_workspace_executor_globals`).

### The local battery entry point

`python scripts/run_tests.py` (also `make test`) is the documented local run. A bare call always means the FULL battery — the node lane (a missing node is a red `NOT_RUN`), then every default-lane test — and never skips a test on the strength of an earlier run; extra arguments are forwarded to pytest as a focused run. It imports `LANE_EXCLUSION_EXPR` and `PARALLEL_PASS_FLAGS` from the gate rather than restating them. Its default mode is ONE xdist run under `--dist loadgroup --serial-shards=N` (`tests/conftest.py::_pin_lane_groups`): every serial FILE is pinned to one of N file-sharded groups, so a serial file never splits across workers and never runs beside another file of its own shard, while every other test is grouped by file; N is one per four workers, at most four, so a four-core machine keeps a single serial group. Serial tests thus also get the parallel flags' 300 s per-test timeout, which the gate's serial pass lacks. `--sequential` keeps the gate's marker split and xdist flags in two passes. The option defaults to 0 and is inert for CI and the commit gate, which keep the two-pass split and stay the authority; a failure seen only in overlapped mode is re-checked with `--sequential` before it is believed. Enforcement: `tests/test_run_tests_script.py`.

### Safe local verification and dirty browser candidates

Before application imports or local tests, use the stdlib boundary launcher. It
checks the helper's imports, scrubs owner configuration and credentials, and
prints isolated roots before Python starts. Use a dependency-only venv: an
editable project could import deleted modules from the source checkout:

```bash
uv sync --locked --extra browser --group dev --no-install-project
python -I -S scripts/safe_test.py -- .venv/bin/python -m pytest tests/test_test_environment.py
export PLAYWRIGHT_BROWSERS_PATH="$(pwd)/.tmp-data-browsers"
python -I -S scripts/safe_test.py -- .venv/bin/python -m playwright install chromium webkit
OUROBOROS_RUN_UI_SMOKE=1 OUROBOROS_EXPECT_BROWSER_ENGINES=chromium,webkit \
  python -I -S scripts/safe_test.py -- .venv/bin/python -m pytest tests/ -m ui_browser --require-ui-browser
```

`--require-ui-browser` fails on narrowed/empty collection, missing engines,
unsettled cases, or skips outside `tests/browser_lane.py`'s reviewed platform
registry. Registered skips name node/reason. `--temp-parent` (`/tmp` on macOS: short socket paths) refuses Git checkouts. Launcher and pytest trees persist (`SAFE_TEST_RETAINED <path>`). The stdlib
`ouroboros/test_environment.py` owns the data, settings, app, HOME, projects,
worktrees, Deliverables, cache and userbase defaults for pytest, preflight and
their server children — including `env=None` children; explicit test roots survive. Bare pytest keeps explicitly supplied
provider/lane controls for integration CI; launcher and preflight scrub them.
Run identities never pass: bare pytest drops launcher authority and the
isolated review's run cap and engine attach selection, so no test reaches a
host engine.
Under that marker `supervisor/git_ops_reset.py` installs no dependencies for ANY
caller; production unchanged. C locale/Git ceilings stabilize probes; pytest Deliverables follows synthetic HOME. `MAC_CHROMIUM_TMPDIR` shares the disposable temp root: macOS Chromium ignores `TMPDIR` for initial download staging. Not an OS sandbox. Money scale: [usage store](../USAGE_STORE.md#4-reads).

`tests/candidate_checkout.py` owns shared UI and keyless wait/repair checkouts: tracked
worktree bytes plus non-ignored new files — staged, unstaged, deleted, executable
and binary alike. It preserves the source HEAD, branch and raw index, verifies
source identity before and after capture and at teardown, and verifies the copy
before each server incarnation and at exit, kept unless all reaped. The copy's
identity binds HEAD, staged entries and their diff (including intent-to-add), and every file's bytes and mode — not the
raw index bytes: a served process runs `git status`, which rewrites the index's
cached stat data without moving one staged entry, and the error names the
metadata field or paths that did move. Ignored artifacts and empty
directories are outside the contract; symlinks, special files, gitlinks,
sparse/assume-unchanged, split and unmerged indexes fail explicitly. Two complete
observations plus per-read metadata checks detect concurrent edits — no lock, no
adversarial-snapshot promise — so keep the source stable for the whole run.
`origin_proof=True` adds bytes no other tree carries: a static sentinel under
`web/` and a `+candidate.<token>` VERSION suffix. Startup AND restart then prove
the bound PID, that the served static tree is this copy, and that `/api/health`'s
`runtime_version` came from Python imported from it, so a server answering with
HEAD's bytes or another checkout's cannot pass. An interpreter carrying an
installed `ouroboros` distribution is refused explicitly. Owners:
`test_candidate_checkout`, `test_candidate_checkout_consumers`, `test_test_environment`, `test_ui_fixture_lifecycle`,
`test_ui_candidate_server`, `test_browser_ci_scope`.

The real-engine custody witness is `python -I -S scripts/safe_test.py --temp-parent /tmp -- <python> scripts/claudexor_lifecycle_smoke.py`. It retains its disclosed temporary roots and always stops its own daemon children, even on a failed assertion. Its fake-only catalog projection is an explicit test exception: the real engine catalog excludes fake harnesses. All delegated admission and custody effects remain production paths; no credentials or vendor task are needed. CI runs it in `claudexor-platform-gate.yml` on all three desktop OSes, separately from the gateway-only fixture lane.

### Reading CI failure evidence

Provider/UI summaries report Actions producer outcomes separately from testcase counts: a passing case cannot override a nonzero session exit. Skips and unavailable credentials remain explicit. Upload/report failures say `diagnostics_incomplete` and do not alter release eligibility, with one exception: the sharded UI lane is proven only by its shards' and its manifest's projections, so a missing, unreadable or red projection there fails `ui-smoke`. `tests.ci_evidence`, registered after isolation, writes safe projections with `--ci-evidence-dir`; raw JUnit stays outside uploads in `runner.temp/ci-private`.

`quick-test` and `full-test` publish that projection for their parallel, serial and size-ratchet passes through `.github/actions/test-evidence`: one summary section per pass, and at most ten error annotations naming failed tests (the summary lists the rest). A dead xdist worker — a hard crash, or the parallel pass's thread-method timeout — is phase `crash`. The controller appends a redacted `events.jsonl` as tests start, report and finish, so a session killed before its final export still names the tests in flight; collected tests that never reported are counted. Each producer step has a wall-clock ceiling whose expiry fails the step, and with it the job, without cancelling the job, so the evidence step still runs after a hang; the serial pass keeps no per-test timeout.

The complete UI lane (`ui-browser.yml`) runs as four `ui-shard` jobs. Each collects and guards the whole lane, then runs every fourth node of the sorted node ids (`--ui-browser-shard=K/4`) under a cooperative 75-minute session budget (`--session-timeout=4500`, no per-test timeout) and a 90-minute step ceiling. `ui-manifest` collects the lane once without the shard option. `ui-smoke` runs `tests/ci_evidence.py reconcile-shards` over the run's artifacts and is red, naming each gap, unless every manifest node executed in its own shard and every judged session (shards, manifest, browser tools) ended with exit status 0; after a re-run the proof is the highest attempt and the summary says which. It is also red when a shard or the manifest job did not succeed, and green without reconciliation only for a documentation-only event.

Provider failures retain every attempt, including earlier retries, with labelled logical, persisted request, received tool-field, partial-assembly and normalized views. Successes are compact. Omitted/redacted views are explicit; original nonstream bodies may be unavailable, and partial assembly cannot exonerate the assembler. Private native data and exception bodies are excluded.

The synthetic viewport, inflight and Main-notice fixtures capture screenshot, trace and event/geometry facts before close on failure; capture/cleanup errors preserve the first exception. Dispatch `ui-browser-push.yml` with `viewport` or `inflight` for Linux PARTIAL DIAGNOSTIC checks without paid calls or browser-tool tests. `full` retains the collection guard and existing assertions/gestures/waits; partial success is never full proof. The accounting test checks causal loop progress under lock contention, not a wall-time tick rate.

### The commit gate mirrors the CI split

`ouroboros/preflight_runner.py::run_hermetic_pytest` mirrors CI in one disposable checkout and scrubbed temporary data root: the node test lane (`cd web && node --test tests/*.test.js`, content-keyed — a candidate without web tests never requires node, while an active web suite cannot silently disappear when node is missing), then the same two logical pytest passes (parallel `not serial`, then flag-free `serial`). The browser no-undef check has two layers: the dependency-free acorn walker in that suite (`web/tests/no_undef.test.js`) is the hermetic gate's, and both CI jobs additionally run ESLint's `no-undef` (`web/eslint.config.js`, exact-pinned, installed with `npm ci`) as an independent second opinion — CI-only, never part of the gate. `LANE_EXCLUSION_EXPR` and `PARALLEL_PASS_FLAGS` are executable SSOTs pinned against both CI jobs; the candidate is captured as one hardened worktree-vs-`HEAD` binary diff, and a capture or apply failure is the typed `PREFLIGHT_CANDIDATE_ASSEMBLY` hard block, never a test failure. The `pyproject.toml` `addopts` line is the single home of the per-test timing report (`--durations=25 --durations-min=1.0`), prepended to every argv, so the same slowest-test evidence appears locally, in both CI jobs and in both gate passes. Contributor rules:

- The candidate cannot weaken the pass: `PYTEST_*`/`NODE_OPTIONS` are scrubbed, and so is owner runtime state — `OUROBOROS_*`, secret-suffixed keys and every settings key `config.apply_settings_to_env` projects (the settings vocabulary read statically by `test_environment.settings_keys`) — so the verdict cannot depend on the operator's install profile; required plugins are probed outside candidate control and forced on with host-owned worker evidence, post-commit checks also inspect `HEAD~1` so suite deletion cannot hide after the commit exists, and exit status owns the verdict — rendered diagnostics do not. `OUROBOROS_PREFLIGHT_SERIAL=1` is the explicit temporary rollback lever, never a silent fallback.
- A red post-commit gate is warning-only for an ordinary commit (the local commit is preserved for forensics); evolution publication refuses to auto-push while the warning stands, and inside a managed update the gate blocks boot promotion and routes through rollback — an incomplete rollback leaves `gate_blocked` so boot retries recovery instead of promoting the rejected merge.
- The managed mandate is "the full suite provably ran green on the exact committed tree", not "run it twice": the reuse authority is the process-held runner proof (`ctx._preflight_test_proof`), never the durable `tests_evidence` record or the event log (what it binds and why every workload binds HEAD: ARCHITECTURE §6 "Git and commit review"). A newly created commit requires a new run, a restart forces a rerun, and a skip, no applicable suite or a mocked `None` return cannot mint a proof. Review-binding and tag-binding mismatches use the same managed failure route.
- Process containment is unconditional, including after a green pass: Windows uses a kill-on-close Job Object; POSIX uses an environment membership token plus a process-group enumeration backstop and promises honest detection with a fail-closed verdict for attributed members, not guaranteed teardown of an arbitrary detached process. A same-uid unreadable stranger is a warning, never membership proof; an unobserved descendant that detached and hid its token remains a disclosed detection gap; known roots, groups and the retained set of observed members still fail closed when unreadable (`tests/test_preflight_process_containment.py`). A crashed worker, a timeout-killed worker, a missing plugin, containment failure and ordinary test failure keep distinct diagnostics.
- Mark process/port/global-state tests `serial`; make a merely slow test faster or split it — marking it serial removes the 300s per-test timeout and lets it consume the remaining total gate budget.

### GitHub Actions: secrets in step-level `if:` conditions

GitHub Actions rejects `secrets.*` inside step-level `if:` expressions, and a step's own `env:` block is not visible to that same step's `if:`. Derive a non-secret boolean in the job-level `env:` block, gate steps with that boolean, and map the actual credentials only inside the first-party steps that need them, so later SBOM and attestation steps inherit none of them:

```yaml
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
    env:
      HAS_APPLE_SIGNING: ${{ matrix.os == 'macos-latest' && secrets.BUILD_CERTIFICATE_BASE64 != '' && secrets.P12_PASSWORD != '' && secrets.KEYCHAIN_PASSWORD != '' && secrets.APPLE_TEAM_ID != '' && 'true' || 'false' }}
    steps:
      - name: Import Apple signing certificate
        if: env.HAS_APPLE_SIGNING == 'true'
        env:
          BUILD_CERTIFICATE_BASE64: ${{ secrets.BUILD_CERTIFICATE_BASE64 }}
          P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
          KEYCHAIN_PASSWORD: ${{ secrets.KEYCHAIN_PASSWORD }}
        run: |
          echo "${BUILD_CERTIFICATE_BASE64}" | base64 -d > cert.p12
          security create-keychain -p "${KEYCHAIN_PASSWORD}" build.keychain
          security import cert.p12 -k build.keychain -P "${P12_PASSWORD}"
      - name: Cleanup keychain
        if: always() && matrix.os == 'macos-latest' && env.HAS_APPLE_SIGNING == 'true'
        run: security delete-keychain build.keychain
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

Prerelease artifacts may intentionally be unsigned and must report that state; stable publication applies the configured signing and notarization policy rather than implying credentials or success that were absent. Only the non-secret `HAS_APPLE_SIGNING` gate is job-wide; certificate/keychain values exist only in the import step and Apple ID notarization values only in the first-party build step. Notary/stapler failures are soft outcomes recorded through `NOTARIZE_OUTCOME`, so a transient Apple service problem does not silently drop an otherwise valid signed artifact; cleanup uses the `always()` plus matrix/env guards, and signing material never persists across runs.

### Windows Authenticode release signing (SSL.com eSigner)

The repository variable `ESIGNER_CERT_SHA1` selects the mode for every `v*`
tag, prereleases included. **Unset** (forks, any repository without a
certificate): no job binds a signing Environment or sees a signing secret, and
the release publishes an explicitly unsigned Windows ZIP whose receipt and
notes say so. **Set**: signing is required. A malformed value, a missing
secret, a tool, signature, timestamp or digest failure stops the whole
seven-asset release; it never falls back to an unsigned ZIP.

The `build` matrix holds no eSigner secret. Its Windows shard always packs the
unsigned ZIP with `scripts/pack_windows_archive.ps1`, the one packer for local
and both release modes (ZipFile keeps hidden files such as `.local-browsers`;
the ZIP is re-extracted and must match every directory and file digest), and
hands it over with its digest as a job output. When configured, a separate
`windows-sign` job on a fresh runner binds the `windows-release-signing`
Environment, checks that digest before extracting, and runs
`scripts/sign_windows_release.ps1`: SSL.com's CodeSignTool 1.3.2 from its
versioned GitHub release asset, SHA-256
`4afc32e8b7f79bbe1de7e4e7049aaad4e0f754357613b9bbec0e3052f06fd36b`, run with
the runner's maintained Java 11 (`JAVA_HOME_11_X64`; the bundled 2019 JDK
lacks the root of `cs.ssl.com`'s current chain) through `java -jar` (the
vendor BAT expands `%*` unquoted). It signs `Ouroboros.exe`, checks
Authenticode status, the configured signer and `signtool verify /pa /tw /v` by
documented exit code (0 passes; 1 fails; 2 warns, e.g. no timestamp), repacks
and outputs the final ZIP digest. Vendor output is discarded, never printed,
redacted or uploaded; each stage fails with its own message and says when the
cause is not visible. The secret-free `windows-proof` runner checks the
producer's digest before extracting, re-verifies the signature when
configured, runs the packaged smoke, and records the signer thumbprint and
publisher (or `unsigned`). Only signed receipts carry `authenticode_signer`,
`timestamp` and `signed_payload_archive_match` checks. Both modes attest the
ZIP and SBOM. Assembly requires a receipt matching `ESIGNER_CERT_SHA1`
and downloads only `ouroboros-*` artifacts; intermediate uploads overwrite on a
re-run.

Setup, by role: the **repository owner** creates the Environment
`windows-release-signing` with a `v*` tag rule and no required reviewer, so a
tag signs without a manual approval. The **certificate holder** stores the
four **environment-scoped** secrets `ESIGNER_USERNAME`, `ESIGNER_PASSWORD`,
`ESIGNER_CREDENTIAL_ID` (the order's *SIGNING CREDENTIALS* section) and
`ESIGNER_TOTP_SECRET` (the `secret code` beside the order's *eSigner.com QR
Code*) with `gh secret set <NAME> --env windows-release-signing`, then sets
`ESIGNER_CERT_SHA1`, a **nonsecret repository-level Actions variable** holding
the certificate's 40-hex SHA-1 thumbprint; setting it turns signing on.
Repository secrets would be readable from any branch push. Never paste values
into an issue, chat, workflow or log; an exposed TOTP seed is regenerated at
SSL.com. Each `windows-sign` run spends one eSigner signing, a re-run of that
job another. The plan's signing count is not a hard stop: SSL.com's service
agreement invoices signings beyond the subscription tier, and this repository
cannot see the holder's balance or order terms.

A green tag run publishes directly. Anyone can check a downloaded ZIP with
`scripts/verify_windows_signature.ps1 -Executable Ouroboros\Ouroboros.exe
-ExpectedThumbprint <signerThumbprint from release-evidence.json>` (Windows
SDK `signtool` required). Until a published release has been checked and
launched on Windows, live signing is **unverified**: these tests prove neither
that the credential signs, nor that the runner accepts the timestamp, nor the
publisher a downloaded app shows. Only `Ouroboros.exe` is signed, not the
bundled runtimes or later Git updates of the code, and SmartScreen still
weighs reputation.

### Release proof capsule

The artifact pipeline — per-platform archive smokes, native Linux packages, the AppImage custody chain, SBOM and attestation binding, and the seven-required-desktop plus optional-Android release job — lives in ARCHITECTURE §8 and `.github/workflows/ci.yml`. The honesty invariants a change must preserve:

- On a valid release tag, `release-preflight` records the tag/VERSION and prerelease state before checking its required job results. A failed test prerequisite makes the preflight red but permits the desktop build to run as a diagnostic rehearsal; macOS signing/notarization can still consume configured services and record attestations, but configured Windows signing requires a green preflight, so a diagnostic red run has no signed ZIP; no GitHub Release is published; the release job still requires a successful preflight. Android publisher builds retain their Android proof dependencies because the signed source/APK pair is optional release content, so failed or skipped Android proofs exclude both optional assets; the diagnostic build itself never invokes the release job; if the required tests later pass on a rerun, that same-tag payload may be published with provenance bound to the same SHA.
- Publication is draft-first with a per-tag concurrency group; the remote
  annotated tag is revalidated against the event SHA immediately before
  draft creation AND again before publication, and a published release is
  never overwritten by a rerun.
- Vendor-distribution smokes (Astra, RED OS) are reported evidence, never
  release authority — third-party registry reachability is outside the
  publication pipeline's control.
- The AppImage smoke deliberately makes no native GTK/Qt claim; packaged
  native webview coverage remains a separate Linux distribution contract.
- `OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1` is only a local-builder escape
  hatch — it skips Playwright's host-library installation, not
  browser-binary bundling — and a build using it must disclose that browser
  host compatibility was not locally proven.
- Never represent a later checksum inventory as build-time provenance, an
  SBOM, or packaged smoke evidence that the original build did not create.
