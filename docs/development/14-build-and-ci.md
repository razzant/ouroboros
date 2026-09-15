# Build & CI

This chapter owns the build and test topology: the dependency-lock change procedure, the seven opt-in pytest marker lanes, the parallel and serial split CI actually runs, the hermetic commit gate that mirrors that split in a disposable checkout, and the GitHub Actions secret-gating shape. It exists because the gate's verdict has to be reproducible and un-weakenable by the candidate it is judging.

### Python dependency locks

The one dependency authority and its packaging projections are ARCHITECTURE
"Build scripts" (`pyproject.toml` + `uv.lock`; `requirements-runtime.lock` and
`requirements.txt` are generated projections, never authorities). A dependency
change updates the metadata, runs `uv lock`, regenerates the
runtime export with the exact README command, and leaves the CI clean-diff
check green. The pinned `tool.uv.required-version` and digest-pinned
`setup-uv` action make resolver changes deliberate rather than an ambient
CI upgrade. Documentation may pair the checkout-free `uv tool install` form
with a full commit SHA to pin the source revision, but must not claim it
locks dependencies or describe it as a release-artifact install or
contributor development environment.

### Pytest marker lanes

Default local pytest excludes seven costly or environment-dependent lanes —
`integration`, `browser`, `ui_browser`, `ui_browser_docker`,
`portable_detail`, `skill_smoke`, and `size_ratchet` — and CI opts into
them explicitly (job topology and provider matrix: ARCHITECTURE "CI
topology"):

- `integration` runs real provider checks, including the trusted
  direct-OpenAI canary rows derived from `OPENAI_DIRECT_DEFAULTS`. Missing
  core credentials are red in the official repository job; explicit
  quota/429/5xx/timeout may be typed inconclusive, while
  contract/auth/model/reasoning/tool 4xx stay red. Secretless request-wire
  and Anthropic-custody contracts remain in ordinary pull-request tests: do
  not move provider secrets into PR jobs or duplicate the trusted lane. The
  KEYLESS `tests/system_e2e/` scenario lane rides the same marker plus `serial`
  and the `OUROBOROS_E2E_DEEP=mock` env gate — real isolated servers, no
  provider keys (ARCHITECTURE "System E2E suite"). Because those three gates
  shut it out of every other pass, the only thing that runs it is the dedicated
  `system-e2e-mock` CI job on its daily schedule, manual dispatch or release
  tag: never an ordinary branch push or pull_request, and carrying no secret.
- `browser` / `ui_browser` / `ui_browser_docker` launch real Playwright
  engines (agent browser tools / the host UI / the `ouroboros-web:test`
  container; the docker lane skips cleanly when Docker is unavailable
  locally). The marker is the source of truth for what the lane collects;
  the four Widgets lifecycle suites listed under "Declarative widgets" run
  in it. `portable_detail` covers build/portable artifact invariants.
  Existing `ui-smoke` runs only `tests/test_skill_publish_browser.py` on PRs,
  using Chromium and real task admission, Main card and history. Manual/tag
  runs keep the full host UI and Chromium/WebKit browser-tool suites. These
  checks do not join the default local pytest run or require a paid provider.
- `skill_smoke` installs the nine pinned official skills from the LIVE
  catalog (list in `tests/test_skill_smoke_official.py`) and runs as the
  dedicated 3-OS CI job in serial pytest invocations with real network and
  real pip; red means investigate — there is deliberately no fallback-skip.
  Its paid review tier runs as a SEPARATE pytest step, ORDERED FIRST and
  ubuntu-only, alone carrying the provider key (why: ARCHITECTURE "CI
  topology"); a missing key is a hard red, not a skip.
- `size_ratchet` carries the live-repo size gates and is the ONLY blocking
  surface for repository size (rules under "Module Size & Complexity"; the
  base fallback and why a resolvable base without a manifest fails closed:
  ARCHITECTURE "CI topology"; only checks against the live repo carry the
  marker).

`skill_smoke` and `size_ratchet` tests must NOT also carry the `serial`
marker or join `_SERIAL_TEST_FILES`: the `and not <lane>` markexprs in
quick/full-test are the lane barrier, and single-lane assignment keeps each
test's placement unambiguous. When adding a new opt-in lane, register the
marker in `pyproject.toml`, add a collect-only zero-test guard in CI, and
keep the default local addopts free of network and Docker requirements.

### Parallel CI and the `serial` marker

CI runs the default suite in parallel — `python -m pytest tests/` with
`-m "not serial and <the seven lane exclusions>"`, `-n auto --dist
loadscope --max-worker-restart=0 --timeout=300 --timeout-method=thread` —
followed by a serial pass for `-m "serial and <the same exclusions>"`
(`.github/workflows/ci.yml`, jobs `quick-test` / `full-test`). Two rules
keep new tests from breaking that:

- Mark real-process / real-port tests, and tests that mutate process-global
  state WITHOUT reliable fixture isolation, `@pytest.mark.serial` (or add
  the file to `_SERIAL_TEST_FILES` in `tests/conftest.py`). Under `-n` such
  a test flakes on kill/reap or port-reclaim timing, or crashes its
  worker — and with `--max-worker-restart=0` a dead worker fails its WHOLE
  co-located batch, showing up as spurious failures in unrelated files.
- Keep every other test parallel-safe so it stays in the fast pass: use
  `tmp_path` (never a fixed path), `monkeypatch.setenv`/`delenv`/`setattr` for
  environment and attribute changes, and no execution-order assumptions. The
  autouse `tests/conftest.py::_os_environ_isolation` snapshot restores
  `os.environ` at every test boundary, so a bare assignment no longer leaks;
  monkeypatch stays the rule because it reverses exactly the named change
  inside the test, before the snapshot runs. A
  module-global mutation that is reliably snapshot-and-restored by a
  fixture may stay in the parallel pass — the pattern is
  `tests/conftest.py::_isolate_workspace_executor_globals`.

### The commit gate mirrors the CI split

`ouroboros/preflight_runner.py::run_hermetic_pytest` mirrors CI in one
disposable checkout and scrubbed temporary data root: the node test lane
(`cd web && node --test tests/*.test.js`, content-keyed — a candidate
without web tests never requires node, while an active web suite cannot
silently disappear when node is missing), then the same two logical pytest
passes (parallel `not serial`, then flag-free `serial`).
The browser no-undef check has two layers: the dependency-free acorn walker
in that suite (`web/tests/no_undef.test.js`) is the hermetic gate's, and
both CI jobs additionally run ESLint's `no-undef` (`web/eslint.config.js`,
exact-pinned, installed with `npm ci` from `web/package-lock.json`) as an
independent second opinion — CI-only, never part of the gate.
`LANE_EXCLUSION_EXPR` and `PARALLEL_PASS_FLAGS` are executable SSOTs pinned
against both CI jobs; the candidate is captured as one hardened
worktree-vs-`HEAD` binary diff, and a capture or apply failure is the typed
`PREFLIGHT_CANDIDATE_ASSEMBLY` hard block, never a test failure.
The `pyproject.toml` `addopts` line is the single home of the per-test timing
report (`--durations=25 --durations-min=1.0`): it is prepended to every argv, so
the same slowest-test evidence appears in a plain local run, in both CI jobs and
in both gate passes without any surface pinning its own copy.
Contributor rules:

- The candidate cannot weaken the pass: `PYTEST_*`/`NODE_OPTIONS` are
  scrubbed, and so is owner runtime state — `OUROBOROS_*`, secret-suffixed
  keys and every settings key `config.apply_settings_to_env` projects
  (derived from `settings_env_keys()`), so the verdict cannot depend on the
  operator's install profile; required plugins are probed outside candidate control and
  forced on with host-owned worker evidence, post-commit checks also
  inspect `HEAD~1` so suite deletion cannot hide after the commit exists,
  and exit status owns the verdict — rendered diagnostics do not.
  `OUROBOROS_PREFLIGHT_SERIAL=1` is the explicit temporary rollback lever,
  never a silent fallback.
- A red post-commit gate is warning-only for an ordinary commit (the local
  commit is preserved for forensics); evolution publication refuses to
  auto-push while the warning stands, and inside a managed update the gate
  blocks boot promotion and routes through rollback — an incomplete
  rollback leaves `gate_blocked` so boot retries recovery instead of
  promoting the rejected merge.
- The managed mandate is "the full suite provably ran green on the exact
  committed tree", not "run it twice": the reuse authority is the process-held
  runner proof (`ctx._preflight_test_proof`), never the durable
  `tests_evidence` record or the event log (what the proof binds, when it is
  reused and why every workload binds HEAD: ARCHITECTURE "Git and commit
  review"). A newly created commit therefore requires a new run, a restart
  forces a rerun, and a skip, no applicable suite or a mocked `None` return
  cannot mint a proof. Review-binding and tag-binding mismatches use the same
  managed failure route.
- Process containment is unconditional, including after a green pass:
  Windows uses a kill-on-close Job Object; POSIX uses an environment
  membership token plus a process-group enumeration backstop and promises
  honest detection with a fail-closed verdict for attributed members, not
  guaranteed teardown of an arbitrary detached process. A same-uid unreadable
  stranger is a warning, never membership proof; an unobserved descendant
  that detached and hid its token remains a disclosed detection gap. Known
  roots, groups and the retained set of observed members still fail closed when unreadable
  (`tests/test_preflight_process_containment.py`). A crashed worker, a timeout-killed
  worker, a missing plugin, containment failure, and ordinary test failure
  keep distinct diagnostics.
- Mark process/port/global-state tests `serial`; make a merely slow test
  faster or split it — marking it serial removes the 300s per-test timeout
  and lets it consume the remaining total gate budget.

### GitHub Actions: secrets in step-level `if:` conditions

GitHub Actions rejects `secrets.*` inside step-level `if:` expressions, and
a step's own `env:` block is not visible to that same step's `if:`. Derive
a non-secret boolean in the job-level `env:` block, gate steps with that
boolean, and map the actual credentials only inside the first-party steps
that need them — later SBOM and attestation steps then inherit none of
them.

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

`tests/test_build_scripts.py::TestMacOSSigning::test_ci_uses_env_context_for_condition`
enforces this for `.github/workflows/ci.yml` only; other workflow files are
not scanned by it.

### Apple signing & notarization (macOS Build job)

Prerelease artifacts may intentionally be unsigned and must report that
state; stable publication applies the configured signing and notarization
policy rather than implying credentials or success that were absent. Only
the non-secret `HAS_APPLE_SIGNING` gate is job-wide; certificate/keychain
values exist only in the import step and Apple ID notarization values only
in the first-party build step. Notary/stapler failures are soft outcomes
recorded through `NOTARIZE_OUTCOME`, so a transient Apple service problem
does not silently drop an otherwise valid signed artifact; cleanup uses the
`always()` plus matrix/env guards, and signing material never persists
across runs.

### Release proof capsule

The artifact pipeline — per-platform archive smokes, native Linux packages,
the AppImage custody chain, SBOM and attestation binding, and the
seven-asset release job — lives in ARCHITECTURE "8. Git Branching, CI, and
Build" and `.github/workflows/ci.yml`. The honesty invariants a change must
preserve:

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
