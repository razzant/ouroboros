"""Node lane of the hermetic commit gate (ouroboros/preflight_node.py).

The lane's whole contract, one test each: content-keyed activation (an empty
candidate glob means NO node requirement at all), typed hard blocks for a
missing/unusable/too-old runtime, a red suite as `NODE_TESTS_FAILED` with the
failing test named, and a green suite that lets the pytest passes proceed.

Real-spawn tests skip on a host without a usable node — but CI provisions node
22 and sets `OUROBOROS_PREFLIGHT_REQUIRE_NODE=1`, which turns that skip into a
hard failure so a provisioning regression can never look like a green run
(the exact self-concealment `OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS` closes for
the pytest plugins).
"""

import json
import os
import pathlib
import re
import subprocess
import textwrap

import pytest

from ouroboros import preflight_node as pn

# Spawns real node/git subprocesses; run_hermetic_pytest's reaper sweeps
# processes referencing its temp root. Same class as test_preflight_runner.py.
pytestmark = pytest.mark.serial

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_REQUIRE_NODE_ENV = "OUROBOROS_PREFLIGHT_REQUIRE_NODE"


def _node_problem() -> str:
    """Whether THIS host can run the lane's real spawn, per the lane's own resolver."""
    node = pn.resolve_node()
    if not node:
        return "no Node.js runtime resolved (bundled or PATH)"
    version = pn.probe_node_version(node)
    if not version:
        return f"{node} did not answer --version"
    if pn._version_tuple(version) < pn.NODE_MIN_VERSION:
        floor = ".".join(str(part) for part in pn.NODE_MIN_VERSION)
        return f"{node} is v{version}, below the v{floor} floor"
    return ""


_NODE_PROBLEM = _node_problem()

_REAL_NODE_SKIP_REASON = (
    "this host cannot run the node lane's real spawn: " + (_NODE_PROBLEM or "-")
    + " — run `bash scripts/download_node_standalone.sh` or install node >= "
    + ".".join(str(part) for part in pn.NODE_MIN_VERSION)
    + f", or set {_REQUIRE_NODE_ENV}=1 to turn this skip into a hard failure"
)

requires_node = pytest.mark.skipif(bool(_NODE_PROBLEM), reason=_REAL_NODE_SKIP_REASON)

_PASSING_TEST = textwrap.dedent(
    """\
    import test from 'node:test';
    import assert from 'node:assert/strict';
    test('passes', () => { assert.equal(1, 1); });
    """
)

_FAILING_TEST = textwrap.dedent(
    """\
    import test from 'node:test';
    import assert from 'node:assert/strict';
    test('boom_marker_test', () => { assert.equal(1, 2); });
    """
)


def _worktree(tmp_path: pathlib.Path, files: dict) -> pathlib.Path:
    """A bare candidate tree; web/package.json mirrors production ESM shape."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    if any(rel.startswith("web/") for rel in files):
        package = worktree / "web" / "package.json"
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_text(json.dumps({"type": "module"}), encoding="utf-8")
    for rel, body in files.items():
        target = worktree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return worktree


# ── Lane activation (content-keyed) ───────────────────────────────────


def test_empty_glob_keeps_the_lane_inactive_without_touching_node(tmp_path, monkeypatch):
    """No web/tests/*.test.js in the candidate → None, and node is never even
    resolved — the gate's synthetic mini-repo fixtures must keep passing on
    hosts with no node at all."""

    def _explode():
        raise AssertionError("resolve_node was called for an inactive lane")

    monkeypatch.setattr(pn, "resolve_node", _explode)

    bare = _worktree(tmp_path, {"tests/test_ok.py": "def test_ok():\n    assert True\n"})
    assert pn.run_node_tests(bare, tmp_path / "t1", 60, 8000) is None

    # An empty web/tests directory — or one holding only non-test files — is
    # inactive too: the glob is `tests/*.test.js`, not "the directory exists".
    (bare / "web" / "tests").mkdir(parents=True)
    (bare / "web" / "tests" / "helper.js").write_text("export {};\n", encoding="utf-8")
    assert pn.run_node_tests(bare, tmp_path / "t2", 60, 8000) is None


def test_candidate_glob_is_sorted_and_web_relative(tmp_path):
    worktree = _worktree(tmp_path, {
        "web/tests/zz.test.js": _PASSING_TEST,
        "web/tests/aa.test.js": _PASSING_TEST,
        "web/tests/not_a_test.js": "export {};\n",
    })
    assert pn.candidate_node_tests(worktree) == ["tests/aa.test.js", "tests/zz.test.js"]


# ── Typed runtime blocks ──────────────────────────────────────────────


def test_missing_node_is_a_typed_hard_block_when_the_lane_is_active(tmp_path, monkeypatch):
    worktree = _worktree(tmp_path, {"web/tests/x.test.js": _FAILING_TEST})
    monkeypatch.setattr(pn, "resolve_node", lambda: None)

    result = pn.run_node_tests(worktree, tmp_path / "t", 60, 8000)

    assert result is not None and result["files"] == 1
    assert "PREFLIGHT_NODE_MISSING" in result["error"]
    # The remediation must name the provisioning script that actually exists.
    assert "scripts/download_node_standalone.sh" in result["error"]
    assert (REPO_ROOT / "scripts" / "download_node_standalone.sh").is_file()


def test_unprobeable_node_is_a_typed_hard_block(tmp_path, monkeypatch):
    """A node that exists but cannot answer `--version` (the Homebrew-SIGKILL
    class) blocks with the MISSING label rather than being run blind."""
    worktree = _worktree(tmp_path, {"web/tests/x.test.js": _PASSING_TEST})
    monkeypatch.setattr(pn, "resolve_node", lambda: "/fake/node")
    monkeypatch.setattr(pn, "probe_node_version", lambda path: "")

    result = pn.run_node_tests(worktree, tmp_path / "t", 60, 8000)

    assert "PREFLIGHT_NODE_MISSING" in result["error"]
    assert "/fake/node" in result["error"]


def test_too_old_node_is_a_typed_hard_block(tmp_path, monkeypatch):
    worktree = _worktree(tmp_path, {"web/tests/x.test.js": _PASSING_TEST})
    monkeypatch.setattr(pn, "resolve_node", lambda: "/fake/node")
    monkeypatch.setattr(pn, "probe_node_version", lambda path: "18.19.1")

    result = pn.run_node_tests(worktree, tmp_path / "t", 60, 8000)

    assert "PREFLIGHT_NODE_TOO_OLD" in result["error"]
    assert "18.19.1" in result["error"]


@pytest.mark.parametrize("version,meets_floor", [
    ("20.11.0", True),
    ("20.11", True),
    ("20.10.9", False),
    ("18.19.1", False),
    ("22.0.0", True),
    ("24.16.0", True),
    ("v20.11.0", True),      # probe strips the v, but the parser tolerates it
    ("21.0.0-nightly", True),  # leading digits only, per piece
    ("garbage", False),        # unparseable degrades to 0 → blocked, not crashed
])
def test_version_floor_comparison(version, meets_floor):
    assert (pn._version_tuple(version) >= pn.NODE_MIN_VERSION) is meets_floor


# ── Real spawn ────────────────────────────────────────────────────────


def test_node_is_provisioned_where_required():
    """Control mirroring the plugins lane: with the CI flag set, a missing node
    is one loud failure instead of a run full of silent skips."""
    if _NODE_PROBLEM and not os.environ.get(_REQUIRE_NODE_ENV, "").strip():
        pytest.skip(
            "unprovisioned host, and this run did not declare "
            f"{_REQUIRE_NODE_ENV}=1 — " + _REAL_NODE_SKIP_REASON
        )
    assert _NODE_PROBLEM == "", (
        f"{_REQUIRE_NODE_ENV} declares this host provisioned, but the node lane "
        "cannot run a real spawn: " + _NODE_PROBLEM
    )


@requires_node
def test_a_failing_web_test_is_node_tests_failed(tmp_path):
    worktree = _worktree(tmp_path, {
        "web/tests/boom.test.js": _FAILING_TEST,
        "web/tests/ok.test.js": _PASSING_TEST,
    })

    result = pn.run_node_tests(worktree, tmp_path / "t", 120, 8000)

    assert result["files"] == 2
    assert result["returncode"] not in (0, None)
    assert "NODE_TESTS_FAILED" in result["error"]
    # The bounded body must still NAME the failing test.
    assert "boom_marker_test" in result["error"]


@requires_node
def test_a_passing_web_suite_is_green(tmp_path):
    worktree = _worktree(tmp_path, {"web/tests/ok.test.js": _PASSING_TEST})

    result = pn.run_node_tests(worktree, tmp_path / "t", 120, 8000)

    assert result["error"] is None
    assert result["returncode"] == 0
    assert result["files"] == 1
    assert result["node"]


# ── Orchestration: the gate runs the lane on the CANDIDATE tree ───────


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)


def _gate_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Tiny committed repo the hermetic gate accepts (pytest suite, no web tests)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "ouroboros")
    (repo / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    serial: serial-pass marker\n", encoding="utf-8"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    _git(repo, "add", ".")
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=str(repo), check=True, capture_output=True, text=True,
    )
    return repo


def _stub_pytest_passes(monkeypatch, calls):
    """Neutralise the pytest side so these tests are about the node lane only
    (same seams tests/test_preflight_runner.py's `stub_passes` documents)."""
    from ouroboros import preflight_runner as pr

    monkeypatch.setattr(pr, "_verify_preflight_plugins", lambda *a, **k: [])
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a, **k: {"gw0", "gw1"})

    def _fake_pass(agent_python, worktree, temp_root, args, timeout):
        calls.append(list(args))
        return (0, "", "")

    monkeypatch.setattr(pr, "_execute_pytest_pass", _fake_pass)


@requires_node
def test_gate_blocks_on_the_candidate_worktrees_web_tests(tmp_path, monkeypatch):
    """The glob runs against the ASSEMBLED candidate: an uncommitted failing
    web test activates the lane, and its block returns before any pytest pass
    starts — the node lane is the first consumer of the shared budget."""
    from ouroboros import preflight_runner as pr

    repo = _gate_repo(tmp_path)
    (repo / "web" / "tests").mkdir(parents=True)
    (repo / "web" / "package.json").write_text(json.dumps({"type": "module"}), encoding="utf-8")
    (repo / "web" / "tests" / "boom.test.js").write_text(_FAILING_TEST, encoding="utf-8")
    calls = []
    _stub_pytest_passes(monkeypatch, calls)

    result = pr.run_hermetic_pytest(repo, timeout=120)

    assert result is not None and "NODE_TESTS_FAILED" in result
    assert calls == [], "a red node lane must return before any pytest pass spawns"


@requires_node
def test_gate_runs_node_then_both_pytest_passes_when_green(tmp_path, monkeypatch):
    """Capability preservation: a green node lane hands over to the unchanged
    two-pass pytest split instead of replacing or reordering it."""
    from ouroboros import preflight_runner as pr

    repo = _gate_repo(tmp_path)
    (repo / "web" / "tests").mkdir(parents=True)
    (repo / "web" / "package.json").write_text(json.dumps({"type": "module"}), encoding="utf-8")
    (repo / "web" / "tests" / "ok.test.js").write_text(_PASSING_TEST, encoding="utf-8")
    calls = []
    _stub_pytest_passes(monkeypatch, calls)

    assert pr.run_hermetic_pytest(repo, timeout=120) is None
    assert len(calls) == 2, "both pytest passes must still run after a green node lane"


def test_gate_import_and_call_site_are_wired():
    """`run_hermetic_pytest` calls the lane by its module-level name (so tests
    and operators can stub `preflight_runner.run_node_tests`), after candidate
    assembly and before the pass loop."""
    import inspect

    from ouroboros import preflight_runner as pr

    assert pr.run_node_tests is pn.run_node_tests
    source = inspect.getsource(pr.run_hermetic_pytest)
    assert "run_node_tests(worktree, temp_root, timeout, max_output)" in source
    assert source.index("_copy_untracked(repo, worktree)") < source.index("run_node_tests(") < source.index("for spec in passes:")


# ── CI parity ─────────────────────────────────────────────────────────


def _ci_job_block(job: str) -> str:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    block = re.search(
        rf"^  {re.escape(job)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:$|\Z)",
        ci, re.MULTILINE | re.DOTALL,
    )
    assert block, f"ci.yml has no `{job}:` job"
    return block.group(1)


@pytest.mark.parametrize("job", ["quick-test", "full-test"])
def test_each_ci_job_runs_the_node_suite_the_gate_runs(job):
    """Both CI jobs carry the exact suite the gate runs, derived from the
    module's own WEB_DIR/TESTS_GLOB constants so the three surfaces (gate, CI,
    web/package.json) cannot drift apart silently."""
    block = _ci_job_block(job)
    expected = f"run: cd {pn.WEB_DIR} && node --test {pn.TESTS_GLOB}"
    assert block.count(expected) == 1, f"{job} does not run the gate's node suite exactly once"
    # The Windows runner in full-test needs bash for the glob; keep the step
    # byte-identical in both jobs so parity stays trivially checkable.
    assert "shell: bash" in block.split(expected)[0].rsplit("- name:", 1)[-1]
    # ...and the job declares the provisioning contract, so a node regression
    # there is a loud failure instead of silent real-spawn skips.
    assert f'{_REQUIRE_NODE_ENV}: "1"' in block


def test_full_test_provisions_node_pinned_by_sha():
    block = _ci_job_block("full-test")
    setup = re.search(r"uses: actions/setup-node@([0-9a-f]{40})", block)
    assert setup, "full-test must provision node via a SHA-pinned actions/setup-node"
    assert "node-version: '22'" in block


def test_package_json_script_matches_the_gate_glob():
    package = json.loads((REPO_ROOT / pn.WEB_DIR / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["test"] == f"node --test {pn.TESTS_GLOB}"


# ── ESLint second layer (owner decision D-13 = A) ─────────────────────
#
# The gate keeps the dependency-free acorn walker (web/tests/no_undef.test.js);
# CI additionally runs ESLint's `no-undef` as an independent second opinion.
# Everything a registry move could change is pinned: the step itself, the
# frozen install, the exact versions, and the one-rule config.

_ESLINT_STEP = "run: cd web && npm ci --no-audit --no-fund && npm run lint:undef"


@pytest.mark.parametrize("job", ["quick-test", "full-test"])
def test_each_ci_job_runs_the_eslint_second_layer_after_the_node_suite(job):
    block = _ci_job_block(job)
    node_suite = f"run: cd {pn.WEB_DIR} && node --test {pn.TESTS_GLOB}"
    assert block.count(_ESLINT_STEP) == 1, f"{job} does not run the ESLint no-undef layer exactly once"
    # A second layer is only a second opinion when the first still runs.
    assert block.index(node_suite) < block.index(_ESLINT_STEP), (
        f"{job} runs ESLint before (or instead of) the gate's node suite"
    )
    # Windows runner in full-test: bash for the `cd web && ...` chain, same as the node step.
    assert "shell: bash" in block.split(_ESLINT_STEP)[0].rsplit("- name:", 1)[-1]


def test_the_eslint_layer_is_exact_pinned_and_lockfile_frozen():
    """`npm ci` refuses to run without a lockfile that matches package.json, so
    the committed lockfile plus exact (no `^`/`~`) devDependency versions make
    the CI verdict independent of what the registry serves tomorrow."""
    web = REPO_ROOT / pn.WEB_DIR
    package = json.loads((web / "package.json").read_text(encoding="utf-8"))
    dev = package["devDependencies"]
    assert set(dev) == {"eslint", "globals"}, "the layer needs eslint and its globals tables, nothing else"
    for name, version in dev.items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), f"{name} is not exact-pinned: {version!r}"
    assert package["scripts"]["lint:undef"] == "eslint --max-warnings=0 app.js modules tests"
    lock = json.loads((web / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["lockfileVersion"] >= 3
    for name, version in dev.items():
        assert lock["packages"][f"node_modules/{name}"]["version"] == version, (
            f"package-lock.json resolves {name} to a version other than the pinned one"
        )
    # An installed tree must never dirty the checkout (build_repo_bundle.py refuses a dirty tree).
    assert "web/node_modules/\n" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")


def test_the_eslint_config_carries_only_no_undef_over_the_gate_scope():
    """One rule, so a divergence between the two layers is a scope-model bug in
    one of them and never a style disagreement; the same file scope and the
    same by-design vendored globals as the acorn walker."""
    config = (REPO_ROOT / pn.WEB_DIR / "eslint.config.js").read_text(encoding="utf-8")
    rules = re.findall(r"rules:\s*\{([^}]*)\}", config)
    assert rules, "eslint.config.js declares no rules"
    for body in rules:
        assert re.findall(r"'([a-z-]+)'\s*:", body) == ["no-undef"], body
    assert "files: ['app.js', 'modules/**/*.js']" in config
    assert "globals.browser" in config and "globals.es2022" in config
    for name in ("Chart", "marked", "DOMPurify", "mermaid", "hljs"):
        assert f"{name}: 'readonly'" in config, f"vendored <script> global {name} missing from the ESLint layer"
    # ...and the first layer is still there: ESLint is a second opinion, not a replacement.
    assert (REPO_ROOT / pn.WEB_DIR / "tests" / "no_undef.test.js").is_file()


@requires_node
def test_node_options_cannot_green_a_red_suite(tmp_path, monkeypatch):
    """Adversarial repro (wave 1): an inherited NODE_OPTIONS test filter
    (--test-name-pattern matching nothing) used to exit 0 on a red suite.
    The lane scrubs NODE_OPTIONS exactly as the pytest lane scrubs PYTEST_*."""
    worktree = _worktree(tmp_path, {
        "web/tests/boom.test.js": _FAILING_TEST,
    })
    monkeypatch.setenv("NODE_OPTIONS", "--test-name-pattern=__no_such_test__")

    result = pn.run_node_tests(worktree, tmp_path / "t", 120, 8000)

    assert result["returncode"] not in (0, None)
    assert "NODE_TESTS_FAILED" in result["error"]
    # The suite genuinely EXECUTED and named its red test — the scrub, not an
    # option-rejection exit, produced the failure (wave-2 hardening).
    assert "boom_marker_test" in result["error"]
