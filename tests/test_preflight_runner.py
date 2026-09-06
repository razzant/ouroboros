import errno
import inspect
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest

from ouroboros.platform_layer import force_kill_pid, pid_is_alive
from ouroboros.settings_defaults import settings_env_keys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The nested fixture repos below are TINY (1-3 probe tests) and pin the xdist
# worker count so the parallel pass costs seconds, not a full `-n auto` fan-out.
_FIXTURE_PYTEST_INI = "[pytest]\nmarkers =\n    serial: real-process/port/global-state test; runs in the serial pass\n"


def _preflight_plugin_problems() -> list:
    """Ask the GATE'S OWN verifier whether this interpreter can host a real pass."""
    from ouroboros.preflight_runner import _verify_preflight_plugins

    with tempfile.TemporaryDirectory(prefix="ouroboros-plugin-probe-") as probe:
        return _verify_preflight_plugins(sys.executable, pathlib.Path(probe))


# Probed ONCE, at import, and stated in exactly ONE place.
#
# The real-spawn tests further down run a NESTED pytest under `sys.executable`,
# and the gate is deliberately fail-closed: `run_hermetic_pytest` returns
# PREFLIGHT_PLUGIN_MISSING before any pass unless that interpreter really carries
# pytest-xdist and pytest-timeout. On an interpreter without them, every one of
# those tests fails on that single environment fact instead of on its own
# subject — a dozen identical failures, none of which is about the behaviour
# under test, and all of which drown the one message that would tell an operator
# what to install.
#
# So the fact is asserted once (below, in
# `test_plugin_verification_passes_on_the_interpreter_running_this_suite`) and
# otherwise carried by this marker. The hermetic and stubbed tests stay
# UNCONDITIONAL — they are the ones that pin the fail-closed behaviour itself,
# and they must never be silenced by the environment they are describing.
#
# The skip must not be able to conceal ITSELF, which is what an earlier revision
# did: the control test carried this same marker, so its `_PREFLIGHT_PLUGIN_
# PROBLEMS == []` assertion was skipped in precisely the case where it would have
# failed, and an unprovisioned run reported a clean suite with a dozen quiet
# skips while every behavioural proof of the parallel-pass machinery went
# unexecuted. `OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS` is the seam that fixes that:
# where the environment is provisioned (CI's `quick-test`/`full-test` set it, and
# a repair-round gate command should too) the control test HARD-FAILS on a
# missing plugin instead of skipping, so the twelve skips can never be silent.
_PREFLIGHT_PLUGIN_PROBLEMS = _preflight_plugin_problems()

_REQUIRE_PLUGINS_ENV = "OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS"

# Names the tests that go dark, so a `-rs` line is actionable rather than a count.
_REAL_SPAWN_SKIP_REASON = (
    "this interpreter cannot host a real preflight pass, so a nested run can only "
    "return PREFLIGHT_PLUGIN_MISSING — install pytest-xdist>=3.5 and "
    "pytest-timeout>=2.1 into it (pyproject.toml declares both), or set "
    f"{_REQUIRE_PLUGINS_ENV}=1 to turn this skip into a hard failure: "
    + "; ".join(_PREFLIGHT_PLUGIN_PROBLEMS)
)

requires_preflight_plugins = pytest.mark.skipif(
    bool(_PREFLIGHT_PLUGIN_PROBLEMS), reason=_REAL_SPAWN_SKIP_REASON
)


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)


def _commit_all(repo: pathlib.Path) -> None:
    _git(repo, "add", ".")
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )


def _delete_loose_object(repo: pathlib.Path, oid: str) -> None:
    obj_path = repo / ".git" / "objects" / oid[:2] / oid[2:]
    assert obj_path.exists(), (
        "fixture assumption: a fresh repo keeps this object loose"
    )
    # git stores loose objects read-only; Windows refuses to unlink a
    # read-only file (WinError 5), so lift the bit first.
    obj_path.chmod(0o644)
    obj_path.unlink()


def _make_repo(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    """Init a tiny git repo whose `tests/` holds only the given probe files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "ouroboros")
    (repo / "pytest.ini").write_text(_FIXTURE_PYTEST_INI, encoding="utf-8")
    (repo / "tests").mkdir()
    for rel, body in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(body), encoding="utf-8")
    _commit_all(repo)
    return repo


@pytest.fixture
def two_pass_env(monkeypatch):
    """Deterministic env for the real-spawn two-pass tests."""
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    # Private seam (scrubbed before the candidate ever sees it), clamped at the
    # >=2 floor: the fixture repos below hold 1-3 probe tests, so a full `-n auto`
    # fan-out would spend minutes on worker startup for nothing.
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    # The operator-environment downgrade the scrub must defeat. Every real-spawn
    # test below therefore ALSO proves the parallel lane stayed parallel: if
    # PYTEST_XDIST_AUTO_NUM_WORKERS were inherited, `-n auto` would resolve to one
    # worker and the "parallel" pass would silently be a serial one.
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "1")
    # This file is serial-only, but never let an outer xdist worker's marker
    # leak into the nested run and turn the lane-partition probes false-red.
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_TESTRUNUID", raising=False)


# ── Pass-spec contract (unit) ─────────────────────────────────────────


def test_two_pass_specs_mirror_ci(monkeypatch):
    """The gate runs CI's exact split: parallel `not serial` pass, then `serial`."""
    from ouroboros import preflight_runner as pr

    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    specs = pr._preflight_pass_specs()
    assert [spec.label for spec in specs] == ["parallel", "serial"]
    parallel, serial = specs

    assert parallel.args[:3] == ["tests/", "-m", f"not serial and {pr.LANE_EXCLUSION_EXPR}"]
    for flag in ("-n", "auto", "--dist", "loadscope", "--max-worker-restart=0",
                 "--timeout=300", "--timeout-method=thread"):
        assert flag in parallel.args, f"parallel pass lost {flag}"

    # The serial pass is flag-free like CI: no -n, no --dist, no per-test timeout.
    assert serial.args == ["tests/", "-m", f"serial and {pr.LANE_EXCLUSION_EXPR}", "-q", "--tb=line", "--no-header"]

    # `parallel` is read off the argv, not the label: it gates the xdist-only
    # diagnoses so they cannot fire on a pass that never ran a worker.
    assert parallel.parallel is True
    assert serial.parallel is False


def test_the_parallel_pass_forces_the_plugins_and_the_worker_probe(monkeypatch):
    """Verifying the INTERPRETER proves the plugins are installed; it does not
    prove the CANDIDATE's pytest configuration lets them load. A repo whose
    `addopts` carries `-p no:xdist -p no:timeout` — with a conftest declaring and
    ignoring `-n`/`--dist`/`--timeout` — would run the nominal parallel lane
    serially and exit 0. So the pass loads them by name, and carries the gate's
    own probe plugin so the worker count can be read back afterwards.

    Entry-point names (`xdist`, `timeout`), not module paths: pytest skips an
    entry point whose name is already registered, so forcing them cannot
    double-register the plugin autoload would have loaded anyway — which
    `-p xdist.plugin` would, failing an otherwise green run."""
    from ouroboros import preflight_runner as pr

    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    parallel, serial = pr._preflight_pass_specs()

    assert pr._FORCED_PLUGIN_FLAGS == ["-p", "xdist", "-p", "timeout"]
    for index in range(0, len(pr._FORCED_PLUGIN_FLAGS), 2):
        pair = pr._FORCED_PLUGIN_FLAGS[index:index + 2]
        assert pair[1] and "." not in pair[1], f"{pair[1]!r} is a module path, not an entry-point name"
    joined = " ".join(parallel.args)
    assert " ".join(pr._FORCED_PLUGIN_FLAGS) in joined, "the parallel pass does not force its plugins"
    assert f"-p {pr._WORKER_PROBE_MODULE}" in joined, "the parallel pass carries no worker probe"

    # The serial/legacy passes use neither plugin, so forcing them there would
    # make a lane that deliberately needs nothing fail on a missing dependency.
    assert "-p" not in serial.args
    assert "-p" not in pr._preflight_pass_specs(["tests/", "-n", "2"])[0].args, (
        "an explicit caller argv must be forwarded verbatim"
    )


def test_lane_expr_matches_pyproject():
    """`LANE_EXCLUSION_EXPR` is the SSOT: a command-line `-m` REPLACES the
    pyproject addopts `-m`, so any drift silently re-admits an excluded lane."""
    from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    addopts = re.search(r"^addopts\s*=\s*\"(.*)\"\s*$", pyproject, re.MULTILINE)
    assert addopts, "pyproject.toml addopts line not found"
    markexpr = re.search(r"-m '([^']+)'", addopts.group(1))
    assert markexpr, "pyproject.toml addopts carries no -m markexpr"
    assert markexpr.group(1) == LANE_EXCLUSION_EXPR


def _ci_pytest_suite_commands(job: str) -> list[tuple[str, str]]:
    """The `(markexpr, trailing_flags)` of ONE ci.yml job's full-suite pytest runs.

    Scoped to a single job on purpose. The gate's claim is "each CI job runs this
    split", and a whole-file substring search cannot see the difference between
    both jobs carrying it and one job carrying it twice.
    """
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    block = re.search(
        rf"^  {re.escape(job)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:$|\Z)",
        ci,
        re.MULTILINE | re.DOTALL,
    )
    assert block, f"ci.yml has no `{job}:` job"
    # `tests/` only — the per-file guard steps run one file and are not the split.
    return re.findall(r'run: python -m pytest tests/ -m "([^"]+)"(.*)', block.group(1))


@pytest.mark.parametrize("job", ["quick-test", "full-test"])
def test_each_ci_job_runs_the_same_split_the_gate_runs(job, monkeypatch):
    """The gate's whole promise is "what CI will do, before you push". Both jobs
    must carry BOTH passes with the gate's own markexprs and the gate's own
    parallel flag vector — checked per job, because a search of the whole
    workflow file stays green while either job alone drifts, and it never
    compared `PARALLEL_PASS_FLAGS` against any CI command at all."""
    from ouroboros import preflight_runner as pr

    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    commands = _ci_pytest_suite_commands(job)
    assert len(commands) == 2, f"{job} runs {len(commands)} full-suite passes, expected 2"
    (ci_parallel_markexpr, ci_parallel_tail), (ci_serial_markexpr, ci_serial_tail) = commands

    gate_parallel, gate_serial = pr._preflight_pass_specs()
    assert ci_parallel_markexpr == gate_parallel.args[2], f"{job} parallel markexpr drifted from the gate"
    assert ci_serial_markexpr == gate_serial.args[2], f"{job} serial markexpr drifted from the gate"

    # The exact flag VECTOR, contiguous and in order: `--dist loadscope` without
    # `--max-worker-restart=0` is a different gate, and so is the same set in an
    # order that separates a flag from its value.
    assert " ".join(pr.PARALLEL_PASS_FLAGS) in ci_parallel_tail, (
        f"{job}'s parallel pass no longer runs PARALLEL_PASS_FLAGS verbatim: {ci_parallel_tail!r}"
    )
    # ...and the serial pass stays flag-free, or CI's serial lane would carry a
    # per-test timeout the gate's serial pass does not.
    for flag in pr.PARALLEL_PASS_FLAGS:
        assert flag not in ci_serial_tail.split(), f"{job}'s serial pass carries {flag}"


def test_explicit_pytest_args_stay_single_pass():
    """An explicit `pytest_args=` keeps today's single-pass behaviour verbatim."""
    from ouroboros import preflight_runner as pr

    specs = pr._preflight_pass_specs(["tests/test_one.py", "-q"])
    assert len(specs) == 1
    assert specs[0].args == ["tests/test_one.py", "-q"]
    assert specs[0].parallel is False
    # ...but a caller who brings their own `-n` still gets the xdist diagnoses.
    assert pr._preflight_pass_specs(["tests/", "-n", "4"])[0].parallel is True


@pytest.mark.parametrize("token,parallel", [
    ("-n", True),
    ("--numprocesses", True),
    ("--dist", True),
    # pytest accepts a short option and its value as ONE token, and xdist really
    # does distribute such a run — so it must reach the xdist-only diagnoses.
    ("-n4", True),
    ("-nauto", True),
    ("-nlogical", True),
    ("--numprocesses=4", True),
    # ...while `--no-header` (in DEFAULT_PYTEST_ARGS, i.e. on EVERY invocation)
    # starts with `-n` as a STRING and must never be read as an xdist flag.
    ("--no-header", False),
    ("--tb=line", False),
    ("-q", False),
])
def test_parallel_detection_reads_the_argv_not_the_label(token, parallel):
    """`PreflightPass.parallel` gates both hard-block diagnoses. Missing a real
    xdist form costs a crashed worker its remediation; matching a non-xdist form
    hands an ordinary failure a mark-it-serial instruction that cannot fix it."""
    from ouroboros import preflight_runner as pr

    assert pr._preflight_pass_specs(["tests/", token])[0].parallel is parallel


def test_explicit_empty_pytest_args_stay_single_pass():
    """An EMPTY explicit argv is still an explicit argv: ONE pass, never the
    two-pass split (which would run tests the caller never asked for under xdist
    requirements they never opted into). Only `pytest_args is None` means "no
    argv supplied".

    But an empty sequence must still select `DEFAULT_PYTEST_ARGS`, because that
    is what the pre-two-pass runner's truthiness test did. Forwarding a literally
    empty argv instead changes the discovery target (rootdir, not `tests/`) and
    drops the output flags — a behaviour change smuggled in as a bug fix."""
    from ouroboros import preflight_runner as pr

    specs = pr._preflight_pass_specs([])
    assert len(specs) == 1, "an explicit empty argv must not expand to two passes"
    assert specs[0].args == pr.DEFAULT_PYTEST_ARGS, "empty argv lost the legacy defaults"
    assert specs[0].label == "single"
    assert specs[0].parallel is False
    # ...and the default (nothing supplied) is still the two-pass split.
    assert len(pr._preflight_pass_specs()) == 2
    assert len(pr._preflight_pass_specs(None)) == 2


def test_serial_escape_hatch_forces_single_pass(monkeypatch):
    """`OUROBOROS_PREFLIGHT_SERIAL=1` is the operator rollback lever to the
    legacy single serial pass. `_preflight_env` scrubs it before the candidate
    suite runs, so it can never change what the tests themselves observe."""
    from ouroboros import preflight_runner as pr

    monkeypatch.setenv("OUROBOROS_PREFLIGHT_SERIAL", "1")
    specs = pr._preflight_pass_specs()
    assert len(specs) == 1
    assert specs[0].args == pr.DEFAULT_PYTEST_ARGS
    assert "-n" not in specs[0].args
    assert "-m" not in specs[0].args
    assert specs[0].parallel is False


def test_preflight_plugins_are_declared_dependencies():
    """The gate fails closed on a missing plugin instead of degrading to serial,
    so the plugins must be declared in EVERY place the environment is provisioned.

    `pyproject.toml` is the dependency authority for source, wheel, and packaged
    installs. Declaring pytest there without pytest-xdist/pytest-timeout produced an
    installation whose every commit hard-blocks with PREFLIGHT_PLUGIN_MISSING.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for pin in ("pytest-xdist>=3.5", "pytest-timeout>=2.1"):
        assert pin in pyproject, f"[project].dependencies does not declare {pin}"


def test_required_plugin_minimums_match_requirements():
    """The probe's version floors and the declared dependencies are the same
    promise, and a probe that accepts an older xdist than the gate depends on is
    not a check."""
    from ouroboros.preflight_runner import _REQUIRED_PREFLIGHT_PLUGINS

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for _module, dist, minimum in _REQUIRED_PREFLIGHT_PLUGINS:
        pin = f"{dist}>=" + ".".join(str(part) for part in minimum)
        assert pin in pyproject, f"the probe requires {pin}, pyproject.toml does not"


def test_plugin_verification_passes_on_the_interpreter_running_this_suite(tmp_path):
    """Live control for the hostile cases below: the gate's own environment must
    verify clean, or a green result there would prove nothing.

    This is the SINGLE place the suite states its own environment, and it must
    NOT carry `requires_preflight_plugins` — that is what made the control
    tautological. The marker's predicate is `bool(_PREFLIGHT_PLUGIN_PROBLEMS)`,
    so a marked control skipped itself in exactly the case its assertion was
    written to catch, and an unprovisioned interpreter produced a clean-looking
    run in which every real-spawn proof of the parallel machinery had silently
    gone unexecuted.

    So the gate is an EXPLICIT provisioning declaration instead. With
    `OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS` set (CI's `quick-test`/`full-test` set
    it at job level) a missing plugin is one loud, actionable failure naming
    what to install. Without it the run is declaring itself unprovisioned, and
    this test reports that as a skip whose reason names both the missing
    distributions and the flag that would have made it fatal.
    """
    from ouroboros.preflight_runner import _verify_preflight_plugins

    if _PREFLIGHT_PLUGIN_PROBLEMS and not os.environ.get(_REQUIRE_PLUGINS_ENV, "").strip():
        pytest.skip(
            "unprovisioned interpreter, and this run did not declare "
            f"{_REQUIRE_PLUGINS_ENV}=1, so every real-spawn preflight regression below "
            "is INERT rather than passing — " + _REAL_SPAWN_SKIP_REASON
        )

    assert _PREFLIGHT_PLUGIN_PROBLEMS == [], (
        f"{_REQUIRE_PLUGINS_ENV} declares this environment provisioned, but "
        f"{sys.executable} cannot host a real preflight pass: "
        + "; ".join(_PREFLIGHT_PLUGIN_PROBLEMS)
    )
    # ...and the skip marker is keyed off the gate's own verifier rather than a
    # private guess: a fresh call from a different directory must agree with the
    # import-time verdict, so a provisioned interpreter really runs the
    # real-spawn lane instead of quietly skipping it.
    assert _verify_preflight_plugins(sys.executable, tmp_path) == []


def test_the_real_spawn_lane_declares_which_tests_go_dark_when_it_skips():
    """The skip must be self-reporting. Twelve `s` characters in pytest's dot
    output are indistinguishable from twelve passes at a glance, and the whole
    point of the marker is that the tests it guards are the ONLY behavioural
    proofs of the forced-plugin and worker-probe machinery the gate injects into
    every real parallel pass. So the reason names the distributions to install
    AND the flag that converts the skip into a failure."""
    assert "pytest-xdist>=3.5" in _REAL_SPAWN_SKIP_REASON
    assert "pytest-timeout>=2.1" in _REAL_SPAWN_SKIP_REASON
    assert _REQUIRE_PLUGINS_ENV in _REAL_SPAWN_SKIP_REASON
    # ...and CI, which provisions the plugins, declares the flag — otherwise no
    # shipped environment ever executes the lane and the pins stay inert.
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for job in ("quick-test", "full-test"):
        block = re.search(
            rf"^  {re.escape(job)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:$|\Z)", ci, re.MULTILINE | re.DOTALL
        )
        assert block, f"ci.yml has no `{job}:` job"
        assert f'{_REQUIRE_PLUGINS_ENV}: "1"' in block.group(1), (
            f"{job} installs the plugins but does not require them, so a provisioning "
            "regression there would show up as silent skips"
        )


def test_plugin_verification_reports_an_absent_module(tmp_path):
    """...and a genuinely absent module is reported rather than assumed present."""
    from ouroboros.preflight_runner import _probe_plugins

    problems = _probe_plugins(
        sys.executable, tmp_path, isolated=True,
        spec=(("ouroboros_definitely_not_installed", "ouroboros-nonesuch", (1, 0)),),
    )
    assert problems and "ouroboros-nonesuch" in problems[0]


def test_plugin_verification_ignores_the_candidate_working_directory(tmp_path):
    """The candidate-controlled import surface is the working directory. A probe
    run from the diff-applied tree could be answered by a candidate-supplied
    `xdist.py`, which is the whole thing the probe exists to rule out."""
    from ouroboros.preflight_runner import _probe_plugins

    (tmp_path / "ouroboros_probe_decoy.py").write_text("VERSION = '9.9'\n", encoding="utf-8")
    spec = (("ouroboros_probe_decoy", "ouroboros-probe-decoy", (1, 0)),)

    problems = _probe_plugins(sys.executable, tmp_path, isolated=True, spec=spec)
    assert problems, "an importable file in the probe cwd satisfied the isolated probe"


# ── Parallelism is real, not nominal (unit) ───────────────────────────


def test_worker_count_can_never_fall_below_two(monkeypatch):
    """A "parallel" pass on ONE worker exercises no concurrency at all, yet the
    argv still says `-n` and the green return is accepted as proof. The count is
    therefore clamped: the private test seam may only lower it TO the floor."""
    from ouroboros import preflight_runner as pr

    monkeypatch.setenv(pr._PREFLIGHT_WORKERS_ENV, "1")
    assert pr._preflight_worker_count() == 2
    monkeypatch.setenv(pr._PREFLIGHT_WORKERS_ENV, "0")
    assert pr._preflight_worker_count() == 2
    monkeypatch.setenv(pr._PREFLIGHT_WORKERS_ENV, "-4")
    assert pr._preflight_worker_count() == 2
    monkeypatch.setenv(pr._PREFLIGHT_WORKERS_ENV, "not-a-number")
    assert pr._preflight_worker_count() >= 2
    monkeypatch.setenv(pr._PREFLIGHT_WORKERS_ENV, "3")
    assert pr._preflight_worker_count() == 3
    monkeypatch.delenv(pr._PREFLIGHT_WORKERS_ENV, raising=False)
    assert pr._preflight_worker_count() >= 2


@pytest.mark.parametrize("hostile", [
    # Decides what `-n auto` resolves to: an inherited "1" runs the whole
    # "parallel" pass on a single worker while the argv still reads parallel.
    ("PYTEST_XDIST_AUTO_NUM_WORKERS", "1"),
    # Can append `-p no:xdist`, its own `-m`, or `-p no:randomly` to EVERY run.
    ("PYTEST_ADDOPTS", "-p no:xdist"),
    # Decide whether the verified plugins load at all.
    ("PYTEST_PLUGINS", "evil_plugin"),
    ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"),
    # Leak the OUTER run's identity into the nested one, so a candidate test can
    # see "I am under xdist" when it is not, or vice versa.
    ("PYTEST_XDIST_WORKER", "gw7"),
    ("PYTEST_XDIST_TESTRUNUID", "deadbeef"),
    ("PYTEST_CURRENT_TEST", "tests/test_outer.py::test_outer (call)"),
])
def test_external_pytest_controls_never_reach_the_candidate_suite(tmp_path, monkeypatch, hostile):
    """Every one of these weakens the pass while the argv still reads like a full
    parallel run, and a green pass under any of them is indistinguishable from a
    green pass under the real gate. They are dropped WHOLESALE (`PYTEST_*`), not
    by name, so a plugin adding tomorrow's control variable is covered too."""
    from ouroboros.preflight_runner import _preflight_env

    key, value = hostile
    monkeypatch.setenv(key, value)
    env = _preflight_env(tmp_path / "root", tmp_path / "root" / "repo")

    if key == "PYTEST_XDIST_AUTO_NUM_WORKERS":
        # Not merely dropped: re-injected with a count THIS process chose, or
        # `-n auto` would fall back to the runner host's cpu count anyway.
        assert int(env[key]) >= 2, "the inherited single-worker downgrade survived"
    else:
        assert key not in env, f"{key} leaked into the candidate suite"


@pytest.mark.parametrize("key", settings_env_keys())
def test_projected_settings_keys_never_reach_the_candidate_suite(tmp_path, monkeypatch, key):
    """`config.apply_settings_to_env` exports every one of these from the owner's
    settings.json into the server process, and most carry neither the OUROBOROS_
    prefix nor a secret suffix (OPENAI_COMPATIBLE_BASE_URL, USE_LOCAL_*,
    LOCAL_MODEL_*, MCP_*, GITHUB_REPO, TOTAL_BUDGET). Inherited, they made the
    hermetic verdict depend on the operator's install profile: with
    OPENAI_COMPATIBLE_BASE_URL alone set, tests/test_settings_effort.py routed on
    it and lost four tests; USE_LOCAL_MAIN lost a different four. The scrub is
    DERIVED from `settings_env_keys()`, so tomorrow's settings key is covered
    without anyone extending a hand-kept list."""
    from ouroboros.preflight_runner import _preflight_env

    monkeypatch.setenv(key, "owner-runtime-state")
    env = _preflight_env(tmp_path / "root", tmp_path / "root" / "repo")
    assert key not in env, f"{key} leaked into the candidate suite"


def test_the_gate_pins_the_worker_count_it_verified(tmp_path, monkeypatch):
    """The injected count is the clamped one, not whatever `os.cpu_count()` or
    the operator environment happened to say."""
    from ouroboros import preflight_runner as pr

    monkeypatch.setenv(pr._PREFLIGHT_WORKERS_ENV, "3")
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "1")
    env = pr._preflight_env(tmp_path / "root", tmp_path / "root" / "repo")
    assert env["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "3"
    # ...and the seam itself is OUROBOROS_*, so the scrub removes it first: a
    # candidate test can neither read nor re-lower it.
    assert pr._PREFLIGHT_WORKERS_ENV not in env


def test_the_worker_probe_is_prepended_to_pythonpath(tmp_path, monkeypatch):
    """`-p ouroboros_preflight_probe` is an ordinary import, so it resolves through
    `sys.path`. PREPENDED, not appended: an inherited `PYTHONPATH` entry (or a
    candidate module of that name) that came first would shadow the gate's probe,
    and a shadowed probe writes no worker files — which reads as
    `PREFLIGHT_PARALLELISM_LOST` on a lane that was fine, or, if the shadow writes
    its own, as parallelism that never happened."""
    from ouroboros import preflight_runner as pr

    monkeypatch.setenv("PYTHONPATH", "/inherited/first")
    root = tmp_path / "root"
    env = pr._preflight_env(root, root / "repo")

    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(pr._probe_dir(root.resolve(strict=False))), (
        "the gate's probe dir is shadowable"
    )
    assert "/inherited/first" in entries, "the inherited PYTHONPATH was discarded, not prepended to"

    module = pr._install_worker_probe(root)
    assert module.startswith(pr._WORKER_PROBE_MODULE + "_"), module
    assert module != pr._WORKER_PROBE_MODULE, "the probe module name carries no nonce"
    assert (pr._probe_dir(root) / f"{module}.py").exists()
    # The RETURNED name is the only one a file is written under, so a caller that
    # discards it and passes the bare stem hands pytest a `-p` import that cannot
    # resolve — a usage error (exit 4) before a single test is collected, on every
    # default two-pass run.
    assert not (pr._probe_dir(root) / f"{pr._WORKER_PROBE_MODULE}.py").exists()
    # Written into the gate's disposable temp root, NEVER into the worktree the
    # candidate diff was applied to — the gate must not add a file to the tree it
    # is judging.
    assert not (root / "repo").exists()


def test_the_parallel_pass_loads_the_probe_module_that_was_actually_written(tmp_path, monkeypatch):
    """The nonce is only worth having if it reaches the argv.

    `-p name` resolves through `sys.path`, and `python -m pytest` puts the
    CANDIDATE worktree at `sys.path[0]`, ahead of the PYTHONPATH entry the gate
    prepends — so a repository containing a top-level `ouroboros_preflight_probe`
    would shadow the gate's probe, report zero workers, and hard-block every green
    parallel run as PREFLIGHT_PARALLELISM_LOST. The nonce defeats that only if the
    pass loads the generated name; building the specs against the stem instead
    breaks the pass outright.
    """
    from ouroboros import preflight_runner as pr

    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    module = pr._install_worker_probe(tmp_path)
    parallel, serial = pr._preflight_pass_specs(probe_module=module)

    assert module in parallel.args, "the parallel pass does not load the generated probe"
    assert parallel.args[parallel.args.index(module) - 1] == "-p", "the probe is not a `-p` operand"
    assert pr._WORKER_PROBE_MODULE not in parallel.args, (
        "the parallel pass loads the bare stem, which names no file on the probe path"
    )
    assert module not in serial.args, "the serial pass needs no worker probe"


def test_serial_file_manifest_entries_exist_and_partition():
    """A renamed/removed file left in `_SERIAL_TEST_FILES` silently shrinks the
    serial lane; this file itself must stay in it (it spawns real pytest trees)."""
    from tests.conftest import _SERIAL_TEST_FILES

    for name in _SERIAL_TEST_FILES:
        assert (REPO_ROOT / "tests" / name).exists(), f"_SERIAL_TEST_FILES names a missing file: {name}"
    assert {"test_preflight_runner.py", "test_preflight_process_containment.py"} <= _SERIAL_TEST_FILES


# ── Result classification (unit, no spawn) ────────────────────────────


def test_classify_plugin_missing():
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "ERROR: usage: pytest [options] [file_or_dir]\n"
        "pytest: error: unrecognized arguments: -n auto --dist loadscope\n"
    )
    result = _classify_pass_result(
        "parallel", 4, output, 8000, parallel=True, agent_python="/opt/py/bin/python3"
    )
    assert result is not None
    assert "PREFLIGHT_PLUGIN_MISSING" in result
    assert "/opt/py/bin/python3" in result
    assert "pytest-xdist" in result and "pytest-timeout" in result


def test_classify_does_not_blame_plugins_for_an_unrelated_usage_error():
    """`-n` is a SUBSTRING of `--no-header`, which `DEFAULT_PYTEST_ARGS` passes on
    every invocation, so a substring test blamed missing xdist for any usage
    error at all. Only a whole-token match against the parallel flags counts."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "pytest: error: unrecognized arguments: --no-header\n"
    result = _classify_pass_result("parallel", 4, output, 8000, parallel=True)
    assert result is not None, "a usage error still blocks"
    assert "PREFLIGHT_PLUGIN_MISSING" not in result
    assert "pytest-xdist" not in result


def test_classify_skips_xdist_diagnoses_on_a_non_parallel_pass():
    """The serial and legacy passes carry no `-n`/`--dist`, so they have no
    workers to crash and no xdist plugin to miss. Both labels would still block
    (nonzero exit), but with a remediation that is wrong by construction."""
    from ouroboros.preflight_runner import _classify_pass_result

    crash_text = "[gw0] node down: Not properly terminated\nworker gw0 crashed while running 'x'\n"
    result = _classify_pass_result("serial", 1, crash_text, 8000, parallel=False)
    assert result is not None
    assert "PARALLEL_WORKER_CRASH" not in result
    assert "@pytest.mark.serial" not in result, "cannot ask a serial-lane test to be marked serial"
    assert "node down" in result, "the raw pytest output is still reported"

    usage = "pytest: error: unrecognized arguments: -n auto\n"
    assert "PREFLIGHT_PLUGIN_MISSING" not in _classify_pass_result("single", 4, usage, 8000, parallel=False)


@pytest.mark.parametrize("returncode,output,label,remediation_marker,body_marker", [
    (4, "pytest: error: unrecognized arguments: -n auto --dist loadscope\n",
     "PREFLIGHT_PLUGIN_MISSING", "pyproject.toml", "unrecognized arguments"),
    (1, "worker gw3 crashed while running 'tests/test_x.py::test_y'\n",
     "PARALLEL_WORKER_CRASH", "@pytest.mark.serial", "worker gw3 crashed"),
])
def test_hard_block_remediation_survives_caller_truncation(
    returncode, output, label, remediation_marker, body_marker
):
    """`review_helpers._run_review_preflight_tests` re-truncates this string from
    the TAIL at the same 8000 limit (ouroboros/utils.py::truncate_review_artifact),
    so a remediation emitted AFTER a full-budget body is the first thing
    destroyed — exactly when the output is long enough to need it."""
    from ouroboros.preflight_runner import _classify_pass_result

    # Comfortably larger than the longest remediation (~430 chars) so this test
    # keeps pinning ORDER, not the exact prose length; the too-small-budget
    # extreme is pinned separately by
    # `test_diagnosis_never_overruns_a_declared_max_output`.
    max_output = 1200
    noisy = output + ("E   assert 0 == 1  # routine failing-suite noise\n" * 200)
    result = _classify_pass_result("parallel", returncode, noisy, max_output, parallel=True)

    assert result is not None
    assert label in result
    assert len(result) <= max_output, f"diagnosis overran the caller's {max_output}-char budget"
    assert result.index(remediation_marker) < result.index(body_marker), (
        "the remediation must precede the pytest body, or a tail cut removes it first"
    )


def test_per_test_timeout_kill_is_not_told_to_mark_the_test_serial():
    """`--timeout-method=thread` does not FAIL a slow test — it `os._exit`s the
    whole worker, which xdist reports with its crash phrasing. The generic
    remediation would then tell the author to mark a merely-slow test
    `@pytest.mark.serial`, and the serial pass carries NO per-test timeout, so
    obeying it moves the hang into the one pass that cannot bound it. Same hard
    block, different instruction."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "+++++++++++++++++++++++++++ Timeout +++++++++++++++++++++++++++\n"
        "~~~~~~~~~~~~~~ Stack of MainThread (123) ~~~~~~~~~~~~~~\n"
        '  File "tests/test_slow.py", line 4, in test_slow\n'
        "    time.sleep(600)\n"
        "+++++++++++++++++++++++++++ Timeout +++++++++++++++++++++++++++\n"
        "[gw0] node down: Not properly terminated\n"
        "worker gw0 crashed and worker restarting disabled\n"
    )
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert result is not None
    assert "PARALLEL_WORKER_CRASH" in result, "a killed worker is still a hard block"
    assert "300s per-test limit" in result
    assert "faster or split" in result
    assert "Do NOT mark it @pytest.mark.serial" in result, (
        "the serial pass has no per-test timeout, so the wrong instruction relocates the hang"
    )
    assert "Find the test that spawns a real process" not in result, (
        "the generic crash remediation must not be emitted for a timeout kill"
    )
    assert "never a flake/retry" not in result
    # The evidence the author needs is still there.
    assert "time.sleep(600)" in result


def test_signal_method_timeout_banner_also_avoids_the_serial_remediation():
    """The signal method spells the same event `Failed: Timeout >300.0s`. Pinned
    so the diagnosis stays right if `--timeout-method` is ever changed."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "E   Failed: Timeout >300.0s\nworker gw2 crashed while running 'tests/test_slow.py::test_slow'\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert "PARALLEL_WORKER_CRASH" in result
    assert "Find the test that spawns a real process" not in result
    assert "300s per-test limit" in result


def test_genuine_crash_still_gets_the_mark_it_serial_remediation():
    """The timeout branch must not swallow the ordinary case: a worker that dies
    with no pytest-timeout banner is the real-process/port/global-state class,
    and `@pytest.mark.serial` IS the fix for it."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "worker gw0 crashed and worker restarting disabled\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert "PARALLEL_WORKER_CRASH" in result
    assert "@pytest.mark.serial" in result
    assert "never a flake/retry" in result
    assert "300s per-test limit" not in result


def test_crash_diagnosis_keeps_the_full_pytest_output():
    """A crash label must never COST the reader the report. The matched xdist
    lines are a highlighted prefix, not a replacement: a worker usually dies
    alongside ordinary failures, and a pattern false positive would otherwise
    delete every real failure line the author needs."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "tests/test_a.py::test_one FAILED\n"
        "[gw0] node down: Not properly terminated\n"
        "tests/test_b.py::test_two FAILED\n"
        "E   assert 0 == 1  # the failure that actually explains the crash\n"
    )
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert result is not None
    assert "PARALLEL_WORKER_CRASH" in result
    assert "node down" in result
    for survivor in ("test_one FAILED", "test_two FAILED", "the failure that actually explains"):
        assert survivor in result, f"crash diagnosis discarded {survivor!r}"


def test_crash_patterns_ignore_a_bare_worker_id_in_test_text():
    """A bare `worker gwN` appears in ordinary assertion text and captured logs.
    Only xdist's controller phrasing counts, so a routine failure keeps its
    ordinary report and its ordinary (absent) remediation."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "tests/test_pool.py::test_scheduling FAILED\n"
        "E   AssertionError: assert 'worker gw1' in queue_label\n"
    )
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert result is not None, "an ordinary failure still blocks"
    assert "PARALLEL_WORKER_CRASH" not in result
    assert "@pytest.mark.serial" not in result
    assert "test_scheduling FAILED" in result


@pytest.mark.parametrize("crash_line", [
    "worker 'gw0' crashed while running 'tests/test_x.py::test_y'",
    "[gw0] node down: Not properly terminated",
    # The parallel pass runs with `--max-worker-restart=0`, so THIS is the
    # phrasing xdist actually emits for the configuration the gate uses.
    "worker gw0 crashed and worker restarting disabled",
    "replacing crashed worker gw1",
    "Maximum crashed workers reached: 0",
])
def test_crash_patterns_cover_xdist_controller_phrasing(crash_line):
    """Tightening the patterns away from a bare worker id must not lose the
    lines xdist really prints — especially the restart-disabled variant that
    `--max-worker-restart=0` selects."""
    from ouroboros.preflight_runner import _classify_pass_result

    result = _classify_pass_result("parallel", 1, crash_line + "\n", 8000, parallel=True)
    assert result is not None
    assert "PARALLEL_WORKER_CRASH" in result, f"unrecognised xdist crash line: {crash_line!r}"


def test_crash_patterns_survive_terminal_decoration():
    """pytest/xdist colour their output. A pattern matched against the raw line
    would miss a controller line wrapped in SGR escapes — a silent downgrade to
    the generic diagnosis for exactly the coloured terminals humans use."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "\x1b[31m[gw0] node down: Not properly terminated\x1b[0m\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)
    assert "PARALLEL_WORKER_CRASH" in result


@pytest.mark.parametrize("innocent_line", [
    # Every crash PHRASE, in text a passing/failing test can legitimately emit.
    "E   AssertionError: assert 'node down: pool drained' in status_banner",
    "E   AssertionError: assert log == 'crashed while running the migration'",
    "INFO     scheduler:pool.py:88 replacing crashed worker in the pool",
    "WARNING  scheduler:pool.py:91 maximum crashed workers reached; giving up",
    "E   ValueError: worker gw1 crashed",
])
def test_crash_patterns_need_the_whole_controller_line_shape(innocent_line):
    """The patterns are UNANCHORED (xdist re-emits `handle_crashitem` mid-line in
    the `-q` short summary), so a free-substring match would label any test that
    reasons about worker pools a `PARALLEL_WORKER_CRASH` and hand its author a
    mark-it-serial instruction the marker cannot satisfy. Only the complete
    shape — phrase plus the worker id or numeric operand xdist always prints —
    counts."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = f"tests/test_pool.py::test_scheduling FAILED\n{innocent_line}\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert result is not None, "an ordinary failure still blocks"
    assert "PARALLEL_WORKER_CRASH" not in result, f"false crash label from: {innocent_line!r}"
    assert "@pytest.mark.serial" not in result
    assert "test_scheduling FAILED" in result


def test_crash_pattern_still_matches_the_mid_line_short_summary_form():
    """`handle_crashitem` reports the crash as a TestReport longrepr, so under
    `-q` pytest re-emits it inside the short-summary line. Pinned because it is
    the reason the patterns may not be `^`-anchored — the real-spawn regression
    `test_worker_crash_is_hard_block` depends on this exact shape."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "FAILED tests/test_x.py::test_y - worker 'gw0' crashed while running 'tests/test_x.py::test_y'\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)
    assert "PARALLEL_WORKER_CRASH" in result


def test_a_genuine_crash_is_not_reclassified_by_timeout_text_elsewhere_in_the_output():
    """`_crash_remediation` INVERTS the instruction, so its patterns must be as
    tight as the crash patterns. Matching a bare `Timeout >30s` anywhere in the
    pass output let one unrelated test's assertion text rewrite a real crash's
    remediation into "make it faster, do NOT mark it serial" — the one
    instruction that leaves the real-process test breaking every parallel run."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "tests/test_banner.py::test_render FAILED\n"
        "E   AssertionError: assert 'Timeout >30s' in banner\n"
        "worker gw0 crashed and worker restarting disabled\n"
    )
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert "PARALLEL_WORKER_CRASH" in result
    assert "@pytest.mark.serial" in result, "a real crash lost its mark-it-serial fix"
    assert "never a flake/retry" in result
    assert "300s per-test limit" not in result


def test_diagnosis_keeps_the_failure_summary_tail_when_output_is_cut():
    """pytest prints FAILURES + the short summary at the END: the bounded
    diagnosis cuts the MIDDLE, never the tail naming the failing tests."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "= session starts =\n" + ("noise PASSED\n" * 400)
        + "= FAILURES =\nE assert 0 == 1\nFAILED tests::test_the_culprit\n"
    )
    result = _classify_pass_result("parallel", 1, output, 2000, parallel=True)
    assert result is not None and len(result) <= 2000
    assert "session starts" in result  # the head survives too
    assert "...(truncated)..." in result
    assert "test_the_culprit" in result  # the tail survives the cut


@pytest.mark.parametrize("max_output", [1, 40, 200])
def test_diagnosis_never_overruns_a_declared_max_output(max_output):
    """`_diagnosis` promises the returned string stays inside the caller's
    limit. The `PREFLIGHT_PLUGIN_MISSING` remediation alone is ~380 chars, so
    the header+remediation prefix must be cut too when the budget is smaller
    than it — not returned whole."""
    from ouroboros.preflight_runner import _classify_pass_result

    usage = "pytest: error: unrecognized arguments: -n auto --dist loadscope\n"
    result = _classify_pass_result("parallel", 4, usage, max_output, parallel=True)
    assert result is not None
    assert len(result) <= max_output


def test_pass_header_reports_that_pass_s_own_duration():
    """`elapsed` is per-pass, not cumulative: a 40s serial pass after a 140s
    parallel one must print 40s, or the header blames the wrong pass for the
    budget it burned."""
    from ouroboros import preflight_runner

    assert "parallel pass, exit 1, 140s" in preflight_runner._classify_pass_result(
        "parallel", 1, "boom", 8000, parallel=False, elapsed=140.0
    )
    src = inspect.getsource(preflight_runner.run_hermetic_pytest)
    assert "pass_started = time.monotonic()" in src
    assert "elapsed = time.monotonic() - pass_started" in src


def test_classify_green_and_empty_pass():
    from ouroboros.preflight_runner import _classify_pass_result

    assert _classify_pass_result("parallel", 0, "", 8000, parallel=True) is None
    # Exit 5 is green PER PASS: a candidate repo may have zero serial tests.
    assert _classify_pass_result("serial", 5, "no tests ran", 8000, parallel=False) is None


# ── Orchestration contract (real worktree, stubbed pytest passes) ─────
#
# These drive `run_hermetic_pytest` end to end — real git worktree, real
# diff/env plumbing — with `_execute_pytest_pass` stubbed, so the budget and
# sweep contracts are pinned WITHOUT depending on pytest-xdist being installed
# in the interpreter running this file.


@pytest.fixture
def stub_passes(monkeypatch):
    """Replace the pytest spawn with a recorder, and log the temp-root sweeps.

    Both are appended to ONE ordered event log so a caller can pin not just how
    often the sweep runs but WHERE it runs relative to each pass.

    The two OTHER real-interpreter seams `run_hermetic_pytest` crosses are
    neutralised here as well, because neither can work when nothing is spawned:
    `_verify_preflight_plugins` shells out to the selected interpreter before the
    worktree exists (so with xdist absent from THIS interpreter every stubbed
    test would fail on `PREFLIGHT_PLUGIN_MISSING` instead of on its own subject),
    and `_observed_worker_ids` reads the worker files a real xdist run writes (a
    recorded pass writes none, so every green two-pass case would fail on
    `PREFLIGHT_PARALLELISM_LOST`). Both behaviours have their own dedicated
    tests, which install their own expectations:
    `test_plugins_are_verified_before_the_candidate_tree_exists`,
    `test_the_legacy_single_pass_does_not_require_the_parallel_plugins`,
    `test_a_nominally_parallel_pass_on_one_worker_is_a_hard_block` and the
    real-spawn `test_the_parallel_pass_really_starts_more_than_one_worker`.
    """
    from ouroboros import platform_layer, preflight_runner

    events: list[tuple] = []

    def _record_sweep(marker: str) -> None:
        events.append(("sweep", marker))

    monkeypatch.setattr(platform_layer, "kill_processes_referencing", _record_sweep)
    monkeypatch.setattr(preflight_runner, "_verify_preflight_plugins", lambda *a, **k: [])
    monkeypatch.setattr(preflight_runner, "_observed_worker_ids", lambda *a, **k: {"gw0", "gw1"})

    def _install(results):
        pending = list(results)

        def _fake_pass(agent_python, worktree, temp_root, args, timeout):
            events.append(("pass", list(args), timeout))
            handler = pending.pop(0)
            result = tuple(handler() if callable(handler) else handler)
            # `_execute_pytest_pass` returns `(returncode, output, reap_error)`.
            # A 2-tuple result means "containment reported nothing wrong", which
            # is what every case that is not ABOUT containment wants to say.
            return result if len(result) == 3 else (result[0], result[1], "")

        monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _fake_pass)
        return events

    return _install


def test_temp_root_is_swept_between_passes_not_only_at_teardown(tmp_path, two_pass_env, stub_passes):
    """A pass-1 escapee (detached child, bound port, stray server) must be reaped
    BEFORE pass 2 reads the same worktree. Pinned positionally in the event log:
    deleting the in-loop sweep leaves the teardown sweep behind, which a bare
    `"kill_processes_referencing" in source` assertion cannot distinguish."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([(0, ""), (0, "")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    assert run_hermetic_pytest(repo, timeout=120) is None

    kinds = [event[0] for event in events]
    assert kinds == ["pass", "sweep", "pass", "sweep", "sweep"], (
        f"expected a sweep after EVERY pass plus one at teardown, got {kinds}"
    )
    # ...and it is the two-pass split that ran, in order.
    assert "not serial and" in events[0][1][2]
    assert events[2][1][2].startswith("serial and")


def test_second_pass_never_starts_once_the_total_budget_is_gone(tmp_path, two_pass_env, stub_passes):
    """The 900s budget is TOTAL. Clamping an exhausted remainder up to one second
    (`max(1, int(...))`) let the serial pass start AFTER the deadline and run for
    another whole second; integer truncation could also gift most of a second
    back. An exhausted budget must return without spawning anything."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    def _burn_the_budget():
        time.sleep(1.3)
        return (0, "")

    events = stub_passes([_burn_the_budget, (0, "")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = run_hermetic_pytest(repo, timeout=1)

    assert result is not None
    assert "serial pass never started" in result, result
    assert "total budget of 1 seconds" in result, result
    assert [event[0] for event in events].count("pass") == 1, "pass 2 ran past the total budget"


def test_a_red_first_pass_stops_the_run_and_returns_only_its_own_output(tmp_path, two_pass_env, stub_passes):
    """Fail-fast is what makes truncation-safety structural: one pass's output can
    never have its failing section squeezed out by a second pass sharing the same
    8000-char budget. Pinned here without xdist; the real-spawn sibling
    `test_worker_crash_is_hard_block` proves the same thing end to end."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    crash = "worker gw0 crashed and worker restarting disabled\n"
    events = stub_passes([(1, crash), (0, "SERIAL PASS OUTPUT")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None
    assert "PARALLEL_WORKER_CRASH" in result, result
    assert "SERIAL PASS OUTPUT" not in result, "output from two passes was merged"
    assert [event[0] for event in events].count("pass") == 1, "fail-fast broken: pass 2 ran"


@pytest.mark.parametrize("results,expected", [
    ([(0, ""), (5, "no tests ran")], None),
    ([(5, "no tests ran"), (0, "")], None),
    ([(5, "no tests ran"), (5, "no tests ran")], "no tests were collected"),
])
def test_exit_5_is_green_per_pass_but_blocks_when_every_pass_is_empty(
    tmp_path, two_pass_env, stub_passes, results, expected
):
    """A candidate repo may legitimately have zero tests in ONE lane, so a blanket
    "exit 5 blocks" would false-block it. The empty-`tests/` invariant is preserved
    at the orchestrator instead: only ALL passes empty is a block."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    stub_passes(results)
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = run_hermetic_pytest(repo, timeout=120)

    if expected is None:
        assert result is None, result
    else:
        assert result is not None and expected in result, result


def test_each_pass_gets_the_exact_remaining_budget(tmp_path, two_pass_env, stub_passes):
    """Pass 2's timeout is `total − elapsed` as a FLOAT, never rounded up: the
    two passes together may not outlive the total the gate advertises."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    def _spend_half_a_second():
        time.sleep(0.5)
        return (0, "")

    events = stub_passes([_spend_half_a_second, (0, "")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    assert run_hermetic_pytest(repo, timeout=60) is None

    spawns = [event for event in events if event[0] == "pass"]
    assert len(spawns) == 2
    first_timeout, second_timeout = spawns[0][2], spawns[1][2]
    assert first_timeout <= 60
    assert second_timeout < first_timeout, "pass 2 was handed a fresh budget, not the remainder"
    assert second_timeout <= 60 - 0.5, "pass 2's share was rounded up past the total budget"


def test_plugins_are_verified_before_the_candidate_tree_exists(tmp_path, two_pass_env, stub_passes, monkeypatch):
    """Missing-plugin detection may not be inferred from the candidate's own
    pytest accepting `-n`/`--dist`/`--timeout`: a conftest can declare those exact
    option names with `pytest_addoption` and ignore them, so with xdist absent the
    nominal parallel lane runs serially, exits 0 and returns GREEN.

    So the interpreter is verified independently, and — pinned here — before the
    worktree that would carry that conftest is even created."""
    from ouroboros import preflight_runner as pr

    events = stub_passes([])
    git_calls: list[list[str]] = []
    real_run_git = pr._run_git

    def _spy(repo_dir, args, **kwargs):
        git_calls.append(list(args))
        return real_run_git(repo_dir, args, **kwargs)

    monkeypatch.setattr(pr, "_run_git", _spy)
    repo = _make_repo(
        tmp_path,
        {
            "conftest.py": """
                def pytest_addoption(parser):
                    # Accepts the gate's own parallel flags and does nothing with
                    # them: proof that "pytest did not reject -n" is not evidence
                    # that xdist is installed.
                    parser.addoption("--dist", action="store", default=None)
                    parser.addoption("--timeout", action="store", default=None)
                    parser.addoption("--timeout-method", action="store", default=None)
                    parser.addoption("--max-worker-restart", action="store", default=None)
            """,
            "tests/test_plain.py": """
                def test_ok():
                    assert True
            """,
        },
    )
    # The one thing a test cannot arrange for real: an interpreter without xdist.
    # (The probe's own behaviour against a genuinely absent module is covered by
    # `test_plugin_verification_reports_an_absent_module`; what is under test HERE
    # is what the orchestrator does with the answer, and WHEN it asks.)
    probe_seen: list[list[str]] = []

    def _missing(_python, _probe_dir):
        probe_seen.append([" ".join(args) for args in git_calls])
        return ["pytest-xdist: xdist is not importable (ModuleNotFoundError: xdist)"]

    monkeypatch.setattr(pr, "_verify_preflight_plugins", _missing)

    result = pr.run_hermetic_pytest(repo, timeout=120)

    assert probe_seen, "the interpreter was never verified for a parallel run"
    assert not any("worktree add" in call for call in probe_seen[0]), (
        "the candidate tree was materialised BEFORE the interpreter was verified"
    )

    assert result is not None, "a missing parallel plugin must block, never degrade to serial"
    assert "PREFLIGHT_PLUGIN_MISSING" in result, result
    assert "pyproject.toml" in result, result
    assert "OUROBOROS_PREFLIGHT_SERIAL=1" in result, "the deliberate-rollback lever is the remediation"
    assert [event[0] for event in events].count("pass") == 0, "a pass ran on an unverified interpreter"
    assert not any(args[:2] == ["worktree", "add"] for args in git_calls), (
        "the candidate tree was materialised before the interpreter was verified"
    )


def test_the_legacy_single_pass_does_not_require_the_parallel_plugins(tmp_path, two_pass_env, stub_passes, monkeypatch):
    """Verification is scoped to passes that actually carry `-n`. The escape
    hatch exists precisely so an operator can commit WHILE provisioning, so it
    must not be gated on the plugins it deliberately does not use."""
    from ouroboros import preflight_runner as pr

    monkeypatch.setenv("OUROBOROS_PREFLIGHT_SERIAL", "1")

    def _never(_python, _probe_dir):
        raise AssertionError("the legacy single pass was gated on the parallel plugins")

    monkeypatch.setattr(pr, "_verify_preflight_plugins", _never)
    events = stub_passes([(0, "")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    assert pr.run_hermetic_pytest(repo, timeout=120) is None
    assert [event[0] for event in events].count("pass") == 1


def test_a_nominally_parallel_pass_on_one_worker_is_a_hard_block(tmp_path, two_pass_env, stub_passes, monkeypatch):
    """Verifying the INTERPRETER proves the plugins are installed; it does not
    prove the CANDIDATE let them load. A `pytest.ini` carrying
    `-p no:xdist -p no:timeout` (or `addopts = -n 0`) plus a conftest that
    declares and ignores the option names leaves a lane that is labelled parallel,
    exits 0, and never ran two things at once — so it proves nothing about the
    parallel-only defects this gate exists to catch, and returns green.

    The forced `-p xdist -p timeout` is the fix; the worker count is the PROOF,
    and it is taken from files the gate's own probe plugin writes, not from
    output the candidate could shape. Fewer workers than the floor is a hard
    block, not a downgrade — `OUROBOROS_PREFLIGHT_SERIAL=1` is how an operator
    takes a serial run deliberately."""
    from ouroboros import preflight_runner as pr

    # Installed AFTER the fixture, so this expectation wins: exactly one worker
    # reported, which is the silently-serial lane.
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a, **k: {"gw0"})
    events = stub_passes([(0, "1 passed"), (0, "1 passed")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = pr.run_hermetic_pytest(repo, timeout=120)

    assert result is not None, "a silently serial parallel lane returned green"
    assert "PREFLIGHT_PARALLELISM_LOST" in result, result
    assert "OUROBOROS_PREFLIGHT_SERIAL=1" in result, "the deliberate-rollback lever is the remediation"
    assert "gw0" in result, "the diagnosis must name what it actually observed"
    assert [event[0] for event in events].count("pass") == 1, "fail-fast broken: the serial pass ran"


def test_a_caller_supplied_parallel_argv_is_not_blocked_for_missing_worker_evidence(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """The control for the block above. An explicit `pytest_args=` is forwarded
    VERBATIM, so it never carries the gate's probe plugin and can never produce
    worker files — keying the block on `spec.parallel` would fail every such call
    with a parallelism claim the gate never made. The block keys on the probe
    being present in the argv instead."""
    from ouroboros import preflight_runner as pr

    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a, **k: set())
    events = stub_passes([(0, "1 passed")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = pr.run_hermetic_pytest(repo, timeout=120, pytest_args=["tests/", "-n", "2"])

    assert result is None, result
    assert [event[0] for event in events].count("pass") == 1
    # Prefix-tested, not equality-tested: the probe module carries a per-run
    # nonce, so `_WORKER_PROBE_MODULE not in args` would pass even if the gate
    # HAD injected the nonce-named probe into the caller's argv.
    assert not any(arg.startswith(pr._WORKER_PROBE_MODULE) for arg in events[0][1]), (
        "the gate injected a probe into a caller's argv"
    )


def test_a_pass_whose_tree_cannot_be_proven_gone_blocks_even_when_it_exits_zero(
    tmp_path, two_pass_env, stub_passes
):
    """The container's failure reason has to reach the verdict, or it is inert.

    `reap()` returns a non-empty string when containment could not be PROVED —
    an unreadable process table, a member whose environment it can no longer read,
    a tree that never stops forking, a Windows Job Object that could not be
    created or whose teardown did not confirm itself.
    Dropping that value (it used to be a bare call in a `finally:`) left the exact
    fail-open the container was written to close: an unreadable table looks
    identical to an empty one, so exit 0 over a tree nothing ever enumerated was
    reported as a clean, green pass.

    The block therefore fires on a pass that exited ZERO, and it fires before the
    second pass runs — a tree that may still be alive must not be handed one.
    """
    from ouroboros import preflight_runner as pr

    events = stub_passes([
        (0, "1 passed", "the live process table could not be enumerated"),
        (0, "1 passed"),
    ])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = pr.run_hermetic_pytest(repo, timeout=120)

    assert result is not None, "a green pass over an unprovable teardown returned green"
    assert "PREFLIGHT_CONTAINMENT_FAILED" in result, result
    assert "the live process table could not be enumerated" in result, (
        "the diagnosis must carry the container's own reason"
    )
    assert [event[0] for event in events].count("pass") == 1, (
        "the serial pass ran on top of a tree that could not be proven gone"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-enumeration containment")
def test_an_unreadable_process_table_is_a_containment_failure_not_an_empty_container(monkeypatch):
    """The other half of the chain above: `[]` and `None` must not be the same answer.

    `pids_with_env_marker` returns `[]` for "enumerated, no members" and `None`
    for "could not read the table". Conflating them makes the container answer
    "reaped" for a tree it never looked at, which is no container at all.
    """
    from ouroboros import process_containment

    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: None)
    container = process_containment.ProcessContainer()
    # A token exists from construction; no process is adopted, so nothing is
    # signalled either way — the subject is purely the enumeration verdict.
    reason = container.reap()

    assert reason, "an unreadable process table was reported as a clean reap"
    assert "enumerated" in reason, reason

    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [])
    assert process_containment.ProcessContainer().reap() == "", (
        "an empty container must still be a SUCCESSFUL reap"
    )


@pytest.mark.parametrize("max_output", [0, -1, -8000])
def test_an_unrenderable_output_budget_blocks_instead_of_passing(tmp_path, two_pass_env, stub_passes, max_output):
    """`_diagnosis` renders INSIDE the caller's budget, so a non-positive one
    produced an empty string for a real failure — and an empty diagnosis was read
    as "no failure". A red pass therefore returned a GREEN gate. A budget that
    cannot render a failure must stop the run before it starts, never silently
    pass it."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = run_hermetic_pytest(repo, timeout=120, max_output=max_output)

    assert result is not None, f"max_output={max_output} produced a green gate"
    assert "max_output" in result, result
    assert [event[0] for event in events].count("pass") == 0, "the run started on an unusable budget"


def test_a_failing_pass_blocks_even_when_its_diagnosis_cannot_be_rendered(tmp_path, two_pass_env, stub_passes):
    """The EXIT CODE decides that a pass failed; the rendered text only decides
    how it reads. Gating the block on a truthy diagnosis let a budget too small
    to hold even the header turn a red pass into a green gate."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    stub_passes([(1, "E   assert 0 == 1\n")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    # 1 char: smaller than the header, so `_diagnosis` can only return a stub.
    result = run_hermetic_pytest(repo, timeout=120, max_output=1)

    assert result, "a nonzero exit returned a falsy result — the gate read it as success"


def test_deleting_the_whole_test_suite_is_a_hard_block(tmp_path, two_pass_env, stub_passes):
    """The all-passes-empty invariant was reachable only THROUGH the passes, and
    a candidate that stages the removal of every test file removes `tests/` with
    them (git does not track empty directories). The live-path check then
    returned success before the worktree was even created — the one change that
    deletes the gate was the one change the gate waved through."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    _git(repo, "rm", "-r", "--quiet", "tests")
    assert not (repo / "tests").exists(), "fixture precondition: tests/ is gone"

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None, "a candidate that deletes tests/ must not pass the gate"
    assert "removes the entire tests/ tree" in result, result
    assert [event[0] for event in events].count("pass") == 0


@pytest.mark.parametrize("entry_point", ["commit", "review"])
def test_the_production_entry_points_do_not_short_circuit_a_deleted_suite(
    tmp_path, two_pass_env, stub_passes, monkeypatch, entry_point
):
    """The block above is only reachable if something CALLS the runner. Both
    production entry points used to check `(repo_dir / "tests").exists()` first
    and return None when it did not — so the candidate that deletes every test
    file was waved through by the callers, and the hard block inside
    `run_hermetic_pytest` was dead code no shipped path could reach.

    Each entry point is driven in the repository state that call site really
    sees, which is NOT the same state:

    * review (`_run_review_preflight_tests`) runs PRE-commit, so the deletion is
      merely staged and `HEAD` still carries the suite;
    * commit (`_run_pre_push_tests`) runs POST-commit — `_post_commit_result`
      is only reached once `commit_sha` exists, which is why its failure text
      says the commit "was already created and preserved". By then the deletion
      is in `HEAD` itself, and a HEAD-only baseline answers "this repository has
      no test suite" and returns green for the change that deleted the gate.
      Staging without committing here would have kept this pin passing against
      a state production never reaches.

    Driven through the entry points themselves for that reason: pinning the
    runner alone is what let this regress."""
    if entry_point == "commit":
        from ouroboros.tools.git import _run_pre_push_tests as under_test
    else:
        from ouroboros.tools.review_helpers import _run_review_preflight_tests as under_test

    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    _git(repo, "rm", "-r", "--quiet", "tests")
    if entry_point == "commit":
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
             "commit", "-m", "delete the suite"],
            cwd=str(repo), check=True, capture_output=True, text=True,
        )
    assert not (repo / "tests").exists(), "fixture precondition: tests/ is gone"

    class _Ctx:
        repo_dir = str(repo)

    result = under_test(_Ctx())

    assert result is not None, f"the {entry_point} entry point skipped the gate for a deleted suite"
    assert "removes the entire tests/ tree" in result, result
    assert [event[0] for event in events].count("pass") == 0


# ── Candidate assembly (universal one-diff capture) ───────────────────
#
# The candidate is assembled the same way for EVERY source-index state: one
# hardened `git diff --binary … HEAD` capture (config-pinning flag tail, raw
# bytes end to end — the exact argv is pinned in the failed-capture test) plus
# the untracked-file copy. The staged/unstaged diff pair it replaced could not
# express an
# unmerged index — the state an assisted managed-update resolver (or any
# merge_pr flow) is in when the advisory preflight runs: `git diff --cached`
# renders each conflicted path as a literal "* Unmerged path" stub and
# `git diff` as a combined `--cc` hunk, which `git apply` REJECTS when the
# payload holds nothing else (rc=128) and silently DROPS when ordinary hunks
# accompany it. These tests pin both halves of the universal scheme: the
# unmerged shapes the old pair corrupted, and the ordinary merged shapes the
# old pair handled — which the one-diff capture must keep handling.


def _start_conflicted_merge(
    repo: pathlib.Path, incoming: dict[str, str], ours: dict[str, str]
) -> None:
    """Drive `repo` into an in-progress merge whose index holds unmerged entries.

    Two real branches, a real `git merge` that stops on the conflict — no mocked
    git anywhere, because the subject under test is git's own rendering of an
    unmerged index. Asserts the fixture really produced unmerged entries so a
    test can never silently pin the ordinary merged path instead.
    """
    _git(repo, "checkout", "-b", "incoming")
    for rel, body in incoming.items():
        (repo / rel).write_text(textwrap.dedent(body), encoding="utf-8")
    _commit_all(repo)
    _git(repo, "checkout", "ouroboros")
    for rel, body in ours.items():
        (repo / rel).write_text(textwrap.dedent(body), encoding="utf-8")
    _commit_all(repo)
    merge = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "merge", "incoming"],
        cwd=str(repo), capture_output=True, text=True,
    )
    assert merge.returncode != 0, (
        f"fixture precondition: the merge must conflict, got rc=0:\n{merge.stdout}{merge.stderr}"
    )
    unmerged = subprocess.run(
        ["git", "ls-files", "-u"], cwd=str(repo),
        capture_output=True, text=True, check=True,
    )
    assert unmerged.stdout.strip(), "fixture precondition: no unmerged index entries"


def _spy_on_candidate(monkeypatch, rel_paths):
    """Replace the pytest spawn with a spy that records candidate-file contents.

    Returns the dict the spy fills: relative path -> file text, or None when the
    path is absent from the candidate worktree. Complements ``stub_passes``
    (whose fixture setup still neutralises the plugin/worker seams): the
    recorder it installs is replaced, because these tests need the WORKTREE
    argument — the one thing the recorder drops.
    """
    from ouroboros import preflight_runner

    seen: dict[str, object] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        wt = pathlib.Path(worktree)
        for rel in rel_paths:
            target = wt / rel
            seen[rel] = target.read_text(encoding="utf-8") if target.is_file() else None
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)
    return seen


def test_a_purely_conflicted_merge_runs_against_the_worktree_resolution(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """A merge whose ONLY change is the conflicted file used to kill the gate
    outright: the staged diff is nothing but the "* Unmerged path" stub and the
    unstaged diff nothing but the `--cc` hunk, so `git apply` returned rc=128
    ("No valid patches in input") and the whole preflight died as "hermetic
    preflight failed" before running a single test. The one-diff capture has no
    such rendering — the conflicted path arrives as plain worktree content — so
    the gate runs, and runs against the RESOLUTION the resolver typed, not
    against HEAD."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
    })
    _start_conflicted_merge(
        repo, incoming={"conflict.txt": "incoming\n"}, ours={"conflict.txt": "ours\n"}
    )
    (repo / "conflict.txt").write_text("resolved\n", encoding="utf-8")  # no `git add`

    stub_passes([])  # seam neutralisation only; the spy below replaces the recorder
    seen = _spy_on_candidate(monkeypatch, ["conflict.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["conflict.txt"] == "resolved\n", (
        f"candidate does not carry the worktree resolution: {seen!r}"
    )


def test_a_mixed_unmerged_index_drops_neither_staged_nor_conflicted_changes(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """The SILENT failure mode, worse than the rc=128 one: with an ordinary hunk
    in each diff stream (an auto-merged staged file, an unstaged edit) alongside
    the conflict, `git apply` exits 0 and just DROPS the `--cc` hunk. The
    candidate then carried the ordinary changes but NOT the resolution — a
    chimera tree nobody has, whose green or red verdict is equally meaningless.
    On the unfixed base this test fails on the conflict-file assertion."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
        "auto.txt": "base auto\n",
        "notes.txt": "base notes\n",
    })
    _start_conflicted_merge(
        repo,
        incoming={"conflict.txt": "incoming\n", "auto.txt": "incoming auto\n"},
        ours={"conflict.txt": "ours\n"},
    )
    # Resolve the conflict and touch an unrelated tracked file — both WITHOUT
    # `git add`, exactly how a resolver's tree looks mid-work.
    (repo / "conflict.txt").write_text("resolved\n", encoding="utf-8")
    (repo / "notes.txt").write_text("edited notes\n", encoding="utf-8")

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["conflict.txt", "auto.txt", "notes.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["auto.txt"] == "incoming auto\n", "staged auto-merged change lost"
    assert seen["notes.txt"] == "edited notes\n", "unstaged ordinary change lost"
    assert seen["conflict.txt"] == "resolved\n", (
        "the conflicted file's resolution was silently dropped — the candidate "
        f"is a chimera: {seen!r}"
    )


def test_an_unmerged_resolution_by_deletion_is_absent_from_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Deleting the conflicted file (plain `rm`, no `git rm`) is a legitimate
    resolution. `git diff --binary HEAD` renders it as an ordinary deletion
    hunk, so the candidate must NOT carry the file — a candidate that resurrects
    it from HEAD would test a tree the resolver explicitly deleted from."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
    })
    _start_conflicted_merge(
        repo, incoming={"conflict.txt": "incoming\n"}, ours={"conflict.txt": "ours\n"}
    )
    (repo / "conflict.txt").unlink()  # resolution by deletion, no `git rm`

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["conflict.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["conflict.txt"] is None, (
        f"a file deleted as the conflict resolution reappeared in the candidate: {seen!r}"
    )


def test_a_staged_delete_with_a_recreated_untracked_file_mirrors_the_live_worktree(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Classification edge: `git rm` stages a deletion, and a NEW same-named file
    written afterwards is untracked (`ls-files --others` lists it). The one-diff
    capture deletes the path from the candidate and the untracked copy then
    restores the reborn content — net effect, the candidate equals the live
    worktree, which is the whole equivalence the one-diff capture promises."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
        "victim.txt": "victim base\n",
    })
    _start_conflicted_merge(
        repo, incoming={"conflict.txt": "incoming\n"}, ours={"conflict.txt": "ours\n"}
    )
    (repo / "conflict.txt").write_text("resolved\n", encoding="utf-8")
    _git(repo, "rm", "-q", "victim.txt")
    (repo / "victim.txt").write_text("reborn\n", encoding="utf-8")  # untracked now

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["conflict.txt", "victim.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["victim.txt"] == "reborn\n", (
        f"candidate diverged from the live worktree on the recreated path: {seen!r}"
    )
    assert seen["conflict.txt"] == "resolved\n"


def test_a_failed_capture_is_a_named_hard_block_not_a_test_failure(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """When the one-diff capture itself fails, the verdict must say so in the
    gate's named-hard-block vocabulary — PREFLIGHT_CANDIDATE_ASSEMBLY, with a
    remediation that owns the failure itself and does NOT blame the merge in
    progress (an unmerged index is a supported source state for this capture,
    per the function's own docstring) — and no pass may run, because there is
    no candidate worth running it against. A bare "hermetic preflight failed"
    here reads as an infrastructure flake and invites a retry that cannot
    succeed. The interception below pins the EXACT capture argv, the whole
    config-pinning tail included (`--no-ext-diff --no-textconv --no-color
    --src-prefix=a/ --dst-prefix=b/`): dropping any of those flags re-opens
    the door to an operator git config — a diff driver, textconv filter,
    colour escapes, or diff.noprefix/srcPrefix — that reshapes the payload
    into something `git apply` cannot re-apply."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
    })
    _start_conflicted_merge(
        repo, incoming={"conflict.txt": "incoming\n"}, ours={"conflict.txt": "ours\n"}
    )

    events = stub_passes([])
    real_run_git = preflight_runner._run_git
    capture_argv = [
        "diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
        "--src-prefix=a/", "--dst-prefix=b/", "HEAD",
    ]

    def _broken_capture(repo_dir, args, **kwargs):
        if list(args) == capture_argv:
            return subprocess.CompletedProcess(
                ["git", *args], 1, "", "synthetic capture failure"
            )
        return real_run_git(repo_dir, args, **kwargs)

    monkeypatch.setattr(preflight_runner, "_run_git", _broken_capture)

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None
    assert "PREFLIGHT_CANDIDATE_ASSEMBLY" in result, result
    assert "hard block" in result, result
    assert "is not a test failure" in result, result
    assert "synthetic capture failure" in result, result
    # The remediation must not send the operator off to "finish the merge":
    # an unmerged index is a state this capture supports, so the block means
    # the capture/apply ITSELF failed and the text says exactly that.
    assert "supported source state" in result, result
    assert "mid-merge" not in result, result
    assert [event[0] for event in events].count("pass") == 0, (
        "a pass ran against a candidate whose capture failed"
    )


@pytest.mark.parametrize(
    "failure_mode, misread",
    [
        pytest.param(
            "capture_timeout", "pytest timed out",
            id="git-diff-timeout-is-not-a-pytest-timeout",
        ),
        pytest.param(
            "untracked_permission", "hermetic preflight failed",
            id="untracked-copy-permission-error-is-not-a-generic-failure",
        ),
    ],
)
def test_a_raised_assembly_exception_is_owned_by_the_assembly_block(
    tmp_path, two_pass_env, stub_passes, monkeypatch, failure_mode, misread
):
    """The assembly block must own RAISED exceptions, not only the rc!=0 path
    the failed-capture test above pins. `_run_git` raises
    subprocess.TimeoutExpired when the diff capture outruns its budget, and
    `_copy_untracked` raises OSErrors (PermissionError, FileNotFoundError) from
    the filesystem copy. On the unfixed base the block caught RuntimeError
    alone, so these flew past it into the OUTER handlers and were misread as a
    pytest timeout ("pytest timed out after N seconds") or a generic "hermetic
    preflight failed" — retryable-looking verdicts for a candidate that was
    never assembled. Both cases raise REAL exceptions through the real code
    path; neither returns a CompletedProcess(rc=1)."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
    })
    events = stub_passes([])

    if failure_mode == "capture_timeout":
        real_run_git = preflight_runner._run_git
        capture_argv = [
            "diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
            "--src-prefix=a/", "--dst-prefix=b/", "HEAD",
        ]

        def _timing_out_capture(repo_dir, args, **kwargs):
            if list(args) == capture_argv:
                raise subprocess.TimeoutExpired(cmd=["git", *args], timeout=30)
            return real_run_git(repo_dir, args, **kwargs)

        monkeypatch.setattr(preflight_runner, "_run_git", _timing_out_capture)
    else:

        def _denied_copy(repo_dir, worktree):
            raise PermissionError(13, "Permission denied", str(worktree))

        monkeypatch.setattr(preflight_runner, "_copy_untracked", _denied_copy)

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None
    assert "PREFLIGHT_CANDIDATE_ASSEMBLY" in result, result
    assert "hard block" in result, result
    assert "is not a test failure" in result, result
    assert "supported source state" in result, result
    assert misread not in result, result
    assert [event[0] for event in events].count("pass") == 0, (
        "a pass ran against a candidate whose assembly raised"
    )


def test_a_zero_context_diff_config_still_assembles_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Hunk WIDTH is the config axis the capture's flag tail cannot pin: a user
    `diff.context=0` (equivalently `GIT_DIFF_OPTS=--unified=0` in the
    environment) makes `git diff` emit zero-context hunks, which `git apply`
    REJECTS by default — so on the unfixed base an ORDINARY tracked edit died
    as PREFLIGHT_CANDIDATE_ASSEMBLY before any test ran. `--unidiff-zero` on
    the apply accepts zero-context hunks and is a no-op for hunks that carry
    context, so one flag covers both the config and the env route. The
    repo-local config below is the real reviewer reproduction, not a mock."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "victim.txt": "a\nb\nc\nd\n",
    })
    _git(repo, "config", "diff.context", "0")
    (repo / "victim.txt").write_text("a\nb\nEDITED\nd\n", encoding="utf-8")

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["victim.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["victim.txt"] == "a\nb\nEDITED\nd\n", (
        f"zero-context capture blocked or corrupted the candidate: {seen!r}"
    )


def test_a_staged_change_reverted_in_the_worktree_lands_as_head_content(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Merged-path regression for the one-diff capture: a change that is staged
    but reverted in the worktree must land as HEAD content. Both schemes model
    the WORKTREE, not the index — the old pair replayed stage(A→B) then
    unstage(B→A) and netted out, the one-diff capture simply emits no hunk —
    so this pins that dropping the two-step replay did not silently start
    honouring the index's intermediate bookkeeping."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "reverted.txt": "base\n",
    })
    (repo / "reverted.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "reverted.txt")
    (repo / "reverted.txt").write_text("base\n", encoding="utf-8")  # back to HEAD

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["reverted.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["reverted.txt"] == "base\n", (
        f"candidate honoured the staged intermediate, not the worktree: {seen!r}"
    )


def test_disposable_index_matches_source_while_files_match_live_worktree(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Tests see both projections: live bytes on disk and staged bytes in Git."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "dual.txt": "head\n",
    })
    (repo / "dual.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "dual.txt")
    (repo / "dual.txt").write_text("live\n", encoding="utf-8")

    stub_passes([])
    seen: dict[str, str] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        candidate = pathlib.Path(worktree)
        seen["live"] = (candidate / "dual.txt").read_text(encoding="utf-8")
        seen["staged"] = subprocess.run(
            ["git", "show", ":dual.txt"], cwd=candidate, check=True,
            capture_output=True, text=True,
        ).stdout
        seen["tree"] = subprocess.run(
            ["git", "write-tree"], cwd=candidate, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    source_tree = subprocess.run(
        ["git", "write-tree"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen == {"live": "live\n", "staged": "staged\n", "tree": source_tree}


def test_non_unmerged_source_write_tree_failure_is_a_hard_block(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    events = stub_passes([])
    real_run_git = preflight_runner._run_git

    def _broken_write_tree(repo_dir, args, **kwargs):
        if pathlib.Path(repo_dir).resolve() == repo.resolve() and list(args) == ["write-tree"]:
            return subprocess.CompletedProcess(
                ["git", *args], 128, "", "synthetic index corruption"
            )
        return real_run_git(repo_dir, args, **kwargs)

    monkeypatch.setattr(preflight_runner, "_run_git", _broken_write_tree)

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None and "PREFLIGHT_SOURCE_INDEX" in result
    assert "synthetic index corruption" in result
    assert [event[0] for event in events].count("pass") == 0


def test_a_chmod_only_change_reaches_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """A mode flip with identical content is a real change (a script that lost
    its executable bit fails differently under test). The capture must carry the
    old mode/new mode header and `git apply` must apply it in the candidate.

    Gated on what `git init` actually PROBED for this filesystem (core.filemode)
    rather than on the OS name: an `os.name` skip is wrong in both directions —
    a FAT/exFAT volume on POSIX cannot track the bit either, and the probe is
    the same signal git itself trusts when deciding whether to emit mode
    hunks."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "tool.sh": "#!/bin/sh\necho hi\n",
    })
    filemode = subprocess.run(
        ["git", "config", "--get", "core.filemode"], cwd=str(repo),
        capture_output=True, text=True,
    ).stdout.strip().lower()
    if filemode != "true":
        pytest.skip(f"this filesystem does not track the executable bit (core.filemode={filemode or 'unset'})")
    os.chmod(repo / "tool.sh", 0o755)  # unstaged mode-only change

    stub_passes([])
    seen: dict[str, int] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        seen["mode"] = (pathlib.Path(worktree) / "tool.sh").stat().st_mode & 0o111
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["mode"], "the executable bit never reached the candidate"


def test_crlf_content_survives_the_capture_byte_for_byte(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """The capture travels through `_run_git`'s binary pipes and `_apply_diff`'s
    UTF-8 re-encode; CRLF line endings are the classic casualty of a text-mode
    hop (a translated diff stops matching the LF worktree and `git apply`
    rejects it wholesale). Pinned as raw bytes — `read_text` would translate the
    very characters under test."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "crlf.txt": "one\n",
    })
    # Pinned, not assumed: an operator/global autocrlf=true would rewrite the
    # very bytes this test is about at add/checkout time and test nothing.
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")  # unstaged CRLF edit

    stub_passes([])
    seen: dict[str, bytes] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        seen["crlf.txt"] = (pathlib.Path(worktree) / "crlf.txt").read_bytes()
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["crlf.txt"] == b"one\r\ntwo\r\n", (
        f"CRLF bytes were translated in transit: {seen!r}"
    )


def test_a_staged_binary_change_reaches_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Non-UTF-8 binary content travels as a base85 "GIT binary patch" section —
    which only exists because the capture passes `--binary`. Dropping the flag
    would degrade the hunk to "Binary files differ", which `git apply` cannot
    replay, so a staged icon/fixture change would kill the whole gate."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
    })
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02")
    _commit_all(repo)
    (repo / "blob.bin").write_bytes(b"\x00\xff\xfe\x00")
    _git(repo, "add", "blob.bin")

    stub_passes([])
    seen: dict[str, bytes] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        seen["blob.bin"] = (pathlib.Path(worktree) / "blob.bin").read_bytes()
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["blob.bin"] == b"\x00\xff\xfe\x00", (
        f"staged binary content did not reach the candidate: {seen!r}"
    )


def test_non_utf8_text_content_survives_the_capture_byte_for_byte(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Git classifies NUL-free content as TEXT even when its bytes are not
    valid UTF-8 (latin-1 logs, cp1251 fixtures), so those bytes travel on plain
    diff lines — never inside a base85 binary section that the previous test
    already covers. The capture→apply hop used to decode the payload with
    errors="replace" and re-encode it: every non-UTF-8 byte on an added line
    became U+FFFD, the apply still succeeded, and the candidate SILENTLY
    diverged from the worktree while the gate stayed green. The payload now
    travels as raw bytes end to end; pinned with `read_bytes`, since a text
    read would mask the very substitution under test."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "latin.txt": "plain\n",
    })
    (repo / "latin.txt").write_bytes(b"plain\ncaf\xe9 au lait\n")  # unstaged latin-1 edit

    stub_passes([])
    seen: dict[str, bytes] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        seen["latin.txt"] = (pathlib.Path(worktree) / "latin.txt").read_bytes()
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["latin.txt"] == b"plain\ncaf\xe9 au lait\n", (
        f"non-UTF-8 text bytes were substituted in transit: {seen!r}"
    )


def test_a_staged_add_removed_from_the_worktree_is_absent_from_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """A file `git add`ed and then deleted from the worktree exists only in the
    index. The worktree-vs-HEAD capture emits no hunk for it (absent on both
    sides) and the untracked copy cannot see it (it is IN the index, so
    `ls-files --others` skips it) — the candidate must not resurrect it. The
    old pair reached the same absence the long way round: staged add, then
    unstaged delete."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
    })
    (repo / "ghost.txt").write_text("ghost\n", encoding="utf-8")
    _git(repo, "add", "ghost.txt")
    (repo / "ghost.txt").unlink()

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["ghost.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["ghost.txt"] is None, (
        f"an index-only file was resurrected in the candidate: {seen!r}"
    )


def test_a_failing_post_commit_gate_stops_publication(monkeypatch):
    """A hard block the MANAGED commit path converts to a warning is not a block.

    `_post_commit_result` must return the failure (not just store it in the
    warning ref) so the managed gate can act on it. For a managed-update merge
    the gate's verdict is read BEFORE the tag and the push — an auto-created
    version tag on an unverified merge is immutable and would strand the
    corrected commit. Ordinary commits deliberately keep the warning-only
    contract (their own commit stays preserved for inspection).
    """
    from ouroboros.tools import git as git_module

    monkeypatch.setattr(git_module, "_log_test_failure", lambda *a, **k: None)
    # Module-global counter the function rebinds; monkeypatch restores it so this
    # pin cannot shift another test's consecutive-failure state.
    monkeypatch.setattr(git_module, "_consecutive_test_failures", 0)
    monkeypatch.setattr(
        git_module, "_git_commit_with_tests",
        lambda ctx, force=False: "⚠️ TESTS_FAILED: Post-commit verification failed.\nPREFLIGHT_PLUGIN_MISSING",
    )

    warning_ref = [""]
    blocking = git_module._post_commit_result(object(), "msg", False, warning_ref)

    assert blocking, "the post-commit gate's failure never left the function"
    assert "PREFLIGHT_PLUGIN_MISSING" in blocking, blocking
    assert not blocking.startswith("OK"), "a red gate produced an OK-prefixed result"
    assert "TESTS_FAILED" in warning_ref[0], "the operator-visible warning was dropped"

    # The control: a green gate returns None, so the ordinary path still pushes.
    monkeypatch.setattr(git_module, "_git_commit_with_tests", lambda ctx, force=False: None)
    assert git_module._post_commit_result(object(), "msg", False, [""]) is None
    # ...and a skipped gate is not a failed one — EXCEPT under force, which the
    # managed gate uses so neither skip_tests nor the env toggle can wave a
    # managed merge through untested.
    assert git_module._post_commit_result(object(), "msg", True, [""]) is None

    # The managed gate must act BEFORE anything publishes, and "publishes"
    # starts at the TAG, not at the push. Pinned in source because driving the
    # whole commit path here would assert on mock scaffolding instead of the
    # ordering that matters.
    src = inspect.getsource(git_module._repo_commit_push)
    assert "gate_failure = _managed_post_commit_tests_gate(" in src
    guard = src.index("if gate_failure:")
    assert guard < src.index("_auto_tag_on_version_bump("), (
        "the version tag is created before the gate's verdict is read"
    )
    assert guard < src.index("_auto_push("), "the push happens before the gate's verdict is read"
    assert guard < src.index("managed_assisted_postcommit("), (
        "the managed-update path runs before the gate's verdict is read"
    )
    # ...and the helper records the terminal failed attempt rather than dropping it.
    helper = inspect.getsource(git_module._managed_post_commit_tests_gate)
    assert 'block_reason="post_commit_tests_failed"' in helper


def test_the_post_commit_gate_record_carries_the_same_review_metadata_as_its_siblings(monkeypatch):
    """A terminal ledger record that drops the review metadata loses the forensics.

    Every OTHER failure record on the commit path carries which triad models ran,
    which scope model ran, their raw results and any degradation reasons — that is
    how an operator reconstructs, after the fact, whether a block came from a real
    verdict or from a degraded review. The post-commit gate is the NEWEST terminal
    outcome and the one least is known about, so a thinner record here is exactly
    the wrong place to economise.
    """
    from ouroboros.tools import git as git_module

    recorded = {}
    monkeypatch.setattr(git_module, "_post_commit_result", lambda *a, **k: "⚠️ TESTS_FAILED: red")
    monkeypatch.setattr(
        git_module, "_managed_commit_gate_failure", lambda reason, message: message,
    )
    monkeypatch.setattr(
        git_module, "_record_commit_attempt",
        lambda ctx, message, status, **kwargs: recorded.update(status=status, **kwargs),
    )

    class _Ctx:
        _last_triad_models = ["m1", "m2"]
        _last_scope_model = "scope-model"
        _last_triad_raw_results = [{"verdict": "approve"}]
        _last_scope_raw_result = {"in_scope": True}
        _review_degraded_reasons = ["one model timed out"]

    assert git_module._managed_post_commit_tests_gate(
        _Ctx(), "msg", time.time(), False, ["⚠️ TESTS_FAILED: red"],
        {"phase": "committing_assisted"},
        fingerprints=({"fingerprint": "pre-abc"}, {"fingerprint": "post-def"}),
    )

    assert recorded.get("status") == "failed"
    assert recorded.get("triad_models") == ["m1", "m2"]
    assert recorded.get("scope_model") == "scope-model"
    assert recorded.get("triad_raw_results") == [{"verdict": "approve"}]
    assert recorded.get("scope_raw_result") == {"in_scope": True}
    assert recorded.get("degraded_reasons") == ["one model timed out"]
    # The fingerprint columns too, and `matched` rather than pending: the gate is only
    # reached once the binding check has tied the created commit to `post_fingerprint`,
    # so the ledger can name WHICH reviewed revision the gate rejected. Leaving these
    # empty for this class alone is the same forensics hole as dropping the triad data.
    assert recorded.get("pre_review_fingerprint") == "pre-abc"
    assert recorded.get("post_review_fingerprint") == "post-def"
    assert recorded.get("fingerprint_status") == "matched"
    # ...and a ctx that carries none of it still records, rather than raising on a
    # missing attribute and losing the whole entry.
    recorded.clear()
    assert git_module._managed_post_commit_tests_gate(
        object(), "msg", time.time(), False, [""], {"phase": "committing_assisted"},
    )
    assert recorded.get("status") == "failed"
    assert recorded.get("pre_review_fingerprint") == ""


def test_a_red_gate_on_a_managed_update_rolls_the_merge_back(monkeypatch):
    """A managed update whose merge fails the gate must not be left mid-transaction.

    The assisted update writes its transaction as `committing_assisted` BEFORE the
    2-parent merge commit, and that phase means one thing to boot recovery: "the
    process died while committing". Returning the gate block on its own left HEAD
    advanced onto the rejected merge, MERGE_HEAD gone, and the tx sitting in that
    phase — so the next boot promoted it to `pending_boot_smoke` and could finalize
    a merge the gate had just refused, without ever rerunning that gate. (An
    immediate retry fared no better: managed precommit verification fails against
    an already-advanced HEAD.)

    The existing failed-update path is the correct terminal state, so the seam
    routes into it: the rejected merge is preserved on a `failed-update-*` branch,
    the tree resets to `pre_update_sha`, and the marker is CLEARED so nothing can
    promote it later.

    A rollback can itself FAIL, though — no `pre_update_sha` in the marker, a
    `checkout -B` that will not run — and clearing the marker is the very thing it
    does last. So a failed rollback leaves the phase it was called to escape, and the
    danger comes straight back. The tx is therefore re-phased to a terminal
    `gate_blocked` that no recovery path advances.

    The gate is not the only return that reaches this state: BOTH review-binding
    mismatches abandon the commit after the same `committing_assisted` write, so all
    three route through the same helper.
    """
    import types

    from ouroboros.tools import git as git_module

    calls, blocked = [], []

    def _rollback(reason):
        calls.append(reason)
        return True, "reset to pre_update_sha"

    fake = types.ModuleType("supervisor.update_merge")
    fake.rollback_managed_update = _rollback
    fake.mark_update_tx_gate_blocked = (
        lambda reason, detail="": blocked.append(reason) or True
    )
    monkeypatch.setitem(sys.modules, "supervisor.update_merge", fake)

    annotated = git_module._managed_commit_gate_failure(
        "assisted_post_commit_tests_failed", "⚠️ TESTS_FAILED: red",
    )

    assert calls == ["assisted_post_commit_tests_failed"], (
        "the update transaction was abandoned in committing_assisted"
    )
    assert "TESTS_FAILED" in annotated, "the rollback swallowed the gate's own verdict"
    assert "rolled back" in annotated, annotated
    assert not blocked, (
        "a SUCCESSFUL rollback already cleared the marker; rewriting one back is how "
        "a finished transaction reappears on the next boot"
    )

    # A rollback that returns False never got as far as clearing the marker, so the
    # phase it was called to escape is still on disk. Re-phase it, or the next boot
    # resumes the merge this gate just refused.
    calls.clear()
    fake.rollback_managed_update = lambda reason: (False, "no pre_update_sha in tx marker")
    annotated = git_module._managed_commit_gate_failure(
        "assisted_post_commit_tests_failed", "⚠️ TESTS_FAILED: red",
    )
    assert blocked == ["assisted_post_commit_tests_failed"], (
        "a failed rollback left the tx in its pre-gate phase, which boot recovery "
        "reads as an interrupted commit"
    )
    assert "MANAGED_UPDATE_GATE_BLOCKED" in annotated, annotated
    assert "marked gate_blocked" in annotated, (
        f"the operator is not told the tx was pinned shut: {annotated}"
    )

    # A rollback that RAISES is no different from one that returns False, and the
    # PERSISTED state is the assertion that matters: `rollback_managed_update` runs
    # several git commands before it clears the marker, so a raise halfway through
    # leaves the same pre-gate phase on disk. The re-phase must run independently
    # of the rollback's own error handling.
    def _explode(reason):
        raise RuntimeError("no pre_update_sha recorded")

    blocked.clear()
    fake.rollback_managed_update = _explode
    annotated = git_module._managed_commit_gate_failure(
        "assisted_post_commit_tests_failed", "⚠️ TESTS_FAILED: red",
    )
    assert blocked == ["assisted_post_commit_tests_failed"], (
        "a RAISED rollback left the tx in its pre-gate phase; the exception path "
        "must attempt the terminal re-phase independently of the rollback"
    )
    assert "MANAGED_UPDATE_GATE_BLOCKED" in annotated, annotated
    assert "TESTS_FAILED" in annotated

    # And when the re-phase ITSELF cannot be written, the message must stop claiming
    # the transaction is pinned. Telling an operator a dangerous marker is terminal
    # when it is not is worse than the failure it is reporting: it is the one line
    # that would have sent them to clear it before the next boot.
    def _explode_mark(reason, detail=""):
        raise OSError("update tx marker is not writable")

    fake.mark_update_tx_gate_blocked = _explode_mark
    annotated = git_module._managed_commit_gate_failure(
        "assisted_post_commit_tests_failed", "⚠️ TESTS_FAILED: red",
    )
    assert "MANAGED_UPDATE_ROLLBACK_FAILED" in annotated, annotated
    assert "could NOT be re-phased" in annotated, (
        f"an unpinned tx is still reported as pinned shut: {annotated}"
    )

    # And the seams that reach it: the managed test gate and BOTH review-binding
    # mismatches route through the shared managed-failure helpers rather than
    # returning bare and abandoning the commit mid-transaction.
    src = inspect.getsource(git_module._repo_commit_push)
    for call in (
        'binding_kind="commit"',
        'binding_kind="tag"',
    ):
        assert call in src, (
            f"{call} is not routed through _review_binding_failure; that return "
            "abandons the commit in its pre-gate phase just as the red gate did"
        )
    assert src.count("return binding_msg\n") == 0, (
        "a binding mismatch still returns bare, leaving a managed tx parked in "
        "its pre-gate phase for boot recovery to resume"
    )
    gate_src = inspect.getsource(git_module._managed_post_commit_tests_gate)
    assert "_managed_commit_gate_failure(" in gate_src
    binding_src = inspect.getsource(git_module._review_binding_failure)
    assert "_managed_commit_gate_failure(" in binding_src


def test_a_gate_blocked_update_tx_is_never_promoted_by_boot_recovery():
    """A gate_blocked tx must never be finalized or resumed by boot recovery.

    It exists only for the path where a check rejected the update AND the rollback
    that should have erased the transaction failed. What is on disk at that point
    is a merge the gate refused, with the marker still naming it. Boot recovery's
    contract for that phase is a fresh ROLLBACK attempt (restoring pre_update_sha)
    — never `pending_boot_smoke` promotion, never assisted resumption, never a
    `finalized: True` report on the refused revision.
    """
    from supervisor import update_merge

    assert update_merge.GATE_BLOCKED_PHASE not in update_merge._ASSISTED_PHASES, (
        "gate_blocked is an assisted phase again, so `_recover_assisted_on_boot` "
        "resumes or promotes the merge a gate refused"
    )
    src = inspect.getsource(update_merge.finalize_managed_update_on_boot)
    gate_branch = src.split("if phase == GATE_BLOCKED_PHASE:", 1)
    assert len(gate_branch) == 2, (
        "the finalizer has no explicit gate_blocked branch; an unhandled phase is "
        "only safe until someone widens the fallthrough"
    )
    branch_body = gate_branch[1].split("return", 1)
    assert "rollback_managed_update(" in branch_body[0], (
        "the gate_blocked branch no longer retries the rollback that restores "
        "pre_update_sha"
    )
    assert '"finalized": False' in branch_body[1].split("\n", 1)[0], (
        "the gate_blocked branch reports the update as finalized"
    )
    assert "_finalize_pending_boot_smoke" not in gate_branch[1].split("if phase", 1)[0], (
        "the gate_blocked branch promotes the refused merge to pending_boot_smoke"
    )


def test_an_unborn_head_is_proven_absent_not_unreadable(
    tmp_path, two_pass_env, stub_passes
):
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "ouroboros")

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert [event[0] for event in events].count("pass") == 0


def test_a_broken_head_ref_does_not_masquerade_as_unborn(
    tmp_path, two_pass_env, stub_passes
):
    """A quiet rev-parse rc=1 is ambiguous until symbolic HEAD is readable."""
    from ouroboros.preflight_runner import PRE_COMMIT_PHASE, run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    _git(repo, "rm", "-r", "--quiet", "tests")
    (repo / ".git" / "refs" / "heads" / "ouroboros").write_text(
        "not-an-object-id\n", encoding="utf-8"
    )

    result = run_hermetic_pytest(repo, timeout=120, phase=PRE_COMMIT_PHASE)
    assert result is not None
    assert "PREFLIGHT_TESTS_BASELINE_UNREADABLE" in result
    assert [event[0] for event in events].count("pass") == 0


def test_a_repository_that_never_had_tests_is_still_out_of_scope(tmp_path, two_pass_env, stub_passes):
    """...and the control: the block keys on the committed history carrying a
    suite, not on the working tree lacking one, so a repo with no test suite at
    all is untouched. (A single-commit repo also has no `HEAD~1`, so the
    post-commit baseline must degrade to False rather than to an error.)"""
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "ouroboros")
    (repo / "value.py").write_text("FLAG = True\n", encoding="utf-8")
    _commit_all(repo)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert [event[0] for event in events].count("pass") == 0


def test_the_post_commit_baseline_reaches_back_exactly_one_commit(tmp_path, two_pass_env, stub_passes):
    """The `HEAD~1` consult is what makes the block reachable from the POST-commit
    gate, and it must not become a permanent one. Only the IMMEDIATELY preceding
    commit counts: one commit after a deliberate removal, neither `HEAD` nor
    `HEAD~1` carries a suite and the repository is out of scope again — otherwise
    a project that genuinely dropped its tests could never commit anything."""
    from ouroboros.preflight_runner import _head_tracks_tests, run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    _git(repo, "rm", "-r", "--quiet", "tests")
    _commit_all(repo)
    assert _head_tracks_tests(repo), "the deletion commit itself must still be in scope"

    (repo / "value.py").write_text("FLAG = True\n", encoding="utf-8")
    _commit_all(repo)

    assert not _head_tracks_tests(repo), "the baseline reached back more than one commit"
    assert run_hermetic_pytest(repo, timeout=120) is None
    assert [event[0] for event in events].count("pass") == 0


def test_an_unreadable_baseline_tree_hard_blocks_instead_of_reading_as_no_tests(
    tmp_path, two_pass_env, stub_passes
):
    """`ls-tree` returning nonzero is not on its own evidence a ref is absent:
    git fails that way too when the ref resolves fine but its tree cannot be
    read (a corrupt or missing object, a permissions/IO error). Reading that
    failure as "this ref never tracked tests" lets a candidate that deletes
    tests/ sail through the hard block below merely because git could not
    read a real, resolvable ref's tree. The corrupted ref here is HEAD~1,
    which legitimately carries the suite the deletion commit removed."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    tree_oid = subprocess.run(
        ["git", "rev-parse", "HEAD:tests"], cwd=str(repo),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    _git(repo, "rm", "-r", "--quiet", "tests")
    _commit_all(repo)

    _delete_loose_object(repo, tree_oid)

    result = run_hermetic_pytest(repo, timeout=120)
    assert result is not None, "an unreadable baseline ref must hard-block, not silently pass"
    assert "PREFLIGHT_TESTS_BASELINE_UNREADABLE" in result
    assert [event[0] for event in events].count("pass") == 0


def test_an_unreadable_head_commit_hard_blocks_the_pre_commit_baseline(
    tmp_path, two_pass_env, stub_passes
):
    from ouroboros.preflight_runner import PRE_COMMIT_PHASE, run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    head_oid = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(repo, "rm", "-r", "--quiet", "tests")
    _delete_loose_object(repo, head_oid)

    result = run_hermetic_pytest(repo, timeout=120, phase=PRE_COMMIT_PHASE)
    assert result is not None
    assert "PREFLIGHT_TESTS_BASELINE_UNREADABLE" in result
    assert [event[0] for event in events].count("pass") == 0


def test_an_unreadable_first_parent_hard_blocks_the_post_commit_baseline(
    tmp_path, two_pass_env, stub_passes
):
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    parent_oid = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(repo, "rm", "-r", "--quiet", "tests")
    _commit_all(repo)
    _delete_loose_object(repo, parent_oid)

    result = run_hermetic_pytest(repo, timeout=120)
    assert result is not None
    assert "PREFLIGHT_TESTS_BASELINE_UNREADABLE" in result
    assert [event[0] for event in events].count("pass") == 0


def test_the_pre_commit_baseline_is_head_only_after_a_deliberate_removal(tmp_path, two_pass_env, stub_passes):
    """HEAD~1 belongs to the POST-commit phase and false-blocks the pre-commit one.

    The pre-commit review runs while the candidate is still a working-tree change,
    so HEAD alone already says whether this change deletes the suite. Consulting
    HEAD~1 there means that for the FIRST unrelated change staged after a
    deliberate test-removal commit, HEAD legitimately carries no suite while
    HEAD~1 still does — and an `any()` over both rejected that change as
    "removes the entire tests/ tree". The one-commit horizon does expire, but only
    once the NEXT commit exists, which is after the pre-commit gate has already
    refused to let it be made.
    """
    from ouroboros.preflight_runner import PRE_COMMIT_PHASE, _head_tracks_tests, run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    _git(repo, "rm", "-r", "--quiet", "tests")
    _commit_all(repo)

    # An unrelated next change, staged but NOT committed — the pre-commit phase.
    (repo / "value.py").write_text("FLAG = True\n", encoding="utf-8")
    _git(repo, "add", "value.py")

    assert _head_tracks_tests(repo), "control: the post-commit baseline still sees HEAD~1's suite"
    assert not _head_tracks_tests(repo, ("HEAD",)), "control: HEAD alone carries no suite"

    assert run_hermetic_pytest(repo, timeout=120, phase=PRE_COMMIT_PHASE) is None, (
        "the pre-commit review rejected an unrelated change for a deletion it did not make"
    )
    assert [event[0] for event in events].count("pass") == 0
    # The post-commit phase keeps the wider baseline: this IS the entry point the
    # HEAD~1 consult exists for, since by then the deletion is already in HEAD.
    assert run_hermetic_pytest(repo, timeout=120) is not None, (
        "the post-commit baseline lost its HEAD~1 consult"
    )


def test_a_timed_out_pass_reports_what_the_killed_child_had_already_flushed(tmp_path, two_pass_env, stub_passes):
    """The serial pass carries no per-test timeout, so a serial hang is exactly
    the case with no other evidence of WHICH test hung — and the post-kill
    `communicate` already collects that evidence. Discarding it left the operator
    with "the serial pass timed out" and nothing else."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    stub_passes([(None, "tests/test_a.py .\ntests/test_hangs_here.py ")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None
    assert "timed out" in result
    assert "test_hangs_here" in result, "the killed pass's own output was collected then thrown away"


def test_a_second_timeout_keeps_the_excerpt_the_first_one_already_carried(tmp_path, monkeypatch):
    """The retry `communicate` is a bonus source of evidence, not the only one.

    On timeout the pass kills the tree and calls `communicate(timeout=10)` again
    to collect what pytest flushed. That retry can itself time out — an escaped
    grandchild holding the inherited pipe open is the exact case the code
    anticipates — and seeding the excerpt as `""` meant the diagnosis then carried
    nothing at all, losing the last-test evidence it exists to preserve. The FIRST
    `TimeoutExpired` already carries that output (as raw bytes, since
    `communicate` joins the partial reads before applying text-mode decoding), so
    it is the seed and the retry may only enrich it.

    The container is also reaped BEFORE the retry, not only after: the descendant
    that would make the retry hang is precisely the one the container can kill.
    """
    from ouroboros import preflight_runner as pr, process_containment

    order: list = []

    class _StuckProc:
        pid = 4242
        returncode = None
        stdout = None
        stderr = None

        def communicate(self, timeout=None):
            order.append("communicate")
            raise subprocess.TimeoutExpired(
                "pytest", timeout,
                output=b"tests/test_a.py .\ntests/test_hangs_here.py ",
                stderr=b"",
            )

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    class _FakeContainer:
        def spawn(self, argv, **kwargs):
            return _StuckProc()

        def reap(self):
            order.append("reap")
            return ""

        def close(self):
            order.append("close")

    monkeypatch.setattr(process_containment, "ProcessContainer", _FakeContainer)
    monkeypatch.setattr(pr, "_terminate_preflight_tree", lambda proc, temp_root: None)

    returncode, output, reap_error = pr._execute_pytest_pass(
        sys.executable, tmp_path, tmp_path, ["tests/"], 0.01
    )

    assert returncode is None, "a timed-out pass must report no exit code"
    assert reap_error == ""
    assert "test_hangs_here" in output, (
        "both collections timed out and the excerpt was thrown away with them"
    )
    assert order[:3] == ["communicate", "reap", "communicate"], (
        f"the container was not reaped before the retry that it unblocks: {order}"
    )


@pytest.mark.parametrize("max_output", [1, 60, 200, 8000])
def test_timeout_excerpt_stays_inside_the_budget_and_keeps_the_tail(max_output):
    """The excerpt shares the caller's 8000-char limit with the message, and the
    caller re-truncates from the TAIL — so it must be bounded here, and it must
    keep the END of the output (progress output stops at the test that never
    finished), not the beginning.

    The bound is UNCONDITIONAL, matching `_diagnosis`: when the message alone
    already fills the budget the message is cut rather than returned whole. An
    earlier revision exempted that case, which made the documented invariant one
    with an exception — and `max_output` is a caller-declared limit, so the one
    thing it must never do is depend on which branch ran."""
    from ouroboros.preflight_runner import _with_timeout_excerpt

    message = "⚠️ PRE_PUSH_TEST_ERROR: pytest timed out after 30 seconds in the serial pass"
    output = ("noise line that is not the answer\n" * 400) + "tests/test_hangs_here.py "
    result = _with_timeout_excerpt(message, output, max_output)

    assert len(result) <= max_output, f"the excerpt overran the {max_output}-char budget"
    assert result.startswith(message) or result == message[:max_output], (
        "the excerpt displaced the message it explains"
    )
    if len(result) > len(message):
        assert result.rstrip().endswith("tests/test_hangs_here.py"), "kept the head instead of the tail"


def test_timeout_message_survives_an_empty_or_missing_excerpt():
    """A pass killed before it flushed anything has no excerpt, and appending an
    empty one would leave a dangling header promising output that never comes."""
    from ouroboros.preflight_runner import _with_timeout_excerpt

    message = "⚠️ PRE_PUSH_TEST_ERROR: pytest timed out"
    assert _with_timeout_excerpt(message, "", 8000) == message
    assert _with_timeout_excerpt(message, "   \n  ", 8000) == message


# ── Real two-pass execution ───────────────────────────────────────────
#
# Every test below spawns a REAL nested pytest, so it needs the parallel-pass
# plugins in `sys.executable`; see `requires_preflight_plugins` at the top of the
# file for why that is a marker and not eleven duplicate failures.


@requires_preflight_plugins
def test_hermetic_pytest_applies_candidate_diff_and_scrubs_live_env(tmp_path, monkeypatch, two_pass_env):
    """BOTH passes must see the candidate diff and the scrubbed env — a probe in
    only one lane would leave the other lane's wiring unproven."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    # 20-space indent: `_make_repo` dedents the 16-space template around it, so
    # these land one level in, inside the probe function body.
    assertions = "\n".join(
        " " * 20 + line for line in [
            'assert value.FLAG is True',
            'assert extra_value.FLAG is True',
            'assert "OUROBOROS_MANAGED_BY_LAUNCHER" not in os.environ',
            'assert "OUROBOROS_SAFETY_MODE" not in os.environ',
            'assert "OUROBOROS_TASK_REVIEW_MODE" not in os.environ',
            'assert "OUROBOROS_FAKE_API_KEY" not in os.environ',
            'assert "ouroboros-preflight-" in os.environ["OUROBOROS_DATA_DIR"]',
            'assert os.environ["OUROBOROS_SETTINGS_PATH"].startswith(os.environ["OUROBOROS_DATA_DIR"])',
            'assert "ouroboros-preflight-" in os.environ["OUROBOROS_REPO_DIR"]',
        ]
    )
    repo = _make_repo(
        tmp_path,
        {
            "value.py": "FLAG = False\n",
            "tests/test_parallel_lane.py": f"""
                import os
                import extra_value
                import value


                def test_candidate_diff_and_env_are_hermetic():
{assertions}
            """,
            "tests/test_serial_lane.py": f"""
                import os

                import pytest

                import extra_value
                import value


                @pytest.mark.serial
                def test_candidate_diff_and_env_are_hermetic_in_serial_pass():
{assertions}
            """,
        },
    )
    # Candidate (uncommitted) changes: a tracked edit plus an untracked new file.
    (repo / "value.py").write_text("FLAG = True\n", encoding="utf-8")
    (repo / "extra_value.py").write_text("FLAG = True\n", encoding="utf-8")

    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "light")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_FAKE_API_KEY", "must-not-reach-tests")
    result = run_hermetic_pytest(repo, timeout=120)

    assert result is None, result


@requires_preflight_plugins
def test_both_passes_execute_and_partition(tmp_path, two_pass_env):
    """One worktree, one env, two passes: the unmarked probe must run under an
    xdist worker and the serial probe must NOT — the lane partition IS the gate."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    marker_parallel = tmp_path / "ran_parallel.txt"
    marker_serial = tmp_path / "ran_serial.txt"
    repo = _make_repo(
        tmp_path,
        {
            "tests/test_parallel_probe.py": f"""
                import os


                def test_runs_in_the_parallel_pass():
                    assert os.environ.get("PYTEST_XDIST_WORKER"), "expected an xdist worker"
                    open(r'{marker_parallel}', "w").write(os.environ["PYTEST_XDIST_WORKER"])
            """,
            "tests/test_serial_probe.py": f"""
                import os

                import pytest


                @pytest.mark.serial
                def test_runs_in_the_serial_pass():
                    assert "PYTEST_XDIST_WORKER" not in os.environ, "serial test ran under xdist"
                    open(r'{marker_serial}', "w").write("serial")
            """,
        },
    )

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is None, result
    assert marker_parallel.exists(), "the parallel pass never ran its probe"
    assert marker_serial.exists(), "the serial pass never ran its probe"


@requires_preflight_plugins
def test_the_parallel_pass_really_starts_more_than_one_worker(tmp_path, two_pass_env):
    """A "parallel" pass on ONE worker exercises no concurrency, yet the argv
    still says `-n`, `PreflightPass.parallel` stays True and the green return is
    accepted as proof. `-n auto` resolves through PYTEST_XDIST_AUTO_NUM_WORKERS,
    which the operator environment can carry — `two_pass_env` pins it to "1",
    exactly the inherited downgrade the scrub must defeat — so this asserts the
    property behaviourally, from inside the candidate suite."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    workers_dir = tmp_path / "observed_workers"
    workers_dir.mkdir()
    # `--dist loadscope` distributes by MODULE, so one file is one work unit.
    # xdist's initial round-robin hands the first unit to each node, so four
    # units against the two-worker floor makes both nodes report.
    files = {}
    for index in range(4):
        files[f"tests/test_scope_{index}.py"] = f"""
            import os
            import pathlib


            def test_records_its_worker():
                # The operator's downgrade never arrived: the count is the one
                # the gate chose, and it is at least two.
                assert int(os.environ["PYTEST_XDIST_AUTO_NUM_WORKERS"]) >= 2
                worker = os.environ["PYTEST_XDIST_WORKER"]
                pathlib.Path(r'{workers_dir}', worker).write_text(worker)
        """
    repo = _make_repo(tmp_path, files)

    assert run_hermetic_pytest(repo, timeout=180) is None

    observed = sorted(path.name for path in workers_dir.iterdir())
    assert len(observed) >= 2, f"the parallel lane ran on a single worker: {observed}"


@requires_preflight_plugins
def test_a_candidate_cannot_switch_the_parallel_plugins_off(tmp_path, two_pass_env):
    """Verifying the interpreter proves the plugins are INSTALLED. It says nothing
    about whether the candidate's own pytest configuration lets them LOAD, and ini
    `addopts` are PREPENDED to the gate's argv, so `-p no:xdist -p no:timeout` in
    the candidate's `pytest.ini` disarms both before the gate's flags are read.

    That is why the parallel pass appends `-p xdist -p timeout`: `consider_preparse`
    walks `-p` entries in order, so the later unblock wins over the earlier block.
    Pinned behaviourally — the candidate says no, and the lane still fans out over
    at least two real workers."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    workers_dir = tmp_path / "hostile_workers"
    workers_dir.mkdir()
    files = {
        "pytest.ini": (
            "[pytest]\n"
            "addopts = -p no:xdist -p no:timeout\n"
            "markers =\n"
            "    serial: real-process/port/global-state test; runs in the serial pass\n"
        ),
    }
    # `--dist loadscope` distributes by MODULE, so four files are four work units
    # and xdist's initial round-robin reaches both nodes.
    for index in range(4):
        files[f"tests/test_scope_{index}.py"] = f"""
            import os
            import pathlib


            def test_records_its_worker():
                worker = os.environ["PYTEST_XDIST_WORKER"]
                pathlib.Path(r'{workers_dir}', worker).write_text(worker)
        """
    repo = _make_repo(tmp_path, files)

    assert run_hermetic_pytest(repo, timeout=180) is None

    observed = sorted(path.name for path in workers_dir.iterdir())
    assert len(observed) >= 2, (
        f"the candidate switched xdist off and the gate accepted it: {observed}"
    )


@requires_preflight_plugins
def test_a_candidate_faking_the_parallel_flags_cannot_earn_a_green_pass(tmp_path, two_pass_env):
    """The full attack, end to end. `pytest.ini` blocks both plugins AND a conftest
    declares the gate's own flags with `pytest_addoption` and ignores them, so
    nothing rejects `-n`: the lane is labelled parallel, runs strictly serially and
    exits 0. Green, on a pass that never ran two things at once.

    The invariant is that this shape cannot be BOTH green and serial. Forcing the
    plugins on makes the conftest's duplicate option definitions collide with the
    real ones, which is a usage error and a block; without the collision the lane
    genuinely fans out. Either outcome is fail-closed — a silent pass is not."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    workers_dir = tmp_path / "faked_workers"
    workers_dir.mkdir()
    files = {
        "pytest.ini": (
            "[pytest]\n"
            "addopts = -p no:xdist -p no:timeout\n"
            "markers =\n"
            "    serial: real-process/port/global-state test; runs in the serial pass\n"
        ),
        "conftest.py": """
            def pytest_addoption(parser):
                # Swallow the gate's parallel flags so a plugin-less pytest accepts
                # them: "pytest did not reject -n" is not evidence of parallelism.
                parser.addoption("-n", "--numprocesses", action="store", default=None)
                parser.addoption("--dist", action="store", default=None)
                parser.addoption("--timeout", action="store", default=None)
                parser.addoption("--timeout-method", action="store", default=None)
                parser.addoption("--max-worker-restart", action="store", default=None)
        """,
    }
    for index in range(4):
        files[f"tests/test_scope_{index}.py"] = f"""
            import os
            import pathlib


            def test_records_its_worker():
                worker = os.environ.get("PYTEST_XDIST_WORKER", "none")
                pathlib.Path(r'{workers_dir}', worker).write_text(worker)
        """
    repo = _make_repo(tmp_path, files)

    result = run_hermetic_pytest(repo, timeout=180)

    observed = sorted(path.name for path in workers_dir.iterdir())
    assert result is not None or len(observed) >= 2, (
        f"a lane that never ran in parallel returned green; workers observed: {observed}"
    )


@requires_preflight_plugins
@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership containment")
def test_a_green_pass_cannot_leak_a_child_into_the_next_pass(tmp_path, two_pass_env):
    """Containment must be UNCONDITIONAL, not a timeout/crash path.

    `communicate()` returning only proves the pytest CONTROLLER exited. A test
    that spawned a child and did not wait for it leaves that child alive, and
    after the controller dies nothing can find it: the `pgrep -P` parent->child
    walk is gone with the ppid links, and the temp-root command-line sweep misses
    an argv that names no sweepable path. Such a child ran on into pass 2 — the
    very cross-pass contamination the inter-pass sweep exists to prevent — and
    then past teardown onto the machine.

    The child here calls `setsid()` (`start_new_session=True`), which is the
    HARDEST shape: it leaves the controller's process group, so the recorded pgid
    no longer names it, and it is the shape a daemonising child naturally takes.
    Only the container's environment membership token — which the kernel copied
    into the child at fork and which `setsid()`, orphaning and closed stdio all
    leave untouched — can still name it once the controller is gone.

    The probe returns IMMEDIATELY after spawning. An earlier revision recorded
    descendants from a 0.5s background poll and this test slept for two seconds
    so a sample would land while the ppid link still existed; that sleep was the
    test accommodating the defect, since a green pass that spawns and returns
    fast is precisely the leak. Membership is resolved from live kernel state at
    reap time, so no sampling has to happen at all.

    Pinned end to end: pass 1 is green, and pass 2 (a different pytest process
    entirely) observes the pid already dead."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    marker = tmp_path / "escapee.pid"
    repo = _make_repo(
        tmp_path,
        {
            # Stdio to DEVNULL so the child does NOT hold the inherited pipe open
            # — otherwise `communicate` blocks and this becomes the timeout case
            # that was already covered. The argv is deliberately path-free, so the
            # temp-root sweep cannot see it either. `start_new_session=True` puts
            # it in its OWN session and process group, so the group handle the
            # container recorded at spawn does not cover it.
            "tests/test_leaks_a_child.py": f"""
                import pathlib
                import subprocess
                import sys


                def test_spawns_a_child_and_passes():
                    child = subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(180)"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    pathlib.Path(r'{marker}').write_text(str(child.pid))
                    # No sleep: the test returns at once, so the child is born,
                    # detached and orphaned with nothing observing it. That is the
                    # shape the containment must survive.
            """,
            "tests/test_checks_the_child_is_gone.py": f"""
                import os
                import pathlib
                import time

                import pytest


                @pytest.mark.serial
                def test_the_parallel_pass_left_nothing_running():
                    pid = int(pathlib.Path(r'{marker}').read_text().strip())
                    deadline = time.time() + 15
                    while time.time() < deadline:
                        try:
                            os.kill(pid, 0)
                        except OSError:
                            return
                        time.sleep(0.2)
                    raise AssertionError(
                        "pid %d from the parallel pass is still alive in the serial pass" % pid
                    )
            """,
        },
    )

    try:
        result = run_hermetic_pytest(repo, timeout=180)
        assert result is None, result
        assert marker.exists(), "the parallel probe never spawned its child"
    finally:
        # A containment regression must not leak a 180s sleeper into the suite.
        if marker.exists():
            leaked = int(marker.read_text().strip())
            if pid_is_alive(leaked):
                force_kill_pid(leaked)


@requires_preflight_plugins
def test_worker_crash_is_hard_block(tmp_path, two_pass_env):
    """A dead xdist worker is a HARD BLOCK with mark-it-serial remediation, never
    an ordinary failure and never a retryable flake. Fail-fast: no pass 2."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    marker_serial = tmp_path / "ran_serial.txt"
    repo = _make_repo(
        tmp_path,
        {
            "tests/test_crash_probe.py": """
                import os


                def test_kills_its_worker():
                    os._exit(1)
            """,
            "tests/test_serial_probe.py": f"""
                import pytest


                @pytest.mark.serial
                def test_should_never_run():
                    open(r'{marker_serial}', "w").write("serial")
            """,
        },
    )

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None, "a crashed worker must block the commit"
    assert "PARALLEL_WORKER_CRASH" in result, result
    assert "@pytest.mark.serial" in result, result
    assert "never a flake/retry" in result, result
    assert not marker_serial.exists(), "fail-fast broken: the serial pass ran after a red pass 1"


@requires_preflight_plugins
def test_empty_serial_lane_is_green(tmp_path, two_pass_env):
    """Exit 5 is green PER PASS — a candidate repo with zero serial tests must
    not be false-blocked by an empty serial lane."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(
        tmp_path,
        {
            "tests/test_plain.py": """
                def test_ok():
                    assert True
            """,
        },
    )

    assert run_hermetic_pytest(repo, timeout=120) is None


@requires_preflight_plugins
def test_both_lanes_empty_blocks(tmp_path, two_pass_env):
    """...but a `tests/` directory that yields NO runnable test in ANY pass keeps
    blocking, preserving the empty-suite invariant."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(
        tmp_path,
        {
            "tests/helpers.py": """
                def not_a_test():
                    return True
            """,
        },
    )

    result = run_hermetic_pytest(repo, timeout=120)
    assert result is not None
    assert "no tests were collected" in result, result


@requires_preflight_plugins
def test_pass2_timeout_names_serial_pass(tmp_path, two_pass_env):
    """The 900s budget is TOTAL; pass 2 gets the remainder and its timeout must
    name the pass so a hung serial test is not mistaken for a hung parallel one."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(
        tmp_path,
        {
            "tests/test_fast.py": """
                def test_ok():
                    assert True
            """,
            "tests/test_slow_serial.py": """
                import time

                import pytest


                @pytest.mark.serial
                def test_hangs():
                    time.sleep(300)
            """,
        },
    )

    result = run_hermetic_pytest(repo, timeout=30)

    assert result is not None
    assert "timed out" in result, result
    assert "serial pass" in result, result
    assert "total budget 30 seconds" in result, result


# ── Reaper / interpreter source pins ──────────────────────────────────


def test_hermetic_pytest_timeout_invokes_full_tree_reaper():
    """The timeout path must delegate to the full-tree reaper (not a bare killpg),
    and that reaper must use the recursive PID-tree kill, escaped process-group
    kill, and the temp-root sweep so detached/reparented children cannot survive."""
    from ouroboros import preflight_runner

    pass_src = inspect.getsource(preflight_runner._execute_pytest_pass)
    assert "_terminate_preflight_tree" in pass_src

    # The process is spawned INSIDE the container, never `Popen`'d and adopted
    # afterwards: on Windows job membership only takes effect at assignment, so a
    # descendant started in that window is outside the job and survives
    # terminate/close — the exact leak the container exists to close.
    assert "container.spawn(" in pass_src
    assert "subprocess.Popen(" not in pass_src, (
        "spawning outside the container reopens the Windows job-assignment race"
    )

    # The container is reaped UNCONDITIONALLY — including after a GREEN pass,
    # which is exactly the case the `proc.poll() is None` guard skips and the case
    # `test_a_green_pass_cannot_leak_a_child_into_the_next_pass` covers. Pinned in
    # source too, because that behavioural test is POSIX-only and Windows relies
    # on the same call reaching the Job Object.
    #
    # It happens BEFORE the return, not only in `finally`: a `finally` block
    # cannot alter an already-computed return tuple, so a reap that runs only
    # there can report a containment FAILURE that no caller can ever see — which
    # is the fail-open the container exists to close. `finally` still carries a
    # reap (for the raising path, where there is no verdict to carry it) and the
    # handle release.
    returning, _, teardown = pass_src.partition("finally:")
    assert "reap_error = container.reap()" in returning, (
        "the reap result cannot reach the caller, so containment fails open"
    )
    assert "return returncode, output, reap_error" in returning
    assert "container.reap()" in teardown
    assert "container.close()" in teardown

    # The inter-pass temp-root sweep is pinned BEHAVIOURALLY by
    # `test_temp_root_is_swept_between_passes_not_only_at_teardown`, not here: a
    # bare `"kill_processes_referencing" in run_hermetic_pytest source` check
    # cannot fail, because the teardown `finally` block contains that same call
    # and predates the two-pass split.
    reaper_src = inspect.getsource(preflight_runner._terminate_preflight_tree)
    assert "kill_process_tree" in reaper_src
    assert "kill_pid_tree" in reaper_src
    assert "kill_process_group_id" in reaper_src
    assert "kill_processes_referencing" in reaper_src
    # Platform-specific process discovery stays behind platform_layer helpers.
    assert "collect_descendant_pids" in reaper_src


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_reap_fails_when_a_member_stays_alive_across_scans(monkeypatch):
    """Quiet has to mean EMPTY, not "no pid I had not already seen".

    The rescan loop used to count a scan as quiet whenever it produced no
    PREVIOUSLY UNSEEN pid, on the theory that a pid still listed after its SIGKILL
    is only a corpse awaiting `wait()`. It is not only that: `force_kill_pid`
    swallows EPERM and every other signalling error, so a member the container
    CANNOT kill is added to `seen` on the first scan, contributes nothing new on
    the second, and the loop returns success — the container reports a reaped tree
    while a token-bearing process is still running, which is the exact fail-open
    the containment work exists to close.

    The kill seam here leaves the same marker-bearing pid visible on every scan,
    which is what a failed signal looks like from inside the loop.
    """
    from ouroboros import platform_layer, process_containment

    survivor = os.getpid() + 1_000_000  # never a live pid; every probe is stubbed
    killed: list[int] = []

    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [survivor])
    monkeypatch.setattr(
        process_containment, "pid_marker_state", lambda pid, marker: process_containment.MARKER_MEMBER
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda pid: killed.append(pid))

    container = process_containment.ProcessContainer()
    error = container.reap()

    assert error, "reap reported success while a marker-bearing member was still alive"
    assert "could not be proven gone" in error, error
    assert killed, "the survivor was never even signalled"

    # The control: once the seam actually clears the member, the SAME loop returns
    # success — the failure above is about liveness, not about the loop refusing
    # to terminate.
    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [])
    assert process_containment.ProcessContainer().reap() == ""


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_reap_does_not_mistake_an_unwaited_corpse_for_a_live_member(monkeypatch):
    """...and the other direction, which is why the liveness test is not just
    `still listed`. `_execute_pytest_pass` reaps the container on the timeout path
    BEFORE it waits pytest, so the SIGKILLed root is a zombie: still holding its
    pid and its pgid in `ps`, executing nothing. Counting it live would spin the
    whole cleanup deadline and then hard-block the run on containment for what was
    really a timeout."""
    from ouroboros import process_containment

    corpse = os.getpid() + 1_000_000

    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [corpse])
    monkeypatch.setattr(
        process_containment, "pid_marker_state", lambda pid, marker: process_containment.MARKER_MEMBER
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: True)

    assert process_containment.ProcessContainer().reap() == "", (
        "an already-exited member was counted as live, so containment blocked a timeout"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_a_member_that_becomes_unreadable_is_a_leak_not_a_clean_reap(monkeypatch):
    """A member the container CANNOT read is not a member it has proven gone.

    Membership is read from the process ENVIRONMENT, which stops being readable the
    moment a member `exec`s something setuid or otherwise nondumpable, or changes
    user. Enumeration claims members positively — deliberately, so a stranger this
    user cannot inspect is never swept into the container — which means such a
    member also DISAPPEARS from the scan. Answering "" there would be the exact
    fail-open the container exists to close: an honest-looking clean teardown for a
    process still running.

    So `reap` keeps its own set of pids it has already seen as members, and a
    member whose recheck comes back UNREADABLE is reported as a leak by pid.
    """
    from ouroboros import platform_layer, process_containment

    ghost = os.getpid() + 1_000_000  # never a live pid; every probe below is stubbed
    scans = []

    def _enumerate(marker, pgid=0, since_ticks=0):
        scans.append(marker)
        # Seen once, then unreadable — so no longer enumerable as a member.
        return [ghost] if len(scans) == 1 else []

    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    monkeypatch.setattr(process_containment, "pids_with_env_marker", _enumerate)
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: process_containment.MARKER_UNREADABLE,
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid",
                        lambda pid: pytest.fail(f"pid {pid} was signalled without revalidation"))

    error = process_containment.ProcessContainer().reap()

    assert error, "a member that vanished into unreadability was reported as reaped"
    assert "could not be determined" in error, error
    assert str(ghost) in error, f"the leaked pid is not named for the operator: {error}"


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_a_root_unreadable_from_the_very_first_scan_is_still_a_leak(monkeypatch):
    """The pid `spawn` started is a member the container KNOWS, not one it re-reads.

    Every other member joins the container by being enumerated, which means being
    positively READ. The root is different: it is a member by construction. An
    earlier revision still learned about it only through enumeration, so a root
    that turned nondumpable — `exec`ing something setuid, dropping privileges — or
    changed credentials before the FIRST scan appeared in no list at all, and the
    two empty scans that followed were reported as a clean teardown of a process
    that was still running. There was no later scan to catch it either: the "once
    seen, always watched" set only holds pids it managed to see once.

    So `spawn` records the root and `reap` seeds itself with it. Here enumeration
    NEVER returns it and its membership probe is never answerable, which is the
    exact shape of that escape; the container must still block, by pid.
    """
    from ouroboros import platform_layer, process_containment

    class _FakeProc:
        pid = os.getpid() + 1_000_000  # never a live pid; every probe below is stubbed

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: _FakeProc())
    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    # The root is invisible to enumeration for the whole reap, from the first scan on.
    monkeypatch.setattr(process_containment, "pids_with_env_marker",
                        lambda marker, pgid=0, since_ticks=0: [])
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: process_containment.MARKER_UNREADABLE,
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid",
                        lambda pid: pytest.fail(f"pid {pid} was signalled without revalidation"))

    container = process_containment.ProcessContainer()
    container.spawn(["pytest"])
    error = container.reap()

    assert error, "a root that was never readable was reported as a clean teardown"
    assert str(_FakeProc.pid) in error, f"the leaked root is not named for the operator: {error}"
    assert "could not be determined" in error, error

    # The control: the same unenumerable root, ANSWERED as gone, is not a leak —
    # otherwise every ordinary pass would block on its own exited pytest.
    monkeypatch.setattr(
        process_containment, "pid_marker_state", lambda pid, marker: process_containment.MARKER_ABSENT
    )
    replacement = process_containment.ProcessContainer()
    replacement.spawn(["pytest"])
    assert replacement.reap() == "", "an exited root was mistaken for an unreadable one"


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_a_descendant_unreadable_before_it_was_ever_seen_is_still_a_leak(monkeypatch):
    """The hole the root seed does NOT plug: a descendant nobody ever managed to read.

    The root is a member by construction, and a member seen once stays in `known`
    forever. Between those two sits the case with no cover at all — a grandchild that
    `exec`s something setuid, or drops privileges, BEFORE the first scan. It was never
    enumerated, so it never entered `known`; it is not the root, so the seed does not
    name it; and its environment is unreadable, so the token can never claim it. Every
    scan came back empty and the container certified a clean teardown of a live tree.

    The process GROUP is what closes it, and only because it is kernel-held: the
    grandchild's pgid is readable from outside no matter what the process did to its
    own environment or credentials. Enumeration therefore takes the group as a second
    input, and once the pid is in `known` the unreadable probe makes it undetermined —
    which fails closed."""
    from ouroboros import platform_layer, process_containment

    class _FakeProc:
        pid = os.getpid() + 1_000_000  # never a live pid; every probe below is stubbed

    hidden = _FakeProc.pid + 7  # a grandchild, not the root

    def _enumerate(marker, pgid=0, since_ticks=0):
        # The token claims nothing: this member has been unreadable since before the
        # first scan. Only the kernel-held group still names it.
        return [hidden] if pgid == _FakeProc.pid else []

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: _FakeProc())
    monkeypatch.setattr(platform_layer, "process_group_id",
                        lambda pid: _FakeProc.pid if pid in (_FakeProc.pid, hidden) else 0)
    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.3)
    monkeypatch.setattr(process_containment, "pids_with_env_marker", _enumerate)
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: (process_containment.MARKER_ABSENT if pid == _FakeProc.pid
                             else process_containment.MARKER_UNREADABLE),
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid",
                        lambda pid: pytest.fail(f"pid {pid} was signalled without revalidation"))

    container = process_containment.ProcessContainer()
    container.spawn(["pytest"])
    assert container._pgid == _FakeProc.pid, (
        "spawn did not record the root's own group, so enumeration has only the token "
        "and a never-readable descendant is invisible for the whole reap"
    )
    error = container.reap()

    assert error, "a descendant that was never readable was reported as a clean teardown"
    assert str(hidden) in error, f"the leaked descendant is not named: {error}"


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_the_deadline_report_names_the_last_scan_that_actually_saw_something(monkeypatch):
    """A block that names no pid is a block the operator cannot act on.

    The remediation tells the operator to go and kill the pids listed, so the report
    has to be built from the last scan that SAW one — not from whichever scan the
    deadline happens to land on. Reporting the current scan means a member that
    flickers out of readability on the final probe produces "nothing is proven gone"
    with no pid at all, from a run that named it moments earlier.

    The member below is alive on the first scan and gone from the second, and the
    deadline is set to expire in the 50ms settle between them — so the scan the
    deadline lands on is empty while the run has already named a pid. One quiet scan
    is not two, so this is a BLOCK either way; the question is whether it is an
    actionable one."""
    from ouroboros import platform_layer, process_containment

    flicker = os.getpid() + 1_000_000  # never a live pid; every probe is stubbed
    scans: list[str] = []

    def _enumerate(marker, pgid=0, since_ticks=0):
        scans.append(marker)
        return [flicker] if len(scans) == 1 else []

    # Shorter than one settle interval, so it expires during the sleep after scan 1
    # and the loop exits on scan 2 — before quiet could ever reach two.
    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.03)
    monkeypatch.setattr(process_containment, "pids_with_env_marker", _enumerate)
    monkeypatch.setattr(
        process_containment, "pid_marker_state",
        lambda pid, marker: (process_containment.MARKER_MEMBER if len(scans) == 1
                             else process_containment.MARKER_ABSENT),
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda pid: None)

    error = process_containment.ProcessContainer().reap()

    assert len(scans) >= 2, (
        f"the reap exited on its first scan ({len(scans)}), so 'the LAST non-empty "
        "scan' is not being exercised at all"
    )
    assert error, "the flickering member was reported as a clean teardown"
    assert str(flicker) in error, (
        "the deadline report was built from the empty final scan, so it names no pid "
        f"for the operator to act on: {error}"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership reaping")
def test_a_member_is_signalled_at_most_once_however_long_the_scans_run(monkeypatch):
    """Killing is ONE bounded sweep; everything after it is scan-only.

    The signal is the one operation here that can hit the wrong process: between
    revalidating a pid as a member and sending SIGKILL, that pid can exit and be
    handed to a stranger. The window cannot be closed — it is the reason the
    contract is detection rather than guaranteed teardown — so the fix is to enter
    it as FEW times as possible. An earlier revision had `force_kill_pid` inside
    the rescan loop, re-signalling every still-visible member roughly every 50ms
    for up to ten seconds: two hundred throws of the same dice, buying nothing,
    since a member that survived the first SIGKILL is one we cannot kill (EPERM)
    and the block is already earned.

    The member below stays visible on every scan, which is what an unkillable one
    looks like from inside the loop; it must be signalled exactly once, and the
    verdict must still be a block.
    """
    from ouroboros import platform_layer, process_containment

    survivor = os.getpid() + 1_000_000  # never a live pid; every probe is stubbed
    killed: list[int] = []
    scans: list[str] = []

    def _enumerate(marker, pgid=0, since_ticks=0):
        scans.append(marker)
        return [survivor]

    monkeypatch.setattr(process_containment, "_REAP_DEADLINE_SEC", 0.5)
    monkeypatch.setattr(process_containment, "pids_with_env_marker", _enumerate)
    monkeypatch.setattr(
        process_containment, "pid_marker_state", lambda pid, marker: process_containment.MARKER_MEMBER
    )
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda pid: killed.append(pid))

    error = process_containment.ProcessContainer().reap()

    assert len(scans) > 2, (
        f"the reap only scanned {len(scans)} time(s), so 'at most once' is vacuous here"
    )
    assert killed == [survivor], (
        f"the sweep is not bounded: {survivor} was signalled {len(killed)} times across "
        f"{len(scans)} scans, re-entering the exit/pid-reuse race on every one"
    )
    assert error and str(survivor) in error, (
        f"signalling once must not weaken the verdict; the leak went unreported: {error}"
    )


def test_the_ps_membership_branch_answers_unreadable_for_a_live_pid(monkeypatch):
    """The same tri-state, on the POSIX systems that have no `/proc`.

    macOS and the BSDs answer membership with `ps -E`, and `ps -E` reports a
    process whose environment it may not print by simply OMITTING the environment
    — byte-identical to a process that never carried the token. Collapsing that
    into "absent" reopened, on exactly those platforms, the escape the tri-state
    was introduced to close: a member turns uninspectable, drops out of
    enumeration, and two quiet scans call it reaped. Only `ps` failing to find the
    pid at all (non-zero exit) may answer ABSENT.

    The `/proc` sibling pin forces the `/proc` branch on every POSIX host, so this
    branch is otherwise unpinned in either direction.
    """
    import types

    from ouroboros import platform_layer, process_containment

    if platform_layer.IS_WINDOWS:
        pytest.skip("POSIX environment-token membership")

    # Both shims lie about `ps`/`/proc` ONLY and delegate everything else, so the
    # branch is pinned on a Linux host too and nothing else running inside this
    # test (pytest's own reporting included) is affected.
    real_isdir = os.path.isdir
    monkeypatch.setattr(platform_layer.os.path, "isdir",
                        lambda path: False if path == "/proc" else real_isdir(path))

    result = {"rc": 0, "out": "/usr/bin/python3 -c pass\n"}
    real_run = subprocess.run

    def _run(argv, **kwargs):
        if not (isinstance(argv, (list, tuple)) and argv and argv[0] == "ps"):
            return real_run(argv, **kwargs)
        return types.SimpleNamespace(returncode=result["rc"], stdout=result["out"], stderr="")

    monkeypatch.setattr(platform_layer.subprocess, "run", _run)

    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_UNREADABLE, (
        "a live pid whose environment `ps` declined to print was reported as proof "
        "of non-membership, so an uninspectable member leaves containment silently"
    )

    # The two answers that ARE answers: the token is there, or the pid is not.
    result["out"] = "/usr/bin/python3 -c pass TOKEN=1\n"
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_MEMBER
    result["rc"], result["out"] = 1, ""
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_ABSENT, (
        "a pid `ps` cannot find must be ABSENT, or every ordinary exit blocks the gate"
    )


def test_an_unanswerable_membership_probe_is_unreadable_not_absent(monkeypatch):
    """The unit beneath that pin: `pid_marker_state` distinguishes three answers.

    Its predecessor returned a BOOLEAN, which folded `PermissionError` into "not a
    member" — the read failed, so the pid looked innocent. Only ESRCH/ENOENT (the
    pid is genuinely gone) may answer ABSENT; every other `OSError` is UNREADABLE.
    """
    import builtins

    from ouroboros import platform_layer, process_containment

    if platform_layer.IS_WINDOWS:
        pytest.skip("POSIX environment-token membership")

    # Pinned against the /proc branch on every POSIX host, so the distinction is
    # not silently unpinned on a machine that happens to lack /proc. Both shims
    # lie about `/proc` ONLY and delegate everything else, so nothing else running
    # inside this test (pytest's own reporting included) is affected.
    real_isdir = os.path.isdir
    monkeypatch.setattr(platform_layer.os.path, "isdir",
                        lambda path: True if path == "/proc" else real_isdir(path))
    real_open = builtins.open

    def _open_raising(errno_value):
        def _open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                raise OSError(errno_value, os.strerror(errno_value))
            return real_open(path, *args, **kwargs)
        return _open

    monkeypatch.setattr(builtins, "open", _open_raising(errno.EACCES))
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_UNREADABLE, (
        "an unreadable environment was reported as proof of non-membership"
    )

    # The control: a pid that is genuinely gone is ANSWERED, not undetermined, or
    # every ordinary exit would block the run.
    monkeypatch.setattr(builtins, "open", _open_raising(errno.ESRCH))
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_ABSENT
    monkeypatch.setattr(builtins, "open", _open_raising(errno.ENOENT))
    assert process_containment.pid_marker_state(1234, "TOKEN") == process_containment.MARKER_ABSENT


def test_a_windows_job_teardown_that_does_not_confirm_itself_is_a_containment_failure(monkeypatch):
    """Win32 reports failure by RETURN VALUE, and a false BOOL was being discarded.

    The Job Object is the one place teardown really is kernel-enforced, which is
    why its result is the whole Windows verdict: if `TerminateJobObject` returns
    false the job's processes are still running, and if `CloseHandle` returns false
    kill-on-close — the backstop for a termination that did not take — never fires
    AND the handle leaks. Both used to be called for effect and ignored, so `reap`
    returned "" for a job it had not torn down.

    The code must be read with `ctypes.get_last_error()`, not `ctypes.GetLastError()`:
    the handle is opened with `use_last_error=True`, which makes ctypes SNAPSHOT the
    thread's last error immediately after each call into its own private slot. The
    raw `GetLastError` reads the live thread value, which ctypes' own bookkeeping
    between the failing call and the read has by then overwritten — so the operator
    is handed an unrelated code for a containment failure.
    """
    import inspect
    import types

    from ouroboros import platform_layer, process_containment

    win_src = inspect.getsource(platform_layer.terminate_job) + inspect.getsource(
        platform_layer.close_job)
    assert "get_last_error" in win_src and "ctypes.GetLastError" not in win_src, (
        "the Win32 failure code is read with the raw GetLastError again; with "
        "use_last_error=True that is not the code the failing call set"
    )

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "ctypes", types.SimpleNamespace(get_last_error=lambda: 5),
                        raising=False)

    results = {"terminate": 0, "close": 1}
    monkeypatch.setattr(
        platform_layer, "_kernel32",
        types.SimpleNamespace(
            TerminateJobObject=lambda job, code: results["terminate"],
            CloseHandle=lambda job: results["close"],
        ),
        raising=False,
    )

    container = process_containment.ProcessContainer()
    container._job = object()
    error = container.reap()
    assert "TerminateJobObject" in error and "5" in error, error

    # A close that fails is equally a leak, and a raised call is not different from
    # a false return — both leave the job unaccounted for.
    results["terminate"], results["close"] = 1, 0
    container = process_containment.ProcessContainer()
    container._job = object()
    assert "kill-on-close never fired" in container.reap()

    def _raise(*args):
        raise OSError("the handle is invalid")

    monkeypatch.setattr(
        platform_layer, "_kernel32",
        types.SimpleNamespace(TerminateJobObject=_raise, CloseHandle=_raise),
        raising=False,
    )
    container = process_containment.ProcessContainer()
    container._job = object()
    error = container.reap()
    assert "the handle is invalid" in error, error

    # The control: a job that confirms both halves is a clean reap.
    monkeypatch.setattr(
        platform_layer, "_kernel32",
        types.SimpleNamespace(TerminateJobObject=lambda job, code: 1, CloseHandle=lambda job: 1),
        raising=False,
    )
    container = process_containment.ProcessContainer()
    container._job = object()
    assert container.reap() == ""


def test_resolve_preflight_timeout_env_override(monkeypatch):
    """`OUROBOROS_PREFLIGHT_TIMEOUT_SEC` overrides the TOTAL two-pass budget."""
    from ouroboros.preflight_runner import _resolve_preflight_timeout

    monkeypatch.delenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raising=False)
    assert _resolve_preflight_timeout(300) == 300
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "450")
    assert _resolve_preflight_timeout(300) == 450
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "not-an-int")
    assert _resolve_preflight_timeout(300) == 300
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "0")
    assert _resolve_preflight_timeout(300) == 300


def test_hermetic_pytest_prefers_agent_python_env():
    from ouroboros import preflight_runner

    runner_src = inspect.getsource(preflight_runner.run_hermetic_pytest)
    assert 'os.environ.get("OUROBOROS_AGENT_PYTHON") or sys.executable' in runner_src
    # ...and every pass is actually spawned with that resolved interpreter.
    pass_src = inspect.getsource(preflight_runner._execute_pytest_pass)
    assert '[agent_python, "-m", "pytest", *args]' in pass_src


@requires_preflight_plugins
@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group reaping behaviour")
def test_hermetic_pytest_timeout_reaps_detached_session_child(tmp_path, two_pass_env):
    """A child that escapes pytest's process group (its own session) must still be
    reaped on timeout — the orphan class the QA hit (96% CPU survivor)."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    marker = tmp_path / "child.pid"
    # NOTE: this fixture is generated user-style code that runs inside the
    # hermetic worktree (where `ouroboros` is not on sys.path), so it must stay
    # stdlib-only. start_new_session=True simulates an arbitrary skill test
    # spawning a detached child in its own session — exactly the orphan class the
    # reaper must catch. The test HARNESS itself uses platform_layer helpers.
    # Unmarked, so it hangs inside the PARALLEL pass (one xdist worker deep).
    repo = _make_repo(
        tmp_path,
        {
            "tests/test_hang.py": f"""
                import sys, subprocess, time

                def test_spawns_detached_child_and_hangs():
                    # Detached child in its OWN session escapes the pytest group's killpg.
                    subprocess.Popen(
                        [sys.executable, "-c",
                         "import os,time;open(r'{marker}','w').write(str(os.getpid()));time.sleep(180)"],
                        start_new_session=True,
                    )
                    time.sleep(180)
            """,
        },
    )

    # Allow xdist startup before timing out the detached-child probe.
    result = run_hermetic_pytest(repo, timeout=30)
    assert result is not None and "timed out" in result

    assert marker.exists(), "probe never started; reaper result is inconclusive"
    child_pid = int(marker.read_text().strip())
    deadline = time.time() + 10
    alive = True
    while time.time() < deadline:
        if not pid_is_alive(child_pid):
            alive = False
            break
        time.sleep(0.2)
    if alive:  # cleanup so a reaping regression does not leak a 180s sleeper
        force_kill_pid(child_pid)
    assert not alive, f"detached child {child_pid} survived preflight timeout reaping"


def test_an_untracked_file_with_a_non_utf8_name_reaches_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """POSIX filenames are BYTES, and Git lists them as such. Decoding that list
    as UTF-8 with `errors="replace"` turned a raw non-UTF-8 byte into U+FFFD, so
    the reconstructed path did not exist, `is_file()` said False, and the file was
    skipped in silence — an inexact candidate with no PREFLIGHT_CANDIDATE_ASSEMBLY
    block. The names are now decoded with the filesystem codec (surrogateescape),
    which round-trips the original bytes."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    if os.name != "posix":
        pytest.skip("byte filenames are a POSIX property")
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    raw_name = b"fixture_\xff.dat"
    try:
        with open(os.path.join(os.fsencode(str(repo)), raw_name), "wb") as handle:
            handle.write(b"untracked payload\n")
    except (OSError, UnicodeError):  # APFS and friends enforce UTF-8 names
        pytest.skip("this filesystem rejects non-UTF-8 filenames")
    decoded_name = os.fsdecode(raw_name)

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, [decoded_name])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen[decoded_name] == "untracked payload\n", (
        f"an untracked file vanished from the candidate on a byte filename: {seen!r}"
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "POSIX-only invariant: the guarantee is that a raw non-UTF-8 filename byte "
        "survives to the copy via os.fsdecode's surrogateescape round-trip. Windows "
        "uses a UTF-16 filesystem where such a name cannot exist, and its fs codec "
        "(utf-8/surrogatepass) raises on the synthetic 0xff byte this test injects — "
        "git never emits such a name on Windows, so the production path is unaffected."
    ),
)
def test_untracked_listing_is_decoded_with_the_filesystem_codec(tmp_path, monkeypatch):
    """Filesystem-independent pin for the same defect (the test above can only run
    where non-UTF-8 names are creatable): the listing is read as BYTES and each
    name goes through `os.fsdecode`, so the original bytes reach the copy instead
    of a U+FFFD name that matches no file on disk. POSIX-only: os.fsdecode is only
    byte-transparent under surrogateescape (POSIX); see the skip marker."""
    from ouroboros import preflight_runner

    seen_kwargs = {}

    def fake_run_git(_repo_dir, args, **kwargs):
        # Mirrors the real seam: bytes only when the caller asks for them, the old
        # utf-8/replace decode otherwise — so this pins the decision, not the stub.
        seen_kwargs.update(kwargs)
        raw = b"fixture_\xff.dat\x00"
        out = raw if kwargs.get("binary_stdout") else raw.decode("utf-8", "replace")
        return subprocess.CompletedProcess(list(args), 0, out, "")

    copied = []
    monkeypatch.setattr(preflight_runner, "_run_git", fake_run_git)
    monkeypatch.setattr(preflight_runner.shutil, "copy2", lambda src, dst: copied.append(src))
    monkeypatch.setattr(pathlib.Path, "is_file", lambda _self: True)

    preflight_runner._copy_untracked(tmp_path, tmp_path / "candidate")

    assert seen_kwargs.get("binary_stdout") is True, "the names must not be decoded by _run_git"
    assert os.fsencode(str(copied[0])).endswith(b"fixture_\xff.dat"), copied
