"""The two-pass plan the preflight gate runs, and the plugins it verifies first.

This module owns the parallel and serial pass specs that must mirror CI, the lane expression
that must match pyproject, the plugin minimums and the verification that runs on the
interpreter hosting the suite, and the worker count the gate pins after probing it.

The diagnosis, the pass orchestration, the candidate capture, the commit gate, the real
hermetic runs and the process containment were split verbatim into
``tests/test_preflight_diagnosis.py``, ``tests/test_preflight_pass_orchestration.py``,
``tests/test_preflight_candidate_capture.py``, ``tests/test_preflight_commit_gate.py``,
``tests/test_preflight_hermetic_runs.py``, ``tests/test_preflight_process_containment.py``
and ``tests/test_preflight_process_reaping.py``; the repo builders they share live in
``tests/_preflight_runner_shared.py``.
"""

from __future__ import annotations

import os
import re
import sys

import pytest


from tests._preflight_runner_shared import (
    REPO_ROOT,
    _PREFLIGHT_PLUGIN_PROBLEMS,
    _REAL_SPAWN_SKIP_REASON,
    _REQUIRE_PLUGINS_ENV,
)


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
    assert "test_preflight_runner.py" in _SERIAL_TEST_FILES
