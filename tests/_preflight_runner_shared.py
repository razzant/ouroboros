"""Repo builders and skip conditions shared by the preflight gate suites.

Split out of ``tests/test_preflight_runner.py`` when that module was divided by theme; the
definitions are verbatim, so every sibling suite builds the same throwaway repository, reads
the same plugin verdict and honours the same real-spawn skip it was written against.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import textwrap

import pytest



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
