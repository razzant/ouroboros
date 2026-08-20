"""How the two passes are run: the budget they share and the blocks they raise.

Split verbatim out of ``tests/test_preflight_runner.py`` by theme. This module owns the
temp root swept between passes, the budget each pass is handed, the empty and red passes
that stop the run, the plugin verification that happens before the candidate tree exists,
and the hard blocks a deleted suite or an unprovable teardown must raise.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest


from tests._preflight_runner_shared import (
    _git,
    _make_repo,
)
from tests._preflight_runner_shared import stub_passes as _stub_passes
from tests._preflight_runner_shared import two_pass_env as _two_pass_env

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
stub_passes = _stub_passes
two_pass_env = _two_pass_env


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
