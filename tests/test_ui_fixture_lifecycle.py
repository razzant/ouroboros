"""The shared UI fixture owns each spawned tree through readiness, restart and teardown.

The sleeper payload keeps process containment real without starting a server or
requiring browser binaries. Real UI consumers run in their existing marker lane.
Cases boot from an owner copy of a small synthetic source: the copy's custody is
under test here, not the repository's bytes.
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, suppress
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros import platform_layer as pl, process_containment as pc
from ouroboros.subagent_worktrees import _force_rmtree
from ouroboros.test_environment import RETENTION_MARKER
from tests import test_ui_smoke_playwright as ui
from tests.candidate_checkout import VERSION_PATH, CandidateError, candidate_checkout
from tests.test_candidate_checkout import source as source_fixture


pytestmark = pytest.mark.serial
source = source_fixture
_REPO = Path(__file__).resolve().parents[1]


def _fixture_python() -> str:
    """Use the real Windows interpreter so Popen.pid names the payload."""
    if os.name == "nt":
        return str(getattr(sys, "_base_executable", sys.executable))
    return sys.executable


_TREE_SCRIPT = r"""
import json, os, pathlib, subprocess, sys, time
from ouroboros.platform_layer import subprocess_new_group_kwargs
receipt, entered, ready = map(pathlib.Path, sys.argv[1:4])
entered.write_text("entered", encoding="utf-8")
child = subprocess.Popen(
    [sys.executable, "-c",
     "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8'); time.sleep(90)",
     str(ready)],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    **subprocess_new_group_kwargs(),
)
pending = receipt.with_suffix(".pending")
pending.write_text(json.dumps({"parent": os.getpid(), "child": child.pid}), encoding="utf-8")
pending.replace(receipt)
if sys.argv[4] == "exited":
    sys.exit(0)
time.sleep(90)
"""


def _wait(predicate, detail, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(detail)
        time.sleep(0.05)


def _gone(pid):
    # A reparented POSIX zombie cannot execute, even before init has waited it.
    return not pl.pid_is_alive(pid) or pc.pid_is_zombie(pid)


def _assert_stopped(probe, run):
    _wait(lambda: _gone(run.proc.pid) and _gone(run.child), "fixture-owned tree survived")
    assert run.proc.poll() is not None
    assert ("reap", run.index) in probe.events
    assert ("close", run.index) in probe.events
    assert probe.sentinel.poll() is None, "cleanup killed an unrelated process"


def _owner_copy(source, tmp_path, monkeypatch):
    """The session-style owner copy the fixture's cases boot from, of a source it reads as REPO_ROOT."""
    (source / "VERSION").write_bytes(b"9.9.9\n")
    (source / "web").mkdir()
    (source / "web" / "index.html").write_bytes(b"<!doctype html>\n")
    monkeypatch.setattr(ui, "REPO_ROOT", str(source))
    return candidate_checkout(source, tmp_path / "candidate" / "repo", origin_proof=True)


def _assert_owner_deletes(probe):
    """A proven-clean case gives the copy back to its owner, whose own retirement deletes it."""
    assert probe.owner.unproven == 0 and probe.owner.path.exists()
    probe.retire()
    assert not probe.owner.path.exists()


def _assert_retained_then_discard(retire, checkout):
    """The case's unproven hold kept the owner's copy; the caller has since proved it idle."""
    with pytest.raises(CandidateError, match="CANDIDATE_RETAINED"):
        retire()
    for marked in (checkout, checkout.parent):  # the owner's retirement, and the case's own at once
        assert "never proven gone" in (marked / RETENTION_MARKER).read_text(encoding="utf-8")
    _force_rmtree(checkout.parent)


@pytest.fixture
def fixture_probe(tmp_path, monkeypatch, source):
    """Use a sleeper tree at the fixture's one spawn seam; keep containment real."""
    original_container = pc.ProcessContainer
    owner = ExitStack()
    probe = SimpleNamespace(
        mode="alive", fail_at="", reap_error="", reap_exception=False,
        runs=[], containers=[], generators=[], events=[], prior_gone=[],
        owner=owner.enter_context(_owner_copy(source, tmp_path, monkeypatch)), retire=owner.close,
    )
    sentinel_ready = tmp_path / "sentinel-ready"
    probe.sentinel = subprocess.Popen(
        [_fixture_python(), "-c",
         "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8'); sys.stdin.read()",
         str(sentinel_ready)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        **pl.subprocess_new_group_kwargs(),
    )

    class ObservedContainer(original_container):
        def __init__(self):
            super().__init__()
            self.index = len(probe.containers)
            self.initial_token = dict(self.containment_env())
            probe.containers.append(self)

        def spawn(self, argv, **kwargs):
            assert Path(argv[1]).name == "server.py", argv
            if probe.runs:
                previous = probe.runs[-1]
                probe.prior_gone.append(
                    _gone(previous.proc.pid) and _gone(previous.child)
                )
            run = SimpleNamespace(
                index=self.index, container=self, proc=None, child=0,
                receipt=tmp_path / f"tree-{len(probe.runs)}.json",
                entered=tmp_path / f"entered-{len(probe.runs)}",
                ready=tmp_path / f"ready-{len(probe.runs)}",
            )
            probe.runs.append(run)
            probe.events.append(("spawn", self.index))
            # The sleeper imports the platform layer from this repository; its cwd is the synthetic copy.
            imports = os.pathsep.join(filter(None, (str(_REPO), kwargs["env"].get("PYTHONPATH"))))
            run.proc = super().spawn(
                [_fixture_python(), "-c", _TREE_SCRIPT,
                 str(run.receipt), str(run.entered), str(run.ready), probe.mode],
                **{**kwargs, "env": {**kwargs["env"], "PYTHONPATH": imports}},
            )
            return run.proc

        def reap(self):
            probe.events.append(("reap", self.index))
            actual = super().reap()
            if actual:
                return actual
            if probe.reap_exception:
                raise RuntimeError("injected reap exception")
            return probe.reap_error

        def close(self):
            probe.events.append(("close", self.index))
            return super().close()

    def health(_url):
        # The OLD bare-Popen fixture must fail without accidentally starting a
        # real server. Its direct Popen is refused below; only this container's
        # script may start.
        assert probe.runs, "the UI fixture did not use ProcessContainer.spawn"
        run = probe.runs[-1]
        _wait(lambda: run.receipt.exists() and run.ready.exists(), "sleeper tree did not start")
        info = json.loads(run.receipt.read_text(encoding="utf-8"))
        assert info["parent"] == run.proc.pid
        run.child = info["child"]
        assert not _gone(run.child), "child must be live before teardown"
        if probe.mode == "exited":
            assert run.proc.wait(timeout=15) == 0
        else:
            assert run.proc.poll() is None
        if probe.fail_at == "health":
            raise RuntimeError("injected health failure")

    def supervisor(_url):
        if probe.fail_at == "supervisor":
            raise RuntimeError("injected supervisor failure")

    original_popen = subprocess.Popen

    def no_real_server(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and len(argv) > 1:
            assert Path(str(argv[1])).name != "server.py", "uncontained real server spawn"
        proc = original_popen(argv, *args, **kwargs)
        if isinstance(argv, list) and len(argv) > 2 and argv[2] == _TREE_SCRIPT:
            # Retain custody even if a later native assignment/resume step raises
            # before ProcessContainer.spawn has returned its Popen to the caller.
            probe.runs[-1].proc = proc
        return proc

    def open_fixture():
        gen = ui.direct_server_with_data.__wrapped__(tmp_path / f"fixture-{len(probe.generators)}", probe.owner)
        probe.generators.append(gen)
        return gen, next(gen)

    monkeypatch.setattr(pc, "ProcessContainer", ObservedContainer)
    monkeypatch.setattr(ui, "MockLLMServer", lambda: nullcontext(
        SimpleNamespace(base_url="http://127.0.0.1:9/v1")
    ))
    monkeypatch.setattr(ui, "_assert_served_candidate", lambda *args: None)
    monkeypatch.setattr(ui, "require_candidate_interpreter", lambda *args: None)
    monkeypatch.setattr(ui, "_free_port", lambda: 27991)  # no port is actually bound
    monkeypatch.setattr(ui, "_wait_health", health)
    monkeypatch.setattr(ui, "_wait_supervisor_ready", supervisor)
    monkeypatch.setattr(subprocess, "Popen", no_real_server)
    probe.open = open_fixture
    try:
        _wait(sentinel_ready.exists, "unrelated sentinel did not start")
        yield probe
    finally:
        # This runs only after test assertions. It cleans our own exact children
        # if a regression made the generator's cleanup fail; never sweeps argv,
        # filenames, global process lists, or another test's historical runs.
        try:
            for gen in probe.generators:
                with suppress(RuntimeError, pytest.fail.Exception):
                    gen.close()
            for run in probe.runs:
                if run.receipt.exists():
                    run.child = json.loads(run.receipt.read_text(encoding="utf-8"))["child"]
                try:
                    original_container.reap(run.container)
                finally:
                    original_container.close(run.container)
                    if run.child and not _gone(run.child):
                        pl.force_kill_pid(run.child)
                    if run.proc is not None:
                        if run.proc.poll() is None:
                            run.proc.kill()
                        run.proc.wait(timeout=15)
            with suppress(CandidateError):
                probe.retire()  # A test that already retired the owner leaves nothing to close.
        finally:
            if probe.sentinel.stdin is not None:
                probe.sentinel.stdin.close()
            if probe.sentinel.poll() is None:
                probe.sentinel.terminate()
            probe.sentinel.wait(timeout=15)


@pytest.mark.parametrize("mode", ["exited", "alive"])
def test_ui_fixture_reaps_descendants_after_parent_exit_or_normal_stop(fixture_probe, mode):
    probe = fixture_probe
    probe.mode = mode
    gen, _ = probe.open()
    run = probe.runs[0]
    assert (run.proc.poll() is not None) == (mode == "exited")
    gen.close()
    _assert_stopped(probe, run)
    _assert_owner_deletes(probe)


@pytest.mark.parametrize("stage", ["health", "supervisor"])
def test_ui_fixture_reaps_after_readiness_failure(fixture_probe, stage):
    probe = fixture_probe
    probe.fail_at = stage
    with pytest.raises(RuntimeError, match=f"injected {stage} failure"):
        probe.open()
    assert len(probe.runs) == 1
    _assert_stopped(probe, probe.runs[0])
    _assert_owner_deletes(probe)  # the failed start was reaped clean, so nothing pins the copy


def test_ui_fixture_restart_retires_one_container_before_spawning_another(fixture_probe):
    probe = fixture_probe
    gen, fixture = probe.open()
    first = probe.runs[0]
    fixture["restart_server"]()
    assert len(probe.runs) == len(probe.containers) == 2
    second = probe.runs[1]
    assert first.container.initial_token != second.container.initial_token
    assert probe.prior_gone == [True]
    assert probe.events.index(("close", 0)) < probe.events.index(("spawn", 1))
    _assert_stopped(probe, first)
    assert second.proc.poll() is None and not _gone(second.child)
    gen.close()
    _assert_stopped(probe, second)
    _assert_owner_deletes(probe)


def test_cases_boot_from_the_owners_one_copy_with_their_own_roots_and_overlay(fixture_probe, monkeypatch):
    """Sequential cases reuse one materialization; a VERSION one case rewrote is back for the next."""
    from tests import candidate_checkout as checkouts

    probe = fixture_probe
    monkeypatch.setattr(checkouts, "_copy_candidate", lambda *args: pytest.fail("a case copied the source again"))
    gen, first = probe.open()
    first["stop_server"]()
    first["candidate"].rewrite(VERSION_PATH, b"9.9.10+restarted\n")  # a stopped case restarts into another VERSION
    first["start_server"]()
    gen.close()
    _assert_stopped(probe, probe.runs[1])
    gen, second = probe.open()
    assert second["repo_dir"] == first["repo_dir"] == probe.owner.path
    assert second["data_dir"] != first["data_dir"] and second["candidate"] is not first["candidate"]
    assert second["candidate_version"] == first["candidate_version"] == probe.owner.version_text
    assert (probe.owner.path / VERSION_PATH).read_text(encoding="utf-8").strip() == probe.owner.version_text
    gen.close()
    _assert_stopped(probe, probe.runs[2])
    _assert_owner_deletes(probe)


@pytest.mark.parametrize("raises", [False, True], ids=["returned-error", "raised-error"])
@pytest.mark.parametrize("operation", ["teardown", "restart"])
def test_ui_fixture_surfaces_reap_failure_and_still_closes(fixture_probe, raises, operation):
    probe = fixture_probe
    gen, fixture = probe.open()
    probe.reap_exception = raises
    probe.reap_error = "injected reap failure"
    with pytest.raises((RuntimeError, pytest.fail.Exception), match="injected reap"):
        if operation == "teardown":
            gen.close()
        else:
            fixture["restart_server"]()
    # A failed retirement cannot start a second generation or erase the error.
    assert len(probe.runs) == 1
    _assert_stopped(probe, probe.runs[0])
    # The generator still reaches its outer finally after restart failed.
    # Root may keep or clear active references; either way this cannot spawn.
    with suppress(RuntimeError, pytest.fail.Exception):
        gen.close()
    assert len(probe.runs) == 1
    # Unproven custody keeps the tree in place for every enclosing layer.
    _assert_retained_then_discard(probe.retire, probe.owner.path)


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows Job and suspended-start APIs")
def test_ui_fixture_native_windows_assigns_suspended_root_then_reaps_orphan(fixture_probe, monkeypatch):
    probe = fixture_probe
    probe.mode = "exited"
    native_popen = subprocess.Popen
    native_assign = pl.assign_pid_to_job
    native_resume = pl.resume_process
    ordering = []
    assigned = set()
    observations = []

    def popen(argv, **kwargs):
        if isinstance(argv, list) and len(argv) > 2 and argv[2] == _TREE_SCRIPT:
            assert int(kwargs.get("creationflags", 0)) & 0x4, "root was not CREATE_SUSPENDED"
            ordering.append("popen")
        return native_popen(argv, **kwargs)

    def assign(job, pid):
        observations.append(("before_assign", probe.runs[-1].entered.exists()))
        result = native_assign(job, pid)
        observations.append(("assigned", result))
        if result:
            assigned.add(pid)
        ordering.append("assign")
        return result

    def resume(pid):
        observations.append(("before_resume", pid in assigned, probe.runs[-1].entered.exists()))
        result = native_resume(pid)
        observations.append(("resumed", result))
        ordering.append("resume")
        return result

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(pl, "assign_pid_to_job", assign)
    monkeypatch.setattr(pl, "resume_process", resume)
    gen, _ = probe.open()
    run = probe.runs[0]
    assert ordering == ["popen", "assign", "resume"]
    assert observations == [
        ("before_assign", False), ("assigned", True),
        ("before_resume", True, False), ("resumed", True),
    ]
    assert run.entered.read_text(encoding="utf-8") == "entered"
    assert run.proc.poll() == 0 and not _gone(run.child)
    gen.close()
    _assert_stopped(probe, run)


@pytest.mark.parametrize("reap_result", ["returned-error", "raised-error", "clean"])
def test_ui_fixture_preserves_reap_diagnostic_when_parent_survives(tmp_path, monkeypatch, source, reap_result):
    waits, closed, probed, spawned = [], [], [], []
    owner = ExitStack()
    copy = owner.enter_context(_owner_copy(source, tmp_path, monkeypatch))

    class UnstoppedProcess:
        pid = 12345  # Synthetic identity; the served-candidate probe is stubbed below.

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired(["fixture-root"], timeout)

    class FailedContainer:
        def spawn(self, argv, **kwargs):
            spawned.append((argv[0], kwargs["env"], kwargs["cwd"]))
            return UnstoppedProcess()

        def reap(self):
            if reap_result == "raised-error":
                raise RuntimeError("owned root could not be proven gone")
            return "owned root could not be proven gone" if reap_result == "returned-error" else ""

        def close(self):
            closed.append(True)

    monkeypatch.setattr(pc, "ProcessContainer", FailedContainer)
    monkeypatch.setattr(ui, "MockLLMServer", lambda: nullcontext(SimpleNamespace(base_url="http://127.0.0.1:9")))
    monkeypatch.setattr(ui, "_assert_served_candidate", lambda *args: None)
    monkeypatch.setattr(ui, "require_candidate_interpreter", lambda *args: probed.append(args))
    monkeypatch.setattr(ui, "_free_port", lambda: 27991)
    monkeypatch.setattr(ui, "_wait_health", lambda _: None)
    monkeypatch.setattr(ui, "_wait_supervisor_ready", lambda _: None)
    gen = ui.direct_server_with_data.__wrapped__(tmp_path, copy)
    next(gen)
    # The interpreter validated is the one launched, with the server's own env and cwd.
    assert probed == spawned and spawned[0][0] == ui._fixture_interpreter()
    expected = subprocess.TimeoutExpired if reap_result == "clean" else RuntimeError
    with pytest.raises(expected) as failure:
        gen.close()
    assert closed == [True]
    assert waits == ([10, 5] if reap_result == "clean" else [10])
    if reap_result != "clean":
        assert "owned root could not be proven gone" in str(failure.value)
    # Even a clean reap is not proof while the parent never exited. The synthetic
    # process owns nothing real, so this test may discard what the fixture kept.
    _assert_retained_then_discard(owner.close, copy.path)
