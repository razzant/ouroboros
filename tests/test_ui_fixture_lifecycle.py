"""The shared UI fixture owns each spawned tree through readiness, restart and teardown.

The sleeper payload keeps process containment real without starting a server or
requiring browser binaries; the separately hosted interface's real core and the
widget ESM network server start it at the same seam. Real UI consumers run in
their existing marker lane.
Cases boot from an owner copy of a small synthetic source: the copy's custody is
under test here, not the repository's bytes.
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, suppress
import io
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
from ouroboros.test_evidence import EVIDENCE_ENV
from tests import test_ui_smoke_playwright as ui
from tests.candidate_checkout import SENTINEL_PATH, VERSION_PATH, CandidateError, candidate_checkout
from tests.test_candidate_checkout import source as source_fixture


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
print("fixture tree stderr", file=sys.stderr, flush=True)
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
def outer_evidence(tmp_path, monkeypatch):
    """The run's own durable evidence root, as it stands before ``fixture_probe`` binds the case's.

    ``fixture_probe`` requests this, so it is in force first whatever order a test lists them in; an export that
    still lands here came from a case without an evidence root of its own."""
    monkeypatch.setenv(EVIDENCE_ENV, str(tmp_path / "outer-evidence"))
    return tmp_path / "outer-evidence"


@pytest.fixture
def fixture_probe(outer_evidence, tmp_path, monkeypatch, source):
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
                index=self.index, container=self, proc=None, child=0, tmpdir=Path(kwargs["env"]["TMPDIR"]),
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

    def health(_url, timeout_sec=None):  # separate_ui_support's Core passes its own bound
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

    def supervisor(_url, timeout_sec=None):
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
    # The case's own evidence root replaces outer_evidence's: a synthetic start failure exports its log here,
    # never into the run's.
    monkeypatch.setenv(EVIDENCE_ENV, str(tmp_path / "evidence-owner"))
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
def test_ui_fixture_reaps_after_readiness_failure(fixture_probe, outer_evidence, stage, capsys):
    probe = fixture_probe
    probe.fail_at = stage
    with pytest.raises(RuntimeError, match=f"injected {stage} failure"):
        probe.open()
    assert len(probe.runs) == 1
    _assert_stopped(probe, probe.runs[0])
    _assert_owner_deletes(probe)  # the failed start was reaped clean, so nothing pins the copy
    # The server's own stderr outlives the case's scratch: a manager child's traceback lands nowhere else.
    reported = [line for line in capsys.readouterr().err.splitlines() if line.startswith("SERVER_STARTUP_LOG ")]
    assert "fixture tree stderr" in (Path(reported[0].split(" ", 1)[1]) / "server-stdio.log").read_text(encoding="utf-8")
    assert not outer_evidence.exists(), "a synthetic failure reached the run's own evidence"


def test_supervisor_readiness_failure_names_the_servers_own_error(monkeypatch):
    answer = json.dumps({"supervisor_ready": False, "supervisor_error": "Supervisor init failed: boom"}).encode()
    reads = iter([io.BytesIO(answer)])

    def urlopen(*_args, **_kwargs):
        try:
            return next(reads)
        except StopIteration:
            raise TimeoutError("timed out") from None  # A later slow read must not replace the server's own words.

    monkeypatch.setattr(ui.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="did not become ready: Supervisor init failed: boom"):
        ui._wait_supervisor_ready("http://127.0.0.1:9", timeout_sec=1)


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
    # One temp for the case's every generation; only the run's owner removes it, after the run.
    assert first.tmpdir == second.tmpdir and second.tmpdir.is_dir()
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
    assert second["env"]["TMPDIR"] != first["env"]["TMPDIR"]
    assert second["candidate_version"] == first["candidate_version"] == probe.owner.version_text
    assert (probe.owner.path / VERSION_PATH).read_text(encoding="utf-8").strip() == probe.owner.version_text
    gen.close()
    _assert_stopped(probe, probe.runs[2])
    _assert_owner_deletes(probe)


@pytest.mark.parametrize(("operation", "raises"), [
    pytest.param("teardown", False, id='teardown-returned-error'),
    pytest.param("restart", True, id='restart-raised-error'),
])
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


# separate_ui_support's real core: its own log outlives a failed start, restart or readiness wait.

def _kept_logs(capsys) -> list[Path]:
    """The exports keep_startup_log reported since the last read."""
    return [Path(line.split(" ", 1)[1]) for line in capsys.readouterr().err.splitlines()
            if line.startswith("SERVER_STARTUP_LOG ")]


@pytest.fixture
def core_probe(fixture_probe, monkeypatch, source, tmp_path):
    """separate_ui_support.real_core over the synthetic source, its server the sleeper tree."""
    from tests import candidate_checkout as checkouts
    from tests import separate_ui_support

    monkeypatch.setattr(separate_ui_support, "REPO_ROOT", source)
    monkeypatch.setattr(separate_ui_support, "_fetch_json", lambda *args, **kwargs: {})  # no port is bound
    monkeypatch.setattr(checkouts, "require_candidate_interpreter", lambda *args: None)
    monkeypatch.setattr(checkouts, "assert_served_candidate", lambda *args: None)
    monkeypatch.setenv(EVIDENCE_ENV, str(tmp_path / "evidence-owner"))  # the run's, outliving the case's root
    fixture_probe.real_core = separate_ui_support.real_core
    return fixture_probe


def test_separate_core_first_start_failure_keeps_its_log_once(core_probe, tmp_path, capsys):
    probe = core_probe
    probe.fail_at = "health"
    with pytest.raises(RuntimeError, match="^injected health failure$"):
        with probe.real_core(tmp_path / "core"):
            pytest.fail("a core that never became ready was handed out")
    _assert_stopped(probe, probe.runs[0])
    kept = _kept_logs(capsys)
    assert len(kept) == 1 and kept[0].is_relative_to(tmp_path / "evidence-owner"), kept  # one report, one copy
    log = (tmp_path / "core" / "core.log").read_bytes()
    assert b"fixture tree stderr" in log and (kept[0] / "core.log").read_bytes() == log


@pytest.mark.parametrize("operation", ["restart", "wait_ready"])
def test_separate_core_restart_or_readiness_failure_keeps_its_log(core_probe, tmp_path, capsys, operation):
    probe = core_probe
    with probe.real_core(tmp_path / "core") as core:
        act = getattr(core, operation)
        act()
        assert _kept_logs(capsys) == [] and not (tmp_path / "evidence-owner").exists()  # success exports nothing
        probe.fail_at = "supervisor"
        with pytest.raises(RuntimeError, match="^injected supervisor failure$"):
            act()
        assert core.process.running  # still real_core's own teardown to stop: no second cleanup owner
    for run in probe.runs:
        _assert_stopped(probe, run)
    kept = _kept_logs(capsys)
    log = (tmp_path / "core" / "core.log").read_bytes()
    assert len(kept) == 1 and kept[0].is_relative_to(tmp_path / "evidence-owner"), kept
    assert (kept[0] / "core.log").read_bytes() == log
    # Each generation appends to the one log, so the failed restart's copy carries the earlier ones too.
    assert log.count(b"fixture tree stderr") == len(probe.runs) == (3 if operation == "restart" else 1)


# test_batch4_owner_controls_browser's owner Restart: it waits for the next generation itself, outside start_server.

def test_owner_restart_wait_keeps_the_servers_log_past_its_case(fixture_probe, tmp_path, capsys):
    from tests import test_batch4_owner_controls_browser as batch4

    probe = fixture_probe
    gen, fixture = probe.open()
    data_dir = fixture["data_dir"]
    root, events = data_dir.parent, data_dir / "logs" / "events.jsonl"

    def confirm():  # the owner's confirmation; the next generation then records its boot
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"type": "startup_verification"}) + "\n")

    page = SimpleNamespace(locator=lambda _selector: SimpleNamespace(click=confirm),
                           wait_for_function=lambda *args, **kwargs: None)
    batch4._owner_restart(page, fixture["url"], data_dir)
    assert _kept_logs(capsys) == [] and not (tmp_path / "evidence-owner").exists()  # success exports nothing
    probe.fail_at = "supervisor"
    with pytest.raises(RuntimeError, match="^injected supervisor failure$"):
        batch4._owner_restart(page, fixture["url"], data_dir)
    run = probe.runs[0]
    # The helper stops nothing: the fixture's generator is still the tree's only owner.
    assert run.proc.poll() is None and not _gone(run.child) and ("reap", 0) not in probe.events
    kept = _kept_logs(capsys)
    assert len(kept) == 1 and kept[0].is_relative_to(tmp_path / "evidence-owner"), kept
    log = (root / "server-stdio.log").read_bytes()
    assert b"fixture tree stderr" in log
    gen.close()
    _assert_stopped(probe, run)
    _assert_owner_deletes(probe)
    _force_rmtree(root)  # the case retires with its tree
    assert (kept[0] / "server-stdio.log").read_bytes() == log


# keep_startup_log's exits: a failure copies the log; every exit leaves as it came, the rest without I/O.

class _UnknownExit(BaseException):
    """An exit the keeper has no name for."""


def _stopped_startup(tmp_path):
    root = tmp_path / "case"
    (root / "data" / "logs").mkdir(parents=True)
    (root / "server-stdio.log").write_bytes(b"manager child traceback\n")
    (root / "data" / "logs" / "supervisor.jsonl").write_bytes(b'{"event": "boot"}\n')
    return root


@pytest.mark.parametrize("raised, keeps", [
    (lambda: RuntimeError("readiness wait failed"), True),
    (lambda: pytest.fail.Exception("Timeout (>180.0s) from pytest-timeout."), True),
    (lambda: pytest.xfail.Exception("an expected failure"), False),
    (lambda: pytest.skip.Exception("not here"), False),
    (lambda: pytest.exit.Exception("stop the session"), False),  # pytest.exit: an Exception, still no failure
    (KeyboardInterrupt, False), (SystemExit, False), (GeneratorExit, False),
    (lambda: __import__("asyncio").CancelledError(), False), (_UnknownExit, False),
], ids=["error", "pytest-failure", "xfail", "skip", "pytest-exit", "interrupt", "exit", "generator-close", "cancelled",
        "unknown"])
def test_only_a_failure_keeps_the_startup_log_and_every_exit_leaves_unchanged(tmp_path, monkeypatch, raised, keeps):
    from ouroboros import test_evidence

    root = _stopped_startup(tmp_path)
    monkeypatch.setenv(EVIDENCE_ENV, str(tmp_path / "evidence"))
    io = []
    for name in ("diagnostic_directory", "report_diagnostic", "diagnostics_incomplete", "preview_file"):
        monkeypatch.setattr(test_evidence, name, lambda *args, _real=getattr(test_evidence, name), _name=name, **kw:
                            io.append(_name) or _real(*args, **kw))
    error = raised()

    def wait_for_readiness():
        raise error

    with pytest.raises(type(error)) as caught:
        with test_evidence.keep_startup_log(root, "server-stdio.log"):
            wait_for_readiness()
    # The very exception, still pointing at where it was raised.
    assert caught.value is error and caught.traceback[-1].name == "wait_for_readiness"
    kept = sorted((tmp_path / "evidence").rglob("*")) if keeps else []
    if keeps:
        assert "report_diagnostic" in io
        log = next(path for path in kept if path.name == "server-stdio.log")
        assert log.read_bytes() == b"manager child traceback\n"
        assert (log.parent / "data" / "logs" / "supervisor.jsonl").read_bytes() == b'{"event": "boot"}\n'
        assert (log.parent / test_evidence.PUBLISHED_DIAGNOSTICS_MARKER).is_file(), "its folder, published"
    else:
        assert io == [] and not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("fault", ["unwritable-evidence", "interrupted-copy"])
def test_a_failed_or_interrupted_export_never_replaces_the_failure_or_the_interruption(tmp_path, monkeypatch, fault):
    from ouroboros import test_evidence

    root = _stopped_startup(tmp_path)
    (tmp_path / "evidence").write_text("a file where the run's evidence should be", encoding="utf-8")
    monkeypatch.setenv(EVIDENCE_ENV, str(tmp_path / "evidence"))
    if fault == "interrupted-copy":
        monkeypatch.setenv(EVIDENCE_ENV, str(tmp_path / "writable"))

        def interrupt(*_args):
            raise KeyboardInterrupt

        monkeypatch.setattr(test_evidence.shutil, "copyfile", interrupt)
    failure = pytest.fail.Exception("Timeout (>3.0s) from pytest-timeout.")
    with pytest.raises(KeyboardInterrupt if fault == "interrupted-copy" else type(failure)) as caught:
        with test_evidence.keep_startup_log(root, "server-stdio.log"):
            raise failure
    try:
        if fault == "interrupted-copy":  # the person's stop wins, and the failure it cut short stays attached
            assert caught.value.__context__ is failure and not (root / RETENTION_MARKER).exists()
        else:  # the original goes on; the case's tree holds what could not be exported
            assert caught.value is failure
            assert "test diagnostics incomplete" in (root / RETENTION_MARKER).read_text(encoding="utf-8")
    finally:
        (root / RETENTION_MARKER).unlink(missing_ok=True)  # this test's synthetic hold


_TIMED_OUT_READINESS = '''
import os, signal, time
import pytest
from ouroboros import test_evidence
from tests import test_ui_smoke_playwright as ui
from tests.test_ui_fixture_lifecycle import fixture_probe, outer_evidence, source  # noqa: F401  the real fixture

RUN = os.environ[test_evidence.EVIDENCE_ENV]  # this session's launcher run, before the probe replaces it


@pytest.mark.timeout({timeout}, method="signal", func_only=True)  # 0: no timer, so only the interruption ends it
def test_readiness_wait(fixture_probe, monkeypatch):
    monkeypatch.setenv(test_evidence.EVIDENCE_ENV, RUN)  # a real case's export goes to the run, as here

    def never_ready(*_args, **_kwargs):
        if {scenario!r} == "interrupt":
            os.kill(os.getpid(), signal.SIGINT)
        time.sleep(120)

    monkeypatch.setattr(ui, "_wait_supervisor_ready", never_ready)
    fixture_probe.open()
'''


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGALRM"), reason="pytest-timeout's signal method needs SIGALRM")
@pytest.mark.parametrize("scenario, exit_status", [("signal-timeout", 1), ("interrupt", 2)])
def test_a_real_signal_timeout_in_the_fixtures_readiness_wait_keeps_its_log_past_the_release(
        tmp_path, scenario, exit_status):
    """#2716 at its process boundary: a nested session under scripts/safe_test.py whose real UI fixture (the probe's
    sleeper tree) never becomes ready. pytest-timeout's signal method raises its Failed inside the wait; the log
    reaches the launcher's run and outlives the released scratch while the timeout stays the reported failure.
    A real SIGINT in the same wait leaves without a copy."""
    from ouroboros.test_evidence import diagnostic_directory

    suite = tmp_path / "suite"
    (suite / "tests").mkdir(parents=True)
    (suite / "tests" / "test_readiness.py").write_text(_TIMED_OUT_READINESS.format(
        scenario=scenario, timeout=10 if scenario == "signal-timeout" else 0), encoding="utf-8")
    entry = tmp_path / "run.py"
    entry.write_text(f"import os, sys\nsys.path.insert(0, {str(_REPO)!r})\nos.chdir({str(suite)!r})\n"
                     "import pytest\nraise SystemExit(pytest.main(sys.argv[1:]))\n", encoding="utf-8")
    evidence = diagnostic_directory(tmp_path / "evidence", "startup-timeout-" + scenario)
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(_REPO / "scripts" / "safe_test.py"), "--evidence-dir", str(evidence), "--",
         sys.executable, str(entry), "tests/", "-p", "tests.conftest", "-o", "addopts=", "--rootdir", str(suite),
         "--confcutdir", str(suite), "-q", "--tb=short", "-p", "no:cacheprovider"],
        cwd=_REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    output = done.stdout + done.stderr
    assert done.returncode == exit_status, output
    run = Path(next(line.split(" ", 1)[1] for line in done.stderr.splitlines() if line.startswith("SAFE_TEST_EVIDENCE ")))
    scratch = Path(done.stderr.strip().splitlines()[-1].removeprefix("SAFE_TEST_RELEASED "))
    assert done.stderr.strip().splitlines()[-1].startswith("SAFE_TEST_RELEASED ") and not scratch.exists(), output
    logs = list(run.glob("server-startup/*/server-stdio.log"))
    if scenario == "interrupt":
        assert "KeyboardInterrupt" in output and logs == []
        return
    # The timeout is still what failed, raised inside the fixture's start (where the keeper is).
    assert "Timeout (>10.0s) from pytest-timeout" in output and "in start_server" in output and "1 failed" in output
    assert [log.read_bytes().count(b"fixture tree stderr") for log in logs] == [1]

_READY = {"supervisor_ready": True, "supervisor_error": None}
_FAILED = {"supervisor_ready": False, "supervisor_error": "Supervisor init failed: boom"}


@pytest.fixture
def esm_probe(fixture_probe, monkeypatch, source, tmp_path):
    """network_server over the synthetic source, its server the sleeper tree, on a virtual clock.

    The server's HTTP answers: health once the tree is live, the copy's own proof files, the password
    gate, and for each /api/state read the next entry of ``probe.script`` (its last one repeats)."""
    import httpx

    from tests import test_widget_esm_browser as esm

    probe, root, client = fixture_probe, tmp_path / "esm", httpx.Client
    probe.script, probe.reads, probe.now, probe.signed_in = [], [], 0.0, False

    def answer(request):
        path, repo = request.url.path, root / "repo"
        if path == "/api/health":
            ui._wait_health(str(request.url))  # this probe's: the sleeper tree is up and live
            return httpx.Response(200, json={"runtime_version": (repo / VERSION_PATH).read_text().strip()})
        if path == "/auth/login":
            probe.signed_in = True
        elif not probe.signed_in:
            return httpx.Response(401)
        if path == "/api/state":
            probe.reads.append(probe.now)
            step = probe.script.pop(0) if len(probe.script) > 1 else probe.script[0]
            if step == "timeout":
                raise httpx.ReadTimeout("no answer within the client timeout", request=request)
            if isinstance(step, int):
                return httpx.Response(step, json={"error": "transient"})
            return httpx.Response(200, **({"json": step} if isinstance(step, dict) else {"content": step}))
        if path == "/static/candidate-sentinel.txt":
            return httpx.Response(200, content=(repo / SENTINEL_PATH).read_bytes())
        return httpx.Response(200, json={"ui_tabs": [{}, {}, {}]} if path == "/api/widgets" else {})

    def sleep(seconds):
        probe.now += seconds

    monkeypatch.setattr(esm, "candidate_checkout", lambda _repo, target, **kw: candidate_checkout(source, target, **kw))
    monkeypatch.setattr(esm, "require_candidate_interpreter", lambda *args: None)
    monkeypatch.setattr(esm, "time", SimpleNamespace(monotonic=lambda: probe.now, sleep=sleep))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(answer), **kwargs))
    monkeypatch.setenv(EVIDENCE_ENV, str(tmp_path / "evidence-owner"))
    request = SimpleNamespace(node=SimpleNamespace(callspec=SimpleNamespace(params={"engine": "chromium"})))
    opened = []

    def open_esm():
        opened.append(esm.network_server.__wrapped__(root, request))
        return opened[-1]

    probe.open_esm = open_esm
    yield probe
    for gen in opened:  # a case that stopped early still stops its server, proxy and model stub here
        gen.close()


@pytest.mark.parametrize("script, outcome", [
    ([500, _READY], "ready"), ([503], "deadline"),
    ([_FAILED, _READY], "error"),
], ids=["transient-500", "persistent-503", "supervisor-error"])
def test_widget_esm_server_reads_state_until_it_answers(esm_probe, tmp_path, script, outcome):
    probe = esm_probe
    probe.script[:] = script
    gen = probe.open_esm()
    if outcome == "ready":
        next(gen)
        assert probe.reads == pytest.approx([0.0, 0.2])  # one interval later, the answer
        gen.close()
    elif outcome == "malformed":
        with pytest.raises(json.JSONDecodeError):  # a malformed success is an error, never another try
            next(gen)
        assert probe.reads == [0.0]
    else:
        with pytest.raises(pytest.fail.Exception, match="supervisor not ready") as failure:
            next(gen)
        # The server's own log goes with the verdict and stays with the run's evidence.
        retained = [*(tmp_path / "evidence-owner").rglob("server.log")]
        assert "fixture tree stderr" in str(failure.value) and len(retained) == 1, retained
        assert b"fixture tree stderr" in retained[0].read_bytes()
        if outcome == "error":
            assert "Supervisor init failed: boom" in str(failure.value) and probe.reads == [0.0]
        else:  # the unchanged 45 s deadline, read by read
            assert "HTTP 503" in str(failure.value) and 45 < probe.reads[-1] < 45.21
    _assert_stopped(probe, probe.runs[0])
