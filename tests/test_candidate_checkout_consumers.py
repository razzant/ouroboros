"""Alternate browser fixtures must carry the complete snapshot into server custody."""
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from devtools.benchmarks.common import server_runner
from ouroboros import process_containment
from ouroboros.test_environment import RETENTION_MARKER
from tests import candidate_checkout as candidate
from tests import test_owner_wait_integration as owner_wait
from tests import test_ui_repair_owner_intent as repair
from tests.system_e2e import harness
from tests.test_candidate_checkout import git, source as source_fixture


pytestmark = pytest.mark.serial
source = source_fixture


@pytest.fixture(params=["wait_clone", "repair_clone"])
def consumer(request, source, tmp_path, monkeypatch):
    """Invoke each real fixture; only its browser availability probe is replaced."""
    module = owner_wait if request.param == "wait_clone" else repair
    monkeypatch.setattr(module, "__file__", str(source / "tests" / "consumer.py"))
    monkeypatch.setenv("OUROBOROS_RUN_UI_SMOKE", "1")
    playwright = ModuleType("playwright.sync_api")
    playwright.Error = RuntimeError
    playwright.sync_playwright = lambda: nullcontext(SimpleNamespace(
        chromium=SimpleNamespace(launch=lambda: SimpleNamespace(close=lambda: None))))
    package = ModuleType("playwright")
    package.__path__ = []
    package.sync_api = playwright
    monkeypatch.setattr("ouroboros.tools.browser._set_playwright_browsers_path_if_bundled", lambda: None)
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", playwright)
    (source / "VERSION").write_bytes(b"9.9.9\n")
    (source / "web").mkdir()
    (source / "web" / "index.html").write_bytes(b"<!doctype html>\n")
    fixture = getattr(module, request.param).__wrapped__
    return lambda: contextmanager(fixture)(tmp_path / "consumer")


@pytest.fixture
def launch_probe(monkeypatch):
    """Keep the consumer/start/stop flow, replacing OS spawn and HTTP only."""
    probe = SimpleNamespace(runs=[], proofs=[], interpreters=[], fail="")
    monkeypatch.setattr(server_runner, "free_port", lambda: 12345)

    class Process:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if probe.fail == "wait":
                raise RuntimeError("parent wait failed")
            assert self.returncode is not None
            return self.returncode

    class Container:
        def __init__(self):
            self.proc = None
            self.reaped = self.closed = False
            probe.runs.append(self)

        def spawn(self, argv, **kwargs):
            self.argv, self.kwargs = argv, kwargs
            self.proc = Process()
            return self.proc

        def reap(self):
            self.reaped = True
            if probe.fail == "reap_raise":
                raise RuntimeError("unreadable process census")
            return "descendant still alive" if probe.fail == "reap_error" else ""

        def close(self):
            self.closed = True
            if probe.fail == "close":
                raise RuntimeError("container close failed")

    def stop(server):
        if server.proc is not None:
            server.proc.returncode = 0

    def ready(server, timeout):
        if probe.fail == "ready":
            raise RuntimeError("readiness failed")

    monkeypatch.setattr(process_containment, "ProcessContainer", Container)
    monkeypatch.setattr(harness.IsolatedServer, "stop", stop)
    monkeypatch.setattr(harness.KeylessIsolatedServer, "_wait_ready", ready)
    monkeypatch.setattr(harness, "require_candidate_interpreter",
                        lambda *args: probe.interpreters.append(args))
    monkeypatch.setattr(harness, "assert_served_candidate",
                        lambda *args: probe.proofs.append(args))
    return probe


def start(checkout, tmp_path):
    return harness.start_server(checkout, tmp_path / "runtime", harness.keyless_settings(
        SimpleNamespace(base_url="http://127.0.0.1:12346")))


def test_alternate_consumers_deliver_all_dirty_inputs_and_preserve_source(
    consumer, source, tmp_path, launch_probe,
):
    (source / "edited").write_bytes(b"staged\n")
    (source / "staged-new").write_bytes(b"staged only\x00\xff")
    (source / "staged-then-removed").write_bytes(b"index only\n")
    git(source, "add", "edited", "staged-new", "staged-then-removed")
    (source / "edited").write_bytes(b"unstaged\r\n\x80\xff\x00")
    (source / "staged-then-removed").unlink()
    git(source, "mv", "deleted", "renamed")
    git(source, "rm", "recreated")
    (source / "recreated").write_bytes(b"untracked after staged deletion")
    name = "new file\nwith tabs\t" if os.name != "nt" else "new file"
    (source / "web" / name).write_bytes(b"new asset\x00\xfe")
    (source / "new_backend.py").write_bytes(b"VALUE = 'untracked'\n")
    (source / "new_backend.py").chmod(0o755)
    (source / "ignored").mkdir()
    (source / "ignored" / "artifact").write_bytes(b"not a candidate input")
    before = candidate.observe_candidate(source)

    with consumer() as checkout:
        assert checkout.state == before
        server = start(checkout, tmp_path)
        try:
            for _ in range(2):
                assert checkout.unproven == 1
                run = launch_probe.runs[-1]
                served = Path(run.kwargs["cwd"])
                assert served == checkout.path == server.clone
                assert (served / ".git" / "index").read_bytes() == before.index
                for path, value in before.files.items():
                    if path in checkout.overlay:
                        continue
                    if value is None:
                        assert not (served / path).exists(), path
                    else:
                        assert (served / path).read_bytes() == value[0], path
                        assert (served / path).stat().st_mode & 0o777 == value[1] & 0o777
                assert git(served, "show", ":edited") == b"staged\n"
                assert git(served, "show", ":staged-then-removed") == b"index only\n"
                assert not (served / "ignored").exists()
                assert launch_probe.proofs[-1][-1] is checkout
                assert launch_probe.interpreters[-1][0] == run.argv[0]
                server.stop()
                assert checkout.unproven == 0 and run.reaped and run.closed
                if len(launch_probe.runs) == 1:
                    server.start()
        finally:
            server.stop()
    assert len(launch_probe.runs) == 2
    assert not checkout.path.exists()
    assert candidate.observe_candidate(source) == before


@pytest.mark.parametrize("phase", ["capture", "source", "checkout"])
def test_alternate_consumers_refuse_concurrent_drift(consumer, source, monkeypatch, phase):
    if phase == "capture":
        copy = candidate._copy_candidate

        def racing_copy(*args):
            copy(*args)
            (source / "late-new").write_bytes(b"arrived during capture")

        monkeypatch.setattr(candidate, "_copy_candidate", racing_copy)
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED"):
        with consumer() as checkout:
            assert phase != "capture", "a mixed source was admitted"
            root = source if phase == "source" else checkout.path
            (root / "new_backend.py").write_bytes(b"drift after capture")


def test_alternate_consumers_refuse_unsupported_inputs(consumer, source):
    git(source, "update-index", "--assume-unchanged", "edited")
    before = candidate._git(source, "status", "--porcelain=v1", "-z")
    index = (source / ".git" / "index").read_bytes()
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_UNSUPPORTED"):
        with consumer():
            pytest.fail("unsupported candidate reached the browser consumer")
    assert candidate._git(source, "status", "--porcelain=v1", "-z") == before
    assert (source / ".git" / "index").read_bytes() == index


def test_alternate_checkout_drift_refuses_restart_before_spawning(consumer, tmp_path, launch_probe):
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED"):
        with consumer() as checkout:
            server = start(checkout, tmp_path)
            server.stop()
            (checkout.path / "edited").write_bytes(b"unexpected checkout edit")
            with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED"):
                server.start()
            assert len(launch_probe.runs) == 1 and checkout.unproven == 0


@pytest.mark.parametrize("failure", ["reap_error", "reap_raise", "close", "wait"])
def test_alternate_consumers_retain_unproven_tree_and_cannot_clear_failed_pin(
    consumer, source, tmp_path, launch_probe, failure,
):
    from tests._shared import retire_synthetic_retention

    before = candidate.observe_candidate(source)
    try:
        with pytest.raises(candidate.CandidateError, match="CANDIDATE_RETAINED"):
            with consumer() as checkout:
                server = start(checkout, tmp_path)
                launch_probe.fail = failure
                with pytest.raises(RuntimeError):
                    server.stop()
                assert server.proc.poll() == 0, "root exit alone must not release the snapshot"
                assert checkout.unproven == 1
                with pytest.raises(candidate.CandidateError, match="CANDIDATE_RETAINED"):
                    server.stop()
                with pytest.raises(candidate.CandidateError, match="CANDIDATE_CUSTODY"):
                    server.start()
        assert (checkout.path / RETENTION_MARKER).is_file()
        assert (server.data_root / RETENTION_MARKER).is_file()
        assert launch_probe.runs[0].reaped and launch_probe.runs[0].closed
        assert candidate.observe_candidate(source) == before
    finally:
        # launch_probe spawns no OS process; these are its injected failures' markers.
        retire_synthetic_retention(checkout.path, "1 process tree(s) started from this checkout were never proven gone")
        retire_synthetic_retention(server.data_root, "CANDIDATE_RETAINED: " + {
            "reap_error": "CANDIDATE_RETAINED: process cleanup failed: descendant still alive",
            "reap_raise": "unreadable process census", "close": "container close failed",
            "wait": "parent wait failed"}[failure])


def test_alternate_readiness_failure_reaps_before_releasing_snapshot(
    consumer, source, tmp_path, launch_probe,
):
    before = candidate.observe_candidate(source)
    launch_probe.fail = "ready"
    with pytest.raises(RuntimeError, match="readiness failed"):
        with consumer() as checkout:
            start(checkout, tmp_path)
    assert checkout.unproven == 0
    assert launch_probe.runs[0].reaped and launch_probe.runs[0].closed
    assert not checkout.path.exists()
    assert candidate.observe_candidate(source) == before


def test_plain_keyless_clone_keeps_mutable_head_scenario_contract(tmp_path, monkeypatch, launch_probe):
    calls = []
    monkeypatch.setattr(harness.IsolatedServer, "start", lambda self, timeout: calls.append(timeout) or self)
    monkeypatch.setattr(harness.IsolatedServer, "stop", lambda self: calls.append("stopped"))
    server = harness.KeylessIsolatedServer(tmp_path / "clone", tmp_path / "data", tmp_path / "settings.json")
    assert server.start(42) is server
    server.stop()
    assert calls == [42, "stopped"]
    assert not launch_probe.runs and server.candidate is None
