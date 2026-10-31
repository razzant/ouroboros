"""Every executable stop must end before recovered capture, after owned teardown."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros import delegate_collection as body, delegate_custody as custody, delegate_custody_current as current
from ouroboros import review_checking as rc
from ouroboros.process_containment import ProcessContainer
from ouroboros.test_environment import isolated_environment, retain_tree
from tests._usage_store_testing import ledger_rows
from tests.test_recovered_capture import ANSWER, EVIDENCE, Engine, prepared

REPO = Path(__file__).resolve().parents[1]
SIGNAL_INGRESS = [pytest.param(name, marks=pytest.mark.skipif(
    sys.platform == "win32", reason="Popen.send_signal requires POSIX signal delivery"))
    for name in ("sigint", "sigterm")]


@pytest.mark.parametrize("existing", [False, True])
def test_capture_gate_publishes_only_complete_json(tmp_path, monkeypatch, existing):
    from tests._server_capture_exit import record

    target = tmp_path / "entered.json"
    if existing:
        target.write_text('{"generation": "old"}', encoding="utf-8")
    writes = []
    write_bytes = Path.write_bytes

    def interrupted_write(path, data):
        write_bytes(path, data[:1])
        writes.append(path)
        if target.exists():
            assert json.loads(target.read_text(encoding="utf-8")) == {"generation": "old"}
        assert target.exists() is existing
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", interrupted_write)
    monkeypatch.setattr(Path, "write_text", lambda path, text, **kw: interrupted_write(path, text.encode("utf-8")))
    record(tmp_path, "entered", announce=False, generation="new")
    assert writes and json.loads(target.read_text(encoding="utf-8"))["generation"] == "new"


def wait_for(predicate, proc, timeout=20):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert proc.poll() is None, f"server exited early: {proc.returncode}"
        assert time.monotonic() < deadline, "subprocess observation timed out"
        time.sleep(0.02)


@contextmanager
def running(scratch, generation, entrypoint="script", ingress="pipe", broken_stream=""):
    gates = scratch / generation
    gates.mkdir()
    env = isolated_environment(scratch, REPO)
    env["OUROBOROS_HOST_SERVICE_PORT"] = "0"
    if broken_stream:
        env["OUROBOROS_TEST_BROKEN_STREAM"] = broken_stream
    if ingress in ("pipe", "eof"):
        env["OUROBOROS_LAUNCHER_STOP_STDIN"] = "1"
    elif ingress.startswith("early-"):
        env["OUROBOROS_TEST_EARLY_SIGNAL"] = ingress[6:].upper()
    container = ProcessContainer(lifeline=True, overload=True)
    proc = None
    try:
        with (gates / "console.log").open("wb") as log:
            proc = container.spawn(
                [sys.executable, "-u", str(REPO / "tests" / "_server_capture_exit.py"),
                 str(scratch), generation, entrypoint], cwd=REPO, env=env,
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
            )
            yield proc, gates
    finally:
        # Failure cleanup cannot satisfy the exit assertion inside the context.
        (gates / "release").touch()
        if proc is not None:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass  # The container owns the bounded fallback, including descendants.
        error = container.reap()
        container.close()
        if proc is not None:
            proc.wait(timeout=5)
        (gates / "reaped.json").write_text(json.dumps({
            "entrypoint": entrypoint, "ingress": ingress,
            "error": error, "returncode": proc.returncode if proc else None,
        }))
        if error:
            retain_tree(scratch, error)
        assert not error, error


def request_stop(proc, ingress):
    if ingress in ("pipe", "eof"):
        if ingress == "pipe":
            proc.stdin.write(b"quit\n")
            proc.stdin.flush()
        proc.stdin.close()
    else:
        proc.send_signal(signal.SIGINT if ingress == "sigint" else signal.SIGTERM)


def stop_before_release(proc, gates, ingress, *, imported=False, expected_code=None):
    assert not (gates / "release").exists()
    started = time.monotonic()
    request_stop(proc, ingress)
    wait_for(lambda: (gates / "lifespan-finished.json").exists(), proc)
    try:
        code = proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        code = None
    observation = {"ingress": ingress, "imported": imported,
                   "exited_before_release": code is not None, "returncode": code,
                   "elapsed_seconds": time.monotonic() - started}
    (gates / "exit.json").write_text(json.dumps(observation))
    print("EXECUTABLE_STOP", json.dumps(observation), flush=True)
    assert not (gates / "release").exists()
    assert (gates / "lifespan-finished.json").is_file()
    assert (gates / "server-finished.json").is_file()
    assert json.loads((gates / "main-socket-closed.json").read_text())["closed"]
    began = json.loads((gates / "stop-began.json").read_text())
    assert any(row.get("stop_requested_at") for row in began["records"])
    stopped = json.loads((gates / "owned-stop.json").read_text())
    assert stopped["outcome"]["state"] == "completed"
    assert stopped["outcome"]["confirmed"] >= 1
    assert stopped["children"] and all(row["code"] is not None for row in stopped["children"])
    assert stopped["maintenance"] and all(row["code"] is not None for row in stopped["maintenance"])
    # Server machinery has its own lifecycle, outside the task-owned stop set.
    # Any ledger remnants must name exactly these proven-exited maintenance children.
    capture_pids = {row["pid"] for row in stopped["maintenance"]}
    assert all(row["purpose"] == "maintenance" and not row["owner_task"]
               and row["host_pid"] in capture_pids for row in stopped["records"])
    expected = expected_code if expected_code is not None else (
        130 if ingress == "sigint" else 143 if ingress == "sigterm" else 0)
    if imported:
        assert code is None, "imported main must leave process exit to its caller"
    else:
        assert code == expected, "executable did not exit while capture remained blocked"


def prepare_capture(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "settings.json").write_text(json.dumps({"OUROBOROS_STARTER_SKILLS": "[]",
        "OUROBOROS_SUBAGENTS": "[]", "OUROBOROS_MAX_WORKERS": 1}))
    entries, areas, sources = {}, {}, {}
    for kind in ("ordinary", "review"):
        entry, areas[kind], sources[kind] = prepared(tmp_path, root, kind)
        detail = Engine().get_run(entry.run_id)
        assert body.observe_run(root, entry, detail)
        assert body.record_terminal_money(root, entry, detail)["settled"]
        entries[kind] = current.run(root, entry.run_id)
    before_money = [r for r in ledger_rows(root) if r.get("kind") == "subscription_session"]
    assert len(before_money) == 2 and len({row["attempt_id"] for row in before_money}) == 2
    return root, entries, areas, sources, before_money


def save_evidence(tmp_path, label):
    out = os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT")
    if out:
        destination = Path(out) / (f"server-capture-exit-{label}-" + str(time.time_ns()))
        for name in ("held", "recovery"):
            if (tmp_path / name).exists():
                shutil.copytree(tmp_path / name, destination / name)
        if (tmp_path / "data" / "logs").exists():
            shutil.copytree(tmp_path / "data" / "logs", destination / "runtime-logs")
        print("EXECUTABLE_STOP_EVIDENCE", destination, flush=True)


@pytest.mark.parametrize("entrypoint", ["script", "console", "cli"])
@pytest.mark.parametrize("ingress", ["pipe", "eof", *SIGNAL_INGRESS])
@pytest.mark.timeout(150)
def test_executable_quit_preserves_owed_capture_for_next_generation(tmp_path, entrypoint, ingress):
    root, entries, areas, sources, before_money = prepare_capture(tmp_path)

    def pending():
        for kind, entry in entries.items():
            row = current.run(root, entry.run_id)
            assert row.terminal_source == entry.terminal_source
            assert body._read_source(root, row, row.terminal_source)["primaryOutput"]["text"] == ANSWER
            assert not row.patch_captured
            assert (areas[kind] / "check.py").read_bytes() == b"# reviewer or delegate output\n"
            assert (areas[kind] / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
        assert not rc._receipt(root, "review", "op-one")

    try:
        with running(tmp_path, "held", entrypoint, ingress) as (proc, gates):
            # The ordinary capture enters in the maintenance child: allow its cold start, as recovery does.
            wait_for(lambda: all((gates / name).exists() for name in (
                "ready.json", "ordinary-entered.json", "review-entered.json")), proc, timeout=45)
            # The server names its own process: on Windows proc.pid can be a venv launcher in front of it.
            server_pid = json.loads((gates / "ready.json").read_text(encoding="utf-8"))["pid"]
            assert json.loads((gates / "ordinary-entered.json").read_text(encoding="utf-8"))["pid"] != server_pid
            assert json.loads((gates / "review-entered.json").read_text(encoding="utf-8"))["pid"] == server_pid
            pending()
            stop_before_release(proc, gates, ingress)
            pending()  # Before any cleanup or capture release: the process really ended.
            assert not (gates / "ordinary-returned.json").exists()
            assert not (gates / "review-returned.json").exists()

        with running(tmp_path, "recovery", entrypoint) as (proc, gates):
            # The new body/child generations reclaim the dead owner's lock and finish owed capture.
            wait_for(lambda: all((gates / name).exists() for name in (
                "ready.json", "ordinary-entered.json", "review-entered.json")), proc, timeout=45)
            pending()
            (gates / "release").touch()
            wait_for(lambda: current.run(root, entries["ordinary"].run_id).patch_captured
                     and rc._receipt(root, "review", "op-one").get("status") == "retained", proc)
            ordinary = current.run(root, entries["ordinary"].run_id)
            cap = custody.delegated_capture_dir(root, "ordinary", custody.capture_key(ordinary))
            wait_for(lambda: (cap / "evidence" / "summary.json").exists(), proc)
            receipt = rc._receipt(root, "review", "op-one")
            for evidence in (cap / "evidence", Path(receipt["evidence_dir"])):
                assert (evidence / "files" / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
            assert receipt["source"]["tree"] == sources["review"]["tree"]
            for patch in (cap / "workspace.patch", Path(receipt["evidence_dir"]) / "delta.patch"):
                content = patch.read_text(encoding="utf-8")
                assert "check.py" in content and "tracked.txt" not in content
            payload = body.retained_payload(root, ordinary)
            assert payload["primary_output"]["text"] == ANSWER
            rows = [r for r in ledger_rows(root) if r.get("kind") == "subscription_session"]
            assert rows == before_money
            (gates / "capture.json").write_text(json.dumps({
                "ordinary": asdict(ordinary), "review": receipt, "accounting": rows,
                "terminal_output": payload,
            }))
            # The live server may release (remove) the disposition lock mid-copy; it is not evidence.
            shutil.copytree(cap, gates / "ordinary-capture", ignore=shutil.ignore_patterns("disposition.lock"))
            shutil.copytree(Path(receipt["evidence_dir"]), gates / "review-capture")
            proc.stdin.write(b"quit\n")
            proc.stdin.flush()
            proc.stdin.close()
            assert proc.wait(timeout=10) == 0
    finally:
        save_evidence(tmp_path, f"{entrypoint}-{ingress}")


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_closed_output_stream_cannot_hold_executable_capture(tmp_path, stream):
    _, _, areas, _, _ = prepare_capture(tmp_path)
    try:
        with running(tmp_path, "held", "console", broken_stream=stream) as (proc, gates):
            wait_for(lambda: all((gates / name).exists() for name in (
                "ready.json", "ordinary-entered.json", "review-entered.json")), proc)
            stop_before_release(proc, gates, "pipe", expected_code=120)
            assert not (gates / "ordinary-returned.json").exists()
            for area in areas.values():
                assert (area / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
    finally:
        save_evidence(tmp_path, "closed-" + stream)


@pytest.mark.parametrize("ingress", ["pipe", SIGNAL_INGRESS[0]])
@pytest.mark.timeout(90)
def test_imported_main_leaves_exit_and_exceptions_to_caller(tmp_path, ingress):
    """Negative control: imported main returns/raises into a caller that controls its own exit."""
    prepare_capture(tmp_path)
    try:
        with running(tmp_path, "held", "imported", ingress) as (proc, gates):
            wait_for(lambda: all((gates / name).exists() for name in (
                "ready.json", "ordinary-entered.json", "review-entered.json")), proc)
            stop_before_release(proc, gates, ingress, imported=True)
            if ingress == "pipe":
                assert json.loads((gates / "main-returned.json").read_text())["result"] == 0
            else:
                assert "KeyboardInterrupt" in (gates / "console.log").read_text()
                assert (gates / "main-interrupted.json").is_file()
            (gates / "release").touch()
            assert proc.wait(timeout=10) == (0 if ingress == "pipe" else -signal.SIGINT)
            assert not (gates / "ordinary-returned.json").exists(), "Stop killed the owned capture child"
    finally:
        save_evidence(tmp_path, f"imported-{ingress}")


@pytest.mark.parametrize("entrypoint", ["script", "console", "cli"])
@pytest.mark.parametrize("ingress", SIGNAL_INGRESS)
def test_signal_before_uvicorn_capture_stops_server(tmp_path, entrypoint, ingress):
    root, entries, areas, sources, before_money = prepare_capture(tmp_path)
    try:
        with running(tmp_path, "held", entrypoint, "early-" + ingress) as (proc, gates):
            wait_for(lambda: (gates / "early-signal.json").exists(), proc)
            assert proc.wait(timeout=12) == (130 if ingress == "sigint" else 143)
            assert not (gates / "release").exists()
            assert (gates / "lifespan-finished.json").exists()
            assert json.loads((gates / "main-socket-closed.json").read_text())["closed"]
            for area in areas.values():
                assert (area / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
    finally:
        save_evidence(tmp_path, f"early-{entrypoint}-{ingress}")


@pytest.mark.parametrize("entrypoint", ["script", "console", "cli"])
@pytest.mark.parametrize("failure,code", [("arguments", 2), ("startup", 1)])
def test_executable_does_not_turn_early_failures_into_success(tmp_path, entrypoint, failure, code):
    env = isolated_environment(tmp_path, REPO)
    env["OUROBOROS_SERVER_PORT"] = "0"
    injection = ""
    args = ["--unknown-option"] if failure == "arguments" else ["--port", "0"]
    if failure == "startup":
        injection = ("from ouroboros import server_entrypoint\n"
                     "def fail(*a, **kw): raise RuntimeError('startup probe failed')\n"
                     "server_entrypoint.find_free_port = fail\n")
    script = ("import sys\nfrom tests._server_capture_exit import invoke\n" + injection +
              "raise SystemExit(invoke(sys.argv[1], sys.argv[2:]))\n")
    container = ProcessContainer(lifeline=True, overload=True)
    proc = None
    try:
        proc = container.spawn([sys.executable, "-c", script, entrypoint, *args], cwd=REPO, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, _ = proc.communicate(timeout=20)
        assert proc.returncode == code, output.decode(errors="replace")
        assert (b"unrecognized arguments" if failure == "arguments" else b"startup probe failed") in output
    finally:
        error = container.reap()
        container.close()
        if proc is not None:
            proc.wait(timeout=5)
        if error:
            retain_tree(tmp_path, error)
        assert not error, error


@pytest.mark.parametrize("signum", [None, signal.SIGINT, signal.SIGTERM] +
                         ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else []))
@pytest.mark.parametrize("panic", [False, True])
@pytest.mark.parametrize("flush_error", [None, BrokenPipeError(), ValueError("closed stream")])
def test_command_boundary_unwinds_before_exit_and_defers_to_panic(monkeypatch, signum, panic, flush_error):
    import server
    from ouroboros import server_control

    calls = []
    signals = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])
    before = {sig: signal.getsignal(sig) for sig in signals}
    import threading
    monkeypatch.setattr(server, "_exit_signalled", threading.Event())
    monkeypatch.setattr(server, "_supervisor_stop", threading.Event())

    def main():
        try:
            if signum is not None:
                signal.getsignal(signum)(signum, None)  # Invoke only this boundary's handler.
                assert server._exit_signalled.is_set() and server._supervisor_stop.is_set()
                calls.append("signal-returned")
            return 2  # A refusal must retain its nonzero status.
        finally:
            calls.append("unwound")

    if flush_error is not None:
        def flush():
            raise flush_error
        monkeypatch.setattr(server.sys, "stdout", SimpleNamespace(flush=flush))
    monkeypatch.setattr(server, "main", main)
    monkeypatch.setattr(server_control, "_restart_stop_requested", panic)
    monkeypatch.setattr(server_control.os, "_exit", lambda code: calls.append(("exit", code)))
    monkeypatch.setattr(server.threading, "Event", lambda: SimpleNamespace(wait=lambda: calls.append("panic-parked")))
    server.executable_main()
    assert calls == (["signal-returned"] if signum is not None else []) + [
        "unwound", "panic-parked" if panic else ("exit", 128 + signum if signum else 2)]
    assert {sig: signal.getsignal(sig) for sig in signals} == before


@pytest.mark.parametrize("error", [RuntimeError("startup probe"), SystemExit(7), KeyboardInterrupt()])
def test_command_boundary_preserves_unrelated_exception_and_restores_handlers(monkeypatch, error):
    import server

    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def fail():
        raise error

    monkeypatch.setattr(server, "main", fail)
    monkeypatch.setattr(server, "exit_after_shutdown", lambda code: pytest.fail(f"unexpected hard exit {code}"))
    with pytest.raises(type(error)) as caught:
        server.executable_main()
    assert caught.value is error
    assert {sig: signal.getsignal(sig) for sig in before} == before
