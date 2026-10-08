"""Per-process logging bootstrap and the error records it makes complete.

Process-wide state (root handlers, ``threading.excepthook``, ``sys.excepthook``) is
exercised in fresh interpreters, never in the pytest process, so no other test
inherits a handler or a hook from these checks.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import pathlib
import subprocess
import sys
import textwrap
import traceback
from types import SimpleNamespace

import pytest

from supervisor import worker_process as _worker_process
from tests.test_terminal_file_boundary import worker as _worker_loop

worker = _worker_loop  # noqa: F811 - real worker loop fixture re-export
REPO = pathlib.Path(__file__).resolve().parents[1]
_REAL_LOG_WORKER_CRASH = _worker_process._log_worker_crash


def _run(code: str, tmp_path: pathlib.Path, *, script: bool = False) -> subprocess.CompletedProcess:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(data), "PYTHONPATH": str(REPO)}
    env.pop("PYTEST_CURRENT_TEST", None)
    source = textwrap.dedent(code)
    if script:
        path = tmp_path / "probe.py"
        path.write_text(source, encoding="utf-8")
        argv = [sys.executable, str(path)]
    else:
        argv = [sys.executable, "-c", source]
    return subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=180)


@pytest.mark.serial
def test_server_bootstrap_writes_server_log_through_redacting_handlers(tmp_path):
    logs = tmp_path / "logs"
    completed = _run(f"""
        import json, logging, pathlib, threading
        from ouroboros.observability import SecretRedactingLogFilter
        from ouroboros.process_logging import configure_process_logging
        configure_process_logging(drive_logs=pathlib.Path({str(logs)!r}))
        configure_process_logging(drive_logs=pathlib.Path({str(logs)!r}))  # idempotent
        root = logging.getLogger()
        logging.getLogger("probe").info("Authorization: Bearer sk-or-v1-{'a' * 64}")
        for handler in root.handlers:
            handler.flush()
        print("STATE", json.dumps({{
            "handlers": sorted(type(h).__name__ for h in root.handlers),
            "filtered": all(any(type(f) is SecretRedactingLogFilter for f in h.filters)
                            for h in root.handlers),
            "httpx": logging.getLogger("httpx").level,
            "thread_hook": getattr(threading.excepthook, "_ouroboros_hook", False),
        }}))
    """, tmp_path)
    assert completed.returncode == 0, completed.stderr[-2000:]
    state = json.loads(next(line for line in completed.stdout.splitlines() if line.startswith("STATE"))[6:])
    assert state["handlers"] == ["RotatingFileHandler", "StreamHandler"]
    assert state["filtered"] is True and state["thread_hook"] is True
    assert state["httpx"] >= logging.WARNING
    server_log = (logs / "server.log").read_text(encoding="utf-8")
    assert "probe" in server_log and "a" * 64 not in server_log


@pytest.mark.serial
def test_httpx_logger_quieted(tmp_path):
    """httpx logs each request URL at INFO; the bootstrap keeps it at WARNING."""
    completed = _run("""
        import logging
        from ouroboros.process_logging import configure_process_logging
        configure_process_logging(drive_logs=None)
        print("LEVELS", logging.getLogger("httpx").level, logging.getLogger("httpcore").level)
    """, tmp_path)
    assert completed.returncode == 0, completed.stderr[-2000:]
    levels = next(line for line in completed.stdout.splitlines() if line.startswith("LEVELS")).split()[1:]
    assert all(int(level) >= logging.WARNING for level in levels)


@pytest.mark.serial
def test_uncaught_thread_exception_is_one_log_record_not_a_raw_print(tmp_path):
    logs = tmp_path / "logs"
    completed = _run(f"""
        import logging, pathlib, threading
        from ouroboros.process_logging import configure_process_logging
        configure_process_logging(drive_logs=pathlib.Path({str(logs)!r}))

        def boom():
            raise ValueError("thread-boom")

        def leave():
            raise SystemExit(0)

        for target, name in ((boom, "probe-thread"), (leave, "leaving-thread")):
            thread = threading.Thread(target=target, name=name)
            thread.start()
            thread.join()
        for handler in logging.getLogger().handlers:
            handler.flush()
    """, tmp_path)
    assert completed.returncode == 0, completed.stderr[-2000:]
    server_log = (logs / "server.log").read_text(encoding="utf-8")
    assert "Uncaught exception in thread probe-thread" in server_log
    assert "ValueError: thread-boom" in server_log
    assert "leaving-thread" not in server_log
    # The interpreter's raw print ("Exception in thread …") is replaced, not doubled.
    assert "Exception in thread probe-thread" not in completed.stderr
    assert completed.stderr.count("ValueError: thread-boom") == 1


@pytest.mark.serial
def test_uncaught_main_thread_exception_is_logged_and_still_exits_nonzero(tmp_path):
    completed = _run("""
        from ouroboros.process_logging import configure_process_logging
        configure_process_logging(drive_logs=None)
        raise RuntimeError("main-boom")
    """, tmp_path)
    assert completed.returncode == 1
    assert "Uncaught exception" in completed.stderr
    assert completed.stderr.count("RuntimeError: main-boom") == 1


@pytest.mark.serial
def test_a_main_thread_hook_installed_earlier_still_runs_after_the_record(tmp_path):
    completed = _run("""
        import sys
        def recorder(exc_type, exc_value, exc_traceback):
            print("CHAINED", exc_type.__name__, flush=True)
        sys.excepthook = recorder
        from ouroboros.process_logging import configure_process_logging
        configure_process_logging(drive_logs=None)
        raise RuntimeError("main-chained")
    """, tmp_path)
    assert completed.returncode == 1
    assert "CHAINED RuntimeError" in completed.stdout
    assert completed.stderr.count("RuntimeError: main-chained") == 1


@pytest.mark.serial
def test_a_hook_installed_earlier_still_runs_after_the_record(tmp_path):
    completed = _run("""
        import threading
        calls = []
        threading.excepthook = lambda args: calls.append(args.exc_type.__name__)
        from ouroboros.process_logging import configure_process_logging
        configure_process_logging(drive_logs=None)

        def boom():
            raise ValueError("chained")

        thread = threading.Thread(target=boom, name="chained-thread")
        thread.start()
        thread.join()
        print("CALLS", calls)
    """, tmp_path)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "CALLS ['ValueError']" in completed.stdout
    assert "Uncaught exception in thread chained-thread" in completed.stderr


@pytest.mark.serial
def test_without_any_handler_the_interpreter_print_stays_in_charge(tmp_path):
    completed = _run("""
        import threading
        from ouroboros.process_logging import install_exception_hooks
        install_exception_hooks()

        def boom():
            raise ValueError("unconfigured")

        thread = threading.Thread(target=boom, name="bare-thread")
        thread.start()
        thread.join()
    """, tmp_path)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "Exception in thread bare-thread" in completed.stderr
    assert "ValueError: unconfigured" in completed.stderr


@pytest.mark.serial
@pytest.mark.parametrize("start_method", [
    "spawn",
    pytest.param("forkserver", marks=pytest.mark.skipif(sys.platform == "win32", reason="forkserver is POSIX-only")),
])
def test_worker_logging_is_configured_only_in_a_real_pool_child(tmp_path, start_method):
    completed = _run(f"""
        import logging, multiprocessing
        from supervisor.worker_process import _configure_worker_logging


        def probe(queue):
            _configure_worker_logging()
            queue.put(sorted(type(h).__name__ for h in logging.getLogger().handlers))


        if __name__ == "__main__":
            _configure_worker_logging()  # not a pool child: the host process owns logging
            print("PARENT", sorted(type(h).__name__ for h in logging.getLogger().handlers))
            context = multiprocessing.get_context({start_method!r})
            queue = context.Queue()
            child = context.Process(target=probe, args=(queue,))
            child.start()
            print("CHILD", queue.get(timeout=150))
            child.join(60)
    """, tmp_path, script=True)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "PARENT []" in completed.stdout
    assert "CHILD ['StreamHandler']" in completed.stdout
    assert not (tmp_path / "data" / "logs" / "server.log").exists()


def test_json_exception_records_only_unexpected_failures(caplog):
    from ouroboros.gateway._helpers import json_exception

    caplog.set_level(logging.ERROR, logger="ouroboros.gateway._helpers")
    try:
        raise RuntimeError("index failed")
    except RuntimeError as exc:
        response = json_exception(exc, context="api index failure")
    assert response.status_code == 500
    assert json.loads(response.body) == {"error": "index failed"}
    records = [r for r in caplog.records if r.name == "ouroboros.gateway._helpers"]
    assert len(records) == 1
    assert records[0].getMessage() == "api index failure" and records[0].exc_info[0] is RuntimeError

    caplog.clear()
    response = json_exception(ValueError("bad input"), 400)
    assert response.status_code == 400
    assert not [r for r in caplog.records if r.name == "ouroboros.gateway._helpers"]


def test_worker_crash_goes_through_the_shared_appender_and_the_log(tmp_path, caplog):
    from supervisor import worker_process

    caplog.set_level(logging.ERROR, logger="supervisor.worker_process")
    try:
        raise RuntimeError("agent build failed")
    except RuntimeError as exc:
        worker_process._log_worker_crash(3, tmp_path, "make_agent", exc, traceback.format_exc())
    ledger = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in ledger.splitlines()]
    assert rows[-1]["type"] == "worker_crash" and rows[-1]["phase"] == "make_agent"
    assert rows[-1]["worker_id"] == 3 and rows[-1]["error"] == "RuntimeError('agent build failed')"
    assert "agent build failed" in rows[-1]["traceback"]
    record = next(r for r in caplog.records if r.name == "supervisor.worker_process")
    assert record.exc_info[0] is RuntimeError


def test_worker_crash_falls_back_to_one_stderr_line_when_the_append_fails(tmp_path, monkeypatch, capsys):
    from ouroboros import utils
    from supervisor import worker_process

    monkeypatch.setattr(utils, "append_jsonl", lambda *_args, **_kwargs: False)
    worker_process._log_worker_crash(1, tmp_path, "extension_reload", None, "Traceback: fake")
    lines = [line for line in capsys.readouterr().err.splitlines() if '"type": "worker_crash"' in line]
    assert len(lines) == 1 and json.loads(lines[0])["phase"] == "extension_reload"
    assert not (tmp_path / "logs" / "supervisor.jsonl").exists()


def test_task_exception_is_logged_with_its_stack(tmp_path, caplog):
    from ouroboros.agent import _task_exception_terminal

    caplog.set_level(logging.ERROR, logger="ouroboros.agent")
    try:
        raise RuntimeError("tool loop exploded")
    except RuntimeError as error:
        _task_exception_terminal(SimpleNamespace(drive_root=tmp_path), {"id": "t-log"}, error, tmp_path)
    record = next(r for r in caplog.records if r.name == "ouroboros.agent" and "t-log" in r.getMessage())
    assert record.exc_info[0] is RuntimeError
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row.get("type") == "task_error" and row.get("task_id") == "t-log" for row in rows)


def test_every_uvicorn_config_in_the_server_leaves_logging_to_the_root_handlers():
    """uvicorn's own logging graph gives its loggers a private stderr handler and
    ``propagate: False``, which would keep "Exception in ASGI application" out of
    server.log and away from every operator handler."""
    tree = ast.parse((REPO / "server.py").read_text(encoding="utf-8"))
    configs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Config"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "uvicorn"
    ]
    assert configs
    for call in configs:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords.get("log_config"), ast.Constant) and keywords["log_config"].value is None


def test_the_launcher_routes_its_uncaught_exceptions_through_the_shared_hooks():
    """The frozen launcher keeps its own launcher.log handlers and shares the
    uncaught-exception hooks, called after its guarded redaction filter loop, and
    the output copier, imported unguarded because the module loads only the stdlib."""
    tree = ast.parse((REPO / "launcher.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "install_exception_hooks"]
    assert len(calls) == 1
    imports = {alias.name: node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == "ouroboros.process_logging" for alias in node.names}
    assert set(imports) == {"install_exception_hooks", "copy_capped_output"}
    start_agent = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "start_agent")
    assert imports["copy_capped_output"] in start_agent.body  # Unguarded, before the server is spawned.


@pytest.mark.serial
def test_the_module_loads_only_the_standard_library():
    """The launcher drains and records a server whose own imports are broken (the crash
    output agent_stdout.log exists for), so nothing project-owned loads with the copier."""
    tree = ast.parse((REPO / "ouroboros" / "process_logging.py").read_text(encoding="utf-8"))
    roots = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    roots |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert roots - {"__future__"} <= set(sys.stdlib_module_names)
    completed = subprocess.run(
        [sys.executable, "-c", "import sys, ouroboros.process_logging; print('ouroboros.observability' in sys.modules)"],
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True, timeout=120)
    assert completed.stdout.strip() == "False", completed.stderr[-2000:]


@pytest.mark.serial
@pytest.mark.parametrize("phase", ["init_baseline", "extension_reload"])
def test_a_nonfatal_worker_startup_failure_keeps_its_exception(worker, monkeypatch, caplog, phase):
    """Both non-fatal startup phases of the real worker loop hand the caught exception
    to the crash record, so an attached error handler receives type, value and frames."""
    import ouroboros.config as config
    import ouroboros.extension_loader as extensions
    from supervisor import worker_process

    worker.task["type"] = "shutdown"
    monkeypatch.setattr(worker_process, "_log_worker_crash", _REAL_LOG_WORKER_CRASH)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic startup failure")

    if phase == "init_baseline":
        monkeypatch.setattr(config, "initialize_runtime_mode_baseline", fail)
    else:
        monkeypatch.setattr(extensions, "reload_all", fail)
    caplog.set_level(logging.ERROR, logger="supervisor.worker_process")
    worker.run()
    records = [r for r in caplog.records if r.name == "supervisor.worker_process" and phase in r.getMessage()]
    assert len(records) == 1 and records[0].exc_info is not None
    assert records[0].exc_info[0] is RuntimeError
    ledger = (worker.root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in ledger.splitlines()]
    assert any(row.get("type") == "worker_crash" and row.get("phase") == phase for row in rows)


@pytest.mark.parametrize("function_name", ["_apply_smart_update_fenced", "_apply_replace_recovery_fenced"])
def test_a_failed_managed_update_is_recorded_once(monkeypatch, caplog, function_name):
    """The update catch records the failure itself; its json_exception call names the
    status, so the same exception is not recorded a second time."""
    from ouroboros.gateway import control
    from supervisor import update_merge

    plan = {"available": True, "kind": "clean", "base_sha": "a" * 40, "target_sha": "b" * 40}
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_kwargs: plan)
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _value: None)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    monkeypatch.setattr(control.update_progress, "begin", lambda: None)
    monkeypatch.setattr(control.update_progress, "_finish", lambda *_args: None)
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    error = RuntimeError("synthetic update failure")

    def fail(_reason):
        raise error

    monkeypatch.setattr(control, "_quiesce_repo_writers", fail)
    caplog.set_level(logging.WARNING)
    response = getattr(control, function_name)(None, expected_base_sha="a" * 40, expected_target_sha="b" * 40)
    assert response.status_code == 500
    records = [r for r in caplog.records if r.exc_info and r.exc_info[1] is error]
    assert len(records) == 1 and records[0].levelno == logging.ERROR


def test_the_bootstrap_is_inert_inside_the_pytest_process(tmp_path):
    """Restart and shutdown tests run ``server.main()`` in the shared xdist worker; the conftest
    keeps the bootstrap from leaving handlers, a root level or hooks behind for later tests."""
    import threading

    from ouroboros.process_logging import configure_process_logging

    root = logging.getLogger()
    before = (list(root.handlers), root.level, threading.excepthook, sys.excepthook)
    configure_process_logging(drive_logs=tmp_path / "logs")
    assert (list(root.handlers), root.level, threading.excepthook, sys.excepthook) == before
    assert not (tmp_path / "logs").exists()
