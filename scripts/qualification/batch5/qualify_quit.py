"""Operator-only real-process source shutdown qualification; no paid model calls.

Drive the production launcher start/stop and Background.request_quit entry point.
This is not native menu-click or packaged-app qualification. Every run uses a new
app/data/home root, 24 real workers, initialized owner state and a loopback model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def rows(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except ValueError:
            continue  # a currently-being-appended final row is retried on the next read
    return result


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def resource_facts(server_pid=None):
    if sys.platform != "win32":
        return {"cpu_count": os.cpu_count()}
    import psutil
    memory = psutil.virtual_memory()
    facts = {"cpu_count": os.cpu_count(), "memory_total_bytes": memory.total,
             "memory_available_bytes": memory.available}
    if server_pid is not None:
        processes = [psutil.Process(server_pid)]
        processes.extend(processes[0].children(recursive=True))
        facts["server_tree_rss_bytes"] = sum(process.memory_info().rss for process in processes)
        facts["server_tree_processes"] = len(processes)
    return facts


def isolated_env(source, run_root, short_temp, parent_env, windows):
    keep = {"PATH", "USER", "LOGNAME", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL"}
    env = {key: value for key, value in parent_env.items() if key.upper() in keep}
    env.update({"HOME": str(run_root / "home"), "USERPROFILE": str(run_root / "home"),
        "APPDATA": str(run_root / "home/AppData/Roaming"),
        "LOCALAPPDATA": str(run_root / "home/AppData/Local"),
        "TMPDIR": str(short_temp), "TMP": str(short_temp), "TEMP": str(short_temp),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(source),
        "PYTHONUTF8": "1", "OUROBOROS_PYTEST_ACTIVE": "1",
        "OUROBOROS_APP_ROOT": str(run_root / "app"), "OUROBOROS_DATA_DIR": str(run_root / "data"),
        "OUROBOROS_SETTINGS_PATH": str(run_root / "data/settings.json"),
        "OUROBOROS_REPO_DIR": str(source), "OUROBOROS_SERVER_HOST": "127.0.0.1",
        "OUROBOROS_SERVER_PORT": str(free_port()), "OUROBOROS_HOST_SERVICE_PORT": str(free_port()),
        "OUROBOROS_MANAGED_BY_LAUNCHER": "1"})
    if windows:
        # Windows environment names are case-insensitive; retain native OS plumbing.
        assert any(key.upper() == "SYSTEMROOT" for key in env), "Windows SystemRoot is required"
    return env


def wait_until(fn, timeout, label):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = fn()
            if last:
                return last
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"timeout waiting for {label}; last={last!r}")


def api(port, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}", data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


class HeldModel:
    """An HTTP call stays physically in flight until the qualification cleans up."""
    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_json({"data": [{"id": "shutdown-probe", "max_model_len": 1000000}]})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                outer.calls.append({"path": self.path, "model": body.get("model"),
                                    "received_monotonic": time.monotonic()})
                outer.release.wait(600)
                try:
                    self.send_json({"error": {"message": "isolated qualification complete"}}, status=503)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def send_json(self, payload, status=200):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


def child(args):
    root = args.run_root.resolve()
    source = args.source.resolve()
    data_root = root / "data"
    sys.path.insert(0, str(source))
    from ouroboros.provider_models import (
        ACTIVE_MODEL_SETTING_KEYS, ALL_PROVIDER_CREDENTIAL_KEYS, LEGACY_MODEL_SETTING_KEYS,
    )
    model = HeldModel()
    settings = {key: "" for key in (*ACTIVE_MODEL_SETTING_KEYS, *LEGACY_MODEL_SETTING_KEYS,
                                    *ALL_PROVIDER_CREDENTIAL_KEYS)}
    settings.update({
        "OPENAI_COMPATIBLE_API_KEY": "local-fixture-only",
        "OPENAI_COMPATIBLE_BASE_URL": model.start(),
        "OUROBOROS_MODEL": "openai-compatible::shutdown-probe",
        "OUROBOROS_MODEL_LIGHT": "openai-compatible::shutdown-probe",
        "OUROBOROS_MAX_WORKERS": args.workers,
        "OUROBOROS_SAFETY_MODE": "off",
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
        "OUROBOROS_RUNTIME_MODE": "light",
        "OUROBOROS_TASK_REVIEW_MODE": "off",
        "OUROBOROS_POST_TASK_EVOLUTION": "false",
        "OUROBOROS_DESKTOP_KEEP_RUNNING": "false",
        "TOTAL_BUDGET": 10.0,
        "OUROBOROS_PER_TASK_COST_USD": 10.0,
    })
    write_json(data_root / "settings.json", settings)
    from devtools.benchmarks.common.server_runner import seed_owner_state
    seed_owner_state(data_root)
    import launcher
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.launcher_background import Background
    from websockets.sync.client import connect

    port = int(os.environ["OUROBOROS_SERVER_PORT"])
    proc = None
    ws = None
    request = None
    worker_pids = []
    task_ids = []
    process_cohort = None
    monitor_stop = threading.Event()
    monitor = None
    report = {"platform": platform.platform(), "python": sys.version,
              "interpreter": sys.executable, "native_windows": sys.platform == "win32",
              "startup_resources": resource_facts(),
              "source": str(source), "run_root": str(root), "workers_requested": args.workers,
              "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
              "source_tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=source, text=True).strip(),
              "entry": "Background.request_quit -> callback -> launcher.stop_agent",
              "final_orphan_port_sweep": "NOT_RUN: host-global port sweep excluded by fixture",
              "native_menu_click": "NOT_RUN", "packaged_app": "NOT_RUN"}
    try:
        if args.require_windows and sys.platform != "win32":
            raise RuntimeError("native Windows qualification cannot run on this platform")
        if args.expected_sha and report["source_sha"] != args.expected_sha:
            raise RuntimeError("source SHA differs from the workflow pin")
        if sys.platform == "win32" and Path(sys.executable).resolve() != Path(sys._base_executable).resolve():
            raise RuntimeError("use the setup-python base interpreter, not a Windows venv PID redirector")
        proc = launcher.start_agent(port)
        report["server_pid"] = proc.pid
        report["native_stop_transport"] = "launcher_private_stdin" if sys.platform == "win32" else "SIGTERM"
        report["launcher_job_created"] = launcher._agent_job is not None
        if sys.platform == "win32" and (proc.stdin is None or launcher._agent_job is None):
            raise RuntimeError("native launcher did not retain both the Job and private Quit pipe")
        write_json(root / "progress.json", {**report, "stage": "server_started"})
        wait_until(lambda: api(port, "/api/state").get("supervisor_ready") is True,
                   args.startup_timeout, "supervisor_ready")
        ready = wait_until(
            lambda: (found if len(found := {row["pid"] for row in rows(data_root / "logs/events.jsonl")
                                           if row.get("type") == "worker_ready"}) >= args.workers else None),
            args.startup_timeout, "all real worker_ready rows")
        worker_pids = sorted(ready)
        report["worker_pids"] = worker_pids
        report["workers_ready"] = len(worker_pids)
        for index in range(args.tasks):
            response = api(port, "/api/tasks", {
                "description": f"Isolated lifecycle qualification task {index}: wait for your loopback model.",
                "memory_mode": "forked", "actor_id": "qualification", "source": "qualification",
                "timeout_sec": 600,
            })
            task_id = response.get("task_id")
            if not task_id:
                raise RuntimeError(f"task admission failed: {response}")
            task_ids.append(task_id)
        report["task_ids"] = task_ids
        write_json(root / "progress.json", {**report, "stage": "tasks_submitted"})
        if task_ids:
            wait_until(lambda: len(model.calls) >= len(task_ids), 180, "all tasks physically in loopback model")
            before = read_json(data_root / "state/queue_snapshot.json", {})
            active = {row.get("id") for row in before.get("running", [])}
            if not set(task_ids) <= active:
                raise RuntimeError(f"not all admitted tasks are running: {active}")
            report["running_before"] = len(set(task_ids) & active)
            write_json(root / "queue_before.json", before)
        if args.extra_processes:
            from process_fixture import ProcessCohort
            process_cohort = ProcessCohort(root, source, task_ids[0], wait_until, args.daemon_descendants)
            report["process_fixture_before"] = process_cohort.start()
        ready_rows = [row for row in rows(data_root / "logs/events.jsonl") if row.get("type") == "worker_ready"
                      and row.get("pid") in worker_pids]
        report["worker_ready_source_shas"] = sorted({row.get("git_sha", "") for row in ready_rows})
        if report["worker_ready_source_shas"] != [report["source_sha"]]:
            raise RuntimeError("workers did not start from the frozen candidate SHA")
        current = subprocess.check_output(["git", "rev-parse", "HEAD", "HEAD^{tree}"], cwd=source, text=True).splitlines()
        if current != [report["source_sha"], report["source_tree"]]:
            raise RuntimeError("source identity moved before Quit")
        report["ready_resources"] = resource_facts(proc.pid)
        ws = connect(f"ws://127.0.0.1:{port}/ws", open_timeout=10)
        request = socket.create_connection(("127.0.0.1", port), timeout=5)
        request.sendall(b"POST /api/settings HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                        b"Content-Type: application/json\r\nContent-Length: 100000\r\n\r\n{\"OUROBOROS_MODEL\":\"")
        write_json(root / "progress.json", {**report, "stage": "quit_ready", "provider_calls": model.calls})
        def exit_launcher():
            launcher._shutdown_event.set()
            launcher.stop_agent()
            launcher.stop_tray_before_exit(launcher.release_pid_lock)
        background = Background(exit_launcher, lambda: port, launcher._shutdown_event)
        def observe_shutdown():
            last = None
            with (root / "shutdown_timeline.jsonl").open("w", encoding="utf-8") as handle:
                while not monitor_stop.is_set():
                    observed = {
                        "alive_workers": sum(pid_is_alive(pid) for pid in worker_pids),
                        "terminal_results": sum(read_json(data_root / "task_results" / (task_id + ".json"), {}).get("status")
                                                in {"cancelled", "interrupted", "failed", "completed"} for task_id in task_ids),
                        "server_exit": proc.poll(),
                        "server_shutdown_rows": len([row for row in rows(data_root / "logs/supervisor.jsonl")
                                                      if row.get("type") == "server_shutdown"]),
                    }
                    if observed != last:
                        handle.write(json.dumps({"monotonic": time.monotonic(), "wall_time": time.time(), **observed}) + "\n")
                        handle.flush()
                        last = observed
                    monitor_stop.wait(.05)
        monitor = threading.Thread(target=observe_shutdown, daemon=True)
        monitor.start()
        started = time.monotonic()
        report["quit_started_wall_time"] = time.time()
        report["quit_started_monotonic"] = started
        background.request_quit()
        report["quit_elapsed_sec"] = time.monotonic() - started
        report["server_exit"] = proc.poll()
        report["workers_surviving_after_quit"] = [pid for pid in worker_pids if pid_is_alive(pid)]
        report["server_shutdown"] = [row for row in rows(data_root / "logs/supervisor.jsonl")
                                     if row.get("type") == "server_shutdown"]
        report["false_supervisor_alarms"] = [row for row in rows(data_root / "logs/chat.jsonl")
            if row.get("system_type") == "supervisor_failure" or "Supervisor loop died" in str(row.get("text", ""))]
        report["task_results"] = {task_id: read_json(data_root / "task_results" / (task_id + ".json"), {})
                                  for task_id in task_ids}
        write_json(root / "queue_after.json", read_json(data_root / "state/queue_snapshot.json", {}))
        report["provider_calls"] = model.calls
        if process_cohort is not None:
            report["process_fixture_after"] = process_cohort.after_quit()
        report["passed"] = (
            report["server_exit"] in ((0,) if sys.platform == "win32" else (0, -signal.SIGTERM))
            and not report["workers_surviving_after_quit"]
            and any(row.get("cause") == ("launcher_quit" if sys.platform == "win32" else "external_signal")
                    for row in report["server_shutdown"])
            and not report["false_supervisor_alarms"]
            and all(row.get("status") in {"cancelled", "interrupted"} for row in report["task_results"].values())
            and report.get("process_fixture_after", {"passed": True})["passed"]
        )
    except Exception:
        report["error"] = traceback.format_exc()
        report["passed"] = False
    finally:
        # Preserve the pre-cleanup observation even if fixture cleanup later raises.
        write_json(root / "result-before-fixture-cleanup.json", report)
        monitor_stop.set()
        if monitor is not None:
            monitor.join(5)
        if request:
            request.close()
        if ws:
            try:
                ws.close()
            except Exception:
                pass
        cleanup_errors = []
        try:
            if proc and proc.poll() is None:
                report["cleanup_required"] = True
                launcher.stop_agent()
            if process_cohort is not None:
                process_cohort.cleanup()
        except Exception:
            cleanup_errors.append(traceback.format_exc())
        finally:
            model.close()
        captured = set(worker_pids)
        if proc is not None:
            captured.add(proc.pid)
        if process_cohort is not None:
            captured.update(pid for ids in process_cohort.markers.values() for pid in ids.values())
        captured.update(row["pid"] for row in rows(data_root / "state/process_ledger.jsonl")
                        if isinstance(row.get("pid"), int) and row["pid"] > 0)
        from ouroboros.process_containment import pid_is_zombie
        report["captured_pids_after_fixture_cleanup"] = sorted(captured)
        report["remaining_pids_after_fixture_cleanup"] = [pid for pid in captured
            if pid_is_alive(pid) and not pid_is_zombie(pid)]
        report["cleanup_errors"] = cleanup_errors
        report["source_status_after"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source, text=True).strip()
        report["source_sha_after"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if cleanup_errors or report["remaining_pids_after_fixture_cleanup"] or report["source_status_after"]:
            report["passed"] = False
        if report["source_sha_after"] != report["source_sha"]:
            report["passed"] = False
        write_json(root / "result.json", report)
    print(json.dumps({k: report.get(k) for k in ["passed", "source_sha", "workers_ready", "running_before",
          "quit_elapsed_sec", "server_exit", "workers_surviving_after_quit", "error"]}), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--extra-processes", action="store_true")
    parser.add_argument("--daemon-descendants", type=int, default=1)
    parser.add_argument("--expected-sha", default="")
    parser.add_argument("--require-windows", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    if args.child:
        return child(args)
    args.run_root.mkdir(parents=True, exist_ok=False)
    for name in ["data/state", "home", "app"]:
        (args.run_root / name).mkdir(parents=True)
    short_temp = Path(tempfile.mkdtemp(prefix="oq-", dir="/tmp" if os.name != "nt" else None))
    # Start from OS plumbing only: no provider/API tokens, model overrides, profile transport,
    # live runtime settings, proxies or operator Python path can reach this fixture.
    env = isolated_env(args.source, args.run_root, short_temp, os.environ, os.name == "nt")
    write_json(args.run_root / "environment.json", env)
    write_json(args.run_root / "driver_provenance.json", {
        "expected_sha": args.expected_sha, "driver": str(Path(__file__).resolve()),
        "scripts_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ("qualify_quit.py", "process_fixture.py")},
        "runner_platform": platform.platform(), "interpreter": sys.executable,
    })
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", "--source", str(args.source),
        "--run-root", str(args.run_root), "--workers", str(args.workers), "--tasks", str(args.tasks),
        "--daemon-descendants", str(args.daemon_descendants),
        "--expected-sha", args.expected_sha, "--startup-timeout", str(args.startup_timeout),
        *(["--require-windows"] if args.require_windows else []),
        *(["--extra-processes"] if args.extra_processes else [])],
        env=env, cwd=args.source)
    write_json(args.run_root / "producer_exit.json", {"exit": completed.returncode})
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
