"""A launcher-managed server's output relay: a stopped or vanished reader of the launcher's pipe
never holds the server's Python output or a worker's completion (``process_logging._OutputRelay``).

Every claim here lives at the process boundary (pipes, fork, exit), so each test runs one
server-shaped probe in its own process: its stdout and stderr are one pipe the test reads, as
the launcher reads the server, and it configures logging as ``server.main()`` does, after
setting the launcher's ``OUROBOROS_MANAGED_BY_LAUNCHER`` mark itself (``tests/conftest.py`` keeps
it out of every test child's environment); a probe child configures it as ``worker_main`` does. The test not reading is the stopped launcher, the test closing its end the launcher gone.
The same probe without the mark is the negative control on the same machine and run. A probe
reports through a side file, never through the pipe under test.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

import pytest

from ouroboros.process_containment import ProcessContainer
from ouroboros.test_environment import retain_tree

REPO = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="these pipe-fill and signal probes require POSIX")

PROBE = r'''
import contextlib, json, os, pathlib, sys, threading, time
repo, progress = sys.argv[1], pathlib.Path(sys.argv[2])
config = json.loads(sys.argv[3])
sys.path.insert(0, repo)
from ouroboros import process_logging


def mark(**facts):
    with progress.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "t": time.monotonic(), **facts}) + "\n")


def wait_for(name, limit=120.0):
    flag, deadline = progress.with_name(name), time.monotonic() + limit
    while not flag.exists() and time.monotonic() < deadline:
        time.sleep(0.02)


def held():
    relay = process_logging._relay
    return 0 if relay is None else relay.queued + len(relay.pending)


def drain_own(limit=60.0):
    deadline = time.monotonic() + limit
    while held() and time.monotonic() < deadline:
        time.sleep(0.02)


def child_writes(count):
    import logging
    process_logging.configure_process_logging(drive_logs=None)  # as worker_main does
    for index in range(count):
        print(f"child line {index:05d} " + "c" * 90, file=sys.stderr)
    logging.getLogger("child").info("the child logged its end")


def flood(records):
    import logging
    log = logging.getLogger("flood")
    for index in range(records):
        log.info("flood %05d %s", index, "f" * 120)


def raw_flood(writes):
    try:
        for _ in range(writes):
            os.write(1, b"f" * 4095 + b"\n")  # straight into the pipe, as native output goes
    except OSError:
        pass


def parked(level, gate):
    if level <= 1:
        gate.wait()
        return
    parked(level - 1, gate)


def fill_pipe():
    """A native writer fills the unread pipe (the stopped launcher); returns once it holds no more."""
    import select
    threading.Thread(target=raw_flood, args=(64,), daemon=True).start()  # 256 KiB: more than any pipe holds
    poller, deadline = select.poll(), time.monotonic() + 60
    poller.register(2, select.POLLOUT)
    while poller.poll(0) and time.monotonic() < deadline:
        time.sleep(0.01)
    mark(phase="pipe_full", full=not poller.poll(0))


if __name__ == "__main__":
    scenario = config["scenario"]
    if config["launcher"]:  # the launcher's mark, for this probe and its children (a test's own env never carries it)
        os.environ["OUROBOROS_MANAGED_BY_LAUNCHER"] = "1"
    launcher = os.fstat(1)
    if scenario == "early_server":
        import builtins, runpy
        fill_pipe()
        original_import = builtins.__import__
        def broken_import(name, *args, **kwargs):
            if name == "starlette.applications":
                mark(phase="early_failure", relayed=process_logging._relay is not None,
                     relay_thread=process_logging._relay.thread is not None)
                raise ImportError("controlled missing server dependency")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = broken_import
        runpy.run_path(str(pathlib.Path(sys.argv[1]) / "server.py"), run_name=config.get("main_name", "__main__"))
    if config.get("early_error"):  # before configure_process_logging: the plain streams, as before
        raise RuntimeError("an error before the relay")
    real_start = threading.Thread.start
    if config.get("refuse_relay_thread"):  # injected: what an exhausted process raises at a thread start
        def refusing(self):
            if self.name == "output-relay":
                raise RuntimeError("can't start new thread")
            return real_start(self)
        threading.Thread.start = refusing
    if config.get("before_relay"):  # block-buffered on a pipe: still in the plain stream when the relay comes
        sys.stdout.write("buffered before the relay " * 40 + "\n")
        fill_pipe()
    started = time.monotonic()
    real_dup = os.dup
    if config.get("refuse_relay_dup"):
        def refused_dup(fd):
            if fd == 2:
                raise OSError(24, "controlled descriptor exhaustion")
            return real_dup(fd)
        os.dup = refused_dup
    if config.get("max_bytes"):
        process_logging._install_output_relay(max_bytes=config["max_bytes"])
    # server.main()'s branch writes server.log; the fork arm leaves it out, since a server.log write of another
    # thread at the fork would wedge the child on that file's lock (the fork hazard of supervisor/workers.py).
    process_logging.configure_process_logging(drive_logs=progress.parent / "logs" if config.get("server_log", True) else None)
    os.dup = real_dup
    mark(phase="configured", relayed=process_logging._relay is not None, sec=round(time.monotonic() - started, 3))
    if scenario == "system_query":
        import subprocess
        from ouroboros import platform_layer
        original = subprocess.check_output
        def noisy_query(command, **kwargs):
            result = b"17179869184" if command[-1] == "hw.memsize" else b"controlled CPU"
            return original([sys.executable, "-I", "-c",
                             "import os; os.write(2, b'S' * 262144); os.write(1, " + repr(result) + ")"], **kwargs)
        subprocess.check_output = noisy_query
        platform_layer.platform.system = lambda: "Darwin"
        fill_pipe()
        value = getattr(platform_layer, config["query"])()
        mark(phase="query_returned", value=value)
        wait_for("release")
        sys.exit(0)
    if scenario == "fatal":
        import faulthandler, resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if config.get("fatal_handler"):
            faulthandler.enable(file=2)
        else:
            faulthandler.disable()
        if config.get("full"):
            fill_pipe()
        mark(phase="aborting")
        os.abort()
    if scenario == "driver":
        from playwright.sync_api import sync_playwright
        from playwright._impl import _transport
        from ouroboros.browser_output import driver_diagnostics
        node, entry = _transport.compute_driver_executable()
        wrapper = progress.parent / "noisy-driver.py"
        wrapper.write_text("import os, sys\nos.write(2, b'D' * 262144)\nos.execv(" +
                           repr(str(node)) + ", [" + repr(str(node)) + ", " + repr(str(entry)) + ", *sys.argv[1:]])\n")
        _transport.compute_driver_executable = lambda: (sys.executable, str(wrapper))
        fill_pipe()
        manager = driver_diagnostics() if config.get("adapt", True) else contextlib.nullcontext()
        with manager, sync_playwright() as pw:
            mark(phase="driver_ready", engine=pw.webkit.name)
        mark(phase="driver_stopped")
        wait_for("release")
        sys.exit(0)
    if scenario == "dependency_install":
        from supervisor import git_ops, git_ops_reset
        git_ops.REPO_DIR = progress.parent
        git_ops.DRIVE_ROOT = progress.parent / "data"
        git_ops_reset._installs_nothing = lambda: ""
        git_ops_reset._dependency_install_plan = lambda root: (
            [sys.executable, "-I", "-c", "import os; os.write(1, b'd' * 262144)"],
            "controlled diagnostic child", progress.parent / "requirements.lock")
        git_ops_reset._dependency_fingerprint = lambda *args: "controlled-diagnostic-child"
        fill_pipe()
        ok, source = git_ops_reset.sync_runtime_dependencies("relay verification")
        mark(phase="installed", ok=ok, source=source)
        wait_for("release")
        sys.exit(0 if ok else 1)
    if scenario == "child":
        import multiprocessing
        filler = threading.Thread(target=raw_flood, args=(64,), daemon=True)  # 256 KiB: more than any pipe holds
        filler.start()
        time.sleep(0.5)
        print("buffered before the start", end="")  # Process.start flushes it (multiprocessing.util._flush_std_streams)
        child = multiprocessing.get_context(config["method"]).Process(target=child_writes, args=(3000,))
        mark(phase="starting")
        try:
            child.start()
        except OSError as exc:
            mark(phase="start_failed", error=type(exc).__name__)
        else:
            mark(phase="started")
            child.join(30)
            mark(phase="joined", exitcode=child.exitcode)
        mark(phase="waiting")
        wait_for("release")
        filler.join(60)
        if child.pid is not None and child.exitcode is None:
            child.join(60)
            mark(phase="joined_after_release", exitcode=child.exitcode)
        drain_own()
        mark(phase="done", held=held())
    elif scenario == "slow_exit":
        sys.stdout.write("x" * config["bytes"])  # queued: the reader takes it slowly
        if config.get("stop_in") is not None:  # a shutdown door started the stop's one grace (owned_shutdown)
            from ouroboros import owned_shutdown
            owned_shutdown._GENERATION_STOP.deadline = time.monotonic() + config["stop_in"]
        mark(phase="exiting")
        from ouroboros.server_control import exit_after_shutdown
        exit_after_shutdown(7)
    elif scenario == "panic_during_handover":
        import logging
        # This handover would outlast Panic's own work by far: any wait for it shows in the exit.
        process_logging._relay.handover_sec = 120.0
        sys.stdout.write("x" * config["bytes"])
        def panic():
            time.sleep(0.3)
            mark(phase="panicking")
            from ouroboros.server_control import execute_panic_stop
            execute_panic_stop(None, lambda: None, data_dir=pathlib.Path(config["data"]), panic_exit_code=99,
                               log=logging.getLogger("panic"))
        threading.Thread(target=panic, daemon=True).start()
        mark(phase="exiting")
        from ouroboros.server_control import exit_after_shutdown
        exit_after_shutdown(7)
    elif scenario == "diagnostic_child":
        import subprocess
        fill_pipe()
        writer = "import os\nfor _ in range(64):\n    os.write(1, {!r} * 4096)\n"
        started = time.monotonic()
        with process_logging.relayed_child_output() as output:
            code = subprocess.call([sys.executable, "-I", "-c", writer.format(b"i")], stdout=output, stderr=output)
        mark(phase="relayed_returned", code=code, sec=round(time.monotonic() - started, 3), seam=output is not None)
        def inherited():  # the same child inheriting descriptors 1 and 2, as the installers do today
            mark(phase="inherited_returned", code=subprocess.call([sys.executable, "-I", "-c", writer.format(b"j")]))
        threading.Thread(target=inherited, daemon=True).start()
        wait_for("release")
        wait_for_phase = time.monotonic() + 60
        while not any('"inherited_returned"' in line for line in progress.read_text().splitlines()) \
                and time.monotonic() < wait_for_phase:
            time.sleep(0.02)
        drain_own()
        mark(phase="done", held=held())
    elif scenario == "browser_install":
        import builtins, subprocess, types
        from ouroboros.tools import browser
        fill_pipe()
        real_check_call = subprocess.check_call
        calls = []
        def noisy(command, **kwargs):
            calls.append(command)
            return real_check_call([sys.executable, "-I", "-c",
                                    "import os; os.write(1, b'b' * 262144)"], **kwargs)
        browser.subprocess.check_call = noisy
        if config["path"] == "package":
            real_import = builtins.__import__
            def missing(name, *args, **kwargs):
                if name == "playwright":
                    raise ImportError("injected missing package")
                return real_import(name, *args, **kwargs)
            builtins.__import__ = missing
            browser._pinned_chromium = lambda **kw: "not launched"
            browser._ensure_playwright_installed()
        elif config["path"] == "engine":
            browser._playwright_ready = False
            browser._playwright_ready_engines.clear()
            class MissingEngine:
                def __enter__(self):
                    return types.SimpleNamespace(webkit=types.SimpleNamespace(executable_path=str(progress / "absent")))
                def __exit__(self, *args):
                    pass
            import playwright.sync_api
            playwright.sync_api.sync_playwright = MissingEngine
            browser._ensure_playwright_installed(engine="webkit")
        else:
            browser._repair_host_dependencies("webkit")
        mark(phase="installed", calls=calls)
        wait_for("release")
        drain_own()
        mark(phase="done", held=held())
    elif scenario == "child_tail":
        import multiprocessing
        child = multiprocessing.get_context(config["method"]).Process(target=child_writes, args=(3000,))
        child.start()
        child.join(60)
        mark(phase="joined", exitcode=child.exitcode)
    elif scenario == "overflow":
        total, peak = 0, 0
        for index in range(4000):
            line = f"payload {index:05d} " + "p" * 84 + "\n"
            sys.stdout.write(line)
            total += len(line)
            peak = max(peak, process_logging._relay.queued)
        mark(phase="written", total=total, peak=peak)
        wait_for("release")
        drain_own()
        mark(phase="done", held=held())
    elif scenario == "server_log":
        import logging
        for index in range(3000):
            logging.getLogger("probe").info("stalled record %05d %s", index, "s" * 100)
        mark(phase="logged")
        wait_for("release")
        drain_own()
        mark(phase="done", held=held())
    elif scenario == "exit":
        if config.get("pipe_full"):
            fill_pipe()
        for index in range(config["lines"]):
            print(f"tail line {index:05d} " + "t" * 90)
        mark(phase="exiting")
        from ouroboros.server_control import exit_after_shutdown
        exit_after_shutdown(7)
    elif scenario == "import_error":
        raise RuntimeError("a broken import after the relay was installed")
    elif scenario == "fork":
        for index in range(200):
            print(f"parent line {index:05d} " + "a" * 90)  # queued: nobody reads yet
        pid = os.fork()
        if pid == 0:
            for index in range(200):
                print(f"child line {index:05d} " + "b" * 90)
            drain_own()
            mark(phase="child_done", held=held(), own=process_logging._relay.owner == os.getpid())
            os._exit(0)
        mark(phase="forked", child=pid)
        wait_for("release")
        drain_own()
        _, status = os.waitpid(pid, 0)
        mark(phase="done", child_status=status)
    elif scenario == "native":
        import ctypes, faulthandler
        gate = threading.Event()
        for _ in range(60):
            threading.Thread(target=parked, args=(15, gate), daemon=True).start()
        time.sleep(0.3)
        faulthandler.dump_traceback(file=2, all_threads=True)  # C-level writes with the GIL held
        mark(phase="dumped")
        library = ctypes.PyDLL(None)  # the GIL stays held across this write
        library.write.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)
        library.write.restype = ctypes.c_ssize_t
        mark(phase="pydll", written=library.write(2, b"n" * 262144, 262144))
    elif scenario == "descriptors":
        import subprocess
        relay = process_logging._relay
        same = [(os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (launcher.st_dev, launcher.st_ino) for fd in (1, 2)]
        check = subprocess.run([sys.executable, "-I", "-c", f"import os\ntry:\n    os.fstat({relay.output})\n"
                                "    print('open')\nexcept OSError:\n    print('closed')\n"],
                               stdout=subprocess.PIPE, text=True, timeout=60)
        mark(phase="descriptors", same=same, inheritable=os.get_inheritable(relay.output),
             in_child=check.stdout.strip(), streams=[type(sys.stdout).__name__, type(sys.stderr).__name__])
    elif scenario == "relay_failure":
        import logging
        relay = process_logging._relay
        print("before the failure", flush=True)
        drain_own()
        def broken():
            raise RuntimeError("injected relay fault")
        relay._take = broken  # injected: the relay thread's next step raises
        print("wakes the relay", flush=True)
        deadline = time.monotonic() + 30
        while not relay.failed and time.monotonic() < deadline:
            time.sleep(0.02)
        mark(phase="failed", failed=relay.failed)
        fill_pipe()
        started = time.monotonic()
        for index in range(200):
            print(f"after the failure {index:04d}", file=sys.stderr)
        logging.getLogger("probe").warning("a record after the failure")
        mark(phase="wrote", sec=round(time.monotonic() - started, 3))
        wait_for("release")
        mark(phase="done")
    elif scenario == "start_refused":
        import logging
        relay = process_logging._relay
        fill_pipe()
        started = time.monotonic()
        for index in range(200):
            print(f"while refused {index:04d} " + "r" * 80, file=sys.stderr)
        logging.getLogger("probe").warning("a record while refused")
        mark(phase="wrote", sec=round(time.monotonic() - started, 3), held=held(), thread=relay.thread is not None)
        threading.Thread.start = real_start
        print("after the start is allowed again", file=sys.stderr)
        mark(phase="allowed", thread=relay.thread is not None and relay.thread.is_alive())
        wait_for("release")
        drain_own()
        mark(phase="done", held=held())
    elif scenario == "before_relay":
        print("after the relay")
        wait_for("release")
        drain_own()
        mark(phase="done", held=held())
    elif scenario == "panic":
        import logging
        threading.Thread(target=flood, args=(1500,), daemon=True).start()
        time.sleep(0.5)
        mark(phase="panicking")
        from ouroboros.server_control import execute_panic_stop
        execute_panic_stop(None, lambda: None, data_dir=pathlib.Path(config["data"]), panic_exit_code=99,
                           log=logging.getLogger("panic"))
'''


class _Probe:
    """One probe process inside a ProcessContainer; its stdout and stderr are one pipe this test reads."""

    def __init__(self, tmp_path: pathlib.Path, *, launcher: bool = True, **config) -> None:
        self.root = tmp_path
        self.progress = tmp_path / "progress.jsonl"
        self.config = {**config, "launcher": launcher}
        self.container = ProcessContainer(lifeline=True, overload=True)
        self.proc: subprocess.Popen | None = None
        self.output = bytearray()
        self._reader: threading.Thread | None = None

    def start(self) -> "_Probe":
        self.root.mkdir(parents=True, exist_ok=True)
        script = self.root / "probe.py"
        script.write_text(PROBE, encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if key != "OUROBOROS_MANAGED_BY_LAUNCHER"}
        env["OUROBOROS_DATA_DIR"] = str(self.root / "data")
        self.proc = self.container.spawn(
            [sys.executable, "-I", str(script), str(REPO), str(self.progress), json.dumps(self.config)],
            cwd=str(self.root), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return self

    def rows(self) -> list:
        if not self.progress.exists():
            return []
        return [json.loads(line) for line in self.progress.read_text(encoding="utf-8").splitlines() if line.strip()]

    def phase(self, name: str) -> dict | None:
        return next((row for row in self.rows() if row.get("phase") == name), None)

    def wait_phase(self, name: str, limit: float) -> dict | None:
        """The phase's row once written, or what the probe left when it exited first."""
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            row = self.phase(name)
            if row is not None:
                return row
            if self.proc.poll() is not None:
                return self.phase(name)
            time.sleep(0.05)
        return None

    def read(self, size: int = 65536, pause: float = 0.0) -> None:
        """Read the pipe to its end, as a launcher that resumed; ``pause`` makes it a slow one."""
        def run():
            while chunk := self.proc.stdout.read1(size):
                self.output += chunk
                time.sleep(pause)
        self._reader = threading.Thread(target=run, daemon=True)
        self._reader.start()

    def flag(self, name: str) -> None:
        self.progress.with_name(name).write_text(name, encoding="utf-8")

    def finish(self, limit: float = 60) -> int:
        code = self.proc.wait(timeout=limit)
        if self._reader is not None:
            self._reader.join(limit)
        return code

    def close(self) -> None:
        error = self.container.reap()
        self.container.close()
        if self.proc is not None:
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=10)
            self.proc.stdout.close()
        if error:
            retain_tree(self.root, error)
        from ouroboros.test_evidence import diagnostic_directory
        import shutil
        kept = diagnostic_directory(self.root / "diagnostics", "output-relay", publish=True)
        (kept / "facts.json").write_text(json.dumps({"config": self.config, "rows": self.rows(),
            "exit": self.proc.returncode if self.proc else None, "teardown_error": error}, default=str))
        (kept / "output.bin").write_bytes(self.output)
        for path in (self.root / "probe.py", self.progress):
            if path.exists():
                shutil.copyfile(path, kept / path.name)
        if (self.root / "logs").exists():
            shutil.copytree(self.root / "logs", kept / "logs", dirs_exist_ok=True)
        assert not error, error


@contextlib.contextmanager
def _probe(tmp_path, **config):
    probe = _Probe(tmp_path, **config).start()
    try:
        yield probe
    finally:
        probe.close()


@pytest.mark.parametrize("method", ["spawn", "forkserver", "fork"])
def test_a_stopped_reader_holds_neither_process_start_nor_the_child_with_the_relay(tmp_path, method):
    """The incident: with the launcher's pipe full, ``Process.start`` stopped in its stream flush and
    every logging thread and worker with it. With the relay the child completes (exit 0) while
    nobody reads; without it the same start is still stuck seconds later, until reading resumes."""
    with _probe(tmp_path / "relay", scenario="child", method=method, server_log=False) as relayed:
        assert relayed.wait_phase("waiting", 60) is not None, relayed.rows()
        assert relayed.phase("configured")["relayed"] is True
        assert relayed.phase("joined")["exitcode"] == 0  # completed while the pipe was unread
        relayed.read()
        relayed.flag("release")
        assert relayed.finish() == 0
        assert relayed.phase("done")["held"] == 0
    delivered = bytes(relayed.output)
    assert delivered.count(b"f" * 4095 + b"\n") == 64 and b"buffered before the start" in delivered
    assert b"[output gap:" not in delivered

    with _probe(tmp_path / "direct", scenario="child", method=method, server_log=False, launcher=False) as direct:
        assert direct.wait_phase("starting", 30) is not None, direct.rows()
        assert direct.phase("configured")["relayed"] is False
        time.sleep(2)
        assert direct.phase("started") is None and direct.phase("start_failed") is None, direct.rows()
        direct.read()  # the launcher reads again
        direct.flag("release")
        assert direct.finish() == 0
        assert direct.phase("joined")["exitcode"] == 0, direct.rows()


def test_a_closed_reader_lets_the_child_finish_with_the_relay(tmp_path):
    """The launcher gone (its read end closed): without the relay ``Process.start``'s flush raises
    EPIPE; with it the server's writes are dropped and the child still exits 0."""
    with _probe(tmp_path / "relay", scenario="child", method="spawn", server_log=False) as relayed:
        relayed.proc.stdout.close()
        relayed.flag("release")
        assert relayed.finish() == 0
        assert relayed.phase("joined")["exitcode"] == 0, relayed.rows()

    with _probe(tmp_path / "direct", scenario="child", method="spawn", server_log=False, launcher=False) as direct:
        direct.proc.stdout.close()
        direct.flag("release")
        direct.finish()
        failed, joined = direct.phase("start_failed"), direct.phase("joined")
        assert (failed and failed["error"] == "BrokenPipeError") or (joined and joined["exitcode"] != 0), direct.rows()


def test_overflow_is_bounded_counted_and_marked_where_it_happened(tmp_path):
    """Unread output beyond the queue is dropped and counted; the count reaches the reader as one
    marker line at the dropped position once it reads again, and nothing else is lost."""
    limit = 64 * 1024
    with _probe(tmp_path, scenario="overflow", max_bytes=limit) as probe:
        written = probe.wait_phase("written", 60)
        assert written is not None, probe.rows()
        assert 0 < written["peak"] <= limit
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0
        assert probe.phase("done")["held"] == 0
    data = bytes(probe.output)
    markers = list(re.finditer(rb"\[output gap: (\d+) bytes dropped by the output relay while this output was not read\]\n", data))
    dropped = sum(int(match.group(1)) for match in markers)
    assert markers and dropped > 0
    delivered = len(data) - sum(len(match.group(0)) for match in markers)
    # A marker starts a line of its own; where the dropped bytes ended mid-line it adds that one newline.
    assert written["total"] - dropped <= delivered <= written["total"] - dropped + len(markers)
    first = data.index(b"[output gap:")
    assert data[:first].endswith(b"\n") and b"payload 00000 " in data[:first]


def test_server_log_keeps_every_record_while_the_stream_drops(tmp_path):
    """``server.log`` stays the server's own complete file: what the relay drops for a stopped
    reader is only the stream's copy, and the stream says how much it dropped."""
    with _probe(tmp_path, scenario="server_log", max_bytes=64 * 1024) as probe:
        assert probe.wait_phase("logged", 60) is not None, probe.rows()
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0
    server_log = (tmp_path / "logs" / "server.log").read_text(encoding="utf-8")
    assert all(f"stalled record {index:05d} " in server_log for index in range(3000))
    stream = bytes(probe.output)
    assert b"[output gap:" in stream and stream.count(b"stalled record ") < 3000


def test_an_ordinary_exit_hands_the_tail_over_and_never_waits_for_a_stopped_reader(tmp_path):
    """``exit_after_shutdown`` lets the relay hand its whole queue to a reading launcher, even a tail
    bigger than any pipe; beside a stopped one it exits within the close's absolute grace, losing what the
    pipe did not take, also when a native writer keeps that pipe full; the same exit without the relay waits."""
    with _probe(tmp_path / "reading", scenario="exit", lines=3000) as reading:
        reading.read()
        assert reading.finish() == 7
    assert bytes(reading.output).count(b"tail line ") == 3000

    for name, config in (("stopped", {"lines": 3000}), ("native_full", {"lines": 200, "pipe_full": True})):
        with _probe(tmp_path / name, scenario="exit", **config) as stopped:
            exiting = stopped.wait_phase("exiting", 60)
            assert exiting is not None, stopped.rows()
            assert stopped.proc.wait(timeout=20) == 7  # nobody read: no wait for the reader
            exited = time.monotonic()
            stopped.read()
            stopped.finish()
        assert exited - exiting["t"] < 5, (name, exited - exiting["t"])  # one host clock for both
        assert bytes(stopped.output).count(b"tail line ") < config["lines"]  # the rest died with the process

    with _probe(tmp_path / "direct", scenario="exit", lines=3000, launcher=False) as direct:
        time.sleep(2)
        assert direct.proc.poll() is None  # stuck in a write to the unread pipe
        direct.read()
        assert direct.finish() == 7


@pytest.mark.parametrize("method", ["spawn", "forkserver", "fork"])
def test_a_pool_child_s_tail_reaches_a_reading_launcher_at_its_own_exit(tmp_path, method):
    """A pool child's relay hands its queue over as the child ends: spawn through the interpreter's exit,
    fork and forkserver through ``threading._shutdown`` in ``multiprocessing``'s bootstrap before
    ``os._exit``. A reading launcher gets every line, the child's last log record included."""
    with _probe(tmp_path, scenario="child_tail", method=method, server_log=False) as probe:
        probe.read()
        assert probe.finish() == 0
        assert probe.phase("joined")["exitcode"] == 0, probe.rows()
    data = bytes(probe.output)
    assert data.count(b"child line ") == 3000 and b"the child logged its end" in data


@pytest.mark.parametrize("stop_in", [None, 1.0])
def test_an_ordinary_exit_ends_its_handover_by_its_deadline_beside_a_slow_reader(tmp_path, stop_in):
    """A reader that keeps taking, slowly, never holds an exit past the handover's absolute deadline:
    ``SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC`` after it starts, or earlier the ordinary stop's own
    deadline once a shutdown door started it (``owned_shutdown.stop_deadline``). What the reader has
    not taken by then is the exit's loss."""
    from ouroboros.runtime_limits import SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC

    total = 900_000  # about nine seconds for this reader, more than either deadline allows
    with _probe(tmp_path, scenario="slow_exit", bytes=total, stop_in=stop_in) as probe:
        probe.read(size=1024, pause=0.01)
        exiting = probe.wait_phase("exiting", 60)
        assert exiting is not None, probe.rows()
        assert probe.proc.wait(timeout=60) == 7
        exited = time.monotonic()
        probe.finish()
    bound = SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC if stop_in is None else stop_in
    took = exited - exiting["t"]  # one host clock for both
    taken = bytes(probe.output).count(b"x")
    print("SLOW_EXIT", json.dumps({"stop_in": stop_in, "bound": bound, "took": round(took, 3), "taken": taken}))
    assert took < bound + 2, (took, bound)
    assert 64 * 1024 < taken < total, taken  # it waited while the reader took, then left the rest


def test_panic_never_waits_for_an_ordinary_exit_s_handover(tmp_path):
    """Panic accepted while an ordinary exit hands over to a slow reader: Panic's own bounded writes
    and ``os._exit(99)`` follow at once; the handover (made to outlast Panic by far) holds nothing."""
    with _probe(tmp_path, scenario="panic_during_handover", bytes=1_000_000, data=str(tmp_path / "panic-data")) as probe:
        probe.read(size=512, pause=0.01)  # about twenty seconds for the whole handover
        panicking = probe.wait_phase("panicking", 60)
        assert panicking is not None, probe.rows()
        code = probe.proc.wait(timeout=60)
        exited = time.monotonic()
        probe.finish()
    taken = bytes(probe.output).count(b"x")
    print("PANIC_DURING_HANDOVER", json.dumps({"code": code, "after_panic": round(exited - panicking["t"], 3), "taken": taken}))
    assert code == 99
    assert (tmp_path / "panic-data" / "state" / "panic_stop.flag").exists()
    assert exited - panicking["t"] < 12, exited - panicking["t"]  # Panic's own work, not the handover's
    assert taken < 1_000_000


def test_a_diagnostic_child_hands_its_output_to_the_relay_and_never_waits_for_a_stopped_reader(tmp_path):
    """A child given ``relayed_child_output`` writes 256 KiB and exits 0 beside a full, unread pipe:
    a thread of the parent drains its private pipe into the bounded relay. The same child inheriting
    descriptors 1 and 2 (the negative control) waits until the reader resumes; then both arrive."""
    with _probe(tmp_path, scenario="diagnostic_child") as probe:
        relayed = probe.wait_phase("relayed_returned", 60)
        assert relayed is not None and relayed["code"] == 0 and relayed["seam"] is True, probe.rows()
        assert probe.phase("pipe_full")["full"] is True
        time.sleep(2)
        assert probe.phase("inherited_returned") is None, probe.rows()  # nobody reads: it waits
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0
        assert probe.phase("inherited_returned")["code"] == 0
        assert probe.phase("done")["held"] == 0
    data = bytes(probe.output)
    assert data.count(b"i") >= 64 * 4096 and data.count(b"j") >= 64 * 4096


def test_an_error_before_the_relay_still_reaches_a_reading_launcher(tmp_path):
    """Output before ``configure_process_logging`` (the server's imports, a worker's preamble) goes to
    the plain streams as before: a reading launcher gets the traceback whole. Beside a stopped reader
    that write still waits, as it did before the relay: the relay covers only what follows setup."""
    with _probe(tmp_path, scenario="exit", lines=1, early_error=True) as probe:
        probe.read()
        assert probe.finish() == 1
        assert probe.phase("configured") is None
    text = bytes(probe.output).decode(errors="replace")
    assert "RuntimeError: an error before the relay" in text and "Traceback" in text


def test_a_failed_import_after_the_relay_still_reaches_the_launcher(tmp_path):
    """An interpreter exit (here an exception at import) runs the relay's ``atexit`` handover, so
    the traceback that ``agent_stdout.log`` exists for is not left in the relay's queue."""
    with _probe(tmp_path, scenario="import_error") as probe:
        probe.read()
        assert probe.finish() == 1
    text = bytes(probe.output).decode(errors="replace")
    assert "RuntimeError: a broken import after the relay was installed" in text and "Traceback" in text


def test_a_forked_child_relays_its_own_output_and_never_repeats_the_parent_s(tmp_path):
    """A forked child starts an empty relay of its own (its thread with its first write): output
    the parent still held at the fork reaches the reader once, the child's own output too."""
    with _probe(tmp_path, scenario="fork") as probe:
        assert probe.wait_phase("forked", 30) is not None, probe.rows()
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0
        child = probe.phase("child_done")
    assert child is not None and child["own"] is True and child["held"] == 0, probe.rows()
    assert probe.phase("done")["child_status"] == 0
    data = bytes(probe.output)
    assert all(data.count(f"parent line {index:05d} ".encode()) == 1 for index in range(200))
    assert all(data.count(f"child line {index:05d} ".encode()) == 1 for index in range(200))


def test_a_native_write_holding_the_gil_never_waits_for_the_relay(tmp_path):
    """C code writing to descriptor 2 with the GIL held (the all-threads traceback dump a fatal
    error uses, or a GIL-holding foreign call) never waits for the relay thread, which needs the
    GIL: descriptors 1 and 2 stay the launcher's pipe, so a reading launcher takes it all. This
    shows no new dependence on the relay's thread; beside a stopped reader such a write still
    waits, as before the relay."""
    with _probe(tmp_path, scenario="native") as probe:
        probe.read()
        assert probe.finish(limit=60) == 0
        assert probe.phase("configured")["relayed"] is True
        assert probe.phase("dumped") is not None and probe.phase("pydll")["written"] == 262144, probe.rows()
    data = bytes(probe.output)
    assert data.count(b"n" * 4096) >= 262144 // 4096 - 1 and b"Thread 0x" in data


def test_the_relay_leaves_the_descriptors_to_the_launcher_and_keeps_its_copy_private(tmp_path):
    """Descriptors 1 and 2 remain the launcher's pipe and the relay writes through a duplicate no
    exec'd child inherits, so a descendant holds exactly what it held before."""
    with _probe(tmp_path, scenario="descriptors") as probe:
        probe.read()
        assert probe.finish() == 0
        facts = probe.phase("descriptors")
    assert facts["same"] == [True, True] and facts["inheritable"] is False and facts["in_child"] == "closed", facts
    assert facts["streams"] == ["_RelayStream", "_RelayStream"], facts


def test_a_failed_relay_says_so_and_drops_later_output_beside_a_stopped_reader(tmp_path):
    """Injected fault (the relay thread's next step raises): it writes one line naming the stop, never
    through logging, and this process's later output is dropped, not written to the plain stream, whose
    descriptor is the same unread pipe: 200 prints and a log record return at once beside a full pipe."""
    with _probe(tmp_path, scenario="relay_failure") as probe:
        wrote = probe.wait_phase("wrote", 60)
        assert wrote is not None and wrote["sec"] < 5, probe.rows()  # nobody reads yet
        assert probe.phase("failed")["failed"] is True and probe.phase("pipe_full")["full"] is True
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0
    text = bytes(probe.output).decode(errors="replace")
    assert "before the failure" in text and "[output relay stopped (RuntimeError)" in text
    assert "after the failure" not in text and "a record after the failure" not in text


def test_a_refused_relay_thread_keeps_output_queued_beside_a_stopped_reader(tmp_path):
    """No thread can start for the relay (injected: what an exhausted process raises): output stays
    queued, bounded, and the writes return at once beside a full pipe; a later write starts the thread
    and a reading launcher then gets everything, in order."""
    with _probe(tmp_path, scenario="start_refused", refuse_relay_thread=True) as probe:
        wrote = probe.wait_phase("wrote", 60)
        assert wrote is not None and wrote["sec"] < 5 and wrote["thread"] is False and wrote["held"] > 0, probe.rows()
        assert probe.wait_phase("allowed", 30)["thread"] is True
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0
        assert probe.phase("done")["held"] == 0
    data = bytes(probe.output)
    assert data.count(b"while refused ") == 200 and b"a record while refused" in data
    assert data.index(b"while refused 0199") < data.index(b"after the start is allowed again")


def test_text_buffered_before_the_relay_goes_out_first_and_never_holds_setup(tmp_path):
    """Text the plain ``sys.stdout`` still buffers when the relay comes (block-buffered on a pipe) is
    flushed by the relay's thread, ahead of relayed output, so setup returns at once beside a full pipe."""
    with _probe(tmp_path, scenario="before_relay", before_relay=True) as probe:
        configured = probe.wait_phase("configured", 60)
        assert configured is not None and configured["sec"] < 5 and configured["relayed"] is True, probe.rows()
        assert probe.phase("pipe_full")["full"] is True
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0
    data = bytes(probe.output)
    assert data.count(b"buffered before the relay ") == 40
    assert data.index(b"buffered before the relay ") < data.index(b"after the relay")


def test_panic_exits_promptly_beside_a_stopped_reader(tmp_path):
    """Panic beside a stopped reader with a full queue: its log calls never wait for the pipe, its
    bounded writes and ``os._exit(99)`` follow, and the relay adds no wait to that exit."""
    with _probe(tmp_path, scenario="panic", data=str(tmp_path / "panic-data")) as probe:
        panicking = probe.wait_phase("panicking", 60)
        assert panicking is not None, probe.rows()
        code = probe.proc.wait(timeout=60)
        exited = time.monotonic()
        probe.read()
        probe.finish()
    assert code == 99
    assert (tmp_path / "panic-data" / "state" / "panic_stop.flag").exists()  # its bounded writes ran
    assert exited - panicking["t"] < 20  # one host clock for both


@pytest.mark.parametrize("path", ["package", "engine", "dependencies"])
def test_browser_installers_complete_with_a_full_unread_launcher_pipe(tmp_path, path):
    """Actual preparation functions; only installer argv and missing-package/binary facts are synthetic."""
    with _probe(tmp_path, scenario="browser_install", path=path) as probe:
        installed = probe.wait_phase("installed", 60)
        assert installed is not None, probe.rows()
        assert probe.phase("pipe_full")["full"]
        assert len(installed["calls"]) == (2 if path == "engine" else 1)
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0


@pytest.mark.parametrize("failure", ["pipe", "thread"])
def test_diagnostic_relay_preparation_failure_refuses_before_child_start(monkeypatch, failure):
    from types import SimpleNamespace
    from ouroboros import process_logging
    from ouroboros.tools import browser

    monkeypatch.setattr(process_logging, "_relay", SimpleNamespace(owner=os.getpid()))
    created = []
    real_pipe = os.pipe
    def pipe():
        if failure == "pipe":
            raise OSError("injected descriptor exhaustion")
        pair = real_pipe()
        created.extend(pair)
        return pair
    def refuse(_thread):
        raise RuntimeError("injected thread exhaustion")
    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(threading.Thread, "start", refuse)
    monkeypatch.setattr(browser.subprocess, "check_call", lambda *a, **kw: pytest.fail("child must not start"))
    with pytest.raises(RuntimeError, match="child not started"):
        browser._diagnostic_install(["unused"])
    for fd in created:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_exit_hooks_share_one_deadline_even_when_reader_makes_no_progress(monkeypatch):
    from types import SimpleNamespace
    from ouroboros import process_logging

    relay = process_logging._OutputRelay(-1, 1024, 0.08)
    relay.thread = SimpleNamespace(is_alive=lambda: True)
    relay.queued = 1
    monkeypatch.setattr(process_logging, "_relay", relay)
    start = time.monotonic()
    process_logging.finish_output_relay()
    first_deadline = relay.finish_deadline
    process_logging.finish_output_relay()
    assert relay.finish_deadline == first_deadline
    assert time.monotonic() - start < 0.14


def test_a_gap_queued_beside_an_idle_writer_reaches_a_reading_launcher_without_waiting(monkeypatch):
    """Astra r1: one write past the bound into an idle relay queued its gap without waking the writer, so a
    reading launcher saw no marker and the exit's handover spent its whole deadline. A real pipe and thread."""
    from ouroboros import process_logging

    read_end, write_end = os.pipe()
    relay = process_logging._OutputRelay(write_end, 1024, 5.0)
    try:
        relay.put(b"first\n")
        assert os.read(read_end, 64) == b"first\n"
        with relay.cond:
            assert relay.cond.wait_for(lambda: relay.done == 6 and not relay.pending, 5)
        time.sleep(0.05)  # the writer is back waiting for an entry
        relay.put(b"x" * 1025)
        start = time.monotonic()
        relay.finish(start + 5.0)
        assert time.monotonic() - start < 2.5 and relay.gaps == 0
        assert os.read(read_end, 4096).startswith(b"[output gap: 1025 bytes dropped")
    finally:
        os.close(read_end)
        relay.put(b"gone\n")  # EPIPE ends the writer and closes its descriptor


def test_a_stdout_pipe_beside_another_stderr_is_never_relayed_even_without_pipe_identity(monkeypatch):
    """Carbon r1: on Windows ``fstat`` gives every pipe, console and NUL dev and inode 0, so a stdout pipe beside a
    console stderr passed the one-pipe check and stdout was merged into stderr's descriptor. Both must be FIFOs."""
    import stat
    from types import SimpleNamespace

    from ouroboros import process_logging

    kinds = {1: stat.S_IFIFO, 2: stat.S_IFCHR}  # the launcher's stdout pipe; an operator's console stderr
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setattr(process_logging, "_relay", None)
    monkeypatch.setattr(process_logging.sys, "stdout", process_logging.sys.stdout)
    monkeypatch.setattr(process_logging.sys, "stderr", process_logging.sys.stderr)
    monkeypatch.setattr(process_logging.os, "fstat", lambda fd: SimpleNamespace(st_mode=kinds[fd], st_dev=0, st_ino=0))
    monkeypatch.setattr(process_logging.os, "dup", lambda fd: pytest.fail("no relay: stdout must keep its own pipe"))
    assert process_logging._install_output_relay(start_thread=False) is False and process_logging._relay is None


def test_initial_descriptor_failure_never_restores_blocking_output(tmp_path):
    with _probe(tmp_path, scenario="exit", lines=200, pipe_full=True, refuse_relay_dup=True) as probe:
        assert probe.wait_phase("configured", 60)["relayed"]
        assert probe.wait_phase("exiting", 60) is not None
        assert probe.finish(limit=15) == 7
        probe.read()
        probe.finish()


@pytest.mark.parametrize("main_name", ["__main__", "__mp_main__"])
def test_server_import_failure_exits_with_a_full_unread_launcher_pipe(tmp_path, main_name):
    with _probe(tmp_path, scenario="early_server", main_name=main_name) as probe:
        assert probe.wait_phase("pipe_full", 60)["full"]
        assert probe.wait_phase("early_failure", 60)["relayed"]
        assert probe.phase("early_failure")["relay_thread"] is (main_name == "__main__")
        assert probe.finish(limit=15) == 1
        probe.read()


def test_runtime_dependency_installer_finishes_with_full_unread_pipe(tmp_path):
    with _probe(tmp_path, scenario="dependency_install") as probe:
        assert probe.wait_phase("pipe_full", 60)["full"]
        assert probe.wait_phase("installed", 60)["ok"]
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0


@pytest.mark.parametrize("query, value", [("get_system_memory", "16.0 GB"), ("get_cpu_info", "controlled CPU")])
def test_system_query_preserves_result_with_full_unread_diagnostic_pipe(tmp_path, query, value):
    with _probe(tmp_path, scenario="system_query", query=query) as probe:
        assert probe.wait_phase("pipe_full", 60)["full"]
        assert probe.wait_phase("query_returned", 60)["value"] == value
        probe.read()
        probe.flag("release")
        assert probe.finish() == 0


@pytest.mark.parametrize("adapt", [True, False])
def test_real_playwright_driver_keeps_protocol_and_diagnostic_channels_separate(tmp_path, adapt):
    with _probe(tmp_path, scenario="driver", adapt=adapt) as probe:
        assert probe.wait_phase("pipe_full", 60)["full"]
        if adapt:
            assert probe.wait_phase("driver_ready", 60)["engine"] == "webkit"
            assert probe.wait_phase("driver_stopped", 60)
        else:
            assert probe.wait_phase("driver_ready", 2) is None
        probe.read()
        assert probe.wait_phase("driver_ready", 60)["engine"] == "webkit"
        probe.flag("release")
        assert probe.finish() == 0


@pytest.mark.parametrize("handler, full", [(False, True), (True, False), (True, True)])
def test_actual_abort_and_optional_fatal_handler(tmp_path, handler, full):
    import signal
    with _probe(tmp_path, scenario="fatal", fatal_handler=handler, full=full) as probe:
        if not full:
            probe.read()
        assert probe.wait_phase("aborting", 60)
        if handler and full:
            # The opt-in C fatal handler writes directly and waits for its reader.
            # This is a measured residual, not proof that buffering covers fatal output.
            time.sleep(0.5)
            assert probe.proc.poll() is None
            probe.read()
        assert probe.finish(limit=15) == -signal.SIGABRT
        if full and not handler:
            probe.read()
