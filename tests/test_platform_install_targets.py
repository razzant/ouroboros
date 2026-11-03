"""Pip targets the invoked venv even when its binary belongs to the bundle."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

import pytest

from ouroboros.launcher_bootstrap import embedded_python_env
from ouroboros.platform_layer import pip_install_target_args
from ouroboros.process_containment import ProcessContainer


def _run_install_command(args, *, env, timeout, capture_output=False, text=False):
    """Run one checked install probe, settling its whole tree even after success."""
    container = ProcessContainer()
    proc = None
    failure = None
    try:
        proc = container.spawn(
            args, env=env, text=text,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        result = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
        result.check_returncode()
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            try:
                try:
                    error = container.reap()
                    if error:
                        raise RuntimeError(error)
                finally:
                    if proc is not None:
                        proc.wait(timeout=10)
                if proc is not None and isinstance(failure, subprocess.TimeoutExpired):
                    # A failed reap must retain its report, not attempt a drain.
                    # Windows' initial timeout has no output; join its readers
                    # only after successful reap, retaining the original timeout.
                    failure.stdout, failure.stderr = proc.communicate(timeout=10)
            finally:
                try:
                    container.close()
                finally:
                    if proc is not None:
                        for stream in (proc.stdout, proc.stderr):
                            if stream is not None:
                                stream.close()
        except BaseException as cleanup_error:
            if failure is not None:
                raise cleanup_error from failure
            raise


@pytest.mark.parametrize("layout", ["bin/python3", "Scripts/python.exe"])
@pytest.mark.parametrize("linked", [False, True], ids=["copied", "symlink"])
def test_confirmed_venv_precedes_embedded_binary_resolution(tmp_path, layout, linked):
    bundled = tmp_path / "python-standalone" / "bin" / "python3"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"interpreter fixture")
    # A copied venv under this ancestor also defeats ancestry-only classification.
    venv = tmp_path / "python-standalone" / "work-environment"
    invocation = venv / pathlib.Path(layout)
    invocation.parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = bundled\n", encoding="utf-8")
    if linked:
        try:
            invocation.symlink_to(bundled)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")
    else:
        shutil.copyfile(bundled, invocation)
    assert pip_install_target_args(str(invocation)) == []


def test_venv_name_and_ambient_env_do_not_override_direct_bundle(tmp_path, monkeypatch):
    bundled = tmp_path / "python-standalone" / ".venv" / "bin" / "python3"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"interpreter fixture")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "unrelated-env"))
    assert pip_install_target_args(str(bundled)) == ["--user"]
    alias = tmp_path / "python-alias"
    try:
        alias.symlink_to(bundled)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    assert pip_install_target_args(str(alias)) == ["--user"]


def test_plain_python_and_nonstandard_cfg_neighbor(tmp_path):
    plain = tmp_path / "usr" / "bin" / "python3"
    assert pip_install_target_args(str(plain)) == []
    bundled = tmp_path / "python-standalone" / "python.exe"
    bundled.parent.mkdir()
    bundled.write_bytes(b"interpreter fixture")
    (bundled.parent / "pyvenv.cfg").write_text("home = elsewhere\n", encoding="utf-8")
    assert pip_install_target_args(str(bundled)) == ["--user"]


@pytest.mark.serial
def test_real_venv_installs_and_imports_local_wheel(tmp_path):
    venv = tmp_path / "work-environment"
    env = embedded_python_env(tmp_path / "data")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    _run_install_command([sys.executable, "-m", "venv", str(venv)], env=env, timeout=90)
    interpreter = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheel = tmp_path / "ouroboros_venv_probe-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ouroboros_venv_probe.py", "VALUE = 'local wheel imported'\n")
        archive.writestr("ouroboros_venv_probe-1.0.dist-info/METADATA",
                         "Metadata-Version: 2.1\nName: ouroboros-venv-probe\nVersion: 1.0\n")
        archive.writestr("ouroboros_venv_probe-1.0.dist-info/WHEEL",
                         "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("ouroboros_venv_probe-1.0.dist-info/RECORD", "")
    _run_install_command(
        [str(interpreter), "-m", "pip", "install", *pip_install_target_args(str(interpreter)),
         "--no-index", "--no-deps", "--no-cache-dir", str(wheel)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    result = _run_install_command(
        [str(interpreter), "-c", "import json,sys,ouroboros_venv_probe as p; "
         "print(json.dumps([sys.prefix,p.__file__,p.VALUE]))"],
        env=env, capture_output=True, text=True, timeout=10,
    )
    prefix, module, value = json.loads(result.stdout)
    assert pathlib.Path(prefix).resolve() == venv.resolve()
    assert venv.resolve() in pathlib.Path(module).resolve().parents
    assert value == "local wheel imported"


@pytest.mark.serial
@pytest.mark.parametrize("outcome", ["timeout", "success", "nonzero", "no-cleanup"])
def test_install_command_settles_descendants_before_fallback(tmp_path, outcome):
    """Observe births/exit state before the independent outer owner reaps anything."""
    from ouroboros import platform_layer as pl
    from ouroboros.process_containment import pid_is_zombie

    def live(identity):
        pid, birth = identity
        assert birth, f"missing birth for {pid}"
        if pl.pid_provably_gone(pid) or pid_is_zombie(pid):
            return False
        current = pl.process_start_time(pid)
        assert current, f"unreadable birth for {pid}"
        return current == birth

    probe = tmp_path / "tree.py"
    repo = pathlib.Path(__file__).resolve().parents[1]
    probe.write_text(f"import sys; sys.path.insert(0, {str(repo)!r})\n" + '''
import json, os, subprocess, sys, time
from pathlib import Path
from ouroboros.platform_layer import process_start_time
root, depth, outcome = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
receipt = root / f"{depth}.tmp"
receipt.write_text(json.dumps([os.getpid(), process_start_time(os.getpid())]))
receipt.replace(root / f"{depth}.json")
if depth < 2:
    subprocess.Popen([sys.executable, __file__, str(root), str(depth + 1), outcome],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.monotonic() + 10
while not all((root / f"{n}.json").exists() for n in range(3)):
    if time.monotonic() >= deadline:
        raise RuntimeError("tree did not start")
    time.sleep(0.02)
if depth == 0:
    print("probe stdout", flush=True)
    print("probe stderr", file=sys.stderr, flush=True)
    if outcome in {"success", "nonzero"}:
        sys.exit(7 if outcome == "nonzero" else 0)
time.sleep(60)
''', encoding="utf-8")
    args = [sys.executable, str(probe), str(tmp_path), "0", outcome]
    # subprocess.run is the old, deliberately leaking negative control. The
    # outer container owns that whole runner, including its orphaned children.
    runner = f'''
import json, os, subprocess, sys
sys.path.insert(0, {str(repo)!r})
from tests.test_platform_install_targets import _run_install_command
try:
    if {outcome == "no-cleanup"!r}:
        result = subprocess.run({args!r}, env=os.environ, timeout=5,
                                capture_output=True, text=True, check=True)
    else:
        result = _run_install_command({args!r}, env=os.environ, timeout=5,
                                      capture_output=True, text=True)
except subprocess.TimeoutExpired as exc:
    assert exc.timeout == 5
    for captured, expected in ((exc.output, "probe stdout"), (exc.stderr, "probe stderr")):
        if isinstance(captured, bytes):
            captured = captured.decode()
        assert captured.splitlines() == [expected]
    print(json.dumps({{"kind": type(exc).__name__}}))
except subprocess.CalledProcessError as exc:
    assert exc.returncode == 7, (exc.returncode, exc.stdout, exc.stderr)
    assert exc.stdout == "probe stdout\\n" and exc.stderr == "probe stderr\\n"
    print(json.dumps({{"kind": type(exc).__name__}}))
else:
    assert result.returncode == 0
    assert result.stdout == "probe stdout\\n" and result.stderr == "probe stderr\\n"
    print(json.dumps({{"kind": "success"}}))
'''
    owners = []
    try:
        for code in ("import time; time.sleep(60)", runner):
            container = ProcessContainer()
            owners.append((container, None))
            proc = container.spawn([sys.executable, "-c", code],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            owners[-1] = (container, proc)
        sentinel = owners[0][1]
        sentinel_identity = (sentinel.pid, pl.process_start_time(sentinel.pid))
        stdout, stderr = proc.communicate(timeout=30)
        assert proc.returncode == 0, (stdout, stderr)
        expected = {"timeout": "TimeoutExpired", "no-cleanup": "TimeoutExpired",
                    "success": "success", "nonzero": "CalledProcessError"}[outcome]
        assert json.loads(stdout) == {"kind": expected}
        identities = [json.loads((tmp_path / f"{n}.json").read_text()) for n in range(3)]
        observed = [live(identity) for identity in identities]
        print(json.dumps({"outcome": outcome, "identities": identities,
                          "live_before_fallback": observed, "sentinel": live(sentinel_identity)}))
        assert observed == ([False, True, True] if outcome == "no-cleanup" else [False] * 3)
        assert live(sentinel_identity)
    finally:
        for container, child in reversed(owners):
            try:
                assert not container.reap()
                if child is not None:
                    child.wait(timeout=10)
            finally:
                container.close()
                if child is not None:
                    child.stdout.close()
                    child.stderr.close()
    assert all(not live(identity) for identity in identities)
    assert not live(sentinel_identity)


@pytest.mark.serial
@pytest.mark.parametrize("timeout", [0.05, 10], ids=["timeout-and-cleanup-error", "success-and-cleanup-error"])
def test_install_command_cleanup_failure_stays_red(monkeypatch, timeout):
    containers = []

    class ReportFailedCleanup(ProcessContainer):
        def spawn(self, *args, **kwargs):
            self.proc = super().spawn(*args, **kwargs)
            containers.append(self)
            communicate = self.proc.communicate

            def before_reap_only(*args, **kwargs):
                assert not getattr(self, "reaped", False), "drain hid failed reap"
                return communicate(*args, **kwargs)

            self.proc.communicate = before_reap_only
            return self.proc

        def reap(self):
            assert not super().reap()
            self.reaped = True
            return "injected cleanup failure after real reap"

        def close(self):
            super().close()
            self.closed = True

    monkeypatch.setattr(sys.modules[__name__], "ProcessContainer", ReportFailedCleanup)
    code = "import time; time.sleep(60)" if timeout < 1 else "pass"
    with pytest.raises(RuntimeError, match="injected cleanup failure") as raised:
        _run_install_command([sys.executable, "-c", code], env=os.environ,
                             timeout=timeout, capture_output=True, text=True)
    if timeout < 1:
        assert isinstance(raised.value.__cause__, subprocess.TimeoutExpired)
        assert raised.value.__cause__.timeout == timeout
    else:
        assert raised.value.__cause__ is None
    assert containers[0].closed and containers[0].proc.returncode is not None
    assert containers[0].proc.stdout.closed and containers[0].proc.stderr.closed


@pytest.mark.serial
@pytest.mark.parametrize("text", [False, True], ids=["bytes", "text"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_install_timeout_drains_initially_missing_output(monkeypatch, text, newline):
    """Real pipes, with the Windows initial-timeout shape also exercised on POSIX."""
    class MissingInitialOutput(ProcessContainer):
        def spawn(self, *args, **kwargs):
            proc = super().spawn(*args, **kwargs)
            communicate = proc.communicate

            def windows_timeout_shape(*args, **kwargs):
                try:
                    return communicate(*args, **kwargs)
                except subprocess.TimeoutExpired as exc:
                    exc.stdout = exc.stderr = None
                    raise

            proc.communicate = windows_timeout_shape
            return proc

    monkeypatch.setattr(sys.modules[__name__], "ProcessContainer", MissingInitialOutput)
    stdout = ("stdout" + newline).encode()
    stderr = ("stderr" + newline).encode()
    code = (f"import sys,time; sys.stdout.buffer.write({stdout!r}); sys.stdout.buffer.flush(); "
            f"sys.stderr.buffer.write({stderr!r}); sys.stderr.buffer.flush(); time.sleep(60)")
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        _run_install_command([sys.executable, "-c", code], env=os.environ,
                             timeout=3, capture_output=True, text=text)
    assert raised.value.timeout == 3
    assert raised.value.stdout == ("stdout\n" if text else stdout)
    assert raised.value.stderr == ("stderr\n" if text else stderr)
