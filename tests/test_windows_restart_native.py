"""Native Windows proof of parent-handle inheritance, venv identity and stdio.

These child scripts exercise production restart, the captured bootstrap handover,
parent observation and Panic's successor stop. They do not start the application
server or its other owners.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import site
import sys
import time
import venv

import pytest


pytestmark = [
    pytest.mark.serial,
    pytest.mark.skipif(sys.platform != "win32", reason="native Win32 handle inheritance"),
]

_STDIO_AND_RECEIPTS = '''import ctypes, ctypes.wintypes, json, os, pathlib, sys
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetStdHandle.argtypes = (ctypes.wintypes.DWORD,)
kernel.GetStdHandle.restype = ctypes.wintypes.HANDLE
kernel.SetStdHandle.argtypes = (ctypes.wintypes.DWORD, ctypes.wintypes.HANDLE)
kernel.SetStdHandle.restype = ctypes.wintypes.BOOL
kernel.GetFileType.argtypes = (ctypes.wintypes.HANDLE,)
kernel.GetFileType.restype = ctypes.wintypes.DWORD

def standard_streams():
    handles = {name: kernel.GetStdHandle(number)
               for name, number in (("stdin", -10), ("stdout", -11), ("stderr", -12))}
    return {"handles": handles,
            "types": {name: kernel.GetFileType(handle) if handle else None
                      for name, handle in handles.items()},
            "stdin_absent": sys.stdin is None, "stderr_absent": sys.stderr is None}

def write_receipt(name, facts):
    path = pathlib.Path(name)
    pending = path.with_suffix(".writing")
    pending.write_text(json.dumps(facts), encoding="utf-8")
    pending.replace(path)
'''

_SUCCESSOR = _STDIO_AND_RECEIPTS + '''from ouroboros import delegate_recovery as recovery
from ouroboros.config import DATA_DIR
from ouroboros.platform_layer import process_start_time
parent = recovery.observe_restart_parent()
recovery._ack_direct_exec_successor(DATA_DIR)
transaction = recovery._read_restart_transaction(DATA_DIR, "native-proof")
streams = standard_streams()
input_text = None if sys.stdin is None else sys.stdin.read()
print("successor-stdout", flush=True)
if sys.stderr is not None:
    print("successor-stderr", file=sys.stderr, flush=True)
write_receipt(sys.argv[1], {"parent": parent, "pid": os.getpid(),
    "birth": process_start_time(os.getpid()), "prefix": sys.prefix, "executable": sys.executable,
    "transaction": transaction,
    "handle_env": os.environ.get("OUROBOROS_PLANNED_RESTART_PARENT_HANDLE"),
    "stdio": streams, "input_text": input_text})
'''

_PARENT = _STDIO_AND_RECEIPTS + '''import logging
from ouroboros import server_control, delegate_recovery as recovery
from ouroboros.config import DATA_DIR
from ouroboros.platform_layer import process_start_time
if sys.argv[5] == "absent":
    for name, fd, number in (("stdin", 0, -10), ("stderr", 2, -12)):
        os.close(fd)
        setattr(sys, name, None)
        setattr(sys, "__" + name + "__", None)
        if not kernel.SetStdHandle(number, None):
            raise ctypes.WinError(ctypes.get_last_error())
args = list(sys.argv)
os.environ[recovery.PLANNED_RESTART_TRANSACTION_ENV] = "native-proof"
recovery._write_restart_transaction(DATA_DIR, {"transaction_id": "native-proof", "status": "prepared",
    "supervisor_pid": os.getpid(), "task_ids": []})
bootstrap_root = os.environ.pop("OUROBOROS_NATIVE_BOOTSTRAP_ROOT", "")
if bootstrap_root:
    os.environ["PYTHONPATH"] = bootstrap_root + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["OUROBOROS_NATIVE_BODY_HANDOVER"] = "1"
sys.argv = args[1:3]
server_control.restart_current_process("127.0.0.1", 8765, repo_dir=pathlib.Path.cwd(),
    log=logging.getLogger("native"))
[child] = server_control._restart_successors
write_receipt(args[3], {"pid": os.getpid(), "birth": process_start_time(os.getpid()),
    "child_pid": child.pid, "prefix": sys.prefix, "executable": sys.executable,
    "stdio": standard_streams()})
os._exit(int(args[4]))
'''


@pytest.mark.parametrize("parent_exit", [42, 99])
@pytest.mark.parametrize("stdio_mode", ["redirected", "absent"])
@pytest.mark.parametrize("body_bootstrap", [False, True], ids=["direct", "body-bootstrap"])
def test_native_restart_inherits_exact_parent_venv_and_streams(tmp_path, parent_exit, stdio_mode, body_bootstrap):
    from ouroboros.process_containment import ProcessContainer

    venv_dir = tmp_path / "restart venv with spaces"
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(venv_dir)
    interpreter = venv_dir / "Scripts" / "python.exe"
    child_script, parent_script = tmp_path / "successor.py", tmp_path / "parent.py"
    child_script.write_text(_SUCCESSOR, encoding="utf-8")
    parent_script.write_text(_PARENT, encoding="utf-8")
    receipt, facts = tmp_path / "successor.json", tmp_path / "parent.json"
    input_file, output_file = tmp_path / "stdin.txt", tmp_path / "stdout.txt"
    input_file.write_text("successor-stdin\n", encoding="utf-8")
    # The temporary venv keeps the runner's dependencies even when pytest itself
    # runs in a dependency-only venv rather than the base installation.
    product_root = Path(__file__).resolve().parents[1]
    import_roots = [str(product_root), *site.getsitepackages()]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(import_roots),
           "OUROBOROS_DATA_DIR": str(tmp_path / "data")}
    if body_bootstrap:
        checkout, helper = tmp_path / "adopting checkout", tmp_path / "captured helper"
        package = checkout / "ouroboros"
        package.mkdir(parents=True)
        (checkout / ".git").mkdir()
        helper.mkdir()
        # The production package hook runs before any product module. Extend its
        # package path only so the miniature checkout can then find real modules.
        (package / "__init__.py").write_text(
            f"__path__.append({str(product_root / 'ouroboros')!r})\n"
            + (product_root / "ouroboros/__init__.py").read_text(encoding="utf-8"), encoding="utf-8")
        (checkout / ".git/ouroboros-body-adoption").write_text(str(helper), encoding="utf-8")
        (helper / "handoff.json").write_text(json.dumps({"data_dir": env["OUROBOROS_DATA_DIR"]}),
                                            encoding="utf-8")
        (helper / "captured.py").write_bytes((product_root / "ouroboros/body_switch.py").read_bytes())
        # Invoke the captured handover once through the real hook. Switching Git
        # phases is covered portably; these native cases prove its OS transfer.
        (helper / "switch.py").write_text('''import os
if os.environ.pop("OUROBOROS_NATIVE_BODY_HANDOVER", "") == "1":
    captured = os.path.join(os.path.dirname(__file__), "captured.py")
    namespace = {"__name__": "native_captured_helper", "__file__": __file__}
    with open(captured, "rb") as source:
        exec(compile(source.read(), captured, "exec"), namespace)
    namespace["_handover"]()
''', encoding="utf-8")
        env["OUROBOROS_NATIVE_BOOTSTRAP_ROOT"] = str(checkout)
    container, parent = ProcessContainer(), None
    try:
        with input_file.open("r", encoding="utf-8") as stdin, output_file.open("w", encoding="utf-8") as stdout:
            # Windows admission is suspended until the Job owns the parent,
            # so timeout cleanup covers even an unreported successor.
            parent = container.spawn(
                [str(interpreter), str(parent_script), str(child_script), str(receipt), str(facts),
                 str(parent_exit), stdio_mode],
                # Passing the file twice gives stderr its own native duplicate:
                # closing fd 2 must not close an alias of stdout's OS handle.
                env=env, stdin=stdin, stdout=stdout, stderr=stdout,
            )
            assert parent.wait(timeout=30) == parent_exit
            deadline = time.monotonic() + 30
            while not receipt.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert receipt.exists(), output_file.read_text(encoding="utf-8")
            observed = json.loads(receipt.read_text(encoding="utf-8"))
            parent_facts = json.loads(facts.read_text(encoding="utf-8"))
            assert observed["parent"] == {
                "pid": parent_facts["pid"], "birth": parent_facts["birth"], "exit_code": parent_exit,
            }
            assert (observed["pid"] == parent_facts["child_pid"]) is (not body_bootstrap)
            assert observed["birth"] and observed["handle_env"] is None
            transaction = observed["transaction"]
            assert (transaction["supervisor_pid"], transaction["supervisor_birth"]) == (
                parent_facts["pid"], parent_facts["birth"])
            if body_bootstrap and parent_exit == 99:
                # Panic does not authorize rebinding to the final generation.
                assert transaction["successor_pid"] == parent_facts["child_pid"]
                assert transaction["successor_birth"]
            else:
                assert (transaction["successor_pid"], transaction["successor_birth"]) == (
                    observed["pid"], observed["birth"])
            if parent_exit == 42:
                assert transaction["status"] == "normal_exit_acknowledged"
                assert transaction["ack_source"] == "windows_direct_parent_handle"
            else:
                assert transaction["status"] == "prepared"
            for process in (parent_facts, observed):
                assert Path(process["prefix"]).resolve() == venv_dir.resolve()
                assert Path(process["executable"]).resolve() == interpreter.resolve()
                streams = process["stdio"]
                assert streams["types"]["stdout"] == 1  # FILE_TYPE_DISK, never a synthesized pipe.
                for name in ("stdin", "stderr"):
                    assert streams[name + "_absent"] is (stdio_mode == "absent")
                    if stdio_mode == "absent":
                        assert streams["handles"][name] is None
                        assert streams["types"][name] is None
                    else:
                        assert streams["handles"][name]
                        assert streams["types"][name] == 1
            assert observed["input_text"] == (None if stdio_mode == "absent" else "successor-stdin\n")
            output = output_file.read_text(encoding="utf-8")
            assert "successor-stdout" in output
            assert ("successor-stderr" in output) is (stdio_mode == "redirected")
    finally:
        cleanup_error = container.reap()
        if parent is not None:
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=10)
        assert not cleanup_error, cleanup_error


_LIVE_SUCCESSOR = _STDIO_AND_RECEIPTS + '''import time
write_receipt(sys.argv[1], {"pid": os.getpid(), "prefix": sys.prefix, "executable": sys.executable})
while True:
    time.sleep(1)
'''

_PANIC_PARENT = _STDIO_AND_RECEIPTS + '''import logging, time
from ouroboros import server_control, delegate_recovery as recovery
os.environ.pop(recovery.PLANNED_RESTART_TRANSACTION_ENV, None)
os.environ.pop(recovery.PLANNED_RESTART_PARENT_ENV, None)
args = list(sys.argv)
sys.argv = args[1:3]
server_control.restart_current_process("127.0.0.1", 8765, repo_dir=pathlib.Path.cwd(),
    log=logging.getLogger("native-panic"))
[child] = server_control._restart_successors
receipt = pathlib.Path(args[2])
deadline = time.monotonic() + 30
while not receipt.exists() and time.monotonic() < deadline:
    time.sleep(0.05)
assert receipt.exists(), "successor did not start"
facts = json.loads(receipt.read_text(encoding="utf-8"))
assert child.pid == facts["pid"], "Panic holds the redirector instead of its server child"
assert child.poll() is None
[request] = server_control.stop_restart_successor()
assert request["requested"], request
exit_code = child.wait(timeout=10)
write_receipt(args[3], {"held_pid": child.pid, "successor": facts, "exit_code": exit_code,
    "prefix": sys.prefix, "executable": sys.executable})
'''


def test_native_no_transaction_venv_panic_stops_the_actual_successor(tmp_path):
    from ouroboros.process_containment import ProcessContainer

    venv_dir = tmp_path / "ordinary restart venv with spaces"
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(venv_dir)
    interpreter = venv_dir / "Scripts/python.exe"
    successor_script, parent_script = tmp_path / "successor.py", tmp_path / "parent.py"
    successor_script.write_text(_LIVE_SUCCESSOR, encoding="utf-8")
    parent_script.write_text(_PANIC_PARENT, encoding="utf-8")
    receipt, facts, output = tmp_path / "successor.json", tmp_path / "parent.json", tmp_path / "output.txt"
    import_roots = [str(Path(__file__).resolve().parents[1]), *site.getsitepackages()]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(import_roots),
           "OUROBOROS_DATA_DIR": str(tmp_path / "data")}
    container, parent = ProcessContainer(), None
    try:
        with output.open("w", encoding="utf-8") as stream:
            parent = container.spawn([str(interpreter), str(parent_script), str(successor_script),
                                      str(receipt), str(facts)], env=env, stdout=stream, stderr=stream)
            assert parent.wait(timeout=60) == 0, output.read_text(encoding="utf-8")
        observed = json.loads(facts.read_text(encoding="utf-8"))
        assert observed["held_pid"] == observed["successor"]["pid"]
        assert observed["exit_code"] != 0
        for process in (observed, observed["successor"]):
            assert Path(process["prefix"]).resolve() == venv_dir.resolve()
            assert Path(process["executable"]).resolve() == interpreter.resolve()
    finally:
        cleanup_error = container.reap()
        if parent is not None:
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=10)
        assert not cleanup_error, cleanup_error
