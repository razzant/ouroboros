"""Windows qualification equipment, NOT product code: placed at the disposable checkout's root, Git-ignored.

Python imports ``sitecustomize`` at interpreter start from the first ``sys.path`` entry that has one; the
launcher starts every server with ``PYTHONPATH`` set to the checkout, so the servers of this installation
import this file before any product module. It does two labelled things, each only when its variable is set
in the installation's environment (the qualification sets both; nothing else does):

* ``WINQUAL_WITNESS_DIR``: one JSON file per interpreter start: which process ran on which interpreter
  prefix (taken BEFORE the package-init hook can redirect anything) and under which parent.
* ``WINQUAL_CHECKS_STUB=1``: the CHECKS-ONLY STUB. ``supervisor.update_checks.workspace_checks`` is replaced,
  once that real module has executed, by a function that runs no tests and returns
  ``{"ok": True, "stub": LABEL, ...}``. Nothing else of preparation, activation, the switch, the probe or
  the launcher is replaced. A lock change would select a test battery far beyond a runner's budget; the
  verdicts read ``checks.stub`` back from the candidate record and report it.

An interpreter ``sitecustomize`` that this file shadows still runs first. Standard library only; it never
imports a third-party or product module and never raises into the interpreter start.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

LABEL = "windows-qualification checks-only stub: workspace checks were NOT run"
_HERE = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
_SHADOWED = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.realpath(p or os.curdir) != _HERE])
if _SHADOWED is not None and _SHADOWED.loader is not None:
    _SHADOWED.loader.exec_module(importlib.util.module_from_spec(_SHADOWED))


def _birth():
    if sys.platform != "win32":
        return ""
    import ctypes
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.GetProcessTimes.argtypes = (w.HANDLE,) + (ctypes.POINTER(w.FILETIME),) * 4
    kernel.GetProcessTimes.restype = w.BOOL
    times = [w.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), *[ctypes.byref(t) for t in times]):
        return ""
    return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)


def _start_witness(directory):
    value = {"schema": "winqual-start/1", "where": "interpreter_start", "pid": os.getpid(), "ppid": os.getppid(),
             "birth": _birth(), "time_ns": time.time_ns(), "executable": sys.executable,
             "base_executable": getattr(sys, "_base_executable", None), "prefix": sys.prefix,
             "base_prefix": sys.base_prefix, "venv_launcher": os.environ.get("__PYVENV_LAUNCHER__"),
             "orig_argv": list(getattr(sys, "orig_argv", []) or []), "path": list(sys.path)}
    with open(os.path.join(directory, "start-%d-%d.json" % (value["pid"], value["time_ns"])), "x",
              encoding="utf-8") as fh:
        json.dump(value, fh)


class _ChecksOnlyStub(importlib.abc.MetaPathFinder):
    """Replaces exactly ``supervisor.update_checks.workspace_checks`` after the real module ran."""

    def find_spec(self, name, path, target=None):
        if name != "supervisor.update_checks":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module

        def exec_module(module):
            original(module)
            real = getattr(module, "workspace_checks", None)

            def workspace_checks(row):
                return {"ok": True, "stub": LABEL, "candidate_id": row.get("candidate_id"),
                        "replaced": getattr(real, "__qualname__", repr(real)), "pid": os.getpid()}

            module.workspace_checks = workspace_checks

        spec.loader.exec_module = exec_module
        return spec


try:
    if os.environ.get("WINQUAL_WITNESS_DIR"):
        _start_witness(os.environ["WINQUAL_WITNESS_DIR"])
except Exception:
    pass
if os.environ.get("WINQUAL_CHECKS_STUB") == "1":
    sys.meta_path.insert(0, _ChecksOnlyStub())
