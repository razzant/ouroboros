"""Fixture builders: the qualification package's wheels and the official target commits (stdlib only).

Fixture, never product: ``ouroboros-qual-probe`` is a tiny package whose versions 1.0-5.0 stand for a
dependency lock change. It carries a real extension module (the interpreter's own ``_bz2.pyd``, loaded as
``ouroboros_qual_probe._bz2``) so a process importing a generation maps a native image from inside it,
and a console script (``ouroboros-qual-tool``) that pip writes into ``Scripts``. Every target commit is
the frozen candidate plus exactly two declared edits: the pin in ``requirements-runtime.lock`` and two
lines in ``ouroboros/gateway/router.py`` (imported by every server and by the switch's import probe) that
import the fixture and leave a witness. Target T4 instead imports ``stable_api``, which only 4.0 lacks:
a syntactically valid tree that cannot run on its own new dependency.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import os
import pathlib
import subprocess
import zipfile

DIST, PACKAGE = "ouroboros-qual-probe", "ouroboros_qual_probe"
WITNESS_ENV, STUB_ENV = "WINQUAL_WITNESS_DIR", "WINQUAL_CHECKS_STUB"
LOCK, ROUTER = "requirements-runtime.lock", "ouroboros/gateway/router.py"
WITHOUT_STABLE_API = "4.0"
# name -> (parent, fixture version, router edit). S0 is the serving base every mode starts from.
PLAN = (("S0", None, "1.0", "witness"), ("T2", "S0", "2.0", "witness"), ("T3", "T2", "3.0", "witness"),
        ("T4", "T3", "4.0", "fault"), ("T5", "S0", "5.0", "witness"))
VERSIONS = {name: version for name, _parent, version, _edit in PLAN}
STUB_LABEL = "windows-qualification checks-only stub: workspace checks were NOT run"  # = stub/sitecustomize.py LABEL
WITNESS_LINES = ("import ouroboros_qual_probe as _winqual_probe  # Windows qualification fixture, not product code",
                 '_winqual_probe.witness("ouroboros.gateway.router")')
FAULT_LINES = ("from ouroboros_qual_probe import stable_api as _winqual_stable_api  # fixture: 4.0 removed it",)
IDENTITY = {"GIT_AUTHOR_NAME": "Windows qualification fixture", "GIT_AUTHOR_EMAIL": "winqual@invalid",
            "GIT_COMMITTER_NAME": "Windows qualification fixture", "GIT_COMMITTER_EMAIL": "winqual@invalid",
            "GIT_AUTHOR_DATE": "2026-10-29T00:00:00Z", "GIT_COMMITTER_DATE": "2026-10-29T00:00:00Z"}

INIT = '''"""Windows Update qualification fixture {version} (equipment, never product code)."""
VERSION = "{version}"
{stable_api}

def _kernel():
    import ctypes
    from ctypes import wintypes as w
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = w.HANDLE
    k.GetProcessTimes.argtypes = (w.HANDLE,) + (ctypes.POINTER(w.FILETIME),) * 4
    k.GetProcessTimes.restype = w.BOOL
    k.GetModuleHandleW.argtypes, k.GetModuleHandleW.restype = (w.LPCWSTR,), w.HMODULE
    k.K32GetMappedFileNameW.argtypes = (w.HANDLE, ctypes.c_void_p, w.LPWSTR, w.DWORD)
    k.K32GetMappedFileNameW.restype = w.DWORD
    k.CreateFileW.argtypes = (w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE)
    k.CreateFileW.restype = w.HANDLE
    k.GetFinalPathNameByHandleW.argtypes = (w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD)
    k.GetFinalPathNameByHandleW.restype = w.DWORD
    k.CloseHandle.argtypes, k.CloseHandle.restype = (w.HANDLE,), w.BOOL
    return ctypes, w, k


def _birth():
    """This process's creation FILETIME as the product's bindings spell it; "" off Windows."""
    import sys
    if sys.platform != "win32":
        return ""
    ctypes, w, k = _kernel()
    times = [w.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(k.GetCurrentProcess(), *[ctypes.byref(t) for t in times]):
        return ""
    return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)


def _final(path):
    """The path a handle opened now resolves to (Windows); the plain realpath elsewhere."""
    import os
    import sys
    if sys.platform != "win32":
        return os.path.realpath(path)
    ctypes, w, k = _kernel()
    handle = k.CreateFileW(str(path), 0, 7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        return None
    try:
        buf = ctypes.create_unicode_buffer(32768)
        return buf.value if k.GetFinalPathNameByHandleW(handle, buf, 32768, 0) else None
    finally:
        k.CloseHandle(handle)


def native():
    """Load this package's real extension module; name the file the loader mapped."""
    import os
    import sys
    try:
        from . import _bz2
    except ImportError as exc:
        return {{"status": "absent", "error": str(exc)}}
    compressor = _bz2.BZ2Compressor()
    data = compressor.compress(b"winqual") + compressor.flush()
    facts = {{"status": "loaded", "file": _bz2.__file__, "realpath": os.path.realpath(_bz2.__file__),
             "roundtrip": _bz2.BZ2Decompressor().decompress(data) == b"winqual"}}
    if sys.platform == "win32":
        ctypes, w, k = _kernel()
        module = k.GetModuleHandleW(_bz2.__file__)
        buf = ctypes.create_unicode_buffer(32768)
        if module and k.K32GetMappedFileNameW(k.GetCurrentProcess(), module, buf, 32768):
            facts["mapped_device"] = buf.value
        facts["expected_device"] = _device(_bz2.__file__)
    return facts


def _device(path):
    import sys
    if sys.platform != "win32":
        return None
    ctypes, w, k = _kernel()
    handle = k.CreateFileW(str(path), 0, 7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        return None
    try:
        buf = ctypes.create_unicode_buffer(32768)
        return buf.value if k.GetFinalPathNameByHandleW(handle, buf, 32768, 2) else None
    finally:
        k.CloseHandle(handle)


def facts(where, with_native=True):
    import os
    import site
    import sys
    import time
    try:
        cwd = os.getcwd()
    except OSError as exc:
        cwd = "error: %s" % exc
    return {{"schema": "winqual-witness/1", "where": where, "pid": os.getpid(), "ppid": os.getppid(),
            "birth": _birth(), "time_ns": time.time_ns(), "executable": sys.executable,
            "base_executable": getattr(sys, "_base_executable", None), "prefix": sys.prefix,
            "base_prefix": sys.base_prefix, "userbase": site.getuserbase(), "user_site": site.ENABLE_USER_SITE,
            "venv_launcher": os.environ.get("__PYVENV_LAUNCHER__"), "argv": list(sys.argv),
            "orig_argv": list(getattr(sys, "orig_argv", []) or []), "cwd": cwd, "cwd_final": _final("."),
            "package": {{"version": VERSION, "file": __file__, "realpath": os.path.realpath(__file__)}},
            "native": native() if with_native else None}}


def witness(where):
    """One JSON file per call in $WINQUAL_WITNESS_DIR; never raises into the importing product."""
    import json
    import os
    try:
        directory = os.environ.get("WINQUAL_WITNESS_DIR")
        if not directory:
            return None
        value = facts(where)
        path = os.path.join(directory, "pkg-%d-%d.json" % (value["pid"], value["time_ns"]))
        with open(path, "x", encoding="utf-8") as fh:
            json.dump(value, fh)
        return path
    except Exception:
        return None


def main(argv=None):
    """``report``: print facts. ``hold``: acknowledge, then answer stdin requests until ``exit``/EOF."""
    import argparse
    import json
    import os
    import sys
    parser = argparse.ArgumentParser(prog="ouroboros-qual-tool")
    parser.add_argument("mode", choices=("report", "hold"))
    parser.add_argument("--native", action="store_true", help="map the package's extension module")
    parser.add_argument("--cwd", help="hold this directory as the working directory")
    args = parser.parse_args(argv)
    if args.native:
        native()
    if args.cwd:
        os.chdir(args.cwd)

    def send(value):
        sys.stdout.write(json.dumps(value) + "\\n")
        sys.stdout.flush()

    def view(event, **extra):
        return {{**facts("tool:" + args.mode, with_native=args.native), "event": event, **extra}}

    if args.mode == "report":
        send(view("report"))
        return 0
    send(view("ready"))
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("op") == "exit":
            send({{"event": "exit", "id": request.get("id")}})
            return 0
        send(view("report", id=request.get("id")))
    return 0
'''
STABLE_API = "\ndef stable_api():\n    return VERSION\n"
MAIN = "from . import main\n\nraise SystemExit(main())\n"


def package_files(version):
    return {f"{PACKAGE}/__init__.py": INIT.format(version=version,
                                                  stable_api="" if version == WITHOUT_STABLE_API else STABLE_API),
            f"{PACKAGE}/__main__.py": MAIN}


def build_wheel(directory, version, pyd=None):
    """A wheel of ``version``; with ``pyd`` (bytes of the target interpreter's ``_bz2.pyd``) a
    ``cp310-cp310-win_amd64`` wheel whose package maps that real native image."""
    tag = "cp310-cp310-win_amd64" if pyd is not None else "py3-none-any"
    info = f"{PACKAGE}-{version}.dist-info"
    files = {name: text.encode("utf-8") for name, text in package_files(version).items()}
    if pyd is not None:
        files[f"{PACKAGE}/_bz2.pyd"] = bytes(pyd)
    files[f"{info}/METADATA"] = f"Metadata-Version: 2.1\nName: {DIST}\nVersion: {version}\n".encode()
    files[f"{info}/WHEEL"] = (f"Wheel-Version: 1.0\nGenerator: winqual\nRoot-Is-Purelib: "
                              f"{'false' if pyd is not None else 'true'}\nTag: {tag}\n").encode()
    files[f"{info}/entry_points.txt"] = f"[console_scripts]\nouroboros-qual-tool = {PACKAGE}:main\n".encode()
    record = [f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')},"
              f"{len(data)}" for name, data in files.items()]
    files[f"{info}/RECORD"] = ("\n".join(record + [f"{info}/RECORD,,"]) + "\n").encode()
    path = pathlib.Path(directory) / f"{PACKAGE}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data)
    return path


def insert_after_future(text, lines, filename=ROUTER):
    """``lines`` after the module docstring and every ``from __future__`` import, then compiled:
    a fixture that cannot even compile would fail for the wrong reason (Astra finding 6)."""
    body = ast.parse(text, filename).body
    end, index = 0, 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        end, index = body[0].end_lineno, 1
    while index < len(body) and isinstance(body[index], ast.ImportFrom) and body[index].module == "__future__":
        end, index = body[index].end_lineno, index + 1
    source = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    if end and not source[end - 1].endswith(("\n", "\r")):
        source[end - 1] += newline
    result = "".join(source[:end]) + "".join(line + newline for line in lines) + "".join(source[end:])
    compile(result, filename, "exec", dont_inherit=True)
    return result


def pin(lock_text, version):
    """The lock with exactly one fixture pin at ``version`` (refuses a lock already naming the fixture)."""
    lines = lock_text.splitlines()
    if any(line.split("==")[0].strip().lower() == DIST for line in lines):
        raise ValueError("the candidate lock already names the fixture package")
    newline = "\r\n" if "\r\n" in lock_text else "\n"
    return newline.join(lines + [f"{DIST}=={version}"]) + newline


def _git(repo, *args, data=None):
    env = {**os.environ, **IDENTITY, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_TERMINAL_PROMPT": "0"}
    done = subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True, env=env, timeout=600)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {done.stderr.decode(errors='replace')[-800:]}")
    return done.stdout.decode("utf-8", "replace").strip()


def build_commits(candidate_repo, candidate_sha, work):
    """Clone the candidate and add the PLAN commits; returns ``{name: sha}`` plus the source facts."""
    work = pathlib.Path(work)
    _git(work.parent, "clone", "--quiet", "--no-checkout", str(candidate_repo), str(work))
    _git(work, "checkout", "--quiet", "--detach", candidate_sha)
    router = subprocess.run(["git", "-C", str(work), "show", f"{candidate_sha}:{ROUTER}"], capture_output=True,
                            check=True).stdout.decode("utf-8")
    lock = subprocess.run(["git", "-C", str(work), "show", f"{candidate_sha}:{LOCK}"], capture_output=True,
                          check=True).stdout.decode("utf-8")
    edits = {"witness": insert_after_future(router, WITNESS_LINES),
             "fault": insert_after_future(router, FAULT_LINES + WITNESS_LINES)}
    commits = {}
    for name, parent, version, edit in PLAN:
        _git(work, "checkout", "--quiet", "--detach", commits.get(parent, candidate_sha))
        (work / ROUTER).write_bytes(edits[edit].encode("utf-8"))
        (work / LOCK).write_bytes(pin(lock, version).encode("utf-8"))
        _git(work, "add", "--", ROUTER, LOCK)
        _git(work, "commit", "--quiet", "--no-verify", "-m", f"winqual fixture {name}: {DIST}=={version} ({edit})")
        commits[name] = _git(work, "rev-parse", "HEAD")
        changed = set(_git(work, "diff", "--name-only", candidate_sha, commits[name]).split())
        if changed != {ROUTER, LOCK}:  # against the frozen candidate: exactly the two declared edits
            raise RuntimeError(f"fixture {name} differs from the candidate in {sorted(changed)}, not only its edits")
        _git(work, "branch", "--force", f"winqual/{name}", commits[name])
    return commits
