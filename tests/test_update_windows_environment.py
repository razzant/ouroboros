"""The unified Update's environment seam on Windows, checked on any host where it is pure bytes and paths.

A Windows venv names its interpreter ``Scripts/python.exe`` and pip writes each console script as
distlib's launcher: a stub, one ``#!<interpreter>`` line and a zip archive found from the end of the
file. Preparation runs pip with the generation's own interpreter, so after the switch renames the
generation into the fixed path those launchers name a path that no longer exists (measured on NTFS:
the console entry point exits 1 while a fresh Python reads the new packages). The switch helper
rewrites only that line (``body_switch._retarget_scripts``). These launchers are real: pip's own
vendored distlib writes them from its shipped ``t64.exe`` stub, with only distlib's ``os.name`` branch
taken as Windows takes it. Running them is the native Windows qualification's part.
"""
from __future__ import annotations

import io
import os
import pathlib
import struct
import sys
import zipfile
import zipimport

import pytest

from ouroboros import body_switch, runtime_venv
from supervisor import update_checks, update_prepare

distlib = pytest.importorskip("pip._vendor.distlib")
scripts = pytest.importorskip("pip._vendor.distlib.scripts")


def _launcher(directory: pathlib.Path, interpreter: str, name: str = "toycli") -> pathlib.Path:
    """``<name>.exe`` exactly as pip writes it on Windows for an environment whose interpreter is ``interpreter``."""
    stubs = pathlib.Path(distlib.__path__[0])
    if not (stubs / "t64.exe").is_file():
        pytest.skip("this pip ships no Windows launcher stubs")
    maker = scripts.ScriptMaker(None, str(directory), add_launchers=True)
    maker._is_nt = True  # what distlib decides from os.name on Windows
    maker.variants = {""}
    original = (scripts.ScriptMaker.__dict__.get("_get_launcher"), scripts.ScriptMaker._build_shebang,
                scripts.get_executable)
    try:
        scripts.ScriptMaker._get_launcher = lambda self, kind: (stubs / f"{kind}64.exe").read_bytes()
        scripts.ScriptMaker._build_shebang = lambda self, executable, post: b"#!" + executable + post + b"\n"  # nt
        scripts.get_executable = lambda: interpreter  # pip leaves it to sys.executable; distlib quotes a space
        return pathlib.Path(maker.make(f"{name} = toypkg.cli:main")[0])
    finally:
        if original[0] is None:
            del scripts.ScriptMaker._get_launcher
        else:
            scripts.ScriptMaker._get_launcher = original[0]
        scripts.ScriptMaker._build_shebang, scripts.get_executable = original[1], original[2]


def _shebang(data: bytes) -> bytes:
    """The line distlib's launcher runs: the end-of-central-directory record from the end, the archive start
    from its sizes, the last ``#!`` before it, up to the end of that line."""
    eocd = data.rfind(b"PK\x05\x06")
    size, offset = struct.unpack("<II", data[eocd + 12:eocd + 20])
    start = data.rfind(b"#!", 0, eocd - size - offset)
    return data[start:min(at for at in (data.find(b"\r", start), data.find(b"\n", start)) if at >= 0)]


@pytest.mark.parametrize("parent", [r"C:\Users\run ner\Ouroboros\data\state", r"D:\a\_temp\data\state"],
                         ids=["path-with-spaces", "plain-path"])
def test_windows_console_launchers_name_the_fixed_path_after_the_switch_and_keep_their_archive(tmp_path, parent):
    generation, current = parent + r"\venv.update-c1", parent + r"\venv"
    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    written = _launcher(venv / "Scripts", generation + r"\Scripts\python.exe")
    other = _launcher(venv / "Scripts", r"C:\Python310\python.exe", name="elsewhere")  # another interpreter: kept
    (venv / "Scripts" / "python.exe").write_bytes((pathlib.Path(distlib.__path__[0]) / "t64.exe").read_bytes())
    legacy = venv / "Scripts" / "legacy-script.py"
    legacy.write_bytes(b'#!"' + generation.encode() + b'\\Scripts\\python.exe" -E\r\nimport toypkg\r\n')
    before = {path.name: path.read_bytes() for path in (venv / "Scripts").iterdir()}
    quoted = " " in parent
    assert _shebang(before[written.name]) == b"#!" + (f'"{generation}\\Scripts\\python.exe"' if quoted
                                                      else f"{generation}\\Scripts\\python.exe").encode()

    # The generation's own spelling may differ in case and slashes from the recorded path: matched as a Windows path.
    body_switch._retarget_scripts(str(venv), generation.lower().replace("\\", "/"), current, windows=True)

    after = written.read_bytes()
    target = f"{current}\\Scripts\\python.exe"
    assert _shebang(after) == b"#!" + (f'"{target}"' if quoted else target).encode()
    line = before[written.name].rfind(b"#!")
    assert after[:line] == before[written.name][:line]  # the stub byte for byte
    archive = zipfile.ZipFile(io.BytesIO(after)).read("__main__.py")
    assert archive == zipfile.ZipFile(io.BytesIO(before[written.name])).read("__main__.py")
    assert zipimport.zipimporter(str(written)).get_code("__main__") is not None  # how ``python toycli.exe`` runs it
    assert other.read_bytes() == before[other.name]  # names another interpreter
    assert (venv / "Scripts" / "python.exe").read_bytes() == before["python.exe"]  # no archive: not a launcher
    assert legacy.read_bytes() == b'#!"' + current.encode() + b'\\Scripts\\python.exe" -E\r\nimport toypkg\r\n'
    again = {path.name: path.read_bytes() for path in (venv / "Scripts").iterdir()}
    body_switch._retarget_scripts(str(venv), generation, current, windows=True)  # a resumed selection: nothing left
    assert {path.name: path.read_bytes() for path in (venv / "Scripts").iterdir()} == again


def test_posix_console_scripts_keep_their_relocation(tmp_path):
    """``bin/``'s two forms, unchanged: a ``#!`` line and distlib's ``/bin/sh`` trampoline for a long path."""
    venv, old, new = tmp_path / "venv", "/data/state/venv.update-c1", "/data/state/venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "short").write_bytes(f"#!{old}/bin/python\nimport toypkg\n".encode())
    (venv / "bin" / "long").write_bytes(f"#!/bin/sh\n'''exec' \"{old}/bin/python\" \"$0\" \"$@\"\n' '''\n".encode())
    os.chmod(venv / "bin" / "short", 0o755)

    body_switch._retarget_scripts(str(venv), old, new, windows=False)

    assert (venv / "bin" / "short").read_bytes() == f"#!{new}/bin/python\nimport toypkg\n".encode()
    assert (venv / "bin" / "short").stat().st_mode & 0o777 == 0o755
    assert (venv / "bin" / "long").read_bytes().splitlines()[1] == f"'''exec' \"{new}/bin/python\" \"$0\" \"$@\"".encode()


def test_a_windows_generation_installs_checks_and_hands_over_on_its_scripts_interpreter(tmp_path, monkeypatch):
    """``Scripts/python.exe``, never a hardcoded ``bin/python``: preparation's install, the W' checks, the hand-over."""
    monkeypatch.setattr(runtime_venv, "IS_WINDOWS", True)
    monkeypatch.setattr(body_switch.sys, "platform", "win32")
    venv = tmp_path / "venv.update-c1"
    seen = []
    monkeypatch.setattr(update_checks, "run_tracked", lambda cmd, cwd, timeout, env=None: seen.append(cmd) or {
        "returncode": 0, "stdout": "", "stderr": ""})
    monkeypatch.setattr(runtime_venv, "system_site", lambda root, visible: None)

    update_prepare._install_lock(venv, tmp_path / "requirements-runtime.lock", "venv")
    update_prepare._install_lock(venv, tmp_path / "requirements-runtime.lock", "private")

    assert [cmd[0] for cmd in seen] == [str(venv / "Scripts" / "python.exe")] * 2
    row = {"environment": {"generation": {"kind": "venv", "user_base": str(venv)}}}
    assert update_checks._interpreters(row, {})[0] == str(venv / "Scripts" / "python.exe")
    assert body_switch._venv_python(str(venv)) == os.path.join(str(venv), "Scripts", "python.exe")


def _seam(monkeypatch, *, platform, targets, prefix, base, launcher=None):
    from ouroboros import platform_layer

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "prefix", prefix)
    monkeypatch.setattr(sys, "base_prefix", base)
    monkeypatch.setattr(platform_layer, "pip_install_target_args", lambda _python: targets)
    monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER", raising=False)
    monkeypatch.delenv(update_prepare.LAUNCHER_HANDOFF_ENV, raising=False)
    if launcher is not None:
        monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
        if launcher:
            monkeypatch.setenv(update_prepare.LAUNCHER_HANDOFF_ENV, "1")
    return update_prepare.serving_environment()


def test_windows_switches_its_packaged_user_site_and_venvs(tmp_path, monkeypatch):
    base = str(tmp_path / "python")
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "data" / "state" / "python-userbase"))
    kind, path = _seam(monkeypatch, platform="win32", targets=["--user"], prefix=base, base=base, launcher=True)
    assert (kind, path) == ("user_site", (tmp_path / "data" / "state" / "python-userbase").resolve())
    venv = str(tmp_path / "runtime-venv")
    assert _seam(monkeypatch, platform="win32", targets=[], prefix=venv, base=base, launcher=True) == (
        "venv", pathlib.Path(venv))


@pytest.mark.parametrize("platform", ["darwin", "win32"])
@pytest.mark.parametrize("launcher,expected", [(None, "private"), (True, "private"), (False, "")],
                         ids=["direct", "launcher-hands-over", "launcher-predating-the-hand-over"])
def test_a_bare_interpreter_gets_a_private_environment_under_a_launcher_that_hands_itself_over(
        tmp_path, monkeypatch, launcher, expected, platform):
    """A system Python gets its private environment on Windows too, run directly or by a launcher that hands over."""
    base = str(tmp_path / "python")
    kind, _path = _seam(monkeypatch, platform=platform, targets=[], prefix=base, base=base, launcher=launcher)
    assert kind == expected
    if not expected:
        assert "cannot restart itself onto new Python packages" in update_prepare._no_seam()


def _windows_private(tmp_path, monkeypatch, repo, *, executable_key=True):
    """A system Python's private environment as Windows' ``venv`` leaves it: ``Scripts/python.exe`` a copied
    redirector (never a link to the interpreter), ``pyvenv.cfg`` naming that interpreter, the checkout's marker."""
    import json

    monkeypatch.setattr(runtime_venv, "IS_WINDOWS", True)
    bare = os.path.realpath(sys.executable)
    data = tmp_path / "data"
    root = data / "state" / runtime_venv.PRIVATE_NAME
    (root / "Scripts").mkdir(parents=True)
    (root / "Scripts" / "python.exe").write_bytes(b"MZ redirector stand-in")
    named = f"executable = {bare}\n" if executable_key else ""
    (root / "pyvenv.cfg").write_text(f"home = {os.path.dirname(bare)}\ninclude-system-site-packages = false\n{named}")
    (root / runtime_venv.PRIVATE_MARKER).write_text(json.dumps({"repo_dir": os.path.realpath(repo)}))
    return bare, data, root


@pytest.mark.parametrize("executable_key", [True, False], ids=["executable", "home-only"])
def test_a_windows_private_environment_is_recognised_by_the_python_its_config_names(tmp_path, monkeypatch,
                                                                                     executable_key):
    repo = tmp_path / "repo"
    repo.mkdir()
    bare, data, root = _windows_private(tmp_path, monkeypatch, repo, executable_key=executable_key)
    if not executable_key:  # home-only: the interpreter is home's python.exe
        (pathlib.Path(bare).parent / "python.exe").exists() or pytest.skip("no python.exe beside this interpreter")
        bare = str(pathlib.Path(bare).parent / "python.exe")
    private = str(root / "Scripts" / "python.exe")
    assert runtime_venv.private_entry_interpreter(data, repo, bare) == private
    assert runtime_venv.dependency_python(data, repo, bare) == private  # installs go there
    other = tmp_path / "other-python.exe"
    other.write_bytes(b"")
    assert runtime_venv.private_entry_interpreter(data, repo, str(other)) == ""  # another interpreter runs as it is
    with pytest.raises(runtime_venv.RuntimeVenvRefused):
        runtime_venv.dependency_python(data, repo, str(other))  # and never installs into its own packages


def test_a_direct_windows_entry_runs_as_the_waited_on_child_in_its_private_environment(tmp_path, monkeypatch):
    """The package-init hook on a Windows system Python: no exec, so the unchanged command's process stays the
    waited-on parent of its entry, started as the base Python in the private venv (``body_switch._wait_as_parent``),
    and exits with the entry's status (a Panic's 99 here)."""
    import ast

    import ouroboros

    repo = pathlib.Path(ouroboros.__file__).resolve().parent.parent
    init = repo / "ouroboros" / "__init__.py"
    hook = next(node for node in ast.parse(init.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == "_private_environment_hook")
    namespace = {"__file__": str(init)}  # the hook deletes itself after the package's own import
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(init), "exec"), namespace)  # noqa: S102
    _bare, data, root = _windows_private(tmp_path, monkeypatch, repo)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.delenv("OUROBOROS_PRIVATE_ENVIRONMENT_ENTRY", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    waited, ended = [], []

    class Ended(BaseException):
        pass

    monkeypatch.setattr(body_switch, "_wait_as_parent", lambda argv, venv=None: waited.append((argv, venv)) or 99)
    monkeypatch.setattr(os, "_exit", lambda code: ended.append(code) or (_ for _ in ()).throw(Ended()))
    with pytest.raises(Ended):
        namespace["_private_environment_hook"]()
    assert waited == [([sys.executable, *sys.orig_argv[1:]], str(root / "Scripts" / "python.exe"))] and ended == [99]
    assert os.environ.pop("OUROBOROS_PRIVATE_ENVIRONMENT_ENTRY") == "1"  # its child reads it once: never a loop


def test_the_windows_waiting_parent_starts_a_venvs_entry_as_its_base_python(tmp_path, monkeypatch):
    """``_windows_handoff`` starts a named venv's entry as the base Python with ``__PYVENV_LAUNCHER__`` (one process,
    as the redirector would start it, minus the redirector), argv as a list (quoted by ``subprocess``)."""
    import types

    monkeypatch.setitem(sys.modules, "_winapi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace())
    monkeypatch.delenv(body_switch._RESTART_PARENT, raising=False)
    started = []
    monkeypatch.setattr(body_switch.subprocess, "Popen", lambda argv, env: started.append((argv, env)) or "child")
    venv = str(tmp_path / "Program Files" / "python-private" / "Scripts" / "python.exe")
    assert body_switch._windows_handoff([venv, "C:/repo with space/server.py", "--x"], venv) == "child"
    base = getattr(sys, "_base_executable", None) or sys.executable
    assert started[0][0] == [base, "C:/repo with space/server.py", "--x"]
    assert started[0][1]["__PYVENV_LAUNCHER__"] == venv


def test_a_windows_hand_over_into_a_private_environment_runs_its_base_python(tmp_path, monkeypatch):
    """After the switch, a direct entry hands over through the waiting parent with the selected venv named."""
    root = tmp_path / "python-private"
    (root / "Scripts").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = C:/Python312\n")
    monkeypatch.setattr(body_switch.sys, "platform", "win32")
    monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER", raising=False)
    waited = []

    class Ended(BaseException):
        pass

    monkeypatch.setattr(body_switch, "_wait_as_parent", lambda argv, venv=None: waited.append((argv, venv)) or 0)
    monkeypatch.setattr(body_switch.os, "_exit", lambda code: (_ for _ in ()).throw(Ended()))
    handoff = {"environment": {"kind": "private", "state": "selected", "current": str(root)}}
    with pytest.raises(Ended):
        body_switch._handover(handoff)
    python = os.path.join(str(root), "Scripts", "python.exe")
    assert waited == [([python, *sys.orig_argv[1:]], python)]


def test_a_windows_launcher_starts_a_private_entry_as_one_process_of_its_base_python(tmp_path, monkeypatch):
    """The launcher's server keeps its PID, Job and stop pipe: the base Python itself runs the private venv
    (``__PYVENV_LAUNCHER__``), never the venv's redirector, whose PID is not the server; a launcher that itself
    runs in that venv starts the server from the venv's base Python."""
    import launcher
    from ouroboros import launcher_bootstrap

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "server.py").write_text("", encoding="utf-8")
    bare, data, root = _windows_private(tmp_path, monkeypatch, repo)
    captured = {}

    class Started(BaseException):
        pass

    monkeypatch.setattr(launcher, "IS_WINDOWS", True)
    monkeypatch.setattr(launcher, "DATA_DIR", data)
    monkeypatch.setattr(launcher, "PORT_FILE", data / "state" / "server_port")
    monkeypatch.setattr(launcher, "REPO_DIR", repo)
    monkeypatch.setattr(launcher, "EMBEDDED_PYTHON", bare)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {})
    monkeypatch.setattr(launcher, "_apply_settings_to_env", lambda _s: None)
    monkeypatch.setattr(launcher, "_hidden_popen", lambda cmd, **kw: captured.update(cmd=cmd, **kw) or (
        _ for _ in ()).throw(Started()))
    with pytest.raises(Started):
        launcher.start_agent(port=1)
    assert captured["cmd"][0] == bare and captured["env"]["__PYVENV_LAUNCHER__"] == str(root / "Scripts" / "python.exe")

    monkeypatch.setattr(launcher_bootstrap.sys, "platform", "win32")
    monkeypatch.setattr(launcher_bootstrap.sys, "executable", str(root / "Scripts" / "python.exe"))
    monkeypatch.setattr(launcher_bootstrap.sys, "_base_executable", bare, raising=False)
    assert launcher_bootstrap.find_embedded_python(tmp_path / "no-bundle", macos=False) == bare
