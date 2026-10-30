"""The optional persistent ``--runtime-venv`` and the managed bootstrap's mount refusal.

A real venv is created at its final path, recorded ``preparing`` before provisioning and
``ready`` only after the lock landed; an unrelated directory, a venv of another Python and
a venv whose ``pyvenv.cfg`` or interpreter disagree are refused untouched with the
recovery step; an interrupted preparation resumes additively. The launcher selects the
venv's lexical interpreter and PATH before bootstrap and world generation, gives the cold
install its own bound and marks it ready only after the install. Bootstrap refuses a
mount point without a checkout and moves nothing.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from ouroboros import runtime_venv
from ouroboros.runtime_venv import MARKER_NAME, RuntimeVenv, RuntimeVenvRefused, mark_ready, prepare_runtime_venv

CURRENT = f"{sys.version_info.major}.{sys.version_info.minor}"


def _tree(root: pathlib.Path) -> dict:
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
            for path in sorted(root.rglob("*"))}


class Run:
    """subprocess.run with a record of every venv creation."""

    def __init__(self, interpreter_version: str | None = None):
        self.creations = 0
        self.interpreter_version = interpreter_version

    def __call__(self, cmd, **kwargs):
        if cmd[1:3] == ["-m", "venv"]:
            self.creations += 1
        if self.interpreter_version is not None and cmd[1] == "-c":
            return subprocess.CompletedProcess(cmd, 0, f"{self.interpreter_version} /x\n", "")
        return subprocess.run(cmd, **kwargs)


@pytest.mark.serial
def test_a_real_venv_is_prepared_once_then_reused_ready(tmp_path):
    root = tmp_path / "runtime volume" / "venv"
    run = Run()
    venv = prepare_runtime_venv(root, run=run)
    assert run.creations == 1 and venv.ready is False
    assert json.loads((root / MARKER_NAME).read_text())["state"] == "preparing"
    assert pathlib.Path(venv.python).parent == venv.bin_dir and pathlib.Path(venv.python).exists()
    probe = subprocess.run([venv.python, "-c", "import sys; print(sys.prefix != sys.base_prefix)"],
                           capture_output=True, text=True, timeout=60)
    assert probe.stdout.strip() == "True", "the lexical interpreter activates the venv"
    assert mark_ready(venv).ready is True
    again = prepare_runtime_venv(root, run=run)
    assert again.ready is True and run.creations == 1, "a ready venv is reused, never recreated"


def test_an_unrelated_directory_is_refused_untouched(tmp_path):
    root = tmp_path / "venv"
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "keep.txt").write_text("owner data", encoding="utf-8")
    before = _tree(root)
    with pytest.raises(RuntimeVenvRefused, match="did not create.*left untouched.*move that directory aside"):
        prepare_runtime_venv(root, run=Run())
    assert _tree(root) == before


@pytest.mark.parametrize("marker,problem", [
    ({"schema_version": 1, "python": "2.7", "state": "ready"}, "created for Python 2.7"),
    ({"schema_version": 9, "python": CURRENT, "state": "ready"}, "unknown schema"),
    ({"schema_version": 1, "python": CURRENT, "state": "rebuilding"}, "recorded state"),
    ({"schema_version": 1, "python": CURRENT, "state": ["preparing"]}, "recorded state"),
])
def test_an_established_venv_that_cannot_be_trusted_is_refused_untouched(tmp_path, marker, problem):
    root = tmp_path / "venv"
    root.mkdir()
    (root / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    (root / "pyvenv.cfg").write_text(f"version = {CURRENT}.0\n", encoding="utf-8")
    before = _tree(root)
    run = Run()
    with pytest.raises(RuntimeVenvRefused, match=problem):
        prepare_runtime_venv(root, run=run)
    assert _tree(root) == before and run.creations == 0


@pytest.mark.parametrize("cfg_version,interpreter_version,label", [
    ("3.9.18", CURRENT, "pyvenv.cfg"),
    (f"{CURRENT}.4", "3.9", "its interpreter"),
])
def test_an_image_changed_python_cannot_impersonate_the_venvs_abi(tmp_path, cfg_version, interpreter_version, label):
    root = tmp_path / "venv"
    bin_dir, python = runtime_venv._layout(root)
    bin_dir.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    (root / MARKER_NAME).write_text(json.dumps({"schema_version": 1, "python": CURRENT, "state": "ready"}))
    (root / "pyvenv.cfg").write_text(f"home = /usr/bin\nversion = {cfg_version}\n", encoding="utf-8")
    with pytest.raises(RuntimeVenvRefused, match=f"{label} reports Python"):
        prepare_runtime_venv(root, run=Run(interpreter_version))


def test_an_interrupted_preparation_resumes_additively(tmp_path):
    root = tmp_path / "venv"
    root.mkdir()
    (root / MARKER_NAME).write_text(json.dumps({"schema_version": 1, "python": CURRENT, "state": "preparing"}))
    (root / "partial-download").write_text("kept", encoding="utf-8")

    class Resume(Run):
        def __call__(self, cmd, **kwargs):
            if cmd[1:3] == ["-m", "venv"]:
                self.creations += 1
                bin_dir, python = runtime_venv._layout(root)
                bin_dir.mkdir(parents=True)
                python.write_text("", encoding="utf-8")
                (root / "pyvenv.cfg").write_text(f"version = {CURRENT}.1\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 0, f"{CURRENT} {root}\n", "")

    venv = prepare_runtime_venv(root, run=Resume())
    assert venv.ready is False and (root / "partial-download").read_text() == "kept"


def test_a_relative_path_is_refused(tmp_path):
    with pytest.raises(RuntimeVenvRefused, match="absolute"):
        prepare_runtime_venv("relative/venv", run=Run())


def test_venv_creation_uses_the_plain_interpreter_not_the_app_entry(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    plain = "/app/Contents/Frameworks/python-standalone/bin/python3"
    monkeypatch.setattr(runtime_venv.sys, "executable", "/app/Contents/MacOS/ouroboros-shell")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["-m", "venv"]:
            bin_dir, python = runtime_venv._layout(root)
            bin_dir.mkdir()
            python.write_text("")
            (root / "pyvenv.cfg").write_text(f"version = {CURRENT}.0\n")
        return subprocess.CompletedProcess(cmd, 0, f"{CURRENT} {root}\n", "")

    prepare_runtime_venv(root, run=run, base_python=plain)
    assert calls[0] == [plain, "-m", "venv", str(root)]
    assert all(cmd[0] != sys.executable for cmd in calls)


@pytest.fixture
def managed_launcher(monkeypatch, tmp_path):
    import atexit

    import launcher

    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(launcher, "_headless", False)
    monkeypatch.setattr(launcher, "_external_ui", False)
    monkeypatch.setattr(launcher, "_external_host_update", None)
    monkeypatch.setattr(launcher, "_runtime_venv", None)
    monkeypatch.setattr(launcher, "EMBEDDED_PYTHON", launcher.EMBEDDED_PYTHON)
    monkeypatch.setattr(launcher, "acquire_pid_lock", lambda: True)
    monkeypatch.setattr(launcher, "check_git", lambda: True)
    monkeypatch.setattr(atexit, "register", lambda *args: None)
    monkeypatch.setenv("PATH", "/usr/bin")
    return launcher


class ReachedLifecycle(Exception):
    pass


# Both PATH separators on one host: the foreign one is simulated and qualifies no native launch there.
@pytest.mark.parametrize("separator", [":", ";"], ids=["posix", "windows-separator"])
def test_the_launcher_runs_body_installs_and_handoff_in_the_runtime_venv(managed_launcher, monkeypatch, tmp_path,
                                                                        separator):
    launcher = managed_launcher
    monkeypatch.setattr(launcher.os, "pathsep", separator)
    root = tmp_path / "body" / "venv"
    prepared = RuntimeVenv(root=root, python=str(root / "bin" / "python"), bin_dir=root / "bin", ready=False)
    seen = {}
    monkeypatch.setattr(launcher, "prepare_runtime_venv", lambda path, **kw: seen.setdefault("path", path) and prepared)

    def bootstrap():
        context = launcher._bootstrap_context()
        seen.update(python=context.embedded_python, timeout=context.deps_timeout_sec,
                    path_env=launcher.os.environ["PATH"])
        return True

    monkeypatch.setattr(launcher, "bootstrap_repo", bootstrap)
    monkeypatch.setattr(launcher, "_install_deps", lambda: seen.setdefault("installed", True))
    monkeypatch.setattr(launcher, "mark_ready", lambda venv: seen.setdefault("ready", venv) and venv)

    def stop():
        raise ReachedLifecycle()

    monkeypatch.setattr(launcher, "_prepare_first_run_settings", stop)
    with pytest.raises(ReachedLifecycle):
        launcher.main(["--no-ui", "--runtime-venv", str(root)])
    assert seen["path"] == root and seen["python"] == prepared.python
    assert seen["path_env"] == separator.join([str(prepared.bin_dir), "/usr/bin"])
    assert seen["timeout"] == runtime_venv.PREPARATION_DEPS_TIMEOUT_SEC
    assert seen["installed"] is True and seen["ready"] is prepared
    assert launcher.EMBEDDED_PYTHON == prepared.python


@pytest.mark.parametrize("entry", ["source", "source-venv", "macos-entry", "frozen-app"])
def test_launcher_venv_creation_keeps_the_running_python_abi(managed_launcher, monkeypatch, tmp_path, entry):
    """A downloaded bundled Python is not the interpreter of an ordinary source launch.

    Run the real launcher-to-preparation seam and its marker/validation; only Python
    subprocesses are simulated, including the mismatching interpreter's actual ABI.
    """
    launcher = managed_launcher
    packaged = entry in {"macos-entry", "frozen-app"}
    version = "3.10" if packaged else "3.11"
    bundled = str(tmp_path / "downloaded-python-3.10" / "python")
    source_python = str(tmp_path / "source-python-3.11" / "python")
    executable = bundled if packaged else source_python
    if entry in {"macos-entry", "frozen-app", "source-venv"}:
        executable = str(tmp_path / entry / "launcher")
    expected = bundled if packaged else source_python
    monkeypatch.setattr(launcher.sys, "frozen", entry == "frozen-app", raising=False)
    monkeypatch.setattr(launcher.sys, "executable", executable)
    monkeypatch.setattr(launcher.sys, "_base_executable", expected, raising=False)
    monkeypatch.setattr(launcher.sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(launcher, "EMBEDDED_PYTHON", bundled)
    monkeypatch.setattr(runtime_venv, "_major_minor", lambda: version)
    root = tmp_path / "body-venv"
    calls = []
    created_version = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["-m", "venv"]:
            created_version.append("3.10" if cmd[0] == bundled else "3.11")
            bin_dir, python = runtime_venv._layout(root)
            bin_dir.mkdir()
            python.write_text("simulated interpreter", encoding="utf-8")
            (root / "pyvenv.cfg").write_text(f"version = {created_version[0]}.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, f"{created_version[0]} {root}\n", "")

    monkeypatch.setattr(launcher, "prepare_runtime_venv",
                        lambda path, **kw: prepare_runtime_venv(path, run=run, base_python=kw.get("base_python")))
    monkeypatch.setattr(launcher, "bootstrap_repo", lambda: True)
    monkeypatch.setattr(launcher, "_install_deps", lambda: True)
    monkeypatch.setattr(launcher, "_prepare_first_run_settings",
                        lambda: (_ for _ in ()).throw(ReachedLifecycle()))
    with pytest.raises(ReachedLifecycle):
        launcher.main(["--no-ui", "--runtime-venv", str(root)])
    assert calls[0] == [expected, "-m", "venv", str(root)]
    assert created_version == [version]
    assert json.loads((root / MARKER_NAME).read_text(encoding="utf-8"))["python"] == version
    assert launcher._runtime_venv.ready


def test_a_refused_runtime_venv_stops_before_bootstrap(managed_launcher, monkeypatch, tmp_path):
    launcher = managed_launcher

    def refuse(path, **kw):
        raise RuntimeVenvRefused("fixture refusal")

    monkeypatch.setattr(launcher, "prepare_runtime_venv", refuse)
    monkeypatch.setattr(launcher, "bootstrap_repo", lambda: pytest.fail("no bootstrap without the venv"))
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--no-ui", "--runtime-venv", str(tmp_path / "venv")])
    assert stopped.value.code == 1


def test_bootstrap_refuses_a_mount_point_without_a_checkout_and_moves_nothing(tmp_path, monkeypatch):
    import logging

    from ouroboros import launcher_bootstrap

    repo = tmp_path / "body" / "repo"
    repo.mkdir(parents=True)
    (repo / "volume-content.txt").write_text("keep", encoding="utf-8")
    before = _tree(repo)
    context = launcher_bootstrap.BootstrapContext(
        bundle_dir=tmp_path / "seed", repo_dir=repo, data_dir=tmp_path / "data", settings_path=tmp_path / "s.json",
        embedded_python=sys.executable, app_version="7.6.0", hidden_run=subprocess.run,
        save_settings=lambda _s: None, log=logging.getLogger("mount-test"))
    monkeypatch.setattr(launcher_bootstrap, "load_bundle_manifest", lambda _context: {})
    monkeypatch.setattr(launcher_bootstrap.os.path, "ismount", lambda path: pathlib.Path(path) == repo)
    with pytest.raises(RuntimeError, match="mount point without a checkout; nothing was moved"):
        launcher_bootstrap.ensure_managed_repo(context)
    assert _tree(repo) == before and not (tmp_path / "data" / "archive").exists()


@pytest.mark.serial
def test_preparing_python_without_pip_is_repaired_and_keeps_an_installed_package(tmp_path):
    import hashlib
    import shutil
    import zipfile

    root = tmp_path / "venv"
    venv = prepare_runtime_venv(root)
    wheel = tmp_path / "sentinel-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in {
            "sentinel.py": "VALUE = 'keep this installed package'\n",
            "sentinel-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: sentinel\nVersion: 1.0\n",
            "sentinel-1.0.dist-info/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            "sentinel-1.0.dist-info/RECORD": "",
        }.items():
            archive.writestr(name, value)
    subprocess.run([venv.python, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
                   check=True, capture_output=True, timeout=60)
    site = pathlib.Path(subprocess.check_output(
        [venv.python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    sentinel = site / "sentinel.py"
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    # The precise interruption: the executable/cfg and earlier packages exist,
    # but ensurepip has not installed pip. Remove only this disposable venv's pip.
    shutil.rmtree(site / "pip")
    for metadata in site.glob("pip-*.dist-info"):
        shutil.rmtree(metadata)
    missing = subprocess.run([venv.python, "-m", "pip", "--version"], capture_output=True, text=True)
    assert missing.returncode != 0 and "No module named pip" in missing.stderr
    repaired = prepare_runtime_venv(root)
    pip = subprocess.run([repaired.python, "-m", "pip", "--version"], capture_output=True, text=True)
    assert pip.returncode == 0, pip.stderr
    assert not repaired.ready and hashlib.sha256(sentinel.read_bytes()).hexdigest() == before
    probe = subprocess.check_output([repaired.python, "-c", "import sentinel; from importlib.metadata import version; "
                                    "print(sentinel.VALUE, version('sentinel'))"], text=True)
    assert probe.strip() == "keep this installed package 1.0"


@pytest.mark.parametrize("problem", ["cfg", "interpreter", "missing-cfg"])
def test_a_preparing_venv_cannot_hide_incompatible_or_unknown_python(tmp_path, problem):
    root = tmp_path / "venv"
    bin_dir, python = runtime_venv._layout(root)
    bin_dir.mkdir(parents=True)
    python.write_text("interrupted python")
    (root / MARKER_NAME).write_text(json.dumps({"schema_version": 1, "python": CURRENT, "state": "preparing"}))
    if problem != "missing-cfg":
        (root / "pyvenv.cfg").write_text(f"version = {'2.7' if problem == 'cfg' else CURRENT}\n")
    before = _tree(root)
    run = Run("2.7" if problem == "interpreter" else CURRENT)
    with pytest.raises(RuntimeVenvRefused, match="left untouched"):
        prepare_runtime_venv(root, run=run)
    assert run.creations == 0 and _tree(root) == before


def test_directory_creation_failure_is_a_typed_refusal(tmp_path):
    parent = tmp_path / "blocked"
    parent.write_text("a file, not a writable directory")
    with pytest.raises(RuntimeVenvRefused, match="cannot create its directory.*may have created or changed"):
        prepare_runtime_venv(parent / "venv")
    assert parent.read_text() == "a file, not a writable directory"


@pytest.mark.parametrize("state", ["preparing", "ready"])
def test_marker_failure_reports_retained_partial_effects(tmp_path, monkeypatch, state):
    from ouroboros import utils

    root = tmp_path / "venv"

    def cannot_write(*args, **kwargs):
        raise PermissionError("read-only fixture")

    monkeypatch.setattr(utils, "atomic_write_json", cannot_write)
    if state == "ready":
        root.mkdir()
        (root / MARKER_NAME).write_text("preparing evidence")
    with pytest.raises(RuntimeVenvRefused, match=f"cannot record {state}.*files; they were retained") as refused:
        if state == "preparing":
            prepare_runtime_venv(root)
        else:
            mark_ready(RuntimeVenv(root, str(root / "bin/python"), root / "bin", False))
    assert "left untouched" not in str(refused.value) and root.is_dir()
    if state == "ready":
        assert (root / MARKER_NAME).read_text() == "preparing evidence"


def test_a_failed_ready_marker_stops_the_launcher_before_starting_the_body(managed_launcher, monkeypatch, tmp_path, capsys):
    launcher = managed_launcher
    root = tmp_path / "venv"
    prepared = RuntimeVenv(root, str(root / "bin/python"), root / "bin", False)
    monkeypatch.setattr(launcher, "prepare_runtime_venv", lambda *a, **kw: prepared)
    monkeypatch.setattr(launcher, "bootstrap_repo", lambda: True)
    monkeypatch.setattr(launcher, "_install_deps", lambda: True)
    def refuse(_venv):
        raise RuntimeVenvRefused("cannot record ready; files retained")
    monkeypatch.setattr(launcher, "mark_ready", refuse)
    monkeypatch.setattr(launcher, "_prepare_first_run_settings", lambda: pytest.fail("no body after marker failure"))
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--no-ui", "--runtime-venv", str(root)])
    assert stopped.value.code == 1
    assert "Ouroboros did not start: cannot record ready" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# An Update's clone pointed at a staged app's Python: real venvs, real interpreter starts
# --------------------------------------------------------------------------- #
_STARTED = "import json, sys; print(json.dumps([sys.prefix, sys.base_prefix]))"


def _started(python: pathlib.Path) -> list:
    """``[sys.prefix, sys.base_prefix]`` of a real start of ``python``, resolved (isolated, no bytecode written)."""
    done = subprocess.run([str(python), "-I", "-B", "-c", _STARTED], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return [os.path.realpath(path) for path in json.loads(done.stdout)]


def _real_venv(python: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    """A venv as CPython's ``venv`` makes it (no pip, so nothing is fetched)."""
    subprocess.run([str(python), "-m", "venv", "--without-pip", str(root)], check=True, capture_output=True,
                   timeout=300)
    return root


def _state(venv: pathlib.Path) -> tuple:
    """Every link of the venv's ``bin/`` with its exact text, and its ``pyvenv.cfg`` bytes."""
    return ({link.name: os.readlink(link) for link in sorted((venv / "bin").iterdir()) if link.is_symlink()},
            (venv / "pyvenv.cfg").read_bytes())


def _clone(venv: pathlib.Path) -> pathlib.Path:
    """The update's private copy, made as ``update_prepare.prepare_generation_environment`` makes it."""
    clone = venv.with_name(f"{venv.name}.update-probe")
    if sys.platform != "darwin" or subprocess.run(["cp", "-c", "-R", str(venv), str(clone)],
                                                  capture_output=True, timeout=300).returncode != 0:
        shutil.rmtree(clone, ignore_errors=True)
        shutil.copytree(venv, clone, symlinks=True)
    return clone


def _app_cpython():
    """The app's own standalone CPython (``Contents/Frameworks/python-standalone/bin/python3``) when it runs these
    tests, else ``None``: a relocatable runtime can stand for two app installations at once."""
    from ouroboros.platform_layer import interpreter_is_embedded

    python = pathlib.Path(sys.executable)
    if sys.platform != "darwin" or not interpreter_is_embedded(sys.executable):
        return None
    if python.parents[2].name != "Frameworks":
        return None
    return python


def _app_runtimes(tmp_path: pathlib.Path) -> tuple:
    """``(apps, pythons, homes)``: two app installations, each a copy-on-write copy of the app's own CPython as the
    app lays it out: ``Contents/Frameworks/python-standalone``, the ``libpython`` its ``bin/python3`` loads from
    ``Contents/Frameworks``, and ``Contents/Resources/python-standalone``, where the bundle keeps the standard
    library its code links to. Each starts as its own runtime."""
    contents = _app_cpython().parents[3]
    parts = [pathlib.Path("Frameworks", "python-standalone"), pathlib.Path("Resources", "python-standalone"),
             *(path.relative_to(contents) for path in (contents / "Frameworks").glob("libpython3*.dylib"))]
    apps = {"installed": tmp_path / "Applications" / "Ouroboros.app",
            "staged": tmp_path / "Applications" / ".Ouroboros-update-9.9.9-probe" / "Ouroboros.app"}
    for app in apps.values():
        for part in parts:
            if (contents / part).exists():
                (app / "Contents" / part).parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["cp", "-c", "-R", str(contents / part), str(app / "Contents" / part)], check=True,
                               timeout=300)
    pythons = {name: app / "Contents" / "Frameworks" / "python-standalone" / "bin" / "python3"
               for name, app in apps.items()}
    homes = {name: os.path.realpath(python.parent.parent) for name, python in pythons.items()}
    assert _started(pythons["staged"])[1] == homes["staged"] != homes["installed"]  # two runtimes, each its own
    return apps, pythons, homes


@pytest.mark.serial
@pytest.mark.skipif(_app_cpython() is None, reason="needs the app's own standalone CPython running these tests")
def test_a_real_venv_of_the_apps_python_runs_the_staged_apps_python_only_while_pointed(tmp_path):
    """The supported case on real interpreters, nothing installed and no network, over two copies of the app's own
    CPython (``_app_runtimes``). The owner's venv is what the launcher makes with the installed app's ``python3``:
    CPython names that Python in ``bin/python3`` and makes ``python`` and ``python3.X`` relative aliases of it. The
    update's clone starts the installed Python, the staged one through every alias while pointed, then the
    installed one again with every link and ``pyvenv.cfg`` exactly as before; the owner's venv never changes."""
    apps, pythons, homes = _app_runtimes(tmp_path)
    owner = _real_venv(pythons["installed"], tmp_path / "owner-venv")
    minor = f"python3.{sys.version_info.minor}"
    assert _state(owner)[0] == {"python": "python3", "python3": str(pythons["installed"]), minor: "python3"}
    before = _state(owner)
    clone = _clone(owner)
    assert _started(clone / "bin" / "python") == [os.path.realpath(clone), homes["installed"]]

    saved = runtime_venv.point_base(clone, str(apps["installed"]), str(apps["staged"]))

    assert runtime_venv.base_interpreter(clone) == str(pythons["staged"])
    for alias in ("python", "python3", minor):
        assert _started(clone / "bin" / alias) == [os.path.realpath(clone), homes["staged"]], alias
    assert _state(owner) == before
    runtime_venv.restore_base(clone, saved)
    assert _started(clone / "bin" / "python") == [os.path.realpath(clone), homes["installed"]]
    assert _state(clone) == before and _state(owner) == before


@pytest.mark.serial
@pytest.mark.skipif(_app_cpython() is None, reason="needs the app's own standalone CPython running these tests")
def test_the_switch_probe_runs_a_real_venv_of_the_apps_python_on_the_staged_apps_python_and_changes_nothing(
        tmp_path):
    """The switch's import probe (``body_switch._probe``) of a venv of the installed app's Python, on real
    interpreters, nothing installed and no network. Before the exchange the selected venv's ``bin/python`` still
    names the installed app's Python, which the exchange then replaces with the staged one: the probe starts the
    staged runtime over that venv's own packages, imports a module only the venv carries and one a ``.pth`` line of
    its site-packages adds, and leaves every link and ``pyvenv.cfg`` of the venv exactly as they were."""
    from ouroboros import body_switch

    _apps, pythons, homes = _app_runtimes(tmp_path)
    venv = _real_venv(pythons["installed"], tmp_path / "owner-venv")
    site = venv / "lib" / f"python3.{sys.version_info.minor}" / "site-packages"
    (site / "venv_only.py").write_text("WHERE = __file__\n")
    added = tmp_path / "added-by-pth"
    added.mkdir()
    (added / "pth_only.py").write_text("ADDED = True\n")
    (site / "probe-extra.pth").write_text(f"{added}\n")
    before = _state(venv)
    out = tmp_path / "probe.json"
    code = ("import json, os, sys, pth_only, venv_only; "  # resolved while the probe's scratch venv exists
            f"open({str(out)!r}, 'w').write(json.dumps([sys.prefix, sys.base_prefix, "
            "os.path.realpath(venv_only.WHERE)]))")
    handoff = {"id": "probe", "probe_command": [str(venv / "bin" / "python"), "-c", code],
               "environment": {"kind": "venv", "state": "selected", "current": str(venv),
                               "staged_python": str(pythons["staged"])}}

    assert body_switch._probe(str(tmp_path), handoff) == ""

    prefix, base_prefix, where = json.loads(out.read_text())
    assert os.path.realpath(base_prefix) == homes["staged"]  # the runtime the exchange puts in place
    assert where == os.path.realpath(site / "venv_only.py")  # the venv's own packages
    assert "ouroboros-switch-probe-" in prefix and not os.path.exists(prefix)  # a scratch venv, gone with the probe
    assert _state(venv) == before
    assert _started(venv / "bin" / "python") == [os.path.realpath(venv), homes["installed"]]


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="a Windows venv copies its interpreter instead of linking it")
def test_a_real_venv_of_another_python_is_refused_and_left_as_it_was(tmp_path):
    """The marker records a Python version, not where that Python lives. A real venv of the Python running these
    tests is not the app's, so pointing it at another app is refused, nothing in it changes and it still starts."""
    venv = _real_venv(pathlib.Path(sys.executable), tmp_path / "venv")
    before = _state(venv)
    apps = [tmp_path / "Applications" / "Ouroboros.app",
            tmp_path / "Applications" / ".Ouroboros-update-9.9.9-probe" / "Ouroboros.app"]
    for app in apps:
        python = app / "Contents" / "Frameworks" / "python-standalone" / "bin" / "python3"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")

    with pytest.raises(RuntimeVenvRefused, match="does not run the Python of"):
        runtime_venv.point_base(venv, str(apps[0]), str(apps[1]))

    assert _state(venv) == before
    assert _started(venv / "bin" / "python")[0] == os.path.realpath(venv)


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="a Windows venv copies its interpreter instead of linking it")
def test_pointing_a_real_venv_follows_its_aliases_and_a_failure_part_way_leaves_it_exactly_as_it_was(
        tmp_path, monkeypatch):
    """A real venv of the Python running these tests, its installation standing for the installed app and a mirror
    of that installation for the staged one. Pointing follows ``bin/python`` through CPython's relative aliases to
    the one link that names a Python, changes that link and ``pyvenv.cfg``'s ``home`` only, and the venv starts
    through every alias while pointed and after. A failure part-way (``pyvenv.cfg`` cannot be written) writes back
    the link it had already changed: the venv is exactly as it was."""
    venv = _real_venv(pathlib.Path(sys.executable), tmp_path / "venv")
    before = _state(venv)
    named = {name: text for name, text in before[0].items() if os.path.isabs(text)}
    assert len(named) == 1 and len(before[0]) > 1, before  # one link names the Python; the others alias it
    [(name, base)] = named.items()
    installed = pathlib.Path(base).parent.parent  # the installation whose bin/ holds that Python
    staged = tmp_path / "staged-installation"
    (staged / "bin").mkdir(parents=True)
    (staged / "bin" / pathlib.Path(base).name).symlink_to(base)
    if (installed / "lib").is_dir():
        (staged / "lib").symlink_to(installed / "lib")  # where a non-framework Python finds its library from home

    saved = runtime_venv.point_base(venv, str(installed), str(staged))

    links, config = _state(venv)
    assert {key: text for key, text in links.items() if key != name} == {
        key: text for key, text in before[0].items() if key != name}  # the aliases are never rewritten
    assert links[name] == runtime_venv.base_interpreter(venv) == str(staged / "bin" / pathlib.Path(base).name)
    assert f"home = {staged / 'bin'}".encode() in config
    for alias in before[0]:
        assert _started(venv / "bin" / alias)[0] == os.path.realpath(venv), alias
    runtime_venv.restore_base(venv, saved)
    assert _state(venv) == before

    def full(path, text):
        raise OSError(28, "No space left on device", str(path))

    monkeypatch.setattr(runtime_venv, "_replace_text", full)
    with pytest.raises(OSError, match="No space left on device"):
        runtime_venv.point_base(venv, str(installed), str(staged))
    assert _state(venv) == before
    assert _started(venv / "bin" / "python")[0] == os.path.realpath(venv)
