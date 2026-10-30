"""A lock change's generation environment, executed: real pip, a real venv and user site, real consumers.

Preparation clones the serving packages and installs the new lock into the clone
(``update_prepare.prepare_generation_environment``); the switch helper renames the clone
into the one path every launcher starts the body with, and a return renames it back
(``body_switch._select_environment`` / ``_restore_environment``). Nothing here is faked
below those owners: pip installs offline from wheels built in the test, and each step is
proven by running what a generation runs from that path: its interpreter importing the
package, a console script pip wrote (for the venv, a path long enough that distlib writes
its ``/bin/sh`` trampoline instead of a plain shebang) and, for the venv, pip itself.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import zipfile

import pytest

from ouroboros import body_switch
from supervisor import update_prepare

pytestmark = pytest.mark.serial  # real interpreters and pip


def _wheel(folder: pathlib.Path, version: str, name: str = "toydep", tool: str = "toytool") -> None:
    """A pure wheel of ``name`` whose console script ``tool`` prints the version and the interpreter prefix."""
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": (f"VERSION = {version!r}\n\n\ndef main():\n    import sys\n"
                                "    print(VERSION, sys.prefix)\n"),
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist}/entry_points.txt": f"[console_scripts]\n{tool} = {name}:main\n",
    }
    record = [f"{path},sha256={base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b'=').decode()},"
              f"{len(text.encode())}" for path, text in files.items()] + [f"{dist}/RECORD,,"]
    with zipfile.ZipFile(folder / f"{name}-{version}-py3-none-any.whl", "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
        archive.writestr(f"{dist}/RECORD", "\n".join(record) + "\n")


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def _candidate(tmp_path) -> tuple:
    """A workspace whose composition pins toydep 2.0, as a prepared candidate's record names it."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "requirements-runtime.lock").write_text("toydep==2.0\n")
    _git(ws, "init", "-q")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "the new lock")
    commit = _git(ws, "rev-parse", "HEAD")
    row = {"candidate_id": "cand1", "path": str(ws), "repo_dir": str(ws),
           "layers": {"new_files": _git(ws, "rev-parse", "HEAD^{tree}")}}
    return row, commit


def _run(argv, env=None):
    done = subprocess.run([str(part) for part in argv], capture_output=True, text=True, timeout=300,
                          env={**os.environ, **(env or {})})
    assert done.returncode == 0, (argv, done.stdout, done.stderr)
    return done.stdout.strip()


@pytest.fixture
def offline_pip(tmp_path, monkeypatch):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for version in ("1.0", "2.0"):
        _wheel(wheels, version)
    for key, value in {"PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(wheels), "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                       "PIP_BREAK_SYSTEM_PACKAGES": "1",  # a scratch user base only
                       "PIP_REQUIRE_VIRTUALENV": "0"}.items():  # the user site case installs outside a venv
        monkeypatch.setenv(key, value)
    return wheels


def _handoff(helper: pathlib.Path, kind: str, current: pathlib.Path, generation: pathlib.Path) -> dict:
    """The environment part of the record ``authorize_update`` writes, the identities of what is where included."""
    helper.mkdir(exist_ok=True)
    return {"id": "adoption1", "phase": "switching",
            "environment": {"state": "prepared", "kind": kind, "current": str(current), "generation": str(generation),
                            "previous": f"{current}.previous-adoption1",
                            "identity": {"current": body_switch._path_identity(current),
                                         "generation": body_switch._path_identity(generation)}}}


def test_a_venv_generation_is_coherent_where_it_is_checked_and_where_it_is_selected(tmp_path, monkeypatch,
                                                                                    offline_pip):
    deep = tmp_path / "Application Support" / "serving environment"  # a space: distlib's /bin/sh trampoline
    deep.mkdir(parents=True)
    venv = deep / "venv"
    _run([sys.executable, "-m", "venv", venv])
    _wheel(offline_pip, "1.0", "keptdep", "kepttool")  # a package beside the lock, which the update leaves alone
    _run([venv / "bin" / "python", "-m", "pip", "install", "-q", "toydep==1.0", "keptdep==1.0"],
         {"PYTHONNOUSERSITE": "1"})
    assert _run([venv / "bin" / "toytool"]).split()[0] == "1.0"
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))
    row, commit = _candidate(tmp_path)

    generation = update_prepare.prepare_generation_environment(row, commit)

    clone = pathlib.Path(generation["user_base"])
    assert generation["kind"] == "venv" and generation["lock"] == "requirements-runtime.lock"
    script = (clone / "bin" / "toytool").read_text()
    assert " " in str(clone) and script.startswith("#!/bin/sh\n'''exec'")  # the trampoline form
    # Checks run the generation where it lives: its script names its own interpreter, not the serving one.
    assert _run([clone / "bin" / "toytool"]).split(" ", 1) == ["2.0", str(clone)]
    # A package the lock leaves as it was keeps the script cloned from the serving venv: pointed at the copy too.
    assert _run([clone / "bin" / "kepttool"]).split(" ", 1) == ["1.0", str(clone)]
    assert str(clone) in _run([clone / "bin" / "pip", "--version"])
    assert _run([venv / "bin" / "toytool"]).split()[0] == "1.0"  # the serving packages are untouched
    assert _run([venv / "bin" / "kepttool"]).split(" ", 1) == ["1.0", str(venv)]
    helper = tmp_path / "helper"
    handoff = _handoff(helper, "venv", venv, clone)

    body_switch._select_environment(str(helper), handoff)

    assert handoff["environment"]["state"] == "selected" and not clone.exists()
    assert _run([venv / "bin" / "toytool"]).split(" ", 1) == ["2.0", str(venv)]  # the retargeted trampoline
    assert _run([venv / "bin" / "kepttool"]).split(" ", 1) == ["1.0", str(venv)]  # back at the fixed path
    assert _run([venv / "bin" / "python", "-c", "import sys, toydep; print(toydep.VERSION, sys.prefix)"],
                {"PYTHONNOUSERSITE": "1"}).split(" ", 1) == ["2.0", str(venv)]
    assert "pip" in _run([venv / "bin" / "pip", "--version"])  # a script cloned from the serving venv
    assert json.loads((helper / "handoff.json").read_text())["environment"]["state"] == "selected"

    body_switch._restore_environment(str(helper), handoff)

    assert handoff["environment"]["state"] == "restored" and clone.is_dir()
    assert _run([venv / "bin" / "toytool"]).split(" ", 1) == ["1.0", str(venv)]  # the previous generation, intact


def _user_site_python() -> str:
    """The interpreter a packaged user site runs: this one, or the Python this run's venv runs, since inside a
    venv pip refuses ``--user`` and no user site is visible (``runtime_venv.base_interpreter``)."""
    from ouroboros.runtime_venv import base_interpreter

    if sys.prefix == sys.base_prefix:
        return sys.executable
    python = base_interpreter(sys.prefix) or getattr(sys, "_base_executable", "")
    # Not by resolved file: a venv's python links to the very binary its base runs, yet runs as the venv.
    if not python or pathlib.Path(sys.prefix) in pathlib.Path(python).parents:
        pytest.skip("this venv names no interpreter outside it to own a user site")
    return python


def test_a_user_site_generation_is_selected_and_returned_under_the_same_interpreter(tmp_path, monkeypatch,
                                                                                    offline_pip):
    monkeypatch.setenv("PYTHONNOUSERSITE", "")  # the packaged interpreter imports its user site (empty = unset)
    userbase = tmp_path / "state" / "python-userbase"
    serving = {"PYTHONUSERBASE": str(userbase)}
    python = _user_site_python()
    _run([python, "-m", "pip", "install", "--user", "-q", "toydep==1.0"], serving)
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("user_site", userbase))
    row, commit = _candidate(tmp_path)

    generation = update_prepare.prepare_generation_environment(row, commit, python=python)

    clone = pathlib.Path(generation["user_base"])
    probe = [python, "-c", "import toydep; print(toydep.VERSION)"]
    assert _run(probe, {"PYTHONUSERBASE": str(clone)}) == "2.0" and _run(probe, serving) == "1.0"
    helper = tmp_path / "helper"
    handoff = _handoff(helper, "user_site", userbase, clone)

    body_switch._select_environment(str(helper), handoff)

    assert _run(probe, serving) == "2.0" and _run([userbase / "bin" / "toytool"], serving).split()[0] == "2.0"
    body_switch._restore_environment(str(helper), handoff)
    assert _run(probe, serving) == "1.0" and _run([userbase / "bin" / "toytool"], serving).split()[0] == "1.0"


def test_a_package_installed_into_the_serving_venv_after_preparation_stales_the_candidate(tmp_path, monkeypatch,
                                                                                          offline_pip):
    """The local model runtime's install (``local_model.LocalModelManager._run_install``: pip into the serving
    interpreter's packages) after an Update that brings its own packages was prepared. Activating that candidate
    would switch in a copy taken before the install, and the package would be gone after the restart: it is
    ``stale:environment_changed`` instead. A fresh preparation copies the serving packages with it, activates,
    and after the real switch the package is still there beside the new lock."""
    import time

    from ouroboros import local_model
    from tests.test_update_activation import Scene

    scene = Scene(tmp_path, monkeypatch)
    venv = tmp_path / "serving-venv"
    _run([sys.executable, "-m", "venv", venv])
    _run([venv / "bin" / "python", "-m", "pip", "install", "-q", "toydep==1.0"], {"PYTHONNOUSERSITE": "1"})
    _wheel(offline_pip, "1.0", "keptdep", "kepttool")  # the runtime the owner installs meanwhile
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))
    scene.official({"requirements-runtime.lock": "toydep==2.0\n", "ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    first = scene.prepare()
    assert first["state"] == "ready" and first["environment"]["generation"]["kind"] == "venv", first.get("reason")

    manager = local_model.LocalModelManager()  # its own install path, pointed at this scene's serving interpreter
    monkeypatch.setattr(local_model, "_get_install_command", lambda: [
        str(venv / "bin" / "python"), "-m", "pip", "install", "-q", "keptdep==1.0"])
    monkeypatch.setattr(manager, "check_runtime", lambda: True)
    manager.install_runtime()
    deadline = time.monotonic() + 120
    while manager._runtime_status == "installing" and time.monotonic() < deadline:
        time.sleep(0.1)
    assert manager._runtime_status == "install_ok", manager._runtime_install_log

    why = update_prepare.activation_check(update_prepare.find(first["candidate_id"]))

    assert why.startswith(f"environment_changed: packages were installed or removed in {venv}"), why
    assert update_prepare.find(first["candidate_id"])["state"] == "stale"
    second = scene.prepare()
    assert second["state"] == "ready" and update_prepare.activation_check(second) == ""
    scene.activate(second)
    assert scene.boot().returncode == 0 and scene.handoff()["environment"]["state"] == "selected"
    shown = _run([venv / "bin" / "python", "-c", "import keptdep, toydep; print(keptdep.VERSION, toydep.VERSION)"],
                 {"PYTHONNOUSERSITE": "1"})
    assert shown == "1.0 2.0"  # the install was kept beside the new lock
