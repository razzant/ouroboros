"""A bare interpreter's private environment, executed: the real interpreter, a real venv, real pip, real entries.

A server that runs on a system Python without a launcher has no seam an Update can switch:
its boot installs the lock into that interpreter's own packages. Preparation now makes the new
generation a venv of exactly that interpreter beside ``<data>/state/python-private``
(``update_prepare.prepare_generation_environment``): every pin of the lock installed in it while
the interpreter's own packages are hidden, which then stay visible underneath. The switch selects
it there (the first time nothing was there before), the probe and the hand-over run its
interpreter, and the package-init hook runs every later entry of that checkout on that
interpreter there, after the handoff is retired too. A return parks it, and the interpreter's own
packages serve again. Nothing is installed into them. The interpreter is this run's own, or the
one its venv was made from (``BARE``): the owner's unchanged command is ``BARE server.py``.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import zipfile

import pytest

from ouroboros import body_adoption, body_switch, config
from supervisor import update_prepare
from tests.body_candidate_support import REPO, git, make_serving
from tests.test_update_activation import Scene
from tests.test_update_environment import _candidate, _run, offline_pip  # noqa: F401 -- the offline wheels fixture

pytestmark = [pytest.mark.serial,  # real interpreters, pip and cold entries
              pytest.mark.skipif(sys.platform == "win32", reason="POSIX layout and exec (Windows: its own tests)")]

def _bare_interpreter() -> str:
    """The interpreter a venv of this run was made from (or this one, run bare), proven bare before any entry: it
    reports no venv of its own. ``sys._base_executable`` names it on most Pythons, but CPython 3.10 POSIX under a
    uv venv reports the venv's own ``bin/python`` there; the venv's links (``runtime_venv.base_interpreter``) and
    their real path name it too."""
    from ouroboros.runtime_venv import base_interpreter

    names = [getattr(sys, "_base_executable", ""), sys.executable]
    if sys.prefix != sys.base_prefix:
        names[1:1] = [base_interpreter(sys.prefix), os.path.realpath(sys.executable)]
    for name in dict.fromkeys(name for name in names if name and os.path.isfile(name)):
        shown = subprocess.run([name, "-I", "-c", "import sys; print(sys.prefix == sys.base_prefix)"],
                               capture_output=True, text=True, timeout=60)
        if shown.returncode == 0 and shown.stdout.strip() == "True":
            return name
    return ""


BARE = _bare_interpreter()  # the owner's unchanged command is ``BARE server.py``


@pytest.fixture(autouse=True)
def _bare_resolved():
    if not BARE:  # never a skip: these cases exist to run on the owner's bare interpreter
        pytest.fail("no bare interpreter was found behind this run (sys._base_executable, the venv's links, "
                    "sys.executable all report a venv)")

REPORTING_SERVER = '''"""Toy entry of the new tree: reports where it runs."""
import json
import os
import sys

import ouroboros  # the package-init hooks run here, before any other body module
from ouroboros import mod_a, mod_b

try:
    import toydep
    TOYDEP = toydep.VERSION
except ImportError:
    TOYDEP = None
BREAKS = False
if BREAKS and TOYDEP:
    raise SystemExit("the new packages break this tree")

if __name__ == "__main__":
    report = {"a": mod_a.GEN, "b": mod_b.GEN, "argv": sys.argv[1:], "prefix": sys.prefix, "toydep": TOYDEP}
    with open(os.environ["TOY_REPORT"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\\n")
'''


@pytest.fixture
def scene(tmp_path, monkeypatch, offline_pip):  # noqa: F811
    scene = Scene(tmp_path, monkeypatch)
    scene.fixed = update_prepare.private_environment_root()
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("private", scene.fixed))
    return scene


def _entry(scene, *argv, cwd=None, python=BARE):
    return subprocess.run([str(python), *argv], cwd=str(cwd or scene.serving), capture_output=True, text=True,
                          timeout=300, env={**os.environ, "TOY_REPORT": str(scene.report)})


def _same(path, other) -> bool:
    return os.path.realpath(str(path)) == os.path.realpath(str(other))


def _lock_change(scene, *, breaks=False):
    scene.official({"requirements-runtime.lock": "toydep==2.0\n", "ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n",
                    "server.py": REPORTING_SERVER.replace("BREAKS = False", f"BREAKS = {breaks}")})
    scene.local_work(commit=False)
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    generation = row["environment"]["generation"]
    assert generation["kind"] == "private" and not scene.fixed.exists()  # preparation alone selects nothing
    handoff = scene.activate(row)
    environment = handoff["environment"]
    assert environment["identity"]["current"] is None and environment["python"] == sys.executable
    assert handoff["probe_command"][0] == str(scene.fixed / "bin" / "python")
    return row, pathlib.Path(generation["user_base"])


def test_a_private_generation_owns_every_pin_and_sees_the_interpreters_packages_underneath(tmp_path, monkeypatch,
                                                                                          offline_pip):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "state").mkdir(parents=True)
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "userbase"))
    monkeypatch.setenv("PYTHONNOUSERSITE", "")  # the interpreter's user site, as a bare one imports it
    user_site = pathlib.Path(_run([BARE, "-c", "import site; print(site.getusersitepackages())"]))
    user_site.mkdir(parents=True)
    # Beside the lock, outside it: a local function's package and the lock's own pin, installed underneath.
    (user_site / "localextra.py").write_text("WHERE = 'underneath'\n")
    zipfile.ZipFile(offline_pip / "toydep-2.0-py3-none-any.whl").extractall(user_site)
    fixed = update_prepare.private_environment_root()
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("private", fixed))
    seen = []
    from supervisor import update_checks

    tracked = update_checks.run_tracked

    def run_tracked(cmd, cwd, timeout, env=None):  # what pip saw of the interpreter's packages while it decided
        seen.append((cmd[1:3], (pathlib.Path(cmd[0]).parent.parent / "pyvenv.cfg").read_text()
                     if cmd[1:3] == ["-m", "pip"] else ""))
        return tracked(cmd, cwd, timeout, env=env)

    monkeypatch.setattr(update_checks, "run_tracked", run_tracked)
    row, commit = _candidate(tmp_path)

    generation = update_prepare.prepare_generation_environment(row, commit)

    root = pathlib.Path(generation["user_base"])
    assert generation["kind"] == "private" and root == fixed.with_name("python-private.update-cand1")
    assert not fixed.exists() and _same(root / "bin" / "python", BARE)  # a venv of exactly that interpreter
    assert [step for step, _cfg in seen] == [["-m", "venv"], ["-m", "pip"]]
    assert "include-system-site-packages = false" in seen[1][1]  # hidden while pip decided what is installed
    assert "include-system-site-packages = true" in (root / "pyvenv.cfg").read_text()
    found = json.loads(_run([root / "bin" / "python", "-c", "import json, localextra, sys, toydep; "
                             "print(json.dumps([toydep.__file__, toydep.VERSION, localextra.WHERE, sys.prefix]))"]))
    assert pathlib.Path(found[0]).resolve().is_relative_to(root.resolve())  # its own pin, not the one underneath
    assert found[1:3] == ["2.0", "underneath"] and _same(found[3], root)
    marker = json.loads((root / update_prepare.PRIVATE_MARKER).read_text())
    assert marker["repo_dir"] == os.path.realpath(row["repo_dir"]) and marker["candidate_id"] == "cand1"


def test_the_first_private_environment_runs_every_entry_of_its_checkout_after_its_handoff_too(scene):
    purelib = _run([BARE, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"])
    before = sorted(os.listdir(purelib))
    _row, generation = _lock_change(scene)

    first = _entry(scene, "server.py")  # the owner's unchanged command on the bare interpreter

    assert first.returncode == 0, first.stderr
    handoff = scene.handoff()
    assert (handoff["phase"], handoff["environment"]["state"]) == ("switched", "selected") and not generation.exists()
    # The hand-over named the private interpreter: one server entry, recorded by the process that serves.
    assert [entry["kind"] for entry in handoff["entries"]] == ["server"]
    report = scene.reports()[-1]
    assert (report["a"], report["toydep"]) == ("GEN_CAND", "2.0") and _same(report["prefix"], scene.fixed)
    assert "pip" in _run([scene.fixed / "bin" / "pip", "--version"])  # its scripts name where it now is

    body_adoption._archive(scene.data, handoff, "adopted")  # what a confirmed adoption does: the pointer goes
    assert not body_adoption.helper_dir(scene.data).exists() and not body_adoption.read(scene.data)
    again = _entry(scene, "server.py")

    assert again.returncode == 0, again.stderr
    report = scene.reports()[-1]
    assert (report["toydep"], _same(report["prefix"], scene.fixed)) == ("2.0", True)
    shown = _entry(scene, "-c", "import ouroboros, sys, toydep; print(sys.prefix); print(toydep.VERSION)")
    assert shown.returncode == 0, shown.stderr
    assert _same(shown.stdout.split()[0], scene.fixed) and shown.stdout.split()[1] == "2.0"
    # Another checkout on this data root, and the owner's own venv of this interpreter, run as they are.
    other = make_serving(scene.tmp / "other")
    elsewhere = _entry(scene, "-c", "import ouroboros, sys; print(sys.prefix)", cwd=other)
    assert elsewhere.returncode == 0 and not _same(elsewhere.stdout.strip(), scene.fixed), elsewhere.stderr
    _run([BARE, "-m", "venv", "--without-pip", scene.tmp / "own-venv"])
    own = _entry(scene, "-c", "import ouroboros, sys; print(sys.prefix)", python=scene.tmp / "own-venv" / "bin" / "python")
    assert own.returncode == 0 and _same(own.stdout.strip(), scene.tmp / "own-venv"), own.stderr
    # Nothing was installed into the interpreter's own packages.
    assert sorted(os.listdir(purelib)) == before
    assert _run([BARE, "-I", "-c", "import importlib.util; print(importlib.util.find_spec('toydep'))"]) == "None"


def test_a_return_of_the_first_private_environment_leaves_the_interpreters_own_packages_serving(scene):
    row, generation = _lock_change(scene, breaks=True)

    first = _entry(scene, "server.py")

    assert first.returncode == 0, first.stderr
    handoff = scene.handoff()
    assert (handoff["return"]["outcome"], handoff["return"]["trigger"]) == ("returned", "import_probe_failed")
    assert (handoff["phase"], handoff["environment"]["state"], handoff["entries"]) == ("returned", "restored", [])
    assert not scene.fixed.exists() and (generation / update_prepare.PRIVATE_MARKER).is_file()  # parked, not selected
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["old_commit"]
    assert [report["a"] for report in scene.reports()] == ["GEN_OLD"]
    plain = _entry(scene, "-c", "import ouroboros, sys; print(sys.prefix)")
    assert plain.returncode == 0 and not _same(plain.stdout.strip(), scene.fixed), plain.stderr


def _selection(helper, adoption, current, generation):
    """The environment part of a handoff exactly as ``authorize_update`` records it for a private generation."""
    return {"id": adoption, "phase": "switching", "environment": {
        "state": "prepared", "kind": "private", "current": str(current), "generation": str(generation),
        "previous": f"{current}.previous-{adoption}", "python": str(current / "bin" / "python"),
        "identity": {"current": body_switch._path_identity(current), "generation": body_switch._path_identity(generation)}}}


def test_a_serving_private_environment_is_cloned_for_the_next_lock_change_and_switched_back(tmp_path, monkeypatch,
                                                                                            offline_pip):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "state").mkdir(parents=True)
    fixed = update_prepare.private_environment_root()
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("private", fixed))
    helper = tmp_path / "helper"
    helper.mkdir()
    row, commit = _candidate(tmp_path)
    first = pathlib.Path(update_prepare.prepare_generation_environment(row, commit)["user_base"])
    selected = _selection(helper, "adoption1", fixed, first)
    body_switch._select_environment(str(helper), selected)  # the first one: nothing was at the fixed path
    assert selected["environment"]["state"] == "selected" and not first.exists()
    (tmp_path / "ws" / "requirements-runtime.lock").write_text("toydep==1.0\n")
    git(tmp_path / "ws", "commit", "-qam", "an older pin")
    row = {**row, "candidate_id": "cand2", "layers": {"new_files": git(tmp_path / "ws", "rev-parse", "HEAD^{tree}")}}
    with monkeypatch.context() as serving:
        serving.setattr(sys, "prefix", str(fixed))  # this process stands in for the interpreter it serves on

        second = update_prepare.prepare_generation_environment(row, git(tmp_path / "ws", "rev-parse", "HEAD"))

    clone = pathlib.Path(second["user_base"])
    assert [_run([root / "bin" / "python", "-c", "import toydep; print(toydep.VERSION)"], {"PYTHONNOUSERSITE": "1"})
            for root in (fixed, clone)] == ["2.0", "1.0"]  # the serving one untouched
    assert json.loads((clone / update_prepare.PRIVATE_MARKER).read_text())["candidate_id"] == "cand2"
    handoff = _selection(helper, "adoption2", fixed, clone)

    body_switch._select_environment(str(helper), handoff)

    assert handoff["environment"]["state"] == "selected" and not clone.exists()
    assert _run([fixed / "bin" / "toytool"]).split(" ", 1) == ["1.0", str(fixed)]  # its script names where it runs
    body_switch._restore_environment(str(helper), handoff)
    assert handoff["environment"]["state"] == "restored" and clone.is_dir()
    assert _run([fixed / "bin" / "toytool"]).split(" ", 1) == ["2.0", str(fixed)]


def test_an_interpreter_that_cannot_make_a_venv_stops_typed_and_changes_nothing(tmp_path, monkeypatch, offline_pip):  # noqa: F811
    """The refusal path, with the interpreter's failure simulated: a Python without venv/ensurepip (Debian's
    split, a stripped image) exits nonzero at ``-m venv``; nothing falls back to its own packages."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "state").mkdir(parents=True)
    fixed = update_prepare.private_environment_root()
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("private", fixed))
    stripped = tmp_path / "python-without-ensurepip"
    stripped.write_text("#!/bin/sh\necho 'The virtual environment was not created successfully because ensurepip is "
                        "not available.' >&2\nexit 1\n")
    stripped.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(stripped))
    row, commit = _candidate(tmp_path)

    with pytest.raises(update_prepare.PrepareRefused) as refused:
        update_prepare.prepare_generation_environment(row, commit)

    assert refused.value.code == "UPDATE_ENVIRONMENT_UNAVAILABLE" and "ensurepip is not available" in refused.value.text
    assert sorted(os.listdir(tmp_path / "data" / "state")) == []  # no environment, no stray generation


def test_an_interpreter_without_ensurepip_gets_a_venv_with_its_own_pip_from_its_own_pip(tmp_path, monkeypatch,
                                                                                      offline_pip):  # noqa: F811
    """Debian and Ubuntu without ``python3-venv``: ``-m venv`` stops at ensurepip. Only that one call is stood in (a
    stand-in refuses it and runs this interpreter for everything else); the rest is real: the venv made alone, pip
    installed into it by this interpreter's own pip (offline, from a pip wheel), the venv's own pip then running with
    the interpreter's packages hidden and installing the whole lock. Nothing is written to the interpreter."""
    import ensurepip
    import shutil

    bundled = sorted(pathlib.Path(ensurepip.__file__).parent.glob("_bundled/pip-*.whl"))
    if not bundled:
        pytest.skip("this interpreter ships no pip wheel to install offline")
    shutil.copy(bundled[-1], offline_pip)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "state").mkdir(parents=True)
    fixed = update_prepare.private_environment_root()
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("private", fixed))
    stand_in = tmp_path / "debian" / "python3"
    stand_in.parent.mkdir()
    stand_in.write_text('#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "venv" ] && [ "$3" != "--without-pip" ]; then\n'
                        "  echo 'The virtual environment was not created successfully because ensurepip is not "
                        "available.' >&2\n  exit 1\nfi\n"
                        f'exec "{os.path.realpath(BARE)}" "$@"\n')
    stand_in.chmod(0o755)
    purelib = pathlib.Path(_run([BARE, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"]))
    before = sorted(os.listdir(purelib))
    monkeypatch.setattr(sys, "executable", str(stand_in))
    row, commit = _candidate(tmp_path)

    generation = update_prepare.prepare_generation_environment(row, commit)

    root = pathlib.Path(generation["user_base"])
    assert generation["kind"] == "private" and generation["pip"] == "interpreter_pip" and not fixed.exists()
    assert _same(root / "bin" / "python", BARE)  # still a venv of exactly that interpreter
    hidden = {"PYTHONNOUSERSITE": "1"}
    found = json.loads(_run([root / "bin" / "python", "-c", "import json, pip, toydep; print(json.dumps("
                             "[pip.__file__, toydep.__file__, toydep.VERSION]))"], hidden))
    assert all(pathlib.Path(path).resolve().is_relative_to(root.resolve()) for path in found[:2]) and found[2] == "2.0"
    assert "include-system-site-packages = true" in (root / "pyvenv.cfg").read_text()  # its packages underneath
    assert sorted(os.listdir(purelib)) == before  # nothing installed into the interpreter's own packages


def test_an_interpreter_without_venv_or_pip_gets_a_venv_layout_and_the_runtime_locks_pip_wheel(tmp_path):
    """Builds that ship neither ``venv`` nor pip. Only the refusals are stood in (``-m venv`` both ways, and the
    interpreter's own pip, which such a build lacks); the rest is real: the files ``venv --without-pip`` writes,
    pip importing itself from its wheel into the venv as ``ensurepip`` runs its bundled one (this interpreter's
    bundled wheel stands in for the fetched ``PIP_WHEEL``, so nothing is downloaded here), and the venv's own pip
    with the interpreter's packages hidden. Nothing is written to the interpreter."""
    import ensurepip
    import shutil

    from ouroboros import runtime_venv

    bundled = sorted(pathlib.Path(ensurepip.__file__).parent.glob("_bundled/pip-*.whl"))
    if not bundled:
        pytest.skip("this interpreter ships no pip wheel to install offline")
    refused = []

    def run(cmd):
        if cmd[1:3] == ["-m", "venv"] or cmd[1:4] == ["-m", "pip", "install"]:
            refused.append(cmd[2])
            return {"returncode": 1, "stdout": "", "stderr": f"No module named {cmd[2]}"}
        done = subprocess.run([str(part) for part in cmd], capture_output=True, text=True, timeout=300,
                              env={**os.environ, "PYTHONNOUSERSITE": "1"})
        return {"returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr}

    def fetch(directory):
        directory.mkdir(parents=True, exist_ok=True)
        return pathlib.Path(shutil.copy(bundled[-1], directory))

    purelib = pathlib.Path(_run([BARE, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"]))
    before = sorted(os.listdir(purelib))
    root = tmp_path / "python-private.gen"

    assert runtime_venv.create_with_pip(root, os.path.realpath(BARE), run, fetch) == "layout+pip_wheel"

    assert refused == ["venv", "venv", "pip"] and not (root / ".pip-wheel").exists()
    assert _same(root / "bin" / "python", BARE) and (root / "bin" / "python3").is_symlink()
    found = json.loads(_run([root / "bin" / "python", "-c", "import json, pip, sys; print(json.dumps([pip.__file__, "
                             "sys.prefix, sys.base_prefix]))"], {"PYTHONNOUSERSITE": "1"}))
    assert pathlib.Path(found[0]).resolve().is_relative_to(root.resolve()) and _same(found[1], root)
    assert not _same(found[2], root) and "include-system-site-packages = false" in (root / "pyvenv.cfg").read_text()
    assert sorted(os.listdir(purelib)) == before  # nothing installed into the interpreter's own packages


def test_the_pip_fetched_for_an_interpreter_without_one_is_the_runtime_locks_pip():
    from ouroboros import runtime_venv

    pinned = next(line.split("==", 1)[1].strip() for line in (REPO / "requirements-runtime.lock").read_text(
        encoding="utf-8").splitlines() if line.startswith("pip=="))
    assert runtime_venv.PIP_WHEEL["url"].endswith(f"/pip-{pinned}-py3-none-any.whl")
    assert runtime_venv.PIP_WHEEL["url"].startswith("https://files.pythonhosted.org/")


@pytest.mark.parametrize("roots", ["data", "app", "home"])
def test_the_package_init_hook_restates_where_the_server_keeps_the_private_environment(tmp_path, monkeypatch, roots):
    """The stdlib hook runs before any import, so it restates ``config.resolve_data_dir`` and the two names. Every
    child of this suite gets a scratch data root, so the hook's own expression is evaluated here, each way a root
    is given (an entry through that root is executed above)."""
    import ast

    hook = next(node for node in ast.parse((REPO / "ouroboros" / "__init__.py").read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == "_private_environment_hook")
    data = next(node.value for node in ast.walk(hook) if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "data")
    names = {node.value for node in ast.walk(hook) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert {update_prepare.PRIVATE_NAME, update_prepare.PRIVATE_MARKER} <= names
    for key in ("OUROBOROS_DATA_DIR", "OUROBOROS_APP_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    if roots == "data":
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data-root"))
    elif roots == "app":
        monkeypatch.setenv("OUROBOROS_APP_ROOT", str(tmp_path / "app-root"))

    restated = eval(compile(ast.Expression(data), "ouroboros/__init__.py", "eval"), {"os": os})  # noqa: S307

    assert pathlib.Path(restated) == config.resolve_data_dir()  # what the server's DATA_DIR is at its import
