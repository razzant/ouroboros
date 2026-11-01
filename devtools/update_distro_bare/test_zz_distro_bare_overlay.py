"""The real distro conditions the private-environment tests stand in for, executed on that interpreter.

``tests/test_update_private_environment.py`` proves the owners on a python-build-standalone interpreter, where a shell
stand-in plays Debian's missing ``python3-venv``. This file runs in a disposable copy of the subject (``cell.sh`` copies
it into ``tests/`` so the subject's conftest applies) on a real ``/usr/bin/python3`` from apt. Nothing in the product is
replaced: no ``sys.executable`` stand-in, no refusing script, ``serving_environment`` is not patched, and the subject's
test boundary (``OUROBOROS_PYTEST_ACTIVE``) stays in every child. ``Q_EXPECTED_PIP`` is the path this cell predicts.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import site
import subprocess
import sys
import sysconfig

import pytest

from ouroboros import body_adoption, body_switch
from supervisor import update_checks, update_prepare
from tests.body_candidate_support import REPO, git
from tests.test_update_activation import Scene
from tests.test_update_environment import _candidate, _run, offline_pip  # noqa: F401 -- the offline wheels fixture
from tests.test_update_private_environment import BARE, _same, _selection

pytestmark = pytest.mark.serial

DRIVER_SITE = os.environ.get("Q_DRIVER_SITE", "")
EVIDENCE = pathlib.Path(os.environ.get("Q_EVIDENCE", "."))
EXPECTED_PIP = os.environ.get("Q_EXPECTED_PIP", "")
STEPS = {  # create_with_pip's real order on each of its paths, then _install_lock
    "ensurepip": ["venv", "lock"],
    "interpreter_pip": ["venv", "venv_without_pip", "own_pip_installing_pip", "pip_version", "lock"],
    "pip_wheel": ["venv", "venv_without_pip", "own_pip_installing_pip", "wheel_installing_pip", "pip_version", "lock"],
    "layout+interpreter_pip": ["venv", "venv_without_pip", "its_version", "own_pip_installing_pip", "pip_version",
                               "lock"],
    "layout+pip_wheel": ["venv", "venv_without_pip", "its_version", "own_pip_installing_pip", "wheel_installing_pip",
                         "pip_version", "lock"],
}
BARE_STEPS = {"venv", "venv_without_pip", "its_version"}  # run by the interpreter itself; every other by the venv's
# Lock pip's own module, run by the interpreter under test from the driver's directory: a child of a real marker read.
PIP_FROM_DRIVER = ("import runpy, sys; sys.path[:0] = [sys.argv.pop(1)]; sys.argv[0] = 'pip'; "
                   "runpy.run_module('pip', run_name='__main__', alter_sys=True)")
# Appended to the Update's own import probe: where the process that completed it really ran.
WHERE = "; import json, sys, starlette; print('RAN=' + json.dumps([sys.prefix, sys.executable, starlette.__file__]))"


def _global_packages() -> dict:
    """Names in every directory this interpreter would import installed packages from (census.py keeps the bytes)."""
    folders = {*site.getsitepackages(), sysconfig.get_path("purelib"), sysconfig.get_path("platlib"),
               "/usr/lib/python3/dist-packages"}
    return {folder: sorted(os.listdir(folder)) if os.path.isdir(folder) else None for folder in sorted(folders)}


def _step(cmd) -> str:
    """Which step of the owners ``cmd`` is, from its exact arguments; an unknown one names itself and fails the list."""
    args = [str(part) for part in cmd[1:]]
    if args[:3] == ["-m", "venv", "--without-pip"]:
        return "venv_without_pip"
    if args[:2] == ["-m", "venv"]:
        return "venv"
    if args[:5] == ["-m", "pip", "install", "-q", "--ignore-installed"]:
        return "own_pip_installing_pip"
    if args[:3] == ["-m", "pip", "--version"]:
        return "pip_version"
    if args[:4] == ["-m", "pip", "install", "-q"] and args[4:5] == ["-r"]:
        return "lock"
    if args[:1] == ["-c"] and "runpy" in args[1]:
        return "wheel_installing_pip"
    if args[:1] == ["-c"] and "sys.version_info" in args[1]:
        return "its_version"
    return " ".join(args[:5])


def _who(cmd, generation) -> str:
    """Which interpreter ``cmd`` names, from its raw ``argv[0]`` unresolved: a venv's ``bin/python`` links to the bare
    interpreter, so its real path would say nothing."""
    first = os.path.abspath(str(cmd[0]))
    if first.startswith(os.path.abspath(str(generation)) + os.sep):
        return "generation"
    return "bare" if first == os.path.abspath(sys.executable) else f"other:{first}"


def _data_root(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "state").mkdir(parents=True)
    return tmp_path / "data"


def _bare_entry(checkout, data, *argv, timeout=300) -> dict:
    """``BARE argv`` from ``checkout``, as the owner starts it: this test chooses no interpreter; the checkout's
    package-init hook does. The repository is named as the entry names it itself (``launch_roots.enter_direct``)."""
    done = subprocess.run([BARE, *argv], cwd=str(checkout), capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_REPO_DIR": str(checkout)})
    return {"argv": [BARE, *(part[:120] for part in argv)], "rc": done.returncode, "stdout": done.stdout[-1500:],
            "stderr": done.stderr[-2500:]}


def test_this_process_is_the_bare_distro_interpreter_under_its_real_policy(tmp_path, offline_pip):  # noqa: F811
    assert sys.prefix == sys.base_prefix and _same(sys.executable, BARE) and os.path.realpath(BARE).startswith("/usr/")
    assert "PIP_BREAK_SYSTEM_PACKAGES" not in os.environ and "PYTHONPATH" not in os.environ
    marker = pathlib.Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED"
    owner = subprocess.run(["dpkg", "-S", str(marker)], capture_output=True, text=True, timeout=60)
    assert marker.is_file() and owner.returncode == 0, owner.stderr  # the distro's file, none this run wrote
    # The serving seam as the product decides it: nothing patched, this interpreter's generations are private ones.
    assert update_prepare.serving_environment() == ("private", update_prepare.private_environment_root())
    before = _global_packages()
    tried = [[BARE, "-c", PIP_FROM_DRIVER, DRIVER_SITE]] + ([[BARE, "-m", "pip"]] if EXPECTED_PIP != "pip_wheel" else [])
    for pip in tried:  # --dry-run without --report is still checked (pip install.py) and cannot write on a broken policy
        refused = subprocess.run([*pip, "install", "--dry-run", "--no-index", "--find-links", str(offline_pip),
                                  "toydep==2.0"], capture_output=True, text=True, timeout=120)
        assert refused.returncode != 0 and "externally-managed-environment" in refused.stdout + refused.stderr, (
            pip, refused.stdout[-800:], refused.stderr[-800:])
    assert _global_packages() == before


def test_preparation_takes_the_path_this_interpreter_really_has(tmp_path, monkeypatch, offline_pip):  # noqa: F811
    """The union of what tests 117, 279 and 321 assert for one path, with the stand-ins gone. Every check runs against
    the path preparation really took; only the last one compares it with this cell's prediction, so another legitimate
    path reads as PATH_NOT_COVERED (``verdict.py``): neither a product failure nor a pass of the predicted branch."""
    _data_root(tmp_path, monkeypatch)
    fixed = update_prepare.private_environment_root()
    expected_root = fixed.with_name(f"{fixed.name}.update-cand1")
    seen, tracked = [], update_checks.run_tracked

    def run_tracked(cmd, cwd, timeout, env=None):
        step = _step(cmd)
        config = pathlib.Path(cmd[0]).parent.parent / "pyvenv.cfg"
        seen.append({"step": step, "who": _who(cmd, expected_root), "argv": [str(part) for part in cmd],
                     "config": config.read_text() if step == "lock" else "",
                     "breaks_policy": "PIP_BREAK_SYSTEM_PACKAGES" in (env or {})})
        return tracked(cmd, cwd, timeout, env=env)

    monkeypatch.setattr(update_checks, "run_tracked", run_tracked)
    before = _global_packages()
    row, commit = _candidate(tmp_path)

    generation = update_prepare.prepare_generation_environment(row, commit)

    taken, root = generation.get("pip", ""), pathlib.Path(generation["user_base"])
    record = {"expected": EXPECTED_PIP, "taken": taken, "held_for_taken_path": False, "seen": seen, **generation}
    (EVIDENCE / "preparation.json").write_text(json.dumps(record, default=str))
    assert generation["kind"] == "private" and taken in STEPS, taken
    assert [call["step"] for call in seen] == STEPS[taken]
    assert [call["who"] for call in seen] == ["bare" if step in BARE_STEPS else "generation" for step in STEPS[taken]]
    assert not any(call["breaks_policy"] for call in seen)
    assert root == expected_root and not fixed.exists()
    assert _same(root / "bin" / "python", BARE)  # a venv of exactly this interpreter
    assert "include-system-site-packages = false" in seen[-1]["config"]  # hidden while pip decided what is installed
    assert "include-system-site-packages = true" in (root / "pyvenv.cfg").read_text()
    found = json.loads(_run([root / "bin" / "python", "-c", "import json, pip, toydep; print(json.dumps([pip.__file__, "
                             "toydep.__file__, toydep.VERSION]))"], {"PYTHONNOUSERSITE": "1"}))
    assert all(pathlib.Path(path).resolve().is_relative_to(root.resolve()) for path in found[:2]) and found[2] == "2.0"
    marker = json.loads((root / update_prepare.PRIVATE_MARKER).read_text())
    assert marker["repo_dir"] == os.path.realpath(row["repo_dir"]) and marker["candidate_id"] == "cand1"
    assert not (root / ".pip-wheel").exists() and _global_packages() == before
    assert _run([BARE, "-I", "-c", "import importlib.util; print(importlib.util.find_spec('toydep'))"]) == "None"
    (EVIDENCE / "preparation.json").write_text(json.dumps({**record, "held_for_taken_path": True}, default=str))
    assert taken == EXPECTED_PIP, f"PATH_NOT_COVERED: preparation took {taken}, this cell predicts {EXPECTED_PIP}"


def test_without_the_index_this_interpreter_stops_typed_and_changes_nothing(tmp_path, monkeypatch, offline_pip):  # noqa: F811
    """Phase N: no network, no pip and no ensurepip, all real (DEPLOYMENT: "without it the Update stops at preparation")."""
    data = _data_root(tmp_path, monkeypatch)
    before = _global_packages()
    row, commit = _candidate(tmp_path)

    with pytest.raises(update_prepare.PrepareRefused) as refused:
        update_prepare.prepare_generation_environment(row, commit)

    (EVIDENCE / "no_index_refusal.json").write_text(json.dumps({"code": refused.value.code, "text": refused.value.text}))
    assert refused.value.code == "UPDATE_ENVIRONMENT_UNAVAILABLE"
    assert "could not make a private environment" in refused.value.text
    assert sorted(os.listdir(data / "state")) == [] and _global_packages() == before


def test_the_real_lock_installs_on_this_interpreter_and_its_unchanged_entry_runs_there_once_selected(tmp_path,
                                                                                                    monkeypatch):
    """The subject's own tree as the checkout, its lock from the real index into a generation of this interpreter (every
    applicable pin there, ``pip check`` clean), selected by the product's selection owner as test 222 drives it. Then
    the owner's unchanged ``BARE server.py`` and the Update's own import probe run from that checkout on ``BARE``, and
    only the package-init hook can carry them into the generation: the interpreter's own packages have no ``starlette``,
    which the same command shows before selection. Import and argument-parser entry only, not serving readiness;
    workspace checks are out of scope."""
    data = _data_root(tmp_path, monkeypatch)
    (data / "logs").mkdir()  # as the data root of the server that ran the Update has it (body_candidate_support.isolate)
    fixed = update_prepare.private_environment_root()
    checkout = tmp_path / "checkout"
    shutil.copytree(REPO, checkout, symlinks=True, ignore=lambda folder, names: {
        name for name in names if name in (".git", "__pycache__") or (_same(folder, REPO) and name == "tests")})
    git(checkout, "init", "-q")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-qm", "the subject's tree")
    commit = git(checkout, "rev-parse", "HEAD")
    row = {"candidate_id": "cand1", "path": str(checkout), "repo_dir": str(checkout),
           "layers": {"new_files": git(checkout, "rev-parse", "HEAD^{tree}")}}
    record, before = {}, _global_packages()

    def keep(**facts):  # written at each stage, so a stop shows where it was
        record.update(facts)
        (EVIDENCE / "real_lock.json").write_text(json.dumps(record, default=str))

    own = _run([BARE, "-I", "-c", "import importlib.util; print(importlib.util.find_spec('starlette'))"])
    unselected = _bare_entry(checkout, data, "server.py", "--help")
    keep(interpreter_finds_starlette=own, unselected=unselected)
    # A control of this cell's state only (apt's python3 and nothing else), not a statement about any boot.
    assert own == "None" and unselected["rc"] != 0 and "ModuleNotFoundError" in unselected["stderr"], unselected

    generation = update_prepare.prepare_generation_environment(row, commit)

    root = pathlib.Path(generation["user_base"])
    python, hidden = root / "bin" / "python", {"PYTHONNOUSERSITE": "1"}
    pins = json.loads(_run([python, "-c", """
import importlib.metadata as md, json, os, re, sys
from pip._vendor.packaging.markers import Marker
norm = lambda name: re.sub(r"[-_.]+", "-", name).lower()
here = {norm(d.name): (d.version, str(d.locate_file(""))) for d in md.distributions(path=[p for p in sys.path if p.startswith(sys.prefix)])}
applicable, missing = 0, []
for line in open(sys.argv[1], encoding="utf-8"):
    pin, _, marker = line.partition(";")
    if not pin.strip() or pin.startswith("#") or (marker.strip() and not Marker(marker.strip()).evaluate()):
        continue
    applicable += 1
    name, _, version = pin.strip().partition("==")
    if here.get(norm(name), ("",))[0] != version:
        missing.append(pin.strip())
print(json.dumps({"applicable": applicable, "missing": missing}))
""", checkout / "requirements-runtime.lock"], hidden))
    checked = subprocess.run([str(python), "-m", "pip", "check"], capture_output=True, text=True, timeout=300,
                             env={**os.environ, **hidden})
    keep(**pins, pip=generation.get("pip"), pip_check=[checked.returncode, checked.stdout[-1500:]])
    assert generation["kind"] == "private" and not pins["missing"] and pins["applicable"] > 50, pins
    assert checked.returncode == 0, checked.stdout
    helper = tmp_path / "helper"
    helper.mkdir()
    selection = _selection(helper, "adoption1", fixed, root)

    body_switch._select_environment(str(helper), selection)

    assert selection["environment"]["state"] == "selected" and fixed.is_dir() and not root.exists()
    assert not body_adoption.read(data)  # no adoption record: the state every entry after a retired handoff meets
    probe = _bare_entry(checkout, data, "-c", body_adoption.UPDATE_PROBE_IMPORTS + WHERE)
    line = next((text for text in probe["stdout"].splitlines() if text.startswith("RAN=")), "")
    ran = json.loads(line[len("RAN="):]) if line else []
    entry = _bare_entry(checkout, data, "server.py", "--help")
    keep(selected=selection["environment"]["state"], probe=probe, ran=ran, entry=entry)
    assert probe["rc"] == 0 and ran, probe
    assert _same(ran[0], fixed) and pathlib.Path(ran[2]).resolve().is_relative_to(fixed.resolve()), ran
    assert entry["rc"] == 0 and "Run the Ouroboros web server." in entry["stdout"], entry
    assert _global_packages() == before


def test_the_real_tree_enters_under_a_live_handoff_once_its_lock_changed(tmp_path, monkeypatch):
    """Test 159's own ``Scene`` with the real tree serving in place of the toy body, and the real import probe (``Scene``
    sets ``import server`` for the toy one). The target is the same tree with one comment line added to its real lock,
    which preparation compares by bytes, so the lock installs into this interpreter's first private generation. After
    ``authorize_update`` and ``arm``, the owner's unchanged ``BARE server.py --help`` itself makes the switch, the probe
    in the helper's own environment and the hand-over (``os.execv`` of the selected interpreter), and the new process
    records its entry before it reaches the parser. Workspace checks stay stubbed as in ``Scene``; nothing serves."""
    probe_imports = body_adoption.UPDATE_PROBE_IMPORTS
    scene = Scene(tmp_path, monkeypatch)
    monkeypatch.setattr(body_adoption, "UPDATE_PROBE_IMPORTS", probe_imports)
    fixed = update_prepare.private_environment_root()
    for path in scene.serving.iterdir():  # the toy body goes; its branch, Git config and history stay
        if path.name != ".git":
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    shutil.copytree(REPO, scene.serving, symlinks=True, dirs_exist_ok=True, ignore=lambda folder, names: {
        name for name in names if name in (".git", "__pycache__") or (_same(folder, REPO) and name == "tests")})
    git(scene.serving, "add", "-A")
    git(scene.serving, "commit", "-qm", "the subject's tree serves")
    lock = (scene.serving / "requirements-runtime.lock").read_text(encoding="utf-8")
    scene.official({"requirements-runtime.lock": lock + "\n# the same pins: a change of the lock's bytes only\n"})
    record, before = {}, _global_packages()

    def keep(**facts):  # written at each stage, so a stop shows where it was
        record.update(facts)
        (EVIDENCE / "live_handoff.json").write_text(json.dumps(record, default=str))

    own = _run([BARE, "-I", "-c", "import importlib.util; print(importlib.util.find_spec('starlette'))"])
    row = scene.prepare()
    keep(interpreter_finds_starlette=own, state=row.get("state"), reason=row.get("reason"),
         generation=(row.get("environment") or {}).get("generation"))
    assert own == "None" and row["state"] == "ready", row.get("reason")
    assert row["environment"]["generation"]["kind"] == "private" and not fixed.exists()
    scene.activate(row)

    entry = _bare_entry(scene.serving, scene.data, "server.py", "--help", timeout=700)  # the probe's own bound is 600 s

    handoff = scene.handoff()
    keep(entry=entry, handoff={key: handoff.get(key) for key in ("phase", "environment", "entries", "return",
                                                                 "events")})
    assert entry["rc"] == 0 and "Run the Ouroboros web server." in entry["stdout"], entry
    assert (handoff["phase"], handoff["environment"]["state"]) == ("switched", "selected") and not handoff.get("return")
    assert [item["kind"] for item in handoff["entries"]] == ["server"] and fixed.is_dir()
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["new_commit"]
    assert _global_packages() == before
