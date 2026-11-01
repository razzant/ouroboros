"""The Update's official macOS app step over real Git and real processes; nothing reaches a network or a real app.

Releases are real tags of the miniature body whose ``launcher.py`` declares a source-entry
protocol; the installed app is a bundle under ``tmp_path`` read through ``FakeMac`` (codesign,
spctl, hdiutil and ditto answered from one signing identity), and its embedded interpreter is this
test's Python; downloads are a fake fetch, where a release without assets answers 404. Activation
is real: the captured helper and installer beside the record, the installer started detached by
the staged entry, held in process custody and named in the record, the old app's launcher and
server as real processes the stand-in ``osascript`` ends (the old app's Quit), the launcher's
single-instance lock, the import probe on the staged interpreter, the exchange through
``renamex_np`` on this volume, the stand-in ``open`` and real cold entries afterwards. Nothing
here is native qualification (plan §11): no signed app, Apple Event, LaunchServices, TCC or
WebKit profile is involved.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import plistlib
import subprocess
import sys
import textwrap
import time
import urllib.error
from types import SimpleNamespace

import pytest

from ouroboros import body_adoption, body_switch, native_package, runtime_venv
from supervisor import update_native, update_prepare
from tests.body_candidate_support import LAUNCHER_PY, REPO, git
from tests.test_native_package import IMAGE, INSTALLED, RUNTIME, TARGET, FakeMac, capsule, make_app
from tests.test_update_activation import EXITED, STAGED, STOPPED, UNSTAGED, Scene

pytestmark = [pytest.mark.serial,
              pytest.mark.skipif(sys.platform != "darwin", reason="the macOS app step: renamex_np and launchd")]
BASE = "https://github.com/razzant/ouroboros/releases/download"


def _release(scene, version, protocol):
    """A published release of the body: a tagged commit whose launcher.py declares ``protocol``."""
    git(scene.serving, "checkout", "-q", "-b", f"release-{version}")
    (scene.serving / "launcher.py").write_text(LAUNCHER_PY + f"\nMACOS_SOURCE_ENTRY_PROTOCOL = {protocol}\n")
    git(scene.serving, "commit", "-q", "-am", f"release {version}")
    git(scene.serving, "tag", f"v{version}")
    sha = git(scene.serving, "rev-parse", "HEAD")
    git(scene.serving, "checkout", "-q", "ouroboros")
    return {"tag": f"v{version}", "sha": sha, "source": "official"}


_real_prepare = update_native.prepare
_real_installed_bundle = update_native.installed_bundle


@pytest.fixture
def scene(tmp_path, monkeypatch):
    scene = Scene(tmp_path, monkeypatch)
    scene.app = make_app(tmp_path / "Applications" / "Ouroboros.app", version=INSTALLED)
    scene.fake = FakeMac()
    scene.fetched, scene.unpublished, scene.installs = [], set(), []
    scene.published = [_release(scene, TARGET, 2), _release(scene, INSTALLED, 1)]
    scene.caskrooms = [tmp_path / "Caskroom"]
    scene.userbase = tmp_path / "userbase"  # the packaged user site, the one path every launcher runs the body with
    (scene.userbase / "lib").mkdir(parents=True)
    (scene.userbase / "lib" / "serving-package").write_text("serving\n")

    def install(target, _lock, kind, python=None):  # pip into the generation only, by the interpreter given
        entry = {"target": pathlib.Path(target), "kind": kind, "python": python}
        if kind == "venv":  # a venv's pip is its own interpreter: what the clone names while it installs
            entry["link"] = _base_named(target)
            config = pathlib.Path(target) / "pyvenv.cfg"
            entry["config"] = config.read_text() if config.is_file() else None
        scene.installs.append(entry)
        (pathlib.Path(target) / "lib" / "generation-package").write_text("generation\n")
        if kind == "venv":  # pip names a console script's interpreter by the path it installed into
            script = pathlib.Path(target) / "bin" / "generation-tool"
            script.write_text(f"#!{target}/bin/python\nimport sys\n")
            script.chmod(0o755)
        return {"returncode": 0}

    def fetch(url, target, _limit):
        scene.fetched.append(url)
        version = url.rsplit("/", 2)[-2].lstrip("v")
        if version in scene.unpublished:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        payload = capsule(version).encode() if url.endswith("SHA256SUMS") else IMAGE
        pathlib.Path(target).write_bytes(payload)
        return len(payload)

    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("user_site", scene.userbase))
    monkeypatch.setattr(update_prepare, "_install_lock", install)
    monkeypatch.setattr(update_native, "installed_bundle", lambda *_args: scene.app)  # what preparation finds
    monkeypatch.setattr(update_native, "prepare", lambda row, **kw: _real_prepare(
        row, run=scene.fake, fetch=fetch, published=scene.published,
        caskrooms=scene.caskrooms, appdirs=[scene.app.parent], **kw))
    monkeypatch.setattr("ouroboros.config.PID_FILE", tmp_path / "ouroboros.pid")  # the launcher's single-instance lock
    return scene


def _target_speaks(scene, protocol):
    """The official target: this body with a launcher declaring ``protocol`` (the serving one declares none)."""
    return scene.official({"VERSION": f"{TARGET}\n", "ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n",
                           "launcher.py": LAUNCHER_PY + f"\nMACOS_SOURCE_ENTRY_PROTOCOL = {protocol}\n"})


def _version_at(app):
    with open(pathlib.Path(app) / "Contents" / "Info.plist", "rb") as stream:
        return plistlib.load(stream)["CFBundleShortVersionString"]


def _runtime_venv(scene, *, marked=True, config=False):
    """The launcher's ``--runtime-venv`` as ``runtime_venv.prepare_runtime_venv`` leaves it under a macOS app: its
    marker, and the layout CPython's ``venv`` gives a venv made by the app's ``python3`` (the launcher's
    ``base_python``): ``bin/python3`` names that Python by its installed path, ``python`` and ``python3.X`` are
    relative aliases of it (``tests/test_runtime_venv.py`` checks that shape on a real venv of the app's own Python).
    ``config`` adds the ``pyvenv.cfg`` naming that Python's folder (left out where the venv really runs: the stand-in
    app has no standard library there)."""
    venv = scene.tmp / "owner-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib").mkdir()
    (venv / "bin" / "python3").symlink_to(_app_python(scene.app))
    for alias in ("python", f"python3.{sys.version_info.minor}"):
        (venv / "bin" / alias).symlink_to("python3")
    (venv / "lib" / "serving-package").write_text("serving\n")
    if config:
        (venv / "pyvenv.cfg").write_text(f"home = {_app_python(scene.app).parent}\ninclude-system-site-packages = "
                                         f"false\nversion = {platform.python_version()}\n")
    if marked:
        (venv / runtime_venv.MARKER_NAME).write_text(json.dumps({"schema_version": 1, "python": RUNTIME,
                                                                 "state": "ready"}))
    scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))
    return venv


def _base_named(venv):
    """The absolute path a venv's ``bin/python`` leads to through its own relative aliases, as Python follows them."""
    path = pathlib.Path(venv) / "bin" / "python"
    while path.is_symlink():
        text = os.readlink(path)
        if os.path.isabs(text):
            return text
        path = path.parent / text
    return ""


def _venv_state(venv):
    """Every link of a venv's ``bin/`` with its exact text, and its ``pyvenv.cfg`` bytes (``None`` without one)."""
    config = pathlib.Path(venv) / "pyvenv.cfg"
    return ({link.name: os.readlink(link) for link in (pathlib.Path(venv) / "bin").iterdir() if link.is_symlink()},
            config.read_bytes() if config.is_file() else None)


def _cask_receipt(scene, version=INSTALLED, form="json"):
    """Homebrew's record of the cask it installed this app from, as ``brew install --cask`` leaves it."""
    receipt = scene.caskrooms[0] / "ouroboros" / ".metadata" / version / "20261027120000.000" / "Casks"
    receipt.mkdir(parents=True)
    if form == "json":
        receipt = receipt / "ouroboros.json"
        receipt.write_text(json.dumps({"token": "ouroboros", "version": version,
                                       "artifacts": [{"app": ["Ouroboros.app"]}, {"zap": []}]}))
    else:
        receipt = receipt / "ouroboros.rb"
        receipt.write_text(f'cask "ouroboros" do\n  version "{version}"\n  app "Ouroboros.app"\nend\n')
    return receipt


# --------------------------------------------------------------------------- #
# Preparation: need, package, origin, staging
# --------------------------------------------------------------------------- #
def test_a_target_that_keeps_the_serving_protocol_needs_no_app(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    row = scene.prepare()
    assert row["state"] == "ready" and row["native"] == {"status": "none", "reason": "nothing_declared"}
    assert scene.fetched == [] and scene.fake.calls == [] and scene.installs == []


def test_a_needed_official_app_is_staged_with_its_own_interpreter_and_private_packages(scene):
    _target_speaks(scene, 2)
    scene.local_work(commit=False)
    before = scene.composition()
    row = scene.prepare()
    staged = row["native"]
    assert row["state"] == "ready", (row.get("reason"), staged)
    assert (staged["status"], staged["origin"], staged["team"], staged["version"]) == ("staged", "official",
                                                                                         "OFFICIAL01", TARGET)
    assert scene.fetched == [f"{BASE}/v{TARGET}/SHA256SUMS", f"{BASE}/v{INSTALLED}/SHA256SUMS",
                             f"{BASE}/v{INSTALLED}/Ouroboros-{INSTALLED}.dmg", f"{BASE}/v{TARGET}/Ouroboros-{TARGET}.dmg"]
    # The staged app's interpreter really ran; it installed the private packages, even with the lock unchanged.
    assert staged["python"].startswith(staged["staged_app"] + "/") and staged["runtime"]["version"] == platform.python_version()
    [install] = scene.installs
    generation = row["environment"]["generation"]
    assert (install["python"], install["kind"], str(install["target"])) == (staged["python"], "user_site",
                                                                            generation["user_base"])
    assert generation["python"] == staged["python"]  # the checks run W' on it (update_checks._environments)
    assert staged["previous_identity"] == body_switch.bundle_identity(str(scene.app))
    assert staged["new_identity"] == body_switch.bundle_identity(staged["staged_app"])
    assert scene.composition() == before and not (scene.userbase / "lib" / "generation-package").exists()
    assert scene.app.is_dir() and not pathlib.Path(staged["backup_app"]).exists()  # nothing replaced yet
    update_prepare.discard(update_prepare.find(row["candidate_id"]))  # a retired candidate's staging goes too
    from ouroboros import subagent_worktrees

    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task_id: False)
    assert not pathlib.Path(staged["staging_dir"]).exists() and scene.app.is_dir()


@pytest.mark.parametrize("app_version, entry, outcome", [
    ("7.8.0", True, ("ready", "none", "protocol_unchanged")),
    ("7.7.0", False, ("ready", "staged", "signed_entry_missing")),
])
def test_an_app_without_the_signed_entry_gets_the_release_that_has_it(scene, app_version, entry, outcome):
    """7.7.0 → current source, offline: the serving and target sources both declare protocol 1; an app
    built without ``ouroboros-shell`` cannot host them as the current launcher requires, whatever the
    protocols say, so the step is the newest published app that has it (7.8.0 here)."""
    scene.published.append(_release(scene, "7.7.0", None))  # its launcher declares no protocol
    scene.app = make_app(scene.tmp / "Installed" / "Ouroboros.app", version=app_version, entry=entry)
    (scene.serving / "launcher.py").write_text(LAUNCHER_PY + "\nMACOS_SOURCE_ENTRY_PROTOCOL = 1\n")
    git(scene.serving, "commit", "-qam", "serving source: protocol 1, run with plain Python by 7.7.0")
    _target_speaks(scene, 1)

    row = scene.prepare()

    native_fact = row["native"]
    assert (row["state"], native_fact["status"], (native_fact.get("need") or native_fact)["reason"]) == outcome
    if not entry:
        assert native_fact["version"] == INSTALLED == "7.8.0" and native_fact["previous_version"] == "7.7.0"


def test_an_unpublished_protocol_has_no_package_to_stage(scene):
    _target_speaks(scene, 3)
    row = scene.prepare()
    assert row["native"]["reason"] == "no_published_package" and "protocol 3" in row["question"]
    assert scene.fetched == []


def test_a_newer_tag_without_published_assets_never_shadows_the_published_app(scene):
    scene.published.insert(0, _release(scene, "7.9.5", 2))  # tagged and pushed, its release not published yet
    scene.unpublished.add("7.9.5")
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert (row["state"], row["native"]["status"], row["native"]["version"]) == ("ready", "staged", TARGET)
    assert f"{BASE}/v7.9.5/SHA256SUMS" in scene.fetched and not any("7.9.5/Ouroboros" in url for url in scene.fetched)


def test_an_official_app_homebrew_installed_takes_the_same_update_and_its_receipt_stays_as_written(scene, monkeypatch):
    """Homebrew without the owner: the cask installs the official image, so the app proves official like any other
    and is staged the same way. The receipt is read, never written: Homebrew keeps listing the version it installed,
    recorded apart from the app's own, into the activation's record too."""
    receipt = _cask_receipt(scene, form="rb")
    written = {path: path.read_bytes() for path in scene.caskrooms[0].rglob("*") if path.is_file()}
    _target_speaks(scene, 2)

    row = scene.prepare()

    staged = row["native"]
    assert (row["state"], staged["status"], staged["origin"]) == ("ready", "staged", "official"), (row.get("reason"),
                                                                                                 staged)
    assert (staged["manager"], staged["manager_version"], staged["previous_version"], staged["version"]) == (
        "Homebrew Cask", INSTALLED, INSTALLED, TARGET)
    # The receipt bypasses nothing: the app's own official release image is fetched and its payload's code-directory
    # hash compared with this app's, exactly as for an app no package manager installed, before the package is.
    assert scene.fetched == [f"{BASE}/v{TARGET}/SHA256SUMS", f"{BASE}/v{INSTALLED}/SHA256SUMS",
                             f"{BASE}/v{INSTALLED}/Ouroboros-{INSTALLED}.dmg",
                             f"{BASE}/v{TARGET}/Ouroboros-{TARGET}.dmg"]
    attached = [call[-1] for call in scene.fake.calls if call[:2] == [native_package.HDIUTIL, "attach"]]
    assert [pathlib.Path(image).name for image in attached] == [f"Ouroboros-{INSTALLED}.dmg", f"Ouroboros-{TARGET}.dmg"]
    assert any(call[:2] == [native_package.CODESIGN, "-dv"] and call[-1] == str(scene.app) for call in scene.fake.calls)
    projected = update_prepare.projection()["native"]
    assert (projected["origin"], projected["manager"], projected["manager_version"]) == (
        "official", "Homebrew Cask", INSTALLED)
    monkeypatch.setattr(update_native, "launcher_pid", lambda: os.getppid())
    handoff = body_adoption.authorize_update(scene.data, row)
    assert (handoff["native"]["manager"], handoff["native"]["manager_version"]) == ("Homebrew Cask", INSTALLED)
    assert {path: path.read_bytes() for path in scene.caskrooms[0].rglob("*") if path.is_file()} == written
    assert receipt.is_file()


def test_a_homebrew_installed_app_that_is_not_the_official_image_is_left_to_homebrew(scene):
    _cask_receipt(scene)
    scene.fake.cdhashes[str(scene.app)] = "0" * 40  # another cask's build: the right signer and tag, other code
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert (row["state"], row["native"]["reason"], row["native"]["manager"]) == (
        "needs_decision", "origin_custom", "Homebrew Cask")
    assert "brew upgrade --cask ouroboros" in row["question"] and scene.installs == []
    assert f"{BASE}/v{TARGET}/Ouroboros-{TARGET}.dmg" not in scene.fetched  # its own release was read, nothing more


def test_the_installed_app_is_official_only_when_its_own_release_payload_has_its_code_directory(scene):
    scene.fake.cdhashes[str(scene.app)] = "0" * 40  # a build with the right signer and tag, other code
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert row["native"]["reason"] == "origin_custom" and row["state"] == "needs_decision"
    assert scene.fetched == [f"{BASE}/v{TARGET}/SHA256SUMS", f"{BASE}/v{INSTALLED}/SHA256SUMS",
                             f"{BASE}/v{INSTALLED}/Ouroboros-{INSTALLED}.dmg"]
    staging = pathlib.Path(row["native"]["staging_dir"])
    assert staging.name == f".Ouroboros-update-{TARGET}-{row['candidate_id']}"  # this candidate's own folder
    assert not (staging / "installed-release").exists()


def test_an_official_legacy_app_gatekeeper_calls_unnotarized_is_replaced_by_a_notarized_one(scene):
    """The official 7.7.0 app and its release payload: Developer ID, reported unnotarized offline."""
    scene.fake.unnotarized.add(INSTALLED)
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert (row["state"], row["native"]["status"], row["native"]["origin"]) == ("ready", "staged", "official")


def test_a_package_gatekeeper_calls_unnotarized_is_never_staged(scene):
    scene.fake.unnotarized.add(TARGET)
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert (row["state"], row["native"]["reason"]) == ("needs_decision", "gatekeeper_rejected")


def test_a_runtime_venv_installation_stages_the_app_and_prepares_its_packages_beside_that_venv(scene):
    """The launcher's explicit ``--runtime-venv`` is no manual step: the new app is staged as for the packaged user
    site, and the generation's packages are a clone beside the venv, never in the serving venv. The clone runs the
    STAGED app's own Python while its lock installs and its checks run (its ``bin/python`` and ``pyvenv.cfg``
    name that app meanwhile), then names the installed app again exactly as before, where the exchange puts those
    bytes. The record names the venv the new app opens with and the interpreter the clone ran."""
    venv = _runtime_venv(scene, config=True)
    written = (venv / "pyvenv.cfg").read_text()
    _target_speaks(scene, 2)
    seen = {}

    def checks(row):  # W' runs <clone>/bin/python (update_checks._environments): what that names right now
        clone = pathlib.Path(row["environment"]["generation"]["user_base"])
        seen.update(link=_base_named(clone), config=(clone / "pyvenv.cfg").read_text())
        return {"ok": True, "stub": True}

    row = update_prepare.prepare(fetch=False, checks=checks)

    staged, generation = row["native"], row["environment"]["generation"]
    assert (row["state"], staged["status"], staged["runtime_venv"]) == ("ready", "staged", str(venv)), (
        row.get("reason"), staged)
    staged_python = _app_python(staged["staged_app"])
    [install] = scene.installs
    assert (install["kind"], install["python"], str(install["target"])) == ("venv", None, generation["user_base"])
    for during in (install, seen):  # the lock installed, and the checks ran, on the staged app's own Python
        assert during["link"] == str(staged_python)
        assert during["config"] == written.replace(str(_app_python(scene.app).parent), str(staged_python.parent))
    assert (generation["kind"], generation["current"], "python" in generation) == ("venv", str(venv), False)
    assert (generation["staged_python"], "links" in generation) == (str(staged_python), False)
    clone = pathlib.Path(generation["user_base"])
    assert clone.parent == venv.parent and (clone / "lib" / "serving-package").exists()
    assert (clone / "lib" / "generation-package").exists() and not (venv / "lib" / "generation-package").exists()
    assert _venv_state(clone) == _venv_state(venv)  # named back, exactly: every link and pyvenv.cfg
    assert (_base_named(clone), (clone / "pyvenv.cfg").read_text()) == (str(_app_python(scene.app)), written)
    assert os.readlink(clone / "bin" / "python") == "python3"  # the aliases themselves never change
    assert _base_named(venv) == str(_app_python(scene.app))  # the serving venv: untouched
    assert update_prepare.projection()["native"]["runtime_venv"] == str(venv)


def test_a_runtime_venv_whose_links_name_no_python_is_refused_and_never_relinked(scene):
    """A venv whose ``bin/python`` is a copy names no Python by its links, so nothing tells whether it runs the
    installed app's: it is never pointed at the new app's Python. Preparation fails with the reason, the clone goes
    and the owner's venv stays as it is."""
    venv = _runtime_venv(scene)
    for name in ("python", "python3"):
        (venv / "bin" / name).unlink()
    (venv / "bin" / "python3").write_bytes(b"\0")  # a copied interpreter, as ``venv --copies`` leaves it
    (venv / "bin" / "python").symlink_to("python3")
    owner = _venv_state(venv)
    _target_speaks(scene, 2)

    row = scene.prepare()

    assert row["state"] == "failed" and row["reason"].startswith("UPDATE_ENVIRONMENT_UNLINKED: "), row.get("reason")
    assert _venv_state(venv) == owner and (venv / "bin" / "python3").read_bytes() == b"\0" and scene.installs == []
    assert not list(venv.parent.glob(f"{venv.name}.update-*"))


def test_the_installed_app_is_the_one_the_launcher_runs_from_never_the_one_the_bodys_python_lives_in(
        tmp_path, monkeypatch):
    """App identity is the launcher's: the bundle it stamps into the server it starts, whichever Python the body runs
    on. A launcher of a source checkout runs no app; only a server no launcher started is read by its interpreter."""
    launched = make_app(tmp_path / "Applications" / "Ouroboros.app", version=INSTALLED)
    other = make_app(tmp_path / "Elsewhere" / "Ouroboros.app", version=INSTALLED)
    body = other / "Contents" / "Frameworks" / "python-standalone" / "bin" / f"python{RUNTIME}"
    body.write_bytes(b"\0")  # the body's interpreter lives in another app (read, never run)
    monkeypatch.setattr(sys, "executable", str(body))
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(launched / "Contents" / "Frameworks"))
    assert _real_installed_bundle() == launched.resolve()
    monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(tmp_path))  # a source launcher stamps its checkout
    assert _real_installed_bundle() is None
    monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER")  # no launcher: the interpreter's bundle, as before
    assert _real_installed_bundle() == other.resolve()


def test_a_runtime_venv_on_the_owners_own_python_is_updated_under_the_launchers_app_and_keeps_that_python(
        scene, monkeypatch):
    """The app is the launcher's fact, the body's Python the owner's choice. A launcher accepts a marked venv of
    any Python of its version (the marker records a version, not where that Python lives) and stamps the bundle it
    runs from into the server it starts (``OUROBOROS_BUNDLE_DIR``), so that app is found and replaced although the
    body runs outside it. The venv keeps its own Python, never pointed at the staged app's: its clone installs the
    lock and runs the checks on it, the import probe runs it, and the new app opens with the venv; the record
    carries that Python and its runtime for the readback (``venv_python``)."""
    monkeypatch.setattr(update_native, "installed_bundle", _real_installed_bundle)
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(scene.app / "Contents" / "Frameworks"))
    venv = _runtime_venv(scene)
    (venv / "bin" / "python3").unlink()
    (venv / "bin" / "python3").symlink_to(sys.executable)  # made by another Python of the app's version
    owner = _venv_state(venv)
    _target_speaks(scene, 2)
    seen = {}

    def checks(row):  # W' runs <clone>/bin/python (update_checks._environments): what that names right now
        generation = (row.get("environment") or {}).get("generation") or {}
        seen["link"] = _base_named(generation["user_base"]) if generation else None
        return {"ok": True, "stub": True}

    row = update_prepare.prepare(fetch=False, checks=checks)

    staged = row["native"]
    assert (row["state"], staged["status"], staged.get("runtime_venv")) == ("ready", "staged", str(venv)), (
        row.get("reason"), staged)
    generation = row["environment"]["generation"]
    assert pathlib.Path(staged["installed"]) == scene.app.resolve()
    assert staged["venv_python"] == os.path.realpath(sys.executable)
    assert staged["runtime"] == {"version": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version}
    [install] = scene.installs
    assert install["link"] == seen["link"] == sys.executable  # its own Python installed and checked, never the app's
    assert "staged_python" not in generation and _venv_state(generation["user_base"]) == owner == _venv_state(venv)
    world = _activate(scene, row)
    try:
        result, lines = _ended(world.log)

        assert result["status"] == "launched", lines
        handoff = scene.handoff()
        assert handoff["native"]["venv_python"] == os.path.realpath(sys.executable)
        assert "staged_python" not in handoff["environment"]  # the probe runs the venv as it is: its own Python
        assert handoff["probe_command"][0] == str(venv / "bin" / "python")
        assert (venv / "lib" / "generation-package").exists() and _base_named(venv) == sys.executable
        assert body_switch.bundle_identity(handoff["native"]["installed"]) == handoff["native"]["new_identity"]
        open_call = [line for line in world.calls.read_text().splitlines() if line.startswith("open ")][-1]
        assert open_call.endswith(f"--args --launch-intent automatic --runtime-venv {venv}"), open_call
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_a_virtual_environment_its_launcher_did_not_select_stays_a_manual_step(scene):
    """Only a venv the launcher's ``runtime_venv`` recorded is one a relaunch with ``--runtime-venv`` selects again;
    any other would be refused by the new app, so nothing is downloaded or staged for it."""
    _runtime_venv(scene, marked=False)
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert (row["state"], row["native"]["reason"]) == ("needs_decision", "environment_venv")
    assert not any(url.endswith(f"Ouroboros-{TARGET}.dmg") for url in scene.fetched) and scene.installs == []


def test_a_failure_while_the_clone_runs_the_staged_python_points_only_the_clone_back_exactly(scene):
    """The checks crash while the clone names the staged app's Python: the candidate fails with that error, the
    clone (the update's private copy) names the installed app's Python again exactly as the owner's venv does,
    the record no longer claims a pointing to undo, and the owner's venv was never touched."""
    venv = _runtime_venv(scene, config=True)
    owner = _venv_state(venv)
    _target_speaks(scene, 2)
    seen = {}

    def checks(row):
        clone = pathlib.Path(row["environment"]["generation"]["user_base"])
        seen.update(clone=clone, during=_base_named(clone), owner=_venv_state(venv))
        raise RuntimeError("a check crashed")

    row = update_prepare.prepare(fetch=False, checks=checks)

    assert (row["state"], row["reason"]) == ("failed", "RuntimeError: a check crashed")
    assert seen["during"] == str(_app_python(row["native"]["staged_app"]))  # it did run the staged Python
    assert seen["owner"] == owner and _venv_state(venv) == owner
    assert _venv_state(seen["clone"]) == owner
    assert "links" not in update_prepare.find(row["candidate_id"])["environment"]["generation"]


def test_activation_refuses_a_body_started_with_another_runtime_venv_than_prepared(scene, monkeypatch):
    venv = _runtime_venv(scene)
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    monkeypatch.setattr(update_native, "launcher_pid", lambda: os.getppid())
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("user_site", scene.userbase))  # no venv now
    with pytest.raises(body_adoption.AdoptionRefused) as refused:
        body_adoption.authorize_update(scene.data, row)
    assert refused.value.code == "ADOPTION_NATIVE_CHANGED" and body_adoption.read(scene.data) == {}
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))
    handoff = body_adoption.authorize_update(scene.data, row)
    assert handoff["native"]["runtime_venv"] == str(venv)
    assert handoff["native"]["launch"]["args"] == ["--launch-intent", "automatic", "--runtime-venv", str(venv)]
    # The tree's import probe runs the interpreter the body will run: the venv's own path, on the staged app's Python
    # that path names once the exchange is done (before it, the previous app's: body_switch._probe_command).
    assert handoff["probe_command"][0] == str(venv / "bin" / "python")
    assert handoff["environment"]["staged_python"] == str(_app_python(row["native"]["staged_app"]))


def test_a_candidate_staged_with_the_earlier_two_field_app_identity_is_stale_and_prepared_again(scene):
    """An app staged before ``Contents`` was part of its identity (``[device, inode]``) proves no bundle any more:
    Restart to update finds the candidate stale and asks for a new preparation before anything stops."""
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    earlier = {**row["native"], "previous_identity": row["native"]["previous_identity"][:2],
               "new_identity": row["native"]["new_identity"][:2]}
    with update_prepare.lifecycle():
        row = update_prepare.mark(row, native=earlier)
        why = update_prepare.activation_check(row)

    assert why.startswith("native_identity_stale: ") and why.endswith("prepare it again"), why
    stale = update_prepare.find(row["candidate_id"])
    assert (stale["state"], stale["reason"]) == ("stale", "native_identity_stale")
    assert update_prepare.projection()["actions"] == ["prepare", "discard_candidate"]


def test_preparing_again_after_a_local_edit_stages_the_new_candidates_app_beside_the_superseded_one(scene, monkeypatch):
    """The same package version, prepared twice: an ordinary local edit, then prepare again. Each candidate
    owns its own staging, so the replacement is ready, and the superseded stage leaves only with its own
    retirement, which waits while an installer may still run from it."""
    from ouroboros import subagent_worktrees

    _target_speaks(scene, 2)
    scene.local_work(commit=False)
    first = scene.prepare()
    assert (first["state"], first["native"]["status"]) == ("ready", "staged"), first.get("reason")
    (scene.serving / "local-scratch.txt").write_text("edited after the first preparation\n")

    second = scene.prepare()

    assert (second["state"], second["native"]["status"]) == ("ready", "staged"), (second.get("reason"),
                                                                                 second["native"])
    assert update_prepare.find(first["candidate_id"])["state"] == "superseded"
    staged = [pathlib.Path(row["native"]["staged_app"]) for row in (first, second)]
    assert staged[0].parent != staged[1].parent and all(path.is_dir() for path in staged)
    assert body_switch.bundle_identity(str(staged[1])) == second["native"]["new_identity"]
    monkeypatch.setattr(update_native, "installer_pending", lambda _data: ["pid 4242 (alive)"])
    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task_id: False)
    assert staged[0].is_dir() and update_prepare.find(first["candidate_id"]) is not None
    monkeypatch.setattr(update_native, "installer_pending", lambda _data: [])
    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task_id: False)
    assert not staged[0].parent.exists() and update_prepare.find(first["candidate_id"]) is None
    assert body_switch.bundle_identity(str(staged[1])) == second["native"]["new_identity"]


# --------------------------------------------------------------------------- #
# The app-only Update: the source is current, the app cannot host it
# --------------------------------------------------------------------------- #
def _current_source_under_an_app_without_its_entry(scene):
    """The owner's 7.7 case: the serving source already is the channel's target and declares protocol 1;
    the app has no signed entry, so it needs the published 7.8.0 app."""
    scene.app = make_app(scene.tmp / "Installed" / "Ouroboros.app", version="7.7.0", entry=False)
    scene.published.append(_release(scene, "7.7.0", None))
    (scene.serving / "launcher.py").write_text(LAUNCHER_PY + "\nMACOS_SOURCE_ENTRY_PROTOCOL = 1\n")
    git(scene.serving, "commit", "-qam", "the current official source: protocol 1")
    target = git(scene.serving, "rev-parse", "HEAD")
    scene.monkeypatch.setattr(update_prepare, "resolve_target", lambda **_kw: {
        "channel": scene.channel, "target_ref": "official", "target_sha": target})
    return target


def test_an_app_only_update_keeps_every_local_layer_exactly_and_makes_no_commit(scene, monkeypatch):
    _current_source_under_an_app_without_its_entry(scene)
    scene.local_work()  # a local commit, a staged and an unstaged edit of one file, an untracked file
    before = scene.composition()
    assert update_prepare.projection()["app_needed"]["reason"] == "signed_entry_missing"  # the panel can say so
    branches = lambda: [line for line in git(scene.serving, "for-each-ref", "--format=%(refname) %(objectname)",
                                             "refs/heads").splitlines() if not line.startswith("refs/heads/candidate/")]
    refs_before = branches()

    row = scene.prepare()

    assert row["state"] == "ready", (row.get("reason"), row.get("native"))
    layers = row["layers"]
    assert (layers["old_commit"], layers["old_index"], layers["old_files"]) == (
        layers["new_commit"], layers["new_index"], layers["new_files"])
    assert layers["new_commit"] == before["head"] and row["merge"]["kind"] == "current"
    assert (row["native"]["status"], row["native"]["version"]) == ("staged", "7.8.0")
    assert scene.composition() == before and branches() == refs_before  # no branch of the owner moved
    assert update_prepare.projection()["merge_kind"] == "current"
    with pytest.raises(body_adoption.AdoptionRefused) as refused:
        body_adoption.authorize_update(scene.data, row)  # not started by the app's launcher: no app step
    assert refused.value.code == "ADOPTION_NATIVE_UNMANAGED" and body_adoption.read(scene.data) == {}
    monkeypatch.setattr(update_native, "launcher_pid", lambda: os.getppid())
    handoff = body_adoption.authorize_update(scene.data, row)
    assert handoff["switch"] == [] and handoff["old"] == handoff["cand"] == before["head"]
    assert handoff["native"]["version"] == "7.8.0" and handoff["environment"]["kind"] == "user_site"
    assert handoff["probe_command"][0] == row["native"]["python"]  # the new app's interpreter probes the tree
    assert (body_adoption.helper_dir(scene.data) / "app_install.py").is_file()


def test_a_current_source_under_an_app_that_hosts_it_is_up_to_date(scene):
    _current_source_under_an_app_without_its_entry(scene)
    scene.app = make_app(scene.tmp / "Hosting" / "Ouroboros.app", version="7.8.0")
    assert "app_needed" not in update_prepare.projection()
    with pytest.raises(update_prepare.PrepareRefused) as refused:
        scene.prepare()
    assert refused.value.code == "UPDATE_CURRENT"


# --------------------------------------------------------------------------- #
# Activation with the real captured installer
# --------------------------------------------------------------------------- #
def _orphan():
    """A sleeping process reparented as under launchd (which reaps it once it is killed)."""
    return int(subprocess.run(["/bin/sh", "-c", f"{json.dumps(sys.executable)} -c 'import time; time.sleep(300)' "
                               ">/dev/null 2>&1 & echo $!"], capture_output=True, text=True, check=True).stdout)


def _tool(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def _stop(pid):
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


def _until(check, timeout=60):
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("the condition did not hold in time")
        time.sleep(0.2)


def _activate(scene, row, *, quits=True, arm=True, quit_body="", after_quit=""):
    """The activation's calls, in order, around the old app's two real processes and the stand-in tools.

    The staged entry runs this interpreter, as the signed entry runs the app's; the stand-in
    ``osascript`` ends the old launcher and server, as the old app's Quit does through its pipe
    (``quit_body`` replaces what it does after recording its call; ``after_quit`` runs once they ended).
    """
    launcher, server = _orphan(), _orphan()
    scene.monkeypatch.setattr(update_native, "launcher_pid", lambda: launcher)
    scene.monkeypatch.setattr(update_native, "server_pid", lambda: server)
    calls, opened = scene.tmp / "tool-calls", scene.tmp / "open-environment"  # what the new app would start with
    tools = {"osascript": _tool(scene.tmp / "osascript", f'echo "osascript $@" >> {json.dumps(str(calls))}\n'
                                + (quit_body or (f"kill {launcher} {server}\n" if quits else "")) + after_quit),
             "open": _tool(scene.tmp / "open", f'echo "open $@" >> {json.dumps(str(calls))}\n'
                                               f'/usr/bin/env -0 > {json.dumps(str(opened))}\n')}
    entry = pathlib.Path(row["native"]["staged_app"]) / "Contents" / "MacOS" / "ouroboros-shell"
    entry.write_text(f"#!/bin/sh\nexec {json.dumps(sys.executable)} \"$@\"\n")
    entry.chmod(0o755)
    handoff = body_adoption.authorize_update(scene.data, row)
    assert body_adoption.bind_update_restart(scene.data)
    command = update_native.installer_command(row["native"], data_dir=scene.data, adoption_id=handoff["id"],
                                              tools=tools)
    assert command["ok"] and command["argv"][:3] == [str(entry), "-I",
                                                     str(body_adoption.helper_dir(scene.data) / "app_install.py")]
    started = update_native.hand_over(command, data_dir=scene.data, adoption_id=handoff["id"])
    assert started["ok"], started
    world = SimpleNamespace(handoff=handoff, launcher=launcher, server=server, calls=calls, started=started,
                            log=pathlib.Path(command["log"]), command=command, opened=opened)
    if arm:  # the restart's exit tail, on the stop owners' evidence
        assert body_adoption.arm(scene.data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                                 owner_restart=True) == "armed"
    return world


def _ended(log, timeout=120):
    """The installer's last line (``end``) and every JSON line of its log."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lines = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")] \
            if log.exists() else []
        ends = [line for line in lines if line.get("step") == "end"]
        if ends:
            return ends[-1], lines
        time.sleep(0.2)
    raise AssertionError("the installer did not finish: " + (log.read_text() if log.exists() else "no log"))


def _app_python(app):
    """The interpreter inside ``app`` (a link to this one here): what the app's entries run."""
    return pathlib.Path(app) / "Contents" / "Frameworks" / "python-standalone" / "bin" / "python3"


def _enter(scene, python):
    """A real cold server entry of the serving checkout under that interpreter."""
    # PYTHONEXECUTABLE: a framework build reports its own binary otherwise; an app's interpreter is its path.
    return subprocess.run([str(python), "server.py"], cwd=str(scene.serving), capture_output=True, text=True,
                          timeout=120, env={**os.environ, "TOY_REPORT": str(scene.report),
                                            "PYTHONEXECUTABLE": str(python)})


def test_the_installer_switches_tree_and_packages_then_exchanges_the_apps_and_opens_the_new_one(scene):
    _target_speaks(scene, 2)
    scene.local_work(commit=False)
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    world = _activate(scene, row)
    try:
        result, lines = _ended(world.log)

        assert result["status"] == "launched", lines
        handoff = scene.handoff()
        native = handoff["native"]
        assert [step["state"] for step in native["steps"]] == [
            "in_custody", "quitting", "locked", "exchanging", "exchanged", "backed_up", "opened"]
        assert native["steps"][0]["ppid"] == 1  # detached from the activating server's tree before anything ran
        assert native["installer"]["pid"] == world.started["pid"]
        # The tree: the merge on the branch; staged, unstaged and untracked work carried, none of it history.
        assert handoff["phase"] == "switched" and git(scene.serving, "rev-parse", "HEAD") == handoff["cand"]
        assert git(scene.serving, "show", ":ouroboros/mod_b.py") + "\n" == STAGED
        assert (scene.serving / "ouroboros" / "mod_b.py").read_text() == UNSTAGED
        assert (scene.serving / "local-scratch.txt").read_text() == "mine, untracked\n"
        # The packages: the generation's at the one serving path, the previous ones kept beside for a return.
        assert (scene.userbase / "lib" / "generation-package").exists()
        previous = pathlib.Path(handoff["environment"]["previous"])
        assert (previous / "lib" / "serving-package").exists() and not (previous / "lib" / "generation-package").exists()
        # The apps: exchanged by identity, the previous one the backup beside, nothing left staged.
        assert body_switch.bundle_identity(native["installed"]) == native["new_identity"]
        assert body_switch.bundle_identity(native["backup"]) == native["previous_identity"]
        assert (_version_at(native["installed"]), _version_at(native["backup"])) == (TARGET, INSTALLED)
        assert not pathlib.Path(native["staged"]).exists() and not (scene.tmp / "ouroboros.pid").exists()
        quit_call, open_call = world.calls.read_text().splitlines()
        assert quit_call == f'osascript -e tell application "{native["installed"]}" to quit'
        assert open_call.startswith(f"open {native['installed']} --env ") and open_call.endswith(
            "--args --launch-intent automatic")
        assert f"--env OUROBOROS_DATA_DIR={scene.data}" in open_call  # the roots the launcher was started with
        assert [entry["kind"] for entry in handoff["entries"]] == ["app"]  # recorded before the new app could run

        # The new app's first server entry imports the switched tree and is recorded.
        booted = _enter(scene, _app_python(native["installed"]))
        assert booted.returncode == 0, booted.stderr
        assert scene.reports()[-1]["a"] == "GEN_CAND"
        assert [entry["kind"] for entry in scene.handoff()["entries"]] == ["app", "server"]
        # This test process is not that app's interpreter, nor started by its signed entry: never adopted.
        body_switch._attest_loaded(str(scene.serving), scene.handoff())
        settled = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
        assert settled["outcome"] == "start_failed" and "successor_entered:app" in settled["reasons"]
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_the_new_app_opens_with_this_installations_configuration_and_none_of_the_old_generations_role(
        scene, monkeypatch):
    """``open`` hands its own environment to the app it starts and ``--env`` only adds to it. The installer
    inherits the activating server's, so it opens the new app with the inputs this installation was launched
    with, the configuration this generation resolved (roots, settings, lock, port file) and its home and
    profile, and without anything the old launcher, server, custody or frozen bootloader put there."""
    from ouroboros import config, process_scope
    from ouroboros.process_census import SCOPE_VALUE
    from ouroboros.settings_integrity import PROJECTED_SETTINGS_ENV, _value_digest

    _target_speaks(scene, 2)
    row = scene.prepare()
    profile, port = scene.tmp / "profile-home", scene.tmp / "relocated" / "server_port"
    profile.mkdir()
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", scene.tmp / "chosen-settings.json")
    inputs = {"CFFIXED_USER_HOME": str(profile), "OUROBOROS_PORT_FILE": str(port),
              "HTTPS_PROXY": "http://proxy.invalid:3128",  # the old launch's own inputs, an operator's UI choice too
              "OUROBOROS_UI_DIR": str(scene.tmp / "operator-ui"),
              # The operator's port, as the launcher stamped it into its server, and documented env-only inputs.
              "OUROBOROS_SERVER_PORT": "28817", "OUROBOROS_WORKER_START_METHOD": "spawn",
              "OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD": "1"}
    port.parent.mkdir(parents=True)
    port.write_text("28818")  # the port that server fell back to: the bound one, never the operator's choice
    projected = "sk-projected-from-the-settings-document"
    role = {"OUROBOROS_APP_VERSION": "7.7.0", "OUROBOROS_SERVER_PROCESS_PID": "3226",
            "OUROBOROS_SERVER_HOST": "127.0.0.1",
            "OUROBOROS_LAUNCHER_STOP_STDIN": "1", "OUROBOROS_MANAGED_BY_LAUNCHER": "1",
            "OUROBOROS_LAUNCH_INTENT": "owner", "PYTHONPATH": str(scene.serving), "PYTHONUSERBASE": str(scene.userbase),
            "_PYI_ARCHIVE_FILE": str(scene.app / "Contents" / "MacOS" / "Ouroboros"), "_PYI_PARENT_PROCESS_LEVEL": "1",
            "XPC_SERVICE_NAME": "application.com.ouroboros.agent.1.2",
            process_scope.installation_marker(scene.data, create=True): SCOPE_VALUE,
            "OPENAI_API_KEY": projected,
            PROJECTED_SETTINGS_ENV: json.dumps({"OPENAI_API_KEY": _value_digest(projected)})}
    for key, value in {**inputs, **role}.items():
        monkeypatch.setenv(key, value)
    world = _activate(scene, row)
    try:
        result, lines = _ended(world.log)

        assert result["status"] == "launched", lines
        handoff = scene.handoff()
        expected = {"OUROBOROS_APP_ROOT": str(config.resolve_app_root()), "OUROBOROS_REPO_DIR": handoff["repo_dir"],
                    "OUROBOROS_DATA_DIR": handoff["data_dir"], "OUROBOROS_SETTINGS_PATH": str(config.SETTINGS_PATH),
                    "OUROBOROS_PID_FILE": str(config.PID_FILE), "OUROBOROS_PORT_FILE": str(port),
                    "OUROBOROS_SERVER_PORT": "28817", "HOME": os.path.expanduser("~"), "CFFIXED_USER_HOME": str(profile)}
        assert handoff["native"]["launch"]["env"] == expected
        open_call = [line for line in world.calls.read_text().splitlines() if line.startswith("open ")][-1]
        assert all(f"--env {key}={value}" in open_call for key, value in expected.items()), open_call
        seen = dict(item.split("=", 1) for item in world.opened.read_text().split("\0") if "=" in item)
        assert {key: seen.get(key) for key in expected} == expected  # what the new app starts with
        operator = ("HTTPS_PROXY", "OUROBOROS_UI_DIR", "OUROBOROS_WORKER_START_METHOD",
                    "OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD")
        assert {key: seen.get(key) for key in operator} == {key: inputs[key] for key in operator}
        assert seen["OUROBOROS_SERVER_PORT"] == "28817"  # the operator's port, never the 28818 fallback
        assert sorted(key for key in role if key in seen) == []
        assert not set(role) & set(handoff["native"]["launch"]["inherit"])  # names only, and none of the role
        recorded = json.dumps(handoff)  # no inherited value is ever written into the record
        assert projected not in recorded and inputs["HTTPS_PROXY"] not in recorded and "28818" not in recorded
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_the_installer_opens_the_new_app_in_the_runtime_venv_whose_packages_it_switched_in_place(scene):
    """The owner's ``--runtime-venv`` survives the app replacement: the venv's path keeps its name and gets the
    generation's packages (its console scripts retargeted to it), the previous packages stay beside it, the probe ran
    the venv's packages on the staged app's Python, and the new app opens with ``--runtime-venv`` at that path. The
    venv's interpreter names the installed app's Python, which the exchange makes the new app's."""
    venv = _runtime_venv(scene)
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    world = _activate(scene, row)
    try:
        result, lines = _ended(world.log)

        assert result["status"] == "launched", lines
        handoff = scene.handoff()
        native = handoff["native"]
        assert (handoff["environment"]["kind"], handoff["environment"]["state"]) == ("venv", "selected")
        assert handoff["probe_command"][0] == str(venv / "bin" / "python")
        assert handoff["environment"]["staged_python"] == str(_app_python(row["native"]["staged_app"]))
        assert (venv / "lib" / "generation-package").exists() and (venv / runtime_venv.MARKER_NAME).is_file()
        previous = pathlib.Path(handoff["environment"]["previous"])
        assert previous.parent == venv.parent and (previous / "lib" / "serving-package").exists()
        assert (venv / "bin" / "generation-tool").read_text().splitlines()[0] == f"#!{venv}/bin/python"
        assert _base_named(venv) == str(_app_python(native["installed"]))
        assert os.readlink(venv / "bin" / "python") == "python3"
        assert body_switch.bundle_identity(native["installed"]) == native["new_identity"]  # that path: the new app
        open_call = [line for line in world.calls.read_text().splitlines() if line.startswith("open ")][-1]
        assert open_call.endswith(f"--args --launch-intent automatic --runtime-venv {venv}"), open_call
    finally:
        _stop(world.launcher)
        _stop(world.server)


@pytest.mark.parametrize("unable, outcome", [("staged", ("not_applied", "returned")),
                                             ("installed", ("launched", None))])
def test_the_switch_probes_a_venv_of_the_apps_python_on_the_staged_apps_python_and_changes_no_venv(
        scene, unable, outcome):
    """A ``--runtime-venv`` of the installed app's Python runs the staged app's once the exchange is done, and the
    switch's import probe runs before the exchange: it runs the selected venv's packages on the staged app's Python,
    never on the previous app's. So a new runtime that cannot import the new tree returns it and reopens the previous
    app, and a previous runtime that could not is no reason to. Neither the owner's venv nor the generation's clone
    changes for the probe: each keeps every link exactly as prepared."""
    venv = _runtime_venv(scene)
    _target_speaks(scene, 2)
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    clone = pathlib.Path(row["environment"]["generation"]["user_base"])
    owner, prepared = _venv_state(venv), _venv_state(clone)
    unable_python = _app_python(row["native"]["staged_app"] if unable == "staged" else scene.app)
    stand_in = _tool(scene.tmp / "unable-python", "echo 'this runtime cannot import the new tree' >&2\nexit 3\n")
    os.replace(stand_in, unable_python)  # replaces the app's link itself, never the interpreter it named
    world = _activate(scene, row)
    try:
        result, lines = _ended(world.log)

        assert (result["status"], result["reason"]) == outcome, lines
        handoff = scene.handoff()
        assert handoff["environment"]["staged_python"] == str(_app_python(row["native"]["staged_app"]))
        if unable == "staged":
            assert (handoff["phase"], handoff["return"]["trigger"]) == ("returned", "import_probe_failed")
            assert "this runtime cannot import the new tree" in handoff["return"]["detail"]
            assert _venv_state(venv) == owner and _venv_state(clone) == prepared  # each back, exactly as it was
            assert (venv / "lib" / "serving-package").exists() and not (venv / "lib" / "generation-package").exists()
            assert body_switch.bundle_identity(handoff["native"]["installed"]) == handoff["native"]["previous_identity"]
        else:
            assert handoff["phase"] == "switched" and (venv / "lib" / "generation-package").exists()
            assert _venv_state(venv) == prepared  # the generation's clone, its links exactly as prepared
            assert _base_named(venv) == str(_app_python(handoff["native"]["installed"]))
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_an_app_another_installer_put_in_place_before_the_swap_is_not_swapped_away(scene):
    """An app another installer moved into place between the old app's Quit and the swap (by hand, or a Homebrew
    upgrade without app-management permission, which replaces the folder): the installed path no longer holds the
    bundle folder this update was prepared against, so the installer swaps nothing, returns the composition and
    reopens what is installed. The check narrows that window and closes none, and an in-place replacement of the
    folder's contents keeps its identity (``app_install``)."""
    _target_speaks(scene, 2)
    row = scene.prepare()
    installed = pathlib.Path(row["native"]["installed"])
    brewed = make_app(scene.tmp / "brew-staging" / "Ouroboros.app", version="7.9.5")  # what the cask now names
    moved = scene.tmp / "moved-by-brew.app"
    world = _activate(scene, row, after_quit=f"mv {json.dumps(str(installed))} {json.dumps(str(moved))}\n"
                                             f"mv {json.dumps(str(brewed))} {json.dumps(str(installed))}\n")
    try:
        result, lines = _ended(world.log)

        assert (result["status"], result["reason"]) == ("not_applied", "exchange_failed"), lines
        handoff = scene.handoff()
        assert (handoff["phase"], handoff["return"]["trigger"]) == ("returned", "app_changed")
        assert git(scene.serving, "rev-parse", "HEAD") == scene.base
        assert (scene.userbase / "lib" / "serving-package").exists()  # the serving packages are back at their path
        assert not (scene.userbase / "lib" / "generation-package").exists()
        assert _version_at(installed) == "7.9.5"  # Homebrew's app stays where it put it
        staged = pathlib.Path(row["native"]["staged_app"])
        assert body_switch.bundle_identity(str(staged)) == row["native"]["new_identity"]  # never swapped in
        assert [step["state"] for step in handoff["native"]["steps"]][-2:] == ["exchange_refused", "reopened"]
        assert not [entry for entry in handoff["entries"] if entry.get("kind") == "app"]  # no new app ever ran
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_an_installer_that_died_after_its_switch_returns_the_tree_at_the_previous_apps_next_entry(scene, monkeypatch):
    _target_speaks(scene, 2)
    scene.local_work(commit=False)
    before = scene.composition()
    row = scene.prepare()
    monkeypatch.setenv("OUROBOROS_APP_INSTALL_FAIL_AT", "after_switch")
    world = _activate(scene, row)
    try:
        _until(lambda: body_switch.process_gone(scene.handoff()["native"]["installer"]))
        handoff = scene.handoff()
        assert handoff["phase"] == "switched" and git(scene.serving, "rev-parse", "HEAD") == handoff["cand"]
        native = handoff["native"]
        assert body_switch.bundle_identity(native["installed"]) == native["previous_identity"]  # never exchanged

        booted = _enter(scene, _app_python(native["installed"]))  # the owner opens the previous app again

        assert booted.returncode == 0, booted.stderr
        assert scene.composition() == before and scene.reports()[-1]["a"] == "GEN_OLD"
        assert (scene.handoff()["phase"], scene.handoff()["return"]["trigger"]) == ("returned", "app_not_exchanged")
        assert (scene.userbase / "lib" / "serving-package").exists()
        assert not (scene.userbase / "lib" / "generation-package").exists()  # its own packages back in place
        settled = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
        assert settled["outcome"] == "returned" and "data was not rolled back" in settled["note"]
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_an_installer_that_died_after_its_exchange_leaves_the_new_app_and_the_backup_to_the_gc(scene, monkeypatch):
    _target_speaks(scene, 2)
    row = scene.prepare()
    monkeypatch.setenv("OUROBOROS_APP_INSTALL_FAIL_AT", "after_exchange")
    world = _activate(scene, row)
    try:
        _until(lambda: body_switch.process_gone(scene.handoff()["native"]["installer"]))
        native = scene.handoff()["native"]
        assert body_switch.bundle_identity(native["installed"]) == native["new_identity"]
        assert body_switch.bundle_identity(native["staged"]) == native["previous_identity"]  # its backup is owed

        booted = _enter(scene, _app_python(native["installed"]))  # the new app's next start goes on

        assert booted.returncode == 0 and scene.reports()[-1]["a"] == "GEN_CAND"
        assert scene.handoff()["phase"] == "switched"
        assert [entry["kind"] for entry in scene.handoff()["entries"]] == ["app", "server"]
        assert update_native.release_staging(row["native"]) is True  # the GC's retirement finishes the backup
        assert body_switch.bundle_identity(native["backup"]) == native["previous_identity"]
        assert not pathlib.Path(row["native"]["staging_dir"]).exists()
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_retirement_keeps_a_previous_app_an_earlier_two_field_identity_names_as_its_backup(tmp_path):
    """A record from before ``Contents`` was part of the identity names its previous app by folder alone. An app the
    installer left at the staged path (stopped between its exchange and the backup) still becomes that backup: it
    is never deleted with the staging folder."""
    folder = tmp_path / "Applications"
    installed = make_app(folder / "Ouroboros.app", version=TARGET)
    left = make_app(folder / f".Ouroboros-update-{TARGET}-cand" / "Ouroboros.app", version=INSTALLED)
    backup = folder / f".Ouroboros-{INSTALLED}-backup.app"
    record = {"installed": str(installed), "staging_dir": str(left.parent), "staged_app": str(left),
              "backup_app": str(backup), "previous_identity": body_switch.bundle_identity(str(left))[:2]}

    assert update_native.release_staging(record)

    assert _version_at(backup) == INSTALLED and not left.parent.exists() and _version_at(installed) == TARGET
    assert body_switch.bundle_identity(str(backup))[:2] == record["previous_identity"]
    assert body_switch.identity_stale(record["previous_identity"])


def test_an_entry_never_switches_an_app_transition_and_abandons_it_once_its_installer_is_gone(scene):
    _target_speaks(scene, 2)
    row = scene.prepare()
    world = _activate(scene, row, quits=False)  # the old app does not quit: the installer waits
    try:
        _until(lambda: world.calls.exists())
        old = _enter(scene, _app_python(scene.app))  # an entry under the previous app meanwhile
        assert old.returncode == 0 and scene.reports()[-1]["a"] == "GEN_OLD"
        assert scene.handoff()["phase"] == "armed" and git(scene.serving, "rev-parse", "HEAD") == scene.base
        assert body_adoption.native_hold(scene.data) == "hold"  # the activating server keeps holding its exit
        _stop(world.started["pid"])
        _until(lambda: body_switch.process_gone(scene.handoff()["native"]["installer"]))

        again = _enter(scene, _app_python(scene.app))

        assert again.returncode == 0 and scene.reports()[-1]["a"] == "GEN_OLD"
        assert scene.handoff()["phase"] == "abandoned" and "installer_gone" in scene.handoff()["events"][-1]["detail"]
        assert git(scene.serving, "rev-parse", "HEAD") == scene.base and _version_at(scene.app) == INSTALLED
        assert body_adoption.native_hold(scene.data) == "recycle"  # nothing is held: the old app restarts
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_the_activating_server_recycles_once_a_dead_installer_is_abandoned_with_nothing_switched(scene):
    _target_speaks(scene, 2)
    row = scene.prepare()
    world = _activate(scene, row, quits=False)
    try:
        _until(lambda: world.calls.exists())
        _stop(world.started["pid"])
        _until(lambda: body_switch.process_gone(scene.handoff()["native"]["installer"]))
        assert body_adoption.native_hold(scene.data) == "recycle"
        assert scene.handoff()["phase"] == "abandoned" and git(scene.serving, "rev-parse", "HEAD") == scene.base
        settled = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
        assert settled["outcome"] == "not_applied"
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_a_new_tree_that_does_not_import_on_the_new_interpreter_is_returned_and_the_previous_app_reopened(scene):
    scene.official({"VERSION": f"{TARGET}\n", "ouroboros/mod_a.py": "GEN = (\n",  # does not even parse
                    "launcher.py": LAUNCHER_PY + "\nMACOS_SOURCE_ENTRY_PROTOCOL = 2\n"})
    scene.local_work(commit=False)
    before = scene.composition()
    row = scene.prepare()
    world = _activate(scene, row)
    try:
        result, lines = _ended(world.log)

        assert (result["status"], result["reason"]) == ("not_applied", "returned"), lines
        handoff = scene.handoff()
        assert (handoff["phase"], handoff["return"]["trigger"]) == ("returned", "import_probe_failed")
        assert scene.composition() == before and (scene.userbase / "lib" / "serving-package").exists()
        native = handoff["native"]
        assert body_switch.bundle_identity(native["installed"]) == native["previous_identity"]
        assert pathlib.Path(native["staged"]).is_dir() and handoff["entries"] == []  # the new app never ran
        quit_call, reopen_call = world.calls.read_text().splitlines()
        assert reopen_call.startswith(f"open {native['installed']} ")  # the previous app reopened, unchanged
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_the_installer_outlives_a_tree_kill_of_the_server_that_started_it_and_stays_in_custody(scene):
    """The old app's forced stop of its server kills that server's descendants: the installer is none of them."""
    from ouroboros.app_install import INSTALLER_PURPOSE
    from ouroboros.platform_layer import kill_pid_tree
    from ouroboros.process_custody import ledger_path

    _target_speaks(scene, 2)
    row = scene.prepare()
    entry = pathlib.Path(row["native"]["staged_app"]) / "Contents" / "MacOS" / "ouroboros-shell"
    entry.write_text(f"#!/bin/sh\nexec {json.dumps(sys.executable)} \"$@\"\n")
    entry.chmod(0o755)
    scene.monkeypatch.setattr(update_native, "launcher_pid", lambda: os.getppid())
    handoff = body_adoption.authorize_update(scene.data, row)
    body_adoption.bind_update_restart(scene.data)
    tools = {"osascript": _tool(scene.tmp / "osascript", "exit 0\n"), "open": _tool(scene.tmp / "open", "exit 0\n")}
    command = update_native.installer_command(row["native"], data_dir=scene.data, adoption_id=handoff["id"],
                                              tools=tools)
    script = scene.tmp / "activating_server.py"
    script.write_text(textwrap.dedent("""
        import json, sys, time
        from supervisor import update_native
        print(json.dumps(update_native.hand_over(json.loads(sys.argv[1]), data_dir=sys.argv[2],
                                                 adoption_id=sys.argv[3])), flush=True)
        time.sleep(300)
    """))
    server = subprocess.Popen([sys.executable, str(script), json.dumps(command), str(scene.data), handoff["id"]],
                              cwd=str(REPO), stdout=subprocess.PIPE, text=True, start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(REPO)})
    installer = 0
    try:
        started = json.loads(server.stdout.readline())
        assert started["ok"], started
        installer = started["pid"]
        parent = subprocess.run(["ps", "-o", "ppid=", "-p", str(installer)], capture_output=True, text=True)
        assert parent.stdout.strip() == "1"  # no descendant of the server that started it

        kill_pid_tree(server.pid, include_process_group=True)  # the launcher's forced stop of that server
        server.wait(timeout=30)

        assert not body_switch.process_gone(scene.handoff()["native"]["installer"])  # it lives on
        rows = [json.loads(line) for line in ledger_path(scene.data).read_text().splitlines()]
        held = [entry for entry in rows if entry.get("purpose") == INSTALLER_PURPOSE]
        assert [(entry["pid"], entry["scope"]) for entry in held] == [(installer, "daemon")]
        assert update_native.installer_pending(scene.data)
        body_adoption.abandon(scene.data, "never_armed", candidate_id=row["candidate_id"])
        result, _lines = _ended(pathlib.Path(command["log"]))
        assert (result["status"], result["reason"]) == ("refused", "not_armed")  # nothing quit, swapped or opened
        assert _version_at(scene.app) == INSTALLED and pathlib.Path(row["native"]["staged_app"]).is_dir()
    finally:
        if server.poll() is None:
            kill_pid_tree(server.pid, include_process_group=True)
        if installer:
            _stop(installer)


def test_panics_installation_census_kills_the_installer_that_every_ordinary_stop_spares(scene, monkeypatch):
    """Custody is membership, not a parent or a label. The installer inherits this installation's scope
    marker (the server plants it before any child), so the activating server's exit-tail stop and the next
    generation's reaper spare it, while Panic's installation census selects and kills it, here while it is
    blocked in its quit step and reads no Panic flag at all (no flag is ever written in this test)."""
    from ouroboros import owned_shutdown, process_scope
    from ouroboros.process_census import SCOPE_VALUE
    from ouroboros.process_custody import reap_orphaned_processes

    # This scratch root's own marker, minted for it alone, planted as ``initialize_installation_scope`` does:
    # no process of any other installation carries it, so the census below can signal nothing else.
    marker = process_scope.installation_marker(scene.data, create=True)
    monkeypatch.setenv(marker, SCOPE_VALUE)
    _target_speaks(scene, 2)
    row = scene.prepare()
    tool_pid = scene.tmp / "quit-tool.pid"
    blocked = (f"echo $$ > {json.dumps(str(tool_pid))}\n"
               f"exec {json.dumps(sys.executable)} -c 'import time; time.sleep(600)'\n")  # Python: its env is readable
    world = _activate(scene, row, quit_body=blocked)
    tool = 0
    try:
        _until(lambda: tool_pid.exists() and tool_pid.read_text().strip())
        tool, installer = int(tool_pid.read_text()), world.started["pid"]
        monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
        stopped = owned_shutdown.stop_owned_work(scene.data)  # the activating server's own exit-tail stop
        reaped = reap_orphaned_processes(scene.data, running_task_ids=set())  # the next generation's reaper
        assert stopped["state"] == "completed" and installer not in reaped
        assert not body_switch.process_gone(scene.handoff()["native"]["installer"])

        outcome = process_scope.panic_scope_census(scene.data, exclude={os.getpid(), os.getppid(), 1})

        assert outcome["requested"] is True and {installer, tool} <= set(outcome["ended"]), outcome
        _until(lambda: body_switch.process_gone(scene.handoff()["native"]["installer"]), timeout=15)
        # Ended inside its quit: the transition stays armed, nothing was switched, swapped or opened.
        assert scene.handoff()["phase"] == "armed" and git(scene.serving, "rev-parse", "HEAD") == scene.base
        assert _version_at(scene.app) == INSTALLED and not body_switch.bundle_identity(row["native"]["backup_app"])
    finally:
        for pid in (tool, world.started["pid"], world.launcher, world.server):
            if pid:
                _stop(pid)


def test_a_panic_before_the_activation_arms_changes_nothing(scene):
    _target_speaks(scene, 2)
    row = scene.prepare()
    world = _activate(scene, row, arm=False)
    try:
        (scene.data / "state" / "panic_stop.flag").write_text("panic")
        assert body_adoption.arm(scene.data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                                 owner_restart=True) == "authorized"  # Panic never arms
        result, _lines = _ended(world.log)
        assert (result["status"], result["reason"]) == ("stopped", "panic")
        assert not world.calls.exists() and _version_at(scene.app) == INSTALLED
        assert body_adoption.native_hold(scene.data) == "panic"
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_a_panic_while_the_old_app_quits_stops_before_the_switch(scene):
    _target_speaks(scene, 2)
    row = scene.prepare()
    world = _activate(scene, row, quits=False)
    try:
        _until(lambda: world.calls.exists())  # the quit was asked; the old app has not exited
        (scene.data / "state" / "panic_stop.flag").write_text("panic")

        result, _lines = _ended(world.log)

        assert (result["status"], result["reason"]) == ("stopped", "panic")
        assert scene.handoff()["phase"] == "abandoned" and git(scene.serving, "rev-parse", "HEAD") == scene.base
        assert _version_at(scene.app) == INSTALLED and pathlib.Path(row["native"]["staged_app"]).is_dir()
        assert not [line for line in world.calls.read_text().splitlines() if line.startswith("open ")]  # nothing opened
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_a_staged_bundle_without_its_signed_entry_is_never_handed_the_switch(tmp_path):
    staged = make_app(tmp_path / "Apps" / f".Ouroboros-update-{TARGET}" / "Ouroboros.app", version=TARGET, entry=False)
    native_fact = {"status": "staged", "installed": str(tmp_path / "Apps" / "Ouroboros.app"),
                   "staged_app": str(staged), "backup_app": str(tmp_path / "Apps" / "backup.app")}
    assert update_native.installer_command(native_fact, data_dir=tmp_path, adoption_id="x", tools={"open": "x"}) == {
        "ok": False, "reason": "staged_entry_missing"}


# --------------------------------------------------------------------------- #
# The activating server's held exit and the new generation's readback
# --------------------------------------------------------------------------- #
def test_a_held_exit_ends_on_the_launchers_quit_on_panic_or_once_nothing_is_held(monkeypatch, tmp_path):
    import threading

    from ouroboros import server_process
    from ouroboros.config import PANIC_EXIT_CODE, RESTART_EXIT_CODE

    verdicts = iter(["hold", "hold", "recycle"])
    monkeypatch.setattr(body_adoption, "native_hold", lambda _data: next(verdicts))
    assert server_process.hold_for_launcher_stop(tmp_path, poll_sec=0.01) == RESTART_EXIT_CODE
    monkeypatch.setattr(body_adoption, "native_hold", lambda _data: "panic")
    assert server_process.hold_for_launcher_stop(tmp_path, poll_sec=0.01) == PANIC_EXIT_CODE
    monkeypatch.setattr(body_adoption, "native_hold", lambda _data: "hold")
    quit_pipe = threading.Timer(0.2, server_process._exit_signalled.set)  # the launcher's ``quit`` line arrives
    try:
        quit_pipe.start()
        assert server_process.hold_for_launcher_stop(tmp_path, poll_sec=0.01) == 0
    finally:
        quit_pipe.cancel()
        server_process._exit_signalled.clear()


def test_the_readback_accepts_only_the_new_apps_interpreter_entry_runtime_environment_and_roots(tmp_path, monkeypatch):
    installed = make_app(tmp_path / "Applications" / "Ouroboros.app", version=TARGET)
    repo = tmp_path / "repo"
    repo.mkdir()
    python = _app_python(installed)
    runtime = {"version": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version}
    handoff = {"data_dir": str(tmp_path / "data"), "repo_dir": str(repo), "environment": {"state": "selected"},
               "native": {"installed": str(installed), "new_identity": body_switch.bundle_identity(str(installed)),
                          "runtime": runtime}}
    command = f"{installed}/Contents/MacOS/ouroboros-shell {repo}/launcher.py --launch-intent automatic"
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr("ouroboros.platform_layer.process_command", lambda _pid: command)
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", handoff["data_dir"])
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(repo))
    assert body_adoption._native_readback(handoff, repo) == {"ok": True, "failed": [], "launcher": command,
                                                              "executable": str(python)}
    monkeypatch.setattr("ouroboros.platform_layer.process_command",
                        lambda _pid: f"{python} {repo}/launcher.py --launch-intent automatic")  # plain Python
    assert body_adoption._native_readback(handoff, repo)["failed"] == ["launcher_is_not_the_signed_source_entry"]
    monkeypatch.setattr("ouroboros.platform_layer.process_command", lambda _pid: command)
    stale = {**handoff, "environment": {"state": "restored"},  # another bundle, in the current identity form
             "native": {**handoff["native"], "new_identity": [0, 0, 0]}}
    assert body_adoption._native_readback(stale, repo)["failed"] == ["installed_app_is_not_the_new_one",
                                                                     "environment_not_selected"]
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "other"))
    assert body_adoption._native_readback(handoff, repo)["failed"] == ["roots_differ"]


def test_a_bundle_whose_contents_another_installer_replaced_in_place_is_another_bundle(tmp_path):
    """Homebrew with app-management permission keeps an app's folder and moves a new ``Contents`` into it (its
    ``moved.rb``, read): the folder's inode stays, the identity does not. A rename keeps the identity."""
    import shutil

    app = make_app(tmp_path / "Applications" / "Ouroboros.app", version=INSTALLED)
    before = body_switch.bundle_identity(str(app))
    os.rename(app, tmp_path / "Elsewhere.app")
    assert body_switch.bundle_identity(str(tmp_path / "Elsewhere.app")) == before
    os.rename(tmp_path / "Elsewhere.app", app)
    brewed = make_app(tmp_path / "brew-staging" / "Ouroboros.app", version=TARGET)
    shutil.rmtree(app / "Contents")
    os.rename(brewed / "Contents", app / "Contents")
    after = body_switch.bundle_identity(str(app))
    assert after[:2] == before[:2] and after != before


def test_an_app_whose_contents_another_installer_replaced_before_the_swap_is_not_swapped_away(scene):
    """Homebrew with app-management permission upgrades in place: the folder stays, ``Contents`` is new. The
    installer finds it just before the swap, as for a replaced folder, and swaps nothing."""
    _target_speaks(scene, 2)
    row = scene.prepare()
    installed = pathlib.Path(row["native"]["installed"])
    brewed = make_app(scene.tmp / "brew-staging" / "Ouroboros.app", version="7.9.5")
    world = _activate(scene, row, after_quit=f"rm -rf {json.dumps(str(installed / 'Contents'))}\n"
                                             f"mv {json.dumps(str(brewed / 'Contents'))} "
                                             f"{json.dumps(str(installed / 'Contents'))}\n")
    try:
        result, lines = _ended(world.log)

        assert (result["status"], result["reason"]) == ("not_applied", "exchange_failed"), lines
        assert (scene.handoff()["phase"], scene.handoff()["return"]["trigger"]) == ("returned", "app_changed")
        assert git(scene.serving, "rev-parse", "HEAD") == scene.base and _version_at(installed) == "7.9.5"
        assert body_switch.bundle_identity(row["native"]["staged_app"]) == row["native"]["new_identity"]
    finally:
        _stop(world.launcher)
        _stop(world.server)


def test_the_readback_holds_a_carried_runtime_venv_to_the_body_and_the_new_apps_python(tmp_path, monkeypatch):
    """A launch that carried ``--runtime-venv`` proves the body runs in that venv (never silently on the new app's
    own packages) and that the venv's interpreter resolves into the new app; one without it keeps the app's rule."""
    installed = make_app(tmp_path / "Applications" / "Ouroboros.app", version=TARGET)
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    real = _app_python(installed).with_name(f"python{RUNTIME}")  # the app's own binary the venv's link resolves to
    real.write_bytes(b"\0")
    venv, stray = tmp_path / "owner-venv", tmp_path / "stray-venv"
    for root, target in ((venv, _app_python(installed)), (stray, pathlib.Path(sys.executable))):
        (root / "bin").mkdir(parents=True)
        (root / "bin" / "python3").symlink_to(target)
        (root / "bin" / "python").symlink_to("python3")
        (root / runtime_venv.MARKER_NAME).write_text(json.dumps({"schema_version": 1, "python": RUNTIME,
                                                                 "state": "ready"}))
    _app_python(installed).unlink()
    _app_python(installed).symlink_to(real.name)  # python3 -> python3.X inside the app, as the app ships it
    for key, value in (("OUROBOROS_DATA_DIR", str(data)), ("OUROBOROS_REPO_DIR", str(repo)),
                       ("OUROBOROS_MANAGED_BY_LAUNCHER", "1")):
        monkeypatch.setenv(key, value)
    runtime = {"version": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version}
    handoff = {"data_dir": str(data), "repo_dir": str(repo), "environment": {"state": "selected"},
               "native": {"installed": str(installed), "new_identity": body_switch.bundle_identity(str(installed)),
                          "runtime": runtime, "runtime_venv": str(venv)}}
    monkeypatch.setattr("ouroboros.platform_layer.process_command",
                        lambda _pid: f"{installed}/Contents/MacOS/ouroboros-shell {repo}/launcher.py "
                                     f"--launch-intent automatic --runtime-venv {venv}")

    def body(python, environment):
        monkeypatch.setattr(sys, "executable", str(python))
        monkeypatch.setattr(update_prepare, "serving_environment", lambda: environment)
        return body_adoption._native_readback(handoff, repo)["failed"]

    assert body(venv / "bin" / "python", ("venv", venv)) == []
    # The venv is the body's; the shell's provenance and the runtime are held exactly as without one.
    monkeypatch.setattr("ouroboros.platform_layer.process_command",
                        lambda _pid: f"{venv}/bin/python {repo}/launcher.py --runtime-venv {venv}")  # not the entry
    assert body(venv / "bin" / "python", ("venv", venv)) == ["launcher_is_not_the_signed_source_entry"]
    monkeypatch.setattr("ouroboros.platform_layer.process_command",
                        lambda _pid: f"{installed}/Contents/MacOS/ouroboros-shell {repo}/launcher.py "
                                     f"--launch-intent automatic --runtime-venv {venv}")
    handoff["native"]["runtime"] = {**runtime, "version": "3.9.0"}  # the staged app measured another Python
    assert body(venv / "bin" / "python", ("venv", venv)) == ["runtime_differs_from_the_prepared_one"]
    handoff["native"]["runtime"] = runtime
    # The new app ignored the option: its body runs on its own packages, a silent replacement of the owner's venv.
    assert body(_app_python(installed), ("user_site", tmp_path / "userbase")) == ["runtime_venv_not_selected"]
    assert body(stray / "bin" / "python", ("venv", stray)) == ["runtime_venv_not_selected"]  # another venv
    (venv / runtime_venv.MARKER_NAME).unlink()  # a directory the launcher would refuse, never select
    assert body(venv / "bin" / "python", ("venv", venv)) == ["runtime_venv_not_selected"]
    (venv / runtime_venv.MARKER_NAME).write_text(json.dumps({"schema_version": 1, "python": RUNTIME, "state": "ready"}))
    (venv / "bin" / "python3").unlink()
    (venv / "bin" / "python3").symlink_to(sys.executable)  # the owner's venv, but its Python is not the new app's
    assert body(venv / "bin" / "python", ("venv", venv)) == ["interpreter_outside_the_app"]
    del handoff["native"]["runtime_venv"]  # nothing carried: a body in some venv is outside the app, as before
    assert body(venv / "bin" / "python", ("venv", venv)) == ["interpreter_outside_the_app"]


def test_the_readback_holds_a_venv_on_the_owners_own_python_to_exactly_that_python(tmp_path, monkeypatch):
    """A carried venv that runs the owner's own Python (``venv_python``) is held to exactly that Python and the
    runtime prepared for it: the same venv moved onto the new app's Python, or onto any other, is not the launch
    the Update intended. The venv, the signed entry and the runtime are held as for a venv of the app's Python."""
    installed = make_app(tmp_path / "Applications" / "Ouroboros.app", version=TARGET)
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    own, other = (tmp_path / name / "bin" / f"python{RUNTIME}" for name in ("owners-python", "another-python"))
    for python in (own, other):  # installations of the owner's, not the app's
        python.parent.mkdir(parents=True)
        python.write_bytes(b"\0")
    venv = tmp_path / "owner-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python3").symlink_to(own)
    (venv / "bin" / "python").symlink_to("python3")
    (venv / runtime_venv.MARKER_NAME).write_text(json.dumps({"schema_version": 1, "python": RUNTIME, "state": "ready"}))
    for key, value in (("OUROBOROS_DATA_DIR", str(data)), ("OUROBOROS_REPO_DIR", str(repo)),
                       ("OUROBOROS_MANAGED_BY_LAUNCHER", "1")):
        monkeypatch.setenv(key, value)
    runtime = {"version": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version}
    handoff = {"data_dir": str(data), "repo_dir": str(repo), "environment": {"state": "selected"},
               "native": {"installed": str(installed), "new_identity": body_switch.bundle_identity(str(installed)),
                          "runtime": runtime, "runtime_venv": str(venv), "venv_python": os.path.realpath(own)}}
    monkeypatch.setattr("ouroboros.platform_layer.process_command",
                        lambda _pid: f"{installed}/Contents/MacOS/ouroboros-shell {repo}/launcher.py "
                                     f"--launch-intent automatic --runtime-venv {venv}")
    monkeypatch.setattr(sys, "executable", str(venv / "bin" / "python"))
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))

    assert body_adoption._native_readback(handoff, repo)["failed"] == []
    handoff["native"]["runtime"] = {**runtime, "version": "3.9.0"}  # prepared on another runtime than it runs
    assert body_adoption._native_readback(handoff, repo)["failed"] == ["runtime_differs_from_the_prepared_one"]
    handoff["native"]["runtime"] = runtime
    for moved_to in (_app_python(installed), other):  # the new app's Python, or another of the owner's
        (venv / "bin" / "python3").unlink()
        (venv / "bin" / "python3").symlink_to(moved_to)
        assert body_adoption._native_readback(handoff, repo)["failed"] == ["interpreter_is_not_the_venvs_own"]


def test_the_readback_never_accepts_an_app_identity_recorded_before_its_contents(tmp_path, monkeypatch):
    """A transition recorded with the earlier two-field identity proves no bundle: the readback says so, whatever the
    folder at the installed path is, and nothing else it checks changes."""
    installed = make_app(tmp_path / "Applications" / "Ouroboros.app", version=TARGET)
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = {"version": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version}
    native_fact = {"installed": str(installed), "new_identity": body_switch.bundle_identity(str(installed)),
                   "runtime": runtime}
    handoff = {"data_dir": str(tmp_path / "data"), "repo_dir": str(repo), "environment": {"state": "selected"},
               "native": native_fact}
    monkeypatch.setattr(sys, "executable", str(_app_python(installed)))
    monkeypatch.setattr("ouroboros.platform_layer.process_command",
                        lambda _pid: f"{installed}/Contents/MacOS/ouroboros-shell {repo}/launcher.py")
    for key, value in (("OUROBOROS_MANAGED_BY_LAUNCHER", "1"), ("OUROBOROS_DATA_DIR", handoff["data_dir"]),
                       ("OUROBOROS_REPO_DIR", str(repo))):
        monkeypatch.setenv(key, value)
    assert body_adoption._native_readback(handoff, repo)["failed"] == []

    native_fact["new_identity"] = native_fact["new_identity"][:2]  # the same folder, by the earlier identity

    assert body_adoption._native_readback(handoff, repo)["failed"] == ["app_identity_stale"]


def test_the_readback_refuses_a_new_app_that_started_under_another_home_or_profile(tmp_path, monkeypatch):
    """The installer carries the configuration this installation ran under; the new generation proves it got it."""
    installed = make_app(tmp_path / "Applications" / "Ouroboros.app", version=TARGET)
    repo, data, python = tmp_path / "repo", tmp_path / "data", _app_python(installed)
    repo.mkdir()
    for key, value in (("HOME", str(tmp_path / "home")), ("CFFIXED_USER_HOME", str(tmp_path / "profile")),
                       ("OUROBOROS_APP_ROOT", str(tmp_path)), ("OUROBOROS_DATA_DIR", str(data)),
                       ("OUROBOROS_REPO_DIR", str(repo)), ("OUROBOROS_MANAGED_BY_LAUNCHER", "1")):
        monkeypatch.setenv(key, value)
    runtime = {"version": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version}
    handoff = {"data_dir": str(data), "repo_dir": str(repo), "environment": {"state": "selected"},
               "native": {"installed": str(installed), "new_identity": body_switch.bundle_identity(str(installed)),
                          "runtime": runtime, "launch": {"env": update_native.launch_configuration(data, repo)}}}
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr("ouroboros.platform_layer.process_command",
                        lambda _pid: f"{installed}/Contents/MacOS/ouroboros-shell {repo}/launcher.py")
    assert body_adoption._native_readback(handoff, repo)["failed"] == []

    monkeypatch.setenv("HOME", str(tmp_path / "another-home"))  # LaunchServices gave the new app another home
    monkeypatch.delenv("CFFIXED_USER_HOME")

    assert body_adoption._native_readback(handoff, repo)["failed"] == ["configuration_differs:CFFIXED_USER_HOME,HOME"]


def test_the_readback_holds_the_new_app_to_the_port_this_installation_was_launched_for(tmp_path, monkeypatch):
    """The operator's port names the window's origin, whose storage WebKit keeps apart: a new app the launcher
    started on another port is not this installation's window, whatever answers there. A port the server fell
    back to stays the port file's; both sides compare the port the launcher was asked for."""
    installed = make_app(tmp_path / "Applications" / "Ouroboros.app", version=TARGET)
    repo, data, python = tmp_path / "repo", tmp_path / "data", _app_python(installed)
    repo.mkdir()
    for key, value in (("HOME", str(tmp_path / "home")), ("OUROBOROS_APP_ROOT", str(tmp_path)),
                       ("OUROBOROS_DATA_DIR", str(data)), ("OUROBOROS_REPO_DIR", str(repo)),
                       ("OUROBOROS_MANAGED_BY_LAUNCHER", "1"), ("OUROBOROS_SERVER_PORT", "28817")):
        monkeypatch.setenv(key, value)
    runtime = {"version": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version}
    launch = update_native.successor_launch(data, repo)
    assert launch["env"]["OUROBOROS_SERVER_PORT"] == "28817" and "OUROBOROS_SERVER_PORT" not in launch["inherit"]
    handoff = {"data_dir": str(data), "repo_dir": str(repo), "environment": {"state": "selected"},
               "native": {"installed": str(installed), "new_identity": body_switch.bundle_identity(str(installed)),
                          "runtime": runtime, "launch": launch}}
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr("ouroboros.platform_layer.process_command",
                        lambda _pid: f"{installed}/Contents/MacOS/ouroboros-shell {repo}/launcher.py")
    (data / "state").mkdir(parents=True)
    (data / "state" / "server_port").write_text("28818")  # this generation's server fell back: still the request
    assert body_adoption._native_readback(handoff, repo)["failed"] == []

    monkeypatch.setenv("OUROBOROS_SERVER_PORT", "8765")  # the new app was launched without the operator's port

    assert body_adoption._native_readback(handoff, repo)["failed"] == ["configuration_differs:OUROBOROS_SERVER_PORT"]


def test_documented_operator_inputs_ride_by_name_and_no_process_role_does(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    operator = {"OUROBOROS_WORKER_START_METHOD": "spawn", "OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD": "1",
                "OUROBOROS_DISABLE_MANAGED_UPDATES": "1"}
    role = {"OUROBOROS_MANAGED_BY_LAUNCHER": "1", "OUROBOROS_LAUNCH_INTENT": "owner", "OUROBOROS_APP_VERSION": "7.7.0",
            "OUROBOROS_SERVER_PROCESS_PID": "3226", "OUROBOROS_SERVER_PORT": "28817"}
    for key, value in {**operator, **role}.items():
        monkeypatch.setenv(key, value)

    launch = update_native.successor_launch(data, tmp_path / "repo")

    assert set(operator) <= set(launch["inherit"]) and not set(role) & set(launch["inherit"])
    assert launch["env"]["OUROBOROS_SERVER_PORT"] == "28817"  # the port is configuration, carried and read back
    assert "spawn" not in json.dumps(launch)  # an inherited input is a name, never its value


@pytest.mark.parametrize("source, carried", [("environment", True), ("settings", False), (None, False)])
def test_a_bind_host_the_operator_launched_with_rides_into_the_new_app_and_a_stamped_one_never(
        tmp_path, monkeypatch, source, carried):
    """The launcher stamps its server's bind host from settings or keeps the one its own launch environment
    carried (``launcher.start_agent``), and records which (``server_host_source``, read back through
    ``server_process.applied_server_host_source``). The operator's choice rides along; a stamp, or a host whose
    provenance no record states, would mask the settings the new app reads, so it stays behind."""
    from ouroboros import server_process

    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_SERVER_HOST", "0.0.0.0")
    monkeypatch.setattr(server_process, "_applied_server_host_source", "launcher")  # a launcher-managed server
    if source:  # the launcher's record of this very server (``launcher._write_server_process_record``)
        server_py = pathlib.Path(server_process.__file__).resolve().parents[1] / "server.py"
        (data / "state" / "server_process.json").write_text(json.dumps(
            {"pid": os.getpid(), "server_path": str(server_py), "server_host_source": source}))

    launch = update_native.successor_launch(data, tmp_path / "repo")

    assert ("OUROBOROS_SERVER_HOST" in launch["inherit"]) is carried
    assert "0.0.0.0" not in json.dumps(launch)  # a name, never the value
