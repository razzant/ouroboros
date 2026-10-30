"""The unified Update's official macOS package step: need, package, origin, staging, hand-over.

A composition needs a new app when the installed app cannot host its source launcher as the
current launcher requires (``native_package.requirement``): a launcher declaring a source-entry
protocol needs the app's signed source entry (7.7.0 has none; source alone cannot add one to a
sealed app), and a protocol the app's own release tree does not declare needs that release. A
source install, or a composition declaring none, needs nothing native. The need is the
composition's, not the source update's: a checkout already at the target whose app cannot host
it is an app-only Update (``app_need``), with H, I and W unchanged and no commit made.

When an app is needed: the package is the newest published official release whose own tree
declares that protocol AND whose release actually carries the macOS image (its checksum capsule
names it: ``native_package.select_package``), a version separate from the source's. The app's
origin is then proven, not claimed: its deep signature, bundle id and published manifest tag
locally, then the official disk image of ITS OWN release, checked against that release's
capsule, must carry a payload with the same code-directory hash (which seals the bundle's
resources and nested code); that payload's team is the official team. Notarization is no part of
that provenance (official 7.7.0 is a Developer ID build Gatekeeper reports unnotarized). Only then
is the package downloaded, checked (capsule SHA-256, read-only attach, signature, notarization,
that team, bundle id, version, the signed source entry, the Python runtime this installation's
packages were prepared for), staged beside the app on its volume, and its interpreter measured;
the volume must swap two directories in one call. The new app runs the body on the packages at
the path this generation runs them from: the packaged user site, or the ``--runtime-venv`` the
launcher selected (``selected_runtime_venv``), which the new app is opened with
(``successor_launch``). The app is the one the launcher runs from (``installed_bundle``),
whichever Python the venv runs, and the generation's packages are a clone beside the venv. A
venv of the installed app's Python has the clone pointed at the staged app's own Python while
its lock installs and its checks run, then back at the installed path, where the exchange puts
those bytes (``update_prepare``, ``runtime_venv.point_base``); the switch-time import probe,
before the exchange, runs its packages on the staged Python as well (``body_switch``). A venv of
another Python of the same version, which a launcher accepts too (the marker records a version,
not where that Python lives), keeps it: the clone installs and is checked on it, the probe runs
it and the readback holds the body to it (``venv_python``). Any other environment is a manual step.

A Homebrew Cask receipt for the app (``native_package.package_manager_receipts``) is recorded and
decides nothing: the official image a cask installed takes this same Update, and the receipt is
never written, so Homebrew keeps listing the version it installed (``manager_version``) beside the
app's own. Nothing excludes Homebrew meanwhile. The bundles are told apart by identity (the
folder's device and inode and its ``Contents`` inode, ``body_switch.bundle_identity``), which a
``brew upgrade`` or reinstall changes whether it replaces the folder or, with app-management
permission, the contents in place (its ``moved.rb``, read, not run): authorization refuses it,
the installer finds it just before its swap, and one between that check and the swap or after
it only the new generation's readback finds (not adopted). Once adopted, the bundle is
Homebrew's to replace like any app.

Activation starts ``hand_over`` before it requests its restart: the staged app's signed entry
runs the installer captured beside the switch helper (``ouroboros/app_install.py``), detached
from the activating server's process tree and in process custody; it acts only once that
restart armed the switch and switches source and environment itself, before the new app opens.
Retirement keeps the candidate while its installer lives. Nothing here is natively qualified
(UPDATE-TECHNICAL-PLAN §11).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import select
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_SEC = 1800
_HAND_OVER_SEC = 30  # the detaching start reports its child at once


def runner(argv) -> subprocess.CompletedProcess:
    """One native tool (``hdiutil``, ``codesign``, ...) through the shell's tracked process owner: refused once this
    server's stop or Emergency Stop closed command admission (``update_prepare.PrepareRefused``, typed), ended by the
    generation's stop while it runs, its tree killed at the deadline."""
    from ouroboros.tools.shell_process import CommandAdmissionClosed, _tracked_subprocess_run

    try:
        return _tracked_subprocess_run([str(part) for part in argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, timeout=600)
    except CommandAdmissionClosed:
        from supervisor.update_prepare import stopped

        stopped()
        raise


def installed_bundle() -> Optional[pathlib.Path]:
    """The Ouroboros macOS ``.app`` this generation's launcher runs from, or ``None`` where none is found (a
    source or container install, or a packaged Windows or Linux build: only a macOS app is ever replaced).

    The app is the launcher's fact, not the body's: a launcher stamps the bundle it runs from into
    every server it starts (``OUROBOROS_BUNDLE_DIR``: the app's ``Contents/Frameworks``, frozen or
    handed that seed), and a ``--runtime-venv`` may run the body on a Python outside the app. A
    server no launcher started is read by the bundle its interpreter resolves into. Only a bundle
    carrying the seed manifest every Ouroboros app build ships counts: a source install's
    interpreter may itself live inside another bundle (Homebrew's ``Python.app``).
    """
    from ouroboros.launcher_bootstrap import BUNDLE_MANIFEST_NAME
    from ouroboros.platform_layer import BUNDLE_DIR_ENV

    stamped = os.environ.get(BUNDLE_DIR_ENV, "") if launcher_pid() else ""
    for parent in pathlib.Path(stamped or sys.executable).resolve().parents:
        if parent.suffix == ".app" and (parent / "Contents" / "Info.plist").is_file():
            seeded = any((parent / "Contents" / folder / BUNDLE_MANIFEST_NAME).is_file()
                         for folder in ("Frameworks", "Resources"))
            return parent if seeded else None
    return None


def _download(url: str, target: pathlib.Path, limit: int) -> int:
    """Stream ``url`` into ``target`` (at most ``limit`` bytes); returns the bytes written."""
    written = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as out:  # noqa: S310 - fixed https
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            written += len(chunk)
            if written > limit:
                raise ValueError(f"{url} is larger than {limit} bytes")
            out.write(chunk)
    return written


def _capsule_reader(fetch: Callable, folder: pathlib.Path) -> Callable[[str], Optional[str]]:
    """``version -> SHA256SUMS text`` of that official release, ``None`` when the release publishes none."""
    from ouroboros import native_package
    from ouroboros.tools import release_sync

    def read(version: str) -> Optional[str]:
        url = release_sync.release_asset_download_url(native_package.ASSET, version).rsplit("/", 1)[0] + "/SHA256SUMS"
        target = folder / f"SHA256SUMS-{version}"
        try:
            fetch(url, target, 1 << 20)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None  # a tag whose release carries no assets (yet)
            raise
        return target.read_text(encoding="utf-8")

    return read


def _manual(base: Dict[str, Any], reason: str, instruction: str = "", **facts: Any) -> Dict[str, Any]:
    return {**base, **facts, "status": "manual", "reason": reason, "instruction": instruction}


def _need(repo: Any, files_tree: str, serving_files: str, bundle: pathlib.Path, run: Callable, receipts: list,
          tags: list) -> tuple:
    from ouroboros import native_package

    target = native_package.protocol_at(repo, files_tree)
    local = native_package.installed_app(bundle, runner=run, published_tags=tags, owners=receipts)
    hosted = native_package.protocol_at(repo, local["release_tag"]) if local.get("release_tag") else None
    need = native_package.requirement(local, target, serving=native_package.protocol_at(repo, serving_files),
                                      hosted=hosted)
    return target, local, need


def app_need(repo: Any, files_tree: str, *, bundle: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Whether the installed app can host ``files_tree`` (the serving composition, for an app-only Update).

    The bundle's own files and Git only (no signature check, no download): the serving
    protocol is the composition's own, so an app with the signed entry that runs it needs
    nothing, and one without it needs the release that has it. Preparation reads everything else.
    """
    from ouroboros import native_package
    from ouroboros.tools import release_sync

    bundle = bundle or installed_bundle()
    if bundle is None:
        return {"status": "none", "reason": "no_app_bundle"}
    target = native_package.protocol_at(repo, files_tree)
    tag = release_sync.normalize_release_tag(str((native_package._manifest(bundle) or {}).get("release_tag") or ""))
    need = native_package.requirement({"entry": native_package._entry(bundle)}, target, serving=target,
                                      hosted=native_package.protocol_at(repo, tag) if tag else None)
    # The app's own version as its bundle says (unread stays absent): the panel names which app this is.
    version = str((native_package._plist(bundle) or {}).get("CFBundleShortVersionString") or "")
    return {**need, **({"installed_version": version} if version else {})}


def prepare(row: Dict[str, Any], *, run: Callable = runner, fetch: Callable = _download,
            bundle: Optional[pathlib.Path] = None, published=None, caskrooms=None,
            appdirs=None) -> Dict[str, Any]:
    """The candidate's native fact: ``none``, ``staged`` (ready to hand over) or ``manual`` with the instruction."""
    from ouroboros import native_package

    repo, new_files = row["repo_dir"], row["layers"]["new_files"]
    if native_package.protocol_at(repo, new_files) is None:
        return {"status": "none", "reason": "nothing_declared"}
    bundle = bundle or installed_bundle()
    if bundle is None:
        return {"status": "none", "reason": "no_app_bundle"}  # no macOS app here: no native package step
    receipts = native_package.package_manager_receipts(bundle, **({"caskrooms": caskrooms} if caskrooms else {}),
                                                       **({"appdirs": appdirs} if appdirs else {}))
    if published is None:
        from supervisor.git_ops import list_official_update_tags

        published = list_official_update_tags()
    tags = [item.get("tag") for item in published if isinstance(item, dict)]
    target, local, need = _need(repo, new_files, row["layers"]["old_files"], bundle, run, receipts, tags)
    if need["status"] == "none":
        return {"status": "none", "reason": need["reason"]}
    base = {"need": need, "installed": local.get("path"), "source_version": _tree_version(repo, new_files),
            "previous_version": local.get("version") or "",
            # Who installed the app and what it last installed: the bundle's own version is ``previous_version``.
            **({"manager": local["manager"], "manager_version": local.get("manager_version") or ""}
               if local.get("manager") else {})}
    if need["status"] != "needed":
        return _manual(base, need["reason"])
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-release-capsules-"))
    try:
        package = native_package.select_package(repo, target, published, _capsule_reader(fetch, scratch))
    except (OSError, ValueError) as exc:
        package = {"unread": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if package is None:
        return _manual(base, "no_published_package", f"No published Ouroboros app hosts source-entry protocol {target} "
                       "yet; this update needs it. Install the release that does once it is published.")
    if package.get("unread"):
        return _manual(base, f"release_assets_unread: {package['unread']}", "The published Ouroboros releases could "
                       "not be read; check the connection and prepare the update again.")
    version = package["version"]
    base.update(version=version, package_tag=package["tag"])
    if local["origin"] == "custom":
        return _manual(base, "origin_custom", native_package.manual_instruction(version, local), origin="custom")
    from ouroboros.runtime_venv import own_python
    from supervisor.update_prepare import serving_environment

    kind, venv = serving_environment()[0], selected_runtime_venv()
    if kind != "user_site" and not venv:  # the new app starts the body on its user site or on a --runtime-venv
        return _manual(base, f"environment_{kind or 'system'}", native_package.manual_instruction(version, local))
    own = own_python(venv, bundle) if venv else ""  # a venv of another Python than the app's keeps that Python
    return _stage(row, {**base, **({"runtime_venv": venv} if venv else {}), **({"venv_python": own} if own else {})},
                  local, package, bundle, receipts, tags, run, fetch)


def selected_runtime_venv() -> str:
    """The ``--runtime-venv`` this generation's body runs in, ``""`` when it runs on the app's own packages.

    The packages' one path (``update_prepare.serving_environment``) is then a venv the launcher's
    ``ouroboros.runtime_venv`` recorded, the one a relaunch with ``--runtime-venv`` at that path
    selects again. Named as this interpreter names it, which is how its launcher named it.
    """
    from ouroboros.runtime_venv import recorded_state
    from supervisor.update_prepare import serving_environment

    kind, root = serving_environment()
    return str(root) if kind == "venv" and recorded_state(root) else ""


def _tree_version(repo: Any, tree: str) -> str:
    from supervisor.update_prepare import _run  # the tracked Git owner: a refused start raises, never reads as ""

    shown = _run(pathlib.Path(repo), "cat-file", "blob", f"{tree}:VERSION", check=False)
    return shown.stdout.decode("utf-8", "replace").strip() if shown.returncode == 0 else ""


def _release_image(version: str, folder: pathlib.Path, fetch: Callable, sums: str = "") -> Dict[str, Any]:
    """The official release's disk image (and checksum capsule, unless already read) for ``version``."""
    from ouroboros import native_package
    from ouroboros.tools import release_sync

    folder.mkdir(parents=True, exist_ok=True)
    url = release_sync.release_asset_download_url(native_package.ASSET, version)
    if not sums:
        fetch(url.rsplit("/", 1)[0] + "/SHA256SUMS", folder / "SHA256SUMS", 1 << 20)
        sums = (folder / "SHA256SUMS").read_text(encoding="utf-8")
    dmg = folder / release_sync.release_asset_name(native_package.ASSET, version)
    fetch(url, dmg, 4 << 30)
    return {"dmg": dmg, "sha256sums": sums}


def staged_interpreter(app: Any) -> Dict[str, Any]:
    """The staged app's own Python (what runs the new tree) and its measured runtime, as it will be installed.

    The runtime is measured by running that interpreter; the executable is named at the
    installed path, where the exchange puts these same bytes.
    """
    from ouroboros.platform_layer import embedded_python_candidates

    frameworks = pathlib.Path(str(app)) / "Contents" / "Frameworks"
    python = next((path for path in embedded_python_candidates(frameworks) if path.is_file()), None)
    if python is None:
        return {"error": f"{app} carries no embedded Python"}
    try:
        shown = runner([str(python), "-I", "-c", "import json, sqlite3, sys; print(json.dumps({'version': "
                        "sys.version.split()[0], 'sqlite': sqlite3.sqlite_version}))"])
        measured = json.loads(shown.stdout) if shown.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, ValueError):
        measured = {}
    if not measured.get("version"):
        return {"error": f"{python} did not run"}
    return {"python": str(python), "runtime": measured}


def _stage(row, base, local, package, bundle, receipts, tags, run, fetch) -> Dict[str, Any]:
    """Prove the installed app's origin against its own official release, then stage the package beside it."""
    from ouroboros import app_install, body_switch, native_package

    version, installed_version = package["version"], str(local.get("version") or "")
    owner = str(row["candidate_id"])  # each candidate stages in its own folder, retired with it alone
    staging = native_package.staging_path(bundle, version, owner)
    base = {**base, "staging_dir": str(staging)}  # whatever happens next, retirement removes this folder
    if local["origin"] == "unknown" and local["checks"].get("team_id") is None and all(
            value is True for name, value in local["checks"].items() if name != "team_id"):
        try:
            own = _release_image(installed_version, staging / "installed-release", fetch)
        except (OSError, ValueError) as exc:
            return _manual(base, f"download_failed: {exc}", native_package.manual_instruction(version, local))
        asset = native_package.release_sync.release_asset_name(native_package.ASSET, installed_version)
        digest, why = native_package._capsule_digest(own["sha256sums"], asset)
        # Provenance only: that release's payload is never launched, so its notarization is no part of it.
        proof = native_package.verify(own["dmg"], digest or "", runner=run, version=installed_version,
                                      notarized=False)
        shutil.rmtree(staging / "installed-release", ignore_errors=True)  # the proof is read; its image goes
        if proof["status"] != "verified" or not proof["cdhash"] or proof["cdhash"] != local.get("cdhash"):
            local = {**local, "origin": "custom" if proof["status"] == "verified" else "unknown"}
            return _manual(base, f"origin_{local['origin']}", native_package.manual_instruction(version, local),
                           origin=local["origin"], detail=proof.get("detail") or why)
        local = native_package.installed_app(bundle, runner=run, official_team_id=proof["team"], published_tags=tags,
                                             owners=receipts)
    if local["origin"] != "official":
        return _manual(base, f"origin_{local['origin']}", native_package.manual_instruction(version, local),
                       origin=local["origin"])
    try:
        image = _release_image(version, staging, fetch, sums=package.get("sha256sums") or "")
    except (OSError, ValueError) as exc:
        return _manual(base, f"download_failed: {exc}", native_package.manual_instruction(version, local))
    planned = native_package.plan(version, local, release_metadata={"sha256sums": image["sha256sums"]}, owner=owner)
    if planned["status"] != "ready":
        return _manual(base, planned["reason"], planned["instruction"] or "", detail=planned["detail"])
    body = body_switch.runtime_identity()  # the Python this body and its packages run on (a venv's, in one)
    checked = native_package.verify(planned["dmg_path"], planned["sha256"], runner=run, version=version,
                                    official_team_id=str(local.get("team") or ""), stage_to=planned["staged_app"],
                                    runtime=".".join(body["version"].split(".")[:2]))
    if checked["status"] != "verified":
        return _manual(base, checked["reason"], planned["instruction"] or "", detail=checked["detail"])
    interpreter = staged_interpreter(planned["staged_app"])
    swaps = app_install.exchange_supported(planned["staging_dir"])
    if interpreter.get("error") or swaps:
        return _manual(base, "staged_runtime_unread" if interpreter.get("error") else "exchange_unsupported",
                       planned["instruction"] or "", detail=interpreter.get("error") or swaps)
    return {**base, "status": "staged", "origin": "official", "staged_app": planned["staged_app"],
            "backup_app": planned["backup_app"], "staging_dir": planned["staging_dir"], "sha256": planned["sha256"],
            "team": local.get("team"), "instruction": planned["instruction"], **interpreter,
            # The runtime the new generation's body runs on: the staged app's, or the venv's own Python it keeps.
            **({"runtime": {key: body[key] for key in ("version", "sqlite")}} if base.get("venv_python") else {}),
            "previous_identity": body_switch.bundle_identity(str(bundle)),
            "new_identity": body_switch.bundle_identity(planned["staged_app"])}


def installer_command(native: Dict[str, Any], *, data_dir: Any, adoption_id: str,
                      tools: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The captured installer under the staged app's signed entry: its argv, or why there is none.

    Nothing starts here. The entry is the launcher's own rule (``launcher_bootstrap.macos_source_entry``)
    applied to the staged bundle: the signed Python host the new app runs source with. ``-I`` keeps
    the installer to the standard library. Every other fact it acts on is in the activation's
    record. ``tools`` names other ``open``/``osascript`` executables (the default is the system's).
    """
    from ouroboros.body_adoption import helper_dir
    from ouroboros.launcher_bootstrap import macos_source_entry

    if sys.platform != "darwin" and not tools:
        return {"ok": False, "reason": "platform"}
    if not native or native.get("status") != "staged":
        return {"ok": False, "reason": "not_staged"}
    entry = macos_source_entry(pathlib.Path(native["staged_app"]) / "Contents" / "Frameworks")
    installer = helper_dir(data_dir) / "app_install.py"
    if entry is None or not installer.is_file():
        return {"ok": False, "reason": "staged_entry_missing" if entry is None else "installer_missing"}
    # The step log stays in the staging folder beside the app (retired with the candidate), not the data root.
    record = pathlib.Path(native["staged_app"]).parent / "install.jsonl"
    argv = [str(entry), "-I", str(installer), "--data-dir", str(data_dir), "--adoption-id", str(adoption_id),
            "--log", str(record)]
    for flag, path in (tools or {}).items():
        argv += [f"--{flag}", str(path)]
    return {"ok": True, "reason": "", "argv": argv, "log": str(record), "entry": str(entry)}


def hand_over(command: Dict[str, Any], *, data_dir: Any, adoption_id: str) -> Dict[str, Any]:
    """Start the installer ``installer_command`` built, detached, in custody, and name it in its transition.

    The start runs the entry with ``--detach``: it forks the installer away and reports its pid,
    so the installer is no descendant of this server (a launcher's tree cleanup of it never
    reaches the installer) and leads its own session. Custody is the existing ledger's: the
    start itself through ``process_custody.spawn_supervised``, the installer it reported through
    ``record_process`` (``daemon`` scope, ``INSTALLER_PURPOSE``), which the generation stop and the
    reaper keep. Panic still ends it: it carries the installation's scope marker (Panic's census)
    and reads Panic itself before each step and while it waits. Activation calls this before it
    requests its restart; a start that cannot be confirmed is ended here and reported.
    """
    from ouroboros.app_install import INSTALLER_PURPOSE
    from ouroboros.body_adoption import bind_installer
    from ouroboros.platform_layer import force_kill_pid
    from ouroboros.process_custody import record_process, spawn_supervised

    record = pathlib.Path(command["log"])
    pid = 0
    try:
        with open(record, "a", encoding="utf-8") as output:
            start = spawn_supervised([*command["argv"], "--detach"], drive_root=pathlib.Path(data_dir),
                                     purpose=INSTALLER_PURPOSE + "_start", scope="daemon", stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=output, close_fds=True, cwd=str(record.parent))
        try:
            ready, _w, _x = select.select([start.stdout], [], [], _HAND_OVER_SEC)
            line = start.stdout.readline() if ready else b""
            pid = int(line.strip() or 0)
            start.wait(timeout=_HAND_OVER_SEC)
        finally:
            start.stdout.close()
        if pid <= 0:
            raise RuntimeError("the installer start reported no process")
        record_process(pathlib.Path(data_dir), pid=pid, cmd=command["argv"], purpose=INSTALLER_PURPOSE, scope="daemon")
        if not bind_installer(data_dir, adoption_id, pid):
            raise RuntimeError("the transition changed before the installer could be named in it")
    except Exception as exc:  # a refused or failed start must reach activation, which abandons the switch
        if pid > 0:
            force_kill_pid(pid)  # unnamed, it would never act: end it rather than leave it waiting
        return {"ok": False, "reason": f"installer_not_started: {type(exc).__name__}: {exc}"}
    return {"ok": True, "reason": "", "entry": command["entry"], "pid": pid}


def installer_pending(data_dir: Any) -> list:
    """The installer rows custody still holds alive or unconfirmed (an unreadable ledger is one too)."""
    from ouroboros.app_install import INSTALLER_PURPOSE
    from ouroboros.process_custody import pending_process_stops

    return pending_process_stops(pathlib.Path(data_dir), {INSTALLER_PURPOSE})


def launcher_pid() -> int:
    """The launcher that owns this server: the one the installer waits for (0 when not launcher-managed)."""
    return os.getppid() if os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") == "1" else 0


def server_pid() -> int:
    """The activating server: it holds its exit for the old app's Quit, and the installer waits for it."""
    return os.getpid()


# The new app's launch (``app_install``). ``open`` hands the app it starts its own environment, to which ``--env``
# only adds, and the installer's is the activating server's: this installation's launch inputs plus what the
# old generation's launcher, server, custody and frozen bootloader put there. The new app starts with the inputs
# alone and the configuration this generation resolved, as a launch of this installation does.
_ROLE_PREFIXES = ("OUROBOROS_", "OURO_", "CLAUDEXOR_RUN_SCOPE_", "PYTHON", "__PYVENV_LAUNCHER__", "VIRTUAL_ENV")
# A settings key left once this tree's projections are gone is an operator's input. The bind host may instead be
# the launcher's stamp of the saved one (``launcher.start_agent``); the launcher records which it is, and only the
# host its own launch environment carried rides along (``server_process.applied_server_host_source``).
_BIND_HOST = "OUROBOROS_SERVER_HOST"
# The documented environment-only inputs that configure this installation and are no settings key (DEPLOYMENT.md,
# ARCHITECTURE §7 env-only rows): an operator's choice, no process role, so they ride along by name. The port is
# the launch configuration's own (``launch_configuration``).
_OPERATOR_INPUTS = ("OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD", "OUROBOROS_DISABLE_MANAGED_UPDATES",
                    "OUROBOROS_WORKER_START_METHOD")
_CARRIED_VALUES = frozenset({"OUROBOROS_SERVER_PORT"})  # compared as written; every other carried value is a path


def launch_configuration(data_dir: Any, repo_dir: Any) -> Dict[str, str]:
    """What this generation runs under: the roots, settings, single-instance lock and port file ``config``
    resolved, the server port it was launched for, the home, and CoreFoundation's profile home when one is set
    (WebKit and preferences follow it, and the port names the window's origin)."""
    from ouroboros import config

    data = pathlib.Path(str(data_dir))
    found = {"OUROBOROS_APP_ROOT": config.resolve_app_root(), "OUROBOROS_REPO_DIR": repo_dir,
             "OUROBOROS_DATA_DIR": data, "OUROBOROS_SETTINGS_PATH": config.SETTINGS_PATH,
             "OUROBOROS_PID_FILE": config.PID_FILE,
             "OUROBOROS_PORT_FILE": config.resolve_port_file(data, hosting_data_dir=data),
             # The port the launcher was asked for and stamped into this server (``launcher._requested_server_port``,
             # ``start_agent``): the operator's choice. A port the server fell back to is the port file's, never this.
             "OUROBOROS_SERVER_PORT": os.environ.get("OUROBOROS_SERVER_PORT"),
             "HOME": os.path.expanduser("~"), "CFFIXED_USER_HOME": os.environ.get("CFFIXED_USER_HOME")}
    return {key: str(value) for key, value in found.items() if value}


def configuration_differs(carried: Dict[str, Any], now: Dict[str, str]) -> List[str]:
    """The carried launch configuration a generation does not run under: paths resolved, the port as written."""
    def same(key: str, value: Any) -> bool:
        if key in _CARRIED_VALUES:
            return now[key] == str(value)
        return os.path.realpath(now[key]) == os.path.realpath(str(value))

    return sorted(key for key, value in carried.items() if key not in now or not same(key, value))


def successor_launch(data_dir: Any, repo_dir: Any) -> Dict[str, Any]:
    """The new app's launch: its configuration (``env``, also ``open --env``), the names of the inherited inputs
    it keeps (``inherit``: the values stay in the environment, never in the record) and its arguments: the
    automatic intent and the ``--runtime-venv`` this generation's body runs in (``selected_runtime_venv``),
    an argument no environment carries. A staged app declares a source-entry protocol, and protocol 1's
    signed entry hands that option on to the body (7.7.1 onward); a successor that runs without it fails
    its readback (``body_adoption._native_readback``).

    Dropped: this tree's settings projections (``settings_integrity.skill_environ``; an operator's own settings
    variable stays), the old launch's descriptors and PyInstaller state (``test_environment``), and the process
    roles: every other ``OUROBOROS_`` key (but the documented operator inputs), the scope and containment markers,
    interpreter and import overrides.
    """
    from ouroboros.server_process import applied_server_host_source
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS
    from ouroboros.settings_integrity import skill_environ
    from ouroboros.test_environment import without_launch_descriptors

    settings = set(SETTINGS_DEFAULTS)
    if applied_server_host_source(pathlib.Path(str(data_dir))) != "environment":
        settings.discard(_BIND_HOST)  # stamped from settings, or of unknown provenance: the saved host decides
    inherited = without_launch_descriptors(skill_environ())
    venv = selected_runtime_venv()
    return {"env": launch_configuration(data_dir, repo_dir),
            "inherit": sorted(key for key in inherited if key in settings or key in _OPERATOR_INPUTS
                              or not key.startswith(_ROLE_PREFIXES)),
            "args": ["--launch-intent", "automatic", *(["--runtime-venv", venv] if venv else [])]}


def release_staging(native: Dict[str, Any]) -> bool:
    """A retired candidate's package staging beside the app (its image, an unused staged copy); True once gone.

    Only the folder preparation named ``.<app>-update-<version>-<candidate>`` beside the installed app; the
    installed app and a backup beside it are never touched. A previous app the installer left at
    the staged path (it stopped between its exchange and the backup) becomes that backup first,
    also when the record names it by an identity from before ``Contents`` was part of it
    (``body_switch.same_bundle``); when it cannot, the folder stays. The GC retires a staged
    candidate only once no installer is alive (``update_prepare.retention_verdict``), as for its
    resolver.
    """
    from ouroboros import body_switch

    staging, installed = str((native or {}).get("staging_dir") or ""), str((native or {}).get("installed") or "")
    if not staging or not installed:
        return True
    path, app = pathlib.Path(staging), pathlib.Path(installed)
    if not (path.parent == app.parent and path.name.startswith(f".{app.stem}-update-") and path.is_dir()
            and not path.is_symlink()):
        return not path.exists()
    previous = native.get("previous_identity")
    staged = str(native.get("staged_app") or "")
    if previous and staged and body_switch.same_bundle(staged, previous):
        try:
            moved = body_switch.native_settle_backup({"staged": staged, "backup": native.get("backup_app") or "",
                                                      "previous_identity": previous})
        except OSError:
            moved = False
        if not moved:
            return False
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def retire_backup(backup: str, installed: str) -> bool:
    """The app backup an earlier Update left, once a newer adoption made its own; True once gone.

    Only a plain ``.<app>-<version>-backup<ext>`` folder beside the installed app (as ``native_package.plan``
    names it) goes; the installed app, a link or any other entry is never touched, and stays reported as kept.
    """
    path, app = pathlib.Path(str(backup or "")), pathlib.Path(str(installed or ""))
    if not (backup and installed and path.parent == app.parent and path.name != app.name
            and path.name.startswith(f".{app.stem}-") and path.name.endswith(f"-backup{app.suffix}")
            and path.is_dir() and not path.is_symlink()):
        return not os.path.lexists(path) if backup else True
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()
