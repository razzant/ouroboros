"""Official macOS package transition: native need, package, origin, verification and staging.

A published 7.7/7.8 frozen app cannot run new frozen code, and its ``--host-update`` hook
needs ``--no-ui`` and never replaces a desktop app: a target that needs a host the installed
app lacks needs a new official package, app and source changing in ONE operation
(UPDATE-TECHNICAL-PLAN §6.8). This module owns only the facts of that app step. It downloads
nothing and keeps no state file; it refuses only the app step, never a source-only update.
OS commands go through the caller's ``runner(argv)`` (``returncode``/``stdout``/``stderr``);
one that raises leaves its fact unread.

NEED. The one contract between a frozen app and the source it runs is the source-entry
protocol: an app with the signed ``Contents/MacOS/ouroboros-shell`` entry hands off only to
a source ``launcher.py`` declaring the ``MACOS_SOURCE_ENTRY_PROTOCOL`` it accepts
(``launcher_bootstrap.source_launcher_importable``); an app without the entry hands off
with plain Python and checks nothing. ``protocol_at`` reads that constant from a tree (its
sole owner, by AST, never importing). A target that declares a protocol needs the entry, the
current launcher's host rule (``macos_source_entry``): an app without it (7.7.0) needs the
newest published release that has it, whatever the protocols say. With the entry, a new app
is needed only when the target's protocol differs from the serving source's (which this app
already runs) AND from the app's own release tree's. No feature or version registry exists.

PACKAGE. Its identity is separate from the source's: ``select_package`` takes the newest
published official tag whose own tree declares the target's protocol and whose release
checksum capsule names the macOS image, never the target tree's VERSION and never a tag
alone. ORIGIN. ``official`` only on every fact at once (deep signature, bundle id, notarized
Gatekeeper acceptance, a seed manifest naming a published tag that matches the app, and the
Team ID of the official payload the caller proved identical to this app by its code-directory
hash). A package-manager receipt for the path (``package_manager_receipts``) names who installed
the app and which version it last installed, never what the app is: the official image a Homebrew
Cask installed is official and takes the same Update, and the receipt stays exactly as Homebrew
wrote it (``manual_instruction`` names that manager for an app that is not proven official).
The code-directory hash is the identity because
the code directory seals the bundle's Info.plist and resource envelope, whose entries carry
every resource's hash and each nested executable's own code-directory hash, and ``--deep
--strict`` verification proves the bundle on disk is what that seal names; a same-signer build
with the same manifest and other bytes has another hash. Staging verifies the image against the
release checksum capsule, the payload's signature, notarization, team, bundle id, version,
signed entry and Python runtime, measures it, and copies it beside the app. The exchange itself
is the installer captured beside the switch helper (``ouroboros/app_install.py``), run by the
staged app's signed entry (``supervisor/update_native.hand_over``). Nothing here is natively
qualified (plan §11).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import pathlib
import plistlib
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ouroboros.launcher_bootstrap import BUNDLE_MANIFEST_NAME, macos_source_entry
from ouroboros.tools import release_sync

BUNDLE_ID = "com.ouroboros.agent"  # Ouroboros.spec bundle_identifier; build.sh signs the entry with it too
ASSET = "macos-arm64"  # the DMG's key in release_sync.RELEASE_ASSET_TEMPLATES
PAYLOAD = "Ouroboros.app"  # build.sh stages the disk image with exactly this bundle name
ORIGINS = ("official", "custom", "unknown")
CODESIGN, SPCTL, HDIUTIL, DITTO, OPEN = (
    "/usr/bin/codesign", "/usr/sbin/spctl", "/usr/bin/hdiutil", "/usr/bin/ditto", "/usr/bin/open")
# An installed app is official on provenance alone, all True: notarization is no part of it (the official 7.7.0
# app is a Developer ID build Gatekeeper reports as not notarized) and is read separately (``notarized``).
CHECKS = ("signature", "team_id", "bundle_id", "release")
_NOTARIZED = "Notarized Developer ID"
_SUMS_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNREAD = (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError)
_ORIGIN_REFUSAL = {"custom": "custom_origin"}  # else unknown_origin
PROTOCOL_OWNER, PROTOCOL_NAME = "launcher.py", "MACOS_SOURCE_ENTRY_PROTOCOL"  # the one declaration of it
CASKROOMS = ("/opt/homebrew/Caskroom", "/usr/local/Caskroom")  # Homebrew's install receipts (both prefixes)


def protocol_at(repo: Any, rev: str) -> Optional[int]:
    """The source-entry protocol ``rev``'s ``launcher.py`` declares (a module-level int literal), else ``None``."""
    from supervisor.update_prepare import _run  # the Update's tracked Git owner: a refused start raises

    try:
        shown = _run(repo, "cat-file", "blob", f"{rev}:{PROTOCOL_OWNER}", check=False)
        if shown.returncode != 0:
            return None
        tree = ast.parse(shown.stdout.decode("utf-8"))
    except (OSError, subprocess.SubprocessError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == PROTOCOL_NAME
                                                for t in node.targets):
            value = node.value
            return value.value if isinstance(value, ast.Constant) and _is_int(value.value) else None
    return None


def package_manager_receipts(app: Any, caskrooms: Iterable[Any] = CASKROOMS,
                             appdirs: Optional[Iterable[Any]] = None) -> list:
    """Homebrew Cask receipts naming this bundle: who installed it, read only, never written.

    Each installed cask keeps its definition under ``<Caskroom>/<token>/.metadata/<version>/
    <timestamp>/Casks/<token>.json`` (or ``.rb``); a cask whose ``app`` artifact is this
    bundle's name installed it when the bundle sits in a folder Homebrew installs apps into
    (``/Applications`` or ``~/Applications``). ``version`` is the one that receipt records
    Homebrew installed, newest timestamp first as Homebrew itself reads it: an in-app Update
    moves the bundle past it and leaves the receipt as it is. A receipt decides no origin
    (``installed_app``); a future manager adds a reader here.
    """
    bundle = pathlib.Path(os.path.abspath(str(app)))
    folders = [pathlib.Path("/Applications"), pathlib.Path.home() / "Applications"] if appdirs is None else [
        pathlib.Path(os.path.abspath(str(folder))) for folder in appdirs]
    if bundle.parent not in folders:
        return []
    receipts = []
    for root in caskrooms:
        found = pathlib.Path(root).glob("*/.metadata/*/*/Casks/*")
        for definition in sorted(found, key=lambda path: (path.parents[1].name, path.parents[2].name, str(path)),
                                 reverse=True):
            try:
                text = definition.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if definition.suffix == ".json":
                try:
                    artifacts = json.loads(text).get("artifacts") or []
                except (ValueError, AttributeError):
                    continue
                apps = [name for entry in artifacts if isinstance(entry, dict)
                        for name in (entry.get("app") or []) if isinstance(name, str)]
            else:
                apps = re.findall(r'^\s*app\s+"([^"]+)"', text, re.MULTILINE)
            if bundle.name in {pathlib.Path(name).name for name in apps}:
                token = definition.stem
                receipts.append({"manager": "Homebrew Cask", "path": str(bundle), "version": definition.parents[2].name,
                                 "upgrade": f"brew upgrade --cask {token}", "receipt": str(definition)})
    return receipts


def _run(runner: Callable, argv: Sequence[Any]) -> tuple:
    """``(returncode, stdout + stderr)``, or ``(None, error)`` when the command could not be read."""
    try:
        done = runner([str(part) for part in argv])
        streams = (s.decode("utf-8", "replace") if isinstance(s, bytes) else str(s or "")
                   for s in (done.stdout, done.stderr))
        return int(done.returncode), "\n".join(streams).strip()
    except _UNREAD as error:
        return None, str(error)


def _all(*values: Optional[bool]) -> Optional[bool]:
    """Three-valued AND: one disproven fact disproves, an unread one leaves it unread."""
    if any(value is False for value in values):
        return False
    return None if any(value is None for value in values) else True


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _plist(app: pathlib.Path) -> Optional[dict]:
    try:
        with open(app / "Contents" / "Info.plist", "rb") as stream:
            data = plistlib.load(stream)
    except Exception:  # absent, unreadable or malformed alike: an unread fact
        return None
    return data if isinstance(data, dict) else None


def _manifest(app: pathlib.Path) -> Optional[dict]:
    """The seed manifest where the launcher reads it (Frameworks), else its data copy; ``{}`` when absent."""
    for folder in ("Frameworks", "Resources"):
        try:
            raw = json.loads((app / "Contents" / folder / BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None
    return {}


def _entry(app: pathlib.Path) -> Optional[bool]:
    """The launcher's own entry rule (``macos_source_entry``); present-but-unusable counts as absent."""
    entry = app / "Contents" / "MacOS" / "ouroboros-shell"
    if macos_source_entry(app / "Contents" / "Frameworks") == entry:
        return True
    try:
        os.lstat(entry)
    except FileNotFoundError:
        return False if (app / "Contents" / "MacOS").is_dir() else None
    except OSError:
        return None
    return False


def _bytes(app: pathlib.Path) -> Optional[int]:
    """Logical bytes of the bundle's files, symlinks not followed; ``None`` when any part is unreadable."""
    total, errors = 0, []
    try:
        for root, _dirs, files in os.walk(app, onerror=errors.append):
            total += sum(os.lstat(os.path.join(root, name)).st_size for name in files)
    except OSError:
        return None
    return None if errors else total


def _signed(app: pathlib.Path, runner: Callable, team_id: Optional[str]) -> dict:
    """Signature, Team ID, bundle id and Gatekeeper facts: True proven, False disproven, None unread."""
    plist = _plist(app)
    code, _ = _run(runner, [CODESIGN, "--verify", "--deep", "--strict", app])
    signature = None if code is None else code == 0
    # codesign names the code-directory hash (``CDHash=``) only from verbosity 3 on; 2 omits it.
    code, shown = _run(runner, [CODESIGN, "-dv", "--verbose=4", app])
    fields = dict(line.split("=", 1) for line in shown.splitlines() if "=" in line) if code == 0 else {}
    team = fields.get("TeamIdentifier", "").strip().replace("not set", "")
    signing_id = fields.get("Identifier", "").strip()
    cdhash = fields.get("CDHash", "").strip().lower()  # the code directory: executable plus sealed resources
    team_ok = None if code is None or not team_id else code == 0 and team == team_id  # unsigned: no Team ID
    plist_id = None if plist is None else plist.get("CFBundleIdentifier") == BUNDLE_ID
    signing_ok = None if code is None else code == 0 and signing_id == BUNDLE_ID
    code, assessed = _run(runner, [SPCTL, "-a", "-t", "exec", "-vv", app])
    source = next((line.split("=", 1)[1].strip() for line in assessed.splitlines()
                   if line.startswith("source=")), None)
    # spctl exits 3 on a rejection; an acceptance without a source (assessment disabled) proves nothing.
    gatekeeper = (source == _NOTARIZED) if code == 0 and source else (False if code == 3 else None)
    return {"signature": signature, "team_id": team_ok, "bundle_id": _all(plist_id, signing_ok),
            "gatekeeper": gatekeeper, "team": team or None, "signing_id": signing_id or None,
            "cdhash": cdhash or None, "source": source, "plist": plist}


def _owner(app: pathlib.Path, owners: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """A package-manager receipt naming exactly this bundle; the caller reads receipts, this only matches."""
    here = os.path.realpath(app)
    for receipt in owners or ():
        if isinstance(receipt, Mapping) and receipt.get("manager") and receipt.get("path") \
                and os.path.realpath(str(receipt["path"])) == here:
            return receipt
    return {}


def installed_app(bundle_path: Any, *, runner: Callable, official_team_id: Optional[str] = None,
                  published_tags: Optional[Iterable[str]] = None, owners: Iterable[Mapping[str, Any]] = ()) -> dict:
    """Origin facts of one installed bundle; ``official`` only when every check in ``CHECKS`` holds.

    ``official_team_id`` and ``published_tags`` are the caller's facts: without them origin stays
    ``unknown``. The team is never this app's own claim: the caller proves it from the official
    payload of this app's release (``verify`` reads its team and code-directory hash, which must
    equal this app's ``cdhash``). ``owners`` are package-manager receipts (``manager``, ``path``,
    optional ``upgrade`` and ``version``); one naming this path is reported (``manager``,
    ``manager_version``: what that manager last installed, apart from the bundle's own ``version``)
    and changes no origin. Whether Gatekeeper reports the app notarized is a fact here
    (``notarized``), never its provenance: the package that replaces it is checked for
    notarization in full (``verify``).
    """
    app = pathlib.Path(os.path.abspath(str(bundle_path)))
    signed = _signed(app, runner, official_team_id) if _plist(app) is not None else {}  # no plist: no bundle
    plist, manifest = signed.get("plist"), _manifest(app)
    version = str(plist.get("CFBundleShortVersionString") or "") if plist else ""
    tag = release_sync.normalize_release_tag(manifest.get("release_tag")) if manifest is not None else ""
    if plist is None or manifest is None:
        release = None
    elif not tag or tag != f"v{version}" or str(manifest.get("app_version") or "") != version:
        release = False  # no tag, or a manifest that disagrees with the app it sits in
    elif published_tags is None:
        release = None
    else:
        release = tag in {release_sync.normalize_release_tag(item) for item in published_tags}
    checks = {**{name: signed.get(name) for name in CHECKS[:-1]}, "release": release}
    owner, verdict = _owner(app, owners), _all(*checks.values())  # verdict: False on a readable disqualifier
    origin = {True: "official", False: "custom"}.get(verdict, "unknown")
    try:
        device = os.stat(app).st_dev
    except OSError:
        device = None
    return {"path": str(app), "origin": origin, "checks": checks,
            "reasons": [name for name, value in checks.items() if value is not True],
            "manager": str(owner["manager"]) if owner else None,
            "manager_upgrade": str(owner.get("upgrade") or "").strip() or None,
            "manager_version": str(owner.get("version") or "").strip() or None,
            "team": signed.get("team"), "signing_id": signed.get("signing_id"), "cdhash": signed.get("cdhash"),
            "notarized": signed.get("gatekeeper"), "gatekeeper_source": signed.get("source"),
            "version": version or None,
            "release_tag": tag or None, "entry": _entry(app),
            "device": device, "bytes": _bytes(app)}


def requirement(installed: Optional[Mapping[str, Any]], target: Optional[int], *, serving: Optional[int],
                hosted: Optional[int]) -> dict:
    """``none``/``needed``/``unknown``: can this app host the target's source launcher as the launcher requires?

    ``target`` and ``serving`` are the protocols the target and the serving trees declare
    (``protocol_at``); ``hosted`` is the one this app's own release tree declares. A target that
    declares one is a source-first shell, and the current launcher's host rule is the signed
    source entry (``launcher_bootstrap.macos_source_entry``, read as ``installed_app``'s
    ``entry``): without it source restarts lose the app profile and notification identity, and
    source alone cannot add it to a sealed app. An app without it (7.7.0) hands off with plain
    Python and checks nothing, so its silence proves nothing: it needs the release that has the
    entry, whatever the protocols say. With the entry, the serving source passed its protocol
    check, so an unchanged protocol needs nothing. Origin and versions are never read here.
    """
    entry = (installed or {}).get("entry")
    if target is None:
        status, reason = "none", "nothing_declared"
    elif entry is False:
        status, reason = "needed", "signed_entry_missing"
    elif entry is not True:
        status, reason = "unknown", "entry_unread"
    elif target == serving:
        status, reason = "none", "protocol_unchanged"
    elif hosted is None:
        status, reason = "unknown", "protocol_unread"
    else:
        status, reason = ("none", "protocol_hosted") if hosted == target else ("needed", "protocol_differs")
    return {"status": status, "reason": reason, "required": target, "serving": serving, "entry": entry,
            "hosted": hosted}


def select_package(repo: Any, protocol: int, published: Iterable[Mapping[str, Any]],
                   capsule: Callable[[str], Optional[str]]) -> Optional[dict]:
    """The newest published official release that hosts ``protocol`` AND carries the macOS image: the app to stage.

    ``published`` lists the official tags newest first (``git_ops.list_official_update_tags``);
    a tag whose tree is not here or declares another protocol is skipped. A tag alone is not a
    package: ``capsule(version)`` reads that release's checksum capsule (``None`` when the release
    publishes none), which must name exactly one SHA-256 for the image, so a newer tag without
    assets never shadows an older release that has them. ``None``: no published app hosts the
    protocol yet. A capsule that cannot be read at all raises: unknown is never a choice.
    """
    for item in published or ():
        tag = release_sync.normalize_release_tag(str(item.get("tag") or ""))
        version = tag[1:] if tag.startswith("v") else ""
        if not release_sync.is_release_version(version) or protocol_at(repo, str(item.get("sha") or tag)) != protocol:
            continue
        sums = capsule(version)
        digest, _why = _capsule_digest(sums, release_sync.release_asset_name(ASSET, version))
        if digest is not None:
            return {"tag": tag, "version": version, "sha": str(item.get("sha") or ""), "sha256": digest,
                    "sha256sums": sums}
    return None


def manual_instruction(target_version: str, installed: Optional[Mapping[str, Any]]) -> str:
    """One English sentence for the owner while the automatic path is off or refused."""
    version = str(target_version or "").strip()
    if not release_sync.is_release_version(version):
        raise ValueError(f"{target_version!r} is not a release version")
    facts = installed or {}
    origin = facts.get("origin")
    if facts.get("manager") and origin != "official":  # a managed app not proven official: its manager updates it
        manager = str(facts["manager"])
        upgrade = f" ({facts['manager_upgrade']})" if facts.get("manager_upgrade") else ""
        return (f"Ouroboros {version} is published and this app is managed by {manager}, so update it through "
                f"{manager}{upgrade} once that offers {version}, instead of replacing the app by hand.")
    url = release_sync.release_asset_download_url(ASSET, version)
    where = f" at {facts['path']}" if facts.get("path") else ""
    if origin == "official":
        return (f"Download Ouroboros {version} from {url}, quit Ouroboros, then drag {PAYLOAD} from that disk "
                f"image over the current app{where}.")
    if origin == "custom":
        return (f"Ouroboros {version} is published at {url}; the app{where} is not that official signed release, so "
                f"it is never replaced automatically: rebuild it from v{version} or install the disk image by hand.")
    return (f"Ouroboros {version} is published at {url}; the origin of the app{where} could not be read, so it is "
            "not replaced automatically: install the disk image by hand.")


def _capsule_digest(capsule: Any, asset: str) -> tuple:
    if not isinstance(capsule, str) or not capsule.strip():
        return None, "No release checksum capsule (SHA256SUMS) was supplied."
    found = {match.group(1).lower() for match in map(_SUMS_LINE.match, capsule.splitlines())
             if match and match.group(2).strip() == asset}
    if len(found) != 1:
        return None, f"The checksum capsule names {'no' if not found else 'conflicting'} SHA-256 for {asset}."
    return found.pop(), ""


def staging_path(app: Any, version: str, owner: str = "") -> pathlib.Path:
    """The folder beside the installed ``app`` where ``owner`` (an Update candidate) stages ``version``: its own,
    so preparing again never meets an earlier candidate's stage, which leaves only with that candidate."""
    app = pathlib.Path(str(app))
    return app.parent / f".{app.stem}-update-{version}{f'-{owner}' if owner else ''}"


def plan(target_version: str, installed: Optional[Mapping[str, Any]], *,
         release_metadata: Optional[Mapping[str, Any]], owner: str = "") -> dict:
    """Checksum, origin and same-volume staging for one package version, or one typed refusal.

    ``release_metadata``: ``sha256sums`` (the release's SHA256SUMS text). Staging (``owner``'s own
    folder, ``staging_path``) and the backup sit beside the installed app on its volume: the backup
    is a rename there and needs no space; the staged copy's need is the payload's measured size,
    checked at staging.
    """
    version, facts = str(target_version or "").strip(), dict(installed or {})
    result: dict = {"status": "refused", "reason": None, "detail": "", "version": version, "asset": None,
                    "url": None, "sha256": None, "staging_dir": None, "dmg_path": None,
                    "staged_app": None, "backup_app": None, "instruction": None}

    def refuse(reason: str, detail: str) -> dict:
        result.update(status="refused", reason=reason, detail=detail)
        return result

    if not release_sync.is_release_version(version):
        return refuse("invalid_version", f"{target_version!r} is not a release version.")
    asset = release_sync.release_asset_name(ASSET, version)
    result.update(asset=asset, url=release_sync.release_asset_download_url(ASSET, version),
                  instruction=manual_instruction(version, facts))
    origin = facts.get("origin")
    if origin != "official":
        return refuse(_ORIGIN_REFUSAL.get(origin, "unknown_origin"),
                      f"The installed app's origin is {origin or 'unknown'}; it is never replaced automatically.")
    digest, why = _capsule_digest((release_metadata or {}).get("sha256sums"), asset)
    if digest is None:
        return refuse("checksum_missing", why)
    result["sha256"] = digest
    app = pathlib.Path(str(facts.get("path") or ""))
    parent, staging = app.parent, staging_path(app, version, owner)
    backup = parent / f".{app.stem}-{facts.get('version') or 'previous'}-backup{app.suffix}"
    result.update(staging_dir=str(staging), dmg_path=str(staging / asset), staged_app=str(staging / app.name),
                  backup_app=str(backup))
    if os.path.lexists(staging) and (os.path.islink(staging) or not staging.is_dir()):
        return refuse("staging_unsafe", f"{staging} exists and is not a plain directory.")
    try:
        devices = {"app": facts.get("device"), "beside": os.stat(parent).st_dev,
                   **({"staging": os.stat(staging).st_dev} if staging.is_dir() else {})}
    except OSError as error:
        return refuse("volume_unread", f"The volume beside the app could not be read: {error}.")
    if not all(_is_int(value) for value in devices.values()):
        return refuse("volume_unread", f"The app's volume could not be read (devices {devices}).")
    if len(set(devices.values())) != 1:
        return refuse("other_volume", f"Staging beside the app is not on the app's volume (devices {devices}).")
    if not os.access(parent, os.W_OK | os.X_OK):
        return refuse("not_writable", f"{parent} is not writable by this user.")
    if os.path.lexists(backup):
        return refuse("backup_exists", f"An earlier backup is still at {backup}; it is never overwritten.")
    result.update(status="ready", detail=f"Staging beside {app}.")
    return result


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_of(app: Any) -> Optional[str]:
    """``"3.10"``: the Python the bundle's embedded interpreter runs (its ``lib/pythonX.Y``), else ``None``."""
    found = sorted(path.name[len("python"):] for path in (
        pathlib.Path(str(app)) / "Contents" / "Frameworks" / "python-standalone" / "lib").glob("python3.*")
        if path.is_dir())
    return found[0] if len(found) == 1 else None


def _payload_problem(app: pathlib.Path, runner: Callable, version: str, team_id: Optional[str],
                     runtime: Optional[str] = None, notarized: bool = True) -> tuple:
    """``(reason, detail)`` for one bundle from the image, ``(None, "")`` when it is that official release.

    With ``team_id`` the payload must be signed by that team; ``runtime`` (``"3.10"``) asks for an
    embedded Python of that version, the one this installation's packages were prepared for, and
    the signed source entry the staged installer runs. ``notarized`` False reads an earlier
    release's payload as a provenance proof only: it is never launched.
    """
    if os.path.islink(app) or not app.is_dir():
        return "payload_missing", f"{app} is not an app bundle."
    signed, manifest = _signed(app, runner, team_id), _manifest(app) or {}
    if signed["signature"] is not True:
        return "signature_invalid", f"codesign --verify --deep --strict did not accept {app}."
    if notarized and signed["gatekeeper"] is not True:
        return "gatekeeper_rejected", f"spctl did not accept {app} as notarized (source {signed['source']})."
    plist = signed["plist"] or {}
    seen = {"bundle_id": plist.get("CFBundleIdentifier"), "signing_id": signed["signing_id"],
            "version": plist.get("CFBundleShortVersionString"), "app_version": manifest.get("app_version"),
            "release_tag": manifest.get("release_tag"), **({"team": signed["team"]} if team_id else {})}
    wanted = {"bundle_id": BUNDLE_ID, "signing_id": BUNDLE_ID, "version": version, "app_version": version,
              "release_tag": f"v{version}", **({"team": team_id} if team_id else {})}
    differs = {key: seen[key] for key in wanted if seen[key] != wanted[key]}
    if differs or not signed["team"] or not signed["cdhash"]:
        return "identity_mismatch", f"{app} is not the official {version} payload: {differs or 'unsigned'}."
    if runtime is not None:
        if _entry(app) is not True:
            return "entry_missing", f"{app} has no signed source entry to run the installer and the source."
        if runtime_of(app) != runtime:
            return "runtime_differs", (f"{app} runs Python {runtime_of(app) or 'unknown'}; this installation's "
                                       f"packages are for {runtime}.")
    return None, ""


def _stage(payload: pathlib.Path, staged: pathlib.Path, runner: Callable, version: str, team_id: Optional[str],
           runtime: Optional[str]) -> tuple:
    """Measure the mounted app, copy it out beside the installed one, and check the copy: it is what runs."""
    if os.path.lexists(staged):  # ditto merges into an existing tree; never stage over one
        return "stage_failed", f"{staged} already exists."
    try:
        staged.parent.mkdir(parents=True, exist_ok=True)
        need, free = _bytes(payload), shutil.disk_usage(staged.parent).free
    except OSError as error:
        return "stage_failed", f"The staging directory could not be prepared: {error}"
    if need is None:
        return "size_unknown", f"The size of {payload} could not be read."
    if need > free:
        return "insufficient_space", f"Staging needs {need} bytes beside the app; {free} are free."
    code, output = _run(runner, [DITTO, payload, staged])
    if code != 0:
        return "stage_failed", f"ditto failed: {' '.join(output.split())[-300:]}"
    return _payload_problem(staged, runner, version, team_id, runtime)


def verify(dmg_path: Any, expected_sha256: Any, *, runner: Callable, version: str,
           official_team_id: Optional[str] = None, stage_to: Any = None, runtime: Optional[str] = None,
           notarized: bool = True) -> dict:
    """Image bytes against the capsule, then the mounted payload; with ``stage_to`` also its staged copy.

    The hash is checked before anything mounts. The image attaches read-only, unbrowsed and
    without auto-open at a private mount point, and detaches on every path. ``team`` and
    ``cdhash`` are the payload's: its signer and code-directory hash, the identity an installed
    app of this release is compared with. Without ``official_team_id`` the payload's own team
    is reported, never trusted by itself. ``notarized`` False: the installed app's own release,
    read for its provenance only; a package to install is always checked for notarization.
    """
    image = pathlib.Path(str(dmg_path))
    result: dict = {"status": "failed", "reason": None, "detail": "", "sha256": None, "detached": None,
                    "staged": None, "team": None, "cdhash": None}

    def fail(reason: str, detail: str) -> dict:
        result.update(status="failed", reason=reason, detail=detail)
        return result

    expected = str(expected_sha256 or "").strip().lower()
    if not _SHA256.match(expected):
        return fail("checksum_missing", "No SHA-256 from the release checksum capsule was supplied.")
    try:
        result["sha256"] = _sha256(image)
    except OSError as error:
        return fail("dmg_missing", f"The disk image could not be read: {error}.")
    if result["sha256"] != expected:
        return fail("sha256_mismatch", f"The disk image's SHA-256 is {result['sha256']}; the capsule names {expected}.")
    mount = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-dmg-"))
    attached, output = _run(runner, [HDIUTIL, "attach", "-readonly", "-nobrowse", "-noautoopen",
                                     "-mountpoint", mount, image])
    reason, detail = (None, "") if attached == 0 else (
        "attach_failed", f"hdiutil attach failed: {' '.join(output.split())[-300:]}")
    try:
        if reason is None:
            reason, detail = _payload_problem(mount / PAYLOAD, runner, version, official_team_id,
                                              notarized=notarized or stage_to is not None)
            if reason is None:
                identity = _signed(mount / PAYLOAD, runner, official_team_id)
                result.update(team=identity["team"], cdhash=identity["cdhash"])
        if reason is None and stage_to is not None:
            reason, detail = _stage(mount / PAYLOAD, pathlib.Path(str(stage_to)), runner, version,
                                    official_team_id, runtime)
            result["staged"] = str(stage_to) if reason is None else None
    finally:  # a refused attach mounted nothing; an unread one may have, so it detaches too
        result["detached"] = None if attached not in (0, None) else any(
            _run(runner, [HDIUTIL, "detach", *force, mount])[0] == 0 for force in ([], ["-force"]))
        if result["detached"] is not False:
            with contextlib.suppress(OSError):
                os.rmdir(mount)  # only an empty directory, never a tree that may still be mounted
    if reason is not None:
        return fail(reason, detail)
    if not result["detached"]:
        return fail("detach_failed", f"The disk image is still attached at {mount}.")
    result.update(status="verified", detail=f"{image.name} is the official {version} release.")
    return result
