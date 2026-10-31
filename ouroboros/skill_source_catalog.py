"""Catalogs of the connected skill sources, normalized into one item shape.

A git-hosted source is read from its cache clone (``skill_source_git``): a repository with ``hub.json``
at its root is a hub (``entries/<name>.json``, the maintainer's ``official.json``, ``categories.json``
and an optional ``signals`` branch); any other repository is a plain skills repository whose folders
with ``SKILL.md`` or ``skill.json`` are its skills. A refresh parses the commit it fetched into a
snapshot beside the clone, so every reader sees one commit and no reader ever waits for the network;
it dates each entry by the newest fetched commit touching its release (``updated_at``).
The built-in source is the bundled ``skills/`` folder with ``skills/builtin.json``; a local folder is
scanned in place; a registry (ClawHub) answers searches only.

The release digest of a tree is ``skill_identity.payload_digest`` (``tree_release_digest`` reads the
manifest once for it); an installed skill's digest is the loader's own content hash. Installed state,
updates and the listing fields of an installed skill are joined here from install records and the
cached snapshots, without network and without writing (the Skills list read stays passive).
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ouroboros.skill_identity import is_release_digest
from ouroboros.skill_source_git import (
    clone_dir,
    fetch_signals,
    last_changed,
    read_meta,
    refresh_in_background,
    view_cache_dir,
    update_clone,
)
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "catalog.json"
SNAPSHOT_SCHEMA_VERSION = 1
REFRESH_AFTER_SEC = 600  # a cache older than this is refreshed in the background when a catalog is shown
_MAX_ABOUT_CHARS = 20_000
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".ouroboros_env", ".github"})
_MEMO_LOCK = threading.Lock()
_BUILTIN_MEMO: Dict[str, Dict[str, Any]] = {}

_PERMISSION_WORDS = {
    "net": "Uses the internet", "tool": "Adds tools Ouroboros can call", "route": "Serves its own web routes",
    "widget": "Shows a widget", "subprocess": "Runs programs on this computer",
    "companion_process": "Keeps a helper program running", "supervised_task": "Runs a background task",
    "inject_chat": "Writes into the chat", "presence": "Can act as a presence", "notify_owner": "Can notify you",
    "subscribe_event": "Follows chat events", "read_settings": "Reads its settings",
}


# --- manifests and the release digest -------------------------------------------------------------

def read_manifest(skill_dir: pathlib.Path) -> Any:
    """The parsed manifest of a tree, or ``None`` when it has none or it does not parse."""
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

    for name in ("SKILL.md", "skill.json"):
        path = pathlib.Path(skill_dir) / name
        if path.is_file():
            try:
                return parse_skill_manifest_text(path.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def tree_release_digest(skill_dir: pathlib.Path, manifest: Any = None, *, bucket: Optional[str] = None) -> str:
    """Release digest of a skill tree (``bucket``: a staged tree's destination); raises when it has
    no readable manifest or payload."""
    from ouroboros.skill_identity import payload_digest

    manifest = manifest if manifest is not None else read_manifest(skill_dir)
    if manifest is None:
        raise ValueError(f"{pathlib.Path(skill_dir).name}: no readable SKILL.md or skill.json")
    return payload_digest(pathlib.Path(skill_dir), manifest_entry=manifest.entry,
                          manifest_scripts=manifest.scripts, manifest=manifest, bucket=bucket)


def installed_release_digest(loaded: Any) -> str:
    """The release digest of an installed skill's current bytes: the loader's content hash ('' when
    its payload is unreadable)."""
    digest = str(getattr(loaded, "content_hash", "") or "")
    return digest if is_release_digest(digest) else ""


def _connector_facts(manifest: Any, *, icon_source: bytes | None = None) -> Dict[str, Any]:
    from ouroboros.extension_ui_validation import validate_contributions
    from ouroboros.icon_geometry import icon_spec

    clean, _ = validate_contributions(manifest) if manifest is not None else ([], [])
    places = [row for row in clean if row.get("provides") == "place"]
    icon = getattr(manifest, "raw_extra", {}).get("icon", "")
    if not icon and places:
        icon = places[0].get("icon", "")
    services = [str(row.get("destination") or "")[len("vendor:"):].strip() for row in places
                if str(row.get("destination") or "").startswith("vendor:")]
    return {"connects": connect_services(services), "icon": icon_spec(icon, icon_source),
            "title": getattr(manifest, "raw_extra", {}).get("title") or next((p["label"] for p in places if p.get("label")), "")}


def connect_services(*values: Any) -> List[str]:
    """The services a connector connects (``Slack``), from manifests or Hub rows; [] for none.

    The Hub index writes a list of service names; anything else names no service.
    """
    names: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple)):
            names.update(str(item).strip() for item in value if isinstance(item, str) and item.strip())
    return sorted(names)


def _catalog_icon(raw: Any, folder: pathlib.Path | None = None) -> Dict[str, str]:
    """Catalog hints are display data, never live extension authority."""
    from ouroboros.icon_geometry import MAX_ICON_BYTES, icon_spec

    if isinstance(raw, dict):
        return icon_spec("ui/icon.svg", raw.get("svg")) if raw.get("kind") == "svg" else icon_spec(raw.get("name", ""))
    source = None
    if raw == "ui/icon.svg" and folder is not None:
        path = _inside(folder, raw)
        try:
            if path is not None:
                with path.open("rb") as stream:
                    source = stream.read(MAX_ICON_BYTES + 1)
        except OSError:
            pass
    return icon_spec(raw, source)


def _manifest_facts(manifest: Any, folder: pathlib.Path | None = None) -> Dict[str, Any]:
    if manifest is None:
        return {"type": "", "needs_keys": [], "permissions": [], "connects": [], "icon": {}}
    keys: List[str] = []
    for raw in manifest.env_from_settings or []:
        key = str(raw or "").strip().upper()
        if key and key not in keys:
            keys.append(key)
    facts = _connector_facts(manifest)
    raw_icon = manifest.raw_extra.get("icon") or next((row.get("icon") for row in manifest.contributes if row.get("icon")), "")
    facts["icon"] = _catalog_icon(raw_icon, folder)
    return {**facts, "type": str(manifest.type or ""), "needs_keys": keys,
            "permissions": [str(item) for item in manifest.permissions or []]}


# --- parsing one source tree into a snapshot -------------------------------------------------------

def _read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _inside(root: pathlib.Path, rel: str) -> Optional[pathlib.Path]:
    rel = str(rel or "").strip().strip("/")
    pure = pathlib.PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    target = (root / pathlib.Path(*pure.parts)) if pure.parts else root
    try:
        target.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return target


def _hub_entry(raw: Any, stem: str, clone: pathlib.Path) -> Dict[str, Any]:
    from ouroboros.skill_loader import _sanitize_skill_name

    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("entry schema_version must be 2")
    name = str(raw.get("name") or "")
    if not name or name != stem or _sanitize_skill_name(name) != name:
        raise ValueError("entry name must equal its file name and be a valid skill name")
    source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    kind = str(source.get("kind") or "")
    digest = str(raw.get("digest") or "")
    if kind not in {"hosted", "git"} or not is_release_digest(digest):
        raise ValueError("entry needs source.kind hosted|git and a release digest")
    category = raw.get("category") if isinstance(raw.get("category"), list) else []
    withdrawn = raw.get("withdrawn") if isinstance(raw.get("withdrawn"), dict) else None
    entry: Dict[str, Any] = {
        "name": name, "description": str(raw.get("description") or ""), "version": str(raw.get("version") or ""),
        "category": [str(item) for item in category][:2], "owner": str(raw.get("owner") or ""),
        "digest": digest, "withdrawn": withdrawn,
        "withdrawn_releases": [item for item in raw.get("withdrawn_releases", []) if isinstance(item, dict)],
    }
    if raw.get("install_specs"):
        entry["install_specs"] = raw["install_specs"]
    if kind == "hosted":
        payload = _inside(clone, str(source.get("path") or f"skills/{name}"))
        if payload is None:
            raise ValueError("hosted path leaves the repository")
        entry["source"] = {"kind": "hosted", "path": payload.relative_to(clone).as_posix()}
        entry.update(_manifest_facts(read_manifest(payload), payload))
    else:
        from ouroboros.skill_sources import validate_git_url

        commit = str(source.get("commit") or "").lower()
        if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            raise ValueError("pointer entry needs a 40-character commit")
        entry["source"] = {"kind": "git", "repo": validate_git_url(str(source.get("repo") or "")),
                           "path": str(source.get("path") or "").strip("/"), "commit": commit}
        entry.update(_manifest_facts(None))
    if "connects" in raw:
        entry["connects"] = connect_services(entry["connects"], raw["connects"])
    if not entry["icon"] and "icon" in raw:
        entry["icon"] = _catalog_icon(raw["icon"])
    return entry


def parse_hub_tree(clone: pathlib.Path) -> Dict[str, Any]:
    """Snapshot of a hub repository tree (``hub.json`` present)."""
    hub = _read_json(clone / "hub.json")
    hub = hub if isinstance(hub, dict) else {}
    entries: List[Dict[str, Any]] = []
    problems: List[str] = []
    entries_dir = clone / "entries"
    for path in sorted(entries_dir.glob("*.json")) if entries_dir.is_dir() else []:
        try:
            entries.append(_hub_entry(_read_json(path), path.stem, clone))
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")
    official: Dict[str, Dict[str, Any]] = {}
    curation = _read_json(clone / "official.json")
    for item in (curation.get("endorsements") if isinstance(curation, dict) else None) or []:
        if isinstance(item, dict) and item.get("name") and is_release_digest(item.get("digest")):
            official[str(item["name"])] = {"digest": str(item["digest"]), "recommended": item.get("recommended") is True}
    categories = _read_json(clone / "categories.json")
    category_rows = [
        {"id": str(item.get("id")), "label": str(item.get("label") or item.get("id"))}
        for item in ((categories.get("categories") if isinstance(categories, dict) else None) or [])
        if isinstance(item, dict) and item.get("id")
    ]
    return {"format": "hub", "hub": {"name": str(hub.get("name") or ""), "maintainer": str(hub.get("maintainer") or "")},
            "entries": entries, "official": official, "categories": category_rows, "problems": problems}


def _skill_folders(root: pathlib.Path) -> Iterable[pathlib.Path]:
    def has_manifest(path: pathlib.Path) -> bool:
        return (path / "SKILL.md").is_file() or (path / "skill.json").is_file()

    if has_manifest(root):
        yield root
        return
    for child in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if child.name in _SKIP_DIRS:
            continue
        nested = [child] if has_manifest(child) else sorted(
            p for p in child.iterdir() if p.is_dir() and not p.name.startswith(".") and has_manifest(p))
        yield from nested


def parse_plain_tree(root: pathlib.Path, *, repo_name: str = "") -> Dict[str, Any]:
    """Snapshot of a plain skills repository or folder: every folder with a manifest is a skill."""
    from ouroboros.skill_loader import _sanitize_skill_name

    entries: List[Dict[str, Any]] = []
    problems: List[str] = []
    seen: set[str] = set()
    if not root.is_dir():
        return {"format": "plain", "entries": [], "categories": [], "official": {}, "problems": ["folder is missing"]}
    for folder in _skill_folders(root):
        manifest = read_manifest(folder)
        single = (getattr(manifest, "name", "") or repo_name or root.name) if folder == root else folder.name
        name = _sanitize_skill_name(single)
        if name in seen:
            problems.append(f"{folder.name}: name {name!r} appears twice")
            continue
        seen.add(name)
        entry: Dict[str, Any] = {"name": name, "description": str(getattr(manifest, "description", "") or ""),
                                 "version": str(getattr(manifest, "version", "") or ""), "category": [],
                                 "owner": "", "withdrawn": None,
                                 "source": {"kind": "folder", "path": folder.relative_to(root).as_posix()},
                                 **_manifest_facts(manifest, folder)}
        try:
            entry["digest"] = tree_release_digest(folder, manifest)
        except Exception as exc:
            entry["digest"] = ""
            problems.append(f"{folder.name}: {exc}")
        entries.append(entry)
    return {"format": "plain", "entries": entries, "official": {}, "categories": [], "problems": problems}


def _parse_signals(raw: Optional[bytes]) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(raw.decode("utf-8")) if raw else None
    except (UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("skills"), dict) else None


def _date_entries(clone: pathlib.Path, snapshot: Dict[str, Any]) -> None:
    """``updated_at`` of each entry: the newest commit in the clone touching its release (a hosted payload,
    a pointer's entry file, a plain repository's skill folder); absent when the history cannot tell."""
    paths = {}
    for entry in snapshot.get("entries") or []:
        source = entry.get("source") or {}
        path = (f"entries/{entry['name']}.json" if source.get("kind") == "git" else
                str(pathlib.PurePosixPath(str(snapshot.get("path") or "."), str(source.get("path") or "."))))
        paths[entry["name"]] = "" if path == "." else path
    try:
        dates = last_changed(clone, paths)
    except Exception:
        log.debug("skill source history unavailable for dates", exc_info=True)
        return
    for entry in snapshot.get("entries") or []:
        if dates.get(entry["name"]):
            entry["updated_at"] = dates[entry["name"]]


def refresh_source(drive_root: pathlib.Path, row: Dict[str, Any], *, wait_sec: float = 180) -> Dict[str, Any]:
    """Fetch a git-hosted source and write the snapshot of the fetched commit; returns its meta."""
    cache = view_cache_dir(drive_root, row)

    def write_snapshot(clone: pathlib.Path, commit: str) -> None:
        base = _inside(clone, str(row.get("path") or "")) or clone
        if (clone / "hub.json").is_file():
            snapshot = parse_hub_tree(clone)
            snapshot["signals"] = _parse_signals(fetch_signals(clone, url=row["url"]))
        else:
            snapshot = parse_plain_tree(base, repo_name=pathlib.PurePosixPath(str(row.get("url") or "")).stem)
            snapshot["path"] = base.relative_to(clone).as_posix() if base != clone else ""
        _date_entries(clone, snapshot)
        snapshot.update({"schema_version": SNAPSHOT_SCHEMA_VERSION, "commit": commit, "read_at": utc_now_iso()})
        atomic_write_json(cache / SNAPSHOT_FILENAME, snapshot)

    return update_clone(drive_root, row, wait_sec=wait_sec, after_fetch=write_snapshot)


def read_snapshot(drive_root: pathlib.Path, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = read_json_dict(view_cache_dir(drive_root, row) / SNAPSHOT_FILENAME)
    return data if data and data.get("schema_version") == SNAPSHOT_SCHEMA_VERSION else None


def builtin_snapshot(repo_dir: pathlib.Path) -> Dict[str, Any]:
    """The bundled skills folder as a source snapshot (memoized per process: it ships with the app);
    a built-in whose manifest ``os`` excludes this host is not offered here."""
    from ouroboros.contracts.skill_manifest import host_platform
    from ouroboros.skill_builtin import applies_here

    key = f"{pathlib.Path(repo_dir).resolve()}\0{host_platform()}"
    with _MEMO_LOCK:
        cached = _BUILTIN_MEMO.get(key)
    if cached is not None:
        return cached
    root = pathlib.Path(repo_dir) / "skills"
    index = _read_json(root / "builtin.json")
    marks = {str(item.get("name")): item for item in ((index or {}).get("skills") or []) if isinstance(item, dict)}
    snapshot = parse_plain_tree(root)
    snapshot["entries"] = [entry for entry in snapshot["entries"]
                           if applies_here(root / entry["source"]["path"])]
    for entry in snapshot["entries"]:
        mark = marks.get(entry["name"], {})
        entry["category"] = [str(item) for item in mark.get("category") or []][:2]
        entry["recommended"] = mark.get("recommended") is True
        if entry.get("digest"):
            snapshot["official"][entry["name"]] = {"digest": entry["digest"], "recommended": entry["recommended"]}
    snapshot["format"] = "builtin"
    with _MEMO_LOCK:
        _BUILTIN_MEMO[key] = snapshot
    return snapshot


def _stale(meta: Dict[str, Any]) -> bool:
    stamp = str(meta.get("fetched_at") or meta.get("error_at") or "")
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() > REFRESH_AFTER_SEC


def snapshot_for(drive_root: pathlib.Path, row: Dict[str, Any], *, repo_dir: Optional[pathlib.Path] = None,
                 refresh: bool = False) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """``(snapshot, status)`` of one source from local data; may start a background refresh."""
    kind = row["kind"]
    if kind == "builtin":
        from ouroboros.config import REPO_DIR

        return builtin_snapshot(pathlib.Path(repo_dir or REPO_DIR)), {"state": "ready"}
    if kind == "local":
        return parse_plain_tree(pathlib.Path(row["path"])), {"state": "ready", "last_read_at": utc_now_iso()}
    if kind == "registry":
        return None, {"state": "search"}
    meta = read_meta(drive_root, row)
    snapshot = read_snapshot(drive_root, row)
    if refresh or snapshot is None or _stale(meta):
        refresh_in_background(drive_root, row, refresh_source)
    from ouroboros.skill_source_git import is_refreshing

    status: Dict[str, Any] = {"state": "ready" if snapshot else ("reading" if is_refreshing(drive_root, row) or not meta
                                                                 else "error"),
                              "last_read_at": str((snapshot or {}).get("read_at") or "")}
    if meta.get("error"):
        status["error"] = str(meta["error"])
    if snapshot is not None and is_refreshing(drive_root, row):
        status["refreshing"] = True
    return snapshot, status


# --- items -------------------------------------------------------------------------------------------

def _in_part(row: Dict[str, Any], entry: Dict[str, Any], official: Dict[str, Any]) -> bool:
    endorsed = official.get(entry["name"])
    is_official = bool(endorsed and endorsed["digest"] == entry.get("digest"))
    part = row.get("part")
    return part is None or (part == "official") == is_official


def _signals(snapshot: Dict[str, Any], entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    signals = snapshot.get("signals")
    if not isinstance(signals, dict):
        return None
    skill = (signals.get("skills") or {}).get(entry["name"])
    repo = (entry.get("source") or {}).get("repo")
    stars = ((signals.get("author_repos") or {}).get(repo) or {}).get("stars") if repo else None
    if not isinstance(skill, dict) and stars is None:
        return None
    skill = skill if isinstance(skill, dict) else {}
    return {"votes": skill.get("votes"), "comments": skill.get("comments"),
            "discussion_url": skill.get("discussion_url"), "author_repo_stars": stars}


def items_for(row: Dict[str, Any], snapshot: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The §3.3 items one source contributes (installed state is joined by the caller)."""
    if not snapshot:
        return []
    official = snapshot.get("official") or {}
    out = []
    for entry in snapshot.get("entries") or []:
        if row["kind"] == "hub" and not _in_part(row, entry, official):
            continue
        endorsed = official.get(entry["name"]) or {}
        is_official = _endorses_release(row, snapshot, entry["name"], entry.get("digest", ""))
        source = entry.get("source") or {}
        item = {
            "id": f"{row['id']}/{entry['name']}", "source_id": row["id"], "source_name": row["name"],
            "trust": row["trust"], "name": entry["name"], "title": entry.get("title", ""), "description": entry.get("description", ""),
            "category": list(entry.get("category") or []), "version": entry.get("version", ""),
            "digest": entry.get("digest", ""), "official": is_official,
            "recommended": is_official and endorsed.get("recommended") is True,
            "owner": entry.get("owner", ""), "signals": _signals(snapshot, entry),
            "withdrawn": entry.get("withdrawn"), "type": entry.get("type", ""),
            "needs_keys": list(entry.get("needs_keys") or []), "permissions": list(entry.get("permissions") or []),
            "installed": None,
            "connects": connect_services(entry.get("connects")), "icon": _catalog_icon(entry.get("icon", "")),
        }
        if row.get("url"):
            item["repo_url"] = row["url"]
        if source.get("kind") == "git":
            item["author_repo"] = source.get("repo", "")
        if entry.get("install_specs"):
            item["install_specs"] = entry["install_specs"]
        if entry.get("updated_at"):
            item["updated_at"] = entry["updated_at"]
        out.append(item)
    return out


def registry_items(row: Dict[str, Any], query: str, *, limit: int = 16) -> List[Dict[str, Any]]:
    """ClawHub search hits as items (network; called on an explicit search only)."""
    from ouroboros.marketplace.clawhub import search
    from ouroboros.skill_loader import _sanitize_skill_name

    page = search(query, limit=limit, registry_url=row.get("url") or None, include_metadata=True, timeout_sec=15)
    out = []
    for summary in page.get("results") or []:
        if summary.is_plugin:
            continue
        out.append({
            "id": f"{row['id']}/{summary.slug}", "source_id": row["id"], "source_name": row["name"],
            "trust": row["trust"], "name": _sanitize_skill_name(summary.slug.split("/")[-1]),
            "registry_slug": summary.slug, "description": summary.summary or summary.description,
            "category": [], "version": summary.latest_version, "digest": "", "official": False,
            "recommended": False, "signals": None, "withdrawn": None, "type": "",
            "needs_keys": list(summary.requires_env or []), "permissions": [], "installed": None,
            "registry_stats": dict(summary.stats or {}), **summary.catalog_facts(),
        })
    return out


# --- installed state ---------------------------------------------------------------------------------

def installed_index(drive_root: pathlib.Path, *, repo_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Installed skill names with their location and install record (no payload hashing)."""
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import _skill_location_inventory

    out: Dict[str, Dict[str, Any]] = {}
    for candidate in _skill_location_inventory(pathlib.Path(drive_root), repo_path=repo_path):
        current = out.setdefault(candidate.name, {"name": candidate.name, "locations": []})
        current["locations"].append(candidate.location)
        current["skill_dir"] = str(candidate.skill_dir)
    for name, current in out.items():
        current["record"] = read_install_record(pathlib.Path(drive_root), name)
    return out


def record_matches_row(record: Dict[str, Any], row: Dict[str, Any]) -> bool:
    """Whether an install record came from this source row (same id, or the same repository)."""
    source = record.get("source") or {}
    if source.get("id") == row["id"]:
        return True
    return bool(source.get("url") and source.get("url") == row.get("url") and source.get("kind") == row["kind"])


def update_state(record: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Update facts for one install against the cached entry of its source."""
    if entry is None:
        return {"update_available": None, "withdrawn": None, "left_source": True}
    release = record.get("release") or {}
    digest = str(entry.get("digest") or "")
    newer = bool(digest and digest != release.get("digest") and digest != record.get("skipped_release_digest"))
    withdrawn = entry.get("withdrawn")
    if not withdrawn:
        withdrawn = next(({"reason": item.get("reason", ""), "version": item.get("version", "")}
                          for item in entry.get("withdrawn_releases", []) if item.get("digest") == release.get("digest")), None)
    return {"update_available": {"version": entry.get("version", ""), "digest": digest} if newer else None,
            "withdrawn": withdrawn, "left_source": False}


def _endorses_release(row: Dict[str, Any], snapshot: Dict[str, Any], name: str, digest: str) -> bool:
    """Official names exact bytes endorsed by an Official hub view or the bundled release."""
    eligible = row["kind"] == "builtin" or (row["kind"] == "hub" and row.get("part") == "official")
    endorsement = (snapshot.get("official") or {}).get(name) or {}
    return bool(eligible and digest and endorsement.get("digest") == digest and any(
        e.get("name") == name and e.get("digest") == digest for e in snapshot.get("entries") or []))


def _entry_for_record(record: Dict[str, Any], snapshots: Dict[str, tuple]) -> tuple[bool, Optional[Dict[str, Any]]]:
    """``(source known, entry)``: the entry of a record's source in the cached snapshots."""
    for row, snapshot in snapshots.values():
        if not record_matches_row(record, row):
            continue
        if snapshot is None:
            return False, None
        name = str(record.get("entry") or "")
        return True, next((entry for entry in snapshot.get("entries") or [] if entry.get("name") == name), None)
    return False, None


def install_listing(drive_root: pathlib.Path, loaded: Any, snapshots: Dict[str, tuple]) -> Dict[str, Any]:
    """``install``, ``category`` and ``removal`` fields of one ``GET /api/extensions`` row (passive);
    ``removal`` is the host's word (``skill_uninstall_state.removal_action``) on both return paths."""
    from ouroboros.skill_install_record import install_listing_fields, read_install_record
    from ouroboros.skill_uninstall_state import removal_action

    record = read_install_record(pathlib.Path(drive_root), loaded.name)
    category: List[str] = []
    removal = removal_action(loaded, record)
    from ouroboros.extension_registry_state import live_host_asset

    connector = _connector_facts(getattr(loaded, "manifest", None),
                                 icon_source=live_host_asset(loaded.name, getattr(loaded, "content_hash", ""), "ui/icon.svg"))
    if record is None:
        builtin = snapshots.get("builtin")
        if builtin and builtin[1] and getattr(loaded, "source", "") == "native":
            entry = next((e for e in builtin[1]["entries"] if e["name"] == loaded.name), None)
            category = list((entry or {}).get("category") or [])
        return {**connector, "install": None, "category": category, "removal": removal}
    known, entry = _entry_for_record(record, snapshots)
    base = str(record.get("base_digest") or "")
    current = installed_release_digest(loaded)
    matching = [(row, snap) for row, snap in snapshots.values() if snap is not None and record_matches_row(record, row)]
    official = any(_endorses_release(row, snap, str(record.get("entry") or ""), current) for row, snap in matching
                   ) if matching else bool(record.get("official"))
    official = bool(official and current and current == (record.get("release") or {}).get("digest"))
    facts = update_state(record, entry) if known else {"update_available": None, "withdrawn": None,
                                                         "left_source": False}
    category = list((entry or {}).get("category") or [])
    source = record.get("source") or {}
    return {**connector, "category": category, "removal": removal, "install": {
        "source": {key: source[key] for key in ("id", "name", "kind", "part") if source.get(key)},
        "official": official, "release": dict(record.get("release") or {}),
        "basis": record.get("basis"), "base_known": bool(base),
        "modified": (current != base) if base and current else None,
        "turn_on_when_ready": bool(record.get("turn_on_when_ready")),
        "installed_by": record.get("installed_by", ""), "installed_at": record.get("installed_at", ""),
        **{key: str(record[key]) for key in ("reason", "task_id") if record.get(key)},
        **facts,
        **install_listing_fields(record),
    }}


def listing_snapshots(drive_root: pathlib.Path) -> Dict[str, tuple]:
    """The connected sources' cached catalogs for one ``GET /api/extensions`` (never raises)."""
    try:
        from ouroboros.skill_sources import load_sources

        return cached_snapshots(drive_root, load_sources())
    except Exception:
        log.warning("cached skill source catalogs unavailable for the listing", exc_info=True)
        return {}


def listing_install_fields(drive_root: pathlib.Path, loaded: Any, snapshots: Dict[str, tuple]) -> Dict[str, Any]:
    """``install_listing`` that degrades to no install facts instead of failing the listing."""
    try:
        return install_listing(drive_root, loaded, snapshots)
    except Exception:
        log.debug("install facts unavailable for %s", getattr(loaded, "name", ""), exc_info=True)
        return {"install": None, "category": [], "removal": ""}


def cached_snapshots(drive_root: pathlib.Path, rows: List[Dict[str, Any]], *,
                     repo_dir: Optional[pathlib.Path] = None) -> Dict[str, tuple]:
    """``{row id: (row, snapshot)}`` from local data only (no refresh is started)."""
    out: Dict[str, tuple] = {}
    for row in rows:
        if row["kind"] == "builtin":
            from ouroboros.config import REPO_DIR

            out[row["id"]] = (row, builtin_snapshot(pathlib.Path(repo_dir or REPO_DIR)))
        elif row["kind"] in {"hub", "git"}:
            out[row["id"]] = (row, read_snapshot(drive_root, row))
        elif row["kind"] == "local":
            out[row["id"]] = (row, None)  # scanned in place: its skills are installed where they live
    return out


def _join_installed(items: List[Dict[str, Any]], index: Dict[str, Dict[str, Any]],
                    rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    for item in items:
        current = index.get(item["name"])
        if current is None:
            continue
        record = current.get("record")
        row = rows_by_id.get(item["source_id"])
        if record and row and record_matches_row(record, row) and record.get("entry") == item["name"]:
            release = record.get("release") or {}
            facts = update_state(record, {"digest": item["digest"], "version": item["version"],
                                          "withdrawn": item["withdrawn"]})
            item["installed"] = {"state": "installed", "version": release.get("version", ""),
                                 "digest": release.get("digest", ""), "modified": None,
                                 "update_available": facts["update_available"] is not None}
        elif row and row["kind"] == "local" and "user_repo" in current["locations"]:
            item["installed"] = {"state": "installed", "version": item["version"], "digest": item["digest"],
                                 "modified": None, "update_available": False}
        else:
            source = (record or {}).get("source") or {}
            item["installed"] = {"state": "name_taken", "source_id": source.get("id", ""),
                                 "source_name": source.get("name", "") or ", ".join(current["locations"])}
        if item["installed"]["state"] == "installed" and current.get("skill_dir"):
            # Older Hub snapshots need no schema refresh to identify an installed connector.
            manifest = read_manifest(pathlib.Path(current["skill_dir"]))
            item["connects"] = connect_services(item.get("connects"), _connector_facts(manifest)["connects"])


def catalog_view(drive_root: pathlib.Path, *, rows: Optional[List[Dict[str, Any]]] = None, refresh: bool = False,
                 repo_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """``GET /api/skill-catalog``: every connected source's items from local data, plus source states."""
    from ouroboros.skill_sources import load_sources

    rows = rows if rows is not None else load_sources()
    items: List[Dict[str, Any]] = []
    categories: Dict[str, Dict[str, Any]] = {}
    statuses = []
    for row in rows:
        snapshot, status = snapshot_for(drive_root, row, repo_dir=repo_dir, refresh=refresh)
        items.extend(items_for(row, snapshot))
        for category in (snapshot or {}).get("categories") or []:
            categories.setdefault(category["id"], category)
        statuses.append({"id": row["id"], "name": row["name"], "kind": row["kind"], "trust": row["trust"],
                         "confirmed": row["confirmed"], **status})
    _join_installed(items, installed_index(drive_root), {row["id"]: row for row in rows})
    return {"items": items, "categories": list(categories.values()), "sources": statuses}


def find_item(drive_root: pathlib.Path, item_id: str, *, rows: Optional[List[Dict[str, Any]]] = None,
              repo_dir: Optional[pathlib.Path] = None) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """``(row, item, snapshot)`` for a catalog id from local data; ``LookupError`` when unknown."""
    from ouroboros.skill_sources import load_sources, source_by_id

    source_id, _, name = str(item_id or "").partition("/")
    rows = rows if rows is not None else load_sources()
    row = source_by_id(rows, source_id)
    if row is None or not name:
        raise LookupError(f"no connected source {source_id!r}")
    if row["kind"] == "registry":
        return row, {"id": item_id, "source_id": row["id"], "source_name": row["name"], "trust": row["trust"],
                     "name": "", "registry_slug": name, "digest": "", "version": "", "withdrawn": None,
                     "official": False}, {}
    snapshot, _status = snapshot_for(drive_root, row, repo_dir=repo_dir)
    item = next((it for it in items_for(row, snapshot) if it["name"] == name), None)
    if item is None:
        raise LookupError(f"{name!r} is not in {row['name']}")
    return row, item, snapshot or {}


def entry_payload_dir(drive_root: pathlib.Path, row: Dict[str, Any], snapshot: Dict[str, Any], name: str, *,
                      repo_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """The local folder holding an entry's files (fetching a pointer's author repository at its commit)."""
    entry = next((e for e in snapshot.get("entries") or [] if e.get("name") == name), None)
    if entry is None:
        raise LookupError(f"{name!r} is not in {row['name']}")
    source = entry.get("source") or {}
    if row["kind"] == "builtin":
        from ouroboros.config import REPO_DIR

        base = pathlib.Path(repo_dir or REPO_DIR) / "skills"
    elif row["kind"] == "local":
        base = pathlib.Path(row["path"])
    elif source.get("kind") == "git":
        from ouroboros.skill_source_git import checkout_at_commit

        base = checkout_at_commit(drive_root, source["repo"], source["commit"])
        return _inside(base, source.get("path", "")) or base
    else:
        base = clone_dir(drive_root, row)
        if source.get("kind") == "folder" and snapshot.get("path"):
            base = _inside(base, snapshot["path"]) or base
    folder = _inside(base, source.get("path", ""))
    if folder is None or not folder.is_dir():
        raise LookupError(f"{name!r} has no files in {row['name']}")
    return folder


def item_detail(drive_root: pathlib.Path, item_id: str, *, rows: Optional[List[Dict[str, Any]]] = None,
                repo_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """``GET /api/skill-catalog/item/{source_id}/{name}``: the detail sheet data."""
    row, item, snapshot = find_item(drive_root, item_id, rows=rows, repo_dir=repo_dir)
    _join_installed([item], installed_index(drive_root), {row["id"]: row})
    about, about_error = "", ""
    manifest = None
    try:
        folder = entry_payload_dir(drive_root, row, snapshot, item["name"], repo_dir=repo_dir)
        manifest = read_manifest(folder)
        about = str(getattr(manifest, "body", "") or "")[:_MAX_ABOUT_CHARS]
    except Exception as exc:
        about_error = str(exc)[:300]
    facts = _manifest_facts(manifest, folder) if manifest is not None else {
        "type": item.get("type", ""), "needs_keys": item.get("needs_keys", []), "permissions": item.get("permissions", [])}
    uses = [{"kind": "permission", "id": perm, "label": _PERMISSION_WORDS.get(perm, perm)}
            for perm in facts["permissions"]]
    uses += [{"kind": "key", "id": key, "label": f"Uses your {key}"} for key in facts["needs_keys"]]
    signals = item.get("signals") or {}
    return {"item": {**item, **facts}, "about": about, "about_error": about_error, "uses": uses,
            "versions": [{"version": item["version"], "digest": item["digest"], "current": True}],
            "links": {key: value for key, value in {"repo": item.get("repo_url"), "author_repo": item.get("author_repo"),
                                                    "discussion": signals.get("discussion_url")}.items() if value}}


def probe_source(drive_root: pathlib.Path, *, url: str = "", path: str = "") -> Dict[str, Any]:
    """What an address or folder holds, read once into a scratch folder (nothing is saved)."""
    from ouroboros.force_remove import force_rmtree
    from ouroboros.skill_source_git import SkillSourceFetchError, fetch_tree, scratch_dir
    from ouroboros.skill_sources import SkillSourceError, repository_owner, validate_git_url

    if path:
        root = pathlib.Path(path).expanduser()
        if not root.is_absolute() or not root.is_dir():
            return {"error": "Choose an existing folder on this computer (an absolute path)."}
        snapshot = parse_plain_tree(root)
        return {"kind": "local", "suggested_name": root.name, "found": [e["name"] for e in snapshot["entries"]]}
    try:
        url = validate_git_url(url)
    except SkillSourceError as exc:
        return {"error": str(exc)}
    warning = {"warning": "not_encrypted"} if url.startswith("http://") else {}
    scratch = scratch_dir(drive_root, "probe")
    try:
        fetch_tree(scratch / "repo", url, "")
        clone = scratch / "repo"
        stem = pathlib.PurePosixPath(url.rstrip("/")).stem or "skills"
        if (clone / "hub.json").is_file():
            snapshot = parse_hub_tree(clone)
            hub = snapshot.get("hub") or {}
            answer = {"kind": "hub", "suggested_name": str(hub.get("name") or stem)}
            owner = str(hub.get("maintainer") or "") or repository_owner(url)
        else:
            snapshot = parse_plain_tree(clone, repo_name=stem)
            answer, owner = {"kind": "git", "suggested_name": stem}, repository_owner(url)
        if owner:
            answer["owner"] = owner
        return {**answer, **warning, "found": [e["name"] for e in snapshot["entries"]]}
    except SkillSourceFetchError as exc:
        return {**warning, "error": f"This computer could not read {url}: {exc}"}
    finally:
        force_rmtree(scratch)  # the clone's git objects are read-only, which a plain rmtree leaves on Windows


__all__ = [
    "REFRESH_AFTER_SEC", "builtin_snapshot", "cached_snapshots", "catalog_view",
    "entry_payload_dir", "find_item", "install_listing", "installed_index", "installed_release_digest",
    "listing_install_fields", "listing_snapshots",
    "item_detail", "items_for", "parse_hub_tree", "parse_plain_tree", "probe_source", "read_manifest", "read_snapshot",
    "record_matches_row", "refresh_source", "registry_items", "snapshot_for", "tree_release_digest", "update_state",
]
