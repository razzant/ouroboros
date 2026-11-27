"""Catalogs of the connected skill sources, normalized into one item shape.

A git-hosted source is read from its cache clone (``skill_source_git``) at the row's folder (``path``;
'' is the top one): a folder with ``hub.json`` is a hub (``entries/<name>.json`` and hosted payloads
inside that folder, the maintainer's ``official.json``, ``categories.json`` and, for a top-level hub,
an optional ``signals`` branch); any other folder holds plain skills, the folders with ``SKILL.md`` or
``skill.json`` up to two levels down and in ``.agents/skills``. A name found twice keeps the first
folder and the other is a ``problem`` naming both paths, as is an unreadable ``.agents/skills``; source
states carry the problems. A refresh
parses the commit it fetched into a
snapshot beside the clone, so every reader sees one commit and no catalog reader ever waits for the network;
it dates each entry by the newest fetched commit touching its release (``updated_at``). An entry's files use
an immutable ``checkout_at_commit`` of the snapshot's commit, fetched locally from the shared clone so a
later refresh cannot change a folder already handed to a reader. A pointer entry's manifest lies in its author's
repository, fetched only when a detail sheet or an install reads it: its items carry no manifest facts.
The built-in source is the bundled ``skills/`` folder with ``skills/builtin.json``; a local folder is
scanned in place; a registry (ClawHub) has no list here: it answers a search, or one page when named alone.
A hub entry or a ``builtin.json`` row may carry the catalog's own words for people (``title``, a one-line
``pitch``, and ``needs``, the prerequisite in words): display data, kept out of the payload manifests whose
digests grants and endorsements name; an entry's ``title`` wins over its manifest's, whose names (its own title,
each place label) stay ``search_names``: the text search finds a skill by every name it is shown by.

The release digest of a tree is ``skill_identity.payload_digest`` (``tree_release_digest`` reads the
manifest once for it); an installed skill's digest is the loader's own content hash. Installed state,
updates and the listing fields of an installed skill are joined here from install records and the
cached snapshots, without network and without writing (the Skills list read stays passive).
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ouroboros.skill_identity import is_release_digest
from ouroboros.skill_source_git import (
    checkout_at_commit,
    clone_dir,
    fetch_signals,
    last_changed,
    read_meta,
    refresh_in_background,
    update_clone,
    view_cache_dir,
)
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "catalog.json"
SNAPSHOT_SCHEMA_VERSION = 1
REFRESH_AFTER_SEC = 600  # a cache older than this is refreshed in the background when a catalog is shown
REGISTRY_BROWSE_PAGE = 24  # one Browse page of a registry
_MAX_ABOUT_CHARS = 20_000
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".ouroboros_env", ".github"})
_NAME_BREAKS = re.compile(r"[\s_-]+")  # one break in a skill's title or name for the text search
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


def _name_form(text: Any) -> str:
    return _NAME_BREAKS.sub(" ", str(text or "").lower()).strip()


def _other_names(names: Iterable[Any], *shown: Any) -> List[str]:
    """``names`` that no ``shown`` name already carries, each once in its first spelling (space, ``_`` and ``-``
    alike, case aside), in order; blanks dropped."""
    seen = {_name_form(name) for name in shown} - {""}
    out: List[str] = []
    for name in names:
        text = str(name or "").strip()
        form = _name_form(text)
        if form and form not in seen:
            seen.add(form)
            out.append(text)
    return out


def _connector_facts(manifest: Any, *, icon_source: bytes | None = None) -> Dict[str, Any]:
    from ouroboros.extension_ui_validation import validate_contributions
    from ouroboros.icon_geometry import icon_spec

    clean, _ = validate_contributions(manifest) if manifest is not None else ([], [])
    places = [row for row in clean if row.get("provides") == "place"]
    raw = getattr(manifest, "raw_extra", {})
    icon = raw.get("icon", "")
    if not icon and places:
        icon = places[0].get("icon", "")
    services = [str(row.get("destination") or "")[len("vendor:"):].strip() for row in places
                if str(row.get("destination") or "").startswith("vendor:")]
    tab = getattr(manifest, "ui_tab", None)  # the widget it declares: its title names it on the Widgets page
    widget = str(tab.get("title") or "").strip() if isinstance(tab, dict) else ""
    # The names a card may print for it, in the order it prefers them: its own title, then each place label.
    names = _other_names([raw.get("title") if isinstance(raw.get("title"), str) else "",
                          *(row.get("label") for row in places)])
    title = names[0] if names else ""
    return {"connects": connect_services(services), "icon": icon_spec(icon, icon_source), "title": title,
            "widget_title": widget, "search_names": _other_names(names[1:], title, widget)}


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
        entry.update(connects=[], icon={})  # its manifest lies in the author's repository: no facts, not empty ones
    if "connects" in raw:
        entry["connects"] = connect_services(entry["connects"], raw["connects"])
    if not entry["icon"] and "icon" in raw:
        entry["icon"] = _catalog_icon(raw["icon"])
    return _with_catalog_words(entry, raw)


CATALOG_WORDS = ("title", "pitch", "needs", "mark")  # a catalog's words for people; a hub publication keeps them


def _with_catalog_words(entry: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    """``entry`` with the catalog's own optional words for people from ``raw`` (a hub entry, a ``builtin.json`` row):
    ``mark`` is ``needs`` in a few words («Gemini key»). A blank ``needs`` is kept: the catalog says the skill needs
    nothing, where an absent one says nothing at all. A catalog ``title`` replaces the one its manifest gives the
    skill as the name a card prints; that name stays one of its ``search_names``."""
    words = {key: raw[key].strip() for key in CATALOG_WORDS
             if isinstance(raw.get(key), str) and (raw[key].strip() or key == "needs")}
    if words.get("title") and entry.get("title"):
        entry["search_names"] = _other_names([entry["title"], *(entry.get("search_names") or [])],
                                             words["title"], entry.get("widget_title"), entry["name"])
    entry.update(words)
    return entry


def parse_hub_tree(root: pathlib.Path) -> Dict[str, Any]:
    """Snapshot of a hub folder (``hub.json`` present); hosted payload paths are relative to it."""
    hub = _read_json(root / "hub.json")
    hub = hub if isinstance(hub, dict) else {}
    entries: List[Dict[str, Any]] = []
    problems: List[str] = []
    entries_dir = root / "entries"
    for path in sorted(entries_dir.glob("*.json")) if entries_dir.is_dir() else []:
        try:
            entries.append(_hub_entry(_read_json(path), path.stem, root))
        except ValueError as exc:
            problems.append(f"entries/{path.name}: {exc}")
    official: Dict[str, Dict[str, Any]] = {}
    curation = _read_json(root / "official.json")
    for item in (curation.get("endorsements") if isinstance(curation, dict) else None) or []:
        if isinstance(item, dict) and item.get("name") and is_release_digest(item.get("digest")):
            official[str(item["name"])] = {"digest": str(item["digest"]), "recommended": item.get("recommended") is True}
    categories = _read_json(root / "categories.json")
    category_rows = [
        {"id": str(item.get("id")), "label": str(item.get("label") or item.get("id"))}
        for item in ((categories.get("categories") if isinstance(categories, dict) else None) or [])
        if isinstance(item, dict) and item.get("id")
    ]
    return {"format": "hub", "hub": {"name": str(hub.get("name") or ""), "maintainer": str(hub.get("maintainer") or "")},
            "entries": entries, "official": official, "categories": category_rows, "problems": problems}


def _skill_folders(root: pathlib.Path, problems: List[str]) -> Iterable[pathlib.Path]:
    from ouroboros.skill_sources import standard_skill_dirs

    def has_manifest(path: pathlib.Path) -> bool:
        return (path / "SKILL.md").is_file() or (path / "skill.json").is_file()

    def listed(folder: pathlib.Path) -> List[pathlib.Path]:  # unreadable: a named problem, never an empty folder
        try:
            return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError as exc:
            problems.append(f"{folder.relative_to(root).as_posix()}: could not be read ({exc.strerror or exc})")
            return []

    if has_manifest(root):
        yield root
        return
    for child in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if child.name in _SKIP_DIRS:
            continue
        nested = [child] if has_manifest(child) else sorted(
            p for p in child.iterdir() if p.is_dir() and not p.name.startswith(".") and has_manifest(p))
        yield from nested
    yield from standard_skill_dirs(root, listdir=listed)


def parse_plain_tree(root: pathlib.Path, *, repo_name: str = "") -> Dict[str, Any]:
    """Snapshot of a plain skills repository or folder: every folder with a manifest is a skill, the first
    folder of a name wins and a later one is a problem naming both paths."""
    from ouroboros.skill_loader import _sanitize_skill_name

    entries: List[Dict[str, Any]] = []
    problems: List[str] = []
    seen: Dict[str, str] = {}
    if not root.is_dir():
        return {"format": "plain", "entries": [], "categories": [], "official": {}, "problems": ["folder is missing"]}
    for folder in _skill_folders(root, problems):
        manifest = read_manifest(folder)
        single = (getattr(manifest, "name", "") or repo_name or root.name) if folder == root else folder.name
        name = _sanitize_skill_name(single)
        where = folder.relative_to(root).as_posix()
        if name in seen:
            problems.append(f"{where}: not offered, {seen[name]} already offers the name {name!r}")
            continue
        seen[name] = where
        entry: Dict[str, Any] = {"name": name, "description": str(getattr(manifest, "description", "") or ""),
                                 "version": str(getattr(manifest, "version", "") or ""), "category": [],
                                 "owner": "", "withdrawn": None, "source": {"kind": "folder", "path": where},
                                 **_manifest_facts(manifest, folder)}
        try:
            entry["digest"] = tree_release_digest(folder, manifest)
        except Exception as exc:
            entry["digest"] = ""
            problems.append(f"{where}: {exc}")
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
        rel = f"entries/{entry['name']}.json" if source.get("kind") == "git" else str(source.get("path") or ".")
        path = str(pathlib.PurePosixPath(str(snapshot.get("path") or "."), rel))
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
        path = base.relative_to(clone).as_posix() if base != clone else ""
        if (base / "hub.json").is_file():
            snapshot = parse_hub_tree(base)  # the signals branch belongs to a hub at the top of its repository
            snapshot["signals"] = None if path else _parse_signals(fetch_signals(clone, url=row["url"]))
        else:
            snapshot = parse_plain_tree(base, repo_name=pathlib.PurePosixPath(str(row.get("url") or "")).stem)
        snapshot["path"] = path
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
        _with_catalog_words(entry, mark)
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


def _problems(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    problems = [str(item) for item in (snapshot or {}).get("problems") or []]
    return {"problems": problems} if problems else {}


def snapshot_for(drive_root: pathlib.Path, row: Dict[str, Any], *, repo_dir: Optional[pathlib.Path] = None,
                 refresh: bool = False) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """``(snapshot, status)`` of one source from local data; may start a background refresh. The status names
    the snapshot's ``problems`` (what it could not offer, and why)."""
    kind = row["kind"]
    if kind == "builtin":
        from ouroboros.config import REPO_DIR

        return builtin_snapshot(pathlib.Path(repo_dir or REPO_DIR)), {"state": "ready"}
    if kind == "local":
        snapshot = parse_plain_tree(pathlib.Path(row["path"]))
        return snapshot, {"state": "ready", "last_read_at": utc_now_iso(), **_problems(snapshot)}
    if kind == "registry":
        return None, {"state": "search"}
    meta = read_meta(drive_root, row)
    snapshot = read_snapshot(drive_root, row)
    # Views of one repository and revision share its fetch: a view behind the last good one reads it again.
    behind = bool(snapshot and meta.get("commit") and not meta.get("error")
                  and snapshot.get("commit") != meta["commit"])
    if refresh or snapshot is None or _stale(meta) or behind:
        refresh_in_background(drive_root, row, refresh_source)
    from ouroboros.skill_source_git import is_refreshing

    status: Dict[str, Any] = {"state": "ready" if snapshot else ("reading" if is_refreshing(drive_root, row) or not meta
                                                                 else "error"),
                              "last_read_at": str((snapshot or {}).get("read_at") or "")}
    if meta.get("error"):
        status["error"] = str(meta["error"])
    if snapshot is not None and is_refreshing(drive_root, row):
        status["refreshing"] = True
    return snapshot, {**status, **_problems(snapshot)}


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


def _manifest_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The manifest facts of an item. A pointer's manifest lies in its author's repository at the pinned commit, which
    no catalog read fetches (``entry_payload_dir`` does, for a detail sheet or an install): its ``type``, ``needs_keys``
    and ``permissions`` are absent, never empty, whatever an older snapshot stored."""
    if (entry.get("source") or {}).get("kind") == "git":
        return {}
    return {"type": entry.get("type", ""), "needs_keys": list(entry.get("needs_keys") or []),
            "permissions": list(entry.get("permissions") or [])}


def items_for(row: Dict[str, Any], snapshot: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The §3.3 items one source contributes (installed state is joined by the caller)."""
    if not snapshot:
        return []
    from ouroboros.skill_sources import source_key

    official = snapshot.get("official") or {}
    key = source_key(row)
    out = []
    for entry in snapshot.get("entries") or []:
        if row["kind"] == "hub" and not _in_part(row, entry, official):
            continue
        endorsed = official.get(entry["name"]) or {}
        is_official = _endorses_release(row, snapshot, entry["name"], entry.get("digest", ""))
        source = entry.get("source") or {}
        item = {
            "id": f"{row['id']}/{entry['name']}", "source_id": row["id"], "source_name": row["name"],
            "trust": row["trust"], "name": entry["name"], "title": entry.get("title", ""),
            "widget_title": entry.get("widget_title", ""), "description": entry.get("description", ""),
            "pitch": entry.get("pitch", ""), "needs": entry.get("needs"), "mark": entry.get("mark", ""),
            "category": list(entry.get("category") or []), "version": entry.get("version", ""),
            "digest": entry.get("digest", ""), "official": is_official,
            "recommended": is_official and endorsed.get("recommended") is True,
            "owner": entry.get("owner", ""), "signals": _signals(snapshot, entry),
            "withdrawn": entry.get("withdrawn"), **_manifest_fields(entry),
            "installed": None, "source_key": key,
            "connects": connect_services(entry.get("connects")), "icon": _catalog_icon(entry.get("icon", "")),
        }
        names = _other_names(entry.get("search_names") or [], item["title"], item["widget_title"], item["name"])
        if names:  # an older snapshot has none until its next refresh
            item["search_names"] = names
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


def _registry_page_items(row: Dict[str, Any], page: Dict[str, Any]) -> List[Dict[str, Any]]:
    from ouroboros.skill_loader import _sanitize_skill_name
    from ouroboros.skill_sources import source_key

    key = source_key(row)
    out = []
    for summary in page.get("results") or []:
        if summary.is_plugin:
            continue
        # The registry's own name for people titles the card (its summary falls back to the slug when it has none);
        # the id, name and slug stay what install and search take.
        display = str(summary.display_name or "").strip()
        out.append({
            "id": f"{row['id']}/{summary.slug}", "source_id": row["id"], "source_name": row["name"],
            "trust": row["trust"], "name": _sanitize_skill_name(summary.slug.split("/")[-1]),
            **({"title": display} if display and display != summary.slug else {}),
            "registry_slug": summary.slug, "description": summary.summary or summary.description,
            "category": [], "version": summary.latest_version, "digest": "", "official": False,
            "recommended": False, "signals": None, "withdrawn": None, "type": "",
            "needs_keys": list(summary.requires_env or []), "permissions": [], "installed": None,
            "registry_stats": dict(summary.stats or {}), "source_key": key, **summary.catalog_facts(),
        })
    return out


def _names(item: Dict[str, Any]) -> List[str]:
    """The names an item is shown and found by: the title its card prints, its widget's title, the other full names
    it is known by (``search_names``) and its id."""
    return [str(name) for name in (item.get("title"), item.get("widget_title"), *(item.get("search_names") or []),
                                   item.get("name")) if name]


def matches_words(item: Dict[str, Any], query: str) -> bool:
    """The text search of the local catalogs (``GET /api/skill-catalog/search``, ``skill_search``): ``query`` is the
    item's catalog id, or every word of it occurs in one of the item's names or its description. In those names
    space, ``_`` and ``-`` are one break, so the title a card prints and its id typed as words find the same entry,
    and a lone ``-`` or ``_`` between words is no word of its own («Claudexor - Quotas»); a query of nothing else
    keeps its literal meaning."""
    if str(query or "").strip().lower() == str(item.get("id") or "").lower():
        return True
    names = _names(item)
    text = " ".join([*names, str(item.get("description") or "")]).lower()
    forms = [form for form in map(_name_form, names) if form]

    def found(word: str) -> bool:
        form = _name_form(word)
        return word in text or bool(form) and any(form in name for name in forms)

    words = str(query or "").lower().split()
    return all(found(word) for word in [word for word in words if _name_form(word)] or words)


def name_rank(item: Dict[str, Any], query: str) -> int:
    """How wholly ``query`` names ``item``, for the order of a capped answer: 0 it is the item's technical identity as
    typed (its name, catalog id or registry slug, case aside), 1 it is one of the names the item is shown by, its id
    spelled out included (space, ``_`` and ``-`` alike), 2 neither (its words matched)."""
    typed = str(query or "").strip().lower()
    if typed and typed in {str(item.get(key) or "").lower() for key in ("name", "id", "registry_slug")}:
        return 0
    wanted = _name_form(query)
    return 1 if wanted and wanted in {_name_form(name) for name in (*_names(item), item.get("registry_slug"))} else 2


def registry_items(row: Dict[str, Any], query: str, *, limit: int = 16) -> List[Dict[str, Any]]:
    """ClawHub search hits as items (network; called on an explicit search only). A query that begins with the
    local ``<source_id>/`` prefix of this row (case aside) is stripped of it for the registry call, so a pasted
    catalog id finds the hit; the rest of the slug, namespace included, is asked with its original case."""
    from ouroboros.marketplace.clawhub import search

    text = str(query or "")
    prefix = f"{row['id']}/"
    if text[:len(prefix)].lower() == prefix.lower():
        text = text[len(prefix):]
    return _registry_page_items(row, search(text, limit=limit, registry_url=row.get("url") or None,
                                              include_metadata=True, timeout_sec=15))


def registry_browse(row: Dict[str, Any], *, sort: str = "popular", cursor: str = "",
                    limit: int = REGISTRY_BROWSE_PAGE) -> Dict[str, Any]:
    """``{items, next_cursor}``: one page of a registry in its own order (network; only for a registry named
    explicitly with nothing to search). ``popular`` (most downloaded) is the default and an unknown order's."""
    from ouroboros.marketplace.clawhub import BROWSE_SORTS, search

    page = search("", limit=limit, sort=sort if sort in BROWSE_SORTS else "popular", cursor=cursor or None,
                  registry_url=row.get("url") or None, include_metadata=True, timeout_sec=15)
    return {"items": _registry_page_items(row, page), "next_cursor": str(page.get("next_cursor") or "")}


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
    """Whether an install record came from this source row.

    A repository record written since rows have views names its view (``path`` present, maybe empty) and matches
    only a row of the same kind reading that view of that address, whatever its id: a removed source's id comes
    back when its name is added again for another address or view. An older record matches its id or its
    repository (every view of it), as before; so do built-in, local and registry records."""
    from ouroboros.skill_sources import source_view

    source = record.get("source") or {}
    same_repository = bool(source.get("url") and source.get("url") == row.get("url")
                           and source.get("kind") == row["kind"])
    if source.get("kind") in {"hub", "git"} and "path" in source:
        return same_repository and source_view(source) == source_view(row)
    return source.get("id") == row["id"] or same_repository


def row_for_record(record: Dict[str, Any], rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The row an install record came from: of the rows it matches, its own id first."""
    own = (record.get("source") or {}).get("id")
    matching = [row for row in rows if record_matches_row(record, row)]
    return next((row for row in matching if row["id"] == own), matching[0] if matching else None)


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
    """``(source known, entry)``: the entry of a record's source in the cached snapshots (its own row first)."""
    own = (record.get("source") or {}).get("id")
    for row, snapshot in sorted(snapshots.values(), key=lambda pair: pair[0]["id"] != own):
        if not record_matches_row(record, row):
            continue
        if snapshot is None:
            return False, None
        name = str(record.get("entry") or "")
        return True, next((entry for entry in snapshot.get("entries") or [] if entry.get("name") == name), None)
    return False, None


def _presenting(record: Dict[str, Any], snapshots: Dict[str, tuple], fields: Dict[str, Any],
                key: Optional[str], current: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The recorded origin presents its copy. A pre-view record may use a unique cached release witness for
    display only; it still has no action key, and update selection is independent. When the recorded key and
    entry are known but every matching source's cache is cold, the detail address is still carried — no title
    or offer is invented from it, and the absent branch is never used as source proof."""
    from ouroboros.skill_sources import source_key, source_view

    legacy = (_unviewed_repo(fields) and record.get("converted_from") != ".ouroboroshub.json"
              and bool(current) and current == (record.get("release") or {}).get("digest"))
    if key is None and not legacy:
        return None, None
    own, name = fields.get("id"), str(record.get("entry") or "")
    witnesses: Dict[str, tuple] = {}
    cold: Dict[str, str] = {}
    for row, snapshot in sorted(snapshots.values(), key=lambda pair: pair[0]["id"] != own):
        row_key = source_key(row)
        if row_key is None or (row_key != key and not (
            legacy and row["kind"] == fields["kind"] and row.get("url") == fields["url"] and not source_view(row)[1]
        )):
            continue
        if snapshot is None:
            # Only the record's own repository row names a detail its cold cache will hold: a registry keeps no
            # catalog (its hits answer a search) and another row of the key may read another part of the hub.
            if not legacy and row_key == key and name and row["kind"] in {"hub", "git"} and row["id"] == own:
                cold[row["id"]] = name
            continue
        entry = next((each for each in (snapshot or {}).get("entries") or [] if each.get("name") == name), None)
        if entry is None or (legacy and entry.get("digest") != current):
            continue
        listed = row["kind"] != "hub" or _in_part(row, entry, snapshot.get("official") or {})
        if row_key not in witnesses or (listed and witnesses[row_key][1] is None):
            witnesses[row_key] = (entry, f"{row['id']}/{name}" if listed else None)
        if listed and not legacy:
            break
    if len(witnesses) == 1:
        return next(iter(witnesses.values()))
    if not witnesses and cold and key is not None and not legacy:
        rid = own if own in cold else next(iter(cold))
        return None, f"{rid}/{cold[rid]}"
    return None, None


def _installed_names(loaded: Any, connector: Dict[str, Any], entry: Optional[Dict[str, Any]],
                     catalog_id: Optional[str]) -> Dict[str, Any]:
    """The names of one ``GET /api/extensions`` row: those its manifest gives it (``title``, ``widget_title``, the
    other full names as ``search_names``), the ``catalog_id`` of its detail sheet (null for none) and the
    ``catalog_title`` of the entry that presents it; a card shows that title only where the skill has none of
    its own."""
    title = str((entry or {}).get("title") or "")
    names = _other_names(connector["search_names"], connector["title"], connector["widget_title"], title, loaded.name)
    return {**{key: value for key, value in connector.items() if key != "search_names"},
            **({"search_names": names} if names else {}), "catalog_id": catalog_id,
            **({"catalog_title": title} if title else {})}


def add_search_names(row: Dict[str, Any], *names: Any) -> None:
    """Adds to an Installed row's ``search_names`` each text of ``names`` it is not already shown or found by."""
    added = _other_names([*(row.get("search_names") or []), *(name for name in names if isinstance(name, str))],
                         row.get("title"), row.get("widget_title"), row.get("catalog_title"), row.get("name"))
    if added:
        row["search_names"] = added


def live_widget_titles() -> Dict[str, tuple]:
    """This process's published widget headers by name, with the folder and revision each came from (one registry
    read, ``extension_registry_state.published_widget_titles``; {} when unreadable): a process holding no publication
    of a skill knows only its declared names."""
    try:
        from ouroboros.extension_registry_state import published_widget_titles

        return published_widget_titles()
    except Exception:
        log.debug("published widget titles unavailable", exc_info=True)
        return {}


def _published_titles(live: Dict[str, tuple], name: str, skill_dir: Any, revision: str = "") -> List[str]:
    """The widget headers in ``live`` (``live_widget_titles``) of ``name``'s publication from exactly ``skill_dir``,
    and of exactly ``revision`` when one is named; [] for another folder's publication, another revision or none."""
    held = live.get(name)
    if not held or not skill_dir:
        return []
    try:
        directory = str(pathlib.Path(skill_dir).resolve())
    except OSError:
        return []
    return list(held[2]) if held[0] == directory and (not revision or held[1] == revision) else []


def install_listing(drive_root: pathlib.Path, loaded: Any, snapshots: Dict[str, tuple], *,
                    live: Optional[Dict[str, tuple]] = None,
                    local_walk: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """``install``, ``category`` and ``removal`` fields of one ``GET /api/extensions`` row with its names (passive);
    ``removal`` is the host's word (``skill_uninstall_state.removal_action``) on both return paths. The headers its
    widgets are published under from this very folder (``live``, read once per listing, else now) join its
    ``search_names``: a publication another folder holds under the same name lends this row none."""
    from ouroboros.skill_install_record import install_listing_fields, read_install_record
    from ouroboros.skill_sources import source_key
    from ouroboros.skill_uninstall_state import removal_action

    record = read_install_record(pathlib.Path(drive_root), loaded.name)
    removal = removal_action(loaded, record)
    from ouroboros.extension_registry_state import live_host_asset

    connector = _connector_facts(getattr(loaded, "manifest", None),
                                 icon_source=live_host_asset(loaded.name, getattr(loaded, "content_hash", ""), "ui/icon.svg"))
    published = _published_titles(live_widget_titles() if live is None else live, loaded.name,
                                  getattr(loaded, "skill_dir", None))
    connector["search_names"] = [*connector["search_names"], *published]
    if record is None:
        builtin, entry = snapshots.get("builtin"), None
        if builtin and builtin[1] and getattr(loaded, "source", "") == "native":
            entry = next((e for e in builtin[1]["entries"] if e["name"] == loaded.name), None)
        listed = f"{builtin[0]['id']}/{entry['name']}" if entry else None
        key: Optional[str] = source_key({"kind": "builtin"}) if entry is not None else None
        if entry is None:
            # A recordless copy in the owner's own folder: the first connected local row whose own snapshot
            # enumerates exactly that package folder presents it under the entry's own name (a root package's
            # manifest name may differ from its folder's); an unrelated root's namesake does not.
            walk: Dict[str, List[str]] = {} if local_walk is None else local_walk  # one walk per folder per listing
            for row, snap in snapshots.values():
                if row["kind"] == "local" and _same_local_copy(row, getattr(loaded, "skill_dir", ""), snap,
                                                                loaded.name, walk):
                    key = source_key(row)
                    listed = f"{row['id']}/{_local_entry_name(row, snap, getattr(loaded, 'skill_dir', ''), loaded.name)}"
                    break
        return {**_installed_names(loaded, connector, entry, listed), "install": None, "source_key": key,
                "category": list((entry or {}).get("category") or []), "removal": removal}
    known, entry = _entry_for_record(record, snapshots)
    base = str(record.get("base_digest") or "")
    current = installed_release_digest(loaded)
    matching = [(row, snap) for row, snap in snapshots.values() if snap is not None and record_matches_row(record, row)]
    official = any(_endorses_release(row, snap, str(record.get("entry") or ""), current) for row, snap in matching
                   ) if matching else bool(record.get("official"))
    official = bool(official and current and current == (record.get("release") or {}).get("digest"))
    facts = update_state(record, entry) if known else {"update_available": None, "withdrawn": None,
                                                         "left_source": False}
    source = record.get("source") or {}
    # The connected source the record matches now, whatever id it was installed under (none: no id); the name,
    # kind and part stay as recorded. The words the copy is shown by come from a row reading its own address.
    matched = row_for_record(record, [row for row, _snapshot in snapshots.values()])
    fields = _record_source_fields(record, getattr(loaded, "skill_dir", ""), local_name=loaded.name,
                                  drive_root=drive_root)
    key = _record_key(record, fields)
    shown, listed = _presenting(record, snapshots, fields, key, current)
    out: Dict[str, Any] = {**_installed_names(loaded, connector, shown, listed),
                           "source_key": key,
                           "category": list((shown or {}).get("category") or []), "removal": removal, "install": {
        "source": {**({"id": matched["id"]} if matched else {}),
                   **{key: source[key] for key in ("name", "kind", "part") if source.get(key)}},
        "official": official, "release": dict(record.get("release") or {}),
        "basis": record.get("basis"), "base_known": bool(base),
        "modified": (current != base) if base and current else None,
        "turn_on_when_ready": bool(record.get("turn_on_when_ready")),
        "installed_by": record.get("installed_by", ""), "installed_at": record.get("installed_at", ""),
        **{key: str(record[key]) for key in ("reason", "task_id") if record.get(key)},
        **facts,
        **install_listing_fields(record),
    }}
    if source.get("kind") == "registry" and record.get("entry"):
        slug = out["registry_slug"] = str(record["entry"])
        own = snapshots.get(source.get("id"))
        if key is not None and own and own[0].get("confirmed") and source_key(own[0]) == key:
            add_search_names(out, f"{own[0]['id']}/{slug}")
    return out


def listing_snapshots(drive_root: pathlib.Path) -> Dict[str, tuple]:
    """The connected sources' cached catalogs for one ``GET /api/extensions`` (never raises)."""
    try:
        from ouroboros.skill_sources import load_sources

        return cached_snapshots(drive_root, load_sources())
    except Exception:
        log.warning("cached skill source catalogs unavailable for the listing", exc_info=True)
        return {}


def listing_install_fields(drive_root: pathlib.Path, loaded: Any, snapshots: Dict[str, tuple], *,
                           live: Optional[Dict[str, tuple]] = None,
                           local_walk: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """``install_listing`` that degrades to no install facts instead of failing the listing."""
    try:
        return install_listing(drive_root, loaded, snapshots, live=live, local_walk=local_walk)
    except Exception:
        log.debug("install facts unavailable for %s", getattr(loaded, "name", ""), exc_info=True)
        return {"install": None, "category": [], "removal": "", "catalog_id": None, "source_key": None}


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
        else:  # a folder is scanned in place and a registry only searched: neither keeps a catalog
            out[row["id"]] = (row, None)
    return out


def _lend_installed_names(item: Dict[str, Any], skill_dir: pathlib.Path, manifest: Any,
                          facts: Dict[str, Any], *, digest: str = "") -> None:
    """A pointer's item knows no manifest names (they lie in its author's repository): the installed copy of exactly
    its release gives them, as search names only, while its bytes are the release's (``digest``: the caller's
    memoized current digest of that folder, else the loader's own digest is computed here; «Modified» compares
    it too). The card keeps the catalog's title, an empty one included."""
    widget = item.get("widget_title") or facts["widget_title"]
    names = _other_names([facts["title"], *facts["search_names"], *(item.get("search_names") or [])],
                         item["title"], widget, item["name"])
    if widget == item.get("widget_title") and names == list(item.get("search_names") or []):
        return  # nothing to add: no digest to read
    try:
        current = digest or tree_release_digest(skill_dir, manifest)
        if current != item["digest"]:
            return
    except Exception:
        return
    item["widget_title"] = widget
    if names:
        item["search_names"] = names


def _local_entry_dir(row: Dict[str, Any], snapshot: Any, name: str) -> str:
    """The resolved package folder of ``name`` in a local source's snapshot — its entry's own ``source.path``
    under the row's root (the root itself when it holds the manifest) — without walking the tree."""
    from ouroboros.skill_sources import resolved_folder

    if not str(row.get("path") or "").strip():
        return ""  # a row without a folder holds no package: never one under the current directory
    root = pathlib.Path(str(row["path"])).expanduser()
    for entry in (snapshot or {}).get("entries") or []:
        if entry.get("name") == name:
            where = str((entry.get("source") or {}).get("path") or "")
            return resolved_folder(root / where) if where else resolved_folder(root)
    return ""


def _local_entry_name(row: Dict[str, Any], snapshot: Any, skill_dir: Any, fallback: str) -> str:
    """The catalog name of the entry whose package folder is exactly ``skill_dir`` — a single-package root's
    manifest name may differ from its folder's basename — else ``fallback``."""
    from ouroboros.skill_sources import resolved_folder

    target = resolved_folder(skill_dir)
    if not target:
        return fallback
    for entry in (snapshot or {}).get("entries") or []:
        name = str(entry.get("name") or "")
        if name and _local_entry_dir(row, snapshot, name) == target:
            return name
    root = pathlib.Path(str(row.get("path") or "")).expanduser()
    if (str(row.get("path") or "").strip() and resolved_folder(root) == target
            and ((root / "SKILL.md").is_file() or (root / "skill.json").is_file())):
        # A directly attached package without a snapshot (a listing reads none): named as parse_plain_tree names it.
        from ouroboros.skill_loader import _sanitize_skill_name

        return _sanitize_skill_name(getattr(read_manifest(root), "name", "") or root.name)
    return fallback


def _local_skill_dirs(root: pathlib.Path) -> List[str]:
    """The resolved skill folders a local source's snapshot enumerates (the same walk ``parse_plain_tree``
    uses: a manifest at the root itself, child manifests up to two levels down, ``.agents/skills``); [] when
    the folder cannot be read."""
    from ouroboros.skill_sources import resolved_folder

    try:
        if not root.is_dir():
            return []
        problems: List[str] = []
        return [resolved_folder(folder) for folder in _skill_folders(root, problems) if resolved_folder(folder)]
    except OSError:
        return []


def _same_local_copy(row: Dict[str, Any], skill_dir: Any, snapshot: Any = None, name: str = "",
                     walk_cache: Optional[Dict[str, List[str]]] = None) -> bool:
    """Whether the installed copy at ``skill_dir`` is one of the skill folders ``row``'s own snapshot
    enumerates — its entry's known package path when the snapshot has it (no walk), else one walk per row —
    never a mere ancestor of it, and never a namesake in an unrelated root."""
    from ouroboros.skill_sources import resolved_folder

    target = resolved_folder(skill_dir)
    if not target:
        return False
    if snapshot is not None and name:
        entry_dir = _local_entry_dir(row, snapshot, name)
        if entry_dir:
            return entry_dir == target
    root = str(row.get("path") or "").strip()
    if not root:
        return False  # no folder: nothing is walked, least of all the current directory
    if walk_cache is not None:
        if root not in walk_cache:
            walk_cache[root] = _local_skill_dirs(pathlib.Path(root).expanduser())
        return target in walk_cache[root]
    return target in _local_skill_dirs(pathlib.Path(root).expanduser())


def _unviewed_repo(fields: Dict[str, Any]) -> bool:
    return bool(fields.get("kind") in {"hub", "git"} and fields.get("url")
                and "path" not in fields and not fields.get("ref") and not fields.get("commit"))


def _record_source_fields(record: Dict[str, Any], skill_dir: Any, *,
                          local_name: str = "", drive_root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """The record's source fields, with the branch an ``.ouroboroshub.json`` sidecar's catalog URL named
    recovered through ``repo_from_catalog_url`` — for presentation only; the record itself is never migrated.
    A registry record the default row installed without naming a URL takes its origin from the durable
    ClawHub provenance the real installer recorded under the copy's actual local name (``local_name``);
    a legacy converted record keeps no origin and stays unknown, and with no local name nothing is lent."""
    from ouroboros.skill_sources import CLAWHUB_ID, repo_from_catalog_url

    fields = dict(record.get("source") or {})
    if (fields.get("kind") == "registry" and not str(fields.get("url") or "").strip()
            and record.get("converted_from") != ".clawhub.json" and drive_root is not None
            and str(local_name or "").strip()):
        from ouroboros.marketplace.provenance import read_provenance

        # The record's ``entry`` is the full registry slug, but the installer wrote payload and
        # provenance under the sanitized local name (``acme/notes`` lands as ``acme__notes``): read
        # the provenance under the actual local name this copy is loaded under, never the raw slug
        # and never a namesake record. The recorded source is the ClawHub provider label (`clawhub`),
        # not the user-assigned Sources row id: a custom registry row (any id) with the same provider
        # provenance lends the URL just as the default row does. The slug and a recorded sanitized
        # name must still name this record before the URL is lent.
        name = str(local_name).strip()
        prov = read_provenance(pathlib.Path(drive_root), name) or {}
        url = str(prov.get("registry_url") or "").strip()
        provider = str(prov.get("source") or "").strip()
        if (url and provider in ("clawhub", CLAWHUB_ID)
                and str(prov.get("slug") or "").strip() == str(record.get("entry") or "").strip()
                and (not str(prov.get("sanitized_name") or "").strip()
                     or str(prov["sanitized_name"]).strip() == name)):
            fields = {**fields, "url": url}
    if fields.get("kind") == "hub" and "path" not in fields and not fields.get("ref") and not fields.get("commit"):
        sidecar = read_json_dict(pathlib.Path(skill_dir) / ".ouroboroshub.json") if skill_dir else None
        if sidecar:
            repo, ref = repo_from_catalog_url(str(sidecar.get("catalog_url") or ""))
            if (sidecar.get("source") != "ouroboroshub" or sidecar.get("slug") != record.get("entry")
                    or (repo and fields.get("url") and repo != fields["url"])):
                return {}  # conflicting provenance lends neither names nor an action key
            if repo and ref:
                fields = {**fields, "url": repo, "ref": ref}
    return fields


def _record_key(record: Dict[str, Any], fields: Dict[str, Any]) -> Optional[str]:
    """The source key of an install record's ORIGINAL origin (never a legacy update matcher's current row),
    over the same recovered presentation fields."""
    from ouroboros.skill_sources import source_key

    if _unviewed_repo(fields):
        return None  # absent pre-view metadata did not explicitly name the default branch
    if fields.get("kind") == "registry" and (
        not str(fields.get("url") or "").strip() or record.get("converted_from") == ".clawhub.json"
    ):
        return None  # legacy conversion copied the configured registry row, not a proven historic origin
    return source_key(fields)


def _join_installed(items: List[Dict[str, Any]], index: Dict[str, Dict[str, Any]],
                    rows_by_id: Dict[str, Dict[str, Any]],
                    snapshots: Optional[Dict[str, Any]] = None,
                    *, drive_root: Optional[pathlib.Path] = None) -> None:
    from ouroboros.skill_sources import source_key

    live: Optional[Dict[str, tuple]] = None  # read once, at the first item an installed copy of its release presents
    local_walk: Dict[str, List[str]] = {}  # one walk per local row, only where the snapshot has no entry path
    folder_facts: Dict[str, tuple] = {}  # folder -> (current release digest, manifest), computed once per folder
    by_folder: Optional[Dict[str, Dict[str, Any]]] = None  # the sole user_repo copy loaded from each folder
    for item in items:
        current = index.get(item["name"])
        row = rows_by_id.get(item["source_id"])
        folder = (_local_entry_dir(row, (snapshots or {}).get(row["id"]), item["name"])
                  if row and row["kind"] == "local" else "")
        if folder and (current or {}).get("skill_dir") != folder:
            # The loader names a directly attached package by its folder, the catalog by its manifest: the copy
            # is the one loaded from this entry's very folder, never a namesake elsewhere.
            if by_folder is None:
                by_folder = {str(each.get("skill_dir")): each for each in index.values()
                             if each.get("locations") == ["user_repo"] and each.get("skill_dir")}
            current = by_folder.get(folder, current)
        if current is None:
            continue
        record = current.get("record")
        skill_dir = current.get("skill_dir")
        key = _record_key(record, _record_source_fields(record, skill_dir, local_name=str(current.get("name") or ""),
                                                        drive_root=drive_root)) if record else None
        this_release = False
        if record and key is not None and key == item["source_key"] and record.get("entry") == item["name"]:
            release = record.get("release") or {}
            this_release = bool(item["digest"]) and release.get("digest") == item["digest"]
            facts = update_state(record, {"digest": item["digest"], "version": item["version"],
                                          "withdrawn": item["withdrawn"]})
            item["installed"] = {"state": "installed", "version": release.get("version", ""),
                                 "digest": release.get("digest", ""), "modified": None,
                                 "update_available": facts["update_available"] is not None,
                                 "source_key": key}
        elif (row and row["kind"] == "local" and "user_repo" in current["locations"]
              and _same_local_copy(row, skill_dir, (snapshots or {}).get(row["id"]), item["name"], local_walk)):
            # A connected local folder is read in place: the copy is the package folder its own snapshot
            # enumerates for this name, and no reload is needed.
            this_release = True
            item["installed"] = {"state": "installed", "version": item["version"], "digest": item["digest"],
                                 "modified": None, "update_available": False, "source_key": source_key(row)}
        else:
            source = (record or {}).get("source") or {}
            item["installed"] = {"state": "name_taken", "source_id": source.get("id", ""),
                                 "source_name": source.get("name", "") or ", ".join(current["locations"]),
                                 "source_key": key}
        if item["installed"]["state"] == "installed" and this_release and skill_dir:
            # Only the installed copy whose current bytes are exactly this release lends its manifest's
            # connector facts and its live widget headers; a copy of another release or edited since borrows
            # nothing. Each folder's digest is read once and memoized, never re-scanned per row.
            folder = str(skill_dir)
            if folder not in folder_facts:
                manifest = read_manifest(pathlib.Path(folder))
                try:
                    digest = tree_release_digest(pathlib.Path(folder), manifest) if manifest is not None else ""
                except Exception:
                    digest = ""
                folder_facts[folder] = (digest, manifest)
            current_digest, manifest = folder_facts[folder]
            if current_digest and current_digest == item.get("digest"):
                declared = _connector_facts(manifest)
                item["connects"] = connect_services(item.get("connects"), declared["connects"])
                if item.get("author_repo"):
                    _lend_installed_names(item, pathlib.Path(folder), manifest, declared, digest=current_digest)
                if len(current["locations"]) == 1:
                    # The headers its widgets are published under, while that one folder's live publication is
                    # exactly this release: search names only, as its Installed card is found by them.
                    live = live_widget_titles() if live is None else live
                    titles = _published_titles(live, str(current.get("name") or item["name"]), current["skill_dir"],
                                               item["digest"])
                    names = _other_names([*(item.get("search_names") or []), *titles],
                                         item["title"], item.get("widget_title"), item["name"])
                    if names:
                        item["search_names"] = names


def catalog_view(drive_root: pathlib.Path, *, rows: Optional[List[Dict[str, Any]]] = None, refresh: bool = False,
                 repo_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """``GET /api/skill-catalog``: every connected source's items from local data, plus source states."""
    from ouroboros.skill_sources import load_sources

    rows = rows if rows is not None else load_sources()
    items: List[Dict[str, Any]] = []
    categories: Dict[str, Dict[str, Any]] = {}
    statuses = []
    snaps: Dict[str, Any] = {}
    for row in rows:
        snapshot, status = snapshot_for(drive_root, row, repo_dir=repo_dir, refresh=refresh)
        snaps[row["id"]] = snapshot
        items.extend(items_for(row, snapshot))
        for category in (snapshot or {}).get("categories") or []:
            categories.setdefault(category["id"], category)
        statuses.append({"id": row["id"], "name": row["name"], "kind": row["kind"], "trust": row["trust"],
                         "confirmed": row["confirmed"], **status})
    _join_installed(items, installed_index(drive_root), {row["id"]: row for row in rows}, snaps, drive_root=drive_root)
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
        from ouroboros.skill_sources import source_key

        return row, {"id": item_id, "source_id": row["id"], "source_name": row["name"], "trust": row["trust"],
                     "name": "", "registry_slug": name, "digest": "", "version": "", "withdrawn": None,
                     "official": False, "source_key": source_key(row)}, {}
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
        base = checkout_at_commit(drive_root, source["repo"], source["commit"])
        return _inside(base, source.get("path", "")) or base
    else:  # Consumers keep this path after return, so even a currently matching shared clone is not stable.
        base = checkout_at_commit(drive_root, clone_dir(drive_root, row).resolve().as_uri(), str(snapshot["commit"]))
        if snapshot.get("path"):
            base = _inside(base, snapshot["path"]) or base
    folder = _inside(base, source.get("path", ""))
    if folder is None or not folder.is_dir():
        raise LookupError(f"{name!r} has no files in {row['name']}")
    return folder


def item_detail(drive_root: pathlib.Path, item_id: str, *, rows: Optional[List[Dict[str, Any]]] = None,
                repo_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """``GET /api/skill-catalog/item/{source_id}/{name}``: the detail sheet data."""
    row, item, snapshot = find_item(drive_root, item_id, rows=rows, repo_dir=repo_dir)
    _join_installed([item], installed_index(drive_root), {row["id"]: row}, {row["id"]: snapshot}, drive_root=drive_root)
    about, about_error = "", ""
    manifest = None
    try:
        folder = entry_payload_dir(drive_root, row, snapshot, item["name"], repo_dir=repo_dir)
        manifest = read_manifest(folder)
        about = str(getattr(manifest, "body", "") or "")[:_MAX_ABOUT_CHARS]
    except Exception as exc:
        about_error = str(exc)[:300]
    # The release's own manifest says what it can use (a pointer's item lacks it), and the services and icon its catalog
    # does not name; the item keeps the catalog's names, so the detail names the entry as its card does, with the same
    # fallback to the id for no title.
    released = _manifest_facts(manifest, folder) if manifest is not None else item
    facts = {"type": released.get("type", ""), "needs_keys": released.get("needs_keys", []),
             "permissions": released.get("permissions", []),
             "connects": item.get("connects") or released.get("connects", []),
             "icon": item.get("icon") or released.get("icon", {})}
    uses = [{"kind": "permission", "id": perm, "label": _PERMISSION_WORDS.get(perm, perm)}
            for perm in facts["permissions"]]
    uses += [{"kind": "key", "id": key, "label": f"Uses your {key}"} for key in facts["needs_keys"]]
    signals = item.get("signals") or {}
    return {"item": {**item, **facts}, "about": about, "about_error": about_error, "uses": uses,
            "versions": [{"version": item["version"], "digest": item["digest"], "current": True}],
            "links": {key: value for key, value in {"repo": item.get("repo_url"), "author_repo": item.get("author_repo"),
                                                    "discussion": signals.get("discussion_url")}.items() if value}}


def probe_source(drive_root: pathlib.Path, *, url: str = "", path: str = "", revision: str = "",
                 subfolder: str = "") -> Dict[str, Any]:
    """What an address (at ``revision``, in ``subfolder``) or a folder holds, read once into a scratch folder
    (nothing is saved). A repository's answer names the exact ``commit`` it read."""
    from ouroboros.force_remove import force_rmtree
    from ouroboros.skill_source_git import SkillSourceFetchError, fetch_tree, scratch_dir
    from ouroboros.skill_sources import SkillSourceError, repository_owner, validate_git_url, view_fields

    def found(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {"found": [e["name"] for e in snapshot["entries"]], **_problems(snapshot)}

    if path:
        root = pathlib.Path(path).expanduser()
        if str(revision or "").strip() or str(subfolder or "").strip():
            return {"error": "A folder on this computer is read as it is: leave the revision and subfolder empty."}
        if not root.is_absolute() or not root.is_dir():
            return {"error": "Choose an existing folder on this computer (an absolute path), or a repository "
                             "address (https://, ssh://, git@host:path or file://)."}
        return {"kind": "local", "suggested_name": root.name, **found(parse_plain_tree(root))}
    try:
        url, view = validate_git_url(url), view_fields(revision, subfolder)
    except SkillSourceError as exc:
        return {"error": str(exc)}
    warning = {"warning": "not_encrypted"} if url.startswith("http://") else {}
    scratch = scratch_dir(drive_root, "probe")
    try:
        commit = fetch_tree(scratch / "repo", url, view.get("commit") or view.get("ref") or "")
        clone = scratch / "repo"
        base = _inside(clone, view.get("path", ""))
        if base is None or not base.is_dir():
            return {**warning, "error": f"{url} has no folder {view.get('path')} at "
                                       f"{view.get('ref') or commit[:12]}."}
        stem = pathlib.PurePosixPath(url.rstrip("/")).stem or "skills"
        if (base / "hub.json").is_file():
            snapshot = parse_hub_tree(base)
            hub = snapshot.get("hub") or {}
            answer = {"kind": "hub", "suggested_name": str(hub.get("name") or stem)}
            owner = str(hub.get("maintainer") or "") or repository_owner(url)
        else:
            snapshot = parse_plain_tree(base, repo_name=stem)
            answer, owner = {"kind": "git", "suggested_name": stem}, repository_owner(url)
        if owner:
            answer["owner"] = owner
        return {**answer, **warning, "commit": commit, **found(snapshot)}
    except SkillSourceFetchError as exc:
        return {**warning, "error": f"This computer could not read {url}: {exc}"}
    finally:
        force_rmtree(scratch)  # the clone's git objects are read-only, which a plain rmtree leaves on Windows


__all__ = [
    "CATALOG_WORDS", "REFRESH_AFTER_SEC", "add_search_names", "builtin_snapshot", "cached_snapshots", "catalog_view",
    "entry_payload_dir", "find_item", "install_listing", "installed_index", "installed_release_digest",
    "listing_install_fields", "listing_snapshots", "live_widget_titles", "matches_words", "name_rank",
    "item_detail", "items_for", "parse_hub_tree", "parse_plain_tree", "probe_source", "read_manifest", "read_snapshot",
    "record_matches_row", "refresh_source", "registry_browse", "registry_items", "row_for_record", "snapshot_for", "tree_release_digest",
    "update_state",
]
