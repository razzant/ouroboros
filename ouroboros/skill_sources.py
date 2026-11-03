"""Skill sources: the owner's list of places skills are discovered, installed and updated from.

One owner-editable settings key, ``OUROBOROS_SKILL_SOURCES`` (a JSON list), names every source with a
trust choice: ``trusted`` installs without review, ``review`` reviews each skill. While the key is absent
the list is DERIVED from the older settings and never written: the built-in skills, the public hub as two
rows over one repository (its maintainer's Official endorsements and the community entries), ClawHub and
the optional local folder. The first change materializes the list; My folder always follows
``OUROBOROS_SKILLS_REPO_PATH``, and removing it clears that setting. Changing a row's trust never rewrites
an existing install: the basis an install was made under lives in its install record. Adding a source is
an owner act (the UI, or Ouroboros on the owner's instruction, or Cyber Pro); this module only validates
and persists the list. An enterprise build ships its own list as settings data; a row it marks
``managed: true`` keeps the mark through edits, so the Sources page can say who set it. The mark locks
nothing, and the source changes made from the Sources page or by Ouroboros never set it. A build may
also offer ``OUROBOROS_SKILL_SOURCE_SUGGESTIONS``: addresses the owner can choose in Add source. A
suggestion connects nothing and carries no trust; the open-source build offers none.

A repository row reads one view of its repository: a revision (``ref``, a branch or tag; or a pinned
``commit``; neither follows the default branch) and a folder (``path``; '' is the top folder). Install
records name that view, so two views of one repository stay two sources.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import stat
import urllib.parse
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

SOURCES_SETTING = "OUROBOROS_SKILL_SOURCES"
SUGGESTIONS_SETTING = "OUROBOROS_SKILL_SOURCE_SUGGESTIONS"
FOLDER_SETTING = "OUROBOROS_SKILLS_REPO_PATH"
# Where a repository keeps skills by a shared convention, beside the folders the ordinary walk finds.
STANDARD_SKILL_FOLDERS = (".agents/skills",)
KINDS = ("builtin", "hub", "git", "local", "registry")
TRUSTS = ("trusted", "review")
PARTS = ("official", "community")
BUILTIN_ID = "builtin"
HUB_OFFICIAL_ID = "ouroboros"
HUB_COMMUNITY_ID = "ouroboros-community"
CLAWHUB_ID = "clawhub"
LOCAL_ID = "my-folder"
RESERVED_IDS = frozenset({BUILTIN_ID, HUB_OFFICIAL_ID, HUB_COMMUNITY_ID, CLAWHUB_ID, LOCAL_ID})
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
# Never a ref name (git check-ref-format), and an option or a refspec in a fetch argument.
_BAD_REF_RE = re.compile(r"^[-+]|\.\.|@\{|[\s:~^?*\[\\\x00-\x1f\x7f]")
_SCP_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:[^\s]+$")
_MAX_SOURCES = 64


class SkillSourceError(ValueError):
    """A source row or a requested source change is invalid; the message is owner-facing."""


def repo_from_catalog_url(url: str) -> tuple[str, str]:
    """``(repository URL, branch)`` behind a hub catalog URL.

    ``https://raw.githubusercontent.com/<owner>/<repo>/<branch>/catalog.json`` is the shape every
    released version configured; any other value is taken as a repository URL as it is.
    """
    text = str(url or "").strip()
    parsed = urllib.parse.urlparse(text)
    if parsed.hostname == "raw.githubusercontent.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3:
            return f"https://github.com/{parts[0]}/{parts[1]}", parts[2]
    if text.endswith("/catalog.json"):
        text = text[: -len("/catalog.json")]
    return text, ""


def validate_git_url(url: str) -> str:
    """The URL as given when git may fetch it, or ``SkillSourceError``.

    http(s), ssh (``git@host:path`` or ``ssh://``), ``file://`` and an absolute path on this computer are
    accepted. Credentials never ride in a URL: the host's own git credentials authenticate.
    """
    text = str(url or "").strip()
    if not text or any(ch.isspace() or ord(ch) < 0x20 for ch in text):
        raise SkillSourceError("The repository address is empty or contains spaces.")
    if _SCP_RE.match(text):
        return text
    if pathlib.Path(text).is_absolute():
        return text
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"https", "ssh", "file", "http"}:
        raise SkillSourceError("Use an http://, https://, ssh:// or file:// repository address, or a folder path.")
    if parsed.password or (parsed.scheme in {"https", "http"} and parsed.username):
        raise SkillSourceError("Remove the credentials from the address; git uses this computer's own sign-in.")
    if parsed.scheme != "file" and not parsed.hostname:
        raise SkillSourceError("The repository address has no host.")
    return text


def repository_owner(url: str) -> str:
    """The account a repository address names (``https://host/<owner>/<repo>``, ``git@host:<owner>/<repo>``),
    or '' for a folder, a ``file://`` address or an address without one."""
    text = str(url or "").strip()
    if _SCP_RE.match(text):
        path = text.split(":", 1)[1]
    else:
        parsed = urllib.parse.urlparse(text)
        path = parsed.path if parsed.scheme in {"https", "http", "ssh"} and parsed.hostname else ""
    parts = [part for part in path.split("/") if part]
    return "/".join(parts[:-1])


def _flag(settings: Dict[str, Any], key: str, default: str) -> bool:
    raw = settings.get(key)
    if raw is None:
        raw = os.environ.get(key, default)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _with_settings_folder(rows: List[Dict[str, Any]], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """My folder is a view of the Settings path; other local sources are independent."""
    folder = str(settings.get(FOLDER_SETTING) or "").strip()
    linked = source_by_id(rows, LOCAL_ID)
    if not folder:
        return [row for row in rows if row["id"] != LOCAL_ID]
    path = str(pathlib.Path(folder).expanduser())
    if linked is not None:
        return [{**row, "path": path} if row["id"] == LOCAL_ID else row for row in rows]
    return [*rows, {"id": LOCAL_ID, "name": "My folder", "kind": "local", "path": path,
                   "trust": "review", "confirmed": True}]


def derived_sources(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The list an install without ``OUROBOROS_SKILL_SOURCES`` behaves as (never written by itself)."""
    from ouroboros.config import SETTINGS_DEFAULTS

    rows: List[Dict[str, Any]] = [{
        "id": BUILTIN_ID, "name": "Built in", "kind": "builtin",
        "trust": "trusted" if _flag(settings, "OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true") else "review",
        "confirmed": True,
    }]
    catalog_url = str(settings.get("OUROBOROS_HUB_CATALOG_URL") or SETTINGS_DEFAULTS["OUROBOROS_HUB_CATALOG_URL"])
    repo, ref = repo_from_catalog_url(catalog_url)
    if repo:  # a raw GitHub catalog address names its branch; a repository address follows its default branch
        hub = {"kind": "hub", "url": repo, **({"ref": ref} if ref else {}), "confirmed": False}
        rows.append({"id": HUB_OFFICIAL_ID, "name": "Ouroboros", "part": "official", "trust": "trusted", **hub})
        rows.append({"id": HUB_COMMUNITY_ID, "name": "Ouroboros Community", "part": "community",
                     "trust": "review", **hub})
    registry = str(settings.get("OUROBOROS_CLAWHUB_REGISTRY_URL") or "").strip()
    rows.append({"id": CLAWHUB_ID, "name": "ClawHub", "kind": "registry",
                 **({"url": registry} if registry else {}), "trust": "review", "confirmed": True})
    return _with_settings_folder(rows, settings)


def _view_folder(path: Any) -> str:
    """One repository-relative spelling, including old records with './' or repeated separators."""
    return "/".join(pathlib.PurePosixPath(str(path or "").strip().strip("/")).parts)


def _checked_view(ref: str, commit: str, path: str, *, where: str = "") -> Dict[str, str]:
    """The ``commit`` (it wins over a ``ref``) or ``ref`` and the ``path`` a repository row stores."""
    ref, commit, path = ref.strip(), commit.strip().lower(), _view_folder(path)
    if commit and not _COMMIT_RE.match(commit):
        raise SkillSourceError(f"{where}commit must be a full 40-character commit id.")
    if ref and _BAD_REF_RE.search(ref):
        raise SkillSourceError(f"{where}{ref!r} is not a branch, tag or commit name.")
    if path and ".." in pathlib.PurePosixPath(path).parts:
        raise SkillSourceError(f"{where}the folder must stay inside the repository.")
    out = {"commit": commit} if commit else ({"ref": ref} if ref else {})
    return {**out, "path": path} if path else out


def view_fields(revision: str = "", subfolder: str = "") -> Dict[str, str]:
    """The row fields of an owner's ``revision`` (a branch, a tag or a full commit id; '' follows the default
    branch) and ``subfolder`` of a repository ('' is its top folder)."""
    revision = str(revision or "").strip()
    commit = revision if _COMMIT_RE.match(revision.lower()) else ""
    return _checked_view("" if commit else revision, commit, str(subfolder or ""))


def source_view(fields: Dict[str, Any]) -> tuple[str, str]:
    """``(revision, folder)`` a repository row or a recorded source names: its pinned commit or ref ('' follows
    the default branch) and its folder ('' is the top one)."""
    return str(fields.get("commit") or fields.get("ref") or ""), _view_folder(fields.get("path"))


def standard_skill_dirs(root: pathlib.Path, taken: Any = (), *, listdir: Any) -> List[pathlib.Path]:
    """The skill folders in ``root``'s ``STANDARD_SKILL_FOLDERS``, without the names in ``taken`` (the ordinary
    walk's finds keep their folders). An absent folder holds none; any other folder is read with ``listdir``, the
    caller's own walk reader, which decides what an unreadable one means."""
    from ouroboros.contracts.skill_manifest import canonical_skill_name

    out: List[pathlib.Path] = []
    for rel in STANDARD_SKILL_FOLDERS:
        folder = pathlib.Path(root, rel)
        try:
            if not stat.S_ISDIR(folder.stat().st_mode):
                continue
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            pass  # present but not stat-able: the reader meets that failure
        out += [p for p in listdir(folder) if ((p / "SKILL.md").is_file() or (p / "skill.json").is_file())
                and canonical_skill_name(p.name) not in taken]
    return out


def normalize_source(row: Any) -> Dict[str, Any]:
    """One validated row with only the known fields, or ``SkillSourceError``."""
    if not isinstance(row, dict):
        raise SkillSourceError("A source must be an object.")
    source_id = str(row.get("id") or "").strip()
    if not _ID_RE.match(source_id):
        raise SkillSourceError(f"Source id {source_id!r} must be lowercase letters, digits, '.', '_' or '-'.")
    kind = str(row.get("kind") or "").strip()
    if kind not in KINDS:
        raise SkillSourceError(f"Source {source_id!r} has an unknown kind {kind!r}.")
    trust = str(row.get("trust") or "").strip()
    if trust not in TRUSTS:
        raise SkillSourceError(f"Source {source_id!r} needs trust 'trusted' or 'review'.")
    out: Dict[str, Any] = {
        "id": source_id, "name": str(row.get("name") or source_id).strip()[:80] or source_id,
        "kind": kind, "trust": trust, "confirmed": row.get("confirmed") is not False,
    }
    if row.get("managed") is True:
        out["managed"] = True
    if kind in {"hub", "git"}:
        out["url"] = validate_git_url(str(row.get("url") or ""))
        out.update(_checked_view(str(row.get("ref") or ""), str(row.get("commit") or ""), str(row.get("path") or ""),
                                 where=f"Source {source_id!r}: "))
        part = str(row.get("part") or "").strip()
        if part:
            if kind != "hub" or part not in PARTS:
                raise SkillSourceError(f"Source {source_id!r}: part is 'official' or 'community' on a hub.")
            out["part"] = part
    elif kind == "local":
        path = str(row.get("path") or "").strip()
        if not path or not pathlib.Path(path).expanduser().is_absolute():
            raise SkillSourceError(f"Source {source_id!r} needs an absolute folder path.")
        out["path"] = str(pathlib.Path(path).expanduser())
    elif kind == "registry":
        url = str(row.get("url") or "").strip()
        if url:
            out["url"] = url
    return out


def parse_sources(raw: Any) -> List[Dict[str, Any]]:
    """Validate a whole list (JSON text or a list); ids are unique."""
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SkillSourceError(f"{SOURCES_SETTING} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise SkillSourceError(f"{SOURCES_SETTING} must be a JSON list.")
    if len(data) > _MAX_SOURCES:
        raise SkillSourceError(f"At most {_MAX_SOURCES} sources are supported.")
    rows = [normalize_source(item) for item in data]
    seen: set[str] = set()
    for row in rows:
        if row["id"] in seen:
            raise SkillSourceError(f"Source id {row['id']!r} appears twice.")
        seen.add(row["id"])
    return rows


def load_sources(settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The saved or derived list, with My folder projected from the current Settings path.

    A saved value that does not validate falls back to the derived list and is logged; the Sources page
    shows the problem through ``sources_problem``.
    """
    if settings is None:
        from ouroboros.config import load_settings

        settings = load_settings()
    raw = settings.get(SOURCES_SETTING)
    if raw in (None, "", []):
        return derived_sources(settings)
    try:
        return _with_settings_folder(parse_sources(raw), settings)
    except SkillSourceError:
        log.warning("Saved %s is invalid; using the derived source list", SOURCES_SETTING, exc_info=True)
        return derived_sources(settings)


def sources_problem(settings: Dict[str, Any]) -> str:
    """Why the saved list is not in use, or ''."""
    raw = settings.get(SOURCES_SETTING)
    if raw in (None, "", []):
        return ""
    try:
        parse_sources(raw)
    except SkillSourceError as exc:
        return str(exc)
    return ""


def source_suggestions(settings: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The build's suggested repositories (``OUROBOROS_SKILL_SOURCE_SUGGESTIONS``, a JSON list of ``{name, url,
    ref|commit, path, about}``), each with ``connected`` when a row already reads that view. An entry that does
    not validate is left out and logged; nothing here connects a source or sets a trust."""
    raw = settings.get(SUGGESTIONS_SETTING)
    try:
        data = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or [])
    except json.JSONDecodeError:
        log.warning("%s is not valid JSON; no source is suggested", SUGGESTIONS_SETTING)
        return []
    out: List[Dict[str, Any]] = []
    for item in (data if isinstance(data, list) else [])[:_MAX_SOURCES]:
        try:
            if not isinstance(item, dict):
                raise SkillSourceError("a suggestion is an object")
            url = validate_git_url(str(item.get("url") or ""))
            fields = _checked_view(str(item.get("ref") or ""), str(item.get("commit") or ""), str(item.get("path") or ""))
        except SkillSourceError:
            log.warning("Skipping a suggested skill source that does not validate: %r", item, exc_info=True)
            continue
        view = source_view(fields)
        out.append({"name": str(item.get("name") or "").strip()[:80] or pathlib.PurePosixPath(url.rstrip("/")).stem,
                    "url": url, **fields, "about": str(item.get("about") or "").strip()[:300],
                    "connected": any(row.get("url") == url and row["kind"] in {"hub", "git"}
                                     and source_view(row) == view for row in rows)})
    return out


def source_by_id(rows: List[Dict[str, Any]], source_id: str) -> Optional[Dict[str, Any]]:
    return next((row for row in rows if row["id"] == str(source_id or "")), None)


def save_sources(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Persist the list and its Settings folder together under the settings lock."""
    from ouroboros.gateway.owner_settings import (
        _owner_update_settings,
        fields_on_saved_document,
        settings_document_mutation,
    )

    valid = parse_sources(rows)
    value = json.dumps(valid, ensure_ascii=False, separators=(",", ":"))
    folder = source_by_id(valid, LOCAL_ID)
    fields = {SOURCES_SETTING: value, FOLDER_SETTING: str((folder or {}).get("path") or "")}
    with settings_document_mutation():
        _owner_update_settings(fields_on_saved_document(fields))
        os.environ.update(fields)  # projected under the same lock as the commit
    return valid


def _new_id(rows: List[Dict[str, Any]], name: str) -> str:
    base = re.sub(r"[^a-z0-9._-]+", "-", str(name or "").strip().lower()).strip("-._")[:48] or "source"
    taken = {row["id"] for row in rows} | set(RESERVED_IDS)
    candidate, index = base, 2
    while candidate in taken or not _ID_RE.match(candidate):
        candidate, index = f"{base}-{index}"[:64], index + 1
    return candidate


def with_added_source(rows: List[Dict[str, Any]], *, kind: str, url: str = "", path: str = "",
                      name: str, trust: str, revision: str = "",
                      subfolder: str = "") -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Add a source; reconnecting the default public hub restores its two views. Nothing is written.

    A repository is read at ``revision`` from ``subfolder`` (``view_fields``); another view of a connected
    repository is another source. A folder on this computer is read as it is."""
    if kind in {"hub", "git"}:
        location: Dict[str, Any] = {"url": url, **view_fields(revision, subfolder)}
    elif str(revision or "").strip() or str(subfolder or "").strip():
        raise SkillSourceError("A folder on this computer is read as it is: leave the revision and subfolder empty.")
    else:
        location = {"path": path}
    view = source_view(location)
    for row in rows:
        if kind in {"hub", "git"} and row.get("url") == url and not row.get("part") and source_view(row) == view:
            raise SkillSourceError(f"{row['name']} already reads this repository"
                                   f"{' at the same revision and folder' if any(view) else ''}.")
        if kind == "local" and row.get("path") == str(pathlib.Path(path).expanduser()):
            raise SkillSourceError(f"{row['name']} already reads this folder.")
    public = [r for r in derived_sources({}) if r["kind"] == "hub"]
    if (kind == "hub" and public and url.rstrip("/").removesuffix(".git") == public[0]["url"]
            and view in {("", ""), source_view(public[0])}):
        missing = [normalize_source(r) for r in public if not source_by_id(rows, r["id"])]
        if missing:
            return [*rows, *missing], missing[0]
        raise SkillSourceError("The public hub is already connected.")
    row = normalize_source({"id": _new_id(rows, name), "name": name, "kind": kind, "trust": trust,
                            "confirmed": True, **location})
    return [*rows, row], row


def with_changed_source(rows: List[Dict[str, Any]], source_id: str, *, name: Any = None,
                        trust: Any = None, confirmed: Any = None) -> List[Dict[str, Any]]:
    """``rows`` with one row renamed, re-trusted or confirmed. Nothing is written."""
    if source_by_id(rows, source_id) is None:
        raise SkillSourceError(f"No source {source_id!r}.")
    out = []
    for row in rows:
        if row["id"] == source_id:
            row = dict(row)
            if name is not None:
                row["name"] = str(name).strip()[:80] or row["name"]
            if trust is not None:
                row["trust"] = str(trust)
            if confirmed is not None:
                row["confirmed"] = bool(confirmed)
            row = normalize_source(row)
        out.append(row)
    return out


def without_source(rows: List[Dict[str, Any]], source_id: str) -> List[Dict[str, Any]]:
    """``rows`` without one source. Installed skills keep working: their records name the source."""
    if source_by_id(rows, source_id) is None:
        raise SkillSourceError(f"No source {source_id!r}.")
    return [row for row in rows if row["id"] != source_id]


__all__ = [
    "BUILTIN_ID", "CLAWHUB_ID", "HUB_COMMUNITY_ID", "HUB_OFFICIAL_ID", "KINDS", "LOCAL_ID", "SOURCES_SETTING",
    "STANDARD_SKILL_FOLDERS", "SUGGESTIONS_SETTING", "SkillSourceError", "TRUSTS", "derived_sources", "load_sources",
    "normalize_source", "parse_sources", "repo_from_catalog_url", "repository_owner", "save_sources",
    "source_by_id", "source_suggestions", "source_view", "sources_problem", "standard_skill_dirs",
    "validate_git_url", "view_fields", "with_added_source", "with_changed_source", "without_source",
]
