"""Where a skill is published, and the hub entry that lists it.

A skill installed from a git-hosted source publishes back to that origin: the hub for a hosted hub
entry, the author's repository for a hub entry that points at one, the repository itself (the exact
folder it came from, including the repository root) for a plain git source. GitHub HTTPS and SSH
origins are supported; publication to other hosts is not yet supported. Everything else (made here, built in, from a registry or
a local folder) publishes to the hub, or to a git source the owner names. The owner's choice arrives as
``{"source_id", "mode"}`` (mode ``origin``, ``entry`` or ``new_entry``, a new name beside an existing
entry) from the Publish dialog, through the publish task's ``skill_publish_target.destination``.

Publishing is always a pull request through the existing consent, scan and receipt path. A hub with
``hub.json`` receives the skill as one entry (``entries/<name>.json``) plus its hosted files; the hub
regenerates its v1 ``catalog.json`` itself, and its owner-update automerge accepts only an entry with
its own payload. A repository without ``hub.json`` is an older hub and gets its v1 catalog row. A git
destination this account may push to merges the request at once (the files are published); otherwise
the request waits for the destination's maintainers. Files reaching a repository and the skill being
listed in a catalog are separate results: a hub lists a skill when its entry PR merges, a plain
repository lists every folder it holds, and a hub entry that points at the author's repository is
listed anew only by its own one-entry hub change to the new commit.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ouroboros.skill_publish_result import SkillPublishDestinationError, parse_skill_publish_destination

_GITHUB_RE = re.compile(r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
                        r"([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$")
MODES = ("origin", "entry", "new_entry")


class PublishListingError(ValueError):
    """The destination's listing refuses this publication; ``repair_hint`` says what to do."""

    def __init__(self, reason_code: str, repair_hint: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.repair_hint = repair_hint


@dataclass(frozen=True)
class PublishDestination:
    """One repository a skill can be published to. ``branch`` empty: the repository's default."""

    kind: str  # hub | git
    owner: str
    repo: str
    branch: str
    folder: str  # git: the folder holding skill folders
    source_id: str
    source_name: str
    pointer: bool = False  # the author's repository behind a hub pointer entry
    skill_path: Optional[str] = None  # None: unknown; "": repository root
    new_entry: bool = False  # hub: a new name, never an update of an existing entry

    @property
    def repository(self) -> str:
        return f"{self.owner}/{self.repo}"

    def skill_folder(self, skill: str) -> str:
        if self.kind == "hub":
            return f"skills/{skill}"  # hosted entries live where v1 clients read them
        if self.skill_path is not None:
            return self.skill_path
        return "/".join(part for part in (self.folder, skill) if part)


def github_repository(url: str) -> tuple[str, str]:
    """``(owner, repository)`` of a GitHub HTTPS or SSH address; publication supports GitHub only."""
    match = _GITHUB_RE.match(str(url or "").strip())
    if not match:
        error = SkillPublishDestinationError("hub_destination_invalid")
        error.args = ("Publishing currently supports GitHub repositories only. Choose a GitHub source.",)
        raise error
    return match.group(1), match.group(2)


def _hub_destination(rows: List[Dict[str, Any]], *, new_entry: bool = False) -> PublishDestination:
    hubs = [row for row in rows if row["kind"] == "hub"]
    row = next((r for r in hubs if r.get("part") == "official"), hubs[0] if hubs else None)
    if row is None:
        from ouroboros.config import get_ouroboroshub_catalog_url

        owner, repo, branch = parse_skill_publish_destination(get_ouroboroshub_catalog_url())
        return PublishDestination("hub", owner, repo, branch, "skills", "hub", "OuroborosHub", new_entry=new_entry)
    owner, repo = github_repository(row.get("url", ""))
    return PublishDestination("hub", owner, repo, str(row.get("ref") or ""), "skills", row["id"], row["name"],
                              new_entry=new_entry)


def _row_destination(row: Dict[str, Any], *, skill_path: Optional[str] = None, new_entry: bool = False) -> PublishDestination:
    if row["kind"] not in {"hub", "git"}:
        raise SkillPublishDestinationError("destination_not_git")
    owner, repo = github_repository(row.get("url", ""))
    if row["kind"] == "hub":
        return PublishDestination("hub", owner, repo, str(row.get("ref") or ""), "skills", row["id"], row["name"],
                                  new_entry=new_entry)
    return PublishDestination("git", owner, repo, str(row.get("ref") or ""), str(row.get("path") or ""),
                              row["id"], row["name"], skill_path=skill_path)


def _origin(record: Mapping[str, Any], rows: List[Dict[str, Any]]) -> Optional[PublishDestination]:
    """The repository the installed release came from, or ``None`` when it came from no git source."""
    from ouroboros.skill_source_catalog import record_matches_row

    release = record.get("release") or {}
    source = record.get("source") or {}
    if release.get("repo"):  # a hub entry pointing at its author's repository
        owner, repo = github_repository(release["repo"])
        return PublishDestination("git", owner, repo, "", "", str(source.get("id") or ""), f"{owner}/{repo}",
                                  pointer=True, skill_path=str(release.get("path") or ""))
    row = next((row for row in rows if record_matches_row(record, row)), None)
    if row is None or row["kind"] not in {"hub", "git"}:
        return None
    return _row_destination(row, skill_path=release.get("path"))


def _requested(requested: Any) -> Tuple[str, str]:
    if isinstance(requested, Mapping):
        source_id, mode = str(requested.get("source_id") or "").strip(), str(requested.get("mode") or "").strip()
        if mode and mode not in MODES:
            raise SkillPublishDestinationError("destination_unknown")
        return source_id, mode or ("entry" if source_id else "origin")
    text = str(requested or "").strip()
    return ("", "origin") if not text else (text, "entry")


def resolve_publish_destination(drive_root: pathlib.Path, skill: str, requested: Any = None, *,
                                rows: Optional[List[Dict[str, Any]]] = None) -> PublishDestination:
    """The destination ``requested`` names (``hub``, a source id, or ``{source_id, mode}``), else the
    skill's origin, else the hub."""
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_sources import load_sources, source_by_id

    rows = rows if rows is not None else load_sources()
    source_id, mode = _requested(requested)
    record = read_install_record(pathlib.Path(drive_root), skill) or {}
    if mode == "origin":
        origin = _origin(record, rows) if record else None
        if origin is not None:
            return origin
        if source_id in ("", "hub"):
            return _hub_destination(rows)
    if source_id == "hub":
        return _hub_destination(rows, new_entry=mode == "new_entry")
    row = source_by_id(rows, source_id)
    if row is None:
        raise SkillPublishDestinationError("destination_unknown")
    if mode == "new_entry" and row["kind"] != "hub":
        raise SkillPublishDestinationError("destination_not_hub")
    release = record.get("release") or {}
    same_origin = bool(record) and (record.get("source") or {}).get("id") == row["id"] and not release.get("repo")
    return _row_destination(row, skill_path=release.get("path") if same_origin else None,
                            new_entry=mode == "new_entry")


def captured_release_digest(snapshot: Any) -> str:
    """The release digest of exactly the files a publication sends with their manifest's declaration:
    the digest a source index states and an installer verifies for those files."""
    from ouroboros.skill_identity import release_digest_from_files

    return release_digest_from_files([{"path": item.path, "sha256": item.sha256} for item in snapshot.public_files],
                                     snapshot.manifest.install_specs())


def hub_entry(skill: str, snapshot: Any, *, login: str, digest: str, existing: Optional[Dict[str, Any]],
              category: List[str]) -> Dict[str, Any]:
    """The one-entry hub record of a hosted skill; an existing entry keeps its recorded owner and categories
    unless new ones are named. Generated fields (version, declaration, digest) follow the published files."""
    manifest = snapshot.manifest
    entry: Dict[str, Any] = {
        "schema_version": 2, "name": skill, "description": manifest.description,
        "category": list(category or (existing or {}).get("category") or []),
        "owner": str((existing or {}).get("owner") or login), "version": manifest.version,
        "source": {"kind": "hosted", "path": f"skills/{skill}"}, "digest": digest, "withdrawn": None,
    }
    if manifest.skill_type:
        entry["type"] = manifest.skill_type
    if manifest.when_to_use:
        entry["when_to_use"] = manifest.when_to_use
    specs = manifest.install_specs()
    if specs:
        entry["install_specs"] = specs
    if (existing or {}).get("author"):
        entry["author"] = existing["author"]
    return entry


def entry_bytes(entry: Mapping[str, Any]) -> bytes:
    """The hub's canonical entry file form (sorted keys, two-space indent, final newline)."""
    return (json.dumps(entry, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def hub_entry_listing(read: Callable[[str], Optional[bytes]], snapshot: Any, *, skill: str, login: str,
                      category: Any, new_entry: bool) -> Tuple[Dict[str, bytes], Optional[str]]:
    """``({"entries/<name>.json": bytes}, previous version or None)`` for a hub with ``hub.json``.

    ``read(path)`` returns a file of the destination at the publication's base commit (``None`` when absent).
    """
    try:
        raw = read(f"entries/{skill}.json")
        existing = json.loads(raw.decode("utf-8")) if raw else None
        known = json.loads((read("categories.json") or b"{}").decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise PublishListingError("upstream_catalog_invalid", "Repair the upstream Hub entry files, then retry.") from exc
    if (existing is not None and not isinstance(existing, dict)) or not isinstance(known, dict):
        raise PublishListingError("upstream_catalog_invalid", "Repair the upstream Hub entry files, then retry.")
    if existing and new_entry:
        raise PublishListingError("entry_exists", f"The hub already lists {skill}: give the skill a new name, review "
                                                  "it, then publish it as a new skill.")
    if existing and (existing.get("source") or {}).get("kind") == "git":
        raise PublishListingError("pointer_entry", "The hub lists this skill from its author's repository: publish "
                                                   "to that repository (destination origin); its hub entry then moves "
                                                   "to the new commit in its own change.")
    previous = str(existing.get("version") or "") if existing else None
    if previous is not None and previous == str(snapshot.manifest.version or ""):
        raise PublishListingError("catalog_version_exists", "Bump the skill version, run a fresh review, then retry.")
    ids = [str(item.get("id")) for item in (known.get("categories") or []) if isinstance(item, dict) and item.get("id")]
    chosen = list(dict.fromkeys(str(item) for item in (category if isinstance(category, list) else
                                                       [category] if category else [])))[:2]
    if ids and any(item not in ids for item in chosen):
        raise PublishListingError("category_unknown", f"Use hub category ids ({', '.join(ids)}), then retry.")
    entry = hub_entry(skill, snapshot, login=login, digest=captured_release_digest(snapshot),
                      existing=existing, category=chosen)
    if not entry["category"]:
        known_ids = ", ".join(ids) or "see the hub's categories.json"
        raise PublishListingError("category_required", f"Name one or two categories ({known_ids}), then retry.")
    return {f"entries/{skill}.json": entry_bytes(entry)}, previous


# --- an older hub (no hub.json): the v1 catalog row lists the skill ---------------------------------

def catalog_row(
    skill: str,
    snapshot: Any,
    payload_files: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The v1 catalog row of the published files (per-file SHA-256 for older clients)."""
    manifest = snapshot.manifest
    entry: Dict[str, Any] = {
        "slug": skill,
        "name": manifest.name or skill,
        "description": manifest.description,
        "version": manifest.version,
        "type": manifest.skill_type,
        "files": [
            {
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
                "size": int(item["size"]),
            }
            for item in payload_files
        ],
    }
    install_specs = manifest.install_specs()
    if install_specs:
        entry["install_specs"] = install_specs
    if manifest.when_to_use:
        entry["when_to_use"] = manifest.when_to_use
    return entry


def _catalog_file_paths(entry: Mapping[str, Any]) -> set[str]:
    files = entry.get("files")
    if files is None:
        return set()
    if not isinstance(files, list):
        raise PublishListingError("upstream_catalog_invalid", "Repair the upstream Hub catalog, then retry.")
    paths: set[str] = set()
    for item in files:
        raw = item.get("path") if isinstance(item, Mapping) else None
        path = str(raw or "")
        pure = pathlib.PurePosixPath(path)
        if (
            not path
            or pure.is_absolute()
            or path != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or path in paths
        ):
            raise PublishListingError("upstream_catalog_invalid", "Repair the upstream Hub catalog, then retry.")
        paths.add(path)
    return paths


def update_catalog(
    catalog: Dict[str, Any],
    entry: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Tuple[str, ...], str | None]:
    """``(mode, catalog with the row, obsolete file paths, previous version)``; rows stay sorted by slug."""
    skills = catalog.get("skills")
    if not isinstance(skills, list):
        raise PublishListingError("upstream_catalog_invalid", "Repair the upstream Hub catalog, then retry.")
    slug = str(entry.get("slug") or "")
    matching_indexes = [
        index for index, item in enumerate(skills) if isinstance(item, dict) and item.get("slug") == slug
    ]
    if len(matching_indexes) > 1:
        raise PublishListingError("upstream_catalog_invalid", "Repair the upstream Hub catalog, then retry.")
    existing_index = matching_indexes[0] if matching_indexes else None
    previous_version = None
    if existing_index is None:
        mode = "add"
        obsolete_paths: Tuple[str, ...] = ()
        skills.append(entry)
    else:
        existing = skills[existing_index]
        previous_version = str(existing.get("version") or "")
        if str(existing.get("version") or "") == str(entry.get("version") or ""):
            raise PublishListingError("catalog_version_exists", "Bump the skill version, run a fresh review, then retry.")
        mode = "update"
        obsolete_paths = tuple(sorted(_catalog_file_paths(existing) - _catalog_file_paths(entry)))
        skills[existing_index] = entry
    skills.sort(key=lambda item: str(item.get("slug") or "") if isinstance(item, dict) else "")
    catalog["skills"] = skills
    return mode, catalog, obsolete_paths, previous_version


def task_destination(ctx: Any) -> Any:
    """The destination the owner chose for a publish task (``skill_publish_target.destination``)."""
    metadata = getattr(ctx, "task_metadata", None)
    target = metadata.get("skill_publish_target") if isinstance(metadata, Mapping) else None
    value = target.get("destination") if isinstance(target, Mapping) else None
    return value if isinstance(value, (Mapping, str)) else None


__all__ = [
    "MODES", "PublishDestination", "PublishListingError", "captured_release_digest", "catalog_row", "entry_bytes",
    "github_repository", "hub_entry", "hub_entry_listing", "resolve_publish_destination", "task_destination",
    "update_catalog",
]
