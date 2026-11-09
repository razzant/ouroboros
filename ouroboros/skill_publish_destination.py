"""Where a skill is published, and the hub entry that lists it.

A skill installed from a git-hosted source publishes back to that origin: the hub for a hosted hub
entry, the author's repository for a hub entry that points at one, the repository itself (the exact
folder it came from, including the repository root) for a plain git source. GitHub HTTPS and SSH
origins are supported; publication to other hosts is not yet supported. Everything else (made here, built in, from a registry or
a local folder) publishes to the hub, or to a git source the owner names. A hub read from a folder of its
repository is never a destination: a hub entry is written at the repository's top; nor is a view pinned to a
commit, which would never read it, except for the release that came from it, nor a view of a tag (refused by
name: ``destination_tag``); the hub chosen by default passes over all three. A branch may be named in full
(``refs/heads/<name>``). The owner's choice arrives as
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

Before anything is written, ``plan_publication`` compares what the publication would change at the
destination's head, only within this skill's footprint (its folder's files, or at a repository root only
the files it sends, plus its entry or v1 row), with the release the local copy was installed from: the
install record's commit, when that release came from exactly this repository folder. A change the
destination made after that release, or any replaced content when no such release is recorded, needs
the mind's explicit decision bound to ``destination_state``; ``merge`` markers and publication receipts
never stand in for that base.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def _tag(row: Dict[str, Any]) -> str:
    """The tag a repository row reads when its ``ref`` names one in full (``refs/tags/<name>``), else ''."""
    ref = str(row.get("ref") or "")
    return ref[len("refs/tags/"):] if ref.startswith("refs/tags/") else ""


def _branch(row: Dict[str, Any]) -> str:
    """The branch a publication to ``row`` reads and targets ('' is the repository's default), named in full or not.
    A tag never moves, so a view of one would never read a publication: it is refused by name, never sent to another
    branch (a bare name the repository holds as a tag is refused when the branch is read: ``fetch_branch_head``)."""
    if _tag(row):
        error = SkillPublishDestinationError("destination_tag")
        error.args = (f"{row['name']} stays at the tag {_tag(row)}, so it would never read this publication. "
                      "Choose a source that follows a branch.",)
        raise error
    return str(row.get("ref") or "").removeprefix("refs/heads/")


def _hub_destination(rows: List[Dict[str, Any]], *, new_entry: bool = False) -> PublishDestination:
    # A hub entry is written at the top of a branch: a hub read from a folder, a commit or a tag never reads it.
    hubs = [row for row in rows if row["kind"] == "hub" and not row.get("path") and not row.get("commit")
            and not _tag(row)]
    row = next((r for r in hubs if r.get("part") == "official"), hubs[0] if hubs else None)
    if row is None:
        from ouroboros.config import get_ouroboroshub_catalog_url

        owner, repo, branch = parse_skill_publish_destination(get_ouroboroshub_catalog_url())
        return PublishDestination("hub", owner, repo, branch, "skills", "hub", "OuroborosHub", new_entry=new_entry)
    owner, repo = github_repository(row.get("url", ""))
    return PublishDestination("hub", owner, repo, _branch(row), "skills", row["id"], row["name"], new_entry=new_entry)


def _row_destination(row: Dict[str, Any], *, skill_path: Optional[str] = None, new_entry: bool = False) -> PublishDestination:
    if row["kind"] not in {"hub", "git"}:
        raise SkillPublishDestinationError("destination_not_git")
    if row["kind"] == "hub" and row.get("path"):  # its entries live in that folder, not where a hub entry goes
        error = SkillPublishDestinationError("hub_folder_unsupported")
        error.args = (f"Publishing to a hub inside a repository folder ({row['path']}) is not supported. "
                      "Choose another destination.",)
        raise error
    owner, repo = github_repository(row.get("url", ""))
    if row["kind"] == "hub":
        return PublishDestination("hub", owner, repo, _branch(row), "skills", row["id"], row["name"],
                                  new_entry=new_entry)
    return PublishDestination("git", owner, repo, _branch(row), str(row.get("path") or ""), row["id"], row["name"],
                              skill_path=skill_path)


def _origin(record: Mapping[str, Any], rows: List[Dict[str, Any]]) -> Optional[PublishDestination]:
    """The repository the installed release came from, or ``None`` when it came from no git source."""
    from ouroboros.skill_source_catalog import row_for_record

    release = record.get("release") or {}
    source = record.get("source") or {}
    if release.get("repo"):  # a hub entry pointing at its author's repository
        owner, repo = github_repository(release["repo"])
        return PublishDestination("git", owner, repo, "", "", str(source.get("id") or ""), f"{owner}/{repo}",
                                  pointer=True, skill_path=str(release.get("path") or ""))
    row = row_for_record(dict(record), rows)
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
    from ouroboros.skill_source_catalog import record_matches_row
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
    release, source = record.get("release") or {}, record.get("source") or {}
    # Back in its own folder only on the row it came from: the view a record names (whatever the id), or an older
    # record's own row, as before; that older record's match of every view of its repository is for updates only.
    from_here = record_matches_row(dict(record), row) if "path" in source else source.get("id") == row["id"]
    same_origin = bool(record) and from_here and not release.get("repo")
    if row.get("commit") and not same_origin:  # a pinned view would never read what is published there
        error = SkillPublishDestinationError("destination_pinned")
        error.args = (f"{row['name']} stays at commit {row['commit'][:7]}, so it would never read this publication. "
                      "Choose a source that follows a branch.",)
        raise error
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


# --- what a publication replaces at its destination ---------------------------------------------------

UNCHANGED, CHANGED, UNKNOWN = "unchanged_since_base", "changed_since_base", "unknown"
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ABSENT = object()  # a path or listing field one side does not have
_UNREAD = object()  # a base listing that could not be read
_DIFF_MAX_BYTES = 256 * 1024
_DIFF_MAX_LINES = 600


@dataclass(frozen=True)
class DestinationReads:
    """Read-only access to one destination repository, supplied by the publisher's GitHub transport."""

    repository: Callable[[], Dict[str, Any]]  # {default_branch, can_push}
    branch_head: Callable[[str], Tuple[str, str]]  # branch ('' default) -> (commit, branch)
    file: Callable[[str, str], Optional[bytes]]  # (commit, path) -> bytes, None when absent
    tree: Callable[..., Optional[Dict[str, str]]]  # (commit, folder, missing_ok) -> {relative path: blob id}
    catalog: Callable[[str], Tuple[Dict[str, Any], str]]  # branch -> (v1 catalog at its head, that commit)
    blob: Callable[[str], bytes]  # blob id -> bytes


@dataclass(frozen=True)
class DestinationComparison:
    """What one publication changes in its destination's copy of a skill, path by path and listing field by
    field, each judged against the release the local copy came from (``upstream``: changed_since_base,
    unchanged_since_base, or unknown without a usable base). ``destination_state`` identifies exactly this
    comparison: the local snapshot, the destination and its footprint of this skill and of the base, never
    the whole head commit, so a change elsewhere in the repository leaves it valid."""

    destination_state: str
    head_commit: str
    base_commit: str  # '' when the local copy has no usable recorded base here
    base_reason: str  # install_record | no_install_record | base_unknown | installed_elsewhere | base_unreadable
    base_version: str
    metadata_path: str  # entries/<name>.json, catalog.json (its v1 row) or '' for a plain repository
    paths: Tuple[Dict[str, Any], ...]  # {path, effect: replaced|deleted|added, upstream, head, base, sent}
    fields: Tuple[Dict[str, Any], ...]  # {field, change: replaced|removed|added, upstream, head, base, sent}

    def decision_rows(self) -> List[Dict[str, Any]]:
        return [row for row in (*self.paths, *self.fields) if needs_decision(row)]

    def projection(self, state: str, artifact: Optional[Mapping[str, Any]] = None, artifact_error: str = "") -> Dict[str, Any]:
        """The result-envelope form (``skill_publish_result`` bounds the rows): decision rows first."""
        def order(row: Mapping[str, Any]) -> Tuple[bool, str]:
            return not needs_decision(row), str(row.get("path") or row.get("field"))
        return {
            "state": state, "destination_state": self.destination_state, "head_commit": self.head_commit,
            "base_commit": self.base_commit, "base_reason": self.base_reason, "base_version": self.base_version,
            "metadata_path": self.metadata_path,
            "paths": [{key: row[key] for key in ("path", "effect", "upstream")} for row in sorted(self.paths, key=order)],
            "fields": [{key: row[key] for key in ("field", "change", "upstream")} for row in sorted(self.fields, key=order)],
            **({"artifact": dict(artifact)} if artifact else {}), **({"artifact_error": artifact_error} if artifact_error else {}),
        }


def needs_decision(row: Mapping[str, Any]) -> bool:
    """Whether a changed path or field replaces content its base does not explain. Adding where the
    destination holds nothing loses nothing, unless the destination removed it after the base."""
    added = row.get("effect", row.get("change")) == "added"
    return row["upstream"] == CHANGED or (row["upstream"] == UNKNOWN and not added)


_DECISION_HINTS = {
    "destination_decision_required": (
        "This publication would replace content at the destination that {why}: read destination_comparison and "
        "its artifact. Bring that content into the local copy and review it again, or publish over it on purpose "
        "by calling again with expected_destination_state and replacement_rationale."),
    "destination_state_changed": (
        "This skill's copy at the destination, its base or the local snapshot changed since that comparison; read "
        "the new destination_comparison before deciding again."),
    "replacement_rationale_required": (
        "Give replacement_rationale (1-2000 characters): why this publication replaces the destination content "
        "the comparison names."),
}


def destination_decision(comparison: DestinationComparison, expected_state: str, rationale: str) -> Tuple[str, str]:
    """``(state, detail)``: ``clear``, or ``decided`` with the normalized rationale, when the publication may
    proceed; else ``decision_required`` with its refusal reason code. Only the mind's explicit choice bound
    to exactly this comparison, with its reason, publishes over content the base does not explain."""
    expected = str(expected_state or "").strip().lower()
    reason = " ".join(str(rationale or "").split())
    if not comparison.decision_rows():
        return "clear", ""
    if expected == comparison.destination_state:
        return ("decided", reason) if 1 <= len(reason) <= 2000 else ("decision_required",
                                                                     "replacement_rationale_required")
    return "decision_required", "destination_state_changed" if expected else "destination_decision_required"


def decision_hint(code: str, comparison: DestinationComparison) -> str:
    why = ("changed after the release this copy was installed from" if comparison.base_commit
           else "has no recorded common base with this copy")
    return _DECISION_HINTS[code].format(why=why)


def replacement_section(comparison: DestinationComparison, rationale: str) -> str:
    """The pull request's account of the destination content the mind decided to replace, and why."""
    items = [f"- `{row['path']}` ({row['effect']})" if "path" in row
             else f"- `{comparison.metadata_path}` field `{row['field']}` ({row['change']})"
             for row in comparison.decision_rows()]
    shown = items[:50] + ([f"- … and {len(items) - 50} more"] if len(items) > 50 else [])
    where = (f"changed there after release `{comparison.base_commit}` that the local copy was installed from"
             if comparison.base_commit else "with no recorded common base with the local copy")
    return ("## Replaced destination content\n"
            f"Before any change this publication was compared with the destination at `{comparison.head_commit}`. "
            f"It replaces this content, {where}:\n" + "\n".join(shown) + f"\n\nAuthor rationale: {rationale}\n")


@dataclass(frozen=True)
class PublicationPlan:
    """What a publication writes, all read at one destination commit, and its comparison."""

    listing: Dict[str, bytes]
    obsolete_paths: Tuple[str, ...]
    mode: str
    base_sha: str  # the destination head the publication builds on
    base_branch: str
    can_push: bool
    previous_version: Optional[str]
    comparison: DestinationComparison


def git_blob_oid(data: bytes, *, width: int = 40) -> str:
    """Git's id of a blob holding ``data`` (``width`` 64: a SHA-256 repository)."""
    return (hashlib.sha256 if width == 64 else hashlib.sha1)(b"blob %d\0" % len(data) + data).hexdigest()


def installed_base(record: Optional[Mapping[str, Any]], target: PublishDestination, skill: str) -> Tuple[str, str]:
    """``(commit, reason)`` of the release the installed copy came from, when the install record names it
    for exactly this destination folder; else ``("", why)``. A requested merge only asks for a later
    release and a publication receipt only says what was sent once: neither shows what the copy holds."""
    if not record:
        return "", "no_install_record"
    release = record.get("release") or {}
    commit = str(release.get("commit") or "").lower()
    if not _COMMIT_RE.fullmatch(commit) or "path" not in release:
        return "", "base_unknown"
    try:
        owner, repo = github_repository(str(release.get("repo") or (record.get("source") or {}).get("url") or ""))
    except SkillPublishDestinationError:
        return "", "installed_elsewhere"
    if ((owner.casefold(), repo.casefold()) != (target.owner.casefold(), target.repo.casefold())
            or str(release.get("path") or "").strip("/") != target.skill_folder(skill).strip("/")):
        return "", "installed_elsewhere"
    return commit, "install_record"


def _catalog_row_of(catalog: Any, skill: str) -> Any:
    rows = catalog.get("skills") if isinstance(catalog, dict) else None
    if not isinstance(rows, list):
        return _UNREAD
    return next((copy.deepcopy(row) for row in rows if isinstance(row, dict) and row.get("slug") == skill), None)


def _listing_at(raw: Optional[bytes], metadata_path: str, skill: str) -> Any:
    """The skill's entry or v1 row in one commit's file (None: it has none; ``_UNREAD``: unreadable)."""
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        return _UNREAD
    if metadata_path == "catalog.json":
        return _catalog_row_of(data, skill)
    return data if isinstance(data, dict) else _UNREAD


def plan_publication(reads: DestinationReads, target: PublishDestination, snapshot: Any,
                     payload_files: Sequence[Mapping[str, Any]], *, skill: str, folder: str, login: str,
                     category: Any, record: Optional[Mapping[str, Any]]) -> PublicationPlan:
    """Read the destination at one commit, build what the publication writes there, and compare that with
    the destination's footprint of this skill at the same commit and at the local copy's base."""
    sent_paths = {str(item["path"]) for item in payload_files}
    listing: Dict[str, bytes] = {}
    previous: Optional[str] = None
    can_push, head_meta, sent_meta, metadata_path = False, None, None, ""
    if target.kind == "git":
        facts = reads.repository()
        can_push = bool(facts["can_push"])
        commit, branch = reads.branch_head(target.branch or facts["default_branch"])
        head = reads.tree(commit, folder) or {}
        # A skill at the repository root shares it with the repository's own files: nothing is deleted there.
        footprint = sent_paths | (set(head) if folder else set())
        obsolete = tuple(sorted(set(head) - sent_paths)) if folder else ()
        mode = "add" if folder and not head else "update"
    else:
        commit, branch = reads.branch_head(target.branch)
        if reads.file(commit, "hub.json") is None:  # an older hub lists a skill by its v1 catalog row alone
            catalog, commit = reads.catalog(branch)
            head_meta = _catalog_row_of(catalog, skill)
            sent_meta = catalog_row(skill, snapshot, payload_files)
            try:
                mode, updated, obsolete, previous = update_catalog(catalog, copy.deepcopy(sent_meta))
                listing["catalog.json"] = json.dumps(
                    updated, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
            except PublishListingError:
                raise
            except (TypeError, ValueError) as exc:
                raise PublishListingError("upstream_catalog_invalid", "Repair the upstream Hub catalog, then retry.") from exc
            metadata_path = "catalog.json"
            footprint = sent_paths | (_catalog_file_paths(head_meta) if isinstance(head_meta, dict) else set())
            head = reads.tree(commit, folder) or {}
        else:
            read_cache: Dict[str, Optional[bytes]] = {}

            def read_at_head(path: str) -> Optional[bytes]:
                if path not in read_cache:
                    read_cache[path] = reads.file(commit, path)
                return read_cache[path]

            listing, previous = hub_entry_listing(read_at_head, snapshot, skill=skill, login=login,
                                                  category=category, new_entry=target.new_entry)
            metadata_path = f"entries/{skill}.json"
            head_meta = _listing_at(read_at_head(metadata_path), metadata_path, skill)
            sent_meta = json.loads(listing[metadata_path].decode("utf-8"))
            head = reads.tree(commit, folder) or {}
            footprint = sent_paths | set(head)
            obsolete = tuple(sorted(set(head) - sent_paths))
            mode = "add" if previous is None else "update"
    base_commit, base_reason = installed_base(record, target, skill)
    base: Optional[Dict[str, str]] = None
    base_meta: Any = _UNREAD
    if base_commit == commit:
        base, base_meta = head, head_meta
    elif base_commit:
        base = reads.tree(base_commit, folder, True)
        if base is None:
            base_commit, base_reason = "", "base_unreadable"
        elif metadata_path:
            base_meta = _listing_at(reads.file(base_commit, metadata_path), metadata_path, skill)
    comparison = compare_destination(
        identity={"kind": target.kind, "repository": target.repository.casefold(), "branch": branch,
                  "folder": folder, "listing": metadata_path},
        snapshot_hash=snapshot.content_hash, sent={item.path: item.content for item in snapshot.public_files},
        footprint=footprint, head=head, base=base, prefix=folder, head_commit=commit, base_commit=base_commit,
        base_reason=base_reason, base_version=str(((record or {}).get("release") or {}).get("version") or "")
        if base_commit else "", metadata_path=metadata_path, head_meta=head_meta, base_meta=base_meta,
        sent_meta=sent_meta)
    return PublicationPlan(listing, obsolete, mode, commit, branch, can_push, previous, comparison)


def _canonical(value: Any) -> str:
    if value is _ABSENT:
        return "\0absent"
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _field_changes(head: Any, base: Any, sent: Any) -> List[Dict[str, Any]]:
    """Listing edits, including recreation of an entry removed since the installed base."""
    if not isinstance(sent, dict):
        return []
    if not isinstance(head, dict):
        if not isinstance(base, dict):
            return []  # a genuinely new listing has no prior fields to overwrite
        head = {}
    rows = []
    for key in sorted(set(head) | set(sent)):
        h, s = head.get(key, _ABSENT), sent.get(key, _ABSENT)
        if _canonical(h) == _canonical(s):
            continue
        b = base.get(key, _ABSENT) if isinstance(base, dict) else _ABSENT
        upstream = UNKNOWN if base is _UNREAD else UNCHANGED if _canonical(b) == _canonical(h) else CHANGED
        row = {"field": str(key), "change": "added" if h is _ABSENT else "removed" if s is _ABSENT else "replaced",
               "upstream": upstream}
        row.update({side: value for side, value in (("head", h), ("base", b), ("sent", s)) if value is not _ABSENT})
        rows.append(row)
    return rows


def compare_destination(*, identity: Mapping[str, str], snapshot_hash: str, sent: Mapping[str, bytes],
                        footprint: Iterable[str], head: Mapping[str, str], base: Optional[Mapping[str, str]],
                        prefix: str, head_commit: str, base_commit: str, base_reason: str, base_version: str = "",
                        metadata_path: str = "", head_meta: Any = None, base_meta: Any = _UNREAD,
                        sent_meta: Any = None) -> DestinationComparison:
    """Compare the footprint (paths relative to ``prefix``) of one skill at the destination head, as blob ids,
    with the bytes this publication sends there, judging each change against ``base`` (None: unknown)."""
    width = len(next(iter(head.values()), "")) or 40
    sent_ids = {path: git_blob_oid(data, width=width) for path, data in sent.items()}
    known = base is not None
    lead = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    state: List[List[Any]] = []
    paths: List[Dict[str, Any]] = []
    for rel in sorted(set(footprint)):
        h, s = head.get(rel), sent_ids.get(rel)
        b = base.get(rel) if base is not None else None
        state.append([rel, h, b if known else "?", s])
        if h == s:
            continue
        paths.append({"path": lead + rel, "effect": "added" if h is None else "deleted" if s is None else "replaced",
                      "upstream": UNKNOWN if not known else UNCHANGED if h == b else CHANGED,
                      "head": h, "base": b, "sent": s})
    base_meta = base_meta if known else _UNREAD
    fields = _field_changes(head_meta, base_meta, sent_meta)

    def listing_fact(value: Any) -> Any:
        return "unread" if value is _UNREAD else value

    destination_state = hashlib.sha256(json.dumps(
        {"schema": 1, "destination": dict(identity), "snapshot": snapshot_hash, "base": base_commit if known else "",
         "paths": state, "listing": [listing_fact(head_meta), listing_fact(base_meta), listing_fact(sent_meta)]},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    return DestinationComparison(destination_state, head_commit, base_commit if known else "", base_reason,
                                 base_version if known else "", metadata_path, tuple(paths), tuple(fields))


_BASE_WORDS = {
    "no_install_record": "this skill has no install record, so no release names its common base with the destination",
    "base_unknown": "its install record names no exact release commit and folder",
    "installed_elsewhere": "it was installed from another repository or folder",
    "base_unreadable": "the release commit its install record names can no longer be read there",
}
_UPSTREAM_WORDS = {CHANGED: "changed at the destination since the base", UNCHANGED: "unchanged since the base",
                   UNKNOWN: "no known base"}


def _diff(before: bytes, after: bytes, before_name: str, after_name: str) -> str:
    if max(len(before), len(after)) > _DIFF_MAX_BYTES:
        return f"(not diffed: {len(before)} → {len(after)} bytes)\n"
    try:
        old, new = before.decode("utf-8"), after.decode("utf-8")
    except UnicodeDecodeError:
        return f"(binary: {len(before)} → {len(after)} bytes)\n"
    lines = list(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                      before_name, after_name))
    shown = "".join(line if line.endswith("\n") else line + "\n" for line in lines[:_DIFF_MAX_LINES])
    omitted = len(lines) - _DIFF_MAX_LINES
    note = f"(… {omitted} more diff lines omitted; read both versions in full)\n" if omitted > 0 else ""
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", shown)), default=0))
    return f"{fence}diff\n{shown}{fence}\n{note}"


def comparison_report(comparison: DestinationComparison, *, skill: str, repository: str, branch: str,
                      sent: Mapping[str, bytes], blob: Callable[[str], bytes]) -> str:
    """Readable Markdown of one comparison: every changed path and listing field, and as diffs the
    destination's change since the base and this publication's change. ``sent`` maps repository paths to
    the bytes the publication sends; ``blob`` reads a destination blob by id."""
    contents: Dict[str, Any] = {}

    def read(oid: Optional[str]) -> Any:
        if oid is None:
            return b""
        if oid not in contents:
            try:
                data = blob(oid)
                contents[oid] = data if git_blob_oid(data, width=len(oid)) == oid else ValueError("bytes do not match")
            except Exception as exc:  # the diff is evidence only; the decision rests on the blob ids above
                contents[oid] = exc
        return contents[oid]

    head, base = comparison.head_commit, comparison.base_commit
    lines = [f"# What publishing {skill} would change at {repository}", "",
             f"- Destination: `{repository}` branch `{branch}` at `{head}`"
             + (f"; listing `{comparison.metadata_path}`." if comparison.metadata_path else "."),
             f"- Base: release {comparison.base_version or '(unnamed)'} at `{base}`, from the install record."
             if base else f"- No usable base: {_BASE_WORDS.get(comparison.base_reason, comparison.base_reason)}.",
             f"- `destination_state`: `{comparison.destination_state}`. It stops matching when this skill's files or "
             "listing at the destination, its base or the local snapshot change.", ""]
    for title, wanted in (("Needs a decision", True), ("Other changes", False)):
        rows = [row for row in (*comparison.paths, *comparison.fields) if needs_decision(row) is wanted]
        if rows:
            lines += [f"## {title}", ""] + [
                f"- `{row.get('path') or comparison.metadata_path + ' ' + row['field']}`: "
                f"{row.get('effect') or row.get('change')}; {_UPSTREAM_WORDS[row['upstream']]}" for row in rows] + [""]
    if comparison.fields:
        lines += [f"## Listing fields in `{comparison.metadata_path}`", ""]
        for row in comparison.fields:
            lines.append(f"- `{row['field']}` ({row['change']}; {_UPSTREAM_WORDS[row['upstream']]})")
            lines += [f"  - {label}: {json.dumps(row[side], ensure_ascii=False, sort_keys=True)}"
                      for side, label in (("head", "destination now"), ("sent", "this publication"), ("base", "base"))
                      if side in row]
        lines.append("")
    if comparison.paths:
        lines += ["## Diffs", ""]
    for row in comparison.paths:
        path = row["path"]
        lines += [f"### `{path}` ({row['effect']}; {_UPSTREAM_WORDS[row['upstream']]})", ""]
        now, then = read(row["head"]), read(row.get("base"))
        if row["upstream"] == CHANGED:
            failed = next((value for value in (then, now) if isinstance(value, Exception)), None)
            lines += [f"(content at the base or the destination unavailable: {failed})", ""] if failed else [
                f"At the destination since the base (`{base[:12]}` → `{head[:12]}`):", "",
                _diff(then, now, f"base/{path}", f"destination/{path}")]
        lines += [f"(destination content unavailable: {now})", ""] if isinstance(now, Exception) else [
            "This publication (destination → local copy):", "",
            _diff(now, sent.get(path, b""), f"destination/{path}", f"local/{path}")]
    return "\n".join(lines).rstrip() + "\n"


def task_destination(ctx: Any) -> Any:
    """The destination the owner chose for a publish task (``skill_publish_target.destination``)."""
    metadata = getattr(ctx, "task_metadata", None)
    target = metadata.get("skill_publish_target") if isinstance(metadata, Mapping) else None
    value = target.get("destination") if isinstance(target, Mapping) else None
    return value if isinstance(value, (Mapping, str)) else None


__all__ = [
    "MODES", "DestinationComparison", "DestinationReads", "PublicationPlan", "PublishDestination",
    "PublishListingError", "captured_release_digest", "catalog_row", "compare_destination", "comparison_report",
    "decision_hint", "destination_decision", "entry_bytes", "git_blob_oid", "github_repository", "hub_entry",
    "hub_entry_listing", "installed_base", "needs_decision", "plan_publication", "replacement_section",
    "resolve_publish_destination", "task_destination", "update_catalog",
]
