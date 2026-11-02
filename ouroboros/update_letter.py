"""Update description: Ouroboros's own account of what a pending official update brings.

No host code writes the prose and no check calls a model. A fetching check (boot, or the
Updates panel's check) records facts only: the range it found (``checked_key``), that range's
exact qualification, and the range material — every commit reachable from the official
target but not from the running base, including merged branches, plus the README "Version
History" rows ADDED along the target's first-parent line. Rows are recovered from the commit
diffs rather than read from any README snapshot, because the history table is capped (rows
roll off inside one range) and untagged releases have no tag to look up. Subjects in the
reverse range describe local-only history separately, not incoming changes or proof of
preserved code. The material lives beside the record (``state/update_material.json``, one
current range), so the record every round and status read parses stays small.

The prose is written by the main mind as ordinary work: on the owner's request from the
Updates panel (one ordinary Main task, ``request_description``) or on its own initiative,
reading the material with ``query_data(kind='update_material')`` and unclear changes with
``vcs_read``, and saved through the ``record_update_description`` tool (``save_description``).
Floor (host code): exact SHAs, the ranges a description may name, one revision every writer
compares, host-derived authorship, and a description that is never deleted after its update
lands. Ceiling (the mind): the shape, length and language of the text and what matters to
this human. The panel renders it through the same sanitizing markdown pipeline the chat uses.

Storage is one record, ``data/state/update_letter.json``: the saved description with its own
range, the last check's facts and the latest owner request. Every write holds one
cross-process lock (``state/update_letter.lock``); Git, GitHub and model work stay outside it
and are compared on write. One reader shape, ``project_letter``, serves the Updates panel
payload, the Apply preflight and the agent's Runtime context (``official_update_projection``).
Exact ranges and recorded ancestry keep Git off hot reads. The panel uses the captured
server-generation baseline; task context retains its own checkout origin. Separate ancestry
witnesses prevent a merge landed on disk from claiming that the still-running server adopted it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pathlib
import re
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros.update_channels import normalize_update_channel
from ouroboros.utils import atomic_write_json, read_json_dict, truncate_within_limit, utc_now_iso

log = logging.getLogger(__name__)

RECORD_REL = pathlib.Path("state") / "update_letter.json"
MATERIAL_REL = pathlib.Path("state") / "update_material.json"
LOCK_REL = pathlib.Path("state") / "update_letter.lock"
# Per-commit body bound inside the material (a disclosed cut, never a silent slice).
COMMIT_BODY_MAX_CHARS = 10000
# EVERY commit subject in the range reaches the author; only the bodies and the older
# release-row texts are bounded, because a long range must not become an invisible one.
DEFAULT_MAX_BODIES = 200
DEFAULT_MAX_ROWS = 60
# Local history accumulates updater merges independently of the incoming release.
LOCAL_SUBJECT_PREVIEW = 20
# A record writer holds the lock only to read, compare and write JSON, so a wait this long
# means a stuck holder, which the owner-aware lock evicts once its process is gone.
RECORD_LOCK_WAIT_SEC = 10.0
# The Runtime context's pointer while an update is on offer: where the range is, and the one writer.
DESCRIPTION_SOURCES = ("query_data(op='json', kind='update_material', select='text', max_bytes=<letter.material"
                       ".text_bytes>); vcs_read(root='system_repo'); save with record_update_description")
_SHA_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_KEY_FIELDS = ("base_sha", "target_sha", "update_channel", "target_ref")

def _row_re() -> "re.Pattern":
    """An added README history row, recognised with the repository's ONE version grammar.

    Retyping it here drifted: this regex used to reject a real ``4.50.0rc1`` row and accept
    a ``4.50.0-foo`` that is not a version at all, which then counted a valid release as an
    omission. ``release_sync`` owns the grammar because it WRITES those rows.
    """
    from ouroboros.tools.release_sync import PRE_SUFFIX

    return re.compile(r"^\+\|\s*(\d+\.\d+\.\d+" + PRE_SUFFIX + r")\s*\|", re.IGNORECASE)
# A README carries several pipe tables, and `git log -p -U0` gives no section context to
# tell them apart. What identifies a RELEASE row is its own first cell: a release row opens
# with a version. So a row whose first cell starts with a digit is a release row — read it,
# or disclose that it could not be read — and any other added row (a provider table, the
# history table's own header and separator) belongs to something else and is passed over
# in silence. Counting those would tell the author a release was withheld when none was.
_ROW_CANDIDATE_RE = re.compile(r"^\+\|\s*\d")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
# One fetching check at a time in this process: a second one waits and then records its own
# facts, so two checks never read GitHub for the same range at once.
_CHECK_FLIGHT = threading.Lock()
# Owner requests come only through this server process; one at a time, so two presses can
# never admit two author tasks for one recorded request.
_REQUEST_FLIGHT = threading.Lock()

GitCapture = Callable[[List[str]], Tuple[int, str, str]]


class MaterialUnavailable(RuntimeError):
    """The range could not be read at all (git failed), which is NOT an empty range.

    Collapsing the two would record "this update has nothing to say" over a repository
    the host could not read — a silent, durable falsehood in the record.
    """


class RecordBusy(RuntimeError):
    """The record lock stayed held past ``RECORD_LOCK_WAIT_SEC``: nothing was written."""


# ---------------------------------------------------------------------------
# Material: the update range as evidence
# ---------------------------------------------------------------------------

def _default_git() -> GitCapture:
    """Local git under the managed-update ceiling (config SSOT); never an unbounded wait
    while the single-flight lock is held."""
    from ouroboros.update_channels import get_managed_update_fetch_timeout_sec
    from supervisor.git_ops import git_capture

    limit = float(get_managed_update_fetch_timeout_sec())
    return lambda cmd: git_capture(cmd, timeout=limit)


def _split_row(line: str) -> Optional[Tuple[str, str, str]]:
    """Split one added history row into (version, date, text) on UNESCAPED pipes.

    Rows carry escaped pipes in prose (``incomplete\\|unknown``), so a raw split
    would shred them. Returns ``None`` when the row does not have exactly three
    cells — the caller counts that as a disclosed omission, never a silent skip.
    """
    cells = [cell.strip() for cell in _UNESCAPED_PIPE.split(line.lstrip("+").strip())]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    if len(cells) != 3:
        return None
    return cells[0], cells[1], cells[2].replace("\\|", "|")


def _commit_subjects(capture: GitCapture, spec: str) -> Tuple[List[Dict[str, Any]], int]:
    """Whole subject inventory of one exact range; malformed records stay counted."""
    rc, out, err = capture(["git", "log", "--format=%H%x1f%aI%x1f%s%x1e", spec])
    if rc != 0:
        raise MaterialUnavailable(f"git log {spec} failed (rc={rc}): {str(err)[:200]}")
    commits, omitted = [], 0
    for chunk in out.split("\x1e") if out.strip() else []:
        if not chunk.strip():
            continue
        parts = chunk.strip("\n").split("\x1f", 2)
        if len(parts) != 3:
            omitted += 1
            continue
        commits.append({"sha": parts[0].strip(), "date": parts[1].strip(),
                        "subject": parts[2].strip(), "body": ""})
    return commits, omitted


def collect_range_material(
    base_sha: str,
    target_sha: str,
    *,
    git: Optional[GitCapture] = None,
    max_bodies: int = DEFAULT_MAX_BODIES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Dict[str, Any]:
    """Collect all base..target commits and primary-line README history additions.

    A release row is recognised by its own first cell (a version), so rows of the README's
    other tables are passed over rather than reported as dropped releases.

    ``commits`` are EVERY commit of the range, in Git's default newest-first order: the subject of
    each one always reaches the author, so a long range is never an invisible one. Only
    the bodies are bounded — the newest ``max_bodies`` are read at all, ``bodies_omitted``
    discloses how many older ones were not. ``releases`` are rows added along the target's
    first-parent line, excluding transient branch rows removed before merge,
    newest-first with first-wins per version; the newest ``max_rows`` keep their text and
    older rows stay as version and date (``rows_summarized``), malformed ones are counted
    in ``omitted_rows`` with the commits that carried them in ``omitted_row_commits``, so an
    omission always names where to read it. ``versions`` are the VERSION files at both ends.
    ``local_commits`` keeps target..base separately. The request shows a counted newest-subject
    preview with that exact Git range, not proof of unique code or a preservation promise.
    """
    capture = git or _default_git()
    from ouroboros.update_qualification import commit_qualification
    spec = f"{base_sha}..{target_sha}"
    material: Dict[str, Any] = {
        "base_sha": base_sha, "target_sha": target_sha,
        "commits": [], "bodies_omitted": 0, "omitted_commit_chunks": 0, "releases": [],
        "omitted_rows": 0, "omitted_row_commits": [], "rows_summarized": 0, "versions": {},
    }
    # Two reads on purpose: the subject line of EVERY commit (small, unbounded by design) and
    # the bodies of only the newest ``max_bodies`` — git never hands this process the bodies
    # of a whole long history just so most of them can be dropped again.
    commits, material["omitted_commit_chunks"] = _commit_subjects(capture, spec)
    # Match the post-merge workflow's integration line, including squash commits.
    rc, out, err = capture(["git", "rev-list", "--first-parent", spec])
    if rc != 0:
        raise MaterialUnavailable(f"git rev-list --first-parent {spec} failed (rc={rc}): {str(err)[:200]}")
    by_sha = {c["sha"]: c for c in commits}
    material["first_parent_commits"] = [by_sha[sha] for sha in out.split() if sha in by_sha]
    material["local_commits"], material["omitted_local_chunks"] = _commit_subjects(
        capture, f"{target_sha}..{base_sha}")
    bodies: Dict[str, str] = {}
    if commits and max_bodies > 0:
        rc, out, err = capture([
            "git", "log", "-n", str(int(max_bodies)), "--format=%H%x1f%b%x1e", spec,
        ])
        if rc != 0:
            raise MaterialUnavailable(f"git log bodies {spec} failed (rc={rc}): {str(err)[:200]}")
        for chunk in out.split("\x1e") if out.strip() else []:
            parts = chunk.strip("\n").split("\x1f", 1)
            if len(parts) == 2 and parts[0].strip():
                bodies[parts[0].strip()] = parts[1].strip()
        for commit in commits[:max_bodies]:
            body = bodies.get(commit["sha"], "")
            commit["qualification"] = commit_qualification(body)
            commit["body"] = truncate_within_limit(body, COMMIT_BODY_MAX_CHARS) if body else ""
    for commit in commits:
        commit["body_read"] = commit["sha"] in bodies
    material["bodies_omitted"] = max(0, len(commits) - max_bodies)
    material["commits"] = commits

    # Release rows describe the target's primary line, not temporary branch-local entries.
    # Commit evidence above independently covers the complete graph. Full/sparse history
    # prevents path simplification from hiding primary-line merges for a divergent base.
    # `--diff-merges=first-parent`, never `-m`: `-m` diffs a merge against EVERY parent, and the
    # second-parent comparison of an official merge re-emits rows that were already on the
    # first-parent line — presenting an old release as added inside this range.
    rc, out, err = capture([
        "git", "log", "--first-parent", "--full-history", "--sparse",
        "--diff-merges=first-parent", "-p", "-U0",
        "--format=%x01%H", spec, "--", "README.md",
    ])
    if rc != 0:
        raise MaterialUnavailable(f"git log -p README.md {spec} failed (rc={rc}): {str(err)[:200]}")
    seen_versions: set = set()
    releases: List[Dict[str, Any]] = []
    row_re = _row_re()
    if out:
        current_sha = ""
        for line in out.splitlines():
            if line.startswith("\x01"):
                current_sha = line[1:].strip()
                continue
            if not _ROW_CANDIDATE_RE.match(line):
                continue
            cells = _split_row(line)
            if cells is None or not row_re.match(line):
                # Wrong cell count, or a first cell that is not a version: either way this
                # row said something the letter's author would otherwise never learn. The
                # commit it came from travels with the count, so the omission is resolvable.
                material["omitted_rows"] += 1
                if current_sha and current_sha not in material["omitted_row_commits"]:
                    material["omitted_row_commits"].append(current_sha)
                continue
            version, date, text = cells
            if version in seen_versions:
                continue
            seen_versions.add(version)
            releases.append({"version": version, "date": date, "text": text, "commit": current_sha})
    for row in releases[max_rows:]:
        row["text"] = ""
        material["rows_summarized"] += 1
    material["releases"] = releases
    for label, sha in (("base", base_sha), ("target", target_sha)):
        material["versions"][label] = _version_at(capture, sha)
    return material


def _version_at(capture: GitCapture, sha: str) -> str:
    rc, out, _err = capture(["git", "show", f"{sha}:VERSION"])
    return out.strip() if rc == 0 else ""


def material_text(material: Dict[str, Any]) -> str:
    """Render the material as compact evidence text for its reader (releases first)."""
    lines: List[str] = []
    releases = material.get("releases") or []
    # The disclosures live OUTSIDE the `releases` branch: a range whose every candidate row
    # was malformed has no rows to print and the omission is exactly what must still be said.
    if releases or material.get("omitted_rows"):
        lines.append("Release notes added along the target's first-parent line in this range (newest first):")
        for row in releases:
            # The row's own commit travels with it: a row whose text is not rendered here
            # still names where to read it in full.
            head = f"- {row.get('version')} ({row.get('date')}, added in {row.get('commit') or 'unknown commit'})"
            lines.append(f"{head}: {row['text']}" if row.get("text") else head)
        if material.get("omitted_rows"):
            where = ", ".join(material.get("omitted_row_commits") or [])
            lines.append(
                f"- [{material['omitted_rows']} unreadable history row(s) omitted"
                + (f"; read them in {where}]" if where else "]")
            )
        if material.get("rows_summarized"):
            lines.append(f"- [the oldest {material['rows_summarized']} row(s) above carry version and date only]")
    commits = material.get("commits") or []
    # Same rule as the rows: a range whose every commit record came back unreadable has no
    # commits to print, and that omission is exactly what must still be said — never
    # "(no commits in this range)".
    if commits or material.get("omitted_commit_chunks"):
        lines.append("")
        lines.append("Commits in this range, including merged branches (newest first, every one of them):")
        for commit in commits:
            # The FULL sha, not a display prefix: the subject is a summary, the sha is the
            # only thing that makes the rest of that commit retrievable.
            lines.append(
                f"- {commit.get('sha') or ''} {commit.get('date', '')[:10]} {commit.get('subject', '')}"
            )
            body = str(commit.get("body") or "").strip()
            if body:
                lines.append("  " + body.replace("\n", "\n  "))
        if material.get("omitted_commit_chunks"):
            lines.append(f"- [{material['omitted_commit_chunks']} commit record(s) git returned unreadably]")
        if material.get("bodies_omitted"):
            lines.append(
                f"- [the bodies of the {material['bodies_omitted']} oldest commit(s) were not read; "
                "their subjects are all listed above]"
            )
    local = material.get("local_commits") or []
    if local or material.get("omitted_local_chunks"):
        lines.append("\nExisting local history absent from the official target (target..base; not incoming changes):")
        lines.extend(f"- {c['sha']} {c['date']} {c['subject']}" for c in local[:LOCAL_SUBJECT_PREVIEW])
        if len(local) > LOCAL_SUBJECT_PREVIEW:
            lines.append(f"[{len(local) - LOCAL_SUBJECT_PREVIEW} older local subjects omitted from this preview; "
                         f"read the full range with git log {material['target_sha']}..{material['base_sha']}]")
        lines.append("These subjects include local updater merges; they do not prove unique code or preservation after apply.")
        if material.get("omitted_local_chunks"):
            lines.append(f"[{material['omitted_local_chunks']} local records unreadable; inspect "
                         f"{material['target_sha']}..{material['base_sha']}]")
    if not lines:
        lines.append("(no commits in this range)")
    return "\n".join(lines)


# Where the rest of the range is, in the reader's own tools: the material is a view.
READ_RECIPE = """How to read further (the installation's own repository holds {base}..{target}):
- One commit with its patch: vcs_read(root="system_repo", action="show", ref="<sha>"); add path="<file>" to narrow it.
- The whole range as one tree diff: vcs_read(root="system_repo", action="diff", base="{base}", head="{target}", stat=true), then path="<file>" for one file.
- What one merged branch brought: vcs_read(root="system_repo", action="diff", base="<merge sha>^1", head="<merge sha>", stat=true).
- Plain Git in that repository: git log --format='%H %s%n%n%b' {base}..{target}"""
# Named only where the record's qualification is this very range's (a check replaces it).
CHECK_FACTS = ("Check facts for this exact range (receipts, post-merge runs; missing evidence is not a pass): "
               "query_data(op=\"json\", kind=\"update_letter\", select=\"qualification\").")


def material_path(drive_root: Optional[pathlib.Path] = None) -> pathlib.Path:
    from ouroboros.config import DATA_DIR

    return pathlib.Path(drive_root or DATA_DIR) / MATERIAL_REL


def _material_document(key: Dict[str, str], material: Dict[str, Any]) -> Dict[str, Any]:
    """The one current range's material as its reader receives it: the counts and
    disclosures as fields, the commits and rows as one text with its read recipe.
    ``text_bytes`` is the reader's own count of ``select='text'`` (``query_data``), so it is
    exactly the ``max_bytes`` that returns the text whole."""
    from ouroboros.tools.query_data import selected_bytes

    commits = material.get("commits") or []
    text = (material_text(material) + "\n\n" + READ_RECIPE.format(base=key["base_sha"], target=key["target_sha"])
            + "\n- " + CHECK_FACTS)
    return {"schema": 1, "key": dict(key), "collected_at": utc_now_iso(), "text_bytes": selected_bytes(text),
            "versions": dict(material.get("versions") or {}), "commits": len(commits),
            "bodies_read": sum(1 for c in commits if c.get("body_read")), "body_max_chars": COMMIT_BODY_MAX_CHARS,
            **{name: len(material.get(name) or []) for name in ("first_parent_commits", "releases", "local_commits")},
            "bodies_omitted": int(material.get("bodies_omitted") or 0), "text": text}


def _stored_material(drive_root: Optional[pathlib.Path]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(document, sha256 of its bytes)`` of the stored material, ``(None, "")`` when absent."""
    try:
        raw = material_path(drive_root).read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None, ""
    return (document, hashlib.sha256(raw).hexdigest()) if isinstance(document, dict) else (None, "")


# ---------------------------------------------------------------------------
# Storage: one record, one cross-process lock
# ---------------------------------------------------------------------------

def record_path(drive_root: Optional[pathlib.Path] = None) -> pathlib.Path:
    from ouroboros.config import DATA_DIR

    return pathlib.Path(drive_root or DATA_DIR) / RECORD_REL


def read_record(drive_root: Optional[pathlib.Path] = None) -> Optional[Dict[str, Any]]:
    record = read_json_dict(record_path(drive_root))
    return record if record and isinstance(record.get("key"), dict) else None


@contextlib.contextmanager
def _record_lock(drive_root: Optional[pathlib.Path]) -> Iterator[None]:
    """The one writer lock of the record, shared by the server's check and request paths and
    a worker's tool write. Holders only read, compare and write JSON."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    path = record_path(drive_root).parent / LOCK_REL.name
    fd = acquire_exclusive_file_lock(path, timeout_sec=RECORD_LOCK_WAIT_SEC, owner_aware_stale=True)
    if fd is None:
        raise RecordBusy("the update description record stayed locked")
    try:
        yield
    finally:
        release_exclusive_file_lock(path, fd)


def _revision(record: Optional[Dict[str, Any]]) -> int:
    value = (record or {}).get("revision")
    return value if type(value) is int and value >= 0 else 0


def runtime_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """Add the startup source baseline without changing checkout/apply pins.

    Target validation belongs to git_ops_updates, including a consumed target.
    This reader never fetches or reconstructs that cache's authority.
    """
    from ouroboros.server_process import server_source_baseline
    from ouroboros.update_generation_facts import projection

    return {**status, "running_sha": server_source_baseline(), "after_update": projection()}


def _key_from_status(status: Dict[str, Any]) -> Dict[str, str]:
    return {
        "base_sha": str(status.get("running_sha", status.get("current_sha")) or ""),
        "target_sha": str(status.get("checked_target_sha") or status.get("latest_sha") or ""),
        "update_channel": str(status.get("update_channel") or ""),
        "target_ref": str(status.get("target_ref") or ""),
    }


def _range_key(candidate: Any) -> Optional[Dict[str, str]]:
    """A key naming a real, non-empty range, else ``None``."""
    if not isinstance(candidate, dict):
        return None
    key = {field: str(candidate.get(field) or "") for field in _KEY_FIELDS}
    return key if key["base_sha"] and key["target_sha"] and key["base_sha"] != key["target_sha"] else None


def _same_range(key: Optional[Dict[str, Any]], base_sha: str, target_sha: str) -> bool:
    return bool(key) and (key.get("base_sha"), key.get("target_sha")) == (base_sha, target_sha)


# ---------------------------------------------------------------------------
# The fetching check: facts only, never prose
# ---------------------------------------------------------------------------

def refresh_after_check(
    status: Dict[str, Any],
    *,
    drive_root: Optional[pathlib.Path] = None,
    git: Optional[GitCapture] = None,
) -> Optional[Dict[str, Any]]:
    """Record what a successful fetching check learned; the description is left as it is.

    The source is the server generation's captured baseline, independently of the checkout
    used to apply an update. Git and GitHub are read before the record lock and the facts
    are written onto whatever the record holds then, so a description saved meanwhile
    stays. A failed or passive check changes nothing.
    """
    try:
        if status.get("check_ok") is not True:
            return read_record(drive_root)
        from ouroboros.update_channels import get_managed_update_fetch_timeout_sec

        if not _CHECK_FLIGHT.acquire(timeout=float(get_managed_update_fetch_timeout_sec())):
            return read_record(drive_root)
        try:
            return _record_check(status, drive_root, git or _default_git())
        finally:
            _CHECK_FLIGHT.release()
    except Exception:
        log.debug("update description check facts were not recorded", exc_info=True)
        return read_record(drive_root)


def _record_check(status: Dict[str, Any], drive_root: Optional[pathlib.Path], git: GitCapture) -> Dict[str, Any]:
    key = _key_from_status(status)
    checkout_sha = str(status.get("current_sha") or "")
    running_sha = str(status.get("running_sha", key["base_sha"]) or "")
    before = read_record(drive_root) or {}
    no_update = (not key["target_sha"] or key["base_sha"] == key["target_sha"]
                 or (not status.get("available") and key["base_sha"] == checkout_sha))
    facts: Dict[str, Any] = {}
    material: Optional[Dict[str, Any]] = None
    if not key["base_sha"]:
        facts["material_error"] = {"kind": "runtime_source_unavailable", "key": key, "at": utc_now_iso(),
                                   "text": "The running server's source baseline is unavailable; no range was read."}
    elif not no_update:
        stored, _digest = _stored_material(drive_root)
        recorded = _same_range(before.get("qualification"), key["base_sha"], key["target_sha"])
        try:
            if not (recorded and stored and _same_range(stored.get("key"), key["base_sha"], key["target_sha"])):
                material = collect_range_material(key["base_sha"], key["target_sha"], git=git)
        except MaterialUnavailable as exc:
            facts["material_error"] = {"kind": "material_unavailable", "key": key, "at": utc_now_iso(),
                                       "text": _safe_text(f"{type(exc).__name__}: {exc}")}
        # The runs of a range already recorded refresh without its material; a new range needs it.
        if recorded or material is not None:
            facts["qualification"] = _check_qualification(before, key, material, str(status.get("official_repo_url") or ""))
    shown_key = shown_letter(before)[0].get("key") if before else None
    shown_target = str((shown_key or {}).get("target_sha") or "") if before.get("state") != "none" else ""
    facts["ancestry"] = {
        "target_sha": shown_target,
        "in_running": _target_in_head(shown_target, running_sha, git),
        "in_head": _target_in_head(shown_target, checkout_sha, git) if checkout_sha != running_sha else None,
    }
    target_version = _version_at(git, key["target_sha"] or key["base_sha"])
    from ouroboros.update_qualification import post_merge_reds

    facts["post_merge_reds"] = post_merge_reds(str(status.get("official_repo_url") or ""))
    with _record_lock(drive_root):
        current = read_record(drive_root)
        record = _with_check_facts(current, key, facts, running_sha=running_sha, checkout_sha=checkout_sha,
                                   target_version=target_version)
        if material is not None and (material.get("commits") or material.get("releases")):
            document = _material_document(key, material)
            atomic_write_json(material_path(drive_root), document)
            # What the record says about the material beside it: its range, size and when it was read.
            record["material"] = {name: document[name] for name in ("key", "collected_at", "text_bytes", "commits",
                                                                    "bodies_read", "bodies_omitted")}
        atomic_write_json(record_path(drive_root), record)
    return record


def _check_qualification(before: Dict[str, Any], key: Dict[str, str], material: Optional[Dict[str, Any]],
                         repo_url: str) -> Optional[Dict[str, Any]]:
    """The exact-range evidence: refreshed in place for a range already recorded, built from
    new material otherwise. It never touches the description: a changed run is a new fact
    about the same range, not a reason to rewrite what the range does."""
    from ouroboros.update_qualification import range_qualification, refresh_qualification

    previous = before.get("qualification")
    if isinstance(previous, dict) and _same_range(previous, key["base_sha"], key["target_sha"]):
        return refresh_qualification(previous, repo_url, previous=previous)[0]
    if material is None:
        return previous if isinstance(previous, dict) else None
    return range_qualification(material, repo_url)


def _with_check_facts(current: Optional[Dict[str, Any]], key: Dict[str, str], facts: Dict[str, Any], *,
                      running_sha: str, checkout_sha: str, target_version: str) -> Dict[str, Any]:
    """Pure: the record with one check's facts. The description fields are never touched."""
    record = dict(current) if current else {"schema": 1, "state": "none", "text": "", "last_good": None}
    record["checked_key"] = dict(key)
    record["checked_target_version"] = target_version
    if record.get("state") == "none":
        # A letterless record follows the check it describes, so the Runtime fact can name
        # the version an up-to-date install is current with.
        record["key"] = dict(key)
        record["target_version"] = target_version
    if facts.get("qualification") is not None:
        record["qualification"] = facts["qualification"]
    from ouroboros.update_qualification import kept_reds

    # A failed read keeps the observation the record holds under the lock, never "none open".
    record["post_merge_reds"] = kept_reds(facts.get("post_merge_reds"), record.get("post_merge_reds"))
    if facts.get("material_error"):
        record["material_error"] = facts["material_error"]
    else:
        record.pop("material_error", None)
    # The ancestry witnesses describe the description shown when the check read Git; a
    # description saved since names another target, and its facts wait for the next check.
    ancestry = facts.get("ancestry") or {}
    shown_target = str((shown_letter(record)[0].get("key") or {}).get("target_sha") or "")
    computed_for = str(ancestry.get("target_sha") or "")
    valid = record.get("state") != "none" and computed_for == shown_target
    record["checked_running_sha"] = running_sha
    record["target_in_running"] = bool(valid and ancestry.get("in_running"))
    record["checked_head_sha"] = checkout_sha
    in_head = ancestry.get("in_running") if ancestry.get("in_head") is None else ancestry.get("in_head")
    record["target_in_head"] = bool(valid and in_head)
    record["target_in_for"] = computed_for
    return record


def _target_in_head(target: str, head_sha: str, git: GitCapture) -> bool:
    """Whether ``target`` is already inside ``head_sha`` without being it.

    Answered here, at the check, because the check runs where git is — the Updates panel and
    the Runtime context then read this ONE recorded fact instead of each proving it their own
    way, which is what kept the two surfaces describing one install differently. A divergent
    install applies an official target as a merge commit (``supervisor/update_merge.py``), so
    the target is never HEAD itself; ancestry is the proof.
    """
    head = str(head_sha or "")
    if not target or not head or target == head:
        return False
    try:
        return git(["git", "merge-base", "--is-ancestor", target, head])[0] == 0
    except Exception:  # noqa: BLE001 — an unprovable ancestry is simply not proof
        log.debug("target-in-head ancestry check failed", exc_info=True)
        return False


def _safe_text(text: str) -> str:
    """Secret-free, bounded text for the record (the same sanitizer every log line uses)."""
    from ouroboros.utils import sanitize_tool_result_for_log

    return sanitize_tool_result_for_log(str(text or ""))[:300]


# ---------------------------------------------------------------------------
# The description writer (the record_update_description tool) and owner requests
# ---------------------------------------------------------------------------

def legal_ranges(record: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """The ranges a description may name: the one the last check offered, the shown
    description's own (a rephrase, an applied update's included), and the one the owner's
    latest request named. Anything else would label text with a range nobody offered."""
    record = record or {}
    provenance, text, _last_good = shown_letter(record)
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    keys: List[Dict[str, str]] = []
    for key in map(_range_key, (record.get("checked_key"), provenance.get("key") if text else None, request.get("key"))):
        if key and not any(_same_range(known, key["base_sha"], key["target_sha"]) for known in keys):
            keys.append(key)
    return keys


def save_description(text: str, base_sha: str, target_sha: str, expected_revision: int, *, author: Dict[str, Any],
                     drive_root: Optional[pathlib.Path] = None, git: Optional[GitCapture] = None) -> Dict[str, Any]:
    """Save one description for one exact range; returns a typed outcome, never raises for a
    refusal. ``author`` is host-derived (``task_id``, ``model``). The writer is the owner's
    recorded request when its task is that request's own work — the requested task, a host
    retry of it or a recorded Continue (``_acts_for``) — and the request keeps that it saved
    (``request_saved``) when a later description replaces this one.

    ``revision_conflict``: someone saved since the caller read ``expected_revision`` — the
    newer text stays. ``range_not_offered``: the range is none of ``legal_ranges``.
    ``saved`` carries the replaced description whole, so the writer's own durable tool
    record keeps it.
    """
    from ouroboros import get_version

    body = str(text or "").strip()
    if not body:
        raise ValueError("text is empty")
    if not (_SHA_RE.fullmatch(str(base_sha or "")) and _SHA_RE.fullmatch(str(target_sha or ""))):
        raise ValueError("base_sha and target_sha must be full lowercase commit SHAs")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("expected_revision must be a non-negative integer")
    target_version = _version_at(git or _default_git(), target_sha)
    stored, digest = _stored_material(drive_root)
    source = ({"material_sha256": digest, "material_collected_at": str(stored.get("collected_at") or "")}
              if stored and _same_range(stored.get("key"), base_sha, target_sha) else {})
    task_id, model = str(author.get("task_id") or ""), str(author.get("model") or "")
    # Task results are read before the lock, like Git; the request is compared on write.
    asked = _request_task_id(read_record(drive_root))
    acts_for = asked if _acts_for(drive_root, asked, task_id) else ""
    with _record_lock(drive_root):
        current = read_record(drive_root) or {}
        revision = _revision(current)
        shown, shown_text, _last_good = shown_letter(current) if current else ({}, "", None)
        if expected_revision != revision:
            return {"outcome": "revision_conflict", "revision": revision,
                    "saved": _public_description(shown, shown_text)}
        key = next((k for k in legal_ranges(current) if _same_range(k, base_sha, target_sha)), None)
        if key is None:
            return {"outcome": "range_not_offered", "revision": revision, "legal_ranges": legal_ranges(current)}
        asked = _request_task_id(current)
        requested = bool(asked) and asked in (task_id, acts_for)
        record = {**current, "schema": 1, "key": key, "state": "ready", "text": body, "revision": revision + 1,
                  "author_version": get_version(), "target_version": target_version, "model": model,
                  "author": {"task_id": task_id, "model": model, "requested": requested},
                  "written_at": utc_now_iso(), "source": source, "error_kind": "", "error_text": "",
                  "attempt_id": "", "attempt_ids": [], "last_good": None,
                  **({"request_saved": asked} if requested else {})}
        atomic_write_json(record_path(drive_root), record)
    return {"outcome": "saved", "revision": record["revision"], "key": key, "written_at": record["written_at"],
            "replaced": _public_description(shown, shown_text)}


def _public_description(provenance: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    return {"text": text, "key": dict(provenance.get("key") or {}), "written_at": str(provenance.get("written_at") or ""),
            "revision": _revision(provenance), "author": provenance.get("author") or {},
            "author_version": str(provenance.get("author_version") or "")}


def request_description(mode: str, base_sha: str, target_sha: str, *, admit: Callable[[str, str, Dict[str, Any]], str],
                        drive_root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Admit ONE ordinary Main task that writes (or rewrites) the description of an offered
    range, then record it as the owner's latest request.

    ``admit(task_id, text, metadata)`` is the producer's own admission (the gateway passes the
    queue's receipt-bearing enqueue) and returns ``""`` or the refusal reason. The task id is
    minted here, never derived from content. A request whose task has not ended refuses a
    second one; nothing repeats on its own after a stop, a failure or an unknown end.
    """
    import uuid

    if mode not in ("write", "rewrite"):
        raise ValueError("mode must be write or rewrite")
    with _REQUEST_FLIGHT:
        current = read_record(drive_root) or {}
        provenance, text, _last_good = shown_letter(current) if current else ({}, "", None)
        offered = [_range_key(current.get("checked_key")), _range_key(provenance.get("key")) if text else None]
        key = next((k for k in offered if k and _same_range(k, base_sha, target_sha)), None)
        if key is None:
            return {"outcome": "range_not_offered", "legal_ranges": [k for k in offered if k]}
        prior = _request_view(current, drive_root, with_status=True)
        if prior and prior["status"] in _LIVE_TASK_STATUSES:
            return {"outcome": "in_flight", "task_id": prior.get("current_task_id", prior["task_id"]), "status": prior["status"]}
        task_id = uuid.uuid4().hex[:8]
        request = {"task_id": task_id, "mode": mode, "key": key, "requested_at": utc_now_iso(),
                   "revision_at_request": _revision(current)}
        # Recorded before admission, so the task's own save already finds itself requested;
        # a refused admission puts the previous request back.
        _swap_request(drive_root, request, expected=current.get("request"))
        refusal = admit(task_id, author_request_text(current, request), {"update_description": dict(request)})
        if refusal:
            _swap_request(drive_root, current.get("request"), expected=request)
            return {"outcome": "refused", "reason": refusal}
    return {"outcome": "queued", "task_id": task_id, "key": key}


def _swap_request(drive_root: Optional[pathlib.Path], request: Any, *, expected: Any) -> None:
    """Replace the recorded request only while it is still ``expected``. Request writers are
    serialized by ``_REQUEST_FLIGHT``; the lock keeps a concurrent description write whole."""
    with _record_lock(drive_root):
        latest = read_record(drive_root)
        if latest is not None and latest.get("request") == expected:
            latest.pop("request", None)
            atomic_write_json(record_path(drive_root), {**latest, **({"request": request} if request else {})})


# Task statuses during which a request holds the door: its task may still save. Every other
# answer — an end, a missing or an unreadable result — lets the owner ask again explicitly;
# the revision every writer compares keeps a late author from erasing a newer description.
_LIVE_TASK_STATUSES = frozenset({"requested", "scheduled", "running", "interrupted", "cancel_requested"})


def _request_task_id(record: Optional[Dict[str, Any]]) -> str:
    request = (record or {}).get("request")
    return str(request.get("task_id") or "") if isinstance(request, dict) else ""


def _acts_for(drive_root: Optional[pathlib.Path], task_id: str, actor_task_id: str) -> bool:
    """Whether ``actor_task_id`` is ``task_id``'s own work, by the existing owner of that question
    (the task itself, its host retry attempts, recorded Continue claims). Unreadable lineage is no."""
    from ouroboros.owner_quiz import question_task_owned_by

    try:
        return bool(task_id and actor_task_id) and question_task_owned_by(
            record_path(drive_root).parent.parent, task_id, actor_task_id)
    except Exception:  # noqa: BLE001 — authorship that cannot be proven is not claimed
        log.debug("update description request lineage unreadable", exc_info=True)
        return False


def _request_view(record: Optional[Dict[str, Any]], drive_root: Optional[pathlib.Path], *,
                  with_status: bool) -> Optional[Dict[str, Any]]:
    """Read the request's current work through the existing retry and admitted-Continue owners.
    Keep its original handle and physical history; ``current_task_id`` names whose status and
    terminal event the panel follows. Hot readers ask without status."""
    task_id = _request_task_id(record)
    if not task_id:
        return None
    request = record["request"]
    view = {"task_id": task_id, "mode": str(request.get("mode") or ""), "key": dict(request.get("key") or {}),
            "requested_at": str(request.get("requested_at") or "")}
    if with_status:
        from ouroboros.question_answers import admitted_successor
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import effective_task_result

        root = record_path(drive_root).parent.parent
        try:
            current, seen = task_id, set()
            while current not in seen:
                seen.add(current)
                stored = load_task_result(root, current, strict=True)
                effective = effective_task_result(root, stored, materialize_artifacts=False) if isinstance(stored, dict) else {}
                current = str(effective.get("task_id") or current)
                successor = admitted_successor(root, current)
                if not successor:
                    break
                current = successor
            else:
                raise ValueError("cyclic description task continuation")
            view["current_task_id"] = current
            view["status"] = str(effective.get("status") or "unknown") if effective else "missing"
        except Exception:  # noqa: BLE001 — an unreadable result is an unknown status, never an end
            view["status"] = "unknown"
        view["saved"] = str(record.get("request_saved") or "") == task_id
    return view


def author_request_text(record: Dict[str, Any], request: Dict[str, Any]) -> str:
    """The owner's request as the ordinary task text: facts, sources, the one writer. The
    shape of the description is the mind's own; nothing here sets a length or a template."""
    key = request["key"]
    base, target = key["base_sha"], key["target_sha"]
    shown, text, _last_good = shown_letter(record) if record else ({}, "", None)
    shown_key = shown.get("key") or {}
    target_version = _target_version(record, target) if record else ""
    saved = ("" if not text else "A description of this range is already saved (query_data(op=\"json\", kind=\"update_letter\", "
             "select=\"text\")); my human asked for a new one, so read the range again rather than restating it. "
             if _same_range(shown_key, base, target) else f"The saved description covers another range "
             f"({shown_key.get('base_sha', '')}..{shown_key.get('target_sha', '')}); it stays until this one is saved. ")
    return (
        "[UPDATE DESCRIPTION REQUEST]\n"
        f"My human asked me in Dashboard → Updates to {'rewrite' if request['mode'] == 'rewrite' else 'write'} "
        f"the description of the official update {base}..{target} "
        f"(channel {key.get('update_channel') or 'unknown'}, ref {key.get('target_ref') or 'unknown'}"
        + (f", target version {target_version}" if target_version else "") + "). "
        + saved + "\n\n" + _request_sources(record, key) + "\n\n"
        "Write it for my human: what changes for them and for me once this update is applied, in the language we use "
        "together, as long and as structured as the change needs. Say honestly what was not verified where it matters; "
        "promise nothing about how applying will go. Save it with record_update_description("
        f"base_sha=\"{base}\", target_sha=\"{target}\", expected_revision={request['revision_at_request']}, text=...), "
        "or text_file={\"root\": ..., \"path\": ...} in place of text for a Markdown file you wrote "
        "— enable it with enable_tools if it is not loaded. If it refuses because a newer description was saved, read "
        "that one and decide. Then finish with a short note.\n"
        "[/UPDATE DESCRIPTION REQUEST]"
    )


def _request_sources(record: Dict[str, Any], key: Dict[str, str]) -> str:
    """Where the requested range is. The record holds ONE range's material and check facts,
    which a later check replaces: each is named only when it is this range's, otherwise the
    request says whose they are and gives this range's own repository recipe."""
    from ouroboros.query_data_sql import MAX_BYTES

    base, target = key["base_sha"], key["target_sha"]
    pointer = record.get("material") if isinstance(record.get("material"), dict) else {}
    facts = record.get("qualification") if isinstance(record.get("qualification"), dict) else {}
    if _same_range(pointer.get("key"), base, target):
        size = pointer.get("text_bytes")  # the reader's own count of the text (``_material_document``)
        budget = min(size, MAX_BYTES) if type(size) is int and size > 0 else MAX_BYTES
        sources = (f"Sources: query_data(op=\"json\", kind=\"update_material\", select=\"text\", max_bytes={budget}) "
                   "holds every commit of this range with the bodies it could read, the first-parent integrations, "
                   "README release rows, local history that is not part of the target, and how to read further. When a "
                   "message does not say what changed, read the change itself with vcs_read(root=\"system_repo\", ...).")
    else:
        stored = _range_text(pointer.get("key"))
        sources = ((f"The stored update material covers {stored}, not this range" if stored else
                    "No update material is stored") + "; read this range in the repository.\n"
                   + READ_RECIPE.format(base=base, target=target))
    if _same_range(facts, base, target):
        return f"{sources}\n{CHECK_FACTS}"
    stored = _range_text(facts)
    return f"{sources}\nNo check facts for this range are recorded{f' (the recorded ones cover {stored})' if stored else ''}."


def _range_text(key: Any) -> str:
    return f"{key.get('base_sha')}..{key.get('target_sha')}" if _range_key(key) else ""


def _target_version(record: Dict[str, Any], sha: str) -> str:
    """The version the record already knows for target ``sha``; '' rather than a Git read."""
    if ((record.get("checked_key") or {}).get("target_sha")) == sha:
        return str(record.get("checked_target_version") or "")
    provenance = shown_letter(record)[0] if record else {}
    return str(provenance.get("target_version") or "") if (provenance.get("key") or {}).get("target_sha") == sha else ""


# ---------------------------------------------------------------------------
# The one projection shared by the panel payload, preflight and the Runtime context
# ---------------------------------------------------------------------------

def shown_letter(record: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]]]:
    """``(provenance, text, last_good)`` — the description a reader will actually SEE.

    A record written before descriptions became the mind's own work may hold a failed
    rewrite that kept the previous good letter, which may be about an OLDER target than the
    range that failed. Relation, key, versions and every ancestry question must therefore be
    asked about this provenance and never about the outer record: asking the outer one
    presents a kept letter as the letter for the update currently on offer, and hides an
    already-applied one behind the range that failed.
    """
    last_good = record.get("last_good") if isinstance(record.get("last_good"), dict) else None
    text = str(record.get("text") or "")
    if record.get("state") != "ready" and not text and last_good:
        return last_good, str(last_good.get("text") or ""), last_good
    return record, text, last_good


def project_letter(
    record: Optional[Dict[str, Any]],
    *,
    head_sha: str,
    latest_sha: str,
    request: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Relate the stored description to the live HEAD and official target.

    ``applied`` is SHA equality, or the fact the last check recorded about this very HEAD:
    a divergent install applies an official target as a merge commit
    (``supervisor/update_merge.py``), so HEAD never equals the target it consumed, and the
    check — which runs where git is — proves the ancestry once (``target_in_head``) for
    both readers. Nothing here infers it from ``latest_sha``, which a passive read leaves
    empty exactly when the cached target has been consumed. ``description_current`` compares
    ranges only: a new post-merge run is a fact about the same range, never a stale text.
    A record without a description projects only while it carries something to act on (an
    offered range or a request); ``request`` is the caller's ``_request_view``.
    """
    if not record:
        return None
    provenance, text, last_good = shown_letter(record)
    if record.get("state") in ("none", None, ""):
        if not _range_key(record.get("checked_key")) and not request:
            return None
        provenance, text = {}, ""
    key = provenance.get("key") if isinstance(provenance.get("key"), dict) else {}
    base, target = str(key.get("base_sha") or ""), str(key.get("target_sha") or "")
    head, latest = str(head_sha or ""), str(latest_sha or "")
    consumed = (bool(record.get("target_in_head")) and head and head == str(record.get("checked_head_sha") or "")
                and record.get("target_in_for", target) == target)
    if not text:
        relation = ""
    elif head and head == target:
        relation = "applied"
    elif consumed and target:
        relation = "applied"
    elif head and head == base and latest == target:
        relation = "pending"
    elif head and head == base:
        relation = "superseded"
    else:
        relation = "other"
    failed = record.get("state") == "failed"
    material_error = record.get("material_error") if isinstance(record.get("material_error"), dict) else {}
    error_kind = str(record.get("error_kind") or material_error.get("kind") or "")
    return {
        "state": "failed" if failed else ("ready" if text else "none"),
        "relation": relation,
        "text": text,
        "author_version": str(provenance.get("author_version") or ""),
        "target_version": str(provenance.get("target_version") or ""),
        "written_at": str(provenance.get("written_at") or ""),
        "error_kind": error_kind,
        "error_text": str(record.get("error_text") or material_error.get("text") or ""),
        "key": dict(key),
        "has_last_good": bool(last_good and last_good.get("text")),
        "description_current": bool(text and base and head == base and latest == target),
        "failed_at": str(record.get("written_at") or "") if failed else str(material_error.get("at") or ""),
        "latest_failed_key": (dict(record.get("key") or {}) if failed
                              else dict(material_error.get("key") or {}) if material_error else None),
        "qualification": record.get("qualification") or provenance.get("qualification"),
        "revision": _revision(record),
        "author": dict(provenance.get("author") or {}),
        "checked_key": _range_key(record.get("checked_key")),
        "material": record.get("material") if isinstance(record.get("material"), dict) else None,
        "request": request,
    }


def project_letter_for_panel(
    status: Dict[str, Any],
    *,
    drive_root: Optional[pathlib.Path] = None,
) -> Optional[Dict[str, Any]]:
    """Project against the server baseline; task context retains its own checkout origin.

    Both use the same reader; separate recorded ancestry facts prevent checkout
    movement from claiming server adoption. No git runs on either read path; the panel
    alone reads the request's task status (one result file).
    """
    record = read_record(drive_root)
    if record and "running_sha" in status:
        record = {**record, "checked_head_sha": record.get("checked_running_sha"),
                  "target_in_head": record.get("target_in_running", False)}
    return project_letter(
        record,
        head_sha=_key_from_status(status)["base_sha"],
        latest_sha=_key_from_status(status)["target_sha"],
        request=_request_view(record, drive_root, with_status=True),
    )


def official_update_projection(
    head_sha: str,
    *,
    drive_root: Optional[pathlib.Path] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The typed fact for the Runtime context: O(1), no git, never raises.

    The retained update's before/after samples are the Runtime section's ``vitals.last_update``
    (``context_runtime_facts``); the panel reads them through ``runtime_status``.
    ``status`` is honest about freshness. It follows the CHECK, not the letter:
    ``update_available``/``up_to_date`` while the last check still describes this HEAD (the
    availability is the cache's, so an available update may carry a failed letter or none at
    all), ``up_to_date`` also when HEAD is the checked target itself, ``moved_since_check``
    when HEAD matches neither, ``unchecked`` before any fetching check has completed.
    ``status_as_of`` names the check the fact comes from.
    ``head_sha`` is the HEAD the caller already resolved for its own repository; a
    child worktree that committed reads ``moved_since_check`` for its own tree, which
    is a true statement about that tree, not about the canonical body.
    """
    try:
        from ouroboros import get_version
        from ouroboros.config import DATA_DIR
        from ouroboros.update_qualification import runtime_letter  # the full record: query_data kind update_letter

        head = str(head_sha or "")
        if not head or head == "unknown":
            # context.py's own sentinel for a git it could not read. Comparing it with real
            # SHAs would report "moved since check" about a body that has not moved at all.
            return {"status": "unknown", "error": "head_unresolved"}
        if state is None:
            # Lock-free read, like the neighbouring drive-state section: this runs on
            # every task round and consciousness cycle, never under the state lock.
            state = read_json_dict(pathlib.Path(drive_root or DATA_DIR) / "state" / "state.json") or {}
        cache = state.get("managed_update_cache") if isinstance(state, dict) else None
        cache = cache if isinstance(cache, dict) else {}
        record = read_record(drive_root)
        # The cached check describes ONE channel. After the owner switches channels it says
        # nothing about the active one (the panel asks for a new check for the same reason),
        # so the fact is honestly "unchecked" there — the letter stays, as history.
        active_channel = normalize_update_channel(os.environ.get("OUROBOROS_UPDATE_CHANNEL"))
        cached_channel = str(cache.get("update_channel") or "")
        if cached_channel and cached_channel != active_channel:
            return {
                "status": "unchecked", "status_as_of": "",
                "running": {"version": get_version(), "sha": head},
                "update_channel": active_channel, "target": None, "behind": None, "ahead": None,
                "local_changes": "unknown", "letter": runtime_letter(project_letter(
                    record, head_sha=head, latest_sha="", request=_request_view(record, drive_root, with_status=False))),
                "post_merge_reds": (record or {}).get("post_merge_reds"),
            }
        latest = str(cache.get("latest_sha") or "")
        checked_head = str((record or {}).get("checked_head_sha") or "")
        shown = shown_letter(record)[0] if record else {}
        record_key = shown.get("key") if isinstance(shown.get("key"), dict) else {}
        letter = runtime_letter(project_letter(record, head_sha=head, latest_sha=latest,
                                               request=_request_view(record, drive_root, with_status=False)))
        if not cache:
            status = "unchecked"
        elif head and head == checked_head:
            status = "update_available" if cache.get("available") else "up_to_date"
        elif head and head == latest:
            status = "up_to_date"
        else:
            status = "moved_since_check"
        target = None
        if latest:
            # Key AND version from the SAME provenance: reading the version off the outer
            # record would pair the newer failed range's version with the kept letter's sha.
            same_target = str(record_key.get("target_sha") or "") == latest
            checked_target = str(((record or {}).get("checked_key") or {}).get("target_sha") or "") == latest
            target = {
                "version": (str(shown.get("target_version") or "") if same_target
                            else str((record or {}).get("checked_target_version") or "") if checked_target else ""),
                "sha": latest,
            }
        return {
            "status": status,
            "status_as_of": str(cache.get("checked_at") or ""),
            "running": {"version": get_version(), "sha": head},
            "update_channel": str(cache.get("update_channel") or ""),
            "target": target,
            "behind": cache.get("behind"),
            "ahead": cache.get("ahead"),
            # Commits ahead are history; this is whether the checked HEAD held content of its own.
            "local_changes": str(cache.get("local_changes") or "unknown")
            if head == str(cache.get("checked_head_sha") or "") else "unknown",
            "letter": letter,
            "post_merge_reds": (record or {}).get("post_merge_reds"),
            # Where the range and the one writer are, for a mind that decides to describe it.
            **({"description_sources": DESCRIPTION_SOURCES} if status == "update_available" else {}),
        }
    except Exception as exc:  # noqa: BLE001 — a fact that raises would take the whole context with it
        log.debug("official_update projection failed", exc_info=True)
        return {"status": "unknown", "error": type(exc).__name__}
