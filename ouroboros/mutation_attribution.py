"""Task-wide mutation baselines and attributed Git candidates.

This module answers one pure evidence question: which paths were clean when
the host captured a task's baseline and changed during the task's observed
window?  The window is observational — nothing here claims or enforces
exclusive ownership of a surface. An explicitly selected predecessor can pass
its exact retained candidates to a new root; the original dirty baseline stays
recorded, and this transfer supplies no reviewer approval. The evidence lives in the existing task
result; no second ledger is introduced, and honest ambiguity (a pre-existing
dirty path that changed, a stale baseline) is reported as blockers for the
reviewing LLM panels to weigh, never as an automatic verdict.

A body candidate's surface may carry the basis its owner established
(``body_candidate``): ``original`` (observed when prepared), ``derived`` (its
clean creation) or ``carried`` (a predecessor's row). Dirt in any other
observation of such a copy is uncertain rather than known foreign: automatic
staging leaves it out, an explicit selection may carry it to review.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import stat
from typing import Any, Iterable, Mapping, Sequence

from ouroboros import work_domain
from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
from ouroboros.utils import safe_relpath, utc_now_iso

MUTATION_EVIDENCE_VERSION = 1
_GIT_SURFACE_TYPES = frozenset({"system_repo", "external_workspace"})
# Full-content hashing is bounded: a dirty artifact bigger than this records a
# size-only fingerprint. Such an observation cannot establish exact predecessor
# content and therefore never authorizes a transfer by matching size alone.
_FINGERPRINT_MAX_BYTES = 32 * 1024 * 1024
# A trashed worktree (build generations, browser caches) can list tens of
# thousands of dirty paths; storing/fingerprinting them would stall task start
# and bloat the durable task result. Past this cap the baseline is honestly
# unusable for attribution instead of expensively pretending.
_BASELINE_DIRTY_PATHS_MAX = 4096
_OBSERVED_EFFECT_STATES = frozenset({"observed_window", "quiescent"})
ESTABLISHED_BASES = frozenset({"original", "derived", "carried"})


def attribution_task_id(results_drive_root: Any, candidates: Iterable[Any], *, repo_root: Any = None,
                        surface_type: str = "") -> str:
    """Return the first task id in lineage order with a durable baseline; with ``repo_root``, one whose baseline
    observes that Git root, and with ``surface_type`` one observing a Git surface of that type (``system_repo``:
    the body, by its serving checkout or a candidate copy). A baseline of another admitted folder says nothing
    about either."""
    root = str(_canonical_root(repo_root)) if repo_root is not None else ""
    for candidate in candidates:
        task_id = str(candidate or "").strip()
        if not task_id:
            continue
        try:
            result = load_task_result(results_drive_root, task_id) or {}
        except Exception:
            continue
        evidence = result.get("mutation_evidence")
        baseline = evidence.get("baseline") if isinstance(evidence, dict) else None
        if isinstance(baseline, dict) and (not (root or surface_type) or any(
                isinstance(row, dict) and isinstance(row.get("git"), dict)
                and (not root or row.get("canonical_root") == root)
                and (not surface_type or row.get("surface_type") == surface_type)
                for row in baseline.get("surfaces") or [])):
            return task_id
    return ""


def _canonical_root(value: Any) -> pathlib.Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("mutation surface root is required")
    return work_domain.settle(pathlib.Path(text).expanduser())  # a work folder keeps its spelling; any other resolves


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_git(root: pathlib.Path, *args: str, timeout: float | None = None) -> str:
    from ouroboros.git_observation import host_git_env

    proc = work_domain.run(  # a work folder's Git, with its config and filters, runs as the work domain
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",  # Git writes paths as UTF-8 bytes, never in the locale's code page (cp1252 on Windows)
        errors="surrogateescape",
        check=False,
        env=host_git_env(),
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "git command failed").strip()
        raise RuntimeError(detail)
    return proc.stdout


_GIT_PROBE_TIMEOUT_SEC = 10.0  # one rev-parse of a surface's HEAD or top level; the status walk keeps no bound


def _current_head(root: pathlib.Path) -> str:
    """HEAD's commit, or "" on a positively unborn branch (a folder initialized with nothing committed yet), read
    with ``workspace_preflight.git_head_basis``'s bounded probes. A HEAD proven neither raises: never a guessed
    empty tree."""
    from ouroboros.workspace_preflight import git_head_basis

    head, state = git_head_basis(root)
    if state == "unknown":
        raise RuntimeError(f"HEAD of {root} is neither a readable commit nor a proven unborn branch")
    return head


def _git_basis(root: pathlib.Path) -> dict[str, str]:
    """The commit and tree a Git surface's observation starts from: HEAD's, or on an unborn branch no commit and the
    empty tree (``workspace_patch_capture._git_empty_tree_oid``, in the repository's own object format), marked
    ``head_state="unborn"``. Its first commit is then a moved base like any other."""
    head = _current_head(root)
    if head:
        return {"base_commit": head,
                "base_tree": _run_git(root, "rev-parse", f"{head}^{{tree}}", timeout=_GIT_PROBE_TIMEOUT_SEC).strip()}
    from ouroboros.workspace_patch_capture import _git_empty_tree_oid

    errors: list[dict[str, Any]] = []
    tree = _git_empty_tree_oid(root, errors)
    if not tree:
        raise RuntimeError(f"the empty tree of unborn {root} could not be named: {errors}")
    return {"base_commit": "", "base_tree": tree, "head_state": "unborn"}


def _git_status_paths(root: pathlib.Path) -> list[str]:
    """Return every path named by porcelain-v1, including both rename sides."""
    raw = _run_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    fields = raw.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(fields):
        row = fields[index]
        index += 1
        if not row:
            continue
        if len(row) < 4:
            raise RuntimeError("malformed git porcelain row")
        status = row[:2]
        path = row[3:]
        if path:
            paths.append(path)
        if "R" in status or "C" in status:
            if index >= len(fields) or not fields[index]:
                raise RuntimeError("malformed git rename/copy row")
            paths.append(fields[index])
            index += 1
    return sorted(dict.fromkeys(paths))


def _git_committed_paths(root: pathlib.Path, base_commit: str, current_head: str) -> list[str]:
    """Return both sides of paths changed after the baseline commit."""
    if not base_commit or not current_head or base_commit == current_head:
        return []
    raw = _run_git(
        root,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        base_commit,
        current_head,
    )
    return sorted(dict.fromkeys(path for path in raw.split("\0") if path))


def _path_fingerprint(path: pathlib.Path) -> dict[str, Any]:
    """Fingerprint one exact known path without walking any parent directory. A work path's metadata and bytes
    come through the confined intake, never a service open of its pathname."""
    try:
        info = work_domain.lstat(path)
    except FileNotFoundError:
        return {"kind": "missing"}
    if stat.S_ISLNK(info.st_mode):
        return {
            "kind": "symlink",
            "target": work_domain.readlink(path),
        }
    if stat.S_ISREG(info.st_mode):
        if int(info.st_size) > _FINGERPRINT_MAX_BYTES:
            return {
                "kind": "file",
                "size": int(info.st_size),
                "sha256_skipped": "over_size_cap",
            }
        if work_domain.is_work(path):
            sha256 = work_domain.measure(path)["sha256"]
        else:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            sha256 = digest.hexdigest()
        return {
            "kind": "file",
            "size": int(info.st_size),
            "sha256": sha256,
        }
    if stat.S_ISDIR(info.st_mode):
        # Deliberately do not recurse.  user_files may be the owner's home and
        # the contract permits only declared targets / existing artifact refs.
        return {"kind": "directory"}
    return {"kind": "other", "size": int(info.st_size)}


def _git_path_fingerprints(root: pathlib.Path, paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Capture content and ordinary Git staging modes with one config/index read.

    Only regular files need an additional mode fact: symlink targets and absence
    already describe their Git type. Missing mode evidence remains unknown.
    ``core.filemode=false`` takes the indexed mode (default 100644) and
    ``core.symlinks=false`` keeps an indexed symlink mode; the index is only read.
    """
    fingerprints = {path: _path_fingerprint(root / path) for path in paths}
    files = {path: row for path, row in fingerprints.items() if row.get("kind") == "file"}
    if not files:
        return fingerprints
    for row in files.values():
        row["git_mode"] = None
    try:
        config = work_domain.run(
            ["git", "config", "--type=bool", "--null", "--get-regexp", r"^core\.(filemode|symlinks)$"],
            cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="surrogateescape", check=False,
        )
        if config.returncode not in (0, 1):
            raise RuntimeError("Git mode configuration unavailable")
        options = dict(row.split("\n", 1) for row in config.stdout.split("\0") if row)
        modes, unmerged = {}, set()
        for entry in _run_git(root, "ls-files", "--stage", "-z").split("\0"):
            if not entry:
                continue
            metadata, path = entry.split("\t", 1)
            mode, _oid, stage = metadata.split()
            if path in files:
                if stage == "0":
                    modes[path] = mode
                else:
                    unmerged.add(path)
        for path, row in files.items():
            if path in unmerged:
                continue
            indexed = modes.get(path)
            if options.get("core.symlinks", "true") == "false" and indexed == "120000":
                row["git_mode"] = indexed
            elif options.get("core.filemode", "true") == "false":
                row["git_mode"] = indexed if indexed in {"100644", "100755"} else "100644"
            else:
                row["git_mode"] = "100755" if work_domain.lstat(root / path).st_mode & stat.S_IXUSR else "100644"
    except (OSError, RuntimeError, ValueError):
        logging.getLogger(__name__).debug("Git fingerprint mode unavailable", exc_info=True)
    return fingerprints


def _foreign_fingerprint_matches(previous: Any, current: Any) -> bool:
    """Unknown legacy mode must not globally block already-excluded foreign WIP."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    if previous.get("git_mode") is not None and current.get("git_mode") is not None:
        return previous == current
    return ({key: value for key, value in previous.items() if key != "git_mode"}
            == {key: value for key, value in current.items() if key != "git_mode"})


def _normalize_known_paths(root: pathlib.Path, values: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        candidate = pathlib.Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            rel = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"known mutation target escapes surface root: {text}") from exc
        if rel not in normalized:
            normalized.append(rel)
    return sorted(normalized)


def _surface_identity(surface: Mapping[str, Any]) -> tuple[str, pathlib.Path]:
    surface_type = str(surface.get("surface_type") or surface.get("type") or "").strip()
    if surface_type not in {
        "system_repo",
        "skill_payload",
        "external_workspace",
        "user_files",
    }:
        raise ValueError(f"unknown mutation surface type: {surface_type!r}")
    # ``host_root`` is the durable surface identity.  Accept the ``root``
    # spelling only when reading transitional in-memory evidence.
    root = _canonical_root(surface.get("host_root") or surface.get("root"))
    return surface_type, root


def _capture_surface(surface: Mapping[str, Any]) -> dict[str, Any]:
    surface_type, root = _surface_identity(surface)
    known_paths = _normalize_known_paths(root, surface.get("known_paths") or [])
    row: dict[str, Any] = {
        "surface_type": surface_type,
        "canonical_root": str(root),
        "known_paths": known_paths,
        "captured_at": utc_now_iso(),
        **{key: dict(surface[key]) for key in ("basis", "candidate") if isinstance(surface.get(key), Mapping)},
    }
    if surface_type in _GIT_SURFACE_TYPES and isinstance(surface.get("git"), Mapping):
        row["git"] = dict(surface["git"])  # a basis the caller established instead of today's files
        return row
    if surface_type in _GIT_SURFACE_TYPES:
        try:
            inside = _run_git(root, "rev-parse", "--is-inside-work-tree").strip() == "true"
        except Exception:
            inside = False
        if inside:
            basis = _git_basis(root)  # an unborn branch starts from the empty tree, before any commit
            dirty_paths = _git_status_paths(root)
            if len(dirty_paths) > _BASELINE_DIRTY_PATHS_MAX:
                row["git"] = {**basis, "dirty_overflow": len(dirty_paths)}
                return row
            row["git"] = {
                **basis,
                "dirty_paths": dirty_paths,
                "dirty_fingerprints": _git_path_fingerprints(root, dirty_paths),
            }
            return row
    row["known_path_fingerprints"] = {
        path: _path_fingerprint(root / path) for path in known_paths
    }
    return row


def capture_mutation_baseline(
    results_drive_root: Any,
    task_id: str,
    surfaces: Sequence[Mapping[str, Any]],
    *,
    owner_kind: str = "task_root",
    owner_id: str = "",
    predecessor_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture and strictly confirm surface baselines for one task.

    The host calls this when a root task starts (and may append a late,
    previously unseen surface before its first write).  Existing roots and
    known-target fingerprints are immutable once captured, unless a caller
    ``supersede``s a row with a basis it established; the whole earlier row
    then stays in ``extensions``. A Git row recorded before rows named their
    copy may join that ``candidate`` identity, its observation unchanged.
    """
    task_id = str(task_id or "").strip()
    if not task_id:
        raise ValueError("task_id is required for mutation evidence")
    existing = load_task_result(results_drive_root, task_id) or {}
    prior = existing.get("mutation_evidence")
    prior_baseline = prior.get("baseline") if isinstance(prior, dict) else None
    captured = [dict(row) for row in (prior_baseline or {}).get("surfaces") or []]
    owner_kind = str(owner_kind or "task_root")
    owner_id = str(owner_id or task_id)
    if isinstance(prior_baseline, dict):
        if (
            str(prior_baseline.get("owner_kind") or "") != owner_kind
            or str(prior_baseline.get("owner_id") or "") != owner_id
        ):
            raise RuntimeError("mutation baseline owner cannot change")
    existing_by_key = {
        (str(row.get("surface_type") or ""), str(row.get("canonical_root") or "")): row
        for row in captured
    }
    added: list[dict[str, Any]] = []
    for surface in surfaces:
        surface_type, root = _surface_identity(surface)
        key = (surface_type, str(root))
        existing_surface = existing_by_key.get(key)
        if existing_surface is not None and not surface.get("supersede"):
            # Git baselines already cover the complete worktree. Bounded
            # non-Git surfaces add a newly declared exact target before that
            # target's first write without re-fingerprinting older targets.
            if not isinstance(existing_surface.get("git"), dict):
                requested = _normalize_known_paths(root, surface.get("known_paths") or [])
                prior_paths = {
                    str(path) for path in existing_surface.get("known_paths") or []
                }
                new_paths = sorted(path for path in requested if path not in prior_paths)
                if new_paths:
                    fingerprints = dict(existing_surface.get("known_path_fingerprints") or {})
                    fingerprints.update({
                        path: _path_fingerprint(root / path) for path in new_paths
                    })
                    existing_surface["known_paths"] = sorted(prior_paths | set(new_paths))
                    existing_surface["known_path_fingerprints"] = fingerprints
                    added.append({
                        "surface_type": key[0],
                        "canonical_root": key[1],
                        "known_paths": new_paths,
                    })
            elif isinstance(surface.get("candidate"), Mapping) and "candidate" not in existing_surface:
                # Which copy it observes: its dates, observation and basis stay exactly as recorded.
                existing_surface["candidate"] = dict(surface["candidate"])
                added.append({"surface_type": key[0], "canonical_root": key[1],
                              "candidate": existing_surface["candidate"]})
            continue
        candidate = _capture_surface(surface)
        if predecessor_source and isinstance(candidate.get("git"), dict) and "git" not in surface:
            candidate["git"]["predecessor_adoption"] = _predecessor_git_changes(
                results_drive_root, task_id, candidate, predecessor_source,
            )
        extension: dict[str, Any] = {"surface_type": key[0], "canonical_root": key[1]}
        if existing_surface is not None:  # never erased: the whole earlier row stays in this history
            captured.remove(existing_surface)
            extension.update(basis=candidate.get("basis") or {}, superseded=existing_surface)
        captured.append(candidate)
        existing_by_key[key] = candidate
        added.append(extension)
    if isinstance(prior_baseline, dict) and not added:
        return dict(prior)
    captured.sort(key=lambda item: (item["surface_type"], item["canonical_root"]))
    baseline = dict(prior_baseline or {})
    baseline.update({
        "captured_at": str(baseline.get("captured_at") or utc_now_iso()),
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "surfaces": captured,
    })
    if prior_baseline and added:
        extensions = [
            dict(row) for row in baseline.get("extensions") or [] if isinstance(row, dict)
        ]
        extensions.append({"captured_at": utc_now_iso(), "surfaces": added})
        baseline["extensions"] = extensions
    baseline.pop("baseline_hash", None)
    baseline["baseline_hash"] = _stable_hash(baseline)
    evidence = dict(prior or {})
    evidence.update({
        "version": MUTATION_EVIDENCE_VERSION,
        "baseline": baseline,
        "effect_state": str(evidence.get("effect_state") or "observed_window"),
        "flags": list(evidence.get("flags") or []),
    })
    status = str(existing.get("status") or STATUS_RUNNING)
    written = write_task_result(
        results_drive_root,
        task_id,
        status,
        mutation_evidence=evidence,
    )
    confirmed = written.get("mutation_evidence") if isinstance(written, dict) else None
    if not isinstance(confirmed, dict) or confirmed.get("baseline") != baseline:
        raise RuntimeError("mutation baseline was not durably confirmed")
    return dict(confirmed)


def capture_root_baseline(
    results_drive_root: Any, task: Mapping[str, Any], *, system_repo: Any, workspace: Any, conversation: bool,
) -> dict[str, Any] | None:
    """The baseline a root task's admitted resources start from, captured before its first operation: the
    serving repository for a queued root (commit staging attributes against it), and the admitted folder it
    works in when that is a Git tree of its own, for every root, a conversation's too (owner, 8 Nov 2026: queue
    membership is not what makes production attributable). A resumed task keeps the surfaces it started with:
    a folder its start never observed is not baselined after work was done. A freshly initialized folder starts
    from the empty tree (``_git_basis``); a surface whose capture still fails is left without a baseline (a named
    gap at its readers), never at the cost of the other."""
    task_id = str(task.get("id") or "").strip()
    surfaces = [] if conversation else [{"surface_type": "system_repo", "host_root": str(system_repo)}]
    folder = _canonical_root(workspace) if str(workspace or "").strip() else None
    if folder is not None and folder != _canonical_root(system_repo):
        try:  # at its own top level cdup is empty: Git in a container prints its own spelling of any address
            own_tree = _run_git(folder, "rev-parse", "--is-inside-work-tree", "--show-cdup",
                                timeout=_GIT_PROBE_TIMEOUT_SEC).strip() == "true"
        except Exception:
            own_tree = False
        evidence = (load_task_result(results_drive_root, task_id) or {}).get("mutation_evidence") if own_tree else None
        prior = evidence.get("baseline") if isinstance(evidence, dict) else None
        if own_tree and (not isinstance(prior, dict) or any(
                isinstance(row, dict) and row.get("canonical_root") == str(folder) for row in prior.get("surfaces") or [])):
            surfaces.append({"surface_type": "external_workspace", "host_root": str(folder)})
    if not surfaces:
        return None
    predecessor = (task.get("predecessor_authority") or {}).get("source")

    def capture(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return capture_mutation_baseline(results_drive_root, task_id, rows, owner_kind="task_root",
                                         owner_id=str(task.get("root_task_id") or task_id),
                                         predecessor_source=predecessor if isinstance(predecessor, Mapping) else None)

    try:
        return capture(surfaces)
    except Exception:
        if len(surfaces) < 2:
            raise
    captured = None
    for surface in surfaces:  # one failing surface never costs the other its baseline
        try:
            captured = capture([surface])
        except Exception:
            logging.getLogger(__name__).warning("%s %s of task %s has no baseline", surface["surface_type"],
                                                surface["host_root"], task_id, exc_info=True)
    return captured


def _predecessor_git_changes(
    results_drive_root: Any, task_id: str, surface: Mapping[str, Any], source: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a host-selected predecessor's exact terminal work at task start.

    A path is adopted only with exact retained content: a sha256, a missing path
    or a symlink; a size-only or mode-unknown regular file proves no transfer.
    """
    from ouroboros.task_status import FINAL_STATUSES, load_effective_task_result

    previous_id = str(source.get("task_id") or "")
    adoption: dict[str, Any] = {"task_id": previous_id, "source": dict(source), "paths": []}
    if not previous_id or previous_id == task_id:
        return adoption
    previous = load_effective_task_result(results_drive_root, previous_id, materialize_artifacts=False) or {}
    if previous.get("status") not in FINAL_STATUSES:
        return adoption
    evidence = previous.get("mutation_evidence") or {}
    baseline = evidence.get("baseline") or {}
    terminal = evidence.get("terminal_candidate_snapshot") or {}
    if (evidence.get("effect_state") != "quiescent"
            or not baseline.get("baseline_hash")
            or terminal.get("baseline_hash") != baseline.get("baseline_hash")):
        return adoption
    root = _canonical_root(surface["canonical_root"])
    matching = [row for row in terminal.get("surfaces") or [] if isinstance(row, dict)
                and row.get("canonical_root") == str(root)
                and row.get("surface_type") == surface.get("surface_type")]
    if len(matching) != 1 or matching[0].get("blockers"):
        return adoption
    retained = matching[0]
    fingerprints = retained.get("candidate_fingerprints") or {}
    git = surface["git"]
    # Unrelated landed changes need not forbid continuation, but a changed base
    # for a retained path cannot be mistaken for that path's original patch.
    previous_head = str(retained.get("base_commit") or "")
    if not previous_head:
        return adoption
    try:
        changed_base = set(_git_committed_paths(root, previous_head, str(git.get("base_commit") or "")))
    except Exception:
        return adoption
    for path in git.get("dirty_paths") or []:
        fingerprint = fingerprints.get(path)
        if (path not in changed_base and path in (retained.get("candidates") or [])
                and isinstance(fingerprint, dict)
                and (fingerprint.get("sha256") or fingerprint.get("kind") in {"missing", "symlink"})
                and (fingerprint.get("kind") != "file"
                     or fingerprint.get("git_mode") in {"100644", "100755", "120000"})
                and fingerprint == (git.get("dirty_fingerprints") or {}).get(path)):
            adoption["paths"].append(path)
    return adoption


def _foreign_dirty_paths(git: Mapping[str, Any]) -> set[str]:
    """Keep the observed dirty set intact while recognizing explicit custody."""
    adoption = git.get("predecessor_adoption") or {}
    return set(git.get("dirty_paths") or []) - set(adoption.get("paths") or [])


def _uncertain_dirt(row: Mapping[str, Any], dirty: set[str], candidate_root: bool) -> set[str]:
    """A body candidate's dirt from any observation but an established basis: maybe this work's, never known foreign."""
    basis = row.get("basis") if isinstance(row.get("basis"), Mapping) else {}
    copy = candidate_root or isinstance(row.get("candidate"), Mapping)
    return set(dirty) if copy and basis.get("kind") not in ESTABLISHED_BASES else set()


def mutation_evidence_projection(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return bounded host evidence suitable for acceptance and outcome gates."""
    if not isinstance(evidence, Mapping) or not evidence:
        return {}

    blockers: list[str] = []
    baseline = evidence.get("baseline")
    surfaces = baseline.get("surfaces") if isinstance(baseline, Mapping) else None
    if not isinstance(baseline, Mapping) or not isinstance(surfaces, list) or not surfaces:
        blockers.append("baseline_missing")
        surfaces = []
    baseline_hash = str(baseline.get("baseline_hash") or "") if isinstance(baseline, Mapping) else ""
    if surfaces and not baseline_hash:
        blockers.append("baseline_hash_missing")

    surface_counts: dict[str, int] = {}
    known_path_count = 0
    preexisting_dirty_count = 0
    adopted_path_count = 0
    for row in surfaces[:64]:
        if not isinstance(row, Mapping):
            blockers.append("baseline_surface_malformed")
            continue
        surface_type = str(row.get("surface_type") or "unknown")[:40]
        surface_counts[surface_type] = surface_counts.get(surface_type, 0) + 1
        known_path_count += len(row.get("known_paths") or [])
        git = row.get("git")
        if isinstance(git, Mapping):
            preexisting_dirty_count += len(git.get("dirty_paths") or [])
            adopted_path_count += len((git.get("predecessor_adoption") or {}).get("paths") or [])
    if len(surfaces) > 64:
        blockers.append("baseline_surface_limit_exceeded")

    effect_state = str(evidence.get("effect_state") or "")[:80]
    if effect_state not in _OBSERVED_EFFECT_STATES:
        blockers.append(effect_state or "effect_state_unknown")
    # ``flags`` is a reserved evidence field: no runtime writer exists in
    # v6.66.0, but a populated flag (from older evidence or a future writer)
    # must still surface as a blocker rather than vanish.
    flags = sorted(dict.fromkeys(
        str(row.get("flag") or "")[:80]
        for row in evidence.get("flags") or []
        if isinstance(row, Mapping) and str(row.get("flag") or "").strip()
    ))[:32]
    blockers.extend(flags)

    terminal = evidence.get("terminal_candidate_snapshot")
    terminal_candidate_count = 0
    terminal_excluded_count = 0
    if terminal is None and effect_state == "quiescent":
        blockers.append("terminal_snapshot_missing")
    elif terminal is not None:
        if not isinstance(terminal, Mapping):
            blockers.append("terminal_snapshot_malformed")
        else:
            terminal_rows = terminal.get("surfaces")
            if not isinstance(terminal_rows, list):
                blockers.append("terminal_snapshot_malformed")
            else:
                for row in terminal_rows[:64]:
                    if not isinstance(row, Mapping):
                        blockers.append("terminal_snapshot_malformed")
                        continue
                    terminal_candidate_count += len(row.get("candidates") or [])
                    excluded = len(row.get("excluded_preexisting_dirty") or [])
                    terminal_excluded_count += excluded
                    if excluded:
                        blockers.append("preexisting_dirty_changed")
                    if row.get("uncertain"):  # a body candidate's unresolved observation, not foreign work
                        blockers.append("attribution_uncertain")
                    blockers.extend(
                        str(item)[:80]
                        for item in row.get("blockers") or []
                        if str(item).strip()
                    )
                if len(terminal_rows) > 64:
                    blockers.append("terminal_surface_limit_exceeded")

    blockers = sorted(dict.fromkeys(item for item in blockers if item))
    # The union above serves retained patch custody. Only the separately
    # observed dirty set answers what remains outside HEAD; a retained clean
    # commit is not an uncommitted file. Old snapshots carry no such claim.
    remainder_rows = [row for row in (terminal.get("surfaces") or [] if isinstance(terminal, Mapping) else [])
                      if isinstance(row, Mapping) and "uncommitted_paths" in row]
    remainder = ({
        "state": "unknown" if any(row.get("uncommitted_state") != "known" for row in remainder_rows) else "known",
        "count": sum(len(row["uncommitted_paths"]) for row in remainder_rows),
        # Exact paths stay in the existing task detail's mutation_evidence.
        # Result axes, terminal rows and WS carry only this disclosed omission.
        "paths_omitted": sum(len(row["uncommitted_paths"]) for row in remainder_rows),
        "snapshot_at": str(terminal.get("captured_at") or ""),
        "baseline_hash": str(terminal.get("baseline_hash") or ""),
    } if remainder_rows else {})
    if remainder.get("state") == "unknown":
        remainder["count"] = None
    return {
        "version": MUTATION_EVIDENCE_VERSION,
        "present": True,
        "baseline_hash": baseline_hash,
        "effect_state": effect_state,
        "flags": flags,
        "surface_counts": dict(sorted(surface_counts.items())),
        "known_path_count": known_path_count,
        "preexisting_dirty_count": preexisting_dirty_count,
        **({"adopted_predecessor_path_count": adopted_path_count} if adopted_path_count else {}),
        "terminal_snapshot_present": isinstance(terminal, Mapping),
        "terminal_candidate_count": terminal_candidate_count,
        "terminal_excluded_count": terminal_excluded_count,
        **({"uncommitted_remainder": remainder} if remainder else {}),
        "blockers": blockers,
        "clean_eligible": not blockers,
    }


def load_mutation_evidence_projection(
    results_drive_root: Any,
    task_id: str,
) -> dict[str, Any]:
    """Load compact attribution; the named task's existing Details read retains exact paths."""
    try:
        result = load_task_result(results_drive_root, str(task_id or "")) or {}
    except Exception:
        return {
            "version": MUTATION_EVIDENCE_VERSION,
            "present": False,
            "effect_state": "unavailable",
            "flags": [],
            "surface_counts": {},
            "blockers": ["mutation_evidence_unavailable"],
            "clean_eligible": False,
        }
    projection = mutation_evidence_projection(result.get("mutation_evidence"))
    if projection.get("uncommitted_remainder"):
        projection["uncommitted_remainder"]["task_id"] = str(task_id)
    return projection


def attributed_git_candidates(
    results_drive_root: Any,
    task_id: str,
    repo_root: Any,
    *,
    candidate_root: bool = False,
) -> dict[str, Any]:
    """Compute the exact clean-at-baseline Git candidate set for one root.

    ``candidate_root`` is the caller's host fact that this root is its bound body
    candidate; it also covers rows recorded before candidate surfaces carried an
    identity. Dirt in such a copy is ``uncertain`` unless an established basis
    recorded it; only known foreign dirt is ``excluded`` and can block.
    """
    root = _canonical_root(repo_root)
    result = load_task_result(results_drive_root, task_id) or {}
    evidence = result.get("mutation_evidence")
    blockers: list[str] = []
    if not isinstance(evidence, dict):
        return {
            "candidates": [],
            "excluded_preexisting_dirty": [],
            "blockers": ["baseline_missing"],
            "baseline_hash": "",
        }
    baseline = evidence.get("baseline")
    if not isinstance(baseline, dict):
        blockers.append("baseline_missing")
        surfaces: list[dict[str, Any]] = []
    else:
        surfaces = [row for row in baseline.get("surfaces") or [] if isinstance(row, dict)]
    matching = [
        row for row in surfaces
        if str(row.get("canonical_root") or "") == str(root)
        and isinstance(row.get("git"), dict)
    ]
    if len(matching) != 1:
        blockers.append("baseline_surface_missing" if not matching else "baseline_surface_ambiguous")
        return {
            "candidates": [],
            "excluded_preexisting_dirty": [],
            "blockers": blockers,
            "baseline_hash": str((baseline or {}).get("baseline_hash") or ""),
        }
    row, git = matching[0], matching[0]["git"]
    if git.get("dirty_overflow"):
        return {
            "candidates": [],
            "excluded_preexisting_dirty": [],
            "blockers": ["baseline_dirty_overflow"],
            "baseline_hash": str(baseline.get("baseline_hash") or ""),
        }
    try:
        current_head = _current_head(root)  # "" while an unborn baseline's branch still has no commit
        changed = _git_status_paths(root)
    except Exception:
        blockers.append("candidate_scan_failed")
        current_head = ""
        changed = []
    if current_head != str(git.get("base_commit") or ""):
        blockers.append("baseline_stale")
    dirty = _foreign_dirty_paths(git)
    copy = candidate_root or isinstance(row.get("candidate"), dict)
    basis = dict(row["basis"]) if isinstance(row.get("basis"), dict) else {"kind": "unrecorded"}
    uncertain_dirty = _uncertain_dirt(row, dirty, copy)
    fingerprints = git.get("dirty_fingerprints") or {}
    excluded = sorted(path for path in changed if path in dirty - uncertain_dirty)
    candidates = sorted(path for path in changed if path not in dirty)
    # A pre-existing dirty path is excluded evidence either way; it becomes a
    # blocker only when known content/mode CHANGED during the observed window
    # (unchanged owner WIP merely persisting must not wedge the task's commits).
    current_fingerprints = _git_path_fingerprints(root, excluded)
    if any(not _foreign_fingerprint_matches(fingerprints.get(path), current_fingerprints[path]) for path in excluded):
        blockers.append("preexisting_dirty_changed")
    blockers.extend(_evidence_blockers(evidence, _OBSERVED_EFFECT_STATES))
    return {
        "candidates": candidates,
        "excluded_preexisting_dirty": excluded,
        "blockers": sorted(dict.fromkeys(blockers)),
        "baseline_hash": str(baseline.get("baseline_hash") or ""),
        "base_commit": str(git.get("base_commit") or ""),
        "base_tree": str(git.get("base_tree") or ""),
        "canonical_root": str(root),
        **({"predecessor_adoption": dict(git["predecessor_adoption"])} if git.get("predecessor_adoption") else {}),
        **({"uncertain": sorted(path for path in changed if path in uncertain_dirty),
            "copy": {"basis": basis, **({"candidate": dict(row["candidate"])}
                                       if isinstance(row.get("candidate"), dict) else {})}} if copy else {}),
    }


def attributed_commit_paths(results_drive_root: Any, task_id: str, repo_root: Any) -> list[str]:
    """Paths the task's own reviewed commits changed in ``repo_root``, read from its ``attributed_commit`` epochs
    (``advance_mutation_baseline``): committed work stays visible after a clean commit. A HEAD move no epoch
    records (a shell commit, a pull) attributes nothing here and stays ``baseline_stale``; base..HEAD alone is
    never authorship."""
    root = _canonical_root(repo_root)
    evidence = (load_task_result(results_drive_root, str(task_id or "")) or {}).get("mutation_evidence")
    baseline = evidence.get("baseline") if isinstance(evidence, dict) else None
    paths: list[str] = []
    for epoch in baseline.get("epochs") or [] if isinstance(baseline, dict) else []:
        if (isinstance(epoch, dict) and epoch.get("reason") == "attributed_commit" and epoch.get("commit")
                and any(isinstance(row, dict) and row.get("canonical_root") == str(root) for row in epoch.get("previous") or [])):
            raw = _run_git(root, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "-z", "--no-renames",
                           str(epoch["commit"]))  # --root: an unborn folder's first commit lists its files too
            paths.extend(path for path in raw.split("\0") if path)
    return sorted(dict.fromkeys(paths))


def _evidence_blockers(evidence: Mapping[str, Any], states: frozenset) -> list[str]:
    """An effect state outside ``states`` and every populated flag, as blockers."""
    effect_state = str(evidence.get("effect_state") or "")
    return ([] if effect_state in states else [effect_state or "effect_state_unknown"]) + [
        str(row["flag"]) for row in evidence.get("flags") or []
        if isinstance(row, dict) and str(row.get("flag") or "")]


def resolve_attributed_git_paths(
    results_drive_root: Any,
    task_id: str,
    repo_root: Any,
    explicit_paths: Sequence[Any] | None,
    *,
    candidate_root: bool = False,
    retained: bool = False,
) -> tuple[list[str], dict[str, Any], str]:
    """Resolve omitted/explicit staging paths against one attributed candidate set.

    Automatic selection stays cautious: any blocker refuses it and uncertain
    paths stay out. On a body candidate copy an explicit selection may also name
    uncertain paths and pass a stale base (an overlapping move, never re-anchored
    here); ``selection`` then carries that gap to review and the author. Known
    foreign dirt is never selectable, and empty never means everything.
    ``retained`` marks a call collecting a review over the index it keeps: with
    paths omitted, that staged composition is the explicit selection, retaining
    its uncertainty and moved-base disclosure rather than selecting fresh work.
    """
    evidence = attributed_git_candidates(results_drive_root, task_id, repo_root, candidate_root=candidate_root)
    blockers = [str(item) for item in evidence.get("blockers") or [] if str(item)]
    if blockers == ["baseline_stale"] and try_reanchor_stale_baseline(
        results_drive_root, task_id, repo_root,
    ):
        evidence = attributed_git_candidates(results_drive_root, task_id, repo_root, candidate_root=candidate_root)
        blockers = [str(item) for item in evidence.get("blockers") or [] if str(item)]
    implicit_collection = explicit_paths is None and retained
    if implicit_collection:
        try:  # the composition the collected review froze stays its selection
            explicit_paths = [path for path in _run_git(_canonical_root(repo_root), "diff", "--cached",
                                                        "--no-renames", "--name-only", "-z").split("\0") if path] or None
        except RuntimeError:
            pass  # unreadable: the fresh rules below answer
    copy = evidence.get("copy") if isinstance(evidence.get("copy"), dict) else None
    if candidate_root and blockers == ["baseline_surface_missing"]:  # a body copy younger than the task's baseline:
        try:  # its own changes are this continued work, each uncertain, as on a moved base (06.11)
            evidence["uncertain"] = sorted(_git_status_paths(_canonical_root(repo_root)))
        except Exception:
            evidence["uncertain"] = []
        copy = evidence.setdefault("copy", {"basis": {"kind": "unrecorded"}})
    gaps = blockers if copy is not None and set(blockers) <= {"baseline_stale", "baseline_surface_missing"} else []
    if blockers and not (gaps and explicit_paths is not None):
        named = f": paths={json.dumps(evidence.get('uncertain') or [], ensure_ascii=False)}" if evidence.get("uncertain") else ""
        return [], evidence, (
            "⚠️ GIT_ATTRIBUTION_BLOCKED: clean task attribution is unavailable "
            f"({', '.join(blockers)}). Automatic staging is disabled; preserve the "
            "unattributed changes and resolve the evidence conflict before review."
            + (" To send this body candidate's composition to review with this gap "
               f"disclosed, name its paths explicitly{named}." if gaps else "")
        )
    candidates = [str(item) for item in evidence.get("candidates") or [] if str(item)]
    uncertain = [str(item) for item in evidence.get("uncertain") or [] if str(item)]
    if explicit_paths is None:
        selected = candidates
    else:
        try:
            selected = sorted(dict.fromkeys(
                safe_relpath(str(path)) for path in explicit_paths if str(path or "").strip()
            ))
        except ValueError as exc:
            return [], evidence, f"⚠️ PATH_ERROR: {exc}"
        outside = sorted(set(selected) - set(candidates) - set(uncertain))
        if outside:
            return [], evidence, (
                "⚠️ GIT_ATTRIBUTION_BLOCKED: explicit paths must be a subset of "
                f"the task-attributed candidates. Outside set: {', '.join(outside)}"
            )
    if not selected:
        return [], evidence, (
            "⚠️ GIT_NO_ATTRIBUTED_CHANGES: no clean-at-baseline task-owned paths "
            "are available to stage."
            + (f" Uncertain attribution in this body candidate: {', '.join(uncertain)}; name such paths "
               "explicitly to send them to review with that gap disclosed." if uncertain else "")
        )
    if copy is not None:
        evidence["selection"] = {**copy, "base_commit": str(evidence.get("base_commit") or ""), "gaps": gaps,
                                 "uncertain": sorted(set(selected) & set(uncertain)),
                                 "explicit_uncertain": [] if implicit_collection else sorted(set(selected) & set(uncertain)),
                                 "left_out": sorted(set(uncertain) - set(selected)),
                                 "excluded_known_foreign": len(evidence.get("excluded_preexisting_dirty") or [])}
    return selected, evidence, ""


def selection_disclosure(selection: Mapping[str, Any] | None) -> str:
    """The attribution facts of one selected body-candidate composition, for its reviewers and author:
    every uncertain name, since an identical diff can hide which of its paths are."""
    if not isinstance(selection, Mapping):
        return ""
    basis = selection.get("basis") if isinstance(selection.get("basis"), Mapping) else {}
    kind, notes = str(basis.get("kind") or "unrecorded"), []
    if kind not in {"original", "derived", "carried"}:
        notes.append(f"the copy's recorded basis is {kind}, a later observation")
    elif kind != "original":
        notes.append(f"the copy's baseline is {kind} from {basis.get('source') or 'an earlier record'}: changes "
                     "made in it since then count as this continued work, which does not prove that each "
                     "line was written by this task")
    if "baseline_surface_missing" in (selection.get("gaps") or ()):
        notes.append("the copy has no recorded task baseline, so which of its changes this task made is unknown")
    elif selection.get("gaps"):
        notes.append(f"its HEAD moved from the recorded base {str(selection.get('base_commit') or '')[:12]} over "
                     "paths it changes, and this explicit selection was not re-anchored")
    for key, label in (("uncertain", "selected with uncertain attribution"), ("left_out", "left out as uncertain")):
        if selection.get(key):
            notes.append(f"{label}: {', '.join(selection[key])}")
    return ("Task attribution of this body-candidate composition: " + "; ".join(notes) + ".") if notes else ""


def record_terminal_mutation_candidates(
    results_drive_root: Any,
    task_id: str,
    *,
    candidate_roots: Iterable[Any] = (),
) -> dict[str, Any]:
    """Persist one quiescent projection from the existing baseline authority.

    ``candidate_roots`` names the lineage's body candidate (rows recorded before
    surfaces carried an identity): its uncertain dirt is listed as such and stays
    in the remainder, as the commit consumer reads the same copy.
    """
    copies = {str(_canonical_root(path)) for path in candidate_roots if str(path or "").strip()}
    current = load_task_result(results_drive_root, task_id) or {}
    evidence = dict(current.get("mutation_evidence") or {})
    # The final task-result write publishes terminal candidates and the
    # ``quiescent`` effect state atomically.  Until it succeeds the durable
    # evidence keeps its earlier state (``observed_window``), so a claimed
    # quiescence without a snapshot stays detectable.
    evidence["effect_state"] = "quiescent"
    baseline = evidence.get("baseline") if isinstance(evidence.get("baseline"), dict) else {}
    rows: list[dict[str, Any]] = []
    for surface in baseline.get("surfaces") or []:
        if not isinstance(surface, dict):
            continue
        root_text = str(surface.get("canonical_root") or "")
        row: dict[str, Any] = {
            "surface_type": str(surface.get("surface_type") or ""),
            "canonical_root": root_text,
        }
        if isinstance(surface.get("git"), dict):
            git = surface["git"]
            blockers: list[str] = []
            if git.get("dirty_overflow"):
                row.update({
                    "candidates": [],
                    "excluded_preexisting_dirty": [],
                    "blockers": ["baseline_dirty_overflow"],
                    # An admitted folder's work is its delivery, never a body candidate's uncommitted remainder.
                    **({} if row["surface_type"] == "external_workspace" else
                       {"uncommitted_paths": [], "uncommitted_state": "unknown"}),
                })
                rows.append(row)
                continue
            try:
                root = _canonical_root(root_text)
                current_head = _current_head(root)
                uncommitted = _git_status_paths(root)
                changed = sorted(dict.fromkeys(
                    uncommitted
                    + _git_committed_paths(
                        root,
                        # An unborn baseline has no commit: its interval starts at the empty tree.
                        str(git.get("base_commit") or git.get("base_tree") or ""),
                        current_head,
                    )
                ))
            except Exception:
                current_head = ""
                uncommitted = []
                changed = []
                blockers.append("candidate_scan_failed")
            dirty = _foreign_dirty_paths(git)
            uncertain = _uncertain_dirt(surface, dirty, root_text in copies)
            fingerprints = git.get("dirty_fingerprints") or {}
            excluded = sorted(path for path in changed if path in dirty - uncertain)
            try:
                current_fingerprints = _git_path_fingerprints(root, changed)
            except OSError:
                current_fingerprints = {}
                blockers.append("candidate_fingerprint_unavailable")
            if any(
                not _foreign_fingerprint_matches(fingerprints.get(path), current_fingerprints.get(path))
                for path in excluded
            ):
                blockers.append("preexisting_dirty_changed")
            blockers.extend(_evidence_blockers(evidence, frozenset({"quiescent"})))
            candidates = sorted(path for path in changed if path not in dirty)
            candidate_fingerprints = {path: current_fingerprints[path] for path in candidates if path in current_fingerprints}
            row.update({
                "candidates": candidates,
                "candidate_fingerprints": candidate_fingerprints,
                "base_commit": current_head,
                "excluded_preexisting_dirty": excluded,
                **({"uncertain": sorted(path for path in changed if path in uncertain)} if uncertain else {}),
                "blockers": sorted(dict.fromkeys(blockers)),
                "uncommitted_paths": sorted(set(uncommitted) - (dirty - uncertain)),
                "uncommitted_state": "unknown" if blockers else "known",
                "head_advanced": bool(
                    current_head
                    and current_head != str(git.get("base_commit") or "")
                ),
            })
            if row["surface_type"] == "external_workspace":  # an admitted folder's work is its delivery, not a
                row.pop("uncommitted_paths"), row.pop("uncommitted_state")  # body candidate's uncommitted remainder
        else:
            root = _canonical_root(root_text)
            prior = surface.get("known_path_fingerprints") or {}
            changed: list[str] = []
            for path in surface.get("known_paths") or []:
                normalized = str(path or "")
                if normalized and _path_fingerprint(root / normalized) != prior.get(normalized):
                    changed.append(normalized)
            row.update({"candidates": sorted(changed), "blockers": []})
        rows.append(row)
    evidence["terminal_candidate_snapshot"] = {
        "captured_at": utc_now_iso(),
        "baseline_hash": str(baseline.get("baseline_hash") or ""),
        "surfaces": rows,
    }
    status = str(current.get("status") or STATUS_RUNNING)
    written = write_task_result(
        results_drive_root,
        task_id,
        status,
        mutation_evidence=evidence,
    )
    confirmed = written.get("mutation_evidence") if isinstance(written, dict) else None
    if (
        not isinstance(confirmed, dict)
        or str(confirmed.get("effect_state") or "") != "quiescent"
        or confirmed.get("terminal_candidate_snapshot")
        != evidence["terminal_candidate_snapshot"]
    ):
        raise RuntimeError("terminal mutation evidence was not durably confirmed")
    return dict(confirmed)


def try_reanchor_stale_baseline(
    results_drive_root: Any,
    task_id: str,
    repo_root: Any,
) -> bool:
    """Heal a stale baseline when HEAD moved past it harmlessly.

    A foreign fast-forward commit that touches NONE of the currently dirty
    paths cannot change what this task may attribute, so wedging every later
    ``commit_reviewed`` on ``baseline_stale`` would punish the task for someone
    else's unrelated work. Conditions are strict: the baseline commit must be
    an ancestor of HEAD and the committed interval must be disjoint from every
    currently dirty path; anything else keeps the fail-closed stale blocker.
    The re-anchor is auditable as a ``foreign_ff_reanchor`` epoch row.
    """
    root = _canonical_root(repo_root)
    result = load_task_result(results_drive_root, str(task_id or "")) or {}
    evidence = result.get("mutation_evidence")
    baseline = evidence.get("baseline") if isinstance(evidence, dict) else None
    if not isinstance(baseline, dict):
        return False
    git_row = next(
        (
            row.get("git")
            for row in baseline.get("surfaces") or []
            if isinstance(row, dict)
            and str(row.get("canonical_root") or "") == str(root)
            and isinstance(row.get("git"), dict)
        ),
        None,
    )
    if not isinstance(git_row, dict):
        return False
    base_commit = str(git_row.get("base_commit") or "")
    try:
        current_head = _run_git(root, "rev-parse", "HEAD").strip()
        if not base_commit or base_commit == current_head:
            return False
        ancestor_probe = work_domain.run(
            ["git", "merge-base", "--is-ancestor", base_commit, current_head],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if ancestor_probe.returncode != 0:
            return False
        interval = set(_git_committed_paths(root, base_commit, current_head))
        current_dirty = set(_git_status_paths(root))
    except Exception:
        return False
    if interval & current_dirty:
        return False
    try:
        advance_mutation_baseline(
            results_drive_root, task_id, root, reason="foreign_ff_reanchor",
        )
    except Exception:
        return False
    return True


def advance_mutation_baseline(
    results_drive_root: Any,
    task_id: str,
    repo_root: Any,
    *,
    reason: str = "attributed_commit",
    commit: str = "",
    attribution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Open a new baseline epoch after the task's own attributed commit.

    The committed content is now part of HEAD, so a fresh capture re-marks the
    surviving foreign dirt as pre-existing and the next attributed staging
    window opens from the new commit instead of failing ``baseline_stale``.
    An unexplained HEAD move (no advance) still surfaces as a stale baseline.
    The epoch row keeps the whole replaced surface and, for a commit, its
    ``selection`` facts, so a gap disclosed at review stays in history.
    """
    root = _canonical_root(repo_root)
    current = load_task_result(results_drive_root, str(task_id or "")) or {}
    evidence = dict(current.get("mutation_evidence") or {})
    baseline = evidence.get("baseline")
    if not isinstance(baseline, dict):
        raise RuntimeError("cannot advance a missing mutation baseline")
    surfaces = [dict(row) for row in baseline.get("surfaces") or [] if isinstance(row, dict)]
    replaced: list[dict[str, Any]] = []
    for idx, row in enumerate(surfaces):
        if str(row.get("canonical_root") or "") != str(root) or not isinstance(row.get("git"), dict):
            continue
        # The new epoch inherits the INTERSECTION of the original pre-existing
        # dirty set with what is still dirty now. A blind re-capture would
        # re-mark the task's own not-yet-committed leftovers as foreign
        # pre-existing dirt and wedge their follow-up commit.
        prior_dirty = {str(path) for path in row["git"].get("dirty_paths") or []}
        still_dirty = sorted(prior_dirty & set(_git_status_paths(root)))
        new_row = dict(row)
        new_row["captured_at"] = utc_now_iso()
        new_row["git"] = {
            "base_commit": _run_git(root, "rev-parse", "HEAD").strip(),
            "base_tree": _run_git(root, "rev-parse", "HEAD^{tree}").strip(),
            "dirty_paths": still_dirty,
            "dirty_fingerprints": _git_path_fingerprints(root, still_dirty),
            **({"predecessor_adoption": {
                **row["git"]["predecessor_adoption"],
                "paths": sorted(set(row["git"]["predecessor_adoption"].get("paths") or []) & set(still_dirty)),
            }} if row["git"].get("predecessor_adoption") else {}),
        }
        surfaces[idx] = new_row
        replaced.append(row)
    if not replaced:
        raise RuntimeError("no git baseline surface matches the committed root")
    epochs = [dict(r) for r in baseline.get("epochs") or [] if isinstance(r, dict)]
    epochs.append({"advanced_at": utc_now_iso(), "reason": str(reason or "attributed_commit"),
                   "previous": replaced,
                   **({"commit": str(commit)} if commit else {}),
                   **({"attribution": dict(attribution)} if attribution else {})})
    baseline = dict(baseline)
    baseline.update({"surfaces": surfaces, "epochs": epochs})
    baseline.pop("baseline_hash", None)
    baseline["baseline_hash"] = _stable_hash(baseline)
    evidence.update({
        "version": MUTATION_EVIDENCE_VERSION,
        "baseline": baseline,
        "effect_state": "observed_window",
    })
    # A terminal snapshot from the previous epoch must not linger as current.
    evidence.pop("terminal_candidate_snapshot", None)
    status = str(current.get("status") or STATUS_RUNNING)
    written = write_task_result(
        results_drive_root,
        task_id,
        status,
        mutation_evidence=evidence,
    )
    confirmed = written.get("mutation_evidence") if isinstance(written, dict) else None
    if not isinstance(confirmed, dict) or confirmed.get("baseline") != baseline:
        raise RuntimeError("advanced mutation baseline was not durably confirmed")
    return dict(confirmed)


__all__ = [
    "MUTATION_EVIDENCE_VERSION",
    "advance_mutation_baseline",
    "attribution_task_id",
    "attributed_commit_paths",
    "attributed_git_candidates",
    "capture_mutation_baseline",
    "capture_root_baseline",
    "load_mutation_evidence_projection",
    "mutation_evidence_projection",
    "record_terminal_mutation_candidates",
    "resolve_attributed_git_paths",
    "try_reanchor_stale_baseline",
]
