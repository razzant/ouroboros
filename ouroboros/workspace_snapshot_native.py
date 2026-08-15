"""Stable snapshot and conflict-safe patch primitives for the execd kernel."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import threading
from collections.abc import Mapping
from typing import Any

from ouroboros.export_policy_contract import (
    MANIFEST_EXCLUSION_ROW_FIELDS,
    QUESTION_EXPORT,
    REASON_EXCLUDED_DIRECTORY,
    REASON_PROTECTED_ARTIFACT,
    AliasIndex,
    build_policy_document,
    export_policy_hash,
    judged_exclusion,
    normalize_export_policy,
    path_under_any,
)
from ouroboros.workspace_diagnostics import ToolExecutionEnvelope
from ouroboros.workspace_native_contract import NativeOperationResult
# The ONE confinement door for every native mutation. `_restore_rows` builds its target
# from workspace-relative components, which is the shape that escaped through a symlink in
# `write_file`'s append arm.
from ouroboros.workspace_native_paths import native_mutation_target

# The TARGET-side PRODUCTION caps: what this walk will read into one snapshot before
# it records a `*_limit_exceeded` failure. Home's ACCEPTANCE caps are derived from
# these (`remote_transfer.MAX_ACCEPTED_SNAPSHOT_*`) rather than restated, because they
# were restated and disagreed: Home accepted 20_000 files while this produced up to
# 25_000, so a clean snapshot of a 22k-file workspace was refused by Home with
# "exceeds the Home file limit" and nothing about the target to fix. Two names for two
# roles is right; one name for two different numbers is how that happened.
MAX_SNAPSHOT_FILES = 25_000
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.Lock] = {}


def snapshot_policy(
    policy: Mapping[str, Any] | None,
    protected_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The document this snapshot is judged by — one object, never two.

    A policy handed down from Home is AUTHORITATIVE and ``protected_paths`` is
    read out of it rather than merged with it: the hash the prepared operation
    carries must describe the exact rules that ran, and a source-side merge would
    make the applied policy something Home never hashed.
    """

    if policy is not None:
        return normalize_export_policy(policy)
    return build_policy_document(
        channel="workspace_snapshot", protected_paths=tuple(protected_paths)
    )


def snapshot_workspace(
    root: pathlib.Path,
    *,
    protected_paths: tuple[str, ...] = (),
    policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Return blobs only after two exact content/git manifest observations."""

    document = snapshot_policy(policy, tuple(protected_paths))
    previous: dict[str, Any] | None = None
    previous_blobs: dict[str, bytes] = {}
    for attempt in range(2):
        manifest, blobs = _snapshot_once(root, document)
        manifest["attempt"] = attempt + 1
        if previous is not None and previous["fingerprint"] == manifest["fingerprint"]:
            return manifest, blobs
        previous, previous_blobs = manifest, blobs
    assert previous is not None
    previous["complete"] = False
    previous["materializable"] = False
    previous["integrity_complete"] = False
    previous["unstable"] = True
    failures = list(previous.get("failures") or [])
    failures.append({"path": "", "reason": "unstable_observation"})
    previous["failures"] = failures
    previous["failure_count"] = len(failures)
    return previous, previous_blobs


def snapshot_operation(
    root: pathlib.Path,
    *,
    protected_paths: tuple[str, ...] = (),
    policy: Mapping[str, Any] | None = None,
) -> NativeOperationResult:
    """Project a stable snapshot into the native operation wire contract."""

    manifest, blobs = snapshot_workspace(
        root, protected_paths=protected_paths, policy=policy
    )
    state = "complete" if manifest["complete"] else "partial"
    return NativeOperationResult(
        ToolExecutionEnvelope(
            text=json.dumps(manifest, sort_keys=True),
            artifacts=tuple(
                {
                    "path": row["path"],
                    "blob_id": row["sha256"],
                    "sha256": row["sha256"],
                    "size": row["size"],
                    "mode": row["mode"],
                    "kind": row["kind"],
                }
                for row in manifest["entries"]
            ),
            trace={"snapshot": manifest, "completion": state},
        ),
        blobs,
    )


def guarded_patch_apply(
    root: pathlib.Path,
    args: Mapping[str, Any],
    blobs: Mapping[str, bytes],
    *,
    protected_paths: tuple[str, ...] = (),
    policy: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    """Check all preconditions, apply once, and restore exact originals on failure."""

    document = snapshot_policy(policy, tuple(protected_paths))
    with _root_lock(root):
        current, current_blobs = snapshot_workspace(
            root,
            policy=document,
        )
        if not snapshot_integrity_ready(current):
            return ToolExecutionEnvelope(
                text=(
                    "⚠️ REMOTE_SNAPSHOT_INTEGRITY_FAILED: guarded apply "
                    "requires an integrity-complete source snapshot."
                ),
                trace={"completion": "not_started", "snapshot": current},
            )
        expected = str(args.get("expected_fingerprint") or "")
        if not expected or current["fingerprint"] != expected:
            return _conflict_envelope(expected, current)
        expected_head = str(args.get("expected_head") or "")
        expected_index = str(args.get("expected_index_sha256") or "")
        git_facts = current.get("git") if isinstance(current.get("git"), dict) else {}
        if expected_head and expected_head != str(git_facts.get("head") or ""):
            return _conflict_envelope(expected, current, "HEAD changed")
        if expected_index and expected_index != str(git_facts.get("index_sha256") or ""):
            return _conflict_envelope(expected, current, "index changed")
        patch = _patch_blob(args, blobs)
        changes = _validated_changes(root, args.get("changes"), current, document)
        declared_paths = {str(change["path"]) for change in changes}
        touched_paths = _patch_numstat_paths(root, patch, reverse=False)
        touched_paths.update(_patch_numstat_paths(root, patch, reverse=True))
        if not touched_paths:
            raise ValueError("cannot prove patch paths: patch touched no paths")
        if touched_paths != declared_paths:
            missing = sorted(touched_paths - declared_paths)
            extra = sorted(declared_paths - touched_paths)
            raise ValueError("patch paths do not exactly match declared changes " f"(missing={missing}, extra={extra})")
        check = _git_apply(root, patch, check=True)
        if check.returncode:
            return ToolExecutionEnvelope(
                text=("⚠️ REMOTE_PATCH_CHECK_FAILED: " + check.stderr.decode("utf-8", errors="replace")),
                trace={"completion": "not_started", "snapshot": current},
            )
        rollback = _rollback_rows(changes, current, current_blobs)
        try:
            applied = _git_apply(root, patch, check=False)
            if applied.returncode:
                raise RuntimeError(applied.stderr.decode("utf-8", errors="replace") or "git apply failed")
            after, _ = snapshot_workspace(
                root,
                policy=document,
            )
            if not snapshot_integrity_ready(after):
                raise RuntimeError(
                    "remote post-state snapshot is partial or unstable"
                )
            expected_content = str(args.get("expected_content_fingerprint") or "")
            if (
                (expected_content and after.get("content_fingerprint") != expected_content)
                or (expected_head and str(after.get("git", {}).get("head") or "") != expected_head)
                or (expected_index and str(after.get("git", {}).get("index_sha256") or "") != expected_index)
            ):
                raise RuntimeError("remote post-state does not match the reviewed mirror")
        except Exception as exc:
            rollback_errors = _restore_rows(root, rollback)
            message = f"{type(exc).__name__}: {exc}"
            if rollback_errors:
                return ToolExecutionEnvelope(
                    text=(
                        "⚠️ ROLLBACK_FAILED: guarded remote apply failed and "
                        f"rollback was incomplete: {message}; {rollback_errors}"
                    ),
                    trace={
                        "completion": "unknown",
                        "rollback_failed": rollback_errors,
                    },
                )
            return ToolExecutionEnvelope(
                text=f"⚠️ REMOTE_PATCH_ROLLED_BACK: {message}",
                trace={"completion": "not_started", "rollback": "complete"},
            )
        return ToolExecutionEnvelope(
            text="OK: guarded remote patch applied.",
            trace={
                "completion": "complete",
                "before": current,
                "after": after,
                "changed": len(changes),
            },
        )


def _snapshot_once(
    root: pathlib.Path,
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    entries: list[dict[str, Any]] = []
    blobs: dict[str, bytes] = {}
    policy_exclusions: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    total = 0
    protected = tuple(document.get("protected_paths") or ())

    def record_failure(path: str, reason: str) -> None:
        row = {"path": path, "reason": reason}
        if row not in failures:
            failures.append(row)

    def walk_error(exc: OSError) -> None:
        raw = pathlib.Path(str(getattr(exc, "filename", "") or root))
        try:
            rel = raw.relative_to(root).as_posix()
        except ValueError:
            rel = ""
        record_failure(rel, "walk_error")

    # ONE alias index for the whole walk. This channel's identity check was INLINE here —
    # and only here, which is why the snapshot got aliases right while `search_code`
    # returned the secret LINE through the same trick and `read_file` returned the whole
    # file. It is `export_policy_contract.judged_exclusion` now, the same call every door
    # makes: a rule that lives in one walk is a rule the next walk lacks.
    aliases = AliasIndex(root, document)

    for dirpath, dirnames, filenames in os.walk(
        root,
        followlinks=False,
        onerror=walk_error,
    ):
        current = pathlib.Path(dirpath)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            path = current / name
            rel = path.relative_to(root).as_posix()
            directory_reason, _sentence, directory_judged = judged_exclusion(
                root, path, rel, document, question=QUESTION_EXPORT, aliases=aliases
            )
            # A whole protected subtree is disclosed as ONE exclusion and pruned;
            # build/VCS directories are pruned silently (they were never exportable
            # content, so counting them would drown the disclosed list). Anything
            # else recurses and is judged file by file, which keeps the disclosed
            # paths precise instead of collapsing a directory the owner can inspect.
            if directory_reason == REASON_PROTECTED_ARTIFACT:
                policy_exclusions.append({
                    "path": rel,
                    "reason": REASON_PROTECTED_ARTIFACT,
                    "judged": directory_judged or rel,
                })
            elif directory_reason == REASON_EXCLUDED_DIRECTORY:
                continue
            elif path.is_symlink():
                filenames.append(name)
            else:
                kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            path = current / name
            rel = path.relative_to(root).as_posix()
            if len(entries) + len(policy_exclusions) >= MAX_SNAPSHOT_FILES:
                record_failure(rel, "file_limit_exceeded")
                break
            # Spelling, resolved identity and alias in ONE call: an alias (hardlink or
            # second name) for an excluded inode exports the same bytes under a harmless
            # name, and the identity half is what makes the rule about CONTENT.
            reason, _sentence, judged = judged_exclusion(
                root, path, rel, document, question=QUESTION_EXPORT, aliases=aliases
            )
            if reason:
                # `judged` is the spelling the policy excluded, which is the entry's own
                # name unless an ALIAS was the finding. Home holds no workspace and cannot
                # re-derive it, so a row without it is a claim nothing can check.
                policy_exclusions.append(
                    {"path": rel, "reason": reason, "judged": judged or rel}
                )
                continue
            try:
                row, data = _read_stable_entry(path, rel, root)
            except RuntimeError:
                record_failure(rel, "changed_during_read")
                continue
            except OSError as exc:
                reason = (
                    "unsafe_symlink"
                    if "symlink escapes" in str(exc)
                    else (
                        "unsupported_file_kind"
                        if "unsupported file kind" in str(exc)
                        else "entry_read_error"
                    )
                )
                record_failure(rel, reason)
                continue
            total += len(data)
            if total > MAX_SNAPSHOT_BYTES:
                record_failure(rel, "byte_limit_exceeded")
                break
            entries.append(row)
            blobs.setdefault(row["sha256"], data)
        if failures:
            break
    entries.sort(key=lambda row: row["path"])
    policy_exclusions = sorted(
        {(
            str(row["path"]),
            str(row["reason"]),
            str(row.get("judged") or row["path"]),
        ) for row in policy_exclusions}
    )
    policy_rows = [
        dict(zip(MANIFEST_EXCLUSION_ROW_FIELDS, row)) for row in policy_exclusions
    ]
    integrity_complete = not failures
    policy_scope = "policy_filtered" if policy_rows else "full"
    complete = integrity_complete and policy_scope == "full"
    git_facts = _git_facts(root)
    content_fingerprint = hashlib.sha256(
        json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    policy_hash = export_policy_hash(document)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "entries": entries,
                "git": git_facts,
                "policy_exclusions": policy_rows,
                "policy_hash": policy_hash,
                "protected_paths": list(protected),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return (
        {
            "schema_version": 3,
            "entries": entries,
            "fingerprint": fingerprint,
            "content_fingerprint": content_fingerprint,
            "git": git_facts,
            "complete": complete,
            "materializable": integrity_complete,
            "integrity_complete": integrity_complete,
            "policy_scope": policy_scope,
            # The hash of the policy that was actually applied. Home compares it
            # with the one it computed and hashed into the prepared operation; a
            # manifest produced under a different policy is a typed refusal, not
            # a partial import, because nobody can say what was filtered.
            "policy_hash": policy_hash,
            "unstable": False,
            "protected_paths": list(protected),
            "policy_exclusions": policy_rows,
            "policy_excluded_count": len(policy_rows),
            "exclusions": policy_rows,
            "excluded_count": len(policy_rows),
            "failures": failures,
            "failure_count": len(failures),
            "total_bytes": total,
        },
        blobs,
    )


def _read_stable_entry(
    path: pathlib.Path,
    rel: str,
    root: pathlib.Path,
) -> tuple[dict[str, Any], bytes]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        link_target = os.readlink(path)
        resolved_target = (path.parent / link_target).resolve(strict=False)
        if pathlib.Path(link_target).is_absolute() or not _path_inside(
            resolved_target,
            root,
        ):
            raise OSError("symlink escapes workspace")
        data = link_target.encode("utf-8", errors="surrogateescape")
        kind = "symlink"
    elif stat.S_ISREG(before.st_mode):
        data = path.read_bytes()
        kind = "file"
    else:
        raise OSError("unsupported file kind")
    after = path.lstat()
    if (
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
    ) != (
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise RuntimeError("file changed during snapshot")
    return (
        {
            "path": rel,
            "kind": kind,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "mode": stat.S_IMODE(before.st_mode),
        },
        data,
    )


def _git_facts(root: pathlib.Path) -> dict[str, str]:
    head = _git_bytes(
        root,
        ["rev-parse", "--verify", "HEAD"],
        allow_failure=True,
    )
    index = _git_bytes(
        root,
        ["ls-files", "--stage", "-z"],
        allow_failure=True,
    )
    status = _git_bytes(
        root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        allow_failure=True,
    )
    return {
        "head": head.decode("utf-8", errors="replace").strip(),
        "unborn": "true" if not head else "false",
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }

def _validated_changes(
    root: pathlib.Path,
    raw: Any,
    current: Mapping[str, Any],
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("changes must be a list")
    current_rows = {
        str(row.get("path") or ""): row for row in list(current.get("entries") or []) if isinstance(row, dict)
    }
    changes: list[dict[str, Any]] = []
    seen: set[str] = set()
    # A path the SOURCE snapshot omitted is off-limits too, not only a path the
    # rules name: the reviewed mirror never contained it, so a change against it
    # was never reviewed. The reasons are the same closed set the policy emits.
    excluded_paths = tuple(
        str(row.get("path") or "")
        for row in list(current.get("exclusions") or [])
        if isinstance(row, dict) and str(row.get("path") or "")
    )
    aliases = AliasIndex(root, document)
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("change rows must be objects")
        path = _safe_relpath(item.get("path"))
        # THE judge, not the spelling: a declared change against an alias of an excluded
        # path is a change against the excluded path, and this admission gate is what
        # stands between a rollback row and a write.
        if judged_exclusion(
            root,
            root.joinpath(*path.split("/")),
            path,
            document,
            question=QUESTION_EXPORT,
            aliases=aliases,
        )[0] or path_under_any(path, excluded_paths):
            raise ValueError(f"change targets an omitted policy path: {path}")
        if path in seen:
            raise ValueError(f"duplicate change path: {path}")
        seen.add(path)
        before = item.get("before")
        after = item.get("after")
        if before is not None and not isinstance(before, dict):
            raise ValueError("change before state must be an object or null")
        if after is not None and not isinstance(after, dict):
            raise ValueError("change after state must be an object or null")
        if current_rows.get(path) != before:
            raise ValueError(f"change precondition mismatch: {path}")
        changes.append({"path": path, "before": before, "after": after})
    return changes


def snapshot_integrity_ready(manifest: Mapping[str, Any]) -> bool:
    failures = manifest.get("failures")
    failure_count = manifest.get("failure_count")
    return (
        manifest.get("integrity_complete") is True
        and manifest.get("materializable") is True
        and manifest.get("unstable") is False
        and isinstance(failures, list)
        and not failures
        and type(failure_count) is int
        and failure_count == 0
    )


def _rollback_rows(
    changes: list[dict[str, Any]],
    current: Mapping[str, Any],
    blobs: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    del current
    rows: list[dict[str, Any]] = []
    for change in changes:
        before = change["before"]
        data = None if before is None else blobs.get(str(before.get("sha256") or ""))
        if before is not None and data is None:
            raise ValueError(f"rollback blob is unavailable: {change['path']}")
        rows.append({"path": change["path"], "before": before, "data": data})
    return rows


def _restore_rows(root: pathlib.Path, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        try:
            # The SAME door every other native mutation uses. `_safe_relpath` already
            # refused `..` LEXICALLY, but a lexically clean spelling still walks through
            # whatever a component currently points at, and this loop deletes, writes,
            # chmods and symlinks. It was the one native mutation site left outside the
            # confinement kernel, which is precisely how the `write_file` append escape
            # started — a mutation whose caller happened to make it unreachable. The
            # errors this raises are collected like any other rollback error, so a refused
            # row is reported rather than silently skipped.
            # `facts=None` states an UNBOUND mutation, and it is the honest answer here
            # rather than a gap: a rollback restores the bytes the reviewed snapshot
            # already held, and `_validated_changes` refused any change naming a policy
            # path before this loop could ever see one.
            target = native_mutation_target(root, row["path"], facts=None)
            before = row["before"]
            if before is None:
                _remove_path(target)
                continue
            _remove_path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = bytes(row["data"])
            if before["kind"] == "symlink":
                os.symlink(data.decode("utf-8", errors="surrogateescape"), target)
            else:
                target.write_bytes(data)
                os.chmod(target, int(before["mode"]) & 0o777)
        except Exception as exc:
            errors.append(f"{row['path']}: {type(exc).__name__}: {exc}")
    return errors


def _patch_blob(args: Mapping[str, Any], blobs: Mapping[str, bytes]) -> bytes:
    blob_id = str(args.get("patch_blob_id") or "")
    if not blob_id:
        raise ValueError("patch_blob_id is required")
    patch = blobs.get(blob_id)
    if patch is None or hashlib.sha256(patch).hexdigest() != blob_id:
        raise ValueError("declared patch blob is unavailable or invalid")
    return bytes(patch)


def _git_apply(
    root: pathlib.Path,
    patch: bytes,
    *,
    check: bool,
) -> subprocess.CompletedProcess[bytes]:
    command = ["git", "apply"]
    if check:
        command.append("--check")
    command.append("-")
    return subprocess.run(
        command,
        cwd=str(root),
        input=patch,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )


def _patch_numstat_paths(
    root: pathlib.Path,
    patch: bytes,
    *,
    reverse: bool,
) -> set[str]:
    command = ["git", "apply", "--numstat", "-z"]
    if reverse:
        command.append("--reverse")
    command.append("-")
    proc = subprocess.run(
        command,
        cwd=str(root),
        input=patch,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if proc.returncode:
        error = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot prove patch paths: {error or 'git rejected patch'}")
    raw = bytes(proc.stdout)
    if not raw or not raw.endswith(b"\0"):
        raise ValueError("cannot prove patch paths: malformed Git numstat output")
    records = raw.split(b"\0")
    records.pop()
    paths: set[str] = set()
    index = 0
    while index < len(records):
        fields = records[index].split(b"\t", 2)
        index += 1
        if len(fields) != 3 or not all(value == b"-" or value.isdigit() for value in fields[:2]):
            raise ValueError("cannot prove patch paths: malformed Git numstat record")
        encoded_path = fields[2]
        if encoded_path:
            paths.add(_safe_relpath(os.fsdecode(encoded_path)))
            continue
        if index + 1 >= len(records):
            raise ValueError("cannot prove patch paths: incomplete rename record")
        old_path, new_path = records[index : index + 2]
        index += 2
        if not old_path or not new_path:
            raise ValueError("cannot prove patch paths: empty rename path")
        paths.add(_safe_relpath(os.fsdecode(old_path)))
        paths.add(_safe_relpath(os.fsdecode(new_path)))
    return paths


def _git_bytes(
    root: pathlib.Path,
    args: list[str],
    *,
    allow_failure: bool,
) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if proc.returncode and not allow_failure:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
    return bytes(proc.stdout) if proc.returncode == 0 else b""


def _safe_relpath(value: Any) -> str:
    text = str(value or "").replace("\\", "/")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("unsafe change path")
    return "/".join(parts)


def _root_lock(root: pathlib.Path) -> threading.Lock:
    key = str(root.resolve(strict=False))
    with _LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.Lock())


def _remove_path(path: pathlib.Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _path_inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _conflict_envelope(
    expected: str,
    current: Mapping[str, Any],
    reason: str = "workspace changed",
) -> ToolExecutionEnvelope:
    return ToolExecutionEnvelope(
        text=(
            "⚠️ SNAPSHOT_FINGERPRINT_MISMATCH: "
            f"{reason} (expected={expected or '<missing>'}, "
            f"actual={current.get('fingerprint') or '<missing>'})."
        ),
        trace={"completion": "not_started", "snapshot": dict(current)},
    )
