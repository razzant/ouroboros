"""Separate result adoption from bulk retention using the existing custody ledger.

The answer, output files, raw QA and verification receipts are completion work.
Historical call/source closure belongs to the off-loop child-ref retry owner. Its
pending row holds the execution drive across restart; readers can still resolve exact
sources there until canonical copies have been verified.
"""

from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from ouroboros.utils import append_jsonl, utc_now_iso


DEFERRED_KIND = "history_retention_deferred"
_EVIDENCE_GAP_KIND = "test_evidence"
_EVIDENCE_GAP_ROWS = 20  # transient facts one result row keeps (the rest is one count); deterministic ones stay exact
_EVIDENCE_COLLISION = "different or unreadable bytes already hold this canonical address"


def prepare_result_retention(parent: Path, child: Path, task_id: str,
                             result: dict[str, Any]) -> tuple[dict, dict]:
    """Copy deliverables, retain source custody, and leave history to maintenance."""
    from ouroboros.artifacts import promote_task_attachment_refs
    from ouroboros.headless import _copy_child_artifacts_to_parent

    copied = copy.deepcopy(result)
    separate = parent.resolve(strict=False) != child.resolve(strict=False)
    state = {"schema_version": 1, "status": "incomplete" if separate else "complete",
             "promoted_ref_count": 0, "promoted_source_handle_count": 0,
             "pending_refs": [], "unavailable_refs": []}
    if not separate:
        return copied, state
    state["pending_refs"].append({"kind": DEFERRED_KIND, "path": str(child.resolve(strict=False)),
                                  "reason": "background_history_retention"})
    promote_task_attachment_refs(parent, child, task_id, copied, state)
    if isinstance(copied.get("artifacts"), list):
        copied["artifacts"] = _copy_child_artifacts_to_parent(
            parent, task_id, child, copied["artifacts"], promotion=state)
    carry_raw_evidence(parent, child, task_id, state)
    # Logs is diagnostic only: the task-result publication owns the durable duty.
    event = {
        "ts": utc_now_iso(), "type": "history_retention", "task_id": task_id,
        **retention_summary({"child_ref_promotion": state}),
        "diagnostics": retention_diagnostics({"child_ref_promotion": state}),
        **{key: result[key] for key in ("chat_id", "project_id", "parent_task_id", "root_task_id") if key in result},
    }
    try:
        append_jsonl(parent / "logs" / "events.jsonl", event)
    except OSError:
        pass  # The task-result duty still owns custody if the diagnostic log is unavailable.
    return copied, state


def carry_raw_evidence(parent: Path, child: Path, task_id: str, state: dict[str, Any]) -> int:
    """Carry a child-local ``source_handles/test-evidence`` to the same canonical relpaths; files placed.

    A context-aware gate writes this family canonical-first, but ``safe_test --evidence-dir`` may select the
    execution drive's copy, which no row, typed reference or store listing carries: an authored index could
    outlive the only files it names. Copy-back, its retry and settlement therefore carry this one writer's
    family, under the task's custody lock and without a row per stream; the other reserved families keep their
    owners, and the index is neither read nor rewritten. Identical bytes are already delivered. Other bytes at
    the address, a family or folder custody cannot read (a link at or above the family included), an unreadable
    file or a refused placement become a pending fact in ``state``: its original holds the drive
    (``task_custody._pending_refs_under``), the parent reads ``raw_evidence_notice``, and nothing is placed beside
    or over the address. Links are never followed on either side; as in the store listing, a member that is no
    regular file (a link, a FIFO) carries no bytes of its own and is left out.
    """
    import os
    import stat

    from ouroboros.artifacts import copy_artifact_file, stream_artifact_file, task_artifact_dir_path
    from ouroboros.observability import _append_promotion_fact, _promotion_fact
    from ouroboros.store import is_wal_index
    from ouroboros.task_custody import _STAGING_DIR, PublicationClosed, _publish, _Retained, _verified

    root, drive = Path(parent).resolve(strict=False), Path(child).resolve(strict=False)
    if root == drive:
        return 0  # one store: nothing to carry
    store = task_artifact_dir_path(drive, task_id, create=False)
    source = store / "source_handles" / "test-evidence"  # the path ``test_evidence.evidence_directory`` writes

    def gap(path: Any, reason: Any, **facts: Any) -> dict[str, Any]:
        text = reason if isinstance(reason, str) else f"{type(reason).__name__}: {reason}"
        return _promotion_fact({"kind": _EVIDENCE_GAP_KIND, "path": str(path), **facts}, text)

    gaps: list[dict[str, Any]] = []
    members: list[Path] = []

    def unlisted(exc: OSError) -> None:
        gaps.append(gap(exc.filename or source, exc))  # this folder only: the readable rest still arrives

    try:
        info = source.lstat()  # ``is_dir`` raises on an unsearchable parent; ``lexists`` would call that absence
    except (FileNotFoundError, NotADirectoryError):
        return 0  # no family in this store
    except OSError as exc:
        gaps.append(gap(source, exc))  # hidden by an unsearchable folder above it: unreadable, never absent
    else:
        if source.resolve(strict=False) != source:
            link = drive
            for part in source.relative_to(drive).parts:
                link = link / part
                if link.is_symlink():
                    break
            try:
                text = os.readlink(link)
            except OSError:
                text = "unreadable link text"
            # Named by the real folder holding the link: a pending path must resolve inside the drive it holds.
            # Deterministic: kept, and walked again only once that folder or the link changes (``RetentionMemo``).
            gaps.append({**gap(link.parent, f"reached through a link, which custody never follows: "
                                            f"{link.relative_to(drive).as_posix()} -> {text}"),
                         "failure_kind": "invalid_source", "watched_paths": [str(link)]})
        elif stat.S_ISDIR(info.st_mode):
            for folder, dirs, files in os.walk(source, onerror=unlisted):  # never into a linked folder
                dirs.sort()
                for name in sorted(files):
                    path = Path(folder) / name
                    try:  # what the store listing admits: regular files (never through a link), no WAL index
                        if stat.S_ISREG(path.lstat().st_mode) and not is_wal_index(path):
                            members.append(path)
                    except OSError as exc:
                        gaps.append(gap(path, exc))
        else:
            return 0  # its writer makes only directories: a file of this name is not that family
    target = task_artifact_dir_path(root, task_id, create=False)
    staging = root / _STAGING_DIR / f"{task_id}-{uuid.uuid4().hex[:12]}"
    placed = 0
    try:
        for path in members:
            rel = path.relative_to(store).as_posix()
            dest = target.joinpath(*rel.split("/"))
            try:
                if dest.parent.resolve(strict=False) != dest.parent or dest.is_symlink():
                    raise _Retained("canonical_store_unsafe")  # a linked component would redirect or fake it
                if dest.exists():
                    observed = stream_artifact_file(path)
                    if _verified(dest, observed) is None:
                        # Deterministic (``source_retention.InvalidRetainedSource``): retried once a file changes.
                        gaps.append({**gap(path, _EVIDENCE_COLLISION, **observed),
                                     "failure_kind": "invalid_source", "watched_paths": [str(dest)]})
                    continue
                staged = staging / str(placed)
                staged.parent.mkdir(parents=True, exist_ok=True)
                copy_artifact_file(path, staged)
                _publish(root, task_id, [{"rel": rel, "staged": staged, "dest_identity": None}])
                placed += 1
            except PublicationClosed:
                raise
            except (OSError, _Retained) as exc:
                gaps.append(gap(path, str(exc) if isinstance(exc, _Retained) else exc))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    # A deterministic fact keeps its exact watched paths: they let ``RetentionMemo`` skip an unchanged retry and
    # notice a repair of any one of them. A transient one is retried anyway, so only those are counted past the bound.
    transient = 0
    for fact in gaps:
        transient += fact.get("failure_kind") != "invalid_source"
        if fact.get("failure_kind") == "invalid_source" or transient <= _EVIDENCE_GAP_ROWS:
            _append_promotion_fact(state["pending_refs"], fact)
    if transient > _EVIDENCE_GAP_ROWS:
        _append_promotion_fact(state["pending_refs"], gap(source, f"{transient - _EVIDENCE_GAP_ROWS} more raw test "
                                                          "evidence facts; every original is under this folder"))
    state["promoted_source_handle_count"] = state.get("promoted_source_handle_count", 0) + placed
    return placed


def raw_evidence_custodied(parent: Path, child: Path, task_id: str, current: dict[str, Any]) -> int | None:
    """Settlement's carry, under its custody lock: files placed, or None once a gap was armed on the row,
    where the parent reads it and the same background duty ``call_inventory_custodied`` arms retries it."""
    from ouroboros.observability import _append_promotion_fact
    from ouroboros.task_custody import attempt_basis, fence_publication
    from ouroboros.task_results import write_task_result

    found: dict[str, Any] = {"pending_refs": []}
    placed = carry_raw_evidence(parent, child, task_id, found)
    if not found["pending_refs"]:
        return placed

    def owe_gap(row, _incoming):
        fence_publication()
        if attempt_basis(row) != attempt_basis(current):
            return None
        state = copy.deepcopy(row.get("child_ref_promotion") or {})
        pending = list(state.get("pending_refs") or [])
        for fact in found["pending_refs"]:
            _append_promotion_fact(pending, fact)
        if state.get("status") == "incomplete" and pending == state.get("pending_refs"):
            return None  # these facts are on the row already: a later pass rewrites nothing
        state.update(schema_version=1, status="incomplete", pending_refs=pending)
        return {"status": row["status"], "child_ref_promotion": state}

    write_task_result(parent, task_id, current["status"], _field_projector=owe_gap, strict_existing_dict=True)
    return None


def raw_evidence_notice(result: Any, *, limit: int = 5) -> str:
    """One bounded host line naming raw QA that is not at its canonical address (``carry_raw_evidence``)."""
    promotion = result.get("child_ref_promotion")
    rows = [row for row in (promotion.get("pending_refs") or []) if isinstance(row, dict)
            and row.get("kind") == _EVIDENCE_GAP_KIND] if isinstance(promotion, dict) else []
    if not rows:
        return ""
    # JSON: a file name is the child's text, and a newline in it must not forge a host line.
    shown = "\n".join("- " + json.dumps({"path": row.get("path"), "reason": row.get("reason")}, ensure_ascii=False)
                      for row in rows[:limit])
    more = f"\n- ... {len(rows) - limit} more" if len(rows) > limit else ""
    return (f"[TEST_EVIDENCE_GAP] {len(rows)} raw test evidence fact(s): these sources did not reach this task's "
            "canonical store, so an index naming them proves no bytes there. Each path is an original kept on the "
            "execution drive, read there by absolute path; its reason says why it was not carried:\n"
            f"{shown}{more}\nThe row keeps each conflict or link exactly; transient failures past {_EVIDENCE_GAP_ROWS} "
            "are counted under the folder holding them: "
            "get_task_result(include_authority=True) -> authority.child_ref_promotion.pending_refs.")


def retention_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Full existing failure facts for Logs/raw detail, excluding routine deferral."""
    state = result.get("child_ref_promotion")
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        return {}
    pending = state.get("pending_refs") or []
    problems = [row for row in pending if isinstance(row, dict) and row.get("kind") != DEFERRED_KIND]
    unavailable = state.get("unavailable_refs") or []
    return {"pending_refs": copy.deepcopy(problems), "unavailable_refs": copy.deepcopy(unavailable)}


def retention_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Group failure reasons for display; preserve every full fact in diagnostics."""
    from collections import Counter

    diagnostics = retention_diagnostics(result)
    if not diagnostics:
        return {}
    state = result["child_ref_promotion"]
    pending = state.get("pending_refs") or []
    problems, unavailable = diagnostics["pending_refs"], diagnostics["unavailable_refs"]
    reasons = Counter(str(row.get("reason") or "No failure reason was recorded.")
                      for row in (*problems, *unavailable) if isinstance(row, dict))
    return {"status": "problem" if problems or unavailable else
            "complete" if state.get("status") == "complete" else "pending",
            "pending_count": len(pending), "problem_count": len(problems) + len(unavailable),
            "problem_reasons": [{"reason": reason, "count": count} for reason, count in reasons.items()],
            "promoted_ref_count": state.get("promoted_ref_count", 0),
            "promoted_source_handle_count": state.get("promoted_source_handle_count", 0)}


def call_inventory_custodied(parent: Path, child: Path, task_id: str,
                             current: dict[str, Any], *, stop=None) -> bool:
    """Arm the same background duty before GC of old/cancelled or newly grown history.

    Manifest writers atomically replace files, changing their directory revision.
    Compare that cheap revision here; scanning and copying calls remains off the
    completion path. The no-live custody check owns whether GC may proceed.
    """
    from ouroboros.task_custody import attempt_basis, fence_publication, publication_fence, task_custody_lock
    from ouroboros.task_results import write_task_result

    directory = child / "observability" / "calls" / task_id
    try:
        revision = directory.stat().st_mtime_ns
    except FileNotFoundError:
        return True
    promotion = current.get("child_ref_promotion") or {}
    if promotion.get("call_inventory_preserved") is True and promotion.get("call_inventory_mtime_ns") == revision:
        return True
    with publication_fence(stop), task_custody_lock(parent, task_id, timeout_sec=0.5) as locked:
        if not locked:
            return False
        def owe(row, _incoming):
            fence_publication()
            if attempt_basis(row) != attempt_basis(current):
                return None
            state = copy.deepcopy(row.get("child_ref_promotion") or {})
            pending = list(state.get("pending_refs") or [])
            marker = {"kind": DEFERRED_KIND, "path": str(child), "reason": "background_history_retention"}
            if marker not in pending:
                pending.append(marker)
            state.update(schema_version=1, status="incomplete", pending_refs=pending)
            return {"status": row["status"], "child_ref_promotion": state}
        write_task_result(parent, task_id, current["status"], _field_projector=owe, strict_existing_dict=True)
    return False
