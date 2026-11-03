"""The Git cleanup of an ended evolution cycle, and the one fact it leaves when it moved something.

A cycle without a candidate gets its base back: the whole dirty checkout goes to a stash and another
HEAD to a local branch before the reset (owner 1 Nov 2026, quiz fb45adf3: keep the action, show the
result). Whose bytes they were is unknown: the stash holds the whole tree. So the fact names what Git
did and where the bytes went, never their author, how long Git keeps them or how often a row was sent.

``evolution_lifecycle`` decides when this runs, owns the campaign file and supplies its data root; this
module owns the guards in front of the move, the move, and its publication: the transaction's
``cleanup_*`` fields, the task result's ``outcome_axes.execution.evolution_cleanup`` (the card's one short
line) and one row of the cycle's card owed through the terminal outbox (the exact addresses; a row alone
on a page without that card). "Moved to a stash" rests on reading the stash back under its exact label
after the push, whatever it exited with: exit 0 also answers dirt Git cannot stash, a failure can follow a
saved entry and removed files, and an unread list is said as unconfirmed, never as either outcome. A branch
keeps its own commit (``cleanup_preserved_ref``, ``cleanup_preserved_head``), apart from the checkout HEAD
this attempt saw (``cleanup_head``). Live writers are the queue's RUNNING roster: every live focus
(project task, conversation-lane turn, inline executor) owns a row there.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import truncate_review_artifact, utc_now_iso

log = logging.getLogger(__name__)


def cleanup_fields(tx: Dict[str, Any]) -> Dict[str, Any]:
    """The transaction's cleanup evidence, as both campaign paths persist it."""
    return {key: value for key, value in tx.items() if key.startswith("cleanup_") or key == "recovery_hint"}


def _labelled_stashes(git_ops: Any, label: str) -> Tuple[bool, List[str]]:
    """(readable, this label's stash OIDs, newest first). The label carries the transaction id, so a
    replay finds what an interrupted attempt moved; other sessions' entries in the shared stack never match."""
    rc, top, _ = git_ops.git_capture(["git", "for-each-ref", "--format=%(objectname)", "refs/stash"])
    if rc != 0:
        return False, []
    if not top.strip():
        return True, []
    rc, listing, _ = git_ops.git_capture(["git", "log", "-g", "--format=%H %gs", "refs/stash", "--"])
    if rc != 0:
        return False, []
    return True, [line.split(" ", 1)[0] for line in listing.splitlines() if line.endswith(f": {label}")]


def _preserved_branch(git_ops: Any, prefix: str) -> Tuple[str, str]:
    """(branch, commit) an earlier attempt of this transaction kept last (its name starts with the time), else empty."""
    rc, listing, _ = git_ops.git_capture([
        "git", "for-each-ref", "--count=1", "--sort=-refname", "--format=%(refname:short) %(objectname)",
        f"refs/heads/{prefix}-*"])
    fields = listing.split() if rc == 0 else []
    return (fields[0], fields[1]) if len(fields) == 2 else ("", "")


def _cleanup_worktree_after_cycle(tx: Dict[str, Any], task_id: str) -> None:
    """Restore an ended cycle's base, preserving dirty/ahead work in stash/local refs.
    Unadmitted cycles, other writers, unknown base, live tests or disabled cleanup
    retain their reason. Never raises (OUROBOROS_EVOLUTION_CYCLE_CLEANUP=false)."""
    if str(os.environ.get("OUROBOROS_EVOLUTION_CYCLE_CLEANUP", "true") or "true").lower() in {"0", "false", "no", "off"}:
        tx["cleanup_status"] = "disabled"
        return
    if tx.get("admission_refused") and not tx.get("commit_sha"):
        tx["cleanup_status"] = "skipped_never_admitted"
        return
    base_head = str(tx.get("base_head") or "").strip()
    if not base_head:
        tx["cleanup_status"] = "skipped_no_base"
        return
    update_lock_fh = None
    release_update_lock = None
    try:
        from ouroboros import body_adoption, body_candidate
        from supervisor import git_ops, queue
        from supervisor.update_merge import acquire_update_lock, release_update_lock
        from supervisor.workers import repo_writer_admission_closed

        # Same protection class as git_ops._guard_live_repo_destructive_git, but
        # covering the stash too: a unit test that never re-pointed
        # git_ops.REPO_DIR must not stash/reset the LIVE repo's working tree.
        if os.environ.get("OUROBOROS_ALLOW_LIVE_REPO_TESTS") != "1":
            import sys as _sys
            try:
                live_repo = git_ops.REPO_DIR.resolve(strict=False) == (pathlib.Path.home() / "Ouroboros" / "repo").resolve(strict=False)
            except OSError:
                live_repo = False
            if live_repo and ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in _sys.modules):
                tx["cleanup_status"] = "skipped_live_repo_test_guard"
                return

        tx_key = str(tx.get("transaction_id") or task_id)
        stash_label = f"evolution-cycle-cleanup-{tx_key}"
        branch_prefix = "evolution-leftover-" + re.sub(r"[^A-Za-z0-9._-]", "-", tx_key)
        # What an interrupted attempt of this transaction already moved stays said, whatever this attempt decides:
        # these reads move nothing and come before every guard below that can end it.
        stashes_known, stashes = _labelled_stashes(git_ops, stash_label)
        kept_ref, kept_head = _preserved_branch(git_ops, branch_prefix)
        if not stashes_known:  # said beside any earlier hint, a rescue pointer included
            note, prior = "the stash list could not be read, so an earlier attempt's stash was not checked", \
                str(tx.get("recovery_hint") or "")
            tx["recovery_hint"] = prior if note in prior else "; ".join(part for part in (prior, note) if part)
        if stashes:
            tx["cleanup_stash"], tx["cleanup_stash_oids"] = stash_label, stashes
        if kept_ref:
            tx["cleanup_preserved_ref"], tx["cleanup_preserved_head"] = kept_ref, kept_head
        if stashes or kept_ref:
            tx.update(cleanup_repo=str(git_ops.current_repo_dir()), cleanup_at=utc_now_iso())

        candidate = body_candidate.find(str(task_id))
        if candidate is not None:                         # its work is in the retained candidate;
            tx["cleanup_status"] = "candidate_retained"    # serving dirt belongs to someone else
            body_adoption.abandon(pathlib.Path(queue.DRIVE_ROOT), "evolution_cycle_ended",
                                  task_id=str(task_id), candidate_id=str(candidate["candidate_id"]))
            return

        try:
            update_lock_fh = acquire_update_lock()
        except RuntimeError:
            tx["cleanup_status"] = "skipped_managed_update_lock_busy"
            return
        admission_reason = repo_writer_admission_closed()
        if admission_reason:
            tx["cleanup_status"] = "skipped_repo_writer_admission_closed"
            tx["cleanup_deferred_reason"] = admission_reason
            return

        # A lock-free RUNNING snapshot, read once: every live focus owns a row there (project tasks, the
        # conversation lane's turns and the inline executors alike), and a reset would destroy another
        # focus's work in the shared checkout. Admission on another thread can add a row just after this
        # read; that turn writes only after a model round, and nothing here fences against it.
        running_others = [tid for tid in list(queue.RUNNING.keys()) if str(tid) != str(task_id)]
        if running_others:
            # The bytes stay where they are; nothing retries this cleanup, and the next cycle starts from them.
            tx["cleanup_status"] = "skipped_other_tasks_running"
            return

        rc_status, status_out, _ = git_ops.git_capture(["git", "status", "--porcelain"])
        rc_head, head_out, _ = git_ops.git_capture(["git", "rev-parse", "HEAD"])
        if rc_status != 0 or rc_head != 0:
            tx["cleanup_status"] = "skipped_git_unavailable"
            return
        dirty = bool(status_out.strip())
        head = head_out.strip()
        if not dirty and head == base_head:  # a clean tree proves no absence of an earlier attempt's move
            tx["cleanup_status"] = "reset_to_base" if kept_ref else "stashed_dirty" if stashes else "already_clean"
            return

        tx.update(cleanup_repo=str(git_ops.current_repo_dir()), cleanup_at=utc_now_iso())
        if head != base_head:
            tx["cleanup_head"] = head
        if dirty:
            rc_stash, _, stash_err = git_ops.git_capture(
                ["git", "stash", "push", "--include-untracked", "-m", stash_label]
            )
            # The exit code proves neither outcome. Dirt only inside a nested repository exits 0 with no save, and
            # Git stores the entry before it clears the tree, so a failure can follow a save and a removed file.
            # The entries under this exact label after the push decide; an earlier attempt's stay named.
            after_known, after = _labelled_stashes(git_ops, stash_label)
            if after_known:
                tx.pop("cleanup_stash_unconfirmed", None)
            if after_known and after:
                tx["cleanup_stash"], tx["cleanup_stash_oids"] = stash_label, after
            elif not after_known:  # Git ran it; whether it saved an entry is unread, said as unconfirmed
                tx.update(cleanup_stash=stash_label, cleanup_stash_oids=stashes, cleanup_stash_unconfirmed=True)
            if rc_stash != 0:
                # Refuse to reset over what may be unsaved (P1: no silent loss), and say what it left.
                cause = truncate_review_artifact(str(stash_err or "").strip(), 400) or f"git stash push exited {rc_stash}"
                tx.update(cleanup_status="skipped_stash_failed", cleanup_error=cause,
                          recovery_hint=f"worktree dirty and stash failed: {cause}")
                tx["cleanup_stash_partial"] = not after_known or after != stashes
                return
            if head == base_head and not tx.get("cleanup_stash"):
                tx["cleanup_status"] = "dirty_not_stashable"  # read back: nothing under this label
                return

        if head != base_head:
            if not kept_ref or kept_head != head:
                preserved, ref_name = git_ops.preserve_local_ref_branch("HEAD", prefix=branch_prefix)
                if not preserved:
                    tx["cleanup_status"] = "skipped_preserve_failed"
                    tx["cleanup_error"] = truncate_review_artifact(str(ref_name or "").strip(), 400)
                    tx["recovery_hint"] = (
                        "HEAD ahead of base_head and preserve-branch failed; left as-is"
                        + (f" (dirty files already saved in stash {tx['cleanup_stash']})." if tx.get("cleanup_stash") else ".")
                    )
                    return
                tx["cleanup_preserved_ref"], tx["cleanup_preserved_head"] = ref_name, head
            rc_reset, _, reset_err = git_ops.git_capture(["git", "reset", "--hard", base_head])
            if rc_reset != 0:
                tx["cleanup_status"] = "reset_failed"
                tx["cleanup_error"] = truncate_review_artifact(str(reset_err or "").strip(), 400)
                tx["recovery_hint"] = f"git reset --hard {base_head[:12]} failed: {tx['cleanup_error']}"
                return
            tx["cleanup_status"] = "reset_to_base"
        else:  # HEAD already at base; only the stash happened, read back or not
            tx["cleanup_status"] = "stashed_dirty" if tx.get("cleanup_stash_oids") else "stash_unconfirmed"
        log.info(
            "Evolution cycle %s cleanup: worktree restored to base %s (stash=%s, preserved=%s)",
            tx_key, base_head[:12], tx.get("cleanup_stash") or "-", tx.get("cleanup_preserved_ref") or "-",
        )
    except Exception as exc:
        tx["cleanup_status"] = "error"
        tx["cleanup_error"] = _cause(exc)
        log.warning("Evolution cycle worktree cleanup failed", exc_info=True)
    finally:
        if update_lock_fh is not None and release_update_lock is not None:
            release_update_lock(update_lock_fh)


def _cause(exc: BaseException) -> str:
    return truncate_review_artifact(f"{type(exc).__name__}: {exc}", 400)


def _code(value: Any) -> str:
    """A literal in the row's Markdown, its bytes kept: a path, an id or a command never renders as formatting."""
    text = str(value)
    fence = "``" if "`" in text else "`"
    return f"{fence}{text}{fence}"


def _repository(repo: str) -> Tuple[str, str, str]:
    """(the row's Repository line, how its commands start, where they run). Only a path without a backtick or line
    break is exact in a code span on every surface: the card, a bubble and Telegram read a backtick inside one
    differently, and a line break in one becomes a space. Any other path, or none, is in no command to copy."""
    if repo and not re.search(r"[`\r\n]", repo):
        return f"- Repository: {_code(repo)}", f"git -C {shlex.quote(repo)}", " (POSIX shell)"
    shown = ("- Repository: not recorded" if not repo else "- Repository, escaped because it holds a backtick or line "
             "break: " + _code(json.dumps(repo, ensure_ascii=False).replace("`", "\\u0060")))
    return shown, "git", ", from inside that repository"


def cleanup_line(tx: Dict[str, Any]) -> str:
    """The short fact in owner words, without ids; empty when the cleanup moved nothing. A stash read back
    under its label is a move; one Git ran without that readback (an older record too) is unconfirmed."""
    stash = ("moved local changes to a Git stash" if tx.get("cleanup_stash_oids") else
             "ran a Git stash of local changes (whether it saved anything is unconfirmed)" if tx.get("cleanup_stash")
             else "")
    branch = "kept local commits on a branch" if tx.get("cleanup_preserved_ref") else ""
    if not stash and not branch:
        return ""
    saved = "Evolution cleanup " + " and ".join(part for part in (stash, branch) if part)
    newer = "the newer " if branch else ""
    return saved + {
        "reset_to_base": ", then reset the checkout to the cycle's starting commit",
        "reset_failed": ", but the reset to the cycle's starting commit failed",
        "skipped_preserve_failed": f", but could not keep {newer}local commits on a branch, so the checkout stays on them",
        "skipped_stash_failed": ", but the stash command failed, so the cleanup stopped there",
        "error": ", then stopped on an error",
    }.get(str(tx.get("cleanup_status") or ""), "")


def _row_text(tx: Dict[str, Any], line: str) -> str:
    """The card row: its first line heads it, the rest is its Details (``chat_activity.cardRowSummary``)."""
    base, status = str(tx.get("base_head") or ""), str(tx.get("cleanup_status") or "")
    label, oids = str(tx.get("cleanup_stash") or ""), list(tx.get("cleanup_stash_oids") or [])
    ref, head = str(tx.get("cleanup_preserved_ref") or ""), str(tx.get("cleanup_head") or "")
    kept = str(tx.get("cleanup_preserved_head") or head)  # an older record's branch shares its one HEAD field
    repository, git, where = _repository(str(tx.get("cleanup_repo") or ""))
    unconfirmed = bool(label) and (bool(tx.get("cleanup_stash_unconfirmed")) or not oids)
    partial = bool(tx.get("cleanup_stash_partial"))
    lines = [f"{line}.", "", repository]
    lines += [f"- Stash: {_code(oid)} ({_code(label)})" for oid in oids]
    if unconfirmed:
        pushed = "failed, and" if status == "skipped_stash_failed" else "exited without error, but"
        lines.append((f"- Whether this attempt saved another entry under {_code(label)}" if oids else
                      f"- {_code('git stash push')} under {_code(label)} {pushed} whether it saved an entry")
                     + f" was not confirmed. Look for it{where}: {_code(git + ' stash list')}")
    if ref:
        lines.append(f"- Branch: {_code(ref)} at {_code(kept)}")
    lines.append(f"- Starting commit: {_code(base)}")
    if head and status in {"reset_failed", "skipped_preserve_failed", "skipped_stash_failed", "error"}:
        lines.append(f"- The checkout is still at {_code(head)}")
    if tx.get("cleanup_error"):  # Git's own words on one line; they are read, never run
        lines.append(f"- What failed: {_code(' '.join(str(tx['cleanup_error']).split()))}")
    lines += [f"- Restore the changes{where}: {_code(f'{git} stash apply {oid}')}" for oid in oids]
    if partial and oids and not tx.get("cleanup_stash_unconfirmed"):  # an unread list may hide a newer entry than these
        lines.append("- While changed files are still in place, that command reports errors: it puts back missing "
                     "untracked files and leaves tracked files alone. To put tracked files back as saved, overwriting "
                     f"what they hold now{where}: {_code(f'{git} restore --source={oids[0]} --worktree -- :/')}")
    if label:
        lines += ["", ("Git can save the entry and remove files before a stash command fails, so some files may already "
                       "be gone from the checkout while others are still in place. " if partial else "")
                  + ("The stash holds" if oids else "A stash under that label holds") + " the whole dirty checkout as "
                  "it was then, so it can include changes made outside this cycle."]
    return "\n".join(lines)


def _record_on_result(root: pathlib.Path, task_id: str, fact: Dict[str, Any]) -> None:
    """The short fact on the task's own result, where its terminal rows and reloads read their causes."""
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document

    def _mutate(current: Dict[str, Any]) -> Any:
        require_writable_task_result_schema(current)
        if not current.get("status"):  # never mint a partial row the result store would refuse
            return None
        axes = current.get("outcome_axes") if isinstance(current.get("outcome_axes"), dict) else {}
        execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
        if execution.get("evolution_cleanup") == fact:
            return None
        return stamp_task_result_schema({
            **current, "outcome_axes": {**axes, "execution": {**execution, "evolution_cleanup": fact}}})

    patch_document(root, task_id, _mutate)


def _live_row(task_id: str) -> Optional[Dict[str, Any]]:
    """The queue's own copy of a task still running or queued (the reaper's room source), else None."""
    from supervisor import queue

    meta = queue.RUNNING.get(str(task_id))
    if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
        return meta["task"]
    return next((row for row in list(queue.PENDING) if isinstance(row, dict) and str(row.get("id")) == str(task_id)), None)


def _owe_card_row(root: pathlib.Path, tx: Dict[str, Any], task_id: str, fact: Dict[str, Any]) -> None:
    """One row of the cycle's card, owed before its send (the outbox's bounded replay; a receipt retires it), one per fact,
    and it stands alone only where no card exists. Typed as a terminal receipt so a transport's progress toggle
    does not hide it, and so a page that already settled the card re-reads its durable terminal line."""
    from ouroboros.task_results import load_task_result
    from supervisor.message_bus import notification_chat_route
    from supervisor.state import load_state
    from supervisor.terminal_delivery import enqueue_terminal_delivery_outcome, lineage_chat_id

    try:
        task = load_task_result(root, task_id, strict=True) or {}  # absent: a task that names no room
    except Exception:
        task = _live_row(task_id)  # the same task's own row while it still settles; never a guessed room
        if task is None:
            log.warning("Evolution cleanup row of %s is not owed: its record is unreadable and no live row names its "
                        "room. The campaign transaction keeps what moved: %s", task_id, json.dumps(fact, sort_keys=True),
                        exc_info=True)
            return
    # The reaper's incident order (C4): its Project room, else the room the task names (0, the hidden
    # partition, is a room and never Main), else the owner's chat for a task that names none.
    chat_id = notification_chat_route(lineage_chat_id(root, task, task_id) or None, task.get("chat_id"))
    if chat_id is None:  # an absent or 0 owner chat is "not configured", never the hidden partition
        chat_id = notification_chat_route(int(load_state().get("owner_chat_id") or 0) or None)
    if chat_id is None:
        log.warning("Evolution cleanup of %s moved work but no chat can carry its row: %s", task_id,
                    json.dumps(fact, sort_keys=True))
        return
    row_id = f"evolution-cleanup:{fact['transaction_id']}"
    text = _row_text(tx, fact["line"])
    # A replay observes its own time and checkout HEAD; the same words owe the same row.
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    sent = enqueue_terminal_delivery_outcome(root, {
        "type": "send_message", "chat_id": chat_id, "task_id": task_id, "text": text, "log_text": text,
        "format": "markdown", "role": "system", "system_type": "evolution_cleanup", "ts": fact["at"] or utc_now_iso(),
        "delivery_id": f"{row_id}:{digest}",
        "progress_meta": {"card_row": "timeline", "card_row_id": row_id, "terminal_receipt": "retained_work",
                          "narration": False}})
    if sent == "unavailable":
        log.warning("Evolution cleanup row of %s was not queued live; only an owed copy that registered is replayed",
                    task_id)


def publish_cleanup(tx: Dict[str, Any], task_id: str, drive_root: Any) -> None:
    """Say what this cleanup moved; a cleanup that moved nothing says nothing. Idempotent and never raises:
    an effect of the move, it never decides a stop, and each surface survives the other's failure."""
    try:
        line = cleanup_line(tx)
        if not line:
            return
        root = pathlib.Path(drive_root)
        fact = {"transaction_id": str(tx.get("transaction_id") or task_id), "status": str(tx.get("cleanup_status") or ""),
                "line": line, "repo": str(tx.get("cleanup_repo") or ""), "at": str(tx.get("cleanup_at") or ""),
                "base_head": str(tx.get("base_head") or ""), "stash_label": str(tx.get("cleanup_stash") or ""),
                "stash_oids": list(tx.get("cleanup_stash_oids") or []),
                "stash_unconfirmed": bool(tx.get("cleanup_stash_unconfirmed")),
                "stash_partial": bool(tx.get("cleanup_stash_partial")),
                "preserved_ref": str(tx.get("cleanup_preserved_ref") or ""),
                "preserved_head": str(tx.get("cleanup_preserved_head") or ""), "head": str(tx.get("cleanup_head") or ""),
                "error": str(tx.get("cleanup_error") or "")}
        for publish in (lambda: _record_on_result(root, str(task_id), fact),
                        lambda: _owe_card_row(root, tx, str(task_id), fact)):
            try:
                publish()
            except Exception:
                log.warning("Evolution cleanup fact of %s was not fully published", task_id, exc_info=True)
    except Exception:
        log.warning("Evolution cleanup fact of %s was not published", task_id, exc_info=True)


def carry_cleanup_fact(drive_root: Any, task_id: str, task_done_event: Dict[str, Any]) -> None:
    """The terminal event the dispatcher publishes next says what the durable result now holds.
    Its caller is a ``finally``: nothing here may replace that terminal's own return."""
    try:
        from ouroboros.task_results import load_task_result

        axes = (load_task_result(pathlib.Path(drive_root), task_id) or {}).get("outcome_axes") or {}
        fact = (axes.get("execution") or {}).get("evolution_cleanup")
        if isinstance(fact, dict):
            event_axes = task_done_event.get("outcome_axes") or {}
            task_done_event["outcome_axes"] = {
                **event_axes, "execution": {**(event_axes.get("execution") or {}), "evolution_cleanup": fact}}
    except Exception:
        log.debug("cleanup fact did not reach the terminal event of %s", task_id, exc_info=True)
