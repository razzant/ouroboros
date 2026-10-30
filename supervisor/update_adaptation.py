"""An Update conflict is adapted in its candidate's own workspace by one reviewed resolver.

A conflict at any step of a preparation (``update_prepare``: the merge C, the staged layer I', the working
layer W') commits the mechanical result WITH its markers (M0) on the candidate branch and hands that
workspace to one resolver task, admitted through the queue's host-producer admission and bound to that
checkout in every state (``resolver_binding``), never to the serving tree. Its settlement accepts exactly
one reviewed commit on M0 that leaves no marker and carries the managed adaptation's qualification, a
complete green full test run of exactly that commit on record (``_adaptation_unqualified``); it records the
step's result and the preparation goes on (``update_prepare._advance``). Anything else is the owner's
decision. Records, locks and steps stay ``update_prepare``'s; this module adds the hand-off and settlement.
"""

from __future__ import annotations

import logging
import pathlib
import threading
import uuid
from typing import Any, Dict, List, Optional

from supervisor import update_prepare as up

log = logging.getLogger(__name__)

_SIDES = {"merge": ("the official update", "this installation's own commits"),
          "index": ("the merged update", "the owner's staged changes"),
          "files": ("the merged update with the staged changes", "the owner's unstaged and untracked changes")}


def _objective(row: Dict[str, Any], step: str, conflicts: List[str]) -> str:
    theirs, ours = _SIDES[step]
    return (
        f"Adapt this installation's local work to the official Ouroboros update "
        f"{str(row['target']['target_sha'])[:12]}. Your working copy is the update's private workspace; the running "
        f"app is not affected. It holds the mechanical result, with conflict markers where {theirs} and {ours} "
        f"changed the same lines: {', '.join(conflicts[:20])}. Resolve every conflict so the update keeps its meaning "
        "and the local function keeps working, and remove every marker (<<<<<<<, =======, >>>>>>>). Then call "
        "commit_reviewed once (full_tests=true with a test_reason) with a message naming what you kept from each "
        "side. Do not move branches or run git checkout/reset/merge/rebase, and do not request a restart: the "
        "preparation continues after your commit. If the local function cannot be kept without changing what the "
        "update means or removing that function, do not commit: end with one concrete question for the owner that "
        "names the options."
    )


def resolver_row(metadata: Any) -> Optional[Dict[str, Any]]:
    """The candidate whose adaptation this task's settlement continues, or ``None``: still ``adapting`` with
    exactly this resolver. Binding the task's workspace is ``resolver_binding``'s, whatever the state."""
    row = resolver_binding(metadata)
    return row if row and row.get("state") == "adapting" else None


def resolver_binding(metadata: Any) -> Optional[Dict[str, Any]]:
    """The candidate an adaptation task works in, in any state (``body_candidate.restore`` binds it).

    A resolver whose candidate was discarded or superseded keeps working in its own
    checkout, which the GC keeps until its exit is proven, never in the serving one.
    """
    candidate_id = metadata.get("update_candidate_resolver") if isinstance(metadata, dict) else ""
    row = up.find(str(candidate_id)) if candidate_id else None
    resolver = (row or {}).get("resolver") or {}
    return row if row and resolver.get("task_id") and resolver.get("task_id") == str(
        metadata.get("root_task_id") or "") else None


def _admit_resolver(task: Dict[str, Any]) -> str:
    """Hand the resolver to the queue's host-producer admission; ``""`` once admitted, else the typed refusal."""
    from supervisor import workers
    from supervisor.task_admission import enqueue_with_admission_receipt

    try:
        if not workers.ensure_worker_pool_started(allow_disabled_restart=True):
            return "worker_pool_unavailable"
        admitted = enqueue_with_admission_receipt(task, receipt_required=False, front=True, durable=True)
    except Exception as exc:
        log.warning("the update adaptation task could not be queued", exc_info=True)
        return f"{type(exc).__name__}: {exc}"
    if not isinstance(admitted, dict):
        return "admission_unknown"
    blocked = admitted.get("_admission_blocked")
    if not blocked:
        return ""
    detail = str(admitted.get("_admission_detail") or "")
    return str(admitted.get("_worker_pool_disabled_reason") or blocked) + (f": {detail}" if detail else "")


def _adapt(row: Dict[str, Any], step: str, parent: str, tree: str, conflicts: List[str], **fields: Any
           ) -> Dict[str, Any]:
    """Commit the mechanical result WITH its markers (M0) on the candidate branch and start one resolver.

    The resolver authors in the workspace like any own-body candidate: its reviewed
    commit's diff is M0 -> resolution, the conflict work alone. Without model budget
    the candidate waits ``paused_budget``; a resolver the queue does not admit leaves it
    ``needs_adaptation`` with the refusal (prepare again retries). The candidate is
    ``adapting`` only once the resolver was admitted; the serving checkout is never
    involved, and the resolver's follow-ups name this checkout, never the serving one.
    """
    from supervisor.state import budget_remaining, load_state

    workspace = pathlib.Path(row["path"])
    m0 = up._out(workspace, "commit-tree", tree, "-p", parent, "-m", f"ouroboros update: mechanical {step} with conflicts")
    up._run(workspace, "reset", "--hard", "--quiet", m0)
    up._event("update_candidate_needs_adaptation", candidate_id=row["candidate_id"], step=step, paths=conflicts[:20])
    try:
        remaining = budget_remaining(load_state() or {}, strict=True)
    except Exception:
        remaining = 0
    resolver = {"step": step, "m0": m0, "parent": parent, "conflicts": conflicts, "task_id": ""}
    if remaining <= 0:
        return up._update(row, state="paused_budget", reason=f"{step}_conflicts", resolver=resolver, **fields)
    task_id = "update_adapt_" + uuid.uuid4().hex[:8]
    with up._lifecycle:
        # Recorded before admission: a worker that starts it at once binds this checkout (``resolver_binding``).
        row = up._update(row, state="needs_adaptation", reason=f"{step}_conflicts",
                         resolver={**resolver, "task_id": task_id}, **fields)
        owner_chat = int((load_state() or {}).get("owner_chat_id") or 0)
        refusal = _admit_resolver({
            "id": task_id, "type": "task", "chat_id": owner_chat, "text": _objective(row, step, conflicts),
            "metadata": {"root_task_id": task_id, "update_candidate_resolver": row["candidate_id"],
                         # The canonical resource intent: a follow-up keeps this checkout as its target
                         # and waits while it is gone, never falling back to the serving repository.
                         "resource_intent": {"kind": "explicit_resource", "root": str(workspace)}}})
        if refusal:
            return up._update(row, state="needs_adaptation", reason=f"{step}_conflicts: resolver not admitted: {refusal}",
                              resolver={**resolver, "refused": refusal})
        return up._update(row, state="adapting")


def resolution_settled(task_id: str, metadata: Any) -> None:
    """A task ended: when it names an update candidate's resolver, continue that adaptation off this thread."""
    if isinstance(metadata, dict) and metadata.get("update_candidate_resolver"):
        threading.Thread(target=_settle_resolution, args=(str(task_id), dict(metadata)), daemon=True,
                         name="update-adaptation").start()


def _settle_resolution(task_id: str, metadata: Dict[str, Any]) -> None:
    with up._running:  # after the step that admitted it: the record says adapting, or the candidate moved on
        row = resolver_row(metadata)
        if row is None or (row.get("resolver") or {}).get("task_id") != task_id:
            return
        with up._lifecycle:
            up._working.add(row["candidate_id"])
        try:
            _settle_locked(row)
        except Exception as exc:
            log.warning("update adaptation could not be continued", exc_info=True)
            up._close_failed(row, f"adaptation: {type(exc).__name__}: {exc}")
        finally:
            with up._lifecycle:
                up._working.discard(row["candidate_id"])
            _announce()


def _announce() -> None:
    """Ask every client to read the Update's status again (the frame ``gateway.update_progress`` sends), through the
    supervisor's own bus, whose broadcast the server binds to its WebSocket clients; without a bus nobody listens."""
    try:
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is not None:
            bridge.broadcast({"type": "update_progress_changed"})
    except Exception:
        log.debug("the update candidate's change was not announced", exc_info=True)


def _settle_locked(row: Dict[str, Any]) -> Dict[str, Any]:
    """The resolver's one reviewed commit on M0 resolves this step; anything else is the owner's decision."""
    workspace, resolver = pathlib.Path(row["path"]), dict(row["resolver"])
    tip = up._out(workspace, "rev-parse", "--verify", "HEAD^{commit}")
    if tip == resolver["m0"]:
        return up._update(row, state="needs_decision", reason="adaptation_question",
                          question=_resolver_answer(resolver["task_id"]))
    parents = up._out(workspace, "rev-list", "--parents", "-n", "1", tip).split()[1:]
    if parents != [resolver["m0"]] or tip not in set(row.get("reviewed_commits") or ()):
        return up._update(row, state="failed", reason="adaptation_unreviewed: the workspace moved beyond one reviewed "
                                                      "commit on the mechanical result")
    left = up._run(workspace, "grep", "-l", "-e", "^<<<<<<< ", "-e", "^>>>>>>> ", tip, check=False).stdout
    if left.strip():
        return up._update(row, state="failed", reason="adaptation_markers_left: " + left.decode("utf-8", "replace")[:300])
    unqualified = _adaptation_unqualified(workspace, tip)
    if unqualified:  # the reviewed commit and the workspace stay as they are for the owner's next step
        return up._update(row, state="failed", reason="adaptation_unqualified: " + unqualified)
    tree = up._out(workspace, "rev-parse", f"{tip}^{{tree}}")
    step, fields = resolver["step"], {}
    if step == "merge":
        target = row["target"]["target_sha"]
        merged = up._out(workspace, "commit-tree", tree, "-p", row["layers_base"]["head"], "-p", target, "-m",
                         f"Merge official Ouroboros update {target[:12]} (adapted, reviewed {tip[:12]})")
        fields["merge"] = {**(row.get("merge") or {}), "kind": "adapted", "commit": merged, "resolution": tip}
    else:
        fields["carry"] = {**(row.get("carry") or {}), f"{step if step == 'index' else 'files'}_tree": tree}
    row = up._update(row, state="preparing", worker=up._worker(), resolver={**resolver, "resolution": tip}, **fields)
    return up._advance(row)


def _adaptation_unqualified(workspace: pathlib.Path, tip: str) -> str:
    """Why the adaptation commit ``tip`` lacks the managed-adaptation qualification, or ``""``: a complete green FULL
    run of exactly its tree on record (``preflight_qualification.receipt_for_commit``, the exact-commit evidence an
    adoption reads, with no selected allowance). The resolver's post-commit gate writes it under its canonical data
    root, the installation's, where this record lives; a request for full tests is intent, never proof."""
    from ouroboros.preflight_qualification import receipt_for_commit

    data, skipped = up._registry()._data_dir(None), []
    if receipt_for_commit(data, workspace, tip, skipped=skipped).get("tier") == "full":
        return ""
    return (f"no complete green full test run of exactly {tip[:12]} is on record in {data} ("
            + ("; ".join(skipped[:3]) if skipped else "no record of that commit") + "); an adaptation commit needs "
            "the full suite (commit_reviewed with full_tests=true)")


def _resolver_answer(task_id: str) -> str:
    try:
        from ouroboros.task_results import load_task_result
        from supervisor import git_ops as g

        result = load_task_result(g.DRIVE_ROOT, task_id) or {}
        return str(result.get("result") or result.get("summary") or "")[-2000:]
    except Exception:
        return ""
