"""The direct chat lane and its resume after a restart.

Each chat turn owns a fresh native agent and a registered execution. A turn
is admitted (``_admit_chat_task``: the seed, the census registration, the
actor, the start receipt) and then executed (``_execute_chat_task``); the
owner's message runs both in one call (``handle_chat_direct``), while a
self-initiated wake-up (``handle_wake_direct``) admits synchronously and
answers with a typed receipt before the body runs on its own thread.
Turns are refused while the repo-writer gate is closed
for a DESTRUCTIVE update window (apply/replace prologue, materialization,
rollback), so a managed update never races a turn that could touch the checkout
mid-reset. While the ONE authorized assisted resolver holds the repository
(``assisted_resolution`` / ``committing_assisted``) the lanes stay open:
conversation admission is not repo-writing permission — the registry's
managed-update guard refuses every repo-mutating tool to any task but that
resolver — and an owner line reaches the resolver through the ordinary
``steer_task`` mailbox path (#283). The server's own owner-control path is
imported BEFORE conflict markers land in the live tree (``preload_owner_control_path``).
"""

from __future__ import annotations

import logging
import json
import pathlib
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union
from supervisor.state import append_jsonl
from ouroboros.utils import utc_now_iso




log = logging.getLogger(__name__)


from supervisor.log_addressing import TurnEventQueue as _TurnEventQueue


def _pool():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import workers

    return workers


# Update phases during which the repository is held by the ONE authorized
# assisted resolver and nothing else moves the tree: conversation stays open.
_CONVERSATION_ADMITTED_PHASES = frozenset({"assisted_resolution", "committing_assisted"})


def conversation_admitted_during_update(gate_reason: str) -> bool:
    """Whether a chat turn may run while ``gate_reason`` closes the repo-writer gate.

    True only while the durable update transaction is VALID and held by the
    authorized assisted resolver — phase ``assisted_resolution`` or
    ``committing_assisted`` — and the process-local latch is either absent (a
    post-restart resume: only the durable marker closes the gate) or the assisted
    latch of that same transaction. Every other closure is a destructive window
    (the apply/replace/rollback prologue, materialization, a corrupt or future
    marker) and keeps refusing. Conversation admission is not repo-writing
    permission: the registry guard still refuses repo tools to a non-resolver.
    """
    from supervisor.update_merge import assisted_writer_gate_reason, read_update_tx_strict

    reason = str(gate_reason or "")
    if not reason:
        return True
    try:
        status, tx = read_update_tx_strict()
    except Exception:
        return False
    if status != "valid" or str(tx.get("phase") or "") not in _CONVERSATION_ADMITTED_PHASES:
        return False
    if reason.startswith("managed_update_tx:"):
        return True
    return reason == assisted_writer_gate_reason(tx)


def wake_gate_open() -> bool:
    """The owner-conversation gate for a consciousness wake-up, WITHOUT the owner's lock
    notice: a refused wake is the alarm's typed ``repo_writer_gate_closed``, retried quietly,
    never a "🔒" line in Main at every attempt."""
    reason = _pool().repo_writer_admission_closed()
    return not reason or conversation_admitted_during_update(reason)


def owner_conversation_admitted(chat_id: int) -> bool:
    """Admit one owner chat turn: open gate, or a resolver-held update.

    A refused turn gets the pool's existing lock notice (``_repo_writer_turn_allowed``).
    """
    reason = _pool().repo_writer_admission_closed()
    if not reason or conversation_admitted_during_update(reason):
        return True
    return bool(_pool()._repo_writer_turn_allowed(chat_id))


def preload_owner_control_path() -> list[str]:
    """Import the server's owner-control path while the live tree is still clean.

    The assisted merge is about to write conflict markers into the checkout this
    process imports from. A module already in ``sys.modules`` keeps working; a
    function-local import that first runs AFTER the markers land raises
    SyntaxError on a conflicted file — the late receipt-wait import that broke
    the update conversation (#283). Called right after the resolver readiness
    proof and before a boot re-materialization. Best effort: a failure is logged
    and returned, never a reason to refuse the update (it would degrade the
    conversation, not the update). First-party packages are discovered so newly
    added function-local imports do not escape a manually maintained list.
    The tool catalog still loads through the chat agent's registry, preserving
    its admission rules and module-failure diagnostics.
    """
    import importlib
    import pkgutil

    failed: list[str] = []
    pending = ["ouroboros", "supervisor"]
    while pending:
        name = pending.pop()
        try:
            module = importlib.import_module(name)
            # Discover from the imported package's own path, including nested
            # packages. The packaged server runs embedded Python against the
            # materialized repo; this never depends on the process cwd or a
            # hand-maintained frozen module list.
            if hasattr(module, "__path__"):
                pending.extend(info.name for info in pkgutil.iter_modules(
                    module.__path__, module.__name__ + ".",
                ))
        except Exception:
            log.warning("owner control path preload: %s failed", name, exc_info=True)
            failed.append(name)
    try:
        from ouroboros.tools.registry import ToolRegistry

        ToolRegistry(pathlib.Path(_pool().REPO_DIR), pathlib.Path(_pool().DRIVE_ROOT))
    except Exception:
        log.warning("owner control path preload: tool catalog failed", exc_info=True)
        failed.append("ouroboros.tools.*")
    return failed


def handle_chat_direct(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    if not owner_conversation_admitted(chat_id):
        return
    _handle_chat_direct_locked(
        chat_id, text, image_data,
        task_constraint=task_constraint, task_metadata=task_metadata,
    )


def _handle_chat_direct_locked(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    from supervisor.state import budget_remaining, load_state
    failure_meta = _host_operation_failure(task_metadata)
    try:
        remaining = budget_remaining(load_state(), strict=True)
    except Exception:
        _pool().send_with_budget(chat_id, "⚠️ Cost accounting is unavailable. Task was not dispatched; retry after ledger recovery.", **failure_meta, role="system", system_type="task_admission_notice")
        return
    if remaining <= 0:
        try:
            _pool().send_with_budget(chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.", **failure_meta, role="system", system_type="task_admission_notice")
        except Exception:
            pass
        return

    _run_chat_task(
        None, chat_id, text, image_data,
        task_constraint=task_constraint, task_metadata=task_metadata,
    )


def _host_operation_failure(metadata: Optional[dict]) -> dict:
    """Optional terminal correlation for the host's preaccepted skill messages."""
    from ouroboros.task_finalization import host_operation_reply_kwargs

    if (metadata or {}).get("_host_operation"):
        return host_operation_reply_kwargs((metadata or {}).get("origin_message_ref"), "failed")
    return {}


def _broadcast_task_named(msg: dict) -> None:
    """Bridge broadcast callback for admission naming (kept tiny + fail-soft)."""
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().broadcast(msg)
    except Exception:
        log.debug("task_named broadcast failed", exc_info=True)


def _run_chat_task(
    agent: Any,
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    """Build the direct-chat task and run it on the given agent, draining events.

    Main/Project turns use the full native task/result/delivery lifecycle:
    admission (``_admit_chat_task``) then execution (``_execute_chat_task``)."""
    admitted = _admit_chat_task(
        agent, chat_id, text, image_data,
        task_constraint=task_constraint, task_metadata=task_metadata,
    )
    if admitted is not None:
        _execute_chat_task(admitted)


def _admit_chat_task(
    agent: Any,
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Build and REGISTER one direct turn; ``None`` when it was refused.

    Everything that happens before the model runs: the seed task, the census
    registration under the repo-writer gate (one transaction with the update
    owner), the actor, the origin/attachment/project layering, the task
    contract and the authoritative start receipt. The returned bundle is a
    registered execution that ``_execute_chat_task`` MUST run (it owns the
    unregister); a refusal or an admission failure has already unregistered
    itself and, for a failure, reported it to the chat.
    """
    client_msg_id = ""
    if task_metadata:
        _cmid_ref = task_metadata.get("origin_message_ref")
        if isinstance(_cmid_ref, dict):
            client_msg_id = str(_cmid_ref.get("client_message_id") or "")
        if not client_msg_id:
            client_msg_id = str(task_metadata.get("client_message_id") or "")
    kind = "direct_chat"
    task: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "type": "task",
        "chat_id": chat_id,
        "text": text,
        "_is_direct_chat": True,
    }
    from supervisor.active_activity import get_direct_activity_registry

    registry = get_direct_activity_registry()
    # Close/check/register is one short transaction with the update owner.
    # The registered execution includes agent construction, attachment staging,
    # the whole native lifecycle and event delivery, not just its LLM rounds.
    from ouroboros.consciousness_authority import is_consciousness_origin

    quiet = is_consciousness_origin(task_metadata)  # a wake: the alarm reports the refusal, not the chat
    with _pool()._repo_writer_gate_lock:
        if not (wake_gate_open() if quiet else owner_conversation_admitted(chat_id)):
            return None
        activity = registry.register(
            task["id"], chat_id,
            client_message_id=client_msg_id,
            project_id=str((task_metadata or {}).get("project_id") or ""),
            kind=kind, origin_message_ref=(task_metadata or {}).get("origin_message_ref"),
            actor=agent,
        )
    admitted: Dict[str, Any] = {
        "task": task, "agent": agent, "activity": activity, "registry": registry,
        "chat_id": chat_id, "client_msg_id": client_msg_id, "kind": kind,
        "task_metadata": task_metadata,
    }
    try:
        if agent is None:
            agent = _pool()._get_chat_agent()
            activity.actor = agent
            admitted["agent"] = agent
        from ouroboros.consciousness_authority import apply_consciousness_authority
        from ouroboros.contracts.task_contract import attach_task_contract

        if task_constraint:
            task["task_constraint"] = dict(task_constraint)
        if task_metadata:
            task["metadata"] = dict(task_metadata)
            if task_metadata.get("late_answer") is not None:
                # A LATE quiz answer: the owner's row holds only their words; the
                # model reads the card they answered, rebuilt from the stored
                # block (disclosed when unreadable) -- the one shared builder.
                from ouroboros.owner_quiz import late_answer_model_text

                task["text"] = late_answer_model_text(
                    _pool().DRIVE_ROOT, task_metadata.get("late_answer"), str(text or ""),
                )
            # The ingress-captured origin identity rides on the TASK RECORD so a
            # later post-hoc "Turn into project" reads it from the persisted
            # result instead of re-deriving identity from content.
            _origin_ref = task_metadata.get("origin_message_ref")
            if isinstance(_origin_ref, dict) and _origin_ref:
                task["origin_message_ref"] = dict(_origin_ref)
                _origin_text = task_metadata.get("origin_message_text")
                if isinstance(_origin_text, str) and _origin_text:
                    task["origin_message_text"] = _origin_text
            # Project-thread conversations scope the direct lane to the
            # project's memory (knowledge/journal/workpad sections).
            pid = str(task_metadata.get("project_id") or "").strip()
            if pid:
                task["project_id"] = pid
        if image_data:
            # image_data is (base64, mime) or (base64, mime, caption). The caption
            # still seeds task['text'] (and the legacy inline image path below) so a
            # caption-only message keeps working even when nothing stages.
            task["image_base64"] = image_data[0]
            task["image_mime"] = image_data[1]
            if len(image_data) > 2 and image_data[2]:
                task["image_caption"] = image_data[2]
                if not text:
                    task["text"] = image_data[2]
        # v6.52.0 (P1, full desktop unify): route the WHOLE desktop attachment set
        # (any type) through the shared staging substrate so the agent gets EVERY
        # attachment — images natively via attachment_images + non-images via the
        # read_file(root='artifact_store', path='attachments/...') manifest — exactly
        # like the CLI/API/GAIA path. The uploads are resolved from data/uploads/ in
        # ws._chat_attachment_uploads and carried as task['metadata'] (like force_plan).
        # On a non-empty manifest we DROP the legacy inline image_base64 so the same
        # image is not double-injected; on absent/empty uploads (older clients, the
        # single-image base64 seam) the legacy inline path above stays untouched.
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        uploads = meta.get("chat_attachment_uploads")
        if uploads:
            from ouroboros.artifacts import (
                attachment_manifest_all_rejected,
                attachment_manifest_has_rejections,
                stage_task_attachments,
            )
            from ouroboros.gateway.tasks import _render_attachment_lines

            manifest = stage_task_attachments(_pool().DRIVE_ROOT, str(task["id"]), uploads)
            rendered = _render_attachment_lines(manifest)
            # Partial staging is the default (В25c, capinv-447); a FULLY-rejected
            # set stays atomic — the task would start with none of its material.
            if attachment_manifest_all_rejected(manifest):
                from ouroboros.artifacts import remove_staged_attachments

                remove_staged_attachments(manifest)
                _pool().send_with_budget(
                    chat_id,
                    f"⚠️ Task not started: every attachment was rejected.\n{rendered}",
                    **_host_operation_failure(task_metadata),
                    role="system", system_type="attachment_notice")
                registry.unregister(task["id"])
                return None
            from ouroboros.artifacts import attachment_manifest_projection
            authority = attachment_manifest_projection(_pool().DRIVE_ROOT, str(task["id"]), manifest)
            rendered = _render_attachment_lines(authority)
            if attachment_manifest_has_rejections(manifest):
                _pool().send_with_budget(
                    chat_id,
                    "⚠️ Some declared attachments could not be staged; the task "
                    f"starts with the rest.\n{rendered}",
                    role="system", system_type="attachment_notice")
            if manifest:
                manifest = [dict(row) for row in manifest]
                task["drive_root"] = str(_pool().DRIVE_ROOT)
                task.update(authority)
                task["attachments"] = authority["attachment_manifest"]
                task["attachment_images"] = [
                    m for m in manifest
                    if str(m.get("status") or "staged") == "staged" and m.get("is_image")
                ]
                if rendered:
                    task["text"] = f"{task.get('text') or ''}\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]"
                task.pop("image_base64", None)
                task.pop("image_mime", None)
        # A rejected initial UI task must leave no partial project assignment.
        # Bind only after all declared attachments have passed admission.
        pid = str(task.get("project_id") or "").strip()
        if pid:
            try:
                from ouroboros.projects_registry import bind_task_to_project

                bind_task_to_project(
                    _pool().DRIVE_ROOT, task["id"], pid, chat_id,
                    origin=_pool()._origin_from_mapping(
                        task_metadata or {}, absent="mid_task_no_origin",
                    ),
                )
            except Exception as exc:
                _pool()._report_binding_failure(task["id"], pid, exc, path="direct_project_turn")
        if not task["text"]:
            task["text"] = "(image attached)" if image_data else ""
        # A Main turn is named lazily: the turn queue below fires the namer on
        # the first non-addressing tool call (owner decision Q7=A, 16.09), so a
        # greeting costs no naming call and a working turn gets a title as its
        # block becomes the task card. A Project-room turn is named by its room.
        # Managed promotes keep their admission names
        # (worker_promotion._admitted_suggested_name).
        # A consciousness wake-up derives its level's disabled_tools and mode cap
        # here, before the contract reads them (consciousness_authority).
        apply_consciousness_authority(task)
        attach_task_contract(task)

        # Announce the authoritative start immediately (owner decision 2A):
        # the client's `Sending...` retires on this receipt (once the census
        # read it triggers has answered), not on a socket echo; the census row
        # carries the same activity<->client_message_id link, and a turn the
        # census never lists is still settled by that read.
        try:
            from supervisor.message_bus import get_bridge

            get_bridge().send_chat_action(
                int(chat_id or 0),
                "typing",
                activity_id=str(task["id"]),
                client_message_id=client_msg_id,
                phase="thinking",
                kind=kind,
            )
        except Exception:
            log.debug("Direct-turn start typing announce failed", exc_info=True)
    except Exception as e:
        _report_direct_chat_error(admitted, e)
        registry.unregister(task["id"])
        return None
    return admitted


def _execute_chat_task(admitted: Dict[str, Any]) -> bool:
    """Run a registered direct turn to its end, draining its events.

    Returns True when the actor's run and the event hand-off completed, False
    when the runner failed (the failure has been reported to the chat). The
    registry entry is released here, whichever way the turn ends.
    """
    task, agent, chat_id = admitted["task"], admitted["agent"], admitted["chat_id"]
    registry = admitted["registry"]
    ok = False
    try:
        # The turn's live emits (loop_llm_call and friends publish
        # straight to the agent's event queue DURING handle_task) and its
        # returned events can be consumed after this registry entry is gone:
        # stamp the authoritative chat identity before handing them off.
        on_first_work = None
        if not task.get("project_id"):
            from ouroboros.project_naming import spawn_turn_namer

            on_first_work = lambda: spawn_turn_namer(  # noqa: E731
                _pool().DRIVE_ROOT, str(task["id"]), task["text"], broadcast=_broadcast_task_named,
            )
        turn_queue = _TurnEventQueue(
            _pool().get_event_q(), task["id"], chat_id,
            initiator=str((admitted.get("task_metadata") or {}).get("initiator") or ""),
            on_first_work=on_first_work,
        )
        prev_queue = getattr(agent, "_event_queue", None)
        agent._event_queue = turn_queue
        try:
            events = agent.handle_task(task)
        finally:
            agent._event_queue = prev_queue
        # An exact budget pause of THIS turn is parked here, in-process and
        # synchronously, while the registry entry below still owns the id (#1196).
        # The turn's ``budget_pause`` event carries its own task record (a direct
        # turn was never in RUNNING). Handing that event to the supervisor loop
        # and unregistering the actor races: between the unregister and the park
        # the SAME id is nowhere — not live, not queued — and a restart in that
        # window fences a saved pause. The direct lane runs inside the supervisor
        # process, so it parks the record itself through the ONE park owner
        # (``events_budget.install_exact_budget_pause``) against the same queue
        # state, under the queue lock, and drops the event from the hand-off. A
        # park that fails leaves the event on the ordinary path (typed, logged),
        # never a silently lost pause. Either way the turn's LOCAL dispatch fence
        # is released: the actor has unwound and the durable row (parked here or
        # by the supervisor loop) owns the hold, so a fence left closed in this
        # process would refuse the resumed turn's sends under the same id. Every
        # other event passes through unchanged.
        remaining: list = []
        for event in list(events or []):
            checkpoint = ((event.get("resource_limit") or {}).get("checkpoint")
                          if isinstance(event, dict) and isinstance(event.get("resource_limit"), dict) else None)
            if not (isinstance(event, dict) and str(event.get("type") or "") == "budget_pause"
                    and event.get("_is_direct_chat") and isinstance(checkpoint, dict)):
                remaining.append(event)
                continue
            task_id = str(event.get("task_id") or task.get("id") or "")
            try:
                from types import SimpleNamespace

                from supervisor import queue as queue_mod
                from supervisor.events_budget import install_exact_budget_pause
                from supervisor.message_bus import get_bridge

                pool = _pool()
                shim = SimpleNamespace(
                    RUNNING=pool.RUNNING, PENDING=pool.PENDING, WORKERS=pool.WORKERS, DRIVE_ROOT=pool.DRIVE_ROOT,
                    sort_pending=queue_mod.sort_pending, persist_queue_snapshot=queue_mod.persist_queue_snapshot,
                    bridge=get_bridge(),
                )
                install_exact_budget_pause(shim, task_id, checkpoint, evt=event, source="direct_turn_inline_park")
                append_jsonl(
                    pool.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {"ts": utc_now_iso(), "type": "direct_turn_budget_pause_parked_inline",
                     "task_id": task_id, "chat_id": event.get("chat_id"),
                     "pause_id": str(checkpoint.get("pause_id") or "")},
                )
            except Exception:
                log.error("Direct turn %s could not be parked inline; its pause event takes the ordinary path",
                          task_id, exc_info=True)
                remaining.append(event)
            finally:
                from ouroboros.budget_pause import end_dispatch_fence

                end_dispatch_fence(task_id)  # quiescent actor unwound; the durable row owns the dispatch hold
        for e in remaining:
            _pool().get_event_q().put(turn_queue.stamp(e))
        ok = True
    except Exception as e:
        _report_direct_chat_error(admitted, e)
    finally:
        registry.unregister(task["id"])
    return ok


def _report_direct_chat_error(admitted: Dict[str, Any], e: BaseException) -> None:
    """Record a direct-turn failure and conclude the turn in the chat."""
    import traceback
    task, chat_id, kind = admitted["task"], admitted["chat_id"], admitted["kind"]
    client_msg_id, task_metadata = admitted["client_msg_id"], admitted.get("task_metadata")
    err_msg = f"⚠️ Error: {type(e).__name__}: {e}"
    append_jsonl(
        _pool().DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "direct_chat_error",
            "task_id": str(task.get("id") or ""),
            "chat_id": int(chat_id or 0),
            "error": repr(e),
            "traceback": str(traceback.format_exc())[:2000],
        },
    )
    try:
        # Key the error final with the turn's activity id so the client
        # concludes exactly this turn (active set, 4A) instead of leaving
        # its `Sending.../Thinking...` state to an unkeyed sweep. If the
        # failure happened before the start announce was broadcast, announce
        # it first: the receipt's census read settles the linked `Sending...`
        # and the keyed final right after concludes the turn's census row.
        failed_task_id = str(task.get("id") or "") if isinstance(task, dict) else ""
        if failed_task_id and client_msg_id:
            try:
                from supervisor.message_bus import get_bridge

                get_bridge().send_chat_action(
                    int(chat_id or 0),
                    "typing",
                    activity_id=failed_task_id,
                    client_message_id=client_msg_id,
                    phase="thinking",
                    kind=kind,
                )
            except Exception:
                log.debug("Failed-turn typing announce failed", exc_info=True)
        failure_meta = _host_operation_failure(task_metadata)
        progress_meta = {"task_terminal_status": "failed"}
        if failure_meta:
            progress_meta["origin_message_ref"] = failure_meta["progress_meta"]["origin_message_ref"]
        initiator = str((task_metadata or {}).get("initiator") or "")
        if initiator:
            progress_meta["initiator"] = initiator
        _pool().send_with_budget(
            chat_id,
            err_msg,
            task_id=failed_task_id,
            progress_meta=progress_meta,
            role="system", system_type="task_error")
    except Exception:
        log.debug("Suppressed exception", exc_info=True)


def handle_wake_direct(
    chat_id: int,
    text: str,
    task_metadata: Optional[dict],
    on_finished: Optional[Callable[[str, bool], None]] = None,
) -> Dict[str, Any]:
    """Start a self-initiated Main turn (a consciousness wake-up) as an
    ordinary direct turn, and answer with a typed receipt.

    The same gates as ``handle_chat_direct`` decide admission, but a refused
    wake gets ``{"admitted": False, "task_id": "", "reason": <typed>}``
    synchronously instead of a chat notice. An admitted wake is REGISTERED
    before this returns (the census lists it, Stop reaches it, the liveness
    read sees it) and its body runs on a daemon thread; the receipt carries
    the registered ``task_id``. ``on_finished(task_id, ok)`` fires once the
    turn has ended, ``ok=False`` when the runner failed, so the alarm clock
    can back off after a failure too. The wake's ``task_metadata`` (its
    origin label, ledger category, reason, autonomy level, model role) rides
    verbatim on ``task["metadata"]``; nothing here pauses or resumes the
    legacy background loop.
    """
    if not wake_gate_open():
        return {"admitted": False, "task_id": "", "reason": "repo_writer_gate_closed"}
    from supervisor.state import budget_remaining, load_state

    try:
        remaining = budget_remaining(load_state(), strict=True)
    except Exception:
        return {"admitted": False, "task_id": "", "reason": "cost_accounting_unavailable"}
    if remaining <= 0:
        return {"admitted": False, "task_id": "", "reason": "budget_exhausted"}
    admitted = _admit_chat_task(
        None, int(chat_id), str(text or ""), None,
        task_constraint=None, task_metadata=dict(task_metadata or {}),
    )
    if admitted is None:
        # The gate can close between the check above and the registration — a silent
        # refusal, nothing in the chat; every other None the lane already reported.
        reason = "repo_writer_gate_closed" if not wake_gate_open() else "admission_failed"
        return {"admitted": False, "task_id": "", "reason": reason}
    task_id = str(admitted["task"]["id"])

    def _run() -> None:
        ok = False
        try:
            ok = _execute_chat_task(admitted)
        finally:
            if on_finished is not None:
                try:
                    on_finished(task_id, ok)
                except Exception:
                    log.debug("wake on_finished callback failed", exc_info=True)

    import threading

    try:
        threading.Thread(target=_run, name=f"wake-turn-{task_id}", daemon=True).start()
    except Exception:
        # A registered turn nobody runs would read as a live owner turn forever.
        log.warning("wake turn %s could not start its thread", task_id, exc_info=True)
        admitted["registry"].unregister(task_id)
        return {"admitted": False, "task_id": "", "reason": "admission_failed"}
    return {"admitted": True, "task_id": task_id, "reason": ""}


def auto_resume_after_restart() -> None:
    """Auto-resume after a recent restart when scratchpad still has work."""
    try:
        owner_restart_flag = _pool().DRIVE_ROOT / "state" / "owner_restart_no_resume.flag"
        if owner_restart_flag.exists():
            owner_restart_flag.unlink(missing_ok=True)
            panic_compat_flag = _pool().DRIVE_ROOT / "state" / "panic_stop.flag"
            try:
                if panic_compat_flag.read_text(encoding="utf-8").strip() == "owner_restart_no_resume":
                    panic_compat_flag.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except Exception:
                log.debug("Failed to consume owner restart compatibility flag", exc_info=True)
            log.info("Owner restart flag detected — skipping auto-resume.")
            return

        # Panic/owner-restart flags suppress auto-resume and are consumed.
        panic_flag = _pool().DRIVE_ROOT / "state" / "panic_stop.flag"
        if panic_flag.exists():
            panic_flag.unlink(missing_ok=True)
            log.info("Panic flag detected — skipping auto-resume.")
            return

        st = _pool().load_state()
        chat_id = st.get("owner_chat_id")
        if not chat_id:
            return

        restart_verify_path = _pool().DRIVE_ROOT / "state" / "pending_restart_verify.json"
        recent_restart = False
        if restart_verify_path.exists():
            recent_restart = True
        else:
            sup_log = _pool().DRIVE_ROOT / "logs" / "supervisor.jsonl"
            if sup_log.exists():
                try:
                    lines = sup_log.read_text(encoding="utf-8").strip().split("\n")
                    for line in reversed(lines[-20:]):
                        if not line.strip():
                            continue
                        evt = json.loads(line)
                        if evt.get("type") in ("launcher_start", "restart"):
                            recent_restart = True
                            break
                except Exception:
                    log.debug("Suppressed exception", exc_info=True)

        if not recent_restart:
            return

        scratchpad_path = _pool().DRIVE_ROOT / "memory" / "scratchpad.md"
        if not scratchpad_path.exists():
            return

        scratchpad = scratchpad_path.read_text(encoding="utf-8")
        stripped = scratchpad.strip()
        if not stripped or stripped == "# Scratchpad" or "(empty" in stripped.lower():
            content_lines = [
                ln.strip() for ln in stripped.splitlines()
                if ln.strip() and not ln.strip().startswith("#") and ln.strip() != "- (empty)"
            ]
            content_lines = [ln for ln in content_lines if not ln.startswith("UpdatedAt:")]
            if not content_lines:
                return

        time.sleep(2)  # Let everything initialize
        if not _pool().chat_turn_liveness():
            import threading
            threading.Thread(
                target=handle_chat_direct,
                args=(int(chat_id),
                      "[auto-resume after restart] Continue your work. Read scratchpad and identity — they contain context of what you were doing.",
                      None),
                daemon=True,
            ).start()
            append_jsonl(
                _pool().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "auto_resume_triggered",
                },
            )
    except Exception as e:
        append_jsonl(_pool().DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(),
            "type": "auto_resume_error",
            "error": repr(e),
        })


DIRECT_TURN_STOP_GONE = "gone"        # the turn had already ended before the stop could be armed
DIRECT_TURN_STOP_ENDED = "ended"      # armed, and the turn reached its boundary within the wait
DIRECT_TURN_STOP_LIVE = "live"        # armed, still inside a step (the sweep retries custody)


def stop_direct_chat_turn(task_id: str, turn: Dict[str, Any], *, deliver: bool = True) -> str:
    """Stop the in-process direct-chat turn COOPERATIVELY; a typed outcome.

    There is no worker process to kill: the turn owns a native actor
    inside the supervisor. The lane writes the typed ``finalize_now``
    control (``REASON_OWNER_STOPPED_DIRECT_TURN``) to the canonical drive's
    owner mailbox — the one the turn's loop drains at every round boundary,
    where it ends the turn with ZERO further model calls — then waits the
    short config-owned bound for the turn to reach that boundary (the same
    custody pass runs on the supervisor sweep, so the wait stays short).

    Two things the write must never do: arm a turn that is already gone (a
    turn that ended between custody's ownership read and this write would get
    a false owner toast over an answer that already landed, and an orphaned
    control in a mailbox the settled-cleanup may already have pruned) — so
    liveness is re-read immediately before the write and ``GONE`` names that
    lane; and re-arm on a retry: the control is written ONCE per turn (the
    turn's stamp is the latch, as the RUNNING row is for a pooled task) and a
    custody pass that finds the stamp already there answers WITHOUT waiting
    (the supervisor tick must not spend the bound on every pass).
    ``deliver=False`` (a cascade sweep, which speaks for the tree once)
    suppresses the owner toast.
    """
    from supervisor import queue as q
    from supervisor import workers
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
    from supervisor.task_reaper import request_finalization_grace
    from ouroboros.config import get_direct_turn_stop_wait_sec

    if turn.get("stop_control_msg_id"):
        return DIRECT_TURN_STOP_LIVE if workers.direct_chat_turn(task_id) is not None else DIRECT_TURN_STOP_ENDED

    def _write_control(live_turn: Dict[str, Any]) -> str:
        # The canonical drive: a direct turn runs on the main data root, and
        # its loop drains that root's owner mailbox (the same root custody
        # settles the intent on).
        return request_finalization_grace(
            pathlib.Path(q.DRIVE_ROOT), task_id, REASON_OWNER_STOPPED_DIRECT_TURN,
            chat_id=int(live_turn.get("chat_id") or 0), stamp=int(time.time()),
            toast_text=(
                f"⏹ The owner stopped chat turn {task_id}; it ends at its next "
                "step without further work."
            ) if deliver else "",
            quiet=not deliver,
        )

    # Atomic against the turn's own completion (its admission lock): a turn
    # that ended first gets no control and no toast.
    if workers.arm_direct_chat_turn(task_id, _write_control) is None:
        return DIRECT_TURN_STOP_GONE
    deadline = time.monotonic() + float(get_direct_turn_stop_wait_sec())
    while workers.direct_chat_turn(task_id) is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    return DIRECT_TURN_STOP_LIVE if workers.direct_chat_turn(task_id) is not None else DIRECT_TURN_STOP_ENDED
