"""The direct and ephemeral chat lanes, and the resume after a restart.

A chat turn runs on the single long-lived agent under its own lock; an ephemeral
turn gets a throwaway one. Both are refused while the repo-writer gate is closed
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
import sys
import time
import uuid
from typing import Any, Dict, Optional, Tuple, Union
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


def owner_conversation_admitted(chat_id: int) -> bool:
    """Admit one owner chat turn: open gate, or a resolver-held update.

    A refused turn gets the pool's existing lock notice (``_repo_writer_turn_allowed``).
    """
    reason = _pool().repo_writer_admission_closed()
    if not reason or conversation_admitted_during_update(reason):
        return True
    return bool(_pool()._repo_writer_turn_allowed(chat_id))


# The server's owner-control path: every first-party module a Main turn may
# import for the first time while answering, steering or waiting for a receipt
# during an update. tests/test_update_owner_conversation.py pins the transitive
# closure against the function-local imports of the routing/steering chain.
OWNER_CONTROL_PATH_MODULES: tuple[str, ...] = (
    "ouroboros.server_owner_routing",
    "ouroboros.server_routing_context",
    "ouroboros.routing_wait",
    "ouroboros.owner_mailbox",
    "ouroboros.owner_hurry",
    "ouroboros.owner_quiz",
    "ouroboros.project_dialogue",
    "ouroboros.project_naming",
    "ouroboros.projects_registry",
    "ouroboros.cancel_intents",
    "ouroboros.artifacts",
    "ouroboros.client_surface",
    "ouroboros.loop_round_limits",
    "ouroboros.post_task_evolution",
    "ouroboros.promotion_source",
    "ouroboros.contracts.task_constraint",
    "ouroboros.contracts.chat_id_policy",
    "ouroboros.gateway.tasks",
    "ouroboros.gateway.task_decision",
    "ouroboros.tools.control_routing",
    "ouroboros.agent",
    "supervisor.steering",
    "supervisor.events",
    "supervisor.events_project_routing",
    "supervisor.message_bus",
    "supervisor.queue",
    "supervisor.active_activity",
    "supervisor.owner_stop",
    "supervisor.task_reaper",
    "supervisor.update_merge",
)


def preload_owner_control_path() -> list[str]:
    """Import the server's owner-control path while the live tree is still clean.

    The assisted merge is about to write conflict markers into the checkout this
    process imports from. A module already in ``sys.modules`` keeps working; a
    function-local import that first runs AFTER the markers land raises
    SyntaxError on a conflicted file — the late receipt-wait import that broke
    the update conversation (#283). Called right after the resolver readiness
    proof and before a boot re-materialization. Best effort: a failure is logged
    and returned, never a reason to refuse the update (it would degrade the
    conversation, not the update). The tool catalog is loaded exactly the way
    the chat agent's registry loads it, so every tool module is resident too.
    """
    import importlib

    failed: list[str] = []
    for name in OWNER_CONTROL_PATH_MODULES:
        try:
            importlib.import_module(name)
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
    with _pool()._chat_agent_lock:
        if not owner_conversation_admitted(chat_id):
            return
        _handle_chat_direct_locked(
            chat_id,
            text,
            image_data,
            task_constraint=task_constraint,
            task_metadata=task_metadata,
        )


def _handle_chat_direct_locked(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    from supervisor.state import budget_remaining, load_state
    try:
        remaining = budget_remaining(load_state(), strict=True)
    except Exception:
        _pool().send_with_budget(chat_id, "⚠️ Cost accounting is unavailable. Task was not dispatched; retry after ledger recovery.")
        return
    if remaining <= 0:
        try:
            _pool().send_with_budget(chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.")
        except Exception:
            pass
        return

    _run_chat_task(
        _pool()._get_chat_agent(), chat_id, text, image_data,
        task_constraint=task_constraint, task_metadata=task_metadata, ephemeral=False,
    )


def _broadcast_task_named(msg: dict) -> None:
    """Bridge broadcast callback for the proactive namer (kept tiny + fail-soft)."""
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
    *,
    ephemeral: bool = False,
) -> None:
    """Build the direct-chat task and run it on the given agent, draining events.

    ``ephemeral`` marks a SHORT-LIVED same-route turn (run on a separate agent
    instance while the shared chat agent is busy): it carries _ephemeral_turn so
    the task pipeline skips long-term memory / reflection / evolution writes."""
    task: Optional[dict] = None
    client_msg_id = ""
    if task_metadata:
        _cmid_ref = task_metadata.get("origin_message_ref")
        if isinstance(_cmid_ref, dict):
            client_msg_id = str(_cmid_ref.get("client_message_id") or "")
        if not client_msg_id:
            client_msg_id = str(task_metadata.get("client_message_id") or "")
    kind = "ephemeral_decision" if ephemeral else "direct_chat"
    task: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "type": "task",
        "chat_id": chat_id,
        "text": text,
        "_is_direct_chat": True,
    }
    try:
        from ouroboros.contracts.task_contract import attach_task_contract

        if ephemeral:
            task["_ephemeral_turn"] = True
        if task_constraint:
            task["task_constraint"] = dict(task_constraint)
        if task_metadata:
            task["metadata"] = dict(task_metadata)
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
                )
                return
            if attachment_manifest_has_rejections(manifest):
                _pool().send_with_budget(
                    chat_id,
                    "⚠️ Some declared attachments could not be staged; the task "
                    f"starts with the rest.\n{rendered}",
                )
            if manifest:
                manifest = [dict(row) for row in manifest]
                task["drive_root"] = str(_pool().DRIVE_ROOT)
                task["attachments"] = manifest
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
        if pid and not ephemeral:
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
        # Cluster B: proactively coin a project name for a fresh MAIN-CHAT direct card
        # (not an ephemeral decision turn, not an already-bound project-thread task) so
        # the card shows a human title up front and turn-into-project reuses it.
        if not ephemeral and not task.get("project_id"):
            from ouroboros.project_naming import spawn_proactive_namer

            spawn_proactive_namer(
                _pool().DRIVE_ROOT, str(task["id"]), task["text"], broadcast=_broadcast_task_named
            )
        attach_task_contract(task)

        pid = str(task.get("project_id") or "")

        from supervisor.active_activity import track_direct_activity

        with track_direct_activity(
            activity_id=str(task["id"]),
            chat_id=int(chat_id or 0),
            client_message_id=client_msg_id,
            project_id=pid,
            kind=kind,
            phase="thinking",
        ):
            # Announce the authoritative start immediately (owner decision 2A):
            # the client's `Sending...` retires on this frame, not on a socket
            # echo, and the frame carries the activity<->client_message_id link
            # so even a turn that fails before its first LLM round concludes
            # cleanly via its keyed error final.
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
            # The turn's live emits (loop_llm_call and friends publish
            # straight to the agent's event queue DURING handle_task) and its
            # returned events are both drained after the registry entry is
            # gone: route them through the turn-scoped addressing proxy.
            turn_queue = _TurnEventQueue(_pool().get_event_q(), task["id"], chat_id)
            prev_queue = getattr(agent, "_event_queue", None)
            agent._event_queue = turn_queue
            try:
                events = agent.handle_task(task)
            finally:
                agent._event_queue = prev_queue
            for e in events:
                _pool().get_event_q().put(turn_queue.stamp(e))
    except Exception as e:
        import traceback
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
            # failure happened before the start announce was broadcast, the
            # client has no activity<->client_message_id link yet, so announce
            # it first: the keyed final right after then retires both the
            # activity and its linked `Sending...` submission.
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
            _pool().send_with_budget(
                chat_id,
                err_msg,
                task_id=failed_task_id,
                progress_meta={"task_terminal_status": "failed"},
            )
        except Exception:
            log.debug("Suppressed exception", exc_info=True)


def handle_chat_ephemeral(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    """The "turn = decision" path (v6.33.0 WS10): when the shared chat agent is
    busy, a new main-chat message runs as a SHORT-LIVED turn on a SEPARATE agent
    instance — bypassing _chat_agent_lock so it never freezes/injects into the
    running turn, while keeping the SAME ROUTE (same make_agent config: model /
    mode / effort, not a cheaper lane). Ephemeral turns are serialized among
    themselves and are barred from long-term memory/reflection/evolution writes."""
    from supervisor.state import budget_remaining, load_state
    try:
        remaining = budget_remaining(load_state(), strict=True)
    except Exception:
        _pool().send_with_budget(chat_id, "⚠️ Cost accounting is unavailable. Task was not dispatched; retry after ledger recovery.")
        return
    if remaining <= 0:
        try:
            _pool().send_with_budget(chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.")
        except Exception:
            pass
        return
    if not getattr(sys, 'frozen', False):
        sys.path.insert(0, str(_pool().REPO_DIR))
    from ouroboros.agent import make_agent

    with _pool()._ephemeral_chat_lock:
        if not owner_conversation_admitted(chat_id):
            return
        agent = make_agent(repo_dir=str(_pool().REPO_DIR), drive_root=str(_pool().DRIVE_ROOT), event_queue=_pool().get_event_q())
        _run_chat_task(
            agent, chat_id, text, image_data,
            task_constraint=task_constraint, task_metadata=task_metadata, ephemeral=True,
        )


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
        agent = _pool()._get_chat_agent()
        if not agent._busy:
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

    There is no worker process to kill: the turn runs on the long-lived chat
    agent inside the supervisor. The lane writes the typed ``finalize_now``
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
