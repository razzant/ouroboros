"""Terminal delivery and sealed final ground truth for root task finalization.

Extracted seams used by ``agent_task_pipeline`` (kept at its module ceiling).
Incident e9108a09: the owner's answer was ready 34 minutes before delivery
because the buffered terminal events waited behind a hung post-task groom; the
idle reaper then killed the worker and the answer never reached chat, while
the reflection — written from the error trace — declared the delivered PDF
missing. The two seams below fix both halves: the answer leaves early over
the live queue, and synthesis receives the delivered outcome as ground truth
(a prompt input, not a validator — durable writes are never blocked on it).
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from typing import Any, Dict, List

from ouroboros.utils import truncate_review_artifact

log = logging.getLogger(__name__)

# Prompt bounds: the overflow count is disclosed instead of silently dropped.
_SEALED_MANIFEST_MAX_FILES = 200
_SEALED_FINAL_TEXT_PROMPT_CHARS = 4000


def deliver_final_message_live(
    event_queue: Any, pending_events: List[Dict[str, Any]], task_id: str,
    *, drive_root: Any = None,
) -> bool:
    """Send the buffered FINAL ``send_message`` through the live worker queue.

    The buffer can also hold proactive ``send_user_message`` events queued
    mid-task (they carry no ``task_id``), so the final answer is selected by
    the finalizing task's id — falling back to the LAST send_message — never
    the first match, which would ship a proactive text early while the answer
    stayed hostage to blocking post-task.

    Never lost, never doubled — without treating ``queue.put()`` as a delivery
    receipt: the buffered copy is KEPT (an event can still die between put and
    supervisor processing), and both copies carry the same ``delivery_id`` so
    the supervisor's send_message handler delivers exactly one of them. On a
    live-transport failure only the buffered copy exists; if the worker dies
    during post-task only the live copy ever left; in the normal case the live
    copy arrives first and the post-return drain's copy is suppressed.

    §8-A2 (ONE seam for normal/cancel/reap): when ``drive_root`` names the
    CANONICAL data root, the answer is registered as OWED in the durable outbox
    (``supervisor/terminal_delivery.py``, cross-process locked — safe from the
    worker) BEFORE it is enqueued, so a supervisor crash between the put and
    the send is replayed on boot/tick instead of losing both copies. The same
    ``delivery_id`` dedupe guarantees nothing double-delivers. Fail-soft: a
    registration failure only costs the crash insurance, never the send.
    """
    tid = str(task_id or "")
    final = fallback = None
    for event in pending_events:
        if isinstance(event, dict) and event.get("type") == "send_message":
            fallback = event
            if str(event.get("task_id") or "") == tid:
                final = event
    final = final if final is not None else fallback
    if final is None:
        return False
    digest = hashlib.sha256(str(final.get("text") or "").encode("utf-8")).hexdigest()[:16]
    final["delivery_id"] = f"final:{tid}:{digest}"
    if drive_root is not None and str(drive_root).strip() and final.get("chat_id"):
        # OWED before ENQUEUED. Rows without a chat id are not registered (the
        # replay could never send them, so they would only age into a false
        # "delivery gave up" disclosure), and an EMPTY root is refused rather
        # than resolved relative to the process CWD.
        try:
            from supervisor.terminal_delivery import register_pending_delivery

            register_pending_delivery(pathlib.Path(drive_root), dict(final))
        except Exception:
            log.debug("final-answer owed registration failed for %s", tid, exc_info=True)
    try:
        event_queue.put(dict(final))
    except Exception:
        log.warning(
            "Live final-answer delivery failed; keeping buffered delivery",
            exc_info=True,
        )
        return False
    return True


def register_final_answer_owed(
    task: Dict[str, Any], send_event: Dict[str, Any], *, env_drive_root: Any,
) -> None:
    """GR2-5 (§8-A2, ONE outbox for EVERY root): owe the final answer durably.

    Called immediately BEFORE durable result persistence for every non-ephemeral
    ROOT (``agent_task_pipeline.emit_task_results`` registers, then stores), so a
    crash in that window leaves an owed row the boot replay delivers instead of
    a persisted result nobody was told about — the cancel lanes are the ones that
    write first and owe before they SETTLE the intent. Registration happens
    regardless of the blocking/nonblocking post-task split: the nonblocking
    lane used to buffer the send with NO delivery_id and NO owed registration,
    so a worker crash before the buffered drain lost the owner's answer with
    nothing to replay. Mints the canonical ``final:<tid>:<digest>`` id onto the
    buffered event and registers it in the durable outbox; the registry's
    delivery_id dedupe keeps the normal path single-send
    (``deliver_final_message_live`` re-registers idempotently and the send
    handler suppresses the second copy). Fail-soft: registration is crash
    insurance, never a gate on the send. Rows without a chat id are skipped
    (the replay could never send them), and an EMPTY canonical root is refused
    rather than resolved relative to the process CWD (AR2-4).
    """
    if not send_event.get("chat_id"):
        return
    try:
        # The CANONICAL data root the supervisor's boot/tick replay reads: the
        # parent/budget root for split children, the task's own drive for an
        # ordinary root (whose ``budget_drive_root`` is legitimately empty).
        outbox_root = str(task.get("budget_drive_root") or env_drive_root or "").strip()
        if not outbox_root:
            return
        from supervisor.terminal_delivery import register_pending_delivery

        tid = str(task.get("id") or "")
        digest = hashlib.sha256(str(send_event.get("text") or "").encode("utf-8")).hexdigest()[:16]
        send_event["delivery_id"] = f"final:{tid}:{digest}"
        register_pending_delivery(pathlib.Path(outbox_root), dict(send_event))
    except Exception:
        log.debug("final-answer owed registration failed for %s", task.get("id"), exc_info=True)


def build_sealed_final_package(result_row: Any, final_text: str) -> Dict[str, Any]:
    """Host-attested final outcome: delivered text + artifact-store manifest.

    The manifest comes from the DURABLE task result the pipeline just stored
    (whose ``artifacts`` were merged from ``collect_task_artifact_records``,
    the one artifact-store authority) via ``artifact_bundle_from_result``,
    which only re-stats existence on the already-known list — never an
    independent filesystem walk (no second source of truth).
    """
    from ouroboros.outcomes import artifact_bundle_from_result

    row = result_row if isinstance(result_row, dict) else {}
    manifest = [
        {"name": record["name"], "size_bytes": int(record.get("size") or 0),
         "status": str(record.get("status") or "")}
        for record in (artifact_bundle_from_result(row).get("artifacts") or [])
        if isinstance(record, dict) and record.get("name")
    ]
    omitted = max(0, len(manifest) - _SEALED_MANIFEST_MAX_FILES)
    return {
        "final_result_text": str(final_text or ""),
        "artifact_manifest": manifest[:_SEALED_MANIFEST_MAX_FILES],
        **({"artifact_manifest_omitted": omitted} if omitted else {}),
    }


def sealed_final_prompt_section(sealed_final: Dict[str, Any] | None) -> str:
    """Render the sealed package as mandatory ground truth for synthesis prompts."""
    if not isinstance(sealed_final, dict) or not sealed_final:
        return ""
    final_text = truncate_review_artifact(
        str(sealed_final.get("final_result_text") or ""),
        limit=_SEALED_FINAL_TEXT_PROMPT_CHARS,
    ).strip() or "(empty final message)"
    rows = [
        f"- {item.get('name')} ({item.get('size_bytes')} bytes)"
        + (f" [{item.get('status')}]" if item.get("status") not in ("", "ready", None) else "")
        for item in (sealed_final.get("artifact_manifest") or [])
        if isinstance(item, dict) and item.get("name")
    ]
    omitted = int(sealed_final.get("artifact_manifest_omitted") or 0)
    if omitted:
        rows.append(f"- ... {omitted} more file(s) exist in the store (list bounded)")
    manifest_text = "\n".join(rows) if rows else "(no files in the artifact store)"
    return (
        "## Sealed final outcome (host-attested ground truth)\n"
        "Below are the final answer the owner actually received and a host-built\n"
        "manifest of this task's durable artifact store (plain filesystem facts).\n"
        "Outcomes stated here OVERRIDE impressions from the error trace: if the\n"
        "trace suggests failure but this package shows a delivered result or\n"
        "artifact, describe the recovery honestly instead of declaring the\n"
        "deliverable missing.\n"
        "Final result text (as delivered to the owner):\n"
        f"{final_text}\n"
        "Artifact store manifest (task_results/artifacts/<task_id>/):\n"
        f"{manifest_text}\n\n"
    )


def build_swarm_efficiency(env: Any, task: Dict[str, Any]) -> Dict[str, Any] | None:
    """Compact derived swarm-efficiency rollup for a task that fanned out subagents.

    (Moved verbatim from ``agent_task_pipeline`` — that module sits at its
    line ceiling; this is the same finalization-time rollup.)

    Computed from the durable ``swarm_fanout`` telemetry this task already emits
    (control.py:_emit_swarm_fanout): the number of children, the number of fan-out
    waves, the summed inter-wave latency, and the set of model lanes REQUESTED —
    fanout events are written before any child starts, so effective lanes are not
    knowable here; they live on each child's own dispatch record.
    Returns None for a plain task (no fan-out), so the block only appears on real
    swarms.

    OMITTED (no reliable structured source today): ``observed_max_concurrency`` —
    child task results carry only ``ts``/``updated_at``, not a per-child running-start
    vs finish timestamp, so true overlap cannot be derived honestly here — and
    ``parent_blocked_wait_sec`` (wait_task returns prose, not a typed duration).
    """
    task_id = str(task.get("id") or task.get("task_id") or "")
    if not task_id:
        return None
    try:
        from ouroboros.utils import iter_jsonl_objects

        drive_root = getattr(env, "drive_root", None)
        if drive_root is None:
            return None
        events_path = pathlib.Path(drive_root) / "logs" / "events.jsonl"
        child_ids: set[str] = set()
        wave_count = 0
        inter_wave_latency_total = 0.0
        lanes: list[str] = []
        # Read the FULL per-task events stream (not a tail window): the swarm_fanout
        # events can occur EARLY in a long fan-out task, so a bounded tail would
        # silently undercount waves/children (P1 no-silent-loss). This runs once at
        # finalization (not a hot path) and only for fan-out tasks.
        for ev in iter_jsonl_objects(events_path):
            if ev.get("type") != "swarm_fanout":
                continue
            if str(ev.get("parent_task_id") or ev.get("task_id") or "") != task_id:
                continue
            wave_count += 1
            for tid in ev.get("task_ids") or []:
                if str(tid or "").strip():
                    child_ids.add(str(tid))
            try:
                inter_wave_latency_total += float(ev.get("inter_wave_latency_sec") or 0.0)
            except (TypeError, ValueError):
                pass
            # The lane a wave ASKED for. A fan-out event is written before any child
            # starts, so it cannot know what they ran on — that is a per-child
            # dispatch fact and lives on each child's own record.
            lane = str(ev.get("requested_model_lane") or "").strip()
            if lane and lane not in lanes:
                lanes.append(lane)
        if not child_ids:
            return None
        return {
            "subagent_count": len(child_ids),
            "wave_count": wave_count,
            "inter_wave_latency_sec_total": round(inter_wave_latency_total, 3),
            "lanes_requested": lanes,
        }
    except Exception:
        log.debug("swarm efficiency rollup failed", exc_info=True)
        return None
