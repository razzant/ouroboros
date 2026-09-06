"""Terminal delivery and sealed final ground truth for root task finalization.

Extracted seams used by ``agent_task_pipeline`` (kept at its module ceiling).
Incident e9108a09: the owner's answer was ready 34 minutes before delivery
because the buffered terminal events waited behind a hung post-task groom; the
idle reaper then killed the worker and the answer never reached chat, while
the reflection — written from the error trace — declared the delivered PDF
missing. The two seams below fix both halves: the answer leaves early over
the live queue, and synthesis receives the delivered outcome as ground truth
(a prompt input, not a validator — durable writes are never blocked on it).

Terminal provenance extends the same custody rule. A provider outage is an
infrastructure failure even when useful text survived. A complete current
model candidate remains Ouroboros's byte-exact answer and host outage facts
become a separate System incident; stale/last-response/deterministic fallback
bytes are host salvage, preserved durably but never promoted to speech. The
provider rail may recover a compacted transcript from persisted LLM output,
and forced finalization still retains its original semantics: owner-stop gets
one logical model call because steering is fenced, while the generic rail may
refresh once for a late owner directive. These invariants belong here because
this module joins producer provenance, durable result, delivered projection,
and sealed post-task ground truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any, Dict, List

from ouroboros.utils import truncate_review_artifact

log = logging.getLogger(__name__)

# Prompt bounds: the overflow count is disclosed instead of silently dropped.
_SEALED_MANIFEST_MAX_FILES = 200
_SEALED_FINAL_TEXT_PROMPT_CHARS = 4000
_COMPLETION_SKILL_PREVIEW_ROWS = 20

# Closed producer vocabulary. Missing remains a valid legacy state and must
# never be inferred from result text or lifecycle status.
TERMINAL_ORIGIN_MODEL_FINAL = "model_final"
TERMINAL_ORIGIN_HOST_SALVAGE = "host_salvage"
# A terminal text the HOST wrote alone (a budget rejection, a round-limit rail
# with nothing to deliver, a scheduled swarm handoff). It is not salvage: its
# own words ARE the answer, so they are published verbatim on every transport
# instead of being replaced by the outage receipt.
TERMINAL_ORIGIN_HOST_NOTICE = "host_notice"
HOST_AUTHORED_TERMINAL_ORIGINS = frozenset({
    TERMINAL_ORIGIN_HOST_SALVAGE, TERMINAL_ORIGIN_HOST_NOTICE,
})
_STAMPED_TERMINAL_ORIGINS = frozenset({
    TERMINAL_ORIGIN_MODEL_FINAL, *HOST_AUTHORED_TERMINAL_ORIGINS,
})
TERMINAL_PLAN_REVIEW_NOTE = (
    "Plan review was still open when the outage forced finalization; "
    "its details remain in the task."
)


def send_provider_death_notice(
    ctx: Any, chat_id: int, task_id: Any, final_result: Dict[str, Any],
) -> bool:
    """Send the secondary incident, unless the primary is already the receipt."""
    if str(final_result.get("terminal_origin") or "") == TERMINAL_ORIGIN_HOST_SALVAGE:
        return False
    plan_note = (
        f"\n\n{TERMINAL_PLAN_REVIEW_NOTE}"
        if final_result.get("terminal_plan_review_open") is True else ""
    )
    notice = str(final_result.get("terminal_provider_notice") or "") or (
        "A model-provider outage stopped this task. Partial work and workspace files "
        "are preserved; inspect the task details before starting another run."
    )
    ctx.send_with_budget(
        chat_id,
        f"🔌 Task {task_id} was NOT completed.\n\n{notice}{plan_note}",
        role="system",
        system_type="terminal_incident",
    )
    return True


def provider_terminal_body(text: str, notice: str) -> str:
    """Render a host-labelled status beside raw text on single-body transports."""
    if not notice:
        return text
    return (text + "\n\n" if text else "") + "[Host status]\n" + notice


def stamp_root_final_phase(
    send_event: Dict[str, Any], task: Dict[str, Any], *, post_task_open: bool, terminal_status: str,
) -> None:
    """Type a root's final frame for the client's live conclusion gate.

    With post-task synthesis still OPEN the owner's answer leaves early: the
    typed phase marker (progress_meta merges into the WS chat payload) holds
    the card on "Finalizing…" until the settled task_done, instead of the
    early final reading as the task's terminal conclusion. With post-task
    already settled a DIRECT turn's bare final IS the turn's terminal word
    (#369), so it names ``terminal_status`` — the status the durable row
    settles to, which the chat row persists and replay reads as the card's
    phase (a stopped turn is ``failed``, never a blanket ``completed``) —
    managed roots keep their task_done conclusion untouched.
    """
    if post_task_open:
        send_event.setdefault("progress_meta", {})["task_phase"] = "finalizing"
    elif task.get("_is_direct_chat"):
        send_event.setdefault("progress_meta", {})["task_terminal_status"] = terminal_status


def prepare_terminal_send_event(
    env_drive_root: Any, task: Dict[str, Any], text: str,
    usage: Dict[str, Any], send_event: Dict[str, Any],
    *, ephemeral: bool, presence: bool,
) -> Dict[str, Any]:
    """Preserve raw host salvage, then build the one live/replay projection."""
    origin = str(usage.get("terminal_origin") or "")
    notice = str(usage.get("terminal_provider_notice") or "")
    if ephemeral and not presence:
        # #369: an ephemeral decision's task_done frame is dropped at the
        # client's log-event entry by design, so this final is the turn's
        # ONLY conclusion vehicle. The typed fact mirrors the direct-error
        # branch (supervisor/workers.py stamps task_terminal_status="failed")
        # and lets the live concludesTurn gate settle the activity.
        send_event.setdefault("progress_meta", {})["task_terminal_status"] = "completed"
    if origin not in _STAMPED_TERMINAL_ORIGINS:
        return send_event
    canonical_root = pathlib.Path(task.get("budget_drive_root") or env_drive_root)
    preserved_path = ""
    if text and (origin == TERMINAL_ORIGIN_HOST_SALVAGE or (ephemeral and notice)):
        try:
            from ouroboros.observability import preserve_salvaged_output

            preserved_path = preserve_salvaged_output(
                canonical_root, str(task.get("id") or ""), text,
            )
        except Exception:
            log.warning("Failed to pre-preserve terminal host salvage", exc_info=True)
        usage["terminal_salvage_path"] = preserved_path
    if presence:
        return send_event  # Presence's existing body renderer owns its delivery outcome.
    if ephemeral:
        if notice:
            body = ("Preserved intermediate output (not a final answer):\n" + text
                    if origin == TERMINAL_ORIGIN_HOST_SALVAGE and text else text)
            body = provider_terminal_body(body, notice)
            send_event.update(text=body, log_text=body)
        return send_event  # no task-details promise on a turn with no durable task row
    from supervisor.terminal_delivery import project_terminal_result_event

    return project_terminal_result_event(
        canonical_root, task, str(task.get("id") or ""),
        result_text=text, terminal_origin=origin, base_event=send_event, provider_notice=notice,
    )


def terminal_result_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Additive durable origin/full-copy fields; unknown producers stay legacy."""
    fields: Dict[str, Any] = {}
    origin = str(usage.get("terminal_origin") or "")
    if origin in _STAMPED_TERMINAL_ORIGINS:
        fields["terminal_origin"] = origin
    path = str(usage.get("terminal_salvage_path") or "")
    if path:
        fields["terminal_salvage_path"] = path
    if usage.get("terminal_plan_review_open") is True:
        fields["terminal_plan_review_open"] = True
    if isinstance(usage.get("terminal_provider_notice"), str) and usage["terminal_provider_notice"]:
        fields["terminal_provider_notice"] = usage["terminal_provider_notice"]
    return fields


def deliver_final_message_live(
    event_queue: Any, pending_events: List[Dict[str, Any]], task_id: str,
    *, drive_root: Any = None,
) -> bool:
    """Send the buffered FINAL ``send_message`` through the live worker queue.

    The buffer can also hold proactive ``send_user_message`` events that fell
    back to deferred delivery mid-task (live-first frames stamp ``task_id``
    too), so the final answer is selected as the LAST send_message matching
    the finalizing task's id — the host appends the terminal frame after all
    tool-time frames, so it wins the last-match scan — never the first match,
    which would ship a proactive text early while the answer stayed hostage
    to blocking post-task.

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
    if not str(final.get("delivery_id") or "").strip():
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
        if not str(send_event.get("delivery_id") or "").strip():
            digest = hashlib.sha256(str(send_event.get("text") or "").encode("utf-8")).hexdigest()[:16]
            send_event["delivery_id"] = f"final:{tid}:{digest}"
        register_pending_delivery(pathlib.Path(outbox_root), dict(send_event))
    except Exception:
        log.debug("final-answer owed registration failed for %s", task.get("id"), exc_info=True)


def build_completion_observations(drive_root: Any, task: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    """Retain task-observed actions for synthesis without inventing delivery receipts.

    The exact recorded returns live in the existing artifact store. Packet-only
    synthesis receives counts and the latest return of EACH send-family tool,
    so an earlier photo is not hidden by a tail of later text sends. Skill state
    is the existing task-related readiness projection, not owner-click authorship.
    """
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.observability import redact_projection
    from ouroboros.skill_readiness import acceptance_skill_lifecycle
    from ouroboros.tool_capabilities import OWNER_DELIVERY_TOOL_NAMES
    from ouroboros.utils import utc_now_iso

    calls = trace.get("tool_calls")
    deliveries = [{key: call[key] for key in (
        "tool", "tool_call_id", "status", "code", "result", "result_partial", "is_error",
    ) if key in call} for call in (calls or [])
        if isinstance(call, dict) and call.get("tool") in OWNER_DELIVERY_TOOL_NAMES]
    coverage: Dict[str, Any] = {}
    try:
        skills = acceptance_skill_lifecycle(
            task.get("budget_drive_root") or drive_root, trace,
            str(task.get("root_task_id") or task.get("id") or ""),
            task_started_at=str(task.get("started_at") or ""), history_coverage=coverage,
        )
    except Exception:
        skills, coverage = [], {"complete": False, "status": "unavailable"}
        log.debug("Completion skill-state observations unavailable", exc_info=True)
    snapshot = redact_projection({
        "observed_at": utc_now_iso(), "trace_available": isinstance(calls, list),
        "delivery_results": deliveries, "skill_state": skills,
        "skill_history_coverage": coverage,
        "delivery_receipt_coverage": "not_observed_by_tool_trace",
        "skill_state_scope": "current_state_of_task_related_skills; not action attribution",
    }).value
    counts: Dict[str, Any] = {}
    latest: Dict[str, Any] = {}
    for row in snapshot["delivery_results"]:
        name = str(row["tool"])
        count = counts.setdefault(name, {"calls": 0, "reported_ok": 0, "status_unknown": 0})
        count["calls"] += 1
        count["reported_ok"] += int(row.get("status") == "ok")
        count["status_unknown"] += int(not row.get("status"))
        latest[name] = {**row, "result": truncate_review_artifact(str(row.get("result") or ""), limit=600)}
    projection = {**snapshot, "delivery_counts": counts, "delivery_results": list(latest.values()),
                  "delivery_results_omitted": len(deliveries) - len(latest),
                  "skill_state": snapshot["skill_state"][:_COMPLETION_SKILL_PREVIEW_ROWS],
                  "skill_state_omitted": max(0, len(skills) - _COMPLETION_SKILL_PREVIEW_ROWS)}
    if not deliveries and not skills:
        return projection  # no full action record to store for an ordinary empty turn
    raw = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    try:
        projection["source_ref"] = store_actor_source_bytes(
            task.get("budget_drive_root") or drive_root, str(task.get("id") or ""),
            category="context_checkpoints", source_id="completion", data=raw, extension="json",
        )
        projection["source_ref"]["reader"] = {
            "tool": "get_task_result",
            "arguments": {"task_id": str(task.get("id") or ""), "include_completion_source": True},
        }
        projection["source_status"] = "available"
    except (OSError, ValueError, TimeoutError):
        projection["source_status"] = "unavailable"
        log.warning("Completion observations full source unavailable", exc_info=True)
    return projection


def completion_source_projection(
    drive_root: Any, task_id: str, result: Dict[str, Any], start_char: Any = None, end_char: Any = None,
) -> Dict[str, Any]:
    """Read the selected task's complete stored observations through its canonical root."""
    from ouroboros.artifacts import read_actor_source_bytes, text_source_range_projection

    unavailable = {"schema": 1, "kind": "task_completion_observations", "status": "unavailable"}
    observations = result.get("completion_observations")
    ref = observations.get("source_ref") if isinstance(observations, dict) else None
    if not isinstance(ref, dict) or ref.get("kind") != "task_source":
        return {**unavailable, "reason": "source_unavailable"}
    try:
        raw = read_actor_source_bytes(drive_root, str(result.get("task_id") or task_id), ref)
        projection, reason = text_source_range_projection(raw.decode("utf-8"), unavailable["kind"], start_char, end_char)
    except ValueError as exc:
        reason = "source_identity_mismatch" if "verification" in str(exc) else "source_ref_invalid"
        return {**unavailable, "reason": reason}
    except (OSError, RuntimeError):
        return {**unavailable, "reason": "source_unavailable"}
    payload = projection or unavailable
    return {**payload, **({"reason": reason} if reason else {})}


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
        "completion_observations": row.get("completion_observations") or {"status": "unavailable"},
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
    observations = json.dumps(sealed_final.get("completion_observations") or {"status": "unavailable"},
                              ensure_ascii=False, default=str)
    return (
        "## Sealed final outcome (host-attested ground truth)\n"
        "Below are the final answer submitted for delivery and a host-built\n"
        "manifest of this task's durable artifact store (plain filesystem facts).\n"
        "Outcomes stated here OVERRIDE impressions from the error trace: if the\n"
        "trace suggests failure but this package shows a delivered result or\n"
        "artifact, describe the recovery honestly instead of declaring the\n"
        "deliverable missing.\n"
        "An empty final answer, manifest, or observation section is not evidence that no action occurred.\n"
        "Tool success records a submitted/queued action, not a chat receipt or proof the owner received it.\n"
        "Skill readiness is current task-related state; it does not attribute an owner's click to this task.\n"
        "Use the inline counts/results and coverage below; source refs are for later agent readers,\n"
        "not additional evidence you have read. Omitted or unavailable facts remain unknown.\n"
        "Final result text (submitted for delivery):\n"
        f"{final_text}\n"
        "Artifact store manifest (task_results/artifacts/<task_id>/):\n"
        f"{manifest_text}\nTask completion observations:\n{observations}\n\n"
    )


def build_swarm_efficiency(env: Any, task: Dict[str, Any]) -> Dict[str, Any] | None:
    """Compact derived swarm-efficiency rollup: observed fan-out, or the
    zero-fanout disclosure block for a host-attested Swarm-intent task.

    (Moved verbatim from ``agent_task_pipeline`` — that module sits at its
    line ceiling; this is the same finalization-time rollup.)

    Computed from the durable ``swarm_fanout`` telemetry this task already emits
    (control.py:_emit_swarm_fanout): the number of children, the number of fan-out
    waves, the summed inter-wave latency, and the set of model lanes REQUESTED —
    fanout events are written before any child starts, so effective lanes are not
    knowable here; they live on each child's own dispatch record.
    Returns None for a plain task (no fan-out), so the block only appears on real
    swarms — with ONE exception: a task admitted with host-attested Swarm intent
    (typed metadata ``force_plan_source == "swarm"``, never prompt inspection) that
    fanned out NOTHING returns a minimal ``no_fanout_observed`` block instead of
    disappearing, so a Swarm-button task that spawned zero children is
    distinguishable from a plain task. ``planned`` in that block is null — never
    inferred as 0 from the absence of events; a real planned figure exists only as
    the waves' ``requested_count`` sum, surfaced under that exact name on
    swarm-intent rollups.

    OMITTED (no reliable structured source today): ``observed_max_concurrency`` —
    child task results carry only ``ts``/``updated_at``, not a per-child running-start
    vs finish timestamp, so true overlap cannot be derived honestly here — and
    ``parent_blocked_wait_sec`` (wait_task returns prose, not a typed duration).
    """
    task_id = str(task.get("id") or task.get("task_id") or "")
    if not task_id:
        return None
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    swarm_intent = metadata.get("force_plan_source") == "swarm"
    try:
        from ouroboros.utils import iter_jsonl_chain_objects

        drive_root = getattr(env, "drive_root", None)
        if drive_root is None:
            return None
        events_path = pathlib.Path(drive_root) / "logs" / "events.jsonl"
        child_ids: set[str] = set()
        wave_count = 0
        requested_count_total = 0
        inter_wave_latency_total = 0.0
        lanes: list[str] = []
        # Read the FULL per-task events stream (not a tail window): the swarm_fanout
        # events can occur EARLY in a long fan-out task, so a bounded tail would
        # silently undercount waves/children (P1 no-silent-loss). This runs once at
        # finalization (not a hot path), for fan-out and Swarm-intent tasks.
        # Chain-aware (CPL4-C1): early fan-out events may already have rotated
        # into archive/events_*.jsonl by finalization time.
        for ev in iter_jsonl_chain_objects(events_path):
            if ev.get("type") != "swarm_fanout":
                continue
            if str(ev.get("parent_task_id") or ev.get("task_id") or "") != task_id:
                continue
            wave_count += 1
            try:
                requested_count_total += int(ev.get("requested_count") or 0)
            except (TypeError, ValueError):
                pass
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
            if swarm_intent:
                # The owner asked for a swarm and nothing fanned out: say so instead
                # of vanishing — this is the zero-fan-out visibility the block exists
                # for. ``planned`` is null because no requested counts were ever
                # recorded; 0 would be an inference from absence, not an observation.
                return {
                    "intent_source": "swarm",
                    "planned": None,
                    "observed_started": 0,
                    "status": "no_fanout_observed",
                }
            return None
        rollup: Dict[str, Any] = {
            "subagent_count": len(child_ids),
            "wave_count": wave_count,
            "inter_wave_latency_sec_total": round(inter_wave_latency_total, 3),
            "lanes_requested": lanes,
        }
        try:
            # Depth is the one swarm fact the root could not see: its own
            # contract carries the request, the subtree carries what was
            # actually reached. Its own try/except — the enclosing one returns
            # None for the WHOLE rollup, and a subtree read failure must not
            # erase the fan-out numbers.
            from ouroboros.depth_evidence import build_depth_summary
            from ouroboros.task_status import find_child_tasks

            canonical = pathlib.Path(task.get("budget_drive_root") or drive_root)
            rollup["depth"] = build_depth_summary(
                task.get("task_contract"),
                find_child_tasks(
                    canonical, parent_task_id=task_id, root_task_id=task_id,
                    scope="subtree", materialize_artifacts=False,
                ),
            )
        except Exception:
            log.debug("swarm depth summary failed", exc_info=True)
        if swarm_intent:
            rollup["intent_source"] = "swarm"
            # The planned figure under its existing event name — the waves'
            # requested_count sum, no synonyms (rc-phaseC, fable 2.3 disposition).
            rollup["requested_count"] = requested_count_total
        return rollup
    except Exception:
        log.debug("swarm efficiency rollup failed", exc_info=True)
        return None
