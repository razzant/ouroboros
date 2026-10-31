"""Post-task self-evolution (V4 envelope + V5 promotion) — owner-gated, LLM-first.

After a qualifying task completes, OPTIONALLY promote one concrete, high-value
self-improvement into the EXISTING gated evolution campaign. The worker writes a
durable request file on the canonical drive; the supervisor's idle tick applies
it via ``start_evolution_campaign`` + enabling evolution, after which the normal
``enqueue_evolution_task_if_needed`` runs the cycle through EVERY safety gate
(idle, restart-verify, 3-fail breaker, budget reserve, advanced/pro,
owner_chat_id).

Invariants (red-team guards — keep intact):
- The worker NEVER enqueues or enables evolution itself; it only writes a durable
  signal that the gated supervisor tick applies (R1.1).
- Promotion never fires from evolution/deep_self_review/subagent tasks (R1.2
  loop guard), nor on a non-canonical (child) dual-run pass.
- The promotion DECISION is a structured LLM judgment, never keyword/threshold
  (R1.3 / BIBLE P5).
- A promoted item that requires plan review carries that obligation into the
  objective (R4.1 advisory->reviewed boundary).
- Default OFF; only the owner enables the envelope.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, Optional

from ouroboros.consciousness_authority import consciousness_origin_metadata, task_disabled_tools
from ouroboros.evolution_fingerprint import _PLAN_REVIEW_SUFFIX
from ouroboros.config import runtime_setting

log = logging.getLogger(__name__)

_REQUEST_REL = "state/post_task_evolution_request.json"
_COUNTER_REL = "state/post_task_evolution_counter.json"
_SKIP_TYPES = frozenset({"evolution", "deep_self_review"})


def drop_pending_request(drive_root: Any) -> None:
    """Best-effort delete of any pending post-task promotion request. Called by the
    owner-stop sites (the fast path); the DURABLE backstop against re-arm is the
    ``evolution_owner_stopped`` state flag checked in :func:`apply_pending_request`,
    which also catches a request a worker re-creates after this delete. Never raises."""
    try:
        (pathlib.Path(str(drive_root)) / _REQUEST_REL).unlink(missing_ok=True)
    except Exception:
        pass


def _resolve(value: Any) -> Optional[pathlib.Path]:
    try:
        return pathlib.Path(str(value)).resolve(strict=False)
    except Exception:
        return None


def _is_canonical_run(env: Any, task: Dict[str, Any]) -> bool:
    """True when ``env.drive_root`` is the canonical drive (a shared task, or the
    parent dual-run pass for forked/empty/workspace). Prevents double promotion
    and targets the canonical backlog/campaign."""
    bdr = str(task.get("budget_drive_root") or "").strip()
    if not bdr:
        return True
    a, b = _resolve(bdr), _resolve(getattr(env, "drive_root", ""))
    return bool(a and b and a == b)


def _eligible(task: Dict[str, Any]) -> bool:
    if str(task.get("type") or "") in _SKIP_TYPES:
        return False
    if str(task.get("delegation_role") or "") == "subagent":
        return False
    # ONE fact: a task whose contract withholds toggle_evolution (a consciousness
    # wake-up below Full, and every root it started) may not propose evolution
    # either — otherwise Act would reach a campaign through one indirection.
    if "toggle_evolution" in task_disabled_tools(task):
        return False
    return True


def _parse_every_n(cadence: str) -> int:
    try:
        if ":" in cadence:
            return max(1, int(cadence.split(":", 1)[1].strip()))
    except (ValueError, TypeError):
        pass
    return 1


def _counter_due(drive_root: pathlib.Path, k: int) -> bool:
    path = drive_root / _COUNTER_REL
    n = 0
    try:
        if path.exists():
            n = int(json.loads(path.read_text(encoding="utf-8")).get("n") or 0)
    except Exception:
        n = 0
    n += 1
    try:
        from ouroboros.utils import atomic_write_json

        atomic_write_json(path, {"n": n})
    except Exception:
        pass
    return (n % max(1, k)) == 0


def _loose_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


_DECISION_PROMPT = """You decide whether Ouroboros should run ONE reviewed self-improvement (evolution) cycle now, based on the task it just finished, its improvement backlog and past attempts.

[JUST-FINISHED TASK — host-recorded facts]
{task_facts}

[IMPROVEMENT BACKLOG — the complete authored note while it is active (its structure, statuses and any counts are its author's text), otherwise its archived or missing state]
{backlog}

[SOLVE-CAPABILITY HISTORY — what past evolution cycles actually landed]
{capability}

[PAST ATTEMPTS — objectives earlier cycles shipped (ABSORBED), dropped (ABANDONED / NO_OP) or found hard-blocked (BLOCKED); evidence about what happened, not a permanent verdict]
{closed}

[ACTIVE CAMPAIGN OBJECTIVE — a reviewed cycle is ALREADY running this; do NOT re-propose it]
{active_objective}

Return ONLY a JSON object:
{{"promote": true|false, "objective": "<one concrete, self-contained improvement to Ouroboros's own code/process; empty if not promoting>", "requires_plan_review": true|false, "backlog_id": "<the backlog entry this maps to, as the note names it; else empty>"}}

Rules: set promote=true ONLY when there is a concrete, high-value, self-contained code/process improvement worth a reviewed cycle right now. Weigh the solve-capability history: objective classes that historically got ABSORBED are better bets than classes that kept ending no_op/abandoned. Bias toward SMALL, TARGETED objectives that directly improve the ability to solve tasks (a sharper tool, a fixed failure mode, a removed bottleneck) over broad refactors or speculative platform work — small reviewed wins absorb; sprawling objectives historically die as no_op. Do NOT restate the ACTIVE CAMPAIGN OBJECTIVE: a cycle is already running it. Proposing any past objective again (absorbed, abandoned, no-op or blocked) needs a reason its earlier outcome does not cover. If nothing is clearly worthwhile, return promote=false."""


def _closed_objectives_digest(drive_root: pathlib.Path) -> Optional[str]:
    """BUG3 Layer A: past attempt outcomes the chooser weighs as evidence.

    Built from the STRUCTURED ledger (state/evolution_checkpoints.jsonl), not prose,
    so it cannot rot when prose formatting changes. An objective is closed when its latest cycle
    outcome is absorbed (already shipped), abandoned/no_op (attempted and dropped), or the
    objective/review axis recorded outcome_tier == "blocked_with_evidence" (hard-blocked).
    A typed review-cap/unavailable stop, or a result review of the task still pending
    in the review ledger, is only this attempt's limit, not a
    permanently closed objective. Its retained task source remains discoverable;
    this readout neither schedules a retry nor resets campaign counters.
    Deduped by the SSOT fingerprint so the same base objective appears once. "" when nothing.
    """
    import json as _json

    from ouroboros.evolution_fingerprint import canonical_objective_fingerprint
    from ouroboros.outcomes import (
        ACCEPTANCE_FINALIZED_UNACCEPTED, REASON_REVIEW_CYCLES_EXHAUSTED,
        REASON_REVIEW_QUORUM_UNREACHABLE,
    )

    path = pathlib.Path(drive_root) / "state" / "evolution_checkpoints.jsonl"
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    by_task: Dict[str, Dict[str, Any]] = {}
    order: list = []
    for line in lines:
        try:
            row = _json.loads(line)
        except Exception:
            return None
        if not isinstance(row, dict):
            return None
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        if task_id not in by_task:
            order.append(task_id)
        merged = by_task.setdefault(task_id, {})
        obj = str(row.get("campaign_objective") or "")
        if obj:
            merged["objective"] = obj
        if str(row.get("kind") or "") == "cycle_outcome" and row.get("cycle_outcome"):
            merged["cycle_outcome"] = str(row.get("cycle_outcome") or "")
        else:
            tx = row.get("transaction") if isinstance(row.get("transaction"), dict) else {}
            merged.setdefault("cycle_outcome", str(tx.get("cycle_outcome") or ""))
        axes = row.get("outcome_axes") if isinstance(row.get("outcome_axes"), dict) else {}
        objective = axes.get("objective") if isinstance(axes.get("objective"), dict) else {}
        review = axes.get("review") if isinstance(axes.get("review"), dict) else {}
        # An old row may carry a task-acceptance decision written before the result review.
        decision = review.get("acceptance_decision") if isinstance(review.get("acceptance_decision"), dict) else {}
        if (objective.get("reason") in {REASON_REVIEW_CYCLES_EXHAUSTED, REASON_REVIEW_QUORUM_UNREACHABLE}
                or (decision.get("status") == ACCEPTANCE_FINALIZED_UNACCEPTED
                    and decision.get("reason") in {REASON_REVIEW_CYCLES_EXHAUSTED, "review_degraded"})):
            merged["waiting_for_review"] = True
        for axis in ("objective", "review"):
            axis_obj = axes.get(axis) if isinstance(axes.get(axis), dict) else {}
            if str(axis_obj.get("outcome_tier") or "") == "blocked_with_evidence":
                merged["blocked"] = True
    pending = _pending_result_review_tasks(drive_root)
    seen = set()
    out: list = []
    for task_id in reversed(order):  # newest first
        info = by_task.get(task_id) or {}
        blocked = bool(info.get("blocked"))
        outcome = str(info.get("cycle_outcome") or "")
        waiting = (info.get("waiting_for_review") or task_id in pending) and outcome not in {"absorbed", "abandoned"}
        if not waiting and outcome not in {"absorbed", "abandoned", "no_op"} and not blocked:
            continue
        objective = str(info.get("objective") or "").strip().replace("\n", " ")
        if not objective:
            continue
        fp = canonical_objective_fingerprint(objective)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        if waiting:
            # The current independently initiated attempt supersedes an older
            # dropped attempt of this objective; its history is still in the
            # solve-capability digest, without a permanent "do not reconsider".
            continue
        tag = "BLOCKED" if blocked else (str(info.get("cycle_outcome") or "DROPPED").upper())
        out.append(f"- [{tag}] {objective}")
    return "\n".join(out)


def _pending_result_review_tasks(drive_root: pathlib.Path) -> set:
    """Tasks whose own result review is still pending, read once from the review
    ledger's hot index (a row count can never exceed the index's byte bound)."""
    try:
        from ouroboros import review_ledger as ledger

        rows = ledger.recent_records(drive_root, limit=ledger.INDEX_MAX_BYTES, hot_only=True, surface="result")
    except Exception:
        log.debug("Review ledger unreadable for the closed-objectives digest", exc_info=True)
        return set()
    return {str(row.get("task_id") or "") for row in rows if row.get("state") == ledger.STATE_PENDING}


def _active_campaign_objective() -> str:
    """BUG3 Layer A: the current campaign objective (a cycle is already running it)."""
    try:
        from supervisor.evolution_lifecycle import _read_evolution_campaign
        return str(_read_evolution_campaign().get("objective") or "").strip()
    except Exception:
        return ""


def _task_facts(task: Dict[str, Any], sealed_final: Optional[Dict[str, Any]]) -> str:
    """The finished run's recorded facts: its exact initial text and its sealed outcome.

    The initial text rides whole; a request the route cannot hold fails the stage visibly.
    A split Project root's globalized view deliberately omits it, and this tool-less
    chooser has no reader to reach it, so the prompt says it is unavailable here.
    The sealed outcome is bounded, not whole: ``task_finalization.sealed_final_prompt_section``
    cuts each recorded text (final answer, internal text, host notice) at
    ``_SEALED_FINAL_TEXT_PROMPT_CHARS`` (4000) under an OMISSION NOTE, and the artifact
    manifest names its omitted count.
    """
    from ouroboros.task_finalization import sealed_final_prompt_section

    text = str(task.get("text") or "")
    initial = (f"Initial text of the run (exact):\n{text}" if text else
               "Initial text of the run: unavailable in this view.")
    sealed = sealed_final_prompt_section(sealed_final) or "Sealed final outcome: not recorded for this run.\n"
    return f"{initial}\n\n{sealed}"


def _decide_promotion(env: Any, task: Dict[str, Any], sealed_final: Optional[Dict[str, Any]],
                      llm_client: Any) -> Optional[Dict[str, Any]]:
    from ouroboros.improvement_backlog import authored_backlog_text
    from ouroboros.utils import truncate_review_artifact

    drive_root = pathlib.Path(str(env.drive_root))
    try:
        from ouroboros.evolution_checkpoints import build_solve_capability_digest
        capability = truncate_review_artifact(build_solve_capability_digest(drive_root), 2000)
    except Exception:
        capability = ""
    # BUG3 Layer A: give the chooser the past attempt outcomes (without them a CLOSED
    # objective was re-suggested 4-5x). Sourced from the structured ledger plus the
    # campaign-local dropped set, deduped by fingerprint; evidence, not a verdict.
    closed = _closed_objectives_digest(drive_root)
    if closed is None:
        log.warning("post_task_evolution: closed-objective history unavailable; abstaining")
        return None
    active_objective = _active_campaign_objective()
    prompt = _DECISION_PROMPT.format(
        task_facts=_task_facts(task, sealed_final), backlog=authored_backlog_text(drive_root),
        capability=capability or "(no evolution-cycle history yet)",
        closed=closed or "(none)", active_objective=active_objective or "(no active campaign)",
    )
    # Failures raise to the post-task stage: only an abstention returns None.
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.llm import LLMClient
    from ouroboros.llm_observability import chat_observed

    client = llm_client or LLMClient()
    # Main-slot chooser (plan 5C): picking the next evolution objective is a
    # high-leverage cognitive decision, not a cheap-lane formatting call.
    chooser_model = str(
        runtime_setting("OUROBOROS_MODEL", "") or SETTINGS_DEFAULTS["OUROBOROS_MODEL"]
    ).strip()
    resp, usage = chat_observed(
        client,
        drive_root=drive_root,
        task_id=str(task.get("id") or "post_task_evolution"),
        call_type="post_task_evolution_decision",
        model_role="main",
        messages=[{"role": "user", "content": prompt}],
        model=chooser_model,
        reasoning_effort="medium",
        max_tokens=8192,
    )
    if usage:
        try:
            from supervisor.state import update_budget_from_usage

            update_budget_from_usage(usage)
        except Exception:
            pass
    obj = _loose_json((resp.get("content") or "").strip())
    if not obj:
        return None
    return {
        "promote": bool(obj.get("promote")),
        "objective": str(obj.get("objective") or "").strip(),
        # Default to requiring plan review (preserve the advisory->reviewed boundary).
        "requires_plan_review": bool(obj.get("requires_plan_review", True)),
        "backlog_id": str(obj.get("backlog_id") or "").strip(),
    }


def _write_request(drive_root: pathlib.Path, decision: Dict[str, Any], task: Dict[str, Any]) -> None:
    from ouroboros.utils import utc_now_iso

    req = {
        "schema_version": 1,
        "ts": utc_now_iso(),
        "objective": decision["objective"],
        "requires_plan_review": bool(decision.get("requires_plan_review", True)),
        "backlog_id": decision.get("backlog_id") or "",
        "source": "post_task",
        "origin_task_id": str(task.get("id") or ""),
        # A Full-level consciousness tree proposing evolution keeps its origin, so
        # the campaign and its cycle tasks stay inside the consciousness allowance.
        **consciousness_origin_metadata(task.get("metadata")),
    }
    path = drive_root / _REQUEST_REL
    # Atomic publish: the supervisor polls every tick, so a partial write must
    # never be observable (else it could parse-fail and drop the signal).
    from ouroboros.utils import atomic_write_json

    atomic_write_json(path, req)


def maybe_promote(env: Any, task: Dict[str, Any], sealed_final: Optional[Dict[str, Any]],
                  llm_client: Any = None) -> Optional[Dict[str, Any]]:
    """Worker-side: write a durable promotion signal if the envelope is on and the
    chooser finds a worthwhile self-improvement in the finished run's recorded facts
    (``sealed_final``) and the authored backlog. Returns the decision or None. NEVER
    enqueues/enables evolution (that is the supervisor's job). A failure raises to
    the post-task promotion stage: None is only a genuine no-op, never a swallowed
    chooser failure."""
    from ouroboros.config import (
        get_post_task_evolution_cadence,
        get_post_task_evolution_enabled,
        get_runtime_mode,
    )

    if not get_post_task_evolution_enabled():
        return None
    if get_runtime_mode() == "light":
        return None
    if not _eligible(task) or not _is_canonical_run(env, task):
        return None
    # A project-scoped task never triggers GLOBAL self-evolution (defense in
    # depth; the post-task pipeline also gates this). Project work stays isolated.
    from ouroboros.project_facts import resolve_project_id

    if resolve_project_id(task):
        return None
    cadence = get_post_task_evolution_cadence()
    if cadence == "off":
        return None
    drive_root = pathlib.Path(str(env.drive_root))
    # A stored every_n schedule says WHEN to ask; whether to evolve stays the chooser's judgment.
    if cadence.startswith("every_n") and not _counter_due(drive_root, _parse_every_n(cadence)):
        return None
    decision = _decide_promotion(env, task, sealed_final, llm_client)
    if not decision or not decision.get("promote") or not decision.get("objective"):
        return None
    _write_request(drive_root, decision, task)
    log.info("post_task_evolution: durable promotion signal written (origin task=%s)",
             str(task.get("id") or ""))
    return decision


def _safe_unlink(path: pathlib.Path) -> None:
    try:
        path.unlink()
    except Exception:
        pass


def apply_pending_request(drive_root: Any) -> bool:
    """Supervisor-side (idle tick): apply a pending durable promotion through the
    gated campaign machinery. Sets the campaign objective + enables evolution so
    the normal idle-tick ``enqueue_evolution_task_if_needed`` runs the cycle under
    all safety gates. Marks ``post_task_autostop`` so the absorbed cycle disables
    evolution again (one-shot). Returns True if a promotion was applied."""
    try:
        from ouroboros.config import get_post_task_evolution_enabled

        if not get_post_task_evolution_enabled():
            return False
        path = pathlib.Path(str(drive_root)) / _REQUEST_REL
        if not path.exists():
            return False
        try:
            req = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # A transient/partial read should not drop the durable signal; leave
            # the file for the next tick (atomic writes make corruption unlikely).
            return False
        objective = str((req or {}).get("objective") or "").strip()
        if not objective:
            _safe_unlink(path)
            return False

        from supervisor.evolution_lifecycle import evolution_block_reason, evolution_stop_reason, start_evolution_campaign
        from supervisor.state import control_value, load_state

        if evolution_block_reason():  # light runtime mode, etc.
            _safe_unlink(path)
            return False
        st = load_state()
        if not all(control_value(st, key)[0] for key in (
                "evolution_owner_stopped", "evolution_mode_enabled", "owner_chat_id")) or evolution_stop_reason():
            # An autonomous re-arm needs KNOWN current controls and no received Stop/Panic
            # (#1307): unknown is neither "not stopped" nor "not enabled". The durable
            # request stays for a later tick; only a known stop drops it (below).
            return False
        if not st.get("owner_chat_id"):
            # Evolution requires an owner-bound chat; without it the cycle could
            # never run. Drop the stale request rather than leaking it.
            _safe_unlink(path)
            return False
        if bool(st.get("evolution_owner_stopped")):
            # The owner EXPLICITLY stopped evolution (/evolve off, toggle_evolution(False),
            # or panic). Do NOT re-arm it from a queued post-task promotion — that would
            # silently flip evolution_mode_enabled back True with no /evolve start. The
            # flag is durable and cleared ONLY by an owner /evolve start, so this is the
            # authoritative gate (the campaign status alone is not: start_evolution_campaign
            # would just mint a fresh active campaign). It also drops a request a worker's
            # maybe_promote re-created after the owner-stop cleared the file (maybe_promote
            # is ungated on this flag), closing that race deterministically.
            _safe_unlink(path)
            return False
        if bool(st.get("evolution_mode_enabled")):
            # A campaign is already enabled (owner-driven or a previous
            # promotion). Activating another would hijack its objective and
            # clear its failure counter; leave the request for a later tick.
            return False
        # Per-window budget floor (V4 envelope): if configured, do not start a
        # post-task cycle unless at least that much budget remains.
        from ouroboros.config import get_post_task_evolution_budget_usd

        budget_floor = get_post_task_evolution_budget_usd()
        if budget_floor > 0:
            from supervisor.state import budget_remaining

            try:
                remaining = budget_remaining(st, strict=True)
            except Exception:
                log.error("Post-task promotion deferred: cost accounting unavailable", exc_info=True)
                return False
            if remaining < budget_floor:
                _safe_unlink(path)
                return False
        if bool(req.get("requires_plan_review", True)):
            objective += _PLAN_REVIEW_SUFFIX
        origin = consciousness_origin_metadata(req)
        if not start_evolution_campaign(objective, source="post_task", **({"origin": origin} if origin else {})):
            return False
        # The chooser's stated backlog entry rides the campaign into its cycle-outcome
        # history as evidence of what it chose from: it is not checked against, and
        # never changes, the authored note, whose author alone revises it.
        backlog_id = str(req.get("backlog_id") or "").strip()
        if backlog_id:
            try:
                from supervisor.evolution_lifecycle import _read_evolution_campaign, _write_evolution_campaign

                camp = _read_evolution_campaign()
                camp["post_task_backlog_id"] = backlog_id
                _write_evolution_campaign(camp)
            except Exception:
                pass
        from supervisor.state import update_state

        def _activate_one_shot(live: dict) -> None:
            # Atomic re-check against a RACED owner stop: the earlier load_state()
            # snapshot can be stale if an owner /evolve off / panic set the sentinel
            # after it (panic can fire from another thread — the toggle/`/evolve` paths
            # are already serialized ahead of this on the supervisor loop). Honor the
            # LIVE flag inside the atomic update so evolution is never enabled against a
            # fresh owner stop, even in that window.
            known, stopped = control_value(live, "evolution_owner_stopped")
            if not known or stopped:
                return
            live["evolution_mode_enabled"] = True
            live["evolution_consecutive_failures"] = 0
            live["post_task_autostop"] = True

        from supervisor.state import StateUnavailable

        try:
            st_after = update_state(_activate_one_shot)
        except StateUnavailable:
            st_after = None  # the request stays durable for a later tick; the minted campaign closes below
        else:
            _safe_unlink(path)
        if st_after is None or not bool(st_after.get("evolution_mode_enabled")):
            # Owner stop won the race: the atomic re-check refused the enable. Terminal-
            # close the campaign this now-stale path minted so no dangling active campaign
            # survives, and do NOT audit a self-enable that did not happen. The durable
            # sentinel keeps evolution off until an owner /evolve start.
            try:
                from supervisor.evolution_lifecycle import complete_evolution_campaign
                complete_evolution_campaign("owner stop raced post-task enable", status="stopped")
            except Exception:
                pass
            return False
        # Audit: this is the ONLY path that flips evolution_mode_enabled True without an
        # owner /evolve start. Record the cause + campaign id so any autonomous self-enable
        # is observable (the other True-writer is the owner /evolve handler).
        try:
            from supervisor.evolution_lifecycle import _read_evolution_campaign as _rc
            _cid = str((_rc() or {}).get("id") or "")
        except Exception:
            _cid = ""
        log.info(
            "post_task_evolution: AUTONOMOUS self-enable (source=post_task, campaign=%s, "
            "objective=%r) — evolution_mode_enabled flipped True with no /evolve start",
            _cid, objective[:120],
        )
        return True
    except Exception:
        log.debug("post_task_evolution.apply_pending_request failed", exc_info=True)
        return False
