"""Task completion facts frozen in the Main mirror and projected from recorded room history."""
from __future__ import annotations

from ouroboros.chat_references import origin_reply
from ouroboros.cost_projection import carry_cost_meta
from ouroboros.outcomes import _merge_objective_warning, normalize_outcome_axes
from ouroboros.task_finalization import schedule_report_run


def completion_facts(root, task_id: str, task: dict, result: dict, event: dict) -> dict:
    """Freeze these facts with the owed delivery, before live or history projection; ``place_origin`` too, the one its
    Main portal has once the census lists the work no more (``subagent_messages.place_origin``, by value)."""
    from ouroboros.subagent_messages import place_origin

    axes = {key: value for source in (event, result)
            for key, value in (source.get("outcome_axes") or {}).items() if isinstance(value, dict)}
    facts = {"outcome_axes": normalize_outcome_axes({**event, **result, "outcome_axes": axes})}
    metadata = {**task, **result}.get("metadata")
    if isinstance(metadata, dict) and (origin := place_origin(metadata)):
        facts["place_origin"] = origin
    reply = origin_reply(root, task_id, {**task, **result}, chat_id=1, binding_fallback=True)
    if reply:
        facts["reply_to"] = reply
    if isinstance(result.get("wake_summary"), str) and result["wake_summary"].strip():
        facts["wake_summary"] = result["wake_summary"]  # the root's own closing sentence, for its Main portal (DESIGN)
    return facts


def recorded_attempt(row):
    from ouroboros.terminal_time import execution_owner_attempt

    terminal = row.get("terminal_time")
    witness = terminal.get("attempt") if isinstance(terminal, dict) else None
    nested = row.get("task_outcome")
    if nested is None and row.get("type") == "task_summary" and not (row.get("status") or row.get("task_terminal_status")):
        nested = row.get("current_outcome")  # a saved summary whose only ending was nested
    for source in (row, witness, nested):
        if isinstance(source, dict):
            if source.get("task_id", row.get("task_id")) != row.get("task_id"):
                continue
            for key in ("task_attempt", "_attempt"):
                if type(source.get(key)) is int and source[key] >= 0:
                    return source[key]
    return execution_owner_attempt(row.get("task_id"), row.get("execution_owner"))


def matches_result_attempt(row, attempt):
    recorded = recorded_attempt(row)
    if row.get("summary_kind") == "host_task_facts":
        # A retained counter row is physical evidence, not an unbound current card.
        # The legacy path is still valid when neither side names an attempt.
        return recorded == attempt
    if row.get("executor_observation") or (recorded is not None and attempt is not None):
        return recorded is not None and recorded == attempt
    return True  # a legacy unnumbered carrier can still display the task's result


def matches_history_attempt(row, attempt):
    """Only the same attempt, or a wholly unnumbered legacy pair, shares a publication's facts."""
    recorded = recorded_attempt(row)
    if row.get("system_type") == "task_summary" and (attempt is not None or recorded is not None):
        return recorded == attempt
    return matches_result_attempt(row, attempt)


def projected_delivery(row):
    """The source warning retained with this attempt's Project publication."""
    from ouroboros.terminal_time import task_attempt_witness

    marker = row.get("canonical_terminal_projection")
    if isinstance(marker, dict) and marker.get("attempt") == task_attempt_witness(row):
        return marker.get("terminal_delivery")
    return None


def _page_delivery(carriers, result):
    """The latest saved disposition belonging to this attempt, including a receipt on a different page row."""
    attempt, latest = recorded_attempt(result), None
    for row in carriers:
        for fact in (row, row.get("task_outcome"), row.get("current_outcome")):
            if not isinstance(fact, dict) or not isinstance(fact.get("terminal_delivery"), dict):
                continue
            source_attempt = recorded_attempt(fact)
            if source_attempt is None:
                source_attempt = recorded_attempt(row)
            when = str(fact.get("revision") or row.get("ts") or "")
            if source_attempt == attempt and (latest is None or when >= latest[0]):
                latest = (when, {**fact["terminal_delivery"], "revision": fact.get("revision")})
    return latest[1] if latest else None


def _separate_current_outcome(current, result, carriers):
    """Current presentation, without inheriting fields from the historical carrier."""
    from ouroboros.task_finalization import terminal_host_notice_text
    from ouroboros.text_sentences import first_sentence

    labels = recorded_ending_labels(result)
    if "wake_summary" in labels:
        labels["wake_summary"] = first_sentence(labels["wake_summary"])
    view = {**labels, **current}
    rounds = ((result.get("loop_outcome") or {}).get("usage") or {}).get("total_rounds")
    if type(rounds) is int and rounds >= 0:
        view["rounds"] = rounds
    solve = result.get("model_execution")
    if isinstance(solve, dict):
        view["model_execution"] = dict(solve)
    if ("wake_silent" in result or result.get("owner_message_sent") or terminal_host_notice_text(result)
            or not isinstance(solve, dict) or solve.get("source") != "usable_solve_response" or not solve.get("llm_call_id")):
        return view
    # The same usable solve can corroborate a legacy publication's quiet flag, not its attempt or other facts.
    flags = {old["quiet_effective"] for row in carriers if (old := row.get("historical_terminal"))
             and recorded_attempt(old) is None and type(old.get("quiet_effective")) is bool
             and (old.get("status"), old.get("phase")) == (current.get("status"), current.get("outcome_phase"))
             and isinstance(source := old.get("model_execution"), dict)
             and (source.get("source"), source.get("llm_call_id")) == (solve["source"], solve["llm_call_id"])}
    if flags == {True}:
        view["quiet_effective"] = True
    return view


def attach_current_outcomes(carriers, result, drive_root, *, offer, open_phase=None):
    """Join one task's page carriers to its current result. Each receipt keeps its own attempt and delivery."""
    from ouroboros.late_review_marks import current_card_outcome, late_current_outcome
    from ouroboros.task_status import FINAL_STATUSES

    if open_phase:
        for carrier in carriers:
            if carrier.get("historical_terminal"):
                carrier.update(task_phase=open_phase, outcome_final=False)
        return {}
    if not carriers or result.get("status") not in FINAL_STATUSES:
        return {}
    attempt = recorded_attempt(result)
    delivery = result.get("terminal_delivery") or _page_delivery(carriers, result) or projected_delivery(result)
    if delivery:
        result = {**result, "terminal_delivery": delivery}
    current = late_current_outcome(drive_root, result)
    late = current is not None
    if not late and isinstance(delivery, dict):
        current = current_card_outcome(drive_root, result)
    separate = None
    for carrier in carriers:
        recorded = carrier.get("historical_terminal") or carrier
        nested = carrier.get("current_outcome") or carrier.get("task_outcome")
        nested_attempt = recorded_attempt(nested) if isinstance(nested, dict) else None
        status = nested
        if isinstance(nested, dict):
            status = nested.get("task_terminal_status") or nested.get("status")
        if not status:
            status = recorded.get("status") if carrier.get("system_type") == "task_summary" else None
        publication_attempt = recorded_attempt(carrier)
        different = (attempt is not None or publication_attempt is not None or nested_attempt is not None) and (
            not matches_history_attempt(carrier, attempt) or nested_attempt not in (None, attempt)
        )
        revised = (nested and (delivery or result.get("reason_code") or result.get("outcome_axes"))
                   or current is not None and carrier.get("terminal_delivery")
                   and carrier["terminal_delivery"] != current.get("terminal_delivery"))
        if not (late or different or revised or (status and status != result["status"])):
            continue
        if current is None:
            current = current_card_outcome(drive_root, result)
        if (isinstance(carrier.get("current_outcome"), dict) and "task_outcome" not in carrier
                and (different or nested.get("terminal_delivery"))
                and (nested_attempt is None or publication_attempt in (None, nested_attempt))):
            carrier["task_outcome"] = nested
        if different and separate is None:
            separate = {**_separate_current_outcome(current, result, carriers), "continuation_offer": offer}
        carrier["current_outcome"] = dict(separate if different else current)
    return current or {}


def same_attempt_ending(result: dict, current: dict, offer, *, final: bool) -> dict:
    """What a row of the stored result's own attempt, or of a wholly unnumbered legacy pair, shows of that result.

    ``current`` is ``attach_current_outcomes``'s answer for the page: a delivery warning there revises the
    axes and the phase. ``final`` is False while post-task finalization is open; a final row also carries the
    attempt's proven end (``terminal_time``), so a row shown before its ending row keeps the clock. A row of
    another attempt takes none of this; it carries the result separately as ``current_outcome``.
    """
    from ouroboros.history_retention import retention_summary
    from ouroboros.project_dialogue import outcome_phase
    from ouroboros.subagent_messages import initiator_meta
    from ouroboros.terminal_time import terminal_time_fact

    axes = normalize_outcome_axes(result)
    phase = outcome_phase(result, {})
    if delivery := current.get("terminal_delivery"):
        _merge_objective_warning(axes["objective"], f"terminal_delivery_{delivery['state']}")
        phase = current["outcome_phase"]
    ending = {
        "outcome_axes": axes, "_is_direct_chat": bool(result.get("_is_direct_chat")),
        "outcome_phase": phase, "outcome_final": final,
        **({"terminal_delivery": delivery} if delivery else {}),
        **initiator_meta(result)}  # + the origin label, from the persisted metadata
    for key in ("model_execution", "cancel_origin", "review_projection", "review_overrides", "finalization_recovery"):
        if isinstance(result.get(key), dict):
            ending[key] = dict(result[key])
    if isinstance(result.get("body_adoptions"), list):  # its settled adoptions, read beside its Reviews
        ending["body_adoptions"] = [dict(row) for row in result["body_adoptions"] if isinstance(row, dict)]
    ending.update({key: str(result[key]) for key in ("wake_summary", "schedule_name", "terminal_origin") if result.get(key)})
    if schedule_report_run(result):
        ending.pop("wake_summary", None)
        ending["schedule_report_only"] = True
    if isinstance(result.get("delegated_runs_unreconciled"), list):  # the current debt (#1583)
        ending["delegated_runs_unreconciled"] = list(result["delegated_runs_unreconciled"])
    retention = retention_summary(result)
    rounds = ((result.get("loop_outcome") or {}).get("usage") or {}).get("total_rounds")
    if type(rounds) is int and rounds >= 0:
        ending["rounds"] = rounds
    if final and (ended := terminal_time_fact(result))["source"] != "unknown":
        ending["terminal_time"] = ended  # observed for this attempt; an unknown end is never shown as one
    if retention:
        ending["history_retention"] = retention
    if result.get("reason_code"):
        ending["reason_code"] = str(result.get("reason_code") or "")
    ending["continuation_offer"] = offer
    ending.update(carry_cost_meta(result))  # ABI-3 fields override the row fallback
    return ending


def recorded_ending_labels(row: dict) -> dict:
    """The names, place and quiet flags that survive on both a summary and its compact replay row."""
    report_only = row.get("schedule_report_only") is True or schedule_report_run(row)
    labels = {key: str(row[key]) for key in ("wake_summary", "suggested_name", "schedule_name")
              if row.get(key) and (key != "wake_summary" or not report_only)}
    if report_only:
        labels["schedule_report_only"] = True
    if isinstance(row.get("place_origin"), dict):
        labels["place_origin"] = dict(row["place_origin"])
    for key in ("wake_silent", "quiet_effective", "schedule_report_only"):
        if type(row.get(key)) is bool:
            labels[key] = row[key]
    return labels


def recorded_ending_details(row: dict) -> dict:
    """Only the publication owns its explanation. Its small lifecycle stamp can also ride on narration."""
    keys = ("reason_code", "terminal_origin", "outcome_axes", "finalization_recovery", "cancel_origin",
            "delegated_runs_unreconciled", "review_projection")
    details = {key: row[key] for key in keys if key in row}
    if type(row.get("_result_text_present")) is bool:
        details["_result_text_present"] = row["_result_text_present"]
    return details


def remember_terminal(latest: dict, row: dict) -> dict:
    """The compact receipt of this publication and the newest such receipt represented on this page."""
    from ouroboros.project_dialogue import historical_terminal_projection

    frozen = historical_terminal_projection(row)
    if not frozen:
        return {}
    facts = recorded_ending_labels(row)
    for key in ("terminal_time", "terminal_delivery"):
        if isinstance(row.get(key), dict):
            facts[key] = dict(row[key])
    attempt = recorded_attempt(row)
    if attempt is not None:
        facts["task_attempt"] = attempt
    frozen.update(facts)
    task_id = str(row["task_id"])
    previous = latest.get(task_id, {})
    prior = previous.get("task_attempt")
    newer = frozen["ts"] >= previous.get("ts", "")
    if attempt is not None and prior is not None and attempt != prior:
        newer = attempt > prior
    if newer:
        latest[task_id] = frozen
    receipt = {**facts, "historical_terminal": frozen}
    # Roots already have their published phase and time. A nested card can have only
    # this receipt to explain its ending; a recovered answer also needs its custody facts.
    if row.get("summary_kind") == "terminal_result_projection" or row.get("finalization_recovery"):
        receipt.update(recorded_ending_details(row))
    _copy_recorded_outcomes(receipt, row)
    return receipt


def _copy_recorded_outcomes(rec: dict, entry: dict) -> None:
    """Some summaries carry their ending as a nested record. Only the facts the card reads travel with it."""
    for field in ("task_outcome", "current_outcome"):
        outcome = entry.get(field)
        if isinstance(outcome, str) and outcome and field == "task_outcome":
            rec[field] = outcome
        elif isinstance(outcome, dict) and outcome.get("task_id", entry.get("task_id")) == entry.get("task_id"):
            facts = recorded_ending_details(outcome)
            facts.update({k: outcome[k] for k in ("status", "task_terminal_status", "outcome_phase", "outcome_final",
                          "task_completion", "artifact_status", "terminal_delivery", "revision") if k in outcome})
            if (isinstance(outcome.get("terminal_time"), dict)
                    and outcome["terminal_time"].get("source") in {"executor_terminal", "supervisor_cancel"}):
                facts["terminal_time"] = dict(outcome["terminal_time"])
            for key, part in (("artifact_bundle", "status"), ("review_status", "status"), ("root_phase_checkpoint", "post_task_synthesis")):
                if isinstance(outcome.get(key), dict) and part in outcome[key]:
                    facts[key] = {part: outcome[key][part]}
            facts.update(recorded_ending_labels(outcome))
            if (attempt := recorded_attempt(outcome)) is not None:
                facts["task_attempt"] = attempt
            if facts:
                rec[field] = facts


def copy_task_summary_metadata(rec: dict, entry: dict) -> None:
    """Copy terminal chat facts for task summaries."""
    if entry.get("type") != "task_summary":
        return
    _copy_recorded_outcomes(rec, entry)
    rec.update(recorded_ending_labels(entry))
    for key in ("model_execution", "review_projection", "history_retention", "terminal_time", "cancel_origin",
                "terminal_delivery", "finalization_recovery"):
        if isinstance(entry.get(key), dict):
            rec[key] = dict(entry[key])
    for key in ("summary_kind", "target_label", "terminal_origin", "status"):
        if entry.get(key): rec[key] = str(entry[key])
    for key in ("tool_calls", "rounds", "tool_errors", "routing_tool_calls", "completion_tool_calls"):
        if key in entry:
            rec[key] = None if entry[key] is None else int(entry[key])
    if "tool_call_counts" in entry:
        rec["tool_call_counts"] = dict(entry["tool_call_counts"]) if isinstance(entry["tool_call_counts"], dict) else None
    # The row's own direct-turn fact (pruned-result fallback; the persisted
    # result overrides it) and the typed routing action as `addressing_only`.
    if "_is_direct_chat" in entry:
        rec["_is_direct_chat"] = bool(entry["_is_direct_chat"])
    if entry.get("typed_routing_action"):
        rec["addressing_only"] = str(entry["typed_routing_action"])
    # A summary can predate axes. Its lifecycle alone records no execution, file or review verdict.
    axes = entry.get("outcome_axes")
    recorded_axis = isinstance(axes, dict) and any(
        name != "lifecycle" and isinstance(value, dict) and value for name, value in axes.items()
    )
    bundle = entry.get("artifact_bundle")
    artifact_status = bundle.get("status") if isinstance(bundle, dict) else None
    if entry.get("result_status") or recorded_axis or artifact_status or entry.get("artifact_status"):
        rec["outcome_axes"] = normalize_outcome_axes(entry)
    elif isinstance(axes, dict) and axes:
        rec["outcome_axes"] = dict(axes)
    if "reason_code" in entry:
        rec["reason_code"] = str(entry.get("reason_code") or "")
    # Row cost is a fallback; persisted result truth overrides it. ABI-3
    # converts legacy pairs deprecated-wins, exposing only honest names.
    rec.update(carry_cost_meta(entry))
    # Live-card axes/origin are row fallbacks; persisted task_results override them.
    rec.update({key: entry[key] for key in ("outcome_phase", "outcome_final", "initiator", "continue_requester") if key in entry})
