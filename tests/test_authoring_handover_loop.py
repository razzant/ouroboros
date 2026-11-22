"""Authoring handovers exercised through the real loop, dispatcher and finalizer.

Only model responses are scripted. Tool calls read a real isolated file; fallback
selection, wait re-preparation, transcript custody and final admission stay real.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import loop, model_wait
from ouroboros.llm import LLMClient
from ouroboros.outcomes import derive_loop_outcome
from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, KIND_OWNER_MODEL, write_owner_message
from ouroboros.tools.registry import ToolRegistry
from tests.test_completion_selection import finish

PRIMARY = "openai/model-a"
SUCCESSOR = "anthropic/model-b"
TASK = "handover-loop"
SCHEDULED_REPORT = {"schedule_occurrence": {"schedule_id": "handover-report", "token": "occurrence-1"}}


def _text(content):
    return {"role": "assistant", "content": content}


def _tool(name="read_file", **args):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "read-" + str(args.get("path", name)), "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }]}


@pytest.fixture
def episode(tmp_path, monkeypatch, provider_catalog_offline):
    monkeypatch.setenv("OUROBOROS_MODEL", PRIMARY)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", SUCCESSOR)
    monkeypatch.setenv("USE_LOCAL_MAIN", "false")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "false")
    monkeypatch.setenv("OUROBOROS_REASONING_SUMMARY", "off")
    from ouroboros import fallback_cooldown
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    (tmp_path / "evidence.txt").write_text("Verified predecessor evidence\n", encoding="utf-8")
    from ouroboros import context
    from tests.test_context_fit_integration import _plan
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: (
        {"model": task["model"], "provider": "openai"},
        SimpleNamespace(route_fp="test-route-" + task["model"], status="confirmed",
                        stale=False, window_tokens=900_000, source="advertised"),
    ))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.context_fit_plan = replace(_plan(preferred="max", window=900_000), model=PRIMARY)
    state = SimpleNamespace(root=tmp_path, tools=registry, calls=[], notes=[], steps=[], task_id=TASK)

    def scripted(_llm, messages, model, _schemas, _effort, _retries, _logs,
                 _task_id, _round, _events, usage, *_args, **kwargs):
        state.calls.append({"model": model, "messages": deepcopy(messages)})
        assert state.steps, "Unexpected model call after the scripted terminal response"
        step = state.steps.pop(0)
        if callable(step):
            step = step(state, usage, kwargs)
        if step is None:
            usage["_last_llm_error_kind"] = "invalid_continuation"
            return None, 0.0
        usage.pop("_last_llm_error_kind", None)
        return deepcopy(step), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", scripted)

    def run(steps, *, task_id=TASK):
        from ouroboros.task_results import write_task_result
        tools = state.tools
        write_task_result(tmp_path, task_id, "running", root_task_id=task_id,
                          _attempt=getattr(tools._ctx, "task_attempt", None) or 1)
        state.steps = list(steps)
        state.task_id = task_id
        state.result = loop.run_llm_loop(
            messages=tools._ctx.context_fit_plan.messages_for("max"),
            tools=tools, llm=LLMClient(),
            drive_logs=tmp_path / "logs", drive_root=tmp_path,
            emit_progress=lambda text, **_meta: state.notes.append(text),
            incoming_messages=queue.Queue(), task_id=task_id,
        )
        assert not state.steps, "The loop finalized before consuming the expected continuation"
        return state.result

    state.run = run
    return state


def _read():
    return _tool(path="evidence.txt")


def _one_response(*messages):
    """One reply carrying every tool call of ``messages``, each under its own call ID."""
    calls = [call for message in messages for call in message["tool_calls"]]
    return {"role": "assistant", "content": None,
            "tool_calls": [{**call, "id": f"{call['id']}-{number}"} for number, call in enumerate(calls, 1)]}


def _handover_notes(trace):
    return [note for note in trace["reasoning_notes"] if "handover recovery" in note.lower()]


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_selection(text, usage, trace, selection):
    """The successor's own selection ended the handover: no warning of its own, and the history says how."""
    assert usage.get("reason_code") != "authoring_handover_incomplete"
    assert "authoring_handover_incomplete" not in trace
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    row = trace["route_handovers"][0]
    assert (row["status"], row["selection"], row["recovery_prompted"]) == ("selected", selection, True)
    assert row["reply_observed"] is True
    assert usage["authoring_handovers"] == [row]  # the history the task result keeps
    return row


@pytest.mark.parametrize("first_status", ["The worktree is clean.", "Finished successfully.", "Работа ещё продолжается."])
def test_successor_selection_after_the_one_reminder_delivers_without_a_handover_warning(episode, first_status):
    text, usage, trace = episode.run([_read(), None, _text(first_status), finish("No additional work done.")])
    assert [call["model"] for call in episode.calls] == [PRIMARY, PRIMARY, SUCCESSOR, SUCCESSOR]
    assert [row["tool"] for row in trace["tool_calls"]] == ["read_file", "finish_task"]
    assert trace["tool_calls"][0]["is_error"] is False
    assert any(row.get("role") == "tool" and "Verified predecessor evidence" in row.get("content", "")
               for row in episode.calls[2]["messages"])
    assert len(_handover_notes(trace)) == 1
    seen = str(episode.calls[3]["messages"])  # the reminder and the answer it can select
    assert "model handover" in seen and f"Retained answer_sha256={_sha(first_status)}" in seen
    assert text == "No additional work done."
    row = _clean_selection(text, usage, trace, "answer")
    assert (row["from_model"], row["to_model"], row["tool_calls_at_handover"]) == (PRIMARY, SUCCESSOR, 1)


@pytest.mark.parametrize("physical_author", [None, "openai/model-d"])
def test_successor_confirms_the_retained_answer_by_its_hash(episode, physical_author):
    status = "The predecessor's checks passed; the report stands as written."

    def reply(message):
        def on_route(_state, usage, _kwargs):
            if physical_author:
                usage["_last_llm_call_meta"] = {"model": physical_author}
            return message
        return on_route

    text, usage, trace = episode.run([_read(), None, reply(_text(status)), reply(finish(answer_sha256=_sha(status)))])
    assert text == status
    assert len(_handover_notes(trace)) == 1
    assert _clean_selection(text, usage, trace, "answer_sha256")["to_model"] == (physical_author or SUCCESSOR)


def _interrupt_handover(episode, monkeypatch, boundary, answer, *, steps=None):
    from ouroboros import working_checkpoint

    save = working_checkpoint.save_or_log

    def interrupt_after_save(limit, point):
        saved = save(limit, point)
        current = getattr(limit.tools._ctx, "_authoring_handover", None)
        selected = getattr(limit.tools._ctx, "_completion_selected", None)
        if not saved:
            return saved
        if boundary == "recovered_candidate":
            rows = limit.accumulated_usage.get("authoring_handovers") or []
            wanted = point == "candidate" and rows and rows[-1]["status"] == "recovered"
        elif boundary == "next_ready":  # the round after the last scripted reply, before its model call
            wanted = point == "ready" and not episode.steps
        elif not current:
            wanted = point == boundary == "post_batch" and not episode.steps
        elif boundary == "selected_candidate":  # the last scripted selection: an earlier one may meet the reminder
            wanted = point == "candidate" and selected and not episode.steps
        elif boundary == "staged_post_batch":
            wanted = (point == "post_batch" and getattr(limit.tools._ctx, "_completion_request", None)
                      and not episode.steps)
        else:
            wanted = point == boundary and current["recovery_prompted"] == (point == "ready")
        if wanted:
            raise RuntimeError("test interruption after checkpoint")
        return saved

    with monkeypatch.context() as interrupted:
        interrupted.setattr(working_checkpoint, "save_or_log", interrupt_after_save)
        with pytest.raises(RuntimeError, match="test interruption after checkpoint"):
            if steps is None:
                steps = [_read(), None, _text(answer)]
                if boundary == "selected_candidate":
                    steps.append(finish(answer_sha256=_sha(answer)))
            episode.run(steps)


def _resume_handover(episode, plan, task_id=TASK):
    from ouroboros import working_checkpoint

    episode.tools = ToolRegistry(repo_dir=episode.root, drive_root=episode.root)
    ctx = episode.tools._ctx
    ctx.task_id, ctx.task_attempt, ctx.context_fit_plan = task_id, 2, plan
    ctx.working_recovery = working_checkpoint.prepare_recovery(episode.root, TASK, from_attempt=1, cause="restart")
    return ctx.working_recovery


def _legacy_checkpoint(episode, monkeypatch, boundary, *replies, prompted=False, awaited=False):
    from ouroboros import loop_nudges, working_checkpoint
    from ouroboros.loop_llm_call import call_llm_with_retry

    def transport(_client, **kwargs):
        episode.calls.append({key: deepcopy(kwargs[key]) for key in ("model", "messages")})
        step = episode.steps.pop(0)
        return deepcopy(step() if callable(step) else step), {
            "cost": 0.0, "provider": "openrouter", "prompt_tokens": 1, "completion_tokens": 1}

    def change_route():
        monkeypatch.setenv("OUROBOROS_MODEL", SUCCESSOR)
        return _read()

    # Script only the transport: let the real call recorder and batch trace write the proof.
    monkeypatch.setattr(loop, "call_llm_with_retry", call_llm_with_retry)
    monkeypatch.setattr(LLMClient, "chat", transport)
    episode.tools._ctx.task_attempt = 1
    with monkeypatch.context() as old:
        # The previous body checked for work only at finalization, not at the batch end.
        old.setattr(loop_nudges, "observe_handover_work", lambda *_: None)
        _interrupt_handover(episode, old, boundary, None, steps=[change_route, *replies])
    saved = working_checkpoint.read_working(episode.root, TASK, 1)[1]
    if awaited:  # the previous body wrote a settings change's pair at the switch, before its target answered
        assert saved["route"].pop("_pending_handover_reason") == "settings_change"
        pair = {"from_model": PRIMARY, "to_model": SUCCESSOR, "reason": "settings_change",
                "tool_calls_at_handover": 1, "recovery_prompted": False, "status": "pending"}
        saved["usage"]["authoring_handovers"], saved["trace"]["route_handovers"] = [dict(pair)], [dict(pair)]
    for rows in (saved["usage"]["authoring_handovers"], saved["trace"]["route_handovers"]):
        row = rows[-1]
        assert row["status"] == "pending" and row["recovery_prompted"] is prompted
        assert awaited or row.pop("reply_observed") is True
    author = PRIMARY if awaited else SUCCESSOR  # of the last reply this save holds
    assert saved["delivery"]["_completion_observation"].pop("model") == author
    receipt = saved["usage"]["_last_llm_call_meta"]
    assert receipt == saved["usage"]["llm_call_refs"][-1] and receipt["usable_solve_response"] is True
    assert receipt["model"] == author and receipt["round"] == saved["round_idx"] - (boundary == "next_ready")
    return working_checkpoint.checkpoint_path(episode.root, TASK, 1), saved


@pytest.mark.parametrize(("boundary", "same_round"), [("post_batch", True), ("post_batch", False), ("next_ready", True)],
                         ids=["proved-batch", "unbound-tool-result", "proved-batch-saved-at-the-next-ready"])
def test_legacy_save_credits_success_only_in_the_proved_successor_round(episode, monkeypatch, boundary, same_round):
    """The previous body looked for work only at finalization, so its saves keep the row pending after the
    successor's successful batch; the receipts credit it at whichever boundary the save was taken."""
    plan = episode.tools._ctx.context_fit_plan
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, _read())
    assert saved["working"]["boundary"] == boundary.removeprefix("next_")
    first, last = saved["trace"]["tool_calls"]
    assert last["round_id"] == saved["usage"]["_last_llm_call_meta"]["round_id"]
    assert last["is_error"] is False and first["round_id"] != last["round_id"]
    if not same_round:
        last["round_id"] = first["round_id"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    answer = "Verified file report."
    text, usage, trace = episode.run([_text(answer)] if same_round else [_text("Checking."), finish(answer)])
    assert text == answer
    assert [call["tool"] for call in trace["tool_calls"] if call["tool"] == "read_file"] == ["read_file"] * 2
    [row] = usage["authoring_handovers"]
    if same_round:
        assert (row["status"], row["work_observed"], row["reply_observed"]) == ("recovered", True, True)
        assert len(episode.calls) == 3 and not _handover_notes(trace)
    else:
        assert len(_handover_notes(trace)) == 1
        _clean_selection(text, usage, trace, "answer")


@pytest.mark.parametrize("proof", ["matching", "refused-round", "unsent-round", "unrecorded-round", "unstamped",
                                   "other-execution", "other-attempt", "other-round", "other-model"])
def test_legacy_candidate_asks_once_without_inventing_an_author(episode, monkeypatch, proof):
    plan, draft = episode.tools._ctx.context_fit_plan, "Still checking the file."
    path, saved = _legacy_checkpoint(episode, monkeypatch, "candidate", _text(draft))
    assert saved["delivery_candidate"]["full_text"] == draft
    assert not saved["delivery_candidate"]["control_episode_seen"]
    usage, proved = saved["usage"], proof in {"matching", "refused-round", "unsent-round"}
    if proof == "unstamped":  # absent facts: no receipt says which reply was usable
        for receipt in (usage["_last_llm_call_meta"], *usage["llm_call_refs"]):
            receipt.pop("usable_solve_response")
    elif proof == "other-attempt":
        usage["_task_attempt"] += 1
    elif proof in {"refused-round", "unsent-round", "unrecorded-round"}:
        # The successor answered one logical round later: the round between ended without a usable reply
        # (a view refused for its size, then renewed).
        reply, saved["round_idx"] = usage["llm_call_refs"][-1], 3
        for receipt in (reply, usage["_last_llm_call_meta"]):
            receipt.update(round=3, round_id=f"{usage['execution_id']}:round:3")
        routes = usage["accepted_routes"]  # the dispatcher's own record of who answered, apart from the receipts
        assert [row["to_binding"][0] for row in routes] == [PRIMARY, SUCCESSOR]
        if proof == "refused-round":  # the provider refused the request: the old body recorded a failed attempt
            usage["llm_call_refs"].insert(-1, {
                "ts": reply["ts"], "llm_call_id": "llm_refused", "round": 2, "model": PRIMARY, "use_local": False,
                "failure_code": "context_overflow", "error": "context window exceeded", "status_code": 400})
        elif proof == "unrecorded-round":  # another route answered there and its receipt is missing
            routes.insert(-1, {**routes[-1], "to_binding": ["openai/model-c", False, ""]})
        # "unsent-round": the host refused the view before sending, which leaves no receipt and changes no route
    elif proof != "matching":
        prior = usage["llm_call_refs"][0]
        keys = {"other-execution": {"execution_id": "other", "round_id": f"other:round:{saved['round_idx']}"},
                "other-round": {key: prior[key] for key in ("round", "round_id")},
                "other-model": {"model": prior["model"]}}[proof]
        usage["_last_llm_call_meta"].update(keys)
        usage["llm_call_refs"][-1].update(keys)
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    confirmed = "File checked."

    before_choice = []

    def choose():
        ctx = episode.tools._ctx
        row, = ctx._execution_trace["route_handovers"]
        before_choice.append((row.copy(), ctx._authoring_handover is None))
        return finish(confirmed)

    fresh = [_text("Fresh words with no choice.")] if proof == "unrecorded-round" else []
    text, usage, trace = episode.run([*fresh, choose])
    (row, unbound), = before_choice
    assert row["status"] == "pending" and row["recovery_prompted"] is True
    if not proved:
        assert unbound and "reply_observed" not in row
        assert "does not confirm its author" in str(episode.calls[-1]["messages"])
    assert text == confirmed and len(episode.calls) == 3 + len(fresh) and len(_handover_notes(trace)) == 1
    if proved:
        assert "model handover" in str(episode.calls[-1]["messages"])
        _clean_selection(text, usage, trace, "answer")
    elif proof == "unrecorded-round":
        # The receipts name the target as the last author past a round they do not account for: its fresh prose
        # after the reminder stays held, its later choice delivers, and the row claims nothing about either.
        from tests._delivery_candidate_shared import holds

        assert [(hold["reason"], hold["subreason"]) for hold in holds(episode.root)] == [
            ("finalization_nudge", "authoring_handover"), ("selection_required", None)]
        [row] = usage["authoring_handovers"]
        assert row["status"] == "pending" and "reply_observed" not in row and "selection" not in row
        assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    elif proof in {"unstamped", "other-execution", "other-attempt"}:
        # No receipt can say anything: the target's real reply after the restore is the first fact about the pair.
        _clean_selection(text, usage, trace, "answer")


@pytest.mark.parametrize("proof", ["matching", "foreign-author", "missing-usable", "foreign-result"])
def test_legacy_candidate_reconciles_work_from_an_earlier_successor_round(episode, monkeypatch, proof):
    plan, answer = episode.tools._ctx.context_fit_plan, "Verified file report."
    path, saved = _legacy_checkpoint(episode, monkeypatch, "candidate", _read(), _text(answer))
    predecessor, earlier, latest = saved["usage"]["llm_call_refs"]
    earlier_result = saved["trace"]["tool_calls"][-1]
    assert earlier_result["round_id"] == earlier["round_id"] != latest["round_id"]
    if proof == "foreign-author":
        earlier["model"] = "openai/model-c"
    elif proof == "missing-usable":
        earlier.pop("usable_solve_response")
    elif proof == "foreign-result":
        earlier_result["round_id"] = predecessor["round_id"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)
    text, usage, trace = episode.run([] if proof == "matching" else [finish(answer_sha256=_sha(answer))])
    [row] = usage["authoring_handovers"]
    assert text == answer and len(episode.calls) == (3 if proof == "matching" else 4)
    assert len(_handover_notes(trace)) == (0 if proof == "matching" else 1)
    assert (row["status"] == "recovered") == (proof == "matching")


@pytest.mark.parametrize("boundary", ["staged_post_batch", "selected_candidate"])
@pytest.mark.parametrize("proof", ["matching", "foreign-result", "different-author"])
def test_legacy_completion_copies_get_only_their_own_round_author(episode, monkeypatch, boundary, proof):
    plan, answer = episode.tools._ctx.context_fit_plan, "Report selected by its digest."
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, _text(answer),
                                     finish(answer_sha256=_sha(answer)), prompted=True)
    field = "_completion_request" if boundary == "staged_post_batch" else "_completion_selected"
    observation = saved["delivery"][field]["observation"]
    assert observation.pop("model") == SUCCESSOR
    control = saved["trace"]["tool_calls"][-1]
    assert control["completion_control"] is True and control["round_id"] == saved["usage"]["_last_llm_call_meta"]["round_id"]
    if proof == "foreign-result":
        control["round_id"] = saved["usage"]["llm_call_refs"][-2]["round_id"]
    elif proof == "different-author":
        observation["model"] = "openai/model-c"
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)
    text, usage, trace = episode.run([])
    assert text == answer and len(episode.calls) == 3 and len(_handover_notes(trace)) == 1
    if proof == "matching":
        _clean_selection(text, usage, trace, "answer_sha256")
    else:
        [row] = usage["authoring_handovers"]
        assert row["status"] == "pending" and "selection" not in row


def _legacy_selection(saved, boundary):
    """The saved copy of the successor's selection, as the previous body wrote it: without its author."""
    observation = saved["delivery"]["_completion_request" if boundary == "staged_post_batch"
                                    else "_completion_selected"]["observation"]
    assert observation.pop("model") == SUCCESSOR
    return saved["usage"], saved["trace"]


def _pending_without_a_choice(usage):
    row = usage["authoring_handovers"][-1]
    assert row["status"] == "pending" and "selection" not in row and "reply_observed" not in row


REVISITED = "Report confirmed after the route came back."
REVISITED_ROWS = [(PRIMARY, SUCCESSOR, 1), (SUCCESSOR, PRIMARY, 1), (PRIMARY, SUCCESSOR, 1)]


def _revisit(monkeypatch, *selections, scripted=False):
    """One tool baseline, A→B→A→B: A reads, B and then A comment, and B answers with ``selections``."""
    def then_route(model, message):
        def reply(*_scripted):
            monkeypatch.setenv("OUROBOROS_MODEL", model)
            return message
        return reply

    return [*([then_route(SUCCESSOR, _read())] if scripted else []),  # the legacy save scripts its own first read
            then_route(PRIMARY, _text("Reading what the first model found.")),
            then_route(SUCCESSOR, _text("The first model again: nothing to add.")), *selections]


def _revisited_legacy_rows(saved):
    """Every switch after the tools kept its own row at the same tool count; none says its target replied."""
    for rows in (saved["usage"]["authoring_handovers"], saved["trace"]["route_handovers"]):
        assert [(row["from_model"], row["to_model"], row["tool_calls_at_handover"]) for row in rows] == REVISITED_ROWS
        assert all(row["status"] == "pending" for row in rows)
        for row in rows[:-1]:
            assert row.pop("reply_observed") is True
        yield rows


def _final_transition_selected(text, usage, trace):
    assert text == REVISITED and len(_handover_notes(trace)) == 2  # the first row's reminder and the last row's
    assert usage.get("reason_code") != "authoring_handover_incomplete" and "authoring_handover_incomplete" not in trace
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    first, returned, final = usage["authoring_handovers"]
    assert (final["status"], final["selection"], final["reply_observed"]) == ("selected", "answer_sha256", True)
    assert trace["route_handovers"][-1] is final
    for earlier in (first, returned):  # history: only the final transition is settled
        assert earlier["status"] == "pending" and "selection" not in earlier
    return first, returned


def test_a_revisited_successor_selects_after_its_own_rows_reminder(episode, monkeypatch):
    """The uninterrupted reference: the route's return opens a new row, whose target meets that row's reminder."""
    text, usage, trace = episode.run(_revisit(
        monkeypatch, finish(REVISITED), finish(answer_sha256=_sha(REVISITED)), scripted=True))
    assert [call["model"] for call in episode.calls] == [PRIMARY, SUCCESSOR, PRIMARY, SUCCESSOR, SUCCESSOR]
    rows = usage["authoring_handovers"]
    assert [(row["from_model"], row["to_model"], row["tool_calls_at_handover"]) for row in rows] == REVISITED_ROWS
    _final_transition_selected(text, usage, trace)


@pytest.mark.parametrize("boundary", ["staged_post_batch", "selected_candidate"])
@pytest.mark.parametrize("proof", ["matching", "unrowed-return", "misnamed-return", "unbound-return", "foreign-return",
                                   "missing-return", "other-attempt"])
def test_legacy_selection_after_a_revisited_author_settles_only_the_final_transition(episode, monkeypatch, boundary, proof):
    """Every switch kept its own row and every reply its receipt, so the last row is the last departure into its
    target, whose selection after that row's one reminder is its own."""
    plan = episode.tools._ctx.context_fit_plan
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, *_revisit(
        monkeypatch, finish(REVISITED), finish(answer_sha256=_sha(REVISITED))), prompted=True)
    usage, trace = _legacy_selection(saved, boundary)
    refs = usage["llm_call_refs"]
    assert [(ref["model"], ref["round"], ref["usable_solve_response"]) for ref in refs] == [
        (PRIMARY, 1, True), (SUCCESSOR, 2, True), (PRIMARY, 3, True), (SUCCESSOR, 4, True), (SUCCESSOR, 5, True)]
    assert [(call["tool"], call["round_id"], call["is_error"]) for call in trace["tool_calls"]] == [
        ("read_file", refs[0]["round_id"], False), ("finish_task", refs[3]["round_id"], False),
        ("finish_task", refs[4]["round_id"], False)]
    for rows in _revisited_legacy_rows(saved):
        if proof == "unrowed-return":  # two departures into the target and one row: which one it names is unknown
            del rows[:-1]
        elif proof == "misnamed-return":  # the rows and the receipts disagree about who answered in between
            rows[1]["to_model"] = "openai/model-c"
        elif proof == "unbound-return":  # a malformed tool count: the row cannot be placed among these changes
            rows[1]["tool_calls_at_handover"] = "1"
    if proof == "foreign-return":  # the receipts name an author the rows do not: the target came back from it
        refs[2]["model"] = "openai/model-c"
    elif proof == "missing-return":  # a round between the two series that no receipt accounts for
        refs[2].pop("usable_solve_response")
    elif proof == "other-attempt":
        usage["_task_attempt"] += 1
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([])
    assert text == REVISITED and len(episode.calls) == 5
    if proof != "matching":
        return _pending_without_a_choice(usage)
    for earlier in _final_transition_selected(text, usage, trace):
        assert "reply_observed" not in earlier


def test_legacy_first_selection_after_a_revisited_author_meets_its_own_rows_reminder(episode, monkeypatch):
    """Saved at the returned target's first selection, before its row's reminder: the restore continues as the
    uninterrupted loop does, with that one reminder and then the target's own choice."""
    from tests._delivery_candidate_shared import holds

    plan = episode.tools._ctx.context_fit_plan
    path, saved = _legacy_checkpoint(episode, monkeypatch, "staged_post_batch", *_revisit(monkeypatch, finish(REVISITED)))
    _legacy_selection(saved, "staged_post_batch")
    assert len(list(_revisited_legacy_rows(saved))) == 2
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([finish(answer_sha256=_sha(REVISITED))])
    assert len(episode.calls) == 5 and "model handover" in str(episode.calls[-1]["messages"])
    assert [(hold["reason"], hold["subreason"]) for hold in holds(episode.root)][-1] == (
        "finalization_nudge", "authoring_handover")
    _final_transition_selected(text, usage, trace)


@pytest.mark.parametrize(("boundary", "proof"), [
    *[(boundary, proof) for boundary in ("staged_post_batch", "selected_candidate")
      for proof in ("matching", "foreign-sibling", "other-command", "other-arguments", "other-request",
                    "unread-arguments", "different-author")],
    ("selected_candidate", "failed-sibling"),  # at the batch end the loop itself reads a failed command again
])
def test_legacy_selection_repeated_in_one_batch_keeps_its_author(episode, monkeypatch, boundary, proof):
    """One response may send the same accepted selection twice under two call IDs (staging admits both): the
    repeat is the same act of the same round, not a gap in the selection's provenance."""
    plan, answer = episode.tools._ctx.context_fit_plan, "Report selected twice by its digest."
    twice = _one_response(*[finish(answer_sha256=_sha(answer))] * 2)
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, _text(answer), twice, prompted=True)
    usage, trace = _legacy_selection(saved, boundary)
    read, first, second = trace["tool_calls"]
    for control in (first, second):
        assert control["completion_control"] is True and control["is_error"] is False
        assert control["round_id"] == usage["_last_llm_call_meta"]["round_id"] != read["round_id"]
    assert first["tool_call_id"] != second["tool_call_id"] and first["args"] == second["args"]

    def respell(call, **changes):
        """Repoint one recorded call's spelling wherever the save carries it: its trace projection and the
        transcript's complete arguments, the two views of one call a real save keeps together."""
        call["args"] = {**call["args"], **changes}
        for row in saved["messages"]:
            for recorded in row.get("tool_calls") or []:
                if recorded.get("id") == call.get("tool_call_id"):
                    recorded["function"]["arguments"] = json.dumps(
                        {**json.loads(recorded["function"]["arguments"]), **changes})

    if proof == "failed-sibling":  # a refused or conflicting sibling: the batch does not say one thing
        second["is_error"] = True
    elif proof == "foreign-sibling":  # a command another round sent
        second["round_id"] = read["round_id"]
    elif proof == "other-command":  # not the command the saved selection names as its source
        second["tool"] = "presence_finish"
    elif proof == "other-arguments":  # two different requests are a conflict, whatever their recorded status
        respell(second, answer_sha256=_sha("Another report."))
    elif proof == "other-request":  # the same answer under another rationale is another request to the staging
        respell(second, rationale="The reviewers have not answered.")
    elif proof == "unread-arguments":  # arguments the command's own contract refuses name no request to compare
        respell(second, outcome="message")
    elif proof == "different-author":  # the saved copy already names another author: nothing to repair
        saved["delivery"]["_completion_request" if boundary == "staged_post_batch"
                          else "_completion_selected"]["observation"]["model"] = "openai/model-c"
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([])
    assert text == answer and len(episode.calls) == 3 and len(_handover_notes(trace)) == 1
    if proof == "matching":
        _clean_selection(text, usage, trace, "answer_sha256")
    else:
        [row] = usage["authoring_handovers"]
        assert row["status"] == "pending" and "selection" not in row


SPELLED_TWICE = "Report selected once and spelled twice."
# How a response repeats the choice it already made: a strict schema makes a model fill the field it did not
# choose, and a default may be named. Each is, to the staging, the request it already holds.
RESPELLED = pytest.mark.parametrize(("selection", "respelled"), [
    ("answer_sha256", {"answer": ""}),
    ("answer_sha256", {"rationale": "", "silent": False, "summary": "", "fold_read": False}),
    ("answer", {"answer_sha256": ""}),
], ids=["empty-unused-answer", "named-defaults", "empty-unused-selector"])


def _spelled_twice(selection, respelled):
    chosen = {"answer": SPELLED_TWICE} if selection == "answer" else {"answer_sha256": _sha(SPELLED_TWICE)}
    return _one_response(finish(**chosen), finish(**{**chosen, **respelled}))


@RESPELLED
def test_a_selection_spelled_two_ways_in_one_response_is_one_choice(episode, selection, respelled):
    """The uninterrupted reference: no conflict, one accepted choice of the successor after its reminder."""
    text, usage, trace = episode.run([_read(), None, _text(SPELLED_TWICE), _spelled_twice(selection, respelled)])
    assert text == SPELLED_TWICE and not trace.get("completion_refusals") and len(episode.calls) == 4
    assert [(call["tool"], call["is_error"]) for call in trace["tool_calls"]] == [
        ("read_file", False), ("finish_task", False), ("finish_task", False)]
    _clean_selection(text, usage, trace, selection)


@pytest.mark.parametrize("boundary", ["staged_post_batch", "selected_candidate"])
@RESPELLED
def test_legacy_selection_spelled_two_ways_in_one_batch_is_one_choice(episode, monkeypatch, boundary, selection,
                                                                     respelled):
    """The live staging admitted the repeat as the request it already held, so the restored batch says one thing
    too: the saved selection gets its round's author and is recorded as that successor's choice."""
    plan, answer, twice = episode.tools._ctx.context_fit_plan, SPELLED_TWICE, _spelled_twice(selection, respelled)
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, _text(answer), twice, prompted=True)
    _usage, trace = _legacy_selection(saved, boundary)
    _read_row, first, second = trace["tool_calls"]
    assert first["args"] != second["args"] and not first["is_error"] and not second["is_error"]  # both staged live
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([])
    assert text == answer and len(episode.calls) == 3 and len(_handover_notes(trace)) == 1
    _clean_selection(text, usage, trace, selection)


@pytest.mark.parametrize("respelled", [{}, {"answer_sha256": ""}], ids=["repeated", "spelled-two-ways"])
def test_legacy_first_selection_repeated_in_one_batch_meets_the_reminder_once(episode, monkeypatch, respelled):
    """Saved at the successor's first selection, before its reminder: the restore continues as the uninterrupted
    loop does (``test_a_selection_as_the_first_act_meets_the_reminder_once``), however its repeat was spelled."""
    from tests._delivery_candidate_shared import holds

    plan, answer = episode.tools._ctx.context_fit_plan, "Report"
    twice = _one_response(finish(answer), finish(answer, **respelled))
    path, saved = _legacy_checkpoint(episode, monkeypatch, "staged_post_batch", twice)
    _legacy_selection(saved, "staged_post_batch")
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([finish(answer_sha256=_sha(answer))])
    assert text == answer and len(episode.calls) == 3
    assert [(hold["reason"], hold["subreason"]) for hold in holds(episode.root)] == [
        ("finalization_nudge", "authoring_handover")]
    assert "Selected completion was held" in str(episode.calls[-1]["messages"])
    _clean_selection(text, usage, trace, "answer_sha256")


@pytest.mark.parametrize(("boundary", "prompted"), [("staged_post_batch", False), ("selected_candidate", True)],
                         ids=["before-the-reminder", "after-the-reminder"])
def test_a_long_selection_spelled_two_ways_is_still_one_accepted_choice(episode, monkeypatch, boundary, prompted):
    """A repeat the live staging admitted can render as two different truncation markers in the recorded
    ``args``; the restore reads the one request the batch staged from the transcript's complete arguments,
    on both sides of the one reminder (#3172)."""
    plan, report = episode.tools._ctx.context_fit_plan, "File status verified."
    summary = " ".join(f"Item {number} remains accounted for." for number in range(130))  # past the log's 3000-char cut
    twice = _one_response(finish(report, summary=summary), finish(report, summary=summary + "\n"))
    replies = (_text("Interim report."), twice) if prompted else (twice,)
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, *replies, prompted=prompted)
    _legacy_selection(saved, boundary)
    first, second = saved["trace"]["tool_calls"][-2:]
    assert first["args"]["summary"] != second["args"]["summary"]  # the rendered markers differ
    whole = [json.loads(call["function"]["arguments"]) for row in saved["messages"]
             for call in (row.get("tool_calls") or [])
             if call["function"]["name"] == "finish_task" and call["function"]["arguments"]]
    assert [row["summary"] for row in whole] == [summary, summary + "\n"]  # the transcript keeps the spellings
    path.write_text(json.dumps(saved), encoding="utf8")
    _resume_handover(episode, plan)

    steps = [] if prompted else [finish(answer_sha256=_sha(report))]
    text, usage, trace = episode.run(steps)
    assert text == report
    _clean_selection(text, usage, trace, "answer_sha256" if not prompted else "answer")


def _repeated_round_rows(tool_calls, other_round):
    """The transcript rows of an earlier round that answered the same provider call IDs under other arguments.

    Providers reuse those IDs (tool_call_log); each fake result row is stamped with the round it answers, the
    way the executor stamps real ones."""
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    rows = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "finish_task", "arguments": json.dumps(
            {"action": "finish", "answer": f"Someone else's choice {number}."})}}
        for number, call_id in enumerate(tool_calls)]}]
    for number, call_id in enumerate(tool_calls):
        rows.append({"role": "tool", "tool_call_id": call_id, "content": "ERROR: COMPLETION_ARGUMENT: held",
                     TOOL_RESULT_RECORD_KEY: {"version": 1, "invocation": {
                         "tool_call_id": call_id, "round_id": other_round}}})
    return rows


@pytest.mark.parametrize("boundary", ["staged_post_batch", "selected_candidate"])
def test_legacy_reused_call_ids_read_their_own_batch_and_not_the_repeated_round(episode, monkeypatch, boundary):
    """Provider call IDs repeat between responses (tool_call_log): the restore reads each recorded command
    from the batch whose result records name its round. A transcript that also holds an earlier round under
    the same IDs and other arguments neither denies the successor its one admitted selection nor lends it
    that round's words (#3172)."""
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    plan, answer = episode.tools._ctx.context_fit_plan, "Report selected under IDs an earlier round reused."
    chosen = {"answer_sha256": _sha(answer)}
    twice = _one_response(finish(**chosen), finish(**{**chosen, "answer": ""}))  # one choice, spelled twice
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, _text(answer), twice, prompted=True)
    usage, trace = _legacy_selection(saved, boundary)
    read_row, first, second = trace["tool_calls"]
    assert first["tool_call_id"] != second["tool_call_id"]
    assert all(call["round_id"] == usage["_last_llm_call_meta"]["round_id"] for call in (first, second))
    reply_round, other_round = first["round_id"], read_row["round_id"]
    rows = saved["messages"]
    start = next(i for i, row in enumerate(rows) if row.get("role") == "tool"
                 and ((row.get(TOOL_RESULT_RECORD_KEY) or {}).get("invocation") or {}).get("round_id") == reply_round)
    start = next(i for i in range(start, -1, -1) if rows[i].get("role") == "assistant")
    saved["messages"] = [*rows[:start], *_repeated_round_rows(
        [first["tool_call_id"], second["tool_call_id"]], other_round), *rows[start:]]
    stamped = [(row.get(TOOL_RESULT_RECORD_KEY) or {}).get("invocation", {}).get("round_id")
               for row in saved["messages"] if row.get("role") == "tool"]
    assert stamped == [other_round, other_round, other_round, reply_round, reply_round]  # read, then the repeat
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([])
    assert text == answer and len(episode.calls) == 3 and len(_handover_notes(trace)) == 1
    _clean_selection(text, usage, trace, "answer_sha256")


def test_a_save_from_before_the_result_records_still_repairs_the_selection_by_the_name_match(episode, monkeypatch):
    """The legitimate older boundary: a transcript whose rows carry no record says which round answered a
    repeated ID. Those saves keep the earlier cautious first-name compatibility unchanged — nothing is
    inferred, and a complete choice the staging admitted in one batch stays one admitted choice."""
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    plan, answer = episode.tools._ctx.context_fit_plan, "Report selected from an unrecorded transcript."
    chosen = {"answer_sha256": _sha(answer)}
    twice = _one_response(finish(**chosen), finish(**{**chosen, "answer": ""}))
    path, saved = _legacy_checkpoint(episode, monkeypatch, "staged_post_batch", _text(answer), twice, prompted=True)
    _legacy_selection(saved, "staged_post_batch")
    for row in saved["messages"]:
        if row.get("role") == "tool":
            assert isinstance(row.pop(TOOL_RESULT_RECORD_KEY, None), dict)
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([])
    assert text == answer and len(episode.calls) == 3 and len(_handover_notes(trace)) == 1
    _clean_selection(text, usage, trace, "answer_sha256")


def test_the_batch_binding_reads_its_own_package_and_keeps_the_recorded_copy_when_unproven():
    """The binding axis alone (#3172): the batch whose result records name the round answers; a command the
    records leave out keeps its own recorded args — a broken link never reads another round's spelling."""
    from ouroboros.loop_nudges import _transcript_arguments
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    round_old, round_new = "exec:round:2", "exec:round:4"

    def declared(call_id, arguments):
        return {"id": call_id, "type": "function",
                "function": {"name": "finish_task", "arguments": json.dumps(arguments)}}

    def result(call_id, stamped_round, stamp=True):
        row = {"role": "tool", "tool_call_id": call_id, "content": "ok"}
        if stamp:
            row[TOOL_RESULT_RECORD_KEY] = {"version": 1, "invocation": {
                "tool_call_id": call_id, "round_id": stamped_round}}
        return row

    own = {"action": "finish", "answer": "This batch says one thing."}
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            declared("call-1", {"action": "finish", "answer": "Someone else's choice"}),
            declared("call-3", {"action": "finish", "answer": "Someone else's other choice"})]},
        result("call-1", round_old), result("call-3", round_old),
        {"role": "assistant", "content": None, "tool_calls": [
            declared("call-1", own),
            declared("call-2", {"action": "finish", "answer": "Doubled"}),
            declared("call-2", {"action": "finish", "answer": "in one response"}),
            declared("call-3", own)]},
        result("call-3", round_old, stamp=True),  # the link disagrees: this row names the earlier round
        result("call-1", round_new), result("call-2", round_new),  # call-4 has no result row at all
    ]
    controls = [
        {"tool": "finish_task", "tool_call_id": "call-1", "round_id": round_new, "args": {"rendered": 1}},
        {"tool": "finish_task", "tool_call_id": "call-2", "round_id": round_new, "args": {"copy": "Y"}},
        {"tool": "finish_task", "tool_call_id": "call-3", "round_id": round_new, "args": {"copy": "Z"}},
        {"tool": "finish_task", "tool_call_id": "call-4", "round_id": round_new, "args": {"copy": "W"}},
    ]
    assert _transcript_arguments(controls, messages) == [own, {"copy": "Y"}, {"copy": "Z"}, {"copy": "W"}]
    unrecorded = [{k: v for k, v in row.items() if k != TOOL_RESULT_RECORD_KEY} for row in messages]
    # A save carrying no record cannot tell the packages apart: its IDs repeat, so two namings of one ID say
    # which round spelled neither. Nothing is read from another round's batch — every command keeps its own
    # recorded copy, the same copy those restores settled on before any transcript was read for it.
    assert _transcript_arguments(controls, unrecorded) == [
        {"rendered": 1}, {"copy": "Y"}, {"copy": "Z"}, {"copy": "W"}]


def _spelling_probe(spelling: object, invocation: dict) -> tuple:
    """One batch declaring one call's arguments as ``spelling``, and the result row that answers it."""
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "finish_task", "arguments": spelling}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok",
         TOOL_RESULT_RECORD_KEY: {"version": 1, "invocation": invocation}},
    ]
    controls = [{"tool": "finish_task", "tool_call_id": "call-1", "round_id": "exec:round:4", "args": {"copy": "C"}}]
    return messages, controls


def test_only_a_whole_spelling_certifies_arguments_and_no_broken_one_raises():
    """The decoder's own contract (#3172): a save may hold the arguments field as an object, a list or a
    number, and reading those raises ``TypeError``, not ``ValueError``. Every unreadable form leaves the call
    its recorded copy — in the record era and in an older save that carries no records — while one whole
    readable spelling still answers for itself."""
    from ouroboros.loop_nudges import _transcript_arguments
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    own = {"action": "finish", "answer": "This batch says one thing."}
    unreadable = [("an object instead of text", {"action": "finish"}), ("a list", ["action"]),
                  ("a number", 42), ("a boolean", True), ("text that is not JSON", "action finish"),
                  ("a JSON string", '"finish"'), ("a JSON number", "42"), ("no spelling", None),
                  ("empty text", "")]
    complete = {"tool_call_id": "call-1", "round_id": "exec:round:4"}
    readable = json.dumps(own)
    for why, spelling in [("one whole spelling", readable), *unreadable]:
        expected = own if spelling == readable else {"copy": "C"}
        messages, controls = _spelling_probe(spelling, complete)
        assert _transcript_arguments(controls, messages) == [expected], why
        unrecorded = [{k: v for k, v in row.items() if k != TOOL_RESULT_RECORD_KEY} for row in messages]
        assert _transcript_arguments(controls, unrecorded) == [expected], why


def test_a_record_naming_another_call_leaves_the_call_its_own_copy():
    """A result row binds its batch by the call ID the record itself carries: a record naming another call,
    or no call, or no round, is the gap ``read_tool_result_record`` reports as ``tool_result_mismatch`` or
    ``invocation_unrecorded``, not a link that grants that batch's words to this command (#3172)."""
    from ouroboros.loop_nudges import _transcript_arguments

    own = {"action": "finish", "answer": "This batch says one thing."}
    readable = json.dumps(own)
    for broken in ({"tool_call_id": "call-2", "round_id": "exec:round:4"},  # another call answered in this row
                   {"round_id": "exec:round:4"},  # a record that names no call: identity is a gap, not proof
                   {"tool_call_id": "call-1"},  # and no round: which round ran it stays unknown
                   {}):
        messages, controls = _spelling_probe(readable, broken)
        assert _transcript_arguments(controls, messages) == [{"copy": "C"}], broken


@pytest.mark.parametrize("boundary", ["staged_post_batch", "selected_candidate"])
def test_an_unrecorded_save_with_reused_ids_repairs_from_its_own_copy(episode, monkeypatch, boundary):
    """The same reused provider IDs in a save from before the result records: the first name match could hand
    an earlier round's complete choice to this batch, and no compatibility rule ever required that pick. The
    doubled naming now leaves each command its own recorded copy, which is what these restores were proven to
    settle: one selection, its round's author, no warning of its own (#3172)."""
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    plan, answer = episode.tools._ctx.context_fit_plan, "Report selected under IDs an unrecorded round reused."
    chosen = {"answer_sha256": _sha(answer)}
    twice = _one_response(finish(**chosen), finish(**{**chosen, "answer": ""}))
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, _text(answer), twice, prompted=True)
    usage, trace = _legacy_selection(saved, boundary)
    read_row, first, second = trace["tool_calls"]
    rows, reply_round, other_round = saved["messages"], first["round_id"], read_row["round_id"]
    start = next(i for i, row in enumerate(rows) if row.get("role") == "tool"
                 and ((row.get(TOOL_RESULT_RECORD_KEY) or {}).get("invocation") or {}).get("round_id") == reply_round)
    start = next(i for i in range(start, -1, -1) if rows[i].get("role") == "assistant")
    saved["messages"] = [*rows[:start], *_repeated_round_rows(
        [first["tool_call_id"], second["tool_call_id"]], other_round), *rows[start:]]
    for row in saved["messages"]:  # the format before the executor stamped anything: records nowhere
        if row.get("role") == "tool":
            assert row.pop(TOOL_RESULT_RECORD_KEY, None) is not None
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([])
    assert text == answer and len(episode.calls) == 3 and len(_handover_notes(trace)) == 1
    _clean_selection(text, usage, trace, "answer_sha256")


@pytest.mark.parametrize("boundary", ["staged_post_batch", "selected_candidate"])
def test_a_saved_spelling_that_is_not_text_still_leaves_the_selection_readable(episode, monkeypatch, boundary):
    """A save may hold one call's arguments as the object itself rather than as JSON text. Reading that is a
    shape error, not a request: the restore names no author from it and must not raise over it, and the
    batch's own recorded copies still say one thing, so the selection keeps its author (#3172)."""
    plan, answer = episode.tools._ctx.context_fit_plan, "Report whose saved spelling is not text at all."
    chosen = {"answer_sha256": _sha(answer)}
    twice = _one_response(finish(**chosen), finish(**{**chosen, "answer": ""}))
    path, saved = _legacy_checkpoint(episode, monkeypatch, boundary, _text(answer), twice, prompted=True)
    usage, trace = _legacy_selection(saved, boundary)
    ids = {call["tool_call_id"] for call in trace["tool_calls"][-2:]}
    for row in saved["messages"]:
        for recorded in row.get("tool_calls") or []:
            if recorded.get("id") in ids:
                recorded["function"]["arguments"] = dict(chosen)  # the object where its JSON text belongs
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)

    text, usage, trace = episode.run([])
    assert text == answer and len(episode.calls) == 3 and len(_handover_notes(trace)) == 1
    _clean_selection(text, usage, trace, "answer_sha256")


@pytest.mark.parametrize(("boundary", "selection"), [("candidate", "answer"), ("ready", "answer_sha256")])
def test_same_id_checkpoint_before_or_after_the_notice_keeps_one_selection(episode, monkeypatch, boundary, selection):
    plan = episode.tools._ctx.context_fit_plan
    answer = "The evidence file contains the verified report."
    _interrupt_handover(episode, monkeypatch, boundary, answer)
    assert _resume_handover(episode, plan)["boundary"] == boundary
    request = finish(answer) if selection == "answer" else finish(answer_sha256=_sha(answer))
    text, usage, trace = episode.run([_text("Still checking the file."), request])

    assert text == answer
    assert [call["model"] for call in episode.calls[-2:]] == [SUCCESSOR, SUCCESSOR]
    assert len(_handover_notes(trace)) == 1
    assert [row["tool"] for row in trace["tool_calls"]] == ["read_file", "finish_task"]
    seen = str(episode.calls[-1]["messages"])
    assert "continued from its working checkpoint" in seen and "No completion selection was made" in seen
    _clean_selection(text, usage, trace, selection)


@pytest.mark.parametrize("boundary", ["ready", "selected_candidate"])
def test_owner_changes_the_model_during_a_same_id_restart(episode, monkeypatch, boundary):
    from ouroboros.task_results import write_task_result

    third = "openai/model-c"
    plan = episode.tools._ctx.context_fit_plan
    _interrupt_handover(episode, monkeypatch, boundary, "Report drafted on B")
    write_task_result(episode.root, TASK, "running",
                      metadata={"lead": {"source": "owner", "model": third, "use_local": False}},
                      owner_model={"request_id": "choose-third", "action": "set"})
    _resume_handover(episode, plan)
    if boundary == "selected_candidate":
        calls_before_resume = len(episode.calls)
        text, usage, trace = episode.run([])
        assert text == "Report drafted on B" and len(episode.calls) == calls_before_resume
        assert _clean_selection(text, usage, trace, "answer_sha256")["to_model"] == SUCCESSOR
        return
    text, usage, trace = episode.run([finish("C's draft"), finish(answer_sha256=_sha("C's draft"))])
    assert text == "C's draft" and [c["model"] for c in episode.calls[-2:]] == [third, third]
    first, second = usage["authoring_handovers"]
    assert first["status"] == "pending" and "selection" not in first
    assert (second["from_model"], second["to_model"], second["selection"]) == (SUCCESSOR, third, "answer_sha256")
    assert len(_handover_notes(trace)) == 2


@pytest.mark.parametrize("boundary", ["candidate", "ready", "selected_candidate"])
def test_new_id_retry_keeps_old_handover_as_history_and_requires_its_own_choice(episode, monkeypatch, boundary):
    from ouroboros import working_checkpoint

    draft, retry = "Recoverable answer", "new-attempt-id"
    plan = episode.tools._ctx.context_fit_plan
    _interrupt_handover(episode, monkeypatch, boundary, draft)
    # A pre-upgrade worker could persist the transient degraded signal beside the held candidate.
    path = working_checkpoint.checkpoint_path(episode.root, TASK, 1)
    saved = working_checkpoint.read_working(episode.root, TASK, 1)[1]
    historical = saved["trace"]["route_handovers"][-1]
    if boundary == "candidate":
        assert historical["status"] == "pending" and not historical["recovery_prompted"]
        assert not saved["delivery_candidate"]["control_episode_seen"]
    else:
        historical.update(status="incomplete", incomplete_observed=True)
        saved["trace"]["authoring_handover_incomplete"] = historical
        saved["usage"].update(execution_status="degraded", reason_code="authoring_handover_incomplete",
                              authoring_handovers=[historical])
        path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan, retry)
    text, usage, trace = episode.run([_text("Interim"), finish(answer_sha256=_sha(draft))], task_id=retry)
    assert text == draft and "No completion selection was made" in str(episode.calls[-1]["messages"])
    assert trace["route_handovers"] == [historical] and "authoring_handovers" not in usage
    assert "authoring_handover_incomplete" not in trace
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"


@pytest.mark.parametrize("next_model", [PRIMARY, "openai/model-c"], ids=["same-model", "changed-main"])
def test_new_id_uses_the_last_answering_route_after_a_completed_batch(episode, monkeypatch, next_model):
    from ouroboros import working_checkpoint

    # PRIMARY's read settles an earlier handover. The retry must preserve its history.
    monkeypatch.setenv("OUROBOROS_MODEL", SUCCESSOR)

    def return_to_primary(*_args):
        monkeypatch.setenv("OUROBOROS_MODEL", PRIMARY)
        return _read()

    plan = episode.tools._ctx.context_fit_plan
    _interrupt_handover(episode, monkeypatch, "post_batch", None, steps=[return_to_primary, _read()])
    saved = working_checkpoint.read_working(episode.root, TASK, 1)[1]
    assert saved["delivery_candidate"] is None
    assert saved["trace"]["accepted_routes"][-1]["to_binding"][0] == PRIMARY
    [prior] = saved["trace"]["route_handovers"]
    assert (prior["from_model"], prior["to_model"], prior["status"]) == (SUCCESSOR, PRIMARY, "recovered")

    retry = "retry-report"
    monkeypatch.setenv("OUROBOROS_MODEL", next_model)
    _resume_handover(episode, plan, retry)
    steps = [_text("Report ready")]
    if next_model != PRIMARY:
        steps.append(finish("Selected report"))
    text, usage, trace = episode.run(steps, task_id=retry)
    assert [call["model"] for call in episode.calls] == [SUCCESSOR, PRIMARY] + [next_model] * len(steps)
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    if next_model == PRIMARY:
        assert text == "Report ready" and not _handover_notes(trace)
        assert trace["route_handovers"] == [prior] and "authoring_handovers" not in usage
    else:
        [row] = usage["authoring_handovers"]
        assert text == "Selected report" and len(_handover_notes(trace)) == 1
        assert (row["from_model"], row["to_model"], row["reason"]) == (PRIMARY, next_model, "settings_change")
        assert (row["status"], row["selection"], row["recovery_prompted"]) == ("selected", "answer", True)
        assert trace["route_handovers"] == [prior, row]


@pytest.mark.parametrize(("boundary", "legacy"), [
    ("ready", False), ("selected_candidate", False), ("selected_candidate", "with-author"),
    ("selected_candidate", "unknown-author"), ("selected_candidate", "saved-choice-only"),
])
def test_following_main_changes_before_the_saved_candidate_is_restored(episode, monkeypatch, boundary, legacy):
    def next_main(*_args):
        monkeypatch.setenv("OUROBOROS_MODEL", SUCCESSOR)
        return _read()

    answer, third = "Report selected by B", "openai/model-c"
    plan = episode.tools._ctx.context_fit_plan
    steps = [next_main, _text(answer)]
    if boundary == "selected_candidate":
        steps.append(finish(answer_sha256=_sha(answer)))
    _interrupt_handover(episode, monkeypatch, boundary, answer, steps=steps)
    if legacy:
        from ouroboros import working_checkpoint
        saved = working_checkpoint.read_working(episode.root, TASK, 1)[1]
        for rows in (saved["trace"]["route_handovers"], saved["usage"]["authoring_handovers"]):
            rows[-1].pop("reply_observed", None)
        if legacy in {"unknown-author", "saved-choice-only"}:
            saved["delivery"]["_completion_observation"].pop("model")
        if legacy == "unknown-author":
            saved["delivery"]["_completion_selected"]["observation"].pop("model")
        working_checkpoint.checkpoint_path(episode.root, TASK, 1).write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_MODEL", third)
    _resume_handover(episode, plan)
    if boundary == "selected_candidate":
        text, usage, trace = episode.run([])
        assert [call["model"] for call in episode.calls] == [PRIMARY, SUCCESSOR, SUCCESSOR]
        [row] = usage["authoring_handovers"]
        if legacy in {"unknown-author", "saved-choice-only"}:
            assert row["status"] == "pending" and "selection" not in row
            assert "reply_observed" not in row
        else:
            assert (row["status"], row["selection"]) == ("selected", "answer_sha256")
        assert usage.get("reason_code") != "authoring_handover_incomplete"
        assert (text, row["from_model"], row["to_model"], row["reason"]) == (answer, PRIMARY, SUCCESSOR, "settings_change")
    else:
        final = "Report selected by C"
        text, usage, trace = episode.run([finish(final), finish(answer_sha256=_sha(final))])
        assert [call["model"] for call in episode.calls] == [PRIMARY, SUCCESSOR, third, third]
        first, second = usage["authoring_handovers"]
        assert (first["from_model"], first["to_model"], first["status"]) == (PRIMARY, SUCCESSOR, "pending")
        assert (second["from_model"], second["to_model"], second["selection"]) == (SUCCESSOR, third, "answer_sha256")
        assert text == final and len(_handover_notes(trace)) == 2


@pytest.mark.parametrize("next_main", [SUCCESSOR, "openai/model-c", PRIMARY],
                         ids=["target-replies", "another-route-replies", "mind-switches-to-the-target"])
def test_legacy_pair_recorded_before_its_reply_needs_its_targets_reply(episode, monkeypatch, next_main):
    """The previous body wrote a settings change's pair at the switch itself. A save taken before the new route
    answered holds a pair no reply confirms: only its target's reply after the author it names completes it,
    and not a reply another route gives or one the mind's own switch brings."""
    plan, draft = episode.tools._ctx.context_fit_plan, "The file has the report."
    path, saved = _legacy_checkpoint(episode, monkeypatch, "next_ready", awaited=True)
    assert saved["working"]["boundary"] == "ready" and len(saved["usage"]["llm_call_refs"]) == 1
    path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_MODEL", next_main)
    _resume_handover(episode, plan)
    if next_main == PRIMARY:  # Main went back; the author the pair names then chooses its target itself
        monkeypatch.setattr(LLMClient, "available_models", lambda _self: [PRIMARY, SUCCESSOR])
        text, usage, trace = episode.run([_tool("switch_model", model=SUCCESSOR), _text(draft)])
        assert text == draft and [call["model"] for call in episode.calls] == [PRIMARY, PRIMARY, SUCCESSOR]
        [pair] = usage["authoring_handovers"]
        assert trace["route_handovers"] == [pair] and not _handover_notes(trace)
        assert pair["status"] == "pending" and "reply_observed" not in pair and "selection" not in pair
        return

    text, usage, trace = episode.run([_text(draft), finish(answer_sha256=_sha(draft))])
    assert text == draft and [call["model"] for call in episode.calls] == [PRIMARY, next_main, next_main]
    pair, *later = usage["authoring_handovers"]
    assert trace["route_handovers"] == [pair, *later] and len(_handover_notes(trace)) == 1
    assert (pair["from_model"], pair["to_model"], pair["reason"]) == (PRIMARY, SUCCESSOR, "settings_change")
    if next_main == SUCCESSOR:
        assert not later
        _clean_selection(text, usage, trace, "answer_sha256")
    else:  # the pair's target never answered: it stays as written beside the actual author's own row
        assert pair["status"] == "pending" and "reply_observed" not in pair and "selection" not in pair
        [row] = later
        assert (row["from_model"], row["to_model"], row["reply_observed"]) == (PRIMARY, next_main, True)
        assert (row["status"], row["selection"]) == ("selected", "answer_sha256")
        assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"


def test_failed_following_main_route_does_not_become_an_author(episode, monkeypatch):
    third = "openai/model-c"

    def change_main(*_args):
        monkeypatch.setenv("OUROBOROS_MODEL", third)
        return _read()

    text, usage, trace = episode.run([change_main, None, _text("Working on the report"), finish("File report")])
    assert [call["model"] for call in episode.calls] == [PRIMARY, third, SUCCESSOR, SUCCESSOR]
    row = _clean_selection(text, usage, trace, "answer")
    assert (row["from_model"], row["to_model"]) == (PRIMARY, SUCCESSOR)


def test_a_retry_restores_an_ordinary_final_after_successful_successor_work(episode, monkeypatch):
    plan, answer, retry = episode.tools._ctx.context_fit_plan, "Checked both files", "retry-report"
    _interrupt_handover(episode, monkeypatch, "recovered_candidate", answer,
                        steps=[_read(), None, _read(), _text(answer)])
    _resume_handover(episode, plan, retry)
    text, usage, trace = episode.run([], task_id=retry)
    assert text == answer and "authoring_handovers" not in usage
    assert trace["route_handovers"][-1]["status"] == "recovered"


@pytest.mark.parametrize("choice", ["silent", "stop"])
@pytest.mark.parametrize("new_id", [False, True])
def test_checkpoint_preserves_quiet_or_stop_only_for_its_own_execution(episode, monkeypatch, choice, new_id):
    episode.tools._ctx.task_metadata = deepcopy(SCHEDULED_REPORT)
    answer = "Partial report"
    request = finish(answer, silent=True) if choice == "silent" else finish(answer, action="stop", rationale="File missing")
    plan = episode.tools._ctx.context_fit_plan
    _interrupt_handover(episode, monkeypatch, "candidate", answer, steps=[_read(), None, request])
    task_id = "retry-report" if new_id else TASK
    _resume_handover(episode, plan, task_id)
    episode.tools._ctx.task_metadata = deepcopy(SCHEDULED_REPORT)
    if new_id:
        text, usage, trace = episode.run([finish("New report")], task_id=task_id)
        assert text == "New report" and not usage.get("wake_silent") and "task_completion" not in trace
        assert "authoring_handovers" not in usage and trace["route_handovers"][-1]["status"] == "pending"
    else:
        text, usage, trace = episode.run([])
        [row] = usage["authoring_handovers"]
        assert (text, row["selection"], row["recovery_prompted"]) == (answer, choice, False)
        assert bool(usage.get("wake_silent")) == (choice == "silent")
        assert (trace.get("task_completion") or {}).get("action") == ("stop" if choice == "stop" else None)


def test_a_selection_as_the_first_act_meets_the_reminder_once(episode):
    from tests._delivery_candidate_shared import holds

    text, usage, trace = episode.run([_read(), None, finish("Report"), finish(answer_sha256=_sha("Report"))])
    assert text == "Report"
    assert [(row["reason"], row["subreason"]) for row in holds(episode.root)] == [
        ("finalization_nudge", "authoring_handover")]
    assert "Selected completion was held" in str(episode.calls[-1]["messages"])
    _clean_selection(text, usage, trace, "answer_sha256")


def test_status_prose_after_the_reminder_stays_held_until_a_selection(episode):
    from tests._delivery_candidate_shared import holds

    verified = "The result is verified."
    text, usage, trace = episode.run([
        _read(), None, _text("Checking the result."), _text(verified), finish(answer_sha256=_sha(verified)),
    ])
    assert text == verified  # the exact held response, selected by its hash
    assert [(row["reason"], row["subreason"]) for row in holds(episode.root)] == [
        ("finalization_nudge", "authoring_handover"), ("selection_required", None)]
    assert f"latest whole held response answer_sha256={_sha(verified)}" in str(episode.calls[-1]["messages"])
    _clean_selection(text, usage, trace, "answer_sha256")


def test_a_stale_selector_after_the_reminder_is_refused_and_completes_nothing(episode):
    def after_refusal(state, _usage, _kwargs):
        row = state.tools._ctx._authoring_handover
        assert row is not None and (row["status"], row["recovery_prompted"]) == ("pending", True)
        return finish("Final answer")

    text, usage, trace = episode.run([
        _read(), None, _text("First status."), _text("Second status."), _text("Third status."),
        finish(answer_sha256=_sha("Second status.")), after_refusal,  # no longer the latest held response
    ])
    assert text == "Final answer"
    assert [row["reason"].split(":", 1)[0] for row in trace["completion_refusals"]] == ["selection_unavailable"]
    _clean_selection(text, usage, trace, "answer")


def test_a_failed_completion_request_is_not_tool_work(episode):
    text, usage, trace = episode.run([
        _read(), None, finish("Report", answer_sha256="b" * 64), _text("Report"), finish("Report"),
    ])
    assert text == "Report"
    assert [(row["tool"], row["is_error"]) for row in trace["tool_calls"]] == [
        ("read_file", False), ("finish_task", True), ("finish_task", False)]
    assert len(_handover_notes(trace)) == 1  # the refused request did not stand in for work or skip the reminder
    _clean_selection(text, usage, trace, "answer")


@pytest.mark.parametrize("before_notice", [True, False], ids=["before_notice", "after_notice"])
def test_a_refused_read_keeps_its_error_and_the_handover_selection(episode, before_notice):
    refusal = _tool("read_file")  # required path omitted
    draft = _text("The draft report is ready.")
    text, usage, trace = episode.run([
        _read(), None, *([refusal, draft] if before_notice else [draft, refusal]),
        _text("Still checking."), finish("The report uses the predecessor's evidence."),
    ])
    assert text == "The report uses the predecessor's evidence."
    assert len(_handover_notes(trace)) == 1
    assert f"latest whole held response answer_sha256={_sha('Still checking.')}" in str(episode.calls[-1]["messages"])
    assert [(row["tool"], row["is_error"]) for row in trace["tool_calls"]] == [
        ("read_file", False), ("read_file", True), ("finish_task", False)]
    _clean_selection(text, usage, trace, "answer")
    errors = derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["unresolved_tool_errors"]
    assert [(row["tool"], row["status"]) for row in errors] == [("read_file", "argument_error")]


def test_recovery_tools_publish_fresh_final_without_json_only_instruction(episode):
    text, usage, trace = episode.run([
        _read(), None, _text("WIP before recovery"), _read(), finish("Fresh verified report"),
    ])
    assert text == "Fresh verified report"
    assert [row["tool"] for row in trace["tool_calls"]] == ["read_file", "read_file", "finish_task"]
    assert len(_handover_notes(trace)) == 1
    assert usage.get("reason_code") != "authoring_handover_incomplete"
    assert "authoring_handover_incomplete" not in trace
    assert trace["route_handovers"][0]["status"] == "recovered"
    assert not any("DELIVERY_FINALIZATION_CONTROL" in str(row.get("content", ""))
                   for call in episode.calls for row in call["messages"])
    assert trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("discovery", [_tool("enable_tools", tools="read_file"), _tool("list_available_tools")])
def test_schema_discovery_leaves_the_successor_owing_its_choice(episode, discovery):
    text, usage, trace = episode.run([_read(), None, discovery, _text("Interim"), finish("Verified")])
    assert trace["tool_calls"][1]["is_error"] is False
    assert len(_handover_notes(trace)) == 1
    _clean_selection(text, usage, trace, "answer")


@pytest.mark.parametrize("after_reminder", [False, True])
def test_author_stop_settles_the_handover_in_its_own_round(episode, after_reminder):
    stop = finish("Partial report", action="stop", rationale="The second part is unfinished.")
    text, usage, trace = episode.run([_read(), None, *([_text("Checking")] if after_reminder else []), stop])
    assert text == "Partial report" and trace["task_completion"]["action"] == "stop"
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["objective"]["reason"] == "author_stop"
    [row] = usage["authoring_handovers"]
    assert (row["status"], row["selection"], row["recovery_prompted"]) == ("selected", "stop", after_reminder)
    assert len(_handover_notes(trace)) == int(after_reminder)


PARTIAL = "Partial report: the second file is unread."


def _read_and_stop(*, rationale="The second file is still unread."):
    """The successor's one response: a read that succeeds beside its explicit stop over a partial answer."""
    return _one_response(_read(), finish(PARTIAL, action="stop", rationale=rationale))


def _worked_and_stopped(text, usage, trace):
    """Both acts of the successor's one response are on its row; the task ends as an author stop."""
    assert text == PARTIAL and trace["task_completion"]["action"] == "stop"
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["objective"]["reason"] == "author_stop"
    assert [(call["tool"], call["is_error"]) for call in trace["tool_calls"]] == [
        ("read_file", False), ("read_file", False), ("finish_task", False)]
    [row] = usage["authoring_handovers"]
    assert trace["route_handovers"] == [row] and (row["from_model"], row["to_model"]) == (PRIMARY, SUCCESSOR)
    assert (row["status"], row["selection"], row["work_observed"], row["reply_observed"]) == (
        "selected", "stop", True, True)
    assert "authoring_handover_incomplete" not in trace and usage.get("reason_code") != "authoring_handover_incomplete"
    return row


@pytest.mark.parametrize("after_reminder", [False, True])
def test_one_response_that_works_and_stops_keeps_both_acts_on_its_row(episode, after_reminder):
    """The batch's successful read ends the handover before the delivery gates accept the stop the same response
    made: the row says both, not the read alone beside a task that records the stop."""
    row = _worked_and_stopped(*episode.run(
        [_read(), None, *([_text("Checking")] if after_reminder else []), _read_and_stop()]))
    assert row["recovery_prompted"] is after_reminder
    assert len(episode.calls) == 3 + after_reminder


def test_a_refused_call_beside_the_work_and_the_stop_stays_its_own_warning(episode):
    refused = _tool("read_file")  # required path omitted
    stop = finish(PARTIAL, action="stop", rationale="The second file is still unread.")
    text, usage, trace = episode.run([_read(), None, _one_response(_read(), refused, stop)])
    [row] = usage["authoring_handovers"]
    assert (row["status"], row["selection"], row["work_observed"]) == ("selected", "stop", True)
    assert text == PARTIAL and trace["task_completion"]["action"] == "stop"
    errors = derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["unresolved_tool_errors"]
    assert [(error["tool"], error["status"]) for error in errors] == [("read_file", "argument_error")]


@pytest.mark.parametrize("after_reminder", [False, True])
def test_a_finish_riding_with_its_note_is_a_recorded_choice_only_after_the_reminder(episode, after_reminder):
    """A receipt write is work and rides with the finish of its response. Before the reminder that finish is an
    ordinary one, so the work alone ends the handover; after it the accepted answer is the successor's choice too."""
    episode.tools.override_handler("chronicle_write", lambda _ctx, **_arguments: "written")
    reply = _one_response(_tool("chronicle_write", kind="page", text="What I checked"), finish("Report"))
    text, usage, trace = episode.run([_read(), None, *([_text("Checking")] if after_reminder else []), reply])
    [row] = usage["authoring_handovers"]
    assert text == "Report" and row["work_observed"] is True and "task_completion" not in trace
    assert (row["status"], row.get("selection")) == (("selected", "answer") if after_reminder else ("recovered", None))
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    assert len(_handover_notes(trace)) == int(after_reminder) and not trace.get("completion_refusals")


def test_a_quiet_finish_riding_with_its_note_keeps_both_acts(episode):
    episode.tools._ctx.task_metadata = deepcopy(SCHEDULED_REPORT)
    episode.tools.override_handler("chronicle_write", lambda _ctx, **_arguments: "written")
    reply = _one_response(_tool("chronicle_write", kind="page", text="What I checked"), finish(silent=True))
    text, usage, trace = episode.run([_read(), None, reply])
    [row] = usage["authoring_handovers"]
    assert text == "" and usage["wake_silent"] is True and not _handover_notes(trace)
    assert (row["status"], row["selection"], row["work_observed"], row["recovery_prompted"]) == (
        "selected", "silent", True, False)


def test_a_revisited_successor_that_works_and_stops_settles_only_its_own_row(episode, monkeypatch):
    """A→B→A→B at one tool count: the returned target's one response reads and stops, and both acts land on the
    row of that last departure alone."""
    text, usage, trace = episode.run(_revisit(monkeypatch, _read_and_stop(), scripted=True))
    assert [call["model"] for call in episode.calls] == [PRIMARY, SUCCESSOR, PRIMARY, SUCCESSOR]
    assert text == PARTIAL and trace["task_completion"]["action"] == "stop"
    first, returned, final = usage["authoring_handovers"]
    assert [(row["from_model"], row["to_model"], row["tool_calls_at_handover"]) for row in (
        first, returned, final)] == REVISITED_ROWS and trace["route_handovers"][-1] is final
    assert (final["status"], final["selection"], final["work_observed"]) == ("selected", "stop", True)
    for earlier in (first, returned):
        assert earlier["status"] == "pending" and not {"selection", "work_observed"} & set(earlier)


@pytest.mark.parametrize("boundary", ["post_batch", "recovered_candidate"])
@pytest.mark.parametrize("new_id", [False, True])
def test_a_restart_between_the_work_and_its_stop_keeps_both_only_for_its_own_execution(episode, monkeypatch,
                                                                                    boundary, new_id):
    plan = episode.tools._ctx.context_fit_plan
    _interrupt_handover(episode, monkeypatch, boundary, None, steps=[_read(), None, _read_and_stop()])
    task_id = "retry-report" if new_id else TASK
    _resume_handover(episode, plan, task_id)
    if new_id:
        text, usage, trace = episode.run([finish("New report")], task_id=task_id)
        assert text == "New report" and "task_completion" not in trace and "authoring_handovers" not in usage
        [history] = trace["route_handovers"]  # the stop the earlier execution staged was never accepted here
        assert (history["status"], history["work_observed"]) == ("recovered", True) and "selection" not in history
    else:
        _worked_and_stopped(*episode.run([]))


def test_legacy_save_of_a_response_that_worked_and_stopped_restores_both_acts(episode, monkeypatch):
    """The previous body left the row pending at the batch end and the staged stop without its author: the
    receipts credit the same round's read and name that stop's author, and the gates then accept it."""
    plan = episode.tools._ctx.context_fit_plan
    path, saved = _legacy_checkpoint(episode, monkeypatch, "staged_post_batch", _read_and_stop())
    _legacy_selection(saved, "staged_post_batch")
    path.write_text(json.dumps(saved), encoding="utf-8")
    _resume_handover(episode, plan)
    _worked_and_stopped(*episode.run([]))
    assert len(episode.calls) == 2


@pytest.mark.parametrize("held", ["unread-result", "contradicted", "owner-follow-up"])
def test_a_completion_held_beside_the_work_never_reaches_the_row(episode, held):
    """Only acceptance records a choice. A request the gates refuse or defer leaves the row as the work left it.
    The stop a later reply makes is accepted for the task, but it is another response's: it does not rewrite the
    row, whether that reply only stops or works again beside its stop."""
    from ouroboros.tools.control_runtime import _finish_task

    later, mailed = "Stopping after the held request.", []
    final = finish(PARTIAL, action="stop", rationale=later)
    if held == "unread-result":  # a finish beside a result it has not read
        reply = _one_response(_read(), finish("Early report"))
    elif held == "contradicted":  # two different requests in one response
        reply = _one_response(_read(), finish(PARTIAL, action="stop", rationale="Unread file."), finish("Whole report"))
    else:  # my human writes while the batch runs: the stop is returned for the new words
        reply, final = _read_and_stop(), _read_and_stop(rationale=later)

        def stop_then_mail(ctx, **arguments):
            receipt = _finish_task(ctx, **arguments)
            if not mailed:
                mailed.append(write_owner_message(episode.root, "Name the unread file too.", TASK, msg_id="follow-up"))
            return receipt

        episode.tools.override_handler("finish_task", stop_then_mail)
    text, usage, trace = episode.run([_read(), None, reply, final])
    [row] = usage["authoring_handovers"]
    assert trace["route_handovers"] == [row] and (row["status"], row["work_observed"]) == ("recovered", True)
    assert "selection" not in row
    assert text == PARTIAL and trace["task_completion"]["rationale"] == later  # the later stop itself was accepted
    if held == "owner-follow-up":
        assert len(mailed) == 1 and "Name the unread file too." in str(episode.calls[-1]["messages"])
    else:
        expected = "unobserved_tool_results" if held == "unread-result" else "contradictory_completion_requests"
        assert [refusal["reason"].split(":", 1)[0] for refusal in trace["completion_refusals"]] == [expected]


@pytest.mark.parametrize(("rail", "late", "worked"), [
    pytest.param("stop_now", False, False, id="direct-stop-after-batch"),
    pytest.param("stop_now", False, True, id="direct-stop-after-a-batch-that-also-worked"),
    pytest.param("deadline_local", False, False, id="deadline-after-batch"),
    pytest.param("deadline_local", False, True, id="deadline-after-a-batch-that-also-worked"),
    pytest.param("stop_now", True, False, id="direct-stop-at-admission"),
    pytest.param("wrap_up", True, False, id="wrap-up-at-admission"),
    pytest.param("wrap_up", True, True, id="wrap-up-at-admission-after-work"),
    pytest.param("presence_stop_now", True, False, id="presence-stop-at-admission"),
])
def test_hard_rail_keeps_a_validated_successor_stop_without_changing_the_exit(episode, monkeypatch, rail, late,
                                                                              worked):
    from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, STOP_POLICY_IMMEDIATE, request_cancel
    from ouroboros.tools.control_runtime import _finish_task
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, owner_stop_control_id

    ctx = episode.tools._ctx
    ctx.is_direct_chat = rail in {"stop_now", "presence_stop_now"}
    text, injected = "Partial report, second file not checked.", []

    def control():
        injected.append(rail)
        if rail == "presence_stop_now":
            ctx.model_wait_context = SimpleNamespace(
                worker_slot_held=False, task={"_presence_turn": True}, canonical_root=episode.root, task_id=TASK)
            request_cancel(episode.root, TASK, requested_stop_policy=STOP_POLICY_IMMEDIATE)
        elif rail == "wrap_up":
            intent = request_cancel(episode.root, TASK, requested_stop_policy=STOP_POLICY_FINALIZE)
            write_owner_message(episode.root, REASON_OWNER_REQUESTED_FINALIZATION, TASK,
                                msg_id=owner_stop_control_id(intent), kind=KIND_FINALIZE_NOW)
        else:
            reason = REASON_OWNER_STOPPED_DIRECT_TURN if rail == "stop_now" else rail
            write_owner_message(episode.root, reason, TASK, kind=KIND_FINALIZE_NOW)

    def stop_then_control(ctx, **kwargs):
        receipt = _finish_task(ctx, **kwargs)
        control()
        return receipt

    if late:
        ctx.owner_message_admission_lock = threading.Lock()
        ctx.owner_message_admission_agent = SimpleNamespace(
            _accepting_owner_messages=True, _busy=True, _current_task_id=TASK)
        finalize_services = loop._finalize_task_services

        def arrive(exit_ctx):
            if not injected and (getattr(ctx, "_completion_selected", None) or {}).get("action") == "stop":
                control()
            return finalize_services(exit_ctx)

        monkeypatch.setattr(loop, "_finalize_task_services", arrive)
    else:
        episode.tools.override_handler("finish_task", stop_then_control)

    stop = finish(text, action="stop", rationale="Work remains.")
    result, usage, trace = episode.run([_read(), None, _one_response(_read(), stop) if worked else stop])
    assert injected == [rail] and result == text and len(episode.calls) == 3 and not _handover_notes(trace)
    [row] = usage["authoring_handovers"]
    assert (row["status"], row["selection"], row["recovery_prompted"]) == ("selected", "stop", False)
    assert trace["route_handovers"] == [row] and row.get("work_observed") is (True if worked else None)
    assert trace["forced_finalization"]["candidate_sha256"] == _sha(text)
    assert usage["reason_code"] == (rail if rail == "deadline_local" else REASON_OWNER_REQUESTED_FINALIZATION)
    assert usage["execution_status"] == "failed" and "task_completion" not in trace


@pytest.mark.parametrize("record", ["", "Private record"])
def test_scheduled_successor_can_finish_quietly_as_its_first_act(episode, record):
    episode.tools._ctx.task_metadata = deepcopy(SCHEDULED_REPORT)
    text, usage, trace = episode.run([_read(), None, finish(record, silent=True)])
    assert text == record and usage["wake_silent"] is True
    [row] = usage["authoring_handovers"]
    assert (row["status"], row["selection"], row["recovery_prompted"]) == ("selected", "silent", False)
    assert not _handover_notes(trace)


def test_each_model_owns_its_batch_even_when_the_next_call_switches(episode):
    # The fallback's own next failure returns the ordinary call to the primary.
    text, usage, trace = episode.run([_read(), None, _read(), None, _text("Returning"), finish("Report")])
    first, second = usage["authoring_handovers"]
    assert (first["to_model"], first["status"], first["work_observed"]) == (SUCCESSOR, "recovered", True)
    assert (second["to_model"], second["status"], second["selection"]) == (PRIMARY, "selected", "answer")
    assert len(_handover_notes(trace)) == 1 and text == "Report"


def test_completed_read_is_recorded_before_a_forced_exit(episode):
    def read_and_wrap(state, _usage, _kwargs):
        write_owner_message(state.root, "deadline", TASK, kind=KIND_FINALIZE_NOW)
        return _read()

    _, usage, trace = episode.run([_read(), None, read_and_wrap, _text("Best effort")])
    [row] = trace["route_handovers"]
    assert row["status"] == "recovered" and row["work_observed"] is True
    assert usage["reason_code"] != "authoring_handover_incomplete" and not _handover_notes(trace)


def test_zero_prior_tools_fallback_remains_normal_final(episode):
    text, usage, trace = episode.run([None, _text("Ordinary answer")])
    assert text == "Ordinary answer"
    assert len(episode.calls) == 2
    assert not trace.get("route_handovers")
    assert not _handover_notes(trace)
    assert usage.get("reason_code") != "authoring_handover_incomplete"


def test_explicit_switch_model_is_not_host_handover(episode, monkeypatch):
    monkeypatch.setattr(LLMClient, "available_models", lambda _self: [PRIMARY, SUCCESSOR])
    text, usage, trace = episode.run([
        _read(), _tool("switch_model", model=SUCCESSOR), _text("Deliberately switched final"),
    ])
    assert text == "Deliberately switched final"
    assert [call["model"] for call in episode.calls] == [PRIMARY, PRIMARY, SUCCESSOR]
    assert all(row["is_error"] is False for row in trace["tool_calls"])
    assert not trace.get("route_handovers")
    assert usage.get("reason_code") != "authoring_handover_incomplete"


@pytest.mark.parametrize("source", ["owner", "owner_words", "owner_press"],
                         ids=["model-sheet", "cited-words", "cited-card-press"])
def test_a_durable_work_model_choice_uses_the_actual_initiator(episode, source):
    from ouroboros.task_results import write_task_result

    def choose_during_reply(state, _usage, _kwargs):
        lead = {"source": source, "model": SUCCESSOR, "use_local": False}
        if source != "owner":
            lead["cited"] = "Use this model"
        write_task_result(state.root, TASK, "running", metadata={"lead": lead},
                          owner_model={"request_id": "new-work-model", "action": "set"})
        write_owner_message(state.root, "owner_model", TASK, kind=KIND_OWNER_MODEL, msg_id="owner_model:new-work-model")
        return _read()

    steps = [choose_during_reply, _text("Report")]
    if source == "owner":
        steps.append(finish(answer_sha256=_sha("Report")))
    text, usage, trace = episode.run(steps)
    assert [call["model"] for call in episode.calls] == [PRIMARY] + [SUCCESSOR] * (len(steps) - 1)
    assert text == "Report"
    if source == "owner":
        row = _clean_selection(text, usage, trace, "answer_sha256")
        assert (row["from_model"], row["to_model"], row["reason"]) == (PRIMARY, SUCCESSOR, "owner_model")
    else:
        assert "route_handovers" not in trace and not _handover_notes(trace)


@pytest.mark.parametrize(("prior_tools", "targets"), [
    pytest.param(False, (PRIMARY,), id='False-targets0'),
    pytest.param(True, (SUCCESSOR,), id='True-targets1'),
    pytest.param(True, (SUCCESSOR, SUCCESSOR), id='True-targets2'),
    pytest.param(True, (SUCCESSOR, "openai/model-c"), id="only-final-wait-choice-answered"),
    pytest.param(True, (SUCCESSOR, PRIMARY), id="returned-to-actual-author"),
])
def test_model_wait_reprepare_preserves_original_author_and_tool_baseline(episode, targets, prior_tools):
    target = targets[-1]
    def switch(_state, _usage, kwargs):
        owner = model_wait.current_model_wait()
        for destination in targets:
            prepared = owner.reprepare(kwargs["model_role"], {
                "model": destination, "model_role": kwargs["model_role"], "use_local": False,
                "messages": episode.calls[-1]["messages"],
            })
            assert isinstance(prepared, model_wait.PreparedModelCall)
        assert any(row.get("role") == "tool" for row in prepared.kwargs["messages"]) is prior_tools
        return _text("Wait switched response")

    steps = [_read(), switch] if prior_tools else [switch]
    if target != PRIMARY and prior_tools:
        steps.append(finish("Second successor response"))
    with model_wait.task_model_wait_scope(task={"id": TASK}, drive_root=episode.root,
                                         event_queue=None, worker_slot_held=False) as owner:
        owner.tool_context = episode.tools._ctx
        text, usage, trace = episode.run(steps)
    if target == PRIMARY or not prior_tools:
        assert text == "Wait switched response"
        assert not trace.get("route_handovers")
        assert not _handover_notes(trace)
    else:
        assert text == "Second successor response"
        assert episode.calls[-1]["model"] == target
        assert trace["route_handovers"][0]["reason"] == "model_wait"
        assert trace["route_handovers"][0]["from_model"] == PRIMARY
        assert trace["route_handovers"][0]["tool_calls_at_handover"] == 1
        assert len(_handover_notes(trace)) == 1
        _clean_selection(text, usage, trace, "answer")


@pytest.mark.parametrize("control", ["owner_stop", "deadline"])
def test_forced_finalization_takes_precedence_over_pending_handover(episode, control):
    def trigger(state, _usage, _kwargs):
        if control == "owner_stop":
            from ouroboros.cancel_intents import request_cancel, STOP_POLICY_FINALIZE
            from supervisor.owner_stop import owner_stop_control_id
            intent = request_cancel(state.root, TASK, requested_stop_policy=STOP_POLICY_FINALIZE)
            write_owner_message(state.root, "owner_requested_finalization", TASK,
                                msg_id=owner_stop_control_id(intent), kind=KIND_FINALIZE_NOW)
        else:
            state.tools._ctx.task_metadata = {
                "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat(),
            }
        return _text("Interim response")

    steps = [_read(), None, trigger]
    if control == "deadline":
        steps.append(_text("Best effort on interruption"))
    text, usage, trace = episode.run(steps)
    assert text, (usage, trace)
    if control == "deadline":
        assert "Best effort on interruption" in text
    assert usage["reason_code"] == ("owner_requested_finalization" if control == "owner_stop" else "deadline_local")
    assert usage.get("reason_code") != "authoring_handover_incomplete"
    assert len(_handover_notes(trace)) == 1


@pytest.mark.parametrize(("reply", "selection", "verdict"), [
    (_text(json.dumps({"action": "finish", "answer": "Final report"})), "answer", "selected"),
    (finish(answer_sha256=_sha("File report")), "answer_sha256", "selected"),
    (finish("Private report", silent=True), "silent", "selected"),
    (finish("Partial report", action="stop", rationale="A file is missing"), "stop", "selected"),
    (finish(answer_sha256="0" * 64), None, "refused"),
    (_text("Best effort"), None, None),
])
def test_forced_exit_records_only_its_accepted_choice(episode, reply, selection, verdict):
    def trigger(state, _usage, _kwargs):
        state.tools._ctx.task_metadata = {**SCHEDULED_REPORT, "deadline_at":
            (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()}
        return _text("File report")

    text, usage, trace = episode.run([_read(), None, trigger, reply])
    [row] = usage["authoring_handovers"]
    assert trace["route_handovers"] == [row] and row["to_model"] == SUCCESSOR
    assert (row["status"], row.get("selection")) == ("selected" if selection else "pending", selection)
    assert (trace.get("forced_completion_request") or {}).get("status") == verdict
    assert usage["reason_code"] == "deadline_local" and len(_handover_notes(trace)) == 1
    if selection:
        assert text == {"answer": "Final report", "answer_sha256": "File report",
                        "silent": "Private report", "stop": "Partial report"}[selection]
        assert bool(usage.get("wake_silent")) == (selection == "silent")
        assert (trace.get("task_completion") or {}).get("action") == ("stop" if selection == "stop" else None)


@pytest.mark.parametrize(("reply", "selection"), [
    (finish(answer_sha256=_sha("File report")), "answer_sha256"),
    (_text("Report from C"), None),
    (None, None),
    (_tool(path="without-an-execution.txt"), None),
], ids=["digest", "complete-prose", "no-reply", "tool-preamble"])
def test_a_different_forced_author_owns_only_its_own_choice(episode, monkeypatch, reply, selection):
    third = "openai/model-c"

    def change_to_b(*_args):
        monkeypatch.setenv("OUROBOROS_MODEL", SUCCESSOR)
        return _read()

    def change_to_c(state, usage, _kwargs):
        usage["_last_llm_call_meta"] = {"model": SUCCESSOR}  # an earlier call's metadata cannot name the forced author
        monkeypatch.setenv("OUROBOROS_MODEL", third)
        state.tools._ctx.task_metadata = {"deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()}
        return _text("File report")

    text, usage, trace = episode.run([change_to_b, change_to_c, reply])
    assert [call["model"] for call in episode.calls] == [PRIMARY, SUCCESSOR, third]
    first, *rest = usage["authoring_handovers"]
    assert (first["from_model"], first["to_model"], first["status"]) == (PRIMARY, SUCCESSOR, "pending")
    if reply is None or reply.get("tool_calls") and not selection:
        assert not rest and trace["forced_finalization"]["source"] != "model"
    else:
        [second] = rest
        assert (second["from_model"], second["to_model"]) == (SUCCESSOR, third)
        assert (second["status"], second.get("selection")) == ("selected" if selection else "pending", selection)
        assert second["reply_observed"] is True and not second["recovery_prompted"]
        assert text == ("File report" if selection else "Report from C")
    assert trace["route_handovers"] == [first, *rest] and usage["reason_code"] == "deadline_local"


@pytest.mark.parametrize("prior_terminal", ["selected", "pending_at_finalize"])
def test_reused_tool_context_does_not_inherit_previous_handover(episode, prior_terminal):
    def interim(state, _usage, _kwargs):
        if prior_terminal == "pending_at_finalize":
            write_owner_message(state.root, "deadline", TASK, kind=KIND_FINALIZE_NOW)
        return _text("First status")

    _text_before, _usage_before, before = episode.run([
        _read(), None, interim,
        _text("Still status") if prior_terminal == "pending_at_finalize" else finish("Still status")])
    # A forced end leaves the handover open on its row; the selection ended it.
    assert before["route_handovers"][0]["status"] == ("pending" if prior_terminal == "pending_at_finalize"
                                                      else "selected")
    text, usage, trace = episode.run([_text("Unrelated new task")], task_id="later-task")
    assert text == "Unrelated new task"
    assert not trace.get("route_handovers")
    assert not trace.get("authoring_handover_incomplete")
    assert usage.get("reason_code") != "authoring_handover_incomplete"


def test_owner_followup_holds_selection_until_the_successor_reads_again(episode, monkeypatch):
    from ouroboros.tools import review_result

    agent = SimpleNamespace(_owner_message_admission_lock=threading.Lock(),
                            _accepting_owner_messages=True, _busy=True, _current_task_id=TASK)
    episode.tools._ctx.owner_message_admission_lock = agent._owner_message_admission_lock
    episode.tools._ctx.owner_message_admission_agent = agent

    # The owner writes while the selected answer is at its delivery step, right before
    # the admission close drains the mailbox (the result review's step in the real path).
    at_delivery = review_result.result_review_at_delivery
    sent = []

    def followup_at_delivery(ctx, content, llm_trace, hold):
        if not sent and content == "Selected before the follow-up":
            assert llm_trace["route_handovers"][0]["status"] == "pending"
            write_owner_message(episode.root, "Read the evidence once more and finish.", TASK, msg_id="followup")
            sent.append(1)
        return at_delivery(ctx, content, llm_trace, hold)

    monkeypatch.setattr(review_result, "result_review_at_delivery", followup_at_delivery)

    def informed_final(state, _usage, _kwargs):
        source = state.tools._ctx._acceptance_observation["owner_source_sha256"]
        assert source in str(state.calls[-1]["messages"])
        return finish("Complete after follow-up", acceptance_subject={"owner_source_sha256": source})

    text, usage, trace = episode.run([
        _read(), None, _text("Initial WIP"), finish("Selected before the follow-up"), _read(), informed_final,
    ])
    assert text == "Complete after follow-up"
    assert sent == [1]
    assert [row["tool"] for row in trace["tool_calls"]] == ["read_file", "finish_task", "read_file", "finish_task"]
    assert len(_handover_notes(trace)) == 1
    assert any("Read the evidence once more" in str(row.get("content", ""))
               for row in episode.calls[-1]["messages"])
    [incident] = usage["authoring_handovers"]
    assert incident["work_observed"] and incident["status"] == "recovered" and "selection" not in incident


def test_audience_refusal_keeps_handover_open_for_the_admitted_choice(episode, monkeypatch):
    from ouroboros import place_reply

    prepare = place_reply.prepare_focus_reply

    def hold_first(ctx, text, *, quiet=False):
        if text == "First choice":
            assert ctx._authoring_handover["status"] == "pending"
            return False, "The destination held this reply."
        return prepare(ctx, text, quiet=quiet)

    monkeypatch.setattr(place_reply, "prepare_focus_reply", hold_first)

    def choose_quietly(state, _usage, _kwargs):
        assert state.tools._ctx._authoring_handover["status"] == "pending"
        return finish(silent=True)

    episode.tools._ctx.task_metadata = deepcopy(SCHEDULED_REPORT)
    text, usage, trace = episode.run([_read(), None, _text("Checking"), finish("First choice"), choose_quietly])
    assert text == "" and usage["wake_silent"] is True
    assert len(_handover_notes(trace)) == 1 and usage["authoring_handovers"][0]["selection"] == "silent"


@pytest.mark.parametrize("stay_quiet", [False, True])
def test_inherited_quiet_delivery_is_not_the_successors_choice(episode, monkeypatch, stay_quiet):
    from ouroboros.delivery_facts import noted
    from ouroboros import loop_nudges

    episode.tools._ctx.task_metadata = deepcopy(SCHEDULED_REPORT)
    real_notice = loop_nudges._claim_pre_delivery_notice

    def show_fact_once(tools, *args):
        ctx = tools._ctx
        if (getattr(ctx, "_completion_selected", None) or {}).get("wake_silent") and not ctx._pre_delivery_noticed:
            ctx._pre_delivery_noticed, ctx._completion_hold_cause = True, "pre_delivery_notice"
            return "A fact was shown once."
        return real_notice(tools, *args)

    monkeypatch.setattr(loop_nudges, "_claim_pre_delivery_notice", show_fact_once)

    def a_chooses_silence(*_args):
        monkeypatch.setenv("OUROBOROS_MODEL", SUCCESSOR)
        return finish("A's private record", silent=True)

    def b_replies(state, _usage, _kwargs):
        held = state.tools._ctx._delivery_candidate
        assert held.quiet and noted(held)
        return _text("B's draft")

    text, usage, trace = episode.run([_read(), a_chooses_silence, b_replies, finish("B's final", silent=stay_quiet)])
    assert text == "B's final" and bool(usage.get("wake_silent")) is stay_quiet
    assert len(_handover_notes(trace)) == 1
    _clean_selection(text, usage, trace, "silent" if stay_quiet else "answer")


def test_selected_handover_keeps_the_real_verification_nudge_and_its_record(episode):
    """The unexamined write returns the first selection and remains visible beside the accepted answer."""
    from ouroboros.outcomes import apply_receipt_absent_flag
    from tests._delivery_candidate_shared import holds

    write = _tool("write_file", path="report.md", content="Unchecked report")
    replacement = finish(answer_sha256=_sha("Selected first"))
    text, usage, trace = episode.run([
        write, None, _text("Initial WIP"), finish("Selected first"), replacement,
    ])
    assert text == "Selected first"
    assert trace["tool_calls"][0]["is_error"] is False
    assert len(_handover_notes(trace)) == 1
    assert [(row["reason"], row["subreason"]) for row in holds(episode.root)] == [
        ("finalization_nudge", "authoring_handover"), ("finalization_nudge", "pre_delivery_notice")]
    _clean_selection(text, usage, trace, "answer_sha256")
    outcome = derive_loop_outcome(text, usage, trace)
    apply_receipt_absent_flag(outcome, trace, episode.root, TASK, receipts=[])
    assert "receipt_absent" in outcome["outcome_axes"]["objective"]["warnings"]


def test_selected_handover_keeps_the_real_skill_readiness_nudge(episode):
    from tests.test_loop_skill_finalization import _write_self_authored_skill

    _write_self_authored_skill(episode.root)
    write = _tool("write_file", path="notes.txt", content="Authored skill notes",
                  root="skill_payload", bucket="external", skill_name="alpha")
    final = finish("Skill is authored but still needs review and enablement.")
    # The skill reminder comes first; the one pre-delivery notice (the write nothing examined) holds the next finish.
    text, usage, trace = episode.run([
        write, None, _text("Initial WIP"), finish("Selected first"), final, final,
    ])
    assert text == "Skill is authored but still needs review and enablement."
    assert trace["tool_calls"][0]["is_error"] is False
    assert (episode.root / "skills" / "external" / "alpha" / "notes.txt").read_text(encoding="utf-8") == "Authored skill notes"
    assert len(_handover_notes(trace)) == 1
    assert any("SKILL_NOT_FINALIZED" in str(row.get("content", ""))
               for row in episode.calls[-1]["messages"])
    _clean_selection(text, usage, trace, "answer")
