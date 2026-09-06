from __future__ import annotations

import pathlib
import queue
from types import SimpleNamespace


def test_progress_thought_keeps_full_content_and_existing_authorship():
    from ouroboros.agent import OuroborosAgent

    events = queue.Queue()
    agent = SimpleNamespace(
        _last_progress_ts=None,
        _event_queue=events,
        _current_chat_id=7,
        _current_task_id="thought-task",
        tools=SimpleNamespace(_ctx=SimpleNamespace(is_ephemeral_turn=False)),
        _subagent_progress_meta=lambda _event: {},
    )
    thought = "long visible reasoning\n" + ("x" * 20_000)

    OuroborosAgent._emit_progress(agent, thought)

    event = events.get_nowait()
    assert event["text"] == f"💬 {thought}"
    assert event["is_progress"] is True
    assert event["task_id"] == "thought-task"
    assert "role" not in event
    assert "system_type" not in event


def test_normal_model_response_stamps_model_final_origin():
    from ouroboros.loop import _handle_text_response

    usage = {}
    text, returned_usage, _trace = _handle_text_response(
        "A complete answer", {"reasoning_notes": []}, usage,
    )
    assert text == "A complete answer"
    assert returned_usage["terminal_origin"] == "model_final"


def test_terminal_result_fields_carry_open_plan_review_for_model_final():
    from ouroboros.task_finalization import terminal_result_fields

    assert terminal_result_fields({
        "terminal_origin": "model_final",
        "terminal_plan_review_open": True,
    }) == {
        "terminal_origin": "model_final",
        "terminal_plan_review_open": True,
    }


def test_terminal_projection_preserves_model_and_legacy_but_receipts_host_salvage(tmp_path):
    from supervisor.terminal_delivery import (
        delivery_id_for,
        project_terminal_result_event,
    )

    raw = "RAW PATCH " * 1000
    base = {
        "type": "send_message",
        "chat_id": 7,
        "task_id": "terminal-a",
        "text": raw,
        "format": "markdown",
    }
    host = project_terminal_result_event(
        tmp_path, {"chat_id": 7}, "terminal-a",
        result_text=raw, terminal_origin="host_salvage", base_event=base,
    )
    assert host["role"] == "system"
    assert host["system_type"] == "terminal_incident"
    assert host["task_id"] == "terminal-a"
    assert host["terminal_origin"] == "host_salvage"
    assert raw not in host["text"]
    assert "model-provider outage" in host["text"]
    assert "terminal-a" not in host["text"]
    assert "task details" in host["text"]
    assert host["delivery_id"] == delivery_id_for("terminal-a", raw)

    model = project_terminal_result_event(
        tmp_path, {"chat_id": 7}, "terminal-a",
        result_text=raw, terminal_origin="model_final", base_event=base,
    )
    assert model["text"] == raw
    assert model["terminal_origin"] == "model_final"
    assert "role" not in model and "system_type" not in model

    legacy = project_terminal_result_event(
        tmp_path, {"chat_id": 7}, "terminal-a",
        result_text=raw, terminal_origin=None, base_event=base,
    )
    assert legacy == {**base, "delivery_id": delivery_id_for("terminal-a", raw)}


def test_host_salvage_live_outbox_and_replay_use_same_projection(tmp_path, monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    from supervisor import state as supervisor_state
    from supervisor.terminal_delivery import build_completed_result_event, pending_deliveries

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setattr(
        pipeline, "_derive_host_bound_loop_outcome",
        lambda *_a, **_k: {
            "outcome_axes": {
                "execution": {"status": "infra_failed"},
                "objective": {"status": "not_evaluated"},
                "review": {"status": "not_evaluated"},
                "artifacts": {"status": "not_applicable"},
            },
            "reason_code": "provider_unavailable",
        },
    )
    monkeypatch.setattr(pipeline, "apply_receipt_absent_flag", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_root_post_task_already_completed", lambda *_a, **_k: False)
    observed = {}
    monkeypatch.setattr(
        pipeline, "_run_post_task_processing_async",
        lambda *_a, **kwargs: observed.update(sealed=kwargs.get("sealed_final")),
    )
    monkeypatch.setattr(supervisor_state, "reconstruct_task_cost", lambda *_a, **_k: {
        "cost_accounting_status": "available", "cost_final": True,
        "cost_usd": 0.0, "total_rounds": 1, "prompt_tokens": 1,
        "completion_tokens": 1, "reserved_usd": 0.0,
        "unresolved_upper_bound_usd": 0.0, "unknown_unmetered": 0,
    })
    raw = "RAW_TOOL_ENVELOPE\n" + ("x" * 20000)
    usage = {
        "rounds": 1,
        "cost": 0.0,
        "reason_code": "provider_unavailable",
        "terminal_origin": "host_salvage",
    }
    pending = []
    pipeline.emit_task_results(
        env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
        memory=object(), llm=object(), pending_events=pending,
        task={"id": "host-live", "type": "task", "chat_id": 7, "text": "do it"},
        text=raw, usage=usage,
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0, drive_logs=logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )

    live = pending[0]
    assert live["role"] == "system"
    assert live["system_type"] == "terminal_incident"
    assert raw not in live["text"]
    preserved = pathlib.Path(usage["terminal_salvage_path"])
    assert preserved.read_text(encoding="utf-8") == raw
    (owed,) = pending_deliveries(tmp_path)
    assert owed["text"] == live["text"]
    assert owed["role"] == live["role"]
    assert owed["system_type"] == live["system_type"]
    assert owed["delivery_id"] == live["delivery_id"]

    replay = build_completed_result_event(
        tmp_path, {"chat_id": 7}, "host-live",
        {"result": raw, "terminal_origin": "host_salvage",
         "terminal_salvage_path": str(preserved)},
    )
    assert replay is not None
    assert replay["text"] == live["text"]
    assert replay["role"] == live["role"]
    assert replay["system_type"] == live["system_type"]
    assert replay["delivery_id"] == live["delivery_id"]
    assert observed["sealed"]["final_result_text"] == live["text"]
    assert raw not in observed["sealed"]["final_result_text"]


def test_task_result_persists_origin_and_full_host_salvage(tmp_path):
    import ouroboros.agent_task_pipeline as pipeline

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    raw = "full raw salvage"
    path = tmp_path / "observability" / "salvaged" / "persisted.txt"
    path.parent.mkdir(parents=True)
    path.write_text(raw, encoding="utf-8")
    pipeline._store_task_result(
        env,
        {"id": "persisted", "root_task_id": "persisted", "type": "task"},
        raw,
        {"rounds": 1, "cost": 0.0, "terminal_origin": "host_salvage",
         "terminal_salvage_path": str(path)},
        {"tool_calls": [], "reasoning_notes": []},
    )
    stored = pipeline.load_task_result(tmp_path, "persisted")
    assert stored["result"] == raw
    assert stored["terminal_origin"] == "host_salvage"
    assert stored["terminal_salvage_path"] == str(path)


def test_project_completion_host_salvage_uses_neutral_details_copy(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.project_dialogue import enqueue_project_completion_summary

    project = create_project(tmp_path, "salvage", name="Salvage Project")
    bind_task_to_project(
        tmp_path, "salvage-root", project["id"], project["chat_id"],
        origin={"absent": "system"},
    )
    queued = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda _root, event, **_kwargs: queued.append(dict(event)) or True,
    )
    raw = "RAW PATCH " * 100
    task = {
        "id": "salvage-root", "chat_id": project["chat_id"],
        "project_id": project["id"], "title": "Interrupted task",
    }
    result = {
        **task, "task_id": "salvage-root", "status": "failed",
        "reason_code": "provider_unavailable", "result": raw,
        "terminal_origin": "host_salvage",
    }
    assert enqueue_project_completion_summary(
        tmp_path, {}, "salvage-root", task, result,
        {"status": "failed", "reason_code": "provider_unavailable"},
    )
    assert len(queued) == 1
    assert raw not in queued[0]["text"]
    assert queued[0]["text"].endswith("Open the Project for details.")


def test_provider_death_arms_always_carry_terminal_origin(monkeypatch):
    """Class contract (nanny-leaf sprint S2): every arm of the provider-death
    rail — including no-call early returns (context_overflow salvage,
    transport_unavailable_no_resend, provider_outcome_unknown_no_resend) —
    stamps a terminal provenance, so host salvage can never publish as a
    model-authored final. Explicit stamps (retained MODEL_FINAL under the
    retry wall) stay authoritative via setdefault."""
    import pathlib
    from types import SimpleNamespace

    import ouroboros.loop as L

    def _ctx(kind):
        return SimpleNamespace(
            messages=[{"role": "user", "content": "do"}],
            llm=None, active_model="m", active_effort="low", max_retries=1,
            drive_logs=pathlib.Path("/tmp"), task_id="t-origin", round_idx=1,
            event_queue=None, accumulated_usage={
                # The round gate always leaves the failing round's typed facts
                # on the usage record before the rail is entered.
                "_last_llm_error_kind": kind,
                "execution_status": "infra_failed", "reason_code": "llm_api_error",
            },
            task_type="", active_use_local=False, max_rounds=10, deadline_ts=None,
            drive_root=None, budget_drive_root=None, root_task_id="", tools=None,
            llm_trace={},
        )

    monkeypatch.setattr(L, "call_llm_with_retry", lambda *a, **k: (None, 0.0))

    # provider_outcome_unknown no-resend arm (the incident arm).
    _t, usage, _tr = L._handle_provider_unavailable(
        _ctx("provider_outcome_unknown"), error_kind="provider_outcome_unknown")
    assert usage.get("terminal_origin") == L.TERMINAL_ORIGIN_HOST_SALVAGE

    # transport-wait no-resend arm.
    _t, usage, _tr = L._handle_provider_unavailable(
        _ctx("transport_unavailable"), wait_cause="transport_unavailable")
    assert usage.get("terminal_origin") == L.TERMINAL_ORIGIN_HOST_SALVAGE

    # context-overflow salvage arm.
    _t, usage, _tr = L._handle_provider_unavailable(
        _ctx("context_overflow"), error_kind="context_overflow")
    assert usage.get("terminal_origin") == L.TERMINAL_ORIGIN_HOST_SALVAGE


def test_deadline_grace_final_is_never_stamped_host_salvage(monkeypatch):
    """Panel blocker (sol/grok/fable, lane B F1): the deadline grace arm can
    return a MODEL-AUTHORED final (``deadline_local``, provider_terminal=False).
    The wrapper must not stamp it HOST_SALVAGE — delivery would replace the
    model's answer with the generic outage receipt."""
    import pathlib
    from types import SimpleNamespace

    import ouroboros.loop as L

    ctx = SimpleNamespace(
        messages=[{"role": "user", "content": "do"}],
        llm=None, active_model="m", active_effort="low", max_retries=1,
        drive_logs=pathlib.Path("/tmp"), task_id="t-deadline", round_idx=1,
        event_queue=None,
        accumulated_usage={"_last_llm_error_kind": "deadline_exhausted"},
        task_type="", active_use_local=False, max_rounds=10, deadline_ts=None,
        drive_root=None, budget_drive_root=None, root_task_id="", tools=None,
        llm_trace={},
    )
    monkeypatch.setattr(
        L, "call_llm_with_retry",
        lambda *a, **k: ({"role": "assistant", "content": "Best answer before deadline."}, 0.0),
    )
    text, usage, _tr = L._handle_provider_unavailable(
        ctx, error_kind="deadline_exhausted")
    assert "Best answer before deadline." in text
    assert usage.get("reason_code") == "deadline_local"
    assert usage.get("terminal_origin") != L.TERMINAL_ORIGIN_HOST_SALVAGE


def test_budget_and_round_limit_rails_stamp_host_notice(monkeypatch):
    """INTENTIONAL behaviour change, not a bug fix of the test: these rails used
    to leave terminal_origin absent because they never enter the provider-death
    wrapper, so "missing" meant both "written before the stamp existed" and "the
    host wrote this text alone". The forced-finalization sink now stamps
    host_notice, and a missing origin identifies only legacy rows."""
    import pathlib
    from types import SimpleNamespace

    import ouroboros.loop as L

    ctx = SimpleNamespace(
        messages=[{"role": "user", "content": "do"}],
        llm=None, active_model="m", active_effort="low", max_retries=1,
        drive_logs=pathlib.Path("/tmp"), task_id="t-rails", round_idx=11,
        event_queue=None, accumulated_usage={},
        task_type="", active_use_local=False, max_rounds=10, deadline_ts=None,
        drive_root=None, budget_drive_root=None, root_task_id="", tools=None,
        llm_trace={},
    )
    monkeypatch.setattr(L, "call_llm_with_retry", lambda *a, **k: (None, 0.0))
    _t, usage, _tr = L._handle_round_limit(ctx)
    assert usage.get("terminal_origin") == L.TERMINAL_ORIGIN_HOST_NOTICE


def _rail_ctx(task_id="t-rail", round_idx=11):
    import pathlib
    from types import SimpleNamespace

    return SimpleNamespace(
        messages=[{"role": "user", "content": "do"}],
        llm=None, active_model="m", active_effort="low", max_retries=1,
        drive_logs=pathlib.Path("/tmp"), task_id=task_id, round_idx=round_idx,
        event_queue=None, accumulated_usage={},
        task_type="", active_use_local=False, max_rounds=10, deadline_ts=None,
        drive_root=None, budget_drive_root=None, root_task_id="", tools=None,
        llm_trace={},
    )


def test_budget_rejection_before_any_work_is_a_host_notice(monkeypatch):
    """The host wrote this text alone, so it must be attributable — and it must
    be published verbatim rather than replaced by the outage receipt."""
    import ouroboros.loop as L

    ctx = _rail_ctx(task_id="t-budget", round_idx=1)
    result = L._check_budget_limits(ctx, 0.0)
    assert result is not None
    text, usage, _trace = result
    assert text.startswith("🚫 Task rejected")
    assert usage["terminal_origin"] == L.TERMINAL_ORIGIN_HOST_NOTICE
    assert usage["reason_code"] == "budget_exhausted"


def test_a_host_notice_publishes_its_own_words_with_its_markdown(tmp_path):
    """A notice is NOT salvage: replacing its text with the outage receipt would
    name the wrong cause, and dropping its markdown would render the host's own
    code spans as escaped plain text. It also carries no system_type, which is
    what lets a replayed task card conclude on it."""
    from supervisor.terminal_delivery import project_terminal_result_event

    text = "🚫 Task rejected. Total budget exhausted.\n\nPlan review left open: `plan_review_advisory`."
    base = {
        "type": "send_message", "chat_id": 7, "task_id": "terminal-n",
        "text": text, "format": "markdown", "log_text": "kept",
    }
    notice = project_terminal_result_event(
        tmp_path, {"chat_id": 7}, "terminal-n",
        result_text=text, terminal_origin="host_notice", base_event=dict(base),
    )
    assert notice["text"] == text
    assert "model-provider outage" not in notice["text"]
    assert notice["role"] == "system"
    assert notice["terminal_origin"] == "host_notice"
    assert "system_type" not in notice
    assert notice["format"] == "markdown"
    assert notice["log_text"] == "kept"


def test_the_durable_result_carries_the_third_producer_word():
    from ouroboros.task_finalization import terminal_result_fields

    assert terminal_result_fields({"terminal_origin": "host_notice"})["terminal_origin"] == "host_notice"
    assert terminal_result_fields({"terminal_origin": "model_final"})["terminal_origin"] == "model_final"
    # An unknown producer stays legacy rather than acquiring a word.
    assert "terminal_origin" not in terminal_result_fields({"terminal_origin": "something_new"})


def test_provider_death_arms_are_not_downgraded_by_the_forced_sink(monkeypatch):
    """Regression for the sink's interaction with the provider rail: the sink
    stamps host_notice FIRST on the three no-call/no-resend provider-death arms,
    so a plain setdefault in the wrapper would have become a no-op and a
    provider-death salvage would have published verbatim."""
    import ouroboros.loop as L

    monkeypatch.setattr(L, "call_llm_with_retry", lambda *a, **k: (None, 0.0))
    for kind, wait_cause in (
        ("provider_outcome_unknown", ""),
        ("provider_unavailable", "transport_unavailable"),
        ("context_overflow", ""),
    ):
        ctx = _rail_ctx(task_id=f"t-{kind}")
        ctx.accumulated_usage = {"terminal_origin": L.TERMINAL_ORIGIN_HOST_NOTICE}
        _text, usage, _trace = L._handle_provider_unavailable(
            ctx, error_kind=kind, wait_cause=wait_cause,
        )
        assert usage.get("terminal_origin") == L.TERMINAL_ORIGIN_HOST_SALVAGE, kind


def test_a_notice_keeps_its_completion_excerpt_unlike_a_salvage():
    """Only salvage hides its text behind the neutral details copy."""
    from ouroboros.project_dialogue import _completion_excerpt

    assert _completion_excerpt({"result": "x", "terminal_origin": "host_notice"}) == "x"
    assert _completion_excerpt({"result": "x", "terminal_origin": "host_salvage"}) == ""


def test_a_non_provider_rail_with_a_complete_candidate_stays_model_final(tmp_path, monkeypatch):
    """The candidate branch only stamped provenance on the provider rail, so a
    round-limit stop that delivered the model's own complete answer went out
    unattributed. The ordinary no-tool final composes the same disclosure glue
    and stays model_final; this rail now says the same thing, and the glue is
    still appended to the model's text."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    answer = "Complete current model answer."
    loop._replace_delivery_candidate(registry, limit_ctx, trace, answer, control="replace")
    monkeypatch.setattr(
        loop, "_force_plan_disclosure",
        lambda *_a, **_k: "\n\nPlan review was left open.",
    )

    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert text.startswith(answer)
    assert text.endswith("Plan review was left open.")
    assert usage["terminal_origin"] == loop.TERMINAL_ORIGIN_MODEL_FINAL
    assert usage["terminal_plan_review_open"] is True


def test_a_non_provider_rail_without_a_candidate_is_a_host_notice(tmp_path, monkeypatch):
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (None, 0.0))

    _text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert usage["terminal_origin"] == loop.TERMINAL_ORIGIN_HOST_NOTICE
