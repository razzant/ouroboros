from __future__ import annotations

import pathlib
import queue
from types import SimpleNamespace

import pytest

from ouroboros.project_dialogue import TASK_CAUSE_PHRASES


def test_progress_thought_keeps_full_content_and_existing_authorship():
    from ouroboros.agent import OuroborosAgent

    events = queue.Queue()
    agent = SimpleNamespace(
        _last_progress_ts=None,
        _event_queue=events,
        _current_chat_id=7,
        _current_task_id="thought-task",
        tools=SimpleNamespace(_ctx=SimpleNamespace()),
        _subagent_progress_meta=lambda _event: {},
    )
    thought = "long visible reasoning\n" + ("x" * 20_000)

    OuroborosAgent._emit_progress(agent, thought, narration=True)

    event = events.get_nowait()
    assert event["text"] == f"💬 {thought}"
    assert event["is_progress"] is True
    assert event["task_id"] == "thought-task"
    assert event["role"] == "assistant"
    assert event["system_type"] == "model_narration"


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


def test_project_completion_host_salvage_labels_its_bytes_and_points(tmp_path, monkeypatch):
    """The Main row over a salvaged task says what survived, in bounded form.

    It used to fall back to the neutral pointer copy, which read as "nothing
    here" over work that had actually been applied. The untruncated bytes still
    never enter the row: they stay with ``get_task_result``, and the pointer to
    them rides beside the excerpt rather than being displaced by it, which is
    the owner's answer (labelled excerpt PLUS pointer) in both forms.
    """
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.project_dialogue import (
        SALVAGE_EXCERPT_LABEL, enqueue_project_completion_summary,
    )

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
    text = queued[0]["text"]
    assert raw not in text
    # The cause speaks its human sentence (TASK_CAUSE_PHRASES twin), never the raw code.
    assert (f"The task could not get a usable model answer, so it could not finish. "
            f"{SALVAGE_EXCERPT_LABEL}: RAW PATCH") in text
    assert "provider_unavailable." not in text
    # The way back to the full copy is the row's Project reference (drawn from `project_id`), never a sentence.
    assert "Open the Project" not in text
    labelled = text.split(f"{SALVAGE_EXCERPT_LABEL}: ", 1)[1]
    assert len(labelled) <= 240 and labelled.endswith("…")


def test_provider_death_arms_always_carry_terminal_origin(monkeypatch):
    """Class contract (nanny-leaf sprint S2): every arm of the provider-death
    rail — including no-call early returns (context_overflow salvage,
    transport_unavailable_no_resend, provider_outcome_unknown_no_resend) —
    stamps a host-authored terminal provenance, so it can never publish as a
    model-authored final. Which one is a fact of the returned bytes: the host's
    own notice (nothing to salvage) is ``host_notice``, retained model output is
    ``host_salvage`` (12.10 incident: a notice was labelled preserved output)."""
    import pathlib
    from types import SimpleNamespace

    import ouroboros.loop as L

    def _ctx(kind, salvage=""):
        return SimpleNamespace(
            messages=[{"role": "user", "content": "do"}] + (
                [{"role": "assistant", "content": salvage}] if salvage else []),
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

    arms = (("provider_outcome_unknown", {"error_kind": "provider_outcome_unknown"}),  # the incident arm
            ("transport_unavailable", {"wait_cause": "transport_unavailable"}),
            ("context_overflow", {"error_kind": "context_overflow"}))
    for kind, kwargs in arms:
        text, usage, _tr = L._handle_provider_unavailable(_ctx(kind), **kwargs)
        assert usage.get("terminal_origin") == L.TERMINAL_ORIGIN_HOST_NOTICE, kind
        assert text.startswith("⚠️"), kind  # the host's own notice, nothing salvaged
        text, usage, _tr = L._handle_provider_unavailable(_ctx(kind, salvage="Partial model work."), **kwargs)
        assert usage.get("terminal_origin") == L.TERMINAL_ORIGIN_HOST_SALVAGE, kind
        assert text == "Partial model work.", kind


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


def test_the_round_one_budget_rejection_still_makes_no_call_and_gains_no_facts_block(monkeypatch):
    """A no-call rail never enters the forced prompt, so the typed facts block that
    rides ``_prepare_forced_prompt`` cannot turn it into an LLM call."""
    import ouroboros.loop as L

    def never(*_a, **_k):
        raise AssertionError("a no-call rail entered _prepare_forced_prompt")

    monkeypatch.setattr(L, "_prepare_forced_prompt", never)
    ctx = _rail_ctx(task_id="t-budget", round_idx=1)
    result = L._check_budget_limits(ctx, 0.0)
    assert result is not None
    text, usage, _trace = result
    assert text.startswith("🚫 Task rejected") and "[TASK_STATE_FACTS]" not in text
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
    stamps host_notice FIRST on the three no-call/no-resend provider-death arms;
    the rail still decides by the returned bytes, so a salvage is labelled
    ``host_salvage`` and the host's own notice stays ``host_notice``."""
    import ouroboros.loop as L

    monkeypatch.setattr(L, "call_llm_with_retry", lambda *a, **k: (None, 0.0))
    for kind, wait_cause in (
        ("provider_outcome_unknown", ""),
        ("provider_unavailable", "transport_unavailable"),
        ("context_overflow", ""),
    ):
        for salvage, origin in (("", L.TERMINAL_ORIGIN_HOST_NOTICE), ("Kept bytes.", L.TERMINAL_ORIGIN_HOST_SALVAGE)):
            ctx = _rail_ctx(task_id=f"t-{kind}")
            ctx.messages += [{"role": "assistant", "content": salvage}] if salvage else []
            ctx.accumulated_usage = {"terminal_origin": L.TERMINAL_ORIGIN_HOST_NOTICE}
            _text, usage, _trace = L._handle_provider_unavailable(
                ctx, error_kind=kind, wait_cause=wait_cause,
            )
            assert usage.get("terminal_origin") == origin, (kind, salvage)


def test_a_notice_speaks_for_itself_while_a_salvage_is_labelled():
    """The notice/salvage distinction survives, the blank card does not.

    A host NOTICE's own words ARE the answer, so they stand unlabelled. A
    SALVAGE is preserved intermediate output: dropping it left a bare headline
    and a reason code over work that had actually been applied, so the row now
    names what the bytes are beside the pointer both writers already append. A
    peer stop receipt reduces the row to the label alone only where that receipt
    actually landed: it goes to the task's OWN lineage chat, so a row written to
    any other chat has never seen it and keeps the bytes.
    """
    from ouroboros.project_dialogue import SALVAGE_EXCERPT_LABEL, _completion_excerpt

    assert _completion_excerpt({"result": "x", "terminal_origin": "host_notice"}) == "x"
    assert _completion_excerpt(
        {"result": "x", "terminal_origin": "host_salvage"},
    ) == f"{SALVAGE_EXCERPT_LABEL}: x"
    receipted = {
        "result": "x", "terminal_origin": "host_salvage", "chat_id": 7,
        "cancel_receipt": {"delivery_id": "cancel:t:1", "delivered_chat_id": 7},
    }
    assert _completion_excerpt(receipted, chat_id=7) == f"{SALVAGE_EXCERPT_LABEL}."
    legacy = {**receipted, "cancel_receipt": {"delivery_id": "cancel:t:1"}}
    assert _completion_excerpt(legacy, chat_id=7) == f"{SALVAGE_EXCERPT_LABEL}: x"
    # Another chat, and an unknown destination, both keep the bytes.
    assert _completion_excerpt(receipted, chat_id=1) == f"{SALVAGE_EXCERPT_LABEL}: x"
    assert _completion_excerpt(receipted) == f"{SALVAGE_EXCERPT_LABEL}: x"
    # Nothing to preserve is still nothing to label.
    assert _completion_excerpt({"result": "", "terminal_origin": "host_salvage"}) == ""


def test_a_non_provider_rail_with_a_complete_candidate_stays_model_final(tmp_path, monkeypatch):
    """The candidate branch only stamped provenance on the provider rail, so a
    round-limit stop that delivered the model's own complete answer went out
    unattributed. Host disclosures stay separately attributed and do not
    replace the model answer or its candidate identity."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    answer = "Complete current model answer."
    loop._replace_delivery_candidate(registry, limit_ctx, trace, answer, control="replace")
    monkeypatch.setattr(
        loop, "_plan_gate_disclosure",
        lambda *_a, **_k: "\n\nPlan review was left open.",
    )

    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert text == answer
    assert usage["terminal_host_notice"] == "Plan review was left open."
    assert usage["terminal_origin"] == loop.TERMINAL_ORIGIN_MODEL_FINAL
    assert usage["terminal_plan_review_open"] is True


def test_the_normal_rail_types_an_open_plan_review(tmp_path, monkeypatch):
    """The forced rails typed the fact; the normal rail left it to the single
    degraded_reason slot, which an unsettled child overwrites, so the plan fact
    survived only as prose. Now every rail that discloses types it, and a clean
    finalization stores no field at all (absent means "not stated")."""
    import queue

    from ouroboros.task_finalization import terminal_result_fields
    from tests.test_delivery_forced_finalization import _forced_test_context

    for suffix, expected in (("\n\nPlan review is still open.", True), ("", False)):
        loop, registry, limit_ctx, trace = _forced_test_context(tmp_path / str(expected))
        monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
        monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)
        monkeypatch.setattr(loop, "_plan_gate_disclosure", lambda *_a, **_k: suffix)
        text, usage, _trace = loop._no_tool_final_answer(
            "Complete answer.", limit_ctx, trace, registry, queue.Queue(), set(), lambda _t: None)
        assert text == "Complete answer."
        assert usage.get("terminal_plan_review_open", False) is expected  # absent = not open
        assert ("terminal_plan_review_open" in terminal_result_fields(usage)) is expected
        assert ("terminal_host_notice" in usage) is expected  # the typed fact never travels alone


# --- The plan clause speaks for the plan as it stands at THIS exit (map seam 3) -------------------------

PLANNED_ANSWER = "Complete answer."


def _withdraw(record):
    """The raiser's own later reply withdraws its objection (``plan_record.fold_objections``)."""
    seat = dict(record["asks"][0]["participants"][0],
                reply={"prose": "Addressed.", "objections": [], "withdrawn": [{"id": "o1", "why": "addressed"}]})
    record["asks"].append({"ask_id": "a2", "ts": "t2", "revision": 1, "message": "", "participants": [seat]})


def _planned_root(tmp_path, monkeypatch, *, owner_marked=True, unreadable=None, known=False):
    """A root's normal finalization whose plan carries one open objection of an owner-marked reviewer (or of an
    unmarked one); ``unreadable`` is a mutable ``[bool]``: while true the record reads as ``_plan_facts`` reports a
    store that refused, an empty record that cannot be judged (``known`` keeps the readable part's objection, as when
    only the earlier-schema state failed)."""
    from ouroboros import owner_hurry as oh
    from ouroboros import plan_record as pr
    from tests.test_delivery_forced_finalization import _forced_test_context
    from tests.test_owner_hurry_s3 import _plan_record

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)
    record = _plan_record("held")
    record["asks"][0]["participants"][0]["owner_marked"] = owner_marked
    monkeypatch.setattr(oh, "_plan_facts", lambda _ctx: (record if known else pr.empty_record(), {}, True)
                        if unreadable and unreadable[0] else (record, {}, False))
    candidate = loop._replace_delivery_candidate(registry, limit_ctx, trace, PLANNED_ANSWER, control="candidate")
    return loop, registry, limit_ctx, trace, record, candidate


STOPPED = TASK_CAUSE_PHRASES["author_stop"].rstrip(".")  # the clause; its words are review's
STILL_OPEN = "The plan review was still open when this answer was delivered"
UNKNOWN = "The plan record could not be read, so the state of the plan review is unknown."


def _recorded(text, usage, trace):
    """This exit as the pipeline records it: its execution axis, and the one card line that both the stored row
    (with the result-only fields) and the live frame (axes alone) read through the host card reader."""
    from ouroboros.outcomes import derive_loop_outcome, normalize_outcome_axes
    from ouroboros.project_dialogue import _completion_verdict
    from ouroboros.task_finalization import terminal_result_fields

    outcome = derive_loop_outcome(text, usage, trace)
    live = {"status": "completed", "reason_code": outcome["reason_code"],
            "outcome_axes": normalize_outcome_axes({"outcome_axes": outcome["outcome_axes"]})}
    lines = {_completion_verdict({**live, **terminal_result_fields(usage)}, {}), _completion_verdict({}, live)}
    assert len(lines) == 1, f"the stored row and the live frame read differently: {lines}"
    return outcome["outcome_axes"]["execution"], lines.pop()


@pytest.mark.parametrize(("prior", "exit_by"), [
    ("", "selection"),
    ("invalid_delivery_control_after_repair", "round_limit"),
])
def test_a_withdrawn_objection_leaves_no_plan_clause_on_the_answer_held_before_it(tmp_path, monkeypatch,
                                                                                  exit_by, prior):
    """The answer is disclosed with an open objection and its result review holds it; the raiser then
    withdraws, and the SAME candidate ends the task (selected again, or kept by a round-limit rail). The typed
    flag, the notice, the candidate's plan mark, the outcome class and the forced call's facts follow the plan at
    that exit; another owner's degradation of the candidate is neither written over nor cleared."""
    import queue

    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.project_dialogue import _terminal_limitations
    from ouroboros.task_finalization import terminal_result_fields
    from tests.test_delivery_forced_finalization import _select_completion

    loop, registry, limit_ctx, trace, record, candidate = _planned_root(tmp_path, monkeypatch)
    candidate.degraded, candidate.degraded_reason = bool(prior), prior
    holds = iter([True, False])
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", lambda *_a, **_k: next(holds))
    assert loop._no_tool_final_answer(
        PLANNED_ANSWER, limit_ctx, trace, registry, queue.Queue(), set(), lambda _t: None) is None
    held = limit_ctx.accumulated_usage
    assert held["terminal_plan_review_open"] is True and "triad_a/o1" in held["terminal_host_notice"]
    assert candidate.degraded_reason == (prior or "plan_review_advisory")
    _withdraw(record)
    prompts: list = []
    if exit_by == "selection":
        _select_completion(registry, limit_ctx, trace, answer_sha256=candidate.content_sha256)
        text, usage, trace = loop._no_tool_final_answer(PLANNED_ANSWER, limit_ctx, trace, registry, queue.Queue(),
                                                        set(), lambda _t: None, explicit_candidate=True)
        assert (candidate.degraded, candidate.degraded_reason) == (bool(prior), prior)
    else:
        monkeypatch.setattr(loop, "call_llm_with_retry", lambda _llm, messages, *_a, **_k: (
            prompts.append(str(messages)) or {"content": PLANNED_ANSWER}, 0.0))
        text, usage, trace = loop._handle_round_limit(limit_ctx)
    assert text == PLANNED_ANSWER and registry._ctx._delivery_candidate.content_sha256 == candidate.content_sha256
    assert not any("plan_review_open" in prompt for prompt in prompts), "the forced call reads the plan now"
    assert "terminal_plan_review_open" not in usage and "terminal_plan_review_open" not in terminal_result_fields(usage)
    assert "triad_a/o1" not in str(usage.get("terminal_host_notice") or "")
    outcome = derive_loop_outcome(text, usage, trace)
    assert "plan_review" not in outcome["outcome_axes"]["execution"]
    stored = {**terminal_result_fields(usage), "outcome_axes": outcome["outcome_axes"]}
    assert not any(_terminal_limitations(stored, {}, outcome["reason_code"])), "no stale limitation on the card"


@pytest.mark.parametrize(("mode", "stop"), [("pro", False), ("cyber_pro", True)])
def test_an_authors_stop_over_objections_that_hold_nothing_never_says_the_work_proceeded(
        tmp_path, monkeypatch, mode, stop):
    """No Access holds an answer over an owner-marked objection. Delivered, the notice and the card say the work
    went on with it; after the author's stop, the notice names the objection and the stop, never work that
    proceeded, and the card reads the stop with the review still open."""
    import queue

    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES
    from tests.test_delivery_forced_finalization import _select_completion

    cyber = mode == "cyber_pro"
    monkeypatch.setattr("ouroboros.tools.review_helpers.cyber_review_authority", lambda: cyber)
    loop, registry, limit_ctx, trace, _record, candidate = _planned_root(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", lambda *_a, **_k: False)
    if stop:
        _select_completion(registry, limit_ctx, trace, answer_sha256=candidate.content_sha256, action="stop",
                           rationale="Stopping with the draft.")
    text, usage, trace = loop._no_tool_final_answer(PLANNED_ANSWER, limit_ctx, trace, registry, queue.Queue(), set(),
                                                    lambda _t: None, explicit_candidate=stop)
    listed = "⚠️ Plan review: 1 open objection remains open — triad_a/o1 (ask a1, revision 1): no; "
    why = ("Ouroboros stopped with the work unfinished" if stop else "Cyber Pro: Ouroboros proceeded by its own judgment"
           if cyber else "work proceeded with them open")
    assert text == PLANNED_ANSWER and usage["terminal_host_notice"] == f"{listed}{why}."
    execution, line = _recorded(text, usage, trace)
    assert execution["plan_review"] == "objections_open"
    assert line == (f"{STOPPED} · Stopping with the draft · {STILL_OPEN}." if stop
                    else TASK_CAUSE_PHRASES["plan_review_objections_open"])


@pytest.mark.parametrize("mode, exit_by", [("pro", "stop"), ("cyber_pro", "stop"), ("pro", "round_limit")])
def test_an_unreadable_plan_record_names_the_authors_stop_and_leaves_other_exits_as_they_were(
        tmp_path, monkeypatch, mode, exit_by):
    """An unreadable record holds a normal answer outside Cyber Pro (its reminder unchanged). The author's stop then
    says the record could not be read, the review's state is unknown and the work stopped unfinished; a rail keeps
    its own wording, and Cyber Pro, which never holds, still discloses nothing. Every exit records the plan review's
    class as unreadable, never the objections a readable part still names, and its card keeps the state unknown:
    beside the author's stop (never work that went on) or the rail's own cause."""
    import queue

    from tests.test_delivery_forced_finalization import _select_completion

    cyber = mode == "cyber_pro"
    monkeypatch.setattr("ouroboros.tools.review_helpers.cyber_review_authority", lambda: cyber)
    loop, registry, limit_ctx, trace, _record, candidate = _planned_root(
        tmp_path, monkeypatch, unreadable=[True], known=exit_by == "stop_known")
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", lambda *_a, **_k: False)
    if not cyber:
        assert loop._no_tool_final_answer(
            PLANNED_ANSWER, limit_ctx, trace, registry, queue.Queue(), set(), lambda _t: None) is None
        assert "plan record cannot be read" in str(limit_ctx.messages[-1]["content"]), "the gate's own hold"
    if exit_by == "round_limit":
        monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: ({"content": PLANNED_ANSWER}, 0.0))
        text, usage, trace = loop._handle_round_limit(limit_ctx)
        assert usage["terminal_host_notice"].startswith(
            "⚠️ Plan review state is unknown (the plan record could not be read); the task-wide rail `round_limit`")
        execution, line = _recorded(text, usage, trace)
        assert execution.get("plan_review") == "unreadable"
        assert line == f"The task hit its round limit before it could finish cleanly · {UNKNOWN}"
        return
    _select_completion(registry, limit_ctx, trace, answer_sha256=candidate.content_sha256, action="stop",
                       rationale="The plan record cannot be read; stopping with the draft.")
    text, usage, trace = loop._no_tool_final_answer(PLANNED_ANSWER, limit_ctx, trace, registry, queue.Queue(), set(),
                                                    lambda _t: None, explicit_candidate=True)
    notice = str(usage.get("terminal_host_notice") or "")
    assert text == PLANNED_ANSWER and trace["task_completion"]["action"] == "stop"
    execution, line = _recorded(text, usage, trace)
    assert execution.get("plan_review") == "unreadable", "the plan review's state is unknown, not its readable part"
    assert line == f"{STOPPED} · The plan record cannot be read; stopping with the draft · {UNKNOWN}"
    if cyber:
        assert notice == "" and "terminal_plan_review_open" not in usage
        return
    beyond = " beyond 1 open objection: triad_a/o1 (ask a1, revision 1): no" if exit_by == "stop_known" else ""
    assert notice == ("⚠️ Plan review: the plan record could not be read, so the state of the plan review is unknown"
                      f"{beyond}; Ouroboros stopped with the work unfinished.")
    assert usage["terminal_plan_review_open"] is True, "typed as the disclosed suffix, as the rail's unreadable one"


@pytest.mark.parametrize("exit_by", ["normal", "round_limit"])
def test_a_normal_answer_after_a_transient_unreadable_read_claims_no_stop(tmp_path, monkeypatch, exit_by):
    """A refused store read held one normal answer of a task with no plan; the next read finds no plan, so no gate
    decides the next exit (a normal answer, or a rail's), and the earlier hold's projection no longer speaks for it:
    nothing names a stop or an unknown plan, in the notice, the recorded class or the card."""
    import queue

    unreadable = [True]
    loop, registry, limit_ctx, trace, record, _candidate = _planned_root(tmp_path, monkeypatch, unreadable=unreadable)
    record.update(revisions=[], asks=[])  # no plan at all once the store reads again
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", lambda *_a, **_k: False)
    assert loop._no_tool_final_answer(
        PLANNED_ANSWER, limit_ctx, trace, registry, queue.Queue(), set(), lambda _t: None) is None
    assert trace["plan_gate"]["status"] == "unreadable", "the hold's own projection"
    unreadable[0] = False
    if exit_by == "round_limit":
        monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: ({"content": PLANNED_ANSWER}, 0.0))
        text, usage, trace = loop._handle_round_limit(limit_ctx)
    else:
        text, usage, trace = loop._no_tool_final_answer(
            PLANNED_ANSWER, limit_ctx, trace, registry, queue.Queue(), set(), lambda _t: None)
    assert text == PLANNED_ANSWER and "plan_gate" not in trace, "no earlier hold's projection speaks for this exit"
    assert "terminal_host_notice" not in usage and "terminal_plan_review_open" not in usage
    execution, line = _recorded(text, usage, trace)
    assert "plan_review" not in execution
    assert line == ("The task hit its round limit before it could finish cleanly." if exit_by == "round_limit" else "")


def test_an_unreadable_record_with_a_pending_ask_stays_unknown_not_merely_awaited(tmp_path, monkeypatch):
    """A readable pending ask plus unreadable legacy plan state may finish in Cyber Pro; the actual reducer
    names both facts, and the recorded/card projection must keep the unreadable state."""
    import queue

    from ouroboros import owner_hurry
    from tests.test_owner_hurry_s3 import _plan_record

    monkeypatch.setattr("ouroboros.tools.review_helpers.cyber_review_authority", lambda: True)
    loop, registry, limit_ctx, trace, _record, _candidate = _planned_root(tmp_path, monkeypatch)
    monkeypatch.setattr(owner_hurry, "_plan_facts", lambda _ctx: (_plan_record("awaiting"), {}, True))
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", lambda *_a, **_k: False)
    text, usage, trace = loop._no_tool_final_answer(
        PLANNED_ANSWER, limit_ctx, trace, registry, queue.Queue(), set(), lambda _t: None)
    assert trace["plan_gate"]["review_only_awaited"] is True and trace["plan_gate"]["unreadable"] is True
    execution, line = _recorded(text, usage, trace)
    assert execution["plan_review"] == "unreadable" and line == UNKNOWN


def test_a_candidateless_fallback_types_the_open_plan_review(tmp_path, monkeypatch):
    """A host-notice fallback (no delivery candidate) disclosed the open review in
    prose while nothing typed carried it; the stamp now happens for both arms."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (None, 0.0))
    monkeypatch.setattr(loop, "_plan_gate_disclosure", lambda *_a, **_k: "\n\nPlan review is still open.")
    _text, usage, _returned = loop._handle_round_limit(limit_ctx)
    assert usage["terminal_origin"] == loop.TERMINAL_ORIGIN_HOST_NOTICE
    assert usage["terminal_plan_review_open"] is True
    assert usage["terminal_host_notice"].startswith("Plan review is still open.")


def test_a_non_provider_rail_without_a_candidate_is_a_host_notice(tmp_path, monkeypatch):
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (None, 0.0))

    _text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert usage["terminal_origin"] == loop.TERMINAL_ORIGIN_HOST_NOTICE


@pytest.mark.parametrize("late_case", ["unknown_outcome", "second_directive"])
def test_a_host_control_notice_on_the_forced_call_stays_host_notice(tmp_path, monkeypatch, late_case):
    """Triad finding (12.10): owner input that arrives while the provider rail's one
    forced call runs ends in the host's own control sentence, never model output. It is
    ``host_notice`` (its words are the row), not a "preserved intermediate output" receipt."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    incoming = queue.Queue()
    loop, _registry, ctx, _trace = _forced_test_context(tmp_path, incoming=incoming)
    ctx.accumulated_usage["_last_llm_error_kind"] = "llm_empty_response"
    calls = []

    def forced_model(*_args, **_kwargs):
        calls.append(1)
        incoming.put("Additional owner directive")
        if late_case == "unknown_outcome":
            ctx.accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
            return {"role": "assistant", "content": ""}, 0.0
        return {"role": "assistant", "content": "Draft superseded by owner input"}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)
    text, usage, _trace = loop._handle_provider_unavailable(ctx, error_kind="llm_empty_response")
    assert text.startswith("⚠️ ") and "Resume the task" in text
    assert len(calls) == (1 if late_case == "unknown_outcome" else 2)
    assert usage["terminal_origin"] == "host_notice", (text, usage["terminal_origin"])
