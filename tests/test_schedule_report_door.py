"""An older schedule's confirmed run selects its held report through a narrow adapter of the old completion (#3687).

Its template copied a frozen Presence ceiling and speaker by value: ``finish_task`` is outside that ceiling and no
current registry publishes ``presence_finish``. Only a run its own admission receipt confirms gets that name back,
as the ordinary staged completion of a quiet record; every other task, chat and turn keeps today's door.
"""
import hashlib
import json
import queue
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop
from ouroboros.loop_completion_schema import completion_schema, refused_door_end
from ouroboros.presence_authority import (
    PresenceArgumentBinding, PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload,
)
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolRegistry
from tests.test_schedule_occurrence import q  # noqa: F401 -- the scheduler's own admission fixture

TASK, SCHEDULE, TOKEN = "sched-run", "reference-weekly", "occ-token-1"
REPORT = "Reference report checked: no confirmed changes."
REVISED_REPORT = "Reference report checked: one pending item."
SELECTOR = hashlib.sha256(REPORT.encode()).hexdigest()
SPEAKER = {"binding_id": "1" * 32, "transport_skill": "slack", "place_id": "place-1",
           "event": {"conversation_key": "slack:acct:C1:0", "provider": "slack", "conversation_id": "C1"}}


def ceiling(*grants):
    draft = PresenceCapabilityCeiling(
        skill_name="slack-place", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=10,
        tool_grants=(PresenceToolGrant("chat_history"), *grants), resource_grants=(), digest="0" * 64)
    return presence_ceiling_payload(draft)


def admit(root, task_id=TASK, *, token=TOKEN, status="accepted", chat_id=1):
    """The receipt the scheduler writes for one accepted occurrence (``schedule_occurrence._write_receipt``)."""
    occurrence = {"schedule_id": SCHEDULE, "token": token, "due_at": "2026-11-17T06:00:00+00:00"}
    frozen = {"id": task_id, "text": "Check the reference report.",
              "metadata": {"schedule_occurrence": occurrence, "presence": SPEAKER}}
    write_task_result(root, task_id, "running", root_task_id=task_id, delegation_role="root", chat_id=chat_id,
                      schedule_admission={"schedule_id": SCHEDULE, "token": token, "due_at": occurrence["due_at"],
                                          "status": status, "dispatch": "possible", "task": frozen})


def run_metadata(task_id=TASK, *, token=TOKEN, speaker=SPEAKER):
    return {"presence": json.loads(json.dumps(speaker)), "schedule_occurrence": {"schedule_id": SCHEDULE, "token": token},
            "root_task_id": task_id, "delegation_role": "root"}


def bind(registry, *, task_id=TASK, metadata=None, contract=None):
    ctx = registry._ctx
    ctx.task_id = task_id
    ctx.task_metadata = run_metadata(task_id) if metadata is None else metadata
    ctx.task_contract = {"capability_ceiling": ceiling()} if contract is None else contract
    registry.set_context(ctx)
    return ctx


def safety_calls(monkeypatch, *verdicts):
    """The Safety Supervisor's answers, in order (allow once none is left), and what it was asked."""
    calls, answers = [], iter(verdicts)

    def check(name, args, **kwargs):
        calls.append((name, dict(args), kwargs.get("declared_policy")))
        return next(answers, (True, ""))
    monkeypatch.setattr("ouroboros.safety.check_safety", check)
    return calls


def door(**arguments):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": f"door-{len(json.dumps(arguments))}", "type": "function",
        "function": {"name": "presence_finish", "arguments": json.dumps(arguments)}}]}


def run(registry, monkeypatch, tmp_path, responses, *, events=None):
    """The real loop over scripted model replies: what each request carried, and what the run published."""
    asked, notes, replies = [], [], iter(responses)

    def respond(_llm, messages, _model, tools, *_a, **_kw):
        asked.append({"messages": json.dumps(messages), "tools": [row["function"]["name"] for row in tools or []]})
        return next(replies), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    text, usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "Check the reference report."}], registry,
        SimpleNamespace(default_model=lambda: "test-model"), tmp_path / "logs",
        lambda line, **_kw: notes.append(line), queue.Queue(), task_id=TASK,
        event_queue=events, drive_root=tmp_path)
    return text, usage, trace, asked, notes


@pytest.fixture
def held(monkeypatch):
    """The first delivery of the run is held once at the host's result-review step (an ordinary debt)."""
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    from tests.test_completion_selection import hold_first_delivery

    return hold_first_delivery(monkeypatch)


def test_only_an_older_schedules_confirmed_run_gets_its_completion_door(tmp_path):
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry)
    assert registry.get_schema_by_name("finish_task") is None, "the frozen ceiling still withholds it"
    assert registry.get_schema_by_name("presence_finish")["function"]["name"] == "presence_finish"
    assert completion_schema(registry, None) == "presence_finish"
    assert ctx._schedule_door == {"task_id": TASK, "schedule_id": SCHEDULE, "token": TOKEN, "root_task_id": TASK}
    assert registry._entries["presence_finish"].safety_policy == "check", "the ordinary check, no historical skip"


def _retry(root, attempt_id, predecessor_id, *, reciprocal=("superseded_by", "retry_task_id")):
    """One hard-timeout retry as the queue writes it: the attempt names its IMMEDIATE predecessor in all three links,
    the predecessor names the attempt back in the reciprocal pair (``reciprocal`` withholds one half to break it)."""
    write_task_result(root, attempt_id, "running", root_task_id=TASK, delegation_role="root",
                      supersedes_task_id=predecessor_id, original_task_id=predecessor_id,
                      timeout_retry_from=predecessor_id)
    write_task_result(root, predecessor_id, "running", **{key: attempt_id for key in reciprocal})


def _retry_metadata(predecessor_id):
    return {**run_metadata(), "original_task_id": predecessor_id, "timeout_retry_from": predecessor_id}


def test_a_hard_timeout_retry_of_the_run_reads_its_roots_receipt(tmp_path):
    """Work, root, schedule and occurrence: a retry is new work under the same root and the same one receipt."""
    admit(tmp_path)
    _retry(tmp_path, "sched-run-retry", TASK)
    ctx = bind(ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), task_id="sched-run-retry",
               metadata=_retry_metadata(TASK))
    assert ctx._schedule_door == {"task_id": "sched-run-retry", "schedule_id": SCHEDULE, "token": TOKEN,
                                  "root_task_id": TASK}
    stray = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)  # another id under that root, without retry markers
    assert getattr(bind(stray, task_id="sched-run-other", metadata=run_metadata()), "_schedule_door", None) is None
    ghost = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)  # retry metadata alone is not the attempt's lineage
    assert getattr(bind(ghost, task_id="sched-run-ghost", metadata=_retry_metadata(TASK)),
                   "_schedule_door", None) is None


def test_a_second_retry_reaches_the_root_through_each_reciprocal_link(tmp_path):
    """root -> retry1 -> retry2: each attempt names its immediate predecessor and is named back by it."""
    admit(tmp_path)
    _retry(tmp_path, "sched-run-retry-1", TASK)
    _retry(tmp_path, "sched-run-retry-2", "sched-run-retry-1")
    ctx = bind(ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), task_id="sched-run-retry-2",
               metadata=_retry_metadata("sched-run-retry-1"))
    assert ctx._schedule_door == {"task_id": "sched-run-retry-2", "schedule_id": SCHEDULE, "token": TOKEN,
                                  "root_task_id": TASK}
    skipping = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)  # retry2 naming the root, not retry1, is no chain
    assert getattr(bind(skipping, task_id="sched-run-retry-2", metadata=_retry_metadata(TASK)),
                   "_schedule_door", None) is None


@pytest.mark.parametrize("half", ["superseded_by", "retry_task_id"])
def test_a_one_sided_reciprocal_link_is_no_admission(tmp_path, half):
    """The predecessor naming the attempt in only one of its two back-links leaves the chain unconfirmed."""
    admit(tmp_path)
    _retry(tmp_path, "sched-run-retry-1", TASK)
    _retry(tmp_path, "sched-run-retry-2", "sched-run-retry-1", reciprocal=(half,))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry, task_id="sched-run-retry-2", metadata=_retry_metadata("sched-run-retry-1"))
    assert getattr(ctx, "_schedule_door", None) is None and "presence_finish" not in registry._entries
    assert completion_schema(registry, None) == "no currently permitted completion tool"


@pytest.mark.parametrize("change", ["no_receipt", "other_occurrence", "not_accepted", "child", "no_ceiling",
                                    "authority_only", "interactive_turn", "broken_ceiling", "unreadable"])
def test_a_run_without_every_confirmed_fact_keeps_todays_door(tmp_path, monkeypatch, change):
    if change != "no_receipt":
        admit(tmp_path, status="refused" if change == "not_accepted" else "accepted")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    metadata, contract = run_metadata(), {"capability_ceiling": ceiling()}
    if change == "other_occurrence":
        metadata["schedule_occurrence"]["token"] = "occ-token-2"
    elif change == "child":
        metadata.update(parent_task_id="parent", delegation_role="subagent", root_task_id="parent")
    elif change == "no_ceiling":
        contract = {}
    elif change == "authority_only":
        metadata.pop("presence")
        metadata["presence_binding_authority"] = {"binding_id": "1" * 32}
    elif change == "interactive_turn":
        metadata.pop("schedule_occurrence")
    elif change == "broken_ceiling":
        contract["capability_ceiling"] = {**contract["capability_ceiling"], "digest": "e" * 64}
    elif change == "unreadable":
        def unreadable(*_a, **_kw):
            raise OSError("the store did not answer")
        monkeypatch.setattr("ouroboros.task_results.load_task_result", unreadable)
    ctx = bind(registry, metadata=metadata, contract=contract)
    assert "presence_finish" not in registry._entries and getattr(ctx, "_schedule_door", None) is None
    if change != "broken_ceiling":  # a ceiling that does not decode refuses every tool, as before
        assert completion_schema(registry, None) == (
            "finish_task" if change in {"no_ceiling", "authority_only"} else "no currently permitted completion tool")
    if change == "unreadable":  # unknown is never a confirmation; the next binding reads again
        monkeypatch.undo()
        registry.set_context(ctx)
        assert completion_schema(registry, None) == "presence_finish"


def test_rebinding_drops_this_registrys_door_and_a_frozen_registry_keeps_its_own(tmp_path):
    from tests.presence_legacy_helpers import _finish_presence, register_legacy_completion

    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry)
    other = bind(registry, task_id="next-task", metadata={"root_task_id": "next-task"}, contract={})
    assert "presence_finish" not in registry._entries and other._schedule_door is None, "the reused context forgets it"
    assert completion_schema(registry, None) == "finish_task"
    frozen = register_legacy_completion(ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path))
    ctx = bind(frozen)
    assert frozen._entries["presence_finish"].handler is _finish_presence, "a pre-upgrade door keeps its contract"
    assert getattr(ctx, "_schedule_door", None) is None


def test_the_door_stays_in_a_request_a_route_ceiling_fits(tmp_path):
    from ouroboros.tool_policy import fit_tool_schemas_to_limit, initial_tool_schemas

    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry)
    schemas = initial_tool_schemas(registry, context_mode="max")
    filler = [{"type": "function", "function": {"name": f"ext_tool_{index}", "parameters": {}}} for index in range(40)]
    fitted, left_out = fit_tool_schemas_to_limit([*filler, *schemas], len(schemas))
    assert "presence_finish" in [row["function"]["name"] for row in fitted] and "presence_finish" not in left_out
    assert [row["function"]["name"] for row in initial_tool_schemas(registry, context_mode="nano")].count("presence_finish") == 1


def test_a_restored_schema_list_frozen_before_the_door_gets_it_before_the_first_round(tmp_path):
    from ouroboros.loop_delivery import resume_delivery_candidate

    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry)
    ctx.messages, ctx._delivery_control_required = [], True
    schemas = [registry.get_schema_by_name("chat_history")]  # what an older checkpoint of the held run saved
    resume_delivery_candidate(registry, {}, schemas)
    assert [row["function"]["name"] for row in schemas] == ["chat_history", "presence_finish"]
    assert "Current permitted completion schema loaded: presence_finish" in ctx.messages[-1]["content"]


def test_the_door_stages_one_ordinary_quiet_request_after_the_ordinary_check(tmp_path, monkeypatch):
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry)
    checks = safety_calls(monkeypatch)
    receipt = json.loads(registry.execute("presence_finish", {"outcome": "message", "answer_sha256": SELECTOR}))
    assert receipt == {"status": "completion_requested", "completion_control": True, "action": "finish"}
    staged = ctx._completion_request
    assert {key: staged[key] for key in ("action", "answer_sha256", "wake_silent", "source", "allow_empty")} == {
        "action": "finish", "answer_sha256": SELECTOR, "wake_silent": True, "source": "presence_finish",
        "allow_empty": False}
    assert "answer" not in staged and "outcome" not in staged, "the older word changes no audience"
    assert checks == [("presence_finish", {"outcome": "message", "answer_sha256": SELECTOR}, "check")]
    assert getattr(ctx, "_presence_completion", None) is None, "a reply carrier quiets the record at the exit"
    ctx._completion_request = None
    registry.execute("presence_finish", {"message": REPORT, "action": "stop", "rationale": "One source was unreadable."})
    assert (ctx._completion_request["answer"], ctx._completion_request["action"]) == (REPORT, "stop")


@pytest.mark.parametrize("arguments, words", [
    ({}, "select exactly one of answer and answer_sha256"),
    ({"message": REPORT, "answer_sha256": SELECTOR}, "select exactly one of answer and answer_sha256"),
    ({"outcome": "speech", "message": REPORT}, "outcome is message, silent, tool_delivered or deferred"),
    ({"message": REPORT, "action": "stop"}, "stop requires a rationale"),
    ({"message": REPORT, "silent": True}, "silent"),
])
def test_a_correctable_argument_stages_nothing(tmp_path, monkeypatch, arguments, words):
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry)
    safety_calls(monkeypatch)
    result = registry.execute_result("presence_finish", arguments)
    assert result.code == "TOOL_ARG_ERROR" and words in result.text
    assert getattr(ctx, "_completion_request", None) is None


def test_bindings_prohibitions_and_identity_keep_the_ordinary_boundary(tmp_path, monkeypatch):
    admit(tmp_path)
    checks = safety_calls(monkeypatch)
    fixed = ceiling(PresenceToolGrant("presence_finish", (
        PresenceArgumentBinding(("message",), "static", static_value="A host-fixed report."),)))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry, contract={"capability_ceiling": fixed})
    result = registry.execute_result("presence_finish", {"answer_sha256": SELECTOR})
    assert result.code == "ACCESS_BLOCKED" and "SCHEDULE_REPORT_FIXED_ARGUMENT" in result.text and "message" in result.text
    assert getattr(ctx, "_completion_request", None) is None, "a fixed value is never the author's selection"
    prohibited = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(prohibited, contract={"capability_ceiling": ceiling(), "disabled_tools": ["presence_finish"]})
    assert prohibited.get_schema_by_name("presence_finish") is None
    assert prohibited.execute_result("presence_finish", {"answer_sha256": SELECTOR}).code == "RESOURCE_CONSTRAINT_BLOCKED"
    assert getattr(ctx, "_completion_request", None) is None and checks == [], "a fixed argument is refused before Safety"
    drifted = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(drifted)
    ctx.task_metadata["schedule_occurrence"]["token"] = "occ-token-2"
    assert "SCHEDULE_REPORT_DOOR_UNAVAILABLE" in drifted.execute("presence_finish", {"answer_sha256": SELECTOR})
    assert getattr(ctx, "_completion_request", None) is None


def test_the_legacy_outcome_binding_reaches_safety_but_does_not_turn_the_report_into_speech(tmp_path, monkeypatch):
    admit(tmp_path)
    checks = safety_calls(monkeypatch)
    fixed = ceiling(PresenceToolGrant("presence_finish", (
        PresenceArgumentBinding(("outcome",), "static", static_value="message"),)))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry, contract={"capability_ceiling": fixed})
    result = registry.execute_result("presence_finish", {"answer_sha256": SELECTOR})
    assert result.code == "OK" and checks == [("presence_finish", {"outcome": "message", "answer_sha256": SELECTOR}, "check")]
    assert ctx._completion_request["wake_silent"] is True
    assert ctx._completion_request["answer_sha256"] == SELECTOR and "outcome" not in ctx._completion_request


def test_without_a_reply_carrier_the_door_stages_the_quiet_presence_completion_itself(tmp_path, monkeypatch):
    admit(tmp_path)
    safety_calls(monkeypatch)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    speaker = {key: value for key, value in SPEAKER.items() if key != "place_id"}  # a binding from before Places
    ctx = bind(registry, metadata=run_metadata(speaker=speaker))
    registry.execute("presence_finish", {"outcome": "message", "message": REPORT})
    assert ctx._presence_completion == {"outcome": "silent", "message": "", "quiet": True}
    assert ctx._presence_completion_accepted is False, "only the common exit accepts it"
    from ouroboros.presence_context import presence_finish_not_accepted_note

    note = presence_finish_not_accepted_note(ctx, ctx._presence_completion)  # a hold voids it: the next round reads
    assert note.startswith("[FINISH_NOT_ACCEPTED]") and "Finish again with presence_finish" in note
    assert "nothing is sent" in note and "tool_delivered" not in note and "finish_task" not in note


def _outcome(root, usage, text, status):
    """The schedule's last-run fact as the scheduler observes it from the stored result."""
    from supervisor.queue_schedules import _scan_outcomes
    from ouroboros.task_finalization import terminal_result_fields

    write_task_result(root, TASK, status, result=text, _terminal_observed=True, **terminal_result_fields(usage))
    return _scan_outcomes(root, {"tasks": [{"id": SCHEDULE, "last_task_id": TASK}]})[SCHEDULE]


def test_a_held_report_is_selected_once_and_stays_the_runs_quiet_record(tmp_path, monkeypatch, held):
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry)
    checks, events = safety_calls(monkeypatch), queue.Queue()
    text, usage, trace, asked, notes = run(registry, monkeypatch, tmp_path,
                                           [{"content": REPORT}, door(outcome="message", answer_sha256=SELECTOR)],
                                           events=events)
    assert text == REPORT and len(asked) == 2 and held == [REPORT, REPORT]
    assert "presence_finish" in asked[0]["tools"] and "finish_task" not in asked[0]["tools"]
    assert "Use presence_finish to finish or stop" in asked[1]["messages"], "the hold names the run's own door"
    assert usage["wake_silent"] is True and usage["presence_completion_outcome"] == "silent"
    assert [name for name, *_ in checks] == ["presence_finish"], "Safety checked the new door; no speech to check"
    assert "task_completion" not in trace and "schedule_report_refused" not in trace
    published = [events.get_nowait() for _ in range(events.qsize())]
    assert not any(REPORT in json.dumps(event) for event in published), "nothing is sent"
    assert not any(REPORT in note for note in notes), "no progress carries the report"
    outcome = _outcome(tmp_path, usage, text, "completed")
    assert {key: outcome[key] for key in ("task_id", "state", "quiet", "notice", "line")} == {
        "task_id": TASK, "state": "completed", "quiet": True, "notice": False, "line": ""}


@pytest.mark.parametrize("refusal, code, why", [
    ("prohibited", "RESOURCE_CONSTRAINT_BLOCKED", "this run's constraints do not allow its completion"),
    ("returned", "SAFETY_RETURNED", "the Safety check returned its completion")])
def test_a_final_refusal_ends_the_run_with_its_report_kept_unsent(tmp_path, monkeypatch, held, refusal, code, why):
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry, contract={"capability_ceiling": ceiling(),
                             **({"disabled_tools": ["presence_finish"]} if refusal == "prohibited" else {})})
    safety_calls(monkeypatch, (False, "⚠️ SAFETY_RETURNED: the finish was returned for review."))
    events = queue.Queue()
    text, usage, trace, asked, notes = run(registry, monkeypatch, tmp_path,
                                           [{"content": REPORT}, door(answer_sha256=SELECTOR)], events=events)
    assert text == REPORT and len(asked) == (1 if refusal == "prohibited" else 2)
    assert [call["tool"] for call in trace["tool_calls"]] == ([] if refusal == "prohibited" else ["presence_finish"])
    published = [events.get_nowait() for _ in range(events.qsize())]
    assert not any(REPORT in json.dumps(event) for event in published), "the refusal sends nothing"
    assert not any(event.get("type") in {"send_message", "presence_result"} for event in published), published
    assert (usage["execution_status"], usage["reason_code"], usage["wake_silent"]) == ("failed", "tool_failure", True)
    notice = usage["terminal_host_notice"]
    assert notice == (f"This scheduled run's report was not confirmed: {why} ({code}). "
                      "The report stays in this task's result; this completion sent nothing."), "the host's own words"
    assert "returned for review" not in notice, "the refusal's own dynamic words stay in its call history"
    assert trace["schedule_report_refused"]["answer_sha256"] == SELECTOR
    assert registry._ctx._presence_forced_declaration["status"] == "missing", "the record never becomes speech"
    outcome = _outcome(tmp_path, usage, text, "failed")
    assert (outcome["state"], outcome["quiet"], outcome["notice"]) == ("failed", False, True)
    assert outcome["line"] == notice.splitlines()[0][:160] and REPORT not in outcome["line"]


@pytest.mark.parametrize("prohibited", [True, False])
def test_owner_controls_time_and_money_decide_before_a_refused_door_ends_the_run(tmp_path, monkeypatch, held, prohibited):
    """The owner/time/money rail precedes both a known prohibition and a refusal from an attempted selection."""
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry, contract={"capability_ceiling": ceiling(),
                             **({"disabled_tools": ["presence_finish"]} if prohibited else {})})
    safety_calls(monkeypatch, (False, "⚠️ SAFETY_RETURNED: the finish was returned for review."))
    real, reads = loop._maybe_early_finalize, []

    def early(limit_ctx, tools, controls, **kwargs):
        reads.append(len(limit_ctx.llm_trace["tool_calls"]))
        if len(reads) == (2 if prohibited else 3):  # the next round checks its own controls first
            return "Ended by its rail.", dict(limit_ctx.accumulated_usage, reason_code="finalization_grace"), limit_ctx.llm_trace
        return real(limit_ctx, tools, controls, **kwargs)
    monkeypatch.setattr(loop, "_maybe_early_finalize", early)
    text, usage, trace, asked, _notes = run(registry, monkeypatch, tmp_path,
                                            [{"content": REPORT}, door(answer_sha256=SELECTOR)])
    assert text == "Ended by its rail." and reads == ([0, 0] if prohibited else [0, 0, 1])
    assert len(asked) == (1 if prohibited else 2)
    assert usage["reason_code"] == "finalization_grace" and "schedule_report_refused" not in trace


def test_an_unavailable_check_and_a_correctable_call_are_not_final(tmp_path, monkeypatch, held):
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry)
    checks = safety_calls(monkeypatch, (False, "⚠️ SAFETY_UNAVAILABLE: the supervisor did not answer."))
    text, usage, trace, asked, _notes = run(registry, monkeypatch, tmp_path, [
        {"content": REPORT}, door(answer_sha256=SELECTOR), door(outcome="silent"), door(answer_sha256=SELECTOR)])
    assert text == REPORT and len(asked) == 4 and len(checks) == 3
    assert usage["wake_silent"] is True and "schedule_report_refused" not in trace


def test_refused_door_end_reads_only_this_runs_final_refusals(tmp_path):
    from ouroboros.loop_delivery import DeliveryCandidate

    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry)
    refused = {"tool": "presence_finish", "is_error": True, "tool_result_code": "SAFETY_RETURNED", "result": "no"}
    limit = SimpleNamespace(llm_trace={"tool_calls": [refused]}, accumulated_usage={}, tools=registry, drive_root=tmp_path,
                            task_id=TASK, root_task_id=TASK, status_drive_root=None, event_queue=None, drive_logs=tmp_path / "logs")
    assert refused_door_end(registry, limit, after_tools=True) is None, "no retained report: nothing to keep"
    ctx._delivery_candidate = DeliveryCandidate(REPORT, SELECTOR, 1, 1, "e", {})
    for calls in ([], [{**refused, "tool_result_code": "SAFETY_UNAVAILABLE"}], [{**refused, "tool": "chat_history"}],
                  [refused, {"tool": "presence_finish", "is_error": False}],
                  [refused, {"tool": "read_file", "is_error": False, "tool_result_code": "OK"}]):  # read it first
        assert refused_door_end(registry, SimpleNamespace(**{**vars(limit), "llm_trace": {"tool_calls": calls}}),
                                after_tools=True) is None
    import ouroboros.loop_messages as messages
    owed = messages.owner_reply_owed
    try:
        messages.owner_reply_owed = lambda _ctx: True
        # A record-only schedule without an owner-send path settles instead of spinning for a reply it cannot send.
        assert refused_door_end(registry, limit, after_tools=True) is not None
        ctx._delivery_candidate = DeliveryCandidate(REPORT, SELECTOR, 1, 1, "e", {})
        registry.get_schema_by_name = lambda name: object()  # a send path restores the ordinary owner-wait guard
        assert refused_door_end(registry, limit, after_tools=True) is None
    finally:
        messages.owner_reply_owed = owed
    ctx._completion_selected = {"action": "finish"}
    assert refused_door_end(registry, limit, after_tools=True) is None, "a selection already made decides"
    ctx._completion_selected, ctx._schedule_door = None, None
    assert refused_door_end(registry, limit, after_tools=True) is None, "no bound door: an ordinary task"


def test_an_authored_stop_through_the_door_stays_unfinished_and_quiet(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry)
    safety_calls(monkeypatch)
    text, usage, trace, asked, _notes = run(registry, monkeypatch, tmp_path, [
        door(action="stop", message=REPORT, rationale="The second announcement channel was unreadable.")])
    assert text == REPORT and len(asked) == 1 and usage["wake_silent"] is True
    assert {key: trace["task_completion"][key] for key in ("action", "source", "answer_sha256")} == {
        "action": "stop", "source": "presence_finish", "answer_sha256": SELECTOR}


class _WorkerDeath(BaseException):
    """The worker process ends: nothing after this point runs, saves or acknowledges."""


@pytest.mark.parametrize("boundary", ["post_batch", "candidate"])
def test_a_cold_return_accepts_the_existing_selection_without_a_repeated_call(tmp_path, monkeypatch, held, boundary):
    """The worker dies after the door's selection was saved: staged (``post_batch``) or consumed with its quiet
    candidate (``candidate``). The same task continues from that state with its door bound again, and accepts the
    same bytes through the remaining ordinary checks without a model call."""
    from ouroboros import working_checkpoint as wc
    from tests.test_working_checkpoint import _registry

    asked = []

    def attempt(number, answers, **facts):
        registry = _registry(tmp_path, monkeypatch, TASK, number)
        for key, value in facts.items():
            setattr(registry._ctx, key, value)
        bind(registry)
        replies = iter(answers)

        def respond(*_a, **_kw):
            asked.append(number)
            return next(replies), 0.0
        monkeypatch.setattr(loop, "call_llm_with_retry", respond)
        return loop.run_llm_loop([{"role": "user", "content": "Weekly glossary check"}], registry,
                                 SimpleNamespace(default_model=lambda: "test-model"), tmp_path / "logs",
                                 lambda *_a, **_kw: None, queue.Queue(), task_id=TASK, drive_root=tmp_path)

    admit(tmp_path)
    safety_calls(monkeypatch)
    saved, seen = wc.save_or_log, []

    def save_then_die(limit_ctx, at):
        result = saved(limit_ctx, at)
        seen.append(at)
        if at == boundary and (boundary == "post_batch" or seen.count("candidate") == 2):
            raise _WorkerDeath
        return result

    with monkeypatch.context() as dying:
        dying.setattr(wc, "save_or_log", save_then_die)
        with pytest.raises(_WorkerDeath):
            attempt(1, [{"content": REPORT}, door(answer_sha256=SELECTOR)])
    state = wc.read_working(tmp_path, TASK, 1)[1]
    selection = state["delivery"].get("_completion_request" if boundary == "post_batch" else "_completion_selected")
    assert (selection["source"], selection["answer_sha256"], selection["wake_silent"]) == ("presence_finish", SELECTOR, True)
    task = {"id": TASK}
    assert wc.attach_recovery(tmp_path, task, source_task_id=TASK, from_attempt=1, cause="worker_crash")
    text, usage, _trace = attempt(2, [], working_recovery=task["_working_recovery"])
    assert text == REPORT and asked == [1, 1], "the existing selection, no repeated command"
    assert usage["wake_silent"] is True


def test_the_schedulers_own_admission_of_an_older_template_confirms_its_run(q, monkeypatch, held):  # noqa: F811
    """A template saved by an older Presence follow-up (its speaker and frozen ceiling by value) goes through the
    real occurrence protocol; the task it admits binds the door from that receipt and ends as a quiet record."""
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr("supervisor.queue_schedules.in_worker_process", lambda: False)

    template = {"type": "task", "text": "Check the reference report.", "chat_id": 1,
                "metadata": {"presence": SPEAKER, "resource_intent": {"kind": "system_repo"}},
                "task_contract": {"capability_ceiling": ceiling()}}
    q.queue.upsert_scheduled_task(
        {"id": SCHEDULE, "name": "Glossary check", "enabled": True, "source": "task_followup",
         "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"}, "task": template},
        writer_kind="consciousness", writer_metadata={"presence": SPEAKER},
        host_followup={"followup_relation": {"kind": "independent"},
                       "followup_origin": {"task_id": "", "root_task_id": ""}})
    q.queue.check_scheduled_tasks()
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert "capability_ceiling" in task["task_contract"] and task["metadata"]["presence"]["place_id"] == "place-1"
    registry = ToolRegistry(repo_dir=q.root, drive_root=q.root)
    ctx = ToolContext(repo_dir=q.root, drive_root=q.root, task_id=task["id"], task_contract=task["task_contract"],
                      task_metadata={**task["metadata"], "root_task_id": task["root_task_id"],
                                     "delegation_role": task["delegation_role"]})
    registry.set_context(ctx)
    assert ctx._schedule_door["token"] == task["metadata"]["schedule_occurrence"]["token"]
    safety_calls(monkeypatch)
    asked, replies = [], iter([{"content": REPORT}, door(answer_sha256=SELECTOR)])

    def respond(*_a, **_kw):
        asked.append(1)
        return next(replies), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    text, usage, _trace = loop.run_llm_loop(
        [{"role": "user", "content": task["text"]}], registry, SimpleNamespace(default_model=lambda: "test-model"),
        q.root / "logs", lambda *_a, **_kw: None, queue.Queue(), task_id=task["id"], drive_root=q.root)
    assert text == REPORT and len(asked) == 2 and usage["wake_silent"] is True


def test_a_salvage_receipt_of_an_older_schedules_run_names_no_excerpt(tmp_path):
    from supervisor.terminal_delivery import build_unreviewed_salvage_event

    row = {"task_id": TASK, "chat_id": 1, "metadata": run_metadata(), "task_contract": {"capability_ceiling": ceiling()}}
    kept = build_unreviewed_salvage_event(tmp_path, row, TASK, outcome="cancelled by Stop now",
                                          salvaged_text=REPORT, settled_status="cancelled")
    assert REPORT not in kept["text"] and "(its output stays in the task details)" in kept["text"]
    ordinary = build_unreviewed_salvage_event(tmp_path, {"task_id": "other", "chat_id": 1}, "other",
                                              outcome="cancelled by Stop now", salvaged_text=REPORT,
                                              settled_status="cancelled")
    assert REPORT in ordinary["text"], "every other task's salvage is unchanged"


def test_no_room_excerpt_of_an_older_schedules_held_report():
    """The shared completion excerpt stays empty for an older schedule's held report (#3687)."""
    from ouroboros.project_dialogue import _completion_excerpt

    row = {"result": REPORT, "terminal_origin": "host_salvage", "metadata": run_metadata(),
           "task_contract": {"capability_ceiling": ceiling()}}
    assert _completion_excerpt(row, chat_id=1, salvage_only=True) == ""
    ordinary = {"result": REPORT, "terminal_origin": "host_salvage"}
    assert _completion_excerpt(ordinary, chat_id=1, salvage_only=True).startswith(
        "Preserved intermediate output")


def test_main_mirror_publishes_no_older_schedule_report():
    """A Main completion row never carries an older schedule's held report as its answer (#3687)."""
    from ouroboros.projects_registry import mirrored_answer

    row = {"result": REPORT, "terminal_origin": "model_final", "metadata": run_metadata(),
           "task_contract": {"capability_ceiling": ceiling()}}
    assert mirrored_answer(row, "done") == {}
    assert mirrored_answer({"result": REPORT, "terminal_origin": "model_final"}, "done") == {
        "completion_answer": REPORT}


def test_replay_of_an_older_schedule_record_without_a_recorded_reply_stays_silent():
    """A stored schedule report with no recorded reply body is never replayed as place speech (#3687)."""
    from ouroboros.presence_runner import presence_result_from_stored

    stored = {"status": "completed", "terminal_origin": "model_final", "result": REPORT,
              "metadata": run_metadata(), "task_contract": {"capability_ceiling": ceiling()}}
    replay = presence_result_from_stored(stored, TASK)
    assert (replay.outcome, replay.text) == ("silent", "")
    spoken = presence_result_from_stored(
        {"status": "completed", "terminal_origin": "model_final", "result": REPORT, "metadata": {}}, TASK)
    assert (spoken.outcome, spoken.text) == ("message", REPORT)


def test_the_completion_wins_send_never_speaks_an_older_schedules_report(tmp_path):
    """A dead worker's completion-wins send (#3687): whichever row carries the frozen geometry — the
    stored result, the queue's task, or both — the seam builds no send, so the adopt path that copies
    ``model_final`` without ``wake_silent`` still keeps the report a record. Every other task's answer
    still ships word for word."""
    from supervisor.terminal_delivery import build_completed_result_event

    geometry = {"metadata": run_metadata(), "task_contract": {"capability_ceiling": ceiling()}}
    stored = {"status": "completed", "terminal_origin": "model_final", "result": REPORT, **geometry}
    assert build_completed_result_event(tmp_path, {"id": TASK, "chat_id": 1}, TASK, stored) is None
    assert build_completed_result_event(tmp_path, None, TASK, stored) is None, "task_row falls back to stored"
    queue_row = {"id": TASK, "chat_id": 1, **geometry}  # the supervisor's own row of the same run
    bare = {"status": "completed", "terminal_origin": "model_final", "result": REPORT}
    assert build_completed_result_event(tmp_path, queue_row, TASK, bare) is None
    ordinary = build_completed_result_event(tmp_path, {"id": "other", "chat_id": 1}, "other", bare)
    assert ordinary["text"] == REPORT and ordinary["task_id"] == "other", "every other task is unchanged"


def test_a_quiet_schedule_record_sends_nothing_again_and_never_reaches_the_main_mirror(tmp_path):
    """A quiet stored schedule row re-delivers nothing (#3687): the completion-wins seam builds no
    send, the terminal projection's own guard (``quiet_finish``, its gate before
    ``project_completion_delivery_outcome``) keeps the Main mirror unreached, and a delivery the run's
    carrier already confirmed stays the only one."""
    from supervisor import terminal_delivery as td
    from ouroboros.task_finalization import quiet_finish

    stored = {"status": "completed", "terminal_origin": "model_final", "result": REPORT, "wake_silent": True,
              "metadata": run_metadata(), "task_contract": {"capability_ceiling": ceiling()}}
    task = {"id": TASK, "chat_id": 1}
    assert td.build_completed_result_event(tmp_path, task, TASK, stored) is None
    assert quiet_finish(stored) is True, "the mirror is never even built for this row"
    assert td.register_delivery(tmp_path, td.delivery_id_for(TASK, REPORT)) is True, "the carrier sent it once"
    assert td.deliver_completed_result(tmp_path, task, TASK, stored) is False
    assert td.pending_deliveries(tmp_path) == [], "one confirmed delivery; nothing re-owed, nothing re-sent"


def _cold_attempt(tmp_path, monkeypatch, number, answers, asked, **facts):
    """One worker attempt of the bound run over the real loop; ``facts`` carry a recovery handoff."""
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, TASK, number)
    for key, value in facts.items():
        setattr(registry._ctx, key, value)
    bind(registry)
    replies = iter(answers)

    def respond(*_a, **_kw):
        asked.append(number)
        return next(replies), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    return loop.run_llm_loop([{"role": "user", "content": "Reference check"}], registry,
                             SimpleNamespace(default_model=lambda: "test-model"), tmp_path / "logs",
                             lambda *_a, **_kw: None, queue.Queue(), task_id=TASK, drive_root=tmp_path)


def _die_at(monkeypatch, tmp_path, boundary, answers, asked, occurrence=1):
    """Attempt 1 dies right after its ``occurrence``-th ``boundary`` save; the working state it saved."""
    from ouroboros import working_checkpoint as wc

    saved, seen = wc.save_or_log, []

    def save_then_die(limit_ctx, at):
        result = saved(limit_ctx, at)
        seen.append(at)
        if at == boundary and seen.count(boundary) == occurrence:
            raise _WorkerDeath
        return result
    with monkeypatch.context() as dying:
        dying.setattr(wc, "save_or_log", save_then_die)
        with pytest.raises(_WorkerDeath):
            _cold_attempt(tmp_path, monkeypatch, 1, answers, asked)
    return wc.read_working(tmp_path, TASK, 1)[1]


def _recover(tmp_path):
    from ouroboros import working_checkpoint as wc

    task = {"id": TASK}
    assert wc.attach_recovery(tmp_path, task, source_task_id=TASK, from_attempt=1, cause="worker_crash")
    return task["_working_recovery"]


@pytest.mark.parametrize("saved", ["staged", "selected"])
def test_a_preupgrade_literal_selection_is_normalized_and_kept_on_a_cold_return(tmp_path, monkeypatch, held, saved):
    """A pre-upgrade working record has no wake_silent/quiet keys on its selection, Presence completion or candidate:
    saved with the literal still staged, or already selected beside its candidate. The real restore marks it a
    record, and the common exit keeps it one: no place reply check, hold or notice."""
    from ouroboros import place_reply
    from ouroboros import working_checkpoint as wc

    admit(tmp_path)
    checks, asked = safety_calls(monkeypatch), []
    boundary, occurrence = ("post_batch", 1) if saved == "staged" else ("candidate", 2)
    state = _die_at(monkeypatch, tmp_path, boundary, [{"content": REPORT}, door(message=REVISED_REPORT)], asked, occurrence)
    delivery, key = state["delivery"], "_completion_request" if saved == "staged" else "_completion_selected"
    delivery[key].pop("wake_silent", None)
    delivery.get("_presence_completion", {}).pop("quiet", None)
    state["delivery_candidate"].pop("quiet", None)
    wc.checkpoint_path(tmp_path, TASK, 1).write_text(json.dumps(state), encoding="utf-8")  # the older record
    older = wc.read_working(tmp_path, TASK, 1)[1]
    assert "quiet" not in older["delivery_candidate"] and "wake_silent" not in older["delivery"][key]
    exits, spoken, real_exit = [], [], place_reply.prepare_focus_reply

    def common_exit(ctx, text, *, quiet=False):
        exits.append((quiet, getattr(ctx._delivery_candidate, "quiet", None)))
        if saved == "selected":
            frozen = older["delivery_candidate"]
            assert (ctx._delivery_candidate.revision, ctx._delivery_candidate.evidence_fingerprint) == (
                frozen["revision"], frozen["evidence_fingerprint"])
        return real_exit(ctx, text, quiet=quiet)
    monkeypatch.setattr(place_reply, "prepare_focus_reply", common_exit)
    monkeypatch.setattr(place_reply, "record_notice", lambda *_a, **kw: spoken.append(kw.get("text")))
    monkeypatch.setattr(place_reply, "reply_hold", lambda *_a, **_kw: spoken.append("hold") or "")
    checks.clear()
    text, usage, _trace = _cold_attempt(tmp_path, monkeypatch, 2, [], asked, working_recovery=_recover(tmp_path))
    assert text == REVISED_REPORT and asked == [1, 1] and usage.get("wake_silent") is True
    assert exits and all(exit == (True, True) for exit in exits), "the restored candidate reaches the exit as a record"
    assert spoken == [] and "place_reply" not in [name for name, *_ in checks]


def test_a_restored_literal_requires_the_same_bytes_as_its_executed_call(tmp_path):
    from ouroboros.loop_completion_schema import _restored_schedule_selection_error

    admit(tmp_path)
    ctx = bind(ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path))
    ctx.messages = []  # an older persisted attempt with the selection in its executed-call receipt
    ctx._delivery_candidate = SimpleNamespace(full_text=REVISED_REPORT)
    request = {"source": "presence_finish", "answer": REVISED_REPORT}
    receipt = {"tool": "presence_finish", "is_error": False, "args": {"message": REPORT}}
    trace = {"tool_calls": [receipt]}
    assert _restored_schedule_selection_error(ctx, request, trace).startswith("selection_mismatch:")
    ctx._delivery_candidate.full_text = request["answer"] = REPORT
    assert not _restored_schedule_selection_error(ctx, request, trace)


def test_a_busy_receipt_read_on_a_cold_return_still_reaches_the_door(tmp_path, monkeypatch):
    """The worker dies with its report held, before any door call. Its next attempt restores while the root's
    receipt read (the door's own) answers StoreBusy; the ordinary registry reads again, and the model's presence_finish reaches the
    adapter (an ordinary quiet selection), never an unknown-tool refusal."""
    import sys
    from ouroboros.store import StoreBusy
    import ouroboros.task_results as task_results

    admit(tmp_path)
    checks, asked = safety_calls(monkeypatch), []
    _die_at(monkeypatch, tmp_path, "candidate", [{"content": REPORT}], asked)
    recovery, real, busy = _recover(tmp_path), task_results.load_task_result, []

    def receipt_read(root, task_id, *args, **kwargs):
        door_read = sys._getframe(1).f_globals.get("__name__") == "ouroboros.loop_completion_schema"
        if door_read and not asked[1:]:
            busy.append(task_id)  # until attempt 2's first model request, the door's receipt read is busy
            raise StoreBusy("the store is busy")
        return real(root, task_id, *args, **kwargs)
    monkeypatch.setattr(task_results, "load_task_result", receipt_read)
    text, usage, trace = _cold_attempt(tmp_path, monkeypatch, 2, [door(answer_sha256=SELECTOR)], asked,
                                       working_recovery=recovery)
    assert busy and asked == [1, 2] and text == REPORT and usage.get("wake_silent") is True
    assert [name for name, *_ in checks] == ["presence_finish"], "the call reached the adapter's ordinary check"
    door_calls = [call for call in trace["tool_calls"] if call.get("tool") == "presence_finish"]
    assert [bool(call.get("is_error")) for call in door_calls] == [False], "the adapter staged it; no UNKNOWN_TOOL"


def test_forced_exit_never_turns_a_schedule_report_into_a_presence_result_or_finish_note(tmp_path):
    from ouroboros.loop_forced_finalization import _presence_forced_armed
    from ouroboros.presence_runner import build_presence_result_event

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry)
    assert _presence_forced_armed(ctx) is False
    ctx._presence_completion = {"outcome": "silent", "message": REPORT, "quiet": True}
    ctx._presence_completion_accepted = True
    task = {"id": TASK, "metadata": run_metadata(), "task_contract": {"capability_ceiling": ceiling()}}
    event = build_presence_result_event(task, REPORT, ctx, terminal_origin="model_final", silent=True)
    assert (event["outcome"], event["text"], event["message"]) == ("silent", "", "")
    assert "finish_note" not in event and task["metadata"]["presence_result_text"] == ""


def test_an_older_candidate_is_quiet_on_the_forced_return_before_the_pipeline(tmp_path):
    from tests.test_delivery_forced_finalization import _forced_test_context

    runner, registry, limit, trace = _forced_test_context(tmp_path)
    admit(tmp_path, "parent1")
    ctx = bind(registry, task_id="parent1")
    runner._replace_delivery_candidate(registry, limit, trace, REPORT, control="replace")
    ctx._delivery_candidate.quiet = False  # retained before the quiet marker existed
    text, usage, _trace = runner._handle_round_limit(limit)
    assert text == REPORT and usage["wake_silent"] is True


def test_cascade_salvage_uses_the_roots_stored_geometry_when_the_routing_row_is_a_descendant(tmp_path):
    from supervisor.terminal_delivery import build_unreviewed_salvage_event

    write_task_result(tmp_path, TASK, "completed", result=REPORT, metadata=run_metadata(),
                      task_contract={"capability_ceiling": ceiling()})
    descendant = {"task_id": "ordinary-child", "chat_id": 1, "parent_task_id": TASK}
    event = build_unreviewed_salvage_event(tmp_path, descendant, TASK, outcome="cancelled",
                                           salvaged_text=REPORT, settled_status="cancelled")
    assert REPORT not in event["text"] and "(its output stays in the task details)" in event["text"]


@pytest.mark.parametrize("borrowed", [True, False])
def test_only_borrowed_routing_needs_the_roots_geometry_to_preview_salvage(tmp_path, monkeypatch, borrowed):
    from ouroboros import task_results
    from ouroboros.store import StoreBusy
    from supervisor.terminal_delivery import build_unreviewed_salvage_event

    read = task_results.load_task_result

    def unavailable(root, task_id, *args, **kwargs):
        if task_id == TASK and kwargs.get("strict"):
            raise StoreBusy("root result pending")
        return read(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", unavailable)
    row = {"task_id": "ordinary-child" if borrowed else TASK, "chat_id": 1}
    if borrowed:
        row["parent_task_id"] = TASK
    event = build_unreviewed_salvage_event(tmp_path, row, TASK, outcome="cancelled",
                                           salvaged_text=REPORT, settled_status="cancelled")
    assert (REPORT in event["text"]) is not borrowed


def test_schedule_last_outcome_never_uses_a_legacy_report_as_its_preview(tmp_path):
    from supervisor.queue_schedules import _scan_outcomes

    write_task_result(tmp_path, TASK, "completed", result=REPORT, metadata=run_metadata(), wake_silent=True,
                      task_contract={"capability_ceiling": ceiling()}, terminal_origin="model_final")
    outcome = _scan_outcomes(tmp_path, {"tasks": [{"id": SCHEDULE, "last_task_id": TASK}]})[SCHEDULE]
    assert outcome["line"] == "" and outcome["quiet"] is True


def test_a_legacy_binding_refusal_is_final_without_another_model_round(tmp_path):
    from ouroboros.loop_delivery import DeliveryCandidate

    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry)
    ctx._delivery_candidate = DeliveryCandidate(REPORT, SELECTOR, 1, 1, "e", {})
    call = {"tool": "presence_finish", "is_error": True, "tool_result_code": "LEGACY_BLOCKED",
            "result": "binding could not resolve"}
    limit = SimpleNamespace(llm_trace={"tool_calls": [call]}, accumulated_usage={}, tools=registry, drive_root=tmp_path,
                            task_id=TASK, root_task_id=TASK, status_drive_root=None, event_queue=None, drive_logs=tmp_path / "logs")
    text, usage, _trace = refused_door_end(registry, limit, after_tools=True)
    assert text == REPORT and usage["wake_silent"] is True
    assert usage["terminal_host_notice"] == (
        "This scheduled run's report was not confirmed: its confirmation attempt was blocked (LEGACY_BLOCKED). "
        "The report stays in this task's result; this completion sent nothing.")


def test_an_explicit_literal_can_replace_the_offered_report_as_a_quiet_record(tmp_path, monkeypatch, held):
    admit(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry)
    checks, events = safety_calls(monkeypatch), queue.Queue()
    text, usage, trace, asked, _notes = run(registry, monkeypatch, tmp_path,
                                           [{"content": REPORT}, door(message=REVISED_REPORT)], events=events)
    assert text == REVISED_REPORT and len(asked) == 2 and usage["wake_silent"] is True
    assert [args["message"] for name, args, *_ in checks if name == "presence_finish"] == [REVISED_REPORT]
    assert not any(call.get("is_error") for call in trace["tool_calls"])
    assert [(event.get("type"), sorted(event)) for event in list(events.queue)
            if REVISED_REPORT in json.dumps(event)] == []


def test_a_busy_store_does_not_unbind_the_door_for_the_rest_of_the_attempt(tmp_path, monkeypatch):
    admit(tmp_path)

    from ouroboros import task_results

    real = task_results.load_task_result

    def busy(*_a, **_kw):
        raise OSError("busy")

    monkeypatch.setattr("ouroboros.task_results.load_task_result", busy)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    bind(registry)
    assert completion_schema(registry, None) == "no currently permitted completion tool"
    monkeypatch.setattr(task_results, "load_task_result", real)
    assert completion_schema(registry, None) == "presence_finish", "the next entry retries the receipt read"


@pytest.mark.parametrize("catalog", ["schemas", "get_schema_by_name"])
def test_after_a_busy_bind_each_catalog_read_publishes_the_door_itself(tmp_path, monkeypatch, catalog):
    """A StoreBusy receipt read at bind leaves the door unbound; once the store answers, this catalog read alone binds
    and publishes it on a fresh registry (dispatch is the cold-return test's own path)."""
    from ouroboros.store import StoreBusy
    import ouroboros.task_results as task_results

    admit(tmp_path)
    real = task_results.load_task_result

    def busy(*_a, **_kw):
        raise StoreBusy("the store is busy")
    monkeypatch.setattr(task_results, "load_task_result", busy)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry)
    assert "presence_finish" not in registry._entries and getattr(ctx, "_schedule_door", None) is None
    monkeypatch.setattr(task_results, "load_task_result", real)
    if catalog == "schemas":
        assert "presence_finish" in [row["function"]["name"] for row in registry.schemas()]
    else:
        assert registry.get_schema_by_name("presence_finish")["function"]["name"] == "presence_finish"
    assert ctx._schedule_door["root_task_id"] == TASK


def test_completion_wins_keeps_only_the_legacy_custody_notice_not_the_report(tmp_path):
    from supervisor.terminal_delivery import build_completed_result_event

    stored = {"status": "completed", "result": REPORT, "wake_silent": True,
              "metadata": run_metadata(), "task_contract": {"capability_ceiling": ceiling()}}
    event = build_completed_result_event(tmp_path, {"id": TASK, "chat_id": 1}, TASK, stored,
                                         unreconciled_runs=["run-unsettled"])
    assert event is not None and "run-unsettled" in event["text"] and REPORT not in event["text"]
