"""``plan_task`` — write / ask / read over the plan record, driven through the real
ToolContext, the real plan record and a FAKE review substrate. Pinned in both
directions: typed $0 refusals write nothing and call nobody; a write records a
revision and calls nobody; an ask sends exactly to the named participants with each
participant's own delivery and continuation; replies settle into the record and the
intention ledger record; only the raiser's withdrawal releases a Blocking hold."""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import plan_record
from ouroboros.tools import plan_ask, plan_review as pr
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar

CLEAN = json.dumps({"prose": "Looks sound to me.", "objections": [], "withdrawn": []})


def _reply(prose="", objections=(), withdrawn=()):
    return json.dumps({"prose": prose, "objections": [{"id": i, "text": t} for i, t in objections],
                       "withdrawn": [{"id": i, "why": w} for i, w in withdrawn]})


def _slots(*specs):
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_runtime import PLAN_REVIEW_MAX_TOKENS

    out = []
    for spec in specs:
        sid, model = spec[0], spec[1]
        session = len(spec) > 2 and spec[2] == "session"
        out.append(ReviewSlot(
            slot_id=sid, model=model, effort="high", role_hint="plan reviewer", subagent_id=sid,
            route=ReviewRouteKind.AGENT_SESSION if session else ReviewRouteKind.API_CHAT,
            session_target="cursor=grok" if session else "", max_tokens=PLAN_REVIEW_MAX_TOKENS, default_temperature=0.2,
            native_retrieval_override=None if session else False))
    return out


class _Substrate:
    """Fake ``run_review_request``: per-slot answers (text, dict row overrides, or callable(request))."""

    def __init__(self, answers):
        self.answers = answers
        self.calls: list = []

    def __call__(self, request, *, slots, drive_root, llm, usage_ctx=None):
        self.calls.append({"request": request, "slots": list(slots)})
        actors = []
        for slot in slots:
            answer = self.answers.get(slot.slot_id, CLEAN)
            answer = answer(request) if callable(answer) else answer
            row = {"slot_id": slot.slot_id, "model": slot.model, "status": "ok", "raw_text": "", "error": "",
                   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "resolved_model": slot.model,
                             "physical_attempt_state": "settled", "cost": 0.01},
                   "prompt_ref": {}, "response_ref": {}, "operation_id": f"op-{slot.slot_id}-{len(self.calls)}",
                   "operation_state": "settled"}
            if isinstance(answer, dict):
                row.update(answer)
            else:
                row["raw_text"] = answer or ""
                if not answer:
                    row.update(status="error", error="transport died")
            actors.append(row)
        return SimpleNamespace(actors=actors)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    system = tmp_path / "repo"
    (system / "docs").mkdir(parents=True)
    (system / "BIBLE.md").write_text("# BIBLE.md\n\n## Principle 0: Agency\n\nbe.\n", encoding="utf-8")
    (system / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros vX — Architecture\n\n## 1. Runtime\n\nthe loop.\n", encoding="utf-8")
    (system / "docs" / "CHECKLISTS.md").write_text(
        "## Plan Review Checklist\n\nFIXTURE CHECKLIST ROW: the bar.\n\n## Coupling questions\n\nx\n", encoding="utf-8")
    (system / "ouroboros").mkdir()
    (system / "ouroboros" / "loop.py").write_text("x = 1\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("deck notes\n", encoding="utf-8")
    drive = tmp_path / "data"
    (drive / "logs").mkdir(parents=True)
    events: queue.Queue = queue.Queue()
    progress: list = []

    def make_ctx(*, active_workspace=True, task_id="task-1", force_plan=False):
        ctx = ToolContext(
            repo_dir=system, system_repo_dir=system, drive_root=drive, task_id=task_id,
            workspace_root=workspace if active_workspace else None, workspace_mode="external" if active_workspace else "",
            task_metadata={"root_task_id": task_id, **({"force_plan": True} if force_plan else {})},
            task_contract={"objective": "Deliver the thing"}, event_queue=events)
        ctx.emit_progress_fn = lambda text, **_kw: progress.append(text)
        return ctx

    state = {"slots": _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "m/c"))}
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(plan_ask, "plan_review_slots", lambda default_effort="": list(state["slots"]))
    monkeypatch.setattr("ouroboros.tools.plan_dialogue.session_input_limits", lambda _slots: {})
    monkeypatch.setattr("ouroboros.tools.review_checklist.REPO_ROOT", system)

    def install(answers):
        import ouroboros.review_substrate as rs

        sub = _Substrate(answers)
        monkeypatch.setattr(rs, "run_review_request", sub)
        return sub

    return SimpleNamespace(system=system, workspace=workspace, drive=drive, events=events, progress=progress,
                           make_ctx=make_ctx, state=state, install=install, monkeypatch=monkeypatch)


def _call(ctx, **kw):
    """The handler under the registry's result sidecar: ``(text, typed ToolResult)``."""
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = pr._handle_plan_task(ctx, **kw)
        result = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert result is not sentinel, "the handler must publish a typed result"
    return text, result


def _record(h, task_id="task-1"):
    return plan_record.load_plan_record(h.drive, task_id)


def _user_text(content):
    if isinstance(content, list):
        return "".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return str(content or "")


def _write(ctx, **over):
    fields = {"goal": "Ship the deck", "plan": "Outline first, then draft each slide.",
              "done_when": [{"id": "c1", "claim": "exactly 5 slides", "support": "count them"}], "affected_paths": []}
    fields.update(over)
    return _call(ctx, **fields)


# ---------------------------------------------------------------- refusals write nothing

def test_old_form_is_refused_at_zero_cost_naming_the_new_form(harness):
    sub = harness.install({})
    ctx = harness.make_ctx()
    text, result = _call(ctx, goal="G", plan="P", spec={"affected_paths": []})
    assert "PLAN_FORM_RETIRED" in text and "spec" in text and "ask='council'" in text
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    text, _ = _call(ctx, review_disposition={"review_fingerprint": "f" * 64, "items": []})
    assert "PLAN_FORM_RETIRED" in text
    assert sub.calls == [] and _record(harness) == plan_record.empty_record()


def test_first_revision_requires_affected_paths_and_schema_errors_write_nothing(harness):
    sub = harness.install({})
    ctx = harness.make_ctx()
    text, result = _call(ctx, goal="G", plan="P")
    assert "PLAN_FIRST_REVISION_NEEDS_AFFECTED_PATHS" in text and result.code == "TOOL_ARG_ERROR"
    for bad in ({"done_when": "five slides", "affected_paths": []}, {"affected_paths": "x"}, {"ask": []},
                {"ask": "everyone"}, {"message": "hello"}, {"reviewer_effort": "turbo", "ask": "council"}, {"nonsense": 1}):
        text, result = _call(ctx, **bad)
        assert "PLAN_ARGS_INVALID" in text and result.code == "TOOL_ARG_ERROR", bad
    assert sub.calls == [] and _record(harness) == plan_record.empty_record()


def test_unknown_participant_refuses_the_whole_call_including_its_write(harness):
    sub = harness.install({})
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config.catalog_review_row",
                                lambda _snapshot, name: (_ for _ in ()).throw(ValueError(f"no row {name}")))
    text, result = _write(harness.make_ctx(), ask=["ghost"])
    assert "PLAN_PARTICIPANT_UNKNOWN" in text and "ghost" in text and result.code == "TOOL_ARG_ERROR"
    assert sub.calls == [] and _record(harness) == plan_record.empty_record()


def test_budget_refusal_writes_nothing(harness):
    sub = harness.install({})
    harness.monkeypatch.setattr("ouroboros.tools.review_helpers.review_wave_budget_gate",
                                lambda *_a, **_k: {"binding_axis": "global", "remaining_usd": 0.0, "estimated_wave_usd": 1.0,
                                                   "global_limit_usd": 5.0, "global_known_usd": 5.0})
    text, result = _write(harness.make_ctx(), ask="council")
    assert "PLAN_ASK_SKIPPED_BUDGET" in text and result.code == "CAPABILITY_UNAVAILABLE"
    assert sub.calls == [] and _record(harness) == plan_record.empty_record()


# ---------------------------------------------------------------- write and read

def test_write_records_a_revision_calls_nobody_and_identical_content_records_nothing(harness):
    sub = harness.install({})
    ctx = harness.make_ctx()
    text, result = _write(ctx)
    assert result.status == "ok" and result.meta["plan_effect"] == "write" and result.meta["plan_revision"] == 1
    assert "Revision 1 recorded" in text and "No ask yet" in text
    assert sub.calls == []
    record = _record(harness)
    assert len(record["revisions"]) == 1 and record["revisions"][0]["done_when"] == [{"id": "c1", "claim": "exactly 5 slides", "support": "count them"}]
    assert record["revisions"][0]["body"] is False
    text, result = _write(ctx)
    assert "Identical content" in text and len(_record(harness)["revisions"]) == 1
    text, result = _call(ctx, plan="Draft the outline, then the slides, then the notes.")
    record = _record(harness)
    assert result.meta["plan_revision"] == 2 and record["revisions"][1]["goal"] == "Ship the deck"  # carried over
    assert record["revisions"][1]["affected_paths"] == [] and record["revisions"][1]["done_when"] == record["revisions"][0]["done_when"]
    assert any("revision 2 recorded" in line for line in harness.progress)


def test_read_is_free_and_shows_the_state(harness):
    sub = harness.install({})
    ctx = harness.make_ctx()
    text, result = _call(ctx)
    assert "No plan revision is recorded" in text and result.meta["plan_effect"] == "read"
    _write(ctx)
    text, result = _call(ctx)
    assert "# Plan — revision 1 of 1" in text and "exactly 5 slides" in text and "Nothing holds your final answer" in text
    assert sub.calls == []


def test_body_plan_is_computed_at_write_from_affected_paths(harness):
    harness.install({})
    ctx = harness.make_ctx(active_workspace=False)
    _write(ctx, affected_paths=["ouroboros/loop.py"])
    revision = _record(harness)["revisions"][0]
    assert revision["body"] is True and "constitutional" in revision["body_note"]
    _write(harness.make_ctx(task_id="task-2"), affected_paths=[str(harness.workspace / "notes.md")])
    assert _record(harness, "task-2")["revisions"][0]["body"] is False


def test_a_declared_new_source_file_is_a_body_plan_before_it_exists(harness):
    """A file the work will CREATE has no bytes and no parent directory on disk yet; writing
    `ouroboros/new/deep/module.py` IS self-modification, so the plan is a body plan and the
    constitutional pack rides (owner 8=A / 9=A, after a live wave)."""
    target = harness.system / "ouroboros" / "new" / "deep" / "module.py"
    assert not target.exists() and not target.parent.exists()
    sub = harness.install({})
    _write(harness.make_ctx(), affected_paths=[str(target)], ask=["s1"])
    revision = _record(harness)["revisions"][0]
    assert revision["body"] is True and "affected_paths" in revision["body_note"] and str(target) in revision["body_note"]
    system_prompt = _user_text(sub.calls[0]["request"].slot_messages["s1"][0]["content"])
    assert "## BIBLE.md (constitution" in system_prompt and "## ARCHITECTURE.md" in system_prompt


def test_reading_a_repository_file_is_not_changing_it(harness):
    """Owner 16=A. The same locator, twice: attached as evidence alone the plan stays ordinary and
    carries no pack; declared as a change target it is a body plan with the pack."""
    bible = str(harness.system / "BIBLE.md")
    sub = harness.install({})
    _write(harness.make_ctx(), affected_paths=[], attach=[bible], ask=["s1"])
    assert _record(harness)["revisions"][0]["body"] is False
    assert "## BIBLE.md (constitution" not in _user_text(sub.calls[0]["request"].slot_messages["s1"][0]["content"])
    sub = harness.install({})
    _write(harness.make_ctx(task_id="task-2"), affected_paths=[bible], ask=["s1"])
    assert _record(harness, "task-2")["revisions"][0]["body"] is True
    assert "## BIBLE.md (constitution" in _user_text(sub.calls[0]["request"].slot_messages["s1"][0]["content"])


# ---------------------------------------------------------------- ask

def test_ask_council_sends_each_participant_its_own_delivery_and_settles_replies(harness):
    sub = harness.install({"s1": _reply("The bar is low.", objections=[("o1", "five slides is not the bar; the owner asked for a story")]),
                           "s2": CLEAN, "s3": _reply("Fine.")})
    ctx = harness.make_ctx()
    text, result = _write(ctx, ask="council", message="First look, please.")
    assert result.meta["plan_effect"] == "write+ask" and result.meta["plan_open_objections"] == 1 and result.meta["plan_holding"] == 1
    assert len(sub.calls) == 1 and [s.slot_id for s in sub.calls[0]["slots"]] == ["s1", "s2", "s3"]
    request = sub.calls[0]["request"]
    assert request.surface == "plan_review" and request.reconciliation_identity["plan_ask_id"] == "a1"
    assert request.policy["output_contract"].startswith("Return ONE JSON object")
    for sid in ("s1", "s2", "s3"):
        user = _user_text(request.slot_messages[sid][-1]["content"])
        assert "## TASK OBJECTIVE" in user and "Deliver the thing" in user and "## THE PLAN" in user
        assert "exactly 5 slides" in user and "First look, please." in user and "first ask on this plan" in user
        assert "You have no open objection" in user
    system_prompt = _user_text(request.slot_messages["s1"][0]["content"])
    assert "FIXTURE CHECKLIST ROW" in system_prompt and "## Reply contract" in system_prompt
    assert "## BIBLE.md" not in system_prompt and "navigation map" not in system_prompt  # non-body: no pack, no maps
    record = _record(harness)
    ask = record["asks"][0]
    assert ask["ask_id"] == "a1" and ask["revision"] == 1 and ask["message"] == "First look, please."
    assert [(p["seat_id"], p["status"], p["owner_marked"]) for p in ask["participants"]] == [
        ("s1", "answered", True), ("s2", "answered", True), ("s3", "answered", True)]
    assert ask["participants"][0]["reply"]["objections"] == [{"id": "o1", "text": "five slides is not the bar; the owner asked for a story"}]
    assert ask["participants"][0]["delivery_ref"]["status"] == "retained" and ask["participants"][0]["reply_ref"]["status"] == "retained"
    [objection] = plan_record.open_objections(record)
    assert objection["seat_id"] == "s1" and objection["holds"] is True
    assert "s1 / o1" in text and "holds your final answer under Blocking" in text and "HELD" in text
    assert any("ask a1 on revision 1 — sending to 3 participants" in line for line in harness.progress)
    assert any("m/a answered — 1 objection" in line for line in harness.progress)
    # The ask's exact texts and costs live in ONE intention ledger record.
    from ouroboros.review_ledger import load_record

    ledger = load_record(harness.drive, ask["record_id"])
    assert ledger["surface"] == "intention" and ledger["state"] == "settled" and ledger["verdict"]["aggregate"] == "NOT_PERFORMED"
    assert ledger["subject"]["plan_ask"] == "a1" and ledger["subject"]["plan_revision"] == 1
    rows = {r["seat_id"]: r for r in ledger["rows"]}
    assert rows["s1"]["intention"]["objections"] == ["o1"] and rows["s1"]["status"] == "responded" and rows["s1"]["usd"] == 0.01
    assert {ref["role"] for ref in rows["s1"]["source_refs"]} >= {"prompt", "response"}
    assert ledger["cost"]["usd"] == pytest.approx(0.03)


def test_ask_names_exactly_its_recipients_and_an_added_critic_never_holds(harness):
    row = SimpleNamespace(slot_id="critic_x", kind="api_chat", target_id="m/critic", effort="", session_target="", profile_id="",
                          subagent_id="critic_x", use_local=False, processing_preference="", delivery="packet", access="",
                          is_session=False, native_retrieval=False)
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config.catalog_review_row",
                                lambda _snapshot, name: row if name == "critic_x" else (_ for _ in ()).throw(ValueError("unknown")))
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config._delivery_slot",
                                lambda r, **kw: _slots(("critic_x", r.target_id))[0])
    sub = harness.install({"critic_x": _reply("No!", objections=[("k1", "wrong shape")]), "s2": CLEAN})
    ctx = harness.make_ctx()
    text, result = _write(ctx, ask=["s2", "critic_x"])
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s2", "critic_x"]
    ask = _record(harness)["asks"][0]
    assert [(p["seat_id"], p["owner_marked"]) for p in ask["participants"]] == [("s2", True), ("critic_x", False)]
    [objection] = plan_record.open_objections(_record(harness))
    assert objection["holds"] is False and result.meta["plan_holding"] == 0 and result.meta["plan_open_objections"] == 1
    assert "added participant — never holds" in text and "Open objections are disclosed in your final answer; nothing holds it" in text


def test_a_body_plans_first_ask_includes_every_owner_marked_reviewer_and_says_so(harness):
    sub = harness.install({})
    ctx = harness.make_ctx(active_workspace=False)
    text, _ = _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["s2"])
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s2", "s1", "s3"]
    ask = _record(harness)["asks"][0]
    assert ask["body_first_ask"] is True and ask["host_added"] == ["s1", "s3"]
    assert ask["panel"]["widened_from"] == ["s2"] and ask["panel"]["strictness"] == "blocking"
    assert "first ask on a body plan" in text and "added s1, s3" in text
    system_prompt = _user_text(sub.calls[0]["request"].slot_messages["s2"][0]["content"])
    assert "## BIBLE.md (constitution" in system_prompt and "## ARCHITECTURE.md" in system_prompt
    # The second ask is the mind's choice.
    text, _ = _call(ctx, ask=["s2"], message="again")
    assert [s.slot_id for s in sub.calls[1]["slots"]] == ["s2"] and _record(harness)["asks"][1]["host_added"] == []


def test_advisory_first_body_ask_requires_reason_and_preserves_the_named_panel(harness):
    harness.monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    sub = harness.install({})
    ctx = harness.make_ctx(active_workspace=False)
    text, result = _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["s2"])
    assert "REVIEW_PANEL_REASON_REQUIRED" in text and result.code == "TOOL_ARG_ERROR"
    assert sub.calls == [] and _record(harness) == plan_record.empty_record()
    _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["s2"], reason="one area changed")
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s2"]
    ask = _record(harness)["asks"][0]
    assert ask["host_added"] == [] and ask["panel"]["widened_from"] == []
    assert ask["reason"] == ask["panel"]["reason"] == "one area changed"
    from ouroboros.review_ledger import load_record
    assert load_record(harness.drive, ask["record_id"])["panel"]["reason"] == "one area changed"


def test_first_body_ask_adviser_alone_refuses_in_advisory_and_is_free_in_cyber(harness):
    critic = _slots(("critic", "m/critic"))[0]
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config.catalog_review_row", lambda *_: critic)
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config._delivery_slot", lambda *_a, **_kw: critic)
    harness.monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    sub = harness.install({"critic": _reply("A concern", objections=[("x", "explain this")])})
    ctx = harness.make_ctx(active_workspace=False)
    text, result = _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["critic"], reason="optional critic")
    assert "REVIEW_PANEL_EMPTY" in text and result.code == "TOOL_ARG_ERROR"
    assert sub.calls == [] and _record(harness) == plan_record.empty_record()
    harness.monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    _, result = _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["critic"])
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["critic"]
    assert result.meta["plan_holding"] == 0


def test_later_ask_continues_each_participant_with_delta_cross_talk_and_own_objections(harness):
    sub = harness.install({"s1": _reply("Low bar.", objections=[("o1", "the bar is low")]), "s2": _reply("Peer says fine."), "s3": CLEAN})
    ctx = harness.make_ctx()
    _write(ctx, ask="council")
    sub.answers["s1"] = _reply("Better.", withdrawn=[("o1", "revision 2 raised the bar")])
    text, result = _call(ctx, done_when=[{"id": "c1", "claim": "a 5-slide story with one claim per slide", "support": "read it"}],
                         ask=["s1"], message="I raised the bar; is o1 answered?")
    request = sub.calls[1]["request"]
    assert [s.slot_id for s in sub.calls[1]["slots"]] == ["s1"]
    transcript = request.slot_messages["s1"]
    assert [m["role"] for m in transcript] == ["system", "user", "assistant", "user"]  # its own exact prior transcript continues
    assert "Low bar." in transcript[2]["content"]
    user = _user_text(transcript[-1]["content"])
    assert "Changes since revision 1" in user and "done_when reworded: c1" in user and "a 5-slide story" in user
    assert "### Ask a1" in user and "s2 (m/b) answered:" in user and "Peer says fine." in user  # cross-talk since its last answer
    assert "## YOUR OPEN OBJECTIONS" in user and "- o1 (raised in ask a1 about revision 1): the bar is low" in user
    assert "I raised the bar; is o1 answered?" in user
    assert plan_record.open_objections(_record(harness)) == [] and result.meta["plan_holding"] == 0
    assert "withdrawn: s1/o1 in ask a2" in text and "Nothing holds your final answer" in text
    ask = _record(harness)["asks"][1]
    assert ask["participants"][0]["continued"] is True


def test_omission_flags_not_adjudicated_and_another_seat_cannot_withdraw(harness):
    sub = harness.install({"s1": _reply("Low bar.", objections=[("o1", "the bar is low")]), "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    _write(ctx, ask="council")
    sub.answers.update({"s1": _reply("Something else entirely."), "s2": _reply("I withdraw s1's point.", withdrawn=[("o1", "I disagree with it")])})
    text, result = _call(ctx, ask=["s1", "s2"], message="again")
    [objection] = plan_record.open_objections(_record(harness))
    assert objection["not_adjudicated"] is True and objection["holds"] is True
    assert "NOT ADJUDICATED" in text and "ignored withdrawals of objections the seat never raised: s2/o1 in ask a2" in text
    assert result.meta["plan_holding"] == 1


def test_unreadable_reply_and_a_failed_seat_change_nothing_and_never_hold(harness):
    sub = harness.install({"s1": "I object to everything, in prose.", "s2": "", "s3": _reply("ok")})
    text, result = _write(harness.make_ctx(), ask="council")
    ask = _record(harness)["asks"][0]
    status = {p["seat_id"]: p["status"] for p in ask["participants"]}
    assert status == {"s1": "unparseable", "s2": "failed", "s3": "answered"}
    assert ask["participants"][0]["parse_error"] and ask["participants"][1]["error"] == "transport died"
    assert result.meta["plan_open_objections"] == 0 and "unparseable" in text and "failed — transport died" in text
    assert any("could not be read as the reply object" in line for line in harness.progress)
    assert len(sub.calls) == 1


def test_pending_seats_settle_from_the_custody_hook_and_wake_the_mailbox(harness):
    pending = {"status": "pending", "raw_text": "", "operation_state": "pending_dispatch", "late_result_pending": True,
               "awaiting_since": "2026-10-15T00:00:00+00:00"}
    sub = harness.install({"s1": pending, "s2": pending, "s3": CLEAN})
    ctx = harness.make_ctx()
    text, result = _write(ctx, ask="council")
    record = _record(harness)
    assert [p["status"] for p in record["asks"][0]["participants"]] == ["pending", "pending", "answered"]
    assert result.meta["plan_pending_asks"] == 1 and "Still working: s1, s2" in text
    decision = plan_record.gate_projection(record, enforcement="blocking", cyber_pro=False, hurry=False)
    assert decision["status"] == "awaiting" and decision["allow"] is True  # silence never holds
    from ouroboros.review_records import ReviewActorRecord

    request = sub.calls[0]["request"]
    slots = {s.slot_id: s for s in sub.calls[0]["slots"]}
    actor = ReviewActorRecord(slot_id="s1", model="m/a", status="ok", raw_text=_reply("Late answer.", objections=[("o9", "late objection")]),
                              operation_id="op-s1-1", operation_state="late_settled", usage={"cost": 0.02, "resolved_model": "m/a"})
    plan_ask.settle_released_reply(ctx, request=request, task_id="task-1", actor=actor, slot=slots["s1"], settled_wave={}, roster_size=3)
    record = _record(harness)
    s1 = record["asks"][0]["participants"][0]
    assert s1["status"] == "answered" and s1["reply"]["objections"] == [{"id": "o9", "text": "late objection"}] and s1["cost"] == 0.02
    def frames_about(ask_id):
        from ouroboros.owner_mailbox import _mailbox_path, mailbox_lines

        path = _mailbox_path(harness.drive, "task-1")
        rows = [json.loads(line) for line in mailbox_lines(path.read_text(encoding="utf-8"))] if path.exists() else []
        return [m for m in rows if f"Plan ask {ask_id}" in str(m.get("text") or "")]

    assert frames_about("a1") == []
    failed = ReviewActorRecord(slot_id="s2", model="m/b", status="error", error="run ended failed", operation_id="op-s2-1",
                               operation_state="settled", failure_code="run_failed")
    plan_ask.settle_released_reply(ctx, request=request, task_id="task-1", actor=failed, slot=slots["s2"],
                                   settled_wave={"s1": "ok", "s2": "error", "s3": "ok"}, roster_size=3)
    record = _record(harness)
    assert record["asks"][0]["participants"][1]["status"] == "failed"
    frames = frames_about("a1")
    assert len(frames) == 1 and "2 answered, 1 failed" in frames[0]["text"] and "1 open objection" in frames[0]["text"]
    from ouroboros.review_ledger import load_record

    ledger = load_record(harness.drive, record["asks"][0]["record_id"])
    assert ledger["state"] == "settled" and {r["seat_id"]: r["status"] for r in ledger["rows"]} == {"s1": "responded", "s2": "error", "s3": "responded"}
    # A late pending echo never re-opens a settled seat.
    echo = ReviewActorRecord(slot_id="s1", model="m/a", status="error", operation_id="op-s1-1", operation_state="in_flight", late_result_pending=True)
    plan_ask.settle_released_reply(ctx, request=request, task_id="task-1", actor=echo, slot=slots["s1"], settled_wave={}, roster_size=3)
    assert _record(harness)["asks"][0]["participants"][0]["status"] == "answered"


def test_an_earlier_schema_wave_settles_nothing_and_writes_no_frame(harness, monkeypatch):
    """A released seat of a v2 wave (no ``plan_ask_id``) and its late result stay history in the
    operation's custody: no collector, no plan record write, no mailbox frame. The organ's own
    ask settles through the same hook (pinned above); another surface is never this hook's."""
    frames = []
    monkeypatch.setattr("ouroboros.owner_mailbox.write_task_message", lambda *a, **k: frames.append(a))
    ctx = harness.make_ctx()
    _write(ctx)
    before = _record(harness)
    for surface in ("plan_review", "task_acceptance"):
        plan_ask.settle_released_reply(ctx, request=SimpleNamespace(surface=surface, reconciliation_identity={"subject_hash": "f" * 64}),
                                       task_id="task-1", actor=SimpleNamespace(slot_id="s1"), settled_wave={"s1": "ok"},
                                       roster_size=1)
    assert frames == [] and _record(harness) == before
    calls = []
    monkeypatch.setattr("ouroboros.observability.read_call_payload",
                        lambda *a, **k: calls.append(k) or ({}, {"request": {"reconciliation_identity": {"subject_hash": "f" * 64}}}, []))
    assert plan_ask.attach_late_reply(harness.drive, {"surface": "plan_review", "task_id": "task-1", "operation_id": "op-v2"}) == 0
    assert calls and _record(harness) == before
    assert plan_ask.attach_late_reply(harness.drive, {"surface": "task_acceptance", "task_id": "task-1", "operation_id": "op-v2"}) == 0
    assert len(calls) == 1  # another surface is not read at all


# ---------------------------------------------------------------- lane health, evidence

def test_a_structurally_dead_seat_is_not_dispatched_and_unknown_health_dispatches_everyone(harness, monkeypatch):
    """Positive structural evidence (a spent window with a future reset, a dead pool) keeps a seat
    out of the dispatch as a typed $0 not-dispatched participant; a snapshot that could not be
    taken (None) is unknown health and every named participant is sent."""
    import ouroboros.tools.plan_review_runtime as runtime

    dead = {"s2": {"failure_code": "subscription_window_exhausted", "reset_at": "2030-01-02T00:00:00+00:00"},
            "s3": {"failure_code": "credential_pool_exhausted", "reset_at": "2030-01-01T00:00:00+00:00"}}
    monkeypatch.setattr(runtime, "plan_panel_health_snapshot", lambda slots: dict(dead))
    sub = harness.install({"s1": CLEAN})
    text, result = _write(harness.make_ctx(), ask="council")
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s1"]
    rows = {p["seat_id"]: p for p in _record(harness)["asks"][0]["participants"]}
    assert rows["s1"]["dispatched"] is True and rows["s1"]["status"] == "answered"
    for sid, code in (("s2", "subscription_window_exhausted"), ("s3", "credential_pool_exhausted")):
        assert rows[sid]["dispatched"] is False and rows[sid]["status"] == "not_dispatched"
        assert f"health_skip[{code}]" in rows[sid]["not_dispatched_reason"]
    assert "not sent: s2 — health_skip[subscription_window_exhausted]" in text and result.meta["plan_holding"] == 0

    monkeypatch.setattr(runtime, "plan_panel_health_snapshot", lambda slots: None)
    sub = harness.install({})
    _write(harness.make_ctx(task_id="task-2"), ask="council")
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s1", "s2", "s3"]
    assert all(p["dispatched"] for p in _record(harness, "task-2")["asks"][0]["participants"])


def test_a_typed_lane_refusal_reaches_the_participant_record_and_an_untyped_failure_stays_absent(harness):
    """A typed lane refusal (code, reset) rides substrate -> plan row -> participant record; a
    prose-only failure (an engine emitting no code) records no code: absence stays absence."""
    harness.install({"s2": {"status": "error", "raw_text": "", "error": "delegated review session run-9 ended failed",
                            "failure_code": "subscription_window_exhausted", "reset_at": "2030-01-01T00:00:00Z"},
                     "s3": {"status": "error", "raw_text": "", "error": "transport died"}})
    _write(harness.make_ctx(), ask="council")
    rows = {p["seat_id"]: p for p in _record(harness)["asks"][0]["participants"]}
    assert rows["s1"]["status"] == "answered" and rows["s1"]["failure_code"] == ""
    assert (rows["s2"]["status"], rows["s2"]["failure_code"], rows["s2"]["reset_at"]) == (
        "failed", "subscription_window_exhausted", "2030-01-01T00:00:00Z")
    assert (rows["s3"]["status"], rows["s3"]["failure_code"], rows["s3"]["reset_at"]) == ("failed", "", "")
    assert rows["s3"]["error"] == "transport died"


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_a_session_participant_reads_the_same_exact_evidence_inline_as_a_packet_seat(harness, newline):
    """A retrieving session seat sees the SAME evidence bytes a packet seat sees, and those are the
    file's own bytes: a secret-shaped line in an attached file is not rewritten for either, and its
    line endings are not normalised."""
    line = "ANTHROPIC_API_KEY=sk-ant-FAKEFIXTURE-not-a-real-key-ABCDEFGHIJKLMNOP"
    expected = f"plan notes{newline}{line}{newline}"
    # Bytes, not text mode: text mode would translate newlines on Windows; the fixture fixes them.
    (harness.workspace / "notes.md").write_bytes(expected.encode("utf-8"))
    harness.state["slots"] = _slots(("api1", "m/a"), ("sess1", "cursor=grok", "session"))
    sub = harness.install({})
    _write(harness.make_ctx(), ask="council", attach=["notes.md"])
    request = sub.calls[0]["request"]
    session = str(request.slot_session_tasks["sess1"])
    assert request.session_root == str(harness.workspace)
    assert "RETRIEVING REVIEWER" in session and "the same bytes every reviewer sees" in session
    assert "REDACTED" not in session and expected in session
    packet = _user_text(request.slot_messages["api1"][-1]["content"])
    assert expected in packet and "secrets_redacted" not in packet
    assert request.policy["output_contract"].startswith("Return ONE JSON object")


def test_the_runtime_data_plane_is_never_attached_whatever_root_the_ask_names(harness, monkeypatch):
    """The live data root and settings file are denied as evidence (``plan_ask._evidence_deny_paths``
    is wired into the attachment manifest); an ordinary workspace file still attaches."""
    from ouroboros import config as cfg

    data_root = harness.workspace / "live_data"
    data_root.mkdir()
    (data_root / "settings.json").write_text('{"OPENROUTER_API_KEY": "sk-live-x"}', encoding="utf-8")
    monkeypatch.setattr(cfg, "DATA_DIR", data_root, raising=False)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", data_root / "settings.json", raising=False)
    ctx = harness.make_ctx()
    assert any(str(data_root) in path for path in plan_ask._evidence_deny_paths(ctx))
    sub = harness.install({})
    _write(ctx, ask="council", attach=["live_data/settings.json", "notes.md"])
    ask = _record(harness)["asks"][0]
    assert {"locator": "live_data/settings.json", "reason": "denied_path"} in ask["omissions"]
    assert [a["locator"] for a in ask["attached"]] == ["notes.md"]
    assert "sk-live-x" not in json.dumps([c["request"].slot_messages for c in sub.calls], default=str)


def test_an_asks_reviewer_effort_orders_the_pool_of_that_ask_only(harness, monkeypatch):
    """``reviewer_effort`` reaches the review pool's builder as the order of THIS ask (its
    clamping and pin rules are ``plan_review_slots``' own) and the ask row records it; an ask
    without one leaves every row at its own level."""
    seen = []
    monkeypatch.setattr(plan_ask, "plan_review_slots",
                        lambda default_effort="": seen.append(default_effort) or list(harness.state["slots"]))
    harness.install({})
    ctx = harness.make_ctx()
    _write(ctx, ask="council", reviewer_effort="low")
    _call(ctx, ask="council")
    assert seen == ["low", ""]
    asks = _record(harness)["asks"]
    assert [a["reviewer_effort"] for a in asks] == ["low", ""]


def test_the_deadline_rail_states_facts_and_never_advises_a_plan_of_its_own(harness):
    """The deadline rail names the remaining time and that no participant was sent; it never
    advises the mind to proceed with a plan of its own, and no deadline is no skip."""
    from ouroboros.deadline_utils import utc_now
    from ouroboros.tools.plan_review_runtime import plan_deadline_skip

    ctx = harness.make_ctx()
    assert plan_deadline_skip(ctx) == ""
    ctx.task_metadata["deadline_at"] = utc_now().isoformat()
    text = plan_deadline_skip(ctx)
    assert text.startswith("PLAN_TASK_SKIPPED_DEADLINE:") and "remaining time 0s" in text
    assert "This call dispatched no reviewer" in text and "The plan record keeps its state" in text
    assert "no plan review is open" not in text and "proceed with your own" not in text.lower()
