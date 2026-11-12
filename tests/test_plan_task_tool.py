"""``plan_task`` — write / ask / read over the plan record, driven through the real
ToolContext, the real plan record and a FAKE review substrate. Pinned in both
directions: typed $0 refusals write nothing and call nobody; a write records a
revision and calls nobody; an ask sends exactly to the named participants with each
participant's own delivery and continuation; replies settle into the record and the
intention ledger record; only the raiser's withdrawal closes an objection, which holds nothing."""
from __future__ import annotations

import json
import pathlib
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
    # LF on every host, as a checkout keeps them (.gitattributes): attachments carry a file's own bytes, and a Windows
    # text-mode write would make them CRLF.
    system = tmp_path / "repo"
    (system / "docs").mkdir(parents=True)
    (system / "BIBLE.md").write_text("# BIBLE.md\n\n## Principle 0: Agency\n\nbe.\n", encoding="utf-8", newline="\n")
    (system / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros vX — Architecture\n\n## 1. Runtime\n\nthe loop.\n",
                                                     encoding="utf-8", newline="\n")
    (system / "docs" / "CHECKLISTS.md").write_text(
        "## Plan Review Checklist\n\nFIXTURE CHECKLIST ROW: the bar.\n\n## Coupling questions\n\nx\n", encoding="utf-8",
        newline="\n")
    (system / "ouroboros").mkdir()
    (system / "ouroboros" / "loop.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("deck notes\n", encoding="utf-8", newline="\n")
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
    _write(harness.make_ctx(), affected_paths=[str(target)], ask=["s1"], reason="one module")
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
    _write(harness.make_ctx(task_id="task-2"), affected_paths=[bible], ask=["s1"], reason="one file")
    assert _record(harness, "task-2")["revisions"][0]["body"] is True
    assert "## BIBLE.md (constitution" in _user_text(sub.calls[0]["request"].slot_messages["s1"][0]["content"])


# ---------------------------------------------------------------- ask

def test_ask_council_sends_each_participant_its_own_delivery_and_settles_replies(harness):
    sub = harness.install({"s1": _reply("The bar is low.", objections=[("o1", "five slides is not the bar; the owner asked for a story")]),
                           "s2": CLEAN, "s3": _reply("Fine.")})
    ctx = harness.make_ctx()
    text, result = _write(ctx, ask="council", message="First look, please.")
    assert result.meta["plan_effect"] == "write+ask" and result.meta["plan_open_objections"] == 1 and "plan_holding" not in result.meta
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
    assert objection["seat_id"] == "s1" and objection["owner_marked"] is True and "holds" not in objection
    assert "s1 / o1" in text and "owner-marked reviewer" in text and "HELD" not in text
    assert "Open objections are disclosed in your final answer; nothing holds it" in text
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


def test_ask_names_exactly_its_recipients_and_an_added_critic_is_named_as_added(harness):
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
    assert objection["owner_marked"] is False and result.meta["plan_open_objections"] == 1
    assert "added participant" in text and "Open objections are disclosed in your final answer; nothing holds it" in text


def test_a_first_body_ask_requires_reason_and_preserves_the_named_panel(harness):
    sub = harness.install({})
    ctx = harness.make_ctx(active_workspace=False)
    text, result = _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["s2"])
    assert "REVIEW_PANEL_REASON_REQUIRED" in text and result.code == "TOOL_ARG_ERROR"
    assert sub.calls == [] and _record(harness) == plan_record.empty_record()
    _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["s2"], reason="one area changed")
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s2"]
    ask = _record(harness)["asks"][0]
    assert ask["host_added"] == [] and "widened_from" not in ask["panel"]
    assert ask["reason"] == ask["panel"]["reason"] == "one area changed"
    from ouroboros.review_ledger import load_record
    assert load_record(harness.drive, ask["record_id"])["panel"]["reason"] == "one area changed"


def test_first_body_ask_adviser_alone_refuses_outside_cyber_and_is_free_in_cyber(harness):
    critic = _slots(("critic", "m/critic"))[0]
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config.catalog_review_row", lambda *_: critic)
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config._delivery_slot", lambda *_a, **_kw: critic)
    sub = harness.install({"critic": _reply("A concern", objections=[("x", "explain this")])})
    ctx = harness.make_ctx(active_workspace=False)
    text, result = _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["critic"], reason="optional critic")
    assert "REVIEW_PANEL_EMPTY" in text and result.code == "TOOL_ARG_ERROR"
    assert sub.calls == [] and _record(harness) == plan_record.empty_record()
    harness.monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    _, result = _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["critic"])
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["critic"]
    assert result.meta["plan_open_objections"] == 1  # disclosed; nothing holds it


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
    assert plan_record.open_objections(_record(harness)) == [] and result.meta["plan_open_objections"] == 0
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
    assert objection["not_adjudicated"] is True and objection["owner_marked"] is True
    assert "NOT ADJUDICATED" in text and "ignored withdrawals of objections the seat never raised: s2/o1 in ask a2" in text
    assert result.meta["plan_open_objections"] == 1


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
    # A pending ask holds no ending; a reply after the task ends stays in its plan record and shows as late feedback
    # only where the room is known (``late_review_marks``): neither text promises an answer "before your final".
    for words in (text, pr._SCHEMA["description"]):
        for fact in ("a pending ask holds no ending", "kept in its plan record", "when the room is known"):
            assert fact in words, fact
        assert "before your final" not in words and "before an unhurried finish" not in words
    decision = plan_record.gate_projection(record, cyber_pro=False)
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
    assert "not sent: s2 — health_skip[subscription_window_exhausted]" in text and "plan_holding" not in result.meta

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
    is wired into the attachment manifest); an ordinary workspace file still attaches. The refusal keeps its
    code and says what to attach instead, once, to the author (the ask's result and a later read) and to
    the participants (their packet); the manifest row stays ``{locator, reason}``."""
    from ouroboros import config as cfg
    from ouroboros.tools.plan_packet import DENIED_ATTACH_HINT

    data_root = harness.workspace / "live_data"
    data_root.mkdir()
    (data_root / "settings.json").write_text('{"OPENROUTER_API_KEY": "sk-live-x"}', encoding="utf-8")
    monkeypatch.setattr(cfg, "DATA_DIR", data_root, raising=False)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", data_root / "settings.json", raising=False)
    ctx = harness.make_ctx()
    assert any(str(data_root) in path for path in plan_ask._evidence_deny_paths(ctx))
    sub = harness.install({})
    asked, _result = _write(ctx, ask="council", attach=["live_data/settings.json", "notes.md"])
    ask = _record(harness)["asks"][0]
    assert {"locator": "live_data/settings.json", "reason": "denied_path"} in ask["omissions"]
    assert [a["locator"] for a in ask["attached"]] == ["notes.md"]
    packets = json.dumps([c["request"].slot_messages for c in sub.calls], default=str)
    assert "sk-live-x" not in packets
    read, _result = _call(ctx)
    for shown in (asked, read):
        assert (f"  evidence not attached: live_data/settings.json (denied_path)\n  {DENIED_ATTACH_HINT}\n" in shown
                and shown.count(DENIED_ATTACH_HINT) == 1), shown
    assert "| live_data/settings.json | denied_path |" in packets and DENIED_ATTACH_HINT in packets
    assert DENIED_ATTACH_HINT.startswith("Settings and runtime files outside your own task stores cannot be attached "
                                         "directly. Use task:<id> for a helper's saved result, chat:<id> for dialogue, "
                                         "or a readable file in your own task_drive/artifact_store or workspace ")


@pytest.mark.parametrize("reason", ["missing", "outside_allowed_roots", "truncated_to_40000"])
def test_only_a_deny_root_says_what_to_attach_instead(reason):
    """The other side: every other omission keeps its code alone, and several denied locators get the
    sentence once per list, not once per row."""
    from ouroboros.tools.plan_packet import omission_hint, render_evidence

    omissions = [{"locator": "x", "reason": reason}]
    assert omission_hint(omissions) == ""
    assert "cannot be attached directly" not in render_evidence({"declared": ["x"], "omissions": omissions})
    denied = omissions + [{"locator": "y", "reason": "denied_path"}, {"locator": "z", "reason": "denied_path"}]
    assert render_evidence({"declared": ["x", "y", "z"], "omissions": denied}).count("cannot be attached directly") == 1


def _own_store(ctx, label):
    from ouroboros.tool_access import resource_root_path

    base = resource_root_path(ctx, label)
    base.mkdir(parents=True, exist_ok=True)
    return base


def test_own_task_store_files_attach_their_exact_bytes_for_packet_and_session_seats(harness):
    """#2756: the caller's own ``artifact_store`` and ``task_drive`` lie beneath the denied runtime data root, and
    their files still attach through ``read_file``'s read admission exactly as a workspace file does: the source's
    own bytes (CRLF and a secret-shaped line unchanged), its full hash, a line selection, a named cut and named
    absences, for a packet seat and a retrieving session seat alike."""
    import hashlib

    from ouroboros.tools.plan_evidence import EVIDENCE_PER_ITEM_BYTES

    raw = b"own planning evidence\r\nOPENROUTER_API_KEY=sk-or-FAKEFIXTURE-not-a-real-key\r\nthird line\r\n"
    big = b"y" * (EVIDENCE_PER_ITEM_BYTES + 7)
    ctx = harness.make_ctx()
    art, drv = _own_store(ctx, "artifact_store"), _own_store(ctx, "task_drive")
    for base in (art, drv):
        (base / "plan-notes.md").write_bytes(raw)
    (art / "big.md").write_bytes(big)
    (drv / "drafts").mkdir()
    a, d = str(art / "plan-notes.md"), str(drv / "plan-notes.md")
    harness.state["slots"] = _slots(("api1", "m/a"), ("sess1", "cursor=grok", "session"))
    sub = harness.install({})
    _write(ctx, ask="council", attach=[a, d, f"{d}::lines=2-2", str(art / "big.md"), str(drv / "drafts"),
                                       str(drv / "absent.md"), "notes.md"])
    ask = _record(harness)["asks"][0]
    digest = hashlib.sha256(raw).hexdigest()
    assert [(row["locator"], row["sha256"], row["bytes"]) for row in ask["attached"]] == [
        (a, digest, len(raw)), (d, digest, len(raw)), (f"{d}::lines=2-2", digest, len(raw)),
        (str(art / "big.md"), hashlib.sha256(big).hexdigest(), len(big)),
        ("notes.md", hashlib.sha256(b"deck notes\n").hexdigest(), len(b"deck notes\n"))]
    assert ask["omissions"] == [
        {"locator": str(art / "big.md"), "reason": f"truncated_to_{EVIDENCE_PER_ITEM_BYTES}"},
        {"locator": str(drv / "drafts"), "reason": "directory"},
        {"locator": str(drv / "absent.md"), "reason": "missing"}]
    request = sub.calls[0]["request"]
    packet = _user_text(request.slot_messages["api1"][-1]["content"])
    for delivery in (packet, str(request.slot_session_tasks["sess1"])):
        for locator in (a, d):
            assert f"----- BEGIN {locator} -----\n{raw.decode()}\n----- END {locator} -----" in delivery
        selected = f"----- BEGIN {d}::lines=2-2 -----\nOPENROUTER_API_KEY=sk-or-FAKEFIXTURE-not-a-real-key\r\n\n"
        assert selected in delivery and "REDACTED" not in delivery and "denied_path" not in delivery


@pytest.mark.parametrize("layout", ["host_child_drive", "external_execution_drive"])
def test_beneath_the_runtime_data_roots_only_the_callers_own_stores_attach(harness, monkeypatch, layout):
    """A forked task's own stores attach from its canonical and its execution drive. Everything else beneath a runtime
    data root (``runtime_data_roots``: the execution drive, either the host's child layout inside the canonical own
    ``task_drive`` or a drive outside every other root, which is itself a readable ``runtime_data`` root; the canonical
    budget drive; the configured data root) stays ``denied_path``: settings, memory and state, a sibling's store, the
    parent's store (which ``read_file`` reads by lineage) and a symlink out of an own store. An execution
    copy the read authority binds to its canonical copy is refused as such (that copy attaches by its own address).
    Sensitive names, lexical or resolved, are refused as anywhere; a real ``resource_policy`` protected artifact,
    outside Cyber Pro, by the read admission under its own code."""
    from ouroboros import config as cfg
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.runtime_mode_policy import runtime_mode_at_least
    from ouroboros.tools.core_file_tools import read_admission

    assert not runtime_mode_at_least(cfg.get_runtime_mode(), "cyber_pro")

    configured = harness.workspace / "live_data"
    (configured / "state").mkdir(parents=True)
    (configured / "settings.json").write_text('{"OPENROUTER_API_KEY": "sk-live-x"}', encoding="utf-8")
    (configured / "state" / "state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cfg, "DATA_DIR", configured, raising=False)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", configured / "settings.json", raising=False)
    canonical = harness.drive
    external = harness.workspace.parent / "exec" / "data"
    child = canonical / "task_drives" / "task-1" if layout == "host_child_drive" else external
    ctx = harness.make_ctx()
    ctx.drive_root, ctx.budget_drive_root = child, str(canonical)
    ctx.task_metadata.update(budget_drive_root=str(canonical), parent_task_id="parent-1")
    own_canonical, own_exec = task_artifact_dir_path(canonical, "task-1"), _own_store(ctx, "task_drive")
    capture = pathlib.Path("delegated_runs", "r1", "out.md")
    files = {"canonical": own_canonical / "canon.md", "execution": own_exec / "exec.md",
             "canonical_capture": own_canonical / capture,
             "execution_capture": _own_store(ctx, "artifact_store") / capture,
             "memory_exec": child / "memory" / "scratchpad.md", "memory_canon": canonical / "memory" / "identity.md",
             "sibling": canonical / "task_drives" / "task-2" / "plan.md",
             "parent": task_artifact_dir_path(canonical, "parent-1") / "plan.md",
             "env": own_canonical / ".env", "oracle": own_canonical / "oracle.md"}
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name} bytes\n", encoding="utf-8")
    for link, target in (("escape.md", "memory_canon"), ("alias.md", "env"), ("id_rsa", "execution")):
        (own_exec / link).symlink_to(files[target])
    ctx.task_contract["resource_policy"] = {"protected_artifacts": [
        {"id": "oracle", "role": "black_box_reference", "paths": [str(files["oracle"])]}]}
    assert read_admission(ctx, "artifact_store", str(files["parent"]))[2] is None  # readable by lineage, not attachable
    sub = harness.install({})
    links = [str(own_exec / link) for link in ("escape.md", "alias.md", "id_rsa")]
    _write(ctx, ask=["s1"], attach=[*map(str, files.values()), *links, str(configured / "settings.json"),
                                    str(configured / "state" / "state.json"), "notes.md"])
    ask = _record(harness)["asks"][0]
    assert [row["locator"] for row in ask["attached"]] == [
        str(files["canonical"]), str(files["execution"]), str(files["canonical_capture"]), "notes.md"]
    denied = ("memory_exec", "memory_canon", "sibling", "parent")
    assert {row["locator"]: row["reason"] for row in ask["omissions"]} == {
        **{str(files[name]): "denied_path" for name in denied}, str(files["execution_capture"]): "read_target_differs",
        str(files["env"]): "sensitive", str(files["oracle"]): "read_blocked:RESOURCE_POLICY_BLOCKED",
        links[0]: "denied_path", links[1]: "sensitive", links[2]: "sensitive",
        str(configured / "settings.json"): "denied_path", str(configured / "state" / "state.json"): "denied_path"}
    packet = _user_text(sub.calls[0]["request"].slot_messages["s1"][-1]["content"])
    assert "canonical_capture bytes" in packet and "sk-live-x" not in packet
    assert not any(f"{name} bytes" in packet for name in (*denied, "execution_capture", "env", "oracle"))


def test_a_protected_workspace_file_is_refused_while_its_neighbour_attaches(harness):
    """Every file locator goes through ``read_file``'s read admission, workspace files included: a real
    ``resource_policy`` protected artifact in the workspace, outside Cyber Pro, is refused under the admission's own
    code, and the ordinary file beside it attaches its bytes as before."""
    from ouroboros import config as cfg
    from ouroboros.runtime_mode_policy import runtime_mode_at_least

    assert not runtime_mode_at_least(cfg.get_runtime_mode(), "cyber_pro")
    (harness.workspace / "oracle.py").write_text("ANSWER = 42\n", encoding="utf-8")
    ctx = harness.make_ctx()
    ctx.task_contract["resource_policy"] = {"protected_artifacts": [
        {"id": "oracle", "role": "black_box_reference", "paths": ["oracle.py"]}]}
    sub = harness.install({})
    _write(ctx, ask=["s1"], attach=["oracle.py", "notes.md"])
    ask = _record(harness)["asks"][0]
    assert [row["locator"] for row in ask["attached"]] == ["notes.md"]
    assert ask["omissions"] == [{"locator": "oracle.py", "reason": "read_blocked:RESOURCE_POLICY_BLOCKED"}]
    packet = _user_text(sub.calls[0]["request"].slot_messages["s1"][-1]["content"])
    assert "ANSWER = 42" not in packet
    assert "----- BEGIN notes.md -----\ndeck notes\n\n----- END notes.md -----" in packet


def test_a_bound_candidate_keeps_serving_governance_files_attachable_with_their_own_bytes(harness):
    """With a body candidate bound, ``system_repo`` names the candidate while ``planning_roots`` keeps the serving
    checkout as the governance root, which attachments have always reached. A serving file attaches its own bytes,
    never the candidate's same-named file, through an exact ``system_repo`` binding to that root that the read
    admission still checks: a protected serving file and a sensitive name are refused, a link from the serving root
    into the data root stays denied, and the candidate's file attaches by its own address."""
    import hashlib

    from ouroboros.tools import plan_evidence

    serving, candidate = harness.system, harness.workspace.parent / "candidate"
    candidate.mkdir()
    (candidate / "BIBLE.md").write_bytes(b"# BIBLE.md\n\ncandidate wording\n")
    (serving / "oracle.md").write_text("serving oracle\n", encoding="utf-8")
    (serving / ".env").write_text("X=1\n", encoding="utf-8")
    (harness.drive / "memory").mkdir()
    (harness.drive / "memory" / "identity.md").write_text("identity-private-bytes\n", encoding="utf-8")
    (serving / "data-link.md").symlink_to(harness.drive / "memory" / "identity.md")
    ctx = harness.make_ctx()
    ctx.serving_repo_dir, ctx.repo_dir, ctx.system_repo_dir = serving, candidate, candidate  # as body_candidate.bind
    ctx.task_metadata["body_candidate"] = {"candidate_id": "c1", "path": str(candidate), "branch": "candidate",
                                           "base_sha": "0" * 40, "repo_dir": str(serving)}
    ctx.task_contract["resource_policy"] = {"protected_artifacts": [
        {"id": "oracle", "role": "black_box_reference", "paths": [str(serving / "oracle.md")]}]}
    serving_bible, candidate_bible = serving / "BIBLE.md", candidate / "BIBLE.md"
    # read_file's own root selection cannot reach the serving checkout while the candidate is system_repo
    assert plan_evidence.evidence_read_admission(ctx)(None, serving_bible.resolve()).startswith("read_blocked:")
    sub = harness.install({})
    _write(ctx, ask=["s1"], attach=[str(serving_bible), str(candidate_bible), str(serving / "oracle.md"),
                                    str(serving / ".env"), str(serving / "data-link.md")])
    ask = _record(harness)["asks"][0]
    assert [(row["locator"], row["sha256"]) for row in ask["attached"]] == [
        (str(serving_bible), hashlib.sha256(serving_bible.read_bytes()).hexdigest()),
        (str(candidate_bible), hashlib.sha256(candidate_bible.read_bytes()).hexdigest())]
    assert ask["omissions"] == [
        {"locator": str(serving / "oracle.md"), "reason": "read_blocked:RESOURCE_POLICY_BLOCKED"},
        {"locator": str(serving / ".env"), "reason": "sensitive"},
        {"locator": str(serving / "data-link.md"), "reason": "denied_path"}]
    packet = _user_text(sub.calls[0]["request"].slot_messages["s1"][-1]["content"])
    assert f"----- BEGIN {serving_bible} -----\n{serving_bible.read_text(encoding='utf-8')}\n----- END" in packet
    assert f"----- BEGIN {candidate_bible} -----\n# BIBLE.md\n\ncandidate wording\n\n----- END" in packet
    assert "serving oracle" not in packet and "identity-private-bytes" not in packet


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_a_serving_file_attaches_beside_a_bound_candidate_as_read_file_binds_it(harness, mode):
    """With a candidate bound, ``read_file``'s own selection decides a serving file first; the exact ``system_repo``
    binding to the serving root decides only what that selection does not admit as itself. In Cyber Pro
    ``user_files`` holds the checkout, so a Presence ceiling granting reads there and none of ``system_repo``
    attaches the serving file exactly as ``read_file`` reads it; in Advanced the checkout is no user file and both
    refuse. The candidate's same-named file stays refused in both: the ceiling grants no ``system_repo`` read."""
    from ouroboros.presence_authority import (PresenceCapabilityCeiling, PresenceResourceGrant, PresenceToolGrant,
                                             presence_ceiling_payload)
    from ouroboros.tools.registry import ToolRegistry

    harness.monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", mode)
    harness.monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(harness.system.parent))
    serving, candidate = harness.system, harness.workspace.parent / "candidate"
    candidate.mkdir()
    (candidate / "BIBLE.md").write_bytes(b"# BIBLE.md\n\ncandidate wording\n")
    ctx = harness.make_ctx()
    ctx.serving_repo_dir, ctx.repo_dir, ctx.system_repo_dir = serving, candidate, candidate  # as body_candidate.bind
    ctx.task_metadata["body_candidate"] = {"candidate_id": "c1", "path": str(candidate), "branch": "candidate",
                                           "base_sha": "0" * 40, "repo_dir": str(serving)}
    ceiling = PresenceCapabilityCeiling("helper", "a" * 64, "b" * 64, "c" * 64, "d" * 64, "main", 10,
                                        (PresenceToolGrant("read_file"), PresenceToolGrant("plan_task")),
                                        (PresenceResourceGrant("user_files", ("read",), "."),), "0" * 64)
    ctx.task_contract["capability_ceiling"] = presence_ceiling_payload(ceiling)
    bible = (serving / "BIBLE.md").resolve()
    registry = ToolRegistry(repo_dir=candidate, drive_root=harness.drive)
    registry.set_context(ctx)
    read = registry.execute("read_file", {"path": str(bible)})
    governance, subject = plan_ask.planning_roots(ctx)
    manifest = plan_ask.attachment_manifest(ctx, [str(bible), str(candidate / "BIBLE.md")], state_root=harness.drive,
                                            task_id="task-1", system_root=governance, active_root=subject)
    refused = {"locator": str(candidate / "BIBLE.md"), "reason": "read_blocked:ACCESS_BLOCKED"}
    if mode == "cyber_pro":
        assert "Principle 0: Agency" in read, read
        assert [(row["locator"], row["text"]) for row in manifest["attached"]] == [(str(bible), bible.read_text("utf-8"))]
        assert manifest["omissions"] == [refused]
    else:
        assert read.startswith("⚠️ USER_FILES_PATH_BLOCKED"), read
        assert manifest["attached"] == []
        assert manifest["omissions"] == [{"locator": str(bible), "reason": "read_blocked:ACCESS_BLOCKED"}, refused]


def test_a_presence_ceiling_decides_attachments_as_it_decides_their_reads(harness):
    """The read admission carries the Presence ceiling, for own stores and ordinary files alike: with only
    ``artifact_store`` reads granted, that store's file attaches; the ungranted ``task_drive`` file and a workspace
    file are refused with the admission's own code."""
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceResourceGrant, presence_ceiling_payload

    ctx = harness.make_ctx()
    ceiling = PresenceCapabilityCeiling("helper", "a" * 64, "b" * 64, "c" * 64, "d" * 64, "main", 10, (),
                                        (PresenceResourceGrant("artifact_store", ("read",), "."),), "0" * 64)
    ctx.task_contract["capability_ceiling"] = presence_ceiling_payload(ceiling)
    art, drv = _own_store(ctx, "artifact_store"), _own_store(ctx, "task_drive")
    (art / "granted.md").write_text("granted\n", encoding="utf-8", newline="\n")  # the attached text is its own bytes
    (drv / "ungranted.md").write_text("ungranted\n", encoding="utf-8", newline="\n")
    manifest = plan_ask.attachment_manifest(ctx, [str(art / "granted.md"), str(drv / "ungranted.md"), "notes.md"],
                                            state_root=harness.drive, task_id="task-1", system_root=harness.system,
                                            active_root=harness.workspace)
    assert [row["text"] for row in manifest["attached"]] == ["granted\n"]
    assert manifest["omissions"] == [{"locator": str(drv / "ungranted.md"), "reason": "read_blocked:ACCESS_BLOCKED"},
                                     {"locator": "notes.md", "reason": "read_blocked:ACCESS_BLOCKED"}]


def test_a_folderless_task_attaches_relative_locators_from_its_own_task_drive(harness):
    """Project work without a folder has its own ``task_drive`` as the subject (``folderless_scratch_dir``): a
    relative locator resolves there and reaches the packet with its own bytes, while a relative step out of the
    store into the data root stays ``denied_path``."""
    ctx = harness.make_ctx(active_workspace=False)
    ctx.project_id = "p1"
    drv = _own_store(ctx, "task_drive")
    (drv / "draft.md").write_bytes(b"folderless draft\r\n")
    (harness.drive / "memory").mkdir()
    (harness.drive / "memory" / "identity.md").write_text("identity-private-bytes\n", encoding="utf-8")
    sub = harness.install({})
    _write(ctx, ask=["s1"], attach=["draft.md", "../../memory/identity.md"])
    ask = _record(harness)["asks"][0]
    assert [row["locator"] for row in ask["attached"]] == ["draft.md"]
    assert ask["omissions"] == [{"locator": "../../memory/identity.md", "reason": "denied_path"}]
    request = sub.calls[0]["request"]
    packet = _user_text(request.slot_messages["s1"][-1]["content"])
    assert request.session_root == str(drv)
    assert "----- BEGIN draft.md -----\nfolderless draft\r\n\n----- END draft.md -----" in packet
    assert "identity-private-bytes" not in packet


def test_an_own_store_a_symlink_moved_admits_nothing_beneath_the_data_root(harness):
    """Only the host's own address of the store counts: a ``task_drive`` replaced by a symlink into the data root
    does not make what it points at attachable."""
    memory = harness.drive / "memory"
    memory.mkdir()
    (memory / "identity.md").write_text("who I am\n", encoding="utf-8")
    (harness.drive / "task_drives").mkdir()
    (harness.drive / "task_drives" / "task-1").symlink_to(memory, target_is_directory=True)
    locator = str(harness.drive / "task_drives" / "task-1" / "identity.md")
    manifest = plan_ask.attachment_manifest(harness.make_ctx(), [locator], state_root=harness.drive, task_id="task-1",
                                            system_root=harness.system, active_root=harness.workspace)
    assert manifest["attached"] == [] and manifest["omissions"] == [{"locator": locator, "reason": "denied_path"}]


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


def test_an_asks_mixed_evidence_reaches_every_seat_and_every_author_view_apart(harness, monkeypatch):
    """One ask attaching mixed evidence, through the real handler, preparation, record and readers: own
    ``artifact_store`` and ``task_drive`` files, a helper's saved result and a workspace selection attach (the
    artifact twice, the large drive file as its head); the settings file is refused and a link is never fetched.
    Packet, session and native seats are prepared the same bytes and the same named absences. The author sees,
    in the ask's result, the free read, the review history and the context index's plan-record reader, what
    attached (whole or in part) apart from what did not, each with its reason: a cut or a repeat beside content
    that attached is never counted as missing. A revision's ask keeps its own evidence; an ask an older writer
    stored without these fields reads as not recorded, never as nothing attached, and one that kept what attached
    but not why says its reasons are not recorded."""
    from dataclasses import replace

    from ouroboros import config as cfg
    from ouroboros.review_history_view import REVIEW_CONTEXT_INDEX_KEY, current_plan_history, review_context_updates
    from ouroboros.tools import control_task_results
    from ouroboros.tools.plan_evidence import EVIDENCE_PER_ITEM_BYTES
    from tests._store_shared import seed_result

    live = harness.workspace / "live_data"
    live.mkdir()
    (live / "settings.json").write_text('{"OPENROUTER_API_KEY": "sk-live-x"}', encoding="utf-8")
    monkeypatch.setattr(cfg, "DATA_DIR", live, raising=False)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", live / "settings.json", raising=False)
    ctx = harness.make_ctx()
    art, drv = _own_store(ctx, "artifact_store"), _own_store(ctx, "task_drive")
    brief_bytes = b"BRIEF-4K: keep the existing Plan details.\n"
    (art / "brief.md").write_bytes(brief_bytes)
    big = b"".join(f"{n:06d} drive log line\n".encode() for n in range(EVIDENCE_PER_ITEM_BYTES // 20 + 400))
    (drv / "long-log.md").write_bytes(big)
    seed_result(harness.drive, "helper-1", {"status": "completed", "result": "HELPER-RESULT-7Z: three call sites."})
    [api, sess, native] = _slots(("api1", "m/a"), ("sess1", "cursor=grok", "session"), ("nat1", "m/native"))
    harness.state["slots"] = [api, sess, replace(native, subagent_id="native-reader", native_retrieval_override=True)]
    sub = harness.install({})
    brief, log = str(art / "brief.md"), str(drv / "long-log.md")
    url = "https://example.com/spec"
    attach = [brief, log, "task:helper-1", brief, "live_data/settings.json", url, "notes.md::lines=1-1"]
    asked, _result = _write(ctx, ask="council", message="Mixed evidence.", attach=attach)

    ask = _record(harness)["asks"][0]
    assert ask["attach"] == attach
    assert [row["locator"] for row in ask["attached"]] == [brief, log, "task:helper-1", "notes.md::lines=1-1"]
    assert ask["omissions"] == [{"locator": log, "reason": f"truncated_to_{EVIDENCE_PER_ITEM_BYTES}"},
                                {"locator": brief, "reason": "duplicate_locator"},
                                {"locator": "live_data/settings.json", "reason": "denied_path"},
                                {"locator": url, "reason": "url_not_fetched"}]
    # Every seat is prepared the same evidence: the artifact once, the drive head, the helper's result, and the
    # absences in its table; the settings bytes reach none.
    request = sub.calls[0]["request"]
    deliveries = {"api1": _user_text(request.slot_messages["api1"][-1]["content"]),
                  "sess1": str(request.slot_session_tasks["sess1"]), "nat1": str(request.slot_session_tasks["nat1"])}
    for seat, delivery in deliveries.items():
        assert delivery.count(f"----- BEGIN {brief} -----\nBRIEF-4K") == 1, seat
        assert big[:EVIDENCE_PER_ITEM_BYTES].decode() in delivery and "HELPER-RESULT-7Z" in delivery, seat
        for row in ask["omissions"]:
            assert f"| {row['locator']} | {row['reason']} |" in delivery, (seat, row)
        assert "sk-live-x" not in delivery, seat
    assert ask["participants"][2]["slot"]["delivery"] == "native"

    whole = (f"  evidence attached (prepared for the participants): {brief} (source {len(brief_bytes)} bytes; "
             f"duplicate_locator); {log} (in part; source {len(big)} bytes; truncated_to_{EVIDENCE_PER_ITEM_BYTES}); task:helper-1 "
             f"(source {ask['attached'][2]['bytes']} bytes); notes.md::lines=1-1 (source 11 bytes)\n"
             f"  evidence not attached: live_data/settings.json (denied_path); {url} (url_not_fetched)\n")
    summary = ("Evidence prepared for the participants: 3 of 6 attached whole, 1 in part, 2 not attached; "
               "the ask below lists each locator and every reason.")
    assert summary in asked and whole in asked, asked

    # A revision's ask carries its own evidence; the earlier ask keeps what it carried.
    _call(ctx, plan="Outline, then the story.", ask=["api1", "sess1", "nat1"], attach=["task:helper-1", url, "absent.md"])
    revised = sub.calls[1]["request"]
    for seat in ("api1", "sess1", "nat1"):
        delivery = (_user_text(revised.slot_messages[seat][-1]["content"]) if seat == "api1" else str(revised.slot_session_tasks[seat]))
        assert "HELPER-RESULT-7Z" in delivery and f"| {url} | url_not_fetched |" in delivery and "| absent.md | missing |" in delivery
        assert "BRIEF-4K" not in delivery.split("## YOUR PANEL SEAT")[-1], seat
    # Asks an older writer stored: one without any evidence field, one that kept what attached but not the reasons.
    notes = {"locator": "notes.md", "sha256": "e" * 64, "bytes": 11}
    for row in ({"ask_id": "a3", "message": "older writer"},
                {"ask_id": "a4", "message": "older partial writer", "attach": ["notes.md", "b.md"], "attached": [notes]}):
        plan_record.append_ask(harness.drive, "task-1", {**row, "ts": "2026-11-07T12:00:00+00:00", "revision": 2, "participants": []})
    read, _result = _call(ctx)
    a1, a2, a3, a4 = (read.index(f"- ask {ask_id} on revision") for ask_id in ("a1", "a2", "a3", "a4"))
    assert a1 < read.index(whole) < a2, read
    second = ("\n  evidence attached (prepared for the participants): task:helper-1 "
              f"(source {ask['attached'][2]['bytes']} bytes)\n  evidence not attached: {url} (url_not_fetched); absent.md (missing)\n")
    assert a2 < read.index(second) < a3, read
    assert read[a3:a4].endswith("\n  evidence: not recorded for this ask\n"), read[a3:a4]
    assert ("\n  evidence attached (prepared for the participants): notes.md (completeness not recorded; source 11 bytes)\n  evidence not attached: b.md\n"
            "  evidence cuts and absences: not recorded for this ask\n") in read[a4:], read[a4:]
    assert "evidence omitted" not in read and read.count("cannot be attached directly") == 1  # the deny hint, once

    # The review history and the plan family's current history carry each ask's fields as recorded; the older ask
    # gains no empty list.
    record = _record(harness)
    history, _operative = current_plan_history(ctx)
    rounds = {row["plan_ask"]: row for row in history["rounds"]}
    assert rounds == {row["plan_ask"]: row for row in plan_record.record_history(harness.drive, "task-1", record)["rounds"]}
    for stored in record["asks"][:2]:
        assert {key: rounds[stored["ask_id"]][key] for key in ("attach", "attached", "omissions")} == {
            key: stored[key] for key in ("attach", "attached", "omissions")}
    assert not {"attach", "attached", "omissions"} & set(rounds["a3"])
    assert (rounds["a4"]["attach"], rounds["a4"]["attached"], "omissions" in rounds["a4"]) == (["notes.md", "b.md"], [notes], False)
    # The context index addresses the plan record for these facts; its named reader returns them exactly.
    updates, _identities = review_context_updates(ctx, families={"plan": ""})
    [index] = [row for row in updates if REVIEW_CONTEXT_INDEX_KEY in row]
    current = json.loads(index["content"].split("\n", 1)[1])["current"]
    [address] = [row for row in current["addresses"].values() if row["what"].startswith("this task's plan record")]
    assert "each locator cut or not attached with its reason" in address["what"]
    assert address["read"] == {"tool": "get_task_result", "arguments": {"task_id": "task-1", "include_authority": True}, "field": "plan_record"}
    authority = json.loads(control_task_results._get_task_result(ctx, **address["read"]["arguments"]))["authority"]
    assert authority[address["read"]["field"]]["asks"] == record["asks"]


@pytest.mark.parametrize("fields,state", [
    ({"attached": [], "omissions": []}, "unknown"),
    ({"attached": [{"locator": "unresolved.md", "bytes": 9}]}, "attached_unknown"),
])
def test_incomplete_preparation_facts_stay_visible(harness, fields, state):
    ctx = harness.make_ctx()
    _write(ctx)
    plan_record.append_ask(harness.drive, "task-1", {
        "ask_id": "a1", "ts": "2026-11-07T12:00:00+00:00", "revision": 1,
        "message": "incomplete record", "participants": [], "attach": ["unresolved.md"], **fields,
    })
    ask = _record(harness)["asks"][0]
    evidence = plan_record.ask_evidence(ask)
    assert len(evidence["items"]) == 1 and evidence["items"][0]["state"] == state
    assert evidence["counts"]["whole"] == 0
    rendered, _ = _call(ctx)
    assert "unresolved.md" in rendered and "not recorded" in rendered
