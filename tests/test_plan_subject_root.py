"""``plan_task``'s subject after ``prepare_self_change`` binds a separate body candidate.

A task's explicit workspace or its room's selected folder stays the root of relative
``affected_paths``, attachments and retrieving seats; a declared locator is body work when it
reaches a recognized body copy (that folder or the bound candidate) in either spelling; an
unavailable folder is refused, never replaced, while the unused candidate's checkout is not
required; a task without such a folder keeps the candidate. The evidence an ask could not
attach is named in the ask's result and in the free read. Real Git, a real candidate, the
real handler and plan record; only the review substrate (provider generation) is fake, and a
native seat's episode then runs for real over the exact request that substrate received.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import body_candidate, plan_record
from ouroboros.tools import plan_ask
from ouroboros.tools import review as _review  # noqa: F401 — binds its REPO_ROOT aliases before a test patches them
from tests.body_candidate_support import git, isolate, make_serving
from tests.test_plan_task_tool import CLEAN, _call, _reply, _slots, _Substrate, _user_text, _write

BRIEF = ".local/research/brief.md"


@pytest.fixture
def env(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    (serving / "docs").mkdir()
    (serving / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros — Architecture\n\n## 1. Runtime\n\nthe loop.\n", encoding="utf-8")
    (serving / "docs" / "CHECKLISTS.md").write_text("## Plan Review Checklist\n\nFIXTURE CHECKLIST ROW.\n", encoding="utf-8")
    git(serving, "add", "-A")
    git(serving, "commit", "-q", "-m", "governance")
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    # A separate clone of the body (recognized through its local origin) and a foreign project.
    clone = tmp_path / "ext-clone"
    git(tmp_path, "clone", "-q", str(serving), str(clone))
    foreign = tmp_path / "foreign"
    (foreign / "src").mkdir(parents=True)
    (foreign / "src" / "app.py").write_text("APP = 1\n", encoding="utf-8")
    git(foreign, "init", "-q")
    git(foreign, "remote", "add", "origin", "https://example.invalid/acme/app.git")
    for root in (clone, foreign):
        (root / BRIEF).parent.mkdir(parents=True)
        (root / BRIEF).write_text(f"integrator brief kept in {root.name}\n", encoding="utf-8")
    monkeypatch.setattr(plan_ask, "plan_review_slots", lambda default_effort="": _slots(("s1", "m/a"), ("sess1", "cursor=grok", "session")))
    monkeypatch.setattr("ouroboros.tools.plan_dialogue.session_input_limits", lambda _slots: {})
    monkeypatch.setattr("ouroboros.tools.plan_review_runtime.plan_panel_health_snapshot", lambda _slots: {})
    monkeypatch.setattr("ouroboros.tools.review_checklist.REPO_ROOT", serving)

    def ctx(task_id, **over):
        from ouroboros.tools.registry import ToolContext

        meta = {"root_task_id": task_id, "work_kind": "owner", **over.pop("meta", {})}
        return ToolContext(repo_dir=serving, system_repo_dir=serving, drive_root=data, task_id=task_id, branch_dev="ouroboros",
                           task_metadata=meta, task_contract={"objective": "Repair the plan root"}, **over)

    def prepare(context):
        """The real tool: binds a candidate and returns it (its receipt names the folder that stays in use)."""
        from ouroboros.tools.registry import ToolRegistry

        registry = ToolRegistry(repo_dir=serving, drive_root=data)
        registry.set_context(context)
        receipt = registry.execute("prepare_self_change", {})
        assert "OK: body candidate" in receipt, receipt
        candidate = pathlib.Path(body_candidate.descriptor(context)["path"]).resolve()
        assert body_candidate.is_bound(context) and candidate not in {clone.resolve(), foreign.resolve(), serving.resolve()}
        return candidate, receipt

    def install(answers=None):
        import ouroboros.review_substrate as rs

        sub = _Substrate(answers or {})
        monkeypatch.setattr(rs, "run_review_request", sub)
        return sub

    return SimpleNamespace(serving=serving, data=data, clone=clone, foreign=foreign, tmp=tmp_path, ctx=ctx,
                           prepare=prepare, install=install, monkeypatch=monkeypatch)


def _revisions(env, task_id):
    return plan_record.load_plan_record(env.data, task_id)["revisions"]


def _asks(env, task_id):
    return plan_record.load_plan_record(env.data, task_id)["asks"]


def _governed(env, request):
    """The serving body's pack reached every seat: inline for the packet seat, as mandatory reads for the session."""
    pack = "## Governance pack (this plan touches Ouroboros's own body) — MANDATORY FULL READS"
    bible = f"`{env.serving.resolve() / 'BIBLE.md'}`"
    return (all(pack in task and bible in task for task in request.slot_session_tasks.values())
            and all("## BIBLE.md (constitution" in _user_text(messages[0]["content"]) for messages in request.slot_messages.values()))


def test_an_external_body_copy_stays_the_plan_subject_after_a_candidate_is_bound(env):
    sub = env.install()
    ctx = env.ctx("t-copy", workspace_root=env.clone, workspace_mode="external")
    clone = env.clone.resolve()
    _write(ctx, affected_paths=["ouroboros/mod_a.py"], ask=["s1", "sess1"], attach=[BRIEF])
    first = _revisions(env, "t-copy")[0]
    assert first["body"] is True and f"({clone})" in first["body_note"]
    assert sub.calls[0]["request"].session_root == str(clone) and "native_candidate_root" not in sub.calls[0]["request"].policy

    candidate, receipt = env.prepare(ctx)
    assert f"still use {env.clone}; plan_task reviews {clone} " in receipt, receipt  # default reviews keep the workspace

    _call(ctx, plan="Same files, revised prose.")
    _call(ctx, affected_paths=[str(clone / "ouroboros" / "mod_a.py")])
    _text, result = _call(ctx, ask=["s1", "sess1"], attach=[BRIEF])
    second, third = _revisions(env, "t-copy")[1:]
    for revision in (second, third):  # relative and absolute spellings of the clone's file agree
        assert revision["body"] is True and revision["body_note"] == first["body_note"], revision["body_note"]
        assert str(candidate) not in revision["body_note"]
    ask = _asks(env, "t-copy")[1]
    assert [a["locator"] for a in ask["attached"]] == [BRIEF] and ask["omissions"] == [] and result.status == "ok"
    request = sub.calls[1]["request"]
    assert request.session_root == str(clone) and f"you review {clone} " in request.slot_session_tasks["sess1"]
    assert request.policy["native_candidate_root"] == str(candidate)  # native seats read it beside the clone
    assert f"integrator brief kept in {env.clone.name}" in _user_text(request.slot_messages["s1"][-1]["content"])
    assert _governed(env, request)


def test_an_unavailable_unused_candidate_does_not_refuse_the_tasks_workspace(env):
    """The plan never reads the unused candidate, so its vanished checkout refuses nothing there; the
    commit gate, which commits that candidate, keeps refusing it (``review_repo_dirs_for``)."""
    import shutil

    from ouroboros.review_substrate import review_repo_dirs_for

    sub = env.install()
    ctx = env.ctx("t-vanished", workspace_root=env.clone, workspace_mode="external")
    candidate, _receipt = env.prepare(ctx)
    shutil.rmtree(candidate)  # deleted outside the product while the task runs (its GC keeps a live owner's)
    text, result = _write(ctx, affected_paths=["ouroboros/mod_a.py"], ask=["sess1"], attach=[BRIEF])
    assert result.status == "ok", text
    [revision] = _revisions(env, "t-vanished")
    assert revision["body"] is True and f"({env.clone.resolve()})" in revision["body_note"]
    assert [a["locator"] for a in _asks(env, "t-vanished")[0]["attached"]] == [BRIEF]
    assert sub.calls[0]["request"].session_root == str(env.clone.resolve()) and _governed(env, sub.calls[0]["request"])
    with pytest.raises(ValueError, match="unavailable governance/subject root"):
        review_repo_dirs_for(ctx)


def test_a_foreign_project_keeps_its_root_and_an_absolute_candidate_path_is_body_work(env):
    sub = env.install()
    ctx = env.ctx("t-foreign", workspace_root=env.foreign, workspace_mode="external")
    foreign = env.foreign.resolve()
    candidate, _receipt = env.prepare(ctx)

    _write(ctx, affected_paths=["src/app.py"])
    _call(ctx, affected_paths=["src/app.py", str(candidate / "ouroboros" / "mod_a.py")], ask=["s1", "sess1"], attach=[BRIEF])
    own, mixed = _revisions(env, "t-foreign")
    assert own["body"] is False and own["body_note"].startswith("not constitutional"), own["body_note"]
    assert mixed["body"] is True and f"({candidate})" in mixed["body_note"]
    ask = _asks(env, "t-foreign")[0]
    assert [a["locator"] for a in ask["attached"]] == [BRIEF] and ask["omissions"] == []
    request = sub.calls[0]["request"]
    assert request.session_root == str(foreign) and f"you review {foreign} " in request.slot_session_tasks["sess1"]
    assert "integrator brief kept in foreign" in _user_text(request.slot_messages["s1"][-1]["content"])
    assert _governed(env, request)  # a body plan carries the pack wherever its session reads


def _native_seat(env):
    """The pool is one native API seat: a configured-subagent row that retrieves through inspection rounds."""
    from dataclasses import replace

    [slot] = _slots(("n1", "m/native"))
    seat = replace(slot, subagent_id="native-reader", native_retrieval_override=True)
    env.monkeypatch.setattr(plan_ask, "plan_review_slots", lambda default_effort="": [seat])


def _native_episode(request, slot, calls, *, call_id):
    """The REAL native executor over ``request``: one scripted round per registered inspection call,
    then the reply. Returns the tool replies in call order and the episode's inspection context."""
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from tests.test_native_tool_round_executor import _ScriptedLLM, _tool_call

    llm = _ScriptedLLM([*({"tool_calls": [_tool_call(name, args, f"c{i}")]} for i, (name, args) in enumerate(calls)),
                        {"content": CLEAN}])
    executor = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id=call_id), llm=llm)
    result = executor.execute()
    assert result.raw_text == CLEAN and len(llm.calls) == len(calls) + 1
    replies = {m["tool_call_id"]: m["content"] for m in llm.calls[-1]["messages"] if m.get("role") == "tool"}
    return [replies[f"c{i}"] for i in range(len(calls))], executor._inspection_ctx


def test_a_native_seat_reads_the_bound_candidate_beside_a_foreign_project(env):
    """The reader itself, in Pro: after ``prepare_self_change`` a mixed plan keeps the foreign
    project as its subject, and the native seat's registered tools read the bound candidate's files by
    their absolute paths beside it, while relative paths stay in the project and governance stays the
    serving checkout. The candidate address is frozen in the ask's request, so a recorded copy of that
    request reads it after the author's binding is gone; another task's candidate stays unreadable and
    nothing is written. The worktree, data and serving folders sit under the user-files home as in
    ``~/Ouroboros``, where ``user_files`` refuses every copy."""
    import dataclasses
    import json

    from ouroboros.review_substrate import ReviewRequest
    from ouroboros.tools import review_helpers

    env.monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(env.tmp))
    sub = env.install()
    ctx = env.ctx("t-native", workspace_root=env.foreign, workspace_mode="external")
    candidate, _receipt = env.prepare(ctx)
    marker = candidate / "ouroboros" / "mod_a.py"
    marker.write_text("GEN = 'CANDIDATE_ONLY'\n", encoding="utf-8")
    other, _receipt = env.prepare(env.ctx("t-other"))  # a different task's private candidate
    (other / "ouroboros" / "mod_a.py").write_text("GEN = 'NOT_THIS_PLAN'\n", encoding="utf-8")
    env.monkeypatch.setattr(review_helpers, "REPO_ROOT", env.serving)  # the installed body the reviewer's governance reads
    _native_seat(env)

    text, result = _write(ctx, affected_paths=["src/app.py", str(marker)], ask=["n1"])
    assert result.status == "ok", text
    [call] = sub.calls
    request, slot = call["request"], call["slots"][0]
    assert request.session_root == str(env.foreign.resolve())
    # The recorded copy custody rebuilds (``ReviewRequest(**stored)``), after the author's binding changed.
    ctx.task_metadata.pop("body_candidate")
    ctx.repo_dir = ctx.system_repo_dir = env.serving
    recorded = ReviewRequest(**json.loads(json.dumps(dataclasses.asdict(request), default=str)))
    replies, reader = _native_episode(recorded, slot, [
        ("read_file", {"path": str(marker)}),  # the read that was refused
        ("list_files", {"path": str(candidate / "ouroboros")}),
        ("search_code", {"query": "CANDIDATE_ONLY", "path": str(candidate)}),
        ("read_file", {"path": "src/app.py"}),
        ("read_file", {"path": str(env.serving / "BIBLE.md")}),
        ("read_file", {"root": "system_repo", "path": "ouroboros/mod_a.py"}),
        ("read_file", {"path": str(other / "ouroboros" / "mod_a.py")}),
        ("write_file", {"path": str(marker), "content": "GEN = 'WRITTEN'\n"}),
    ], call_id="native-candidate-read")
    candidate_read, listing, search, project, bible, serving, unrelated, write = replies
    assert "CANDIDATE_ONLY" in candidate_read and "BLOCKED" not in candidate_read, candidate_read
    assert "mod_a.py" in listing and "CANDIDATE_ONLY" in search, (listing, search)
    assert "APP = 1" in project and "# Constitution" in bible, (project, bible)
    assert "GEN_OLD" in serving and "CANDIDATE_ONLY" not in serving, serving
    assert "USER_FILES_PATH_BLOCKED" in unrelated and "NOT_THIS_PLAN" not in unrelated, unrelated
    assert marker.read_text(encoding="utf-8") == "GEN = 'CANDIDATE_ONLY'\n" and "WRITTEN" not in write, write
    # The reader's own roots are unchanged: the project is its subject, the serving checkout its governance;
    # the candidate is only the address the ask froze.
    assert pathlib.Path(reader.repo_dir) == env.foreign.resolve() and pathlib.Path(reader.system_repo_dir) == env.serving
    assert recorded.policy["native_candidate_root"] == request.policy["native_candidate_root"] == str(candidate)


def test_a_rooms_selected_folder_stays_the_subject_and_an_unavailable_one_is_refused(env):
    sub = env.install()
    room = env.ctx("t-room", is_direct_chat=True, meta={"_project_room_dir": str(env.clone)})
    env.prepare(room)
    _write(room, affected_paths=["ouroboros/mod_a.py"], ask=["sess1"], attach=[BRIEF])
    [revision] = _revisions(env, "t-room")
    assert revision["body"] is True and f"({env.clone.resolve()})" in revision["body_note"]
    assert [a["locator"] for a in _asks(env, "t-room")[0]["attached"]] == [BRIEF]
    assert sub.calls[0]["request"].session_root == str(env.clone.resolve()) and _governed(env, sub.calls[0]["request"])

    missing = env.ctx("t-gone", is_direct_chat=True, meta={"_project_room_dir": str(env.tmp / "moved-away")})
    _candidate, receipt = env.prepare(missing)
    assert f"still use {env.tmp / 'moved-away'}; plan_task refuses (unavailable governance/subject root" in receipt, receipt
    noted = env.ctx("t-noted", is_direct_chat=True, meta={"_project_room_note": "The room's folder is unavailable: ROOM_NOTE_FIXTURE."})
    candidate, receipt = env.prepare(noted)  # the real tool: preparation succeeded, and the receipt says so
    assert "is unavailable (The room's folder is unavailable: ROOM_NOTE_FIXTURE.)" in receipt, receipt
    assert f"workspace_root={candidate} now use it" in receipt and "still use" not in receipt and "plan_task" not in receipt
    from ouroboros.review_substrate import review_repo_dirs_for
    from ouroboros.tools.review import _gate_governance_root
    from ouroboros.tools.tool_resolution import active_repo_dir_for

    with pytest.raises(ValueError, match="ROOM_NOTE_FIXTURE"):  # no folder replaces the room after preparation
        active_repo_dir_for(noted)

    for context, why in ((missing, "moved-away"), (noted, "ROOM_NOTE_FIXTURE")):
        text, result = _write(context, affected_paths=["ouroboros/mod_a.py"], ask=["s1"])
        assert "PLAN_SUBJECT_ROOT_INVALID" in text and why in text and result.code == "TOOL_ERROR", text
        assert plan_record.load_plan_record(env.data, context.task_id) == plan_record.empty_record()
        # Only the plan refuses: the commit gate's governance never depended on the task's folder.
        assert review_repo_dirs_for(context)[0] == _gate_governance_root(context) == env.serving.resolve()
    assert len(sub.calls) == 1  # nobody was called for a refused subject


def test_a_workspace_without_its_mode_stays_refused_when_a_candidate_is_bound(env):
    """The binding's own validation runs before the folder rule: an ambiguous workspace is refused,
    never resolved into the candidate."""
    sub = env.install()
    ambiguous = env.ctx("t-ambiguous", workspace_root=env.foreign, workspace_mode="")
    body_candidate.prepare(ambiguous)
    text, result = _write(ambiguous, affected_paths=["src/app.py"], ask=["s1"])
    assert "PLAN_SUBJECT_ROOT_INVALID" in text and "workspace_root is set without workspace_mode" in text, text
    assert plan_record.load_plan_record(env.data, "t-ambiguous") == plan_record.empty_record() and sub.calls == []


def test_without_its_own_folder_a_task_keeps_the_bound_candidate_as_subject(env):
    from ouroboros.tools.tool_resolution import active_repo_dir_for

    sub = env.install()
    folderless = env.ctx("t-folderless", project_id="proj-1")
    scratch = active_repo_dir_for(folderless).resolve()
    candidate, receipt = env.prepare(folderless)
    assert active_repo_dir_for(folderless).resolve() == scratch != candidate  # the default folder is the task scratch
    assert (f"review_change without root, commands without cwd and delegate_start runs still use {scratch}; "
            f"plan_task reviews {candidate} " in receipt), receipt  # the receipt tells the two review defaults apart
    plain = env.ctx("t-plain")
    plain_candidate, _receipt = env.prepare(plain)
    for context, bound in ((folderless, candidate), (plain, plain_candidate)):
        _write(context, affected_paths=["ouroboros/mod_a.py"], ask=["sess1"], attach=[BRIEF])
        [revision] = _revisions(env, context.task_id)
        assert revision["body"] is True and f"({bound})" in revision["body_note"], revision["body_note"]
        assert sub.calls[-1]["request"].session_root == str(bound) and _governed(env, sub.calls[-1]["request"])
        assert "native_candidate_root" not in sub.calls[-1]["request"].policy  # the candidate is the subject itself
        assert _asks(env, context.task_id)[0]["omissions"] == [{"locator": BRIEF, "reason": "missing"}]


def test_evidence_an_ask_could_not_attach_is_named_in_its_result_and_in_the_free_read(env):
    env.install()
    ctx = env.ctx("t-omit", workspace_root=env.foreign, workspace_mode="external")
    (env.foreign / "notes").mkdir()
    (env.foreign / "notes" / "long.md").write_text("x" * 50_000, encoding="utf-8")
    text, _result = _write(ctx, affected_paths=["src/app.py"], ask=["s1"], attach=["notes/absent.md", "notes/long.md"])
    recorded = _asks(env, "t-omit")[0]["omissions"]
    assert recorded == [{"locator": "notes/absent.md", "reason": "missing"}, {"locator": "notes/long.md", "reason": "truncated_to_40000"}]
    read, _result = _call(ctx)
    line = "\n  evidence omitted: notes/absent.md (missing); notes/long.md (truncated_to_40000)\n"
    for shown in (text, read):
        assert line in shown and shown.index("- ask a1 on revision 1") < shown.index(line), shown
        assert "cannot be attached directly" not in shown  # what to attach instead is said only for a deny root


def test_a_late_reply_settles_from_its_recorded_assignment_after_the_candidate_is_bound(env):
    """The paid send's recorded assignment, never the context's current roots: a free read after
    ``prepare_self_change`` collects the pending seat by its row and retry key, re-sends nothing
    and leaves the ask's delivery and attachments as they were sent."""
    pending = {"status": "pending", "raw_text": "", "operation_state": "pending_dispatch", "late_result_pending": True}
    sub = env.install({"sess1": pending})
    ctx = env.ctx("t-late", workspace_root=env.clone, workspace_mode="external")
    _write(ctx, affected_paths=["ouroboros/mod_a.py"], ask=["sess1"], attach=[BRIEF])
    sent = _asks(env, "t-late")[0]
    env.prepare(ctx)
    collected = []

    def collector(root, *, task_id, surface, row, retry_key, usage_ctx=None, **_kw):
        collected.append((pathlib.Path(root), task_id, surface, row["seat_id"], retry_key))
        return {"slot_id": "sess1", "model": "cursor=grok", "status": "ok", "raw_text": _reply("Late, read from the clone."),
                "error": "", "operation_id": row.get("operation_id"), "operation_state": "late_settled", "usage": {"cost": 0.02}}

    env.monkeypatch.setattr("ouroboros.review_operation.collect_recorded_review_actor", collector)
    _call(ctx)
    assert collected == [(env.data, "t-late", "plan_review", "sess1", sent["retry_key"])] and len(sub.calls) == 1
    settled = _asks(env, "t-late")[0]
    assert settled["participants"][0]["status"] == "answered"
    assert settled["participants"][0]["delivery_ref"] == sent["participants"][0]["delivery_ref"]
    assert settled["attached"] == sent["attached"]
    from ouroboros.review_ledger import read_source

    delivered = read_source(env.data, "t-late", sent["participants"][0]["delivery_ref"]).decode("utf-8")
    assert f"you review {env.clone.resolve()} " in delivered
