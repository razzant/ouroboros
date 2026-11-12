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

import json
import os
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import body_candidate, plan_record
from ouroboros.tools import plan_ask
from ouroboros.tools import review as _review  # noqa: F401 — binds its REPO_ROOT aliases before a test patches them
from tests.body_candidate_support import git, isolate, make_serving
from tests.test_plan_task_tool import CLEAN, _call, _reply, _slots, _Substrate, _user_text, _write

WHY = "the session reads the copy itself"  # a first body ask that names fewer than all ticked reviewers says why

BRIEF = ".local/research/brief.md"
MANAGED_URL = "https://example.invalid/razzant/ouroboros-managed.git"


@pytest.fixture
def env(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    (serving / "docs").mkdir()
    (serving / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros — Architecture\n\n## 1. Runtime\n\nthe loop.\n", encoding="utf-8")
    (serving / "docs" / "CHECKLISTS.md").write_text("## Plan Review Checklist\n\nFIXTURE CHECKLIST ROW.\n", encoding="utf-8")
    git(serving, "add", "-A")
    git(serving, "commit", "-q", "-m", "governance")
    git(serving, "remote", "add", "managed", MANAGED_URL)  # a source install's maintained update remote
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
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


def test_a_nested_named_body_clone_is_the_plan_subject(env):
    """A workspace may be a containing folder with a separately cloned body inside it."""
    import shutil

    sub = env.install()
    container = env.tmp / "workspace-container"
    nested = container / "nested-body"
    container.mkdir()
    shutil.copytree(env.clone, nested)
    for task_id, locator in (
        ("t-nested-absolute", str(nested / "ouroboros" / "mod_a.py")),
        ("t-nested-relative", "nested-body/ouroboros/mod_a.py"),
    ):
        ctx = env.ctx(task_id, workspace_root=container, workspace_mode="external")
        _write(ctx, affected_paths=[locator], ask=["s1", "sess1"])
        [revision] = _revisions(env, task_id)
        assert revision["body"] is True
        assert str(nested.resolve()) in revision["body_note"]
        request = sub.calls[-1]["request"]
        assert request.session_root == str(container.resolve())
        assert "## BIBLE.md (constitution" in _user_text(request.slot_messages["s1"][0]["content"])
        assert "## Governance pack (this plan touches Ouroboros's own body)" in request.slot_session_tasks["sess1"]


def test_nested_probe_reuse_is_local_and_does_not_scan_past_a_body_match(env):
    from ouroboros import review_body_fact

    container = env.tmp / "container"
    target = container / "source"
    target.mkdir(parents=True)
    ctx = env.ctx("t-probe-reuse", workspace_root=container, workspace_mode="external")
    probes = []
    real_git = review_body_fact._git

    def observed_git(root, *args):
        if args == ("rev-parse", "--show-toplevel"):
            probes.append(root)
        return real_git(root, *args)

    env.monkeypatch.setattr(review_body_fact, "_git", observed_git)
    paths = [str(target / f"new-{i}" / "module.py") for i in range(500)]
    assert plan_ask.plan_is_body(ctx, paths, system_root=env.serving, active_root=container)[0] is False
    assert probes == [target.resolve()]

    # The same parent becomes a recognized body repository between calls.
    # A persistent negative cache would now incorrectly keep the plan core-only.
    git(target, "init", "-q")
    git(target, "remote", "add", "origin", str(env.serving))
    probes.clear()
    # Unrelated later folders should not be probed after the first body match.
    paths.extend(str(container / f"other-{i}" / "module.py") for i in range(500))
    assert plan_ask.plan_is_body(ctx, paths, system_root=env.serving, active_root=container)[0] is True
    assert probes == [target.resolve()]


def test_a_nested_foreign_clone_stays_foreign(env):
    """Nested discovery uses the repository's remotes, not its folder name."""
    import shutil

    sub = env.install()
    container = env.tmp / "foreign-container"
    nested = container / "nested-body-like-name"
    container.mkdir()
    shutil.copytree(env.foreign, nested)
    ctx = env.ctx("t-nested-foreign", workspace_root=container, workspace_mode="external")

    _write(ctx, affected_paths=[str(nested / "src" / "app.py")], ask=["s1", "sess1"])
    [revision] = _revisions(env, "t-nested-foreign")
    assert revision["body"] is False
    request = sub.calls[0]["request"]
    assert request.session_root == str(container.resolve())
    assert "## BIBLE.md (constitution" not in _user_text(request.slot_messages["s1"][0]["content"])
    assert "## Governance pack (this plan touches Ouroboros's own body)" not in request.slot_session_tasks["sess1"]


def test_an_unavailable_unused_candidate_does_not_refuse_the_tasks_workspace(env):
    """The plan never reads the unused candidate, so its vanished checkout refuses nothing there; the
    commit gate, which commits that candidate, keeps refusing it (``review_repo_dirs_for``)."""
    import shutil

    from ouroboros.review_substrate import review_repo_dirs_for

    sub = env.install()
    ctx = env.ctx("t-vanished", workspace_root=env.clone, workspace_mode="external")
    candidate, _receipt = env.prepare(ctx)
    shutil.rmtree(candidate)  # deleted outside the product while the task runs (its GC keeps a live owner's)
    text, result = _write(ctx, affected_paths=["ouroboros/mod_a.py"], ask=["sess1"], attach=[BRIEF], reason=WHY)
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
    _write(room, affected_paths=["ouroboros/mod_a.py"], ask=["sess1"], attach=[BRIEF], reason=WHY)
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
        text, result = _write(context, affected_paths=["ouroboros/mod_a.py"], ask=["s1"], reason=WHY)
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
        _write(context, affected_paths=["ouroboros/mod_a.py"], ask=["sess1"], attach=[BRIEF], reason=WHY)
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
    # The head attached in part; only the absent file is not attached, each with its reason.
    line = ("\n  evidence attached (prepared for the participants): notes/long.md (in part; source 50000 bytes; truncated_to_40000)"
            "\n  evidence not attached: notes/absent.md (missing)\n")
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
    _write(ctx, affected_paths=["ouroboros/mod_a.py"], ask=["sess1"], attach=[BRIEF], reason=WHY)
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


# ------------------------------------------------- a second body copy outside the folder and candidate (#2918)


def _delivered(request):
    """Every text an ask handed its seats: each packet message and each session task."""
    return [*(_user_text(m["content"]) for messages in request.slot_messages.values() for m in messages),
            *request.slot_session_tasks.values()]


def _ungoverned(request):
    return not any("## BIBLE.md (constitution" in text or "## Governance pack" in text for text in _delivered(request))


def _sibling(env, kind):
    """A second copy of the body outside the task's folder and its candidate (#2918), recognized only by its own Git
    identity: a linked worktree of the serving repository, a clone with a local origin, a linked worktree of a clone
    whose one remote is the serving repository's ``managed`` URL, or a clone of that URL with a partial-clone filter.
    Its own BIBLE and ARCHITECTURE carry markers no reviewer may be given: the pack is always the serving checkout's."""
    if kind == "worktree":
        root = env.tmp / "sibling-worktree"
        git(env.serving, "worktree", "add", "-q", "--detach", str(root))
    elif kind == "clone":
        root = env.clone
    elif kind == "partial":  # #3004: the filter makes its remote read `<url> (fetch) [blob:none]` in `git remote -v`
        root = env.tmp / "partial-clone"
        git(env.tmp, "clone", "-q", str(env.serving), str(root))
        git(root, "remote", "set-url", "origin", MANAGED_URL)
        git(root, "config", "remote.origin.partialclonefilter", "blob:none")
    else:
        mirror = env.tmp / "managed-mirror"
        git(env.tmp, "clone", "-q", str(env.serving), str(mirror))
        git(mirror, "remote", "set-url", "origin", MANAGED_URL)
        root = env.tmp / "managed-worktree"
        git(mirror, "worktree", "add", "-q", "--detach", str(root))
    (root / "BIBLE.md").write_text("# SIBLING_OWN_BIBLE\n", encoding="utf-8")
    (root / "docs" / "ARCHITECTURE.md").write_text("# SIBLING_OWN_ARCHITECTURE\n", encoding="utf-8")
    return root.resolve()


def _copy_note(root):
    return f"body: the declared paths change a recognized copy of Ouroboros's own body ({root})"


def _serving_pack(env, request):
    """The serving checkout's BIBLE and ARCHITECTURE reached both seats: their exact text in the packet seat's standing
    frame, both locators as mandatory reads in the session seat's task (``_governed`` alone checks the headers)."""
    from ouroboros.tools.review_helpers import load_governance_doc

    frame, task, docs = _user_text(request.slot_messages["s1"][0]["content"]), request.slot_session_tasks["sess1"], (
        "BIBLE.md", "docs/ARCHITECTURE.md")
    return (_governed(env, request) and all(load_governance_doc(env.serving, rel) in frame for rel in docs)
            and all(f"`{env.serving.resolve() / rel}`" in task for rel in docs))


def _sent(request):
    """Every byte an ask handed the substrate, as one text."""
    import dataclasses

    return json.dumps(dataclasses.asdict(request), default=str, ensure_ascii=False)


def _policy_values(request):
    """The request's policy; only the mandatory-read size follows each plan's own text."""
    policy = dict(request.policy)
    assert isinstance(policy.pop("native_mandatory_read_chars"), int)
    return policy


@pytest.mark.parametrize("kind, how", [("worktree", "git_common_dir"), ("clone", "remote_chain"), ("managed", "managed_remote"),
                                       ("partial", "managed_remote")])
def test_a_folderless_plan_in_a_second_body_copy_is_a_body_plan_and_moves_nothing(env, kind, how):
    """A folderless task binds candidate A, then declares only absolute paths inside another copy B of the body. B's
    own Git identity makes it body work, and nothing else moves: A stays the subject, the task's folder and read roots
    stay, the request's policy is that of a plan inside A itself (no address for B), and the pack is the serving
    checkout's, never B's own."""
    from ouroboros import review_body_fact
    from ouroboros.tool_access import profile_readable_root_paths
    from ouroboros.tools.tool_resolution import active_repo_dir_for

    sub = env.install()
    sibling = _sibling(env, kind)
    assert review_body_fact.body_fact(sibling, system_repo=env.serving, data_dir=env.data).how == how
    control = env.ctx("t-in-candidate", project_id="proj-1")
    env.prepare(control)
    _write(control, affected_paths=["ouroboros/mod_a.py"], ask=["s1", "sess1"], attach=[BRIEF])
    ctx = env.ctx("t-sibling", project_id="proj-1")
    scratch = active_repo_dir_for(ctx).resolve()
    candidate, _receipt = env.prepare(ctx)
    reads = profile_readable_root_paths(ctx, operation="read")
    text, result = _write(ctx, affected_paths=[str(sibling / "ouroboros" / "mod_a.py")], ask=["s1", "sess1"], attach=[BRIEF])
    assert result.status == "ok", text
    [revision] = _revisions(env, "t-sibling")
    assert revision["body"] is True and revision["body_note"] == _copy_note(sibling), revision["body_note"]
    plain, request = (call["request"] for call in sub.calls)
    assert request.session_root == str(candidate) and _serving_pack(env, request)
    assert "SIBLING_OWN" not in _sent(request) and str(sibling / "BIBLE.md") not in _sent(request)
    # Nothing moves: the task's own folder, its read roots, what an attachment resolves against, the request's policy.
    assert active_repo_dir_for(ctx).resolve() == scratch != candidate
    assert profile_readable_root_paths(ctx, operation="read") == reads
    assert not any(root == sibling or sibling in root.parents for _label, root in reads)
    missing = [{"locator": BRIEF, "reason": "missing"}]
    assert _asks(env, "t-sibling")[0]["omissions"] == _asks(env, "t-in-candidate")[0]["omissions"] == missing
    assert _policy_values(request) == _policy_values(plain) and "native_candidate_root" not in request.policy


def test_every_spelling_of_a_path_in_a_second_body_copy_reaches_its_own_repository(env):
    """A file not created yet, ``file://``, a line selector, a relative path leaving the planning root and a list
    that also names a foreign project all reach the second copy's own repository, which the note names."""
    sibling = _sibling(env, "worktree")
    ctx = env.ctx("t-spellings", project_id="proj-1")
    candidate, _receipt = env.prepare(ctx)
    target = sibling / "ouroboros" / "mod_a.py"
    for n, paths in enumerate((
        [str(sibling / "ouroboros" / "brand_new" / "module.py")],
        [f"file://{target}"],
        [f"{target}::lines=1-1"],
        [os.path.relpath(target, candidate)],  # relative paths resolve against the planning root, here the candidate
        [str(env.foreign / "src" / "app.py"), str(target)],
    ), start=1):
        text, result = (_write if n == 1 else _call)(ctx, affected_paths=paths)
        assert result.status == "ok", text
        revision = _revisions(env, "t-spellings")[-1]
        assert (revision["n"], revision["body"], revision["body_note"]) == (n, True, _copy_note(sibling)), paths


def test_foreign_and_unknown_repositories_beside_the_candidate_stay_core(env):
    """Git identity decides, never names: a repository whose files carry the body's names and its linked worktree, a
    folder in no repository, a ``file://`` path to nothing and no paths at all stay core, and so does a core change
    that only attaches a file of a recognized second copy (evidence never decides). No pack reaches any seat."""
    sub = env.install()
    lookalike = env.tmp / "lookalike"
    for relative in ("BIBLE.md", "docs/ARCHITECTURE.md", "ouroboros/__init__.py", "ouroboros/mod_a.py"):
        (lookalike / relative).parent.mkdir(parents=True, exist_ok=True)
        (lookalike / relative).write_text(f"lookalike {relative}\n", encoding="utf-8")
    git(lookalike, "init", "-q")
    git(lookalike, "remote", "add", "origin", "https://example.invalid/acme/app.git")
    git(lookalike, "add", "-A")
    git(lookalike, "commit", "-q", "-m", "lookalike")
    linked = env.tmp / "lookalike-linked"
    git(lookalike, "worktree", "add", "-q", "--detach", str(linked))
    loose = env.tmp / "loose"
    loose.mkdir()
    sibling = _sibling(env, "worktree")
    ctx = env.ctx("t-core", project_id="proj-1")
    env.prepare(ctx)
    for n, paths in enumerate((
        [str(lookalike / "ouroboros" / "mod_a.py"), str(lookalike / "BIBLE.md"), str(linked / "ouroboros" / "mod_a.py")],
        [str(loose / "notes.md")],
        ["file:///nonexistent-path"],
        [],
    ), start=1):
        text, result = (_write if n == 1 else _call)(ctx, affected_paths=paths, **({"ask": ["s1", "sess1"]} if n == 1 else {}))
        assert result.status == "ok", text
        revision = _revisions(env, "t-core")[-1]
        assert (revision["n"], revision["body"]) == (n, False), (paths, revision["body_note"])
    text, result = _call(ctx, affected_paths=[str(loose / "notes.md")], ask=["s1", "sess1"],
                         attach=[str(sibling / "ouroboros" / "mod_a.py")])
    assert result.status == "ok", text
    revision = _revisions(env, "t-core")[-1]
    assert (revision["n"], revision["body"]) == (5, False), revision["body_note"]
    assert len(sub.calls) == 2 and all(_ungoverned(call["request"]) for call in sub.calls)


def test_a_foreign_repository_nested_in_a_body_copy_keeps_its_classification(env):
    """Only targets outside the recognized copies gain anything: a foreign repository nested in the task's own body
    clone stays body work by that clone's containment, as before, and one nested in a second copy is judged by its own
    identity, which is foreign; the step never climbs from a path's own repository to a containing one."""
    sibling = _sibling(env, "worktree")
    for root in (env.clone, sibling):
        nested = root / "vendor" / "app"
        (nested / "src").mkdir(parents=True)
        (nested / "src" / "app.py").write_text("APP = 1\n", encoding="utf-8")
        git(nested, "init", "-q")
        git(nested, "remote", "add", "origin", "https://example.invalid/acme/app.git")
    ctx = env.ctx("t-nested-in-copy", workspace_root=env.clone, workspace_mode="external")
    _write(ctx, affected_paths=["vendor/app/src/app.py"])
    _call(ctx, affected_paths=[str(sibling / "vendor" / "app" / "src" / "app.py")])
    inside, beside = _revisions(env, "t-nested-in-copy")
    assert inside["body"] is True and inside["body_note"] == _copy_note(env.clone.resolve()), inside["body_note"]
    assert beside["body"] is False, beside["body_note"]


def test_a_repository_is_judged_once_and_probes_are_reused_within_one_call_only(env):
    """Probes are keyed by the nearest existing folder and reused within one call; a repository is judged once however
    many folders its paths name. 500 new files under a recognized copy cost one probe and nothing declared after it is
    probed or judged; a folder that becomes a copy between calls is recognized by the next call."""
    from ouroboros import review_body_fact

    sibling = _sibling(env, "worktree")
    folder = env.tmp / "task-folder"
    folder.mkdir()
    ctx = env.ctx("t-sibling-probes", workspace_root=folder, workspace_mode="external")
    probes, judged = [], []
    real_git, real_fact = review_body_fact._git, review_body_fact.body_fact

    def observed_git(root, *args):
        if args == ("rev-parse", "--show-toplevel"):
            probes.append(root)
        return real_git(root, *args)

    def observed_fact(root, **kwargs):
        judged.append(pathlib.Path(root))
        return real_fact(root, **kwargs)

    env.monkeypatch.setattr(review_body_fact, "_git", observed_git)
    env.monkeypatch.setattr(review_body_fact, "body_fact", observed_fact)

    def classify(paths):
        probes.clear()
        judged.clear()
        return plan_ask.plan_is_body(ctx, paths, system_root=env.serving, active_root=folder)

    parts = [env.foreign / f"part-{i}" for i in range(5)]
    for part in parts:
        part.mkdir()
    assert classify([str(parts[i % 5] / f"new-{i}" / "module.py") for i in range(500)])[0] is False
    assert sorted(probes) == sorted(part.resolve() for part in parts) and judged == [env.foreign.resolve()]
    fresh = env.tmp / "becomes-a-copy"
    fresh.mkdir()
    later = [str(fresh / f"new-{i}" / "module.py") for i in range(500)]
    assert classify(later)[0] is False and probes == [fresh.resolve()] and judged == []
    git(fresh, "init", "-q")
    git(fresh, "remote", "add", "origin", str(env.serving))
    assert classify(later) == (True, _copy_note(fresh.resolve())) and probes == [fresh.resolve()]
    paths = [str(sibling / f"new-{i}" / "module.py") for i in range(500)]
    assert classify(paths + later) == (True, _copy_note(sibling)) and probes == [sibling] and judged == [sibling]


def test_a_task_folder_stays_the_subject_when_its_plan_reaches_a_second_body_copy(env):
    """A task with its own foreign folder W and a bound candidate A: the revision that reaches copy B is a body plan,
    W stays the subject, native seats keep reading A at the address the request freezes, and the packet seat that
    answered the core revision continues its own transcript under the current frame, which now carries the pack."""
    sub = env.install()
    sibling = _sibling(env, "worktree")
    ctx = env.ctx("t-folder-sibling", workspace_root=env.foreign, workspace_mode="external")
    candidate, _receipt = env.prepare(ctx)
    _write(ctx, affected_paths=["src/app.py"], ask=["s1", "sess1"])
    _call(ctx, affected_paths=["src/app.py", str(sibling / "ouroboros" / "mod_a.py")], ask=["s1", "sess1"])
    target = sibling / "ouroboros" / "mod_b.py"
    from_folder, from_candidate = os.path.relpath(target, env.foreign.resolve()), os.path.relpath(target, candidate)
    assert from_folder != from_candidate
    _call(ctx, affected_paths=[from_folder])
    _call(ctx, affected_paths=[from_candidate])
    core, body, relative, elsewhere = _revisions(env, "t-folder-sibling")
    assert core["body"] is False and body["body"] is True and body["body_note"] == _copy_note(sibling), body["body_note"]
    assert relative["body"] is True and relative["body_note"] == _copy_note(sibling), relative["body_note"]
    assert elsewhere["body"] is False, elsewhere["body_note"]  # relative paths resolve against W, never the candidate
    first, second = (call["request"] for call in sub.calls)
    assert _ungoverned(first) and _serving_pack(env, second) and "SIBLING_OWN" not in _sent(second)
    assert [p["continued"] for p in _asks(env, "t-folder-sibling")[1]["participants"] if p["seat_id"] == "s1"] == [True]
    assert [m["role"] for m in second.slot_messages["s1"]] == ["system", "user", "assistant", "user"]
    for request in (first, second):
        assert request.session_root == str(env.foreign.resolve()) and request.policy["native_candidate_root"] == str(candidate)
    assert _policy_values(first) == _policy_values(second)


def test_an_exempt_payload_path_in_a_second_copy_does_not_hide_an_ordinary_one(env):
    """Exemption is per locator, judgement per repository: a skill-payload path in a second copy stays data and leaves
    that copy unjudged, so an ordinary path in the same copy declared after it still makes the plan body work."""
    holder = env.data / "skills"  # a linked worktree of the body that holds the drive's skill payloads
    git(env.serving, "worktree", "add", "-q", "--detach", str(holder))
    folder = env.tmp / "task-folder"
    folder.mkdir()
    ctx = env.ctx("t-payload-sibling", workspace_root=folder, workspace_mode="external")
    payload, ordinary = str(holder / "external" / "toy-skill" / "plugin.py"), str(holder / "ouroboros" / "mod_a.py")
    assert plan_ask.plan_is_body(ctx, [payload], system_root=env.serving, active_root=folder)[0] is False
    assert plan_ask.plan_is_body(ctx, [payload, ordinary], system_root=env.serving, active_root=folder) == (
        True, _copy_note(holder.resolve()))
    env.monkeypatch.setattr(plan_ask, "plan_payload_roots", lambda _ctx, _paths: [])  # only the exemption kept it data
    assert plan_ask.plan_is_body(ctx, [payload], system_root=env.serving, active_root=folder)[0] is True


def test_a_name_the_filesystem_refuses_stays_core_instead_of_failing_the_write(env):
    """A sentence pasted as a path can be a name the filesystem refuses (over 255 bytes): it lies in no repository, so
    the revision is recorded as core in either spelling instead of failing (relative ones failed before #2918 too)."""
    folder = env.tmp / "task-folder"
    folder.mkdir()
    ctx = env.ctx("t-long-name", workspace_root=folder, workspace_mode="external")
    name = "Обновить_README_" * 30
    for n, locator in enumerate((f"{name}/a.py", str(folder / name / "a.py")), start=1):
        text, result = (_write if n == 1 else _call)(ctx, affected_paths=[locator])
        assert result.status == "ok", text
        assert [r["body"] for r in _revisions(env, "t-long-name")] == [False] * n
