"""A filled optional argument that asks for nothing is not a refusal.

Models may fill every key of a tool schema. These tests pin, through the REAL registry,
that such a value takes the omitted path with one disclosure line, and that a
genuine argument mistake is refused once, typed, naming the value it received.
"""
import queue

import pytest

from ouroboros.owner_wait import direct_owner_wait
from ouroboros.task_results import load_task_result
from ouroboros.tools.arg_feedback import argument_refusal, ignored_argument_note
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _registry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="root-task",
                      is_direct_chat=True, current_chat_id=1, event_queue=queue.Queue())
    ctx.owner_wait_callback = direct_owner_wait
    registry = ToolRegistry(repo_dir=repo, drive_root=tmp_path)
    registry.set_context(ctx)
    return registry, ctx


def test_helpers_name_the_field_the_value_and_the_repair(tmp_path):
    assert ignored_argument_note("timezone", "UTC", "run_at carries its own offset") == (
        "timezone='UTC' ignored: run_at carries its own offset")
    _registry_unused, ctx = _registry(tmp_path)
    text = argument_refusal(ctx, "DEMO_INVALID", ["a=1 must be 2.", "b is missing"],
                            effect="Nothing was recorded.")
    assert text == "⚠️ DEMO_INVALID: a=1 must be 2; b is missing. Nothing was recorded."


@pytest.mark.parametrize("reflex", [0, 1])
def test_escalate_ignores_a_wait_bound_on_a_quiz_that_does_not_wait(tmp_path, reflex):
    """The live loop: `max_wait_minutes` 0 or 1 beside `wait_for_answer=false` and a real
    assumption was refused eight times in one day. The quiz is sent; the receipt says
    the bound was ignored; the ignored bound is never persisted with the quiz."""
    registry, ctx = _registry(tmp_path)
    result = registry.execute_result("escalate", {
        "question": "Which theme?", "options": ["Light", "Dark"], "stake": "",
        "assumption": "Light meanwhile", "wait_for_answer": False, "max_wait_minutes": reflex,
    })
    assert result.status == "ok" and result.text.startswith("OK: quiz ")
    assert "max_wait_minutes ignored: it applies only to wait_for_answer=true" in result.text
    quiz_id = ctx.event_queue.get_nowait()["quiz_id"]
    block = load_task_result(tmp_path, "root-task")["owner_quiz"][quiz_id]
    assert "max_wait_minutes" not in block


def test_escalate_refuses_a_bound_that_cannot_hold_and_names_the_repair(tmp_path, monkeypatch):
    """A REQUIRED wait keeps the refusal: `0` and a bound past the task's own ceiling both
    ask for something the wait cannot serve, so each is one typed refusal that names the
    repair. The tolerant path is the OPTIONAL question above, not a silent reinterpretation."""
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "21600")  # 360 minutes
    registry, ctx = _registry(tmp_path)
    for asks_for_nothing in (0, 100000):
        refused = registry.execute_result("escalate", {
            "question": "Continue?", "options": ["Yes", "No"],
            "wait_for_answer": True, "max_wait_minutes": asks_for_nothing})
        assert refused.status == "error" and refused.code == "TOOL_ARG_ERROR"
        assert refused.text.startswith("⚠️ QUIZ_WAIT_BOUND_INVALID")
        assert "omit it for an unbounded wait" in refused.text  # the repair, not just the rule
        assert refused.text.endswith("The quiz was not sent.")
    assert not getattr(ctx, "_owner_wait_max_minutes", None) and ctx.event_queue.empty()


@pytest.mark.parametrize("ceiling,budget,limit", [
    ("21600", None, 360), ("21600", 0, 360), ("21600", 50000, 360), ("21600", 1800, 30), (None, 1800, 30)])
def test_escalate_caps_a_required_wait_by_the_task_execution_limit(tmp_path, monkeypatch, ceiling, budget, limit):
    """#939: an owner wait counts as execution, so an explicit task budget narrower than the
    global ceiling ends the task first. Absent or zero adds no bound: the ceiling decides."""
    if ceiling is None:
        monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)
    else:
        monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", ceiling)
    registry, ctx = _registry(tmp_path)
    ctx.task_metadata = {} if budget is None else {"execution_timeout_sec": budget}
    ask = {"question": "Continue?", "options": ["Yes", "No"], "wait_for_answer": True}
    refused = registry.execute_result("escalate", {**ask, "max_wait_minutes": limit + 1})
    assert refused.code == "TOOL_ARG_ERROR" and f"at most {limit} (the task's execution time limit)" in refused.text
    assert ctx.event_queue.empty()
    accepted = registry.execute_result("escalate", {**ask, "max_wait_minutes": limit})
    assert accepted.status == "ok" and f"waits up to {limit} minutes" in accepted.text


@pytest.mark.parametrize("budget", [0.5, 30, 59.999])
def test_escalate_subminute_budget_refuses_a_whole_minute_but_keeps_unbounded_wait(tmp_path, monkeypatch, budget):
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "unlimited")
    registry, ctx = _registry(tmp_path)
    ctx.task_metadata = {"execution_timeout_sec": budget}
    ask = {"question": "Continue?", "options": ["Yes", "No"], "wait_for_answer": True}
    refused = registry.execute_result("escalate", {**ask, "max_wait_minutes": 1})
    assert refused.code == "TOOL_ARG_ERROR" and "shorter than one minute" in refused.text
    assert ctx.event_queue.empty()
    assert registry.execute_result("escalate", ask).status == "ok"


def test_escalate_genuine_argument_mistake_is_one_typed_refusal(tmp_path):
    registry, ctx = _registry(tmp_path)
    result = registry.execute_result("escalate", {
        "question": "Continue?", "options": ["Yes", "No"], "wait_for_answer": True, "max_wait_minutes": -3})
    assert result.status == "error" and result.code == "TOOL_ARG_ERROR"
    assert result.text.startswith("⚠️ QUIZ_WAIT_BOUND_INVALID: max_wait_minutes must be ")
    assert result.text.endswith("The quiz was not sent.") and ctx.event_queue.empty()


def test_a_blocked_link_stays_a_policy_denial_and_a_malformed_one_an_argument_fault(tmp_path):
    registry, _ctx = _registry(tmp_path)
    blocked = registry.execute_result("send_links", {"links": [{"label": "x", "url": "javascript:alert(1)"}]})
    assert blocked.status == "blocked" and blocked.text.startswith("⚠️ SEND_LINKS_URL_BLOCKED")
    assert blocked.text.endswith("No links were sent.")
    malformed = registry.execute_result("send_links", {"links": "nope"})
    assert malformed.status == "error" and malformed.text.startswith("⚠️ SEND_LINKS_")



def test_a_refusal_names_a_repair_call_only_where_the_caller_may_make_it(tmp_path):
    """Both directions, one helper for every refusal that names a working call: an ordinary root may name a
    core tool; a contract's disabled_tools, a subagent profile without it, a Presence ceiling without it and a
    Presence grant with host-bound arguments each make the host unable to show the named call works."""
    from ouroboros.presence_authority import (
        PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload,
    )
    from ouroboros.presence_capabilities import PresenceArgumentBinding
    from ouroboros.tools.arg_feedback import repair_call_available

    def ceiling(*grants):
        return {"capability_ceiling": presence_ceiling_payload(PresenceCapabilityCeiling(
            skill_name="helper", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
            state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=10,
            tool_grants=grants, resource_grants=(), digest="e" * 64))}

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root-task")
    assert repair_call_available(ctx, "get_task_result")
    ctx.task_metadata = {"disabled_tools": ["get_task_result"]}
    assert not repair_call_available(ctx, "get_task_result")
    ctx.task_metadata = {"tool_profile": "local_readonly_subagent"}
    assert repair_call_available(ctx, "get_task_result") and not repair_call_available(ctx, "continue_task")
    ctx.task_metadata = {}
    ctx.task_contract = ceiling(PresenceToolGrant("get_task_result"))
    assert repair_call_available(ctx, "get_task_result") and not repair_call_available(ctx, "wait_tasks")
    bound = PresenceArgumentBinding(("mode",), "static", static_value="warm")
    ctx.task_contract = ceiling(PresenceToolGrant("await_messages", (bound,)))
    assert not repair_call_available(ctx, "await_messages")


def _file_registry(tmp_path, monkeypatch):
    """A real registry over an external project workspace (no body candidate in between)."""
    import ouroboros.safety as safety
    from tests._workspace_git import git_repository

    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    system, project = git_repository(tmp_path / "system"), git_repository(tmp_path / "project")
    registry = ToolRegistry(repo_dir=system, drive_root=tmp_path / "drive")
    registry.set_context(ToolContext(repo_dir=system, system_repo_dir=system, drive_root=tmp_path / "drive",
                                     workspace_root=project, workspace_mode="external", task_id="root-task"))
    return registry, project


def test_edit_text_count_one_is_the_single_replacement_and_a_larger_count_names_the_edits_form(tmp_path, monkeypatch):
    """`count` is outside edit_text's schema. count=1 (and a null) ask for its default and run, named in the
    result; any other count is refused with the same tool's counted `edits` form, keeping root and the other
    arguments, so the ready call edits the same file; nothing is written by the refused call."""
    registry, repo = _file_registry(tmp_path, monkeypatch)
    (repo / "m.txt").write_text("a a a\n", encoding="utf-8")
    plain = registry.execute_result("edit_text", {"path": "m.txt", "old_str": "a a a", "new_str": "b a a"})
    assert plain.status == "ok" and "count=" not in plain.text
    one = registry.execute_result("edit_text", {"path": "m.txt", "old_str": "b a a", "new_str": "c a a", "count": 1})
    assert one.status == "ok" and "count=1 ignored: one replacement is edit_text's default" in one.text
    null = registry.execute_result("edit_text", {"path": "m.txt", "old_str": "c a a", "new_str": "d a a", "count": None})
    assert null.status == "ok" and "count=None ignored: empty optional value" in null.text
    assert (repo / "m.txt").read_text(encoding="utf-8") == "d a a\n"
    assert registry.execute_result("write_file", {"root": "task_drive", "path": "m.txt", "content": "x x\n"}).status == "ok"
    for count in (2, 0, "all"):
        refused = registry.execute_result("edit_text", {
            "root": "task_drive", "path": "m.txt", "old_str": "x", "new_str": "y", "count": count})
        assert refused.status == "error" and refused.code == "TOOL_ARG_ERROR", refused.text
        assert ('edit_text(path="m.txt", root="task_drive", edits=[{"old_str": <this call\'s old_str>, '
                '"new_str": <this call\'s new_str>, "count": ' + ("2" if count == 2 else "<occurrences>") + "}])"
                ) in refused.text
        assert refused.text.endswith("Nothing was written.")
    ready = registry.execute_result("edit_text", {
        "path": "m.txt", "root": "task_drive", "edits": [{"old_str": "x", "new_str": "y", "count": 2}]})
    assert ready.status == "ok", ready.text
    assert registry.execute_result("read_file", {"root": "task_drive", "path": "m.txt"}).text.endswith("y y\n")
    assert (repo / "m.txt").read_text(encoding="utf-8") == "d a a\n"


def test_read_file_end_line_reads_its_inclusive_window_and_an_unclear_end_names_the_call(tmp_path, monkeypatch):
    """`end_line` is outside read_file's schema and has one reading, the inclusive last line counted from the
    start the reader uses (start_line floored at 1), so it reads that window and says so, also for a memory
    file the default read shows whole. A conflicting max_lines, an end before the start, a value that is not a
    whole line number and `end_char` (window or file?) are refused with the ready call keeping the read's root
    and cursor; the ordinary call is unchanged and a refused read leaves no earlier read's view behind."""
    registry, repo = _file_registry(tmp_path, monkeypatch)
    (repo / "f.txt").write_text("".join(f"line {n}\n" for n in range(1, 11)), encoding="utf-8")
    plain = registry.execute_result("read_file", {"path": "f.txt", "start_line": 3, "max_lines": 3})
    assert "lines 3\u20135 of 10" in plain.text and "end_line" not in plain.text
    read = registry.execute_result("read_file", {"path": "f.txt", "start_line": 3, "end_line": 5})
    assert read.status == "ok" and "lines 3\u20135 of 10" in read.text and "line 6" not in read.text
    assert read.text.endswith("end_line=5 read as max_lines=3 from start_line=3")
    for args, lines in (({"end_line": 2, "max_lines": 2}, "1\u20132"), ({"start_line": 0, "end_line": 2}, "1\u20132"),
                        ({"start_line": -5, "end_line": 3}, "1\u20133"), ({"end_line": "4", "max_lines": 2000}, "1\u20134")):
        assert f"lines {lines} of 10" in registry.execute_result("read_file", {"path": "f.txt", **args}).text, args
    assert registry._ctx.last_read_view is not None
    for args, says in (({"start_line": 3, "end_line": 5, "max_lines": 9}, "asks for 3 lines from start_line=3"),
                       ({"start_line": 6, "end_line": 5}, "end_line=5 is before start_line=6"),
                       ({"end_line": True}, "end_line=True is not a whole line number"),
                       ({"end_line": 2.9}, "end_line=2.9 is not a whole line number"),
                       ({"end_line": "five"}, "end_line='five' is not a whole line number"),
                       ({"end_line": "--2", "max_chars": 1}, "end_line='--2' is not a whole line number"),
                       ({"end_line": "\u00b2"}, "end_line='\u00b2' is not a whole line number"),
                       ({"end_line": 2001, "max_lines": 2000}, "asks for 2001 lines from start_line=1"),
                       ({"max_lines": 1, "end_char": 40}, "end_char=40 could count"),
                       ({"start_line": 2, "start_char": 4, "end_char": 40},
                        "end_char=40 could count from the window or from the file")):
        refused = registry.execute_result("read_file", {"path": "f.txt", "root": "active_workspace", **args})
        assert refused.status == "error" and refused.code == "TOOL_ARG_ERROR", refused.text
        assert says in refused.text and 'read_file(path="f.txt", root="active_workspace"' in refused.text
        assert refused.text.endswith("Nothing was read.") and "line 1\n" not in refused.text
        assert registry._ctx.last_read_view is None
        other = "max_lines" if "end_char" in args else "max_chars"  # the call's other limit stays in the named call
        assert other not in args or f"{other}={args[other]}, start_line=" in refused.text, refused.text
    assert 'start_char=4, start_line=2, max_chars=<characters>)' in refused.text
    (repo / "g.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    for window in ({"end_line": 1}, {"max_lines": 0}):  # one line, by end_line or by max_lines brought to 1
        refused = registry.execute_result("read_file", {"path": "g.txt", "end_char": 40, **window})
        assert 'read_file(path="g.txt", max_lines=1, start_line=1, max_chars=<characters>)' in refused.text, refused.text
        retry = registry.execute_result("read_file", {"path": "g.txt", "max_lines": 1, "start_line": 1, "max_chars": 40})
        assert "alpha" in retry.text and "beta" not in retry.text
    world = tmp_path / "drive" / "memory" / "WORLD.md"
    world.parent.mkdir(parents=True, exist_ok=True)
    world.write_text("".join(f"w {n}\n" for n in range(1, 2002)), encoding="utf-8")
    for default in ({}, {"max_lines": 2000}, {"start_line": 1, "max_lines": 2000}):  # asks for nothing: whole
        whole = registry.execute_result("read_file", {"root": "runtime_data", "path": "memory/WORLD.md", **default})
        assert "w 2001" in whole.text, default
    bounded = registry.execute_result("read_file", {"root": "runtime_data", "path": "memory/WORLD.md", "end_line": 2000})
    assert "w 2000\n" in bounded.text and "w 2001" not in bounded.text, bounded.text[-300:]
