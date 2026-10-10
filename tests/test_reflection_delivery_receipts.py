"""Transport receipts the host already recorded reach every NEW reflection.

The real chain: ``PresenceDeliveryRecorder.record`` appends the rows, the real
``_run_reflection`` walks the lineage once and reads one captured chat-chain
window, the real ``generate_reflection`` builds and retains the prompt. Only the
model boundary (``llm_observability.chat_observed``) is a fake, capturing the
prompt it would send (and, where named, asking for the retained record through the
reflection's own ``read_file``). What these tests prove is that the evidence reaches
the prompt and its retained sources, not what a model then writes.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
from types import SimpleNamespace

import pytest

import ouroboros.presence_delivery as delivery
from ouroboros import post_task_synthesis
from ouroboros.task_results import write_task_result
from ouroboros.utils import utc_now_iso
from tests.test_consolidator_context_fit import fit  # noqa: F401 — offline Light route/window facts
from tests.test_late_phase_pause_resume import ANSWER, ROOT, _join, _phase, _resume, _spawned, late  # noqa: F401

HEADING = "## Transport delivery reports (read for this reflection)"
_NOW = dt.datetime.now(dt.timezone.utc)
OLD = (_NOW - dt.timedelta(hours=3)).isoformat()      # rows written before the task
START = (_NOW - dt.timedelta(hours=1)).isoformat()    # the task's recorded start


@pytest.fixture
def root(tmp_path, monkeypatch):
    import supervisor.message_bus as bus
    import supervisor.state as state

    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    for name, value in {"DRIVE_ROOT": tmp_path, "STATE_PATH": tmp_path / "state/state.json",
                        "STATE_LAST_GOOD_PATH": tmp_path / "state/state.last_good.json",
                        "STATE_LOCK_PATH": tmp_path / "locks/state.lock"}.items():
        monkeypatch.setattr(state, name, value)
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture
def prompts(monkeypatch):
    from ouroboros import llm_observability

    captured: list[str] = []

    def chat_observed(*_args, **kwargs):
        if kwargs.get("call_type") == "task_reflection":
            captured.append(kwargs["messages"][0]["content"])
        return {"content": "Lesson."}, {}

    monkeypatch.setattr(llm_observability, "chat_observed", chat_observed)
    return captured


def _payload(**changes):
    return {"schema_version": 1, "delivery_id": "send:one", "part_id": "0", "state": "delivered",
            "provider": "slack", "account_id": "T-work", "conversation_id": "C-daily", "thread_id": "",
            "text": "Daily report", "format": "markdown", "message": {"provider_message_id": "1700.1"},
            "origin": {"kind": "tool"}, **changes}


def _record(root, skill, task_id="", **changes):
    origin = {"kind": changes.pop("kind", "tool"), **({"task_id": task_id} if task_id else {})}
    return delivery.PresenceDeliveryRecorder(root).record(skill, _payload(origin=origin, **changes))


def _presence_turn(root, task_id, skill, **fields):
    write_task_result(root, task_id, "completed", result="reply", started_at=START, metadata={"presence": {
        "transport_skill": skill, "binding_id": "b" * 32, "event": {"source_event_id": f"in-{task_id}"}}}, **fields)


def _filler(root, text, ts=OLD):
    from supervisor.message_bus import log_chat

    log_chat("in", 1, 0, text, ts=ts, drive_root=root, require_write=True)


def _archive(root, name):
    """The rotator's rename of the live generation, at a chosen archive stamp."""
    (root / "archive").mkdir(exist_ok=True)
    (root / "logs" / "chat.jsonl").rename(root / "archive" / name)


def _reflect(root, task, prompts, *, env_root=None, sealed_final=None):
    """The real reflection stage of one root; returns (its receipts section, entry)."""
    env = SimpleNamespace(drive_root=env_root or root, repo_dir=root)
    entry = post_task_synthesis._run_reflection(
        env, None, {"type": "task", "text": "Send the daily report", **task}, {"rounds": 20, "cost": 0.01},
        {"tool_calls": []}, {}, sealed_final=sealed_final)
    assert entry is not None and prompts, "a 20-round run reflects"
    prompt = prompts[-1]
    section = prompt[prompt.index(HEADING):].split("\n\n", 1)[0]
    # The retained exact input is the record of what was shown.
    ref = entry["source_ref"]
    retained = (pathlib.Path(ref["canonical_root"]) / ref["read"]["arguments"]["path"]).read_text(encoding="utf-8")
    assert section in retained
    return section, entry


def _receipts(section):
    return [line for line in section.splitlines() if line.startswith("- ")]


def _bound(line):
    """``(binding, task id)`` of one rendered receipt line."""
    return re.match(r"- #\d+ \S+ (host_bound|skill_claimed) task (\S+?): ", line).groups()


def _delivery(line):
    return line.split(" delivery ", 1)[1].split(" ", 1)[0]


def _record_arguments(section):
    """The ``read_file`` arguments the section names for its retained whole selection."""
    return json.loads(section.split("\nComplete record (", 1)[1].split("): read_file ", 1)[1].splitlines()[0])


def _read_record(root, task_id, arguments):
    """Through the reflection's own reader (``KnowledgeReadContext``), never a bare path."""
    from ouroboros.consolidator import KnowledgeReadContext
    from ouroboros.tools.registry import ToolContext

    reader = KnowledgeReadContext(ToolContext(repo_dir=root, drive_root=root, task_id=task_id), "task_reflection")
    return reader.read_call({"id": "read-record", "function": {"name": "read_file",
                                                               "arguments": json.dumps(arguments)}})["result"]


def test_presence_slack_two_parts_cross_conversation_and_frozen_observations(root, prompts):
    from ouroboros.task_finalization import build_completion_observations, build_sealed_final_package

    _presence_turn(root, "turn-1", "slack-bridge")
    _record(root, "slack-bridge", "turn-1", text="Part one")
    _record(root, "slack-bridge", "turn-1", part_id="1", text="Part two", message={"provider_message_id": "1700.2"})
    _record(root, "slack-bridge", "turn-1", delivery_id="send:elsewhere", conversation_id="C-other", text="FYI")
    row = {"task_id": "turn-1", "status": "completed", "result": "reply", "metadata": {"presence": {
        "transport_skill": "slack-bridge"}, "presence_result_text": "reply", "presence_outcome": "message"}}
    row["completion_observations"] = build_completion_observations(root, {"id": "turn-1"}, {"tool_calls": []})
    sealed = build_sealed_final_package(row, "reply")

    section, _entry = _reflect(root, {"id": "turn-1", "budget_drive_root": str(root)}, prompts, sealed_final=sealed)

    lines = _receipts(section)
    assert [_bound(line) for line in lines] == [("host_bound", "turn-1")] * 3
    assert "part 0 delivered" in lines[0] and '"Part one"' in lines[0] and "part 1 delivered" in lines[1]
    assert "slack:T-work:C-daily:0" in lines[0] and "slack:T-work:C-other:0" in lines[2]
    assert all(" row:" in line for line in lines), "each fact names its exact chat row"
    assert "Reports: 3 (delivered 3, host_bound 3); all shown in chain order" in section
    assert "coverage complete" in section and "NONEXHAUSTIVE" not in section
    assert "never checked with the provider by the host" in section and "not that\na person read it" in section
    # The frozen package keeps its meaning beside the new section.
    prompt = prompts[-1]
    assert "it is not a provider delivery receipt" in prompt
    assert '"delivery_receipt_coverage": "not_observed_by_tool_trace"' in prompt
    assert sealed["completion_observations"]["delivery_receipt_coverage"] == "not_observed_by_tool_trace"


def test_telegram_status_part_email_accepted_and_skill_claimed_tasks(root, prompts):
    # An ordinary task and a consciousness wake are known only through the skill's claim.
    write_task_result(root, "main-1", "completed", result="sent", started_at=START)
    _record(root, "telegram-bot", "main-1", provider="telegram", account_id="bot-1", conversation_id="-100",
            delivery_id="telegram:bot-1:-100:out-7", text="Hello")
    _record(root, "telegram-bot", "main-1", provider="telegram", account_id="bot-1", conversation_id="-100",
            delivery_id="telegram:bot-1:-100:out-7", part_id="status", state="uncertain", text="Second part unconfirmed")
    _record(root, "email-presence", "main-1", provider="email", account_id="ops@example.test",
            conversation_id="reader@example.test", delivery_id="mail:1", state="accepted",
            message={"message_id": "<1@example.test>", "accepted": ["reader@example.test"], "refused": []})

    section, _ = _reflect(root, {"id": "main-1"}, prompts)
    lines = _receipts(section)
    assert [_bound(line) for line in lines] == [("skill_claimed", "main-1")] * 3
    assert "delivery telegram:bot-1:-100:out-7 part 0 delivered" in lines[0]
    assert "part status uncertain" in lines[1]
    assert "part 0 accepted" in lines[2] and "delivered" not in lines[2]
    assert "SMTP acceptance, not inbox\narrival" in section
    assert "Reports: 3 (accepted 1, delivered 1, skill_claimed 3, uncertain 1)" in section

    prompts.clear()
    write_task_result(root, "wake-1", "completed", result="noted", started_at=START,
                      metadata={"initiator": "consciousness"})
    _record(root, "slack-bridge", "wake-1", delivery_id="send:wake", kind="tool")
    section, _ = _reflect(root, {"id": "wake-1"}, prompts)
    [line] = _receipts(section)
    assert _bound(line) == ("skill_claimed", "wake-1") and _delivery(line) == "send:wake"


def test_exact_binding_excludes_foreign_rows_and_keeps_every_reported_state(root, prompts):
    _presence_turn(root, "turn-a", "slack-bridge")
    _presence_turn(root, "turn-foreign", "slack-bridge")
    _record(root, "slack-bridge", "turn-a", delivery_id="send:dup", account_id="T-one", state="uncertain")
    _record(root, "slack-bridge", "turn-a", delivery_id="send:dup", account_id="T-one", state="delivered")
    _record(root, "slack-bridge", "turn-a", delivery_id="send:dup", account_id="T-two")
    _record(root, "slack-bridge", "turn-foreign", delivery_id="send:foreign", text="not ours")
    _record(root, "slack-bridge", "someone-else", delivery_id="send:claimed-elsewhere", text="not ours either")
    # A row bound by the host to another task is never re-read through its origin claim.
    path = root / "logs" / "chat.jsonl"
    foreign = json.loads(path.read_text(encoding="utf-8").splitlines()[3])
    assert foreign["task_id"] == "turn-foreign"
    foreign["transport"]["origin"]["task_id"] = "turn-a"
    foreign["transport"]["delivery"]["delivery_id"] = "send:mismatch"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(foreign) + "\n")

    section, _ = _reflect(root, {"id": "turn-a"}, prompts)
    lines = _receipts(section)
    assert len(lines) == 3 and "not ours" not in section and "send:mismatch" not in section
    assert "T-one" in lines[0] and "uncertain" in lines[0]
    assert "T-one" in lines[1] and "part 0 delivered" in lines[1], "no latest-wins fold, no downgrade guard"
    assert "T-two" in lines[2]


def test_split_root_child_beyond_the_evidence_cap_and_retry_grandchild(root, prompts, tmp_path_factory):
    project_drive = tmp_path_factory.mktemp("project-drive")
    write_task_result(root, "split-root", "completed", result="done", started_at=START,
                      root_task_id="split-root", parent_task_id="", delegation_role="root")
    for index in range(6):  # verbose children overflow the 6000-char child evidence text
        write_task_result(root, f"kid-{index}", "completed", result="R" * 3000, trace_summary="T" * 2000,
                          parent_task_id="split-root", root_task_id="split-root", delegation_role="subagent")
    _record(root, "slack-bridge", "kid-5", delivery_id="send:kid")
    split = {"id": "split-root", "root_task_id": "split-root", "budget_drive_root": str(root)}
    section, _ = _reflect(root, split, prompts)
    assert "OMISSION NOTE" in prompts[-1], "the child evidence text really was capped"
    [line] = _receipts(section)
    assert _bound(line) == ("skill_claimed", "kid-5")

    # Executed on another drive, the child evidence walks that drive; the receipts read the canonical root.
    prompts.clear()
    section, _ = _reflect(root, split, prompts, env_root=project_drive)
    assert "kid-5" not in prompts[-1].split("## Related child/subtask evidence")[1].split("##")[0]
    [line] = _receipts(section)
    assert _bound(line) == ("skill_claimed", "kid-5")

    # A hard-timeout retry keeps the logical root; its grandchildren and the first attempt count.
    prompts.clear()
    write_task_result(root, "logical", "failed", result="timed out", started_at=START, root_task_id="logical",
                      parent_task_id="", delegation_role="root")
    write_task_result(root, "retry", "completed", result="done", root_task_id="logical", parent_task_id="",
                      delegation_role="root", original_task_id="logical", timeout_retry_from="logical",
                      started_at=(_NOW - dt.timedelta(minutes=5)).isoformat())
    write_task_result(root, "child", "completed", result="c", parent_task_id="retry", root_task_id="logical",
                      delegation_role="subagent")
    write_task_result(root, "grandchild", "completed", result="g", parent_task_id="child", root_task_id="logical",
                      delegation_role="subagent")
    _record(root, "slack-bridge", "grandchild", delivery_id="send:grand")
    _record(root, "slack-bridge", "logical", delivery_id="send:first-attempt")
    _record(root, "slack-bridge", "kid-5", delivery_id="send:other-tree")
    section, _ = _reflect(root, {"id": "retry", "root_task_id": "logical"}, prompts)
    assert [_bound(line)[1] for line in _receipts(section)] == ["grandchild", "logical"]
    assert f"earliest recorded start {START}" in section
    children = prompts[-1].split("## Related child/subtask evidence")[1].split("##")[0]
    assert '"task_id": "child"' in children
    assert '"task_id": "logical"' not in children and '"task_id": "grandchild"' not in children, \
        "the receipt lineage never feeds the child evidence"


def test_the_receipt_lineage_never_widens_reflection_admission(root, prompts, monkeypatch, tmp_path_factory):
    """Admission reads the unchanged child walk; a failure only the receipt lineage sees admits nothing."""
    calls: list = []
    real = delivery.task_delivery_receipts
    monkeypatch.setattr(delivery, "task_delivery_receipts", lambda *a, **k: calls.append(a) or real(*a, **k))
    short, trace = {"rounds": 2, "cost": 0.0}, {"tool_calls": []}
    env = SimpleNamespace(drive_root=root, repo_dir=root)
    write_task_result(root, "first", "failed", result="timed out", started_at=START, root_task_id="first",
                      parent_task_id="", delegation_role="root")
    write_task_result(root, "again", "completed", result="done", root_task_id="first", parent_task_id="",
                      delegation_role="root", original_task_id="first", timeout_retry_from="first")
    write_task_result(root, "helper", "completed", result="c", parent_task_id="again", root_task_id="first",
                      delegation_role="subagent")
    write_task_result(root, "nested", "failed", result="boom", parent_task_id="helper", root_task_id="first",
                      delegation_role="subagent")
    _record(root, "slack-bridge", "nested", delivery_id="send:nested")
    retry = {"id": "again", "root_task_id": "first", "type": "task", "text": "short"}
    assert post_task_synthesis._run_reflection(env, None, retry, short, trace, {}) is None
    # A split root's failed child is on the canonical root; admission walks the execution drive.
    write_task_result(root, "split-2", "completed", result="done", started_at=START, root_task_id="split-2")
    write_task_result(root, "split-kid", "failed", result="boom", parent_task_id="split-2", root_task_id="split-2",
                      delegation_role="subagent")
    split = {"id": "split-2", "root_task_id": "split-2", "budget_drive_root": str(root), "type": "task", "text": "s"}
    assert post_task_synthesis._run_reflection(SimpleNamespace(drive_root=tmp_path_factory.mktemp("exec"),
                                                               repo_dir=root), None, split, short, trace, {}) is None
    assert calls == [] and prompts == []
    # A failed direct child still admits, as before; only then is the lineage read, with the nested receipt.
    write_task_result(root, "direct", "failed", result="boom", parent_task_id="again", root_task_id="first",
                      delegation_role="subagent")
    assert post_task_synthesis._run_reflection(env, None, retry, short, trace, {}) is not None
    assert len(calls) == 1 and "delivery send:nested" in prompts[-1]


def test_projection_is_bounded_with_counted_omissions(root, prompts):
    write_task_result(root, "busy", "completed", result="sent", started_at=START, root_task_id="busy")
    for index in range(13):
        write_task_result(root, f"busy-kid-{index:02d}", "completed", result="ok", parent_task_id="busy",
                          root_task_id="busy", delegation_role="subagent")
    for part in range(43):
        _record(root, "slack-bridge", "busy", delivery_id="send:long", part_id=str(part), text="x" * 400)
    section, _ = _reflect(root, {"id": "busy", "root_task_id": "busy"}, prompts)
    lines = _receipts(section)
    assert ("Reports: 43 (delivered 43, skill_claimed 43); newest 40 shown in chain order (#4–#43), 3 older not "
            "shown (#1–#3: delivered 3, skill_claimed 3; each whole in the record)") in section
    assert len(lines) == 40 and lines[0].startswith("- #4 ") and " part 3 delivered" in lines[0]
    assert lines[-1].startswith("- #43 ") and " part 42 delivered" in lines[-1]
    assert all("…[+240 chars]" in line for line in lines), "each text preview names what it left out"
    assert "busy-kid-10 and 2 more (this task" in section
    record = _read_record(root, "busy", _record_arguments(section))
    assert '"n": 3' in record and '"part_id": "2"' in record and "x" * 400 in record
    assert '"busy-kid-12"' in record, "every task id, not only the twelve named inline"


def test_reflection_reads_the_whole_record_after_the_chat_rotates(root, monkeypatch, fit):  # noqa: F811
    """The model asks for the record in the reflection's own tool loop after the rows left the live chat."""
    from ouroboros import llm_observability
    from supervisor.state import rotate_chat_log_if_needed

    write_task_result(root, "many", "completed", result="sent", started_at=START)
    refusal = "Refused by the provider: " + "r " * 200 + "END-OF-REFUSAL"
    _record(root, "slack-bridge", "many", delivery_id="send:many", state="failed", text=refusal,
            message={"error": "channel_not_found"})
    for part in range(41):  # a later report of part 0 contradicts the failed one
        _record(root, "slack-bridge", "many", delivery_id="send:many", part_id=str(part), text=f"part {part}")
    requests: list = []

    def chat_observed(*_args, **kwargs):
        if kwargs.get("call_type") != "task_reflection":
            return {"content": "Lesson."}, {}
        requests.append(kwargs)
        if len(requests) > 1:
            return {"content": "Lesson."}, {}
        rotate_chat_log_if_needed(root, max_bytes=1)
        live = root / "logs" / "chat.jsonl"
        assert "END-OF-REFUSAL" not in (live.read_text(encoding="utf-8") if live.exists() else "")
        arguments = _record_arguments(kwargs["messages"][0]["content"])
        return {"content": "", "tool_calls": [{"id": "read-record", "type": "function", "function": {
            "name": "read_file", "arguments": json.dumps(arguments)}}]}, {}

    monkeypatch.setattr(llm_observability, "chat_observed", chat_observed)
    env = SimpleNamespace(drive_root=root, repo_dir=root)
    entry = post_task_synthesis._run_reflection(
        env, None, {"type": "task", "text": "Send the report", "id": "many"}, {"rounds": 20, "cost": 0.01},
        {"tool_calls": []}, {})
    assert entry is not None and entry["reflection"] == "Lesson." and len(requests) == 2
    tools = {tool["function"]["name"] for tool in requests[0]["tools"]}
    assert "read_file" in tools and "memory_read" not in tools, "the reader a row address would need is absent"
    prompt = requests[0]["messages"][0]["content"]
    assert "END-OF-REFUSAL" not in prompt and "channel_not_found" not in prompt
    assert ("Reports: 42 (delivered 41, failed 1, skill_claimed 42); newest 40 shown in chain order (#3–#42), "
            "2 older not shown (#1–#2: delivered 1, failed 1, skill_claimed 2; each whole in the record)") in prompt
    answered = json.dumps(requests[1]["messages"][1:], ensure_ascii=False)
    assert "END-OF-REFUSAL" in answered and "channel_not_found" in answered
    assert '\\"n\\": 1' in answered and '\\"read_coverage\\": \\"complete\\"' in answered


def test_every_inline_identity_value_is_bounded_and_whole_in_the_record(root, prompts):
    task_id = "task-" + "t" * 120  # the longest a task id may be is 128
    write_task_result(root, task_id, "completed", result="sent", started_at=START)
    long = {"provider": "prov-" + "p" * 150, "account_id": "acct-" + "a" * 150,
            "conversation_id": "conv-" + "c" * 150, "delivery_id": "send-" + "d" * 150, "part_id": "1" * 150}
    skill = "skill-" + "s" * 150
    _record(root, skill, task_id, text="first line\nsecond line", **long)
    # Host-stamped values are bounded the same way: a reported_at and row stamp of any length.
    path = root / "logs" / "chat.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    row["transport"]["delivery"].update(reported_at="2026-" + "9" * 150, delivery_id="send:stamped")
    row["ts"] = "2026-" + "8" * 150
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    section, _ = _reflect(root, {"id": task_id}, prompts)
    first, stamped = _receipts(section)
    # The record's own read_file path names its task directory (a task id is at most 128 chars).
    inline = "\n".join(line for line in section.splitlines() if not line.startswith("Complete record ("))

    def clipped(value):
        return f"{value[:96]}…[+{len(value) - 96} chars]"

    key = ":".join([long["provider"], long["account_id"], long["conversation_id"], "0"])  # conversation_key
    shown = (task_id, skill, long["provider"], long["account_id"], key, long["delivery_id"], long["part_id"])
    assert all(clipped(value) in first for value in shown)
    assert not any(value in inline for value in (*shown, long["conversation_id"]))
    assert f"tasks {clipped(task_id)} (this task" in section
    assert '"first line second line"' in first, "a value stays one line"
    assert clipped("2026-" + "9" * 150) in stamped and "8" * 97 not in stamped
    assert max(len(line) for line in (first, stamped)) < 1600
    record = _read_record(root, task_id, _record_arguments(section))
    for value in (task_id, skill, *long.values(), "2026-" + "9" * 150, "2026-" + "8" * 150):
        assert value in record
    assert "first line\\nsecond line" in record


def test_an_unretained_record_is_a_named_gap_and_keeps_every_shown_report(root, prompts, monkeypatch):
    import ouroboros.chat_chain as chat_chain

    real = chat_chain.retain_memory_source

    def retain(context, source_id, data, extension="md"):
        if source_id == "task_delivery_receipts":
            raise OSError("artifact store unwritable")
        return real(context, source_id, data, extension)

    monkeypatch.setattr(chat_chain, "retain_memory_source", retain)
    write_task_result(root, "nosrc", "completed", result="sent", started_at=START)
    _record(root, "slack-bridge", "nosrc", delivery_id="send:n", state="uncertain", text="maybe")
    for part in range(41):
        _record(root, "slack-bridge", "nosrc", delivery_id="send:n", part_id=str(part))
    section, _ = _reflect(root, {"id": "nosrc"}, prompts)
    assert len(_receipts(section)) == 40
    assert "coverage gapped (complete_record_unavailable)" in section and "coverage complete" not in section
    assert "Complete record unavailable (OSError): the values below are bounded previews" in section
    assert "read_file" not in section
    assert ("2 older not shown (#1–#2: delivered 1, skill_claimed 2, uncertain 1; only counted here)") in section

    prompts.clear()  # zero reports: the read stays disclosed as gapped, still never as non-delivery
    write_task_result(root, "nosrc-quiet", "completed", result="sent", started_at=START)
    section, _ = _reflect(root, {"id": "nosrc-quiet"}, prompts)
    assert "not evidence of non-delivery" in section and "coverage gapped (complete_record_unavailable)" in section


def test_a_failed_lineage_walk_is_named_and_keeps_the_task_s_own_receipts(root, prompts, monkeypatch):
    import ouroboros.task_results as task_results

    write_task_result(root, "walkless", "completed", result="sent", started_at=START)
    _record(root, "slack-bridge", "walkless", delivery_id="send:own")

    def unreadable(*_args, **_kwargs):
        raise OSError("results root unreadable")

    monkeypatch.setattr(task_results, "list_task_results", unreadable)
    section, _ = _reflect(root, {"id": "walkless", "started_at": START}, prompts)
    assert [_delivery(line) for line in _receipts(section)] == ["send:own"]
    assert "tasks walkless (the task-results walk failed: this task, its root and only the ids found" in section


def test_window_reads_whole_boundary_segment_same_second_archives_and_discloses_older(root, prompts, monkeypatch):
    import supervisor.message_bus as bus

    write_task_result(root, "task-w", "completed", result="sent", started_at=START)
    _filler(root, "even older", ts=(_NOW - dt.timedelta(hours=5)).isoformat())
    _record(root, "slack-bridge", "task-w", delivery_id="send:oldest")  # in a generation older than the window
    _archive(root, "chat_20261001T090000.jsonl")
    _filler(root, "before the start")
    _record(root, "slack-bridge", "task-w", delivery_id="send:same-second")
    _archive(root, "chat_20261001T100000.jsonl")
    _filler(root, "before the start, rotated in the same second")
    # A receipt whose own stamp precedes the start still belongs to the whole boundary segment.
    with monkeypatch.context() as stamp:
        stamp.setattr(bus, "utc_now_iso", lambda: OLD)
        _record(root, "slack-bridge", "task-w", delivery_id="send:reordered")
    _archive(root, "chat_20261001T100000_1.jsonl")
    _record(root, "slack-bridge", "task-w", delivery_id="send:live")

    section, _ = _reflect(root, {"id": "task-w"}, prompts)
    assert [_delivery(line) for line in _receipts(section)] == ["send:same-second", "send:reordered", "send:live"]
    assert "3 of 4 chat generations read newest first back to the one holding the earliest" in section
    assert "NONEXHAUSTIVE: 1 older archived generation(s) not read" in section
    assert "coverage complete" in section, "clean bytes are still not an exhaustive history"


def test_no_recorded_start_reads_a_limited_window_and_says_so(root, prompts):
    _record(root, "slack-bridge", "no-start", delivery_id="send:oldest")
    _archive(root, "chat_20261001T080000.jsonl")
    _record(root, "slack-bridge", "no-start", delivery_id="send:archived")
    _archive(root, "chat_20261001T090000.jsonl")
    _record(root, "slack-bridge", "no-start", delivery_id="send:live")
    section, _ = _reflect(root, {"id": "no-start"}, prompts)
    assert [_delivery(line) for line in _receipts(section)] == ["send:live"]
    assert ("1 of 3 chat generations read newest first back to the newest nonempty one: a limited window, "
            "no start recorded for these tasks") in section
    assert "NONEXHAUSTIVE: 2 older archived generation(s) not read" in section

    # Just after a rotation the live file is empty: the window reaches the newest archive with a row.
    prompts.clear()
    _archive(root, "chat_20261001T100000.jsonl")
    (root / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    section, _ = _reflect(root, {"id": "no-start"}, prompts)
    assert [_delivery(line) for line in _receipts(section)] == ["send:live"]
    assert "2 of 4 chat generations" in section and "NONEXHAUSTIVE: 2 older" in section


@pytest.mark.parametrize("contents", ['{"ts": "bad-date"}\n', '{broken\n', '{"unfinished":'])
def test_no_anchor_does_not_scan_old_history_for_parseable_timestamps(root, prompts, contents):
    _record(root, "slack-bridge", "no-start", delivery_id="send:old")
    _archive(root, "chat_20261001T080000.jsonl")
    (root / "logs" / "chat.jsonl").write_text(contents, encoding="utf-8")
    section, _ = _reflect(root, {"id": "no-start"}, prompts)
    assert not _receipts(section)
    assert "1 of 2 chat generations" in section
    assert "NONEXHAUSTIVE: 1 older archived generation(s) not read" in section
    assert "not evidence of non-delivery" in section


@pytest.mark.parametrize("stamped", ["own", "root"])
def test_a_persisted_own_or_root_start_anchors_an_input_task_without_one(root, prompts, stamped):
    task_id = "attempt-2"
    write_task_result(root, "attempt-1", "failed", result="timed out", root_task_id="attempt-1",
                      **({"started_at": START} if stamped == "root" else {}))
    write_task_result(root, task_id, "completed", result="sent", root_task_id="attempt-1",
                      **({"started_at": START} if stamped == "own" else {}))
    _filler(root, "long before", ts=(_NOW - dt.timedelta(hours=5)).isoformat())
    _record(root, "slack-bridge", task_id, delivery_id="send:too-old")
    _archive(root, "chat_20261001T080000.jsonl")
    _filler(root, "before the start")
    _record(root, "slack-bridge", task_id, delivery_id="send:boundary")
    _archive(root, "chat_20261001T090000.jsonl")
    _record(root, "slack-bridge", task_id, delivery_id="send:live")
    section, _ = _reflect(root, {"id": task_id, "root_task_id": "attempt-1"}, prompts)  # the input has no stamp
    assert [_delivery(line) for line in _receipts(section)] == ["send:boundary", "send:live"]
    assert f"2 of 3 chat generations read newest first back to the one holding the earliest recorded start {START}" \
        in section
    assert "NONEXHAUSTIVE: 1 older archived generation(s) not read" in section


def test_zero_reports_are_unobserved_never_undelivered(root, prompts):
    write_task_result(root, "quiet", "completed", result="sent", started_at=START)
    _record(root, "slack-bridge", "someone-else")
    section, _ = _reflect(root, {"id": "quiet"}, prompts)
    assert _receipts(section) == []
    assert "No transport delivery report bound to these tasks was observed in what was read" in section
    assert "not evidence of non-delivery" in section and "coverage complete" in section


def test_bytes_after_the_capture_are_not_claimed_across_append_and_rotation(root, prompts, monkeypatch):
    from supervisor.state import rotate_chat_log_if_needed

    write_task_result(root, "race", "completed", result="sent", started_at=START)
    _record(root, "slack-bridge", "race", delivery_id="send:before-capture")
    raced: list = []

    def captured_at():  # stamped right after the capture: a writer and the rotator race in here
        if not raced:
            raced.append(True)  # the recorder stamps its own row through this same name
            _record(root, "slack-bridge", "race", delivery_id="send:after-capture")
            rotate_chat_log_if_needed(root, max_bytes=1)
            assert list((root / "archive").glob("chat_*.jsonl")), "the captured live generation rotated"
        return utc_now_iso()

    monkeypatch.setattr(delivery, "utc_now_iso", captured_at)
    section, _ = _reflect(root, {"id": "race"}, prompts)
    assert raced and "send:before-capture" in section and "send:after-capture" not in section
    assert "1 of 1 chat generations" in section and "coverage complete" in section
    # The late row is canonical history: the next new reflection reads it.
    section, _ = _reflect(root, {"id": "race"}, prompts)
    assert "send:after-capture" in section


def _deny(monkeypatch, target, *, directory=False):
    """An OS read refusal on one path (portable: chmod is bypassed by root and Windows)."""
    import os

    if directory:
        real_scandir = os.scandir

        def scandir(path="."):
            if pathlib.Path(path) == target:
                raise PermissionError(f"denied {path}")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", scandir)
        return
    real_open = pathlib.Path.open

    def open_(self, *args, **kwargs):
        if self == target:
            raise PermissionError(f"denied {self}")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", open_)


@pytest.mark.parametrize("damage", ["bad_line", "torn_live", "unreadable_window", "unreadable_oldest",
                                    "unreadable_listing"])
def test_gaps_keep_every_positive_receipt_and_name_themselves(root, prompts, monkeypatch, damage):
    write_task_result(root, "gap", "completed", result="sent", started_at=START)  # every generation is newer
    _record(root, "slack-bridge", "gap", delivery_id="send:oldest-archive")
    _archive(root, "chat_20261001T080000.jsonl")
    _record(root, "slack-bridge", "gap", delivery_id="send:middle-archive")
    _archive(root, "chat_20261001T090000.jsonl")
    _record(root, "slack-bridge", "gap", delivery_id="send:live")
    live = root / "logs" / "chat.jsonl"
    expected = {"send:oldest-archive", "send:middle-archive", "send:live"}
    if damage == "bad_line":
        with (root / "archive" / "chat_20261001T090000.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{broken\n")
        gap = "malformed_jsonl"
    elif damage == "torn_live":
        with live.open("ab") as handle:
            handle.write(b'{"type": "presence_delivery", "task_id": "gap"')
        gap = "incomplete_live_line"
    elif damage == "unreadable_window":
        _deny(monkeypatch, root / "archive" / "chat_20261001T090000.jsonl")
        expected.discard("send:middle-archive")
        gap = "unreadable_source"
    elif damage == "unreadable_oldest":  # the strict capture itself fails on it
        _deny(monkeypatch, root / "archive" / "chat_20261001T080000.jsonl")
        expected.discard("send:oldest-archive")
        gap = "unreadable_source"
    else:
        _deny(monkeypatch, root / "archive", directory=True)
        expected = {"send:live"}
        gap = "unreadable_source"
    section, _ = _reflect(root, {"id": "gap"}, prompts)
    assert {_delivery(line) for line in _receipts(section)} == expected
    assert f"coverage gapped ({gap})" in section


def test_an_unexpected_failure_mid_read_keeps_the_receipts_already_found(root, prompts, monkeypatch):
    import ouroboros.chat_chain as chat_chain

    write_task_result(root, "odd", "completed", result="sent", started_at=START)
    _record(root, "slack-bridge", "odd", delivery_id="send:archived")
    _archive(root, "chat_20261001T090000.jsonl")
    _record(root, "slack-bridge", "odd", delivery_id="send:live")
    real = chat_chain.row_address

    def row_address(row, **kwargs):
        if row["transport"]["delivery"]["delivery_id"] == "send:archived":
            raise RuntimeError("unexpected projection failure")
        return real(row, **kwargs)

    monkeypatch.setattr(chat_chain, "row_address", row_address)
    section, _ = _reflect(root, {"id": "odd"}, prompts)
    assert [_delivery(line) for line in _receipts(section)] == ["send:live"]
    assert "coverage gapped (read_failed)" in section
    assert "NONEXHAUSTIVE: 1 older archived generation(s) not read" in section


def test_capture_failure_is_unavailable_never_undelivered(root, prompts, monkeypatch):
    import ouroboros.jsonl_tail as jsonl_tail

    def broken(*_args, **_kwargs):
        raise RuntimeError("chain capture failed")

    monkeypatch.setattr(jsonl_tail, "JsonlChainSnapshot", broken)
    section, _ = _reflect(root, {"id": "any"}, prompts)
    assert section.startswith(HEADING + "\nUnavailable: the chat history could not be captured (RuntimeError)")
    assert "delivery outcomes stay unknown, not failed" in section


def test_no_reflection_and_ordinary_finalization_never_read_the_chat_chain(root, prompts, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline

    captures: list = []
    real = delivery.task_delivery_receipts
    monkeypatch.setattr(delivery, "task_delivery_receipts", lambda *a, **k: captures.append(a) or real(*a, **k))
    _record(root, "slack-bridge", "clean", delivery_id="send:clean")
    env = SimpleNamespace(drive_root=root, repo_dir=root, drive_path=lambda rel: root / rel)
    task = {"id": "clean", "type": "task", "text": "short", "chat_id": 1}
    assert post_task_synthesis._run_reflection(env, None, task, {"rounds": 2, "cost": 0.0},
                                               {"tool_calls": []}, {}) is None
    pipeline._store_task_result(env, task, "Done.", {"rounds": 2, "cost": 0.0}, {"tool_calls": []})
    assert captures == [] and prompts == []


# --- a reflection created after a restart or a Resume reads its own horizon ---------------


def _real_reflection(f, monkeypatch):
    calls: list = []
    real = delivery.task_delivery_receipts
    monkeypatch.setattr(f.pipeline, "_run_reflection", post_task_synthesis._run_reflection)
    monkeypatch.setattr(delivery, "task_delivery_receipts", lambda *a, **k: calls.append(a) or real(*a, **k))
    return calls


def _reflection_prompts(f):
    return [send["prompt"] for send in f.light.sends if HEADING in send["prompt"]]


def _pause_on(f, kind, trace):
    """The owner's Pause lands while the first ``kind`` send of a 20-round root's late phase is on the wire."""
    import threading

    from ouroboros.usage_accounting import UsageScope, usage_scope
    from supervisor.owner_pause_control import request_owner_pause

    entered, release = threading.Event(), threading.Event()

    def hook():
        entered.set()
        assert release.wait(10)

    f.light.hooks[kind] = [hook]
    scope = UsageScope(drive_root=f.root, task_id=ROOT, root_task_id=ROOT, category="task", source="agent.task",
                       root_limit_usd=3.0, root_limit_source="task_admission")
    with usage_scope(scope):
        threads = _spawned(lambda: f.pipeline._run_post_task_processing_async(
            f.env, f.task, {"rounds": 20}, trace, {}, f.root / "logs"))[1]
    assert entered.wait(10)
    answer = request_owner_pause(ROOT, request_id=f"pause-{kind}")
    release.set()
    _join(threads)
    assert answer["ok"] and _phase(f) == "paused", answer


def test_pending_once_recovery_reflects_on_receipts_recorded_before_it(late, monkeypatch):  # noqa: F811
    f = late
    calls = _real_reflection(f, monkeypatch)
    write_task_result(f.root, ROOT, "completed", total_rounds=20, started_at=START)
    _record(f.root, "slack-bridge", ROOT, delivery_id="send:before-restart")
    _, threads = _spawned(lambda: f.pipeline.recover_pending_root_post_task_synthesis(f.root, f.root))
    _join(threads)
    [prompt] = _reflection_prompts(f)
    assert "delivery send:before-restart" in prompt and len(calls) == 1
    assert _phase(f) == "completed"


def test_resume_before_reflection_reads_the_fresh_horizon(late, monkeypatch):  # noqa: F811
    f = late
    calls = _real_reflection(f, monkeypatch)
    _pause_on(f, "scratchpad", {"tool_calls": []})
    assert _reflection_prompts(f) == [] and calls == []
    _record(f.root, "slack-bridge", ROOT, delivery_id="send:while-paused")  # e.g. an automatic reply's receipt
    assert _resume(f)["ok"]
    [prompt] = _reflection_prompts(f)
    assert "delivery send:while-paused" in prompt and len(calls) == 1
    assert _phase(f) == "completed"


def test_resumed_published_reflection_is_not_rerun_for_a_late_receipt(late, monkeypatch):  # noqa: F811
    f = late
    calls = _real_reflection(f, monkeypatch)
    _record(f.root, "slack-bridge", ROOT, delivery_id="send:first")
    # An errored call admits the nested Pattern Register write; the Pause lands while the
    # reflection's own send is on the wire, so that write is the refused next paid send.
    _pause_on(f, "other", {"tool_calls": [{"tool": "run_command", "status": "error", "is_error": True,
                                           "result": "boom"}]})
    from ouroboros.post_task_checkpoint import late_phase_pause_record
    from ouroboros.task_results import load_task_result

    assert late_phase_pause_record(load_task_result(f.root, ROOT, strict=True))["stage"] == "reflection"
    [prompt] = _reflection_prompts(f)
    assert "delivery send:first" in prompt and len(calls) == 1
    _record(f.root, "slack-bridge", ROOT, delivery_id="send:late")
    assert _resume(f)["ok"]
    assert len(_reflection_prompts(f)) == 1 and len(calls) == 1, "the published reflection is not written again"
    reflections = (f.root / "logs" / "task_reflections.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["task_id"] for line in reflections] == [ROOT]
    assert _phase(f) == "completed"


def test_native_presence_post_task_is_detached_and_its_capture_orders_the_reply_receipts(late, monkeypatch):  # noqa: F811
    """The real dispatch of a native Presence turn returns before synthesis; the adapter's reports
    land around the reflection's capture in a fixed order: one before it is read, one after."""
    import threading

    from ouroboros.usage_accounting import UsageScope, usage_scope

    f = late
    monkeypatch.setattr(f.pipeline, "_run_reflection", post_task_synthesis._run_reflection)
    presence = {"transport_skill": "slack-bridge", "binding_id": "b" * 32, "event": {"source_event_id": "in-1"}}
    write_task_result(f.root, ROOT, "completed", total_rounds=20, started_at=START, metadata={"presence": presence})
    task = {**f.task, "_is_direct_chat": True, "metadata": {"presence": presence}}
    released, captured, resume = threading.Event(), threading.Event(), threading.Event()
    readers: list = []
    real_read, real_now = delivery.task_delivery_receipts, delivery.utc_now_iso

    def read(*args, **kwargs):
        readers.append(threading.current_thread())
        assert released.wait(10)  # the reflection is admitted and waits to capture
        return real_read(*args, **kwargs)

    def now():  # stamped right after the capture: only the reflection's read pauses here
        if threading.current_thread() in readers and not captured.is_set():
            captured.set()
            assert resume.wait(10)
        return real_now()

    monkeypatch.setattr(delivery, "task_delivery_receipts", read)
    monkeypatch.setattr(delivery, "utc_now_iso", now)
    scope = UsageScope(drive_root=f.root, task_id=ROOT, root_task_id=ROOT, category="task", source="agent.task",
                       root_limit_usd=3.0, root_limit_source="task_admission")
    with usage_scope(scope):
        threads = _spawned(lambda: f.pipeline._dispatch_root_post_task(
            f.env, task, ANSWER, None, [], {"rounds": 20}, {"tool_calls": []}, {}, f.root / "logs",
            budget_drive_root=str(f.root), split_drive=False, project_scoped=False, project_task=False,
            parent_env=None, parent_task=None))[1]
    assert threads and any(thread.is_alive() for thread in threads), "dispatch returned before synthesis"
    assert not captured.is_set()
    _record(f.root, "slack-bridge", ROOT, delivery_id="send:before-capture")  # the adapter posted the reply
    released.set()
    assert captured.wait(10)
    _record(f.root, "slack-bridge", ROOT, delivery_id="send:after-capture", part_id="status", state="uncertain")
    resume.set()
    _join(threads)

    [prompt] = _reflection_prompts(f)
    section = prompt[prompt.index(HEADING):].split("\n\n", 1)[0]
    assert [(_bound(line), _delivery(line)) for line in _receipts(section)] == [
        (("host_bound", ROOT), "send:before-capture")]
    assert "send:after-capture" not in prompt and "reports written after the capture stay in chat history" in section
    assert "send:after-capture" in (f.root / "logs" / "chat.jsonl").read_text(encoding="utf-8")
    assert _phase(f) == "completed"
