"""A held dependency reaches a model round before any wait on it parks; the same wait then parks.

The production loop runs (``run_llm_loop``: batch execution, fitted delivery, the post-batch
checkpoint, ``wait_after_tools``) for the resumed root of ``test_current_stop_readers``' real held
tree; only the model's turns are scripted and the park callback records the park instead of
lending a worker. Exposure is the delivered view, as for executor questions (``test_sleep_questions``).
"""
from __future__ import annotations

import copy
import json
import queue as stdqueue
from types import SimpleNamespace

import pytest

from tests.test_current_stop_readers import _held_tree

pytestmark = pytest.mark.serial  # the queue, its workers and the supervisor seams are module state


def _sleep(call_id, **arguments):
    return {"id": call_id, "type": "function",
            "function": {"name": "await_messages", "arguments": json.dumps({"mode": "warm", **arguments})}}


def _turn(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def _shown(messages, call_id):
    """The JSON the model was shown for one call, read from the transcript it was sent."""
    content = next(row["content"] for row in messages if row.get("role") == "tool" and row.get("tool_call_id") == call_id)
    return json.JSONDecoder().raw_decode(content)[0]


def _resumed_root(tmp_path, monkeypatch):
    """The held tree, its root's earlier mail already in its transcript (acknowledged), and the root's own registry."""
    from ouroboros.owner_mailbox import acknowledge_transcript_entry, drain_owner_entries
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    tree = _held_tree(tmp_path, monkeypatch)
    attempt = int(tree.workers.RUNNING["root-1"]["attempt"])
    for entry in drain_owner_entries(tmp_path, "root-1", set(), attempt):
        assert acknowledge_transcript_entry(tmp_path, "root-1", entry)
    write_task_result(tmp_path, "plain", STATUS_RUNNING, chat_id=0)  # an ordinary dependency that simply runs
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt, ctx.root_task_id, ctx.budget_drive_root = "root-1", attempt, "root-1", tmp_path
    ctx.task_metadata = {"root_task_id": "root-1"}
    parks = []
    ctx.owner_wait_callback = lambda _ctx, checkpoint: parks.append(copy.deepcopy(checkpoint["sleep"])) or "owner_text"
    return tree, registry, parks


class _End(BaseException):
    """The script's last word: leaves the loop where it stands, after its checks (a root whose children are
    still held cannot finish, so a scripted answer would only be asked again)."""


def _end(check):
    def turn(messages):
        check(messages)
        raise _End()
    return turn


def _run(tmp_path, monkeypatch, registry, turns):
    """The real loop over scripted turns; each turn may inspect the transcript it is answering."""
    from ouroboros import loop as loop_mod
    from ouroboros.loop import run_llm_loop

    sent, failures = [], []

    def call(_llm, messages, *_a, **_k):
        sent.append(copy.deepcopy(messages))
        if len(sent) > len(turns):
            failures.append(AssertionError(f"unscripted model round {len(sent)}"))
            raise _End()
        turn = turns[len(sent) - 1]
        try:
            return (turn(messages) if callable(turn) else turn), 0.0
        except AssertionError as exc:  # the loop would treat it as a provider failure and ask again
            failures.append(exc)
            raise _End() from exc

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    with pytest.raises(_End):
        run_llm_loop(
            messages=[{"role": "user", "content": "wait for the child"}], tools=registry,
            llm=SimpleNamespace(default_model=lambda: "test-model"), drive_logs=tmp_path,
            emit_progress=lambda *_a, **_k: None, incoming_messages=stdqueue.Queue(),
            task_id="root-1", drive_root=tmp_path)
    if failures:
        raise failures[0]
    return sent


def test_the_hold_reaches_the_next_round_before_any_park_and_the_same_request_then_parks(tmp_path, monkeypatch):
    _tree, registry, parks = _resumed_root(tmp_path, monkeypatch)

    def second(messages):
        shown = _shown(messages, "c1")
        assert shown["reason"] == "dependency_held" and shown["slept"] is False
        assert shown["held"][0]["task_id"] == "child-run" and shown["held"][0]["door"] == "resume_child_task"
        assert parks == [], "nothing parked before this model round"
        return _turn(_sleep("c2", tasks=["child-run"], senders=["child-run"]))

    def third(messages):
        assert _shown(messages, "c2")["reason"] == "sleep_armed"
        assert [park["tasks"] for park in parks] == [["child-run"]], "the deliberate wait parked once"
        assert any("were woken by the owner's message or control" in str(row.get("content")) for row in messages)

    sent = _run(tmp_path, monkeypatch, registry, [
        _turn(_sleep("c1", tasks=["child-run"], senders=["child-run"])), second, _end(third)])
    assert len(sent) == 3
    identity = registry._ctx._held_dependencies_seen["child-run"]
    from ouroboros.task_status import load_effective_task_result

    assert identity == load_effective_task_result(tmp_path, "child-run")["current_stop"]["identity"]


@pytest.mark.parametrize("order", ["held_first", "held_last"])
def test_two_waits_in_one_batch_cannot_park_before_the_hold_is_seen(tmp_path, monkeypatch, order):
    _tree, registry, parks = _resumed_root(tmp_path, monkeypatch)
    held, plain = _sleep("h", tasks=["child-run"]), _sleep("p", tasks=["plain"])
    batch = [held, plain] if order == "held_first" else [plain, held]

    def second(messages):
        assert parks == [], "neither call of the batch parked before this round"
        assert _shown(messages, "h")["reason"] == "dependency_held"
        other = _shown(messages, "p")
        if order == "held_first":  # the hold is still undelivered: the later call shows it too, armed nothing
            assert other["reason"] == "dependency_held" and other["held"][0]["task_id"] == "child-run"
            assert other["held"][0]["from_earlier_call"] is True, "said as the earlier call's, not this wait's"
        else:  # the earlier call armed; the later one withdrew it, saying so
            assert other["reason"] == "sleep_armed" and "withdrawn" in _shown(messages, "h")["note"]
        return _turn(_sleep("p2", tasks=["plain"]))

    def third(messages):
        assert _shown(messages, "p2")["reason"] == "sleep_armed" and [park["tasks"] for park in parks] == [["plain"]]

    assert len(_run(tmp_path, monkeypatch, registry, [_turn(*batch), second, _end(third)])) == 3


def _wait(call_id, name):
    arguments = {"task_id": "child-run"} if name == "wait_task" else {"task_ids": ["child-run"]}
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps({**arguments, "timeout_sec": 0})}}


def _content(messages, call_id):
    return next(row["content"] for row in messages if row.get("role") == "tool" and row.get("tool_call_id") == call_id)


@pytest.mark.parametrize("name", ["wait_task", "wait_tasks"])
@pytest.mark.parametrize("order", ["sleep_first", "wait_first"])
def test_a_task_wait_beside_a_sleep_shows_the_hold_before_either_parks(tmp_path, monkeypatch, order, name):
    """The nanny's falsifier (an armed sleep, then an in-slot wait on the held child) and its mirror."""
    _tree, registry, parks = _resumed_root(tmp_path, monkeypatch)
    sleep, wait = _sleep("p", tasks=["plain"]), _wait("w", name)

    def second(messages):
        assert parks == [], "no call of the batch parked before this round"
        text = _content(messages, "w")
        if name == "wait_task":
            assert text.startswith("Task wait returned to show a held dependency") and '"task_id": "child-run"' in text
            note = text
        else:
            early = _shown(messages, "w")["early_return"]
            assert early["reason"] == "dependency_held" and early["held"][0]["task_id"] == "child-run"
            note = _shown(messages, "w")["early_return_note"]
        if order == "sleep_first":  # the wait withdrew the sleep armed before it, saying so
            assert _shown(messages, "p")["reason"] == "sleep_armed"
            assert "The sleep an earlier call of this batch armed is withdrawn." in note
        else:  # the later sleep shows the hold as the earlier call's and arms nothing
            other = _shown(messages, "p")
            assert other["reason"] == "dependency_held" and other["held"][0]["from_earlier_call"] is True
            assert "withdrawn" not in note
        return _turn(_wait("w2", name), _sleep("p2", tasks=["plain"]))

    def third(messages):
        assert "held dependency" not in _content(messages, "w2") and "dependency_held" not in _content(messages, "w2"), \
            "the delivered hold is not shown again: the wait is an ordinary wait on it"
        assert _shown(messages, "p2")["reason"] == "sleep_armed" and [park["tasks"] for park in parks] == [["plain"]]

    batch = [sleep, wait] if order == "sleep_first" else [wait, sleep]
    assert len(_run(tmp_path, monkeypatch, registry, [_turn(*batch), second, _end(third)])) == 3


def test_only_a_complete_delivered_view_records_the_hold(tmp_path, monkeypatch):
    from ouroboros import loop_tool_execution as loop
    from ouroboros.tools.control_task_results import _await_messages
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR
    from tests.test_current_stop_readers import _drain, _root_ctx

    _held_tree(tmp_path, monkeypatch)
    ctx = _drain(_root_ctx(tmp_path), tmp_path)
    text = _await_messages(ctx, mode="warm", tasks=["child-run"])
    meta = dict(getattr(ctx, _TOOL_RESULT_ATTR).meta)

    def deliver(*, partial=False, error=False):
        monkeypatch.setattr(loop, "_deliver_tool_results", lambda *_a, **_k: [{"result": text, "result_partial": partial}])
        loop.process_tool_results([{"fn_name": "await_messages", "tool_call_id": "c", "result": text, "is_error": error,
                                    "args_for_log": {}, "tool_args": {}, "result_meta": {"tool_result_meta": meta}}],
                                  [], {"tool_calls": []}, lambda *_a, **_k: None, tools=SimpleNamespace(_ctx=ctx))

    assert ctx._held_dependencies_pending and not getattr(ctx, "_held_dependencies_seen", None)
    deliver(partial=True)
    assert not getattr(ctx, "_held_dependencies_seen", None) and ctx._held_dependencies_pending == []
    assert json.loads(_await_messages(ctx, mode="warm", tasks=["child-run"]))["reason"] == "dependency_held"
    deliver(error=True)
    assert not getattr(ctx, "_held_dependencies_seen", None)
    _await_messages(ctx, mode="warm", tasks=["child-run"])
    deliver()
    assert json.loads(_await_messages(ctx, mode="warm", tasks=["child-run"]))["reason"] == "sleep_armed"


def test_a_saved_wait_keeps_what_it_was_shown_and_the_supervisor_never_wakes_on_a_hold(tmp_path, monkeypatch):
    from ouroboros import owner_wait
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tools.control_task_results import _await_messages
    from supervisor.restart_retention import retain_sleep_checkpoint
    from supervisor.sleep_wake import _readiness
    from tests.test_current_stop_readers import _deliver, _drain, _published, _root_ctx

    restore = owner_wait.restore_continuation_state  # the tree's own root consumption stubs it afterwards
    _held_tree(tmp_path, monkeypatch)
    ctx = _drain(_root_ctx(tmp_path), tmp_path)
    assert json.loads(_await_messages(ctx, mode="warm", tasks=["child-run"]))["reason"] == "dependency_held"
    _deliver(ctx, _published(ctx))
    assert json.loads(_await_messages(ctx, mode="warm", tasks=["child-run"]))["reason"] == "sleep_armed"
    messages = [{"role": "user", "content": "wait for the child"}]
    checkpoint = owner_wait.checkpoint_owner_wait(ctx, messages, {}, {}, 1, [], set())
    owner_wait.set_owner_wait(tmp_path, "root-1", {**checkpoint, "state": "waiting"})
    _wait, saved = owner_wait.saved_sleep_checkpoint(tmp_path, "root-1", int(ctx.task_attempt))
    assert saved["held_dependencies_seen"] == ctx._held_dependencies_seen
    restored = _drain(_root_ctx(tmp_path), tmp_path)
    restore(SimpleNamespace(_ctx=restored), saved, [], {}, {}, set())
    assert restored._held_dependencies_seen == ctx._held_dependencies_seen
    assert json.loads(_await_messages(restored, mode="warm", tasks=["child-run"]))["reason"] == "sleep_armed", \
        "the restored wait keeps what it was shown while the tree's resume basis stands"
    legacy = _drain(_root_ctx(tmp_path), tmp_path)
    restore(SimpleNamespace(_ctx=legacy), {
        key: value for key, value in saved.items() if key != "held_dependencies_seen"}, [], {}, {}, set())
    assert json.loads(_await_messages(legacy, mode="warm", tasks=["child-run"]))["reason"] == "dependency_held", \
        "a saved state from before this map shows the hold once more, never loses it"

    retained = retain_sleep_checkpoint(tmp_path, "root-1", int(ctx.task_attempt))
    source = json.loads(read_actor_source_bytes(tmp_path, "root-1", retained["source_ref"]))
    assert source["held_dependencies_seen"] == ctx._held_dependencies_seen
    task = {"id": "root-1", "_attempt": int(ctx.task_attempt), "drive_root": str(tmp_path)}
    assert _readiness(SimpleNamespace(DRIVE_ROOT=tmp_path), task, retained) == "", \
        "the supervisor's cold wake reads terminals, never the hold: a re-armed wait is not woken by it"
    # The restart kept the sleep as the root's own new pause: a new resume basis, under which an earlier child
    # selection would be refused too. The restored wait shows the hold once more, at its new identity.
    after = _drain(_root_ctx(tmp_path), tmp_path)
    restore(SimpleNamespace(_ctx=after), source, [], {}, {}, set())
    shown = json.loads(_await_messages(after, mode="warm", tasks=["child-run"]))
    assert shown["reason"] == "dependency_held"
    assert shown["held"][0]["identity"] != ctx._held_dependencies_seen["child-run"]
    _deliver(after, _published(after))
    assert json.loads(_await_messages(after, mode="warm", tasks=["child-run"]))["reason"] == "sleep_armed"


def _in_slot(call_id):
    return {"id": call_id, "type": "function",
            "function": {"name": "await_messages", "arguments": json.dumps({"mode": "in_slot", "timeout_sec": 30})}}


def test_a_hold_an_earlier_call_returned_ends_a_later_in_slot_mail_wait_at_once_then_it_waits_as_before(
        tmp_path, monkeypatch):
    """Astra's R1 falsifier (a returned hold, then the default in-slot mail wait in the same batch): the wait gives
    the batch's undelivered hold back at once, from that batch's own state; once delivered, the same wait is the
    ordinary mail wait again. Only the model's turns and the wait's clock are scripted."""
    from ouroboros import task_status
    from ouroboros.tools import control_task_results as controls

    _tree, registry, parks = _resumed_root(tmp_path, monkeypatch)
    tick, slept, reads = [0.0], [], []

    def sleep(seconds):
        slept.append(seconds)
        tick[0] += seconds

    monkeypatch.setattr(controls, "time", SimpleNamespace(monotonic=lambda: tick[0], sleep=sleep))
    real_read = task_status.load_effective_task_result

    def read(root, task_id, **kwargs):
        reads.append(task_id)
        return real_read(root, task_id, **kwargs)

    monkeypatch.setattr(task_status, "load_effective_task_result", read)
    in_slot_reads = []

    def await_messages(ctx, **kwargs):
        before = len(reads)
        try:
            return controls._await_messages(ctx, **kwargs)
        finally:
            if kwargs.get("mode") == "in_slot":
                in_slot_reads.append(len(reads) - before)

    registry.override_handler("await_messages", await_messages)

    def second(messages):
        assert _shown(messages, "hold")["reason"] == "dependency_held" and parks == []
        mail = _shown(messages, "mail")
        assert (mail["reason"], mail["elapsed_sec"], slept) == ("dependency_held", 0.0, []), "no tick of the wait ran"
        assert mail["held"][0]["task_id"] == "child-run" and mail["held"][0]["from_earlier_call"] is True
        assert in_slot_reads == [0], "the batch already holds the hold: the in-slot wait reads no task"
        return _turn(_in_slot("again"))

    def third(messages):
        again = _shown(messages, "again")
        assert (again["reason"], again["elapsed_sec"], sum(slept)) == ("timeout", 30.0, 30.0), \
            "the delivered hold leaves the ordinary mail wait its whole window"
        assert "held" not in again and parks == []

    _run(tmp_path, monkeypatch, registry, [
        _turn(_sleep("hold", tasks=["child-run"]), _in_slot("mail")), second, _end(third)])


def _running_child(tree, tmp_path, task_id="plain"):
    """A second running child of the root, as a successful dispatch publishes it."""
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, task_id, STATUS_RUNNING, chat_id=0, parent_task_id="root-1", root_task_id="root-1",
                      delegation_role="subagent")
    tree.workers.RUNNING[task_id] = {"task": {
        "id": task_id, "type": "task", "chat_id": 0, "root_task_id": "root-1", "parent_task_id": "root-1",
        "delegation_role": "subagent"}, "worker_id": 99, "attempt": 1}
    assert tree.queue.persist_queue_snapshot(reason="second-child-running")


def _authority_pause(tree, tmp_path, task_id="plain"):
    """The child's own task-scope pause after a failed authority read (``loop_budget._handle_authority_unavailable``),
    installed exactly: it holds that child alone and sends its parent nothing."""
    from ouroboros import budget_pause
    from ouroboros.task_status import load_effective_task_result
    from supervisor.events_budget import install_exact_budget_pause
    from tests._budget_pause_exact_helpers import _loop_ctx

    child, limit = _loop_ctx(tmp_path, task_id)
    child.root_task_id = "root-1"
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit, rail=budget_pause.RAIL_CONTROL_AUTHORITY, scope="task",
                                   reason_text="dispatch-fence authority read failed")
    budget_pause.end_dispatch_fence(task_id)
    install_exact_budget_pause(tree.sup, task_id, budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    stop = load_effective_task_result(tmp_path, task_id)["current_stop"]
    assert (stop["state"], stop["cause"]) == ("held", "authority")
    return stop


def test_a_child_held_after_its_parent_armed_a_warm_sleep_reaches_the_mind_first_and_the_same_request_then_parks(
        tmp_path, monkeypatch):
    """Astra's R1 falsifier: a later tool of the batch that armed the sleep sees the selected child enter its own
    authority pause. One addressed refresh at the actual park returns the hold to the mind, recorded with the
    transcript before that model round; the deliberate repeat then parks, and the hold never wakes it."""
    from ouroboros import model_sleep, working_checkpoint

    tree, registry, parks = _resumed_root(tmp_path, monkeypatch)
    _running_child(tree, tmp_path)
    held = []

    def child_pauses_during_a_later_tool(ctx, **_kwargs):
        assert ctx._model_sleep["tasks"] == ["plain"], "the parent's sleep is already armed"
        held.append(_authority_pause(tree, tmp_path))
        assert model_sleep.wake_reason(ctx, ctx._model_sleep) == "", "a child-only hold sends no wake"
        return "independent read completed"

    registry.override_handler("knowledge_read", child_pauses_during_a_later_tool)
    read = {"id": "read", "type": "function",
            "function": {"name": "knowledge_read", "arguments": json.dumps({"topic": "placeholder"})}}

    def second(messages):
        assert _shown(messages, "arm")["reason"] == "sleep_armed" and parks == [], "nothing parked"
        [notice] = [row["content"] for row in messages if "sleep was not taken" in str(row.get("content"))]
        assert notice.startswith("[SYSTEM NOTICE]\nYour warm sleep was not taken: since it was armed")
        assert '"task_id": "plain"' in notice and held[0]["identity"] in notice and '"cause": "authority"' in notice
        ctx = registry._ctx
        assert ctx._held_dependencies_seen["plain"] == held[0]["identity"] and not ctx._model_sleep
        saved = json.loads(working_checkpoint.checkpoint_path(tmp_path, "root-1", ctx.task_attempt).read_text())
        assert saved["held_dependencies_seen"]["plain"] == held[0]["identity"], "durable before this model round"
        assert notice in [row.get("content") for row in saved["messages"]]
        return _turn(_sleep("again", tasks=["plain"]))

    def third(messages):
        assert _shown(messages, "again")["reason"] == "sleep_armed"
        assert [park["tasks"] for park in parks] == [["plain"]], "the deliberate repeat on the same stop parked once"
        assert len([row for row in messages if "sleep was not taken" in str(row.get("content"))]) == 1

    _run(tmp_path, monkeypatch, registry, [_turn(_sleep("arm", tasks=["plain"]), read), second, _end(third)])


def test_a_dependency_held_after_a_cold_sleep_was_armed_returns_before_the_cold_park_and_the_same_request_parks(
        tmp_path, monkeypatch):
    """The cold park boundary (``enter_cold_sleep``): the same refresh, before the exact pause is requested."""
    from ouroboros import budget_pause, model_sleep
    from tests._budget_pause_exact_helpers import _fast_hold, _mock_pause_observation
    from tests.test_current_stop_readers import _continued

    tree = _held_tree(tmp_path, monkeypatch)
    ctx, limit = _continued(tree, tmp_path)  # a Continue episode: its cold sleep may select another tree's child
    _running_child(tree, tmp_path)
    assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, tasks=["plain"]), "cold")["reason"] == "sleep_armed"
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    stop = _authority_pause(tree, tmp_path)
    before = len(limit.messages)
    budget_pause.enter_cold_sleep(limit)  # returned: no exact pause was requested
    [notice] = [row["content"] for row in limit.messages[before:]]
    assert notice.startswith("[SYSTEM NOTICE]\nYour cold sleep was not taken: since it was armed")
    assert stop["identity"] in notice and ctx._model_sleep is None and not ctx._owner_wait_requested
    assert ctx._held_dependencies_seen == {"plain": stop["identity"]}
    assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, tasks=["plain"]), "cold")["reason"] == "sleep_armed"
    with pytest.raises(budget_pause.BudgetPauseRequested):
        budget_pause.enter_cold_sleep(limit)  # the deliberate repeat parks
    assert len(limit.messages) == before + 1


def test_the_park_refresh_reads_each_selected_dependency_once_and_parks_on_an_unchanged_or_unreadable_one(
        tmp_path, monkeypatch, caplog):
    import logging

    from ouroboros import model_sleep, task_status
    from ouroboros.tools.control_task_results import _await_messages
    from tests.test_current_stop_readers import _deliver, _published, _root_ctx

    _held_tree(tmp_path, monkeypatch)
    root, chosen = _root_ctx(tmp_path), {"tasks": ["child-run", "child-new"], "senders": ["child-run"]}
    assert json.loads(_await_messages(root, mode="warm", **chosen))["reason"] == "dependency_held"
    _deliver(root, _published(root))
    assert json.loads(_await_messages(root, mode="warm", **chosen))["reason"] == "sleep_armed"
    reads, real_read = [], task_status.load_effective_task_result
    monkeypatch.setattr(task_status, "load_effective_task_result",
                        lambda drive, task_id, **kwargs: reads.append(task_id) or real_read(drive, task_id, **kwargs))
    messages = []
    assert model_sleep.withdraw_for_new_holds(root, messages) is False and messages == [] and root._model_sleep
    assert reads == ["child-run", "child-new"], "one addressed read per selected dependency, no other source"

    def unreadable(*_args, **_kwargs):
        raise OSError("the task store is busy")

    monkeypatch.setattr(task_status, "load_effective_task_result", unreadable)
    with caplog.at_level(logging.WARNING, logger="ouroboros.model_sleep"):
        assert model_sleep.withdraw_for_new_holds(root, messages) is False and root._model_sleep, "no new gate"
    assert "unreadable before root-1 parks" in caplog.text and messages == []
    root._model_sleep = {"sleep_id": "time-only", "mode": "warm", "tasks": [], "senders": [], "wake_at": "x"}
    reads.clear()
    monkeypatch.setattr(task_status, "load_effective_task_result", real_read)
    assert model_sleep.withdraw_for_new_holds(root, messages) is False and reads == []


def test_a_held_wait_set_of_any_supported_size_carries_one_bounded_flag_and_only_its_complete_view_records_it(
        tmp_path, monkeypatch):
    """Astra's R1 probe at the supported fan-out: the holds a ``wait_tasks`` shows ride in its text, the meta is
    one flag, and the delivered view decides what was shown (complete: every identity; partial or error: none)."""
    from unittest.mock import patch

    from ouroboros import loop_tool_execution as loop
    from ouroboros.config import MAX_ACTIVE_SUBAGENTS_HARD_CAP
    from ouroboros.model_sleep import HOLD_NOTICE
    from ouroboros.tools import control_task_results as readers
    from ouroboros.tools.tool_result import _MAX_META_BYTES, _TOOL_RESULT_ATTR
    from tests.test_current_stop_readers import _root_ctx

    for count in (32, 160, MAX_ACTIVE_SUBAGENTS_HARD_CAP):
        (tmp_path / str(count)).mkdir()
        ctx = _root_ctx(tmp_path / str(count))
        rows = {f"{n:08x}": {"task_id": f"{n:08x}", "status": "scheduled", "root_task_id": "root-1",
                             "parent_task_id": "root-1", "current_stop": {
                                 "state": "held", "scope": "member", "release": "mind", "cause": "awaiting_selection",
                                 "code": "awaiting_selection", "identity": f"{n:016x}"}} for n in range(1, count + 1)}

        def load(_root, task_id, **_kwargs):
            return rows.get(task_id, {})

        with patch.object(readers, "load_effective_task_result", load), \
                patch("ouroboros.task_status.load_effective_task_result", load), \
                patch("ouroboros.task_status._load_queue_snapshot", lambda _root: {"pending": [], "running": []}), \
                patch("ouroboros.loop_transport._owner_signal_pending", lambda *_a, **_k: False):
            def wait():
                text = readers._wait_for_tasks(ctx, list(rows), timeout_sec=0)
                return text, json.loads(text), dict(getattr(ctx, _TOOL_RESULT_ATTR).meta)

            def deliver(text, meta, *, partial=False, error=False):
                monkeypatch.setattr(loop, "_deliver_tool_results",
                                    lambda *_a, **_k: [{"result": text, "result_partial": partial}])
                loop.process_tool_results([{
                    "fn_name": "wait_tasks", "tool_call_id": "w", "result": text, "is_error": error, "args_for_log": {},
                    "tool_args": {}, "result_meta": {"tool_result_meta": meta}}],
                    [], {"tool_calls": []}, lambda *_a, **_k: None, tools=SimpleNamespace(_ctx=ctx))

            text, waited, meta = wait()
            assert waited["early_return"]["reason"] == "dependency_held", count
            assert len(waited["early_return"]["held"]) == count and meta == {HOLD_NOTICE: True}
            assert len(json.dumps(meta).encode()) < _MAX_META_BYTES // 100, "bounded whatever the wait set's size"
            deliver(text, meta, partial=True)
            deliver(text, meta, error=True)
            assert not getattr(ctx, "_held_dependencies_seen", None), "a partial or failed view records nothing"
            text, waited, meta = wait()
            assert waited["early_return"]["reason"] == "dependency_held", "so it is shown again"
            deliver(text, meta)
            assert ctx._held_dependencies_seen == {tid: row["current_stop"]["identity"] for tid, row in rows.items()}
            assert "early_return" not in wait()[1], "shown once in full: the same wait is an ordinary wait"


_FRAME = {"window": 200_000, "reserve": 16_000}  # Astra's R2 frame: the supported wait arrives shortened


def _held_children(tmp_path, monkeypatch, count):
    """``count`` held children behind the real readers (the task store stubbed, as in the R1 probe), and a root
    whose results reach its transcript through the loop's own measured delivery (``process_tool_results``)."""
    from ouroboros import task_status
    from ouroboros.tools import control_task_results as readers
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root-1")
    ctx.task_attempt, ctx.root_task_id, ctx.budget_drive_root, ctx.task_metadata = 1, "root-1", tmp_path, {}
    ctx._loop_mailbox_seen_ids = set()
    setattr(ctx, _TOOL_RESULT_ATTR, None)
    rows = {f"{n:08x}": {"task_id": f"{n:08x}", "status": "scheduled", "root_task_id": "root-1",
                         "parent_task_id": "root-1", "current_stop": {
                             "state": "held", "scope": "member", "release": "mind", "cause": "awaiting_selection",
                             "code": "awaiting_selection", "identity": f"{n:016x}"}} for n in range(1, count + 1)}

    def load(_root, task_id, **_kwargs):
        return rows.get(task_id, {})

    monkeypatch.setattr(readers, "load_effective_task_result", load)
    monkeypatch.setattr(task_status, "load_effective_task_result", load)
    monkeypatch.setattr(task_status, "_load_queue_snapshot", lambda _root: {"pending": [], "running": []})
    monkeypatch.setattr("ouroboros.loop_transport._owner_signal_pending", lambda *_a, **_k: False)
    return ctx, rows


def _identities(rows):
    return {tid: row["current_stop"]["identity"] for tid, row in rows.items()}


def _waited(ctx, rows, select=None):
    """``wait_tasks`` over every child (or ``select``), delivered at the frame: its text, the transcript and the
    call's trace row."""
    from ouroboros.tools import control_task_results as readers
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR
    from tests._tool_result_delivery_shared import measured_fit
    from tests.test_tool_result_delivery import _deliver, _row

    ids = list(select or rows)
    text = readers._wait_for_tasks(ctx, ids, timeout_sec=0)
    messages, trace = _deliver(ctx, [_row("wait", "wait_tasks", text, args={"task_ids": ids, "timeout_sec": 0},
                                          meta={"tool_result_meta": dict(getattr(ctx, _TOOL_RESULT_ATTR).meta)})],
                               measured_fit(**_FRAME))
    return text, messages, trace["tool_calls"][0]


def _key(ref):
    """A pending hold source is the file its shortened view named: its retained path and its exact bytes."""
    return f"{ref['path']}|{ref['sha256']}"


def _read(ctx, args, *, messages=None, fit=None, error=False):
    """One ``read_file`` call delivered as the loop delivers it; returns the transcript and the call's trace row."""
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR
    from tests._tool_result_delivery_shared import measured_fit
    from tests.test_tool_result_delivery import _deliver, _row

    text = _read_file(ctx, **args)
    messages, trace = _deliver(ctx, [_row(f"read-{len(messages or ())}", "read_file", text, args=args, is_error=error,
                                          meta={"tool_result_meta": dict(getattr(ctx, _TOOL_RESULT_ATTR).meta)})],
                               fit or measured_fit(**_FRAME), messages=messages)
    return messages, trace["tool_calls"][0]


def _page(ref, start, end):
    return {"root": ref["root"], "path": ref["path"], "start_line": 1, "max_lines": 100_000,
            "start_char": start, "max_chars": end - start}


def test_a_shortened_hold_view_read_whole_from_its_exact_source_lets_the_same_wait_wait(tmp_path, monkeypatch):
    """Astra's R2 probe, its premise and final assertion kept: the supported 500-child wait reaches a 200k window
    shortened, its complete text kept as an exact source; every byte of that source then read through ``read_file``
    in explicit pages, each delivered whole, shows every hold, so the deliberate repeat of the same wait waits."""
    from ouroboros.config import MAX_ACTIVE_SUBAGENTS_HARD_CAP
    from ouroboros.tools import control_task_results as readers
    from tests.test_tool_result_delivery import _source_ref

    ctx, rows = _held_children(tmp_path, monkeypatch, MAX_ACTIVE_SUBAGENTS_HARD_CAP)
    full, messages, shown = _waited(ctx, rows)
    assert shown.get("result_partial"), "the supported 500-child wait is projected at a 200k window"
    ref = _source_ref(messages[0]["content"])
    assert not getattr(ctx, "_held_dependencies_seen", None)
    for start in range(0, len(full), 20_000):
        messages, page = _read(ctx, _page(ref, start, min(start + 20_000, len(full))), messages=messages)
        assert ctx.last_read_view["source_start_char"] == start
        assert not page.get("result_partial"), "every selected page reached the transcript whole"
    assert ctx._held_dependencies_seen == _identities(rows), "read from the retained source itself"
    assert ctx._held_dependency_sources == {}, "a completed source is done"
    repeated = json.loads(readers._wait_for_tasks(ctx, list(rows), timeout_sec=0))
    assert "early_return" not in repeated, "a fully read hold source must allow the deliberate repeat wait"


def test_only_what_was_delivered_of_that_exact_source_shows_a_shortened_hold(tmp_path, monkeypatch):
    """The shortened view's own head and tail count, so its advertised reader of the omitted middle completes it.
    A failed read, a page the frame cut, the same bytes under another name and a file that merely claims the holds
    credit nothing they did not deliver of that source."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tools import control_task_results as readers
    from tests._tool_result_delivery_shared import measured_fit
    from tests.test_tool_result_delivery import _source_ref

    ctx, rows = _held_children(tmp_path, monkeypatch, 160)
    full, messages, shown = _waited(ctx, rows)
    ref, omitted = _source_ref(messages[0]["content"]), shown["result_source_view"]["read_omitted"]
    assert shown.get("result_partial") and list(ctx._held_dependency_sources) == [_key(ref)]
    first, last = omitted["start_char"], omitted["start_char"] + omitted["max_chars"]
    store = task_artifact_dir_path(tmp_path, "root-1")
    (store / "copy.txt").write_bytes((store / ref["path"]).read_bytes())  # the same bytes under another name
    (store / "claim.json").write_text(json.dumps({"held": [{"task_id": tid, "identity": identity}
                                                           for tid, identity in _identities(rows).items()]}))
    for name in ("copy.txt", "claim.json"):
        whole = {"root": "artifact_store", "path": name, "start_line": 1, "max_lines": 100_000}
        assert not _read(ctx, whole)[1].get("result_partial"), "each is delivered whole"
    assert not getattr(ctx, "_held_dependencies_seen", None), "another name or a mere claim is not that source"
    middle = list(range(first, last, 20_000))
    assert len(middle) >= 3
    _read(ctx, _page(ref, middle[1], min(middle[1] + 20_000, last)), error=True)  # a failed read of a real page
    for start in [middle[0], *middle[2:]]:
        assert not _read(ctx, _page(ref, start, min(start + 20_000, last)))[1].get("result_partial")
    # A chosen page is cut only when it exceeds the usable input by itself: a frame smaller than the page.
    cut = _read(ctx, _page(ref, middle[1], min(middle[1] + 20_000, last)), fit=measured_fit(window=3_000))[1]
    assert cut.get("result_partial"), "this page reached the transcript shortened"
    assert not getattr(ctx, "_held_dependencies_seen", None), "one page of that source was never delivered whole"
    again = json.loads(readers._wait_for_tasks(ctx, list(rows), timeout_sec=0))
    assert again["early_return"]["reason"] == "dependency_held", "so the same wait still returns to show them"
    _read(ctx, _page(ref, middle[1], min(middle[1] + 20_000, last)))
    assert ctx._held_dependencies_seen == _identities(rows), "the head and tail it was shown, then the middle"
    assert "early_return" not in json.loads(readers._wait_for_tasks(ctx, list(rows), timeout_sec=0)), \
        "the same wait now waits"


def test_a_saved_continuation_keeps_a_shortened_hold_views_source_and_a_new_process_reads_it_again(
        tmp_path, monkeypatch):
    """What waits to be read whole rides the continuation beside what was shown; coverage is the reading process's
    own, so the restored actor proves it by what is delivered to it, and a saved state from before this map keeps
    the holds unshown, never shown."""
    from ouroboros import delegate_output, owner_wait
    from ouroboros.tools import control_task_results as readers
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR
    from tests.test_tool_result_delivery import _source_ref, _view_json

    ctx, rows = _held_children(tmp_path, monkeypatch, 160)
    full, messages, _shown = _waited(ctx, rows)
    ref = _source_ref(messages[0]["content"])
    state = owner_wait.continuation_state(ctx, messages, {}, {}, 1, [], set())
    kept = {_key(ref): {**ref, "order": 1}}  # with the place its shortened view arrived at
    assert state["held_dependency_sources"] == kept and state["held_dependencies_seen"] == {}

    def restored(saved):
        fresh = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root-1")
        fresh.task_attempt, fresh.root_task_id, fresh.budget_drive_root, fresh.task_metadata = 1, "root-1", tmp_path, {}
        setattr(fresh, _TOOL_RESULT_ATTR, None)
        owner_wait.restore_continuation_state(SimpleNamespace(_ctx=fresh), saved, [], {}, {}, set())
        return fresh

    legacy = restored({key: value for key, value in state.items() if key != "held_dependency_sources"})
    assert legacy._held_dependency_sources == {}
    fresh = restored(json.loads(json.dumps(state)))
    assert fresh._held_dependency_sources == kept
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})  # a new worker: nothing was delivered to it yet
    view = _view_json(messages[0]["content"])
    first, last = view["omitted_ranges"][0]
    for start in range(first, last, 20_000):
        _read(fresh, _page(ref, start, min(start + 20_000, last)))
    assert not getattr(fresh, "_held_dependencies_seen", None), "the head and tail reached the earlier process"
    for start, end in view["shown_ranges"]:
        _read(fresh, _page(ref, start, end))
    assert fresh._held_dependencies_seen == _identities(rows) and fresh._held_dependency_sources == {}
    assert "early_return" not in json.loads(readers._wait_for_tasks(fresh, list(rows), timeout_sec=0))


def test_an_older_shortened_hold_view_stays_completable_however_many_follow_it(tmp_path, monkeypatch):
    """Waits over different selected sets show different holds, so a later shortened view never stands in for
    an earlier one. Ten real shortened deliveries in turn, then the oldest advertised source read whole: its
    holds count, its same wait now waits, and every later source still waits to be read."""
    from ouroboros.tools import control_task_results as readers
    from tests.test_tool_result_delivery import _source_ref

    ctx, rows = _held_children(tmp_path, monkeypatch, 169)
    sets = [list(rows)[start:start + 160] for start in range(10)]
    shown = [_waited(ctx, rows, select) for select in sets]
    assert all(row.get("result_partial") for _text, _messages, row in shown), "each arrived shortened"
    refs = [_source_ref(messages[0]["content"]) for _text, messages, _row in shown]
    assert len({ref["sha256"] for ref in refs}) == 10, "ten distinct exact sources"
    oldest, full = refs[0], shown[0][0]
    for start in range(0, len(full), 20_000):
        assert not _read(ctx, _page(oldest, start, min(start + 20_000, len(full))))[1].get("result_partial")
    assert {tid: (getattr(ctx, "_held_dependencies_seen", None) or {}).get(tid) for tid in sets[0]} == {
        tid: rows[tid]["current_stop"]["identity"] for tid in sets[0]}, "the oldest source, read whole, shows its holds"
    assert "early_return" not in json.loads(readers._wait_for_tasks(ctx, sets[0], timeout_sec=0)), \
        "its same wait now waits"
    assert sorted(ctx._held_dependency_sources) == sorted(_key(ref) for ref in refs[1:]), \
        "each later source is still there to be read"


def _restop(rows, task_id, **fields):
    """That child's stop as it reads now: a newer one, or its recovery (a ``None`` field is gone)."""
    stop = {**rows[task_id]["current_stop"], **fields}
    rows[task_id]["current_stop"] = {key: value for key, value in stop.items() if value is not None}


def _rearm(ctx, task_ids):
    """The same wait, deliberately armed again as a warm sleep (``request_sleep``, the store's rows stubbed)."""
    from ouroboros import model_sleep

    chosen = {"senders": [], "tasks": list(task_ids), "runs": [], "wake_at": "", "any_mail": False}
    return model_sleep.request_sleep(ctx, chosen, "warm")["reason"]


def _older_then_newer(tmp_path, monkeypatch):
    """Astra's R3 premise at the readers: an older wait over 160 held children arrives shortened, its exact source
    kept; then one child is paused anew and another's root resume basis becomes unreadable, a newer wait over those
    two reaches the transcript whole, and that same wait is deliberately armed as a sleep."""
    from tests.test_tool_result_delivery import _source_ref

    ctx, rows = _held_children(tmp_path, monkeypatch, 160)
    old, (paused, unknown) = _identities(rows), list(rows)[:2]
    full, messages, shown = _waited(ctx, rows)
    assert shown.get("result_partial"), "the older view arrived shortened"
    _restop(rows, paused, cause="pause", scope="tree", identity="newer-pause")
    _restop(rows, unknown, resume_basis="unreadable", identity="basis-unreadable")
    assert not _waited(ctx, rows, [paused, unknown])[2].get("result_partial"), "the newer view arrived whole"
    assert ctx._held_dependencies_seen == {paused: "newer-pause", unknown: "basis-unreadable"}
    assert _rearm(ctx, [paused, unknown]) == "sleep_armed", "shown once: the same wait arms"
    return ctx, rows, old, full, _source_ref(messages[0]["content"])


def _read_whole(ctx, ref, full):
    """A later tool of the batch that armed the sleep reads the older source whole, page by page."""
    for start in range(0, len(full), 20_000):
        assert not _read(ctx, _page(ref, start, min(start + 20_000, len(full))))[1].get("result_partial")


def _park_refresh(ctx):
    """The park's one addressed refresh (``withdraw_for_new_holds``): the holds its notice shows, or ``None`` when
    the sleep parks."""
    from ouroboros import model_sleep

    notices = []
    if not model_sleep.withdraw_for_new_holds(ctx, notices):
        assert notices == [] and ctx._model_sleep
        return None
    [notice] = [row["content"] for row in notices]
    return [(row["task_id"], row["identity"]) for row in json.loads(
        notice.split("[HELD_DEPENDENCY]\n", 1)[1].split("\n[/HELD_DEPENDENCY]", 1)[0])]


def test_an_older_shortened_hold_view_read_whole_later_keeps_the_newer_stops_so_the_rearmed_sleep_parks(
        tmp_path, monkeypatch):
    """Astra's R3 finding: the older view counts at the place it arrived. Its other holds count and its source is
    done, while the newer stops stay what was shown, so the deliberately re-armed sleep parks; a Pause newer than
    every view is still shown before the park, and the same request then parks."""
    from ouroboros.tools import control_task_results as readers

    ctx, rows, old, full, ref = _older_then_newer(tmp_path, monkeypatch)
    paused, unknown = list(rows)[:2]
    _read_whole(ctx, ref, full)
    assert ctx._held_dependency_sources == {}, "the older source still completes"
    assert _park_refresh(ctx) is None, "nothing the re-armed sleep waits for changed: it parks"
    assert ctx._held_dependencies_seen == {**old, paused: "newer-pause", unknown: "basis-unreadable"}, \
        "the older view's other holds count; what the newer view showed stays"
    assert "early_return" not in json.loads(readers._wait_for_tasks(ctx, list(rows), timeout_sec=0)), \
        "the older wait itself now waits"
    _restop(rows, paused, identity="newest-pause")
    assert _park_refresh(ctx) == [(paused, "newest-pause")], "a stop newer than every view is shown before the park"
    assert _rearm(ctx, [paused, unknown]) == "sleep_armed" and _park_refresh(ctx) is None, \
        "the same request then parks"


def test_a_recovery_to_the_stop_an_older_view_showed_is_shown_after_that_view_is_read_whole(tmp_path, monkeypatch):
    """The unreadable basis reads again, back at the very identity the older view showed: after the newer view
    that is new information, and reading the older source whole later does not make it seen."""
    ctx, rows, old, full, ref = _older_then_newer(tmp_path, monkeypatch)
    unknown = list(rows)[1]
    _read_whole(ctx, ref, full)
    _restop(rows, unknown, resume_basis=None, identity=old[unknown])
    assert _park_refresh(ctx) == [(unknown, old[unknown])], "the recovered basis is shown once before the park"
    assert _rearm(ctx, list(rows)[:2]) == "sleep_armed" and _park_refresh(ctx) is None


def test_a_restored_actor_keeps_each_hold_record_at_its_place_so_an_older_source_read_there_keeps_the_newer_stop(
        tmp_path, monkeypatch):
    """The places ride the continuation beside what was shown and what waits to be read: a new process that reads
    the older source whole keeps what the newer view showed, and the same wait arms and parks."""
    from ouroboros import delegate_output, owner_wait
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    ctx, rows, old, full, ref = _older_then_newer(tmp_path, monkeypatch)
    paused, unknown = list(rows)[:2]
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, [], {}, {}, 1, [], set())))
    fresh = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root-1")
    fresh.task_attempt, fresh.root_task_id, fresh.budget_drive_root, fresh.task_metadata = 1, "root-1", tmp_path, {}
    setattr(fresh, _TOOL_RESULT_ATTR, None)
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=fresh), state, [], {}, {}, set())
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})  # a new worker: nothing was delivered to it yet
    _read_whole(fresh, ref, full)
    assert fresh._held_dependency_sources == {}
    assert fresh._held_dependencies_seen == {**old, paused: "newer-pause", unknown: "basis-unreadable"}
    assert _rearm(fresh, [paused, unknown]) == "sleep_armed" and _park_refresh(fresh) is None


def test_a_shortened_hold_view_newer_than_a_tasks_record_replaces_it_once_read_whole(tmp_path, monkeypatch):
    """The same order the other way: a child shown whole, then paused anew and shown again by a newer wait that
    arrives shortened. Read whole, the newer view's stop is what was shown: its same wait waits, the sleep parks."""
    from ouroboros.tools import control_task_results as readers
    from tests.test_tool_result_delivery import _source_ref

    ctx, rows = _held_children(tmp_path, monkeypatch, 160)
    task_id = list(rows)[0]
    assert not _waited(ctx, rows, [task_id])[2].get("result_partial")
    assert ctx._held_dependencies_seen == {task_id: rows[task_id]["current_stop"]["identity"]}
    _restop(rows, task_id, cause="pause", scope="tree", identity="newer-pause")
    full, messages, shown = _waited(ctx, rows)
    assert shown.get("result_partial"), "the newer view arrived shortened"
    _read_whole(ctx, _source_ref(messages[0]["content"]), full)
    assert ctx._held_dependencies_seen == _identities(rows), "the newer view replaced the older record"
    assert "early_return" not in json.loads(readers._wait_for_tasks(ctx, list(rows), timeout_sec=0))
    assert _rearm(ctx, [task_id]) == "sleep_armed" and _park_refresh(ctx) is None


_SLEEP_FRAME = {"window": 8_000, "reserve": 2_000}  # Astra's R4 probe: a sleep selects at most 32 tasks


def _sleepable_children(tmp_path, monkeypatch, count):
    """``_held_children`` whose root may request its sleep (``await_messages``), as in Astra's R4 probe: an owner
    to sleep under, and each child a task this installation knows. Its holds return the same bytes in every test,
    so each starts from a fresh reading ledger, as a new worker does."""
    from ouroboros import delegate_output

    ctx, rows = _held_children(tmp_path, monkeypatch, count)
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})
    ctx.owner_wait_callback = lambda *_args: "owner_text"
    monkeypatch.setattr("ouroboros.task_results.load_task_result",
                        lambda _root, task_id, **_kwargs: rows.get(task_id, {}))
    return ctx, rows


def _awaited(ctx, rows, call_id):
    """The same wait as a warm sleep request over every child, delivered at the frame as call ``call_id``. A hold it
    returns names no elapsed time, so the same holds return the same text byte for byte: its text, the transcript
    and the source its shortened view advertised."""
    from ouroboros.tools import control_task_results as readers
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR
    from tests._tool_result_delivery_shared import measured_fit
    from tests.test_tool_result_delivery import _deliver, _row, _source_ref

    text = readers._await_messages(ctx, mode="warm", tasks=list(rows))
    assert json.loads(text)["reason"] == "dependency_held"
    messages, trace = _deliver(ctx, [_row(call_id, "await_messages", text, args={"mode": "warm", "tasks": list(rows)},
                                          meta={"tool_result_meta": dict(getattr(ctx, _TOOL_RESULT_ATTR).meta)})],
                               measured_fit(**_SLEEP_FRAME))
    assert trace["tool_calls"][0].get("result_partial"), "it arrived shortened"
    return text, messages, _source_ref(messages[0]["content"])


def _same_text_twice(ctx, rows):
    """Astra's R4 premise: two calls return byte-identical holds, each shortened and kept under its own call's
    retained path. That text, each call's transcript and each source."""
    (text, first, ref), (again, second, ref_again) = (_awaited(ctx, rows, call_id) for call_id in ("first", "second"))
    assert again == text and ref["sha256"] == ref_again["sha256"] and ref["path"] != ref_again["path"]
    return text, (first, second), (ref, ref_again)


def _read_span(ctx, ref, start, end):
    """``read_file`` pages of ``[start, end)`` of that source, each delivered whole."""
    for at in range(start, end, 20_000):
        assert not _read(ctx, _page(ref, at, min(at + 20_000, end)))[1].get("result_partial")


def _pending(ctx):
    """Each advertised file still waiting to be read whole, at the place its view arrived."""
    return {ref["path"]: ref["order"] for ref in (getattr(ctx, "_held_dependency_sources", None) or {}).values()}


def test_two_identical_shortened_hold_views_keep_two_sources_each_completed_by_what_was_delivered_of_it(
        tmp_path, monkeypatch):
    """Astra's R4 finding: the later of two identical views no longer replaces the earlier one's source. The earlier,
    its omitted middle read, shows its holds and the same request arms, while the later still waits at its own place:
    what was delivered of one file never covers the other, and the later, its own middle read, records there."""
    from ouroboros.tools import control_task_results as readers
    from tests.test_tool_result_delivery import _view_json

    ctx, rows = _sleepable_children(tmp_path, monkeypatch, 32)
    _text, transcripts, (first, second) = _same_text_twice(ctx, rows)
    omitted = [_view_json(messages[0]["content"])["omitted_ranges"] for messages in transcripts]
    for start, end in omitted[0]:
        _read_span(ctx, first, start, end)
    assert getattr(ctx, "_held_dependencies_seen", None) == _identities(rows), \
        "the earlier file, its head and tail shown and its middle read, shows its holds"
    assert set(ctx._held_dependencies_order.values()) == {1}, "at the place its view arrived"
    assert _pending(ctx) == {second["path"]: 2}, "the later file still waits to be read"
    assert json.loads(readers._await_messages(ctx, mode="warm", tasks=list(rows)))["reason"] == "sleep_armed", \
        "the same request arms"
    start, end = omitted[1][0]
    assert end - start > 1_000
    _read_span(ctx, second, start, start + 1_000)
    assert _pending(ctx) == {second["path"]: 2}, "one page of the later file does not complete it"
    for start, end in omitted[1]:
        _read_span(ctx, second, start, end)
    assert _pending(ctx) == {} and set(ctx._held_dependencies_order.values()) == {2}, \
        "read whole by what was delivered of it, the later file records at its own place"


def test_a_restored_actor_completes_each_of_two_identical_hold_sources_at_its_own_place_in_either_order(
        tmp_path, monkeypatch):
    """Both files and their places ride the continuation. A new process reads the later one whole first: it records
    at its own place while the earlier still waits; the earlier, read whole after it, completes and never replaces
    that newer record."""
    from ouroboros import delegate_output, owner_wait
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    ctx, rows = _sleepable_children(tmp_path, monkeypatch, 32)
    text, _transcripts, (first, second) = _same_text_twice(ctx, rows)
    assert _pending(ctx) == {first["path"]: 1, second["path"]: 2}, "two files, each at the place its view arrived"
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, [], {}, {}, 1, [], set())))
    assert state["held_dependency_sources"] == {_key(first): {**first, "order": 1},
                                                _key(second): {**second, "order": 2}}
    fresh = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root-1")
    fresh.task_attempt, fresh.root_task_id, fresh.budget_drive_root, fresh.task_metadata = 1, "root-1", tmp_path, {}
    setattr(fresh, _TOOL_RESULT_ATTR, None)
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=fresh), state, [], {}, {}, set())
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})  # a new worker: nothing was delivered to it yet
    _read_whole(fresh, second, text)
    assert fresh._held_dependencies_seen == _identities(rows) and set(fresh._held_dependencies_order.values()) == {2}
    assert _pending(fresh) == {first["path"]: 1}, "the earlier file still waits to be read"
    _read_whole(fresh, first, text)
    assert _pending(fresh) == {} and set(fresh._held_dependencies_order.values()) == {2}, \
        "the earlier completes and never replaces the later one's record"
    assert _rearm(fresh, list(rows)) == "sleep_armed" and _park_refresh(fresh) is None


def test_the_older_of_two_identical_hold_sources_read_whole_keeps_the_newer_stops_until_the_later_one_is_read(
        tmp_path, monkeypatch):
    """R4 beside R3's order. After a newer view, both children return to the very stops the older view showed, and a
    later request returns the older view's text byte for byte under its own call's path. The older file, read whole,
    counts at the place it arrived: the newer stops stay what was shown, so the same request shows the returned
    stops before it arms. The later file, read whole, records them at its own place; the same request then parks."""
    ctx, rows = _sleepable_children(tmp_path, monkeypatch, 32)
    old, (paused, unknown) = _identities(rows), list(rows)[:2]
    full, _messages, older = _awaited(ctx, rows, "older")
    _restop(rows, paused, cause="pause", scope="tree", identity="newer-pause")
    _restop(rows, unknown, resume_basis="unreadable", identity="basis-unreadable")
    assert not _waited(ctx, rows, [paused, unknown])[2].get("result_partial"), "the newer view arrived whole"
    _restop(rows, paused, cause="awaiting_selection", scope="member", identity=old[paused])
    _restop(rows, unknown, resume_basis=None, identity=old[unknown])
    again, _messages, later = _awaited(ctx, rows, "later")
    assert again == full and later["sha256"] == older["sha256"] and later["path"] != older["path"], \
        "the older view's very text, under the later call's own path"
    _read_whole(ctx, older, full)
    assert ctx._held_dependencies_seen == {**old, paused: "newer-pause", unknown: "basis-unreadable"}, \
        "the older file counts where it arrived: its other holds count, the newer stops stay what was shown"
    assert _pending(ctx) == {later["path"]: 3}, "the later file still waits to be read"
    assert _rearm(ctx, [paused, unknown]) == "dependency_held", "so the stops they returned to are shown first"
    _read_whole(ctx, later, again)
    assert _pending(ctx) == {} and ctx._held_dependencies_seen == old
    assert _rearm(ctx, [paused, unknown]) == "sleep_armed" and _park_refresh(ctx) is None, \
        "the same request then arms and parks"
