"""A held completion names its deciding branch in one diagnostic events row (issue #2339).

The rows come from the real loop (``run_llm_loop`` with a scripted model) and its real
finalization producers: the selection refusal, the loop's early owner/resume/left-review
branches, the delivery gates, the nudges and the result review. They are read back from the
canonical ``logs/events.jsonl`` through its existing readers (``query_data`` kind=events_log
and the task event replay), also after a split execution drive is removed. Observation decides
nothing: ordinary tool rounds, clean finishes, suppressed nudges and terminal stops write none,
and a hold whose row cannot be written plays out exactly as one that was written.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import shutil
import threading
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop
from ouroboros.tools.registry import ToolRegistry
from tests.test_presence_owner_contact import contact_runtime as _contact_runtime
from tests.test_presence_owner_contact import focus  # noqa: F401  (the fixture contact_runtime requests)
from tests.test_presence_owner_stop import owner_stop as _owner_stop  # the public Stop over a live turn

contact_runtime, owner_stop = _contact_runtime, _owner_stop


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finish(answer=None, *, action="finish", **kw):
    arguments = {"action": action, **kw}
    if answer is not None:
        arguments["answer"] = answer
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "complete", "type": "function",
        "function": {"name": "finish_task", "arguments": json.dumps(arguments)},
    }]}


def tool_call(name, call_id="work"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def act(name, **arguments):
    """One assistant round calling ``name`` with ``arguments``."""
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def write_round():
    """A round that changes the result (a reviewable write) and leaves it unexamined."""
    return act("write_file", path="report.md", content="Report")


def holds(root):
    """The completion_held rows of the events log under ``root`` (absent file: none)."""
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "task_checkpoint" and row.get("checkpoint_kind") == "completion_held"]


def facts(row):
    return (row["round"], row["origin"], row["reason"], row["subreason"], row["answer_sha256"])


def authored(calls):
    """Each request's assistant turns, tool results and host notices. The per-round facts and
    roster rows are left out: their sampled clock and the other runs they list differ between
    any two runs, as do a tool result's invocation ids and paths beside its content."""
    return [[(row.get("role"), row.get("content"), [c["function"]["name"] for c in row.get("tool_calls") or []])
             for row in call if not row.get("_host_context_kind")] for call in calls]


def runner(root, monkeypatch, *, task_id="author-task"):
    """A direct turn on ``root`` driven by scripted model responses (callables run at their call)."""
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx.is_direct_chat = True
    calls, incoming = [], queue.Queue()

    def run(responses, **kwargs):
        iterator = iter(responses)

        def respond(_llm, messages, *_a, **_kw):
            calls.append(json.loads(json.dumps(messages)))
            response = next(iterator)
            return (response() if callable(response) else response), 0.0

        monkeypatch.setattr(loop, "call_llm_with_retry", respond)
        return loop.run_llm_loop([{"role": "user", "content": "Please help"}], registry,
                                 SimpleNamespace(default_model=lambda: "test-model"), root / "logs",
                                 lambda _text, **_kw: None, incoming, task_id=task_id, drive_root=root, **kwargs)

    return SimpleNamespace(registry=registry, calls=calls, incoming=incoming, root=root, run=run)


@pytest.fixture
def turn(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    return runner(tmp_path, monkeypatch)


def hold_first_delivery(monkeypatch):
    """A result-review double that holds the first delivery once, naming no cause."""
    seen = []

    def review(_ctx, content, _trace, _hold):
        seen.append(content)
        return len(seen) == 1

    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", review)
    return seen


# --- Selection refusals: one row beside the historical trace record ------------------------------

@pytest.mark.parametrize("refusal", ["selection_unavailable", "unobserved_tool_results",
                                     "owner_input_changed"])
def test_a_refused_selection_is_one_typed_hold_beside_the_unchanged_trace_record(turn, refusal):
    retained = None
    if refusal == "selection_unavailable":
        first = finish(answer_sha256="0" * 64)
    elif refusal == "unobserved_tool_results":
        turn.registry.override_handler("chat_history", lambda _ctx, **kw: "read result")
        first = finish("Chosen before the result")
        first["tool_calls"].insert(0, tool_call("chat_history", "read"))
        retained = sha("Chosen before the result")  # the selection was retained, then refused
    elif refusal == "contradictory_completion_requests":
        first = finish("first")
        second = finish("second")["tool_calls"][0]
        second["id"] = "second"
        first["tool_calls"].append(second)
    else:
        def first():
            turn.registry._ctx._owner_directives.append({"source": "owner", "content": "New requirement"})
            return finish("Answer predating the owner input")

    text, usage, trace = turn.run([first, finish("Now informed")])

    assert text == "Now informed" and len(turn.calls) == 2
    [refused] = trace["completion_refusals"]
    assert set(refused) == {"reason", "request"} and refused["reason"].startswith(refusal)
    [row] = holds(turn.root)
    assert facts(row) == (1, "selection", "selection_refused", refusal, retained)
    assert row["task_id"] == "author-task" and row["execution_id"] == usage["execution_id"]
    assert "candidate_revision" in row and (row["candidate_revision"] is None) is (retained is None)
    assert "Now informed" not in json.dumps(row) and "Answer predating" not in json.dumps(row)


def test_a_refused_acceptance_subject_is_named_without_its_prose(turn):
    response = finish("The answer", acceptance_subject={"owner_source_sha256": "not-an-observed-source"})
    text, _usage, trace = turn.run([response, finish("Selected again")])
    assert text == "Selected again" and len(turn.calls) == 2
    [refused] = trace["completion_refusals"]
    [row] = holds(turn.root)
    assert facts(row) == (1, "selection", "selection_refused", "acceptance_subject", None)
    assert refused["reason"] not in json.dumps(row), "the refusal's host prose stays in the trace record"


# --- After an accepted selection, and before one -------------------------------------------------

@pytest.mark.parametrize("origin", ["selection", "prose"])
def test_a_real_nudge_hold_names_its_nudge_for_a_selection_and_for_prose(turn, origin):
    first = finish("Answer") if origin == "selection" else {"content": "Answer"}
    text, _usage, trace = turn.run([write_round(), first, finish("Answer")])

    assert text == "Answer" and len(turn.calls) == 3
    assert not trace.get("completion_refusals"), "a selection held after acceptance was not refused"
    [row] = holds(turn.root)
    assert facts(row) == (2, origin, "finalization_nudge", "pre_delivery_notice", sha("Answer"))
    assert row["candidate_revision"] == 1


def test_owner_words_after_an_accepted_selection_hold_it_in_the_loop_owner(turn):
    def answer():
        # Queued while the model call is out: drained after the batch consumed the selection.
        turn.incoming.put({"text": "Also cover the second case.", "msg_id": "owner-2"})
        return finish("Answer predating the owner words")

    text, _usage, trace = turn.run([answer, finish("Now considered")])

    assert text == "Now considered" and len(turn.calls) == 2
    assert not trace.get("completion_refusals"), "an accepted selection, then held, is no refusal"
    [row] = holds(turn.root)
    assert facts(row) == (1, "selection", "owner_input_changed", None, sha("Answer predating the owner words"))
    assert "second case" not in json.dumps(row)


def canonical_holds_now(root):
    """What the canonical events reader returns at this moment (``query_data`` kind=events_log)."""
    from ouroboros.tools.query_data import _query_data
    from ouroboros.tools.registry import ToolContext

    reply = json.loads(_query_data(ToolContext(repo_dir=root, drive_root=root, task_id="reader"), op="jsonl",
                                   kind="events_log",
                                   filter={"field": "checkpoint_kind", "op": "eq", "value": "completion_held"}))
    return [facts(row) for row in reply["data"]] if reply["status"] == "ok" else reply.get("code")


def left_review(turn, wait, park_owner=None):
    """The author's prose answers while a review it left is its intention (set during the model call)."""
    turn.registry._ctx.task_attempt = 1  # the agent sets it; the park's checkpoint names its attempt
    if park_owner is not None:
        turn.registry._ctx.owner_wait_callback = park_owner

    def waiting():
        turn.registry._ctx._left_review_wait = wait
        return {"content": "Waiting for the review I left."}

    return waiting


@pytest.mark.parametrize("case", ["custody_restored", "wake_notice", "settled"])
def test_prose_beside_a_left_review_is_one_hold_naming_its_case_recorded_before_any_wait(turn, monkeypatch, case):
    from ouroboros.review_pause import PROCESS_TOKEN

    wait = {"owner": PROCESS_TOKEN, "binding": "b-1", "surface": "change"}
    if case == "custody_restored":
        wait["owner"] = "another-process"
    elif case == "settled":
        wait.update(done=True, noticed=False, notice="The review settled.")
    elif case == "wake_notice":  # only the wake decision is scripted; its notice and the hold are the park's own
        monkeypatch.setattr("ouroboros.review_waiting.wake_reason", once("mail", ""))
    inside = []

    def park_owner(ctx, checkpoint):  # entered with the task parked on its review
        inside.append((checkpoint["review_binding"], canonical_holds_now(turn.root)))
        ctx._left_review_wait = None  # the wave's notice was read: this park is over
        return ""

    final = finish("Done.") if case == "wake_notice" else {"content": "Done."}  # an explicit completion ends it
    text, _usage, _trace = turn.run([left_review(turn, wait, park_owner), final])

    expected = (1, "prose", "left_review_wait", case, None)
    assert text == "Done." and len(turn.calls) == 2
    assert inside == ([("b-1", [expected])] if case == "parked" else []), "readable while the task is parked"
    assert [facts(row) for row in holds(turn.root)] == [expected], "one row per hold"


def test_a_park_its_owner_ends_keeps_its_recorded_hold(turn):
    from ouroboros.review_pause import PROCESS_TOKEN

    def park_owner(_ctx, _checkpoint):
        raise RuntimeError("the worker ended while parked")

    wait = {"owner": PROCESS_TOKEN, "binding": "b-1", "surface": "change"}
    with pytest.raises(RuntimeError, match="ended while parked"):
        turn.run([left_review(turn, wait, park_owner)])
    assert canonical_holds_now(turn.root) == [(1, "prose", "left_review_wait", "parked", None)]


@pytest.mark.parametrize("cause", ["owner_input_changed", "quiet_selection_lost"])
def test_a_resumed_candidate_hold_is_attributed_to_the_resume(tmp_path, cause):
    from ouroboros.loop_messages import owner_source_sha256
    from tests.test_acceptance_preparation_loop import _delivery_ctx

    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, limit_ctx = _delivery_ctx(tmp_path, trace)
    candidate = loop._replace_delivery_candidate(registry, limit_ctx, trace, "Saved answer", control="candidate")
    if cause == "owner_input_changed":
        candidate.owner_source_sha256 = owner_source_sha256(registry._ctx)
        registry._ctx._owner_directives = [{"source": "owner", "content": "A newer requirement"}]
    else:
        candidate.quiet = True  # a wake's silent selection, lost with its execution id

    assert loop._finalize_loop_candidate(None, limit_ctx, registry, lambda *_a, **_k: None,
                                         resume_candidate=True) is None

    [row] = holds(tmp_path)
    assert facts(row) == (1, "resume", cause, None, sha("Saved answer"))
    assert row["task_id"] == "parent1" and row["candidate_revision"] == candidate.revision


def test_a_later_hold_names_the_current_candidate_never_an_earlier_held_response(turn, monkeypatch):
    seen = hold_first_delivery(monkeypatch)
    interim, held_hash = "Interim, unselected", []

    def select_interim():
        held_hash.append(turn.registry._ctx._completion_held_sha256)  # what round 3 decided beside
        return finish(answer_sha256=sha(interim))

    text, _usage, _trace = turn.run([{"content": "Report"}, {"content": interim},
                                     finish(answer_sha256="0" * 64), select_interim])

    assert text == interim and seen == ["Report", interim] and len(turn.calls) == 4
    assert [facts(row) for row in holds(turn.root)] == [
        (1, "prose", "result_review", None, sha("Report")),
        (2, "prose", "selection_required", None, sha("Report")),
        (3, "selection", "selection_refused", "selection_unavailable", sha("Report")),
    ]
    assert held_hash == [sha(interim)], "the held response's hash was live, and no row borrowed it"


def once(value, otherwise):
    """A gate predicate that decides ``value`` on its first call and ``otherwise`` after it."""
    seen = []

    def gate(*_a, **_k):
        seen.append(1)
        return value if len(seen) == 1 else otherwise

    return gate


@pytest.mark.parametrize("reason", ["plan_gate", "focus_reply_refused",
                                    "admission_seal"])
def test_each_delivery_gate_names_itself_in_its_hold(turn, monkeypatch, reason):
    """The gate's decision is scripted once (the owner follow-up is a real admission drain); the
    branch it takes, its host rows and the row it writes are the production ones."""
    answer, origin, script = "Answer", "prose", [{"content": "Answer"}, finish("Answer")]
    if reason == "plan_gate":
        monkeypatch.setattr(loop, "_enforce_plan_gate", once(True, False))
    elif reason == "subagent_handoff":
        monkeypatch.setattr(loop, "_compute_subagent_handoff", once("A child finished.", ""))
    elif reason == "focus_reply_refused":
        monkeypatch.setattr("ouroboros.place_reply.prepare_focus_reply", once((False, "[SYSTEM NOTICE] Not here."), (True, "")))
    elif reason == "admission_seal":  # a held stop is selected again only by a later model round
        answer, origin = "Partial", "selection"
        script = [finish("Partial", action="stop", rationale="Unfinished")] * 2
        monkeypatch.setattr("ouroboros.loop_delivery._seal_admission_before_delivery", once(False, True))
    else:
        turn.registry._ctx.owner_message_admission_lock = threading.Lock()
        turn.registry._ctx.owner_message_admission_agent = SimpleNamespace(
            _accepting_owner_messages=True, _busy=False, _current_task_id="")

        def answered():  # the owner writes while this answer is out; its admission close drains them
            turn.incoming.put({"text": "One more thing.", "msg_id": "owner-3"})
            return {"content": "Answer"}

        script = [answered, {"content": "Answer"}]

    text, _usage, _trace = turn.run(script)

    assert text == answer and len(turn.calls) == len(script)
    assert [facts(row) for row in holds(turn.root)] == [(1, origin, reason, None, sha(answer))]


def test_a_refused_stop_reply_waits_for_a_later_model_round(turn, monkeypatch):
    """A stop whose reply the place's check refuses is held, never republished by the host (CAD group, 04.11
    20:57:04 → 20:57:34): nothing is offered again until the next model round selects the stop, and that round's
    identical stop is delivered."""
    offered = []

    def prepare(_ctx, _text, quiet=False):
        offered.append(len(turn.calls))  # how many model rounds had run when the reply was offered
        return (False, "[SYSTEM NOTICE] Not here.") if len(offered) == 1 else (True, "")

    monkeypatch.setattr("ouroboros.place_reply.prepare_focus_reply", prepare)
    text, _usage, _trace = turn.run([finish("Partial", action="stop", rationale="Unfinished")] * 2)
    assert text == "Partial" and len(turn.calls) == 2
    assert offered == [1, 2]
    assert [facts(row) for row in holds(turn.root)] == [(1, "selection", "focus_reply_refused", None, sha("Partial"))]


@pytest.mark.parametrize("standing", ["addressed", "context"])
def test_a_completed_stop_is_held_once_for_an_addressed_carried_row_and_stands_for_a_context_row(tmp_path, monkeypatch,
                                                                                                    standing):
    """The completed stop reads a carried row of a closing helper: addressed, the selection is cleared, the hold is
    observed once and the next round reads the row before the same stop is published; context, the stop stands.
    The owner corpus and its acknowledgement hash are unchanged either way (the row enters no corpus)."""
    from ouroboros.loop_messages import owner_source_sha256
    from tests.test_helper_owner_words_flow import _carried_line, _carried_mail, _late_row

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    turn = runner(tmp_path, monkeypatch, task_id="helper")
    turn.registry._ctx.is_direct_chat = False
    turn.registry._ctx.task_metadata = {"delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root"}
    turn.registry._ctx.owner_message_admission_lock = threading.Lock()
    turn.registry._ctx.owner_message_admission_agent = SimpleNamespace(_busy=True, _current_task_id="helper",
                                                                        _accepting_owner_messages=True)
    row, address = _late_row(tmp_path)
    before = []

    def stop():
        before.append((owner_source_sha256(turn.registry._ctx), json.dumps(turn.registry._ctx._owner_directives, default=str)))
        _carried_mail(tmp_path, "helper", row, address, went_to="" if standing == "addressed" else "sibling")
        return finish("Partial", action="stop", rationale="Unfinished")

    text, _usage, trace = turn.run([stop, finish("Partial", action="stop", rationale="Unfinished")])

    assert text == "Partial" and trace["task_completion"]["action"] == "stop"
    assert (owner_source_sha256(turn.registry._ctx), json.dumps(turn.registry._ctx._owner_directives, default=str)) == before[0]
    if standing == "addressed":
        assert len(turn.calls) == 2 and [facts(held) for held in holds(tmp_path)] == [
            (1, "selection", "owner_carried", None, sha("Partial"))]
        assert any(_carried_line(row, address) in str(m.get("content")) for m in turn.calls[1])
    else:
        assert len(turn.calls) == 1 and holds(tmp_path) == []


# --- Review answers settled since the mind's last read reach it before delivery (cec586e9) ------

REFUTATION = "o1: the parser already reads fenced JSON"


def review_settles(monkeypatch):
    """The review history producer as the loop calls it: one new index row per settled answer, never twice."""
    from ouroboros.review_history_view import REVIEW_CONTEXT_INDEX_KEY

    meta = {"family": "plan", "repo_root": "", "sha256": "decisions-1", "visible_sha256": "v1"}

    def updates(ctx, *, families=None, messages=(), selection=None):
        pending = families if families is not None else getattr(ctx, "_pending_review_context", {})
        if not pending or any(row.get(REVIEW_CONTEXT_INDEX_KEY) == meta for row in messages):
            return [], {}
        return [{"role": "user", "content": "[Current review decisions] " + REFUTATION, REVIEW_CONTEXT_INDEX_KEY: dict(meta)}], {}

    monkeypatch.setattr("ouroboros.review_history_view.review_context_updates", updates)


def settle_during(turn, response):
    """A council seat settles while this model call is out (the hook queues the author's update)."""
    def call():
        turn.registry._ctx._pending_review_context = {"plan": ""}
        return response
    return call


@pytest.mark.parametrize("origin", ["selection", "prose"])
def test_review_answers_settled_during_the_final_call_are_read_before_delivery(turn, monkeypatch, origin):
    review_settles(monkeypatch)
    first = finish("Answer before the review") if origin == "selection" else {"content": "Answer before the review"}

    text, _usage, _trace = turn.run([settle_during(turn, first), finish("Answer after the review")])

    assert text == "Answer after the review" and len(turn.calls) == 2
    assert REFUTATION in json.dumps(turn.calls[1]), "the second call reads the settled answer"
    assert [facts(row) for row in holds(turn.root)] == [(1, origin, "review_update", None, sha("Answer before the review"))]


@pytest.mark.parametrize("origin", ["selection", "prose"])
def test_review_answers_the_mind_already_read_hold_nothing(turn, monkeypatch, origin):
    review_settles(monkeypatch)
    turn.registry.override_handler("chat_history", lambda _ctx, **kw: "history")
    work = {"content": None, "tool_calls": [tool_call("chat_history")]}
    final = finish("Answer") if origin == "selection" else {"content": "Answer"}

    text, _usage, _trace = turn.run([settle_during(turn, work), final])

    assert text == "Answer" and len(turn.calls) == 2
    assert REFUTATION in json.dumps(turn.calls[1]), "the rows reached the call that wrote the final"
    assert holds(turn.root) == []


@pytest.mark.parametrize("case", ["author_stop", "owner_hurry", "plan_gate_first"])
def test_stop_and_hurry_deliver_and_the_plan_gate_keeps_its_place(turn, monkeypatch, case):
    review_settles(monkeypatch)
    answer = finish("Partial", action="stop", rationale="Unfinished") if case == "author_stop" else finish("Answer")

    def call():  # the seat settles, and the owner hurries, while the final call is out
        turn.registry._ctx._pending_review_context = {"plan": ""}
        if case == "owner_hurry":
            turn.registry._ctx._owner_hurry_latch = {"mailbox_id": "hurry-1"}
        return answer

    script = [call]
    if case == "plan_gate_first":
        monkeypatch.setattr(loop, "_enforce_plan_gate", once(True, False))
        script.append(finish("Answer"))

    text, _usage, _trace = turn.run(script)

    assert len(turn.calls) == len(script) and text in {"Partial", "Answer"}
    expected = [(1, "selection", "plan_gate", None, sha("Answer"))] if case == "plan_gate_first" else []
    assert [facts(row) for row in holds(turn.root)] == expected


# --- What is not a hold -------------------------------------------------------------------------

@pytest.mark.parametrize("script", ["tool_round_then_plain_final", "author_stop", "suppressed_nanny_nudge"])
def test_ordinary_work_finishes_and_suppressed_nudges_write_no_hold(turn, script):
    responses = {
        "tool_round_then_plain_final": [{"content": None, "tool_calls": [tool_call("chat_history")]},
                                        {"content": "Done."}],
        "voluntary_finish": [finish("Hi")],
        "plain_final": [{"content": "Plain answer."}],
        "author_stop": [finish("Partial", action="stop", rationale="Not finished.")],
        "receipt_sibling": [finish("The asked fact")],
        "suppressed_nanny_nudge": [{"content": "Done by my own calls."}],
    }[script]
    if script == "tool_round_then_plain_final":
        turn.registry.override_handler("chat_history", lambda _ctx, **kw: "history")
    if script == "receipt_sibling":
        turn.registry.override_handler("update_scratchpad", lambda _ctx, **kw: "written")
        responses[0]["tool_calls"].insert(0, {"id": "note", "type": "function", "function": {
            "name": "update_scratchpad", "arguments": json.dumps({"content": "Next: the docs"})}})
    if script == "suppressed_nanny_nudge":  # a nanny finishing by its own calls: no open run, no child, no effect
        turn.registry._ctx._nanny_route_dispatched = True
        turn.registry._ctx.task_contract = {"disabled_tools": ["delegate_start"]}

    text, _usage, trace = turn.run(responses)

    assert len(turn.calls) == len(responses) and text
    assert not trace.get("completion_refusals")
    assert holds(turn.root) == []


def test_an_owner_hard_stop_of_a_direct_turn_is_a_terminal_not_a_hold(turn):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
    from ouroboros.tools.control_runtime import _finish_task
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    def stop(ctx, **kw):
        result = _finish_task(ctx, **kw)
        write_owner_message(turn.root, REASON_OWNER_STOPPED_DIRECT_TURN, "author-task", kind=KIND_FINALIZE_NOW)
        return result

    turn.registry.override_handler("finish_task", stop)
    text, usage, _trace = turn.run([finish("Selected partial", action="stop", rationale="Unfinished")])
    assert text == "Selected partial" and usage["reason_code"] == "owner_requested_finalization"
    assert holds(turn.root) == []


def test_a_presence_answer_met_by_stop_now_is_a_terminal_not_a_hold(owner_stop, monkeypatch):
    """A real Place focus turn: its final answer meets the owner's Stop now at the closing admission."""
    from ouroboros.presence_runner import PresenceTurnError
    from tests.test_presence_contact_wait import start
    from tests.test_presence_owner_stop import LATE_ANSWER, _scripted

    r, press = owner_stop
    stops = {}
    h, generated = _scripted(monkeypatch, r, lambda task_id: stops.setdefault("stop", press(task_id, cascade=False)),
                             {"content": LATE_ANSWER, "tool_calls": []})
    with pytest.raises(PresenceTurnError) as stopped:
        start(r, h).result.result(timeout=90)

    assert stopped.value.code == "presence_turn_cancelled" and stops["stop"].status_code == 202 and generated == [1]
    task_id = h.agents[0].tools._ctx.task_id
    rows = [json.loads(line) for path in r.root.rglob("events.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]
    assert any(row.get("task_id") == task_id for row in rows), "the scanned logs hold this turn's own rows"
    assert [row for row in rows if row.get("checkpoint_kind") == "completion_held"] == []


# --- The canonical plane outlives a split execution drive ---------------------------------------

def test_a_split_child_hold_survives_its_drive_removal_in_both_canonical_readers(tmp_path, monkeypatch):
    from ouroboros.gateway.task_events import iter_task_events
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.query_data import _query_data
    from ouroboros.tools.registry import ToolContext
    from supervisor.log_addressing import TurnEventQueue

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    canonical = tmp_path / "data"
    child = canonical / "state" / "headless_tasks" / "child-task" / "data"
    (child / "logs").mkdir(parents=True)
    lineage = {"parent_task_id": "root-task", "root_task_id": "root-task", "delegation_role": "subagent"}
    write_task_result(canonical, "root-task", "running")
    write_task_result(canonical, "child-task", "running", child_drive_root=str(child), **lineage)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=child)
    registry._ctx.budget_drive_root = str(canonical)
    registry._ctx.task_metadata = {"budget_drive_root": str(canonical), **lineage}
    live = queue.Queue()
    frames = TurnEventQueue(live, "child-task", None, lineage=lineage, direct_chat=False, telemetry_only=True)
    responses = iter([finish(answer_sha256="0" * 64), finish("Child answer")])
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (next(responses), 0.0))

    text, _usage, _trace = loop.run_llm_loop(
        [{"role": "user", "content": "Do the child work"}], registry, SimpleNamespace(default_model=lambda: "m"),
        child / "logs", lambda _text, **_kw: None, queue.Queue(), task_id="child-task", drive_root=child,
        event_queue=frames)

    assert text == "Child answer"
    assert holds(child) == [], "one row in one plane: the execution drive keeps no copy"
    [written] = holds(canonical)
    assert {key: written[key] for key in ("task_id", *lineage)} == {"task_id": "child-task", **lineage}
    assert facts(written) == (1, "selection", "selection_refused", "selection_unavailable", None)
    shutil.rmtree(child)
    assert not child.exists()

    reader = ToolContext(repo_dir=tmp_path, drive_root=canonical, task_id="root-task")
    reply = json.loads(_query_data(reader, op="jsonl", kind="events_log",
                                   filter={"field": "checkpoint_kind", "op": "eq", "value": "completion_held"}))
    assert reply["status"] == "ok" and reply["data"] == [written]
    for task in ("child-task", "root-task"):
        replayed = [event["data"] for event in iter_task_events(canonical, task)
                    if event["type"] == "task_checkpoint" and event["data"].get("checkpoint_kind") == "completion_held"]
        assert replayed == [written], task


# --- Observation decides nothing ----------------------------------------------------------------

@pytest.mark.parametrize("failure", ["emitter_raises", "events_log_unwritable"])
def test_an_unrecorded_hold_plays_out_exactly_as_a_recorded_one(tmp_path, monkeypatch, caplog, failure):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    # The pre-delivery notice holds the draft (a written result nothing examined); interim prose is then no
    # selection; a stale selector is refused.
    script = [write_round(), {"content": "Draft"}, {"content": "Interim."}, finish(answer_sha256="0" * 64),
              finish("Final")]
    recorded_root, failing_root = tmp_path / "recorded", tmp_path / "failing"

    def written(_ctx, *, _resolved_binding=None, **_kwargs):  # the same tool row in both runs
        return "written"

    recorded = runner(recorded_root, monkeypatch)
    recorded.registry.override_handler("write_file", written)
    expected = recorded.run(list(script))
    assert [row["reason"] for row in holds(recorded_root)] == [
        "finalization_nudge", "selection_required", "selection_refused"]

    if failure == "emitter_raises":
        real = loop._emit_checkpoint_event

        def emit(event_queue, task_id, drive_logs, data):
            if data.get("checkpoint_kind") == "completion_held":
                raise RuntimeError("observation sink down")
            return real(event_queue, task_id, drive_logs, data)

        monkeypatch.setattr(loop, "_emit_checkpoint_event", emit)
    else:
        (failing_root / "logs" / "events.jsonl").mkdir(parents=True)  # no row can be appended there
    failing = runner(failing_root, monkeypatch)
    failing.registry.override_handler("write_file", written)
    with caplog.at_level(logging.WARNING):
        observed = failing.run(list(script))

    assert observed[0] == expected[0] == "Final" and len(failing.calls) == len(recorded.calls) == 5
    assert observed[2]["completion_refusals"] == expected[2]["completion_refusals"]
    assert authored(failing.calls) == authored(recorded.calls), "the same model requests, round for round"
    disclosed = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    if failure == "emitter_raises":
        assert disclosed.count("Completion hold of author-task was not observed") == 3
        assert holds(failing_root) == []
    else:
        assert any("events.jsonl" in message for message in disclosed), "the failed append is logged"
        assert (failing_root / "logs" / "events.jsonl").is_dir(), "and nothing was written elsewhere instead"


# --- The producers name their causes -------------------------------------------------------------

def _nudge_case(case, tmp_path, monkeypatch):
    ctx = SimpleNamespace(drive_root=tmp_path, task_contract={}, task_metadata={})
    trace = {"reasoning_notes": [], "tool_calls": []}
    content = "done"
    monkeypatch.setattr(loop, "_skill_finalization_message", lambda *_a, **_k: "")
    if case == "authoring_handover":
        trace["tool_calls"] = [{"tool": "read_file", "status": "ok"}]
        ctx._authoring_handover = {"tool_calls_at_handover": 1, "from_model": "a", "to_model": "b"}
    elif case == "skill_not_finalized":
        monkeypatch.setattr(loop, "_skill_finalization_message", lambda *_a, **_k: "⚠️ SKILL_NOT_FINALIZED: x")
    elif case == "pre_delivery_notice":  # a written result nothing examined since
        trace["tool_calls"] = [{"tool": "write_file", "status": "ok", "args": {"path": "report.md"}}]
    elif case == "final_answer_marker":
        ctx.task_contract = {"answer_protocol": "final_answer_line"}
        trace["tool_calls"] = [{"tool": "read_file", "status": "ok"}]
    return SimpleNamespace(_ctx=ctx), trace, content


@pytest.mark.parametrize("case", ["pre_delivery_notice"])
def test_each_finalization_nudge_names_its_own_hold_cause(tmp_path, monkeypatch, case):
    tools, trace, content = _nudge_case(case, tmp_path, monkeypatch)
    assert loop._maybe_inject_finalization_nudges(tools, tmp_path, "t", trace, content, [], lambda *_: None) is True
    assert getattr(tools._ctx, "_completion_hold_cause", None) == case
    from ouroboros.loop_delivery import _hold_cause

    assert _hold_cause(tools._ctx) == case and _hold_cause(tools._ctx) == "", "read once"


@pytest.mark.parametrize("after_write", ["nothing", "declared", "red_check", "result_review"])
def test_an_unexamined_result_is_noticed_once_and_an_examined_one_is_not(turn, monkeypatch, after_write):
    """Audit 11.11, item 18: a written result nothing examined since its last change is noticed once at the first
    finish (cause ``pre_delivery_notice``) and the second finish is accepted. An executed check, a red one too, or a
    result review asked for after the last change examined it: the door is silent. A declaration that nothing is
    machine-checkable examines nothing: the notice still comes, and the record keeps ``receipt_absent``."""
    import sys

    from ouroboros.outcomes import apply_receipt_absent_flag, derive_loop_outcome

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))  # the check runs, unjudged
    examine = {
        "nothing": [],
        "declared": [act("verify_and_record", contract_kind="no_visible_machine_contract", check="read it through")],
        "red_check": [act("verify_and_record", contract_kind="explicit_command",
                          check=[sys.executable, "-c", "raise SystemExit(1)"])],
        "result_review": [act("review_change", subject="result")],
    }[after_write]
    if after_write == "result_review":  # the request is what the record reads; its panel does not run here
        turn.registry.override_handler("review_change", lambda _ctx, **_kw: json.dumps({"status": "requested"}))
    noticed = after_write in {"nothing", "declared"}
    script = [write_round(), *examine, finish("Answer"), finish("Answer")]

    text, usage, trace = turn.run(script)

    assert text == "Answer" and len(turn.calls) == len(script) - (0 if noticed else 1)
    assert [facts(row) for row in holds(turn.root)] == (
        [(len(script) - 1, "selection", "finalization_nudge", "pre_delivery_notice", sha("Answer"))] if noticed else [])
    outcome = derive_loop_outcome(text, usage, trace)
    apply_receipt_absent_flag(outcome, trace, turn.root, "author-task")
    warnings = outcome["outcome_axes"]["objective"].get("warnings") or []
    assert ("receipt_absent" in warnings) is noticed, warnings
    assert ("verification_red_unreconciled" in warnings) is (after_write == "red_check"), warnings


def test_only_a_current_bound_result_review_examines_the_result():
    """A bound result review speaks for the subject it was bound to. The next delivery candidate supersedes it (a
    change after it is unexamined again), a run of an earlier version stands beside the delivered bytes as that
    version's, and a $0 refusal examined nothing; the exact subject delivered again is current again."""
    from ouroboros.outcomes import result_unexamined
    from ouroboros.tools.review_result import _bind_review_run, supersede_result_runs

    def record(aggregate="PASS"):
        return {"record_id": "r1", "verdict": {"aggregate": aggregate}, "subject": {"diff_sha": "s1"}}

    write = {"tool": "write_file", "status": "ok", "args": {"path": "report.md"}}
    trace = {"tool_calls": [dict(write)], "review_runs": []}
    run = _bind_review_run(trace, record(), "s1", "advisory")
    assert result_unexamined(trace) is False, "bound to this result"
    trace["tool_calls"].append(dict(write))
    supersede_result_runs(trace, "candidate-2")  # the next door's candidate, before its notice
    assert result_unexamined(trace) is True, "the result changed after the binding"
    _bind_review_run(trace, record(), "s2", "advisory", earlier_revision=True)
    assert run["earlier_revision"] and result_unexamined(trace) is True, "an earlier version's review"
    _bind_review_run(trace, record(), "s1", "advisory")
    assert result_unexamined(trace) is False, "the exact subject it reviewed is delivered again"
    trace["review_runs"] = [{"surface": "result", "aggregate": "NOT_DISPATCHED"}]
    assert result_unexamined(trace) is True, "a refusal dispatched no reviewer"


def test_the_notice_shown_before_a_continuation_is_not_shown_again_after_it(tmp_path, monkeypatch):
    """The notice comes once for the task, not once per process: a continuation (an owner wait, a restart, a budget
    pause) saves the shown notice and the handed-off child results with the other delivery facts, so the resumed
    execution shows neither again. A new turn on a reused context starts its own notice."""
    from ouroboros import owner_wait
    from ouroboros.loop import _reset_turn_state
    from ouroboros.loop_nudges import _claim_pre_delivery_notice
    from tests._budget_pause_exact_helpers import _loop_ctx

    monkeypatch.setattr("ouroboros.config.get_task_review_mode", lambda: "auto")
    trace = {"reasoning_notes": [], "tool_calls": [{"tool": "write_file", "status": "ok", "args": {"path": "r.md"}}]}
    ctx, limit = _loop_ctx(tmp_path, "t")
    assert _claim_pre_delivery_notice(SimpleNamespace(_ctx=ctx), tmp_path, "t", trace, []), "the first door shows it"
    ctx._subagent_handoff_shown = ["child-1:abc"]
    saved = json.loads(json.dumps(owner_wait.continuation_state(ctx, limit.messages, {}, {}, 3, [], set()), default=str))
    unsaved, _ = _loop_ctx(tmp_path, "t")
    assert _claim_pre_delivery_notice(SimpleNamespace(_ctx=unsaved), tmp_path, "t", trace, []), "nothing carried it"
    resumed, _ = _loop_ctx(tmp_path, "t")
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=resumed), saved, [], {}, {}, set())
    assert _claim_pre_delivery_notice(SimpleNamespace(_ctx=resumed), tmp_path, "t", trace, []) == ""
    assert resumed._subagent_handoff_shown == ["child-1:abc"], "the handed-off results stay handed off"
    _reset_turn_state(resumed)
    assert _claim_pre_delivery_notice(SimpleNamespace(_ctx=resumed), tmp_path, "t", trace, []), "a new turn's answer"


@pytest.mark.parametrize("mode", ["auto", "off", "auto_without_review_tool"])
def test_the_notice_offers_a_result_review_only_where_this_actor_can_ask_for_one(tmp_path, monkeypatch, mode):
    """Task review Off refuses every result review (``review_result.refuse_when_off``), and an actor whose tools
    lack ``review_change`` cannot call it: there the notice offers the check and the honest delivery only; under
    Auto, with the tool, it also offers the review call."""
    from ouroboros.loop_nudges import _pre_delivery_notice

    tools, trace, _content = _nudge_case("pre_delivery_notice", tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.config.get_task_review_mode", lambda: mode.split("_")[0])
    if mode == "auto_without_review_tool":
        tools.available_tools = lambda: ["write_file", "verify_and_record", "finish_task"]
    notice = _pre_delivery_notice(tools, tmp_path, "t", trace, [])
    assert notice.startswith("Before delivery: no recorded check (verify_and_record) or result review has examined")
    assert ('review_change with subject="result"' in notice) is (mode == "auto")
    assert ("off on this install" in notice) is (mode == "off")
    assert "verify_and_record runs the check" in notice and "or deliver it as it is" in notice


@pytest.mark.parametrize("cause", ["findings", "pass"])
def test_each_result_review_hold_names_its_own_cause(tmp_path, monkeypatch, cause):
    import ouroboros.review_substrate as substrate
    import ouroboros.tools.review_change as review_change_mod
    from ouroboros.tools.review_result import result_review_at_delivery
    from tests.test_result_review import CRITICAL, _floor_ctx, _produced_answer_protocol, _seam

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends, shown, trace = [], [], {"tool_calls": []}
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot",
                        _seam(sends, {} if cause in {"earlier_request", "pass"} else {"t1": CRITICAL}))
    if cause == "pending":
        monkeypatch.setattr(review_change_mod, "run_review_change",
                            lambda ctx, **kwargs: {"state": "pending", "record_id": "rl-running"})
    if cause == "earlier_request":
        assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True  # its PASS, first
        ctx._completion_hold_cause = ""  # the delivery owner reads each cause once
        assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False, "then it delivers"
        assert ctx._completion_hold_cause == "", "a delivery names no hold cause"
        ctx._owner_directives = [*(getattr(ctx, "_owner_directives", None) or []),
                                 {"source": "owner_mailbox", "content": "Give it in kilometres.", "msg_id": "m-2"}]
    answer = "FINAL ANSWER: 2"
    ctx._completion_hold_cause = ""

    assert result_review_at_delivery(ctx, answer, trace, shown.append) is True
    assert ctx._completion_hold_cause == cause


def test_a_real_result_review_hold_reaches_the_row_through_the_real_loop(tmp_path, monkeypatch):
    import ouroboros.review_substrate as substrate
    from tests.test_result_review import CRITICAL, _floor_ctx, _produced_answer_protocol, _seam

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    answer = "FINAL ANSWER: 2"
    responses = iter([{"content": answer}, finish(answer)])
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (next(responses), 0.0))

    text, _usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "How many?"}], registry, SimpleNamespace(default_model=lambda: "m"),
        ctx.drive_root / "logs", lambda _text, **_kw: None, queue.Queue(), task_id="task-project",
        drive_root=ctx.drive_root)

    assert text == answer and len(sends) == 3, "one host review; the informed redelivery buys none"
    [row] = holds(ctx.drive_root)
    assert facts(row) == (1, "prose", "result_review", "findings", sha(answer))
    assert getattr(ctx, "_completion_hold_cause", "") == "", "the delivery owner read the cause once"
