"""The local lane never cuts a request: the serving instance's exact count decides, or the server does.

When the instance counts exactly, a candidate is measured whole: a fit is sent as prepared; a
shortfall is the typed size refusal with the refused candidate's own facts, before any send.
Without exact support, and in Low and Max, the whole prepared candidate is sent too (the
server's own refusal is the same typed refusal): no markdown compactor runs on any path.
With no number of the owner's the count is a request the built-in server answers one at a time,
so it waits for the server's one slot before its connection opens; a count whose answer is only
advice declines a busy slot instead and leaves its estimate in force.
"""
import contextlib
import copy
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import config, llm_local, local_model, model_concurrency as mc, send_clock as sc, task_pacing, usage_accounting as ua
from ouroboros.context_budget import LocalContextTooLargeError
from ouroboros.llm import LLMClient
from ouroboros.local_model_server import input_fingerprint
from tests.test_physical_candidate_capture import _Response, _rows, _scope, data_root as _data_root

data_root = _data_root
MESSAGES = [{"role": "system", "content": [
    {"type": "text", "text": "## BIBLE.md\n\n" + "keep " * 200 + "\n\n## Other governance\n\n" + "cut " * 20_000},
    {"type": "text", "text": "## Identity\n\n" + "me " * 100},
    {"type": "text", "text": "## Runtime context\n\n" + "now " * 100},
]}, {"role": "user", "content": "the task"}]


def _nano(window):
    return ua.PhysicalAttemptContext(
        profile="owner_nano", rendered_mode="nano", measurement_basis="cold_estimate", route_fp="local", round_id="x:round:1",
        target_total_tokens=85_000, capacity_total_tokens=window, context_target_miss=False, automatic_pass_used=False)


@pytest.fixture
def lane(data_root, monkeypatch):
    """A fake owned server: ``count(payload)`` is its exact tokenizer; ``sent`` what reached it."""
    state = {"window": 16_384, "pid": 123, "count": lambda payload: 5_000, "measured": [], "sent": []}
    monkeypatch.setenv("LOCAL_MODEL_PORT", "1")  # the fake owns the endpoint even the bound lookahead names
    evidence = lambda **kwargs: {"context_window": state["window"], "confirmed": True,  # noqa: E731
                                 "process_id": state["pid"], "source": "published_server_arguments", "port": 1}

    def measure(payload, **kwargs):
        state["measured"].append(copy.deepcopy(payload))
        count = state["count"](payload)
        if count is None:
            return {"supported": False, "input_is_exact": False, "reason": "selected_formatter_not_measurable"}
        return {"supported": True, "input_is_exact": True, "input_tokens": count, "context_window": state["window"],
                "process_id": 123, "native_input_sha256": input_fingerprint(payload),
                "output_limit_enforced": True, "reasoning_included_in_limit": True}

    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        serving_context_evidence=evidence, measure_prepared_input=measure))
    client = LLMClient(api_key="unused")
    monkeypatch.setattr(client, "_get_local_client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **candidate: state["sent"].append(copy.deepcopy(candidate)) or _Response(text="local")))))
    state["client"], state["root"] = client, data_root
    return state


def _send(lane, physical, clock=None):
    with ua.usage_scope(_scope(lane["root"])), ua.bind_physical_attempt_context(physical), (
            clock.bound() if clock is not None else sc.MainSendClock(None).bound()):
        return lane["client"]._chat_local(copy.deepcopy(MESSAGES), None, max_tokens=65_536, tool_choice="auto")


def _prepared(lane):
    """The whole prepared candidate: every section's text, the same bytes on every path."""
    return lane["client"]._build_local_candidate(copy.deepcopy(MESSAGES), None, 65_536, "auto")[1]["messages"]


def test_an_exact_fit_is_sent_whole_with_one_clock_line(lane):
    clock = sc.MainSendClock(sc.SendClockPolicy("UTC"))
    message, _usage = _send(lane, _nano(16_384), clock)
    [sent] = lane["sent"]
    assert message["content"] == "local" and len(lane["measured"]) == 1
    assert "cut cut" in sent["messages"][0]["content"] and "Compacted" not in sent["messages"][0]["content"]
    assert sent["messages"][:-1] == _prepared(lane)
    assert sent["max_tokens"] == 4_096  # the quarter-window ceiling: never raised, the reply floor is the ceiling
    assert len(clock.notes) == 1 and sent["messages"][-1] == {"role": "user", "content": clock.notes[0]}
    assert lane["measured"][0]["messages"][-1] == sent["messages"][-1]  # the whole candidate carried the same line
    assert [row["state"] for row in _rows(lane["root"])] == ["settled"]  # the store keeps one row per attempt


def test_a_262k_window_sends_an_80k_exact_input_with_the_whole_ceiling(lane):
    lane["window"], lane["count"] = 262_144, lambda payload: 80_000
    _send(lane, _nano(262_144))
    [sent] = lane["sent"]
    assert sent["max_tokens"] == 65_536 and "cut cut" in sent["messages"][0]["content"]


def test_an_exact_shortfall_is_the_typed_refusal_with_its_own_candidate_facts(lane):
    """Both directions with the fit above: the same candidate, counted past the window, is refused
    before any send with its own facts; nothing is cut, re-measured or resent."""
    lane["count"] = lambda payload: 15_000
    with pytest.raises(LocalContextTooLargeError) as caught:
        _send(lane, _nano(16_384))
    facts = caught.value.refused_candidate
    assert not lane["sent"] and len(lane["measured"]) == 1
    assert not _rows(lane["root"])  # nothing reserved, nothing sent
    assert facts["provider"] == "local" and facts["model"] == "local-model" and facts["max_completion_tokens"] == 4_096
    assert facts["physical_context"]["rendered_mode"] == "nano" and facts["input_tokens"] == 15_000
    assert len(facts["candidate_raw_sha256"]) == 64 and facts["candidate_context_size_bytes"] > 0
    assert "below the floor" in str(caught.value)


def test_without_exact_support_the_whole_candidate_is_sent_as_prepared(lane):
    lane["count"] = lambda payload: None
    _send(lane, _nano(16_384))
    [sent] = lane["sent"]
    assert len(lane["measured"]) == 1 and sent["messages"] == _prepared(lane)  # no clock policy: no clock line
    assert "cut cut" in sent["messages"][0]["content"] and sent["max_tokens"] == 4_096


@pytest.mark.parametrize("physical", [None, ua.PhysicalAttemptContext(
    profile="owner_max", rendered_mode="max", measurement_basis="cold_estimate", route_fp="local", round_id="x:round:1",
    target_total_tokens=None, capacity_total_tokens=16_384, context_target_miss=False, automatic_pass_used=False)])
def test_low_and_max_send_the_whole_candidate_too(lane, physical):
    _send(lane, physical)
    [sent] = lane["sent"]
    assert len(lane["measured"]) == 1 and sent["messages"] == _prepared(lane)  # measured once, sent whole
    assert sent["max_tokens"] == 4_096


def test_a_changed_serving_instance_is_still_a_preparation_failure(lane):
    def count(payload):
        lane["pid"] = 456  # the instance that measured is gone before dispatch
        return 5_000

    lane["count"] = count
    with pytest.raises(ua.PhysicalAttemptPreparationFailed, match="instance changed"):
        _send(lane, _nano(16_384))
    assert not lane["sent"]


def test_unbound_helper_measures_before_rejecting_an_overestimated_input(lane, monkeypatch):
    from ouroboros import context_fit
    monkeypatch.setattr(context_fit, "bounded_prompt_tokens_for_payload", lambda *a, **kw: 90014)
    lane["window"], lane["count"] = 80000, lambda payload: 50000
    _send(lane, None)
    assert len(lane["measured"]) == 1 and len(lane["sent"]) == 1
    assert lane["sent"][0]["max_tokens"] == 20000
    assert lane["sent"][0]["messages"] == _prepared(lane)


def test_unbound_helper_rejects_an_exact_overflow_after_measurement(lane):
    lane["window"], lane["count"] = 80000, lambda payload: 79000
    with pytest.raises(LocalContextTooLargeError) as caught:
        _send(lane, None)
    assert len(lane["measured"]) == 1 and not lane["sent"]
    assert caught.value.refused_candidate["input_tokens"] == 79000


# The built-in server answers one request at a time, and its count waits on the lock a generation holds.
BUILT_IN = {"provider": "local"}  # the slot of the lane's own route: the same model and endpoint as its target


def _queue_files(lane, suffix):
    return list((lane["root"] / "state" / "model_concurrency").glob(f"*/*.{suffix}"))


def _limit(lane, monkeypatch, number=None):
    """No number of the owner's (the shipped blank), or ``number`` through the environment; no document is saved."""
    monkeypatch.setattr(config, "SETTINGS_PATH", lane["root"] / "settings.json")
    if number is None:
        monkeypatch.delenv("OUROBOROS_MODEL_MAX_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", str(number))


def _watch(lane, monkeypatch):
    """The count and the send in the order they reach the fake server, each with the slots held at that moment."""
    events = []
    lane["count"] = lambda payload: events.append(("count", len(_queue_files(lane, "slot")))) or 5_000

    def create(**candidate):
        events.append(("send", len(_queue_files(lane, "slot"))))
        lane["sent"].append(copy.deepcopy(candidate))
        return _Response(text="local")

    monkeypatch.setattr(lane["client"], "_get_local_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    return events


def _call_in_thread(work):
    outcome = {}

    def run():
        try:
            outcome["result"] = work()
        except BaseException as error:  # reported by the test, not lost in the thread
            outcome["error"] = error

    return threading.Thread(target=run, name="queued-local-call", daemon=True), outcome


def _wait_until(predicate, timeout=30):
    until = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < until, "bounded condition did not become true"
        time.sleep(0.01)


def test_a_waiting_local_request_opens_no_connection_for_its_count_before_it_holds_the_slot(lane, monkeypatch):
    """No number chosen: a call behind another's generation holds its count back too, so nothing reaches the server
    and nothing is reserved while it waits. Then the count and the send follow in order, each under one slot and
    never two, and the exact count sizes the request."""
    _limit(lane, monkeypatch)
    events = _watch(lane, monkeypatch)
    exact, attempt_request = {}, llm_local._attempt_request

    def spy(target, candidate, **kwargs):
        request = attempt_request(target, candidate, **kwargs)
        exact["count"] = (request.prompt_tokens_exact, request.context_window_tokens_exact)
        return request

    monkeypatch.setattr(llm_local, "_attempt_request", spy)
    call, outcome = _call_in_thread(lambda: _send(lane, _nano(16_384)))
    try:
        with mc.model_call_slot("local-model", True, route=BUILT_IN):  # another call generates on the one slot
            call.start()
            _wait_until(lambda: _queue_files(lane, "ticket"))
            assert not lane["measured"] and not lane["sent"] and not events
            assert not _rows(lane["root"])
    finally:
        call.join(30)
    assert not call.is_alive() and "error" not in outcome, outcome.get("error")
    assert outcome["result"][0]["content"] == "local"
    assert events == [("count", 1), ("send", 1)]
    assert len(lane["measured"]) == len(lane["sent"]) == 1
    assert exact["count"] == (5_000, 16_384) and lane["sent"][0]["max_tokens"] == 4_096
    assert [row["state"] for row in _rows(lane["root"])] == ["settled"]
    assert not _queue_files(lane, "slot") and not _queue_files(lane, "ticket")


def test_a_number_of_the_owners_keeps_the_count_outside_the_queue(lane, monkeypatch):
    """With the owner's number the count is not held back: it reaches the server at once while both slots are taken,
    as before, and only the send waits in line."""
    _limit(lane, monkeypatch, 2)
    events = _watch(lane, monkeypatch)
    call, outcome = _call_in_thread(lambda: _send(lane, _nano(16_384)))
    holders = contextlib.ExitStack()
    try:
        for _ in range(2):
            holders.enter_context(mc.model_call_slot("local-model", True, route=BUILT_IN))
        call.start()
        _wait_until(lambda: lane["measured"] and _queue_files(lane, "ticket"))
        assert events == [("count", 2)] and not lane["sent"]
    finally:
        holders.close()
        call.join(30)
    assert not call.is_alive() and "error" not in outcome, outcome.get("error")
    assert [name for name, _ in events] == ["count", "send"] and events[0] == ("count", 2) and len(lane["sent"]) == 1


def test_a_local_request_waiting_for_its_count_ends_at_its_callers_deadline_with_nothing_sent(lane, monkeypatch):
    _limit(lane, monkeypatch)
    events = _watch(lane, monkeypatch)
    with mc.model_call_slot("local-model", True, route=BUILT_IN):
        with pytest.raises(mc.ModelSlotTimeout, match="the request was not sent"), \
                mc.model_call_scope("local-model", True, deadline_ts=time.time() + 0.3):
            _send(lane, _nano(16_384))
        assert not events and not lane["measured"] and not lane["sent"]
        assert not _rows(lane["root"]) and not _queue_files(lane, "ticket")


def test_the_budget_lookahead_declines_a_busy_slot_and_prices_from_its_estimate(lane, monkeypatch):
    """The lookahead (``task_pacing.prospective_wrapup_attempt_request``) prices advice, never queues and never
    refuses: while a generation holds the server's one slot its count admits nothing, opens no connection and
    leaves no place in line, so the request it returns carries its estimate and no exact count."""
    _limit(lane, monkeypatch)
    events = _watch(lane, monkeypatch)
    with mc.model_call_slot("local-model", True, route=BUILT_IN), \
            mc.model_call_scope("local-model", True, deadline_ts=time.time() + 5):  # a queued count would end here
        with ua.usage_scope(_scope(lane["root"])), ua.bind_physical_attempt_context(_nano(16_384)), \
                sc.MainSendClock(None).bound():
            request = task_pacing.prospective_wrapup_attempt_request(
                llm=lane["client"], messages=copy.deepcopy(MESSAGES), model="local-model",
                reasoning_effort="medium", use_local=True)
        assert not _queue_files(lane, "ticket")  # the declined count left no place in line behind the generation
    assert events == [] and not lane["measured"] and not lane["sent"]
    assert request.provider == "local" and request.model == "local-model" and request.source == "llm.local"
    assert request.max_completion_tokens == 4_096  # the quarter-window ceiling of the confirmed 16_384 window
    assert request.prompt_tokens_estimate > 0  # the estimate stays in force
    assert request.prompt_tokens_exact is None and request.context_window_tokens_exact is None  # nothing was counted


def test_the_budget_lookahead_counts_once_under_the_free_slot(lane, monkeypatch):
    """With the slot free the same lookahead measures as the sender does — one count, under the one slot, and no
    send of its own — and the request it returns carries the server's exact count beside the estimate."""
    _limit(lane, monkeypatch)
    events = _watch(lane, monkeypatch)
    with ua.usage_scope(_scope(lane["root"])), ua.bind_physical_attempt_context(_nano(16_384)), \
            sc.MainSendClock(None).bound():
        request = task_pacing.prospective_wrapup_attempt_request(
            llm=lane["client"], messages=copy.deepcopy(MESSAGES), model="local-model",
            reasoning_effort="medium", use_local=True)
    assert events == [("count", 1)] and len(lane["measured"]) == 1 and not lane["sent"]
    assert request.prompt_tokens_exact == 5_000 and request.context_window_tokens_exact == 16_384
    assert request.prompt_tokens_estimate > 0 and request.max_completion_tokens == 4_096
    assert not _queue_files(lane, "slot") and not _queue_files(lane, "ticket")
