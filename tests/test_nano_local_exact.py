"""The local lane never cuts a request: the serving instance's exact count decides, or the server does.

When the instance counts exactly, a candidate is measured whole: a fit is sent as prepared; a
shortfall is the typed size refusal with the refused candidate's own facts, before any send.
Without exact support, and in Low and Max, the whole prepared candidate is sent too (the
server's own refusal is the same typed refusal): no markdown compactor runs on any path.
"""
import copy
from types import SimpleNamespace

import pytest

from ouroboros import local_model, send_clock as sc, usage_accounting as ua
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
    evidence = lambda: {"context_window": state["window"], "confirmed": True, "process_id": state["pid"],  # noqa: E731
                        "source": "published_server_arguments", "port": 1}

    def measure(payload):
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
