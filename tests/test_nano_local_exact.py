"""The local lane in a rendered Nano: the exact count decides before the approximate compactor (T4).

When the serving instance counts exactly, a Nano candidate is measured whole: a fit is sent
uncompacted; a shortfall applies today's markdown compactor ONCE, re-measures, and sends or
raises the typed overflow with the refused candidate's own facts. Without exact support, and
in Low and Max, today's compacted path is unchanged.
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
COMPACTED = "[Compacted for local-model context"
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


def _todays_compacted(lane):
    return lane["client"]._build_local_candidate(copy.deepcopy(MESSAGES), None, 65_536, "auto")[1]["messages"]


def test_an_exact_fit_is_sent_whole_without_the_compactor(lane):
    message, _usage = _send(lane, _nano(16_384))
    [sent] = lane["sent"]
    assert message["content"] == "local" and len(lane["measured"]) == 1
    assert "cut cut" in sent["messages"][0]["content"] and COMPACTED not in sent["messages"][0]["content"]
    assert sent["max_tokens"] == 4_096  # the quarter-window ceiling: never raised, the reply floor is the ceiling
    assert [row["state"] for row in _rows(lane["root"])] == ["settled"]  # the store keeps one row per attempt
    assert _todays_compacted(lane)[0]["content"] != sent["messages"][0]["content"]  # today's path would have cut it


def test_a_262k_window_sends_an_80k_exact_input_with_the_whole_ceiling(lane):
    lane["window"], lane["count"] = 262_144, lambda payload: 80_000
    _send(lane, _nano(262_144))
    [sent] = lane["sent"]
    assert sent["max_tokens"] == 65_536 and COMPACTED not in sent["messages"][0]["content"]


def test_an_exact_shortfall_compacts_once_remeasures_and_sends_with_one_clock_line(lane):
    lane["count"] = lambda payload: 3_000 if COMPACTED in payload["messages"][0]["content"] else 15_000
    clock = sc.MainSendClock(sc.SendClockPolicy("UTC"))
    _send(lane, _nano(16_384), clock)
    [sent] = lane["sent"]
    assert len(lane["measured"]) == 2 and COMPACTED in sent["messages"][0]["content"]
    assert sent["max_tokens"] == 4_096
    assert len(clock.notes) == 1 and sent["messages"][-1] == {"role": "user", "content": clock.notes[0]}
    assert lane["measured"][0]["messages"][-1] == sent["messages"][-1]  # the whole candidate carried the same line


def test_a_second_shortfall_is_the_typed_refusal_with_its_own_candidate_facts(lane):
    lane["count"] = lambda payload: 15_000
    with pytest.raises(LocalContextTooLargeError) as caught:
        _send(lane, _nano(16_384))
    facts = caught.value.refused_candidate
    assert not lane["sent"] and len(lane["measured"]) == 2
    assert not _rows(lane["root"])  # nothing reserved, nothing sent
    assert facts["provider"] == "local" and facts["model"] == "local-model" and facts["max_completion_tokens"] == 4_096
    assert facts["physical_context"]["rendered_mode"] == "nano" and facts["input_tokens"] == 15_000
    assert len(facts["candidate_raw_sha256"]) == 64 and facts["candidate_context_size_bytes"] > 0
    assert "below the floor" in str(caught.value)


def test_without_exact_support_todays_compacted_path_is_unchanged(lane):
    lane["count"] = lambda payload: None
    _send(lane, _nano(16_384))
    [sent] = lane["sent"]
    assert len(lane["measured"]) == 2 and COMPACTED in sent["messages"][0]["content"]
    assert lane["measured"][1]["messages"] == sent["messages"]  # today's one measurement of the compacted twin
    assert sent["messages"] == _todays_compacted(lane) and sent["max_tokens"] == 4_096


def test_a_compacted_twin_the_instance_can_count_keeps_its_exact_allowance(lane):
    # The whole candidate cannot be counted (a transient refusal), its compacted twin can: today's path measured
    # that twin, so its exact count (not the small approximate estimate) still sizes the reply.
    counts = iter((None, 200_000))
    lane["window"], lane["count"] = 262_144, lambda payload: next(counts)
    _send(lane, _nano(262_144))
    [sent] = lane["sent"]
    assert len(lane["measured"]) == 2 and lane["measured"][1]["messages"] == sent["messages"]
    assert sent["max_tokens"] == 262_144 - 200_000  # exact: no slack, below the 65,536 ceiling


@pytest.mark.parametrize("physical", [None, ua.PhysicalAttemptContext(
    profile="owner_max", rendered_mode="max", measurement_basis="cold_estimate", route_fp="local", round_id="x:round:1",
    target_total_tokens=None, capacity_total_tokens=16_384, context_target_miss=False, automatic_pass_used=False)])
def test_low_and_max_keep_todays_compacted_path_even_with_an_exact_count(lane, physical):
    _send(lane, physical)
    [sent] = lane["sent"]
    assert len(lane["measured"]) == 1 and sent["messages"] == _todays_compacted(lane)  # measured once, re-sealed, as today
    assert sent["max_tokens"] == 4_096


def test_a_changed_serving_instance_is_still_a_preparation_failure(lane):
    def count(payload):
        lane["pid"] = 456  # the instance that measured is gone before dispatch
        return 5_000

    lane["count"] = count
    with pytest.raises(ua.PhysicalAttemptPreparationFailed, match="instance changed"):
        _send(lane, _nano(16_384))
    assert not lane["sent"]
