"""Commit/skill delivery vectors retain the pool's captured processing binding."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import reviewer_slot_config as slots
from ouroboros.review_execution import ApiChatReviewExecutor, ReviewAssignment, ReviewRouteKind
from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
from ouroboros.tools.review_admission import drop_api_rows, seat_vectors
from tests.review_pool_rosters import pool_roster, pool_seat
from tests.test_review_pool import provider_call


MODEL = "openai::gpt-5.6-sol"


@pytest.fixture
def dispatch(monkeypatch, tmp_path):
    """Run the real vector fan-out/constructor, then inspect both executors' sends.

    The substrate is replaced before any custody or provider work; provider_call
    below separately exercises the stock client's resolution with fake transports.
    """
    from ouroboros.tools import review, review_multi_model

    monkeypatch.setenv("OPENAI_API_KEY", "sk-present-key")
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", "{}")
    monkeypatch.setattr(review, "LLMClient", lambda: object())
    monkeypatch.setattr(review, "_parse_model_response", lambda _model, payload, _headers: payload)
    monkeypatch.setattr(review_multi_model, "triad_api_messages", lambda *a, **kw: (
        [{"role": "user", "content": "Review this packet."}], ""))
    seen = {}

    def run(request, *, slots, **_kwargs):
        [slot] = slots
        assignment = ReviewAssignment(request, slot)
        if slot.native_retrieval:
            kwargs = NativeToolRoundReviewExecutor(assignment)._chat_kwargs([], [], 8)
        else:
            kwargs = ApiChatReviewExecutor(assignment)._kwargs()
        seen[slot.slot_id] = (slot, kwargs)
        return SimpleNamespace(actors=[{"slot_id": slot.slot_id, "status": "ok", "raw_text": "[]"}])

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", run)

    def deliver(plan):
        seen.clear()
        ctx = SimpleNamespace(task_id="processing-regression", drive_root=tmp_path, task_metadata={})
        result = asyncio.run(review_multi_model._multi_model_review_async(
            "review evidence", "review prompt", plan["models"], ctx,
            routes=plan["routes"], row_plan=plan, session_task="Read the change.",
            session_root=str(tmp_path), checking_source={}, layer="core"))
        assert result["model_count"] == len(plan["models"])
        assert all("error" not in row for row in result["results"]), result
        assert len(seen) == len(plan["models"])
        return dict(seen)

    return deliver


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("processing", ["economy", ""])
def test_empty_pool_main_processing_survives_commit_vectors(monkeypatch, dispatch, local, processing):
    settings = {
        "OUROBOROS_SUBAGENTS": pool_roster(), "OUROBOROS_MODEL": MODEL,
        "USE_LOCAL_MAIN": str(local), "OUROBOROS_MODEL_ACCOUNTS": '{"main":"main-pin"}',
        "OUROBOROS_PROCESSING_PREFERENCE": "standard" if processing else "",
        "OUROBOROS_MODEL_PROCESSING_PREFERENCES": json.dumps({"main": processing}),
    }
    before = dict(settings)
    with task_settings_scope(task_settings_snapshot(settings, settings)):
        plan = seat_vectors(slots.commit_triad_delivery())
    # Dispatch outside the captured plane: ambient processing is Fast.
    assert plan["processing_preferences"] == [processing]
    seat, kwargs = dispatch(plan)["main"]
    assert seat.native_retrieval and seat.subagent_id == ""
    assert seat.effort == plan["efforts"][0]
    assert seat.processing_preference == kwargs["processing_preference"] == processing
    assert provider_call(monkeypatch, kwargs) == (MODEL, "reviewer:main", "main-pin", processing, local)
    assert settings == before


@pytest.mark.parametrize("asynchronous", [False, True], ids=["chat", "chat_async"])
@pytest.mark.parametrize("native", [False, True], ids=["packet", "native"])
@pytest.mark.parametrize("processing", ["", "economy", None])
def test_local_candidate_keeps_captured_processing_or_inherits_when_unspecified(
        monkeypatch, dispatch, asynchronous, native, processing):
    from ouroboros.llm import LLMClient

    plan = {"models": [MODEL], "routes": [ReviewRouteKind.API_CHAT], "slot_ids": ["main"],
            "use_local": [True], "retrieves": [native], "processing_preferences": [processing]}
    _, kwargs = dispatch(plan)["main"]
    seen = []

    class CandidateBuilt(BaseException):
        """Stop after real construction, before measurement or physical dispatch."""

    def finalized(_client, target, payload):
        seen.append((target, payload))
        raise CandidateBuilt()

    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda: SimpleNamespace(
        serving_context_evidence=lambda: {"confirmed": True, "context_window": 8192}))
    monkeypatch.setattr(LLMClient, "_get_local_client", lambda _client: object())
    monkeypatch.setattr(LLMClient, "_finalize_local_candidate", finalized)
    client = LLMClient()
    with pytest.raises(CandidateBuilt):
        if asynchronous:
            asyncio.run(client.chat_async(**kwargs))
        else:
            client.chat(**kwargs)
    [(target, payload)] = seen
    assert target["provider"] == "local" and payload["model"] == "local-model"
    assert target["processing_preference"] == ("fast" if processing is None else processing)


@pytest.mark.parametrize("delivery", ["native", "packet"])
def test_catalog_processing_survives_commit_vectors(monkeypatch, dispatch, delivery):
    settings = {"OUROBOROS_SUBAGENTS": pool_roster(pool_seat(
        "critic", MODEL, delivery=delivery, effort="high",
        processing_preference="economy")), "OUROBOROS_PROCESSING_PREFERENCE": "standard"}
    with task_settings_scope(task_settings_snapshot(settings, settings)):
        plan = seat_vectors(slots.commit_triad_delivery())
    seat, kwargs = dispatch(plan)["critic"]
    assert seat.native_retrieval == (delivery == "native")
    assert (seat.subagent_id, seat.effort) == ("critic", "high")
    assert provider_call(monkeypatch, kwargs) == (MODEL, "reviewer:critic", "", "economy", False)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("processing", [None, ["", None]])
def test_missing_processing_vector_or_tail_keeps_unbound_compatibility(monkeypatch, dispatch, native, processing):
    plan = {"models": [MODEL] * 3, "routes": [ReviewRouteKind.API_CHAT] * 3,
            "slot_ids": ["first", "second", "tail"], "use_local": [False] * 3,
            "retrieves": [native] * 3}
    if processing is not None:
        plan["processing_preferences"] = processing
    sent = dispatch(plan)
    for index, name in enumerate(plan["slot_ids"]):
        expected = processing[index] if processing is not None and index < len(processing) else None
        seat, kwargs = sent[name]
        assert seat.processing_preference == expected
        assert provider_call(monkeypatch, kwargs)[3] == ("fast" if expected is None else expected)


def test_filtered_vectors_keep_processing_with_the_selected_seats(dispatch):
    settings = {"OUROBOROS_SUBAGENTS": pool_roster(
        pool_seat("packet", MODEL, processing_preference="fast"),
        pool_seat("native", MODEL, delivery="native", processing_preference="economy"),
        pool_seat("native-empty", MODEL, delivery="native")),
        "OUROBOROS_PROCESSING_PREFERENCE": ""}
    with task_settings_scope(task_settings_snapshot(settings, settings)):
        plan = seat_vectors(slots.commit_triad_delivery())
    filtered = drop_api_rows(plan)
    assert filtered["slot_ids"] == ["native", "native-empty"]
    sent = dispatch(filtered)
    assert sent["native"][0].processing_preference == "economy"
    assert sent["native-empty"][0].processing_preference == ""


def test_filtered_session_rights_stay_with_their_original_seats():
    settings = {"OUROBOROS_SUBAGENTS": pool_roster(
        pool_seat("packet", MODEL, processing_preference="fast"),
        pool_seat("write", "codex", kind="agent_session", access="workspace_write",
                  processing_preference="economy"),
        pool_seat("native", MODEL, delivery="native"),
        pool_seat("packet-tail", MODEL),
        pool_seat("full", "claude", kind="agent_session", access="full",
                  processing_preference="standard")), "OUROBOROS_PROCESSING_PREFERENCE": ""}
    with task_settings_scope(task_settings_snapshot(settings, settings)):
        plan = seat_vectors(slots.commit_triad_delivery())
    filtered = drop_api_rows(plan)
    assert filtered["slot_ids"] == ["write", "native", "full"]
    assert filtered["session_accesses"] == ["workspace_write", "", "full"]
    assert filtered["processing_preferences"] == ["economy", "", "standard"]
    assert plan["session_accesses"] == ["", "workspace_write", "", "", "full"]
