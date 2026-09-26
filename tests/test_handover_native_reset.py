"""Active-turn repair survives Main's actual wait and canonical reprepare."""

import asyncio
from copy import deepcopy

import pytest

from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY
from ouroboros import loop, usage_accounting as ua
from ouroboros.llm_claudexor import ModelTurnState
from ouroboros.loop_model_call import _reprepare_waiting_main
from tests.test_llm_claudexor import EARLIER, MODEL, ROUTE, TURN, ledger, result
from tests.test_llm_claudexor import setup as gateway_fixture, turn_engine as engine_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_subscription_main_wait import ROUTE_B, main_call as main_fixture

setup = gateway_fixture
turn_engine = engine_fixture
live_wait = wait_fixture
main_call = main_fixture


# Main dispatch is synchronous; the async transport also uses the native-repair callback.
@pytest.mark.parametrize("asynchronous,quota_wait", [(False, False), (False, True), (True, False)],
                         ids=["sync-repair", "sync-quota-wait", "async-repair"])
@pytest.mark.parametrize("route", [ROUTE, ROUTE_B], ids=["same-account", "changed-account"])
def test_main_dual_token_reset_survives_wait_reprepare_and_adopts_new_envelope(
    main_call, turn_engine, asynchronous, route, quota_wait,
):
    ctx, gateway, controller, events, _decide, observations = main_call
    original = deepcopy(ctx.messages)
    slot = ModelTurnState(deepcopy(EARLIER))
    ctx.tools._ctx.model_turn_state = slot
    failure = result(outcome="failed", route=route,
                     problem={"code": "invalid_continuation", "message": "refused"})
    landed = {**result(route=route), "nativeContinuation": deepcopy(TURN)}
    gateway.results = [failure, landed]
    gateway.dispatch = ["not_started", "response_received"]
    if quota_wait:
        gateway.results.insert(0, result(outcome="failed", problem={
            "code": "subscription_window_exhausted", "message": "wait for capacity"}))
        gateway.dispatch.insert(0, "not_started")

    if asynchronous:
        disposition = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
        with controller.register_reprepare("main", lambda values: _reprepare_waiting_main(ctx, values)):
            with ua.bind_physical_attempt_context(loop._physical_context_for_fit(disposition)):
                answer, _usage = asyncio.run(ctx.llm.chat_async(
                    ctx.messages, MODEL, model_role="main", model_turn_state=slot))
    else:
        answer, _cost, _mode = loop._call_round_model(ctx)

    assert answer and ctx.tools._ctx.model_turn_state is slot and slot.envelope == TURN
    assert gateway.uploads[0][0]["nativeContinuation"] == EARLIER
    assert gateway.uploads[-1][0]["nativeContinuation"] is None
    assert len(gateway.accepted_operations) == 2 + int(quota_wait)
    expected = deepcopy(original)
    for message in expected:
        message.pop("nativeContinuation", None)
    assert ctx.messages == expected
    # The canonical system message carries the builder's host-only stable-prefix
    # declaration; the Codex send copy pops it (llm_claudexor._request).
    wire_expected = deepcopy(expected)
    wire_expected[0].pop(STABLE_PREFIX_BLOCKS_KEY, None)
    assert gateway.uploads[-1][0]["messages"] == wire_expected
    assert ctx.context_fit_plan.core_sha256 == "a" * 64
    assert any(item["model_route"] == route for item in observations)
    rows = ledger(ctx.drive_root)
    assert [row["state"] for row in rows].count("released") == 1 + int(quota_wait)
    assert [row["state"] for row in rows].count("settled") == 1
    final_dispatch = [row for row in rows if row["state"] == "dispatched"][-1]
    assert final_dispatch["physical_context"]["route_fp"] == f"capacity-{route['credentialProfileId']}"
    if quota_wait:
        waits = [event for event in list(events.queue) if event.get("type") == "task_model_wait"]
        assert waits and waits[-1]["state"] == "resolved"
