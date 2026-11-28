"""Which declared number binds a start on OpenAI's realtime route, at the adapter's own session candidate.

The catalogue states two numbers for ``gpt-realtime-2.1`` (spoken probes of 27.11.2026): the conversation window, the
largest size at which a whole six-turn call kept the spoken fact and the late recall, and a larger input the endpoint
accepted over three turns. Only the window binds admission (its usable input is the window less the working reserve
once the reply allowance has priced down to fit); the accepted input is evidence. The real adapter's session candidate
is driven with the socket trapped on one protected owner row sized a few tokens under and over that usable input.
"""
from __future__ import annotations

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.contracts.realtime import RealtimeSetupError
from tests._live_world import host_world as host_world
from tests.test_live_conversation import packages as packages


def _candidate(host, provider, base, owner_chars):
    from ouroboros.context_budget import bound_host_row
    from ouroboros.realtime_setup import session_candidate

    def trapped(*args, **kwargs):
        def connect(*_a, **_kw):
            pytest.fail("sizing tried to connect")
        return provider["session"](*args, connect=connect, **kwargs)

    host.messages[:] = [*base, bound_host_row("owner_words", "w" * owner_chars)]
    host._context_changed()
    return session_candidate(host, trapped, LiveOptions(), observation_fp=f"openai-window-edge-{owner_chars}")


def test_the_conversation_window_binds_admission_and_the_accepted_input_does_not(host_world, packages):
    from ouroboros.realtime_setup import loaded_provider, session_refusal_notice

    host = host_world.host
    provider = loaded_provider("openai_realtime:openai")
    model = provider["catalog"].model("gpt-realtime-2.1")
    facts = model.facts
    host.model, host.turn.native_model = model, model.id
    base = list(host.messages)
    usable = facts.conversation_window_tokens - facts.working_reserve_tokens
    assert facts.accepted_input_tokens > facts.conversation_window_tokens  # two different numbers, one of them binds

    chars, measured = 400_000, None
    for _ in range(4):  # the wire's own fragments make the estimate not quite linear in characters: converge on it
        _session, _request, proof = _candidate(host, provider, base, chars)
        measured = proof["fit"]["measurement"]
        if usable - 8 <= measured["estimated_input_tokens"] < usable:
            break
        chars += 4 * (usable - 4 - measured["estimated_input_tokens"])
    under = measured["estimated_input_tokens"]
    assert usable - 8 <= under < usable, measured
    assert measured["capacity_total_tokens"] == facts.conversation_window_tokens
    assert measured["capacity_deficit_tokens"] == 0
    assert under + measured["response_reserve_tokens"] <= facts.conversation_window_tokens

    # Over the edge: the ladder may first fold this room's memory to its floor (lawful, about a thousand tokens here)
    # and admit what then fits; once the floor is reached, a few tokens more are refused, far under the accepted input.
    refusal = None
    for _ in range(4):
        try:
            _session, _request, proof = _candidate(host, provider, base, chars + 4 * (usable - under + 4))
        except RealtimeSetupError as exc:
            refusal = exc
            break
        folded = proof["fit"]["measurement"]
        assert folded["estimated_input_tokens"] + folded["response_reserve_tokens"] <= facts.conversation_window_tokens
        chars, under = chars + 4 * (usable - under + 4), folded["estimated_input_tokens"]
    assert refusal is not None and refusal.code == "context_capacity"
    refused = host.trace["fit_refusal"]["fit"]["measurement"]
    assert usable < refused["estimated_input_tokens"] <= usable + 64  # just over the window's usable input
    assert refused["estimated_input_tokens"] < facts.accepted_input_tokens  # the accepted input does not bind
    details = session_refusal_notice(host)["notice"]["details"]
    assert f"Window: {facts.conversation_window_tokens:,} " in details
    assert facts.source in details and facts.observed in details
