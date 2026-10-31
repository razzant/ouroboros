"""One host-facts sentence for every session with a scoped native-state home (owner 3B, #1772).

The engine provides ``CLAUDEXOR_OPERATOR_HOME`` and, when the operator's browser cache
exists, ``PLAYWRIGHT_BROWSERS_PATH``; the host tells the model those facts exist, once, in
the delegated host instructions and the reviewer agent-session preamble (the builder that
first-send sizing also uses), and never in a raw API reviewer prompt, which has no shell
HOME. No command list.
"""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros.delegate_start_instructions import (
    HOST_FACTS_INSTRUCTION, HOST_INSTRUCTIONS, SHARED_ENVIRONMENT_INSTRUCTION,
)
from ouroboros.review_session_preparation import render_review_session_prompt
from ouroboros.subagents import delegated_run_shape
from ouroboros.tools.delegate import _host_instructions


def test_the_sentence_names_the_two_engine_facts_as_present_when_provided_and_lists_no_commands():
    assert "CLAUDEXOR_OPERATOR_HOME" in HOST_FACTS_INSTRUCTION and "PLAYWRIGHT_BROWSERS_PATH" in HOST_FACTS_INSTRUCTION
    assert "when present" in HOST_FACTS_INSTRUCTION and "scoped native-state home" in HOST_FACTS_INSTRUCTION
    assert "scratch" not in HOST_FACTS_INSTRUCTION, "a delegated lane home persists across turns"
    assert HOST_FACTS_INSTRUCTION.count(".") == 1, "one sentence"
    for command in ("ls ", "find ", "npx", "playwright install", "cd "):
        assert command not in HOST_FACTS_INSTRUCTION
    assert HOST_FACTS_INSTRUCTION not in HOST_INSTRUCTIONS, "one shared helper, not a second copy inside the ban text"


def test_every_delegated_run_shape_carries_the_sentence_exactly_once():
    for shape in (delegated_run_shape(False), delegated_run_shape(False, "readonly"), delegated_run_shape(True),
                  delegated_run_shape(True, "full")):
        text = _host_instructions(shape)
        assert text.count(HOST_FACTS_INSTRUCTION) == 1
        assert text.count(SHARED_ENVIRONMENT_INSTRUCTION) == 1
        assert text.index(HOST_FACTS_INSTRUCTION) < text.index("ACCESS:"), "facts before the access line"
    payload = _host_instructions(delegated_run_shape(True), payload_skill="alpha", assignment="ASSIGNMENT\nx")
    assert payload.count(HOST_FACTS_INSTRUCTION) == 1 and payload.endswith("\n\nASSIGNMENT\nx")
    assert payload.count(SHARED_ENVIRONMENT_INSTRUCTION) == 1


def test_the_reviewer_agent_session_preamble_carries_it_and_the_raw_api_prompt_does_not():
    request = SimpleNamespace(surface="plan", policy={})
    for access in ("readonly", "workspace_write"):
        slot = SimpleNamespace(slot_id="s1", role_hint="", session_access=access)
        prompt = render_review_session_prompt(request, slot, "review this")
        assert prompt.count(HOST_FACTS_INSTRUCTION) == 1
        assert prompt.count(SHARED_ENVIRONMENT_INSTRUCTION) == 1
        assert prompt.splitlines()[0].endswith(HOST_FACTS_INSTRUCTION.strip()), "one line: the preamble sentence"
    from ouroboros import review_execution, review_native_episode

    assert HOST_FACTS_INSTRUCTION not in review_native_episode.native_episode_prompt("plan", "", "task", "contract", "s1")
    source = open(review_execution.__file__, encoding="utf-8").read()
    assert "HOST_FACTS_INSTRUCTION" not in source, "the raw API reviewer prompt has no shell HOME"
