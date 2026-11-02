"""The owner's standing rules reach every acting mind through the read the Supervisor makes.

One source: ``runtime_setting`` serves the actor's settings snapshot, so a captured context and the
checks of that task hold the same bytes, a later snapshot belongs to a later capture or actor, and a
delegated session gets them in its start request, never in its canonical work order.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import context
from ouroboros.safety_config import OWNER_RULES_KEY, owner_rules_section
from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests.test_book_context_capture import _plan
from tests.test_doc_context import _make_env_and_memory

RULES = "Rule one: keep my private work private.\nRule two: hold a disputed send and ask me in Main."
LIVE = "LIVE ENVIRONMENT TEXT, NOT THIS TASK'S SNAPSHOT"


def _scope(rules):
    return task_settings_scope(task_settings_snapshot({}, {**os.environ, OWNER_RULES_KEY: rules}))


def _bundle(prompt):
    return json.loads(prompt.split("Host evidence (provenance is not consent):\n", 1)[1]
                      .split("\nProposed tool call:", 1)[0])


@pytest.mark.parametrize("role", ["main", "api", "declared_api", "declared_nanny"])
def test_rules_reach_every_author_role_and_mode_once_from_the_task_snapshot(tmp_path, monkeypatch, role):
    monkeypatch.setenv(OWNER_RULES_KEY, LIVE)
    env, memory = _make_env_and_memory(tmp_path)
    task = {"id": "rules-author", "type": "task", "text": "Do the work"}
    if role != "main":
        task.update(delegation_role="subagent", configured_subagent={"route": {
            "kind": "agent_session" if role == "declared_nanny" else "api_model"}})
    if role.startswith("declared"):
        task["task_contract"] = {"input_sources": "declared", "context_requires_self_body_docs": False}
    with _scope(RULES):
        core = context._capture_context_core(env, memory, task, None, None)
    plan = _plan(env, core, task, "max")
    captured = {}
    for mode in ("max", "low", "nano"):
        blocks = plan.projection(mode).system_message()["content"]
        text = "\n".join(block["text"] for block in blocks)
        assert text.count(RULES.splitlines()[0]) == 1 and LIVE not in text
        # Governance-like and stable: it rides a cached block, so the shared prefix is unchanged by tasks.
        assert any("## My human's standing rules" in block["text"] and RULES in block["text"]
                   for block in blocks if block.get("cache_control"))
        captured[mode] = json.dumps(plan.messages_for(mode), ensure_ascii=False)
    with _scope("A later rule."):  # a later actor's snapshot is a new capture, never a rewrite of this one
        later = context._capture_context_core(env, memory, task, None, None)
    assert "A later rule." in later.semi_stable_text and RULES not in later.semi_stable_text
    assert {mode: json.dumps(plan.messages_for(mode), ensure_ascii=False) for mode in captured} == captured


def test_the_supervisor_and_the_author_read_the_same_snapshot(tmp_path, monkeypatch):
    from ouroboros import safety

    monkeypatch.setenv(OWNER_RULES_KEY, LIVE)
    ctx = SimpleNamespace(task_id="t", task_metadata={"work_kind": "owner"}, drive_root=tmp_path)
    for rules in (RULES, "Another task's rule.", ""):
        with _scope(rules):
            bundle = _bundle(safety._build_check_prompt("run_command", {"cmd": "ls"}, ctx=ctx))
            section = owner_rules_section()
        assert bundle["standing_delegation"]["owner_rules"] == rules
        assert section.endswith("\n\n" + rules) if rules else section.endswith("None set: "
            "Settings › Behavior › Safety Supervisor › Your rules is empty.")
        assert LIVE not in section


def test_cleared_and_unreadable_rules_are_stated_never_guessed(monkeypatch):
    with _scope("  \n"):
        assert "None set" in owner_rules_section() and "None set" in owner_rules_section(session=True)

    def unreadable(_key, _default=None):
        raise RuntimeError("settings unreadable")
    monkeypatch.setattr("ouroboros.config.runtime_setting", unreadable)
    text = owner_rules_section()
    assert "Unreadable now (RuntimeError)" in text and "None set" not in text


@pytest.mark.parametrize("input_sources", ["shared", "declared"])
def test_session_start_carries_the_rules_beside_an_unchanged_work_order(tmp_path, monkeypatch, input_sources):
    from ouroboros.subagent_work_order import compile_external_work_order, work_order_fingerprint
    from tests.test_session_design_navigation import _start

    monkeypatch.setenv(OWNER_RULES_KEY, RULES)
    request, _row, task, _ctx, _live, _prepared = _start(
        tmp_path, monkeypatch, compiled=True, kind="foreign_copy", input_sources=input_sources)
    instructions = request["instructions"]
    assert "OWNER RULES DELIVERY (host instructions, separate from canonical work order)" in instructions
    assert instructions.count(RULES) == 1
    canonical, fingerprint = compile_external_work_order(task), work_order_fingerprint(task)
    assert request["prompt"] == canonical and RULES not in canonical
    monkeypatch.setenv(OWNER_RULES_KEY, "A later rule.")  # retained work-order provenance does not move
    assert compile_external_work_order(task) == canonical and work_order_fingerprint(task) == fingerprint


def test_an_unknown_start_retry_replays_the_rules_it_was_prepared_with(tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_delegate_continuation import _start

    monkeypatch.setenv(OWNER_RULES_KEY, RULES)
    _select_transport_access(monkeypatch, "readonly")
    original, unknown, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="rules-session", prompt="Do the work",
        start_error=ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503))
    assert original["instructions"].count(RULES) == 1
    monkeypatch.setenv(OWNER_RULES_KEY, "A later rule.")
    retry, payload, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="rules-session", prompt="Do the work",
        start_kwargs={"retry_of": unknown["pending_invocation_id"]})
    assert payload["status"] == "started" and retry == original


def test_supervisor_prompt_keeps_machine_read_verdict_format():
    """The parser's verdict vocabulary is structural; prompt meaning belongs to review."""
    text = (Path(__file__).resolve().parents[1] / "prompts" / "SAFETY.md").read_text(encoding="utf-8")
    assert '"status": "SAFE" | "SUSPICIOUS" | "DANGEROUS"' in text
