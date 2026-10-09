"""Every deep-review row delivers by RETRIEVAL — a bare api row included.

The `deep_review` row has one delivery class: the reviewer reads the repository
itself. An `api_chat` row runs the bounded native inspection episode whether or
not a configured subagent binds it (nothing is ever assembled and handed over),
and a row stored as `openai/<slug>` still runs on an install whose only OpenAI
access is the direct API, because availability resolves the direct route and the
episode is sent on THAT route.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.deep_self_review import deep_review_route, run_deep_self_review
from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS
from ouroboros.reviewer_slot_config import ConfiguredReviewerSlot, reviewer_slot_last_executions
from tests.test_deep_review_slot import _DEEP_SLOT_ID, _ScriptedLLM, _tool_call

_BIBLE = "# BIBLE\n\n## Principle 0: Agency\n\nOuroboros is a becoming personality.\n" * 3
_REPORT = "Read: BIBLE.md in full; memory inline.\n\n# Deep self-review\n\nCRITICAL: loop.py finalization race.\n"


def _bare_row(target: str = "openai/fake-deep", **fields) -> ConfiguredReviewerSlot:
    """The row an install has without configuring one: an api route, no subagent."""
    return ConfiguredReviewerSlot(slot_id=_DEEP_SLOT_ID, kind="api_chat", target_id=target, **fields)


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "BIBLE.md").write_text(_BIBLE, encoding="utf-8")
    (root / "docs" / "ARCHITECTURE.md").write_text("# Arch\n\n## Review stack\n\ntext\n", encoding="utf-8")
    (root / "docs" / "DEVELOPMENT.md").write_text("# Dev\n\n## Rules\n\nx\n", encoding="utf-8")
    (root / "docs" / "CHECKLISTS.md").write_text("# Checks\n\n## Change Review Checklist\n\ny\n", encoding="utf-8")
    (root / "ouroboros").mkdir()
    (root / "ouroboros" / "loop.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    return root


@pytest.fixture()
def drive(tmp_path):
    root = tmp_path / "drive"
    (root / "memory" / "knowledge").mkdir(parents=True)
    (root / "memory" / "identity.md").write_text("I am Ouroboros.\n", encoding="utf-8")
    (root / "memory" / "scratchpad.md").write_text("Working notes.\n", encoding="utf-8")
    (root / "state").mkdir()
    (root / "logs").mkdir()
    return root


def test_a_bare_api_row_runs_the_native_inspection_episode(repo, drive, monkeypatch):
    """No subagent binds the row, and the delivery is still an episode: the
    reviewer gets the retrieving TASK plus read-only tools, its repository reads
    are host-observed, and the header names the native delivery."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md"}, "c1")]},
        {"content": _REPORT},
    ])
    progress: list[str] = []
    text, usage = run_deep_self_review(repo, drive, llm, progress.append,
                                       task_id="bare-deep-1", slot=_bare_row())

    assert text.endswith(_REPORT)
    header = text.split("\n")[0]
    assert "delivery=native_tool_rounds" in header and "attestation=host_observed" in header
    assert "coverage=BIBLE.md:delivered_inline" in header and "incomplete=none" in header
    assert usage["native_rounds"] == 2 and len(usage["native_tool_receipts"]) == 1
    assert usage["host_file_read_attestation"] == "host_observed"
    assert "execution_status" not in usage
    # The first send is the retrieving task (tools to read WITH), never a pack.
    first = llm.calls[0]
    task = next(m["content"] for m in first["messages"] if m["role"] == "user")
    assert "delivered IN FULL below" in task and "read_file" in task
    assert "## FILE: drive/memory/identity.md\nI am Ouroboros.\n" in task
    # The map arrives as navigation with the read instruction, never whole.
    assert "Governance navigation (read on demand)" in task
    assert "ARCHITECTURE.md (navigation map)" in task
    assert task.count(_BIBLE) == 1, "tier 1 is actually delivered, not just promised"
    assert [tool["function"]["name"] for tool in first["tools"] or []], "read-only tools ride the send"
    assert any("native_tool_rounds" in line for line in progress)
    last = reviewer_slot_last_executions()[_DEEP_SLOT_ID]
    assert last["surface"] == "deep_self_review" and last["status"] == "responded"
    assert last["effective"]["model"] == "openai/fake-deep"


@pytest.mark.parametrize("pin", ["personal", ""])
def test_w5_a_bare_review_on_main_sends_under_mains_pinned_account(repo, drive, monkeypatch, pin):
    """No row named: the deep self-review runs on the direct Main row, and every send
    of the episode carries Main's pinned account as its ``model_account_override``
    under ``model_role=reviewer:main``. The engine treats an EMPTY override as Auto
    (never a lookup of Main's role), so a pinned Main must ride here or the review
    would run on whatever account the engine picked; an unpinned Main stays Auto."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai/fake-deep")
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"main": pin, "light": "someone-else"}))
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md"}, "c1")]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(repo, drive, llm, lambda _m: None, task_id="main-pin-1")

    assert text.endswith(_REPORT) and usage["native_rounds"] == 2
    assert len(llm.calls) == 2
    assert [call["model_role"] for call in llm.calls] == ["reviewer:main", "reviewer:main"]
    assert [call["model_account_override"] for call in llm.calls] == [pin, pin]
    assert reviewer_slot_last_executions()["main"]["requested"]["profile_id"] == pin


def test_a_stored_openrouter_spelling_runs_on_the_direct_openai_route(repo, drive, monkeypatch):
    """An install whose only OpenAI access is the direct API (no OpenRouter key,
    no `OPENAI_BASE_URL` redirect) keeps a row saved as `openai/<slug>`:
    availability resolves the direct route, the episode is SENT on it, and the
    row's own pin and effort ride along — the effective route, not just the
    stored spelling."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-direct-test")

    row = _bare_row("openai/gpt-5.5", effort="xhigh", profile_id="account-a")
    assert deep_review_route(row) == ("", "openai::gpt-5.5")
    # A router-only `-pro` slug is not an OpenAI model id: it lands on the
    # direct route's own deep-review default instead of a mechanical rewrite.
    assert deep_review_route(_bare_row("openai/gpt-5.6-sol-pro")) == (
        "", OPENAI_DIRECT_DEFAULTS["deep_self_review"])

    llm = _ScriptedLLM([{"content": _REPORT}])
    progress: list[str] = []
    text, usage = run_deep_self_review(repo, drive, llm, progress.append,
                                       task_id="direct-deep-1", slot=row)

    assert text.endswith(_REPORT)
    assert llm.calls[0]["model"] == "openai::gpt-5.5", "the wire carries the resolved direct route"
    assert llm.calls[0]["reasoning_effort"] == "xhigh"
    assert usage["resolved_model"] == "openai::gpt-5.5"
    assert "model=openai::gpt-5.5" in text.split("\n")[0]
    assert any("openai::gpt-5.5" in line for line in progress)
    last = reviewer_slot_last_executions()[_DEEP_SLOT_ID]
    assert last["effective"]["model"] == "openai::gpt-5.5"
    assert last["requested"]["profile_id"] == "account-a"


def test_an_uncredentialed_api_row_is_refused_typed_before_any_send(repo, drive, monkeypatch):
    """The direct-route resolution is a fallback, never a bypass: with neither
    OpenRouter nor direct OpenAI credentials the row is the typed refusal."""
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    llm = _ScriptedLLM([{"content": _REPORT}])
    text, usage = run_deep_self_review(repo, drive, llm, lambda _m: None, slot=_bare_row("openai/gpt-5.5"))
    assert text.startswith("❌ Deep self-review unavailable: no OpenRouter or direct OpenAI credentials for openai/gpt-5.5")
    assert usage == {"execution_status": "infra_failed", "reason_code": "deep_self_review_unavailable"}
    assert llm.calls == []
