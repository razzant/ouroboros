"""Unit tests for the portable conversation-resume module (``ouroboros/resume.py``).

Covers the two seams (capture / load) and the invariants that must not regress:
sanitize blocklist (H1: preserve tool_calls/tool_call_id/name, drop reasoning/cache
metadata), the path-traversal guard (H2), the closing-assistant-turn append, the
prior-system-turn drop, the ephemeral cache breakpoint, and the opt-in / empty-case
no-ops. The module is stdlib-only and reads OUROBOROS_DATA_DIR, so these tests need no
server, LLM, or network.
"""
from __future__ import annotations

import json

from ouroboros import resume


def _capture_env(monkeypatch, tmp_path):
    """Enable capture + point the resume dir at a temp OUROBOROS_DATA_DIR."""
    monkeypatch.setenv("OUROBOROS_RESUME_CAPTURE", "1")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))


def test_capture_load_round_trip(monkeypatch, tmp_path):
    _capture_env(monkeypatch, tmp_path)
    resume.capture(None, "taskA",
                   [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "remember BANANA-42"}],
                   "ok, BANANA-42 stored")
    turns = resume.load_resume_turns(None, {"resume_from_task_id": "taskA"})
    roles = [t.get("role") for t in turns]
    # prior system turn dropped; user + appended closing assistant remain
    assert roles == ["user", "assistant"]
    assert any(t["role"] == "assistant" and "BANANA-42" in str(t["content"]) for t in turns)


def test_capture_appends_closing_assistant_turn(monkeypatch, tmp_path):
    # run_llm_loop returns the final text but does not append it to `messages`;
    # capture must append it so the resumed conversation contains the agent's own turn.
    _capture_env(monkeypatch, tmp_path)
    resume.capture(None, "t", [{"role": "user", "content": "q"}], "final answer")
    path = tmp_path / "state" / "resume" / "t.json"
    msgs = json.loads(path.read_text())["messages"]
    assert msgs[-1] == {"role": "assistant", "content": "final answer"}


def test_capture_no_double_assistant_turn(monkeypatch, tmp_path):
    # If messages already end with an assistant turn, don't append a duplicate.
    _capture_env(monkeypatch, tmp_path)
    resume.capture(None, "t",
                   [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                   "a")
    msgs = json.loads((tmp_path / "state" / "resume" / "t.json").read_text())["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_capture_is_opt_in(monkeypatch, tmp_path):
    # Without OUROBOROS_RESUME_CAPTURE set, capture is a no-op.
    monkeypatch.delenv("OUROBOROS_RESUME_CAPTURE", raising=False)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    resume.capture(None, "t", [{"role": "user", "content": "q"}], "a")
    assert not (tmp_path / "state" / "resume" / "t.json").exists()


def test_sanitize_preserves_reasoning_and_tool_calls_drops_only_cache():
    # Mirror the live-path replay policy: preserve structural keys (tool_calls/name) AND
    # reasoning continuity (reasoning_details); drop ONLY cache_control (stale breakpoints).
    out = resume.sanitize_turn({
        "role": "assistant",
        "content": "x",
        "tool_calls": [{"id": "c1"}],
        "name": "f",
        "reasoning_details": [{"type": "reasoning.text", "text": "plan", "signature": "sig"}],
        "cache_control": {"type": "ephemeral"},
    })
    assert out["tool_calls"] == [{"id": "c1"}]
    assert out["name"] == "f"
    assert out["reasoning_details"][0]["signature"] == "sig"   # reasoning continuity preserved
    assert "cache_control" not in out


def test_sanitize_preserves_thinking_blocks_with_signatures():
    # Same-model replay accepts thinking blocks; Anthropic signatures are cross-provider
    # portable (llm.py live probe). Keep them; strip only the stale cache_control.
    out = resume.sanitize_turn({
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "reasoning...", "signature": "s"},
            {"type": "text", "text": "answer", "cache_control": {"type": "ephemeral"}},
        ],
    })
    kinds = [b.get("type") for b in out["content"]]
    assert kinds == ["thinking", "text"]                       # thinking block kept
    assert out["content"][0]["signature"] == "s"               # with its signature
    assert all("cache_control" not in b for b in out["content"])


def test_load_drops_prior_system_turn(monkeypatch, tmp_path):
    _capture_env(monkeypatch, tmp_path)
    resume.capture(None, "t",
                   [{"role": "system", "content": "old system"},
                    {"role": "user", "content": "u"}],
                   "a")
    turns = resume.load_resume_turns(None, {"resume_from_task_id": "t"})
    assert all(t["role"] != "system" for t in turns)


def test_load_marks_cache_breakpoint_at_end(monkeypatch, tmp_path):
    _capture_env(monkeypatch, tmp_path)
    resume.capture(None, "t", [{"role": "user", "content": "u"}], "final")
    turns = resume.load_resume_turns(None, {"resume_from_task_id": "t"})
    last = turns[-1]
    # string content is promoted to a text block carrying the ephemeral breakpoint
    assert isinstance(last["content"], list)
    assert last["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_load_empty_when_no_resume_id(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    assert resume.load_resume_turns(None, {}) == []
    assert resume.load_resume_turns(None, {"resume_from_task_id": ""}) == []


def test_load_empty_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    assert resume.load_resume_turns(None, {"resume_from_task_id": "never_captured"}) == []


def test_path_traversal_guard(monkeypatch, tmp_path):
    # H2: resume_from_task_id is attacker-reachable via the unauthenticated task body and is
    # interpolated into a filesystem path — reject anything that could escape the resume dir.
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    for bad in ["../evil", "..", "a/b", "a\\b", ".hidden", "../../etc/passwd"]:
        assert resume.load_resume_turns(None, {"resume_from_task_id": bad}) == []
