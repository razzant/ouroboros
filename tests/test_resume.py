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


def test_capture_load_round_trip_default_is_continuation(monkeypatch, tmp_path):
    _capture_env(monkeypatch, tmp_path)
    resume.capture(None, "taskA",
                   [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "remember BANANA-42"}],
                   "ok, BANANA-42 stored")
    # DEFAULT (no resume_mode) = CONTINUATION: original system comes back verbatim
    turns = resume.load_continuation(None, {"resume_from_task_id": "taskA"})
    roles = [t.get("role") for t in turns]
    assert roles == ["system", "user", "assistant"]
    assert turns[0]["content"] == "sys"                          # verbatim
    assert any(t["role"] == "assistant" and "BANANA-42" in str(t["content"]) for t in turns)
    # and the splice loader stays out of the way unless explicitly requested
    assert resume.load_resume_turns(None, {"resume_from_task_id": "taskA"}) == []
    spliced = resume.load_resume_turns(None, {"resume_from_task_id": "taskA",
                                              "resume_mode": "splice"})
    assert [x["role"] for x in spliced] == ["user", "assistant"]  # legacy splice drops system


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
    turns = resume.load_resume_turns(None, {"resume_from_task_id": "t",
                                            "resume_mode": "splice"})
    assert all(t["role"] != "system" for t in turns)


def test_load_marks_cache_breakpoint_at_end(monkeypatch, tmp_path):
    _capture_env(monkeypatch, tmp_path)
    resume.capture(None, "t", [{"role": "user", "content": "u"}], "final")
    turns = resume.load_resume_turns(None, {"resume_from_task_id": "t",
                                            "resume_mode": "splice"})
    last = turns[-1]
    # string content is promoted to a text block carrying the ephemeral breakpoint
    assert isinstance(last["content"], list)
    assert last["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_load_empty_when_no_resume_id(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    assert resume.load_resume_turns(None, {}) == []
    assert resume.load_resume_turns(None, {"resume_from_task_id": ""}) == []
    assert resume.load_continuation(None, {}) == []
    assert resume.load_continuation(None, {"resume_from_task_id": ""}) == []


def test_load_empty_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    assert resume.load_continuation(None, {"resume_from_task_id": "never_captured"}) == []


def test_path_traversal_guard(monkeypatch, tmp_path):
    # H2: resume_from_task_id is attacker-reachable via the unauthenticated task body and is
    # interpolated into a filesystem path — reject anything that could escape the resume dir.
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    for bad in ["../evil", "..", "a/b", "a\\b", ".hidden", "../../etc/passwd"]:
        assert resume.load_continuation(None, {"resume_from_task_id": bad}) == []
        assert resume.load_resume_turns(None, {"resume_from_task_id": bad,
                                               "resume_mode": "splice"}) == []


# ---------------------------------------------------------------- continuation mode

def _mk_capture(tmp_path, tid="tA", msgs=None):
    (tmp_path / "state" / "resume").mkdir(parents=True, exist_ok=True)
    default = [
        {"role": "system", "content": [
            {"type": "text", "text": "STATIC", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "DYNAMIC"},
        ]},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "a1", "cache_control": {"type": "ephemeral"}}]},
    ]
    (tmp_path / "state" / "resume" / f"{tid}.json").write_text(
        json.dumps({"messages": msgs if msgs is not None else default}))


def test_continuation_returns_stored_system_verbatim(monkeypatch, tmp_path):
    # CC --resume analog: the ORIGINAL system message comes back byte-identical (its own
    # cache_control breakpoints preserved) so the prefix is byte-stable => prompt-cache hits.
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _mk_capture(tmp_path)
    turns = resume.load_continuation(None, {"resume_from_task_id": "tA",
                                            "resume_mode": "continuation"})
    assert turns and turns[0]["role"] == "system"
    assert turns[0]["content"][0]["cache_control"] == {"type": "ephemeral"}  # verbatim
    assert turns[0]["content"][0]["text"] == "STATIC"
    # non-system turns: stale breakpoints dropped, ONE fresh one at the end of the prefix
    assert turns[1]["role"] == "user" and turns[2]["role"] == "assistant"
    assert turns[-1]["content"][-1].get("cache_control") == {"type": "ephemeral"}
    mid_blocks = [b for t in turns[1:-1] for b in
                  (t["content"] if isinstance(t["content"], list) else [])]
    assert all("cache_control" not in b for b in mid_blocks if isinstance(b, dict))


def test_mode_selection_continuation_default_splice_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _mk_capture(tmp_path)
    # DEFAULT (no mode) = continuation fires, splice stays out of the way
    assert resume.load_continuation(None, {"resume_from_task_id": "tA"}) != []
    assert resume.load_resume_turns(None, {"resume_from_task_id": "tA"}) == []
    # explicit continuation identical to default
    assert resume.load_continuation(None, {"resume_from_task_id": "tA",
                                           "resume_mode": "continuation"}) != []
    # splice ONLY on explicit opt-in, and it still drops the system turn
    assert resume.load_continuation(None, {"resume_from_task_id": "tA",
                                           "resume_mode": "splice"}) == []
    spliced = resume.load_resume_turns(None, {"resume_from_task_id": "tA",
                                              "resume_mode": "splice"})
    assert spliced and all(t["role"] != "system" for t in spliced)
    # unknown mode falls back LOUDLY to the default (continuation)
    assert resume.load_continuation(None, {"resume_from_task_id": "tA",
                                           "resume_mode": "continuatoin"}) != []


def test_continuation_without_system_falls_back_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _mk_capture(tmp_path, msgs=[{"role": "user", "content": "q"},
                                {"role": "assistant", "content": "a"}])
    assert resume.load_continuation(None, {"resume_from_task_id": "tA",
                                           "resume_mode": "continuation"}) == []


def test_continuation_path_traversal_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    for bad in ["../evil", "..", "a/b", ".hidden"]:
        assert resume.load_continuation(None, {"resume_from_task_id": bad,
                                               "resume_mode": "continuation"}) == []


# ------------------------------------------------------------ note_final_msg hook

def test_note_final_msg_appends_full_msg_when_armed(monkeypatch):
    monkeypatch.setenv("OUROBOROS_RESUME_CAPTURE", "1")
    messages = [{"role": "user", "content": "q"}]
    msg = {"role": "assistant", "content": "final", "tool_calls": None,
           "reasoning_details": [{"type": "reasoning.text", "text": "t", "signature": "s"}],
           "response_id": "gen-123", "cache_control": {"type": "ephemeral"}}
    resume.note_final_msg(messages, msg)
    last = messages[-1]
    assert last["reasoning_details"][0]["signature"] == "s"   # fidelity preserved
    assert last["response_id"] == "gen-123"
    assert "tool_calls" not in last                            # None values dropped
    assert "cache_control" not in last                         # stale breakpoint dropped
    assert last["content"] == "final"


def test_note_final_msg_noop_when_capture_disarmed(monkeypatch):
    monkeypatch.delenv("OUROBOROS_RESUME_CAPTURE", raising=False)
    messages = [{"role": "user", "content": "q"}]
    resume.note_final_msg(messages, {"role": "assistant", "content": "x"})
    assert len(messages) == 1                                  # normal runs untouched


def test_note_final_msg_then_capture_no_duplicate(monkeypatch, tmp_path):
    # loop appends the full final msg via the hook; capture's append-guard must then NOT
    # add a second bare-text assistant turn.
    _capture_env(monkeypatch, tmp_path)
    messages = [{"role": "user", "content": "q"}]
    resume.note_final_msg(messages, {"role": "assistant", "content": "final",
                                     "reasoning": "think"})
    resume.capture(None, "t", messages, "final")
    msgs = json.loads((tmp_path / "state" / "resume" / "t.json").read_text())["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[-1].get("reasoning") == "think"                # full msg won, not bare text


# ------------------------------------------------------------------ chain trim

def test_continuation_trims_oldest_when_over_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_RESUME_MAX_CHARS", "2000")
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(20):  # ~20 x ~120 chars >> 2000 cap
        msgs.append({"role": "user", "content": f"question {i} " + "x" * 80})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "y" * 80})
    _mk_capture(tmp_path, msgs=msgs)
    turns = resume.load_continuation(None, {"resume_from_task_id": "tA"})
    assert turns[0]["role"] == "system" and turns[0]["content"] == "SYS"  # system survives
    assert "omitted to fit the context window" in str(turns[1]["content"])  # explicit note
    joined = str(turns)
    assert "question 19" in joined            # newest kept
    assert "question 0 " not in joined        # oldest dropped
    # under the cap -> untouched, no note
    monkeypatch.setenv("OUROBOROS_RESUME_MAX_CHARS", "600000")
    turns2 = resume.load_continuation(None, {"resume_from_task_id": "tA"})
    assert "omitted" not in str(turns2[1].get("content"))
