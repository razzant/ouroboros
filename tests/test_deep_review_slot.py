"""The optional ``deep_review`` reviewer row (Ф3, owner decisions R6/R7).

Deep self-review joins the shared reviewer-row vocabulary as ONE optional
singleton row: absent, the packed api row is synthesized from the legacy model
key ``OUROBOROS_MODEL_DEEP_SELF_REVIEW`` (the invisible migration source), so
every existing install keeps today's exact delivery; present, the row picks the
delivery through the same ``retrieves`` predicate every other surface uses, and
its own effort outranks the surface key.
"""

import asyncio
import json
import ntpath
import os
import types

import pytest

from ouroboros.reviewer_slot_config import (
    DEEP_REVIEW_SLOT_ID,
    REVIEWER_SLOTS_ENV,
    deep_review_slot,
    load_reviewer_slot_config,
    parse_reviewer_slots,
    reviewer_slot_save_check,
    row_effort,
)

_ROSTER = {
    "enabled": True,
    "items": [
        {"subagent_id": "api-critic", "name": "API critic", "recommended_use": "x",
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-terra"}, "effort": "medium"},
        {"subagent_id": "session-critic", "name": "Session critic", "recommended_use": "y",
         "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol",
                   "credential_profile_id": "profile-1"}, "effort": "high"},
    ],
}


def _payload(deep_review=None, **extra):
    body = {
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-luna"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-terra"}}],
        **extra,
    }
    if deep_review is not None:
        body["deep_review"] = deep_review
    return json.dumps(body)


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(_ROSTER))
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL",
                "OUROBOROS_ADVISORY_REVIEW_ROUTE", REVIEWER_SLOTS_ENV):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL_DEEP_SELF_REVIEW", "openai/legacy-deep-model")
    monkeypatch.setenv("OUROBOROS_EFFORT_DEEP_SELF_REVIEW", "low")
    return monkeypatch


def _get_endpoint():
    from starlette.requests import Request

    from ouroboros.gateway.settings import api_reviewer_slots

    request = Request({"type": "http", "method": "GET", "path": "/api/reviewer-slots",
                       "headers": [], "query_string": b""})
    return json.loads(asyncio.run(api_reviewer_slots(request)).body)


def test_deep_review_row_parses_on_the_shared_vocabulary(env):
    """Direct api, direct session (with the manual pin) and a configured-subagent
    reference all parse through the ONE row parser; the identity is fixed."""
    api = parse_reviewer_slots(_payload(
        {"route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-sol-pro"}, "effort": "xhigh"})).deep_review
    assert api.slot_id == DEEP_REVIEW_SLOT_ID and api.kind == "api_chat"
    assert api.target_id == "openai/gpt-5.6-sol-pro" and api.effort == "xhigh"
    assert api.retrieves is False and api.native_retrieval is False

    session = parse_reviewer_slots(_payload(
        {"route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol", "profile_id": "koshak"}})).deep_review
    assert session.is_session and session.session_target == "codex=gpt-5.6-sol"
    assert session.profile_id == "koshak" and session.retrieves is True

    actor = parse_reviewer_slots(_payload({"subagent_id": "api-critic"})).deep_review
    assert actor.subagent_id == "api-critic" and actor.kind == "api_chat"
    assert actor.target_id == "openai/gpt-5.6-terra" and actor.effort == "medium"
    assert actor.native_retrieval is True and actor.retrieves is True

    # Absent is absent — never an empty placeholder row.
    assert parse_reviewer_slots(_payload()).deep_review is None


@pytest.mark.parametrize("row, fragment", [
    ({"route": {"kind": "api_chat", "target_id": "m"}, "slot_id": "mine"}, "unknown keys"),
    ({"route": {"kind": "api_chat", "target_id": "m"}, "enabled": True}, "unknown keys"),
    ({"route": {"kind": "api_chat", "target_id": "m"}, "bogus": 1}, "unknown keys"),
    ({"route": {"kind": "api_chat", "target_id": "m"}, "subagent_id": "api-critic"}, "either route or"),
    ({"subagent_id": "nobody"}, "does not resolve"),
    ({"route": {"kind": "agent_session", "target_id": "off"}}, "concrete harness route"),
    ({"route": {"kind": "api_chat", "target_id": "m"}, "effort": "turbo"}, "unknown effort"),
    ("openai/x", "must be an object"),
])
def test_deep_review_row_refuses_typed_like_every_other_row(env, row, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_reviewer_slots(_payload(row))


def test_deep_review_identity_is_fixed_and_cannot_be_reused_by_another_row(env):
    """The singleton's id lives in the SAME identity space as the other rows:
    a triad row squatting on it collides, so receipts keep ONE history."""
    body = json.loads(_payload({"route": {"kind": "api_chat", "target_id": "m"}}))
    body["triad"][0]["slot_id"] = DEEP_REVIEW_SLOT_ID
    with pytest.raises(ValueError, match="appears twice"):
        parse_reviewer_slots(json.dumps(body))


def test_deep_review_slot_synthesizes_the_packed_row_from_the_model_key(env):
    """No row saved (structured without the key, or legacy comma keys): the
    delivery is today's exact one — a packed api row on the legacy model key,
    NEVER a retrieving row an install did not ask for."""
    for setup in ("structured", "legacy"):
        if setup == "structured":
            env.setenv(REVIEWER_SLOTS_ENV, _payload())
        else:
            env.delenv(REVIEWER_SLOTS_ENV, raising=False)
        row = deep_review_slot()
        assert row.slot_id == DEEP_REVIEW_SLOT_ID and row.kind == "api_chat"
        assert row.target_id == "openai/legacy-deep-model"
        assert row.retrieves is False and row.subagent_id == "" and row.effort == ""
    # A saved row wins over the key.
    env.setenv(REVIEWER_SLOTS_ENV, _payload({"route": {"kind": "api_chat", "target_id": "openai/saved"}}))
    assert deep_review_slot().target_id == "openai/saved"
    # A caller that already parsed the setting hands its config over (no second parse).
    config = load_reviewer_slot_config()
    assert deep_review_slot(config) is config.deep_review


def test_deep_review_row_effort_outranks_the_surface_key_only_when_set(env):
    """R6: the row's effort is the authority when it names one; the synthesized
    row (and a saved row with no effort) keeps the surface key."""
    env.setenv(REVIEWER_SLOTS_ENV, _payload({"route": {"kind": "api_chat", "target_id": "m"}, "effort": "xhigh"}))
    assert row_effort(deep_review_slot(), "deep_self_review") == "xhigh"
    env.setenv(REVIEWER_SLOTS_ENV, _payload({"route": {"kind": "api_chat", "target_id": "m"}}))
    assert row_effort(deep_review_slot(), "deep_self_review") == "low"
    env.setenv(REVIEWER_SLOTS_ENV, _payload())
    assert row_effort(deep_review_slot(), "deep_self_review") == "low"
    # A compound Cursor slug on a session row carries its own effort (shared rule).
    env.setenv(REVIEWER_SLOTS_ENV, _payload({"route": {"kind": "agent_session", "target_id": "cursor=cursor-grok-4.6-xhigh"}}))
    assert row_effort(deep_review_slot(), "deep_self_review") == "xhigh"


def test_malformed_deep_review_refuses_the_whole_setting(env):
    """The parser is ONE authority: a bad deep_review row is a save-time 400 and
    a runtime typed error, never a silent fallback onto the model key."""
    bad = _payload({"route": {"kind": "api_chat", "target_id": "m"}, "bogus": 1})
    with pytest.raises(ValueError, match="deep_review has unknown keys"):
        reviewer_slot_save_check(bad)
    env.setenv(REVIEWER_SLOTS_ENV, bad)
    with pytest.raises(ValueError, match="deep_review has unknown keys"):
        deep_review_slot()
    # A valid row passes the save check (and produces no acceptance warning).
    assert reviewer_slot_save_check(_payload({"subagent_id": "session-critic"})) == ""


def test_reviewer_slots_endpoint_reports_the_deep_review_row_and_its_limit(env):
    env.setenv(REVIEWER_SLOTS_ENV, _payload())
    body = _get_endpoint()
    assert body["limits"]["deep_review"] == 1
    # Synthesized: the effective row is shown AND labeled as not saved yet.
    assert body["deep_review"] == {
        "route": {"kind": "api_chat", "target_id": "openai/legacy-deep-model"},
        "effort": "",
        "synthesized_from": "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    }
    # Saved direct session row: the stored form round-trips with its pin, unlabeled.
    env.setenv(REVIEWER_SLOTS_ENV, _payload(
        {"route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol", "profile_id": "koshak"}, "effort": "high"}))
    body = _get_endpoint()
    assert body["deep_review"] == {
        "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol", "profile_id": "koshak"},
        "effort": "high",
    }
    # Saved reference: the subagent_id IS the stored form; the route is disclosure only.
    env.setenv(REVIEWER_SLOTS_ENV, _payload({"subagent_id": "api-critic"}))
    row = _get_endpoint()["deep_review"]
    assert row["subagent_id"] == "api-critic" and "route" not in row and "slot_id" not in row
    assert row["resolved_route"] == {"kind": "api_chat", "target_id": "openai/gpt-5.6-terra"}
    # Unconfigured install: the synthesized row is reported the same way.
    # ABI 7.0 (ABI-10) retired the comma-list "legacy" source, so an install
    # without the structured key reports the shipped default panel instead.
    env.delenv(REVIEWER_SLOTS_ENV, raising=False)
    body = _get_endpoint()
    assert body["source"] == "default"
    assert body["deep_review"]["synthesized_from"] == "OUROBOROS_MODEL_DEEP_SELF_REVIEW"


# ---------------------------------------------------------------------------
# The three deliveries of ``run_deep_self_review`` on the row.
# ---------------------------------------------------------------------------

import copy  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from unittest import mock  # noqa: E402

from ouroboros import deep_self_review  # noqa: E402
from ouroboros.deep_self_review import (  # noqa: E402
    _REPORT_CONTRACT,
    _SYSTEM_PROMPT,
    deep_review_route,
    run_deep_self_review,
)
from ouroboros.review_execution import ReviewAttemptResult, ReviewRouteKind, ReviewRouteUnavailable  # noqa: E402
from ouroboros.reviewer_slot_config import ConfiguredReviewerSlot, reviewer_slot_last_executions  # noqa: E402

# The packed system prompt of the pre-row deep review (v6.114.0): the packed
# delivery's wire payload is byte-identical after the row landed.
_PACKED_PROMPT_SHA256 = "1bc81d4cde90757119d1cefd76c863232cf5c234aa0eef1568781149ac9e9aa5"


def test_packed_system_prompt_is_byte_identical_to_the_pre_row_review():
    assert hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest() == _PACKED_PROMPT_SHA256


class _ScriptedLLM:
    """chat() replays a script; captures every messages payload it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs.get("messages"))})
        if not self.script:
            raise AssertionError("script exhausted — the executor made an extra call")
        return self.script.pop(0), {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}


def _tool_call(name, args, call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


_BIBLE = "# BIBLE\n\n## Principle 0: Agency\n\nOuroboros is a becoming personality.\n" * 3
_REPORT = "Read: BIBLE.md in full; memory inline.\n\n# Deep self-review\n\nCRITICAL: loop.py finalization race.\n"


@pytest.fixture()
def review_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "BIBLE.md").write_text(_BIBLE, encoding="utf-8")
    (repo / "docs" / "ARCHITECTURE.md").write_text("# Arch\n\n## Review stack\n\ntext\n\n#### Deep self-review\n\nmore\n", encoding="utf-8")
    (repo / "docs" / "DEVELOPMENT.md").write_text("# Dev\n\n## Rules\n\nx\n", encoding="utf-8")
    (repo / "docs" / "CHECKLISTS.md").write_text("# Checks\n\n## Repo Commit Checklist\n\ny\n", encoding="utf-8")
    (repo / "ouroboros").mkdir()
    (repo / "ouroboros" / "loop.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    return repo


@pytest.fixture()
def review_drive(tmp_path):
    drive = tmp_path / "drive"
    (drive / "memory" / "knowledge").mkdir(parents=True)
    (drive / "memory" / "identity.md").write_text("I am Ouroboros.\n", encoding="utf-8")
    (drive / "memory" / "scratchpad.md").write_text("Working notes.\n", encoding="utf-8")
    (drive / "memory" / "knowledge" / "patterns.md").write_text("## Patterns\n- class A\n", encoding="utf-8")
    (drive / "state").mkdir()
    (drive / "logs").mkdir()
    return drive


def _row(kind="api_chat", target="openai/fake-deep", **fields):
    return ConfiguredReviewerSlot(slot_id=DEEP_REVIEW_SLOT_ID, kind=kind, target_id=target, **fields)


def _native_row():
    return _row(subagent_id="api-critic")


def _session_row():
    return _row("agent_session", "codex=gpt-5.6-sol", session_target="codex=gpt-5.6-sol", profile_id="koshak")


def test_native_row_runs_the_inspection_episode_over_repo_and_memory(review_repo, review_drive, monkeypatch):
    """A configured-subagent api row is a NATIVE episode through the shared
    executor seam: the task carries the role prompt, the memory whitelist
    inline byte-exact, BIBLE.md as a mandatory read and the governance
    navigation maps; the data plane is the REAL runtime root; the report
    comes back behind the host header with host-observed coverage."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md"}, "c1")]},
        {"tool_calls": [_tool_call("read_file", {"path": "memory/identity.md", "root": "runtime_data"}, "c2")]},
        {"content": _REPORT},
    ])
    progress = []
    text, usage = run_deep_self_review(review_repo, review_drive, llm, progress.append,
                                       task_id="dsr-1", slot=_native_row())
    header, body = text.split("\n\n", 1)
    assert body == _REPORT
    assert header.startswith(
        "<!-- deep-review provenance: delivery=native_tool_rounds, model=openai/fake-deep, memory=3/7, "
        "memory_missing=registry.md,WORLD.md,index-full.md,improvement-backlog.md, "
        "coverage=BIBLE.md:read, incomplete=none, attestation=host_observed, rounds=3, tool_calls=2, receipts=2, "
        "end_reason=final_answer, transcript=")
    assert "_Deep self-review: native inspection episode on openai/fake-deep — 3 rounds, 2 tool calls" in header
    assert "BIBLE.md read in full; memory 3/7 inlined (omitted: registry.md missing, WORLD.md missing, index-full.md missing, improvement-backlog.md missing); complete_" in header
    assert usage["deep_review_memory"]["inlined"] == 3 and usage["deep_review_memory"]["dispositions"]["memory/WORLD.md"] == "missing"
    assert usage["native_rounds"] == 3 and usage["host_file_read_attestation"] == "host_observed"
    assert usage["resolved_model"] == "openai/fake-deep" and "execution_status" not in usage
    assert not [d for d in usage.get("capability_delta", []) if str(d.get("reason", "")).startswith("deep_review_")]
    # The episode task: role prompt + method, memory inline byte-exact, BIBLE
    # as a sized mandatory read, nav maps, and the REPORT contract (never the
    # JSON array the executors fall back to).
    first = llm.calls[0]["messages"]
    task = next(m["content"] for m in first if m["role"] == "user")
    assert "deep self-review of the Ouroboros project" in task
    assert "`BIBLE.md` IN FULL first" in task and f"about {len(_BIBLE):,} chars" in task
    assert "## FILE: drive/memory/identity.md\nI am Ouroboros.\n" in task
    assert "## FILE: drive/memory/knowledge/patterns.md\n## Patterns\n- class A\n" in task
    assert "Memory dispositions (7 whitelisted): memory/identity.md inlined; memory/scratchpad.md inlined; memory/registry.md missing" in task
    assert "docs/ARCHITECTURE.md (navigation map)" in task and "Deep self-review" in task
    assert "Deliver the report itself as plain markdown prose" in task
    assert "Begin with one line naming what you read" in task
    assert "JSON array" not in task
    # The REAL data root is the episode's data plane (R5): memory is readable.
    tool_msgs = [m for m in llm.calls[2]["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c1" and "becoming personality" in tool_msgs[0]["content"]
    assert tool_msgs[1]["tool_call_id"] == "c2" and "I am Ouroboros." in tool_msgs[1]["content"]
    # «Выполняется как» for the deep-review row, and the progress names the delivery.
    last = reviewer_slot_last_executions()[DEEP_REVIEW_SLOT_ID]
    assert last["surface"] == "deep_self_review" and last["status"] == "responded"
    assert last["requested"]["subagent_id"] == "api-critic" and last["effective"]["model"] == "openai/fake-deep"
    assert any("native_tool_rounds" in line for line in progress)


def test_native_row_missing_mandatory_read_is_disclosed_not_refused(review_repo, review_drive, monkeypatch):
    """R8: a native episode that never opened BIBLE.md still delivers — with
    the miss in the header, a typed capability_delta, and the runs-as row."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "ouroboros/loop.py"}, "c1")]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    assert text.endswith(_REPORT)
    assert "coverage=BIBLE.md:missing" in text and "BIBLE.md NOT read; memory 3/7 inlined" in text and text.split("\n")[1].endswith("; complete_")
    deltas = [d for d in usage["capability_delta"] if d["reason"] == "deep_review_mandatory_read_missing"]
    assert deltas and deltas[0]["requested"] == "mandatory full read of BIBLE.md"
    assert reviewer_slot_last_executions()[DEEP_REVIEW_SLOT_ID]["capability_delta"] == usage["capability_delta"]

    def cov(receipts, calls=None):
        return deep_self_review._native_read_coverage(
            {"native_tool_calls": len(receipts) if calls is None else calls, "native_tool_receipts": receipts}, review_repo)["BIBLE.md"]

    def rec(path, start, end, total, root="", outcome="executed", opened=None):
        # Shaped like a real receipt: `opened_path` is what the reader opened
        # (the raw `path` is the model's spelling; a receipt that rendered
        # nothing has no opened path).
        return {"tool": "read_file", "path": path, "root": root, "outcome": outcome,
                "start_line": start, "end_line": end, "total_lines": total,
                "opened_path": path if opened is None else opened}

    # Full coverage only when the merged intervals cover the whole file.
    assert cov([rec("BIBLE.md", 1, 12, 12)])["state"] == "read"
    two = cov([rec("BIBLE.md", 1, 6, 12), rec("BIBLE.md", 7, 12, 12)])
    assert two["state"] == "read" and two["covered_lines"] == 12
    one = cov([rec("BIBLE.md", 1, 1, 12)])
    assert one["state"] == "partial" and one["fraction"] == round(1 / 12, 3)
    overlap = cov([rec("BIBLE.md", 1, 6, 12), rec("BIBLE.md", 1, 6, 12), rec("BIBLE.md", 3, 8, 12)])
    assert overlap["state"] == "partial" and overlap["covered_lines"] == 8
    # Coverage folds on the OPENED path, never on the model's spelling: every
    # spelling the REAL registry reads as BIBLE.md (absolute in-repo, whitespace-
    # padded, redundant `repo/` prefix, root-qualified `/`, dot-prefixed) credits
    # `read` — one scripted episode per spelling, through the real registry.
    total = len(_BIBLE.splitlines())
    for spelled in (str(review_repo / "BIBLE.md"), " BIBLE.md", "repo/BIBLE.md", "/BIBLE.md", "./BIBLE.md", "BIBLE.md"):
        llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": spelled}, "c1")]}, {"content": _REPORT}])
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
        receipt = usage["native_tool_receipts"][0]
        assert (receipt["path"], receipt["opened_path"], receipt["eof"], receipt["total_lines"]) == (spelled, "BIBLE.md", True, total), receipt
        assert "coverage=BIBLE.md:read" in text and "BIBLE.md read in full" in text, (spelled, text.split("\n")[0])
        assert not [d for d in usage.get("capability_delta", []) if d["reason"].startswith("deep_review_")]
    # ...and on the OPENED root: a padded root spelling the registry reads as a
    # repository root credits `read` (the raw `root` stays the model's spelling).
    for root in (" system_repo ", "active_workspace ", "system_repo"):
        llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md", "root": root}, "c1")]}, {"content": _REPORT}])
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
        receipt = usage["native_tool_receipts"][0]
        assert (receipt["root"], receipt["opened_root"], receipt["opened_path"], receipt["eof"]) == (root, root.strip(), "BIBLE.md", True), receipt
        assert "coverage=BIBLE.md:read" in text, (root, text.split("\n")[0])
        assert not [d for d in usage.get("capability_delta", []) if d["reason"].startswith("deep_review_")]
    # A receipt without an opened path is matched by its raw spelling, where a
    # `..` component names nothing (refused before dispatch, nothing rendered).
    assert cov([{"tool": "read_file", "path": "a/../BIBLE.md", "root": "", "outcome": "executed"}])["state"] == "missing"
    # A data-plane read of a same-named file is NOT the repository read.
    assert cov([rec("BIBLE.md", 1, 12, 12, root="runtime_data")])["state"] == "missing"
    # Refused / withheld reads are not reads; capped receipts or a receipt
    # without an extent make absence `unobserved`, never `missing`.
    assert cov([rec("BIBLE.md", 1, 12, 12, outcome="withheld")])["state"] == "missing"
    assert cov([rec("ouroboros/loop.py", 1, 2, 2)], calls=5)["state"] == "unobserved"
    assert cov([{"tool": "read_file", "path": "BIBLE.md", "root": "", "outcome": "executed"}])["state"] == "unobserved"
    assert cov([rec("BIBLE.md", 1, 6, 12)], calls=3)["state"] == "unobserved"  # partial AND capped: the tail may hold the rest
    # One measured receipt beside one extent-less receipt: the unmeasured one may
    # hold the rest — `unobserved`, never `partial` (full coverage must be PROVEN).
    assert cov([rec("BIBLE.md", 1, 6, 12), {"tool": "read_file", "path": "BIBLE.md", "root": "", "outcome": "executed"}])["state"] == "unobserved"
    assert cov([rec("BIBLE.md", 1, 12, 12), {"tool": "read_file", "path": "BIBLE.md", "root": "", "outcome": "executed"}])["state"] == "read"
    # A measured EMPTY delivery (cursor past the window, start past EOF) delivered
    # nothing of the file: `missing`, never an inverted claim.
    assert cov([rec("BIBLE.md", 13, 12, 12)])["state"] == "missing"
    assert cov([rec("BIBLE.md", 13, 12, 12), rec("BIBLE.md", 1, 3, 12)])["state"] == "partial"


def test_native_row_exhaustion_delivers_the_draft_marked_incomplete(review_repo, review_drive, monkeypatch):
    """R13/Ф1: the report shape delivers the collected draft when the
    transcript bound lands first; the header says INCOMPLETE and why."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (review_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    draft = "# Deep self-review (draft)\n\nCRITICAL: loop.py finalization race.\n"
    script = [{"content": draft, "tool_calls": [_tool_call("read_file", {"path": "big.txt"}, "c1")]}] + [
        {"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md"}, f"c{i}")]} for i in range(2, 40)
    ]
    llm = _ScriptedLLM(script)
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    assert text.endswith("\n\n" + draft)
    assert "incomplete=transcript_bound" in text and "INCOMPLETE (transcript_bound)_" in text
    assert usage["native_incomplete"] == "transcript_bound" and usage["native_end_reason"] == "transcript_bound"
    assert any(d["reason"] == "native_transcript_bound_before_final_answer" for d in usage["capability_delta"])
    assert "execution_status" not in usage  # a partial product is a product, not a failure
    assert llm.script  # the bound landed before the script ran out


class _FakeSessionExecutor:
    """Stands in for the session executor at the ONE transport seam."""

    instances: list = []

    def __init__(self, assignment, *, llm=None):
        self.assignment = assignment
        self.llm = llm
        type(self).instances.append(self)

    def prompt_payload(self):
        return {"session_prompt": "p"}

    def failure_custody(self):
        return {"delegated_run_started": False, "delegated_run_id": "", "pending_invocation_id": ""}

    def execute(self):
        return ReviewAttemptResult(
            message={"content": _REPORT, "session_transcript": _REPORT, "delegated_run_id": "run-1", "verdict_method": "report"},
            usage={"provider": "claudexor", "resolved_model": "gpt-5.6-sol", "delegated_route": "codex",
                   "delegated_run_id": "run-1", "verdict_method": "report", "cost_disclosed_usd": 0.4},
            raw_text=_REPORT,
        )


def test_session_row_runs_through_the_session_executor_with_the_report_contract(review_repo, review_drive, monkeypatch):
    """An agent_session row: the same hand-built request rides the seam, the
    report contract and the real data root travel in the policy, the slot
    carries the row's target/pin and an explicit logical window narrowed by
    the owner deadline, and coverage is honestly `unobserved`."""
    import ouroboros.review_execution as review_execution

    _FakeSessionExecutor.instances = []
    monkeypatch.setattr(review_execution, "_review_route_executor", _FakeSessionExecutor)
    monkeypatch.setattr(deep_self_review, "_session_route_reason", lambda row: "")
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat()
    before = time.monotonic()
    text, usage = run_deep_self_review(review_repo, review_drive, object(), lambda _m: None,
                                       task_id="dsr-2", deadline_at=deadline, slot=_session_row())
    assert text.endswith(_REPORT)
    assert text.startswith(
        "<!-- deep-review provenance: delivery=agent_session, model=gpt-5.6-sol, memory=3/7, "
        "memory_missing=registry.md,WORLD.md,index-full.md,improvement-backlog.md, "
        "coverage=BIBLE.md:unobserved, incomplete=unobserved, attestation=unobserved -->\n"
        "_Deep self-review: agent session codex=gpt-5.6-sol (model gpt-5.6-sol) — reads not host-observed "
        "(coverage unobserved); memory 3/7 inlined (omitted: registry.md missing, WORLD.md missing, index-full.md missing, "
        "improvement-backlog.md missing); completeness not host-observed_\n\n")
    # A session carries NO round/receipt facts — by construction, not by key absence.
    comment = text.split("\n", 1)[0]
    assert "rounds=" not in comment and "receipts=" not in comment and "tool_calls=" not in comment
    executor = _FakeSessionExecutor.instances[0]
    request, slot = executor.assignment.request, executor.assignment.slot
    assert request.surface == "deep_self_review" and request.session_root == str(review_repo)
    assert request.policy["output_contract"] is _REPORT_CONTRACT
    assert request.policy["native_data_root"] == str(review_drive)
    assert request.max_tokens == 100_000 and request.no_proxy is True and request.deadline_at == deadline
    assert "## FILE: drive/memory/identity.md\nI am Ouroboros.\n" in request.session_task
    assert slot.route is ReviewRouteKind.AGENT_SESSION and slot.session_target == "codex=gpt-5.6-sol"
    assert slot.session_profile == "koshak" and slot.slot_id == DEEP_REVIEW_SLOT_ID
    assert slot.max_tokens == 100_000 and slot.role_hint == "deep self-reviewer"
    # The logical window: the task ceiling narrowed by the owner deadline (never the 300 s default).
    assert 0 < slot.timeout_sec < 600
    assert before < executor._logical_deadline_monotonic < before + 600
    assert executor.assignment.custody_root == review_drive and executor.assignment.call_type == "deep_self_review"
    last = reviewer_slot_last_executions()[DEEP_REVIEW_SLOT_ID]
    assert last["effective"] == {"route": "agent_session:codex", "model": "gpt-5.6-sol", "verdict_method": "report"}
    assert last["requested"]["session_target"] == "codex=gpt-5.6-sol" and last["requested"]["profile_id"] == "koshak"
    # Without an owner deadline the window is the task's absolute ceiling.
    _FakeSessionExecutor.instances = []
    from ouroboros.config import get_task_abs_ceiling_sec
    run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=_session_row())
    assert _FakeSessionExecutor.instances[0].assignment.slot.timeout_sec == float(get_task_abs_ceiling_sec())


def test_retrieving_failure_is_typed_and_recorded_never_a_report(review_repo, review_drive, monkeypatch):
    import ouroboros.review_execution as review_execution

    class _Refusing(_FakeSessionExecutor):
        def execute(self):
            raise ReviewRouteUnavailable("delegated review route unavailable: route_disabled", code="route_disabled")

    monkeypatch.setattr(review_execution, "_review_route_executor", _Refusing)
    monkeypatch.setattr(deep_self_review, "_session_route_reason", lambda row: "")
    text, usage = run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=_session_row())
    assert text.startswith("❌ Deep self-review failed: ReviewRouteUnavailable: delegated review route unavailable")
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "deep_self_review_error"
    # The typed failure usage carries the memory fact and the executor's failure
    # custody — the same usage the «Выполняется как» error row was recorded from.
    assert usage["deep_review_memory"]["total"] == 7 and usage["delegated_run_started"] is False
    last = reviewer_slot_last_executions()[DEEP_REVIEW_SLOT_ID]
    assert last["status"] == "error" and last["surface"] == "deep_self_review"


def test_memory_fact_precedes_every_runs_as_record_and_rides_the_returned_usage(review_repo, review_drive, monkeypatch):
    """Round 3, ONE class: on all three retrieving paths — responded, empty
    response, executor exception — the usage handed to the «Выполняется как»
    record carries `deep_review_memory`, and so does the usage the caller
    receives (a typed failure included). The durable D22 projection persists
    route/model/status/capability_delta and typed failure facts ONLY: the
    memory fact is intentionally absent there (no deep-review-only field on a
    cross-surface SSOT) — its durable disclosure is the header and the usage."""
    import ouroboros.review_execution as review_execution

    class _Empty(_FakeSessionExecutor):
        def execute(self):
            return ReviewAttemptResult(message={"content": " "}, usage={"resolved_model": "gpt-5.6-sol"}, raw_text=" ")

    class _Boom(_FakeSessionExecutor):
        def execute(self):
            raise RuntimeError("socket reset")

    recorded = []
    real_record = deep_self_review._record_execution

    def spy(slot, usage, *, status, error=""):
        recorded.append((status, dict(usage)))
        real_record(slot, usage, status=status, error=error)

    monkeypatch.setattr(deep_self_review, "_record_execution", spy)
    monkeypatch.setattr(deep_self_review, "_session_route_reason", lambda row: "")
    for executor_cls, status, prefix in (
        (_FakeSessionExecutor, "responded", "<!-- deep-review provenance"),
        (_Empty, "error", "⚠️ Model returned an empty response"),
        (_Boom, "error", "❌ Deep self-review failed: RuntimeError: socket reset"),
    ):
        recorded.clear()
        monkeypatch.setattr(review_execution, "_review_route_executor", executor_cls)
        text, usage = run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=_session_row())
        assert text.startswith(prefix), (executor_cls.__name__, text[:80])
        assert [s for s, _ in recorded] == [status]
        handed = recorded[0][1]
        assert handed["deep_review_memory"]["total"] == 7 and handed["deep_review_memory"]["inlined"] == 3
        assert usage["deep_review_memory"] == handed["deep_review_memory"]
        if status == "error":
            assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "deep_self_review_error"
        last = reviewer_slot_last_executions()[DEEP_REVIEW_SLOT_ID]
        assert last["status"] == status and last["surface"] == "deep_self_review"
        assert last["requested"]["session_target"] == "codex=gpt-5.6-sol" and "capability_delta" in last
        assert "deep_review_memory" not in json.dumps(last)  # intentionally absent from the durable projection


def test_availability_follows_the_row_not_the_model_key(env, monkeypatch):
    """Route-aware availability (`deep_review_route`, the ONE availability
    reader — agent, tool and runner all call it): the packed row keeps the
    ≥1M/OPENAI_BASE_URL rule, a native row needs its model's credentials, a
    session row needs a healthy delegated route (the substrate's own reader),
    and a malformed setting is the typed reason — never a fallback onto the key."""
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    reason, identity = deep_review_route(_row())
    assert reason.startswith("no OpenRouter or direct OpenAI credentials for openai/fake-deep") and identity is None
    assert deep_review_route(_native_row())[0] == "no provider credentials for openai/fake-deep"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert deep_review_route(_row()) == ("", "openai/fake-deep")
    assert deep_review_route(_native_row()) == ("", "openai/fake-deep")

    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagents as subagents

    class _Gateway:
        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda **_k: _Gateway())
    health = {"answer": ("", "")}
    seen = []

    def _route_health(gateway, route_id, shape, *, route_model="", pinned_profile=""):
        seen.append((route_id, route_model, pinned_profile))
        return health["answer"]

    monkeypatch.setattr(subagents, "route_health", _route_health)
    assert deep_review_route(_session_row()) == ("", "codex=gpt-5.6-sol")
    assert seen == [("codex", "gpt-5.6-sol", "koshak")]  # the pin narrows the quota judgement
    health["answer"] = ("subscription_window_exhausted", "2026-09-03T00:00:00Z")
    assert deep_review_route(_session_row()) == ("subscription_window_exhausted", None)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda **_k: (_ for _ in ()).throw(RuntimeError("daemon down")))
    assert deep_review_route(_session_row())[0].startswith("agent_service_unavailable: RuntimeError")
    assert deep_review_route(_row("agent_session", "=bad", session_target="=bad")) == ("session_target_unparsable", None)
    # Malformed structured setting: the parser's typed text is the reason.
    env.setenv(REVIEWER_SLOTS_ENV, _payload({"route": {"kind": "api_chat", "target_id": "m"}, "bogus": 1}))
    reason, identity = deep_review_route()
    assert "deep_review has unknown keys" in reason and identity is None


def test_unavailable_row_never_runs_and_returns_typed_usage(review_repo, review_drive, monkeypatch, env):
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    llm = mock.Mock()
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
    assert text.startswith("❌ Deep self-review unavailable: no OpenRouter or direct OpenAI credentials for openai/fake-deep")
    assert usage == {"execution_status": "infra_failed", "reason_code": "deep_self_review_unavailable"}
    assert not llm.chat.called
    env.setenv(REVIEWER_SLOTS_ENV, _payload({"route": {"kind": "api_chat", "target_id": "m"}, "bogus": 1}))
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None)
    assert "deep_review has unknown keys" in text and usage["reason_code"] == "deep_self_review_unavailable"


def test_packed_row_keeps_the_wire_shape_and_records_its_execution(review_repo, review_drive, monkeypatch):
    """The packed delivery is unchanged on the wire — two messages, the golden
    system prompt, tools=None, the 100K output reserve — and now also leaves
    its runs-as row like every other reviewer row."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OUROBOROS_EFFORT_DEEP_SELF_REVIEW", "high")
    llm = mock.Mock()
    llm.chat.return_value = ({"content": "Review result."}, {"cost": 0.02, "prompt_tokens": 10})
    pack = "y" * 400
    with mock.patch.object(deep_self_review, "build_review_pack",
                           return_value=(pack, {"file_count": 3, "total_chars": len(pack), "skipped": [], "context_manifest": {"ok": 1}})):
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None,
                                           slot=_row(effort="xhigh"))
    kwargs = llm.chat.call_args.kwargs
    assert kwargs["messages"] == [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": pack}]
    assert kwargs["model"] == "openai/fake-deep" and kwargs["tools"] is None and kwargs["temperature"] is None
    assert kwargs["max_tokens"] == 100_000 and kwargs["no_proxy"] is True
    assert kwargs["reasoning_effort"] == "xhigh"  # the row's effort outranks the surface key (R6)
    assert text == (
        "<!-- deep-review provenance: delivery=api_packet, model=openai/fake-deep, memory=0/7, "
        "coverage=pack:3_files, incomplete=none, attestation=packed, window=assumed_1000000 -->\n"
        "_Deep self-review: one packed API review on openai/fake-deep — 3 files; memory 0/7 inlined; "
        "window 1,000,000 (unknown, full window assumed); complete_\n\n"
        "Review result."
    )
    assert usage["deep_review_memory"] == {"inlined": 0, "total": 7, "dispositions": {}}  # the mocked pack carried no memory fact
    assert usage["resolved_model"] == "openai/fake-deep" and usage["cost"] == 0.02
    last = reviewer_slot_last_executions()[DEEP_REVIEW_SLOT_ID]
    assert last["effective"]["route"] == "api_chat" and last["effective"]["model"] == "openai/fake-deep"
    assert last["requested"]["effort"] == "xhigh" and last["status"] == "responded"


def test_agent_keeps_the_previous_report_when_the_review_fails(tmp_path, monkeypatch):
    """`memory/deep_review.md` is overwritten ONLY by a delivered report: a
    typed failure goes to the task result and a typed `task_error` event."""
    import ouroboros.agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent

    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive"
    (drive / "memory").mkdir(parents=True)
    (drive / "logs").mkdir()
    (drive / "memory" / "deep_review.md").write_text("PREVIOUS REPORT", encoding="utf-8")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **_k: ([], {}))
    monkeypatch.setattr(agent_module, "emit_task_results", lambda *_a, **_k: None)
    outcome = {"value": ("❌ Deep self-review unavailable: no provider credentials for openai/x. Configure …",
                         {"execution_status": "infra_failed", "reason_code": "deep_self_review_unavailable"})}
    monkeypatch.setattr(deep_self_review, "run_deep_self_review", lambda **_k: outcome["value"])
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    task = {"id": "dsr-agent", "type": "deep_self_review", "chat_id": 1, "text": "owner:/review",
            "metadata": {"deadline_at": "2099-01-01T00:00:00+00:00"}}
    events = agent.handle_task(task)
    assert (drive / "memory" / "deep_review.md").read_text(encoding="utf-8") == "PREVIOUS REPORT"
    rows = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = [r for r in rows if r.get("type") == "task_error" and r.get("task_id") == "dsr-agent"]
    assert errors and errors[0]["reason_code"] == "deep_self_review_unavailable"
    assert any(e.get("type") == "llm_usage" and e.get("category") == "deep_self_review" for e in events)
    # A delivered report overwrites it, and the deadline reaches the review.
    seen = {}

    def _ok(**kwargs):
        seen.update(kwargs)
        return "<!-- deep-review provenance: delivery=api_packet -->\n_x_\n\nNEW REPORT", {"resolved_model": "openai/x", "cost": 0.0}

    monkeypatch.setattr(deep_self_review, "run_deep_self_review", _ok)
    events = agent.handle_task(task)
    assert (drive / "memory" / "deep_review.md").read_text(encoding="utf-8").endswith("NEW REPORT")
    assert seen["task_id"] == "dsr-agent" and seen["deadline_at"] == "2099-01-01T00:00:00+00:00"
    usage_events = [e for e in events if e.get("type") == "llm_usage"]
    assert usage_events and usage_events[0]["model"] == "openai/x"


# ---------------------------------------------------------------------------
# Fix batch №1 — optional-key wire and repair save (items 1, 10).
# ---------------------------------------------------------------------------


def test_endpoint_carries_the_synthesized_row_beside_a_config_error(env):
    """A malformed structured value must not blank the deep-review editor: the
    legacy-derived REPAIR PLACEHOLDER (the row synthesized from the model key,
    labeled `synthesized_from`) rides beside the typed config_error so the
    repair save starts from a real row — it is NOT an effective row: none is
    effective until the setting is repaired (`deep_review_slot()` raises)."""
    env.setenv(REVIEWER_SLOTS_ENV, "{broken")
    body = _get_endpoint()
    assert "not valid JSON" in body["config_error"]
    assert body["deep_review"] == {
        "route": {"kind": "api_chat", "target_id": "openai/legacy-deep-model"},
        "effort": "",
        "synthesized_from": "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    }
    assert "triad" not in body  # the rows themselves are still unparseable


def test_repair_save_without_the_optional_key_succeeds_and_an_emptied_target_is_refused(env):
    """The optional key absent on the wire = the runtime synthesizes the row;
    an EXPLICITLY emptied api target is the typed 400 (owner fork 3 = A).

    Exercised at the save-check seam the POST handler calls
    (`_check_reviewer_slots_against_incoming_roster`) — never through a real
    `POST /api/settings`, whose apply rebinds the PROCESS-WIDE settings
    authority (`config.SETTINGS_PATH`, bound session-wide by conftest) and
    leaked this test's roster into later tests in the same worker."""
    from ouroboros.gateway.settings import _check_reviewer_slots_against_incoming_roster

    env.setenv(REVIEWER_SLOTS_ENV, "{broken")  # the stored value is malformed (config_error)
    # Repair: a valid value WITHOUT deep_review passes the boundary check (no
    # warning, no refusal) and the singleton stays synthesized from the key.
    assert _check_reviewer_slots_against_incoming_roster({REVIEWER_SLOTS_ENV: _payload()}) == ""
    assert deep_review_slot(parse_reviewer_slots(_payload())).target_id == "openai/legacy-deep-model"
    # Explicitly emptied target: typed refusal at the parser and at the boundary seam.
    emptied = _payload({"route": {"kind": "api_chat", "target_id": ""}})
    with pytest.raises(ValueError, match="deep_review route.target_id"):
        reviewer_slot_save_check(emptied)
    with pytest.raises(ValueError, match="deep_review route.target_id"):
        _check_reviewer_slots_against_incoming_roster({REVIEWER_SLOTS_ENV: emptied})
    # An explicit CLEAR of the setting is a clear, not a validation subject.
    assert _check_reviewer_slots_against_incoming_roster({REVIEWER_SLOTS_ENV: ""}) == ""


# ---------------------------------------------------------------------------
# Fix batch №1 — provenance truthfulness (items 11, 13, 15, 23).
# ---------------------------------------------------------------------------


def test_header_sanitizes_hostile_values_and_builds_session_facts_by_construction(review_repo, review_drive, monkeypatch):
    """Item 13: a resolved model carrying `-->` and a newline cannot close the
    comment or break the line; a session's fact set has no rounds/receipts and
    `attestation=unobserved` by construction; long values are bounded."""
    import ouroboros.review_execution as review_execution

    hostile = "gpt-->\ninjected --> " + "x" * 300

    class _Hostile(_FakeSessionExecutor):
        def execute(self):
            result = super().execute()
            usage = dict(result.usage, resolved_model=hostile, native_rounds=9, native_tool_receipts=[{"tool": "read_file"}])
            return ReviewAttemptResult(message=result.message, usage=usage, raw_text=result.raw_text)

    monkeypatch.setattr(review_execution, "_review_route_executor", _Hostile)
    monkeypatch.setattr(deep_self_review, "_session_route_reason", lambda row: "")
    text, _usage = run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=_session_row())
    comment, human = text.split("\n")[0], text.split("\n")[1]
    assert comment.startswith("<!-- deep-review provenance: ") and comment.endswith(" -->")
    assert comment.count("-->") == 1 and "\n" not in comment
    assert "model=gpt-> injected -> xxxx" in comment and "OMISSION NOTE" in comment  # bounded, disclosed
    assert "rounds=" not in comment and "receipts=" not in comment and "attestation=unobserved" in comment
    assert human.startswith("_") and human.endswith("_") and "\n" not in human
    # The HUMAN line is bounded and sanitized too: the hostile model rides it
    # through `_header_value` (no comment terminator, bounded with the disclosed marker).
    assert "-->" not in human and "OMISSION NOTE" in human and "x" * 121 not in human
    # A hostile session TARGET on the row is bounded the same way.
    hostile_row = ConfiguredReviewerSlot(slot_id=DEEP_REVIEW_SLOT_ID, kind="agent_session",
                                         target_id="codex=" + "t" * 200 + "-->\nx", session_target="codex=" + "t" * 200 + "-->\nx")
    text2, _u = run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=hostile_row)
    human2 = text2.split("\n")[1]
    assert "-->" not in human2 and "\n" not in human2 and "OMISSION NOTE" in human2 and "t" * 121 not in human2


def test_packed_incomplete_follows_the_provider_finish_reason(review_repo, review_drive, monkeypatch):
    """Item 11: a packed report cut by the output reserve is labelled so
    (`response_finish_reason == "length"`), never "complete"."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = mock.Mock()
    pack = "y" * 200
    stats = {"file_count": 2, "total_chars": len(pack), "skipped": [],
             "memory": {"inlined": 1, "total": 7, "dispositions": {"memory/identity.md": "inlined"}}}
    for finish, expected in (("length", "output_reserve"), ("stop", "none"), (None, "none")):
        usage_in = {"cost": 0.0, **({"response_finish_reason": finish} if finish else {})}
        llm.chat.return_value = ({"content": "Cut repo"}, usage_in)
        with mock.patch.object(deep_self_review, "build_review_pack", return_value=(pack, stats)):
            text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
        comment = text.split("\n")[0]
        assert f"incomplete={expected}" in comment, (finish, comment)
        assert ("INCOMPLETE (output_reserve" in text) == (expected == "output_reserve")
        assert usage["deep_review_memory"] == stats["memory"] and "memory=1/7" in comment
        assert "execution_status" not in usage  # a cut report is a product, disclosed — not a failure
    # The direct-Anthropic lane (the shipped `anthropic::` deep default) sets
    # NO usage finish reason; its cut marker is the message's `stop_reason`.
    for message, expected in (
        ({"content": "Cut repo", "stop_reason": "max_tokens"}, "output_reserve"),
        ({"content": "Whole repo", "stop_reason": "end_turn"}, "none"),
        # Fail-safe for a NON-normalized message shape only: the OpenAI-compatible
        # normalizer keeps finish_reason in usage (`response_finish_reason`), so a
        # message-level finish_reason is not a shipped contract — this pins the
        # arm's fail-safe reading of an unexpected shape, nothing more.
        ({"content": "Cut repo", "finish_reason": "length"}, "output_reserve"),
    ):
        llm.chat.return_value = (message, {"cost": 0.0})
        with mock.patch.object(deep_self_review, "build_review_pack", return_value=(pack, stats)):
            text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
        assert f"incomplete={expected}" in text.split("\n")[0], (message, text.split("\n")[0])
        assert ("INCOMPLETE (output_reserve" in text) == (expected == "output_reserve")


def test_memory_dispositions_are_disclosed_per_whitelisted_entry(review_repo, tmp_path, monkeypatch):
    """Item 23: a partially initialized data root — one inlined, one empty, one
    oversized, four missing — is disclosed per entry in the task text, the
    usage fact and the header, on the retrieving delivery and the packed one."""
    drive = tmp_path / "partial"
    (drive / "memory" / "knowledge").mkdir(parents=True)
    (drive / "state").mkdir()
    (drive / "memory" / "identity.md").write_text("I am Ouroboros.\n", encoding="utf-8")
    (drive / "memory" / "scratchpad.md").write_text("   \n", encoding="utf-8")
    (drive / "memory" / "WORLD.md").write_text("w" * (1_048_576 + 1), encoding="utf-8")
    task, facts = deep_self_review._retrieving_task(review_repo, drive)
    expected = {
        "memory/identity.md": "inlined", "memory/scratchpad.md": "empty", "memory/registry.md": "missing",
        "memory/WORLD.md": "oversized", "memory/knowledge/index-full.md": "missing",
        "memory/knowledge/patterns.md": "missing", "memory/knowledge/improvement-backlog.md": "missing",
    }
    assert facts["memory"] == {"inlined": 1, "total": 7, "dispositions": expected}
    assert "## FILE: drive/memory/identity.md\nI am Ouroboros.\n" in task
    assert "## FILE: drive/memory/scratchpad.md" not in task and "## FILE: drive/memory/WORLD.md" not in task
    line = next(l for l in task.splitlines() if l.startswith("Memory dispositions (7 whitelisted): "))
    for rel, disposition in expected.items():
        assert f"{rel} {disposition}" in line
    # The native episode carries the same fact into usage and the header.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = _ScriptedLLM([{"content": _REPORT}])
    text, usage = run_deep_self_review(review_repo, drive, llm, lambda _m: None, slot=_native_row())
    assert usage["deep_review_memory"]["dispositions"] == expected
    comment = text.split("\n")[0]
    assert "memory=1/7" in comment
    assert ("memory_missing=registry.md,index-full.md,patterns.md,improvement-backlog.md, "
            "memory_empty=scratchpad.md, memory_oversized=WORLD.md, coverage=") in comment
    # Worst case — every whitelisted file omitted under ONE disposition — still fits the value bound.
    from ouroboros.deep_self_review import _HEADER_VALUE_MAX_CHARS, _MEMORY_WHITELIST
    worst = ",".join(rel.rsplit("/", 1)[-1] for rel in _MEMORY_WHITELIST)
    assert len(worst) <= _HEADER_VALUE_MAX_CHARS
    assert "memory 1/7 inlined (omitted: scratchpad.md empty, registry.md missing, WORLD.md oversized" in text.split("\n")[1]
    # The packed pack states the same dispositions in its omission section.
    dulwich_index = mock.Mock(); dulwich_index.__iter__ = mock.Mock(return_value=iter([b"ouroboros/loop.py", b"BIBLE.md"]))
    dulwich_repo = mock.Mock(); dulwich_repo.open_index.return_value = dulwich_index
    with mock.patch("dulwich.repo.Repo", mock.Mock(return_value=dulwich_repo)):
        pack, stats = deep_self_review.build_review_pack(review_repo, drive)
    assert stats["memory"]["dispositions"] == expected and stats["memory"]["inlined"] == 1
    omitted = pack[pack.index("## OMITTED FILES"):]
    assert "drive/memory/scratchpad.md (empty: no content)" in omitted
    assert "drive/memory/WORLD.md (oversized: >1024KB)" in omitted
    assert "drive/memory/registry.md (missing: not present under the data root)" in omitted


def test_native_read_extent_rides_the_receipts_and_drives_coverage(review_repo, review_drive, monkeypatch):
    """Item 20 end to end: the reader's own window facts reach the receipts
    (extended contract: start_line/end_line/total_lines/eof), two chunks that
    cover BIBLE.md read as `read`, one line as `partial`, a data-root BIBLE.md
    as `missing`, and an episode-truncated read counts only delivered lines."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    total = len(_BIBLE.splitlines())
    half = total // 2
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md", "max_lines": half}, "c1")]},
        {"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md", "start_line": half + 1, "max_lines": 500}, "c2")]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    receipts = usage["native_tool_receipts"]
    assert receipts[0]["outcome"] == "executed"  # the outcome vocabulary is unchanged
    assert (receipts[0]["start_line"], receipts[0]["end_line"], receipts[0]["total_lines"], receipts[0]["eof"]) == (1, half, total, False)
    assert (receipts[1]["start_line"], receipts[1]["end_line"], receipts[1]["eof"]) == (half + 1, total, True)
    assert "coverage=BIBLE.md:read" in text and "BIBLE.md read in full" in text
    assert not [d for d in usage.get("capability_delta", []) if d["reason"].startswith("deep_review_")]

    # One line: partial, with the fraction in the header and a typed delta.
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md", "max_lines": 1}, "c1")]}, {"content": _REPORT}])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    assert f"coverage=BIBLE.md:partial({1 / total:.2f})" in text
    assert f"BIBLE.md {1 / total:.0%} read (1/{total} lines)" in text
    delta = next(d for d in usage["capability_delta"] if d["reason"] == "deep_review_mandatory_read_partial")
    assert delta["effective"] == f"1 of {total} lines of BIBLE.md delivered (merged receipts)"

    # A BIBLE.md under the DATA plane does not satisfy the repository read.
    (review_drive / "BIBLE.md").write_text(_BIBLE, encoding="utf-8")
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md", "root": "runtime_data"}, "c1")]}, {"content": _REPORT}])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    assert usage["native_tool_receipts"][0]["root"] == "runtime_data" and usage["native_tool_receipts"][0]["eof"] is True
    assert usage["native_tool_receipts"][0]["opened_root"] == "runtime_data"  # the opened root never credits a data-plane read
    assert "coverage=BIBLE.md:missing" in text

    # The episode's own result bound cut the body: only complete delivered lines count.
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (review_repo / "BIBLE.md").write_text("".join(f"line {i:05d} " + "b" * 60 + "\n" for i in range(1500)), encoding="utf-8")
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md"}, "c1")]}, {"content": _REPORT}])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    receipt = usage["native_tool_receipts"][0]
    assert receipt["total_lines"] == 1500 and receipt["start_line"] == 1
    assert receipt["end_line"] < 1500 and receipt["eof"] is False
    tool_msg = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"][0]["content"]
    assert "RESULT TRUNCATED" in tool_msg and f"line {receipt['end_line']:05d}" in tool_msg
    assert f"line {receipt['end_line'] + 1:05d} " + "b" * 60 + "\n" not in tool_msg  # the first cut line is not counted
    assert "coverage=BIBLE.md:partial(" in text


def test_a_registry_refused_read_never_inherits_the_previous_reads_extent(review_repo, review_drive, monkeypatch):
    """The stamp-leak class (round 3): a `read_file` the registry refuses BEFORE
    dispatch (its binding layer — path traversal) never reaches the reader, so
    it carries NO extent, and its `..` path is never folded onto `BIBLE.md`:
    after a real read of another file the mandatory read is `missing` with
    its typed delta; after a real PARTIAL read of BIBLE.md the traversal
    shapes never lift it to `read`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    # ONE traversal shape at the coverage level; the refusal trio itself is the
    # executor suite's receipt-level pin (test_read_file_receipts_carry_the_delivered_extent).
    shapes = ("a/../BIBLE.md",)
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "docs/ARCHITECTURE.md"}, "c1")]
                       + [_tool_call("read_file", {"path": p}, f"c{i}") for i, p in enumerate(shapes, 2)]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    receipts = usage["native_tool_receipts"]
    assert receipts[0]["path"] == "docs/ARCHITECTURE.md" and receipts[0]["eof"] is True
    tool_msgs = {m["tool_call_id"]: m["content"] for m in llm.calls[1]["messages"] if m.get("role") == "tool"}
    for i, p in enumerate(shapes, 1):
        assert tool_msgs[f"c{i + 1}"].startswith("⚠️ READ_FILE_ERROR") and receipts[i]["path"] == p
        assert receipts[i]["outcome"] == "executed"  # the registry answered with text; the vocabulary is unchanged
        assert not any(k in receipts[i] for k in ("start_line", "end_line", "total_lines", "eof", "opened_path", "opened_root")), receipts[i]
    assert "coverage=BIBLE.md:missing" in text and "BIBLE.md NOT read" in text
    assert [d["reason"] for d in usage["capability_delta"]] == ["deep_review_mandatory_read_missing"]
    # A real PARTIAL read followed by a traversal shape stays partial — never `read`.
    total = len(_BIBLE.splitlines())
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md", "max_lines": 1}, "c1"),
                        _tool_call("read_file", {"path": "a/../BIBLE.md"}, "c2")]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    assert f"coverage=BIBLE.md:partial({1 / total:.2f})" in text
    assert "total_lines" not in usage["native_tool_receipts"][1]
    # The path rule itself: `..` is kept as spelled (matches no mandatory read);
    # a clean relative or in-repo absolute spelling still normalizes.
    assert deep_self_review._repo_relative("a/../BIBLE.md", review_repo) == "a/../BIBLE.md"
    assert deep_self_review._repo_relative("BIBLE.md/../BIBLE.md", review_repo) == "BIBLE.md/../BIBLE.md"
    assert deep_self_review._repo_relative("./docs//ARCHITECTURE.md", review_repo) == "docs/ARCHITECTURE.md"
    assert deep_self_review._repo_relative(str(review_repo / "BIBLE.md"), review_repo) == "BIBLE.md"
    # The key is POSIX on EVERY host OS (a Windows runner's `os.path` IS ntpath,
    # whose normpath renders `docs\ARCHITECTURE.md`): with the module's OS-native
    # path module swapped for ntpath, relative and backslash spellings still fold
    # onto the POSIX mandatory-read key, and `..` still stays as spelled.
    monkeypatch.setattr(deep_self_review, "os", types.SimpleNamespace(path=ntpath, environ=os.environ))
    assert deep_self_review._repo_relative("./docs//ARCHITECTURE.md", review_repo) == "docs/ARCHITECTURE.md"
    assert deep_self_review._repo_relative(".\\docs\\ARCHITECTURE.md", review_repo) == "docs/ARCHITECTURE.md"
    assert deep_self_review._repo_relative("a\\..\\BIBLE.md", review_repo) == "a/../BIBLE.md"



# ---------------------------------------------------------------------------
# Fix batch №1 — custody / ownership (items 4, 14, 8, 17).
# ---------------------------------------------------------------------------


def test_empty_retrieving_response_is_an_error_row_never_a_responded_review(review_repo, review_drive, monkeypatch):
    import ouroboros.review_execution as review_execution

    class _Empty(_FakeSessionExecutor):
        def execute(self):
            result = super().execute()
            return ReviewAttemptResult(message={"content": "  "}, usage=result.usage, raw_text="  ")

    monkeypatch.setattr(review_execution, "_review_route_executor", _Empty)
    monkeypatch.setattr(deep_self_review, "_session_route_reason", lambda row: "")
    text, usage = run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=_session_row())
    assert text.startswith("⚠️ Model returned an empty response")
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "deep_self_review_error"
    last = reviewer_slot_last_executions()[DEEP_REVIEW_SLOT_ID]
    assert last["status"] == "error" and last["surface"] == "deep_self_review"


def test_coverage_deltas_never_mutate_the_executors_usage(review_repo, review_drive, monkeypatch):
    """Item 14: `dict(attempt.usage)` is shallow — the appended coverage delta
    must land on THIS record's copy, never on the list the executor owns."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    import ouroboros.review_native_episode as native_episode

    attempts = []
    original = native_episode.NativeToolRoundReviewExecutor.execute

    def spy(self):
        result = original(self)
        attempts.append(result)
        return result

    monkeypatch.setattr(native_episode.NativeToolRoundReviewExecutor, "execute", spy)
    # An exhausted report episode: the EXECUTOR itself owns a non-empty delta
    # list (`native_transcript_bound_before_final_answer`); the record then
    # appends its coverage delta — to ITS copy only.
    (review_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    draft = "# Draft\n\nCRITICAL: something.\n"
    llm = _ScriptedLLM([{"content": draft, "tool_calls": [_tool_call("read_file", {"path": "big.txt"}, "c1")]}] + [
        {"tool_calls": [_tool_call("read_file", {"path": "ouroboros/loop.py"}, f"c{i}")]} for i in range(2, 40)
    ])
    _text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    executor_list = attempts[0].usage["capability_delta"]
    assert [d["reason"] for d in executor_list] == ["native_transcript_bound_before_final_answer"]
    assert [d["reason"] for d in usage["capability_delta"]] == [
        "native_transcript_bound_before_final_answer", "deep_review_mandatory_read_missing"]
    assert usage["capability_delta"] is not executor_list and len(executor_list) == 1


def test_budget_exhaustion_propagates_out_of_the_review(review_repo, review_drive, monkeypatch):
    """Item 8: the paid ledger's refusal is budget vocabulary, not a review
    error — it must reach agent.py's `except BudgetExceeded: raise` rail."""
    import ouroboros.review_execution as review_execution
    from ouroboros.usage_accounting import BudgetExceeded

    class _Broke(_FakeSessionExecutor):
        def execute(self):
            raise BudgetExceeded("root budget exhausted")

    monkeypatch.setattr(review_execution, "_review_route_executor", _Broke)
    monkeypatch.setattr(deep_self_review, "_session_route_reason", lambda row: "")
    with pytest.raises(BudgetExceeded):
        run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=_session_row())
    # ...while any other executor failure stays a typed, returned review error.
    class _Boom(_FakeSessionExecutor):
        def execute(self):
            raise RuntimeError("socket reset")

    monkeypatch.setattr(review_execution, "_review_route_executor", _Boom)
    text, usage = run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=_session_row())
    assert text.startswith("❌ Deep self-review failed: RuntimeError: socket reset")
    assert usage["reason_code"] == "deep_self_review_error"


def test_slot_override_with_an_empty_target_or_unknown_kind_is_refused_typed(review_repo, review_drive, monkeypatch):
    """Item 17: a caller-built row never buys a paid call with `model=""`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = mock.Mock()
    for row, fragment in (
        (_row(target=""), "has no target"),
        (_row(target="   "), "has no target"),
        (_row("agent_session", "", session_target=""), "has no target"),
        (ConfiguredReviewerSlot(slot_id=DEEP_REVIEW_SLOT_ID, kind="bogus", target_id="openai/x"), "unknown route kind 'bogus'"),
    ):
        reason, identity = deep_review_route(row)
        assert fragment in reason and identity is None, (row, reason)
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=row)
        assert text.startswith("❌ Deep self-review unavailable: ") and fragment in text
        assert usage["reason_code"] == "deep_self_review_unavailable"
    assert not llm.chat.called



# ---------------------------------------------------------------------------
# Fix batch №1 — the packed delivery's ≥1M floor (codex sol, item 22).
# ---------------------------------------------------------------------------


def test_packed_row_refuses_a_confirmed_sub_1m_window_and_discloses_an_unknown_one(review_repo, review_drive, monkeypatch):
    """A packed review's guarantee IS its ≥1M pack: a route whose evidence puts
    the window below the floor is refused typed (never a silently shrunk
    pack); an unknown window keeps the documented full-window assumption and
    says so in the header; a confirmed ≥1M route runs unlabeled."""
    from ouroboros import reviewer_window
    from ouroboros.reviewer_window import REVIEWER_FULL_WINDOW, ReviewerWindow

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = mock.Mock()
    llm.chat.return_value = ({"content": "Review result."}, {"cost": 0.0})
    pack = "y" * 200
    stats = {"file_count": 2, "total_chars": len(pack), "skipped": [], "memory": {"inlined": 0, "total": 7, "dispositions": {}}}
    windows = {"answer": ReviewerWindow(window_tokens=200_000, status="confirmed", model="openai/fake-deep")}
    monkeypatch.setattr(reviewer_window, "resolve_reviewer_window", lambda model_id, **_k: windows["answer"])

    reason, identity = deep_review_route(_row())
    assert identity is None and "needs a ≥1,000,000-token window" in reason
    assert "openai/fake-deep is confirmed at 200,000 tokens" in reason and "native or session deep_review row" in reason
    with mock.patch.object(deep_self_review, "build_review_pack", return_value=(pack, stats)):
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
    assert text.startswith("❌ Deep self-review unavailable: the packed deep review needs a ≥1,000,000-token window")
    assert usage["reason_code"] == "deep_self_review_unavailable" and not llm.chat.called

    # Evidence landing between the availability read and the run is caught by the runner itself.
    calls = iter([ReviewerWindow(window_tokens=0), ReviewerWindow(window_tokens=131_072, status="confirmed")])
    monkeypatch.setattr(reviewer_window, "resolve_reviewer_window", lambda model_id, **_k: next(calls))
    with mock.patch.object(deep_self_review, "build_review_pack", return_value=(pack, stats)):
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
    assert "131,072 tokens" in text and not llm.chat.called

    # Unknown window: dispatched on the full-window assumption, disclosed.
    windows["answer"] = ReviewerWindow(window_tokens=0)
    monkeypatch.setattr(reviewer_window, "resolve_reviewer_window", lambda model_id, **_k: windows["answer"])
    assert deep_review_route(_row()) == ("", "openai/fake-deep")
    with mock.patch.object(deep_self_review, "build_review_pack", return_value=(pack, stats)):
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
    assert llm.chat.call_count == 1 and f"window=assumed_{REVIEWER_FULL_WINDOW}" in text.split("\n")[0]
    assert "window 1,000,000 (unknown, full window assumed)" in text.split("\n")[1]

    # Confirmed ≥1M: available, the window stated as a fact.
    windows["answer"] = ReviewerWindow(window_tokens=REVIEWER_FULL_WINDOW, status="confirmed")
    with mock.patch.object(deep_self_review, "build_review_pack", return_value=(pack, stats)):
        text, _usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
    assert llm.chat.call_count == 2 and f"window={REVIEWER_FULL_WINDOW}" in text.split("\n")[0]
    assert "(unknown" not in text
    # ONE window fact per run: the object validated against the floor IS the
    # object the pack is sized and labeled with. Availability reads once, the
    # run reads once — a third evidence value (here a confirmed 200K) is never
    # consumed, so it can never size a call the floor check did not see.
    seen = []
    sequence = iter([ReviewerWindow(window_tokens=0), ReviewerWindow(window_tokens=0),
                     ReviewerWindow(window_tokens=200_000, status="confirmed")])

    def _resolve(model_id, **_k):
        seen.append(model_id)
        return next(sequence)

    monkeypatch.setattr(reviewer_window, "resolve_reviewer_window", _resolve)
    before = llm.chat.call_count
    with mock.patch.object(deep_self_review, "build_review_pack", return_value=(pack, stats)):
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
    assert len(seen) == 2 and llm.chat.call_count == before + 1
    assert f"window=assumed_{REVIEWER_FULL_WINDOW}" in text.split("\n")[0] and "200" not in text.split("\n")[0]
    assert next(sequence).window_tokens == 200_000  # the third fact was never read
