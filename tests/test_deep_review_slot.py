"""The deep self-review's reviewer row (Ф3 R6/R7; PR-3 decision 3A).

Deep self-review runs on ONE reviewer row of the shared vocabulary: the direct
Main row when nothing names another (the retired ``deep_review`` lane row and
``OUROBOROS_MODEL_DEEP_SELF_REVIEW`` key reach it only through the migration), or
the row a caller hands in (``review_change(subject=system)`` naming a catalog
row). Delivery is retrieval either way — an api row is the bounded native
inspection episode, a session row a delegated run.
"""

import json
import ntpath

import pytest

# The row identity the retired lane minted for its singleton; the executions
# below are keyed by the row they ran on, so any fixed id serves the tests.
_DEEP_SLOT_ID = "deep_review_slot_1"

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


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(_ROSTER))
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_MODEL_DEEP_SELF_REVIEW", "OUROBOROS_EFFORT_DEEP_SELF_REVIEW"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# The two deliveries of ``run_deep_self_review`` on the row.
# ---------------------------------------------------------------------------

import copy  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from unittest import mock  # noqa: E402

from ouroboros import deep_self_review  # noqa: E402
from ouroboros.deep_self_review import (  # noqa: E402
    _REPORT_CONTRACT,
    deep_review_route,
    run_deep_self_review,
)
from ouroboros.review_execution import ReviewAttemptResult, ReviewRouteKind, ReviewRouteUnavailable  # noqa: E402
from ouroboros.reviewer_slot_config import ConfiguredReviewerSlot, reviewer_slot_last_executions  # noqa: E402


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
    (repo / "docs" / "CHECKLISTS.md").write_text("# Checks\n\n## Change Review Checklist\n\ny\n", encoding="utf-8")
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
    return ConfiguredReviewerSlot(slot_id=_DEEP_SLOT_ID, kind=kind, target_id=target, **fields)


def _native_row():
    return _row(subagent_id="api-critic")


def _session_row():
    return _row("agent_session", "codex=gpt-5.6-sol", session_target="codex=gpt-5.6-sol", profile_id="koshak")


def test_native_row_runs_the_inspection_episode_over_repo_and_memory(review_repo, review_drive, monkeypatch):
    """A configured-subagent api row is a NATIVE episode through the shared
    executor seam: the task carries the role prompt, the memory whitelist
    inline byte-exact, BIBLE.md delivered inline and the governance
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
        "coverage=BIBLE.md:delivered_inline, incomplete=none, attestation=host_observed, rounds=3, tool_calls=2, receipts=2, "
        "end_reason=final_answer, transcript=")
    assert "_Deep self-review: native inspection episode on openai/fake-deep — 3 rounds, 2 tool calls" in header
    assert "BIBLE.md delivered inline in full; memory 3/7 inlined (omitted: registry.md missing, WORLD.md missing, index-full.md missing, improvement-backlog.md missing); complete_" in header
    assert usage["deep_review_memory"]["inlined"] == 3 and usage["deep_review_memory"]["dispositions"]["memory/WORLD.md"] == "missing"
    assert usage["native_rounds"] == 3 and usage["host_file_read_attestation"] == "host_observed"
    assert usage["resolved_model"] == "openai/fake-deep" and "execution_status" not in usage
    assert not [d for d in usage.get("capability_delta", []) if str(d.get("reason", "")).startswith("deep_review_")]
    # The episode task: role prompt + method, memory inline byte-exact, BIBLE
    # inline in full, nav maps, and the REPORT contract (never the
    # JSON array the executors fall back to).
    first = llm.calls[0]["messages"]
    task = next(m["content"] for m in first if m["role"] == "user")
    assert "deep self-review of the Ouroboros project" in task
    assert "delivered IN FULL below" in task and f"{len(_BIBLE):,} chars" in task
    assert task.count(_BIBLE) == 1
    assert "## FILE: drive/memory/identity.md\nI am Ouroboros.\n" in task
    assert "## FILE: drive/memory/knowledge/patterns.md\n## Patterns\n- class A\n" in task
    assert "Memory dispositions (7 whitelisted): memory/identity.md inlined; memory/scratchpad.md inlined; memory/registry.md missing" in task
    assert "ARCHITECTURE.md (navigation map)" in task and "Deep self-review" in task
    assert "Deliver the report itself as plain markdown prose" in task
    assert "Begin with one line naming what you read" in task
    assert "JSON array" not in task
    # The REAL data root is the episode's data plane (R5): memory is readable.
    tool_msgs = [m for m in llm.calls[2]["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c1" and "becoming personality" in tool_msgs[0]["content"]
    assert tool_msgs[1]["tool_call_id"] == "c2" and "I am Ouroboros." in tool_msgs[1]["content"]
    # «Выполняется как» for the deep-review row, and the progress names the delivery.
    last = reviewer_slot_last_executions()[_DEEP_SLOT_ID]
    assert last["surface"] == "deep_self_review" and last["status"] == "responded"
    assert last["requested"]["subagent_id"] == "api-critic" and last["effective"]["model"] == "openai/fake-deep"
    assert any("native_tool_rounds" in line for line in progress)


def test_native_row_needs_no_second_read_of_inline_bible(review_repo, review_drive, monkeypatch):
    """Inline delivery satisfies the exact constitution source without a tool read.
    The older line-only receipt fold remains explicit below."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "ouroboros/loop.py"}, "c1")]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    assert text.endswith(_REPORT)
    assert "coverage=BIBLE.md:delivered_inline" in text and "BIBLE.md delivered inline in full; memory 3/7 inlined" in text
    assert text.split("\n")[1].endswith("; complete_")
    assert not [d for d in usage.get("capability_delta", []) if d["reason"].startswith("deep_review_")]
    assert reviewer_slot_last_executions()[_DEEP_SLOT_ID]["status"] == "responded"

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
    # padded, redundant `repo/` prefix, dot-prefixed) credits
    # `read` — one scripted episode per spelling, through the real registry.
    total = len(_BIBLE.splitlines())
    for spelled in (str(review_repo / "BIBLE.md"), " BIBLE.md", "repo/BIBLE.md", "./BIBLE.md", "BIBLE.md"):
        llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": spelled}, "c1")]}, {"content": _REPORT}])
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
        receipt = usage["native_tool_receipts"][0]
        assert (receipt["path"], receipt["opened_path"], receipt["eof"], receipt["total_lines"]) == (spelled, "BIBLE.md", True, total), receipt
        assert "coverage=BIBLE.md:delivered_inline" in text and "BIBLE.md delivered inline in full" in text, (spelled, text.split("\n")[0])
        assert not [d for d in usage.get("capability_delta", []) if d["reason"].startswith("deep_review_")]
    # ...and on the OPENED root: a padded root spelling the registry reads as a
    # repository root credits `read` (the raw `root` stays the model's spelling).
    for root in (" system_repo ", "active_workspace ", "system_repo"):
        llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md", "root": root}, "c1")]}, {"content": _REPORT}])
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
        receipt = usage["native_tool_receipts"][0]
        assert (receipt["root"], receipt["opened_root"], receipt["opened_path"], receipt["eof"]) == (root, root.strip(), "BIBLE.md", True), receipt
        assert "coverage=BIBLE.md:delivered_inline" in text, (root, text.split("\n")[0])
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
    the owner deadline, and tool reading stays distinct from observed inline delivery."""
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
        "coverage=BIBLE.md:delivered_inline, incomplete=unobserved, attestation=unobserved, "
        f"generated_at={usage['deep_review_generated_at']}, source_revision=unknown -->\n"
        "_Deep self-review: agent session codex=gpt-5.6-sol (model gpt-5.6-sol) — tool reads not host-observed; "
        "BIBLE.md delivered inline in full; memory 3/7 inlined (omitted: registry.md missing, WORLD.md missing, index-full.md missing, "
        "improvement-backlog.md missing); completeness not host-observed_\n"
        f"Report generated at {usage['deep_review_generated_at']}; reviewed source revision: unknown (not captured).\n\n")
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
    assert slot.session_profile == "koshak" and slot.slot_id == _DEEP_SLOT_ID
    assert slot.max_tokens == 100_000 and slot.role_hint == "deep self-reviewer"
    # The logical window: the task ceiling narrowed by the owner deadline (never the 300 s default).
    assert 0 < slot.timeout_sec < 600
    assert before < executor._logical_deadline_monotonic < before + 600
    assert executor.assignment.custody_root == review_drive and executor.assignment.call_type == "deep_self_review"
    last = reviewer_slot_last_executions()[_DEEP_SLOT_ID]
    assert last["effective"] == {"route": "agent_session:codex", "model": "gpt-5.6-sol", "verdict_method": "report"}
    assert last["requested"]["session_target"] == "codex=gpt-5.6-sol" and last["requested"]["profile_id"] == "koshak"
    # Without an owner deadline the window is the task's operation window: its finite
    # absolute lifetime, else the finite operation fallback (never an unbounded session).
    _FakeSessionExecutor.instances = []
    from ouroboros.config import get_task_abs_ceiling_sec, operation_window_sec
    run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=_session_row())
    assert _FakeSessionExecutor.instances[0].assignment.slot.timeout_sec == operation_window_sec(
        get_task_abs_ceiling_sec())


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
    last = reviewer_slot_last_executions()[_DEEP_SLOT_ID]
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
        last = reviewer_slot_last_executions()[_DEEP_SLOT_ID]
        assert last["status"] == status and last["surface"] == "deep_self_review"
        assert last["requested"]["session_target"] == "codex=gpt-5.6-sol" and "capability_delta" in last
        assert "deep_review_memory" not in json.dumps(last)  # intentionally absent from the durable projection


def test_availability_follows_the_row_not_the_model_key(env, monkeypatch):
    """Route-aware availability (`deep_review_route`, the ONE availability
    reader — agent, tool and runner all call it): an api row needs its model's
    credentials (a bare route and a subagent reference read the SAME rule), a
    session row needs a healthy delegated route (the substrate's own reader),
    and with no row named it is the direct Main row's (decision 3A)."""
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    reason, identity = deep_review_route(_row())
    assert reason.startswith("no OpenRouter or direct OpenAI credentials for openai/fake-deep") and identity is None
    assert deep_review_route(_native_row())[0] == reason
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
    # No row named: the direct Main row — the catalog's review marks do not
    # decide `/review`'s executor.
    env.setenv("OUROBOROS_MODEL", "openai/main-model")
    assert deep_review_route() == ("", "openai/main-model")


def test_unavailable_row_never_runs_and_returns_typed_usage(review_repo, review_drive, monkeypatch, env):
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    llm = mock.Mock()
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_row())
    assert text.startswith("❌ Deep self-review unavailable: no OpenRouter or direct OpenAI credentials for openai/fake-deep")
    assert usage == {"execution_status": "infra_failed", "reason_code": "deep_self_review_unavailable"}
    assert not llm.chat.called
    env.setenv("OUROBOROS_MODEL", "openai/main-model")
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None)
    assert text.startswith("❌ Deep self-review unavailable: no OpenRouter or direct OpenAI credentials for openai/main-model")
    assert usage["reason_code"] == "deep_self_review_unavailable" and not llm.chat.called


def test_agent_keeps_the_previous_report_when_the_review_fails(tmp_path, monkeypatch):
    """`memory/deep_review.md` is overwritten ONLY by a delivered report: a
    typed failure goes to the task result and a typed `task_error` event. The
    worker runs `review_change(subject=system, surface=system)` and links its
    record in the task's answer."""
    import ouroboros.agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.review_ledger import load_record

    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive"
    (drive / "memory").mkdir(parents=True)
    (drive / "logs").mkdir()
    (drive / "memory" / "deep_review.md").write_text("PREVIOUS REPORT", encoding="utf-8")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **_k: ([], {}))
    answers = []
    monkeypatch.setattr(agent_module, "emit_task_results", lambda *a, **_k: answers.append(a[5]))
    outcome = {"value": ("❌ Deep self-review unavailable: no provider credentials for openai/x. Run /review …",
                         {"execution_status": "infra_failed", "reason_code": "deep_self_review_unavailable"})}
    monkeypatch.setattr(deep_self_review, "run_deep_self_review", lambda *_a, **_k: outcome["value"])
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    task = {"id": "dsr-agent", "type": "deep_self_review", "chat_id": 1, "text": "owner:/review",
            "metadata": {"deadline_at": "2099-01-01T00:00:00+00:00"}}
    events = agent.handle_task(task)
    assert (drive / "memory" / "deep_review.md").read_text(encoding="utf-8") == "PREVIOUS REPORT"
    rows = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = [r for r in rows if r.get("type") == "task_error" and r.get("task_id") == "dsr-agent"]
    assert errors and errors[0]["reason_code"] == "deep_self_review_unavailable"
    assert any(e.get("type") == "llm_usage" and e.get("category") == "deep_self_review" for e in events)
    # A delivered report overwrites it, the deadline reaches the review, and the
    # answer links the surface=system record that keeps the report.
    seen = {}

    def _ok(*_args, **kwargs):
        seen.update(kwargs)
        return "<!-- deep-review provenance: delivery=native_tool_rounds -->\n_x_\n\nNEW REPORT", {"resolved_model": "openai/x", "cost": 0.0}

    monkeypatch.setattr(deep_self_review, "run_deep_self_review", _ok)
    events = agent.handle_task(task)
    assert (drive / "memory" / "deep_review.md").read_text(encoding="utf-8").endswith("NEW REPORT")
    assert seen["task_id"] == "dsr-agent" and seen["deadline_at"] == "2099-01-01T00:00:00+00:00"
    assert seen["slot"].slot_id == "main"
    usage_events = [e for e in events if e.get("type") == "llm_usage"]
    assert usage_events and usage_events[0]["model"] == "openai/x"
    body, link = answers[-1].rsplit("\n\nReview record: ", 1)
    assert body.endswith("NEW REPORT") and link.endswith(" (surface=system)")
    record = load_record(drive, link.split(" ", 1)[0])
    assert record["surface"] == "system" and [seat["seat_id"] for seat in record["rows"]] == ["main"]


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
    hostile_row = ConfiguredReviewerSlot(slot_id=_DEEP_SLOT_ID, kind="agent_session",
                                         target_id="codex=" + "t" * 200 + "-->\nx", session_target="codex=" + "t" * 200 + "-->\nx")
    text2, _u = run_deep_self_review(review_repo, review_drive, object(), lambda _m: None, slot=hostile_row)
    human2 = text2.split("\n")[1]
    assert "-->" not in human2 and "\n" not in human2 and "OMISSION NOTE" in human2 and "t" * 121 not in human2


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


def test_native_read_extent_rides_the_receipts_and_drives_coverage(review_repo, review_drive, monkeypatch):
    """Item 20 end to end: the reader's own window facts reach the receipts
    (extended contract: start_line/end_line/total_lines/eof), two chunks that
    cover inspection.md read as `read`, one line as `partial`, a data-root inspection.md
    as `missing`, and an episode-truncated read counts only delivered lines."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    from ouroboros.tools.scope_required_sources import source_text_identity

    (review_repo / "inspection.md").write_text(_BIBLE, encoding="utf-8")

    def required():
        return [{"root": "system_repo", "path": "inspection.md",
                 **source_text_identity((review_repo / "inspection.md").read_bytes())}]
    total = len(_BIBLE.splitlines())
    half = total // 2
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "inspection.md", "max_lines": half}, "c1")]},
        {"tool_calls": [_tool_call("read_file", {"path": "inspection.md", "start_line": half + 1, "max_lines": 500}, "c2")]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row(), required_sources=required())
    receipts = usage["native_tool_receipts"]
    assert receipts[0]["outcome"] == "executed"  # the outcome vocabulary is unchanged
    assert (receipts[0]["start_line"], receipts[0]["end_line"], receipts[0]["total_lines"], receipts[0]["eof"]) == (1, half, total, False)
    assert (receipts[1]["start_line"], receipts[1]["end_line"], receipts[1]["eof"]) == (half + 1, total, True)
    assert "coverage=inspection.md:read" in text and "inspection.md read in full" in text
    assert not [d for d in usage.get("capability_delta", []) if d["reason"].startswith("deep_review_")]

    # One line: partial, with the fraction in the header and a typed delta.
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "inspection.md", "max_lines": 1}, "c1")]}, {"content": _REPORT}])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row(), required_sources=required())
    assert f"coverage=inspection.md:partial({len(_BIBLE.splitlines(keepends=True)[0]) / len(_BIBLE):.2f})" in text
    assert f"inspection.md {len(_BIBLE.splitlines(keepends=True)[0]) / len(_BIBLE):.0%} read ({len(_BIBLE.splitlines(keepends=True)[0])}/{len(_BIBLE)} characters)" in text
    delta = next(d for d in usage["capability_delta"] if d["reason"] == "deep_review_mandatory_read_partial")
    assert delta["effective"] == f"{len(_BIBLE.splitlines(keepends=True)[0])} of {len(_BIBLE)} characters of inspection.md delivered (merged receipts)"

    # An inspection.md under the DATA plane does not satisfy the repository read.
    (review_drive / "inspection.md").write_text(_BIBLE, encoding="utf-8")
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "inspection.md", "root": "runtime_data"}, "c1")]}, {"content": _REPORT}])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row(), required_sources=required())
    assert usage["native_tool_receipts"][0]["root"] == "runtime_data" and usage["native_tool_receipts"][0]["eof"] is True
    assert usage["native_tool_receipts"][0]["opened_root"] == "runtime_data"  # the opened root never credits a data-plane read
    assert "coverage=inspection.md:missing" in text

    # The episode's own result bound cut the body: only complete delivered lines count.
    # The bound must leave room for a CUT-but-inline result beside the task the
    # governance tiers deliver, or the result rides a stored source instead.
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "120000")
    (review_repo / "inspection.md").write_text("".join(f"line {i:05d} " + "b" * 60 + "\n" for i in range(1500)), encoding="utf-8")
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "inspection.md"}, "c1")]}, {"content": _REPORT}])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row(), required_sources=required())
    receipt = usage["native_tool_receipts"][0]
    assert receipt["total_lines"] == 1500 and receipt["start_line"] == 1
    assert receipt["end_line"] < 1500 and receipt["eof"] is False
    tool_msg = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"][0]["content"]
    # Source labels begin at zero; receipt line addresses begin at one.
    body = tool_msg.split("\n⚠️ RESULT TRUNCATED", 1)[0]  # the notice opens with its OWN newline: judge the delivered body only
    assert body != tool_msg and f"line {receipt['end_line'] - 1:05d} " + "b" * 60 + "\n" in body
    assert f"line {receipt['end_line']:05d} " + "b" * 60 + "\n" not in body  # a cut on a line's last character is not that line complete
    assert "coverage=inspection.md:partial(" in text


def test_a_registry_refused_read_never_inherits_the_previous_reads_extent(review_repo, review_drive, monkeypatch):
    """The stamp-leak class (round 3): a `read_file` the registry refuses BEFORE
    dispatch (its binding layer — path traversal) never reaches the reader, so
    it carries NO extent, and its `..` path is never folded onto `inspection.md`:
    after a real read of another file the mandatory read is `missing` with
    its typed delta; after a real PARTIAL read of inspection.md the traversal
    shapes never lift it to `read`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    from ouroboros.tools.scope_required_sources import source_text_identity

    (review_repo / "inspection.md").write_text(_BIBLE, encoding="utf-8")

    def required():
        return [{"root": "system_repo", "path": "inspection.md",
                 **source_text_identity((review_repo / "inspection.md").read_bytes())}]
    # ONE traversal shape at the coverage level; the refusal trio itself is the
    # executor suite's receipt-level pin (test_read_file_receipts_carry_the_delivered_extent).
    shapes = ("a/../inspection.md", "/inspection.md")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "docs/ARCHITECTURE.md"}, "c1")]
                       + [_tool_call("read_file", {"path": p}, f"c{i}") for i, p in enumerate(shapes, 2)]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row(), required_sources=required())
    receipts = usage["native_tool_receipts"]
    assert receipts[0]["path"] == "docs/ARCHITECTURE.md" and receipts[0]["eof"] is True
    tool_msgs = {m["tool_call_id"]: m["content"] for m in llm.calls[1]["messages"] if m.get("role") == "tool"}
    for i, p in enumerate(shapes, 1):
        assert tool_msgs[f"c{i + 1}"].startswith("⚠️ READ_FILE_ERROR") and receipts[i]["path"] == p
        assert receipts[i]["outcome"] == "executed"  # the registry answered with text; the vocabulary is unchanged
        assert not any(k in receipts[i] for k in ("start_line", "end_line", "total_lines", "eof", "opened_path", "opened_root")), receipts[i]
    assert "coverage=inspection.md:missing" in text and "inspection.md NOT read" in text
    assert "deep_review_mandatory_read_missing" in [d["reason"] for d in usage["capability_delta"]]
    # A real PARTIAL read followed by a traversal shape stays partial — never `read`.
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "inspection.md", "max_lines": 1}, "c1"),
                        _tool_call("read_file", {"path": "a/../inspection.md"}, "c2")]},
        {"content": _REPORT},
    ])
    text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row(), required_sources=required())
    assert f"coverage=inspection.md:partial({len(_BIBLE.splitlines(keepends=True)[0]) / len(_BIBLE):.2f})" in text
    assert "total_lines" not in usage["native_tool_receipts"][1]
    # The path rule itself: `..` is kept as spelled (matches no mandatory read);
    # a clean relative or in-repo absolute spelling still normalizes.
    assert deep_self_review._repo_relative("a/../inspection.md", review_repo) == "a/../inspection.md"
    assert deep_self_review._repo_relative("inspection.md/../inspection.md", review_repo) == "inspection.md/../inspection.md"
    assert deep_self_review._repo_relative("./docs//ARCHITECTURE.md", review_repo) == "docs/ARCHITECTURE.md"
    assert deep_self_review._repo_relative(str(review_repo / "inspection.md"), review_repo) == "inspection.md"
    # Feed an actual Windows-normalized spelling through the POSIX receipt owner.
    # It uses posixpath directly and no longer imports an OS-native path module.
    windows_path = ntpath.normpath("./docs//ARCHITECTURE.md")
    assert deep_self_review._repo_relative(windows_path, review_repo) == "docs/ARCHITECTURE.md"
    assert deep_self_review._repo_relative("./docs//ARCHITECTURE.md", review_repo) == "docs/ARCHITECTURE.md"
    assert deep_self_review._repo_relative(".\\docs\\ARCHITECTURE.md", review_repo) == "docs/ARCHITECTURE.md"
    assert deep_self_review._repo_relative("a\\..\\inspection.md", review_repo) == "a/../inspection.md"



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
    last = reviewer_slot_last_executions()[_DEEP_SLOT_ID]
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
    # keeps its own delta list, separate from the executor's.
    (review_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    draft = "# Draft\n\nCRITICAL: something.\n"
    llm = _ScriptedLLM([{"content": draft, "tool_calls": [_tool_call("read_file", {"path": "big.txt"}, "c1")]}] + [
        {"tool_calls": [_tool_call("read_file", {"path": "ouroboros/loop.py"}, f"c{i}")]} for i in range(2, 40)
    ])
    _text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=_native_row())
    executor_list = attempts[0].usage["capability_delta"]
    assert [d["reason"] for d in executor_list] == ["native_transcript_bound_before_final_answer"]
    assert [d["reason"] for d in usage["capability_delta"]] == [
        "native_transcript_bound_before_final_answer"]
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
        (ConfiguredReviewerSlot(slot_id=_DEEP_SLOT_ID, kind="bogus", target_id="openai/x"), "unknown route kind 'bogus'"),
    ):
        reason, identity = deep_review_route(row)
        assert fragment in reason and identity is None, (row, reason)
        text, usage = run_deep_self_review(review_repo, review_drive, llm, lambda _m: None, slot=row)
        assert text.startswith("❌ Deep self-review unavailable: ") and fragment in text
        assert usage["reason_code"] == "deep_self_review_unavailable"
    assert not llm.chat.called
