"""Plain-text contract for system Project lifecycle rows (owner bug report 2026-08-31).

A Project completion card in Main rendered as a huge H1 with a literal
``## Короткий вывод`` glued mid-line: the excerpt producer flattened newlines
while KEEPING markdown markers, the durable ``chat.jsonl`` row stored them raw
(only the live send stripped), history replayed raw text, and the client
rendered every system row as markdown. These tests pin the server half of the
contract: one shared stripper (``ouroboros.utils.strip_markdown``), strip
BEFORE flatten in the producer, no inherited markdown format on host-salvage
``terminal_incident`` rows, read-side normalization of old persisted rows
without rewriting the log, and VERBATIM live delivery (the bridge no longer
strips — live frame == durable row == history replay, owner D14). The client
render arm is pinned in ``web/tests/chat_plain_system_rows.test.js``.
"""

from __future__ import annotations

import asyncio
import json
import types
from types import SimpleNamespace

MARKDOWN_RESULT = (
    "# Report title\n\n## Short conclusion\n\nBody with `inline code` and **bold**."
)
PLAIN_EXCERPT = "Report title Short conclusion Body with inline code and bold."


def test_completion_excerpt_strips_markers_before_flattening_newlines():
    """RO6: flatten-first would glue ``##`` mid-line where the line-anchored
    heading pattern (and every renderer) can no longer treat it as markup —
    exactly the owner's screenshot symptom."""
    from ouroboros.project_dialogue import _completion_excerpt

    excerpt = _completion_excerpt({"summary": MARKDOWN_RESULT})
    assert excerpt == PLAIN_EXCERPT
    for marker in ("#", "**", "`"):
        assert marker not in excerpt
    # The naive flatten-then-strip order would have produced this glued form.
    assert "## Short conclusion" not in excerpt


def test_completion_excerpt_leaves_plain_text_untouched():
    from ouroboros.project_dialogue import _completion_excerpt

    assert _completion_excerpt({"result": "Release shipped."}) == "Release shipped."


def test_completion_excerpt_strip_applies_before_length_cap():
    from ouroboros.project_dialogue import _completion_excerpt

    long_plain = "word " * 100  # 500 chars once flattened
    excerpt = _completion_excerpt({"summary": "## Heading\n" + long_plain})
    assert excerpt.startswith("Heading word")
    assert len(excerpt) <= 240
    assert excerpt.endswith("…")


def test_completion_summary_event_text_is_plain_and_fully_normalized(
    tmp_path, monkeypatch,
):
    from ouroboros.project_dialogue import enqueue_project_completion_summary
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.utils import strip_markdown

    project = create_project(tmp_path, "launch", name="Launch 🚀")
    bind_task_to_project(
        tmp_path, "root-project", project["id"], project["chat_id"],
        origin={"absent": "system"},
    )
    queued = []

    def _enqueue(_root, event, **_kwargs):
        queued.append(dict(event))
        return True

    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery", _enqueue,
    )
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path)
    root = {
        "id": "root-project", "project_id": "launch",
        "title": "Ship release", "chat_id": project["chat_id"],
    }
    result = {
        "task_id": "root-project", "status": "completed",
        "project_id": "launch", "title": "Ship release",
        "result": MARKDOWN_RESULT,
    }
    done = {"status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}

    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, {"status": "completed"}, "root-project", root, result, done,
    ) is True
    assert queued[0]["text"] == (
        f"Launch 🚀 › Ship release · Done\n{PLAIN_EXCERPT}"
    )
    for marker in ("#", "**", "`"):
        assert marker not in queued[0]["text"]
    # RO4 convergence: the producer already normalized, so the verbatim live
    # delivery and the durable row carry the same fully-plain text (stripping
    # the producer's output again is an identity).
    assert strip_markdown(queued[0]["text"]) == queued[0]["text"]
    # Structural fields stay intact next to the plain text.
    assert queued[0]["progress_meta"]["target_label"] == "Launch 🚀 › Ship release"
    assert queued[0]["system_type"] == "project_completion_summary"


def test_host_salvage_terminal_incident_drops_inherited_markdown_format(tmp_path):
    """RO9: the fixed host-salvage receipt must not inherit ``format:
    "markdown"`` from the completed-answer base event; the completed paths
    keep it (bug report #7: markdown system rows still render rich)."""
    from supervisor.terminal_delivery import project_terminal_result_event

    raw = "## Salvage heading\nRAW PATCH " * 20
    base = {
        "type": "send_message", "chat_id": 7, "task_id": "terminal-a",
        "text": raw, "format": "markdown",
    }
    host = project_terminal_result_event(
        tmp_path, {"chat_id": 7}, "terminal-a",
        result_text=raw, terminal_origin="host_salvage", base_event=dict(base),
    )
    assert host["system_type"] == "terminal_incident"
    assert "format" not in host

    model = project_terminal_result_event(
        tmp_path, {"chat_id": 7}, "terminal-a",
        result_text=raw, terminal_origin="model_final", base_event=dict(base),
    )
    assert model["format"] == "markdown"

    legacy = project_terminal_result_event(
        tmp_path, {"chat_id": 7}, "terminal-a",
        result_text=raw, terminal_origin=None, base_event=dict(base),
    )
    assert legacy["format"] == "markdown"

    # A host NOTICE is not salvage: its own words are the answer, and its
    # markdown must survive or the host's own code spans render escaped.
    notice = project_terminal_result_event(
        tmp_path, {"chat_id": 7}, "terminal-a",
        result_text=raw, terminal_origin="host_notice", base_event=dict(base),
    )
    assert notice["format"] == "markdown"
    assert notice["role"] == "system"
    assert "system_type" not in notice
    assert notice["text"] == raw


def test_history_normalizes_old_project_rows_on_read_without_rewriting_log(tmp_path):
    """Bug report #10: rows persisted BEFORE the producer stripped markdown are
    normalized on read; ``chat.jsonl`` stays byte-identical; only the two
    Project lifecycle types are touched (assistant markdown and other system
    types replay verbatim)."""
    from ouroboros.gateway.history import make_chat_history_endpoint

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    old_project_text = "Launch › Ship · Completed\n## Short conclusion\n**bold** `code`"
    rows = (
        {
            "ts": "2026-08-21T00:00:01Z", "direction": "system", "chat_id": 1,
            "type": "project_started", "task_id": "root-project",
            "project_id": "launch", "project_name": "Launch",
            "target_label": "Launch › Ship",
            "text": "# Launch › Ship · Started\nWork is running in this Project.",
        },
        {
            "ts": "2026-08-21T00:00:02Z", "direction": "system", "chat_id": 1,
            "type": "project_completion_summary", "task_id": "root-project",
            "project_id": "launch", "project_name": "Launch",
            "target_label": "Launch › Ship", "status": "completed",
            "text": old_project_text,
        },
        {
            "ts": "2026-08-21T00:00:03Z", "direction": "out", "chat_id": 1,
            "format": "markdown", "text": "## Assistant heading stays markdown",
        },
        {
            "ts": "2026-08-21T00:00:04Z", "direction": "system", "chat_id": 1,
            "type": "cancel_receipt", "task_id": "root-project",
            "text": "⚠️ Task cancelled. Below is the preserved text.\n\n## Verbatim heading",
        },
    )
    chat_path = logs / "chat.jsonl"
    chat_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    before = chat_path.read_bytes()

    endpoint = make_chat_history_endpoint(tmp_path)
    payload = json.loads(
        asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1"}))).body
    )
    by_type = {row.get("system_type"): row for row in payload["messages"] if row.get("role") == "system"}

    started = by_type["project_started"]
    assert started["text"] == "Launch › Ship · Started\nWork is running in this Project."
    assert started["markdown"] is False

    completion = by_type["project_completion_summary"]
    assert completion["text"] == "Launch › Ship · Completed\nShort conclusion\nbold code"
    assert completion["markdown"] is False
    for marker in ("#", "**", "`"):
        assert marker not in completion["text"]
    # Structural fields survive normalization untouched.
    assert completion["project_id"] == "launch"
    assert completion["project_name"] == "Launch"
    assert completion["target_label"] == "Launch › Ship"
    assert completion["status"] == "completed"

    # Only the two lifecycle types are normalized: the cancel_receipt salvage
    # stays verbatim (owner D14) and the assistant markdown row is untouched.
    receipt = by_type["cancel_receipt"]
    assert "## Verbatim heading" in receipt["text"]
    assistant = next(row for row in payload["messages"] if row.get("role") == "assistant")
    assert assistant["text"] == "## Assistant heading stays markdown"
    assert assistant["markdown"] is True

    # The durable log was read, never rewritten.
    assert chat_path.read_bytes() == before


def test_strip_markdown_is_the_single_shared_stripper():
    """RO8: the excerpt producer and history's read-side normalization share
    ONE stripper; the live bridge no longer strips AT ALL (verbatim delivery,
    owner D14) so a divergent second copy cannot reappear there."""
    import inspect

    from ouroboros.utils import strip_markdown
    from supervisor import message_bus

    assert "strip_markdown" not in inspect.getsource(message_bus)
    # Idempotent on its own output for the bug-report fixture: normalizing an
    # already-normalized producer row cannot change it.
    once = strip_markdown(MARKDOWN_RESULT)
    assert strip_markdown(once) == once


def test_cancel_receipt_rides_verbatim_live_durable_and_on_replay(
    monkeypatch, tmp_path,
):
    """Owner D14 + sol review: a cancel_receipt with raw markdown markers is
    delivered VERBATIM through the real ``send_with_budget`` — the live WS
    frame, the durable ``chat.jsonl`` row, and the history replay carry
    byte-equal text (the client renders all three as escaped plain text)."""
    import supervisor.message_bus as message_bus
    from ouroboros.gateway.history import make_chat_history_endpoint

    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(
        message_bus, "load_state", lambda: {"session_id": "s-1", "owner_id": 7},
    )
    monkeypatch.setattr(
        message_bus, "_advance_project_visible_revision", lambda _chat_id: None,
    )
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)

    verbatim = (
        "⚠️ Task t-1 was cancelled. Below is the last persisted intermediate "
        "model message, preserved WITHOUT review.\n\n"
        "## Heading\n**bold** and `code` and an unclosed `backtick"
    )
    message_bus.send_with_budget(
        1, verbatim, task_id="t-1", role="system", system_type="cancel_receipt",
    )

    live = next(frame for frame in frames if frame.get("type") == "chat")
    assert live["content"] == verbatim
    assert live["markdown"] is False
    assert live["role"] == "system"
    assert live["system_type"] == "cancel_receipt"

    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    durable = next(row for row in rows if row.get("type") == "cancel_receipt")
    assert durable["text"] == verbatim

    endpoint = make_chat_history_endpoint(tmp_path)
    payload = json.loads(
        asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1"}))).body
    )
    replayed = next(
        row for row in payload["messages"]
        if row.get("system_type") == "cancel_receipt"
    )
    assert replayed["text"] == verbatim
    assert replayed["markdown"] is False


def _parity_cases():
    import pathlib

    fixture = pathlib.Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures" / "outcome_phase_parity.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["cases"]


def test_host_status_phase_mirrors_the_browser_over_the_shared_fixture():
    """S5-03: one status-word family. The same fixture is read by
    ``web/tests/reason_detail.test.js``, so a divergence between the browser's
    severity fold and the host's durable label fails on both sides."""
    from ouroboros.project_dialogue import completion_status_label, outcome_phase

    cases = _parity_cases()
    assert len(cases) >= 10
    for case in cases:
        record = case["record"]
        assert outcome_phase(record, {}) == case["phase"], case["name"]
        assert completion_status_label(record, {}) == case["headline"], case["name"]
        # The event frame is the other half of the same merge: a record read
        # from the task_done event alone must resolve identically.
        assert outcome_phase({}, record) == case["phase"], case["name"]


def test_host_status_phase_folds_the_legacy_partial_result_status_to_a_warning():
    """The browser already shows a warning here (the gateway normalizes axes on
    read); the host label used to agree only by accident, through a 'partial'
    member of its own degraded set."""
    from ouroboros.project_dialogue import completion_status_label, outcome_phase

    legacy = {"status": "completed", "result_status": "partial"}
    assert outcome_phase(legacy, {}) == "warn"
    assert completion_status_label(legacy, {}) == "Done with warnings"


def test_owner_requested_stop_is_done_on_the_host_row_too():
    from ouroboros.project_dialogue import _completion_verdict, completion_status_label

    stopped = {
        "status": "completed", "reason_code": "owner_requested_finalization",
        "outcome_axes": {"execution": {"status": "best_effort"}},
    }
    assert completion_status_label(stopped, {}) == "Done"
    assert _completion_verdict(stopped, {}) == ""


A4_DECISION = {
    "status": "finalized_unaccepted",
    "rationale": "Acceptance reviewers did not reach a valid quorum.",
}
A4_CLAUSE = "Acceptance: finalized_unaccepted — Acceptance reviewers did not reach a valid quorum."


def _a4_result(**overrides):
    result = {
        "task_id": "root-project", "status": "completed", "reason_code": "final_message",
        "project_id": "launch", "title": "Ship release", "result": "Release shipped.",
        "outcome_axes": {
            "execution": {"status": "ok"},
            "review": {"status": "degraded", "acceptance_decision": dict(A4_DECISION)},
        },
    }
    result.update(overrides)
    return result


def test_host_verdict_states_an_unaccepted_acceptance_decision_in_its_own_words():
    """S5-04: a warning caused by REVIEW used to be explained by the execution
    reason that happened to sit beside it (``Reason: final_message``), which
    named the delivery step rather than the cause."""
    from ouroboros.project_dialogue import _completion_verdict

    assert _completion_verdict(_a4_result(), {}) == A4_CLAUSE
    # The stored rationale already ends in a period; the clause must not double it.
    assert not _completion_verdict(_a4_result(), {}).endswith("..")


def test_host_verdict_keeps_the_execution_reason_when_acceptance_was_reached():
    from ouroboros.project_dialogue import _completion_verdict

    accepted = _a4_result(outcome_axes={
        "execution": {"status": "ok"},
        "review": {"status": "pass", "acceptance_decision": {"status": "accepted"}},
    })
    assert _completion_verdict(accepted, {}) == "Reason: final_message."
    assert _completion_verdict({"status": "completed"}, {}) == ""
    # A hard failure explains itself by its execution reason, not by a decision.
    assert _completion_verdict(
        _a4_result(status="failed", reason_code="delegated_custody_unreconciled",
                   outcome_axes={"execution": {"status": "failed"},
                                 "review": {"acceptance_decision": dict(A4_DECISION)}}),
        {},
    ) == "Reason: delegated_custody_unreconciled."


def test_host_verdict_flattens_and_strips_a_markdown_rationale():
    """These are durable plain-text rows: the rationale is free owner-visible
    text up to 500 characters and may carry newlines and markdown markers."""
    from ouroboros.project_dialogue import _completion_verdict

    noisy = _a4_result(outcome_axes={
        "execution": {"status": "ok"},
        "review": {"status": "degraded", "acceptance_decision": {
            "status": "revision_requested",
            "rationale": "## Verdict\n\nThe **tests** never ran with `pytest`",
        }},
    })
    verdict = _completion_verdict(noisy, {})
    assert verdict == "Acceptance: revision_requested — Verdict The tests never ran with pytest."
    for marker in ("#", "**", "`", "\n"):
        assert marker not in verdict

    # A rationale that already terminates itself keeps its own punctuation: a
    # question mark is as terminal as a period, and appending one would render
    # "…did the tests run?." to the owner.
    asking = _a4_result(outcome_axes={
        "execution": {"status": "ok"},
        "review": {"status": "degraded", "acceptance_decision": {
            "status": "revision_requested", "rationale": "Did the tests ever run?",
        }},
    })
    assert _completion_verdict(asking, {}) == "Acceptance: revision_requested — Did the tests ever run?"


def test_host_verdict_keeps_the_full_bounded_acceptance_rationale():
    from ouroboros.project_dialogue import _completion_verdict

    rationale = ("Review evidence " + ("remains material and owner-visible. " * 9)).strip()
    result = _a4_result(outcome_axes={
        "execution": {"status": "ok"},
        "review": {"status": "degraded", "acceptance_decision": {
            "status": "revision_requested", "rationale": rationale,
        }},
    })

    assert len(rationale) > 240
    assert _completion_verdict(result, {}) == f"Acceptance: revision_requested — {rationale}"


def test_host_verdict_states_a_decision_without_a_rationale_alone():
    from ouroboros.project_dialogue import _completion_verdict

    bare = _a4_result(outcome_axes={
        "execution": {"status": "degraded"},
        "review": {"status": "degraded", "acceptance_decision": {"status": "revision_requested"}},
    })
    assert _completion_verdict(bare, {}) == "Acceptance: revision_requested."


def test_host_verdict_leads_both_lifecycle_rows(tmp_path, monkeypatch):
    """The verdict must reach the owner where the owner looks: the Main row's
    second line and the Project thread's terminal row."""
    from ouroboros.project_dialogue import (
        append_terminal_task_projection, enqueue_project_completion_summary,
    )
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "launch", name="Launch 🚀")
    bind_task_to_project(
        tmp_path, "root-project", project["id"], project["chat_id"],
        origin={"absent": "system"},
    )
    queued = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda _root, event, **_kwargs: queued.append(dict(event)) or True,
    )
    root = {
        "id": "root-project", "project_id": "launch",
        "title": "Ship release", "chat_id": project["chat_id"],
    }
    result = _a4_result()
    done = {"status": "completed", "outcome_axes": result["outcome_axes"]}

    assert enqueue_project_completion_summary(
        tmp_path, {"status": "completed"}, "root-project", root, result, done,
    ) is True
    assert queued[0]["text"] == (
        f"Launch 🚀 › Ship release · Done with warnings\n{A4_CLAUSE} Release shipped."
    )
    assert "final_message" not in queued[0]["text"]

    ordinary = _a4_result(
        reason_code="budget_exhausted",
        outcome_axes={"execution": {"status": "degraded"}},
    )
    assert enqueue_project_completion_summary(
        tmp_path, {"status": "completed"}, "root-project", root, ordinary,
        {"status": "completed", "outcome_axes": ordinary["outcome_axes"]},
    ) is True
    assert queued[1]["text"] == (
        "Launch 🚀 › Ship release · Done with warnings\n"
        "Reason: budget_exhausted. Release shipped."
    )

    assert append_terminal_task_projection(tmp_path, "root-project", root, result, done)
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    projection = next(row for row in rows if row.get("summary_kind") == "terminal_root_projection")
    assert A4_CLAUSE in projection["text"]
    assert "Reason: final_message" not in projection["text"]
    assert projection["reason_code"] == "final_message"
    assert projection["outcome"] == "Done with warnings"
    assert projection["text"].endswith('Details: get_task_result(task_id="root-project")')


def test_host_verdict_and_the_card_line_compose_the_same_sentence():
    """The shared fixture is the only place the two languages agree; a clause
    present there must be produced by the host verdict as well."""
    from ouroboros.project_dialogue import _completion_verdict

    for case in _parity_cases():
        clause = case.get("acceptance_clause") or ""
        if clause:
            assert _completion_verdict(case["record"], {}) == clause, case["name"]


def test_terminal_row_reports_the_depth_request_only_when_one_exists(tmp_path):
    """S3-05: the owner-visible row is where a nested swarm's depth becomes
    checkable — numbers first, and no line at all on swarms nobody asked to
    nest, so an ordinary task's row stays byte-unchanged."""
    from ouroboros.project_dialogue import append_terminal_task_projection

    task = {"id": "swarm-root", "chat_id": 3, "role": "root"}
    result = {
        "task_id": "swarm-root", "status": "completed", "result": "Shipped.",
        "outcome_axes": {"execution": {"status": "ok"}},
        "swarm_efficiency": {
            "subagent_count": 3,
            "depth": {
                "requested_depth": 2, "permitted_depth": 4, "attempted_depth": 2,
                "achieved_depth": 2, "status": "achieved", "host_visible_only": True,
            },
        },
    }
    done = {"chat_id": 3, "status": "completed", "outcome_axes": result["outcome_axes"]}
    assert append_terminal_task_projection(tmp_path, "swarm-root", task, result, done)

    flat = {"id": "flat-root", "chat_id": 3, "role": "root"}
    flat_result = {**result, "task_id": "flat-root", "swarm_efficiency": {"subagent_count": 3}}
    assert append_terminal_task_projection(tmp_path, "flat-root", flat, flat_result, done)

    rows = {
        json.loads(line)["task_id"]: json.loads(line)
        for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "Depth requested=2, permitted=4, achieved=2 (achieved)." in rows["swarm-root"]["text"]
    assert "Depth" not in rows["flat-root"]["text"]
    for marker in ("#", "**", "`"):
        assert marker not in rows["swarm-root"]["text"]
