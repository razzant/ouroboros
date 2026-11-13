"""Typed Claudexor text fragments must not become punctuated event labels."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ouroboros.delegate_progress import (
    WindowObservations, present_payload, timeline_tail,
)


def text_event(text, *, kind="thinking", delta=True, attempt="a01", harness="cursor"):
    return {
        "type": "harness.event", "title": text[:500], "detail": text,
        "severity": "info", "textKind": kind, "textDelta": delta,
        "attemptId": attempt, "harnessId": harness,
    }


def test_text_body_wins_over_the_engine_title_preview():
    event = text_event("  phrase\n\nnext paragraph ")
    event["title"] = "phrase next paragraph"
    original = deepcopy(event)
    assert timeline_tail({"timeline": [event]})[0]["title"] == event["detail"]
    assert event == original


@pytest.mark.parametrize("metadata", [
    {}, {"textKind": None, "textDelta": False},
    {"textKind": "thinking", "textDelta": True, "detail": None},
])
def test_legacy_or_unknown_metadata_keeps_event_labels(metadata):
    rows = [{"type": "harness.event", "title": word, **metadata} for word in ["UX", "plan"]]
    assert timeline_tail({"timeline": rows}) == [
        {"type": "harness.event", "title": word, "severity": None} for word in ["UX", "plan"]
    ]


def test_verbose_text_remains_disclosed_and_the_payload_fits():
    events = [text_event("x" * 20000, kind="message", delta=False) for _ in range(16)]
    seen = WindowObservations()
    advance = seen.record({"timeline": events}, 38)
    assert all("original length 20000" in row["title"] for row in advance.events)
    assert all("detail" not in row for row in advance.events)
    payload = present_payload(
        run_id="run-1", state="running", last_seq=38, moved=True,
        elapsed_seconds=120, max_seconds=1800, waiting_on_user=False,
        detail={"timeline": events}, budget=15000,
    )
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    assert len(raw) <= 15000
    assert json.loads(raw) == payload


def test_start_receipt_names_the_serving_engine_without_claiming_review_success():
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _started_payload, get_tools
    from ouroboros.delegate_start_instructions import HOST_INSTRUCTIONS

    payload = json.loads(_started_payload(
        {"runDir": "/fixture/run"}, "run-1", SimpleNamespace(route_id="selected", model="m", effort="high"),
        "readonly", delegated_run_shape(False), "/fixture/repo", durable=True, recovering=False,
        invocation_id="invocation-1", snapshot_id="", target_root="", baseline_sha="",
        engine_version="3.9.7",
    ).text)
    assert payload["engine_version"] == "3.9.7"
    assert payload["status"] == "started"
    assert "review_passed" not in payload
    description = next(tool.schema["description"] for tool in get_tools() if tool.name == "delegate_start")
    assert "requests no extra Claudexor review panel" in description
    assert "recovered historical run" in description
    assert "self_worktree capture separately requires an unchanged HEAD" in HOST_INSTRUCTIONS
    assert "preserve committed changes" in HOST_INSTRUCTIONS


def test_an_open_run_says_what_ending_this_task_does_to_it():
    """The nanny reads, where the run's state is read, that ending now cancels the open run on the next custody
    pass and which endings spare it (audit 11.11, item 18: the fact at its birth, not a return at the finish door)."""
    payload = present_payload(run_id="run-1", state="running", last_seq=3, moved=False, elapsed_seconds=60,
                              max_seconds=None, waiting_on_user=False, detail={}, budget=15000)
    fact = payload["owner_end"]
    assert "cancels the run (owner_task_gone)" in fact and "spares it" in fact
    assert "deadline, budget or round limit" in fact, "a forced wrap-up is a deliberate end too"

