"""Presentation truth on the host side: voice, child warnings, cancel cause.

Three small contracts that the browser already keeps and the host did not:

* #1011 a host note sent through the shared seam is typed as the HOST's voice,
  so a reader cannot promote it to the card's title the way an absent key
  (legacy) still means narration;
* #1087 a child that finished with warnings reads that way in chat whichever
  axis carries the degradation, through the SHARED normalized projection;
* #1061 a cancelled row states the RECORDED cause and only the relation the
  record proves, word for word with the browser twin.
"""
from __future__ import annotations

from tests import _dialogue as dialogue_test

import json

import pytest


# ---------------------------------------------------------------------------
# #1011 — the voice of a host note
# ---------------------------------------------------------------------------

class TestHostNoteVoice:
    def _bus(self, tmp_path, monkeypatch):
        import supervisor.message_bus as mb

        monkeypatch.setattr(mb, "DATA_DIR", tmp_path)
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(mb, "load_state", lambda: {"owner_id": 1})
        sent: list[dict] = []
        monkeypatch.setattr(mb, "get_bridge", lambda: type("B", (), {
            "send_message": staticmethod(lambda *a, **kw: sent.append(dict(kw))),
        })())
        return mb, sent

    def test_an_omitted_voice_is_normalized_to_the_host_on_every_new_send(self, tmp_path, monkeypatch):
        mb, sent = self._bus(tmp_path, monkeypatch)
        mb.send_with_budget(1, "startup notice", is_progress=True, task_id="t1",
                            role="system", system_type="startup_notice")
        rows = dialogue_test.log_rows(tmp_path / "logs" / "progress.jsonl")
        # ONE meta dict reaches the durable record and the live bridge.
        assert rows[-1]["narration"] is False
        assert sent[-1]["progress_meta"]["narration"] is False

    def test_an_explicit_voice_is_preserved_in_both_directions(self, tmp_path, monkeypatch):
        mb, sent = self._bus(tmp_path, monkeypatch)
        mb.send_with_budget(1, "the model speaking", is_progress=True, task_id="t1",
                            narration=True)
        assert sent[-1]["progress_meta"]["narration"] is True
        mb.send_with_budget(1, "host note", is_progress=True, task_id="t1", narration=False)
        assert sent[-1]["progress_meta"]["narration"] is False
        # A caller that already put the fact on its own meta keeps it.
        mb.send_with_budget(1, "carried", is_progress=True, task_id="t1",
                            progress_meta={"narration": True, "card_row": "timeline"})
        from ouroboros.chat_references import resolve_progress_row
        meta = dict(sent[-1]["progress_meta"])
        stored = dialogue_test.log_rows(tmp_path / "logs" / "progress.jsonl")[-1]
        assert resolve_progress_row(tmp_path, meta.pop("row_address")) == stored
        assert meta == {"narration": True, "card_row": "timeline"}

    def test_markdown_and_the_callers_other_meta_are_untouched(self, tmp_path, monkeypatch):
        mb, sent = self._bus(tmp_path, monkeypatch)
        captured: list[tuple] = []
        monkeypatch.setattr(mb, "_send_markdown",
                            lambda *a, **kw: captured.append((a, kw)) or (True, ""))
        mb.send_with_budget(1, "**bold**", fmt="markdown", is_progress=True, task_id="t1",
                            progress_meta={"card_row": "timeline"})
        from ouroboros.chat_references import resolve_progress_row
        meta = dict(captured[-1][1]["progress_meta"])
        stored = dialogue_test.log_rows(tmp_path / "logs" / "progress.jsonl")[-1]
        assert resolve_progress_row(tmp_path, meta.pop("row_address")) == stored
        assert meta == {"card_row": "timeline", "narration": False}

    def test_the_legacy_read_is_unchanged(self):
        # ABSENT still means narration for every row written before the fact —
        # the normalization is a write-side default, never a rewrite of history.
        from ouroboros.gateway.contracts import ChatOutbound

        assert "narration" in ChatOutbound.__annotations__

    def test_nonprogress_host_voice_survives_real_history_replay(self, tmp_path, monkeypatch):
        import asyncio
        from types import SimpleNamespace
        from ouroboros.gateway.history import make_chat_history_endpoint

        mb, sent = self._bus(tmp_path, monkeypatch)
        mb.send_with_budget(1, "Host note", task_id="t1", role="system", system_type="host_progress")
        rows = dialogue_test.log_rows(tmp_path / "logs/chat.jsonl")
        assert rows[-1]["narration"] is sent[-1]["progress_meta"]["narration"] is False
        response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"})))
        replay = next(row for row in json.loads(response.body)["messages"] if row.get("text") == "Host note")
        assert replay["role"] == "system" and replay["narration"] is False


# ---------------------------------------------------------------------------
# #1087 — a child's warnings, from the shared projection
# ---------------------------------------------------------------------------

class TestChildWarnings:
    @pytest.mark.parametrize("axes", [
        {"execution": {"status": "degraded"}},
        {"objective": {"status": "degraded", "source": "task_acceptance_review"}},
        {"review": {"status": "degraded"}},
        {"objective": {"status": "best_effort", "source": "task_acceptance_review"}},
        {"objective": {"status": "pass", "warning": True}},
    ])
    def test_every_axis_the_card_folds_also_warns_in_chat(self, axes):
        from supervisor.events_task_done import _finished_with_warnings

        event = {"status": "completed", "outcome_axes": axes}
        assert _finished_with_warnings(event, {"status": "completed", "outcome_axes": axes})
        # The result alone is enough: the frame need not repeat the axes.
        assert _finished_with_warnings({"status": "completed"},
                                       {"status": "completed", "outcome_axes": axes})

    def test_a_clean_child_stays_clean_and_precedence_is_the_callers(self):
        from supervisor.events_task_done import _finished_with_warnings

        clean = {"status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}
        assert not _finished_with_warnings(clean, clean)
        # A FAILED or CANCELLED child is not "warnings": this predicate answers
        # only the warning question, and the caller asks it for completed rows.
        failed = {"status": "failed", "outcome_axes": {"execution": {"status": "failed"}}}
        assert not _finished_with_warnings(failed, failed)

    def test_a_reason_code_without_axes_still_warns(self):
        from supervisor.events_task_done import _finished_with_warnings

        event = {"status": "completed", "reason_code": "configured_actor_incomplete"}
        assert _finished_with_warnings(event, {})

    @pytest.mark.parametrize('status', ['failed', 'cancelled'])
    def test_terminal_failure_or_cancel_outranks_a_degraded_event(self, status):
        from supervisor.events_task_done import _finished_with_warnings

        event = {'status': 'completed', 'outcome_axes': {'execution': {'status': 'degraded'}}}
        assert not _finished_with_warnings(event, {'status': status})

    def test_the_host_and_browser_folds_agree_on_the_same_record(self):
        from ouroboros.project_dialogue import outcome_phase

        record = {"status": "completed", "outcome_axes": {"review": {"status": "degraded"}}}
        assert outcome_phase(record, {}) == "warn"

    def test_a_completed_lifecycle_with_a_failed_outcome_never_reads_as_a_clean_completion(self):
        # R2 (#1087): `_finished_with_warnings` answers only the warning question
        # (False for `error`), so the chat line took ✅ from the lifecycle while the
        # card folded the same record to Failed. The display helper speaks the phase.
        from ouroboros.project_dialogue import outcome_phase
        from supervisor.events_task_done import _completed_lifecycle_display

        failed_review = {"status": "completed", "outcome_axes": {"review": {"status": "fail"}}}
        assert outcome_phase(failed_review, {}) == "error"
        assert _completed_lifecycle_display({"status": "completed"}, failed_review) == ("❌", "finished with a failed outcome")
        warned = {"status": "completed", "outcome_axes": {"review": {"status": "degraded"}}}
        assert _completed_lifecycle_display({"status": "completed"}, warned) == ("⚠️", "finished with warnings")
        clean = {"status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}
        assert _completed_lifecycle_display(clean, clean) is None


def test_cancel_cause_python_browser_fixture_parity():
    from pathlib import Path
    from ouroboros.project_dialogue import _completion_verdict

    cases = json.loads((Path(__file__).parents[1] / "web/tests/fixtures/cancel_cause_parity.json").read_text(encoding="utf-8"))
    for case in cases:
        assert _completion_verdict(case["record"], {}) == case["text"], case


def test_cancel_cause_is_bounded_by_unicode_characters():
    from ouroboros.project_dialogue import _completion_verdict

    assert _completion_verdict({"status": "cancelled", "cancel_origin": {
        "reason": "🙂" * 200,
    }}, {}) == "🙂" * 159 + "… (preview; the full reason is kept with the task)"
    assert _completion_verdict({"status": "cancelled", "cancel_origin": {
        "reason": "🙂" * 160,
    }}, {}) == "🙂" * 160 + "."


def test_settled_early_speech_keeps_canonical_warning_axes(tmp_path):
    from ouroboros.gateway.history import _assemble_history_response
    from ouroboros.task_finalization import stamp_root_final_phase
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor.message_bus import log_chat

    axes = {'execution': {'status': 'degraded'}}
    event = {'progress_meta': {'outcome_axes': axes}}
    stamp_root_final_phase(event, {}, post_task_open=True, terminal_status='completed')
    log_chat('out', 1, 1, 'Retained early answer', task_id='early',
             message_meta=event['progress_meta'], drive_root=tmp_path)
    write_task_result(tmp_path, 'early', 'completed', outcome_axes=axes,
                      root_phase_checkpoint={'post_task_synthesis': 'completed'})
    append_jsonl(tmp_path/'logs/progress.jsonl', {'task_id': 'early', 'chat_id': 1,
        'content': 'Working', 'ts': '2026-09-23T10:00:00Z'})
    append_jsonl(tmp_path/'logs/chat.jsonl', {'task_id': 'early', 'chat_id': 1,
        'direction': 'system', 'type': 'task_summary', 'text': 'Summary',
        'ts': '2026-09-23T10:00:01Z'})
    messages = json.loads(_assemble_history_response(tmp_path, 1, 50, 200))['messages']
    speech = next(row for row in messages if row.get('text') == 'Retained early answer')
    assert speech['task_terminal_status'] == 'completed'
    assert speech['outcome_axes']['execution']['status'] == 'degraded'
    assert speech['outcome_phase'] == 'warn'
