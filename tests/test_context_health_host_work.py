"""The health block's fact about host work past one pass: present only when a law's half is broken, from the receipts."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import context_health, transcript_prefix


def _events(tmp_path, *rows):
    (tmp_path / "logs").mkdir()
    path = tmp_path / "logs" / "events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return SimpleNamespace(drive_path=lambda relative: tmp_path / relative, drive_root=tmp_path, budget_drive_root=None)


def _receipt(code="runningcode1234", **work):
    return {"type": "context_memory_view", "task_id": "t", "code": code, "work": work}


@pytest.fixture(autouse=True)
def running(monkeypatch):
    monkeypatch.setattr(transcript_prefix, "_running_code", lambda: "runningcode1234")


def _checks(env):
    checks = []
    context_health._collect_log_analysis_checks(env, checks)
    return [check for check in checks if "HOST WORK" in check]


def test_a_healthy_state_adds_nothing(tmp_path):
    env = _events(tmp_path, _receipt(view_renders=2, view_texts_kept=2, request_serializations=6, rows_read=31, rows_open=30, rows_new=1))
    assert _checks(env) == []


def test_rows_read_past_those_shown_and_new_are_named_with_their_numbers(tmp_path):
    env = _events(tmp_path, _receipt(rows_read=74963, rows_open=13834, rows_new=0))
    [line] = _checks(env)
    assert line.startswith("INFO: HOST WORK BEYOND ONE PASS under the running code (runningcode1)")
    assert "read 74,963 rows of the chat tail for 13,834 shown and 0 new" in line and "should" not in line


def test_work_done_again_is_named_with_its_verb_and_count(tmp_path):
    env = _events(tmp_path, _receipt(view_pieces=9, view_pieces_again=2, rows_read=5, rows_open=5),
                  _receipt(view_renders=2, rows_read=5, rows_open=5))
    [line] = _checks(env)
    assert "1 of 2 recent builds did work again on an input they already had (view_pieces 2)" in line
    assert "rows of the chat tail" not in line  # that half is whole


def test_an_older_code_s_receipts_and_empty_work_are_not_read(tmp_path):
    env = _events(tmp_path, _receipt(code="oldcode", rows_read=99, rows_open=1, view_pieces_again=7),
                  {"type": "context_memory_view", "code": "runningcode1234"})
    assert _checks(env) == []
