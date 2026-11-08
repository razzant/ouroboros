"""A finished task's notifications name its work (``target_label``) beside the outcome word.

The root's terminal row keeps its own ``text`` byte for byte ("Done. Root task …": Ouroboros's memory and the wake
fixtures read it); the row and the ``task_done`` event add the work's name from the rows the writer already holds, and
nothing when the task has none. The browser banner and Telegram read the label; neither invents a "Task".
"""
from __future__ import annotations

import asyncio
import json

from ouroboros import terminal_projection
from ouroboros.task_finalization import work_label
from tests.test_completion_notices import _record
from tests.test_task_done_durable_vs_transport import _mk_ctx


def test_the_root_row_names_its_work_and_keeps_its_own_text():
    result = {**_record("clean"), "title": "Publish the release notes"}
    row = terminal_projection._project_row("rel", result, {}, {"chat_id": 1})
    assert row["target_label"] == "Publish the release notes"
    assert row["text"].startswith(f"{row['outcome']}. Root task rel.") and "Publish" not in row["text"]
    # A coined name speaks when the request has no title; without either nothing is invented.
    coined = terminal_projection._project_row("rel", {**_record("clean"), "suggested_name": "Release notes"}, {}, {"chat_id": 1})
    assert coined["target_label"] == "Release notes"
    unnamed = terminal_projection._project_row("bare", _record("clean"), {}, {"chat_id": 1})
    assert "target_label" not in unnamed and unnamed["text"].startswith(f"{unnamed['outcome']}. Root task bare.")
    assert work_label({}, None, {"title": "  "}) == {}


def test_the_task_done_event_carries_the_work_name_on_its_log_and_live_channel(tmp_path):
    from ouroboros.headless import prepare_terminal_task_files
    from ouroboros.task_results import write_task_result
    from supervisor.events import _handle_task_done

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    for task_id, title in (("named", "Rename the config keys"), ("bare", "")):
        sent, pushed = [], []
        ctx = _mk_ctx(tmp_path, sent, pushed)
        task = {"id": task_id, "type": "task", "chat_id": 1, **({"title": title} if title else {})}
        ctx.RUNNING[task_id] = {"task": task}
        write_task_result(tmp_path, task_id, "completed", result="done", chat_id=1, **({"title": title} if title else {}))
        assert not prepare_terminal_task_files(tmp_path, task)["error"]
        _handle_task_done({"type": "task_done", "task_id": task_id, "task_type": "task", "worker_id": 0, "chat_id": 1,
                           "ts": "2026-11-03T00:00:00Z", "_files_prepared_attempt": 1}, ctx)
        rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        (logged,) = [r for r in rows if r.get("type") == "task_done" and r.get("task_id") == task_id]
        (live,) = [p for p in pushed if p.get("type") == "task_done"]
        expected = {"target_label": title} if title else {}
        assert {key: logged[key] for key in ("target_label",) if key in logged} == expected
        assert {key: live[key] for key in ("target_label",) if key in live} == expected


def test_the_live_banner_says_the_name_word_and_cause_from_the_frame_the_producer_pushes(tmp_path):
    """A root's ending goes live only as its ``task_done`` event (the ``task_summary`` row is stored, never sent): the
    banner that frame rings names the work and its word, then the cause the card says over the same facts."""
    import pathlib
    import subprocess

    from ouroboros.headless import prepare_terminal_task_files
    from ouroboros.preflight_node import resolve_node
    from ouroboros.task_results import write_task_result
    from supervisor.events import _handle_task_done

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    frames = {}
    for task_id, status, facts in (("crashed", "failed", {"reason_code": "worker_crash_signal"}), ("clean", "completed", {})):
        sent, pushed = [], []
        ctx = _mk_ctx(tmp_path, sent, pushed)
        task = {"id": task_id, "type": "task", "chat_id": 1, "title": "Rename the config keys"}
        ctx.RUNNING[task_id] = {"task": task}
        write_task_result(tmp_path, task_id, status, result="done", chat_id=1, title=task["title"], **facts)
        assert not prepare_terminal_task_files(tmp_path, task)["error"]
        _handle_task_done({"type": "task_done", "task_id": task_id, "task_type": "task", "worker_id": 0, "chat_id": 1,
                           "status": status, "ts": "2026-11-03T00:00:00Z", "_files_prepared_attempt": 1}, ctx)
        (frames[task_id],) = [p for p in pushed if p.get("type") == "task_done"]
        assert "reason_detail" not in frames[task_id], "the event carries the facts, not the sentence"
        assert not [p for p in pushed if "task_summary" in (p.get("type"), p.get("system_type"))]
        assert not [s for s in sent if "task_summary" in repr(s)]
    node = resolve_node()
    assert node, "the live banner proof requires Node"
    script = """import { classifyLiveFrame } from './web/modules/notifications.js';
let text = ''; for await (const chunk of process.stdin) text += chunk;
const said = Object.entries(JSON.parse(text)).map(([id, frame]) => [id, classifyLiveFrame(frame, { kind: 'log' })?.body]);
console.log(JSON.stringify(Object.fromEntries(said)));"""
    # Node writes UTF-8 to a pipe on every host; a bare text=True would decode it with the locale's codec (cp1252).
    run = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(frames), text=True, encoding="utf-8",
                         cwd=pathlib.Path(__file__).resolve().parents[1], capture_output=True, timeout=60, check=True)
    assert json.loads(run.stdout) == {"crashed": "Rename the config keys · Failed: The worker process running the task crashed",
                                      "clean": "Rename the config keys · Done"}


def test_history_replays_the_rows_work_name():
    from ouroboros.gateway.history import _copy_task_summary_metadata

    rec: dict = {}
    _copy_task_summary_metadata(rec, {"type": "task_summary", "target_label": "Release notes", "outcome_phase": "done"})
    assert rec["target_label"] == "Release notes"
    rec = {}
    _copy_task_summary_metadata(rec, {"type": "task_summary", "outcome_phase": "done"})
    assert "target_label" not in rec


def test_telegram_says_the_work_name_and_only_falls_back_to_the_task_id_without_one(tmp_path, monkeypatch):
    from tests.test_telegram_notifications import _api, _load, _Rec, _write_rows

    nt = _load()
    api, data = _api(tmp_path)
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    named = terminal_projection._project_row("rel", {**_record("partial"), "title": "Publish the release notes"},
                                            {}, {"chat_id": 1})
    bare = terminal_projection._project_row("bare", _record("partial"), {}, {"chat_id": 1})
    _write_rows(data, [named, bare])
    _Rec.sent = []
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
    first_lines = [message.split("\n")[0] for _chat, message, *_ in _Rec.sent]
    assert any(line.startswith("⚠️ Publish the release notes · ") for line in first_lines), first_lines
    assert any("Task bare" in line for line in first_lines), first_lines
    assert not any("Root task" in message for _chat, message, *_ in _Rec.sent)
