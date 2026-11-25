"""A host notice: one phrase, its next step and its details (DESIGN principle 7, ``ouroboros/host_notice.py``).

The producer writes the phrase as the row's text and the typed notice beside it; the row keeps the notice through
persistence and history; readers are sent one line, the phrase and its step in the install's language; the mind reads
the step and the raw details beside the phrase, so a raw error it read in an older row's words stays in its view.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import typing
from types import SimpleNamespace

import pytest

from ouroboros import host_notice
from ouroboros.dialogue_provenance import dialogue_text
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.history_contracts import ChatHistoryMessage
from ouroboros.gateway.schema import validate_ingress
from supervisor import message_bus
from tests.test_telegram_i18n import _seed

RAW = "PhysicalAttemptPreparationFailed: StoreBusy: current store lock unavailable: /Users/x/packs_publication.lock"


def test_a_notice_carries_a_known_step_and_its_raw_details_and_refuses_words_nobody_wrote():
    assert host_notice.notice("task_exception", details=RAW) == {"code": "task_exception", "details": RAW}
    assert host_notice.notice("budget_exhausted", step="raise_total_budget", details="  ") == {
        "code": "budget_exhausted", "step": "raise_total_budget"}
    with pytest.raises(ValueError):
        host_notice.notice("x", step="press_the_magic_button")
    assert host_notice.stored({"notice": {"code": "c", "details": "d", "line": "the reader's line"}}) == {
        "code": "c", "details": "d"}, "a row persists no reader's line"
    assert host_notice.notice_of({"notice": {"details": "no code"}}) == {}


def test_the_raw_error_the_mind_read_in_an_older_row_stays_in_its_view():
    """BIBLE P1: the old row said the exception in its words; the notice row says a phrase, and the mind reads the
    same exception beside it (``dialogue_text``, the one reader of history, search and memory rows)."""
    old = {"direction": "system", "role": "system", "text": f"⚠️ Error during processing: {RAW}"}
    notice = {"direction": "system", "role": "system", "text": "This task stopped on an unexpected error.",
              **host_notice.fields("task_exception", details=RAW)}
    assert RAW in dialogue_text(old) and RAW in dialogue_text(notice)
    assert dialogue_text(notice) == f"This task stopped on an unexpected error.\n[Details: {RAW}]"
    stepped = {"text": "The total budget is spent, so this turn did not start.",
               **host_notice.fields("budget_exhausted", step="raise_total_budget")}
    assert dialogue_text(stepped).endswith(f"\n[Step: {host_notice.STEPS['raise_total_budget']}]")
    from ouroboros.memory import Memory

    line = Memory._format_chat_line({**notice, "ts": "2026-11-21T12:00:00Z"}, compact=True)
    assert line.startswith("📋 12:00 [system] This task stopped") and RAW in line
    assert RAW in Memory.summarize_progress(SimpleNamespace(), [{**notice, "ts": "2026-11-21T12:00:00Z"}])


def test_the_one_line_joins_the_phrase_and_its_step_each_translated_first(tmp_path, monkeypatch):
    phrase = "The total budget is spent, so this turn did not start."
    row = {"text": phrase, "role": "system", **host_notice.fields("budget_exhausted", step="raise_total_budget")}
    assert host_notice.notice_line(row, "") == f"{phrase} {host_notice.STEPS['raise_total_budget']}"
    _seed(tmp_path, "ru", {phrase: {"text": "Бюджет исчерпан."},
                           "code:notice.step.raise_total_budget": {"text": "Поднимите его."}})
    assert host_notice.notice_line(row, "ru", drive_root=tmp_path) == "Бюджет исчерпан. Поднимите его."
    # While the phrase has no translation the step stays English too: the line is in one language (RV4 17).
    _seed(tmp_path, "de", {"code:notice.step.raise_total_budget": {"text": "Erhöhen."}})
    assert host_notice.notice_line(row, "de", drive_root=tmp_path) == f"{phrase} {host_notice.STEPS['raise_total_budget']}"
    # The technical pause is translated by its text, as Telegram did before notices; other rows keep their words.
    pause = {"text": "Paused for its authority.", "role": "system", "system_type": "task_pause_notice",
             "pause_reason": "authority"}
    _seed(tmp_path, "ru", {pause["text"]: {"text": "Пауза."}, "Some host words.": {"text": "Нет."}})
    assert host_notice.notice_line(pause, "ru", drive_root=tmp_path) == "Пауза."
    assert host_notice.notice_line({**pause, "pause_reason": "owner"}, "ru", drive_root=tmp_path) == pause["text"]
    assert host_notice.notice_line({"text": "Some host words.", "role": "system"}, "ru", drive_root=tmp_path) == "Some host words."


@pytest.fixture
def chat(tmp_path, monkeypatch):
    """The real bridge over a disposable data root; frames are what a socket would receive."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "progress.jsonl").touch()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    events: list = []
    monkeypatch.setattr(message_bus, "publish_event", lambda _kind, event: events.append(event))
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    bridge = message_bus.LocalChatBridge({})
    frames: list = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    return SimpleNamespace(frames=frames, events=events, root=tmp_path)


def test_a_notice_row_keeps_its_notice_and_every_reader_is_sent_its_line(chat):
    phrase = "This turn stopped on an unexpected error."
    message_bus.send_with_budget(1, phrase, task_id="t1", role="system", system_type="task_error",
                                 progress_meta=host_notice.fields("task_exception", details=RAW))
    message_bus.send_with_budget(1, "No conversation carrier could start, so this turn continues inside the server "
                                    "process.", is_progress=True, task_id="t1", role="system", system_type="worker_failure",
                                 progress_meta=host_notice.fields("conversation_degraded_executor", details="boot"))
    live = [frame for frame in chat.frames if frame.get("notice")]
    assert [frame["notice"]["line"] for frame in live] == [phrase, live[1]["content"]]
    assert live[0]["notice"]["details"] == RAW and live[0]["content"] == phrase, "the raw error is not its words"
    assert [event["notice"]["line"] for event in chat.events if event.get("notice")] == [phrase, live[1]["content"]]
    response = asyncio.run(make_chat_history_endpoint(chat.root)(SimpleNamespace(query_params={})))
    rows = [row for row in json.loads(response.body.decode("utf-8"))["messages"] if row.get("notice")]
    assert [(row["text"], row["notice"]) for row in rows] == [(frame["content"], frame["notice"]) for frame in live]
    for row in rows:
        assert not validate_ingress(row, ChatHistoryMessage), validate_ingress(row, ChatHistoryMessage)
    assert "notice" in typing.get_type_hints(ChatHistoryMessage)


def test_the_persisted_row_keeps_the_notice_without_any_readers_line(tmp_path):
    from supervisor.message_ingress import _chat_record

    _root, record = _chat_record("system", 1, 7, "A phrase.", drive_root=tmp_path, message_meta={
        "notice": {"code": "c", "step": "restart", "details": "raw", "line": "A phrase. Restart Ouroboros to recover."}})
    assert record["notice"] == {"code": "c", "step": "restart", "details": "raw"}
    _root, plain = _chat_record("system", 1, 7, "A phrase.", drive_root=tmp_path, message_meta={})
    assert "notice" not in plain


def test_a_host_notice_terminal_says_its_phrase_while_the_record_keeps_the_error(tmp_path):
    """``agent._task_exception_terminal``'s notice reaches the row through the terminal seam; a published text the
    host joined other words to (a suffix) keeps its whole text instead."""
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE, exception_notice, prepare_terminal_send_event

    notice = exception_notice(ValueError("boom"))
    assert notice["record"] == "This task stopped on an unexpected error: ValueError: boom"
    usage = {"terminal_origin": TERMINAL_ORIGIN_HOST_NOTICE, "host_notice_row": notice,
             "execution_status": "infra_failed", "reason_code": "task_exception"}
    task = {"id": "t-ex", "chat_id": 7}
    base = {"type": "send_message", "chat_id": 7, "task_id": "t-ex", "text": notice["record"], "log_text": notice["record"]}
    event = prepare_terminal_send_event(tmp_path, task, notice["record"], dict(usage), dict(base), presence=False)
    assert (event["text"], event["log_text"], event["role"]) == ("This task stopped on an unexpected error.",) * 2 + ("system",)
    assert event["progress_meta"]["notice"] == {"code": "task_exception", "details": "ValueError: boom"}
    joined = notice["record"] + "\n\nPlan review was still open."
    kept = prepare_terminal_send_event(tmp_path, task, joined, dict(usage), {**base, "text": joined}, presence=False)
    assert kept["text"] == joined and "notice" not in (kept.get("progress_meta") or {})
    # The outcome is classified by its typed status and reason, not by its words (no prefix matches them now).
    outcome = derive_loop_outcome(notice["record"], usage, {})
    assert outcome["reason_code"] == "task_exception" and outcome["failure"]["kind"] == "runtime"
    review = derive_loop_outcome("Deep self-review failed: X: y", {"execution_status": "infra_failed",
                                                                  "reason_code": "deep_self_review_error"}, {})
    assert review["failure"]["kind"] == "runtime", "a deep review's own failure is the runtime's, as its old prefix said"


# --- RV4: the review's accepted findings, each pinned where it would fail without its fix ---------------------------


def test_a_variable_phrase_translates_its_template_and_never_its_words(tmp_path):
    """RV4 3: the mind's note, a name or a count is put back as written; only the constant template is translated."""
    text, said = host_notice.phrase("project_closed", "Ouroboros closed this project. {note}", step="reopen_project",
                                    note="Готово, всё сдано.")
    assert text == "Ouroboros closed this project. Готово, всё сдано."
    row = {"text": text, "role": "system", **said}
    assert host_notice.stored(row)["values"] == {"note": "Готово, всё сдано."}, "the template and values persist"
    _seed(tmp_path, "ru", {"Ouroboros closed this project. {note}": {"text": "Уроборос закрыл проект. {note}"},
                           "code:notice.step.reopen_project": {"text": "Сообщение здесь откроет его снова."}})
    assert host_notice.notice_line(row, "ru", drive_root=tmp_path) == (
        "Уроборос закрыл проект. Готово, всё сдано. Сообщение здесь откроет его снова.")


def test_a_read_never_offers_a_phrase_and_the_send_offers_its_template_in_its_own_root(tmp_path, monkeypatch):
    """RV4 3 and 19: history, search and an operation read without writing; the send offers the template once, and a
    refused offer is offered again."""
    from ouroboros import i18n_memory

    host_notice._MISSES.clear()
    text, said = host_notice.phrase("skill_review_interrupted", "Skill review: `{skill}` — interrupted", skill="alpha")
    row = {"text": text, "role": "system", **said}
    host_notice.view(row, drive_root=tmp_path)  # a read, on a Russian install
    monkeypatch.setattr(i18n_memory, "current_language", lambda: "ru")
    host_notice.view(row, drive_root=tmp_path)
    assert not i18n_memory.pending_path(tmp_path, "ru").exists(), "a read writes nothing and wakes nothing"
    offered = []
    real = i18n_memory.record_missing
    monkeypatch.setattr(i18n_memory, "record_missing", lambda root, tag, items: (
        offered.append((root, [item["key"] for item in items])), {"accepted": 0, "dropped": 1, "pending": 0})[1])
    host_notice.view(row, drive_root=tmp_path, record_misses=True)
    host_notice.view(row, drive_root=tmp_path, record_misses=True)
    assert offered == [(tmp_path, ["Skill review: `{skill}` — interrupted"])] * 2, "a refused offer is offered again"
    monkeypatch.setattr(i18n_memory, "record_missing", real)
    from ouroboros import ui_translation
    woke = []  # the generator's wake is recorded, not run: a live generator would drain the queue this test reads
    monkeypatch.setattr(ui_translation, "on_language_event", lambda *event: woke.append(event))
    host_notice.view(row, drive_root=tmp_path, record_misses=True)
    host_notice.view(row, drive_root=tmp_path, record_misses=True)
    pending = json.loads(i18n_memory.pending_path(tmp_path, "ru").read_text(encoding="utf-8"))["items"]
    assert list(pending) == ["Skill review: `{skill}` — interrupted"] and pending[list(pending)[0]]["count"] == 1
    assert woke == [("missing", tmp_path, "ru")], "an accepted offer wakes the generator once"


def test_search_and_the_room_snapshot_carry_the_notice(tmp_path):
    """RV4 2 and 16: a search candidate's words are the line the bubble draws; the plan review's snapshot keeps it."""
    from ouroboros.dialogue_evidence import _row_projection
    from ouroboros.gateway.chat_search import _candidate
    from ouroboros.gateway.history_contracts import ChatSearchCandidate

    phrase = "The total budget is spent, so this turn did not start."
    entry = {"history_id": "chat:1", "row_address": "row:1@t#0", "role": "system", "direction": "system",
             "system_type": "task_not_started", "ts": "2026-11-22T10:00:00Z", "text": phrase,
             **host_notice.fields("budget_exhausted", step="raise_total_budget", details="setting: TOTAL_BUDGET")}
    found = _candidate(entry, tmp_path)
    assert found["notice"]["line"] == f"{phrase} {host_notice.STEPS['raise_total_budget']}"
    assert not validate_ingress(found, ChatSearchCandidate) and "notice" in typing.get_type_hints(ChatSearchCandidate)
    assert _row_projection(entry, "chat", 1)["notice"]["details"] == "setting: TOTAL_BUDGET"


def test_a_refused_turn_keeps_its_step_in_the_result_and_the_key_in_details():
    """RV4 4: the result is the canonical record (phrase and step); the setting's key is the notice's detail."""
    from supervisor.worker_chat_lane import _budget_spent_notice

    said = _budget_spent_notice()
    assert said["record"] == f"{said['text']} {host_notice.STEPS['raise_total_budget']}"
    assert said["notice"] == {"code": "budget_exhausted", "step": "raise_total_budget", "details": "setting: TOTAL_BUDGET"}


def test_an_update_outcome_is_worded_from_its_observed_outcome_and_its_verified_stash():
    """RV4 6, RVD4 6 and Fable N1: the phrase follows the outcome the boot branch observed, never a finalized flag
    alone; a deferred rollback has only recorded its switch; "not lost" is said only of a stash Git verifiably kept."""
    from supervisor.update_merge import boot_outcome_notice

    said = lambda **result: boot_outcome_notice({"stash_note": "the note", **result})["text"]
    assert said(finalized=True, stash_restore_status="restored") == "The update is in place, and your local changes were put back."
    assert said(finalized=True, observed={"outcome": "applied"}) == "The update is in place."
    assert said(finalized=True, observed={"outcome": "rolled_back"}) == "The last update was rolled back."
    assert said(finalized=True, observed={"outcome": "not_applied"}) == "The last update was not applied."
    assert said(rolled_back=True, rollback_deferred=True) == "The last update will be rolled back as the app restarts."
    assert said(rolled_back=True) == "The last update was rolled back."
    assert said(finalized=False, stash_restore_status="preserved") == (
        "The last update did not finish cleanly; Git kept your local changes aside, so they are not lost.")
    assert said(finalized=True, stash_restore_status="incomplete") == (
        "The update is in place; your local changes have not been put back yet.")
    kept = boot_outcome_notice({"finalized": False, "stash_restore_status": "preserved", "stash_note": "git stash apply X"})
    assert kept["progress_meta"]["notice"]["details"] == "git stash apply X" and "Details" not in kept["text"]


def test_a_second_boot_after_a_rollback_and_a_deferred_rollback_say_what_happened(monkeypatch):
    """RVD4 6 on the boot branches themselves: the restored generation's marker cleanup observed a rollback, and the
    boot-smoke door that only recorded its switch leaves HEAD on the update until the restart."""
    from ouroboros import deployment_policy
    from supervisor import update_merge

    monkeypatch.setattr(update_merge, "_log_supervisor", lambda _row: True)
    monkeypatch.setattr(deployment_policy, "body_replacement_refusal", lambda: "")
    monkeypatch.setattr(update_merge, "clear_update_tx", lambda: True)
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("ok", {
        "phase": update_merge.MARKER_CLEANUP_RETRY_PHASE, "gate_blocked_reason": "rollback_restart_pending",
        "gate_blocked_detail": "rolled back to 0123456789ab", "attempt_id": "a1", "pre_update_sha": "0123456789ab"}))
    cleanup = update_merge._finalize_managed_update_locked(True)
    assert cleanup["finalized"] is True and cleanup["observed"]["outcome"] == "rolled_back"
    assert update_merge.boot_outcome_notice(cleanup)["text"] == "The last update was rolled back."
    monkeypatch.setattr(update_merge, "_boot_smoke_rollback",
                        lambda _tx, _reason: (True, "authority switch recorded", {"rollback_deferred": True}))
    monkeypatch.setattr(update_merge._g, "git_capture", lambda _cmd: (0, "f" * 40, ""))
    deferred = update_merge._finalize_pending_boot_smoke(
        {"merge_commit": "e" * 40, "strategy": "replace", "boot_attempts": 1}, True)
    assert deferred["rolled_back"] is True and deferred["rollback_deferred"] is True
    assert update_merge.boot_outcome_notice(deferred)["text"] == "The last update will be rolled back as the app restarts."


KEPT_SHA = "0123456789abcdef0123"
KEPT_NOTE = ('conflicting paths: "a.py"; the original local changes remain in stash 0123456789ab and current files '
             f"should be inspected before recovering with `git stash apply {KEPT_SHA}`\nGit diagnostics:\nCONFLICT (content)")


def test_only_a_kept_stash_sends_the_owner_to_its_git_command_and_the_window_line_stays_one_line():
    """RV4 15 and RVD4 15 (Q1=C): a kept stash carries the step and its exact command as the notice's typed one-line
    instruction (an interrupted apply's step asks a look at the current files first), the whole note as its details; a
    restored, rolled-back or unfinished restore carries neither. The closed table still refuses a step nobody wrote,
    and the window's line holds no command and names no Details."""
    from supervisor.update_merge import boot_outcome_notice

    command = f"git stash apply {KEPT_SHA}"
    kept = boot_outcome_notice({"finalized": True, "stash_restore_status": "preserved", "stash_note": KEPT_NOTE,
                                "stash_command": command, "stash_interrupted": False})
    assert kept["progress_meta"]["notice"] == {
        "code": "managed_update_finalized", "step": "restore_kept_changes", "details": KEPT_NOTE, "instruction": command}
    interrupted = boot_outcome_notice({"finalized": True, "stash_restore_status": "preserved", "stash_note": KEPT_NOTE,
                                       "stash_command": command, "stash_interrupted": True})["progress_meta"]["notice"]
    assert (interrupted["step"], interrupted["instruction"]) == ("inspect_then_restore_kept_changes", command)
    assert "check the current files first" in host_notice.STEPS["inspect_then_restore_kept_changes"]
    for result in ({"finalized": True, "stash_restore_status": "restored", "stash_note": "local changes restored"},
                   {"rolled_back": True, "stash_note": "rolled back to 0123456789ab"},
                   {"finalized": False, "stash_restore_status": "incomplete", "stash_note": "Do not apply it.",
                    "stash_command": command}):
        notice = boot_outcome_notice(result)["progress_meta"]["notice"]
        assert "step" not in notice and "instruction" not in notice, result
    with pytest.raises(ValueError):
        host_notice.notice("managed_update_finalized", step="restore_changes", details=KEPT_NOTE)
    assert host_notice.notice("c", instruction=" git stash\n apply  x ")["instruction"] == "git stash apply x", "one line"
    line = host_notice.notice_line({"text": kept["text"], "role": "system", **kept["progress_meta"]}, "")
    assert line == f"{kept['text']} {host_notice.STEPS['restore_kept_changes']}" and "git stash" not in line
    assert "Details" not in host_notice.STEPS["restore_kept_changes"]
    assert host_notice.stored({"notice": kept["progress_meta"]["notice"]})["instruction"] == command, "history keeps it"


def test_crash_recovery_keeps_the_record_a_notice_row_was_sent_with(tmp_path, monkeypatch):
    """RV4 7: the outbox and the delivered receipt keep the canonical record beside the phrase; recovery adopts it."""
    from ouroboros import working_checkpoint
    from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE, exception_notice, prepare_terminal_send_event
    from supervisor import terminal_delivery

    notice = exception_notice(ValueError("boom"))
    usage = {"terminal_origin": TERMINAL_ORIGIN_HOST_NOTICE, "host_notice_row": notice}
    event = prepare_terminal_send_event(tmp_path, {"id": "t-rec", "chat_id": 7}, notice["record"], usage,
                                        {"type": "send_message", "chat_id": 7, "text": notice["record"]}, presence=False)
    assert event["terminal_record"] == notice["record"] and event["text"] == notice["text"]
    monkeypatch.setattr(terminal_delivery, "terminal_answer_receipts", lambda root, task_id: {"owed": [
        {**event, "delivery_id": "final:task:t-rec", "registered_at": "2026-11-22T10:00:00Z"}]})
    assert working_checkpoint.registered_answer(tmp_path, "t-rec")["text"] == notice["record"]
    monkeypatch.setattr(terminal_delivery, "terminal_answer_receipts", lambda root, task_id: {"owed": [
        {"text": "An older row's text.", "delivery_id": "final:task:t-old", "registered_at": "2026-11-22T10:00:00Z"}]})
    assert working_checkpoint.registered_answer(tmp_path, "t-old")["text"] == "An older row's text.", "older rows as before"


def test_a_typed_answer_is_never_classified_by_an_old_prefix():
    """RV4 8 and RVD4 8: the prefix table reads only text whose producer typed nothing. The loop stamps its model's
    final answer ``model_final`` and no status (``_handle_text_response``); that answer quoting an old host prefix stays
    the model's answer. Host text that typed nothing keeps the old reading."""
    from ouroboros.loop_messages import _handle_text_response
    from ouroboros.outcomes import derive_loop_outcome

    text, usage, trace = _handle_text_response("⚠️ Error during processing: quoted from a log", {"reasoning_notes": []}, {})
    assert usage == {"terminal_origin": "model_final"}, "the loop's own stamping, no execution status"
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    typed = derive_loop_outcome(text, {"execution_status": "ok", "reason_code": "final_message"}, {})
    assert typed["outcome_axes"]["execution"]["status"] == "ok"
    untyped = derive_loop_outcome("⚠️ Error during processing: RuntimeError: x", {}, {})
    assert untyped["outcome_axes"]["execution"]["status"] == "infra_failed"


def test_a_failed_review_keeps_its_notice_with_its_ledger_record(tmp_path):
    """RV4 9: the review's ledger suffix joins the failure's record and details, so the notice survives."""
    from ouroboros.deep_self_review import deep_review_unavailable, with_review_record
    from ouroboros.task_finalization import prepare_terminal_send_event

    text, usage = deep_review_unavailable("subscription_window_exhausted")
    joined = with_review_record(text, usage, "rv-1")
    assert joined == usage["host_notice_row"]["record"] and joined.endswith("Review record: rv-1 (surface=system)")
    assert usage["host_notice_row"]["notice"]["details"] == "subscription_window_exhausted; Review record: rv-1 (surface=system)"
    event = prepare_terminal_send_event(tmp_path, {"id": "t-rv", "chat_id": 7}, joined, usage,
                                        {"type": "send_message", "chat_id": 7, "text": joined}, presence=False)
    assert event["text"] == "Deep self-review cannot run on this reviewer row." and "notice" in event["progress_meta"]


def test_a_provider_outage_says_a_stable_phrase_with_its_diagnostic_behind_details():
    """RV4 10: the provider's own words (its last code, a plan note) and the task's id are details, never the line."""
    from ouroboros.task_finalization import send_provider_death_notice

    sent = []
    ctx = SimpleNamespace(send_with_budget=lambda chat_id, text, **kw: sent.append((text, kw)))
    diagnostic = "⏳ The provider kept answering 529 overloaded_error; Ouroboros waited 12 minutes."
    assert send_provider_death_notice(ctx, 7, "t-down", {"terminal_provider_notice": diagnostic})
    [(text, kw)] = sent
    assert text == "A model-provider outage stopped this task; its partial work and workspace files are kept."
    notice = kw["progress_meta"]["notice"]
    assert notice["step"] == "inspect_task" and diagnostic in notice["details"] and "task t-down" in notice["details"]


def test_a_full_local_slot_is_worded_as_local_capacity_never_a_provider_outage(tmp_path):
    """RVD4 10: the terminal's typed cause words the incident. The loop's own provider terminal over a full local model
    slot (nothing dispatched) stamps the install's capacity; with a retained model answer as the primary, the incident
    names that capacity, not a provider outage, while a provider terminal keeps the outage phrase."""
    from ouroboros.task_finalization import send_provider_death_notice, terminal_result_fields
    from tests.test_delivery_forced_finalization import _forced_test_context

    slot = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_provider_code": "model_capacity_unavailable",
            "_last_llm_attempt_custody": "released"}  # what loop_llm_call stamps from the limiter's refusal
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path, usage=dict(slot))
    _text, usage, _trace = loop._handle_provider_unavailable(
        limit_ctx, error_kind="transport_unavailable", wait_cause="transport_unavailable", waited_sec=60.0)
    assert (usage["reason_code"], usage["terminal_provider_cause"]) == ("provider_unavailable", "local_capacity")
    sent = []
    ctx = SimpleNamespace(send_with_budget=lambda chat_id, text, **kw: sent.append((text, kw)))
    assert send_provider_death_notice(ctx, 7, "t-slot", {**terminal_result_fields(usage), "terminal_origin": "model_final"})
    assert send_provider_death_notice(ctx, 7, "t-down", {"terminal_origin": "model_final", "terminal_provider_notice": "529"})
    (local, local_kw), (outage, outage_kw) = sent
    assert "outage" not in local and "local model slot" in local
    assert local_kw["progress_meta"]["notice"]["code"] == "local_model_capacity"
    assert outage.startswith("A model-provider outage stopped this task") and outage_kw["progress_meta"]["notice"]["code"] == "provider_outage"


def test_a_refused_evolution_start_says_a_phrase_with_its_lock_reason_behind_details(monkeypatch):
    """RV4 12: the state lock's reason is the notice's detail, never the words of the command's or switch's reply."""
    from supervisor import events_runtime_controls, evolution_lifecycle, state

    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")

    def refused(*_a, **_k):
        raise state.StateUnavailable("lock_timeout")

    monkeypatch.setattr(state, "update_state", refused)
    refusal = events_runtime_controls.owner_evolution_start("tidy", source="owner_chat")
    assert "lock_timeout" not in refusal["text"] and refusal["notice"]["details"] == "lock_timeout"


def test_an_agent_evolution_start_that_hits_the_state_lock_says_the_owners_phrase(monkeypatch):
    """RVD4 12: the agent's own start whose activation the state lock refused sends the phrase the owner's start sends,
    with the lock's reason behind Details, never "ON" with the raw reason."""
    from supervisor import events_runtime_controls, evolution_lifecycle, state

    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(evolution_lifecycle, "evolution_stop_reason", lambda *_a: "")
    monkeypatch.setattr(evolution_lifecycle, "start_evolution_campaign", lambda *_a, **_k: {"id": "c1"})

    def refused(*_a, **_k):
        raise state.StateUnavailable("lock_timeout")

    monkeypatch.setattr(state, "update_state", refused)
    sent = []
    ctx = SimpleNamespace(send_with_budget=lambda chat, text, **kw: sent.append((text, kw)),
                          load_state=lambda: {"owner_chat_id": 7, "evolution_owner_stopped": False})
    events_runtime_controls._handle_toggle_evolution({"enabled": True, "objective": "tidy", "source": "agent_tool"}, ctx)
    [(text, kw)] = sent
    assert text == "Evolution did not turn on: the runtime state could not be saved." and "lock_timeout" not in text
    assert kw["progress_meta"]["notice"] == {"code": "evolution_start_unsaved", "details": "lock_timeout"}


def test_a_split_custody_row_never_inherits_the_answers_notice():
    """RV4 18: a custody row says its own fact; the answer's step and raw details stay the answer's."""
    from supervisor.events_chat_delivery import bundled_rows

    answer, rows = bundled_rows({"type": "send_message", "chat_id": 7, "task_id": "t-c", "delivery_id": "final:task:t-c",
                                 "text": "The total budget is spent.", "terminal_custody_notice": "A delegated run may be live.",
                                 "progress_meta": host_notice.fields("budget_exhausted", step="raise_total_budget")})
    assert answer["progress_meta"]["notice"]["code"] == "budget_exhausted"
    assert rows and "notice" not in rows[0]["progress_meta"]


def test_a_version_switch_chooses_its_words_by_its_typed_cause_and_its_step_by_who_pressed(tmp_path, monkeypatch):
    """RV4 20 and 21: free text that happens to equal a known sentence is Git's output, kept whole in details; an
    automatic rollback has no "press it again"."""
    from ouroboros import store_switch
    from supervisor import message_bus as bus

    sent = []
    monkeypatch.setattr(bus, "send_with_budget", lambda chat, text, **kw: sent.append((text, kw)))
    progress = {"outcomes": [
        {"intent": {"door": "boot_smoke_rollback", "target_ref": "v7.6.0"}, "message": "the export failed twice and was abandoned"},
        {"intent": {"door": "ui_rollback", "target_ref": "v7.6.0"}, "reason": "second_failure"}]}
    (tmp_path / store_switch.PROGRESS_REL).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / store_switch.PROGRESS_REL).write_text(json.dumps(progress), encoding="utf-8")
    store_switch.report_outcomes(tmp_path, 7)
    (free, free_kw), (typed, typed_kw) = sent
    assert free == "Automatic rollback did not happen. The app is running the current version."
    assert "step" not in free_kw["progress_meta"]["notice"]
    assert free_kw["progress_meta"]["notice"]["details"].endswith(": the export failed twice and was abandoned")
    assert typed == "Rollback did not happen: the export failed twice and was abandoned. The app is running the current version."
    assert typed_kw["progress_meta"]["notice"]["step"] == "press_again"


def test_a_switch_failure_names_its_own_record_which_the_next_switch_never_overwrites(tmp_path, monkeypatch):
    """RVD4 21: Git's multiline output is cut to its headline and the address of this failure's own outcome record, never
    the shared receipt the next switch rewrites; after the next switch fails too, that record still holds the whole text."""
    from ouroboros import store_switch
    from supervisor import message_bus as bus

    sent, stamps = [], iter(["2026-11-22T10:00:00.000001+00:00", "reported", "2026-11-22T11:00:00.000002+00:00"])
    monkeypatch.setattr(bus, "send_with_budget", lambda chat, text, **kw: sent.append(kw))
    monkeypatch.setattr(store_switch, "utc_now_iso", lambda: next(stamps))
    intent = {"door": "ui_rollback", "target_ref": "v7.6.0", "target_sha": "a" * 40, "requested_at": "first"}
    first = "error: Your local changes would be overwritten by checkout:\n\tweb/app.js\nAborting"
    store_switch._progress(tmp_path, "switch_failed", intent=intent, message=first)
    store_switch.report_outcomes(tmp_path, 7)
    details = sent[0]["progress_meta"]["notice"]["details"]
    assert details.startswith("Rollback to v7.6.0: error: Your local changes would be overwritten by checkout:")
    assert f"{store_switch.PROGRESS_REL}, outcome at 2026-11-22T10:00:00.000001+00:00" in details
    assert str(store_switch.RECEIPT_REL) not in details
    (tmp_path / store_switch.RECEIPT_REL).write_text(json.dumps({"intent": {**intent, "requested_at": "next"}}), encoding="utf-8")
    store_switch._progress(tmp_path, "switch_failed", intent={**intent, "requested_at": "next"}, message="fatal: other\nmore")
    outcomes = json.loads((tmp_path / store_switch.PROGRESS_REL).read_text(encoding="utf-8"))["outcomes"]
    assert [row["message"] for row in outcomes if f"outcome at {row['at']}" in details] == [first]


def test_an_agent_rollback_is_worded_from_its_deferral_and_restart_facts(monkeypatch):
    """RV4 20: a deferred switch only recorded its intent; a restart that could not be requested says so, with Restart."""
    from ouroboros.gateway import control
    from supervisor import events_runtime_controls

    sent = []
    ctx = SimpleNamespace(request_restart=lambda: None, load_state=lambda: {"owner_chat_id": 7},
                          send_with_budget=lambda chat, text, **kw: sent.append((text, kw)))
    monkeypatch.setattr(events_runtime_controls.threading, "Thread", lambda target, **_k: SimpleNamespace(start=target))
    monkeypatch.setattr(control, "rollback_through_door", lambda *_a, **_k: (200, {
        "status": "restart_required", "message": "recorded", "restarting": False, "switch_deferred": True}))
    events_runtime_controls._handle_rollback_request({"target": "abc123def4567890"}, ctx)
    [(text, kw)] = sent
    assert text.startswith("Ouroboros recorded a rollback at its own request") and "could not be requested" in text
    assert kw["progress_meta"]["notice"]["step"] == "restart"


def test_the_duplicate_job_and_a_clipped_progress_row_keep_their_evidence(monkeypatch):
    """RV4 22: the original job's id is the pointer notice's detail; a summary cut at its cap names its whole row."""
    from ouroboros import skill_lifecycle_queue as queue
    from ouroboros.memory import Memory
    from supervisor import message_bus as bus

    sent = []
    monkeypatch.setattr(bus, "send_with_budget", lambda chat, text, **kw: sent.append((chat, text, kw)))
    job = lambda chat, jid: SimpleNamespace(kind="install", target="alpha", status="running", id=jid, chat_id=chat,
                                            group_id="", task_id="", root_task_id="", origin_task_id="",
                                            origin_root_task_id="", presentation_owner_task_id="", source="")
    monkeypatch.setattr(queue, "_effective_chat_id", lambda j: j.chat_id)
    queue._notify_duplicate_pointer(job(2, "job-new"), job(1, "job-original"))
    [(_chat, text, kw)] = sent
    assert "job-original" not in text and kw["progress_meta"]["notice"]["details"] == "job job-original"
    long = {"text": "x" * 790, "ts": "2026-11-22T10:00:00Z", "history_id": "progress:42",
            **host_notice.fields("skill_failed", details="ENOENT: no such file")}
    assert Memory.summarize_progress(SimpleNamespace(), [long]).endswith("[whole row: progress:42]")


def test_the_offer_memory_is_bounded(monkeypatch):
    """RV4 23: a process remembers a bounded number of offered phrases per language."""
    host_notice._MISSES.clear()
    monkeypatch.setattr(host_notice, "_MISSES_CAP", 3)
    import ouroboros.i18n_memory as memory
    monkeypatch.setattr(memory, "record_missing", lambda *_a, **_k: {"accepted": 1})
    import ouroboros.ui_translation as ui_translation
    monkeypatch.setattr(ui_translation, "on_language_event", lambda *_a, **_k: None)
    for index in range(7):
        host_notice._miss(f"phrase {index}", "ru", {}, pathlib.Path("/tmp"))
    assert len(host_notice._MISSES["ru"]) <= 3


def test_manual_actions_are_typed_steps_and_an_objective_is_never_the_phrase(monkeypatch, tmp_path):
    """RV4 11: a supported manual action is a step from the closed table; the evolution objective is its own detail."""
    from ouroboros import review_overrides, skill_review_runner
    from supervisor import evolution_lifecycle, message_bus as bus, state

    sent = []
    monkeypatch.setattr(bus, "send_with_budget", lambda chat, text, **kw: sent.append((text, kw)))
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr(review_overrides, "landed_outcome", lambda *_a: False)
    campaign = {"objective": "Cut the cold start\nby half"}
    evolution_lifecycle.notify_owner_cycle_outcome(campaign, {"cycle_outcome": "absorbed", "commit_sha": "c" * 40})
    evolution_lifecycle.notify_owner_cycle_outcome(campaign, {"cycle_outcome": "abandoned", "abandoned_reason": "red tests"})
    (absorbed, kept), (abandoned, gone) = sent
    assert "Cut the cold start" not in absorbed + abandoned and "\n" not in absorbed + abandoned
    assert "Reply if" not in absorbed and kept["progress_meta"]["notice"]["step"] == "reply_to_revert"
    assert all("objective: Cut the cold start" in kw["progress_meta"]["notice"]["details"] for kw in (kept, gone))
    rows = []
    monkeypatch.setattr(skill_review_runner.dialogue_store, "append_quietly", lambda root, stream, row: rows.append(row))
    skill_review_runner._append_interrupted_review_progress(tmp_path, "alpha", {"job_id": "j1"}, ts="2026-11-22T10:00:00Z")
    [row] = rows
    assert row["notice"]["step"] == "review_again" and row["notice"]["values"] == {"skill": "alpha"}
