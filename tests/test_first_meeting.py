"""My human's press of Start opens one first meeting (``supervisor.first_meeting``): admitted once, in Main whatever
owner chat is bound, as the host's prompt for the press; never twice across starts, never without a record, and
not at all when nothing is owed or a message of mine to my human already exists."""

from __future__ import annotations

import json

import pytest

from supervisor import first_meeting as meeting

PRESSED = "2026-11-21T10:47:12.000000+00:00"


@pytest.fixture
def boot(monkeypatch, tmp_path):
    """One supervisor start over an in-memory state, a recorded queue and a stubbed dialogue read."""
    from ouroboros import dialogue_store
    from supervisor import queue, state

    store = {"owner_chat_id": None, "owner_id": None, meeting.FIRST_MEETING_KEY: meeting.owed(PRESSED)}
    admitted, answers, exchange = [], [], {"value": (False, False)}
    modes = {"chosen": "advanced", "process": "advanced"}  # the access level Finish saved, and the one this process runs
    monkeypatch.setattr(state, "load_state", lambda: json.loads(json.dumps(store)))
    monkeypatch.setattr("ouroboros.config.load_settings", lambda *a, **k: {"OUROBOROS_RUNTIME_MODE": modes["chosen"]})
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: modes["process"])

    def update_state(mutator, **_kwargs):
        mutator(store)
        return dict(store)

    def enqueue(task):
        admitted.append(task)
        return answers.pop(0) if answers else dict(task)

    def owner_exchange(_root):
        if isinstance(exchange["value"], Exception):
            raise exchange["value"]
        return exchange["value"]

    monkeypatch.setattr(state, "update_state", update_state)
    monkeypatch.setattr(queue, "enqueue_with_admission_receipt", enqueue)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr(dialogue_store, "owner_exchange", owner_exchange)
    (tmp_path / "logs").mkdir()

    def rows():
        path = tmp_path / "logs" / "events.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    yield type("Boot", (), {"store": store, "admitted": admitted, "answers": answers, "exchange": exchange, "modes": modes,
                            "root": tmp_path, "rows": staticmethod(rows),
                            "start": staticmethod(lambda: meeting.admit_first_meeting(tmp_path))})


def test_a_fresh_setup_opens_one_meeting_in_main_and_a_second_start_none(boot):
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID

    assert boot.start() == "admitted"
    [root] = boot.admitted
    assert (root["chat_id"], root["delegation_role"], root["root_task_id"]) == (WEB_UI_CHAT_ID, "root", root["id"])
    assert root["metadata"]["objective_author"] == {"kind": "ui_control", "control": "setup_start"}
    assert root["metadata"]["ui_control"] == {"name": "setup_start", "pressed_at": PRESSED}
    assert root["text"] == meeting.objective(PRESSED) and PRESSED in root["text"]
    assert root["_require_unique_task_id"] is True and root["_require_worker_pool"] is True
    record = boot.store[meeting.FIRST_MEETING_KEY]
    assert (record["state"], record["task_id"], record["chat_id"]) == ("admitted", root["id"], WEB_UI_CHAT_ID)
    assert boot.start() == "none" and len(boot.admitted) == 1  # the next start, and every later one: nothing
    assert [row["outcome"] for row in boot.rows()] == ["admitted"]


def test_the_press_s_root_goes_to_main_whatever_owner_chat_is_bound_or_unread(boot):
    """Owner item 1: the press was made in setup, so its work is in Main, never an external chat bound in between,
    and a binding that cannot be read does not hold back work whose room the owner already chose."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID

    boot.store["owner_chat_id"] = 4242  # my human's first message arrived before this start (a Telegram chat)
    assert boot.start() == "admitted" and boot.admitted[-1]["chat_id"] == WEB_UI_CHAT_ID != 4242
    boot.store.update({meeting.FIRST_MEETING_KEY: meeting.owed(PRESSED), "_recovery": {"unconfirmed": ["owner_chat_id"]}})
    assert boot.start() == "admitted" and boot.admitted[-1]["chat_id"] == WEB_UI_CHAT_ID and len(boot.admitted) == 2
    assert boot.store[meeting.FIRST_MEETING_KEY]["chat_id"] == WEB_UI_CHAT_ID


def test_a_restart_into_the_chosen_access_level_comes_first_and_the_meeting_waits_for_it(boot):
    """Setup chose an access level this process does not run (it applies from the next start: ``restart_required``):
    nothing is admitted here, the meeting stays owed, and the restarted process running that level admits it, so it
    never runs under a level my human did not choose and a desktop restart never consumes it."""
    boot.modes.update(chosen="light", process="advanced")
    assert boot.start() == "deferred" and boot.admitted == []
    assert boot.store[meeting.FIRST_MEETING_KEY] == meeting.owed(PRESSED)
    assert boot.rows()[-1]["reason"] == "a restart into the chosen access level (light) comes first"
    boot.modes["process"] = "light"  # the restarted process
    assert boot.start() == "admitted" and len(boot.admitted) == 1


def test_nothing_is_owed_without_a_press_and_an_existing_message_of_mine_settles_it(boot):
    from ouroboros.store import StoreBusy

    boot.store.pop(meeting.FIRST_MEETING_KEY)  # a state minted without a recorded setup (Colab, headless, older)
    assert boot.start() == "none" and boot.admitted == [] and boot.rows() == []
    boot.store[meeting.FIRST_MEETING_KEY] = meeting.owed(PRESSED)
    boot.exchange["value"] = StoreBusy("dialogue store busy")
    assert boot.start() == "deferred" and boot.store[meeting.FIRST_MEETING_KEY]["state"] == "owed"
    boot.exchange["value"] = (True, True)  # we have already talked
    assert boot.start() == "not_needed" and boot.admitted == []
    assert boot.store[meeting.FIRST_MEETING_KEY]["state"] == "not_needed"
    assert boot.start() == "none"


def test_a_state_that_cannot_be_read_defers_the_meeting_and_never_fails_the_start(boot, monkeypatch):
    """«A start never fails on this» covers the state read too: unreadable, the meeting waits for the next start with a
    row saying why; readable again, it is admitted (rv2 C8)."""
    from supervisor import state

    readable = state.load_state
    monkeypatch.setattr(state, "load_state", lambda: (_ for _ in ()).throw(state.StateUnavailable("no_readable_copy")))
    assert boot.start() == "deferred" and boot.admitted == []
    assert boot.rows()[-1]["outcome"] == "deferred" and "StateUnavailable" in boot.rows()[-1]["reason"]
    monkeypatch.setattr(state, "load_state", readable)
    assert boot.start() == "admitted" and len(boot.admitted) == 1


def test_the_claim_precedes_the_admission_so_a_crash_retries_the_same_id_and_never_admits_twice(boot):
    boot.answers.append({"_admission_blocked": "worker_pool_unavailable", "_worker_pool_disabled_reason": "budget"})
    assert boot.start() == "refused"
    first = boot.admitted[-1]["id"]
    record = boot.store[meeting.FIRST_MEETING_KEY]
    assert (record["state"], record["task_id"], record["last_refusal"]) == ("admitting", first, "budget")
    # A crash after the claim: the next start repeats the very same id, which the queue admits at most once.
    boot.answers.append({"_admission_blocked": "duplicate_task_id"})  # an earlier start's admission had landed
    assert boot.start() == "admitted" and boot.admitted[-1]["id"] == first
    assert boot.store[meeting.FIRST_MEETING_KEY]["state"] == "admitted"
    assert [row["outcome"] for row in boot.rows()] == ["refused", "admitted"]
