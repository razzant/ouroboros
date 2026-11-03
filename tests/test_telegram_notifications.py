import asyncio, importlib.util, json, sys, types
from pathlib import Path

import pytest
from ouroboros import dialogue_store as ds


def _load():
    """Load plugin (which imports the lib modules) and return the notifier module
    where the budget/task notification helpers now live (post lib split)."""
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    pkg = types.ModuleType("tg_nt"); pkg.__path__ = [str(root)]; sys.modules["tg_nt"] = pkg
    spec = importlib.util.spec_from_file_location("tg_nt.plugin", root / "plugin.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return sys.modules["tg_nt.lib.telegram_notifier"]


class _Rec:
    sent = []
    def __init__(self, token, **_kwargs): pass
    async def send_message(self, chat_id, text, parse_mode="HTML"):
        _Rec.sent.append((chat_id, text)); return 1


def _api(tmp_path):
    data = tmp_path / "data"
    sd = data / "state" / "skills" / "telegram-bridge"; sd.mkdir(parents=True)
    (data / "logs").mkdir(parents=True, exist_ok=True)

    class A:
        def get_state_dir(self): return str(sd)
        def get_settings(self, k): return {}  # _Rec ignores the token value
        def log(self, *a, **k): pass
    return A(), data


def test_budget_threshold_notify(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    snapshots = iter([
        {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 85},
        {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 85},
        {"spent_usd": 920, "budget_limit": 1000, "budget_pct": 92},
    ])

    async def runtime_state(_api):
        return next(snapshots)

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    # This compatibility projection is deliberately stale. Notifications must
    # follow the authoritative /api/state snapshot above instead.
    (data / "state" / "state.json").write_text(json.dumps({"spent_usd": 9999}), encoding="utf-8")
    (data / "settings.json").write_text(json.dumps({"TOTAL_BUDGET": 1}), encoding="utf-8")
    settings = {"TELEGRAM_NOTIFY_BUDGET": "on"}; state = {}
    asyncio.run(nt._check_budget_notify(api, settings, 42, state, "en"))
    assert len(_Rec.sent) == 1 and "85%" in _Rec.sent[0][1] and state["budget_threshold"] == 80
    asyncio.run(nt._check_budget_notify(api, settings, 42, state, "en"))   # same → no new
    assert len(_Rec.sent) == 1
    asyncio.run(nt._check_budget_notify(api, settings, 42, state, "en"))
    assert len(_Rec.sent) == 2 and "92%" in _Rec.sent[1][1] and state["budget_threshold"] == 90


def test_budget_notification_uses_authoritative_percentage(tmp_path, monkeypatch):
    nt = _load(); api, _data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)

    async def runtime_state(_api):
        return {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 79}

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    asyncio.run(nt._check_budget_notify(api, {"TELEGRAM_NOTIFY_BUDGET": "on"}, 42, {}, "en"))
    assert _Rec.sent == []


def test_budget_notification_is_silent_without_bounded_accounting(tmp_path, monkeypatch):
    nt = _load(); api, _data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    snapshots = iter([
        {},
        {"spent_usd": None, "budget_limit": 1000, "budget_pct": None},
        {"spent_usd": "unavailable", "budget_limit": 1000, "budget_pct": "unavailable"},
        {"spent_usd": 850, "budget_limit": 0, "budget_pct": 0},
    ])

    async def runtime_state(_api):
        return next(snapshots)

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    settings = {"TELEGRAM_NOTIFY_BUDGET": "on"}
    state = {}
    for _ in range(4):
        asyncio.run(nt._check_budget_notify(api, settings, 42, state, "en"))
    assert _Rec.sent == []
    assert "budget_threshold" not in state


# A root finish in Main: the audience of opt-in completion notifications.
_OWNER_ROOT = {"summary_kind": "terminal_root_projection", "chat_id": 1}


def _root(tid, phase, **fields):
    return {"type": "task_summary", "task_id": tid, "outcome_final": True, **_OWNER_ROOT, "outcome_phase": phase,
            "outcome_axes": {"lifecycle": {"status": "completed"}}, **fields}


def _write_rows(data, rows):
    for row in rows:
        ds.append(data, "chat", row)


def test_effectively_quiet_runs_skip_push_but_notice_bearing_runs_notify(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    _write_rows(data, [
        _root("quiet", "done", wake_silent=True, quiet_effective=True),
        _root("notice", "warn", wake_silent=True, quiet_effective=False,
              completion_notice={"neutral": False, "detail": "A tool failed.", "phase": "warn"}),
        _root("ordinary", "done", wake_silent=False, quiet_effective=False),
    ])
    state = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    assert len(_Rec.sent) == 2
    assert "A tool failed." in _Rec.sent[0][1] and "ordinary" in _Rec.sent[1][1]
    assert state["notified_task_ids"] == ["quiet", "notice", "ordinary"]


def test_off_suppresses_every_phase_and_consumes_every_id(tmp_path, monkeypatch):
    """OFF includes failures/cancellations; seen IDs prevent a backlog on enable."""
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    _write_rows(data, [
        _root("done1", "done"), _root("warn1", "warn"), _root("err1", "error", chat_id=1000000),
        _root("stop1", "cancelled"), _root("child1", "warn", summary_kind="terminal_result_projection"),
        _root("hidden1", "warn", chat_id=0), _root("a2a1", "warn", chat_id=-1001),
        {k: v for k, v in _root("nochat1", "warn").items() if k != "chat_id"},
        {k: v for k, v in _root("nokind1", "warn").items() if k != "summary_kind"},
    ])
    state = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {}, 42, state, "en"))
    assert _Rec.sent == []
    assert state["notified_task_ids"] == [
        "done1", "warn1", "err1", "stop1", "child1", "hidden1", "a2a1", "nochat1", "nokind1"]
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    assert _Rec.sent == []


def test_enabled_notices_never_widen_the_audience(tmp_path, monkeypatch):
    """With task notifications on, a ROOT finish in an owner room is announced; a
    child, the hidden partition, an A2A room and a chat-less or kind-less row still send
    nothing (the toggle once pushed every finish, children included), and all are seen."""
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    _write_rows(data, [
        _root("done1", "done"), _root("warn1", "warn", chat_id=1000000),
        _root("child1", "done", summary_kind="terminal_result_projection"),
        _root("hidden1", "done", chat_id=0), _root("a2a1", "done", chat_id=-1001),
        {k: v for k, v in _root("nochat1", "done").items() if k != "chat_id"},
        {k: v for k, v in _root("nokind1", "done").items() if k != "summary_kind"},
    ])
    state = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    assert [text for _chat, text in _Rec.sent] == ["✅ Task done1 done", "⚠️ Task warn1 done with warnings"]
    assert state["notified_task_ids"] == ["done1", "warn1", "child1", "hidden1", "a2a1", "nochat1", "nokind1"]


def test_enabled_notices_are_one_producer_and_never_double(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    _write_rows(data, [_root("warn1", "warn")])
    state = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    assert [text for _chat, text in _Rec.sent] == ["⚠️ Task warn1 done with warnings"]


def test_task_line_carries_the_card_reason_sentence(tmp_path, monkeypatch):
    """The line carries the card's own clause, read off the row: a reworded card
    sentence updates the card and the phone together, or fails here."""
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES

    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    _write_rows(data, [
        _root("adv1", "warn", reason_detail=TASK_CAUSE_PHRASES["plan_review_advisory"]),
        _root("plain1", "warn"),
    ])
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
    assert [text for _chat, text in _Rec.sent] == [
        f"⚠️ Task adv1 done with warnings\n{TASK_CAUSE_PHRASES['plan_review_advisory']}",
        "⚠️ Task plain1 done with warnings",
    ]


def test_phase_words_are_derived_from_the_host_headline_table():
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE

    nt = _load()
    assert nt._PHASE_WORDS == {
        phase: word.lower() for phase, word in OUTCOME_PHASE_HEADLINE.items() if phase != "working"}
    assert "working" not in nt._PHASE_WORDS
    assert nt._PHASE_WORDS["warn"] == "done with warnings"
    # Another language reads the SAME headline table by code through the install's memory.
    assert nt._phase_word("warn", "") == "done with warnings"


def test_enabling_task_notifications_later_never_blasts_the_backlog(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    _write_rows(data, [_root("ok1", "done"), _root("ok2", "done"), _root("ok3", "done")])
    state = {}
    asyncio.run(nt._check_tasks_notify(api, {}, 42, state, "en"))  # seeds
    asyncio.run(nt._check_tasks_notify(api, {}, 42, state, "en"))  # toggle off: seen, unsent
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    assert _Rec.sent == [] and set(state["notified_task_ids"]) == {"ok1", "ok2", "ok3"}
    _write_rows(data, [_root("ok4", "done")])
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    assert [text for _chat, text in _Rec.sent] == ["✅ Task ok4 done"]


def test_notifier_never_edits_the_silent_mode_tracked_message():
    """Structurally, not by fixture: this lane reaches Telegram only through its own
    sendMessage, so it cannot replace the answer bubble the mirror tracks."""
    nt = _load()
    source = Path(nt.__file__).read_text(encoding="utf-8")
    for name in ("edit_message_text", "_set_silent_msg", "_get_silent_msg", "silent_msg", "TELEGRAM_SILENT_MODE"):
        assert name not in source, name
    assert "client.send_message(" in source  # its own sendMessage is the only transport


def test_the_notifier_loop_runs_the_tasks_lane_with_both_toggles_off(tmp_path, monkeypatch):
    """The disabled lane still observes IDs, but must not send anything."""
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    monkeypatch.setattr(nt, "_load_settings", lambda _api: {"TELEGRAM_CHAT_ID": "42"})
    _write_rows(data, [_root("warn1", "warn")])
    nt._save_notif_state(api, {"notified_task_ids": []})
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        raise RuntimeError("stop the loop")

    monkeypatch.setattr(nt.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError):
        asyncio.run(nt._make_notifier(api)())
    assert _Rec.sent == []
    assert nt._load_notif_state(api)["notified_task_ids"] == ["warn1"]
    assert sleeps == [30]


def test_tasks_notify_primes_then_fires(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    chat = data / "logs" / "chat.jsonl"
    chat.write_text(json.dumps({"type": "task_summary", **_OWNER_ROOT, "task_id": "old1", "rounds": 3,
                                "outcome_axes": {"lifecycle": "completed"}}) + "\n", encoding="utf-8")
    settings = {"TELEGRAM_NOTIFY_TASKS": "on"}; state = {}
    asyncio.run(nt._check_tasks_notify(api, settings, 42, state, "en"))   # primes, no send
    assert _Rec.sent == [] and "old1" in state["notified_task_ids"]
    ds.append(data, "chat", {"type": "task_summary", **_OWNER_ROOT, "task_id": "new1", "rounds": 5,
                             "outcome_axes": {"lifecycle": "completed"}})
    asyncio.run(nt._check_tasks_notify(api, settings, 42, state, "en"))
    assert len(_Rec.sent) == 1 and "new1" in _Rec.sent[0][1] and "5r" in _Rec.sent[0][1]


def test_task_summary_scan_uses_bounded_live_tail(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path)
    path = data / "logs" / "chat.jsonl"
    seen = {}

    def tail(actual_path, *, max_entries, tail_bytes):
        seen.update(path=actual_path, max_entries=max_entries, tail_bytes=tail_bytes)
        return ([{"type": "task_summary", "task_id": "bounded"}], True)

    monkeypatch.setattr(nt, "_jsonl_tail", tail)
    rows = nt._summary_ids_in_tail(api, limit=123)

    assert rows == [("bounded", {"type": "task_summary", "task_id": "bounded"})]
    assert seen == {"path": path, "max_entries": 123, "tail_bytes": 256 * 1024}


@pytest.mark.parametrize("problem", ["pending", "mismatch", "invalid"])
def test_unknown_dialogue_coverage_never_advances_notified_ids(tmp_path, problem):
    from ouroboros import dialogue_import as di
    from tests._dialogue import seed

    nt = _load()
    api, data = _api(tmp_path)
    row = _root("older", "done")
    if problem == "invalid":
        seed(data, live={"chat": [row, "{invalid\n"]})
    else:
        seed(data, archives={"chat_20261016T000000.jsonl": [row]})
    ds.ensure(data)
    if problem == "mismatch":
        (data / "archive/chat_20261016T000000.jsonl").write_text("changed\n", encoding="utf-8")
        di.run_dialogue_import(data)
    state = {}
    error, delivered = asyncio.run(nt._check_tasks_notify(api, {}, 42, state, "en"))
    assert isinstance(error, ds.StoreError) and not delivered and state == {}
    assert {"pending": "still importing", "mismatch": "could not be imported",
            "invalid": "invalid intervals"}[problem] in str(error)
    if problem == "pending":
        di.run_dialogue_import(data)
        assert asyncio.run(nt._check_tasks_notify(api, {}, 42, state, "en")) == (None, False)
        assert state == {"notified_task_ids": ["older"]}


def test_tasks_notify_waits_for_latest_final_summary_for_each_task(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    chat = data / "logs" / "chat.jsonl"
    working = {
        "type": "task_summary", **_OWNER_ROOT, "task_id": "same1", "outcome_final": False,
        "outcome_phase": "working",
        "outcome_axes": {"lifecycle": {"status": "completed"},
                         "execution": {"status": "ok"}},
    }
    chat.write_text(json.dumps(working) + "\n", encoding="utf-8")
    state = {"notified_task_ids": []}

    asyncio.run(nt._check_tasks_notify(
        api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en",
    ))
    assert _Rec.sent == []
    assert state["notified_task_ids"] == []

    terminal = {
        **working, "outcome_final": True, "outcome_phase": "warn",
        "outcome_axes": {"lifecycle": {"status": "completed"},
                         "execution": {"status": "ok"},
                         "review": {"status": "degraded"}},
    }
    ds.append(data, "chat", terminal)

    asyncio.run(nt._check_tasks_notify(
        api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en",
    ))
    assert _Rec.sent == [(42, "⚠️ Task same1 done with warnings")]
    assert state["notified_task_ids"] == ["same1"]


def test_notify_disabled_is_silent(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    (data / "state" / "state.json").write_text(json.dumps({"spent_usd": 999}), encoding="utf-8")
    (data / "settings.json").write_text(json.dumps({"TOTAL_BUDGET": 1000}), encoding="utf-8")
    asyncio.run(nt._check_budget_notify(api, {"TELEGRAM_NOTIFY_BUDGET": "off"}, 42, {}, "en"))
    assert _Rec.sent == []


def test_budget_notification_retries_before_advancing_ledger(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path)

    class Flaky:
        attempts = 0
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            Flaky.attempts += 1
            if Flaky.attempts == 1:
                raise nt.TelegramTransportError("offline")
            return 1

    monkeypatch.setattr(nt, "TelegramClient", Flaky)

    async def runtime_state(_api):
        return {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 85}

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    settings = {"TELEGRAM_NOTIFY_BUDGET": "on"}; state = {}
    asyncio.run(nt._check_budget_notify(api, settings, 42, state, "en"))
    assert "budget_threshold" not in state
    asyncio.run(nt._check_budget_notify(api, settings, 42, state, "en"))
    assert Flaky.attempts == 2 and state["budget_threshold"] == 80


def test_task_notification_retries_before_advancing_ledger(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path)

    class Flaky:
        attempts = 0
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            Flaky.attempts += 1
            if Flaky.attempts == 1:
                raise nt.TelegramTransportError("offline")
            return 1

    monkeypatch.setattr(nt, "TelegramClient", Flaky)
    (data / "logs" / "chat.jsonl").write_text(
        json.dumps({"type": "task_summary", **_OWNER_ROOT, "task_id": "retry1",
                    "outcome_axes": {"lifecycle": "completed"}}) + "\n",
        encoding="utf-8",
    )
    settings = {"TELEGRAM_NOTIFY_TASKS": "on"}; state = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, settings, 42, state, "en"))
    assert state["notified_task_ids"] == []
    asyncio.run(nt._check_tasks_notify(api, settings, 42, state, "en"))
    assert Flaky.attempts == 2 and state["notified_task_ids"] == ["retry1"]


def test_tasks_notify_stops_batch_on_first_transient_failure(tmp_path, monkeypatch):
    """A transient send failure ends the batch: with three pending summaries
    exactly ONE send is attempted in the cycle (each further send would burn
    its full transport timeout against the same dead network) and no summary
    is marked seen, so all three retry next cycle."""
    nt = _load(); api, data = _api(tmp_path)

    class AlwaysOffline:
        attempts = 0
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            AlwaysOffline.attempts += 1
            raise nt.TelegramTransportError("offline")

    monkeypatch.setattr(nt, "TelegramClient", AlwaysOffline)
    with open(data / "logs" / "chat.jsonl", "w", encoding="utf-8") as f:
        for tid in ("t1", "t2", "t3"):
            f.write(json.dumps({"type": "task_summary", **_OWNER_ROOT, "task_id": tid,
                                "outcome_axes": {"lifecycle": "completed"}}) + "\n")
    state = {"notified_task_ids": []}
    transient, delivered = asyncio.run(
        nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en")
    )

    assert AlwaysOffline.attempts == 1
    assert transient is not None and delivered is False
    assert state["notified_task_ids"] == []


def test_notifier_cycle_makes_one_send_attempt_when_transport_is_down(tmp_path, monkeypatch):
    """One dead-network cycle = exactly one send attempt, then backoff: a
    transient budget-send failure skips the tasks batch entirely."""
    nt = _load(); api, data = _api(tmp_path)

    class AlwaysOffline:
        attempts = 0
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            AlwaysOffline.attempts += 1
            raise nt.TelegramTransportError("offline")

    monkeypatch.setattr(nt, "TelegramClient", AlwaysOffline)

    async def runtime_state(_api):
        return {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 85}

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    with open(data / "logs" / "chat.jsonl", "w", encoding="utf-8") as f:
        for tid in ("t1", "t2", "t3"):
            f.write(json.dumps({"type": "task_summary", **_OWNER_ROOT, "task_id": tid,
                                "outcome_axes": {"lifecycle": "completed"}}) + "\n")
    tasks_checks = []
    real_tasks_check = nt._check_tasks_notify

    async def counting_tasks_check(*args, **kwargs):
        tasks_checks.append(True)
        return await real_tasks_check(*args, **kwargs)

    monkeypatch.setattr(nt, "_check_tasks_notify", counting_tasks_check)
    monkeypatch.setattr(
        nt,
        "_load_settings",
        lambda _api: {
            "TELEGRAM_NOTIFY_BUDGET": "on",
            "TELEGRAM_NOTIFY_TASKS": "on",
            "TELEGRAM_CHAT_ID": "42",
        },
    )
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(nt.asyncio, "sleep", record_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(nt._make_notifier(api)())

    assert AlwaysOffline.attempts == 1  # budget send only; tasks batch skipped
    assert tasks_checks == []
    assert sleeps == [5]  # backoff, not the healthy 30s pacing sleep


def test_notifier_ignores_ambient_chat_id(monkeypatch):
    nt = _load()
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert nt._pinned_chat_id({}) == 0


def test_permanent_notification_rejection_skips_and_persists(tmp_path, monkeypatch):
    """A permanent send rejection consumes the notification (state advances)
    instead of raising into the supervised-restart budget and replaying the
    same dead send forever."""
    nt = _load(); api, data = _api(tmp_path)

    class Rejected:
        attempts = 0
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            Rejected.attempts += 1
            raise nt.TelegramRequestRejected("rejected", status_code=401)

    monkeypatch.setattr(nt, "TelegramClient", Rejected)

    async def runtime_state(_api):
        return {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 85}

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    settings = {"TELEGRAM_NOTIFY_BUDGET": "on"}; state = {}
    asyncio.run(nt._check_budget_notify(api, settings, 42, state, "en"))
    assert state["budget_threshold"] == 80
    asyncio.run(nt._check_budget_notify(api, settings, 42, state, "en"))
    assert Rejected.attempts == 1  # consumed: the dead send is not replayed

    (data / "logs" / "chat.jsonl").write_text(
        json.dumps({"type": "task_summary", "task_id": "dead1", "outcome_axes": {"lifecycle": "completed"}}) + "\n",
        encoding="utf-8",
    )
    task_state = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, task_state, "en"))
    assert task_state["notified_task_ids"] == ["dead1"]


def test_notifier_loop_survives_permanent_rejection_and_saves_state(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path)

    class Rejected:
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            raise nt.TelegramRequestRejected("rejected", status_code=401)

    monkeypatch.setattr(nt, "TelegramClient", Rejected)

    async def runtime_state(_api):
        return {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 85}

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    monkeypatch.setattr(
        nt,
        "_load_settings",
        lambda _api: {"TELEGRAM_NOTIFY_BUDGET": "on", "TELEGRAM_CHAT_ID": "42"},
    )

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(nt.asyncio, "sleep", stop_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(nt._make_notifier(api)())

    sd = Path(api.get_state_dir())
    saved = json.loads((sd / "notif_state.json").read_text(encoding="utf-8"))
    assert saved["budget_threshold"] == 80


def test_notifier_transient_backoff_grows_and_resets_with_transition_logs(tmp_path, monkeypatch):
    nt = _load(); _api_obj, data = _api(tmp_path)

    logs = []

    class LoggingApi:
        def __init__(self, inner): self._inner = inner
        def get_state_dir(self): return self._inner.get_state_dir()
        def get_settings(self, k): return self._inner.get_settings(k)
        def log(self, level, message, **fields): logs.append((level, message))

    api = LoggingApi(_api_obj)
    outcomes = iter([
        nt.TelegramTransportError("offline one"),
        nt.TelegramTransportError("offline two"),
        1,  # delivered → backoff resets
        nt.TelegramTransportError("offline again"),
    ])

    class Flapping:
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr(nt, "TelegramClient", Flapping)
    budgets = iter([85, 85, 85, 92, 100])

    async def runtime_state(_api):
        pct = next(budgets)
        return {"spent_usd": 10 * pct, "budget_limit": 1000, "budget_pct": pct}

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    monkeypatch.setattr(
        nt,
        "_load_settings",
        lambda _api: {"TELEGRAM_NOTIFY_BUDGET": "on", "TELEGRAM_CHAT_ID": "42"},
    )
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 4:
            raise asyncio.CancelledError

    monkeypatch.setattr(nt.asyncio, "sleep", record_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(nt._make_notifier(api)())

    # transient backoff, grown backoff, healthy pacing sleep, reset backoff
    assert sleeps[0] == 5
    assert sleeps[1] > sleeps[0]
    assert sleeps[2] == 30
    assert sleeps[3] == 5
    degraded = [m for _lvl, m in logs if "degraded" in m]
    recovered = [m for _lvl, m in logs if "recovered" in m]
    assert len(degraded) == 2 and "TelegramTransportError" in degraded[0]
    assert len(recovered) == 1


def test_notifier_clears_degraded_silently_when_pending_work_evaporates(tmp_path, monkeypatch):
    """'recovered' is declared only when a send was actually delivered. When
    the pending work merely evaporates (here: the toggle flips off) the
    degraded episode ends WITHOUT the recovered line, and a later new failure
    opens a fresh episode with its own transition warning."""
    nt = _load(); _api_obj, data = _api(tmp_path)

    logs = []

    class LoggingApi:
        def __init__(self, inner): self._inner = inner
        def get_state_dir(self): return self._inner.get_state_dir()
        def get_settings(self, k): return self._inner.get_settings(k)
        def log(self, level, message, **fields): logs.append((level, message))

    api = LoggingApi(_api_obj)

    class AlwaysOffline:
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            raise nt.TelegramTransportError("offline")

    monkeypatch.setattr(nt, "TelegramClient", AlwaysOffline)

    async def runtime_state(_api):
        return {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 85}

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    settings_cycles = iter([
        {"TELEGRAM_NOTIFY_BUDGET": "on", "TELEGRAM_CHAT_ID": "42"},
        {"TELEGRAM_NOTIFY_BUDGET": "off", "TELEGRAM_CHAT_ID": "42"},
        {"TELEGRAM_NOTIFY_BUDGET": "on", "TELEGRAM_CHAT_ID": "42"},
    ])
    monkeypatch.setattr(nt, "_load_settings", lambda _api: next(settings_cycles))
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(nt.asyncio, "sleep", record_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(nt._make_notifier(api)())

    # degraded backoff, silent episode end (healthy pacing), fresh episode
    assert sleeps == [5, 30, 5]
    degraded = [m for _lvl, m in logs if "degraded" in m]
    recovered = [m for _lvl, m in logs if "recovered" in m]
    assert len(degraded) == 2, degraded
    assert recovered == [], recovered


def test_notifier_transient_send_failures_log_at_debug_only(tmp_path, monkeypatch):
    """Per-attempt transient failures stay at debug; the degraded transition
    warning is the one owner-visible line of the episode."""
    nt = _load(); _api_obj, data = _api(tmp_path)

    logs = []

    class LoggingApi:
        def __init__(self, inner): self._inner = inner
        def get_state_dir(self): return self._inner.get_state_dir()
        def get_settings(self, k): return self._inner.get_settings(k)
        def log(self, level, message, **fields): logs.append((level, message))

    api = LoggingApi(_api_obj)

    class AlwaysOffline:
        def __init__(self, _token, **_kwargs): pass
        async def send_message(self, *_args, **_kwargs):
            raise nt.TelegramTransportError("offline")

    monkeypatch.setattr(nt, "TelegramClient", AlwaysOffline)

    async def runtime_state(_api):
        return {"spent_usd": 850, "budget_limit": 1000, "budget_pct": 85}

    monkeypatch.setattr(nt, "_load_runtime_state", runtime_state)
    transient, delivered = asyncio.run(
        nt._check_budget_notify(api, {"TELEGRAM_NOTIFY_BUDGET": "on"}, 42, {}, "en")
    )

    assert transient is not None and delivered is False
    attempt_logs = [(lvl, m) for lvl, m in logs if "Telegram notify failed" in m]
    assert attempt_logs and all(lvl == "debug" for lvl, _m in attempt_logs)


def test_notifier_local_failure_reaches_supervisor(tmp_path, monkeypatch):
    nt = _load(); api, _data = _api(tmp_path)

    def broken_settings(_api):
        raise RuntimeError("local notifier defect")

    monkeypatch.setattr(nt, "_load_settings", broken_settings)
    with pytest.raises(RuntimeError, match="local notifier defect"):
        asyncio.run(nt._make_notifier(api)())


def test_tasks_notify_reads_lifecycle_status_and_severity_icon(tmp_path, monkeypatch):
    """The lifecycle AXIS is a dict — read `.status`, never str() the container.

    Pushes once read `⚠️ Task 8023b715 done · 25r · $7.98 · {'status':
    'completed'}`: every task looked degraded and leaked a Python dict repr.
    The word now comes from `outcome_axes.lifecycle.status` (legacy bare-string
    rows still resolve), and the icon additionally warns on a degraded/
    best_effort execution axis. This adapter is deliberately NARROWER than the
    web card's `taskOutcomeSeverity`: failed and cancelled are shown by their
    lifecycle word with the same ⚠️, not with a distinct icon.
    """
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    rows = [
        {"task_id": "ok1", "rounds": 25,
         "outcome_axes": {"lifecycle": {"status": "completed"},
                          "execution": {"status": "ok", "reason_code": ""}}},
        {"task_id": "deg1",
         "outcome_axes": {"lifecycle": {"status": "completed"},
                          "execution": {"status": "degraded",
                                        "reason_code": "plan_review_advisory"}}},
        {"task_id": "fail1",
         "outcome_axes": {"lifecycle": {"status": "failed"},
                          "execution": {"status": "failed", "reason_code": "boom"}}},
        {"task_id": "legacy1", "outcome_axes": {"lifecycle": "completed"}},
    ]
    with open(data / "logs" / "chat.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"type": "task_summary", **_OWNER_ROOT, **row}) + "\n")
    state = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))

    assert [text for _chat, text in _Rec.sent] == [
        "✅ Task ok1 done · 25r",
        "⚠️ Task deg1 done",
        "⚠️ Task fail1 done · failed",
        "✅ Task legacy1 done",
    ]
    # The container is never stringified into an owner-visible push.
    assert all("{'status'" not in text for _chat, text in _Rec.sent)


def test_tasks_notify_consumes_the_host_status_phase_when_the_row_carries_one(
    tmp_path, monkeypatch,
):
    """S5-03: the host stamps ONE phase on its own task rows, so the notifier
    consumes it instead of re-deriving a third status ladder.

    Two owner-visible flips follow from the card's rule: a task whose execution
    was clean but whose acceptance review degraded now warns (it used to read
    ✅), and an owner-requested stop is a success (it used to warn because the
    stop leaves a best_effort execution axis). Legacy rows without the field
    keep the axes rule; a pre-finalization ``working`` row is not a candidate.
    """
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    rows = [
        {"task_id": "review1", "outcome_phase": "warn",
         "outcome_axes": {"lifecycle": {"status": "completed"},
                          "execution": {"status": "ok"},
                          "review": {"status": "degraded"}}},
        {"task_id": "stop1", "outcome_phase": "done",
         "outcome_axes": {"lifecycle": {"status": "completed"},
                          "execution": {"status": "best_effort"}}},
        {"task_id": "open1", "outcome_phase": "working",
         "outcome_axes": {"lifecycle": {"status": "completed"},
                          "execution": {"status": "degraded"}}},
    ]
    with open(data / "logs" / "chat.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"type": "task_summary", **_OWNER_ROOT, **row}) + "\n")
    state = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))

    assert [text for _chat, text in _Rec.sent] == [
        "⚠️ Task review1 done with warnings",
        "✅ Task stop1 done",
    ]
    assert state["notified_task_ids"] == ["review1", "stop1"]


def test_tasks_notify_word_and_icon_follow_the_host_phase(tmp_path, monkeypatch):
    """#538: a failed or cancelled task is never announced as "done". The word
    and icon come from the stamped phase (the web card's own vocabulary); the
    lifecycle tail is not repeated as a second status word. Russian follows."""
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    rows = [
        {"task_id": "fail1", "outcome_phase": "error", "rounds": 3,
         "outcome_axes": {"lifecycle": {"status": "failed"}, "execution": {"status": "failed"}}},
        {"task_id": "stop1", "outcome_phase": "cancelled",
         "outcome_axes": {"lifecycle": {"status": "cancelled"}, "execution": {"status": "best_effort"}}},
        {"task_id": "warn1", "outcome_phase": "warn",
         "outcome_axes": {"lifecycle": {"status": "completed"}, "review": {"status": "degraded"}}},
        {"task_id": "ok1", "outcome_phase": "done",
         "outcome_axes": {"lifecycle": {"status": "completed"}}},
        {"task_id": "odd1", "outcome_phase": "someday",
         "outcome_axes": {"lifecycle": {"status": "failed"}}},
    ]
    with open(data / "logs" / "chat.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"type": "task_summary", **_OWNER_ROOT, **row}) + "\n")
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
    assert [text for _chat, text in _Rec.sent] == [
        "❌ Task fail1 failed · 3r",
        "🚫 Task stop1 stopped",
        "⚠️ Task warn1 done with warnings",
        "✅ Task ok1 done",
        "⚠️ Task odd1 done · failed",  # unknown phase: the legacy axes rule
    ]
    _Rec.sent = []
    # Russian is not a second table: the install's translation memory holds the push
    # template and the host's headline words, written by the generator or an import.
    from ouroboros import i18n_memory as memory

    nt.telegram_i18n.configure(data)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(data, "ru", lambda doc: (memory.apply_generated(doc, {
        "code:tg.notify.task_finished": {"text": "{icon} Задача {id} {word}{tail}"},
        "code:task.headline.error": {"text": "Ошибка"}, "code:task.headline.cancelled": {"text": "Остановлена"},
        "code:task.headline.warn": {"text": "Готова с предупреждениями"}, "code:task.headline.done": {"text": "Готова"},
    }, model="test"), doc)[1], create=True)
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "ru"))
    assert [text for _chat, text in _Rec.sent] == [
        "❌ Задача fail1 ошибка · 3r",
        "🚫 Задача stop1 остановлена",
        "⚠️ Задача warn1 готова с предупреждениями",
        "✅ Задача ok1 готова",
        "⚠️ Задача odd1 готова · failed",
    ]


def _tree_cost(amount, *, final=True, unpriced=False, open_=False, rows=True, scope="root_tree"):
    return {"scope": scope, "tracked_amount": amount, "has_unpriced": unpriced, "tracked_final": final,
            "accounting_open": open_, "has_rows": rows}


def _notify_one_root(nt, api, data, monkeypatch, result):
    _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda _root, _tid: result)
    row = {"type": "task_summary", **_OWNER_ROOT, "task_id": "cost1",
           "outcome_axes": {"lifecycle": {"status": "completed"}}}
    (data / "logs" / "chat.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
    return [text for _chat, text in _Rec.sent]


@pytest.mark.parametrize(("result", "line"), [
    ({"cost_presentation": _tree_cost(3.42)}, "✅ Task cost1 done · $3.42"),
    ({"cost_presentation": _tree_cost(3.42, final=False, open_=True)}, "✅ Task cost1 done · up to $3.42"),
    ({"cost_presentation": _tree_cost(3.42, final=False, unpriced=True)}, "✅ Task cost1 done · Tracked: up to $3.42"),
    ({"cost_presentation": _tree_cost(3.42, unpriced=True)}, "✅ Task cost1 done · Tracked: $3.42"),
    ({"cost_presentation": _tree_cost(0.125)}, "✅ Task cost1 done · $0.13"),  # toFixed(2) rounds a tie up
    ({"cost_presentation": _tree_cost(0.0)}, "✅ Task cost1 done · $0.00"),  # a priced zero is a real zero
])
def test_task_notify_shows_the_cost_in_the_cards_words(tmp_path, monkeypatch, result, line):
    """#971: the phone line carries the card's cost words for the root's whole tree."""
    nt = _load(); api, data = _api(tmp_path)
    assert _notify_one_root(nt, api, data, monkeypatch, result) == [line]


@pytest.mark.parametrize("result", [
    {"cost_presentation": _tree_cost(None, final=False, open_=True, rows=False)},  # the card's "Cost unknown"
    {"cost_presentation": _tree_cost(None, final=False, unpriced=True)},  # only unpriced rows: no measured zero
    {"cost_presentation": None, "cost_accounting_status": "unavailable"},
    {"cost_presentation": _tree_cost(1.5, scope="own")},  # a root's own share is not the tree's cost
    {"cost_usd": 7.98},  # the retired alias is not a current price
    {"accounted_upper_bound_usd": 7.98, "cost_usd": 7.98},
])
def test_task_notify_says_nothing_about_money_it_does_not_know(tmp_path, monkeypatch, result):
    nt = _load(); api, data = _api(tmp_path)
    assert _notify_one_root(nt, api, data, monkeypatch, result) == ["✅ Task cost1 done"]


def test_task_notify_reads_the_cost_the_task_result_store_keeps(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    write_task_result(data, "cost2", STATUS_COMPLETED, cost_presentation=_tree_cost(12.5, final=False))
    row = {"type": "task_summary", **_OWNER_ROOT, "task_id": "cost2",
           "outcome_axes": {"lifecycle": {"status": "completed"}}}
    (data / "logs" / "chat.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
    assert [text for _chat, text in _Rec.sent] == ["✅ Task cost2 done · up to $12.50"]
