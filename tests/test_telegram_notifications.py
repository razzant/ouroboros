import asyncio, importlib.util, json, sys, types
from pathlib import Path

import pytest


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


def test_tasks_notify_primes_then_fires(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    chat = data / "logs" / "chat.jsonl"
    chat.write_text(json.dumps({"type": "task_summary", "task_id": "old1", "rounds": 3,
                                "outcome_axes": {"lifecycle": "completed"}}) + "\n", encoding="utf-8")
    settings = {"TELEGRAM_NOTIFY_TASKS": "on"}; state = {}
    asyncio.run(nt._check_tasks_notify(api, settings, 42, state, "en"))   # primes, no send
    assert _Rec.sent == [] and "old1" in state["notified_task_ids"]
    with open(chat, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "task_summary", "task_id": "new1", "rounds": 5,
                            "outcome_axes": {"lifecycle": "completed"}}) + "\n")
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


def test_tasks_notify_waits_for_latest_final_summary_for_each_task(tmp_path, monkeypatch):
    nt = _load(); api, data = _api(tmp_path); _Rec.sent = []
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    chat = data / "logs" / "chat.jsonl"
    working = {
        "type": "task_summary", "task_id": "same1", "outcome_final": False,
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
    with open(chat, "a", encoding="utf-8") as f:
        f.write(json.dumps(terminal) + "\n")

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
        json.dumps({"type": "task_summary", "task_id": "retry1", "outcome_axes": {"lifecycle": "completed"}}) + "\n",
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
            f.write(json.dumps({"type": "task_summary", "task_id": tid,
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
            f.write(json.dumps({"type": "task_summary", "task_id": tid,
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
            f.write(json.dumps({"type": "task_summary", **row}) + "\n")
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
            f.write(json.dumps({"type": "task_summary", **row}) + "\n")
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
            f.write(json.dumps({"type": "task_summary", **row}) + "\n")
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
    assert [text for _chat, text in _Rec.sent] == [
        "❌ Task fail1 failed · 3r",
        "🚫 Task stop1 cancelled",
        "⚠️ Task warn1 done with warnings",
        "✅ Task ok1 done",
        "⚠️ Task odd1 done · failed",  # unknown phase: the legacy axes rule
    ]
    _Rec.sent = []
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "ru"))
    assert [text for _chat, text in _Rec.sent] == [
        "❌ Задача fail1 ошибка · 3r",
        "🚫 Задача stop1 отменена",
        "⚠️ Задача warn1 готова с предупреждениями",
        "✅ Задача ok1 готова",
        "⚠️ Задача odd1 готова · failed",
    ]
