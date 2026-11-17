"""A Telegram request that fails after the bot token changed belongs to the replaced token.

The production TelegramClient, token binding, notifier, poller, typing cue and Connection status run over
``httpx.MockTransport`` with synthetic tokens; nothing reaches the network. The token changes while a request
is in flight (the transport applies the save as the token observer does), then Telegram's answer is a refusal,
a broken transport or an unreadable body. Each such failure ends as ``TelegramBindingChanged`` carrying the
parts Telegram accepted, so the policy of a token change decides it: a notice the old bot never took goes once
through the token in force, one it partly took is never sent again, nothing more is asked with the old token,
and no status of the new token inherits the old answer. Controls: the same answers without a change keep their
ordinary meaning for the token in force (a refusal is consumed, a broken transport retries the whole notice).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time

import httpx
import pytest

from ouroboros import dialogue_store as ds
from tests.test_telegram_proxy import _Api, _load

_ASYNC_CLIENT = httpx.AsyncClient  # the real one, before any test patches it
KEY = "TELEGRAM_BOT_TOKEN"
OLD = "111:synthetic-old"
CHANGES = {"same_bot": "111:synthetic-rotated", "new_bot": "222:synthetic-new", "cleared": ""}
LONG = "word " * 1500  # two Telegram messages
KINDS = ("refused", "not_ok", "server_error", "transport", "timeout", "invalid_json", "not_an_object")
# What each answer means for the token in force, as before this change.
ORDINARY = {
    "refused": ("skipped", "TelegramRequestRejected"),
    "not_ok": ("skipped", "TelegramRequestRejected"),
    "server_error": ("transient", "TelegramRequestRejected"),
    "transport": ("transient", "TelegramTransportError"),
    "timeout": ("transient", "TelegramTransportError"),
    "invalid_json": ("transient", "TelegramTransportError"),
    "not_an_object": ("transient", "TelegramTransportError"),
}


def _failure(kind: str, request: httpx.Request) -> httpx.Response:
    if kind == "transport":
        raise httpx.ReadError("synthetic interrupted answer", request=request)
    if kind == "timeout":
        raise httpx.ReadTimeout("synthetic timeout", request=request)
    if kind == "refused":
        return httpx.Response(401, json={"ok": False, "error_code": 401, "description": "Unauthorized"})
    if kind == "not_ok":
        return httpx.Response(200, json={"ok": False, "error_code": 403, "description": "Forbidden"})
    if kind == "server_error":
        return httpx.Response(502, json={"ok": False, "error_code": 502, "description": "Bad Gateway"})
    if kind == "invalid_json":
        return httpx.Response(200, content=b"<html>not the Bot API</html>")
    return httpx.Response(200, json=["not", "an", "object"])


def _ok(result=None) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": {"message_id": 5} if result is None else result})


def _transport(monkeypatch, telegram_api, handler) -> list[str]:
    """Answer every Telegram request of the test with ``handler(request, number)``; return the paths asked."""
    asked: list[str] = []

    def respond(request):
        asked.append(request.url.path)
        return handler(request, len(asked))

    monkeypatch.setattr(telegram_api.httpx, "AsyncClient",
                        lambda **kw: _ASYNC_CLIENT(transport=httpx.MockTransport(respond), **kw))
    return asked


def _world(tmp_path, token: str = OLD):
    plugin, telegram_api, notifier = _load()
    connection = sys.modules[plugin.__package__ + ".lib.telegram_connection"]
    state = tmp_path / "state"
    state.mkdir()
    (state / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    api = _Api(state, {KEY: token})
    binding = notifier.binding_for(api)
    binding.read_now(api)
    return plugin, telegram_api, notifier, connection, api, binding


def _save(api, binding, token: str) -> None:
    """A Settings save the token observer applies."""
    api.granted[KEY] = token
    binding.read_now(api)


@pytest.mark.parametrize("taken", [0, 1], ids=["none_taken", "one_part_taken"])
@pytest.mark.parametrize("change", [None, *CHANGES], ids=["no_change", *CHANGES])
@pytest.mark.parametrize("kind", KINDS)
def test_a_notice_whose_failure_arrives_after_a_token_change_follows_the_change(tmp_path, monkeypatch, kind, change,
                                                                                 taken):
    _plugin, telegram_api, notifier, _connection, api, binding = _world(tmp_path)

    def handler(request, number):
        if number <= taken:
            return _ok()
        if change is not None:
            _save(api, binding, CHANGES[change])
        return _failure(kind, request)

    asked = _transport(monkeypatch, telegram_api, handler)
    outcome, error = asyncio.run(notifier._push_notification(api, 42, LONG if taken else "Notice"))
    assert asked == [f"/bot{OLD}/sendMessage"] * (taken + 1), "nothing more is asked, through either token"
    if change is None:
        assert (outcome, type(error).__name__) == ORDINARY[kind]
        return
    assert type(error).__name__ == "TelegramBindingChanged" and error.accepted == taken
    # Untaken: the next cycle sends it through the token in force. Partly taken: those parts stay, the rest is dropped.
    assert outcome == ("skipped" if taken else "transient")
    assert OLD not in str(error) and "synthetic" not in str(error)


def _summaries(api, *ids):
    for tid in ids:
        ds.append(api.state_dir, "chat", {"type": "task_summary", "task_id": tid, "outcome_final": True,
                                          "summary_kind": "terminal_root_projection", "chat_id": 1,
                                          "outcome_phase": "done",
                                          "outcome_axes": {"lifecycle": {"status": "completed"}}})


@pytest.mark.parametrize("taken", [0, 1], ids=["never_taken", "partly_taken"])
def test_the_next_cycle_sends_an_untaken_notice_once_through_the_new_bot_and_never_resends_a_partial_one(
        tmp_path, monkeypatch, taken):
    _plugin, telegram_api, notifier, _connection, api, binding = _world(tmp_path)
    monkeypatch.setattr(notifier, "_completion_text", lambda row, tid, lang, parts=(): LONG if taken else f"Done {tid}")
    _summaries(api, "first")
    settings, state = {"TELEGRAM_NOTIFY_TASKS": "on"}, {"notified_task_ids": []}

    def handler(request, number):
        if request.url.path.startswith(f"/bot{OLD}/") and number == taken + 1:
            _save(api, binding, CHANGES["new_bot"])
            raise httpx.ReadError("synthetic interrupted answer", request=request)
        return _ok()

    asked = _transport(monkeypatch, telegram_api, handler)
    transient, delivered = asyncio.run(notifier._check_tasks_notify(api, settings, 42, state, "en"))
    assert delivered is False
    if taken:
        assert transient is None and state["notified_task_ids"] == ["first"], "the old bot's parts consume it"
    else:
        assert type(transient).__name__ == "TelegramBindingChanged" and state["notified_task_ids"] == []
    transient, delivered = asyncio.run(notifier._check_tasks_notify(api, settings, 42, state, "en"))
    assert transient is None and state["notified_task_ids"] == ["first"]
    new_bot = f"/bot{CHANGES['new_bot']}/sendMessage"
    assert asked == [f"/bot{OLD}/sendMessage"] * (taken + 1) + ([] if taken else [new_bot])
    assert delivered is (not taken)


async def _until(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("change", ["same_bot", "new_bot"])
@pytest.mark.parametrize("stage", ["getMe", "getUpdates"])
@pytest.mark.parametrize("kind", ["refused", "server_error", "transport"])
def test_a_replaced_tokens_late_poll_failure_never_becomes_the_new_tokens_status(tmp_path, monkeypatch, stage,
                                                                                  change, kind):
    """The poller records nothing for the replaced token's late answer (no refusal, no wait, no park); the new
    token is validated and polled at once, and its own status is the only one shown."""
    plugin, telegram_api, _notifier, connection, api, binding = _world(tmp_path)
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    monkeypatch.setattr(plugin, "TELEGRAM_RETRY_INITIAL_SEC", 0.01)  # the unpatched tree's retry, kept short
    new = CHANGES[change]
    written = []
    real_save = connection._save_bridge_status

    def save(api_, state_, reason_code="", bot="", generation=None):
        written.append((state_, reason_code, bot))
        real_save(api_, state_, reason_code, bot, generation)

    monkeypatch.setattr(plugin, "_save_bridge_status", save)
    monkeypatch.setattr(connection, "_save_bridge_status", save)

    def handler(request, _number):
        method = request.url.path.rsplit("/", 1)[-1]
        if request.url.path.startswith(f"/bot{OLD}/") and method == stage:
            _save(api, binding, new)
            return _failure(kind, request)
        if method == "getMe":
            return _ok({"id": int(new.split(":")[0] if request.url.path.startswith(f"/bot{new}/") else 111)})
        return _ok([] if method == "getUpdates" else True)

    asked = _transport(monkeypatch, telegram_api, handler)

    async def scenario():
        poller = asyncio.create_task(plugin._poller(api))
        try:
            await _until(lambda: f"/bot{new}/getUpdates" in asked)
            return connection._connection_fields(api)
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller

    status = asyncio.run(scenario())
    assert not [row for row in written if row[0] in ("error", "degraded")], written
    assert ("ready", "", new.split(":")[0]) in written
    assert status["bridge"]["state"] == "ready" and status["bridge_status"] == "Connected"
    assert not [message for level, message in api.logs if level in ("warning", "error")], api.logs
    old_requests = [path for path in asked if path.startswith(f"/bot{OLD}/")]
    assert old_requests[-1].endswith(f"/{stage}"), "the replaced token asked nothing after its late answer"


def test_a_same_bot_rotation_does_not_inherit_the_replaced_tokens_refusal_or_wait(tmp_path):
    """Astra's counterexample and its siblings: a refusal, a conflict or an outage recorded for the old token
    reads as connecting for the new one, which has not been tried yet; the new token's own record shows."""
    _plugin, _telegram_api, _notifier, connection, api, binding = _world(tmp_path)
    for state, reason, words in (("error", "telegram_startup_failed", "Stopped: Telegram refused this token"),
                                 ("error", "telegram_rejected", "Stopped: Telegram refused this token"),
                                 ("degraded", "telegram_conflict", "Waiting: a webhook or another copy"),
                                 ("degraded", "telegram_unreachable", "Can't reach Telegram now")):
        connection._save_bridge_status(api, state, reason, bot="111")
        before = connection._connection_fields(api)
        assert before["bridge"]["state"] == state and before["bridge_status"].startswith(words)
        _save(api, binding, f"111:synthetic-{reason}")
        after = connection._connection_fields(api)
        assert after["bridge"]["state"] == "connecting", (reason, after["bridge"])
        assert after["bridge_status"] == "Connecting with the saved token…" and "Save" not in after["bridge_status"]


def test_a_record_of_a_replaced_generation_is_dropped_and_the_current_one_stays(tmp_path):
    _plugin, _telegram_api, _notifier, connection, api, binding = _world(tmp_path)
    replaced = binding.snapshot()[1]
    _save(api, binding, CHANGES["same_bot"])
    connection._save_bridge_status(api, "connecting", "telegram_token_changed", bot="111", generation=binding.snapshot()[1])
    connection._save_bridge_status(api, "error", "telegram_rejected", bot="111", generation=replaced)
    assert json.loads((api.state_dir / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "connecting", "reason_code": "telegram_token_changed", "bot": "111"}, "the late refusal wrote nothing"
    assert connection._bridge_status(api)["state"] == "connecting"


def test_a_cold_start_proves_nothing_from_an_earlier_loads_file(tmp_path):
    """The saved record of an earlier load (here a refusal of this very token) is not this load's status; this
    load's poller recording the same refusal is. getMe validation and a later refusal stay separate facts."""
    _plugin, _telegram_api, _notifier, connection, api, binding = _world(tmp_path)
    (api.state_dir / "bridge_status.json").write_text(
        json.dumps({"state": "error", "reason_code": "telegram_rejected", "bot": "111"}), encoding="utf-8")
    assert connection._bridge_status(api)["state"] == "connecting"
    binding.mark_validated(binding.snapshot()[1])
    connection._save_bridge_status(api, "ready", bot="111")
    assert connection._bridge_status(api)["state"] == "ready"
    connection._save_bridge_status(api, "error", "telegram_rejected", bot="111")  # getUpdates refused after getMe
    assert binding.is_validated() and connection._bridge_status(api) == {
        "state": "error", "token": "configured", "owner_bound": True, "poller": "failed",
        "command_mode": "full_access", "mirror_mode": "all", "reason_code": "telegram_rejected"}
    _save(api, binding, CHANGES["same_bot"])
    assert connection._bridge_status(api)["state"] == "connecting", "neither the proof nor the refusal carries over"
    _save(api, binding, "")
    assert connection._bridge_status(api)["state"] == "missing_token"


def test_a_download_failing_after_a_token_change_reads_no_local_copy_and_keeps_the_update_unconfirmed(tmp_path,
                                                                                                      monkeypatch):
    plugin, telegram_api, notifier, _connection, api, binding = _world(tmp_path)
    store = tmp_path / "store"
    (store / "photos").mkdir(parents=True)
    (store / "photos" / "a.jpg").write_bytes(b"local copy")

    def run(change: bool) -> tuple[list[str], object]:
        def handler(request, _number):
            if request.url.path.endswith("/getFile"):
                return _ok({"file_path": "photos/a.jpg"})
            if change:
                _save(api, binding, CHANGES["same_bot"])
            return httpx.Response(404, content=b"gone")

        asked = _transport(monkeypatch, telegram_api, handler)
        token, generation = binding.snapshot()
        client = telegram_api.TelegramClient(token, files_dir=str(store), binding=binding, generation=generation)
        try:
            return asked, asyncio.run(client.download_file("file-1"))
        except Exception as exc:  # noqa: BLE001 — the outcome is the subject
            return asked, exc

    asked, outcome = run(change=False)
    assert outcome == b"local copy", "the token in force still falls back to the local store"
    asked, outcome = run(change=True)
    assert type(outcome).__name__ == "TelegramBindingChanged" and outcome.accepted == 1
    assert [path.rsplit("/", 1)[-1] for path in asked] == ["getFile", "a.jpg"]


def test_the_typing_cue_is_not_muted_by_a_replaced_tokens_late_refusal(tmp_path, monkeypatch):
    plugin, telegram_api, _notifier, _connection, api, binding = _world(tmp_path)

    def run(change: bool):
        def handler(request, _number):
            if change:
                _save(api, binding, CHANGES["same_bot"] if binding.snapshot()[0] == OLD else OLD)
            return _failure("refused", request)

        _transport(monkeypatch, telegram_api, handler)
        cue = plugin._typing_cue(api)  # the bridge's own: its client factory and the accepted token's generation
        cue._rescope()  # as each read does before its sends
        asyncio.run(cue._send(42))
        return cue

    def warnings():
        return [message for level, message in api.logs if level == "warning"]

    cue = run(change=True)
    assert 42 not in cue._muted and not warnings(), "a token change waits, never mutes"
    assert 42 not in cue._retry_at and 42 not in cue._delay, "nor defers the replacement's next action"
    cue = run(change=False)
    assert 42 in cue._muted and len(warnings()) == 1, "the token in force refused: muted once, loudly"


def _typing_world(tmp_path, monkeypatch, handler):
    """The bridge's own typing cue (``plugin._typing_cue``: its client factory, mirror policy and the accepted
    token's generation) over ``handler(token)`` (async), the host's projection scripted to one prepared reply
    and its clocks shortened. Old-token backoff keeps its real 5 s start, so a cue it delayed is seen."""
    import types

    plugin, telegram_api, _notifier, _connection, api, binding = _world(tmp_path)
    typing = sys.modules[plugin.__package__ + ".lib.telegram_typing"]
    monkeypatch.setattr(typing, "_REFRESH_SEC", 0.1)
    monkeypatch.setattr(typing, "_MIN_READ_GAP_SEC", 0.01)
    asked: list[str] = []

    async def respond(request):
        asked.append(request.url.path)
        return await handler(request.url.path.split("/")[1][len("bot"):])

    monkeypatch.setattr(telegram_api.httpx, "AsyncClient",
                        lambda **kw: _ASYNC_CLIENT(transport=httpx.MockTransport(respond), **kw))
    cue = plugin._typing_cue(api)
    reads = []
    cue._read_state = lambda: reads.append(1) or {"complete": True, "chats": [{"chat_id": 1, "transport": {}}]}
    return types.SimpleNamespace(
        api=api, binding=binding, cue=cue, reads=reads,
        typing=lambda token: [path for path in asked if path == f"/bot{token}/sendChatAction"],
        asked_with=lambda token: [path for path in asked if path.startswith(f"/bot{token}/")],
        warnings=lambda: [message for level, message in api.logs if level == "warning"])


async def _stopped(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def test_a_replacement_brings_back_typing_its_replaced_tokens_refusal_had_muted(tmp_path, monkeypatch):
    """Old token refused (401): the chat is muted once, loudly, and stays so. The owner saves a working token: on
    the change's wake the cue starts typing through it at once, with no message delivered first, and nothing more
    goes out with the old token. Then the replacement's own refusal mutes the chat again (same generation) until a
    message reaches it: the change starts the suppression over, it does not switch it off."""
    refusing = {OLD}
    new = CHANGES["new_bot"]

    async def handler(token):
        return _failure("refused", None) if token in refusing else _ok(True)

    world = _typing_world(tmp_path, monkeypatch, handler)
    cue = world.cue

    async def scenario():
        task = asyncio.ensure_future(cue.run())
        try:
            await _until(lambda: 42 in cue._muted)
            assert world.typing(OLD) == [f"/bot{OLD}/sendChatAction"] and len(world.warnings()) == 1
            await asyncio.sleep(0.25)  # the renewal pause it was in ends with one more read, then it idles
            reads = len(world.reads)
            await asyncio.sleep(0.4)  # four renewal spacings
            assert len(world.typing(OLD)) == 1 and len(world.reads) == reads, "a muted-only chat idles"
            _save(world.api, world.binding, new)
            cue.wake()  # the token observer's wake (its own test: test_telegram_poll_binding)
            await _until(lambda: world.typing(new))
            assert len(world.asked_with(OLD)) == 1, "nothing more with the replaced token"
            await _until(lambda: len(world.typing(new)) >= 3)  # renewed at its spacing
            refusing.add(new)
            await _until(lambda: 42 in cue._muted)
            refused = len(world.typing(new))
            assert len(world.warnings()) == 2
            for _ in range(5):
                cue.wake()  # a read or a wake never unmutes; nor does an unchanged token
                await asyncio.sleep(0.05)
            assert len(world.typing(new)) == refused and 42 in cue._muted
            refusing.discard(new)
            async with cue.message(42) as delivered:
                delivered()
            await _until(lambda: len(world.typing(new)) > refused)
        finally:
            await _stopped(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("answer", ["refused", "server_error", "transport", "ok"])
def test_a_replaced_tokens_late_answer_neither_mutes_nor_delays_the_replacement(tmp_path, monkeypatch, answer):
    """The old token's action is on the wire when the owner saves a replacement; its answer arrives after the change.
    A refusal, a failure or a success of the replaced token records nothing: the replacement types promptly (well
    inside the 5 s backoff a failure of the token in force would start), unmuted and with no warning."""
    new = CHANGES["new_bot"]
    gate = {}

    async def handler(token):
        if token == OLD:
            gate["entered"].set()
            await gate["release"].wait()
            if answer == "transport":
                raise httpx.ReadError("synthetic interrupted answer")
            return _ok(True) if answer == "ok" else _failure(answer, None)
        return _ok(True)

    world = _typing_world(tmp_path, monkeypatch, handler)
    cue = world.cue

    async def scenario():
        gate.update(entered=asyncio.Event(), release=asyncio.Event())
        task = asyncio.ensure_future(cue.run())
        try:
            await asyncio.wait_for(gate["entered"].wait(), 2)
            _save(world.api, world.binding, new)
            reads = len(world.reads)
            await _until(lambda: len(world.reads) > reads)  # the cue's next read sees the change (in flight: waits)
            gate["release"].set()
            started = time.monotonic()
            await _until(lambda: world.typing(new), timeout=1.5)
            assert time.monotonic() - started < 1.5
            assert 42 not in cue._muted and 42 not in cue._retry_at and not world.warnings()
        finally:
            await _stopped(task)

    asyncio.run(scenario())


def test_a_replaced_tokens_late_success_keeps_the_replacements_backoff(tmp_path, monkeypatch):
    """Per chat one action is in flight, so the cue itself never interleaves these; the rule is pinned directly:
    the replacement's transient failure backs its chat off, and the old token's success arriving afterwards
    clears nothing of it. A success of the token in force still clears its own backoff."""
    new = CHANGES["new_bot"]
    gate = {}

    async def handler(token):
        if token == OLD:
            await gate["release"].wait()
            return _ok(True)
        return _failure("server_error", None) if gate["failing"] else _ok(True)

    world = _typing_world(tmp_path, monkeypatch, handler)
    world.cue._read_state = lambda: world.reads.append(1) or {"complete": True, "chats": []}
    cue = world.cue

    async def scenario():
        gate.update(release=asyncio.Event(), failing=True)
        task = asyncio.ensure_future(cue.run())
        try:
            await _until(lambda: world.reads)  # its first read takes the token in force
            old = asyncio.ensure_future(cue._send(42))
            await asyncio.sleep(0.05)
            _save(world.api, world.binding, new)
            reads = len(world.reads)
            cue.wake()
            await _until(lambda: len(world.reads) > reads)
            await cue._send(42)  # the replacement's own transient failure
            backoff = (cue._delay.get(42), cue._retry_at.get(42))
            assert backoff[0] is not None and backoff[1] is not None
            gate["release"].set()
            await asyncio.wait_for(old, 2)
            assert (cue._delay.get(42), cue._retry_at.get(42)) == backoff, "the late success cleared the backoff"
            gate["failing"] = False
            await cue._send(42)
            assert 42 not in cue._delay and 42 not in cue._retry_at, "its own success clears it"
        finally:
            await _stopped(task)

    asyncio.run(scenario())
