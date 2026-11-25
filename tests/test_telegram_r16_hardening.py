"""Behavioral regression tests for Council Round 16 concurrency, teardown, and revocation hardening."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import threading
import time
import types

import pytest

from tests.test_telegram_security_boundaries import _StubApi, _load_plugin


def _open_fd_set() -> set[int]:
    proc_fd = pathlib.Path("/proc/self/fd")
    if not proc_fd.is_dir():
        return set()
    return {int(p.name) for p in proc_fd.iterdir() if p.name.isdigit()}


def _fd_open(fd: int) -> bool:
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


def test_runtime_open_verified_cancel_leaks_zero_fds(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_r16_rt")
    from tg_r16_rt.lib.presence import custody, outbox_files, runtime as rt_mod
    from tg_r16_rt.scripts.telegram_settings import merge_settings

    state = tmp_path / "state"
    api = _StubApi(state)
    merge_settings(state, {"TELEGRAM_CHAT_ID": "18"})
    src = tmp_path / "doc.bin"
    src.write_bytes(b"PAYLOAD_VERIFIED_R16")
    frozen = outbox_files.keep(src, state, 1024)

    store = custody.CustodyStore(state / "presence.sqlite3")
    payload = {
        "kind": "document",
        "account_id": "123456",
        "chat_id": "-1001",
        "topic_id": "",
        "file_path": frozen["path"],
        "_frozen": [frozen],
    }
    store.enqueue_outbox("d-cancel-1", payload)

    entered = threading.Event()
    release = threading.Event()
    opened_fds: list[int] = []
    real_open_verified = outbox_files.open_verified

    def slow_open_verified(p, **kw):
        res = real_open_verified(p, **kw)
        opened_fds.extend(fd for fd, _sz in res.values())
        entered.set()
        assert release.wait(5.0)
        return res

    monkeypatch.setattr(outbox_files, "open_verified", slow_open_verified)

    fake_submitter = types.SimpleNamespace(
        submit=lambda *a, **k: asyncio.sleep(0),
        poll=lambda *a, **k: asyncio.sleep(0),
    )
    fake_client = types.SimpleNamespace(bot_id="123456", download_limit=20 * 1024 * 1024)
    rt = rt_mod.PresenceRuntime(
        api=api, state_dir=state, submitter=fake_submitter, client_factory=lambda: fake_client
    )
    rt.store = store
    rt._bot_id = "123456"
    rt._validation_gen = 1
    monkeypatch.setattr(rt, "_ready_client", lambda: asyncio.sleep(0, result=fake_client))
    monkeypatch.setattr(rt, "_client_is_current", lambda _c, _g: True)
    monkeypatch.setattr(rt, "_resolved_reporting", lambda _p: asyncio.sleep(0, result={}))
    monkeypatch.setattr(rt, "_check_place", lambda *a, **k: asyncio.sleep(0))

    async def scenario():
        task = asyncio.create_task(rt.process_one_outbox())
        assert await asyncio.to_thread(entered.wait, 5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        deadline = time.monotonic() + 5.0
        while any(_fd_open(fd) for fd in opened_fds) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert opened_fds and all(not _fd_open(fd) for fd in opened_fds)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        store.close()


def test_transport_multipart_open_cancel_leaks_zero_fds_and_repeated_path_streams_both(tmp_path, monkeypatch):
    _load_plugin("tg_r16_tr")
    from tg_r16_tr.lib.presence import outbox_files, transport

    state = tmp_path / "state"
    src = tmp_path / "photo.bin"
    src.write_bytes(b"ALBUM_ITEM_BYTES_12345")
    frozen = outbox_files.keep(src, state, 1024)
    # 1) Repeated path across two album items streams BOTH parts completely without truncation
    vfds = outbox_files.open_verified(
        {"kind": "album", "items": [{"file_path": frozen["path"]}, {"file_path": frozen["path"]}], "_frozen": [frozen, frozen]},
        state_dir=state,
        require_for_media=True,
    )
    try:
        stream = transport._AsyncMultipartStream.open(
            {"chat_id": "1"},
            [("attach0", pathlib.Path(frozen["path"])), ("attach1", pathlib.Path(frozen["path"]))],
            vfds,
        )
        async def _read_all():
            return b"".join([chunk async for chunk in stream])

        body = asyncio.run(_read_all())
        assert body.count(b"ALBUM_ITEM_BYTES_12345") == 2
    finally:
        outbox_files.close_verified(vfds)

    # 2) Cancelling _post_multipart mid-_AsyncMultipartStream.open closes opened fds once thread completes
    entered = threading.Event()
    release = threading.Event()
    stream_fds: list[int] = []
    real_open = transport._AsyncMultipartStream.open

    def slow_open(fields, file_specs, verified_fds=None):
        st = real_open(fields, file_specs, verified_fds)
        stream_fds.extend(fd for _h, fd, _sz in st._file_parts)
        entered.set()
        assert release.wait(5.0)
        return st

    monkeypatch.setattr(transport._AsyncMultipartStream, "open", staticmethod(slow_open))
    client = transport.PresenceBotClient("123456:TOK_A")

    async def cancel_scenario():
        task = asyncio.create_task(client._post_multipart("sendDocument", {"chat_id": "1"}, [("document", src)]))
        assert await asyncio.to_thread(entered.wait, 5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        deadline = time.monotonic() + 5.0
        while any(_fd_open(fd) for fd in stream_fds) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert stream_fds and all(not _fd_open(fd) for fd in stream_fds)

    try:
        asyncio.run(cancel_scenario())
    finally:
        release.set()


def test_unload_from_worker_thread_cancels_cmd_sync_on_debug_loop_and_stops_presence(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_r16_un")
    state = tmp_path / "state"
    api = _StubApi(state)
    unload_cbs = []
    api.on_unload = lambda cb: unload_cbs.append(cb)
    api.register_supervised_task = lambda *a, **k: None
    api.subscribe_event = lambda *a, **k: None
    api.register_route = lambda *a, **k: None
    api.register_settings_section = lambda *a, **k: None
    api.register_tool = lambda *a, **k: None

    stopped = threading.Event()
    fake_presence = types.SimpleNamespace(
        run=lambda: asyncio.sleep(0),
        stop=lambda wait_sec=2.0: stopped.set(),
        on_token_changed=lambda: None,
    )
    monkeypatch.setattr(plugin, "_presence_runtime", lambda _a: fake_presence)
    monkeypatch.setattr(plugin, "register_miniapp", lambda _a: False)
    monkeypatch.setattr(plugin, "register_connection", lambda _a: None)
    monkeypatch.setattr(plugin, "register_presence_tools", lambda _a, _p: None)
    plugin.register(api)
    assert len(unload_cbs) == 1

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_debug(True)
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocking_sync(*_a, **_k):
            started.set()
            try:
                await asyncio.sleep(30.0)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(plugin, "_sync_bot_commands", blocking_sync)
        fake_cl = types.SimpleNamespace(bot_id="123456", api_base_origin="https://api.telegram.org")
        plugin._schedule_command_sync(api, fake_cl, "strict", "en", "18")
        await asyncio.wait_for(started.wait(), timeout=2.0)

        # Invoke _unload from a dedicated worker thread, matching extension_loader._run_unload_callback.
        worker_errors = []

        def run_unload():
            try:
                unload_cbs[0]()
            except BaseException as exc:
                worker_errors.append(exc)

        t = threading.Thread(target=run_unload)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive() and worker_errors == []
        assert stopped.is_set()
        await asyncio.wait_for(cancelled.wait(), timeout=2.0)

    asyncio.run(scenario())


def test_merge_settings_invalidates_pairing_before_publishing_owner_and_propagates_unlink_error(tmp_path, monkeypatch):
    _load_plugin("tg_r16_set")
    from tg_r16_set.lib import telegram_owner_auth as oa
    from tg_r16_set.scripts import telegram_settings as ts

    state = tmp_path / "state"
    ts.merge_settings(state, {"TELEGRAM_CHAT_ID": "18"})
    code = oa.issue_owner_pairing_code(state)["code"]
    pairing_file = state / "owner_pairing.json"
    assert pairing_file.is_file()

    real_unlink = pathlib.Path.unlink

    def fail_pairing_unlink(self, *args, **kwargs):
        if self.name == "owner_pairing.json":
            raise PermissionError("simulated unlink denial")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", fail_pairing_unlink)
    with pytest.raises(ts.TelegramSettingsError, match="invalidate pending owner pairing"):
        ts.merge_settings(state, {"TELEGRAM_CHAT_ID": ""})

    # Owner setting was NOT revoked because pairing invalidation failed first!
    assert ts.load_settings(state)["TELEGRAM_CHAT_ID"] == "18"
    assert json.loads(pairing_file.read_text(encoding="utf-8"))["code"] == code


def test_pairing_group_burn_bare_code_and_concurrent_reissue_safety(tmp_path):
    _load_plugin("tg_r16_pair")
    from tg_r16_pair.lib import telegram_owner_auth as oa

    state = tmp_path / "state"
    code1 = oa.issue_owner_pairing_code(state)["code"]
    # Bare code pasted in a group without /pair is burned immediately
    assert oa.is_pairing_command(state, {"chat": {"id": -1001, "type": "supergroup"}, "text": f"oops `{code1}`"}) is True
    assert not (state / "owner_pairing.json").exists()
    assert oa.redeem_owner_pairing_code(state, 99, f"/pair {code1}") is False

    # Atomic compare-and-burn never deletes a newer concurrently re-issued code
    code2 = oa.issue_owner_pairing_code(state)["code"]
    assert oa._burn_active_code_if_matches(state, code1) is False
    assert (state / "owner_pairing.json").exists()
    assert oa.redeem_owner_pairing_code(state, 18, f"/pair {code2}") is True


def test_receipt_visible_to_fails_closed_on_binding_lookup_error_and_strict_mode_lang_regex(monkeypatch):
    plugin = _load_plugin("tg_r16_misc")
    from tg_r16_misc.lib.presence import file_access

    monkeypatch.setattr(
        "ouroboros.presence_authority.presence_caller_binding",
        lambda _ctx: (_ for _ in ()).throw(RuntimeError("corrupt binding")),
    )
    ctx = types.SimpleNamespace(task_id="t1", task_metadata={})
    assert file_access.receipt_visible_to(ctx, {"account_id": "123", "chat_id": "1"}) is False
    assert plugin._MODE_CMD_RE.match("/mode@bot/restart") is None
    assert plugin._LANG_ARG_RE.match("/language@bot/restart") is None
    assert plugin._MODE_CMD_RE.match("/mode@my_bot presence") is not None
    assert plugin._LANG_ARG_RE.match("/language@my_bot Quenya") is not None
