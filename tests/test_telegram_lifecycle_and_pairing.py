"""Behavioral regressions for Telegram skill lifecycle, FD ownership, pairing revocation, and command matching."""

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


def _fd_open(fd: int) -> bool:
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


def test_runtime_open_verified_cancel_and_loop_shutdown_leak_zero_fds(tmp_path, monkeypatch):
    _load_plugin("tg_lc_rt")
    from tg_lc_rt.lib.presence import custody, outbox_files, runtime as rt_mod
    from tg_lc_rt.scripts.telegram_settings import merge_settings

    state = tmp_path / "state"
    api = _StubApi(state)
    merge_settings(state, {"TELEGRAM_CHAT_ID": "18"})
    src = tmp_path / "doc.bin"
    src.write_bytes(b"PAYLOAD_VERIFIED_LC")
    frozen = outbox_files.keep(src, state, 1024)

    real_open_verified = outbox_files.open_verified
    for mode in ("cancel_live", "loop_shutdown"):
        store = custody.CustodyStore(state / f"presence_{mode}.sqlite3")
        payload = {
            "kind": "document",
            "account_id": "123456",
            "chat_id": "-1001",
            "topic_id": "",
            "file_path": frozen["path"],
            "_frozen": [frozen],
        }
        store.enqueue_outbox(f"d-{mode}", payload)

        entered = threading.Event()
        release = threading.Event()
        opened_fds: list[int] = []

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
            if mode == "cancel_live":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                release.set()
                deadline = time.monotonic() + 5.0
                while any(_fd_open(fd) for fd in opened_fds) and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
            else:
                # Let asyncio.run() begin full teardown while worker thread is still blocked!
                threading.Timer(0.05, release.set).start()
                task.cancel()

        try:
            asyncio.run(scenario())
            deadline = time.monotonic() + 5.0
            while any(_fd_open(fd) for fd in opened_fds) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert opened_fds and all(not _fd_open(fd) for fd in opened_fds)
        finally:
            release.set()
            store.close()


def test_transport_multipart_cancel_shutdown_verified_fds_isolation_and_repeated_path(tmp_path, monkeypatch):
    _load_plugin("tg_lc_tr")
    from tg_lc_tr.lib.presence import outbox_files, transport

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

    # 2) Cancelling _post_multipart mid-open WITH populated verified_fds must NOT touch caller's closed/reused FD
    #    AND must close all owned FDs even across event-loop shutdown!
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"0123456789ABCDEF")
    entered = threading.Event()
    release = threading.Event()
    stream_fds: list[int] = []
    real_open = transport._AsyncMultipartStream.open

    def slow_open(fields, file_specs, verified_fds=None):
        entered.set()
        assert release.wait(5.0)
        st = real_open(fields, file_specs, verified_fds)
        stream_fds.extend(fd for _h, fd, _sz in st._file_parts)
        return st

    monkeypatch.setattr(transport._AsyncMultipartStream, "open", staticmethod(slow_open))
    client = transport.PresenceBotClient("123456:TOK_A")
    caller_vfds = outbox_files.open_verified(
        {"kind": "document", "file_path": frozen["path"], "_frozen": [frozen]},
        state_dir=state,
        require_for_media=True,
    )
    victim_fd: int | None = None

    async def cancel_with_reused_fd():
        nonlocal victim_fd
        task = asyncio.create_task(
            client._post_multipart(
                "sendDocument",
                {"chat_id": "1"},
                [("document", pathlib.Path(frozen["path"]))],
                verified_fds=caller_vfds,
            )
        )
        assert await asyncio.to_thread(entered.wait, 5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Simulate runtime.py's finally closing caller_vfds and an unrelated open reusing the integer FD
        outbox_files.close_verified(caller_vfds)
        victim_fd = os.open(victim, os.O_RDONLY)
        assert os.lseek(victim_fd, 7, os.SEEK_SET) == 7
        threading.Timer(0.03, release.set).start()

    try:
        asyncio.run(cancel_with_reused_fd())
        deadline = time.monotonic() + 5.0
        while (not stream_fds or any(_fd_open(fd) for fd in stream_fds)) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stream_fds and all(not _fd_open(fd) for fd in stream_fds)
        assert victim_fd is not None and os.lseek(victim_fd, 0, os.SEEK_CUR) == 7  # Zero FD reuse corruption!
    finally:
        release.set()
        outbox_files.close_verified(caller_vfds)
        if victim_fd is not None:
            os.close(victim_fd)

    # 3) Cancelling __aiter__ mid-read defers FD close until in-flight os.read exits (no EBADF or FD reuse race)
    read_entered = threading.Event()
    read_release = threading.Event()
    real_os_read = os.read
    st3 = real_open({"chat_id": "1"}, [("document", pathlib.Path(frozen["path"]))])
    target_fd = st3._file_parts[0][1]
    monkeypatch.setattr(
        transport.os,
        "read",
        lambda fd, n: (read_entered.set(), read_release.wait(5.0), real_os_read(fd, n))[2] if fd == target_fd else real_os_read(fd, n),
    )

    async def cancel_mid_read():
        async def consume():
            async for _chunk in st3:
                pass

        t = asyncio.create_task(consume())
        assert await asyncio.to_thread(read_entered.wait, 5.0)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert _fd_open(target_fd)  # still open while worker os.read is in flight
        read_release.set()

    try:
        asyncio.run(cancel_mid_read())
        deadline = time.monotonic() + 5.0
        while _fd_open(target_fd) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _fd_open(target_fd)
    finally:
        read_release.set()
        st3.close_sync()


def test_unload_from_worker_thread_cancels_cmd_sync_on_debug_loop_and_stops_presence(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_lc_un")
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
    _load_plugin("tg_lc_set")
    from tg_lc_set.lib import telegram_owner_auth as oa
    from tg_lc_set.scripts import telegram_settings as ts

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

    assert ts.load_settings(state)["TELEGRAM_CHAT_ID"] == "18"
    assert json.loads(pairing_file.read_text(encoding="utf-8"))["code"] == code


def test_pairing_group_burn_all_forms_error_propagation_surrogates_and_reissue_safety(tmp_path, monkeypatch):
    _load_plugin("tg_lc_pair")
    from tg_lc_pair.lib import telegram_owner_auth as oa
    from tg_lc_pair.scripts import telegram_settings as ts

    state = tmp_path / "state"
    for fmt in (
        "oops `{c}`",
        "/pair `{c}`",
        "/pair <{c}>",
        "/pair extra {c}",
        "here is code={c}",
        "https://t.me/mybot?start={c}",
    ):
        code = oa.issue_owner_pairing_code(state)["code"]
        msg = {"chat": {"id": -1001, "type": "supergroup"}, "text": fmt.format(c=code)}
        assert oa.is_pairing_command(state, msg) is True, fmt
        assert not (state / "owner_pairing.json").exists(), fmt
        assert oa.redeem_owner_pairing_code(state, 99, f"/pair {code}") is False

    # Lone surrogate does not raise UnicodeEncodeError
    code_s = oa.issue_owner_pairing_code(state)["code"]
    assert oa.is_pairing_command(state, {"chat": {"id": -1001, "type": "supergroup"}, "text": "bad \ud800 text"}) is False
    assert (state / "owner_pairing.json").exists()

    # Unlink failure during group burn propagates TelegramSettingsError so poller halts & retries, AND in-memory denylist blocks redeem
    real_unlink = pathlib.Path.unlink
    monkeypatch.setattr(
        pathlib.Path,
        "unlink",
        lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("denied")) if self.name == "owner_pairing.json" else real_unlink(self, *a, **k),
    )
    with pytest.raises(ts.TelegramSettingsError):
        oa.is_pairing_command(state, {"chat": {"id": -1001, "type": "supergroup"}, "text": f"/pair {code_s}"})
    assert oa.redeem_owner_pairing_code(state, 99, f"/pair {code_s}") is False
    monkeypatch.setattr(pathlib.Path, "unlink", real_unlink)

    # Atomic compare-and-burn never deletes a newer concurrently re-issued code
    code2 = oa.issue_owner_pairing_code(state)["code"]
    assert oa._burn_active_code_if_matches(state, code_s) is False
    assert (state / "owner_pairing.json").exists()
    assert oa.redeem_owner_pairing_code(state, 18, f"/pair {code2}") is True


def test_receipt_visible_to_fails_closed_and_exact_bot_command_dispatch(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_lc_misc")
    from tg_lc_misc.lib.presence import file_access
    from tg_lc_misc.scripts.telegram_settings import merge_settings

    monkeypatch.setattr(
        "ouroboros.presence_authority.presence_caller_binding",
        lambda _ctx: (_ for _ in ()).throw(RuntimeError("corrupt binding")),
    )
    ctx = types.SimpleNamespace(task_id="t1", task_metadata={})
    assert file_access.receipt_visible_to(ctx, {"account_id": "123", "chat_id": "1"}) is False

    # End-to-end owner handler dispatch for malformed mentions & multiline /language
    api = _StubApi(tmp_path / "cmd_state")
    merge_settings(tmp_path / "cmd_state", {"TELEGRAM_CHAT_ID": "18"})
    sent_plain: list[str] = []
    sent_keyboards: list[str] = []
    chosen_langs: list[str] = []
    fake_cl = types.SimpleNamespace(
        bot_id="123456",
        api_base_origin="https://api.telegram.org",
        send_message=lambda cid, txt, **k: asyncio.sleep(0, result=sent_plain.append(txt) or 1),
        send_message_with_inline_keyboard=lambda cid, txt, kb, **k: asyncio.sleep(0, result=sent_keyboards.append(txt) or 1),
    )
    monkeypatch.setattr(plugin, "_choose_language", lambda _a, req, _l, _r: asyncio.sleep(0, result=chosen_langs.append(req)))

    for malformed in ("/mode@bot/restart", "/language@bot/restart", "/menu@bot/restart", "/help@bot/panic"):
        upd = {"update_id": 1, "message": {"chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 5, "text": malformed}}
        asyncio.run(plugin._handle_owner_message_inner(api, fake_cl, upd, 1, "strict", "en"))
    assert sent_keyboards == [] and len(sent_plain) == 4

    upd_ml = {"update_id": 2, "message": {"chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 6, "text": "/language@my_bot\nQuenya"}}
    asyncio.run(plugin._handle_owner_message_inner(api, fake_cl, upd_ml, 2, "strict", "en"))
    assert sent_keyboards == [] and chosen_langs == ["Quenya"]
