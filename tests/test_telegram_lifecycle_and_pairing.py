"""Behavioral regressions for Telegram skill lifecycle, FD ownership, pairing revocation, and command matching."""

from __future__ import annotations

import asyncio
import errno
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


def test_pairing_read_eio_text_link_private_bare_and_unbound_batch_ordering(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_lc_r19")
    from tg_lc_r19.lib import telegram_owner_auth as oa, telegram_poll_binding as pb, telegram_presence_bridge as bridge
    from tg_lc_r19.lib.presence import events as pres_events
    from tg_lc_r19.scripts import telegram_settings as ts

    state = tmp_path / "state"

    # 1) EIO reading owner_pairing.json raises TelegramSettingsError (never silently False)
    code_eio = oa.issue_owner_pairing_code(state)["code"]
    real_read_text = pathlib.Path.read_text

    def fail_pairing_read(self, *a, **k):
        if self.name == "owner_pairing.json":
            raise OSError(errno.EIO, "simulated EIO")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", fail_pairing_read)
    with pytest.raises(ts.TelegramSettingsError, match="read owner pairing"):
        oa.is_pairing_command(state, {"chat": {"id": -1001, "type": "supergroup"}, "text": f"/pair {code_eio}"})
    monkeypatch.setattr(pathlib.Path, "read_text", real_read_text)
    assert (state / "owner_pairing.json").exists()

    # 2) text_link entity URL (plain & percent-encoded) burns active code in groups
    for raw_url_tpl in ("https://t.me/mybot?start={c}", "https://t.me/mybot?start=%{h}{r}"):
        code_link = oa.issue_owner_pairing_code(state)["code"]
        url = raw_url_tpl.format(c=code_link, h=f"{ord(code_link[0]):02X}", r=code_link[1:])
        msg = {
            "chat": {"id": -1001, "type": "supergroup"},
            "text": "tap to pair",
            "entities": [{"type": "text_link", "offset": 0, "length": 11, "url": url}],
        }
        assert oa.is_pairing_command(state, msg) is True
        assert not (state / "owner_pairing.json").exists()
        assert oa.redeem_owner_pairing_code(state, 99, f"/start {code_link}") is False

    # 3) Unbound private DM bare paste suppresses without entering custody, yet lets /pair redeem next
    code_priv = oa.issue_owner_pairing_code(state)["code"]
    assert oa.is_pairing_command(state, {"chat": {"id": 18, "type": "private"}, "text": code_priv}) is True
    assert (state / "owner_pairing.json").exists()
    assert oa.redeem_owner_pairing_code(state, 18, f"/pair {code_priv}") is True
    ts.merge_settings(state, {"TELEGRAM_CHAT_ID": ""})

    # 4) Unbound poll batch: presence_only_defer (3 updates) and runtime=None (2 updates) NEVER let attacker leapfrog burn
    for scenario in ("presence_defer", "runtime_none"):
        sub_state = tmp_path / f"batch_{scenario}"
        api = _StubApi(sub_state)
        code_b = oa.issue_owner_pairing_code(sub_state)["code"]
        cursor = pb.PollCursor(sub_state / "cursor.json", "https://api.telegram.org", "123456")
        binding = types.SimpleNamespace(current=lambda _g: True)
        sent: list[tuple[int, str]] = []
        cl = types.SimpleNamespace(
            bot_id="123456",
            api_base_origin="https://api.telegram.org",
            send_message=lambda cid, txt, **k: asyncio.sleep(0, result=sent.append((cid, txt)) or 1),
        )
        if scenario == "presence_defer":
            updates = [
                {"update_id": 1, "message": {"chat": {"id": -1001, "type": "supergroup"}, "from": {"id": 7}, "message_id": 1, "text": "hello"}},
                {"update_id": 2, "message": {"chat": {"id": -1001, "type": "supergroup"}, "from": {"id": 7}, "message_id": 2, "text": f"/pair {code_b}"}},
                {"update_id": 3, "message": {"chat": {"id": 99, "type": "private"}, "from": {"id": 99}, "message_id": 3, "text": f"/pair {code_b}"}},
            ]

            async def non_owner_defer(a, c, u, p):
                if u["update_id"] == 1:
                    raise pb.InjectNotAccepted("custody busy")
                await bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: None)

            non_owner_cb = non_owner_defer
        else:
            updates = [
                {"update_id": 1, "message": {"chat": {"id": -1001, "type": "supergroup"}, "from": {"id": 7}, "message_id": 1, "text": f"/pair {code_b}"}},
                {"update_id": 2, "message": {"chat": {"id": 99, "type": "private"}, "from": {"id": 99}, "message_id": 2, "text": f"/pair {code_b}"}},
            ]

            async def non_owner_none(a, c, u, p):
                await bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: None)

            non_owner_cb = non_owner_none

        asyncio.run(
            pb.handle_poll_batch(
                api, cl, updates, cursor, binding, 1, "", 10, "strict", "en",
                is_panic=lambda _u: False,
                authorize=plugin._authorized_owner,
                on_callback=lambda *a, **k: asyncio.sleep(0, result="en"),
                on_message=plugin._handle_owner_message_inner,
                on_non_owner=non_owner_cb,
            )
        )
        assert ts.load_settings(sub_state).get("TELEGRAM_CHAT_ID", "") == "", scenario
        assert sent == [], scenario
        if scenario == "runtime_none":
            assert not (sub_state / "owner_pairing.json").exists()

    # 5) Lone surrogates across ingress helpers never raise UnicodeEncodeError
    sur = "bad \ud800 /panic"
    assert isinstance(pres_events.telegram_wake_facts({"chat": {"type": "group"}, "text": sur, "entities": [{"type": "mention", "offset": 0, "length": 3}]}, "1", "mybot"), list)
    assert pb.update_key({"edited_message": {"chat": {"id": 1}, "message_id": 2, "text": sur}}).startswith("em:1:2:")
    assert oa.is_literal_slash_message({"text": sur, "entities": [{"type": "code", "offset": 0, "length": 1}]}) is False


def test_post_multipart_pre_dup_emfile_keeps_outbox_retryable_not_uncertain(tmp_path, monkeypatch):
    _load_plugin("tg_lc_dup")
    from tg_lc_dup.lib.presence import custody, outbox_files, runtime as rt_mod, transport
    from tg_lc_dup.scripts.telegram_settings import merge_settings

    state = tmp_path / "state"
    api = _StubApi(state)
    merge_settings(state, {"TELEGRAM_CHAT_ID": "18"})
    src = tmp_path / "doc.bin"
    src.write_bytes(b"EMFILE_RETRYABLE_PAYLOAD")
    frozen = outbox_files.keep(src, state, 1024)
    store = custody.CustodyStore(state / "presence_emfile.sqlite3")
    store.enqueue_outbox(
        "d-emfile",
        {
            "kind": "document",
            "account_id": "123456",
            "chat_id": "-1001",
            "topic_id": "",
            "file_path": frozen["path"],
            "_frozen": [frozen],
        },
    )

    real_dup = os.dup
    dup_calls = 0

    def fail_dup(_fd):
        nonlocal dup_calls
        dup_calls += 1
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(transport.os, "dup", fail_dup)
    client = transport.PresenceBotClient("123456:TOK_A")
    http_posts: list[str] = []
    monkeypatch.setattr(transport.httpx.AsyncClient, "post", lambda *a, **k: http_posts.append("post"))
    fake_submitter = types.SimpleNamespace(submit=lambda *a, **k: asyncio.sleep(0), poll=lambda *a, **k: asyncio.sleep(0))
    rt = rt_mod.PresenceRuntime(api=api, state_dir=state, submitter=fake_submitter, client_factory=lambda: client)
    rt.store = store
    rt._bot_id = "123456"
    rt._validation_gen = 1
    monkeypatch.setattr(rt, "_ready_client", lambda: asyncio.sleep(0, result=client))
    monkeypatch.setattr(rt, "_client_is_current", lambda _c, _g: True)
    monkeypatch.setattr(rt, "_resolved_reporting", lambda _p: asyncio.sleep(0, result={}))
    monkeypatch.setattr(rt, "_check_place", lambda *a, **k: asyncio.sleep(0))

    try:
        assert asyncio.run(rt.process_one_outbox()) is True
        receipt = store.delivery_receipt("d-emfile")
        assert dup_calls == 1 and http_posts == []
        assert receipt["state"] == "pending" and receipt["attempts"] == 1
        assert "uncertain" not in str(receipt.get("error") or "").lower()
    finally:
        monkeypatch.setattr(transport.os, "dup", real_dup)
        store.close()


def test_reply_quote_pairing_custody_bound_panic_under_bad_pairing_file_and_transient_eio_redeem(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_lc_r20")
    from tg_lc_r20.lib import telegram_owner_auth as oa, telegram_poll_binding as pb, telegram_presence_bridge as bridge
    from tg_lc_r20.lib.presence import custody, events as pres_events, runtime as rt_mod
    from tg_lc_r20.scripts import telegram_settings as ts

    # 1) Unbound private DM bare paste + reply/quote referencing it NEVER enters SQLite custody AND keeps /pair working
    s1 = tmp_path / "reply_custody"
    api1 = _StubApi(s1)
    code1 = oa.issue_owner_pairing_code(s1)["code"]
    store1 = custody.CustodyStore(s1 / "presence.sqlite3")
    rt1 = rt_mod.PresenceRuntime(api=api1, state_dir=s1, submitter=types.SimpleNamespace(submit=lambda *a, **k: asyncio.sleep(0)))
    rt1.store = store1
    rt1._bot_id = "123456"
    cursor1 = pb.PollCursor(s1 / "cursor.json", "https://api.telegram.org", "123456")
    binding1 = types.SimpleNamespace(current=lambda _g: True)
    sent1: list[tuple[int, str]] = []
    cl1 = types.SimpleNamespace(
        bot_id="123456",
        api_base_origin="https://api.telegram.org",
        send_message=lambda cid, txt, **k: asyncio.sleep(0, result=sent1.append((cid, txt)) or 1),
    )
    u_bare = {"update_id": 1, "message": {"chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 10, "date": 1700000000, "text": code1}}
    u_reply = {
        "update_id": 2,
        "message": {
            "chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 11, "date": 1700000001,
            "text": "did you get it?",
            "reply_to_message": {"message_id": 10, "text": code1},
        },
    }
    try:
        asyncio.run(
            pb.handle_poll_batch(
                api1, cl1, [u_bare, u_reply], cursor1, binding1, 1, "", 10, "strict", "en",
                is_panic=lambda _u: False, authorize=plugin._authorized_owner,
                on_callback=lambda *a, **k: asyncio.sleep(0, result="en"),
                on_message=plugin._handle_owner_message_inner,
                on_non_owner=lambda a, c, u, p: bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: rt1),
            )
        )
        with store1._connect() as conn:
            rows = conn.execute("SELECT payload_json FROM inbox").fetchall()
        assert rows == []
        assert (s1 / "owner_pairing.json").exists()
        assert oa.redeem_owner_pairing_code(s1, 18, f"/pair {code1}") is True
    finally:
        store1.close()

    # 2) Bound owner /panic succeeds even when owner_pairing.json has corrupt UTF-8, chmod 000, or is a directory
    for mode in ("bad_utf8", "chmod_000", "directory"):
        s2 = tmp_path / f"panic_{mode}"
        api2 = _StubApi(s2)
        ts.merge_settings(s2, {"TELEGRAM_CHAT_ID": "18"})
        pfile = s2 / "owner_pairing.json"
        if mode == "bad_utf8":
            pfile.write_bytes(b"\xff\xfe\x00garbage")
        elif mode == "chmod_000":
            pfile.write_text('{"code":"abc"}', encoding="utf-8")
            pfile.chmod(0)
        else:
            pfile.mkdir()
        injected: list[dict] = []
        monkeypatch.setattr(plugin, "_inject", lambda _a, payload: asyncio.sleep(0, result=injected.append(payload)))
        monkeypatch.setattr(plugin, "_POLL_GENERATION_STARTED", 1)
        cursor2 = pb.PollCursor(s2 / "cursor.json", "https://api.telegram.org", "123456")
        cl2 = types.SimpleNamespace(
            bot_id="123456",
            api_base_origin="https://api.telegram.org",
            send_message=lambda cid, txt, **k: asyncio.sleep(0, result=1),
        )
        u_panic = {"update_id": 10, "message": {"chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 20, "date": 1700000000, "text": "/panic"}}
        try:
            asyncio.run(
                pb.handle_poll_batch(
                    api2, cl2, [u_panic], cursor2, types.SimpleNamespace(current=lambda _g: True), 1, "18", 10, "strict", "en",
                    is_panic=plugin._panic_update, authorize=plugin._authorized_owner,
                    on_callback=lambda *a, **k: asyncio.sleep(0, result="en"),
                    on_message=plugin._handle_owner_message_inner,
                    on_non_owner=lambda *a, **k: asyncio.sleep(0),
                )
            )
            assert len(injected) == 1 and injected[0]["text"] == "/panic", mode
            if mode == "bad_utf8":
                # Non-panic messages also do not crash with UnicodeDecodeError
                assert oa.is_pairing_command(s2, {"chat": {"id": 18, "type": "private"}, "text": "hello"}) is False
        finally:
            if mode == "chmod_000":
                pfile.chmod(0o600)

    # 3) Single transient EIO on first read during legitimate private /pair leaves update unconfirmed & preserves challenge for retry
    s3 = tmp_path / "eio_redeem"
    api3 = _StubApi(s3)
    code3 = oa.issue_owner_pairing_code(s3)["code"]
    cursor3 = pb.PollCursor(s3 / "cursor.json", "https://api.telegram.org", "123456")
    sent3: list[tuple[int, str]] = []
    cl3 = types.SimpleNamespace(
        bot_id="123456",
        api_base_origin="https://api.telegram.org",
        send_message=lambda cid, txt, **k: asyncio.sleep(0, result=sent3.append((cid, txt)) or 1),
    )
    u_pair = {"update_id": 4, "message": {"chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 40, "date": 1700000000, "text": f"/pair {code3}"}}
    real_read_text = pathlib.Path.read_text
    failed_once = {"done": False}

    def eio_once(self, *a, **k):
        if self.name == "owner_pairing.json" and not failed_once["done"]:
            failed_once["done"] = True
            raise OSError(errno.EIO, "transient read error")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", eio_once)
    with pytest.raises(ts.TelegramSettingsError):
        asyncio.run(
            pb.handle_poll_batch(
                api3, cl3, [u_pair], cursor3, types.SimpleNamespace(current=lambda _g: True), 1, "", 10, "strict", "en",
                is_panic=lambda _u: False, authorize=plugin._authorized_owner,
                on_callback=lambda *a, **k: asyncio.sleep(0, result="en"),
                on_message=plugin._handle_owner_message_inner,
                on_non_owner=lambda a, c, u, p: bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: None),
            )
        )
    assert cursor3.offset == 0 and (s3 / "owner_pairing.json").exists()
    assert ts.load_settings(s3).get("TELEGRAM_CHAT_ID", "") == ""
    # Retry succeeds and binds owner 18 cleanly
    asyncio.run(
        pb.handle_poll_batch(
            api3, cl3, [u_pair], cursor3, types.SimpleNamespace(current=lambda _g: True), 1, "", 10, "strict", "en",
            is_panic=lambda _u: False, authorize=plugin._authorized_owner,
            on_callback=lambda *a, **k: asyncio.sleep(0, result="en"),
            on_message=plugin._handle_owner_message_inner,
            on_non_owner=lambda a, c, u, p: bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: None),
        )
    )
    assert ts.load_settings(s3).get("TELEGRAM_CHAT_ID") == "18" and cursor3.offset == 5

    # 4) Custody SQLite accepts lone-surrogate text cleanly without UnicodeEncodeError
    s4 = tmp_path / "sur_custody"
    store4 = custody.CustodyStore(s4 / "presence.sqlite3")
    try:
        ev = pres_events.parse_telegram_update(
            {"update_id": 7, "message": {"chat": {"id": -1001, "type": "supergroup", "title": "grp"}, "from": {"id": 5, "first_name": "A\ud800"}, "message_id": 7, "date": 1700000000, "text": "hi \ud800 there"}},
            bot_account_id="123456", bot_username="mybot", download_limit=1024,
        )
        assert ev is not None
        assert store4.commit_update(7, ev, []) is True
    finally:
        store4.close()


def test_async_client_emfile_and_html_fallback_before_send_oserror_stay_retryable(tmp_path, monkeypatch):
    _load_plugin("tg_lc_cli")
    from tg_lc_cli.lib.presence import custody, outbox_files, runtime as rt_mod, transport
    from tg_lc_cli.scripts.telegram_settings import merge_settings

    # 1) Text message send where httpx.AsyncClient() hits EMFILE stays retryable (pending, NOT uncertain)
    state = tmp_path / "state"
    api = _StubApi(state)
    merge_settings(state, {"TELEGRAM_CHAT_ID": "18"})
    store = custody.CustodyStore(state / "presence_msg_emfile.sqlite3")
    store.enqueue_outbox(
        "d-msg-emfile",
        {"kind": "message", "account_id": "123456", "chat_id": "-1001", "topic_id": "", "text": "hello"},
    )
    monkeypatch.setattr(transport.httpx, "AsyncClient", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EMFILE, "Too many open files")))
    client = transport.PresenceBotClient("123456:TOK_A")
    fake_submitter = types.SimpleNamespace(submit=lambda *a, **k: asyncio.sleep(0), poll=lambda *a, **k: asyncio.sleep(0))
    rt = rt_mod.PresenceRuntime(api=api, state_dir=state, submitter=fake_submitter, client_factory=lambda: client)
    rt.store = store
    rt._bot_id = "123456"
    rt._validation_gen = 1
    monkeypatch.setattr(rt, "_ready_client", lambda: asyncio.sleep(0, result=client))
    monkeypatch.setattr(rt, "_client_is_current", lambda _c, _g: True)
    monkeypatch.setattr(rt, "_resolved_reporting", lambda _p: asyncio.sleep(0, result={}))
    monkeypatch.setattr(rt, "_check_place", lambda *a, **k: asyncio.sleep(0))

    try:
        assert asyncio.run(rt.process_one_outbox()) is True
        receipt = store.delivery_receipt("d-msg-emfile")
        assert receipt["state"] == "pending" and receipt["attempts"] == 1
        assert "uncertain" not in str(receipt.get("error") or "").lower()
    finally:
        store.close()

    # 2) HTML-400 fallback before_send() ValueError/OSError maps to TelegramRequestNotSent (retryable, not uncertain)
    src = tmp_path / "doc.bin"
    src.write_bytes(b"HTML_400_FALLBACK")
    frozen = outbox_files.keep(src, state, 1024)
    cl2 = transport.PresenceBotClient("123456:TOK_A")
    monkeypatch.setattr(
        cl2,
        "_post_multipart",
        lambda *a, **k: (_ for _ in ()).throw(transport.TelegramApiError("sendDocument", "Bad Request: can't parse entities", 400)),
    )

    async def fail_before():
        raise ValueError("the copy of doc.bin kept when it was queued cannot be read (OSError); not sent")

    with pytest.raises(transport.TelegramRequestNotSent):
        asyncio.run(
            cl2.send_media(
                "document", "-1001", pathlib.Path(frozen["path"]),
                prepared_caption={"text": "<b>hi</b>", "parse_mode": "HTML"},
                before_send=fail_before,
            )
        )

    # 3) send_text_chunk HTML-400 fallback before_send() OSError also translates to TelegramRequestNotSent
    monkeypatch.setattr(
        cl2,
        "_post_json",
        lambda *a, **k: (_ for _ in ()).throw(transport.TelegramApiError("sendMessage", "Bad Request: can't parse entities", 400)),
    )

    async def fail_before_os():
        raise OSError(errno.EMFILE, "Too many open files")

    with pytest.raises(transport.TelegramRequestNotSent):
        asyncio.run(cl2.send_text_chunk("-1001", "<b>hi</b>", parse_mode="HTML", before_send=fail_before_os))


def test_pairing_settings_replace_eio_recovery_max_attempts_durability_and_universal_carriers(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_lc_r21p")
    from tg_lc_r21p.lib import telegram_owner_auth as oa, telegram_poll_binding as pb, telegram_presence_bridge as bridge
    from tg_lc_r21p.scripts import telegram_settings as ts

    # 1) Transient EIO on settings.json os.replace during valid private /pair recovers on exact-chat replay, refuses stranger
    s1 = tmp_path / "settings_eio"
    api1 = _StubApi(s1)
    code1 = oa.issue_owner_pairing_code(s1)["code"]
    cursor1 = pb.PollCursor(s1 / "cursor.json", "https://api.telegram.org", "123456")
    sent1: list[tuple[int, str]] = []
    cl1 = types.SimpleNamespace(
        bot_id="123456", api_base_origin="https://api.telegram.org",
        send_message=lambda cid, txt, **k: asyncio.sleep(0, result=sent1.append((cid, txt)) or 1),
    )
    u_pair = {"update_id": 12, "message": {"chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 120, "date": 1700000000, "text": f"/pair {code1}"}}
    real_replace = os.replace
    failed_once = {"done": False}

    def fail_settings_replace(src, dst):
        if pathlib.Path(dst).name == "settings.json" and not failed_once["done"]:
            failed_once["done"] = True
            raise OSError(errno.EIO, "transient settings replace error")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_settings_replace)
    with pytest.raises(ts.TelegramSettingsError):
        asyncio.run(
            pb.handle_poll_batch(
                api1, cl1, [u_pair], cursor1, types.SimpleNamespace(current=lambda _g: True), 1, "", 10, "strict", "en",
                is_panic=lambda _u: False, authorize=plugin._authorized_owner,
                on_callback=lambda *a, **k: asyncio.sleep(0, result="en"),
                on_message=plugin._handle_owner_message_inner,
                on_non_owner=lambda a, c, u, p: bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: None),
            )
        )
    assert cursor1.offset == 0 and ts.load_settings(s1).get("TELEGRAM_CHAT_ID", "") == ""
    assert oa.redeem_owner_pairing_code(s1, 99, f"/pair {code1}") is False
    asyncio.run(
        pb.handle_poll_batch(
            api1, cl1, [u_pair], cursor1, types.SimpleNamespace(current=lambda _g: True), 1, "", 10, "strict", "en",
            is_panic=lambda _u: False, authorize=plugin._authorized_owner,
            on_callback=lambda *a, **k: asyncio.sleep(0, result="en"),
            on_message=plugin._handle_owner_message_inner,
            on_non_owner=lambda a, c, u, p: bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: None),
        )
    )
    assert ts.load_settings(s1).get("TELEGRAM_CHAT_ID") == "18" and cursor1.offset == 13

    # 2) 5th bad attempt persists terminal failed_attempts on disk even if unlink fails
    s2 = tmp_path / "max_attempts_eio"
    code2 = oa.issue_owner_pairing_code(s2)["code"]
    for _ in range(4):
        assert oa.redeem_owner_pairing_code(s2, 18, "/pair WRONG") is False
    real_unlink = pathlib.Path.unlink
    monkeypatch.setattr(
        pathlib.Path, "unlink",
        lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("denied")) if self.name == "owner_pairing.json" else real_unlink(self, *a, **k),
    )
    assert oa.redeem_owner_pairing_code(s2, 18, "/pair WRONG") is False
    tombstone = json.loads((s2 / "owner_pairing.json").read_text(encoding="utf-8"))
    assert tombstone["failed_attempts"] == 5 and "code" not in tombstone
    oa._BURNED_HASHES.clear()
    oa._BURNED_ORDER.clear()
    assert oa.redeem_owner_pairing_code(s2, 18, f"/pair {code2}") is False

    # 2b) When BOTH unlink and write fail on 5th bad attempt AND its retry, cursor stays unchanged until storage recovers
    s2b = tmp_path / "max_attempts_retry_io"
    api2b = _StubApi(s2b)
    oa.issue_owner_pairing_code(s2b)
    monkeypatch.setattr(pathlib.Path, "unlink", real_unlink)
    for _ in range(4):
        assert oa.redeem_owner_pairing_code(s2b, 18, "/pair WRONG") is False
    cur2b = pb.PollCursor(s2b / "cursor.json", "https://api.telegram.org", "123456")
    u_bad = {"update_id": 20, "message": {"chat": {"id": 18, "type": "private"}, "from": {"id": 18}, "message_id": 200, "date": 1700000000, "text": "/pair WRONG"}}
    real_wpj = oa._write_private_json
    monkeypatch.setattr(pathlib.Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("denied")) if self.name == "owner_pairing.json" else real_unlink(self, *a, **k))
    monkeypatch.setattr(oa, "_write_private_json", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "write denied")))
    for _ in range(2):
        with pytest.raises(ts.TelegramSettingsError):
            asyncio.run(pb.handle_poll_batch(api2b, cl1, [u_bad], cur2b, types.SimpleNamespace(current=lambda _g: True), 1, "", 10, "strict", "en", is_panic=lambda _u: False, authorize=plugin._authorized_owner, on_callback=lambda *a, **k: asyncio.sleep(0, result="en"), on_message=plugin._handle_owner_message_inner, on_non_owner=lambda a, c, u, p: bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: None)))
        assert cur2b.offset == 0
    monkeypatch.setattr(pathlib.Path, "unlink", real_unlink)
    monkeypatch.setattr(oa, "_write_private_json", real_wpj)
    asyncio.run(pb.handle_poll_batch(api2b, cl1, [u_bad], cur2b, types.SimpleNamespace(current=lambda _g: True), 1, "", 10, "strict", "en", is_panic=lambda _u: False, authorize=plugin._authorized_owner, on_callback=lambda *a, **k: asyncio.sleep(0, result="en"), on_message=plugin._handle_owner_message_inner, on_non_owner=lambda a, c, u, p: bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: None)))
    assert cur2b.offset == 21 and not (s2b / "owner_pairing.json").exists()

    # 3) Double-encoded link_preview_options.url, poll.question, and document.file_name burn active codes in groups
    s3 = tmp_path / "carriers"
    for carrier_fn in (
        lambda c: {"chat": {"id": -1001, "type": "supergroup"}, "text": "preview", "link_preview_options": {"url": f"https://t.me/bot?start=%25{ord(c[0]):02X}{c[1:]}"}},
        lambda c: {"chat": {"id": -1001, "type": "supergroup"}, "poll": {"question": f"code {c}", "options": [{"text": "yes"}]}},
        lambda c: {"chat": {"id": -1001, "type": "supergroup"}, "document": {"file_name": f"{c}.pdf"}},
    ):
        c = oa.issue_owner_pairing_code(s3)["code"]
        assert oa.is_pairing_command(s3, carrier_fn(c)) is True
        assert not (s3 / "owner_pairing.json").exists()
        assert oa.redeem_owner_pairing_code(s3, 99, f"/pair {c}") is False


def test_lone_surrogates_end_to_end_poll_sightings_custody_host_fifo_and_outbox(tmp_path, monkeypatch):
    _load_plugin("tg_lc_r21s")
    from tg_lc_r21s.lib import telegram_poll_binding as pb, telegram_presence_bridge as bridge
    from tg_lc_r21s.lib.presence import custody, file_access, host, outbox_worker, runtime as rt_mod, transport

    state = tmp_path / "sur_e2e"
    api = _StubApi(state)
    store = custody.CustodyStore(state / "presence.sqlite3")
    submitted_bodies: list[bytes] = []

    class _FakeResp:
        status_code = 200
        headers: dict[str, str] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def aiter_bytes(self):
            yield b'{"status":"completed","outcome":"silent","turn_ref":"t-1"}'

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, *, headers=None, content=None):
            if content is not None:
                submitted_bodies.append(content)
            return _FakeResp()

    monkeypatch.setattr(host.httpx, "AsyncClient", _FakeClient)
    host_client = host.PresenceHostClient(
        state_dir=state,
        token_provider=lambda: "tok",
        host_base="http://127.0.0.1:8765",
        transport=host.UrllibPresenceHostTransport(),
    )
    host_client.delivery_reporting_status = "unsupported"
    host_client._identity_discovered_at = time.monotonic()
    monkeypatch.setattr(host, "load_binding_id", lambda _s: "b-1")
    monkeypatch.setattr(host_client, "bind_place", None)
    rt = rt_mod.PresenceRuntime(api=api, state_dir=state, submitter=host_client)
    rt.store = store
    rt._bot_id = "123456"
    rt._bot_username = "mybot"
    cursor = pb.PollCursor(state / "cursor.json", "https://api.telegram.org", "123456")
    cl = types.SimpleNamespace(bot_id="123456", api_base_origin="https://api.telegram.org")
    updates = [
        {"update_id": 10, "message": {"chat": {"id": -1001, "type": "supergroup", "title": "Team \ud800"}, "from": {"id": 5, "first_name": "A\ud800"}, "message_id": 10, "date": 1700000000, "media_group_id": "mg-\ud800", "text": "hi \ud800 there"}},
        {"update_id": 11, "message": {"chat": {"id": -1001, "type": "supergroup", "title": "Team"}, "from": {"id": 5, "first_name": "A"}, "message_id": 11, "date": 1700000001, "text": "clean follow-up"}},
        {"update_id": 12, "message": {"chat": {"id": 42, "type": "private", "first_name": "Bob\ud800"}, "from": {"id": 42, "first_name": "Bob\ud800"}, "message_id": 12, "date": 1700000002, "text": "dm \ud83d"}},
    ]
    try:
        asyncio.run(
            pb.handle_poll_batch(
                api, cl, updates, cursor, types.SimpleNamespace(current=lambda _g: True), 1, "18", 10, "strict", "en",
                is_panic=lambda _u: False, authorize=lambda *a, **k: asyncio.sleep(0, result=None),
                on_callback=lambda *a, **k: asyncio.sleep(0, result="en"),
                on_message=lambda *a, **k: asyncio.sleep(0),
                on_non_owner=lambda a, c, u, p: bridge.handle_non_owner_update(a, c, u, p, runtime_getter=lambda _a: rt),
            )
        )
        assert cursor.offset == 13
        with store._connect() as conn:
            inbox_rows = conn.execute("SELECT payload_json FROM inbox ORDER BY update_id").fetchall()
            group_rows = conn.execute("SELECT chat_id, title FROM telegram_groups").fetchall()
        assert len(inbox_rows) == 3 and len(group_rows) == 2
        for (raw_json,) in inbox_rows:
            json.dumps(json.loads(raw_json), ensure_ascii=False).encode("utf-8")

        # Also verify legacy DB row with escaped \ud800 submits cleanly through UrllibPresenceHostTransport without deadlocking FIFO
        with store._connect() as conn:
            cur = conn.execute(
                "UPDATE inbox SET payload_json = replace(payload_json, ?, ?) WHERE update_id = 10",
                ("hi ? there", "hi \\ud800 there"),
            )
            assert cur.rowcount == 1
            conn.commit()
            assert "\\ud800" in conn.execute("SELECT payload_json FROM inbox WHERE update_id = 10").fetchone()[0]
        for i in range(3):
            assert asyncio.run(rt.process_one_inbox()) is True, f"iter={i} err={rt._last_error!r} bodies={len(submitted_bodies)}"
        assert len(submitted_bodies) == 3
        decoded = [json.loads(b.decode("utf-8")) for b in submitted_bodies]
        assert any("clean follow-up" in str(d) for d in decoded)

        # Outbox own_message_row, idempotency_fingerprint, and transport._post_json all handle lone surrogates cleanly
        row = outbox_worker.own_message_row("123456", "-1001", "", {"message_id": 99, "text": "sent \ud800", "media_group_id": "mg-\ud800"}, "")
        assert row is not None and "\ud800" not in row[8] and "\ud800" not in row[9]
        assert len(file_access.idempotency_fingerprint({"kind": "message", "text": "x \ud800"})) == 64
        sent_json: list[dict] = []

        class _CaptureClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, _url, *, json=None, **k):
                sent_json.append(json)
                return types.SimpleNamespace(status_code=200, json=lambda: {"ok": True, "result": {"message_id": 1}}, text="")

        tr_cl = transport.PresenceBotClient("123456:TOK_A")
        monkeypatch.setattr(tr_cl, "_new_async_client", lambda *_a, **_k: _CaptureClient())
        asyncio.run(tr_cl._post_json("sendMessage", {"chat_id": "-1001", "text": "out \ud800"}))
        assert sent_json == [{"chat_id": "-1001", "text": "out ?"}]
        # Surrogate error text in format_error and CustodyStore last_error binds never raises UnicodeEncodeError
        assert "\ud800" not in outbox_worker.format_error(RuntimeError("Bad \ud800 thing"))
        store.enqueue_outbox("d-sur-err", {"kind": "message", "account_id": "123456", "chat_id": "-1001", "text": "hi"})
        lease = store.claim_outbox()
        assert lease is not None and store.release_outbox(lease.delivery_id, reason="Bad \ud800 err", retry_after_sec=0, attempts=lease.attempts, lease_token=lease.lease_token) is True
    finally:
        store.close()
