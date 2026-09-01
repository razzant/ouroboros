"""Regression checks for restart/reconnect client behavior."""

import os
import pathlib

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_ws_has_error_handler_and_reconnect_timer():
    source = _read("web/modules/ws.js")
    assert "socket.onerror" in source
    assert "_scheduleReconnect" in source
    assert "_scheduleUiRecovery" in source
    assert "_uiRecoveryTimer" in source
    assert "_watchdogTimer" in source
    assert "_lastMessageAt" in source
    assert "_startWatchdog" in source
    assert "window.location.replace" in source
    assert "location.reload()" not in source


def test_ws_queues_outbound_messages_when_disconnected():
    source = _read("web/modules/ws.js")
    assert "_pendingMessages" in source
    assert "status: 'queued'" in source
    assert "outbound_sent" in source


def test_chat_marks_pending_messages_until_reconnect():
    source = _read("web/modules/chat.js")
    assert "pendingUserBubbles" in source
    assert "Queued until reconnect" in source
    assert "result?.status === 'queued'" in source


def test_chat_resyncs_history_after_reconnect():
    source = _read("web/modules/chat.js")
    assert "async function syncHistory" in source
    # perf2 P3: the default history request sends NO quota params — the server's
    # window constants govern; the dead `?limit=1000` placebo is gone.
    assert "`/api/chat/history${isMain ? '' : `?chat_id=${chatId}`}`" in source
    assert "limit=1000" not in source
    assert "cache: 'no-store'" in source
    assert "syncHistory({ includeUser: !historyLoaded, fromReconnect: isReconnect })" in source
    assert "const expectedDisconnect = socketState !== WebSocket.OPEN" in source
    assert "if (expectedDisconnect && err instanceof TypeError)" in source


def test_final_answer_marker_uses_ordinary_message_presentation():
    source = _read("web/modules/chat.js")
    css = _read("web/style.css")

    assert "renderAssistantWithAnswerChip" not in source
    assert "final-answer-chip" not in css
    assert ": renderChatMarkdown(text);" in source
    # Both live and history assistant/system paths feed the same addMessage renderer;
    # reconnect notices do too, so marker-shaped text never gets a separate capsule.
    assert "addMessage(msg.content, msg.role" in source
    assert "addMessage(msg.text, msg.role" in source
    assert "systemType: 'reconnect'" in source


def test_server_enables_ws_ping_and_heartbeat():
    server_source = _read("server.py")
    helper_source = _read("ouroboros/server_runtime.py")
    assert "ws_heartbeat_loop" in server_source or "ws_heartbeat_loop" in helper_source
    assert '"type": "heartbeat"' in server_source or '"type": "heartbeat"' in helper_source
    assert "ws_ping_interval=20" in server_source
    assert "ws_ping_timeout=20" in server_source


def test_index_includes_reconnect_overlay():
    source = _read("web/index.html")
    assert 'id="reconnect-overlay"' in source


def test_index_page_disables_cache():
    server_source = _read("server.py")
    helper_source = _read("ouroboros/server_web.py")
    assert "cache-control" in server_source.lower() or "cache-control" in helper_source.lower()


def test_find_free_port_waits_for_preferred_port():
    """find_free_port should retry the preferred port before falling back."""
    import socket
    import threading
    from ouroboros.server_entrypoint import find_free_port

    # Grab any available port for isolation
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        preferred = s.getsockname()[1]

    # Block the preferred port with a listener that releases after a short delay
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", preferred))
    blocker.listen(1)

    def _release():
        import time
        time.sleep(1.0)
        blocker.close()

    threading.Thread(target=_release, daemon=True).start()

    # With wait_retries=6 × 0.5s = 3s budget, it should get the preferred port
    result = find_free_port("127.0.0.1", preferred, wait_retries=6, wait_interval=0.5)
    assert result == preferred, f"Expected preferred port {preferred}, got {result}"


def test_find_free_port_falls_back_when_stuck():
    """find_free_port should fall back to a nearby port if preferred stays busy.

    On Windows, SO_REUSEADDR allows binding to a port already in LISTEN state,
    so the real-socket blocking approach does not reliably prevent rebind.
    We use monkeypatching to guarantee the preferred port stays unavailable.
    """
    from ouroboros import server_entrypoint

    preferred = 51234  # arbitrary high port unlikely to collide

    _original = getattr(server_entrypoint, "_can_bind_port", None)

    def _fake_can_bind(host: str, port: int) -> bool:
        if port == preferred:
            return False  # always blocked
        # For fallback ports, allow the first one
        return True

    server_entrypoint._can_bind_port = _fake_can_bind
    try:
        result = server_entrypoint.find_free_port(
            "127.0.0.1", preferred, max_tries=10,
            wait_retries=2, wait_interval=0.05,
        )
        assert result != preferred, f"Should have fallen back, but got preferred {preferred}"
    finally:
        if _original is not None:
            server_entrypoint._can_bind_port = _original


def test_find_free_port_retries_fallback_range_until_port_frees(monkeypatch):
    """Fallback scanning should keep retrying instead of returning the busy preferred port."""
    from ouroboros import server_entrypoint

    preferred = 41000
    attempts = {preferred: 0, preferred + 1: 0, preferred + 2: 0, preferred + 3: 0}

    def fake_can_bind(_host: str, port: int) -> bool:
        attempts[port] += 1
        if port == preferred:
            return False
        if port == preferred + 1:
            return attempts[port] >= 3
        return False

    monkeypatch.setattr(server_entrypoint, "_can_bind_port", fake_can_bind)

    result = server_entrypoint.find_free_port(
        "127.0.0.1",
        preferred,
        max_tries=4,
        wait_retries=3,
        wait_interval=0,
    )

    assert result == preferred + 1
    assert attempts[preferred] == 3
    assert attempts[preferred + 1] == 3


def test_find_free_port_raises_when_range_stays_busy(monkeypatch):
    """find_free_port should fail clearly instead of returning a known-busy port."""
    from ouroboros import server_entrypoint

    monkeypatch.setattr(server_entrypoint, "_can_bind_port", lambda _host, _port: False)

    try:
        server_entrypoint.find_free_port(
            "127.0.0.1",
            42000,
            max_tries=3,
            wait_retries=2,
            wait_interval=0,
        )
    except OSError as exc:
        assert "42000-42002" in str(exc)
    else:
        raise AssertionError("Expected find_free_port to raise when every candidate stays busy")


def test_chat_shows_reconnect_banner_after_reconnect_and_reload():
    """chat.js should show reconnect status after both soft reconnect and restart reload."""
    source = _read("web/modules/chat.js")
    assert "wsHasConnectedOnce" in source, "Missing reconnect-once tracking flag"
    assert "Reconnected" in source, "Missing reconnect banner text"
    # reconnectBannerText moved verbatim to chat_activity.js (chat.js byte ceiling).
    assert "Restart complete" in _read("web/modules/chat_activity.js"), (
        "Missing restart-complete banner text"
    )
    assert "_ouro_reason" in source, "Missing restart reload reason handling"
    assert "history.replaceState" in source, "Reconnect params should be cleared after showing banner"
    # Ensure banner is ephemeral (not persisted to history)
    assert "ephemeral: true" in source


def test_progress_bubbles_have_subdued_styling():
    """Progress/reasoning bubbles should be visually muted compared to regular ones."""
    css = _read("web/style.css")
    assert ".chat-bubble.progress .message" in css, "Missing progress-specific message style"
    progress_block = css.split(".chat-bubble.progress .message", 1)[1]
    progress_block = progress_block[: progress_block.index("}")]
    assert "font-size: var(--type-body)" in progress_block, (
        "Progress font should be the 14px body token, smaller than the "
        "16px --type-section chat default"
    )


def test_working_live_cards_are_subdued_and_expandable():
    """Visible working/thinking cards should stay compact but support expanding per block."""
    css = _read("web/style.css")
    assert ".chat-live-title" in css, "Missing live-card title styling"
    assert ".chat-live-line-toggle" in css, "Missing clickable live-line toggle"
    assert ".chat-live-line-expand-label" in css, "Missing expand/collapse label"
    assert '.chat-live-line[data-expanded="1"] .chat-live-line-body' in css


def test_live_card_blocks_can_expand_to_full_text():
    """chat.js should preserve expansion state and render per-block toggles."""
    source = _read("web/modules/chat.js")
    assert "expandedLineKeys" in source, "Missing per-line expansion state"
    assert "data-live-line-toggle" in source, "Missing per-block toggle markup"
    assert "fullHeadline" in source, "Missing full headline preservation"
    assert "fullBody" in source, "Missing full body preservation"


def test_live_event_summaries_preserve_full_text_for_expansion():
    """log event summaries should keep full text so expanded live blocks can reveal it."""
    source = _read("web/modules/log_events.js")
    assert "describeText" in source
    assert "fullHeadline" in source
    assert "fullBody" in source


def test_task_done_live_summary_distinguishes_typed_failure():
    source = _read("web/modules/log_events.js")
    assert "export function taskOutcomeSeverity" in source
    assert "export function taskPresentation" in source
    assert "outcome_axes?.execution?.status" in source
    assert "['degraded', 'best_effort'].includes(execution)" in source
    assert "['degraded', 'best_effort'].includes(objective)" in source
    assert "const severity = taskOutcomeSeverity(evt);" in source
    # v6.82 (P5): the severity -> card-phase mapping is ONE shared reducer, so
    # live task_done, history task_summary rows and the terminal replay
    # fallback cannot drift (a cancelled root must never render as done).
    assert "export function taskTerminalPhase" in source
    assert "if (severity === 'cancelled') return 'cancelled';" in source
    assert "if (severity === 'error') return 'error';" in source
    assert "if (severity === 'warn') return 'warn';" in source
    assert source.count("taskTerminalPhase(evt)") >= 2
    assert "const terminal = taskDoneIsTerminal(evt);" in source
    assert "const presentation = taskPresentation(terminal ? taskTerminalPhase(evt) : 'working');" in source
    assert "headline: presentation.headline" in source


def test_chat_warning_task_summaries_force_visible_cards():
    source = _read("web/modules/chat.js")
    assert "summary.terminal && summary.phase === 'warn'" in source
    assert (
        "const needsVisibleTerminal = severity === 'error' || severity === 'warn'"
        " || severity === 'cancelled';"
    ) in source


def test_chat_scrolls_to_bottom_after_first_history_load():
    """syncHistory must scroll to bottom on first load (restart/open) but
    respect user scroll position on subsequent reconnect syncs."""
    source = _read("web/modules/chat.js")
    media_source = _read("web/modules/chat_media.js")
    # First-load guard: wasFirstLoad captures pre-call state
    assert "wasFirstLoad = !historyLoaded" in source, \
        "Missing first-load detection before setting historyLoaded"
    # Conditional scroll: first load always scrolls, reconnect only when near bottom
    assert "if (wasFirstLoad || (fromReconnect ? scrollBeforeSync.nearBottom : isNearBottom()))" in source, \
        "Missing conditional scroll-to-bottom after history sync"
    assert "anchor: captureVisibleTimelineAnchor()" in source
    assert "restoreVisibleTimelineAnchor(scrollBeforeSync.anchor)" in source, \
        "Reconnect must restore a visible DOM anchor, not apply total height growth"
    # The anchor pair lives in chat_render_batch.js (extracted verbatim from
    # chat.js at the byte ratchet); the identity contract is unchanged.
    anchor_source = _read("web/modules/chat_render_batch.js")
    assert "taskId: card.dataset?.taskId || ''" in anchor_source, \
        "Nested viewport anchors must retain stable task identity"
    assert "createTimelineAnchors({ messagesDiv, liveCardRecords })" in source, \
        "The anchors factory must receive the live-card registry it reads"
    assert "liveCardRecords.get(entry.taskId)" in anchor_source, \
        "A rebuilt live card whose earliest timestamp changed needs canonical task lookup"
    assert "reorderExisting: anchorMovedEarlier" in source, \
        "A mounted task card must be re-sorted if a later event lowers its anchor"
    assert "record._anchorOrderDirty = true;" in source
    assert "reorderDirtyCardIfNeeded(rec);" in source, \
        "Pass 2 must re-sort connected cards dirtied by routine history sync"
    assert "const parent = liveCardRecords.get(record.parentGroupId);" in source
    assert "seen.has(record.groupId)" in source, \
        "Nested subagent timestamps must propagate to the top-level ancestor safely"
    assert "if (wasFirstLoad) scrollToBottomAfterLayout();" in source, \
        "First load must pin the fresh feed to the newest message explicitly"
    # The shared photo/video builder and the separate document builder must
    # each stamp sortable data-ts from the raw source timestamp.
    bubble_frame = media_source.split("function bubbleFrame", 1)[1].split(
        "function mediaSource", 1
    )[0]
    media_builder = media_source.split("function buildMediaBubble", 1)[1].split(
        "function buildDocumentBubble", 1
    )[0]
    document_builder = media_source.split("function buildDocumentBubble", 1)[1].split(
        "function buildLinksMessage", 1
    )[0]
    assert "const rawTs = msg.ts ||" in bubble_frame
    assert "stampNodeTimestamp(bubble, rawTs);" in bubble_frame
    assert "bubbleFrame(msg" in media_builder, \
        "shared photo/video bubbles must use the raw-timestamp-stamped frame"
    assert "bubbleFrame(msg" in document_builder, \
        "document bubbles must use the raw-timestamp-stamped frame"
    assert "stampNodeTimestamp(bubble, ts);" in source, \
        "chat text bubbles must carry raw-timestamp data-ts"


def test_restart_watchdog_waits_for_uvicorn_exit():
    source = _read("server.py")
    assert "_uvicorn_exited = threading.Event()" in source
    assert "_uvicorn_exited.wait(timeout=force_exit_timeout_sec)" in source
    assert "_uvicorn_exited.set()" in source


def test_owner_restart_copy_is_explicit_about_stopped_task():
    source = _read("server.py")
    assert 'ctx.send_with_budget(chat_id, "♻️ Restarting.")' in source
    assert "Stopping active task. New settings apply to the next message." in source
    assert "owner_restart_no_resume.flag" in source
    assert "owner_restart_no_resume" in source
    assert "panic_stop.flag" in source
    assert "owner_restart_flag.unlink(missing_ok=True)" in source
    assert "stable_skip_flag.unlink(missing_ok=True)" in source
    assert source.index("_safe_restart_serialized(") < source.index("Stopping active task. New settings apply to the next message.")
    assert source.index("owner_restart_no_resume.flag") < source.index("Stopping active task. New settings apply to the next message.")


def test_auto_resume_skips_owner_restart_no_resume_flag(tmp_path, monkeypatch):
    import supervisor.workers as workers

    original_drive = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    flag = tmp_path / "state" / "owner_restart_no_resume.flag"
    flag.parent.mkdir(parents=True)
    flag.write_text("owner_restart", encoding="utf-8")
    compat_flag = tmp_path / "state" / "panic_stop.flag"
    compat_flag.write_text("owner_restart_no_resume", encoding="utf-8")

    def fail_load_state():
        raise AssertionError("owner restart flag should be checked before load_state")

    monkeypatch.setattr(workers, "load_state", fail_load_state)
    try:
        workers.auto_resume_after_restart()
    finally:
        workers.DRIVE_ROOT = original_drive

    assert not flag.exists()
    assert not compat_flag.exists()


def test_owner_restart_cleans_flags_when_worker_shutdown_fails(tmp_path, monkeypatch):
    import server
    import supervisor.message_bus as message_bus

    messages = []

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": offset,
                "message": {
                    "chat": {"id": 1},
                    "from": {"id": 1},
                    "text": "/restart",
                },
            }]

    class Ctx:
        consciousness = None

        def load_state(self):
            return {}

        def save_state(self, _state):
            return None

        def update_state(self, mutator):
            live = {}
            mutator(live)
            return live

        def send_with_budget(self, _chat_id, text):
            messages.append(text)

        def safe_restart(self, **_kwargs):
            return True, "OK"

        def kill_workers(self, force=True, **kwargs):
            assert kwargs["terminal_status"] == "cancelled"
            assert kwargs["result_reason"] == "Owner restart stopped this task before process restart."
            raise RuntimeError("shutdown failed")

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        server,
        "_request_restart_exit",
        lambda: (_ for _ in ()).throw(AssertionError("restart exit should not be requested")),
    )

    server._process_bridge_updates(Bridge(), 0, Ctx())

    assert not (tmp_path / "state" / "owner_restart_no_resume.flag").exists()
    assert not (tmp_path / "state" / "panic_stop.flag").exists()
    assert "Stopping active task. New settings apply to the next message." not in messages
    assert "⚠️ Restart cancelled: failed to stop workers." in messages


def test_ws_sha_reload_decision_is_single_sourced():
    """Both the post-open refresh and the recovery probe must route through the
    one exported SHA decision helper (keep / reload_changed / reload_unknown).

    The behavioral contract — same/changed/missing/empty/malformed SHA, the
    first-ever connection, the probe-vs-open race bail, the healthy-probe fuse,
    and outbound-queue survival — is executable in
    web/tests/ws_recovery.test.js (node --test). These are wiring pins only.
    """
    source = _read("web/modules/ws.js")
    assert "export function decide(" in source
    assert "RECOVERY_HEALTHY_PROBE_LIMIT = 4" in source
    recovery_body = source.split("_scheduleUiRecovery(reason, delay = 15000) {", 1)[1].split(
        "_startWatchdog(socket) {", 1
    )[0]
    assert "_applyShaDecision(servedSha, this._wasConnected, false)" in recovery_body, (
        "the recovery probe must consult the shared SHA decision WITHOUT "
        "remembering the served SHA (a pre-connect probe adopting a restarted "
        "server's SHA would leave stale assets unhealed)"
    )
    assert "this._uiRecoveryProbeInFlight" in recovery_body, (
        "arming recovery must be gated on the in-flight probe flag so hung "
        "probes cannot pile up and multi-count the healthy fuse"
    )
    refresh_body = source.split("_refreshStateAfterOpen(previouslyConnected) {", 1)[1].split(
        "_flushPendingMessages() {", 1
    )[0]
    assert "_applyShaDecision(servedSha, previouslyConnected, true)" in refresh_body, (
        "the post-open refresh must consult the shared SHA decision and is the "
        "ONLY path that remembers the served SHA"
    )
    apply_body = source.split(
        "_applyShaDecision(servedSha, previouslyConnected, storeServedSha) {", 1
    )[1].split("_reloadForShaDecision(decision) {", 1)[0]
    assert "decide(this._lastSha, servedSha, previouslyConnected)" in apply_body, (
        "the shared applier must delegate to the exported SSOT decide()"
    )


def test_only_an_owner_restart_asks_for_the_runtime_mode_to_be_re_read(tmp_path, monkeypatch):
    """The owner raises the mode in Settings, presses Restart, and the mode never changes.

    ``restart_current_process`` re-execs with ``os.environ.copy()``, so the ratchet
    pin ``OUROBOROS_BOOT_RUNTIME_MODE`` rode into the replacement and the child
    re-pinned the OLD baseline from it — verified live: the process held
    ``OUROBOROS_RUNTIME_MODE=advanced`` under ``OUROBOROS_BOOT_RUNTIME_MODE=light``
    and ``/api/state`` kept reporting light, on that restart and every later one.

    One fact decides it, and the two directions are pinned here: the owner's
    ``/restart`` marks the restart owner-initiated, the agent's restart request
    (via the supervisor) does not.
    """
    import server
    import supervisor.message_bus as message_bus

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": offset,
                "message": {"chat": {"id": 1}, "from": {"id": 1}, "text": "/restart"},
            }]

    class Ctx:
        consciousness = None

        def load_state(self):
            return {}

        def save_state(self, _state):
            return None

        def update_state(self, mutator):
            live = {}
            mutator(live)
            return live

        def send_with_budget(self, _chat_id, _text):
            return None

        def safe_restart(self, **_kwargs):
            return True, "OK"

        def kill_workers(self, force=True, **_kwargs):
            return None

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)

    server._owner_restart_requested.clear()
    server._restart_requested.clear()
    try:
        server._process_bridge_updates(Bridge(), 0, Ctx())
        assert server._owner_restart_requested.is_set(), \
            "the owner's Restart button must mark the restart owner-initiated"

        # The agent's own restart request goes through the same exit signal and
        # must NOT claim owner intent: its baseline stays pinned.
        server._owner_restart_requested.clear()
        server._restart_requested.clear()
        server._request_restart_exit()
        assert server._restart_requested.is_set()
        assert not server._owner_restart_requested.is_set()
    finally:
        server._owner_restart_requested.clear()
        server._restart_requested.clear()
