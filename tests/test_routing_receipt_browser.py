"""#968 in the real SPA: the routing receipt under an owner message reads the same live and after
a reload, whichever producer wrote it (owner choice A: a late reply from Main names its work and
that work's Project, in the receipt's existing look).

Frames come from the real producers through the real ``LocalChatBridge``; the reload reads the
real history endpoint over the same data root. No runtime, provider or Telegram.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests import test_subscription_setup_browser as setup
from tests.test_human_rows_browser import _evidence, _host, _push, _serve_history
from tests.test_sidebar_activity_browser import FRAMES, activity, mount_sidebar
from tests.ui_chat_viewport_smoke import wait_drawer_closed

subscription_ui = setup.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
TITLE = "Fix duplicate posts in the RSS import"
WORDS = "Keep the old feed address and add a redirect from /feed.xml."
LINE = """cmid => {
    const note = document.querySelector(`.chat-bubble.user[data-client-message-id="${cmid}"] .msg-routing-annotation`);
    if (!note) return null;
    const lead = note.querySelector(':scope > span');
    const reference = note.querySelector('[data-intent="open-project"]');
    return { lead: lead?.textContent || '', reference: reference?.getAttribute('aria-label') || '',
             name: reference?.querySelector('.chat-live-project-name')?.textContent || '',
             cut: [note, lead].filter(Boolean).some((node) => node.scrollWidth > node.clientWidth + 1),
             status: note.dataset.annotationStatus || '' };
}"""


def _receipt(producer, root, bridge, cmid):
    if producer == "late_reply":  # the direct server path: words saved into the work's mailbox
        from ouroboros.project_dialogue import routing_target_label
        from ouroboros.server_owner_routing import _record_routing_receipt

        _record_routing_receipt(bridge, SimpleNamespace(DRIVE_ROOT=root), chat_id=1, client_message_id=cmid,
                                action="mailbox_delivery", target="work-1",
                                target_label=routing_target_label(root, "mailbox_delivery", "work-1"), status="delivered")
    else:  # the supervisor's token-bound relay receipt
        from supervisor.events_project_routing import _emit_routing_receipt

        _emit_routing_receipt(SimpleNamespace(DRIVE_ROOT=root, bridge=bridge),
                              {"client_message_id": cmid, "routing_token": "relay-1", "relay_id": "relay-1",
                               "chat_id": 1, "relay_target": {"id": "work-1", "title": TITLE}},
                              action="steer_task", target="work-1", status="delivered")


@pytest.mark.parametrize("theme,width", [("light", 1110), ("dark", 1110), ("light", 390)])
@pytest.mark.parametrize("producer", ["late_reply", "relay"])
def test_a_routed_message_receipt_reads_the_same_live_and_after_reload(
        subscription_ui, tmp_path, monkeypatch, producer, theme, width):
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result
    from supervisor import message_bus

    ui, out = subscription_ui, _evidence(tmp_path)
    page = ui["page"]
    bridge, frames = _host(tmp_path, monkeypatch)
    project = create_project(tmp_path, "blog", name="Blog")
    bind_task_to_project(tmp_path, "work-1", "blog", project["chat_id"], origin={"absent": "system"})
    write_task_result(tmp_path, "work-1", "running", chat_id=project["chat_id"], title=TITLE, result="working")
    room = dict(id="blog", name="Blog", chat_id=project["chat_id"], lifecycle="active", visible_revision=0)
    # The owner's saved message in Main (its ingress already confirmed), then the receipt that answers it.
    cmid = "web-late-answer-0001"
    message_bus.log_chat("in", 1, 7, WORDS, source="web", client_message_id=cmid, drive_root=tmp_path)
    _serve_history(page, tmp_path)
    mounted = mount_sidebar(ui, width, theme, False, [room], [activity("work-1", "blog")])
    if width < 700:  # the narrow layout opened the navigation drawer; choosing Main Chat closes it
        page.locator('#primary-sidebar [data-nav-page="chat"]').click()
        wait_drawer_closed(page)
    page.locator(f'.chat-bubble.user[data-client-message-id="{cmid}"]').wait_for()

    sent = len(frames)
    _receipt(producer, tmp_path, bridge, cmid)
    _push(mounted, frames, sent)
    bubble = page.locator(f'.chat-bubble.user[data-client-message-id="{cmid}"]')
    bubble.locator(".msg-routing-annotation").wait_for()
    page.evaluate(FRAMES)
    live = page.evaluate(LINE, cmid)
    bubble.scroll_into_view_if_needed()
    bubble.screenshot(path=str(out / f"receipt-{producer}-live-{theme}-{width}.png"), animations="disabled")

    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => window.__ouroWs?.ws?.readyState === WebSocket.OPEN")
    bubble.locator(".msg-routing-annotation").wait_for()
    page.evaluate(FRAMES)
    replay = page.evaluate(LINE, cmid)
    bubble.scroll_into_view_if_needed()
    bubble.screenshot(path=str(out / f"receipt-{producer}-reload-{theme}-{width}.png"), animations="disabled")
    (out / f"receipt-{producer}-{theme}-{width}.json").write_text(json.dumps({"live": live, "reload": replay}, indent=1))

    assert live == {"lead": f"Passed on to «{TITLE}» in", "reference": "Open project Blog", "name": "Blog",
                    "cut": False, "status": "delivered"}, live
    assert replay == live
    assert not ui["errors"].unexpected(), ui["errors"].report()


# The compact work names (the reported case: msg-1793137452751-1 → 8a5774f9). An older host stored a nameless work's
# whole text as its title; today's producer names a work as its card does, and the line shows any stored name compact.
LONG_MESSAGE = ("Проверь пожалуйста обработку длинных сообщений и покажи понятный краткий результат " * 12)[:824]
PREVIEW_VIEWS = {"webkit-1165": dict(width=1165, height=851, touch=False), "mobile-390": dict(width=390, height=844, touch=True)}
CASES = {"short": "Keep the old feed address.", "assignment": "Resume the community watch.",
         "one_line": "Do the deep pass now.", "multiple": "Both works should know this.", "unnamed": "Resume it now."}
SHOWN = """cmid => {
    const note = document.querySelector(`.chat-bubble.user[data-client-message-id="${cmid}"] .msg-routing-annotation`);
    if (!note) return null;
    const lead = note.querySelector(':scope > span');
    return { lead: lead?.textContent || note.textContent,
             reference: note.querySelector('[data-intent="open-project"]')?.getAttribute('aria-label') || '',
             cut: [note, lead].filter(Boolean).some((node) => node.scrollWidth > node.clientWidth + 1),
             stored_text_in_dom: ['Read the current thread', 'краткий результат Проверь'].some((part) => note.outerHTML.includes(part)) };
}"""


def _older_host_receipt(root, bridge, cmid, *, targets, titles, project):
    """A receipt row an older host stored and sent, its work titles kept whole as they were written."""
    from ouroboros.project_dialogue import append_chat_annotation

    fields = dict(action="steer_task", target=targets[0], status="delivered", routing_token=f"older-{cmid}",
                  project_id=project["id"], project_chat_id=project["chat_id"], project_name=project["name"],
                  target_task_ids=list(targets), target_titles=list(titles))
    assert append_chat_annotation(root, cmid, **fields)
    bridge.send_routing_ack(1, client_message_id=cmid, **fields)


def _preview_sheet(page, path):
    """One small image of the five receipts as painted (each bubble shot, stacked): the compiled preview."""
    from PIL import Image

    shots = []
    for case in CASES:
        bubble = page.locator(f'.chat-bubble.user[data-client-message-id="preview-{case}"]')
        bubble.evaluate("el => el.scrollIntoView({block: 'center'})")  # clear of the floating header and the composer
        page.evaluate(FRAMES)
        shot = path.with_name(f"{path.stem}-{case}.png")
        bubble.screenshot(path=str(shot), animations="disabled")
        shots.append(Image.open(shot))
    sheet = Image.new("RGB", (max(s.width for s in shots), sum(s.height + 12 for s in shots) - 12), "white")
    top = 0
    for shot in shots:
        sheet.paste(shot, (0, top))
        top += shot.height + 12
    sheet.save(path)


@pytest.mark.parametrize("view", list(PREVIEW_VIEWS))
def test_receipt_work_names_show_compact_live_after_reload_and_reconnect(subscription_ui, tmp_path, monkeypatch, view):
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor import message_bus
    from supervisor.events_project_routing import _emit_routing_receipt
    from tests.test_routing_receipt_facts import assignment

    ui, out, spec = subscription_ui, _evidence(tmp_path), PREVIEW_VIEWS[view]
    bridge, frames = _host(tmp_path, monkeypatch)
    blog, watch = create_project(tmp_path, "blog", name="Blog"), create_project(tmp_path, "community-watch")
    for task, project in (("w-short", blog), ("8a5774f9", watch), ("w-owner", watch)):
        bind_task_to_project(tmp_path, task, project["id"], project["chat_id"], origin={"absent": "system"})
    for case, words in CASES.items():
        message_bus.log_chat("in", 1, 7, words, source="web", client_message_id=f"preview-{case}", drive_root=tmp_path)
    page = ui["new_page"](viewport={"width": spec["width"], "height": spec["height"]}, has_touch=spec["touch"])
    history_reads: list = []
    page.on("request", lambda request: history_reads.append(request.url) if "/api/chat/history" in request.url else None)
    _serve_history(page, tmp_path)
    rooms = [dict(id=p["id"], name=p["name"], chat_id=p["chat_id"], lifecycle="active", visible_revision=0) for p in (blog, watch)]
    mounted = mount_sidebar({**ui, "page": page}, spec["width"], "light", False, rooms, [])
    if spec["width"] < 700:  # the narrow layout opened the navigation drawer; choosing Main Chat closes it
        page.locator('#primary-sidebar [data-nav-page="chat"]').click()
        wait_drawer_closed(page)
    page.set_viewport_size({"width": spec["width"], "height": spec["height"]})
    for case in CASES:
        page.locator(f'.chat-bubble.user[data-client-message-id="preview-{case}"]').wait_for()

    # Live: today's producer for a named and a nameless work, and three rows as an older host stored and sent them.
    sent, ctx = len(frames), SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge)
    _emit_routing_receipt(ctx, {"client_message_id": "preview-short", "routing_token": "r-short", "relay_id": "r-short",
                                "chat_id": 1, "relay_target": {"id": "w-short", "suggested_name": TITLE}},
                          action="steer_task", target="w-short", status="delivered")
    _older_host_receipt(tmp_path, bridge, "preview-assignment", targets=["8a5774f9"], titles=[assignment()], project=watch)
    _older_host_receipt(tmp_path, bridge, "preview-one_line", targets=["w-owner"], titles=[LONG_MESSAGE], project=watch)
    _older_host_receipt(tmp_path, bridge, "preview-multiple", targets=["w-short", "8a5774f9"], titles=[TITLE, assignment()],
                        project=watch)
    _emit_routing_receipt(ctx, {"client_message_id": "preview-unnamed", "routing_token": "r-unnamed", "relay_id": "r-unnamed",
                                "chat_id": 1, "relay_target": {"id": "8a5774f9", "text": assignment()}},
                          action="steer_task", target="8a5774f9", status="delivered")
    _push(mounted, frames, sent)

    def painted(phase):
        for case in CASES:
            page.locator(f'.chat-bubble.user[data-client-message-id="preview-{case}"] .msg-routing-annotation').wait_for()
        page.evaluate(FRAMES)
        _preview_sheet(page, out / f"receipt-preview-{view}-{phase}.png")
        return {case: page.evaluate(SHOWN, f"preview-{case}") for case in CASES}

    seen = {"live": painted("live")}
    ui["errors"].watch(page)  # the reload may cut off this page's own reads; other pages stay strict
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => window.__ouroWs?.ws?.readyState === WebSocket.OPEN")
    seen["reload"] = painted("reload")
    # Reconnect: the socket drops, the client opens a new one and reads the room's history again.
    sockets, reads = len(mounted["sockets"]), len(history_reads)
    mounted["sockets"][-1].close()
    for _ in range(150):
        if len(mounted["sockets"]) > sockets and len(history_reads) > reads:
            break
        page.wait_for_timeout(200)
    page.wait_for_function("() => window.__ouroWs?.ws?.readyState === WebSocket.OPEN")
    page.wait_for_timeout(600)
    seen["reconnect"] = painted("reconnect")
    (out / f"receipt-preview-{view}.json").write_text(json.dumps(seen, ensure_ascii=False, indent=1))

    assert len(mounted["sockets"]) > sockets and len(history_reads) > reads, "the reconnect read the history again"
    watch_ref, blog_ref = "Open project community-watch", "Open project Blog"
    one_line = seen["live"]["one_line"]["lead"]
    assert one_line.startswith("Passed on to «Проверь пожалуйста обработку длинных сообщений") and one_line.endswith("…» in")
    assert len(one_line) <= len("Passed on to «» in") + 81
    assert {case: (row["lead"], row["reference"]) for case, row in seen["live"].items()} == {
        "short": (f"Passed on to «{TITLE}» in", blog_ref),
        "assignment": ("Passed on to «Community watch — one six-hourly GitHub pass…» in", watch_ref),
        "one_line": (one_line, watch_ref),
        "multiple": (f"Passed on to «{TITLE}», «Community watch — one six-hourly GitHub pass…» in", watch_ref),
        "unnamed": ("Passed on to the work in", watch_ref),
    }
    for phase, rows in seen.items():
        assert rows == seen["live"], phase
        assert not any(row["cut"] or row["stored_text_in_dom"] for row in rows.values()), (phase, rows)
    assert not ui["errors"].unexpected(), ui["errors"].report()
