"""Source labels (owner's choice C) through registered bridge handlers and the actual Bot API encoder.

A Project's message names its Project; named work of its own in Main names that work; a direct
reply, a typed completion/refusal notice and an unknown source carry no label."""
from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from tests.test_telegram_reply_references import bridge, _state  # noqa: F401 - shared recording Bot API wire
from ouroboros.task_results import write_task_result
from ouroboros.projects_registry import create_project, update_project
from supervisor import message_bus
from supervisor.message_origin import message_origin

_ANCESTRY = ("Main", "Task:", "Worker:", "Project:", "›")


@pytest.fixture
def wired(bridge, monkeypatch):  # noqa: F811 - pytest fixture imported for the shared Bot API wire
    b = bridge
    data = b.api.state_dir / "data"
    data.mkdir(exist_ok=True)
    subscribers = {}
    b.api.register_supervised_task = lambda *args, **kwargs: None
    b.api.subscribe_event = lambda topic, handler: subscribers.update({topic: handler})
    b.api.register_route = lambda *args, **kwargs: None
    b.api.register_settings_section = lambda *args, **kwargs: None
    b.api.on_unload = lambda *args: None
    monkeypatch.setattr(b.plugin, "register_miniapp", lambda api: False)
    b.plugin.register(b.api)
    monkeypatch.setattr(message_bus, "DATA_DIR", data)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 42})
    monkeypatch.setattr(message_bus, "_chat_media_download_url", lambda *args: ("", ""))
    events = []

    def publish(topic, event):
        events.append((topic, event))
        if topic in subscribers:
            asyncio.run(subscribers[topic](event))

    monkeypatch.setattr(message_bus, "publish_event", publish)
    bus = message_bus.LocalChatBridge({})
    return SimpleNamespace(**vars(b), data=data, bus=bus, events=events, subscribers=subscribers)


def tasks(w, *, depth=1, root="root-work", title="Build export", **root_fields):
    write_task_result(w.data, root, "running", **({"title": title} if title else {}), chat_id=1, **root_fields)
    parent = root
    for index in range(depth):
        tid = f"{root}-worker-{index}"
        write_task_result(w.data, tid, "running", title=f"Reader {index}", chat_id=1,
                          parent_task_id=parent, root_task_id=root, delegation_role="subagent")
        parent = tid
    return parent


def texts(w):
    return [data["text"] for method, data in w.calls if method == "sendMessage"]


def label_of(w, event):
    return sys.modules[w.plugin.__package__ + ".lib.telegram_origin"].origin_label(event, "en")


def test_main_labels_only_named_work_never_direct_replies_or_ancestry(wired):
    w = wired
    write_task_result(w.data, "direct", "running", _is_direct_chat=True, title="Hello", chat_id=1)
    w.bus.send_message(1, "Direct answer", task_id="direct")
    child = tasks(w)
    w.bus.send_message(1, "Work answer", task_id="root-work")
    w.bus.send_message(1, "Worker note", task_id=child, progress_meta={"system_type": "proactive_message"})
    # A worker of the direct reply speaks for that reply: still no heading.
    write_task_result(w.data, "direct-kid", "running", title="Lookup", chat_id=1, parent_task_id="direct",
                      root_task_id="direct", delegation_role="subagent")
    w.bus.send_message(1, "Helper note", task_id="direct-kid", progress_meta={"system_type": "proactive_message"})
    # A bare request is not a name; an unknown task and a host row without a task stay unknown.
    write_task_result(w.data, "unnamed", "running", objective="Fix the parser crash on empty input", chat_id=1)
    w.bus.send_message(1, "Unnamed answer", task_id="unnamed")
    w.bus.send_message(1, "Unknown answer", task_id="missing")
    w.bus.send_message(1, "Host line")
    assert texts(w) == ["Direct answer", "Build export\nWork answer", "Build export\nWorker note", "Helper note",
                        "Unnamed answer", "Unknown answer", "Host line"]
    assert not any(word in text for text in texts(w) for word in _ANCESTRY)
    assert w.api.logs == []


def test_project_name_follows_room_rename_and_project_work_delivered_in_main(wired):
    w = wired
    child = tasks(w, depth=2)
    project = create_project(w.data, "source-export", name="Source export")
    w.bus.send_message(project["chat_id"], "Read the writer", task_id=child,
                       progress_meta={"system_type": "proactive_message"})
    update_project(w.data, project["id"], name="New name")
    w.bus.send_message(project["chat_id"], "Renamed", task_id=child,
                       progress_meta={"system_type": "proactive_message"})
    # The Project's own work names its Project, not itself, also when it speaks in Main.
    write_task_result(w.data, "import", "running", title="Import", chat_id=project["chat_id"])
    w.bus.send_message(1, "Open the sheet for me", task_id="import")
    # A Project the owner never named has no name to show.
    unnamed = create_project(w.data, "proj_0a1b2c")
    w.bus.send_message(unnamed["chat_id"], "Quiet", task_id=child)
    assert texts(w) == ["Source export\nRead the writer", "New name\nRenamed", "New name\nOpen the sheet for me", "Quiet"]


def test_project_quiz_uses_one_label_and_unnamed_project_stays_unlabelled(wired):
    w = wired
    from ouroboros.owner_quiz import record_asked

    named = create_project(w.data, "quiz-project", name="Recipes")
    write_task_result(w.data, "quiz-project-task", "running", chat_id=1, title="Ask")
    record_asked(w.data, "quiz-project-task", quiz_id="named-q", question="Pick?", options=["A"],
                 assumption="continue")
    ok, error = w.bus.send_quiz(
        named["chat_id"], quiz_id="named-q", question="Pick?", options=[{"label": "A"}],
        task_id="quiz-project-task", assumption="continue",
    )
    assert (ok, error) == (True, "ok")
    assert w.calls[0][1]["text"].startswith("Recipes\nQuestion: Pick?")
    assert "Project: Recipes" not in w.calls[0][1]["text"]

    w.calls.clear()
    unnamed = create_project(w.data, "proj_0a1b2c")
    write_task_result(w.data, "unnamed-quiz-task", "running", chat_id=1, title="Ask")
    record_asked(w.data, "unnamed-quiz-task", quiz_id="unnamed-q", question="Pick?", options=["A"],
                 assumption="continue")
    ok, error = w.bus.send_quiz(
        unnamed["chat_id"], quiz_id="unnamed-q", question="Pick?", options=[{"label": "A"}],
        task_id="unnamed-quiz-task", assumption="continue",
    )
    assert (ok, error) == (True, "ok")
    assert not w.calls[0][1]["text"].startswith("Project:")


@pytest.mark.parametrize(("origin", "project_name", "first_line"), [
    ({"label": "Recipes"}, "Recipes", "Recipes"),  # the label already names it: one line
    ({}, "Recipes", "Project: Recipes"),  # a failed host lookup keeps the legacy line
    (None, "Recipes", "Project: Recipes"),  # an older host's event
    ({}, "proj_0a1b2c", "Question: Pick?"),  # a minted id is no name
    (None, "proj_0a1b2c", "Question: Pick?"),
    ({}, "blog", "Project: blog"),  # an owner's own short name stays
], ids=["label", "failed-lookup", "older-host", "minted", "minted-older-host", "owner-short-name"])
def test_quiz_project_line_only_when_no_label_and_a_real_name(wired, origin, project_name, first_line):
    w = wired
    event = {"chat_id": 1, "task_id": "root-work", "quiz_id": "pick", "question": "Pick?",
             "options": [{"label": "A"}], "project_name": project_name,
             **({} if origin is None else {"message_origin": origin})}
    asyncio.run(w.subscribers["chat.quiz"](event))
    lines = w.calls[0][1]["text"].split("\n")
    assert lines[0] == first_line
    assert sum(line.endswith(project_name) for line in lines) == (0 if project_name.startswith("proj_") else 1)


def _bind_owner(monkeypatch, **bindings):
    """A stand-in for the owner bindings ``state.json`` proves; any other control is unknown.
    The real file, witness and producers run in the ``admitted`` tests below."""
    from supervisor import state

    monkeypatch.setattr(state, "control_in_copy", lambda _path, key: (key in bindings, bindings.get(key)))


@pytest.mark.parametrize("metadata", [
    {},  # no transport-specific stamp at all
    {"work_kind": "owner", "schedule_writer_kind": "owner"},  # the stamps a schedule writer leaves
    {"client_surface": {"channel": "web"}},  # owner ingress from another surface
], ids=["no-stamp", "schedule-stamps", "other-surface"])
@pytest.mark.parametrize("binding", ["owner_external_chat_id", "owner_chat_id"])
def test_named_standalone_owner_work_in_the_owners_bound_chat_gets_label(wired, monkeypatch, metadata, binding):
    w = wired
    _bind_owner(monkeypatch, **{binding: 42})
    write_task_result(w.data, "owner-work", "running", chat_id=42, title="Export report", metadata=metadata)
    assert message_origin(w.data, 42, "owner-work") == {"label": "Export report"}
    # Its worker speaks for it there too, never naming its chain.
    write_task_result(w.data, "owner-kid", "running", title="Reader", chat_id=42, parent_task_id="owner-work",
                      root_task_id="owner-work", delegation_role="subagent")
    assert message_origin(w.data, 42, "owner-kid") == {"label": "Export report"}


def test_confirmed_primary_owner_provenance_labels_unbound_external_work(wired, monkeypatch):
    w = wired
    # Telegram-first/strict-command installs can have a confirmed owner but no
    # owner_external_chat_id: no host slash command has ever been forwarded.
    _bind_owner(monkeypatch, owner_id=1, owner_external_chat_id=None, owner_chat_id=1)
    write_task_result(
        w.data, "telegram-owner", "running", chat_id=42, title="Export report",
        metadata={
            "client_surface": {"channel": "telegram"},
        },
        origin_message_ref={"chat_id": 42, "client_message_id": "m-1"},
    )
    assert message_origin(w.data, 42, "telegram-owner") == {"label": "Export report"}

    # The scheduler intentionally strips transport-specific metadata, but its
    # admitted schedule identity remains a transport-neutral owner fact.
    write_task_result(
        w.data, "telegram-scheduled", "running", chat_id=42, title="Nightly export",
        metadata={"schedule_id": "nightly", "source": "scheduled_task"},
    )
    assert message_origin(w.data, 42, "telegram-scheduled") == {"label": "Nightly export"}

    # Confirmed owner state alone never labels an unrelated chat.
    write_task_result(w.data, "other-chat", "running", chat_id=77, title="Other")
    assert message_origin(w.data, 42, "other-chat") == {}


def test_owner_chat_label_needs_a_proven_binding_and_this_chats_named_owner_work(wired, monkeypatch):
    w = wired
    write_task_result(w.data, "owner-work", "running", chat_id=42, title="Export report")
    # Unknown, unreadable or another chat's binding proves nothing.
    _bind_owner(monkeypatch)
    assert message_origin(w.data, 42, "owner-work") == {}
    _bind_owner(monkeypatch, owner_external_chat_id=77, owner_chat_id=None)
    assert message_origin(w.data, 42, "owner-work") == {}
    _bind_owner(monkeypatch, owner_external_chat_id="42")
    assert message_origin(w.data, 42, "owner-work") == {}
    _bind_owner(monkeypatch, owner_external_chat_id=42)
    # A direct reply, Presence-carried work and a bare request stay unlabelled.
    write_task_result(w.data, "direct", "running", chat_id=42, title="Hello", _is_direct_chat=True)
    write_task_result(w.data, "presence-turn", "running", chat_id=42, title="Reply", _presence_turn=True)
    write_task_result(w.data, "presence-source", "running", chat_id=42, title="Reply", source="presence")
    write_task_result(w.data, "unnamed", "running", chat_id=42, objective="Fix the parser")
    for task_id in ("direct", "presence-turn", "presence-source", "unnamed", "missing"):
        assert message_origin(w.data, 42, task_id) == {}, task_id
    # The work must belong to this very chat: another room, A2A or an unknown room has no guessed source.
    write_task_result(w.data, "elsewhere", "running", chat_id=55555, title="Elsewhere")
    assert message_origin(w.data, 42, "elsewhere") == {}
    assert message_origin(w.data, 55555, "owner-work") == {} and message_origin(w.data, 55555, "elsewhere") == {}
    write_task_result(w.data, "a2a", "running", chat_id=-1001, title="Peer")
    assert message_origin(w.data, -1001, "a2a") == {}
    # A Project the owner never named has no name, even for work of the owner's chat.
    unnamed = create_project(w.data, "proj_0a1b2c")
    assert message_origin(w.data, unnamed["chat_id"], "owner-work") == {}


def test_owner_chat_label_reads_the_work_row_once(wired, monkeypatch):
    import supervisor.message_origin as source

    w = wired
    _bind_owner(monkeypatch, owner_external_chat_id=42)
    write_task_result(w.data, "owner-work", "running", chat_id=42, title="Export report")
    reads = []
    real = source.open_store

    def counted(*args, **kwargs):
        reads.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(source, "open_store", counted)
    assert message_origin(w.data, 42, "owner-work") == {"label": "Export report"}
    assert len(reads) == 1


_TELEGRAM_OWNER = {"owner_id": 7, "owner_chat_id": 42, "owner_external_id": 7, "owner_external_chat_id": 42}
_DESKTOP_OWNER = {"owner_id": 1, "owner_chat_id": 1, "owner_external_id": 7, "owner_external_chat_id": 42}


@pytest.fixture
def admitted(wired, monkeypatch):
    """The real admission producers on the bridge's data root, with a real ``state.json`` and its completed
    initialization witness (``state.init_state``); the owner bindings are written as its writers write them.
    Nothing stubs ``control_in_copy``; the session's roots come back afterwards (``test_state_authority.root``)."""
    from supervisor import queue, state
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_root_effort_ingress import _pool_ready

    prior = state.DRIVE_ROOT, queue.DRIVE_ROOT, state.TOTAL_BUDGET_LIMIT
    _queue, _state, workers = _install_queue(wired.data, monkeypatch)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda *_a, **_k: None)
    assert state.init_state().quality == "current"
    _pool_ready(monkeypatch, workers)
    yield SimpleNamespace(**vars(wired), state=state, queue=queue, workers=workers)
    state.init(prior[0], prior[2])
    queue.init(prior[1])


def _continued(a):
    """``admit_continuation`` (``POST /api/tasks/{id}/continue``) of named owner work interrupted in chat 42."""
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE, _interrupted

    _interrupted(a.data, chat_id=42, origin_message_ref={"chat_id": 42, "client_message_id": "m-1"})
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    return load_task_result(a.data, ack["successor_task_id"])


def _scheduled(a, monkeypatch, producer):
    """Build one due schedule through its real writer and payload builder.

    The scheduler's claim/admission loop also depends on a live supervisor tick and
    worker-pool timing.  The origin contract is decided before that loop, so this
    test persists the exact task produced by ``_task_from_schedule`` and drives the
    real message-origin reader against it.  No owner metadata is injected here.
    """
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import followup
    from supervisor import queue_schedules
    from tests.test_schedule_followup import _ctx

    due = "2000-01-01T00:00:00+00:00"
    if producer == "followup":  # a run in the owner's chat schedules its own follow-up
        ctx = _ctx(a.data.parent)
        ctx.current_chat_id = 42
        ctx.task_metadata["resource_intent"] = {"kind": "system_repo"}
        write_task_result(a.data, "root-1", "running", root_task_id="root-1", chat_id=42)
        monkeypatch.setattr(followup, "_is_delegated_subagent", lambda _ctx: False)
        out = followup._handle_schedule_followup(ctx, relation="independent", run_at=due,
                                                 objective="Check the export again", name="Weekly export check")
        assert out.startswith("FOLLOWUP_SCHEDULED"), out
    else:  # the owner's Schedules door: no chat in the template, so the run goes to the owner's own chat
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from ouroboros.gateway.schedules import api_schedules_upsert

        monkeypatch.setattr(a.queue, "load_state", a.state.load_state)  # the scheduler reads the real binding
        app = Starlette(routes=[Route("/api/schedules", api_schedules_upsert, methods=["POST"])])
        app.state.drive_root = a.data
        written = TestClient(app).post("/api/schedules", json={
            "id": "export", "name": "Weekly export check", "trigger": {"type": "once", "run_at": due},
            "task": {"type": "task", "text": "Check the export again"}})
        assert written.status_code == 200, written.text
    [row] = queue_schedules.load_schedule_store(a.data)["tasks"]
    fired = queue_schedules._task_from_schedule(row, task_id=f"scheduled-{producer}")
    write_task_result(
        a.data, fired["id"], "running",
        **{key: value for key, value in fired.items() if key != "id"},
    )
    return load_task_result(a.data, fired["id"])


def _assert_owner_labelled(a, stored, label):
    """The persisted row carries no transport stamp; its label reaches the phone through the real bridge."""
    assert a.state.control_in_copy(a.state.STATE_PATH, "owner_external_chat_id") == (True, 42)  # the real file
    assert stored["chat_id"] == 42 and stored["title"] == label
    assert not {"client_surface", "owner_external_chat_id"} & set(stored.get("metadata") or {})
    assert message_origin(a.data, 42, stored["task_id"]) == {"label": label}
    a.bus.send_message(42, "Report body", task_id=stored["task_id"])
    assert texts(a)[-1] == f"{label}\nReport body"


@pytest.mark.parametrize("bound", [_TELEGRAM_OWNER, _DESKTOP_OWNER], ids=["telegram-first", "desktop-first"])
def test_a_real_continue_of_owner_work_in_the_bound_chat_is_labelled(admitted, bound):
    a = admitted
    a.state.update_state(lambda st: st.update(bound))
    _assert_owner_labelled(a, _continued(a), "Continue: Friday report")  # the successor's own admitted name


@pytest.mark.parametrize("producer", ["followup", "owner-door"])
def test_a_real_scheduled_run_in_the_bound_owner_chat_is_labelled(admitted, monkeypatch, producer):
    a = admitted
    a.state.update_state(lambda st: st.update(_TELEGRAM_OWNER))
    _assert_owner_labelled(a, _scheduled(a, monkeypatch, producer), "Weekly export check")


@pytest.mark.parametrize("binding", ["unbound", "unconfirmed"])
def test_a_real_continue_in_an_unproven_owner_chat_stays_unlabelled(admitted, binding):
    a = admitted
    path = a.state.STATE_PATH
    if binding == "unconfirmed":  # a recovered copy keeps the values but cannot prove them (``mark_unconfirmed``)
        def unproven(st):
            st.update(_TELEGRAM_OWNER)
            for key in ("owner_chat_id", "owner_external_chat_id"):
                a.state.mark_unconfirmed(st, key)
        a.state.update_state(unproven)
        assert a.state.read_state_copy(path)[1]["owner_external_chat_id"] == 42
    expected = (False, None) if binding == "unconfirmed" else (True, None)
    assert [a.state.control_in_copy(path, key) for key in ("owner_chat_id", "owner_external_chat_id")] == [expected] * 2
    stored = _continued(a)
    assert stored["chat_id"] == 42 and stored["title"] == "Continue: Friday report"
    assert message_origin(a.data, 42, stored["task_id"]) == {}
    a.bus.send_message(42, "Report body", task_id=stored["task_id"])
    assert texts(a)[-1] == "Report body"


def test_worker_lifecycle_edit_keeps_bubble_and_current_project(wired):
    w = wired
    child = tasks(w)
    project = create_project(w.data, "cards", name="Cards")
    for state, text in (("scheduled", "Queued"), ("running", "Working")):
        w.bus.send_message(project["chat_id"], text, task_id=child, is_progress=True,
                           progress_meta={"subagent_event": state, "subagent_task_id": child})
        update_project(w.data, project["id"], name="Cards now")
    assert [method for method, _ in w.calls] == ["sendMessage", "editMessageText"]
    assert w.calls[0][1]["text"].startswith("Cards\n")
    assert w.calls[1][1]["text"].startswith("Cards now\n") and "Worker:" not in w.calls[1][1]["text"]
    assert w.calls[1][1]["message_id"] == "101"


def test_label_is_literal_and_multipart_body_and_references_survive(wired):
    w = wired
    title = "<b>😀 *R&D*</b>"
    project = create_project(w.data, "literal", name=title)
    body = "**α & β** 😀\n" * 700
    w.bus.send_message(project["chat_id"], body, parse_mode="Markdown", task_id="missing")
    assert len(w.calls) > 2
    stripped = []
    for method, data in w.calls:
        assert method == "sendMessage" and w.transport._u16len(data["text"]) <= 4096
        assert data["text"].startswith("<i>&lt;b&gt;😀 *R&amp;D*&lt;/b&gt;</i>\n")
        stripped.append(data["text"].split("</i>\n", 1)[1])
        mid = 100 + len(stripped)
        assert _state(w.plugin)._reply_reference(w.api, "", 42, mid) == "task:missing"
    expected = w.transport.markdown_to_telegram_html(body.strip())  # existing outbound handler trims its body
    assert "".join(w.transport._telegram_html_to_plain(part) for part in stripped) == w.transport._telegram_html_to_plain(expected)


@pytest.mark.parametrize("kind", ["photo", "video", "document", "links"])
def test_media_and_links_carry_label_and_exact_reference(wired, kind):
    w = wired
    child = tasks(w)
    if kind == "photo":
        w.bus.send_photo(1, b"photo", caption="Caption", task_id=child)
    elif kind == "video":
        w.bus.send_video(1, b"video", caption="Caption", task_id=child)
    elif kind == "document":
        w.bus.send_document(1, b"document", filename="report.txt", caption="Caption", task_id=child)
    else:
        w.bus.send_links(1, [{"label": "Read", "url": "https://example.com"}], title="Links", task_id=child)
    assert len(w.calls) == 1
    method, data = w.calls[0]
    field = "text" if kind == "links" else "caption"
    shown = w.transport._telegram_html_to_plain(data[field])
    assert shown.startswith("Build export\n") and not any(word in shown for word in _ANCESTRY)
    event = w.events[-1][1]
    assert event["task_id"] == child and event.get("row_address")
    assert _state(w.plugin)._reply_reference(w.api, "", 42, 101) == event["row_address"]
    assert w.api.logs == []


def test_a_file_without_caption_gets_the_label_alone_and_a_direct_file_none(wired):
    w = wired
    project = create_project(w.data, "files", name="Recipes")
    w.bus.send_document(project["chat_id"], b"csv", filename="dupes.csv", task_id="missing")
    write_task_result(w.data, "direct", "running", _is_direct_chat=True, title="Hello", chat_id=1)
    w.bus.send_document(1, b"csv", filename="menu.csv", caption="Menu", task_id="direct")
    captions = [w.transport._telegram_html_to_plain(data.get("caption") or "") for _, data in w.calls]
    assert captions == ["Recipes", "Menu"]
    # The label alone is still the HTML-italic line on the wire, as it is on every other message.
    assert w.calls[0][1]["caption"] == "<i>Recipes</i>" and w.calls[0][1]["parse_mode"] == "HTML"


@pytest.mark.parametrize("kind", ["document", "audio"])
@pytest.mark.parametrize("caption", ["", "Caption 😀 & <b> " * 150], ids=["empty", "spilled"])
def test_file_and_audio_label_keeps_html_on_the_wire(wired, kind, caption):
    w = wired
    project = create_project(w.data, "files", name="R&D <files>")
    filename, mime = ("a.mp3", "audio/mpeg") if kind == "audio" else ("dupes.csv", "text/csv")
    w.bus.send_document(project["chat_id"], b"bytes", filename=filename, caption=caption, mime=mime, task_id="missing")
    method, data = w.calls[-1]
    assert method == ("sendAudio" if kind == "audio" else "sendDocument")
    assert data["caption"] == "<i>R&amp;D &lt;files&gt;</i>" and data["parse_mode"] == "HTML"
    spilled = [data for method, data in w.calls if method == "sendMessage"]
    assert bool(spilled) is bool(caption)
    if caption:
        text = "".join(w.transport._telegram_html_to_plain(part["text"].split("</i>\n", 1)[1]) for part in spilled)
        assert text == caption and all(part["parse_mode"] == "HTML" for part in spilled)


def test_long_caption_kept_as_ordered_text_before_media(wired):
    w = wired
    child = tasks(w)
    caption = "A 😀 & B " * 400
    w.bus.send_photo(1, b"photo", caption=caption, task_id=child)
    assert w.calls[-1][0] == "sendPhoto"
    assert w.transport._u16len(w.calls[-1][1]["caption"]) <= 1024
    assert w.transport._telegram_html_to_plain(w.calls[-1][1]["caption"]) == "Build export"
    text = "".join(w.transport._telegram_html_to_plain(data["text"].split("</i>\n", 1)[1])
                   for method, data in w.calls if method == "sendMessage")
    assert text == caption
    ref = w.events[-1][1]["row_address"]
    assert all(_state(w.plugin)._reply_reference(w.api, "", 42, 101 + i) == ref for i in range(len(w.calls)))


def test_quiz_edit_retains_label_after_fresh_client(wired):
    w = wired
    child = tasks(w)
    q = sys.modules[w.plugin.__package__ + ".lib.telegram_quiz"]
    event = {"chat_id": 1, "task_id": child, "quiz_id": "choose", "question": "Choose?",
             "options": [{"label": "A"}], "message_origin": message_origin(w.data, 1, child)}
    asyncio.run(w.subscribers["chat.quiz"](event))
    asyncio.run(w.subscribers["chat.quiz_state"]({"task_id": child, "quiz_id": "choose", "state": "closed", "closed_reason": "Done"}))
    assert [method for method, _ in w.calls] == ["sendMessage", "editMessageText"]
    assert w.calls[0][1]["text"].startswith("Build export\nQuestion: Choose?")
    assert w.calls[1][1]["text"].startswith("Build export\nQuestion: Choose?")
    assert q.quiz_for_message(w.api, "", 42, 101)["task_id"] == child


def test_a_direct_replys_quiz_is_unlabelled(wired):
    w = wired
    write_task_result(w.data, "direct", "running", _is_direct_chat=True, title="Hello", chat_id=1)
    origin = message_origin(w.data, 1, "direct")
    assert origin == {}
    asyncio.run(w.subscribers["chat.quiz"]({"chat_id": 1, "task_id": "direct", "quiz_id": "direct-q", "question": "Which one?",
                                            "options": [{"label": "A"}], "message_origin": origin}))
    assert w.calls and w.calls[0][1]["text"].startswith("Question: Which one?")


def test_unknown_cyclic_and_unrecorded_sources_stay_unlabelled(wired, monkeypatch):
    w = wired
    deep = tasks(w, depth=9)
    # A deep worker with its recorded root names that work, never its chain.
    assert message_origin(w.data, 1, deep) == {"label": "Build export"}
    # Outside Main and a Project nothing is nameable: no guessed room.
    assert message_origin(w.data, 55555, deep) == {}
    assert message_origin(w.data, 0) == {} and message_origin(None, 1, "root-work") == {}
    # An early lifecycle event before the worker's row: only its recorded lineage names it.
    assert message_origin(w.data, 1, event={"subagent_task_id": "early-child"}) == {}
    assert message_origin(w.data, 1, event={"subagent_task_id": "early-child", "parent_task_id": "root-work",
                                            "root_task_id": "root-work", "delegation_role": "subagent"}) == {"label": "Build export"}
    write_task_result(w.data, "cycle", "running", title="Loop", chat_id=1, parent_task_id="cycle", delegation_role="subagent")
    write_task_result(w.data, "orphan", "running", title="Orphan", chat_id=1, parent_task_id="ghost", delegation_role="subagent")
    assert message_origin(w.data, 1, "cycle") == {} and message_origin(w.data, 1, "orphan") == {}
    # An older producer's event, a malformed origin and a NUL-padded name.
    assert label_of(w, {"chat_id": 444}) == "" and label_of(w, {"message_origin": "Main"}) == ""
    assert label_of(w, {"message_origin": {"label": "  Re\x00cipes \n catalog "}}) == "Recipes catalog"
    import supervisor.message_origin as source
    monkeypatch.setattr(source, "project_of_room", lambda *args: (_ for _ in ()).throw(OSError("unreadable")))
    assert message_origin(w.data, 1, "root-work") == {}


@pytest.mark.parametrize("answer_kind", ["callback", "reply", "closed"])
@pytest.mark.parametrize("question", ["Q" * 3300, "Long question 😀\n" * 800], ids=["short", "overflow"])
def test_native_quiz_answer_preserves_label_without_polluter(wired, answer_kind, question):
    w = wired
    child = tasks(w)
    q = sys.modules[w.plugin.__package__ + ".lib.telegram_quiz"]
    event = {"chat_id": 1, "task_id": child, "quiz_id": "native", "question": question,
             "options": [{"label": "A"}], "message_origin": message_origin(w.data, 1, child)}
    asyncio.run(w.subscribers["chat.quiz"](event))
    token = q.mint_token(child, "native")
    record = q.quiz_for_token(w.api, "", token)
    heading = "Build export\n"
    for method, data in w.calls:
        assert method == "sendMessage" and data["chat_id"] == "42"
        assert data["text"].startswith(heading) and data["text"].count(heading) == 1
        assert w.transport._u16len(data["text"]) <= 4096
    assert record["text"].startswith(heading) and record["text"].count(heading) == 1
    client = w.plugin._telegram_client(w.api)  # the poller's long-lived client, no event

    async def post(*args):
        if answer_kind == "closed":
            return 409, {"state": "closed", "outcome": "closed", "closed_reason": "Finished"}
        return 200, {"status": "answered", "state": "answered", "accepted": True,
                     "answered_index": 0, "comment": "😀" * 200}

    async def act():
        if answer_kind == "callback":
            await q.answer_from_callback(w.api, client, f"qz:{token}:0", cb_id="tap", request_key="tap", lang="en", post=post)
        else:
            await q.answer_from_reply(w.api, client, record, "Answer", chat_id=42, request_key="reply", lang="en", post=post)
        await client.send_message(42, "Unrelated response", parse_mode="")

    asyncio.run(act())
    edits = [data for method, data in w.calls if method == "editMessageText"]
    assert edits and edits[-1]["text"].startswith(heading)
    assert edits[-1]["text"].count(heading) == 1
    assert w.transport._u16len(edits[-1]["text"]) <= 4096
    assert w.calls[-1][1]["text"] == "Unrelated response"
    assert client.origin_label == ""


def test_quiz_creation_keeps_event_label_and_published_answer_edits_once(wired, monkeypatch):
    w = wired
    child = tasks(w)
    clients = []
    real_client = w.plugin._telegram_client

    def client(*args, **kwargs):
        value = real_client(*args, **kwargs)
        clients.append((value, value.origin_label))
        return value

    monkeypatch.setattr(w.plugin, "_telegram_client", client)
    asyncio.run(w.subscribers["chat.quiz"]({
        "chat_id": 1, "task_id": child, "quiz_id": "published", "question": "Already answered?",
        "options": [{"label": "A"}], "state": "answered", "answered_index": 0,
        "message_origin": message_origin(w.data, 1, child)}))
    heading = "Build export\n"
    assert [method for method, _ in w.calls] == ["sendMessage", "editMessageText"]
    assert all(data["text"].startswith(heading) and data["text"].count(heading) == 1 for _, data in w.calls)
    assert w.calls[-1][1]["text"] == heading + "Question: Already answered?\n1. A\nAnswered: 1. A"
    assert clients[0][1] and all(value.origin_label == original for value, original in clients)
    assert w.api.logs == []


def test_quiz_label_counts_in_the_answered_edit_budget(wired):
    w = wired
    child = tasks(w)
    q = w.plugin.telegram_quiz
    heading = "Build export\n"
    hint = q.hint("en")
    # Without the label this fits exactly with the edit reserve; with it the
    # card must use the compact answer door, not spend that reserve on chrome.
    size = 4096 - q.answered_edit_reserve("en") - w.transport._u16len("Question: \n1. A\n" + hint)
    event = {"chat_id": 1, "task_id": child, "quiz_id": "budget", "question": "Q" * size,
             "options": [{"label": "A"}], "message_origin": message_origin(w.data, 1, child)}
    asyncio.run(w.subscribers["chat.quiz"](event))
    assert len(w.calls) == 2
    assert "reply_markup" not in w.calls[0][1] and "reply_markup" in w.calls[1][1]
    assert w.calls[0][1]["text"] == heading + "Question: " + "Q" * size + "\n1. A"
    record = q.quiz_for_message(w.api, "", 42, 102)
    assert record["text"] == heading + "1. A"
    assert w.api.logs == []


def test_audio_document_fallback_does_not_repeat_long_caption(wired):
    w = wired
    child = tasks(w)
    real = w.transport.TelegramClient.call

    async def call(client, method, **kwargs):
        if method == "sendAudio":
            raise w.transport.TelegramRequestRejected("Unsupported audio", status_code=400)
        return await real(client, method, **kwargs)

    from unittest.mock import patch
    caption = "Caption 😀 " * 150
    with patch.object(w.transport.TelegramClient, "call", call):
        w.bus.send_document(1, b"audio", filename="a.mp3", caption=caption, mime="audio/mpeg", task_id=child)
    assert [method for method, _ in w.calls] == ["sendMessage", "sendDocument"]
    text = w.transport._telegram_html_to_plain(w.calls[0][1]["text"].split("</i>\n", 1)[1])
    assert text == caption
    assert w.calls[1][1]["caption"] == "<i>Build export</i>" and w.calls[1][1]["parse_mode"] == "HTML"


def test_labelled_multipart_keeps_the_native_owner_reply(wired):
    from tests.test_telegram_reply_references import _accept_owner

    w = wired
    ref = _accept_owner(w, 10)
    _accept_owner(w, 11)
    event = {"chat_id": 1, "task_id": "root-work", "reply_to": ref, "text": "Reply " * 2000,
             "message_origin": {"label": "Export"}}
    asyncio.run(w.subscribers["chat.outbound"](event))
    assert len(w.calls) > 2
    assert all(data.get("reply_to_message_id") == 10 for _, data in w.calls)
    assert all(data["text"].startswith("<i>Export</i>\n") for _, data in w.calls)
    assert w.api.logs == []


def test_long_name_is_bounded_by_the_host_but_authored_text_is_not(wired):
    w = wired
    title = "Широкая длинная задача 😀 " * 5
    write_task_result(w.data, "long", "running", title=title, chat_id=1)
    w.bus.send_message(1, "Complete body\nsecond line", task_id="long")
    label, body = w.calls[0][1]["text"].split("\n", 1)
    assert label.endswith("…") and len(label) <= 80 and body == "Complete body\nsecond line"


def test_card_fallback_has_own_label_and_reference(wired, monkeypatch):
    from ouroboros.card_store import write_card
    from supervisor import card_delivery

    w = wired
    child = tasks(w)
    record = write_card(w.data, chat_id=1, task_id=child, title="Overview", fallback_text="Whole fallback",
                        kind="components", body={"components": [{"type": "markdown", "text": "Hello"}]})
    monkeypatch.setattr(card_delivery, "publish_event", message_bus.publish_event)
    card_delivery.send_card(w.bus, record, 1, task_id=child, interactive=False)
    assert w.calls[-1][1]["text"] == "Build export\nWhole fallback"  # a plain card stays plain
    assert _state(w.plugin)._reply_reference(w.api, "", 42, 101).startswith("row:1@")


def test_typed_completion_and_refusal_headings_are_not_repeated(wired):
    w = wired
    w.plugin.merge_settings(w.api.state_dir, {"TELEGRAM_NOTIFY_TASKS": "on"})
    origin = {"label": "Recipes"}
    completion = {"role": "system", "system_type": "project_completion_summary", "chat_id": 1,
                  "task_id": "root-work", "target_label": "Recipes › Import", "text": "Recipes › Import / Done",
                  "completion_notice": {"neutral": False, "detail": "", "phase": "done"}, "message_origin": origin}
    refusal = {"chat_id": 1, "annotation_type": "routing_ack", "routing_notice_carrier": "routing_ack",
               "target_label": "Import", "cause": "Not delivered: the work has finished", "message_origin": origin}
    system_refusal = {"chat_id": 1, "role": "system", "system_type": "steer_not_delivered", "task_id": "root-work",
                      "routing_notice_carrier": "system", "text": "Import · Not delivered", "message_origin": origin}
    for event in (completion, refusal, system_refusal):
        asyncio.run(w.subscribers["chat.outbound"](event))
    sent = texts(w)
    assert sent[0] == w.plugin._completion_text(completion, "root-work", "en") and "Recipes › Import" in sent[0]
    assert sent[1:] == ["Import · Not delivered: the work has finished", "Import · Not delivered"]
    assert not any(text.startswith("Recipes\n") or text.startswith("<i>Recipes") for text in sent)


def test_label_does_not_change_mirror_or_child_final_policy(wired):
    w = wired
    child = tasks(w)
    origin = message_origin(w.data, 1, child)
    assert origin == {"label": "Build export"}
    asyncio.run(w.subscribers["chat.outbound"]({"chat_id": 1, "task_id": child, "text": "Private child report",
                "delegation_role": "subagent", "message_origin": origin}))
    assert w.calls == []
    (w.api.state_dir / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "telegram_only"}), encoding="utf-8")
    w.bus.send_message(1, "Web-only", task_id="root-work")
    assert w.calls == []
