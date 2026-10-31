"""``show_card``: the card validator, the tool's bodies and refusals, and the card store's bounds.

A card is stored before its ``send_card`` event leaves the task (the supervisor half is
``tests/test_living_cards_delivery.py``); every refusal names what was wrong and that the
card was not shown, and stores nothing.
"""

from __future__ import annotations

import json
import queue

import pytest

from ouroboros.card_store import (
    MAX_APP_BYTES, MAX_SUBMISSIONS, CardStoreError, load_card, read_app_bytes, record_submission,
    submissions_summary, write_card,
)
from ouroboros.contracts.plugin_api import ExtensionRegistrationError
from ouroboros.extension_ui_validation import validate_card_components, validate_ui_render
from ouroboros.tools.cards import _show_card
from ouroboros.tools.registry import ToolContext

ACTION = {"type": "action", "label": "Book it", "submit": {"id": "book"}}
FORM = {"type": "form", "title": "Dates", "fields": [{"name": "from"}, {"name": "to"}],
        "submit": {"id": "dates", "once": True}}


def _ctx(tmp_path, *, chat_id=1, live=True):
    ctx = ToolContext(repo_dir=tmp_path / "work", drive_root=tmp_path)
    (tmp_path / "work").mkdir(exist_ok=True)
    ctx.current_chat_id = chat_id
    ctx.task_id = "task-1"
    ctx.event_queue = queue.Queue() if live else None
    return ctx


def _events(ctx):
    events = []
    while ctx.event_queue is not None and not ctx.event_queue.empty():
        events.append(ctx.event_queue.get_nowait())
    return events + list(ctx.pending_events)


def _cards(tmp_path):
    folder = tmp_path / "state" / "cards"
    return sorted(p.name for p in folder.glob("card_*.json")) if folder.exists() else []


# --- the CARD mode of the declarative validator -------------------------------------------

def test_a_card_tree_indexes_its_commit_controls_with_their_drawn_labels():
    checked = validate_card_components([
        {"type": "markdown", "text": "Two flights"},
        {"type": "group", "layout": "grid", "components": [ACTION, FORM]},
        {"type": "table", "path": "rows", "columns": [{"path": "time", "label": "Time"}]},
    ])
    assert checked["actions"] == {
        "book": {"kind": "action", "once": False, "label": "Book it"},
        "dates": {"kind": "form", "once": True, "label": "Dates", "fields": ["from", "to"]},
    }
    assert checked["components"][1]["components"][0] == ACTION


@pytest.mark.parametrize(("tree", "message"), [
    ([{"type": "poll", "route": "x"}], "components[0] type 'poll' is not available in a card"),
    ([{"type": "subscription", "event": "tick"}], "components[0] type 'subscription' is not available"),
    ([{"type": "file", "src": "data:x"}], "components[0] type 'file' is not available"),
    ([{"type": "action", "label": "Go"}], "components[0].submit.id is required"),
    ([{"type": "action", "label": "Go", "submit": {"id": "Bad Id"}}], "components[0].submit.id is required"),
    ([{"type": "group", "components": [{"type": "action", "route": "r", "submit": {"id": "a"}}]}],
     "components[0].components[0].route is not available in a card"),
    ([ACTION, {**FORM, "submit": {"id": "book"}}], "components[1].submit.id duplicates 'book' first used at components[0]"),
    ([{"type": "markdown", "text": "x", "submit": {"id": "a"}}], "components[0].submit binds only action and form"),
    ([{"type": "action", "submit": {"id": "a", "once": "yes"}}], "components[0].submit.once must be boolean"),
    ([{"type": "gallery", "items": [{"type": "file", "src": "data:x"}]}], "components[0].items[0].type 'file'"),
    ([{"type": "kanban", "columns": [{"id": "a"}], "on_move": {"route": "m"}}], "components[0].on_move is not available"),
    ([], "components must be a non-empty list"),
])
def test_a_bad_card_tree_is_refused_at_its_exact_path(tree, message):
    with pytest.raises(ExtensionRegistrationError) as refused:
        validate_card_components(tree)
    assert message in str(refused.value)


def test_widget_validation_is_unchanged_outside_card_mode():
    """The same catalog still demands a route for a widget's action, and accepts one."""
    with pytest.raises(ExtensionRegistrationError, match="requires route or api_route"):
        validate_ui_render({"kind": "declarative", "components": [{"type": "action", "label": "Go"}]})
    clean = validate_ui_render({"kind": "declarative", "components": [{"type": "action", "route": "go"}]})
    assert clean["components"][0]["route"] == "go"


# --- the tool -------------------------------------------------------------------------------

def test_a_component_card_is_stored_before_its_event_leaves_the_task(tmp_path):
    ctx = _ctx(tmp_path)
    receipt = json.loads(_show_card(ctx, title="Flights", fallback_text="A 9:00 or B 12:00",
                                    components=[ACTION, {"type": "table", "path": "rows", "columns": [{"path": "t"}]}],
                                    data={"rows": [{"t": "9:00"}]}))
    assert receipt == {"card_id": receipt["card_id"], "revision": 1, "shown": True}
    card = load_card(tmp_path, receipt["card_id"])
    assert card["kind"] == "components" and card["chat_id"] == 1 and card["task_id"] == "task-1"
    assert card["data"] == {"rows": [{"t": "9:00"}]} and card["submissions"] == [] and card["revision"] == 1
    [event] = _events(ctx)
    assert event["type"] == "send_card" and event["card_id"] == receipt["card_id"]
    assert event["interactive"] is True and event["replace"] is False
    assert event["fallback_text"] == "A 9:00 or B 12:00" and event["task_id"] == "task-1"


def test_a_display_only_card_is_not_interactive_and_a_queued_card_says_so(tmp_path):
    ctx = _ctx(tmp_path, live=False)
    receipt = json.loads(_show_card(ctx, title="Status", fallback_text="All green",
                                    components=[{"type": "metric", "label": "Uptime", "value": "99.9"}]))
    assert receipt["shown"] is False and receipt["queued"] is True
    [event] = _events(ctx)
    assert event["interactive"] is False


@pytest.mark.parametrize(("kwargs", "fragment"), [
    ({}, "give exactly one body"),
    ({"components": [ACTION], "app": {"html": "<p>x</p>"}}, "give exactly one body"),
    ({"widget": {"skill": "s", "tab_id": "t"}, "data": {}}, "data belongs to a components body"),
    ({"components": [ACTION], "data": [1]}, "data must be an object"),
    ({"components": [{"type": "stream", "route": "s"}]}, "components[0] type 'stream'"),
    ({"app": {"path": "a.html", "html": "<p/>"}}, "exactly one of path or html"),
    ({"app": {"path": "missing.html"}}, "app file not found"),
    ({"widget": {"skill": "s"}}, "widget needs skill and tab_id"),
])
def test_a_refused_card_names_the_problem_and_stores_nothing(tmp_path, kwargs, fragment):
    ctx = _ctx(tmp_path)
    text = _show_card(ctx, title="T", fallback_text="plain", **kwargs)
    assert text.startswith("⚠️ SHOW_CARD_ARG_ERROR") and fragment in text and "The card was not shown." in text
    assert _cards(tmp_path) == [] and _events(ctx) == []


def test_title_and_fallback_are_required_and_bounded(tmp_path):
    text = _show_card(_ctx(tmp_path), title="x" * 121, fallback_text="", components=[ACTION])
    assert "title is required text of at most 120" in text and "fallback_text is required" in text
    assert _cards(tmp_path) == []
    ok = _show_card(_ctx(tmp_path), title="x" * 120, fallback_text="y" * 4000, components=[ACTION])
    assert json.loads(ok)["revision"] == 1


def test_no_chat_means_no_card(tmp_path):
    text = _show_card(_ctx(tmp_path, chat_id=None), title="T", fallback_text="p", components=[ACTION])
    assert "No active chat" in text and _cards(tmp_path) == []


def test_a_widget_card_needs_a_live_tab_and_teaches_the_live_ones(tmp_path, monkeypatch):
    import ouroboros.extension_loader as loader

    monkeypatch.setattr(loader, "live_widget_projection",
                        lambda skill: None if skill != "weather" else [{"tab": {"tab_id": "now"}, "revision": "r"}])
    refused = _show_card(_ctx(tmp_path), title="T", fallback_text="p", widget={"skill": "weather", "tab_id": "week"})
    assert "skill 'weather' has no live widget tab 'week' (live tabs: now)" in refused
    gone = _show_card(_ctx(tmp_path), title="T", fallback_text="p", widget={"skill": "radio", "tab_id": "x"})
    assert "skill 'radio' is not live" in gone and _cards(tmp_path) == []
    ctx = _ctx(tmp_path)
    receipt = json.loads(_show_card(ctx, title="Weather", fallback_text="Sunny",
                                    widget={"skill": "weather", "tab_id": "now", "input": {"city": "Rome"}}))
    card = load_card(tmp_path, receipt["card_id"])
    assert card["kind"] == "widget" and card["widget"] == {"skill": "weather", "tab_id": "now", "input": {"city": "Rome"}}
    assert _events(ctx)[0]["interactive"] is True


def test_an_app_is_copied_content_addressed_from_the_workspace_or_inline(tmp_path):
    ctx = _ctx(tmp_path)
    page = "<!doctype html><button>Roll</button>"
    (tmp_path / "work" / "dice.html").write_text(page, encoding="utf-8")
    from_file = load_card(tmp_path, json.loads(_show_card(ctx, title="Dice", fallback_text="A dice roller",
                                                          app={"path": "dice.html"}))["card_id"])
    inline = load_card(tmp_path, json.loads(_show_card(ctx, title="Dice", fallback_text="A dice roller",
                                                       app={"html": page}))["card_id"])
    assert from_file["kind"] == "app" and from_file["app"] == inline["app"]
    assert from_file["app"]["size"] == len(page.encode()) and read_app_bytes(tmp_path, from_file["app"]["sha256"]) == page.encode()
    assert len(list((tmp_path / "state" / "cards" / "apps").iterdir())) == 1


def test_an_oversized_or_binary_app_is_refused(tmp_path):
    ctx = _ctx(tmp_path)
    (tmp_path / "work" / "big.html").write_bytes(b"x" * (MAX_APP_BYTES + 1))
    (tmp_path / "work" / "bin.html").write_bytes(b"\xff\xfe\x00")
    assert "limit 524288" in _show_card(ctx, title="T", fallback_text="p", app={"path": "big.html"})
    assert "not UTF-8" in _show_card(ctx, title="T", fallback_text="p", app={"path": "bin.html"})
    assert "limit 524288" in _show_card(ctx, title="T", fallback_text="p", app={"html": "x" * (MAX_APP_BYTES + 1)})
    assert _cards(tmp_path) == []


def test_replace_updates_a_card_of_the_same_chat_in_place(tmp_path):
    ctx = _ctx(tmp_path)
    card_id = json.loads(_show_card(ctx, title="Vote", fallback_text="v1", components=[ACTION]))["card_id"]
    record_submission(tmp_path, card_id, request_id="r1", action_id="book", label="Book it", values=None, once=False, revision=1)
    _events(ctx)
    receipt = json.loads(_show_card(ctx, title="Vote (closed)", fallback_text="v2",
                                    components=[{"type": "callout", "text": "Closed"}], replace=card_id))
    assert receipt["card_id"] == card_id and receipt["revision"] == 2
    card = load_card(tmp_path, card_id)
    assert card["title"] == "Vote (closed)" and card["components"] == [{"type": "callout", "text": "Closed"}]
    assert [row["request_id"] for row in card["submissions"]] == ["r1"]  # presses survive a new body
    [event] = _events(ctx)
    assert event["replace"] is True and event["revision"] == 2
    other = _show_card(_ctx(tmp_path, chat_id=2), title="T", fallback_text="p", components=[ACTION], replace=card_id)
    assert other.startswith("⚠️ CARD_REPLACE_OTHER_CHAT") and load_card(tmp_path, card_id)["revision"] == 2
    unknown = _show_card(ctx, title="T", fallback_text="p", components=[ACTION], replace="card_0000000000000000")
    assert unknown.startswith("⚠️ CARD_REPLACE_NOT_FOUND")


# --- the store ------------------------------------------------------------------------------

def test_presses_are_idempotent_per_request_and_once_keeps_the_first(tmp_path):
    card = write_card(tmp_path, chat_id=1, task_id="t", title="T", fallback_text="p", kind="components",
                      body={"components": [ACTION]})
    first = record_submission(tmp_path, card["card_id"], request_id="r1", action_id="dates", label="Dates",
                              values={"from": "Mon"}, once=True, revision=1)
    assert first["duplicate"] is False and first["submission"]["delivered"] == "owed"
    again = record_submission(tmp_path, card["card_id"], request_id="r1", action_id="dates", label="Dates",
                              values={"from": "Tue"}, once=True, revision=1)
    assert again["duplicate"] is True and again["submission"]["values"] == {"from": "Mon"}
    later = record_submission(tmp_path, card["card_id"], request_id="r2", action_id="dates", label="Dates",
                              values=None, once=True, revision=1)
    assert later["winner"]["request_id"] == "r1" and "submission" not in later
    many = record_submission(tmp_path, card["card_id"], request_id="r3", action_id="book", label="Book it",
                             values=None, once=False, revision=1)
    assert [row["action_id"] for row in submissions_summary(many["record"])] == ["dates", "book"]
    with pytest.raises(CardStoreError):
        record_submission(tmp_path, "card_ffffffffffffffff", request_id="r", action_id="a", label="a",
                          values=None, once=False, revision=1)


def test_the_oldest_presses_are_dropped_and_counted_beyond_the_bound(tmp_path):
    card = write_card(tmp_path, chat_id=1, task_id="t", title="T", fallback_text="p", kind="components",
                      body={"components": [ACTION]})
    for index in range(MAX_SUBMISSIONS + 3):
        outcome = record_submission(tmp_path, card["card_id"], request_id=f"r{index}", action_id="book",
                                    label="Book it", values=None, once=False, revision=1)
    stored = outcome["record"]
    assert len(stored["submissions"]) == MAX_SUBMISSIONS and stored["submissions_dropped"] == 3
    assert stored["submissions"][0]["request_id"] == "r3"


def test_a_malformed_card_id_never_reaches_the_filesystem(tmp_path):
    assert load_card(tmp_path, "../state/state") is None
    assert load_card(tmp_path, "card_ABC") is None
    assert read_app_bytes(tmp_path, "../x") is None


def test_a_work_turned_into_a_project_replaces_its_own_card_there_and_no_other_room_can(tmp_path):
    """A card a work showed in Main lives in its Project once the work moves there (its row does too):
    the same work, readmitted in the Project, may replace it; a work of an unrelated Project may not."""
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    nova, other = create_project(tmp_path, "nova", name="Nova"), create_project(tmp_path, "other", name="Other")
    write_task_result(tmp_path, "task-1", "running", chat_id=1, title="task-1")
    card_id = json.loads(_show_card(_ctx(tmp_path), title="Plan", fallback_text="v1",
                                    components=[{"type": "markdown", "text": "v1"}]))["card_id"]
    bind_task_to_project(tmp_path, "task-1", nova["id"], nova["chat_id"], origin={"absent": "system"})
    moved = json.loads(_show_card(_ctx(tmp_path, chat_id=nova["chat_id"]), title="Plan", fallback_text="v2",
                                  components=[{"type": "markdown", "text": "v2"}], replace=card_id))
    assert moved["revision"] == 2 and load_card(tmp_path, card_id)["fallback_text"] == "v2"
    stranger = _ctx(tmp_path, chat_id=other["chat_id"])
    stranger.task_id = "stranger"
    write_task_result(tmp_path, "stranger", "running", chat_id=other["chat_id"], title="stranger")
    refused = _show_card(stranger, title="Plan", fallback_text="hijack", components=[{"type": "markdown", "text": "x"}],
                         replace=card_id)
    assert refused.startswith("⚠️ CARD_REPLACE_OTHER_CHAT") and load_card(tmp_path, card_id)["revision"] == 2
