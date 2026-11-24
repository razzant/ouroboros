"""Converting a task into a Project adds a read lens without copying dialogue.

- the owner's ORIGINAL canonical row is projected into the Project room;
- a subagent's progress classifies into the root project's thread by lineage;
- the main chat keeps one owner row and no physical mirror bubble.
"""

from __future__ import annotations

from ouroboros import dialogue_store
from tests import _dialogue as dialogue_test

import asyncio
import json
from types import SimpleNamespace

import pytest


def _request(tmp_path, body):
    async def _json():
        return body

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
        json=_json,
    )


def _history(tmp_path, chat_id):
    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    resp = asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": str(chat_id), "limit": "50"})))
    return json.loads(resp.body.decode("utf-8"))["messages"]


def test_conversion_projects_canonical_owner_request_and_rehomes_subagents(tmp_path, monkeypatch):
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import project_binding_for_task, project_chat_for_task
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    dialogue_store.append(
        tmp_path, "chat",
        # The owner's original request lives in the MAIN chat (chat_id 1). It is
        # logged at receive time with NO task_id. The immutable binding stores a
        # source reference so the Project lens can render this same row.
        {"ts": "2026-06-18T00:00:00Z", "direction": "in", "chat_id": 1,
         "client_message_id": "owner-1", "text": "build cyberpunk racing"},
    )
    # v6.73.0: the origin identity is captured at ingress and persisted on the
    # TASK RECORD; post-hoc conversion reads it from there (never a content lookup).
    write_task_result(
        tmp_path, "root", STATUS_RUNNING,
        objective="build cyberpunk racing",
        origin_message_ref=build_owner_message_ref(
            chat_id=1, client_message_id="owner-1",
            ts="2026-06-18T00:00:00Z", text="build cyberpunk racing",
        ),
        origin_message_text="build cyberpunk racing",
    )

    async def coin(text, **kwargs):
        assert text == "build cyberpunk racing"
        return "Cyber Racing Game"

    monkeypatch.setattr(project_naming, "llm_project_name_async", coin)
    resp = asyncio.run(api_project_from_task(_request(
        tmp_path,
        {"task_id": "root", "id": "task-root", "objective_hint": "Not the original request"},
    )))
    payload = json.loads(resp.body.decode("utf-8"))
    proj_chat = int(payload["project"]["chat_id"])
    assert proj_chat == project_chat_for_task(tmp_path, "root") > 0
    binding = project_binding_for_task(tmp_path, "root")
    assert binding["source_ref"]["client_message_id"] == "owner-1"
    assert binding["source_text"] == "build cyberpunk racing"

    # A subagent emits progress; its rows carry main chat_id but lineage roots at "root".
    dialogue_store.append(tmp_path, "progress", {
        "ts": "2026-06-18T00:01:00Z", "content": "child working",
        "chat_id": 1, "task_id": "child", "parent_task_id": "root",
        "root_task_id": "root", "subagent_event": "progress",
    })

    project_view = _history(tmp_path, proj_chat)
    # The canonical owner request is the first Project message...
    assert any(m.get("role") == "user" and "cyberpunk racing" in m.get("text", "") for m in project_view)
    # ...and the subagent progress re-homes into the project thread by lineage (C4.4).
    assert any(m.get("task_id") == "child" for m in project_view)
    # BUG 1 (ordering): the owner's row sorts to the TOP using its ORIGINAL send ts
    # (00:00:00), ahead of the working bubble (00:01:00) — not stamped 'now' at the
    # bottom. History replay sorts purely by ts.
    owner_idx = next(i for i, m in enumerate(project_view)
                     if m.get("role") == "user" and "cyberpunk racing" in m.get("text", ""))
    child_idx = next(i for i, m in enumerate(project_view) if m.get("task_id") == "child")
    assert owner_idx < child_idx, f"owner row must precede the working bubble (got {owner_idx} vs {child_idx})"
    assert str(project_view[owner_idx].get("ts", "")).startswith("2026-06-18T00:00:00"), \
        project_view[owner_idx].get("ts")

    main_view = _history(tmp_path, 1)
    # Main still renders that same canonical row exactly once.
    owner_user_rows = [m for m in main_view if m.get("role") == "user" and "cyberpunk racing" in m.get("text", "")]
    assert len(owner_user_rows) == 1
    raw_rows = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    assert len([row for row in raw_rows if "cyberpunk racing" in row.get("text", "")]) == 1
    # The subagent's raw chat stays out of main; only sanitized progress may mirror.
    assert all(not (m.get("task_id") == "child" and m.get("role") == "user") for m in main_view)


def test_repeat_conversion_does_not_duplicate_owner_message(tmp_path):
    """A repeated conversion keeps one canonical row and one immutable ref."""
    from ouroboros.gateway.projects import api_project_from_task

    dialogue_store.append(
        tmp_path, "chat", {
            "ts": "2026-06-18T00:00:00Z",
            "direction": "in",
            "chat_id": 1,
            "client_message_id": "owner-repeat",
            "text": "build cyberpunk racing",
        },
    )

    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(
        tmp_path, "root", STATUS_RUNNING,
        objective="build cyberpunk racing",
        origin_message_ref=build_owner_message_ref(
            chat_id=1, client_message_id="owner-repeat",
            ts="2026-06-18T00:00:00Z", text="build cyberpunk racing",
        ),
        origin_message_text="build cyberpunk racing",
    )
    body = {"task_id": "root", "id": "task-root", "name": "Racing project"}
    first = asyncio.run(api_project_from_task(_request(tmp_path, body)))
    repeat = asyncio.run(api_project_from_task(_request(tmp_path, {**body, "name": ""})))
    assert first.status_code == repeat.status_code == 200
    assert json.loads(first.body)["handoff_id"] == json.loads(repeat.body)["handoff_id"]

    rows = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    canonical = [r for r in rows if r.get("direction") == "in" and "cyberpunk racing" in r.get("text", "")]
    assert len(canonical) == 1


def test_conversion_reuses_proactive_suggested_name(tmp_path, monkeypatch):
    """The producer attests the actual title, not a reusable flag on the slot."""
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(
        tmp_path, "root", STATUS_RUNNING,
        suggested_name="Cyber Racing Game",
        model_coined_name="Cyber Racing Game",
        objective="build a cyberpunk racing game",
    )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("a proven model title needs no extra Light call")

    monkeypatch.setattr(project_naming, "llm_project_name_async", forbidden)
    resp = asyncio.run(api_project_from_task(_request(
        tmp_path, {"task_id": "root", "id": "task-root", "objective_hint": "build a cyberpunk racing game"},
    )))
    payload = json.loads(resp.body.decode("utf-8"))
    assert payload["project"]["name"] == "Cyber Racing Game", payload["project"]["name"]


def test_naming_refusal_and_later_click_keep_one_terminal_origin(tmp_path, monkeypatch):
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import get_project, project_binding_for_task
    from ouroboros.task_results import write_task_result

    text = "Map the two tributaries\nand label their source."
    row = {"chat_id": 1, "client_message_id": "origin", "ts": "2026-06-18T00:00:00Z", "text": text}
    dialogue_store.append(tmp_path, "chat", {**row, "direction": "in"})
    write_task_result(tmp_path, "root", "completed", result="done",
                      origin_message_ref=build_owner_message_ref(**row), origin_message_text=text,
                      suggested_name="Map the two tributaries")
    calls = []

    async def coin(received, **kwargs):
        calls.append(received)
        return "" if len(calls) == 1 else "Tributary map"

    monkeypatch.setattr(project_naming, "llm_project_name_async", coin)
    body = {"task_id": "root", "objective_hint": "Ignore the tributaries"}
    first = asyncio.run(api_project_from_task(_request(tmp_path, body)))
    assert first.status_code == 409 and json.loads(first.body)["reason_code"] == "project_name_unavailable"
    assert calls == [text]
    assert get_project(tmp_path, "task-root") is None and project_binding_for_task(tmp_path, "root") is None
    assert sum(m.get("text") == text for m in _history(tmp_path, 1)) == 1
    second = asyncio.run(api_project_from_task(_request(tmp_path, body)))
    assert second.status_code == 200 and calls == [text, text]
    project = json.loads(second.body)["project"]
    assert project["name"] == "Tributary map"
    assert sum(m.get("text") == text for m in _history(tmp_path, int(project["chat_id"]))) == 1
    assert sum(m.get("text") == text for m in _history(tmp_path, 1)) == 1


@pytest.mark.parametrize("same_destination", [True, False])
def test_automatic_name_that_finishes_after_another_binding_cannot_rename_it(tmp_path, monkeypatch, same_destination):
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import get_project
    from ouroboros.task_results import write_task_result

    row = {"chat_id": 1, "client_message_id": "start", "ts": "2026-06-18T00:00:00Z", "text": "Sketch a garden"}

    def interleave(tid, late_name):
        write_task_result(tmp_path, tid, "running", origin_message_ref=build_owner_message_ref(**row),
                          origin_message_text=row["text"])

        async def coin(_text, **_kwargs):
            won = await api_project_from_task(_request(tmp_path, {"task_id": tid, "name": "Owner's room"}))
            assert won.status_code == 200
            return late_name

        monkeypatch.setattr(project_naming, "llm_project_name_async", coin)
        requested_id = f"task-{tid}" if same_destination else f"alternate-{tid}"
        reply = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": tid, "id": requested_id})))
        payload = json.loads(reply.body)
        assert reply.status_code == 200 and payload["adopted"] is True, payload
        assert payload["binding"]["project_id"] == f"task-{tid}"
        assert get_project(tmp_path, f"alternate-{tid}") is None
        assert get_project(tmp_path, f"task-{tid}")["name"] == "Owner's room"

    interleave("root", "")
    interleave("second", "Garden sketch")


def test_conversion_adoption_checks_the_winning_bindings_admission(tmp_path, monkeypatch):
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import project_binding_for_task, update_project
    from ouroboros.task_results import write_task_result

    text = "Sketch the coastal observatory."
    ref = build_owner_message_ref(chat_id=1, client_message_id="start", ts="2026-06-18T00:00:00Z", text=text)
    write_task_result(tmp_path, "root", "running", origin_message_ref=ref, origin_message_text=text)
    new_folder = tmp_path / "new-folder"
    new_folder.mkdir()
    won = {}

    async def coin(_text, **_kwargs):
        reply = await api_project_from_task(_request(tmp_path, {"task_id": "root", "name": "Observatory"}))
        assert reply.status_code == 200
        won.update(project_binding_for_task(tmp_path, "root"))
        update_project(tmp_path, "task-root", working_dir=str(new_folder))
        return "Different observatory"

    monkeypatch.setattr(project_naming, "llm_project_name_async", coin)
    reply = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": "root"})))
    assert reply.status_code == 409 and json.loads(reply.body)["reason_code"] == "project_routing_fence_changed"
    # The exact task's own admission is the comparison: it is neither replaced by a fresh read nor rebound.
    assert project_binding_for_task(tmp_path, "root") == won and won["placement_admission"]["project"]["working_dir"] == ""


@pytest.mark.parametrize("authored", [True, False])
def test_continue_conversion_reuses_the_work_title_not_its_wrapper(tmp_path, monkeypatch, authored):
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    title = "Documenting how the tidal observatory estimates tomorrow's coastal conditions"
    fields = {"title": title, "suggested_name": title} if authored else {
        "title": None, "suggested_name": "Coastal observatory", "model_coined_name": "Coastal observatory"}
    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, "first", **fields)
    ack = admit_continuation("first", action_nonce=NONCE)
    assert ack["ok"], ack
    successor = ack["successor_task_id"]
    assert load_task_result(tmp_path, successor)["title"].startswith("Continue: ")

    async def forbidden(*_args, **_kwargs):
        pytest.fail("The first work's title makes another name call unnecessary")

    monkeypatch.setattr(project_naming, "llm_project_name_async", forbidden)
    result = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": successor})))
    assert result.status_code == 200, result.body
    assert json.loads(result.body)["project"]["name"] == fields["suggested_name"]


@pytest.mark.parametrize("kind", ["api", "unaccepted-api", "late-answer", "saved-press", "unwitnessed"])
def test_conversion_uses_the_verified_opening_of_this_episode(tmp_path, monkeypatch, kind):
    from ouroboros import card_store, project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result

    text = "  Include the two northern piers."
    fields = {"objective": "A different objective"}
    expected = text
    if kind in ("api", "unaccepted-api"):
        fields.update(description=text, metadata={"client_surface": {"channel": "api_task"}},
                      api_admission={"status": "accepted" if kind == "api" else "requested", "token": "admission"})
        fields.update(context="The subject is the harbor redesign.", expected_output="A plan", constraints="Avoid red ink")
        expected = json.dumps({key: fields[key] for key in ("description", "context", "expected_output", "constraints")})
    elif kind == "late-answer":
        fields["metadata"] = {"late_answer": {"task_id": "asking-task", "quiz_id": "missing-quiz"}}
    elif kind == "saved-press":
        text = "Make the map\nlegend: Ports"
        fields["metadata"] = {"card_press": {"task_id": "asking-task", "card_id": "card_1234567890abcdef",
                                            "request_id": "accepted-press", "submission": {
                                                "title": "Harbor access", "label": "Make the map",
                                                "values": {"legend": "Ports"}}}}
        expected = f"Harbor access\n{text}"

        def not_a_live_card(*_args):
            pytest.fail("The accepted submission outlives the presentation card")

        monkeypatch.setattr(card_store, "load_card", not_a_live_card)
    if kind in ("late-answer", "saved-press"):
        fields.update(origin_message_ref=build_owner_message_ref(
            chat_id=1, client_message_id="owner-start", ts="2026-06-18T00:00:00Z", text=text),
            origin_message_text=text)
    write_task_result(tmp_path, "root", "completed", **fields)
    calls = []

    async def coin(sent, **_kwargs):
        calls.append(sent)
        return "Northern piers"

    monkeypatch.setattr(project_naming, "llm_project_name_async", coin)
    result = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": "root", "objective_hint": "Do not use me"})))
    if kind in ("unaccepted-api", "unwitnessed"):
        assert result.status_code == 409 and not calls, result.body
        assert json.loads(result.body)["reason_code"] == "project_name_source_unavailable"
    else:
        assert result.status_code == 200 and calls == [expected], result.body
        assert json.loads(result.body)["project"]["name"] == "Northern piers"


def test_live_conversion_uses_ingress_not_live_progress(tmp_path, monkeypatch):
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor import queue, workers

    text = "Survey the street trees"
    ref = build_owner_message_ref(chat_id=1, client_message_id="owner-live", ts="2026-06-18T00:00:00Z", text=text)
    task = {"id": "live", "chat_id": 1, "origin_message_ref": ref, "origin_message_text": text,
            "objective": "Status: counting blocks", "workspace_root": "original", "project_id": ""}
    running = {"live": {"task": task}}
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(workers, "RUNNING", running)
    calls = []

    async def coin(received, **kwargs):
        calls.append(received)
        return "Street tree survey"

    monkeypatch.setattr(project_naming, "llm_project_name_async", coin)
    response = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": "live"})))
    assert response.status_code == 200 and calls == [text]
    assert json.loads(response.body)["project"]["name"] == "Street tree survey"
    assert task["workspace_root"] == "original" and task["project_id"] == ""


@pytest.mark.parametrize("uploads, expected", [
    ([], ""), ([{}], ""), ([{"name": "design.pdf", "unavailable": "too_large"}], ""),
    ([{"name": "design.pdf", "upload": "fixture"}], '(file attached)\nAttached filenames: ["design.pdf"]'),
])
def test_namer_uses_only_witnessed_attachment_placeholders(tmp_path, uploads, expected):
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.project_naming import captured_project_request

    placeholder = "(file attached)"
    ref = build_owner_message_ref(chat_id=1, client_message_id="files", ts="2026-06-18T00:00:00Z", text=placeholder)
    sources = [{"origin_message_ref": ref, "metadata": {
        "origin_message_snapshot": {"attachments": uploads, "placeholder_text": placeholder}}}]
    origin = _owner_task_origin(tmp_path, "files", sources=sources)
    assert captured_project_request(tmp_path, "files", sources, origin) == expected


def test_a_legacy_continue_without_an_admission_record_does_not_supply_a_title(tmp_path, monkeypatch):
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "legacy", "completed", title="Continue: Old harbor work",
                      metadata={"continuation": {"predecessor_task_id": "retired"}})

    async def forbidden(*_args, **_kwargs):
        pytest.fail("The original opening is unavailable, so no request should be sent")

    monkeypatch.setattr(project_naming, "llm_project_name_async", forbidden)
    reply = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": "legacy"})))
    assert reply.status_code == 409 and json.loads(reply.body)["reason_code"] == "project_name_source_unavailable"


def test_file_only_opening_uses_the_recovered_canonical_row(tmp_path, monkeypatch):
    from ouroboros.chat_uploads import store_upload
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros import project_naming
    from ouroboros.task_results import write_task_result

    _, upload = store_upload(b"Synthetic list; its bytes must not be sent", "harbors.txt", data_dir=tmp_path)
    row = {"ts": "2026-06-18T00:00:00Z", "direction": "in", "chat_id": 1, "client_message_id": "file-1",
           "text": "(file attached)", "text_placeholder": True, "attachments": [upload]}
    dialogue_store.append(tmp_path, "chat", row)
    ref = build_owner_message_ref(chat_id=1, client_message_id="file-1", ts=row["ts"], text=row["text"])
    write_task_result(tmp_path, "files", "completed", origin_message_ref=ref)
    sends = []

    async def name(text, **_kwargs):
        sends.append(text)
        return "Harbor list"

    monkeypatch.setattr(project_naming, "llm_project_name_async", name)
    reply = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": "files"})))
    assert reply.status_code == 200, reply.body
    assert sends == ['(file attached)\nAttached filenames: ["harbors.txt"]']


def test_late_input_admission_preserves_the_authored_title_without_another_call(tmp_path, monkeypatch):
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    from supervisor.worker_chat_lane import _seed_task

    text, title = "Add the harbor compass", "Map of the northern coast and its old breakwater for the harbor archive"
    ref = build_owner_message_ref(chat_id=1, client_message_id="late", ts="2026-06-18T00:00:00Z", text=text)
    admitted = _seed_task(1, text, None, None, {"title": title, "origin_message_ref": ref, "origin_message_text": text,
        "late_answer": {"task_id": "question-task", "quiz_id": "retired-quiz"}})
    write_task_result(tmp_path, admitted["id"], "completed", title=admitted["title"],
                      suggested_name=admitted["suggested_name"], origin_message_ref=ref, origin_message_text=text)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("An admitted title needs no naming call")

    monkeypatch.setattr(project_naming, "llm_project_name_async", forbidden)
    reply = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": admitted["id"]})))
    assert reply.status_code == 200 and json.loads(reply.body)["project"]["name"] == title


def test_default_conversion_joins_the_winning_episode_without_releasing_its_live_lease(tmp_path, monkeypatch):
    import copy
    from ouroboros import project_naming
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import list_sidebar_projects, project_binding_for_task
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    text = "Draw the coastline near the old harbor."
    ref = build_owner_message_ref(chat_id=1, client_message_id="request", ts="2026-06-18T00:00:00Z", text=text)
    _interrupted(tmp_path, "first", title=None, chat_id=1, origin_message_ref=ref, origin_message_text=text)
    ack = admit_continuation("first", action_nonce=NONCE)
    assert ack["ok"], ack
    successor = ack["successor_task_id"]
    running = workers.PENDING.pop()
    workers.RUNNING[successor] = {"task": running}
    lease = copy.deepcopy({key: running.get(key) for key in ("project_id", "workspace_root", "_project_admission")})
    seen = []

    async def convert(tid):
        response = await api_project_from_task(_request(tmp_path, {"task_id": tid}))
        assert response.status_code == 200, response.body
        return json.loads(response.body)

    async def coin(request, *, task_id, **_kwargs):
        seen.append((task_id, request))
        if task_id == "first":
            assert (await convert(successor))["project"]["name"] == "Old harbor map"
            return "Discard this later title"
        return "Old harbor map"

    monkeypatch.setattr(project_naming, "llm_project_name_async", coin)
    winner = asyncio.run(convert("first"))
    assert winner["adopted"] and winner["project"]["name"] == "Old harbor map"
    assert project_binding_for_task(tmp_path, successor)["project_id"] == winner["project"]["id"]
    assert len(list_sidebar_projects(tmp_path)) == 1
    assert asyncio.run(convert(successor))["handoff_id"] == asyncio.run(convert(successor))["handoff_id"]
    assert {key: running.get(key) for key in lease} == lease
    assert seen == [("first", text), (successor, text)]
