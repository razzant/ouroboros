"""``/api/state`` is one versioned display publication and ``/api/health`` a liveness read (cost laws L6, L12).

Equality laws, not timings: concurrent readers share one production; a cancelled reader starts nothing;
an unchanged revision serves the held publication without reading a task body; a real change produces
again; the health answer takes no executor hop and no store read; the client shape is unchanged.
"""
import asyncio
import json
import threading
import types

import pytest
from starlette.requests import Request

from ouroboros import store, task_store, usage_store
from ouroboros.gateway import state as gateway_state, state_publication as publication
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_hot_facts import _task

# The StateResponse fields /api/state answered before the publication (gateway/state.py at 1d88f843b).
STATE_FIELDS = {
    "uptime", "workers_alive", "workers_total", "pending_count", "running_count", "spent_usd", "budget_limit",
    "budget_pct", "branch", "sha", "evolution_enabled", "bg_consciousness_enabled", "state_quality",
    "evolution_cycle", "evolution_state", "bg_consciousness_state", "spent_calls", "supervisor_ready",
    "supervisor_error", "boot", "runtime_mode", "context_mode", "subagents_mode", "subagents_availability",
    "context_mode_auto_low", "effort_range", "safety_mode", "skills_repo_configured", "github_token_configured",
    "accounting", "projects", "project_messages_revision", "project_listable_count", "project_chat_ids",
    "main_visible_revision", "task_bindings", "task_bindings_revision", "task_bindings_unchanged", "active_direct_turns", "active_chat_activities",
    "active_chat_activities_complete", "current_questions", "current_questions_complete", "current_questions_gaps",
    "main_attention", "main_attention_complete", "main_attention_gaps", "works", "rooms", "work_summary",
    "works_complete", "works_gaps", "served_ui",
} | {"main_route"}  # Main chat writes Main itself


@pytest.fixture
def serving(tmp_path, monkeypatch):
    from ouroboros import config, utils
    from supervisor import queue as queue_mod
    from supervisor.active_activity import get_direct_activity_registry

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    usage_store.migrate_from_journal(root)
    queue, _state, workers = _install_queue(root, monkeypatch)
    monkeypatch.setattr(config, "REPO_DIR", tmp_path / "no-checkout")
    monkeypatch.setattr(queue_mod, "get_evolution_status_snapshot", lambda **_kwargs: {})
    # The files this test writes are seconds old; the racy-stamp guard is about coarse clocks, not APFS.
    monkeypatch.setattr(utils, "_RACY_NS", 0)
    get_direct_activity_registry().clear()
    app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=root, app_start=0.0))
    yield types.SimpleNamespace(root=root, app=app, queue=queue, workers=workers)
    get_direct_activity_registry().clear()


def _request(app, query=b""):
    return Request({"type": "http", "method": "GET", "path": "/api/state", "headers": [], "query_string": query,
                    "scheme": "http", "server": ("test", 80), "client": ("test", 1), "app": app})


def _get(app, query=b""):
    return asyncio.run(gateway_state.api_state(_request(app, query)))


def _payload(response):
    def no_duplicates(pairs):
        keys = [key for key, _ in pairs]
        assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
        return dict(pairs)

    assert response.status_code == 200, response.body
    return json.loads(response.body, object_pairs_hook=no_duplicates)


def _counting(monkeypatch, target, name):
    calls, real = [], getattr(target, name)

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, counted)
    return calls


def _gated_production(monkeypatch):
    calls, gate, real = [], threading.Event(), gateway_state._state_snapshot

    def gated(request, direct=None):
        calls.append(1)
        assert gate.wait(30), "the test never released the production"
        return real(request, direct=direct)

    monkeypatch.setattr(gateway_state, "_state_snapshot", gated)
    return calls, gate


def test_the_answer_keeps_the_client_shape_and_names_its_generation(serving):
    response = _get(serving.app)
    payload = _payload(response)
    assert set(payload) == STATE_FIELDS
    assert response.headers["cache-control"] == "no-store"
    assert response.headers[publication.GENERATION_HEADER] == "1"
    assert float(response.headers[publication.AGE_HEADER]) >= 0.0
    assert isinstance(payload["task_bindings"], dict) and payload["task_bindings_unchanged"] is False
    assert payload["boot"]["phase"] in {"starting", "serving", "ready", "failed"}


def test_state_keeps_the_sidebar_feed_capped_and_reports_the_full_listable_count(serving, monkeypatch):
    """The feed and its count come from one registry read: a later read here would see one more room, as when another
    writer adds one between two reads, and the count would then claim a room the feed's own read never saw."""
    from ouroboros import projects_registry as registry

    for index in range(60):
        registry.create_project(serving.root, f"project-{index:03d}", name=f"Project {index:03d}", origin="owner_ui")
    reads, real = [], registry.list_sidebar_projects

    def spliced(drive_root):
        rows = real(drive_root)
        reads.append(len(rows))
        return rows if len(reads) == 1 else [*rows, {**rows[0], "id": "created-between-reads", "chat_id": 99_999}]

    monkeypatch.setattr(registry, "list_sidebar_projects", spliced)
    payload = _payload(_get(serving.app))
    assert reads == [60], "the state production read the registry's listable rows more than once"
    assert len(payload["projects"]) == 50
    assert payload["project_listable_count"] == 60


def test_a_room_that_is_the_owners_turn_always_has_its_row_in_the_feed(serving, monkeypatch):
    """The Projects header's amber stands for a row the owner can reach: a listable Project whose room is the owner's
    turn joins the feed beyond its recent slice, in message order after the pins. Without such a room the feed is the
    capped one, byte for byte, also when the amber room is already in it."""
    from ouroboros import projects_registry as registry
    from ouroboros.gateway import work_projection

    for index in range(60):
        registry.create_project(serving.root, f"project-{index:03d}", name=f"Project {index:03d}", origin="owner_ui")
    capped = gateway_state._state_snapshot(_request(serving.app))["projects"]
    everything = registry.projects_summary(serving.root, limit=None, by_message=True)
    [beyond] = [row for row in everything if row["id"] not in {row["id"] for row in capped}][:1]
    real = work_projection.works_snapshot
    amber = []

    def with_amber(*args, **kwargs):
        works = real(*args, **kwargs)
        return {**works, "rooms": {**works["rooms"], **{str(chat): {"amber": True, "turn": [{}]} for chat in amber}}}

    monkeypatch.setattr(work_projection, "works_snapshot", with_amber)
    amber[:] = [capped[0]["chat_id"]]
    assert gateway_state._state_snapshot(_request(serving.app))["projects"] == capped
    amber[:] = [beyond["chat_id"]]
    fed = gateway_state._state_snapshot(_request(serving.app))["projects"]
    assert len(fed) == len(capped) + 1 and beyond in fed
    assert [row["id"] for row in fed] == [row["id"] for row in everything if row in capped or row == beyond]


def test_concurrent_readers_share_one_production_even_when_nothing_could_be_reused(serving, monkeypatch):
    real_probe = publication.probe
    monkeypatch.setattr(publication, "probe", lambda *a, **k: (*real_probe(*a, **k), object()))  # never equal
    calls, gate = _gated_production(monkeypatch)

    async def scenario():
        readers = [asyncio.ensure_future(gateway_state.api_state(_request(serving.app))) for _ in range(5)]
        await asyncio.sleep(0.2)  # every reader now waits on the one refresh, blocked inside its production
        gate.set()
        return await asyncio.gather(*readers)

    responses = asyncio.run(scenario())
    assert len(calls) == 1, "five concurrent readers ran five productions"
    assert {response.headers[publication.GENERATION_HEADER] for response in responses} == {"1"}
    assert len({json.dumps({k: v for k, v in _payload(r).items() if k != "uptime"}, sort_keys=True)
                for r in responses}) == 1


def test_a_cancelled_reader_starts_nothing_and_its_refresh_serves_the_next(serving, monkeypatch):
    calls, gate = _gated_production(monkeypatch)

    async def scenario():
        first = asyncio.ensure_future(gateway_state.api_state(_request(serving.app)))
        await asyncio.sleep(0.2)  # its refresh is in flight
        first.cancel()
        second = asyncio.ensure_future(gateway_state.api_state(_request(serving.app)))
        await asyncio.sleep(0.2)
        assert len(calls) == 1, "the cancellation or the next reader started a second production"
        gate.set()
        response = await second
        with pytest.raises(asyncio.CancelledError):
            await first
        return response

    response = asyncio.run(scenario())
    assert len(calls) == 1
    assert response.headers[publication.GENERATION_HEADER] == "1"

    async def abandoned():
        lone = asyncio.ensure_future(gateway_state.api_state(_request(serving.app)))
        await asyncio.sleep(0.05)
        lone.cancel()
        held = serving.app.state.state_publication
        while held._flight is not None and not held._flight[1].done():
            await asyncio.sleep(0.02)  # the shared refresh finishes for nobody: no new one starts

    asyncio.run(abandoned())
    assert len(calls) == 1, "an unchanged revision needs no production, even for a reader that left"


def test_an_unchanged_revision_serves_the_held_publication_without_reading_a_task_body(serving, monkeypatch):
    with store.open_store(serving.root, write=True) as txn:
        _task(txn, "queued-task", queued_at="2026-11-02T00:00:00Z", chat_id=1)
        _task(txn, "working-task", chat_id=1)
    serving.workers.PENDING[:] = [{"id": "queued-task", "queued_at": "2026-11-02T00:00:00Z", "chat_id": 1}]
    serving.workers.RUNNING["working-task"] = {"task": {"id": "working-task", "chat_id": 1}, "attempt": 1,
                                               "started_at": 1}
    first = _get(serving.app)
    assert {row["activity_id"] for row in _payload(first)["active_chat_activities"]} == {"queued-task",
                                                                                         "working-task"}
    productions = _counting(monkeypatch, gateway_state, "_state_snapshot")
    bodies = _counting(monkeypatch, task_store, "read_document")
    second = _get(serving.app)
    assert productions == [] and bodies == [], "an unchanged revision re-read the census"
    assert second.headers[publication.GENERATION_HEADER] == first.headers[publication.GENERATION_HEADER]
    assert {k: v for k, v in _payload(second).items() if k != "uptime"} == \
        {k: v for k, v in _payload(first).items() if k != "uptime"}
    # Companion: a real change (a store write) is produced again and shows.
    with store.open_store(serving.root, write=True) as txn:
        _task(txn, "working-task", chat_id=1, status="running", focus="moved")
    third = _get(serving.app)
    assert len(productions) == 1 and third.headers[publication.GENERATION_HEADER] == "2"


def test_publication_reprobes_an_effective_child_root_after_it_changes(serving, monkeypatch):
    """A split-drive effective result is a publication dependency, not only a budget root."""
    child = str(serving.root / "child-drive")
    revision = [1]
    productions = []

    def fake_probe(_root, roots=frozenset(), **_kwargs):
        return (tuple(sorted(roots)), revision[0])

    def produce(_direct):
        productions.append(1)
        return {"fields": {}, "task_bindings": {}, "roots": [child], "jobs": {}}

    monkeypatch.setattr(publication, "probe", fake_probe)
    held = publication.Publisher(str(serving.root))
    held.refresh(produce)
    held.refresh(produce)
    # The first refresh learns the split-drive dependency; the following
    # refresh publishes that expanded probe key, then the key is stable.
    held.refresh(produce)
    assert productions == [1, 1]
    revision[0] = 2
    held.refresh(produce)
    assert productions == [1, 1, 1]


def test_bindings_the_client_holds_are_not_sent_again(serving):
    first = _payload(_get(serving.app))
    revision = first["task_bindings_revision"]
    assert revision
    again = _get(serving.app, f"bindings_since={revision}".encode())
    payload = _payload(again)
    assert payload["task_bindings"] is None and payload["task_bindings_unchanged"] is True
    assert again.headers[publication.GENERATION_HEADER] == "1"
    stale = _payload(_get(serving.app, b"bindings_since=other"))
    assert stale["task_bindings"] == first["task_bindings"] and stale["task_bindings_unchanged"] is False


def test_only_project_times_move_the_public_message_revision(serving):
    from ouroboros import dialogue_store as ds, projects_registry as registry, room_messages

    root = serving.root
    room = registry.create_project(root, "messages", name="Messages", origin="owner_ui")["chat_id"]
    ds.ensure(root)
    room_messages.run(root, lambda: False)
    before = _payload(_get(serving.app))["project_messages_revision"]
    assert before
    for index, (chat, direction) in enumerate(((1, "in"), (1, "out"), (12345, "in"))):
        ds.append(root, "chat", {"ts": f"2026-01-01T00:0{index}:00Z", "chat_id": chat,
                                 "direction": direction, "text": "Outside Projects"}, feed=True, appearances=[])
        assert _payload(_get(serving.app))["project_messages_revision"] == before
    ds.append(root, "chat", {"ts": "2026-01-01T01:00:00Z", "chat_id": room,
                             "direction": "in", "text": "Owner request"}, feed=True, appearances=[])
    after = _payload(_get(serving.app))
    assert after["project_messages_revision"] != before
    project = next(row for row in after["projects"] if row["chat_id"] == room)
    assert project["last_message_at"] == "2026-01-01T01:00:00Z" and project["visible_revision"] == 1
    # An older delivery moves arrival/unread, but not the latest original event time.
    ds.append(root, "chat", {"ts": "2026-01-01T00:00:00Z", "chat_id": room,
                             "direction": "out", "text": "Delayed answer"}, feed=True, appearances=[])
    assert _payload(_get(serving.app))["project_messages_revision"] == after["project_messages_revision"]
    ds.append(root, "chat", {"ts": "2026-01-01T02:00:00Z", "chat_id": 1, "direction": "out",
                             "type": "main_notice", "text": "Also shown in the Project"},
              feed=True, appearances=[f"room:{room}"], shown_in=[room])
    assert _payload(_get(serving.app))["project_messages_revision"] != after["project_messages_revision"]


def _key(serving):
    return publication.probe(str(serving.root))


def test_each_source_the_body_reads_moves_the_revision(serving):
    from supervisor import state as supervisor_state
    from supervisor.active_activity import get_direct_activity_registry

    base = _key(serving)
    assert _key(serving) == base, "an unchanged root probes equal"
    with store.open_store(serving.root, write=True) as txn:
        _task(txn, "new-task")
    after_store = _key(serving)
    assert after_store != base
    serving.workers.PENDING.append({"id": "new-task"})
    after_queue = _key(serving)
    assert after_queue != after_store
    serving.workers.RUNNING["new-task"] = {"task": {"id": "new-task"}, "attempt": 1}
    after_running = _key(serving)
    serving.workers.RUNNING["new-task"]["active_llm_call"] = {"model_activity": {"phase": "reasoning"}}
    assert _key(serving) != after_running, "a live model activity the queue documents omit"
    after_activity = _key(serving)
    serving.workers.RUNNING["new-task"]["turn_phase"] = "context_building"
    assert _key(serving) != after_activity, "a start step its frames land (#2862) the queue documents omit"
    after_activity = _key(serving)
    serving.workers.RUNNING["new-task"]["context_attempt"] = {"attempt_id": "a1", "ledger_state": "dispatched"}
    assert _key(serving) != after_activity, "a sent attempt's context the queue documents omit"
    after_activity = _key(serving)
    supervisor_state.STATE_PATH.write_text(json.dumps({"evolution_cycle": 3}), encoding="utf-8")
    after_state = _key(serving)
    assert after_state != after_activity
    get_direct_activity_registry().register("direct-turn", 1)
    after_direct = _key(serving)
    assert after_direct != after_state
    (serving.root / "state" / "project_task_bindings.json").write_text("{}", encoding="utf-8")
    assert _key(serving) != after_direct


def test_a_recorded_stop_and_the_presence_author_census_move_the_revision(serving, monkeypatch):
    """Two body facts no queue or store revision announces: a recorded Stop a live Presence turn has not taken yet
    (its work's ``stop_requested``: the cancel-intent projection) and whether the Presence author census could read
    every held witness (one released makes the census whole again)."""
    from ouroboros import conversation_lease
    from ouroboros.cancel_intents import request_cancel

    def presence(key):
        return next(part for part in key if part[0] == "presence")

    before = presence(_key(serving))
    request_cancel(serving.root, "a-live-turn", source="http_cascade", requested_stop_policy="immediate")
    assert presence(_key(serving)) != before
    monkeypatch.setattr(conversation_lease, "live_presence_roots", lambda root: {"roots": [], "complete": False})
    gap = publication._presence_authors(str(serving.root))
    monkeypatch.setattr(conversation_lease, "live_presence_roots", lambda root: {"roots": [], "complete": True})
    assert publication._presence_authors(str(serving.root)) != gap


def test_an_unprovable_source_is_never_reused(serving, monkeypatch):
    monkeypatch.setattr("ouroboros.projects_registry.bindings_revision", lambda _root: "")
    assert _key(serving) != _key(serving)


def test_health_answers_without_an_executor_hop_or_a_store_read(serving, monkeypatch):
    from ouroboros import maintenance_jobs, server_liveness

    maintenance_jobs.record(serving.root, "evidence_packing", state="done", done=3, total=3)
    state_answer = _payload(_get(serving.app))
    assert state_answer["boot"]["jobs"]["evidence_packing"] == {"state": "done", "done": 3, "total": 3}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the health path left the loop or read the store")

    monkeypatch.setattr(asyncio, "to_thread", forbidden)
    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "run_in_executor", forbidden)
    monkeypatch.setattr(store, "open_store", forbidden)
    monkeypatch.setattr(server_liveness, "job_facts", forbidden)
    response = asyncio.run(gateway_state.api_health(_request(serving.app)))
    payload = json.loads(response.body)
    assert response.status_code == 200 and payload["status"] == "ok"
    assert payload["boot"]["jobs"] == state_answer["boot"]["jobs"], "the jobs the last publication read"


def test_health_before_any_publication_reports_no_jobs(serving):
    payload = json.loads(asyncio.run(gateway_state.api_health(_request(serving.app))).body)
    assert payload["boot"]["jobs"] == {}
