"""A lost presence turn is neither spoken nor final, and its retry never regenerates it.

Whether a lost attempt's model/tool work and sends took effect is unproven — absent receipts,
logs or ledger rows are no proof of no effect — so a retry of an event whose persisted row is
RUNNING, INTERRUPTED or the reconciler's placeholder fails closed before any agent exists
(``presence_attempt_outcome_unknown``; the Host answers 409 ``retry``) until a canonical terminal
replays. An event with no running row of its own (an attempt that died before its running write,
a turn thread that never started, another event) still runs, and a real terminal still replays.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.gateway import host_service
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.outcomes import infra_failed_axes
from ouroboros.presence_runner import (
    PresenceTurnError,
    PresenceTurnGate,
    _cached_result,
    _live_task_rows,
    _previous_turn_path,
    _read_previous_turn,
    _stable_numeric_id,
    _task_id,
    _turn_sends,
    presence_result_from_stored,
    presence_event_identity,
    presence_turn_is_live,
    run_presence_turn,
)
from ouroboros.task_results import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    is_reconciled_presence_placeholder,
    load_task_result,
    reopen_reconciled_presence_placeholder,
    task_result_path,
    write_task_result,
)
from ouroboros.task_status import reconcile_orphaned_running_tasks
from ouroboros.tools.presence import _finish_presence
from ouroboros.utils import append_jsonl
from tests.test_host_service_api import _seed_presence_behavior, _seed_token
from tests.test_presence_runner import _admission, _event

_PRESENCE_METADATA = {"source": "presence", "presence": {"binding_id": "1" * 32, "delivery_reporting_version": 0}}
_NOW = 1_800_000_000.0  # 2027-01-15T08:00:00Z, the fresh queue snapshot's own time
_OUTCOME_UNKNOWN = "presence_attempt_outcome_unknown"


def _sweep(tmp_path, monkeypatch, task_id, status=STATUS_RUNNING, *, seed=True, boot="2026-05-28T00:00:02+00:00",
           metadata=_PRESENCE_METADATA, **fields):
    """The real reconciler: a stale row, a later worker boot, a fresh empty queue, one sweep."""
    with monkeypatch.context() as patch:
        patch.setattr(time, "time", lambda: _NOW)
        if seed:
            write_task_result(tmp_path, task_id, status, result="Task is running.",
                              ts="2026-05-28T00:00:00+00:00", metadata=dict(metadata), **fields)
        (tmp_path / "state").mkdir(exist_ok=True)
        (tmp_path / "state" / "queue_snapshot.json").write_text(
            '{"ts": "2027-01-15T08:00:00+00:00", "pending": [], "running": []}', encoding="utf-8")
        events = tmp_path / "logs" / "events.jsonl"
        append_jsonl(events, {"ts": "2026-05-28T00:00:01+00:00", "type": "llm_round", "task_id": task_id})
        append_jsonl(events, {"ts": boot, "type": "worker_boot"})
        healed = reconcile_orphaned_running_tasks(tmp_path)
    return healed, load_task_result(tmp_path, task_id)


def _reconciled(tmp_path, monkeypatch, task_id, status=STATUS_RUNNING, **kwargs):
    healed, row = _sweep(tmp_path, monkeypatch, task_id, status, **kwargs)
    assert healed == 1 and row["status"] == STATUS_FAILED and row["status_reconciled_from"] == status
    assert row["reason_code"] == ("interrupted_retry_lost" if status == STATUS_INTERRUPTED
                                  else "orphaned_running_after_worker_restart")
    assert "TASK_ORPHAN_RECONCILED" in row["result"] and not row.get("terminal_origin")
    return row


def test_reconciled_turn_replays_silent_while_completed_legacy_row_still_speaks(tmp_path, monkeypatch):
    row = _reconciled(tmp_path, monkeypatch, "orphan-turn")
    replay = presence_result_from_stored(row, "orphan-turn")
    assert replay.outcome == "silent" and replay.text == ""
    assert _cached_result(tmp_path, "orphan-turn") is None
    # The unknown-origin compatibility path itself survives for completed rows.
    legacy = presence_result_from_stored({"status": "completed", "result": "Old reply", "metadata": {}}, "old")
    assert legacy.outcome == "message" and legacy.text == "Old reply"


@pytest.mark.parametrize("status", [STATUS_RUNNING, STATUS_INTERRUPTED])
def test_reopen_moves_the_host_mark_aside_exactly_once(tmp_path, monkeypatch, status):
    row = _reconciled(tmp_path, monkeypatch, "orphan-turn", status)
    assert reopen_reconciled_presence_placeholder(tmp_path, "orphan-turn") is True
    reopened = load_task_result(tmp_path, "orphan-turn")
    assert reopened["status"] == STATUS_RUNNING and reopened["metadata"] == row["metadata"]
    assert reopened["superseded_placeholder"] == {
        "status": STATUS_FAILED, "reason_code": row["reason_code"], "status_reconciled_from": status,
        "ts": row["ts"], "result": row["result"][-500:],
        # the failed transition's terminal-projection provenance goes aside with the mark
        "canonical_terminal_projection_origin": "terminal_transition",
    }
    for cleared in ("reason_code", "outcome_axes", "artifact_status", "artifact_bundle", "result",
                    "status_reconciled_from", "canonical_terminal_projection_origin"):
        assert cleared not in reopened
    # A running row is no placeholder: the transition cannot fire twice.
    before = task_result_path(tmp_path, "orphan-turn").read_bytes()
    assert reopen_reconciled_presence_placeholder(tmp_path, "orphan-turn") is False
    assert task_result_path(tmp_path, "orphan-turn").read_bytes() == before


def _outcome_failure(tmp_path, monkeypatch):
    healed, row = _sweep(tmp_path, monkeypatch, "axes-turn", reason_code="provider_unavailable",
                         outcome_axes=infra_failed_axes("provider_unavailable"))
    # task_status: a RUNNING row whose own axes already failed is marked from those axes.
    assert healed == 1 and row["status"] == STATUS_FAILED and row["status_reconciled_from"] == STATUS_RUNNING
    assert row["reason_code"] == "provider_unavailable" and "TASK_ORPHAN_RECONCILED" not in row["result"]
    return "axes-turn"


def _ordinary_failure(tmp_path, _monkeypatch):
    write_task_result(tmp_path, "failed-turn", STATUS_FAILED, result="Authored partial reply",
                      terminal_origin="model_final", reason_code="round_limit",
                      metadata={**_PRESENCE_METADATA, "presence_outcome": "message"})
    return "failed-turn"


def _non_presence_orphan(tmp_path, monkeypatch):
    _reconciled(tmp_path, monkeypatch, "ordinary-task", metadata={"source": "web"})
    return "ordinary-task"


@pytest.mark.parametrize("seed", [_outcome_failure, _ordinary_failure, _non_presence_orphan])
def test_only_the_orphan_placeholder_of_a_presence_turn_reopens(tmp_path, monkeypatch, seed):
    task_id = seed(tmp_path, monkeypatch)
    before = task_result_path(tmp_path, task_id).read_bytes()
    assert reopen_reconciled_presence_placeholder(tmp_path, task_id) is False
    assert task_result_path(tmp_path, task_id).read_bytes() == before
    if seed is _ordinary_failure:
        assert _cached_result(tmp_path, task_id) is not None  # the model's own terminal replays
    elif seed is _outcome_failure:
        # A host infrastructure terminal answered nothing: it is never reopened as a placeholder, and
        # replay keeps the event with the transport instead of acknowledging it as silent.
        with pytest.raises(PresenceTurnError) as refused:
            _cached_result(tmp_path, task_id)
        assert refused.value.code == "presence_attempt_outcome_unknown"
    write_task_result(tmp_path, task_id, STATUS_COMPLETED, result="Late answer")
    assert load_task_result(tmp_path, task_id)["status"] == STATUS_FAILED  # sticky terminal unchanged


def _complete(task, reply, drive_root):
    """The real pipeline's terminal write for *task*: an accepted presence_finish, then emit_task_results."""
    task["_skip_post_task_synthesis"] = True
    ctx = SimpleNamespace(task_contract=task["task_contract"], task_metadata=task["metadata"])
    _finish_presence(ctx, "message", reply)
    ctx._presence_completion_accepted = True
    pending: list = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=drive_root, repo_dir=drive_root), None, None, pending, task, reply,
        {"terminal_origin": "model_final"}, {"tool_calls": [], "reasoning_notes": []}, 0.0,
        drive_root / "logs", ctx=ctx,
    )
    return pending


def _answering_agent(calls, reply, drive_root, *, during=None, lost=False):
    """Real durable pipeline: the RUNNING start write, then the terminal write (or a lost worker)."""

    class Agent:
        def handle_task(self, task):
            calls.append(task)
            task["_skip_post_task_synthesis"] = True
            write_task_result(drive_root, task["id"], STATUS_RUNNING, metadata=task["metadata"],
                              result="Task is running.", ts="2026-05-28T00:00:00+00:00" if lost else "")
            if during is not None:
                during(task)
            if lost:
                raise RuntimeError("worker lost")
            return _complete(task, reply, drive_root)

    return Agent()


def _chat_rows(drive_root):
    chat = drive_root / "logs" / "chat.jsonl"
    return [json.loads(line) for line in chat.read_text(encoding="utf-8").splitlines()] if chat.exists() else []


def _trace(drive_root, task_id, conversation_key=_event().conversation_key):
    """What a refused retry must leave untouched: the turn's row, the conversation's pointer and dialogue."""
    row, pointer = task_result_path(drive_root, task_id), _previous_turn_path(drive_root, conversation_key)
    chat_id = _stable_numeric_id("presence-conversation", conversation_key)
    dialogue = [item for item in _chat_rows(drive_root)
                if item.get("task_id") == task_id or item.get("chat_id") == chat_id]
    return (row.read_bytes() if row.exists() else None, pointer.read_bytes() if pointer.exists() else None, dialogue)


def _retry_kwargs(tmp_path, built, *, version=1, reply="Regenerated answer", **overrides):
    """A retry whose agent factory records every build: a refused retry must fail before the first."""
    calls: list = []

    def factory(**kwargs):
        built.append(kwargs)
        return _answering_agent(calls, reply, tmp_path)

    return dict(admission=_admission(), event=replace(_event(), delivery_reporting_version=version),
                repo_dir=tmp_path, drive_root=tmp_path, agent_factory=factory, gate=PresenceTurnGate(1), **overrides)


def _refused(kwargs):
    """The typed refusal: the event stays with its transport, named by the turn it could not settle."""
    with pytest.raises(PresenceTurnError) as raised:
        run_presence_turn(**kwargs)
    assert (raised.value.code, raised.value.field, getattr(raised.value, "turn_ref", None)) == (
        _OUTCOME_UNKNOWN, "source_event_id", _task_id(kwargs["admission"], kwargs["event"]))


def _assert_fails_closed(tmp_path, task_id, kwargs, built):
    """Two retries, each refused before any agent exists, and neither leaves a trace to inherit."""
    before = _trace(tmp_path, task_id, kwargs["event"].conversation_key)
    for _second in (False, True):  # a second retry is refused exactly like the first, never promoted
        _refused(kwargs)
        assert built == [] and not presence_turn_is_live(task_id)
        assert _trace(tmp_path, task_id, kwargs["event"].conversation_key) == before
    assert _cached_result(tmp_path, task_id) is None  # still no result to replay


def _prior_sends(tmp_path, task_id):
    """The receipt facts a lost attempt left in the live generation, settled by the runner's own rule."""
    return _turn_sends(_live_task_rows(tmp_path, task_id, _event().conversation_key))


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("status", [STATUS_RUNNING, STATUS_INTERRUPTED])
def test_reconciled_turn_fails_closed_and_is_never_regenerated(tmp_path, monkeypatch, status, version):
    """Formerly the placeholder re-ran with 'what the lost attempt sent' as model context. That context
    cannot prove the lost model/tool work had no effect, so the retry is refused and the host mark stays."""
    task_id = _task_id(_admission(), _event())
    row = _reconciled(tmp_path, monkeypatch, task_id, status)
    assert _cached_result(tmp_path, task_id) is None  # the placeholder is still not a result
    built: list = []
    _assert_fails_closed(tmp_path, task_id, _retry_kwargs(tmp_path, built, version=version), built)
    stored = load_task_result(tmp_path, task_id)
    assert is_reconciled_presence_placeholder(stored) and stored == row  # never reopened, never moved aside
    assert "superseded_placeholder" not in stored


def test_reconciler_repersist_keeps_the_placeholder(tmp_path, monkeypatch):
    row = _reconciled(tmp_path, monkeypatch, "orphan-turn")
    write_task_result(tmp_path, "orphan-turn", STATUS_FAILED, result=row["result"],
                      status_reconciled_from=row["status_reconciled_from"])
    assert load_task_result(tmp_path, "orphan-turn")["status_reconciled_from"] == STATUS_RUNNING
    assert _cached_result(tmp_path, "orphan-turn") is None


def test_ordinary_failed_turn_still_replays_and_stays_failed(tmp_path):
    task_id = _task_id(_admission(), _event())
    write_task_result(tmp_path, task_id, STATUS_FAILED, result="Authored partial reply",
                      terminal_origin="model_final", reason_code="round_limit",
                      metadata={**_PRESENCE_METADATA, "presence_outcome": "message",
                                "presence_result_text": "Authored partial reply",
                                "presence_event_identity": presence_event_identity(_admission().binding_id, _event())})
    calls: list = []
    result = run_presence_turn(admission=_admission(), event=_event(), repo_dir=tmp_path, drive_root=tmp_path,
                               agent_factory=lambda **_kw: _answering_agent(calls, "New answer", tmp_path),
                               gate=PresenceTurnGate(1))
    assert calls == [] and result.outcome == "message" and result.text == "Authored partial reply"
    write_task_result(tmp_path, task_id, STATUS_COMPLETED, result="New answer")
    assert load_task_result(tmp_path, task_id)["status"] == STATUS_FAILED
    # A failed row whose origin is unknown is host text, never a reply.
    write_task_result(tmp_path, "unknown-origin", STATUS_FAILED, result="Error during processing",
                      metadata=dict(_PRESENCE_METADATA))
    assert presence_result_from_stored(load_task_result(tmp_path, "unknown-origin"), "unknown-origin").text == ""


@pytest.mark.parametrize("origin", ["", "host_notice", "host_salvage"])
def test_failed_deferred_turn_keeps_its_work_reference(tmp_path, origin):
    write_task_result(tmp_path, "deferred-turn", STATUS_FAILED, result="Host diagnostic", terminal_origin=origin,
                      metadata={**_PRESENCE_METADATA, "presence_outcome": "deferred",
                                "presence_work_ref": "presence-work-1"})
    replay = _cached_result(tmp_path, "deferred-turn")
    assert (replay.outcome, replay.text, replay.work_ref) == ("deferred", "", "presence-work-1")


def test_reconciled_non_presence_task_keeps_the_sticky_terminal(tmp_path, monkeypatch):
    _reconciled(tmp_path, monkeypatch, "ordinary-task", metadata={"source": "web"})
    write_task_result(tmp_path, "ordinary-task", STATUS_COMPLETED, result="Late answer")
    row = load_task_result(tmp_path, "ordinary-task")
    assert row["status"] == STATUS_FAILED and row["status_reconciled_from"] == STATUS_RUNNING


_HOST_EVENT_ID = "telegram:bot-1:42"


def _host(tmp_path, agents):
    """The real Host over the real runner; each turn builds the next queued agent (an exception raises)."""
    _seed_token(tmp_path, skill="telegram-bot", token="presence-token",
                permissions=["presence"], manifest_permissions=["presence"])
    binding_id = _seed_presence_behavior(tmp_path)

    def factory(**_kw):
        agent = agents.pop(0)
        if isinstance(agent, BaseException):
            raise agent
        return agent

    def run_real_presence(**kwargs):
        return run_presence_turn(repo_dir=tmp_path, drive_root=tmp_path, agent_factory=factory,
                                 gate=PresenceTurnGate(1), **kwargs)

    app = create_host_service_app(tmp_path, presence_runner=run_real_presence)
    return app, binding_id


def _host_body(binding_id, source_event_id=_HOST_EVENT_ID):
    return {"binding_id": binding_id, "delivery_reporting_version": 1, "event": {
        "source_event_id": source_event_id, "provider": "telegram", "account_id": "bot-1",
        "conversation_id": "room-1", "thread_id": "topic-1", "conversation_key": "ignored",
        "actor": {"platform_actor_id": "user-7"}, "conversation": {}, "message": {"message_id": "42"},
        "text": "Hello",
    }}


def _host_post(tmp_path, agents):
    app, binding_id = _host(tmp_path, agents)
    client = TestClient(app)

    def post(source_event_id=_HOST_EVENT_ID):
        return client.post("/presence/turn", headers={"X-Skill-Token": "presence-token"},
                           json=_host_body(binding_id, source_event_id))

    task_id = "presence-" + hashlib.sha256(f"{binding_id}\0{_HOST_EVENT_ID}".encode("utf-8")).hexdigest()[:24]
    return app.state.host_service_context, post, task_id


def _host_refused(response, task_id):
    body = response.json()
    assert response.status_code == 409 and body["ok"] is False and not body.get("text"), body
    assert (body["code"], body["disposition"], body["field"], body["turn_ref"]) == (
        _OUTCOME_UNKNOWN, "retry", "source_event_id", task_id)


@pytest.mark.parametrize("prior_send", [None, "delivered", "uncertain"])
def test_host_retry_of_a_lost_turn_is_a_409_retry_never_a_rerun(tmp_path, monkeypatch, prior_send):
    """Through POST /presence/turn: a lost v1 attempt, retries before and after the reconciler, a second
    retry — all refused, whatever the lost attempt's receipts said. Formerly the placeholder re-ran with
    the prior sends as context; a confirmed send, an unconfirmed one or none proves nothing about the rest."""
    agents: list = []
    ctx, post, task_id = _host_post(tmp_path, agents)
    recorder = ctx.presence_deliveries
    sweeps = []

    def lost_attempt_effects(task):
        if prior_send:  # the lost attempt had already reported one transport send
            recorder.record("telegram-bot", {
                "schema_version": 1, "delivery_id": "send:early", "part_id": "0", "state": prior_send,
                "provider": "telegram", "account_id": "bot-1", "conversation_id": "room-1", "thread_id": "topic-1",
                "text": "Early part", "format": "markdown", "message": {"provider_message_id": "501"},
                "origin": {"kind": "tool", "task_id": task["id"], "source_event_id": _HOST_EVENT_ID},
            })
        # stale by clock and by a later boot, but executing in this process: the live set protects it
        sweeps.append(_sweep(tmp_path, monkeypatch, task["id"], seed=False, boot="2027-01-15T07:59:00+00:00")[0])
        assert load_task_result(tmp_path, task["id"])["status"] == STATUS_RUNNING

    calls: list = []
    agents.append(_answering_agent(calls, "", tmp_path, during=lost_attempt_effects, lost=True))
    assert post().status_code == 500 and load_task_result(tmp_path, task_id)["status"] == STATUS_RUNNING
    assert sweeps == [0] and len(calls) == 1
    must_not_run = _answering_agent(calls, "Regenerated answer", tmp_path)
    agents.append(must_not_run)
    before = _trace(tmp_path, task_id)
    _host_refused(post(), task_id)  # the unreconciled RUNNING row: nothing of this turn is live here any more
    assert _trace(tmp_path, task_id) == before
    _reconciled(tmp_path, monkeypatch, task_id, seed=False)
    before = _trace(tmp_path, task_id)
    for _second in (False, True):
        _host_refused(post(), task_id)
        assert agents == [must_not_run] and len(calls) == 1 and _trace(tmp_path, task_id) == before
        assert ctx.presence_turns.live() == [] and not any(ctx._inflight.values())
    assert is_reconciled_presence_placeholder(load_task_result(tmp_path, task_id))
    assert _prior_sends(tmp_path, task_id) == ((["Early part"] if prior_send == "delivered" else []),
                                               int(prior_send == "uncertain"))
    inbound = [row for row in _chat_rows(tmp_path) if row.get("direction") == "in" and row.get("task_id") == task_id]
    assert len(inbound) == 1  # the refusals never log the correspondent's message again
    # Fail-closed is per turn, and the refusals kept no capacity: another event of the room still runs.
    agents[:] = [_answering_agent(calls, "Fresh answer", tmp_path)]
    fresh = post("telegram:bot-1:43")
    assert fresh.status_code == 200 and fresh.json()["text"] == "Fresh answer" and agents == []
    assert "previous_attempt" not in calls[-1]["metadata"]["presence"]
    _host_refused(post(), task_id)


@pytest.mark.parametrize("status", [STATUS_RUNNING, STATUS_INTERRUPTED])
def test_host_replays_the_late_canonical_terminal_of_a_refused_turn(tmp_path, status):
    """An older worker still finishing the turn writes its terminal after a refusal: the retry answers
    from that durable row, once, and never builds an agent."""
    agents: list = []
    _ctx, post, task_id = _host_post(tmp_path, agents)
    calls: list = []
    agents.append(_answering_agent(calls, "", tmp_path, lost=True))
    assert post().status_code == 500
    if status == STATUS_INTERRUPTED:
        write_task_result(tmp_path, task_id, STATUS_INTERRUPTED, result="Task interrupted.")
    must_not_run = _answering_agent(calls, "Regenerated answer", tmp_path)
    agents.append(must_not_run)
    _host_refused(post(), task_id)
    _complete(calls[0], "Late answer", tmp_path)  # the lost attempt's own terminal lands after all
    first = post()
    assert first.status_code == 200 and first.json()["text"] == "Late answer" and first.json()["turn_ref"] == task_id
    assert agents == [must_not_run] and len(calls) == 1
    assert _read_previous_turn(tmp_path, _event().conversation_key)["task_id"] == task_id  # repaired on replay
    replay = post()
    assert replay.status_code == 200 and replay.json() == first.json() and agents == [must_not_run]
    stored = load_task_result(tmp_path, task_id)
    assert stored["status"] == STATUS_COMPLETED and "superseded_placeholder" not in stored


def test_host_retry_of_a_live_turn_joins_it_instead_of_refusing(tmp_path):
    """A RUNNING row whose execution is live in this Host is not a lost attempt: the retry joins it."""
    agents: list = []
    app, binding_id = _host(tmp_path, agents)
    ctx = app.state.host_service_context
    started, release, calls, joins = threading.Event(), threading.Event(), [], []

    def hold(_task):
        started.set()
        assert release.wait(5)

    agents.append(_answering_agent(calls, "Real answer", tmp_path, during=hold))
    real_start_or_join = ctx.presence_turns.start_or_join

    async def scenario():
        second_joined = asyncio.Event()

        def recording(turn_id, **kwargs):
            outcome = real_start_or_join(turn_id, **kwargs)
            joins.append(outcome[1])
            if len(joins) == 2:
                second_joined.set()
            return outcome

        ctx.presence_turns.start_or_join = recording

        async def json_body():
            return _host_body(binding_id)

        def request():
            return SimpleNamespace(app=app, headers={"x-skill-token": "presence-token"}, json=json_body)

        first = asyncio.ensure_future(host_service._api_presence_turn(request()))
        assert await asyncio.to_thread(started.wait, 5)
        assert load_task_result(tmp_path, calls[0]["id"])["status"] == STATUS_RUNNING
        second = asyncio.ensure_future(host_service._api_presence_turn(request()))
        await asyncio.wait_for(second_joined.wait(), 5)
        release.set()
        return await asyncio.wait_for(asyncio.gather(first, second), 5)

    first, second = asyncio.run(scenario())
    assert joins == [True, False] and len(calls) == 1
    assert first.status_code == second.status_code == 200 and first.body == second.body
    assert json.loads(first.body)["text"] == "Real answer"


def test_host_turn_that_never_started_runs_on_its_retry(tmp_path, monkeypatch):
    """No thread, or no agent before the running write: nothing of the turn ran, so the retry answers it."""
    agents: list = []
    ctx, post, task_id = _host_post(tmp_path, agents)
    start = threading.Thread.start

    def refuse_turn_threads(thread):
        if thread.name.startswith("presence-turn-"):
            raise RuntimeError("can't start new thread")
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", refuse_turn_threads)
    refused = post()
    assert refused.status_code == 503 and (refused.json()["code"], refused.json()["disposition"]) == (
        "presence_turn_not_started", "retry")
    assert load_task_result(tmp_path, task_id) is None and _chat_rows(tmp_path) == []
    assert ctx.presence_turns.live() == [] and not any(ctx._inflight.values())
    monkeypatch.setattr(threading.Thread, "start", start)
    calls: list = []
    agents[:] = [RuntimeError("worker died before the running write"), _answering_agent(calls, "Real answer", tmp_path)]
    assert post().status_code == 500 and load_task_result(tmp_path, task_id) is None
    first = post()
    assert first.status_code == 200 and first.json()["text"] == "Real answer" and len(calls) == 1
    assert "previous_attempt" not in calls[0]["metadata"]["presence"]
    assert [row["direction"] for row in _chat_rows(tmp_path) if row.get("task_id") == task_id].count("in") == 1
    replay = post()
    assert replay.status_code == 200 and replay.json() == first.json() and len(calls) == 1


def _lost_v1_attempt(tmp_path, task_id, *, chat_id):
    """A v1 attempt that logged its inbound row, delivered one part through a tool, then died."""
    write_task_result(tmp_path, task_id, STATUS_RUNNING, result="Task is running.", ts="2026-05-28T00:00:00+00:00",
                      metadata={"source": "presence", "presence": {"binding_id": "1" * 32, "delivery_reporting_version": 1}})
    chat = tmp_path / "logs" / "chat.jsonl"
    append_jsonl(chat, {"direction": "in", "chat_id": chat_id, "client_message_id": "telegram:bot-1:42",
                        "text": "Hello", "task_id": task_id})
    append_jsonl(chat, {"type": "presence_delivery", "direction": "out", "chat_id": chat_id, "text": "Early part",
                        "task_id": task_id, "transport": {"conversation_key": _event().conversation_key, "delivery": {
                            "state": "delivered", "delivery_id": "send:early", "part_id": "0"}}})
    return chat


def _v1_kwargs(tmp_path, calls, reply="Real answer", **overrides):
    return dict(admission=_admission(), event=replace(_event(), delivery_reporting_version=1), repo_dir=tmp_path,
                drive_root=tmp_path, agent_factory=lambda **_kw: _answering_agent(calls, reply, tmp_path),
                gate=PresenceTurnGate(1), **overrides)


def _lose_attempt(tmp_path, monkeypatch, kind):
    """A real v1 attempt: inbound row, running write, one confirmed tool send, then its worker is lost.

    ``kind`` is how the lost row persists: still RUNNING, marked INTERRUPTED, or ``reopened`` — a
    placeholder an earlier release had already moved back to RUNNING for a re-run that was lost too.
    """
    calls: list = []

    def confirmed_send(task):
        append_jsonl(tmp_path / "logs" / "chat.jsonl", {
            "type": "presence_delivery", "direction": "out", "chat_id": task["chat_id"], "text": "Early part",
            "task_id": task["id"], "transport": {"conversation_key": _event().conversation_key, "delivery": {
                "state": "delivered", "delivery_id": "send:early", "part_id": "0"}}})

    with pytest.raises(RuntimeError, match="worker lost"):
        run_presence_turn(**{**_v1_kwargs(tmp_path, calls), "agent_factory": lambda **_kw: _answering_agent(
            calls, "", tmp_path, during=confirmed_send, lost=True)})
    task_id = calls[0]["id"]
    if kind == STATUS_INTERRUPTED:
        write_task_result(tmp_path, task_id, STATUS_INTERRUPTED, result="Task interrupted.")
    elif kind == "reopened":
        _reconciled(tmp_path, monkeypatch, task_id, seed=False)
        assert reopen_reconciled_presence_placeholder(tmp_path, task_id) is True
    assert load_task_result(tmp_path, task_id)["status"] == (STATUS_INTERRUPTED if kind == STATUS_INTERRUPTED
                                                             else STATUS_RUNNING)
    return calls[0]


@pytest.mark.parametrize("kind", [STATUS_RUNNING, STATUS_INTERRUPTED, "reopened"])
def test_stale_running_row_fails_closed_before_the_reconciler_runs(tmp_path, monkeypatch, kind):
    """An adapter retry inside the reconciler's grace window finds the dead attempt's row: formerly it re-ran
    with the confirmed send as context, now it is refused until the attempt's own terminal replays."""
    lost = _lose_attempt(tmp_path, monkeypatch, kind)
    task_id = lost["id"]
    assert _cached_result(tmp_path, task_id) is None and _prior_sends(tmp_path, task_id) == (["Early part"], 0)
    built: list = []
    kwargs = _retry_kwargs(tmp_path, built)
    _assert_fails_closed(tmp_path, task_id, kwargs, built)
    _complete(lost, "Late answer", tmp_path)  # a later canonical terminal
    first = run_presence_turn(**kwargs)
    assert (first.outcome, first.text, first.task_id) == ("message", "Late answer", task_id) and built == []
    assert run_presence_turn(**kwargs) == first and built == []
    stored = load_task_result(tmp_path, task_id)
    assert stored["status"] == STATUS_COMPLETED and stored["terminal_origin"] == "model_final"
    assert [row["direction"] for row in _chat_rows(tmp_path) if row.get("task_id") == task_id].count("in") == 1


def test_a_lost_attempt_blocks_only_its_own_event(tmp_path, monkeypatch):
    """Another event of the same conversation has no running row of its own: it runs as a fresh turn,
    and neither touches the lost row nor unblocks it."""
    lost = _lose_attempt(tmp_path, monkeypatch, STATUS_RUNNING)
    lost_row = task_result_path(tmp_path, lost["id"]).read_bytes()
    built: list = []
    fresh_kwargs = _retry_kwargs(tmp_path, built, reply="Fresh answer")
    fresh_kwargs["event"] = replace(fresh_kwargs["event"], source_event_id="telegram:bot-1:43")
    fresh = run_presence_turn(**fresh_kwargs)
    assert fresh.text == "Fresh answer" and fresh.task_id != lost["id"] and len(built) == 1
    assert task_result_path(tmp_path, lost["id"]).read_bytes() == lost_row
    built.clear()
    _assert_fails_closed(tmp_path, lost["id"], _retry_kwargs(tmp_path, built), built)


@pytest.mark.parametrize("reconciled", [False, True])
def test_a_refused_retry_asks_the_owner_once_and_never_the_correspondent(tmp_path, monkeypatch, reconciled):
    """With the installation's bridge bound, the refusal raises one owner-side question: a second retry
    does not repeat it, the correspondent's conversation never hears it, its once-stamp is the only row
    change, and an event with no running row of its own asks nothing."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from supervisor import message_bus

    lost = _lose_attempt(tmp_path, monkeypatch, STATUS_RUNNING)
    if reconciled:
        _reconciled(tmp_path, monkeypatch, lost["id"], seed=False)
    before, dialogue = load_task_result(tmp_path, lost["id"]), _trace(tmp_path, lost["id"])[2]
    notices: list = []
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: object())
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *args, **kwargs: notices.append((args, kwargs)))
    built: list = []
    kwargs = _retry_kwargs(tmp_path, built)
    for _second in (False, True):
        _refused(kwargs)
        assert built == [] and len(notices) == 1 and _trace(tmp_path, lost["id"])[2] == dialogue
    args, meta = notices[0]
    assert args[0] == WEB_UI_CHAT_ID != lost["chat_id"] and lost["id"] in args[1]
    assert meta["system_type"] == "presence_recovery_required"
    after, stamp = load_task_result(tmp_path, lost["id"]), {"presence_recovery_owner_notified", "updated_at"}
    assert after["presence_recovery_owner_notified"] and is_reconciled_presence_placeholder(after) is reconciled
    assert {k: v for k, v in after.items() if k not in stamp} == {k: v for k, v in before.items() if k not in stamp}

    class Quiet:
        def handle_task(self, task):
            write_task_result(tmp_path, task["id"], STATUS_COMPLETED, metadata=task["metadata"], result="")
            return [{"type": "presence_result", "outcome": "silent", "text": "", "work_ref": ""}]

    fresh = run_presence_turn(**{**kwargs, "event": replace(kwargs["event"], source_event_id="telegram:bot-1:43"),
                                 "agent_factory": lambda **_kw: Quiet()})
    assert fresh.outcome == "silent" and len(notices) == 1


def test_rotation_between_the_lost_attempt_and_its_retry_keeps_prior_sends_unknown(tmp_path):
    """Receipts in a rotated archive are not counted as zero, and a refused retry never re-logs the
    message into the live generation (formerly the re-run was told the count was unknown)."""
    task_id = _task_id(_admission(), _event())
    chat = _lost_v1_attempt(tmp_path, task_id, chat_id=7)
    (tmp_path / "archive").mkdir()
    chat.rename(tmp_path / "archive" / "chat_20260528T000100.jsonl")  # the live generation rotated
    built: list = []
    _assert_fails_closed(tmp_path, task_id, _retry_kwargs(tmp_path, built), built)
    assert not chat.exists() and _prior_sends(tmp_path, task_id) == (None, 0)


def test_a_second_refusal_after_the_rotation_keeps_the_count_unknown(tmp_path):
    """No retry may leave a fresh inbound row a later reader mistakes for complete receipt coverage:
    formerly a second death proved it for the re-run; now neither refusal writes into the live file."""
    task_id = _task_id(_admission(), _event())
    chat = _lost_v1_attempt(tmp_path, task_id, chat_id=7)
    (tmp_path / "archive").mkdir()
    chat.rename(tmp_path / "archive" / "chat_20260528T000100.jsonl")
    chat.touch()  # the rotator leaves a fresh live generation behind, as in production
    built: list = []
    kwargs = _retry_kwargs(tmp_path, built)
    for _attempt in ("A", "B"):
        _refused(kwargs)
        assert built == [] and chat.read_bytes() == b"" and _prior_sends(tmp_path, task_id) == (None, 0)


def test_rejected_build_leaves_the_placeholder_and_the_next_retry_still_fails_closed(tmp_path, monkeypatch):
    """The host mark never moves aside. Formerly a rejected build kept it for the re-run that followed; now
    the lost attempt is refused before any build or staging, whatever files the retry carries."""
    task_id = _task_id(_admission(), _event())
    row = _reconciled(tmp_path, monkeypatch, task_id)
    built: list = []
    kwargs = _retry_kwargs(tmp_path, built, version=0)
    _refused({**kwargs, "staged_files": [tmp_path / "missing.png"]})
    assert load_task_result(tmp_path, task_id) == row and built == []  # still the placeholder
    _assert_fails_closed(tmp_path, task_id, kwargs, built)
    assert load_task_result(tmp_path, task_id) == row


def test_uncertain_receipts_fail_closed_and_stay_a_floor(tmp_path):
    """A timed-out send the provider never confirmed is neither counted nor forgotten — and it is exactly
    the effect a regenerated turn could duplicate, so the retry is refused (formerly it re-ran)."""
    task_id = _task_id(_admission(), _event())
    chat = _lost_v1_attempt(tmp_path, task_id, chat_id=7)
    append_jsonl(chat, {"type": "presence_delivery", "direction": "system", "chat_id": 7, "text": "Maybe part",
                        "task_id": task_id, "transport": {"conversation_key": _event().conversation_key, "delivery": {
                            "state": "uncertain", "delivery_id": "send:late", "part_id": "0"}}})
    built: list = []
    _assert_fails_closed(tmp_path, task_id, _retry_kwargs(tmp_path, built), built)
    assert _prior_sends(tmp_path, task_id) == (["Early part"], 1)


@pytest.mark.parametrize("states, uncertain", [(("uncertain", "failed"), 0), (("failed", "uncertain"), 1)])
def test_a_part_settles_by_its_latest_receipt(tmp_path, states, uncertain):
    """A timed-out part the provider later refused is neither delivered nor possibly landed; a refused
    part whose retry timed out may have landed after all. Either way the lost turn is not regenerated."""
    task_id = _task_id(_admission(), _event())
    chat = _lost_v1_attempt(tmp_path, task_id, chat_id=7)
    for state in states:
        append_jsonl(chat, {"type": "presence_delivery", "direction": "system", "chat_id": 7, "text": "Maybe part",
                            "task_id": task_id, "transport": {"conversation_key": _event().conversation_key, "delivery": {
                                "state": state, "delivery_id": "send:late", "part_id": "0"}}})
    built: list = []
    _assert_fails_closed(tmp_path, task_id, _retry_kwargs(tmp_path, built), built)
    assert _prior_sends(tmp_path, task_id) == (["Early part"], uncertain)


def test_an_attempt_that_died_before_its_running_write_still_logs_the_message_once(tmp_path):
    """Only the inbound row survives such a death; the retry must not repeat the correspondent."""
    calls: list = []
    kwargs = _v1_kwargs(tmp_path, calls)
    agents = [None, _answering_agent(calls, "Real answer", tmp_path)]

    def factory(**_kw):
        agent = agents.pop(0)
        if agent is None:
            raise RuntimeError("worker died before the running write")
        return agent

    with pytest.raises(RuntimeError):
        run_presence_turn(**{**kwargs, "agent_factory": factory})
    first = run_presence_turn(**{**kwargs, "agent_factory": factory})
    assert first.text == "Real answer" and "previous_attempt" not in calls[0]["metadata"]["presence"]
    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["direction"] for row in rows if row.get("task_id") == first.task_id].count("in") == 1


def test_a_confirmed_part_later_refused_is_not_delivered(tmp_path):
    """One latest-state rule for confirmed and uncertain parts alike; zero confirmed sends is still no proof
    that the lost model/tool work had no effect, so the retry is refused (formerly it re-ran)."""
    task_id = _task_id(_admission(), _event())
    chat = _lost_v1_attempt(tmp_path, task_id, chat_id=7)
    append_jsonl(chat, {"type": "presence_delivery", "direction": "system", "chat_id": 7, "text": "Early part",
                        "task_id": task_id, "transport": {"conversation_key": _event().conversation_key, "delivery": {
                            "state": "failed", "delivery_id": "send:early", "part_id": "0"}}})
    built: list = []
    _assert_fails_closed(tmp_path, task_id, _retry_kwargs(tmp_path, built), built)
    assert _prior_sends(tmp_path, task_id) == ([], 0)


def test_a_receipt_addressed_to_another_conversation_does_not_count(tmp_path):
    """A tool send elsewhere with the same body is not this conversation's delivery."""
    task_id = _task_id(_admission(), _event())
    chat = _lost_v1_attempt(tmp_path, task_id, chat_id=7)
    append_jsonl(chat, {"type": "presence_delivery", "direction": "out", "chat_id": 8, "text": "Early part",
                        "task_id": task_id, "transport": {"conversation_key": "telegram:bot-1:other-room:0", "delivery": {
                            "state": "delivered", "delivery_id": "send:elsewhere", "part_id": "0"}}})
    built: list = []
    _assert_fails_closed(tmp_path, task_id, _retry_kwargs(tmp_path, built), built)
    assert _prior_sends(tmp_path, task_id) == (["Early part"], 0)


def test_reconciler_skips_a_row_whose_retry_went_live_after_the_decision(tmp_path, monkeypatch):
    """The orphan decision is taken outside the row lock; a presence retry that registered meanwhile
    cancels the write, and the same sweep heals once nothing is live."""
    from ouroboros import presence_runner, task_status

    task_id = "presence-raced"
    real_effective, order = task_status.load_effective_task_result, []

    def effective_then_retry_registers(root, tid, *args, **kwargs):
        effective = real_effective(root, tid, *args, **kwargs)
        if tid == task_id and not order:  # the retry goes live right after the sweep decided
            order.append("live")
            with presence_runner._LIVE_LOCK:
                presence_runner._LIVE_PRESENCE_TASKS.add(task_id)
        return effective

    monkeypatch.setattr(task_status, "load_effective_task_result", effective_then_retry_registers)
    try:
        healed, row = _sweep(tmp_path, monkeypatch, task_id)
        assert (healed, row["status"], order) == (0, STATUS_RUNNING, ["live"])  # decision dropped, row untouched
    finally:
        with presence_runner._LIVE_LOCK:
            presence_runner._LIVE_PRESENCE_TASKS.discard(task_id)
    healed, row = _sweep(tmp_path, monkeypatch, task_id, seed=False)
    assert healed == 1 and row["status"] == STATUS_FAILED and row["status_reconciled_from"] == STATUS_RUNNING


def test_reconciler_settles_nothing_when_the_row_was_requeued_after_the_decision(tmp_path, monkeypatch):
    """A row requeued (scheduled) between the decision and the write is neither healed nor cleaned up."""
    from ouroboros import owner_quiz, task_status

    task_id = "presence-requeued"
    real_effective, cleanups = task_status.load_effective_task_result, []

    def effective_then_requeue(root, tid, *args, **kwargs):
        effective = real_effective(root, tid, *args, **kwargs)
        if tid == task_id:
            write_task_result(tmp_path, task_id, "scheduled", result="New authority")
        return effective

    monkeypatch.setattr(task_status, "load_effective_task_result", effective_then_requeue)
    monkeypatch.setattr(owner_quiz, "reconcile_terminal", lambda root, tid: cleanups.append(tid))
    healed, row = _sweep(tmp_path, monkeypatch, task_id)
    assert (healed, row["status"], row["result"], cleanups) == (0, "scheduled", "New authority", [])
    # The same sweep over a genuine orphan still heals and still runs the terminal cleanup.
    monkeypatch.setattr(task_status, "load_effective_task_result", real_effective)
    healed, row = _sweep(tmp_path, monkeypatch, "presence-orphan")
    assert (healed, row["status"], cleanups) == (1, STATUS_FAILED, ["presence-orphan"])


def test_an_unwritable_inbound_row_fails_the_turn_before_the_model_runs(tmp_path, monkeypatch):
    """The no-re-log rule assumes the inbound row landed; a failed append is a failed turn, then a retry logs it."""
    from ouroboros import presence_runner

    calls: list = []
    kwargs = _v1_kwargs(tmp_path, calls)
    real_append = presence_runner.append_jsonl
    monkeypatch.setattr(presence_runner, "append_jsonl", lambda path, obj=None, **_kw: False)
    with pytest.raises(PresenceTurnError) as raised:
        run_presence_turn(**kwargs)
    assert raised.value.code == "chat_log_unwritable" and calls == []
    assert load_task_result(tmp_path, _task_id(_admission(), kwargs["event"])) is None  # no lost attempt to inherit
    monkeypatch.setattr(presence_runner, "append_jsonl", real_append)
    first = run_presence_turn(**kwargs)
    rows = [r for r in _chat_rows(tmp_path) if r.get("task_id") == first.task_id]
    # The retry logged the inbound row once and spoke nothing itself; the root's free host-facts
    # row (a system row with empty text) is bookkeeping in the room's history, not speech.
    facts = [r for r in rows if r.get("summary_kind") == "host_task_facts"]
    assert first.text == "Real answer" and [r["direction"] for r in rows if r not in facts] == ["in"]
    assert rows[0]["source"] == "presence:telegram" and all(r["direction"] == "system" and r["text"] == "" for r in facts)


def _sending_agent(calls, reply, drive_root, *, part, rotate_first=False):
    """A real-pipeline agent whose turn records one delivered receipt for this conversation mid-turn."""
    def send(task):
        chat = drive_root / "logs" / "chat.jsonl"
        if rotate_first:  # the live generation rotates while the turn runs
            (drive_root / "archive").mkdir(exist_ok=True)
            chat.rename(drive_root / "archive" / "chat_20260528T000200.jsonl")
        append_jsonl(chat, {"type": "presence_delivery", "direction": "out", "chat_id": task["chat_id"], "text": part,
                            "task_id": task["id"], "transport": {"conversation_key": _event().conversation_key,
                                                                 "delivery": {"state": "delivered", "delivery_id": f"send:{part}", "part_id": "0"}}})
    return _answering_agent(calls, reply, drive_root, during=send)


def test_a_refused_retry_after_a_rotation_claims_no_sends_of_its_own(tmp_path):
    """Formerly the re-run's own receipts in the fresh live generation became its pointer's confirmed sends.
    A refused retry sends nothing and names no turn: the conversation's pointer is never written."""
    task_id = _task_id(_admission(), _event())
    chat = _lost_v1_attempt(tmp_path, task_id, chat_id=7)
    (tmp_path / "archive").mkdir()
    chat.rename(tmp_path / "archive" / "chat_20260528T000100.jsonl")
    chat.touch()  # the rotator leaves a fresh live generation behind, as in production
    calls: list = []
    kwargs = {**_v1_kwargs(tmp_path, calls), "agent_factory": lambda **_kw: _sending_agent(
        calls, "Real answer", tmp_path, part="Retry part")}
    _refused(kwargs)
    assert calls == [] and chat.read_bytes() == b"" and _read_previous_turn(tmp_path, _event().conversation_key) is None


def test_a_rotation_during_the_turn_leaves_its_sends_unknown(tmp_path):
    calls: list = []
    kwargs = _v1_kwargs(tmp_path, calls)
    run_presence_turn(**{**kwargs, "agent_factory": lambda **_kw: _sending_agent(
        calls, "Real answer", tmp_path, part="Mid part", rotate_first=True)})
    pointer = _read_previous_turn(tmp_path, kwargs["event"].conversation_key)
    assert (pointer["transport_sends"], pointer["delivery"]) == ([], "unknown")


def test_a_presence_placeholder_owes_no_terminal_projection_even_when_its_retry_is_refused(tmp_path, monkeypatch):
    """The target's terminal-projection sweep must not post the placeholder's failure into the room, and a
    refused retry originates no terminal of its own (formerly the re-run's completion did), even when a
    stale marker was inherited."""
    from ouroboros.terminal_projection import (
        SETTLEMENT_NONE,
        reconcile_terminal_projections,
        settle_terminal_projection,
    )

    task_id = _task_id(_admission(), _event())
    _reconciled(tmp_path, monkeypatch, task_id)
    assert reconcile_terminal_projections(tmp_path) == 0  # nothing owed for a placeholder
    assert not [row for row in _chat_rows(tmp_path) if row.get("type") == "task_summary"]
    # An install on the previous release already recorded readiness for the placeholder (its chat append
    # failed): the direct settlement path, as startup recovery calls it, must not publish it either.
    write_task_result(tmp_path, task_id, STATUS_FAILED, canonical_terminal_projection_ready={
        "summary_id": f"task-terminal:{task_id}", "token": "stale", "attempt": {}, "task_done_ts": "2026-05-28T00:00:05+00:00",
        "chat_id": 7})
    assert settle_terminal_projection(tmp_path, task_id) == SETTLEMENT_NONE
    assert not [row for row in _chat_rows(tmp_path) if row.get("type") == "task_summary"]
    # An install that ran the sweep before this rule left a failed marker on the placeholder.
    write_task_result(tmp_path, task_id, STATUS_FAILED, canonical_terminal_projection={
        "summary_id": f"task-terminal:{task_id}", "summary_kind": "terminal_root_projection", "attempt": {}})
    built: list = []
    _assert_fails_closed(tmp_path, task_id, _retry_kwargs(tmp_path, built, version=0), built)
    stored = load_task_result(tmp_path, task_id)
    assert is_reconciled_presence_placeholder(stored) and "superseded_placeholder" not in stored
    assert {"canonical_terminal_projection", "canonical_terminal_projection_ready"} <= set(stored)  # not moved aside
    assert reconcile_terminal_projections(tmp_path) == 0
    assert not [row for row in _chat_rows(tmp_path) if row.get("type") == "task_summary"]
    # An ordinary orphaned root (no presence identity) still gets its failed terminal row.
    _sweep(tmp_path, monkeypatch, "plain-orphan", metadata={"source": "chat"})
    assert reconcile_terminal_projections(tmp_path) == 1
    assert [row["status"] for row in _chat_rows(tmp_path)
            if row.get("type") == "task_summary" and row.get("task_id") == "plain-orphan"] == ["failed"]
