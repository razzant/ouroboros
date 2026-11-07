"""#2780: a new work from a Presence caller, its destination named by the mind (owner's choice A, 4 November 2026).

A live turn that receives the owner's answer to its private question may start the owner's own work: the same
root a private late answer starts, with the Main answer row as its owner origin, the place's read and stop kept and
nothing of the turn's speaker, focus or limits. Omission or a wrong selection refuses before any effect, and only
an outward start touches the conversation's handoff. A schedule names its destination too, and the owner's work is not
scheduled from there: that root schedules for itself (owner's decision, 5 November 2026). Real question, HTTP answer,
drain, tool dispatch, promote handler, readers, Continue and startup check; the model, the queue row store and Safety are
doubles (the shared ``contact_runtime``). Nothing is sent anywhere. The connector door, room and scheduler consumers:
``test_presence_private_consumers``.
"""
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.owner_quiz import quiz_states
from ouroboros.presence_owner_contact import flush_owner_events
from ouroboros.tools.registry import ToolContext
from supervisor import queue as task_queue
from supervisor import workers
from tests.test_place_focus import focus as focus  # fixture
from tests.test_presence_owner_contact import _answer, _ask, _exercise_related_root, _pooled_metadata, _run
from tests.test_presence_owner_contact import contact_runtime as contact_runtime  # fixture

PRIVATE_START = {"objective": "Investigate the source", "title": "Source research"}
SPEAKER_KEYS = {"presence", "presence_final_route", "presence_binding_authority"}
FOCUS_KEYS = {"initiator", "usage_category", "consciousness_autonomy", "consciousness_cause", "focus", "wake_cause",
              "origin_wake_task_id", "disabled_tools", "runtime_mode_cap", "root_cost_ceiling_usd", "model", "model_role"}


def _answered(r, ctx):
    """The real private question, the owner's HTTP answer and this live turn's own drain: the handle it then holds."""
    from ouroboros.owner_relay import held_owner_messages

    assert _ask(ctx).startswith("OK:")
    qid, = quiz_states(r.root, ctx.task_id)
    flush_owner_events(r.root, ctx.pending_events)
    r.pump()
    assert _answer(r, ctx.task_id, qid).status_code == 200
    _drain_incoming_messages([], queue.Queue(), r.root, ctx.task_id, None, set(), owner_ctx=ctx)
    [handle] = held_owner_messages(ctx)
    return handle


def _admitting(r, monkeypatch, ctx):
    """The real promote handler and queue on this runtime's root; the turn's other events go on as before."""
    from ouroboros.utils import atomic_write_json
    from supervisor.events import _handle_promote_chat_to_task

    emitted, turn_events = [], ctx.event_queue
    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    atomic_write_json(r.root / "state" / "queue_snapshot.json", {"worker_total": 1, "pending": [], "running": []})
    host = SimpleNamespace(DRIVE_ROOT=r.root, WORKERS={0: SimpleNamespace()}, PENDING=task_queue.PENDING, bridge=None,
        append_jsonl=lambda *_a, **_kw: None, persist_queue_snapshot=lambda **_kw: True,
        enqueue_task=task_queue.enqueue_task, load_state=lambda: {"owner_chat_id": 1})

    def put_nowait(event):
        if event.get("type") != "promote_chat_to_task":
            return turn_events.put_nowait(event)
        emitted.append(dict(event))
        return _handle_promote_chat_to_task(event, host)

    ctx.event_queue = SimpleNamespace(put_nowait=put_nowait)
    return host, emitted


def _worker_ctx(r, row, **kw):
    return ToolContext(repo_dir=r.root, drive_root=r.root, budget_drive_root=r.root, task_id=row["id"],
                       current_chat_id=row["chat_id"], task_metadata=_pooled_metadata(row),
                       task_contract=row["task_contract"], is_direct_chat=False, **kw)


def test_a_live_answer_starts_the_owners_private_work_like_a_late_answer_and_leaves_the_turn_as_it_was(
        contact_runtime, monkeypatch):
    """The reported incident: a live Presence turn receives the answer and starts work. Real question, answer, drain,
    dispatch and admission; the turn stays the place's, and the root is the same ordinary owner work a late answer
    starts: the same event shape, the same reader results and the whole related-root exerciser."""
    from copy import deepcopy

    from ouroboros.consciousness_authority import is_consciousness_origin, task_mode_capped_light, work_kind
    from ouroboros.dialogue_provenance import presence_caller_binding, run_origin
    from ouroboros.owner_relay import held_owner_messages
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_routing import _start_task
    from ouroboros.tools.followup import _handle_schedule_followup
    from ouroboros.tools.registry_guards import _disabled_tools

    r, seen = contact_runtime, {}
    binding = r.admission.binding_id

    def during(ctx, task, _agent):
        handle = _answered(r, ctx)
        host, emitted = _admitting(r, monkeypatch, ctx)
        task["metadata"]["chat_attachment_uploads"] = [{"upload": "the-place-sent-this", "name": "external.txt"}]
        before = deepcopy(task["metadata"])
        out = ctx._consumer_tools.execute("start_task", {
            **PRIVATE_START, "context_requires_self_body_docs": True, "destination": "owner", "owner_quote": handle})
        assert out.startswith("OK: task") and "owner's private work" in out, out
        assert task["metadata"] == before and ctx.task_metadata is task["metadata"]
        assert not hasattr(ctx, "_swarm_handoff_attempt")
        seen.update(ctx=ctx, host=host, event=emitted[0], root=host.PENDING[0], tid=ctx.task_id)

    assert _run(r, during).outcome == "silent"
    ctx, host, event, root = seen["ctx"], seen["host"], seen["event"], seen["root"]
    words = "Read the public context only; do not promise implementation."

    # The event asks for nothing of the speaker, and for nothing of its focus.
    assert not (SPEAKER_KEYS | FOCUS_KEYS | {"task_contract"}) & set(event), sorted(set(event))
    assert (event["chat_id"], event["work_kind"]) == (1, "owner") and event["attachment_uploads"] == []
    assert event["presence_binding_relation"] == {"binding_id": binding}
    [question] = quiz_states(r.root, seen["tid"])
    assert event["source_text"] == words and event["client_message_id"] == f"quiz_answer:{seen['tid']}:{question}"
    assert event["source_ref"]["client_message_id"] == event["client_message_id"]  # the Main row, never the Presence event's id

    # Late-answer parity: the same start, field by field, from the owner's own late turn.
    block = dict(next(iter(quiz_states(r.root, seen["tid"]).values())), answered_after_terminal=True)
    late = SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, drive_root=r.root, budget_drive_root=str(r.root),
        project_id="", task_id="owner-late-turn", is_direct_chat=True,
        task_metadata={"origin_message_ref": {"chat_id": 1, "client_message_id": "late", "ts": "t", "text_sha256": "0" * 64},
                       "origin_message_text": words, "client_message_id": "late",
                       "late_answer": {"task_id": seen["tid"], "quiz_id": block["quiz_id"],
                                       "answer_snapshot": {"task_id": seen["tid"], "quiz": block}}})
    with monkeypatch.context() as scoped:
        scoped.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                       lambda _ctx, evt: (seen.update(late=dict(evt)) or "live", {"status": "scheduled"}))
        _start_task(late, PRIVATE_START["objective"], title=PRIVATE_START["title"], context_requires_self_body_docs=True)
    assert set(event) - {"destination", "owner_quote"} == set(seen["late"])
    assert {key: seen["late"][key] for key in ("presence_binding_relation", "work_kind", "chat_id", "source_text")} == {
        key: event[key] for key in ("presence_binding_relation", "work_kind", "chat_id", "source_text")}

    # Admitted as the owner's ordinary work: own contract, the owner's words, no limit of the focus.
    assert root["source"] == "promote_chat_to_task" and root["chat_id"] == 1 and not root.get("actor_id")
    assert not root.get("project_id")  # no owner project was selected; never inherit the Presence project's room
    assert root["origin_message_ref"] == event["source_ref"] and root["origin_message_text"] == words
    assert not (SPEAKER_KEYS | FOCUS_KEYS) & set(root["metadata"]), sorted(root["metadata"])
    assert "capability_ceiling" not in root["task_contract"] and not root["task_contract"].get("disabled_tools")
    assert root["metadata"]["work_kind"] == "owner" == work_kind(root["metadata"]) and not is_consciousness_origin(root["metadata"])
    assert words in str(root["metadata"]["owner_corpus"]) and r.event.text not in str(root)
    assert run_origin(root)["owner_ingress"] is True
    stored = load_task_result(r.root, root["id"])
    assert stored["status"] == "scheduled" and stored["metadata"]["presence_binding_relation"] == {"binding_id": binding}
    assert stored.get("source") != "presence_promote"
    billing = root["metadata"]["billing_group"]  # the real queue's binding: this work's own group, as any owner root's
    assert billing["billing_group_id"] == root["id"] and billing["billing_group_limit_source"] == "initial_task_admission"
    worker = _worker_ctx(r, root)
    assert presence_caller_binding(worker) is None and not _disabled_tools(worker)
    assert not task_mode_capped_light(worker.task_metadata)

    # The Main row itself: the worker holds it as a quotable owner source, and its reference resolves.
    held = held_owner_messages(worker)
    assert [row["client_message_id"] for row in held.values()] == [event["client_message_id"]]

    # Two follow-ups keep the relation; then the whole late-answer exerciser (nested start, children, tools, stop).
    for args in ({"notify": True}, {"relation": "independent"}):
        assert _handle_schedule_followup(worker, objective="Revisit it", run_at="2030-01-01T00:00:00Z", **args).startswith("FOLLOWUP_SCHEDULED")
        assert "FOLLOWUP_DESTINATION_NOT_APPLICABLE" in _handle_schedule_followup(
            worker, objective="Revisit it", run_at="2030-01-01T00:00:00Z", destination="owner", **args)
    [followup] = [row for row in task_queue.list_scheduled_tasks(r.root)["tasks"] if row.get("kind") != "notify"]
    fired = task_queue._task_from_schedule(followup)["metadata"]
    assert fired["presence_binding_relation"] == {"binding_id": binding} and fired["work_kind"] == "owner"
    assert not (SPEAKER_KEYS | FOCUS_KEYS) & set(fired)
    _exercise_related_root(r, ctx, root, host, monkeypatch, seen["tid"])


def test_a_presence_start_refuses_a_missing_or_unusable_destination_before_any_effect(contact_runtime, monkeypatch):
    """Owner's choice A: no default and no guess. Every refusal happens before a durable intent, an event or the slot."""
    from copy import deepcopy

    from ouroboros.task_results import list_task_results, write_task_result
    from ouroboros.tools.control_routing import _start_task

    r = contact_runtime

    def refused(ctx, emitted, words, *, direct=False, **args):
        known = {row.get("task_id") for row in list_task_results(r.root)}
        out = (_start_task(ctx, **PRIVATE_START, **args) if direct
               else ctx._consumer_tools.execute("start_task", {**PRIVATE_START, **args}))
        assert "TOOL_ARG_ERROR (start_task)" in out and words in out and "Nothing was scheduled" in out, (args, out)
        assert not emitted and {row.get("task_id") for row in list_task_results(r.root)} == known, args
        assert not hasattr(ctx, "_swarm_handoff_attempt"), args

    def during(ctx, task, _agent):
        handle = _answered(r, ctx)
        _host, emitted = _admitting(r, monkeypatch, ctx)
        for args, words in (({}, "destination must be"), ({"destination": "nowhere"}, "destination must be"),
                            ({"destination": "Owner", "owner_quote": handle}, "destination must be"),
                            ({"destination": ["owner"], "owner_quote": handle}, "destination must be"),
                            ({"owner_quote": handle}, "destination must be"),
                            ({"destination": "place", "owner_quote": handle}, "owner_quote belongs"),
                            ({"destination": "place", "owner_quote": []}, "owner_quote belongs"),
                            ({"destination": "place", "owner_quote": False}, "owner_quote belongs"),
                            ({"destination": "owner"}, "owner_quote must be"),  # one held answer is no default either
                            ({"destination": "owner", "owner_quote": "row:1@nowhere#000000000000"}, "owner_quote must be"),
                            ({"destination": "owner", "owner_quote": [handle]}, "owner_quote must be")):
            refused(ctx, emitted, words, **args)
        assert handle in _start_task(ctx, "x", title="X", destination="owner")  # the refusal names what is held

        held = ctx._held_owner_messages
        original = held[handle]

        def holding(row):
            held[handle] = row
        # An ordinary owner row held by relay proves no private question.
        holding({"direction": "in", "chat_id": 1, "text": "Please do it", "ts": "2026-01-01T00:00:00Z"})
        refused(ctx, emitted, "owner_quote must be", destination="owner", owner_quote=handle)
        quiz = original["quiz"]
        # Another place's question, and a question another work asked for this very place.
        holding({**original, "quiz": {**quiz, "presence_contact": {**quiz["presence_contact"], "binding_id": "other"}}})
        refused(ctx, emitted, "another place's question", destination="owner", owner_quote=handle)
        write_task_result(r.root, "other-work", "completed", delegation_role="root")
        holding({**original, "task_id": "other-work", "quiz": {**quiz, "presence_contact": {**quiz["presence_contact"], "task_id": "other-work"}}})
        refused(ctx, emitted, "another work asked", destination="owner", owner_quote=handle)
        # Held words that differ from the saved answer, and a saved row a store cannot read now.
        holding({**original, "quiz": {**quiz, "comment": "Publish everything."}})
        refused(ctx, emitted, "no matching saved row", destination="owner", owner_quote=handle)
        holding(original)
        from ouroboros.store import StoreBusy

        def busy(*_a, **_k):
            raise StoreBusy("record busy", reason="contention")

        with monkeypatch.context() as scoped:
            scoped.setattr("ouroboros.project_dialogue.resolve_owner_message_source", busy)
            refused(ctx, emitted, "cannot be read now", destination="owner", owner_quote=handle)
        with monkeypatch.context() as scoped:
            scoped.setattr("ouroboros.owner_quiz.question_task_owned_by", busy)
            refused(ctx, emitted, "question ownership cannot be read now", destination="owner", owner_quote=handle)
        # A legacy ceiling keeps narrowing, and a delegated caller reports upward (the row stays held, so only these decide).
        saved = deepcopy(ctx.task_contract)
        ctx.task_contract = {**saved, "capability_ceiling": {}}
        refused(ctx, emitted, "legacy capability ceiling", direct=True, destination="owner", owner_quote=handle)
        ctx.task_contract = saved
        ctx.task_metadata = {**task["metadata"], "parent_task_id": "parent", "delegation_role": "subagent"}
        refused(ctx, emitted, "reports to its parent", direct=True, destination="owner", owner_quote=handle)
        ctx.task_metadata = task["metadata"]

    _run(r, during)
    # Work holding no Presence authority has nothing to choose, and cannot name a choice.
    plain = SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, drive_root=r.root, budget_drive_root=str(r.root),
                            project_id="", task_id="owner-turn", is_direct_chat=True, task_metadata={}, task_contract={})
    for args in ({"destination": "owner"}, {"destination": "place"}, {"owner_quote": "row:1@x#y"},
                 *({key: value} for key in ("destination", "owner_quote") for value in (0, False, [], {}))):
        out = _start_task(plain, "x", title="X", **args)
        assert "apply only to a start from a Presence" in out and not plain.pending_events, args


def test_admission_reads_the_private_origin_itself_before_any_project_or_source_effect(contact_runtime, monkeypatch):
    """A producer's claim is not the answer: a changed reference or a carrier beside it refuses at the supervisor,
    before the source's off-loop work; the refusal reserves no handoff and leaves a durable typed record."""
    from ouroboros.presence_owner_contact import private_start_refusal
    from ouroboros.task_results import load_task_result

    r, effects = contact_runtime, []
    monkeypatch.setattr("ouroboros.promotion_source.resolve_promote_source", lambda *a, **k: effects.append(a))

    def during(ctx, task, _agent):
        handle = _answered(r, ctx)
        host, emitted = _admitting(r, monkeypatch, ctx)
        admit = ctx.event_queue.put_nowait

        for tamper, reason in ((lambda e: e["source_ref"].update(client_message_id="quiz_answer:other:other"), "private_owner_source_invalid"),
                               (lambda e: e.update(presence_binding_relation={"binding_id": "other"}), "private_owner_source_invalid"),
                               (lambda e: e.update(presence={"binding_id": r.admission.binding_id}), "private_owner_source_invalid"),
                               (lambda e: e.update(source_text="Different words."), "private_owner_source_invalid"),
                               (lambda e: e.update(chat_id=77), "private_owner_source_invalid"),
                               (lambda e: e.update(client_message_id="other"), "private_owner_source_invalid"),
                               (lambda e: e.update(attachment_uploads=[{"upload": "foreign"}]), "private_owner_source_invalid"),
                               (lambda e: e.update(work_kind="consciousness"), "private_owner_source_invalid"),
                               (lambda e: e.update(owner_quote="another-answer"), "private_owner_source_invalid"),
                               (lambda e: e.update(source="https://example.test/repo.git"), "private_owner_source_invalid"),
                               (lambda e: e.update(task_contract={}), "private_owner_source_invalid"),
                               (lambda e: e.update(destination="place", presence={"binding_id": r.admission.binding_id}),
                                "private_owner_source_invalid")):
            ctx.event_queue = SimpleNamespace(put_nowait=lambda event: (tamper(event), admit(event))[1])
            out = ctx._consumer_tools.execute("start_task", {**PRIVATE_START, "source": "https://example.test/repo.git",
                                                             "destination": "owner", "owner_quote": handle})
            assert out.startswith("⚠️ PROMOTE_REJECTED") and "its owner answer could not be verified" in out, out
            assert not host.PENDING and not effects and not hasattr(ctx, "_swarm_handoff_attempt")
            assert load_task_result(r.root, emitted[-1]["task_id"])["promotion_admission"]["reason"] == reason
        ctx.event_queue = SimpleNamespace(put_nowait=admit)
        assert private_start_refusal(r.root, {"destination": "owner"}) == "private_owner_source_invalid"
        assert private_start_refusal(r.root, {"presence": {}, "source_ref": "anything"}) == ""  # not this reader's
        with monkeypatch.context() as scoped:
            from ouroboros.store import StoreBusy

            def busy(*_a, **_k):
                raise StoreBusy("record busy", reason="contention")

            scoped.setattr("ouroboros.project_dialogue.resolve_owner_message_source", busy)
            busy_event = dict(emitted[0])
            busy_event["source"] = ""
            assert private_start_refusal(r.root, busy_event) == "private_owner_source_unavailable"

    _run(r, during)


@pytest.mark.parametrize("private_outcome", ["scheduled", "refused", "unconfirmed", "no_workers"])
@pytest.mark.parametrize("first", ["private", "outward"])
def test_only_the_outward_start_holds_the_first_handoff_in_either_order(contact_runtime, monkeypatch, first, private_outcome):
    """A private start takes no part in the conversation's handoff, whatever its outcome and in either order; the
    outward start keeps exactly what it has without one: the speaker carrier, the held slot, the published work."""
    from ouroboros.presence_context import PRESENCE_FINAL_ROUTE_KEY as KEY
    from ouroboros.presence_continuation import _handoff_ref
    from ouroboros.presence_runner import build_presence_result_event
    from ouroboros.tools.control_routing import _start_task

    r = contact_runtime

    def during(ctx, task, _agent):
        handle = _answered(r, ctx)
        host, emitted = _admitting(r, monkeypatch, ctx)

        def private():
            if private_outcome == "scheduled":
                return _start_task(ctx, "Owner's research", title="Research", destination="owner", owner_quote=handle)
            status = "rejected" if private_outcome == "refused" else "unconfirmed"
            with monkeypatch.context() as scoped:
                scoped.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                               lambda _ctx, _evt: ("live", {"status": status, "reason": "worker_pool_unavailable"}))
                if private_outcome == "no_workers":
                    scoped.setattr("ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot", lambda _ctx: "no_workers")
                return _start_task(ctx, "Owner's research", title="Research", destination="owner", owner_quote=handle)

        def outward():
            out = _start_task(ctx, "Answer them", title="Answer", destination="place")
            assert out.startswith("OK: task"), out
            return next(row for row in host.PENDING if "presence" in row["metadata"])

        if first == "private":
            private()
            assert not hasattr(ctx, "_swarm_handoff_attempt")
            root = outward()
        else:
            root = outward()
            held = dict(ctx._swarm_handoff_attempt)
            private()
            assert ctx._swarm_handoff_attempt == held
        assert root["metadata"][KEY] == {"held": True, "work_ref": root["id"], "source_task_id": ctx.task_id}
        assert root["source"] == "presence_promote"
        assert root["metadata"]["presence"]["event"] == task["metadata"]["presence"]["event"]
        assert _handoff_ref(ctx) == root["id"]
        frame = build_presence_result_event(task, "On it.", ctx, terminal_origin="model_final")
        assert (frame["outcome"], frame["work_ref"]) == ("deferred", root["id"])
        if private_outcome == "scheduled":
            [mine] = [row for row in host.PENDING if row["id"] != root["id"]]
            assert KEY not in mine["metadata"] and "presence" not in mine["metadata"]

    _run(r, during)


def _published(r, tid):
    """What the connector's retry reads back from the turn's record."""
    from ouroboros.presence_runner import presence_result_from_stored
    from ouroboros.task_results import load_task_result

    result = presence_result_from_stored(load_task_result(r.root, tid), tid, r.root)
    return result.outcome, result.work_ref, result.text


@pytest.mark.parametrize("first", ["private", "outward", None], ids=["private-first", "outward-first", "private-only"])
def test_a_quiet_final_defers_only_to_the_outward_start_and_says_nothing(contact_runtime, monkeypatch, first):
    """The mind's silent finish, whichever start came first; private work alone leaves the turn plainly silent."""
    from ouroboros.presence_runner import build_presence_result_event
    from ouroboros.tools.control_routing import _start_task

    r, seen = contact_runtime, {}

    def during(ctx, task, _agent):
        handle = _answered(r, ctx)
        host, _emitted = _admitting(r, monkeypatch, ctx)
        private = {"objective": "Owner's research", "title": "Research", "destination": "owner", "owner_quote": handle}
        outward = {"objective": "Answer them", "title": "Answer", "destination": "place"}
        for args in {"private": (private, outward), "outward": (outward, private), None: (private,)}[first]:
            assert _start_task(ctx, **args).startswith("OK: task")
        seen.update(frame=build_presence_result_event(task, "Noted for myself.", ctx, terminal_origin="model_final", silent=True),
                    outward=next((row["id"] for row in host.PENDING if "presence" in row["metadata"]), ""), tid=ctx.task_id)

    _run(r, during)
    frame, outward = seen["frame"], seen["outward"]
    assert (frame["outcome"], frame["work_ref"], frame["text"]) == ("deferred" if first else "silent", outward, "")
    assert frame["finish_note"] == "Noted for myself." and bool(outward) is bool(first)
    assert _published(r, seen["tid"]) == (frame["outcome"], outward, "")


def test_one_private_answer_can_start_two_roots(contact_runtime, monkeypatch):
    """No one-shot rule: the answer stays held and each start is its own ordinary root under the same owner origin."""
    from ouroboros.task_results import load_task_result

    r, seen = contact_runtime, {}

    def during(ctx, _task, _agent):
        handle = _answered(r, ctx)
        host, emitted = _admitting(r, monkeypatch, ctx)
        for title in ("First research", "Second research"):
            out = ctx._consumer_tools.execute("start_task", {"objective": f"{title} of the source", "title": title,
                                                             "destination": "owner", "owner_quote": handle})
            assert out.startswith("OK: task"), out
        assert not hasattr(ctx, "_swarm_handoff_attempt")
        seen.update(roots=list(host.PENDING), emitted=emitted)

    _run(r, during)
    first, second = seen["roots"]
    assert first["id"] != second["id"] and first["title"] != second["title"]
    assert first["origin_message_ref"] == second["origin_message_ref"] == seen["emitted"][0]["source_ref"]
    assert first["origin_message_text"] == second["origin_message_text"]
    for root in (first, second):
        assert load_task_result(r.root, root["id"])["status"] == "scheduled"
        assert not (SPEAKER_KEYS | FOCUS_KEYS) & set(root["metadata"])
        assert root["metadata"]["billing_group"]["billing_group_id"] == root["id"]


def test_a_private_start_from_a_project_bound_turn_is_in_main_unless_the_owner_names_a_project(contact_runtime, monkeypatch):
    """The turn's Project is the place's default for outward work and no default for the owner's: this is the omission."""
    from ouroboros.projects_registry import create_project, project_chat_for_task, project_id_for_task
    from ouroboros.tools.control_routing import _inherited_project_scope
    from tests.test_presence_project_activity import _move

    r, seen = contact_runtime, {}
    placed = create_project(r.root, "place-room", name="Place room")
    chosen = create_project(r.root, "owner-research", name="Owner research")

    def during(ctx, _task, _agent):
        handle = _answered(r, ctx)
        host, _emitted = _admitting(r, monkeypatch, ctx)
        _move(r.root, ctx.task_id, placed["id"])
        assert _inherited_project_scope(ctx) == placed["id"]
        for args in ({"destination": "owner", "owner_quote": handle}, {"destination": "place", "workspace": "none"},
                     {"destination": "owner", "owner_quote": handle, "project_id": chosen["id"], "workspace": "none"}):
            out = ctx._consumer_tools.execute("start_task", {**PRIVATE_START, **args})
            assert out.startswith("OK: task"), out
        seen.update(roots=list(host.PENDING))

    _run(r, during)
    unnamed, outward, named = seen["roots"]
    assert not unnamed.get("project_id") and unnamed["chat_id"] == 1
    assert project_id_for_task(r.root, unnamed["id"]) == "" and not project_chat_for_task(r.root, unnamed["id"])
    assert unnamed["metadata"]["resource_intent"] == {"kind": "system_repo"}
    assert outward["project_id"] == placed["id"] and project_id_for_task(r.root, outward["id"]) == placed["id"]
    assert named["project_id"] == chosen["id"] and named["chat_id"] == chosen["chat_id"] == project_chat_for_task(r.root, named["id"])


def test_a_private_start_alone_is_no_deferred_work_and_its_room_is_the_owners(contact_runtime, monkeypatch):
    from ouroboros.presence_runner import build_presence_result_event
    from ouroboros.projects_registry import create_project, project_chat_for_task

    r, seen = contact_runtime, {}
    project = create_project(r.root, "owner-research", name="Owner research")

    def during(ctx, task, _agent):
        handle = _answered(r, ctx)
        host, emitted = _admitting(r, monkeypatch, ctx)
        ctx._consumer_tools.execute("start_task", {**PRIVATE_START, "project_id": project["id"], "workspace": "none",
                                                   "destination": "owner", "owner_quote": handle})
        frame = build_presence_result_event(task, "Nothing to add here.", ctx, terminal_origin="model_final")
        assert frame["outcome"] != "deferred" and not frame["work_ref"]
        seen.update(root=host.PENDING[0], event=emitted[0], place_room=task["chat_id"])

    _run(r, during)
    assert seen["root"]["chat_id"] == project["chat_id"] != seen["place_room"]  # a room of the owner's, never the place's
    assert project_chat_for_task(r.root, seen["root"]["id"]) == project["chat_id"]


def test_a_later_arrival_and_the_final_stay_out_of_the_private_root(contact_runtime, monkeypatch):
    """No connector reads it and no place message reaches it; the outward root beside it is the negative control."""
    from ouroboros.conversation_lease import open_root_intake, root_conversation_key
    from ouroboros.dialogue_provenance import is_presence_task, presence_provenance_from_task, presence_related_work
    from ouroboros.place_intake import hand_own_arrival
    from ouroboros.place_reply import reply_carrier
    from ouroboros.presence_context import ROUTELESS, final_route
    from ouroboros.room_reader import conversation_works
    from ouroboros.utils import atomic_write_json

    r, seen = contact_runtime, {}
    binding, key = r.admission.binding_id, r.event.conversation_key

    def during(ctx, task, _agent):
        handle = _answered(r, ctx)
        host, _emitted = _admitting(r, monkeypatch, ctx)
        ctx._consumer_tools.execute("start_task", {**PRIVATE_START, "destination": "owner", "owner_quote": handle})
        ctx._consumer_tools.execute("start_task", {"objective": "Answer them", "title": "Answer", "destination": "place"})
        private, outward = host.PENDING
        atomic_write_json(r.root / "state" / "queue_snapshot.json", {"worker_total": 1, "running": [], "pending": [
            {"id": row["id"], "task": row} for row in (private, outward)]})
        assert hand_own_arrival(r.root, task["metadata"], private["id"], True, ctx.task_id) == "arrival_target_not_listed"
        seen.update(private=private, outward=outward)

    _run(r, during)
    private, outward = seen["private"], seen["outward"]
    assert root_conversation_key(private) == "" and root_conversation_key(outward) == key
    assert open_root_intake(r.root, private) is False  # its loop reads no conversation
    assert [row["task_id"] for row in conversation_works(r.root, key, [binding])] == [outward["id"]]
    assert presence_related_work(binding, private) and not is_presence_task(private)
    assert reply_carrier(private["metadata"]) == {} and reply_carrier(outward["metadata"])
    assert final_route(r.root, private["metadata"], private["id"]) == (ROUTELESS, "it carries no place conversation")
    assert presence_provenance_from_task(private) == {}


def test_observe_keeps_its_dispatch_refusal_with_every_selector(contact_runtime, monkeypatch):
    """The owner's limit stays at dispatch: no argument brings start_task back at Observe. Act reaches the handler."""
    r = contact_runtime
    monkeypatch.setattr("ouroboros.config.get_consciousness_autonomy", lambda: "observe")

    def during(ctx, task, _agent):
        assert task["metadata"]["consciousness_autonomy"] == "observe" and "start_task" in task["metadata"]["disabled_tools"]
        handle = _answered(r, ctx)
        host, emitted = _admitting(r, monkeypatch, ctx)
        for args in ({"destination": "owner", "owner_quote": handle}, {"destination": "place"}, {}):
            out = ctx._consumer_tools.execute("start_task", {**PRIVATE_START, **args})
            assert "RESOURCE_CONSTRAINT_BLOCKED" in out and "start_task" in out, out
        for args in ({"destination": "owner"}, {"destination": "place"}, {}):
            out = ctx._consumer_tools.execute("schedule_followup", {
                "objective": "Revisit it", "run_at": "2030-01-01T00:00:00Z", "relation": "independent", **args})
            assert "RESOURCE_CONSTRAINT_BLOCKED" in out and "schedule_followup" in out, out
        assert task_queue.list_scheduled_tasks(r.root)["tasks"] == []
        assert not emitted and not host.PENDING and not hasattr(ctx, "_swarm_handoff_attempt")
        assert "disabled_tools" in task["metadata"] and task["metadata"]["consciousness_autonomy"] == "observe"

    _run(r, during)


def test_continue_and_cold_resume_keep_the_private_roots_owner_origin_and_relation(contact_runtime, monkeypatch):
    """The saved Main answer row is the origin both ways: the startup check accepts it, a fresh worker holds it, a
    Continue keeps the relation, the owner's words and the owner's kind, and no carrier appears on either."""
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.dialogue_provenance import presence_caller_binding, presence_related_work
    from ouroboros.owner_relay import held_owner_messages
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE

    r, seen = contact_runtime, {}
    binding = r.admission.binding_id

    def during(ctx, task, _agent):
        handle = _answered(r, ctx)
        host, _emitted = _admitting(r, monkeypatch, ctx)
        ctx._consumer_tools.execute("start_task", {**PRIVATE_START, "destination": "owner", "owner_quote": handle})
        seen.update(root=host.PENDING[0])

    _run(r, during)
    root = seen["root"]
    env = SimpleNamespace(drive_root=r.root, budget_drive_root=r.root)
    words = root["origin_message_text"]

    # Resume: a fresh worker built from the saved row passes the startup source check and holds the answer again.
    resumed = json.loads(json.dumps({key: value for key, value in root.items() if not key.startswith("_")}))
    assert "unavailable" not in str(validate_task_authority_sources(env, resumed)), resumed
    worker = _worker_ctx(r, resumed)
    assert presence_caller_binding(worker) is None
    assert [row["type"] for row in held_owner_messages(worker).values()] == ["quiz_answer"]

    # Continue: the same work, admitted from its interrupted record.
    _queue, _state, pool = _install_queue(r.root, monkeypatch)
    write_task_result(r.root, root["id"], "running", chat_id=1, root_task_id=root["id"], delegation_role="root",
                      origin_message_ref=root["origin_message_ref"], origin_message_text=words, metadata=root["metadata"],
                      billing_group={"billing_group_id": root["id"], "billing_group_limit_usd": 20.0,
                                     "billing_group_limit_source": "initial_task_admission",
                                     "billing_group_limit_revision": "admission-1"})
    write_task_result(r.root, root["id"], "cancelled", result="interrupted",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    ack = admit_continuation(root["id"], action_nonce=NONCE)
    assert ack["ok"], ack
    successor = next(row for row in pool.PENDING if row["id"] == ack["successor_task_id"])
    metadata = successor["metadata"]
    assert metadata["presence_binding_relation"] == {"binding_id": binding} and metadata["work_kind"] == "owner"
    assert metadata["origin_message_ref"] == root["origin_message_ref"] and metadata["origin_message_text"] == words
    assert not (SPEAKER_KEYS | {"initiator", "usage_category", "consciousness_autonomy", "focus"}) & set(metadata)
    assert presence_related_work(binding, successor) and "presence" not in load_task_result(r.root, successor["id"])["metadata"]
    assert "unavailable" not in str(validate_task_authority_sources(env, {
        **successor, "origin_message_ref": metadata["origin_message_ref"], "origin_message_text": words}))


def test_the_recorded_successor_holding_the_answer_starts_it_and_a_card_row_is_an_owner_source_only_when_answered(
        contact_runtime, monkeypatch):
    """A Continue of the asking turn holds only the binding: the answer is still its to act on. The row is an owner
    source by the owner's words alone: not its host frame, and never a closed, open or form-outcome card."""
    from ouroboros.owner_relay import held_owner_messages
    from ouroboros.project_dialogue import _entry_source_identities, _text_sha256
    from ouroboros.tools.control_routing import _start_task

    r, seen = contact_runtime, {}

    def during(ctx, _task, _agent):
        seen.update(handle=_answered(r, ctx), held=dict(held_owner_messages(ctx)), asker=ctx.task_id)

    _run(r, during)
    binding, row = r.admission.binding_id, seen["held"][seen["handle"]]
    monkeypatch.setattr("ouroboros.owner_continue.recorded_continuation",
                        lambda _root, asker, task: (asker, task) == (seen["asker"], "successor"))
    metadata = {"presence_binding_authority": {"binding_id": binding}, "root_task_id": "successor",
                "delegation_role": "root", "continuation": {"owner_sources": {"held_owner_messages": seen["held"]}}}
    events = []
    successor = SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, drive_root=r.root, project_id="",
                                budget_drive_root=str(r.root), task_id="successor", is_direct_chat=False,
                                task_metadata=metadata, task_contract={})
    monkeypatch.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                        lambda _ctx, evt: (events.append(evt) or "live", {"status": "scheduled"}))
    assert _start_task(successor, "Do it", title="Do it", destination="owner", owner_quote=seen["handle"]).startswith("OK: task")
    assert events[0]["presence_binding_relation"] == {"binding_id": binding}
    assert "presence_binding_authority" not in events[0] and events[0]["work_kind"] == "owner"

    words, quiz = events[0]["source_text"], row["quiz"]
    assert _text_sha256(words) in {key[3] for key in _entry_source_identities(row)}
    assert _text_sha256(row["text"]) not in {key[3] for key in _entry_source_identities(row)}
    for changed in ({"state": "closed"}, {"state": "open"}, {"secret_request": {"plugin_name": "x"}}):
        assert _entry_source_identities({**row, "quiz": {**quiz, **changed}}) == set()
    assert _entry_source_identities({**row, "type": "quiz_closed"}) == _entry_source_identities({**row, "source": "other"}) == set()


CARRIERS = {
    "speaker": ({"presence": {"binding_id": "b" * 32}}, {}),
    "work-acting-for-a-place": ({"presence_binding_authority": {"binding_id": "mail-room"}}, {}),
    "retained-legacy-ceiling": ({"presence": {"binding_id": "b" * 32}}, {"capability_ceiling": {"tools": ["schedule_followup"]}}),
    "lost-carrier-under-a-ceiling": ({}, {"capability_ceiling": {"tools": ["schedule_followup"]}}),
}


def _schedule_writes(ctx):
    """Every durable schedule fact of this root: the table's rows and its audit stream."""
    events = ctx.drive_root / "logs" / "events.jsonl"
    audit = [row for row in events.read_text(encoding="utf-8").splitlines() if "schedule_mutation" in row] if events.exists() else []
    return task_queue.list_scheduled_tasks(ctx.drive_root)["tasks"], audit


def _carrying(tmp_path, carrier):
    from tests.test_schedule_followup import _ctx

    ctx = _ctx(tmp_path)
    metadata, contract = CARRIERS[carrier]
    ctx.task_metadata.update(metadata)
    ctx.task_contract.update(contract)
    return ctx


@pytest.mark.parametrize("carrier", CARRIERS)
def test_independent_work_from_a_carrier_names_its_destination_before_any_schedule_or_audit_write(tmp_path, carrier):
    """Owner's choice A at the schedule: no default, and no reading of its objective. The owner's route is named, not taken."""
    from tests.test_schedule_followup import _followup

    ctx = _carrying(tmp_path, carrier)
    for args in ({}, {"destination": None}, {"destination": ""}, {"destination": "nowhere"}, {"destination": "Owner"},
                 {"destination": ["owner"]}, {"destination": 0}):
        out = _followup(ctx, **args)
        assert "FOLLOWUP_DESTINATION_REQUIRED" in out and "Nothing was scheduled" in out, args
        assert _schedule_writes(ctx) == ([], []) and not ctx.pending_events, args
    out = _followup(ctx, destination="owner")
    assert "FOLLOWUP_DESTINATION_OWNER" in out and "start_task(destination='owner', owner_quote=" in out, out
    assert "that new root" in out and "Nothing was scheduled" in out
    assert _schedule_writes(ctx) == ([], []) and not ctx.pending_events

    out = _followup(ctx, destination="place")
    assert out.startswith("FOLLOWUP_SCHEDULED"), out
    [row], audit = _schedule_writes(ctx)
    assert audit and "destination" not in json.dumps(row)  # the choice is this call's, never a field of the promise
    metadata = row["task"]["metadata"]
    assert {key: metadata[key] for key in CARRIERS[carrier][0]} == CARRIERS[carrier][0]
    assert row["followup_relation"]["kind"] == "independent"
    assert row["task"]["task_contract"].get("capability_ceiling") == CARRIERS[carrier][1].get("capability_ceiling")


@pytest.mark.parametrize("destination", ["place", "owner", "nowhere", ["place"], 0])
def test_a_destination_named_where_there_is_no_choice_refuses_rather_than_being_ignored(tmp_path, destination):
    """Ordinary work, a note and related work have nothing to choose; a contradictory choice is never silently dropped."""
    from tests.test_schedule_followup import _ctx, _followup

    ctx = _ctx(tmp_path)
    for args in ({}, {"notify": True}, {"notify": True, "relation": "related"}, {"cron": "0 9 * * 1", "run_at": None}):
        out = _followup(ctx, destination=destination, **args)
        assert "FOLLOWUP_DESTINATION_NOT_APPLICABLE" in out and "Nothing was scheduled" in out, args
    assert _schedule_writes(ctx) == ([], [])
    assert "FOLLOWUP_SUBAGENT_REFUSED" in _followup(_ctx(tmp_path, role="subagent"), destination=destination)
    assert "FOLLOWUP_NOTIFY_REFUSED" in _followup(_carrying(tmp_path, "speaker"), notify=True, destination=destination)
    assert _schedule_writes(ctx) == ([], [])


@pytest.mark.parametrize("empty", [None, ""])
def test_an_ordinary_call_keeps_working_without_a_destination(tmp_path, empty):
    from tests.test_schedule_followup import _ctx, _followup

    ctx = _ctx(tmp_path)
    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED")
    assert _followup(ctx, notify=True).startswith("FOLLOWUP_SCHEDULED")
    assert _followup(ctx, destination=empty).startswith("FOLLOWUP_SCHEDULED"), "models may send null"


def test_both_schemas_and_the_retry_notice_name_the_destination(tmp_path):
    from ouroboros.promotion_intent import promotion_intent_notice
    from ouroboros.tools.control import get_tools
    from ouroboros.tools.followup import get_tools as schedule_tools

    [tool] = [entry for entry in get_tools() if entry.name == "start_task"]
    properties = tool.schema["parameters"]["properties"]
    assert properties["destination"]["enum"] == ["place", "owner"] and "owner_quote" in properties
    assert set(tool.schema["parameters"]["required"]) == {"objective", "title"}  # ordinary callers keep their call
    assert "without one nothing starts" in tool.schema["description"]
    assert "schedules its own follow-ups" in properties["destination"]["description"]
    [schedule] = [entry for entry in schedule_tools() if entry.name == "schedule_followup"]
    properties = schedule.schema["parameters"]["properties"]
    assert properties["destination"]["enum"] == ["place", "owner"] and "owner_quote" not in properties
    assert set(schedule.schema["parameters"]["required"]) == {"objective", "relation"}
    assert "names destination" in schedule.schema["description"] and "Elsewhere omit it" in schedule.schema["description"]
    assert "start_task(destination='owner', owner_quote=...)" in properties["destination"]["description"]
    notice = promotion_intent_notice("t1", {"promotion_admission": {"status": "not_started"}, "promotion_intent": {
        "objective": "x", "title": "X", "destination": "owner", "owner_quote": "row:1@t#0"}})
    assert '"destination": "owner"' in notice and '"owner_quote": "row:1@t#0"' in notice
