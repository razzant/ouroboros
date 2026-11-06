"""Where a Presence-bound run's words land, told truthfully: the final-route reader, the send family's receipts, the pooled frame.

Real Place admission and focus construction, the real ``start_task`` producer, supervisor promote admission, the real
scheduler, the real ``build_presence_result_event`` publishing a turn's handoff (called, never changed here),
``build_llm_messages``, the real worker body, ``dispatch_event`` and ``GET /presence/work``; only the model is scripted.
Synthetic data; nothing leaves the process and no connector runs.
"""
from __future__ import annotations

import base64
import copy
import json
import pathlib
import queue as queue_module
import types

import pytest

from ouroboros.dialogue_provenance import presence_root_carrier
from ouroboros.presence_context import PRESENCE_FINAL_ROUTE_KEY as KEY, ROUTED, ROUTELESS, UNKNOWN, final_route
from ouroboros.presence_owner_contact import flush_owner_events
from ouroboros.tools.owner_delivery import UNDELIVERED, deliver_owner_event
from tests.test_place_conversation_work import (
    HEADERS, SILENT, call, dispatched, event_of, finish, promote, publish, run_root,
)
from tests.test_place_conversation_work import host as host  # fixture
from tests.test_presence_final_audience import HEADING, THREAD, _promote, _source_turn
from tests.test_presence_promises import _due_promise
from tests.test_presence_promises import fired_context as fired_context  # fixture
from tests.test_presence_project_activity import root as root  # fixture

pytestmark = [pytest.mark.usefixtures("provider_catalog_offline")]

NO_ROUTE = "No connector reads this root's final"
UNSETTLED = "whether a connector reads its final there is not settled"
SPEECH = "new speech there"
FINAL = "Nightly check: two of the three services are degraded."
LOCAL_ROOM = 'the local room of place "'
NOT_THERE = "(Ouroboros side, not the place conversation)"
SAYS_ROUTED = "Your final answer has a route there"
SAYS_ROUTELESS = "From this run only a connector tool reaches it."
SAYS_UNKNOWN = "Whether your final answer reaches it is not settled yet"


def _registry_for(root, task):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=task["id"], task_metadata=dict(task["metadata"]),
                      task_contract=task.get("task_contract") or {}, current_chat_id=task["chat_id"])
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry.set_context(ctx)
    return registry


def _rendered(first_request, task):
    return first_request({"type": "task", "text": "Compile it", **task})


def _audience(text):
    return text.split(HEADING, 1)[1].split("\\n##", 1)[0] if HEADING in text else ""


def _fired(root, pending, registry, objective="Check back on the report"):
    from supervisor.queue import check_scheduled_tasks

    _due_promise(registry, objective)
    check_scheduled_tasks()
    return next(task for task in pending if (task.get("metadata") or {}).get("schedule_occurrence"))


def _state(root, task):
    return final_route(root, task["metadata"], task["id"])[0]


def _running(root, task):
    from ouroboros.task_results import write_task_result

    write_task_result(root, task["id"], "running", metadata=task["metadata"])


def _ended(root, task, registry, answer="On it."):
    """The run's end through the real producer of its published references, then its terminal row."""
    from ouroboros.presence_runner import build_presence_result_event
    from ouroboros.task_results import write_task_result

    frame = build_presence_result_event(task, answer, registry._ctx, terminal_origin="model_final")
    write_task_result(root, task["id"], "completed", metadata=task["metadata"], result=answer)
    return frame


# --- the final-route reader: published references decide, never the pre-submission candidate -------------------


def test_only_the_held_handoff_of_a_turn_is_routed_and_only_once_the_turn_published_it(fired_context, monkeypatch):
    from ouroboros.presence_continuation import _handoff_ref

    root, _pending, first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    assert _state(root, turn) == ROUTED  # the turn's connector holds the request
    first = _promote(root, monkeypatch, turn, registry)
    second = _promote(root, monkeypatch, turn, registry)
    assert _handoff_ref(registry._ctx) == first["id"]
    assert first["metadata"][KEY] == {"held": True, "work_ref": first["id"], "source_task_id": turn["id"]}
    assert second["metadata"][KEY] == {"held": False, "work_ref": second["id"], "source_task_id": turn["id"]}
    assert KEY not in first["metadata"]["presence"]  # beside the carrier a schedule copies, never inside it

    # A true candidate is not publication: before the turn's row, and while it runs, the route is unknown.
    assert final_route(root, first["metadata"], first["id"]) == (UNKNOWN, "the run that handed it off has no record yet")
    _running(root, turn)
    assert final_route(root, first["metadata"], first["id"]) == (UNKNOWN, "the run that handed it off has not ended yet")
    told = _audience(_rendered(first_request, first))
    assert UNSETTLED in told and "has not ended yet" in told and "If one does, a non-silent" in told
    assert NO_ROUTE not in told

    frame = _ended(root, turn, registry)
    assert (frame["outcome"], frame["work_ref"]) == ("deferred", first["id"])  # what the connector receives
    assert _state(root, first) == ROUTED
    told = _audience(_rendered(first_request, first))
    assert told.startswith("\\n\\nThis root answers the place conversation") and SPEECH in told and UNSETTLED not in told

    assert final_route(root, second["metadata"], second["id"]) == (
        ROUTELESS, "it is not the first work the run that started it handed off")
    told = _audience(_rendered(first_request, second))
    assert NO_ROUTE in told and SPEECH not in told and "People receive only what a connector tool sends" in told


def test_a_parked_author_publishes_its_child_while_it_still_runs(fired_context, monkeypatch):
    """Not every published reference waits for a terminal: the continuing author's own poll names its child."""
    from ouroboros.presence_continuation import work_view
    from ouroboros.task_results import load_task_result, write_task_result

    root, _pending, _first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    _running(root, turn)
    assert _state(root, first) == UNKNOWN
    write_task_result(root, turn["id"], "running", metadata=turn["metadata"], presence_continuation={
        "child_work_ref": first["id"], "initial": {"status": "continuing", "outcome": "deferred", "text": "",
                                                   "output_ref": "", "work_ref": ""}})
    status, view = work_view(load_task_result(root, turn["id"]), turn["id"], root)
    assert (status, view["status"], view["child_work_ref"]) == (202, "pending", first["id"])  # what a connector polls
    assert final_route(root, first["metadata"], first["id"]) == (
        ROUTED, "that conversation's turn published it as the work its connector observes")


def test_a_retained_turn_that_kept_its_child_reference_published_it(fired_context, monkeypatch):
    """A lost attempt answers with a retained refusal naming its real child: the connector observes that child."""
    from ouroboros.presence_runner import _refused_turn
    from ouroboros.task_results import load_task_result

    root, _pending, _first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    _running(root, turn)
    refusal = _refused_turn(root, turn["id"], load_task_result(root, turn["id"]), "presence_attempt_outcome_unknown",
                            work_ref=first["id"])
    assert (refusal.disposition, refusal.work_ref) == ("retained", first["id"])  # what the connector receives
    assert _state(root, first) == ROUTED


def test_a_turn_ending_silent_beside_its_handoff_passes_no_work_to_a_connector(fired_context, monkeypatch):
    """Its row names the child, but both installed connectors observe a turn's work_ref only when it is deferred."""
    root, _pending, first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    _running(root, turn)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (False, "flagged"))  # refused at its end
    frame = _ended(root, turn, registry)
    assert (frame["outcome"], frame["work_ref"]) == ("silent", first["id"])
    assert final_route(root, first["metadata"], first["id"]) == (
        ROUTELESS, "the run that handed it off ended without passing it to a connector")
    assert NO_ROUTE in _audience(_rendered(first_request, first))


def test_a_refused_first_attempt_is_still_the_held_one_so_a_later_root_gets_no_route(fired_context, monkeypatch):
    from ouroboros.presence_continuation import _handoff_ref
    from ouroboros.tools.control_routing import _finish_swarm_handoff

    root, _pending, _first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    _finish_swarm_handoff(registry._ctx, {"task_id": "refused"}, "refused", status="rejected", reason="pool")
    late = _promote(root, monkeypatch, turn, registry)
    assert _handoff_ref(registry._ctx) == "" and late["metadata"][KEY]["held"] is False
    assert _state(root, late) == ROUTELESS


def test_a_first_attempt_whose_confirmation_timed_out_is_unknown_then_routeless_when_admitted_late(fired_context, monkeypatch):
    """No marker: the turn never names an unconfirmed attempt, so its terminal alone settles the late root."""
    from ouroboros.presence_continuation import _handoff_ref
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_routing import _start_task
    from supervisor import queue, workers
    from supervisor.events import _handle_promote_chat_to_task
    from tests.test_promote_chat_flow import _confirm_promote

    root, _pending, first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    ctx = registry._ctx
    _running(root, turn)
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue, "DRIVE_ROOT", str(root))
    monkeypatch.setattr("ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    sent = []
    with monkeypatch.context() as timed_out:
        # The supervisor's answer never reached the producer in time: the real timeout's outcome.
        timed_out.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                          lambda _ctx, evt: sent.append(evt) or ("live", {"status": "unconfirmed", "reason": "confirmation_timeout"}))
        assert _start_task(ctx, "Compile the requested report", title="Report", destination="place").startswith("Not confirmed yet")
    assert ctx._swarm_handoff_attempt["status"] == "unconfirmed" and _handoff_ref(ctx) == ""
    assert "presence_work_ref" not in (load_task_result(root, turn["id"]).get("metadata") or {})  # no reference yet

    # Then the supervisor admits that very request.
    _confirm_promote(monkeypatch)
    pending: list = []
    handler = types.SimpleNamespace(
        DRIVE_ROOT=root, WORKERS={0: types.SimpleNamespace()}, PENDING=pending, bridge=None,
        append_jsonl=lambda *_a, **_k: None, persist_queue_snapshot=lambda **_k: True,
        enqueue_task=lambda task: pending.append(dict(task)) or pending[-1], load_state=lambda: {"owner_chat_id": 1})
    _handle_promote_chat_to_task(sent[0], handler)
    [late] = pending
    assert late["metadata"][KEY] == {"held": True, "work_ref": late["id"], "source_task_id": turn["id"]}
    assert _state(root, late) == UNKNOWN  # the turn still runs: a candidate, not a route
    assert UNSETTLED in _audience(_rendered(first_request, late))

    frame = _ended(root, turn, registry)
    assert frame["work_ref"] == "" and frame["outcome"] != "deferred"  # the turn named no work
    assert final_route(root, late["metadata"], late["id"]) == (ROUTELESS, "the run that handed it off ended without passing it to a connector")
    told = _audience(_rendered(first_request, late))
    assert NO_ROUTE in told and SPEECH not in told


def test_a_root_started_by_a_promoted_root_is_never_told_it_is_routed(fired_context, monkeypatch):
    """Depth two: the turn's connector polls the first root; whether it follows that root's own child is the
    connector's (telegram-bot does, slack-bridge's ordinary paths do not), so a named child is unknown, not routed."""
    root, _pending, first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    _ended(root, turn, registry)
    assert _state(root, first) == ROUTED
    first_registry = _registry_for(root, first)
    child = _promote(root, monkeypatch, first, first_registry)
    later = _promote(root, monkeypatch, first, first_registry)
    assert child["metadata"][KEY]["held"] is True and later["metadata"][KEY]["held"] is False
    # Its source's row exists from admission on (``scheduled``), then runs: neither publishes the child yet.
    assert final_route(root, child["metadata"], child["id"]) == (UNKNOWN, "the run that handed it off has not ended yet")
    _running(root, first)
    assert _state(root, child) == UNKNOWN
    frame = _ended(root, first, first_registry, answer="The report is started.")
    assert (frame["outcome"], frame["work_ref"]) == ("deferred", child["id"])  # its view names the child
    assert final_route(root, child["metadata"], child["id"]) == (
        UNKNOWN, "another root's result names it, and not every connector follows such work")
    assert UNSETTLED in _audience(_rendered(first_request, child))
    assert _state(root, later) == ROUTELESS


def test_a_promoted_root_that_ends_without_naming_its_child_leaves_it_routeless(fired_context, monkeypatch):
    from ouroboros.task_results import write_task_result

    root, _pending, _first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    _ended(root, turn, registry)
    child = _promote(root, monkeypatch, first, _registry_for(root, first))
    write_task_result(root, first["id"], "completed", metadata=first["metadata"], result="Done without a handoff.")
    assert final_route(root, child["metadata"], child["id"]) == (ROUTELESS, "the run that handed it off ended without passing it to a connector")


def test_a_schedule_born_root_and_a_root_it_starts_have_no_route_and_a_schedule_cannot_copy_one(fired_context, monkeypatch):
    root, pending, first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    _ended(root, turn, registry)
    assert _state(root, first) == ROUTED
    # The root holding a routed handoff schedules a promise: the real producer and the real scheduler.
    fired = _fired(root, pending, _registry_for(root, first))
    assert fired["metadata"]["presence"]["event"] == first["metadata"]["presence"]["event"]
    assert KEY not in fired["metadata"] and KEY not in fired["metadata"]["presence"]
    assert final_route(root, fired["metadata"], fired["id"]) == (ROUTELESS, "it started from a schedule")
    told = _rendered(first_request, fired)
    assert "nothing it finishes with is sent to the conversation by itself" in told and HEADING not in told
    # Even a copied fact cannot route: the schedule outranks it, and a fact names one work only.
    assert _state(root, {**fired, "metadata": {**fired["metadata"], KEY: dict(first["metadata"][KEY])}}) == ROUTELESS
    assert final_route(root, first["metadata"], "another-root") == (UNKNOWN, "its handoff record names other work")

    _running(root, fired)
    monkeypatch.setattr("ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    child = _promote(root, monkeypatch, fired, _registry_for(root, fired))
    assert child["metadata"][KEY]["held"] is True  # its source's first handoff, yet no connector reads that source
    assert final_route(root, child["metadata"], child["id"]) == (ROUTELESS, "no connector reads the run that handed it off")
    told = _audience(_rendered(first_request, child))
    assert NO_ROUTE in told and SPEECH not in told


def test_a_root_admitted_before_this_fact_is_unknown_never_routeless_and_its_children_inherit_no_answer(fired_context, monkeypatch):
    root, _pending, first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    _ended(root, turn, registry)
    legacy = {**first, "metadata": {key: value for key, value in first["metadata"].items() if key != KEY}}
    assert final_route(root, legacy["metadata"], legacy["id"]) == (UNKNOWN, "it was admitted before the host recorded handoffs")
    told = _audience(_rendered(first_request, legacy))
    assert UNSETTLED in told and "If one does, a non-silent finish_task answer is new speech there" in told
    child = _promote(root, monkeypatch, legacy, _registry_for(root, legacy))
    assert child["metadata"][KEY]["held"] is True and _state(root, child) == UNKNOWN


def test_an_unreadable_source_record_is_unknown(fired_context, monkeypatch):
    import ouroboros.task_results as task_results

    root, _pending, _first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    _ended(root, turn, registry)
    real = task_results.load_task_result

    def unreadable(drive_root, task_id, **kwargs):
        if task_id == turn["id"]:
            raise OSError("the store is busy")
        return real(drive_root, task_id, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", unreadable)
    assert final_route(root, first["metadata"], first["id"]) == (
        UNKNOWN, "the record of the run that handed it off is unreadable now")


def _forced(root, task):
    from ouroboros.loop_forced_finalization import _presence_forced_contract

    tools = types.SimpleNamespace(task_contract={"capability_ceiling": {}}, drive_root=root, task_id=task["id"],
                                  task_metadata=task["metadata"])
    return _presence_forced_contract(types.SimpleNamespace(task_id=task["id"], messages=[], tools=None), tools)


def test_the_forced_final_contract_reads_the_same_reader_and_adds_no_round(fired_context, monkeypatch):
    root, _pending, _first_request = fired_context
    monkeypatch.setattr("ouroboros.presence_context.presence_send_facts", lambda *_a: "none")
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    second = _promote(root, monkeypatch, turn, registry)
    unknown = _forced(root, first)
    assert "whether a connector reads its final there is not settled (the run that handed it off has no record yet)" in unknown
    _ended(root, turn, registry)
    routed, routeless = _forced(root, first), _forced(root, second)
    assert "This task answers a Presence conversation" in routed and "not settled" not in routed
    assert "no connector reads its final there (it is not the first work" in routeless
    assert "receive only what you declare" not in routeless
    for contract in (unknown, routed, routeless):
        assert contract.count("[PRESENCE_DELIVERY]") == 1 and '"presence_finish"' in contract  # the one call, no new round
    assert "This task answers a Presence conversation" in _forced(root, turn)  # the inline turn itself


def test_a_scheduled_root_keeps_its_published_contract_in_the_forced_final_too(fired_context, monkeypatch):
    root, pending, _first_request = fired_context
    registry, _turn = _source_turn(root, THREAD)
    fired = _fired(root, pending, registry)
    monkeypatch.setattr("ouroboros.presence_context.presence_send_facts", lambda *_a: "none")
    assert "no connector reads its final there (it started from a schedule)" in _forced(root, fired)


# --- receipts: one seam, the actual audience and the same reader ------------------------------------------------


def _family(ctx, tmp_path):
    from ouroboros.tools.cards import _show_card
    from ouroboros.tools.control_runtime import _send_user_message
    from ouroboros.tools.core_artifacts import _send_file, _send_links, _send_photo, _send_video

    image, video, doc = tmp_path / "a.png", tmp_path / "a.mp4", tmp_path / "a.txt"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 4096)
    video.write_bytes(b"0" * 64)
    doc.write_text("a report", encoding="utf-8")
    return {
        "send_user_message": lambda: _send_user_message(ctx, "A word."),
        "send_photo": lambda: _send_photo(ctx, image_base64=base64.b64encode(image.read_bytes()).decode()),
        "send_video": lambda: _send_video(ctx, file_path=str(video)),
        "send_file": lambda: _send_file(ctx, file_path=str(doc)),
        "send_links": lambda: _send_links(ctx, links=[{"label": "Docs", "url": "https://example.test/x"}]),
        "show_card": lambda: _show_card(ctx, title="Status", fallback_text="All green",
                                        components=[{"type": "metric", "label": "Up", "value": "9"}]),
    }


def _words(name, receipt):
    """A card's receipt is its identity plus the family's words."""
    return json.loads(receipt)["delivery"] if name == "show_card" else receipt


def _turn_ctx(root, *, event_queue):
    registry, turn = _source_turn(root, THREAD)
    ctx = registry._ctx
    ctx.event_queue, ctx.pending_events = event_queue, []
    return ctx, turn


class _Refusing:
    def put_nowait(self, _event):
        raise OSError("the live queue refused")


def test_an_inline_place_turn_is_told_its_local_room_for_the_whole_family_with_its_route(fired_context, tmp_path):
    root, _pending, _first_request = fired_context
    live = queue_module.Queue()
    ctx, turn = _turn_ctx(root, event_queue=live)
    for name, send in _family(ctx, tmp_path).items():
        receipt = _words(name, send())
        assert "accepted for delivery to " + LOCAL_ROOM in receipt, name
        assert "not the place conversation" in receipt and SAYS_ROUTED in receipt, name
        assert "owner chat" not in receipt and "shown" not in receipt, name
    assert live.qsize() == 6 and not ctx.pending_events
    assert {frame["chat_id"] for frame in list(live.queue)} == {turn["chat_id"]}


def test_each_promoted_roots_receipt_reads_its_route_again_at_each_send(fired_context, monkeypatch, tmp_path):
    from ouroboros.tools.control_runtime import _send_user_message

    root, _pending, _first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    second = _promote(root, monkeypatch, turn, registry)
    sends = {}
    for task in (first, second):
        ctx = _registry_for(root, task)._ctx
        ctx.event_queue, ctx.pending_events = queue_module.Queue(), []
        sends[task["id"]] = ctx
    early = _send_user_message(sends[first["id"]], "Working on it.")
    assert LOCAL_ROOM in early and NOT_THERE in early and SAYS_UNKNOWN in early and SAYS_ROUTED not in early
    _ended(root, turn, registry)  # the turn publishes the first root as its handoff
    later = _send_user_message(sends[first["id"]], "Half done.")
    assert SAYS_ROUTED in later and SAYS_UNKNOWN not in later
    other = _send_user_message(sends[second["id"]], "Working on it.")
    assert LOCAL_ROOM in other and NOT_THERE in other and SAYS_ROUTELESS in other and "final answer" not in other


def test_a_project_bound_root_names_its_project_for_the_whole_family(fired_context, monkeypatch, tmp_path):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    root, _pending, _first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    first = _promote(root, monkeypatch, turn, registry)
    project = create_project(root, "launch", name="Launch")
    bind_task_to_project(root, first["id"], project["id"], project["chat_id"], origin={"absent": "system"})
    ctx = _registry_for(root, first)._ctx
    ctx.event_queue, ctx.pending_events = queue_module.Queue(), []
    for name, send in _family(ctx, tmp_path).items():
        receipt = _words(name, send())
        assert 'Project room "Launch"' in receipt and "not the place conversation" in receipt, name
        assert "owner chat" not in receipt and "Main" not in receipt and "local room" not in receipt, name


def test_ordinary_project_and_main_receipts_name_only_acceptance(tmp_path):
    from tests.test_main_notice import _ctx, _project

    ctx = _ctx(tmp_path, _project(tmp_path), event_queue=queue_module.Queue())
    family = _family(ctx, tmp_path)
    assert family["send_file"]() == "OK: file 'a.txt' accepted for delivery to Project room \"Launch\"."
    assert family["send_user_message"]() == 'OK: message accepted for delivery to Project room "Launch".'
    assert json.loads(family["show_card"]())["delivery"] == 'OK: card accepted for delivery to Project room "Launch".'
    plain = _ctx(tmp_path / "other", 1, event_queue=queue_module.Queue())
    family = _family(plain, tmp_path)
    assert family["send_user_message"]() == "OK: message accepted for delivery to owner chat."
    assert family["send_file"]() == "OK: file 'a.txt' accepted for delivery to owner chat."
    assert json.loads(family["show_card"]())["delivery"] == "OK: card accepted for delivery to owner chat."


@pytest.mark.parametrize("reader", ["_load_bindings", "project_thread_name"])
def test_unreadable_delivery_audience_never_guesses_owner_chat(tmp_path, monkeypatch, reader):
    from ouroboros import projects_registry
    from tests.test_main_notice import _ctx, _project

    chat = _project(tmp_path)
    ctx = _ctx(tmp_path, chat, event_queue=queue_module.Queue())
    family = _family(ctx, tmp_path)

    def unavailable(*_args, **_kwargs):
        raise OSError("registry temporarily unavailable")

    with monkeypatch.context() as failure:
        failure.setattr(projects_registry, reader, unavailable)
        for name, send in family.items():
            receipt = _words(name, send())
            assert "accepted for delivery" in receipt and "destination unavailable" in receipt, name
            assert "owner chat" not in receipt and "shown" not in receipt, name
    assert ctx.event_queue.qsize() == 6 and not ctx.pending_events
    assert {frame["chat_id"] for frame in list(ctx.event_queue.queue)} == {chat}
    assert 'Project room "Launch"' in family["send_user_message"]()


def test_hidden_delivery_is_not_described_as_an_owner_visible_chat(tmp_path):
    from tests.test_main_notice import _ctx

    ctx = _ctx(tmp_path, 0, event_queue=queue_module.Queue())
    for name, send in _family(ctx, tmp_path).items():
        receipt = _words(name, send())
        assert "accepted for delivery to the hidden task record" in receipt, name
        assert "not an owner-visible chat" in receipt and "owner chat" not in receipt, name
    assert ctx.event_queue.qsize() == 6
    assert {frame["chat_id"] for frame in list(ctx.event_queue.queue)} == {0}


@pytest.mark.parametrize("room_count", [10, 1000])
def test_receipts_and_bus_share_unchanged_room_labels_without_listing_projects(tmp_path, monkeypatch, room_count):
    from ouroboros import projects_registry as projects
    from ouroboros.tools.control_runtime import _send_user_message
    from tests.test_main_notice import _ctx, _project

    chat = _project(tmp_path)
    data = projects._load(tmp_path)
    data["projects"].extend({"id": f"other-{i}", "name": f"Other {i}", "chat_id": 2_000_000_000 + i,
                             "lifecycle": "tombstoned"} for i in range(room_count))
    projects._save(tmp_path, data)
    ctx = _ctx(tmp_path, chat, event_queue=queue_module.Queue())
    original, reads = projects.display_registry_snapshot, []

    def snapshot(root):
        reads.append(root)
        return original(root)

    def no_listing(*_args, **_kwargs):
        raise AssertionError("a receipt must not list/sort every Project")

    monkeypatch.setattr(projects, "display_registry_snapshot", snapshot)
    monkeypatch.setattr(projects, "list_reserved_projects", no_listing)
    for _ in range(20):
        assert 'Project room "Launch"' in _send_user_message(ctx, "Progress")
        frame = ctx.event_queue.get_nowait()
        projects.stamp_project_thread(tmp_path, frame)
        assert frame["project_thread"] is True
    assert reads == [tmp_path]
    projects.update_project(tmp_path, "launch", name="Renamed")
    assert 'Project room "Renamed"' in _send_user_message(ctx, "Progress")
    assert reads == [tmp_path, tmp_path]


def test_room_label_cache_keeps_source_failure_and_ambiguous_names_unknown(tmp_path):
    from ouroboros import projects_registry as projects
    from ouroboros.tools.control_runtime import _send_user_message
    from tests.test_main_notice import _ctx, _project

    chat = _project(tmp_path)
    ctx = _ctx(tmp_path, chat, event_queue=queue_module.Queue())
    assert 'Project room "Launch"' in _send_user_message(ctx, "Progress")
    path = projects._registry_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text('{"projects": [], "projects": []}', encoding="utf-8")
    assert "destination unavailable" in _send_user_message(ctx, "Progress")
    path.unlink()  # the committed witness remains: this is not a new empty install
    assert "destination unavailable" in _send_user_message(ctx, "Progress")
    projects._save(tmp_path, data)
    assert 'Project room "Launch"' in _send_user_message(ctx, "Progress")
    data["projects"].append({"id": "duplicate-chat", "chat_id": chat, "name": "Other"})
    projects._save(tmp_path, data)
    assert "destination unavailable" in _send_user_message(ctx, "Progress")
    assert chat in projects.project_thread_chat_ids(tmp_path)  # membership still known
    other = _ctx(tmp_path / "empty-install", 1, event_queue=queue_module.Queue())
    assert _send_user_message(other, "Progress") == "OK: message accepted for delivery to owner chat."


@pytest.mark.parametrize("event_queue", [None, _Refusing()], ids=["no_queue", "refused"])
def test_an_inline_turn_with_no_consumer_is_told_nothing_was_delivered(fired_context, tmp_path, event_queue):
    from ouroboros.tools.control_runtime import _send_user_message

    root, _pending, _first_request = fired_context
    ctx, _turn = _turn_ctx(root, event_queue=event_queue)
    family = _family(ctx, tmp_path)
    for name, send in family.items():
        receipt = _words(name, send())
        assert "LOCAL_ROOM_UNAVAILABLE" in receipt and "not delivered" in receipt and LOCAL_ROOM in receipt, name
    assert not ctx.pending_events and not getattr(ctx, "_owner_message_sent", False)
    assert _send_user_message(ctx, "Again.").startswith("⚠️")


def test_a_refused_live_send_leaves_at_the_turns_end_through_the_carriers_sink_once_and_in_order(fired_context, monkeypatch, tmp_path):
    from ouroboros.tools.control_runtime import _send_user_message

    root, _pending, _first_request = fired_context
    monkeypatch.setattr("ouroboros.utils.in_worker_process", lambda: True)
    ctx, turn = _turn_ctx(root, event_queue=_Refusing())
    for words in ("First.", "Second."):
        receipt = _send_user_message(ctx, words)
        assert "queued for delivery to " + LOCAL_ROOM in receipt
    final = {"type": "send_message", "chat_id": turn["chat_id"], "text": "the final", "task_id": turn["id"]}
    result = {"type": "presence_result", "task_id": turn["id"], "text": "the final"}
    events = [*ctx.pending_events, final, result]
    received: list = []
    flush_owner_events(root, events, sink=received.extend)
    assert [row["text"] for row in received] == ["First.", "Second."]  # never the final, never the result frame
    assert events == [final, result]
    flush_owner_events(root, events, sink=received.extend)
    assert len(received) == 2


def test_a_buffered_local_row_reaches_the_attached_bus_and_a_lost_bus_never_fails_the_turn(fired_context, monkeypatch):
    from ouroboros.tools.control_runtime import _send_user_message
    from supervisor import workers

    root, _pending, _first_request = fired_context
    bus = queue_module.Queue()
    monkeypatch.setattr(workers, "_EVENT_Q", bus)
    ctx, _turn = _turn_ctx(root, event_queue=None)
    assert "queued for delivery to " + LOCAL_ROOM in _send_user_message(ctx, "A word.")
    events = list(ctx.pending_events)
    flush_owner_events(root, events)
    frame = bus.get_nowait()
    assert frame["text"] == "A word." and frame["system_type"] == "proactive_message" and not events

    monkeypatch.setattr(workers, "_EVENT_Q", None)
    kept = [dict(frame)]
    flush_owner_events(root, kept)  # logged and left; the turn's answer is not held hostage to it
    assert kept == [frame]


def test_private_main_contact_is_not_a_local_row_and_keeps_its_custody_without_a_consumer(fired_context):
    from ouroboros.tools.owner_delivery import local_room_frame

    root, _pending, _first_request = fired_context
    ctx, turn = _turn_ctx(root, event_queue=None)
    frame = {"type": "send_message", "chat_id": 1, "text": "private", "task_id": turn["id"],
             "system_type": "main_notice", "progress_meta": {"presence_contact": {"task_id": turn["id"]}}}
    assert not local_room_frame(frame)
    assert deliver_owner_event(ctx, frame) == "deferred" and ctx.pending_events == [frame]
    assert UNDELIVERED != "deferred" and ctx._owner_message_sent is True


def test_presence_main_contact_stamps_before_snapshot_and_outbox_registration(tmp_path, monkeypatch):
    from ouroboros import presence_owner_contact

    captured = {}
    ctx = types.SimpleNamespace(task_id="wake-contact", task_metadata={"initiator": "consciousness"},
                                current_chat_id=1, drive_root=tmp_path, pending_events=[])

    def fake_patch(_root, _task_id, updater):
        captured["stored"] = copy.deepcopy(updater({"owner_contacts": {}}))

    monkeypatch.setattr("ouroboros.task_store.patch_document", fake_patch)
    monkeypatch.setattr("supervisor.terminal_delivery.register_pending_delivery",
                        lambda _root, event: captured.setdefault("registered", copy.deepcopy(event)) or True)
    monkeypatch.setattr("ouroboros.tools.owner_delivery.deliver_owner_event",
                        lambda _ctx, event: captured.setdefault("delivered", copy.deepcopy(event)) or "deferred")

    result = presence_owner_contact.send_main_message(ctx, "wake contact", {"task_id": "source"}, "req-1")
    assert "Main message" in result
    stored = next(iter(captured["stored"]["owner_contacts"].values()))["event"]
    assert stored["progress_meta"]["initiator"] == "consciousness"
    assert captured["registered"]["progress_meta"]["initiator"] == "consciousness"
    assert captured["delivered"]["progress_meta"]["initiator"] == "consciousness"


# --- the pooled frame: declared, content free ---------------------------------------------------------------------


def _ledger(root):
    ctx = types.SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, WORKERS={}, PENDING=[], rows=[])
    ctx.append_jsonl = lambda path, row: ctx.rows.append((pathlib.Path(path).name, dict(row)))
    return ctx


def test_every_frame_of_a_pooled_root_has_a_handler_and_its_final_reaches_no_log(host):
    from supervisor.event_taxonomy import TELEMETRY_ONLY, disposition_for
    from supervisor.events import EVENT_HANDLERS, dispatch_event

    world = host()
    task = promote(world, event=event_of(world, "the-promoter"))
    publish(world)
    from ouroboros.dialogue_provenance import presence_queue_task

    restored = presence_queue_task(world.data, task["id"])  # what a restart's queue restore reads
    assert restored["metadata"][KEY] == {"held": True, "work_ref": task["id"], "source_task_id": ""}
    assert _state(world.data, restored) == UNKNOWN  # this harness's producer context names no source run
    box = run_root(world, dispatched(world, task), finish(FINAL))
    box["thread"].join(120)
    assert not box["thread"].is_alive() and "error" not in box, box.get("error")
    frames = []
    while not box["out"].empty():
        frames.append(box["out"].get_nowait())
    assert not [frame.get("type") for frame in frames if frame.get("type") not in EVENT_HANDLERS]
    assert not [frame for frame in frames if frame.get("type") == "send_message" and FINAL in str(frame.get("text"))]

    frame = next(row for row in frames if row.get("type") == "presence_result")
    assert frame["text"] == FINAL  # the inline runner's copy stays whole; the absorption is the supervisor's
    assert disposition_for("presence_result").tier == TELEMETRY_ONLY
    ledger = _ledger(world.data)
    dispatch_event(dict(frame), ledger)
    [(name, row)] = ledger.rows
    assert name == "events.jsonl" and row["type"] == "presence_result" and row["task_id"] == task["id"]
    assert row["outcome"] == "message" and row["output_ref"] == frame["output_ref"] and row["text_chars"] == len(FINAL)
    assert FINAL not in json.dumps(row)

    # One final, read by its reference: a repeated poll returns the same output to deduplicate on.
    views = [world.client.get(f"/presence/work/{task['id']}", headers=HEADERS,
                              params={"binding_id": world.place.place_id}).json() for _ in range(2)]
    assert views[0]["text"] == FINAL and views[0]["output_ref"] == frame["output_ref"]
    assert views[0] == views[1]


def test_intentional_silence_stays_silent_and_a_tool_send_is_not_repeated_by_the_frame(host):
    world = host()
    task = promote(world, event=event_of(world, "the-quiet"))
    box = run_root(world, dispatched(world, task),
                   call(("send_user_message", {"text": "Local note."})), SILENT)
    box["thread"].join(120)
    assert not box["thread"].is_alive() and "error" not in box, box.get("error")
    frames = []
    while not box["out"].empty():
        frames.append(box["out"].get_nowait())
    notes = [frame for frame in frames if frame.get("type") == "send_message" and frame.get("text") == "Local note."]
    assert len(notes) == 1 and notes[0]["system_type"] == "proactive_message"
    frame = next(row for row in frames if row.get("type") == "presence_result")
    assert frame["outcome"] == "silent" and frame["text"] == ""


def test_a_silent_place_holds_a_promoted_roots_final(host):
    from ouroboros import places

    world = host()
    task = promote(world, event=event_of(world, "the-silenced"))
    places.edit_place(world.data, world.place.place_id, "set_state", author="owner", state="silent")
    box = run_root(world, dispatched(world, task), finish(FINAL))
    box["thread"].join(120)
    assert not box["thread"].is_alive() and "error" not in box, box.get("error")
    view = world.client.get(f"/presence/work/{task['id']}", headers=HEADERS,
                            params={"binding_id": world.place.place_id}).json()
    assert view["status"] == "completed" and view["outcome"] == "silent" and view["text"] == ""


@pytest.mark.parametrize("frame,error", [
    ({"type": "no_such_event", "task_id": "t1", "text": "a private final", "nested": {"message": "private too"}}, ""),
    ({"task_id": "t1", "text": "a private final"}, "missing event.type"),
    (["a private final"], "event is not dict"),
])
def test_a_frame_nobody_answers_leaves_its_shape_never_its_body(tmp_path, frame, error):
    from supervisor.events import dispatch_event

    ledger = _ledger(tmp_path)
    dispatch_event(frame, ledger)
    [(name, row)] = ledger.rows
    assert name == "supervisor.jsonl" and "event_repr" not in row and "private" not in json.dumps(row)
    assert row["type"] == ("unknown_worker_event" if not error else "invalid_worker_event")
    assert row.get("error", "") == error
    if isinstance(frame, dict):
        assert row["task_id"] == "t1" and "text" in row["event_keys"]


# --- Continue of a handed-off root (A01): the successor's route is what it actually has ---------------------------


def _continued(world):
    """A handed-off root ended by a provider failure, then the real Continue admission."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.worker_pool_support import affirm_ready

    task = promote(world, event=event_of(world, "the-handed-off"))
    dispatched(world, task)
    world.queue.RUNNING.pop(task["id"], None)
    write_task_result(world.data, task["id"], "failed", reason_code="provider_unavailable",
                      # ``promote``'s bare producer context has no task id for the author stamp a real turn writes.
                      metadata={**task["metadata"], "objective_author": {"kind": "task", "task_id": "the-turn"}},
                      description="Check the three services",
                      result="The model provider was unavailable.")
    offer = continuation_offer(load_task_result(world.data, task["id"]), task["id"])
    assert offer["eligible"] is True, offer
    affirm_ready(world.workers.WORKERS[1])
    ack = admit_continuation(task["id"], action_nonce="nonce-1234567890")
    assert ack["ok"] is True, ack
    return task, next(row for row in world.queue.PENDING if row["id"] == ack["successor_task_id"])


def _work(world, work_ref):
    return world.client.get(f"/presence/work/{work_ref}", headers=HEADERS, params={"binding_id": world.place.place_id})


def test_a_continued_handoff_has_no_route_and_its_private_answer_is_not_served_by_the_old_reference(host):
    """A01's residual, pinned: a successor has no speaker carrier, so nothing it finishes with is speech today."""
    from ouroboros.place_reply import reply_carrier
    from ouroboros.task_results import write_task_result

    world = host()
    task, successor = _continued(world)
    metadata = successor["metadata"]
    assert KEY not in metadata and "presence" not in metadata and reply_carrier(metadata) == {}
    assert metadata["presence_binding_authority"] == {"binding_id": task["metadata"]["presence"]["binding_id"]}
    assert final_route(world.data, metadata, successor["id"]) == (ROUTELESS, "it carries no place conversation")
    assert presence_root_carrier(metadata) == {"presence_binding_authority": metadata["presence_binding_authority"]}

    write_task_result(world.data, successor["id"], "completed", metadata=metadata, result="An owner-side private report.")
    view = _work(world, task["id"])
    assert view.status_code == 200 and view.json()["status"] == "failed"
    assert "private report" not in json.dumps(view.json())
    assert _work(world, successor["id"]).status_code == 404  # no connector's work: it answers its owner's room


@pytest.mark.parametrize("failed_row", [None, "First.", "Second."], ids=["healthy", "first_refused", "second_refused"])
def test_saved_turn_flush_preserves_order_through_the_worker_drain(root, monkeypatch, failed_row):
    """A scripted author, real saved-turn carrier/finalization and worker drain; only the bus fails once."""
    from ouroboros import presence_carrier
    from ouroboros.presence_runner import _read_previous_turn, build_presence_result_event
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.control_runtime import _send_user_message
    from supervisor.worker_process import run_worker_task
    from tests.test_presence_moved_lifecycle import _seed_and_built
    from tests.test_presence_project_activity import PROJECT, _move
    from tests.test_presence_runner import _event

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    write_task_result(root, tid, "scheduled", metadata=built["metadata"], chat_id=built["chat_id"],
                      _is_direct_chat=True, source="presence", task_contract=built["task_contract"], result="Task paused.")
    _move(root, tid, PROJECT)
    monkeypatch.setattr("ouroboros.utils.in_worker_process", lambda: True)
    seen, frames, attempts = {}, [], []

    class Author:
        def handle_task(self, task):
            ctx = _registry_for(root, task)._ctx
            ctx.model_wait_context = types.SimpleNamespace(control_reason=lambda: None)
            state = presence_carrier.reenter(ctx)
            assert state and state.lease.held()
            ctx.event_queue = _Refusing()  # producer buffers in send order for the carrier's end
            for words in ("First.", "Second.", "Third."):
                assert "queued for delivery" in _send_user_message(ctx, words)
            frame = build_presence_result_event(task, FINAL, ctx, terminal_origin="model_final")
            write_task_result(root, tid, "completed", metadata=task["metadata"], result=FINAL,
                              terminal_origin="model_final")
            seen["events"] = [*ctx.pending_events, frame, {"type": "task_done", "task_id": tid, "status": "completed"}]
            return seen["events"]

    def put(frame):
        if frame.get("system_type") == "proactive_message":
            words = frame["text"]
            attempts.append(words)
            if words == failed_row and attempts.count(words) == 1:
                raise OSError("scripted single enqueue refusal")
        frames.append(dict(frame))

    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_kw: Author())
    resumed = {**built, "_attempt": 1, "_budget_pause_resume": {"grant_id": "g", "pause_id": "p"},
               "_execution_owner": {"kind": "pooled", "lane": "project"}}
    run_worker_task(types.SimpleNamespace(_event_queue=None, owner_wait_callback=None), resumed,
                    wid=91, out_q=types.SimpleNamespace(put=put), repo_dir=str(root), drive_root=str(root))
    words = [frame["text"] for frame in frames if frame.get("system_type") == "proactive_message"]
    previous = _read_previous_turn(root, _event().conversation_key)
    print(json.dumps({"failed_row": failed_row, "attempts": attempts, "delivered": words,
                      "final": previous.get("message"), "returned": [row["type"] for row in seen["events"]]}))
    assert load_task_result(root, tid)["status"] == "completed" and previous["message"] == FINAL
    assert frames[-1]["type"] == "task_done" and not any(row["type"] == "presence_result" for row in frames)
    assert words == ["First.", "Second.", "Third."]
    assert [row.get("text") for row in seen["events"] if row.get("system_type") == "proactive_message"] == (
        [] if failed_row is None else ["First.", "Second.", "Third."][["First.", "Second."].index(failed_row):])


@pytest.mark.parametrize("name", ["send_user_message", "send_photo", "show_card"])
def test_main_receipts_claim_only_queue_acceptance(tmp_path, name):
    from tests.test_main_notice import _ctx

    bus = queue_module.Queue()
    ctx = _ctx(tmp_path, 1, event_queue=bus)
    receipt = _family(ctx, tmp_path)[name]()
    print(json.dumps({"tool": name, "receipt": receipt, "undrained_frames": bus.qsize()}))
    assert bus.qsize() == 1 and not ctx.pending_events
    if name == "show_card":
        card = json.loads(receipt)
        assert card["card_id"] == bus.queue[0]["card_id"] and card["revision"] == 1
        assert "shown" not in card
    words = _words(name, receipt)
    assert "accepted for delivery to" in words and "sent" not in words


def test_main_notice_receipt_claims_only_queue_acceptance(tmp_path):
    from ouroboros.tools.control_runtime import _send_user_message
    from tests.test_main_notice import _ctx, _project

    bus = queue_module.Queue()
    ctx = _ctx(tmp_path, _project(tmp_path), event_queue=bus)
    receipt = _send_user_message(ctx, "A notice.", destination="main")
    print(json.dumps({"receipt": receipt, "undrained_frames": bus.qsize()}))
    assert bus.qsize() == 1 and bus.queue[0]["chat_id"] == 1
    assert "accepted for delivery to the main chat" in receipt and "sent" not in receipt


def test_deep_schedule_handoffs_remain_routeless_in_all_consumers(fired_context, monkeypatch):
    root, pending, first_request = fired_context
    registry, _turn = _source_turn(root, THREAD)
    source = _fired(root, pending, registry)
    monkeypatch.setattr("ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    observed = []
    for depth in range(1, 11):
        registry = _registry_for(root, source)
        child = _promote(root, monkeypatch, source, registry)
        _ended(root, source, registry)
        source = child
        if depth in (9, 10):
            route = final_route(root, child["metadata"], child["id"])
            observed.append({"depth": depth, "route": route})
            print(json.dumps(observed[-1]))
            assert route == (ROUTELESS, "no connector reads the run that handed it off")
            assert NO_ROUTE in _audience(_rendered(first_request, child))
            assert "no connector reads its final there" in _forced(root, child)
            from ouroboros.tools.control_runtime import _send_user_message
            ctx = _registry_for(root, child)._ctx
            ctx.event_queue = queue_module.Queue()
            assert SAYS_ROUTELESS in _send_user_message(ctx, "A local note.")


@pytest.mark.parametrize("case", ["cycle", "unpublished_terminal", "deep_schedule"])
def test_ancestry_walk_stops_on_evidence_or_a_cycle_without_a_depth_cap(tmp_path, monkeypatch, case):
    """A scripted result store isolates traversal; the ten real admissions above cover its durable producer."""
    from ouroboros.presence_context import handoff_fact

    def metadata(task, source):
        return {"presence": {"place_id": "place"}, KEY: handoff_fact(source, task, held=True)}

    count = 1100 if case == "deep_schedule" else 3
    rows = {str(n): {"status": "completed", "metadata": {
        **metadata(str(n), str(n + 1)), "presence_work_ref": str(n - 1)}} for n in range(1, count + 1)}
    if case == "cycle":
        rows[str(count)]["metadata"][KEY] = handoff_fact("1", str(count), held=True)
    elif case == "unpublished_terminal":
        rows["1"]["metadata"].pop("presence_work_ref")
    else:
        rows[str(count)]["metadata"]["schedule_occurrence"] = {"id": "scheduled"}
    reads = []

    def load(_root, task_id, *, strict):
        assert strict
        reads.append(task_id)
        return rows.get(task_id)

    monkeypatch.setattr("ouroboros.task_results.load_task_result", load)
    state, why = final_route(tmp_path, metadata("0", "1"), "0")
    if case == "cycle":
        assert (state, why) == (UNKNOWN, "its handoff records form a cycle")
    elif case == "unpublished_terminal":
        assert (state, why) == (ROUTELESS, "the run that handed it off ended without passing it to a connector")
    else:
        assert (state, why) == (ROUTELESS, "no connector reads the run that handed it off")
    assert reads == [str(n) for n in range(1, (1 if case == "unpublished_terminal" else count) + 1)]
