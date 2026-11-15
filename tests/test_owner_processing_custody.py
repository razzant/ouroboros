"""The owner's tier act on a live work: a durable decision beside the primary's, applied from the next call.

``POST /api/tasks/{id}/model`` with ``action: processing`` writes ``metadata.processing`` and its own
``owner_processing`` record only; the worker mirrors the stored carrier into its metadata, tells the mind once
and moves nothing of the route. Default is a decision too: stale copies, checkpoints and a Continue keep it.
The card says the chosen tier beside the newest receipt of the work's own calls.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import loop, owner_model, owner_wait, route_choice
from ouroboros.model_slots import MODEL_PROCESSING_PREFERENCES_KEY as ROLES
from ouroboros.model_slots import PROCESSING_PREFERENCE_KEY as GLOBAL
from ouroboros.model_slots import task_processing_preference
from ouroboros.task_results import load_task_result, write_task_result
from tests import test_owner_model_custody as _custody
from tests._task_settings_helpers import isolated_runtime, save_owner  # noqa: F401 -- the shared isolated root
from tests.test_owner_model_custody import A, B, C, act, drain_and_apply, queued, running, turn

host = _custody.host  # the live queue, its worker maps and the model-act endpoint
FAST = {"source": "owner", "tier": "fast"}
DEFAULT = {"source": "default"}
ROOM_ECONOMY = {"source": "room", "tier": "economy"}
UNCHANGED = " from its next call; the route and model are unchanged."
FAST_WORDS = "[ROUTE FACTS] My human set this work's processing tier to Fast" + UNCHANGED
FLEX_WORDS = "[ROUTE FACTS] My human set this work's processing tier to Flex (economy)" + UNCHANGED
STANDARD_WORDS = "[ROUTE FACTS] My human set this work's processing tier to Standard" + UNCHANGED
DEFAULT_WORDS = "[ROUTE FACTS] My human returned this work's processing tier to the default" + UNCHANGED


def tier(host, processing, request_id="tier", task_id="work"):
    return act(host, "processing", request_id=request_id, task_id=task_id, processing=processing)


def live(host, metadata):
    """A running root admitted with these carriers."""
    task = queued(host, metadata=copy.deepcopy(metadata))
    host[2].PENDING.remove(task)
    host[2].RUNNING["work"] = {"task": task, "attempt": 1, "lead": {"model": A, "use_local": False, "basis": "main"}}
    write_task_result(host[0], "work", "running")
    return task


@pytest.fixture
def mail(monkeypatch):
    from ouroboros import owner_mailbox

    written, real = [], owner_mailbox.write_owner_message

    def record(*args, **kwargs):
        written.append((args, kwargs))
        return real(*args, **kwargs)
    monkeypatch.setattr(owner_mailbox, "write_owner_message", record)
    return written


# --- The act: validation, durability, receipts and wakes ----------------------------------------------------------


def test_a_queued_tier_act_is_durable_at_start_with_an_exact_retry_and_no_wake(host, mail, monkeypatch):
    from supervisor import queue as q

    reasons, persist = [], q.persist_queue_snapshot
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **kw: reasons.append(kw.get("reason")) or persist(**kw))
    task = queued(host, metadata={"lead": {"source": "main"}, "processing": ROOM_ECONOMY})
    response = tier(host, "fast", request_id="first")
    assert response.status_code == 200, response.text
    receipt = {"ok": True, "task_id": "work", "applied": "at_start", "processing": {"choice": "fast", "resolved": "fast", "source": "owner"}}
    assert response.json() == receipt
    stored = load_task_result(host[0], "work")
    assert stored["metadata"]["processing"] == FAST == task["metadata"]["processing"]
    assert stored["metadata"]["lead"] == {"source": "main"} == task["metadata"]["lead"]
    assert stored["owner_processing"] == {"request_id": "first", "receipts": {"first": receipt}} == task["owner_processing"]
    assert "owner_model" not in stored and "owner_model" not in task
    assert reasons == ["owner_processing"], "the queue snapshot mirrors the act"
    assert not mail, "a queued work reads its tier when it starts"
    saved = copy.deepcopy(task)
    assert tier(host, "economy", request_id="first").json() == receipt  # the same press answers its own receipt
    assert task == saved and load_task_result(host[0], "work")["metadata"]["processing"] == FAST
    # Default is a decision of its own: recorded, so nothing later re-reads the room's tier for this work.
    reset = tier(host, "", request_id="reset")
    assert reset.json() == {**receipt, "processing": {"choice": "", "resolved": ""}}
    assert load_task_result(host[0], "work")["metadata"]["processing"] == DEFAULT == task["metadata"]["processing"]
    assert tier(host, "fast", request_id="first").json() == receipt, "an older press never undoes a later one"
    assert load_task_result(host[0], "work")["metadata"]["processing"] == DEFAULT
    assert not (host[0] / "state/ui_preferences.json").exists(), "a tier act records no model recent"
    assert not mail


def test_a_running_tier_act_wakes_the_work_by_its_own_id(host, mail):
    from ouroboros.owner_mailbox import KIND_OWNER_MODEL

    running(host)
    response = tier(host, "economy", request_id="r-tier")
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "task_id": "work", "applied": "next_step",
                               "processing": {"choice": "economy", "resolved": "economy", "source": "owner"}}
    assert len(mail) == 1
    args, kwargs = mail[0]
    assert args[1:3] == ("owner_model", "work")
    assert kwargs == {"msg_id": "owner_processing:r-tier", "kind": KIND_OWNER_MODEL}
    assert tier(host, "fast", request_id="r-tier").json() == response.json()
    assert {call[1]["msg_id"] for call in mail} == {"owner_processing:r-tier"}
    assert load_task_result(host[0], "work")["metadata"]["processing"] == {"source": "owner", "tier": "economy"}


def test_a_work_takes_ultrafast_and_its_mind_hears_it(host, monkeypatch):
    running(host)
    _tools, ctx = turn(host, monkeypatch)
    response = tier(host, "ultrafast", request_id="uf")
    assert response.status_code == 200, response.text
    assert response.json()["processing"] == {"choice": "ultrafast", "resolved": "ultrafast", "source": "owner"}
    assert load_task_result(host[0], "work")["metadata"]["processing"] == {"source": "owner", "tier": "ultrafast"}
    assert owner_model.apply_current(ctx) is False
    assert ctx.task_metadata["processing"] == {"source": "owner", "tier": "ultrafast"}
    assert ctx._route_facts_pending == "[ROUTE FACTS] My human set this work's processing tier to Ultrafast" + UNCHANGED
    assert task_processing_preference({"task_metadata": ctx.task_metadata}) == "ultrafast"


def test_a_lost_tier_wake_reports_its_failure_after_the_decision_was_saved(host, monkeypatch):
    from ouroboros import owner_mailbox

    running(host)
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *a, **k: False)
    response = tier(host, "fast", request_id="lost")
    assert response.status_code == 503 and response.json()["reason_code"] == "model_wake_failed"
    assert load_task_result(host[0], "work")["metadata"]["processing"] == FAST


@pytest.mark.parametrize("body", [
    {"action": "processing"},  # a tier act names its tier
    {"action": "processing", "processing": "fast", "model": B},  # and never a model
    {"action": "processing", "processing": "fast", "use_local": False},
    {"action": "processing", "processing": "turbo"},
    {"action": "processing", "processing": "auto"},
    {"action": "processing", "processing": 1},
    {"action": "processing", "processing": None},
    {"action": "processing", "processing": ["fast"]},
    {"action": "set", "model": B, "processing": "fast"},  # only a tier act names a tier
    {"action": "follow", "processing": ""},
    {"action": "return", "processing": "fast"},
    {"action": "processing", "processing": "fast", "text": "hi"},
])
def test_a_malformed_tier_act_writes_nothing(host, mail, body):
    task = queued(host)
    before = copy.deepcopy(load_task_result(host[0], "work"))
    response = host[3].post("/api/tasks/work/model", json={"request_id": "bad", **body})
    assert response.status_code == 400, response.text
    assert response.json()["reason_code"] == "task_model_invalid"
    assert load_task_result(host[0], "work") == before
    assert "processing" not in task["metadata"] and "owner_processing" not in task and not mail


@pytest.mark.parametrize("fields,code", [
    ({"parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent"}, "not_a_root_task"),
    ({"_presence_turn": True}, "presence_model_unsupported"),
])
def test_a_tier_act_is_refused_where_a_model_act_is(host, fields, code):
    queued(host, **fields)
    response = tier(host, "fast")
    assert response.status_code == 409 and response.json()["reason_code"] == code
    assert "processing" not in load_task_result(host[0], "work")["metadata"]


def test_unknown_and_finished_work_take_no_tier(host):
    assert tier(host, "fast").json()["reason_code"] == "task_not_live"
    queued(host)
    write_task_result(host[0], "work", "completed")
    response = tier(host, "fast")
    assert response.status_code == 404 and response.json()["reason_code"] == "task_not_live"
    assert "processing" not in load_task_result(host[0], "work")["metadata"]


def test_a_tier_act_and_a_model_act_never_touch_each_others_decision(host):
    task = queued(host)
    assert act(host, model=B, request_id="model").status_code == 200
    stored = load_task_result(host[0], "work")
    control = copy.deepcopy(stored["owner_model"])
    recents = (host[0] / "state/ui_preferences.json").read_bytes()
    assert tier(host, "fast", request_id="tier").status_code == 200
    stored = load_task_result(host[0], "work")
    assert stored["owner_model"] == control and task["owner_model"] == control
    assert stored["metadata"]["lead"] == {"source": "owner", "model": B, "use_local": False}
    assert (host[0] / "state/ui_preferences.json").read_bytes() == recents
    # A later model act and a follow keep the tier and its record.
    assert act(host, model=C, request_id="model-2").status_code == 200
    assert act(host, "follow", request_id="follow").status_code == 200
    stored = load_task_result(host[0], "work")
    assert stored["metadata"]["processing"] == FAST == task["metadata"]["processing"]
    assert stored["owner_processing"]["request_id"] == "tier" == task["owner_processing"]["request_id"]


# --- Consumers: the worker applies both axes, each on its own ------------------------------------------------------


def test_a_model_act_then_a_tier_act_before_the_next_round_apply_both_axes(host, monkeypatch):
    running(host)
    _tools, ctx = turn(host, monkeypatch)
    assert act(host, model=B, request_id="r1").status_code == 200
    assert tier(host, "fast", request_id="r2").status_code == 200
    assert load_task_result(host[0], "work")["owner_model"]["request_id"] == "r1"
    assert owner_model.apply_current(ctx) is True
    assert ctx.task_model_override == B and ctx.active_model_override == B
    assert ctx._owner_model_request_id == "r1", "the model handover waits for the round boundary"
    assert ctx.task_metadata["processing"] == FAST
    assert ctx.task_metadata["lead"] == {"source": "owner", "model": B, "use_local": False}
    assert load_task_result(host[0], "work")["owner_model"]["request_id"] == "r1"
    assert task_processing_preference({"task_metadata": ctx.task_metadata}) == "fast"
    assert task_processing_preference({"task_metadata": ctx.task_metadata}, model_role="lead") == "fast"
    assert f"[ROUTE FACTS] My human set this work's primary: {B}." in ctx._route_facts_pending
    assert FAST_WORDS in ctx._route_facts_pending


@pytest.mark.parametrize("tier_first", [False, True])
def test_both_wakes_drained_in_one_round_tell_the_mind_of_both_acts(host, monkeypatch, tier_first):
    running(host)
    tools, ctx = turn(host, monkeypatch)
    acts = [lambda: act(host, model=B, request_id="r-model"), lambda: tier(host, "fast", request_id="r-tier")]
    for press in (reversed(acts) if tier_first else acts):
        assert press().status_code == 200
    controls = drain_and_apply(host, tools, ctx)
    assert controls.get("owner_model")
    assert ctx.active_model == B and ctx._owner_model_applied_id == "r-model" and ctx._owner_model_request_id is None
    assert ctx.task_metadata["processing"] == FAST and ctx._owner_processing_applied_id == "r-tier"
    assert f"[ROUTE FACTS] My human set this work's primary: {B}." in ctx._route_facts_pending
    assert FAST_WORDS in ctx._route_facts_pending, ctx._route_facts_pending


ROUTE_KEYS = ("route_choice", "active_model", "active_use_local", "active_model_override", "active_use_local_override",
              "active_role_override", "active_effort_override", "route_wait_on_primary", "primary_route",
              "task_model_override", "task_use_local_override", "task_route_source", "_owner_model_request_id",
              "_owner_model_applied_id", "_owner_model_primary_changed", "_lead_return_from", "_lead_episode")


def _route_state(ctx) -> dict:
    return {key: copy.deepcopy(getattr(ctx, key, None)) for key in ROUTE_KEYS}


def _hold_a_route(ctx):
    """A route the owner and a wait card already shaped: a tier act must leave every piece of it."""
    ctx.model_wait_context = SimpleNamespace(overrides={"main": {"model": C, "use_local": False,
                                                                 "model_account_override": "card-account"}})
    ctx.active_model_override, ctx.active_use_local_override, ctx.active_role_override = C, False, "main"
    ctx.active_effort_override = "high"
    ctx.route_wait_on_primary = True
    ctx.route_choice = route_choice.explicit("wait_card")


def test_a_tier_act_alone_moves_nothing_of_the_route(host, monkeypatch):
    running(host)
    tools, ctx = turn(host, monkeypatch)
    monkeypatch.setattr(owner_model, "primary_action", lambda *_a, **_k: pytest.fail("a tier act never acts on the primary"))
    _hold_a_route(ctx)
    turn_state, before = ctx.model_turn_state, _route_state(ctx)
    overrides = copy.deepcopy(ctx.model_wait_context.overrides)
    assert tier(host, "economy", request_id="tier").status_code == 200
    assert owner_model.apply_current(ctx) is False
    assert _route_state(ctx) == before
    assert ctx.model_wait_context.overrides == overrides and ctx.model_turn_state is turn_state
    assert ctx.task_metadata["processing"] == {"source": "owner", "tier": "economy"}
    assert ctx._route_facts_pending == FLEX_WORDS
    assert owner_model.apply_current(ctx) is False
    assert ctx._route_facts_pending == FLEX_WORDS, "the same act is told once"
    # Through the mail: the wake is consumed and no round handover is asked for.
    assert not drain_and_apply(host, tools, ctx).get("owner_model")
    assert _route_state(ctx) == before and ctx.model_turn_state is turn_state
    assert ctx._route_facts_pending == FLEX_WORDS
    # The Default reset is told as such, once.
    assert tier(host, "", request_id="reset").status_code == 200
    assert owner_model.apply_current(ctx) is False
    assert ctx.task_metadata["processing"] == DEFAULT
    assert ctx._route_facts_pending == f"{FLEX_WORDS} {DEFAULT_WORDS}"
    assert _route_state(ctx) == before


def test_a_tier_act_ends_no_owner_wait_or_sleep_and_reaches_the_next_call_at_the_drain(host, monkeypatch):
    """A work parked on the owner's question (warm, inline) or sleeping (warm, or cold through the supervisor's own
    ``wake_reason``) is not woken by a tier act: its mail stays unread until a real wake, and the round's drain then
    applies the tier to the next call. A model act, the owner's words and an answer still wake."""
    import pathlib

    from ouroboros import model_sleep
    from ouroboros.owner_mailbox import OwnerMailboxPeek, write_owner_message

    running(host)
    tools, ctx = turn(host, monkeypatch)
    ctx.drive_root, ctx._loop_mailbox_seen_ids = host[0], set()
    root = pathlib.Path(host[0])

    def woken():
        seen = set(ctx._loop_mailbox_seen_ids)
        return (OwnerMailboxPeek().pending(root, "work", seen, 1),  # the parked owner wait's poll
                OwnerMailboxPeek(waking_only=True).pending(root, "work", seen, 1),  # inline wait, waits, transport
                owner_wait.classify_wake(owner_wait._wait_entries(ctx), "q-1"),
                model_sleep.wake_reason(ctx, {"any_mail": True}))  # a warm sleep, and the cold sleep's supervisor pass
    assert woken() == (False, False, "unknown", "")
    assert tier(host, "economy", request_id="tier").status_code == 200
    assert woken() == (False, False, "unknown", ""), "a tier act alone ends no wait and no sleep"
    assert ctx.task_metadata.get("processing") is None, "nothing applied it yet"
    # Controls: a model act still wakes both, as does the owner's own answer.
    assert act(host, "set", request_id="model", model=B).status_code == 200
    assert woken() == (True, True, "model", "control:owner_model")
    # The round's drain after the wake applies both acts; the next call asks the owner's tier.
    drain_and_apply(host, tools, ctx, seen=ctx._loop_mailbox_seen_ids)
    assert ctx.task_metadata["processing"] == {"source": "owner", "tier": "economy"}
    assert task_processing_preference({"task_metadata": ctx.task_metadata}) == "economy"
    assert FLEX_WORDS in ctx._route_facts_pending
    assert woken() == (False, False, "unknown", "")
    assert tier(host, "fast", request_id="tier-2").status_code == 200
    assert write_owner_message(root, "my answer", "work", msg_id="quiz_answer:t:q-1", kind="quiz_answer")
    assert woken() == (True, True, "answer", "owner_text"), "the answer wakes even beside a newer tier act"


def test_a_running_loop_reads_a_tier_act_only_through_its_mail(host, monkeypatch):
    """Why the producer keeps the tier act's mail: a running loop rereads the durable tier only when a round drains an
    owner_model entry (a fresh start and a cold restore read it too). Without the mail the next call keeps the old tier."""
    from ouroboros import owner_mailbox

    running(host)
    tools, ctx = turn(host, monkeypatch)
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *_a, **_k: True)  # the act saved, no mail written
    assert tier(host, "economy", request_id="silent").status_code == 200
    assert load_task_result(host[0], "work")["metadata"]["processing"] == {"source": "owner", "tier": "economy"}
    drain_and_apply(host, tools, ctx)
    assert ctx.task_metadata.get("processing") is None, "an ordinary round boundary does not reread the stored tier"
    assert task_processing_preference({"task_metadata": ctx.task_metadata}) != "economy"


@pytest.mark.parametrize("row_tier", ["fast", "standard"])
def test_a_card_tier_on_a_fallback_row_is_what_its_next_call_and_its_new_helpers_ask(host, monkeypatch, row_tier):
    """The card says Flex from the next call: on a fallback row whose authored tier is another, that call asks Flex
    (the row's tier is a Settings default), and so does a new helper; Default gives the row its own tier back."""
    from ouroboros.supervisor_route import work_tier

    running(host)
    _tools, ctx = turn(host, monkeypatch)
    save_owner({"OUROBOROS_MODEL": A, ROLES: json.dumps({"fallback": [row_tier]})})
    ctx.route_choice, ctx.active_model, ctx.active_model_role = route_choice.explicit("fallback"), B, "fallback:0"
    next_call = lambda: task_processing_preference({"task_metadata": ctx.task_metadata}, model_role="fallback:0")  # noqa: E731
    assert next_call() == row_tier, "no tier of its own: the row's"
    response = tier(host, "economy", request_id="flex")
    assert response.status_code == 200, response.text
    assert response.json()["applied"] == "next_step"
    assert response.json()["processing"] == {"choice": "economy", "resolved": "economy", "source": "owner"}
    assert owner_model.apply_current(ctx) is False
    assert ctx.route_choice == route_choice.explicit("fallback"), "a tier act moves nothing of the route"
    assert ctx._route_facts_pending == FLEX_WORDS
    assert next_call() == "economy" == work_tier(ctx, {})
    assert tier(host, "", request_id="reset").status_code == 200
    assert owner_model.apply_current(ctx) is False
    assert ctx.task_metadata["processing"] == DEFAULT and next_call() == row_tier


def test_a_tier_act_after_an_applied_model_act_moves_nothing_of_the_route(host, monkeypatch):
    running(host)
    tools, ctx = turn(host, monkeypatch)
    assert act(host, model=B, request_id="r1").status_code == 200
    drain_and_apply(host, tools, ctx)
    assert ctx._owner_model_applied_id == "r1" and ctx.active_model == B
    monkeypatch.setattr(owner_model, "primary_action", lambda *_a, **_k: pytest.fail("a tier act never acts on the primary"))
    ctx._route_facts_pending = ""  # the round after the handover told the mind
    _hold_a_route(ctx)
    turn_state, before = ctx.model_turn_state, _route_state(ctx)
    overrides = copy.deepcopy(ctx.model_wait_context.overrides)
    assert tier(host, "standard", request_id="r2").status_code == 200
    assert not drain_and_apply(host, tools, ctx).get("owner_model")
    assert _route_state(ctx) == before
    assert ctx.model_wait_context.overrides == overrides and ctx.model_turn_state is turn_state
    assert ctx.task_metadata["processing"] == {"source": "owner", "tier": "standard"}
    assert ctx._route_facts_pending == STANDARD_WORDS
    assert load_task_result(host[0], "work")["owner_model"]["request_id"] == "r1"


@pytest.mark.parametrize("stored", [None, FAST])
def test_applying_the_stored_tier_keeps_sharing_the_works_metadata(host, monkeypatch, stored):
    """Whoever else holds the work's metadata (a deadline written during a round) keeps seeing what the loop reads."""
    running(host)
    if stored:
        write_task_result(host[0], "work", "running", metadata={"lead": {"source": "main"}, "processing": stored})
    _tools, ctx = turn(host, monkeypatch, {"lead": {"source": "main"}, **({"processing": stored} if stored else {})})
    held = ctx.task_metadata
    for _ in range(2):  # every fresh loop start and wake applies the stored decision again
        owner_model.apply_current(ctx)
        assert ctx.task_metadata is held
    held["deadline_at"] = "2026-11-14T12:00:00+00:00"
    assert ctx.task_metadata["deadline_at"] == "2026-11-14T12:00:00+00:00"
    assert tier(host, "economy", request_id="flex").status_code == 200  # a new tier reaches the same object
    owner_model.apply_current(ctx)
    assert ctx.task_metadata is held and held["processing"] == {"source": "owner", "tier": "economy"}
    assert held["deadline_at"] == "2026-11-14T12:00:00+00:00"


def test_a_row_without_a_tier_clears_a_stale_one_from_the_work(host, monkeypatch):
    running(host)  # admitted before tiers: the stored row has no carrier
    _tools, ctx = turn(host, monkeypatch, {"lead": {"source": "main"}, "processing": FAST})
    assert owner_model.apply_current(ctx) is False
    assert "processing" not in ctx.task_metadata and ctx._route_facts_pending == ""
    assert ctx.task_metadata["lead"] == {"source": "main"}


def test_a_checkpoint_keeps_what_the_mind_was_told_and_a_newer_act_is_told_once(host, monkeypatch):
    running(host)
    _tools, ctx = turn(host, monkeypatch)
    assert tier(host, "fast", request_id="t1").status_code == 200
    owner_model.apply_current(ctx)
    ctx._route_facts_pending = ""  # the next round told the mind
    saved = json.loads(json.dumps(owner_wait.continuation_state(ctx, [], {}, {}, 3, [], set())))
    assert saved["route"]["_owner_processing_applied_id"] == "t1"
    fresh_tools, fresh = turn(host, monkeypatch)
    owner_wait.restore_continuation_state(fresh_tools, saved, [], {}, {}, set())
    assert fresh.task_metadata["processing"] == FAST and fresh._route_facts_pending == ""
    assert tier(host, "standard", request_id="t2").status_code == 200
    second_tools, second = turn(host, monkeypatch)
    owner_wait.restore_continuation_state(second_tools, saved, [], {}, {}, set())
    assert second.task_metadata["processing"] == {"source": "owner", "tier": "standard"}
    assert second._route_facts_pending == STANDARD_WORDS
    loop._reset_turn_state(second)  # a new turn has told its mind nothing yet
    assert second._owner_processing_applied_id is None


def test_a_default_reset_survives_stale_copies_checkpoints_and_a_continue(host, monkeypatch):
    from ouroboros import projects_registry as projects
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.composer_cognition import update_project_processing_choice
    from supervisor.continuation_admission import _successor_task
    from tests.test_continue_work_identity import SOURCES

    root = host[0]
    save_owner({"OUROBOROS_MODEL": A, ROLES: json.dumps({"main": "standard"}), GLOBAL: "economy"})
    projects.create_project(root, "alpha", origin="owner_ui")
    update_project_processing_choice(root, "alpha", "economy")
    # The owner's work as admission stamps it (``work_kind``): a Continue of it is the owner's work again.
    task = live(host, {"lead": {"source": "main"}, "processing": ROOM_ECONOMY, "work_kind": "owner"})
    _tools, ctx = turn(host, monkeypatch, task["metadata"])
    assert task_processing_preference({"task_metadata": ctx.task_metadata}) == "economy"
    assert tier(host, "fast", request_id="fast").status_code == 200
    owner_model.apply_current(ctx)
    assert ctx.task_metadata["processing"] == FAST
    assert task_processing_preference({"task_metadata": ctx.task_metadata}) == "fast"
    ctx._route_facts_pending = ""
    checkpoint = json.loads(json.dumps(owner_wait.continuation_state(ctx, [], {}, {}, 3, [], set())))
    stale = copy.deepcopy(task)  # the worker's copy: the explicit tier
    assert stale["metadata"]["processing"] == FAST
    assert tier(host, "", request_id="reset").status_code == 200
    assert load_task_result(root, "work")["metadata"]["processing"] == DEFAULT
    # (i) A stale worker copy cannot bring the explicit tier back.
    write_task_result(root, "work", "running", metadata=stale["metadata"], owner_processing={"request_id": "stale"})
    stored = load_task_result(root, "work")
    assert stored["metadata"]["processing"] == DEFAULT and stored["owner_processing"]["request_id"] == "reset"
    # (ii) A checkpoint taken before the reset, restored over the stale copy, reads the stored Default and says so once.
    fresh_tools, fresh = turn(host, monkeypatch, stale["metadata"])
    assert fresh.task_metadata["processing"] == FAST
    owner_wait.restore_continuation_state(fresh_tools, checkpoint, [], {}, {}, set())
    assert fresh.task_metadata["processing"] == DEFAULT
    assert fresh._route_facts_pending == DEFAULT_WORDS
    assert task_processing_preference({"task_metadata": fresh.task_metadata}) == "standard"  # Main's own, again
    owner_model.restore_current(fresh)
    assert fresh.task_metadata["processing"] == DEFAULT and fresh._route_facts_pending == DEFAULT_WORDS
    _plain_tools, plain = turn(host, monkeypatch, stale["metadata"])
    owner_model.restore_current(plain)
    assert plain.task_metadata["processing"] == DEFAULT
    # The finishing pipeline's stale copy keeps it too.
    env = SimpleNamespace(drive_root=root, repo_dir=root, drive_path=lambda rel: root / rel)
    _store_task_result(env, stale, "Done.", {"rounds": 1, "cost": 0}, {"tool_calls": [], "reasoning_notes": []})
    stored = load_task_result(root, "work")
    assert stored["status"] == "completed" and stored["metadata"]["processing"] == DEFAULT
    # (iii) A Continue keeps the stored Default; the room's newer tier reaches only new work.
    update_project_processing_choice(root, "alpha", "fast")
    successor = _successor_task(SimpleNamespace(DRIVE_ROOT=root), "work", stored,
                                {"successor_task_id": "next", "chat_id": 7, "project_id": "alpha"},
                                {"cause": "context_overflow"}, SOURCES, {})
    assert successor["project_id"] == "alpha" and successor["metadata"]["processing"] == DEFAULT
    route_choice.capture_lead(successor, root)
    assert successor["metadata"]["processing"] == DEFAULT
    bare = copy.deepcopy(successor)
    bare["metadata"].pop("processing")
    route_choice.capture_lead(bare, root)
    assert bare["metadata"]["processing"] == {"source": "room", "tier": "fast"}, "the control: the room would stamp it"


@pytest.mark.parametrize("carried", [FAST, DEFAULT, ROOM_ECONOMY])
def test_a_continue_keeps_the_predecessors_stored_tier(host, carried):
    from tests.test_continue_work_identity import _successor

    write_task_result(host[0], "pred", "failed", metadata={"lead": {"source": "main"}, "processing": carried})
    assert _successor(host[0], "pred")["metadata"]["processing"] == carried


def test_a_continue_of_work_admitted_before_tiers_keeps_the_default_and_never_the_rooms_newer_tier(host):
    from ouroboros import projects_registry as projects
    from ouroboros.composer_cognition import update_project_processing_choice
    from supervisor.continuation_admission import _successor_task
    from tests.test_continue_work_identity import SOURCES

    root = host[0]
    save_owner({"OUROBOROS_MODEL": A, GLOBAL: "economy"})
    projects.create_project(root, "alpha", origin="owner_ui")
    write_task_result(root, "legacy", "failed", metadata={"lead": {"source": "main"}, "work_kind": "owner"})
    update_project_processing_choice(root, "alpha", "fast")  # the room chose Fast after that work was admitted
    successor = _successor_task(SimpleNamespace(DRIVE_ROOT=root), "legacy", load_task_result(root, "legacy"),
                                {"successor_task_id": "next", "chat_id": 7, "project_id": "alpha"},
                                {"cause": "context_overflow"}, SOURCES, {})
    assert successor["metadata"]["processing"] == DEFAULT, "the decided Default its predecessor ran on"
    route_choice.capture_lead(successor, root)  # its admission (queue.enqueue_task's own call)
    assert successor["metadata"]["processing"] == DEFAULT
    assert task_processing_preference({"task_metadata": successor["metadata"]}) == "economy"
    fresh = {"id": "new", "type": "task", "chat_id": 7, "project_id": "alpha", "metadata": {"work_kind": "owner"}}
    route_choice.capture_lead(fresh, root)
    assert fresh["metadata"]["processing"] == {"source": "room", "tier": "fast"}, "new work takes the room's tier"


OWNER_ACT = {"request_id": "reset", "receipts": {"reset": {"ok": True}}}


@pytest.mark.parametrize("existing,fields,expected", [
    ({"metadata": {"processing": DEFAULT}, "owner_processing": OWNER_ACT},
     {"metadata": {"processing": FAST, "note": 1}, "owner_processing": {"request_id": "stale"}},
     {"metadata": {"processing": DEFAULT, "note": 1}, "owner_processing": OWNER_ACT}),
    ({"metadata": {"processing": ROOM_ECONOMY}}, {"metadata": {"processing": FAST}},
     {"metadata": {"processing": ROOM_ECONOMY}}),
    ({"metadata": {"processing": FAST}}, {"metadata": {}}, {"metadata": {"processing": FAST}}),
    ({"metadata": {}, "owner_processing": OWNER_ACT}, {"metadata": {"processing": FAST}},
     {"metadata": {}, "owner_processing": OWNER_ACT}),
    ({"metadata": {"processing": FAST}}, {"status": "running"}, {"status": "running"}),
])
def test_a_workers_older_copy_never_replaces_the_stored_tier(existing, fields, expected):
    assert owner_model.preserve_carrier(existing, copy.deepcopy(fields)) == expected


# --- The card: the chosen tier beside the newest receipt of the work's own calls -----------------------------------


RECEIPT = {"requested": "fast", "submitted": "fast", "submittedNative": "priority", "observed": "fast",
           "observedNative": ["ultrafast"], "reason": None}


def _usage(ctx, source=None, receipt=None, task_id="work"):
    """One llm_usage event through the supervisor's handler; returns the receipt object the event carried."""
    from supervisor.events_budget import _handle_llm_usage

    sent = copy.deepcopy(receipt or RECEIPT)
    event = {"type": "llm_usage", "task_id": task_id, "model": A,
             "usage": {"prompt_tokens": 3, "completion_tokens": 1, "processing": sent}}
    if source is not None:
        event["source"] = source
    _handle_llm_usage(event, ctx)
    return sent


def test_the_works_own_loop_receipt_reaches_its_card_and_a_helpers_never_does(host):
    running(host)
    (host[0] / "logs").mkdir(exist_ok=True)
    meta = host[2].RUNNING["work"]
    ctx = SimpleNamespace(DRIVE_ROOT=host[0], RUNNING=host[2].RUNNING)
    _usage(ctx, "loop", {**RECEIPT, "source": "native_session", "extra": "dropped"})
    assert meta["processing_last"] == RECEIPT
    for source in ("review", "web_search", "safety_check", "vision_caption"):
        _usage(ctx, source, {"requested": "economy", "submitted": "economy", "observed": "economy", "reason": None})
    assert meta["processing_last"] == RECEIPT, "another source's call never lands on the work's card"
    # Only the loop's own events (``pricing.emit_llm_usage_event`` stamps ``source: "loop"``) land; one without a
    # source (a tool's own emitter, e.g. vision) is not the work's call.
    _usage(ctx, None, {**RECEIPT, "observed": "standard", "observedNative": ["default"]})
    assert meta["processing_last"] == RECEIPT
    sent = _usage(ctx, "loop")
    sent["observedNative"].append("mutated")  # the card keeps its own copy of what the call reported
    assert meta["processing_last"] == RECEIPT
    _usage(ctx, "loop", {**RECEIPT, "observed": "economy"}, task_id="another-task")  # another id: no row of this one
    assert meta["processing_last"] == RECEIPT


def test_a_newer_call_without_a_receipt_clears_the_older_one_from_the_card(host):
    """A call that asked no tier (a route that sends none) has no receipt: the card never says an older call's."""
    from ouroboros.lead_fact import processing_fields
    from supervisor.events_budget import _handle_llm_usage

    task = running(host)
    (host[0] / "logs").mkdir(exist_ok=True)
    meta = host[2].RUNNING["work"]
    ctx = SimpleNamespace(DRIVE_ROOT=host[0], RUNNING=host[2].RUNNING)
    _usage(ctx, "loop")
    assert meta["processing_last"] == RECEIPT
    _handle_llm_usage({"type": "llm_usage", "task_id": "work", "model": A, "source": "review",
                       "usage": {"prompt_tokens": 3, "completion_tokens": 1}}, ctx)
    assert meta["processing_last"] == RECEIPT, "another source's call without a receipt changes nothing either"
    _handle_llm_usage({"type": "llm_usage", "task_id": "work", "model": A, "source": "loop",
                       "usage": {"prompt_tokens": 3, "completion_tokens": 1}}, ctx)
    assert "processing_last" not in meta
    assert "last" not in processing_fields({**task, "processing_last": meta.get("processing_last")})["processing"]


def test_a_vision_helpers_receipt_never_lands_on_the_works_card(host):
    import queue

    from ouroboros.tools.vision import _emit_usage
    from supervisor.events_budget import _handle_llm_usage

    running(host)
    (host[0] / "logs").mkdir(exist_ok=True)
    meta = host[2].RUNNING["work"]
    ctx = SimpleNamespace(DRIVE_ROOT=host[0], RUNNING=host[2].RUNNING)
    _usage(ctx, "loop")
    # The vision tool reports its own helper call under the work's id, as the tool emits it.
    tool = SimpleNamespace(event_queue=queue.Queue(), task_id="work", current_task_type="task")
    _emit_usage(tool, {"prompt_tokens": 3, "completion_tokens": 1, "processing": {
        "requested": "economy", "submitted": "economy", "submittedNative": "flex", "observed": "economy",
        "observedNative": ["flex"], "reason": None}}, "openai::vision-model")
    [event] = list(tool.event_queue.queue)
    _handle_llm_usage(event, ctx)
    assert meta["processing_last"] == RECEIPT, "a helper's call is not the work's own call"


def test_the_works_card_shows_its_tier_and_what_ran(host, monkeypatch):
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot

    running(host)
    (host[0] / "logs").mkdir(exist_ok=True)

    def card():
        raw, owners = {}, []
        [activity] = _chat_activities_snapshot_safe(host[0], work_facts=raw, decision_owners=owners)
        [work] = works_snapshot(host[0], owners, raw)["works"]
        assert activity["processing"] == work["processing"]
        return work["processing"]
    assert card() == {"choice": "", "resolved": ""}, "admitted before tiers: the default resolution"
    assert tier(host, "fast", request_id="tier").status_code == 200
    assert card() == {"choice": "fast", "resolved": "fast", "source": "owner"}
    _usage(SimpleNamespace(DRIVE_ROOT=host[0], RUNNING=host[2].RUNNING), "loop")
    assert card() == {"choice": "fast", "resolved": "fast", "source": "owner", "last": RECEIPT}
    assert tier(host, "", request_id="reset").status_code == 200
    assert card() == {"choice": "", "resolved": "", "last": RECEIPT}


def test_a_queued_works_card_says_its_admitted_tier(host):
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot

    queued(host, metadata={"lead": {"source": "main"}, "processing": ROOM_ECONOMY})
    raw, owners = {}, []
    [activity] = _chat_activities_snapshot_safe(host[0], work_facts=raw, decision_owners=owners)
    [work] = works_snapshot(host[0], owners, raw)["works"]
    assert activity["processing"] == work["processing"] == {"choice": "economy", "resolved": "economy", "source": "room"}


def test_api_state_publishes_the_running_works_tier(host):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.state import api_state

    running(host)
    (host[0] / "logs").mkdir(exist_ok=True)
    assert tier(host, "economy", request_id="tier").status_code == 200
    _usage(SimpleNamespace(DRIVE_ROOT=host[0], RUNNING=host[2].RUNNING), "loop")
    app = Starlette(routes=[Route("/api/state", api_state)])
    app.state.drive_root, app.state.app_start = host[0], 0.0
    response = TestClient(app).get("/api/state")
    assert response.status_code == 200, response.text
    [work] = response.json()["works"]  # the one live root
    assert work["processing"] == {"choice": "economy", "resolved": "economy", "source": "owner", "last": RECEIPT}
    assert response.json()["processing_tiers"] == {"main": "", "default": ""}


def test_a_conversation_turns_admitted_tier_reaches_its_row(host, monkeypatch):
    from ouroboros.lead_fact import processing_fields
    from supervisor import conversation_lane

    task = {"id": "turn", "type": "task", "chat_id": 7, "metadata": {}, "_is_direct_chat": True}
    host[2].RUNNING["turn"] = {"task": task, "attempt": 2}
    ctx = SimpleNamespace(RUNNING=host[2].RUNNING)
    bound = {"type": "direct_turn_bound", "task_id": "turn", "task_attempt": 2}
    conversation_lane.handle_direct_turn_bound({**bound, "facts": {"processing": {"source": "room", "tier": "turbo"}}}, ctx)
    assert "processing" not in task["metadata"], "an unreadable carrier is not a tier"
    conversation_lane.handle_direct_turn_bound({**bound, "task_attempt": 1, "facts": {"processing": ROOM_ECONOMY}}, ctx)
    assert "processing" not in task["metadata"], "another attempt's frame moves nothing"
    conversation_lane.handle_direct_turn_bound({**bound, "facts": {"processing": ROOM_ECONOMY}}, ctx)
    assert task["metadata"]["processing"] == ROOM_ECONOMY
    assert processing_fields({**task, "processing_last": RECEIPT}) == {
        "processing": {"choice": "economy", "resolved": "economy", "source": "room", "last": RECEIPT}}
    assert processing_fields({**task, "processing_last": {**RECEIPT, "observed": ""}}) == {
        "processing": {"choice": "economy", "resolved": "economy", "source": "room"}}, "a receipt that saw nothing says nothing ran"


@pytest.mark.parametrize("owner", ["fast", ""])  # the owner's Fast, and the owner's Default (recorded as a decision)
def test_an_older_admission_frame_never_replaces_the_owners_tier_act_on_the_row(host, owner):
    """The owner set the tier while the turn was still preparing (no lead fact yet); the carrier's admission frame
    arrives after. The row keeps the owner's decision; the frame's Project facts still land."""
    from ouroboros.lead_fact import processing_fields
    from supervisor import conversation_lane

    task = live(host, {})
    host[2].RUNNING["work"].pop("lead")  # preparation: no lead fact yet
    assert tier(host, owner).status_code == 200
    decided = copy.deepcopy(task["metadata"]["processing"])
    assert decided == ({"source": "owner", "tier": owner} if owner else DEFAULT)
    ctx = SimpleNamespace(RUNNING=host[2].RUNNING)
    conversation_lane.handle_direct_turn_bound({"type": "direct_turn_bound", "task_id": "work", "task_attempt": 1,
        "facts": {"processing": ROOM_ECONOMY, "project_id": "room-a"}}, ctx)
    assert task["metadata"]["processing"] == decided and task["project_id"] == "room-a"
    assert processing_fields(task)["processing"]["choice"] == (owner or "")
    # Control: a row without an owner act takes the admitted room tier from the same frame.
    task.pop("owner_processing")
    conversation_lane.handle_direct_turn_bound({"type": "direct_turn_bound", "task_id": "work", "task_attempt": 1,
        "facts": {"processing": ROOM_ECONOMY}}, ctx)
    assert task["metadata"]["processing"] == ROOM_ECONOMY


@pytest.mark.parametrize('own,expected', [('', 'economy'), ('standard', 'standard'), ('fast', 'fast')])
def test_card_resolves_the_acting_fallback_and_pending_return_through_server(host, own, expected):
    from ouroboros.lead_fact import lead_processing_fields
    from ouroboros.model_slots import task_processing_preference
    from tests._task_settings_helpers import save_owner

    save_owner({'OUROBOROS_PROCESSING_PREFERENCE': 'standard',
                'OUROBOROS_MODEL_PROCESSING_PREFERENCES': {'main': 'fast', 'fallback': ['economy']}})
    task = live(host, {'lead': {'source': 'main'}, 'processing': DEFAULT})
    lead = {'model': B, 'use_local': False, 'basis': 'fallback', 'role': 'fallback:0',
            'primary': {'model': A, 'use_local': False, 'basis': 'main'}}
    host[2].RUNNING['work']['lead'] = lead
    receipt = tier(host, own).json()['processing']
    assert receipt['resolved'] == expected
    fact = lead_processing_fields(lead, {**task, 'processing_last': RECEIPT})['processing']
    assert fact['resolved'] == task_processing_preference(task, model_role='fallback:0') == expected
    assert fact['last'] == RECEIPT  # a new choice never rewrites the last actual call
    if not own:
        task['owner_model'] = {'request_id': 'return-to-main'}
        pending = lead_processing_fields(lead, task)['processing']
        assert pending['resolved'] == 'fast'
