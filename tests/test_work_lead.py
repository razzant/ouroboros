"""One acting fact through the loop, host census, work projection and routing manifest."""
from __future__ import annotations

import json
import queue
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import lead_fact as leads, route_choice
from ouroboros.primary_route_observation import record_round_route_result, route_facts_text
from tests._task_settings_helpers import isolated_runtime, save_owner  # noqa: F401
from tests.test_place_focus import focus as focus

A, B, C = "openai::a", "openai::b", "openai::c"


def context(source="main", *, model=A, local=False, role="main", metadata=None):
    explicit = source != "main" and source not in {"consciousness", "light"}
    return SimpleNamespace(task_id="work", task_attempt=1, task_metadata=metadata or {},
        active_model=model, active_use_local=local, event_queue=queue.Queue(),
        task_model_override=model if explicit else None, task_route_source=source if explicit else None,
        task_use_local_override=local if explicit else None,
        primary_route={"model": model, "use_local": local, "role": role},
        route_choice=route_choice.explicit(source) if explicit else route_choice.following(source),
        context_fit_plan=SimpleNamespace(model_role=role), _execution_trace={})


@pytest.mark.parametrize("source,role,metadata,basis", [
    ("main", "main", {}, "main"), ("consciousness", "consciousness", {"model_role": "consciousness"}, "consciousness"),
    ("room", "lead", {}, "room"), ("owner", "lead", {}, "owner"), ("owner_words", "lead", {}, "owner_words"),
])
def test_primary_fact_names_its_chooser_without_a_departure(source, role, metadata, basis):
    ctx = context(source, role=role, metadata=metadata)
    assert leads.lead_fact(ctx) == {"model": A, "use_local": False, "basis": basis}
    leads.publish_lead(ctx)
    leads.publish_lead(ctx)
    assert ctx.event_queue.qsize() == 1
    assert ctx.event_queue.get()["lead"]["basis"] == basis


def place_context(focus, source):
    from ouroboros.presence_context import focus_metadata

    root, place, admission, event = focus
    admission = replace(admission, place=replace(place, model=A if source == "place" else ""),
                        circle={"model": A} if source == "circle" else None)
    metadata = focus_metadata(admission, event, "work", 7, root)
    ctx = context("consciousness", role="consciousness", metadata=metadata)
    ctx.context_fit_plan = None
    route_choice.bind_task_route(ctx, metadata, presence=True)
    model, _effort, local, *_ = route_choice.initial_round_route(ctx, None, "medium")
    ctx.active_model, ctx.active_use_local = model, local
    return ctx


@pytest.mark.parametrize("source", ["place", "circle", "consciousness"])
def test_place_turn_publishes_the_carriers_chooser_or_follows_consciousness(focus, source):
    save_owner({"OUROBOROS_MODEL_CONSCIOUSNESS": C})
    ctx = place_context(focus, source)
    assert ctx.task_metadata["model_role"] == "consciousness"
    carried = source != "consciousness"
    assert ctx.primary_route["role"] == ("lead" if carried else "consciousness")
    assert ("lead" in ctx.task_metadata) is carried
    leads.publish_lead(ctx)
    assert ctx.event_queue.get_nowait()["lead"] == {
        "model": A if carried else C, "use_local": False, "basis": source}


@pytest.mark.parametrize("obsolete_layer", [None, "circle"])
def test_legacy_light_presence_keeps_its_basis_without_a_place_carrier(focus, obsolete_layer):
    from ouroboros.presence_runner import _build_task

    save_owner({"OUROBOROS_MODEL_LIGHT": B})
    root, _place, admission, event = focus
    task = _build_task(replace(admission, place=None, circle=None, model_slot="light"), event,
                       drive_root=root, staged_files=())
    if obsolete_layer:
        task["metadata"]["model_layer"] = obsolete_layer  # this annotation cannot claim circle provenance
    ctx = context(metadata=task["metadata"])
    ctx.context_fit_plan = None
    route_choice.bind_task_route(ctx, ctx.task_metadata, presence=True)
    model, _effort, local, *_ = route_choice.initial_round_route(ctx, None, "medium")
    ctx.active_model, ctx.active_use_local = model, local
    assert ctx.task_route_source == "presence" and "lead" not in ctx.task_metadata
    leads.publish_lead(ctx)
    assert ctx.event_queue.get_nowait()["lead"] == {"model": B, "use_local": False, "basis": "place"}


@pytest.mark.parametrize("source,basis", [("switch_model", "mind"), ("fallback", "fallback")])
def test_departure_has_primary_date_cause_and_emits_only_on_change(source, basis):
    ctx = context("owner", role="lead")
    leads.publish_lead(ctx)
    ctx.active_model, ctx.route_choice = B, route_choice.explicit(source)
    ctx.context_fit_plan.model_role = "fallback:1" if basis == "fallback" else "lead"
    first = leads.lead_fact(ctx)
    assert first["primary"] == {"model": A, "use_local": False, "basis": "owner"}
    assert first["basis"] == basis and first["since"] and first["cause"]
    assert (first.get("role") == "fallback:1") is (basis == "fallback")
    leads.publish_lead(ctx)
    leads.publish_lead(ctx)
    assert ctx.event_queue.qsize() == 2
    # A checkpoint preserves the episode's original date, including JSON's tuple conversion.
    ctx._lead_episode = json.loads(json.dumps(ctx._lead_episode))
    assert leads.lead_fact(ctx) == first
    ctx.active_model, ctx.route_choice = A, route_choice.opening_choice(ctx)
    ctx.context_fit_plan.model_role = "lead"
    leads.publish_lead(ctx)
    assert ctx.event_queue.qsize() == 3
    assert leads.lead_fact(ctx) == {"model": A, "use_local": False, "basis": "owner"}


def test_same_model_on_another_chain_account_is_still_a_fallback():
    model = "claudexor::codex=example"
    save_owner({"OUROBOROS_MODEL": model, "OUROBOROS_MODEL_ACCOUNTS": json.dumps({
        "main": "primary-account", "fallback": ["first-account", "second-account"]})})
    ctx = context(model=model)
    ctx.route_choice = route_choice.explicit("fallback")
    first = leads.lead_fact(ctx, role="fallback:0")
    second = leads.lead_fact(ctx, role="fallback:1")
    assert first["basis"] == second["basis"] == "fallback"
    assert first["primary"]["model"] == second["model"] == model
    assert first["role"] == "fallback:0" and second["role"] == "fallback:1"
    assert "primary" not in leads.lead_fact(ctx, role="main")


@pytest.mark.parametrize("source,role", [("main", "main"), ("owner", "lead")])
def test_wait_card_move_names_the_owner_and_unchanged_primary_without_pending(source, role):
    save_owner({"OUROBOROS_MODEL": A})
    ctx = context(source, role=role)
    ctx.model_wait_context = SimpleNamespace(overrides={role: {"model": B, "use_local": False}})
    ctx.active_model, ctx.route_choice = B, route_choice.explicit("wait_card")
    fact = leads.lead_fact(ctx)
    assert fact["model"] == B and fact["basis"] == "owner"
    assert fact["primary"] == {"model": A, "use_local": False, "basis": source}
    assert fact["since"] and fact["cause"] == "wait_card"
    save_owner({"OUROBOROS_MODEL": C})
    assert leads.project_lead(fact) == fact  # this explicit route has no next-call move
    assert ctx.primary_route["model"] == A
    ctx.model_wait_context.overrides.clear()
    ctx.active_model, ctx.route_choice = A, route_choice.opening_choice(ctx)
    returned = leads.lead_fact(ctx)
    assert returned == {"model": A, "use_local": False, "basis": source}
    projected = leads.project_lead(returned)
    assert (projected.get("pending") == {"model": C, "use_local": False}) is (source == "main")


@pytest.mark.parametrize("action", ["return", "wait"])
@pytest.mark.parametrize("explicit_switch", [False, True], ids=["primary", "explicit-light"])
def test_return_request_keeps_fallback_until_its_own_accepted_reply(monkeypatch, action, explicit_switch):
    from ouroboros import loop
    from ouroboros.loop_routes import bind_round_route
    from ouroboros.tools.control_runtime import _switch_model

    save_owner({"OUROBOROS_MODEL": A, "OUROBOROS_MODEL_LIGHT": C})
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    ctx = context("room", role="lead")
    ctx.active_model, ctx.active_effort_override = B, None
    leads.adopt_fallback(ctx, False, {"failure_code": "transport_unavailable"}, "fallback:0")
    departed = ctx.event_queue.get()["lead"]
    assert _switch_model(ctx, primary=action).startswith("OK:")
    assert leads.lead_fact(ctx) == departed  # requesting is not returning
    assert "Unknown model" in _switch_model(ctx, model="openai::not-configured")
    assert leads.lead_fact(ctx) == departed  # a refused switch changes nothing
    if explicit_switch:
        assert _switch_model(ctx, model=C).startswith("OK:")  # same tool batch, before the boundary
    tools = SimpleNamespace(_ctx=ctx)
    model, local, *_ = bind_round_route(tools, [], (B, False, "medium"), ctx.context_fit_plan, "max", "max", [])
    assert (model, local, ctx.route_wait_on_primary) == (C if explicit_switch else A, False,
                                                      action == "wait" and not explicit_switch)
    leads.publish_lead(ctx)
    if explicit_switch:
        current = ctx.event_queue.get_nowait()["lead"]
        assert (current["model"], current["basis"], current["cause"]) == (C, "mind", "switch_model")
        assert current["primary"] == {"model": A, "use_local": False, "basis": "room"}
        assert current["since"] and "role" not in current
    else:
        current = departed
    assert ctx.event_queue.empty() and leads.lead_fact(ctx) == current
    call = SimpleNamespace(tools=tools, active_model=model, active_use_local=local, accumulated_usage={})
    record_round_route_result(call, "lead", accepted=False)
    assert leads.lead_fact(ctx) == current
    record_round_route_result(call, "lead", accepted=True)
    if explicit_switch:
        assert leads.lead_fact(ctx) == current  # Light's reply cannot revive the cancelled return hold
    else:
        returned = ctx.event_queue.get()["lead"]
        assert returned == {"model": A, "use_local": False, "basis": "room"}
    assert ctx.event_queue.empty()


def test_a_fallback_adopting_the_primary_ends_the_fact_without_a_return_request():
    ctx = context("owner", role="lead")
    ctx.active_model = B
    leads.adopt_fallback(ctx, False, {"failure_code": "quota"}, "fallback:0")
    ctx.active_model = A
    leads.adopt_fallback(ctx, True, {"failure_code": "bad_request"}, "lead")
    assert [ctx.event_queue.get()["lead"]["basis"] for _ in range(2)] == ["fallback", "owner"]


def test_live_followers_show_pending_without_relabelling_their_last_call():
    save_owner({"OUROBOROS_MODEL": A})
    fact = {"model": A, "use_local": False, "basis": "main"}
    assert leads.project_lead(fact) == fact
    save_owner({"OUROBOROS_MODEL": B, "USE_LOCAL_MAIN": True})
    assert leads.project_lead(fact) == {**fact, "pending": {"model": B, "use_local": True}}
    assert "pending" not in fact
    assert "pending" not in leads.project_lead({**fact, "basis": "owner"})
    assert "pending" not in leads.project_lead({**fact, "basis": "fallback", "role": "fallback:0"})


@pytest.mark.parametrize("carrier", [{"source": "main"}, {"source": "room", "model": C, "use_local": True}])
def test_queue_fact_reads_the_carrier_and_only_a_follow_reads_main(carrier):
    save_owner({"OUROBOROS_MODEL": B})
    fact = leads.project_lead(task={"metadata": {"lead": carrier}}, queued=True)
    assert fact == {"model": B if carrier["source"] == "main" else C,
                    "use_local": carrier.get("use_local", False), "basis": carrier["source"]}
    assert leads.project_lead(task={"metadata": {"model": C}}, queued=True) == {}


def test_owner_lead_facts_name_the_chooser_and_keep_the_primary_return_affordances():
    ctx = context("owner_words", role="lead")
    assert leads.model_fact_words(ctx, A) == f"model {A} — set by my human's words"
    text = route_facts_text(ctx, model=B, use_local=False, role="fallback:0", failed_model=A,
                            failure={"failure_code": "quota"}, waiter=None)
    assert "set by my human's words" in text
    assert 'switch_model(primary="return")' in text and 'switch_model(primary="wait")' in text
    assert leads.model_fact_words(context(), A) == f"model {A}"
    child = context(metadata={"delegation_role": "subagent"})
    leads.publish_lead(child)
    assert child.event_queue.empty() and leads.lead_fact(child) == {}


@pytest.mark.parametrize("span", ["Use this route for the whole report.", "Длинная фраза 🙂 " * 100], ids=["whole", "clipped"])
@pytest.mark.parametrize("source", ["owner_words", "owner", "mind"])
def test_only_the_cited_primary_publishes_its_bounded_words_and_keeps_them_when_away(span, source):
    carrier = {"source": source, "model": A, "use_local": False, "cited": span}
    ctx = context(source, role="lead", metadata={"lead": carrier})
    citation = {"text": span[:leads.MAX_CITED_CHARS], "clipped": len(span) > leads.MAX_CITED_CHARS}
    expected = {"cited": citation} if source == "owner_words" else {}
    fact = leads.lead_fact(ctx)
    assert fact == {"model": A, "use_local": False, "basis": source, **expected}
    assert leads.project_lead(task={"metadata": ctx.task_metadata}, queued=True) == fact
    snapshot = route_choice.lead_snapshot(carrier)
    assert snapshot.get("cited") == (span if source == "owner_words" else None)
    words = leads.model_fact_words(ctx, A)
    assert (f"«{citation['text']}»" in words) is (source == "owner_words")
    assert ("[quote clipped]" in words) is (source == "owner_words" and citation["clipped"])
    ctx.active_model, ctx.route_choice = B, route_choice.explicit("fallback")
    away = leads.lead_fact(ctx, role="fallback:0")
    assert "cited" not in away and away["primary"] == fact
    facts_line = route_facts_text(ctx, model=B, use_local=False, role="fallback:0", failed_model=A,
                                 failure={"failure_code": "quota"}, waiter=None)
    assert (f"«{citation['text']}»" in facts_line) is (source == "owner_words")
    ctx.active_model, ctx.route_choice = A, route_choice.opening_choice(ctx)
    assert leads.lead_fact(ctx) == fact
    assert ctx.task_metadata["lead"]["cited"] == span  # publication never truncates the record


@pytest.mark.parametrize("carrier", ["worker", "lane", "queued"])
def test_fact_crosses_the_event_census_work_and_manifest_boundaries(tmp_path, monkeypatch, carrier):
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot
    from ouroboros.server_routing_context import addressable_roots
    from ouroboros.task_results import write_task_result
    from supervisor.events import _handle_work_lead
    from tests._budget_pause_exact_helpers import _install_queue

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    save_owner({"OUROBOROS_MODEL": A})
    task = {"id": "work", "type": "task", "chat_id": 7, "root_task_id": "work",
            "metadata": {"lead": {"source": "room", "model": B, "use_local": False}}}
    write_task_result(tmp_path, "work", "pending" if carrier == "queued" else "running", **{k: v for k, v in task.items() if k != "id"})
    meta = {"task": task, "attempt": 1, "started_at": 100.0}
    if carrier == "queued":
        workers.PENDING.append(task)
        expected = {"model": B, "use_local": False, "basis": "room"}
    else:
        if carrier == "lane":
            from supervisor.conversation_lane import LANE_CONVERSATION
            task["_is_direct_chat"] = True
            meta.update(lane=LANE_CONVERSATION, worker_id=3)
        workers.RUNNING["work"] = meta
        ctx = context("owner", model=C, role="lead")
        leads.publish_lead(ctx)
        evt = ctx.event_queue.get()
        _handle_work_lead({**evt, "task_attempt": 2}, q)
        assert "lead" not in meta
        _handle_work_lead(evt, q)
        expected = evt["lead"]
    raw, owners = {}, []
    [activity] = _chat_activities_snapshot_safe(tmp_path, work_facts=raw, decision_owners=owners)
    [work] = works_snapshot(tmp_path, owners, raw)["works"]
    [root] = addressable_roots(q).rows()
    assert activity["lead"] == work["lead"] == root["lead"] == expected


def test_direct_turn_folds_only_its_live_attempt(tmp_path, monkeypatch):
    from supervisor import active_activity
    from supervisor.events import _handle_work_lead

    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: registry)
    ctx = context("room", role="lead")
    ctx.model_wait_context = SimpleNamespace(task_id="work", attempt=2, closed=False, overrides={})
    actor = SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), drive_root=tmp_path)
    entry = registry.register("work", 7, actor=actor)
    leads.publish_lead(ctx)
    evt = ctx.event_queue.get()
    host = SimpleNamespace(RUNNING={})
    _handle_work_lead(evt, host)
    assert entry.lead is None
    _handle_work_lead({**evt, "task_attempt": 2}, host)
    assert registry.snapshot()[0]["lead"] == evt["lead"]
    registry.unregister("work")
    _handle_work_lead({**evt, "task_attempt": 2}, host)
    assert registry.snapshot() == []


@pytest.mark.parametrize("source", ["place", "circle"])
def test_inline_presence_uses_the_same_published_fact(focus, monkeypatch, source):
    from ouroboros import conversation_lease, task_results

    ctx = place_context(focus, source)
    leads.publish_lead(ctx)
    actor = SimpleNamespace(tools=SimpleNamespace(_ctx=ctx))
    monkeypatch.setattr(conversation_lease, "live_presence_roots", lambda root: {
        "complete": True, "roots": [{"task_id": "work", "local": True}]})
    monkeypatch.setattr(conversation_lease, "presence_author_agent", lambda *args: actor)
    monkeypatch.setattr(task_results, "load_task_result", lambda *args, **kwargs: {
        "status": "running", "execution_owner": {"kind": "presence"}})
    [fact] = conversation_lease.resolved_presence_turns(focus[0])["roots"]
    assert fact["row"]["lead"] == ctx._lead_last_sent == {"model": A, "use_local": False, "basis": source}
