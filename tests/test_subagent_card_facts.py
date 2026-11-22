"""Actual live and history frames feed the card's shared Python/JS fixture."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from ouroboros.subagent_messages import SUBAGENT_MESSAGE_FIELDS, subagent_message_meta
from ouroboros.subagent_runtime import cognitive_route_billing_basis
from tests._cases import case_folders

CASES = json.loads((Path(__file__).parent / "fixtures/subagent_card_facts.json").read_text())
COORDINATOR_FIELDS = {"parent_cognitive_route", "coordinator_route", "coordinator_billing",
                      "coordinator_provenance", "use_local_model"}


def test_producer_metadata_carries_the_card_facts(subtests):
    for case in CASES:
        with subtests.test(case=case):
            from ouroboros.supervisor_route import cognitive_route_provenance

            frame = subagent_message_meta(case["task"], task_id="child", event="running")
            expected = {key: "" for key in ("subagent_task_id", "root_task_id", "parent_task_id", "delegation_role",
                "subagent_role", "write_surface", "task_group_id", "model_lane", "effective_model_lane", "model",
                "executor_route", "effort_level", "effort_requested", "effort_source")}
            expected.update(subagent_task_id="child", delegation_role="subagent", subagent_event="running", **case["facts"])
            assert frame == expected
            assert set(frame) <= set(SUBAGENT_MESSAGE_FIELDS), "speech persistence must retain the same facts"
            assert subagent_message_meta({"model": "root-model"}) == {}
            assert (frame.get("coordinator_provenance") or cognitive_route_provenance(
                frame.get("parent_cognitive_route"))) == case["provenance"]


def test_history_keeps_only_recorded_card_facts_without_task_result(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for recorded in [True, False]:
        for case in CASES:
            tmp_path = next(case_paths)
            with subtests.test(recorded=recorded, case=case):
                from ouroboros.gateway.contracts import ChatOutbound
                from ouroboros.gateway.history import _load_terminal_result, make_chat_history_endpoint

                frame = subagent_message_meta({**case["task"], "root_task_id": "parent", "parent_task_id": "parent"},
                                              task_id="child", event="running")
                facts = case["facts"] if recorded else {k: v for k, v in case["facts"].items() if k not in COORDINATOR_FIELDS}
                if not recorded:
                    frame = {k: v for k, v in frame.items() if k not in COORDINATOR_FIELDS}
                assert set(frame) <= get_type_hints(ChatOutbound).keys()
                unrelated = {"raw_prompt": "private", "coordinator_account": "not a card fact", "chosen_by": "delegator",
                             "source_role": "fallback", "selected_supervisor_snapshot": {"handle": "unrelated"}}
                logs = tmp_path / "logs"
                logs.mkdir()
                (logs / "chat.jsonl").write_text("", encoding="utf-8")
                rows = [{"ts": "2026-10-22T00:00:00Z", "task_id": "parent", "content": "parent working"},
                        {"ts": "2026-10-22T00:00:01Z", "task_id": "child", "content": "child working", **frame, **unrelated}]
                (logs / "progress.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                assert _load_terminal_result(tmp_path, "child", {}) == {"_history_result_absent": True}
                response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
                assert response.status_code == 200
                replay = next(row for row in json.loads(response.body)["messages"] if row.get("task_id") == "child")
                assert replay["is_progress"] and {key: replay[key] for key in frame} == frame
                assert {key: replay[key] for key in facts} == facts, "the web fixture is the actual replayed card input"
                assert not unrelated.keys() & replay.keys(), "replay must not broaden its metadata whitelist"
                if not recorded:
                    assert not COORDINATOR_FIELDS & replay.keys(), "replay cannot reconstruct absent facts"


def test_history_names_a_childs_selection_from_its_task_for_rows_of_the_same_attempt(tmp_path):
    """A row recorded before the selection rode the frames takes it from its task's display view on replay:
    the queued, running and terminal card name the model the row selected, its account pin never."""
    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests._store_shared import seed_result

    route = {"kind": "agent_session", "target_id": "codex=gpt-5.5"}
    seed_result(tmp_path, "child", {"task_id": "child", "status": "running", "delegation_role": "subagent",
                                    "parent_task_id": "parent", "root_task_id": "parent",
                                    "configured_subagent": {"schema": 1, "selected_subagent_id": "row", "config_fingerprint": "fp",
                                                            "route": {**route, "credential_profile_id": "work-account"}}})
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [{"ts": "2026-11-22T00:00:00Z", "task_id": "parent", "content": "parent working"},
            {"ts": "2026-11-22T00:00:01Z", "task_id": "child", "content": "child working", "subagent_event": "running",
             "subagent_task_id": "child", "parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent"}]
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    replay = next(row for row in json.loads(response.body)["messages"] if row.get("task_id") == "child")
    assert replay["requested_executor_route"] == route


SELECTED = {"kind": "agent_session", "target_id": "codex=gpt-5.5"}
SNAPSHOT = {"schema": 1, "selected_subagent_id": "row", "config_fingerprint": "fp",
            "route": {**SELECTED, "credential_profile_id": "work-account"}}
COGNITION = {"model": "openai/gpt-6-astra", "use_local_model": False, "chosen_by": "inherited"}


def test_history_admission_notices_use_the_refused_children_and_keep_the_parent_anchor(tmp_path, monkeypatch):
    from ouroboros import task_store
    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests._store_shared import seed_result

    evidence = {"delegated_runs_started": 1}
    seed_result(tmp_path, "parent", {"status": "completed", "delegation_role": "subagent", "chat_id": 1,
                                    "parent_task_id": "root", "root_task_id": "root", "model": "parent-model",
                                    "executor_route": "claude=claude-opus-5-5", "coordinator_route": "parent-model",
                                    "subagent_envelope": {"execution_evidence": evidence, "actual_substrate": "harness_used"}})
    child_ids = ["refused-a", "refused-b"]
    for tid in child_ids:
        seed_result(tmp_path, tid, {"status": "failed", "delegation_role": "subagent", "chat_id": 1,
                                   "parent_task_id": "parent", "root_task_id": "root", "configured_subagent": SNAPSHOT,
                                   "parent_cognitive_route": COGNITION})
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [{"ts": "2026-11-22T00:00:00Z", "task_id": "parent", "content": "Parent working."}]
    rows += [{"ts": f"2026-11-22T00:00:0{i}Z", "task_id": "parent", "subagent_task_id": tid,
              "parent_task_id": "parent", "root_task_id": "root", "delegation_role": "subagent",
              "role": "system", "system_type": "subagent_admission_notice", "content": "Request refused.",
              "subagent_event": "rejected", "accepted": False, "status": "failed"} for i, tid in enumerate(child_ids, 1)]
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    batches = []
    display_documents = task_store.display_documents

    def record_batch(root, task_ids, *args, **kwargs):
        batches.append(set(task_ids))
        return display_documents(root, task_ids, *args, **kwargs)

    monkeypatch.setattr(task_store, "display_documents", record_batch)
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "20"})))
    assert response.status_code == 200
    replay = json.loads(response.body)["messages"]
    parent = next(row for row in replay if row["text"] == "Parent working.")
    assert parent["executor_route"] == "claude=claude-opus-5-5"
    assert parent["task_terminal_status"] == "completed" and parent["execution_evidence"] == evidence
    notices = [row for row in replay if row.get("system_type") == "subagent_admission_notice"]
    assert [row["subagent_task_id"] for row in notices] == child_ids
    for notice in notices:
        assert notice["task_id"] == "parent" and notice["status"] == "failed"
        assert notice["requested_executor_route"] == SELECTED and notice["parent_cognitive_route"] == COGNITION
        assert notice["coordinator_route"] == COGNITION["model"]
        assert not notice.get("executor_route") and not notice.get("model")
        assert not {"task_terminal_status", "execution_evidence", "actual_substrate", "tool_evidence"} & notice.keys()
    assert [batch & set(child_ids) for batch in batches if batch.intersection(child_ids)] == [set(child_ids)]


@pytest.mark.parametrize("producer", ["invalid_depth", "project_state", "preflight_unreadable", "preparation_unknown",
                                     "preparation_live", "preparation_recovered", "preparation_write_failed",
                                     "preparation_recovered_write_failed"])
def test_every_rejection_producer_names_the_refused_childs_selection_live_and_on_replay(tmp_path, monkeypatch, producer):
    """Admission and preparation refusals retain the request even when the result cannot answer."""
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.task_results import STATUS_REQUESTED, load_task_result, write_task_result
    from supervisor import events, events_subagent_admission, message_bus, queue, task_admission
    from supervisor import workspace_preparation as prep

    (tmp_path / "logs").mkdir()
    live = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = live.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
    monkeypatch.setattr(events_subagent_admission, "_bound_project_chat_id", lambda *_: None)
    for name, value in (("DRIVE_ROOT", tmp_path), ("PENDING", []), ("RUNNING", {}), ("ADMISSION_RESERVATIONS", {})):
        monkeypatch.setattr(queue, name, value)
    write_task_result(tmp_path, "parent", "running", chat_id=1, delegation_role="root", root_task_id="parent")
    tid, token = f"refused-{producer}", "claim-token"
    # The producer's REQUESTED row and event (control_scheduling): both carry the row's whole snapshot.
    facts = {"objective": f"check {tid}", "expected_output": "a note", "role": "checker", "parent_task_id": "parent",
             "root_task_id": "parent", "delegation_role": "subagent", "chat_id": 1, "configured_subagent": SNAPSHOT,
             "parent_cognitive_route": COGNITION}
    marker = {"token": token, "phase": prep.QUEUED, "generation": "gen", "owner": {}, "surface": "",
              "claimed_at": "2026-11-22T00:00:00Z", "updated_at": "2026-11-22T00:00:00Z"}
    write_task_result(tmp_path, tid, STATUS_REQUESTED, description=facts["objective"],
                      **({prep.MARKER: marker} if producer.startswith("preparation") else {}), **facts)
    event = {**facts, "type": "schedule_subagent", "task_id": tid, "description": facts["objective"], "depth": 1}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={}, WORKERS={0: SimpleNamespace(busy_task_id=None)},
                          send_with_budget=message_bus.send_with_budget, load_state=lambda: {"owner_chat_id": 1},
                          persist_queue_snapshot=lambda reason="": None, enqueue_task=lambda task: pytest.fail("enqueued"))
    def inaccessible(*_args, **_kwargs):
        raise OSError("task authority unavailable")

    unrecorded = producer in {"preflight_unreadable", "preparation_unknown", "preparation_write_failed",
                              "preparation_recovered_write_failed"}
    if producer == "invalid_depth":
        events._handle_schedule_task({**event, "depth": "not-a-depth"}, ctx)
    elif producer == "project_state":
        from ouroboros import projects_registry

        def unreadable(*_args, **_kwargs):
            raise RuntimeError("registry unreadable")

        with monkeypatch.context() as fault:  # only admission's read fails; the replay below reads the registry
            fault.setattr(projects_registry, "project_binding_for_task", unreadable)
            events._handle_schedule_task({**event, "project_id": "proj-1"}, ctx)
    elif producer == "preflight_unreadable":
        monkeypatch.setattr(task_admission, "subagent_schedule_owned", inaccessible)
        events._handle_schedule_task(event, ctx)
    else:
        if "recovered" not in producer:
            claim = prep.PreparationClaim(task_id=tid, ctx=ctx, token=token, generation=None, generation_token="gen",
                                          owner={}, event=event, route={"chat_id": 1, "delegation_role": "subagent",
                                          "parent_id": "parent", "role": "checker", "root_task_id": "parent"})
        else:  # after a restart: only the durable row and its marker, no event
            row = load_task_result(tmp_path, tid, strict=True)
            claim = prep.PreparationClaim.recovered(tmp_path, row, row[prep.MARKER])
        if producer == "preparation_unknown":
            claim.disclose_unknown()
        else:
            if unrecorded:
                monkeypatch.setattr(task_admission, "persist_never_admitted_refusal", inaccessible)
            assert claim.refuse("Subagent not scheduled: its workspace preparation failed: boom",
                                "admission_preparation_failed") is (not unrecorded)
    stored = load_task_result(tmp_path, tid, strict=True)
    assert stored["status"] == (STATUS_REQUESTED if unrecorded else "failed")
    assert stored["configured_subagent"] == SNAPSHOT and stored["parent_cognitive_route"] == COGNITION
    notices = [frame for frame in live if frame.get("subagent_event") == "rejected" and frame.get("subagent_task_id") == tid]
    assert len(notices) == 1, live
    assert notices[0]["requested_executor_route"] == SELECTED
    assert notices[0]["parent_cognitive_route"] == COGNITION
    assert "work-account" not in json.dumps(notices[0]), "the row's account pin never travels with the selection"
    if unrecorded:  # The refusal's own progress row must suffice even while its exact authority is unreadable.
        from ouroboros.task_reads import TaskReads

        effective = TaskReads.effective
        monkeypatch.setattr(TaskReads, "effective", lambda reads, task_id: (
            inaccessible() if task_id == tid else effective(reads, task_id)))
        assert ("unreadable" in notices[0]["error"] or "scheduled_result_authority_unknown" in notices[0]["error"])
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "20"})))
    replays = [row for row in json.loads(response.body)["messages"] if row.get("subagent_task_id") == tid
               and row.get("subagent_event") == "rejected"]
    assert [row.get("requested_executor_route") for row in replays] == [SELECTED]
    assert replays[0]["parent_cognitive_route"] == COGNITION


@pytest.mark.parametrize(("model", "local", "cost", "billing", "price"), [
    ("openai/gpt-5", False, None, "unknown", None),
    ("openai/gpt-5", False, .125, "measured", .001),
    ("future-provider/reviewer", False, 0.0, "measured", None),
    ("local-model", True, None, "unknown", .001),
    ("claudexor::codex=gpt-5", False, None, "subscription", None),
])
def test_billing_basis_ignores_price_cache_and_requires_observed_cost(monkeypatch, price, model, local, cost, billing):
    consulted = []
    def estimate(*_args, **_kwargs):
        consulted.append(True)
        return price
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", estimate)
    route, actual = cognitive_route_billing_basis(model, use_local=local, cost=cost)
    assert route == model + (" (local)" if local else "") and actual == billing
    # A cumulative total names no accepted call: whatever it sums, it labels no route.
    ctx = SimpleNamespace(_accumulated_usage={"cost": cost, "cost_final": True,
                                              "_last_llm_call_meta": {"model": model}})
    assert cognitive_route_billing_basis(model, use_local=local, ctx=ctx) == (
        route, "subscription" if billing == "subscription" else "unknown")
    assert not consulted, "a billing fact cannot depend on the cache or pricing service"


def _call(usage, tmp_path, monkeypatch, model, *, cost=None, final=None, estimate=None, served="", round_idx=1,
          content="usable", use_local=False, error=None):
    """One call through the loop's own producer; ``usage`` then holds the records the billing label reads.

    The route is a synthetic object and an unpriced reply meets a stubbed tariff: no catalog is read."""
    from ouroboros.loop_llm_call import call_llm_with_retry

    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", lambda *_args, **_kwargs: estimate)

    class Route:
        def chat(self, **_kwargs):
            if error is not None:
                raise error
            return {"content": content}, {"prompt_tokens": 9, "completion_tokens": 2, "cost": cost,
                                           **({} if final is None else {"cost_final": final}),
                                           "provider": "openai-compatible", "resolved_model": served or model}

    return call_llm_with_retry(Route(), [{"role": "user", "content": "task"}], model, None, "medium", 1,
                               tmp_path / "logs", "billing-basis", round_idx, None, usage, "task", use_local,
                               attempt_cap=1)


@pytest.mark.parametrize(("cost", "final", "estimate", "billing"), [
    (0.125, True, None, "measured"),
    (0.0, True, None, "measured"),  # a legitimate reported zero
    (0.125, False, None, "unknown"),  # an amount its producer did not call final
    (None, None, None, "unknown"),  # unpriced
    (None, None, 0.001, "unknown"),  # a tariff estimate
    ("bogus", True, None, "unknown"),  # an invalid amount
])
def test_the_label_is_the_last_accepted_calls_own_money_whatever_the_history(
        tmp_path, monkeypatch, cost, final, estimate, billing):
    """The nanny route says measured only for money its own last accepted call reported.

    The history before it would decide the other way if the cumulative labelled the route: an unpriced call
    on another route before reported money, exact money on this route before a call that reports none."""
    usage = {}
    if billing == "measured":
        _call(usage, tmp_path, monkeypatch, "other/route")
        assert (usage["cost_partial"], usage["cost_final"]) == (True, False), "the cumulative is unknown from here on"
    else:
        _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=0.125, final=True)
    _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=cost, final=final, estimate=estimate, round_idx=2)
    ctx = SimpleNamespace(_accumulated_usage=usage)
    consulted = []
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", lambda *_args, **_kwargs: consulted.append(True))
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", billing)
    assert cognitive_route_billing_basis("other/route", ctx=ctx) == ("other/route", "unknown")
    assert not consulted, "the label reads the recorded call, never a price"


@pytest.mark.parametrize(("accepted", "local", "asked", "asked_local"), [
    ("openai/gpt-5", False, "openai/gpt-5.1", False),  # another model
    ("openai/gpt-5", False, "openai::gpt-5", False),  # another provider's spelling of the same served name
    ("openai/gpt-5", False, "served/gpt-5", False),  # the served name itself is not the route that was asked
    ("qwen/qwen3-coder", False, "qwen/qwen3-coder", True),  # the same id on another locality
    ("qwen/qwen3-coder", True, "qwen/qwen3-coder", False),
])
def test_an_accepted_call_labels_only_the_route_it_was_requested_on(
        tmp_path, monkeypatch, accepted, local, asked, asked_local):
    """Exact money names its own route: the configured id and locality its call record requested, no alias."""
    usage = {}
    _call(usage, tmp_path, monkeypatch, accepted, cost=0.125, final=True, served="served/gpt-5", use_local=local)
    ctx = SimpleNamespace(_accumulated_usage=usage)
    label = accepted + (" (local)" if local else "")
    assert cognitive_route_billing_basis(accepted, use_local=local, ctx=ctx) == (label, "measured")
    assert cognitive_route_billing_basis(label, ctx=ctx) == (label, "measured"), "a display suffix is locality"
    assert cognitive_route_billing_basis(asked, use_local=asked_local, ctx=ctx)[1] == "unknown"


def test_a_reply_without_a_usable_answer_accepts_nothing_for_its_route(tmp_path, monkeypatch):
    """Such a reply moves the last call record and writes no accepted call: neither route is then labelled."""
    usage = {}
    _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=0.125, final=True)
    accepted = dict(usage["_last_round_cache_usage"])
    reply, _cost = _call(usage, tmp_path, monkeypatch, "fallback/route", cost=0.25, final=True, round_idx=2, content="")
    assert reply is None and usage["_last_round_cache_usage"] == accepted
    assert usage["_last_llm_call_meta"]["model"] == "fallback/route" and usage["_last_llm_call_meta"]["failure_code"]
    assert usage["cost"] == 0.375 and usage.get("cost_final") is not False, (
        "the cumulative still reads as reported money")
    ctx = SimpleNamespace(_accumulated_usage=usage)
    assert cognitive_route_billing_basis("fallback/route", ctx=ctx) == ("fallback/route", "unknown")
    # The earlier accepted call is no longer the one the last call record names: unknown until the next accepted call.
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", "unknown")
    _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=0.0, final=True, round_idx=3)
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", "measured")


def test_a_send_that_raised_leaves_the_last_accepted_call_the_fact(tmp_path, monkeypatch):
    """No reply came back: the call record and the accepted call both stay the earlier call's."""
    usage = {}
    _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=0.0, final=True)
    records = (dict(usage["_last_llm_call_meta"]), dict(usage["_last_round_cache_usage"]))
    reply, _cost = _call(usage, tmp_path, monkeypatch, "fallback/route", round_idx=2, error=RuntimeError("no reply"))
    assert reply is None and usage["_last_llm_failure"]["model"] == "fallback/route"
    assert (usage["_last_llm_call_meta"], usage["_last_round_cache_usage"]) == records
    ctx = SimpleNamespace(_accumulated_usage=usage)
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", "measured")
    assert cognitive_route_billing_basis("fallback/route", ctx=ctx) == ("fallback/route", "unknown")


def test_explicit_cost_and_subscription_keep_their_kind():
    """The caller's own amount is taken as given; a subscription route stays a subscription."""
    assert cognitive_route_billing_basis("openai/gpt-5", cost=0.0) == ("openai/gpt-5", "measured")
    assert cognitive_route_billing_basis("openai/gpt-5", cost=None) == ("openai/gpt-5", "unknown")
    assert cognitive_route_billing_basis("qwen/qwen3-coder", use_local=True, cost=0.0) == (
        "qwen/qwen3-coder (local)", "measured")
    assert cognitive_route_billing_basis("claudexor::codex=gpt-5", cost=0.0) == (
        "claudexor::codex=gpt-5", "subscription")
    zero = SimpleNamespace(_accumulated_usage={"cost": 0.0, "cost_final": True,
                                              "_last_llm_call_meta": {"model": "claudexor::codex=gpt-5"}})
    assert cognitive_route_billing_basis("claudexor::codex=gpt-5", ctx=zero) == (
        "claudexor::codex=gpt-5", "subscription")
