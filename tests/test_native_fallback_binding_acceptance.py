"""Independent #1407/#1409 native request proofs, proposed for the frozen candidate.

Uses existing main_call/setup fixtures and the real loop, rebind, LLMClient,
Claudexor serializer and physical ledger. No provider calls. Not run by the scout.
"""

from copy import deepcopy
from dataclasses import replace
import json
import queue
from types import SimpleNamespace

import pytest
import httpx

from ouroboros import context, fallback_cooldown, llm_claudexor, loop
from ouroboros import loop_transport, model_wait, working_checkpoint
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY, TRANSPORT_DEATHS_KEY, call_llm_with_retry
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.model_wait import ModelWaitInterrupted
from ouroboros.task_results import write_task_result
from ouroboros.tools.control_runtime import _switch_model
from tests._usage_store_testing import attempt_rows_in_start_order
from tests._working_checkpoint import checkpoint_registry
from tests._llm_claudexor_fixtures import MODEL, REF, ROUTE, result
from tests._usage_store_testing import ledger_rows as ledger
from tests._forced_model_binding_fixtures import acting as forced_rail_fixture
from tests._forced_model_binding_fixtures import CARRIED_WORDS
from tests._forced_model_binding_fixtures import carried_quote as _carried_quote
from tests._forced_model_binding_fixtures import carry_to_work as _carry_to_work
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_model_wait_controls import _loop_tools
from tests.test_subscription_main_wait import main_call as main_call_fixture
from tests._cases import case_folders, chosen_rows

setup = gateway_fixture
live_wait = wait_fixture
main_call = main_call_fixture
acting = forced_rail_fixture
ROUTE_B = {**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"}
ROUTE_C = {**ROUTE, "credentialProfileId": "account-c", "accountFingerprint": "fingerprint-c"}


def _reply(route, message):
    row = result(route=route)
    row["message"] = message
    return row


def _bounded_creates(monkeypatch, gateway, maximum):
    create = gateway.create_model_operation

    def checked(*args, **kwargs):
        if len(gateway.creates) >= maximum:
            pytest.fail("unexpected extra model operation; the test must not retry into success")
        return create(*args, **kwargs)

    monkeypatch.setattr(gateway, "create_model_operation", checked)


@pytest.mark.parametrize(("primary_account", "role", "direct"), [
    pytest.param("account-a", "main", False, id='pinned-A-main-False'),
    pytest.param("account-a", "consciousness", True, id='pinned-A-consciousness-True'),
    pytest.param("", "main", True, id='Auto-main-True'),
])
def test_native_loop_fallback_then_symbolic_return_uses_real_role_account_and_fit(
        main_call, monkeypatch, role, direct, primary_account):
    """A -> B -> primary: equal model names do not make equal account/role bindings.

    The fallback MODEL is deliberately identical to the primary MODEL. A wrong
    fallback:0 role on return would send pin B, and a global-Main return from
    consciousness would send the decoy model/account. Both fail physical assertions.
    """
    ctx, gateway, owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "4")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    monkeypatch.setenv("OUROBOROS_MODEL", MODEL if role == "main" else "claudexor::codex=decoy-main")
    monkeypatch.setenv("OUROBOROS_MODEL_CONSCIOUSNESS", MODEL)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.delenv("USE_LOCAL_MAIN", raising=False)
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    pins = {"main": "decoy-main-pin", "consciousness": "decoy-consciousness-pin", "fallback": ["account-b"]}
    pins[role] = primary_account
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps(pins))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    # The primary's own lost answer takes one upstream observation first; a miss walks the configured routes.
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *_args: False)

    # Metadata is the only route observation stub. _rebind_context_fit_plan,
    # task_model_binding and every physical preparation/send stay real.
    def route_evidence(task, **_kwargs):
        actual_role = task["model_role"]
        assert actual_role in (role, "fallback:0")
        assert task["model"] == MODEL and not task["use_local_model"]
        observed = task.get("model_route") or {}
        explicit = task.get("credential_profile_id")
        configured = "account-b" if actual_role == "fallback:0" else primary_account
        profile = (explicit if explicit is not None else configured) or observed.get("credentialProfileId") or "account-c"
        window = 240_000 if profile == "account-b" else 900_000
        return ({"model": MODEL, "provider": "claudexor"}, SimpleNamespace(
            route_fp=f"capacity-{profile}", status="confirmed", stale=False, window_tokens=window,
            source_id="codex", source="advertised", credential_profile_id=profile,
            account_fingerprint=f"fingerprint-{profile.removeprefix('account-')}"))

    monkeypatch.setattr(context, "_context_fit_route", route_evidence)
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model_role=role, model_route=deepcopy(ROUTE))
    tools = _loop_tools(ctx, owner)
    tools._ctx.task_metadata = {"model_role": role}
    tools._ctx.is_direct_chat = direct
    tools._ctx.model_wait_context = owner
    owner.worker_slot_held = not direct
    original_tools = deepcopy([row for row in ctx.messages if row.get("role") == "tool"])
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
    ctx.messages.append({"role": "user", "content": [{"type": "text", "text": "Keep the diagram"}, image]})
    return_call = {"id": "return-primary", "type": "function", "function": {
        "name": "switch_model", "arguments": json.dumps({"primary": "return"})}}
    answered_route = ROUTE if primary_account else ROUTE_C
    gateway.results = [
        result(outcome="unknown", route=ROUTE),
        _reply(ROUTE_B, {"role": "assistant", "content": "Returning to the original role.",
                         "tool_calls": [return_call]}),
        _reply(answered_route, {"role": "assistant", "content": "Primary answered after return."}),
    ]
    gateway.dispatch = ["unknown", "response_received", "response_received"]
    _bounded_creates(monkeypatch, gateway, 3)

    text, usage, trace = loop.run_llm_loop(
        ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None,
        queue.Queue(), task_id="task-one", drive_root=ctx.drive_root, event_queue=events,
        initial_effort="high",
    )

    assert text == "Primary answered after return."
    assert len(gateway.uploads) == len(gateway.creates) == len(gateway.accepted_operations) == 3
    payloads = [payload for payload, _key in gateway.uploads]
    assert all(payload["model"] == "exact-model" for payload in payloads)
    assert payloads[1]["account"] == {"mode": "pin", "profileId": "account-b"}
    for index in (0, 2):
        account = payloads[index]["account"]
        if primary_account:
            assert account == {"mode": "pin", "profileId": "account-a"}
        else:
            assert account["mode"] == "auto" and "profileId" not in account
    # A returned Auto account is observation, not permission to convert Auto to pin C.
    assert tools._ctx.context_fit_plan.model_role == role
    assert tools._ctx.context_fit_plan.model_route["credentialProfileId"] == answered_route["credentialProfileId"]
    assert tools._ctx.primary_route == {"model": MODEL, "use_local": False, "role": role}
    request_manifests = [json.loads((ctx.drive_root / "observability" / "calls" / "task-one" /
                         f"{attempt}_model_request.json").read_text(encoding="utf-8"))
                         for attempt in gateway.creates]
    assert [manifest["model_role"] for manifest in request_manifests] == [role, "fallback:0", role]
    # One current row per attempt, in start order; each keeps its dispatch context.
    rows = attempt_rows_in_start_order(ctx.drive_root)
    assert [(row["state"], row["revision"]) for row in rows] == [("unresolved", 4), ("settled", 3), ("settled", 3)]
    assert rows[0]["physical_failure"]["stage"] == "raised_exception"
    dispatched = rows
    assert len({row["attempt_id"] for row in dispatched}) == 3
    # Each attempt's frame: the line its request is admitted at (the stated total less the reply floor) plus the reserve.
    assert [row["physical_context"]["capacity_total_tokens"] for row in dispatched] == [
        window - 8_192 + 65_536 for window in (900_000, 240_000, 900_000)]
    assert [row["physical_context"]["route_fp"] for row in dispatched] == [
        "capacity-account-a", "capacity-account-b", f"capacity-{answered_route['credentialProfileId']}"]
    assert usage["transport_recovery"]["previous_attempt"]["physical_attempt_id"] == dispatched[0]["attempt_id"]
    for payload in payloads:
        sent = payload["messages"]
        for completed in original_tools:
            matches = [row for row in sent if row.get("role") == "tool" and
                       row.get("tool_call_id") == completed["tool_call_id"]]
            assert len(matches) == 1 and matches[0]["content"] == completed["content"]
        assert any(image in row.get("content", []) for row in sent if isinstance(row.get("content"), list))
    returned_tool = [row for row in payloads[2]["messages"] if row.get("role") == "tool" and
                     row.get("tool_call_id") == "return-primary"]
    assert len(returned_tool) == 1 and "OK: switching to primary route" in returned_tool[0]["content"]
    assert [call["tool"] for call in trace["tool_calls"]] == ["switch_model"]


def _rejoin_after_404(monkeypatch, gateway, *, missing_reads):
    gateway.pending = True
    original_get = gateway.get_model_operation
    observed_reads = []

    def read(operation_id, **kwargs):
        observed_reads.append(operation_id)
        if len(observed_reads) <= missing_reads:
            raise ClaudexorUnavailable("http_404", "accepted operation temporarily not readable", status_code=404)
        gateway.pending = False
        return original_get(operation_id, **kwargs)

    monkeypatch.setattr(gateway, "get_model_operation", read)
    now = [0.0]
    monkeypatch.setattr(llm_claudexor, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    return observed_reads


@pytest.mark.parametrize("direct", [False, True], ids=["queued", "direct"])
def test_accepted_get_404_recovers_same_native_operation_for_both_turns(live_wait, monkeypatch, direct):
    root, gateway, client, owner, _events, _decide = live_wait
    owner.tool_context = SimpleNamespace(task_id="task-one", is_direct_chat=direct, task_metadata={})
    owner.worker_slot_held = not direct
    observed_reads = _rejoin_after_404(monkeypatch, gateway, missing_reads=2)
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    _bounded_creates(monkeypatch, gateway, 1)
    message, usage = client.chat([{"role": "user", "content": "recover this response"}], MODEL,
                                 model_role="main", timeout=1)
    assert message == result()["message"]
    assert observed_reads == ["op-0"] * 3
    assert len(gateway.uploads) == len(gateway.creates) == len(gateway.accepted_operations) == 1
    assert [(row["state"], row["revision"]) for row in ledger(root)] == [("settled", 3)]
    assert len(usage["ledger_attempt_ids"]) == 1 and not gateway.cancels
    events = [json.loads(line) for line in (root / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event.get("detail") == "same_model_operation_rejoined" for event in events)


def _saved_native_round(setup, monkeypatch, shape):
    """Lose the first worker after a real reservation/create, then restore through the loop entry point."""
    root, gateway, client = setup
    task_id = "task-one"
    monkeypatch.setenv("OUROBOROS_MODEL", MODEL)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::unused")
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_a, **_kw: pytest.fail("restore cannot open a new-generation wait"))
    dead = checkpoint_registry(root, monkeypatch, task_id, 1)
    dead._ctx.active_model = MODEL
    messages = [{"role": "user", "content": "Continue the recorded task."}]
    usage = {"execution_id": "original"}
    checkpoint = SimpleNamespace(tools=dead, task_id=task_id, messages=messages, accumulated_usage=usage,
        llm_trace={"tool_calls": []}, round_idx=1, tool_schemas=[], owner_msg_seen=set(), context_fit_plan=dead._ctx.context_fit_plan)
    write_task_result(root, task_id, "running")
    if shape == "sidecar":
        assert working_checkpoint.save_round(checkpoint, "ready")
    gateway.results, gateway.dispatch = [result(outcome="unknown")], ["unknown"]
    with monkeypatch.context() as initial:
        if shape in {"sidecar", "pending_readable", "unmarked", "timeout_repeat"}:
            # An accepted create whose GET fails leaves a rejoinable receipt.
            initial.setattr(gateway, "pending", True)
            initial.setattr(llm_claudexor._ModelInvocation, "_control_outage", lambda *_: False)

            def missing(operation_id, **_):
                assert operation_id == gateway.detail(0)["id"]
                raise ClaudexorUnavailable("http_404", "accepted operation temporarily unreadable", status_code=404)

            initial.setattr(gateway, "get_model_operation", missing)
        elif shape == "no_operation_id":
            create = gateway.create_model_operation

            def lose_receipt(*args, **kwargs):
                create(*args, **kwargs)
                raise ClaudexorUnavailable("http_404", "Lost the accepted create reply.", status_code=404)

            initial.setattr(gateway, "create_model_operation", lose_receipt)
        assert call_llm_with_retry(client, messages, MODEL, [], "low", 1, root / "logs", task_id, 1, None,
                                   usage, checkpoint_call=checkpoint if shape == "sidecar" else None)[0] is None
    assert len(gateway.creates) == 1
    first, = ledger(root)
    pending = usage["_pending_transport_outcome"]
    assert pending["physical_attempt_id"] == first["attempt_id"] and first["state"] == "unresolved"
    assert pending["operation_id"] == ("" if shape == "no_operation_id" else gateway.detail(0)["id"])
    if shape == "count_only":
        del usage["_pending_transport_outcome"]  # 60a17a0a535 predates the independent pending field
    elif shape == "unmarked":
        assert pending.pop("same_operation_recoverable") is True
    elif shape == "pending_unknown":
        assert "same_operation_recoverable" not in pending  # Its terminal wire body actually reported unknown.
    elif shape == "timeout_repeat":
        # The repeat claim saves this physical key before create; a later non-start cannot renew the grant.
        usage[TIMEOUT_REPEAT_KEY] = {"state": "dispatched", "round_id": "original:round:1",
            "repeat": {"round": 1, "execution_id": "original", "round_id": "original:round:1",
                       "llm_call_id": usage["_last_llm_failure"]["llm_call_id"], "physical_attempt_id": first["attempt_id"],
                       "attempt_custody_state": "reserved", "model": MODEL}}
    if shape != "sidecar":
        if shape not in {"unmarked", "timeout_repeat"}:
            usage[TRANSPORT_DEATHS_KEY] = {"round_id": "original:round:1", "count": 1}
        assert working_checkpoint.save_round(checkpoint, "ready")
    _, saved = working_checkpoint.read_working(root, task_id, 1)
    if shape == "sidecar":
        assert saved["working"]["native_attempt"]["physical_attempt_id"] == first["attempt_id"]
        assert "_pending_transport_outcome" not in saved["usage"]
    task = {"id": task_id}
    assert working_checkpoint.attach_recovery(root, task, source_task_id=task_id, from_attempt=1, cause="worker_crash")
    fresh = checkpoint_registry(root, monkeypatch, task_id, 2)
    fresh._ctx.working_recovery = task["_working_recovery"]
    remote = client._chat_remote

    def only_native(target, *args, **kwargs):
        if target["provider"] != "claudexor":
            pytest.fail("restore opened a paid fallback")
        return remote(target, *args, **kwargs)

    monkeypatch.setattr(client, "_chat_remote", only_native)
    monkeypatch.setattr(client, "_chat_local", lambda *_a, **_kw: pytest.fail("restore opened a local sender"))

    def resume():
        with model_wait.task_model_wait_scope(task={"id": task_id, "_attempt": 2}, drive_root=root,
                                             event_queue=None, worker_slot_held=True, owner_control=lambda: None) as owner:
            owner.tool_context = fresh._ctx
            return loop.run_llm_loop([], fresh, client, root / "logs", lambda *_a, **_kw: None,
                                     queue.Queue(), task_id=task_id, drive_root=root)

    return SimpleNamespace(root=root, gateway=gateway, ctx=fresh._ctx, first=first, saved=saved, resume=resume)


@pytest.mark.parametrize("ending", ["answer", "bad_request", "lost", "rejoin"])
def test_old_ready_round_sends_once_and_never_retries_its_unknown(setup, monkeypatch, ending):
    saved = _saved_native_round(setup, monkeypatch, "count_only")
    gateway = saved.gateway
    reply = result(outcome="unknown" if ending == "lost" else "completed")
    reply["message"] = {"role": "assistant", "content": "Restored round answered."}
    if ending == "bad_request":
        reply = result(outcome="failed", problem={"code": "invalid_request", "message": "Invalid argument",
                                                   "context": {"httpStatus": 400}})
    gateway.results.append(reply)
    gateway.dispatch.append("unknown" if ending == "lost" else "response_received")
    _bounded_creates(monkeypatch, gateway, 2)
    if ending == "rejoin":
        reads = _rejoin_after_404(monkeypatch, gateway, missing_reads=1)
    text, after, _ = saved.resume()
    if ending == "rejoin":
        assert reads == [gateway.detail(1)["id"]] * 2
    assert len(gateway.creates) == len(gateway.accepted_operations) == len(gateway.uploads) == 2
    old, new = ledger(saved.root)
    assert old["attempt_id"] == saved.first["attempt_id"] != new["attempt_id"]
    assert old["state"] == "unresolved" and old.get("cost_usd") is None
    assert new["state"] == ("unresolved" if ending == "lost" else "settled")
    if ending in {"answer", "rejoin"}:
        assert text == "Restored round answered." and TRANSPORT_DEATHS_KEY not in after
    else:
        assert after[TRANSPORT_DEATHS_KEY]["count"] == 1
        if ending == "lost":
            assert after["_pending_transport_outcome"]["physical_attempt_id"] == new["attempt_id"]
        else:
            assert not after.get("_pending_transport_outcome")


@pytest.mark.parametrize("shape", ["pending_unknown", "no_operation_id"])
def test_identified_unknown_stays_with_its_original_attempt(setup, monkeypatch, shape):
    saved = _saved_native_round(setup, monkeypatch, shape)
    gateway = saved.gateway
    _bounded_creates(monkeypatch, gateway, 1)
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: pytest.fail("no outstanding readable result"))
    _, usage, trace = saved.resume()
    only, = ledger(saved.root)
    pending = usage["_pending_transport_outcome"]
    assert pending["physical_attempt_id"] == only["attempt_id"] == saved.first["attempt_id"]
    assert pending["operation_id"] == ("" if shape == "no_operation_id" else "op-0")
    assert only["state"] == "unresolved" and only.get("cost_usd") is None
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert len(gateway.creates) == len(gateway.uploads) == 1 and not gateway.reads and not gateway.acks


def test_unmarked_pending_requires_its_own_saved_round_and_physical_attempt(setup, monkeypatch):
    saved = _saved_native_round(setup, monkeypatch, "unmarked")
    state, task_id = saved.saved, saved.ctx.task_id
    usage = state["usage"]
    failure, pending = usage["_last_llm_failure"], usage["_pending_transport_outcome"]

    def bind(**overrides):
        return loop_transport.ready_operation(state, {**deepcopy(usage), **overrides}, saved.root, task_id)

    assert bind()["operation_id"] == "op-0"
    assert bind(_last_llm_failure={**failure, "llm_call_id": None})["operation_id"] == "op-0"
    assert bind(_last_llm_failure={**failure, "round": 0}) is None
    assert bind(_last_llm_failure={**failure, "llm_call_id": "other-call"}) is None
    assert bind(_last_llm_failure={**failure, "physical_attempt_id": "other-attempt"}) is None
    assert bind(_last_llm_failure={**failure, "model": "different-model"}) is None
    assert bind(execution_id="other-execution") is None
    assert bind(_pending_transport_outcome={**pending, "model": "other-model"}) is None
    assert bind(_pending_transport_outcome={**pending, "operation_id": "other-operation"})["operation_id"] == ""
    foreign = {**state, "task_id": "other-task"}
    assert loop_transport.ready_operation(foreign, deepcopy(usage), saved.root, "other-task") is None
    assert not saved.gateway.reads and not saved.gateway.acks


@pytest.mark.parametrize(("shape", "ending"), [
    ("unmarked", "answer"), ("pending_readable", "answer"), ("sidecar", "answer"),
    ("sidecar", "stop"), ("unmarked", "deadline"),
    ("unmarked", "refusal"), ("pending_readable", "lost"), ("sidecar", "foreign-id"),
    ("sidecar", "existing-source"),
])
def test_cold_restore_waits_for_the_same_operation(setup, monkeypatch, shape, ending):
    from ouroboros import owner_mailbox
    from ouroboros.observability import persist_call, read_call_payload
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from ouroboros.review_evidence import _commit_source_payload
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    saved = _saved_native_round(setup, monkeypatch, shape)
    gateway, attempt = saved.gateway, saved.first["attempt_id"]
    _bounded_creates(monkeypatch, gateway, 1)
    reply = result(cash=0.17, knowledge="exact")
    reply["message"] = {"role": "assistant", "content": "The recovered answer."}
    if ending in {"answer", "existing-source"}:
        reply["failureEvidence"] = {"note": "fixture-only-private-source"}
    if ending == "existing-source":
        logical_call = saved.saved["working"]["native_attempt"]["llm_call_id"]
        earlier = persist_call(saved.root, task_id=saved.ctx.task_id, call_id=f"{logical_call}_response",
            call_type="llm_response", payload={"message": {"content": "Earlier sample."}, "usage": {}})
    if ending == "refusal":
        reply = result(outcome="failed", problem={"code": "provider_content_refused", "retryable": False,
                                                   "message": "Synthetic refusal"})
    elif ending == "lost":
        reply = result(outcome="unknown")
    gateway.results = [reply]
    gateway.dispatch = ["unknown" if ending == "lost" else "response_received"]
    original_get = gateway.get_model_operation

    def read(operation_id, **kwargs):
        assert operation_id == "op-0"
        assert saved.ctx._accumulated_usage.get("execution_status") != "infra_failed"
        poll = len(gateway.reads) + 1
        if poll > 3 or (poll > 2 and ending in {"stop", "deadline"}):
            pytest.fail("restore kept polling past the scripted terminal or control")
        gateway.pending = poll < 3
        detail = original_get(operation_id, **kwargs)
        if poll == 1:
            detail["state"], detail["dispatch"]["state"] = "queued", "not_started"
        elif poll == 2 and ending == "stop":
            owner_mailbox.write_owner_message(saved.root, REASON_OWNER_STOPPED_DIRECT_TURN, saved.ctx.task_id,
                                              msg_id="stop", kind=owner_mailbox.KIND_FINALIZE_NOW)
        elif poll == 2 and ending == "deadline":
            saved.ctx.task_metadata["deadline_at"] = "2000-01-01T00:00:00+00:00"
        elif poll == 3 and ending == "foreign-id":
            detail["id"] = "another-operation"
        return detail

    monkeypatch.setattr(gateway, "get_model_operation", read)
    text, usage, _ = saved.resume()
    only, = ledger(saved.root)
    controlled = ending in {"stop", "deadline"}
    assert gateway.creates == [attempt] and len(gateway.uploads) == 1
    assert gateway.reads == ["op-0"] * (2 if controlled else 3)
    assert gateway.cancels == ([("op-0", "host_cancelled")] if controlled else [])
    assert only["attempt_id"] == attempt and usage["ledger_attempt_ids"] == [attempt]
    if controlled or ending in {"foreign-id", "lost"}:
        assert only["state"] == "unresolved" and only.get("cost_usd") is None
        assert usage["_pending_transport_outcome"]["physical_attempt_id"] == attempt
        assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
        if ending == "stop":
            assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION and saved.ctx._skip_post_task_synthesis
        elif ending == "deadline":
            assert usage["reason_code"] == "deadline_local"
    else:
        assert only["state"] == "settled" and "_pending_transport_outcome" not in usage
        if ending in {"answer", "existing-source"}:
            assert text == "The recovered answer." and only["cost_usd"] == usage["cost"] == 0.17
            meta = usage["_last_llm_call_meta"]
            assert meta["physical_attempt_id"] == attempt and usage["completion_tokens"] == 7
            logical = _commit_source_payload(saved.ctx, meta["response_ref"], manifest=True)
            assert logical["message"] == reply["message"] and "fixture-only-private-source" not in json.dumps(logical)
            raw_manifest, source, physical = read_call_payload(saved.root, task_id=saved.ctx.task_id, call_id=f"{attempt}_model_response")
            assert raw_manifest["call_type"] == "llm_claudexor_response" and json.loads(source["result_json_utf8"]) == reply
            assert logical["usage"]["claudexor"]["result_custody"]["retained_manifest_ref"] == physical["manifest_ref"] != meta["response_ref"]
            if ending == "existing-source":
                previous = _commit_source_payload(saved.ctx, earlier["manifest_ref"], manifest=True)
                assert previous["message"]["content"] == "Earlier sample." and earlier["manifest_ref"] != meta["response_ref"]
        else:
            assert text == llm_claudexor.ClaudexorContentRefused.notice and usage["reason_code"] == "provider_content_refused"
    assert gateway.acks == ([] if controlled or ending == "foreign-id" else [("op-0", REF["sha256"])])


def test_cold_restore_leaves_original_operation_owed_when_the_store_changes(setup, monkeypatch):
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway = saved.gateway
    replacement = type(gateway)()
    replacement.store_identity = "empty-fixture-store"
    pending = loop_transport.ready_operation(saved.saved, deepcopy(saved.saved["usage"]), saved.root, saved.ctx.task_id)
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: replacement)
    before = ledger(saved.root)
    with pytest.raises(llm_claudexor.ClaudexorModelError) as held:
        llm_claudexor.recover_chat_reply(saved.root, saved.ctx.task_id, pending, None)
    assert held.value.code == "model_outcome_unknown" and held.value.same_operation_recoverable
    assert ledger(saved.root) == before and before[0]["state"] == "unresolved"
    assert not (replacement.reads or replacement.creates or replacement.acks or replacement.cancels)
    assert replacement.closed == 1

    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    gateway.results = [result(cash=0.17, knowledge="exact")]
    gateway.results[0]["message"] = {"role": "assistant", "content": "Original store recovered."}
    gateway.dispatch = ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    text, usage, _ = saved.resume()
    [row] = ledger(saved.root)
    assert text == "Original store recovered." and row["state"] == "settled"
    assert row["attempt_id"] == before[0]["attempt_id"] and usage["cost"] == 0.17
    assert gateway.reads == ["op-0"] and gateway.acks == [("op-0", REF["sha256"])]


@pytest.mark.parametrize("words", [False, True], ids=["unchanged-source", "owner-words-while-down"])
def test_a_restored_reply_finishes_only_the_owner_input_it_saw(setup, monkeypatch, words):
    """The round's real mailbox drain runs before the restored read: words it delivers were never seen by that send."""
    from ouroboros import owner_mailbox

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, attempt = saved.gateway, saved.first["attempt_id"]
    old, new = result(cash=0.17, knowledge="exact"), result(cash=0.05, knowledge="exact")
    old["message"] = {"role": "assistant", "content": "The recovered answer.", "tool_calls": result()["message"]["tool_calls"]} if words else {
        "role": "assistant", "content": "The recovered answer."}
    new["message"] = {"role": "assistant", "content": "The answer to the new words."}
    gateway.results, gateway.dispatch = [old, new], ["response_received", "response_received"]
    _bounded_creates(monkeypatch, gateway, 2 if words else 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("a stale reply's tools never run"))
    if words:
        owner_mailbox.write_owner_message(saved.root, "Please also check the second file.", saved.ctx.task_id, msg_id="late-words")
    text, usage, _ = saved.resume()

    first, *later = ledger(saved.root)
    assert first["attempt_id"] == attempt and first["state"] == "settled" and first["cost_usd"] == 0.17
    assert gateway.reads == ["op-0"] and gateway.creates[0] == attempt  # the old operation is read, never created again
    if not words:
        assert text == "The recovered answer." and not later and len(gateway.creates) == 1
        return
    second, = later
    assert second["attempt_id"] != attempt and gateway.creates == [attempt, second["attempt_id"]]
    assert text == "The answer to the new words." and usage["cost"] == pytest.approx(0.22)
    assert {attempt, second["attempt_id"]} <= set(usage["ledger_attempt_ids"])
    sent = gateway.uploads[1][0]["messages"]
    place = {name: next(i for i, row in enumerate(sent) if name in json.dumps(row.get("content")))
             for name in ("The recovered answer.", "Please also check the second file.")}
    assert sent[place["The recovered answer."]]["role"] == "assistant"
    assert place["The recovered answer."] < place["Please also check the second file."]  # its true place
    assert not any(row.get("role") == "tool" or row.get("tool_calls") for row in sent)  # words only: no stale tool effect


def test_retained_main_reply_belongs_to_this_calls_own_settled_attempt():
    from ouroboros import usage_accounting
    from ouroboros.loop_output_exhaustion import retain_interrupted_main_reply

    def interrupted(attempt):
        error = ModelWaitInterrupted("deadline", role="main")
        error.operation_id = "op"
        error.physical_attempt_capture = usage_accounting.PhysicalAttemptCapture(attempt, MODEL, "claudexor", "settled", "opaque")
        error.model_result = {"outcome": "completed", "message": {"role": "assistant", "content": "A reply."}}
        error.usage = {"claudexor": {"operation_id": "op", "model_role": "main", "outcome": "completed",
                                     "dispatch_state": "response_received"}}
        return error

    own, foreign = interrupted("this-attempt"), interrupted("an-earlier-attempt")  # both settled, one not this call's
    for error in (own, foreign):
        retain_interrupted_main_reply(error, ("this-attempt", "owner-source"))
    assert own.received_main_reply == ({"role": "assistant", "content": "A reply."}, own.usage, "owner-source")
    assert not hasattr(foreign, "received_main_reply")


def _during_final_receipt(monkeypatch, gateway, control):
    get_result = gateway.get_model_result

    def read(operation_id, **kwargs):
        body = get_result(operation_id, **kwargs)
        assert operation_id == "op-0"
        control()
        return body

    monkeypatch.setattr(gateway, "get_model_result", read)


def test_warm_paused_refusal_resumes_without_recounting(setup, monkeypatch):
    root, gateway, client = setup
    task_id, received = "task-one", []
    gateway.results = [{**result(outcome="failed", cash=0.17, knowledge="exact", problem={
        "code": "provider_content_refused", "retryable": False}), "message": None}]
    write_task_result(root, task_id, "running")
    _bounded_creates(monkeypatch, gateway, 1)
    _during_final_receipt(monkeypatch, gateway, lambda: received.append(True))
    total = {"execution_id": "original"}
    with model_wait.task_model_wait_scope(task={"id": task_id}, drive_root=root, event_queue=None, worker_slot_held=False,
                                         owner_control=lambda: "owner_pause" if received else None):
        with pytest.raises(ModelWaitInterrupted) as held:
            call_llm_with_retry(client, [{"role": "user", "content": "Sample note."}], MODEL, [], "low", 1,
                                root / "logs", task_id, 1, None, total)
    assert held.value.control_reason == "owner_pause" and isinstance(held.value.received_refusal, llm_claudexor.ClaudexorContentRefused)
    row, = ledger(root)
    state = {"task_id": task_id, "round_idx": 1, "resume_point": {"round_idx": 1}, "context_model_role": "main"}
    restored = deepcopy(total)
    pending = loop_transport.ready_operation(state, restored, root, task_id)
    assert (pending["physical_attempt_id"], pending["operation_id"]) == (row["attempt_id"], "op-0")
    ctx = SimpleNamespace(task_id=task_id, round_idx=1, accumulated_usage=restored, drive_logs=root / "logs",
                          status_drive_root=root, tools=checkpoint_registry(root, monkeypatch, task_id, 2))
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: pytest.fail("Resume must use the retained receipt"))
    with pytest.raises(llm_claudexor.ClaudexorContentRefused):
        loop_transport.read_ready_operation(ctx, pending)
    for usage in (total, restored):
        assert (usage["cost"], usage["prompt_tokens"], usage["completion_tokens"], usage["cached_tokens"]) == (0.17, 20, 7, 12)
        assert usage["ledger_attempt_ids"] == [row["attempt_id"]]
    assert "_pending_transport_outcome" not in restored and ledger(root) == [row]
    assert gateway.creates == [row["attempt_id"]] and gateway.acks == [("op-0", REF["sha256"])] and not gateway.reads


@pytest.mark.parametrize(("receipt", "control"), [
    ("model_substituted", "stop"), ("response_rejected", "deadline"),
    ("processing_unavailable", "stop"), ("not_started", "wrap"), ("tools", "stop"),
    ("model_substituted", "execution_deadline"),
])
def test_control_during_final_receipt_keeps_the_physical_result(setup, monkeypatch, receipt, control):
    from ouroboros import cancel_intents, owner_mailbox
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, owner_stop_control_id
    from tests._llm_claudexor_fixtures import processing_refusal as refusal

    reply = result(cash=0.17, knowledge="exact")
    if receipt == "model_substituted":
        reply["modelMismatch"] = {"requested": ROUTE["model"], "observed": "another-model"}
        reply["route"]["model"] = "another-model"
    elif receipt == "response_rejected":
        reply["message"] = {"role": "assistant", "content": " "}
    elif receipt == "processing_unavailable":
        monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "economy")
        monkeypatch.setattr(llm_claudexor, "current_engine", lambda: None)
        monkeypatch.setattr(llm_claudexor, "model_sources", lambda **_: {
            "sources": [{"id": "codex", "processingPreferences": ["economy", "standard"]}]})
        reply = refusal()
        reply["usage"] = {"input_tokens": 0, "output_tokens": 0}
        reply["cost"].update(knowledge="exact", cashUsd=0.0)
    elif receipt == "not_started":
        reply = result(outcome="failed", problem={"code": "model_request_invalid", "message": "Unsent"})
        reply["message"] = None
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, task_id = saved.gateway, saved.ctx.task_id
    if receipt == "processing_unavailable":
        assert saved.first["processing_preference"] == gateway.uploads[0][0]["options"]["processingPreference"] == "economy"
    gateway.results, gateway.dispatch = [reply], ["not_started" if receipt == "not_started" else "response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *a, **kw: pytest.fail("tool ran after control"))

    def arrive():
        if control == "deadline":
            saved.ctx.task_metadata["deadline_at"] = "2000-01-01T00:00:00+00:00"
        elif control == "execution_deadline":
            model_wait.current_model_wait().owner_control = lambda: "execution_deadline"
        elif control == "wrap":
            intent = cancel_intents.request_cancel(saved.root, task_id, requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
            owner_mailbox.write_owner_message(saved.root, REASON_OWNER_REQUESTED_FINALIZATION, task_id,
                                              msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
        else:
            owner_mailbox.write_owner_message(saved.root, REASON_OWNER_STOPPED_DIRECT_TURN, task_id,
                                              msg_id="stop", kind=owner_mailbox.KIND_FINALIZE_NOW)

    _during_final_receipt(monkeypatch, gateway, arrive)
    _, usage, trace = saved.resume()
    row, = ledger(saved.root)
    assert gateway.creates == usage["ledger_attempt_ids"] == [row["attempt_id"]] == [saved.first["attempt_id"]]
    assert len(gateway.uploads) == 1 and gateway.reads == ["op-0"] and not gateway.cancels
    assert gateway.acks == [("op-0", REF["sha256"])] and "_pending_transport_outcome" not in usage
    assert usage["reason_code"] == {"deadline": "deadline_local", "execution_deadline": "finalization_grace"}.get(
        control, REASON_OWNER_REQUESTED_FINALIZATION)
    if control == "stop":
        assert saved.ctx._skip_post_task_synthesis
    else:
        assert trace["forced_finalization"]["control_reason"] == ("finalize_requested" if control == "wrap" else control)
    if receipt in {"not_started", "processing_unavailable"}:
        assert row["state"] == "released"
        return
    assert row["state"] == "settled" and row["cost_usd"] == reply["cost"]["cashUsd"]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (reply["usage"]["input_tokens"], reply["usage"]["output_tokens"])
    if receipt in {"response_rejected", "tools", "model_substituted"}:
        assert (usage["cost"], usage["prompt_tokens"], usage["completion_tokens"]) == (0.17, 20, 7)
    if receipt in {"response_rejected", "tools"}:
        assert usage["_last_llm_call_meta"]["operation_id"] == "op-0"
    if receipt != "tools":
        failure = usage["_last_llm_failure"]
        assert (failure["recovered_operation_id"], failure["physical_attempt_id"]) == ("op-0", row["attempt_id"])
        assert failure["attempt_custody_state"] == "settled"
        if receipt != "response_rejected":
            assert failure["observed_route"] == reply["route"]
        assert failure["failure_code"] == receipt


def test_pause_during_final_receipt_resumes_the_settled_refusal_once(setup, monkeypatch):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.artifacts import read_actor_source_bytes

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    root, gateway, task_id = saved.root, saved.gateway, saved.ctx.task_id
    gateway.results = [result(outcome="failed", cash=0.17, knowledge="exact", problem={
        "code": "provider_content_refused", "retryable": False, "message": "Synthetic refusal"})]
    gateway.dispatch = ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    _during_final_receipt(monkeypatch, gateway, lambda: owner_pause.install_fence(root, task_id, request_id="late-pause"))
    try:
        # Only a read is owed, even when the original send window has closed.
        with model_wait.calendar_scope("2000-01-01T00:00:00+00:00", dispatch_only=True):
            with pytest.raises(budget_pause.BudgetPauseRequested) as held:
                saved.resume()
        pause = held.value.pause
        assert pause["rail"] == budget_pause.RAIL_OWNER_PAUSE
        row, = ledger(root)
        assert (row["attempt_id"], row["state"], row["cost_usd"]) == (saved.first["attempt_id"], "settled", 0.17)
        state = json.loads(read_actor_source_bytes(root, task_id, pause["source_ref"]))
        owed = loop_transport.ready_operation(state, deepcopy(state["usage"]), root, task_id)
        assert (owed["physical_attempt_id"], owed["operation_id"]) == (row["attempt_id"], "op-0")
        assert state["usage"].get("completion_tokens", 0) == 0

        grant = {"grant_id": "resume", "generation": 1, "owner_pause_fence_id": pause["owner_fence_id"]}
        budget_pause.set_budget_pause(root, task_id, {**pause, "state": budget_pause.STATE_RESUME_GRANTED, "grant": grant})
        saved.ctx.budget_pause_resume = {"pause_id": pause["pause_id"], "pause_generation": pause["pause_generation"],
                                        "grant_id": grant["grant_id"], "grant_generation": grant["generation"]}
        monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: pytest.fail("Resume must use the retained receipt"))
        text, usage, _ = saved.resume()
    finally:
        budget_pause.end_dispatch_fence(task_id)
    assert text == llm_claudexor.ClaudexorContentRefused.notice and usage["reason_code"] == "provider_content_refused"
    assert usage["_last_llm_failure"]["physical_attempt_id"] == row["attempt_id"]
    assert (usage["cost"], usage["completion_tokens"]) == (0.17, 7)
    assert ledger(root) == [row] and len(gateway.uploads) == 1 and gateway.creates == [row["attempt_id"]]
    assert gateway.reads == ["op-0"] and gateway.acks == [("op-0", REF["sha256"])] and not gateway.cancels


@pytest.mark.parametrize("already_released", [True, False])
@pytest.mark.parametrize("shape", ["sidecar", "timeout_repeat"])
def test_a_reasonless_unsent_receipt_restores_the_ordinary_round_without_a_success_receipt(setup, monkeypatch, shape, already_released):
    from ouroboros.transport_custody import ProviderNotDispatched

    saved = _saved_native_round(setup, monkeypatch, shape)
    gateway = saved.gateway
    _bounded_creates(monkeypatch, gateway, 1 if shape == "timeout_repeat" else 2)
    unsent = result(outcome="failed", problem=None)  # unsent and typed with no reason: nothing for a round's own call to meet
    unsent["message"] = None
    answer = result(cash=0.17, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "Fresh request answered."}
    gateway.results, gateway.dispatch = [unsent, answer], ["not_started", "response_received"]
    if already_released:
        bound = loop_transport.ready_operation(saved.saved, deepcopy(saved.saved["usage"]), saved.root, saved.ctx.task_id)
        with pytest.raises(ProviderNotDispatched):
            llm_claudexor.recover_chat_reply(saved.root, saved.ctx.task_id, bound, None)
        assert ledger(saved.root)[0]["state"] == "released"
    text, usage, trace = saved.resume()
    if shape == "timeout_repeat":
        only, = ledger(saved.root)
        assert only["state"] == "released" and usage["ledger_attempt_ids"] == [only["attempt_id"]]
        assert gateway.creates == [only["attempt_id"]] and len(gateway.uploads) == 1
        assert gateway.reads == ["op-0"] and gateway.acks == [("op-0", REF["sha256"])]
        assert usage["_last_llm_error_kind"] == "model_operation_not_started" and "_pending_transport_outcome" not in usage
        assert trace["forced_finalization"]["source"] == "timeout_repeat_spent_no_resend"
        assert not any(row.get("usable_solve_response") for row in usage.get("llm_call_refs", []))
        return
    first, second = ledger(saved.root)
    assert first["attempt_id"] == saved.first["attempt_id"] and first["state"] == "released"
    assert second["attempt_id"] != first["attempt_id"] and second["state"] == "settled"
    assert gateway.creates == [first["attempt_id"], second["attempt_id"]] and len(gateway.uploads) == 2
    assert first["attempt_id"] in usage["ledger_attempt_ids"]
    assert gateway.reads == ["op-0"] and [op for op, _digest in gateway.acks] == ["op-0", "op-1"]
    assert text == answer["message"]["content"] and usage["cost"] == second["cost_usd"] == 0.17
    assert usage["completion_tokens"] == 7 and "_pending_transport_outcome" not in usage
    assert [row["physical_attempt_id"] for row in usage["llm_call_refs"] if row.get("usable_solve_response")] == [second["attempt_id"]]


@pytest.mark.parametrize("case", ["standard", "already-released", "changed-tier", "unsupported", "unknown-dispatch"])
def test_cold_processing_refusal_releases_before_ordinary_standard_admission(setup, monkeypatch, case):
    from tests._llm_claudexor_fixtures import processing_receipt, processing_refusal

    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "economy")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", "{}")
    monkeypatch.setattr(llm_claudexor, "current_engine", lambda: None)
    preferences = ["economy", "fast", "standard"]
    monkeypatch.setattr(llm_claudexor, "model_sources", lambda **_: {
        "sources": [{"id": "codex", "processingPreferences": list(preferences)}]})
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway = saved.gateway
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    selected = "fast" if case == "changed-tier" else "standard"
    mode = "priority" if selected == "fast" else "default"
    answer = {**result(cash=0.17, knowledge="exact"), "processing": processing_receipt(selected, mode, selected),
              "message": {"role": "assistant", "content": "The next attempt answered."}}
    refusal = processing_refusal()
    gateway.results = [refusal, answer]
    no_send = case in {"unsupported", "unknown-dispatch"}
    gateway.dispatch = ["unknown" if case == "unknown-dispatch" else "response_received", "response_received"]
    _bounded_creates(monkeypatch, gateway, 1 if no_send else 2)
    if case == "already-released":
        pending = loop_transport.ready_operation(saved.saved, deepcopy(saved.saved["usage"]), saved.root, saved.ctx.task_id)
        with pytest.raises(llm_claudexor.ClaudexorModelNotDispatched):
            llm_claudexor.recover_chat_reply(saved.root, saved.ctx.task_id, pending, None)
        assert ledger(saved.root)[0]["state"] == "released"
        monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: pytest.fail("released receipt is retained"))
    elif case == "changed-tier":
        monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"main":"fast"}')
    elif case == "unsupported":
        preferences.remove("standard")

    text, usage, _ = saved.resume()
    first, *later = ledger(saved.root)
    assert first["attempt_id"] == saved.first["attempt_id"] and first["processing_preference"] == "economy"
    assert first["state"] == ("unresolved" if case == "unknown-dispatch" else "released")
    assert gateway.reads == ["op-0"] and gateway.uploads[0][0]["options"]["processingPreference"] == "economy"
    if no_send:
        assert not later and len(gateway.creates) == len(gateway.uploads) == 1
        return
    [second] = later
    assert text == answer["message"]["content"] and usage["cost"] == second["cost_usd"] == 0.17
    assert gateway.creates == [first["attempt_id"], second["attempt_id"]]
    assert [operation for operation, _ in gateway.acks] == ["op-0", "op-1"]
    assert second["state"] == "settled" and second["submitted_processing_mode"] == selected
    assert second["processing_preference"] == ("fast" if case == "changed-tier" else "economy")
    assert gateway.uploads[1][0]["options"]["processingPreference"] == selected


def _ready_native_call(setup, monkeypatch, task_id="task-one"):
    """A real native Main send bound to its loop's checkpoint writer, as a round makes it."""
    root, gateway, client = setup
    monkeypatch.setenv("OUROBOROS_MODEL", MODEL)
    registry = checkpoint_registry(root, monkeypatch, task_id, 1)
    registry._ctx.active_model = MODEL
    messages, usage = [{"role": "user", "content": "Continue the recorded task."}], {"execution_id": "original"}
    checkpoint = SimpleNamespace(tools=registry, task_id=task_id, messages=messages, accumulated_usage=usage,
        llm_trace={"tool_calls": []}, round_idx=1, tool_schemas=[], owner_msg_seen=set(),
        context_fit_plan=registry._ctx.context_fit_plan)
    write_task_result(root, task_id, "running")

    def send():
        with working_checkpoint.model_call_scope(checkpoint):
            return call_llm_with_retry(client, messages, MODEL, [], "low", 1, root / "logs", task_id, 1, None, usage,
                                       checkpoint_call=checkpoint)

    address = working_checkpoint.checkpoint_path(root, task_id, 1).with_suffix(".operation.json")
    return SimpleNamespace(root=root, gateway=gateway, ctx=registry._ctx, checkpoint=checkpoint, messages=messages,
                           usage=usage, send=send, address=address, task_id=task_id)


def _refuse_checkpoint_writes(monkeypatch, failures):
    """The rolling file's atomic replace fails while ``failures`` holds entries; its other writes stay real."""
    from ouroboros import utils

    write = utils.write_bytes_atomic

    def flaky(path, content, **kwargs):
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        if failures and name == f"{working_checkpoint.FILE_PREFIX}1.json":  # the rolling state, never its address
            failures.pop()
            raise OSError("synthetic: the working state could not be written")
        return write(path, content, **kwargs)

    monkeypatch.setattr(utils, "write_bytes_atomic", flaky)


@pytest.mark.parametrize("prior", [False, True], ids=["first-ready-write-failed", "stale-earlier-sequence-and-a-changed-view"])
@pytest.mark.parametrize("lands", [False, True], ids=["never-lands", "lands-under-the-hold"])
def test_a_native_create_binds_only_its_rounds_own_landed_ready_state(setup, monkeypatch, prior, lands):
    """A ready write that began and did not land is owed: no native create leaves over an absent sequence, nor over an
    earlier one once the view changed. The existing typed hold writes it first; a control that ends the hold refuses that
    send unsent (released, nothing created), and a write that lands lets exactly one create bind the fresh sequence."""
    from ouroboros import budget_pause
    from tests._budget_pause_exact_helpers import _controls

    call = _ready_native_call(setup, monkeypatch)
    published = []
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.001)
    monkeypatch.setattr(budget_pause, "_publish_hold", lambda _ctx, hold: published.append(dict(hold)))
    if prior:
        assert working_checkpoint.save_round(call.checkpoint, "ready") and call.ctx._working_checkpoint_seq == 1
        call.messages.append({"role": "user", "content": "Owner words the earlier sequence never held."})
    failures = [1] * (2 if lands else 50)
    _refuse_checkpoint_writes(monkeypatch, failures)
    assert working_checkpoint.save_or_log(call.checkpoint, "ready") is False, "the round's own ready write fails"
    assert call.ctx._working_checkpoint_owed is True
    assert getattr(call.ctx, "_working_checkpoint_seq", None) == (1 if prior else None)
    if not lands:
        monkeypatch.setattr(budget_pause, "_hold_control_reason", _controls("", "cancelled"))
        with pytest.raises(ModelWaitInterrupted) as refused:
            call.send()
        assert refused.value.control_reason == "cancelled"
        assert call.gateway.creates == [] and not call.address.exists(), "no create and no address over a stale state"
        assert [row["state"] for row in ledger(call.root)] == ["released"]
        if prior:
            state = working_checkpoint.read_working(call.root, call.task_id, 1)[1]
            assert state["working"]["seq"] == 1 and len(state["messages"]) == 1, "the earlier sequence was never rebound"
    else:
        message, _cost = call.send()
        assert message["content"] == result()["message"]["content"] and len(call.gateway.creates) == 1 and not failures
        row, = ledger(call.root)
        bound = json.loads(call.address.read_text(encoding="utf-8"))
        state = working_checkpoint.read_working(call.root, call.task_id, 1)[1]
        assert bound["seq"] == call.ctx._working_checkpoint_seq == state["working"]["seq"] == (2 if prior else 1)
        assert bound["physical_attempt_id"] == row["attempt_id"] == call.gateway.creates[0]
        assert state["working"]["native_attempt"]["physical_attempt_id"] == row["attempt_id"]
        assert len(state["messages"]) == (2 if prior else 1), "the address names the view that was sent"
        assert call.ctx._working_checkpoint_owed is False and "budget_pause_hold" not in call.usage
    assert [hold["hold_reason"] for hold in published] == [working_checkpoint.HOLD_UNWRITABLE], "the existing hold, once"


@pytest.mark.parametrize("saved", [True, False], ids=["current-ready-landed", "no-write-ever-attempted"])
def test_a_landed_ready_state_binds_at_once_and_an_unsupported_continuation_is_no_failed_save(setup, monkeypatch, saved):
    """The confirmed current ready state is bound without another write or any hold. A call that never attempted a
    write (no ready boundary: a continuation this call does not support) sends as before, with no address and no hold."""
    from ouroboros import budget_pause

    call = _ready_native_call(setup, monkeypatch)
    monkeypatch.setattr(budget_pause, "_publish_hold", lambda *_a: pytest.fail("nothing is owed: no hold"))
    if saved:
        assert working_checkpoint.save_round(call.checkpoint, "ready") and call.ctx._working_checkpoint_owed is False
    message, _cost = call.send()
    assert message["content"] == result()["message"]["content"] and len(call.gateway.creates) == 1
    if saved:
        assert call.usage["working_checkpoint"]["saves"] == 1, "the landed ready write is the one the address names"
        assert json.loads(call.address.read_text(encoding="utf-8"))["seq"] == call.ctx._working_checkpoint_seq == 1
    else:
        assert not call.address.exists() and getattr(call.ctx, "_working_checkpoint_seq", None) is None


def test_a_native_create_is_refused_while_its_ready_state_is_owed_and_no_writer_is_bound(tmp_path):
    """The hold lands an owed write or ends the call: only a call outside every loop's model-call scope has no writer bound,
    and it cannot name that state either: the create is refused unsent, no address is written, no sequence named."""
    from ouroboros.usage_accounting import PhysicalAttemptPreparationFailed

    owner = SimpleNamespace(_working_checkpoint_owed=True, _working_checkpoint_seq=3, task_attempt=1)
    call = SimpleNamespace(tools=SimpleNamespace(_ctx=owner), task_id="task-one", context_fit_plan=None, accumulated_usage={})
    reservation = SimpleNamespace(provider="claudexor", attempt_id="attempt-1", model=MODEL, drive_root=tmp_path)
    with pytest.raises(PhysicalAttemptPreparationFailed):
        loop_transport.ReadyRoundAttempts(call, None, {}, {})(reservation)
    assert not working_checkpoint.checkpoint_path(tmp_path, "task-one", 1).with_suffix(".operation.json").exists()


def _unsent_refusal(code, **context):
    """The engine's terminal body of an operation it refused before starting it: a typed problem, no message."""
    problem = {"code": code, "retryable": False, "message": f"Synthetic {code}.", **({"context": context} if context else {})}
    return {**result(outcome="failed", route={}, problem=problem), "message": None}


# name: (the typed code, its context, what the live wait owner reads of it, the kind the live classifier gives it)
_UNSENT_REFUSALS = {
    "quota": ("subscription_window_exhausted", {"source": "codex", "poolCause": "quota", "resetsAt": "2099-01-01T00:00:00Z"},
              "quota", "subscription_window_exhausted"),
    "auth": ("auth_required", {}, "auth", "auth_error"),
    "non-resource": ("unsupported_parameter", {"parameter": "temperature"}, "", "bad_request"),
}


def _restored_read(saved):
    """The loop's own read of the round's bound operation: its context, its pending record and its usage."""
    usage = deepcopy(saved.saved["usage"])
    pending = loop_transport.ready_operation(saved.saved, usage, saved.root, saved.ctx.task_id)
    ctx = SimpleNamespace(task_id=saved.ctx.task_id, round_idx=1, accumulated_usage=usage, drive_logs=saved.root / "logs",
                          status_drive_root=saved.root, tools=SimpleNamespace(_ctx=saved.ctx))
    return usage, pending, ctx


def _meet_in_the_rounds_own_call(saved, client, usage, *, retries=1):
    """The round's own next call, as the loop makes it after the read: ``call_llm_with_retry`` over the same usage."""
    messages = [{"role": "user", "content": "Continue the recorded task."}]
    checkpoint = SimpleNamespace(tools=SimpleNamespace(_ctx=saved.ctx), task_id=saved.ctx.task_id, messages=messages,
        accumulated_usage=usage, llm_trace={"tool_calls": []}, round_idx=1, tool_schemas=[], owner_msg_seen=set(),
        context_fit_plan=saved.ctx.context_fit_plan)
    return call_llm_with_retry(client, messages, MODEL, [], "low", retries, saved.root / "logs", saved.ctx.task_id, 1, None,
                               usage, checkpoint_call=checkpoint)


@pytest.mark.parametrize("start", ["unresolved", "released-and-retained", "released-without-its-receipt"])
@pytest.mark.parametrize("kind", list(_UNSENT_REFUSALS))
def test_a_released_typed_refusal_is_met_by_its_rounds_own_call_and_never_sent_over(setup, monkeypatch, kind, start):
    """The engine's typed refusal of an unsent operation, quota, auth or any other, reaches the round's own call exactly as the
    live call raised it: the live owners read its type, and nothing is sent over it. It is read from the engine once, then from
    its retained receipt offline; only a release with no retained reason is the ordinary owed round, admitted again."""
    from ouroboros.observability import call_manifest_path

    code, context, reason, failure_code = _UNSENT_REFUSALS[kind]
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, task_id, first = saved.gateway, saved.ctx.task_id, saved.first["attempt_id"]
    gateway.results, gateway.dispatch = [_unsent_refusal(code, **context)], ["not_started"]
    _bounded_creates(monkeypatch, gateway, 1)
    if start != "unresolved":  # an earlier restore read the engine once, released the row and retained the receipt
        bound = loop_transport.ready_operation(saved.saved, deepcopy(saved.saved["usage"]), saved.root, task_id)
        with pytest.raises(llm_claudexor.ClaudexorModelNotDispatched) as earlier:
            llm_claudexor.recover_chat_reply(saved.root, task_id, bound, None)
        assert earlier.value.problem["code"] == code and ledger(saved.root)[0]["state"] == "released"
        monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: pytest.fail("a released receipt is retained"))
    if start == "released-without-its-receipt":
        call_manifest_path(saved.root, task_id, f"{first}_model_response").unlink()
    usage, pending, ctx = _restored_read(saved)
    if start == "released-without-its-receipt":
        assert pending is None and first in usage["ledger_attempt_ids"], "no retained reason: the owed round is admitted again"
        assert gateway.creates == [first] and gateway.reads == ["op-0"] and len(gateway.acks) == 1
        return
    assert pending["physical_attempt_id"] == first and pending["operation_id"] == "op-0"
    with pytest.raises(llm_claudexor.RestoredEnd):
        loop_transport.read_ready_operation(ctx, pending)
    end = usage[llm_claudexor.RESTORED_END_KEY]
    assert (end["custody"], end["problem"]["code"], end["physical_attempt_id"], end["operation_id"]) == (
        "released", code, first, "op-0")
    assert end["round_id"] == "original:round:1" and not usage.get("_last_llm_error_kind") and "_pending_transport_outcome" not in usage
    assert llm_claudexor.RESTORED_END_KEY not in working_checkpoint._TRANSIENT_USAGE, "a saved state carries it until a call meets it"
    assert [row["state"] for row in ledger(saved.root)] == ["released"] and gateway.creates == [first]
    assert gateway.reads == ["op-0"] and len(gateway.acks) == 1, "the engine was read once, its retained receipt after that"

    answer, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage)
    assert answer is None and gateway.creates == [first] and not usage.get(llm_claudexor.RESTORED_END_KEY), "met: spent in place"
    failure = usage["_last_llm_failure"]
    assert (failure["failure_code"], failure["resource_refusal"], failure["physical_attempt_id"],
            failure["attempt_custody_state"], failure["operation_id"]) == (failure_code, reason, first, "released", "op-0")
    assert len(ledger(saved.root)) == 1 and first in usage["ledger_attempt_ids"]


# name: (the engine's terminal body, the creates the round's own call makes after it, the kind the live classifier gives it)
_RECEIVED_FAILURES = {
    "transient": (lambda: result(outcome="failed", cash=0.17, knowledge="exact", problem={
        "code": "provider_failed", "retryable": True, "message": "Upstream overloaded.", "context": {"httpStatus": 503}}),
        1, "provider_transient"),
    "bad-input": (lambda: result(outcome="failed", cash=0.17, knowledge="exact", problem={
        "code": "invalid_request", "message": "Invalid argument", "context": {"httpStatus": 400}}), 0, "bad_request"),
    "substituted": (lambda: {**result(cash=0.17, knowledge="exact"), "modelMismatch": {
        "requested": ROUTE["model"], "observed": "other-fixture-model"}}, 0, "model_substituted"),
}


@pytest.mark.parametrize("failure", list(_RECEIVED_FAILURES))
def test_a_received_failure_is_met_by_its_rounds_own_call(setup, monkeypatch, failure):
    """A failure the engine settled before the restore is the round's own known end: the read settles and counts it once,
    and the round's call meets it as the live call raised it. Only the live burst repeats a temporary failure, on its own
    route; bad input repeats nothing, nor does another model's answer whose older record names no redo its admission left.
    No send is made by the read, and the met end's row keeps that attempt's own operation, so nothing is bought over it."""
    from ouroboros import loop_llm_call
    from ouroboros.observability import update_call_manifest_metadata

    body, creates, kind = _RECEIVED_FAILURES[failure]
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, first = saved.gateway, saved.first["attempt_id"]
    if failure == "substituted":  # a record written before the allowance was kept with it: no unused redo is proved
        update_call_manifest_metadata(saved.root, task_id=saved.ctx.task_id, call_id=f"{first}_model_request",
                                      metadata={"substitution_redos_left": None})
    answer = result(cash=0.17, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "The round's own call answered."}
    gateway.results, gateway.dispatch = [body(), answer], ["response_received", "response_received"]
    _bounded_creates(monkeypatch, gateway, 1 + creates)
    monkeypatch.setattr(loop_llm_call, "_retry_backoff_sec", lambda *_args: 0.0)
    usage, pending, ctx = _restored_read(saved)
    with pytest.raises(llm_claudexor.RestoredEnd):
        loop_transport.read_ready_operation(ctx, pending)
    assert usage[llm_claudexor.RESTORED_END_KEY]["custody"] == "settled" and not usage.get("_last_llm_error_kind")
    read_status = {key: usage["_last_llm_failure"][key] for key in ("operation_state", "operation_outcome")}
    assert all(read_status.values()), "the receipt's own final state and outcome, as the read stamped them"
    settled, = ledger(saved.root)
    assert (settled["attempt_id"], settled["state"], settled["cost_usd"]) == (first, "settled", 0.17)
    assert usage["cost"] == 0.17 and gateway.creates == [first] and gateway.reads == ["op-0"]

    reply, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage, retries=3)
    rows = ledger(saved.root)
    assert len(gateway.creates) == len(rows) == 1 + creates and rows[0]["cost_usd"] == 0.17
    if creates:
        assert reply["content"] == answer["message"]["content"] and [row["state"] for row in rows] == ["settled", "settled"]
        assert gateway.creates[1] == rows[1]["attempt_id"] != first
    else:
        assert reply is None and usage["_last_llm_failure"]["failure_code"] == kind
        assert usage["_last_llm_failure"]["physical_attempt_id"] == first and usage["cost"] == 0.17, "counted once"
        assert (usage["_last_llm_failure"]["recovered_operation_id"], usage["_last_llm_failure"]["restored_attempt"]) == (
            "op-0", True), "the read's own operation: the no-call rail buys no request over it"
        assert loop_llm_call.provider_no_call_source(usage, False) == ("recovered_operation_no_resend", False)
    replayed = next(row for row in usage["llm_call_refs"] if row.get("physical_attempt_id") == first)
    assert {key: replayed[key] for key in read_status} == read_status, "the live owner's own row keeps them, not the read's alone"
    assert not usage.get(llm_claudexor.RESTORED_END_KEY), "a known end is met once and never kept for a later call"
    event = next(row for row in _event_rows(saved.root) if row.get("type") == "llm_api_error" and row.get("physical_attempt_id") == first)
    assert (event["llm_call_id"], event["request_ref"], event["execution_id"], event["round"]) == (
        pending["llm_call_id"], pending["request_ref"], pending["execution_id"], 1), "the call and request its own record names"
    assert event["current_llm_call_id"] != event["llm_call_id"] and replayed["llm_call_id"] == event["llm_call_id"]
    assert event["restored_fields"] == ["attempt", "execution_id", "llm_call_id", "round", "round_id"], "all its record names"
    assert event["operation_id"] == "op-0" and {key: event[key] for key in read_status} == read_status
    assert (event["status_code"], event["attempt_custody_state"]) == (replayed["status_code"], "settled")


def _event_rows(root):
    return [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("record", ["its-last-attempt", "an-unnumbered-record", "a-raised-setting"])
def test_a_restored_temporary_failure_keeps_its_own_place_in_the_burst(setup, monkeypatch, record):
    """A temporary failure a restored read met takes its own attempt's place in its round's burst: the one it reached last
    leaves no repeat, and a record that names no attempt proves no unused one. Either way the round's own call meets it and
    sends nothing; the class alone would have repeated it (the burst's first attempt does:
    ``test_a_received_failure_is_met_by_its_rounds_own_call``)."""
    from ouroboros import loop_llm_call

    monkeypatch.setenv("OUROBOROS_TRANSIENT_RETRY_MAX", "3")
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, first = saved.gateway, saved.first["attempt_id"]
    native = saved.saved["working"]["native_attempt"]
    assert (native["attempt_budget"], native["max_retries"]) == (3, 1), "the burst its send was admitted in, recorded before it"
    if record == "an-unnumbered-record":
        del native["attempt"]  # a receipt recorded without the ordinal
    else:
        native["attempt"] = 3  # the third of the burst of three it was admitted in
    if record == "a-raised-setting":
        monkeypatch.setenv("OUROBOROS_TRANSIENT_RETRY_MAX", "6")  # a larger burst today renews none of that one
    answer = result(cash=0.17, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "A repeat answered."}
    gateway.results, gateway.dispatch = [_RECEIVED_FAILURES["transient"][0](), answer], ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 1)
    monkeypatch.setattr(loop_llm_call, "_retry_backoff_sec", lambda *_args: 0.0)
    usage, pending, ctx = _restored_read(saved)
    with pytest.raises(llm_claudexor.RestoredEnd):
        loop_transport.read_ready_operation(ctx, pending)
    assert usage[llm_claudexor.RESTORED_END_KEY]["origin"].get("attempt") == (None if record == "an-unnumbered-record" else 3)

    reply, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage, retries=3)
    assert reply is None and gateway.creates == [first] and len(ledger(saved.root)) == 1, "no repeat is bought"
    failure = usage["_last_llm_failure"]
    assert (failure["failure_code"], failure["physical_attempt_id"], failure["recovered_operation_id"]) == (
        "provider_transient", first, "op-0")
    assert usage[loop_llm_call.RETRY_WALL_EXHAUSTED_KEY] is True and usage["cost"] == pytest.approx(0.17)
    event = next(row for row in _event_rows(saved.root) if row.get("type") == "llm_api_error" and row.get("physical_attempt_id") == first)
    assert (event["attempt"], event["retry_same_request"]) == (3, True), "its place is the last; its class alone would repeat"
    assert ("attempt" in event["restored_fields"]) is (record != "an-unnumbered-record"), "the ordinal is the record's only if named"


@pytest.mark.parametrize("left", [2, 0, "pinned"], ids=["redos-left", "none-left", "pinned-then-auto"])
def test_a_restored_substituted_answer_spends_only_the_redos_its_admission_recorded(setup, monkeypatch, left):
    """Another model answered a restored attempt. Its request recorded, before dispatch, the redos its call still had (on
    Auto with none spent: the configured two). With some left, the round's own call decides the redo by the live policy and
    asks the round once more as a new operation, the discarded answer disclosed beside the accepted one. With none left
    nothing is asked again: the typed refusal is the round's failure. Nothing beyond that allowance is ever bought."""
    from ouroboros.observability import read_call_payload, update_call_manifest_metadata

    pinned = left == "pinned"
    if pinned:  # its send was pinned to one account: its admission allowed no redo, whatever the round is bound to later
        monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a"}))
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, first, task_id = saved.gateway, saved.first["attempt_id"], saved.ctx.task_id
    recorded = read_call_payload(saved.root, task_id=task_id, call_id=f"{first}_model_request")[0]["substitution_redos_left"]
    assert recorded == (0 if pinned else 2), "what admission's own rule allowed, recorded before dispatch"
    if pinned:
        monkeypatch.delenv(MODEL_ACCOUNTS_KEY)  # Auto now: today's binding alone would allow a redo; its admission did not
        left = 0
    elif not left:
        update_call_manifest_metadata(saved.root, task_id=task_id, call_id=f"{first}_model_request",
                                      metadata={"substitution_redos_left": 0})
    answer = result(cash=0.23, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "The requested model answered the redo."}
    gateway.results, gateway.dispatch = [_RECEIVED_FAILURES["substituted"][0](), answer], ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 2 if left else 1)
    usage, pending, ctx = _restored_read(saved)
    with pytest.raises(llm_claudexor.RestoredEnd):
        loop_transport.read_ready_operation(ctx, pending)
    assert usage[llm_claudexor.RESTORED_END_KEY]["problem"]["context"]["redos_left"] == left
    assert gateway.creates == [first] and gateway.reads == ["op-0"] and usage["cost"] == pytest.approx(0.17)

    reply, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage)
    rows = ledger(saved.root)
    assert (rows[0]["attempt_id"], rows[0]["cost_usd"]) == (first, 0.17)
    if left:
        assert reply["content"] == answer["message"]["content"] and len(gateway.creates) == len(rows) == 2
        assert [(row["disposition"], row["result_custody"]["operation_id"]) for row in usage["_model_substitutions"]] == [
            ("redo", "op-0")], "the discarded answer, disclosed beside the accepted one"
        assert usage["cost"] == pytest.approx(0.40)
    else:
        assert reply is None and gateway.creates == [first] and len(rows) == 1
        failure = usage["_last_llm_failure"]
        assert (failure["failure_code"], failure["physical_attempt_id"], failure["recovered_operation_id"]) == (
            "model_substituted", first, "op-0")


def test_a_restored_reply_with_nothing_usable_is_a_rejected_end_its_rounds_call_meets(setup, monkeypatch):
    """A restored read received a completed reply with nothing usable in it. It counts that receipt once and ends known, a
    whole response judged unusable (``response_rejected``), as the engine's own undeliverable verdict does: the round's own
    call meets it, so the configured walk is the live one's, and neither a same-model repeat nor a summary is bought over it."""
    from ouroboros import loop_llm_call

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, first = saved.gateway, saved.first["attempt_id"]
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    empty = result(cash=0.17, knowledge="exact")
    empty["message"] = {"role": "assistant", "content": " \n"}
    answer = result(cash=0.23, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "A same-model repeat answered."}
    gateway.results, gateway.dispatch = [empty, answer], ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 1)
    usage, pending, ctx = _restored_read(saved)
    with pytest.raises(llm_claudexor.RestoredEnd):
        loop_transport.read_ready_operation(ctx, pending)
    end = usage[llm_claudexor.RESTORED_END_KEY]
    assert (end["physical_attempt_id"], end["problem"]["code"], end["rejected"]) == (first, "response_rejected", True)
    assert usage["cost"] == pytest.approx(0.17) and first in usage["_counted_error_attempts"], "its receipt, counted once"
    assert usage["_last_llm_failure"]["failure_code"] == "response_rejected", "the read's own row keeps what it received"

    reply, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage, retries=3)
    assert reply is None and gateway.creates == [first] and len(ledger(saved.root)) == 1, "no same-model repeat is bought"
    failure = usage["_last_llm_failure"]
    assert (failure["failure_code"], failure["physical_attempt_id"], failure["recovered_operation_id"]) == (
        "provider_error", first, "op-0")
    assert usage["cost"] == pytest.approx(0.17), "never counted twice"
    assert loop_llm_call.provider_no_call_source(usage, False) == ("recovered_operation_no_resend", False), "nor a summary"


@pytest.mark.parametrize("case", ["content-refusal", "unbound-store"])
def test_what_is_not_carried_ends_as_it_did_and_sends_nothing_over(setup, monkeypatch, case):
    """Not met by the round's own call: a definitive content refusal (terminal on every route) and a request that recorded no
    store (unknown, read from nowhere, its cost unknown)."""
    from ouroboros.observability import update_call_manifest_metadata

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, first = saved.gateway, saved.first["attempt_id"]
    _bounded_creates(monkeypatch, gateway, 1)
    if case == "content-refusal":
        gateway.results, gateway.dispatch = [result(outcome="failed", cash=0.17, knowledge="exact", problem={
            "code": "provider_content_refused", "retryable": False, "message": "Synthetic refusal"})], ["response_received"]
    else:
        update_call_manifest_metadata(saved.root, task_id=saved.ctx.task_id, call_id=f"{first}_model_request",
                                      metadata={"model_store_binding": None})  # a request an earlier version recorded
        gateway.results, gateway.dispatch = [result(cash=0.17, knowledge="exact")], ["response_received"]  # readable, unread
    usage, pending, ctx = _restored_read(saved)
    if case == "content-refusal":
        with pytest.raises(llm_claudexor.ClaudexorContentRefused):
            loop_transport.read_ready_operation(ctx, pending)
        assert ledger(saved.root)[0]["state"] == "settled" and usage["cost"] == 0.17
    else:
        assert loop_transport.read_ready_operation(ctx, pending) == (None, None)
        row, = ledger(saved.root)
        assert row["state"] == "unresolved" and row.get("cost_usd") is None and not gateway.reads and not gateway.acks
        assert usage["_last_llm_error_kind"] == "provider_outcome_unknown" and usage["execution_status"] == "infra_failed"
        assert usage["_last_llm_provider_code"] == "model_control_store_unbound", "why it stays unknown"
    assert not usage.get(llm_claudexor.RESTORED_END_KEY) and gateway.creates == [first]


@pytest.mark.parametrize("case", ["auth_changed", "auth_changed-spent", "auth_changed-legacy", "invalid_continuation"])
def test_a_cold_auth_changed_release_keeps_only_the_reissue_its_own_record_proves_unused(setup, monkeypatch, case):
    """Every new physical attempt records its call's one no-start ``auth_changed`` reissue (``auth_reissue_left``: 1 unused,
    0 spent). A restored released ``auth_changed`` whose own record says 1 is met by the round's own call as that one reissue:
    a new operation on the same route, recorded 0, and the released attempt's record is marked 0 before its create, so a
    repeated restore of the same release (a crash before the checkpoint moved) renews nothing. A spent (0) or older (no field)
    record, and every other code, stay as before: a typed end (``bad_request``) and nothing sent over it. The live one-shot
    repair is unchanged (``tests/test_auth_changed_reissue.py``)."""
    from ouroboros.observability import call_manifest_path, update_call_manifest_metadata

    code, task_id = case.split("-", 1)[0], "task-one"
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, first = saved.gateway, saved.first["attempt_id"]

    def recorded(attempt):
        return json.loads(call_manifest_path(saved.root, task_id, f"{attempt}_model_request").read_text(encoding="utf-8"))

    assert recorded(first)["auth_reissue_left"] == 1, "the first send records its call's unused allowance"
    gateway.results, gateway.dispatch = [_unsent_refusal(code), result(cash=0.17, knowledge="exact")], ["not_started", "response_received"]
    if code == "auth_changed" and case != "auth_changed":
        update_call_manifest_metadata(saved.root, task_id=task_id, call_id=f"{first}_model_request",
                                      metadata={"auth_reissue_left": 0 if case.endswith("spent") else None})
    allowed = case == "auth_changed"
    _bounded_creates(monkeypatch, gateway, 2 if allowed else 1)
    usage, pending, ctx = _restored_read(saved)
    with pytest.raises(llm_claudexor.RestoredEnd):
        loop_transport.read_ready_operation(ctx, pending)
    reply, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage)
    assert not usage.get(llm_claudexor.RESTORED_END_KEY), "met: spent in place"
    if not allowed:
        assert reply is None and gateway.creates == [first] and [row["state"] for row in ledger(saved.root)] == ["released"]
        failure = usage["_last_llm_failure"]
        assert (failure["failure_code"], failure["physical_attempt_id"], failure["operation_id"]) == ("bad_request", first, "op-0")
        return
    assert reply is not None and len(gateway.creates) == 2 and gateway.creates[0] == first != gateway.creates[1]
    second = gateway.creates[1]
    assert [row["state"] for row in ledger(saved.root)] == ["released", "settled"]
    assert recorded(second)["auth_reissue_left"] == 0 and recorded(first)["auth_reissue_left"] == 0, "spent on both records"
    usage, pending, ctx = _restored_read(saved)  # the same release restored again: its record now proves the reissue spent
    with pytest.raises(llm_claudexor.RestoredEnd):
        loop_transport.read_ready_operation(ctx, pending)
    assert usage[llm_claudexor.RESTORED_END_KEY]["auth_reissue_left"] == 0
    reply, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage)
    assert reply is None and len(gateway.creates) == 2, "no renewal after the attempted reissue"


@pytest.mark.parametrize("custody", ["settled", "released"])
def test_a_restored_size_refusal_meets_the_size_owner_and_sends_nothing_over(setup, monkeypatch, custody):
    """A typed size refusal, received or released, is the round's own known end: the size owner reads it through the live
    call, never an ordinary admission that sends the same request over it. What it records stays unknown: nothing in the record
    proves the refused request's bytes or tokens (the engine's words state no count), and the package the round measures now
    is not that request, so its estimate is never taken for the refused size."""
    from ouroboros import loop_model_call
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY, request_window_fact

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, first = saved.gateway, saved.first["attempt_id"]
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    context = {"httpStatus": 400, "vendorCode": "context_length_exceeded", "parameter": "input"}
    gateway.results, gateway.dispatch = ([_unsent_refusal("invalid_request", **context)], ["not_started"]) if custody == "released" else (
        [result(outcome="failed", cash=0.17, knowledge="exact", problem={
            "code": "invalid_request", "message": "Codex model request was refused (HTTP 400).", "retryable": False,
            "context": context})], ["response_received"])
    _bounded_creates(monkeypatch, gateway, 1)
    seen, remember = [], loop_model_call.remember_size_refusal

    def spy(ctx, refused, **kwargs):  # the size owner's own call, with what it was given and what it recorded
        remember(ctx, refused, **kwargs)
        seen.append((refused, kwargs["measured"], dict(ctx.accumulated_usage[SIZE_REFUSAL_KEY])))

    monkeypatch.setattr(loop_model_call, "remember_size_refusal", spy)
    _text, usage, _trace = saved.resume()
    row, = ledger(saved.root)
    assert (row["attempt_id"], row["state"]) == (first, custody) and gateway.creates == [first] and gateway.reads == ["op-0"]
    (refused, measured, slot), = seen
    assert isinstance(refused, llm_claudexor.RestoredCapture) and refused.attempt_id == first and refused.state == custody
    assert measured is False and (slot["tokens"], slot["basis"]) == (0, ""), "unknown, never this round's own estimate"
    final = request_window_fact(  # the terminal's size words read that record, not this round's view
        {**usage, SIZE_REFUSAL_KEY: slot, "_context_prompt_estimate": 12345}, overflow=True, round_idx=slot["round"])
    assert "request size unknown" in final and "12,345" not in final
    assert (usage.get("cost") or 0) == (0.17 if custody == "settled" else 0), "the read counted it once; the replay counts nothing"
    origin = saved.saved["working"]["native_attempt"]
    shown = [row for row in _event_rows(saved.root) if row.get("current_llm_call_id")]  # only a restored end's rows name its evaluator
    assert {row["type"] for row in shown} == {"llm_api_error", "remote_context_overflow"}
    assert all(row["llm_call_id"] == origin["llm_call_id"] != row["current_llm_call_id"] and "estimated_prompt_tokens" not in row
               for row in shown), "the old call, never the unsent one; none of its fit estimate"
    assert next(row for row in shown if row["type"] == "llm_api_error")["request_ref"] == origin["request_ref"]


@pytest.mark.parametrize("kind", ["request_too_large", "bad_request"])
def test_a_restored_failures_terminal_never_reads_this_rounds_estimate_as_its_request_size(setup, monkeypatch, kind):
    """The terminal's size words for a failure kind that keeps no size record (a body-size refusal, a bad request) read the usage's
    ``_context_prompt_estimate``. For a live failure that is the request just sent; for a restored attempt's end it is the package
    this round measured, which is not the refused request: its size and window stay unknown, whatever the usage holds."""
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway = saved.gateway
    gateway.results, gateway.dispatch = [_RECEIVED_FAILURES["bad-input"][0]()], ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    usage, pending, ctx = _restored_read(saved)
    with pytest.raises(llm_claudexor.RestoredEnd):
        loop_transport.read_ready_operation(ctx, pending)
    reply, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage)
    assert reply is None and usage["_last_llm_failure"]["restored_attempt"] is True
    usage.update(_context_prompt_estimate=12345, _last_llm_error_kind=kind)  # the new package's estimate stands in the usage

    def words():
        return loop_transport._provider_terminal_text(usage, is_context_overflow=False, is_transport_wait=False, waited_sec=0.0,
                                                      is_deadline_exhausted=False, round_idx=1)

    restored = words()
    assert "request size unknown" in restored and "12,345" not in restored
    del usage["_last_llm_failure"]["restored_attempt"]  # the same failure as a live one would be recorded: its estimate is its own
    assert "request input estimated 12,345 tokens" in words()


def test_a_restored_resource_refusal_waits_before_it_sends_when_no_route_follows(setup, monkeypatch):
    """With no configured route the live call waits in place on a quota verdict and sends only once the wait ends. The same
    verdict read after restoration takes that wait before any create: the unsent round is never resent blind."""
    code, context, reason, _kind = _UNSENT_REFUSALS["quota"]
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, first = saved.gateway, saved.first["attempt_id"]
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    answer = result(cash=0.17, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "Sent after the wait."}
    gateway.results, gateway.dispatch = [_unsent_refusal(code, **context), answer], ["not_started", "response_received"]
    _bounded_creates(monkeypatch, gateway, 2)
    waited = []

    def wait(_owner, _llm, error, kwargs, *_args, **_kw):
        waited.append((len(gateway.creates), model_wait.model_wait_reason(error), error.operation_id,
                       error.physical_attempt_capture.state, error.physical_attempt_capture.attempt_id,
                       isinstance(error, llm_claudexor.ClaudexorModelNotDispatched)))
        return kwargs  # the resource is available again: the same call goes on

    monkeypatch.setattr(model_wait.TaskModelWait, "wait", wait)
    text, usage, _trace = saved.resume()
    released, sent = ledger(saved.root)
    assert waited == [(1, reason, "op-0", "released", first, True)], "the wait precedes the round's only new create"
    assert (released["attempt_id"], released["state"]) == (first, "released") and sent["state"] == "settled"
    assert gateway.creates == [first, sent["attempt_id"]] and text == answer["message"]["content"]
    assert usage["cost"] == sent["cost_usd"] == 0.17 and "_pending_transport_outcome" not in usage


@pytest.mark.parametrize("custody", ["released", "settled"])
def test_the_known_end_survives_a_snapshot_between_its_read_and_its_decision(setup, monkeypatch, custody):
    """The worker dies after the round-top ready write and before the round's own call. The saved state, read back by the real
    restore consumers, still holds the end the read knew: no pending read to read again (no second engine read, no second
    charge), no native address a new create could bind, and the round's call meets the end from that snapshot, sending nothing."""
    code, context, _reason, _kind = _UNSENT_REFUSALS["quota"]
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, task_id, first = saved.gateway, saved.ctx.task_id, saved.first["attempt_id"]
    gateway.results, gateway.dispatch = (([_unsent_refusal(code, **context)], ["not_started"]) if custody == "released"
                                         else ([_RECEIVED_FAILURES["bad-input"][0]()], ["response_received"]))
    _bounded_creates(monkeypatch, gateway, 1)

    class Died(BaseException):
        """The worker's process ends before its round's own call."""

    def die(_call):
        raise Died

    monkeypatch.setattr(loop, "_call_round_model", die)
    with pytest.raises(Died):
        saved.resume()
    state = working_checkpoint.read_working(saved.root, task_id, 2)[1]
    usage = deepcopy(state["usage"])
    end = usage[llm_claudexor.RESTORED_END_KEY]
    assert (end["custody"], end["physical_attempt_id"], end["operation_id"], end["round_id"]) == (
        custody, first, "op-0", "original:round:1")
    assert state["working"]["boundary"] == "ready" and "native_attempt" not in state["working"]
    assert "_pending_transport_outcome" not in usage and loop_transport.ready_operation(state, usage, saved.root, task_id) is None
    assert gateway.reads == ["op-0"] and gateway.creates == [first], "one engine read, and nothing sent"
    counted = usage.get("cost")
    reply, _cost = _meet_in_the_rounds_own_call(saved, setup[2], usage)
    assert reply is None and gateway.creates == [first] and usage.get("cost") == counted, "met from the snapshot: counted once"
    assert usage["_last_llm_failure"]["physical_attempt_id"] == first and not usage.get(llm_claudexor.RESTORED_END_KEY)


def test_a_known_end_is_met_once_and_only_by_its_own_round_model_and_role():
    """The end a restored read hands on is the round's own call's to meet, once: another round takes it for nothing, and a call on
    another model or role drops it unmet. Met, it is the typed error of the live call, carrying the restored attempt's own
    capture as the process's last, so no size or image owner can read an earlier round's candidate as the refused one."""
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.loop_model_call import _failed_capture_is_comparable
    from ouroboros.usage_accounting import last_physical_attempt_capture

    origin = {"llm_call_id": "llm-old", "execution_id": "x", "round": 2, "attempt": 1, "request_ref": {"call_id": "llm-old_request"}}
    end = {"model": MODEL, "model_role": "main", "operation_id": "op-0", "physical_attempt_id": "attempt-1", "round_id": "x:round:2",
           "problem": {"code": "provider_failed", "retryable": True, "context": {"httpStatus": 503}}, "custody": "settled", "route": {},
           "receipt": {"operation_state": "failed", "outcome": "failed", "requested_profile": "account-a"}, "origin": origin}
    usage = {llm_claudexor.RESTORED_END_KEY: dict(end)}
    assert llm_claudexor.restored_round(usage, "x:round:1") == {} and llm_claudexor.RESTORED_END_KEY not in usage, "another round's: dropped"
    usage = {llm_claudexor.RESTORED_END_KEY: dict(end)}
    carrier = llm_claudexor.restored_round(usage, "x:round:2")
    assert carrier == {"end": end} and carrier["end"] is usage[llm_claudexor.RESTORED_END_KEY], "the saved usage's own record"
    assert llm_claudexor.restored_round(usage, "x:round:2") == carrier, "handing it to a call spends nothing: only meeting it does"
    for target, role in (({"usage_model": "claudexor::codex=another-model"}, "main"), ({"usage_model": MODEL}, "review")):
        unmet = {**target, llm_claudexor.RESTORED_PROCESSING_KEY: {"end": dict(end)}}
        assert llm_claudexor._meet_restored_end(unmet, {"model_role": role}) is None
        assert "end" not in unmet[llm_claudexor.RESTORED_PROCESSING_KEY]
    wrong_role = {llm_claudexor.RESTORED_END_KEY: dict(end)}  # a call of another role does not meet it, and spends it
    assert llm_claudexor._meet_restored_end({"usage_model": MODEL, llm_claudexor.RESTORED_PROCESSING_KEY:
                                             llm_claudexor.restored_round(wrong_role, "x:round:2")}, {"model_role": "review"}) is None
    assert wrong_role[llm_claudexor.RESTORED_END_KEY] == {} and llm_claudexor.restored_round(wrong_role, "x:round:2") == {}, (
        "the matching role's later call finds no stale end to replay")
    met = {"usage_model": MODEL, llm_claudexor.RESTORED_PROCESSING_KEY: carrier}
    with pytest.raises(llm_claudexor.ClaudexorModelError) as raised:
        llm_claudexor._meet_restored_end(met, {"model_role": "main"})
    error = raised.value
    assert type(error) is llm_claudexor.ClaudexorModelError and error.problem == end["problem"] and error.operation_id == "op-0"
    assert error.physical_attempt_capture.state == "settled" and last_physical_attempt_capture() is error.physical_attempt_capture
    assert isinstance(error.physical_attempt_capture, llm_claudexor.RestoredCapture) and error.physical_attempt_capture.restored
    assert error.physical_attempt_capture.candidate_measurement_kind == "opaque", "no record proves the refused request"
    recorded = replace(error.physical_attempt_capture, candidate_measurement_kind="canonical_json_v1", candidate_raw_sha256="a" * 64,
                       candidate_context_size_bytes=10, physical_context=SimpleNamespace())
    assert not _failed_capture_is_comparable(error.physical_attempt_capture) and _failed_capture_is_comparable(recorded), (
        "the size owner's own reader decides on the record's facts, and the restored capture has none")
    assert error.received_status == {"operation_state": "failed", "operation_outcome": "failed"}, "the receipt's own state and outcome"
    assert set(error.usage) == {"claudexor"} and error.status_code == 503, "no counters: nothing counts twice; the code's own status"
    assert error.ledger_attempt_ids == ["attempt-1"] and "end" not in met[llm_claudexor.RESTORED_PROCESSING_KEY]
    assert error.restored_origin == origin
    identity = {"execution_id": "x", "round_id": "x:round:2", "llm_call_id": "llm-new", "round": 2, "attempt": 3, "model": MODEL}
    new_fit, new_ref = {"estimated_prompt_tokens": 12345}, {"call_id": "llm-new_request"}
    shown, fit, ref, facts = llm_claudexor.restored_view(error, identity, new_fit, new_ref)
    assert (shown["llm_call_id"], shown["attempt"], shown["current_llm_call_id"], shown["current_attempt"]) == ("llm-old", 1, "llm-new", 3)
    assert (fit, ref) == ({}, origin["request_ref"]), "the old request; none of the unsent call's fit facts"
    assert facts == {"restored_attempt": True, "recovered_operation_id": "op-0"}, "a settled end keeps its own operation"
    assert shown["restored_fields"] == ["attempt", "execution_id", "llm_call_id", "round"] and shown["round_id"] == "x:round:2"
    error.restored_origin = {}  # a record that names no call: its fields stay the evaluating call's, named as none of its own
    shown, _fit, ref, _facts = llm_claudexor.restored_view(error, identity, new_fit, new_ref)
    assert {key: shown[key] for key in ("execution_id", "round_id", "llm_call_id", "round", "attempt")} == {
        key: identity[key] for key in ("execution_id", "round_id", "llm_call_id", "round", "attempt")}, "never null, never invented"
    assert (shown["restored_fields"], ref, shown["current_llm_call_id"]) == ([], None, "llm-new")
    released = llm_claudexor.ClaudexorModelNotDispatched(end["problem"], operation_id="op-0")
    released.restored_origin = origin
    assert llm_claudexor.restored_view(released, identity, new_fit, new_ref)[3] == {"restored_attempt": True}, "unsent: no read"
    assert llm_claudexor.restored_view(ValueError("live"), identity, new_fit, new_ref) == (identity, new_fit, new_ref, {})
    rejected = {llm_claudexor.RESTORED_PROCESSING_KEY: {"end": {**end, "rejected": True, "problem": {
        "code": "provider_incomplete", "message": "No usable message."}}}, "usage_model": MODEL}
    with pytest.raises(llm_claudexor.ClaudexorModelError) as unusable:
        llm_claudexor._meet_restored_end(rejected, {"model_role": "main"})
    assert unusable.value.stream_rejected is True, "a whole response judged unusable stays that across the restore"
    assert classify_llm_exception(unusable.value).retry_same_request is False, "never the same request again"
    assert usage[llm_claudexor.RESTORED_END_KEY] == {}, "spent in place: a later call finds no end to meet"
    for code in ("auth_changed", "invalid_continuation"):  # no transport repair rides a restored end: raised like any typed refusal
        released = {**end, "custody": "released", "problem": {"code": code, "message": "Synthetic."}, "receipt": {}}
        with pytest.raises(llm_claudexor.ClaudexorModelNotDispatched) as typed:
            llm_claudexor._meet_restored_end({"usage_model": MODEL, llm_claudexor.RESTORED_PROCESSING_KEY: {"end": released}},
                                             {"model_role": "main"})
        assert typed.value.code == code and typed.value.received_status == {}


@pytest.mark.parametrize("row,allowed", [
    ({"restored_attempt": True, "rejected_receipt": True, "recovered_operation_id": "op-0"}, False),
    ({"restored_attempt": True, "recovered_operation_id": "op-1", "failure_code": "provider_failed"}, True),
    ({"failure_code": "provider_error", "operation_id": "op-2", "rejected_receipt": True}, False),
    ({"failure_code": "provider_error"}, True),
])
def test_a_restored_rejected_receipt_never_opens_the_fallback_chain(row, allowed):
    """The shared route authority reads the failure row's own restored-rejected mark: a restored read whose
    receipt was judged unusable (no message, or completed whitespace/unusable) ended known and terminal, so no
    configured route buys a new generation over it; a non-rejected restored end and an ordinary known failure
    keep the chain policy unchanged (``loop_transport.fallback_chain_allowed``)."""
    from ouroboros.loop_transport import fallback_chain_allowed

    ctx = SimpleNamespace(task_id="t", exact_model_route=False, route_wait_on_primary=False)
    usage = {"_last_llm_error_kind": "provider_error", "_last_llm_failure": row}
    assert fallback_chain_allowed(ctx, "provider_error", None, usage, primary_acting=True) is allowed


def test_a_restored_unusable_reply_marks_its_failure_row_terminal():
    """A restored end whose receipt was rejected (``response_rejected``: no message, or completed but unusable)
    is met by its round's own call as a stream-rejected error, and ``restored_view`` adds ``rejected_receipt`` to
    the failure row — the owned status the route authority and the no-call rail read; the typed receipt (state,
    outcome, profile) rides the row unchanged, and the wrap-up rail's ``recovered_operation_no_resend`` still
    applies (``loop_llm_call.provider_no_call_source``)."""
    from ouroboros.loop_transport import fallback_chain_allowed

    end = {"model": MODEL, "model_role": "main", "operation_id": "op-9", "physical_attempt_id": "attempt-9",
           "round_id": "x:round:3", "rejected": True, "custody": "settled", "route": {},
           "problem": {"code": "response_rejected", "message": "The recorded operation returned no usable model reply."},
           "receipt": {"operation_state": "succeeded", "outcome": "completed", "requested_profile": "account-a"},
           "origin": {"llm_call_id": "llm-old", "execution_id": "x", "round": 3, "attempt": 1}}
    target = {"usage_model": MODEL, llm_claudexor.RESTORED_PROCESSING_KEY: {"end": dict(end)}}
    with pytest.raises(llm_claudexor.ClaudexorModelError) as raised:
        llm_claudexor._meet_restored_end(target, {"model_role": "main"})
    error = raised.value
    assert error.stream_rejected is True and error.code == "response_rejected"
    assert error.received_status == {"operation_state": "succeeded", "operation_outcome": "completed"}, "the typed receipt rides the met end"
    identity = {"execution_id": "x", "round_id": "x:round:3", "llm_call_id": "llm-new", "round": 3, "attempt": 2, "model": MODEL}
    *_head, facts = llm_claudexor.restored_view(error, identity, {}, None)
    assert facts == {"restored_attempt": True, "recovered_operation_id": "op-9", "rejected_receipt": True}
    live = llm_claudexor.ClaudexorModelError(end["problem"], model_role="main", operation_id="op-10")
    assert llm_claudexor.restored_view(live, identity, {}, None)[3] == {"rejected_receipt": True}, "a live accepted receipt too"
    unaccepted = llm_claudexor.ClaudexorModelError(end["problem"], model_role="main")
    assert llm_claudexor.restored_view(unaccepted, identity, {}, None)[3] == {}, "no accepted operation: nothing to mark"
    other = llm_claudexor.ClaudexorModelError({"code": "provider_failed", "message": "Synthetic."}, operation_id="op-11")
    assert llm_claudexor.restored_view(other, identity, {}, None)[3] == {}, "an ordinary failure adds nothing"
    ctx =SimpleNamespace(task_id="t", exact_model_route=False, route_wait_on_primary=False)
    usage = {"_last_llm_error_kind": "provider_error", "_last_llm_failure": {**facts, "failure_code": "provider_error"}}
    assert fallback_chain_allowed(ctx, "provider_error", None, usage, primary_acting=True) is False


def test_arrivals_bound_to_a_read_stay_with_exactly_that_read():
    """What a round's drains bound to a pending read (``reply_start``, ``owner_input``, ``owner_carried_addressed``) is kept
    for that attempt and operation alone: nothing is derived from a length or a hash, nothing moves to another read. The
    one vocabulary lives beside the writer that binds it (``note_arrivals``)."""
    from ouroboros.loop_output_exhaustion import ARRIVAL_KEYS, kept_arrivals, note_arrivals

    read = {"physical_attempt_id": "attempt-1", "operation_id": "op-1"}
    facts = {"reply_start": 2, "owner_input": True, "owner_carried_addressed": ["chat 1 / carried-words"]}
    saved = {**read, **facts, "paused_receipt_read": True}
    assert set(facts) == set(ARRIVAL_KEYS) and kept_arrivals(saved, read) == facts
    assert kept_arrivals({**saved, "operation_id": "op-2"}, read) == {}
    assert kept_arrivals({**saved, "physical_attempt_id": "attempt-2"}, read) == {}
    assert kept_arrivals(read, read) == {} and kept_arrivals({}, read) == {}
    bound = dict(read)  # every key the writer binds is one the carry keeps
    note_arrivals(bound, place=2, owner_input=True, carried=["chat 1 / carried-words"])
    assert kept_arrivals(bound, read) == facts


def test_a_pause_keeps_the_arrivals_bound_to_its_own_read_for_a_cold_resume(setup, monkeypatch):
    """The Pause that parks a restored read keeps its pending record through ``setdefault``; the record its Resume rebuilds
    (``ready_operation``) carries what the round-top drain bound to that very read, as the reader saw it before parking."""
    from ouroboros import budget_pause, owner_pause
    from ouroboros.artifacts import read_actor_source_bytes

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    root, gateway, task_id = saved.root, saved.gateway, saved.ctx.task_id
    gateway.results = [result(outcome="failed", cash=0.17, knowledge="exact", problem={
        "code": "provider_content_refused", "retryable": False, "message": "Synthetic refusal"})]
    gateway.dispatch = ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    _during_final_receipt(monkeypatch, gateway, lambda: owner_pause.install_fence(root, task_id, request_id="late-pause"))
    drain = {"reply_start": 2, "owner_input": True, "owner_carried_addressed": ["chat 1 / carried-words"]}
    read = loop_transport.read_ready_operation

    def after_the_round_top_drain(ctx, pending):
        pending.update(drain)  # what the round's mail drain bound to this read
        return read(ctx, pending)

    monkeypatch.setattr(loop, "read_ready_operation", after_the_round_top_drain)
    try:
        with model_wait.calendar_scope("2000-01-01T00:00:00+00:00", dispatch_only=True):
            with pytest.raises(budget_pause.BudgetPauseRequested) as held:
                saved.resume()
        state = json.loads(read_actor_source_bytes(root, task_id, held.value.pause["source_ref"]))
    finally:
        budget_pause.end_dispatch_fence(task_id)
    kept = state["usage"]["_pending_transport_outcome"]
    assert kept["paused_receipt_read"] is True and {key: kept[key] for key in drain} == drain
    owed = loop_transport.ready_operation(state, deepcopy(state["usage"]), root, task_id)
    assert (owed["physical_attempt_id"], owed["operation_id"]) == (saved.first["attempt_id"], "op-0")
    assert {key: owed[key] for key in drain} == drain, "a cold Resume reads the same facts, none derived afresh"


@pytest.mark.parametrize("ending", ["answer", "response-only", "stop", "deadline", "refusal-at-stop", "refusal-at-deadline",
                                    "wrong-state", "conflicting-receipt", "reader-error", "budget",
                                    "deadline-with-mail", "deadline-with-pause", "deadline-with-held-answer",
                                    "deadline-with-tools", "deadline-with-commentary", "deadline-with-cut",
                                    "deadline-with-marker", "deadline-missing-marker"])
def test_ready_checkpoint_reads_a_retained_answer_offline(main_call, monkeypatch, ending):
    from ouroboros import owner_mailbox as mailbox
    from ouroboros.observability import update_call_manifest_metadata
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION, derive_loop_outcome
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    ctx, gateway, owner, _events, _decide, _observations = main_call
    root = ctx.tools._ctx.drive_root = ctx.drive_root
    checkpoint = SimpleNamespace(tools=ctx.tools, messages=ctx.messages, accumulated_usage=ctx.accumulated_usage,
        task_id=ctx.task_id, drive_root=root, drive_logs=ctx.drive_logs, round_idx=ctx.round_idx,
        llm_trace={}, tool_schemas=ctx.tool_schemas, owner_msg_seen=set())
    answer = "  The retained answer. 🐍\n"
    if ending == "deadline-with-marker":
        answer += "FINAL ANSWER: 42\n"
    refused = ending in {"refusal-at-stop", "refusal-at-deadline"}
    if refused:
        gateway.results[0] = result(outcome="failed", problem={
            "code": "provider_content_refused", "retryable": False, "message": "Synthetic refusal"})
    else:
        msg = {"role": "assistant", "content": answer}
        if ending == "deadline-with-tools":
            msg["tool_calls"] = result()["message"]["tool_calls"]
        elif ending == "deadline-with-commentary":
            msg["finish_reason"] = "commentary"
        elif ending == "deadline-with-cut":
            gateway.results[0] = result(outcome="incomplete", problem={
                "code": "provider_incomplete", "context": {"reason": "max_output_tokens"}})
        gateway.results[0]["message"] = msg
    write_task_result(root, ctx.task_id, "running")
    task = {"id": ctx.task_id}
    create = gateway.create_model_operation
    if ending == "deadline-with-held-answer":
        loop._finalize_limit_ctx(checkpoint, ctx.tools, checkpoint.llm_trace)
        held = loop._replace_delivery_candidate(ctx.tools, checkpoint, checkpoint.llm_trace, "Earlier held answer.",
                                               control="hold_for_verification")
        ctx.messages.extend([{"role": "assistant", "content": held.full_text},
                             {"role": "user", "content": "[SYSTEM NOTICE]\nThe earlier answer is held."}])

    def before_acceptance(ref, *, idempotency_key, **kwargs):
        _, saved = working_checkpoint.read_working(root, ctx.task_id, 1)
        assert saved["working"]["boundary"] == "ready"
        assert working_checkpoint.attach_recovery(root, task, source_task_id=ctx.task_id, from_attempt=1, cause="worker_crash")
        assert saved["working"]["native_attempt"]["physical_attempt_id"] == idempotency_key
        pending = loop_transport.ready_operation(saved, saved["usage"], root, ctx.task_id)
        assert pending["physical_attempt_id"] == idempotency_key and pending["operation_id"] == ""
        row, = ledger(root)
        assert row["state"] == "dispatched" and row["attempt_id"] == idempotency_key
        return create(ref, idempotency_key=idempotency_key, **kwargs)

    monkeypatch.setattr(gateway, "create_model_operation", before_acceptance)
    working_checkpoint.save_ready(checkpoint)
    checkpoint_file = working_checkpoint.checkpoint_path(root, ctx.task_id, 1)
    ready_bytes = checkpoint_file.read_bytes()
    with working_checkpoint.model_call_scope(checkpoint):
        if refused:
            with pytest.raises(llm_claudexor.ClaudexorRefusalError):
                loop._call_round_model(ctx)
        else:
            # A cut or a commentary reply is no ordinary answer, so the live call returns none; its accepted,
            # settled operation is exactly what the restore below reads.
            live = loop._call_round_model(ctx)[0]
            assert live is None if ending in {"deadline-with-commentary", "deadline-with-cut"} else live["content"] == answer
    assert checkpoint_file.read_bytes() == ready_bytes
    row, = ledger(root)
    _bounded_creates(monkeypatch, gateway, 1)

    def unexpected_send(*_args, **_kwargs):
        pytest.fail("reading an accepted operation dispatched a model request or a tool batch")

    monkeypatch.setattr(ctx.llm, "_chat_remote", unexpected_send)
    monkeypatch.setattr(ctx.llm, "_chat_local", unexpected_send)
    monkeypatch.setattr(loop, "handle_tool_calls", unexpected_send)
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: pytest.fail("the answer is already retained"))
    fresh = checkpoint_registry(root, monkeypatch, ctx.task_id, 2)
    owner.tool_context = fresh._ctx
    fresh._ctx.working_recovery = task["_working_recovery"]
    if ending in {"deadline-with-marker", "deadline-missing-marker"}:
        fresh._ctx.task_metadata["task_contract"] = {"answer_protocol": "final_answer_line"}
    _, saved = working_checkpoint.read_working(root, ctx.task_id, 1)
    state = {**saved, "_working_handoff": task["_working_recovery"]}
    if ending == "response-only":
        update_call_manifest_metadata(root, task_id=ctx.task_id, call_id=f"{row['attempt_id']}_model_request",
                                      metadata={"operation_id": None})
    pending = loop_transport.ready_operation(state, {}, root, ctx.task_id)
    assert pending["physical_attempt_id"] == row["attempt_id"] and pending["model_role"] == "main"
    assert loop_transport.ready_operation(state, {}, root, "another-task") is None
    assert loop_transport.ready_operation({**state, "round_idx": ctx.round_idx + 1}, {}, root, ctx.task_id) is None
    assert loop_transport.ready_operation(state, {"execution_id": "other-execution"}, root, ctx.task_id) is None
    assert loop_transport.ready_operation({**state, "context_model_role": "fallback:0"}, {}, root, ctx.task_id) is None
    working = state["working"]
    assert loop_transport.ready_operation({**state, "working": {**working, "seq": working["seq"] + 1}}, {}, root, ctx.task_id) is None
    assert llm_claudexor.recover_chat_reply(root, "another-task", pending, None) is None
    for change in ({"model_role": "fallback:0"}, {"model": "different"}, {"operation_id": "wrong-operation"}):
        assert llm_claudexor.recover_chat_reply(root, ctx.task_id, {**pending, **change}, None) is None
    if ending == "wrong-state":
        update_call_manifest_metadata(root, task_id=ctx.task_id, call_id=f"{row['attempt_id']}_model_response",
                                      metadata={"operation_state": "interrupted"})
        with pytest.raises(llm_claudexor.ClaudexorModelError) as undeliverable:
            llm_claudexor.recover_chat_reply(root, ctx.task_id, pending, None)
        assert undeliverable.value.code == "response_rejected" and undeliverable.value.operation_id == pending["operation_id"]
        assert undeliverable.value.received_status == {"operation_state": "interrupted", "operation_outcome": "completed"}
        assert ledger(root)[0]["state"] == "settled" and len(gateway.creates) == len(gateway.uploads) == 1
        return  # a settled known end: its round's own call meets it (test_a_received_failure_is_met_by_its_rounds_own_call)
    elif ending == "conflicting-receipt":
        update_call_manifest_metadata(root, task_id=ctx.task_id, call_id=f"{row['attempt_id']}_model_response",
                                      metadata={"operation_id": "a-different-operation"})
    reads = []
    recover = llm_claudexor._read_model_receipt

    def read_original(root, row, *, invocation):
        reads.append(invocation.operation_id)
        if ending == "reader-error":
            raise RuntimeError("unexpected reader error")
        if ending == "budget":
            from ouroboros.usage_accounting import BudgetExceeded
            raise BudgetExceeded("No new reservation", limit_scope="root")
        receipt = recover(root, row, invocation=invocation)
        if refused or ending in {"stop", "deadline-with-mail"}:
            mailbox.write_owner_message(root, "Keep the ordinary mail.", ctx.task_id, msg_id="words")
        if ending == "deadline-with-pause":
            from ouroboros.owner_pause import install_fence
            install_fence(root, ctx.task_id, request_id="late-pause")
        if ending.startswith("deadline") or ending == "refusal-at-deadline":
            fresh._ctx.task_metadata["deadline_at"] = "2000-01-01T00:00:00+00:00"
        elif ending in {"stop", "refusal-at-stop"}:
            mailbox.write_owner_message(root, REASON_OWNER_STOPPED_DIRECT_TURN, ctx.task_id,
                                        msg_id="stop", kind=mailbox.KIND_FINALIZE_NOW)
        return receipt

    monkeypatch.setattr(llm_claudexor, "_read_model_receipt", read_original)

    def resume():
        return loop.run_llm_loop(messages=[], tools=fresh, llm=ctx.llm, drive_logs=root / "logs",
            emit_progress=lambda *_a, **_kw: None, incoming_messages=queue.Queue(), task_id=ctx.task_id, drive_root=root)

    if ending == "reader-error":
        with pytest.raises(RuntimeError, match="unexpected reader error"):
            resume()
    elif ending == "budget":
        from ouroboros import budget_pause
        from ouroboros.artifacts import read_actor_source_bytes
        try:
            with pytest.raises(budget_pause.BudgetPauseRequested) as paused:
                resume()
            source = json.loads(read_actor_source_bytes(root, ctx.task_id, paused.value.pause["source_ref"]))
            assert source["resume_point"]["budget_tail"] == "no_tool"
            owed = loop_transport.ready_operation(source, source["usage"], root, ctx.task_id)
            assert (owed["physical_attempt_id"], owed["operation_id"]) == (row["attempt_id"], reads[0])
        finally:
            budget_pause.end_dispatch_fence(ctx.task_id)
    else:
        text, usage, trace = resume()
    assert reads == ([] if ending == "conflicting-receipt" else [gateway.detail(0)["id"]])
    assert not checkpoint_file.exists() and not checkpoint_file.with_suffix(".operation.json").exists()
    assert len(gateway.creates) == len(gateway.uploads) == len(gateway.acks) == 1
    assert ledger(root)[0]["attempt_id"] == row["attempt_id"] and ledger(root)[0]["state"] == "settled"
    if ending in {"reader-error", "budget"}:
        return
    if ending == "conflicting-receipt":
        assert usage["_pending_transport_outcome"]["operation_id"] == ""
        assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    elif refused:
        failure = usage["_last_llm_failure"]
        assert failure["failure_code"] == "provider_content_refused" and failure["attempt_custody_state"] == "settled"
        assert usage["reason_code"] == ("deadline_local" if ending == "refusal-at-deadline" else REASON_OWNER_REQUESTED_FINALIZATION)
        assert failure["operation_id"] == reads[0] and failure["physical_attempt_id"] == row["attempt_id"]
        if ending == "refusal-at-stop":
            assert fresh._ctx._skip_post_task_synthesis
        assert any("Keep the ordinary mail." in str(message.get("content")) for message in fresh._ctx.messages)
        assert [m["tool_call_id"] for m in fresh._ctx.messages if m.get("role") == "tool"] == ["a", "b"]
    else:
        meta = usage["_last_llm_call_meta"]
        from ouroboros.task_finalization import model_execution_projection
        assert model_execution_projection(usage)["used_model"] == row["model"]
        assert meta["llm_call_id"] == ctx.accumulated_usage["_last_llm_call_meta"]["llm_call_id"]
        assert meta["response_ref"] == ctx.accumulated_usage["_last_llm_call_meta"]["response_ref"]
        assert (meta["physical_attempt_id"], meta["operation_id"], meta["model_role"]) == (row["attempt_id"], reads[0], "main")
        if ending in {"answer", "response-only"}:
            assert text == answer
        else:
            assert "_pending_transport_outcome" not in usage
            assert usage["reason_code"] == ("deadline_local" if ending.startswith("deadline") else REASON_OWNER_REQUESTED_FINALIZATION)
            assert [m["tool_call_id"] for m in fresh._ctx.messages if m.get("role") == "tool"] == ["a", "b"]
            if ending in {"deadline", "deadline-with-marker"}:
                assert text == answer and usage["terminal_origin"] == "model_final"
                assert trace["forced_finalization"]["source"] == "model_wait_received_reply"
                assert trace["forced_finalization"]["control_reason"] == "deadline"
                outcome = derive_loop_outcome(text, usage, trace)
                assert outcome["outcome_axes"]["execution"]["status"] == "best_effort"
                assert outcome["final_answer"] == ("42" if ending == "deadline-with-marker" else "")
            else:
                assert text != answer and trace["forced_finalization"]["source"] != "model_wait_received_reply"
                if ending in {"deadline-with-tools", "deadline-with-commentary", "deadline-with-cut",
                              "deadline-missing-marker", "deadline-with-mail", "deadline-with-pause"}:
                    # No usable words of this call and no candidate: the previous
                    # turn's ordinary line is salvaged model bytes and the origin
                    # says so; the host's own phrase stays host_notice and this
                    # call's proven reply model_final.
                    assert text == result()["message"]["content"]
                    assert usage["terminal_origin"] == "host_salvage"
                if ending == "deadline-with-held-answer":
                    assert text == held.full_text and trace["delivery_candidate"]["content_sha256"] == held.content_sha256
                elif ending == "deadline-with-mail":
                    assert any("Keep the ordinary mail." in str(m.get("content")) for m in fresh._ctx.messages)
                elif ending == "stop":
                    assert fresh._ctx._skip_post_task_synthesis
                    assert usage["terminal_origin"] == "host_notice"


def _main_reply_run(setup, monkeypatch, restored, *, owner_control=None):
    """Use the ordinary loop, or lose its first receiver and read that exact accepted operation. ``owner_control``
    stands for the live wait's own control reader (none by default)."""
    if restored:
        saved = _saved_native_round(setup, monkeypatch, "sidecar")
        return saved.ctx, saved.resume
    root, _gateway, client = setup
    task_id = "task-one"
    tools = checkpoint_registry(root, monkeypatch, task_id, 1)
    tools._ctx.active_model = MODEL
    monkeypatch.setenv("OUROBOROS_MODEL", MODEL)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::unused")
    write_task_result(root, task_id, "running")

    def run():
        with model_wait.task_model_wait_scope(task={"id": task_id, "_attempt": 1}, drive_root=root,
                                             event_queue=None, worker_slot_held=True,
                                             owner_control=owner_control or (lambda: None)) as owner:
            owner.tool_context = tools._ctx
            return loop.run_llm_loop([{"role": "user", "content": "Continue the recorded task."}], tools, client,
                root / "logs", lambda *_a, **_kw: None, queue.Queue(), task_id=task_id, drive_root=root)

    return tools._ctx, run


@pytest.mark.parametrize("restored", [False, True], ids=["live", "restored"])
def test_empty_native_cut_rejoins_the_length_policy_without_replaying_the_attempt(setup, monkeypatch, restored):
    root, gateway, _client = setup
    _ctx, run = _main_reply_run(setup, monkeypatch, restored)
    cut = result(outcome="incomplete", cash=0.17, knowledge="exact",
                 problem={"code": "provider_incomplete", "context": {"reason": "max_output_tokens"}})
    cut["message"] = {"role": "assistant", "content": " \n"}
    cut["usage"].update(output_tokens=11, reasoning_tokens=11)
    answer = result(cash=0.23, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "Answered on the informed turn."}
    gateway.results, gateway.dispatch = [cut, answer], ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 2)
    text, usage, _ = run()

    first, informed = attempt_rows_in_start_order(root)
    assert first["state"] == informed["state"] == "settled"
    assert [first["attempt_id"], informed["attempt_id"]] == gateway.creates
    assert first["attempt_id"] != informed["attempt_id"] and len(gateway.uploads) == 2
    assert gateway.reads == (["op-0"] if restored else [])
    assert [op for op, _ in gateway.acks] == ["op-0", "op-1"]
    assert (first["cost_usd"], informed["cost_usd"], usage["cost"]) == pytest.approx((0.17, 0.23, 0.40))
    assert usage["completion_tokens"] == 18 and usage["rounds"] == 1
    assert "_pending_transport_outcome" not in usage
    second_send = str(gateway.uploads[1][0]["messages"])
    assert "The provider reported 11 reasoning tokens for it." in second_send
    assert "The reply allowance sent was" not in second_send
    assert text == answer["message"]["content"]


@pytest.mark.parametrize("restored", [False, True], ids=["live", "restored"])
@pytest.mark.parametrize("case", ["plain", "text-blocks", "mixed", "ordinary-marker", "refusal-marker",
                                  "tools", "ordinary-control", "mail", "hold", "stop", "cut", "carried"])
def test_main_deadline_uses_only_its_completed_received_reply(setup, monkeypatch, restored, case):
    from ouroboros import owner_mailbox as mailbox
    from ouroboros.outcomes import selected_final_answer
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
    from tests._model_replies import MIXED_REFUSAL_PARTS, MIXED_REFUSAL_TEXT

    root, gateway, _client = setup
    gateway.operation_catalog = [{"method": "POST", "path": "/v2/model-operations", "parameters": [
        {"name": "typedRefusals", "location": "query", "enum": ["true"]}]}]
    ctx, run = _main_reply_run(setup, monkeypatch, restored)
    expected = "  First.\nLast. 🐍\n"
    message = {"role": "assistant", "content": expected}
    if case == "text-blocks":
        message["content"] = [{"type": "text", "text": "  First.\n"}, {"type": "text", "text": "Last. 🐍\n"}]
    elif case in {"mixed", "ordinary-marker", "refusal-marker"}:
        message["content"] = deepcopy(MIXED_REFUSAL_PARTS)
        expected = MIXED_REFUSAL_TEXT
        if case != "mixed":
            ctx.task_metadata["task_contract"] = {"answer_protocol": "final_answer_line"}
        if case == "ordinary-marker":
            message["content"][-1]["text"] += "\nFINAL ANSWER: 42\n"
            expected += "\nFINAL ANSWER: 42\n"
    elif case == "tools":
        message["tool_calls"] = result()["message"]["tool_calls"]
    elif case == "ordinary-control":
        message["content"] = expected = '{"action":"finish","answer":"Not selected."}'
    terminal = ({"outcome": "incomplete", "problem": {"code": "provider_incomplete",
                "context": {"reason": "max_output_tokens"}}} if case == "cut" else {})
    gateway.results, gateway.dispatch = [{**result(cash=0.19, knowledge="exact", **terminal), "message": message}], ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("a deadline never dispatches tools"))
    read_body = gateway.get_model_result

    def deadline_after_receive(*args, **kwargs):
        reply = read_body(*args, **kwargs)
        if case == "mail":
            mailbox.write_owner_message(root, "Change the example.", ctx.task_id, msg_id="new-words")
        elif case == "hold":
            ctx._delivery_control_required = True
        elif case == "stop":
            mailbox.write_owner_message(root, REASON_OWNER_STOPPED_DIRECT_TURN, ctx.task_id,
                                        msg_id="stop", kind=mailbox.KIND_FINALIZE_NOW)
        ctx.task_metadata["deadline_at"] = "2000-01-01T00:00:00+00:00"
        return reply

    monkeypatch.setattr(gateway, "get_model_result", deadline_after_receive)
    if case == "carried":  # drained at the round top: the live request carries it, the restored one never saw it
        _carry_to_work(root, ctx.task_id, _carried_quote(root))
    text, usage, trace = run()

    [settled] = ledger(root)
    assert settled["state"] == "settled" and settled["cost_usd"] == 0.19
    assert len(gateway.creates) == len(gateway.uploads) == 1
    if case == "carried" and restored:  # the row moved no owner hash; the read still bound it, so no old prose answers
        assert text != expected and usage["cost"] == pytest.approx(0.19)  # its receipt counted once, its words taken by none
    if case in {"plain", "text-blocks", "mixed", "ordinary-marker"} or case == "carried" and not restored:
        assert text == expected and usage["terminal_origin"] == "model_final"
        assert trace["forced_finalization"]["source"] == "model_wait_received_reply"
        assert trace["forced_finalization"]["operation_id"] == "op-0"
        assert usage["cost"] == 0.19 and usage["completion_tokens"] == 7
        assert selected_final_answer(text, trace) == ("42" if case == "ordinary-marker" else "")
    else:
        assert text != expected and trace["forced_finalization"]["source"] != "model_wait_received_reply"
        if case == "stop":
            # The Stop terminal carries the settled receipt's cost and tokens too:
            # the summary matches the ledger row in both branches, so a restart
            # reconciles nothing back in and no second request was needed.
            assert (usage["cost"], usage["completion_tokens"]) == (settled["cost_usd"], 7)


def test_main_deadline_does_not_take_the_preparing_helpers_reply(setup, monkeypatch):
    from ouroboros import loop_llm_call, usage_accounting

    _ctx, run = _main_reply_run(setup, monkeypatch, False)
    capture = usage_accounting.PhysicalAttemptCapture("helper-attempt", MODEL, "claudexor", "settled", "opaque")
    response = "A caption is not the requested answer."

    def interrupted_helper(*_args, **_kwargs):
        error = ModelWaitInterrupted("deadline", role="vision")
        error.operation_id, error.physical_attempt_capture = "helper-op", capture
        error.model_result = {"outcome": "completed", "message": {"role": "assistant", "content": response}}
        error.usage = {"claudexor": {"operation_id": "helper-op", "model_role": "vision", "outcome": "completed",
                                     "operation_state": "succeeded", "dispatch_state": "response_received"}}
        raise error

    monkeypatch.setattr(loop_llm_call, "_prepare_main_messages", interrupted_helper)
    text, _usage, trace = run()
    assert text != response and trace["forced_finalization"]["source"] != "model_wait_received_reply"
    root, gateway, _client = setup
    assert not gateway.creates and not ledger(root)


@pytest.mark.parametrize("case", ["same-input", "new-owner-words", "no-ready-checkpoint", "foreign-sidecar",
                                  "own-manifest-mismatch"])
def test_a_withdrawn_wrap_up_reads_the_settled_reply_and_never_sends_it_again(setup, monkeypatch, case):
    """Wrap up reaches this Main call once its own reply has arrived, and is withdrawn before the loop reads it. The
    redial reads that exact attempt from its ready checkpoint instead of sending the request again: unchanged owner
    input is answered by it. Owner words that came with the withdrawal keep it as words before them and get a request
    of their own. With no address for this very attempt (no ready checkpoint, a same-task sidecar naming another
    attempt, or this attempt's own request manifest naming another operation) the round ends on the no-resend rail:
    no foreign attempt enters this round's usage, and the settled receipt is counted once beside that unknown outcome."""
    from ouroboros import owner_mailbox as mailbox
    from ouroboros.observability import update_call_manifest_metadata

    root, gateway, _client = setup
    exact, later = "The exact settled reply.", "The later words are answered."
    pressed = []

    def wrap_up_then_withdrawn():  # the live wait's control reader: Wrap up once the reply is in, then gone
        if gateway.acks and not pressed:
            pressed.append(len(gateway.creates))
            if case == "new-owner-words":  # they reach the control's own drain, before any redial
                mailbox.write_owner_message(root, "Also name the second fixture.", "task-one", msg_id="later")
            elif case == "foreign-sidecar":  # the round's ready sidecar now names another attempt of the same task
                sidecar = working_checkpoint.checkpoint_path(root, "task-one", 1).with_suffix(".operation.json")
                saved = json.loads(sidecar.read_text(encoding="utf-8"))
                sidecar.write_text(json.dumps({**saved, "physical_attempt_id": "other-attempt"}), encoding="utf-8")
            elif case == "own-manifest-mismatch":  # this attempt's request manifest names another operation
                update_call_manifest_metadata(root, task_id="task-one", call_id=f"{gateway.creates[0]}_model_request",
                                              metadata={"operation_id": "op-foreign"})
            return "finalize_requested"
        return None

    _ctx, run = _main_reply_run(setup, monkeypatch, False, owner_control=wrap_up_then_withdrawn)
    gateway.results = [{**result(cash=0.19, knowledge="exact"), "message": {"role": "assistant", "content": exact}},
                       {**result(cash=0.23, knowledge="exact"), "message": {"role": "assistant", "content": later}}]
    gateway.dispatch = ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 2 if case == "new-owner-words" else 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("a plain reply runs no tool"))
    if case == "no-ready-checkpoint":
        monkeypatch.setattr(working_checkpoint, "read_working", lambda *_a, **_k: (b"", {}))
    text, usage, trace = run()

    assert pressed == [1]  # the control met the first call, after its receipt was acknowledged
    assert ledger(root)[0]["state"] == "settled"
    if case == "same-input":
        assert text == exact and len(gateway.creates) == 1
        assert usage["cost"] == pytest.approx(0.19) and "_pending_transport_outcome" not in usage
    elif case == "new-owner-words":
        assert text == later and len(gateway.creates) == 2
        assert usage["cost"] == pytest.approx(0.42)  # the read reply once, the new request once
        sent = [json.dumps(row.get("content"), ensure_ascii=False) for row in gateway.uploads[1][0]["messages"]]
        reply_at = next(index for index, row in enumerate(sent) if exact in row)
        words_at = next(index for index, row in enumerate(sent) if "Also name the second fixture." in row)
        assert reply_at < words_at  # the old reply answers only what it saw: it stays before the new words
    else:
        assert len(gateway.creates) == 1 and usage["cost"] == pytest.approx(0.19)  # the settled receipt, once
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
        assert "other-attempt" not in usage.get("ledger_attempt_ids", [])
        unknown = usage.get("_pending_transport_outcome") or {}
        assert unknown.get("physical_attempt_id") in (None, gateway.creates[0])  # never the sidecar's attempt
        if case == "own-manifest-mismatch":  # the owner's own record of this attempt keeps its outcome unknown
            assert unknown["physical_attempt_id"] == gateway.creates[0] and unknown["operation_id"] == ""


@pytest.mark.parametrize("shape", ["deadline", "no-reply", "foreign-sidecar"])
def test_only_this_calls_own_attempt_binds_a_withdrawn_read_before_any_effect(setup, shape):
    """Another interruption, or one that received no reply of its own, keeps the ordinary redial and touches nothing.
    A real same-task ready sidecar naming another attempt is refused by the real owner (``ready_operation``) before it
    changes this round's usage: its own outcome (infra_failed here) stays, no foreign pending outcome or ledger id
    enters, and this attempt ends unknown, its settled receipt counted once beside that outcome."""
    from ouroboros import loop_output_exhaustion, usage_accounting
    from ouroboros.loop_round_limits import _RoundLimitContext
    from ouroboros.tools.registry import ToolRegistry

    root, _gateway, _client = setup
    root.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry(repo_dir=root, drive_root=root)
    tools._ctx.task_attempt = 1
    usage = {"execution_id": "exec", "execution_status": "infra_failed", "ledger_attempt_ids": ["earlier"]}
    ctx = _RoundLimitContext([], None, MODEL, "high", 1, root / "logs", "task-one", 1, None, usage, "", False, 3,
                             drive_root=root, tools=tools)
    ready = working_checkpoint.checkpoint_path(root, "task-one", 1, create=True)
    ready.write_text(json.dumps({"task_id": "task-one", "task_attempt": 1, "round_idx": 1, "messages": [],
                                 "working": {"boundary": "ready", "seq": 3}, "usage": {}}), encoding="utf-8")
    ready.with_suffix(".operation.json").write_text(json.dumps({  # what an earlier send of this round could leave
        "physical_attempt_id": "other-attempt", "seq": 3, "round": 1, "llm_call_id": "other-call", "execution_id": "exec",
        "round_id": "exec:round:1", "model_role": "main", "model": MODEL}), encoding="utf-8")
    error = ModelWaitInterrupted("deadline" if shape == "deadline" else "finalize_requested", role="main")
    error.physical_attempt_capture = usage_accounting.PhysicalAttemptCapture(
        "attempt-1", MODEL, "claudexor", "settled", "opaque")
    error.usage = {"cost": 0.19, "prompt_tokens": 3, "completion_tokens": 2}
    if shape != "no-reply":
        error.received_main_reply = ({"role": "assistant", "content": "A reply."}, error.usage, "source")
    before = deepcopy(usage)
    bound = loop_output_exhaustion.withdrawn_read(ctx, error)
    if shape != "foreign-sidecar":
        assert bound is None and usage == before
        return
    assert bound == {"physical_attempt_id": "attempt-1", "operation_id": "", "model_role": "main", "round": 1}
    assert usage["execution_status"] == "infra_failed" and "_pending_transport_outcome" not in usage
    assert usage["ledger_attempt_ids"] == ["earlier", "attempt-1"]  # its own receipt, never the sidecar's attempt
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown" and usage["cost"] == pytest.approx(0.19)


def _place_of(messages, text):
    sent = [json.dumps(row.get("content"), ensure_ascii=False) for row in messages]
    return next(index for index, row in enumerate(sent) if text in row)


@pytest.mark.parametrize("received", ["failure", "no-message"])
def test_a_withdrawn_wrap_up_reads_a_received_failure_once_and_never_sends_again(setup, monkeypatch, received):
    """This Main call's own receipt settled carrying usage but no model result: a failed outcome, or a completed one
    without a message. A Wrap up that met it and is withdrawn before the loop drains it is no license to send the round
    again: the redial reads that exact attempt from its ready checkpoint and consumes what it received, once. The round's
    own call meets that end with no configured route to walk: neither a same-model repeat (a whole response with no usable
    message stays unusable) nor a summary request is bought over it, and its row names that attempt and operation."""
    root, gateway, _client = setup
    pressed = []

    def wrap_up_then_withdrawn():  # the live wait's control reader: Wrap up once the receipt is in, then gone
        if gateway.acks and not pressed:
            pressed.append(len(gateway.creates))
            return "finalize_requested"
        return None

    _ctx, run = _main_reply_run(setup, monkeypatch, False, owner_control=wrap_up_then_withdrawn)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    first = ({**result(outcome="failed", cash=0.17, knowledge="exact", problem={
        "code": "invalid_request", "message": "Invalid argument", "context": {"httpStatus": 400}}), "message": None}
        if received == "failure" else {**result(cash=0.17, knowledge="exact"), "message": None})
    fresh = {**result(cash=0.23, knowledge="exact"), "message": {"role": "assistant", "content": "A second send."}}
    gateway.results, gateway.dispatch = [first, fresh], ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("a received failure runs no tool"))
    text, usage, trace = run()

    [row] = ledger(root)
    assert pressed == [1] and len(gateway.creates) == len(gateway.uploads) == 1  # never generated twice
    assert row["state"] == "settled" and row["cost_usd"] == 0.17 and "A second send." not in text
    assert row["attempt_id"] == gateway.creates[0] and usage["cost"] == pytest.approx(0.17)  # its receipt, counted once
    failure = usage["_last_llm_failure"]  # the round's own call met what that attempt received: its row, not a new one's
    assert (failure["physical_attempt_id"], failure["operation_id"], failure["recovered_operation_id"]) == (
        row["attempt_id"], "op-0", "op-0") and failure["restored_attempt"] is True
    assert failure["attempt_custody_state"] == "settled"
    assert failure["failure_code"] == ("bad_request" if received == "failure" else "provider_error")
    assert trace["forced_finalization"]["source"] == "recovered_operation_no_resend"  # no summary bought over it
    assert "_pending_transport_outcome" not in usage


def test_a_withdrawn_wrap_up_keeps_its_reply_as_words_when_a_carried_row_came_with_it(setup, monkeypatch):
    """A row of my human's carried to this work reaches the control's own drain with the withdrawal. It enters no
    corpus, so the owner hash the send bound is unchanged, yet that send never saw it: the settled reply is read once
    and kept as words before that row, none of its tools runs, and the row gets a request of its own."""
    root, gateway, _client = setup
    exact, later = "The exact settled reply.", "The carried words are answered."
    quote, pressed = _carried_quote(root), []

    def wrap_up_then_withdrawn():
        if gateway.acks and not pressed:
            pressed.append(len(gateway.creates))
            _carry_to_work(root, "task-one", quote)  # it reaches the control's own drain, before any redial
            return "finalize_requested"
        return None

    _ctx, run = _main_reply_run(setup, monkeypatch, False, owner_control=wrap_up_then_withdrawn)
    gateway.results = [{**result(cash=0.19, knowledge="exact"), "message": {"role": "assistant", "content": exact,
                                                                             "tool_calls": result()["message"]["tool_calls"]}},
                       {**result(cash=0.23, knowledge="exact"), "message": {"role": "assistant", "content": later}}]
    gateway.dispatch = ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 2)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("a stale reply's tools never run"))
    text, usage, _trace = run()

    assert pressed == [1] and text == later and len(gateway.creates) == len(gateway.uploads) == 2
    assert usage["cost"] == pytest.approx(0.42)  # the read reply once, the new request once
    sent = gateway.uploads[1][0]["messages"]
    assert _place_of(sent, exact) < _place_of(sent, CARRIED_WORDS)  # it answers only what it saw: before the row
    assert not any(row.get("role") == "tool" or row.get("tool_calls") for row in sent)


@pytest.mark.parametrize("arrival", ["same-input", "new-owner-words", "carried-addressed", "owner-pause"])
def test_a_withdrawn_wrap_up_leaves_a_restored_reply_that_reads_own_result(setup, monkeypatch, arrival):
    """A restored read counts its reply, and only then meets a Wrap up that the control rails find withdrawn when they
    drain it. The read is read once more, offline and never counted again: its reply finishes unchanged owner input as an
    ordinary reply does, and words or a carried row that came with the withdrawal keep it as words before them, its
    tools unrun, for a request of their own. With the owner's Pause closed at that moment the second read takes its own
    Pause rail: the reply is not adopted, no tool runs, nothing is sent, and the exact pause still owes that read, its
    receipt counted once."""
    from ouroboros import budget_pause, owner_mailbox, owner_pause
    from ouroboros.artifacts import read_actor_source_bytes

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, attempt, task_id = saved.gateway, saved.first["attempt_id"], saved.ctx.task_id
    old, new = result(cash=0.17, knowledge="exact"), result(cash=0.05, knowledge="exact")
    old["message"] = {"role": "assistant", "content": "The recovered answer."}
    if arrival != "same-input":
        old["message"]["tool_calls"] = result()["message"]["tool_calls"]
    new["message"] = {"role": "assistant", "content": "The answer to the new words."}
    gateway.results, gateway.dispatch = [old, new], ["response_received", "response_received"]
    answered_again = arrival in {"new-owner-words", "carried-addressed"}
    _bounded_creates(monkeypatch, gateway, 2 if answered_again else 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("a stale reply's tools never run"))
    quote, pressed = _carried_quote(saved.root) if arrival == "carried-addressed" else None, []

    def counted_then_withdrawn():  # the restored reader has counted the reply: only its last control check meets this
        if (getattr(saved.ctx, "_accumulated_usage", None) or {}).get("cost") and not pressed:
            pressed.append(True)
            if arrival == "new-owner-words":
                assert owner_mailbox.write_owner_message(saved.root, "Please also check the second file.", task_id,
                                                         msg_id="late-words")
            elif arrival == "carried-addressed":
                _carry_to_work(saved.root, task_id, quote)
            elif arrival == "owner-pause":
                owner_pause.install_fence(saved.root, task_id, request_id="owner-pause")
            return "finalize_requested"
        return None

    scope = model_wait.task_model_wait_scope  # the restored wait's own control reader is this one
    monkeypatch.setattr(model_wait, "task_model_wait_scope",
                        lambda *args, **kwargs: scope(*args, **{**kwargs, "owner_control": counted_then_withdrawn}))
    if arrival == "owner-pause":
        try:
            with pytest.raises(budget_pause.BudgetPauseRequested) as held:
                saved.resume()
            state = json.loads(read_actor_source_bytes(saved.root, task_id, held.value.pause["source_ref"]))
        finally:
            budget_pause.end_dispatch_fence(task_id)
        assert held.value.pause["rail"] == budget_pause.RAIL_OWNER_PAUSE and pressed == [True]
        row, = ledger(saved.root)  # nothing was sent, and the one engine read was not repeated
        assert (row["attempt_id"], row["state"], row["cost_usd"]) == (attempt, "settled", 0.17)
        assert gateway.creates == [attempt] and gateway.reads == ["op-0"]
        owed = loop_transport.ready_operation(state, deepcopy(state["usage"]), saved.root, task_id)
        assert (owed["physical_attempt_id"], owed["operation_id"]) == (attempt, "op-0"), "the exact pause owes this read"
        assert state["usage"]["cost"] == pytest.approx(0.17), "counted once by its first reader, adopted by nothing"
        assert attempt in state["usage"]["_counted_error_attempts"], "a Resume's read applies it without counting it again"
        assert owner_pause.read_fence(saved.root, task_id)["state"] == "requested"
        return
    text, usage, _ = saved.resume()

    first, *later = ledger(saved.root)
    assert pressed == [True] and first["attempt_id"] == attempt and first["state"] == "settled"
    assert gateway.creates[0] == attempt and gateway.reads == ["op-0"]  # one read of the old operation, never another
    if arrival == "same-input":
        assert text == "The recovered answer." and not later and len(gateway.creates) == 1
        assert usage["cost"] == pytest.approx(0.17) and usage["rounds"] == 1
    else:
        second, = later
        assert second["attempt_id"] != attempt and text == "The answer to the new words."
        assert usage["cost"] == pytest.approx(0.22)
        sent = gateway.uploads[1][0]["messages"]
        arrived = CARRIED_WORDS if arrival == "carried-addressed" else "Please also check the second file."
        assert sent[_place_of(sent, "The recovered answer.")]["role"] == "assistant"
        assert _place_of(sent, "The recovered answer.") < _place_of(sent, arrived)
        assert not any(row.get("role") == "tool" or row.get("tool_calls") for row in sent)


@pytest.mark.parametrize("received", ["failure", "empty-cut"])
def test_a_withdrawn_wrap_up_returns_a_restored_failure_or_cut_to_its_policy_once(setup, monkeypatch, received):
    """A restored read applies what it received, a refused request or an empty output cut, and only then meets a Wrap up
    the control rails find withdrawn when they drain it. That ends nothing: the read is read once more, offline and
    counted once, and what it received takes its ordinary policy exactly once, as it would have without the control: the
    refusal is met by the round's own call and ends on the no-resend terminal, the cut buys its one informed turn."""
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, attempt = saved.gateway, saved.first["attempt_id"]
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    if received == "failure":
        old = _RECEIVED_FAILURES["bad-input"][0]()
    else:
        old = result(outcome="incomplete", cash=0.17, knowledge="exact",
                     problem={"code": "provider_incomplete", "context": {"reason": "max_output_tokens"}})
        old["message"] = {"role": "assistant", "content": " \n"}
        old["usage"].update(output_tokens=11, reasoning_tokens=11)
    answer = result(cash=0.23, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "Answered on the informed turn."}
    gateway.results, gateway.dispatch = [old, answer], ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 1 if received == "failure" else 2)
    pressed = []

    def applied_then_withdrawn():  # the reader applied what it received: only its last control check meets this
        if (getattr(saved.ctx, "_accumulated_usage", None) or {}).get("cost") and not pressed:
            pressed.append(True)
            return "finalize_requested"
        return None

    scope = model_wait.task_model_wait_scope  # the restored wait's own control reader is this one
    monkeypatch.setattr(model_wait, "task_model_wait_scope",
                        lambda *args, **kwargs: scope(*args, **{**kwargs, "owner_control": applied_then_withdrawn}))
    text, usage, trace = saved.resume()

    first, *later = ledger(saved.root)
    assert pressed == [True] and first["attempt_id"] == attempt == gateway.creates[0] and first["state"] == "settled"
    assert gateway.reads == ["op-0"], "one engine read; the second read is its retained receipt"
    assert usage.get("reason_code") != "finalization_grace", "a withdrawn control ends nothing"
    if received == "failure":
        assert not later and usage["cost"] == pytest.approx(0.17), "counted once"
        failure = usage["_last_llm_failure"]
        assert (failure["failure_code"], failure["physical_attempt_id"], failure["recovered_operation_id"]) == (
            "bad_request", attempt, "op-0")
        assert trace["forced_finalization"]["source"] == "recovered_operation_no_resend"
    else:
        second, = later
        assert text == answer["message"]["content"] and gateway.creates == [attempt, second["attempt_id"]]
        assert usage["cost"] == pytest.approx(0.40) and usage["rounds"] == 1
        assert "The provider reported 11 reasoning tokens for it." in str(gateway.uploads[1][0]["messages"])


@pytest.mark.parametrize("arrival", ["carried-addressed", "legacy-owner-words", "legacy-unchanged",
                                     "legacy-carried-context", "legacy-hurry"])
def test_a_restored_reply_answers_only_what_its_send_provably_saw(setup, monkeypatch, arrival):
    """The round's drain runs before the restored read. A row carried to this work leaves the owner hash the send bound
    unchanged by design, and an older ready state bound no hash at all; neither proves the old send saw what that drain
    delivered. The reply then stays words before it, its tools unrun, and the arrival gets its own request. With no
    such arrival, the reply of a send that bound no hash still finishes as before: a row carried here that went to
    another work, or a Hurry press, is shown typed as my human's but is no arrival for this read."""
    from ouroboros import owner_mailbox

    saved = _saved_native_round(setup, monkeypatch, "sidecar" if arrival == "carried-addressed" else "unmarked")
    gateway, attempt, task_id = saved.gateway, saved.first["attempt_id"], saved.ctx.task_id
    stale = arrival in {"carried-addressed", "legacy-owner-words"}
    old, new = result(cash=0.17, knowledge="exact"), result(cash=0.05, knowledge="exact")
    old["message"] = {"role": "assistant", "content": "The recovered answer."}
    if stale:
        old["message"]["tool_calls"] = result()["message"]["tool_calls"]
    new["message"] = {"role": "assistant", "content": "The answer to the new words."}
    gateway.results, gateway.dispatch = [old, new], ["response_received", "response_received"]
    _bounded_creates(monkeypatch, gateway, 2 if stale else 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("a stale reply's tools never run"))
    if arrival in {"carried-addressed", "legacy-carried-context"}:
        _carry_to_work(saved.root, task_id, _carried_quote(saved.root),
                       went_to="another-work" if arrival == "legacy-carried-context" else "")
    elif arrival == "legacy-owner-words":
        assert owner_mailbox.write_owner_message(saved.root, "Please also check the second file.", task_id,
                                                 msg_id="late-words")
    elif arrival == "legacy-hurry":
        assert owner_mailbox.write_owner_message(saved.root, "owner_hurry", task_id, msg_id="hurry:late",
                                                 kind=owner_mailbox.KIND_HURRY)
    text, usage, _ = saved.resume()

    first, *later = ledger(saved.root)
    assert first["attempt_id"] == attempt == gateway.creates[0] and gateway.reads == ["op-0"]
    if not stale:
        assert text == "The recovered answer." and not later and usage["cost"] == pytest.approx(0.17)
        return
    second, = later
    assert second["attempt_id"] != attempt and text == "The answer to the new words."
    assert usage["cost"] == pytest.approx(0.22)
    sent = gateway.uploads[1][0]["messages"]
    arrived = CARRIED_WORDS if arrival == "carried-addressed" else "Please also check the second file."
    assert _place_of(sent, "The recovered answer.") < _place_of(sent, arrived)
    assert not any(row.get("role") == "tool" or row.get("tool_calls") for row in sent)


def test_keep_unseen_reply_proves_coverage_only_from_every_drain_since_the_send():
    """Each drain binds what it delivered to the pending read, at the place of the first (``note_arrivals``). A carried
    row addressed here keeps a bound reply as words with its corpus unchanged; my human's rows that entered the corpus
    keep it so even when the send bound no hash; a typed row with neither fact (a Hurry, a carried context row) does
    not, and a read with no arrival delivers as before."""
    from ouroboros import loop_output_exhaustion as recovery
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
    from ouroboros.loop_messages import owner_source_sha256

    owner = SimpleNamespace(_owner_directives=[{"source": "owner_mailbox", "content": "Start."}])
    saw = owner_source_sha256(owner)
    reply = {"role": "assistant", "content": "Old reply.", "tool_calls": [{"id": "c1"}]}
    typed = {"role": "user", "content": "Words.", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND}

    def view(*rows):
        return SimpleNamespace(tools=SimpleNamespace(_ctx=owner), messages=[{"role": "user", "content": "Start."}, *rows])

    read = {"owner_source_sha256": saw}
    recovery.note_arrivals(read, place=1, owner_input=False, carried=None)  # the first drain: nothing arrived
    recovery.note_arrivals(read, place=3, owner_input=False, carried=["chat 1 / carried-words"])  # a later one
    assert read == {"owner_source_sha256": saw, "reply_start": 1, "owner_carried_addressed": ["chat 1 / carried-words"]}
    ctx = view(typed, typed)
    assert recovery.keep_unseen_reply(ctx, read, reply, 3) is True
    assert ctx.messages[1] == {"role": "assistant", "content": "Old reply."}  # words only, at the first drain's place
    legacy = {"owner_source_sha256": None}
    recovery.note_arrivals(legacy, place=1, owner_input=True, carried=None)
    assert recovery.keep_unseen_reply(view(typed), legacy, reply, 1) is True
    for operation in ({"owner_source_sha256": saw}, {"owner_source_sha256": None}, {}, None):  # no arrival bound
        assert recovery.keep_unseen_reply(view(typed), operation, reply, 1) is False
    recovery.note_arrivals(None, place=1, owner_input=True, carried=["x"])  # no pending read: nothing to bind


@pytest.mark.parametrize("case", ["same-input", "new-owner-words", "carried-addressed", "received-failure", "unsent",
                                  "untracked"])
def test_a_withdrawn_wrap_up_ends_the_forced_rail_on_its_own_settled_request(acting, monkeypatch, case):
    """The forced rail's own request settles and meets a Wrap up there that is withdrawn before the rail drains it.
    That paid attempt is never sent again, whatever allowance is left: its receipt counts once, and its reply ends the
    rail through forced delivery, as a restored read's would (owner words, or a row carried to this work, that came
    with the withdrawal end it as a late directive). A received failure ends on the no-send terminal. A settled
    attempt the rail recorded no claim to (no ready tracker) is never read as unsent either: the unknown no-resend
    terminal. Only a request the control stopped inside its own preparation, before any create, is admitted again."""
    from ouroboros import loop_forced_finalization as forced
    from ouroboros import owner_mailbox

    ctx, root, gateway = acting.ctx, acting.root, acting.gateway
    ctx.status_drive_root = root
    ctx.messages = [{"role": "user", "content": "Summarize the verified fixture."}]
    ctx.llm_trace = {"reasoning_notes": [], "tool_calls": []}
    if case != "untracked":  # without a task id on the tool context the forced call binds no ready tracker
        ctx.tools._ctx.task_id, ctx.tools._ctx.task_attempt = ctx.task_id, 1
    quote = _carried_quote(root) if case == "carried-addressed" else None
    answer = "The fixture was verified."
    gateway.results[0] = ({**result(outcome="failed", cash=0.17, knowledge="exact", problem={
        "code": "invalid_request", "message": "Invalid argument", "context": {"httpStatus": 400}}), "message": None}
        if case == "received-failure" else
        {**result(cash=0.17, knowledge="exact"), "message": {"role": "assistant", "content": answer}})
    preparing, pressed = [], []
    if case == "unsent":
        prepare = llm_claudexor._ModelInvocation.prepare

        def observed_prepare(self, *args, **kwargs):  # the real preparation, marked while it runs
            preparing.append(True)
            try:
                return prepare(self, *args, **kwargs)
            finally:
                preparing.pop()

        monkeypatch.setattr(llm_claudexor._ModelInvocation, "prepare", observed_prepare)

    def wrap_up_then_withdrawn():  # the forced wait's own control reader
        if (preparing if case == "unsent" else gateway.acks) and not pressed:
            pressed.append(len(gateway.creates))
            if case == "new-owner-words":  # they reach the control's own drain
                assert owner_mailbox.write_owner_message(root, "Also verify the second fixture.", ctx.task_id,
                                                         msg_id="later")
            elif case == "carried-addressed":  # so does a row carried to this work, which moves no owner hash
                _carry_to_work(root, ctx.task_id, quote)
            return "finalize_requested"
        return None

    with model_wait.task_model_wait_scope(task={"id": ctx.task_id, "_attempt": 1}, drive_root=root, event_queue=None,
                                         worker_slot_held=True, owner_control=wrap_up_then_withdrawn) as owner:
        owner.tool_context = ctx.tools._ctx
        text, usage, trace = forced._forced_final_answer(ctx, prompt="Best effort only", fallback_text="No answer.",
                                                        reason_code="round_limit")

    [sent] = [row for row in ledger(root) if row["state"] != "released"]  # one request ever left the host
    assert pressed == [0 if case == "unsent" else 1] and len(gateway.creates) == len(gateway.uploads) == 1
    assert sent["state"] == "settled" and sent["cost_usd"] == 0.17 and sent["attempt_id"] == gateway.creates[0]
    if case == "untracked":  # no claim proves this receipt the rail's own: its outcome stays the unknown one, unsent again
        assert answer not in text and trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
        return
    assert usage["cost"] == pytest.approx(0.17) and usage["reason_code"] == "round_limit"  # counted once
    if case in {"same-input", "unsent"}:
        assert text == answer and trace["forced_finalization"]["source"] == "model"
    else:
        assert answer not in text
    if case != "unsent":  # the control stays recorded beside the forced reason
        assert trace["forced_finalization"]["control_reason"] == "finalize_requested"
    if case in {"new-owner-words", "carried-addressed"}:
        assert "not answered" in text and trace["forced_finalization"]["source"] == "late_owner_directive_requires_resume"
    elif case == "received-failure":
        assert "No usable answer" in text and "No new request was sent." in text


@pytest.mark.parametrize(("shape", "arrival"), [
    ("unmarked", "owner-words"), ("unmarked", "carried-addressed"), ("unmarked", "carried-context"),
    ("sidecar", "carried-addressed"),
])
def test_what_a_redrained_round_top_drained_still_keeps_the_restored_reply_as_words(setup, monkeypatch, shape, arrival):
    """The round's first drain delivers what reached this work after the restored request; the round then goes back to
    its top without a model call (a delegate hold's redrain: only that admission decision is synthetic here), and the
    next drain brings nothing. What the first drain delivered stays bound to the pending read: my human's words or a
    row carried to this work keep the reply as words at its place before them, its tools unrun, and the arrival gets a
    request of its own, even when the old send bound no owner hash. A row carried here that went to another work is
    no such arrival: the reply of a send that bound no hash finishes as before."""
    from ouroboros import owner_mailbox

    saved = _saved_native_round(setup, monkeypatch, shape)
    gateway, attempt, task_id = saved.gateway, saved.first["attempt_id"], saved.ctx.task_id
    stale = arrival != "carried-context"
    old, new = result(cash=0.17, knowledge="exact"), result(cash=0.05, knowledge="exact")
    old["message"] = {"role": "assistant", "content": "The recovered answer."}
    if stale:
        old["message"]["tool_calls"] = result()["message"]["tool_calls"]
    new["message"] = {"role": "assistant", "content": "The answer to the new words."}
    gateway.results, gateway.dispatch = [old, new], ["response_received", "response_received"]
    _bounded_creates(monkeypatch, gateway, 2 if stale else 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("a stale reply's tools never run"))
    if arrival == "owner-words":
        assert owner_mailbox.write_owner_message(saved.root, "Please also check the second file.", task_id,
                                                 msg_id="late-words")
    else:
        _carry_to_work(saved.root, task_id, _carried_quote(saved.root),
                       went_to="another-work" if arrival == "carried-context" else "")
    hold, holds = loop._delegate_hold_round_top, []

    def redrain_once(*args, **kwargs):  # the hold's admission decision alone: back to the round top once
        holds.append(True)
        return "redrain" if len(holds) == 1 else hold(*args, **kwargs)

    monkeypatch.setattr(loop, "_delegate_hold_round_top", redrain_once)
    text, usage, _ = saved.resume()

    first, *later = ledger(saved.root)
    assert len(holds) >= 2 and first["attempt_id"] == attempt == gateway.creates[0] and gateway.reads == ["op-0"]
    if not stale:
        assert text == "The recovered answer." and not later and usage["cost"] == pytest.approx(0.17)
        return
    second, = later
    assert second["attempt_id"] != attempt and text == "The answer to the new words."
    assert usage["cost"] == pytest.approx(0.22)  # the read reply once, the new request once
    sent = gateway.uploads[1][0]["messages"]
    arrived = "Please also check the second file." if arrival == "owner-words" else CARRIED_WORDS
    assert sent[_place_of(sent, "The recovered answer.")]["role"] == "assistant"
    assert _place_of(sent, "The recovered answer.") < _place_of(sent, arrived)
    assert not any(row.get("role") == "tool" or row.get("tool_calls") for row in sent)


@pytest.mark.parametrize("arrival", ["round-top", "forced-drain", "forced-drain-with-stop"])
def test_a_restored_forced_reply_never_answers_a_carried_row_its_send_never_saw(acting, monkeypatch, arrival):
    """A restored wrap-up read meets a row of my human's carried to this work after its request: shown by the round's
    drain before the read, or by the forced parser's own drain after it. The row moves no owner hash, yet the send never
    saw it: the receipt stays counted once on its one row, nothing more is sent, the forced reply is not adopted, and
    the existing late-directive terminal ends the rail. With a direct turn's Stop now in that same drain, the Stop's own
    terminal ends it, and the reply is not kept as its answer either."""
    from ouroboros import loop_forced_finalization as forced
    from ouroboros import owner_mailbox
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    ctx, root, gateway = acting.ctx, acting.root, acting.gateway
    ctx.status_drive_root = root
    ctx.messages = [{"role": "user", "content": "Summarize the verified fixture."}]
    ctx.llm_trace = {"reasoning_notes": [], "tool_calls": []}
    ctx.tools._ctx.task_id, ctx.tools._ctx.task_attempt = ctx.task_id, 1
    answer = "The fixture was verified."
    gateway.results[0] = {**result(cash=0.17, knowledge="exact"), "message": {"role": "assistant", "content": answer}}
    task, create = {"id": ctx.task_id}, gateway.create_model_operation

    def worker_lost_once_accepted(ref, *, idempotency_key, **kwargs):  # the restore below reads this very operation
        assert working_checkpoint.attach_recovery(root, task, source_task_id=ctx.task_id, from_attempt=1,
                                                  cause="worker_crash")
        return create(ref, idempotency_key=idempotency_key, **kwargs)

    monkeypatch.setattr(gateway, "create_model_operation", worker_lost_once_accepted)
    forced._forced_final_answer(ctx, prompt="Best effort only", fallback_text="No answer.", reason_code="round_limit")
    [row] = ledger(root)
    quote = _carried_quote(root)
    if arrival == "round-top":
        _carry_to_work(root, ctx.task_id, quote)
    else:
        read, arrived = loop.read_ready_operation, []

        def carried_after_the_read(limit, pending):  # the real reader; the row lands once the reply is in hand
            reply = read(limit, pending)
            if not arrived:
                arrived.append(_carry_to_work(root, ctx.task_id, quote))
                if arrival == "forced-drain-with-stop":
                    assert owner_mailbox.write_owner_message(root, REASON_OWNER_STOPPED_DIRECT_TURN, ctx.task_id,
                                                             msg_id="stop", kind=owner_mailbox.KIND_FINALIZE_NOW)
            return reply

        monkeypatch.setattr(loop, "read_ready_operation", carried_after_the_read)
    fresh = checkpoint_registry(root, monkeypatch, ctx.task_id, 2)
    fresh._ctx.task_metadata = deepcopy(ctx.tools._ctx.task_metadata)
    fresh._ctx.working_recovery = task["_working_recovery"]
    monkeypatch.setattr(ctx.llm, "chat", lambda *_a, **_k: pytest.fail("Recovery must not open a sender"))
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("A forced reply cannot run tools"))
    text, usage, trace = loop.run_llm_loop([], fresh, ctx.llm, ctx.drive_logs, lambda *_a, **_k: None,
                                         queue.Queue(), task_id=ctx.task_id, drive_root=root)

    assert gateway.creates == [row["attempt_id"]] and ledger(root) == [row]  # read, never sent again
    assert answer not in text and usage["cost"] == pytest.approx(0.17) and trace["tool_calls"] == []
    if arrival == "forced-drain-with-stop":
        assert usage["reason_code"] == "owner_requested_finalization"
        assert trace["forced_finalization"]["source"] == "owner_stopped_direct_turn"
        assert fresh._ctx._skip_post_task_synthesis is True
    else:
        assert usage["reason_code"] == "round_limit" and "not answered" in text
        assert trace["forced_finalization"]["source"] == "late_owner_directive_requires_resume"


def test_lost_create_reply_without_operation_id_cannot_authorize_new_generation(main_call, monkeypatch):
    """A generic create 4xx is not model_request_invalid: accepted custody may exist.

    Same-key rereads are allowed. After the second lost reply the scripted owner
    stops observation; neither another physical id nor a fallback send is allowed.
    An immediate honest recoverable-unknown terminal is also no-new-generation.
    """
    ctx, gateway, owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b"]}))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_args, **_kwargs: pytest.fail("metadata cannot authorize a new generation"))
    tools = _loop_tools(ctx, owner)
    tools._ctx.is_direct_chat = True
    owner.worker_slot_held = False
    create = gateway.create_model_operation
    original_key = []

    def lost_reply(ref, *, idempotency_key, **kwargs):
        if original_key and idempotency_key != original_key[0]:
            pytest.fail("lost create reply caused a NEW physical id/fallback generation")
        original_key[:] = [idempotency_key]
        create(ref, idempotency_key=idempotency_key, **kwargs)  # engine accepted; host never sees its id
        raise ClaudexorUnavailable("http_404", "create reply lost after admission", status_code=404)

    monkeypatch.setattr(gateway, "create_model_operation", lost_reply)
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    owner.owner_control = lambda: "cancelled" if len(gateway.creates) >= 2 else None
    now = [0.0]
    monkeypatch.setattr(llm_claudexor, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    with pytest.raises(ModelWaitInterrupted, match="cancelled"):
        loop.run_llm_loop(
            ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None,
            queue.Queue(), task_id="task-one", drive_root=ctx.drive_root, event_queue=events, initial_effort="high",
        )
    assert len(gateway.accepted_operations) == len(gateway.uploads) == 1
    assert set(gateway.creates) == {original_key[0]}
    assert gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "account-a"}
    rows = ledger(ctx.drive_root)
    assert [(row["state"], row["revision"]) for row in rows] == [("unresolved", 4)]
    assert rows[0]["physical_failure"]["stage"] == "raised_exception"
    assert len({row["attempt_id"] for row in rows}) == 1
    assert not gateway.acks


@pytest.mark.parametrize("ending", ["unknown", "stop", "deadline", "late-wrap-up", "released"])
def test_cold_create_without_receipt_uses_dispatch_evidence(main_call, monkeypatch, ending):
    from ouroboros import cancel_intents, owner_mailbox, usage_accounting
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, owner_stop_control_id

    ctx, gateway, owner, *_ = main_call
    root = ctx.tools._ctx.drive_root = ctx.drive_root
    task = {"id": ctx.task_id}
    checkpoint = SimpleNamespace(tools=ctx.tools, messages=ctx.messages, accumulated_usage=ctx.accumulated_usage,
        round_idx=ctx.round_idx, llm_trace={}, tool_schemas=ctx.tool_schemas, owner_msg_seen=set(), budget_tail="no_tool")
    working_checkpoint.save_ready(checkpoint)
    dispatch = usage_accounting.mark_dispatched
    create = gateway.create_model_operation

    def before_dispatch(reservation, **kwargs):
        _, state = working_checkpoint.read_working(root, ctx.task_id, 1)
        assert state["working"]["native_attempt"]["physical_attempt_id"] == reservation.attempt_id
        assert ledger(root)[0]["state"] == "reserved"
        assert loop_transport.ready_operation(state, state["usage"], root, ctx.task_id) is None
        return dispatch(reservation, **kwargs)

    def lost_create(ref, *, idempotency_key, **kwargs):
        if ending != "released":
            create(ref, idempotency_key=idempotency_key, **kwargs)  # accepted, reply never reached the host
        assert working_checkpoint.attach_recovery(root, task, source_task_id=ctx.task_id, from_attempt=1, cause="worker_crash")
        if ending == "released":
            raise ClaudexorUnavailable("model_request_invalid", "Rejected before enqueue.", status_code=400)
        raise ClaudexorUnavailable("http_404", "Create receipt unavailable.", status_code=404)

    with monkeypatch.context() as first, working_checkpoint.model_call_scope(checkpoint):
        first.setattr(usage_accounting, "mark_dispatched", before_dispatch)
        first.setattr(gateway, "create_model_operation", lost_create)
        first.setattr(llm_claudexor._ModelInvocation, "_control_outage", lambda *_: False)
        assert loop._call_round_model(ctx)[0] is None
    old, = ledger(root)
    assert old["state"] == ("released" if ending == "released" else "unresolved")
    fresh = checkpoint_registry(root, monkeypatch, ctx.task_id, 2)
    fresh._ctx.working_recovery = task["_working_recovery"]
    owner.tool_context = fresh._ctx
    monkeypatch.setenv("OUROBOROS_MODEL", MODEL)
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: pytest.fail("no operation ID to read"))
    _bounded_creates(monkeypatch, gateway, 1)
    remote = ctx.llm._chat_remote

    def restored_send(target, *args, **kwargs):
        if ending != "released" or target["provider"] != "claudexor":
            pytest.fail("restore opened a sender without proof of non-dispatch")
        return remote(target, *args, **kwargs)

    monkeypatch.setattr(ctx.llm, "_chat_remote", restored_send)
    monkeypatch.setattr(ctx.llm, "_chat_local", lambda *_a, **_kw: pytest.fail("unexpected local generation"))
    if ending == "released":
        gateway.results[0]["message"] = {"role": "assistant", "content": "Retried the unsent request."}
    if ending == "stop":
        owner_mailbox.write_owner_message(root, REASON_OWNER_STOPPED_DIRECT_TURN, ctx.task_id,
                                          msg_id="stop", kind=owner_mailbox.KIND_FINALIZE_NOW)
    elif ending == "deadline":
        fresh._ctx.task_metadata["deadline_at"] = "2000-01-01T00:00:00+00:00"
    elif ending == "late-wrap-up":
        read = loop.read_ready_operation

        def wrap_up(limit, pending):
            intent = cancel_intents.request_cancel(root, ctx.task_id, requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
            owner_mailbox.write_owner_message(root, REASON_OWNER_REQUESTED_FINALIZATION, ctx.task_id,
                                              msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
            return read(limit, pending)

        monkeypatch.setattr(loop, "read_ready_operation", wrap_up)

    text, usage, trace = loop.run_llm_loop([], fresh, ctx.llm, root / "logs", lambda *_a, **_kw: None,
                                          queue.Queue(), task_id=ctx.task_id, drive_root=root)
    assert len(gateway.accepted_operations) == len(gateway.creates) == 1 and not gateway.reads
    if ending == "released":
        first, retry = ledger(root)
        assert first["attempt_id"] == old["attempt_id"] != retry["attempt_id"]
        assert retry["attempt_id"] == gateway.creates[0] and retry["state"] == "settled"
        assert len(gateway.uploads) == 2 and text == "Retried the unsent request."
        return
    only, = ledger(root)
    assert only["state"] == "unresolved" and only.get("cost_usd") is None
    assert usage["_pending_transport_outcome"]["physical_attempt_id"] == only["attempt_id"] == gateway.creates[0]
    assert usage["_pending_transport_outcome"]["operation_id"] == "" and usage["ledger_attempt_ids"] == [only["attempt_id"]]
    assert not loop_transport.new_generation_after_unknown(fresh._ctx, usage)
    assert len(gateway.uploads) == 1 and not gateway.acks and not gateway.cancels
    if ending == "unknown":
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    elif ending == "deadline":
        assert usage["reason_code"] == "deadline_local"
    else:
        assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
        if ending == "stop":
            assert fresh._ctx._skip_post_task_synthesis
        else:
            assert trace["forced_finalization"]["control_reason"] == "finalize_requested"


def test_duplicate_model_slots_keep_their_distinct_native_accounts(main_call, monkeypatch):
    ctx, gateway, owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", f"{MODEL},{MODEL}")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b", "account-c"]}))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    # The primary's own lost answer takes one upstream observation first; a miss walks both configured slots.
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *_args: False)
    tools = _loop_tools(ctx, owner)
    gateway.results = [result(outcome="unknown", route=ROUTE), result(outcome="unknown", route=ROUTE_B),
                       _reply(ROUTE_C, {"role": "assistant", "content": "Third account answered."})]
    gateway.dispatch = ["unknown", "unknown", "response_received"]
    _bounded_creates(monkeypatch, gateway, 3)
    text, _usage, _trace = loop.run_llm_loop(
        ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None,
        queue.Queue(), task_id="task-one", drive_root=ctx.drive_root, event_queue=events)
    assert text == "Third account answered."
    assert [payload["account"] for payload, _ in gateway.uploads] == [
        {"mode": "pin", "profileId": pin} for pin in ("account-a", "account-b", "account-c")]
    assert tools._ctx.context_fit_plan.model_role == "fallback:1"
    assert [row["state"] for row in ledger(ctx.drive_root)].count("unresolved") == 2


@pytest.mark.parametrize(("wire", "status", "reason", "access_reason"), [
    ("http", 401, "auth", "api_key"),
    ("http", 402, "quota", "api_billing"),
    ("http", 403, "auth", ""),
    ("sse", 402, "quota", ""),
    ("sse", 400, "", ""),
])
def test_primary_wait_honors_a_direct_api_access_refusal(main_call, monkeypatch, status, reason, access_reason, wire):
    """A declared wait must not be undone by a paid fallback merely because access uses an API key."""
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch

    ctx, gateway, owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    ctx.active_model = "openai::primary-test"
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model=ctx.active_model, provider="openai", model_route={})
    ctx.tools._ctx.context_fit_plan = ctx.context_fit_plan
    ctx.tools._ctx.primary_route = {"model": ctx.active_model, "use_local": False, "role": "main"}
    ctx.tools._ctx.model_wait_context = owner
    _switch_model(ctx.tools._ctx, primary="wait")
    sent = []
    original_send = ctx.llm._chat_remote

    def send(target, messages, schemas, *args, **kwargs):
        if target["provider"] != "openai":
            return original_send(target, messages, schemas, *args, **kwargs)
        assert target["provider"] == "openai"
        sent.append(target["provider"])
        body = {"messages": deepcopy(messages), "tools": schemas or [], "model": "primary-test"}
        request = _attempt_request(target, body)

        def refused():
            if wire == "sse":
                from ouroboros.llm_stream import ChatAccumulator
                ChatAccumulator().accept("message", json.dumps({"error": {"code": status, "message": "Access refused"}}))
            raise httpx.HTTPStatusError("Access refused", request=httpx.Request("POST", "https://fixture.invalid"),
                                       response=httpx.Response(status))

        return ua.execute_physical_attempt(request, refused, before_dispatch=_candidate_before_dispatch(body, request))

    monkeypatch.setattr(ctx.llm, "_chat_remote", send)
    monkeypatch.setattr(ctx.llm, "claudexor_model_sources", lambda: pytest.fail("API wait has no subscription observer"))
    owner.owner_control = lambda: "cancelled" if any(row.get("state") == "waiting" for row in owner.waits.values()) else None
    if not reason:  # Invalid request is not unavailable access: ordinary fallback still works.
        from ouroboros.loop_model_call import _recover_failed_round
        assert loop._call_round_model(ctx)[0] is None
        gateway.results = [_reply(ROUTE_B, {"role": "assistant", "content": "Alternative accepted the request."})]
        gateway.dispatch = ["response_received"]
        fallback_cooldown.reset_for_tests()
        reply, *_ = _recover_failed_round(ctx, ctx.tools, None, None, context_fit_plan=ctx.context_fit_plan,
            active_context_mode="max", emit_progress=lambda *_a, **_k: None)
        assert reply["content"] == "Alternative accepted the request."
        assert not owner.waits and sent == ["openai"] and len(gateway.creates) == 1
        return
    with pytest.raises(ModelWaitInterrupted, match="cancelled"):
        loop._call_round_model(ctx)
    waits = [row for row in list(events.queue) if row.get("type") == "task_model_wait"]
    assert waits[0]["state"] == "waiting" and waits[0]["reason"] == reason
    assert {wait["access_reason"] for wait in waits} == {access_reason}
    assert waits[0]["availability_observation"] == "unavailable" and waits[0]["auto_continue"] is False
    assert waits[0]["credential_harness"] == "" and waits[-1]["resolution"] == "cancelled"
    assert sent == ["openai"] and gateway.creates == []


@pytest.mark.parametrize("primary_account,fallback_account", [("account-a", "account-b"), ("", "account-b"), ("account-a", "")])
def test_transient_cooldown_keeps_other_native_account_available(main_call, monkeypatch, primary_account, fallback_account):
    from ouroboros import loop_model_call

    ctx, gateway, owner, _events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL)
    monkeypatch.setenv("OUROBOROS_FALLBACK_COOLDOWN_ENABLED", "true")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": primary_account, "fallback": ["decoy"]}))
    owner.overrides["fallback:0"] = {"model_account_override": fallback_account}
    inner = ctx.tools._ctx
    inner.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    ctx.accumulated_usage["_last_llm_error_kind"] = "provider_transient"
    gateway.results = [_reply(ROUTE_B, {"role": "assistant", "content": "Other account answered."})]
    gateway.dispatch = ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    fallback_cooldown.reset_for_tests()
    try:
        reply, *_ = loop_model_call._run_cross_model_fallback_chain(
            llm=ctx.llm, ctx=inner, tools=ctx.tools, messages=ctx.messages, active_model=MODEL,
            active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1,
            drive_logs=ctx.drive_logs, task_id=inner.task_id, round_idx=1, event_queue=ctx.event_queue,
            accumulated_usage=ctx.accumulated_usage, task_type="task", emit_progress=lambda *_a, **_k: None,
            context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
        assert reply["content"] == "Other account answered."
        sent = gateway.uploads[0][0]["account"]
        if fallback_account:
            assert sent == {"mode": "pin", "profileId": fallback_account}
        else:
            assert sent["mode"] == "auto" and "profileId" not in sent
        assert fallback_cooldown.is_cooling_down(MODEL, False, primary_account)
        assert not fallback_cooldown.is_cooling_down(MODEL, False, fallback_account)
        # Resource-refusal deferral must use the same account key as the actual walk.
        assert loop_model_call._route_follows([(MODEL, "fallback:0", False, False)])
        assert not loop_model_call._route_follows([(MODEL, "main", False, True)])
    finally:
        fallback_cooldown.reset_for_tests()


@pytest.mark.parametrize(("primary_account", "choice"), [("account-a", "return"), ("", "wait")])
def test_primary_choice_survives_shared_cold_continuation(main_call, monkeypatch, choice, primary_account):
    from ouroboros import budget_pause, loop_model_call, owner_wait
    from ouroboros.model_slots import route_binding

    ctx, _gateway, owner, _events, _decide, _observations = main_call
    # This route-only fixture has no resident or current tool catalog.
    monkeypatch.setattr(ctx.tools, "schemas", lambda: [], raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", MODEL)  # the primary follows Main: its current slot is this model
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "decoy", "fallback": ["account-b"]}))
    owner.overrides["main"] = {"model_account_override": primary_account}
    inner = ctx.tools._ctx
    inner.model_wait_context = owner
    inner.active_effort, inner.active_use_local = "high", False
    inner.context_fit_plan = replace(ctx.context_fit_plan, model_role="fallback:0")
    inner.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    loop._reset_turn_state(inner)
    assert "OK:" in _switch_model(inner, primary=choice)
    inner._route_facts_pending = "dated refusal fact"
    state = owner_wait.continuation_state(inner, ctx.messages, {}, {}, 1, [], set())
    # JSON storage and a fresh turn can replace the initial role before restore.
    state = json.loads(json.dumps(state))
    loop._reset_turn_state(inner)
    inner.primary_route = {"model": "decoy", "use_local": False, "role": "fallback:0"}
    inner.budget_drive_root = inner.drive_root = ctx.drive_root
    state["_pause_row"] = {"pause_id": "primary-pause", "state": budget_pause.STATE_RESUME_GRANTED,
                           "rail": budget_pause.RAIL_GLOBAL_EXHAUSTED, "grant": {"grant_id": "owner-resume"}}
    budget_pause.set_budget_pause(ctx.drive_root, inner.task_id, state["_pause_row"])
    model, effort, local, mode, _round, restored_plan = budget_pause.resume_paused_loop(
        ctx.tools, state, ctx.messages, {}, {}, set(), budget_remaining_usd=5.0)
    assert budget_pause.budget_pause_row(ctx.drive_root, inner.task_id)["state"] == budget_pause.STATE_RESUMED
    assert any("after an explicit Resume (" in str(row.get("content")) for row in ctx.messages)  # a grant naming nobody
    model, local, effort, plan, _mode = loop_model_call._apply_round_route_overrides(
        inner, ctx.tools, ctx.messages, (model, local, effort), restored_plan, mode, "max", [])
    assert plan.model_role == "main" and inner.route_wait_on_primary is (choice == "wait")
    assert route_binding(model, local, plan.model_role, overrides=owner.overrides) == (MODEL, False, primary_account)
    assert inner.primary_route == {"model": MODEL, "use_local": False, "role": "main"}
    assert inner._route_facts_pending == "dated refusal fact" and effort == "high"


_REFUSAL_CASES = [
    (wire, status, ending) for wire in ("managed", "http", "sse")
    for status in (429, 503, 529) for ending in ("answer", "stop", "deadline")
] + [("managed", 503, "wrap")]

# All wire/status, wire/ending and status/ending pairs. The complete table below
# proves that each wire/status becomes the same refusal policy before an ending
# is applied; these cases additionally compose it with the loop and real ledger.
_REFUSAL_LOOP_CASES = [
    (wire, status, ("answer", "stop", "deadline")[(wire_index + status_index) % 3])
    for wire_index, wire in enumerate(("managed", "http", "sse"))
    for status_index, status in enumerate((429, 503, 529))
] + [("managed", 503, "wrap")]


def _wire_refusal(wire, status):
    if wire == "managed":
        return llm_claudexor.ClaudexorModelError(json.loads(json.dumps({
            "code": "provider_failed", "message": "Upstream refused", "retryable": True,
            "context": {"httpStatus": status}})), model_role="main", route=ROUTE)
    if wire == "sse":
        from ouroboros.llm_stream import ChatAccumulator, ProviderStreamError

        with pytest.raises(ProviderStreamError) as raised:
            ChatAccumulator().accept("message", json.dumps({
                "error": {"code": status, "message": "Temporary refusal"}}))
        return raised.value
    return httpx.HTTPStatusError("Temporary refusal",
        request=httpx.Request("POST", "https://fixture.invalid"), response=httpx.Response(status))


def test_primary_refusal_policy_preserves_every_wire_status_and_ending(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for wire, status, ending in _REFUSAL_CASES:
        tmp_path = next(case_paths)
        with subtests.test(wire=wire, status=status, ending=ending), monkeypatch.context() as monkeypatch:
            from datetime import datetime, timezone
            from ouroboros.loop_llm_call import classify_llm_exception
            from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
            from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

            refusal = classify_llm_exception(_wire_refusal(wire, status))
            assert (refusal.kind, refusal.retry_same_request, refusal.status_code) == ("provider_transient", True, status)
            if wire == "managed":
                payload = llm_claudexor._request({"source": ROUTE["source"], "resolved_model": ROUTE["model"]},
                    [{"role": "user", "content": "Keep waiting on the primary."}], None,
                    {"model_role": "main", "model_account_override": "account-a", "prospective": True})
                assert json.loads(json.dumps(payload))["account"] == {"mode": "pin", "profileId": "account-a"}
            ctx = SimpleNamespace(route_wait_on_primary=True, is_direct_chat=ending != "wrap", task_metadata={})
            assert not loop_transport.fallback_chain_allowed(ctx, refusal.kind, None)
            ctx.route_wait_on_primary = False
            assert loop_transport.fallback_chain_allowed(ctx, refusal.kind, None)
            ctx.route_wait_on_primary = True
            now, waits, notes = [1000.0], [], []
            monkeypatch.setattr(loop_transport, "time", SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0]))

            def sleep(seconds, _wake):
                waits.append(seconds)
                now[0] += seconds
                return False

            monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", sleep)
            common = {"drive_logs": tmp_path, "task_id": "task-one", "model": MODEL,
                      "emit_progress": lambda text, **_kw: notes.append(text)}
            episode = loop_transport.reconcile_transport_wait(None, ctx, msg_present=False, error_kind=refusal.kind, **common)
            assert episode.wait_cause == refusal.kind and episode.interactive is (ending != "wrap")
            step = {**common, "tools": SimpleNamespace(_ctx=ctx), "drive_root": None,
                    "incoming_messages": None, "owner_msg_seen": set()}
            assert loop_transport.transport_wait_step(episode, error_kind=refusal.kind, **step)
            assert waits == [4.0]
            if ending == "answer":
                # Another real refusal preserves this episode: it cannot reset its
                # pacing or claim that an available catalog was a recovered generation.
                assert loop_transport.reconcile_transport_wait(episode, ctx, msg_present=False,
                                                               error_kind=refusal.kind, **common) is episode
                assert loop_transport.transport_wait_step(episode, error_kind=refusal.kind, **step)
                assert waits == [4.0, 8.0] and episode.redials == 2
                assert loop_transport.reconcile_transport_wait(episode, ctx, msg_present=True,
                                                               error_kind="", **common) is None
            else:
                control = {"stop": REASON_OWNER_STOPPED_DIRECT_TURN, "wrap": REASON_OWNER_REQUESTED_FINALIZATION}.get(ending, "")

                def terminal(**facts):
                    assert facts["error_kind"] == facts["wait_cause"] == refusal.kind
                    assert facts["control_reason"] == control
                    return loop_transport.provider_terminal_fallback_text(
                        {"_last_llm_error_kind": refusal.kind, "_last_llm_status_code": status},
                        is_context_overflow=False, is_transport_wait=True, is_deadline_exhausted=ending == "deadline",
                        waited_sec=facts["waited_sec"], interactive=facts["interactive"],
                        control_reason=facts["control_reason"], wait_cause=facts["wait_cause"])

                if ending == "deadline":
                    ctx.task_metadata["deadline_at"] = datetime.fromtimestamp(now[0] - 1, timezone.utc).isoformat()
                    assert not loop_transport.transport_wait_step(episode, error_kind="deadline_exhausted", **step)
                    notice = terminal(error_kind=refusal.kind, wait_cause=episode.wait_cause,
                                      waited_sec=episode.waited_sec, interactive=episode.interactive, control_reason="")
                else:
                    notice = loop_transport.finalize_now_transport_terminal(episode, drive_logs=tmp_path,
                        task_id="task-one", model=MODEL, handle_provider_unavailable=terminal, control_reason=control)
                assert "No new summary request was sent" in notice
                assert "provider connection was unavailable" not in notice and "$0" not in notice
                assert "refusing requests" in notice
                assert waits == [4.0] and episode.redials == 1
            assert any("primary provider temporarily refused" in note for note in notes)
            assert not any("$0" in note or "connection restored" in note.lower() for note in notes)
            if ending == "wrap":
                assert any(note.endswith("Stop cancels.") for note in notes)
            events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            assert events[0]["detail"] == refusal.kind
            assert events[-1]["phase"] == ("recovered" if ending == "answer" else "ended")


@pytest.mark.parametrize("wire,status,ending", chosen_rows(_REFUSAL_LOOP_CASES, 0, 4, 9))
def test_primary_refusal_wait_paces_real_requests_without_catalog_recovery(
        main_call, monkeypatch, wire, status, ending):
    from datetime import datetime, timedelta, timezone
    from ouroboros import loop_llm_call, owner_mailbox, usage_accounting as ua
    from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    ctx, gateway, owner, events, _decide, _observations = main_call
    primary = MODEL if wire == "managed" else "openai::primary-test"
    for key, value in {"OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_TASK_REVIEW_MODE": "off",
                       "OUROBOROS_SAFETY_MODE": "off", "MCP_ENABLED": "false",
                       "OUROBOROS_TRANSIENT_RETRY_MAX": "6", "OUROBOROS_MODEL_FALLBACKS": MODEL,
                       "OPENAI_API_KEY": "fixture-key"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b"]}))
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model=primary,
        provider="claudexor" if wire == "managed" else "openai", model_route=ROUTE if wire == "managed" else {})
    tools = _loop_tools(ctx, owner)
    tools._ctx.task_model_override, tools._ctx.is_direct_chat = primary, ending != "wrap"
    tools._ctx.model_wait_context = owner
    owner.worker_slot_held = ending == "wrap"
    wait_call = {"id": "wait-primary", "type": "function", "function": {
        "name": "switch_model", "arguments": json.dumps({"primary": "wait"})}}
    first = {"role": "assistant", "content": "", "tool_calls": [wait_call]}
    final = {"role": "assistant", "content": "The primary answered."}
    failures = 12
    gateway.results = [_reply(ROUTE, first)] + [result(outcome="failed", problem={
        "code": "provider_failed", "message": "Upstream refused", "retryable": True,
        "context": {"httpStatus": status}}) for _ in range(failures)] + [_reply(ROUTE, final)]
    gateway.dispatch = ["response_received"] * (failures + 2)
    _bounded_creates(monkeypatch, gateway, failures + 2)
    sent = []
    if wire != "managed":
        def send(target, messages, schemas, *args, **kwargs):
            assert target["provider"] == "openai", "a chosen primary wait must not dispatch a fallback"
            sent.append(target["provider"])
            body = {"messages": deepcopy(messages), "tools": schemas or [], "model": "primary-test"}
            request = _attempt_request(target, body)
            def physical():
                if len(sent) in (1, failures + 2):
                    return (first if len(sent) == 1 else final), {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.01, "cost_final": True}
                if wire == "sse":
                    from ouroboros.llm_stream import ChatAccumulator
                    ChatAccumulator().accept("message", json.dumps({"error": {"code": status, "message": "Temporary refusal"}}))
                raise httpx.HTTPStatusError("Temporary refusal", request=httpx.Request("POST", "https://fixture.invalid"),
                                           response=httpx.Response(status))
            return ua.execute_physical_attempt(request, physical,
                extractor=lambda value: (value[1], 0.01, True), before_dispatch=_candidate_before_dispatch(body, request))
        monkeypatch.setattr(ctx.llm, "_chat_remote", send)
    def no_probe(*_a, **_kw):
        pytest.fail("an available catalog does not prove generation recovery")
    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", no_probe)
    monkeypatch.setattr(ctx.llm, "claudexor_model_sources", no_probe)
    bursts, waits, notes = [], [], []
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda seconds, *_a, **_k: bursts.append(seconds) or True)
    def pause(seconds, _wake):
        waits.append(seconds)
        if ending in {"stop", "wrap"}:
            from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
            reason = REASON_OWNER_STOPPED_DIRECT_TURN if ending == "stop" else REASON_OWNER_REQUESTED_FINALIZATION
            msg_id = "stop-primary-wait"
            if ending == "wrap":
                from ouroboros import cancel_intents
                from supervisor.owner_stop import owner_stop_control_id
                intent = cancel_intents.request_cancel(ctx.drive_root, "task-one",
                    requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
                msg_id = owner_stop_control_id(intent)
            owner_mailbox.write_owner_message(ctx.drive_root, reason, "task-one",
                msg_id=msg_id, kind=owner_mailbox.KIND_FINALIZE_NOW)
        elif ending == "deadline":
            tools._ctx.task_metadata["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        return ending != "answer"
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", pause)
    text, usage, _trace = loop.run_llm_loop(
        ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda text, **_kw: notes.append(text), queue.Queue(),
        task_id="task-one", drive_root=ctx.drive_root, event_queue=events)
    attempts = len(gateway.creates) if wire == "managed" else len(sent)
    assert attempts == (failures + 2 if ending == "answer" else 7)
    assert waits == ([4.0, 8.0] if ending == "answer" else [4.0])
    assert bursts == [4.0, 8.0, 16.0, 32.0, 60.0] * (2 if ending == "answer" else 1)
    assert not owner.waits  # no immediate resource_available row/new paid call loop
    if wire == "managed":
        assert all(payload["account"] == {"mode": "pin", "profileId": "account-a"} for payload, _ in gateway.uploads)
    rows = ledger(ctx.drive_root)  # one current row per attempt: every sent attempt
    assert sum(row["state"] in {"dispatched", "settled", "unresolved"} for row in rows) == attempts
    assert not any(row["state"] == "released" for row in rows)
    assert any("primary provider temporarily refused" in note for note in notes)
    assert not any("$0" in note or "connection restored" in note.lower() for note in notes)
    if ending == "answer":
        assert text == final["content"]
    else:
        notice = usage.get("terminal_provider_notice", text)
        if ending == "stop":
            assert "owner stopped this chat turn" in notice
        else:
            assert "No new summary request was sent" in notice
        if ending == "wrap":
            assert "owner requested Wrap up while the primary provider was refusing requests" in notice
            assert any(note.endswith("Stop cancels.") for note in notes)
        assert "provider connection was unavailable" not in notice and "$0" not in notice


@pytest.mark.parametrize("arrival", ["none", "owner-words", "stop-now"])
def test_mail_landing_after_a_restored_read_binds_before_its_reply_or_tools(setup, monkeypatch, arrival):
    """What lands once a restored read's polling ended, before its reply is admitted, is drained and bound to that read
    (``admit_read_reply``): new owner words keep the reply as words, its tools never run, and they get a round of their
    own; a direct turn's Stop now that drain consumed takes its own rail at once, zero further calls, the read's receipt
    counted once. With nothing new the reply's own tools run, as before (paired control)."""
    from ouroboros import loop_output_exhaustion as recovery
    from ouroboros import owner_mailbox
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, attempt, task_id = saved.gateway, saved.first["attempt_id"], saved.ctx.task_id
    old, new = result(cash=0.17, knowledge="exact"), result(cash=0.05, knowledge="exact")
    old["message"] = {"role": "assistant", "content": "The recovered answer.", "tool_calls": result()["message"]["tool_calls"]}
    new["message"] = {"role": "assistant", "content": "The answer to the new words."}
    gateway.results, gateway.dispatch = [old, new], ["response_received"] * 2
    _bounded_creates(monkeypatch, gateway, 2 if arrival == "owner-words" else 1)

    class ToolsRan(Exception):
        pass

    def tools_ran(*_a, **_k):
        raise ToolsRan()

    monkeypatch.setattr(loop, "handle_tool_calls", tools_ran)
    reconcile = recovery.reconcile_read

    def after_the_read(ctx, operation, msg):  # the read's polling is over; nothing of its reply is admitted yet
        if arrival == "owner-words":
            owner_mailbox.write_owner_message(saved.root, "Please also check the second file.", task_id, msg_id="late-words")
        elif arrival == "stop-now":
            owner_mailbox.write_owner_message(saved.root, REASON_OWNER_STOPPED_DIRECT_TURN, task_id,
                                              msg_id="stop", kind=owner_mailbox.KIND_FINALIZE_NOW)
        return reconcile(ctx, operation, msg)

    monkeypatch.setattr(recovery, "reconcile_read", after_the_read)
    if arrival == "none":
        with pytest.raises(ToolsRan):
            saved.resume()
        assert gateway.creates == [attempt]
        return
    text, usage, trace = saved.resume()
    first, *later = ledger(saved.root)
    assert first["attempt_id"] == attempt and first["state"] == "settled" and first["cost_usd"] == 0.17
    if arrival == "stop-now":
        assert not later and gateway.creates == [attempt], "Stop now: zero further calls"
        assert usage["reason_code"] == "owner_requested_finalization" and saved.ctx._skip_post_task_synthesis is True
        assert trace["forced_finalization"]["source"] == "owner_stopped_direct_turn"
        assert usage["cost"] == pytest.approx(0.17), "the read's receipt, counted once"
        return
    second, = later
    assert gateway.creates == [attempt, second["attempt_id"]] and text == "The answer to the new words."
    assert usage["cost"] == pytest.approx(0.22)
    sent = gateway.uploads[1][0]["messages"]
    assert any("Please also check the second file." in json.dumps(row.get("content")) for row in sent)
    assert not any(row.get("role") == "tool" or row.get("tool_calls") for row in sent), "words only: no stale tool effect"


def test_a_pause_while_a_restored_operation_still_runs_parks_its_read_and_cancels_nothing(setup, monkeypatch):
    """The owner's Pause met while a restored read polls an operation that still runs: the poll observes the fence
    (``poll_read_control``), the operation is neither cancelled nor sent again, the read stays owed with its identity
    (``paused_read``), and Resume reads that same operation once it answers."""
    from ouroboros import budget_pause, owner_pause
    from ouroboros.artifacts import read_actor_source_bytes

    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    root, gateway, task_id, attempt = saved.root, saved.gateway, saved.ctx.task_id, saved.first["attempt_id"]
    answer = result(cash=0.17, knowledge="exact")
    answer["message"] = {"role": "assistant", "content": "The recovered answer."}
    gateway.results, gateway.dispatch, gateway.pending = [answer], ["response_received"], True
    _bounded_creates(monkeypatch, gateway, 1)
    read = gateway.get_model_operation

    def running_then_paused(operation_id, **kwargs):
        detail = read(operation_id, **kwargs)
        if len(gateway.reads) == 1:  # it still runs; the owner pauses the tree meanwhile
            owner_pause.install_fence(root, task_id, request_id="pause-while-running")
        return detail

    monkeypatch.setattr(gateway, "get_model_operation", running_then_paused)
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested) as held:
            saved.resume()
        pause = held.value.pause
        assert pause["rail"] == budget_pause.RAIL_OWNER_PAUSE
        assert not gateway.cancels and gateway.creates == [attempt] and len(gateway.uploads) == 1
        assert ledger(root)[0]["state"] == "unresolved"
        state = json.loads(read_actor_source_bytes(root, task_id, pause["source_ref"]))
        kept = state["usage"]["_pending_transport_outcome"]
        assert kept["paused_receipt_read"] is True and (kept["physical_attempt_id"], kept["operation_id"]) == (attempt, "op-0")
        owed = loop_transport.ready_operation(state, deepcopy(state["usage"]), root, task_id)
        assert (owed["physical_attempt_id"], owed["operation_id"]) == (attempt, "op-0"), "Resume owes the same read"

        gateway.pending = False
        grant = {"grant_id": "resume", "generation": 1, "owner_pause_fence_id": pause["owner_fence_id"]}
        budget_pause.set_budget_pause(root, task_id, {**pause, "state": budget_pause.STATE_RESUME_GRANTED, "grant": grant})
        saved.ctx.budget_pause_resume = {"pause_id": pause["pause_id"], "pause_generation": pause["pause_generation"],
                                        "grant_id": grant["grant_id"], "grant_generation": grant["generation"]}
        text, usage, _ = saved.resume()
    finally:
        budget_pause.end_dispatch_fence(task_id)
    assert text == "The recovered answer." and gateway.creates == [attempt] and not gateway.cancels
    assert ledger(root)[0]["state"] == "settled" and usage["cost"] == pytest.approx(0.17)


@pytest.mark.parametrize(("fence", "expected"), [
    ({}, None),
    ({"state": "paused", "fence_id": "fixture"}, "owner_pause"),
    ({"state": "unknown", "reason": "owner_pause_authority_unreadable"}, "owner_pause_authority_unreadable"),
    ("busy", None),
], ids=["open", "installed", "unreadable", "busy-store"])
def test_a_restored_reads_poll_types_the_owner_fence_as_the_owner_seam_reads_it(monkeypatch, fence, expected):
    """The poll's fence read is typed as ``_handle_model_wait_control`` routes it: installed parks, unreadable takes the
    technical rail; a busy store proves neither and the poll goes on. Neither reason cancels the operation
    (``llm_claudexor`` ``check_control``)."""
    from ouroboros import loop_output_exhaustion as recovery
    from ouroboros import loop_round_limits, owner_pause
    from ouroboros.store import StoreBusy

    def read(_owner):
        if fence == "busy":
            raise StoreBusy("fixture contention", reason="contention")
        return fence

    monkeypatch.setattr(owner_pause, "member_fence", read)
    monkeypatch.setattr(loop_transport, "transport_repeat_stop_requested", lambda *_a, **_k: False)
    monkeypatch.setattr(loop_round_limits, "local_deadline_remaining", lambda *_a, **_k: None)
    assert recovery.poll_read_control(SimpleNamespace(deadline_ts=None), object(), object()) == expected
    assert expected is None or expected in recovery.OWNER_PAUSE_CONTROLS


@pytest.mark.parametrize("receipt", ["rejected", "ordinary-bad-request"])
def test_a_live_rejected_receipt_ends_known_before_any_route_wrap_up_or_grace(setup, monkeypatch, receipt):
    """A live accepted operation whose terminal response is judged unusable (``response_rejected``) is recorded as its own
    known end (``restored_view`` → ``rejected_receipt``): the configured routes and the no-call rail, a deadline's grace
    included, read that row and send nothing over it; its settlement stays the provider's. An ordinary settled failure
    keeps the chain policy unchanged (paired control)."""
    from ouroboros.loop_llm_call import provider_no_call_source
    from ouroboros.loop_transport import fallback_chain_allowed

    root, gateway, client = setup
    task_id = "task-one"
    write_task_result(root, task_id, "running")
    if receipt == "rejected":
        reply = result(cash=0.17, knowledge="exact", problem={"code": "response_rejected", "message": "Synthetic unusable response."})
        reply["message"] = None
    else:
        reply = result(outcome="failed", cash=0.17, knowledge="exact", problem={
            "code": "invalid_request", "message": "Invalid argument", "context": {"httpStatus": 400}})
    gateway.results, gateway.dispatch = [reply], ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    usage = {"execution_id": "live"}
    msg, _cost = call_llm_with_retry(client, [{"role": "user", "content": "Sample note."}], MODEL, [], "low", 1,
                                     root / "logs", task_id, 1, None, usage)
    failure = usage["_last_llm_failure"]
    assert msg is None and len(gateway.creates) == 1 and failure["operation_id"] == "op-0"
    assert ledger(root)[0]["state"] == "settled", "the provider's settlement stands"
    ctx = SimpleNamespace(task_id=task_id, exact_model_route=False, route_wait_on_primary=False)
    if receipt == "rejected":
        assert failure["rejected_receipt"] is True
        assert fallback_chain_allowed(ctx, "provider_error", None, usage, primary_acting=True) is False
        for deadline in (False, True):  # a deadline's grace buys no wrap-up over it either
            assert provider_no_call_source(usage, deadline) == ("recovered_operation_no_resend", False)
    else:
        assert "rejected_receipt" not in failure
        assert fallback_chain_allowed(ctx, "provider_error", None, usage, primary_acting=True) is True
        assert provider_no_call_source(usage, True)[0] != "recovered_operation_no_resend"


def test_a_restored_rejected_receipt_ends_the_round_before_any_new_admission(setup, monkeypatch):
    """A restored read whose receipt is judged unusable is terminal in the read itself (``ordinary_read``): no round's own
    call, configured route (``openai::unused`` stays unasked) or wrap-up is admitted over it, and its receipt counts once."""
    saved = _saved_native_round(setup, monkeypatch, "sidecar")
    gateway, attempt = saved.gateway, saved.first["attempt_id"]
    empty = result(cash=0.17, knowledge="exact")
    empty["message"] = {"role": "assistant", "content": " \n"}
    gateway.results, gateway.dispatch = [empty], ["response_received"]
    _bounded_creates(monkeypatch, gateway, 1)
    monkeypatch.setattr(loop, "handle_tool_calls", lambda *_a, **_k: pytest.fail("no reply was admitted"))
    _text, usage, _trace = saved.resume()
    row, = ledger(saved.root)
    assert gateway.creates == [attempt] and len(gateway.uploads) == 1 and row["state"] == "settled"
    failure = usage["_last_llm_failure"]
    assert failure["rejected_receipt"] is True and failure["recovered_operation_id"] == "op-0"
    assert usage["cost"] == pytest.approx(0.17) and attempt in usage["_counted_error_attempts"]


def _forced_rail(acting):
    ctx = acting.ctx
    ctx.messages = [{"role": "user", "content": "Summarize the verified fixture."}]
    ctx.llm_trace = {"reasoning_notes": [], "tool_calls": []}
    ctx.tools._ctx.task_id, ctx.tools._ctx.task_attempt = ctx.task_id, 1
    return ctx


@pytest.mark.parametrize("stop", [False, True], ids=["no-stop", "stop-now"])
@pytest.mark.parametrize("exit_", ["window-elapsed", "authority", "refusal"])
def test_a_stop_now_any_forced_drain_reads_outranks_every_exit_of_the_wrap_up(acting, monkeypatch, exit_, stop):
    """A direct turn's Stop now read by the wrap-up's own drains (its prompt's, or an exceptional exit's) ends the rail on
    the existing hard-stop terminal before an early exit, an authority fallback or a received refusal's raise; nothing is
    sent. Without it each exit keeps its own terminal (paired control)."""
    from ouroboros import loop_forced_finalization as forced
    from ouroboros import owner_mailbox
    from ouroboros.usage_accounting import BillingAuthorityUnavailable
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    ctx, root, gateway = _forced_rail(acting), acting.root, acting.gateway

    def stop_arrives():
        if stop:
            assert owner_mailbox.write_owner_message(root, REASON_OWNER_STOPPED_DIRECT_TURN, ctx.task_id,
                                                     msg_id="stop", kind=owner_mailbox.KIND_FINALIZE_NOW)

    reason = "finalization_grace" if exit_ == "authority" else "round_limit"
    if exit_ == "window-elapsed":
        ctx.deadline_ts = 1.0
        stop_arrives()  # read by the prompt's own drain, before every early exit
    else:
        def refused_send(_ctx, **_kwargs):
            stop_arrives()  # read only by the exceptional exit's drain
            if exit_ == "authority":
                raise BillingAuthorityUnavailable("Synthetic unreadable authority.", cause="fixture")
            raise llm_claudexor.ClaudexorRefusalError(llm_claudexor.ClaudexorRefusalError.terminal_problem())

        monkeypatch.setattr(loop, "_call_forced_model_once", refused_send)
    if exit_ == "refusal" and not stop:
        with pytest.raises(llm_claudexor.ClaudexorRefusalError):  # the refusal's own terminal belongs to the loop
            forced._forced_final_answer(ctx, prompt="Best effort only", fallback_text="No answer.", reason_code=reason)
        assert not gateway.creates
        return
    _text, usage, trace = forced._forced_final_answer(ctx, prompt="Best effort only", fallback_text="No answer.",
                                                      reason_code=reason)
    assert not gateway.creates, "no wrap-up request was sent"
    source = trace["forced_finalization"]["source"]
    if stop:
        assert source == "owner_stopped_direct_turn" and usage["reason_code"] == "owner_requested_finalization"
        assert ctx.tools._ctx._skip_post_task_synthesis is True
    else:
        assert source == {"window-elapsed": "round_limit_window_elapsed", "authority": "forced_authority_unavailable"}[exit_]


@pytest.mark.parametrize("first", ["rejected", "ordinary-failure"])
def test_a_wrap_ups_own_rejected_receipt_is_never_followed_by_a_second_forced_send(acting, monkeypatch, first):
    """New owner words arriving while the wrap-up's own request is read: after its own accepted receipt was judged
    unusable (``rejected_receipt``) the rail ends on the late-directive terminal, one request in all; an ordinary settled
    failure keeps the existing refresh and its second forced request (paired control)."""
    from ouroboros import loop_forced_finalization as forced
    from ouroboros import owner_mailbox

    ctx, root, gateway = _forced_rail(acting), acting.root, acting.gateway
    if first == "rejected":
        gateway.results[0] = result(cash=0.17, knowledge="exact", problem={
            "code": "response_rejected", "message": "Synthetic unusable response."})
        gateway.results[0]["message"] = None
    else:
        gateway.results[0] = result(outcome="failed", cash=0.17, knowledge="exact", problem={
            "code": "invalid_request", "message": "Invalid argument", "context": {"httpStatus": 400}})
    gateway.results[1] = {**result(cash=0.05, knowledge="exact"),
                          "message": {"role": "assistant", "content": "The fixture was verified."}}
    get_result, arrived = gateway.get_model_result, []

    def words_land_during_the_first_read(operation_id, **kwargs):
        body = get_result(operation_id, **kwargs)
        if not arrived:
            arrived.append(owner_mailbox.write_owner_message(root, "Please also check the second file.", ctx.task_id,
                                                             msg_id="late-words"))
        return body

    monkeypatch.setattr(gateway, "get_model_result", words_land_during_the_first_read)
    text, usage, trace = forced._forced_final_answer(ctx, prompt="Best effort only", fallback_text="No answer.",
                                                     reason_code="round_limit")
    assert arrived
    if first == "rejected":
        assert len(gateway.creates) == 1 and usage["_last_llm_failure"]["rejected_receipt"] is True
        assert trace["forced_finalization"]["source"] == "late_owner_directive_requires_resume"
    else:
        assert len(gateway.creates) == 2 and text == "The fixture was verified."
