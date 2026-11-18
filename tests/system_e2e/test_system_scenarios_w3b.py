"""S11-S13 — Ф4 wave 3b of the deep-integration suite (v7next plan §8).

Two surfaces on the wave-1/2 skeleton, keyless throughout, every assertion a
DURABLE artifact or a recorded WIRE fact (never an HTTP 200 alone):

* S11 — DELEGATED TRANSPORT: a scripted top-level nanny drives ``delegate_start``
  → ``delegate_wait`` against a ``FakeClaudexorDaemon`` serving the exact client
  contract (protocol-3 handshake, capability/quota answers, Idempotency-Key'd
  ``POST /v2/runs``, run detail summary, terminal settlement facts). Pinned: the
  full custody chain in the canonical events.jsonl (START_REQUESTED → STARTED →
  LEDGER_RECORDED → SETTLED, one run id, wire Idempotency-Key == the durable
  invocation id), the wire body the SHAPE derived (authPreference=subscription,
  mode=ask, access=readonly, pinned one-element ``harnesses`` == primaryHarness,
  no ``execution`` block for a read-only run), the honest requested-vs-applied
  ``state/subagent_last_delegation.json`` receipt, the terminal facts reaching
  the model (cost_final, FAKE_RUN_RESULT, the requested-vs-applied
  capability_delta disclosure). The first boot retires the fresh pre-POST refusal's
  registration. The second retires the finished survey's registration. The 400
  and pinned-profile 409 retain their keys, bodies and respective existing and
  new registrations. Three starts reach the wire across both boots.
* S12 — NO-ORPHANS RESTART RECOVERY: the nanny's ``delegate_wait`` on a hung run
  is a COLD sleep (owner Q2 = A): its process is released once the body accepted
  the run's custody, and it parks as a sleeper. The WHOLE server tree is then
  SIGKILLed (hard crash, no cleanup) and a new generation boots on the same clone
  + data root: the sleeper is still parked, its run still open and observed by the
  new body, never cancelled or settled (a boot sweep that reads it leaves it live),
  and nothing wakes the nanny. The original run then finishes; another boot settles
  it once (one settlement and ledger receipt, matching SQLite custody) and its
  terminal wakes the sleeper, which finishes with the owed wake, with EXACTLY ONE
  physical ``POST /v2/runs`` across all generations, no unsolicited cancellation,
  and no process outside the live server tree.
* S13 — SKILLS LIFECYCLE E2E over the same HTTP surface the UI drives: a local
  extension payload lands in ``skills/external/``, the review endpoint runs the
  REAL triad panel against the loopback stub (canned all-PASS skill verdict →
  durable review.json status=clean), the requested ``inject_chat`` permission is
  granted, enable live-loads the extension, a scripted task dispatches the
  extension tool (durable tools.jsonl row with the ABI-9
  ``tool_result_meta.extension_generation`` digest), the CPL-7 Model Experience
  prose is visible in the model's own context (Installed Skills section in a
  recorded agent body), disable unloads the surface, and delete removes payload
  AND state dir (this tree's uninstall contract). Its HOT-ADOPTION variant drops
  the restart: the worker pool is up and every registry closed before the
  payload exists, so the tool call can land only by a running worker noticing
  the server's published extension generation (W3B-F1) — pinned against the
  supervisor's own durable roster (no process replaced between enable and
  dispatch) and the adopting worker's own durable event.

The default-lane tests pin the FakeClaudexorDaemon contract WITH THE REAL
GATEWAY CLIENT (handshake, capability catalog through ``route_health``, project
registry, run replay semantics, typed refusals, cancel) and the canned skill
review verdict against the tree's own aggregation policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import signal
import sys
import uuid

import pytest

from tests._usage_store_testing import ledger_rows
from tests.system_e2e.harness import (
    KEYLESS_REVIEW_ROWS,
    LANE_MOCK,
    SKILL_REVIEW_MARKER,
    TRIAD_USER_MARKER,
    ArtifactOracle,
    ScriptedStubModel,
    body_text,
    classify_call,
    keyless_settings,
    message_text,
    pids_with_env_value,
    process_tree_pids,
    require_lane,
    scripted_completion,
    start_server,
    submit_running,
    wait_durable_result,
    wait_until,
    ws_url,
)
from tests.system_e2e.interfaces import (
    FAKE_HANG_MARKER,
    FAKE_REFUSE_MARKER,
    FakeClaudexorDaemon,
)

from devtools.benchmarks.common.server_runner import _api

# ===========================================================================
# Default lane: the fake-daemon contract, pinned with the REAL gateway client.
# ===========================================================================


def test_fake_daemon_serves_the_real_gateway_contract(tmp_path):
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at
    from ouroboros.subagent_route_health import route_health
    from ouroboros.subagents import delegated_run_shape

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        descriptor = daemon.install(tmp_path / "cx")
        assert descriptor.is_file()
        endpoint = discover_daemon_at(tmp_path / "cx")
        with ClaudexorGateway(endpoint) as gateway:
            body = gateway.handshake()
            assert body.get("compatible") is True
            assert gateway.engine_version == daemon.engine_version
            assert all(op["path"] != "/v2/runs/lookup" for op in gateway.operations())
            # The dispatcher's ONE health reader sees a healthy route: catalog row
            # present + enabled, quota empty => fail-open healthy.
            assert route_health(
                gateway, daemon.harness_id, delegated_run_shape(False),
                route_model="mock-model",
            ) == ("", "")
            # Project registry: idempotent per root, Idempotency-Key required.
            project_id = gateway.register_project(str(tmp_path))
            assert project_id
            assert gateway.find_project_id(str(tmp_path)) == project_id
            handshake_posts = daemon.calls("POST", "/v2/handshake")
            assert handshake_posts
            assert all(int(p["body"].get("protocolMajor") or 0) == 3 for p in handshake_posts)
            assert all(p["protocol_major"] == "3" for p in daemon.run_start_posts() or handshake_posts)


def test_fake_daemon_run_replay_refusals_and_cancel(tmp_path):
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway,
        ClaudexorUnavailable,
        discover_daemon_at,
        final_attempt_facts,
    )

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs", start_lookup_unavailable=True) as daemon:
        daemon.install(tmp_path / "cx")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "cx")) as gateway:
            gateway.handshake()
            project_id = gateway.register_project(str(tmp_path))
            request = {
                "prompt": "probe", "instructions": "i", "authPreference": "subscription",
                "mode": "ask", "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "readonly", "maxSeconds": 60,
            }
            handle = gateway.start_run(request, idempotency_key="inv-e2e-1")
            run_id = str(handle.get("runId") or "")
            assert run_id
            # Engine replay contract: same key + byte-identical body -> the
            # ORIGINAL handle; same key + different digest -> 409 typed.
            assert gateway.start_run(request, idempotency_key="inv-e2e-1") == handle
            with pytest.raises(ClaudexorUnavailable) as conflict:
                gateway.start_run({**request, "prompt": "different"},
                                  idempotency_key="inv-e2e-1")
            assert conflict.value.code == "idempotency_conflict"
            assert conflict.value.status_code == 409
            # Scripted typed refusals: prompt marker (400) and ghost pin (409).
            with pytest.raises(ClaudexorUnavailable) as refused:
                gateway.start_run({**request, "prompt": FAKE_REFUSE_MARKER + " no"},
                                  idempotency_key="inv-e2e-2")
            assert refused.value.code == "fake_route_refused"
            assert refused.value.status_code == 400
            with pytest.raises(ClaudexorUnavailable) as ghost:
                gateway.start_run(
                    {**request, "credentialProfileId": daemon.ghost_profile},
                    idempotency_key="inv-e2e-3")
            assert ghost.value.code == "credential_profile_unknown"
            assert ghost.value.status_code == 409
            operation, = [op for op in gateway.operations() if op["path"] == "/v2/runs/lookup"]
            assert (operation["method"], operation["mutability"], operation["idempotency"]) == (
                "POST", "read_only", "key_required")
            posts = daemon.run_start_posts()
            refused_posts = posts[-2:]
            assert [post["idempotency_key"] for post in refused_posts] == ["inv-e2e-2", "inv-e2e-3"]
            for post in refused_posts:
                with pytest.raises(ClaudexorUnavailable) as unreadable:
                    gateway.lookup_run(post["body"], idempotency_key=post["idempotency_key"])
                assert (unreadable.value.status_code, unreadable.value.code) == (503, "lookup_unavailable")
            lookups = daemon.calls("POST", "/v2/runs/lookup")
            assert [(row["idempotency_key"], row["body"]) for row in lookups] == [
                (post["idempotency_key"], post["body"]) for post in refused_posts]
            assert all((row["status"], row["response"]["code"]) == (503, "lookup_unavailable") for row in lookups)
            assert daemon.run_start_posts() == posts
            # Success run: terminal detail carries the settlement facts.
            detail = gateway.get_run(run_id)
            summary = detail.get("summary") or {}
            assert summary.get("state") == "succeeded"
            assert summary.get("model") == daemon.applied_model
            from ouroboros.gateways.claudexor import final_attempt_facts

            assert final_attempt_facts(detail, run_id) == {
                "attempt_id": "a01", "harness_id": daemon.harness_id,
                "model": daemon.applied_model, "profile_id": daemon.applied_profile,
            }
            assert summary.get("spendUsd") == 0.0 and summary.get("spendEstimated") is False
            assert (summary.get("authRoute") or {}).get("profileId") == daemon.applied_profile
            assert final_attempt_facts(detail, run_id) == {
                "attempt_id": "a01", "harness_id": daemon.harness_id,
                "model": daemon.applied_model, "profile_id": daemon.applied_profile,
            }
            assert detail.get("primaryOutput", {}).get("truncated") is False
            # A hung run stays live until cancelled; cancel flips it terminal.
            hung = gateway.start_run({**request, "prompt": FAKE_HANG_MARKER + " go"},
                                     idempotency_key="inv-e2e-4")
            hung_id = str(hung.get("runId") or "")
            assert (gateway.get_run(hung_id).get("summary") or {}).get("state") == "running"
            with pytest.raises(ClaudexorUnavailable) as active:
                gateway.remove_project(project_id)
            assert (active.value.code, active.value.status_code) == ("project_has_active_run", 409)
            assert gateway.find_project_id(str(tmp_path)) == project_id
            receipt = gateway.cancel_run(hung_id, reason="e2e probe")
            assert receipt.get("accepted") is True
            assert (gateway.get_run(hung_id).get("summary") or {}).get("state") == "cancelled"
            daemon.set_busy_project(tmp_path)
            with pytest.raises(ClaudexorUnavailable) as busy:
                gateway.remove_project(project_id)
            assert (busy.value.status_code, busy.value.code) == (409, "project_has_active_run")
            assert gateway.find_project_id(str(tmp_path)) == project_id
            other = gateway.register_project(str(tmp_path / "other"))
            assert gateway.remove_project(other)["registryRemoved"] is True
            daemon.set_busy_project(None)
            removed = gateway.remove_project(project_id)
            assert removed["projectId"] == project_id and removed["registryRemoved"] is True
            assert removed["artifactsRetained"] is True and removed["root"] == str(tmp_path)
            assert not gateway.find_project_id(str(tmp_path))


def test_skill_review_classification_and_canned_verdict():
    """The skill-review branch classifies BEFORE every other marker branch (its
    pack can quote them), and its findings-only clean answer is a responsive
    PASS under the same parser the skill review wave uses."""
    from ouroboros.triad_review import parse_seat_answers

    body = {"messages": [{"role": "user", "content":
                          SKILL_REVIEW_MARKER + "\npack quoting: " + TRIAD_USER_MARKER
                          + " [FINALIZE_NOW]"}]}
    assert classify_call(body) == "skill_review"
    kind, message = scripted_completion(body, 1, lambda _b: None, "x")
    assert kind == "skill_review"
    assert message["content"] == "[]\nNO_FINDINGS"
    parsed = parse_seat_answers(
        {"results": [{"slot_id": "review-t1", "model": "stub", "text": message["content"]}]},
        {"review-t1": ("skill",)})
    assert parsed.findings == [] and not parsed.errors
    assert parsed.responsive_models == ["stub [review-t1]"]
    answer = parsed.actor_records[0].answers["skill"]
    assert answer["status"] == "responded" and answer["verdict"] == "PASS", answer


# ===========================================================================
# Shared delegated-transport pieces
# ===========================================================================

_RUN_ID_RE = re.compile(r'"run_id": "([0-9a-f]{16})"')


def _wait_step(body: dict) -> dict:
    ids = _RUN_ID_RE.findall(body_text(body))
    if not ids:
        return {"final": "E2E_SCRIPT_ERROR: no delegated run id visible in the transcript"}
    return {"tool": "delegate_wait", "arguments": {"run_id": ids[-1]}}


def _roster(*rows: dict) -> str:
    return json.dumps({"enabled": True, "items": list(rows)})


_SCOUT_ROW = {
    "subagent_id": "cx-scout",
    "recommended_use": "Delegated scout for the system_e2e transport scenarios.",
    "route": {"kind": "agent_session", "target_id": "fake-harness=mock-model"},
    "access": "readonly",  # these wire/custody scenarios deliberately use ask mode (#1726)
    "effort": "low",
}
_PINNED_ROW = {
    "subagent_id": "cx-pinned",
    "recommended_use": "Pinned-profile refusal probe for the system_e2e transport scenarios.",
    "route": {"kind": "agent_session", "target_id": "fake-harness=mock-model",
              "credential_profile_id": "ghost-profile"},
    "access": "readonly",
    "effort": "low",
}


def _custody_rows(oracle: ArtifactOracle, event_type: str, run_id: str = "") -> list:
    return [row for row in oracle.events(event_type)
            if not run_id or str(row.get("run_id") or "") == run_id]


# ===========================================================================
# S11 — delegated transport: full nanny run + typed refusals
# ===========================================================================

S11_PARENT_MARKER = "S11_PARENT_FINAL_e2e_w3b"


def _s11_start(actor, prompt, **options):
    return {"tool": "delegate_start", "arguments": {"subagent_id": actor, "prompt": prompt, **options}}


def _s11_tool_messages(body):
    issued = {}
    for message in body["messages"]:
        if message["role"] == "assistant":
            issued = {call["id"]: call["function"]["name"] for call in message.get("tool_calls") or ()}
        elif message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id in issued:
                yield (call_id, issued.pop(call_id)), message
        elif message["role"] == "user":
            issued.clear()


def _s11_payload(message):
    # The producer's JSON can be followed by a host delivery note.
    return json.JSONDecoder().raw_decode(message_text(message).lstrip())[0]


def _s11_task_replies(oracle, task_id, objective, model_calls, *tools):
    calls = [(row["tool_call_id"], row["tool"]) for row in oracle.task_drive(task_id).tools_rows()
             if row.get("type") == "tool_call" and row["task_id"] == task_id
             and row["tool"] in ("delegate_start", "delegate_wait")]
    assert [name for _, name in calls] == list(tools), (task_id, calls)
    assert len({call_id for call_id, _ in calls}) == len(calls)
    replies = {}
    for kind, body in model_calls:
        # The scripted stub labels an ordinary tool-bearing turn by its answer: agent or final.
        if kind not in ("agent", "final") or not body.get("tools"):
            continue
        if not any(m["role"] == "user" and objective in message_text(m) for m in body["messages"]):
            continue
        for key, message in _s11_tool_messages(body):
            if key in calls and key not in replies:
                replies[key] = _s11_payload(message)
    assert set(calls) == replies.keys(), (task_id, set(calls) - replies.keys())
    return [replies[key] for key in calls]


def _assert_s11_removed(row, project_id):
    assert row["path"] == f"/v2/projects/{project_id}" and 200 <= row["status"] < 300
    assert row["response"]["projectId"] == project_id and row["response"]["registryRemoved"] is True


def _assert_s11_refusal(receipt, failure, *, actor, task, pin, outcome):
    assert receipt["selected_subagent_id"] == failure["selected_subagent_id"] == actor
    assert receipt["invocation_id"] == receipt["run_id"] == failure["invocation_id"]
    assert receipt["task_id"] == failure["task_id"] == task
    assert receipt["occurred_at"] == failure["ts"]
    assert receipt["outcome"] == outcome
    assert receipt["failure_code"] == failure["reason"]
    assert receipt["route"] == failure["route"] == "fake-harness"
    assert receipt["requested_model"] == failure["model"] == "mock-model"
    assert receipt["requested_profile"] == pin
    assert receipt["applied_model"] == receipt["applied_profile"] == ""


@pytest.mark.integration
def test_s11_delegated_transport_wire_custody_and_typed_refusals(
        e2e_clone, tmp_path_factory):
    from ouroboros.delegate_custody import invocation_record, replay
    from ouroboros.delegate_pending import pending_invocations, request_body
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at
    from ouroboros.subagent_history import subagent_last_delegation

    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s11")
    data_root = root / "data"
    survey, existing, pinned = root / "survey", root / "existing", root / "pinned"
    for workspace in (survey, existing, pinned):
        workspace.mkdir()
    document = pinned / "survey.txt"
    original_input = f"S11 survey {uuid.uuid4().hex}\n".encode()
    document.write_bytes(original_input)
    roster = _roster(_SCOUT_ROW, _PINNED_ROW)
    captured = {}
    oracle = ArtifactOracle(data_root)

    def json_value(value):
        return json.dumps(value, sort_keys=True)

    def events_by_type(oracle):
        by_type = {}
        custody_payloads = {}
        for row in oracle.events():
            kind = str(row.get("type") or "")
            event_id = row.get("event_id")
            if kind.startswith("delegate_run_") and event_id:
                # An outbox redelivery preserves the entire event, including JSON value types.
                payload = json_value(row)
                if event_id in custody_payloads:
                    assert payload == custody_payloads[event_id], f"Conflicting custody event {event_id}"
                    continue
                custody_payloads[event_id] = payload
            by_type.setdefault(kind, []).append(row)
        return by_type, custody_payloads

    def start_pinned(_body):
        captured["unrun"] = subagent_last_delegation(data_root)
        return _s11_start("cx-pinned", "Survey the selected file.", max_seconds=60,
                          input_files=[{"root": "active_workspace", "path": document.name}])

    def finish_survey(_body):
        # The settled run can still be continued at this point in its owner's turn.
        captured["success"] = subagent_last_delegation(data_root)
        captured["live_deletes"] = [row["path"] for row in daemon.calls("DELETE", "/v2/projects/")]
        return {"final": S11_PARENT_MARKER}

    def wait_for_survey(body):
        starts = [msg for (_, name), msg in _s11_tool_messages(body) if name == "delegate_start"]
        if len(starts) != 1 or not (run_id := _s11_payload(starts[0]).get("run_id")):
            return {"final": "E2E_SCRIPT_ERROR: expected one delegate_start result with a run id"}
        # The body collects the fake's immediate success independently. Ask for an
        # already-ready result so delegate_wait itself returns it in this call.
        if not wait_until(lambda: _custody_rows(oracle, "delegate_run_settled", run_id), timeout=60):
            return {"final": f"E2E_SCRIPT_ERROR: survey run {run_id} never settled"}
        return {"tool": "delegate_wait", "arguments": {"run_id": run_id}}

    script = [_s11_start("cx-scout", "Survey the repository root and report what you find."),
              wait_for_survey, finish_survey]
    with (
        FakeClaudexorDaemon(reports_created=True, require_registration=True, start_lookup_unavailable=True) as daemon,
        ScriptedStubModel(script, final_answer=S11_PARENT_MARKER) as stub,
    ):
        daemon.install(data_root / "claudexor")
        daemon.set_busy_project(survey)
        with ClaudexorGateway(discover_daemon_at(data_root / "claudexor")) as gateway:
            existing_id, created_here = gateway.project_registration(str(existing))
            assert created_here is True
        settings = keyless_settings(stub, OUROBOROS_SUBAGENTS=roster, OUROBOROS_TASK_ABS_CEILING_SEC="unlimited")
        server = start_server(e2e_clone, root, settings)
        try:
            objective = "Delegate the survey to your configured scout, wait for the result, then finish."
            task = server.submit(objective, workspace_root=str(survey), timeout_sec=0)
            stored = wait_durable_result(oracle, task, timeout=600)
            assert stored["status"] == "completed" and S11_PARENT_MARKER in stored["result"], stored
            assert stub.script_consumed()
            start_reply, wait_reply = _s11_task_replies(
                oracle, task, objective, stub.calls[:], "delegate_start", "delegate_wait")

            # This task adopts a registration that predates the host. The completed
            # survey has a separate, exclusively owned registration it may retire.
            with stub._lock:
                stub.script.append(_s11_start("cx-scout", FAKE_REFUSE_MARKER + " refuse this survey"))
            objective = "Delegate a survey of the existing project to the scout."
            existing_task = server.submit(objective, workspace_root=str(existing), timeout_sec=0)
            stored = wait_durable_result(oracle, existing_task, timeout=600)
            assert stored["status"] == "completed" and S11_PARENT_MARKER in stored["result"], stored
            assert stub.script_consumed()
            refused_reply, = _s11_task_replies(oracle, existing_task, objective, stub.calls[:], "delegate_start")

            # The engine has no keyed document import. Selecting a staged input
            # refuses before submission, retiring the new registration. The next
            # start on this same root must acquire its own, distinct registration.
            with stub._lock:
                stub.script.extend([_s11_start("cx-pinned", "Read the attached survey.",
                                              attachment_paths=["attachments/survey.txt"]), start_pinned])
            objective = "Try the configured pinned profile with the survey, then without the attachment."
            created = _api(server.base_url, "POST", "/api/tasks", {
                "description": objective,
                "workspace_mode": "external", "workspace_root": str(pinned), "memory_mode": "forked",
                "attachments": [{"path": str(document)}], "allow_partial_attachments": False,
            })
            pinned_task = created["task_id"]
            attachment, = created["attachment_manifest"]
            assert (attachment["status"], attachment["relpath"]) == ("staged", "attachments/survey.txt")
            stored = wait_durable_result(oracle, pinned_task, timeout=600)
            assert stored["status"] == "completed" and S11_PARENT_MARKER in stored["result"], stored
            assert stub.script_consumed()
            unrun_reply, ghost_reply = _s11_task_replies(
                oracle, pinned_task, objective, stub.calls[:], "delegate_start", "delegate_start")

            events, _ = events_by_type(oracle)
            refused, unrun, ghost = events["delegate_run_start_failed"]
            assert [row["reason"] for row in (refused, unrun, ghost)] == [
                "fake_route_refused", "document_import_unsupported", "credential_profile_unknown"]
            assert unrun["definite"] is True and unrun["project_retired"] is True
            assert all(row["definite"] is False and row["project_retired"] is False for row in (refused, ghost))
            blocked, = events["delegate_run_start_blocked"]
            assert (blocked["task_id"], blocked["reason"]) == (pinned_task, unrun["reason"])
            entry, = events["delegate_run_started"]
            run_id, success_id = entry["run_id"], entry["invocation_id"]
            assert re.fullmatch(r"[0-9a-f]{16}", run_id)
            posts = daemon.run_start_posts()
            assert [post["status"] for post in posts] == [200, 400, 409]
            wire_keys = [post["idempotency_key"] for post in posts]
            assert wire_keys == [success_id, refused["invocation_id"], ghost["invocation_id"]]
            all_keys = [unrun["invocation_id"], *wire_keys]
            assert all(all_keys) and len(set(all_keys)) == 4
            requested = events["delegate_run_start_requested"]
            assert [row["invocation_id"] for row in requested] == wire_keys
            success, adopted, owned = requested
            assert [row["task_id"] for row in requested] == [task, existing_task, pinned_task]
            assert entry["task_id"] == task and entry["selected_subagent_id"] == "cx-scout"
            assert (entry["route"], entry["model"], entry["access"], entry["mode"]) == (
                daemon.harness_id, "mock-model", "readonly", "ask")
            ledger, = events["delegate_run_ledger_recorded"]
            settled, = events["delegate_run_settled"]
            assert ledger["run_id"] == settled["run_id"] == run_id
            assert (settled["state"], settled["model"], settled["credential_profile_id"]) == (
                "succeeded", daemon.applied_model, daemon.applied_profile)
            assert settled["cost_usd"] == 0.0 and settled["cost_final"] is True

            registrations = daemon.calls("POST", "/v2/projects")
            registration_keys = [row["idempotency_key"] for row in registrations]
            assert all(registration_keys) and len(set(registration_keys)) == 5
            assert [(row["body"]["root"], row["response"]["created"]) for row in registrations] == [
                (str(existing), True), (str(survey), True), (str(existing), False), (str(pinned), True), (str(pinned), True)]
            seeded, project, borrowed, pre_project, held = [row["response"]["id"] for row in registrations]
            assert seeded == borrowed == existing_id
            assert all((project, borrowed, pre_project, held)) and len({project, borrowed, pre_project, held}) == 4
            assert (unrun["project_id"], refused["project_id"], ghost["project_id"]) == (pre_project, "", held)
            assert [(row["project_id"], row["project_owned"]) for row in requested] == [
                (project, True), (borrowed, False), (held, True)]
            assert entry["project_id"] == project and entry["project_owned"] is True
            assert captured["live_deletes"] == []

            receipt = captured["success"]
            assert (receipt["task_id"], receipt["invocation_id"], receipt["run_id"]) == (task, success_id, run_id)
            assert (receipt["outcome"], receipt["selected_subagent_id"], receipt["route"]) == (
                "succeeded", "cx-scout", daemon.harness_id)
            assert (receipt["requested_model"], receipt["applied_model"]) == ("mock-model", daemon.applied_model)
            assert (receipt["requested_profile"], receipt["applied_profile"]) == ("", daemon.applied_profile)
            _assert_s11_refusal(captured["unrun"], unrun, actor="cx-pinned", task=pinned_task,
                                pin=daemon.ghost_profile, outcome="not_started")
            receipt = subagent_last_delegation(data_root)
            latest = receipt["latest_by_subagent"]
            assert set(latest) == {"cx-scout", "cx-pinned"}
            _assert_s11_refusal(latest["cx-scout"], refused, actor="cx-scout", task=existing_task,
                                pin="", outcome="unknown")
            _assert_s11_refusal(latest["cx-pinned"], ghost, actor="cx-pinned", task=pinned_task,
                                pin=daemon.ghost_profile, outcome="unknown")
            assert {k: v for k, v in receipt.items() if k != "latest_by_subagent"} == latest["cx-pinned"]
            assert (start_reply["status"], start_reply["run_id"], start_reply["invocation_id"]) == (
                "started", run_id, success_id)
            assert (wait_reply["status"], wait_reply["run_id"], wait_reply["state"]) == ("terminal", run_id, "succeeded")
            assert f"FAKE_RUN_RESULT {run_id}" in wait_reply["primary_output"]["text"]
            assert wait_reply["cost"]["cost_final"] is True
            assert any(delta["reason"] == "session_route_resolves_its_own_model" for delta in wait_reply["capability_delta"])
            for reply, row in ((refused_reply, refused), (unrun_reply, unrun), (ghost_reply, ghost)):
                assert (reply["status"], reply["tool"], reply["reason"]) == ("refused", "delegate_start", row["reason"])
                assert reply["project_retired"] is row["project_retired"]
            assert unrun_reply["definitely_unrun"] is True and "pending_invocation_id" not in unrun_reply
            assert (unrun_reply["project_id"], ghost_reply["project_id"]) == (pre_project, held)
            for reply, row in ((refused_reply, refused), (ghost_reply, ghost)):
                assert reply["pending_invocation_id"] == row["invocation_id"]
                assert reply.get("definitely_unrun", False) is False
            assert {row["invocation_id"] for row in pending_invocations(data_root)} == set(wire_keys[1:])
        finally:
            server.stop()

        # Include A's shutdown. An early sweep could try the busy survey, but
        # only the pre-POST registration could actually be removed in A.
        assert not server.stop_unconfirmed and server.proc.poll() is not None
        deletes_a = daemon.calls("DELETE", "/v2/projects/")
        survey_path = f"/v2/projects/{project}"
        pre_delete, = [row for row in deletes_a if row["path"] != survey_path]
        _assert_s11_removed(pre_delete, pre_project)
        for row in deletes_a:
            if row["path"] == survey_path:
                assert (row["status"], row["response"]["code"]) == (409, "project_has_active_run")
        events_a, original_events = events_by_type(oracle)
        assert not events_a.get("delegate_run_project_retired")
        before_boot = replay(data_root)[run_id]
        assert before_boot.settled and before_boot.ledger_recorded and before_boot.project_owned
        assert before_boot.project_id == project
        with ClaudexorGateway(discover_daemon_at(data_root / "claudexor")) as gateway:
            assert gateway.find_project_id(str(survey)) == project
        previous_lookups = len(daemon.calls("POST", "/v2/runs/lookup"))
        document.write_text("Revised after the original submission.\n", encoding="utf-8")
        # Boot must read both original requests. An unavailable lookup leaves them pending.
        daemon.set_busy_project(None)
        with ScriptedStubModel([]) as stub_b:
            server_b = start_server(e2e_clone, root, keyless_settings(stub_b, OUROBOROS_SUBAGENTS=roster))
            try:
                def boot_observed():
                    reads = daemon.calls("POST", "/v2/runs/lookup")[previous_lookups:]
                    return (set(wire_keys[1:]) <= {row["idempotency_key"] for row in reads}
                            and _custody_rows(oracle, "delegate_run_project_retired", run_id))

                assert wait_until(boot_observed, timeout=60), "missing boot lookup or settled project retirement"
                boot_events, restored_events = events_by_type(ArtifactOracle(server_b.data_root))
                for event_id, payload in original_events.items():
                    assert restored_events[event_id] == payload, f"Changed custody event {event_id} after restart"
                assert [row["event_id"] for row in boot_events["delegate_run_start_requested"]] == [
                    row["event_id"] for row in requested]
                assert [row["event_id"] for row in boot_events["delegate_run_start_blocked"]] == [blocked["event_id"]]
                retired, = boot_events["delegate_run_project_retired"]
                assert retired["event_id"] not in original_events
                assert (retired["task_id"], retired["run_id"], retired["project_id"]) == (task, run_id, project)
                assert not retired.get("project_kept")
                bound, = replay(data_root).values()
                assert bound.settled and bound.ledger_recorded and not bound.project_owned
                assert (bound.task_id, bound.run_id, bound.invocation_id, bound.project_id) == (
                    task, run_id, success_id, project)
                assert json_value(daemon.calls("POST", "/v2/projects")) == json_value(registrations)
                deletions = daemon.calls("DELETE", "/v2/projects/")
                boot_delete, = deletions[len(deletes_a):]
                _assert_s11_removed(boot_delete, project)
                with ClaudexorGateway(discover_daemon_at(data_root / "claudexor")) as gateway:
                    assert gateway.find_project_id(str(existing)) == borrowed
                    assert gateway.find_project_id(str(pinned)) == held
                    assert not gateway.find_project_id(str(survey))
                assert json_value(subagent_last_delegation(data_root)) == json_value(receipt)

                assert json_value(daemon.run_start_posts()) == json_value(posts)
                assert posts[0]["response"]["runId"] == run_id
                assert [post["response"]["code"] for post in posts[1:]] == [refused["reason"], ghost["reason"]]
                for row, post in zip(requested, posts, strict=True):
                    body = post["body"]
                    assert json_value(request_body(data_root, row)) == json_value(body)
                    assert body["authPreference"] == "subscription"
                    assert body["mode"] == "ask" and body["access"] == "readonly" and "execution" not in body
                    assert body["harnesses"] == [body["primaryHarness"]] == [daemon.harness_id]
                    assert body["model"] == "mock-model" and body["instructions"].strip()
                    assert body["scope"]["kind"] == "project"
                    if row["max_seconds_basis"] == "unbounded":
                        assert row["max_seconds"] == 0 and "maxSeconds" not in body
                    else:
                        assert body["maxSeconds"] == row["max_seconds"] > 0
                assert [row["max_seconds_basis"] for row in requested] == ["unbounded", "unbounded", "requested"]
                assert entry["max_seconds_basis"] == success["max_seconds_basis"] and entry["max_seconds"] == 0
                assert [post["body"]["scope"]["root"] for post in posts] == [str(survey), str(existing), str(pinned)]
                assert [post["body"].get("credentialProfileId", "") for post in posts] == ["", "", daemon.ghost_profile]
                assert FAKE_REFUSE_MARKER in posts[1]["body"]["prompt"] and posts[2]["body"]["maxSeconds"] == 60

                # Both open requests keep the original body and registration.
                pending = {row["invocation_id"]: row for row in pending_invocations(data_root)}
                assert set(pending) == set(wire_keys[1:])
                lookups = daemon.calls("POST", "/v2/runs/lookup")[previous_lookups:]
                assert {row["idempotency_key"] for row in lookups} == set(pending)
                for lookup in lookups:
                    assert lookup["status"] == 503 and lookup["response"]["code"] == "lookup_unavailable"
                    assert json_value(lookup["body"]) == json_value(request_body(data_root, pending[lookup["idempotency_key"]]))
                for row, post in zip((adopted, owned), posts[1:], strict=True):
                    survivor = pending[post["idempotency_key"]]
                    assert json_value(request_body(data_root, survivor)) == json_value(post["body"])
                    assert (survivor["task_id"], survivor["project_id"], survivor["project_owned"]) == (
                        row["task_id"], row["project_id"], row["project_owned"])
                selected, = invocation_record(data_root, ghost["invocation_id"])["input_files"]
                assert [selected] == owned["input_files"]
                assert selected["source"] == str(document)
                assert selected["sha256"] == hashlib.sha256(original_input).hexdigest()
                assert selected["local"] in posts[2]["body"]["instructions"]
                assert pathlib.Path(selected["retained"]).read_bytes() == original_input
            finally:
                server_b.stop()
        assert not server_b.stop_unconfirmed and server_b.proc.poll() is not None
        assert json_value(daemon.run_start_posts()) == json_value(posts)
        assert json_value(daemon.calls("DELETE", "/v2/projects/")) == json_value(deletions)


# ===========================================================================
# S12 — SIGKILL mid-run -> restart -> boot custody sweep, no orphans
# ===========================================================================

S12_SCRIPT = [
    {"tool": "delegate_start", "arguments": {
        "subagent_id": "cx-scout",
        "prompt": FAKE_HANG_MARKER + " keep surveying until told otherwise"}},
    _wait_step,
]


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "linux",
                    reason="process-group SIGKILL + /proc scans are Linux-only")
def test_s12_sigkill_mid_run_restart_recovers_custody_without_orphans(
        e2e_clone, tmp_path_factory):
    from ouroboros.delegate_custody import open_runs, replay

    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s12")
    data_root = pathlib.Path(root) / "data"
    roster = _roster(_SCOUT_ROW)
    with FakeClaudexorDaemon() as daemon:
        daemon.install(data_root / "claudexor")
        run_id = ""
        with ScriptedStubModel(S12_SCRIPT) as stub:
            server = start_server(e2e_clone, root, keyless_settings(
                stub, OUROBOROS_SUBAGENTS=roster))
            oracle = ArtifactOracle(server.data_root)
            killed = False
            try:
                task_id = submit_running(server, "Delegate a long survey and babysit it until stopped.")
                # The physical run is LIVE: durable STARTED row, and the body is actually
                # observing it on the daemon (wire GETs observed).
                assert wait_until(lambda: _custody_rows(oracle, "delegate_run_started"), 180), (
                    "delegated run never reached a durable STARTED row")
                run_id = str(_custody_rows(oracle, "delegate_run_started")[-1]["run_id"])
                assert wait_until(lambda: daemon.calls("GET", f"/v2/runs/{run_id}"), 60)
                entered = wait_until(lambda: _custody_rows(
                    oracle, "delegate_supervision_wait_entered", run_id), 60)
                assert entered and entered[-1].get("mode") == "cold", entered
                # The wait with no named mode sleeps cold: the nanny is parked, its process released.
                assert wait_until(lambda: oracle.task_result(task_id).get("reason_code") == "model_sleeping", 120), (
                    oracle.task_result(task_id))
                assert len(daemon.run_start_posts()) == 1

                # HARD CRASH: SIGKILL the whole server tree mid-wait, no cleanup.
                pids = process_tree_pids(server.proc.pid)
                os.killpg(os.getpgid(server.proc.pid), signal.SIGKILL)
                for pid in pids:            # belt: any child outside the group
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                server.proc.wait(timeout=30)
                killed = True
                assert wait_until(
                    lambda: not pids_with_env_value(str(server.data_root)), 30), (
                    "processes carrying the data root survived the SIGKILL")
            finally:
                if not killed:
                    server.stop()

        # A crash is not an owner Stop and no wake: the sleeper stays parked and its
        # healthy leaf live (ARCHITECTURE §6 "Durable custody"), observed by the new body.
        observed_before = len(daemon.calls("GET", f"/v2/runs/{run_id}"))
        with ScriptedStubModel([]) as stub_b:
            server_b = start_server(e2e_clone, root, keyless_settings(
                stub_b, OUROBOROS_SUBAGENTS=roster))
            try:
                oracle = ArtifactOracle(server_b.data_root)
                assert wait_until(lambda: len(daemon.calls("GET", f"/v2/runs/{run_id}")) > observed_before, 240), (
                    "the new body never observed the sleeper's run")
                reconciled = _custody_rows(oracle, "delegate_run_reconciled", run_id)
                assert all(row.get("action") == "left_live" for row in reconciled), reconciled
                held = oracle.task_result(task_id)
                assert held.get("status") == "scheduled" and held.get("reason_code") == "model_sleeping", held
                assert {run.run_id for run in open_runs(data_root)} == {run_id}
                assert replay(data_root)[run_id].task_id == task_id
                assert not _custody_rows(oracle, "delegate_run_cancel_outcome", run_id)
                assert not _custody_rows(oracle, "delegate_run_settled", run_id)
                assert not daemon.calls("POST", f"/v2/runs/{run_id}/control")
                assert len(daemon.run_start_posts()) == 1, (
                    "restart recovery re-POSTed a run that already had a STARTED row")
                assert not stub_b.calls, "a crash wakes no sleeper: its run has not ended"

                # No orphans: every live pid carrying this data root sits inside
                # the gen-B server tree.
                assert wait_until(
                    lambda: set(pids_with_env_value(str(server_b.data_root)))
                    <= set(process_tree_pids(server_b.proc.pid)), 45), (
                    "a process carrying the data root lives outside the gen-B tree")
            finally:
                server_b.stop()

        # The engine completes the ORIGINAL run. A further boot's body settles it, and its
        # terminal wakes the sleeper, which finishes with the owed wake in its next request.
        with daemon._lock:
            daemon.runs[run_id]["hang"] = False
        with ScriptedStubModel([]) as stub_c:
            server_c = start_server(e2e_clone, root, keyless_settings(
                stub_c, OUROBOROS_SUBAGENTS=roster))
            try:
                oracle = ArtifactOracle(data_root)
                assert wait_until(lambda: _custody_rows(oracle, "delegate_run_settled", run_id), 240)
                result = server_c.wait_task(task_id, timeout=600)
                assert result.get("status") == "completed", result
                assert stub_c.calls, "the run's terminal woke the sleeping nanny"
                settled = _custody_rows(oracle, "delegate_run_settled", run_id)
                assert len(settled) == 1 and settled[0].get("state") == "succeeded", settled
                assert len(_custody_rows(oracle, "delegate_run_ledger_recorded", run_id)) == 1
                sessions = [row for row in ledger_rows(data_root) if row.get("kind") == "subscription_session"]
                assert len(sessions) == 1 and sessions[0]["task_id"] == task_id, sessions
                assert sessions[0]["session_id_sha256"] == hashlib.sha256(run_id.encode()).hexdigest()
                assert sessions[0]["cost_final"] is True and sessions[0]["cost_usd"] == 0.0
                custody = replay(data_root)[run_id]
                assert custody.settled and custody.ledger_recorded and custody.terminal_state == "succeeded"
                assert not open_runs(data_root)
                assert not _custody_rows(oracle, "delegate_run_cancel_outcome", run_id)
                assert not daemon.calls("POST", f"/v2/runs/{run_id}/control")
                assert len(daemon.run_start_posts()) == 1
                assert wait_until(
                    lambda: set(pids_with_env_value(str(data_root)))
                    <= set(process_tree_pids(server_c.proc.pid)), 45)
            finally:
                server_c.stop()
        assert wait_until(lambda: not pids_with_env_value(str(data_root)), 30)


# ===========================================================================
# S13 — skills lifecycle E2E + Model Experience (CPL-7) on a live server
# ===========================================================================

S13_SKILL = "e2e_probe"
S13_MX_PROSE = "E2E-MX-MARKER adds a loopback probe echo tool to the toolbox"
S13_SKILL_MD = f"""---
name: {S13_SKILL}
description: Loopback E2E probe extension for the system_e2e lifecycle scenario.
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.0"
permissions: ["tool", "inject_chat"]
model_experience:
  what_model_sees: '{S13_MX_PROSE}'
  token_effect: 'one catalogue line'
---
E2E probe extension body.
"""
S13_PLUGIN = (
    "def _echo(ctx, message='hi'):\n"
    "    return f'echo: {message}'\n"
    "\n"
    "def register(api):\n"
    "    api.register_tool(\n"
    "        'echo', _echo, description='echo probe',\n"
    "        schema={'type': 'object', 'properties': {'message': {'type': 'string'}}},\n"
    "    )\n"
)
S13_DISPATCH_MARKER = "S13_DISPATCH_DONE_e2e_w3b"


def _skill_entry(server, name: str) -> dict:
    listing = _api(server.base_url, "GET", "/api/extensions", timeout=30)
    rows = listing if isinstance(listing, list) else (
        listing.get("extensions") or listing.get("skills") or listing.get("catalog") or [])
    for row in rows:
        if isinstance(row, dict) and str(row.get("name") or "") == name:
            return row
    return {}


@pytest.mark.integration
def test_s13_skills_lifecycle_review_grants_enable_dispatch_disable_delete(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    from ouroboros.extension_loader import extension_surface_name

    surface = extension_surface_name(S13_SKILL, "echo")
    root = tmp_path_factory.mktemp("s13")

    # ---- Phase 1: install payload -> review -> grants -> enable (live server A).
    with ScriptedStubModel([]) as stub_a:
        server = start_server(e2e_clone, root, keyless_settings(stub_a))
        try:
            data_root = pathlib.Path(server.data_root)
            oracle = ArtifactOracle(data_root)
            payload_dir = data_root / "skills" / "external" / S13_SKILL
            payload_dir.mkdir(parents=True)
            (payload_dir / "SKILL.md").write_text(S13_SKILL_MD, encoding="utf-8")
            (payload_dir / "plugin.py").write_text(S13_PLUGIN, encoding="utf-8")

            # REVIEW: the real triad panel over the loopback stub's canned verdict.
            review = _api(server.base_url, "POST",
                          f"/api/skills/{S13_SKILL}/review", {}, timeout=600)
            assert review.get("status") == "clean", review
            # Findings are problems only (#2123). A clean projection keeps its
            # explicit status; prove every seat answered PASS in the canonical
            # ledger, so an empty or missing review cannot pass this guard.
            review_state = oracle._json(f"state/skills/{S13_SKILL}/review.json")
            assert review_state.get("status") == "clean", review_state
            assert review_state.get("findings") == [], review_state
            content_hash = review_state.get("content_hash")
            assert content_hash and content_hash == review.get("content_hash"), review_state
            record_id = review_state.get("record_id")
            assert record_id and record_id == review.get("record_id"), review_state
            record = oracle._json(f"state/review_ledger/{record_id}.json")
            assert record.get("state") == "settled", record
            subject = record.get("subject") or {}
            assert subject.get("skill") == S13_SKILL and subject.get("tree_sha") == content_hash, subject
            verdict = record.get("verdict") or {}
            assert verdict.get("aggregate") == "PASS", verdict
            assert verdict.get("per_row") == dict.fromkeys(KEYLESS_REVIEW_ROWS, "PASS"), verdict
            seats = record.get("rows") or []
            assert len(seats) == len(KEYLESS_REVIEW_ROWS), seats
            assert {seat.get("seat_id") for seat in seats} == set(KEYLESS_REVIEW_ROWS), seats
            for seat in seats:
                answer = (seat.get("answers") or {}).get("skill") or {}
                assert seat.get("status") == answer.get("status") == "responded", seat
                assert answer.get("verdict") == "PASS" and answer.get("findings") == [], seat
            assert stub_a.kinds().count("skill_review") >= len(KEYLESS_REVIEW_ROWS), stub_a.kinds()

            # GRANTS: the manifest-requested privileged permission, owner-granted.
            grants = _api(server.base_url, "POST", f"/api/skills/{S13_SKILL}/grants",
                          {"items": ["inject_chat"]}, timeout=120)
            assert (grants.get("grants") or {}).get("all_granted") is True, grants
            grants_state = oracle._json(f"state/skills/{S13_SKILL}/grants.json")
            assert grants_state.get("granted_permissions") == ["inject_chat"], grants_state

            # ENABLE: the live reconcile loads the extension IN THE SERVER PROCESS.
            toggled = _api(server.base_url, "POST", f"/api/skills/{S13_SKILL}/toggle",
                           {"enabled": True}, timeout=300)
            assert toggled.get("enabled") is True, toggled
            assert not toggled.get("error"), toggled
            entry = _skill_entry(server, S13_SKILL)
            assert entry.get("live_loaded") is True, entry
            assert entry.get("dispatch_live") is True, entry
        finally:
            server.stop()

    # ---- Phase 2: restart, then dispatch + Model Experience, then disable and
    # delete. The restart here pins that the ENABLED state survives a reboot; it
    # is no longer the only way a worker sees the skill — the hot-adoption
    # variant below removes it (W3B-F1).
    script = [{"tool": surface, "arguments": {"message": "ping-e2e"}}]
    with ScriptedStubModel(script,
                           final_answer=f"{S13_DISPATCH_MARKER}: echo absorbed.") as stub:
        server = start_server(e2e_clone, root, keyless_settings(stub))
        try:
            data_root = pathlib.Path(server.data_root)
            oracle = ArtifactOracle(data_root)
            payload_dir = data_root / "skills" / "external" / S13_SKILL
            entry = _skill_entry(server, S13_SKILL)
            assert entry.get("enabled") is True, entry
            assert entry.get("review_status") == "clean", entry
            assert entry.get("live_loaded") is True, entry

            # DISPATCH through a live task: durable tools.jsonl row + ABI-9 digest.
            task_id = submit_running(server, "Call the probe echo tool once, then finish.")
            result = server.wait_task(task_id, timeout=300)
            assert result.get("status") == "completed", result
            stored = wait_durable_result(oracle, task_id)
            assert S13_DISPATCH_MARKER in str(stored.get("result") or ""), stored
            assert stub.script_consumed(), "S13 dispatch script was not fully consumed"
            task_oracle = oracle.task_drive(task_id)
            rows = [row for row in task_oracle.tools_rows()
                    if str(row.get("tool") or "") == surface]
            assert rows, "extension tool call missing from the task tools log"
            assert "echo: ping-e2e" in json.dumps(rows), rows
            meta = rows[-1].get("tool_result_meta") or {}
            digest = str(meta.get("extension_generation") or "")
            assert re.fullmatch(r"[0-9a-f]{8,64}", digest), (
                f"ABI-9 generation digest missing from the durable row: {rows[-1]}")

            # CPL-7 Model Experience: the prose is IN the model's own context.
            agent_texts = "\n".join(
                body_text(call_body) for kind, call_body in stub.calls
                if kind in ("agent", "final"))
            assert "Model experience:" in agent_texts, (
                "Installed Skills section carries no Model experience line")
            assert "E2E-MX-MARKER" in agent_texts

            # DISABLE: the surface leaves the live registry of this process.
            untoggled = _api(server.base_url, "POST", f"/api/skills/{S13_SKILL}/toggle",
                             {"enabled": False}, timeout=300)
            assert not untoggled.get("error"), untoggled
            entry = _skill_entry(server, S13_SKILL)
            assert entry.get("enabled") is False, entry
            assert entry.get("live_loaded") is False, entry
            assert entry.get("dispatch_live") is False, entry

            # DELETE: payload AND durable state dir removed (uninstall contract).
            deleted = _api(server.base_url, "POST", f"/api/skills/{S13_SKILL}/delete",
                           {}, timeout=120)
            assert not deleted.get("error"), deleted
            assert not payload_dir.exists(), "payload dir survived delete"
            assert not (data_root / "state" / "skills" / S13_SKILL).exists(), (
                "state dir survived delete")
            assert _skill_entry(server, S13_SKILL) == {}, "deleted skill still listed"
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# S13, hot-adoption variant — the SAME lifecycle with the restart removed.
# ---------------------------------------------------------------------------

S13H_SKILL = "e2e_hot_probe"
S13H_SKILL_MD = f"""---
name: {S13H_SKILL}
description: Loopback E2E probe extension enabled AFTER the worker pool spawned.
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.0"
permissions: ["tool"]
---
E2E hot-adoption probe extension body.
"""
S13H_DISPATCH_MARKER = "S13H_HOT_DISPATCH_DONE_e2e_w3b"
# The probe's echo names the process that ran it: an extension handler runs in-process in
# the worker carrying the task (``ouroboros/tools/extension_dispatch.py``), so the durable
# tools row itself says which process adopted the skill.
S13H_PLUGIN = (
    "import os\n"
    "\n"
    "def _echo(ctx, message='hi'):\n"
    "    return f'echo: {message} pid={os.getpid()}'\n"
    "\n"
    "def register(api):\n"
    "    api.register_tool(\n"
    "        'echo', _echo, description='echo probe',\n"
    "        schema={'type': 'object', 'properties': {'message': {'type': 'string'}}},\n"
    "    )\n"
)
_S13H_PID_RE = re.compile(r"echo: hot-ping pid=(\d+)")


def _worker_boot_pids(oracle: ArtifactOracle) -> set:
    return {int(row.get("pid") or 0) for row in oracle.events("worker_ready")}


def _worker_ready_count(oracle: ArtifactOracle, pid: int) -> int:
    return sum(1 for row in oracle.events("worker_ready") if int(row.get("pid") or 0) == pid)


def _owner_turn_id(oracle: ArtifactOracle, client_message_id: str) -> str:
    for row in oracle.events("task_received"):
        task = row.get("task") or {}
        origin = (task.get("metadata") or {}).get("origin_message_ref") or {}
        if origin.get("client_message_id") == client_message_id:
            return str(task.get("id") or "")
    return ""


@pytest.mark.integration
def test_s13_hot_enable_reaches_an_already_spawned_worker_without_a_restart(
        e2e_clone, tmp_path_factory):
    """The W3B-F1 defect, at the product surface: enable AFTER boot must reach the
    workers that are ALREADY running.

    The lifecycle scenario above proves the same skill dispatches after a
    restart, which is exactly what made the defect invisible: a task worker
    loaded extensions once, at spawn, so a skill enabled later stayed unknown to
    every task that worker went on to serve while ``/api/extensions`` reported
    it live. Here the pool spawned BEFORE the payload existed and no process is
    replaced, so the tool call can only succeed by the worker noticing the
    server's published generation and adopting it.

    Since #2654 the pool at rest is the conversation lane's one ready spare: project
    workers are bought on demand and retired when idle, so a pooled task dispatched
    after the enable always cold-starts a worker that loads the skill at spawn and
    proves nothing. The probe is therefore the owner's Main turn over the SAME /ws the
    SPA opens: it takes the spare that booted before the payload existed. The lane
    replenishes its spare at once, so a NEW worker appears; the proof is not "no new
    PID" but the tools row naming the process that ran the echo — a worker that booted
    once, before the payload — together with that worker's recorded adoption.
    """
    require_lane(LANE_MOCK)
    from ouroboros.extension_loader import extension_surface_name

    surface = extension_surface_name(S13H_SKILL, "echo")
    root = tmp_path_factory.mktemp("s13h")
    script = [{"tool": surface, "arguments": {"message": "hot-ping"}}]
    with ScriptedStubModel(script,
                           final_answer=f"{S13H_DISPATCH_MARKER}: echo absorbed.") as stub:
        server = start_server(e2e_clone, root, keyless_settings(stub))
        try:
            data_root = pathlib.Path(server.data_root)
            oracle = ArtifactOracle(data_root)

            # The premise: the WHOLE pool is up and every worker has closed its
            # extension registry before the payload exists. The roster is the
            # supervisor's own durable one (state/process_ledger.jsonl, written when
            # the pool is spawned), so "already spawned" is a read fact — a
            # worker starting after the enable would load the skill the ordinary
            # way and prove nothing.
            from tests._shared import live_worker_pids
            roster = wait_until(lambda: live_worker_pids(data_root), timeout=180)
            assert roster, "the supervisor recorded no worker pool"
            assert wait_until(lambda: _worker_boot_pids(oracle) >= roster, timeout=180), (
                f"pool never fully announced ready: {_worker_boot_pids(oracle)} vs {roster}")
            boot_workers = _worker_boot_pids(oracle)

            # INSTALL + REVIEW + ENABLE, all after those workers closed their
            # registries — the same HTTP surface the UI drives.
            payload_dir = data_root / "skills" / "external" / S13H_SKILL
            payload_dir.mkdir(parents=True)
            (payload_dir / "SKILL.md").write_text(S13H_SKILL_MD, encoding="utf-8")
            (payload_dir / "plugin.py").write_text(S13H_PLUGIN, encoding="utf-8")
            review = _api(server.base_url, "POST",
                          f"/api/skills/{S13H_SKILL}/review", {}, timeout=600)
            assert review.get("status") == "clean", review
            toggled = _api(server.base_url, "POST", f"/api/skills/{S13H_SKILL}/toggle",
                           {"enabled": True}, timeout=300)
            assert toggled.get("enabled") is True and not toggled.get("error"), toggled
            entry = _skill_entry(server, S13H_SKILL)
            assert entry.get("live_loaded") is True, entry

            # The durable carrier the workers read: the server published its live
            # set, and the generation is NOT the per-publication ABI-9 digest
            # (that one is minted fresh per publication and could never compare
            # across processes) but the content identity of what is loaded.
            generation = json.loads(
                (data_root / "state" / "extension_generation.json").read_text(encoding="utf-8"))
            assert generation.get("generation"), generation

            # DISPATCH into one of those very workers — no restart anywhere: the owner's
            # Main turn takes the lane's ready spare, which booted before the payload.
            from websockets.sync.client import connect as ws_connect

            client_message_id = f"e2e-s13h-{uuid.uuid4().hex[:12]}"
            with ws_connect(ws_url(server), open_timeout=30, proxy=None) as ws:
                ws.send(json.dumps({"type": "chat", "chat_id": 1, "client_message_id": client_message_id,
                                    "content": "Call the probe echo tool once, then finish."}))
                task_id = wait_until(lambda: _owner_turn_id(oracle, client_message_id), 120)
                assert task_id, "the owner's turn never wrote its task_received row"
                stored = wait_durable_result(oracle, task_id, timeout=300)
            assert stored.get("status") == "completed", stored
            assert S13H_DISPATCH_MARKER in str(stored.get("result") or ""), stored
            assert stub.script_consumed(), "the hot-adoption dispatch script was not consumed"
            rows = [row for row in oracle.task_drive(task_id).tools_rows()
                    if str(row.get("tool") or "") == surface]
            assert rows, "the extension tool call never reached the tools log"
            ran_in = {int(pid) for pid in _S13H_PID_RE.findall(json.dumps(rows))}
            assert len(ran_in) == 1, rows

            # ...and it really was an EXISTING worker: the process that ran the echo
            # booted once, before the payload existed. The lane's replenished spare (a
            # worker_ready after the enable) never ran it.
            (carrier,) = ran_in
            assert carrier in boot_workers, {
                "echo_ran_in": carrier, "booted_before_the_payload": sorted(boot_workers),
                "booted_since": sorted(_worker_boot_pids(oracle) - boot_workers)}
            assert _worker_ready_count(oracle, carrier) == 1, "the carrier re-announced readiness"
            adoptions = [row for row in oracle.events("extension_generation_adopted")
                         if S13H_SKILL in (row.get("skills") or [])]
            assert adoptions, "no worker recorded adopting the published generation"
            assert any(row.get("converged") is True
                       and row.get("published_generation") == generation["generation"] for row in adoptions), (
                adoptions, generation)

            # DISABLE symmetry: the owner's disable republishes at once, so the
            # same channel carries the retraction (a worker adopting it unloads —
            # pinned directly in tests/test_extension_generation_adoption.py).
            untoggled = _api(server.base_url, "POST", f"/api/skills/{S13H_SKILL}/toggle",
                             {"enabled": False}, timeout=300)
            assert not untoggled.get("error"), untoggled
            assert _skill_entry(server, S13H_SKILL).get("live_loaded") is False
            retracted = json.loads(
                (data_root / "state" / "extension_generation.json").read_text(encoding="utf-8"))
            assert retracted.get("generation") != generation.get("generation"), retracted
        finally:
            server.stop()
