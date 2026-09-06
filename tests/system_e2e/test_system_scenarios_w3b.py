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
  capability_delta disclosure), and two TYPED START refusals (a scripted 400 and
  the pinned-profile 409) landing as definite START_FAILED rows that retire
  their invocations — three POSTs, three DISTINCT logical invocation ids.
* S12 — NO-ORPHANS RESTART RECOVERY: the nanny sits in ``delegate_wait`` on a
  hung run; the WHOLE server tree is SIGKILLed (hard crash, no cleanup), a new
  generation boots on the same clone + data root, and its startup custody sweep
  (``_startup_custody_sweep`` → ``reconcile_orphaned_runs``) adopts the
  ownerless run from the DURABLE rows: cancel control delivered, terminal
  verified (CANCEL_OUTCOME outcome=confirmed), settled (SETTLED state=cancelled),
  RECONCILED action=cancelled — with EXACTLY ONE physical ``POST /v2/runs``
  across both generations (recovery is custody adoption, never a second
  attempt) and no process carrying the data root outside the live gen-B tree.
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

import json
import os
import pathlib
import re
import signal
import sys

import pytest

from tests.system_e2e.harness import (
    LANE_MOCK,
    SKILL_REVIEW_MARKER,
    TRIAD_USER_MARKER,
    ArtifactOracle,
    ScriptedStubModel,
    body_text,
    classify_call,
    keyless_settings,
    pids_with_env_value,
    process_tree_pids,
    require_lane,
    scripted_completion,
    start_server,
    submit_running,
    wait_durable_result,
    wait_until,
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
    )

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        daemon.install(tmp_path / "cx")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "cx")) as gateway:
            gateway.handshake()
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
            # Success run: terminal detail carries the settlement facts.
            detail = gateway.get_run(run_id)
            summary = detail.get("summary") or {}
            assert summary.get("state") == "succeeded"
            assert summary.get("model") == daemon.applied_model
            assert summary.get("spendUsd") == 0.0 and summary.get("spendEstimated") is False
            assert (summary.get("authRoute") or {}).get("profileId") == daemon.applied_profile
            assert detail.get("primaryOutput", {}).get("truncated") is False
            # A hung run stays live until cancelled; cancel flips it terminal.
            hung = gateway.start_run({**request, "prompt": FAKE_HANG_MARKER + " go"},
                                     idempotency_key="inv-e2e-4")
            hung_id = str(hung.get("runId") or "")
            assert (gateway.get_run(hung_id).get("summary") or {}).get("state") == "running"
            receipt = gateway.cancel_run(hung_id, reason="e2e probe")
            assert receipt.get("accepted") is True
            assert (gateway.get_run(hung_id).get("summary") or {}).get("state") == "cancelled"


def test_skill_review_classification_and_canned_verdict():
    """The skill-review branch classifies BEFORE every other marker branch (its
    pack can quote them), and the canned all-PASS verdict covers the tree's own
    checklist and aggregates to ``clean`` under the tree's own policy."""
    from ouroboros.skill_review_prompt import _SKILL_REVIEW_ITEMS
    from ouroboros.skill_review_status import aggregate_skill_review_status

    body = {"messages": [{"role": "user", "content":
                          SKILL_REVIEW_MARKER + "\npack quoting: " + TRIAD_USER_MARKER
                          + " [FINALIZE_NOW]"}]}
    assert classify_call(body) == "skill_review"
    kind, message = scripted_completion(body, 1, lambda _b: None, "x")
    assert kind == "skill_review"
    items = json.loads(message["content"])
    assert {row["item"] for row in items} == set(_SKILL_REVIEW_ITEMS)
    assert all(row["verdict"] == "PASS" and row["reason"].strip() for row in items)
    assert aggregate_skill_review_status(items, "extension") == "clean"


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
    "effort": "low",
}
_PINNED_ROW = {
    "subagent_id": "cx-pinned",
    "recommended_use": "Pinned-profile refusal probe for the system_e2e transport scenarios.",
    "route": {"kind": "agent_session", "target_id": "fake-harness=mock-model",
              "credential_profile_id": "ghost-profile"},
    "effort": "low",
}


def _custody_rows(oracle: ArtifactOracle, event_type: str, run_id: str = "") -> list:
    return [row for row in oracle.events(event_type)
            if not run_id or str(row.get("run_id") or "") == run_id]


# ===========================================================================
# S11 — delegated transport: full nanny run + typed refusals
# ===========================================================================

S11_PARENT_MARKER = "S11_PARENT_FINAL_e2e_w3b"

S11_SCRIPT = [
    {"tool": "delegate_start", "arguments": {
        "subagent_id": "cx-scout",
        "prompt": "Survey the repository root and report what you find."}},
    _wait_step,
    {"tool": "delegate_start", "arguments": {
        "subagent_id": "cx-scout",
        "prompt": FAKE_REFUSE_MARKER + " this start must be refused typed"}},
    {"tool": "delegate_start", "arguments": {
        "subagent_id": "cx-pinned",
        "prompt": "the pinned-profile start must be refused typed"}},
]


@pytest.mark.integration
@pytest.mark.serial
def test_s11_delegated_transport_wire_custody_and_typed_refusals(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s11")
    data_root = pathlib.Path(root) / "data"
    with FakeClaudexorDaemon() as daemon, \
            ScriptedStubModel(S11_SCRIPT,
                              final_answer=f"{S11_PARENT_MARKER}: delegated run absorbed.") as stub:
        daemon.install(data_root / "claudexor")
        settings = keyless_settings(stub, OUROBOROS_SUBAGENTS=_roster(_SCOUT_ROW, _PINNED_ROW))
        server = start_server(e2e_clone, root, settings)
        try:
            task_id = submit_running(
                server, "Delegate the survey to your configured scout, wait it out, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)
            assert S11_PARENT_MARKER in str(stored.get("result") or ""), stored
            assert stub.script_consumed(), "S11 script was not fully consumed"

            # -- durable custody chain (canonical events.jsonl) --------------
            requested = _custody_rows(oracle, "delegate_run_start_requested")
            started = _custody_rows(oracle, "delegate_run_started")
            assert len(requested) == 3, requested   # every POST had its pre-wire row
            assert len(started) == 1, started
            entry = started[0]
            run_id = str(entry.get("run_id") or "")
            invocation_id = str(entry.get("invocation_id") or "")
            assert re.fullmatch(r"[0-9a-f]{16}", run_id), entry
            assert invocation_id, entry
            assert entry.get("task_id") == task_id, entry
            assert entry.get("route") == "fake-harness", entry
            assert entry.get("model") == "mock-model", entry
            assert entry.get("access") == "readonly" and entry.get("mode") == "ask", entry
            assert entry.get("selected_subagent_id") == "cx-scout", entry
            assert _custody_rows(oracle, "delegate_run_ledger_recorded", run_id)
            settled = _custody_rows(oracle, "delegate_run_settled", run_id)
            assert settled, "no durable settlement row"
            assert settled[-1].get("state") == "succeeded", settled[-1]
            assert settled[-1].get("model") == daemon.applied_model, settled[-1]
            assert settled[-1].get("cost_usd") == 0.0, settled[-1]
            assert settled[-1].get("cost_final") is True, settled[-1]
            assert settled[-1].get("credential_profile_id") == daemon.applied_profile, settled[-1]

            # Typed refusals: durable, DEFINITE, invocation-retiring.
            failed = _custody_rows(oracle, "delegate_run_start_failed")
            reasons = {str(row.get("reason") or "") for row in failed}
            assert "fake_route_refused" in reasons, failed
            assert "credential_profile_unknown" in reasons, failed
            assert all(row.get("definite") is True for row in failed), failed

            # -- wire truth (the daemon's own recording) ---------------------
            posts = daemon.run_start_posts()
            assert len(posts) == 3, [p["body"].get("prompt") for p in posts]
            keys = [p["idempotency_key"] for p in posts]
            assert len(set(keys)) == 3, keys        # fresh logical invocation per intended start
            assert keys[0] == invocation_id, (keys, invocation_id)
            body = posts[0]["body"]
            assert body.get("authPreference") == "subscription", body
            assert body.get("mode") == "ask" and body.get("access") == "readonly", body
            assert body.get("harnesses") == [daemon.harness_id], body
            assert body.get("primaryHarness") == daemon.harness_id, body
            assert body.get("model") == "mock-model", body
            assert "execution" not in body, body    # readonly run carries no isolation block
            assert int(body.get("maxSeconds") or 0) > 0, body
            assert str(body.get("instructions") or "").strip(), body
            assert (body.get("scope") or {}).get("kind") == "project", body
            project_posts = daemon.calls("POST", "/v2/projects")
            assert project_posts and all(p["idempotency_key"] for p in project_posts)
            assert daemon.calls("DELETE", "/v2/projects/"), (
                "the settled readonly run's owned registration was never retired")

            # -- «last delegated run» receipt: requested vs applied, honest --
            receipt = oracle._json("state/subagent_last_delegation.json")
            assert receipt.get("run_id") == run_id, receipt
            assert receipt.get("route") == "fake-harness", receipt
            assert receipt.get("requested_model") == "mock-model", receipt
            assert receipt.get("applied_model") == daemon.applied_model, receipt
            assert receipt.get("requested_profile") == "", receipt
            assert receipt.get("applied_profile") == daemon.applied_profile, receipt
            assert receipt.get("selected_subagent_id") == "cx-scout", receipt

            # -- the terminal facts REACHED the model (transcript truth) -----
            transcript = "\n".join(body_text(call_body) for _kind, call_body in stub.calls)
            assert f"FAKE_RUN_RESULT {run_id}" in transcript
            assert '"cost_final": true' in transcript
            assert "session_route_resolves_its_own_model" in transcript
            assert "fake_route_refused" in transcript
        finally:
            server.stop()


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
@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "linux",
                    reason="process-group SIGKILL + /proc scans are Linux-only")
def test_s12_sigkill_mid_run_restart_recovers_custody_without_orphans(
        e2e_clone, tmp_path_factory):
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
                submit_running(server, "Delegate a long survey and babysit it until stopped.")
                # The physical run is LIVE: durable STARTED row, and the nanny is
                # actually polling the daemon (wire GETs observed).
                assert wait_until(lambda: _custody_rows(oracle, "delegate_run_started"), 180), (
                    "delegated run never reached a durable STARTED row")
                run_id = str(_custody_rows(oracle, "delegate_run_started")[-1]["run_id"])
                assert wait_until(lambda: daemon.calls("GET", f"/v2/runs/{run_id}"), 60)
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

        # Generation B: same clone + data root. The startup custody sweep must
        # adopt the ownerless run from the DURABLE rows alone.
        with ScriptedStubModel([]) as stub_b:
            server_b = start_server(e2e_clone, root, keyless_settings(
                stub_b, OUROBOROS_SUBAGENTS=roster))
            try:
                oracle = ArtifactOracle(server_b.data_root)
                assert wait_until(
                    lambda: _custody_rows(oracle, "delegate_run_cancel_outcome", run_id), 240), (
                    "boot custody sweep never cancelled the ownerless run")
                cancel_rows = _custody_rows(oracle, "delegate_run_cancel_outcome", run_id)
                assert cancel_rows[-1].get("outcome") == "confirmed", cancel_rows
                settled = wait_until(
                    lambda: _custody_rows(oracle, "delegate_run_settled", run_id) or None, 60)
                assert settled and settled[-1].get("state") == "cancelled", settled
                reconciled = _custody_rows(oracle, "delegate_run_reconciled", run_id)
                assert reconciled and reconciled[-1].get("action") == "cancelled", reconciled

                # Wire truth: the cancel control landed, and there was NEVER a
                # second physical attempt — recovery is custody adoption.
                assert daemon.calls("POST", f"/v2/runs/{run_id}/control")
                assert len(daemon.run_start_posts()) == 1, (
                    "restart recovery re-POSTed a run that already had a STARTED row")

                # No orphans: every live pid carrying this data root sits inside
                # the gen-B server tree.
                assert wait_until(
                    lambda: set(pids_with_env_value(str(server_b.data_root)))
                    <= set(process_tree_pids(server_b.proc.pid)), 45), (
                    "a process carrying the data root lives outside the gen-B tree")
            finally:
                server_b.stop()


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
@pytest.mark.serial
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
            # Durable verdict semantics on this tree: a review WITH real
            # PASS/FAIL verdicts persists NO status key (status re-derives from
            # the findings on every load), so the durable pin is the findings
            # plane itself: all-PASS from the triad panel.
            review_state = oracle._json(f"state/skills/{S13_SKILL}/review.json")
            findings = review_state.get("findings") or []
            assert findings and all(
                str(f.get("verdict") or "") == "PASS" for f in findings), review_state
            assert "status" not in review_state, review_state
            assert stub_a.kinds().count("skill_review") >= 3, stub_a.kinds()

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


def _worker_boot_pids(oracle: ArtifactOracle) -> set:
    return {int(row.get("pid") or 0) for row in oracle.events("worker_ready")}


@pytest.mark.integration
@pytest.mark.serial
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
            # supervisor's own durable one (state/worker_pids.json, written when
            # the pool is spawned), so "already spawned" is a read fact — a
            # worker starting after the enable would load the skill the ordinary
            # way and prove nothing.
            roster = wait_until(
                lambda: {int(row.get("pid") or 0)
                         for row in (oracle._json("state/worker_pids.json").get("workers") or [])},
                timeout=180)
            assert roster, "the supervisor recorded no worker pool"
            assert wait_until(lambda: _worker_boot_pids(oracle) >= roster, timeout=180), (
                f"pool never fully announced ready: {_worker_boot_pids(oracle)} vs {roster}")
            boot_workers = _worker_boot_pids(oracle)

            # INSTALL + REVIEW + ENABLE, all after those workers closed their
            # registries — the same HTTP surface the UI drives.
            payload_dir = data_root / "skills" / "external" / S13H_SKILL
            payload_dir.mkdir(parents=True)
            (payload_dir / "SKILL.md").write_text(S13H_SKILL_MD, encoding="utf-8")
            (payload_dir / "plugin.py").write_text(S13_PLUGIN, encoding="utf-8")
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

            # DISPATCH into one of those very workers — no restart anywhere.
            task_id = submit_running(server, "Call the probe echo tool once, then finish.")
            result = server.wait_task(task_id, timeout=300)
            assert result.get("status") == "completed", result
            stored = wait_durable_result(oracle, task_id)
            assert S13H_DISPATCH_MARKER in str(stored.get("result") or ""), stored
            assert stub.script_consumed(), "the hot-adoption dispatch script was not consumed"
            rows = [row for row in oracle.task_drive(task_id).tools_rows()
                    if str(row.get("tool") or "") == surface]
            assert rows, "the extension tool call never reached the tools log"
            assert "echo: hot-ping" in json.dumps(rows), rows

            # ...and it really was an EXISTING worker: no process was replaced
            # between the enable and the dispatch.
            assert _worker_boot_pids(oracle) == boot_workers, (
                "the pool respawned, so this proves nothing about hot adoption")
            adoptions = [row for row in oracle.events("extension_generation_adopted")
                         if S13H_SKILL in (row.get("skills") or [])]
            assert adoptions, "no worker recorded adopting the published generation"
            assert any(row.get("converged") is True for row in adoptions), adoptions

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
