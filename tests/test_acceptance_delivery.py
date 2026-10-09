"""Task acceptance on the review pool (owner decisions R0/R2/R3, 2026-09-01,
Ф2 of the agentic-review sprint; PR-3: the pool replaces the triad lane).

ONE builder — ``reviewer_slot_config.review_pool_slots`` — turns the catalog's
review-eligible rows into ``ReviewSlot`` objects for plan review, skill/commit
review (as the aligned vectors of ``commit_triad_delivery``) and task acceptance,
so acceptance carries every row's own delivery, effort, credential pin and
stable id (the catalog id) instead of an api-pinned projection. A malformed
catalog refuses acceptance typed (DEGRADED) exactly as it refuses plan and skill
review; a comma list without a catalog is an EMPTY pool and a typed refusal, not
a default panel; child-task and ``off``-mode acceptance follow the pool rows'
delivery (a child with at most one row, #1334).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.review_execution import ReviewRouteKind
from ouroboros.reviewer_slot_config import review_pool_slots


def _row(subagent_id, target, *, kind="api_model", review_eligible=True, **extra):
    route = {"kind": kind, "target_id": target}
    if kind == "agent_session":
        route["credential_profile_id"] = extra.pop("profile_id", "")
    row = {"subagent_id": subagent_id, "recommended_use": f"{subagent_id} reviewer.", "route": route, **extra}
    if review_eligible:
        row["review_eligible"] = True
    return row


def _roster(*rows):
    return {"enabled": True, "items": list(rows)}


# The pool: a packet api row, a pinned agent-session row and a native api row.
_POOL_ROWS = (
    _row("t_api", "openai/gpt-5.6-luna", effort="high", delivery="packet"),
    _row("t_sess", "codex=gpt-5.6-sol", kind="agent_session", profile_id="acct-1", effort="xhigh"),
    _row("t_actor", "openai/gpt-5.6-terra", effort="medium"),
)
_ROSTER = _roster(*_POOL_ROWS)


@pytest.fixture(autouse=True)
def _provider_catalog_stays_off_the_wire(provider_catalog_offline):
    """The work orders here measure the native row's window; see `provider_catalog_offline`."""


@pytest.fixture()
def structured_env(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(_ROSTER))
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_REVIEW_ROUTES", "OUROBOROS_REVIEW_SESSION_ROUTE"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _acceptance_ctx(tmp_path, *, evidence=None, task_metadata=None, content="deliverable",
                    fresh_result=True, max_improvement_passes=0, **tool_ctx_fields):
    """A root acceptance context whose wallet claim can be exercised for real."""
    from ouroboros import loop as loop_mod
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.review_substrate import build_review_binding
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    contract = build_task_contract({"budget_profile": {"max_improvement_passes": max_improvement_passes}})
    metadata = {
        "root_task_id": "root-delivery", "delegation_role": "root",
        "budget_drive_root": str(tmp_path), "task_contract": contract,
        **(task_metadata or {}),
    }
    tool_ctx = SimpleNamespace(
        task_id="root-delivery", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_contract=contract, task_metadata=metadata, pending_events=[],
        **tool_ctx_fields,
    )
    if fresh_result:
        write_task_result(
            tmp_path, "root-delivery", STATUS_RUNNING, root_task_id="root-delivery",
            delegation_role="root", task_contract=contract,
        )
    evidence = evidence if evidence is not None else {"evidence": "complete"}
    return loop_mod._TaskAcceptanceContext(
        tools=SimpleNamespace(_ctx=tool_ctx), content=content, task_id="root-delivery",
        task_type="task", llm_trace={"tool_calls": []}, drive_root=tmp_path,
        messages=[{"role": "system", "content": "policy"}, {"role": "user", "content": "goal"}],
        emit_progress=lambda _text, *, incident=None: None, mode="required", subtree_statuses=[],
        budget_profile=contract["budget_profile"], passes_done=0, evidence=evidence,
        review_binding=build_review_binding(
            candidate=content, evidence=evidence, fence_token_or_state="delivery-test",
        ),
    )


def _capture_panel(monkeypatch):
    """Stub the substrate call and the wave gate; return the captured (request, kwargs)."""
    import ouroboros.review_substrate as rs
    from ouroboros.tools import review_helpers

    captured = []

    def _run(request, **kwargs):
        captured.append((request, kwargs))
        return SimpleNamespace(aggregate_signal="PASS", actors=[])

    monkeypatch.setattr(rs, "run_review_request", _run)
    monkeypatch.setattr(review_helpers, "review_wave_budget_gate", lambda *_a, **_k: None)
    return captured


# ---------------------------------------------------------------------------
# One builder for plan review, skill/commit vectors and task acceptance.
# ---------------------------------------------------------------------------


def test_review_pool_slots_is_the_one_builder_shared_by_plan_and_commit_vectors(structured_env):
    from ouroboros.reviewer_slot_config import commit_triad_delivery, triad_delivery_slots
    from ouroboros.tools.plan_review_runtime import (
        PLAN_REVIEW_MAX_TOKENS,
        plan_review_slots,
    )

    acceptance = review_pool_slots(role_hint="task acceptance")
    plan = plan_review_slots()
    identity = lambda s: (s.slot_id, s.model, s.route, s.session_target, s.session_profile, s.subagent_id)  # noqa: E731
    assert [identity(s) for s in plan] == [identity(s) for s in acceptance]
    assert [identity(s) for s in triad_delivery_slots(role_hint="task acceptance")] == [identity(s) for s in acceptance]
    assert [s.slot_id for s in acceptance] == ["t_api", "t_sess", "t_actor"]  # the catalog ids
    # Plan review keeps its own slot properties on the shared rows.
    assert all(s.role_hint == "plan reviewer" and s.max_tokens == PLAN_REVIEW_MAX_TOKENS for s in plan)
    assert all(s.role_hint == "task acceptance" for s in acceptance)
    # Effort: each row's own value, on every pool surface alike.
    assert [s.effort for s in plan] == ["high", "xhigh", "medium"]
    # The commit/skill vectors are a projection of the same slots.
    vectors = commit_triad_delivery()
    assert vectors["slot_ids"] == [s.slot_id for s in acceptance]
    assert vectors["models"] == [s.model for s in acceptance]
    assert vectors["routes"] == [s.route for s in acceptance]
    assert vectors["session_profiles"] == ["", "acct-1", ""]
    assert vectors["subagent_ids"] == ["t_api", "t_sess", "t_actor"]
    assert vectors["retrieves"] == [False, True, True]
    assert vectors["legacy_skill_fingerprint"] is False


def test_acceptance_panel_carries_each_rows_identity_effort_pin_and_binding(structured_env, tmp_path):
    from ouroboros import loop as loop_mod
    from ouroboros.config import adaptive_quorum

    captured = _capture_panel(structured_env)
    result = loop_mod._execute_task_acceptance_panel(_acceptance_ctx(tmp_path))
    assert result.aggregate_signal == "PASS"
    (request, kwargs), = captured
    slots = kwargs["slots"]
    assert [s.slot_id for s in slots] == ["t_api", "t_sess", "t_actor"]  # catalog ids, not slot_N
    assert [s.route for s in slots] == [
        ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION, ReviewRouteKind.API_CHAT,
    ]
    assert [s.effort for s in slots] == ["high", "xhigh", "medium"]  # per-row, not one global effort
    assert slots[1].session_target == "codex=gpt-5.6-sol" and slots[1].session_profile == "acct-1"
    assert slots[2].subagent_id == "t_actor" and slots[2].native_retrieval and not slots[0].native_retrieval
    assert request.policy["min_successful_slots"] == adaptive_quorum(3)


def test_malformed_catalog_refuses_acceptance_typed(structured_env, tmp_path):
    """R3: the same typed refusal plan and skill review give — never the silently
    projected default panel the retired residual used to run."""
    import ouroboros.review_substrate as rs
    from ouroboros import loop as loop_mod

    structured_env.setenv("OUROBOROS_SUBAGENTS", "{broken")
    structured_env.setattr(
        rs, "run_review_request",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no reviewer may be called")),
    )
    result = loop_mod._execute_task_acceptance_panel(_acceptance_ctx(tmp_path))
    assert result.aggregate_signal == "DEGRADED" and result.degraded
    assert any(
        r.startswith("reviewer_slot_config_invalid:") and "no reviewer was called" in r
        for r in result.degraded_reasons
    )
    assert result.actors == []


def test_a_comma_list_without_a_catalog_is_an_empty_pool_and_a_typed_refusal(monkeypatch, tmp_path):
    """PR-3: the lane-era comma list projects NO panel any more. With no catalog
    (or no marked row) the pool is empty and acceptance refuses typed — nobody
    is called, no default rows are invented. The benchmark classes that used to
    ride the comma list carry explicit catalog rows instead."""
    from ouroboros import loop as loop_mod

    for key in ("OUROBOROS_REVIEW_ROUTES", "OUROBOROS_SUBAGENTS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "openai/a,openai/b,openai/c")
    captured = _capture_panel(monkeypatch)
    result = loop_mod._execute_task_acceptance_panel(_acceptance_ctx(tmp_path))
    assert captured == []
    assert result.aggregate_signal == "DEGRADED" and result.degraded_reasons == ["no_review_slots"]


def test_child_and_off_acceptance_follow_the_configured_rows(structured_env, tmp_path):
    """Child-task and `off`-mode acceptance is advisory evidence (#1334): every
    row keeps its configured delivery. An off-mode root's explicit call runs the
    configured panel's breadth; a child reviews with at most ONE row, named when
    several are configured and checked for membership by the host. A refusal is
    typed and calls nobody."""
    import ouroboros.review_substrate as rs
    from ouroboros import review_evidence as re_mod
    from ouroboros.tools.review import _handle_task_acceptance_review

    calls = []
    structured_env.setattr(re_mod, "collect_turn_diff", lambda ctx, **kwargs: "")
    structured_env.setattr(rs, "build_improvement_capsule", lambda _result: "")
    structured_env.setattr(rs, "dissent_findings", lambda _result: [])

    def fake_run(request, **kwargs):
        calls.append(([s.slot_id for s in kwargs["slots"]], dict(request.slot_session_tasks)))
        return SimpleNamespace(aggregate_signal="PASS", actors=[], parsed_findings=[])

    structured_env.setattr(rs, "run_review_request", fake_run)
    structured_env.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    root = SimpleNamespace(
        drive_root=str(tmp_path), task_id="root", root_task_id="root",
        task_metadata={"root_task_id": "root"}, task_contract={},
    )
    # Off-mode root: the whole configured panel, retrieving rows with their work order.
    json.loads(_handle_task_acceptance_review(root, claim="root done"))
    (slot_ids, orders), = calls
    assert slot_ids == ["t_api", "t_sess", "t_actor"] and set(orders) == {"t_sess", "t_actor"}

    child = SimpleNamespace(
        drive_root=str(tmp_path), task_id="child", root_task_id="root",
        task_metadata={"root_task_id": "root", "parent_task_id": "root"}, task_contract={},
    )
    payload = json.loads(_handle_task_acceptance_review(child, claim="child done"))
    assert payload["status"] == "not_dispatched" and payload["reason"] == "reviewer_selection_required"
    assert [row["slot_id"] for row in payload["reviewer_rows"]] == ["t_api", "t_sess", "t_actor"]
    payload = json.loads(_handle_task_acceptance_review(child, claim="child done", reviewer_slot_id="nope"))
    assert payload["reason"] == "reviewer_slot_unknown" and len(calls) == 1
    _handle_task_acceptance_review(child, claim="child done", reviewer_slot_id="t_actor")
    assert calls[-1][0] == ["t_actor"] and set(calls[-1][1]) == {"t_actor"}

    # A single pool row needs no name.
    structured_env.setenv("OUROBOROS_SUBAGENTS", json.dumps(_roster(_POOL_ROWS[1])))
    _handle_task_acceptance_review(child, claim="child done")
    assert calls[-1][0] == ["t_sess"]

    # Malformed catalog: the same typed refusal, never a default panel.
    structured_env.setenv("OUROBOROS_SUBAGENTS", "{broken")
    payload = json.loads(_handle_task_acceptance_review(root, claim="root done"))
    assert payload["status"] == "not_dispatched" and "invalid review pool configuration" in payload["error"]
    assert len(calls) == 3


def _consume_acceptance_results(results):
    from ouroboros.loop_tool_execution import process_tool_results

    trace = {"tool_calls": [], "reasoning_notes": []}
    process_tool_results([{
        "fn_name": "task_acceptance_review", "tool_call_id": f"call-{index}", "result": result,
        "is_error": False, "args_for_log": {}, "tool_args": {}, "result_meta": {"status": "ok"},
    } for index, result in enumerate(results)], [], trace, emit_progress=lambda _msg, *, incident=None: None)
    return trace


def test_a_typed_predispatch_refusal_is_tool_evidence_not_a_degraded_review_run(structured_env, tmp_path):
    """#1318: the child-selection refusal the REAL tool returns reaches the ordinary
    tool-result consumer; no reviewer ran, so it is recorded as this call's tool
    result and never as a review run that alone degrades the review axis. A
    dispatched run -- malformed, overflowed or degraded -- still counts."""
    from ouroboros.outcomes import _objective_axis, _review_axis
    from ouroboros.tools.review import _handle_task_acceptance_review

    structured_env.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    structured_env.setenv("OUROBOROS_SUBAGENTS", json.dumps(_roster(*_POOL_ROWS[1:])))
    # #1334 now permits the off-mode ROOT's full configured panel. A child
    # facing multiple rows still needs a selection and is genuinely zero-run.
    ctx = SimpleNamespace(drive_root=str(tmp_path), task_id="child", root_task_id="root",
                          task_metadata={"root_task_id": "root", "parent_task_id": "root"}, task_contract={})
    refused = _handle_task_acceptance_review(ctx, claim="child done")
    assert json.loads(refused)["reason"] == "reviewer_selection_required"
    structured_env.setenv("OUROBOROS_SUBAGENTS", "{broken")
    misconfigured = _handle_task_acceptance_review(ctx, claim="root done")
    trace = _consume_acceptance_results([refused, misconfigured])
    assert "review_runs" not in trace
    assert [row["result"] for row in trace["tool_calls"]] == [refused, misconfigured]
    review = _review_axis(trace)
    assert review["status"] == "skipped" and review["run_count"] == 0
    assert _objective_axis(review)["status"] != "degraded"

    dispatched_malformed = {"request": {"surface": "task_acceptance"}, "actors": [{"status": "error"}],
                            "parsed_findings": [], "aggregate_signal": ""}
    overflowed = {"request": {"surface": "task_acceptance"}, "actors": [], "parsed_findings": [],
                  "aggregate_signal": "DEGRADED", "degraded": True,
                  "degraded_reasons": ["__immutable_core_overflow__"]}
    # A status field alone is not a refusal when run evidence rides beside it.
    evidenced = {"status": "not_dispatched", "aggregate_signal": "DEGRADED", "actors": [{"status": "not_dispatched"}]}
    trace = _consume_acceptance_results([json.dumps(dispatched_malformed), json.dumps(overflowed), json.dumps(evidenced)])
    assert len(trace["review_runs"]) == 3
    assert _review_axis(trace)["status"] == "degraded"


# ---------------------------------------------------------------------------
# The retrieving work order (R1/R4/R5/R15/R23) and the route-aware gates.
# ---------------------------------------------------------------------------

_ACCEPTANCE_PACKET = {
    "task_contract": {"objective": "ship it", "acceptance_claims": [{"id": "claim_1", "claim": "game boots"}]},
    "acceptance_support_refs": [{
        "criterion_id": "claim_1", "support_status": "supported",
        "support_refs": [{"ref": "verification_receipts[0]", "status": "pass"}],
    }],
    "verification_summary": {"count": 1, "failed_count": 0},
    "verification_receipts": [{
        "ref": "verification_receipts[0]", "status": "pass", "matched": True,
        "provenance": "host_attested", "criterion_id": "claim_1", "check": "pytest -q",
    }],
    "acceptance_obligations": [],
    "artifacts": [{"name": "report/summary.md", "size": 29, "preview": "PREVIEW-BYTES-OF-THE-ARTIFACT"}],
    "repo_diff": "diff --git a/x b/x",
    "tool_trajectory": [{"tool": "run_command", "status": "ok", "result": "TRAJECTORY-RESULT-3-passed"}],
    "reasoning_notes": "I believe the feature works.",
    "__provenance__": {
        "task_contract": "host_attested", "acceptance_support_refs": "host_attested",
        "verification_summary": "host_attested", "verification_receipts": "host_attested",
        "acceptance_obligations": "host_attested", "repo_diff": "host_attested",
        "artifacts": "artifact", "tool_trajectory": "tool_result", "reasoning_notes": "agent_supplied",
    },
}

_CLEAN_VERDICT = {
    "verdict": "PASS", "outcome_tier": "solved", "completion_coach": "",
    "criteria_used": [
        {"criterion": "game boots", "status": "supported", "evidence_refs": ["verification_receipts[0]"]},
        {"criterion": "receipt beside a fabricated ref", "status": "supported",
         "evidence_refs": ["verification_receipts[0]", "made_up_ref"]},
    ],
    "dialogue_status": "unreachable_here",
    "findings": [], "summary": "verified against the receipts",
}

_ROW_API = _row("t_api", "openai/fake-reviewer", delivery="packet")
_ROW_NATIVE = _row("t_actor", "openai/fake-reviewer")
_ROW_SESSION = _row("t_sess", "fake-review=fake-small", kind="agent_session")


def _offline_env(monkeypatch, *rows):
    """An offline review pool (fake model ids never reach a provider)."""
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(_roster(*rows)))
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_REVIEW_ROUTES", "OUROBOROS_REVIEW_SESSION_ROUTE"):
        monkeypatch.delenv(key, raising=False)


class _EpisodeLLM:
    """Scripted `chat()` that crosses the real durable attempt ledger on every
    send — the production `LLMClient` does, and both the packet row and the
    native episode bind the acceptance stamp around that crossing — so the
    wallet claim fires exactly where production fires it."""

    def __init__(self, drive_root, script, native_script=None, scoped=False, reservation_usd=0.0):
        self.drive_root = drive_root
        self.script = list(script)
        self.native_script = None if native_script is None else list(native_script)
        self.scoped = scoped  # True: the bound usage scope owns task/root ids and the root limit
        self.reservation_usd = float(reservation_usd)  # a PRICED send: reserved against the root wallet, then settled
        self.calls = []

    def _reply(self, kwargs):
        self.calls.append(dict(kwargs))
        script = self.native_script if ("tools" in kwargs and self.native_script is not None) else self.script
        if not script:
            raise AssertionError("script exhausted — an extra reviewer send was made")
        return script.pop(0), {"prompt_tokens": 10, "completion_tokens": 5, "cost": self.reservation_usd}

    def chat(self, **kwargs):
        from ouroboros import usage_accounting as ua

        ids = {} if self.scoped else {"task_id": "review", "root_task_id": "review"}
        # A PRICED send runs under the priced identity (the seeded catalog row), so
        # reservation and settlement follow the real wallet path; an unpriced send
        # keeps the free local identity the older ledger-shape pins expect.
        identity = ({"model": "openai/fake-reviewer", "provider": "openrouter"} if self.reservation_usd > 0
                    else {"model": "local-review-test", "provider": "local"})
        request = ua.AttemptRequest(
            reservation_usd=self.reservation_usd, drive_root=self.drive_root, **identity, **ids,
        )
        return ua.execute_physical_attempt(request, lambda: self._reply(kwargs))


def _real_panel(monkeypatch, llm, *, stub_gate=True):
    """Run the REAL substrate under the panel with a scripted LLM; capture requests.
    ``stub_gate=False`` leaves the real wave budget gate in place (it needs a
    bound usage scope and a priced model to decide anything)."""
    import ouroboros.review_substrate as rs
    from ouroboros.tools import review_helpers

    seen = []
    original = rs.run_review_request

    def _run(request, **kwargs):
        seen.append(request)
        return original(request, llm=llm, **kwargs)

    monkeypatch.setattr(rs, "run_review_request", _run)
    if stub_gate:
        monkeypatch.setattr(review_helpers, "review_wave_budget_gate", lambda *_a, **_k: None)
    return seen


def _priced_offline_model(monkeypatch):
    """Seed the PRICE SOURCE (the provider catalog cache, marked fresh) so the
    real gate can price `openai/fake-reviewer` at $1/M in, $1/M out — ≈$0.07 per
    send with the 65 536-token completion reserve. No gate code is patched."""
    import time

    from ouroboros import pricing

    monkeypatch.setitem(pricing._cached_pricing, "openrouter", {"openai/fake-reviewer": (1.0, None, None, 1.0)})
    monkeypatch.setitem(pricing._pricing_fetched_at, "openrouter", time.time())


def _root_scope(tmp_path, *, root_limit_usd):
    from ouroboros import usage_accounting as ua

    return ua.UsageScope(drive_root=tmp_path, task_id="root-delivery", root_task_id="root-delivery",
                         root_limit_usd=root_limit_usd)


def _seed_root_ledger(scope, *, cost=0.0):
    """`usage_projection(root_task_id)` derives the root limit from EXISTING
    ledger rows; before the first send there is none and the wave gate fails
    open. One scoped physical attempt makes the wallet real for the gate."""
    from ouroboros import usage_accounting as ua

    request = ua.AttemptRequest(model="openai/fake-reviewer", provider="openrouter", reservation_usd=cost)
    with ua.usage_scope(scope):
        ua.execute_physical_attempt(
            request, lambda: ({"content": "seed"}, {"prompt_tokens": 1, "completion_tokens": 1, "cost": cost}))


def _spy_admission(monkeypatch):
    """Record every real `review_wave_admission` call (models and result) and
    call through — the gate is observed, never replaced."""
    from ouroboros import usage_admission as ua

    calls = []
    original = ua.review_wave_admission

    def _spy(drive_root, *, models, **kwargs):
        result = original(drive_root, models=models, **kwargs)
        calls.append({"models": list(models), **result})
        return result

    monkeypatch.setattr(ua, "review_wave_admission", _spy)
    return calls


def _roots(tmp_path):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.outcome_receipt_store import append_verification_receipt
    from ouroboros.utils import append_jsonl

    governance, workspace = tmp_path / "governance", tmp_path / "workspace"
    governance.mkdir(exist_ok=True)
    workspace.mkdir(exist_ok=True)
    (workspace / "greeting.txt").write_text("hello native reviewer\n", encoding="utf-8")
    # The packet names actual producer sources, retained by the real coordinator
    # before our offline native/session executor is allowed to dispatch.
    artifact = task_artifact_dir_path(tmp_path, 'root-delivery', create=True) / 'report/summary.md'
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(_ACCEPTANCE_PACKET['artifacts'][0]['preview'], encoding='utf-8')
    for receipt in _ACCEPTANCE_PACKET['verification_receipts']:
        assert append_verification_receipt(tmp_path, 'root-delivery', receipt)
    for row in _ACCEPTANCE_PACKET['tool_trajectory']:
        append_jsonl(tmp_path / 'logs/tools.jsonl', {'task_id': 'root-delivery', **row})
    return governance, workspace


def _tool_call(name, args, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _fake_session(monkeypatch):
    """The offline Claudexor /v2 surface answering the acceptance object verdict."""
    from tests.test_review_agent_session_route import FakeGateway, _terminal_detail
    from ouroboros import claudexor_daemon
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways import claudexor as gateway_module

    FakeGateway.reset()
    FakeGateway.detail = _terminal_detail(json.dumps(_CLEAN_VERDICT), conformance="passed")
    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", FakeGateway)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: gateway_module.ClaudexorGateway())
    custody._CUSTODY.clear()
    return FakeGateway


def test_trap_retrieving_row_receipt_ref_resolves_against_the_full_packet(monkeypatch, tmp_path):
    """THE trap (brief §6.2 item 6): a retrieving row that cites a real receipt
    ref is resolved CLEAN — against the FULL packet the host built, never against
    the tail-less projection it was sent."""
    from ouroboros import loop as loop_mod
    from ouroboros.review_substrate import task_acceptance_is_clean

    _offline_env(monkeypatch, _ROW_NATIVE)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}])
    seen = _real_panel(monkeypatch, llm)
    governance, workspace = _roots(tmp_path)
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    result = loop_mod._execute_task_acceptance_panel(ctx)

    (actor,) = result.actors
    assert actor["status"] == "ok" and result.aggregate_signal == "PASS"
    assert task_acceptance_is_clean(result) is True
    (row,) = actor["criteria_refs_unresolved"]  # only the criterion carrying the fabricated ref
    assert row["supported_evidence_resolves"] is True
    assert row["refs"][0]["ref"] == "verification_receipts[0]" and row["refs"][0]["resolved_as"]
    assert row["refs"][1] == {"ref": "made_up_ref", "resolved_as": ""}
    assert actor["usage"]["host_file_read_attestation"] == "host_observed"

    (request,) = seen
    order = request.slot_session_tasks["t_actor"]
    assert "TRAJECTORY-RESULT-3-passed" not in order and "PREVIEW-BYTES" not in order  # tail withheld
    assert "verification_receipts[0]" in order and "RETRIEVAL POINTERS" in order and str(tmp_path) in order
    assert request.evidence["tool_trajectory"][0]["result"] == "TRAJECTORY-RESULT-3-passed"  # FULL dict intact
    closure = request.policy['review_source_closure']
    reader = Path(closure['read_root'])
    assert request.policy['native_data_root'] == str(reader)
    assert reader.is_relative_to(tmp_path / 'task_results/artifacts/root-delivery/source_handles/review_inputs')
    retained = {row['name']: Path(row['retained_path']).read_text()
                for row in closure['sources'] if row['status'] == 'retained'}
    assert json.loads(retained['task-result'])['task_id'] == 'root-delivery'
    assert retained['artifact:report/summary.md'] == _ACCEPTANCE_PACKET['artifacts'][0]['preview']
    assert json.loads(retained['verification-receipts']) == _ACCEPTANCE_PACKET['verification_receipts'][0]
    assert json.loads(retained['tool-trajectory'])[0]['result'] == 'TRAJECTORY-RESULT-3-passed'
    assert str(reader) in order
    assert request.session_root == str(workspace)
    sent = json.dumps(llm.calls[0]["messages"])
    assert "RETRIEVAL POINTERS" in sent and "TRAJECTORY-RESULT-3-passed" not in sent


def test_acceptance_request_carries_the_route_owned_work_order(structured_env, tmp_path):
    from ouroboros import loop as loop_mod
    from ouroboros.review_execution import (
        ReviewAssignment,
        _render_prompt_parts,
        _review_route_executor,
        review_output_contract,
    )
    from ouroboros.triad_review import ACCEPTANCE_SURFACE_RULES, TIER_CLASSIFICATION_RULES

    captured = _capture_panel(structured_env)
    governance, workspace = _roots(tmp_path)
    deadline = "2030-01-01T00:00:00+00:00"
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), task_metadata={"deadline_at": deadline},
                          repo_dir=str(governance), workspace_root=str(workspace), workspace_mode="project")
    loop_mod._execute_task_acceptance_panel(ctx)
    (request, kwargs), = captured
    api, sess, actor = kwargs["slots"]
    # R23, R5 and the carried Ф1 finding: owner deadline, real data root, THE acceptance contract.
    assert request.deadline_at == deadline
    assert request.session_root == str(workspace)  # the ACTIVE workspace, not the governance repo
    assert request.policy["native_data_root"] == str(tmp_path)
    contract = request.policy["output_contract"]
    assert contract == review_output_contract(request)
    assert ACCEPTANCE_SURFACE_RULES in contract and "criteria_used" in contract and "dialogue_status" in contract
    for slot in (sess, actor):
        executor = _review_route_executor(ReviewAssignment(request=request, slot=slot, call_id="c"))
        assert executor._output_contract() == contract
    # Session row: the FULL packet, absolute pointers and the access disclosure.
    order = request.slot_session_tasks
    assert set(order) == {"t_sess", "t_actor"}  # packet rows carry no work order
    assert "TRAJECTORY-RESULT-3-passed" in order["t_sess"] and "PREVIEW-BYTES" in order["t_sess"]
    assert "not guaranteed" in order["t_sess"] and str(workspace) in order["t_sess"]
    # Native row: the same packet minus its freely degradable tail, manifested.
    assert "TRAJECTORY-RESULT-3-passed" not in order["t_actor"] and "PREVIEW-BYTES" not in order["t_actor"]
    assert "retrieving_delivery" in order["t_actor"] and "report/summary.md" in order["t_actor"]
    assert "verification_receipts[0]" in order["t_actor"]
    # Both carry the task-stable contract the packet rows render; the FULL packet stays the authority.
    _stable, task_stable, _dynamic = _render_prompt_parts(request, api)
    assert task_stable.rstrip() in order["t_sess"] and task_stable.rstrip() in order["t_actor"]
    # Exercise the real root checklist, not only the synthetic pre-seam goldens:
    # each route carries one coaching owner without the old contradictory copy.
    for delivered in (_stable + task_stable + _dynamic,
                      contract + order["t_sess"], contract + order["t_actor"]):
        assert delivered.count(TIER_CLASSIFICATION_RULES) == 1
        assert "move it one tier up" not in delivered
        assert "not a weaker surrogate self-test" in delivered
    assert request.evidence["tool_trajectory"][0]["result"] == "TRAJECTORY-RESULT-3-passed"
    # The executor labels the slot itself: a work order carries no `Slot:` line of its own.
    assert "Slot:" not in order["t_sess"] and "Slot:" not in order["t_actor"]
    # The api pack states the contract once: route-owned keys never enter its rendered Policy JSON.
    assert "native_data_root" not in task_stable and "output_contract" not in task_stable


def test_wave_budget_gate_prices_only_the_api_money(structured_env, tmp_path):
    import ouroboros.review_substrate as rs
    from ouroboros import loop as loop_mod
    from ouroboros.tools import review_helpers

    gate_calls = []
    structured_env.setattr(
        rs, "run_review_request", lambda request, **kw: SimpleNamespace(aggregate_signal="PASS", actors=[]))
    structured_env.setattr(review_helpers, "review_wave_budget_gate", lambda _ctx, **kw: gate_calls.append(kw))
    governance, workspace = _roots(tmp_path)
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    loop_mod._execute_task_acceptance_panel(ctx)
    (kw,) = gate_calls
    # The session row is subscription, not API money; the native row is one episode send.
    assert kw["models"] == ["openai/gpt-5.6-luna", "openai/gpt-5.6-terra"] and kw["prompt_chars"] > 0
    # An all-session panel spends no API money: the gate is not consulted at all.
    structured_env.setenv("OUROBOROS_SUBAGENTS", json.dumps(_roster(_POOL_ROWS[1])))
    loop_mod._execute_task_acceptance_panel(ctx)
    assert len(gate_calls) == 1


def test_partial_source_refusal_spares_retrieving_rows_and_core_overflow_refuses_packet_rows(monkeypatch, tmp_path):
    from ouroboros import loop as loop_mod

    _offline_env(monkeypatch, _ROW_API, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}])
    _real_panel(monkeypatch, llm)
    partial = {**_ACCEPTANCE_PACKET, "__unresolved_partial_artifacts__": [{
        'tool': 'run_command', 'status': 'source_unavailable',
        'reason': 'fixture_packet_projection_unavailable', 'source_ref': {}}]}
    result = loop_mod._execute_task_acceptance_panel(_acceptance_ctx(
        tmp_path, evidence=partial, repo_dir=str(governance),
        workspace_root=str(workspace), workspace_mode="project"))
    by_id = {a["slot_id"]: a for a in result.actors}
    # The packet row is refused free (a partial PROJECTION is not complete evidence)
    # as a typed `not_dispatched` transport state — never a verdict — with the
    # refusal cause on the row; the native row reads the exact source itself and runs.
    assert by_id["t_api"]["status"] == "not_dispatched" and by_id["t_api"]["parsed"] is None
    assert "partial" in str(by_id["t_api"]["error"])
    assert by_id["t_actor"]["parsed"]["verdict"] == "PASS"
    assert len(llm.calls) == 1 and "tools" in llm.calls[0]
    # The immutable-core overflow refuses every PACKET row (no owner requirement is
    # truncated for anyone); the native row reads the complete packet as one exact
    # source instead of inheriting that refusal (#1329, tests/test_acceptance_source_first.py).
    fresh = tmp_path / "overflow"  # a separate task wallet: the panel above spent this one's cycle
    fresh.mkdir()
    governance, workspace = _roots(fresh)
    llm.script.append({"content": json.dumps(_CLEAN_VERDICT)})
    overflow = {**_ACCEPTANCE_PACKET, "__immutable_core_overflow__": True}
    result = loop_mod._execute_task_acceptance_panel(_acceptance_ctx(
        fresh, evidence=overflow, repo_dir=str(governance),
        workspace_root=str(workspace), workspace_mode="project"))
    by_id = {a["slot_id"]: a for a in result.actors}
    assert by_id["t_api"]["status"] == "not_dispatched" and "overflow" in str(by_id["t_api"]["error"])
    assert by_id["t_actor"]["parsed"]["verdict"] == "PASS"
    assert len(llm.calls) == 2 and "tools" in llm.calls[1]  # the packet row sent nothing


def test_retrieving_row_gets_no_format_repair_resend(monkeypatch, tmp_path):
    """A retrieving row canonicalizes its own answer (strict parse, then
    extraction over the collected transcript); the packet rows' second send for
    format repair never buys it a second episode."""
    from ouroboros import loop as loop_mod

    _offline_env(monkeypatch, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [{"content": "I looked around and it seems fine; no structured verdict."},
                                 {"content": "{}"}, {"content": "{}"}, {"content": "{}"}])
    _real_panel(monkeypatch, llm)
    result = loop_mod._execute_task_acceptance_panel(_acceptance_ctx(
        tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
        workspace_root=str(workspace), workspace_mode="project"))
    assert result.aggregate_signal == "DEGRADED"
    assert sum(1 for call in llm.calls if "tools" in call) == 1  # one episode; no repair resend


@pytest.mark.parametrize("rows", [(_ROW_API,), (_ROW_NATIVE,), (_ROW_API, _ROW_NATIVE)])
def test_wallet_stamp_claims_once_per_panel_on_api_native_and_mixed_rows(monkeypatch, tmp_path, rows):
    """R11: the paid identity is material, not route. One strict claim per
    panel whatever the rows' deliveries, and a spent wallet refuses a NEW paid
    identity before any reviewer is sent — on every delivery alike."""
    from ouroboros import loop as loop_mod
    from ouroboros.loop_acceptance_review import _total_paid_acceptance_cycles

    _offline_env(monkeypatch, *rows)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}] * len(rows))
    _real_panel(monkeypatch, llm)
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    result = loop_mod._execute_task_acceptance_panel(ctx)
    assert result.aggregate_signal == "PASS"
    assert _total_paid_acceptance_cycles(ctx) == 1
    assert len(llm.calls) == len(rows)
    # max_improvement_passes=0 → the tree may buy ONE panel: a changed candidate
    # (new paid identity) is refused fail-closed before any send.
    again = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), content="deliverable v2",
                            fresh_result=False, repo_dir=str(governance),
                            workspace_root=str(workspace), workspace_mode="project")
    refused = loop_mod._execute_task_acceptance_panel(again)
    assert refused.aggregate_signal == "DEGRADED"
    assert len(llm.calls) == len(rows) and _total_paid_acceptance_cycles(again) == 1


def test_session_row_claims_the_same_wallet_and_receives_the_full_packet(monkeypatch, tmp_path):
    from ouroboros import loop as loop_mod
    from ouroboros.loop_acceptance_review import _total_paid_acceptance_cycles
    from ouroboros.review_substrate import task_acceptance_is_clean

    FakeGateway = _fake_session(monkeypatch)
    _offline_env(monkeypatch, _ROW_SESSION)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [])
    _real_panel(monkeypatch, llm)
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    result = loop_mod._execute_task_acceptance_panel(ctx)
    assert result.aggregate_signal == "PASS" and task_acceptance_is_clean(result)
    assert _total_paid_acceptance_cycles(ctx) == 1
    assert llm.calls == []  # a conformant verdict: no extraction, no api pack
    (start,) = FakeGateway.instances[0].start_requests
    wire = json.dumps(start)
    assert "RETRIEVAL POINTERS" in wire and "TRAJECTORY-RESULT-3-passed" in wire and "not guaranteed" in wire
    # The start request carries the workspace root: as the run scope (parsed
    # field, never a substring of the serialized wire — json.dumps escapes the
    # backslashes of an OS-native path) and as the work order's retrieval pointer.
    assert start["scope"] == {"kind": "project", "root": str(workspace)}
    assert str(workspace) in start["prompt"]


def test_replayed_panel_keeps_the_delivery_it_actually_ran_on(monkeypatch, tmp_path):
    """Re-routing the triad after a panel does not re-buy it: the identical
    submission replays the recorded run, whose actors still say how they
    were executed (a native episode), not what the rows say today."""
    from ouroboros import loop as loop_mod
    from ouroboros.loop_acceptance_review import _prior_acceptance_run

    _offline_env(monkeypatch, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}])
    _real_panel(monkeypatch, llm)
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    record = loop_mod._record_host_acceptance_run(ctx, loop_mod._execute_task_acceptance_panel(ctx))
    assert record["actors"][0]["usage"]["delivery"] == "native_tool_rounds"
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(_roster(_ROW_API)))
    _cache, prior = _prior_acceptance_run(ctx.tools._ctx, ctx.llm_trace, ctx.review_binding["binding_hash"])
    assert prior is record and prior["actors"][0]["usage"]["delivery"] == "native_tool_rounds"
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# The timing row: telemetry written after a paid panel, read back by nothing.
# ---------------------------------------------------------------------------


def _timing(events, **row):
    from ouroboros.utils import append_jsonl

    append_jsonl(events, {"type": "task_acceptance_review_timing", **row})


def test_timing_event_names_the_deliveries_and_the_native_rounds(monkeypatch, tmp_path):
    from ouroboros import loop as loop_mod
    from ouroboros import task_pacing
    from ouroboros.utils import iter_jsonl_objects

    _offline_env(monkeypatch, _ROW_API, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}] * 2)
    _real_panel(monkeypatch, llm)
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    assert loop_mod._execute_task_acceptance_panel(ctx).aggregate_signal == "PASS"
    (event,) = [e for e in iter_jsonl_objects(task_pacing.acceptance_timing_events_path(ctx.tools._ctx))
                if e.get("type") == "task_acceptance_review_timing"]
    assert event["delivery"] == "native_tool_rounds"
    assert event["deliveries"] == ["api_chat", "native_tool_rounds"]
    assert event["native_rounds"] == 1 and event["native_rows"] == 1 and event["duration_sec"] > 0


def test_the_wave_gate_decides_on_one_work_order_send_per_paid_row(monkeypatch, tmp_path):
    """The money admission boundary, unchanged and now unconditional: the REAL
    wave gate (a seeded, priced wallet) prices ONE send per PAID row and
    nothing else — the session row rides the owner's subscription and is never
    API money, and a native row that goes on to take two real rounds is still
    admitted on one send. The panel prices the wave EXACTLY ONCE: there is no
    second, read-only pricing pass any more, and no rounds multiplier."""
    import ouroboros.review_substrate as rs
    from ouroboros import loop as loop_mod, task_pacing
    from ouroboros import usage_accounting as ua
    from ouroboros.utils import iter_jsonl_objects

    FakeGateway = _fake_session(monkeypatch)
    _offline_env(monkeypatch, _ROW_API, _ROW_SESSION, _ROW_NATIVE)
    _priced_offline_model(monkeypatch)
    admissions = _spy_admission(monkeypatch)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(
        tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}],
        native_script=[{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})]},
                       {"content": json.dumps(_CLEAN_VERDICT)}],
        scoped=True,
    )
    _real_panel(monkeypatch, llm, stub_gate=False)
    scope = _root_scope(tmp_path, root_limit_usd=50.0)
    _seed_root_ledger(scope)
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    with ua.usage_scope(scope):
        assert loop_mod._execute_task_acceptance_panel(ctx).aggregate_signal == "PASS"
    assert len(FakeGateway.instances[0].start_requests) == 1
    # ONE admission, over the two paid rows (api + native); the session row is absent.
    assert [len(a["models"]) for a in admissions] == [2]
    assert admissions[0]["models"] == ["openai/fake-reviewer"] * 2
    (event,) = [e for e in iter_jsonl_objects(task_pacing.acceptance_timing_events_path(ctx.tools._ctx))
                if e.get("type") == "task_acceptance_review_timing"]
    # The panel really did take two native rounds — recorded, and priced nowhere.
    assert event["native_rounds"] == 2 and event["native_rows"] == 1
    assert event["deliveries"] == ["api_chat", "agent_session", "native_tool_rounds"]

    del admissions[:]
    monkeypatch.setattr(
        rs, "run_review_request", lambda request, **kw: SimpleNamespace(aggregate_signal="PASS", actors=[]))
    with ua.usage_scope(scope):
        loop_mod._execute_task_acceptance_panel(ctx)
    # The recorded two-round history changes the next panel's price by nothing.
    assert [len(a["models"]) for a in admissions] == [2]
    assert all(a["fits"] and a["limit_usd"] == 50.0 and a["unpriced_slots"] == 0 for a in admissions)


def test_native_projection_never_turns_a_malformed_manifest_into_its_keys():
    from ouroboros.loop_acceptance_review import _retrieving_packet_projection

    projected = _retrieving_packet_projection({**_ACCEPTANCE_PACKET, "omissions_manifest": {"bad": "shape"}})
    assert [row["section"] for row in projected["omissions_manifest"]] == ["tool_trajectory", "artifact_previews"]
    assert "bad" not in json.dumps(projected["omissions_manifest"])
    # A well-formed manifest is extended, never replaced.
    kept = _retrieving_packet_projection({**_ACCEPTANCE_PACKET, "omissions_manifest": [{"section": "x", "reason": "y"}]})
    assert kept["omissions_manifest"][0] == {"section": "x", "reason": "y"} and len(kept["omissions_manifest"]) == 3
    # Nothing to omit: a PRESENT manifest of any non-list shape is normalized (None, dict,
    # str → []; a tuple is a sequence and is kept as a list); an absent key is not invented.
    bare = {k: v for k, v in _ACCEPTANCE_PACKET.items() if k not in ("tool_trajectory", "artifacts")}
    for malformed in (None, {"bad": "shape"}, "junk"):
        assert _retrieving_packet_projection({**bare, "omissions_manifest": malformed})["omissions_manifest"] == []
    row = {"section": "x", "reason": "y"}
    assert _retrieving_packet_projection({**bare, "omissions_manifest": (row,)})["omissions_manifest"] == [row]
    assert _retrieving_packet_projection({**_ACCEPTANCE_PACKET, "omissions_manifest": (row,)})["omissions_manifest"][0] == row
    assert "omissions_manifest" not in _retrieving_packet_projection(bare)


@pytest.mark.parametrize("tail", ["\n", "\r\n", "\n\n"])
def test_a_renderer_tail_ending_in_a_newline_still_loses_its_slot_label(structured_env, tmp_path, tail):
    """The executor labels the slot itself; the work order must carry no `Slot:`
    line even if the renderer's dynamic segment ever grows a trailing newline
    (or CRLF) after the label — the `.rstrip()` branch of the trim."""
    from ouroboros import review_execution
    from ouroboros.loop_acceptance_review import acceptance_retrieving_work_order
    from ouroboros.review_substrate import ReviewRequest

    original = review_execution._render_prompt_parts

    def _with_tail(request, slot):
        stable, task_stable, dynamic = original(request, slot)
        assert dynamic.endswith(f"Slot: {slot.slot_id}")  # the renderer's real tail today
        return stable, task_stable, dynamic + tail

    structured_env.setattr(review_execution, "_render_prompt_parts", _with_tail)
    request = ReviewRequest(surface="task_acceptance", goal="ship it", subject="deliverable",
                            evidence=dict(_ACCEPTANCE_PACKET), task_id="root-delivery",
                            policy={"classify_outcome_tier": True})
    retrieving = [slot for slot in review_pool_slots(role_hint="task acceptance") if slot.retrieves]
    assert [slot.slot_id for slot in retrieving] == ["t_sess", "t_actor"]
    acceptance_retrieving_work_order(request, retrieving, session_root=str(tmp_path), data_root=tmp_path)
    for slot_id, order in request.slot_session_tasks.items():
        assert "Slot:" not in order and not order.endswith(("\n", "\r"))
        assert "RETRIEVAL POINTERS" in order and "verification_receipts[0]" in order, slot_id
