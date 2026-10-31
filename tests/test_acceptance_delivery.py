"""The review pool's one slot builder, and offline reviewer fixtures other suites import.

ONE builder -- ``reviewer_slot_config.review_pool_slots`` -- turns the catalog's
review-eligible rows into ``ReviewSlot`` objects for plan review and for the
commit/skill/result review vectors (``commit_triad_delivery``), so every surface
carries each row's own delivery, effort, credential pin and stable id (the catalog
id). The task-acceptance panel this module once pinned is gone (a task's result is
reviewed as ``review_change(subject="result")``, see ``tests/test_result_review.py``);
the offline fixtures below (a root delivery context, the scripted ledger-crossing
reviewer, the offline Claudexor session, the pool rows) stay importable for the
suites that use them.
"""

import json
from types import SimpleNamespace

import pytest

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
    """Slot construction may measure a native row's window; see `provider_catalog_offline`."""


@pytest.fixture()
def structured_env(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(_ROSTER))
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_REVIEW_ROUTES", "OUROBOROS_REVIEW_SESSION_ROUTE"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# One builder for plan review and the commit/skill/result review vectors.
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
    # The commit/skill/result vectors are a projection of the same slots.
    vectors = commit_triad_delivery()
    assert vectors["slot_ids"] == [s.slot_id for s in acceptance]
    assert vectors["models"] == [s.model for s in acceptance]
    assert vectors["routes"] == [s.route for s in acceptance]
    assert vectors["session_profiles"] == ["", "acct-1", ""]
    assert vectors["subagent_ids"] == ["t_api", "t_sess", "t_actor"]
    assert vectors["retrieves"] == [False, True, True]
    assert vectors["legacy_skill_fingerprint"] is False


# ---------------------------------------------------------------------------
# Offline reviewer fixtures shared with other suites.
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
# The source/wallet fixture promises the original Ask shape explicitly.
_ROW_SESSION = _row("t_sess", "fake-review=fake-small", kind="agent_session", access="readonly")


def _acceptance_ctx(tmp_path, *, evidence=None, task_metadata=None, content="deliverable",
                    fresh_result=True, max_improvement_passes=0, **tool_ctx_fields):
    """A root delivery context over a real RUNNING root task (``root-delivery``): its tool
    context, the answer under delivery, a trace and the conversation. It once built the
    task-acceptance panel's context; that panel and its review binding are gone, so it
    carries only the root-task fields a delivery or review test reads."""
    from ouroboros.contracts.task_contract import build_task_contract
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
    return SimpleNamespace(
        tools=SimpleNamespace(_ctx=tool_ctx), content=content, task_id="root-delivery",
        task_type="task", llm_trace={"tool_calls": []}, drive_root=tmp_path,
        messages=[{"role": "system", "content": "policy"}, {"role": "user", "content": "goal"}],
        emit_progress=lambda _text, *, incident=None: None,
        evidence=evidence if evidence is not None else {"evidence": "complete"},
    )


def _offline_env(monkeypatch, *rows):
    """An offline review pool (fake model ids never reach a provider)."""
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(_roster(*rows)))
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_REVIEW_ROUTES", "OUROBOROS_REVIEW_SESSION_ROUTE"):
        monkeypatch.delenv(key, raising=False)


class _EpisodeLLM:
    """Scripted `chat()` that crosses the real durable attempt ledger on every
    send — the production `LLMClient` does, and both the packet row and the
    native episode bind the review stamp around that crossing — so the
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
    """The offline Claudexor /v2 surface answering the clean object verdict."""
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
