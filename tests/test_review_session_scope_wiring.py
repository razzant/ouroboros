"""Surface wiring: the one review wave delivers retrieving seats without packs.

Split by theme out of ``tests/test_review_agent_session_route.py``. This module
owns the wave's surface wiring: session rows never build the API pack, the
mixed fan-out keeps one route per row, a coupling-only seat folded from the
old scope rows rides the same wave over its own route, and a retrieving seat's
findings never change severity based on its working-window size. The brief
every retrieving seat receives — the two parts, the repository index, the
governance tiers, the staged diff inline or paged — is pinned here as well.
"""

import asyncio
import hashlib
import json
import re
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.review_execution import REVIEW_SESSION_ROUTE_ENV
from ouroboros.reviewer_window import ReviewerWindow
from ouroboros.triad_review import parse_seat_answers

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport
from tests._review_session_route_shared import fake_route as __fake_route

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport
fake_route = __fake_route

from tests._review_session_route_shared import FakeGateway, FakeLLM, _terminal_detail
from tests._usage_store_testing import ledger_rows

# ---------------------------------------------------------------------------
# 5.2/5.6/5.7 — surface wiring: the wave delivers sessions without packs
# ---------------------------------------------------------------------------

def _coupling_matrix_rows():
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS

    return [
        {"item": item, "verdict": "PASS", "severity": "advisory",
         "reason": "checked the relevant code path and its consumers thoroughly"}
        for item in sorted(SCOPE_REQUIRED_ITEMS)
    ]


def _two_part_answer(rows=None, change=()):
    return {"change": list(change), "change_clean": not change, "coupling": rows or _coupling_matrix_rows()}


def _wave_ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext

    gov = tmp_path / "gov"
    drive = tmp_path / "data"
    gov.mkdir(exist_ok=True)
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=gov, drive_root=drive)
    ctx.task_id = "one-wave-wiring"
    return ctx


def _empty_plan():
    return {"models": [], "routes": [], "efforts": [], "session_targets": [], "session_profiles": [],
            "subagent_ids": [], "use_local": [], "slot_ids": [], "retrieves": []}


def _dispatch_plan(ctx, plan, *, session_task="BRIEF", session_root=""):
    """The real seat loop over ``plan`` (``_multi_model_review_async``), parsed."""
    from ouroboros.tools.review_multi_model import _multi_model_review_async

    result = asyncio.run(_multi_model_review_async(
        "staged diff", "", list(plan["models"]), ctx, routes=list(plan["routes"]), row_plan=plan,
        session_task=session_task, session_root=session_root))
    parsed = parse_seat_answers(result, dict(zip(plan["slot_ids"], plan["parts"])))
    return result, parsed


def test_mixed_retrieving_pool_sends_each_row_over_its_own_route(tmp_path, monkeypatch):
    """A MIXED pool of retrieving rows (a session and a natively reading api row)
    joins the one wave, each delivered over the route it was configured with
    (the catalog is the SSOT; ABI-10: the phase-5 route envs are retired). Both
    retrieve: both carry the brief, are asked both parts, and neither receives an
    assembled pack."""
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    from ouroboros.tools import review_admission as admission
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    set_review_pool(monkeypatch, pool_roster(
        pool_seat("pool_slot_1", "fake-review=fake-small", kind="agent_session"),
        pool_seat("pool_slot_2", "m/api", delivery="native")))
    dispatched: list = []

    def _capture(request, *, slots, drive_root, llm, usage_ctx=None):
        slot = slots[0]
        dispatched.append((slot.slot_id, slot.model, slot.route.value,
                           bool(request.session_task), bool(request.messages)))
        return SimpleNamespace(actors=[{
            "slot_id": slot.slot_id, "model": slot.model, "status": "ok",
            "raw_text": json.dumps(_two_part_answer()), "usage": {}, "prompt_ref": {}, "response_ref": {},
        }])

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", _capture)

    plan = admission.seat_vectors(commit_triad_delivery())
    assert plan["parts"] == [("change", "coupling"), ("change", "coupling")]
    _result, parsed = _dispatch_plan(_wave_ctx(tmp_path), plan)

    assert sorted(dispatched) == [
        ("pool_slot_1", "fake-review=fake-small", "agent_session", True, False),
        ("pool_slot_2", "m/api", "api_chat", True, False),
    ], dispatched
    assert [r.status for r in parsed.actor_records] == ["responded", "responded"]
    assert all(r.answers["coupling"]["verdict"] == "PASS" for r in parsed.actor_records)


def _brief_over(repo_dir, *, delegated, parts=("change", "coupling"), slot_id="slot_1", model="api/model"):
    from ouroboros.tools.review_brief_coupling import BriefInputs, build_retrieving_brief

    return build_retrieving_brief(repo_dir, BriefInputs(
        commit_message="session-delivery run", parts=tuple(parts), delegated=delegated, model=model,
        slot_id=slot_id))


def test_delegated_seat_goes_out_as_the_brief_and_never_builds_the_pack(tmp_path, fake_route, monkeypatch):
    """5.2 on the wave: a delegated seat goes out as the two-part brief —
    checklists, contract and intent context intact (5.3), the repository index
    and governance navigation instead of an assembled repository pack — and the
    coverage manifest is the pre-run disclosure record, not a gate (5.6): the
    expected read provenance rides as a non-blocking fact on a run that PASSES.
    This tree is not a git repository, so the host cannot capture the staged
    diff and the brief discloses that the reviewer retrieves it."""
    from ouroboros.review_execution import ReviewRouteKind

    ctx = _wave_ctx(tmp_path)
    brief, manifest = _brief_over(ctx.repo_dir, delegated=True, model="fake-review=fake-small")
    # D-12's ratified spelling: the field names the DELIVERY (the reviewer
    # retrieved the surface itself), not the transport.
    assert manifest["delivery"] == "agentic_retrieval"
    assert manifest["coverage"] == "agent_retrieval"
    assert manifest["parts"] == ["change", "coupling"]
    # Pre-run truth about provenance: a delegated session's reads are recovered
    # from its harness journal.
    assert manifest["read_provenance_expected"] == "harness_observed"
    assert manifest["diff_delivery"] == "retrieved_by_reviewer"
    assert "staged_diff_capture_failed" in manifest["diff_reason"]
    assert "coverage_incomplete" not in manifest  # retired framing (BIBLE P3 amendment)
    # Measured across `ouroboros/` and `web/` this key has exactly one writer and
    # no reader: the policy is disclosed, not defended by machinery.
    assert manifest["excluded_sensitive"] == {"policy": "preserved", "host_enforced": False}
    assert "Coupling questions" in brief
    assert "intent_alignment" in brief and "implicit_contracts" in brief
    assert "git diff --cached" in brief         # the disclosed retrieval pointer
    assert "Governance navigation (read on demand)" in brief   # the map, never whole
    assert "no empty \"coupling\"" in brief    # the matrix contract (contract B)

    fake_route.detail = _terminal_detail(json.dumps(_two_part_answer()), conformance="passed")
    plan = {**_empty_plan(), "models": ["fake-review=fake-small"], "routes": [ReviewRouteKind.AGENT_SESSION],
            "efforts": [""], "session_targets": ["fake-review=fake-small"], "session_profiles": [""], "subagent_ids": [""],
            "use_local": [None], "slot_ids": ["slot_1"], "retrieves": [True], "parts": [("change", "coupling")],
            "session_tasks": [brief]}
    _result, parsed = _dispatch_plan(ctx, plan, session_root=str(ctx.repo_dir))

    record = parsed.actor_records[0]
    assert record.status == "responded", (record.raw_text, _result["results"][0])
    assert len(record.parsed_items) == 8
    assert record.answers["change"]["verdict"] == "PASS" and record.answers["coupling"]["verdict"] == "PASS"
    start = fake_route.instances[0].start_requests[0]
    assert brief in start["prompt"]       # the brief (under the seat header), not a pack
    assert "Coupling questions" in start["prompt"]
    assert start["outputSchema"]["properties"]["coupling"]["minItems"] == 1


def _coupling_matrix_with_critical():
    rows = _coupling_matrix_rows()
    rows[0] = {**rows[0], "verdict": "FAIL", "severity": "critical",
               "reason": "the change contradicts a documented invariant on a live path"}
    return rows


@pytest.mark.parametrize(
    "window, provenance",
    [
        # The conservative fallback resolves to exactly the session floor NUMBER. It is
        # not evidence, and a numeric-only floor would have admitted it.
        (200_000, "unknown_conservative"),
        # Sourced, but genuinely below the floor.
        (131_072, "confirmed"),
        # Sourced and large.
        (1_000_000, "confirmed"),
        # The designated-default sentinel is a routing grant, never a measurement.
        (1_000_000, "designated_default_sentinel"),
    ],
)
def test_retrieving_seat_keeps_its_findings_whatever_its_window_evidence(
    tmp_path, fake_route, monkeypatch, window, provenance
):
    """A retrieving seat keeps its findings and its authority whatever its window
    evidence says: a small, unsourced or sentinel window neither demotes a
    critical finding nor removes the seat (BIBLE P3 — authority rests on the
    required-source manifest and its recorded coverage). The wave reduces to
    FAIL here because the seat reported a CRITICAL, which is the only reason a
    coupling answer ever blocks — on the delegated and the native delivery alike."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_ledger import build_rows, coupling_outcome, reduce_verdict
    from ouroboros.tools import scope_window as sw

    if provenance in ("confirmed", "asserted"):
        resolved = ReviewerWindow(window_tokens=int(window), status=provenance)
    elif provenance == "designated_default_sentinel":
        resolved = ReviewerWindow(window_tokens=int(window), status="")
    else:
        resolved = ReviewerWindow(window_tokens=0, status="")
    monkeypatch.setattr(sw, "scope_window", lambda *_a, **_k: resolved)
    fake_route.detail = _terminal_detail(json.dumps(_two_part_answer(_coupling_matrix_with_critical())),
                                         conformance="passed")

    def _native(request, *, slots, drive_root, llm, usage_ctx=None):
        return SimpleNamespace(actors=[{
            "slot_id": slots[0].slot_id, "model": slots[0].model, "status": "ok",
            "raw_text": json.dumps(_two_part_answer(_coupling_matrix_with_critical())),
            "usage": {}, "prompt_ref": {}, "response_ref": {},
        }])

    ctx = _wave_ctx(tmp_path)
    for route in (ReviewRouteKind.AGENT_SESSION, ReviewRouteKind.API_CHAT):
        if route is ReviewRouteKind.API_CHAT:
            monkeypatch.setattr("ouroboros.review_substrate.run_review_request", _native)
        plan = {**_empty_plan(), "models": ["fake-review=fake-small"], "routes": [route], "efforts": [""],
                "session_targets": ["fake-review=fake-small"], "session_profiles": [""], "subagent_ids": [""],
                "use_local": [None], "slot_ids": ["slot_1"], "retrieves": [True], "parts": [("change", "coupling")]}
        _result, parsed = _dispatch_plan(ctx, plan, session_root=str(ctx.repo_dir))
        rows = build_rows({"triad_raw": [r.to_dict() for r in parsed.actor_records]})
        verdict = reduce_verdict(rows)
        assert verdict["aggregate"] == "FAIL", (route, window, provenance, verdict)
        outcome = coupling_outcome(verdict, rows)
        reasons = " ".join(str(f.get("reason") or "") for f in outcome.critical_findings)
        assert "contradicts a documented invariant" in reasons
        assert "[advisory-only session scope reviewer]" not in reasons
        items = {str(f.get("item") or "") for f in outcome.advisory_findings}
        assert "scope_review_session_window_unproven" not in items
        assert not any(f.get("item") == "scope_review_sub_floor" for f in outcome.advisory_findings)


def test_all_retrieving_wave_answers_without_a_window_authority_gate(tmp_path, fake_route, monkeypatch):
    """A wave of two delegated seats with no window evidence at all answers
    authoritatively, and so does the same wave on a sourced 200K window: an
    unknown or small window neither arms nor disarms the gate. The asymmetry
    this replaced was the fail-open measured on a6a3c1f, where the same panel
    shape gave a BLOCKING api row and a non-blocking retrieving one."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_ledger import build_rows, coupling_outcome, reduce_verdict
    from ouroboros.tools import scope_window as sw

    fake_route.detail = _terminal_detail(json.dumps(_two_part_answer()), conformance="passed")
    sequence = {"value": 0}

    def unique_start(self, request, *, idempotency_key=""):
        self.start_requests.append(dict(request))
        self.start_keys.append(str(idempotency_key))
        sequence["value"] += 1
        return {"runId": f"run-wave-{sequence['value']}", "runDir": "/tmp/fake-run"}

    monkeypatch.setattr(FakeGateway, "start_run", unique_start)
    ctx = _wave_ctx(tmp_path)
    for resolved in (ReviewerWindow(window_tokens=0, status=""), ReviewerWindow(window_tokens=200_000, status="confirmed")):
        monkeypatch.setattr(sw, "scope_window", lambda *_a, **_k: resolved)
        plan = {**_empty_plan(), "models": ["fake-review=fake-small"] * 2,
                "routes": [ReviewRouteKind.AGENT_SESSION] * 2, "efforts": ["", ""],
                "session_targets": ["fake-review=fake-small"] * 2, "session_profiles": ["", ""],
                "subagent_ids": ["", ""], "use_local": [None, None], "slot_ids": ["scope_slot_1", "scope_slot_2"],
                "retrieves": [True, True], "parts": [("change", "coupling")] * 2}
        _result, parsed = _dispatch_plan(ctx, plan, session_root=str(ctx.repo_dir))
        rows = build_rows({"triad_raw": [r.to_dict() for r in parsed.actor_records]})
        verdict = reduce_verdict(rows)
        assert verdict["aggregate"] == "PASS", verdict
        assert verdict["quorum"] == {**verdict["quorum"], "assigned": 2, "responded": 2}
        outcome = coupling_outcome(verdict, rows)
        assert outcome.blocked is False and outcome.status == "responded"
        assert [s["slot_id"] for s in outcome.seats] == ["scope_slot_1", "scope_slot_2"]
        assert [s["status"] for s in outcome.seats] == ["responded", "responded"]


def test_wave_keeps_an_answered_seat_with_incomplete_read_coverage():
    """The same answer and findings count while the seat's read coverage stays
    visible on its record: read coverage is a diagnostic, never a withdrawn
    verdict and never a host-authored FAIL."""
    from ouroboros.review_ledger import build_rows, coupling_outcome, reduce_verdict

    partial = _coupling_matrix_rows()
    partial[1] = {**partial[1], "verdict": "FAIL", "severity": "advisory",
                  "reason": "a real observation the row still contributes"}
    results = [
        {"model": "api/big", "slot_id": "scope_slot_1", "verdict": "OK",
         "text": json.dumps({"coupling": _coupling_matrix_rows()}),
         "native_read_coverage": {"status": "complete", "sources": []}},
        {"model": "api/partial", "slot_id": "scope_slot_2", "verdict": "OK",
         "text": json.dumps({"coupling": partial}),
         "native_read_coverage": {"status": "incomplete", "sources": [
             {"path": "prompts/SYSTEM.md", "status": "incomplete"}]}},
    ]
    parsed = parse_seat_answers({"results": results}, {"scope_slot_1": ("coupling",), "scope_slot_2": ("coupling",)})
    records = [r.to_dict() for r in parsed.actor_records]
    rows = build_rows({"triad_raw": records})
    verdict = reduce_verdict(rows)

    # Both configured seats answered; read coverage does not withdraw a verdict.
    assert verdict["quorum"]["responded"] == 2 and verdict["per_question"]["coupling"] == "PASS"
    outcome = coupling_outcome(verdict, rows)
    assert outcome.blocked is False
    assert [f["item"] for f in outcome.advisory_findings] == [partial[1]["item"]]
    rowed = {r["slot_id"]: r for r in records}
    assert rowed["scope_slot_2"]["status"] == "responded"
    assert rowed["scope_slot_2"]["coverage"] == "incomplete"
    assert rowed["scope_slot_2"]["context_manifest"]["native_read_coverage"]["sources"][0]["path"] == "prompts/SYSTEM.md"
    assert rowed["scope_slot_1"]["coverage"] == "complete"


def test_triad_mixed_panel_builds_the_pack_once_for_api_rows_only(tmp_path, fake_route, monkeypatch):
    """5.2/5.3 on the wave: one panel, two deliveries. The packet row gets the
    historical pack; the session row gets the brief and answers contract B; an
    all-session panel never assembles the pack at all."""
    import ouroboros.tools.review as review_mod
    from ouroboros.review_execution import ReviewRouteKind

    chat_calls = []

    class PanelLLM:
        def chat(self, **kwargs):
            chat_calls.append(kwargs)
            return {"content": "[]\nNO_FINDINGS"}, {"prompt_tokens": 4, "completion_tokens": 2}

    monkeypatch.setattr(review_mod, "LLMClient", PanelLLM)
    monkeypatch.setattr(review_mod, "review_drive_root", lambda _ctx: tmp_path)
    fake_route.detail = _terminal_detail(json.dumps(_two_part_answer()), conformance="passed")

    result = json.loads(review_mod._handle_multi_model_review(
        None,
        content="Review the staged diff and context provided in the instructions above.",
        prompt="INSTRUCTIONS BODY",
        models=["api/model-a", "api/model-b"],
        stable_prefix_len=0,
        routes=[ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION],
        session_task="Review the staged diff: run `git diff --cached` yourself.",
        session_root="/tmp/fake-repo",
    ))
    rows = result["results"]
    assert len(rows) == 2
    assert rows[0]["slot_id"] == "slot_1" and rows[0]["text"] == "[]\nNO_FINDINGS"
    assert rows[1]["slot_id"] == "slot_2" and json.loads(rows[1]["text"])["change_clean"] is True
    assert len(chat_calls) == 1  # ONE api send; the session row never used chat
    # The session start carried the brief, not the giant pack.
    session_prompt = fake_route.instances[0].start_requests[0]["prompt"]
    assert "git diff --cached" in session_prompt
    assert "INSTRUCTIONS BODY" not in session_prompt
    # And the pack never reaches the session slot's DURABLE record either: the
    # api pack text must appear only in the api row's persisted prompt (gzip
    # content-addressed blobs), never in the session row's request payload.
    import gzip

    hits = []
    for record in tmp_path.rglob("*.gz"):
        text = gzip.decompress(record.read_bytes()).decode("utf-8", errors="replace")
        if "INSTRUCTIONS BODY" in text:
            hits.append(text)
    assert hits, "the api row's own durable prompt record should carry the pack"
    assert not any('"slot_id": "slot_2"' in text for text in hits)

    # All-session panel: the api pack (prompt) may be empty and nothing chats.
    chat_calls.clear()
    fake_route.reset()
    fake_route.detail = _terminal_detail(json.dumps(_two_part_answer()), conformance="passed")
    result = json.loads(review_mod._handle_multi_model_review(
        None,
        content="Review the staged diff and context provided in the instructions above.",
        prompt="",
        models=["api/model-a"],
        stable_prefix_len=0,
        routes=[ReviewRouteKind.AGENT_SESSION],
        session_task="Review the staged diff yourself.",
        session_root="/tmp/fake-repo",
    ))
    assert "error" not in result
    assert json.loads(result["results"][0]["text"])["coupling"]
    assert chat_calls == []


def test_triad_session_task_carries_criteria_and_nav_maps_not_evidence():
    import ouroboros.tools.review as review_mod
    from ouroboros.tools.review_subject import build_triad_session_task

    # One builder: the wave's two-part brief calls review_subject directly, and
    # the old `review._triad_session_task` shim that only tests exercised is gone.
    assert not hasattr(review_mod, "_triad_session_task")
    task = build_triad_session_task(
        goal_section="## Goal\nDo the thing.",
        scope_section="## Scope\nOnly here.",
        checklist_section="## Review Checklist\n- correctness",
        rebuttal_section="",
        review_history_section="",
        dev_guide_text="# Dev\n\n## Rules\n\ntext\n",
        architecture_text="## Parent\nbody\n### Child\nbody\n#### Detail\nbody\n",
    )
    assert "## Review Checklist" in task
    assert "## Goal" in task and "## Scope" in task
    assert "git diff --cached" in task           # subject pointer, not the diff
    assert "DEVELOPMENT.md (navigation map)" in task
    assert "ARCHITECTURE.md (navigation map)" in task
    assert "- Parent — lines 1-6" in task
    assert "  - Child — lines 3-6" in task
    assert "    - Detail — lines 5-6" in task
    assert "Read BIBLE.md and docs/DESIGN.md in full" in task


def test_session_schema_floor_matches_each_surfaces_clean_contract():
    """`{"findings": []}` is the honest clean verdict for an ordinary advisory
    session, but Skill Review (mandatory matrix rows) demands ``minItems: 1``; the
    commit gate's wave asks for contract B's one object. The floor lets a
    conforming engine regenerate instead."""
    from ouroboros.review_execution import (
        REVIEW_SESSION_OUTPUT_SCHEMA,
        review_session_output_schema,
    )
    from ouroboros.triad_review import TWO_PART_SESSION_OUTPUT_SCHEMA

    assert review_session_output_schema("multi_model_review") is TWO_PART_SESSION_OUTPUT_SCHEMA
    assert TWO_PART_SESSION_OUTPUT_SCHEMA["properties"]["coupling"]["minItems"] == 1
    # Advisory keeps the clean-capable shared schema: its ORDINARY mode's required
    # clean verdict is exactly the empty array, so a floor would starve it of the
    # one answer its contract demands (checklist coverage is checked downstream).
    assert review_session_output_schema("advisory_review") is REVIEW_SESSION_OUTPUT_SCHEMA
    assert "minItems" not in REVIEW_SESSION_OUTPUT_SCHEMA["properties"]["findings"]
    assert review_session_output_schema("skill_review")["properties"]["findings"]["minItems"] == 1
    # A shaped copy, never a mutation of the shared schema.
    assert "minItems" not in REVIEW_SESSION_OUTPUT_SCHEMA["properties"]["findings"]


def test_skill_review_all_session_composition_uses_strict_schema_and_no_api_fallback(
    tmp_path, fake_route, monkeypatch,
):
    """Skill Review → pass runner → shared substrate stays agentic end to end."""
    import ouroboros.tools.review as review_tool
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.skill_review_passes import run_skill_review_passes

    required = ("manifest_schema", "secrets_hygiene")
    fake_route.detail = _terminal_detail(json.dumps({"findings": [
        {"item": item, "verdict": "PASS", "severity": "advisory", "reason": "checked"}
        for item in required
    ]}), conformance="passed", model="fake-small")
    sequence = {"value": 0}

    def unique_start(self, request, *, idempotency_key=""):
        self.start_requests.append(dict(request))
        self.start_keys.append(str(idempotency_key))
        sequence["value"] += 1
        return {"runId": f"run-skill-{sequence['value']}", "runDir": "/tmp/fake-run"}

    monkeypatch.setattr(FakeGateway, "start_run", unique_start)
    no_api = FakeLLM()
    monkeypatch.setattr(review_tool, "LLMClient", lambda: no_api)
    ctx = _wave_ctx(tmp_path)
    ctx.task_id = "skill-review-task"
    root = tmp_path / "source-repo"
    root.mkdir()
    models = ["fake-review=fake-small", "fake-review=fake-small"]
    row_plan = {
        "routes": [ReviewRouteKind.AGENT_SESSION, ReviewRouteKind.AGENT_SESSION],
        "efforts": ["high", "high"],
        "session_targets": models,
        "session_profiles": ["profile-a", "profile-b"],
        "slot_ids": ["skill-slot-a", "skill-slot-b"],
    }

    _prompt, _evidence, result_text, error = run_skill_review_passes(
        ctx, ctx.drive_root, SimpleNamespace(name="happy_farm"),
        evidence={
            "manifest_dump": "{}", "content_hash": "hash", "history": [],
            "review_rebuttal": "", "required_items": required,
        },
        file_packs=["frozen payload bytes"], models=models, row_plan=row_plan,
        session_root=str(root),
        usage_attribution={"review_skill": "happy_farm", "review_wave_id": "wave-agentic"},
        build_prompt=lambda *_a, **_k: (
            "STABLE ONLY\nDYNAMIC SKILL", len("STABLE ONLY\n"), {}),
        run_review=review_tool._handle_multi_model_review,
    )

    assert error == ""
    result = json.loads(result_text)
    assert result["model_count"] == 2
    assert [row["slot_id"] for row in result["results"]] == [
        "skill-slot-a", "skill-slot-b",
    ]
    starts = [request for instance in fake_route.instances for request in instance.start_requests]
    assert len(starts) == 2
    assert no_api.calls == []
    assert {request["credentialProfileId"] for request in starts} == {"profile-a", "profile-b"}
    assert all(request["scope"]["root"] == str(root) for request in starts)
    assert all(request["outputSchema"]["properties"]["findings"]["minItems"] == 1
               for request in starts)
    assert all("exact frozen skill evidence" in request["prompt"] for request in starts)
    assert all("DYNAMIC SKILL" in request["prompt"] and "STABLE ONLY" not in request["prompt"]
               for request in starts)
    assert all("docs/ARCHITECTURE.md" in request["prompt"] and
               "docs/CREATING_SKILLS.md" in request["prompt"] for request in starts)
    assert all("Empty arrays and NO_FINDINGS are invalid" in request["prompt"]
               and "manifest_schema" in request["prompt"] for request in starts)
    ledger = ledger_rows(ctx.drive_root)
    sessions = [row for row in ledger if row.get("kind") == "subscription_session"]
    assert len(sessions) == 2
    assert {row["review_slot_id"] for row in sessions} == {"skill-slot-a", "skill-slot-b"}
    assert all(row["review_skill"] == "happy_farm" and
               row["review_wave_id"] == "wave-agentic" for row in sessions)

def test_skill_review_legacy_session_dispatch_keeps_shared_profile_pin(
    tmp_path, fake_route, monkeypatch,
):
    import ouroboros.tools.review as review_tool
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    from ouroboros.skill_review_passes import run_skill_review_passes

    # The session row + its credential pin are configured through the review
    # pool's catalog row (the phase-5 route envs are retired and ignored).
    from tests.review_pool_rosters import pool_roster, pool_seat

    monkeypatch.delenv(REVIEW_SESSION_ROUTE_ENV, raising=False)
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", pool_roster(pool_seat(
        "t1", "fake-review=fake-small", kind="agent_session", profile_id="legacy-profile", effort="high")))
    delivery = commit_triad_delivery()
    assert delivery["session_profiles"] == ["legacy-profile"]
    fake_route.detail = _terminal_detail(json.dumps({"findings": [
        {"item": "manifest_schema", "verdict": "PASS", "severity": "advisory",
         "reason": "checked"},
    ]}), conformance="passed")
    no_api = FakeLLM()
    monkeypatch.setattr(review_tool, "LLMClient", lambda: no_api)
    ctx = _wave_ctx(tmp_path)

    _prompt, _evidence, _result, error = run_skill_review_passes(
        ctx, ctx.drive_root, SimpleNamespace(name="happy_farm"),
        evidence={
            "manifest_dump": "{}", "content_hash": "hash", "history": [],
            "review_rebuttal": "", "required_items": ("manifest_schema",),
        },
        file_packs=["frozen"], models=delivery["models"], row_plan=delivery,
        session_root=str(tmp_path),
        usage_attribution={"review_skill": "happy_farm", "review_wave_id": "wave-legacy"},
        build_prompt=lambda *_a, **_k: ("STRICT LEGACY PROMPT", 0, {}),
        run_review=review_tool._handle_multi_model_review,
    )

    assert error == "" and no_api.calls == []
    starts = [request for instance in fake_route.instances for request in instance.start_requests]
    assert len(starts) == 1
    assert starts[0]["harnesses"] == ["fake-review"]
    assert starts[0]["credentialProfileId"] == "legacy-profile"
    assert starts[0]["model"] == "fake-small"
    assert starts[0]["effort"] == "high"


def test_brief_book_navigation_uses_physical_chapter_sources(tmp_path):
    """The brief's governance navigation indexes the map by the PHYSICAL chapter
    a section lives in, carries the read instruction, and inlines no chapter
    body. A book whose membership cannot be assembled keeps its name in the
    navigation and states the reason in the governance manifest (BIBLE P1)."""
    from ouroboros.tools.governance_context import governance_context
    from tests.test_reference_books import sources

    for path, raw in sources().items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    text = governance_context(
        tmp_path, surface="scope", touched_paths=(), delivery="retrieving",
        checklist_section_text="(coupling checklist)").navigation
    assert "docs/architecture/runtime.md" in text
    assert "Processes carry the work" in text          # the heading is indexed
    assert "The full startup mechanism" not in text    # the body is not
    assert 'root="system_repo"' in text
    (tmp_path / "docs/architecture/runtime.md").unlink()
    broken = governance_context(
        tmp_path, surface="scope", touched_paths=(), delivery="retrieving",
        checklist_section_text="(coupling checklist)")
    assert "ARCHITECTURE.md" in broken.navigation
    row = next(r for r in broken.manifest if r["path"] == "docs/ARCHITECTURE.md")
    assert row["disposition"] == "navigation"
    assert "runtime.md" in row["reason"]


# ---------------------------------------------------------------------------
# The brief a retrieving seat receives: intent and manifests, the repository
# index, the governance tiers, and the staged diff inline or paged (D2v2).
# ---------------------------------------------------------------------------


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


DIFF_MARKER = "UNIQUE_DIFF_BODY_MARKER"
BRIEF_TASK_ID = "two-part-brief-task"


def _staged_subject(tmp_path, *, payload_chars=0, newline="\n"):
    """A real repository with a staged change of a chosen size."""
    repo = tmp_path / "subject"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    # The `newline` parameter is the subject's EOL: pin Git so a host-level
    # autocrlf cannot silently turn the CRLF case back into the LF case.
    _git(repo, "config", "core.autocrlf", "false")
    (repo / ".gitignore").write_text(".review-drive/\n", encoding="utf-8", newline=newline)
    (repo / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8", newline=newline)
    (repo / "beta.py").write_text("import alpha\n\n\ndef beta():\n    return alpha.alpha()\n",
                                  encoding="utf-8", newline=newline)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "alpha.py").write_text(
        f"def alpha():\n    return 2  # {DIFF_MARKER}\n", encoding="utf-8", newline=newline)
    # A touched prompt is owed in full, so the required-source manifest is real.
    (repo / "prompts").mkdir(exist_ok=True)
    (repo / "prompts" / "SYSTEM.md").write_text("You are the runtime prompt.\n", encoding="utf-8", newline=newline)
    payload = "".join(f"PAYLOAD_LINE_{index:08d}\n" for index in range(payload_chars // 21))
    (repo / "gamma.py").write_text(f"gamma = 1\n{payload}", encoding="utf-8", newline=newline)
    _git(repo, "add", "-A")
    return repo


def _brief_inputs(repo, drive, **overrides):
    from ouroboros.tools.review_brief_coupling import BriefInputs, BriefIntent
    from ouroboros.tools.scope_required_sources import (
        required_sources_ref, scope_required_sources, staged_touched_paths,
        staged_tree_identity, touched_manifest,
    )

    touched = staged_touched_paths(repo)
    tree = staged_tree_identity(repo)
    rows = scope_required_sources(repo, touched, staged_tree_sha=tree)
    fields = dict(
        commit_message="fix: the brief carries what the reviewer needs",
        intent=BriefIntent(goal="Deliver the brief", scope="Only the coupling surface"),
        touched_paths=tuple(path for _status, path in touched),
        touched_manifest=touched_manifest(repo, touched),
        required_sources=rows,
        required_sources_ref=required_sources_ref(rows, staged_tree_sha=tree),
        model="api/scope-model",
        slot_id="slot_1",
        task_id=BRIEF_TASK_ID,
        source_root=str(drive),
    )
    fields.update(overrides)
    return BriefInputs(**fields)


def _seat_brief(ctx, repo, *, delegated, managed_subject=None):
    """ONE retrieving seat's brief and manifest the way the wave builds them."""
    from ouroboros.tools.review_admission import retrieving_brief_for_seat
    from ouroboros.tools.review_brief_coupling import BriefIntent
    from ouroboros.tools.review_helpers import load_checklist_section

    return retrieving_brief_for_seat(
        review_root=repo, governance_root=repo, path_subject=managed_subject, managed_subject=managed_subject,
        diff_text=None, layer="body", checklist_section=load_checklist_section("Change Review Checklist"),
        commit_message="review a subject", intent=BriefIntent(), parts=("change", "coupling"), delegated=delegated,
        model="fixture/model", slot_id="slot_1", drive_root=ctx.drive_root, task_id=BRIEF_TASK_ID,
        source_root=str(ctx.drive_root))


def test_the_brief_carries_the_index_the_governance_tiers_and_both_manifests(tmp_path):
    """One brief, four deliveries of context: the repository index (no bodies),
    the governance tiers with tier 1 ahead of every change-relative section, the
    navigation with its exact read instruction, and the two manifests. Each of
    them is recorded in the pre-run disclosure manifest as well as rendered."""
    from ouroboros.tools.review_brief_coupling import build_retrieving_brief
    from ouroboros.tools.review_helpers import REPO_ROOT

    repo = _staged_subject(tmp_path)
    drive = tmp_path / "data"
    drive.mkdir()
    task, manifest = build_retrieving_brief(
        repo, _brief_inputs(repo, drive, governance_repo_dir=REPO_ROOT))

    # The index: every tracked path's class, the touched paths' facts and their
    # importers — and no file body (beta.py imports alpha.py, so it is listed).
    assert "## Repository index" in task
    # An ordinary path is a bare row (the `indexed` label is implied, not repeated).
    assert re.search(r"^alpha\.py$", task, re.M) and "beta.py" in task
    assert "indexed\talpha.py" not in task
    assert manifest["repository_index"]["strategy"] == "repository_index"
    # alpha.py, gamma.py, prompts/SYSTEM.md
    assert manifest["repository_index"]["touched_count"] == 3
    assert manifest["repository_index"]["importer_count"] == 1  # beta.py
    assert manifest["repository_index"]["index_chars"] > 0

    # Tier 1 is inline and STABLE-FIRST: nothing change-relative precedes it.
    assert "## BIBLE.md" in task and "Philosophy version" in task
    assert "standing disclosures" in task          # CHECKLISTS_ARCHIVE rides with checklist item 7
    head = task.index("## BIBLE.md")
    for change_relative in ("## Intended transformation", "### Staged diff",
                            "## Repository index", "TOUCHED PATHS", "REQUIRED SOURCES"):
        assert head < task.index(change_relative), change_relative
    tiers = {row["path"]: row for row in manifest["governance_manifest"]}
    assert tiers["BIBLE.md"]["tier"] == 1 and tiers["BIBLE.md"]["disposition"] == "inline"
    assert tiers["docs/CHECKLISTS_ARCHIVE.md"]["disposition"] == "inline"
    assert tiers["docs/ARCHITECTURE.md"]["disposition"] == "navigation"

    # The navigation names what is not inlined and says exactly how to read it.
    assert "Governance navigation (read on demand)" in task
    assert 'read_file(root="system_repo", path=..., start_line=A, max_lines=N)' in task

    # Both manifests: what changed, and what the reviewer is owed in full.
    assert "TOUCHED PATHS" in task and "alpha.py (modified" in task
    assert "REQUIRED SOURCES" in task and "list is a MINIMUM" in task
    assert "prompts/SYSTEM.md (added" in task          # a touched prompt is owed in full
    assert manifest["native_required_sources_ref"]["policy"] == "v2"
    assert manifest["native_required_sources_ref"]["required_source_count"] == 1
    assert manifest["brief_chars"] == len(task)
    # The two parts are both there, and the Part 2 brief is hashed on its own.
    assert "## Part 1 — The change" in task and "## Part 2 — Coupling questions" in task
    assert manifest["sha"]["coupling_brief_sha"] and manifest["sha"]["change_prompt_sha"]


def test_a_staged_diff_that_fits_the_first_send_is_inlined(tmp_path):
    """The reviewer reads the change itself, not a pointer to it, whenever the
    whole first send lands under the seat's own bound."""
    from ouroboros.tools.review_brief_coupling import build_retrieving_brief

    repo = _staged_subject(tmp_path)
    drive = tmp_path / "data"
    drive.mkdir()
    task, manifest = build_retrieving_brief(repo, _brief_inputs(repo, drive))

    assert manifest["diff_delivery"] == "inline"
    assert DIFF_MARKER in task
    assert "--- a/alpha.py" in task and "+++ b/alpha.py" in task
    assert manifest["first_send_chars"] < manifest["first_send_ceiling"]
    assert manifest["diff_chars"] > 0
    assert "diff_source" not in manifest


@pytest.mark.parametrize("delegated", [False, True], ids=["native", "session"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_a_staged_diff_above_the_first_send_is_paged_as_one_exact_source(tmp_path, delegated, newline):
    """A diff too large for the first send is not refused and not truncated: it
    is stored ONCE, byte-exactly, at an address the seat's own reader reaches —
    the task artifact store for a native episode, the review's git-ignored
    project view for a delegated session — and the brief carries the address,
    the size and the digest instead of the body."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tools.review_binary_context import capture_staged_diff
    from ouroboros.tools.scope_required_sources import required_sources_ref, source_text_identity
    from ouroboros.tools.registry import ToolContext

    repo = _staged_subject(tmp_path, payload_chars=900_000, newline=newline)
    drive = tmp_path / "data"
    drive.mkdir()
    expected = capture_staged_diff(repo)
    assert len(expected) > 900_000

    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=BRIEF_TASK_ID)
    task, manifest = _seat_brief(ctx, repo, delegated=delegated)

    assert manifest["diff_delivery"] == "paged", manifest.get("diff_paging_reason")
    assert manifest["first_send_chars"] < manifest["first_send_ceiling"]
    assert manifest["diff_chars"] == len(expected)
    # The body is NOT in the brief; its address, size and digest are.
    assert DIFF_MARKER not in task
    assert "PAYLOAD_LINE_00000001" not in task
    source = manifest["diff_source"]
    rows = manifest["native_required_sources"]
    diff_row = next(row for row in rows if row["disposition"] == "review_subject")
    assert diff_row == source["required_row"]
    assert all(diff_row[key] == value for key, value in source_text_identity(expected.encode()).items())
    assert diff_row["candidate_tree"] == manifest["native_required_sources_ref"]["staged_tree_sha"]
    assert manifest["native_data_root"] == str(drive)
    assert manifest["native_required_sources_ref"] == required_sources_ref(
        rows, staged_tree_sha=diff_row["candidate_tree"])
    assert manifest["native_required_sources_ref"]["required_source_count"] == 2
    assert source["sha256"] in task and f"{len(expected):,} chars" in task
    assert "in ranges" in task
    # And the stored source round-trips byte-exactly.
    assert read_actor_source_bytes(drive, BRIEF_TASK_ID, source).decode("utf-8") == expected
    if delegated:
        from ouroboros.review_session_reads import fold_session_coverage, session_source_reader

        relative = source["session_relative_path"]
        assert relative in task
        assert (repo / relative).read_bytes().decode("utf-8") == expected
        assert diff_row["root"] == "session_root" and diff_row["path"] == relative
        coverage = fold_session_coverage([], rows, resolve_file=session_source_reader(str(repo)))
    else:
        from ouroboros.review_native_episode import NativeToolRoundReviewExecutor

        assert source["path"] in task
        assert 'read_file(root="artifact_store"' in task
        assert diff_row["root"] == "artifact_store" and diff_row["path"] == source["path"]
        coverage = NativeToolRoundReviewExecutor._read_coverage(SimpleNamespace(
            assignment=SimpleNamespace(request=SimpleNamespace(policy={"native_required_sources": rows})),
            _inspection_ctx=ctx, _tool_receipts=[]))
    assert coverage["status"] == "incomplete"
    assert next(row for row in coverage["sources"] if row["path"] == diff_row["path"])["missing_ranges"] == [
        [0, source_text_identity(expected.encode())["complete_chars"]]]


@pytest.mark.parametrize("delegated", [False, True], ids=["native", "session"])
@pytest.mark.parametrize("managed", [False, True], ids=["ordinary", "managed"])
@pytest.mark.parametrize("git_autocrlf", ["false", "true"], ids=["raw_blob", "lf_blob"])
def test_renamed_prompt_preimage_is_readable_and_covered_without_a_paged_diff(
    tmp_path, delegated, managed, git_autocrlf,
):
    """Rename detection can omit the body; the exact old prompt stays readable
    and has its own diagnostic row on both transports."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tools.registry import ToolContext

    repo = _staged_subject(tmp_path)
    _git(repo, "config", "core.autocrlf", git_autocrlf)
    old = repo / "prompts/SYSTEM.md"
    raw = "Original α prompt.\r\nSecond line.\r\n".encode("utf-8")
    old.write_bytes(raw)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prompt baseline")
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip()
    # The preimage contract names Git's blob, which can differ from the CRLF
    # worktree file. Exercise both forms on every host instead of inheriting Git's default.
    blob = subprocess.check_output(["git", "show", f"{baseline}:prompts/SYSTEM.md"], cwd=repo)
    assert blob == (raw.replace(b"\r\n", b"\n") if git_autocrlf == "true" else raw)
    (repo / "docs").mkdir()
    old.rename(repo / "docs/renamed.md")
    _git(repo, "add", "-A")
    subject = None
    if managed:
        from ouroboros.tools.review_subject import ManagedReviewSubject
        from ouroboros.tools.scope_required_sources import staged_tree_identity

        tree = staged_tree_identity(repo)
        subject = ManagedReviewSubject(
            repo_dir=str(repo), m0_tree=baseline, staged_tree=tree, m0_missing_reason="",
            pre_update_sha=baseline, target_sha=baseline, conflict_paths=(),
            diff=subprocess.check_output(["git", "diff", baseline, tree], cwd=repo, text=True),
            name_status=(("R100", "docs/renamed.md"),), full_candidate_paths=1,
            resolution_paths=1, fallback_full_diff=False)
    drive = tmp_path / "data"
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=BRIEF_TASK_ID)
    task, manifest = _seat_brief(ctx, repo, delegated=delegated, managed_subject=subject)
    assert manifest["diff_delivery"] == "inline"
    assert manifest["native_data_root"] == str(drive)
    assert len(manifest["native_required_sources"]) == 1
    row = manifest["native_required_sources"][0]
    assert row["disposition"] == "deleted_preimage" and row["preimage_of"] == "prompts/SYSTEM.md"
    assert row["preimage"] == f"{baseline if managed else 'HEAD'}:prompts/SYSTEM.md"
    assert row["candidate_tree"] == manifest["native_required_sources_ref"]["staged_tree_sha"]
    source = manifest["preimage_sources"][0]
    assert read_actor_source_bytes(drive, BRIEF_TASK_ID, source) == blob
    assert row["source_revision"] == hashlib.sha256(blob).hexdigest()
    assert row["path"] in task and "preimage of prompts/SYSTEM.md" in task
    if delegated:
        from ouroboros.review_session_reads import (
            fold_session_coverage, parse_session_read_receipts, session_source_reader,
        )

        journal = tmp_path / "events.jsonl"
        tool = {"name": "shell", "kind": "command", "use_id": "read-preimage",
                "target": f"cat {row['path']}"}
        journal.write_text("\n".join(json.dumps(event) for event in (
            {"type": "tool_call", "tool": tool},
            {"type": "tool_result", "tool": {**tool, "status": "ok", "exit_code": 0}},
        )), encoding="utf-8")
        receipts = parse_session_read_receipts([journal], scope_root=str(repo))
        coverage = fold_session_coverage(receipts, [row], resolve_file=session_source_reader(str(repo)))
    else:
        from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
        from ouroboros.tools.core_file_tools import _read_file

        executor = object.__new__(NativeToolRoundReviewExecutor)
        executor.assignment = SimpleNamespace(request=SimpleNamespace(policy={"native_required_sources": [row]}))
        executor._inspection_ctx = ctx
        body = _read_file(ctx, row["path"], root=row["root"])
        receipt = executor._read_extent(body, len(body))
        executor._tool_receipts = [{"tool": "read_file", "outcome": "executed", "delivered": True,
                                    "opened_root": row["root"], "opened_path": row["path"], **receipt}]
        coverage = executor._read_coverage()
    assert coverage["status"] == "complete"
    assert coverage["sources"][0]["covered_chars"] == len(blob.decode("utf-8").replace("\r\n", "\n"))


@pytest.mark.parametrize("available_store", [False, True], ids=["no_store", "missing_blob"])
def test_unavailable_required_preimage_stays_a_diagnostic_row(tmp_path, available_store):
    from ouroboros.review_session_reads import fold_session_coverage, session_source_reader
    from ouroboros.tools.review_brief_coupling import build_retrieving_brief
    from ouroboros.tools.scope_required_sources import required_sources_ref

    repo = _staged_subject(tmp_path)
    drive = tmp_path / "data"
    drive.mkdir()
    # The first case has a real source and no store; the second has a store
    # but names a missing baseline source. Neither claims a delivered preimage.
    row = {"root": "active_workspace", "path": "prompts/REMOVED.md", "disposition": "deleted",
           "coverage_basis": "preimage_unavailable",
           "preimage": "HEAD:absent.md" if available_store else "HEAD:alpha.py"}
    task, manifest = build_retrieving_brief(repo, _brief_inputs(
        repo, drive, required_sources=[row], required_sources_ref=required_sources_ref([row]),
        source_root=str(drive) if available_store else ""))
    rows = manifest["native_required_sources"]
    assert len(rows) == 1 and rows[0]["coverage_basis"] == "preimage_unavailable"
    assert "preimage not delivered" in rows[0]["reason"] and "preimage not delivered" in task
    assert manifest["native_required_sources_ref"] == required_sources_ref(rows)
    assert not manifest["preimage_sources"]
    coverage = fold_session_coverage([], rows, resolve_file=session_source_reader(str(repo)))
    assert coverage["status"] == "unobserved" and coverage["required_source_count"] == 1
    assert coverage["sources"][0]["status"] == "unobserved"


@pytest.mark.parametrize("matching_governance", [False, True], ids=["different_text", "exact_text"])
def test_inline_governance_satisfies_only_the_exact_candidate_source(tmp_path, matching_governance):
    from ouroboros.tools.review_brief_coupling import build_retrieving_brief

    repo = _staged_subject(tmp_path)
    (repo / "BIBLE.md").write_text("Candidate constitution.\n", encoding="utf-8")
    _git(repo, "add", "BIBLE.md")
    governance = repo
    if not matching_governance:
        governance = tmp_path / "other-governance"
        governance.mkdir()
        (governance / "BIBLE.md").write_text("Different constitution.\n", encoding="utf-8")
    drive = tmp_path / "data"
    drive.mkdir()
    task, manifest = build_retrieving_brief(
        repo, _brief_inputs(repo, drive, governance_repo_dir=governance))
    row = next(row for row in manifest["native_required_sources"] if row["path"] == "BIBLE.md")
    assert row["coverage_basis"] == ("delivered_inline" if matching_governance else "candidate_blob")
    if matching_governance:
        assert "Candidate constitution." in task
        assert "delivered inline in full, no second read needed" in task


def test_the_brief_of_a_three_file_change_on_the_real_tree_is_measured(tmp_path):
    """The two-part brief's size on the REAL repository, without the staged
    diff, with its composition printed. The number is the whole reason the
    packet is gone: the retired scope packet's fixed part alone was 1.21 MB.

    The ceiling is measured, not aspirational — it is the sum of the owner's own
    decisions: tier-1 BIBLE plus the standing disclosures inline, the repository
    index over the tracked paths, the DEVELOPMENT chapters this change activates,
    the book navigation, and now BOTH checklists (the change checklist of Part 1
    beside the coupling questions of Part 2).
    """
    from ouroboros.tools.review_brief_coupling import BriefInputs, build_retrieving_brief
    from ouroboros.tools.review_helpers import REPO_ROOT, load_checklist_section
    from ouroboros.tools.scope_required_sources import (
        required_sources_ref, scope_required_sources, touched_manifest,
    )

    touched = [("M", "ouroboros/tools/review_brief_coupling.py"),
               ("M", "ouroboros/tools/review_admission.py"),
               ("M", "ouroboros/review_ledger.py")]
    rows = scope_required_sources(REPO_ROOT, touched)
    task, manifest = build_retrieving_brief(REPO_ROOT, BriefInputs(
        commit_message="one brief, two parts",
        touched_paths=tuple(path for _status, path in touched),
        touched_manifest=touched_manifest(REPO_ROOT, touched),
        required_sources=rows,
        required_sources_ref=required_sources_ref(rows),
        checklist_section=load_checklist_section("Change Review Checklist"),
        model="api/scope-model", slot_id="slot_1",
    ))
    sections = manifest["brief_sections"]
    without_diff = len(task) - sections["diff_slot"]
    print(f"\ntwo-part brief on the real tree: {len(task):,} chars "
          f"({without_diff:,} without the diff slot)")
    for name, chars in sorted(sections.items(), key=lambda item: -item[1]):
        print(f"  {name:32s} {chars:>9,}")
    # Measured without the diff slot (2026-10-08): 207,462 on packet B's tree,
    # 210,553 on the integrated PR-3 tree, whose canon adds the BIBLE review
    # paragraph and the Coupling questions section (the retired scope brief
    # alone measured 209,106). The ceiling keeps about the same headroom for the
    # next index growth (~23K), not a rounding of the measurement.
    # The engine facts/lifecycle merge (2026-10-08) adds 161 index chars inside this headroom.
    assert without_diff < 234_000, without_diff
    assert sections["repository_index"] > 20_000          # the index really ran
    assert sections["governance_stable_inline"] > 40_000  # BIBLE really inline
    assert sections["change_checklist"] > 0 and sections["coupling_checklist"] > 0
