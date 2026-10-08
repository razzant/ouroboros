"""Surface wiring: scope and triad deliver sessions without packs.

Split by theme out of ``tests/test_review_agent_session_route.py``. This module
owns the scope/triad surface wiring: session rows never build the API pack, the
mixed fanout keeps one route per row, and a retrieving review's independent
findings no longer change severity based on its working-window size.
"""

import hashlib
import json
import re
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.review_execution import (
    REVIEW_SESSION_ROUTE_ENV,
)
from ouroboros.review_substrate import (
    scope_reviewer_slots,
)
from ouroboros.reviewer_window import ReviewerWindow

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport
from tests._review_session_route_shared import fake_route as __fake_route

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport
fake_route = __fake_route

from tests._review_session_route_shared import (
    FakeGateway,
    FakeLLM,
    _terminal_detail,
)
from tests._usage_store_testing import ledger_rows

# ---------------------------------------------------------------------------
# 5.2/5.6/5.7 — surface wiring: scope and triad deliver sessions without packs
# ---------------------------------------------------------------------------

def _scope_matrix_rows():
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS

    return [
        {"item": item, "verdict": "PASS", "severity": "advisory",
         "reason": "checked the relevant code path and its consumers thoroughly"}
        for item in sorted(SCOPE_REQUIRED_ITEMS)
    ]


def _scope_ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext

    gov = tmp_path / "gov"
    drive = tmp_path / "data"
    gov.mkdir(exist_ok=True)
    drive.mkdir(exist_ok=True)
    return ToolContext(repo_dir=gov, drive_root=drive)

def test_mixed_scope_fanout_sends_each_row_over_its_own_route(tmp_path, monkeypatch):
    """A MIXED scope configuration must deliver each row over the route it was
    configured with.

    `_call_scope_llm` rebuilt its slot from `scope_reviewer_slots([model])`, and a
    one-element rebuild historically re-derived ROUTES **row 1** — so on a mixed
    panel the configured api row inherited agent_session while its request
    carried the api pack and no session task: a deterministic
    ReviewRouteUnavailable error actor that failed the blocking scope gate. The
    mixed panel now comes from the structured SSOT (ABI-10: the phase-5 route
    envs are retired) and the caller's fanned-out route stays authoritative.
    """
    import ouroboros.tools.scope_review as scope_mod

    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [
            {"slot_id": "t_api", "route": {"kind": "api_chat", "target_id": "m/api"}},
        ],
        "scope": [
            {"slot_id": "scope_slot_1",
             "route": {"kind": "agent_session", "target_id": "m/session"}},
            {"slot_id": "scope_slot_2",
             "route": {"kind": "api_chat", "target_id": "m/api"}},
        ],
    }))
    dispatched: list = []

    def _capture(request, *, slots, drive_root, llm, usage_ctx=None):
        slot = slots[0]
        dispatched.append((slot.slot_id, slot.model, slot.route.value,
                           bool(request.session_task), bool(request.messages)))
        return SimpleNamespace(actors=[{
            "slot_id": slot.slot_id, "model": slot.model, "status": "ok",
            "raw_text": json.dumps(_scope_matrix_rows()),
            "usage": {}, "prompt_ref": {}, "response_ref": {},
        }])

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", _capture)
    monkeypatch.setattr(scope_mod, "_scope_window",
                        lambda *_a, **_k: ReviewerWindow(
                            window_tokens=1_000_000, status="confirmed"))

    for slot in scope_reviewer_slots():
        scope_mod.run_scope_review(
            _scope_ctx(tmp_path), "mixed route fan-out",
            scope_model=slot.model, slot_id=slot.slot_id, route=slot.route,
        )

    # Row 1 is the delegated session, row 2 the native inspection episode: both
    # retrieve, so both carry the brief and neither receives an assembled pack.
    assert dispatched == [
        ("scope_slot_1", "m/session", "agent_session", True, False),
        ("scope_slot_2", "m/api", "api_chat", True, False),
    ], dispatched

def test_scope_session_delivery_never_builds_the_pack(tmp_path, fake_route, monkeypatch):
    """5.2 on scope: a delegated scope row goes out as the brief — checklist,
    contract and intent context intact (5.3), the repository index and
    governance navigation instead of an assembled repository pack — and the
    coverage manifest is the pre-run disclosure record, not a gate (5.6): the
    expected read provenance rides as a non-blocking fact on a run that
    PASSES. This tree is not a git repository, so the host cannot capture the
    staged diff and the brief discloses that the reviewer retrieves it."""
    import ouroboros.tools.scope_review as scope_mod
    from ouroboros.review_execution import ReviewRouteKind

    monkeypatch.setattr(scope_mod, "_scope_window",
                        lambda *_a, **_k: ReviewerWindow(
                            window_tokens=1_000_000, status="confirmed"))
    fake_route.detail = _terminal_detail(
        json.dumps({"findings": _scope_matrix_rows()}), conformance="passed",
    )
    result = scope_mod.run_scope_review(
        _scope_ctx(tmp_path), "session-delivery scope run",
        scope_model="api/scope-model", slot_id="scope_slot_1",
        route=ReviewRouteKind.AGENT_SESSION,
    )
    assert result.blocked is False
    assert result.status == "responded"
    assert len(result.parsed_items) == 8
    manifest = result.context_manifest
    # D-12's ratified spelling: the field names the DELIVERY (the reviewer
    # retrieved the surface itself), not the transport — `agent_session` is the
    # route kind's own name, and the manifest used to answer with it.
    assert manifest["delivery"] == "agentic_retrieval"
    assert manifest["coverage"] == "agent_retrieval"
    # Pre-run truth about provenance, not an attestation nobody performed: a
    # delegated session's reads are recovered from its harness journal.
    assert manifest["read_provenance_expected"] == "harness_observed"
    assert manifest["diff_delivery"] == "retrieved_by_reviewer"
    assert "staged_diff_capture_failed" in manifest["diff_reason"]
    assert "coverage_incomplete" not in manifest  # retired framing (BIBLE P3 amendment)

    # D-12 also asked that readers stay compatible with the old spelling. There
    # is nothing to be compatible WITH: measured across `ouroboros/` and `web/`,
    # this key has exactly one writer and no reader — the manifest is a durable
    # forensic row whose audience is a person. So the clause had no subject, and
    # that is DISCLOSED here rather than defended by machinery. I built the
    # defence twice before writing this line (a compatibility helper, then a
    # repo-wide reader sweep) and both were guards over an empty set; the rule
    # they broke is that a disclosed residual beats a widened patch.
    assert manifest["excluded_sensitive"] == {"policy": "preserved", "host_enforced": False}

    start = fake_route.instances[0].start_requests[0]
    prompt = start["prompt"]
    assert "Intent / Scope Review Checklist" in prompt
    assert "intent_alignment" in prompt and "implicit_contracts" in prompt
    assert "git diff --cached" in prompt         # the disclosed retrieval pointer
    assert "Governance navigation (read on demand)" in prompt   # the map, never whole
    assert "There is no all-clear shortcut in this mode" in prompt  # matrix contract


def _run_session_scope(tmp_path, fake_route, monkeypatch, *, window, provenance, rows=None):
    """One session-delivered scope row under a given window evidence pair."""
    import ouroboros.tools.scope_review as scope_mod
    from ouroboros.review_execution import ReviewRouteKind

    # Ported onto the evidence-typed resolver (ReviewerWindow): sourced provenance
    # rides `status`; the conservative fallback is NO evidence (window_tokens=0,
    # sizing falls back); the designated-default sentinel is a NUMBER with no
    # status — a routing grant, never a measurement.
    if provenance in ("confirmed", "asserted"):
        _resolved = ReviewerWindow(window_tokens=int(window), status=provenance)
    elif provenance == "designated_default_sentinel":
        _resolved = ReviewerWindow(window_tokens=int(window), status="")
    else:
        _resolved = ReviewerWindow(window_tokens=0, status="")
    monkeypatch.setattr(scope_mod, "_scope_window", lambda *_a, **_k: _resolved)
    fake_route.detail = _terminal_detail(
        json.dumps({"findings": rows if rows is not None else _scope_matrix_rows()}),
        conformance="passed",
    )
    return scope_mod.run_scope_review(
        _scope_ctx(tmp_path), "session-delivered scope row",
        scope_model="session/reviewer", slot_id="scope_slot_1",
        route=ReviewRouteKind.AGENT_SESSION,
    )


def _scope_matrix_with_critical():
    rows = _scope_matrix_rows()
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
        # The designated-default sentinel is a routing grant, never a measurement.
        (1_000_000, "designated_default_sentinel"),
    ],
)
def test_session_scope_preserves_findings_when_window_evidence_is_small_or_unknown(
    tmp_path, fake_route, monkeypatch, window, provenance
):
    """A retrieving scope row keeps its findings and its authority whatever its
    window evidence says: a small, unsourced or sentinel window neither demotes a
    critical finding nor removes the row's seat (BIBLE P3 — authority rests on
    the required-source manifest and its recorded coverage). The commit blocks
    here because the row reported a CRITICAL under blocking enforcement, which is
    the only reason a scope row ever blocks.
    """
    from ouroboros import config as cfg
    monkeypatch.setattr(cfg, "get_review_enforcement", lambda: "blocking")
    result = _run_session_scope(
        tmp_path, fake_route, monkeypatch, window=window, provenance=provenance,
        rows=_scope_matrix_with_critical(),
    )

    assert result.status == "responded", result.status
    assert result.blocked is True
    reasons = " ".join(str(f.get("reason") or "") for f in result.critical_findings)
    assert "[advisory-only session scope reviewer]" not in reasons
    assert "contradicts a documented invariant" in reasons
    items = {str(f.get("item") or "") for f in result.advisory_findings}
    assert "scope_review_session_window_unproven" not in items


def test_session_scope_with_sourced_window_evidence_keeps_blocking_authority(
    tmp_path, fake_route, monkeypatch
):
    """Sourced window evidence changes nothing either: the row's criticals gate
    the commit and it counts as an authoritative responder."""
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "get_review_enforcement", lambda: "blocking")
    for window in (200_000, 1_000_000):
        result = _run_session_scope(
            tmp_path, fake_route, monkeypatch, window=window, provenance="confirmed",
            rows=_scope_matrix_with_critical(),
        )
        assert result.status == "responded", (window, result.status)
        assert result.blocked is True, window
        assert result.critical_findings, window
        items = {str(f.get("item") or "") for f in result.advisory_findings}
        assert "scope_review_session_window_unproven" not in items, window


def test_api_scope_row_window_size_never_removes_its_authority(
    tmp_path, fake_route, monkeypatch
):
    """A bare api scope row is a retrieving reviewer, so its window sizes its
    working view and nothing else: a 200K route answers authoritatively and its
    criticals gate the commit (BIBLE P3 — window is not a condition of
    authority; the required-source manifest is)."""
    import ouroboros.tools.scope_review as scope_mod
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "get_review_enforcement", lambda: "blocking")
    monkeypatch.setattr(scope_mod, "_scope_window",
                        lambda *_a, **_k: ReviewerWindow(
                            window_tokens=200_000, status="confirmed"))
    monkeypatch.setattr(
        scope_mod, "_call_scope_llm",
        lambda *_a, **_k: (json.dumps(_scope_matrix_with_critical()), {}, ""),
    )
    result = scope_mod.run_scope_review(
        _scope_ctx(tmp_path), "api row, small window",
        scope_model="api/small-window", slot_id="scope_slot_1",
    )
    assert result.status == "responded", result.status
    assert result.blocked is True and result.critical_findings
    assert "does not establish the required >=1M floor" not in result.block_message


def test_scope_quorum_keeps_an_answered_row_with_incomplete_read_coverage(tmp_path, monkeypatch):
    """The same answer and findings count while its read coverage remains visible."""
    from ouroboros.tools import parallel_review, review
    from ouroboros.tools.scope_review import ScopeReviewResult

    rows = {
        "api/big": ScopeReviewResult(blocked=False, status="responded", model_id="api/big",
                                     coverage="complete"),
        "api/partial": ScopeReviewResult(
            blocked=False, status="responded", model_id="api/partial", coverage="incomplete",
            context_manifest={"native_read_coverage": {"status": "incomplete", "sources": [
                {"path": "prompts/SYSTEM.md", "status": "incomplete"},
            ]}},
            advisory_findings=[{
                "verdict": "FAIL", "severity": "advisory", "item": "architecture_fit",
                "reason": "a real observation the row still contributes",
            }],
        ),
    }
    monkeypatch.setattr(parallel_review, "run_scope_review",
                        lambda _ctx, _msg, **kwargs: rows[kwargs["scope_model"]])
    monkeypatch.setattr(parallel_review, "scope_reviewer_slots", lambda *_a, **_k: [
        SimpleNamespace(model="api/big", slot_id="scope_slot_1", route=None,
                        effort="", session_target="", session_profile="", retrieves=True),
        SimpleNamespace(model="api/partial", slot_id="scope_slot_2", route=None,
                        effort="", session_target="", session_profile="", retrieves=True),
    ])
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(review, "_prepare_unified_review", lambda *_a, **_k: (None, None, True))
    from ouroboros.tools import review_admission
    monkeypatch.setattr(review_admission, "prepare_scope_review",
                        lambda *_a, **_k: ({"brief": 1}, None))

    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="scope-quorum",
        pending_events=[], _review_history=[], _review_advisory=[], _scope_review_history={},
    )
    args = parallel_review.run_parallel_review(ctx, "quorum commit")
    blocked, message, _reason, _findings, advisory = parallel_review.aggregate_review_verdict(
        *args, ctx, "quorum commit", 0.0, tmp_path)
    assert blocked is False and message is None
    assert [row["item"] for row in advisory] == ["architecture_fit"]

    manifest = (ctx._last_scope_raw_result or {}).get("context_manifest") or {}
    # Both configured rows answered; read coverage does not withdraw a verdict.
    assert manifest["scope_responded_count"] == 2, manifest
    assert manifest["scope_coverage_incomplete_count"] == 1, manifest
    assert manifest["scope_degraded_reasons"] == [], manifest
    assert manifest["scope_coverage_diagnostics"][1]["uncovered_sources"] == ["prompts/SYSTEM.md"]
    rowed = {row["slot_id"]: row for row in ctx._last_scope_raw_results}
    assert rowed["scope_slot_2"]["status"] == "responded"
    assert rowed["scope_slot_2"]["failure_phase"] == ""
    # The reviewer keeps its findings, with no host-authored FAIL for read telemetry.
    assert rowed["scope_slot_2"]["advisory_findings"][0]["item"] == "architecture_fit"


def test_triad_mixed_panel_builds_the_pack_once_for_api_rows_only(tmp_path, fake_route, monkeypatch):
    """5.2/5.3 on the triad: one panel, two deliveries. The api row gets the
    historical pack; the session row gets the compact task; an all-session
    panel never assembles the pack at all."""
    import ouroboros.tools.review as review_mod
    from ouroboros.review_execution import ReviewRouteKind

    chat_calls = []

    class PanelLLM:
        def chat(self, **kwargs):
            chat_calls.append(kwargs)
            return {"content": "[]\nNO_FINDINGS"}, {"prompt_tokens": 4, "completion_tokens": 2}

    monkeypatch.setattr(review_mod, "LLMClient", PanelLLM)
    monkeypatch.setattr(review_mod, "review_drive_root", lambda _ctx: tmp_path)
    fake_route.detail = _terminal_detail('{"findings": []}', conformance="passed")

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
    assert rows[1]["slot_id"] == "slot_2" and rows[1]["text"] == "[]"
    assert len(chat_calls) == 1  # ONE api send; the session row never used chat
    # The session start carried the compact task, not the giant pack.
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
    assert result["results"][0]["text"] == "[]"
    assert chat_calls == []


def test_triad_session_task_carries_criteria_and_nav_maps_not_evidence():
    import ouroboros.tools.review as review_mod

    task = review_mod._triad_session_task(
        None,
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


# ---------------------------------------------------------------------------
# The blocking scope gate must not fail OPEN on an all-retrieving panel.
# ---------------------------------------------------------------------------


def _all_session_scope_panel(tmp_path, monkeypatch, *, window, provenance):
    """The REAL fan-out + aggregate over a panel of two retrieving rows.

    Only the two genuinely external things are faked: the reviewer's window
    evidence and the model call. Everything the gate actually decides with —
    `run_scope_review`, `run_parallel_review`'s quorum, `aggregate_review_verdict`
    — runs for real.
    """
    from ouroboros import config as cfg
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools import parallel_review, review
    import ouroboros.tools.scope_review as scope_mod

    if provenance:
        resolved = ReviewerWindow(window_tokens=int(window), status=provenance)
    else:
        resolved = ReviewerWindow(window_tokens=0, status="")
    monkeypatch.setattr(cfg, "get_review_enforcement", lambda: "blocking")
    monkeypatch.setattr(scope_mod, "_scope_window", lambda *_a, **_k: resolved)
    monkeypatch.setattr(
        scope_mod, "_call_scope_llm",
        lambda *_a, **_k: (json.dumps(_scope_matrix_rows()), {}, ""),
    )
    monkeypatch.setattr(parallel_review, "scope_reviewer_slots", lambda *_a, **_k: [
        ReviewSlot(slot_id="scope_slot_1", model="codex=gpt-5.6-sol",
                   route=ReviewRouteKind.AGENT_SESSION, session_target="codex=gpt-5.6-sol"),
        ReviewSlot(slot_id="scope_slot_2", model="claude=fable-5",
                   route=ReviewRouteKind.AGENT_SESSION, session_target="claude=fable-5"),
    ])
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(review, "_prepare_unified_review", lambda *_a, **_k: (None, None, True))

    ctx = _scope_ctx(tmp_path)
    ctx._review_history = []
    ctx._review_advisory = []
    ctx._scope_review_history = {}
    ctx.task_id = "scope-fail-open"
    ctx.pending_events = []
    args = parallel_review.run_parallel_review(ctx, "all-retrieving scope panel")
    blocked, message, reason, _findings, _advisory = parallel_review.aggregate_review_verdict(
        *args, ctx, "all-retrieving scope panel", 0.0, tmp_path,
    )
    manifest = (ctx._last_scope_raw_result or {}).get("context_manifest") or {}
    return blocked, message or "", reason, manifest


def test_all_retrieving_scope_panel_uses_findings_without_window_authority_gate(tmp_path, monkeypatch):
    """A scope panel of retrieving rows with no window evidence at all answers
    authoritatively: authority rests on the required-source manifest and the
    recorded coverage, so an unknown window neither arms nor disarms the gate.

    The asymmetry this replaced was the fail-open measured on a6a3c1f, where the
    same panel shape gave a BLOCKING api row and a non-blocking retrieving one.
    """
    blocked, message, reason, manifest = _all_session_scope_panel(
        tmp_path, monkeypatch, window=0, provenance="",
    )

    assert blocked is False
    assert manifest["scope_responded_count"] == 2, manifest
    assert manifest["scope_coverage_incomplete_count"] == 0, manifest


def test_retrieving_and_api_panels_agree_on_an_unestablished_window(tmp_path, monkeypatch):
    """The asymmetry itself was the defect, and it is gone in both directions:
    both scope deliveries retrieve, so an unestablished or small window neither
    grants nor removes authority on either of them."""
    import ouroboros.tools.scope_review as scope_mod

    blocked, _msg, _reason, manifest = _all_session_scope_panel(
        tmp_path, monkeypatch, window=200_000, provenance="confirmed",
    )
    assert blocked is False
    assert manifest["scope_responded_count"] == 2, manifest

    # The api row's twin: same small window, same authoritative outcome.
    monkeypatch.setattr(scope_mod, "_scope_window",
                        lambda *_a, **_k: ReviewerWindow(
                            window_tokens=200_000, status="confirmed"))
    monkeypatch.setattr(
        scope_mod, "_call_scope_llm",
        lambda *_a, **_k: (json.dumps(_scope_matrix_rows()), {}, ""),
    )
    api_result = scope_mod.run_scope_review(
        _scope_ctx(tmp_path), "api row, small window", scope_model="api/small",
        slot_id="scope_slot_1",
    )
    assert api_result.blocked is False and api_result.status == "responded"


def test_session_schema_floor_matches_each_surfaces_clean_contract():
    """`{"findings": []}` is the honest clean verdict for a TRIAD session, but on
    scope and Skill Review (mandatory matrix rows) it is schema-conformant but
    downstream-invalid. The floor lets a conforming engine regenerate instead."""
    from ouroboros.review_execution import (
        REVIEW_SESSION_OUTPUT_SCHEMA,
        review_session_output_schema,
    )

    assert review_session_output_schema("commit_review") is REVIEW_SESSION_OUTPUT_SCHEMA
    # Advisory keeps the clean-capable shared schema: its ORDINARY mode's required
    # clean verdict is exactly the empty array, so a floor would starve it of the
    # one answer its contract demands (checklist coverage is checked downstream).
    assert review_session_output_schema("advisory_review") is REVIEW_SESSION_OUTPUT_SCHEMA
    assert "minItems" not in REVIEW_SESSION_OUTPUT_SCHEMA["properties"]["findings"]
    shaped = review_session_output_schema("scope_review")
    assert shaped["properties"]["findings"]["minItems"] == 1
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
    ctx = _scope_ctx(tmp_path)
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

    # ABI-10: the session row + its credential pin are configured through the
    # structured slots (the phase-5 route envs are retired and ignored).
    monkeypatch.delenv(REVIEW_SESSION_ROUTE_ENV, raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1",
                   "route": {"kind": "agent_session",
                             "target_id": "fake-review=fake-small",
                             "profile_id": "legacy-profile"},
                   "effort": "high"}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "m/scope"}}],
    }))
    delivery = commit_triad_delivery()
    assert delivery["session_profiles"] == ["legacy-profile"]
    fake_route.detail = _terminal_detail(json.dumps({"findings": [
        {"item": "manifest_schema", "verdict": "PASS", "severity": "advisory",
         "reason": "checked"},
    ]}), conformance="passed")
    no_api = FakeLLM()
    monkeypatch.setattr(review_tool, "LLMClient", lambda: no_api)
    ctx = _scope_ctx(tmp_path)

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


def test_scope_book_navigation_uses_physical_chapter_sources(tmp_path):
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
        checklist_section_text="(scope checklist)").navigation
    assert "docs/architecture/runtime.md" in text
    assert "Processes carry the work" in text          # the heading is indexed
    assert "The full startup mechanism" not in text    # the body is not
    assert 'root="system_repo"' in text
    (tmp_path / "docs/architecture/runtime.md").unlink()
    broken = governance_context(
        tmp_path, surface="scope", touched_paths=(), delivery="retrieving",
        checklist_section_text="(scope checklist)")
    assert "ARCHITECTURE.md" in broken.navigation
    row = next(r for r in broken.manifest if r["path"] == "docs/ARCHITECTURE.md")
    assert row["disposition"] == "navigation"
    assert "runtime.md" in row["reason"]


# ---------------------------------------------------------------------------
# The brief a scope reviewer receives: intent and manifests, the repository
# index, the governance tiers, and the staged diff inline or paged (D2v2).
# ---------------------------------------------------------------------------


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


DIFF_MARKER = "UNIQUE_DIFF_BODY_MARKER"
BRIEF_TASK_ID = "scope-brief-task"


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
    from ouroboros.tools.scope_required_sources import (
        required_sources_ref, scope_required_sources, staged_touched_paths,
        staged_tree_identity, touched_manifest,
    )
    from ouroboros.tools.scope_review_session import ScopeBriefInputs, ScopeIntentContext

    touched = staged_touched_paths(repo)
    tree = staged_tree_identity(repo)
    rows = scope_required_sources(repo, touched, staged_tree_sha=tree)
    fields = dict(
        commit_message="fix: the scope brief carries what the reviewer needs",
        intent=ScopeIntentContext(goal="Deliver the brief", scope="Only the scope surface"),
        touched_paths=tuple(path for _status, path in touched),
        touched_manifest=touched_manifest(repo, touched),
        required_sources=rows,
        required_sources_ref=required_sources_ref(rows, staged_tree_sha=tree),
        scope_model="api/scope-model",
        slot_id="scope_slot_1",
        task_id=BRIEF_TASK_ID,
        source_root=str(drive),
    )
    fields.update(overrides)
    return ScopeBriefInputs(**fields)


def test_the_brief_carries_the_index_the_governance_tiers_and_both_manifests(tmp_path):
    """One brief, four deliveries of context: the repository index (no bodies),
    the governance tiers with tier 1 ahead of every change-relative section, the
    navigation with its exact read instruction, and the two manifests. Each of
    them is recorded in the pre-run disclosure manifest as well as rendered."""
    from ouroboros.tools.review_helpers import REPO_ROOT
    from ouroboros.tools.scope_review_session import build_scope_session_task

    repo = _staged_subject(tmp_path)
    drive = tmp_path / "data"
    drive.mkdir()
    task, manifest = build_scope_session_task(
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
    assert "## docs/CHECKLISTS_ARCHIVE.md" in task
    head = task.index("## BIBLE.md")
    for change_relative in ("## Intended transformation", "## Staged diff",
                            "## Repository index", "TOUCHED PATHS", "REQUIRED SOURCES"):
        assert head < task.index(change_relative), change_relative
    tiers = {row["path"]: row for row in manifest["governance_manifest"]}
    assert tiers["BIBLE.md"]["tier"] == 1 and tiers["BIBLE.md"]["disposition"] == "inline"
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


def test_a_staged_diff_that_fits_the_first_send_is_inlined(tmp_path):
    """The reviewer reads the change itself, not a pointer to it, whenever the
    whole first send lands under the row's own bound."""
    from ouroboros.tools.scope_review_session import build_scope_session_task

    repo = _staged_subject(tmp_path)
    drive = tmp_path / "data"
    drive.mkdir()
    task, manifest = build_scope_session_task(repo, _brief_inputs(repo, drive))

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
    is stored ONCE, byte-exactly, at an address the row's own reader reaches —
    the task artifact store for a native episode, the review's git-ignored
    project view for a delegated session — and the brief carries the address,
    the size and the digest instead of the body."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.review_admission import prepare_scope_review
    from ouroboros.tools.review_binary_context import capture_staged_diff
    from ouroboros.tools.scope_required_sources import required_sources_ref, source_text_identity
    from ouroboros.tools.registry import ToolContext

    repo = _staged_subject(tmp_path, payload_chars=900_000, newline=newline)
    drive = tmp_path / "data"
    drive.mkdir()
    expected = capture_staged_diff(repo)
    assert len(expected) > 900_000

    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=BRIEF_TASK_ID)
    prepared, final = prepare_scope_review(
        ctx, "review a paged subject", scope_model="fixture/model", slot_id="scope_slot_1",
        route=ReviewRouteKind.AGENT_SESSION if delegated else ReviewRouteKind.API_CHAT)
    assert final is None
    task, manifest = prepared["session_task"], prepared["context_manifest"]

    assert manifest["diff_delivery"] == "paged", manifest.get("diff_paging_reason")
    assert manifest["first_send_chars"] < manifest["first_send_ceiling"]
    assert manifest["diff_chars"] == len(expected)
    # The body is NOT in the brief; its address, size and digest are.
    assert DIFF_MARKER not in task
    assert "PAYLOAD_LINE_00000001" not in task
    source = manifest["diff_source"]
    rows = prepared["required_sources"]
    diff_row = next(row for row in rows if row["disposition"] == "review_subject")
    assert diff_row == source["required_row"]
    assert all(diff_row[key] == value for key, value in source_text_identity(expected.encode()).items())
    assert diff_row["candidate_tree"] == prepared["required_sources_ref"]["staged_tree_sha"]
    assert prepared["native_data_root"] == str(drive)
    assert manifest["native_required_sources"] == rows
    assert prepared["required_sources_ref"] == manifest["native_required_sources_ref"] == required_sources_ref(
        rows, staged_tree_sha=diff_row["candidate_tree"])
    assert prepared["required_sources_ref"]["required_source_count"] == 2
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
    tmp_path, monkeypatch, delegated, managed, git_autocrlf,
):
    """Rename detection can omit the body; the exact old prompt stays readable
    and has its own diagnostic row on both transports."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.review_admission import prepare_scope_review
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
        monkeypatch.setattr("ouroboros.tools.review_subject.managed_review_subject", lambda *_: subject)
    drive = tmp_path / "data"
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=BRIEF_TASK_ID)
    prepared, final = prepare_scope_review(
        ctx, "rename a prompt", scope_model="fixture/model", slot_id="scope_slot_1",
        route=ReviewRouteKind.AGENT_SESSION if delegated else ReviewRouteKind.API_CHAT)
    assert final is None
    manifest = prepared["context_manifest"]
    assert manifest["diff_delivery"] == "inline"
    assert prepared["native_data_root"] == str(drive)
    assert len(prepared["required_sources"]) == 1
    row = prepared["required_sources"][0]
    assert row["disposition"] == "deleted_preimage" and row["preimage_of"] == "prompts/SYSTEM.md"
    assert row["preimage"] == f"{baseline if managed else 'HEAD'}:prompts/SYSTEM.md"
    assert row["candidate_tree"] == prepared["required_sources_ref"]["staged_tree_sha"]
    source = manifest["preimage_sources"][0]
    assert read_actor_source_bytes(drive, BRIEF_TASK_ID, source) == blob
    assert row["source_revision"] == hashlib.sha256(blob).hexdigest()
    assert row["path"] in prepared["session_task"] and "preimage of prompts/SYSTEM.md" in prepared["session_task"]
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
    from ouroboros.tools.scope_review_session import build_scope_session_task
    from ouroboros.tools.scope_required_sources import required_sources_ref

    repo = _staged_subject(tmp_path)
    drive = tmp_path / "data"
    drive.mkdir()
    # The first case has a real source and no store; the second has a store
    # but names a missing baseline source. Neither claims a delivered preimage.
    row = {"root": "active_workspace", "path": "prompts/REMOVED.md", "disposition": "deleted",
           "coverage_basis": "preimage_unavailable",
           "preimage": "HEAD:absent.md" if available_store else "HEAD:alpha.py"}
    task, manifest = build_scope_session_task(repo, _brief_inputs(
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
    from ouroboros.tools.scope_review_session import build_scope_session_task

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
    task, manifest = build_scope_session_task(
        repo, _brief_inputs(repo, drive, governance_repo_dir=governance))
    row = next(row for row in manifest["native_required_sources"] if row["path"] == "BIBLE.md")
    assert row["coverage_basis"] == ("delivered_inline" if matching_governance else "candidate_blob")
    if matching_governance:
        assert "Candidate constitution." in task
        assert "delivered inline in full, no second read needed" in task


def test_the_brief_of_a_three_file_change_on_the_real_tree_is_measured(tmp_path):
    """The brief's size on the REAL repository, without the staged diff, with
    its composition printed. The number is the whole reason the packet is gone:
    the retired scope packet's fixed part alone was 1.21 MB.

    The ceiling is measured, not aspirational — it is the sum of the owner's own
    decisions: tier-1 BIBLE plus the standing disclosures inline (~65k chars),
    the repository index over ~2,400 tracked paths (~49k), the DEVELOPMENT
    chapters this change activates (~31k) and the book navigation (~28k).
    """
    from ouroboros.tools.review_helpers import REPO_ROOT
    from ouroboros.tools.scope_review_session import ScopeBriefInputs, build_scope_session_task
    from ouroboros.tools.scope_required_sources import (
        required_sources_ref, scope_required_sources, touched_manifest,
    )

    touched = [("M", "ouroboros/tools/scope_review_session.py"),
               ("M", "ouroboros/tools/review_admission.py"),
               ("M", "ouroboros/tools/scope_review.py")]
    rows = scope_required_sources(REPO_ROOT, touched)
    task, manifest = build_scope_session_task(REPO_ROOT, ScopeBriefInputs(
        commit_message="scope review by retrieval",
        touched_paths=tuple(path for _status, path in touched),
        touched_manifest=touched_manifest(REPO_ROOT, touched),
        required_sources=rows,
        required_sources_ref=required_sources_ref(rows),
        scope_model="api/scope-model", slot_id="scope_slot_1",
    ))
    sections = manifest["brief_sections"]
    without_diff = len(task) - sections["diff_slot"]
    print(f"\nscope brief on the real tree: {len(task):,} chars "
          f"({without_diff:,} without the diff slot)")
    for name, chars in sorted(sections.items(), key=lambda item: -item[1]):
        print(f"  {name:32s} {chars:>9,}")
    # 200_000 -> 205_000 (PR #940 rework, 2026-10-03): the base sat 285 chars under the
    # ceiling, so the repository index alone (+~1k chars: the interface-language modules,
    # their tests and the per-chapter grant files) crossed it; still a measured sum, raised
    # by what those files are, not by aspiration.
    # 205_000 -> 210_000 (checklist layers, 2026-10-07): the base measured 203,286; the
    # split of the commit checklist grew the tier-1 inline (the archive's renumbering
    # note and the shared-section intro, +1.5k) and the DEVELOPMENT governance chapter
    # that names scope_review.py (+1.9k), to 206,750 — the same measured sum, re-read.
    # Re-measured on the assembled subject-operation tree (2026-10-07): 209,106 — the
    # repository index grew to 56,982 over the review subject / body-fact / operation
    # modules, their tests and chapters, the selected governance inline to 35,084. 894
    # chars under the ceiling: the next index growth must re-read this sum, not round it.
    # 210_000 -> 210_161 (engine facts/lifecycle, 2026-10-08): the d00559d7f
    # membership renders 57,064 index chars; the new module, smoke script and
    # grant directory add exactly 161 (57,225). The full brief measures 210,080;
    # retain the base's 81-char slack, without rounding up the measured growth.
    assert without_diff < 210_161, without_diff
    assert sections["repository_index"] > 20_000          # the index really ran
    assert sections["governance_stable_inline"] > 40_000  # BIBLE really inline
