"""Surface wiring: scope and triad deliver sessions without packs.

Split by theme out of ``tests/test_review_agent_session_route.py``. This module
owns the scope/triad surface wiring: session rows never build the API pack, the
mixed fanout keeps one route per row, sourced window evidence alone carries
blocking authority, and an all-retrieving scope panel blocks instead of failing
open.
"""

import json
from types import SimpleNamespace

import pytest

from ouroboros.review_execution import (
    SCOPE_REVIEW_ROUTES_ENV,
)
from ouroboros.review_substrate import (
    scope_reviewer_slots,
)

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport
from tests._review_session_route_shared import fake_route as __fake_route

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport
fake_route = __fake_route

from tests._review_session_route_shared import (
    _terminal_detail,
)

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
    one-element list always re-reads ROUTES **row 1** — so with
    `agent_session,api_chat` the configured api row inherited agent_session while
    its request carried the api pack and no session task: a deterministic
    ReviewRouteUnavailable error actor that failed the blocking scope gate.
    """
    import ouroboros.tools.scope_review as scope_mod

    monkeypatch.setenv(SCOPE_REVIEW_ROUTES_ENV, "agent_session,api_chat")
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
    monkeypatch.setattr(scope_mod, "_build_scope_prompt",
                        lambda *_a, **_k: ("assembled api pack", None))
    monkeypatch.setattr(scope_mod, "_scope_window",
                        lambda *_a, **_k: scope_mod.ReviewerWindow(
                            window_tokens=1_000_000, status="confirmed"))

    for slot in scope_reviewer_slots(["m/session", "m/api"]):
        scope_mod.run_scope_review(
            _scope_ctx(tmp_path), "mixed route fan-out",
            scope_model=slot.model, slot_id=slot.slot_id, route=slot.route,
        )

    # Row 1 is the session (task, no api pack); row 2 is api (pack, no task).
    assert dispatched == [
        ("scope_slot_1", "m/session", "agent_session", True, False),
        ("scope_slot_2", "m/api", "api_chat", False, True),
    ], dispatched

def test_scope_session_delivery_never_builds_the_pack(tmp_path, fake_route, monkeypatch):
    """5.2 on scope: a delegated scope row goes out as a compact session task —
    checklist, contract and intent context intact (5.3), retrieval pointers and
    nav maps (5.7) instead of the assembled diff/touched/atlas pack — and the
    coverage manifest is forensics, not a gate (5.6): host_file_read_attestation rides
    as a non-blocking fact on a run that PASSES.

    The row is given SOURCED window evidence so it clears the session authority
    floor: coverage is then the only thing that could possibly gate it — and does
    not. (Authority itself is covered by the session-floor tests below.)"""
    import ouroboros.tools.scope_review as scope_mod
    from ouroboros.review_execution import ReviewRouteKind

    def _pack_must_not_build(*_a, **_k):  # pragma: no cover - the point is silence
        raise AssertionError("the api pack builder ran for a session slot")

    monkeypatch.setattr(scope_mod, "_build_scope_prompt", _pack_must_not_build)
    monkeypatch.setattr(scope_mod, "_scope_window",
                        lambda *_a, **_k: scope_mod.ReviewerWindow(
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
    assert manifest["host_file_read_attestation"] == "unobserved"  # forensic, non-blocking
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
    assert "session delivery" in prompt          # retrieval pointers, not packs
    assert "git diff --cached" in prompt
    assert "navigation map" in prompt            # 5.7: atlas as a map
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
        _resolved = scope_mod.ReviewerWindow(window_tokens=int(window), status=provenance)
    elif provenance == "designated_default_sentinel":
        _resolved = scope_mod.ReviewerWindow(window_tokens=int(window), status="")
    else:
        _resolved = scope_mod.ReviewerWindow(window_tokens=0, status="")
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
def test_session_scope_without_sourced_window_evidence_is_advisory_only(
    tmp_path, fake_route, monkeypatch, window, provenance
):
    """A retrieving scope row's FINDINGS certify nothing without SOURCED window
    evidence >= the session floor — and the row BLOCKS, as its api twin does.

    The previous shape skipped `_apply_scope_authority` for `agent_session`
    entirely — a session verdict gated commits with NO window test at all, while
    its own manifest recorded host_file_read_attestation/host_enforced=False.

    Two facts live in one result and are easy to conflate: the findings are
    demoted to ADVISORY (an unestablished window cannot certify a verdict), and
    the commit is BLOCKED (the panel is short an authoritative verdict). Returning
    `blocked=False` here is what made the P3 gate fail open — the api row's
    `sub_floor` twin blocked on the identical panel shape.
    """
    result = _run_session_scope(
        tmp_path, fake_route, monkeypatch, window=window, provenance=provenance,
        rows=_scope_matrix_with_critical(),
    )

    assert result.status == "session_advisory", result.status
    assert result.blocked is True
    assert "authoritative scope verdict required to commit" in result.block_message
    # The critical was preserved as advisory evidence, not discarded...
    assert result.critical_findings == []
    reasons = " ".join(str(f.get("reason") or "") for f in result.advisory_findings)
    assert "[advisory-only session scope reviewer]" in reasons
    assert "contradicts a documented invariant" in reasons
    # ...and the reason it cannot gate is disclosed on the record.
    items = {str(f.get("item") or "") for f in result.advisory_findings}
    assert "scope_review_session_window_unproven" in items, items
    assert "SCOPE_SESSION_ADVISORY_ONLY" in reasons


def test_session_scope_with_sourced_window_evidence_keeps_blocking_authority(
    tmp_path, fake_route, monkeypatch
):
    """Sourced evidence at or above the session floor IS authority: the row's
    criticals gate the commit and it counts as an authoritative responder."""
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


def test_api_scope_row_keeps_the_1m_floor_and_still_blocks_sub_floor(
    tmp_path, fake_route, monkeypatch
):
    """The api (push) delivery is untouched: its authority rests on the assembled
    pack fitting, so a sub-1M reviewer is still the loud `sub_floor` block."""
    import ouroboros.tools.scope_review as scope_mod

    monkeypatch.setattr(scope_mod, "_scope_window",
                        lambda *_a, **_k: scope_mod.ReviewerWindow(
                            window_tokens=200_000, status="confirmed"))
    monkeypatch.setattr(scope_mod, "_build_scope_prompt",
                        lambda *_a, **_k: ("assembled api pack", None))
    monkeypatch.setattr(
        scope_mod, "_call_scope_llm",
        lambda *_a, **_k: (json.dumps(_scope_matrix_with_critical()), {}, ""),
    )
    result = scope_mod.run_scope_review(
        _scope_ctx(tmp_path), "api row, sub-floor window",
        scope_model="api/small-window", slot_id="scope_slot_1",
    )
    assert result.status == "sub_floor", result.status
    assert result.blocked is True
    assert "does not establish the required >=1M floor" in result.block_message


def test_scope_quorum_refuses_a_session_advisory_row_as_authoritative(tmp_path, monkeypatch):
    """The scope quorum must not count a non-host-attested session row as the
    authoritative verdict, and must disclose the shortfall it leaves."""
    from ouroboros.tools import parallel_review, review
    from ouroboros.tools.scope_review import ScopeReviewResult

    rows = {
        "api/big": ScopeReviewResult(blocked=False, status="responded", model_id="api/big"),
        "session/row": ScopeReviewResult(
            blocked=False, status="session_advisory", model_id="session/row",
            advisory_findings=[{
                "verdict": "FAIL", "severity": "advisory",
                "item": "scope_review_session_window_unproven",
                "reason": "SCOPE_SESSION_ADVISORY_ONLY: window not sourced-proven",
            }],
        ),
    }
    monkeypatch.setattr(parallel_review, "run_scope_review",
                        lambda _ctx, _msg, **kwargs: rows[kwargs["scope_model"]])
    monkeypatch.setattr(parallel_review, "scope_reviewer_slots", lambda *_a, **_k: [
        SimpleNamespace(model="api/big", slot_id="scope_slot_1", route=None,
                        effort="", session_target="", session_profile=""),
        SimpleNamespace(model="session/row", slot_id="scope_slot_2", route=None,
                        effort="", session_target="", session_profile=""),
    ])
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(review, "_run_unified_review", lambda *_a, **_k: None)

    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="scope-quorum",
        pending_events=[], _review_history=[], _review_advisory=[], _scope_review_history={},
    )
    parallel_review.run_parallel_review(ctx, "quorum commit")

    manifest = (ctx._last_scope_raw_result or {}).get("context_manifest") or {}
    # Two configured rows, adaptive quorum 2 — but only ONE authoritative verdict.
    assert manifest["scope_responded_count"] == 1, manifest
    assert manifest["scope_session_advisory_only_count"] == 1, manifest
    assert any("scope_session_advisory_only" in str(r)
               for r in manifest["scope_degraded_reasons"]), manifest


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
    assert "Read BIBLE.md in full" in task


# ---------------------------------------------------------------------------
# The blocking scope gate must not fail OPEN on an all-retrieving panel.
# ---------------------------------------------------------------------------


def _all_session_scope_panel(tmp_path, monkeypatch, *, window, provenance):
    """The REAL fan-out + aggregate over a panel of two retrieving rows.

    Only the two genuinely external things are faked: the reviewer's window
    evidence and the model call. Everything the gate actually decides with —
    `run_scope_review`, `_apply_scope_authority`, `session_scope_authority`,
    `run_parallel_review`'s quorum, `aggregate_review_verdict` — runs for real.
    """
    from ouroboros import config as cfg
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools import parallel_review, review
    import ouroboros.tools.scope_review as scope_mod

    if provenance:
        resolved = scope_mod.ReviewerWindow(window_tokens=int(window), status=provenance)
    else:
        resolved = scope_mod.ReviewerWindow(window_tokens=0, status="")
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
    monkeypatch.setattr(review, "_run_unified_review", lambda *_a, **_k: None)

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


def test_all_retrieving_scope_panel_blocks_instead_of_failing_open(tmp_path, monkeypatch):
    """A scope panel of retrieving rows with no sourced window evidence yields ZERO
    authoritative verdicts — and must BLOCK, exactly as the api panel does.

    This is the fail-open the adversarial panel measured on a6a3c1f: the same panel
    shape gave `api_chat status=sub_floor -> BLOCKED=True` and
    `agent_session status=session_advisory -> BLOCKED=False`. Nothing downstream
    could recover it — `partial_quorum_shortfall` only fires above zero responders,
    so a zero-authoritative run walked straight through the blocking scope gate of
    BIBLE P3 while looking armed.
    """
    blocked, message, reason, manifest = _all_session_scope_panel(
        tmp_path, monkeypatch, window=0, provenance="",
    )

    assert blocked is True, "the blocking scope gate must not pass a zero-authoritative panel"
    assert reason == "scope_blocked", reason
    assert "SCOPE_REVIEW_BLOCKED" in message
    assert "authoritative scope verdict required to commit" in message
    # The shortfall is still disclosed, not merely converted into a block.
    assert manifest["scope_responded_count"] == 0, manifest
    assert manifest["scope_session_advisory_only_count"] == 2, manifest
    assert any("scope_session_advisory_only" in str(r)
               for r in manifest["scope_degraded_reasons"]), manifest


def test_retrieving_and_api_panels_agree_on_an_unestablished_window(tmp_path, monkeypatch):
    """The asymmetry itself is the defect: an unestablished window blocks on BOTH
    deliveries, and SOURCED evidence at the row's own floor authorises on both."""
    import ouroboros.tools.scope_review as scope_mod

    # Retrieving row, SOURCED at the session floor -> authoritative, no block.
    blocked, _msg, _reason, manifest = _all_session_scope_panel(
        tmp_path, monkeypatch, window=200_000, provenance="confirmed",
    )
    assert blocked is False, "sourced >=200K evidence must restore an authoritative verdict"
    assert manifest["scope_responded_count"] == 2, manifest

    # api row, window below its own floor -> blocks (the twin, unchanged).
    monkeypatch.setattr(scope_mod, "_build_scope_prompt",
                        lambda *_a, **_k: ("assembled api pack", None))
    monkeypatch.setattr(scope_mod, "_scope_window",
                        lambda *_a, **_k: scope_mod.ReviewerWindow(
                            window_tokens=200_000, status="confirmed"))
    monkeypatch.setattr(
        scope_mod, "_call_scope_llm",
        lambda *_a, **_k: (json.dumps(_scope_matrix_rows()), {}, ""),
    )
    api_result = scope_mod.run_scope_review(
        _scope_ctx(tmp_path), "api row, sub-floor window", scope_model="api/small",
        slot_id="scope_slot_1",
    )
    assert api_result.blocked is True and api_result.status == "sub_floor"


def test_a_retrieving_row_can_actually_reach_sourced_evidence(tmp_path, monkeypatch):
    """The >=200K floor must be REACHABLE, not decorative.

    Retrieving rows were excluded from Capability-Evidence probing and their opaque
    `harness[=model]` target does not resolve through `provider_for_model`, so no
    product path could ever take such a row to `confirmed`/`asserted`: advisory-only
    was the mode's ONLY possible outcome. The settings save now offers the row its
    ack against its own floor, and acking that exact route restores authority.
    """
    from ouroboros import capability_evidence as ce
    from ouroboros.gateway import settings as smod
    from ouroboros.reviewer_window import SESSION_ROUTE_PROVIDER
    from ouroboros.tools.scope_review_session import (
        SESSION_WINDOW_FLOOR,
        session_window_is_authoritative,
    )
    from ouroboros.tools.scope_window import scope_window

    monkeypatch.setattr(ce, "DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(smod, "_candidate_scope_models", lambda _s: [])
    slots = json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "api/m"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "agent_session",
                                              "target_id": "codex=gpt-5.6-sol"}}],
    })

    notices = smod._review_capability_notices({"OUROBOROS_REVIEWER_SLOTS": slots})
    assert len(notices) == 1, notices
    notice = notices[0]
    assert notice["surface"] == "scope_review_session"
    assert notice["floor_tokens"] == SESSION_WINDOW_FLOOR
    ack_route = notice["needs_ack"]
    assert ack_route["provider"] == SESSION_ROUTE_PROVIDER, ack_route
    assert ack_route["model"] == "codex=gpt-5.6-sol"

    # Before the ack the row cannot authorise...
    before = scope_window("codex=gpt-5.6-sol", session=True)
    assert session_window_is_authoritative(before.window_tokens, before.status) is False

    # ...and the ack the UI records against that exact route is what restores it.
    ce.record_owner_ack(tmp_path, provider=ack_route["provider"], model=ack_route["model"],
                        base_url=ack_route["base_url"], window_tokens=SESSION_WINDOW_FLOOR)
    after = scope_window("codex=gpt-5.6-sol", session=True)
    assert session_window_is_authoritative(after.window_tokens, after.status) is True


def test_session_schema_floor_matches_each_surfaces_clean_contract():
    """`{"findings": []}` is the honest clean verdict for a TRIAD session, but on
    scope (eight mandatory rows) and advisory (empty checklist rejected by design)
    it is a schema-conformant answer that can only land as parse_failure and block
    the commit. The floor rides the schema so a conforming engine refuses the empty
    answer up front while the session can still regenerate."""
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
    # A shaped copy, never a mutation of the shared schema.
    assert "minItems" not in REVIEW_SESSION_OUTPUT_SCHEMA["properties"]["findings"]
