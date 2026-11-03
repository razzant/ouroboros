"""v6.78.0 (phase P4) — what outlives the host acceptance-decision state machine.

The decision writer, its terminal branches, the improvement capsule and the acceptance
packet are retired with task acceptance: nothing writes ``acceptance_decision`` now. What
stays: the owner-follow-up release of the root admission seal, the forced-rail record of a
budget rejection before any work, the readers of OLD acceptance decisions (their projection
and the review axis, which still render stored task cards), the advisory red-receipt nudge,
the one capability-omission formatter and the host's ``retrieval`` fact. The PURE receipt
identity/reconciliation core those nudges consult lives in
`test_v678_receipt_reconciliation.py`.

Table-driven and offline: no live reviewer panel, no ambient checkout state.
"""

from __future__ import annotations

from types import SimpleNamespace

import ouroboros.loop as loop_mod
from ouroboros.outcomes import (
    ACCEPTANCE_FINALIZED_UNACCEPTED,
    derive_loop_outcome,
)


# ---------------------------------------------------------------------------
# The root admission seal's owner-follow-up release


def test_an_owner_followup_releases_the_seal_without_writing_a_decision():
    """The owner-follow-up release (``delivery_seal._release_for_owner_followup``, which
    replaced the acceptance supersede): the selected completion is void, the fence ends as a
    revision, the generation-mismatch flag that caused it is cleared, and no acceptance
    decision is written."""
    ends: list = []
    ctx = SimpleNamespace(
        _task_acceptance_fence_token="tok",
        _task_acceptance_owner_generation=0,
        owner_message_admission_agent=SimpleNamespace(_owner_message_generation=1),
        end_acceptance_fence=lambda **kwargs: ends.append(kwargs) or {"ok": True, "status": "released"},
    )
    trace = {"review_runs": [], "task_completion": {"action": "finish", "answer_sha256": "a" * 64}}
    released = loop_mod._release_for_owner_followup(ctx, trace)
    assert released and released.status == "ok"
    assert "task_completion" not in trace and "acceptance_decision" not in trace
    assert [call["outcome"] for call in ends] == ["revision"]
    assert ctx._task_acceptance_fence_generation_mismatch is False


# ---------------------------------------------------------------------------
# Forced rails and the readers of old acceptance decisions


def test_round_one_budget_rejection_is_also_a_covered_forced_sink(tmp_path):
    """The total-budget rejection at round<=1 returns its notice directly instead
    of going through `_forced_final_answer`; the common forced-finalization record
    still says what happened. Nothing was produced, so the record is the whole
    remedy, and it says no candidate existed instead of inventing one; no acceptance
    decision is written."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    limit_ctx.round_idx = 1

    result = loop._check_budget_limits(limit_ctx, 0.0)

    assert result is not None
    text, usage, trace = result
    assert text.startswith("🚫 Task rejected.")
    assert usage["reason_code"] == "budget_exhausted"
    assert "acceptance_decision" not in trace
    # No candidate existed — the record says so instead of inventing one.
    assert trace["forced_finalization"]["candidate_sha256"] == ""
    assert trace["forced_finalization"]["source"] == "host_budget_rejection_before_work"

    axes = derive_loop_outcome(text, usage, trace)["outcome_axes"]
    assert axes["review"]["eligibility"] == "not_eligible"
    assert axes["review"]["run_count"] == 0


def test_acceptance_projection_carries_the_typed_reason():
    from ouroboros.outcomes import _acceptance_decision_projection

    out = _acceptance_decision_projection({
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "review_degraded",
        "source": "task_acceptance_review",
        "rationale": "no quorum",
        "open_obligations": ["ob-1"],
    })
    assert out["status"] == ACCEPTANCE_FINALIZED_UNACCEPTED
    assert out["reason"] == "review_degraded"
    assert out["open_obligations"] == ["ob-1"]
    # Historical records (pre-collapse tokens, no reason) still project verbatim:
    # the projection is a passthrough and no normalizer was introduced.
    legacy = _acceptance_decision_projection({"status": "best_effort_open_obligations"})
    assert legacy["status"] == "best_effort_open_obligations"
    assert legacy["reason"] == ""


# ---------------------------------------------------------------------------
# P4.3 — the reconciliation narrowing stays ADVISORY


def test_semantically_different_green_yields_one_advisory_nudge_and_no_gate(monkeypatch, tmp_path):
    """The panel's required case: a red receipt plus a later, semantically different
    passing receipt (`pytest tests/x.py` -> `pytest tests/x.py -v`) produces AT MOST
    ONE advisory nudge, and the forced finalization path is untouched (it returns
    before the injector runs)."""
    from ouroboros.outcomes import append_verification_receipt

    drive_root = tmp_path
    append_verification_receipt(drive_root, "t", {"status": "fail", "check": "pytest tests/x.py"})
    append_verification_receipt(drive_root, "t", {"status": "pass", "check": "pytest tests/x.py -v"})

    tool_ctx = SimpleNamespace(drive_root=str(tmp_path))
    tools = SimpleNamespace(_ctx=tool_ctx)
    trace = {"reasoning_notes": [], "tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages: list = []
    monkeypatch.setattr(loop_mod, "_skill_finalization_message", lambda *_a, **_k: "")

    fired = [
        loop_mod._maybe_inject_finalization_nudges(
            tools, tmp_path, "t", trace, "answer", messages, lambda _m: None,
        )
        for _ in range(4)
    ]
    assert fired[0] is True                     # one advisory nudge
    assert fired.count(True) == 1               # ...exactly one, latched
    nudges = [m for m in messages if "RED" in str(m.get("content") or "")]
    assert len(nudges) == 1
    assert "advisory" in str(nudges[0]["content"])


def test_red_receipt_reconciled_by_the_same_check_raises_no_nudge(monkeypatch, tmp_path):
    from ouroboros.outcomes import append_verification_receipt

    append_verification_receipt(tmp_path, "t", {"status": "fail", "check": "pytest tests/x.py"})
    append_verification_receipt(tmp_path, "t", {"status": "pass", "check": "pytest  tests/x.py "})
    monkeypatch.setattr(loop_mod, "_skill_finalization_message", lambda *_a, **_k: "")
    tools = SimpleNamespace(_ctx=SimpleNamespace(drive_root=str(tmp_path)))
    trace = {"reasoning_notes": [], "tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    assert loop_mod._maybe_inject_finalization_nudges(
        tools, tmp_path, "t", trace, "answer", [], lambda _m: None,
    ) is False


# ---------------------------------------------------------------------------
# P4.4 — transparency


def test_capability_omission_manifest_has_one_formatter_with_the_richest_detail():
    from ouroboros.tool_policy import CAPABILITY_OMISSION_HEADER, format_capability_omissions

    lines = format_capability_omissions([
        {"surface": "tools", "reason": "disabled_by_contract", "tools": ["web_search", "browse_page"]},
        {"surface": "extensions", "reason": "resource_blocked", "resource": "network=false"},
        {"surface": "mcp", "reason": "load_failed", "error": "boom"},
        "not-a-dict",
    ])
    assert lines[0] == CAPABILITY_OMISSION_HEADER
    # The real withheld NAMES are rendered (owner Q20), not "no detail".
    assert lines[1] == "- tools: disabled_by_contract (web_search, browse_page)"
    assert lines[2] == "- extensions: resource_blocked (network=false)"
    assert lines[3] == "- mcp: load_failed (boom)"
    assert len(lines) == 4
    assert format_capability_omissions([], header="") == []


def test_only_one_capability_omission_formatter_remains_in_the_tree():
    """The five copies are gone: no module renders the manifest line itself."""
    import pathlib

    import ouroboros.loop_tool_execution as lte  # noqa: F401  (tree sanity import)

    root = pathlib.Path(loop_mod.__file__).parent
    hits = [
        path.name
        for path in root.rglob("*.py")
        if "item.get('reason', 'unknown')" in path.read_text(encoding="utf-8")
        or 'item.get("reason", "unknown")' in path.read_text(encoding="utf-8")
    ]
    assert hits == ["tool_policy.py"], hits


def test_retrieval_fold_accumulates_counts_and_capped_urls():
    from ouroboros.loop_llm_call import fold_retrieval_usage

    accumulated: dict = {}
    fold_retrieval_usage(accumulated, {"prompt_tokens": 10})
    assert "retrieval" not in accumulated              # no search -> no fact at all

    fold_retrieval_usage(accumulated, {
        "server_tool_use": {"web_search_requests": 2},
        "web_search_sources": [
            {"url": "https://a.example/1", "title": "t", "content": "snippet"},
            {"url": "https://a.example/1"},
        ],
    })
    fold_retrieval_usage(accumulated, {
        "web_search_sources": [{"url": "https://b.example/2"}],
    })
    record = accumulated["retrieval"]
    assert record["web_search_requests"] == 3          # 2 + citations-without-counter
    assert record["source_count"] == 3
    assert record["urls"] == ["https://a.example/1", "https://b.example/2"]

    assert record["urls_omitted"] == 0                # nothing dropped -> say so
    assert len(record["urls_identity_sha256"]) == 64  # full-set identity, always

    # cap + no titles/snippets ever — and the cap DISCLOSES what it dropped (round 2)
    many = {"web_search_sources": [{"url": f"https://c.example/{i}"} for i in range(40)]}
    fresh: dict = {}
    fold_retrieval_usage(fresh, many)
    assert len(fresh["retrieval"]["urls"]) == 20
    assert fresh["retrieval"]["urls_omitted"] == 20   # exact count, not silence
    assert fresh["retrieval"]["source_count"] == 40
    assert all(isinstance(u, str) for u in fresh["retrieval"]["urls"])
    long_url = {"web_search_sources": [{"url": "https://d.example/" + "x" * 500}]}
    fresh2: dict = {}
    fold_retrieval_usage(fresh2, long_url)
    # per-URL bound goes through the SSOT truncator -> carries its own omission note
    assert "OMISSION NOTE" in fresh2["retrieval"]["urls"][0]
    assert fresh2["retrieval"]["urls"][0].startswith("https://d.example/xxx")


def test_canonical_statuses_reach_the_review_projection_without_new_required_keys():
    """The three-state strings ride the frozen `review_projection` surface; only the
    additive optional `reason` key was introduced."""
    from ouroboros.outcomes import _review_axis

    axis = _review_axis({
        "acceptance_decision": {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED, "reason": "dialogue_terminal",
            "source": "task_acceptance_review", "rationale": "terminal",
        },
        "review_decision": {"eligibility": "eligible", "trigger": "auto_nondirect"},
        "review_runs": [{"authority": "host_root", "aggregate_signal": "FAIL", "actors": []}],
    })
    assert set(axis["acceptance_decision"]) == {
        "status", "reason", "source", "rationale", "agent_disposition", "agent_rationale",
    }


def test_retrieval_urls_are_bounded_with_a_disclosed_count_and_a_full_set_hash():
    """Round-2 CRITICAL 2: `retrieval.urls` are "the URLs it fetched", so silently
    dropping URLs past the cap (and silently shortening each one) misdescribes the
    host-recorded fact. The bound stays; the silence does not."""
    from ouroboros.loop_llm_call import fold_retrieval_usage

    accumulated: dict = {}
    fold_retrieval_usage(accumulated, {
        "web_search_sources": [{"url": f"https://e.example/{i}"} for i in range(33)],
    })
    record = accumulated["retrieval"]
    assert len(record["urls"]) == 20 and record["urls_omitted"] == 13

    # The full-set hash is order-sensitive and covers the URLs the capped list dropped:
    # a different fetched set cannot share it.
    other: dict = {}
    fold_retrieval_usage(other, {
        "web_search_sources": [{"url": f"https://e.example/{i}"} for i in range(32)],
    })
    assert other["retrieval"]["urls"] == record["urls"]  # identical bounded evidence...
    assert other["retrieval"]["urls_identity_sha256"] != record["urls_identity_sha256"]


def test_two_distinct_urls_sharing_a_rendered_prefix_are_never_silently_collapsed():
    """Round-3 BIBLE P1: `fold_retrieval_usage` deduplicated AFTER rendering. Two
    DISTINCT long URLs with the same retained prefix and the same length render
    identically, so the second was dropped while `urls_omitted` still said 0 — a
    fetched URL lost from evidence that promises an EXACT omission count. Dedup is on
    the full RAW url, so every distinct URL is either carried or counted."""
    from ouroboros.loop_llm_call import fold_retrieval_usage

    stem = "https://f.example/" + "q" * 400
    first, second = stem + "/alpha", stem + "/omega"   # same prefix, equal length
    assert len(first) == len(second) and first != second

    acc: dict = {}
    fold_retrieval_usage(acc, {"web_search_sources": [{"url": first}, {"url": second}]})
    record = acc["retrieval"]
    # The two render byte-identically — which is exactly what made the loss silent.
    assert record["source_count"] == 2
    assert len(set(record["urls"])) == 1

    # The invariant: 2 distinct fetched URLs, fully represented OR exactly accounted for.
    assert len(record["urls"]) + record["urls_omitted"] == 2
    assert (len(record["urls"]) == 2 and record["urls_omitted"] == 0) or (
        len(record["urls"]) == 1 and record["urls_omitted"] == 1
    )

    # A genuine REPEAT of the same raw url is still a repeat, not a phantom second fetch.
    again: dict = {}
    fold_retrieval_usage(again, {"web_search_sources": [{"url": first}, {"url": first}]})
    assert len(again["retrieval"]["urls"]) == 1
    assert again["retrieval"]["urls_omitted"] == 0

    # The rolling chain hash already covers the RAW values, so it separates the two sets
    # the rendered list cannot: it must not inherit the same collision.
    assert record["urls_identity_sha256"] != again["retrieval"]["urls_identity_sha256"]
