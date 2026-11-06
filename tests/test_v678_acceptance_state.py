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


from ouroboros.outcomes import (
    ACCEPTANCE_FINALIZED_UNACCEPTED,
)


# ---------------------------------------------------------------------------
# The root admission seal's owner-follow-up release


# ---------------------------------------------------------------------------
# Forced rails and the readers of old acceptance decisions


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
