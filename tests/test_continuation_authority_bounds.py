"""Focused continuation-authority regressions for terminal provenance."""

from __future__ import annotations

import json
from types import SimpleNamespace


def _authority_source(task_id: str) -> dict:
    return {
        "kind": "task_result",
        "task_id": task_id,
        "tool": "get_task_result",
        "arguments": {"task_id": task_id, "include_authority": True},
    }


def _write_result(root, row: dict) -> None:
    result_dir = root / "task_results"
    result_dir.mkdir(exist_ok=True)
    (result_dir / f"{row['task_id']}.json").write_text(
        json.dumps({"_schema_version": 1, **row}), encoding="utf-8",
    )


def test_host_salvage_automatic_authority_is_bounded_but_explicit_read_is_full(
    tmp_path,
):
    from ouroboros.agent_startup_checks import (
        _AUTOMATIC_HOST_SALVAGE_RESULT_CHARS,
        task_result_authority_projection,
        validate_task_authority_sources,
    )
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext

    task_id = "failed-predecessor"
    tail = "FULL_SALVAGE_TAIL_REMAINS_EXPLICITLY_READABLE"
    full_result = "unreviewed provider output\n" + (
        "x" * (_AUTOMATIC_HOST_SALVAGE_RESULT_CHARS + 5_000)
    ) + tail
    row = {
        "task_id": task_id,
        "status": "failed",
        "reason_code": "provider_unavailable",
        "terminal_origin": "host_salvage",
        "result": full_result,
        "outcome_axes": {"objective": "best_effort", "process": "failed"},
    }
    _write_result(tmp_path, row)
    source = _authority_source(task_id)
    task = {
        "id": "continuation",
        "budget_drive_root": str(tmp_path),
        "predecessor_authority_source": source,
    }
    env = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path)

    assert validate_task_authority_sources(env, task) == {}
    automatic = task["predecessor_authority"]
    result = automatic["result"]
    assert result["kind"] == "unreviewed_host_salvage"
    assert len(result["preview"]) <= _AUTOMATIC_HOST_SALVAGE_RESULT_CHARS
    assert result["full_chars"] >= len(full_result)  # serialized chars
    assert result["source_ref"] == {**source, "field": "authority.result"}
    # The carried prefix is byte-exact and SUBSTANTIAL - a preview carrying a
    # token prefix plus a marker would pass a startswith(200) check while
    # discarding the budget it promises.
    carried = len(result["preview"].split("\n\u26a0", 1)[0])
    assert carried > _AUTOMATIC_HOST_SALVAGE_RESULT_CHARS - 1_000
    assert result["preview"].startswith(full_result[:carried])
    assert "OMISSION NOTE" in result["preview"]
    assert tail not in json.dumps(automatic)
    assert len(json.dumps(automatic)) < 25_000
    # The bounded continuation envelope (delegation-usefulness, owner
    # 2026-08-30): identity + digest + pull source instead of the recursive
    # full-body copy that compiled 300K+ work orders.
    assert automatic["kind"] == "bounded_continuation_envelope"
    assert automatic["authority_chars"] > len(full_result)
    assert len(automatic["authority_sha256"]) == 64
    assert automatic["status"] == "failed"
    assert automatic["reason_code"] == "provider_unavailable"
    assert automatic["outcome_axes"] == row["outcome_axes"]
    nested_contract = automatic.get("task_contract") or {}
    assert "predecessor_authority" not in nested_contract, "no recursion carrier"

    # The shared projection and the explicit tool remain full. Only startup's
    # automatic predecessor injection is narrowed.
    assert task_result_authority_projection(row)["result"] == full_result
    explicit = json.loads(_get_task_result(
        ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            budget_drive_root=str(tmp_path),
        ),
        task_id,
        include_authority=True,
    ))
    assert explicit["authority"]["result"] == full_result
    assert tail in explicit["authority"]["result"]


def test_automatic_authority_rides_whole_or_pointer(tmp_path):
    """Decoupled contract (delegation-usefulness, owner 2026-08-30): the
    automatic predecessor injection is a bounded envelope for EVERY origin.
    A result that fits one ordinary tool-result budget rides whole (no pull
    round for small hops); a bigger one rides as a bounded preview beside the
    named pull source - the full body stays in task_results, explicitly
    readable (the sibling salvage test pins that half)."""
    from ouroboros.agent_startup_checks import (
        _AUTOMATIC_HOST_SALVAGE_RESULT_CHARS,
        validate_task_authority_sources,
    )

    env = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path)
    small_result = "concise model answer"
    big_result = "complete model answer\n" + (
        "m" * (_AUTOMATIC_HOST_SALVAGE_RESULT_CHARS + 5_000)
    )
    for task_id, full_result in (("model-small", small_result), ("model-big", big_result)):
        _write_result(tmp_path, {
            "task_id": task_id, "status": "completed", "result": full_result,
            "terminal_origin": "model_final",
        })
        task = {
            "id": f"continue-{task_id}",
            "budget_drive_root": str(tmp_path),
            "predecessor_authority_source": _authority_source(task_id),
        }
        assert validate_task_authority_sources(env, task) == {}
        automatic = task["predecessor_authority"]
        assert automatic["kind"] == "bounded_continuation_envelope"
        if full_result is small_result:
            assert automatic["result"] == full_result, "small hops ride whole"
        else:
            assert automatic["result"]["kind"] == "bounded_field_preview"
            # full_chars counts the SERIALIZED body (escapes included).
            assert automatic["result"]["full_chars"] >= len(full_result)
            preview = automatic["result"]["preview"]
            carried = len(preview.split("\n\u26a0", 1)[0])
            assert carried > _AUTOMATIC_HOST_SALVAGE_RESULT_CHARS - 1_000
            assert preview.startswith(full_result[:carried])
            assert "OMISSION NOTE" in preview


def test_legacy_collapse_fires_only_on_growth_carriers_and_is_idempotent():
    """build_task_contract collapses a legacy predecessor body only when it
    carries growth - a nested recursion carrier or an oversized string.
    Bounded bodies ride byte-identical (exact strings are authority), and a
    collapsed envelope survives a rebuild untouched."""
    from ouroboros.contracts.task_contract import build_task_contract

    bounded = {
        "source": {"kind": "task_result", "task_id": "flat"},
        "result": "small durable answer",
        "task_contract": {"objective": "carry on"},
    }
    contract = build_task_contract({"objective": "next", "predecessor_authority": bounded})
    assert contract["predecessor_authority"] == bounded, "no growth - no re-dressing"

    fat = {
        "source": {"kind": "task_result", "task_id": "deep"},
        "result": "r" * 25_000,
        "capability_delta": {"granted": ["net"]},
        "task_contract": {
            "objective": "grandparent objective",
            "predecessor_authority": {
                "result": "grandparent body",
                "source": {"kind": "task_result", "task_id": "grandpa"},
            },
        },
    }
    contract = build_task_contract({"objective": "next", "predecessor_authority": fat})
    envelope = contract["predecessor_authority"]
    assert envelope["kind"] == "bounded_continuation_envelope"
    assert envelope["collapsed_from"] == "legacy_full_body"
    assert "predecessor_authority" not in envelope["task_contract"], "recursion carrier dropped"
    assert envelope["task_contract"]["objective"] == "grandparent objective"
    assert envelope["capability_delta"] == {"granted": ["net"]}, "compact facts inherit"
    preview = envelope["result"]
    assert preview["kind"] == "bounded_field_preview"
    assert preview["full_chars"] == 25_000 and "OMISSION NOTE" in preview["preview"]
    assert preview["source_ref"]["task_id"] == "deep", "the pull route is named"
    assert envelope["source"] == fat["source"], "the pull route survives"
    assert envelope["previous_task_id"] == "grandpa", (
        "the cursor names the hop BEFORE this body's subject - the binding "
        "rule - never the subject's own id (a self-loop)")

    again = build_task_contract({"objective": "next-hop", "predecessor_authority": envelope})
    assert again["predecessor_authority"] == envelope, "envelope rebuild is a no-op"


def _plan_review_state(claims, finding_id: str) -> dict:
    """A two-wave v2 plan-review state whose reviewer transport dwarfs its decision."""
    def _wave(cycle_index: int, fingerprint: str) -> dict:
        return {
            "schema_version": 2, "cycle_index": cycle_index,
            "request_fingerprint": fingerprint, "spec_hash": "s" * 64,
            "goal": "Ship the deck", "aggregate": "REVIEW_REQUIRED",
            "spec": {"goal": "Ship the deck", "acceptance_claims": list(claims)},
            "findings": [{"id": finding_id, "finding_id": f"s1:{finding_id}",
                          "class": "blocking", "breaks": "goal", "summary": "thin"}],
            "dispositions": [{"finding_id": f"s1:{finding_id}", "decision": "reject"}],
            "closed": False, "paid": True, "retry_key": f"plan_review:{fingerprint}:{cycle_index}",
            "counts": {"blocking": 1},
            "actors": [{"slot_id": f"s{n}", "model": "m/a", "status": "ok",
                        "raw_text": "a" * 4_000} for n in range(1, 4)],
            "actors_degraded": [], "health_epoch": [{"slot_id": "s1", "health": "unknown"}],
            "reasons": ["need_evidence_repeat:gone.md"],
            "evidence_manifest": {"declared": ["notes.md"], "attached": [{"locator": "notes.md",
                                                                          "text": "n" * 3_000}]},
        }
    return {"schema_version": 2, "cycles_paid": 2, "need_evidence_seen": [],
            "waves": [_wave(1, "a" * 64), _wave(2, "b" * 64)]}


def test_plan_review_authority_core_keeps_the_decision_and_drops_the_transport():
    """Unit: identity, claims, findings, dispositions and closure survive because they
    are not transport keys; older waves compact; v1 and empty states pass through."""
    from ouroboros.task_results import plan_review_authority_core

    claims = ["exactly 5 slides", "every slide has a chart"]
    state = _plan_review_state(claims, "f1")
    core = plan_review_authority_core(state)

    assert state["waves"][-1]["actors"], "the input is not mutated"
    old, new = core["waves"]
    assert old["compact"] is True and old["request_fingerprint"] == "a" * 64
    for key in ("actors", "actors_degraded", "evidence_manifest", "health_epoch",
                "reasons", "retry_key"):
        assert key not in new, key
    assert new["request_fingerprint"] == "b" * 64 and new["spec_hash"] == "s" * 64
    assert new["spec"]["acceptance_claims"] == claims and new["goal"] == "Ship the deck"
    assert [f["id"] for f in new["findings"]] == ["f1"] and new["findings"][0]["class"] == "blocking"
    assert new["dispositions"] and new["closed"] is False and new["aggregate"] == "REVIEW_REQUIRED"
    core["waves"][-1]["spec"]["acceptance_claims"].append("mutated")
    assert state["waves"][-1]["spec"]["acceptance_claims"] == claims, "deepcopy, not a view"

    assert plan_review_authority_core({"schema_version": 1, "waves": [{"a": 1}]})["schema_version"] == 1
    assert plan_review_authority_core({}) == {}
    assert plan_review_authority_core(None) is None


def test_plan_review_authority_core_preserves_an_unknown_schema_version():
    from ouroboros.task_results import plan_review_authority_core

    state = {"schema_version": "corrupt", "waves": [{"actors": ["untrusted"]}]}

    assert plan_review_authority_core(state) is state


def test_automatic_plan_review_authority_carries_claims_not_reviewer_transport(tmp_path):
    """The continuation envelope's `plan_review_state` must reach the acceptance claims
    and blocking finding ids; reviewer transport used to fill the preview before them."""
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_results import plan_review_authority_core as plan_core
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext

    claims = ["exactly 5 slides", "every slide has a chart"]
    state = _plan_review_state(claims, "f1")
    _write_result(tmp_path, {
        "task_id": "planned", "status": "completed", "result": "done",
        "terminal_origin": "model_final", "plan_review_state": state,
    })
    env = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path)
    task = {
        "id": "continue-planned", "budget_drive_root": str(tmp_path),
        "predecessor_authority_source": _authority_source("planned"),
    }
    assert validate_task_authority_sources(env, task) == {}
    carried = task["predecessor_authority"]["plan_review_state"]
    text = json.dumps(carried)
    projection = carried["projection"]
    assert projection["projected_from"] == "plan_review_authority_core"
    assert projection["dropped_keys"] == [
        "actors", "actors_degraded", "evidence_manifest", "health_epoch", "reasons", "retry_key",
    ]
    assert projection["full_chars"] == len(json.dumps(state, ensure_ascii=False, sort_keys=True, default=str))
    assert projection["source_ref"] == {
        **_authority_source("planned"), "field": "authority.plan_review_state",
    }
    assert all("actors" not in wave for wave in carried["waves"])
    for claim in (*claims, "f1", "REVIEW_REQUIRED"):
        assert claim in text, claim
    if isinstance(carried, dict) and carried.get("kind") == "bounded_field_preview":
        assert carried["source_ref"]["field"] == "authority.plan_review_state"
        assert carried["full_chars"] >= len(json.dumps(plan_core(state)))

    # The named pull source stays complete, and a minted envelope rebuilds untouched.
    explicit = json.loads(_get_task_result(
        ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=str(tmp_path)),
        "planned", include_authority=True,
    ))
    assert explicit["authority"]["plan_review_state"]["waves"][-1]["actors"]
    envelope = task["predecessor_authority"]
    again = build_task_contract({"objective": "next", "predecessor_authority": envelope})
    assert again["predecessor_authority"] == envelope
