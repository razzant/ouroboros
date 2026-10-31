"""Focused continuation-authority regressions for terminal provenance."""

from __future__ import annotations

import json
from types import SimpleNamespace
from tests._store_shared import seed_result


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
    seed_result(root, row['task_id'], {"_schema_version": 1, **row})


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
    # A Continue successor inherits notes and addresses (THESIS I-7, owner 19.10): the salvaged
    # answer stays behind the exact reader with its size named, never as a preview in the brief.
    assert "result" not in automatic and automatic["omitted_fields"]["result"] > 0
    assert automatic["source"] == source
    assert tail not in json.dumps(automatic) and "unreviewed provider output" not in json.dumps(automatic)
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


def test_automatic_authority_carries_every_answer_as_its_reader(tmp_path):
    """The automatic predecessor injection is a bounded envelope for EVERY origin
    (owner 2026-08-30). A Continue successor inherits notes and addresses (THESIS
    I-7, owner 19.10): small and big answers alike stay behind the named pull
    source with their size, explicitly readable (the sibling salvage test pins that half)."""
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
        assert "result" not in automatic and automatic["omitted_fields"]["result"] > 0
        assert automatic["source"] == _authority_source(task_id)
        assert full_result[:20] not in json.dumps(automatic)


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


def test_automatic_plan_review_authority_stays_behind_its_reader(tmp_path):
    """A Continue successor never inherits the predecessor's review history (THESIS I-7,
    owner 19.10): the brief names `plan_review_state` and its size beside the reader,
    and the explicit read returns the complete state."""
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.contracts.task_contract import build_task_contract
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
    brief = task["predecessor_authority"]
    assert "plan_review_state" not in brief and brief["omitted_fields"]["plan_review_state"] > 0
    assert brief["source"] == _authority_source("planned")
    assert not any(claim in json.dumps(brief) for claim in claims)

    # The named pull source stays complete, and a minted envelope rebuilds untouched.
    explicit = json.loads(_get_task_result(
        ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=str(tmp_path)),
        "planned", include_authority=True,
    ))
    assert explicit["authority"]["plan_review_state"]["waves"][-1]["actors"]
    assert explicit["authority"]["plan_review_state"]["waves"][-1]["spec"]["acceptance_claims"] == claims
    envelope = task["predecessor_authority"]
    again = build_task_contract({"objective": "next", "predecessor_authority": envelope})
    assert again["predecessor_authority"] == envelope


def test_an_owner_source_in_history_still_importing_is_a_typed_refusal_never_a_crash(tmp_path):
    """A start naming an owner message the store cannot answer yet (its archive segment still imports)
    is refused with the typed, retryable outcome; once the segment is imported the same start passes."""
    from ouroboros import dialogue_import
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.project_dialogue import build_owner_message_ref
    from tests import _dialogue as dialogue_test

    row = {"ts": "2026-09-01T00:00:01Z", "chat_id": 1, "direction": "in", "text": "start the audit",
           "client_message_id": "owner-old"}
    dialogue_test.seed(tmp_path, archives={"chat_20260901T000000.jsonl": [row]},
                       live={"chat": [{**row, "client_message_id": "owner-new", "text": "later"}]})
    dialogue_import.cutover(tmp_path)  # the archive is planned, not imported yet
    env = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path)
    ref = build_owner_message_ref(chat_id=1, client_message_id="owner-old", ts=row["ts"], text=row["text"])
    refusal = validate_task_authority_sources(env, {"id": "resume", "origin_message_ref": ref})
    assert refusal["reason_code"] == "authority_source_unavailable" and "still being upgraded" in refusal["detail"]
    dialogue_test.import_all(tmp_path)
    task = {"id": "resume", "origin_message_ref": ref}
    assert validate_task_authority_sources(env, task) == {} and task["origin_message_text"] == "start the audit"

