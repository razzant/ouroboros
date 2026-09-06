"""Focused regressions for the first P3 acceptance-packet fix round."""

import json
from pathlib import Path
from types import SimpleNamespace


def test_prompt_projection_keeps_panels_ahead_of_an_oversized_lens():
    from ouroboros.review_evidence import format_review_evidence_for_prompt

    rendered = format_review_evidence_for_prompt(
        {"has_evidence": True, "task_id": "task-panel", "oversized_lens": "L" * 30_000},
        max_chars=1_100,
        acceptance_panels=[{
            "panel_id": "panel-must-survive",
            "surface": "task_acceptance",
            "aggregate_signal": "PASS",
            "transport_status": "success",
            "parse_status": "valid",
            "reason": "deciding finding " + "R" * 1_000,
            "actors": [{
                "slot_id": "slot_1",
                "response_ref": {"call_id": "response-must-survive"},
            }],
        }, {
            "panel_id": "panel-dropped-whole",
            "surface": "task_acceptance",
            "aggregate_signal": "PASS",
            "reason": "trailing record " + "T" * 1_000,
        }],
    )

    assert rendered.startswith("TASK ACCEPTANCE PANELS:")
    assert "panel-must-survive" in rendered
    assert '"aggregate_signal": "PASS"' in rendered
    assert "deciding finding" in rendered
    assert '"reason_omitted_chars"' in rendered
    assert "response-must-survive" in rendered
    panel_json = rendered.split("\n\n", 1)[0].split("\n", 1)[1]
    panel_projection = json.loads(panel_json)
    assert panel_projection["records"][0]["panel_id"] == "panel-must-survive"
    assert panel_projection["records_omitted"] == 1
    assert "panel-dropped-whole" not in rendered
    assert "OMISSION NOTE" in rendered
    assert "canonical source_ref" in rendered
    assert '"reader": "get_task_result"' in rendered


def test_partial_trajectory_is_non_resolving_but_complete_trajectory_resolves():
    from ouroboros.review_dispatch import task_acceptance_zero_physical_refusal
    from ouroboros.review_evidence import annotate_criteria_evidence_resolution
    from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary
    from ouroboros.review_substrate import task_acceptance_is_clean

    def _actor():
        return {
            "signal": "PASS",
            "parsed": {
                "outcome_tier": "solved",
                "criteria_used": [{
                    "criterion": "tool outcome",
                    "status": "supported",
                    "evidence_refs": ["tool_trajectory"],
                }],
            },
        }

    def _result(actor):
        return SimpleNamespace(
            aggregate_signal="PASS", degraded=False, actors=[actor],
        )

    partial = {
        "tool_trajectory": [{"tool": "read_file", "result_complete": False}],
        "__provenance__": {"tool_trajectory": "tool_result"},
    }
    partial_actor = _actor()
    annotate_criteria_evidence_resolution([partial_actor], partial)
    assert task_acceptance_zero_physical_refusal(partial) == {}
    assert acceptance_evidence_ref_vocabulary(partial)["tool_trajectory"] == "partial"
    assert task_acceptance_is_clean(_result(partial_actor)) is False

    complete = {
        "tool_trajectory": [{"tool": "read_file", "result_complete": True}],
        "__provenance__": {"tool_trajectory": "tool_result"},
    }
    complete_actor = _actor()
    annotate_criteria_evidence_resolution([complete_actor], complete)
    assert acceptance_evidence_ref_vocabulary(complete)["tool_trajectory"] == "packet_section"
    assert task_acceptance_is_clean(_result(complete_actor)) is True


def test_budget_truncated_repo_diff_cannot_resolve_clean_acceptance():
    from ouroboros.review_evidence import (
        _accept_enforce_budget,
        annotate_criteria_evidence_resolution,
    )
    from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary
    from ouroboros.review_substrate import task_acceptance_is_clean

    packet = _accept_enforce_budget({
        "repo_diff": "diff --git a/a b/a\n" + "x" * 50_000,
        "repo_diff_source_ref": {"task_id": "task-diff", "artifact": "repo_diff.patch"},
        "__provenance__": {"repo_diff": "host_attested"},
    }, budget=25_000)
    actor = {
        "signal": "PASS",
        "parsed": {"outcome_tier": "solved", "criteria_used": [{
            "criterion": "the patch is correct", "status": "supported",
            "evidence_refs": ["repo_diff"],
        }]},
    }
    annotate_criteria_evidence_resolution([actor], packet)

    assert packet["repo_diff_complete"] is False
    assert acceptance_evidence_ref_vocabulary(packet)["repo_diff"] == "partial"
    result = SimpleNamespace(aggregate_signal="PASS", degraded=False, actors=[actor])
    assert task_acceptance_is_clean(result) is False


def test_leading_trajectory_omission_from_packet_producer_cannot_resolve_clean(tmp_path):
    from ouroboros.review_dispatch import task_acceptance_zero_physical_refusal
    from ouroboros.review_evidence import (
        _ACCEPT_TRAJECTORY_MAX_CALLS,
        annotate_criteria_evidence_resolution,
        build_task_acceptance_evidence,
    )
    from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary
    from ouroboros.review_substrate import task_acceptance_is_clean
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task-traj")
    packet = build_task_acceptance_evidence(
        ctx, drive_root=tmp_path, task_id="task-traj",
        llm_trace={"tool_calls": [
            {"tool": "read_file", "status": "ok", "result": str(index)}
            for index in range(_ACCEPT_TRAJECTORY_MAX_CALLS + 1)
        ]},
    )
    actor = {
        "signal": "PASS",
        "parsed": {"outcome_tier": "solved", "criteria_used": [{
            "criterion": "the execution was sound", "status": "supported",
            "evidence_refs": ["tool_trajectory"],
        }]},
    }
    annotate_criteria_evidence_resolution([actor], packet)

    assert packet["tool_trajectory_omitted_leading"] == 1
    partial = next(
        row for row in packet["__unresolved_partial_artifacts__"]
        if row["tool"] == "tool_trajectory"
    )
    assert partial["status"] == "not_materialized_for_reviewer"
    assert partial["source_ref"] == packet["tool_trajectory_source_ref"]
    assert partial["source_ref"]["root"] == "artifact_store"
    assert partial["source_ref"]["reader"] == "read_file"
    assert task_acceptance_zero_physical_refusal(packet) == {}
    assert acceptance_evidence_ref_vocabulary(packet)["tool_trajectory"] == "partial"
    result = SimpleNamespace(aggregate_signal="PASS", degraded=False, actors=[actor])
    assert task_acceptance_is_clean(result) is False


def test_omitted_trajectory_corpus_round_trips_through_artifact_reader(tmp_path, monkeypatch):
    from ouroboros import artifacts
    from ouroboros.review_evidence import _ACCEPT_TRAJECTORY_MAX_CALLS, build_task_acceptance_evidence
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.registry import ToolContext

    calls = [
        {"tool": "read_file", "status": "ok", "args": {"path": f"file-{index}"}, "result": str(index)}
        for index in range(_ACCEPT_TRAJECTORY_MAX_CALLS + 1)
    ]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="trajectory-corpus")
    packet = build_task_acceptance_evidence(
        ctx, drive_root=tmp_path, task_id=ctx.task_id, llm_trace={"tool_calls": calls},
    )

    ref = packet["tool_trajectory_source_ref"]
    assert ref["root"] == "artifact_store"
    assert ref["artifact_ref"].startswith(f"artifact_store:{ref['path']}#chars=0-")
    rendered = _read_file(ctx, root=ref["root"], path=ref["path"])
    recovered = json.loads(rendered.split("\n", 1)[1])
    assert recovered == calls
    partial = next(
        row for row in packet["__unresolved_partial_artifacts__"]
        if row["tool"] == "tool_trajectory"
    )
    assert partial["status"] == "not_materialized_for_reviewer"
    assert partial["source_ref"] == ref

    recapped = build_task_acceptance_evidence(
        ctx, drive_root=tmp_path, task_id=ctx.task_id,
        llm_trace={"tool_calls": [
            {"tool": "run_command", "status": "ok", "result": str(index) * 10_000}
            for index in range(2)
        ]},
        budget_chars=5_000,
    )
    assert all(row["result_complete"] is False for row in recapped["tool_trajectory"])
    assert {
        row["status"] for row in recapped["__unresolved_partial_artifacts__"]
        if row["tool"] == "run_command"
    } == {"not_materialized_for_reviewer"}
    assert recapped["tool_trajectory_source_ref"]["root"] == "artifact_store"

    monkeypatch.setattr(
        artifacts, "store_actor_source_bytes",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("artifact store unavailable")),
    )
    unavailable_root = tmp_path / "unavailable"
    unavailable_ctx = ToolContext(
        repo_dir=tmp_path, drive_root=unavailable_root, task_id="trajectory-unavailable",
    )
    unavailable = build_task_acceptance_evidence(
        unavailable_ctx, drive_root=unavailable_root, task_id=unavailable_ctx.task_id,
        llm_trace={"tool_calls": calls},
    )
    missing = next(
        row for row in unavailable["__unresolved_partial_artifacts__"]
        if row["tool"] == "tool_trajectory"
    )
    assert "tool_trajectory_source_ref" not in unavailable
    assert missing["status"] == "source_unavailable"
    assert missing["source_ref"] == {}


def test_budget_recapped_trajectory_from_producer_cannot_resolve_clean():
    from ouroboros.review_evidence import (
        _accept_enforce_budget,
        annotate_criteria_evidence_resolution,
    )
    from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary
    from ouroboros.review_substrate import task_acceptance_is_clean

    packet = _accept_enforce_budget({
        "tool_trajectory": [{"tool": "run_command", "status": "ok", "result": "x" * 30_000}],
        "__provenance__": {"tool_trajectory": "tool_result"},
    }, budget=5_000)
    actor = {
        "signal": "PASS",
        "parsed": {"outcome_tier": "solved", "criteria_used": [{
            "criterion": "the command passed", "status": "supported",
            "evidence_refs": ["tool_trajectory"],
        }]},
    }
    annotate_criteria_evidence_resolution([actor], packet)

    assert packet["tool_trajectory"][0]["result_complete"] is False
    assert acceptance_evidence_ref_vocabulary(packet)["tool_trajectory"] == "partial"
    result = SimpleNamespace(aggregate_signal="PASS", degraded=False, actors=[actor])
    assert task_acceptance_is_clean(result) is False


def test_mixed_source_budget_recap_refuses_a_false_clean_pass(tmp_path):
    from ouroboros.review_dispatch import task_acceptance_zero_physical_refusal
    from ouroboros.review_evidence import _accept_enforce_budget
    from ouroboros.review_substrate import (
        ReviewRequest, ReviewSlot, run_review_request, task_acceptance_is_clean,
    )

    packet = _accept_enforce_budget({
        "tool_trajectory": [
            {"tool": "read_file", "result": "r" * 20_000,
             "result_source_ref": {"kind": "artifact", "path": "results/read.txt"}},
            {"tool": "run_command", "result": "c" * 20_000},
        ],
        "__provenance__": {"tool_trajectory": "tool_result"},
    }, budget=5_000)
    partials = packet["__unresolved_partial_artifacts__"]

    assert [row["status"] for row in partials] == [
        "not_materialized_for_reviewer", "source_unavailable",
    ]
    assert task_acceptance_zero_physical_refusal(packet)["status"] == "degraded_partial_source"
    leading = _accept_enforce_budget({
        "tool_trajectory": [
            {"tool": "run_command", "result": "lost"},
            *[{"tool": "read_file", "result": "kept"} for _ in range(20)],
        ],
        "owner_requirements_and_decisions": "x" * 10_000,
    }, budget=1_000)
    assert any(
        row["tool"] == "tool_trajectory" and row["status"] == "source_unavailable"
        for row in leading["__unresolved_partial_artifacts__"]
    )

    class _PassReviewer:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            return {"content": json.dumps({"verdict": "PASS", "findings": []})}, {}

    reviewer = _PassReviewer()
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="review", evidence=packet),
        slots=[ReviewSlot(slot_id="slot_1", model="reviewer")],
        drive_root=tmp_path,
        llm=reviewer,
    )
    assert reviewer.calls == 0
    assert result.aggregate_signal == "DEGRADED"
    assert task_acceptance_is_clean(result) is False


def test_skill_history_root_task_projection_avoids_whole_history_reads(monkeypatch, tmp_path):
    from ouroboros.skill_readiness import _skill_names_from_review_history
    from ouroboros import utils

    skill_dir = tmp_path / "state" / "skills" / "large-skill"
    skill_dir.mkdir(parents=True)
    history = skill_dir / "review_history.jsonl"
    history.write_text(
        (json.dumps({"root_task_id": "some-other-root", "padding": "x" * 200}) + "\n")
        * 20_000,
        encoding="utf-8",
    )
    (tmp_path / "state" / "skill_review_root_tasks.jsonl").write_text(
        json.dumps({"root_task_id": "root-wanted", "skill": "large-skill"}) + "\n",
        encoding="utf-8",
    )
    original = Path.read_text
    original_iter = utils.iter_jsonl_objects
    bounded_projection_reads = []

    def _guarded_read(path, *args, **kwargs):
        if path.name == "review_history.jsonl":
            raise AssertionError("acceptance rebuilt the full skill review history")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guarded_read)
    def _bounded_iter(path, *args, **kwargs):
        if path.name == "skill_review_root_tasks.jsonl":
            bounded_projection_reads.append(kwargs)
            assert kwargs.get("tail_bytes")
            assert kwargs.get("max_entries")
        return original_iter(path, *args, **kwargs)

    monkeypatch.setattr(utils, "iter_jsonl_objects", _bounded_iter)
    history = _skill_names_from_review_history(tmp_path, "root-wanted")
    assert history["names"] == ["large-skill"]
    assert history["coverage"]["complete"] is True
    assert len(bounded_projection_reads) == 1


def test_terminal_skill_review_updates_the_root_task_projection(tmp_path):
    from ouroboros.skill_review_runner import _append_terminal_history

    assert _append_terminal_history(
        tmp_path,
        "projected-skill",
        {"job_id": "job-1", "root_task_id": "root-1"},
        status="pass",
        terminal_reason="review_complete",
        ts="2026-09-03T00:00:00+00:00",
    )
    assert _append_terminal_history(
        tmp_path,
        "projected-skill",
        {"job_id": "job-1", "root_task_id": "root-1"},
        status="pass",
        terminal_reason="review_complete",
        ts="2026-09-03T00:00:00+00:00",
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "state" / "skill_review_root_tasks.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{
        "ts": "2026-09-03T00:00:00+00:00",
        "root_task_id": "root-1",
        "skill": "projected-skill",
        "job_id": "job-1",
    }]


def test_projection_failure_is_durable_in_coverage_and_runner_receipt(monkeypatch, tmp_path):
    from ouroboros import skill_review_history
    from ouroboros.skill_readiness import _skill_names_from_review_history
    from ouroboros.skill_review_runner import _append_terminal_history

    monkeypatch.setattr(
        skill_review_history, "_append_root_task_projection_once", lambda *_a, **_k: False,
    )
    assert not _append_terminal_history(
        tmp_path,
        "gap-skill",
        {"job_id": "job-gap", "root_task_id": "root-gap"},
        status="pass",
        terminal_reason="review_complete",
        ts="2026-09-04T00:00:00+00:00",
    )

    gap_path = tmp_path / "state" / "skill_review_root_tasks.gaps.jsonl"
    gap = json.loads(gap_path.read_text(encoding="utf-8").splitlines()[0])
    assert gap == {
        "ts": "2026-09-04T00:00:00+00:00",
        "root_task_id": "root-gap",
        "skill": "gap-skill",
        "job_id": "job-gap",
        "reason": "root_task_projection_append_failed",
    }
    history = _skill_names_from_review_history(tmp_path, "root-gap")
    assert history["names"] == []
    assert history["coverage"]["complete"] is False
    assert "root_task_projection_append_failed" in history["coverage"]["gap_reasons"]

    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    receipt = next(row for row in events if row.get("type") == "skill_review_history_append_failed")
    assert "root-task projection" in receipt["reason"]


def test_skill_review_projection_retry_finds_identity_before_bounded_tail(tmp_path):
    from ouroboros.skill_review_history import _append_root_task_projection_once

    original = {"job_id": "job-original", "root_task_id": "root-original"}
    assert _append_root_task_projection_once(tmp_path, "projected-skill", original)
    for index in range(129):
        assert _append_root_task_projection_once(
            tmp_path, "projected-skill",
            {"job_id": f"job-{index}", "root_task_id": f"root-{index}"},
        )
    assert _append_root_task_projection_once(tmp_path, "projected-skill", original)

    rows = [
        json.loads(line)
        for line in (tmp_path / "state" / "skill_review_root_tasks.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        row.get("root_task_id") == "root-original"
        and row.get("skill") == "projected-skill"
        and row.get("job_id") == "job-original"
        for row in rows
    ) == 1


def test_skill_review_projection_is_enrolled_as_a_hot_store():
    from ouroboros.agent_startup_checks import _hot_store_thresholds

    assert "state/skill_review_root_tasks.jsonl" in {
        relative for relative, _threshold, _remediation in _hot_store_thresholds()
    }


def test_acceptance_slot_fit_uses_packet_density():
    from ouroboros.review_dispatch import acceptance_slot_fit

    chars = 330_001

    cap, estimated = acceptance_slot_fit(
        SimpleNamespace(model="reviewer", max_tokens=16_384),
        SimpleNamespace(prompt_chars=lambda: chars),
        slot_input_caps={"reviewer": 200_000},
    )

    assert cap == 200_000
    assert estimated == 100_001
    assert estimated > chars // 4


def test_acceptance_slot_fit_reuses_packet_budget_caps(monkeypatch):
    from ouroboros.review_dispatch import acceptance_slot_fit
    from ouroboros.review_evidence import acceptance_packet_budget_chars
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools import review_synthesis

    calls = []
    caps = {"wide": 800_000, "narrow": 20_000}

    def _caps(models, **_kwargs):
        calls.append(list(models))
        return {model: caps[model] for model in models}

    monkeypatch.setattr(review_synthesis, "per_slot_input_token_limits", _caps)
    slots = [
        ReviewSlot(slot_id="slot_1", model="wide"),
        ReviewSlot(slot_id="slot_2", model="narrow"),
    ]
    budget = acceptance_packet_budget_chars(slots)
    monkeypatch.setattr(
        review_synthesis,
        "per_slot_input_token_limits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dispatch recalibrated cached packet caps")
        ),
    )

    cap, estimated = acceptance_slot_fit(
        slots[1], SimpleNamespace(prompt_chars=lambda: 99_000),
        slot_input_caps=budget.slot_input_caps,
    )

    assert calls == [["wide", "narrow"]]
    assert cap == 20_000
    assert estimated == 30_000


def test_budget_ladder_stops_shedding_after_predecessor_fits():
    from ouroboros.review_evidence import _accept_enforce_budget

    trajectory = [{"tool": f"tool-{index}", "result": "ok"} for index in range(21)]
    evidence = {
        "task_contract": {
            "requirements": "keep the trajectory",
            "predecessor_authority": {
                "previous_task_id": "prior-task",
                "envelope": "x" * 20_000,
            },
        },
        "tool_trajectory": trajectory,
    }
    compact_without_envelope = {
        **evidence,
        "task_contract": {
            "requirements": "keep the trajectory",
            "predecessor_authority": {
                "kind": "predecessor_authority_omitted_for_budget",
                "previous_task_id": "prior-task",
                "omitted_chars": len(json.dumps(
                    evidence["task_contract"]["predecessor_authority"]
                )),
            },
        },
        "omissions_manifest": [{
            "section": "task_contract.predecessor_authority",
            "omitted": len(json.dumps(
                evidence["task_contract"]["predecessor_authority"]
            )),
            "reason": "evidence_budget",
        }],
    }
    budget = len(json.dumps(compact_without_envelope)) + 10
    compact_without_envelope["__budget_note__"] = (
        f"⚠️ OMISSION NOTE: evidence exceeded {budget} chars; "
        f"omitted the predecessor authority envelope ({compact_without_envelope['omissions_manifest'][0]['omitted']} chars). "
        "Full content is durable off-axis."
    )
    budget = len(json.dumps(compact_without_envelope, ensure_ascii=False)) + 10

    result = _accept_enforce_budget(evidence, budget=budget)

    assert result["tool_trajectory"] == trajectory
    assert len(json.dumps(result, ensure_ascii=False, default=str)) <= budget
    assert not any(
        row.get("section") == "tool_trajectory"
        for row in result["omissions_manifest"]
    )


def test_acceptance_docs_have_complete_sentence_boundaries():
    development = Path("docs/DEVELOPMENT.md").read_text(encoding="utf-8")
    architecture = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "silent false green. The\n  Every forced rail" not in development
    assert "never certify success; An OPEN plan wave" not in architecture
    assert "never had claims. an unresolved reference" not in architecture


def test_refused_slot_reuses_the_already_persisted_prompt(monkeypatch, tmp_path):
    import ouroboros.review_substrate as substrate

    calls = []

    def _persist(*_args, **kwargs):
        calls.append(kwargs["call_type"])
        return {"manifest_ref": kwargs["call_id"]}

    monkeypatch.setattr(substrate, "persist_call", _persist)
    request = substrate.ReviewRequest(
        surface="task_acceptance",
        goal="review",
        task_id="prompt-reuse",
        evidence={"__unresolved_partial_artifacts__": [{
            "status": "source_unavailable",
        }]},
    )
    actor = substrate.ReviewCoordinator(drive_root=tmp_path)._run_slot(
        request,
        substrate.ReviewSlot(slot_id="slot_1", model="reviewer"),
        operation_id="operation-1",
    )

    assert actor.status == "not_dispatched"
    assert actor.prompt_ref == {"manifest_ref": "operation-1_prompt"}
    assert len([call for call in calls if call.endswith("_prompt")]) == 1
    assert len(calls) == 2


def test_broken_reflection_panel_keeps_commit_advisory_lens(monkeypatch):
    from ouroboros.reflection import generate_reflection
    from ouroboros import review_substrate

    captured = {}

    class _Llm:
        def chat(self, **kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            return {"content": "Reflection completed."}, {}

    monkeypatch.setattr(
        review_substrate,
        "compact_review_projection",
        lambda _runs: (_ for _ in ()).throw(ValueError("broken panel")),
    )

    generate_reflection(
        task={"id": "reflection-task", "text": "reflect"},
        llm_trace={"tool_calls": [], "review_runs": [{"broken": True}]},
        trace_summary="completed",
        llm_client=_Llm(),
        usage_dict={"rounds": 2, "cost": 0.1},
        review_evidence={"has_evidence": True, "lens_marker": "lens-survives"},
    )

    assert "lens-survives" in captured["prompt"]
    assert "review evidence unavailable" not in captured["prompt"]
