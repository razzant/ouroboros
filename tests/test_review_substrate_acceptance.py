"""The evidence diff and the host acceptance panel.

Split by theme out of ``tests/test_review_substrate_v2.py``. This module owns
the host-owned acceptance review: collect_turn_diff evidence (tracked,
untracked, committed, redacted), host eligibility and retry-root markers, the
root acceptance checkpoint, stale-lineage protection and the applied
enforcement impact record.
"""

import json
from types import SimpleNamespace

from ouroboros.review_substrate import ReviewSlot


def test_collect_turn_diff_surfaces_tracked_and_untracked(tmp_path):
    """T1 (v6.35.0): collect_turn_diff must surface BOTH tracked modifications and
    untracked NEW files (a self-authored test the agent just wrote) so the
    reviewer can judge evidence independence."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "src.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
           cwd=repo, check=True, capture_output=True)
    (repo / "src.py").write_text("x = 2\n", encoding="utf-8")            # tracked mod
    (repo / "test_new.py").write_text("def test_x(): pass\n", encoding="utf-8")  # untracked new

    diff = collect_turn_diff(NS(repo_dir=repo))
    assert "src.py" in diff
    assert "test_new.py" in diff  # the untracked self-authored test is visible


def test_collect_turn_diff_untracked_survives_large_tracked_diff(tmp_path):
    """T1 round-2 fix: a large tracked diff must NOT clip away the untracked
    new-file names (independent truncation)."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "big.py").write_text("x = 0\n", encoding="utf-8")
    sp.run(["git", "add", "big.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
           cwd=repo, check=True, capture_output=True)
    # >20000-char tracked modification, plus an untracked self-authored test.
    (repo / "big.py").write_text("\n".join(f"v{i} = {i}" for i in range(5000)), encoding="utf-8")
    (repo / "test_self.py").write_text("def test_self(): assert True\n", encoding="utf-8")

    diff = collect_turn_diff(NS(repo_dir=repo))
    assert "test_self.py" in diff  # untracked name survives despite the huge tracked diff
    assert "Untracked working-tree files" in diff

def test_acceptance_review_evidence_diff_is_host_owned(monkeypatch, tmp_path):
    """T1 (v6.35.0): the host-collected repo_diff must override any agent-supplied
    repo_diff so the EVIDENCE-INDEPENDENCE judgment can't be steered by a stale
    value passed through the public task_acceptance_review tool."""
    from types import SimpleNamespace as NS

    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    captured = {}

    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kw: "HOST_DIFF_REAL")

    def _fake_run(request, **kwargs):
        captured["evidence"] = dict(request.evidence)
        return NS(aggregate_signal="PASS")

    monkeypatch.setattr(rs, "run_review_request", _fake_run)
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **k: [ReviewSlot(slot_id="a", model="m")])

    ctx = NS(
        drive_root=str(tmp_path), task_id="t",
        task_metadata={"root_task_id": "root", "parent_task_id": "root"},
    )
    _handle_task_acceptance_review(ctx, claim="done", evidence={"repo_diff": "STALE_AGENT_DIFF"})

    # v6.51.0: host repo_diff stays host-owned; the agent value is demoted (not promoted) under
    # the clearly-tagged `agent_supplied` block (was a top-level key pre-v6.51.0).
    assert captured["evidence"]["repo_diff"] == "HOST_DIFF_REAL"
    assert captured["evidence"]["agent_supplied"]["agent_supplied_repo_diff"] == "STALE_AGENT_DIFF"


def test_acceptance_review_empty_host_diff_does_not_fall_back_to_agent(monkeypatch, tmp_path):
    """T1 (v6.35.0): an EMPTY host diff is a valid fact (clean repo), not a reason
    to promote the agent-supplied diff to host-fact status — else the agent could
    steer EVIDENCE-INDEPENDENCE simply by acting when the host diff is empty."""
    from types import SimpleNamespace as NS

    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    captured = {}
    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kw: "")

    def _fake_run(request, **kwargs):
        captured["evidence"] = dict(request.evidence)
        return NS(aggregate_signal="PASS")

    monkeypatch.setattr(rs, "run_review_request", _fake_run)
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **k: [ReviewSlot(slot_id="a", model="m")])

    ctx = NS(
        drive_root=str(tmp_path), task_id="t",
        task_metadata={"root_task_id": "root", "parent_task_id": "root"},
    )
    _handle_task_acceptance_review(ctx, claim="done", evidence={"repo_diff": "FABRICATED_AGENT_DIFF"})

    # repo_diff stays the (empty) host fact; the agent value is only the demoted, tagged key
    # under `agent_supplied` (v6.51.0 relocation — was top-level).
    assert captured["evidence"]["repo_diff"] == ""
    assert captured["evidence"]["agent_supplied"]["agent_supplied_repo_diff"] == "FABRICATED_AGENT_DIFF"


def test_acceptance_review_records_agent_disposition(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    captured = {}
    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kw: "")

    def _fake_run(request, **kwargs):
        captured["evidence"] = dict(request.evidence)
        return NS(aggregate_signal="PASS", actors=[], parsed_findings=[])

    monkeypatch.setattr(rs, "run_review_request", _fake_run)
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **k: [ReviewSlot(slot_id="a", model="m")])
    monkeypatch.setattr(rs, "build_improvement_capsule", lambda _result: "")

    ctx = NS(
        drive_root=str(tmp_path), drive_logs=lambda: tmp_path / "logs", task_id="t",
        task_metadata={"root_task_id": "root", "parent_task_id": "root"},
    )
    raw = _handle_task_acceptance_review(
        ctx,
        claim="done",
        agent_disposition="rejected",
        rationale="Reviewer asked for a benchmark-specific workaround; I reject it as scope drift.",
    )
    payload = json.loads(raw)

    assert captured["evidence"]["agent_supplied"]["agent_decision"]["disposition"] == "rejected"
    assert payload["agent_decision"]["disposition"] == "rejected"
    assert "scope drift" in payload["agent_decision"]["rationale"]
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().strip())
    assert event["type"] == "deprecated_task_acceptance_alias"
    assert event["aliases"] == ["agent_disposition"]
    assert event["removal"] == "next_major"


def test_root_acceptance_tool_defers_to_host_without_model_calls(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setattr(
        rs,
        "run_review_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model review must not run")),
    )
    monkeypatch.setattr(
        rs,
        "triad_delivery_slots",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("review slots must not resolve")),
    )
    ctx = NS(
        drive_root=str(tmp_path),
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={},
    )

    first = json.loads(_handle_task_acceptance_review(
        ctx,
        claim="complete",
        goal="ship the result",
        checklist="tests pass",
        evidence={"verification_receipt": "receipt-1"},
    ))
    second = json.loads(_handle_task_acceptance_review(
        ctx,
        claim="complete",
        goal="ship the result",
        checklist="tests pass",
        evidence={"verification_receipt": "receipt-1"},
    ))
    changed_claim = json.loads(_handle_task_acceptance_review(
        ctx,
        claim="complete with a documented limitation",
        goal="ship the result",
        checklist="tests pass and limitation is disclosed",
        evidence={"verification_receipt": "receipt-1"},
    ))

    assert first["status"] == "deferred_to_host_acceptance"
    assert first["authoritative"] is False
    assert first["request"]["checklist"] == "tests pass"
    assert len(first["evidence_revision"]) == 64
    assert second["evidence_revision"] == first["evidence_revision"]
    assert changed_claim["evidence_revision"] != first["evidence_revision"]


def test_typed_retry_root_defers_self_review_and_is_host_eligible(
    monkeypatch, tmp_path,
):
    from types import SimpleNamespace as NS

    import ouroboros.loop as loop_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setattr(
        rs,
        "run_review_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("normalized root self-call must not run a model review")
        ),
    )
    monkeypatch.setattr(
        rs,
        "triad_delivery_slots",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("normalized root self-call must not resolve review slots")
        ),
    )
    retry_id = "retry-root"
    prior_attempt_id = "logical-root"
    metadata = {
        "task_id": retry_id,
        "root_task_id": prior_attempt_id,
        "parent_task_id": "",
        "delegation_role": "root",
        "original_task_id": prior_attempt_id,
        "timeout_retry_from": prior_attempt_id,
    }
    tool_ctx = NS(
        drive_root=str(tmp_path),
        task_id=retry_id,
        task_metadata=metadata,
        task_contract={},
    )

    payload = json.loads(
        _handle_task_acceptance_review(tool_ctx, claim="retry complete")
    )
    assert payload["status"] == "deferred_to_host_acceptance"

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = retry_id
    registry._ctx.task_metadata = metadata
    registry._ctx.task_contract = {}
    seen = {}
    real_eligible = loop_mod._task_acceptance_eligible

    def capture_eligible(mode, trace, direct, **kwargs):
        result = real_eligible(mode, trace, direct, **kwargs)
        seen.update(is_root_task=kwargs["is_root_task"], result=result)
        return result

    monkeypatch.setattr(loop_mod, "_task_acceptance_eligible", capture_eligible)
    monkeypatch.setattr(
        loop_mod,
        "_begin_task_acceptance_fence",
        lambda *_args, **_kwargs: (False, None),
    )
    assert loop_mod._run_task_acceptance_review_once(
        tools=registry,
        content="retry complete",
        task_id=retry_id,
        task_type="task",
        llm_trace={"tool_calls": []},
        drive_root=tmp_path,
        messages=[],
        emit_progress=lambda _message, *, incident=None: None,
    ) is True
    assert seen == {
        "is_root_task": True,
        "result": (True, "auto_nondirect"),
    }


def test_retry_root_markers_must_agree_before_acceptance_authority(
    monkeypatch, tmp_path,
):
    from types import SimpleNamespace as NS

    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    calls = []
    monkeypatch.setattr(
        rs,
        "triad_delivery_slots",
        lambda **kwargs: [ReviewSlot(slot_id="legacy", model="m")],
    )
    monkeypatch.setattr(rs, "build_improvement_capsule", lambda _result: "")
    monkeypatch.setattr(rs, "dissent_findings", lambda _result: [])

    def fake_run(request, **kwargs):
        calls.append(request.task_id)
        return NS(aggregate_signal="PASS", actors=[], parsed_findings=[])

    monkeypatch.setattr(rs, "run_review_request", fake_run)
    metadata = {
        "root_task_id": "logical-root",
        "parent_task_id": "",
        "delegation_role": "root",
        "original_task_id": "prior-a",
        "timeout_retry_from": "prior-b",
    }
    payload = json.loads(_handle_task_acceptance_review(
        NS(
            drive_root=str(tmp_path),
            task_id="malformed-retry",
            task_metadata=metadata,
            task_contract={},
        ),
        claim="done",
    ))

    assert payload["aggregate_signal"] == "PASS"
    assert calls == ["malformed-retry"]


def test_typed_retry_root_receives_root_acceptance_checkpoint():
    import ouroboros.loop as loop_mod

    trace = {}
    ctx = SimpleNamespace(
        task_id="retry-2",
        task_metadata={
            "root_task_id": "logical-root",
            "parent_task_id": "",
            "delegation_role": "root",
            "original_task_id": "retry-1",
            "timeout_retry_from": "retry-1",
        },
    )

    loop_mod._mark_root_acceptance_checkpoint(
        ctx, trace, status="pass", pass_index=1,
    )

    assert trace["root_phase_checkpoint"] == {
        "phase": "task_acceptance",
        "status": "pass",
        "pass_index": 1,
        "post_task_synthesis": "pending_once",
    }


def test_root_acceptance_agent_refs_reach_host_packet_beyond_trajectory_cap(
    monkeypatch, tmp_path,
):
    from types import SimpleNamespace as NS

    import ouroboros.loop as loop_mod
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    # v6.71.1 evidence-parity: the trajectory per-result cap rose from a hidden 700 to
    # the actor's default window (_ACCEPT_RESULT_CAP == DEFAULT_TOOL_RESULT_LIMIT). Push
    # the ref past the NEW cap so the test still exercises "beyond the trajectory cap →
    # still reaches the host packet via the agent_supplied path".
    agent_evidence = {
        "long_note": "x" * 16000,
        "receipt_ref": "artifact://receipt-123",
        "trailing_note": "y" * 5000,
    }
    tool_ctx = NS(
        drive_root=str(tmp_path),
        repo_dir=tmp_path,
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={},
    )
    raw = _handle_task_acceptance_review(
        tool_ctx,
        claim="complete",
        goal="ship the verified result",
        checklist="receipt is present",
        evidence=agent_evidence,
    )
    payload = json.loads(raw)
    assert payload["agent_supplied"]["receipt_ref"] == "artifact://receipt-123"

    trace = {"tool_calls": []}
    process_tool_results(
        [{
            "fn_name": "task_acceptance_review",
            "tool_call_id": "acceptance-call",
            "result": raw,
            "is_error": False,
            "args_for_log": {
                "claim": "complete",
                "evidence": agent_evidence,
            },
            "tool_args": {},
            "result_meta": {"status": "ok"},
        }],
        [],
        trace,
        emit_progress=lambda _message, *, incident=None: None,
    )

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "root"
    registry._ctx.root_task_id = "root"
    registry._ctx.task_metadata = {"root_task_id": "root"}
    registry._ctx.task_contract = {}
    host_ctx = loop_mod._TaskAcceptanceContext(
        tools=registry,
        content="complete",
        task_id="root",
        task_type="task",
        llm_trace=trace,
        drive_root=tmp_path,
        messages=[],
        emit_progress=lambda _message, *, incident=None: None,
        mode="auto",
        subtree_statuses=[],
        budget_profile=None,
        passes_done=0,
    )
    host_evidence = loop_mod._build_host_acceptance_evidence(host_ctx)

    assert host_evidence["agent_supplied"]["receipt_ref"] == (
        "artifact://receipt-123"
    )
    assert "artifact://receipt-123" not in json.dumps(
        host_evidence.get("tool_trajectory") or [], ensure_ascii=False,
    )


def test_off_mode_root_and_auto_mode_child_keep_existing_model_review(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    calls = []
    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kwargs: "")
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **kwargs: [ReviewSlot(slot_id="a", model="m")])
    monkeypatch.setattr(rs, "build_improvement_capsule", lambda _result: "")
    monkeypatch.setattr(rs, "dissent_findings", lambda _result: [])

    def fake_run(request, **kwargs):
        calls.append((request.task_id, request.surface))
        return NS(aggregate_signal="PASS", actors=[], parsed_findings=[])

    monkeypatch.setattr(rs, "run_review_request", fake_run)

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    root_ctx = NS(
        drive_root=str(tmp_path),
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={},
    )
    root_payload = json.loads(_handle_task_acceptance_review(root_ctx, claim="root done"))

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    child_ctx = NS(
        drive_root=str(tmp_path),
        task_id="child",
        root_task_id="root",
        parent_task_id="root",
        delegation_role="subagent",
        task_metadata={
            "root_task_id": "root",
            "parent_task_id": "root",
            "delegation_role": "subagent",
        },
        task_contract={},
    )
    child_payload = json.loads(_handle_task_acceptance_review(child_ctx, claim="child done"))

    assert calls == [("root", "task_acceptance"), ("child", "task_acceptance")]
    assert root_payload["aggregate_signal"] == "PASS"
    assert child_payload["aggregate_signal"] == "PASS"


def test_stale_parent_lineage_cannot_trigger_a_second_host_panel(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.loop as loop_mod
    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kwargs: "")
    monkeypatch.setattr(
        rs,
        "triad_delivery_slots",
        lambda **kwargs: [ReviewSlot(slot_id="a", model="m")],
    )
    monkeypatch.setattr(rs, "build_improvement_capsule", lambda _result: "")
    monkeypatch.setattr(rs, "dissent_findings", lambda _result: [])
    calls = []

    def fake_run(request, **kwargs):
        calls.append((request.task_id, request.surface))
        return NS(aggregate_signal="PASS", actors=[], parsed_findings=[])

    monkeypatch.setattr(rs, "run_review_request", fake_run)
    metadata = {
        # Legacy/malformed snapshot: root id is absent but an old parent remains.
        "parent_task_id": "missing-parent",
        "delegation_role": "root",
    }
    tool_ctx = NS(
        drive_root=str(tmp_path),
        task_id="restored-task",
        task_metadata=metadata,
        task_contract={},
    )
    payload = json.loads(
        _handle_task_acceptance_review(tool_ctx, claim="restored result")
    )
    assert payload["aggregate_signal"] == "PASS"
    assert calls == [("restored-task", "task_acceptance")]

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "restored-task"
    registry._ctx.task_metadata = metadata
    registry._ctx.task_contract = {}
    monkeypatch.setattr(
        loop_mod,
        "_begin_task_acceptance_fence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale-parent lineage must not reach the host panel")
        ),
    )
    trace = {"tool_calls": [], "review_runs": []}
    assert loop_mod._run_task_acceptance_review_once(
        tools=registry,
        content="restored result",
        task_id="restored-task",
        task_type="task",
        llm_trace=trace,
        drive_root=tmp_path,
        messages=[],
        emit_progress=lambda _message, *, incident=None: None,
    ) is False
    assert trace["review_decision"] == {
        "eligibility": "not_eligible",
        "trigger": "skipped_child_advisory",
    }
    assert calls == [("restored-task", "task_acceptance")]

def test_host_acceptance_enforcement_impact_records_applied_action(tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.loop as loop_mod

    tool_ctx = NS(_task_acceptance_seen_bindings={})
    ctx = loop_mod._TaskAcceptanceContext(
        tools=NS(_ctx=tool_ctx),
        content="candidate",
        task_id="impact",
        task_type="task",
        llm_trace={"review_runs": []},
        drive_root=tmp_path,
        messages=[],
        emit_progress=lambda _message, *, incident=None: None,
        mode="required",
        subtree_statuses=[],
        budget_profile={},
        passes_done=0,
        review_binding={"binding_hash": "b" * 64},
    )
    degraded = NS(
        aggregate_signal="DEGRADED",
        degraded=True,
        actors=[],
        parsed_findings=[],
        degraded_reasons=["no quorum"],
        request={},
    )

    record = loop_mod._record_host_acceptance_run(ctx, degraded)
    assert record["enforcement_impact"] == "degrades_completion"
    loop_mod._set_applied_host_acceptance_impact(
        record,
        degraded,
        requires_revision=True,
    )
    assert record["enforcement_impact"] == "requires_revision"
    loop_mod._set_applied_host_acceptance_impact(
        record,
        degraded,
        requires_revision=False,
    )
    assert record["enforcement_impact"] == "degrades_completion"

def test_task_acceptance_review_schema_exposes_agent_disposition():
    from ouroboros.tools.review import get_tools

    tool = next(entry for entry in get_tools() if entry.name == "task_acceptance_review")
    props = tool.schema["parameters"]["properties"]

    assert props["agent_disposition"]["enum"] == ["accepted", "rejected", "partial", "deferred"]
    assert "rationale" in props

def test_collect_turn_diff_redacts_secrets(tmp_path):
    """T1 (v6.35.0): a tracked credential edit must be REDACTED before the diff
    reaches reviewer LLM slots (no raw secret exfiltration)."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "conf.py").write_text('API_KEY = "placeholder"\n', encoding="utf-8")
    sp.run(["git", "add", "conf.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
           cwd=repo, check=True, capture_output=True)
    # Assemble the fake provider key from chunks so this test FILE contains no
    # contiguous provider-key literal (secret scanners match source, not runtime).
    # The concatenated runtime value is what the redactor must catch.
    secret = "sk-" + "or-" + "v1-" + "abcdef1234567890" * 2 + "deadbeef"
    (repo / "conf.py").write_text(f'API_KEY = "{secret}"\n', encoding="utf-8")

    diff = collect_turn_diff(NS(repo_dir=repo))
    assert secret not in diff           # the literal secret value is gone
    assert "REDACTED" in diff           # replaced with a redaction marker
    assert "conf.py" in diff            # the file/path (evidence-independence fact) survives


def test_collect_turn_diff_surfaces_committed_change(tmp_path):
    """T1 (v6.35.0): when the turn's work was already committed, `git diff HEAD`
    is empty — collect_turn_diff must still surface the committed files via the
    most recent commit so evidence independence can be judged."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "base"],
           cwd=repo, check=True, capture_output=True)
    # Commit the turn's work, so `git diff HEAD` is empty.
    (repo / "feature.py").write_text("def feat():\n    return 1\n", encoding="utf-8")
    sp.run(["git", "add", "feature.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "feature"],
           cwd=repo, check=True, capture_output=True)

    # Without a current-turn commit signal, the unrelated HEAD commit is NOT shown.
    assert "feature.py" not in collect_turn_diff(NS(repo_dir=repo))
    # With the commit signal (this turn committed), the committed work IS surfaced.
    diff = collect_turn_diff(NS(repo_dir=repo), include_recent_commit=True)
    assert "feature.py" in diff
    assert "committed this turn" in diff


def test_collect_turn_diff_disables_git_exec_drivers(tmp_path):
    """v6.35.0 security: the active workspace may be an UNTRUSTED repo, so
    collect_turn_diff must run git with --no-ext-diff AND --no-textconv — a
    repo-configured textconv/external-diff driver must never execute on the host
    while collecting review evidence (Bible P3)."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    marker = tmp_path / "pwned"
    # A malicious textconv driver that would create a marker file if git ran it.
    sp.run(["git", "config", "diff.evil.textconv", f"sh -c 'touch {marker}'; cat"],
           cwd=repo, check=True, capture_output=True)
    (repo / ".gitattributes").write_text("*.secret diff=evil\n", encoding="utf-8")
    (repo / "f.secret").write_text("one\n", encoding="utf-8")
    sp.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "x"],
           cwd=repo, check=True, capture_output=True)
    # Modify the attributed file so the tracked diff would render it via textconv.
    (repo / "f.secret").write_text("two\n", encoding="utf-8")

    # Exercises both the `git diff HEAD` and the `git show HEAD` code paths.
    collect_turn_diff(NS(repo_dir=repo), include_recent_commit=True)
    assert not marker.exists()   # the textconv driver must NOT have executed


def test_collect_turn_diff_does_not_assert_untracked_authorship(tmp_path):
    """T1 (v6.35.0): untracked files are labeled honestly as working-tree state,
    NOT asserted as authored 'this turn' — the host has no baseline, so it must
    not steer the reviewer's EVIDENCE-INDEPENDENCE judgment with a false claim."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "base"],
           cwd=repo, check=True, capture_output=True)
    # A pre-existing untracked file (the host cannot prove it was authored now).
    (repo / "preexisting_test.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")

    diff = collect_turn_diff(NS(repo_dir=repo))
    assert "preexisting_test.py" in diff          # surfaced as evidence
    assert "this turn" not in diff.lower()         # but NOT asserted as authored now
    assert "working-tree" in diff.lower()          # honestly labeled


def test_collect_turn_diff_includes_commit_even_with_leftover_dirty(tmp_path):
    """T1 (v6.35.0): a turn that commits AND leaves further dirty tracked changes
    must surface BOTH — the committed patch is no longer dropped just because the
    working tree is also dirty."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "base"],
           cwd=repo, check=True, capture_output=True)
    # This turn: commit feature.py ...
    (repo / "feature.py").write_text("def feat():\n    return 1\n", encoding="utf-8")
    sp.run(["git", "add", "feature.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "feature"],
           cwd=repo, check=True, capture_output=True)
    # ... then leave a further dirty tracked edit (so `git diff HEAD` is NON-empty).
    (repo / "a.py").write_text("x = 2  # tweaked\n", encoding="utf-8")

    diff = collect_turn_diff(NS(repo_dir=repo), include_recent_commit=True)
    assert "tweaked" in diff                       # the leftover dirty tracked change
    assert "feature.py" in diff                    # AND the committed patch
    assert "committed this turn" in diff
