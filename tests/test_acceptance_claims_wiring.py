"""W2 claims wiring: the ONE pure seam ``effective_acceptance_claims`` and its two
consumers — the acceptance-evidence packet (ingress-contract claims first, the
CLOSED plan wave's frozen claims only when ingress is empty) and the child-contract
builder (claims re-stated AFTER the **parent_contract spread). The running task
contract is NEVER mutated: plan-frozen claims live in plan_review_state and are
resolved at read time."""

import tempfile
import types as _t
from pathlib import Path

from ouroboros.contracts.task_contract import effective_acceptance_claims
from ouroboros.review_evidence import build_task_acceptance_evidence


def _v2_wave(fingerprint: str, claims: list, *, aggregate: str = "GREEN", closed: bool = True) -> dict:
    from ouroboros.tools.plan_spec import normalize_spec, spec_hash

    spec, errors = normalize_spec({"goal": "g", "acceptance_claims": claims})
    assert not errors
    return {
        "schema_version": 2, "cycle_index": 1, "request_fingerprint": fingerprint,
        "spec": spec, "spec_hash": spec_hash(spec), "findings": [], "aggregate": aggregate,
        "closed": closed, "dispositions": [], "paid": True,
    }


def _close_green_wave(root: Path, task_id: str, fingerprint: str, claims: list) -> None:
    from ouroboros.task_results import STATUS_RUNNING, record_plan_review_wave, write_task_result

    write_task_result(root, task_id, STATUS_RUNNING, result="running")
    record_plan_review_wave(root, task_id, _v2_wave(fingerprint, claims))


def test_effective_acceptance_claims_ingress_wins():
    wave = {"acceptance_claims": ["plan claim"]}
    claims, source = effective_acceptance_claims(
        {"acceptance_claims": [{"id": "c1", "claim": "ingress claim"}]}, wave,
    )
    assert source == "ingress_contract"
    assert [c["claim"] for c in claims] == ["ingress claim"]


def test_effective_acceptance_claims_plan_fallback_and_empty():
    claims, source = effective_acceptance_claims(
        {"acceptance_claims": []}, {"acceptance_claims": ["game boots", "score persists"]},
    )
    assert source == "plan_review"
    assert [c["claim"] for c in claims] == ["game boots", "score persists"]
    assert [c["id"] for c in claims] == ["claim_1", "claim_2"]
    assert effective_acceptance_claims({}, None) == ([], "")
    assert effective_acceptance_claims(None, {}) == ([], "")


def test_effective_acceptance_claims_accepts_task_like_mapping():
    task = {"task_contract": {"acceptance_claims": ["from contract"]}}
    claims, source = effective_acceptance_claims(task)
    assert source == "ingress_contract"
    assert claims[0]["claim"] == "from contract"


def test_packet_uses_plan_frozen_claims_when_ingress_empty():
    dr = Path(tempfile.mkdtemp())
    _close_green_wave(dr, "acc", "a" * 64, ["game boots", "score persists"])
    ctx = _t.SimpleNamespace(
        task_contract={"requirements": "do X", "expected_output": "42"},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )

    ev = build_task_acceptance_evidence(ctx, llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc")

    packet_claims = ev["task_contract"]["acceptance_claims"]
    assert [c["claim"] for c in packet_claims] == ["game boots", "score persists"]
    assert ev["acceptance_claims_source"] == "plan_review"
    assert ev["__provenance__"]["acceptance_claims_source"] == "host_attested"


def test_packet_prefers_ingress_claims_over_plan_wave():
    dr = Path(tempfile.mkdtemp())
    _close_green_wave(dr, "acc", "a" * 64, ["plan claim"])
    ctx = _t.SimpleNamespace(
        task_contract={"acceptance_claims": [{"id": "adapter_1", "claim": "adapter claim"}]},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )

    ev = build_task_acceptance_evidence(ctx, llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc")

    packet_claims = ev["task_contract"]["acceptance_claims"]
    assert [c["claim"] for c in packet_claims] == ["adapter claim"]
    assert ev["acceptance_claims_source"] == "ingress_contract"


def test_packet_claims_lookup_is_fail_soft():
    dr = Path(tempfile.mkdtemp())
    # Malformed task-result JSON: the plan-state lookup raises inside, packet survives.
    results = dr / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "acc.json").write_text("{not json", encoding="utf-8")
    ctx = _t.SimpleNamespace(
        task_contract={"requirements": "do X"},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )

    ev = build_task_acceptance_evidence(ctx, llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc")

    assert "acceptance_claims_source" not in ev
    assert not ev["task_contract"].get("acceptance_claims")


def test_schedule_subagent_publishes_claims_param_via_ssot():
    from ouroboros.tools import control

    props = control.schedule_subagent_properties()
    claims = props["acceptance_claims"]
    assert claims["type"] == "array"
    assert claims["items"] == {"type": "string"}
    # No min-constraints (v6.65.1/.2) and no empty-string enum members (Gemini 400).
    assert "minItems" not in claims
    assert "enum" not in claims["items"]
    # Handler allowed-keys DERIVE from the same mapping (BIBLE P7).
    assert "acceptance_claims" in control.schedule_subagent_param_names()


def test_child_contract_restates_claims_after_parent_spread():
    from ouroboros.tools.control import _build_child_subagent_contract

    parent_contract = {
        "acceptance_claims": [{"id": "p1", "claim": "parent-only claim"}],
        "success_criteria": ["parent criterion"],
        "deadline_at": "",
        "context": "API_CONTEXT_" + "x" * 900 + "_DECISIVE_TAIL",
    }
    base_spec = {
        "tid": "child1", "objective": "do child work", "expected_output": "a result",
        "constraints": "", "parent_contract": parent_contract,
        "parent_task_id": "parent1", "root_task_id": "root1", "session_id": "s1",
        "child_delegation_budget": None, "deadline_at": "",
    }

    # No child claims: EMPTY is re-stated — parent claims/criteria never leak.
    bare = _build_child_subagent_contract(dict(base_spec))
    assert bare["acceptance_claims"] == []
    assert bare["success_criteria"] == []
    assert bare["context"].endswith("_DECISIVE_TAIL")

    # Explicit child claims land normalized with positional ids.
    claimed = _build_child_subagent_contract(
        {**base_spec, "acceptance_claims": ["module compiles", "tests green"]}
    )
    assert [c["claim"] for c in claimed["acceptance_claims"]] == [
        "module compiles", "tests green",
    ]
    assert [c["id"] for c in claimed["acceptance_claims"]] == ["claim_1", "claim_2"]
    assert claimed["success_criteria"] == []


def test_schedule_subagent_claims_end_to_end(tmp_path, monkeypatch):
    import queue

    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)

    event_queue: queue.Queue = queue.Queue()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = event_queue
    ctx.task_metadata = {
        "root_task_id": "root1", "session_id": "sess1",
        "task_contract": {"acceptance_claims": [{"id": "p1", "claim": "parent claim"}]},
    }

    result = _schedule_task(
        ctx,
        subagent_id=subagent_id,
        objective="Build the collision module",
        expected_output="A working module",
        acceptance_claims=["hull overlap is rejected", "  ", ""],
    )

    assert "TOOL_ARG_ERROR" not in result
    event = event_queue.get_nowait()
    contract = event["task_contract"]
    assert [c["claim"] for c in contract["acceptance_claims"]] == ["hull overlap is rejected"]
    assert contract["success_criteria"] == []


def test_schedule_subagent_rejects_malformed_claims(tmp_path):
    import queue

    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {}

    for bad in ("not-a-list", [{"claim": "object"}], [1, 2]):
        result = _schedule_task(
            ctx, objective="o", expected_output="e", acceptance_claims=bad,
        )
        assert "TOOL_ARG_ERROR (schedule_subagent): acceptance_claims" in result


def test_success_criteria_is_an_input_alias_not_a_second_carrier():
    from ouroboros.contracts.task_contract import build_task_contract

    # Criteria given only as success_criteria arrive normalized into claims.
    aliased = build_task_contract({"task_contract": {"success_criteria": ["report delivered"]}})
    assert [c["claim"] for c in aliased["acceptance_claims"]] == ["report delivered"]
    assert aliased["success_criteria"] == []

    # Real claims win; the alias list is still not double-persisted.
    both = build_task_contract({"task_contract": {
        "success_criteria": ["alias criterion"],
        "acceptance_claims": ["real claim"],
    }})
    assert [c["claim"] for c in both["acceptance_claims"]] == ["real claim"]
    assert both["success_criteria"] == []

    # No criteria anywhere: both carriers empty, exactly as before.
    empty = build_task_contract({"task_contract": {}})
    assert empty["acceptance_claims"] == []
    assert empty["success_criteria"] == []

    # Explicit empty claims beside a criteria list keep the single-carrier rule:
    # claims stay empty and the raw list persists alone (no double carrier).
    explicit_empty = build_task_contract({"task_contract": {
        "success_criteria": ["kept criterion"],
        "acceptance_claims": [],
    }})
    assert explicit_empty["acceptance_claims"] == []
    assert explicit_empty["success_criteria"] == ["kept criterion"]


def test_wave_freeze_and_bind_preserve_reviewed_claim_whitespace():
    """G3-6: the review panel sees the normalized spec — per-item strip with internal
    whitespace PRESERVED. The frozen v2 wave (the whole reviewed spec) and the read-time
    binder must carry that text byte-for-byte, including decisive tails beyond the old cap;
    the historical ``" ".join(split())`` rewrite made acceptance bind an exact-output
    claim DIFFERENT from what the panel reviewed."""
    from ouroboros.task_results import closed_plan_review_wave, load_plan_review_state
    from ouroboros.tools.plan_spec import normalize_spec

    raw = "stdout is exactly:\n    def f():\n        return  'a  b'"
    spec, _errors = normalize_spec({"goal": "g", "acceptance_claims": [f"  {raw}  "]})
    reviewed = [c["claim"] for c in spec["acceptance_claims"]]
    assert reviewed == [raw]  # the panel-reviewed surface preserves internal whitespace

    dr = Path(tempfile.mkdtemp())
    _close_green_wave(dr, "acc", "a" * 64, reviewed)
    wave = closed_plan_review_wave(load_plan_review_state(dr, "acc"))
    assert [c["claim"] for c in wave["spec"]["acceptance_claims"]] == [raw]  # frozen byte-for-byte

    claims, source = effective_acceptance_claims({"acceptance_claims": []}, wave)
    assert source == "plan_review"
    assert [c["claim"] for c in claims] == [raw]  # bound text == reviewed text

    # Review/acceptance authority is not a preview: the old 600-char cut cannot
    # certify a prefix as the complete claim.
    long_claim = "line with    significant\tspacing\n" * 40
    bounded_spec, _ = normalize_spec({"goal": "g", "acceptance_claims": [long_claim]})
    bounded = bounded_spec["acceptance_claims"][0]["claim"]
    assert bounded == long_claim.strip()
    assert "OMISSION NOTE" not in bounded


def test_packet_open_plan_wave_binds_no_claims():
    from ouroboros.review_execution import _render_prompt_parts
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot
    from ouroboros.task_results import STATUS_RUNNING, record_plan_review_wave, write_task_result

    dr = Path(tempfile.mkdtemp())
    write_task_result(dr, "acc", STATUS_RUNNING, result="running")
    record_plan_review_wave(dr, "acc", _v2_wave("a" * 64, ["unreviewed claim"],
                                                aggregate="REVIEW_REQUIRED", closed=False))
    ctx = _t.SimpleNamespace(
        task_contract={"requirements": "do X"},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )

    ev = build_task_acceptance_evidence(ctx, llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc")

    # An OPEN (unclosed) wave's claims never bind acceptance, but AP5 discloses
    # why the contract has no claims instead of silently omitting that fact.
    assert ev["acceptance_claims_source"] == "none_open_plan_wave"
    assert not ev["task_contract"].get("acceptance_claims")
    assert "acceptance_support_refs" not in ev

    # The acceptance prompt keeps the disclosure at packet top level. Its
    # claim-specific reviewer rule names only task_contract.acceptance_claims,
    # so the typed source cannot be mistaken for a binding claim.
    stable, _task, dynamic = _render_prompt_parts(
        ReviewRequest(surface="task_acceptance", goal="do X", evidence=ev),
        ReviewSlot("s0", "m"),
    )
    assert "`task_contract.acceptance_claims` is present" in stable
    assert '"acceptance_claims_source": "none_open_plan_wave"' in dynamic
    assert '"task_contract": {\n    "requirements": "do X"\n  }' in dynamic
