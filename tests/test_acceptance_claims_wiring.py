"""W2 claims wiring: the ONE pure seam ``effective_acceptance_claims`` and its two
consumers — the one read-time binder ``review_evidence_sections._accept_effective_claims``
that the result review's subject reads (ingress-contract claims first, the CLOSED plan
wave's frozen claims only when ingress is empty) and the child-contract builder (claims
re-stated AFTER the **parent_contract spread). The running task contract is NEVER
mutated: plan-frozen claims live in plan_review_state and are resolved at read time."""

import tempfile
import types as _t
from pathlib import Path

from ouroboros.contracts.task_contract import effective_acceptance_claims
from ouroboros.review_evidence_sections import _accept_effective_claims, _accept_task_contract


def _v2_wave(fingerprint: str, claims: list, *, aggregate: str = "GREEN", closed: bool = True) -> dict:
    """A finished legacy (v2) wave as the retired engine stored it: its spec's claims in the
    acceptance claim grammar (the normalizer the spec parser used)."""
    from ouroboros.contracts.task_contract import normalize_acceptance_claims
    from tests import _legacy_plan_shared as legacy

    spec = {"goal": "g", "acceptance_claims": normalize_acceptance_claims(claims)}
    return legacy.wave(1, fingerprint=fingerprint, spec=spec, aggregate=aggregate, closed=closed)


def _close_green_wave(root: Path, task_id: str, fingerprint: str, claims: list) -> None:
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from tests import _legacy_plan_shared as legacy

    write_task_result(root, task_id, STATUS_RUNNING, result="running")
    legacy.seed_plan_review_state(root, task_id, legacy.v2_state(_v2_wave(fingerprint, claims)))


def _bind(ctx, root: Path, task_id: str):
    """The read-time binder over the task's contract, as the result review's subject reads it."""
    return _accept_effective_claims(ctx, _accept_task_contract(ctx), root, task_id)


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


def test_binder_uses_plan_frozen_claims_when_ingress_empty():
    dr = Path(tempfile.mkdtemp())
    _close_green_wave(dr, "acc", "a" * 64, ["game boots", "score persists"])
    ctx = _t.SimpleNamespace(
        task_contract={"requirements": "do X", "expected_output": "42"},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )

    claims, source, _exhibit = _bind(ctx, dr, "acc")

    assert [c["claim"] for c in claims] == ["game boots", "score persists"]
    assert source == "plan_review"
    assert "acceptance_claims" not in ctx.task_contract  # resolved at read time, never written back


def test_binder_prefers_ingress_claims_over_plan_wave():
    dr = Path(tempfile.mkdtemp())
    _close_green_wave(dr, "acc", "a" * 64, ["plan claim"])
    ctx = _t.SimpleNamespace(
        task_contract={"acceptance_claims": [{"id": "adapter_1", "claim": "adapter claim"}]},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )

    claims, source, _exhibit = _bind(ctx, dr, "acc")

    assert [c["claim"] for c in claims] == ["adapter claim"]
    assert source == "ingress_contract"


def test_binder_claims_lookup_is_fail_soft():
    dr = Path(tempfile.mkdtemp())
    # Malformed task-result JSON: the plan-state lookup raises inside, the binder survives.
    results = dr / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "acc.json").write_text("{not json", encoding="utf-8")
    ctx = _t.SimpleNamespace(
        task_contract={"requirements": "do X"},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )

    claims, source, _exhibit = _bind(ctx, dr, "acc")

    assert source == ""
    assert not claims


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


def test_wave_and_plan_bind_preserve_reviewed_claim_whitespace():
    """G3-6: a claim is stripped at its edges with internal whitespace PRESERVED. A finished
    legacy wave's reviewed claim and a plan revision's ``done_when`` must bind byte-for-byte,
    including decisive tails beyond the old cap; the historical ``" ".join(split())``
    rewrite made acceptance bind an exact-output claim DIFFERENT from what was reviewed."""
    from ouroboros import plan_record
    from ouroboros.contracts.task_contract import normalize_acceptance_claims
    from ouroboros.task_results import current_plan_review_wave, load_plan_review_state

    raw = "stdout is exactly:\n    def f():\n        return  'a  b'"
    dr = Path(tempfile.mkdtemp())
    _close_green_wave(dr, "acc", "a" * 64, [f"  {raw}  "])
    wave = current_plan_review_wave(load_plan_review_state(dr, "acc"))
    assert [c["claim"] for c in wave["spec"]["acceptance_claims"]] == [raw]  # stored byte-for-byte

    claims, source = effective_acceptance_claims({"acceptance_claims": []}, wave)
    assert source == "plan_review"
    assert [c["claim"] for c in claims] == [raw]  # bound text == reviewed text
    ctx = _t.SimpleNamespace(task_contract={"requirements": "do X"}, task_metadata={}, drive_root=str(dr),
                             task_id="acc", repo_dir=str(dr))
    assert [c["claim"] for c in _bind(ctx, dr, "acc")[0]] == [raw]
    assert [row["claim"] for row in plan_record.normalize_done_when([{"claim": f"  {raw}  "}])] == [raw]

    # Review/acceptance authority is not a preview: no cut can certify a prefix as the
    # complete claim.
    long_claim = "line with    significant\tspacing\n" * 40
    bounded = normalize_acceptance_claims([long_claim])[0]["claim"]
    assert bounded == long_claim.strip()
    assert "OMISSION NOTE" not in bounded


def test_binder_open_plan_wave_binds_no_claims():
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from tests import _legacy_plan_shared as legacy

    dr = Path(tempfile.mkdtemp())
    write_task_result(dr, "acc", STATUS_RUNNING, result="running")
    legacy.seed_plan_review_state(dr, "acc", legacy.v2_state(_v2_wave("a" * 64, ["unreviewed claim"],
                                                                      aggregate="REVIEW_REQUIRED", closed=False)))
    ctx = _t.SimpleNamespace(
        task_contract={"requirements": "do X"},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )

    claims, source, exhibit = _bind(ctx, dr, "acc")

    # A legacy v2 wave the old organ never closed selected nothing: the one binder binds
    # no claim, and the plan-history exhibit discloses the legacy record instead of the
    # binding silently omitting the fact.
    assert source == "" and not claims
    assert exhibit["binding"] == "none" and exhibit["revisions"] == [] and exhibit["legacy"]["fingerprint"] == "a" * 64
