"""v6.71.1 acceptance-review convergence: what outlives task acceptance.

The ONE claims binder (AP5) — ingress claims win, else the plan's latest promises, an open
legacy wave binds nothing — is the read-time binder the result review's subject reads
(``review_evidence_sections._accept_effective_claims``). An old acceptance decision whose
forced-rail reason closed a dangling revision keeps its best-effort objective. The rebuttal
wiring, partial coherence, evidence parity and packet budget of the acceptance panel are
retired with task acceptance."""

from __future__ import annotations

import tempfile
import types as _t
from pathlib import Path

from ouroboros.review_evidence_sections import _accept_effective_claims, _accept_task_contract


def _acc_ctx(dr: Path):
    return _t.SimpleNamespace(
        task_contract={"requirements": "do X", "expected_output": "42"},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )


def _bind(ctx, root: Path, task_id: str):
    """The read-time binder over the task's contract, as the result review's subject reads it."""
    return _accept_effective_claims(ctx, _accept_task_contract(ctx), root, task_id)


# ── AP5: ONE binder — ingress claims win, else the plan's latest promises; an open legacy wave binds nothing ──

def _record_wave(root: Path, task_id: str, *, closed: bool, cycle: int = 1) -> None:
    """A legacy (v2) wave as finished installs hold it, planted directly (the retired writer is gone)."""
    from tests import _legacy_plan_shared as legacy

    legacy.seed_plan_review_state(root, task_id, legacy.v2_state(legacy.wave(
        cycle, spec={"goal": "g", "acceptance_claims": [
            {"id": "claim_1", "claim": "the widget renders", "priority": "must"},
            {"id": "claim_2", "claim": "the counter increments", "priority": "should"},
        ]}, aggregate="GREEN" if closed else "REVISE_PLAN", closed=closed)))


def test_an_open_plan_wave_is_disclosed_as_binding_nothing():
    """A legacy (v2) wave still open at upgrade selected nothing: no source, no claims, and the
    exhibit says so (``binding: none``)."""
    dr = Path(tempfile.mkdtemp())
    _record_wave(dr, "acc", closed=False)
    ctx = _acc_ctx(dr)
    claims, source, exhibit = _bind(ctx, dr, "acc")
    assert source == ""
    assert exhibit["binding"] == "none" and exhibit["revisions"] == [] and exhibit["open_objections"] == []
    # Nothing was bound: no claims, and the task's contract keeps none.
    assert claims == []
    assert "acceptance_claims" not in ctx.task_contract


def test_a_closed_legacy_wave_still_binds_and_the_exhibit_names_its_provenance():
    dr = Path(tempfile.mkdtemp())
    _record_wave(dr, "acc", closed=True)
    claims, source, exhibit = _bind(_acc_ctx(dr), dr, "acc")
    assert source == "plan_review"
    assert exhibit["binding"] == "legacy_closed_wave"
    assert [row["id"] for row in claims] == ["claim_1", "claim_2"]


def test_ingress_claims_win_over_an_open_wave_and_the_exhibit_says_so():
    dr = Path(tempfile.mkdtemp())
    _record_wave(dr, "acc", closed=False)
    ctx = _t.SimpleNamespace(
        task_contract={
            "requirements": "do X", "expected_output": "42",
            "acceptance_claims": [{"id": "ingress_1", "claim": "the ingress claim"}],
        },
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )
    claims, source, exhibit = _bind(ctx, dr, "acc")
    assert source == "ingress_contract"
    assert exhibit["binding"] == "ingress_contract"
    assert "adopted_promises" not in exhibit  # no plan revision: nothing was adopted
    assert [row["id"] for row in claims] == ["ingress_1"]


def test_a_task_with_no_wave_at_all_stays_silent():
    dr = Path(tempfile.mkdtemp())
    assert _bind(_acc_ctx(dr), dr, "acc") == ([], "", {})


# ── AP6: an old forced-rail decision keeps its objective best effort ─────────

def test_the_terminal_pair_keeps_the_objective_best_effort():
    from ouroboros.outcomes import _ACCEPTANCE_BLOCKED_TERMINAL_REASONS

    assert "revision_unavailable_on_forced_rail" not in _ACCEPTANCE_BLOCKED_TERMINAL_REASONS
