"""Focused regressions for the bounded synthesis runtime review fixes."""

from __future__ import annotations

import pathlib
import re
from types import SimpleNamespace

import pytest
from tests._governance_docs_shared import governance_doc_text


@pytest.fixture(autouse=True)
def _clear_delegate_custody_memo():
    from ouroboros import delegate_custody as custody

    custody._CUSTODY.clear()
    yield
    custody._CUSTODY.clear()


_CHILD = {"parent_task_id": "root-1", "root_task_id": "root-1", "delegation_role": "subagent"}
_OPEN_AUDIT = {"trigger": "loop_exit", "audit_status": "ok", "unreconciled": ["run-open"],
               "open_run_ids": ["run-open"], "pending_invocation_ids": [], "undisposed_patch_run_ids": [],
               "terminal_runs": []}


@pytest.mark.parametrize(("reason", "execution", "headline", "healed"), [
    ("completed", {"status": "ok", "reason_code": ""}, "Done with warnings", "Done"),
    ("provider_unavailable", {"status": "infra_failed", "reason_code": "provider_unavailable"}, "Failed", "Failed"),
    ("round_limit", {"status": "best_effort", "reason_code": "round_limit"}, "Done with warnings", "Done with warnings"),
], ids=["ordinary", "provider-failure", "truncation-rail"])
def test_an_open_custody_debt_stores_beside_the_cause_it_never_replaces(tmp_path, reason, execution, headline, healed):
    """#1583: an unreconciled own run is a DEBT beside the task's cause, never its cause.

    Through the real store path, the row keeps the derived reason and every earned
    axis byte-for-byte (no custody warning is added, nothing is laundered into an
    infrastructure failure, a truncation rail keeps its code), while the current
    audit lands as the row's own list and envelope, the facts that heal."""
    import copy

    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict, completion_status_label
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

    write_task_result(tmp_path, "t-debt", STATUS_RUNNING)
    derived = {"reason_code": reason, "outcome_axes": {
        "execution": dict(execution), "objective": {"status": "met", "source": "review"},
        "review": {"status": "changes_requested", "trigger": "acceptance"}, "artifacts": {"status": "present"}}}
    before = copy.deepcopy(derived)
    _store_task_result(SimpleNamespace(drive_root=tmp_path), {**_CHILD, "id": "t-debt", "budget_drive_root": str(tmp_path)},
                       "the answer", {"delegate_terminal_reconciliation": dict(_OPEN_AUDIT)}, {}, loop_outcome=derived)
    stored = load_task_result(tmp_path, "t-debt")
    assert derived == before, "the caller's derived outcome is not mutated"
    assert stored["reason_code"] == reason
    axes = stored["outcome_axes"]
    earned = normalize_outcome_axes({"outcome_axes": before["outcome_axes"]})
    assert {key: axes[key] for key in ("objective", "review")} == {key: earned[key] for key in ("objective", "review")}
    assert "warnings" not in axes["objective"], "no custody warning is added to what the task earned"
    assert axes["execution"]["status"] == execution["status"]
    assert axes["execution"].get("reason_code", "") == execution["reason_code"]
    assert stored["delegated_runs_unreconciled"] == ["run-open"]
    assert stored["delegate_terminal_reconciliation"]["open_run_ids"] == ["run-open"]
    assert completion_status_label(stored, {}) == headline
    assert _completion_verdict(stored, {}).endswith(TASK_CAUSE_PHRASES["delegated_custody_unreconciled"])
    # The write-side heal rewrites only the list; the earned cause and axes then speak alone.
    write_task_result(tmp_path, "t-debt", stored["status"], delegated_runs_unreconciled=[])
    healed_row = load_task_result(tmp_path, "t-debt")
    assert healed_row["reason_code"] == reason and healed_row["outcome_axes"] == axes
    assert completion_status_label(healed_row, {}) == healed
    assert TASK_CAUSE_PHRASES["delegated_custody_unreconciled"] not in _completion_verdict(healed_row, {})


def test_delegate_start_recipes_match_the_fresh_start_schema():
    from ouroboros.tools import delegate

    schema = next(
        entry.schema for entry in delegate.get_tools() if entry.name == "delegate_start"
    )["parameters"]
    assert schema["required"] == []  # retry_of may omit prompt (#1614); recipes below still name it
    assert not ({"anyOf", "oneOf", "allOf"} & schema.keys())
    assert "required for direct starts including continue_from" in schema["properties"]["subagent_id"]["description"]
    assert "Omit subagent_id or receive a typed conflict" in (
        schema["properties"]["retry_of"]["description"]
    )

    repo = pathlib.Path(__file__).parents[1]
    recipe_paths = (
        "prompts/SYSTEM.md",
        "docs/ARCHITECTURE.md",
        "docs/DEVELOPMENT.md",
        # #447 stage 3: the standing disclosures (which carry the delegate_start
        # recipe) moved to the binding archive; its recipes must stay
        # schema-valid too. The live checklist has none TODAY, so it is scanned
        # tolerantly below rather than dropped from coverage.
        "docs/CHECKLISTS_ARCHIVE.md",
        # v7 D01 split: the dispatch-note pair (and its delegate_start recipe
        # strings) moved to the agent_dispatch leaf; sdn re-exports the pair.
        "ouroboros/agent_dispatch.py",
        "ouroboros/tools/control.py",
        "ouroboros/tools/delegate.py",
        "ouroboros/tools/delegate_integration.py",
    )
    # The live checklist may legitimately have zero recipes (they moved to the
    # archive), but any it GAINS must stay schema-valid. prompts/SYSTEM.md is
    # tolerant for the same reason since the prompt audit: the recipe lives in
    # the delegate_start schema (the SSOT sent every round), and the prompt
    # only names the lane — a copy there would be the duplication class the
    # audit removed. Any recipe the prompt GAINS must still be schema-valid.
    # The two reference books are map-level: whether they carry a recipe at all
    # is prose, not contract, so they are tolerant too (decision 7A); any recipe
    # they do carry must still be schema-valid.
    tolerant = {
        "docs/CHECKLISTS.md", "prompts/SYSTEM.md",
        "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md",
    }
    for relative in (*recipe_paths, "docs/CHECKLISTS.md"):
        text = governance_doc_text(relative, repo)
        recipes = re.findall(r"\bdelegate_start\(([^)]*)\)", text, flags=re.DOTALL)
        assert recipes or relative in tolerant, (
            f"expected at least one delegate_start recipe in {relative}"
        )
        for recipe in recipes:
            assert re.search(r"\bprompt\s*=", recipe), (relative, recipe)
            if not re.search(r"\bretry_of\s*=", recipe):
                # continue_from is the schema's third start selector: a NEW run
                # bound to a finished predecessor.
                assert "continue_from" in schema["properties"]
                direct_selector = re.search(
                    r"\b(?:subagent_id|continue_from)\s*=", recipe)
                actor_first_snapshot = re.fullmatch(
                    r"\s*prompt\s*=\s*(['\"])\1\s*", recipe,
                )
                assert direct_selector or actor_first_snapshot, (relative, recipe)


def test_delegate_recovery_uses_platform_pid_probe(monkeypatch):
    import ouroboros.delegate_recovery as recovery
    import ouroboros.platform_layer as platform_layer

    seen = []
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: seen.append(pid) or True)
    assert recovery._pid_alive(4312) is True
    assert seen == [4312]
