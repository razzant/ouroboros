"""The plan spec's resource form: `affected_paths` is the ONE list the host resolves.

Owner decisions 8=A / 9=A / 16=A, taken after a live wave: the old `affected_resources` list
was read as "everything that is not a URL is a file path", so the prose item «Отдельный проект
«TSMC — 10-летний инвестиционный анализ»» resolved to a path under the Ouroboros repository, the
plan was marked constitutional, and every reviewer of a deck-shaped plan received BIBLE.md plus
ARCHITECTURE.md in full (~470k tokens per cycle). Across the live history not one of the 14
constitutional verdicts came from a real file path in that list.

What is pinned here: a declared CHANGE target decides (existing or not); prose never resolves
and never reaches the filesystem; an evidence READ of a repository file is named but buys
nothing; a spec submitted in the old mixed form is refused before any dispatch without writing
anything; and a wave recorded under the old shape still closes for free.
"""

from __future__ import annotations

import copy
import json

from ouroboros.tools import plan_spec
from tests.test_plan_review_engine import (  # noqa: F401 — `harness` is the fixture
    CLEAN,
    DECK_SPEC,
    _call,
    _control,
    _finding,
    _state,
    harness,
)

# Verbatim from the live wave of task 12102247e791ff1e (structural-health sprint evidence,
# plan-resource-live-examples.json): the list that made a deck plan constitutional.
TSMC_RESOURCES = [
    "Отдельный проект «TSMC — 10-летний инвестиционный анализ»",
    "Файлы исследования, моделей и PDF внутри его project workspace",
    "Финальные пользовательские deliverables",
]


# The three plan-review waves recorded on the live install while `affected_resources` was still
# resolved as a path list (structural-health sprint evidence, plan-resource-live-examples.json:
# tasks 12102247e791ff1e, 0be6fb0690f44bbd, 6596589bb3674aaa). Their resource lists are verbatim;
# the plan bodies stay on the owner's machine. The digests were taken from this module BEFORE
# `affected_paths` existed — a stored wave must keep its identity or the task it belongs to can
# never collect or dispose of it again.
STORED_LEGACY_WAVES = (
    ("12102247e791ff1e",
     ["Отдельный проект «TSMC — 10-летний инвестиционный анализ»",
      "Файлы исследования, моделей и PDF внутри его project workspace",
      "Финальные пользовательские deliverables"],
     [],
     "8832d9db81cbd61dea584c193130e25b3c63fea25b007b7978456f07f92e1f82",
     "40bc4f76e9053c02066ee0900a6ebde424af70450adfe2e2fdf6ccff1b87d337"),
    ("0be6fb0690f44bbd",
     ["/Users/anton/Ouroboros/projects/TSMC___10-____________________________"],
     [],
     "a8aea6bb3adbb7dcb635dbebb7161d238d34168502420b7be4fd4f4c1692f06f",
     "8e652cd2dbbce9ebfa1ecafe147cea4adf41672551995e83659b2e712d869bcb"),
    ("6596589bb3674aaa",
     ["/Users/anton/Ouroboros/data/skills/external/context-lens",
      "/Users/anton/Ouroboros/projects/Context_Lens___________________________________"],
     ["data/skills/external/context-lens/SKILL.md"],
     "2e2764e09bc74383c082bae1eabc44c9acb24c5c8cd4048c4e77187375bbc240",
     "9acfa5071662c07c1f716c5e15a0fd1648c56b64df96a3e5aae1b9289e91fbff"),
)


def test_stored_specs_from_before_affected_paths_keep_their_recorded_identity():
    """`affected_paths` is never defaulted onto input that did not declare it: normalizing a
    spec recorded under the old shape yields the same key list, the same spec hash and the same
    plan fingerprint it was recorded with."""
    for task_id, resources, evidence, spec_digest, fingerprint in STORED_LEGACY_WAVES:
        raw = {"goal": f"stored plan wave {task_id}", "affected_resources": list(resources),
               "evidence": list(evidence)}
        spec, errors = plan_spec.normalize_spec(raw)
        assert errors == []
        assert "affected_paths" not in spec
        assert list(spec) == [
            "goal", "in_scope", "non_goals", "acceptance_claims", "invariants", "decisions",
            "deferred", "affected_resources", "evidence", "normalization_omissions",
        ]
        assert plan_spec.spec_hash(spec) == spec_digest
        assert plan_spec.plan_fingerprint(
            spec["goal"], "stored prose", spec, "manifest-hash", False) == fingerprint


def test_spec_delta_sees_the_affected_paths_claim_appear_and_then_move():
    """Absent ("stored before the field existed") and `[]` ("this work changes no files") are
    different claims, so the delta the next reviewer reads must not call them the same."""
    base = {"goal": "g", "affected_resources": ["the uploader"]}
    legacy, _ = plan_spec.normalize_spec(base)
    declares_none, _ = plan_spec.normalize_spec({**base, "affected_paths": []})
    declares_file, _ = plan_spec.normalize_spec({**base, "affected_paths": ["ouroboros/uploader.py"]})

    appeared = plan_spec.spec_delta(legacy, declares_none)
    assert appeared["changed"] and appeared["declared_keys_changed"]
    assert appeared["prev_hash"] != appeared["hash"]

    moved = plan_spec.spec_delta(declares_none, declares_file)
    assert moved["changed"] and moved["declared_keys_changed"] is False
    assert moved["lists"]["affected_paths"] == {"added": ["ouroboros/uploader.py"], "removed": []}
    assert plan_spec.spec_delta(declares_file, declares_file)["changed"] is False


def _system_prompt(substrate, index=0):
    return substrate.calls[index]["request"].messages[0]["content"][0]["text"]


def test_a_declared_new_source_file_is_constitutional_before_it_exists(harness):  # noqa: F811
    """A file the work will CREATE has no bytes and no parent directory on disk yet; writing
    `ouroboros/new/deep/module.py` IS self-modification, so the full pack must ride."""
    target = harness.system / "ouroboros" / "new" / "deep" / "module.py"
    assert not target.exists() and not target.parent.exists()
    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})

    out = _call(harness.make_ctx(), spec={**DECK_SPEC, "affected_paths": [str(target)]})

    wave = _state(harness)["waves"][-1]
    assert wave["constitutional"] is True
    assert "affected_paths" in wave["constitutional_note"] and str(target) in wave["constitutional_note"]
    system_prompt = _system_prompt(substrate)
    assert "## BIBLE.md (constitution" in system_prompt and "6. Governance" in system_prompt
    assert "## ARCHITECTURE.md (architecture and data flow" in system_prompt
    assert "on-demand pointer" not in system_prompt
    assert "REMINDER" not in out


def test_prose_resources_never_reach_the_filesystem_and_never_buy_the_pack(harness, monkeypatch):  # noqa: F811
    """The TSMC list, exactly as the live wave sent it: descriptions of a project, its files and
    its deliverables. With no change target declared the plan is ordinary, the constitutional
    resolver resolves nothing at all, and the host reminds the agent that the list it CAN use
    for that is `affected_paths`."""
    resolved: list[str] = []
    real = plan_spec._resolve_locator_path
    monkeypatch.setattr(
        plan_spec, "_resolve_locator_path",
        lambda locator, root: (resolved.append(locator), real(locator, root))[1],
    )
    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    spec = {**DECK_SPEC, "affected_paths": [], "affected_resources": list(TSMC_RESOURCES)}

    # `active_workspace=False` binds the task to the system repo itself: the one situation where
    # an empty change-target list is worth a reminder (D29 keeps the binding from deciding).
    out = _call(harness.make_ctx(active_workspace=False), spec=spec)

    assert resolved == []
    wave = _state(harness)["waves"][-1]
    assert wave["constitutional"] is False
    assert wave["constitutional_note"] == (
        "not constitutional: no declared affected_paths locator resolves under the "
        "Ouroboros system repository"
    )
    assert "REMINDER: affected_paths is empty" in out
    system_prompt = _system_prompt(substrate)
    assert "## BIBLE.md (constitution" not in system_prompt and "on-demand pointer" in system_prompt


def test_reading_a_repository_file_is_not_changing_it(harness):  # noqa: F811
    """Owner 16=A. The same locator, twice: as evidence alone it is named in the note and the
    pack stays a pointer; listed as a change target too, it escalates."""
    bible = str(harness.system / "BIBLE.md")
    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})

    _call(harness.make_ctx(), spec={**DECK_SPEC, "affected_paths": [], "evidence": [bible]})

    read_only = _state(harness)["waves"][-1]
    assert read_only["constitutional"] is False
    assert "EVIDENCE reads" in read_only["constitutional_note"] and bible in read_only["constitutional_note"]
    assert "## BIBLE.md (constitution" not in _system_prompt(substrate)

    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx(task_id="task-2"),
          spec={**DECK_SPEC, "affected_paths": [bible], "evidence": [bible]})

    changing = _state(harness, "task-2")["waves"][-1]
    assert changing["constitutional"] is True
    assert "affected_paths" in changing["constitutional_note"]
    assert "## BIBLE.md (constitution" in _system_prompt(substrate)


def test_a_legacy_form_submission_is_refused_and_records_nothing(harness):  # noqa: F811
    """Owner 9=A: the old mixed form is refused BEFORE any dispatch. The refusal quotes the
    field with an example, and — because its code is not the `PLAN_SPEC_INVALID` family that
    supersedes the current attempt — the task's recorded plan state does not move a byte."""
    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx))["closed"] is True
    before = json.dumps(_state(harness), sort_keys=True, ensure_ascii=False)

    legacy = {key: value for key, value in DECK_SPEC.items() if key != "affected_paths"}
    legacy["affected_resources"] = list(TSMC_RESOURCES)
    out = _call(ctx, spec=legacy)

    assert "PLAN_RESOURCE_FORM_REQUIRED" in out and "spec.affected_paths is required" in out
    assert '"affected_paths": ["ouroboros/tools/plan_review.py"]' in out
    assert '"affected_resources": ["the plan-review organ"' in out
    assert len(substrate.calls) == 1  # the refused submission called no reviewer
    assert json.dumps(_state(harness), sort_keys=True, ensure_ascii=False) == before


def test_a_legacy_form_with_another_error_is_still_refused_before_the_superseding_path(harness):  # noqa: F811
    """A legacy-form spec that ALSO fails ordinary validation must take the non-superseding
    refusal, not the `PLAN_SPEC_INVALID` path that records a new attempt over an open wave."""
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", summary="who signs off?")])
    substrate = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx)) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    before = json.dumps(_state(harness), sort_keys=True, ensure_ascii=False)
    calls_before = len(substrate.calls)

    legacy = {key: value for key, value in DECK_SPEC.items() if key != "affected_paths"}
    out = _call(ctx, spec=legacy, reviewer_effort="galactic")

    assert "PLAN_RESOURCE_FORM_REQUIRED" in out and "PLAN_SPEC_INVALID" not in out
    assert len(substrate.calls) == calls_before
    assert json.dumps(_state(harness), sort_keys=True, ensure_ascii=False) == before


def test_a_legacy_form_submission_over_an_open_wave_points_at_the_free_exit(harness):  # noqa: F811
    """The expensive mistake this prevents: re-submitting into a refusal while an OPEN wave is
    still the live obligation. The refusal names that wave and the $0 way to answer it."""
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", summary="who signs off?")])
    harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx)) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    open_fingerprint = _state(harness)["waves"][-1]["request_fingerprint"]
    before = json.dumps(_state(harness), sort_keys=True, ensure_ascii=False)

    legacy = {key: value for key, value in DECK_SPEC.items() if key != "affected_paths"}
    out = _call(ctx, spec=legacy)

    assert "PLAN_RESOURCE_FORM_REQUIRED" in out
    assert open_fingerprint in out and "review_disposition" in out
    assert json.dumps(_state(harness), sort_keys=True, ensure_ascii=False) == before


def test_a_wave_stored_before_affected_paths_still_closes_at_zero_cost(harness):  # noqa: F811
    """No grace mechanism was built for the waves already open (owner, disclosed) — they must
    simply stay answerable. A wave recorded under the old spec shape closes through
    `review_disposition` with no reviewer call, and its recorded identity does not move."""
    from ouroboros import task_results
    from ouroboros.tools import plan_review_artifacts as artifacts
    from ouroboros.tools.plan_review import _apply_disposition

    legacy_spec, errors = plan_spec.normalize_spec({
        "goal": "Deliver the TSMC analysis",
        "affected_resources": list(TSMC_RESOURCES),
    })
    assert errors == [] and "affected_paths" not in legacy_spec
    stored_hash = plan_spec.spec_hash(legacy_spec)
    fingerprint = "c" * 64
    wave = {
        "schema_version": 2, "cycle_index": 1, "request_fingerprint": fingerprint,
        "goal": legacy_spec["goal"], "spec": legacy_spec, "spec_hash": stored_hash,
        "aggregate": "REVIEW_REQUIRED", "closed": False, "paid": True, "dispositions": [],
        "findings": [{"finding_id": "s1:f1", "id": "f1", "class": "need_evidence", "breaks": "goal",
                      "locator": "", "summary": "who signs this off?", "recommendation": ""}],
    }
    artifacts.record_exact_wave(
        harness.drive, "task-1", wave, copy.deepcopy(wave), need_evidence_seen=[], page_size=32,
    )
    task_results.record_plan_review_attempt(
        harness.drive, "task-1", fingerprint=fingerprint, status="open",
    )
    paid_before = int(_state(harness).get("cycles_paid") or 0)
    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})

    out = _apply_disposition(harness.make_ctx(), {"review_fingerprint": fingerprint, "items": [
        {"finding_id": "s1:f1", "decision": "accept", "rationale": "the owner signs it off"},
    ]})

    assert _control(out) == {"outcome": "REVIEW_REQUIRED", "closed": True}
    assert substrate.calls == []
    state = _state(harness)
    assert int(state.get("cycles_paid") or 0) == paid_before
    closed = state["waves"][-1]
    assert closed["request_fingerprint"] == fingerprint and closed["closed"] is True
    assert closed["spec_hash"] == stored_hash and closed["aggregate"] == "REVIEW_REQUIRED"
    assert "affected_paths" not in closed["spec"]
