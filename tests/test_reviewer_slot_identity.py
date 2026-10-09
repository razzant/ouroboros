"""Reviewer-slot identity: one row, one id, minted in exactly one place.

The scope surface was fixed in v6.87.22. Its twin — the commit triad, the gate
every commit passes through — still ran a row under one id and wrote a durable
actor record under another. These tests pin both halves of the contract:

  * the durable record CARRIES the id the substrate ran, it does not re-derive
    one from the record's position (position is not row whenever an oversized
    skill review merges several chunk passes into one results list);
  * every configured-reviewer surface takes that id from ``slot_id_for_row``,
    so the "minted in exactly one place" claim in ARCHITECTURE.md is testable
    rather than aspirational.
"""

import json
import pathlib
import re
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parent.parent


def _fake_ctx(tmp_path):
    return SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="slot-identity",
        pending_events=[], drive_logs=lambda: tmp_path,
    )


def _substrate_stub(recorder, *, raise_for=()):
    """Stand in for ``run_review_request``, recording the id each row ran under."""
    rows = [{"item": "x", "verdict": "PASS", "severity": "advisory", "reason": "checked"}]

    def _run(request, *, slots, drive_root, llm, usage_ctx=None):
        # The triad dispatches one slot per call; plan review dispatches the
        # whole row list in one. Record whatever this call was asked to run.
        recorder.extend(slot.slot_id for slot in slots)
        if any(slot.model in raise_for for slot in slots):
            raise RuntimeError("transport died")
        return SimpleNamespace(actors=[{
            "slot_id": slot.slot_id,
            "model": slot.model,
            "status": "ok",
            "raw_text": json.dumps(rows),
            "usage": {},
            "prompt_ref": {},
            "response_ref": {},
        } for slot in slots])

    return _run


def _merged_chunk_results(ctx, models, *, chunks, raise_for=()):
    """Drive the real triad fan-out once per chunk and union the results.

    Mirrors ``skill_review_passes`` over an oversized skill: C passes across M
    configured rows are merged into ONE ``{"results": [...]}`` envelope, so the
    Nth entry is row ``N % M``, not row ``N``.
    """
    from ouroboros.tools import review

    merged = []
    for part in range(chunks):
        out = json.loads(review._handle_multi_model_review(
            ctx, content=f"PART {part + 1} of {chunks}", prompt="p", models=list(models),
        ))
        merged.extend(out.get("results") or [])
    return merged


def test_durable_actor_record_carries_the_row_the_substrate_ran(tmp_path, monkeypatch):
    """One row, one durable id — even when position stops matching row.

    Before this fix the record re-derived ``slot_{position + 1}``: over 2 rows x 2
    chunk passes it stamped slot_1..slot_4, naming two rows that are not
    configured and giving the SAME configured row two different identities, while
    the prompt/response refs it points at said slot_1/slot_2.
    """
    from ouroboros import review_substrate
    from ouroboros.tools import review

    ran_as: list = []
    monkeypatch.setattr(
        review_substrate, "run_review_request",
        _substrate_stub(ran_as, raise_for={"m/two"}),
    )
    ctx = _fake_ctx(tmp_path)
    merged = _merged_chunk_results(ctx, ["m/one", "m/two"], chunks=2)

    review._collect_review_findings(ctx, merged)
    durable = [str(r.get("slot_id") or "") for r in (ctx._last_triad_raw_results or [])]

    # The substrate ran two rows, twice. The durable records must say the same.
    assert ran_as == ["slot_1", "slot_2", "slot_1", "slot_2"], ran_as
    assert durable == ran_as, durable
    # Nothing may name a row that was never configured.
    assert set(durable) == {"slot_1", "slot_2"}, durable
    # The failing row is still a row: its errored record carries its own id, not
    # a position. (m/two raised transport, so it is the error envelope.)
    errored = [r for r in ctx._last_triad_raw_results if r.get("status") == "error"]
    assert errored, "expected the raising row to produce an error actor record"
    assert {r["slot_id"] for r in errored} == {"slot_2"}, errored


def _repoint_the_mint(monkeypatch):
    """Repoint the ONE mint. Every surface that reads it must follow."""
    from ouroboros import review_substrate
    from ouroboros.tools import review

    def _fake(index, *, prefix=review_substrate.SLOT_ID_PREFIX):
        return f"{prefix}_row{int(index)}"

    monkeypatch.setattr(review_substrate, "slot_id_for_row", _fake)
    # review.py binds the mint by name at import time; repoint that binding too,
    # so a module that spelled its own literal instead still fails this test.
    monkeypatch.setattr(review, "slot_id_for_row", _fake)


def test_triad_row_ids_come_from_the_one_mint(tmp_path, monkeypatch):
    """The commit triad must not spell its own row id (it said multi_model_slot_N,
    disagreeing with the slot_N its own durable records carried)."""
    from ouroboros import review_substrate
    from ouroboros.tools import review

    ran_as: list = []
    monkeypatch.setattr(review_substrate, "run_review_request", _substrate_stub(ran_as))
    _repoint_the_mint(monkeypatch)

    ctx = _fake_ctx(tmp_path)
    merged = _merged_chunk_results(ctx, ["m/one", "m/two"], chunks=1)
    review._collect_review_findings(ctx, merged)
    durable = [str(r.get("slot_id") or "") for r in (ctx._last_triad_raw_results or [])]

    assert ran_as == ["slot_row1", "slot_row2"], ran_as
    assert durable == ran_as, durable


# Plan review's rows are the review POOL's catalog rows (PR-3): their ids are
# the rows' own stored ids, not minted — see ``tests/test_review_pool.py``
# (``test_plan_review_rows_are_the_pool_rows_and_duplicate_models_stay_distinct``).


def test_skill_review_dispatches_through_the_slot_id_stamping_entry():
    """The to_dict positional fallback in ReviewActorRecord is LEGACY-READ-ONLY:
    that holds only while every live producer stamps slot_id. skill_review's one
    dispatch is review._handle_multi_model_review — the stamping entry the chunk
    -merge identity tests drive — so pin the call site itself."""
    source = (REPO / "ouroboros" / "skill_review.py").read_text(encoding="utf-8")
    assert source.count("run_review=") == 1, "skill_review grew a second review dispatch"
    assert "run_review=_handle_multi_model_review" in source
    assert "from ouroboros.tools.review import _handle_multi_model_review" in source


# An interpolated slot id built anywhere but the mint is the defect class itself:
# it is a second identity for a row that already has one. Constant sentinels
# (``slot_id="scope_slot_error"``) name no row and are not matched.
_INTERPOLATED_SLOT_ID = re.compile(r"""slot_id["']?\s*[:=]\s*f["'][^"']*\{""")


def test_no_surface_derives_a_row_id_outside_the_one_mint():
    """ARCHITECTURE.md says row identity is minted in exactly one place. This is
    the check that keeps that true when the next reviewer surface is added."""
    offenders = []
    for path in sorted((REPO / "ouroboros").rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "slot_id_for_row" in line:
                continue
            if _INTERPOLATED_SLOT_ID.search(line):
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{lineno}: {line.strip()}")
    assert not offenders, (
        "row identity must come from review_substrate.slot_id_for_row, not a local "
        "f-string:\n" + "\n".join(offenders)
    )
