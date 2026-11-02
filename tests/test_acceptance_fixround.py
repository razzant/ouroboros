"""Survivors of the first P3 acceptance-packet fix round.

The acceptance packet, its evidence-ref vocabulary, its budget ladder and its
zero-physical refusals are gone with the task-acceptance machine. Two fixes of
that round outlive it: a root task's reviewed skills are named from a bounded
index (today the review ledger's hot index, one row per ``surface="skill"``
record) never from whole per-skill histories, and a refused review seat reuses
the prompt record it already persisted.
"""

import json
from pathlib import Path


def test_root_task_skill_lookup_reads_the_ledger_index_never_a_skill_history(monkeypatch, tmp_path):
    from ouroboros import review_ledger
    from ouroboros.skill_readiness import _skill_names_from_review_history
    from tests.test_review_ledger import _answer, _facts, _raw

    skill_dir = tmp_path / "state" / "skills" / "large-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "review_history.jsonl").write_text(
        (json.dumps({"root_task_id": "root-wanted", "padding": "x" * 200}) + "\n") * 20_000, encoding="utf-8")
    rows = [_raw("s1", "openai/gpt-5", parts=("skill",), answers={"skill": _answer("skill")})]
    subject = {"root_kind": "skill_payload", "root": "/skills/external/large-skill", "kind": "skill",
               "skill": "large-skill", "base": "", "head": "", "tree_sha": "h" * 64, "diff_sha": "d" * 64, "checkout": ""}
    record = review_ledger.build_wave_record(_facts(rows, task_id="root-wanted", subject=subject), surface="skill",
                                             drive_root=tmp_path)
    # The wave merges its subject's own identity (review_change._finish_record): the skill's name rides the index row.
    record.subject = {**dict(record.subject), "skill": "large-skill", "content_hash": "h" * 64}
    review_ledger.write_record(tmp_path, record)
    original = Path.read_text

    def _guarded_read(path, *args, **kwargs):
        if path.name == "review_history.jsonl":
            raise AssertionError("the root-task lookup rebuilt the full skill review history")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guarded_read)
    history = _skill_names_from_review_history(tmp_path, "root-wanted")
    assert history["names"] == ["large-skill"]
    assert history["coverage"]["complete"] is True
    assert history["coverage"]["source_ref"]["kind"] == "review_ledger_index"
    assert _skill_names_from_review_history(tmp_path, "some-other-root")["names"] == []


def test_refused_slot_reuses_the_already_persisted_prompt(monkeypatch, tmp_path):
    """A seat refused before its physical send (here: the owner deadline is already
    spent) carries the prompt record it persisted, never a second synthetic one."""
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
        deadline_at="2020-01-01T00:00:00+00:00",
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
