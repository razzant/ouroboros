"""X3 (owner 11=B): hash-bound skill repair — immutable admission hash, CAS on
every payload write, typed stale terminalization (no restore promises), and
repair receipts that never invent task ids."""

from __future__ import annotations

import pathlib


from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.skill_repair_admission import (
    STATUS_STALE,
    advance_repair_expected_hash,
    load_repair_admission,
    record_repair_admission,
    repair_write_cas_error,
)


def _seed_skill(drive_root: pathlib.Path, name: str = "alpha") -> pathlib.Path:
    payload = drive_root / "skills" / "external" / name
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
    (payload / "main.py").write_text("print('v1')\n", encoding="utf-8")
    return payload


def _constraint(name: str = "alpha") -> TaskConstraint:
    return TaskConstraint(mode="skill_repair", skill_name=name,
                          payload_root=f"skills/external/{name}",
                          allow_enable=False, allow_review=True)


def _hash(payload: pathlib.Path) -> str:
    from ouroboros.skill_loader import compute_content_hash

    return compute_content_hash(payload)


class TestCasChain:
    def test_clean_chain_allows_writes_and_advances(self, tmp_path):
        payload = _seed_skill(tmp_path)
        constraint = _constraint()
        record_repair_admission(tmp_path, "alpha", task_id="repair-1",
                                base_content_hash=_hash(payload))
        assert repair_write_cas_error(tmp_path, constraint, task_id="repair-1") == ""
        # The repair's own write, then the chain re-pins on the new state.
        (payload / "main.py").write_text("print('v2')\n", encoding="utf-8")
        advance_repair_expected_hash(tmp_path, constraint, task_id="repair-1")
        assert repair_write_cas_error(tmp_path, constraint, task_id="repair-1") == ""
        record = load_repair_admission(tmp_path, "alpha")
        assert record["expected_content_hash"] == _hash(payload)
        # base stays IMMUTABLE — forensics keep the admitted state.
        assert record["base_content_hash"] != record["expected_content_hash"]

    def test_foreign_drift_is_typed_stale_terminal(self, tmp_path):
        payload = _seed_skill(tmp_path)
        constraint = _constraint()
        record_repair_admission(tmp_path, "alpha", task_id="repair-1",
                                base_content_hash=_hash(payload))
        # A CONCURRENT actor edits the payload mid-repair.
        (payload / "main.py").write_text("print('foreign')\n", encoding="utf-8")
        error = repair_write_cas_error(tmp_path, constraint, task_id="repair-1")
        assert "SKILL_REPAIR_STALE" in error
        assert "fresh repair" in error
        # NO restore promises: last_known_good has no payload bytes.
        assert "No restore is possible" in error
        assert load_repair_admission(tmp_path, "alpha")["status"] == STATUS_STALE
        # The terminalization is sticky: further writes stay refused.
        assert "SKILL_REPAIR_STALE" in repair_write_cas_error(
            tmp_path, constraint, task_id="repair-1")

    def test_foreign_lanes_are_not_blocked(self, tmp_path):
        # Proportionality: the repair verifies ITS OWN chain; another task (or
        # the owner's light-mode edit lane) is never gated here.
        payload = _seed_skill(tmp_path)
        constraint = _constraint()
        record_repair_admission(tmp_path, "alpha", task_id="repair-1",
                                base_content_hash=_hash(payload))
        (payload / "main.py").write_text("print('foreign')\n", encoding="utf-8")
        assert repair_write_cas_error(tmp_path, constraint, task_id="other-task") == ""
        assert repair_write_cas_error(tmp_path, constraint, task_id="") == ""

    def test_no_admission_record_means_no_enforcement_for_ordinary_lanes(self, tmp_path):
        # An ordinary payload-editing lane (a short-form bucket+skill_name
        # selector synthesizes the SAME constraint shape) needs no admission.
        _seed_skill(tmp_path)
        assert repair_write_cas_error(tmp_path, _constraint(), task_id="repair-1") == ""

    def test_missing_admission_fails_closed_for_a_repair_task(self, tmp_path):
        # F8: the repair task itself must never write unverified. Without a record
        # every CAS check no-opped, so a repair whose admission was lost ran fully
        # unbound — the exact fail-open this mechanism exists to remove.
        _seed_skill(tmp_path)
        error = repair_write_cas_error(tmp_path, _constraint(), task_id="repair-1",
                                       repair_task=True)
        assert "SKILL_REPAIR_STALE" in error and "NO readable" in error
        # Sticky: nothing changes, so the next write is refused identically.
        assert "SKILL_REPAIR_STALE" in repair_write_cas_error(
            tmp_path, _constraint(), task_id="repair-1", repair_task=True)

    def test_superseded_repair_task_is_refused(self, tmp_path):
        payload = _seed_skill(tmp_path)
        # A FRESH repair took the binding; the older task's writes are no longer
        # accepted (the record's own docstring promised this; the code allowed it).
        record_repair_admission(tmp_path, "alpha", task_id="repair-2",
                                base_content_hash=_hash(payload))
        error = repair_write_cas_error(tmp_path, _constraint(), task_id="repair-1",
                                       repair_task=True)
        assert "SKILL_REPAIR_STALE" in error and "SUPERSEDED" in error
        # The NEWER repair keeps working.
        assert repair_write_cas_error(tmp_path, _constraint(), task_id="repair-2",
                                      repair_task=True) == ""

    def test_unreadable_payload_fails_closed(self, tmp_path):
        payload = _seed_skill(tmp_path)
        constraint = _constraint()
        record_repair_admission(tmp_path, "alpha", task_id="repair-1",
                                base_content_hash=_hash(payload))
        import shutil

        shutil.rmtree(payload)
        # A deleted payload reads either as drift (empty-tree hash) or as
        # unverifiable, depending on how the hasher answers absence — both are
        # the SAME typed stale terminalization, never a silent write.
        error = repair_write_cas_error(tmp_path, constraint, task_id="repair-1")
        assert "SKILL_REPAIR_STALE" in error


class TestAdmissionCapture:
    def test_promoted_repair_admission_binds_base_hash_and_real_task_id(self, tmp_path, monkeypatch):
        import supervisor.workers as workers

        monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
        payload = _seed_skill(tmp_path)
        canonical, error = workers._canonical_promoted_repair_constraint({
            "mode": "skill_repair", "skill_name": "alpha",
            "payload_root": "skills/external/alpha",
            "allow_enable": True,
        })
        assert error == "" and canonical is not None
        assert canonical["allow_enable"] is False
        assert canonical["_base_content_hash"] == _hash(payload)
        # The binding site pops the hash and records the admission keyed by the
        # REAL task id (exercised end-to-end by the promote-flow tests); here we
        # prove the record round-trips.
        record_repair_admission(tmp_path, "alpha", task_id="repair77",
                                base_content_hash=canonical.pop("_base_content_hash"))
        record = load_repair_admission(tmp_path, "alpha")
        assert record["task_id"] == "repair77"
        assert record["base_content_hash"] == _hash(payload)
        # The stored constraint stays canonical (no smuggled keys).
        assert "_base_content_hash" not in canonical

    def test_unwritable_admission_refuses_promotion(self, tmp_path, monkeypatch):
        # F8: a repair admitted WITHOUT its binding CAS-checks nothing. The
        # promote seam must refuse it exactly like an unreadable payload, instead
        # of logging a warning and enqueuing a drift-blind repair.
        import supervisor.workers as workers

        monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
        _seed_skill(tmp_path)
        import ouroboros.skill_repair_admission as admission

        def _boom(*a, **k):
            raise OSError("read-only drive")

        monkeypatch.setattr(admission, "record_repair_admission", _boom)
        out = workers.promote_chat_to_task(
            {
                "task_id": "repair-fail-1",
                "objective": "fix the alpha skill",
                "task_constraint": {
                    "mode": "skill_repair", "skill_name": "alpha",
                    "payload_root": "skills/external/alpha",
                },
            },
            workers,
        )
        assert out["status"] == "needs_manual_target"
        assert out["reason"] == "skill_repair_admission_unwritable"

    def test_unreadable_payload_refuses_admission(self, tmp_path, monkeypatch):
        import supervisor.workers as workers

        monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
        payload = _seed_skill(tmp_path)
        (payload / "SKILL.md").unlink()
        (payload / "main.py").unlink()
        payload_sub = payload / "sub"
        payload_sub.mkdir()
        # An empty-but-present dir still hashes; force unreadability instead.
        import ouroboros.skill_loader as skill_loader

        def _boom(*a, **k):
            raise skill_loader.SkillPayloadUnreadable("boom")

        monkeypatch.setattr(skill_loader, "compute_content_hash", _boom)
        canonical, error = workers._canonical_promoted_repair_constraint({
            "mode": "skill_repair", "skill_name": "alpha",
            "payload_root": "skills/external/alpha",
        })
        assert canonical is None and error == "skill_repair_payload_unreadable"


class TestDataWriteSeam:
    def test_admitted_repair_write_is_cas_gated_end_to_end(self, tmp_path):
        from ouroboros.tools.core import _data_write
        from ouroboros.tools.registry import ToolContext

        payload = _seed_skill(tmp_path)
        constraint = _constraint()
        ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path)
        ctx.task_constraint = constraint
        ctx.task_id = "repair-1"
        record_repair_admission(tmp_path, "alpha", task_id="repair-1",
                                base_content_hash=_hash(payload))
        out = _data_write(ctx, "main.py", "print('fixed')\n")
        assert "⚠️" not in out, out
        assert (payload / "main.py").read_text(encoding="utf-8") == "print('fixed')\n"
        # The chain advanced with the repair's own write.
        assert load_repair_admission(tmp_path, "alpha")["expected_content_hash"] == _hash(payload)
        # Foreign drift now blocks the NEXT write, typed.
        (payload / "SKILL.md").write_text("# tampered\n", encoding="utf-8")
        blocked = _data_write(ctx, "main.py", "print('again')\n")
        assert "SKILL_REPAIR_STALE" in blocked
        assert (payload / "main.py").read_text(encoding="utf-8") == "print('fixed')\n"


class TestReceipts:
    def test_control_receipt_never_invents_an_id(self):
        # The bare "skill_repair" literal was persisted into the durable chat
        # log as if it were a task id. Now: caller-supplied id passes through,
        # absent id is empty + typed pending marker.
        import inspect

        from ouroboros.gateway import control

        source = inspect.getsource(control.api_command)
        assert 'or "skill_repair"' not in source
        assert "task_id_pending" in source

