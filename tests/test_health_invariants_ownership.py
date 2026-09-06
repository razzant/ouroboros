"""Delegated-run health obligations shape their instruction by ownership.

The warnings stay globally visible (a preserved-and-invisible result is how
work rots on disk), but only the OWNER task receives the call-shaped
instruction: ``integrate_delegated_patch`` refuses a non-owner with
``run_not_owned`` while the owner task is LIVE, and the read acknowledgement
only credits the owner, so a foreign reader must be told WHO can act, never
handed a ready-to-paste call that structurally refuses. Once the owner task is
terminal, apply needs matching target authority but reject does not, so the
clause states that split without minting a callable shape.
"""

import types

import ouroboros.context_health as context_health
from ouroboros.delegate_custody import RunCustody


def _env(tmp_path):
    env = types.SimpleNamespace()
    env.drive_root = tmp_path
    env.repo_path = lambda *parts: tmp_path / "repo-none" / "/".join(parts)
    env.drive_path = lambda *parts: tmp_path / "/".join(parts)
    return env


def _foreign_runs(monkeypatch):
    unread = RunCustody(run_id="run-u", task_id="task-owner")
    unread.output_artifact = "delegated/run-u.out"
    patch = RunCustody(run_id="run-p", task_id="task-owner")
    patch.patch_captured = True
    patch.target_root = "/tmp/target"
    monkeypatch.setattr(
        context_health, "build_health_invariants", context_health.build_health_invariants
    )
    import ouroboros.delegate_custody as custody

    monkeypatch.setattr(custody, "settled_unread_outputs", lambda root: [unread])
    monkeypatch.setattr(custody, "undisposed_patches", lambda root: [patch])


def test_non_owner_gets_no_call_shaped_instruction(tmp_path, monkeypatch):
    _foreign_runs(monkeypatch)
    text = context_health.build_health_invariants(_env(tmp_path), task_id="task-other")
    assert "DELEGATED RESULT NEVER READ" in text
    assert "DELEGATED PATCH AWAITS DISPOSITION" in text
    assert "owner task task-owner" in text.lower() or "task-owner" in text
    # The obligation is visible; the DIRECTLY CALLABLE shape is not. The clause
    # names the tool only inside the conditional rule (a terminal owner's orphan
    # has distinct apply/reject authority), never as a
    # ready-to-paste call this foreign reader could make against a LIVE owner.
    assert "integrate_delegated_patch(run_id='run-p'" not in text
    assert "once that task is terminal" in text
    assert "apply requires the caller's active Git root or fresh payload binding" in text
    assert "reject may release it even from a different active root" in text
    assert "disposition row records who acted" in text
    assert "read_file" not in text
    assert "run_not_owned" in text


def test_architecture_states_the_terminal_owner_apply_reject_authority_split():
    import pathlib

    architecture = (pathlib.Path(__file__).resolve().parents[1] / "docs" /
                    "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "Apply requires the caller's active Git root or fresh payload binding" in architecture
    assert "Reject requires only the owner's proven terminality" in architecture
    assert "a live top-level task with a different active root may reject and release" in architecture
    assert "disposition row records who did it" in architecture


def test_owner_keeps_the_call_shaped_instruction(tmp_path, monkeypatch):
    _foreign_runs(monkeypatch)
    text = context_health.build_health_invariants(_env(tmp_path), task_id="task-owner")
    assert "integrate_delegated_patch(run_id='run-p'" in text
    assert "read_file" in text


def test_unattributed_reader_keeps_the_call_shape(tmp_path, monkeypatch):
    """Background Consciousness and legacy callers pass no task id; they may
    be the owner, so the call-shaped wording survives."""
    _foreign_runs(monkeypatch)
    text = context_health.build_health_invariants(_env(tmp_path))
    assert "integrate_delegated_patch(run_id='run-p'" in text
    assert "read_file" in text
