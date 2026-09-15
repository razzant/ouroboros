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
from tests._governance_docs_shared import architecture_text


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

    monkeypatch.setattr(
        custody, "settled_unread_outputs", lambda root, state=None: [unread])
    monkeypatch.setattr(
        custody, "undisposed_patches", lambda root, state=None: [patch])


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
    architecture = architecture_text()
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


def test_one_custody_replay_per_health_invariant_build(tmp_path, monkeypatch):
    """I18 (partially closed): the two delegated obligations share ONE replay.

    Reading the rotated custody chain twice per context build cost a full
    parse of the whole archive each time; the pass is shared now, though the
    read is still O(history) and a compact projection remains open work.
    """
    import ouroboros.delegate_custody as custody

    calls: list = []
    real_replay = custody.replay
    monkeypatch.setattr(
        custody, "replay",
        lambda root, rows=None: calls.append(root) or real_replay(root, rows),
    )

    context_health.build_health_invariants(_env(tmp_path), task_id="task-owner")

    assert len(calls) == 1


# --- P6-3: the same obligation, for a reader whose own root already reaches it ---
#
# The recovery root's prompt carried sixteen near-identical abstract rows: the
# foreign branch keys on task identity alone, so it printed the rule even where
# the run's recorded target WAS that reader's own active root. `active_root` is
# threaded from the call site (the module has no ctx) and answered with the same
# predicate the apply gate asks, so the two can never disagree.


def _foreign_patch_targeting(monkeypatch, target_root):
    """One FOREIGN undisposed patch recorded against ``target_root``."""
    patch = RunCustody(run_id="run-p", task_id="task-owner")
    patch.patch_captured = True
    patch.target_root = str(target_root)
    import ouroboros.delegate_custody as custody

    monkeypatch.setattr(custody, "settled_unread_outputs", lambda root, *_a, **_k: [])
    monkeypatch.setattr(custody, "undisposed_patches", lambda root, *_a, **_k: [patch])


def _patch_row(text):
    return next(line for line in text.splitlines()
                if "DELEGATED PATCH AWAITS DISPOSITION" in line)


def test_reader_whose_active_root_is_the_target_gets_the_concrete_call(tmp_path, monkeypatch):
    target = tmp_path / "target"
    _foreign_patch_targeting(monkeypatch, target)
    text = context_health.build_health_invariants(
        _env(tmp_path), task_id="task-other", active_root=str(target))
    # The static rule still leads; the call FOLLOWS it.
    assert "apply requires the caller's active Git root or fresh payload binding" in text
    assert "Your own active root already satisfies that target" in text
    assert "integrate_delegated_patch(run_id='run-p', decision='apply'|'reject')" in text


def test_reader_whose_active_root_contains_a_host_minted_target_gets_the_call(tmp_path, monkeypatch):
    """The aggregator shape: the target is a clone NESTED in the reader's own
    root, both under the host-minted subagent-projects root."""
    projects_root = tmp_path / "projects"
    parent = projects_root / "P"
    target = parent / "contributions" / "tools"
    target.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    _foreign_patch_targeting(monkeypatch, target)
    text = context_health.build_health_invariants(
        _env(tmp_path), task_id="task-other", active_root=str(parent))
    assert "integrate_delegated_patch(run_id='run-p', decision='apply'|'reject')" in text


def test_an_unrelated_active_root_keeps_the_static_row_byte_for_byte(tmp_path, monkeypatch):
    target = tmp_path / "target"
    _foreign_patch_targeting(monkeypatch, target)
    static = _patch_row(context_health.build_health_invariants(
        _env(tmp_path), task_id="task-other"))
    unrelated = _patch_row(context_health.build_health_invariants(
        _env(tmp_path), task_id="task-other", active_root=str(tmp_path / "elsewhere")))
    assert unrelated == static
    assert "integrate_delegated_patch(run_id='run-p'" not in unrelated
