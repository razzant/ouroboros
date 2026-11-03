"""Delegated-run health obligations follow ownership.

The obligations stay globally visible (a preserved-and-invisible result is how
work rots on disk), but only the OWNER task receives each run's line with its
call-shaped instruction: ``integrate_delegated_patch`` refuses a non-owner with
``run_not_owned`` while the owner task is LIVE, and the read acknowledgement
only credits the owner. All foreign runs share one line with the authority rule
and a pointer to the exact owner/run list in the existing source store.
"""

import json
import re
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


def _foreign_source(root, text):
    from ouroboros.tools.query_data import _query_data
    from ouroboros.tools.registry import ToolContext

    pointer = re.search("query_data\\(op='blob', sha256='([a-f0-9]{64})'\\)", text)
    assert pointer
    reply = json.loads(_query_data(ToolContext(repo_dir=root, drive_root=root, task_id="reader"),
                                  op="blob", sha256=pointer.group(1)))
    assert reply["status"] == "ok"
    return reply["data"]


def test_foreign_obligations_share_one_line_with_a_readable_exact_list(tmp_path, monkeypatch):
    _foreign_runs(monkeypatch)
    text = context_health.build_health_invariants(_env(tmp_path), task_id="task-other")
    rows = [line for line in text.splitlines() if "DELEGATED WORK" in line]
    assert len(rows) == 1 and "WARNING:" in rows[0]
    assert _foreign_source(tmp_path, rows[0]) == {
        "task-owner": {"unread": ["run-u"], "patches": ["run-p"], "yours": []}}
    assert "DELEGATED RESULT NEVER READ" not in text and "DELEGATED PATCH AWAITS DISPOSITION" not in text
    assert "integrate_delegated_patch(run_id='run-p'" not in text
    assert "after it ends, a live top-level task may reject from any root" in rows[0]
    assert "apply to the original target when location and current write rights permit" in rows[0]
    assert "payloads need a fresh binding" in rows[0]
    assert "read_file" not in text


def test_hundreds_of_foreign_owners_still_get_one_line_and_own_runs_stay_full(tmp_path, monkeypatch):
    import ouroboros.delegate_custody as custody
    from ouroboros.delegate_custody import RunCustody

    def unread(run_id, task):
        run = RunCustody(run_id=run_id, task_id=task)
        run.output_artifact = f"delegated/{run_id}.out"
        return run

    runs = [unread(f"run-a{i}", f"task-{i}") for i in range(300)] + [unread("run-own", "me")]
    monkeypatch.setattr(custody, "settled_unread_outputs", lambda root, state=None: runs)
    monkeypatch.setattr(custody, "undisposed_patches", lambda root, state=None: [])
    text = context_health.build_health_invariants(_env(tmp_path), task_id="me")
    folded = [line for line in text.splitlines() if "DELEGATED WORK" in line]
    assert len(folded) == 1 and len(folded[0]) < 700 and "INFO:" in folded[0]
    source = _foreign_source(tmp_path, folded[0])
    assert len(source) == 300 and all(source[f"task-{i}"]["unread"] == [f"run-a{i}"] for i in range(300))
    assert "DELEGATED RESULT NEVER READ — run run-own" in text and "run-own" not in "".join(folded)


def test_architecture_states_the_terminal_owner_apply_reject_authority_split():
    architecture = architecture_text()
    assert "PATCH_DISPOSED" in architecture
    assert "disposed_by_task_id" in architecture


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


def test_one_custody_read_per_health_invariant_build(tmp_path, monkeypatch):
    """I18: the delegated-run lines share ONE custody read, and it is selected.

    Reading every custody row per block, and then per build, decoded the whole
    settled history each time; the shared read now selects the rows the lines
    can name (own, held, maybe-unread) and no full read runs unless it fails.
    SQL still visits every row, so the read stays O(rows) in the store.
    """
    import ouroboros.delegate_custody as custody
    import ouroboros.delegate_custody_current as current

    calls: list = []
    replays: list = []
    real_select, real_replay = current.health_runs, custody.replay
    monkeypatch.setattr(
        current, "health_runs",
        lambda root, task_id="": calls.append(root) or real_select(root, task_id),
    )
    monkeypatch.setattr(
        custody, "replay",
        lambda root, rows=None: replays.append(root) or real_replay(root, rows),
    )

    context_health.build_health_invariants(_env(tmp_path), task_id="task-owner")

    assert len(calls) == 1
    assert replays == []


# --- P6-3: the same obligation, for a reader whose own root already reaches it ---
#
# The recovery root's prompt carried sixteen near-identical abstract rows: the
# foreign branch keys on task identity alone, so it printed the rule even where
# the run's recorded target WAS that reader's own active root. `active_root` is
# threaded from the call site (the module has no ctx); 'yours' keeps the exact-or-
# nested relationship, not the apply gate's wider orphan geometry or write rights.


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
                if "OTHER TASKS' DELEGATED WORK" in line)


def test_reader_whose_active_root_is_the_target_keeps_that_fact_in_the_source(tmp_path, monkeypatch):
    target = tmp_path / "target"
    _foreign_patch_targeting(monkeypatch, target)
    text = context_health.build_health_invariants(
        _env(tmp_path), task_id="task-other", active_root=str(target))
    assert _foreign_source(tmp_path, text)["task-owner"]["yours"] == ["run-p"]


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
    assert _foreign_source(tmp_path, text)["task-owner"]["yours"] == ["run-p"]


def test_an_unrelated_active_root_keeps_the_static_row_byte_for_byte(tmp_path, monkeypatch):
    target = tmp_path / "target"
    _foreign_patch_targeting(monkeypatch, target)
    static = _patch_row(context_health.build_health_invariants(
        _env(tmp_path), task_id="task-other"))
    unrelated = _patch_row(context_health.build_health_invariants(
        _env(tmp_path), task_id="task-other", active_root=str(tmp_path / "elsewhere")))
    assert unrelated == static
    assert "integrate_delegated_patch(run_id='run-p'" not in unrelated


def test_failed_foreign_list_retention_discloses_the_gap_and_keeps_the_list(tmp_path, monkeypatch):
    from ouroboros import observability
    _foreign_runs(monkeypatch)
    monkeypatch.setattr(observability, "write_blob", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    text = context_health.build_health_invariants(_env(tmp_path), task_id="task-other")
    assert "retention unavailable; exact list:" in text
    assert '"task-owner"' in text and '"run-u"' in text and '"run-p"' in text


def test_sibling_host_targets_are_not_mislabelled_as_this_readers_own_folder(tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    target, active = projects_root / "A", projects_root / "B"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    _foreign_patch_targeting(monkeypatch, target)
    text = context_health.build_health_invariants(_env(tmp_path), task_id="other", active_root=str(active))
    assert _foreign_source(tmp_path, text)["task-owner"]["yours"] == []
