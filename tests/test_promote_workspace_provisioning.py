"""A file-less project promotion auto-provisions its genesis workspace.

Split verbatim out of ``tests/test_promote_chat_flow.py`` by theme. This module owns
the empty working_dir that gets a provisioned workspace bound to it, the explicit
opt-out that must stay an opt-out, and the loud failures — a broken working dir and a
failed provisioning — that may never degrade into a silent file-less promotion.
"""

from __future__ import annotations

import types


from tests._promote_chat_shared import _isolated_projects_root  # noqa: F401  (autouse fixture applies on import)


# --- Q10=A (owner, 2026-08-08): file-less project promotes auto-provision -----

def _promote_ctx(enqueued):
    return types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )


def test_promote_fileless_project_autoprovisions_and_binds_workspace(tmp_path, monkeypatch):
    """A project promoted with an EMPTY working_dir gets a genesis workspace via
    the existing ensure_project_workspace seam and the task is BOUND to it
    (external profile, forked memory, lease lane) — the submarine shape fix."""
    import os

    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "fileless1",
        "objective": "Build the submarine game",
        "project_id": "sunken-city",
        "project_name": "Sunken City",
        "chat_id": 1,
    }, _promote_ctx(enqueued))

    assert outcome["status"] == "scheduled"
    task = enqueued[0]
    ws = str(task.get("workspace_root") or "")
    assert ws, "file-less project promote must bind an auto-provisioned workspace"
    projects_root = os.environ["OUROBOROS_SUBAGENT_PROJECTS_ROOT"]
    assert ws.startswith(str(pathlib_resolve(projects_root)))
    assert task["workspace_mode"] == "external"
    assert task["memory_mode"] == "forked"
    assert task["metadata"]["workspace_autoprovisioned"] is True
    assert "[HEADLESS_WORKSPACE]" in task["text"]
    # The registry carries the provisioned working_dir for later waves/promotes.
    assert get_project(tmp_path, "sunken-city")["working_dir"] == ws
    # Idempotency: a second promote reuses the SAME tree (no sunken-city_1 mint).
    enqueued2 = []
    workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "fileless2",
        "objective": "Continue the submarine game",
        "project_id": "sunken-city",
        "chat_id": 1,
    }, _promote_ctx(enqueued2))
    assert enqueued2[0]["workspace_root"] == ws


def pathlib_resolve(p):
    import pathlib

    return pathlib.Path(p).resolve()


def test_promote_workspace_none_still_opts_out_of_autoprovision(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "optout1",
        "objective": "Pure research, no folder",
        "project_id": "folderless",
        "workspace": "none",
        "chat_id": 1,
    }, _promote_ctx(enqueued))

    assert outcome["status"] == "scheduled"
    assert not enqueued[0].get("workspace_root")
    # The opt-out means NO provisioning side effect either.
    assert str(get_project(tmp_path, "folderless").get("working_dir") or "") == ""


def test_promote_broken_working_dir_loud_fails_never_blind_ensures(tmp_path, monkeypatch):
    """v6.58.0 invariant preserved: a NON-EMPTY broken working_dir loud-fails;
    auto-provision fires ONLY on the empty string and never papers over a broken
    folder with a fresh empty repo."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, get_project, update_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    create_project(tmp_path, "brokenp", name="BrokenP")
    gone = tmp_path / "gone-folder"
    update_project(tmp_path, "brokenp", working_dir=str(gone))  # never existed

    enqueued = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "broken1",
        "objective": "Continue",
        "project_id": "brokenp",
        "chat_id": 1,
    }, _promote_ctx(enqueued))

    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "workspace_unusable"
    assert enqueued == []
    # The broken value is preserved for the owner to fix — not overwritten.
    assert get_project(tmp_path, "brokenp")["working_dir"] == str(gone)


def test_promote_provisioning_failure_loud_fails_not_silent_fileless(tmp_path, monkeypatch):
    """Bind-or-fail: if auto-provisioning fails, the promote fails LOUDLY instead
    of silently degrading to a workspace-less self_modification-profile task."""
    import supervisor.workers as workers
    from ouroboros import projects_registry

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(projects_registry, "ensure_project_workspace", lambda *a, **k: "")
    enqueued = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "provfail1",
        "objective": "Build",
        "project_id": "provfail-proj",
        "chat_id": 1,
    }, _promote_ctx(enqueued))

    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "workspace_provisioning_failed"
    assert enqueued == []
