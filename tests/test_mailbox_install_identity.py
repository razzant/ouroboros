"""A stored mailbox pointer never authorizes writing another install's files."""
from types import SimpleNamespace

import pytest

from ouroboros import install_identity as identity, owner_mailbox as mailbox, store
from ouroboros.tools.core_artifacts import _escalate


@pytest.fixture
def installs(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    for root in (a, b):
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        identity.initialize_install()
    yield a, b, tmp_path / "bare"
    store.close_connections()


def _tree(root):
    return {str(path.relative_to(root)): None if path.is_dir() else path.read_bytes()
            for path in root.rglob("*")}


def _produce(kind, root):
    if kind == "owner":
        return mailbox.write_owner_message(root, "Owner input", "parent", msg_id="message")
    if kind == "task":
        return mailbox.write_task_message(root, "Task input", "parent", source_task_id="child",
                                          msg_id="message")
    return mailbox.acknowledge_task_messages(root, "parent", ["message"], wake_id="test")


@pytest.mark.parametrize("kind", ["owner", "task", "ack"])
@pytest.mark.parametrize("layout", ["", "task_drives/parent", "state/headless_tasks/parent/data"])
def test_mailbox_producers_refuse_before_mkdir_and_keep_own_and_bare_roots(installs, kind, layout):
    a, b, bare = installs
    before = _tree(a)
    with pytest.raises(identity.StoreForeign) as refused:
        _produce(kind, a / layout)
    assert refused.value.reason == "foreign_install_root"
    assert refused.value.path == (a / layout).resolve()
    assert _tree(a) == before  # Includes directories, mailbox rows and lock sidecars.
    assert not (a / layout / "memory/owner_mailbox").exists()
    for root in (b / layout, bare / layout):
        assert _produce(kind, root)
        path = mailbox._ack_path(root, "parent") if kind == "ack" else mailbox._mailbox_path(root, "parent")
        assert path.is_file() and '"msg_id": "message"' in path.read_text()
    assert not (bare / identity.IDENTITY_PATH).exists()


def _child_context(root):
    return SimpleNamespace(
        task_id="child", drive_root=root, budget_drive_root=str(root), pending_events=[],
        task_metadata={"parent_task_id": "parent", "root_task_id": "parent",
                       "budget_drive_root": str(root)},
        _active_builtin_tool_result=None,
    )


@pytest.mark.parametrize("field,layout", [
    ("drive_root", ""),
    ("child_drive_root", "task_drives/parent"),
    ("headless_child_drive_root", "state/headless_tasks/parent/data"),
])
def test_parent_escalation_exposes_foreign_reason_and_keeps_own_and_bare_delivery(
        installs, monkeypatch, field, layout):
    from ouroboros import task_status

    a, b, bare = installs
    parent = {"status": "scheduled", field: str(a / layout)}
    monkeypatch.setattr(task_status, "load_effective_task_result", lambda *_: parent)
    before = _tree(a)
    ctx = _child_context(b)
    text = _escalate(ctx, question="Which destination?", assumption="Keep the current destination")
    result = ctx._active_builtin_tool_result
    assert result.status == "blocked" and result.code == "ACCESS_BLOCKED"
    assert result.meta["reason"] == "foreign_install_root"
    assert result.meta["path"] == str((a / layout).resolve())
    assert "foreign_install_root" in text and str(a.resolve()) in text
    assert "move it explicitly to work on it here" in text
    assert _tree(a) == before and not ctx.pending_events
    for root in (b / layout, bare / layout):
        parent[field] = str(root)
        own_ctx = _child_context(b)
        text = _escalate(own_ctx, question="Which destination?", assumption="Keep the current destination")
        assert text.startswith("OK: escalated to parent task parent")
        assert own_ctx._active_builtin_tool_result is None
        assert "Which destination?" in mailbox._mailbox_path(root, "parent").read_text()
    assert not (bare / identity.IDENTITY_PATH).exists()
