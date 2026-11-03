"""A stored root cannot authorize another install's process ledger or spawn."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import install_identity, owned_shutdown, process_custody, tool_custody


def _tree(root):
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
            for path in root.rglob("*")}


@pytest.mark.parametrize("operation", ["record", "spawn"])
@pytest.mark.parametrize("layout", ["", "task_drives/task", "state/headless_tasks/task/data"])
def test_process_custody_refuses_foreign_before_any_effect_but_owns_local_and_bare(
        tmp_path, monkeypatch, operation, layout):
    a, b, bare = tmp_path / "A", tmp_path / "B", tmp_path / "isolated"
    for root in (a, b):
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        install_identity.initialize_install()
    spawns, handoffs = [], []
    proc = SimpleNamespace(pid=12345)
    monkeypatch.setattr(process_custody.subprocess, "Popen", lambda *a, **k: spawns.append(a) or proc)
    monkeypatch.setattr(process_custody, "process_group_id", lambda pid: 0)
    monkeypatch.setattr(process_custody, "process_fingerprint", lambda *a: {"start_time": "test"})
    monkeypatch.setattr(tool_custody, "external_handoff", lambda *a: handoffs.append(a) or (lambda: None))

    def act(root):
        fields = {"drive_root": root, "purpose": "test", "scope": "task", "owner_task_id": "task"}
        if operation == "spawn":
            return process_custody.spawn_supervised(["unused"], **fields)
        return process_custody.record_process(pid=proc.pid, cmd=["unused"], **fields)

    before = _tree(a)
    with pytest.raises(install_identity.StoreForeign) as refused:
        act(a / layout)
    assert refused.value.reason_code == "foreign_install_root"
    assert _tree(a) == before and not spawns and not handoffs

    for root in (b / layout, bare / layout):
        result = act(root)
        entry = json.loads(process_custody.ledger_path(root).read_text())
        assert entry["pid"] == proc.pid and entry["owner_task"] == "task"
        assert owned_shutdown.owned_records(root)[0]["drive_root"] == str(root)
        if operation == "spawn":
            assert result is proc and proc._ouroboros_custody == entry
    assert len(spawns) == len(handoffs) == (2 if operation == "spawn" else 0)
    assert not (bare / install_identity.IDENTITY_PATH).exists()
