"""The destructive bootstrap path requires launcher identity, not a copied marker."""

from __future__ import annotations


class _GitOps:
    def __init__(self):
        self.safe_restart_calls = []
        self.local_sync_calls = []

    def init(self, **_kwargs):
        return None

    def ensure_repo_present(self):
        return None

    def safe_restart(self, **kwargs):
        self.safe_restart_calls.append(kwargs)
        return True, "reset"

    def sync_runtime_dependencies(self, **kwargs):
        self.local_sync_calls.append(kwargs)
        return True, "ok"

    def import_test(self):
        return {"ok": True}


def test_bootstrap_marker_without_matching_repo_identity_skips_reset(monkeypatch, tmp_path):
    import server

    git = _GitOps()
    monkeypatch.setattr(server, "REPO_DIR", tmp_path / "external")
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", True)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED_REPO_DIR", str(tmp_path / "other"))
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *args: None)

    ok, message = server._bootstrap_supervisor_repo({}, git)

    assert ok is True
    assert "local-dev" in message
    assert git.safe_restart_calls == []
    assert git.local_sync_calls == [{"reason": "bootstrap_local_dev"}]


def test_bootstrap_matching_repo_identity_keeps_managed_reset(monkeypatch, tmp_path):
    import server

    repo = tmp_path / "managed"
    repo.mkdir()
    git = _GitOps()
    monkeypatch.setattr(server, "REPO_DIR", repo)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", True)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED_REPO_DIR", str(repo.resolve()))
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *args: None)
    monkeypatch.setattr(server, "_has_active_evolution_transaction", lambda: False)
    monkeypatch.setattr(server, "_safe_restart_serialized", lambda fn, **kwargs: fn(**kwargs))

    ok, message = server._bootstrap_supervisor_repo({}, git)

    assert ok is True
    assert message == "reset"
    assert git.safe_restart_calls == [{"reason": "bootstrap", "unsynced_policy": "keep"}]  # the cold boot keeps local work
    assert git.local_sync_calls == []


def test_bootstrap_legacy_launcher_metadata_keeps_managed_reset(monkeypatch, tmp_path):
    import server

    repo = tmp_path / "managed"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "ouroboros-managed.json").write_text("{}", encoding="utf-8")
    git = _GitOps()
    monkeypatch.setattr(server, "REPO_DIR", repo)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", True)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED_REPO_DIR", "")
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *args: None)
    monkeypatch.setattr(server, "_has_active_evolution_transaction", lambda: False)
    monkeypatch.setattr(server, "_safe_restart_serialized", lambda fn, **kwargs: fn(**kwargs))

    ok, message = server._bootstrap_supervisor_repo({}, git)

    assert ok is True and message == "reset"
    assert git.safe_restart_calls == [{"reason": "bootstrap", "unsynced_policy": "keep"}]  # the cold boot keeps local work
