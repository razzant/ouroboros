"""Real current-generation Restart input for lifecycle tests (no legacy flags)."""
import threading


def live_restart(root, monkeypatch):
    from ouroboros import server_restart
    from ouroboros.delegate_recovery import prepare_owner_restart

    row = prepare_owner_restart(root)
    monkeypatch.setattr(server_restart, "DATA_DIR", root)
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    server_restart._OWNER_STOP_BEGAN.set()
    return row


def live_panic(root, monkeypatch):
    from ouroboros import config, server_control

    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(server_control, "_restart_stop_requested", True)
