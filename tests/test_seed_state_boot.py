"""The real devtools seed producer must survive the server's real state boot."""
from __future__ import annotations

import json
import pathlib

import pytest

from devtools.benchmarks.common.server_runner import seed_owner_state
from supervisor import state, state_initialization



@pytest.fixture
def boot(tmp_path, monkeypatch):
    import server

    prior_root, prior_budget = state.DRIVE_ROOT, state.TOTAL_BUDGET_LIMIT
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda: None)
    monkeypatch.setattr(state, "_openrouter_ledger_settled", lambda: 0.0)
    monkeypatch.setattr(state, "_openrouter_key_fingerprint", lambda: "")
    yield lambda: server._initialize_runtime_state({"TOTAL_BUDGET": 1})
    state.init(prior_root, prior_budget)


@pytest.mark.parametrize("evolution", [False, True])
def test_seed_then_real_server_boot_completes_same_identity(tmp_path, boot, evolution):
    seed_owner_state(tmp_path, evolution_enabled=evolution)
    seed = json.loads((tmp_path / "state/state.json").read_text())
    boot()
    current = state.read_state()
    assert current.quality == "current", current.reason
    assert state.control_value(current.projection(), "owner_chat_id") == (True, 1)
    assert state.control_value(current.projection(), "evolution_mode_enabled") == (True, evolution)
    assert state.control_value(current.projection(), "bg_consciousness_enabled") == (True, False)
    assert current.values["initialization_id"] == seed["initialization_id"]
    assert current.values["session_id"] == seed["session_id"]
    assert state_initialization.read_witness(tmp_path)[1]["phase"] == "complete"
    assert state.STATE_PATH.read_bytes() == state.STATE_LAST_GOOD_PATH.read_bytes()


def test_old_partial_seed_is_not_legacy_initialization_evidence(tmp_path, boot):
    path = tmp_path / "state/state.json"
    path.parent.mkdir()
    raw = b'{"owner_chat_id": 1, "evolution_mode_enabled": true}'
    path.write_bytes(raw)
    boot()
    assert path.read_bytes() == raw
    assert state.control_value(state.read_state().projection(), "evolution_mode_enabled")[0] is False
    assert state_initialization.read_witness(tmp_path)[0] == "missing"


def test_seeded_identity_does_not_make_lost_control_known(tmp_path, boot):
    seed_owner_state(tmp_path, evolution_enabled=True)
    path = tmp_path / "state/state.json"
    seed = json.loads(path.read_text())
    seed.pop("evolution_owner_stopped")
    path.write_text(json.dumps(seed))
    boot()
    current = state.read_state()
    assert state.control_value(current.projection(), "owner_chat_id") == (True, 1)
    assert state.control_value(current.projection(), "evolution_mode_enabled") == (True, True)
    assert state.control_value(current.projection(), "evolution_owner_stopped")[0] is False


@pytest.mark.parametrize("case", ["existing", "invalid_witness", "unreadable"])
def test_seed_refuses_prior_or_unknown_root_without_overwriting(tmp_path, monkeypatch, case):
    path = tmp_path / "state/state.json"
    path.parent.mkdir()
    if case in {"existing", "backup"}:
        (path if case == "existing" else path.with_name("state.last_good.json")).write_text('{"owner_chat_id": 7}')
    elif case == "lost":
        identity = state_initialization.mark_pending(tmp_path, origin="first_boot")
        assert state_initialization.complete(tmp_path, identity, adopted=False)
    elif case == "history":
        (path.parent / "evolution_campaign.json").write_text('{"status": "stopped"}')
    elif case == "invalid_witness":
        state_initialization.witness_path(tmp_path).write_text("{broken")
    else:
        original = pathlib.Path.read_bytes

        def unreadable(candidate):
            if candidate == path:
                raise PermissionError("isolated unavailable state")
            return original(candidate)

        monkeypatch.setattr(pathlib.Path, "read_bytes", unreadable)
    before = {p.name: p.read_text() for p in path.parent.iterdir()}
    with pytest.raises((ValueError, state.StateUnavailable)):
        seed_owner_state(tmp_path, evolution_enabled=True)
    assert {p.name: p.read_text() for p in path.parent.iterdir()} == before
