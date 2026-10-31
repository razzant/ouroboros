"""A failed switch owes one durable owner notice after publication, never a retry switch."""
import json
import pathlib

import pytest

from ouroboros import store, store_switch
from ouroboros.store_import import ensure_current_store
from ouroboros.utils import atomic_write_json
from supervisor import message_bus, state
from tests.test_store_switch import Owner, intent
from tests.test_store_switch import switch_root as switch_root, target_repo as target_repo


@pytest.fixture
def owner_chat(switch_root, monkeypatch):
    state.init(switch_root)
    atomic_write_json(state.STATE_PATH, {"owner_chat_id": 7})  # the legacy install already bound its owner
    sent = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat, text, **kw: sent.append((chat, text, kw)))
    return sent


def progress(root):
    return json.loads((root / store_switch.PROGRESS_REL).read_text(encoding="utf-8"))


@pytest.mark.parametrize("outcome", ["refused", "abandoned", "switch_not_performed", "switch_failed", "switched"])
def test_next_boot_reports_failed_switch_once(switch_root, target_repo, owner_chat, outcome):
    root, (repo, old, _) = switch_root, target_repo
    intent(root, old)

    def interrupted(_):
        raise KeyboardInterrupt("stopped after consuming the intent")

    def failed_handles(_):
        raise OSError("the retained files could not be restored")

    if outcome == "refused":
        assert store_switch.run_switch_seam(root, repo_dir=repo, blockers=["worker:42"])["status"] == "refused"
    elif outcome == "abandoned":
        for _ in range(2):
            assert store_switch.run_switch_seam(root, repo_dir=repo, restore_handles=failed_handles)["status"] == "failed"
        assert store_switch.resume_pending_switch(root, repo_dir=repo)["status"] == "refused"
    elif outcome == "switch_not_performed":
        with pytest.raises(KeyboardInterrupt):
            store_switch.run_switch_seam(root, repo_dir=repo, owner=interrupted)
    else:
        owner = Owner(repo) if outcome == "switched" else lambda _: (False, "The checkout could not be changed.")
        assert store_switch.run_switch_seam(root, repo_dir=repo, owner=owner)["status"] == outcome
    assert owner_chat == []
    store_switch.resume_pending_switch(root, repo_dir=repo)
    ensure_current_store(root)
    store_switch.startup_notice(root)
    if outcome == "switched":
        assert owner_chat == []
        return
    [(chat, text, voice)] = owner_chat
    assert chat == 7 and voice == {"role": "system", "system_type": "storage_switch_notice", "require_write": True}
    assert "Rollback" in text and "v7.6.0" in text and "did not happen" in text
    assert "current version" in text and "again" in text and "\n" not in text
    reasons = {"refused": "could not confirm", "abandoned": "failed twice",
               "switch_not_performed": "stopped", "switch_failed": "checkout could not be changed"}
    assert reasons[outcome] in text
    assert all(row["reported"] for row in progress(root)["outcomes"])
    state.init(root)  # a new boot reloads durable state, not a process-local sent flag
    store_switch.resume_pending_switch(root, repo_dir=repo)
    ensure_current_store(root)
    store_switch.startup_notice(root)
    assert len(owner_chat) == 1


def test_notice_waits_for_owner_and_publication_and_preserves_each_outcome(switch_root, target_repo, owner_chat):
    root, (repo, old, _) = switch_root, target_repo
    for door in ("vcs_rollback", "update_replace"):
        intent(root, old, door)
        store_switch.run_switch_seam(root, repo_dir=repo, blockers=["worker:42"])
    atomic_write_json(state.STATE_PATH, {"owner_chat_id": 0})
    store_switch.startup_notice(root)
    assert owner_chat == []
    atomic_write_json(state.STATE_PATH, {"owner_chat_id": 7})
    with store.importing(root):
        store_switch.startup_notice(root)
    assert owner_chat == []
    store_switch.startup_notice(root)
    assert len(owner_chat) == 2
    assert "Agent rollback" in owner_chat[0][1] and "Update" in owner_chat[1][1]
    store_switch.startup_notice(root)
    assert len(owner_chat) == 2


def test_supervisor_wires_switch_notice_after_publication_and_bus_initialization():
    source = (pathlib.Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")
    body = source.split("def _run_supervisor(settings: dict) -> None:", 1)[1].split("\ndef ", 1)[0]
    dialogue = source.split("def _open_dialogue(", 1)[1].split("\ndef ", 1)[0]
    assert "bus_init(" in dialogue, "the bus is initialized where the dialogue opens"
    assert body.index("store_switch.require_published(DATA_DIR)") < body.index("_open_dialogue(")
    assert body.index("_open_dialogue(") < body.index("store_switch.startup_notice(DATA_DIR)")


def test_previous_body_outcome_and_failed_send_remain_owed(switch_root, target_repo, owner_chat, monkeypatch):
    root, (repo, old, _) = switch_root, target_repo
    intent(root, old, "boot_smoke_rollback")
    store_switch.run_switch_seam(root, repo_dir=repo, blockers=["worker:42"])
    legacy = progress(root)
    legacy.pop("outcomes")
    legacy.pop("intent")
    atomic_write_json(root / store_switch.PROGRESS_REL, legacy)
    with monkeypatch.context() as patch:
        patch.setattr(message_bus, "send_with_budget", lambda *a, **kw: (_ for _ in ()).throw(OSError("chat unavailable")))
        store_switch.startup_notice(root)
    assert owner_chat == [] and progress(root) == legacy
    store_switch.startup_notice(root)
    assert len(owner_chat) == 1 and "Automatic rollback to v7.6.0" in owner_chat[0][1]
    store_switch.startup_notice(root)
    assert len(owner_chat) == 1 and progress(root)["outcomes"][0]["reported"]
