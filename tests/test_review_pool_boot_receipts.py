"""The supervisor boot receipts only the migrations that decide the settings document on disk.

The review-lane -> review-pool migration runs at the settings read seam of whichever process
reads a document, and the process keeps every outcome it computed: the never-configured
outcome of the defaults it read while no settings file existed, a draft it normalized, the
N-1 document it read. A write gives receipts to the migration of the document it replaces and
of the one it saves (``review_pool_receipts.persist_write_receipts``); the boot
(``server_maintenance._startup_review_pool_notice``) applies the same rule to the document on
disk now — with no file, to the defaults the install runs — instead of receipting its whole
memory. Pinned both ways: a fresh install whose wizard saved a catalog of its own is told
nothing about the factory rows the read seam minted before the wizard ran, and an N-1 document
the server read and no save replaced still gets its one snapshot and its one owner message
while the other outcomes in memory get none; without a file the defaults' outcome is receipted
and a draft's is not; a migration whose result is the file still is (the save's own receipt
failed).
"""

from __future__ import annotations

import json

import pytest

import ouroboros.configured_subagents as cs
from ouroboros import config as cfg
from ouroboros import review_pool_migration as m
from ouroboros import server_maintenance
from ouroboros.gateway import settings as gw_settings
from supervisor import message_bus, state
from tests.test_onboarding_complete_endpoint import (
    LIVE_SNAPSHOT,
    onboarding as onboarding,  # explicit fixture re-export: the real wizard endpoint over a tmp settings file
)
from tests.test_review_pool_migration import N1_DOC, SLOTS, SUBAGENTS, anton_document, catalog, session_row

MAIN = "claudexor::codex=default"
MODEL_CATALOG = [{"value": MAIN, "is_default": True, "input_modalities": ["text", "image"]}]
NOTICE = "review_pool_migration_notice"


@pytest.fixture(autouse=True)
def _fresh_seam(monkeypatch):
    monkeypatch.setattr(cs, "MAX_CONFIGURED_SUBAGENTS", 26)
    m._MIGRATIONS_SEEN.clear()
    yield
    m._MIGRATIONS_SEEN.clear()


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """This process's supervisor state bound to ``tmp_path`` with an owner chat; the chat captured."""
    state.init(tmp_path)
    for sub in ("state", "locks"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    state.save_state({"owner_chat_id": 7})
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    for key in m._SHA_PRESENCE_KEYS:
        monkeypatch.delenv(key, raising=False)
    sent: list = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat_id, text, *a, **k: sent.append((chat_id, text, k)))
    return tmp_path, sent


def _snapshots(root):
    return sorted((root / "state" / "review_migrations").glob("*-slots-to-pool.json"))


def _notices(sent):
    return [text for _chat_id, text, kwargs in sent if kwargs.get("system_type") == NOTICE]


def test_a_wizard_install_hears_nothing_about_the_factory_rows_minted_before_its_wizard_ran(boot, onboarding,
                                                                                          monkeypatch):
    """The fresh install: the server starts without a settings file and its read seam computes the
    never-configured outcome of the defaults (the factory rows); the owner finishes the subscription
    wizard, which saves the document with ITS OWN catalog (session reviewers); the supervisor starts
    in the same process. Neither that outcome nor a draft the process normalized decides the document
    on disk: their receipts would tell the owner that rows run which never ran."""
    root, sent = boot
    started: list = []
    monkeypatch.setattr(gw_settings, "_start_supervisor_if_needed_for_request",
                        lambda _request, settings: started.append(dict(settings)) or True)
    cfg.load_settings()  # the server's start: no settings file yet
    assert [o.trigger for o in cfg.review_pool_migrations_seen()] == [m.TRIGGER_NEVER_CONFIGURED]
    cfg.normalize_settings_raw({"OUROBOROS_MODEL": MAIN})  # a wizard draft this process normalized at the seam
    onboarding.calls["snapshot_payload"] = {**LIVE_SNAPSHOT, "model_catalog": MODEL_CATALOG}
    completed = onboarding.client.post("/api/onboarding/complete", json={"subscriptionsConnected": True})
    assert completed.status_code == 200, completed.text
    reviewers = [row for row in json.loads(onboarding.saved()[SUBAGENTS])["items"] if row.get("review_eligible")]
    assert reviewers and {row["route"]["kind"] for row in reviewers} == {"agent_session"}
    assert len(cfg.review_pool_migrations_seen()) == 2 and _snapshots(root) == [], "the wizard's save receipted neither"

    (settings,) = started  # what the supervisor generation is started with
    server_maintenance._startup_review_pool_notice(settings)

    assert _snapshots(root) == [] and server_maintenance.review_pool_migration_records() == {}
    assert _notices(sent) == []


def test_the_n1_document_the_server_read_keeps_its_one_receipt_beside_outcomes_that_decide_nothing(boot, monkeypatch):
    """The normal upgrade: the server read the N-1 document no save has replaced yet; its migration
    gets the snapshot, the record and ONE owner message at the first boot, and a revival adds
    nothing. Another document this process normalized (a draft, another root's document) decides
    nothing on disk and gets none."""
    root, sent = boot
    path = root / "settings.json"
    path.write_text(json.dumps(anton_document()), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    cfg.normalize_settings_raw(dict(N1_DOC))  # a document this process normalized that nobody saves here
    settings = cfg.load_settings_lock_held(_settings_lock_held=False)  # the server's read of the file
    assert len(cfg.review_pool_migrations_seen()) == 2

    server_maintenance._startup_review_pool_notice(settings)
    server_maintenance._startup_review_pool_notice(settings)  # a supervisor revival

    (snapshot_file,) = _snapshots(root)
    snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    assert snapshot["input_sha256"] == m.input_sha256(anton_document())
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["snapshot"] == f"state/review_migrations/{snapshot_file.name}" and record["reported"]
    (outcome,) = [o for o in cfg.review_pool_migrations_seen() if o.input_sha256 == snapshot["input_sha256"]]
    assert _notices(sent) == [m.owner_message(outcome, record["snapshot"])]


def test_without_a_settings_file_the_boot_receipts_the_defaults_the_install_runs_not_a_draft(boot, monkeypatch):
    """No settings file: the defaults run, so their never-configured outcome gets the snapshot and
    the one owner message; a draft whose rows differ decides nothing and gets none."""
    root, sent = boot
    monkeypatch.setattr(cfg, "SETTINGS_PATH", root / "absent" / "settings.json")
    monkeypatch.delenv(SUBAGENTS, raising=False)
    cfg.normalize_settings_raw({"OPENAI_API_KEY": "present"})  # a draft minting rows of its own: the OpenAI panel
    settings = cfg.load_settings()  # the server's read: no file, so the factory rows of the defaults run
    draft, defaults = cfg.review_pool_migrations_seen()
    assert draft.catalog_after != defaults.catalog_after == settings[SUBAGENTS]
    server_maintenance._startup_review_pool_notice(settings)
    (snapshot_file,) = _snapshots(root)
    assert json.loads(snapshot_file.read_text(encoding="utf-8"))["input_sha256"] == defaults.input_sha256
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert _notices(sent) == [m.owner_message(defaults, record["snapshot"])]
    assert _notices(sent)[0].startswith("⚙️ Review pool initialized. This install had no review settings")


def test_a_save_whose_receipt_failed_is_receipted_by_the_boot_that_finds_its_result_on_disk(boot, monkeypatch):
    """A save whose own receipt failed (logged; the save landed) leaves the migrated catalog on disk:
    that migration decides the document, so the boot still writes its snapshot and tells the owner once."""
    from ouroboros import review_pool_receipts as receipts

    root, sent = boot
    path = root / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", root)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    loaded = cfg.normalize_settings_raw(anton_document())
    with monkeypatch.context() as disk:
        disk.setattr(receipts, "read_snapshots", lambda data_dir: (_ for _ in ()).throw(OSError("disk")))
        cfg.save_settings(dict(loaded))
    assert SLOTS not in json.loads(path.read_text(encoding="utf-8")) and _snapshots(root) == []
    server_maintenance._startup_review_pool_notice(cfg.load_settings_lock_held(_settings_lock_held=False))
    (snapshot_file,) = _snapshots(root)
    assert json.loads(snapshot_file.read_text(encoding="utf-8"))["input_sha256"] == m.input_sha256(anton_document())
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["reported"] and _notices(sent) == [m.owner_message(cfg.review_pool_migrations_seen()[0], record["snapshot"])]


@pytest.mark.parametrize("wizard_saved", [True, False], ids=["wizard_saved_its_own_catalog", "document_unchanged"])
def test_a_receipt_deferred_past_the_wizard_never_calls_the_saved_catalog_the_environments(boot, monkeypatch,
                                                                                            wizard_saved):
    """NEW-H1. The fresh install's first boot has no settings file and no owner chat yet: the read
    seam mints the factory rows for the defaults, the boot receipts them (the defaults ARE what runs)
    and the message waits for a chat. The owner then finishes the wizard, which saves a document with a
    catalog of its own, and the next boot binds the chat. That deferred receipt no longer decides the
    document on disk, and the catalog that runs is the OWNER'S — ``OUROBOROS_SUBAGENTS`` is not in the
    process environment — so the record is closed without a message, never delivered as "the
    environment's pool is in force" because the catalog differs from the minted rows. Control: while
    the document is unchanged (still no file) the deferred receipt is delivered once at the first boot
    with a chat, as before. The environment direction — a catalog the environment really carries over
    an N-1 document — is pinned in ``test_review_pool_migration`` (N1)."""
    root, sent = boot
    state.save_state({})  # no owner chat bound yet
    path = root / "settings.json"
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    monkeypatch.delenv(SUBAGENTS, raising=False)
    first = cfg.load_settings()  # the server's first start: no file, the factory rows of the defaults run
    (defaults,) = cfg.review_pool_migrations_seen()
    server_maintenance._startup_review_pool_notice(first)
    (snapshot_file,) = _snapshots(root)
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["reported"] is None and _notices(sent) == []

    own = catalog(session_row("reviewer", MAIN, review_eligible=True))
    if wizard_saved:
        path.write_text(json.dumps({"OUROBOROS_MODEL": MAIN, SUBAGENTS: own}), encoding="utf-8")
    m._MIGRATIONS_SEEN.clear()  # the next start is a fresh process
    state.update_state(lambda st: st.__setitem__("owner_chat_id", 7))
    settings = cfg.load_settings()
    server_maintenance._startup_review_pool_notice(settings)
    server_maintenance._startup_review_pool_notice(settings)  # a supervisor revival adds nothing

    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["reported"] and _snapshots(root) == [snapshot_file], "the receipt itself stays: it is history"
    if wizard_saved:
        assert settings[SUBAGENTS] == own != defaults.catalog_after
        assert _notices(sent) == []
    else:
        assert settings[SUBAGENTS] == defaults.catalog_after
        assert _notices(sent) == [m.owner_message(defaults, record["snapshot"])]
