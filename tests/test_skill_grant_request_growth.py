"""A skill's key request grown by saved Settings keeps the owner's earlier approvals of the same bytes.

``requested_core_setting_keys`` asks a grant for a custom key only once that key is present in
Settings, so saving a new key a skill declares (from the Settings page or the protected secret form)
grows every declaring skill's request. The grown request must add the new key as missing and keep
every approval the owner already gave for those bytes: the reader (``grant_status_for_skill``), the
partial-approval writer (``save_skill_grants``) and a real consumer (a script's environment) agree.
New bytes still reset approvals, and nothing here grants a key or permission the owner did not.
A read that exposes values classifies them from the very document it read, so an owner save landing
mid-read (adding or deleting such a key) never hands the skill an ungranted value.
"""

from __future__ import annotations

import json
import pathlib
import threading
import types

import pytest

from ouroboros.contracts.skill_manifest import SkillManifest
from ouroboros.skill_loader import (
    LoadedSkill,
    SkillReviewState,
    grant_status_for_skill,
    load_skill_grants,
    requested_core_setting_keys,
    save_skill_grants,
)

HASH = "bytes-1"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from ouroboros import config

    path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    def write(**values):
        path.write_text(json.dumps(values), encoding="utf-8")

    write(MAILER_HOST_TOKEN="host-synthetic")
    return write


def _skill(tmp_path: pathlib.Path, name: str, env, *, permissions=None, content_hash=HASH) -> LoadedSkill:
    (tmp_path / name).mkdir(exist_ok=True)
    manifest = SkillManifest(name=name, description="grant growth", version="0.1", type="script",
                             env_from_settings=list(env), permissions=list(permissions or []))
    return LoadedSkill(name=name, skill_dir=tmp_path / name, manifest=manifest, content_hash=content_hash,
                       review=SkillReviewState(status="pass", content_hash=content_hash))


def _grant(drive: pathlib.Path, skill: LoadedSkill, keys, permissions=()):
    from ouroboros.skill_loader import requested_skill_permissions

    save_skill_grants(drive, skill.name, list(keys), content_hash=skill.content_hash,
                      requested_keys=requested_core_setting_keys(list(skill.manifest.env_from_settings)),
                      granted_permissions=list(permissions),
                      requested_permissions=requested_skill_permissions(list(skill.manifest.permissions), []))


def test_a_new_shared_key_joins_the_request_missing_and_keeps_other_approvals(tmp_path, settings):
    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"])
    other = _skill(tmp_path, "other", ["SHARED_API_KEY"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN"])
    before = grant_status_for_skill(drive, mailer)
    assert before["requested_keys"] == ["MAILER_HOST_TOKEN"] and before["granted_keys"] == ["MAILER_HOST_TOKEN"]
    assert grant_status_for_skill(drive, other)["requested_keys"] == []  # an absent custom key is not requested

    settings(MAILER_HOST_TOKEN="host-synthetic", SHARED_API_KEY="shared-synthetic")

    after = grant_status_for_skill(drive, mailer)
    assert after["requested_keys"] == ["MAILER_HOST_TOKEN", "SHARED_API_KEY"]
    assert after["granted_keys"] == ["MAILER_HOST_TOKEN"], "the earlier approval of the same bytes survives"
    assert after["missing_keys"] == ["SHARED_API_KEY"], "the newly present key is never granted by itself"
    assert after["all_granted"] is False and after["usable"] is False
    other_after = grant_status_for_skill(drive, other)
    assert other_after["granted_keys"] == [] and other_after["missing_keys"] == ["SHARED_API_KEY"]
    assert load_skill_grants(drive, "other")["granted_keys"] == []  # reading wrote nothing


def test_a_partial_approval_after_growth_adds_the_key_without_revoking_keys_or_permissions(tmp_path, settings):
    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"], permissions=["inject_chat"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN"], ["inject_chat"])
    settings(MAILER_HOST_TOKEN="host-synthetic", SHARED_API_KEY="shared-synthetic")

    _grant(drive, mailer, ["SHARED_API_KEY"])  # the owner approves only the new key
    stored = load_skill_grants(drive, "mailer")
    assert sorted(stored["granted_keys"]) == ["MAILER_HOST_TOKEN", "SHARED_API_KEY"]
    assert stored["granted_permissions"] == ["inject_chat"]
    status = grant_status_for_skill(drive, mailer)
    assert status["granted_keys"] == ["MAILER_HOST_TOKEN", "SHARED_API_KEY"] and status["all_granted"] is True


def test_the_writer_never_grants_beyond_the_current_request(tmp_path, settings):
    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN", "SHARED_API_KEY", "UNDECLARED_KEY"])
    # SHARED_API_KEY is absent, so it is not requested yet: approving it now records nothing.
    assert load_skill_grants(drive, "mailer")["granted_keys"] == ["MAILER_HOST_TOKEN"]
    settings(MAILER_HOST_TOKEN="host-synthetic", SHARED_API_KEY="shared-synthetic")
    assert grant_status_for_skill(drive, mailer)["missing_keys"] == ["SHARED_API_KEY"]


def test_new_bytes_still_reset_every_approval(tmp_path, settings):
    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"], permissions=["inject_chat"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN"], ["inject_chat"])
    changed = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"], permissions=["inject_chat"],
                     content_hash="bytes-2")
    status = grant_status_for_skill(drive, changed)
    assert status["granted_keys"] == [] and status["granted_permissions"] == []
    _grant(drive, changed, [])  # a write under the new bytes carries nothing from the old ones
    assert load_skill_grants(drive, "mailer")["granted_keys"] == []
    assert load_skill_grants(drive, "mailer")["granted_permissions"] == []


def test_a_script_environment_follows_the_kept_approval(tmp_path, settings):
    from ouroboros.tools.skill_exec import _scrub_env

    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN"])
    settings(MAILER_HOST_TOKEN="host-synthetic", SHARED_API_KEY="shared-synthetic")
    granted = grant_status_for_skill(drive, mailer)["granted_keys"]
    env = _scrub_env(list(mailer.manifest.env_from_settings), tmp_path / "state", "mailer", granted_keys=granted)
    assert env.get("MAILER_HOST_TOKEN") == "host-synthetic"
    assert "SHARED_API_KEY" not in env  # requested now, not granted: the script does not receive it


RACE_KEY, RACE_VALUE = "RACE_CUSTOM_SECRET", "synthetic-race-only-value"


@pytest.fixture
def loaded_reader(tmp_path, monkeypatch):
    """A real PluginAPI 2.4 extension that declares ``RACE_KEY``, loaded by the real loader over an empty document,
    with every settings read it makes (its own reader and the classification owner's) routed through
    ``reads.through``; ``reads.pause_first(thread)`` holds that thread's first read until ``reads.saved``."""
    from ouroboros import config, extension_loader
    from ouroboros.extension_registry_state import _extensions
    from tests._extension_loader_shared import _prepare_extension
    from tests._shared import clean_extension_runtime_state

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    loaded, repo, drive = _prepare_extension(tmp_path, "readrace", "def register(api):\n    pass\n",
                                             permissions=["read_settings"], env_from_settings=[RACE_KEY],
                                             extra_frontmatter='plugin_api: "2.4"\n')
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    config.SETTINGS_PATH.write_text("{}", encoding="utf-8")
    original = config.load_settings
    reads = types.SimpleNamespace(count=[], thread="", reached=threading.Event(), saved=threading.Event())

    def through(*, migrate=False, strict=False):
        document = original(migrate=migrate, strict=strict)
        if threading.current_thread().name == reads.thread:
            reads.count.append(strict)
            if not reads.reached.is_set():
                reads.reached.set()
                assert reads.saved.wait(5), "the owner's save never landed"
        return document

    reads.through = through
    reads.pause_first = lambda thread: setattr(reads, "thread", thread)
    monkeypatch.setattr(config, "load_settings", through)
    assert extension_loader.load_extension(loaded, through, drive_root=drive, repo_path=str(repo),
                                           _force_in_process=True) is None
    try:
        yield _extensions[loaded.name].api_instances[0], loaded, drive, reads
    finally:
        clean_extension_runtime_state()


def _owner_saves(change: str) -> None:
    """The owner's own writers: a secret saved onto the document, or a whole-document save without the key."""
    from ouroboros.gateway.owner_settings import _owner_update_settings, _owner_write_settings, fields_on_saved_document

    if change == "added":
        _owner_update_settings(fields_on_saved_document({RACE_KEY: RACE_VALUE}))
    else:
        _owner_write_settings({})  # the document held only that key


@pytest.mark.parametrize("change", ["added", "deleted"])
@pytest.mark.parametrize("reader", ["get_settings_state", "get_settings", "script_env"])
def test_an_owner_save_landing_mid_read_never_exposes_an_ungranted_custom_key(loaded_reader, monkeypatch, reader,
                                                                              change):
    """Each reader reads a document; the owner then adds (or, after an earlier save, deletes) a custom key the skill
    declares but was never granted; then the read goes on. The key's value never reaches the skill: its readers
    classify the mapping their values come from, so no second read can disagree with the first. Before this, the
    strict state classified first and read later (an added key leaked) and the ordinary read and a script's
    environment read first and classified later (a deleted key leaked)."""
    from ouroboros.skill_loader import find_skill
    from ouroboros.tools import skill_exec

    api, loaded, drive, reads = loaded_reader
    if change == "deleted":  # saved after the load, not granted: present and hidden until the owner deletes it
        _owner_saves("added")
        assert api.get_settings([RACE_KEY]) == {}
        assert api.get_settings_state([RACE_KEY]) == {"status": "ok", "values": {}, "missing": []}
    monkeypatch.setattr(skill_exec, "load_settings", reads.through)  # a script's own read, as the alias it uses
    read = {"get_settings_state": lambda: api.get_settings_state([RACE_KEY]),
            "get_settings": lambda: api.get_settings([RACE_KEY]),
            "script_env": lambda: skill_exec._scrub_env([RACE_KEY], drive / "state", loaded.name, granted_keys=[])}
    result, errors = [], []

    def run():
        try:
            result.append(read[reader]())
        except BaseException as exc:  # noqa: BLE001 — reported below
            errors.append(exc)

    reads.pause_first("skill-read")
    thread = threading.Thread(target=run, name="skill-read")
    thread.start()
    try:
        assert reads.reached.wait(5), "the read never made its settings read"
        _owner_saves(change)
    finally:
        reads.saved.set()
        thread.join(5)
    assert not thread.is_alive() and not errors, errors
    assert RACE_VALUE not in json.dumps(result, default=str), f"{reader} exposed the ungranted key: {result}"
    if reader == "get_settings_state":  # answered as of its one read: absent then (missing) or present and hidden
        assert result == [{"status": "ok", "values": {}, "missing": [RACE_KEY] if change == "added" else []}]
    assert len(reads.count) == 1, f"{reader} read the document {len(reads.count)} times for one answer"
    current = find_skill(drive, loaded.name, repo_path=str(loaded.skill_dir.parent))
    assert grant_status_for_skill(drive, current)["missing_keys"] == ([RACE_KEY] if change == "added" else [])
    reads.pause_first("")
    assert api.get_settings([RACE_KEY]) == {}  # after the save, as it reads now
    assert api.get_settings_state([RACE_KEY])["values"] == {}


def test_granted_and_empty_reads_keep_their_answers_with_one_read_or_none(monkeypatch, loaded_reader):
    """Positive controls: a granted custom key is read (state, ordinary read, script environment), each from one
    settings read; asking nothing readable answers the access alone, with no read and without the API's lock;
    unload closes it."""
    from ouroboros import extension_loader
    from ouroboros.extension_registry_state import _extensions
    from ouroboros.tools import skill_exec

    api, loaded, drive, reads = loaded_reader
    _owner_saves("added")
    _grant(drive, loaded, [RACE_KEY])
    extension_loader.unload_extension(loaded.name)
    assert extension_loader.load_extension(loaded, reads.through, drive_root=drive,
                                           repo_path=str(loaded.skill_dir.parent), _force_in_process=True) is None
    api = _extensions[loaded.name].api_instances[0]
    monkeypatch.setattr(skill_exec, "load_settings", reads.through)
    reads.reached.set()  # count only, never pause
    for read, expected in (
            (lambda: api.get_settings_state([RACE_KEY]), {"status": "ok", "values": {RACE_KEY: RACE_VALUE},
                                                          "missing": []}),
            (lambda: api.get_settings([RACE_KEY]), {RACE_KEY: RACE_VALUE}),
            (lambda: skill_exec._scrub_env([RACE_KEY], drive / "state", loaded.name,
                                           granted_keys=[RACE_KEY])[RACE_KEY], RACE_VALUE)):
        reads.count.clear()
        reads.pause_first(threading.current_thread().name)
        assert read() == expected
        assert len(reads.count) == 1, "one document read classifies and answers"
    reads.count.clear()
    held, release = threading.Event(), threading.Event()

    def hold_lock():
        with api._api_lock:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert held.wait(5)
        assert api.get_settings_state([]) == {"status": "ok", "values": {}, "missing": []}
        assert api.get_settings_state(["NOT_DECLARED"]) == {"status": "ok", "values": {}, "missing": []}
        assert reads.count == [], "nothing readable asked: no settings read"
    finally:
        release.set()
        holder.join(5)
    extension_loader.unload_extension(loaded.name)
    assert api.get_settings_state([]) == {"status": "access_closed", "values": {}, "missing": []}
    assert api.get_settings_state([RACE_KEY])["status"] == "access_closed" and api.get_settings([RACE_KEY]) == {}
