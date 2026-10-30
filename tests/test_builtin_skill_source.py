"""The built-in skill source (``ouroboros/skill_builtin.py``).

Built-ins are an ordinary source: installing one writes the install record every
installer writes (trusted by default), never a synthetic review verdict. A new install
gets only what its starter selection ticks; an existing install keeps what it has.
Built-ins ride app releases: a changed bundled version re-seeds an unmodified copy in
place (state, the owner's enablement and unchanged-request grants kept) and never
rewrites local changes; nothing resurrects a built-in the owner removed; a built-in the
app stops bundling is uninstalled when nobody changed it and keeps running as its owner's
copy when someone did.
"""

from __future__ import annotations

import io
import json
import logging
import pathlib
import subprocess
import tarfile
import textwrap

import pytest

SKILL_TEMPLATE = textwrap.dedent(
    """
    ---
    name: NAME
    description: Test fixture skill.
    version: VERSION
    type: instruction
    ---

    # NAME
    """
).strip() + "\n"


def _write_skill(parent: pathlib.Path, slug: str, version: str = "1.0.0") -> pathlib.Path:
    skill_dir = parent / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(SKILL_TEMPLATE.replace("NAME", slug).replace("VERSION", version),
                                        encoding="utf-8")
    return skill_dir


@pytest.fixture
def env(tmp_path, monkeypatch):
    """(seed_dir, drive_root) with built-in trust governed by the env flag: the
    settings path is pinned to a per-test file so another test's settings.json
    cannot decide it."""
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    drive_root = tmp_path / "data"
    (drive_root / "skills" / "native").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive_root / "settings.json")
    seed_dir = tmp_path / "repo_skills"
    seed_dir.mkdir()
    return seed_dir, drive_root


def _install(name, seed_dir, drive_root, **kwargs):
    from ouroboros.skill_builtin import install_builtin_skill

    return install_builtin_skill(name, drive_root=drive_root, seed_dir=seed_dir, **kwargs)


# --- install: an install record with the source's basis, no synthetic review ---------


def test_install_writes_trusted_record_and_the_gate_admits_exact_bytes(env):
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill, skill_state_dir_path

    seed_dir, drive_root = env
    _write_skill(seed_dir, "weather")
    result = _install("weather", seed_dir, drive_root, turn_on_when_ready=True, installed_by="owner")
    assert result["ok"] and result["basis"] == "trusted"
    target = drive_root / "skills" / "native" / "weather"
    assert (target / ".seed-origin").is_file()
    record = read_install_record(drive_root, "weather")
    assert record["source"]["id"] == "builtin"
    assert record["release"]["digest"] == record["base_digest"] == result["digest"]
    assert record["turn_on_when_ready"] is True
    assert record["official"] is True  # the bundled release is the app's own
    assert not (skill_state_dir_path(drive_root, "weather") / "review.json").exists()
    skill = find_skill(drive_root, "weather")
    gate = skill.review.gate_for(skill.content_hash)
    assert gate["executable_review"] is True and gate["blocking_reason"] == "trusted_source"
    assert not skill.enabled  # enablement is the turn-on reconcile's, inside a lifecycle job

    # An edit of the installed bytes leaves the release: no longer admitted.
    (target / "notes.txt").write_text("local edit\n", encoding="utf-8")
    edited = find_skill(drive_root, "weather")
    assert edited.review.gate_for(edited.content_hash)["executable_review"] is False


def test_install_with_trust_off_records_review_basis(env, monkeypatch):
    from ouroboros.skill_loader import find_skill

    seed_dir, drive_root = env
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "false")
    _write_skill(seed_dir, "weather")
    assert _install("weather", seed_dir, drive_root)["basis"] == "review"
    skill = find_skill(drive_root, "weather")
    assert skill.review.gate_for(skill.content_hash)["executable_review"] is False


def test_the_builtin_row_of_the_source_list_decides_the_basis(env, monkeypatch):
    from ouroboros.skill_builtin import builtin_trust

    seed_dir, drive_root = env
    row = {"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "review", "confirmed": True}
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([row]))
    _write_skill(seed_dir, "weather")
    assert _install("weather", seed_dir, drive_root)["basis"] == "review", "the owner set built-ins to review"
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "false")
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([{**row, "trust": "trusted"}]))
    assert builtin_trust() == "trusted", "a saved row outranks the older switch"
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([{**row, "id": "mine", "kind": "local", "path": "/x"}]))
    assert builtin_trust() == "review", "a list without a built-in row keeps the older switch"
    monkeypatch.delenv("OUROBOROS_SKILL_SOURCES")
    assert builtin_trust() == "review", "the derived row follows the older switch"


def test_install_auto_grants_only_under_the_owner_setting(env, monkeypatch):
    import ouroboros.config as config
    from ouroboros.skill_loader import load_skill_grants

    seed_dir, drive_root = env
    skill = _write_skill(seed_dir, "bridge")
    text = (skill / "SKILL.md").read_text(encoding="utf-8").replace(
        "type: instruction", "type: script\nruntime: python3\nenv_from_settings: [TELEGRAM_BOT_TOKEN]\n"
                             "scripts:\n  - name: run.py")
    (skill / "SKILL.md").write_text(text, encoding="utf-8")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(config, "get_auto_grant_enabled", lambda: False)
    assert _install("bridge", seed_dir, drive_root)["auto_granted_keys"] == []
    assert load_skill_grants(drive_root, "bridge")["granted_keys"] == []
    monkeypatch.setattr(config, "get_auto_grant_enabled", lambda: True)
    result = _install("bridge", seed_dir, drive_root)
    assert result["auto_granted_keys"] == ["TELEGRAM_BOT_TOKEN"]
    assert load_skill_grants(drive_root, "bridge")["granted_keys"] == ["TELEGRAM_BOT_TOKEN"]


def test_install_refuses_a_taken_name_and_a_user_managed_native_copy(env):
    seed_dir, drive_root = env
    _write_skill(seed_dir, "weather")
    _write_skill(drive_root / "skills" / "external", "weather")
    result = _install("weather", seed_dir, drive_root)
    assert result["ok"] is False and result["code"] == "name_taken"
    assert not (drive_root / "skills" / "native" / "weather").exists()

    _write_skill(seed_dir, "radio")
    user_copy = _write_skill(drive_root / "skills" / "native", "radio", version="9.9.9")
    result = _install("radio", seed_dir, drive_root)
    assert result["ok"] is False and "user-managed" in result["error"]
    assert "version: 9.9.9" in (user_copy / "SKILL.md").read_text(encoding="utf-8")
    assert _install("missing", seed_dir, drive_root)["ok"] is False


def test_startup_never_installs_a_built_in(env, monkeypatch):
    """A new install gets built-ins from its starter selection only; the startup hook
    re-seeds what is already installed and never adds or resurrects anything."""
    from ouroboros import skill_builtin

    seed_dir, drive_root = env
    _write_skill(seed_dir, "telegram")
    monkeypatch.setattr(skill_builtin, "bundled_skills_dir", lambda: seed_dir)
    assert skill_builtin.ensure_builtin_skills(drive_root) == 0
    assert not (drive_root / "skills" / "native" / "telegram").exists()
    assert _install("telegram", seed_dir, drive_root)["ok"]
    assert (drive_root / "skills" / "native" / "telegram").is_dir()


# --- re-seed in place on a changed bundled version ----------------------------------


def _reseed(seed_dir, drive_root, log=None):
    from ouroboros.skill_builtin import reseed_changed_builtins

    return reseed_changed_builtins(drive_root, seed_dir=seed_dir, log_obj=log or logging.getLogger("t.builtin"))


def test_reseed_skips_deleted_and_user_managed_and_identical(env):
    seed_dir, drive_root = env
    native = drive_root / "skills" / "native"
    _write_skill(seed_dir, "deleted", "2.0.0")
    _write_skill(seed_dir, "mine", "2.0.0")
    _write_skill(native, "mine", "1.0.0")  # no launcher marker: the owner's
    _write_skill(seed_dir, "same", "1.0.0")
    _install("same", seed_dir, drive_root)
    assert _reseed(seed_dir, drive_root) == 0
    assert not (native / "deleted").exists()
    assert "version: 1.0.0" in (native / "mine" / "SKILL.md").read_text(encoding="utf-8")


def test_reseed_replaces_payload_keeps_state_and_owner_choice(env):
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill, load_enabled, save_enabled

    seed_dir, drive_root = env
    _write_skill(seed_dir, "weather", "1.0.0")
    assert _install("weather", seed_dir, drive_root, turn_on_when_ready=True)["ok"]
    target = drive_root / "skills" / "native" / "weather"
    state = drive_root / "state" / "skills" / "weather"
    (state / "settings.json").write_text('{"CITY":"Paris"}', encoding="utf-8")
    save_enabled(drive_root, "weather", False, actor="owner_ui")
    _write_skill(seed_dir, "weather", "0.9.0")  # a downgrade is a change too (opaque versions)
    (seed_dir / "weather" / "forecast.md").write_text("new in 0.9.0\n", encoding="utf-8")
    assert _reseed(seed_dir, drive_root) == 1
    assert "version: 0.9.0" in (target / "SKILL.md").read_text(encoding="utf-8")
    assert (target / "forecast.md").is_file()
    assert (state / "settings.json").read_text(encoding="utf-8") == '{"CITY":"Paris"}'
    assert load_enabled(drive_root, "weather") is False
    record = read_install_record(drive_root, "weather")
    skill = find_skill(drive_root, "weather")
    assert record["release"]["digest"] == skill.content_hash and record["turn_on_when_ready"] is True
    assert skill.review.gate_for(skill.content_hash)["blocking_reason"] == "trusted_source"


def test_reseed_keeps_a_copy_with_local_changes(env, caplog):
    """An app update never rewrites the owner's edits: a built-in whose bytes differ from
    its recorded base stays as it is (Modified); the owner chooses on update."""
    from ouroboros.skill_install_record import read_install_record

    seed_dir, drive_root = env
    _write_skill(seed_dir, "weather", "1.0.0")
    assert _install("weather", seed_dir, drive_root)["ok"]
    target = drive_root / "skills" / "native" / "weather"
    record = read_install_record(drive_root, "weather")
    (target / "user_notes.txt").write_text("the owner's change\n", encoding="utf-8")
    _write_skill(seed_dir, "weather", "2.0.0")
    log = logging.getLogger("t.builtin.modified")
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _reseed(seed_dir, drive_root, log) == 0
    assert "has local changes" in caplog.text
    assert "version: 1.0.0" in (target / "SKILL.md").read_text(encoding="utf-8")
    assert (target / "user_notes.txt").is_file()
    assert read_install_record(drive_root, "weather")["release"] == record["release"]
    (target / "user_notes.txt").unlink()  # back to the recorded base: the next start re-seeds
    assert _reseed(seed_dir, drive_root, log) == 1
    assert "version: 2.0.0" in (target / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("drift", [False, True])
def test_same_version_hash_drift_is_only_diagnostic(env, caplog, drift):
    seed_dir, drive_root = env
    seed = _write_skill(seed_dir, "weather")
    assert _install("weather", seed_dir, drive_root)["ok"]
    (seed / "payload.txt").write_text("seed", encoding="utf-8")
    installed = drive_root / "skills" / "native" / "weather"
    (installed / "payload.txt").write_text("local" if drift else "seed", encoding="utf-8")
    before = {str(p.relative_to(drive_root)): p.read_bytes() for p in drive_root.rglob("*") if p.is_file()}
    log = logging.getLogger("t.builtin.drift")
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _reseed(seed_dir, drive_root, log) == 0
    after = {str(p.relative_to(drive_root)): p.read_bytes() for p in drive_root.rglob("*") if p.is_file()}
    assert after == before
    assert ("installed files retained because the manifest version is unchanged" in caplog.text) is drift


def test_hash_comparison_failure_does_not_stop_next_skill(env, caplog, monkeypatch):
    import ouroboros.skill_loader as loader

    seed_dir, drive_root = env
    for name in ("first", "second"):
        _write_skill(seed_dir, name, "1.0.0")
        assert _install(name, seed_dir, drive_root)["ok"]
    _write_skill(seed_dir, "second", "2.0.0")
    original = loader.compute_content_hash

    def unreadable(path, **kwargs):
        if pathlib.Path(path).name == "first":
            raise OSError("private error detail")
        return original(path, **kwargs)

    monkeypatch.setattr(loader, "compute_content_hash", unreadable)
    log = logging.getLogger("t.builtin.hash")
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _reseed(seed_dir, drive_root, log) == 1
    assert "comparison unavailable (OSError)" in caplog.text
    assert "private error detail" not in caplog.text
    assert "version: 2.0.0" in (drive_root / "skills" / "native" / "second" / "SKILL.md").read_text(encoding="utf-8")


def test_a_retired_built_in_nobody_changed_is_uninstalled_and_its_grants_survive(env):
    from ouroboros.skill_builtin import release_removed_builtins
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill, skill_state_dir
    from ouroboros.skill_uninstall_state import UNINSTALL_TOMBSTONE_FILENAME

    seed_dir, drive_root = env
    seed = _write_skill(seed_dir, "weather")
    assert _install("weather", seed_dir, drive_root)["ok"]
    grants = skill_state_dir(drive_root, "weather") / "grants.json"
    grants.write_text('{"granted_keys": ["WEATHER_KEY"]}', encoding="utf-8")
    import shutil

    shutil.rmtree(seed)
    assert release_removed_builtins(drive_root, seed_dir=seed_dir, retire=False) is False, "a failed re-seed defers it"
    assert (drive_root / "skills" / "native" / "weather" / ".seed-origin").is_file(), "still marked for the next boot"
    assert release_removed_builtins(drive_root, seed_dir=seed_dir) is True
    assert not (drive_root / "skills" / "native" / "weather").exists()
    assert find_skill(drive_root, "weather") is None and read_install_record(drive_root, "weather") is None
    assert (skill_state_dir(drive_root, "weather") / UNINSTALL_TOMBSTONE_FILENAME).is_file()
    assert grants.is_file(), "the owner's grants outlive the payload"


def test_a_retired_built_in_whose_name_another_folder_uses_stays_as_the_owners_copy(env):
    """The uninstall refuses a name two folders share; the copy stays, released, and the pass completes."""
    from ouroboros.skill_builtin import release_removed_builtins

    seed_dir, drive_root = env
    seed = _write_skill(seed_dir, "weather")
    assert _install("weather", seed_dir, drive_root)["ok"]
    _write_skill(drive_root / "skills" / "external", "weather")
    import shutil

    shutil.rmtree(seed)
    assert release_removed_builtins(drive_root, seed_dir=seed_dir) is True, "a refused uninstall does not loop every boot"
    target = drive_root / "skills" / "native" / "weather"
    assert target.is_dir() and not (target / ".seed-origin").exists()
    assert (drive_root / "skills" / "external" / "weather" / "SKILL.md").is_file()


def test_a_retired_built_in_someone_edited_keeps_running_as_the_owners_copy(env):
    from ouroboros.skill_builtin import release_removed_builtins
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill

    seed_dir, drive_root = env
    seed = _write_skill(seed_dir, "weather")
    assert _install("weather", seed_dir, drive_root)["ok"]
    target = drive_root / "skills" / "native" / "weather"
    (target / "SKILL.md").write_text((target / "SKILL.md").read_text(encoding="utf-8") + "\nMy own line.\n",
                                     encoding="utf-8")
    import shutil

    shutil.rmtree(seed)
    for retire in (False, True):  # an edited copy is the owner's whether or not this pass may retire
        release_removed_builtins(drive_root, seed_dir=seed_dir, retire=retire)
        assert target.is_dir() and not (target / ".seed-origin").exists()
    assert read_install_record(drive_root, "weather") is not None
    skill = find_skill(drive_root, "weather")
    assert skill.source == "external" and "My own line." in (target / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("text,expected", [
    ("---\nname: x\ndescription: y\nversion: 0.2.0 # bumped manually\ntype: instruction\n---\n", "0.2.0"),
    ("this is not a manifest at all", ""),
])
def test_version_reads_through_the_shared_parser(tmp_path, text, expected):
    from ouroboros.skill_builtin import _version

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(text, encoding="utf-8")
    assert _version(skill) == expected
    json_skill = tmp_path / "json_skill"
    json_skill.mkdir()
    (json_skill / "skill.json").write_text('{"name":"foo","description":"y","version":"0.2.0","type":"instruction"}',
                                           encoding="utf-8")
    assert _version(json_skill) == "0.2.0"
    assert _version(tmp_path / "missing") == ""


def test_bundled_telegram_reseed_delivers_the_current_version(tmp_path, monkeypatch):
    """The bundled ``SKILL.md`` version is the RESYNC KEY: a payload change shipped
    without a bump never reaches an existing install. Re-pin this whenever
    skills/telegram changes."""
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", tmp_path / "settings.json")
    seed_dir = pathlib.Path(__file__).resolve().parents[1] / "skills"
    drive_root = tmp_path / "data"
    installed = _write_skill(drive_root / "skills" / "native", "telegram", version="1.2.0")
    (installed / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    from ouroboros import skill_builtin
    from ouroboros.skill_loader import compute_content_hash

    digest = compute_content_hash(installed)  # an unmodified recorded install (a bare legacy copy is kept)
    skill_builtin._write_record(drive_root, "telegram", version="1.2.0", digest=digest, base_digest=digest,
                                turn_on_when_ready=False, installed_by="ouroboros")
    assert _reseed(seed_dir, drive_root) == 1
    assert "version: 1.3.9" in (installed / "SKILL.md").read_text(encoding="utf-8")
    for path in ("plugin.py", "lib/telegram_api.py", "lib/telegram_quiz.py", "lib/telegram_state.py"):
        assert (installed / path).read_bytes() == (seed_dir / "telegram" / path).read_bytes()


@pytest.mark.serial
@pytest.mark.parametrize("name,source,old_version", [
    ("telegram", "7f71fe8d29c975e25b35a235e1c821e1b547674f", "1.3.1"),
    ("telegram", "6014e8896dabe371f5c06cf8adc5dc9902f53c2d", "1.3.2"),
    ("telegram", "08ed0f824df498a1c43c8ff9e04dd42e5c64ee76", "1.3.3"),
    ("unix_computer_use", "ec3fc6e5b5ae5ee99a746b5e1a0f6343a41926d4", "0.4.2"),
    ("unix_computer_use", "1e237c1bc89f0600a7362f69eddafdff4ddd1f2d", "0.4.3"),
    ("unix_computer_use", "9e2bd962903044a6516917bcb10a933b2d1acafd", "0.5.0"),
])
def test_reseed_from_a_real_previous_seed(tmp_path, monkeypatch, name, source, old_version):
    """Full preceding payload from history: the re-seed delivers the bundled bytes,
    keeps the state directory, and the new release is admitted as trusted."""
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill, save_enabled

    repo = pathlib.Path(__file__).resolve().parents[1]
    drive = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive / "settings.json")
    # The Unix release applies on macOS/Linux only; pin a host it applies to.
    monkeypatch.setattr("ouroboros.contracts.skill_manifest.host_platform", lambda: "linux")
    installed = drive / "skills" / "native" / name
    installed.mkdir(parents=True)
    archived = subprocess.run(["git", "archive", f"{source}:skills/{name}"], cwd=repo,
                              capture_output=True, check=True, timeout=30)
    with tarfile.open(fileobj=io.BytesIO(archived.stdout)) as archive:
        for member in archive:
            if member.isdir():
                continue
            target = installed / member.name
            assert member.isfile() and target.resolve().is_relative_to(installed.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    (installed / ".seed-origin").write_text(f"seeded_from={source}\n", encoding="utf-8")
    save_enabled(drive, name, False)
    state = drive / "state" / "skills" / name
    (state / "settings.json").write_text('{"TELEGRAM_CHAT_ID":"12345"}', encoding="utf-8")
    preserved = {path.name: path.read_bytes() for path in state.iterdir() if path.is_file()}
    assert old_version in (installed / "SKILL.md").read_text(encoding="utf-8")
    assert _reseed(repo / "skills", drive) == 1
    skill = find_skill(drive, name)
    from ouroboros.skill_loader import compute_content_hash

    assert skill.content_hash == compute_content_hash(repo / "skills" / name)
    for path, payload in preserved.items():
        assert (state / path).read_bytes() == payload
    assert read_install_record(drive, name)["release"]["digest"] == skill.content_hash
    assert skill.review.gate_for(skill.content_hash)["blocking_reason"] == "trusted_source"


# --- the one-time conversion gives existing launcher seeds their install record ------


def _legacy_native_seed(drive_root, seed_dir, name="weather", version="1.0.0", *, edit_after=False):
    from ouroboros.skill_identity import legacy_content_hashes

    _write_skill(seed_dir, name, version)
    installed = _write_skill(drive_root / "skills" / "native", name, version)
    (installed / ".seed-origin").write_text("seeded_from=skills\n", encoding="utf-8")
    legacy = legacy_content_hashes(installed)[0]
    state = drive_root / "state" / "skills" / name
    state.mkdir(parents=True, exist_ok=True)
    (state / "review.json").write_text(json.dumps({
        "status": "clean", "content_hash": legacy, "findings": [],
        "reviewer_models": ["repo_commit_gate"], "review_profile": "native_seed"}), encoding="utf-8")
    if edit_after:
        (installed / "extra.txt").write_text("owner edit\n", encoding="utf-8")
    return installed, legacy


def test_conversion_records_the_trusted_release_of_an_unchanged_seed(env, monkeypatch):
    from ouroboros import skill_builtin
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill

    seed_dir, drive_root = env
    monkeypatch.setattr(skill_builtin, "bundled_skills_dir", lambda: seed_dir)
    _legacy_native_seed(drive_root, seed_dir)
    skill_builtin.ensure_builtin_skills(drive_root)
    skill = find_skill(drive_root, "weather")
    record = read_install_record(drive_root, "weather")
    assert record["basis"] == "trusted" and record["converted_from"] == ".seed-origin"
    assert record["release"]["digest"] == record["base_digest"] == skill.content_hash
    assert skill.review.content_hash == skill.content_hash  # the verdict moved with the bytes
    assert skill.review.gate_for(skill.content_hash)["executable_review"] is True


def test_conversion_leaves_an_edited_seed_unreviewed(env, monkeypatch):
    from ouroboros import skill_builtin
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill

    seed_dir, drive_root = env
    monkeypatch.setattr(skill_builtin, "bundled_skills_dir", lambda: seed_dir)
    _, legacy = _legacy_native_seed(drive_root, seed_dir, edit_after=True)
    skill_builtin.ensure_builtin_skills(drive_root)
    skill = find_skill(drive_root, "weather")
    record = read_install_record(drive_root, "weather")
    assert record["release"]["digest"] == record["base_digest"] == ""
    assert record["modified"] is True and record["legacy"] is True
    assert skill.review.content_hash == legacy  # stale stays stale
    assert skill.review.gate_for(skill.content_hash)["executable_review"] is False


@pytest.mark.parametrize("edit_after", [False, True])
def test_first_start_preserves_proven_edits_and_reseeds_unchanged_launcher_copy(env, monkeypatch, edit_after):
    """A stale native-seed review proves local edits even when the original base is unknown."""
    from ouroboros import skill_builtin
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill

    seed_dir, drive_root = env
    monkeypatch.setattr(skill_builtin, "bundled_skills_dir", lambda: seed_dir)
    installed, _ = _legacy_native_seed(drive_root, seed_dir, edit_after=edit_after)
    _write_skill(seed_dir, "weather", "2.0.0")
    assert skill_builtin.ensure_builtin_skills(drive_root) == (0 if edit_after else 1)
    assert f"version: {'1.0.0' if edit_after else '2.0.0'}" in (installed / "SKILL.md").read_text(encoding="utf-8")
    skill = find_skill(drive_root, "weather")
    record = read_install_record(drive_root, "weather")
    if edit_after:
        assert record["modified"] is True and record["base_digest"] == ""
        assert (installed / "extra.txt").read_text(encoding="utf-8") == "owner edit\n"
        assert not skill.review.gate_for(skill.content_hash)["executable_review"]
    else:
        assert record["base_digest"] == record["release"]["digest"] == skill.content_hash
        assert skill.review.gate_for(skill.content_hash)["blocking_reason"] == "trusted_source"


def test_legacy_builtin_without_base_is_not_reseeded(env, monkeypatch):
    """No review and no recorded base: nothing proves the copy is the shipped one, so the
    version re-seed keeps it (and the owner's edit) instead of overwriting it."""
    from ouroboros import skill_builtin

    seed_dir, drive = env
    monkeypatch.setattr(skill_builtin, "bundled_skills_dir", lambda: seed_dir)
    installed, _ = _legacy_native_seed(drive, seed_dir, edit_after=True)
    (drive / "state" / "skills" / "weather" / "review.json").unlink()
    _write_skill(seed_dir, "weather", "2.0.0")
    assert skill_builtin.ensure_builtin_skills(drive) == 0
    assert "version: 1.0.0" in (installed / "SKILL.md").read_text(encoding="utf-8")
    assert (installed / "extra.txt").read_text(encoding="utf-8") == "owner edit\n"


@pytest.mark.parametrize("edited", [False, True])
def test_a_legacy_copy_is_reseeded_only_when_its_committed_release_proves_it_unchanged(env, monkeypatch, edited):
    import subprocess

    from ouroboros import skill_builtin

    seed_dir, drive = env
    monkeypatch.setattr(skill_builtin, "bundled_skills_dir", lambda: seed_dir)
    installed, _ = _legacy_native_seed(drive, seed_dir, edit_after=edited)
    (drive / "state" / "skills" / "weather" / "review.json").unlink()
    repo = seed_dir.parent

    def git(*args):
        subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True,
                       env={**__import__("os").environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})

    git("init", "-q")
    git("add", "repo_skills")
    git("commit", "-q", "-m", "release 1.0.0 of weather")  # the bundle's own history carries the old release
    _write_skill(seed_dir, "weather", "2.0.0")
    assert skill_builtin.ensure_builtin_skills(drive) == (0 if edited else 1)
    assert f"version: {'1.0.0' if edited else '2.0.0'}" in (installed / "SKILL.md").read_text(encoding="utf-8")
