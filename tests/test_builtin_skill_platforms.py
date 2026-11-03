"""Bundled built-ins by platform: what this host is offered, keeps, shows and readies.

These run the built-in source's real install, re-seed, acceptance and dependency step over
the payloads in ``repo/skills`` (and, for updates, over a real earlier Unix release from Git
history). The host platform is pinned through ``skill_manifest.host_platform``; nothing here
touches a desktop, a network or the live data root. Real Windows UI Automation is exercised
only by ``tests/test_windows_computer_use_native.py``.
"""

from __future__ import annotations

import io
import json
import logging
import pathlib
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from ouroboros import skill_builtin
from ouroboros.skill_builtin import _dependency_pass, install_builtin_skill, reseed_changed_builtins
from ouroboros.skill_dependencies import skill_deps_not_ready
from ouroboros.skill_loader import (
    SkillReviewState,
    discover_skills,
    find_skill,
    load_enabled,
    save_enabled,
    save_review_state,
)
from ouroboros.skill_peer_inventory import discover_skill_peers
from tests._shared import clean_extension_runtime_state, current_python_env

REPO = pathlib.Path(__file__).resolve().parents[1]
SEEDS = REPO / "skills"
UNIX = "unix_computer_use"
WINDOWS = "windows_computer_use"
EVERYWHERE = ["ui_design", "templates", "device_voice",  # on by default on every host
              "live_conversation", "gemini_live", "grok_realtime", "openai_realtime"]  # inert until Calls (beta)
UNIX_043 = "1e237c1bc89f0600a7362f69eddafdff4ddd1f2d"  # official history: os any, permissions with net
log = logging.getLogger(__name__)


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    root = tmp_path / "data"
    (root / "skills").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", root / "settings.json")
    return root


def on(monkeypatch, platform: str) -> None:
    monkeypatch.setattr("ouroboros.contracts.skill_manifest.host_platform", lambda: platform)


def reseed(data_root, seeds=SEEDS):
    return reseed_changed_builtins(data_root, seed_dir=seeds, log_obj=log)


def names(data_root):
    return {skill.name for skill in discover_skills(data_root)}


def assert_peer_matches(data_root):
    full = {skill.name: (skill.enabled, tuple(skill.conflicts)) for skill in discover_skills(data_root)}
    peers = {skill.name: (skill.enabled, skill.conflicts) for skill in discover_skill_peers(data_root)}
    assert peers == full


def bumped_seeds(tmp_path, name):
    """A copy of the bundled tree in which ``name`` has a newer version (a later app release)."""
    seeds = tmp_path / "later" / "skills"
    shutil.copytree(SEEDS, seeds, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = seeds / name / "SKILL.md"
    version = skill_builtin._version(seeds / name)
    manifest.write_text(manifest.read_text(encoding="utf-8").replace(f"version: {version}", "version: 9.9.9", 1),
                        encoding="utf-8")
    return seeds


def git_payload(source: str, path: str, dest: pathlib.Path) -> None:
    archived = subprocess.run(["git", "archive", f"{source}:{path}"], cwd=REPO,
                              capture_output=True, check=True, timeout=30)
    with tarfile.open(fileobj=io.BytesIO(archived.stdout)) as archive:
        for member in archive:
            if member.isfile():
                target = dest / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())


def earlier_install(data_root, *, unix_marker=True) -> pathlib.Path:
    """An install from an earlier release: Unix 0.4.3 (unrestricted, with ``net``) seeded."""
    native = data_root / "skills" / "native"
    native.mkdir(parents=True, exist_ok=True)
    git_payload(UNIX_043, f"skills/{UNIX}", native / UNIX)
    if unix_marker:
        (native / UNIX / ".seed-origin").write_text(f"seeded_from={UNIX_043}\n", encoding="utf-8")
    state = data_root / "state" / "skills" / UNIX
    state.mkdir(parents=True, exist_ok=True)
    (state / "connections.json").write_text('{"active": "bench", "connections": {"bench": {"backend": '
                                            '"osworld_http", "target": "http://10.0.0.5:5000"}}}\n', encoding="utf-8")
    return native


def state_bytes(data_root, name):
    state = data_root / "state" / "skills" / name
    return {path.name: path.read_bytes() for path in state.iterdir() if path.is_file()}


# ── what each host is offered ──

@pytest.mark.parametrize("platform,own,other", [
    ("darwin", UNIX, WINDOWS), ("linux", UNIX, WINDOWS), ("win32", WINDOWS, UNIX),
])
def test_each_host_is_offered_and_installs_only_its_own_computer_use_skill(data, monkeypatch, platform, own, other):
    from ouroboros.skill_source_catalog import builtin_snapshot
    from ouroboros.skill_starter import starter_groups

    on(monkeypatch, platform)
    offered = {entry["name"] for entry in builtin_snapshot(REPO)["entries"]}
    assert {own, "telegram"} <= offered and other not in offered
    rows = [{"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True}]
    starter = json.dumps(starter_groups(data, rows=rows, repo_dir=REPO))
    assert f"builtin/{own}" in starter and f"builtin/{other}" not in starter

    refused = install_builtin_skill(other, drive_root=data, seed_dir=SEEDS)
    assert refused["ok"] is False and refused["code"] == "other_platform"
    assert not (data / "skills" / "native" / other).exists()
    installed = install_builtin_skill(own, drive_root=data, seed_dir=SEEDS, installed_by="owner")
    assert installed["ok"], installed
    skill = find_skill(data, own)
    assert skill.review.gate_for(skill.content_hash)["blocking_reason"] == "trusted_source"
    assert own in names(data) and other not in names(data)
    assert_peer_matches(data)


# ── the Unix copy kept on Windows for remote machines ──

@pytest.mark.parametrize("actor", ["owner_ui", None])
@pytest.mark.parametrize("backend", ["osworld_http", "ssh_macos"])
def test_a_legacy_unix_remote_choice_on_windows_survives_off_reseed_and_on(data, monkeypatch, tmp_path, actor, backend):
    from ouroboros import extension_loader, skill_lifecycle_actions

    on(monkeypatch, "win32")
    native = earlier_install(data)
    state = data / "state" / "skills" / UNIX
    if actor is None:
        # Unknown provenance: manual ON and inherited early auto-ON look alike.
        (state / "enabled.json").write_text('{"enabled": true}\n', encoding="utf-8")
    else:
        save_enabled(data, UNIX, True, actor=actor)
    registry = {"active": "bench", "connections": {"bench": {
        "backend": backend, "target": "http://10.0.0.5:5000", "ssh_alias": "bench-mac", "enabled": True}}}
    (state / "connections.json").write_text(json.dumps(registry) + "\n", encoding="utf-8")
    (state / "active_connection.txt").write_text("bench\n", encoding="utf-8")
    retained = {name: (state / name).read_bytes() for name in ("connections.json", "active_connection.txt", "enabled.json")}
    assert reseed(data) == 1
    assert not (native / WINDOWS).exists()  # a re-seed alone installs nothing new
    assert_peer_matches(data)
    assert UNIX in names(data)
    assert "legacy_remote_platform=win32" in (native / UNIX / ".seed-origin").read_text(encoding="utf-8").splitlines()
    assert all((state / name).read_bytes() == value for name, value in retained.items())

    # Real extension registration and resolved handlers; only the actual remote
    # transport is replaced. This never opens a desktop, SSH session or socket.
    calls = []

    def exercise_remote():
        extension_loader.reload_all(data, lambda: {})
        assert UNIX in extension_loader.snapshot()["extensions"]
        module = extension_loader._extension_modules[UNIX]

        def osworld(self, conn, argv, *, timeout):
            calls.append(("osworld_http", conn, argv, timeout))
            return {"status": 200, "result": {"status": "success", "returncode": 0, "output": "remote-ok"}}

        def ssh(self, conn, command, *, timeout):
            calls.append(("ssh_macos", conn, command, timeout))
            return 0, "remote-ok", ""

        monkeypatch.setattr(module._ComputerUse, "_osworld_execute", osworld)
        monkeypatch.setattr(module._ComputerUse, "_ssh_run", ssh)
        tool = extension_loader.get_tool(extension_loader.extension_surface_name(UNIX, "remote_exec"))
        assert tool is not None
        result = json.loads(tool["handler"](command="printf remote-ok"))
        assert result["ok"] and result["backend"] == backend
        assert calls[-1][0] == backend
        assert calls[-1][1] == registry["connections"]["bench"]
        assert calls[-1][2] == (["bash", "-lc", "printf remote-ok"] if backend == "osworld_http" else "printf remote-ok")

    clean_extension_runtime_state()
    try:
        exercise_remote()
        disabled = skill_lifecycle_actions._toggle(find_skill(data, UNIX), data, "", False, "owner_ui", "test OFF")
        assert disabled["ok"] and not disabled["enabled"]
        off = (state / "enabled.json").read_bytes()
        assert extension_loader.get_tool(extension_loader.extension_surface_name(UNIX, "remote_exec")) is None
        clean_extension_runtime_state()  # the next boot has no old module/registration
        assert reseed(data) == 0
        assert find_skill(data, UNIX) is not None and not find_skill(data, UNIX).enabled
        assert_peer_matches(data)
        extension_loader.reload_all(data, lambda: {})
        assert UNIX not in extension_loader.snapshot()["extensions"]
        # A later release's re-seed keeps the eligibility AND the newer OFF.
        assert reseed(data, bumped_seeds(tmp_path, UNIX)) == 1
        assert (state / "enabled.json").read_bytes() == off and not load_enabled(data, UNIX)
        assert find_skill(data, UNIX) is not None
        assert_peer_matches(data)
        enabled = skill_lifecycle_actions._toggle(find_skill(data, UNIX), data, "", True, "owner_ui", "intentional ON")
        assert enabled["ok"] and enabled["enabled"], enabled
        exercise_remote()
        assert all((state / name).read_bytes() == retained[name] for name in ("connections.json", "active_connection.txt"))
        clean_extension_runtime_state()
        shutil.rmtree(native / UNIX)
        assert reseed(data) == 0 and find_skill(data, UNIX) is None
    finally:
        clean_extension_runtime_state()


@pytest.mark.parametrize("choice", [{"enabled": False, "actor": "owner_ui"},
                                    {"enabled": True, "actor": "native_seed"}, {"enabled": "true"}])
def test_an_old_unix_copy_without_a_real_on_choice_stays_hidden_on_windows(data, monkeypatch, choice):
    on(monkeypatch, "win32")
    native = earlier_install(data)
    state = data / "state" / "skills" / UNIX
    (state / "enabled.json").write_text(json.dumps(choice), encoding="utf-8")
    before = (state / "enabled.json").read_bytes()

    assert reseed(data) == 1

    assert UNIX not in names(data) and find_skill(data, UNIX) is None
    assert_peer_matches(data)
    assert "legacy_remote_platform" not in (native / UNIX / ".seed-origin").read_text(encoding="utf-8")
    assert (state / "enabled.json").read_bytes() == before


def test_legacy_visibility_confers_no_trust_when_builtin_trust_is_off(data, monkeypatch):
    from ouroboros import extension_loader

    on(monkeypatch, "win32")
    earlier_install(data)
    save_enabled(data, UNIX, True, actor="owner_ui")
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "false")
    assert reseed(data) == 1
    legacy = find_skill(data, UNIX)
    assert legacy is not None and legacy.enabled
    assert not legacy.review.gate_for(legacy.content_hash)["executable_review"]
    assert_peer_matches(data)
    clean_extension_runtime_state()
    try:
        assert extension_loader.load_extension(legacy, lambda: {}, drive_root=data) is not None
        assert UNIX not in extension_loader.snapshot()["extensions"]
        assert extension_loader.get_tool(extension_loader.extension_surface_name(UNIX, "remote_exec")) is None
    finally:
        clean_extension_runtime_state()


@pytest.mark.parametrize("platform,name", [("darwin", UNIX), ("win32", WINDOWS)])
def test_an_explicit_off_survives_a_later_release(data, monkeypatch, tmp_path, platform, name):
    on(monkeypatch, platform)
    assert install_builtin_skill(name, drive_root=data, seed_dir=SEEDS, installed_by="owner")["ok"]
    save_enabled(data, name, False, actor="owner_ui")

    assert reseed(data, bumped_seeds(tmp_path, name)) == 1

    assert skill_builtin._version(data / "skills" / "native" / name) == "9.9.9"
    assert load_enabled(data, name) is False


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_a_markerless_native_payload_stays_its_owners(data, monkeypatch, platform):
    on(monkeypatch, platform)
    native = earlier_install(data, unix_marker=False)
    save_enabled(data, UNIX, True, actor="owner_ui")
    before = state_bytes(data, UNIX)

    assert reseed(data) == 0

    assert skill_builtin._version(native / UNIX) == "0.4.3"
    assert find_skill(data, UNIX).source == "external"
    assert state_bytes(data, UNIX) == before
    assert_peer_matches(data)


def test_another_platforms_copy_left_by_an_earlier_release_stays_hidden_and_inert(data, monkeypatch):
    """An earlier launcher copied every bundled seed; on macOS that left a Windows skill on disk."""
    on(monkeypatch, "darwin")
    native = data / "skills" / "native"
    shutil.copytree(SEEDS / WINDOWS, native / WINDOWS, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The narrow migration line is not a generic foreign-copy exemption.
    (native / WINDOWS / ".seed-origin").write_text("seeded_from=skills\nlegacy_remote_platform=win32\n",
                                                  encoding="utf-8")
    save_enabled(data, WINDOWS, True, actor="native_seed")
    before = state_bytes(data, WINDOWS)
    installs = []
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, *args, **kwargs: installs.append(packages) or [])
    from ouroboros import extension_loader
    from ouroboros.skill_loader import summarize_skills

    clean_extension_runtime_state()
    try:
        results = extension_loader.reload_all(data, lambda: {})
        live = extension_loader.snapshot()["extensions"]
    finally:
        clean_extension_runtime_state()

    assert WINDOWS not in names(data) and find_skill(data, WINDOWS) is None
    assert_peer_matches(data)
    assert WINDOWS not in results and WINDOWS not in live
    assert WINDOWS not in json.dumps(summarize_skills(data))
    assert reseed(data) == 0
    assert _dependency_pass(data, [WINDOWS, UNIX], log) == [] and installs == []
    assert state_bytes(data, WINDOWS) == before


# ── the dependency step bytes accepted without a review get instead of a paid review ──

@pytest.fixture
def windows_installed(data, monkeypatch):
    """A Windows install that asked to turn on when ready: accepted, off, comtypes not ready yet."""
    on(monkeypatch, "win32")
    assert install_builtin_skill(WINDOWS, drive_root=data, seed_dir=SEEDS, turn_on_when_ready=True,
                                 installed_by="owner")["ok"]
    installs, reconciles = [], []
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, env_root, *args, **kwargs: installs.append(list(packages)) or current_python_env(env_root) or [])
    monkeypatch.setattr("ouroboros.extension_loader.reconcile_extension",
                        lambda name, *args, **kwargs: reconciles.append(name) or {"action": "extension_loaded"})
    from ouroboros import skill_lifecycle_actions

    monkeypatch.setattr(skill_lifecycle_actions, "_reconcile", lambda *args, **kwargs: {"action": "extension_loaded"})
    return data, installs, reconciles


def test_the_boot_pass_readies_comtypes_and_turns_the_requested_install_on(windows_installed):
    data, installs, reconciles = windows_installed
    skill = find_skill(data, WINDOWS)
    assert skill_deps_not_ready(data, skill)[1] == "status" and not skill.enabled
    bundled = sorted(path.name for path in SEEDS.iterdir() if path.is_dir())

    assert _dependency_pass(data, bundled, log) == [WINDOWS]

    assert installs == [["comtypes==1.4.17"]] and reconciles == [WINDOWS]
    assert skill_deps_not_ready(data, find_skill(data, WINDOWS)) == ("", "")
    assert load_enabled(data, WINDOWS) is True
    assert not (data / "state" / "skills" / WINDOWS / "review_job.json").exists()
    # Ready dependencies are not installed twice.
    assert _dependency_pass(data, [WINDOWS], log) == [] and len(installs) == 1


def test_an_owner_off_waits_and_its_enable_installs_the_dependency(windows_installed):
    from ouroboros import skill_lifecycle_actions

    data, installs, _reconciles = windows_installed
    save_enabled(data, WINDOWS, False, actor="owner_ui")
    assert _dependency_pass(data, [WINDOWS], log) == [] and installs == []

    result = skill_lifecycle_actions._toggle(find_skill(data, WINDOWS), data, "", True, "owner_ui", "test")

    assert result["ok"] is True and result["enabled"] is True
    assert installs == [["comtypes==1.4.17"]] and load_enabled(data, WINDOWS) is True


def test_a_failed_install_names_the_failure_and_the_next_boot_retries(windows_installed, monkeypatch):
    data, installs, reconciles = windows_installed

    def offline(*args, **kwargs):
        raise RuntimeError("pip install failed: no network")

    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages", offline)
    assert _dependency_pass(data, [WINDOWS], log) == [WINDOWS]
    assert load_enabled(data, WINDOWS) is False  # turning on waits for its dependencies
    assert "no network" in json.loads((data / "state" / "skills" / WINDOWS / "deps.json").read_text(encoding="utf-8"))["error"]
    assert reconciles == [WINDOWS]

    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, env_root, *args, **kwargs: installs.append(list(packages)) or current_python_env(env_root) or [])
    assert _dependency_pass(data, [WINDOWS], log) == [WINDOWS]
    assert installs == [["comtypes==1.4.17"]] and load_enabled(data, WINDOWS) is True


def test_the_trust_opt_out_installs_without_acceptance_or_dependencies(data, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "false")
    on(monkeypatch, "win32")
    installs = []
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, *args, **kwargs: installs.append(packages) or [])

    assert install_builtin_skill(WINDOWS, drive_root=data, seed_dir=SEEDS, turn_on_when_ready=True)["ok"]

    skill = find_skill(data, WINDOWS)
    assert not skill.review.gate_for(skill.content_hash)["executable_review"]
    assert load_enabled(data, WINDOWS) is False
    assert _dependency_pass(data, [WINDOWS], log) == [] and installs == []


def test_bytes_a_review_admits_get_their_packages_at_enable_like_accepted_ones(windows_installed):
    """Inverted pin: review-admitted bytes used to be sent to a paid review for packages while accepted
    bytes got them at Enable; whatever admits the bytes admits their packages (one owner, one next action)."""
    from ouroboros import skill_lifecycle_actions
    from ouroboros.skill_install_record import write_install_record, read_install_record
    from ouroboros.skill_readiness import _DEPENDENCY_ACTION

    data, installs, _reconciles = windows_installed
    accepted = find_skill(data, WINDOWS)
    # The same bytes admitted by a review row instead of an acceptance fact.
    write_install_record(data, WINDOWS, {**read_install_record(data, WINDOWS), "basis": "review"})
    save_review_state(data, WINDOWS, SkillReviewState(status="clean", content_hash=accepted.content_hash,
                                                      findings=[{"item": "x", "verdict": "PASS"}]))
    reviewed = find_skill(data, WINDOWS)
    assert reviewed.review.gate_for(reviewed.content_hash)["blocking_reason"] == "ready"
    assert _DEPENDENCY_ACTION["tool"] == "toggle_skill" and "skill_review" not in _DEPENDENCY_ACTION["reason"]
    result = skill_lifecycle_actions._toggle(reviewed, data, "", True, "owner_ui", "test")
    assert result["ok"] is True and installs == [["comtypes==1.4.17"]] and load_enabled(data, WINDOWS) is True


# ── the startup receipt: an unchanged committed tree costs no payload work ──

SKILL_TEMPLATE = "---\nname: NAME\ndescription: Receipt fixture.\nversion: VERSION\ntype: instruction\n---\nBody.\n"


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, timeout=30)


def _write(skills, name, version):
    (skills / name).mkdir(parents=True, exist_ok=True)
    (skills / name / "SKILL.md").write_text(SKILL_TEMPLATE.replace("NAME", name).replace("VERSION", version),
                                            encoding="utf-8")


@pytest.fixture
def seed_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    repo, drive = tmp_path / "repo", tmp_path / "data"
    (drive / "skills").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive / "settings.json")
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@ouroboros")
    for name in ("first", "second"):
        _write(repo / "skills", name, "1.0.0")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bundled tree")
    for name in ("first", "second"):
        assert install_builtin_skill(name, drive_root=drive, seed_dir=repo / "skills")["ok"]
    return repo, drive


def ensure(repo, drive):
    return skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log, dependency_pass=False)


def receipt(drive):
    path = drive / "state" / "native_seed.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def test_an_unchanged_committed_tree_needs_no_payload_work(seed_repo, monkeypatch):
    repo, drive = seed_repo
    assert ensure(repo, drive) == 0
    first = receipt(drive)
    assert first["schema_version"] == 4 and first["seed_tree"]

    def forbidden(*args, **kwargs):
        raise AssertionError("an unchanged bundled tree must not be reconciled again")

    monkeypatch.setattr(skill_builtin, "reseed_changed_builtins", forbidden)
    monkeypatch.setattr(skill_builtin, "release_removed_builtins", forbidden)
    assert ensure(repo, drive) == 0 and receipt(drive) == first


def test_a_new_version_reseeds_and_names_the_new_tree(seed_repo):
    repo, drive = seed_repo
    ensure(repo, drive)
    before = receipt(drive)
    _write(repo / "skills", "first", "1.1.0")
    _git(repo, "commit", "-qam", "first 1.1.0")

    assert ensure(repo, drive) == 1

    assert skill_builtin._version(drive / "skills" / "native" / "first") == "1.1.0"
    assert receipt(drive)["seed_tree"] != before["seed_tree"]


def test_a_dirty_tree_is_reconciled_without_a_receipt(seed_repo):
    repo, drive = seed_repo
    _write(repo / "skills", "second", "2.0.0")  # not committed

    assert ensure(repo, drive) == 1

    assert skill_builtin._version(drive / "skills" / "native" / "second") == "2.0.0"
    assert receipt(drive) is None


@pytest.mark.parametrize("phase", ["reseed", "release"])
def test_a_failed_phase_writes_no_receipt_and_the_next_boot_retries(seed_repo, monkeypatch, phase):
    repo, drive = seed_repo
    ensure(repo, drive)
    shutil.rmtree(repo / "skills" / "second")  # merged into `first`: retired only by a pass where `first` arrived
    _write(repo / "skills", "first", "1.1.0")
    _git(repo, "commit", "-qam", "first 1.1.0 takes in second")
    if phase == "reseed":
        real = skill_builtin.install_builtin_skill
        monkeypatch.setattr(skill_builtin, "install_builtin_skill",
                            lambda *args, **kwargs: {"ok": False, "error": "disk full"})
    else:
        monkeypatch.setattr(skill_builtin, "release_removed_builtins", lambda *args, **kwargs: False)

    ensure(repo, drive)
    assert receipt(drive) is None  # the stale receipt went before the pass; none replaced it
    assert (drive / "skills" / "native" / "second" / ".seed-origin").is_file(), "the retired partner waits"

    monkeypatch.undo()
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    if phase == "reseed":
        assert skill_builtin.install_builtin_skill is real
    ensure(repo, drive)
    assert receipt(drive) is not None and not (drive / "skills" / "native" / "second").exists()
    assert skill_builtin._version(drive / "skills" / "native" / "first") == "1.1.0"


@pytest.mark.parametrize("pending", [["first"], None])
def test_pending_packages_do_not_invalidate_the_bundle_receipt(seed_repo, monkeypatch, pending):
    """Inverted contract: package retries no longer require repeating bundle reconciliation."""
    repo, drive = seed_repo
    monkeypatch.setattr(skill_builtin, "_skills_needing_work", lambda *args: pending)
    ensure(repo, drive)
    assert receipt(drive) is not None


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_an_earlier_launcher_receipt_reruns_the_pass_once(seed_repo, monkeypatch, schema):
    repo, drive = seed_repo
    tree = subprocess.run(["git", "rev-parse", "HEAD:skills"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    (drive / "state").mkdir(exist_ok=True)
    (drive / "state" / "native_seed.json").write_text(json.dumps(
        {"schema_version": schema, "repo": str(repo.resolve()), "seed_tree": tree, "post_bootstrap_seeds": []}),
        encoding="utf-8")
    calls = []
    real = skill_builtin.reseed_changed_builtins
    monkeypatch.setattr(skill_builtin, "reseed_changed_builtins", lambda *a, **k: calls.append(1) or real(*a, **k))

    ensure(repo, drive)
    ensure(repo, drive)

    assert calls == [1] and receipt(drive)["schema_version"] == 4


def test_the_pass_starts_the_dependency_step_off_the_boot_path(seed_repo, monkeypatch):
    repo, drive = seed_repo
    started = []

    class Thread:
        def __init__(self, *, target, args, name, daemon):
            started.append((name, daemon, target, args[1]))

        def start(self):
            pass

    monkeypatch.setattr(skill_builtin, "threading", SimpleNamespace(Thread=Thread))  # this module's threads only
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    assert [(name, daemon, target) for name, daemon, target, _names in started] == [
        ("skill-dependencies", True, skill_builtin._dependency_pass)]
    assert started[0][3] is None  # every installed skill, not only the bundle (owner decision D2 = A)
    started.clear()
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)  # receipt matches: no payload work...
    # ...but the package pass still starts (inverted pin: it used to stop at the matching receipt, so a failed
    # install of a built-in was never retried at the next boot once the receipt was written).
    assert [name for name, *_ in started] == ["skill-dependencies"]


# ── work a later startup pass must still see ──

class _SyncThread:
    """Runs the startup pass's background work inline so a test can observe it."""

    def __init__(self, *, target, args, name, daemon):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def _stub_lifecycle_effects(monkeypatch):
    from ouroboros import skill_lifecycle_actions

    reconciles = []
    monkeypatch.setattr("ouroboros.extension_loader.reconcile_extension",
                        lambda name, *args, **kwargs: reconciles.append(name) or {"action": "extension_loaded"})
    monkeypatch.setattr(skill_lifecycle_actions, "_reconcile", lambda *args, **kwargs: {"action": "extension_loaded"})
    return reconciles


def test_a_builtin_installed_after_the_receipt_retries_its_dependencies_at_the_next_boot(tmp_path, monkeypatch):
    """A new install writes the receipt before onboarding; the starter's later built-in install must
    not hide a failed dependency install from the next boot."""
    on(monkeypatch, "win32")
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    repo, drive = tmp_path / "repo", tmp_path / "data"
    (drive / "skills").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", drive / "settings.json")
    shutil.copytree(SEEDS / WINDOWS, repo / "skills" / WINDOWS, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@ouroboros")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bundled tree")
    monkeypatch.setattr(skill_builtin, "threading", SimpleNamespace(Thread=_SyncThread))  # this module's threads only
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    assert receipt(drive) is not None  # the first start, before onboarding
    _stub_lifecycle_effects(monkeypatch)
    installs = []

    def offline(*args, **kwargs):
        raise RuntimeError("pip install failed: no network")

    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages", offline)
    assert install_builtin_skill(WINDOWS, drive_root=drive, seed_dir=repo / "skills", turn_on_when_ready=True,
                                 installed_by="owner")["ok"]
    assert receipt(drive) is None  # the install leaves work for the next pass
    assert _dependency_pass(drive, [WINDOWS], log) == [WINDOWS] and not load_enabled(drive, WINDOWS)

    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, env_root, *args, **kwargs: installs.append(list(packages)) or current_python_env(env_root) or [])
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)  # the network is back
    assert installs == [["comtypes==1.4.17"]] and load_enabled(drive, WINDOWS) is True
    assert receipt(drive) is not None  # the seed receipt certifies the bundle, not package work
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    assert receipt(drive) is not None and len(installs) == 1


def test_the_boot_pass_finishes_a_requested_turn_on_whose_dependencies_were_ready(windows_installed):
    """The install prepared the dependencies, then the process stopped before it turned the skill on."""
    from ouroboros.skill_dependencies import ensure_skill_dependencies

    data, installs, reconciles = windows_installed
    assert ensure_skill_dependencies(data, find_skill(data, WINDOWS))[0] == "installed"
    assert skill_deps_not_ready(data, find_skill(data, WINDOWS)) == ("", "") and not load_enabled(data, WINDOWS)
    assert _dependency_pass(data, [WINDOWS], log) == [WINDOWS]

    assert load_enabled(data, WINDOWS) is True
    assert installs == [["comtypes==1.4.17"]]  # prepared once, never again


def test_a_turn_on_waiting_for_grants_stays_off(data, monkeypatch):
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    on(monkeypatch, "darwin")
    assert install_builtin_skill("telegram", drive_root=data, seed_dir=SEEDS, turn_on_when_ready=True)["ok"]
    _stub_lifecycle_effects(monkeypatch)

    assert _dependency_pass(data, ["telegram"], log) == ["telegram"]  # tried at a slow start
    assert load_enabled(data, "telegram") is False  # still waits for the owner's grants


def test_an_unreadable_installed_version_leaves_the_pass_unfinished(seed_repo):
    repo, drive = seed_repo
    manifest = drive / "skills" / "native" / "first" / "SKILL.md"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text("not a manifest\n", encoding="utf-8")
    ensure(repo, drive)
    assert receipt(drive) is None  # an unknown version is unfinished work, never a completed pass
    manifest.write_text(text, encoding="utf-8")
    ensure(repo, drive)
    assert receipt(drive) is not None


def test_the_dependency_job_carries_a_kind_the_skills_card_labels(windows_installed):
    import re

    from ouroboros.gateway.projects import _SKILL_LIFECYCLE_KINDS
    from ouroboros.skill_lifecycle_queue import queue_snapshot

    data, _installs, _reconciles = windows_installed
    assert _dependency_pass(data, [WINDOWS], log) == [WINDOWS]

    job = [event for event in queue_snapshot()["events"]
           if event["target"] == WINDOWS and event["source"] == "startup"][-1]
    card = (REPO / "web" / "modules" / "skill_card_renderer.js").read_text(encoding="utf-8")
    labels = re.search(r"const LIFECYCLE_PENDING_LABELS = \{(.*?)\};", card, re.S).group(1)
    assert job["kind"] in re.findall(r"^\s*(\w+):", labels, re.M)  # else the card says only "Working…"
    assert _SKILL_LIFECYCLE_KINDS[job["kind"]] == "Skill dependencies"


def test_os_declarations_parse_as_platform_lists():
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

    def manifest(os_line):
        return parse_skill_manifest_text(f"---\nname: x\ndescription: x\nversion: 1\ntype: instruction\n{os_line}---\n")

    assert manifest("").os == "any" and manifest("").applies_to_platform("win32")
    listed = manifest("os: [darwin, Linux]\n")
    assert listed.os == "darwin,linux"
    assert listed.applies_to_platform("linux") and not listed.applies_to_platform("win32")
    assert manifest("os: windows\n").applies_to_platform("win32")
    assert manifest("os: macos, win32\n").applies_to_platform("darwin")
    assert not manifest("os: win32\n").applies_to_platform("darwin")
    unix, windows = skill_builtin._manifest(SEEDS / UNIX), skill_builtin._manifest(SEEDS / WINDOWS)
    assert unix.os == "darwin,linux" and windows.os == "win32"
    assert windows.permissions == ["tool"]
    assert windows.raw_extra["install"] == [{"kind": "pip", "package": "comtypes==1.4.17"}]


def test_unknown_dependency_readiness_stays_unknown_beside_source_receipt(windows_installed, monkeypatch):
    data, installs, _reconciles = windows_installed
    assert _dependency_pass(data, [WINDOWS], log) == [WINDOWS] and len(installs) == 1  # ready and on
    identity = {"schema_version": skill_builtin.RECEIPT_SCHEMA_VERSION, "seed_tree": "fixed"}
    monkeypatch.setattr(skill_builtin, "_seed_identity", lambda repo_dir: identity)
    monkeypatch.setattr(skill_builtin, "reseed_changed_builtins", lambda *args, **kwargs: 0)
    monkeypatch.setattr(skill_builtin, "release_removed_builtins", lambda *args, **kwargs: True)

    def boot():
        skill_builtin.ensure_builtin_skills(data, repo_dir=REPO, log_obj=log, dependency_pass=False)
        return receipt(data)

    import ouroboros.skill_dependencies as deps

    def unreadable(*args, **kwargs):
        raise PermissionError("deps state unreadable")

    with monkeypatch.context() as broken:
        broken.setattr(deps, "auto_install_specs_for_skill", unreadable)
        assert deps.skill_deps_not_ready(data, find_skill(data, WINDOWS)) == ("", "")  # execution readers
        assert skill_builtin._skills_needing_work(data, [WINDOWS], log) is None
        assert boot() == identity  # source completion never certifies package readiness
        assert len(installs) == 1
    assert boot() == identity


# ── computer use is on by default (owner decision 16.10; question 4 = A on 18.10) ──

def _starter(drive, state, **fields):
    from ouroboros.skill_starter import _write_marker

    _write_marker(drive, state=state, **fields)


def _boot(drive, monkeypatch):
    """One startup's built-in work, its dependency pass run inline."""
    monkeypatch.setattr(skill_builtin, "threading", SimpleNamespace(Thread=_SyncThread))
    (drive / "state" / "native_seed.json").unlink(missing_ok=True)
    return skill_builtin.ensure_builtin_skills(drive, repo_dir=REPO, log_obj=log)


@pytest.fixture
def lifecycle(monkeypatch):
    installs = []
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, env_root, *args, **kwargs: installs.append(list(packages)) or current_python_env(env_root) or [])
    _stub_lifecycle_effects(monkeypatch)
    return installs


@pytest.mark.parametrize("platform,own,other", [("darwin", UNIX, WINDOWS), ("linux", UNIX, WINDOWS),
                                                ("win32", WINDOWS, UNIX)])
def test_a_new_install_starts_with_its_own_computer_use_skill_ticked_and_on(data, monkeypatch, lifecycle,
                                                                             platform, own, other):
    from ouroboros import skill_starter

    on(monkeypatch, platform)
    monkeypatch.setattr("ouroboros.skill_review.review_skill", lambda *a, **k: pytest.fail("trusted starter review"))
    rows = [{"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True}]
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(rows))
    assert skill_builtin.default_on_names(SEEDS) == [own, *EVERYWHERE]
    ticked = skill_starter.default_starter_ids(data, rows, repo_dir=REPO)
    assert f"builtin/{own}" in ticked and f"builtin/{other}" not in ticked
    assert all(f"builtin/{name}" in ticked for name in EVERYWHERE)

    marker = skill_starter.run_starter(data, None, origin="first_start", repo_dir=REPO)

    assert marker["results"][f"builtin/{own}"]["status"] == "installed"
    assert marker["defaults_offered"] == sorted([own, *EVERYWHERE])
    for name in EVERYWHERE:
        assert marker["results"][f"builtin/{name}"]["status"] == "installed"
        assert load_enabled(data, name) is True
        skill = find_skill(data, name)
        assert skill.review.gate_for(skill.content_hash)["blocking_reason"] == "trusted_source"
        assert not (data / "state" / "skills" / name / "review.json").exists()
    _dependency_pass(data, [own], log)  # dependencies an offline first start missed
    assert load_enabled(data, own) is True and not (data / "skills" / "native" / other).exists()
    assert lifecycle == ([["comtypes==1.4.17"]] if own == WINDOWS else [])


def test_an_unticked_default_stays_off_the_install(data, monkeypatch, lifecycle):
    from ouroboros import skill_starter

    on(monkeypatch, "win32")
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(
        [{"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True}]))
    skill_starter.run_starter(data, ["builtin/telegram"], origin="onboarding", repo_dir=REPO)
    assert _boot(data, monkeypatch) == 0
    assert not (data / "skills" / "native" / WINDOWS).exists()


def test_an_existing_windows_install_gets_the_windows_skill_once_on_update(data, monkeypatch, lifecycle):
    on(monkeypatch, "win32")
    _starter(data, "upgrade")

    _boot(data, monkeypatch)

    assert load_enabled(data, WINDOWS) is True and lifecycle == [["comtypes==1.4.17"]]
    skill = find_skill(data, WINDOWS)
    assert skill.review.gate_for(skill.content_hash)["blocking_reason"] == "trusted_source"
    from ouroboros.skill_starter import read_marker

    assert read_marker(data)["defaults_offered"] == sorted([WINDOWS, *EVERYWHERE])
    shutil.rmtree(data / "skills" / "native" / WINDOWS)  # the owner removes it
    _boot(data, monkeypatch)
    assert not (data / "skills" / "native" / WINDOWS).exists()


@pytest.mark.parametrize("choice", [None, {"enabled": False, "actor": "owner_ui"}])
def test_an_existing_unix_copy_is_turned_on_once_unless_the_owner_chose(data, monkeypatch, lifecycle, choice):
    on(monkeypatch, "darwin")
    native = earlier_install(data)
    state = data / "state" / "skills" / UNIX
    if choice is not None:
        (state / "enabled.json").write_text(json.dumps(choice), encoding="utf-8")
    connections = (state / "connections.json").read_bytes()
    _starter(data, "upgrade")

    _boot(data, monkeypatch)

    assert skill_builtin._version(native / UNIX) == skill_builtin._version(SEEDS / UNIX)
    assert load_enabled(data, UNIX) is (choice is None)
    assert (state / "connections.json").read_bytes() == connections
    if choice is None:
        save_enabled(data, UNIX, False, actor="owner_ui")  # a later Off stands at every start after it
        _boot(data, monkeypatch)
        assert load_enabled(data, UNIX) is False


def test_a_default_an_earlier_release_offered_and_the_owner_removed_stays_removed(data, monkeypatch, lifecycle):
    on(monkeypatch, "win32")
    native = data / "skills" / "native"
    native.mkdir(parents=True)
    (native / f".post-bootstrap-seed-{WINDOWS}").write_text("offered\n", encoding="utf-8")
    _starter(data, "upgrade")
    _boot(data, monkeypatch)
    assert not (native / WINDOWS).exists() and lifecycle == []


def test_a_new_install_offers_nothing_before_its_starter_runs(data, monkeypatch, lifecycle):
    on(monkeypatch, "win32")
    _starter(data, "fresh")
    _boot(data, monkeypatch)
    assert not (data / "skills" / "native" / WINDOWS).exists()


def test_a_name_held_by_another_skill_keeps_the_default_owed(data, monkeypatch, lifecycle):
    on(monkeypatch, "win32")
    held = data / "skills" / "external" / WINDOWS
    held.mkdir(parents=True)
    (held / "SKILL.md").write_text(f"---\nname: {WINDOWS}\ndescription: mine\nversion: 1.0.0\ntype: instruction\n---\nMine.\n",
                                   encoding="utf-8")
    _starter(data, "upgrade")
    _boot(data, monkeypatch)
    assert not (data / "skills" / "native" / WINDOWS).exists() and receipt(data) is None
    from ouroboros.skill_starter import read_marker

    assert WINDOWS not in (read_marker(data).get("defaults_offered") or [])


def test_the_trust_opt_out_offers_without_acceptance_or_dependencies(data, monkeypatch, lifecycle):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "false")
    on(monkeypatch, "win32")
    _starter(data, "upgrade")
    _boot(data, monkeypatch)
    skill = find_skill(data, WINDOWS)
    assert skill is not None and not skill.review.gate_for(skill.content_hash)["executable_review"]
    assert load_enabled(data, WINDOWS) is False and lifecycle == []


@pytest.mark.parametrize("platform,own", [("darwin", UNIX), ("win32", WINDOWS)])
def test_unticking_everything_at_onboarding_stands_after_an_update(data, monkeypatch, lifecycle, platform, own):
    from ouroboros import skill_starter

    on(monkeypatch, platform)
    skill_starter.classify_first_start(data, {})
    assert skill_starter.after_onboarding(data, [], repo_dir=REPO) == "done"
    assert skill_starter.read_marker(data)["defaults_offered"] == sorted([own, *EVERYWHERE])
    _boot(data, monkeypatch)  # a later app update re-runs the startup pass
    assert not (data / "skills" / "native" / own).exists() and lifecycle == []
    assert not any((data / "skills" / "native" / name).exists() for name in EVERYWHERE), "unticked stays off"


# Real upgrade receipts: no helper unlinks them between boots.
@pytest.fixture
def instruction_upgrade(seed_repo, monkeypatch):
    from ouroboros import skill_review

    repo, drive = seed_repo
    _starter(drive, "upgrade")
    ensure(repo, drive)
    assert receipt(drive) is not None
    _write(repo / "skills", "new_instruction", "1.0.0")
    (repo / "skills" / "builtin.json").write_text(json.dumps({"skills": [{
        "name": "new_instruction", "category": ["design"], "recommended": True, "default_on": True,
    }]}), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add default instruction")
    monkeypatch.setattr(skill_builtin, "threading", SimpleNamespace(Thread=_SyncThread))

    def forbidden(*args, **kwargs):
        pytest.fail("an instruction turn-on must not review or load an extension")

    monkeypatch.setattr(skill_review, "review_skill", forbidden)
    monkeypatch.setattr("ouroboros.extension_loader.reconcile_extension", forbidden)
    return repo, drive


def test_upgrade_turns_on_a_new_default_instruction_without_review_or_extension_load(instruction_upgrade):
    repo, drive = instruction_upgrade
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    skill = find_skill(drive, "new_instruction")
    assert skill.enabled and skill.review.gate_for(skill.content_hash)["blocking_reason"] == "trusted_source"
    assert not (drive / "state/skills/new_instruction/review.json").exists()
    assert receipt(drive) is not None  # package/turn-on work is independent of this source receipt
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    assert receipt(drive)["schema_version"] == 4


def test_fresh_instruction_starter_needs_no_review_or_extension_load(instruction_upgrade, monkeypatch):
    from ouroboros import skill_starter
    from ouroboros.skill_source_catalog import builtin_snapshot

    repo, _drive = instruction_upgrade
    fresh = repo.parent / "fresh_data"
    fresh.mkdir()
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", fresh / "settings.json")
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([
        {"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True},
    ]))
    marker = skill_starter.run_starter(fresh, None, origin="first_start", repo_dir=repo)
    assert marker["state"] == "done" and marker["defaults_offered"] == ["new_instruction"]
    assert load_enabled(fresh, "new_instruction") is True
    assert not (fresh / "state/skills/new_instruction/review.json").exists()
    catalog = {row["name"]: row for row in builtin_snapshot(repo)["entries"]}
    assert catalog["new_instruction"]["category"] == ["design"]
    assert catalog["new_instruction"]["recommended"] is True


@pytest.mark.parametrize("choice", ["off", "removed"])
def test_instruction_owner_choice_survives_later_boots_and_release(instruction_upgrade, choice):
    repo, drive = instruction_upgrade
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    if choice == "off":
        save_enabled(drive, "new_instruction", False, actor="owner_ui")
    else:
        shutil.rmtree(drive / "skills/native/new_instruction")
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    assert receipt(drive) is not None
    _write(repo / "skills", "new_instruction", "1.1.0")
    _git(repo, "commit", "-qam", "update instruction")
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    skill = find_skill(drive, "new_instruction")
    assert (skill is None) if choice == "removed" else (not skill.enabled)
    assert receipt(drive) is not None


def test_instruction_upgrade_respects_review_trust_opt_out(instruction_upgrade, monkeypatch):
    repo, drive = instruction_upgrade
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "false")
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    skill = find_skill(drive, "new_instruction")
    assert not skill.enabled and not skill.review.gate_for(skill.content_hash)["executable_review"]
    assert receipt(drive) is not None  # the review choice is not unfinished startup work


@pytest.mark.parametrize("interruption", ["disabled_pass", "thread_never_ran", "old_receipt"])
def test_instruction_turn_on_is_retried_after_interrupted_startup(instruction_upgrade, monkeypatch, interruption):
    repo, drive = instruction_upgrade
    with monkeypatch.context() as interrupted:
        if interruption == "thread_never_ran":
            interrupted.setattr(skill_builtin, "threading", SimpleNamespace(
                Thread=lambda **kwargs: SimpleNamespace(start=lambda: None)))
        skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log,
                                            dependency_pass=interruption == "thread_never_ran")
    assert not load_enabled(drive, "new_instruction") and receipt(drive) is not None
    if interruption == "old_receipt":
        identity = {**skill_builtin._seed_identity(repo), "schema_version": 2}
        (drive / "state/native_seed.json").write_text(json.dumps(identity), encoding="utf-8")
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    assert load_enabled(drive, "new_instruction") is True
    skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, log_obj=log)
    assert receipt(drive)["schema_version"] == 4
