"""A deployment's required extensions at their real owners, each case beside personal OSS (no policy file).

The file names a package by canonical name, directory and the digest the loader computes for its bytes. Exactly
that package reads as on over the employee's saved Off, which stays saved; nothing else is granted: review, grants,
conflicts and dependencies still decide. Supported doors that would turn it off, remove, replace, shadow, update,
repair or edit it refuse before they unload or write; ordinary unload still unloads it. Its fixed settings reach an
extension only through that extension's grant, in process and in an out-of-process child of the same generation, and
never through a projection, an error or a companion's manifest env. Under ``hold`` the carrier does not start work
while the package is not live in that very process; under ``warn`` it starts; both record the change once.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import extension_loader
from tests._extension_loader_shared import _clear_loader_state, _write_ext_skill  # noqa: F401
from tests.test_deployment_policy import _settings_app, manage, write_document
from tests.test_deployment_policy import install as install_fixture
from tests.test_marketplace_dependency_order import registry as clawhub  # noqa: F401  (controlled transport)

install = install_fixture  # the shared isolated data root, as a fixture of this module

NAME = "monitor"
SECRET = "https://fixture-key@errors.invalid/7"
PLUGIN = (
    "API = None\n"
    "def _read(ctx):\n"
    "    return repr(sorted(API.get_settings(['ORG_DSN']).items()))\n"
    "def register(api):\n"
    "    global API\n"
    "    API = api\n"
    "    api.register_tool('read', _read, description='read', schema={})\n"
)


@pytest.fixture
def org(install, monkeypatch, tmp_path):
    """One extension in ``skills/external`` of an isolated data root, reviewed, its key not yet granted."""
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_review_state

    empty = tmp_path / "no-local-skills"
    empty.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(empty))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    skill_dir = _write_ext_skill(install.data / "skills" / "external", NAME, plugin_body=PLUGIN,
                                 permissions=["tool", "read_settings"], env_from_settings=["ORG_DSN"])
    loaded = load_skill(skill_dir, install.data)
    save_review_state(install.data, NAME, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    install.skill_dir, install.content_hash = loaded.skill_dir, loaded.content_hash
    return install


def require(org, monkeypatch, *, mode="hold", path=None, content_hash=None, settings=None, name=NAME):
    return manage(org, monkeypatch, {"version": 1, "extensions": {"when_unmet": mode, "required": [{
        "name": name, "path": str(path or org.skill_dir), "content_hash": content_hash or org.content_hash,
        **({"settings": settings} if settings is not None else {})}]}})


def grant(org, keys=("ORG_DSN",)):
    from ouroboros.skill_loader import save_skill_grants

    save_skill_grants(org.data, NAME, list(keys), content_hash=org.content_hash, requested_keys=list(keys),
                      granted_permissions=[], requested_permissions=[])


def reconcile(org):
    from ouroboros.config import load_settings

    return extension_loader.reconcile_extension(NAME, org.data, load_settings)


def enabled_file(org) -> dict:
    return json.loads((org.data / "state" / "skills" / NAME / "enabled.json").read_text(encoding="utf-8"))


def events(org, kind: str) -> list:
    path = org.data / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


# --- the file -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("section,reason", [
    ({"required": [{"name": "a", "path": "/opt/a", "content_hash": "0" * 64}]}, "must say whether ordinary work"),
    ({"when_unmet": "later", "required": []}, "extensions.when_unmet must be one of"),
    ({"when_unmet": "hold", "required": [{"name": "My Skill", "path": "/opt/a", "content_hash": "0" * 64}]},
     "canonical skill name"),
    ({"when_unmet": "hold", "required": [{"name": "a", "path": "skills/a", "content_hash": "0" * 64}]},
     "absolute package directory"),
    ({"when_unmet": "hold", "required": [{"name": "a", "path": "/opt/a", "content_hash": "abc"}]}, "64-hex digest"),
    ({"when_unmet": "hold", "required": [{"name": "a", "path": "/opt/a", "content_hash": "0" * 64,
                                          "settings": {"OPENROUTER_API_KEY": "x"}}]}, "only custom settings"),
    ({"when_unmet": "hold", "required": [{"name": "a", "path": "/opt/a", "content_hash": "0" * 64,
                                          "settings": {"OUROBOROS_X": "x"}}]}, "only custom settings"),
    ({"when_unmet": "hold", "required": [
        {"name": "a", "path": "/opt/a", "content_hash": "0" * 64, "settings": {"ORG_DSN": "x"}},
        {"name": "b", "path": "/opt/b", "content_hash": "1" * 64, "settings": {"ORG_DSN": "y"}}]}, "fixed twice"),
    ({"when_unmet": "hold", "required": [{"name": "a", "path": "/opt/a", "content_hash": "0" * 64, "extra": 1}]},
     "unsupported field 'extra'"),
])
def test_the_extensions_section_is_strict_and_names_its_mode(section, reason):
    from ouroboros.deployment_policy import DeploymentPolicyError, parse_policy

    with pytest.raises(DeploymentPolicyError, match=reason):
        parse_policy(json.dumps({"version": 1, "extensions": section}).encode())
    policy = parse_policy(json.dumps({"version": 1, "extensions": {"when_unmet": "warn", "required": [
        {"name": "a", "path": "/opt/a", "content_hash": "0" * 64, "settings": {"ORG_DSN": SECRET}}]}}).encode())
    assert (policy.when_unmet, policy.extensions[0].name) == ("warn", "a")
    assert SECRET not in repr(policy), "a fixed value never prints"
    assert parse_policy(b'{"version": 1, "extensions": {"required": []}}').when_unmet == ""


# --- readers --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_the_required_package_reads_on_over_the_saved_off_which_stays_saved(org, monkeypatch, managed):
    from ouroboros.extension_liveness import runtime_state_for_skill_name
    from ouroboros.skill_loader import find_skill, load_enabled, save_enabled
    from ouroboros.skill_peer_inventory import discover_skill_peers

    save_enabled(org.data, NAME, False, actor="owner_ui")
    saved = enabled_file(org)
    if managed:
        require(org, monkeypatch, mode="warn")
    state = reconcile(org)
    assert find_skill(org.data, NAME).enabled is managed
    assert [peer.enabled for peer in discover_skill_peers(org.data) if peer.name == NAME] == [managed]
    assert runtime_state_for_skill_name(NAME, org.data)["desired_live"] is managed
    assert (NAME in extension_loader.snapshot()["extensions"]) is managed, state
    assert load_enabled(org.data, NAME) is False and enabled_file(org) == saved, "the employee's Off stays saved"


@pytest.mark.parametrize("change", ["bytes", "path", "collision", "missing", "review"])
def test_only_the_pinned_package_is_the_required_one_and_nothing_is_granted(org, monkeypatch, change):
    from ouroboros.deployment_extensions import state_rows
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state

    require(org, monkeypatch)
    if change == "bytes":
        (org.skill_dir / "plugin.py").write_text(PLUGIN + "# edited\n", encoding="utf-8")
    elif change == "path":
        moved = org.data / "skills" / "clawhub" / NAME
        moved.parent.mkdir(parents=True)
        shutil.move(str(org.skill_dir), str(moved))
    elif change == "collision":
        _write_ext_skill(org.data / "skills" / "clawhub", NAME, plugin_body=PLUGIN, permissions=["tool"])
    elif change == "missing":
        shutil.rmtree(org.skill_dir)
    elif change == "review":
        save_review_state(org.data, NAME, SkillReviewState(status="pending", content_hash=org.content_hash))
    reconcile(org)
    [row] = state_rows(org.data)
    reason = {"bytes": "changed", "path": "elsewhere", "collision": "collision", "missing": "missing",
              "review": "review"}[change]
    assert (row["state"], row["reason"]) == ("unmet", reason), row
    assert NAME not in extension_loader.snapshot()["extensions"]
    skill = find_skill(org.data, NAME)
    assert skill is None or skill.enabled is (change == "review"), "only the pinned bytes read as on"
    assert row["process"]["pid"] and "role" in row["process"], "a state names the process that observed it"


def test_a_fixed_setting_needs_the_extensions_own_grant_and_never_leaves_as_a_value(org, monkeypatch):
    from ouroboros.deployment_extensions import state_rows
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    write_document(org, {"ORG_DSN": "https://employee-own@mine.invalid/1"})
    require(org, monkeypatch, settings={"ORG_DSN": SECRET})
    reconcile(org)
    [row] = state_rows(org.data)
    assert (row["reason"], row["settings"]) == ("grants", ["ORG_DSN"]) and SECRET not in json.dumps(row)
    grant(org)
    reconcile(org)
    assert state_rows(org.data)[0]["state"] == "met"
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=org.data)
    registry.set_context(ToolContext(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=org.data,
                                     task_metadata={"work_kind": "owner"}))
    tool = extension_loader.extension_surface_name(NAME, "read")
    assert SECRET in registry.execute(tool, {}), "the organization's value, read through the grant"
    assert json.loads(org.path.read_text())["ORG_DSN"] == "https://employee-own@mine.invalid/1", "raw stays"


def test_settings_show_neither_a_prefix_nor_the_value_and_keep_the_raw_choice(org, monkeypatch):
    from ouroboros.gateway import settings_secrets

    write_document(org, {"ORG_DSN": "https://employee-own@mine.invalid/1", "TOTAL_BUDGET": 50})
    require(org, monkeypatch, settings={"ORG_DSN": SECRET})
    app = _settings_app(org, monkeypatch)
    app.router.routes.insert(0, Route("/api/settings/secret", settings_secrets.api_settings_secret, methods=["POST"]))
    with TestClient(app) as client:
        shown = client.get("/api/settings").json()
        revealed = client.post("/api/settings/secret", json={"key": "ORG_DSN"})
        form = {key: value for key, value in shown.items() if key != "_meta"}
        saved_as_shown = client.post("/api/settings", json={**form, "TOTAL_BUDGET": 25})
        changed = client.post("/api/settings", json={"ORG_DSN": "https://elsewhere.invalid/2"})
    deployment = shown["_meta"]["policy_state"]["deployment"]
    assert shown["ORG_DSN"] == "***set***" and deployment["extensions"]["required"][0]["settings"] == ["ORG_DSN"]
    assert (revealed.status_code, revealed.json()["code"]) == (403, "managed_by_deployment")
    assert saved_as_shown.status_code == 200 and changed.status_code == 403, changed.text
    document = json.loads(org.path.read_text())
    assert (document["ORG_DSN"], document["TOTAL_BUDGET"]) == ("https://employee-own@mine.invalid/1", 25)
    for text in (json.dumps(shown), revealed.text, changed.text):
        assert SECRET not in text and "fixture-key" not in text


def test_an_out_of_process_child_reads_the_same_generation(org, monkeypatch):
    """A child that rereads settings gets the policy path and pin after its scrub: the fixed value through the
    grant; without them it would read the raw document (the gap this closes)."""
    from ouroboros.deployment_policy import GENERATION_ENV, POLICY_ENV
    from ouroboros.extension_process_runner import _child_env
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from tests._extension_loader_shared import _mark_isolated_deps_installed

    (org.skill_dir / "SKILL.md").write_text((org.skill_dir / "SKILL.md").read_text().replace(
        "type: extension\n", "type: extension\ndependencies:\n  - dummy_pkg\n"), encoding="utf-8")
    loaded = find_skill(org.data, NAME)
    save_review_state(org.data, NAME, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    org.content_hash = loaded.content_hash
    _mark_isolated_deps_installed(org.data, loaded)
    policy = require(org, monkeypatch, settings={"ORG_DSN": SECRET})
    grant(org)
    reconcile(org)
    tool = extension_loader.get_tool(extension_loader.extension_surface_name(NAME, "read"))
    assert tool is not None and tool["out_of_process"] is True
    env = _child_env(drive_root=org.data, repo_dir=pathlib.Path(__file__).resolve().parents[1], skill_name=NAME,
                     skill_dir=loaded.skill_dir, env_allowlist=[], granted_keys=[])
    assert (env[POLICY_ENV], len(env[GENERATION_ENV])) == (str(policy), 64) and "ORG_DSN" not in env
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=org.data)
    registry.set_context(ToolContext(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=org.data,
                                     task_metadata={"work_kind": "owner"}))
    assert SECRET in registry.execute(extension_loader.extension_surface_name(NAME, "read"), {})


@pytest.mark.parametrize("managed", [True, False])
def test_a_companion_manifest_env_never_stands_in_for_a_fixed_value(org, monkeypatch, managed):
    from ouroboros.deployment_policy import POLICY_ENV
    from ouroboros.extension_companion import companion_spawn_env

    if managed:
        require(org, monkeypatch, settings={"ORG_DSN": SECRET})
    spec = {"env": {"ORG_DSN": "https://manifest.invalid/9", POLICY_ENV: "/tmp/elsewhere.json", "MODE": "x"}}
    env = companion_spawn_env(spec, "token", env_allow=["ORG_DSN"], granted_upper={"ORG_DSN"}, skill=NAME,
                              skill_dir=org.skill_dir, state_dir=org.data / "state" / "skills" / NAME)
    assert env["MODE"] == "x"
    if managed:
        assert env.get("ORG_DSN") == SECRET and POLICY_ENV not in env, "the organization's value, granted"
    else:
        assert env["ORG_DSN"] == "https://manifest.invalid/9" and env[POLICY_ENV] == "/tmp/elsewhere.json"


# --- writers: refused before any effect --------------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_off_and_delete_refuse_before_unload_or_write_and_ordinary_unload_still_unloads(org, monkeypatch, managed):
    from ouroboros.skill_lifecycle_actions import run_skill_action
    from ouroboros.skill_loader import save_enabled
    from ouroboros.tools.registry import ToolContext

    save_enabled(org.data, NAME, True, actor="owner_ui")
    if managed:
        require(org, monkeypatch)
    reconcile(org)
    assert NAME in extension_loader.snapshot()["extensions"]
    before = enabled_file(org)
    ctx = ToolContext(repo_dir=org.data, drive_root=org.data)
    off = run_skill_action(ctx, NAME, "disable", _owner_actor="owner_ui")
    live_after_off, saved_after_off = NAME in extension_loader.snapshot()["extensions"], enabled_file(org)
    deleted = run_skill_action(ctx, NAME, "delete", _owner_actor="owner_ui", payload_root=f"skills/external/{NAME}")
    if managed:
        assert (off["ok"], off["status_code"], off["code"]) == (False, 403, "managed_by_deployment"), off
        assert (deleted["ok"], deleted["code"]) == (False, "managed_by_deployment"), deleted
        assert live_after_off and saved_after_off == before == enabled_file(org)
        assert (org.skill_dir / "plugin.py").read_text() == PLUGIN
        extension_loader.unload_extension(NAME)  # shutdown, Stop and Panic unload as before
        assert NAME not in extension_loader.snapshot()["extensions"]
    else:
        assert off["ok"] and not live_after_off and saved_after_off["enabled"] is False
        assert deleted["ok"] and not org.skill_dir.exists()


def test_a_collision_refuses_the_off_before_its_unload(org, monkeypatch):
    """The toggle unloaded on a collision before refusing to persist; a required name now refuses first."""
    from ouroboros.skill_lifecycle_actions import _toggle
    from ouroboros.skill_loader import find_skill

    require(org, monkeypatch)
    reconcile(org)
    assert NAME in extension_loader.snapshot()["extensions"]
    _write_ext_skill(org.data / "skills" / "clawhub", NAME, plugin_body=PLUGIN, permissions=["tool"])
    colliding = find_skill(org.data, NAME)
    assert colliding.identity_collision
    refused = _toggle(colliding, org.data, "", False, "owner_ui", "test")
    assert refused["code"] == "managed_by_deployment" and NAME in extension_loader.snapshot()["extensions"]


@pytest.mark.parametrize("managed", [True, False])
def test_source_and_marketplace_doors_refuse_before_download_or_landing(org, monkeypatch, managed):
    from ouroboros.deployment_extensions import install_refusal
    from ouroboros.marketplace import install as marketplace
    from ouroboros.skill_source_install import SkillInstallError, uninstall_installed, update_installed

    if managed:
        require(org, monkeypatch)
    monkeypatch.setattr(marketplace, "read_provenance", lambda *_a, **_k: pytest.fail("no update read"))
    update = marketplace.update_skill(org.data, org.data, sanitized_name=NAME)
    removed = marketplace.uninstall_skill(org.data, sanitized_name=NAME)
    with pytest.raises(SkillInstallError) as source_update:
        update_installed(org.data, NAME, mode="replace")
    with pytest.raises(SkillInstallError) as source_removal:
        uninstall_installed(org.data, NAME)
    if managed:
        assert (update.ok, update.error_code, update.error_status) == (False, "managed_by_deployment", 403)
        assert removed.ok is False and "required by this deployment" in removed.error
        assert {source_update.value.code, source_removal.value.code} == {"managed_by_deployment"}
        assert install_refusal(org.data, NAME) and not install_refusal(org.data, "other")
        shutil.rmtree(org.skill_dir)
        assert install_refusal(org.data, NAME) == "", "a missing required name may still be installed"
    else:
        assert "required by this deployment" not in update.error + removed.error
        assert source_update.value.code != "managed_by_deployment" and install_refusal(org.data, NAME) == ""
    assert not managed or (org.data / "skills" / "external").is_dir()


@pytest.mark.parametrize("managed", [True, False])
def test_a_clawhub_reinstall_over_the_required_package_stops_before_landing(install, clawhub, monkeypatch, managed):  # noqa: F811
    """ClawHub's install door end to end: over the installed package the file requires, a reinstall ends with the
    typed refusal before the landing and the landed bytes stay; without the file the same reinstall lands."""
    from ouroboros.marketplace import install as marketplace
    from ouroboros.skill_loader import find_skill

    first = marketplace.install_skill(clawhub.drive, clawhub.repo, slug="demo")
    assert first.ok, first.error
    loaded = find_skill(clawhub.drive, first.sanitized_name, repo_path="")
    landed = {path: path.read_bytes() for path in loaded.skill_dir.rglob("*") if path.is_file()}
    if managed:
        manage(install, monkeypatch, {"version": 1, "extensions": {"when_unmet": "hold", "required": [{
            "name": loaded.name, "path": str(loaded.skill_dir), "content_hash": loaded.content_hash}]}})
    again = marketplace.install_skill(clawhub.drive, clawhub.repo, slug="demo", overwrite=True)
    if managed:
        assert (again.ok, again.error_code, again.error_status) == (False, "managed_by_deployment", 403), again.error
        assert "required by this deployment" in again.error
        assert {path: path.read_bytes() for path in loaded.skill_dir.rglob("*") if path.is_file()} == landed
    else:
        assert again.ok and again.sanitized_name == loaded.name, again.error


@pytest.mark.parametrize("managed", [True, False])
def test_a_kept_update_of_the_package_the_file_then_pins_is_not_rolled_back(install, clawhub, monkeypatch, managed):  # noqa: F811
    """An update landed with its review still to come; then the file pinned the landed package. The later settlement
    refuses to put the old bytes back before any unload or write and accepts nothing: the update stays pending, its
    kept base stays, and its review still decides through liveness. Without the file the same settlement restores."""
    from ouroboros.marketplace import install as marketplace
    from ouroboros.skill_install_record import read_install_record, settle_pending_update, update_backup_path
    from ouroboros.skill_loader import find_skill

    first = marketplace.install_skill(clawhub.drive, clawhub.repo, slug="demo")
    assert first.ok, first.error
    before = (first.target_dir / "SKILL.md").read_bytes()
    landed = marketplace.update_skill(clawhub.drive, clawhub.repo, sanitized_name="demo", version="2.0.0",
                                      auto_review=False)
    assert landed.ok, landed.error
    update = find_skill(clawhub.drive, "demo", repo_path="")
    after = (update.skill_dir / "SKILL.md").read_bytes()
    assert after != before and read_install_record(clawhub.drive, "demo")["pending_update"]["version"] == "2.0.0"
    if managed:
        manage(install, monkeypatch, {"version": 1, "extensions": {"when_unmet": "hold", "required": [{
            "name": update.name, "path": str(update.skill_dir), "content_hash": update.content_hash}]}})
    calls = []
    monkeypatch.setattr("ouroboros.extension_loader.unload_extension", lambda name, *a, **k: calls.append(("unload", name)))
    monkeypatch.setattr("ouroboros.extension_loader.reconcile_extension",
                        lambda name, *a, **k: calls.append(("reconcile", name)))
    settled = settle_pending_update(clawhub.drive, "demo")
    record = read_install_record(clawhub.drive, "demo")
    if managed:
        assert (settled["status"], settled["code"], settled["reason"]) == ("refused", "managed_by_deployment",
                                                                         "review_unfinished"), settled
        assert "cannot be rolled back" in settled["error"]
        assert (update.skill_dir / "SKILL.md").read_bytes() == after and calls == [("reconcile", "demo")]
        assert record["pending_update"]["version"] == "2.0.0" and not record.get("update_rejected")
        assert update_backup_path(clawhub.drive, "demo").exists(), "the kept base stays for its administrator"
        kept = find_skill(clawhub.drive, "demo", repo_path="")
        assert kept.content_hash == update.content_hash
        assert not kept.review.gate_for(kept.content_hash)["executable_review"], "its review still decides"
    else:
        assert (settled["status"], settled["reason"]) == ("restored", "review_unfinished"), settled
        assert (first.target_dir / "SKILL.md").read_bytes() == before
        assert calls == [("unload", "demo"), ("reconcile", "demo")]
        assert record["update_rejected"]["version"] == "2.0.0" and not record.get("pending_update")


@pytest.mark.parametrize("managed", [True, False])
def test_repair_admission_and_file_editors_refuse_a_required_package(org, monkeypatch, managed):
    from starlette.applications import Starlette

    from ouroboros.gateway import files
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from supervisor import worker_promotion, workers

    if managed:
        require(org, monkeypatch)
    monkeypatch.setattr(workers, "DRIVE_ROOT", org.data)
    constraint, error = worker_promotion._canonical_promoted_repair_constraint(
        {"mode": "skill_repair", "skill_name": NAME, "payload_root": f"skills/external/{NAME}"})
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=org.data)
    registry.set_context(ToolContext(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=org.data,
                                     task_metadata={"work_kind": "owner"}))
    edit = registry.execute("write_file", {"root": "skill_payload", "bucket": "external", "skill_name": NAME,
                                           "path": "plugin.py", "content": "def register(api):\n    pass\n"})
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(org.data))  # the owner's file browser over the data root
    with TestClient(Starlette(routes=files.file_browser_routes())) as client:
        browser_edit = client.post("/api/files/write", json={"path": f"skills/external/{NAME}/notes.md", "content": "x",
                                                             "create": True})
        shadow = client.post("/api/files/mkdir", json={"path": "skills", "name": NAME})  # a second package's folder
        removal = client.post("/api/files/delete", json={"path": "skills/external"})
    if managed:
        assert (constraint, error) == (None, "managed_by_deployment")
        assert str(edit).startswith("⚠️ MANAGED_BY_DEPLOYMENT: monitor is required"), edit
        assert {browser_edit.status_code, shadow.status_code, removal.status_code} == {403}
        assert {answer.json()["code"] for answer in (browser_edit, shadow, removal)} == {"managed_by_deployment"}
        assert (org.skill_dir / "plugin.py").read_text() == PLUGIN and not (org.skill_dir / "notes.md").exists()
        assert not (org.data / "skills" / NAME).exists()
    else:
        assert error == "" and constraint is not None and "MANAGED_BY_DEPLOYMENT" not in str(edit)
        assert (browser_edit.status_code, shadow.status_code, removal.status_code) == (200, 200, 200)


# --- running state: hold and warn at the carrier ----------------------------------------------------------

def test_a_fixed_setting_the_package_never_reads_is_one_gap_for_settings_the_carrier_and_the_queue(org, monkeypatch):
    """The pinned package runs, but it does not declare a key the file fixes for it, so the value can never reach
    it: Settings, the carrier and the queue read that same gap, and ``hold`` holds."""
    from ouroboros.deployment_extensions import HOLD_REASON, hold_reason, queue_held, state_rows, unmet_here

    require(org, monkeypatch, settings={"ORG_DSN": SECRET, "ORG_REGION": "eu-west"})
    grant(org)
    reconcile(org)
    assert NAME in extension_loader.snapshot()["extensions"], "the package itself runs"
    [row] = state_rows(org.data)
    assert (row["state"], row["reason"]) == ("unmet", "settings_undeclared") and "ORG_REGION" in row["detail"]
    assert [gap["reason"] for gap in unmet_here(org.data)] == ["settings_undeclared"]
    assert hold_reason(org.data) == HOLD_REASON and queue_held() is True
    assert SECRET not in json.dumps(row) and "eu-west" not in json.dumps(row)


@pytest.mark.parametrize("mode", ["hold", "warn"])
def test_the_carrier_holds_or_warns_on_its_own_registry_and_records_a_change_once(org, monkeypatch, mode):
    from ouroboros.deployment_extensions import HOLD_REASON, hold_reason, hold_text, queue_held

    require(org, monkeypatch, mode=mode)  # reviewed, not yet loaded in this process
    first, second = hold_reason(org.data), hold_reason(org.data)
    assert first == second == (HOLD_REASON if mode == "hold" else "")
    assert queue_held() is (mode == "hold")
    [unmet] = events(org, HOLD_REASON)
    assert unmet["when_unmet"] == mode and unmet["requirements"][0]["name"] == NAME
    assert "monitor: It is not running in this process yet." in hold_text(org.data)
    reconcile(org)
    assert hold_reason(org.data) == "" and queue_held() is False
    assert len(events(org, "deployment_requirement_met")) == 1 and len(events(org, HOLD_REASON)) == 1


def test_without_a_policy_the_carrier_and_queue_check_nothing(org, monkeypatch):
    from ouroboros.deployment_extensions import hold_reason, queue_held, state_rows

    monkeypatch.setattr("ouroboros.deployment_extensions._live_gaps", lambda *_a, **_k: pytest.fail("read nothing"))
    assert (hold_reason(org.data), queue_held(), state_rows(org.data)) == ("", False, [])


def test_a_held_task_ends_typed_with_no_body_and_no_traceback(org, monkeypatch):
    from ouroboros.deployment_extensions import HOLD_REASON
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.task_results import load_task_result
    from supervisor.worker_process import _record_task_exception

    require(org, monkeypatch)
    error = PhysicalDispatchInterrupted(HOLD_REASON)
    error.before_task_start = True
    runner = type("Runner", (), {"env": type("Env", (), {"drive_root": org.data})()})()
    _record_task_exception(runner, {"id": "held-1", "chat_id": 1}, error, org.data)
    result = load_task_result(org.data, "held-1")
    assert (result["status"], result["reason_code"]) == ("failed", HOLD_REASON)
    assert result["result"].startswith("🔒 Not started: your organization requires") and NAME in result["result"]
    assert not events(org, "task_error"), "a deliberate hold is no exception terminal"


def test_a_held_conversation_turn_ends_typed_with_its_origin_and_one_not_started_final(org, monkeypatch):
    """The conversation lane's before-start hold: the turn's typed terminal carries the owner message it answers,
    as its Stop sibling's does, and the owner gets one keyed ``task_not_started`` final."""
    import queue
    from types import SimpleNamespace

    from ouroboros.deployment_extensions import HOLD_REASON
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import load_task_result
    from supervisor.worker_chat_lane import conversation_exception_terminal

    require(org, monkeypatch)
    ref = build_owner_message_ref(chat_id=1, client_message_id="cm-held", ts="2026-11-08T00:00:00+00:00", text="go")
    error = PhysicalDispatchInterrupted(HOLD_REASON)
    error.before_task_start = True
    agent = SimpleNamespace(env=SimpleNamespace(drive_root=org.data, drive_path=lambda name: org.data / name))
    frames = queue.Queue()
    task = {"id": "held-turn", "chat_id": 1, "metadata": {"_host_operation": True, "origin_message_ref": ref}}
    conversation_exception_terminal(agent, task, error, frames, org.data)
    result = load_task_result(org.data, "held-turn")
    assert (result["status"], result["reason_code"], result["origin_message_ref"]) == ("failed", HOLD_REASON, ref)
    [final] = [frames.get_nowait() for _ in range(frames.qsize())]
    assert (final["system_type"], final["progress_meta"]["task_terminal_status"]) == ("task_not_started", "failed")
    assert final["progress_meta"]["origin_message_ref"] == ref and final["text"] == result["result"]
    assert not events(org, "task_error"), "a deliberate hold is no exception terminal"


@pytest.mark.parametrize("mode", ["hold", "warn", None])
def test_a_carrier_ends_a_held_task_itself_and_never_runs_its_body(org, monkeypatch, mode):
    """The pool worker's (and inline executor's) task body: under ``hold`` the task's own typed end comes back as
    events, with no model call and no exception that would end the worker; otherwise the body runs."""
    from types import SimpleNamespace

    import ouroboros.agent_task_pipeline as pipeline
    from ouroboros.deployment_extensions import HOLD_REASON
    from supervisor import worker_process

    if mode:
        require(org, monkeypatch, mode=mode)  # reviewed, not loaded in this process
    ended = []

    def emit(_env, _memory, _llm, events, task, text, usage, trace, *_args, **kwargs):
        ended.append((text, usage, kwargs.get("ctx")))
        events.append({"type": "task_done", "task_id": task["id"], "reason_code": usage["reason_code"]})

    monkeypatch.setattr(pipeline, "emit_task_results", emit)
    ran = []
    runner = SimpleNamespace(env=SimpleNamespace(drive_root=org.data, drive_path=lambda name: org.data / name),
                             memory=None, llm=None, handle_task=lambda task: ran.append(task["id"]) or [])
    events = worker_process._run_task_body(runner, {"id": "queued-1", "chat_id": 1}, str(org.data), None)
    if mode == "hold":
        assert ran == [] and events == [{"type": "task_done", "task_id": "queued-1", "reason_code": HOLD_REASON}]
        [(text, usage, ctx)] = ended
        assert text.startswith("🔒 Not started: your organization requires an extension that isn’t running")
        assert (usage["terminal_origin"], ctx) == ("host_notice", None), "no other task's facts ride along"
    else:
        assert ran == ["queued-1"] and ended == []


@pytest.mark.parametrize("mode", ["hold", "warn", None])
def test_the_queue_keeps_ordinary_work_waiting_while_the_server_does_not_run_a_requirement(org, monkeypatch, mode):
    from supervisor import queue, worker_assignment

    if mode:
        require(org, monkeypatch, mode=mode)
    reached = []
    monkeypatch.setattr(worker_assignment, "continuation_refusal",
                        lambda candidate: reached.append(candidate["id"]) or True)  # the gate after it
    with queue._queue_lock:
        held = worker_assignment._candidate_ineligible(queue, {"id": "queued-2", "metadata": {}}, 5.0, {})
    assert held is True
    assert reached == ([] if mode == "hold" else ["queued-2"]), "hold stops at its own gate; otherwise the row goes on"
    reconcile(org)
    reached.clear()
    with queue._queue_lock:
        worker_assignment._candidate_ineligible(queue, {"id": "queued-3", "metadata": {}}, 5.0, {})
    assert reached == ["queued-3"], "once the server runs it, nothing is held"
