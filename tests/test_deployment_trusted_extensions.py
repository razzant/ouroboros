"""A deployment's trusted extensions under an execution domain, at their real owners, beside the unmanaged install.

With ``execution`` the service runs the code of exactly the packages ``extensions.trusted`` pins by name, folder and
bytes. Any other extension is never imported, cataloged, registered or launched (each plugin here appends a line to
its own log when imported, so an empty log proves none of it ran), and the staged copy a trusted one is imported
from must hash to its pin. Trust admits code and nothing more: review, grants and the employee's switch still decide,
and a trusted package keeps its whole PluginAPI, in process and in an out-of-process child. Its files, dependencies
and staged imports are its administrator's; turning it off stays the employee's. Without ``execution`` (unmanaged,
or a policy naming none) nothing reads trust.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from starlette.testclient import TestClient

from ouroboros import extension_loader
from tests._extension_loader_shared import _clear_loader_state, _mark_isolated_deps_installed, _write_ext_skill  # noqa: F401
from tests.test_deployment_policy import install as install_fixture
from tests.test_deployment_policy import manage

install = install_fixture  # the shared isolated data root, as a fixture of this module

NAME = "audit_widget"
REPO = pathlib.Path(__file__).resolve().parents[1]
MODES = ("unmanaged", "no_execution", "trusted", "untrusted")


def plugin(log: pathlib.Path) -> str:
    """An extension whose import appends one line to ``log`` and whose one tool answers a fixed text."""
    return (f"with open({str(log)!r}, 'a', encoding='utf-8') as _log:\n"
            "    _log.write('imported\\n')\n"
            "def _hello(ctx):\n"
            "    return 'hello from trusted code'\n"
            "def register(api):\n"
            "    api.register_tool('hello', _hello, description='hello', schema={})\n")


def imports(org) -> int:
    return len(org.log.read_text(encoding="utf-8").splitlines()) if org.log.exists() else 0


def write_package(org, *, name=NAME, bucket="external", frontmatter="", log=None):
    """A reviewed, enabled extension package at ``skills/<bucket>/<name>``; returns its loaded skill."""
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_enabled, save_review_state

    skill_dir = _write_ext_skill(org.data / "skills" / bucket, name, plugin_body=plugin(log or org.log),
                                 permissions=["tool"], extra_frontmatter=frontmatter)
    loaded = load_skill(skill_dir, org.data)
    save_review_state(org.data, name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    save_enabled(org.data, name, True, actor="owner_ui")
    return load_skill(skill_dir, org.data)


@pytest.fixture
def org(install, monkeypatch, tmp_path):
    """An isolated data root with one reviewed, enabled extension in ``skills/external`` and a work folder."""
    empty = tmp_path / "no-local-skills"
    empty.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(empty))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    install.log = tmp_path / "imports.log"
    install.work = tmp_path / "work"
    install.work.mkdir()
    install.skill = write_package(install)
    return install


def execution(org, **extra) -> dict:
    return {"executor": {"type": "docker_exec", "container_name": "ouro-work", "network": "host",
                         "path_mappings": [{"host_path": str(org.work), "backend_path": str(org.work)}]}, **extra}


def pin(skill, **changes) -> dict:
    return {"name": skill.name, "path": str(skill.skill_dir), "content_hash": skill.content_hash, **changes}


def protect(org, monkeypatch, trusted=None, **extensions):
    """Name the deployment's file the way its launch does: an execution domain and the packages it trusts."""
    rows = [pin(org.skill)] if trusted is None else trusted
    return manage(org, monkeypatch, {"version": 1, "execution": execution(org),
                                     "extensions": {"trusted": rows, **extensions}})


def apply_mode(org, monkeypatch, mode: str) -> None:
    if mode == "no_execution":
        manage(org, monkeypatch, {"version": 1, "rules": "Customer records stay inside."})
    elif mode == "trusted":
        protect(org, monkeypatch)
    elif mode == "untrusted":
        protect(org, monkeypatch, trusted=[])


def reconcile(org, name=NAME):
    from ouroboros.config import load_settings

    return extension_loader.reconcile_extension(name, org.data, load_settings)


def call_hello(org, name=NAME) -> str:
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=REPO, drive_root=org.data)
    registry.set_context(ToolContext(repo_dir=REPO, drive_root=org.data, task_metadata={"work_kind": "owner"}))
    return str(registry.execute(extension_loader.extension_surface_name(name, "hello"), {}))


# --- the file -------------------------------------------------------------------------------------------

DIGEST = "0" * 64


def test_trust_exists_only_with_an_execution_domain_and_is_strict(org):
    from ouroboros.deployment_policy import DeploymentPolicyError, parse_policy

    def parse(extensions, **document):
        return parse_policy(json.dumps({"version": 1, "extensions": extensions, **document}).encode())

    with pytest.raises(DeploymentPolicyError, match="extensions.trusted applies only with execution"):
        parse({"trusted": []})
    row = {"name": "a", "path": "/opt/plugins/a", "content_hash": DIGEST}
    for bad, reason in [({**row, "name": "My Plugin"}, "canonical skill name"),
                        ({**row, "path": "plugins/a"}, "absolute package directory"),
                        ({**row, "content_hash": "abc"}, "64-hex digest"),
                        ({**row, "settings": {}}, "unsupported field 'settings'")]:
        with pytest.raises(DeploymentPolicyError, match=reason):
            parse({"trusted": [bad]}, execution=execution(org))
    with pytest.raises(DeploymentPolicyError, match="distinct canonical skill name"):
        parse({"trusted": [row, {**row, "path": "/opt/plugins/b"}]}, execution=execution(org))
    with pytest.raises(DeploymentPolicyError, match="at most 64"):
        parse({"trusted": [{**row, "name": f"p{i}", "path": f"/opt/p{i}"} for i in range(65)]}, execution=execution(org))
    policy = parse({"trusted": [row]}, execution=execution(org))
    assert [(item.name, item.path, item.content_hash) for item in policy.trusted] == [("a", "/opt/plugins/a", DIGEST)]
    assert parse({}, execution=execution(org)).trusted == () and parse({}).trusted == ()


@pytest.mark.parametrize("where", ["inside", "holding", "state_root"])
def test_a_trusted_package_never_lies_where_the_work_domain_writes(org, where):
    from ouroboros.deployment_policy import DeploymentPolicyError, parse_policy

    state_root = org.work / "skill-state"
    path = {"inside": org.work / "plugins" / "a", "holding": org.work.parent, "state_root": state_root / "a"}[where]
    document = {"version": 1, "execution": execution(org, skill_state_root=str(state_root)),
                "extensions": {"trusted": [{"name": "a", "path": str(path), "content_hash": DIGEST}]}}
    with pytest.raises(DeploymentPolicyError, match="overlaps the work domain"):
        parse_policy(json.dumps(document).encode())


def test_with_execution_every_required_package_is_one_of_the_trusted(org):
    from ouroboros.deployment_policy import DeploymentPolicyError, parse_policy

    required = {"name": "a", "path": "/opt/plugins/a", "content_hash": DIGEST}

    def parse(trusted, *, domain=True):
        return parse_policy(json.dumps({"version": 1, **({"execution": execution(org)} if domain else {}),
                                        "extensions": {"when_unmet": "hold", "required": [required],
                                                       **({"trusted": trusted} if domain else {})}}).encode())

    for trusted in ([], [{**required, "content_hash": "1" * 64}], [{**required, "path": "/opt/plugins/b"}]):
        with pytest.raises(DeploymentPolicyError, match=r"extensions.required\[0\] must also be in extensions.trusted"):
            parse(trusted)
    assert parse([required]).extensions[0].name == "a"
    assert parse(None, domain=False).trusted == (), "without execution a required row stands alone, as before"


def test_settings_name_the_trusted_pins_only_with_an_execution_domain(org, monkeypatch):
    from ouroboros.deployment_policy import policy_state

    manage(org, monkeypatch, {"version": 1, "extensions": {"when_unmet": "warn", "required": [pin(org.skill)]}})
    assert "trusted" not in policy_state()["extensions"]
    protect(org, monkeypatch)
    assert policy_state()["extensions"] == {"when_unmet": "", "required": [], "trusted": [pin(org.skill)]}


# --- the loader: nothing untrusted is imported, cataloged or registered -----------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_only_trusted_bytes_load_in_process_and_keep_their_tools(org, monkeypatch, mode):
    from ouroboros.extension_liveness import runtime_state_for_skill_name

    apply_mode(org, monkeypatch, mode)
    state = reconcile(org)
    live = NAME in extension_loader.snapshot()["extensions"]
    if mode == "untrusted":
        assert (live, imports(org), state["reason"]) == (False, 0, "deployment_untrusted"), state
        assert "isn’t among them" in runtime_state_for_skill_name(NAME, org.data)["load_error"]
        refused = extension_loader.load_extension(org.skill, lambda: {}, drive_root=org.data)
        assert refused.startswith(f"skill {NAME!r} does not run on this service") and imports(org) == 0
    else:
        assert live and imports(org) == 1, state
        assert "hello from trusted code" in call_hello(org)


@pytest.mark.parametrize("change", ["bytes", "folder"])
def test_other_bytes_or_another_folder_of_a_trusted_name_never_load(org, monkeypatch, change):
    from ouroboros.extension_liveness import runtime_state_for_skill_name

    protect(org, monkeypatch)
    if change == "bytes":
        entry = org.skill.skill_dir / "plugin.py"
        entry.write_text(entry.read_text(encoding="utf-8") + "\n# edited after delivery\n", encoding="utf-8")
    else:
        protect(org, monkeypatch, trusted=[pin(org.skill, path=str(org.root / "elsewhere" / NAME))])
    reconcile(org)
    state = runtime_state_for_skill_name(NAME, org.data)
    assert NAME not in extension_loader.snapshot()["extensions"] and imports(org) == 0
    assert state["reason"] == "deployment_untrusted" and not state["desired_live"]
    assert ("differ from the version" if change == "bytes" else "another folder") in state["load_error"]


def test_a_staged_copy_that_is_not_the_pinned_bytes_is_replaced_before_anything_imports(org, monkeypatch):
    """A cached tree the loader would reuse (``complete.json`` present) is checked against the pin first."""
    from ouroboros.extension_import_staging import extension_import_identity
    from ouroboros.skill_loader import skill_state_dir

    protect(org, monkeypatch)
    planted_log = org.root / "planted.log"
    tree = skill_state_dir(org.data, NAME) / "__extension_imports" / extension_import_identity(org.skill)
    (tree / "skill").mkdir(parents=True)
    (tree / "skill" / "SKILL.md").write_text((org.skill.skill_dir / "SKILL.md").read_text(encoding="utf-8"),
                                             encoding="utf-8")
    (tree / "skill" / "plugin.py").write_text(plugin(planted_log), encoding="utf-8")
    (tree / "complete.json").write_text(json.dumps({"identity": tree.name, "content_hash": org.skill.content_hash,
                                                    "owner_pid": 1}), encoding="utf-8")
    reconcile(org)
    assert NAME in extension_loader.snapshot()["extensions"] and imports(org) == 1
    assert not planted_log.exists(), "the planted tree was never imported"
    assert (tree / "skill" / "plugin.py").read_text(encoding="utf-8") == plugin(org.log)


def test_a_copy_that_changes_while_staged_is_refused_and_not_imported(org, monkeypatch):
    from ouroboros import extension_import_staging

    protect(org, monkeypatch)
    real_copy = extension_import_staging._copy_extension_tree

    def racing_copy(skill, import_root):
        real_copy(skill, import_root)
        (import_root / "skill" / "plugin.py").write_text("raise SystemExit('swapped')\n", encoding="utf-8")

    monkeypatch.setattr(extension_import_staging, "_copy_extension_tree", racing_copy)
    error = extension_loader.load_extension(org.skill, lambda: {}, drive_root=org.data)
    assert "changed while it was staged" in error and imports(org) == 0
    assert NAME not in extension_loader.snapshot()["extensions"]


@pytest.mark.parametrize("trusted", [True, False])
def test_an_out_of_process_extension_runs_only_trusted_and_keeps_its_child_dispatch(org, monkeypatch, trusted):
    """The catalog child and every per-call child load the package themselves: trusted, the tool answers through
    them; untrusted, the service refuses before any child starts."""
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state

    skill_md = org.skill.skill_dir / "SKILL.md"
    skill_md.write_text(skill_md.read_text(encoding="utf-8").replace(
        "type: extension\n", "type: extension\ndependencies:\n  - dummy_pkg\n"), encoding="utf-8")
    loaded = find_skill(org.data, NAME)
    save_review_state(org.data, NAME, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    _mark_isolated_deps_installed(org.data, loaded)
    protect(org, monkeypatch, trusted=[pin(loaded)] if trusted else [])
    reconcile(org)
    calls = loaded.skill_dir.parents[2] / "state" / "skills" / NAME / "extension_calls"
    if trusted:
        tool = extension_loader.get_tool(extension_loader.extension_surface_name(NAME, "hello"))
        assert tool is not None and tool["out_of_process"] is True
        assert "hello from trusted code" in call_hello(org)
        assert imports(org) >= 2, "the catalog child and the call's child each imported the trusted bytes"
    else:
        assert NAME not in extension_loader.snapshot()["extensions"] and imports(org) == 0
        assert not calls.exists() or not any(calls.iterdir()), "no child was ever started"


def test_a_child_asked_to_load_an_untrusted_package_refuses_before_import(org, monkeypatch):
    from ouroboros.extension_process_runner import ExtensionProcessError, _load_child_extension

    protect(org, monkeypatch, trusted=[])
    with pytest.raises(ExtensionProcessError, match="does not run on this service"):
        _load_child_extension(NAME, org.data, REPO, org.root / "no-local-skills")
    assert imports(org) == 0


def test_host_service_tokens_speak_only_for_trusted_code(org, monkeypatch):
    from ouroboros.extension_plugin_api import mint_skill_token
    from ouroboros.gateway.host_service import HostServiceAuthError, HostServiceContext
    from ouroboros.skill_loader import skill_state_dir

    protect(org, monkeypatch)
    token = mint_skill_token(skill_state_dir(org.data, NAME), NAME, org.skill.skill_dir)
    assert HostServiceContext(org.data).authenticate_token_payload(token)[0] == NAME
    protect(org, monkeypatch, trusted=[])
    with pytest.raises(HostServiceAuthError, match="not code this deployment's service runs"):
        HostServiceContext(org.data).authenticate_token_payload(token)


# --- the doors: the switch is the employee's, the files the administrator's --------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_turning_on_an_untrusted_extension_refuses_before_any_write_and_off_stays_the_employees(org, monkeypatch, mode):
    from ouroboros.skill_lifecycle_actions import run_skill_action
    from ouroboros.skill_loader import save_enabled
    from ouroboros.tools.registry import ToolContext

    save_enabled(org.data, NAME, False, actor="owner_ui")
    apply_mode(org, monkeypatch, mode)
    enabled_path = org.data / "state" / "skills" / NAME / "enabled.json"
    before = enabled_path.read_text(encoding="utf-8")
    ctx = ToolContext(repo_dir=org.data, drive_root=org.data)
    on = run_skill_action(ctx, NAME, "enable", _owner_actor="owner_ui")
    if mode == "untrusted":
        assert (on["ok"], on["status_code"], on["code"]) == (False, 403, "managed_by_deployment"), on
        assert enabled_path.read_text(encoding="utf-8") == before and imports(org) == 0
        return
    assert on["ok"] and NAME in extension_loader.snapshot()["extensions"], on
    off = run_skill_action(ctx, NAME, "disable", _owner_actor="owner_ui")
    assert off["ok"] and NAME not in extension_loader.snapshot()["extensions"], off


@pytest.mark.parametrize("mode", ["unmanaged", "trusted"])
def test_a_trusted_packages_files_dependencies_and_staged_imports_refuse_every_editor(org, monkeypatch, mode):
    from starlette.applications import Starlette

    from ouroboros.deployment_extensions import change_refusal, install_refusal, subtree_refusal
    from ouroboros.gateway import files
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    apply_mode(org, monkeypatch, mode)
    original = (org.skill.skill_dir / "plugin.py").read_text(encoding="utf-8")
    registry = ToolRegistry(repo_dir=REPO, drive_root=org.data)
    registry.set_context(ToolContext(repo_dir=REPO, drive_root=org.data, task_metadata={"work_kind": "owner"}))
    edit = registry.execute("write_file", {"root": "skill_payload", "bucket": "external", "skill_name": NAME,
                                           "path": "plugin.py", "content": "def register(api):\n    pass\n"})
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(org.data))  # the owner's file browser over the data root
    targets = (f"skills/external/{NAME}/notes.md", f"state/skills/{NAME}/__extension_imports/sha256-planted/skill/plugin.py")
    dependency = f"skills/external/{NAME}/.ouroboros_env/python/lib/injected.py"
    for path in (*targets, dependency):  # the same folders in every mode, so only the refusal differs
        (org.data / path).parent.mkdir(parents=True, exist_ok=True)
    with TestClient(Starlette(routes=files.file_browser_routes())) as client:
        answers = [client.post("/api/files/write", json={"path": path, "content": "x", "create": True})
                   for path in targets]
        injected = client.post("/api/files/write", json={"path": dependency, "content": "x", "create": True})
    # A package's dependency folder is a control folder for every editor in every mode (skill_payload_policy).
    assert injected.status_code in (400, 403) and not (org.data / dependency).exists(), injected.text
    verbs = {verb: change_refusal(NAME, verb) for verb in
             ("disable", "update", "uninstall", "delete", "rollback", "repair", "edit", "install")}
    if mode == "trusted":
        assert str(edit).startswith(f"⚠️ MANAGED_BY_DEPLOYMENT: {NAME} is delivered by this deployment"), edit
        assert [answer.status_code for answer in answers] == [403, 403], [a.text for a in answers]
        assert {answer.json()["code"] for answer in answers} == {"managed_by_deployment"}
        assert verbs.pop("disable") == "" and all(verbs.values()), verbs
        assert install_refusal(org.data, NAME) and subtree_refusal(org.data / "skills" / "external")
        assert (org.skill.skill_dir / "plugin.py").read_text(encoding="utf-8") == original
        assert not any((org.data / path).exists() for path in targets), "nothing was written"
    else:
        assert "MANAGED_BY_DEPLOYMENT" not in str(edit)
        assert [answer.status_code for answer in answers] == [200, 200]
        assert not any(verbs.values()) and install_refusal(org.data, NAME) == ""


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("mode", ["unmanaged", "trusted", "untrusted"])
def test_the_dependency_owner_installs_nothing_into_a_trusted_package(org, monkeypatch, mode, staged):
    """With a ``staging_root`` too, an extension's packages stay its own folder's: nothing of it is copied into the
    work domain (``work_skill_payloads.deps_home``), which only script skills use."""
    from ouroboros.marketplace import isolated_deps
    from ouroboros.skill_dependencies import ensure_skill_dependencies
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state

    skill_md = org.skill.skill_dir / "SKILL.md"
    skill_md.write_text(skill_md.read_text(encoding="utf-8").replace(
        "type: extension\n", "type: extension\ndependencies:\n  - dummy_pkg\n"), encoding="utf-8")
    loaded = find_skill(org.data, NAME)
    save_review_state(org.data, NAME, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    loaded = find_skill(org.data, NAME)
    staging = org.work / "staging"
    if mode != "unmanaged":
        protect(org, monkeypatch, trusted=[pin(loaded)] if mode == "trusted" else [])
    if mode != "unmanaged" and staged:
        manage(org, monkeypatch, {"version": 1, "execution": execution(org, staging_root=str(staging)),
                                  "extensions": {"trusted": [pin(loaded)] if mode == "trusted" else []}})
    installs = []
    monkeypatch.setattr(isolated_deps, "install_isolated_dependencies",
                        lambda *args, **kwargs: installs.append(args[1]) or {"status": "installed"})
    status, detail = ensure_skill_dependencies(org.data, loaded)
    if mode == "trusted":
        assert (status, installs) == ("failed", []) and "administrator" in detail
        assert not (loaded.skill_dir / ".ouroboros_env").exists(), "nothing was written into the package"
    else:
        assert (status, installs) == ("installed", [NAME]), detail
    assert not (staging / "skills").exists(), "no extension is copied into the work domain"
    _mark_isolated_deps_installed(org.data, loaded)  # what the administrator delivers with the package
    assert ensure_skill_dependencies(org.data, find_skill(org.data, NAME))[0] == "installed"


# --- what Skills shows --------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_skills_name_each_extensions_trust_only_with_an_execution_domain(org, monkeypatch, mode):
    from ouroboros.gateway.extensions import _build_extensions_index

    write_package(org, name="notes_helper", log=org.root / "other.log")
    apply_mode(org, monkeypatch, mode)
    rows = {row["name"]: row for row in _build_extensions_index(org.data, str(org.root / "no-local-skills"))["skills"]}
    trust = {name: row.get("trust") for name, row in rows.items()}
    if mode in ("unmanaged", "no_execution"):
        assert trust == {NAME: None, "notes_helper": None}
        return
    unlisted = {"state": "untrusted", "reason": "not_delivered", "administered": False,
                "detail": "Only extensions your organization’s administrator delivers run here, and this one isn’t "
                          "among them."}
    assert trust["notes_helper"] == unlisted
    assert trust[NAME] == ({"state": "trusted", "administered": True} if mode == "trusted" else unlisted)
    assert rows["notes_helper"]["live_reason"] == "deployment_untrusted"


@pytest.mark.parametrize("protected", [False, True])
def test_the_dictation_chooser_offers_no_built_in_the_service_would_not_run(org, monkeypatch, protected):
    """Installed built-ins not yet on and bundled ones not installed are offered as engines; under an execution
    domain only those the policy trusts, since turning on any other one is refused."""
    from ouroboros import skill_builtin
    from ouroboros.gateway import contributions

    bundled = org.root / "bundled"
    for name in ("voice_a", "voice_b"):
        _write_ext_skill(bundled, name, plugin_body="def register(api):\n    pass\n", permissions=[])
    native = write_package(org, name="voice_c", bucket="native", log=org.root / "voice_c.log")
    (native.skill_dir / skill_builtin.SEED_MARKER).write_text("builtin\n", encoding="utf-8")  # a seeded built-in
    monkeypatch.setattr(skill_builtin, "bundled_skills_dir", lambda: bundled)
    monkeypatch.setattr(skill_builtin, "applies_here", lambda _folder: True)
    monkeypatch.setattr("ouroboros.extension_ui_validation.validate_contributions", lambda manifest: (
        [{"id": "stt", "label": manifest.name, "title": manifest.name, "destination": "", "provides": "stt"}], []))
    if protected:
        protect(org, monkeypatch, trusted=[pin(org.skill), {"name": "voice_b", "path": "/opt/plugins/voice_b",
                                                            "content_hash": DIGEST}])
    offers = contributions.contribution_catalog(org.data, str(org.root / "no-local-skills"))["offers"]["stt"]
    assert sorted(row["skill"] for row in offers) == (["voice_b"] if protected else ["voice_a", "voice_b", "voice_c"])


# --- a failure of a package whose settings the organization fixes -------------------------------------------

DSN = "https://fixture-key@errors.invalid/7"
FRAGMENT = DSN[:24]  # what an SDK's DSN parser quotes back: part of the value, which whole-value redaction misses
FAILING = ("API = None\n"
           "def register(api):\n"
           "    dsn = api.get_settings(['ORG_DSN']).get('ORG_DSN', '')\n"
           "    raise ValueError(f'Unsupported DSN {dsn[:24]!r}')\n")


def failing_package(org, monkeypatch, case: str):
    """A reviewed, granted, enabled extension whose ``register`` quotes part of its ORG_DSN. ``case``: ``fixed`` (the
    organization requires it and fixes ORG_DSN), ``required_unfixed`` (required, ORG_DSN the employee's own) or
    ``unmanaged`` (the employee's own setting, no policy); ``_oop`` loads it in an out-of-process child."""
    from ouroboros.skill_loader import (
        SkillReviewState, load_skill, save_enabled, save_review_state, save_skill_grants,
    )

    frontmatter = "dependencies:\n  - dummy_pkg\n" if case.endswith("_oop") else ""
    skill_dir = _write_ext_skill(org.data / "skills" / "external", "org_monitor", plugin_body=FAILING,
                                 permissions=["tool", "read_settings"], env_from_settings=["ORG_DSN"],
                                 extra_frontmatter=frontmatter)
    loaded = load_skill(skill_dir, org.data)
    save_review_state(org.data, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    save_enabled(org.data, loaded.name, True, actor="owner_ui")
    if frontmatter:
        _mark_isolated_deps_installed(org.data, loaded)
    row = {"name": loaded.name, "path": str(loaded.skill_dir), "content_hash": loaded.content_hash}
    if case.startswith("fixed"):
        manage(org, monkeypatch, {"version": 1, "extensions": {"when_unmet": "warn", "required": [
            {**row, "settings": {"ORG_DSN": DSN}}]}})
    else:
        org.path.write_text(json.dumps({"ORG_DSN": DSN}), encoding="utf-8")  # the employee's own value
        if case.startswith("required_unfixed"):
            manage(org, monkeypatch, {"version": 1, "extensions": {"when_unmet": "warn", "required": [row]}})
    # A custom key is requested once a value is present, so the employee grants it after the value exists.
    save_skill_grants(org.data, loaded.name, ["ORG_DSN"], content_hash=loaded.content_hash, requested_keys=["ORG_DSN"],
                      granted_permissions=[], requested_permissions=[])
    return loaded


def failure_texts(org, name, caplog) -> dict:
    """Every place a failure is told: the reconcile's answer, liveness, Skills, the manifest read, the requirement
    row, the durable health record and the logs."""
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros.deployment_extensions import state_rows
    from ouroboros.extension_health import read_extension_health
    from ouroboros.extension_liveness import runtime_state_for_skill_name
    from ouroboros.gateway.extensions import _build_extensions_index, api_extension_manifest

    answer = reconcile(org, name)
    app = Starlette(routes=[Route("/api/extensions/{skill}/manifest", api_extension_manifest)])
    app.state.drive_root = org.data
    with TestClient(app) as client:
        manifest = client.get(f"/api/extensions/{name}/manifest").json()
    listing = {row["name"]: row for row in _build_extensions_index(org.data, str(org.root / "no-local-skills"))["skills"]}
    health = read_extension_health(org.data, name) or {}
    return {"reconcile": str(answer.get("load_error") or ""),
            "liveness": str(runtime_state_for_skill_name(name, org.data).get("load_error") or ""),
            "skills": str(listing[name].get("load_error") or ""), "manifest": str(manifest.get("load_error") or ""),
            "requirement": json.dumps(state_rows(org.data), ensure_ascii=False),
            "health": json.dumps(health, ensure_ascii=False),
            "logs": "\n".join(record.getMessage() + "\n" + (record.exc_text or "") for record in caplog.records)}


@pytest.mark.parametrize("case", ["fixed", "fixed_oop"])
def test_a_failure_never_quotes_an_organization_fixed_value_yet_keeps_its_category(org, monkeypatch, caplog, case):
    import logging

    caplog.set_level(logging.DEBUG)
    loaded = failing_package(org, monkeypatch, case)
    texts = failure_texts(org, loaded.name, caplog)
    leaks = {where: text for where, text in texts.items() if FRAGMENT in text or "fixture-key" in text}
    assert not leaks, leaks
    category = "out-of-process catalog failure: ExtensionProcessError" if case == "fixed_oop" else "load failure: ValueError"
    for where in ("reconcile", "liveness", "skills", "manifest"):
        assert category in texts[where] and "withheld" in texts[where], (where, texts[where])
    assert "It could not start" in texts["requirement"] and "withheld" in texts["requirement"]
    assert "load_error" in texts["health"] and "withheld" in texts["health"]
    assert NAME not in extension_loader.snapshot()["extensions"] and loaded.name not in extension_loader.snapshot()["extensions"]


@pytest.mark.parametrize("case", ["unmanaged", "required_unfixed", "unmanaged_oop"])
def test_a_failure_with_no_organization_value_keeps_its_own_words(org, monkeypatch, caplog, case):
    """The employee's own setting is no organization's secret: the diagnostic stays whole, as it always was."""
    import logging

    caplog.set_level(logging.DEBUG)
    loaded = failing_package(org, monkeypatch, case)
    texts = failure_texts(org, loaded.name, caplog)
    for where in ("reconcile", "liveness", "skills", "manifest"):
        assert FRAGMENT in texts[where] and "withheld" not in texts[where], (where, texts[where])
