"""The deployment policy: one administrator file constrains what settings read, never the employee's document.

Every behaviour is proved in pairs at a real consumer: managed (a policy file named by the launch) and
unmanaged (personal OSS, which must not change). Readers are the loader, ``runtime_setting``,
``runtime_environ``, ``runtime_settings`` and ``get_runtime_mode``; writers are the transactional owner writer,
``config.save_settings``, the generic Settings save and the single-decision endpoints; the Safety Supervisor's
route walk, hold and second check; model-wait switches and restores; the organization's rules in every mind;
and the server start, where an unusable file stops the process before it serves.
"""

from __future__ import annotations


def test_policy_on_another_windows_drive_is_disjoint_not_a_startup_error(monkeypatch):
    import ntpath
    from pathlib import PureWindowsPath
    from types import SimpleNamespace
    import pytest
    from ouroboros import deployment_policy as policy, launch_roots

    monkeypatch.setattr(policy, "Path", PureWindowsPath)
    monkeypatch.setattr(policy, "os", SimpleNamespace(path=ntpath))
    monkeypatch.setattr(launch_roots, "configured_roots", lambda: SimpleNamespace(
        data_dir=r"C:\data", repo_dir=r"C:\repo"))
    assert str(policy._policy_path(r"D:\admin\policy.json")) == r"D:\admin\policy.json"
    assert str(policy._policy_path(r"C:\admin\policy.json")) == r"C:\admin\policy.json"
    for path in (r"C:\data\policy.json", r"C:\repo\policy.json"):
        with pytest.raises(policy.DeploymentPolicyError, match="lies inside"):
            policy._policy_path(path)

import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests._safety_fixtures import reset_safety_stores, supervisor as supervisor_fixture

supervisor = supervisor_fixture
REPO = pathlib.Path(__file__).resolve().parents[1]
CHECKER = "anthropic@safety::claude-opus-5-5"
FALLBACK = "openai@safety::gpt-6-luna"
RULES = "Keep customer records inside the corporate network."
POLICY = {"version": 1, "access": {"max": "pro"},
          "safety": {"min_coverage": "full", "checker": {"model": CHECKER, "fallbacks": [FALLBACK]}},
          "rules": RULES}
CONNECTIONS = [
    {"kind": "anthropic", "name": "safety", "label": "Organization checker", "api_key": "sk-ant-organization"},
    {"kind": "openai", "name": "safety", "label": "Organization fallback", "api_key": "sk-organization-fallback"},
    {"kind": "openai", "name": "mine", "label": "My key", "api_key": "sk-employee"},
]
# The employee's document as it stood before the installation became managed.
EMPLOYEE = {"OUROBOROS_RUNTIME_MODE": "cyber_pro", "OUROBOROS_SAFETY_MODE": "off",
            "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "off", "OUROBOROS_MODEL_SAFETY": "openai::employee-checker",
            "OUROBOROS_MODEL_ACCOUNTS": {"safety": "mine", "light": "theirs"}, "TOTAL_BUDGET": 50.0,
            "OUROBOROS_CONNECTIONS": CONNECTIONS}
SECOND_CHECK = {"enabled": True, "items": [{
    "subagent_id": "second", "recommended_use": "Check flags", "effort": "high", "delivery": "packet",
    "rechecks_flags": True, "route": {"kind": "api_model", "target_id": "openai@mine::reviewer"}}]}


@pytest.fixture
def install(tmp_path, monkeypatch):
    """An isolated data root: no inherited setting, policy file, generation pin or cached policy read."""
    from ouroboros import config, deployment_policy, route_choice

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    for key in (*config.SETTINGS_DEFAULTS, "OUROBOROS_MODEL_FALLBACK", config.BOOT_RUNTIME_MODE_ENV_KEY,
                deployment_policy.POLICY_ENV, deployment_policy.GENERATION_ENV):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(route_choice, "_VIEW", None)
    config.reset_runtime_mode_baseline_for_tests()
    deployment_policy.reset_for_tests()
    yield SimpleNamespace(root=tmp_path, data=data, path=data / "settings.json")
    deployment_policy.reset_for_tests()
    config.reset_runtime_mode_baseline_for_tests()


def manage(install, monkeypatch, document=None) -> pathlib.Path:
    """Name a policy file the way a deployment's launch does, outside the data root."""
    from ouroboros import deployment_policy

    path = install.root / "admin" / "deployment-policy.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(POLICY if document is None else document), encoding="utf-8")
    monkeypatch.setenv(deployment_policy.POLICY_ENV, str(path))
    deployment_policy.reset_for_tests()
    return path


def write_document(install, document: dict) -> None:
    install.path.write_text(json.dumps(document), encoding="utf-8")


def stored(install) -> dict:
    return json.loads(install.path.read_text(encoding="utf-8"))


# --- the file and its generation ------------------------------------------------------------------------

def test_without_a_named_file_personal_oss_is_unchanged(install, monkeypatch):
    from ouroboros import config, deployment_policy

    write_document(install, EMPLOYEE)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    assert deployment_policy.current() is None and deployment_policy.policy_state() is None
    effective = config.load_settings()
    assert (effective["OUROBOROS_SAFETY_MODE"], effective["OUROBOROS_MODEL_SAFETY"]) == ("off", "openai::employee-checker")
    assert config.get_runtime_mode() == "cyber_pro"
    proposed = {"OUROBOROS_SAFETY_MODE": "light"}
    assert deployment_policy.prepare_write(proposed, lambda: pytest.fail("an unmanaged write reads nothing more")) is proposed
    assert deployment_policy.choice_refusal("OUROBOROS_SAFETY_MODE", "off") == ""


@pytest.mark.parametrize("document,reason", [
    (None, "does not exist"),
    (b"{not json", "not a UTF-8 JSON document"),
    (b'{"version": 1, "version": 1}', "appears twice"),
    ({"version": 2}, "version 2 is unsupported"),
    ({"version": True}, "is unsupported"),
    ({"version": "1"}, "is unsupported"),
    ({"version": 1, "monitoring": {}}, "unsupported field 'monitoring'"),
    ({"version": 1, "access": {"max": "cyber_pro"}}, "access.max must be one of"),
    ({"version": 1, "safety": {"min_coverage": "off"}}, "safety.min_coverage must be one of"),
    ({"version": 1, "safety": {"checker": {"model": "claudexor@pin::codex=model"}}}, "safety.checker.model"),
    ({"version": 1, "safety": {"checker": {"model": CHECKER, "account": "pinned"}}}, "needs a claudexor:: model"),
    ({"version": 1, "safety": {"checker": {"model": CHECKER, "fallbacks": [FALLBACK] * 9}}}, "at most 8 routes"),
    ({"version": 1, "safety": {"checker": {"model": CHECKER, "effort": "high"}}}, "unsupported field 'effort'"),
    *(({"version": 1, "safety": {"allowed_exits": exits}}, "safety.allowed_exits must list distinct exits")
      for exits in ("release_unjudged", None, ["send_anyway"], [True], ["release_unjudged", "release_unjudged"])),
    ({"version": 1, "rules": "x" * 64_001}, "rules must be text"),
])
def test_a_named_policy_that_cannot_be_used_never_reads_as_unmanaged(install, monkeypatch, document, reason):
    from ouroboros import config, deployment_policy

    path = install.root / "deployment-policy.json"
    if document is not None:
        path.write_bytes(document if isinstance(document, bytes) else json.dumps(document).encode("utf-8"))
    monkeypatch.setenv(deployment_policy.POLICY_ENV, str(path))
    deployment_policy.reset_for_tests()
    for read in (deployment_policy.current, config.load_settings, config.get_runtime_mode,
                 lambda: config.runtime_setting("TOTAL_BUDGET")):
        with pytest.raises(deployment_policy.DeploymentPolicyError, match=reason):
            read()


@pytest.mark.parametrize("where,reason", [("relative", "must be an absolute path"), ("data", "inside the data root")])
def test_the_file_must_lie_outside_what_the_workspace_writes(install, monkeypatch, where, reason):
    from ouroboros import deployment_policy

    (install.data / "policy.json").write_text('{"version": 1}', encoding="utf-8")
    monkeypatch.setenv(deployment_policy.POLICY_ENV, "policy.json" if where == "relative" else str(install.data / "policy.json"))
    deployment_policy.reset_for_tests()
    with pytest.raises(deployment_policy.DeploymentPolicyError, match=reason):
        deployment_policy.current()


def test_a_server_generation_pins_its_bytes_for_every_child(install, monkeypatch):
    from ouroboros import deployment_policy

    path = manage(install, monkeypatch)
    policy = deployment_policy.load_for_generation()
    assert os.environ[deployment_policy.GENERATION_ENV] == policy.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    deployment_policy.reset_for_tests()  # a child process: one fresh read under the inherited pin
    assert deployment_policy.current() == policy
    path.write_text(json.dumps({**POLICY, "rules": "Edited while serving."}), encoding="utf-8")
    deployment_policy.reset_for_tests()
    with pytest.raises(deployment_policy.DeploymentPolicyError, match="restart to apply it"):
        deployment_policy.current()
    monkeypatch.delenv(deployment_policy.POLICY_ENV)
    deployment_policy.reset_for_tests()
    with pytest.raises(deployment_policy.DeploymentPolicyError, match="managed server generation"):
        deployment_policy.current()
    monkeypatch.setenv(deployment_policy.POLICY_ENV, str(path))
    assert deployment_policy.load_for_generation().rules == "Edited while serving.", "the next start applies the edit"
    assert os.environ[deployment_policy.GENERATION_ENV] == hashlib.sha256(path.read_bytes()).hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_a_server_never_starts_unmanaged_under_an_unusable_policy(tmp_path):
    """The process boundary: an unsupported file stops ``server.py`` with exit 2 before it serves or writes settings."""
    from ouroboros.deployment_policy import POLICY_ENV
    from ouroboros.process_containment import ProcessContainer

    data, home, policy = tmp_path / "data", tmp_path / "home", tmp_path / "admin-policy.json"
    home.mkdir()
    policy.write_text('{"version": 2}', encoding="utf-8")
    env = {**os.environ, "HOME": str(home), "OUROBOROS_APP_ROOT": str(tmp_path / "app"),
           "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_SETTINGS_PATH": str(data / "settings.json"),
           "OUROBOROS_PID_FILE": str(tmp_path / "server.pid"), "OUROBOROS_SERVER_HOST": "127.0.0.1",
           "OUROBOROS_SERVER_PORT": str(_free_port()), "OUROBOROS_HOST_SERVICE_PORT": str(_free_port()),
           POLICY_ENV: str(policy)}
    container = ProcessContainer()
    try:
        server = container.spawn([sys.executable, "server.py"], cwd=str(REPO), env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = server.communicate(timeout=180)
    finally:
        try:
            error = container.reap()
        finally:
            container.close()
    assert not error, error
    assert server.returncode == 2, out
    assert "Server not started" in out and "version 2 is unsupported" in out, out
    assert not (data / "settings.json").exists()


# --- readers --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_every_reader_applies_the_policy_after_each_view_and_the_document_stays_raw(install, monkeypatch, managed):
    from ouroboros import config
    from ouroboros.model_slots import get_safety_model
    from ouroboros.settings_integrity import (
        runtime_environ, runtime_setting, runtime_settings, selection_settings, task_settings_scope,
        task_settings_snapshot,
    )

    write_document(install, EMPLOYEE)
    before = install.path.read_bytes()
    for key, value in (("OUROBOROS_SAFETY_MODE", "off"), ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off"),
                       ("OUROBOROS_RUNTIME_MODE", "cyber_pro")):
        monkeypatch.setenv(key, value)  # environment overrides, as a launch or a child's parent may leave them
    if managed:
        manage(install, monkeypatch)
    coverage, access, checker = ("full", "pro", CHECKER) if managed else ("off", "cyber_pro", "openai::employee-checker")
    effective = config.load_settings()
    assert (effective["OUROBOROS_SAFETY_MODE"], effective["OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"]) == (coverage, coverage)
    assert (effective["OUROBOROS_RUNTIME_MODE"], effective["OUROBOROS_MODEL_SAFETY"]) == (access, checker)
    # The checker role's options are the policy's (Auto); every other role keeps the employee's own.
    assert json.loads(effective["OUROBOROS_MODEL_ACCOUNTS"]) == ({"light": "theirs"} if managed else EMPLOYEE["OUROBOROS_MODEL_ACCOUNTS"])
    assert (config.get_runtime_mode(), get_safety_model()) == (access, checker)
    assert config.get_safety_mode() == config.get_safety_mode("consciousness") == coverage
    # An old task snapshot, the live route view and the projected child environment never widen it again.
    old = task_settings_snapshot({**effective, "OUROBOROS_SAFETY_MODE": "off"},
                                 {**os.environ, "OUROBOROS_SAFETY_MODE": "off", "OUROBOROS_MODEL_SAFETY": "openai::old"})
    with task_settings_scope(old):
        assert runtime_setting("OUROBOROS_SAFETY_MODE") == runtime_settings()["OUROBOROS_SAFETY_MODE"] == coverage
        assert runtime_setting("OUROBOROS_MODEL_SAFETY") == selection_settings()["OUROBOROS_MODEL_SAFETY"] == checker
        child = runtime_environ()
        assert (child["OUROBOROS_SAFETY_MODE"], child["OUROBOROS_RUNTIME_MODE"]) == (coverage, access)
    assert install.path.read_bytes() == before, "a read never writes"


def test_a_missing_settings_document_still_reads_under_the_policy(install, monkeypatch):
    from ouroboros import config
    from ouroboros.model_slots import get_safety_model

    for key, value in (("OUROBOROS_SAFETY_MODE", "off"), ("OUROBOROS_MODEL_SAFETY", "openai::x"),
                       ("OUROBOROS_RUNTIME_MODE", "cyber_pro")):
        monkeypatch.setenv(key, value)
    manage(install, monkeypatch)
    assert not install.path.exists()
    assert (config.runtime_setting("OUROBOROS_SAFETY_MODE"), get_safety_model(), config.get_runtime_mode()) == (
        "full", CHECKER, "pro")
    assert config.load_settings()["OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"] == "full"
    assert not install.path.exists()


def test_the_access_ceiling_reads_over_the_raw_ratchet_pin(install, monkeypatch):
    from ouroboros import config

    write_document(install, EMPLOYEE)
    manage(install, monkeypatch)
    config.initialize_runtime_mode_baseline("cyber_pro")
    assert config.get_runtime_mode() == "pro" and os.environ[config.BOOT_RUNTIME_MODE_ENV_KEY] == "cyber_pro"
    config.save_settings({**config.load_settings(), "TOTAL_BUDGET": 25})  # no false elevation from the shadowed raw mode
    assert (stored(install)["OUROBOROS_RUNTIME_MODE"], stored(install)["TOTAL_BUDGET"]) == ("cyber_pro", 25)


# --- writers --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("writer", ["owner_document", "save_settings"])
def test_an_unrelated_save_keeps_every_shadowed_raw_choice(install, monkeypatch, writer):
    from ouroboros import config
    from ouroboros.gateway.owner_settings import _owner_write_settings

    write_document(install, EMPLOYEE)
    manage(install, monkeypatch)
    effective = config.load_settings()
    # The whole effective view comes back, with one unrelated change and one change of another role's account.
    view = {**effective, "TOTAL_BUDGET": 25, "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"light": "another"})}
    (_owner_write_settings if writer == "owner_document" else config.save_settings)(view)
    saved = stored(install)
    assert saved["TOTAL_BUDGET"] == 25
    assert {key: saved[key] for key in ("OUROBOROS_RUNTIME_MODE", "OUROBOROS_SAFETY_MODE", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE",
                                        "OUROBOROS_MODEL_SAFETY")} == {
        key: EMPLOYEE[key] for key in ("OUROBOROS_RUNTIME_MODE", "OUROBOROS_SAFETY_MODE",
                                       "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "OUROBOROS_MODEL_SAFETY")}
    assert json.loads(saved["OUROBOROS_MODEL_ACCOUNTS"]) == {"light": "another", "safety": "mine"}


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("target", ["foreign", "same", "alias"])
def test_the_colab_writer_is_exempt_only_for_another_physical_root(install, monkeypatch, managed, target):
    """Another data root is another deployment, written as generated; this process's own document, under any
    spelling, goes through the owner's whole-document writer: the policy judges it first (managed), and
    without one the notebook owner's choices land as the foreign branch writes them."""
    from ouroboros.colab_bootstrap import write_colab_settings
    from ouroboros.config import serialize_settings
    from ouroboros.deployment_policy import DeploymentPolicyRefusal

    write_document(install, {"TOTAL_BUDGET": 50.0})
    if managed:
        manage(install, monkeypatch)
    (install.root / "alias").symlink_to(install.data, target_is_directory=True)
    data_dir = {"foreign": install.root / "drive" / "data", "same": install.data, "alias": install.root / "alias"}[target]
    generated = {"OUROBOROS_SAFETY_MODE": "off", "TOTAL_BUDGET": 25.0}
    before = install.path.read_bytes()
    if target == "foreign":
        assert write_colab_settings(data_dir, generated).read_text(encoding="utf-8") == serialize_settings(generated)
        assert install.path.read_bytes() == before
        return
    if not managed:
        write_colab_settings(data_dir, generated)
        assert install.path.read_text(encoding="utf-8") == serialize_settings(generated), "the owner's bytes, as before"
        return
    with pytest.raises(DeploymentPolicyRefusal, match="deployment's policy file"):
        write_colab_settings(data_dir, generated)
    assert install.path.read_bytes() == before, "refused before any byte"
    write_colab_settings(data_dir, {"TOTAL_BUDGET": 25.0})  # an ordinary change still lands
    assert stored(install) == {"TOTAL_BUDGET": 25.0}

def _rows(**api_keys) -> list:
    return [{**row, "api_key": api_keys.get(f"{row['kind']}_{row['name']}", row["api_key"])} for row in CONNECTIONS]


ATTEMPTS = [
    ("OUROBOROS_SAFETY_MODE", "light"),
    ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off"),
    ("OUROBOROS_RUNTIME_MODE", "cyber_pro"),
    ("OUROBOROS_MODEL_SAFETY", "openai::another-checker"),
    ("OUROBOROS_MODEL_ACCOUNTS", {"safety": "other", "light": "theirs"}),
    ("OUROBOROS_MODEL_CONTEXT_WINDOWS", {"safety": 4096}),
    ("OUROBOROS_MODEL_PROCESSING_PREFERENCES", {"safety": "fast"}),
    ("OUROBOROS_CONNECTIONS", _rows(anthropic_safety="sk-ant-redirected")),
    ("OUROBOROS_CONNECTIONS", _rows(openai_safety="sk-fallback-redirected")),
    ("OUROBOROS_SUBAGENTS", SECOND_CHECK),
]


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("key,value", ATTEMPTS)
def test_a_managed_change_is_refused_before_any_effect_and_free_without_a_policy(install, monkeypatch, managed, key, value):
    from ouroboros.deployment_policy import DeploymentPolicyRefusal
    from ouroboros.gateway.owner_settings import _owner_write_settings

    document = {**EMPLOYEE, "OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_SAFETY_MODE": "full",
                "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "full", "OUROBOROS_MODEL_SAFETY": CHECKER,
                "OUROBOROS_MODEL_ACCOUNTS": {"light": "theirs"}}
    write_document(install, document)
    if managed:
        manage(install, monkeypatch)
    receipts: list = []
    monkeypatch.setattr("ouroboros.review_pool_receipts.persist_write_receipts", lambda *args, **_kw: receipts.append(args))
    before = install.path.read_bytes()

    def write():  # an owner writer that may lower Safety: the policy binds it all the same
        _owner_write_settings({**document, key: value}, allow_safety_lowering=True)

    if managed:
        with pytest.raises(DeploymentPolicyRefusal, match="deployment's policy file"):
            write()
        assert install.path.read_bytes() == before and receipts == [], "refused before receipts or bytes"
    else:
        write()
        saved = stored(install)[key]
        assert (json.loads(saved) if isinstance(saved, str) and isinstance(value, dict) else saved) == value


def test_choices_the_policy_allows_still_save(install, monkeypatch):
    from ouroboros.gateway.owner_settings import _owner_write_settings

    write_document(install, {**EMPLOYEE, "OUROBOROS_SUBAGENTS": SECOND_CHECK})
    manage(install, monkeypatch)
    _owner_write_settings({**stored(install), "OUROBOROS_RUNTIME_MODE": "light",  # below the ceiling
                           "OUROBOROS_MODEL_LIGHT": "openai@mine::light-2",
                           "OUROBOROS_MODEL_ACCOUNTS": {"safety": "mine", "light": "new"},
                           "OUROBOROS_CONNECTIONS": _rows(openai_mine="sk-employee-rotated"),  # their own connection
                           "OUROBOROS_SUBAGENTS": {"enabled": True, "items": []}})  # a second-check row may go
    saved = stored(install)
    assert (saved["OUROBOROS_RUNTIME_MODE"], saved["OUROBOROS_MODEL_LIGHT"]) == ("light", "openai@mine::light-2")
    assert json.loads(saved["OUROBOROS_MODEL_ACCOUNTS"]) == {"safety": "mine", "light": "new"}
    assert saved["OUROBOROS_CONNECTIONS"][2]["api_key"] == "sk-employee-rotated"


def test_a_wizard_form_with_blank_managed_fields_completes_a_fresh_install(install, monkeypatch):
    """The setup form writes every model slot it shows; a blank Safety field over no saved value authors nothing."""
    from ouroboros import config
    from ouroboros.gateway.owner_settings import _owner_write_settings

    manage(install, monkeypatch)
    form = {**config.load_settings(), "OUROBOROS_MODEL_SAFETY": "", "OUROBOROS_MODEL": "openai@mine::main"}
    _owner_write_settings(form, authored_keys=("OUROBOROS_SAFETY_MODE", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"),
                          allow_safety_lowering=True)
    saved = stored(install)
    assert saved["OUROBOROS_MODEL"] == "openai@mine::main"
    assert "OUROBOROS_MODEL_SAFETY" not in saved and "OUROBOROS_SAFETY_MODE" not in saved, "no effective value authored"
    assert config.load_settings()["OUROBOROS_MODEL_SAFETY"] == CHECKER


@pytest.mark.parametrize("managed", [True, False])
def test_an_incidental_writer_names_explicit_choices(install, monkeypatch, managed):
    from ouroboros.deployment_policy import DeploymentPolicyRefusal
    from ouroboros.gateway.owner_settings import _owner_update_settings, fields_on_saved_document

    write_document(install, EMPLOYEE)  # the raw document already says cyber_pro
    if managed:
        manage(install, monkeypatch)
    before = install.path.read_bytes()
    write = lambda: _owner_update_settings(fields_on_saved_document({"OUROBOROS_RUNTIME_MODE": "cyber_pro"}))  # noqa: E731
    if managed:
        with pytest.raises(DeploymentPolicyRefusal, match="at most pro"):
            write()
        assert install.path.read_bytes() == before
    else:
        write()


def _owner_app(install):
    from ouroboros.gateway import safety_settings, settings as settings_gateway

    app = Starlette(routes=[
        Route("/api/owner/safety-mode", safety_settings.api_owner_safety_mode, methods=["POST"]),
        Route("/api/owner/runtime-mode", settings_gateway.api_owner_runtime_mode, methods=["POST"]),
        Route("/api/owner/safety-model", safety_settings.api_owner_safety_model, methods=["POST"]),
        Route("/api/settings", settings_gateway.api_settings_post, methods=["POST"])])
    app.state.drive_root = install.data
    app.state.repo_dir = install.data
    return app


@pytest.mark.parametrize("managed", [True, False])
def test_single_decisions_are_judged_as_requested_even_when_the_document_agrees(install, monkeypatch, managed):
    write_document(install, EMPLOYEE)  # raw: Safety off, Access cyber_pro, the employee's checker
    if managed:
        manage(install, monkeypatch)
    before = install.path.read_bytes()
    with TestClient(_owner_app(install)) as client:
        answers = [client.post("/api/owner/safety-mode", json={"mode": "off"}),
                   client.post("/api/owner/runtime-mode", json={"mode": "cyber_pro"}),
                   client.post("/api/owner/safety-model", json={"settings": {
                       "OUROBOROS_MODEL_SAFETY": "openai::employee-checker"}})]
    if managed:
        assert [answer.status_code for answer in answers] == [403, 403, 403], [answer.text for answer in answers]
        assert all(answer.json()["code"] == "managed_by_deployment" and answer.json()["saved"] is False for answer in answers)
        assert install.path.read_bytes() == before
    else:
        assert [answer.status_code for answer in answers] == [200, 200, 200], [answer.text for answer in answers]


@pytest.mark.parametrize("managed", [True, False])
def test_the_generic_save_answers_typed_and_keeps_raw_choices_on_a_round_trip(install, monkeypatch, managed):
    from ouroboros import config
    from ouroboros.gateway import settings as settings_gateway

    write_document(install, EMPLOYEE)
    if managed:
        manage(install, monkeypatch)
    monkeypatch.setattr(settings_gateway, "apply_runtime_provider_defaults", lambda value: (value, False, []))
    monkeypatch.setattr(settings_gateway, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)
    monkeypatch.setattr(settings_gateway, "_apply_settings_to_env", lambda *_a, **_k: None)
    monkeypatch.setattr(settings_gateway, "_apply_settings_save_side_effects", lambda *_a, **_k: None)
    shown = config.load_settings()["OUROBOROS_MODEL_ACCOUNTS"]
    with TestClient(_owner_app(install)) as client:
        redirect = client.post("/api/settings", json={"OUROBOROS_CONNECTIONS": _rows(anthropic_safety="sk-ant-redirected")})
        unrelated = client.post("/api/settings", json={"TOTAL_BUDGET": 25, "OUROBOROS_MODEL_ACCOUNTS": shown})
    assert unrelated.status_code == 200, unrelated.text
    saved = stored(install)
    assert saved["TOTAL_BUDGET"] == 25
    assert json.loads(saved["OUROBOROS_MODEL_ACCOUNTS"])["safety"] == "mine", "a shadowed raw entry survives the round-trip"
    if managed:
        assert redirect.status_code == 403 and redirect.json()["code"] == "managed_by_deployment", redirect.text
        assert saved["OUROBOROS_CONNECTIONS"][0]["api_key"] == "sk-ant-organization"
    else:
        assert redirect.status_code == 200, redirect.text
        assert saved["OUROBOROS_CONNECTIONS"][0]["api_key"] == "sk-ant-redirected"


# --- the checker's connection, as dispatch reaches it ---------------------------------------------------

COMPATIBLE = "openai-compatible::organization-checker"
LOCAL = "organization-checker (local)"


@pytest.fixture
def booted(monkeypatch):
    """Project the saved settings into this process's environment as a server's boot does (pytest restores it)."""
    def project():
        from ouroboros import config

        env = dict(os.environ)
        config.apply_settings_to_env(config.load_settings(), environ=env)
        for key in set(os.environ) - set(env):
            monkeypatch.delenv(key)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    return project


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("cleared,unmanaged_target", [
    ("OPENAI_COMPATIBLE_BASE_URL", ("https://employee.invalid/v1", "fixture-employee-key")),  # the legacy pair
    ("OPENAI_COMPATIBLE_API_KEY", ("https://organization.invalid/v1", "")),  # a keyless endpoint
])
def test_a_cleared_checker_field_is_judged_as_dispatch_reads_it(install, monkeypatch, booted, managed, cleared, unmanaged_target):
    """Boot mirrors every saved value into the environment, and dispatch never lets that echo stand in for a cleared
    key: neither does the judgement of the save, so a clear cannot move the checker to another endpoint or key."""
    from ouroboros.gateway import settings as settings_gateway
    from ouroboros.llm_routing import resolve_target

    write_document(install, {"TOTAL_BUDGET": 50, "OUROBOROS_MODEL_SAFETY": COMPATIBLE,
                             "OPENAI_COMPATIBLE_BASE_URL": "https://organization.invalid/v1",
                             "OPENAI_COMPATIBLE_API_KEY": "fixture-organization-key",
                             "OPENAI_BASE_URL": "https://employee.invalid/v1", "OPENAI_API_KEY": "fixture-employee-key"})
    if managed:
        manage(install, monkeypatch, {"version": 1, "access": {"max": "pro"},
                                      "safety": {"min_coverage": "full", "checker": {"model": COMPATIBLE}}})
    booted()
    for name in ("_start_supervisor_if_needed_for_request", "_apply_settings_to_env", "_apply_settings_save_side_effects"):
        monkeypatch.setattr(settings_gateway, name, lambda *_a, **_k: None)
    before = install.path.read_bytes()
    with TestClient(_owner_app(install)) as client:
        answer = client.post("/api/settings", json={cleared: ""})
    target = resolve_target(COMPATIBLE)  # the live resolver a send uses
    if managed:
        assert answer.status_code == 403 and answer.json()["code"] == "managed_by_deployment", answer.text
        assert install.path.read_bytes() == before
        assert (target["base_url"], target["api_key"]) == ("https://organization.invalid/v1", "fixture-organization-key")
    else:
        assert answer.status_code == 200, answer.text
        assert stored(install)[cleared] == ""
        assert (target["base_url"], target["api_key"]) == unmanaged_target


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("route,connection,change", [
    ("cloudru::organization-checker", {"CLOUDRU_FOUNDATION_MODELS_API_KEY": "fixture-key"},
     ("CLOUDRU_FOUNDATION_MODELS_BASE_URL", "https://employee.invalid/v1")),
    ("gigachat::organization-checker", {"GIGACHAT_CREDENTIALS": "fixture-credentials"},
     ("GIGACHAT_BASE_URL", "https://employee.invalid/v1")),
    (LOCAL, {"LOCAL_MODEL_SOURCE": "/fixture/organization.gguf"}, ("LOCAL_MODEL_PORT", 8777)),
])
def test_defaulted_checker_fields_read_as_dispatch_defaults_them(install, monkeypatch, managed, route, connection, change):
    """A whole-document writer carries the defaults its read filled in, before any boot projected them: no change of
    the checker. Setting one of those fields to another value is."""
    from ouroboros import config
    from ouroboros.deployment_policy import DeploymentPolicyRefusal
    from ouroboros.gateway.owner_settings import _owner_write_settings

    write_document(install, {"TOTAL_BUDGET": 50, "OUROBOROS_MODEL_SAFETY": route, **connection})
    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": route}}})
    _owner_write_settings({**config.load_settings(), "TOTAL_BUDGET": 25})
    assert stored(install)["TOTAL_BUDGET"] == 25
    before, (key, value) = install.path.read_bytes(), change
    if managed:
        with pytest.raises(DeploymentPolicyRefusal, match="deployment's policy file"):
            _owner_write_settings({**config.load_settings(), key: value})
        assert install.path.read_bytes() == before
    else:
        _owner_write_settings({**config.load_settings(), key: value})
        assert stored(install)[key] == value


@pytest.mark.parametrize("managed", [True, False])
def test_boot_autostart_starts_exactly_the_bound_local_server(install, monkeypatch, managed):
    """Boot's autostart (``server_boot`` hands it the boot's own settings) takes no requested identity: it starts the
    model, template and port the saved settings name, which is the binding ``local_launch_refusal`` admits; so the
    guard stays at the one door that takes a request (``/api/local-model/start``), with no second check here."""
    from ouroboros import config, deployment_policy, local_model
    from ouroboros.local_model_autostart import auto_start_local_model

    write_document(install, {"OUROBOROS_MODEL_SAFETY": LOCAL, "LOCAL_MODEL_SOURCE": "/fixture/organization.gguf",
                             "LOCAL_MODEL_FILENAME": "organization.gguf", "LOCAL_MODEL_PORT": 8766,
                             "LOCAL_MODEL_CHAT_FORMAT": "chatml"})
    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": LOCAL}}})
    starts: list = []
    manager = SimpleNamespace(is_running=False, check_runtime=lambda: True,
                              download_model=lambda source, filename: f"/models/{filename}",
                              start_server=lambda path, **kw: starts.append((path, kw)))
    monkeypatch.setattr(local_model, "get_manager", lambda: manager)
    auto_start_local_model(config.load_settings())
    (path, started), = starts
    assert (path, started["source"], started["filename"], started["port"], started["chat_format"]) == (
        "/models/organization.gguf", "/fixture/organization.gguf", "organization.gguf", 8766, "chatml")
    assert deployment_policy.local_launch_refusal(source=started["source"], filename=started["filename"],
                                                 port=started["port"], chat_format=started["chat_format"]) == ""
    assert (deployment_policy.local_launch_refusal(source="/fixture/employee.gguf", filename="", port=8766,
                                                   chat_format="chatml") != "") is managed


LEGACY_PAIR = {"OPENAI_BASE_URL": "https://organization.invalid/v1", "OPENAI_API_KEY": "fixture-shared-key"}
DEDICATED_PAIR = {"OPENAI_COMPATIBLE_BASE_URL": "https://organization.invalid/v1",
                  "OPENAI_COMPATIBLE_API_KEY": "fixture-organization-key", "OPENAI_API_KEY": "fixture-employee-key"}


def _settings_app(install, monkeypatch):
    """The real Settings GET and POST, with the boot/supervisor side effects of a save left out."""
    from ouroboros.gateway import settings as settings_gateway

    for name in ("_start_supervisor_if_needed_for_request", "_apply_settings_to_env", "_apply_settings_save_side_effects"):
        monkeypatch.setattr(settings_gateway, name, lambda *_a, **_k: None)
    app = _owner_app(install)
    app.router.routes.insert(0, Route("/api/settings", settings_gateway.api_settings_get, methods=["GET"]))
    return app


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("pair", ["legacy", "dedicated"])
def test_settings_fix_exactly_the_fields_a_built_in_checker_reads(install, monkeypatch, managed, pair):
    """A built-in compatible checker reads one pair (``provider_models.compatible_endpoint``). Settings names those
    flat fields and the checker's own built-in row as fixed, and refuses changing them; the whole form saved as
    shown, the employee's Main key (shared with the legacy pair or not) being disabled, or a field the checker does
    not read stays the employee's. Unmanaged, every one of these saves lands."""
    document = {"OUROBOROS_MODEL_SAFETY": COMPATIBLE, "OUROBOROS_MODEL": "openai::employee-main", "TOTAL_BUDGET": 50,
                **(LEGACY_PAIR if pair == "legacy" else DEDICATED_PAIR)}
    write_document(install, document)
    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": COMPATIBLE}}})
    bound = {"legacy": [("OPENAI_API_KEY", "fixture-other-key"), ("OPENAI_BASE_URL", "https://employee.invalid/v1"),
                        ("OPENAI_COMPATIBLE_BASE_URL", "https://employee.invalid/v1")],
             "dedicated": [("OPENAI_COMPATIBLE_API_KEY", "fixture-other-key"), ("OPENAI_COMPATIBLE_BASE_URL", ""),
                           ("OPENAI_COMPATIBLE_BASE_URL", "https://employee.invalid/v1")]}[pair]
    free = {"legacy": {}, "dedicated": {"OPENAI_API_KEY": "fixture-new-employee-key",
                                         "OPENAI_BASE_URL": "https://employee.invalid/v1"}}[pair]
    with TestClient(_settings_app(install, monkeypatch)) as client:
        shown = client.get("/api/settings").json()
        deployment = shown["_meta"]["policy_state"].get("deployment")
        form = {key: value for key, value in shown.items() if key != "_meta"}
        answers = {"form": client.post("/api/settings", json={**form, "TOTAL_BUDGET": 25}),
                   "main_key_off": client.post("/api/settings", json={"OUROBOROS_CONNECTIONS": [
                       {"kind": "openai", "name": "default", "enabled": False}]})}
        answers.update({f"free {key}": client.post("/api/settings", json={key: value}) for key, value in free.items()})
        refused = {f"{key}={value}": client.post("/api/settings", json={key: value}) for key, value in bound}
    assert {name: answer.status_code for name, answer in answers.items()} == dict.fromkeys(answers, 200), {
        name: answer.text for name, answer in answers.items()}
    assert stored(install)["TOTAL_BUDGET"] == 25
    if not managed:
        assert deployment is None and {key: answer.status_code for key, answer in refused.items()} == dict.fromkeys(refused, 200)
        return
    assert deployment["checker"]["connections"] == ["openai-compatible"], "the OpenAI row is not the checker's"
    assert deployment["checker"]["fields"] == (["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL"]
                                               if pair == "legacy" else ["OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"])
    assert {key: answer.status_code for key, answer in refused.items()} == dict.fromkeys(refused, 403), {
        key: answer.text for key, answer in refused.items()}
    kept = stored(install)
    assert {key: kept.get(key) for key, _value in bound} == {key: document.get(key) for key, _value in bound}
    assert {key: kept.get(key) for key in free} == free, "what the checker does not read stays the employee's"


@pytest.mark.parametrize("managed", [True, False])
def test_a_launch_provided_checker_address_is_kept_out_of_the_document_by_a_form_save(install, monkeypatch, managed):
    """The legacy address comes from the deployment's launch, absent from the document: a save carrying it back as
    the form showed it, or blank, keeps it absent (managed) where an unmanaged save writes what it carried; another
    address is refused managed."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://organization.invalid/v1")
    write_document(install, {"OUROBOROS_MODEL_SAFETY": COMPATIBLE, "OPENAI_API_KEY": "fixture-shared-key",
                             "TOTAL_BUDGET": 50})
    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": COMPATIBLE}}})
    with TestClient(_settings_app(install, monkeypatch)) as client:
        shown = client.get("/api/settings").json()
        same = client.post("/api/settings", json={"OPENAI_BASE_URL": shown.get("OPENAI_BASE_URL", ""), "TOTAL_BUDGET": 25})
        same_saved = dict(stored(install))
        blank = client.post("/api/settings", json={"OPENAI_BASE_URL": "", "TOTAL_BUDGET": 30})
        blank_saved = dict(stored(install))
        other = client.post("/api/settings", json={"OPENAI_BASE_URL": "https://employee.invalid/v1"})
    assert (same.status_code, blank.status_code) == (200, 200), (same.text, blank.text)
    assert (same_saved["TOTAL_BUDGET"], blank_saved["TOTAL_BUDGET"]) == (25, 30)
    if managed:
        assert "OPENAI_BASE_URL" not in same_saved and "OPENAI_BASE_URL" not in blank_saved
        assert other.status_code == 403 and "OPENAI_BASE_URL" not in stored(install)
    else:
        assert other.status_code == 200 and stored(install)["OPENAI_BASE_URL"] == "https://employee.invalid/v1"


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("position", ["primary", "fallback"])
def test_only_the_bound_local_server_starts_for_a_local_checker(install, monkeypatch, managed, position):
    """A checker route answering locally binds the server its saved settings name: Start refuses any other artifact,
    template or port before the runtime probe, a download or a start, still starts the bound one; Stop stays free."""
    from ouroboros import local_model
    from ouroboros.gateway.models import api_local_model_start, api_local_model_stop

    bound = {"source": "/fixture/organization.gguf", "port": 8766, "chat_format": "chatml"}
    write_document(install, {"OUROBOROS_MODEL_SAFETY": LOCAL, "LOCAL_MODEL_SOURCE": bound["source"],
                             "LOCAL_MODEL_PORT": bound["port"], "LOCAL_MODEL_CHAT_FORMAT": bound["chat_format"]})
    if managed:
        checker = {"model": LOCAL} if position == "primary" else {"model": CHECKER, "fallbacks": [LOCAL]}
        manage(install, monkeypatch, {"version": 1, "access": {"max": "pro"},
                                      "safety": {"min_coverage": "full", "checker": checker}})
    effects: list = []
    manager = SimpleNamespace(is_running=False, check_runtime=lambda: effects.append("probe") or True,
                              download_model=lambda source, _filename: effects.append(("download", source)) or source,
                              start_server=lambda path, **kw: effects.append(("start", path, kw["port"], kw["chat_format"])),
                              stop_server=lambda: effects.append("stop"))
    monkeypatch.setattr(local_model, "get_manager", lambda: manager)
    app = Starlette(routes=[Route("/api/local-model/start", api_local_model_start, methods=["POST"]),
                            Route("/api/local-model/stop", api_local_model_stop, methods=["POST"])])
    others = [{**bound, "source": "/fixture/employee.gguf"}, {**bound, "port": 8767}, {**bound, "chat_format": "arbitrary"}]
    with TestClient(app) as client:
        assert client.post("/api/local-model/stop").status_code == 200
        answers = [client.post("/api/local-model/start", json=body) for body in others]
        assert client.post("/api/local-model/start", json=bound).status_code == 200
    started = [effect for effect in effects if effect[0] == "start"]
    if managed:
        assert [answer.status_code for answer in answers] == [403] * 3
        assert all(answer.json()["code"] == "managed_by_deployment" for answer in answers)
        assert effects == ["stop", "probe", ("download", bound["source"]), ("start", bound["source"], 8766, "chatml")]
    else:
        assert [answer.status_code for answer in answers] == [200] * 3 and len(started) == 4


# --- the Safety Supervisor ------------------------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_the_safety_walk_follows_only_the_policy_routes(install, monkeypatch, managed):
    from ouroboros import safety

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai@mine::employee-fallback")
    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {
            "model": "claudexor::codex=org-model", "account": "org", "fallbacks": [FALLBACK, "org-local (local)"]}}})
    switched = {"safety": {"model": "openai::switched", "use_local": False, "model_account_override": ""}}
    routes = safety._safety_routes("openai::employee-checker", False, switched)
    if managed:
        assert routes == [("claudexor::codex=org-model", False, "safety", ("claudexor::codex=org-model", False, "org")),
                          (FALLBACK, False, "safety", (FALLBACK, False, "")),
                          ("org-local", True, "safety", ("org-local", True, ""))]
    else:
        assert [(route[0], route[2]) for route in routes] == [
            ("openai::switched", "safety"), ("openai@mine::employee-fallback", "fallback:0")]


@pytest.mark.parametrize("managed", [True, False])
def test_the_managed_supervisor_sends_its_own_account_and_decides_alone(install, monkeypatch, supervisor, managed):
    h = supervisor
    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": "claudexor::codex=org-model",
                                                                            "account": "org"}}})
    h.values["OUROBOROS_SUBAGENTS"] = json.dumps(SECOND_CHECK)
    h.replies.extend([{"status": "DANGEROUS", "reason": "first"}, {"status": "SAFE", "reason": "overturned"}])
    allowed, _message = h.module.check_safety("write_remote", {"x": 1}, ctx=h.ctx)
    if managed:
        assert not allowed and len(h.calls) == 1, "the organization's Supervisor decides alone"
        first = h.calls[0]
        assert (first["light_model"], first["account"], first["processing"]) == ("claudexor::codex=org-model", "org", "")
    else:
        assert allowed and len(h.calls) == 2 and "account" not in h.calls[0]


@pytest.mark.parametrize("managed", [True, False])
def test_an_unavailable_mandatory_checker_holds_only_the_action_that_needs_it(install, monkeypatch, managed):
    from ouroboros import place_guard, safety, tool_effects

    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": CHECKER}}})
    events: list = []
    monkeypatch.setattr(safety, "_emit_durable_safety_event", lambda _ctx, event: events.append(event))
    monkeypatch.setattr(tool_effects, "effect_of", lambda _name, **_kw: tool_effects.BUILTIN)
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *_a, **_kw: {})
    monkeypatch.setattr(safety, "_safety_model_call", lambda **_kw: pytest.fail("no checker connection is configured"))
    reset_safety_stores()
    ctx = SimpleNamespace(task_id="work", task_metadata={"work_kind": "owner"}, drive_root=install.data)
    allowed, message = safety.check_safety("write_remote", {"path": "notes.txt"}, ctx=ctx)
    assert safety.check_safety("read_file", {"path": "notes.txt"}, ctx=ctx) == (True, ""), "reads stay available"
    if managed:
        assert not allowed and message.startswith("⚠️ SAFETY_UNAVAILABLE: not run.") and "no verdict" in message
        assert events[-1]["type"] == "safety_hold"
    else:
        assert allowed and message.startswith("⚠️ SAFETY_NOTE: let through without a Safety reading (")


@pytest.fixture
def walk(install, monkeypatch):
    """The real check path (routing, the walk, repair, holds); only the physical send, the event sink, tool
    effects and place facts are faked. ``answers[model]`` lists what each route returns, in call order."""
    from ouroboros import place_guard, safety, tool_effects

    calls: list = []
    events: list = []
    answers: dict = {}

    def send(**kwargs):
        calls.append(kwargs)
        queue = answers.get(kwargs["light_model"])
        assert queue, f"no request may reach {kwargs['light_model']}"
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return {"content": reply if isinstance(reply, str) else json.dumps(reply)}, {}

    monkeypatch.setattr(safety, "_safety_model_call", send)
    monkeypatch.setattr(safety, "_emit_durable_safety_event", lambda _ctx, event: events.append(event))
    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 0)
    monkeypatch.setattr(safety, "_CHAIN_STORMS", {})
    monkeypatch.setattr(tool_effects, "effect_of", lambda _name, **_kw: tool_effects.BUILTIN)
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *_a, **_kw: {})
    reset_safety_stores()
    ctx = SimpleNamespace(task_id="work", task_metadata={"work_kind": "owner"}, drive_root=install.data)
    yield SimpleNamespace(check=lambda: safety.check_safety("write_remote", {"path": "notes.txt"}, ctx=ctx),
                          calls=calls, events=events, answers=answers)
    reset_safety_stores()


SUBSCRIPTION_FALLBACK = "claudexor::codex=org-fallback"


@pytest.mark.parametrize("fallback", [SUBSCRIPTION_FALLBACK, FALLBACK])
def test_an_approved_fallback_answers_when_the_primary_route_has_no_credential(install, monkeypatch, walk, fallback):
    """The primary's connection has no key (never sent); the approved fallback is asked on its own binding."""
    monkeypatch.setenv("OUROBOROS_CONNECTIONS", json.dumps([CONNECTIONS[1]]))  # the API fallback's own row only
    manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": CHECKER, "fallbacks": [fallback]}}})
    walk.answers[fallback] = [{"status": "SAFE", "reason": "fine"}]
    assert walk.check() == (True, "")
    assert [(call["light_model"], call["model_role"], call["account"], call["processing"]) for call in walk.calls] == [
        (fallback, "safety", "", "")]
    route_event = next(event for event in walk.events if event["type"] == "safety_route_fallback")
    assert [row["model"] for row in route_event["unavailable"]] == [CHECKER]
    assert route_event["answered"] == {"model": fallback, "role": "safety"}


def test_a_fallback_answer_keeps_its_own_binding_through_the_repair(install, monkeypatch, walk):
    """The primary answers nothing; the fallback's unparseable answer is repaired on the fallback's own route,
    account and preference, outside the primary's storm latch."""
    primary, backup = "claudexor::codex=org-model", "claudexor::claude=backup"
    manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": primary, "account": "org",
                                                                        "fallbacks": [backup]}}})
    walk.answers[primary] = [RuntimeError("provider error 500: overloaded")]
    walk.answers[backup] = ["not a verdict", {"status": "SAFE", "reason": "fine"}]
    assert walk.check() == (True, "")
    shape = [(call["light_model"], call["call_type"], call.get("account"), call.get("processing"), call["latch"])
             for call in walk.calls]
    assert shape == [(primary, "safety_supervisor", "org", "", True),
                     (backup, "safety_supervisor", "", "", False),
                     (backup, "safety_supervisor_repair", "", "", False)]
    assert walk.calls[1]["wait_for_resources"] is False, "a route with another after it never waits in its call"


def test_the_owner_question_after_an_unanswered_walk_keeps_the_first_routes_binding(install, monkeypatch, walk):
    """The first route's quota refusal waits for no one inside the walk; when no route answers, its owner
    question (the model wait) comes after the chain, on that route's own account and preference."""
    primary, backup = "claudexor::codex=org-model", "claudexor::claude=backup"
    manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": primary, "account": "org",
                                                                        "fallbacks": [backup]}}})
    quota = RuntimeError("usage limit reached")
    quota.task_resource_wait_reason = "quota"
    walk.answers[primary] = [quota, {"status": "SAFE", "reason": "fine"}]
    walk.answers[backup] = [RuntimeError("provider error 503: unavailable")]
    assert walk.check() == (True, "")
    assert [(call["light_model"], call.get("account"), call.get("processing"), call.get("wait_for_resources"))
            for call in walk.calls] == [(primary, "org", "", False), (backup, "", "", False), (primary, "org", "", None)]


def test_an_unknown_outcome_on_the_primary_holds_without_asking_a_fallback(install, monkeypatch, walk):
    primary = "claudexor::codex=org-model"
    manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": primary,
                                                                        "fallbacks": [SUBSCRIPTION_FALLBACK]}}})
    unknown = RuntimeError("stream cut after dispatch")
    unknown.physical_attempt_capture = SimpleNamespace(state="dispatched")
    walk.answers[primary] = [unknown]
    allowed, message = walk.check()
    assert not allowed and message.startswith("⚠️ SAFETY_UNAVAILABLE")
    assert [call["light_model"] for call in walk.calls] == [primary], "an unknown outcome is never replayed elsewhere"


@pytest.mark.parametrize("fallback_answer", ["definite", "no_credential"])
def test_a_walk_without_any_answer_holds_the_action(install, monkeypatch, walk, fallback_answer):
    """No route answers: the action is held (typed), whether the last route failed or could not be asked."""
    manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": CHECKER, "fallbacks": [
        SUBSCRIPTION_FALLBACK if fallback_answer == "definite" else FALLBACK]}}})
    walk.answers[SUBSCRIPTION_FALLBACK] = [RuntimeError("provider error 503: unavailable")]
    allowed, message = walk.check()
    assert not allowed and message.startswith("⚠️ SAFETY_UNAVAILABLE: not run.") and "no verdict" in message
    assert [call["light_model"] for call in walk.calls] == (
        [SUBSCRIPTION_FALLBACK] if fallback_answer == "definite" else [])
    route_event = next(event for event in walk.events if event["type"] == "safety_route_fallback")
    assert CHECKER in [row["model"] for row in route_event["unavailable"]] and route_event["answered"] is None


@pytest.mark.parametrize("managed", [True, False])
def test_a_wait_card_cannot_switch_the_managed_supervisor(install, monkeypatch, managed):
    from ouroboros import owner_mailbox
    from ouroboros.gateway import task_model_wait as gateway
    from ouroboros.model_wait import mutate_wait
    from ouroboros.task_results import write_task_result
    from supervisor import queue as task_queue

    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": CHECKER}}})
    root, task = install.data, {"id": "task-one", "_attempt": 1, "drive_root": str(install.data)}
    monkeypatch.setattr(gateway, "_live_task", lambda _task_id: dict(task))
    monkeypatch.setattr(task_queue, "DRIVE_ROOT", root)
    write_task_result(root, "task-one", "running")
    mutate_wait(root, "task-one", "wait-one",
                lambda _previous: {"state": "waiting", "revision": 1, "task_attempt": 1, "role": "safety"})
    response = gateway._decide(root, {"request_id": "r1", "decision_id": "model_wait:task-one:wait-one", "revision": 1,
                                      "action": "switch", "model": "openai::switched", "credential_profile_id": "",
                                      "use_local": False})
    controls = owner_mailbox.drain_owner_entries(root, "task-one", kinds={owner_mailbox.KIND_MODEL_WAIT})
    if managed:
        assert response.status_code == 403 and json.loads(response.body)["reason_code"] == "managed_by_deployment"
        assert controls == [], "nothing reached the task"
    else:
        assert response.status_code == 202 and len(controls) == 1


@pytest.mark.parametrize("managed", [True, False])
def test_a_wait_row_from_another_generation_is_served_as_the_door_now_answers(install, monkeypatch, managed):
    """Rows an earlier generation wrote (history, the task's stored rows, Main's list, a carried wait's live event)
    carry ``managed`` exactly when the switch door now refuses that role; no other role ever carries it."""
    from ouroboros.gateway import main_decisions, task_model_wait as gateway
    from ouroboros.gateway.state import _task_activity_facts
    from ouroboros.model_wait import mutate_wait
    from ouroboros.task_results import write_task_result
    from supervisor.task_model_wait import handle_task_model_wait

    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": CHECKER}}})
    base = {"state": "waiting", "revision": 2, "task_attempt": 1, "reason": "quota", "requires_owner_action": True}
    rows = {"before": {**base, "wait_id": "before", "role": "safety"},  # written before the policy
            "stamped": {**base, "wait_id": "stamped", "role": "safety", "managed": True},  # by a managed generation
            "main": {**base, "wait_id": "main", "role": "main"}}
    expected = {"before": True if managed else None, "stamped": True if managed else None, "main": None}

    def marks(served):
        return {key: served[key].get("managed") for key in rows}

    _other, owners, _cut = gateway.history_wait_overlay(
        [{"system_type": "task_model_wait", "task_id": "t", "ts": "1", "model_waits": rows}], 5)
    assert marks(owners[0]["model_waits"]) == expected
    write_task_result(install.data, "t", "running")
    for key, row in rows.items():
        mutate_wait(install.data, "t", key, lambda _previous, row=row: dict(row))
    assert marks(_task_activity_facts(install.data, "t")["display"]["model_waits"]) == expected
    waits, _resume = main_decisions.owner_actions({"phase": "running", "task_attempt": 1, "model_waits": rows})
    assert marks(waits) == expected
    pushed: list = []
    ctx = SimpleNamespace(RUNNING={"t": {"task": {"_attempt": 1, "model_waits": {}}, "attempt": 1}},
                          DRIVE_ROOT=install.data, bridge=SimpleNamespace(push_log=pushed.append))
    for row in rows.values():
        handle_task_model_wait({"type": "task_model_wait", "task_id": "t", "ts": "1", **row}, ctx)
    assert {event["wait_id"]: event.get("managed") for event in pushed} == expected
    assert marks(ctx.RUNNING["t"]["task"]["model_waits"]) == expected


@pytest.mark.parametrize("managed", [True, False])
def test_a_restored_switch_never_moves_the_managed_supervisor(install, monkeypatch, managed):
    from ouroboros import model_wait

    if managed:
        manage(install, monkeypatch, {"version": 1, "safety": {"checker": {"model": CHECKER}}})
    controller = model_wait.TaskModelWait(task={"id": "task-one"}, drive_root=install.data, event_queue=None,
                                          worker_slot_held=False)
    controller.restore_continuation({"overrides": {"safety": {"model": "openai::switched"},
                                                   "main": {"model": "openai::main-switch"}}}, started_at=None)
    assert set(controller.overrides) == ({"main"} if managed else {"main", "safety"})


# --- the organization's rules ---------------------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_organization_rules_reach_every_mind_under_their_own_provenance(install, monkeypatch, managed):
    from ouroboros import place_guard, safety
    from ouroboros.safety_config import owner_rules_section

    monkeypatch.setenv("OUROBOROS_SAFETY_OWNER_RULES", "Ask me before spending money.")
    if managed:
        manage(install, monkeypatch, {"version": 1, "rules": RULES})
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *_a, **_kw: {})
    acting, session = owner_rules_section(), owner_rules_section(session=True)
    prompt = safety._build_check_prompt("write_remote", {}, ctx=SimpleNamespace(task_id="work", task_metadata={}))
    bundle = json.loads(prompt.split("Host evidence (provenance is not consent):\n", 1)[1].split("\nProposed tool call:")[0])
    standing = bundle["standing_delegation"]
    assert standing["owner_rules"] == "Ask me before spending money."
    owner_part = acting.split("## My organization's rules")[0]
    assert "Ask me before spending money." in owner_part and "Your rules" in owner_part and RULES not in owner_part
    if managed:
        organization = acting.split("## My organization's rules", 1)[1]
        assert RULES in organization and "policy file" in organization and "Your rules" not in organization
        delivery = session.split("ORGANIZATION RULES DELIVERY", 1)[1]
        assert RULES in delivery and "not from Settings" in delivery
        assert standing["organization_rules"] == RULES
    else:
        assert "organization" not in acting.lower() and "ORGANIZATION" not in session
        assert "organization_rules" not in standing
