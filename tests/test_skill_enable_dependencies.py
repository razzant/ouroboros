"""One owner readies a skill's packages whenever a runnable skill is requested (sprint 2026-10-19).

Whatever admits the exact bytes (``gate_for(content_hash).executable_review``) admits the
declarations they carry: Enable installs them, a refusal is typed and names its remedy, a slow
owner action answers its running job, the startup pass runs for every skill that should be on,
and the host, not the card, names the removal a skill offers. Every guard asserts both
directions (fires on the broken case, quiet on the working one).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import pathlib
import threading
import time
import urllib.error
from types import SimpleNamespace

import pytest

from ouroboros import skill_lifecycle_actions
from ouroboros.skill_dependencies import dependency_sentence, ensure_skill_dependencies
from ouroboros.skill_install_record import write_install_record
from ouroboros.skill_lifecycle_actions import reconcile_turn_on_when_ready, turn_on_blocker
from ouroboros.skill_loader import SkillReviewState, find_skill, load_enabled, save_enabled, save_review_state, skill_state_dir
from tests._shared import current_python_env

REPO = pathlib.Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

MANIFEST = (
    "---\nname: {name}\ndescription: Needs a package.\nversion: 0.1.0\ntype: script\nruntime: python3\n"
    "timeout_sec: 30\ninstall_specs:\n  - kind: pip\n    package: wheel\n{extra}"
    "scripts:\n  - name: hello.py\n    description: Print hello.\n---\n# body\n"
)
ADMITTING = ["trusted_source", "self_check", "owner_skip", "owner_edit", "cyber_authority", "author_accepted_advisory",
             "ready", "warnings_do_not_block_execution", "blockers_allowed_by_advisory_enforcement"]
REFUSING = ["review_pending", "review_stale", "blocker_findings_need_owner", "blocker_findings_under_blocking_enforcement"]


@pytest.fixture
def drive(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    (drive / "skills" / "external").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    return drive


@pytest.fixture
def installs(monkeypatch):
    calls: list = []
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, env_root, *args, **kwargs: calls.append(list(packages)) or current_python_env(env_root) or [])
    return calls


def make_skill(drive: pathlib.Path, name: str = "demo", *, bucket: str = "external", specs: bool = True, extra: str = "") -> pathlib.Path:
    skill_dir = drive / "skills" / bucket / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    manifest = MANIFEST.format(name=name, extra=extra)
    if not specs:
        manifest = manifest.replace("install_specs:\n  - kind: pip\n    package: wheel\n", "")
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    return skill_dir


def reviewed(drive: pathlib.Path, name: str = "demo", status: str = "clean"):
    loaded = find_skill(drive, name, repo_path="")
    save_review_state(drive, name, SkillReviewState(status=status, content_hash=loaded.content_hash))
    return find_skill(drive, name, repo_path="")


def request_turn_on(drive: pathlib.Path, name: str = "demo", basis: str = "review") -> None:
    write_install_record(drive, name, {"basis": basis, "release": {"digest": "d"}, "source": {"id": "hub"},
                                       "turn_on_when_ready": True})


def enable(drive: pathlib.Path, skill) -> dict:
    return skill_lifecycle_actions._toggle(skill, drive, "", True, "owner_ui", "test")


def offline(*args, **kwargs):
    raise RuntimeError("pip install failed: no network")


# ── the invariant: whatever admits the bytes admits their packages ──

@pytest.mark.parametrize("reason", ADMITTING + REFUSING)
def test_enable_installs_for_every_admitting_gate_reason_and_refuses_before_any_install_otherwise(drive, installs, monkeypatch, reason):
    make_skill(drive)
    skill = find_skill(drive, "demo", repo_path="")
    admits = reason in ADMITTING
    monkeypatch.setattr(skill.review, "gate_for", lambda _hash: {
        "executable_review": admits, "blocking_reason": reason, "review_enforcement": "advisory", "summary": ""})
    monkeypatch.setattr(skill.review, "is_stale_for", lambda _hash: False)

    result = enable(drive, skill)

    if admits:
        assert result["ok"] is True and installs == [["wheel"]] and load_enabled(drive, "demo") is True, result
    else:
        assert result["ok"] is False and result["code"] == "acceptance" and installs == [], result
        assert result["error"] == "It turns on after a review passes." and result["detail"]
        assert not (skill_state_dir(drive, "demo") / "enabled.json").exists()


def test_a_failed_install_is_a_typed_refusal_that_keeps_the_switch_off_and_the_next_enable_retries(drive, installs, monkeypatch):
    make_skill(drive)
    skill = reviewed(drive)
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages", offline)

    failed = enable(drive, skill)

    assert failed["ok"] is False and failed["code"] == "dependencies_failed" and failed["deps_status"] == "failed"
    assert failed["error"] == dependency_sentence("dependencies_failed") and "no network" in failed["detail"]
    assert not (skill_state_dir(drive, "demo") / "enabled.json").exists()
    assert json.loads((skill_state_dir(drive, "demo") / "deps.json").read_text(encoding="utf-8"))["status"] == "failed"
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, env_root, *args, **kwargs: installs.append(list(packages)) or current_python_env(env_root) or [])
    assert enable(drive, skill)["ok"] is True and installs == [["wheel"]] and load_enabled(drive, "demo") is True
    # A ready environment runs no installer: Enable on an enabled skill is an idempotent repair.
    assert enable(drive, skill)["ok"] is True and installs == [["wheel"]]


def test_specs_inapplicable_on_this_host_are_recorded_and_the_skill_turns_on_without_pip(drive, installs):
    make_skill(drive, extra="    platforms: [no-such-os]\n")
    skill = reviewed(drive)

    assert enable(drive, skill)["ok"] is True and installs == [] and load_enabled(drive, "demo") is True
    state = json.loads((skill_state_dir(drive, "demo") / "deps.json").read_text(encoding="utf-8"))
    assert state["status"] == "installed" and state["installed"][0]["status"] == "not_applicable"
    assert ensure_skill_dependencies(drive, find_skill(drive, "demo", repo_path="")) == ("installed", "")


def test_the_seam_does_not_run_an_installer_for_bytes_the_gate_refuses(drive, installs):
    make_skill(drive)
    assert ensure_skill_dependencies(drive, find_skill(drive, "demo", repo_path="")) == ("not_admitted", "")
    assert installs == [] and not (skill_state_dir(drive, "demo") / "deps.json").exists()
    make_skill(drive, "plain", specs=False)
    assert ensure_skill_dependencies(drive, find_skill(drive, "plain", repo_path="")) == ("not_required", "")


# ── turn-on when ready goes through the toggle's own checks ──

def test_turn_on_when_ready_installs_missing_packages_and_maps_a_refusal_to_what_is_missing(drive, installs, monkeypatch):
    make_skill(drive)
    request_turn_on(drive)
    assert turn_on_blocker(drive, "demo") == ""
    assert reconcile_turn_on_when_ready(drive, "demo", repo_path="") == {
        "skill": "demo", "action": "not_ready", "enabled": False, "missing": ["acceptance"]}
    reviewed(drive)
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages", offline)
    assert reconcile_turn_on_when_ready(drive, "demo", repo_path="")["missing"] == ["dependencies"]
    assert load_enabled(drive, "demo") is False
    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages",
                        lambda packages, env_root, *args, **kwargs: installs.append(list(packages)) or current_python_env(env_root) or [])
    assert reconcile_turn_on_when_ready(drive, "demo", repo_path="")["action"] == "enabled"
    assert installs == [["wheel"]] and load_enabled(drive, "demo") is True
    assert turn_on_blocker(drive, "demo") == "already_on"


def test_only_a_load_error_revert_leaves_the_turn_on_request_open(drive):
    make_skill(drive)
    request_turn_on(drive)
    save_enabled(drive, "demo", False, actor="owner_ui")
    assert turn_on_blocker(drive, "demo") == "owner_choice"
    save_enabled(drive, "demo", False, actor="load_error_revert")
    assert turn_on_blocker(drive, "demo") == ""
    make_skill(drive, "plain", specs=False)
    assert turn_on_blocker(drive, "plain") == "not_requested"


def test_a_failure_just_recorded_is_not_re_run_by_the_turn_on_in_the_same_job(drive, monkeypatch):
    from ouroboros import skill_builtin

    make_skill(drive)
    request_turn_on(drive)
    reviewed(drive)
    attempts: list = []

    def failing(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("pip install failed: no network")

    monkeypatch.setattr("ouroboros.marketplace.isolated_deps._install_python_packages", failing)
    assert skill_builtin._dependency_pass(drive, ["demo"], log) == ["demo"]
    assert attempts == [1] and load_enabled(drive, "demo") is False
    # The next pass retries once more (and once only).
    assert skill_builtin._dependency_pass(drive, ["demo"], log) == ["demo"] and attempts == [1, 1]


def test_an_accepted_install_turns_on_only_after_its_packages_succeeded(drive, monkeypatch):
    from ouroboros.skill_source_install import accept_and_turn_on

    make_skill(drive)
    request_turn_on(drive, basis="trusted")
    reviewed(drive)
    turn_ons: list = []
    monkeypatch.setattr("ouroboros.skill_review.run_acceptance_preflight", lambda *a, **k: {"ok": True})
    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.reconcile_turn_on_when_ready",
                        lambda *a, **k: turn_ons.append(1) or {"action": "enabled"})
    monkeypatch.setattr("ouroboros.skill_dependencies.ensure_skill_dependencies", lambda *a, **k: ("failed", "pip exploded"))
    out = accept_and_turn_on(drive, "demo", repo_path="")
    assert out["accepted"] and out["deps_status"] == "failed" and out["deps_error"] == "pip exploded"
    assert out["turn_on"] == "not_ready" and turn_ons == []
    monkeypatch.setattr("ouroboros.skill_dependencies.ensure_skill_dependencies", lambda *a, **k: ("installed", ""))
    assert accept_and_turn_on(drive, "demo", repo_path="")["turn_on"] == "enabled" and turn_ons == [1]


# ── the startup pass (owner decision D2 = A) ──

def test_the_startup_pass_readies_every_skill_that_should_be_on_and_leaves_the_rest_alone(drive, installs):
    from ouroboros import skill_builtin

    make_skill(drive, "on")
    save_enabled(drive, "on", True, actor="owner_ui")
    reviewed(drive, "on")
    make_skill(drive, "requested")
    request_turn_on(drive, "requested")
    reviewed(drive, "requested")
    make_skill(drive, "off")
    save_enabled(drive, "off", False, actor="owner_ui")
    reviewed(drive, "off")
    make_skill(drive, "idle")
    reviewed(drive, "idle")
    make_skill(drive, "unreviewed")
    request_turn_on(drive, "unreviewed")

    assert sorted(skill_builtin._dependency_pass(drive, None, log)) == ["on", "requested"]

    assert installs == [["wheel"], ["wheel"]]
    assert load_enabled(drive, "requested") is True and load_enabled(drive, "off") is False
    assert not (skill_state_dir(drive, "idle") / "deps.json").exists()
    assert not (skill_state_dir(drive, "unreviewed") / "deps.json").exists()
    assert skill_builtin._dependency_pass(drive, None, log) == [] and len(installs) == 2  # ready: nothing to do


def test_startup_inspection_failure_does_not_hide_another_requested_skill(drive, installs, monkeypatch, caplog):
    from ouroboros import skill_builtin, skill_dependencies

    for name in ("broken", "good"):
        make_skill(drive, name)
        request_turn_on(drive, name)
        reviewed(drive, name)
    original = skill_dependencies.auto_install_specs_for_skill

    def inspect(root, skill):
        if skill.name == "broken":
            raise PermissionError("declaration unreadable")
        return original(root, skill)

    monkeypatch.setattr(skill_dependencies, "auto_install_specs_for_skill", inspect)
    with caplog.at_level(logging.WARNING):
        assert skill_builtin._dependency_pass(drive, ["broken", "good"], log) == ["good"]
    assert installs == [["wheel"]] and load_enabled(drive, "good") is True
    assert load_enabled(drive, "broken") is False
    assert "broken" in caplog.text and "could not be inspected" in caplog.text


# ── answer or job: a slow owner action hands back its running job ──

def test_answer_or_job_answers_fast_work_and_hands_back_a_slow_job(monkeypatch):
    from ouroboros import skill_lifecycle_queue as queue

    monkeypatch.setattr(queue, "JOB_ANSWER_WAIT_SEC", 0.2)
    seen: list = []

    def fast(handle):
        return {"ok": True, "enabled": True}

    def slow(handle):
        handle.bind(SimpleNamespace(id="skill-job-1", message="Installing packages…"))
        time.sleep(0.6)
        return {"ok": True, "enabled": True}

    async def run():
        a = await queue.answer_or_job(fast, after=seen.append)
        b = await queue.answer_or_job(slow, after=seen.append, allow_pending=True)
        await asyncio.sleep(0.8)
        return a, b

    a, b = asyncio.run(run())
    assert a == {"ok": True, "enabled": True} and seen[0] is a  # today's bytes; the after-hook ran before answering
    assert b == {"ok": True, "pending": True, "job_id": "skill-job-1", "message": "Installing packages…"}
    assert len(seen) == 2  # the slow job's after-hook (the lifecycle broadcast) ran at its end


def test_an_install_whose_packages_failed_succeeds_and_says_so_in_one_line():
    from ouroboros.gateway.skill_sources import _start_job
    from ouroboros.skill_lifecycle_queue import queue_snapshot

    async def run():
        job_id = await _start_job("install", "demo", "Installing demo", lambda progress: {
            "ok": True, "name": "demo", "acceptance": {"deps_error": "RuntimeError: pip install failed: no network\nmore"}})
        await asyncio.sleep(0.5)
        return job_id

    job_id = asyncio.run(run())
    event = [item for item in queue_snapshot()["events"] if item["id"] == job_id][-1]
    assert event["status"] == "succeeded" and event["error"] == ""
    assert event["message"] == "install demo: done; its packages did not install: RuntimeError: pip install failed: no network"


# ── the author finish waits behind a held lane ──

def test_the_author_finish_package_step_waits_behind_a_held_lane(drive, installs, monkeypatch):
    from ouroboros.skill_lifecycle_queue import _get_lock
    from ouroboros.skill_source_install import run_in_lane

    make_skill(drive)
    skill = reviewed(drive)
    lock = _get_lock()
    lock.acquire()
    released: dict = {}

    def release_later():
        time.sleep(0.4)
        released["at"] = time.monotonic()
        lock.release()

    threading.Thread(target=release_later, daemon=True).start()
    status, _detail = run_in_lane(drive, "deps", "demo", lambda: (released.setdefault("ran", time.monotonic()), ensure_skill_dependencies(drive, skill))[1])
    assert status == "installed" and installs == [["wheel"]]
    assert released["ran"] >= released["at"]


# ── the launcher grant goes through the server ──

def test_the_launcher_grant_goes_through_the_server(tmp_path, monkeypatch):
    import launcher
    from tests._extension_loader_shared import _write_ext_skill

    drive = tmp_path / "data"
    payload = _write_ext_skill(drive / "skills" / "external", "demo", permissions=["inject_chat"],
                               plugin_body="def register(api):\n    return None\n")
    from ouroboros.skill_loader import load_skill

    loaded = load_skill(payload, drive)
    save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=loaded.content_hash))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(launcher, "DATA_DIR", drive)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {})
    monkeypatch.setattr(launcher, "_read_port_file", lambda: 9876)
    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.run_skill_action",
                        lambda *a, **k: pytest.fail("the launcher must not run the grant owner in its own process"))
    requests: list = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout=0):
        requests.append((req.full_url, req.get_method(), json.loads(req.data.decode("utf-8")), timeout))
        return Response(b'{"ok": true, "turn_on_when_ready": "enabled"}')

    monkeypatch.setattr(launcher.urllib.request, "urlopen", urlopen)
    result = launcher._request_skill_key_grant("demo", ["inject_chat"], lambda *a: True, allow_pending=True)
    assert result == {"ok": True, "turn_on_when_ready": "enabled"}
    assert requests == [("http://127.0.0.1:9876/api/skills/demo/grants", "POST",
                         {"items": ["inject_chat"], "expected_content_hash": loaded.content_hash}, 20)]
    assert launcher._request_skill_key_grant("demo", ["inject_chat"], lambda *a: False) == {"ok": False, "error": "Skill grant cancelled."}
    assert len(requests) == 1  # a cancelled confirm sends nothing

    prompts = []
    assert launcher._request_skill_key_grant("demo", ["inject_chat"],
        lambda title, message: prompts.append(message) or True, enable_after_grant=True)["ok"]
    assert "Then turn this skill on." in prompts[0]
    assert requests[-1][2]["enable_after_grant"] is True

    def refused(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 409, "Conflict", None, io.BytesIO(b'{"ok": false, "error": "It turns on after a review passes.", "code": "acceptance"}'))

    monkeypatch.setattr(launcher.urllib.request, "urlopen", refused)
    assert launcher._request_skill_key_grant("demo", ["inject_chat"], lambda *a: True)["code"] == "acceptance"


# ── removal: the host names what each remover accepts ──

def test_removal_names_what_each_remover_accepts():
    from ouroboros.skill_uninstall_state import removal_action

    def loaded(location, source, collision=False):
        return SimpleNamespace(location=location, source=source, identity_collision=collision)

    assert removal_action(loaded("ouroboroshub", "ouroboroshub"), {"basis": "trusted"}) == "uninstall"
    assert removal_action(loaded("clawhub", "clawhub"), {"basis": "review"}) == "uninstall"
    assert removal_action(loaded("clawhub", "clawhub"), None) == "uninstall"
    assert removal_action(loaded("native", "native"), {"basis": "trusted"}) == "uninstall"
    assert removal_action(loaded("native", "native"), None) == "uninstall"  # an orphaned built-in copy
    assert removal_action(loaded("external", "external"), None) == "delete"
    assert removal_action(loaded("external", "self_authored"), None) == "delete"
    assert removal_action(loaded("user_repo", "external"), None) == ""
    assert removal_action(loaded("external", "external", collision=True), None) == ""
    assert removal_action(loaded("external", "clawhub"), None) == ""  # a sourced payload that lost its record
    assert removal_action(None, None) == ""


def test_both_removers_admit_by_the_listing_word(drive, monkeypatch):
    from ouroboros.skill_source_catalog import install_listing
    from ouroboros.skill_source_install import SkillInstallError, uninstall_installed
    from ouroboros.skill_uninstall_state import delete_local_skill

    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    made = make_skill(drive, "made")
    assert install_listing(drive, find_skill(drive, "made", repo_path=""), {})["removal"] == "delete"
    with pytest.raises(SkillInstallError) as refusal:
        uninstall_installed(drive, "made")
    assert refusal.value.code == "not_from_source"
    assert delete_local_skill(drive, find_skill(drive, "made", repo_path=""), repo_path="")["ok"] is True and not made.exists()

    (drive / "skills" / "native").mkdir()
    orphan = make_skill(drive, "orphan", bucket="native", specs=False)
    loaded = find_skill(drive, "orphan", repo_path="")
    assert install_listing(drive, loaded, {})["removal"] == "uninstall"
    assert delete_local_skill(drive, loaded, repo_path="")["ok"] is False
    assert uninstall_installed(drive, "orphan")["ok"] is True and not orphan.exists()
    assert (skill_state_dir(drive, "orphan") / "uninstalled.json").is_file()


def test_a_removed_built_in_that_later_turns_on_by_default_is_not_installed_again(tmp_path, monkeypatch):
    from ouroboros import skill_builtin
    from ouroboros.skill_starter import _write_marker, read_marker
    from ouroboros.skill_uninstall_state import write_uninstall_tombstone

    seeds = REPO / "skills"
    names = skill_builtin.default_on_names(seeds)
    if not names:
        pytest.skip("no default-on built-in applies on this host")
    name = names[0]
    drive = tmp_path / "data"
    (drive / "skills" / "native").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    _write_marker(drive, state="done")
    write_uninstall_tombstone(drive, name, source="builtin")

    skill_builtin._offer_default_builtins(drive, seeds, log, [])

    assert not (drive / "skills" / "native" / name).exists()
    assert name in read_marker(drive)["defaults_offered"]
    # Never removed: the same offer installs it (the guard fires only on the tombstone).
    other = tmp_path / "other"
    (other / "skills" / "native").mkdir(parents=True)
    _write_marker(other, state="done")
    skill_builtin._offer_default_builtins(other, seeds, log, [])
    assert (other / "skills" / "native" / name).is_dir()


# ── one sentence per dependency state, everywhere it is shown ──

def test_liveness_and_the_loader_show_the_dependency_sentence_and_keep_the_code(drive, monkeypatch):
    from ouroboros.extension_liveness import _apply_deps_block
    from tests._extension_loader_shared import _write_ext_skill
    from ouroboros.skill_loader import load_skill

    payload = _write_ext_skill(drive / "skills" / "external", "ext", permissions=["tool"],
                               plugin_body="def register(api):\n    return None\n",
                               extra_frontmatter="install_specs:\n  - kind: pip\n    package: wheel\n")
    skill = load_skill(payload, drive)
    state = _apply_deps_block({"desired_live": True, "reason": "ready", "load_error": None}, drive, skill)
    assert state["reason"] == "deps_missing" and state["load_error"] == "Its packages are not installed."
    assert state["desired_live"] is False
    untouched = _apply_deps_block({"desired_live": False, "reason": "disabled", "load_error": None}, drive, skill)
    assert untouched == {"desired_live": False, "reason": "disabled", "load_error": None}
    assert dependency_sentence("deps_declaration_desync") == "Its package list does not match its files."
    assert dependency_sentence("deps_failed") == dependency_sentence("dependencies_failed") == "Its packages did not install."


def test_agent_facing_texts_send_packages_to_the_enable_owner_never_to_a_paid_review(drive):
    from ouroboros.skill_readiness import _DEPENDENCY_ACTION, skill_readiness_for_execution
    from ouroboros.tools.skill_exec import _TOGGLE_SCHEMA, _skill_deps_exec_block

    make_skill(drive)
    skill = reviewed(drive)
    readiness = skill_readiness_for_execution(drive, skill, require_enabled=False, skills=[skill])
    assert "deps_not_ready:pending" in readiness.blockers
    assert [action for action in readiness.next_actions if action["phase"] == "dependencies"] == [_DEPENDENCY_ACTION]
    assert _DEPENDENCY_ACTION["tool"] == "toggle_skill" and "skill_review" not in _DEPENDENCY_ACTION["reason"]
    block = _skill_deps_exec_block(drive, skill)
    assert "toggle_skill enabled=true installs them" in block and "skill_review" not in block
    assert "enabling installs or refreshes the declared isolated dependencies" in _TOGGLE_SCHEMA["description"]
    assert "accepted without a review" not in _TOGGLE_SCHEMA["description"]
