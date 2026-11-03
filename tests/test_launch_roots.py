"""One APP_ROOT supplies every default; established layouts keep theirs (``launch_roots``, #1239).

The launch tuple is derived once for config, the direct server, the source CLI and the packaged
CLI. Work roots without a saved or environment value follow the data root's basis, carried by the
initialization witness: ``home`` (the historical ``~/Ouroboros/<leaf>``) for an install initialized
before the rule, ``app_root`` for a new one, ``unknown`` (refuse, never guess) when the witness
cannot be read. A direct server given only APP_ROOT refuses while ``~/Ouroboros/data`` may hold
memory. Consumers keep what they classify: GC never forgets a recorded copy it cannot remove, the
safe Deliverables container is reachable wherever APP_ROOT put it, and children inherit the basis.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from ouroboros import launch_roots
from supervisor import state_initialization as witness

REPO = pathlib.Path(__file__).resolve().parents[1]
LEAVES = (("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "subagent_worktrees"),
          ("OUROBOROS_SUBAGENT_PROJECTS_ROOT", "projects"), ("OUROBOROS_DELIVERABLES_ROOT", "Deliverables"))
PATH_KEYS = ("OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH",
             "OUROBOROS_PID_FILE", "OUROBOROS_PORT_FILE", "OUROBOROS_USER_FILES_ROOT", launch_roots.BASIS_ENV,
             *(key for key, _leaf in LEAVES))


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A synthetic HOME (so ``~/Ouroboros`` is H) and a separate lab APP_ROOT A, no path overrides."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for key in PATH_KEYS:
        monkeypatch.delenv(key, raising=False)
    return SimpleNamespace(home=home, historical=home / "Ouroboros", app=tmp_path / "lab")


def _write_witness(root, **fields):
    path = launch_roots.witness_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "initialization_id": "init-1", "phase": "complete",
                                "origin": "first_boot", **fields}), encoding="utf-8")


def _corrupt_primary(root, backup):
    """A primary state the owner cannot parse beside the backup it recovers from."""
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state" / "state.json").write_text("{", encoding="utf-8")
    (root / "state" / "state.last_good.json").write_text(json.dumps(backup), encoding="utf-8")


def _roots():
    from ouroboros import config

    return {leaf: getattr(config, getter)() for leaf, getter in (
        ("subagent_worktrees", "get_subagent_worktree_root"), ("projects", "get_subagent_projects_root"),
        ("Deliverables", "get_deliverables_root"))}


def _use_lab(monkeypatch, layout, data=None):
    data = data or layout.app / "data"
    monkeypatch.setenv("OUROBOROS_APP_ROOT", str(layout.app))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    return data


# --------------------------------------------------------------------------- unchanged default install

def test_an_unconfigured_install_keeps_every_default_under_home_ouroboros(layout):
    h = layout.historical
    assert launch_roots.configured_roots() == (h, h / "repo", h / "data", h / "data" / "settings.json",
                                               h / "ouroboros.pid", h / "data" / "state" / "server_port")
    expected = {leaf: os.path.expanduser(os.path.join("~", "Ouroboros", leaf)) for _key, leaf in LEAVES}
    assert launch_roots.durable_basis(h / "data")[0] == launch_roots.APP_ROOT_BASIS  # a new root ...
    assert _roots() == expected  # ... whose APP_ROOT is ~/Ouroboros: the same folders, same spelling
    _write_witness(h / "data")  # an install initialized before the rule
    assert launch_roots.durable_basis(h / "data") == (launch_roots.HOME_BASIS, "witness")
    assert _roots() == expected


# --------------------------------------------------------------------------- precedence

def test_work_roots_keep_saved_then_environment_then_basis_precedence(layout, monkeypatch, tmp_path):
    from ouroboros import config

    data = _use_lab(monkeypatch, layout)
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    assert config.get_subagent_projects_root() == str(layout.app / "projects"), "a new root: APP_ROOT's"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "env-projects"))
    assert config.get_subagent_projects_root() == str(tmp_path / "env-projects")
    settings.write_text(json.dumps({"OUROBOROS_SUBAGENT_PROJECTS_ROOT": str(tmp_path / "saved")}), encoding="utf-8")
    config.apply_settings_to_env(config.load_settings())
    assert config.get_subagent_projects_root() == str(tmp_path / "saved"), "a nonempty saved value wins"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "env-projects"))
    settings.write_text(json.dumps({"OUROBOROS_SUBAGENT_PROJECTS_ROOT": ""}), encoding="utf-8")
    config.apply_settings_to_env(config.load_settings())
    assert config.get_subagent_projects_root() == str(tmp_path / "env-projects"), "a blank saved value allows env"
    monkeypatch.delenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT")
    before = settings.read_bytes()
    assert config.get_subagent_projects_root() == str(layout.app / "projects")
    assert settings.read_bytes() == before and not (data / "state").exists(), "a getter pins and writes nothing"


# --------------------------------------------------------------------------- the basis and its witness

@pytest.mark.parametrize("prepare, expected", [
    (lambda root: None, launch_roots.APP_ROOT_BASIS),
    (lambda root: _write_witness(root), launch_roots.HOME_BASIS),
    (lambda root: _write_witness(root, work_root_basis="app_root"), launch_roots.APP_ROOT_BASIS),
    (lambda root: _write_witness(root, work_root_basis="home"), launch_roots.HOME_BASIS),
    (lambda root: _write_witness(root, work_root_basis="elsewhere"), launch_roots.UNKNOWN_BASIS),
    (lambda root: (root / "state").mkdir(parents=True) or launch_roots.witness_path(root).write_text("{broken"),
     launch_roots.UNKNOWN_BASIS),
    (lambda root: launch_roots.witness_path(root).mkdir(parents=True), launch_roots.UNKNOWN_BASIS),
    (lambda root: (root / "state").mkdir(parents=True) or (root / "state" / "state.json").write_text(
        json.dumps({"session_id": "s", "created_at": "t"})), launch_roots.HOME_BASIS),
    (lambda root: (root / "state").mkdir(parents=True) or (root / "state" / "state.json").write_text(
        json.dumps({"initialization_id": "lost"})), launch_roots.UNKNOWN_BASIS),
    (lambda root: (root / "state").mkdir(parents=True) or (root / "state" / "state.json").write_text("{"),
     launch_roots.UNKNOWN_BASIS),
    (lambda root: _corrupt_primary(root, {"session_id": "s", "created_at": "t"}), launch_roots.HOME_BASIS),
    (lambda root: _corrupt_primary(root, {"initialization_id": "lost"}), launch_roots.UNKNOWN_BASIS),
    (lambda root: (root / "task_results").mkdir(parents=True) or (root / "task_results" / "t.json").write_text("{}"),
     launch_roots.UNKNOWN_BASIS),
], ids=["new", "legacy-witness", "recorded-app-root", "recorded-home", "unrecognized", "invalid-witness",
        "unreadable-witness", "pre-witness-state", "state-lost-its-witness", "unreadable-state",
        "corrupt-primary-pre-witness-backup", "corrupt-primary-backup-lost-its-witness", "history-without-state"])
def test_the_witness_decides_the_basis_and_never_guesses_home(layout, monkeypatch, prepare, expected):
    data = _use_lab(monkeypatch, layout)
    data.mkdir(parents=True)
    prepare(data)
    assert launch_roots.durable_basis(data)[0] == expected
    if expected == launch_roots.UNKNOWN_BASIS:
        with pytest.raises(launch_roots.WorkRootUnavailable) as refused:
            _roots()
        message = str(refused.value)
        assert str(layout.historical / "subagent_worktrees") in message and str(layout.app / "subagent_worktrees") in message
        assert "OUROBOROS_SUBAGENT_WORKTREE_ROOT" in message
        monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(layout.app / "explicit"))
        from ouroboros import config
        assert config.get_subagent_worktree_root() == str(layout.app / "explicit"), "an explicit root still works"
        monkeypatch.delenv("OUROBOROS_APP_ROOT")  # the two candidates coincide: nothing to decide
        assert config.get_subagent_projects_root() == str(layout.historical / "projects")
    else:
        parent = layout.app if expected == launch_roots.APP_ROOT_BASIS else layout.historical
        assert _roots() == {leaf: str(parent / leaf) for _key, leaf in LEAVES}


def test_an_empty_dialogue_store_is_scaffolding_only_to_its_own_reader(layout, monkeypatch):
    """The cutover creates the store before init: the server's full reader sees no chat rows and
    keeps a new root new, while the stdlib fallback, which cannot count rows, stays undecided."""
    from ouroboros import dialogue_store

    assert launch_roots.DIALOGUE_STORE_REL == dialogue_store.STORE_REL, "one name for the store's evidence"
    data = _use_lab(monkeypatch, layout)
    dialogue_store.create(data)
    assert witness.supervisor_evidence(data) == ("none", "")
    assert launch_roots.durable_basis(data, evidence=witness.supervisor_evidence)[0] == launch_roots.APP_ROOT_BASIS
    assert launch_roots.durable_basis(data)[0] == launch_roots.UNKNOWN_BASIS


# --------------------------------------------------------------------------- first boot, interruption, reset

def test_first_boot_records_app_root_and_keeps_it_through_interruption_completion_and_reset(layout, monkeypatch):
    from ouroboros import server_entrypoint
    from ouroboros.gateway import control

    data = _use_lab(monkeypatch, layout)
    monkeypatch.setenv(launch_roots.BASIS_ENV, launch_roots.HOME_BASIS)  # an inherited value names no root
    assert server_entrypoint.pin_work_root_basis(data) == launch_roots.APP_ROOT_BASIS
    assert os.environ[launch_roots.BASIS_ENV] == launch_roots.APP_ROOT_BASIS
    first = witness.initialization_decision(data)
    status, pending = witness.read_witness(data)
    assert first["create"] and status == "ok" and pending["phase"] == "pending"
    assert pending[launch_roots.BASIS_KEY] == launch_roots.APP_ROOT_BASIS
    again = witness.initialization_decision(data)  # the boot was interrupted before the state was written
    assert again["initialization_id"] == first["initialization_id"]
    assert witness.complete(data, first["initialization_id"], adopted=False)
    assert witness.read_witness(data)[1][launch_roots.BASIS_KEY] == launch_roots.APP_ROOT_BASIS
    monkeypatch.delenv(launch_roots.BASIS_ENV)
    assert launch_roots.durable_basis(data) == (launch_roots.APP_ROOT_BASIS, "witness"), "the second boot"

    # Owner Reset wipes state/ and writes a pending witness; the work folders it keeps keep their basis.
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    monkeypatch.setattr(control, "_request_restart", lambda _request: True)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=data)))
    response = asyncio.run(control.api_reset(request))
    assert response.status_code == 200
    status, reset = witness.read_witness(data)
    assert reset["origin"] == "owner_reset" and reset[launch_roots.BASIS_KEY] == launch_roots.APP_ROOT_BASIS
    assert witness.initialization_decision(data)["initialization_id"] == reset["initialization_id"]


@pytest.mark.parametrize("prepare, expected", [
    (lambda root: _write_witness(root), launch_roots.HOME_BASIS),
    (lambda root: (root / "state").mkdir(parents=True) or launch_roots.witness_path(root).write_text("{"),
     launch_roots.UNKNOWN_BASIS),
], ids=["legacy", "unreadable"])
def test_reset_carries_a_legacy_or_undecided_basis_instead_of_switching_it(layout, monkeypatch, prepare, expected):
    from ouroboros.gateway import control

    data = _use_lab(monkeypatch, layout)
    prepare(data)
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    monkeypatch.setattr(control, "_request_restart", lambda _request: True)
    asyncio.run(control.api_reset(SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=data)))))
    assert witness.read_witness(data)[1][launch_roots.BASIS_KEY] == expected


def test_legacy_adoption_keeps_home_and_a_lost_witness_stays_undecided(layout, monkeypatch):
    data = _use_lab(monkeypatch, layout)
    state = data / "state" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"session_id": "s", "created_at": "t"}), encoding="utf-8")
    witness.prepare_adoption(data, {"session_id": "s", "created_at": "t"})
    assert witness.read_witness(data)[1][launch_roots.BASIS_KEY] == launch_roots.HOME_BASIS
    launch_roots.witness_path(data).unlink()
    state.write_text(json.dumps({"initialization_id": "kept", "session_id": "s", "created_at": "t"}), encoding="utf-8")
    assert witness.prepare_adoption(data, {"initialization_id": "kept"}) == "kept"
    assert witness.read_witness(data)[1][launch_roots.BASIS_KEY] == launch_roots.UNKNOWN_BASIS


@pytest.mark.parametrize("primary", ["readable", "corrupt"])
def test_the_boot_exports_the_basis_its_state_initialization_settled(layout, monkeypatch, primary):
    """Negative control on the staged tree (Astra R1): admission pinned ``unknown`` beside a corrupt
    primary, recovery from the backup wrote a ``home`` witness, and the pin stayed ``unknown``."""
    from ouroboros import config, server_entrypoint
    from supervisor import state

    data = _use_lab(monkeypatch, layout)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    prior = {"created_at": "2026-07-30T00:00:00+00:00", "session_id": "old-session", "owner_id": 7}
    _corrupt_primary(data, prior)
    if primary == "readable":
        (data / "state" / "state.json").write_text(json.dumps(prior), encoding="utf-8")
    (layout.historical / "projects" / "kept").mkdir(parents=True)
    monkeypatch.setattr(state, "_openrouter_ledger_settled", lambda: None)
    monkeypatch.setattr(state._OpenRouterDiagnostic, "start", lambda *_args: None)
    old_drive = state.DRIVE_ROOT
    try:
        # Admission reads the copy the state owner adopts, the backup beside a corrupt primary.
        assert server_entrypoint.pin_work_root_basis(data) == launch_roots.HOME_BASIS
        state.init(data)
        monkeypatch.setenv(launch_roots.BASIS_ENV, launch_roots.UNKNOWN_BASIS)  # a pin initialization outdated
        outcome = state.init_state()
        assert outcome.quality in ("current", "recovered") and outcome.values["session_id"] == "old-session"
        assert witness.read_witness(data)[1][launch_roots.BASIS_KEY] == launch_roots.HOME_BASIS
        assert os.environ[launch_roots.BASIS_ENV] == launch_roots.HOME_BASIS, "workers inherit the settled basis"
        assert _roots() == {leaf: str(layout.historical / leaf) for _key, leaf in LEAVES}
    finally:
        state.init(old_drive)


# --------------------------------------------------------------------------- the direct entry

def _child_env(layout, **extra):
    env = {key: value for key, value in os.environ.items() if key not in PATH_KEYS}
    env.update(HOME=str(layout.home), USERPROFILE=str(layout.home), PYTHONPATH=str(REPO), **extra)
    return env


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in pathlib.Path(root).rglob("*")) if pathlib.Path(root).exists() else []


# The suite's child-env guard re-adds its scratch DATA_DIR/SETTINGS_PATH to every child; the probe
# drops them first so the real entry sees an operator's "APP_ROOT only" environment (HOME is synthetic).
_SERVER_PROBE = r'''
import json, os
for key in ("OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH"):
    os.environ.pop(key, None)
import server
from ouroboros import config, server_process
print(json.dumps({"server": str(server.DATA_DIR), "process": str(server_process.DATA_DIR),
                  "config": str(config.DATA_DIR), "env": os.environ.get("OUROBOROS_DATA_DIR"),
                  "settings": str(config.SETTINGS_PATH), "repo": str(server.REPO_DIR)}))
'''


@pytest.mark.serial
def test_a_direct_server_given_only_app_root_refuses_while_home_holds_history(layout, tmp_path):
    historical = layout.historical / "data"
    _write_witness(historical)
    before = _tree(historical)
    run = subprocess.run([sys.executable, "-c", _SERVER_PROBE], cwd=str(tmp_path), capture_output=True, text=True,
                         timeout=180, env=_child_env(layout, OUROBOROS_APP_ROOT=str(layout.app)))
    assert run.returncode == 2, run.stderr[-2000:]
    for choice in (f"OUROBOROS_DATA_DIR={historical}", f"OUROBOROS_SETTINGS_PATH={layout.app / 'data' / 'settings.json'}",
                   f"OUROBOROS_DATA_DIR={layout.app / 'data'}"):
        assert choice in run.stderr
    assert not layout.app.exists() and _tree(historical) == before, "nothing opened or written"


@pytest.mark.serial
def test_a_direct_server_given_only_app_root_uses_its_own_data_when_home_has_none(layout, tmp_path):
    (layout.historical / "data" / "state").mkdir(parents=True)  # bootstrap scaffolding is not memory
    (layout.historical / "data" / "settings.json").write_text("{}", encoding="utf-8")
    run = subprocess.run([sys.executable, "-c", _SERVER_PROBE], cwd=str(tmp_path), capture_output=True, text=True,
                         timeout=180, env=_child_env(layout, OUROBOROS_APP_ROOT=str(layout.app)))
    assert run.returncode == 0, run.stderr[-2000:]
    facts = json.loads(run.stdout.strip().splitlines()[-1])
    own = str(layout.app / "data")
    assert facts["server"] == facts["process"] == facts["config"] == facts["env"] == own
    assert facts["settings"] == str(layout.app / "data" / "settings.json") and facts["repo"] == str(REPO)
    assert not (layout.app / "data" / "state" / "server.lock").exists(), "importing the server admits nothing"


# --------------------------------------------------------------------------- a first session that ends before Setup

def _serve_and_quit(layout, data, label, logs):
    """One real server generation on the lab root, started with the tuple the desktop app passes and the
    launcher's private stop pipe, left to finish its boot (providerless readiness or the supervisor's),
    then stopped the way Quit stops it."""
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.runtime_limits import LAUNCHER_STOP_GRACE_SEC

    ports = []
    for _ in range(2):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            ports.append(probe.getsockname()[1])
    env = _child_env(layout, OUROBOROS_APP_ROOT=str(layout.app), OUROBOROS_REPO_DIR=str(REPO),
                     OUROBOROS_DATA_DIR=str(data), OUROBOROS_SETTINGS_PATH=str(data / "settings.json"),
                     OUROBOROS_SERVER_HOST="127.0.0.1", OUROBOROS_SERVER_PORT=str(ports[0]),
                     OUROBOROS_HOST_SERVICE_PORT=str(ports[1]), OUROBOROS_LAUNCHER_STOP_STDIN="1")
    port_file, stderr_path = data / "state" / "server_port", logs / f"{label}.stderr"
    port_file.unlink(missing_ok=True)
    container = ProcessContainer()
    with open(stderr_path, "wb") as stderr:
        proc = container.spawn([sys.executable, "server.py"], cwd=str(REPO), env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=stderr)
    try:
        deadline, last = time.monotonic() + 120, ""
        while proc.poll() is None and time.monotonic() < deadline:
            try:
                port = int(port_file.read_text(encoding="utf-8"))
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("supervisor_ready") is True or payload.get("supervisor_error"):
                    break
            except (OSError, ValueError) as exc:
                last = str(exc)
            time.sleep(0.25)
        else:
            pytest.fail(f"{label}: the boot never finished ({last}), exit {proc.poll()}")
        proc.stdin.write(b"quit\n")
        proc.stdin.flush()
        proc.stdin.close()
        code = proc.wait(timeout=LAUNCHER_STOP_GRACE_SEC + 20)
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            error = container.reap()
        finally:
            container.close()
        proc.wait(timeout=10)
    assert (code, error) == (0, ""), stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]


def _initialization_facts(data):
    """What the next state initialization and every default work root decide from."""
    status, record = witness.read_witness(data)
    events = data / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if events.exists() else []
    return {"witness": [status, record.get("phase"), record.get("origin"), record.get(launch_roots.BASIS_KEY)],
            "state": (data / "state" / "state.json").is_file(),
            "history": list(witness.supervisor_evidence(data)),
            "basis": list(launch_roots.durable_basis(data, evidence=witness.supervisor_evidence)),
            "refused": [row.get("reason") for row in rows if row.get("type") == "state_unavailable_at_boot"]}


@pytest.mark.serial
def test_a_first_session_that_ends_before_setup_leaves_the_root_new(layout, tmp_path):
    """A new root's first start has no provider, so no supervisor runs, and the owner (or a Homebrew
    upgrade) quits before Setup: the configured boot after it initializes the root with the ``app_root``
    basis, a later boot without a provider rewrites none of the history a supervisor wrote, and that
    history still refuses a lost state (#2412). Negative control on the staged tree (native qualification
    7c080838, released 7.8.0 alike): the providerless shutdown published an empty queue snapshot, the
    basis became unknown and the configured boot refused ``prior_history_without_state`` while serving."""
    data, logs = layout.app / "data", tmp_path / "server-logs"
    logs.mkdir()
    snapshot = data / "state" / "queue_snapshot.json"
    _serve_and_quit(layout, data, "providerless", logs)
    assert not (data / "settings.json").exists(), "a boot without Setup creates no settings"
    first = _initialization_facts(data)
    model = "openai-compatible::never-called"
    (data / "settings.json").write_text(json.dumps({
        "OPENAI_COMPATIBLE_API_KEY": "first-session-test-key",
        "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9/v1",  # the discard port: nothing answers it
        "OUROBOROS_MODEL": model, "OUROBOROS_MODEL_LIGHT": model, "OUROBOROS_MODEL_FALLBACKS": model,
        "OUROBOROS_MAX_WORKERS": 1, "OUROBOROS_RUNTIME_MODE": "light",
        "OUROBOROS_STARTER_SKILLS": "[]",  # no skill-source clone from GitHub on a new root's first boot
    }), encoding="utf-8")
    _serve_and_quit(layout, data, "configured", logs)
    second = _initialization_facts(data)
    kept = snapshot.read_bytes()
    (data / "settings.json").write_text("{}", encoding="utf-8")  # the provider is removed again
    _serve_and_quit(layout, data, "providerless-again", logs)
    third = {**_initialization_facts(data), "snapshot_kept": snapshot.read_bytes() == kept}
    recorded = str(pathlib.Path("state") / "queue_snapshot.json")
    initialized = {"witness": ["ok", "complete", "first_boot", launch_roots.APP_ROOT_BASIS], "state": True,
                   "history": ["present", recorded], "basis": [launch_roots.APP_ROOT_BASIS, "witness"], "refused": []}
    assert (first, second, third) == (
        {"witness": ["missing", None, None, None], "state": False, "history": ["none", ""],
         "basis": [launch_roots.APP_ROOT_BASIS, "a new data root"], "refused": []},
        initialized, {**initialized, "snapshot_kept": True},
    ), json.dumps({"after_providerless": first, "after_configured": second, "after_providerless_again": third})

    for name in ("state.json", "state.last_good.json"):
        (data / "state" / name).unlink(missing_ok=True)
    assert witness.initialization_decision(data)["reason"] == "initialized_state_lost"
    launch_roots.witness_path(data).unlink()
    assert witness.initialization_decision(data) == {
        "create": False, "reason": "prior_history_without_state", "detail": recorded}
    assert not launch_roots.witness_path(data).exists(), "a refusal mints no identity"


@pytest.mark.parametrize("prepare", [
    lambda h: (h / "state").mkdir(parents=True) or (h / "state" / "dialog.sqlite").write_bytes(b""),
    lambda h: h.mkdir(parents=True) or (h / "state").write_text("not a directory"),
    lambda h: (h / "logs").mkdir(parents=True) or (h / "logs" / "chat.jsonl").write_text('{"x":1}\n'),
], ids=["any-dialogue-store", "unexaminable", "older-live-chat"])
def test_historical_memory_is_probed_conservatively_and_by_name_only(layout, prepare):
    prepare(layout.historical / "data")
    with pytest.raises(launch_roots.AmbiguousDataRoot):
        launch_roots.direct_data_dir({"OUROBOROS_APP_ROOT": str(layout.app)})


def test_explicit_values_stay_independent_and_a_new_launch_may_choose_its_own_data(layout, monkeypatch, tmp_path):
    historical = layout.historical / "data"
    _write_witness(historical)
    old_split = {"OUROBOROS_APP_ROOT": str(layout.app), "OUROBOROS_DATA_DIR": str(historical),
                 "OUROBOROS_SETTINGS_PATH": str(layout.app / "data" / "settings.json"),
                 "OUROBOROS_REPO_DIR": str(tmp_path / "checkout")}
    assert launch_roots.direct_data_dir(old_split) == historical
    assert launch_roots.configured_roots(old_split) == (
        layout.app, tmp_path / "checkout", historical, layout.app / "data" / "settings.json",
        layout.app / "ouroboros.pid", historical / "state" / "server_port")
    chosen = {"OUROBOROS_APP_ROOT": str(layout.app), "OUROBOROS_DATA_DIR": str(layout.app / "data")}
    assert launch_roots.direct_data_dir(chosen) == layout.app / "data", "beside a legacy root, by explicit choice"
    assert launch_roots.direct_entry_environment(tmp_path / "exec", chosen) == {
        "OUROBOROS_REPO_DIR": str(tmp_path / "exec")}, "the executing checkout is added; explicit data is kept"


def _case_insensitive(directory: pathlib.Path) -> bool:
    probe = directory / "CaseProbe"
    probe.mkdir()
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.rmdir()


def test_one_place_under_two_spellings_is_one_root(layout, tmp_path):
    historical = layout.historical / "data"
    _write_witness(historical)
    alias = tmp_path / "alias"
    alias.symlink_to(layout.historical, target_is_directory=True)
    env = {"OUROBOROS_APP_ROOT": str(alias)}
    assert launch_roots.direct_data_dir(env) == alias / "data", "a link to ~/Ouroboros is ~/Ouroboros"
    assert "OUROBOROS_DATA_DIR" not in launch_roots.direct_entry_environment(REPO, env), \
        "the live root keeps its unexported default"
    variant = {"OUROBOROS_APP_ROOT": str(layout.home / "ouroboros")}
    if _case_insensitive(layout.home):
        assert launch_roots.direct_data_dir(variant) == layout.home / "ouroboros" / "data"
    else:
        with pytest.raises(launch_roots.AmbiguousDataRoot):  # two real directories: never lowercased together
            launch_roots.direct_data_dir(variant)
    (tmp_path / "Aa").mkdir()
    assert launch_roots.same_location(tmp_path / "Aa", tmp_path / "Aa" / ".." / "Aa")
    assert not launch_roots.same_location(tmp_path / "Aa", tmp_path / "missing")


# --------------------------------------------------------------------------- consumers

def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True)


def _source_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init")
    (repo / "f.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed")
    return repo


def test_a_recorded_copy_outside_the_current_worktree_root_keeps_files_record_and_pins(tmp_path, monkeypatch):
    """Negative control on the baseline: the copy's record (and a snapshot's pins) were dropped
    while its files stayed outside the new root, so nothing could ever retire them again. Now
    it is kept like a surviving tree: this process does not handle it again, a new one retries."""
    from ouroboros import subagent_worktrees as wt
    from ouroboros import worktree_removal

    repo, data = _source_repo(tmp_path), tmp_path / "data"
    old_root, new_root = tmp_path / "old-root", tmp_path / "new-root"
    handle = wt.provision_worktree(repo_dir=repo, task_id="t-out", worktree_root=old_root, data_dir=data)
    path = pathlib.Path(handle.path)
    assert wt.remove_worktree(task_id="t-out", worktree_root=new_root, data_dir=data) is False
    record = wt._record_at(path, data)
    assert path.exists() and record is not None and record.get("removal_failed"), "kept, with its evidence"
    assert _git(repo, "branch", "--list", handle.branch).stdout.strip(), "the branch pin stays"
    retired = []
    monkeypatch.setattr(wt, "_retire_worktree", lambda *a, **k: retired.append(a) or False)
    assert wt.prune_orphans(worktree_root=new_root, data_dir=data, live=lambda _t: False) == {"removed": 0, "kept": 1}
    assert retired == [], "the same process does not handle it again on every pass"
    monkeypatch.undo()
    worktree_removal._failed_here.clear()  # a new process: the root was restored meanwhile
    assert wt.remove_worktree(task_id="t-out", worktree_root=old_root, data_dir=data) is True
    assert not path.exists() and wt._record_at(path, data) is None

    snap = wt.provision_execution_snapshot(target_root=repo, task_id="t-snap", snapshot_id="snap-out",
                                           worktree_root=old_root, data_dir=data)
    assert wt.remove_execution_snapshot("snap-out", worktree_root=new_root, data_dir=data) is False
    assert pathlib.Path(snap.path).exists() and wt.find_execution_snapshot("snap-out", data).get("removal_failed")
    assert _git(repo, "rev-parse", "--verify", snap.baseline_ref).stdout.strip(), "the baseline pin stays"
    worktree_removal._failed_here.clear()
    assert wt.remove_execution_snapshot("snap-out", worktree_root=old_root, data_dir=data) is True

    # An arbitrary existing folder a stray record names: never deleted, no git run on its branch.
    foreign = tmp_path / "someone-elses-folder"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("mine", encoding="utf-8")
    _git(repo, "branch", "subagent/t-arb")
    wt._put_record({"path": str(foreign), "task_id": "t-arb", "branch": "subagent/t-arb", "repo_dir": str(repo),
                    "created_at": 0}, data)
    assert wt.remove_worktree(task_id="t-arb", worktree_root=new_root, data_dir=data) is False
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "mine" and wt._record_at(foreign, data)
    assert _git(repo, "branch", "--list", "subagent/t-arb").stdout.strip()

    gone = wt.provision_worktree(repo_dir=repo, task_id="t-gone", worktree_root=old_root, data_dir=data)
    wt._force_rmtree(pathlib.Path(gone.path))  # removed by hand: nothing outside the root is touched by git
    assert wt.remove_worktree(task_id="t-gone", worktree_root=new_root, data_dir=data) is True
    assert wt._record_at(pathlib.Path(gone.path), data) is None
    assert _git(repo, "branch", "--list", gone.branch).stdout.strip(), "no branch named by a stray record is deleted"


def _tool_ctx(layout, data):
    from ouroboros.tools.registry import ToolContext

    for path in (layout.app / "repo", data, layout.app / "ws"):
        path.mkdir(parents=True, exist_ok=True)
    return ToolContext(repo_dir=layout.app / "repo", drive_root=data, workspace_root=layout.app / "ws",
                       workspace_mode="", task_id="deliverables-outside-home")


def test_deliverables_outside_home_are_reachable_while_their_siblings_and_jail_stay_closed(layout, monkeypatch):
    """Negative controls on the baseline: the outside-home refusal ran before the Deliverables carve-out,
    and on the staged tree (Astra R1) a shell's absolute cwd there was refused while cwd=user_files ran."""
    from ouroboros.tool_access import filesystem_affordance_map, resolve_shell_cwd, user_files_path_block_reason
    from ouroboros.tool_access_user_files import UserFilesPathBlockedError, resolve_user_file_path
    from ouroboros.tools.shell import _run_shell

    data = _use_lab(monkeypatch, layout)
    ctx = _tool_ctx(layout, data)
    deliverables = layout.app / "Deliverables"
    assert user_files_path_block_reason(ctx, deliverables / "report.md", operation="write_file") == ""
    assert resolve_user_file_path(ctx, "report.md", operation="write_file") == (deliverables / "report.md").resolve()
    assert "outside user home" in user_files_path_block_reason(ctx, layout.app / "projects" / "x.md", operation="write_file")
    assert "outside user home" in user_files_path_block_reason(ctx, layout.app / "x.md", operation="read_file")
    assert user_files_path_block_reason(ctx, deliverables / ".git" / "config", operation="write_file")
    (deliverables / "sub").mkdir(parents=True)
    (layout.app / "projects").mkdir()
    for cwd in ("user_files", str(deliverables), str(deliverables / "sub")):  # the named and absolute forms agree
        assert resolve_shell_cwd(ctx, cwd)[0] == (deliverables if cwd == "user_files" else pathlib.Path(cwd)).resolve()
    ran = _run_shell(ctx, [sys.executable, "-c", "print('ran in deliverables')"], cwd=str(deliverables))
    assert "ran in deliverables" in str(ran) and "CWD_BLOCKED" not in str(ran)
    for sibling in (layout.app, layout.app / "projects", data):
        with pytest.raises(ValueError):
            resolve_shell_cwd(ctx, str(sibling))
    assert filesystem_affordance_map(ctx)["allowed_shell_cwd_roots"].count("user_files") == 1
    (deliverables / "escape").symlink_to(data, target_is_directory=True)
    assert user_files_path_block_reason(ctx, deliverables / "escape" / "settings.json", operation="write_file")
    with pytest.raises(ValueError):
        resolve_shell_cwd(ctx, str(deliverables / "escape"))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(data / "Deliverables"))  # overlaps the data drive
    assert user_files_path_block_reason(ctx, data / "Deliverables" / "r.md", operation="write_file")
    with pytest.raises(ValueError):
        resolve_shell_cwd(ctx, str(data / "Deliverables"))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    jail = layout.home / "jail"
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(jail))
    assert "outside user home" in user_files_path_block_reason(ctx, deliverables / "r.md", operation="write_file")
    with pytest.raises(ValueError):
        resolve_shell_cwd(ctx, str(deliverables))
    monkeypatch.delenv("OUROBOROS_DELIVERABLES_ROOT")
    assert resolve_user_file_path(ctx, "r.md", operation="write_file") == (jail / "Deliverables" / "r.md").resolve()
    with pytest.raises(UserFilesPathBlockedError):
        resolve_user_file_path(ctx, str(layout.app / "projects" / "x.md"), operation="write_file")


def test_cli_discovery_reads_the_tuple_the_direct_server_uses_and_start_refuses_an_ambiguous_one(layout, monkeypatch):
    from ouroboros import cli

    monkeypatch.setenv("OUROBOROS_APP_ROOT", str(layout.app))
    monkeypatch.delenv("OUROBOROS_URL", raising=False)
    port = layout.app / "data" / "state" / "server_port"
    port.parent.mkdir(parents=True)
    port.write_text("9411", encoding="utf-8")
    assert cli._default_base_url().endswith(":9411")
    _write_witness(layout.historical / "data")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: pytest.fail("an ambiguous start spawned a server"))
    with pytest.raises(cli.CLIError, match="OUROBOROS_DATA_DIR="):
        cli._start_local_server("http://127.0.0.1:9411")


def _packaged(monkeypatch, tmp_path):
    from ouroboros import packaged_cli

    monkeypatch.setattr(packaged_cli, "_find_bundle_root", lambda _paths: tmp_path / "bundle")
    monkeypatch.setattr(packaged_cli, "_find_embedded_python", lambda _root: tmp_path / "python")
    monkeypatch.setattr(packaged_cli, "_read_version", lambda _root: "0.0.0")
    return packaged_cli


def test_the_packaged_cli_uses_app_roots_tuple_and_carries_it_into_the_macos_app(layout, monkeypatch, tmp_path):
    """Negative control on the staged tree (Astra R1): a custom APP_ROOT refused `run --start`, and a PID
    file or work root alone was dropped by a bare `open`, so the app started on its default folders."""
    packaged_cli = _packaged(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_APP_ROOT", str(layout.app))
    runtime = packaged_cli.resolve_packaged_runtime()
    a = layout.app
    assert (runtime.app_root, runtime.repo_dir, runtime.data_dir, runtime.settings_path, runtime.pid_file,
            runtime.port_file) == (a, a / "repo", a / "data", a / "data" / "settings.json", a / "ouroboros.pid",
                                   a / "data" / "state" / "server_port")
    contexts: list = []
    monkeypatch.setattr(packaged_cli, "check_git", lambda _windows: True)
    monkeypatch.setattr(packaged_cli, "bootstrap_repo", contexts.append)
    packaged_cli._bootstrap_runtime(runtime)
    assert contexts[0].settings_path == a / "data" / "settings.json" and contexts[0].repo_dir == a / "repo"
    app = tmp_path / "Ouroboros.app"
    launched: list = []
    monkeypatch.setattr(packaged_cli, "IS_MACOS", True)
    monkeypatch.setattr(packaged_cli, "_desktop_app_path", lambda _root: app)
    monkeypatch.setattr(packaged_cli.subprocess, "Popen", lambda args, **kwargs: launched.append((args, kwargs)))
    monkeypatch.setattr(packaged_cli.subprocess, "run", lambda *a, **k: pytest.fail(f"no other command: {a}"))

    def opened() -> dict:
        """The one `open` argv: a new instance whose environment is exactly the `--env` pairs."""
        args, kwargs = launched.pop()
        assert not launched and kwargs == {} and args[:2] == ["open", "-n"] and args[-1] == str(app)
        pairs = args[2:-1]
        assert pairs[::2] == ["--env"] * (len(pairs) // 2)
        return dict(pair.split("=", 1) for pair in pairs[1::2])

    packaged_cli._launch_desktop_app(runtime)
    carried = opened()  # the app resolves exactly this CLI's tuple from what it is handed
    assert launch_roots.configured_roots(carried) == (a, a / "repo", a / "data", a / "data" / "settings.json",
                                                      a / "ouroboros.pid", a / "data" / "state" / "server_port")
    monkeypatch.delenv("OUROBOROS_APP_ROOT")
    h = layout.historical
    for key, value in (("OUROBOROS_PID_FILE", tmp_path / "second.pid"), ("OUROBOROS_PORT_FILE", tmp_path / "port"),
                       ("OUROBOROS_DELIVERABLES_ROOT", tmp_path / "out"), ("OUROBOROS_DATA_DIR", tmp_path / "d"),
                       ("OUROBOROS_SETTINGS_PATH", tmp_path / "s.json")):
        monkeypatch.setenv(key, str(value))
        packaged_cli._launch_desktop_app(packaged_cli.resolve_packaged_runtime())
        carried = opened()
        assert carried[key] == str(value) and carried["OUROBOROS_APP_ROOT"] == str(h), key
        monkeypatch.delenv(key)
    packaged_cli._launch_desktop_app(packaged_cli.resolve_packaged_runtime())
    carried = opened()
    assert carried["OUROBOROS_APP_ROOT"] == str(h), "default CLI must not reactivate a custom-root instance"


@pytest.mark.parametrize("bound,target,accepted", [
    ("127.0.0.2", "127.0.0.1", False), ("127.0.0.1", "127.0.0.2", False),
    ("127.0.0.1", "127.0.0.1", True), ("::1", "::1", True),
    ("::1", "127.0.0.1", False), ("0.0.0.0", "127.0.0.1", True),
    ("::", "::1", True),
])
def test_packaged_implicit_endpoint_matches_the_bound_address(layout, monkeypatch, bound, target, accepted):
    from ouroboros import packaged_cli, server_process

    data = _use_lab(monkeypatch, layout)
    server_process.record_service_binding(data, "main", bound, 19401, pid=os.getpid())
    host = f"[{target}]" if ":" in target else target
    url = f"http://{host}:19401"
    monkeypatch.setattr(packaged_cli, "_gateway_supervisor_ready", lambda _url: True)
    runtime = SimpleNamespace(data_dir=data)
    assert packaged_cli._selected_ready(url, runtime, explicit=False) is accepted
    assert packaged_cli._selected_ready(url, runtime, explicit=True) is True


class _ReadySupervisor(http.server.BaseHTTPRequestHandler):
    """A loopback endpoint answering the two requests `--start` reads, as any ready installation does."""

    def do_GET(self):  # noqa: N802 — http.server's handler name
        body = json.dumps({"status": "ok"} if self.path == "/api/health" else {"supervisor_ready": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def test_start_reaches_only_the_selected_data_roots_own_server(layout, monkeypatch, tmp_path):
    """Negative control on the staged tree (Astra R1): with a new APP_ROOT (no port file yet) and another
    installation ready on the fallback port, `--start` launched nothing and sent the task there."""
    from ouroboros import server_process

    packaged_cli = _packaged(monkeypatch, tmp_path)
    servers = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ReadySupervisor) for _ in range(2)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    other, own = (f"http://127.0.0.1:{server.server_port}" for server in servers)
    other_data = tmp_path / "other" / "data"  # another installation; the suite never writes ~/Ouroboros/data
    try:
        monkeypatch.setattr(packaged_cli, "DEFAULT_PORT", servers[0].server_port)  # the fallback port it holds
        server_process.record_service_binding(other_data, "main", "127.0.0.1", servers[0].server_port, pid=os.getpid())
        monkeypatch.delenv("OUROBOROS_URL", raising=False)
        monkeypatch.setenv("OUROBOROS_CLI_START_TIMEOUT", "10")
        monkeypatch.setenv("OUROBOROS_APP_ROOT", str(layout.app))
        runtime = packaged_cli.resolve_packaged_runtime()
        launched: list = []

        def app_publishes_its_server(started):  # what the opened app's server records for its own root
            launched.append(started.data_dir)
            started.port_file.parent.mkdir(parents=True, exist_ok=True)
            started.port_file.write_text(str(servers[1].server_port), encoding="utf-8")
            server_process.record_service_binding(started.data_dir, "main", "127.0.0.1", servers[1].server_port,
                                                  pid=os.getpid())

        monkeypatch.setattr(packaged_cli, "_launch_desktop_app", app_publishes_its_server)
        assert packaged_cli._prepare_start_if_requested(["run", "--start", "task"], runtime) == ["run", "task"]
        assert launched == [layout.app / "data"] and os.environ["OUROBOROS_URL"] == own
        monkeypatch.delenv("OUROBOROS_URL")
        packaged_cli._prepare_start_if_requested(["run", "--start", "again"], runtime)
        assert len(launched) == 1 and os.environ["OUROBOROS_URL"] == own, "its running server: no second launch"
        monkeypatch.delenv("OUROBOROS_URL")
        packaged_cli._prepare_start_if_requested(["--url", other, "run", "--start", "named"], runtime)
        assert len(launched) == 1 and os.environ["OUROBOROS_URL"] == other, "a named URL is the caller's choice"
        monkeypatch.delenv("OUROBOROS_URL")
        monkeypatch.delenv("OUROBOROS_APP_ROOT")
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(other_data))  # that installation's own CLI, no port file
        packaged_cli._prepare_start_if_requested(["run", "--start", "its own"], packaged_cli.resolve_packaged_runtime())
        assert len(launched) == 1 and os.environ["OUROBOROS_URL"] == other, "it finds its server on the fallback port"
        monkeypatch.delenv("OUROBOROS_URL")
        monkeypatch.delenv("OUROBOROS_DATA_DIR")
        monkeypatch.setenv("OUROBOROS_APP_ROOT", str(tmp_path / "never-started"))
        monkeypatch.setenv("OUROBOROS_CLI_START_TIMEOUT", "1")
        monkeypatch.setattr(packaged_cli, "_launch_desktop_app", lambda _runtime: None)
        with pytest.raises(packaged_cli.PackagedCLIError, match="is not the one for .*pass --url"):
            packaged_cli._prepare_start_if_requested(["run", "--start", "lost"], packaged_cli.resolve_packaged_runtime())
        assert not (tmp_path / "never-started").exists(), "the identity check only reads"
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_source_start_uses_only_its_data_roots_own_server(layout, monkeypatch, tmp_path, capsys):
    """Negative control on the staged tree (review R3): with a new APP_ROOT (no port file yet) and another
    installation ready on the default port, source `run --start` attached to it before any root check."""
    import argparse

    from ouroboros import cli, server_process

    servers = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ReadySupervisor) for _ in range(2)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    other, own = (f"http://127.0.0.1:{server.server_port}" for server in servers)
    data, real_start = layout.app / "data", cli._start_local_server
    started: list = []

    class Spawned:  # the server `--start` spawned: the default port is busy, so it takes a fallback later
        returncode, polls = None, 0

        def poll(self):
            self.polls += 1
            if self.polls == 3:
                (data / "state").mkdir(parents=True, exist_ok=True)
                (data / "state" / "server_port").write_text(str(servers[1].server_port), encoding="utf-8")
                server_process.record_service_binding(data, "main", "127.0.0.1", servers[1].server_port,
                                                      pid=os.getpid())
            return None

    def client(url="", *, start=True):
        return cli._client(argparse.Namespace(url=url), start=start).base_url

    try:
        monkeypatch.setattr(cli, "DEFAULT_PORT", servers[0].server_port)  # held by the other installation
        server_process.record_service_binding(tmp_path / "other" / "data", "main", "127.0.0.1",
                                              servers[0].server_port, pid=os.getpid())
        monkeypatch.delenv("OUROBOROS_URL", raising=False)
        monkeypatch.setenv("OUROBOROS_APP_ROOT", str(layout.app))
        monkeypatch.setattr(cli, "_start_local_server", lambda url: started.append(url) or Spawned())
        monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
        assert client() == own and started == [other], "its own server, on the port that server took"
        assert "is not the one for" in capsys.readouterr().err
        assert client() == own and len(started) == 1, "its running server: no second start"
        assert client(other) == other and len(started) == 1, "a named URL is the caller's choice"
        monkeypatch.setenv("OUROBOROS_URL", other)
        assert client() == other and len(started) == 1, "so is OUROBOROS_URL"
        monkeypatch.delenv("OUROBOROS_URL")
        (data / "state" / "server_port").unlink()
        assert client(start=False) == other, "without --start a command attaches to what answers, as before"
        _write_witness(layout.historical / "data")  # an ambiguous APP_ROOT: the root check comes first
        monkeypatch.setattr(cli, "_start_local_server", real_start)
        monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: pytest.fail("an ambiguous start spawned"))
        with pytest.raises(cli.CLIError, match="OUROBOROS_DATA_DIR="):
            client()
        assert len(started) == 1
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_colab_keeps_git_checkouts_local_while_a_new_installs_projects_follow_drive(layout, monkeypatch, tmp_path):
    from ouroboros import colab_bootstrap

    drive_data = tmp_path / "drive" / "Ouroboros" / "data"
    drive_data.mkdir(parents=True)
    env = colab_bootstrap.export_colab_env(tmp_path / "repo", drive_data, drive_data / "settings.json")
    local = str(layout.historical / "subagent_worktrees")
    assert env["OUROBOROS_SUBAGENT_WORKTREE_ROOT"] == local and env["OUROBOROS_APP_ROOT"] == str(drive_data.parent)
    roots = _roots()
    assert roots["subagent_worktrees"] == local
    assert roots["projects"] == str(drive_data.parent / "projects")
    assert roots["Deliverables"] == str(drive_data.parent / "Deliverables")
    _write_witness(drive_data)  # an existing Colab install keeps its historical roots
    assert _roots() == {leaf: str(layout.historical / leaf) for _key, leaf in LEAVES}
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "mine"))
    assert colab_bootstrap.export_colab_env(tmp_path / "repo", drive_data, drive_data / "settings.json").get(
        "OUROBOROS_SUBAGENT_WORKTREE_ROOT") is None, "the operator's own export is left alone"


def test_a_rooted_android_layout_follows_the_same_rule(layout, monkeypatch, tmp_path):
    """The provisioner's explicit APP_ROOT/REPO/DATA/SETTINGS with HOME=/root, simulated on disk."""
    opt = tmp_path / "opt" / "ouroboros"
    for key, value in (("OUROBOROS_APP_ROOT", opt), ("OUROBOROS_REPO_DIR", opt / "repo"),
                       ("OUROBOROS_DATA_DIR", opt / "data"), ("OUROBOROS_SETTINGS_PATH", opt / "data" / "settings.json")):
        monkeypatch.setenv(key, str(value))
    assert _roots() == {leaf: str(opt / leaf) for _key, leaf in LEAVES}
    _write_witness(opt / "data")
    assert _roots() == {leaf: str(layout.historical / leaf) for _key, leaf in LEAVES}


def test_children_inherit_the_hosts_basis_and_app_root(layout, monkeypatch, tmp_path):
    from ouroboros import extension_process_runner as runner
    from ouroboros import process_custody, startup_historical_audit

    data = _use_lab(monkeypatch, layout, data=tmp_path / "split" / "data")
    monkeypatch.setenv(launch_roots.BASIS_ENV, launch_roots.APP_ROOT_BASIS)
    env = runner._child_env(drive_root=data, repo_dir=REPO, skill_name="probe", skill_dir=tmp_path,
                            env_allowlist=[], granted_keys=[])
    assert env["OUROBOROS_APP_ROOT"] == str(layout.app), "never inferred from the drive's parent"
    assert env[launch_roots.BASIS_ENV] == launch_roots.APP_ROOT_BASIS
    captured: dict = {}
    monkeypatch.setattr(process_custody, "spawn_supervised", lambda argv, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "session")
    startup_historical_audit.spawn_maintenance_job(data, REPO, "storage")
    assert captured["env"]["OUROBOROS_SUBAGENT_PROJECTS_ROOT"] == str(layout.app / "projects")
    assert captured["env"][launch_roots.BASIS_ENV] == launch_roots.APP_ROOT_BASIS
    monkeypatch.setenv(launch_roots.BASIS_ENV, launch_roots.UNKNOWN_BASIS)
    startup_historical_audit.spawn_maintenance_job(data, REPO, "storage")
    assert "OUROBOROS_SUBAGENT_PROJECTS_ROOT" not in captured["env"], "undecided: omitted, never guessed"
    # Negative control on the staged tree (Astra R1): without the host's APP_ROOT the child chose
    # ~/Ouroboros/<leaf>, the folders the host refused. Its data and settings stay the host's own.
    from ouroboros import config

    assert captured["env"]["OUROBOROS_APP_ROOT"] == str(layout.app)
    assert (captured["env"]["OUROBOROS_DATA_DIR"], captured["env"]["OUROBOROS_SETTINGS_PATH"]) == (
        str(data), str(config.SETTINGS_PATH)), "never derived from APP_ROOT or the drive's parent"
    probe = ("import json, os; from ouroboros.storage import place_roots; print(json.dumps({row['id']: "
             "row.get('undecided') or row['roots'] for row in place_roots(os.environ['OUROBOROS_DATA_DIR'])}))")
    child = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO), env=captured["env"], capture_output=True,
                           text=True, timeout=120)
    assert child.returncode == 0, child.stderr[-2000:]
    places = json.loads(child.stdout.strip().splitlines()[-1])
    for place, key in (("project_folders", "OUROBOROS_SUBAGENT_PROJECTS_ROOT"),
                       ("working_copies", "OUROBOROS_SUBAGENT_WORKTREE_ROOT"), ("deliverables", "OUROBOROS_DELIVERABLES_ROOT")):
        assert "is undecided" in places[place] and key in places[place], places[place]


def test_classifiers_treat_an_undecided_default_root_as_containing_nothing(layout, monkeypatch, tmp_path):
    from ouroboros import coop_checkpoint, delegate_shared, project_sources, storage

    data = _use_lab(monkeypatch, layout)
    monkeypatch.setenv(launch_roots.BASIS_ENV, launch_roots.UNKNOWN_BASIS)
    places = {row["id"]: row for row in storage.place_roots(data)}
    for place_id, key in (("project_folders", "OUROBOROS_SUBAGENT_PROJECTS_ROOT"),
                          ("working_copies", "OUROBOROS_SUBAGENT_WORKTREE_ROOT"), ("deliverables", "OUROBOROS_DELIVERABLES_ROOT")):
        assert places[place_id]["roots"] == [] and key in places[place_id]["undecided"]
    inside = layout.app / "projects" / "p"
    assert delegate_shared.orphan_apply_target_ok(inside, layout.app / "elsewhere") is False
    assert delegate_shared.orphan_apply_target_ok(inside, inside) is True, "equality still qualifies"
    assert coop_checkpoint._task_tree_coop_roots(data, "root-task") == []
    path, code, detail = project_sources.clone_project_repo("https://example.invalid/r.git")
    assert (path, code) == ("", "work_root_unavailable") and "OUROBOROS_SUBAGENT_PROJECTS_ROOT" in detail


def test_the_storage_count_shows_an_undecided_work_root_as_unknown_never_as_absent(layout, monkeypatch, tmp_path):
    """Negative control on the staged tree (Astra R1): the real count saved the project folders of a
    root whose witness was lost as absent with zero bytes, and the Storage list hid them."""
    from ouroboros import storage

    data = _use_lab(monkeypatch, layout)
    (data / "state").mkdir(parents=True)
    (data / "settings.json").write_text("{}", encoding="utf-8")
    state = {"initialization_id": "prior", "session_id": "prior-session", "created_at": "2026-10-27T00:00:00Z"}
    (data / "state" / "state.json").write_text(json.dumps(state), encoding="utf-8")
    assert witness.complete(data, witness.prepare_adoption(data, state), adopted=True)  # the lost witness, re-minted
    kept = layout.app / "projects" / "retained-work.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("retained project work\n", encoding="utf-8")
    (data / "task_results").mkdir()
    (data / "task_results" / "t.json").write_text("{}", encoding="utf-8")
    (layout.app / "repo").mkdir()  # the counted program; the child imports this checkout through PYTHONPATH
    env = _child_env(layout, OUROBOROS_APP_ROOT=str(layout.app), OUROBOROS_DATA_DIR=str(data),
                     OUROBOROS_SETTINGS_PATH=str(data / "settings.json"), OUROBOROS_REPO_DIR=str(layout.app / "repo"))
    run = subprocess.run([sys.executable, "-m", "ouroboros.storage", "--data-root", str(data)], cwd=str(REPO),
                         env=env, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr[-2000:]
    saved = json.loads((data / storage.MEASUREMENT_FILE).read_text(encoding="utf-8"))["latest"]
    places = {place["id"]: place for place in saved["places"]}
    for place_id in ("project_folders", "working_copies", "deliverables"):
        place = places[place_id]
        assert (place["present"], place["bytes"], place["unreadable"], place["roots"]) == (True, None, 1, [])
        assert "is undecided" in place["undecided"]
    assert places["task_results"]["bytes"] > 0 and "undecided" not in places["task_results"], "resolved places count"
    assert saved["unreadable"] >= 3 and kept.read_text(encoding="utf-8") == "retained project work\n"
    # The free-space tray's preview fails typed and names the setting, not just the error's type.
    from ouroboros import storage_free

    monkeypatch.setenv(launch_roots.BASIS_ENV, launch_roots.UNKNOWN_BASIS)
    free = storage_free.FreeSpace(live=lambda _task: None, offered=(storage_free.GROUP_LOST_COPIES,))
    free._list(data, REPO)
    tray = free.projection(data)
    assert tray["state"] == "failed" and "OUROBOROS_SUBAGENT_WORKTREE_ROOT" in tray["error"], tray["error"]
