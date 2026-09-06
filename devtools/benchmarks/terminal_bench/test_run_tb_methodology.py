"""Focused tests for the leaderboard-faithfulness invariants of run_tb.py.

Run explicitly (it lives under devtools, not tests/, to stay merge-clean):
    PYTHONPATH=<repo> python -m pytest devtools/benchmarks/terminal_bench/test_run_tb_methodology.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

import pytest

from devtools.benchmarks.terminal_bench import run_tb


@pytest.fixture(autouse=True)
def _hermetic_process_environment():
    """`run_tb.apply_all_model` (and `main --all-model` through it) writes the
    fixed-model contract into `os.environ` directly — the launcher's real
    behaviour, kept. Under xdist that leaked `OUROBOROS_REVIEWER_SLOTS` and the
    forwarded slot keys into every later test of the same worker (the
    `benchmark-scope-1` contamination class). Every test here runs on a
    snapshot of the environment that is restored afterwards, whatever it wrote —
    and starts WITHOUT the operator shell's reviewer panel or legacy comma-list
    keys, in the spirit of tests/conftest.py's
    `_scrub_inherited_subagent_selection` but with a deliberately different key
    set: conftest drops the subagent roster, the account pin and the structured
    panel; this suite reads BOTH panel forms, so it drops the structured panel
    AND the legacy comma-list keys (`OUROBOROS_REVIEW_MODELS`,
    `OUROBOROS_SCOPE_REVIEW_MODELS`, `OUROBOROS_SCOPE_REVIEW_MODEL`). A
    panel-reading test here is hermetic against the shell either way."""
    saved = dict(os.environ)
    for key in ("OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_REVIEW_MODELS",
                "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL"):
        os.environ.pop(key, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


# --- validate_methodology gates -------------------------------------------------

def test_k_below_5_raises_without_allow():
    with pytest.raises(ValueError):
        run_tb.validate_methodology(k=1, timeout_multiplier=1.0, resource_overrides=[])


def test_k_below_5_allowed_with_flag():
    run_tb.validate_methodology(k=1, timeout_multiplier=1.0, resource_overrides=[], allow_low_k=True)


def test_setup_build_multiplier_raises_without_allow():
    with pytest.raises(ValueError):
        run_tb.validate_methodology(k=5, timeout_multiplier=1.0, resource_overrides=[], setup_timeout_multiplier=4.0)
    with pytest.raises(ValueError):
        run_tb.validate_methodology(k=5, timeout_multiplier=1.0, resource_overrides=[], build_timeout_multiplier=4.0)


def test_setup_build_multiplier_allowed_with_flag():
    run_tb.validate_methodology(
        k=5, timeout_multiplier=1.0, resource_overrides=[],
        setup_timeout_multiplier=4.0, build_timeout_multiplier=4.0, allow_setup_build_multipliers=True,
    )


# --- harbor_command output ------------------------------------------------------

def _cfg(**over):
    base = dict(
        dataset="terminal-bench/terminal-bench-2-1", model="google/gemini-3.5-flash", k=5,
        jobs_dir=pathlib.Path("/tmp/jd"), harbor_bin="harbor", n_concurrent=1, task_filters=[],
        settings_path=pathlib.Path("/tmp/s.json"), execute=False, light_model="google/gemini-3.5-flash",
    )
    base.update(over)
    return run_tb.HarborCommandConfig(**base)


def test_faithful_command_omits_multiplier_flags_and_gates_web(tmp_path):
    cmd = run_tb.harbor_command(_cfg(jobs_dir=tmp_path))
    assert "--agent-setup-timeout-multiplier" not in cmd
    assert "--environment-build-timeout-multiplier" not in cmd
    config = json.loads((tmp_path / "agent_job_config.json").read_text(encoding="utf-8"))
    assert config["agents"][0]["kwargs"]["disable_agent_web"] is True


def test_local_override_emits_multiplier_flags():
    cmd = run_tb.harbor_command(_cfg(setup_timeout_multiplier=4.0, build_timeout_multiplier=2.0))
    assert "--agent-setup-timeout-multiplier" in cmd and "4.0" in cmd
    assert "--environment-build-timeout-multiplier" in cmd and "2.0" in cmd


def test_allow_agent_web_flips_kwarg(tmp_path):
    run_tb.harbor_command(_cfg(jobs_dir=tmp_path, disable_agent_web=False))
    config = json.loads((tmp_path / "agent_job_config.json").read_text(encoding="utf-8"))
    assert config["agents"][0]["kwargs"]["disable_agent_web"] is False


def test_pip_cache_mount_is_env_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("OBO_TB_PIP_CACHE", raising=False)
    assert "--mounts" not in run_tb.harbor_command(_cfg())

    cache = tmp_path / "pip-cache"
    monkeypatch.setenv("OBO_TB_PIP_CACHE", str(cache))
    cmd = run_tb.harbor_command(_cfg())
    idx = cmd.index("--mounts")
    mounts = json.loads(cmd[idx + 1])
    assert mounts == [{"type": "bind", "source": str(cache), "target": "/opt/ouro-pip-cache"}]
    assert cache.is_dir()


def test_pip_cache_mount_rejects_repo_path(monkeypatch):
    repo = pathlib.Path(run_tb.__file__).resolve().parents[3]
    monkeypatch.setenv("OBO_TB_PIP_CACHE", str(repo / ".bad-pip-cache"))
    with pytest.raises(ValueError, match="must not be under repo"):
        run_tb.harbor_command(_cfg())


# --- apply_all_model + metadata -------------------------------------------------

def _poison_fixed_actor_env(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "foreign-t", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol"}}],
        "scope": [{"slot_id": "foreign-s", "route": {
            "kind": "api_chat", "target_id": "foreign/scope"}}],
        "advisory": {"enabled": True, "route": {
            "kind": "api_chat", "target_id": "foreign-advisory"}},
    }))
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "foreign-sdk-model")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "foreign/heavy")
    for key in ("USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
                "USE_LOCAL_CONSCIOUSNESS"):
        monkeypatch.setenv(key, "true")


def test_apply_all_model_sets_forwarded_slots(monkeypatch):
    for key in run_tb._ALL_MODEL_SLOT_KEYS + (
        "OUROBOROS_REVIEW_MODELS", "OUROBOROS_SUBAGENTS",
        "OUROBOROS_REVIEWER_SLOTS", "CLAUDE_CODE_MODEL",
        "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS",
        "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
        "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
        "USE_LOCAL_CONSCIOUSNESS", "OUROBOROS_MODEL_HEAVY", "USE_LOCAL_HEAVY",
        "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
    ):
        monkeypatch.delenv(key, raising=False)
    _poison_fixed_actor_env(monkeypatch)
    actor = run_tb.apply_all_model("google/gemini-3.5-flash")
    import os
    assert actor["mismatches"] == []
    assert actor["reviewer_slots"]["advisory"]["enabled"] is False
    assert "OUROBOROS_MODEL_HEAVY" not in os.environ
    for key in run_tb._ALL_MODEL_SLOT_KEYS:
        assert os.environ[key] == "google/gemini-3.5-flash"
    # Single-model run defaults to ONE reviewer at low effort (3 identical = monoculture, no diversity).
    assert os.environ["OUROBOROS_REVIEW_MODELS"] == "google/gemini-3.5-flash"
    assert os.environ["OUROBOROS_EFFORT_REVIEW"] == "low"
    assert os.environ["OUROBOROS_EFFORT_SCOPE_REVIEW"] == "low"
    actors = json.loads(os.environ["OUROBOROS_SUBAGENTS"])
    assert [row["route"]["target_id"] for row in actors["items"]] == ["google/gemini-3.5-flash"]
    reviewers = json.loads(os.environ["OUROBOROS_REVIEWER_SLOTS"])
    assert [row["route"]["target_id"] for row in reviewers["triad"]] == [
        "google/gemini-3.5-flash"
    ]
    assert [row["route"]["target_id"] for row in reviewers["scope"]] == [
        "google/gemini-3.5-flash"
    ]
    assert reviewers["advisory"]["enabled"] is False
    assert all(os.environ[key] == "false" for key in (
        "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
        "USE_LOCAL_CONSCIOUSNESS",
    ))
    # Configurable: the 3-identical-reviewer / medium-effort path is still available.
    run_tb.apply_all_model("google/gemini-3.5-flash", review_slots=3, review_effort="medium")
    assert os.environ["OUROBOROS_REVIEW_MODELS"] == "google/gemini-3.5-flash,google/gemini-3.5-flash,google/gemini-3.5-flash"
    assert os.environ["OUROBOROS_EFFORT_REVIEW"] == "medium"
    reviewers = json.loads(os.environ["OUROBOROS_REVIEWER_SLOTS"])
    assert len(reviewers["triad"]) == 3
    assert {row["effort"] for row in reviewers["triad"]} == {"medium"}


def test_all_model_actor_is_durable_before_tb_external_probe(tmp_path, monkeypatch):
    model = "openai/gpt-5.5"
    run_root = tmp_path / "run"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    _poison_fixed_actor_env(monkeypatch)

    def external_probe(_binary):
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        actor = manifest["harness"]["fixed_model_actor"]
        assert actor["mismatches"] == []
        assert not any(actor["local_routes"].values())
        assert actor["reviewer_slots"]["advisory"]["enabled"] is False
        assert {row["route"]["target_id"] for row in actor["reviewer_slots"]["triad"]} == {model}
        return "test-harbor"

    monkeypatch.setattr(run_tb, "harbor_version", external_probe)
    assert run_tb.main([
        "--all-model", model, "--allow-low-k", "--allow-dirty-seed",
        "--run-root", str(run_root), "--submission-root", str(tmp_path / "submission"),
        "--settings-path", str(settings),
    ]) == 0


def test_malformed_reviewer_panel_is_a_typed_refusal_on_the_durable_manifest(tmp_path, monkeypatch):
    """The launcher structural gate: nothing reads files before admission, so a
    malformed panel (here: in the host settings file the adapter forwards) is
    refused INSIDE the finalize seam — recorded on the durable manifest with the
    launcher's own vocabulary, no traceback, and no submission tree built."""
    model = "openai/gpt-5.5"
    run_root = tmp_path / "run"
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"OUROBOROS_REVIEWER_SLOTS": "{not json"}), encoding="utf-8")
    _poison_fixed_actor_env(monkeypatch)
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setattr(run_tb, "harbor_version", lambda _binary: "test-harbor")
    assert run_tb.main([
        "--model", model, "--allow-low-k", "--allow-dirty-seed",
        "--run-root", str(run_root), "--submission-root", str(tmp_path / "submission"),
        "--settings-path", str(settings),
    ]) == 1
    manifest_text = (run_root / "run_manifest.json").read_text(encoding="utf-8")
    assert '"leaderboard_metadata"' in manifest_text and '"refused"' in manifest_text
    assert "OUROBOROS_REVIEWER_SLOTS" in manifest_text  # the typed reason names the key
    assert not list((tmp_path / "submission").rglob("metadata.yaml"))


def test_adapter_forwards_fixed_model_execution_contract(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    model = "openai/gpt-5.5"
    for key in (
        *run_tb._ALL_MODEL_SLOT_KEYS,
        "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_MODEL_CONSCIOUSNESS",
        "OUROBOROS_MODEL_VISION", "OUROBOROS_SUBAGENTS", "OUROBOROS_REVIEWER_SLOTS",
        "CLAUDE_CODE_MODEL", "OUROBOROS_REVIEW_MODELS", "OUROBOROS_EFFORT_REVIEW",
        "OUROBOROS_EFFORT_SCOPE_REVIEW", "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT",
        "USE_LOCAL_FALLBACK", "USE_LOCAL_CONSCIOUSNESS", "OUROBOROS_MODEL_HEAVY",
        "USE_LOCAL_HEAVY",
    ):
        monkeypatch.delenv(key, raising=False)

    run_tb.apply_all_model(model)
    env = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)._container_env()
    reviewers = json.loads(env["OUROBOROS_REVIEWER_SLOTS"])
    assert {row["route"]["target_id"] for row in reviewers["triad"]} == {model}
    assert {row["route"]["target_id"] for row in reviewers["scope"]} == {model}
    assert reviewers["advisory"]["enabled"] is False
    assert "CLAUDE_CODE_MODEL" not in env
    assert all(env[key] == "false" for key in (
        "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
        "USE_LOCAL_CONSCIOUSNESS",
    ))


@pytest.mark.parametrize(
    "light_model,uses_local_light",
    (
        ("anthropic/claude-sonnet-4.6", False),
        ("owner/light (local)", True),
    ),
)
def test_harbor_smoke_child_uses_the_durable_pinned_actor(
    tmp_path, monkeypatch, light_model, uses_local_light
):
    from devtools.benchmarks.terminal_bench import run_harbor_smoke as smoke
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    model = "openai/gpt-5.5"
    run_root = tmp_path / "smoke"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    _poison_fixed_actor_env(monkeypatch)

    def fake_harbor(cmd, **kwargs):
        assert cmd[0] == "harbor"
        agent_kwargs = {
            cmd[index + 1].split("=", 1)[0]: cmd[index + 1].split("=", 1)[1]
            for index, token in enumerate(cmd[:-1])
            if token == "--agent-kwarg"
        }
        assert agent_kwargs["ouroboros_model"] == model
        assert agent_kwargs["ouroboros_light_model"] == light_model
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        actor = manifest["harness"]["fixed_model_actor"]
        env = kwargs["env"]
        forwarded_keys = (
            "OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT",
            "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_SUBAGENTS",
            "OUROBOROS_REVIEWER_SLOTS", "CLAUDE_CODE_MODEL",
            "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
            "USE_LOCAL_CONSCIOUSNESS", "OUROBOROS_MODEL_HEAVY",
            "USE_LOCAL_HEAVY",
        )
        for key in forwarded_keys:
            if key in env:
                monkeypatch.setenv(key, env[key])
            else:
                monkeypatch.delenv(key, raising=False)
        container_env = tb_agent.OuroborosTerminalBenchAgent(
            logs_dir=tmp_path / "adapter-logs",
            ouroboros_model=agent_kwargs["ouroboros_model"],
            ouroboros_light_model=agent_kwargs["ouroboros_light_model"],
            host_settings_path=agent_kwargs["host_settings_path"],
        )._container_env()
        reviewers = json.loads(container_env["OUROBOROS_REVIEWER_SLOTS"])
        subagents = json.loads(container_env["OUROBOROS_SUBAGENTS"])
        assert actor["mismatches"] == []
        assert actor["model_slots"]["OUROBOROS_MODEL"] == model
        assert actor["model_slots"]["OUROBOROS_MODEL_LIGHT"] == light_model
        assert actor["model_slots"]["OUROBOROS_MODEL_FALLBACKS"] == model
        assert actor["local_routes"] == {
            "USE_LOCAL_MAIN": False,
            "USE_LOCAL_LIGHT": uses_local_light,
            "USE_LOCAL_FALLBACK": False,
            "USE_LOCAL_CONSCIOUSNESS": False,
        }
        assert manifest["model_slots"] == {
            key: actor["model_slots"][key]
            for key in (
                "OUROBOROS_MODEL",
                "OUROBOROS_MODEL_LIGHT",
                "OUROBOROS_MODEL_FALLBACKS",
            )
        }
        assert container_env["OUROBOROS_MODEL"] == model
        assert container_env["OUROBOROS_MODEL_LIGHT"] == light_model
        assert container_env["OUROBOROS_MODEL_FALLBACK"] == model
        assert container_env["OUROBOROS_MODEL_FALLBACKS"] == model
        assert [row["route"]["target_id"] for row in subagents["items"]] == [model]
        assert {row["route"]["target_id"] for row in reviewers["triad"]} == {model}
        assert {row["route"]["target_id"] for row in reviewers["scope"]} == {model}
        assert reviewers["advisory"]["enabled"] is False
        assert container_env["USE_LOCAL_MAIN"] == "false"
        assert container_env["USE_LOCAL_LIGHT"] == str(uses_local_light).lower()
        assert container_env["USE_LOCAL_FALLBACK"] == "false"
        assert container_env["USE_LOCAL_CONSCIOUSNESS"] == "false"
        assert "CLAUDE_CODE_MODEL" not in container_env
        assert "OUROBOROS_MODEL_HEAVY" not in container_env
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(smoke.subprocess, "run", fake_harbor)
    monkeypatch.setattr(smoke.sys, "argv", [
        "run_harbor_smoke.py", "--model", model,
        "--ouroboros-light-model", light_model,
        "--execute", "--allow-dirty-seed",
        "--run-root", str(run_root), "--settings-path", str(settings),
    ])
    assert smoke.main() == 1


def test_metadata_omits_web_search_when_web_disabled(monkeypatch):
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "openai/gpt-5.2")
    roles_on = dict(run_tb._effective_helper_models("google/gemini-3.5-flash", "google/gemini-3.5-flash", disable_agent_web=False))
    roles_off = dict(run_tb._effective_helper_models("google/gemini-3.5-flash", "google/gemini-3.5-flash", disable_agent_web=True))
    assert any("web_search" in r for r in roles_on.values())
    assert not any("web_search" in r for r in roles_off.values())


_PANEL = {
    "triad": [
        {"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.5"}},
        {"slot_id": "t2", "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"}},
    ],
    "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "google/gemini-3.5-pro"}}],
    "advisory": {"enabled": False},
}


def test_metadata_declares_what_the_container_executes_from_the_structured_panel(monkeypatch):
    """The container runs the structured panel the adapter forwards (operator
    env, else the host settings file). Inside a TB task nothing commits: the
    panel reaches the run through task acceptance, which runs every row on its
    own delivery (owner R2, 2026-09-01) — but a task container structurally
    cannot run an agent-session row (no harness CLI/daemon, no harness
    credentials in the forwarded env). Metadata therefore declares the api rows
    by model id and NEVER the session row (a declared-but-never-run model would
    misrepresent the submission); the session row is a typed disclosure,
    `triad_rows_not_executable_in_container`, and neither a stale legacy comma
    key nor a shipped default the container does not run is declared."""

    monkeypatch.delenv("OUROBOROS_WEBSEARCH_MODEL", raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "foreign/stale-triad")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "foreign/stale-scope")
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps(_PANEL))
    roles = dict(run_tb._effective_helper_models("openai/gpt-5.5", "google/gemini-3.5-flash", disable_agent_web=True))
    assert "foreign/stale-triad" not in roles and "foreign/stale-scope" not in roles
    assert roles["openai/gpt-5.5"] == "agent+commit_review_triad"
    # The session row is disclosed, not declared: nothing in the container runs it.
    assert "codex=gpt-5.6-sol" not in roles and not any("agent_session" in r for r in roles.values())
    assert run_tb.triad_rows_not_executable_in_container("openai/gpt-5.5") == ["codex=gpt-5.6-sol"]
    # Scope review is a commit-time gate: it never fires inside a task, so its
    # rows are not declared (the same honesty rule as the advisory).
    assert "google/gemini-3.5-pro" not in roles and "scope_review" not in roles.values()
    assert roles["google/gemini-3.5-flash"] == "light_safety_post_task_synthesis"
    meta = run_tb.leaderboard_metadata(
        agent_name="Ouroboros", org_name="Ouroboros", model="openai/gpt-5.5",
        light_model="google/gemini-3.5-flash", disable_agent_web=True,
    )
    assert 'model_name: "codex=gpt-5.6-sol"' not in meta and 'model_provider: "codex"' not in meta
    # The disclosure rides metadata.yaml as a COMMENT (the leaderboard owns its keys)
    # and run_manifest.json as the typed field.
    assert '# triad_rows_not_executable_in_container: ["codex=gpt-5.6-sol"]' in meta

    # An all-session triad declares NO reviewer row — and not the shipped defaults
    # either: nothing in the container runs them.
    all_session = {**_PANEL, "triad": [_PANEL["triad"][1]]}
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps(all_session))
    roles = dict(run_tb._effective_helper_models("openai/gpt-5.5", "google/gemini-3.5-flash", disable_agent_web=True))
    assert roles["openai/gpt-5.5"] == "agent" and "commit_review_triad" not in "+".join(roles.values())
    # The shipped triad defaults: ABI 7.0 retired the OUROBOROS_REVIEW_MODELS
    # settings key, and the launcher itself reads the SSOT list the key used
    # to be derived from (run_tb._effective_helper_models).
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    for helper in OPENROUTER_REVIEW_DEFAULTS["triad"]:
        assert helper not in roles
    assert run_tb.triad_rows_not_executable_in_container("openai/gpt-5.5") == ["codex=gpt-5.6-sol"]

    # Settings-file fallback, exactly like the container adapter's env → settings order.
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    settings = {"OUROBOROS_REVIEWER_SLOTS": json.dumps(_PANEL)}
    roles = dict(run_tb._effective_helper_models(
        "openai/gpt-5.5", "google/gemini-3.5-flash", disable_agent_web=True, settings=settings))
    assert roles["openai/gpt-5.5"] == "agent+commit_review_triad" and "foreign/stale-triad" not in roles
    assert run_tb.triad_rows_not_executable_in_container("openai/gpt-5.5", settings) == ["codex=gpt-5.6-sol"]
    # No structured panel at all: nothing to disclose.
    assert run_tb.triad_rows_not_executable_in_container("openai/gpt-5.5") == []


_PANEL_WITH_TWO_SESSIONS = {
    "triad": [
        {"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.5"}},
        {"slot_id": "t2", "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"}},
        {"slot_id": "t3", "route": {"kind": "agent_session", "target_id": "cursor=openai/gpt-5"}},
    ],
    "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "google/gemini-3.5-pro"}}],
    "advisory": {"enabled": False},
}


def test_run_manifest_and_metadata_carry_the_rows_the_container_cannot_run(tmp_path, monkeypatch, capsys):
    """End to end through `main` (command generation, no harbor): the durable
    `run_manifest.json` carries `extra.triad_rows_not_executable_in_container`
    with the session rows' targets verbatim and in row order — a target with
    its own `/` (`cursor=openai/gpt-5`) included — `metadata.yaml` carries the
    same list as a comment while declaring no session row as a model, and
    (owner R40) admission prints ONE loud stderr warning naming each row and
    the typed degradation while the run continues."""
    model = "openai/gpt-5.5"
    run_root = tmp_path / "run"
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"OUROBOROS_REVIEWER_SLOTS": json.dumps(_PANEL_WITH_TWO_SESSIONS)}), encoding="utf-8")
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.delenv("OUROBOROS_WEBSEARCH_MODEL", raising=False)
    monkeypatch.setattr(run_tb, "harbor_version", lambda _binary: "test-harbor")
    assert run_tb.main([
        "--model", model, "--allow-low-k", "--allow-dirty-seed",
        "--run-root", str(run_root), "--submission-root", str(tmp_path / "submission"),
        "--settings-path", str(settings),
    ]) == 0
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    extra = manifest["extra"]
    assert extra["outcome"] == "command_generated"
    assert extra["triad_rows_not_executable_in_container"] == ["codex=gpt-5.6-sol", "cursor=openai/gpt-5"]
    meta = pathlib.Path(extra["metadata_yaml"]).read_text(encoding="utf-8")
    assert '# triad_rows_not_executable_in_container: ["codex=gpt-5.6-sol", "cursor=openai/gpt-5"]' in meta
    assert 'model_name: "codex' not in meta and 'model_name: "cursor' not in meta
    assert f'model_name: "{model}"' in meta and 'role: "agent+commit_review_triad"' in meta
    err = capsys.readouterr().err
    assert err.count("[run_tb] WARNING: the configured reviewer triad carries agent-session rows") == 1
    assert "codex=gpt-5.6-sol, cursor=openai/gpt-5" in err and "degrades typed" in err
    assert "Configure api/native triad rows" in err


def test_an_api_only_panel_prints_no_container_warning(tmp_path, monkeypatch, capsys):
    """The R40 warning is for session rows only: an api-only panel admits silently."""
    run_root = tmp_path / "run"
    settings = tmp_path / "settings.json"
    api_only = {**_PANEL_WITH_TWO_SESSIONS, "triad": [_PANEL_WITH_TWO_SESSIONS["triad"][0]]}
    settings.write_text(json.dumps({"OUROBOROS_REVIEWER_SLOTS": json.dumps(api_only)}), encoding="utf-8")
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setattr(run_tb, "harbor_version", lambda _binary: "test-harbor")
    assert run_tb.main([
        "--model", "openai/gpt-5.5", "--allow-low-k", "--allow-dirty-seed",
        "--run-root", str(run_root), "--submission-root", str(tmp_path / "submission"),
        "--settings-path", str(settings),
    ]) == 0
    assert "[run_tb] WARNING: the configured reviewer triad" not in capsys.readouterr().err
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["triad_rows_not_executable_in_container"] == []


def test_metadata_parses_the_panel_under_the_container_roster(monkeypatch):
    """A subagent-bound row resolves against the CONTAINER's one-model roster
    (the adapter replaces the operator roster), never the operator shell's."""
    from devtools.benchmarks.common.model_slots import BENCHMARK_SUBAGENT_ID

    monkeypatch.delenv("OUROBOROS_SUBAGENTS", raising=False)
    panel = {**_PANEL, "triad": [{"slot_id": "t1", "subagent_id": BENCHMARK_SUBAGENT_ID}]}
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps(panel))
    roles = dict(run_tb._effective_helper_models("openai/gpt-5.5", "google/gemini-3.5-flash", disable_agent_web=True))
    # The benchmark actor row RETRIEVES (native tool rounds) on the measured
    # model: acceptance executes it, so the measured model carries the triad
    # role and no shipped default is declared.
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    assert roles["openai/gpt-5.5"] == "agent+commit_review_triad"
    assert not any(h in roles for h in OPENROUTER_REVIEW_DEFAULTS["triad"])
    # An operator-roster reference the container cannot resolve is a typed refusal.
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({**panel, "triad": [{"slot_id": "t1", "subagent_id": "operator-critic"}]}))
    with pytest.raises(ValueError, match="operator-critic"):
        run_tb._effective_helper_models("openai/gpt-5.5", "google/gemini-3.5-flash", disable_agent_web=True)


def test_metadata_never_declares_the_retired_claude_code_role(monkeypatch):
    """The Claude-SDK transport (claude_code_edit) is retired: a stale
    CLAUDE_CODE_MODEL in the operator env must not add a metadata role."""
    monkeypatch.delenv("OUROBOROS_WEBSEARCH_MODEL", raising=False)
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "anthropic/claude-opus-4.8")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "google/gemini-3.5-flash")
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "google/gemini-3.5-flash")
    roles = dict(run_tb._effective_helper_models("google/gemini-3.5-flash", "google/gemini-3.5-flash", disable_agent_web=True))
    assert not any("claude_code_edit" in r for r in roles.values())
    assert list(roles.keys()) == ["google/gemini-3.5-flash"]


# --- report_grade (D: low-k variance warning) -----------------------------------

def test_report_grade_k1_debug_only():
    grade, warn = run_tb.report_grade(k=1, leaderboard_valid=False)
    assert grade == "debug_only" and warn and "k=1" in warn


def test_report_grade_low_k():
    grade, warn = run_tb.report_grade(k=3, leaderboard_valid=False)
    assert grade == "local_low_k" and warn and "k=3" in warn


def test_report_grade_leaderboard_valid_no_warning():
    grade, warn = run_tb.report_grade(k=5, leaderboard_valid=True)
    assert grade == "leaderboard_valid" and warn == ""


def test_report_grade_configurable_floor():
    grade, warn = run_tb.report_grade(k=7, leaderboard_valid=False, low_k_floor=10)
    assert grade == "local_low_k" and "< 10" in warn


def test_report_grade_valid_overrides_floor():
    # a leaderboard-valid run is ALWAYS leaderboard_valid regardless of the floor knob
    grade, warn = run_tb.report_grade(k=5, leaderboard_valid=True, low_k_floor=10)
    assert grade == "leaderboard_valid" and warn == ""


def test_report_grade_k5_not_valid_is_local():
    # k>=5 but a non-faithful setting (e.g. web-on) => not leaderboard_valid => local_low_k,
    # and the warning must NOT falsely claim "k < floor" (the reason is the off-spec setting).
    grade, warn = run_tb.report_grade(k=5, leaderboard_valid=False)
    assert grade == "local_low_k" and warn
    assert "< 5" not in warn and "not leaderboard-valid" in warn.lower()


# --- disclosure ledger ----------------------------------------------------------

def _write_trial(d: pathlib.Path, task: str, reward, exc=None, reason=None, truncated=False):
    d.mkdir(parents=True, exist_ok=True)
    meta = {"turns": 3}
    if reason is not None:
        meta["summary"] = {"reason_code": reason, "infra_failed": False, "truncated": truncated,
                           "resource_limit": ({"status": "resource_limited", "scope": "root"}
                                              if truncated else {})}
    (d / "result.json").write_text(json.dumps({
        "task_name": task, "trial_name": d.name,
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": ({"exception_type": exc} if exc else None),
        "agent_result": {"cost_usd": 0.01, "metadata": meta},
    }), encoding="utf-8")


def _write_run_summary(d: pathlib.Path, captured_after_cancellation: bool):
    adir = d / "agent"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "ouroboros-run-summary.json").write_text(
        json.dumps({"captured_after_cancellation": captured_after_cancellation, "status": "failed"}),
        encoding="utf-8",
    )


def test_disclosure_ledger_provider_unavailable_vs_cancellation(tmp_path):
    """A terminal `provider_unavailable` (captured_after_cancellation=False) is a real
    provider/infra failure; the same reason_code captured during a Harbor cancellation
    snapshot (captured_after_cancellation=True) is a wall-clock cancellation, not provider."""
    jobs = tmp_path / "job"
    _write_trial(jobs / "p1", "alpha", 0.0, reason="provider_unavailable")
    _write_run_summary(jobs / "p1", False)  # normal terminal finish
    _write_trial(jobs / "p2", "beta", 0.0, reason="provider_unavailable")
    _write_run_summary(jobs / "p2", True)  # interrupted/teardown snapshot
    led = run_tb.write_disclosure_ledger(jobs_dir=jobs, out_path=tmp_path / "led.json", run_meta={})
    assert led["provider_or_infra_failure_count"] == 1  # only p1 (genuine mid-run provider death)
    assert led["wall_clock_cancellation_count"] == 1  # only p2 (teardown artifact)
    assert led["genuine_failure_count"] == 0


def test_disclosure_ledger_separates_cost_truncation_from_a_fair_shot_wrong_answer(tmp_path):
    """A trial the RUNTIME's own resource rail stopped is not a `genuine` failure.

    `genuine` asserts a FAIR SHOT — reward 0 having reached a real terminal. A trial cut off
    by the per-task USD reservation bound (a worst-case estimate that reached the rail at
    $0.45 of actual spend in the v6.81.0 OSWorld smoke) never got one, and counting it as a
    wrong answer overstates the failure as a capability result. It is not provider/infra
    either: nothing was unavailable.
    """
    jobs = tmp_path / "job"
    _write_trial(jobs / "b1", "alpha", 0.0, reason="budget_exhausted", truncated=True)
    _write_trial(jobs / "b2", "beta", 0.0)  # a real fair-shot wrong answer
    led = run_tb.write_disclosure_ledger(jobs_dir=jobs, out_path=tmp_path / "led.json", run_meta={})
    assert led["cost_truncated_count"] == 1
    assert led["genuine_failure_count"] == 1  # b2 only
    assert led["provider_or_infra_failure_count"] == 0
    assert led["wall_clock_cancellation_count"] == 0
    truncated_row = next(t for t in led["trials"] if t["task_name"] == "alpha")
    assert truncated_row["truncated"] is True
    assert truncated_row["resource_limit"]["scope"] == "root"
    assert "cost_truncated" in led["exception_note"]


def test_disclosure_ledger_counts(tmp_path):
    jobs = tmp_path / "job"
    _write_trial(jobs / "t1", "alpha", 1.0)
    _write_trial(jobs / "t2", "alpha", 0.0, exc="AgentTimeoutError")
    _write_trial(jobs / "t3", "beta", None, exc="RuntimeError")  # provider/infra (Harbor exception)
    _write_trial(jobs / "t4", "beta", 0.0, reason="provider_unavailable")  # clean reward-0, 429 artifact
    _write_trial(jobs / "t5", "gamma", 0.0)  # genuine wrong answer (no exc, no provider reason)
    led = run_tb.write_disclosure_ledger(jobs_dir=jobs, out_path=tmp_path / "led.json", run_meta={})
    assert led["n_trials"] == 5
    assert led["agent_timeout_count"] == 1
    # RuntimeError (t3) + provider_unavailable reason_code (t4) both count; AgentTimeoutError does NOT
    assert led["provider_or_infra_failure_count"] == 2
    assert led["reason_code_histogram"].get("provider_unavailable") == 1
    assert led["genuine_failure_count"] == 1  # only t5 is a real wrong answer
    assert led["reward_distribution"].get("1.0") == 1  # normalized bucket (not split '1' vs '1.0')
    assert led["per_task_pass_rate"]["alpha"] == 0.5


# --- v6.79.0 readiness flags must not change the leaderboard-faithful command ---

def test_default_command_emits_no_env_passthrough_or_base_config(tmp_path):
    """The new readiness knobs are strictly opt-in: an unflagged run's argv is unchanged, and
    the job config still contains exactly our agents[] block."""
    cmd = run_tb.harbor_command(_cfg(jobs_dir=tmp_path))
    assert "--ae" not in cmd and "--ve" not in cmd
    assert run_tb.redacted_command(cmd) == cmd
    config = json.loads((tmp_path / "agent_job_config.json").read_text(encoding="utf-8"))
    assert list(config) == ["agents"]
    assert config["agents"][0]["kwargs"]["dataset"] == "terminal-bench/terminal-bench-2-1"


# A stand-in for the thing this guard exists to stop: a shell-expanded credential arriving where
# a NAME=VALUE pair was expected. Deliberately NOT shaped like a real key -- the point of the test
# is that this string never appears in an artifact, and a realistic-looking token in a fixture is
# itself a small leak of the pattern scrubbers look for.
_UNSPLITTABLE_TOKEN = "NOT-AN-ASSIGNMENT-PLACEHOLDER"


@pytest.mark.parametrize("flag", ["--agent-env", "--verifier-env"])
def test_env_passthrough_without_equals_is_refused_before_any_artifact(flag, tmp_path, capsys):
    """A `--ve $OPENAI_API_KEY` typo hands this launcher one token with no `=`. Every consumer
    split on the first `=` and called the left half a NAME, so the WHOLE token -- the credential --
    was persisted as `verifier_env_keys` in run_manifest.json, printed as `<token>=<redacted>` in
    the official command, and named in the passthrough warning. It must be refused at parse, with
    the token never echoed, and nothing written."""
    run_root = tmp_path / "run"
    with pytest.raises(SystemExit) as excinfo:
        run_tb.main(["--run-root", str(run_root), flag, _UNSPLITTABLE_TOKEN])
    assert excinfo.value.code == 2
    streams = capsys.readouterr()
    assert _UNSPLITTABLE_TOKEN not in streams.out
    assert _UNSPLITTABLE_TOKEN not in streams.err
    assert "NAME=VALUE" in streams.err
    # No manifest, no run root, nothing on disk to leak from.
    assert not list(tmp_path.rglob("run_manifest.json"))
    assert not run_root.exists()


@pytest.mark.parametrize(
    "bad",
    [_UNSPLITTABLE_TOKEN, "=novalue", "1LEADING_DIGIT=x", "HAS SPACE=x", "HAS\nNEWLINE=x", "a-b=x"],
)
def test_env_assignment_requires_a_posix_name_and_an_explicit_equals(bad):
    """The name is the only half this launcher ever writes down, so it must BE a name: no `=` at
    all, an empty name, or one carrying a space/newline/dash would inject into the manifest and the
    warning, or produce a scrub marker nothing can match."""
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        run_tb.env_assignment(bad)
    assert bad not in str(excinfo.value)
    assert run_tb.env_assignment("OPENAI_API_KEY=x=y") == "OPENAI_API_KEY=x=y"
    assert run_tb.env_assignment("_UNDERSCORE0=") == "_UNDERSCORE0="


def test_redacted_command_never_publishes_a_malformed_token_as_a_name():
    """Defence in depth for an argv assembled outside `main()`: a passthrough value that is not a
    well-formed `NAME=…` is replaced WHOLESALE, instead of being emitted as `<secret>=<redacted>`."""
    out = run_tb.redacted_command(["harbor", "--ve", _UNSPLITTABLE_TOKEN, "--ae", "OPENAI_API_KEY=x"])
    assert out == ["harbor", "--ve", "<redacted>", "--ae", "OPENAI_API_KEY=<redacted>"]
    assert _UNSPLITTABLE_TOKEN not in " ".join(out)


def test_tb21_submission_subtree_is_unchanged():
    """A derived path must reproduce the published TB2.1 tree exactly."""
    assert run_tb.submission_subtree(run_tb.DEFAULT_DATASET) == ("terminal-bench", "2.1")


# --- Frontier-Bench readiness (dataset identity + execution backend disclosure) ---

def test_frontier_bench_identity_threads_through_job_config_and_argv(tmp_path):
    """Frontier-Bench is a DATASET, not a second launcher: the same identity that already feeds
    harbor's `--dataset` must reach the adapter kwarg it uses to pick the per-task cache subtree
    (`~/.cache/harbor/tasks/packages/frontier-bench/<task>/<digest>`). A parallel mechanism here
    would silently make FB runs deadline-blind, which is the exact bug v6.79.0 fixed for TB2.1."""
    cmd = run_tb.harbor_command(_cfg(jobs_dir=tmp_path, dataset=run_tb.FRONTIER_BENCH_DATASET))
    assert cmd[cmd.index("--dataset") + 1] == "frontier-bench/frontier-bench"
    config = json.loads((tmp_path / "agent_job_config.json").read_text(encoding="utf-8"))
    assert config["agents"][0]["kwargs"]["dataset"] == "frontier-bench/frontier-bench"


def test_frontier_bench_submission_subtree_carries_no_invented_version():
    """FB carries its version in the harbor REF (`@v0.1.0`), never in the dataset name, so the
    name-based derivation must NOT fabricate one: it yields an empty version component that main()
    drops, instead of mis-parsing `frontier-bench` into a `frontier/bench`-shaped path."""
    assert run_tb.submission_subtree(run_tb.FRONTIER_BENCH_DATASET) == ("frontier-bench", "")


def test_submission_subtree_strips_a_pinned_harbor_ref():
    """A reproducible FB run pins an immutable ref, and `latest` is mutable — so pinning is the
    NORMAL case, not an edge one. The ref is registry addressing: it must not leak a literal `@`
    into a submission directory name, and TB2.1's derivation must survive being pinned too."""
    assert run_tb.submission_subtree("frontier-bench/frontier-bench@v0.1.0") == ("frontier-bench", "")
    assert run_tb.submission_subtree("frontier-bench/frontier-bench@sha256:abc123") == ("frontier-bench", "")
    assert run_tb.submission_subtree("terminal-bench/terminal-bench-2-1@5") == ("terminal-bench", "2.1")


def test_harbor_env_backend_is_opt_in_and_absent_by_default(tmp_path):
    """Backend selection must not perturb the published TB2.1 argv: with no flag, harbor's own
    default (docker) applies and no `--env` token is emitted."""
    assert "--env" not in run_tb.harbor_command(_cfg(jobs_dir=tmp_path))


def test_harbor_env_backend_is_emitted_when_chosen(tmp_path):
    """A cloud backend is reachable without a base job config — upstream FB CI uses `--env modal`,
    and our local docker verification does not make that path unreachable."""
    cmd = run_tb.harbor_command(_cfg(jobs_dir=tmp_path, harbor_env="modal"))
    assert cmd[cmd.index("--env") + 1] == "modal"


def test_harbor_version_reports_a_fake_binary_and_swallows_failures(monkeypatch):
    """Harness provenance is best-effort by design: a working binary is reported verbatim, and an
    un-interrogable one records "" (visibly unknown) instead of aborting a run.

    The three outcomes are injected at the `subprocess.run` seam rather than acted out with a real
    `#!/bin/sh` script: Windows does not honour a shebang on direct execution, so the script form
    made `harbor_version` return "" on the Windows CI shard and failed the 0.20.0 assertion there
    while passing everywhere the author could see."""
    calls: list[list[str]] = []

    def _fake_run(outcome):
        def run(cmd, **kwargs):
            calls.append(list(cmd))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return run

    monkeypatch.setattr(
        run_tb.subprocess, "run",
        _fake_run(subprocess.CompletedProcess(args=[], returncode=0, stdout="0.20.0\n", stderr="")),
    )
    assert run_tb.harbor_version("harbor") == "0.20.0"
    assert calls[-1] == ["harbor", "--version"]

    monkeypatch.setattr(
        run_tb.subprocess, "run",
        _fake_run(subprocess.CompletedProcess(args=[], returncode=3, stdout="", stderr="boom\n")),
    )
    assert run_tb.harbor_version("harbor") == ""

    monkeypatch.setattr(run_tb.subprocess, "run", _fake_run(FileNotFoundError("no such binary")))
    assert run_tb.harbor_version("does-not-exist") == ""
