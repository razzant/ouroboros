"""Focused tests for the OSWorld step-loop adapter (no network, no docker).

Covers the OSWorld 2.0 alignment surface: protocol defaults, checkout
variant/commit detection, provider preflight failures, the official result
persistence contract, and the cu_bridge sample-60 final_answer fix.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools.benchmarks.osworld import run_step_agent as rsa


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_bench_runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_BENCH_RUNS_ROOT", str(tmp_path / "bench_runs"))


def _agent(tmp_path: Path) -> rsa.OuroborosStepAgent:
    result_dir = tmp_path / "example"
    result_dir.mkdir(parents=True, exist_ok=True)
    return rsa.OuroborosStepAgent(rsa.StepAgentConfig(
        ouroboros_bin="/nonexistent/ouroboros",
        ouroboros_url="http://127.0.0.1:9",
        repo_dir=REPO_ROOT,
        data_dir=tmp_path / "data",
        settings_path=tmp_path / "settings.json",
        result_dir=result_dir,
        task_id="test-task",
        model="anthropic/claude-sonnet-4.6",
        timeout_sec=5,
        max_obs_chars=4000,
        screenshot_check_only=False,
    ))


def _fake_run(payload: dict) -> object:
    def fake(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    return fake


def test_arg_parser_osworld2_defaults():
    defaults = {a.dest: a.default for a in rsa.build_arg_parser()._actions}
    assert defaults["max_steps"] == 500  # OSWorld 2.0 protocol budget
    assert defaults["disable_tools"] == "claude_code_edit"
    assert defaults["observation_type"] == "screenshot_a11y_tree"
    assert defaults["provider_name"] == "vmware"


def test_aligned_upstream_pin_recorded():
    pin = rsa.ALIGNED_UPSTREAM
    assert pin["repo"] == "https://github.com/xlang-ai/OSWorld-V2"
    assert pin["commit"] == "c261cb57a699bd18db128787ca4e71b749141762"
    assert pin["protocol_max_steps"] == 500
    assert pin["protocol_checkpoint_steps"] == [150, 300]


def test_osworld_checkout_info_variant_markers(tmp_path):
    v2 = tmp_path / "v2"
    (v2 / "evaluation_examples").mkdir(parents=True)
    (v2 / "evaluation_examples" / "test_v2.json").write_text("{}", encoding="utf-8")
    (v2 / "desktop_env").mkdir()
    (v2 / "desktop_env" / "desktop_env.py").write_text("# stub\n", encoding="utf-8")
    info = rsa.osworld_checkout_info(v2)
    assert info["variant"] == "v2"
    assert info["has_desktop_env"] is True
    assert info["matches_aligned_commit"] is False  # no git checkout here

    v1 = tmp_path / "v1"
    (v1 / "evaluation_examples").mkdir(parents=True)
    (v1 / "evaluation_examples" / "test_all.json").write_text("{}", encoding="utf-8")
    assert rsa.osworld_checkout_info(v1)["variant"] == "v1"

    bare = tmp_path / "bare"
    (bare / "evaluation_examples").mkdir(parents=True)
    assert rsa.osworld_checkout_info(bare)["variant"] == "examples_only"

    assert rsa.osworld_checkout_info(tmp_path / "missing")["variant"] == "unknown"


def test_provider_preflight_rejects_unsupported_provider_loudly():
    failures = rsa.provider_preflight_failures("aws", "")
    assert failures, "unsupported provider must fail preflight"
    assert "not supported" in failures[0]
    assert "vmware" in failures[0] and "docker" in failures[0]


def test_provider_preflight_vmware_missing_vm(tmp_path):
    failures = rsa.provider_preflight_failures("vmware", str(tmp_path / "missing.vmx"))
    assert any("VM path not found" in failure for failure in failures)


def test_provider_preflight_docker_missing_cli(monkeypatch):
    monkeypatch.setattr(rsa.shutil, "which", lambda name: None)
    failures = rsa.provider_preflight_failures("docker", "")
    assert any("docker CLI not found" in failure for failure in failures)


def test_persist_evaluation_result_official_contract(tmp_path):
    # Legacy float result -> result.txt only.
    score = rsa._persist_evaluation_result(0.5, tmp_path)
    assert score == 0.5
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "0.5\n"
    assert not (tmp_path / "result.json").exists()
    # OSWorld 2.0 dict result -> canonical score in result.txt + full result.json.
    score = rsa._persist_evaluation_result({"score": 0.25, "checkpoints": [1, 0]}, tmp_path)
    assert score == 0.25
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "0.25\n"
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["score"] == 0.25


def test_prompt_declares_vm_state_grading_and_final_answer(tmp_path):
    agent = _agent(tmp_path)
    agent.step_idx = 1
    prompt = agent._prompt("Find the similar names to Carl", {"accessibility_tree": ""}, "", max_steps=500)
    assert "VM STATE" in prompt
    assert "final_answer" in prompt
    assert "ACTIVE TAB URL" in prompt
    assert "chat" in prompt.lower()


def test_predict_captures_final_answer_on_done(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    monkeypatch.setattr(rsa.subprocess, "run", _fake_run({
        "response": "Navigated to the answer page; finishing.",
        "final_answer": "Henry, Charles, Mason",
        "actions": [{"type": "done"}],
    }))
    response, actions, debug = agent.predict("task", {"screenshot": b"png-bytes"}, max_steps=500)
    assert actions == ["DONE"]
    assert agent.terminal_action == "DONE"
    assert agent.final_answer == "Henry, Charles, Mason"
    assert debug["returncode"] == 0


def test_predict_falls_back_to_terminal_response_when_final_answer_missing(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    monkeypatch.setattr(rsa.subprocess, "run", _fake_run({
        "response": "Done. The names are Henry and Charles.",
        "actions": ["DONE"],
    }))
    agent.predict("task", {}, max_steps=500)
    assert agent.final_answer == "Done. The names are Henry and Charles."
    assert agent.terminal_action == "DONE"


def test_predict_records_fail_as_infeasible_terminal(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    monkeypatch.setattr(rsa.subprocess, "run", _fake_run({
        "response": "Spotify cannot be installed here.",
        "final_answer": "TASK_INFEASIBLE",
        "actions": [{"type": "fail"}],
    }))
    _response, actions, _debug = agent.predict("task", {}, max_steps=500)
    assert actions == ["FAIL"]
    assert agent.terminal_action == "FAIL"
    assert agent.final_answer == "TASK_INFEASIBLE"
    # reset() must clear terminal state for the next episode
    agent.reset()
    assert agent.final_answer == "" and agent.terminal_action == ""


def test_predict_passes_disable_tools_to_ouroboros_cli(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    seen: dict[str, list[str]] = {}

    def fake(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"response": "ok", "actions": ["WAIT"]}), stderr="")

    monkeypatch.setattr(rsa.subprocess, "run", fake)
    agent.predict("task", {}, max_steps=500)
    cmd = seen["cmd"]
    assert "--disable-tools" in cmd
    assert cmd[cmd.index("--disable-tools") + 1] == "claude_code_edit"


def test_settings_template_follows_sprint_scaffold_defaults():
    template = json.loads(
        (REPO_ROOT / "devtools" / "benchmarks" / "osworld" / "settings_base.json").read_text(encoding="utf-8")
    )
    assert template["OUROBOROS_MAX_WORKERS"] == 4
    assert template["OUROBOROS_SAFETY_MODE"] == "light"
    assert template["OUROBOROS_RUNTIME_MODE"] == "pro"  # isolated VM bench
    for key, value in template.items():
        if key.endswith(("_API_KEY", "_TOKEN")):
            assert value == "", f"secret {key} must ship blank"
    # single-model scaffold: solver slots all point at the same model
    assert (
        template["OUROBOROS_MODEL"]
        == template["OUROBOROS_MODEL_HEAVY"]
        == template["OUROBOROS_MODEL_LIGHT"]
        == template["OUROBOROS_MODEL_FALLBACKS"]
    )


def test_run_step_agent_refuses_live_desktop_server_url(tmp_path, monkeypatch):
    """Real runs must not write bench steps into the LIVE desktop server: the
    default http://127.0.0.1:8765 URL is refused without --allow-live-server."""
    import pytest as _pytest

    from devtools.benchmarks.osworld import run_step_agent as rsa

    parser = rsa.build_arg_parser()
    args = parser.parse_args(["--task", "t.json", "--result_dir", str(tmp_path)])
    assert args.ouroboros_url.rstrip("/") == "http://127.0.0.1:8765"
    assert not args.allow_live_server
    monkeypatch.setattr(rsa.sys, "argv",
                        ["run_step_agent.py", "--task", "t.json", "--result_dir", str(tmp_path)])
    with _pytest.raises(SystemExit, match="refusing the default desktop server port"):
        rsa.main()


def test_preflight_verifies_server_scaffold_settings(tmp_path, monkeypatch):
    """`ouroboros run --url` cannot configure the executing server via env, so
    the preflight must verify the TARGET server's effective settings against
    the disclosed scaffold (pro/light/4 + the requested model) and fail loudly
    on drift; --allow-scaffold-mismatch downgrades to a recorded detail."""
    from devtools.benchmarks.osworld import run_step_agent as rsa

    osworld_root = tmp_path / "osworld"
    (osworld_root / "evaluation_examples").mkdir(parents=True)
    (osworld_root / "desktop_env").mkdir()
    (osworld_root / "desktop_env" / "desktop_env.py").write_text("# stub\n", encoding="utf-8")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.55.0\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    (data_dir / "uploads").mkdir(parents=True)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"OUROBOROS_MODEL": "anthropic/claude-sonnet-4.6",
                                         "ANTHROPIC_API_KEY": "k"}), encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text("{}", encoding="utf-8")
    vmx = tmp_path / "vm.vmx"
    vmx.write_text("cfg", encoding="utf-8")
    monkeypatch.setattr(rsa, "provider_preflight_failures", lambda *a, **k: [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # slash-form model routes via OpenRouter

    drifted = {"OUROBOROS_RUNTIME_MODE": "light", "OUROBOROS_SAFETY_MODE": "full",
               "OUROBOROS_MAX_WORKERS": 1, "OUROBOROS_MODEL": "openai/gpt-5.5",
               "OUROBOROS_REVIEW_ENFORCEMENT": "advisory"}

    def fake_http(url, timeout=5):
        if url.endswith("/api/state"):
            return {"supervisor_ready": True, "runtime_mode": drifted["OUROBOROS_RUNTIME_MODE"]}
        if url.endswith("/api/settings"):
            return dict(drifted)
        raise AssertionError(url)

    monkeypatch.setattr(rsa, "_http_json", fake_http)
    cfg = rsa.PreflightConfig(
        osworld_root=osworld_root, task_path=task_path, path_to_vm=str(vmx),
        repo_dir=repo_dir, data_dir=data_dir, settings_path=settings_path,
        result_root=tmp_path / "results", ouroboros_url="http://127.0.0.1:8770",
        model="anthropic/claude-sonnet-4.6",
    )
    result = rsa._preflight(cfg)
    joined = " ".join(result["failures"])
    assert not result["ok"]
    assert "OUROBOROS_RUNTIME_MODE" in joined and "OUROBOROS_SAFETY_MODE" in joined
    assert "OUROBOROS_MAX_WORKERS" in joined and "OUROBOROS_MODEL" in joined
    # Adversarial review r2 #6: the blocking review lane is part of the scaffold gate.
    assert "OUROBOROS_REVIEW_ENFORCEMENT" in joined

    # matching server -> ok; mismatch + explicit ablation flag -> ok but recorded
    matching = {"OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_SAFETY_MODE": "light",
                "OUROBOROS_MAX_WORKERS": 4, "OUROBOROS_MODEL": "anthropic/claude-sonnet-4.6",
                "OUROBOROS_REVIEW_ENFORCEMENT": "blocking"}

    def fake_http_ok(url, timeout=5):
        if url.endswith("/api/state"):
            return {"supervisor_ready": True, "runtime_mode": "pro"}
        if url.endswith("/api/settings"):
            return dict(matching)
        raise AssertionError(url)

    monkeypatch.setattr(rsa, "_http_json", fake_http_ok)
    assert rsa._preflight(cfg)["ok"]

    monkeypatch.setattr(rsa, "_http_json", fake_http)
    import dataclasses as _dc
    relaxed = _dc.replace(cfg, allow_scaffold_mismatch=True)
    result3 = rsa._preflight(relaxed)
    assert result3["ok"]
    assert result3["details"]["scaffold_mismatch_allowed"]

