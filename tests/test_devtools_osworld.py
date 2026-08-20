"""OSWorld: the preflight before the VM and the step agent's own honesty.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
review blockers the preflight refuses, the output and data-root isolation the CLI enforces,
the log normalizer, and the shell action and prompt the step agent sends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


from devtools.benchmarks.osworld.normalize_logs import normalize_bundle

from tests._devtools_benchmarks_shared import REPO_ROOT
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


def test_osworld_shell_action_does_not_fabricate_bash_history():
    """NW-6 methodology integrity: the OSWorld shell action must NOT write the
    command into ~/.bash_history to satisfy terminal-task evaluators (hidden
    verifier knowledge / answer fitting). The only allowed mention is the
    docstring documenting that we deliberately do not do it."""
    src = (REPO_ROOT / "devtools" / "benchmarks" / "osworld" / "step_agent_actions.py").read_text("utf-8")
    # No history-file write in the emitted snippet, no record_history plumbing.
    assert "hist.open(" not in src
    assert "record_history" not in src
    assert ".bash_history'" not in src  # the f.write to the history path is gone

def test_osworld_logs_only_normalizer(tmp_path):
    bundle = tmp_path / "osworld_logs"
    (bundle / "sample1").mkdir(parents=True)
    (bundle / "SUMMARY.json").write_text(json.dumps({"count": 1}), encoding="utf-8")
    (bundle / "sample_manifest.json").write_text(json.dumps({"samples": ["sample1"]}), encoding="utf-8")
    (bundle / "trace_manifest.json").write_text(json.dumps({"traces": ["sample1/traj.jsonl"]}), encoding="utf-8")
    (bundle / "sample1" / "traj.jsonl").write_text(
        json.dumps({"type": "start"}) + "\n" + json.dumps({"type": "end"}) + "\n",
        encoding="utf-8",
    )

    normalized = normalize_bundle(bundle)

    assert normalized["traj_count"] == 1
    assert normalized["traces"][0]["events"] == 2
    assert normalized["traces"][0]["last_type"] == "end"

def test_osworld_logs_only_normalizer_accepts_nested_trace_manifests(tmp_path):
    bundle = tmp_path / "osworld_logs"
    sample = bundle / "chrome" / "sample1"
    (sample / "traces").mkdir(parents=True)
    (bundle / "SUMMARY.json").write_text(json.dumps({"count": 1}), encoding="utf-8")
    (bundle / "sample_manifest.json").write_text(json.dumps({"samples": ["sample1"]}), encoding="utf-8")
    (sample / "traces" / "trace_manifest.json").write_text(json.dumps({"trace": "sample1"}), encoding="utf-8")
    (sample / "traj.jsonl").write_text(json.dumps({"event": "done"}) + "\n", encoding="utf-8")

    normalized = normalize_bundle(bundle)

    assert normalized["trace_manifest"]["trace_manifest_paths"] == ["chrome/sample1/traces/trace_manifest.json"]
    assert normalized["traj_count"] == 1

def test_osworld_preflight_rejects_unix_computer_use_review_blockers(tmp_path):
    from devtools.benchmarks.osworld.osworld_adapter_skeleton import preflight
    from ouroboros.skill_loader import compute_content_hash

    osworld = tmp_path / "OSWorld"
    osworld.mkdir()
    (osworld / "evaluation_examples").mkdir()
    data_root = tmp_path / "data"
    payload = tmp_path / "unix_computer_use"
    payload.mkdir()
    (payload / "SKILL.md").write_text("# unix_computer_use\n", encoding="utf-8")
    content_hash = compute_content_hash(payload)
    state_dir = data_root / "state" / "skills" / "unix_computer_use"
    state_dir.mkdir(parents=True)
    (state_dir / "review.json").write_text(json.dumps({"status": "blockers", "content_hash": content_hash}), encoding="utf-8")
    (state_dir / "enabled.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")

    result = preflight(
        osworld_root=osworld,
        ouroboros_url="http://127.0.0.1:9",
        osworld_server_url="http://127.0.0.1:9",
        unix_computer_use_payload=payload,
        unix_computer_use_state_dir=state_dir,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        data_root=data_root,
    )

    assert result["ok"] is False
    assert any("fresh executable pass/advisory_pass" in failure for failure in result["failures"])

def test_osworld_preflight_rejects_stale_unix_computer_use_review(tmp_path):
    from devtools.benchmarks.osworld.osworld_adapter_skeleton import preflight

    osworld = tmp_path / "OSWorld"
    osworld.mkdir()
    (osworld / "evaluation_examples").mkdir()
    data_root = tmp_path / "data"
    payload = tmp_path / "unix_computer_use"
    payload.mkdir()
    (payload / "SKILL.md").write_text("# unix_computer_use\n", encoding="utf-8")
    (payload / "tool.py").write_text("print('v1')\n", encoding="utf-8")
    state_dir = data_root / "state" / "skills" / "unix_computer_use"
    state_dir.mkdir(parents=True)
    (state_dir / "review.json").write_text(
        json.dumps({"status": "pass", "content_hash": "stale-hash"}),
        encoding="utf-8",
    )
    (state_dir / "enabled.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")

    result = preflight(
        osworld_root=osworld,
        ouroboros_url="http://127.0.0.1:9",
        osworld_server_url="http://127.0.0.1:9",
        unix_computer_use_payload=payload,
        unix_computer_use_state_dir=state_dir,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        data_root=data_root,
    )

    assert result["ok"] is False
    assert any("review_stale" in failure for failure in result["failures"])

def test_osworld_preflight_rejects_nonisolated_unix_computer_use_state(tmp_path):
    from devtools.benchmarks.osworld.osworld_adapter_skeleton import preflight
    from ouroboros.skill_loader import compute_content_hash

    osworld = tmp_path / "OSWorld"
    osworld.mkdir()
    (osworld / "evaluation_examples").mkdir()
    payload = tmp_path / "unix_computer_use"
    payload.mkdir()
    (payload / "SKILL.md").write_text("# unix_computer_use\n", encoding="utf-8")
    content_hash = compute_content_hash(payload)
    state_dir = tmp_path / "live-state" / "skills" / "unix_computer_use"
    state_dir.mkdir(parents=True)
    (state_dir / "review.json").write_text(
        json.dumps({"status": "pass", "content_hash": content_hash}),
        encoding="utf-8",
    )
    (state_dir / "enabled.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    (state_dir / "grants.json").write_text(json.dumps({"missing_grants": []}), encoding="utf-8")

    result = preflight(
        osworld_root=osworld,
        ouroboros_url="http://127.0.0.1:9",
        osworld_server_url="http://127.0.0.1:9",
        unix_computer_use_payload=payload,
        unix_computer_use_state_dir=state_dir,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        data_root=tmp_path / "isolated-data",
    )

    assert result["ok"] is False
    assert any("under isolated data root" in failure for failure in result["failures"])

def test_osworld_cli_default_repo_root_blocks_repo_internal_output(tmp_path, monkeypatch):
    import devtools.benchmarks.osworld.osworld_adapter_skeleton as osworld_adapter

    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    for path in (repo_root, data_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    monkeypatch.setattr(osworld_adapter, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(osworld_adapter, "DEFAULT_DATA_ROOT", data_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "osworld_adapter_skeleton.py",
            # This test pins OUTPUT ISOLATION, not seed provenance: its repo_root is a bare
            # directory with no git identity, so the v6.75.0 clean-seed gate would refuse
            # first and mask what is under test. The gate itself is covered separately
            # (test_benchmark_manifest_seed_gate_fails_closed_by_default) against a real repo.
            "--allow-dirty-seed",
            "--osworld-root",
            str(osworld),
            "--osworld-server-url",
            "http://127.0.0.1:9",
            "--unix-computer-use-payload",
            str(payload),
            "--output-root",
            str(repo_root / "bad-output"),
        ],
    )

    assert osworld_adapter.main() == 2
    assert not (repo_root / "bad-output" / "osworld_preflight.ledger.jsonl").exists()

def test_osworld_cli_omitted_data_root_defaults_to_output_isolation(tmp_path, monkeypatch):
    import devtools.benchmarks.osworld.osworld_adapter_skeleton as osworld_adapter

    repo_root = tmp_path / "repo"
    live_data_root = tmp_path / "live-data"
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    output_root = tmp_path / "runs" / "osworld"
    for path in (repo_root, live_data_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    monkeypatch.setattr(osworld_adapter, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(osworld_adapter, "DEFAULT_DATA_ROOT", live_data_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "osworld_adapter_skeleton.py",
            # This test pins OUTPUT ISOLATION, not seed provenance: its repo_root is a bare
            # directory with no git identity, so the v6.75.0 clean-seed gate would refuse
            # first and mask what is under test. The gate itself is covered separately
            # (test_benchmark_manifest_seed_gate_fails_closed_by_default) against a real repo.
            "--allow-dirty-seed",
            "--osworld-root",
            str(osworld),
            "--osworld-server-url",
            "http://127.0.0.1:9",
            "--unix-computer-use-payload",
            str(payload),
            "--output-root",
            str(output_root),
        ],
    )

    assert osworld_adapter.main() == 2
    manifest = json.loads((output_root / "osworld_preflight.run_manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["isolated_data_root"]) == output_root / "isolated_data"
    assert not str(manifest["isolated_data_root"]).startswith(str(live_data_root))

def test_osworld_cli_rejects_explicit_live_data_root(tmp_path, monkeypatch):
    import devtools.benchmarks.osworld.osworld_adapter_skeleton as osworld_adapter

    repo_root = tmp_path / "repo"
    live_data_root = tmp_path / "data"
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    output_root = tmp_path / "runs" / "osworld"
    for path in (repo_root, live_data_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    monkeypatch.setattr(osworld_adapter, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(osworld_adapter, "DEFAULT_DATA_ROOT", live_data_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "osworld_adapter_skeleton.py",
            # This test pins OUTPUT ISOLATION, not seed provenance: its repo_root is a bare
            # directory with no git identity, so the v6.75.0 clean-seed gate would refuse
            # first and mask what is under test. The gate itself is covered separately
            # (test_benchmark_manifest_seed_gate_fails_closed_by_default) against a real repo.
            "--allow-dirty-seed",
            "--osworld-root",
            str(osworld),
            "--osworld-server-url",
            "http://127.0.0.1:9",
            "--unix-computer-use-payload",
            str(payload),
            "--output-root",
            str(output_root),
            "--data-root",
            str(live_data_root),
        ],
    )

    assert osworld_adapter.main() == 2
    rows = [json.loads(line) for line in (output_root / "osworld_preflight.ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "live Ouroboros data root" in rows[0]["error"]

def test_osworld_step_shell_action_uses_temp_script_without_raw_pkill_pattern():
    from devtools.benchmarks.osworld.run_step_agent import _shell_action

    rendered = _shell_action("pkill -f chromium || true", timeout=12)

    assert "base64.b64decode" in rendered
    assert "pkill -f chromium" not in rendered
    assert "NamedTemporaryFile" in rendered
    assert "subprocess.run(['/bin/bash', script_path]" in rendered

def test_osworld_step_prompt_carries_image_and_in_app_done_guidance(tmp_path):
    from devtools.benchmarks.osworld.run_step_agent import OuroborosStepAgent

    agent = OuroborosStepAgent(
        ouroboros_bin="ouroboros",
        ouroboros_url="http://127.0.0.1:8765",
        repo_dir=tmp_path,
        data_dir=tmp_path,
        settings_path=tmp_path / "settings.json",
        result_dir=tmp_path,
        task_id="task",
        model="anthropic/claude-opus-4-7",
        timeout_sec=1,
        max_obs_chars=2000,
        screenshot_check_only=False,
    )
    prompt = agent._prompt(
        "Use LibreOffice Calc to make a pivot table",
        {"accessibility_tree": "<desktop-frame/>"},
        "/tmp/step.png",
        max_steps=50,
    )

    assert "screenshot is attached" in prompt
    assert "step 0 of at most 50" in prompt
    assert "In app-named tasks, work in the named app first" in prompt
    assert "Use done only after independently checking" in prompt
    assert "Cross-step notes" in prompt

def test_osworld_step_predict_attaches_screenshot(tmp_path, monkeypatch):
    from devtools.benchmarks.osworld.run_step_agent import OuroborosStepAgent

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout='{"response":"wait","notes":"remember","actions":[{"type":"wait"}]}', stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    agent = OuroborosStepAgent(
        ouroboros_bin="ouroboros",
        ouroboros_url="http://127.0.0.1:9999",
        repo_dir=tmp_path,
        data_dir=tmp_path / "data",
        settings_path=tmp_path / "settings.json",
        result_dir=tmp_path,
        task_id="task",
        model="anthropic/claude-opus-4-7",
        timeout_sec=1,
        max_obs_chars=2000,
        screenshot_check_only=False,
    )
    response, actions, debug = agent.predict("look", {"screenshot": b"png", "accessibility_tree": ""}, max_steps=3)

    assert response == "wait"
    assert actions == ["WAIT"]
    assert "--attach" in calls["cmd"]
    assert "http://127.0.0.1:9999" in calls["cmd"]
    assert debug["screenshot_upload_path"].endswith("step_001.png")
    assert agent.notes == ["remember"]
