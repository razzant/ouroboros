from __future__ import annotations

import asyncio
import contextlib
import io
import inspect
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools.benchmarks.common.official_commands import programbench_eval_cmd, swebench_eval_cmd
from devtools.benchmarks.osworld.normalize_logs import normalize_bundle
from devtools.benchmarks.common.manifests import benchmark_run_manifest, repo_provenance
from devtools.benchmarks.programbench.programbench_adapter import (
    build_ouroboros_task_body,
    create_submission_tarball,
    preflight_cleanroom_container,
)
from devtools.benchmarks.swe_bench.presets import resolve_preset


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASH_CAPTURE_AVAILABLE = sys.platform != "win32" and shutil.which("bash") is not None


@pytest.fixture(autouse=True)
def _isolate_bench_runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_BENCH_RUNS_ROOT", str(tmp_path / "bench_runs"))


def _git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "app.py").write_text("print('base')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_runtime_core_does_not_import_devtools():
    runtime_paths = [REPO_ROOT / "ouroboros", REPO_ROOT / "server.py"]
    offenders: list[str] = []
    for root in runtime_paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "import devtools" in text or "from devtools" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders


def test_official_command_builders_do_not_replace_scoring():
    # The builders stringify the Path via str(); compare against the platform
    # spelling so the argv-structure assertion stays valid on Windows too
    # (str(Path("/runs/pb")) == "\\runs\\pb" there).
    pb_run = str(Path("/runs/pb"))
    preds = str(Path("/runs/predictions.jsonl"))
    assert programbench_eval_cmd(Path("/runs/pb")) == ["programbench", "eval", pb_run]
    assert swebench_eval_cmd("princeton-nlp/SWE-bench_Verified", Path("/runs/predictions.jsonl"), "ouroboros", 2) == [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "princeton-nlp/SWE-bench_Verified",
        "--predictions_path",
        preds,
        "--max_workers",
        "2",
        "--run_id",
        "ouroboros",
    ]


def test_benchmark_manifest_records_provenance_without_diff_text(tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")

    provenance = repo_provenance(repo)
    manifest = benchmark_run_manifest(
        benchmark="unit",
        run_root=tmp_path / "run",
        repo_dir=repo,
        requested_task_ids=["task-1"],
        metadata={"argv": ["bench", "--task", "task-1"]},
    )

    assert provenance["dirty"] is True
    assert provenance["tracked_diff_sha256"]
    assert "print('changed')" not in json.dumps(provenance)
    assert manifest["requested_count"] == 1
    assert manifest["source"]["tracked_diff_sha256"]


def test_benchmark_common_helpers_keep_compact_api_surface():
    from devtools.benchmarks.common.result_index import task_result_row

    manifest_params = inspect.signature(benchmark_run_manifest).parameters
    row_params = inspect.signature(task_result_row).parameters

    assert len(manifest_params) <= 8
    assert len(row_params) <= 8


def test_benchmark_manifest_model_slots_cover_runtime_model_settings():
    from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS
    from ouroboros.config import SETTINGS_DEFAULTS

    # These match the OUROBOROS_MODEL* prefix but are a concurrency CAP / slot-wait
    # CEILING, not model-id slots, so they are not part of the model-slot manifest.
    _non_model_slot = {"OUROBOROS_MODEL_MAX_CONCURRENCY", "OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC"}
    relevant = {
        key
        for key in SETTINGS_DEFAULTS
        if key not in _non_model_slot
        and (
            key.startswith("OUROBOROS_MODEL")
            or key in {"CLAUDE_CODE_MODEL", "OUROBOROS_WEBSEARCH_MODEL", "OUROBOROS_REVIEW_MODELS"}
            or key.startswith("OUROBOROS_SCOPE_REVIEW_MODEL")
        )
    }

    assert relevant.issubset(set(MODEL_SLOT_KEYS))


def test_benchmark_default_paths_derive_from_workspace_root(monkeypatch):
    from devtools.benchmarks.common import run_roots
    from devtools.benchmarks.common import secrets

    monkeypatch.delenv("OUROBOROS_BENCH_RUNS_ROOT", raising=False)
    monkeypatch.delenv("OUROBOROS_SETTINGS_PATH", raising=False)

    workspace = REPO_ROOT.parent
    assert run_roots.DEFAULT_BENCH_RUNS_ROOT == workspace / "bench_runs"
    assert run_roots.default_settings_path() == workspace / "data" / "settings.json"
    assert secrets.settings_path() == workspace / "data" / "settings.json"


def test_benchmark_manifest_explicit_falsy_kwargs_override_metadata(tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)

    manifest = benchmark_run_manifest(
        benchmark="unit",
        run_root=tmp_path / "run",
        repo_dir=repo,
        requested_task_ids=["task-1"],
        argv=[],
        dataset="",
        isolated_data_root="",
        metadata={"argv": ["stale"], "dataset": "stale-ds", "isolated_data_root": "/tmp/stale"},
    )

    assert manifest["argv"] == []
    assert manifest["dataset"] == ""
    assert manifest["isolated_data_root"] == ""


def test_task_result_row_explicit_falsy_kwargs_override_metadata():
    from devtools.benchmarks.common.result_index import task_result_row

    row = task_result_row(
        benchmark="unit",
        instance_id="task-1",
        status="failed",
        reason_code="",
        prediction_written=False,
        official_eval_status="not_run",
        error="",
        metadata={
            "reason_code": "stale_success",
            "prediction_written": True,
            "official_eval_status": "completed",
            "error": "stale",
        },
    )

    assert row["reason_code"] == ""
    assert row["prediction_written"] is False
    assert row["official_eval_status"] == "not_run"
    assert row["error"] == ""


def test_pyproject_does_not_package_devtools_runtime_assets():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"devtools*"' not in pyproject
    assert "devtools = [" not in pyproject
    assert '"benchmarks/**/*.sh"' not in pyproject
    assert '"benchmarks/**/*.md"' not in pyproject


def test_executable_devtools_entrypoints_support_direct_help():
    scripts = [
        "devtools/benchmarks/programbench/run_programbench.py",
        "devtools/benchmarks/harness_bench_fast/ouroboros_cli_wrapper.py",
        "devtools/benchmarks/terminal_bench/run_harbor_smoke.py",
        "devtools/benchmarks/terminal_bench/run_tb.py",
        "devtools/benchmarks/swe_bench/swebench_predictions.py",
        "devtools/benchmarks/swe_bench_pro/grade_pro.py",
        "devtools/benchmarks/swe_bench_pro/pro_predictions.py",
        "devtools/benchmarks/swe_bench_pro/e1v2/auto_run.py",
        "devtools/benchmarks/swe_bench_pro/e1v2/build_predictions.py",
        "devtools/benchmarks/swe_bench_pro/e1v2/plot_e1v2_curves.py",
        "devtools/benchmarks/swe_bench_pro/e1v2/run_pro.py",
        "devtools/benchmarks/gaia/run_gaia.py",
        "devtools/benchmarks/gaia/score_gaia.py",
        "devtools/benchmarks/osworld/normalize_logs.py",
        "devtools/benchmarks/osworld/osworld_adapter_skeleton.py",
        "devtools/benchmarks/osworld/run_step_agent.py",
    ]
    for rel in scripts:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / rel), "--help"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        assert proc.returncode == 0, f"{rel} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        assert "usage:" in proc.stdout.lower()


def test_harness_bench_fast_wrapper_builds_ouroboros_run_command():
    # The upgraded harness-bench-fast wrapper builds the `ouroboros run` command inline in
    # main() (per-task logs, retries, --result-json-out, --start). Verify the command shape
    # and the v6.39 Phase-2 slot rename (HEAVY/FALLBACKS, never the legacy CODE/FALLBACK).
    from devtools.benchmarks.harness_bench_fast import ouroboros_cli_wrapper as w

    assert hasattr(w, "main")
    src = (
        REPO_ROOT / "devtools" / "benchmarks" / "harness_bench_fast" / "ouroboros_cli_wrapper.py"
    ).read_text(encoding="utf-8")
    for token in ('"run",', '"--memory-mode",', '"--quiet",', '"--result-json-out",', '"--actor-id",'):
        assert token in src, token
    assert '"OUROBOROS_MODEL_HEAVY": args.model' in src
    assert "OUROBOROS_MODEL_CODE" not in src


def test_swe_pro_e1v2_port_has_csv_option_a_heal_and_no_secrets():
    e1v2 = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2"
    csv_path = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "task_order_pro_70.csv"

    assert csv_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 71
    entrypoint = (e1v2 / "entrypoint_pro.sh").read_text(encoding="utf-8")
    # NW-7 (nq10): the harness-side Option A heal is restored so a dangling
    # committed evolution transaction from the previous task does not poison
    # enqueue for all subsequent tasks (E1v2 -> E1) on agents whose core lacks
    # boot reconciliation. It must keep its merge-base reachability guard so a
    # rolled-back commit is ABANDONED, not falsely marked absorbed. With a
    # newer core's own boot reconciliation it is a harmless no-op.
    assert "Option A:" in entrypoint
    assert "merge-base" in entrypoint and "--is-ancestor" in entrypoint
    assert "boot reconciliation" in entrypoint  # documents the no-op interaction
    assert "/opt/ouroboros-ro/devtools/benchmarks/swe_bench_pro/capture_patch.sh" in entrypoint
    assert '"/opt/capture_patch.sh"' not in (e1v2 / "run_pro.py").read_text(encoding="utf-8")
    assert 'post-task evolution=disabled baseline' in entrypoint
    assert 'reason":"evolution_disabled' in entrypoint
    assert 'if [ "${OBO_SELFIMPROVE:-0}" = "1" ]' in entrypoint
    assert "view_image" in entrypoint
    # owner_chat_id must be seeded BEFORE the budget reset (else native
    # post-task evolution is dropped on fresh volumes -> E1v2 silently == E0).
    assert entrypoint.index('printf \'{"owner_chat_id": 1}\'') < entrypoint.index('reset_per_task_budget("/obo-data"')
    for name in ("settings_base.json", "_run_settings.example.json"):
        payload = json.loads((e1v2 / name).read_text(encoding="utf-8"))
        for key, value in payload.items():
            if any(token in key for token in ("API_KEY", "TOKEN", "PASSWORD", "CREDENTIAL")):
                assert value in ("", None, False), (name, key)
        if name == "settings_base.json":
            assert payload["OUROBOROS_TASK_REVIEW_MODE"] == "required"
            assert payload["OUROBOROS_POST_TASK_EVOLUTION"] == "false"

    from ouroboros.config import SETTINGS_DEFAULTS

    assert SETTINGS_DEFAULTS["OUROBOROS_TASK_REVIEW_MODE"] == "auto"
    run_pro = (e1v2 / "run_pro.py").read_text(encoding="utf-8")
    assert "default fixed-model baseline" in run_pro
    assert "default E1v2 (post-task evolution on)" not in run_pro


def test_swe_pro_e1v2_curve_rows(tmp_path):
    from devtools.benchmarks.swe_bench_pro.e1v2.plot_e1v2_curves import curve_rows, load_e0, load_e1v2_results

    csv_path = tmp_path / "order.csv"
    csv_path.write_text("idx,instance_id,verdict\n1,a,pass\n2,b,fail\n", encoding="utf-8")
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"instance_id":"a","resolved":false}\n{"instance_id":"b","resolved":true}\n', encoding="utf-8")

    rows = curve_rows(load_e0(csv_path), load_e1v2_results(results_path), window=2)

    assert rows[-1]["e0_window_rate"] == 0.5
    assert rows[-1]["e1v2_window_rate"] == 0.5


def test_gaia_adapter_wires_settings_and_solver(tmp_path):
    import types
    import devtools.benchmarks.gaia.run_gaia as run_gaia
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    base_settings_path = REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "settings_base.json"
    settings_path = run_gaia._render_run_settings(base_settings_path, "openai/gpt-5.5", tmp_path)
    env = run_gaia._settings_env(settings_path, "google/gemini-2.5-pro", tmp_path)
    assert env["OUROBOROS_SETTINGS_PATH"] == str(settings_path)
    assert env["OUROBOROS_DATA_DIR"].startswith(str(tmp_path))
    assert env["OUROBOROS_MODEL"] == "google/gemini-2.5-pro"
    assert json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == "openai/gpt-5.5"
    assert env["OUROBOROS_SCOPE_REVIEW_MODELS"] == "google/gemini-2.5-pro"
    assert env["OUROBOROS_TASK_REVIEW_MODE"] == "required"
    assert env.get("CLAUDE_CODE_MODEL") != "google/gemini-2.5-pro"
    assert env["GAIA_OUROBOROS_URL"].startswith("http://127.0.0.1:")
    for key in run_gaia._GAIA_PINNED_MODEL_KEYS:
        if key.startswith("OUROBOROS_EFFORT_"):
            continue
        assert env[key]
    assert env.get("OUROBOROS_WEBSEARCH_MODEL") != "google/gemini-2.5-pro"

    argv = run_gaia.build_inspect_argv(
        types.SimpleNamespace(split="validation", level=1, limit=1),
        tmp_path,
    )
    assert any("ouroboros_solver.py@ouroboros_solver" in part for part in argv)
    assert "inspect_evals/gaia" in argv
    assert "subset=2023_level1" in argv
    assert "--log-format" in argv and "json" in argv
    assert callable(ouroboros_solver.ouroboros_solver())
    args = types.SimpleNamespace(split="validation", level=1, limit=3, solve_model="google/gemini-2.5-pro")
    run_gaia._write_manifest(tmp_path, args, argv, settings_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["official_command"] == argv
    assert manifest["requested_count"] == 3
    assert manifest["model_slots"]["OUROBOROS_MODEL"] == "google/gemini-2.5-pro"
    assert "web_search" in open(REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "inspect_solver" / "ouroboros_solver.py", encoding="utf-8").read()
    assert "claude_code_edit" in open(REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "inspect_solver" / "ouroboros_solver.py", encoding="utf-8").read()


def test_gaia_sanitized_env_keeps_only_needed_provider_key(monkeypatch):
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    monkeypatch.setenv("OPENROUTER_API_KEY", "router")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("GITHUB_TOKEN", "github")
    monkeypatch.setenv("OUROBOROS_MODEL", "host/model")
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")

    env = run_gaia._sanitized_host_env("google/gemini-2.5-pro")

    assert env["OPENROUTER_API_KEY"] == "router"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "OUROBOROS_MODEL" not in env
    assert "USE_LOCAL_MAIN" not in env


def test_gaia_sanitized_env_preserves_keys_for_all_model_knobs(monkeypatch):
    # Config A: anthropic main + gpt-4o vision -> BOTH provider keys must survive,
    # else the vision route cannot authenticate.
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router")

    env = run_gaia._sanitized_host_env("anthropic::claude-sonnet-4.5", "openai::gpt-4o", "")
    assert env["ANTHROPIC_API_KEY"] == "anthropic"  # solve model
    assert env["OPENAI_API_KEY"] == "openai"  # vision model — preserved (the fix)


def test_gaia_credential_keys_tolerate_leading_whitespace():
    # A "a, b"-split review-model list leaves leading spaces; the provider match must
    # still resolve the right credential keys (not silently fall through to OpenRouter).
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    assert "ANTHROPIC_API_KEY" in run_gaia._credential_keys_for_model(" anthropic::claude-sonnet-4.5")
    assert "OPENAI_API_KEY" in run_gaia._credential_keys_for_model("openai::gpt-4o ")


def test_gaia_sanitized_env_preserves_pinned_websearch_backend_key(monkeypatch):
    # Config C: opus solve (anthropic key) + 'openai' web_search backend -> the OpenAI key
    # is unrelated to any model but must survive, else web_search cannot authenticate.
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router")

    env = run_gaia._sanitized_host_env("anthropic::claude-opus-4.8", websearch_backend="openai")
    assert env["ANTHROPIC_API_KEY"] == "anthropic"  # solve model
    assert env["OPENAI_API_KEY"] == "openai"  # pinned web_search backend — preserved

    # ddgs pin needs no provider key (pure retrieval).
    env_ddgs = run_gaia._sanitized_host_env("anthropic::claude-opus-4.8", websearch_backend="ddgs")
    assert "OPENAI_API_KEY" not in env_ddgs


def test_gaia_openai_websearch_pin_drops_base_url(monkeypatch):
    # Official OpenAI web_search is disabled when OPENAI_BASE_URL is set, so an 'openai'
    # web pin must drop it EVEN when an openai:: model would otherwise carry it.
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compat.example/v1")

    env = run_gaia._sanitized_host_env("openai::gpt-5.5", websearch_backend="openai")
    assert env["OPENAI_API_KEY"] == "openai"
    assert "OPENAI_BASE_URL" not in env  # dropped so official web_search stays enabled


def test_gaia_render_injects_keys_and_free_host_service_port(tmp_path, monkeypatch):
    # Out-of-the-box coexistence with a running desktop app: the rendered settings must
    # carry a FREE Host-Service port (not the default 8767) and the REAL provider key for
    # the configured model (empty placeholders would be popped by apply_settings_to_env,
    # erasing the env keys -> "No supported provider configured").
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")  # resolved first, before data/settings.json
    base = REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "settings_base.json"

    hsp = run_gaia._free_port()
    assert hsp not in (8765, 8767) and 1024 < hsp < 65536  # a usable free port, not the app's

    # Pin ddgs so only the model's provider (OpenRouter, for the slash-format gemini) is
    # needed — 'auto' would deliberately pull every available key for the web cascade.
    out = run_gaia._render_run_settings(
        base, "google/gemini-2.5-pro", tmp_path, websearch_backend="ddgs", host_service_port=hsp,
    )
    s = json.loads(out.read_text(encoding="utf-8"))
    assert s["OPENROUTER_API_KEY"] == "test-or-key"  # injected (gemini slash -> OpenRouter route)
    assert s["OUROBOROS_HOST_SERVICE_PORT"] == hsp  # free port, avoids the live desktop app
    # Only the NEEDED provider is injected — an unused provider's placeholder stays empty.
    assert not str(s.get("ANTHROPIC_API_KEY", "")).strip()


def test_gaia_settings_env_filters_custom_settings_secrets(tmp_path):
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "OPENROUTER_API_KEY": "from-settings",
        "GITHUB_TOKEN": "gh",
        "ANTHROPIC_API_KEY": "anthropic",
        "OUROBOROS_MODEL": "host/model",
    }), encoding="utf-8")

    env = run_gaia._settings_env(settings, "google/gemini-2.5-pro", tmp_path)

    assert "OPENROUTER_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["OUROBOROS_MODEL"] == "google/gemini-2.5-pro"


def test_gaia_score_parses_inspect_json_logs(tmp_path):
    from devtools.benchmarks.gaia.score_gaia import summarize

    log_dir = tmp_path / "inspect_logs"
    log_dir.mkdir()
    (log_dir / "sample.json").write_text(json.dumps({
        "samples": [
            {
                "output": {"completion": " FINAL ANSWER: 42 "},
                "scores": {"gaia_scorer": {"value": True}},
            },
            {
                "output": {"completion": "wrong"},
                "scores": {"gaia_scorer": {"value": False}},
            },
            {
                "output": {"completion": "string correct"},
                "scores": {"gaia_scorer": {"value": "C"}},
            },
            {
                "output": {"completion": "string incorrect"},
                "scores": {"gaia_scorer": {"value": "I"}},
            },
        ]
    }), encoding="utf-8")

    summary = summarize(tmp_path)
    assert summary["official_scored"] == 4
    assert summary["official_correct"] == 2
    assert summary["official_accuracy"] == 0.5


def test_gaia_score_prefers_official_eval_rows_when_result_json_exists(monkeypatch, tmp_path):
    import devtools.benchmarks.gaia.score_gaia as score_gaia

    sample_dir = tmp_path / "samples" / "s1"
    sample_dir.mkdir(parents=True)
    (sample_dir / "result.json").write_text(json.dumps({"final_answer": "local only"}), encoding="utf-8")
    monkeypatch.setattr(score_gaia, "_rows_from_eval_logs", lambda _root: [{
        "path": "official.eval",
        "raw_answer": "official",
        "local_normalized": "official",
        "official_score": True,
    }])

    summary = score_gaia.summarize(tmp_path)

    assert summary["official_scored"] == 1
    assert summary["official_correct"] == 1


def test_gaia_solver_disable_tools_before_prompt(monkeypatch, tmp_path):
    from ouroboros import cli
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        result_path = tmp_path / "samples" / "sample" / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"final_answer": "ok"}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("GAIA_OUROBOROS_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "ouroboros_data"))
    monkeypatch.setattr(ouroboros_solver.subprocess, "run", fake_run)
    result = ouroboros_solver.run_ouroboros("question", sample_id="sample")
    assert result["final_answer"] == "ok"
    parser = cli.build_parser()
    ns = parser.parse_args(seen["cmd"][3:])
    assert ns.disable_tools == ["web_search,claude_code_edit"]
    assert ns.result_json_out
    # The prompt is the question plus the official GAIA "FINAL ANSWER:" protocol suffix.
    assert ns.prompt and ns.prompt[0].startswith("question")
    assert "FINAL ANSWER:" in ns.prompt[0]


def test_gaia_solver_retries_transient_supervisor_startup(monkeypatch, tmp_path):
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    calls = {"count": 0}

    def fake_run(cmd, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(returncode=2, stdout="", stderr="error: HTTP 503: supervisor is still starting")
        result_path = tmp_path / "samples" / "sample" / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"final_answer": "ok"}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("GAIA_OUROBOROS_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(ouroboros_solver.subprocess, "run", fake_run)
    monkeypatch.setattr(ouroboros_solver.time, "sleep", lambda _seconds: None)

    result = ouroboros_solver.run_ouroboros("question", sample_id="sample")

    assert calls["count"] == 2
    assert result["final_answer"] == "ok"


def test_gaia_solver_extracts_shared_files_and_denies_secrets(monkeypatch, tmp_path):
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    shared = tmp_path / "shared_files"
    shared.mkdir()
    image = shared / "chart.png"
    image.write_bytes(b"png")
    secret_dir = tmp_path / ".ssh"
    secret_dir.mkdir()
    secret = secret_dir / "id_rsa"
    secret.write_text("secret", encoding="utf-8")
    monkeypatch.setenv("GAIA_SHARED_FILES_ROOT", str(shared))
    sample_dir = tmp_path / "run" / "samples" / "s1"
    state = SimpleNamespace(metadata={"attachments": [str(secret)]})

    attachments = ouroboros_solver._attachment_paths_from_state(
        state,
        sample_dir,
        "Use /shared_files/chart.png to answer.",
    )

    assert len(attachments) == 1
    assert attachments[0].name.startswith("chart-")
    assert attachments[0].read_bytes() == b"png"


def test_gaia_attachment_reads_files_dict_keys(monkeypatch, tmp_path):
    # GAIA's TaskState.files maps a SANDBOX path (key) -> host path (value); on this
    # inspect version the real host file is the KEY. Staging must read keys too.
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    host = tmp_path / "data.csv"
    host.write_text("a,b\n1,2\n", encoding="utf-8")
    sample_dir = tmp_path / "run" / "samples" / "s1"
    state = SimpleNamespace(files={str(host): "/sandbox/data.csv"})  # host path is the KEY

    attachments = ouroboros_solver._attachment_paths_from_state(state, sample_dir, "")
    assert len(attachments) == 1
    assert attachments[0].read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_gaia_solver_isolates_generic_subprocess_error(monkeypatch, tmp_path):
    # Crash isolation: a non-timeout spawn/OS failure must become a terminal per-sample
    # result, never propagate and abort the whole eval.
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    def boom(cmd, **kwargs):
        raise OSError("posix_spawn failed")

    monkeypatch.setenv("GAIA_OUROBOROS_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(ouroboros_solver.subprocess, "run", boom)

    result = ouroboros_solver.run_ouroboros("question", sample_id="sample")
    assert result["returncode"] == -1
    assert result["final_answer"] == ""
    assert "SUBPROCESS ERROR" in result["stderr_tail"]


def test_programbench_task_body_sets_executor_and_protected_policy(tmp_path):
    workspace = tmp_path / "workspace"
    _git_repo(workspace)

    body = build_ouroboros_task_body(
        instruction="solve",
        workspace_host_path=workspace,
        container_name="pb-cleanroom",
        protected_backend_paths=["/workspace/executable"],
    )

    assert body["allowed_resources"] == {"web": False, "network": False, "internet": False}
    assert body["actor_id"] == "programbench"
    assert body["source"] == "programbench"
    assert "actor_id" not in body["metadata"]
    assert body["executor_ref"]["type"] == "docker_exec"
    assert body["executor_ref"]["network"] == "none"
    protected = body["resource_policy"]["protected_artifacts"][0]
    assert protected["role"] == "black_box_reference"
    assert protected["allow"] == ["execute"]
    assert {"read_bytes", "hash", "static_introspection", "dynamic_trace", "debug"} <= set(protected["deny"])


def test_programbench_git_workspace_does_not_commit_protected_reference(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_text("protected-bytes\n", encoding="utf-8")

    build_ouroboros_task_body(
        instruction="solve",
        workspace_host_path=workspace,
        container_name="pb-cleanroom",
        protected_backend_paths=["/workspace/executable"],
    )

    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    show = subprocess.run(["git", "show", "HEAD:executable"], cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert head.returncode != 0
    assert show.returncode != 0


def test_programbench_submission_tarball_excludes_repo_noise(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "HEAD").write_text("ref\n", encoding="utf-8")
    (workspace / ".ouroboros").mkdir()
    (workspace / ".ouroboros" / "trace.json").write_text("{}\n", encoding="utf-8")
    (workspace / "node_modules" / "pkg").mkdir(parents=True)
    (workspace / "node_modules" / "pkg" / "index.js").write_text("junk\n", encoding="utf-8")
    (workspace / "build").mkdir()
    (workspace / "build" / "out.o").write_text("junk\n", encoding="utf-8")
    (workspace / "dist").mkdir()
    (workspace / "dist" / "bundle.js").write_text("junk\n", encoding="utf-8")
    (workspace / "executable").write_text("protected\n", encoding="utf-8")
    (workspace / "solution.py").write_text("print('ok')\n", encoding="utf-8")

    tar_path = create_submission_tarball(
        workspace,
        tmp_path / "submission.tar.gz",
        protected_paths=["/workspace/executable", "executable"],
    )

    with tarfile.open(tar_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "solution.py" in names
    assert ".git/HEAD" not in names
    assert ".ouroboros/trace.json" not in names
    assert "node_modules/pkg/index.js" not in names
    assert "build/out.o" not in names
    assert "dist/bundle.js" not in names
    assert "executable" not in names


def test_programbench_instance_path_stays_under_run_root(tmp_path):
    from devtools.benchmarks.common.run_roots import safe_join_under

    root = tmp_path / "programbench-run"
    assert safe_join_under(root, "cheat/cheat") == root.resolve(strict=False) / "cheat" / "cheat"
    with pytest.raises(ValueError, match="escapes run root"):
        safe_join_under(root, "../escape")
    with pytest.raises(ValueError, match="escapes run root"):
        safe_join_under(root, "/tmp/escape")


def test_programbench_cleanroom_preflight_requires_task_cleanroom_and_no_network(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps([
                {
                    "Config": {"Image": "ghcr.io/facebookresearch/programbench/foo:task_cleanroom"},
                    "HostConfig": {"NetworkMode": "none"},
                }
            ]),
            stderr="",
        )

    import devtools.benchmarks.programbench.programbench_adapter as adapter

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    assert preflight_cleanroom_container("pb") == {
        "image": "ghcr.io/facebookresearch/programbench/foo:task_cleanroom",
        "network": "none",
    }
    assert calls[0][:2] == ["docker", "inspect"]


def test_programbench_preflight_failure_writes_blocker_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    output = tmp_path / "programbench-ledger.jsonl"
    manifest = tmp_path / "programbench-manifest.json"
    monkeypatch.setattr(
        run_programbench,
        "preflight_cleanroom_container",
        lambda _: (_ for _ in ()).throw(RuntimeError("docker missing")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench.py",
            "--workspace",
            str(workspace),
            "--instruction-file",
            str(instruction),
            "--container-name",
            "missing",
            "--instance-id",
            "case1",
            "--ledger-output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
    )

    with pytest.raises(RuntimeError, match="docker missing"):
        run_programbench.main()
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["status"] == "blocked"
    assert row["reason_code"] == "cleanroom_preflight_failed"
    assert manifest_json["requested_task_ids"] == ["case1"]


def test_programbench_submission_failure_writes_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    output = tmp_path / "programbench-ledger.jsonl"
    manifest = tmp_path / "programbench-manifest.json"
    monkeypatch.setattr(run_programbench, "preflight_cleanroom_container", lambda _: {"image": "task_cleanroom", "network": "none"})
    monkeypatch.setattr(
        run_programbench,
        "create_submission_tarball",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("tar failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench.py",
            "--workspace",
            str(workspace),
            "--instruction-file",
            str(instruction),
            "--container-name",
            "pb",
            "--instance-id",
            "case2",
            "--ledger-output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
    )

    with pytest.raises(RuntimeError, match="tar failed"):
        run_programbench.main()
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["reason_code"] == "submission_failed"
    assert row["official_eval_status"] == "not_run"
    assert manifest_json["requested_task_ids"] == ["case2"]
    assert manifest_json["extra"]["failure_reason_code"] == "submission_failed"


def test_programbench_official_eval_failure_writes_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    output = tmp_path / "programbench-ledger.jsonl"
    manifest = tmp_path / "programbench-manifest.json"
    submission = tmp_path / "submission.tar.gz"
    monkeypatch.setattr(run_programbench, "preflight_cleanroom_container", lambda _: {"image": "task_cleanroom", "network": "none"})
    monkeypatch.setattr(run_programbench, "create_submission_tarball", lambda *_args, **_kwargs: submission)
    monkeypatch.setattr(
        run_programbench,
        "run_official_eval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("eval failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench.py",
            "--workspace",
            str(workspace),
            "--instruction-file",
            str(instruction),
            "--container-name",
            "pb",
            "--instance-id",
            "case3",
            "--ledger-output",
            str(output),
            "--manifest-output",
            str(manifest),
            "--eval",
        ],
    )

    with pytest.raises(RuntimeError, match="eval failed"):
        run_programbench.main()
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["reason_code"] == "official_eval_failed"
    assert row["official_eval_status"] == "failed"
    assert manifest_json["requested_task_ids"] == ["case3"]
    assert manifest_json["extra"]["failure_reason_code"] == "official_eval_failed"


def test_swe_verified_preset_uses_official_dataset_name():
    assert resolve_preset("verified") == "princeton-nlp/SWE-bench_Verified"
    assert resolve_preset("SWE-bench/SWE-bench_Verified") == "princeton-nlp/SWE-bench_Verified"


def test_terminal_bench_harbor_adapter_is_optional_import():
    spec = importlib.util.spec_from_file_location(
        "tb_harbor_adapter",
        REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench" / "harbor_installed_agent.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OuroborosTerminalBenchAgent.name() == "Ouroboros Installed"


def test_terminal_bench_adapter_does_not_commit_target_workspace():
    adapter = (REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench" / "harbor_installed_agent.py").read_text(encoding="utf-8")
    assert "git add -A" not in adapter
    assert "git commit --allow-empty" not in adapter


def test_osworld_shell_action_does_not_fabricate_bash_history():
    """NW-6 methodology integrity: the OSWorld shell action must NOT write the
    command into ~/.bash_history to satisfy terminal-task evaluators (hidden
    verifier knowledge / answer fitting). The only allowed mention is the
    docstring documenting that we deliberately do not do it."""
    src = (REPO_ROOT / "devtools" / "benchmarks" / "osworld" / "run_step_agent.py").read_text(encoding="utf-8")
    # No history-file write in the emitted snippet, no record_history plumbing.
    assert "hist.open(" not in src
    assert "record_history" not in src
    assert ".bash_history'" not in src  # the f.write to the history path is gone


def test_terminal_bench_metadata_declares_all_assisting_models(monkeypatch):
    """NW-6: with task_review_mode=required the review triad (incl. a frontier
    model) assists the measured run; metadata.yaml must declare every assisting
    model, not only the measured one."""
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "tb_run_for_meta", REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench" / "run_tb.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(_sys.modules, spec.name, module)  # dataclass field resolution needs this
    spec.loader.exec_module(module)
    monkeypatch.delenv("OUROBOROS_REVIEW_MODELS", raising=False)
    meta = module.leaderboard_metadata(
        agent_name="Ouroboros", org_name="Ouroboros",
        model="openai/gpt-5.5", light_model="google/gemini-3.5-flash")
    # The default review triad includes a frontier helper that must be visible.
    assert "anthropic/claude-opus-4.8" in meta
    assert "commit_review_triad" in meta
    assert meta.count("model_name:") >= 3


def test_terminal_bench_adapter_defaults_to_required_acceptance_review(tmp_path):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)
    env = agent._container_env()
    assert env["OUROBOROS_TASK_REVIEW_MODE"] == "required"
    assert env["OUROBOROS_MODEL_LIGHT"] == "google/gemini-3.5-flash"

    agent = tb_agent.OuroborosTerminalBenchAgent(
        logs_dir=tmp_path,
        task_review_mode="auto",
        ouroboros_model="openai/gpt-5.5",
        ouroboros_light_model="google/gemini-3.5-flash",
    )
    env = agent._container_env()
    assert env["OUROBOROS_TASK_REVIEW_MODE"] == "auto"
    assert env["OUROBOROS_MODEL"] == "openai/gpt-5.5"
    # v6.39 slot rename: the bulk lane is OUROBOROS_MODEL_HEAVY (legacy _CODE retired);
    # the container HEAVY lane reads os.environ["OUROBOROS_MODEL_HEAVY"], not _CODE.
    assert env["OUROBOROS_MODEL_HEAVY"] == "openai/gpt-5.5"
    assert env["OUROBOROS_MODEL_LIGHT"] == "google/gemini-3.5-flash"


def test_terminal_bench_source_copy_excludes_secret_shaped_files(tmp_path):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "module.py").write_text("print('ok')\n", encoding="utf-8")
    secret_names = (
        ".env",
        ".env.example",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "aws-credentials.json",
        "credentials.json",
        "gcp-service-account.json",
        "id_rsa",
        "openrouter.token.txt",
        "prod.env",
        "repo.bundle",
        "repo_bundle_manifest.json",
        "secrets.json",
        "service-account.json",
    )
    for name in secret_names:
        (source / name).write_text("secret\n", encoding="utf-8")
    (source / "cert.pem").write_text("secret\n", encoding="utf-8")
    (source / "python-standalone").mkdir()
    (source / "python-standalone" / "python").write_text("binary\n", encoding="utf-8")

    tb_agent._copy_clean_source(source, target)

    assert (target / "module.py").exists()
    for name in (*secret_names, "cert.pem", "python-standalone"):
        assert not (target / name).exists()


def test_terminal_bench_source_provenance_hashes_copied_tree(tmp_path):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    source = tmp_path / "source"
    clean = tmp_path / "clean"
    source.mkdir()
    (source / "module.py").write_text("print('v1')\n", encoding="utf-8")
    (source / "untracked.txt").write_text("copied\n", encoding="utf-8")
    tb_agent._copy_clean_source(source, clean)

    provenance = tb_agent._source_copy_provenance(source, clean)

    assert provenance["copy_policy"]["secret_shaped_file_copy_allowed"] is False
    assert provenance["copied_tree"]["files"] == 2
    assert provenance["copied_tree"]["sha256"]


def test_terminal_bench_network_preflight_uses_configured_provider(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    class Env:
        def __init__(self) -> None:
            self.command = ""

        async def exec(self, *, command, timeout_sec=None, env=None, cwd=None):
            self.command = command
            script = command.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            stdout = io.StringIO()
            code = 0
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(script, {})
            except SystemExit as exc:
                code = int(exc.code or 0)
            return SimpleNamespace(return_code=code, stdout=stdout.getvalue(), stderr="")

    from types import SimpleNamespace

    env = Env()
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)

    asyncio.run(agent._network_preflight(env, {"OPENAI_API_KEY": "sk-test"}))

    assert "api.openai.com" in env.command
    assert "openrouter.ai" not in env.command
    assert "urllib.error.HTTPError" in env.command
    assert "openai_preflight_status 401" in (tmp_path / "network-preflight.txt").read_text(encoding="utf-8")


def test_terminal_bench_openrouter_credit_preflight_blocks_low_credit(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"data":{"total_credits":10,"total_usage":9.75}}'

    def fake_urlopen(req, timeout=0):
        assert req.headers["Authorization"] == "Bearer or-key"
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, openrouter_min_credit_usd=1.0)

    with pytest.raises(RuntimeError, match="remaining \\$0.25 below threshold \\$1.00"):
        agent._openrouter_credit_preflight({})

    payload = json.loads((tmp_path / "openrouter-credit-preflight.json").read_text(encoding="utf-8"))
    assert payload["remaining_usd"] == 0.25


def test_run_ouroboros_task_terminal_nonzero_exit_is_not_interruption(tmp_path):
    """The in-container runner exits 2 to SIGNAL a terminal infra_failed result; that is a real
    terminal task outcome (status completed/failed), NOT a Harbor wall-clock interruption.
    _run_ouroboros_task must RETURN such a summary (so run() sets reached_terminal_result=True and
    the captured summary is not mislabeled captured_after_cancellation). A nonzero exit with NO
    terminal summary (a genuine runner crash) still raises."""
    import asyncio
    from types import SimpleNamespace
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)

    class _Env:
        def __init__(self, return_code, stdout):
            self._rc, self._out = return_code, stdout

        async def exec(self, *, command, timeout_sec=None, env=None, cwd=None):
            return SimpleNamespace(return_code=self._rc, stdout=self._out, stderr="")

    terminal = json.dumps(
        {"status": "failed", "reason_code": "provider_unavailable", "infra_failed": True, "return_code": 2}
    )
    out = asyncio.run(agent._run_ouroboros_task(_Env(2, terminal), {}))
    assert out["status"] == "failed" and out["reason_code"] == "provider_unavailable"

    with pytest.raises(RuntimeError):
        asyncio.run(agent._run_ouroboros_task(_Env(2, "Traceback: boom\nnot-json"), {}))


def test_terminal_bench_openrouter_credit_preflight_skips_when_unconfigured(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)

    agent._openrouter_credit_preflight({})

    assert not (tmp_path / "openrouter-credit-preflight.json").exists()


def test_terminal_bench_network_preflight_supports_openai_compatible(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    class Env:
        def __init__(self) -> None:
            self.command = ""

        async def exec(self, *, command, timeout_sec=None, env=None, cwd=None):
            self.command = command
            script = command.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            stdout = io.StringIO()
            code = 0
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(script, {})
            except SystemExit as exc:
                code = int(exc.code or 0)
            return SimpleNamespace(return_code=code, stdout=stdout.getvalue(), stderr="")

    env = Env()
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)

    asyncio.run(
        agent._network_preflight(
            env,
            {
                "OPENAI_COMPATIBLE_API_KEY": "sk-compatible",
                "OPENAI_COMPATIBLE_BASE_URL": "https://provider.example.invalid/v1",
            },
        )
    )

    assert "provider.example.invalid/v1/models" in env.command
    assert "openai_compatible_preflight_status 401" in (tmp_path / "network-preflight.txt").read_text(encoding="utf-8")


def test_terminal_bench_adapter_forwards_gigachat_and_preflights_direct_provider(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    monkeypatch.setenv("OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS", "1")
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "gigachat-test-credentials")
    monkeypatch.setenv("GIGACHAT_BASE_URL", "https://gigachat.example.invalid/api/v1")

    class Env:
        def __init__(self) -> None:
            self.command = ""

        async def exec(self, *, command, timeout_sec=None, env=None, cwd=None):
            self.command = command
            script = command.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            stdout = io.StringIO()
            code = 0
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(script, {})
            except SystemExit as exc:
                code = int(exc.code or 0)
            return SimpleNamespace(return_code=code, stdout=stdout.getvalue(), stderr="")

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)
    injected = agent._container_env()
    env = Env()

    asyncio.run(agent._network_preflight(env, injected))

    assert injected["GIGACHAT_CREDENTIALS"] == "gigachat-test-credentials"
    assert "gigachat.example.invalid/api/v1/models" in env.command
    assert "gigachat_preflight_status 401" in (tmp_path / "network-preflight.txt").read_text(encoding="utf-8")


def test_terminal_bench_adapter_refuses_container_secret_injection_by_default(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    monkeypatch.delenv("OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-container-secret")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)
    injected = agent._container_env()

    assert "OPENROUTER_API_KEY" not in injected
    with pytest.raises(RuntimeError, match="refuses to inject long-lived provider credentials"):
        agent._enforce_container_secret_policy(injected)


def test_terminal_bench_task_body_uses_top_level_actor_id():
    adapter = (REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench" / "harbor_installed_agent.py").read_text(encoding="utf-8")
    assert '"actor_id": "harbor-terminal-bench"' in adapter
    assert '"metadata": {{"source": "terminal-bench", "delegation_role": "root"}}' in adapter
    assert '"metadata": {{"actor_id": "harbor-terminal-bench"' not in adapter


@pytest.mark.skipif(not _BASH_CAPTURE_AVAILABLE, reason="capture_patch.sh is a POSIX shell helper; Python wrappers are covered separately")
def test_swe_pro_capture_keeps_untracked_text_and_drops_binary(tmp_path):
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    (repo / "new_file.py").write_text("print('new')\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[tool.example]\nvalue = true\n", encoding="utf-8")
    (repo / "setup.py").write_text("from setuptools import setup\nsetup()\n", encoding="utf-8")
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (repo / "poetry.lock").write_text("# lock\n", encoding="utf-8")
    (repo / "binary.bin").write_bytes(b"\x00\x01\x02\x03")
    (repo / "build").mkdir()
    (repo / "build" / "out.txt").write_text("junk\n", encoding="utf-8")
    (repo / "dist").mkdir()
    (repo / "dist" / "out.txt").write_text("junk\n", encoding="utf-8")
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    capture = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"
    out = tmp_path / "patch.diff"

    subprocess.run(["bash", str(capture), str(repo), base, str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    patch = out.read_text(encoding="utf-8")

    assert "new_file.py" in patch
    assert "pyproject.toml" in patch
    assert "setup.py" in patch
    assert "package-lock.json" not in patch
    assert "poetry.lock" in patch
    assert "app.py" in patch
    assert "binary.bin" not in patch
    assert "build/out.txt" not in patch
    assert "dist/out.txt" not in patch


@pytest.mark.skipif(not _BASH_CAPTURE_AVAILABLE, reason="capture_patch.sh is a POSIX shell helper; Python wrappers are covered separately")
def test_swe_pro_capture_preserves_pure_lockfile_patch(tmp_path):
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    capture = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"
    out = tmp_path / "patch.diff"

    subprocess.run(["bash", str(capture), str(repo), base, str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    patch = out.read_text(encoding="utf-8")

    assert "package-lock.json" in patch


@pytest.mark.skipif(not _BASH_CAPTURE_AVAILABLE, reason="capture_patch.sh is a POSIX shell helper; Python wrappers are covered separately")
def test_swe_pro_capture_requires_valid_base_and_external_output(tmp_path):
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    capture = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"

    missing_output = subprocess.run(["bash", str(capture), str(repo), base], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bad_base = subprocess.run(
        ["bash", str(capture), str(repo), "not-a-commit", str(tmp_path / "bad.diff")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    internal_output = REPO_ROOT / "devtools" / "should-not-write.diff"
    internal_dir = REPO_ROOT / "_test_rejected_capture_output_dir"
    nested_internal_output = internal_dir / "out.diff"
    shutil.rmtree(internal_dir, ignore_errors=True)
    try:
        repo_internal = subprocess.run(
            ["bash", str(capture), str(repo), base, str(internal_output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        nested_repo_internal = subprocess.run(
            ["bash", str(capture), str(repo), base, str(nested_internal_output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        internal_output.unlink(missing_ok=True)
        shutil.rmtree(internal_dir, ignore_errors=True)

    assert missing_output.returncode != 0
    assert bad_base.returncode != 0
    assert repo_internal.returncode != 0
    assert "outside the Ouroboros repo" in repo_internal.stderr
    assert nested_repo_internal.returncode != 0
    assert "outside the Ouroboros repo" in nested_repo_internal.stderr
    assert not internal_dir.exists()


def test_swe_pro_grade_runs_official_eval_with_raw_sample(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.grade_pro as grade_pro

    eval_repo = tmp_path / "SWE-bench_Pro-os"
    helper = eval_repo / "helper_code"
    helper.mkdir(parents=True)
    raw_sample = helper / "sweap_eval_full_v2.jsonl"
    raw_sample.write_text(json.dumps({"instance_id": "x", "FAIL_TO_PASS": [], "PASS_TO_PASS": []}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"instance_id": "x", "model_patch": "diff --git a/a b/a\n", "model_name_or_path": "m"}) + "\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grade_pro.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grade_pro.py",
            "--predictions",
            str(predictions),
            "--out-dir",
            str(tmp_path / "out"),
            "--eval-repo",
            str(eval_repo),
        ],
    )

    assert grade_pro.main() == 0
    assert "--raw_sample_path" in captured["cmd"]
    assert str(raw_sample) in captured["cmd"]
    assert captured["cwd"] == str(eval_repo)


def test_swe_pro_grade_rejects_repo_internal_output(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.grade_pro as grade_pro

    eval_repo = tmp_path / "SWE-bench_Pro-os"
    helper = eval_repo / "helper_code"
    helper.mkdir(parents=True)
    raw_sample = helper / "sweap_eval_full_v2.jsonl"
    raw_sample.write_text(json.dumps({"instance_id": "x", "FAIL_TO_PASS": [], "PASS_TO_PASS": []}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"instance_id": "x", "model_patch": "diff --git a/a b/a\n", "model_name_or_path": "m"}) + "\n", encoding="utf-8")
    internal_out = REPO_ROOT / "_test_rejected_grade_output_dir"
    shutil.rmtree(internal_out, ignore_errors=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grade_pro.py",
            "--predictions",
            str(predictions),
            "--out-dir",
            str(internal_out),
            "--eval-repo",
            str(eval_repo),
            "--skip-run",
        ],
    )
    try:
        with pytest.raises(ValueError, match="under repo"):
            grade_pro.main()
        assert not internal_out.exists()
    finally:
        shutil.rmtree(internal_out, ignore_errors=True)


def test_swe_pro_prediction_capture_rejects_empty_patch(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.pro_predictions as pro_predictions

    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "empty.diff"

    def fake_run(cmd, **kwargs):
        out.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(pro_predictions.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="empty patch"):
        pro_predictions._capture_patch(repo, "HEAD", out)


def test_swe_pro_predictions_continue_on_error_writes_denominator_ledger(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.pro_predictions as pro_predictions

    repo = tmp_path / "repo"
    repo.mkdir()
    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    input_jsonl.write_text(
        json.dumps({"instance_id": "case1", "repo_dir": str(repo), "base_commit": "HEAD"}) + "\n",
        encoding="utf-8",
    )

    def fake_capture(repo_dir, base_commit, out_path):
        raise RuntimeError(f"capture_patch.sh produced an empty patch for {repo_dir}")

    monkeypatch.setattr(pro_predictions, "_capture_patch", fake_capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pro_predictions.py",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
            "--continue-on-error",
        ],
    )

    assert pro_predictions.main() == 0
    assert output_jsonl.read_text(encoding="utf-8") == ""
    ledger = [json.loads(line) for line in (tmp_path / "predictions.jsonl.ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [json.loads(line) for line in (tmp_path / "predictions.jsonl.errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger[0]["instance_id"] == "case1"
    assert ledger[0]["status"] == "empty_patch"
    assert errors[0]["reason_code"] == "empty_patch"


def test_swe_pro_predictions_fail_fast_marks_remaining_requested_tasks(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.pro_predictions as pro_predictions

    repo = tmp_path / "repo"
    repo.mkdir()
    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    input_jsonl.write_text(
        json.dumps({"instance_id": "case1", "repo_dir": str(repo), "base_commit": "HEAD"})
        + "\n"
        + json.dumps({"instance_id": "case2", "repo_dir": str(repo), "base_commit": "HEAD"})
        + "\n",
        encoding="utf-8",
    )

    def fake_capture(repo_dir, base_commit, out_path):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(pro_predictions, "_capture_patch", fake_capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pro_predictions.py",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
        ],
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        pro_predictions.main()
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl.ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["case1", "case2"]
    assert rows[0]["status"] == "failed"
    assert rows[1]["status"] == "not_attempted"
    assert rows[1]["reason_code"] == "aborted_after_prior_error"


def test_swe_predictions_rejects_unsafe_instance_id_before_logs_escape(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench.swebench_predictions as swe_predictions

    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    logs_dir = tmp_path / "logs"
    input_jsonl.write_text(
        json.dumps({"instance_id": "../escape", "workspace_root": "/missing", "problem_statement": "fix"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swebench_predictions.py",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
            "--logs-dir",
            str(logs_dir),
            "--continue-on-error",
        ],
    )

    assert swe_predictions.main() == 0
    errors = json.loads((tmp_path / "predictions.jsonl.errors.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ledger = json.loads((tmp_path / "predictions.jsonl.ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert errors["reason_code"] == "invalid_instance_id"
    assert ledger["reason_code"] == "invalid_instance_id"
    assert ledger["status"] == "failed"
    assert not (tmp_path / "escape").exists()


def test_swe_predictions_fail_fast_still_writes_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench.swebench_predictions as swe_predictions

    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    input_jsonl.write_text(
        json.dumps({"instance_id": "case1", "workspace_root": "/missing", "problem_statement": "fix"})
        + "\n"
        + json.dumps({"instance_id": "case2", "workspace_root": "/also-missing", "problem_statement": "fix"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swebench_predictions.py",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
        ],
    )

    with pytest.raises(RuntimeError, match="workspace_root is not a directory"):
        swe_predictions.main()
    assert output_jsonl.exists()
    assert (tmp_path / "predictions.jsonl.errors.jsonl").exists()
    assert (tmp_path / "predictions.jsonl.ledger.jsonl").exists()
    assert (tmp_path / "predictions.jsonl.run_manifest.json").exists()
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl.ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((tmp_path / "predictions.jsonl.run_manifest.json").read_text(encoding="utf-8"))
    assert [row["instance_id"] for row in ledger_rows] == ["case1", "case2"]
    assert ledger_rows[0]["reason_code"] == "invalid_workspace"
    assert ledger_rows[1]["status"] == "not_attempted"
    assert ledger_rows[1]["reason_code"] == "aborted_after_prior_error"
    assert manifest["requested_task_ids"] == ["case1", "case2"]


def test_swe_pro_predictions_rejects_unsafe_instance_id_before_patch_path(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.pro_predictions as pro_predictions

    repo = tmp_path / "repo"
    repo.mkdir()
    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    patch_dir = tmp_path / "patches"
    input_jsonl.write_text(
        json.dumps({"instance_id": "../escape", "repo_dir": str(repo), "base_commit": "HEAD"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pro_predictions, "_capture_patch", lambda *a, **k: pytest.fail("unsafe id should fail before capture"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pro_predictions.py",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
            "--patch-dir",
            str(patch_dir),
        ],
    )

    with pytest.raises(ValueError, match="single safe path component"):
        pro_predictions.main()
    assert not (tmp_path / "escape").exists()


def test_benchmark_output_helpers_reject_repo_internal_outputs(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench.swebench_predictions as swe_predictions
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke
    from devtools.benchmarks.common.run_roots import ensure_file_output_outside_repo

    input_jsonl = tmp_path / "instances.jsonl"
    input_jsonl.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["swebench_predictions.py", "--input", str(input_jsonl), "--output", str(REPO_ROOT / "devtools" / "bad.jsonl")])
    with pytest.raises(ValueError, match="benchmark run output must not be under repo"):
        swe_predictions.main()

    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--run-root", str(REPO_ROOT / "devtools" / "bad_run")])
    with pytest.raises(ValueError, match="benchmark run output must not be under repo"):
        harbor_smoke.main()

    live_data = tmp_path / "live-data"
    live_data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(live_data))
    with pytest.raises(ValueError, match="live runtime data"):
        ensure_file_output_outside_repo(live_data / "bench" / "result_index.jsonl", REPO_ROOT)

    monkeypatch.setattr(sys, "argv", ["swebench_predictions.py", "--input", str(input_jsonl), "--output", str(live_data / "predictions.jsonl")])
    with pytest.raises(ValueError, match="live runtime data"):
        swe_predictions.main()


def test_terminal_bench_smoke_writes_manifest_and_planned_ledger(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb-run"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_harbor_smoke.py",
            "--run-root",
            str(run_root),
            "--model",
            "google/gemini-3.5-flash",
            "--settings-path",
            str(settings),
        ],
    )

    assert harbor_smoke.main() == 0
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["benchmark"] == "terminal_bench"
    assert manifest["requested_count"] == 5
    assert manifest["requested_task_ids"] == []
    assert manifest["extra"]["selection"]["mode"] == "deterministic_first_n"
    assert len(manifest["extra"]["selection"]["requested_slots"]) == 5
    assert "--jobs-dir" in manifest["official_command"]
    assert "--output-dir" not in manifest["official_command"]
    assert f"host_settings_path={settings}" in manifest["official_command"]
    assert rows and {row["status"] for row in rows} == {"planned"}
    assert {row["instance_id"] for row in rows} == {f"selection-slot-{idx}" for idx in range(1, 6)}
    assert all(row["official_eval_status"] == "not_run" for row in rows)


def test_terminal_bench_parses_harbor_task_outcomes(tmp_path):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "stats": {
                    "evals": {
                        "eval": {
                            "reward_stats": {
                                "reward": {
                                    "1.0": ["task-b"],
                                    "0.0": ["task-a"],
                                }
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert harbor_smoke._harbor_task_outcomes(result_path) == [
        {"instance_id": "task-a", "reward": 0.0},
        {"instance_id": "task-b", "reward": 1.0},
    ]


def test_terminal_bench_resolves_only_new_harbor_result(tmp_path):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    old = tmp_path / "old" / "result.json"
    old.parent.mkdir()
    old.write_text("{}", encoding="utf-8")
    before = set(harbor_smoke._harbor_results(tmp_path))
    new = tmp_path / "new" / "result.json"
    new.parent.mkdir()
    new.write_text("{}", encoding="utf-8")

    assert harbor_smoke._new_harbor_result(tmp_path, before) == new.resolve(strict=False)


def test_terminal_bench_ambiguous_harbor_result_fails_closed(tmp_path):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    before: set[Path] = set()
    for name in ("a", "b"):
        result = tmp_path / name / "result.json"
        result.parent.mkdir()
        result.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly one new Harbor result"):
        harbor_smoke._new_harbor_result(tmp_path, before)


def test_terminal_bench_explicit_execute_uses_requested_denominator(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"
    commands = []

    def fake_run(cmd, cwd=None, env=None):
        commands.append(cmd)
        assert env and str(REPO_ROOT) in env.get("PYTHONPATH", "")
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(
            json.dumps({"stats": {"evals": {"eval": {"reward_stats": {"reward": {"1.0": ["task-a", "task-b"]}}}}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_harbor_smoke.py", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
    )

    assert harbor_smoke.main() == 0
    assert commands[0][commands[0].index("--n-tasks") + 1] == "2"
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-a", "task-b"]
    assert {row["status"] for row in rows} == {"harness_completed"}


def test_terminal_bench_explicit_execute_rejects_unexpected_observed_task(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(
            json.dumps({"stats": {"evals": {"eval": {"reward_stats": {"reward": {"1.0": ["unexpected-task"]}}}}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_harbor_smoke.py", "--run-root", str(run_root), "--task", "task-a", "--execute"],
    )

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-a"]
    assert rows[0]["status"] == "harness_failed"
    assert rows[0]["reason_code"] == "harbor_result_unresolved"
    assert "unexpected-task" in rows[0]["error"]


def test_terminal_bench_explicit_execute_rejects_missing_requested_task(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(
            json.dumps({"stats": {"evals": {"eval": {"reward_stats": {"reward": {"1.0": ["task-a"]}}}}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_harbor_smoke.py", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
    )

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-a", "task-b"]
    assert {row["status"] for row in rows} == {"harness_failed"}
    assert all(row["reason_code"] == "harbor_result_unresolved" for row in rows)
    assert all("task-b" in row["error"] for row in rows)


def test_terminal_bench_execute_fails_closed_on_unparseable_harbor_result(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--run-root", str(run_root), "--execute"])

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert {row["status"] for row in rows} == {"harness_failed"}
    assert all(row["reason_code"] == "harbor_result_unresolved" for row in rows)


def test_terminal_bench_execute_fails_closed_on_partial_deterministic_result(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(
            json.dumps({"stats": {"evals": {"eval": {"reward_stats": {"reward": {"1.0": ["task-a"]}}}}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--run-root", str(run_root), "--n-tasks", "2", "--execute"])

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"harness_failed"}
    assert all("expected 2" in row["error"] for row in rows)


def test_terminal_bench_execute_writes_ledger_when_harbor_invocation_fails(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        raise FileNotFoundError("harbor missing")

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_harbor_smoke.py", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
    )

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-a", "task-b"]
    assert {row["status"] for row in rows} == {"harness_failed"}
    assert {row["reason_code"] for row in rows} == {"harbor_invocation_failed"}
    assert all("harbor missing" in row["error"] for row in rows)


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


def test_terminal_bench_adapter_quotes_hostile_workspace_dir(tmp_path):
    from devtools.benchmarks.terminal_bench.harbor_installed_agent import OuroborosTerminalBenchAgent

    class FakeResult:
        return_code = 0
        stdout = '{"return_code": 0}\n'
        stderr = ""

    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, **kwargs):
            self.calls.append(kwargs)
            return FakeResult()

    hostile = "/tmp/ws'; touch /tmp/pwn; echo '"
    agent = OuroborosTerminalBenchAgent(logs_dir=tmp_path, workspace_dir=hostile, task_timeout_sec=900)
    environment = FakeEnvironment()

    asyncio.run(agent._resolve_workspace_dir(environment))
    asyncio.run(agent._ensure_workspace_git_root(environment))
    summary = asyncio.run(agent._run_ouroboros_task(environment, {}))

    assert summary["return_code"] == 0
    quoted = shlex.quote(hostile)
    assert environment.calls[0]["command"] == f"test -d {quoted}"
    git_command = environment.calls[1]["command"]
    assert f"workspace_dir={quoted}" in git_command
    assert "cd \"$workspace_dir\"" in git_command
    runner_command = environment.calls[-1]["command"]
    runner = runner_command.split("cat > /tmp/run_ouroboros_task.py <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    assert f'"workspace_root": {json.dumps(hostile)}' in runner
    assert '"service_teardown": "keep"' in runner
    assert 'task_body["timeout_sec"] = task_timeout' in runner
    assert "task_timeout = 870" in runner
    compile(runner, "run_ouroboros_task.py", "exec")


def test_terminal_bench_run_tb_validates_leaderboard_methodology():
    from devtools.benchmarks.terminal_bench.run_tb import validate_methodology

    validate_methodology(k=5, timeout_multiplier=1.0, resource_overrides=[])
    with pytest.raises(ValueError, match="k >= 5"):
        validate_methodology(k=1, timeout_multiplier=1.0, resource_overrides=[])
    with pytest.raises(ValueError, match="timeout_multiplier"):
        validate_methodology(k=5, timeout_multiplier=2.0, resource_overrides=[])
    with pytest.raises(ValueError, match="forbids resource overrides"):
        validate_methodology(k=5, timeout_multiplier=1.0, resource_overrides=["cpus=8"])


def test_terminal_bench_run_tb_builds_required_agent_kwargs(tmp_path):
    from devtools.benchmarks.terminal_bench.run_tb import HarborCommandConfig, harbor_command

    cmd = harbor_command(HarborCommandConfig(
        dataset="terminal-bench/terminal-bench-2-1",
        model="openai/gpt-5.5",
        k=5,
        jobs_dir=tmp_path / "jobs",
        harbor_bin="harbor",
        n_concurrent=1,
        task_filters=["pypi-server"],
        settings_path=tmp_path / "settings.json",
        execute=True,
        light_model="google/gemini-3.5-flash",
    ))

    joined = " ".join(cmd)
    assert "-k 5" in joined
    assert "task_review_mode=required" in cmd
    assert "ouroboros_light_model=google/gemini-3.5-flash" in cmd
    assert "--include-task-name" in cmd
    assert "pypi-server" in cmd
    assert "--force-build" in cmd
    # 6a: leaderboard-faithful default — Harbor static_validation REJECTS the
    # setup/build timeout multipliers (static_validation.py
    # _trial_timeout_override_fields rejects agent_setup_timeout_multiplier +
    # environment_build_timeout_multiplier), so harbor_command omits them by default;
    # they appear only under the local --allow-setup-build-multipliers opt-in (covered
    # in test_run_tb_methodology.py). Task/verifier timeout multipliers stay 1.0 too.
    assert "--agent-setup-timeout-multiplier" not in cmd
    assert "--environment-build-timeout-multiplier" not in cmd
    assert "--agent-timeout-multiplier" not in cmd


def test_container_env_never_forwards_model_fallback(tmp_path, monkeypatch):
    """6b: the benchmark metric is single-model — a host-configured
    OUROBOROS_MODEL_FALLBACK must never leak into the container env."""
    import json as _json

    from devtools.benchmarks.terminal_bench.harbor_installed_agent import (
        OuroborosTerminalBenchAgent,
    )

    settings = tmp_path / "settings.json"
    settings.write_text(_json.dumps({
        "OUROBOROS_MODEL": "openai/gpt-5.5",
        "OUROBOROS_MODEL_FALLBACK": "google/gemini-3.5-flash",
    }), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "google/gemini-3.5-flash")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-5.5")

    agent = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(settings),
        ouroboros_model="openai/gpt-5.5",
    )
    env = agent._container_env()
    # The fallback is PINNED to the measured model (not absent: the container
    # has no settings.json, so absence would resurrect the SETTINGS_DEFAULTS
    # fallback — a different model — inside the container).
    assert env.get("OUROBOROS_MODEL_FALLBACK") == "openai/gpt-5.5"
    assert env.get("OUROBOROS_MODEL") == "openai/gpt-5.5"

    # No explicit kwarg: the pin follows the forwarded host main model.
    agent_no_kwarg = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(settings),
    )
    env2 = agent_no_kwarg._container_env()
    assert env2.get("OUROBOROS_MODEL_FALLBACK") == env2.get("OUROBOROS_MODEL") == "openai/gpt-5.5"

    # No model anywhere: the pin falls back to the packaged default main model
    # (fallback == main holds in EVERY reachable configuration).
    monkeypatch.delenv("OUROBOROS_MODEL", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    empty_settings = tmp_path / "empty_settings.json"
    empty_settings.write_text("{}", encoding="utf-8")
    agent_bare = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(empty_settings),
    )
    env3 = agent_bare._container_env()
    from ouroboros.config import SETTINGS_DEFAULTS
    assert env3.get("OUROBOROS_MODEL_FALLBACK") == SETTINGS_DEFAULTS["OUROBOROS_MODEL"]


def test_harbor_agent_defaults_max_workers_two_and_probes_context_timeout(tmp_path):
    """6c: plan_task needs >=2 workers; 6d: per-task timeout adopted from the
    harbor AgentContext when a future harbor exposes it (today: metadata probe)."""
    import types as _types

    from devtools.benchmarks.terminal_bench.harbor_installed_agent import (
        OuroborosTerminalBenchAgent,
    )

    agent = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(tmp_path / "settings.json"),
    )
    assert agent.max_workers == 2
    assert agent.task_timeout_sec is None

    ctx = _types.SimpleNamespace(metadata={"task_timeout_sec": 900})
    assert agent._context_task_timeout_sec(ctx) == 900
    ctx_attr = _types.SimpleNamespace(agent_timeout_sec=600, metadata=None)
    assert agent._context_task_timeout_sec(ctx_attr) == 600
    ctx_none = _types.SimpleNamespace(metadata={})
    assert agent._context_task_timeout_sec(ctx_none) is None
    # Explicit kwarg still wins over the probe.
    agent_explicit = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(tmp_path / "settings.json"),
        task_timeout_sec=300,
    )
    assert agent_explicit.task_timeout_sec == 300
