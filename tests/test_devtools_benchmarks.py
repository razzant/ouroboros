from __future__ import annotations

import ast
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
    build_instruction,
    build_ouroboros_task_body,
    classify_infra_failure,
    cleanroom_image_ref,
    container_name_for_instance,
    create_submission_tarball,
    prepare_seeded_workspace,
    preflight_cleanroom_container,
    seed_workspace_from_image,
    start_cleanroom_container,
    submit_and_wait,
    terminal_task_status,
    verify_reference_executable_runnable,
)
from devtools.benchmarks.swe_bench.presets import resolve_preset


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASH_CAPTURE_AVAILABLE = sys.platform != "win32" and shutil.which("bash") is not None


@pytest.fixture(autouse=True)
def _isolate_bench_runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_BENCH_RUNS_ROOT", str(tmp_path / "bench_runs"))
    # Command-construction tests inspect the raw solver argv; the GAIA bwrap
    # answer-cache isolation (default-on at runtime) would prepend a `bwrap … --`
    # prefix and SystemExit where bwrap is absent (CI). Disable by default; the
    # dedicated bwrap test re-enables it explicitly.
    monkeypatch.setenv("GAIA_BWRAP_ISOLATE", "0")


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


def test_official_command_builders_do_not_replace_scoring(monkeypatch):
    from devtools.benchmarks.common import official_commands

    monkeypatch.setattr(official_commands, "resolve_programbench_cli", lambda: ["/opt/homebrew/bin/programbench"])
    monkeypatch.delenv("PROGRAMBENCH_DOCKER_CPUS", raising=False)
    # The builders stringify the Path via str(); compare against the platform
    # spelling so the argv-structure assertion stays valid on Windows too
    # (str(Path("/runs/pb")) == "\\runs\\pb" there).
    pb_run = str(Path("/runs/pb"))
    preds = str(Path("/runs/predictions.jsonl"))
    assert programbench_eval_cmd(Path("/runs/pb")) == [
        "/opt/homebrew/bin/programbench",
        "eval",
        pb_run,
        "--docker-cpus",
        "4",
    ]
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
    # require_clean=False: this test asserts the provenance RECORD on a deliberately dirty
    # checkout. The gate itself (default require_clean=True) is asserted separately below.
    manifest = benchmark_run_manifest(
        benchmark="unit",
        run_root=tmp_path / "run",
        repo_dir=repo,
        requested_task_ids=["task-1"],
        require_clean=False,
        metadata={"argv": ["bench", "--task", "task-1"]},
    )

    assert provenance["dirty"] is True
    assert provenance["tracked_diff_sha256"]
    assert "print('changed')" not in json.dumps(provenance)
    assert manifest["requested_count"] == 1
    assert manifest["source"]["tracked_diff_sha256"]
    assert manifest["seed_gate"] == {
        "require_clean": False,
        "allow_dirty_seed": True,
        "expect": "",
        "git_available": True,
        "status_available": True,
        "dirty": True,
        "describe": manifest["source"]["describe"],
        "reason": "seed_dirty",
        "ok": False,
    }


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
        "devtools/benchmarks/programbench/run_programbench_e2e.py",
        "devtools/benchmarks/programbench/export_programbench_submissions.py",
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
        "devtools/benchmarks/cybergym/run_cybergym.py",
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
    # Inline CLI shape plus the fixed-model actor, without legacy model slots.
    # Encoder owns bytes; test checks only the launcher wiring.
    from devtools.benchmarks.harness_bench_fast import ouroboros_cli_wrapper as w

    assert hasattr(w, "main")
    src = (
        REPO_ROOT / "devtools" / "benchmarks" / "harness_bench_fast" / "ouroboros_cli_wrapper.py"
    ).read_text(encoding="utf-8")
    for token in ('"run",', '"--memory-mode",', '"--quiet",', '"--result-json-out",', '"--actor-id",'):
        assert token in src, token
    assert "fixed_model_actor_snapshot(args.model, target=env)\n" in src
    assert '"OUROBOROS_MODEL_HEAVY": args.model' not in src
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
    # allow_dirty_seed=True keeps this assertion independent of the AMBIENT checkout state:
    # the seed gate is exercised deterministically in the dedicated test below.
    args = types.SimpleNamespace(
        split="validation", level=1, limit=3, solve_model="google/gemini-2.5-pro",
        allow_dirty_seed=True,
    )
    admitted = run_gaia._admit_run(tmp_path, args, argv)
    run_gaia._augment_manifest(admitted, args, tmp_path, settings_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["official_command"] == argv
    assert manifest["requested_count"] == 3
    # `model_slots` is settings-derived, so it exists only on the augmented (retained) dict --
    # the file itself is rewritten with it by the finalization seam in main().
    assert admitted["model_slots"]["OUROBOROS_MODEL"] == "google/gemini-2.5-pro"
    assert "web_search" in open(REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "inspect_solver" / "ouroboros_solver.py", encoding="utf-8").read()
    assert "claude_code_edit" in open(REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "inspect_solver" / "ouroboros_solver.py", encoding="utf-8").read()


def test_gaia_profile_defaults_are_not_silent_web_off():
    import argparse
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    args = argparse.Namespace(
        profile="strict_ddgs", disable_tools=None, websearch_backend="",
        main_web_search="off", main_web_search_engine="auto", max_workers=1,
    )
    run_gaia._apply_profile_defaults(args)
    assert args.disable_tools == "claude_code_edit"
    assert args.websearch_backend == "ddgs"

    quality = argparse.Namespace(
        profile="quality_openrouter_web", disable_tools=None, websearch_backend="",
        main_web_search="off", main_web_search_engine="auto", max_workers=1,
    )
    run_gaia._apply_profile_defaults(quality)
    assert quality.disable_tools == "web_search,claude_code_edit"
    assert quality.main_web_search == "openrouter"
    # v6.55.0: the parser default is 4; an explicit --max-workers value (here 1,
    # the strict-baseline ablation) must never be silently bumped by a profile.
    assert quality.max_workers == 1


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


@pytest.mark.serial
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
    assert s["OUROBOROS_MAIN_WEB_SEARCH"] == "off"


def test_gaia_render_records_main_web_settings(tmp_path, monkeypatch):
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    monkeypatch.setenv("OPENROUTER_API_KEY", "router")
    base = REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "settings_base.json"
    out = run_gaia._render_run_settings(
        base, "openai/gpt-5.5", tmp_path,
        main_web_search="openrouter", main_web_search_engine="auto",
        main_web_search_max_total_results=7,
    )
    settings = json.loads(out.read_text(encoding="utf-8"))
    assert settings["OUROBOROS_MAIN_WEB_SEARCH"] == "openrouter"
    assert settings["OUROBOROS_MAIN_WEB_SEARCH_ENGINE"] == "auto"
    assert settings["OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS"] == 7


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
    # --disable-tools stays BEFORE the prompt transport on argv: the REMAINDER
    # positional would otherwise swallow it (the original bug class), and with
    # the C5 file transport a later flag must still never shadow it.
    assert seen["cmd"].index("--disable-tools") < seen["cmd"].index("--prompt-file")
    parser = cli.build_parser()
    ns = parser.parse_args(seen["cmd"][3:])
    assert ns.disable_tools == ["web_search,claude_code_edit"]
    assert ns.result_json_out
    # C5 E2BIG hygiene: the prompt travels as a FILE, never as an argv tail.
    assert not ns.prompt
    prompt_path = Path(ns.prompt_file)
    assert prompt_path.is_file()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    # The prompt is the question plus the official GAIA "FINAL ANSWER:" protocol suffix.
    assert prompt_text.startswith("question")
    assert "FINAL ANSWER:" in prompt_text


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


def test_gaia_solver_returns_real_host_paths_and_denies_secrets(monkeypatch, tmp_path):
    # v6.52.0 (P1): the solver no longer copies into sample_dir/attachments/ nor
    # parses phantom /shared_files paths out of the prompt. It returns the REAL host
    # file paths (the core stage_task_attachments stages them); secret sources are
    # still denied as defense-in-depth.
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    image = tmp_path / "chart.png"
    image.write_bytes(b"png")
    secret_dir = tmp_path / ".ssh"
    secret_dir.mkdir()
    secret = secret_dir / "id_rsa"
    secret.write_text("secret", encoding="utf-8")
    state = SimpleNamespace(metadata={"attachments": [str(secret), str(image)]})

    attachments = ouroboros_solver._attachment_paths_from_state(state)

    assert len(attachments) == 1
    # Real host path is returned as-is (no copy / no rename).
    assert attachments[0] == image.resolve()
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


def test_gaia_attachment_copy_avoids_duplicate_basenames(tmp_path):
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    src1 = tmp_path / "one" / "same.txt"
    src2 = tmp_path / "two" / "same.txt"
    src1.parent.mkdir()
    src2.parent.mkdir()
    src1.write_text("one", encoding="utf-8")
    src2.write_text("two", encoding="utf-8")

    attachments = ouroboros_solver._attachment_paths_from_state(
        SimpleNamespace(files={str(src1): str(src1), str(src2): str(src2)}),
        sample_dir=tmp_path / "sample",
        prompt="",
    )
    assert [p.name for p in attachments] == ["same.txt", "same_2.txt"]
    assert attachments[0].read_text(encoding="utf-8") == "one"
    assert attachments[1].read_text(encoding="utf-8") == "two"


def test_gaia_attachment_falls_back_to_shared_files_root_and_rewrites_prompt(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    # v6.74.0 (C1): the shared-root fallback is an EXACT relative lookup —
    # /shared_files/doc.pdf resolves only <root>/doc.pdf. The old broad
    # name-anywhere rglob (which could stage an unrelated same-named file from
    # any subdirectory) was removed; an unresolvable declared attachment is a
    # typed staging error at the solve boundary instead.
    attached = shared / "doc.pdf"
    attached.write_bytes(b"%PDF")
    (shared / "2023" / "validation").mkdir(parents=True)
    (shared / "2023" / "validation" / "unrelated.pdf").write_bytes(b"nope")
    monkeypatch.setenv("GAIA_SHARED_FILES_ROOT", str(shared))
    prompt = "Please inspect /shared_files/doc.pdf and answer."
    attachments = ouroboros_solver._attachment_paths_from_state(SimpleNamespace(files={}), prompt=prompt)
    assert attachments == [attached.resolve()]
    rewritten = ouroboros_solver._rewrite_shared_file_prompt(prompt, attachments)
    assert "/shared_files/doc.pdf" not in rewritten
    assert "[ATTACHMENTS]" in rewritten
    assert "doc.pdf" in rewritten


def test_gaia_exact_lookup_does_not_stage_name_anywhere_matches(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    shared = tmp_path / "shared"
    nested = shared / "2023" / "validation"
    nested.mkdir(parents=True)
    (nested / "doc.pdf").write_bytes(b"%PDF")  # exists ONLY at a nested path
    monkeypatch.setenv("GAIA_SHARED_FILES_ROOT", str(shared))
    prompt = "Please inspect /shared_files/doc.pdf and answer."
    attachments = ouroboros_solver._attachment_paths_from_state(SimpleNamespace(files={}), prompt=prompt)
    assert attachments == []  # no broad basename search; typed error surfaces at solve


def test_gaia_sandbox_staging_and_typed_error(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    sample_dir = tmp_path / "sample"
    # No sandbox available (inspect_ai.util import fails in tests) and no host
    # resolution -> a DECLARED file becomes the typed staging error.
    state = SimpleNamespace(files={"/shared_files/missing.bin": "/shared_files/missing.bin"}, metadata={})
    with pytest.raises(ouroboros_solver.GaiaAttachmentStagingError):
        asyncio.run(ouroboros_solver._stage_sandbox_attachments(state, sample_dir, []))
    # A declared file already resolved by the host path stays satisfied.
    resolved = tmp_path / "doc.pdf"
    resolved.write_bytes(b"%PDF")
    state2 = SimpleNamespace(files={"/shared_files/doc.pdf": str(resolved)}, metadata={})
    out = asyncio.run(ouroboros_solver._stage_sandbox_attachments(state2, sample_dir, [resolved]))
    assert out == [resolved]


def test_gaia_real_taskstate_shape_declares_via_prompt(tmp_path):
    # codex final review: the REAL inspect_ai TaskState has NO `files` attribute
    # (verified on 0.3.244) — the prompt's /shared_files path is the declaration
    # channel in the official harness. A prompt-declared file with no host
    # resolution and no sandbox must raise the typed staging error, never solve
    # silently without its input.
    import asyncio
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    class _RealShapeState:  # no files/attachments attributes, like TaskState
        metadata: dict = {}

    prompt = "Please read /shared_files/2023/validation/doc.pdf and answer."
    with pytest.raises(ouroboros_solver.GaiaAttachmentStagingError):
        asyncio.run(ouroboros_solver._stage_sandbox_attachments(
            _RealShapeState(), tmp_path / "s", [], prompt=prompt,
        ))
    # ...and a host-resolved copy of the same basename satisfies the declaration.
    resolved = tmp_path / "doc.pdf"
    resolved.write_bytes(b"%PDF")
    out = asyncio.run(ouroboros_solver._stage_sandbox_attachments(
        _RealShapeState(), tmp_path / "s", [resolved], prompt=prompt,
    ))
    assert out == [resolved]


def test_gaia_shared_files_fallback_prefers_prompt_subpath_over_basename(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    shared = tmp_path / "shared"
    wanted = shared / "a" / "doc.pdf"
    wrong = shared / "b" / "doc.pdf"
    wanted.parent.mkdir(parents=True)
    wrong.parent.mkdir(parents=True)
    wanted.write_bytes(b"wanted")
    wrong.write_bytes(b"wrong")
    monkeypatch.setenv("GAIA_SHARED_FILES_ROOT", str(shared))

    attachments = ouroboros_solver._attachment_paths_from_state(
        SimpleNamespace(files={}),
        prompt="Please inspect /shared_files/a/doc.pdf.",
    )

    assert attachments == [wanted.resolve()]


def test_gaia_shared_files_fallback_blocks_traversal(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    shared = tmp_path / "shared"
    shared.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.setenv("GAIA_SHARED_FILES_ROOT", str(shared))

    attachments = ouroboros_solver._attachment_paths_from_state(
        SimpleNamespace(files={}),
        prompt="Please inspect /shared_files/../outside.txt.",
    )

    assert attachments == []


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
        protected_backend_paths=["/workspace/reference_executable"],
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
    # House rule: benches measure the single-model Ouroboros harness.
    assert body["disabled_tools"] == ["claude_code_edit", "schedule_subagent"]
    # POST /api/tasks accepts no top-level task_contract field; the pacing block
    # rides in metadata.budget_profile and must already be in the normalized
    # contract shape so build_task_contract() adopts it verbatim.
    assert "task_contract" not in body
    profile = body["metadata"]["budget_profile"]
    assert profile == {
        "cost_hard_stop_pct": 0,
        "improvement_policy": "fixed",
        "max_improvement_passes": 6,
        "reserve_finalization_pct": 15,
    }
    # Advisory acceptance claims ride the body top-level (gateway-normalized);
    # the wording stays task-general (no benchmark-specific oracle taxonomy).
    claims = body["acceptance_claims"]
    assert len(claims) == 1 and claims[0]["id"] == "behavioral_equivalence"
    assert claims[0]["priority"] == "must"
    from ouroboros.contracts.task_contract import build_task_contract, normalize_budget_profile

    assert normalize_budget_profile(profile) == profile
    assert build_task_contract(body)["budget_profile"] == profile


def test_programbench_git_workspace_does_not_commit_protected_reference(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reference_executable").write_text("protected-bytes\n", encoding="utf-8")

    build_ouroboros_task_body(
        instruction="solve",
        workspace_host_path=workspace,
        container_name="pb-cleanroom",
        protected_backend_paths=["/workspace/reference_executable"],
    )

    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    show = subprocess.run(["git", "show", "HEAD:reference_executable"], cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
    (workspace / "reference_executable").write_text("protected\n", encoding="utf-8")
    (workspace / "solution.py").write_text("print('ok')\n", encoding="utf-8")

    tar_path = create_submission_tarball(
        workspace,
        tmp_path / "submission.tar.gz",
        protected_paths=["/workspace/reference_executable", "reference_executable"],
    )

    with tarfile.open(tar_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "solution.py" in names
    assert ".git/HEAD" not in names
    assert ".ouroboros/trace.json" not in names
    assert "node_modules/pkg/index.js" not in names
    assert "build/out.o" not in names
    assert "dist/bundle.js" not in names
    assert "reference_executable" not in names


def test_programbench_submission_excludes_both_root_binaries(tmp_path):
    """Source-submission contract: neither the agent-built ./executable nor the
    reference binary may enter submission.tar.gz — the official eval rebuilds
    via compile.sh, and a shipped binary would mask compile failures. Nested
    files that merely SHARE the name stay in (they are ordinary source tree
    content)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF-agent-built")
    (workspace / "reference_executable").write_bytes(b"\x7fELF-reference")
    (workspace / "compile.sh").write_text("#!/bin/sh\ncc -o executable main.c\n", encoding="utf-8")
    (workspace / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    (workspace / "tools").mkdir()
    (workspace / "tools" / "executable").write_text("just a source file\n", encoding="utf-8")

    tar_path = create_submission_tarball(workspace, tmp_path / "submission.tar.gz")

    with tarfile.open(tar_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "compile.sh" in names
    assert "main.c" in names
    assert "tools/executable" in names
    assert "executable" not in names
    assert "reference_executable" not in names


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
            "--allow-dirty-seed",
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


def test_programbench_prepare_seeded_workspace_is_idempotent_on_solved_tree(tmp_path):
    """Re-running prepare on an ALREADY-normalized workspace (reference present,
    agent-built ./executable beside it after a solve) must preserve the real
    reference and leave the agent's build product alone — never rename the
    agent binary over the protected reference."""
    from devtools.benchmarks.programbench.programbench_adapter import prepare_seeded_workspace

    root = tmp_path / "ws"
    root.mkdir()
    (root / "reference_executable").write_bytes(b"REAL-REFERENCE")
    (root / "executable").write_bytes(b"AGENT-BUILD")
    layout = prepare_seeded_workspace(root)
    assert (root / "reference_executable").read_bytes() == b"REAL-REFERENCE"
    assert (root / "executable").read_bytes() == b"AGENT-BUILD"
    assert layout["reference_host_path"] == str(root / "reference_executable")


def test_programbench_prepare_only_normalizes_raw_workspace(tmp_path, monkeypatch):
    """run_programbench (prepare-only) must run prepare_seeded_workspace before
    body/submission creation: a raw cleanroom workspace has the REAL reference
    at ./executable — unrenamed it would ship in the tarball while the task
    body points agents at a nonexistent ./reference_executable."""
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF-raw-seeded-reference")
    (workspace / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    output = tmp_path / "ledger.jsonl"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(run_programbench, "preflight_cleanroom_container",
                        lambda _: {"image": "task_cleanroom", "network": "none"})
    monkeypatch.setattr(sys, "argv", [
        "run_programbench.py", "--allow-dirty-seed", "--workspace", str(workspace),
        "--instruction-file", str(instruction), "--container-name", "pb",
        "--instance-id", "case-prep", "--ledger-output", str(output),
        "--manifest-output", str(manifest),
    ])
    run_programbench.main()

    assert (workspace / "reference_executable").is_file()
    assert not (workspace / "executable").exists()
    with tarfile.open(next(tmp_path.rglob("submission.tar.gz")), "r:gz") as tar:
        names = set(tar.getnames())
    assert "main.c" in names
    assert "reference_executable" not in names
    assert "executable" not in names


def test_programbench_submission_failure_writes_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF-seeded-reference")
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
            "--allow-dirty-seed",
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
    (workspace / "executable").write_bytes(b"\x7fELF-seeded-reference")
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
            "--allow-dirty-seed",
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


def test_programbench_client_poll_error_keeps_container_when_task_live(tmp_path, monkeypatch):
    """A client-side poll failure (timeout OR any transient mid-poll error) after a
    task was submitted must NOT tear down the cleanroom container — the checkpoint
    holds a live task_id and the next run reattaches to it. A failure with NO
    submitted task (creation itself failed) falls to the normal teardown path."""
    import json as _json

    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    stopped: list[str] = []
    monkeypatch.setattr(e2e, "pull_cleanroom_image", lambda name: {"image": name})
    monkeypatch.setattr(e2e, "seed_workspace_from_image", lambda name, ws: {"seeded": True})
    monkeypatch.setattr(e2e, "start_cleanroom_container",
                        lambda *a, **k: {"preflight": {"ok": True}})
    monkeypatch.setattr(e2e, "stop_cleanroom_container", lambda name: stopped.append(name))
    monkeypatch.setattr(e2e, "build_ouroboros_task_body",
                        lambda **k: {"description": "x", "metadata": {}})

    cfg = e2e.InstanceRunConfig(
        out_root=tmp_path, ouroboros_url="http://127.0.0.1:1", timeout_sec=1.0,
        cpus="1", memory="1g", protected_paths=[], dry_run=False,
        skip_pull=False, redo_existing=False,
    )

    def _fake_submit(reason_exc):
        # Mirror the real submit_and_wait: it writes the checkpoint with a task_id
        # (task submitted) BEFORE polling, then raises on the poll failure.
        def _inner(base_url, body, *, timeout_sec, checkpoint_path):
            Path(checkpoint_path).write_text(
                _json.dumps({"task_id": "tsk-live", "status": "running"}), encoding="utf-8")
            raise reason_exc
        return _inner

    # (a) timeout after submit -> kept alive, timeout reason code
    monkeypatch.setattr(e2e, "submit_and_wait", _fake_submit(TimeoutError("did not finish")))
    row = e2e._process_instance({"instance_id": "inst-a", "image_name": "img-a"}, cfg)
    assert row["status"] == "failed"
    assert row["reason_code"] == "client_poll_timeout_reattachable"
    assert row["details"]["container_left_running"] is True
    assert stopped == []

    # (b) transient NON-timeout error after submit -> ALSO kept alive (r1 #10)
    monkeypatch.setattr(e2e, "submit_and_wait", _fake_submit(RuntimeError("transient 502")))
    row2 = e2e._process_instance({"instance_id": "inst-b", "image_name": "img-b"}, cfg)
    assert row2["status"] == "failed"
    assert row2["reason_code"] == "client_poll_error_reattachable"
    assert stopped == []  # a live task's container must survive a transient poll error

    # (c) failure with NO submitted task (checkpoint never written) -> teardown
    def _creation_failed(*a, **k):
        raise RuntimeError("task creation returned no id")

    monkeypatch.setattr(e2e, "submit_and_wait", _creation_failed)
    row3 = e2e._process_instance({"instance_id": "inst-c", "image_name": "img-c"}, cfg)
    assert row3["status"] == "failed"
    assert row3["reason_code"] == "RuntimeError"
    assert stopped == [e2e.container_name_for_instance("inst-c")]


def test_programbench_resume_skipped_rows_are_successful():
    """A resume-only run (everything already has submission.tar.gz) must exit 0:
    skipped rows are successful prior work for exit-code/failed_count purposes."""
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    assert e2e._row_successful({"status": "completed"})
    assert e2e._row_successful({"status": "skipped"})
    assert not e2e._row_successful({"status": "failed"})
    assert not e2e._row_successful({})


def test_programbench_second_run_reattaches_without_cleanroom_reset(tmp_path, monkeypatch):
    """After a client_poll_timeout_reattachable row, the NEXT run must honor the
    live checkpoint: no image pull, no workspace reseed, no container restart
    (start would stop the namesake executor first) — straight to reattach."""
    import json as _json

    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    def _forbidden(*a, **k):
        raise AssertionError("fresh cleanroom work must not run on the reattach path")

    stopped: list[str] = []
    monkeypatch.setattr(e2e, "pull_cleanroom_image", _forbidden)
    monkeypatch.setattr(e2e, "seed_workspace_from_image", _forbidden)
    monkeypatch.setattr(e2e, "start_cleanroom_container", _forbidden)
    monkeypatch.setattr(e2e, "stop_cleanroom_container", lambda name: stopped.append(name))
    monkeypatch.setattr(e2e, "build_ouroboros_task_body",
                        lambda **k: {"description": "x", "metadata": {}})
    monkeypatch.setattr(e2e, "ouroboros_api_request",
                        lambda *a, **k: {"task_id": "tsk-9", "status": "running"})
    monkeypatch.setattr(e2e, "submit_and_wait",
                        lambda *a, **k: {"task_id": "tsk-9", "status": "completed"})
    monkeypatch.setattr(e2e, "create_submission_tarball",
                        lambda ws, dest, protected_paths: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                           dest.write_bytes(b"x"), dest)[-1])

    cfg = e2e.InstanceRunConfig(
        out_root=tmp_path, ouroboros_url="http://127.0.0.1:1", timeout_sec=1.0,
        cpus="1", memory="1g", protected_paths=[], dry_run=False,
        skip_pull=False, redo_existing=False,
    )
    inst_dir = tmp_path / "inst-a"
    inst_dir.mkdir()
    (inst_dir / e2e.TASK_CHECKPOINT_BASENAME).write_text(
        _json.dumps({"task_id": "tsk-9", "status": "running"}), encoding="utf-8")

    row = e2e._process_instance({"instance_id": "inst-a", "image_name": "img-a"}, cfg)
    assert row["status"] == "completed"
    assert row["details"]["harness"]["reattached_task_id"] == "tsk-9"
    # settled result re-arms normal teardown
    assert stopped == [e2e.container_name_for_instance("inst-a")]


def test_programbench_settled_failed_checkpoint_retries_fresh(tmp_path, monkeypatch):
    """Adversarial review r2 #5: a checkpoint naming a task that already SETTLED
    as FAILED must NOT reattach (that replays the old failure as zero work) — the
    resume must drop the stale checkpoint and re-solve in a fresh cleanroom."""
    import json as _json

    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    fresh_work: list[str] = []
    monkeypatch.setattr(e2e, "pull_cleanroom_image", lambda img: fresh_work.append("pull") or "sha")
    monkeypatch.setattr(e2e, "seed_workspace_from_image", lambda img, ws: fresh_work.append("seed"))
    monkeypatch.setattr(e2e, "start_cleanroom_container",
                        lambda *a, **k: fresh_work.append("start") or {"container": "c"})
    monkeypatch.setattr(e2e, "stop_cleanroom_container", lambda name: None)
    monkeypatch.setattr(e2e, "build_ouroboros_task_body",
                        lambda **k: {"description": "x", "metadata": {}})
    # The reattach honor-check GET returns a SETTLED-FAILED payload.
    monkeypatch.setattr(e2e, "ouroboros_api_request",
                        lambda *a, **k: {"task_id": "tsk-old", "status": "failed"})
    monkeypatch.setattr(e2e, "submit_and_wait",
                        lambda *a, **k: {"task_id": "tsk-new", "status": "completed"})
    monkeypatch.setattr(e2e, "create_submission_tarball",
                        lambda ws, dest, protected_paths: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                           dest.write_bytes(b"x"), dest)[-1])

    cfg = e2e.InstanceRunConfig(
        out_root=tmp_path, ouroboros_url="http://127.0.0.1:1", timeout_sec=1.0,
        cpus="1", memory="1g", protected_paths=[], dry_run=False,
        skip_pull=False, redo_existing=False,
    )
    inst_dir = tmp_path / "inst-f"
    inst_dir.mkdir()
    checkpoint = inst_dir / e2e.TASK_CHECKPOINT_BASENAME
    checkpoint.write_text(_json.dumps({"task_id": "tsk-old", "status": "running"}), encoding="utf-8")

    row = e2e._process_instance({"instance_id": "inst-f", "image_name": "img-f"}, cfg)
    assert row["details"]["harness"]["reattached_task_id"] == ""  # did NOT reattach
    assert fresh_work == ["pull", "seed", "start"]  # fresh cleanroom ran
    assert row["status"] == "completed"


def test_programbench_build_instruction_renders_instance_fields(tmp_path):
    template = tmp_path / "instruction.md"
    template.write_text("id={{instance_id}} repo={{repository}} lang={{language}} diff={{difficulty}}\n", encoding="utf-8")
    text = build_instruction(
        {
            "instance_id": "foo__bar.abc123",
            "repository": "foo/bar",
            "language": "c",
            "difficulty": "easy",
        },
        template_path=template,
    )
    assert "id=foo__bar.abc123" in text
    assert "repo=foo/bar" in text
    assert "lang=c" in text
    assert "diff=easy" in text


def test_programbench_cleanroom_image_ref_and_container_name():
    assert cleanroom_image_ref("programbench/foo") == "programbench/foo:task_cleanroom_v6"
    assert cleanroom_image_ref("programbench/foo:task_cleanroom_v6") == "programbench/foo:task_cleanroom_v6"
    assert container_name_for_instance("abishekvashok__cmatrix.5c082c6").startswith("ouroboros-pb-")


def test_programbench_seed_workspace_from_image(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    workspace = tmp_path / "workspace"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["docker", "create", "--platform"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="seed-cid\n", stderr="")
        if cmd[:2] == ["docker", "cp"]:
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "executable").write_text("bin\n", encoding="utf-8")
            (workspace / "README.md").write_text("docs\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    result = seed_workspace_from_image("programbench/demo", workspace)
    assert result["seeded_from"] == "/workspace"
    assert (workspace / "reference_executable").is_file()
    assert not (workspace / "executable").exists()
    if sys.platform != "win32":  # execute bit is a POSIX concept (bench runs in Linux containers)
        assert (workspace / "reference_executable").stat().st_mode & 0o111
    assert "/reference_executable" in (workspace / ".gitignore").read_text(encoding="utf-8")
    assert calls[0][:4] == ["docker", "create", "--platform", "linux/amd64"]
    assert calls[1][:2] == ["docker", "cp"]
    assert ["docker", "rm", "-f", "seed-cid"] in calls


def test_programbench_start_cleanroom_container_invokes_docker_run(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="running-cid\n", stderr="")
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps([{"Config": {"Image": "programbench/demo:task_cleanroom_v6"}, "HostConfig": {"NetworkMode": "none"}}]),
                stderr="",
            )
        if cmd[:3] == ["docker", "exec", "pb-demo"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    result = start_cleanroom_container("pb-demo", "programbench/demo", workspace, cpus="2", memory="8g")
    run_cmd = next(cmd for cmd in calls if cmd[:2] == ["docker", "run"])
    assert "--network" in run_cmd and "none" in run_cmd
    assert "-v" in run_cmd
    assert result["container_name"] == "pb-demo"
    assert result["preflight"]["network"] == "none"
    assert result["reference_probe"]["probe_returncode"] == 0


def test_programbench_prepare_seeded_workspace_moves_reference_and_sets_execute_bit(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF")

    layout = prepare_seeded_workspace(workspace)

    assert layout["reference_backend_path"] == "/workspace/reference_executable"
    assert (workspace / "reference_executable").is_file()
    assert not (workspace / "executable").exists()
    if sys.platform != "win32":  # execute bit is a POSIX concept (bench runs in Linux containers)
        assert (workspace / "reference_executable").stat().st_mode & 0o111
        assert (workspace / "reference_executable").stat().st_mode & 0o400
    gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
    assert "/reference_executable" in gitignore
    assert "/executable" in gitignore


def test_programbench_verify_reference_executable_runnable(monkeypatch):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    result = verify_reference_executable_runnable("pb-demo")
    assert result["probe_returncode"] == 0
    assert calls[0][0] == "docker"
    assert calls[0][2] == "pb-demo"
    assert "reference_executable" in calls[0][-1]


def test_programbench_terminal_status_reads_explicit_payload_status():
    assert terminal_task_status({"status": "completed"}) == "completed"
    assert terminal_task_status({"status": "failed"}) == "failed"
    assert terminal_task_status({"status": "running"}) == ""
    # cancel_requested is the cancel-intent latch, not the settled record.
    assert terminal_task_status({"status": "cancel_requested"}) == ""
    assert terminal_task_status({}) == ""
    # A completed task with stale provider noise in reason_code stays completed
    # (the harness must never demote it heuristically) but IS flagged as infra
    # noise for the ledger when the axes say so.
    assert terminal_task_status({"status": "completed", "reason_code": "provider_unavailable"}) == "completed"
    assert classify_infra_failure({"reason_code": "llm_api_error"}) is True
    assert classify_infra_failure({"outcome_axes": {"execution": {"status": "infra_failed"}}}) is True
    assert classify_infra_failure({"status": "failed", "reason_code": "task_not_completed"}) is False


def test_programbench_submit_and_wait_polls_until_terminal(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    calls: list[tuple[str, str]] = []

    def fake_api(base_url, method, path, body=None, **kwargs):
        calls.append((method, path))
        if method == "POST":
            return {"task_id": "task-123"}
        if len(calls) == 2:
            return {"task_id": "task-123", "status": "running"}
        return {"task_id": "task-123", "status": "completed", "result": "done"}

    monkeypatch.setattr(adapter, "ouroboros_api_request", fake_api)
    monkeypatch.setattr(adapter.time, "sleep", lambda *_args, **_kwargs: None)
    checkpoint = tmp_path / "checkpoint.json"
    result = submit_and_wait(
        "http://127.0.0.1:8765",
        {"description": "solve"},
        timeout_sec=30,
        poll_interval_sec=0,
        checkpoint_path=checkpoint,
    )
    assert result["status"] == "completed"
    assert calls[0] == ("POST", "/api/tasks")
    assert any(path.endswith("/api/tasks/task-123") for _, path in calls)
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["task_id"] == "task-123"
    assert saved["status"] == "completed"
    assert saved["task_result"]["result"] == "done"


def test_programbench_submit_and_wait_resumes_from_checkpoint_without_resubmit(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"task_id": "task-999", "status": "running"}), encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_api(base_url, method, path, body=None, **kwargs):
        calls.append((method, path))
        assert method == "GET", "a live checkpoint must re-attach, never re-submit"
        return {"task_id": "task-999", "status": "completed", "result": "done"}

    monkeypatch.setattr(adapter, "ouroboros_api_request", fake_api)
    monkeypatch.setattr(adapter.time, "sleep", lambda *_args, **_kwargs: None)
    result = submit_and_wait(
        "http://127.0.0.1:8765",
        {"description": "solve"},
        timeout_sec=30,
        poll_interval_sec=0,
        checkpoint_path=checkpoint,
    )
    assert result["status"] == "completed"
    assert calls == [("GET", "/api/tasks/task-999")]


def test_programbench_submit_and_wait_stale_checkpoint_falls_back_to_fresh_submit(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"task_id": "task-gone", "status": "running"}), encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_api(base_url, method, path, body=None, **kwargs):
        calls.append((method, path))
        if path.endswith("/api/tasks/task-gone"):
            raise RuntimeError("Ouroboros API GET /api/tasks/task-gone failed (404): task not found")
        if method == "POST":
            return {"task_id": "task-new"}
        return {"task_id": "task-new", "status": "completed"}

    monkeypatch.setattr(adapter, "ouroboros_api_request", fake_api)
    monkeypatch.setattr(adapter.time, "sleep", lambda *_args, **_kwargs: None)
    result = submit_and_wait(
        "http://127.0.0.1:8765",
        {"description": "solve"},
        timeout_sec=30,
        poll_interval_sec=0,
        checkpoint_path=checkpoint,
    )
    assert result["status"] == "completed"
    assert ("POST", "/api/tasks") in calls
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["task_id"] == "task-new"


# Registry-derived so a newly registered provider can never leak ambient routing.
from ouroboros.provider_models import PROVIDER_CREDENTIAL_GROUPS as _CRED_GROUPS

_PROVIDER_ROUTE_ENV_KEYS = tuple(
    key for group in _CRED_GROUPS.values() for key in group
)


def _scrub_model_route_env(monkeypatch):
    from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS

    for key in (*_PROVIDER_ROUTE_ENV_KEYS, *MODEL_SLOT_KEYS):
        monkeypatch.delenv(key, raising=False)


def test_programbench_model_preflight_rejects_legacy_ids_on_direct_route(tmp_path, monkeypatch):
    from devtools.benchmarks.programbench.run_programbench_e2e import preflight_model_slots

    _scrub_model_route_env(monkeypatch)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"OPENAI_API_KEY": "test-key", "OUROBOROS_MODEL": "openai/gpt-5.5-mini"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="openai::gpt-5.5-mini"):
        preflight_model_slots(settings)


def test_programbench_model_preflight_keeps_openrouter_ids_and_checks_solve_model(tmp_path, monkeypatch):
    from devtools.benchmarks.common.model_slots import single_model_subagents_setting
    from devtools.benchmarks.programbench.run_programbench_e2e import preflight_model_slots

    _scrub_model_route_env(monkeypatch)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "OPENROUTER_API_KEY": "test-key",
                "OUROBOROS_MODEL": "openai/gpt-5.5-mini",
                "OUROBOROS_SUBAGENTS": single_model_subagents_setting("openai/gpt-5.5-mini"),
                "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5-mini,openai/gpt-5.5-mini",
            }
        ),
        encoding="utf-8",
    )
    # provider/model is the canonical OpenRouter form: no rewrite, no error.
    slots = preflight_model_slots(settings, solve_model="openai/gpt-5.5-mini")
    assert slots["OUROBOROS_MODEL"] == "openai/gpt-5.5-mini"
    assert slots["OUROBOROS_REVIEW_MODELS"] == "openai/gpt-5.5-mini,openai/gpt-5.5-mini"
    with pytest.raises(SystemExit, match="does not match settings OUROBOROS_MODEL"):
        preflight_model_slots(settings, solve_model="anthropic/claude-sonnet-4.6")


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


def test_terminal_bench_harbor_adapter_reads_canonical_version(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    monkeypatch.setattr(tb_agent, "_repo_root", lambda: tmp_path)
    (tmp_path / "VERSION").write_text("6.64.2\n", encoding="utf-8")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path / "logs")

    assert agent.version() == "6.64.2"
    (tmp_path / "VERSION").unlink()
    assert agent.version() is None


def test_terminal_bench_harbor_context_uses_physical_metrics(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, task_timeout_sec=900)
    monkeypatch.setattr(agent, "_container_env", lambda: {})
    monkeypatch.setattr(agent, "_enforce_container_secret_policy", lambda _env: None)
    monkeypatch.setattr(agent, "_openrouter_credit_preflight", lambda _settings: None)
    monkeypatch.setattr(agent, "_host_settings", lambda: {})

    async def _noop(*_args, **_kwargs):
        return None

    async def _run(*_args, **_kwargs):
        return {"cost_usd": 0.2, "prompt_tokens": 10, "completion_tokens": 5}

    async def _physical(*_args, **_kwargs):
        return {
            "cost_usd": 0.6,
            "prompt_tokens": 34,
            "completion_tokens": 14,
            "cached_tokens": 13,
            "cost_final": True,
            "accounting_authority": "physical_attempt_ledger",
        }

    for name in (
        "_network_preflight",
        "_resolve_workspace_dir",
        "_ensure_workspace_git_root",
        "_start_server",
        "_capture_current_task_summary",
        "_stop_server",
    ):
        monkeypatch.setattr(agent, name, _noop)
    monkeypatch.setattr(agent, "_run_ouroboros_task", _run)
    monkeypatch.setattr(agent, "_emit_trajectory", _physical)

    class Environment:
        async def upload_file(self, *_args, **_kwargs):
            return None

    context = SimpleNamespace(metadata={})
    asyncio.run(agent.run("Solve it", Environment(), context))

    assert context.cost_usd == 0.6
    assert context.n_input_tokens == 34
    assert context.n_output_tokens == 14
    assert context.n_cache_tokens == 13
    assert context.metadata["summary"]["cost_final"] is True


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
    # Every shipped default is read from the config SSOT and must be visible.
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    for helper in OPENROUTER_REVIEW_DEFAULTS["triad"]:
        assert helper in meta
    assert "scope_review" not in meta
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
    actor = json.loads(env["OUROBOROS_SUBAGENTS"])["items"][0]["route"]
    assert actor == {"kind": "api_model", "target_id": "openai/gpt-5.5"}
    assert "OUROBOROS_MODEL_HEAVY" not in env
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


def test_terminal_bench_openrouter_credit_preflight_uses_authoritative_limit_remaining(tmp_path, monkeypatch):
    """v6.79.0: the preflight reads `/api/v1/key` `limit_remaining` through the shared helper.

    The old `/api/v1/credits` arithmetic (`total_credits − total_usage`) is the metric documented
    to lie on a nearly exhausted key, so this pins BOTH facts: the endpoint actually called, and
    that the credits-style body no longer decides anything."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    calls = []

    class _Response:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._body

    def fake_urlopen(req, timeout=0):
        assert req.headers["Authorization"] == "Bearer or-key"
        calls.append(req.full_url)
        # A body that the DEAD credits arithmetic would have read as $10 of headroom.
        return _Response(b'{"data":{"limit_remaining":0.25,"total_credits":10,"total_usage":0}}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, openrouter_min_credit_usd=1.0)

    with pytest.raises(RuntimeError, match="remaining \\$0.25 below threshold \\$1.00"):
        agent._openrouter_credit_preflight({})

    assert calls == ["https://openrouter.ai/api/v1/key"]
    payload = json.loads((tmp_path / "openrouter-credit-preflight.json").read_text(encoding="utf-8"))
    assert payload["remaining_usd"] == 0.25
    assert payload["source"] == "openrouter:/api/v1/key:limit_remaining"


def test_terminal_bench_openrouter_preflight_admits_an_uncapped_key(tmp_path, monkeypatch):
    """`limit: null` means NO cap, not "$0 left" — an uncapped key must not be refused."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"data":{"limit":null,"usage":123.0}}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: _Response())
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, openrouter_min_credit_usd=1.0)

    agent._openrouter_credit_preflight({})

    payload = json.loads((tmp_path / "openrouter-credit-preflight.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True and payload["uncapped"] is True and payload["remaining_usd"] is None


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
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
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
def test_swe_pro_capture_excludes_base_untracked_snapshot(tmp_path):
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    (repo / "auth.yaml").write_text("pre-existing secret-ish fixture\n", encoding="utf-8")
    (repo / "new_agent_file.py").write_text("print('agent-created')\n", encoding="utf-8")
    snapshot = tmp_path / "base_untracked.snapshot"
    snapshot.write_bytes(b"auth.yaml\0")
    capture = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"
    out = tmp_path / "patch.diff"

    subprocess.run(
        ["bash", str(capture), str(repo), base, str(out), str(snapshot)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    patch = out.read_text(encoding="utf-8")
    post_status = (tmp_path / "patch.status.post.txt").read_text(encoding="utf-8")

    assert "auth.yaml" not in patch
    assert "new_agent_file.py" in patch
    assert "auth.yaml" not in post_status
    assert "new_agent_file.py" in post_status


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
            "--allow-dirty-seed",
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
            "--allow-dirty-seed",
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
            "--allow-dirty-seed",
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
            "--allow-dirty-seed",
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
            "--allow-dirty-seed",
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

    monkeypatch.setattr(sys, "argv", ["swebench_predictions.py", "--allow-dirty-seed", "--input", str(input_jsonl), "--output", str(REPO_ROOT / "devtools" / "bad.jsonl")])
    with pytest.raises(ValueError, match="benchmark run output must not be under repo"):
        swe_predictions.main()

    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(REPO_ROOT / "devtools" / "bad_run")])
    with pytest.raises(ValueError, match="benchmark run output must not be under repo"):
        harbor_smoke.main()

    live_data = tmp_path / "live-data"
    live_data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(live_data))
    with pytest.raises(ValueError, match="live runtime data"):
        ensure_file_output_outside_repo(live_data / "bench" / "result_index.jsonl", REPO_ROOT)

    monkeypatch.setattr(sys, "argv", ["swebench_predictions.py", "--allow-dirty-seed", "--input", str(input_jsonl), "--output", str(live_data / "predictions.jsonl")])
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
            # State-independent: this asserts ledger/denominator behaviour, not the seed gate.
            "--allow-dirty-seed",
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
        ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
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
        ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--task", "task-a", "--execute"],
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
        ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
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
    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--execute"])

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
    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--n-tasks", "2", "--execute"])

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
        ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
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
    assert "task_timeout = 795" in runner  # 900 - _DEADLINE_SAFETY_SEC (105)
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


def test_terminal_bench_run_tb_builds_required_agent_kwargs(tmp_path, monkeypatch):
    import json as _json

    from devtools.benchmarks.terminal_bench.run_harbor_smoke import AGENT_IMPORT
    from devtools.benchmarks.terminal_bench.run_tb import HarborCommandConfig, harbor_command

    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "medium")
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
    # The agent MUST go through a job config (-c): the bare --agent-import-path
    # flag records agents[0].name = null, which the TB2.1 leaderboard static
    # analysis can never match (terminal-bench-2-1#121).
    assert "--agent-import-path" not in cmd
    assert "--agent-kwarg" not in cmd
    assert "--config" in cmd
    cfg_path = cmd[cmd.index("--config") + 1]
    agent_cfg = _json.loads(open(cfg_path, encoding="utf-8").read())["agents"][0]
    assert agent_cfg["name"] == "Ouroboros Installed"
    assert agent_cfg["import_path"] == AGENT_IMPORT
    assert agent_cfg["model_name"] == "ouroboros-openai-gpt-5.5"
    kw = agent_cfg["kwargs"]
    assert kw["task_review_mode"] == "required"
    assert kw["ouroboros_light_model"] == "google/gemini-3.5-flash"
    assert kw["disable_agent_web"] is True
    # Effort labeling: OUROBOROS_EFFORT_TASK becomes the declared submission
    # effort; the adapter forwards it back into the container env.
    assert kw["reasoning_effort"] == "medium"
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


def test_harbor_agent_defaults_max_workers_four_and_probes_context_timeout(tmp_path):
    """6c: 4 decomposition slots for the agent's own subagents (root takes one
    lane; container memory caps the pool — plan review needs no pool);
    6d: per-task timeout adopted from the harbor AgentContext when a future
    harbor exposes it (today: metadata probe)."""
    import types as _types

    from devtools.benchmarks.terminal_bench.harbor_installed_agent import (
        OuroborosTerminalBenchAgent,
    )

    agent = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(tmp_path / "settings.json"),
    )
    assert agent.max_workers == 4
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


def test_bench_template_scaffold_defaults_v655(tmp_path):
    """v6.55.0 shared bench-template decisions: safety light inside the jail,
    claude_code_edit disabled regardless of the web gate, the raised
    finalization margin, and the workers=4 templates across GAIA/SWE-pro."""
    import json as _json
    import pathlib as _pathlib

    from devtools.benchmarks.terminal_bench.harbor_installed_agent import (
        OuroborosTerminalBenchAgent,
    )

    agent = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(tmp_path / "settings.json"),
    )
    env = agent._container_env()
    assert env["OUROBOROS_SAFETY_MODE"] == "light"
    assert env["OUROBOROS_MAX_WORKERS"] == "4"
    # claude_code_edit is withheld in BOTH web modes; the web group must mirror
    # the registry's REAL _WEB_TOOLS set (the adapter list had drifted when
    # youtube_transcript joined _WEB_TOOLS in v6.52.1), and view_image stays
    # available.
    from ouroboros.tools.registry import _WEB_TOOLS

    assert set(OuroborosTerminalBenchAgent._WEB_TOOLS_MIRROR) == set(_WEB_TOOLS)
    web_off = agent._disabled_tools()
    assert web_off[-2:] == ["claude_code_edit", "schedule_subagent"]
    assert set(_WEB_TOOLS) <= set(web_off)
    assert {"analyze_screenshot", "vlm_query"} <= set(web_off)
    assert "view_image" not in web_off
    agent.disable_agent_web = False
    assert agent._disabled_tools() == ["claude_code_edit", "schedule_subagent"]
    assert OuroborosTerminalBenchAgent._DEADLINE_SAFETY_SEC == 105

    bench_root = _pathlib.Path(__file__).resolve().parents[1] / "devtools" / "benchmarks"
    gaia = _json.loads((bench_root / "gaia" / "settings_base.json").read_text(encoding="utf-8"))
    assert gaia["OUROBOROS_MAX_WORKERS"] == 4
    assert gaia["OUROBOROS_SAFETY_MODE"] == "light"
    swepro = _json.loads((bench_root / "swe_bench_pro" / "e1v2" / "settings_base.json").read_text(encoding="utf-8"))
    assert swepro["OUROBOROS_MAX_WORKERS"] == 4
    assert swepro["OUROBOROS_SAFETY_MODE"] == "light"
    assert swepro["OUROBOROS_RUNTIME_MODE"] == "pro"


def test_gaia_runner_default_workers_four_strict_baseline_ablation():
    """run_gaia defaults to the disclosed 4-slot worker pool; an explicit
    --max-workers 1 remains the strict-baseline ablation (no silent bump)."""
    import argparse
    import inspect

    from devtools.benchmarks.gaia import run_gaia as rg

    # Pin the runner's own parser default (source-level: main() builds the
    # parser inline, and invoking main() would launch inspect_ai).
    main_src = inspect.getsource(rg.main)
    assert '"--max-workers", type=int, default=4' in main_src

    args = argparse.Namespace(
        profile="quality_openrouter_web", disable_tools=None,
        websearch_backend="", main_web_search="", main_web_search_engine="",
        max_workers=1,
    )
    rg._apply_profile_defaults(args)
    assert args.max_workers == 1  # explicit strict baseline is preserved
    assert "claude_code_edit" in args.disable_tools


def test_gaia_requested_task_ids_honors_sample_id_and_argv_lockstep():
    # The manifest denominator must match what build_inspect_argv actually runs:
    # --sample-id records those exact ids; otherwise the limit-derived level list.
    from devtools.benchmarks.gaia import run_gaia

    sel = SimpleNamespace(sample_id="A, B ,C", split="validation", level=2, limit=99)
    assert run_gaia._requested_task_ids(sel) == ["A", "B", "C"]
    # argv path mirrors it (uses --sample-id, NOT --limit)
    argv_sel = run_gaia.build_inspect_argv(
        SimpleNamespace(sample_id="A,B,C", split="validation", level=2, limit=99,
                        max_samples=1, max_sandboxes=1, epochs=1),
        Path("/tmp/gaia-run"),
    )
    assert "--sample-id" in argv_sel and "--limit" not in argv_sel

    nolist = SimpleNamespace(sample_id="", split="validation", level=1, limit=2)
    assert run_gaia._requested_task_ids(nolist) == ["validation:level1:1", "validation:level1:2"]
    argv_lim = run_gaia.build_inspect_argv(
        SimpleNamespace(sample_id="", split="validation", level=1, limit=2,
                        max_samples=1, max_sandboxes=1, epochs=1),
        Path("/tmp/gaia-run"),
    )
    assert "--limit" in argv_lim and "--sample-id" not in argv_lim


# --- GAIA anti-lookup + leakage audit v2 + full-trace harness capture (2026-07-04) ---

def test_gaia_anti_leak_instruction_shape_and_all_solvers():
    """The SSOT anti-lookup instruction must (a) exist, (b) NOT name the benchmark
    or contain the FINAL ANSWER marker, (c) not self-trip the leak-query regex, and
    (d) be appended by all four solvers alongside the format instruction."""
    from devtools.benchmarks.gaia.inspect_solver import (
        GAIA_ANTI_LEAK_INSTRUCTION,
        GAIA_FORMAT_INSTRUCTION,
    )
    from devtools.benchmarks.gaia.leak_targets import LEAK_QUERY_RE

    assert GAIA_ANTI_LEAK_INSTRUCTION.strip()
    assert "gaia" not in GAIA_ANTI_LEAK_INSTRUCTION.lower()
    assert "FINAL ANSWER" not in GAIA_ANTI_LEAK_INSTRUCTION
    # neither SSOT instruction may match the answer-hunting query regex (self-flag guard)
    assert not LEAK_QUERY_RE.search(GAIA_ANTI_LEAK_INSTRUCTION)
    assert not LEAK_QUERY_RE.search(GAIA_FORMAT_INSTRUCTION)

    gaia_dir = REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "inspect_solver"
    for fname in ("ouroboros_solver.py", "codex_solver.py", "hermes_solver.py", "claude_code_solver.py"):
        src = (gaia_dir / fname).read_text(encoding="utf-8")
        assert "GAIA_ANTI_LEAK_INSTRUCTION" in src, f"{fname} does not append the anti-leak instruction"


def test_gaia_epistemic_instruction_shape_and_all_solvers():
    """v6.79.0 (owner Q20=1+4 / Q22): the epistemic-grounding rule is a GAIA-adapter prompt
    constant appended by all four solvers, under the same wording locks as the anti-leak text.

    It is a DISCLOSURE duty, not a retrieval duty — the owner's stated worry was Ouroboros
    googling trivia it already knows — so the text must not order the agent to search."""
    from devtools.benchmarks.gaia.inspect_solver import (
        GAIA_ANTI_LEAK_INSTRUCTION,
        GAIA_EPISTEMIC_INSTRUCTION,
        GAIA_FORMAT_INSTRUCTION,
    )
    from devtools.benchmarks.gaia.leak_targets import LEAK_QUERY_RE

    assert GAIA_EPISTEMIC_INSTRUCTION.strip()
    assert GAIA_EPISTEMIC_INSTRUCTION not in (GAIA_ANTI_LEAK_INSTRUCTION, GAIA_FORMAT_INSTRUCTION)
    assert "gaia" not in GAIA_EPISTEMIC_INSTRUCTION.lower()
    assert "FINAL ANSWER" not in GAIA_EPISTEMIC_INSTRUCTION
    assert not LEAK_QUERY_RE.search(GAIA_EPISTEMIC_INSTRUCTION)
    lowered = GAIA_EPISTEMIC_INSTRUCTION.lower()
    # Disclosure, not a search mandate: it must not demand searching/browsing, and it must
    # keep the explicit carve-out for facts the model already knows.
    for banned in ("search the web", "always search", "must search", "use web_search", "browse the web"):
        assert banned not in lowered, banned
    assert "already know" in lowered
    assert "unverified" in lowered

    gaia_dir = REPO_ROOT / "devtools" / "benchmarks" / "gaia" / "inspect_solver"
    for fname in ("ouroboros_solver.py", "codex_solver.py", "hermes_solver.py", "claude_code_solver.py"):
        src = (gaia_dir / fname).read_text(encoding="utf-8")
        assert "GAIA_EPISTEMIC_INSTRUCTION" in src, f"{fname} does not append the epistemic instruction"

    # The leakage audit strips every SSOT instruction before scanning, so an echoed prompt
    # cannot self-flag a sample.
    from devtools.benchmarks.gaia import audit_leakage as audit

    assert GAIA_EPISTEMIC_INSTRUCTION in audit._PROMPT_BOILERPLATE
    assert audit._strip_prompt_boilerplate("Q." + GAIA_EPISTEMIC_INSTRUCTION).strip() == "Q."


def test_epistemic_rule_stays_out_of_the_global_system_prompt():
    """Owner Q20/Q22 scoped the rule to the GAIA adapter ONLY: no global grounding duty in
    `prompts/SYSTEM.md` (it would push the runtime into searching for trivia) and no typed
    contract field. This is the invariant that keeps a future 'while we are here' edit honest."""
    system_md = (REPO_ROOT / "prompts" / "SYSTEM.md").read_text(encoding="utf-8").lower()
    for banned in (
        "epistemic honesty",
        "source your external claims",
        "source your claims",
        "cite a primary source",
        "check it against a primary source",
    ):
        assert banned not in system_md, f"SYSTEM.md must not carry the GAIA grounding rule: {banned}"

    contracts = (REPO_ROOT / "ouroboros" / "contracts" / "task_contract.py").read_text(encoding="utf-8")
    assert "epistemic" not in contracts.lower(), "Q20/Q22 explicitly rejected a typed contract field"


def test_gaia_claude_code_solver_uses_stream_json_and_writes_trace(monkeypatch, tmp_path):
    from devtools.benchmarks.gaia.inspect_solver import claude_code_solver as cc

    seen = {}
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch", "input": {"query": "python docs"}}]}},
        {"type": "result", "result": "FINAL ANSWER: 42", "total_cost_usd": 0.12, "usage": {"output_tokens": 5}, "is_error": False},
    ]
    raw = "\n".join(json.dumps(e) for e in events)

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout=raw, stderr="")

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    trace = tmp_path / "claude_code_trace.jsonl"
    result = cc.run_claude_code("q", sample_id="s", trace_path=trace)
    assert "stream-json" in seen["cmd"]
    assert "--verbose" in seen["cmd"]
    assert result["final_answer"] == "42"
    assert result["cost_usd"] == 0.12
    assert trace.read_text(encoding="utf-8") == raw  # full NDJSON dump captured for the audit


def test_gaia_codex_solver_uses_json_and_writes_trace(monkeypatch, tmp_path):
    from devtools.benchmarks.gaia.inspect_solver import codex_solver as cx

    seen = {}
    stdout = "\n".join(json.dumps(e) for e in [
        {"type": "item", "text": "searching"},
        {"type": "item", "tool": "web_search", "query": "python docs"},
    ])

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        work = Path(kwargs.get("cwd"))
        (work / ".codex_last_message.txt").write_text("FINAL ANSWER: 7", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(cx.subprocess, "run", fake_run)
    trace = tmp_path / "codex_trace.jsonl"
    result = cx.run_codex("q", sample_id="s", workdir=tmp_path / "wd", trace_path=trace)
    assert "--json" in seen["cmd"]
    assert result["final_answer"] == "7"
    assert trace.read_text(encoding="utf-8") == stdout


def test_gaia_leak_targets_match_real_cheats_and_spare_legit():
    from devtools.benchmarks.gaia.leak_targets import LEAK_QUERY_RE, LEAK_URL_RE

    # real cheat queries/URLs observed in the 2026-07-04 contaminated runs
    assert LEAK_QUERY_RE.search('GAIA benchmark "Thinking Machine" "sooner" scientist answer')
    assert LEAK_QUERY_RE.search('"Of the authors" "Pie Menus" "FINAL ANSWER"')
    assert LEAK_URL_RE.search("https://huggingface.co/spaces/agents-course/Final_Assignment_Template/raw/refs/pr/63/metadata.jsonl")
    assert LEAK_URL_RE.search("https://raw.githubusercontent.com/apooravmalik/GAIA-AI-AGENT/main/metadata.jsonl")
    assert LEAK_URL_RE.search("https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/GAIA_web.jsonl")
    assert LEAK_URL_RE.search("https://datasets-server.huggingface.co/rows?dataset=gaia")
    # legitimate content must NOT flag (ESA Gaia telescope, unrelated github, prompt echo)
    assert not LEAK_QUERY_RE.search("orbital period in the ESA Gaia telescope catalogue")
    assert not LEAK_URL_RE.search("https://github.com/psf/requests/blob/main/README.md")
    assert not LEAK_URL_RE.search("https://en.wikipedia.org/wiki/Gaia_(mythology)")


def test_gaia_audit_strip_boilerplate_prevents_self_flag():
    import devtools.benchmarks.gaia.audit_leakage as audit
    from devtools.benchmarks.gaia.inspect_solver import GAIA_ANTI_LEAK_INSTRUCTION

    # a trace that is ONLY the echoed anti-leak instruction must scan clean
    stripped = audit._strip_prompt_boilerplate("Query: solve this." + GAIA_ANTI_LEAK_INSTRUCTION)
    assert not audit.LEAK_QUERY_RE.search(stripped)


def test_gaia_audit_gold_verbatim_alone_is_weak_only(tmp_path):
    """Gold appearing in a NORMAL page is weak (not deterministically flagged);
    gold from a leak source is strong."""
    import devtools.benchmarks.gaia.audit_leakage as audit

    # one act: gold present, but no leak URL in results -> weak, not flagged
    weak_act = {"tool": "web_search", "requested_leak_urls": [], "suspicious_query": False,
                "result_leak_refs": [], "result_text": "The population is 883305 people.", "args_text": ""}
    strong_act = {"tool": "browse_page", "requested_leak_urls": [], "suspicious_query": False,
                  "result_leak_refs": ["https://huggingface.co/datasets/gaia-benchmark/GAIA"],
                  "result_text": "answer: 883305", "args_text": ""}
    gold = "883305"
    # replicate the row logic's gold classification
    def classify(acts):
        gold_verbatim = gold_from_leak = False
        for a in acts:
            if gold in a["result_text"]:
                gold_verbatim = True
                if a["result_leak_refs"]:
                    gold_from_leak = True
        return gold_verbatim, gold_from_leak
    gv, gfl = classify([weak_act])
    assert gv and not gfl
    gv2, gfl2 = classify([strong_act])
    assert gv2 and gfl2
    assert audit._distinctive_gold(gold)



def test_gaia_score_leakage_adjusted(tmp_path):
    from devtools.benchmarks.gaia import score_gaia

    run_dir = tmp_path / "run"
    (run_dir / "inspect_logs").mkdir(parents=True)
    log = {"samples": [
        {"id": "s1", "output": {"completion": "a"}, "scores": {"gaia_scorer": {"value": "C"}}},
        {"id": "s2", "output": {"completion": "b"}, "scores": {"gaia_scorer": {"value": "C"}}},
        {"id": "s3", "output": {"completion": "c"}, "scores": {"gaia_scorer": {"value": "I"}}},
    ]}
    (run_dir / "inspect_logs" / "log.json").write_text(json.dumps(log), encoding="utf-8")
    # s1 is a STRONG-flagged (cheated) sample
    audit_rows = [
        {"sample_id": "s1", "deterministic_flag": True},
        {"sample_id": "s2", "deterministic_flag": False},
        {"sample_id": "s3", "deterministic_flag": False},
    ]
    audit_path = run_dir / "leakage_audit.jsonl"
    audit_path.write_text("\n".join(json.dumps(r) for r in audit_rows), encoding="utf-8")
    summary = score_gaia.summarize(run_dir, leakage_audit=audit_path)
    assert summary["official_correct"] == 2
    assert summary["official_accuracy"] == 2 / 3
    assert summary["leakage_flagged_among_scored"] == 1
    assert summary["leakage_adjusted_correct"] == 1  # s1 zeroed
    assert summary["leakage_adjusted_accuracy"] == 1 / 3


def test_gaia_bwrap_isolate_masks_answer_cache_and_fails_loud(monkeypatch):
    """bwrap prefix masks the GAIA answer-cache dirs when enabled; fails loudly if
    bwrap is missing; no-op when disabled."""
    import devtools.benchmarks.gaia.bwrap_isolate as bw

    # disabled -> passthrough
    monkeypatch.setenv("GAIA_BWRAP_ISOLATE", "0")
    assert bw.wrap(["codex", "exec"]) == ["codex", "exec"]

    # enabled + bwrap present -> prefix wraps the command and masks the cache dirs
    monkeypatch.setenv("GAIA_BWRAP_ISOLATE", "1")
    monkeypatch.setattr(bw.shutil, "which", lambda _n: "/usr/bin/bwrap")
    monkeypatch.setattr(bw, "_mask_dirs", lambda: ["/home/u/.cache/inspect_evals"])
    wrapped = bw.wrap(["codex", "exec", "q"])
    assert wrapped[0] == "/usr/bin/bwrap"
    assert wrapped[-3:] == ["codex", "exec", "q"]
    assert "--tmpfs" in wrapped and "/home/u/.cache/inspect_evals" in wrapped
    assert "--" in wrapped and wrapped.index("--") < wrapped.index("codex")

    # enabled + bwrap missing -> loud failure (never silently unprotected)
    monkeypatch.setattr(bw.shutil, "which", lambda _n: None)
    with pytest.raises(SystemExit):
        bw.wrap(["codex", "exec"])


def test_gaia_sandbox_declarations_are_confined_to_shared_files(tmp_path, capsys):
    # commit triad sol #3 (anti-cheat): traversal/off-root declarations are
    # dropped loudly and never reach sandbox().read_file or the typed error.
    import asyncio
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    state = SimpleNamespace(files={
        "/shared_files/../../tests/secret": "x",
        "/etc/passwd": "x",
        "relative/doc.pdf": "x",
    }, metadata={})
    prompt = "see /shared_files/../hidden.bin too"
    out = asyncio.run(ouroboros_solver._stage_sandbox_attachments(
        state, tmp_path / "s", [], prompt=prompt,
    ))
    assert out == []  # nothing staged, NO GaiaAttachmentStagingError (no DoS)
    err = capsys.readouterr().err
    assert "non-confined attachment declaration" in err


def test_gaia_sandbox_read_success_path_stages_bytes_and_provenance(tmp_path, monkeypatch):
    # commit triad r2 #3: exercise the SUCCESSFUL sandbox().read_file path.
    import asyncio
    import json as _json
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    class _FakeSandbox:
        async def read_file(self, path, text=True):
            assert path == "/shared_files/2023/validation/doc.pdf"
            assert text is False
            return b"%PDF-SANDBOX"

    # inspect_ai is an optional benchmark dep absent on CI runners: inject a
    # fake module so the solver's in-function import resolves everywhere.
    import sys
    import types as _types
    fake_util = _types.ModuleType("inspect_ai.util")
    fake_util.sandbox = lambda *a, **k: _FakeSandbox()
    fake_pkg = _types.ModuleType("inspect_ai")
    fake_pkg.util = fake_util
    monkeypatch.setitem(sys.modules, "inspect_ai", fake_pkg)
    monkeypatch.setitem(sys.modules, "inspect_ai.util", fake_util)

    state = SimpleNamespace(metadata={})  # real TaskState shape: no files attr
    prompt = "Please read /shared_files/2023/validation/doc.pdf and answer."
    out = asyncio.run(ouroboros_solver._stage_sandbox_attachments(
        state, tmp_path / "s", [], prompt=prompt,
    ))
    assert len(out) == 1
    staged = out[0]
    assert staged.read_bytes() == b"%PDF-SANDBOX"
    assert staged.parent == (tmp_path / "s" / "attachments").resolve(strict=False) or staged.parent == tmp_path / "s" / "attachments"
    rows = _json.loads((tmp_path / "s" / "attachments" / "provenance.json").read_text())
    assert rows[-1]["method"] == "sandbox_read"
    assert rows[-1]["source"] == "/shared_files/2023/validation/doc.pdf"


def test_gaia_distinct_same_basename_declarations_both_stage(tmp_path, monkeypatch):
    # commit triad r2 advisory: /shared_files/a/doc.pdf and /shared_files/b/doc.pdf
    # must BOTH stage (uniquified names), not collapse on basename.
    import asyncio
    from types import SimpleNamespace
    from devtools.benchmarks.gaia.inspect_solver import ouroboros_solver

    class _FakeSandbox:
        async def read_file(self, path, text=True):
            return path.encode()

    import sys
    import types as _types
    fake_util = _types.ModuleType("inspect_ai.util")
    fake_util.sandbox = lambda *a, **k: _FakeSandbox()
    fake_pkg = _types.ModuleType("inspect_ai")
    fake_pkg.util = fake_util
    monkeypatch.setitem(sys.modules, "inspect_ai", fake_pkg)
    monkeypatch.setitem(sys.modules, "inspect_ai.util", fake_util)

    state = SimpleNamespace(metadata={})
    prompt = "see /shared_files/a/doc.pdf and /shared_files/b/doc.pdf"
    out = asyncio.run(ouroboros_solver._stage_sandbox_attachments(
        state, tmp_path / "s", [], prompt=prompt,
    ))
    assert len(out) == 2
    contents = sorted(p.read_bytes() for p in out)
    assert contents == [b"/shared_files/a/doc.pdf", b"/shared_files/b/doc.pdf"]


def test_programbench_instruction_states_tree_ships_as_is():
    """v6.74.4: the PB instruction must carry the true submission model (live
    tree, .git dropped, uncommitted edits ship) and the final compile.sh check,
    and must no longer claim a fresh checkout."""
    template = " ".join((
        Path(__file__).resolve().parents[1]
        / "devtools" / "benchmarks" / "programbench" / "instruction_template.md"
    ).read_text(encoding="utf-8").split())
    assert "CURRENT state of your working tree" in template
    assert "uncommitted edits DO ship" in template
    assert "The exporter also excludes" in template
    assert "`.ouroboros/`" in template and "at ANY depth" in template
    assert "run `./compile.sh` one final time" in template
    # The negated truth stays; the old false claim must be gone.
    assert "not from a fresh checkout" in template
    assert "on a fresh checkout" not in template


def test_programbench_submission_tarball_contract(tmp_path):
    """v6.74.4 (codex finding 1): the instruction's submission model must match
    the exporter — uncommitted source ships from the LIVE tree; .git, root
    binaries and build/cache noise do not."""
    import tarfile

    from devtools.benchmarks.programbench.programbench_adapter import (
        create_submission_tarball,
    )

    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (ws / "build").mkdir()
    (ws / "build" / "obj.o").write_text("obj")
    (ws / "figlet_clone.c").write_text("int main(void){return 0;}\n")  # uncommitted source
    (ws / ".ouroboros").mkdir()
    (ws / ".ouroboros" / "required.h").write_text("#define X 1\n")
    (ws / "compile.sh").write_text("#!/bin/sh\ncc figlet_clone.c -o executable\n")
    (ws / "executable").write_text("bin")
    (ws / "reference_executable").write_text("refbin")
    (ws / "probe.log").write_text("log")
    out = create_submission_tarball(ws, tmp_path / "sub.tar.gz")
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
    assert "figlet_clone.c" in names and "compile.sh" in names
    assert not any(n == "executable" or n == "reference_executable" for n in names)
    assert not any(n.startswith(".git") or n.startswith("build") for n in names)
    assert not any(n.startswith(".ouroboros") for n in names)
    assert "probe.log" not in names


# --------------------------------------------------------------------------------------
# v6.75.0 (P1) — run provenance: clean seed, runtime attestation, tri-state grading,
# append-only ledger, atomic sidecars, authoritative key headroom.
# --------------------------------------------------------------------------------------


def _git_commit_all(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "seed"],
        check=True,
        capture_output=True,
    )


def test_benchmark_manifest_seed_gate_fails_closed_by_default(tmp_path):
    """Owner Q19=B: an unreproducible seed refuses the run BY DEFAULT, with a recorded escape.

    Three refusal classes, all before any paid task: a dirty working tree (the manifest would
    say `-dirty` and the run would not be submittable), a checkout with no git identity at all
    (the source cannot be named), and a seed that does not match an explicit `expect` pin. The
    `expect` mismatch is NOT waivable by --allow-dirty-seed: 'dirty' and 'wrong commit' are
    different facts.
    """
    repo = tmp_path / "repo"
    _git_repo(repo)
    clean = benchmark_run_manifest(
        benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
    )
    assert clean["seed_gate"]["ok"] is True
    assert clean["seed_gate"]["require_clean"] is True
    assert clean["seed_gate"]["allow_dirty_seed"] is False

    head = clean["source"]["head"]
    pinned = benchmark_run_manifest(
        benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        expect=head[:12],
    )
    assert pinned["seed_gate"]["expect"] == head[:12]

    (repo / "app.py").write_text("print('dirty')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reason=seed_dirty"):
        benchmark_run_manifest(
            benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        )
    waived = benchmark_run_manifest(
        benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        require_clean=False,
    )
    assert waived["seed_gate"]["reason"] == "seed_dirty"
    assert waived["seed_gate"]["allow_dirty_seed"] is True

    with pytest.raises(RuntimeError, match="reason=seed_mismatch"):
        benchmark_run_manifest(
            benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
            require_clean=False, expect="0" * 40,
        )

    not_git = tmp_path / "plain"
    not_git.mkdir()
    with pytest.raises(RuntimeError, match="reason=seed_identity_unavailable"):
        benchmark_run_manifest(
            benchmark="unit", run_root=tmp_path / "run", repo_dir=not_git, requested_task_ids=["t"],
        )


def test_benchmark_seed_gate_refuses_when_cleanliness_cannot_be_determined(tmp_path):
    """The fourth refusal class: the cleanliness probe itself did not answer.

    `git status` can fail for real (a corrupt `.git/index`, or the 10s timeout on a huge
    untracked tree / CephFS). Coercing that into `dirty: False` let a genuinely dirty seed pass
    the gate with `seed_gate.ok: true`, which is exactly the `-dirty`-provenance run owner
    Q19=B exists to prevent. Reproduced with a REAL corrupted index, not a mock: `rev-parse
    HEAD` still works (so the seed has an identity) while `status` fails.
    """
    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "app.py").write_text("print('dirty and unreportable')\n", encoding="utf-8")
    (repo / ".git" / "index").write_bytes(b"DIRC\x00\x00\x00\xffnot-an-index")

    provenance = repo_provenance(repo)
    assert provenance["git_available"] is True          # the commit is still readable
    assert provenance["status_available"] is False      # the cleanliness probe is not
    assert provenance["dirty"] is False                 # ... and its value carries no information

    with pytest.raises(RuntimeError, match="reason=seed_status_unavailable"):
        benchmark_run_manifest(
            benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        )
    # The recorded escape keeps working and keeps saying WHY it was needed.
    waived = benchmark_run_manifest(
        benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        require_clean=False,
    )
    assert waived["seed_gate"]["reason"] == "seed_status_unavailable"
    assert waived["seed_gate"]["ok"] is False
    assert waived["seed_gate"]["status_available"] is False


def test_benchmark_write_json_is_atomic_and_byte_identical(tmp_path):
    """write_json became atomic without changing a single byte of any existing sidecar.

    The atomic helper defaults to NO trailing newline, so the call must pass
    trailing_newline=True — otherwise every manifest/ledger sidecar silently changes shape.
    Also asserts no temp sibling survives a successful write.
    """
    from devtools.benchmarks.common.manifests import write_json

    payload = {"b": 1, "a": ["x", "ю"], "nested": {"k": None}}
    target = tmp_path / "deep" / "run_manifest.json"
    write_json(target, payload)
    legacy = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    assert target.read_text(encoding="utf-8") == legacy
    assert sorted(p.name for p in target.parent.iterdir()) == ["run_manifest.json"]

    write_json(target, {"replaced": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"replaced": True}


def test_benchmark_manifests_module_stays_stdlib_only_at_import():
    """`common/manifests.py` is imported by every launcher, including the container-side
    Terminal-Bench agent, so the atomic-write dependency on the runtime package must be a LAZY
    import inside write_json — a module-level `import ouroboros` would make the runtime a hard
    dependency of all benchmark families."""
    source = (REPO_ROOT / "devtools" / "benchmarks" / "common" / "manifests.py").read_text(encoding="utf-8")
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "ouroboros" in line
    ]
    assert module_level == []

    # Cross-launcher import smoke: every P1-owned launcher imports the shared module cleanly.
    for module in (
        "devtools.benchmarks.common.manifests",
        "devtools.benchmarks.programbench.run_programbench",
        "devtools.benchmarks.programbench.run_programbench_e2e",
        "devtools.benchmarks.swe_bench.swebench_predictions",
        "devtools.benchmarks.swe_bench_pro.pro_predictions",
        "devtools.benchmarks.harness_bench_fast.run_harness_bench_fast",
    ):
        importlib.import_module(module)


def test_openrouter_key_remaining_uses_authoritative_field(monkeypatch):
    """`limit_remaining` is the source of truth; `limit - usage` is only a FALLBACK, and an
    uncapped key is None (not 0.0, not 'plenty'). The credit-endpoint arithmetic this replaces
    lied on a nearly exhausted key and burned half a run."""
    from devtools.benchmarks.common.manifests import openrouter_key_remaining

    bodies: list[bytes] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return bodies.pop(0)

    def fake_urlopen(req, timeout=0):
        assert req.full_url == "https://openrouter.ai/api/v1/key"
        assert req.headers["Authorization"] == "Bearer or-key"
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    bodies.append(b'{"data":{"limit":100,"usage":97.5,"limit_remaining":0.23}}')
    assert openrouter_key_remaining("or-key") == 0.23
    bodies.append(b'{"data":{"limit":100,"usage":97.5}}')
    assert openrouter_key_remaining("or-key") == pytest.approx(2.5)
    bodies.append(b'{"data":{"limit":null,"usage":12.0}}')
    assert openrouter_key_remaining("or-key") is None

    with pytest.raises(RuntimeError, match="requires an API key"):
        openrouter_key_remaining("  ")


def test_runtime_attestation_records_both_facts_and_fails_closed(tmp_path, monkeypatch):
    """Owner Q7=B / Q8: record the HTTP runtime_version AND the local commit, and hard-stop on
    a skew unless the named override is set (the override is itself recorded)."""
    from devtools.benchmarks.common import manifests

    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "VERSION").write_text("6.75.0\n", encoding="utf-8")
    _git_commit_all(repo)

    served = {"runtime_version": "6.75.0"}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps(served).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.delenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, raising=False)

    ok = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert ok["ok"] is True and ok["reason"] == ""
    assert ok["runtime_version"] == "6.75.0"
    assert ok["repo_version"] == "6.75.0"
    assert len(ok["repo_head"]) == 40
    assert ok["overridden"] is False

    served["runtime_version"] = "6.74.5"
    with pytest.raises(RuntimeError, match="reason=runtime_skew"):
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)

    monkeypatch.setenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, "1")
    overridden = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert overridden["reason"] == "runtime_skew" and overridden["overridden"] is True
    assert overridden["ok"] is False


def test_runtime_attestation_override_waives_only_the_evolved_runtime_reason(tmp_path, monkeypatch):
    """`OBO_ALLOW_EVOLVED_VOLUME` authorises a deliberately evolved / version-skewed runtime and
    NOTHING else. It used to be applied to every failure reason, so with the override exported
    ProgramBench admission continued after an unreachable `/api/health` — the attestation gate
    fail-open the phase exists to remove. Per reason, with the override SET: `runtime_skew`
    proceeds and is recorded; `runtime_unreachable` (no live identity at all) and
    `commit_unavailable` (no commit to attribute the numbers to) still raise."""
    from devtools.benchmarks.common import manifests

    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "VERSION").write_text("6.75.0\n", encoding="utf-8")
    _git_commit_all(repo)

    served: dict = {"runtime_version": "6.74.5"}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps(served).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, "1")

    assert manifests.OVERRIDABLE_ATTESTATION_REASONS == ("runtime_skew",)

    skewed = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert skewed["reason"] == "runtime_skew"
    assert skewed["overridden"] is True and skewed["override_set"] is True
    assert skewed["override_waives"] == ["runtime_skew"]
    assert skewed["ok"] is False

    # (a) transport/parse failure -> no live runtime identity was established AT ALL.
    def _boom(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="reason=runtime_unreachable") as unreachable:
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert "does NOT waive" in str(unreachable.value)
    assert "override_set=True" in str(unreachable.value)

    # ... including a 200 whose body is not the health contract (parse failure, same class).
    class _Garbage(_Resp):
        def read(self):
            return b"<html>not json</html>"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Garbage())
    with pytest.raises(RuntimeError, match="reason=runtime_unreachable"):
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)

    # (b) no local commit -> nothing to attribute the numbers to. `repo_dir` outside git makes
    # `repo_head` empty, and the version pin removes the skew reason so the missing commit is the
    # one under test (no dependence on the AMBIENT checkout: this is a fresh tmp dir).
    served["runtime_version"] = "6.75.0"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    (bare / "VERSION").write_text("6.75.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reason=commit_unavailable") as no_commit:
        manifests.runtime_attestation("http://127.0.0.1:9/", bare, expected_version="6.75.0")
    assert "does NOT waive" in str(no_commit.value)


def test_runtime_attestation_lineage_allows_descendants_only(tmp_path):
    """Evolution legitimately moves HEAD forward, so provenance compares a LINE OF DESCENT
    (`merge-base --is-ancestor`), never equality — and an unknown commit is False, not
    'probably fine'."""
    from devtools.benchmarks.common.manifests import commit_lineage_ok

    repo = tmp_path / "repo"
    _git_repo(repo)
    seed = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    (repo / "evolved.py").write_text("print('evolved')\n", encoding="utf-8")
    _git_commit_all(repo)
    evolved = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()

    assert commit_lineage_ok(seed, seed, repo) is True
    assert commit_lineage_ok(seed, evolved, repo) is True
    assert commit_lineage_ok(evolved, seed, repo) is False
    assert commit_lineage_ok(seed, "", repo) is False
    assert commit_lineage_ok("0" * 40, evolved, repo) is False


def test_runtime_attestation_is_wired_into_url_attaching_readiness_paths():
    """Owner Q9=A+B: the shared helper exists AND every launcher that attaches to a live server
    URL calls it from its own readiness/admission path. This meta-test names the CONCRETE entry
    points, with their ARITY, so a call that would TypeError cannot pass as "wired". CLB's
    host-engine path is covered through IsolatedServer; the CLB-docker stand-in never calls
    `_wait_ready`, so its attestation arrives via the tracked operator patch and is asserted in
    `tests/test_continual_learning_launcher.py`. TB and GAIA are structurally immune (owner
    Q10) and deliberately have no lines here."""
    bench = REPO_ROOT / "devtools" / "benchmarks"
    wired = {
        # shared readiness seam: every IsolatedServer driver (evolve_smoke + CLB host engine)
        bench / "common" / "server_runner.py": "runtime_attestation(self.base_url, self.clone)",
        bench / "programbench" / "run_programbench_e2e.py": "runtime_attestation(str(args.ouroboros_url), repo_dir)",
        # OSWorld: the step loop attests inside `_preflight`, the cu_bridge before its first
        # POST /api/tasks, and the preflight-only skeleton alongside its reachability probes.
        bench / "osworld" / "run_step_agent.py": "runtime_attestation(config.ouroboros_url, config.repo_dir)",
        bench / "osworld" / "run_cu_bridge_agent.py": "runtime_attestation(args.ouroboros_url, repo_dir)",
        bench / "osworld" / "osworld_adapter_skeleton.py": "runtime_attestation(ouroboros_url, repo_root)",
    }
    for path, call in wired.items():
        assert call in path.read_text(encoding="utf-8"), f"{path.name} lost its attestation call"

    # SWE-Pro attests inside the container (it has no host-side URL): one-shot, after readiness
    # and before the paid solve.
    entrypoint = (bench / "swe_bench_pro" / "e1v2" / "entrypoint_pro.sh").read_text(encoding="utf-8")
    assert "/api/health" in entrypoint and "runtime_skew" in entrypoint

    # Every wired call above must actually BIND against the shared helper's signature: a
    # name-only check would pass a call missing the required `repo_dir` positional (which is
    # how the commit half of owner Q7=B is reported) and only fail at run time.
    import ast

    from devtools.benchmarks.common.manifests import runtime_attestation
    signature = inspect.signature(runtime_attestation)
    for call in wired.values():
        node = ast.parse(call, mode="eval").body
        signature.bind(*node.args, **{kw.arg: kw.value for kw in node.keywords})


def test_swe_pro_grade_reports_tri_state_verdicts(tmp_path, monkeypatch):
    """Owner Q17=B: an instance the official evaluator never scored is `ungraded`, not a FAIL.
    The official headline FORMULA is unchanged (pass over submitted); `ungraded=N/total` is
    printed next to it and the shrunken-denominator percentage is explicitly labelled
    diagnostic / not leaderboard-valid."""
    import devtools.benchmarks.swe_bench_pro.grade_pro as grade_pro

    eval_repo = tmp_path / "SWE-bench_Pro-os"
    helper = eval_repo / "helper_code"
    helper.mkdir(parents=True)
    (helper / "sweap_eval_full_v2.jsonl").write_text(
        json.dumps({"instance_id": "won", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []}) + "\n"
        + json.dumps({"instance_id": "lost", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []}) + "\n"
        + json.dumps({"instance_id": "crashed", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []}) + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"instance_id": iid, "model_patch": "diff --git a/a b/a\n", "model_name_or_path": "m"})
            for iid in ("won", "lost", "crashed", "not_in_dataset")
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    for iid, tests in (("won", [{"name": "t1", "status": "PASSED"}]), ("lost", [{"name": "t1", "status": "FAILED"}])):
        (out_dir / iid).mkdir(parents=True)
        (out_dir / iid / "ours_output.json").write_text(json.dumps({"tests": tests}), encoding="utf-8")
    # "crashed" has no official output at all -> ungraded, not a model failure.

    monkeypatch.setattr(
        sys, "argv",
        ["grade_pro.py", "--predictions", str(predictions), "--out-dir", str(out_dir),
         "--eval-repo", str(eval_repo), "--skip-run"],
    )
    assert grade_pro.main() == 0

    summary = json.loads((out_dir / "grade_summary.json").read_text(encoding="utf-8"))
    assert summary["submitted"] == 4
    assert summary["pass"] == 1
    assert summary["fail"] == 1
    assert summary["ungraded"] == 2
    assert summary["headline_raw_pass_at_1_pct"] == 25.0          # UNCHANGED formula: 1/4
    assert summary["diagnostic_pass_over_graded_pct"] == 50.0     # 1/2, labelled diagnostic
    assert summary["diagnostic_not_leaderboard_valid"] is True
    by_id = {row["instance_id"]: row for row in summary["verdicts"]}
    assert by_id["won"]["verdict"] == "pass"
    assert by_id["lost"]["verdict"] == "fail"
    assert by_id["crashed"]["verdict"] == "ungraded"
    assert by_id["crashed"]["reason"] == "no_official_output"
    assert by_id["not_in_dataset"]["reason"] == "instance_not_in_dataset"


def test_swe_pro_grade_ungraded_covers_unparseable_and_empty_requirements(tmp_path):
    """The other two ungraded classes: an official output we cannot parse, and a dataset row
    with no required tests (an empty `need` set used to silently read as FAIL)."""
    from devtools.benchmarks.swe_bench_pro.grade_pro import instance_verdict

    broken = tmp_path / "ours_output.json"
    broken.write_text("{not json", encoding="utf-8")
    verdict, reason, _ = instance_verdict(broken, {"FAIL_TO_PASS": ["t1"]})
    assert verdict == "ungraded" and reason.startswith("output_unparseable")

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"tests": [{"name": "t1", "status": "PASSED"}]}), encoding="utf-8")
    assert instance_verdict(empty, {"FAIL_TO_PASS": [], "PASS_TO_PASS": []})[:2] == ("ungraded", "no_required_tests")
    assert instance_verdict(empty, None)[:2] == ("ungraded", "instance_not_in_dataset")

    # Valid JSON with an UNEXPECTED SHAPE is also unparseable output, never a headline: the row
    # extraction has to sit inside the same guard as json.loads (a raised TypeError/KeyError here
    # would abort the whole grading pass).
    for payload in ({"tests": {"t1": "PASSED"}}, {"tests": [{"status": "PASSED"}]}, {"tests": [None]}):
        odd = tmp_path / f"odd_{abs(hash(str(payload)))}.json"
        odd.write_text(json.dumps(payload), encoding="utf-8")
        verdict, reason, column = instance_verdict(odd, {"FAIL_TO_PASS": ["t1"]})
        assert verdict == "ungraded" and reason.startswith("output_unparseable") and column == "-"


def _write_programbench_actor_settings(_e2e, path):
    from devtools.benchmarks.common.model_slots import pin_single_model

    model = "openai/gpt-5.5"
    payload = {}
    pin_single_model(model, target=payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_programbench_e2e_ledger_is_append_only_and_manifest_is_written_first(tmp_path, monkeypatch):
    """ProgramBench records target-bound manifests before append-only instance rows."""
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    run_root = tmp_path / "pb-run"
    settings = tmp_path / "settings.json"
    target_settings = _write_programbench_actor_settings(e2e, settings)
    instances = [{"instance_id": "inst-a", "image_name": "img-a"}, {"instance_id": "inst-b", "image_name": "img-b"}]
    monkeypatch.setattr(e2e, "_load_instances", lambda **_k: list(instances))
    monkeypatch.setattr(e2e, "runtime_attestation", lambda url, repo: {"ok": True, "runtime_version": "6.75.0"})
    monkeypatch.setattr(e2e, "ouroboros_api_request", lambda *_a, **_k: target_settings)
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: run_root)

    seen: list[str] = []

    def _fake_process(instance, cfg):
        seen.append(str(instance["instance_id"]))
        if len(seen) == 2:
            lines = (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()
            assert [json.loads(line)["instance_id"] for line in lines] == ["inst-a"]
            gate = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))["seed_gate"]
            assert set(gate) >= {"ok", "reason", "require_clean", "allow_dirty_seed", "dirty", "git_available"}
            assert gate["require_clean"] is False and gate["allow_dirty_seed"] is True
            assert gate["ok"] is (not gate["reason"])
        return e2e.task_result_row(
            benchmark="programbench", instance_id=str(instance["instance_id"]),
            status="completed", reason_code="submission_prepared",
        )

    monkeypatch.setattr(e2e, "_process_instance", _fake_process)
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench_e2e.py", "--allow-dirty-seed", "--settings-path", str(settings),
         "--ouroboros-url", "http://127.0.0.1:9"],
    )

    assert e2e.main() == 0
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["inst-a", "inst-b"]

    # A resume APPENDS; readers dedup by instance_id with the last row winning.
    monkeypatch.setattr(e2e, "_load_instances", lambda **_k: [instances[1]])
    monkeypatch.setattr(e2e, "_process_instance", lambda instance, cfg: e2e.task_result_row(
        benchmark="programbench", instance_id="inst-b", status="failed", reason_code="task_not_completed"))
    assert e2e.main() == 1
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["inst-a", "inst-b", "inst-b"]
    latest = {row["instance_id"]: row for row in rows}
    assert latest["inst-b"]["status"] == "failed"
    assert latest["inst-a"]["status"] == "completed"

    # Every processed row reached BOTH ledgers, which is the contract programbench/README.md
    # states without qualification.
    for iid in ("inst-a", "inst-b"):
        per_instance = (run_root / iid / "result_index.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["instance_id"] for line in per_instance] == [iid] * len(per_instance)

    # ... and so does a SKIP row. A resume narrows the work, never the ledger: the instance that
    # is skipped because it already has a submission gets its skip event appended at the run root
    # AND in its own directory. Only the run root was written, so a resumed instance's own history
    # silently omitted the resume while the README claimed both locations.
    submission = run_root / "inst-a" / "submission.tar.gz"
    submission.write_bytes(b"tarball")
    root_before = len((run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines())
    instance_before = len((run_root / "inst-a" / "result_index.jsonl").read_text(encoding="utf-8").splitlines())
    monkeypatch.setattr(e2e, "_load_instances", lambda **_k: list(instances))
    monkeypatch.setattr(e2e, "_process_instance", lambda instance, cfg: e2e.task_result_row(
        benchmark="programbench", instance_id=str(instance["instance_id"]),
        status="completed", reason_code="submission_prepared"))
    assert e2e.main() == 0

    root_rows = [json.loads(line) for line in
                 (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    instance_rows = [json.loads(line) for line in
                     (run_root / "inst-a" / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(root_rows) == root_before + 2          # inst-a skipped + inst-b processed
    assert len(instance_rows) == instance_before + 1
    assert instance_rows[-1]["status"] == "skipped"
    assert instance_rows[-1]["reason_code"] == "skipped_existing_submission"
    assert instance_rows[-1] == next(r for r in root_rows if r["status"] == "skipped")


def test_harness_bench_fast_manifest_is_durable_and_records_the_final_outcome(tmp_path, monkeypatch):
    """The third P1 launcher's half of the manifest lifecycle. It wrote its manifest inline and
    never touched it again, so a run's own record never said how the run ENDED. It is now built
    once, on disk before the harness subprocess starts (asserted from inside the subprocess
    stand-in, i.e. before anything is spent), retained, and rewritten with the final outcome and
    exit code — including the harness's own non-zero exit."""
    from devtools.benchmarks.harness_bench_fast import run_harness_bench_fast as hbf

    out_root = tmp_path / "hbf-run"
    manifest_path = out_root / "run_manifest.json"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hbf, "_read_task_ids", lambda root, ids, task_file="": ["task_1"])

    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["manifest_before_spend"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="")

    monkeypatch.setattr(hbf.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["run_harness_bench_fast.py", "--run-root", str(out_root), "--allow-dirty-seed",
         "--settings-path", str(settings), "--bench-root", str(tmp_path / "bench")],
    )

    assert hbf.main() == 7
    # Durable BEFORE the harness ran, and the seed gate's SHAPE is in it (never its verdict: `ok`
    # mirrors the ambient checkout and would flip between a dirty tree and clean CI).
    early = seen["manifest_before_spend"]
    assert early["extra"]["outcome"] == "started"
    assert set(early["seed_gate"]) >= {"ok", "reason", "require_clean", "allow_dirty_seed"}
    assert early["seed_gate"]["require_clean"] is False

    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert final["extra"]["outcome"] == "harness_nonzero_exit"
    assert final["extra"]["exit_code"] == 7
    assert final["requested_task_ids"] == ["task_1"]

    # A dry run records that it was a dry run rather than leaving `started` behind forever.
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--dry-run"])
    assert hbf.main() == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["extra"]["outcome"] == "dry_run"


def test_benchmark_admission_persists_the_refusal_before_enforcement_raises(tmp_path):
    """The provenance lifecycle is now ENFORCED, not promised.

    `_seed_gate` used to raise from inside `benchmark_run_manifest`, i.e. before the dict reached
    any caller, so no launcher could persist the refusal the contract promises: a refused run left
    nothing but a stderr line that a shard launcher discards. Admission now builds the COMPLETE
    payload, `admit_benchmark_run` writes it, and only then does enforcement raise — and the typed
    exception carries the same payload so any other caller can persist it too.
    """
    from devtools.benchmarks.common.manifests import (
        BenchmarkAdmissionRefused,
        admit_benchmark_run,
    )

    repo = tmp_path / "repo"
    _git_repo(repo)

    admitted_path = tmp_path / "admitted" / "run_manifest.json"
    admitted = admit_benchmark_run(
        admitted_path, benchmark="unit", run_root=tmp_path / "run", repo_dir=repo,
        requested_task_ids=["t"],
    )
    assert admitted["seed_gate"]["ok"] is True
    assert json.loads(admitted_path.read_text(encoding="utf-8"))["seed_gate"]["ok"] is True
    assert "refusal" not in admitted["extra"]

    (repo / "app.py").write_text("print('dirty')\n", encoding="utf-8")
    refused_path = tmp_path / "refused" / "run_manifest.json"
    with pytest.raises(BenchmarkAdmissionRefused, match="reason=seed_dirty") as refused:
        admit_benchmark_run(
            refused_path, benchmark="unit", run_root=tmp_path / "run", repo_dir=repo,
            requested_task_ids=["t"],
        )
    # Still a RuntimeError for every pre-existing caller, and the payload rode on the exception.
    assert isinstance(refused.value, RuntimeError)
    assert refused.value.manifest["seed_gate"]["reason"] == "seed_dirty"

    persisted = json.loads(refused_path.read_text(encoding="utf-8"))
    assert persisted["seed_gate"]["reason"] == "seed_dirty"
    assert persisted["seed_gate"]["ok"] is False
    assert persisted["requested_task_ids"] == ["t"]
    # Same terminal vocabulary a completed run uses, so both read the same way in an audit.
    assert persisted["extra"]["outcome"] == "refused"
    assert persisted["extra"]["exit_code"] == 1
    assert persisted["extra"]["refusal"] == {
        "stage": "seed_gate", "reason": "seed_dirty", "exit_code": 1}

    # The `expect` pin is refused even WITH the dirty-seed escape, and is just as durable.
    pinned_path = tmp_path / "pinned" / "run_manifest.json"
    with pytest.raises(BenchmarkAdmissionRefused, match="reason=seed_mismatch"):
        admit_benchmark_run(
            pinned_path, benchmark="unit", run_root=tmp_path / "run", repo_dir=repo,
            requested_task_ids=["t"], require_clean=False, expect="0" * 40,
        )
    pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    assert pinned["extra"]["refusal"]["reason"] == "seed_mismatch"
    assert pinned["seed_gate"]["allow_dirty_seed"] is True


def test_finalize_run_manifest_records_a_typed_outcome_on_every_exit_path(tmp_path):
    """The ONE finalization seam. Its whole point is the paths a launcher does NOT think about:
    an early typed return and an escaping exception. Several migrated launchers only ever updated
    counts, so their own record still said `started` after they had finished or died."""
    from devtools.benchmarks.common.manifests import finalize_run_manifest

    target = tmp_path / "deep" / "run_manifest.json"

    def _extra():
        return json.loads(target.read_text(encoding="utf-8"))["extra"]

    manifest = {"extra": {"outcome": "started"}}
    with finalize_run_manifest(target, manifest) as final:
        assert final["outcome"] == "completed"
    assert _extra() == {"outcome": "completed", "exit_code": 0}

    manifest = {"extra": {"outcome": "started"}}
    with finalize_run_manifest(target, manifest) as final:
        final.update({"outcome": "refused", "exit_code": 3,
                      "refusal": {"stage": "seed_shape", "reason": "seed_is_not_a_git_directory"}})
    recorded = _extra()
    assert recorded["outcome"] == "refused" and recorded["exit_code"] == 3
    assert recorded["refusal"]["stage"] == "seed_shape"

    manifest = {"extra": {"outcome": "started"}}
    with pytest.raises(ZeroDivisionError):
        with finalize_run_manifest(target, manifest):
            raise ZeroDivisionError("boom")
    recorded = _extra()
    assert recorded["outcome"] == "crashed" and recorded["exit_code"] == 1
    assert recorded["error"] == {"type": "ZeroDivisionError", "message": "boom"}

    # A launcher that NAMED its outcome before re-raising keeps that name; the typed error is
    # recorded NEXT to it rather than replacing it.
    manifest = {"extra": {}}
    with pytest.raises(RuntimeError):
        with finalize_run_manifest(target, manifest) as final:
            final.update({"outcome": "stopped_instance_error", "exit_code": 1})
            raise RuntimeError("instance blew up")
    recorded = _extra()
    assert recorded["outcome"] == "stopped_instance_error"
    assert recorded["error"]["type"] == "RuntimeError"

    # BaseException (SIGINT / SystemExit) must not slip past the seam either.
    manifest = {"extra": {}}
    with pytest.raises(KeyboardInterrupt):
        with finalize_run_manifest(target, manifest):
            raise KeyboardInterrupt
    recorded = _extra()
    assert recorded["outcome"] == "crashed" and recorded["error"]["type"] == "KeyboardInterrupt"

    # ... and a SystemExit keeps its REAL status: flattening it to 1 made the record disagree
    # with the code the process exits with (auto_run's campaign-fatal refusal exits 2).
    manifest = {"extra": {}}
    with pytest.raises(SystemExit):
        with finalize_run_manifest(target, manifest):
            raise SystemExit(2)
    recorded = _extra()
    assert recorded["outcome"] == "crashed" and recorded["exit_code"] == 2
    # A non-integer status (SystemExit("message")) has no numeric meaning -> generic failure.
    manifest = {"extra": {}}
    with pytest.raises(SystemExit):
        with finalize_run_manifest(target, manifest):
            raise SystemExit("no numeric status")
    assert _extra()["exit_code"] == 1


# The structural launcher gate (devtools/benchmarks/common/launcher_audit.py).
# Formerly test-local `ast` helpers knowing ONE launcher shape and ONE hop of
# LOCAL helpers; now a module auditing the real launchers plus a SYNTHETIC
# violating one — the only way to tell "the gate works" from "clean today".

# A synthetic launcher-shaped module for pinning the pre-admission resolver itself.
# Deliberately not a real launcher: the gate's BEHAVIOUR is what must not regress.
_GUARD_PROBE_SOURCE = '''
def _looks_innocent(path):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path)

def _two_levels_down(path):
    return _looks_innocent(path)

def _three_levels_down(path):
    return _two_levels_down(path)

def _pure(a, b):
    return f"{a}/{b}"

def _steps_aside(root):
    root.mkdir(parents=True, exist_ok=True)
    return None

def main():
    args = parse_args()
    if args.collect_only:
        _steps_aside(args.out)
        return 0
    label = _pure(args.a, args.b)
    provenance = _looks_innocent(args.repo)
    manifest = admit_benchmark_run(args.out, label=label, extra=provenance)
    return finish(manifest)
'''

# A synthetic launcher that violates BOTH invariants, in the exact shapes round 6 found:
# `ensure_outside_repo` (an IMPORTED helper that mkdirs what it validates) called before
# admission, and an output path confined against a module-level constant while the run's
# provenance is attested against the checkout the launcher was HANDED.
_VIOLATING_LAUNCHER_SOURCE = '''
import pathlib
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import ensure_outside_repo

REPO = pathlib.Path(__file__).resolve().parents[3]


def main():
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).expanduser()
    out = ensure_outside_repo(pathlib.Path(args.out_dir), REPO)
    manifest = admit_benchmark_run(out / "run_manifest.json", run_root=out, repo_dir=repo_dir)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        return 0
'''

# The same launcher with both invariants honoured: the pure `assert_*` form (no mkdir) before
# admission, and the handed-in checkout as the confinement authority.
_CLEAN_LAUNCHER_SOURCE = _VIOLATING_LAUNCHER_SOURCE.replace(
    "import ensure_outside_repo", "import assert_outside_repo",
).replace(
    "out = ensure_outside_repo(pathlib.Path(args.out_dir), REPO)",
    "out = assert_outside_repo(pathlib.Path(args.out_dir), repo_dir)",
)


# INVARIANT C. A synthetic launcher that publishes its manifest from inside the seam, in the
# exact shape the real ones had: a helper named for the RECORDS it keeps, whose body happens to
# write the manifest too. The name says nothing; only the body does.
_SEAM_PUBLICATION_DEFECT_SOURCE = '''
import pathlib
from devtools.benchmarks.common.manifests import (
    admit_benchmark_run, finalize_run_manifest, write_json,
)
from devtools.benchmarks.common.run_roots import assert_outside_repo


def _write_records(run_dir, manifest, outcome):
    write_json(run_dir / "task_outcome.json", outcome)
    write_json(run_dir / "task_run_manifest.json", manifest)
    return outcome


def main():
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).expanduser()
    out = assert_outside_repo(pathlib.Path(args.out_dir), repo_dir)
    manifest = admit_benchmark_run(out / "run_manifest.json", run_root=out, repo_dir=repo_dir)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        final["outcome"] = "completed"
        return _write_records(out, manifest, {"ok": True})
'''

# The corrected twin: the records helper keeps its OUTCOME sidecar and stops publishing the
# manifest, which the seam writes on every exit path anyway.
_SEAM_PUBLICATION_FIXED_SOURCE = _SEAM_PUBLICATION_DEFECT_SOURCE.replace(
    '    write_json(run_dir / "task_run_manifest.json", manifest)\n', "")

# The same publication with the filename moved one line up into a local — the `run_pro` shape,
# which a check that only read the call site would wave through.
_SEAM_PUBLICATION_INDIRECT_SOURCE = _SEAM_PUBLICATION_DEFECT_SOURCE.replace(
    '    write_json(run_dir / "task_run_manifest.json", manifest)',
    '    manifest_path = run_dir / "task_run_manifest.json"\n'
    '    write_json(manifest_path, manifest)')


def test_the_launcher_gate_forbids_publishing_a_manifest_inside_the_seam():
    """INVARIANT C, pinned against a violator, its corrected twin and its indirect form.

    `finalize_run_manifest` merges the terminal outcome/exit_code/refusal into the manifest when
    its context EXITS. Anything written from inside publishes a PRE-MERGE record — for a refusal,
    the admission seam's generic payload saying exit_code 1 while the process will exit 2. Two
    review rounds fixed this in `run_cu_bridge_agent` and a by-hand sweep still missed
    `run_step_agent` and `run_pro`, because the sweep asked "is there a second copy that can go
    stale?" when the hazard is "is anything published before the merge?" — true of a single-path
    launcher too. Hence a gate.

    Judged by EFFECT: the helper is called `_write_records`, the real ones `_write_task_records`
    and `_write_cu_outcome`. No name-based check finds any of the three.
    """
    from devtools.benchmarks.common import launcher_audit

    # The offending helper is not named anywhere in the gate -- resolution is the rule.
    assert "_write_records" not in launcher_audit.WRITE_PRIMITIVES
    assert not (launcher_audit.WRITE_PRIMITIVES
                & {"_write_task_records", "_write_cu_outcome", "_write_records"})

    violations = launcher_audit.audit_source(_SEAM_PUBLICATION_DEFECT_SOURCE, name="seam.py")
    assert len(violations) == 1, violations
    assert "publishes a manifest from INSIDE an active finalize_run_manifest" in violations[0]
    assert "_write_records -> write_json" in violations[0]

    # ...the same defect with the filename bound to a local one line earlier is still caught...
    indirect = launcher_audit.audit_source(_SEAM_PUBLICATION_INDIRECT_SOURCE, name="seam.py")
    assert len(indirect) == 1 and "_write_records -> write_json" in indirect[0], indirect

    # ...and the corrected twin passes, so the invariant is not simply always-red.
    assert launcher_audit.audit_source(_SEAM_PUBLICATION_FIXED_SOURCE, name="seam.py") == []


def test_every_migrated_launcher_routes_through_both_manifest_seams():
    """Fix the CLASS, not the cases: the seams are pointless if a launcher can pair
    `benchmark_run_manifest()` with its own `write_json()` again (no durable refusal) or skip the
    finalization block (no final outcome). Named files, so a new launcher cannot join silently and
    the launchers whose migration belongs to a LATER phase cannot be silently claimed."""
    # v6.76.0 promoted these three helpers out of this test module and into the shared gate;
    # this test uses that SSOT rather than keeping a second, weaker copy of the same walk.
    from devtools.benchmarks.common.launcher_audit import (
        _dotted_callee, calls_before as _calls_before,
        denied_pre_admission_call as _denied_pre_admission_call,
    )

    bench = REPO_ROOT / "devtools" / "benchmarks"
    migrated = [
        bench / "programbench" / "run_programbench.py",
        bench / "programbench" / "run_programbench_e2e.py",
        bench / "swe_bench" / "swebench_predictions.py",
        bench / "swe_bench_pro" / "pro_predictions.py",
        bench / "harness_bench_fast" / "run_harness_bench_fast.py",
        bench / "swe_bench_pro" / "e1v2" / "run_pro.py",
        bench / "swe_bench_pro" / "e1v2" / "auto_run.py",
        bench / "gaia" / "run_gaia.py",
        bench / "terminal_bench" / "run_tb.py",
        bench / "terminal_bench" / "run_harbor_smoke.py",
        bench / "continual_learning" / "run_clb.py",
        bench / "cybergym" / "run_cybergym.py",
        bench / "osworld" / "run_step_agent.py",
        bench / "osworld" / "run_cu_bridge_agent.py",
        bench / "osworld" / "osworld_adapter_skeleton.py",
        bench / "editbench" / "run_editbench.py",
    ]
    for path in migrated:
        source = path.read_text(encoding="utf-8")
        assert "admit_benchmark_run(" in source, f"{path.name} bypasses the admission seam"
        assert "finalize_run_manifest(" in source, f"{path.name} records no final outcome"
        assert "benchmark_run_manifest(" not in source, (
            f"{path.name} calls the builder directly again: its refusal would never be persisted"
        )
        # Python evaluates ARGUMENTS before entering the callee, so a gate called inside the
        # admission call's argument list refuses BEFORE the manifest can be written — the durable
        # refusal defeated by evaluation order. Attestation belongs after admission.
        call = source.split("admit_benchmark_run(", 1)[1].split("\n    )\n", 1)[0]
        assert "runtime_attestation(" not in call, (
            f"{path.name} evaluates runtime_attestation inside the admission argument list"
        )
        # ADMISSION IS THE OUTER BOUNDARY. Everything a launcher does before it must be argument
        # parsing and pure local derivation: no filesystem assertion, no docker, no subprocess, no
        # network, no state mutation. Walked with `ast` over the function that performs admission
        # AND, when that is not `main()`, over the statements of `main()` that precede it.
        tree = ast.parse(source)
        functions = {node.name: node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)}
        owner = next(
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(isinstance(inner, ast.Call)
                    and _dotted_callee(inner.func).endswith("admit_benchmark_run")
                    for inner in ast.walk(node))
        )
        prefix = _calls_before(functions[owner], "admit_benchmark_run")
        if owner != "main":
            prefix += _calls_before(functions["main"], owner)
        for dotted in prefix:
            denied = _denied_pre_admission_call(dotted)
            assert not denied, (
                f"{path.name}: {dotted}() runs BEFORE admit_benchmark_run() in {owner}() -- a "
                f"refusal there leaves no durable manifest (denied token: {denied})"
            )
    # The pending set is EMPTY on this tree: CL-Bench and the three OSWorld launchers migrated
    # in v6.76.0, GAIA and both Terminal-Bench launchers in v6.79.0. Asserted against the gate's
    # own list so the two enumerations cannot drift apart silently.
    from devtools.benchmarks.common import launcher_audit

    assert launcher_audit.PENDING_LAUNCHERS == ()
    assert sorted(path.relative_to(bench).as_posix() for path in migrated) == sorted(
        launcher_audit.MIGRATED_LAUNCHERS
    )


# One synthetic per CALL FORM a write primitive can wear. The destination model is derived from
# each primitive's real signature, so this matrix is what proves the derivation covers the forms
# rather than asserting it. The first two are the ones a reviewer found missing from the
# hand-written position table it replaced.
_SEAM_FORM_TEMPLATE = '''
import json
import os
import pathlib
import shutil
from devtools.benchmarks.common.manifests import (
    admit_benchmark_run, finalize_run_manifest, write_json, write_jsonl,
)
from devtools.benchmarks.common.run_roots import assert_outside_repo
from ouroboros.utils import atomic_write_json, write_text_atomic


def main():
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).expanduser()
    out = assert_outside_repo(pathlib.Path(args.out_dir), repo_dir)
    manifest = admit_benchmark_run(out / "run_manifest.json", run_root=out, repo_dir=repo_dir)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        final["outcome"] = "completed"
        {statement}
        return 0
'''

_SEAM_WRITE_FORMS = (
    # (label, statement, the callee the report must name)
    ("os.rename publishes to argument ONE",
     'os.rename(tmp, out / "run_manifest.json")', "os.rename"),
    ("standalone write_text takes the path positionally",
     'write_text(out / "run_manifest.json", body)', "write_text"),
    ("standalone write_bytes takes the path positionally",
     'write_bytes(out / "run_manifest.json", blob)', "write_bytes"),
    ("receiver-style write_text names its destination as the receiver",
     '(out / "run_manifest.json").write_text(body)', "write_text"),
    ("receiver-style rename publishes to its target argument",
     'tmp.rename(out / "run_manifest.json")', "rename"),
    ("os.replace publishes to argument ONE",
     'os.replace(tmp, out / "run_manifest.json")', "os.replace"),
    ("shutil.move publishes to argument ONE",
     'shutil.move(tmp, out / "run_manifest.json")', "shutil.move"),
    ("json.dump publishes to its fp argument",
     'json.dump(manifest, open(out / "run_manifest.json", "w"))', "json.dump"),
    ("the destination may arrive as a KEYWORD",
     'write_json(path=out / "run_manifest.json", payload=manifest)', "write_json"),
    ("write_jsonl", 'write_jsonl(out / "run_manifest.json", rows)', "write_jsonl"),
    ("atomic_write_json", 'atomic_write_json(out / "run_manifest.json", manifest)',
     "atomic_write_json"),
    ("write_text_atomic", 'write_text_atomic(out / "run_manifest.json", text)',
     "write_text_atomic"),
    # ...and the local hop, which is how `run_pro` spelled it.
    ("the destination bound to a local one line earlier",
     'manifest_path = out / "run_manifest.json"\n        write_json(manifest_path, manifest)',
     "write_json"),
)


@pytest.mark.parametrize("label, statement, callee", _SEAM_WRITE_FORMS,
                         ids=[form[2] + "/" + form[0][:28] for form in _SEAM_WRITE_FORMS])
def test_invariant_c_places_the_destination_of_every_write_form(label, statement, callee):
    """Every call form a write primitive wears is caught, and the coverage is PROVEN per form.

    The first cut of Invariant C carried a hand-enumerated position table, and it was wrong in
    exactly the way hand-enumerated tables are: `rename` was mapped to argument 0 although
    `os.rename(src, dst)` publishes to argument 1, and standalone `write_text(path, ...)` had no
    positional destination at all — so an in-seam `os.rename(tmp, .../run_manifest.json)` passed
    silently. A gate whose whole subject is incomplete models of where a write goes cannot carry
    one. Destinations now come from each primitive's REAL signature, and this matrix is the proof
    that the derivation covers the forms rather than an assertion that it does.
    """
    from devtools.benchmarks.common import launcher_audit

    source = _SEAM_FORM_TEMPLATE.format(statement=statement)
    violations = launcher_audit.audit_source(source, name="form.py")
    assert len(violations) == 1, (label, violations)
    assert "publishes a manifest from INSIDE an active finalize_run_manifest" in violations[0]
    assert callee in violations[0], (label, violations[0])
    assert launcher_audit.UNRESOLVED_WRITE not in violations[0]

    # The same form writing a NON-manifest artefact is not a publication -- per form, so the
    # matrix cannot pass by being uniformly red.
    benign = launcher_audit.audit_source(
        source.replace("run_manifest.json", "task_outcome.json").replace(
            'admit_benchmark_run(out / "task_outcome.json"',
            'admit_benchmark_run(out / "run_manifest.json"').replace(
            'finalize_run_manifest(out / "task_outcome.json"',
            'finalize_run_manifest(out / "run_manifest.json"'),
        name="form.py")
    assert benign == [], (label, benign)


def test_invariant_c_derives_destinations_from_real_signatures_not_a_hand_written_table():
    """The positions come from the callable, so they cannot drift out of step with it."""
    import os

    from devtools.benchmarks.common import launcher_audit

    # Each primitive resolves to at least one REAL signature...
    for leaf in launcher_audit.WRITE_PRIMITIVES:
        assert launcher_audit.primitive_signatures(leaf), leaf

    # ...and those signatures are the live ones, not a copy. `rename` is the case in point: two
    # different callables share the name, and the union of both is what closes the hole.
    assert ("src", "dst") in {positional for positional, _every
                              in launcher_audit.primitive_signatures("rename")}
    assert ("self", "target") in {positional for positional, _every
                                  in launcher_audit.primitive_signatures("rename")}
    assert tuple(inspect.signature(os.rename).parameters)[:2] == ("src", "dst")


def test_invariant_c_fails_closed_on_a_write_form_it_cannot_place(monkeypatch):
    """An unplaceable write is REPORTED, never assumed harmless.

    A write whose destination no signature can name is the state the hand-written table was
    silently in for every form it omitted. Failing closed converts that silence into a report:
    the gate says it cannot tell, instead of saying there is nothing there.
    """
    from devtools.benchmarks.common import launcher_audit

    source = _SEAM_FORM_TEMPLATE.format(statement='write_json(out / "run_manifest.json", manifest)')
    assert launcher_audit.audit_source(source, name="closed.py")      # placed: a plain violation

    # Strip the primitive's home so nothing can place it, exactly as an unmodelled form is.
    monkeypatch.setitem(launcher_audit._PRIMITIVE_HOMES, "write_json", ())
    launcher_audit.primitive_signatures.cache_clear()
    try:
        violations = launcher_audit.audit_source(source, name="closed.py")
    finally:
        # Drop the patched answer BEFORE monkeypatch restores the table, so no later test in this
        # process sees a cached "unplaceable" verdict for a primitive that is placeable again.
        launcher_audit.primitive_signatures.cache_clear()
    assert len(violations) == 1, violations
    assert launcher_audit.UNRESOLVED_WRITE in violations[0]
    assert "no real signature places its destination" in violations[0]


def test_the_launcher_gate_does_not_confuse_a_recorded_manifest_path_with_a_publication():
    """Recording a manifest PATH in a payload is not writing to it — the vacuity guard.

    CL-Bench's `collect_results` writes `results.json` whose payload lists pointers to the
    external runner's sidecar manifests (`.../cl_bench/*/run_manifest.json`). A first cut of
    Invariant C inspected every argument of the write and reported that as a publication. Only
    the DESTINATION counts; an always-red gate is as useless as a vacuously green one.
    """
    from devtools.benchmarks.common import launcher_audit

    pointer_payload = _SEAM_PUBLICATION_FIXED_SOURCE.replace(
        '    write_json(run_dir / "task_outcome.json", outcome)',
        '    write_json(run_dir / "results.json",\n'
        '               {"sidecars": sorted(str(p) for p in run_dir.glob("*/run_manifest.json"))})')
    assert launcher_audit.audit_source(pointer_payload, name="pointers.py") == []


def test_the_launcher_gate_catches_a_synthetic_violator_of_both_invariants():
    """The gate is pinned against a launcher that BREAKS it, not only against clean ones.

    Round 6 found `ensure_outside_repo` running before admission in four launchers, and the
    guard had missed it for six rounds because it is IMPORTED: the resolver followed only local
    definitions, so an imported mutator was invisible unless somebody had thought to name it in
    the denylist. A denylist is a list of yesterday's bugs. This asserts the RESOLUTION: the
    two `ensure_*` names are NOT in the denylist, and the violation is still reported — by
    reading, one module over, what the helper's body actually does.
    """
    from devtools.benchmarks.common import launcher_audit

    assert not (launcher_audit.PRE_ADMISSION_DENIED_NAMES
                & {"ensure_outside_repo", "ensure_file_output_outside_repo"})
    assert launcher_audit.denied_pre_admission_call("ensure_outside_repo") == ""

    violations = launcher_audit.audit_source(_VIOLATING_LAUNCHER_SOURCE, name="synthetic.py")
    # INVARIANT A, caught through the imported hop and reported as `helper -> what it does`.
    assert any("BEFORE admit_benchmark_run()" in v and "ensure_outside_repo -> mkdir" in v
               for v in violations), violations
    # INVARIANT B: the run is attested against `--repo-dir` but confined against `REPO`.
    assert any("confines paths ONLY against module scope" in v and "REPO" in v
               for v in violations), violations
    assert len(violations) == 2

    # ...and the corrected launcher passes, so the gate is not simply always-red.
    assert launcher_audit.audit_source(_CLEAN_LAUNCHER_SOURCE, name="synthetic.py") == []


def test_the_launcher_gate_reproduces_both_round_six_confinement_defects():
    """Invariant B, on the two real shapes: a helper that resolves its own authority, and a
    launcher that validates its out-dir against its own checkout instead of the executed one."""
    from devtools.benchmarks.common import launcher_audit

    # The `confined_claims_dir` shape: the authority came from `repo_root_from_devtools()`, so
    # `--repo-dir /other/clone --claim-dir /other/clone/.claims` wrote lock and marker state
    # into the execution checkout.
    claims_defect = '''
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import assert_outside_repo, repo_root_from_devtools


def confined_claims_dir(claims_dir):
    return assert_outside_repo(claims_dir, repo_root_from_devtools())


def main():
    args = parse_args()
    repo_dir = args.repo_dir
    claims = confined_claims_dir(args.claim_dir)
    manifest = admit_benchmark_run(args.out, repo_dir=repo_dir)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
'''
    violations = launcher_audit.audit_source(claims_defect, name="claims_defect.py")
    assert any("confined_claims_dir() confines paths ONLY against module scope" in v
               and "repo_root_from_devtools" in v for v in violations), violations

    # The `run_clb.main` shape: `--out-dir` validated against the launcher's own REPO, so
    # admission artefacts could land inside the execution clone being attested.
    clb_defect = '''
import pathlib
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import assert_outside_repo

REPO = pathlib.Path(__file__).resolve().parents[3]


def main():
    args = parse_args()
    execution_clone = pathlib.Path(args.ouroboros_clone)
    out = assert_outside_repo(pathlib.Path(args.out_dir), REPO)
    manifest = admit_benchmark_run(out / "run_manifest.json", repo_dir=execution_clone)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        return 0
'''
    violations = launcher_audit.audit_source(clb_defect, name="clb_defect.py")
    assert any("main() confines paths ONLY against module scope" in v and "REPO" in v
               for v in violations), violations
    # Confining against BOTH checkouts — which is what run_clb.py does now — is accepted: the
    # invariant is agreement with the attested checkout, not a ban on constants.
    fixed = clb_defect.replace(
        "    out = assert_outside_repo(pathlib.Path(args.out_dir), REPO)",
        "    out = pathlib.Path(args.out_dir)\n"
        "    for authority in (REPO, execution_clone):\n"
        "        out = assert_outside_repo(out, authority)",
    )
    assert launcher_audit.audit_source(fixed, name="clb_fixed.py") == []


def test_the_launcher_gate_leaves_static_launchers_alone():
    """A launcher that attests a STATICALLY derived root and confines against that same root is
    CONSISTENT, and flagging it would push the gate straight back toward per-case exemptions.
    The in-repo prediction writers (`swebench_predictions`, `pro_predictions`) are exactly this
    shape, and there is no other checkout for them to be wrong about."""
    from devtools.benchmarks.common import launcher_audit

    static_launcher = '''
import pathlib
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import assert_file_output_outside_repo

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def main():
    args = parse_args()
    output = assert_file_output_outside_repo(pathlib.Path(args.output), REPO_ROOT)
    manifest = admit_benchmark_run(args.manifest_output, repo_dir=REPO_ROOT)
    with finalize_run_manifest(args.manifest_output, manifest) as final:
        return 0
'''
    assert launcher_audit.audit_source(static_launcher, name="static.py") == []


def test_pre_admission_resolver_sees_through_helpers_and_past_step_aside_branches():
    """Pin the RESOLVER, not just its current verdict.

    Two rounds in a row, pre-admission work slipped past it by living one level down inside a
    local helper the denylist does not name (`_ensure_vmrun_on_path` probing the filesystem,
    `_install_optional_dependency_stubs` mutating `sys.modules`, `repo_provenance` shelling out
    to git, `_read_task_ids` running `uv run ... list` with a 60s timeout). So the guard is
    maintained by what a helper DOES. The complement matters too: a branch that always leaves
    the function is not on the path to admission — those are the deliberate step-aside paths
    that exist to leave no footprint — and flagging them would push the guard back toward the
    per-case exemptions it is supposed to replace.
    """
    from devtools.benchmarks.common import launcher_audit

    unit = launcher_audit._Unit(ast.parse(_GUARD_PROBE_SOURCE), "probe.py")
    prefix = launcher_audit.calls_before(unit.functions["main"], "admit_benchmark_run")

    # The helper that hides a subprocess IS caught, and the report names the helper.
    assert launcher_audit.resolve_denied("_looks_innocent", unit) == "_looks_innocent -> subprocess"
    # ...which is exactly what walking main()'s pre-admission statements now reports.
    denied = [d for d in (launcher_audit.resolve_denied(c, unit) for c in prefix) if d]
    assert denied == ["_looks_innocent -> subprocess"]
    # A pure helper is not flagged.
    assert launcher_audit.resolve_denied("_pure", unit) == ""
    # The step-aside branch (`if args.collect_only: ...; return 0`) never reaches admission, so
    # its mutating helper is not on the guarded path -- though the helper itself is still
    # recognised as mutating, so the exclusion is about the PATH, not about the denylist.
    assert "_steps_aside" not in prefix
    assert launcher_audit.resolve_denied("_steps_aside", unit) == "_steps_aside -> mkdir"
    # The branch TEST runs on the way past, so it is still walked.
    assert "parse_args" in prefix
    # TWO hops are resolved, and a hop now CROSSES MODULES — both are the round-6 fix. The old
    # guard resolved ONE hop of LOCAL definitions only, which is why an imported helper whose
    # own body called another imported helper was invisible twice over. A three-hop chain is
    # still out of the gate's reach and stays a review question; asserted so the real depth is
    # documented rather than implied.
    assert launcher_audit.resolve_denied("_two_levels_down", unit) == \
        "_two_levels_down -> _looks_innocent -> subprocess"
    assert launcher_audit.resolve_denied("_three_levels_down", unit) == ""


def test_swe_pro_manifest_records_the_derived_model_not_the_template(tmp_path, monkeypatch):
    """The manifest must name the model that RAN.

    A live SWE-Pro smoke found `run_manifest.json` reporting `anthropic/claude-sonnet-4.5`
    while `_run_settings.json`, the container environment and the in-container settings all
    agreed the run was on `openai/gpt-5.5`. `model_slot_snapshot` had been handed `--settings`
    — the TEMPLATE — while `derive_run_settings` applies `pin_single_model(--solve-model)` on
    top of it. Nothing in the artefact contradicted an auditor who believed the wrong name,
    which is precisely the failure this release exists to remove.

    Note what a weaker test would have done here: `model_slots["OUROBOROS_MODEL"]` is non-empty
    in the BUGGY case too. So this pins it to the DERIVED file and asserts the two disagree.
    """
    import importlib

    from devtools.benchmarks.common.manifests import model_slot_snapshot

    run_pro = importlib.import_module("devtools.benchmarks.swe_bench_pro.e1v2.run_pro")
    template = tmp_path / "settings_template.json"
    template.write_text(json.dumps({
        "OUROBOROS_MODEL": "anthropic/claude-sonnet-4.5",
        "OUROBOROS_MODEL_HEAVY": "anthropic/claude-sonnet-4.5",
        "TOTAL_BUDGET": 50.0,
    }), encoding="utf-8")
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    derived = run_pro.derive_run_settings(str(template), out_dir, "openai/gpt-5.5", 50.0, 5.0)
    assert derived == out_dir / "_run_settings.json"

    # The container is handed the FILE and a fresh environment, so the launcher's own env is
    # not part of that server's configuration and must not be reported as if it were.
    monkeypatch.setenv("OUROBOROS_MODEL", "some/host-env-model")
    assert model_slot_snapshot(derived, env_overrides=False)["OUROBOROS_MODEL"] == "openai/gpt-5.5"
    # ...and the template, which is what the manifest used to record, names a DIFFERENT model:
    # the exact disagreement the smoke observed.
    assert model_slot_snapshot(template, env_overrides=False)["OUROBOROS_MODEL"] == \
        "anthropic/claude-sonnet-4.5"

    # The call site takes the value `derive_run_settings` RETURNED, not `args.settings`.
    tree = ast.parse((REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2"
                      / "run_pro.py").read_text(encoding="utf-8"))
    snapshots = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", "") == "model_slot_snapshot"]
    assert len(snapshots) == 1
    assert getattr(snapshots[0].args[0], "id", "") == "seed"
    assert [(kw.arg, kw.value.value) for kw in snapshots[0].keywords] == [("env_overrides", False)]


def test_the_gate_catches_pre_admission_reads_parses_probes_and_nested_admission_args():
    """Round 7: the gate documented a WIDER class than it enforced.

    It denied MUTATION, but the invariant it states is that nothing which can FAIL may precede
    the persisted manifest — and a run that dies parsing its dataset leaves no manifest at all,
    so it is invisible rather than merely footprint-free, which is strictly worse. Four migrated
    launchers were still doing exactly that (`_records`/`_rows` reading `--input`,
    `preflight_model_slots` reading settings, `read_csv_order`/`load_pro_rows` reading the task
    order and downloading the dataset), and a fifth shape hid in plain sight: a call nested in
    the admission call's own ARGUMENT LIST, which Python evaluates before entering the callee.

    The four shapes are pinned here as synthetic launchers, then the corrected launcher is
    asserted to PASS, so the widening cannot be satisfied by a gate that is always red.
    """
    from devtools.benchmarks.common import launcher_audit

    def audit(body, name):
        return launcher_audit.audit_source(
            "import pathlib\n"
            "from devtools.benchmarks.common.manifests import "
            "admit_benchmark_run, finalize_run_manifest\n"
            "from devtools.benchmarks.common.run_roots import assert_outside_repo\n"
            "\nREPO = pathlib.Path(__file__).resolve().parents[3]\n\n" + body,
            name=name,
        )

    # 1. A DATASET READ one hop down, the `_records`/`_rows` shape.
    read = audit('''
def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    args = parse_args()
    rows = _records(pathlib.Path(args.input))
    manifest = admit_benchmark_run(args.out, repo_dir=REPO, requested_task_ids=rows)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "read.py")
    assert any("_records() runs BEFORE" in v and "_records -> read_text" in v
               for v in read), read

    # 2. A PARSE that opens the file itself, the `read_csv_order` shape.
    parse = audit('''
def read_csv_order(path):
    with path.open(encoding="utf-8") as handle:
        return sorted(csv.DictReader(handle), key=lambda row: int(row["idx"]))


def main():
    args = parse_args()
    order = read_csv_order(pathlib.Path(args.csv))
    manifest = admit_benchmark_run(args.out, repo_dir=REPO, requested_task_ids=order)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "parse.py")
    assert any("read_csv_order -> open" in v for v in parse), parse

    # 3. A MODEL-SLOT PROBE that reads settings and refuses, the `preflight_model_slots` shape.
    #    Reported by the read; the refusal is what made it fatal.
    probe = audit('''
def preflight_model_slots(settings_path):
    settings = json.loads(pathlib.Path(settings_path).read_text(encoding="utf-8"))
    if not settings:
        raise SystemExit("model slot preflight failed")
    return settings


def main():
    args = parse_args()
    slots = preflight_model_slots(args.settings)
    manifest = admit_benchmark_run(args.out, repo_dir=REPO, harness=slots)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "probe.py")
    assert any("preflight_model_slots -> read_text" in v for v in probe), probe

    # 4. A CALL NESTED IN THE ADMISSION ARGUMENTS, the `_collect_attestations` shape. Evaluated
    #    before `admit_benchmark_run` is even entered, and previously invisible because the
    #    walk STOPPED at the statement holding the admission call.
    nested = audit('''
def _collect_attestations(paths):
    return [json.loads(pathlib.Path(raw).read_text(encoding="utf-8")) for raw in paths]


def main():
    args = parse_args()
    manifest = admit_benchmark_run(
        args.out, repo_dir=REPO,
        extra={"runtime_attestations": _collect_attestations(args.attestation)},
    )
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "nested.py")
    assert any("_collect_attestations() runs BEFORE" in v and "read_text" in v
               for v in nested), nested

    # 5. A DEFERRED NON-STDLIB IMPORT, the `load_pro_rows`/`_load_instances` shape. Not a call
    #    at all, so no callee-name rule could ever have seen it; its ImportError (or an offline
    #    hub) killed the process with nothing on disk.
    dataset = audit('''
def load_pro_rows(ids):
    from datasets import load_dataset
    return load_dataset("ScaleAI/SWE-bench_Pro", split="test")


def main():
    args = parse_args()
    rows = load_pro_rows(args.ids)
    manifest = admit_benchmark_run(args.out, repo_dir=REPO, requested_task_ids=rows)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "dataset.py")
    assert any("load_pro_rows -> deferred import datasets" in v for v in dataset), dataset

    # THE CORRECTED SHAPE PASSES. Declared selector at admission, resolved ids amended after —
    # the chicken-and-egg has one answer and this is it.
    fixed = audit('''
def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    args = parse_args()
    manifest = admit_benchmark_run(
        args.out, repo_dir=REPO, requested_task_ids=[], extra={"input": str(args.input)},
    )
    with finalize_run_manifest(args.out, manifest) as final:
        rows = _records(pathlib.Path(args.input))
        manifest["requested_task_ids"] = [row["instance_id"] for row in rows]
        manifest["requested_count"] = len(rows)
        return 0
''', "fixed.py")
    assert fixed == [], fixed


def test_the_gate_separates_argv_shaped_refusals_from_state_shaped_ones():
    """Where the widened invariant draws its line, pinned so it is not re-litigated.

    Argument parsing and pure path arithmetic MUST precede admission — they compute the
    manifest's own path — and their refusals are a deterministic function of argv. A bare
    existence probe is the one permitted middle: it reads no content, cannot fail on malformed
    input, and is what lets `scored_claim_state` answer "another lane already scored this" and
    step aside leaving zero footprint. The combination is what is denied: a helper that PROBES
    and can also REFUSE produces a refusal no argv can explain, which is exactly the class that
    needs a durable manifest.
    """
    from devtools.benchmarks.common import launcher_audit

    source = '''
import pathlib


def refuse_live_repo_clone(clone):
    resolved = pathlib.Path(clone).expanduser().resolve(strict=False)
    if resolved == LIVE:
        raise SystemExit("--ouroboros-clone must never be the live repo")
    return resolved


def scored_claim_state(claims_dir, key):
    if (claims_dir / f"{key}.scored").exists():
        return "already_scored"
    return ""


def check_clone(clone):
    if not (clone / "devtools").exists():
        raise SystemExit("not an Ouroboros checkout")
'''
    unit = launcher_audit._Unit(ast.parse(source), "line.py")
    # Pure-argv refusal: allowed before admission.
    assert launcher_audit.resolve_denied("refuse_live_repo_clone", unit) == ""
    # Probe that only RETURNS: allowed, and this is deliberate, not an oversight.
    assert launcher_audit.resolve_denied("scored_claim_state", unit) == ""
    # Probe + refusal: denied.
    assert launcher_audit.resolve_denied("check_clone", unit) == \
        "check_clone -> refuses on probed state"
    # The probe names are recognised, and none of them is denied on its own.
    assert "exists" in launcher_audit.STATE_PROBE_NAMES
    assert not (launcher_audit.STATE_PROBE_NAMES & launcher_audit.PRE_ADMISSION_DENIED_NAMES)
    # A stdlib deferred import is not a dependency on the state of the world.
    assert launcher_audit.resolve_denied("_is_default_desktop_server", launcher_audit._Unit(
        ast.parse('''
def _is_default_desktop_server(url):
    from urllib.parse import urlparse
    return urlparse(url).port == 8765
'''), "stdlib.py")) == ""


def test_the_gate_catches_a_refusal_authority_derived_from___file__():
    """Invariant B's second shape, found by a live CL-Bench smoke rather than by review.

    `run_clb.refuse_live_repo_clone` compared `--ouroboros-clone` against `REPO`, a
    `__file__`-derived module constant, so running a PINNED SEED's own launcher and handing it
    that same seed — the recipe METHODOLOGY prescribes — was refused, while the live repo the
    guard exists to protect went unmentioned. The two trees coincide only in the development
    workspace. Same class as the `confined_claims_dir` finding, different syntax (a comparison
    rather than a call), which is why the call-shaped detector missed it.
    """
    from devtools.benchmarks.common import launcher_audit

    defect = '''
import pathlib
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import assert_outside_repo

REPO = pathlib.Path(__file__).resolve().parents[3]


def refuse_live_repo_clone(clone):
    resolved = pathlib.Path(clone).expanduser().resolve(strict=False)
    if resolved == REPO.resolve(strict=False):
        raise SystemExit("--ouroboros-clone must be a dedicated CLONE, never the live repo")
    return resolved


def main():
    args = parse_args()
    execution_clone = refuse_live_repo_clone(pathlib.Path(args.ouroboros_clone))
    out = assert_outside_repo(pathlib.Path(args.out_dir), execution_clone)
    manifest = admit_benchmark_run(out / "run_manifest.json", repo_dir=execution_clone)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        return 0
'''
    violations = launcher_audit.audit_source(defect, name="refusal_defect.py")
    assert any("refuse_live_repo_clone() REFUSES against ['REPO']" in v
               and "__file__" in v for v in violations), violations

    # Refusing against the LIVE runtime instead — what run_clb.py does now — passes.
    fixed = defect.replace(
        "    if resolved == REPO.resolve(strict=False):\n"
        '        raise SystemExit("--ouroboros-clone must be a dedicated CLONE, never the live repo")',
        "    for live in live_repo_roots():\n"
        "        if resolved == live.expanduser().resolve(strict=False):\n"
        '            raise SystemExit("--ouroboros-clone must never be the LIVE repo")',
    )
    assert launcher_audit.audit_source(fixed, name="refusal_fixed.py") == []


def test_the_gate_resolves_imported_first_party_helpers_only():
    """The resolver opens FIRST-PARTY modules only. Stdlib and third-party callees stay
    unresolved (the gate must not depend on what happens to be installed) and are covered by
    the name/prefix denylist instead."""
    from devtools.benchmarks.common import launcher_audit

    source = '''
from devtools.benchmarks.common.run_roots import (
    assert_outside_repo, ensure_file_output_outside_repo, ensure_outside_repo,
)
from json import dumps
import shutil


def _wrapper(path, repo):
    from devtools.benchmarks.common.manifests import write_json
    return write_json(path, {})
'''
    unit = launcher_audit._Unit(ast.parse(source), "imports.py")
    assert unit.imports["ensure_outside_repo"] == "devtools.benchmarks.common.run_roots"
    # A first-party import is opened and its body read: BOTH `ensure_*` helpers are caught by
    # what they do, one and two modules-hops away, with neither of them in the denylist.
    assert launcher_audit.resolve_denied("ensure_outside_repo", unit) == \
        "ensure_outside_repo -> mkdir"
    assert launcher_audit.resolve_denied("ensure_file_output_outside_repo", unit) == \
        "ensure_file_output_outside_repo -> ensure_outside_repo -> mkdir"
    # The pure `assert_*` form is what a pre-admission caller must use, and it is NOT flagged.
    assert launcher_audit.resolve_denied("assert_outside_repo", unit) == ""
    # A FUNCTION-LEVEL import is in the map too — the OSWorld launchers import their shared
    # claim helpers inside the functions that use them, and an import the resolver cannot see
    # is an imported mutator it cannot follow.
    assert unit.imports["write_json"] == "devtools.benchmarks.common.manifests"
    # A stdlib import is not opened; nothing is claimed about it.
    assert launcher_audit.resolve_denied("dumps", unit) == ""
    # ...but the name/prefix denylist still covers third-party mutators without resolving them:
    # the name hit wins when there is one, and the prefix catches whole families.
    assert launcher_audit.denied_pre_admission_call("shutil.rmtree") == "rmtree"
    assert launcher_audit.denied_pre_admission_call("shutil.copytree") == "shutil"
    assert launcher_audit.denied_pre_admission_call("docker_pull_if_missing") == \
        "docker_pull_if_missing"


def test_every_migrated_launcher_passes_the_structural_gate():
    """THE GATE. Every launcher under the admission contract, both invariants, one report.

    Fix the CLASS, not the cases. Six review rounds produced eighteen criticals whose per-round
    count went UP, because each round patched the call sites it happened to find. This answers
    the question for the whole family at once, and a launcher that joins the family later joins
    the gate with it. The seams themselves are pointless if a launcher can pair
    `benchmark_run_manifest()` with its own `write_json()` again (no durable refusal) or skip
    the finalization block (no final outcome), so those are checked here too.
    """
    from devtools.benchmarks.common import launcher_audit

    assert launcher_audit.audit_all_launchers() == []
    # Named files, so a new launcher cannot join silently and the launchers whose migration
    # belongs to a LATER phase cannot be silently claimed.
    for path in launcher_audit.launcher_paths():
        assert path.is_file(), path
    for rel in launcher_audit.PENDING_LAUNCHERS:
        source = (launcher_audit.BENCH_ROOT / rel).read_text(encoding="utf-8")
        assert "benchmark_run_manifest(" in source

def test_runtime_attestation_decides_commit_availability_before_skew(tmp_path, monkeypatch):
    """Reason ORDER is part of the fail-closed contract. A checkout with no readable commit that
    ALSO disagrees on the version was labelled `runtime_skew` — an OVERRIDABLE reason — so
    `OBO_ALLOW_EVOLVED_VOLUME=1` waived a run with no commit to attribute its numbers to."""
    from devtools.benchmarks.common import manifests

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b'{"runtime_version": "6.75.0"}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, "1")

    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    (bare / "VERSION").write_text("6.74.5\n", encoding="utf-8")   # skew AND no commit
    with pytest.raises(RuntimeError, match="reason=commit_unavailable") as refused:
        manifests.runtime_attestation("http://127.0.0.1:9/", bare)
    assert "does NOT waive" in str(refused.value)

    # With a real commit the same version disagreement IS the waivable skew.
    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "VERSION").write_text("6.74.5\n", encoding="utf-8")
    _git_commit_all(repo)
    skewed = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert skewed["reason"] == "runtime_skew" and skewed["overridden"] is True


def test_programbench_launcher_records_a_typed_outcome_on_its_failure_path(tmp_path, monkeypatch):
    """Failure path of the per-instance ProgramBench launcher: it only ever wrote
    `failure_reason_code`, so its manifest still claimed the run was `started` after it died."""
    from devtools.benchmarks.programbench import run_programbench as pb

    out_root = tmp_path / "pb"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    instruction = tmp_path / "task.txt"
    instruction.write_text("do it", encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pb, "run_root", lambda *_a, **_k: out_root)

    def _boom(container_name):
        raise RuntimeError("cleanroom container is not running")

    monkeypatch.setattr(pb, "preflight_cleanroom_container", _boom)
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench.py", "--workspace", str(workspace), "--instruction-file",
         str(instruction), "--container-name", "c", "--instance-id", "inst-a",
         "--settings-path", str(settings), "--allow-dirty-seed"],
    )
    with pytest.raises(RuntimeError, match="cleanroom container is not running"):
        pb.main()

    extra = json.loads((out_root / "inst-a" / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "blocked"
    assert extra["exit_code"] == 1
    assert extra["refusal"]["stage"] == "cleanroom_preflight_failed"
    assert extra["error"]["type"] == "RuntimeError"
    assert extra["failure_reason_code"] == "cleanroom_preflight_failed"


def test_programbench_e2e_records_a_typed_outcome_on_its_failure_paths(tmp_path, monkeypatch):
    """Failure paths of the biggest spender: a completed run whose instances failed gets a NAMED
    outcome (not just exit 1), and an instance that RAISES leaves `crashed`, never `started`."""
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    out_root = tmp_path / "pb-e2e"
    settings = tmp_path / "settings.json"
    target_settings = _write_programbench_actor_settings(e2e, settings)
    monkeypatch.setattr(e2e, "_load_instances",
                        lambda **_k: [{"instance_id": "inst-a", "image_name": "img-a"}])
    monkeypatch.setattr(e2e, "runtime_attestation", lambda url, repo: {"ok": True})
    monkeypatch.setattr(e2e, "ouroboros_api_request", lambda *_a, **_k: target_settings)
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: out_root)
    monkeypatch.setattr(e2e, "_process_instance", lambda instance, cfg: e2e.task_result_row(
        benchmark="programbench", instance_id="inst-a", status="failed",
        reason_code="task_not_completed"))
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench_e2e.py", "--allow-dirty-seed", "--settings-path", str(settings),
         "--ouroboros-url", "http://127.0.0.1:9"],
    )
    assert e2e.main() == 1
    extra = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "instances_failed" and extra["exit_code"] == 1

    def _boom(instance, cfg):
        raise RuntimeError("docker exec died")

    monkeypatch.setattr(e2e, "_process_instance", _boom)
    with pytest.raises(RuntimeError, match="docker exec died"):
        e2e.main()
    extra = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "crashed" and extra["error"]["type"] == "RuntimeError"


def test_swebench_predictions_records_a_typed_outcome_when_it_stops_on_an_error(tmp_path, monkeypatch):
    """Failure path of the SWE-bench predictions launcher: it re-raises the first instance error,
    which used to escape with the manifest's `outcome` never written at all."""
    from devtools.benchmarks.swe_bench import swebench_predictions as sp

    input_path = tmp_path / "instances.jsonl"
    input_path.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    output = tmp_path / "preds.jsonl"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sp, "_run_prediction_rows",
                        lambda args, rows, **_k: ([], [], [], RuntimeError("agent never started")))
    monkeypatch.setattr(
        sys, "argv",
        ["swebench_predictions.py", "--input", str(input_path), "--output", str(output),
         "--settings-path", str(settings), "--allow-dirty-seed"],
    )
    with pytest.raises(RuntimeError, match="agent never started"):
        sp.main()

    extra = json.loads(Path(str(output) + ".run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "stopped_instance_error"
    assert extra["exit_code"] == 1
    assert extra["error"]["type"] == "RuntimeError"
    assert extra["prediction_count"] == 0


def test_pro_predictions_records_a_typed_outcome_when_it_stops_on_an_error(tmp_path, monkeypatch):
    """Failure path of the SWE-Pro prediction packer, driven by a REAL malformed input row."""
    from devtools.benchmarks.swe_bench_pro import pro_predictions as pp

    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    output = tmp_path / "preds.jsonl"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["pro_predictions.py", "--input", str(input_path), "--output", str(output),
         "--patch-dir", str(tmp_path / "patches"), "--settings-path", str(settings),
         "--allow-dirty-seed"],
    )
    with pytest.raises(RuntimeError, match="each row must include"):
        pp.main()

    extra = json.loads(Path(str(output) + ".run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "stopped_instance_error"
    assert extra["exit_code"] == 1
    assert extra["error"]["type"] == "RuntimeError"


def test_harness_bench_fast_records_a_crash_instead_of_leaving_started(tmp_path, monkeypatch):
    """The exceptional path of `harness_bench_fast`: its `_finish` helper covered every INTENDED
    exit, so an unhandled failure (missing harness runner) left `outcome: started` forever."""
    from devtools.benchmarks.harness_bench_fast import run_harness_bench_fast as hbf

    out_root = tmp_path / "hbf-run"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hbf, "_read_task_ids", lambda root, ids, task_file="": ["task_1"])

    def _boom(cmd, **kwargs):
        raise FileNotFoundError("harness runner is not installed")

    monkeypatch.setattr(hbf.subprocess, "run", _boom)
    monkeypatch.setattr(
        sys, "argv",
        ["run_harness_bench_fast.py", "--run-root", str(out_root), "--allow-dirty-seed",
         "--settings-path", str(settings), "--bench-root", str(tmp_path / "bench")],
    )
    with pytest.raises(FileNotFoundError):
        hbf.main()

    extra = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "crashed"
    assert extra["exit_code"] == 1
    assert extra["error"]["type"] == "FileNotFoundError"


def test_programbench_e2e_persists_the_manifest_when_attestation_refuses(tmp_path, monkeypatch, capsys):
    """A runtime-attestation refusal must leave the seed-admission manifest ON DISK.

    Attestation used to be evaluated inside `admit_benchmark_run(...)`'s argument list, and Python
    evaluates arguments before entering the callee — so `runtime_unreachable` /
    `commit_unavailable` / `runtime_skew` raised with no `run_manifest.json` written at all,
    defeating the durable-refusal contract by evaluation order alone.
    """
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    out_root = tmp_path / "pb-attest"
    settings = tmp_path / "settings.json"
    target_settings = _write_programbench_actor_settings(e2e, settings)
    monkeypatch.setattr(e2e, "_load_instances",
                        lambda **_k: [{"instance_id": "inst-a", "image_name": "img-a"}])
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: out_root)
    monkeypatch.setattr(e2e, "ouroboros_api_request", lambda *_a, **_k: target_settings)

    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused

    record = {"schema": "ouroboros.benchmark.runtime_attestation.v1",
              "reason": "runtime_unreachable", "ok": False, "runtime_version": "",
              "repo_head": "a" * 40, "repo_version": "6.75.0", "override_set": False,
              "http_error": "OSError: connection refused"}

    def _refuse(url, repo):
        raise RuntimeAttestationRefused(
            "runtime attestation failed reason=runtime_unreachable", record)

    monkeypatch.setattr(e2e, "runtime_attestation", _refuse)
    # An instance stand-in that must NEVER be reached: the refusal precedes all spend.
    monkeypatch.setattr(e2e, "_process_instance",
                        lambda instance, cfg: pytest.fail("an instance ran after the refusal"))
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench_e2e.py", "--allow-dirty-seed", "--settings-path", str(settings),
         "--ouroboros-url", "http://127.0.0.1:9"],
    )
    # RETURNS the recorded code. It used to re-raise, which exits the process with status 1 while
    # the manifest said 3 — the record and reality disagreeing (see
    # test_migrated_launcher_exit_status_matches_the_recorded_exit_code).
    assert e2e.main() == 3
    assert "reason=runtime_unreachable" in capsys.readouterr().err

    manifest = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))
    # The seed gate's SHAPE is on disk (never its verdict: `ok` mirrors the ambient checkout).
    assert set(manifest["seed_gate"]) >= {"ok", "reason", "require_clean", "allow_dirty_seed"}
    assert manifest["seed_gate"]["require_clean"] is False
    assert manifest["seed_gate"]["ok"] is (not manifest["seed_gate"]["reason"])
    extra = manifest["extra"]
    assert extra["outcome"] == "refused"
    assert extra["exit_code"] == 3
    # The EXACT typed reason, not a generic message: the helper builds the record and the launcher
    # persists it, so the manifest keeps the facts the provenance contract exists to preserve.
    assert extra["refusal"] == {"stage": "runtime_attestation", "exit_code": 3,
                                "reason": "runtime_unreachable"}
    assert extra["runtime_attestation"]["reason"] == "runtime_unreachable"
    assert extra["runtime_attestation"]["runtime_version"] == ""
    assert extra["runtime_attestation"]["repo_head"] == "a" * 40
    assert extra["runtime_attestation"]["repo_version"] == "6.75.0"
    # No `error` key: nothing escaped, because the refusal is RETURNED. The record is the report.
    assert "error" not in extra

    # A refusal that carries NO record still refuses and still records a durable manifest, with the
    # generic reason as the documented fallback.
    def _bare(url, repo):
        raise RuntimeError("attestation blew up with no record")

    monkeypatch.setattr(e2e, "runtime_attestation", _bare)
    assert e2e.main() == 3
    assert "no record" in capsys.readouterr().err
    extra = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["refusal"]["reason"] == "runtime_attestation_failed"
    assert extra["runtime_attestation"] == {"pending": "not_attested_yet"}


# --------------------------------------------------------------------------------------
# The recorded exit status must BE the process's exit status. Three review rounds found a
# fresh instance of "recorded != real" (a SystemExit flattened to 1, a re-raise after
# recording 2, a re-raise after recording 3), so the invariant is asserted behaviourally,
# once per migrated launcher, by driving main() into a refusal path.
# --------------------------------------------------------------------------------------


def _process_status_of(main) -> int:
    """The status a process would exit with, exactly as ``raise SystemExit(main())`` computes it."""
    try:
        return int(main() or 0)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException:
        return 1                      # any other escaping exception: CPython exits 1


def _refusal_case_programbench(tmp_path, monkeypatch):
    from devtools.benchmarks.programbench import run_programbench as pb

    out_root = tmp_path / "pb"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    instruction = tmp_path / "task.txt"
    instruction.write_text("do it", encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    def _boom(container_name):
        raise RuntimeError("cleanroom container is not running")

    monkeypatch.setattr(pb, "run_root", lambda *_a, **_k: out_root)
    monkeypatch.setattr(pb, "preflight_cleanroom_container", _boom)
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench.py", "--workspace", str(workspace), "--instruction-file",
         str(instruction), "--container-name", "c", "--instance-id", "inst-a",
         "--settings-path", str(settings), "--allow-dirty-seed"],
    )
    return pb.main, out_root / "inst-a" / "run_manifest.json"


def _refusal_case_programbench_e2e(tmp_path, monkeypatch):
    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    out_root = tmp_path / "pb-e2e"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    def _refuse(url, repo):
        raise RuntimeAttestationRefused("runtime attestation failed reason=runtime_unreachable",
                                        {"reason": "runtime_unreachable", "ok": False})

    monkeypatch.setattr(e2e, "_load_instances",
                        lambda **_k: [{"instance_id": "inst-a", "image_name": "img-a"}])
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: out_root)
    monkeypatch.setattr(e2e, "runtime_attestation", _refuse)
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench_e2e.py", "--allow-dirty-seed", "--settings-path", str(settings),
         "--ouroboros-url", "http://127.0.0.1:9"],
    )
    return e2e.main, out_root / "run_manifest.json"


def _refusal_case_swebench_predictions(tmp_path, monkeypatch):
    from devtools.benchmarks.swe_bench import swebench_predictions as sp

    input_path = tmp_path / "instances.jsonl"
    input_path.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    output = tmp_path / "preds.jsonl"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sp, "_run_prediction_rows",
                        lambda args, rows, **_k: ([], [], [], RuntimeError("agent never started")))
    monkeypatch.setattr(
        sys, "argv",
        ["swebench_predictions.py", "--input", str(input_path), "--output", str(output),
         "--settings-path", str(settings), "--allow-dirty-seed"],
    )
    return sp.main, Path(str(output) + ".run_manifest.json")


def _refusal_case_pro_predictions(tmp_path, monkeypatch):
    from devtools.benchmarks.swe_bench_pro import pro_predictions as pp

    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    output = tmp_path / "preds.jsonl"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["pro_predictions.py", "--input", str(input_path), "--output", str(output),
         "--patch-dir", str(tmp_path / "patches"), "--settings-path", str(settings),
         "--allow-dirty-seed"],
    )
    return pp.main, Path(str(output) + ".run_manifest.json")


def _refusal_case_harness_bench_fast(tmp_path, monkeypatch):
    from devtools.benchmarks.harness_bench_fast import run_harness_bench_fast as hbf

    out_root = tmp_path / "hbf-run"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hbf, "_read_task_ids", lambda root, ids, task_file="": ["task_1"])
    monkeypatch.setattr(hbf.subprocess, "run",
                        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 7, stdout="", stderr=""))
    monkeypatch.setattr(
        sys, "argv",
        ["run_harness_bench_fast.py", "--run-root", str(out_root), "--allow-dirty-seed",
         "--settings-path", str(settings), "--bench-root", str(tmp_path / "bench")],
    )
    return hbf.main, out_root / "run_manifest.json"


def _refusal_case_run_pro(tmp_path, monkeypatch):
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    out_dir = tmp_path / "out"
    seed = tmp_path / "worktree-seed"
    seed.mkdir()
    (seed / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    monkeypatch.setattr(run_pro, "SRC", seed)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(run_pro, "read_full_order", lambda: ["inst__a"])
    monkeypatch.setattr(run_pro, "load_pro_rows", lambda ids: {})
    monkeypatch.setattr(sys, "argv", ["run_pro.py", "--full-set", "--out-dir", str(out_dir),
                                      "--allow-dirty-seed"])
    return run_pro.main, out_dir / "run_manifest.json"


def _refusal_case_auto_run(tmp_path, monkeypatch):
    from devtools.benchmarks.common.manifests import SeedShapeRefused
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    out_dir = tmp_path / "auto"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    fake_run_pro = SimpleNamespace()

    def _refuse(path):
        raise SeedShapeRefused("seed_is_not_a_git_directory", "no real .git directory")

    fake_run_pro.assert_seed_is_git_directory = _refuse
    fake_run_pro.ensure_util_image = lambda: None
    monkeypatch.setitem(sys.modules, "devtools.benchmarks.swe_bench_pro.e1v2.run_pro", fake_run_pro)
    monkeypatch.setattr(sys, "argv", ["auto_run.py", "--start", "1", "--end", "1",
                                      "--out-dir", str(out_dir), "--allow-dirty-seed"])
    return auto_run.main, out_dir / "auto_run_manifest.json"


def _refusal_case_run_clb(tmp_path, monkeypatch):
    """CL-Bench refuses on the EXECUTION clone's provenance. The clone here is a bare
    non-git directory, so the verdict is a property of the fixture, never of the ambient
    checkout (and no `--allow-dirty-seed`, because the refusal IS what is under test)."""
    from devtools.benchmarks.continual_learning import run_clb

    clone = tmp_path / "execution-clone"
    (clone / "devtools" / "benchmarks" / "common").mkdir(parents=True)
    out = tmp_path / "clb-run"
    monkeypatch.setattr(
        sys, "argv",
        ["run_clb.py", "--ouroboros-clone", str(clone), "--out-dir", str(out), "--dry-run"],
    )
    return run_clb.main, out / "run_manifest.json"


def _refusal_case_run_step_agent(tmp_path, monkeypatch):
    from devtools.benchmarks.osworld import run_step_agent

    repo_dir = tmp_path / "repo"                 # bare dir: no git identity, ambient-free
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    task = tmp_path / "OSWorld" / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv",
        ["run_step_agent.py", "--osworld-root", str(tmp_path / "OSWorld"), "--task", str(task),
         "--result_dir", str(results), "--repo-dir", str(repo_dir),
         "--data-dir", str(tmp_path / "data"), "--settings-path", str(tmp_path / "settings.json"),
         "--ouroboros-url", "http://127.0.0.1:9", "--provider_name", "docker",
         "--model", "m"],
    )
    manifest = (results / "pyautogui" / "screenshot_a11y_tree" / "m" / "chrome" / "abc"
                / "task_run_manifest.json")
    return run_step_agent.main, manifest


def _refusal_case_run_cu_bridge_agent(tmp_path, monkeypatch):
    """Admitted, then refused by the runtime attestation (nothing listens on the URL), so the
    finalization seam — not the admission payload — has to record the real status."""
    from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb

    osworld = tmp_path / "OSWorld"
    (osworld / "evaluation_examples" / "examples" / "chrome").mkdir(parents=True)
    task = osworld / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv",
        ["run_cu_bridge_agent.py", "--osworld-root", str(osworld), "--provider_name", "docker",
         "--path_to_vm", "/vm/Ubuntu.qcow2", "--task", str(task), "--result_dir", str(results),
         "--repo-dir", str(repo_dir), "--data-dir", str(tmp_path / "data"),
         "--settings-path", str(tmp_path / "settings.json"),
         "--ouroboros-url", "http://127.0.0.1:9",
         "--target-file", str(tmp_path / "target.txt"), "--allow-dirty-seed"],
    )
    return rcb.main, results / "chrome" / "abc" / "task_run_manifest.json"


def _refusal_case_run_cu_bridge_agent_seed_gate(tmp_path, monkeypatch):
    """The SEED-GATE refusal, with NO `--claim-dir`, so this attempt OWNS the task.

    Owning it means the launcher keeps two copies of the record: the append-only
    `attempts/<id>/task_run_manifest.json` and the shared canonical
    `run_dir/task_run_manifest.json` a scorer reads — and the case above cannot reach this
    branch at all, because it passes `--allow-dirty-seed`. The seed is a REAL git repo left
    deliberately dirty, so the refusal is a property of the fixture and never of whatever
    checkout (or sandbox layout) the test itself happens to run inside.
    """
    from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb

    osworld = tmp_path / "OSWorld"
    (osworld / "evaluation_examples" / "examples" / "chrome").mkdir(parents=True)
    task = osworld / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    repo_dir = tmp_path / "repo"
    _git_repo(repo_dir)
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")   # uncommitted => seed_dirty
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv",
        ["run_cu_bridge_agent.py", "--osworld-root", str(osworld), "--provider_name", "docker",
         "--path_to_vm", "/vm/Ubuntu.qcow2", "--task", str(task), "--result_dir", str(results),
         "--repo-dir", str(repo_dir), "--data-dir", str(tmp_path / "data"),
         "--settings-path", str(tmp_path / "settings.json"),
         "--ouroboros-url", "http://127.0.0.1:9",
         "--target-file", str(tmp_path / "target.txt")],
    )
    return rcb.main, results / "chrome" / "abc" / "task_run_manifest.json"


def _refusal_case_osworld_adapter_skeleton(tmp_path, monkeypatch):
    from devtools.benchmarks.osworld import osworld_adapter_skeleton as skeleton

    repo_root = tmp_path / "repo"                # bare dir: no git identity, ambient-free
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    output_root = tmp_path / "runs" / "osworld"
    for path in (repo_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    monkeypatch.setattr(skeleton, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(skeleton, "DEFAULT_DATA_ROOT", tmp_path / "live-data")
    monkeypatch.setattr(
        sys, "argv",
        ["osworld_adapter_skeleton.py", "--osworld-root", str(osworld),
         "--ouroboros-url", "http://127.0.0.1:9", "--osworld-server-url", "http://127.0.0.1:9",
         "--unix-computer-use-payload", str(payload), "--output-root", str(output_root)],
    )
    return skeleton.main, output_root / "osworld_preflight.run_manifest.json"


_REFUSAL_CASES = (
    _refusal_case_programbench,
    _refusal_case_programbench_e2e,
    _refusal_case_swebench_predictions,
    _refusal_case_pro_predictions,
    _refusal_case_harness_bench_fast,
    _refusal_case_run_pro,
    _refusal_case_auto_run,
    _refusal_case_run_clb,
    _refusal_case_run_step_agent,
    _refusal_case_run_cu_bridge_agent,
    _refusal_case_run_cu_bridge_agent_seed_gate,
    _refusal_case_osworld_adapter_skeleton,
)


@pytest.mark.parametrize(
    "build_case", _REFUSAL_CASES,
    ids=[case.__name__[len("_refusal_case_"):] for case in _REFUSAL_CASES],
)
def test_migrated_launcher_exit_status_matches_the_recorded_exit_code(build_case, tmp_path, monkeypatch):
    """The manifest's `exit_code` must BE the status the process exits with, per launcher.

    Asserted as a PROPERTY rather than as syntax: each case drives the launcher into a failing
    path and compares the status `raise SystemExit(main())` would produce against the
    `extra.exit_code` the run's own record claims. Recording a code and then letting a plain
    exception escape silently reports 1 instead — which is how three separate review rounds each
    found a fresh instance of the record disagreeing with reality.
    """
    main, manifest_path = build_case(tmp_path, monkeypatch)
    status = _process_status_of(main)
    extra = json.loads(manifest_path.read_text(encoding="utf-8"))["extra"]
    assert status == extra["exit_code"], (
        f"process would exit {status} but the manifest records exit_code={extra['exit_code']} "
        f"(outcome={extra.get('outcome')!r})"
    )
    assert status != 0                      # every case here is a failure path
    assert extra["outcome"] not in ("started", "completed")


def test_cu_bridge_refusal_mirrors_the_terminal_record_to_the_canonical_manifest(
    tmp_path, monkeypatch, capsys
):
    """The SHARED canonical manifest must carry the SAME terminal record as the attempt's own.

    `run_cu_bridge_agent` is the one launcher whose record lives in two places: the attempt's
    append-only copy, which the finalization seam writes, and the canonical
    `run_dir/task_run_manifest.json`, which a separate mirror writes for whichever attempt owns
    the task. The mirror is only correct AFTER the seam's context manager has exited, because
    that exit is what merges the terminal `outcome`/`exit_code`/`refusal` into the manifest.
    The seed-gate branch used to mirror from INSIDE its seam and then `return` past the outer
    `finally`, so the artefact a scorer reads kept the admission seam's GENERIC refusal —
    `exit_code` 1 and no terminal outcome — while the process really exited 2. That is the
    "recorded status != real status" defect this release exists to eliminate, inside the
    machinery built to forbid it, on a path any operator hits with a dirty seed and no
    `--claim-dir`.
    """
    main, canonical = _refusal_case_run_cu_bridge_agent_seed_gate(tmp_path, monkeypatch)
    status = _process_status_of(main)
    capsys.readouterr()
    assert status == 2

    recorded = json.loads(canonical.read_text(encoding="utf-8"))
    extra = recorded["extra"]
    assert recorded["seed_gate"]["reason"] == "seed_dirty"      # the fixture's own verdict
    assert extra["outcome"] == "refused"
    assert extra["exit_code"] == status
    assert extra["refusal"] == {"stage": "seed_gate", "reason": "seed_dirty", "exit_code": status}
    assert extra["allow_dirty_seed"] is False
    # The canonical OUTCOME sidecar names the same refusal in this launcher's own vocabulary.
    outcome = json.loads((canonical.parent / "task_outcome.json").read_text(encoding="utf-8"))
    assert (outcome["status"], outcome["reason_code"]) == ("blocked", "seed_gate_failed")

    # ...and it is byte-for-byte the attempt's OWN record, not merely a plausible one: the two
    # copies of a single run's provenance may never tell different stories about how it ended.
    attempts = sorted((canonical.parent / "attempts").iterdir())
    assert len(attempts) == 1
    assert json.loads((attempts[0] / "task_run_manifest.json").read_text(encoding="utf-8")) == recorded


def test_step_agent_refusal_writes_its_manifest_only_on_admission_and_on_seam_exit(
    tmp_path, monkeypatch, capsys
):
    """Invariant C, behaviourally, on the launcher whose canonical path IS the seam's own.

    `run_step_agent` keeps ONE manifest, so round nine's "is there a second copy that can go
    stale?" sweep cleared it — wrongly, because the hazard is publishing before the merge, which
    a single-path launcher does just as readily. `_write_task_records` wrote the manifest from
    inside the seam, so a reader could observe `exit_code` 1 on a run that exits 2, and an
    interruption in that window left it durable.

    The property is the WRITE SEQUENCE at that path: the deliberate admission record, then the
    seam's terminal write on exit, and nothing in between. Asserting only the final content
    passes just as happily with an extra pre-merge publication.
    """
    from devtools.benchmarks.common import manifests
    from devtools.benchmarks.osworld import run_step_agent

    main, manifest_path = _refusal_case_run_step_agent(tmp_path, monkeypatch)
    target = manifest_path.resolve(strict=False)
    writes: list[dict] = []
    real_write_json = manifests.write_json

    def _recording_write_json(path, payload):
        if Path(path).resolve(strict=False) == target:
            writes.append(json.loads(json.dumps(payload)))      # snapshot exactly AS WRITTEN
        return real_write_json(path, payload)

    # Both bindings: the seam writes through `manifests`, the launcher through its own import,
    # so watching one name only would miss half the writes to the very path under test.
    monkeypatch.setattr(manifests, "write_json", _recording_write_json)
    monkeypatch.setattr(run_step_agent, "write_json", _recording_write_json)
    assert _process_status_of(main) == 2
    capsys.readouterr()

    states = [((w.get("extra") or {}).get("outcome"), (w.get("extra") or {}).get("exit_code"))
              for w in writes]
    # The admission record is written BEFORE any seam is open and is deliberately durable — that
    # is the whole point of `admit_benchmark_run`. Any write BETWEEN it and the seam's exit is
    # the forbidden pre-merge publication; before the fix there were three.
    assert states == [("refused", 1), ("refused", 2)], states


def test_cu_bridge_refusal_publishes_the_canonical_manifest_exactly_once(
    tmp_path, monkeypatch, capsys
):
    """The canonical manifest is published ONCE, after the seam — never in a pre-merge state.

    Asserting the FINAL content is not enough: it passes just as happily when the record is
    published TWICE — first from inside `finalize_run_manifest` carrying the admission seam's
    generic `exit_code` 1, then corrected on seam exit — which is exactly how this window
    survived the round that fixed the final artefact. The intermediate publish is observable:
    OSWorld ships multi-lane in this release, the canonical path is what a concurrent reader
    consumes, and an interruption inside the window leaves the wrong record durably. So the
    property under test is the WRITE SEQUENCE at that path, not its last element.
    """
    from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb

    main, canonical = _refusal_case_run_cu_bridge_agent_seed_gate(tmp_path, monkeypatch)
    target = canonical.resolve(strict=False)
    published: list[dict] = []
    real_write_json = rcb.write_json

    def _recording_write_json(path, payload):
        if Path(path).resolve(strict=False) == target:
            published.append(json.loads(json.dumps(payload)))      # snapshot exactly AS WRITTEN
        return real_write_json(path, payload)

    monkeypatch.setattr(rcb, "write_json", _recording_write_json)
    assert _process_status_of(main) == 2
    capsys.readouterr()

    states = [((p.get("extra") or {}).get("outcome"), (p.get("extra") or {}).get("exit_code"))
              for p in published]
    assert len(published) == 1, f"canonical manifest published {len(published)} times: {states}"
    # ...and the single published state is the real one, so no reader can ever observe a record
    # disagreeing with the status the process exits with.
    assert states == [("refused", 2)]


def test_runtime_attestation_requires_the_contracted_runtime_version_field(tmp_path, monkeypatch):
    """Only the CONTRACTED field counts as a runtime identity.

    `runtime_version` is part of the frozen `HealthResponse` (`ouroboros/gateway/contracts.py`).
    The helper used to fall back to a generic `version` key, so ANY unrelated HTTP server that
    answered `{"version": "6.75.0"}` attested successfully and ProgramBench's default admission
    path would bless a server that is not Ouroboros at all. Its absence is now the distinct,
    NON-overridable reason `runtime_version_absent` — the endpoint answered, but not with the
    health contract, so no live runtime identity was established.
    """
    from devtools.benchmarks.common import manifests

    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "VERSION").write_text("6.75.0\n", encoding="utf-8")
    _git_commit_all(repo)

    served: dict = {"version": "6.75.0"}          # a stranger's field, not the contract's

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps(served).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.delenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, raising=False)

    with pytest.raises(RuntimeError, match="reason=runtime_version_absent"):
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)

    # ... and the override does NOT rescue it: it waives a deliberate skew only.
    monkeypatch.setenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, "1")
    with pytest.raises(RuntimeError, match="reason=runtime_version_absent") as refused:
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert "does NOT waive" in str(refused.value)
    assert "runtime_version_absent" not in manifests.OVERRIDABLE_ATTESTATION_REASONS

    # The contracted field attests, with the same payload otherwise unchanged.
    served.clear()
    served["runtime_version"] = "6.75.0"
    attested = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert attested["ok"] is True and attested["reason"] == ""
    assert attested["runtime_version"] == "6.75.0"


# --- v6.79.0 P5.3/P5.4: harbor dataset identity, env passthrough, GAIA/TB seed gate ---

def _write_cached_task(cache_root, org, name, digest, timeout_sec):
    task = cache_root / org / name / digest
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(
        f"[agent]\ntimeout_sec = {timeout_sec}\n", encoding="utf-8"
    )
    return task / "task.toml"


def test_harbor_task_cache_lookup_uses_dataset_org_not_a_hardcoded_one(tmp_path, monkeypatch):
    """The adapter used to hardcode org `terminal-bench`, so every non-TB dataset silently ran
    deadline-blind. The org now comes from the threaded dataset identity."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    cache = tmp_path / "packages"
    _write_cached_task(cache, "terminal-bench", "shared-name", "aaa", 600)
    _write_cached_task(cache, "harbor-index", "shared-name", "bbb", 1800)
    monkeypatch.setattr(tb_agent.OuroborosTerminalBenchAgent, "_PACKAGE_CACHE_DIR", cache)

    logs = tmp_path / "logs" / "shared-name__trialhash" / "agent"
    logs.mkdir(parents=True)
    for dataset, expected in (
        ("terminal-bench/terminal-bench-2-1", 600),
        ("harbor-index/harbor-index-1-0", 1800),
    ):
        agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=logs, dataset=dataset)
        assert agent._cached_task_toml("shared-name").parent.parent.parent.name == dataset.split("/")[0]
        assert agent._resolve_task_timeout_from_dataset(object()) == expected


def test_harbor_task_cache_lookup_refuses_an_ambiguous_task_name(tmp_path, monkeypatch):
    """Same-named tasks in two orgs and no dataset org at all: returning either one would hand
    the agent a FOREIGN wall-clock cap, so the name-only lookup refuses (deadline-blind is the
    honest degradation). A single owner is still resolved when the dataset names no org."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    cache = tmp_path / "packages"
    _write_cached_task(cache, "terminal-bench", "collide", "aaa", 600)
    _write_cached_task(cache, "scale-ai", "collide", "bbb", 1200)
    _write_cached_task(cache, "gaia", "only-here", "ccc", 900)
    monkeypatch.setattr(tb_agent.OuroborosTerminalBenchAgent, "_PACKAGE_CACHE_DIR", cache)

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, dataset="")
    assert agent._cached_task_toml("collide") is None
    assert agent._cached_task_toml("only-here") is not None
    assert agent._cached_task_toml("absent") is None


def test_harbor_task_cache_lookup_never_borrows_another_orgs_timeout(tmp_path, monkeypatch):
    """An EXPLICIT dataset org is authoritative: no cross-owner fallback, ever.

    This previously fell back from a missing configured org to "any unique cache owner", and an
    earlier revision of the ambiguity test asserted that borrow as intended behaviour. It is not
    a lenient fallback — the borrowed field is the wall-clock cap, so the trial silently runs
    under another benchmark's deadline. Frontier-Bench (600s verifier caps) next to
    Terminal-Bench 2.1 (3600s) is the live 6x case, and both are routinely cached side by side
    on the same host."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent
    from devtools.benchmarks.terminal_bench import run_tb

    cache = tmp_path / "packages"
    # Only terminal-bench has this task cached; frontier-bench does not.
    _write_cached_task(cache, "terminal-bench", "borrowed-task", "aaa", 3600)
    monkeypatch.setattr(tb_agent.OuroborosTerminalBenchAgent, "_PACKAGE_CACHE_DIR", cache)

    logs = tmp_path / "logs" / "borrowed-task__trialhash" / "agent"
    logs.mkdir(parents=True)
    fb = tb_agent.OuroborosTerminalBenchAgent(logs_dir=logs, dataset=run_tb.FRONTIER_BENCH_DATASET)
    assert fb._cached_task_toml("borrowed-task") is None
    assert fb._resolve_task_timeout_from_dataset(object()) is None   # deadline-blind, not 3600
    # ...while the org that really owns the task still resolves its own cap.
    tb = tb_agent.OuroborosTerminalBenchAgent(logs_dir=logs, dataset=run_tb.DEFAULT_DATASET)
    assert tb._resolve_task_timeout_from_dataset(object()) == 3600


def test_frontier_bench_wall_clock_cap_resolves_from_its_own_cache_org(tmp_path, monkeypatch):
    """Frontier-Bench needs NO adapter change: harbor caches its tasks under org `frontier-bench`
    (verified against harbor 0.18.0, which populates
    `~/.cache/harbor/tasks/packages/frontier-bench/<task>/<digest>/task.toml`), so the already
    dataset-parametric lookup resolves FB's own cap even while TB2.1 caches a same-named task.
    FB caps are an order above TB2.1's (median 7200s), so picking the wrong org is not cosmetic."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent
    from devtools.benchmarks.terminal_bench import run_tb

    cache = tmp_path / "packages"
    _write_cached_task(cache, "terminal-bench", "bun-sourcemap-leak", "aaa", 600)
    _write_cached_task(cache, "frontier-bench", "bun-sourcemap-leak", "bbb", 1800)
    monkeypatch.setattr(tb_agent.OuroborosTerminalBenchAgent, "_PACKAGE_CACHE_DIR", cache)

    logs = tmp_path / "logs" / "bun-sourcemap-leak__trialhash" / "agent"
    logs.mkdir(parents=True)
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=logs, dataset=run_tb.FRONTIER_BENCH_DATASET)
    assert agent._cached_task_toml("bun-sourcemap-leak").parent.parent.parent.name == "frontier-bench"
    assert agent._resolve_task_timeout_from_dataset(object()) == 1800


def test_run_tb_job_config_carries_dataset_and_deep_merges_a_base_config(tmp_path):
    from devtools.benchmarks.terminal_bench import run_tb

    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "environment": {"env": {"UPSTREAM": "keep"}, "type": "docker"},
        "agents": [{"name": "Upstream Agent", "kwargs": {"dropped": True}}],
        "verifier": {"timeout_multiplier": 1.0},
    }), encoding="utf-8")
    cfg = run_tb.HarborCommandConfig(
        dataset="harbor-index/harbor-index-1-0", model="m", k=5, jobs_dir=tmp_path / "jd",
        harbor_bin="harbor", n_concurrent=1, task_filters=[], settings_path=tmp_path / "s.json",
        execute=False, light_model="m", base_job_config=base,
    )
    path = run_tb._write_agent_job_config(cfg)
    written = json.loads(path.read_text(encoding="utf-8"))

    # Upstream keys survive untouched; our agents[] block wins whole (name must stay ours,
    # a null/foreign agents[0].name permanently invalidates a submission).
    assert written["environment"] == {"env": {"UPSTREAM": "keep"}, "type": "docker"}
    assert written["verifier"] == {"timeout_multiplier": 1.0}
    assert len(written["agents"]) == 1
    assert written["agents"][0]["name"] == "Ouroboros Installed"
    assert "dropped" not in written["agents"][0]["kwargs"]
    assert written["agents"][0]["kwargs"]["dataset"] == "harbor-index/harbor-index-1-0"


def test_run_tb_forwards_agent_and_verifier_env_without_leaking_values(tmp_path):
    from devtools.benchmarks.terminal_bench import run_tb

    cfg = run_tb.HarborCommandConfig(
        dataset=run_tb.DEFAULT_DATASET, model="m", k=5, jobs_dir=tmp_path, harbor_bin="harbor",
        n_concurrent=1, task_filters=["t1"], settings_path=tmp_path / "s.json", execute=False,
        light_model="m", agent_env=("AWS_REGION=us-east-1",), verifier_env=("OPENAI_API_KEY=sk-secret",),
    )
    cmd = run_tb.harbor_command(cfg)

    assert cmd[cmd.index("--ae") + 1] == "AWS_REGION=us-east-1"
    assert cmd[cmd.index("--ve") + 1] == "OPENAI_API_KEY=sk-secret"
    safe = run_tb.redacted_command(cmd)
    assert "sk-secret" not in " ".join(safe)
    assert "OPENAI_API_KEY=<redacted>" in safe and "AWS_REGION=<redacted>" in safe
    # Nothing else about the command changes.
    assert [tok for tok in safe if "=" not in tok] == [tok for tok in cmd if "=" not in tok]


def _harbor_job_tree(root, *, cleartext: str, partial: str) -> dict:
    """A synthetic harbor 0.18.0 job tree, using harbor's REAL filenames and layout.

    Ground truth (installed harbor 0.18.0, `harbor/job.py`): the job config is
    `<jobs-dir>/<job_name>/config.json` — one timestamp level below the `--jobs-dir` our
    launcher passes — written as `self.config.model_dump_json(indent=4, exclude_defaults=True)`,
    and the same env dicts are re-serialized into the job `lock.json` and every trial's
    `config.json` / `lock.json` / `result.json`. `--ae` lands in `agents[].env`, `--ve` in
    `verifier.env`. Harbor's own `templatize_sensitive_env` writes a value VERBATIM when the
    NAME does not match `KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH`, and only partially
    (`value[:4] + "****" + value[-3:]`) when it does — both forms are planted here."""
    job = root / "job" / "2026-07-25__12-00-00"
    trial = job / "some-task__abc123"
    (trial / "agent").mkdir(parents=True)
    written = {}
    env_block = {"JUDGE_API_KEY": partial, "MY_BEARER": cleartext}
    for name in ("config.json", "lock.json"):
        p = job / name
        p.write_text(json.dumps({"verifier": {"env": env_block}, "agents": [{"env": env_block}]},
                                indent=4), encoding="utf-8")
        written[str(p)] = True
    for name in ("config.json", "lock.json", "result.json"):
        p = trial / name
        p.write_text(json.dumps({"config": {"verifier": {"env": env_block}}}, indent=4),
                     encoding="utf-8")
        written[str(p)] = True
    # harbor's only un-redacted path (`trial.py` writes `traceback.format_exc()`), plus an
    # agent-written log: both can carry the resolved cleartext value.
    (trial / "exception.txt").write_text(f"RuntimeError: Command: docker -e MY_BEARER={cleartext}\n",
                                         encoding="utf-8")
    (trial / "agent" / "session.log").write_text(f"env MY_BEARER={cleartext}\n", encoding="utf-8")
    return written


def test_scrub_covers_the_harbor_written_job_config_for_ae_ve_values(tmp_path, monkeypatch):
    """The leak this closes: harbor persists its own JobConfig (and lock/result files) into the
    job dir that gets PUBLICLY uploaded, so a `--ve` value is on disk even though the launcher's
    own artifacts carry names only. The scrub must sweep the whole tree BY VALUE.

    Deterministic and self-contained: the tree is built here, the secret is obviously fake, and
    nothing depends on the ambient checkout or on a real key."""
    from devtools.benchmarks.terminal_bench import run_tb
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    fake = "FAKEfake-judge-key-0000000000deadbeef"   # obviously fake; never a real credential
    # 1. The launcher really does hand this value to harbor's `--ve`.
    cmd = run_tb.harbor_command(run_tb.HarborCommandConfig(
        dataset=run_tb.DEFAULT_DATASET, model="m", k=5, jobs_dir=tmp_path / "jobs",
        harbor_bin="harbor", n_concurrent=1, task_filters=["t1"],
        settings_path=tmp_path / "s.json", execute=False, light_model="m",
        verifier_env=(f"JUDGE_API_KEY={fake}",),
    ))
    assert cmd[cmd.index("--ve") + 1] == f"JUDGE_API_KEY={fake}"
    assert fake not in " ".join(run_tb.redacted_command(cmd))

    # 2. A submission copy of the job dir, in harbor's real shape.
    root = tmp_path / "job_copy"
    root.mkdir()
    partial = scrub.harbor_redacted_form(fake)
    assert partial and partial != fake                    # harbor leaks 7 chars, not zero
    _harbor_job_tree(root, cleartext=fake, partial=partial)
    sources = tmp_path / "fake_settings.json"
    sources.write_text(json.dumps({"OPENROUTER_API_KEY": "FAKEfake-other-value-1111"}),
                       encoding="utf-8")

    # 3. Scrub, then assert the value is gone from EVERY file in the tree.
    monkeypatch.setattr(sys, "argv", ["scrub", "--root", str(root), "--secrets-from", str(sources),
                                      "--env-passthrough", f"JUDGE_API_KEY={fake}"])
    assert scrub.main() == 0
    files = [p for p in root.rglob("*") if p.is_file()]
    assert len(files) >= 7
    for path in files:
        raw = path.read_bytes()
        assert fake.encode() not in raw, f"cleartext survived in {path}"
        assert partial.encode() not in raw, f"harbor's partial form survived in {path}"
    # Structure preserved: the config is still valid JSON with the key present, value redacted.
    cfg = json.loads((root / "job" / "2026-07-25__12-00-00" / "config.json").read_text(encoding="utf-8"))
    assert cfg["verifier"]["env"]["MY_BEARER"] == "<REDACTED:JUDGE_API_KEY>"


def test_scrub_keeps_every_passthrough_occurrence_of_a_repeated_env_name(tmp_path, monkeypatch):
    """One env NAME, two DIFFERENT values (agent phase vs verifier phase) — the real shape when a
    judge key and an agent key share a name, or when a flag is repeated. Keying the passthrough
    needles on the NAME alone dropped the earlier value, so a CORRECT scrub invocation published
    that credential in harbor's job tree. Every distinct occurrence must survive collection."""
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    agent_value = "FAKEfake-agent-value-000000000000aaaa"    # obviously fake; never a credential
    verifier_value = "FAKEfake-verifier-value-11111111bbbb"
    name = "SHARED_API_KEY"

    # Collection alone must retain both values (plus each one's harbor partial form).
    needles, refusals = scrub.collect_env_passthrough(
        [f"{name}={agent_value}", f"{name}={verifier_value}"]
    )
    assert refusals == []
    assert sorted(needles.values()) == sorted([
        agent_value, verifier_value,
        scrub.harbor_redacted_form(agent_value), scrub.harbor_redacted_form(verifier_value),
    ])
    # An exact repeat is the same secret, not a second one: it must not inflate the needle set.
    repeated, _ = scrub.collect_env_passthrough([f"{name}={agent_value}"] * 3)
    assert len(repeated) == 2   # the value plus its harbor partial form

    # End-to-end: both values are planted in harbor's own job tree and both must be gone.
    root = tmp_path / "job_copy"
    job = root / "job" / "2026-07-25__12-00-00"
    job.mkdir(parents=True)
    (job / "config.json").write_text(
        json.dumps({"agents": [{"env": {name: agent_value}}],
                    "verifier": {"env": {name: verifier_value}}}, indent=4),
        encoding="utf-8",
    )
    sources = tmp_path / "fake_settings.json"
    sources.write_text(json.dumps({"OPENROUTER_API_KEY": "FAKEfake-other-value-1111"}),
                       encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "scrub", "--root", str(root), "--secrets-from", str(sources),
        "--env-passthrough", f"{name}={agent_value}",
        "--env-passthrough", f"{name}={verifier_value}",
    ])
    assert scrub.main() == 0
    raw = (job / "config.json").read_bytes()
    assert agent_value.encode() not in raw and verifier_value.encode() not in raw
    # Structure preserved: both entries are still present, each redacted under its own label.
    cfg = json.loads((job / "config.json").read_text(encoding="utf-8"))
    assert cfg["agents"][0]["env"][name].startswith("<REDACTED:")
    assert cfg["verifier"]["env"][name].startswith("<REDACTED:")
    assert cfg["agents"][0]["env"][name] != cfg["verifier"]["env"][name]


def test_scrub_fails_closed_on_an_unsweepable_ae_ve_value_and_changes_nothing(tmp_path, monkeypatch):
    """A value we cannot sweep safely must ABORT before any write: a maybe-scrubbed tree that
    then gets uploaded is strictly worse than no submission at all."""
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    root = tmp_path / "job_copy"
    root.mkdir()
    target = root / "job" / "2026-07-25__12-00-00"
    target.mkdir(parents=True)
    before = json.dumps({"verifier": {"env": {"SHORT_TOKEN": "ab1"}}}, indent=4)
    (target / "config.json").write_text(before, encoding="utf-8")
    sources = tmp_path / "fake_settings.json"
    sources.write_text(json.dumps({"OPENROUTER_API_KEY": "FAKEfake-other-value-1111"}),
                       encoding="utf-8")

    for bad in ("SHORT_TOKEN=ab1",           # too short to sweep safely
                "WORDY_TOKEN=onlyletters",   # not credential-shaped
                "NOEQUALS"):                 # malformed pair
        monkeypatch.setattr(sys, "argv", ["scrub", "--root", str(root), "--secrets-from",
                                          str(sources), "--env-passthrough", bad])
        assert scrub.main() == 2, bad
        assert (target / "config.json").read_text(encoding="utf-8") == before, bad


def test_scrub_sweeps_and_verifies_json_escaped_forms_not_only_the_literal(tmp_path, monkeypatch):
    r"""The false all-clear this closes. Harbor persists env values through JSON serializers, so
    a value containing a quote, a backslash, a control character or a non-ASCII character is on
    disk ESCAPED (``abc"1234`` is stored as ``abc\"1234``). A literal-only sweep walked past it
    AND the literal-only verify then printed zero leftovers — a scrubber that misses a secret and
    reports success is worse than no scrubber, because it turns "check this by hand" into a tool
    verdict. Both passes must see every persisted form.

    Self-contained: values are obviously fake, the tree is built here, nothing reads the ambient
    checkout, the cwd or any real credential source."""
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    # One awkward value per escape class the reviewer named. Obviously fake, never credentials.
    awkward = {
        "QUOTE_TOKEN": 'FAKEfake-quote-"-0000000001',
        "BACKSLASH_TOKEN": "FAKEfake-backslash-\\-0000002",
        "CONTROL_TOKEN": "FAKEfake-control-\x01-0000003",
        "UNICODE_TOKEN": "FAKEfake-nonascii-é-000004",
    }

    # 1. The encoded forms are the SERIALIZER's output, not a hand-kept escape table.
    assert scrub.json_encoded_forms(awkward["QUOTE_TOKEN"]) == ['FAKEfake-quote-\\"-0000000001']
    assert scrub.json_encoded_forms(awkward["BACKSLASH_TOKEN"]) == ["FAKEfake-backslash-\\\\-0000002"]
    assert scrub.json_encoded_forms(awkward["CONTROL_TOKEN"]) == ["FAKEfake-control-\\u0001-0000003"]
    # ensure_ascii=True escapes the non-ASCII char; ensure_ascii=False leaves it == the literal,
    # which is already swept as its own needle, so exactly one extra form is produced.
    assert scrub.json_encoded_forms(awkward["UNICODE_TOKEN"]) == ["FAKEfake-nonascii-\\u00e9-000004"]
    # A value with nothing to escape adds no needle at all.
    assert scrub.json_encoded_forms("FAKEfake-plain-00000005") == []
    expanded = scrub.expand_encoded_forms({"QUOTE_TOKEN": awkward["QUOTE_TOKEN"]})
    assert expanded["QUOTE_TOKEN"] == awkward["QUOTE_TOKEN"]
    assert expanded["QUOTE_TOKEN:json"] == 'FAKEfake-quote-\\"-0000000001'

    # 2. A harbor-shaped tree holding each value in BOTH serializer configurations, plus the raw
    #    literals in an un-redacted log (harbor's traceback path).
    root = tmp_path / "job_copy"
    job = root / "job" / "2026-07-25__12-00-00"
    job.mkdir(parents=True)
    payload = {"verifier": {"env": dict(awkward)}, "agents": [{"env": dict(awkward)}]}
    (job / "config.json").write_text(       # pydantic/serde shape: raw UTF-8
        json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    (job / "lock.json").write_text(         # python json default shape: \uXXXX
        json.dumps(payload, indent=4, ensure_ascii=True), encoding="utf-8")
    (job / "exception.txt").write_text(
        "".join(f"RuntimeError: docker -e {name}={value}\n" for name, value in awkward.items()),
        encoding="utf-8")
    # A --secrets-from value needs the same treatment; expansion is not passthrough-only.
    from_source = 'FAKEfake-source-"-000000006'
    sources = tmp_path / "fake_settings.json"
    sources.write_text(json.dumps({"OPENROUTER_API_KEY": from_source}), encoding="utf-8")
    (job / "settings_echo.json").write_text(
        json.dumps({"OPENROUTER_API_KEY": from_source}, indent=4), encoding="utf-8")

    argv = ["scrub", "--root", str(root), "--secrets-from", str(sources)]
    for name, value in awkward.items():
        argv += ["--env-passthrough", f"{name}={value}"]
    monkeypatch.setattr(sys, "argv", list(argv))

    # 3. Every form of every value must be gone — this is what failed before the fix.
    assert scrub.main() == 0
    files = [p for p in root.rglob("*") if p.is_file()]
    for path in files:
        raw = path.read_bytes()
        for value in (*awkward.values(), from_source):
            assert value.encode() not in raw, f"literal survived in {path}"
            for form in scrub.json_encoded_forms(value):
                assert form.encode() not in raw, f"JSON-escaped form survived in {path}"
    # Structure preserved: still valid JSON, keys intact, values redacted.
    for name in ("config.json", "lock.json"):
        cfg = json.loads((job / name).read_text(encoding="utf-8"))
        assert sorted(cfg["verifier"]["env"]) == sorted(awkward)
        for redacted in cfg["verifier"]["env"].values():
            assert redacted.startswith("<REDACTED:")

    # 4. The VERIFY pass must refuse to declare success while an escaped form remains. Re-planting
    #    only the escaped form is precisely the case that used to exit 0 with the secret on disk.
    (job / "lock.json").write_text(
        json.dumps({"verifier": {"env": {"QUOTE_TOKEN": awkward["QUOTE_TOKEN"]}}},
                   indent=4, ensure_ascii=False),
        encoding="utf-8")
    planted = (job / "lock.json").read_text(encoding="utf-8")
    assert 'FAKEfake-quote-\\"-0000000001' in planted        # escaped form only
    assert awkward["QUOTE_TOKEN"] not in planted             # literal is NOT on disk
    monkeypatch.setattr(scrub, "_sweep_file", lambda path, secrets: (0, []))  # verify pass alone
    monkeypatch.setattr(sys, "argv", list(argv))
    assert scrub.main() == 1


def test_run_tb_submission_subtree_is_derived_from_the_dataset():
    from devtools.benchmarks.terminal_bench import run_tb

    # TB2.1 keeps its published layout byte-identical.
    assert run_tb.submission_subtree("terminal-bench/terminal-bench-2-1") == ("terminal-bench", "2.1")
    # Another dataset no longer lands in the TB2.1 tree.
    assert run_tb.submission_subtree("harbor-index/harbor-index-1-0") == ("harbor-index", "1.0")
    family, version = run_tb.submission_subtree("some-org/unversioned")
    assert (family, version) == ("unversioned", "")


def test_run_tb_submission_subtree_components_are_confined():
    """`submission_root` is validated, but the job dir is DERIVED from it — so the components that
    are about to be created are what must be checked, not their already-checked ancestor. Pure
    function: no env, no cwd, no repo path, nothing derived from `__file__`."""
    from devtools.benchmarks.terminal_bench import run_tb

    # Accepted shapes are unchanged, derived and explicit alike (trailing slash still tolerated).
    assert run_tb.confined_submission_subtree("", dataset="terminal-bench/terminal-bench-2-1") == [
        "terminal-bench", "2.1",
    ]
    assert run_tb.confined_submission_subtree("terminal-bench/2.1", dataset="ignored") == ["terminal-bench", "2.1"]
    assert run_tb.confined_submission_subtree("frontier-bench/", dataset="ignored") == ["frontier-bench"]

    for escape in ("..", "../..", "../../../etc", "terminal-bench/../../..", ".", "./x", "/abs/path", "/"):
        with pytest.raises(ValueError):
            run_tb.confined_submission_subtree(escape, dataset="terminal-bench/terminal-bench-2-1")

    # Windows forms: `\` is a separator and `C:` a drive qualifier there, and this repo's CI matrix
    # runs all three OSes — a value that is an inert directory name on POSIX escapes on Windows.
    for windows_form in ("..\\..\\evil", "sub\\dir", "C:\\evil", "C:/evil", "C:evil", "\\\\server\\share", "\\evil"):
        with pytest.raises(ValueError):
            run_tb.confined_submission_subtree(windows_form, dataset="terminal-bench/terminal-bench-2-1")

    # The DERIVED path is untrusted too: `--dataset` reaches it through submission_subtree(), which
    # splits the name without judging it.
    assert run_tb.submission_subtree("org/..-2-1") == ("..", "2.1")
    with pytest.raises(ValueError):
        run_tb.confined_submission_subtree("", dataset="org/..-2-1")


def test_run_tb_refuses_an_escaping_subtree_before_creating_anything(tmp_path, monkeypatch):
    """The refusal must land ahead of the first mkdir, so a rejected run leaves no directories at
    all. Hermetic: cwd is redirected into tmp_path, so a regression that reaches the run-root
    default writes there and is caught by the emptiness assertion instead of touching a checkout."""
    from devtools.benchmarks.terminal_bench import run_tb

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="single safe path component"):
        run_tb.main([
            "--model", "anthropic/claude-sonnet-5",
            "--submission-root", str(tmp_path / "submission"),
            "--run-root", str(tmp_path / "run"),
            "--submission-subtree", "../../../escaped",
        ])
    assert list(tmp_path.iterdir()) == []


def test_gaia_and_tb_launchers_run_the_shared_seed_gate(tmp_path, monkeypatch):
    """P5.4: GAIA and TB dropped their v6.75.0 `require_clean=False` pins AND route both manifest
    seams, so the refusal is DURABLE: the record reaches disk and no other artefact does. Asserting
    only `pytest.raises` is what let an inert handler pass review, so every launcher's PERSISTED
    outcome is checked here. Deterministic — the gate runs against a PURPOSE-BUILT dirty repo,
    never the ambient checkout."""
    import devtools.benchmarks.gaia.run_gaia as run_gaia
    from devtools.benchmarks.common.manifests import BenchmarkAdmissionRefused
    from devtools.benchmarks.terminal_bench import run_harbor_smoke, run_tb

    seed = tmp_path / "seed"
    _git_repo(seed)
    (seed / "VERSION").write_text("6.79.0\n", encoding="utf-8")
    _git_commit_all(seed)
    monkeypatch.setattr(run_gaia, "REPO", seed)
    monkeypatch.setattr(run_tb, "repo_root_from_devtools", lambda: seed)
    monkeypatch.setattr(run_harbor_smoke, "repo_root_from_devtools", lambda: seed)
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    def _extra(run_dir):
        return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["extra"]

    # Clean seed: GAIA admits, records the gate verdict, augments the manifest with the
    # settings-derived slots, and the finalization seam names the terminal outcome.
    clean = tmp_path / "clean"
    assert run_gaia.main(["--out-dir", str(clean), "--solve-model", "m", "--dry-run"]) == 0
    manifest = json.loads((clean / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed_gate"]["ok"] is True and manifest["seed_gate"]["require_clean"] is True
    assert manifest["model_slots"]["OUROBOROS_MODEL"] == "m"
    assert manifest["extra"]["outcome"] == "dry_run" and manifest["extra"]["exit_code"] == 0

    # Dirty seed: refused before anything is spent -- and the REFUSAL is on disk, so a shard
    # wrapper reading run_manifest.json can tell "refused" from "never started" or "crashed".
    (seed / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")

    dirty = tmp_path / "dirty-gaia"
    with pytest.raises(BenchmarkAdmissionRefused, match="seed_dirty"):
        run_gaia.main(["--out-dir", str(dirty), "--solve-model", "m", "--dry-run"])
    extra = _extra(dirty)
    assert extra["outcome"] == "refused" and extra["exit_code"] == 1
    assert extra["refusal"] == {"stage": "seed_gate", "reason": "seed_dirty", "exit_code": 1}
    # The renderer that injects LIVE provider keys into the run dir never ran.
    assert not (dirty / "settings.json").exists()

    def _no_probe(_harbor_bin):
        raise AssertionError("harbor --version was probed before admission")

    monkeypatch.setattr(run_tb, "harbor_version", _no_probe)
    tb_root = tmp_path / "dirty-tb"
    with pytest.raises(BenchmarkAdmissionRefused, match="seed_dirty"):
        run_tb.main(["--model", "m", "--run-root", str(tb_root), "--settings-path", str(settings)])
    assert _extra(tb_root)["refusal"]["stage"] == "seed_gate"
    # No half-built submission tree (no job dir, no metadata.yaml).
    assert not (tb_root / "submission" / "submissions").exists()

    smoke_root = tmp_path / "dirty-smoke"
    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--run-root", str(smoke_root),
                                      "--settings-path", str(settings)])
    with pytest.raises(BenchmarkAdmissionRefused, match="seed_dirty"):
        run_harbor_smoke.main()
    assert _extra(smoke_root)["refusal"]["stage"] == "seed_gate"
    assert not (smoke_root / "harbor_command.json").exists()
    assert not (smoke_root / "result_index.jsonl").exists()

    # ...unless the escape is recorded.
    escaped = tmp_path / "escaped"
    assert run_gaia.main(["--out-dir", str(escaped), "--solve-model", "m", "--dry-run",
                          "--allow-dirty-seed"]) == 0
    recorded = json.loads((escaped / "run_manifest.json").read_text(encoding="utf-8"))
    assert recorded["seed_gate"]["allow_dirty_seed"] is True
    assert recorded["seed_gate"]["reason"] == "seed_dirty"
    assert recorded["extra"]["outcome"] == "dry_run"


def _inspect_eval_log(status: str, samples: list[dict], *, error: dict | None = None) -> dict:
    """A minimal inspect eval log in the shape `--log-format json` writes and run_gaia reads."""
    log: dict = {"version": 2, "status": status, "eval": {"task": "inspect_evals/gaia"},
                 "plan": {}, "stats": {}, "samples": samples}
    if error is not None:
        log["error"] = error
    return log


def test_run_gaia_cannot_record_a_dead_inspect_eval_as_completed(tmp_path, monkeypatch):
    """A DEAD eval must reach BOTH the outcome and the exit code — the fail-open this release
    exists to remove, found inside the release's own machinery.

    In the v6.81.0 GAIA smoke every sample died in `RuntimeError: Timed out executing setup
    command in sandbox`, nothing was scored, and the run manifest recorded
    `outcome="completed", exit_code=0`, because `inspect eval` has NO non-zero exit path for a
    task that raised: it reports the failure in its log and still returns 0. Every leg below
    therefore pins `harness_exit_code == 0` — the harness lies in all of them, so an
    implementation that reads the return code cannot pass, and one that only ensured the field
    is PRESENT cannot either.

    The three outcomes are kept apart deliberately: an eval that raised, an eval that scored
    nothing, and an eval that scored genuine zeros are different facts, and only the last is a
    result. Hermetic by construction — purpose-built seed repo, tmp settings, tmp run roots, the
    port picker and provider-key resolver stubbed, and the eval injected at the `subprocess.run`
    seam, so nothing depends on OUROBOROS_* env, the cwd, or the ambient checkout.
    """
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    seed = tmp_path / "seed"
    _git_repo(seed)
    (seed / "VERSION").write_text("6.81.0\n", encoding="utf-8")
    _git_commit_all(seed)
    monkeypatch.setattr(run_gaia, "REPO", seed)
    monkeypatch.setattr(run_gaia, "_free_port", lambda: 19999)
    monkeypatch.setattr(run_gaia, "_resolve_provider_keys", lambda needed: {})
    base_settings = tmp_path / "settings_base.json"
    base_settings.write_text("{}", encoding="utf-8")

    def _run(name: str, log: dict | None) -> tuple[int, dict]:
        run_dir = tmp_path / name

        def fake_run(cmd, **kwargs):
            if log is not None:
                log_dir = Path(cmd[cmd.index("--log-dir") + 1])
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "eval.json").write_text(json.dumps(log), encoding="utf-8")
            # Exactly what the real CLI does after a dead eval: return 0.
            return subprocess.CompletedProcess(args=list(cmd), returncode=0)

        monkeypatch.setattr(run_gaia.subprocess, "run", fake_run)
        code = run_gaia.main(["--out-dir", str(run_dir), "--solve-model", "m",
                              "--settings", str(base_settings), "--sample-id", "task-a,task-b"])
        extra = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
        return code, extra

    # 1. The eval RAISED: an infra zero. The benchmark did not run, so it is not `completed` and
    #    the process must not exit 0 — a shard wrapper reads that exit code.
    raised = _inspect_eval_log(
        "error",
        [{"id": "task-a", "scores": {}, "error": {"message": "RuntimeError('Timed out executing setup command in sandbox')"}}],
        error={"message": "RuntimeError('Timed out executing setup command in sandbox')"},
    )
    code, extra = _run("raised", raised)
    assert extra["outcome"] == "eval_error"
    assert extra["exit_code"] != 0 and code == extra["exit_code"]
    assert extra["harness_exit_code"] == 0  # the harness claimed success
    assert "Timed out executing setup command in sandbox" in extra["inspect_eval"]["error"]
    assert extra["inspect_eval"]["scored_samples"] == 0

    # 2. The eval FINISHED and scored nothing: still not a result, and still not `completed`.
    code, extra = _run("unscored", _inspect_eval_log("success", []))
    assert extra["outcome"] == "no_scored_samples"
    assert extra["exit_code"] != 0 and code == extra["exit_code"]
    assert extra["harness_exit_code"] == 0

    # 3. GENUINE zeros: samples that reached the official scorer and were marked incorrect. This
    #    IS a result — real capability data — and must stay `completed` with exit 0, or the
    #    honest zero becomes indistinguishable from the infra zero in the other direction.
    scored_zero = _inspect_eval_log("success", [
        {"id": "task-a", "scores": {"gaia_scorer": {"value": "I"}}},
        {"id": "task-b", "scores": {"gaia_scorer": {"value": "I"}}},
    ])
    code, extra = _run("genuine_zero", scored_zero)
    assert extra["outcome"] == "completed" and extra["exit_code"] == 0 and code == 0
    assert extra["inspect_eval"]["scored_samples"] == 2

    # 4. No readable log at all: fail CLOSED. Unknown success is not success — the same rule the
    #    seed gate applies to unknown cleanliness.
    code, extra = _run("nolog", None)
    assert extra["outcome"] == "eval_status_unavailable"
    assert extra["exit_code"] != 0 and code == extra["exit_code"]


def test_run_gaia_never_silently_clips_the_harness_error_it_records(tmp_path):
    """The record of an infrastructure failure must not itself destroy the evidence.

    The first cut of this fix clipped the message at a hardcoded `[:1000]` — a silent truncation
    (BIBLE P1 / docs/DEVELOPMENT.md "No silent truncation") in the one place it hurts most: a deep
    traceback from a sandbox that died is exactly the error whose TAIL is informative. Messages now
    pass through whole; an implausibly large one is cut only through the shared
    `truncate_review_artifact` seam, which discloses the cut and the original length, and
    `error_log` always names the file holding the untouched message and its traceback."""
    import devtools.benchmarks.gaia.run_gaia as run_gaia

    def _summary(message: str) -> dict:
        log_path = tmp_path / f"eval-{len(message)}.json"
        log_path.write_text(json.dumps(_inspect_eval_log(
            "error", [], error={"message": message})), encoding="utf-8")
        return run_gaia.read_inspect_eval_summary([log_path]), log_path

    # A 4000-char traceback — four times the old cap — survives INTACT, tail included.
    long_error = "RuntimeError: sandbox died\n" + "".join(
        f'  File "frame{i}.py", line {i}, in run\n' for i in range(100)) + "TAIL-MARKER"
    assert len(long_error) > 1000
    summary, log_path = _summary(long_error)
    assert summary["error"] == long_error
    assert summary["error"].endswith("TAIL-MARKER")
    assert summary["error_log"] == str(log_path)

    # Beyond the disclosed budget the cut is DISCLOSED, never silent, and names the true length.
    huge = "x" * (run_gaia._INSPECT_ERROR_DISCLOSED_LIMIT + 5000)
    summary, log_path = _summary(huge)
    assert "⚠️ OMISSION NOTE" in summary["error"]
    assert str(len(huge)) in summary["error"]
    # ...and the reader reaches the whole thing without guessing which file to open.
    assert summary["error_log"] == str(log_path)


def test_run_tb_classifies_a_harbor_job_by_its_trials_not_its_exit_code():
    """The sibling swallow: `harbor run` has no non-zero exit path for a job whose trials all
    ERRORED either (2026-07-04: a job wrote 444 trial `result.json` files and zero rewards while
    looking healthy), so run_tb decides from the disclosure ledger it already builds.

    Same three-way distinction as GAIA, and the same reason for it: an all-zero reward
    distribution over SCORED trials is a genuine result, while trials that never reached the
    verifier are not."""
    from devtools.benchmarks.terminal_bench.run_tb import classify_harbor_outcome

    # Scored trials, all zero -> a genuine result.
    assert classify_harbor_outcome({"n_trials": 4, "reward_distribution": {"0.0": 4}}, 0) == ("completed", 0)
    assert classify_harbor_outcome({"n_trials": 4, "reward_distribution": {"0.0": 3, "1.0": 1}}, 0) == ("completed", 0)
    # Nothing reached the verifier -> an infra zero, non-zero exit despite harbor's 0.
    assert classify_harbor_outcome({"n_trials": 444, "reward_distribution": {"null": 444}}, 0) == ("no_scored_trials", 1)
    assert classify_harbor_outcome({"n_trials": 0, "reward_distribution": {}}, 0) == ("no_scored_trials", 1)
    # Ledger unavailable -> no evidence of a result; fail closed rather than claim `completed`.
    assert classify_harbor_outcome(None, 0) == ("trials_unverified", 1)
    # A harness that DID fail keeps its own status.
    assert classify_harbor_outcome({"n_trials": 4, "reward_distribution": {"1.0": 4}}, 2) == ("harness_nonzero_exit", 2)


def test_run_tb_manifest_records_the_model_the_run_actually_resolved(tmp_path, monkeypatch):
    """Record the model that ran, not decoy env/settings templates.

    The ``--all-model`` leg also pins post-parse override truth. The seed repo, settings,
    run roots, cwd and harbor probe are isolated from the operator workspace.
    """
    from devtools.benchmarks.common import model_slots
    from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS
    from devtools.benchmarks.terminal_bench import run_tb

    seed = tmp_path / "seed"
    _git_repo(seed)
    monkeypatch.setattr(run_tb, "repo_root_from_devtools", lambda: seed)
    monkeypatch.setattr(run_tb, "harbor_version", lambda _harbor_bin: "")
    monkeypatch.chdir(tmp_path)
    for key in (
        *MODEL_SLOT_KEYS, *model_slots._ACTIVE_LOCAL_ROUTE_KEYS,
        model_slots.SUBAGENTS_SETTING, model_slots.REVIEWER_SLOTS_ENV, "USE_LOCAL_HEAVY",
    ):
        monkeypatch.setenv(key, "")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"OUROBOROS_MODEL": "decoy/template-main",
                    "OUROBOROS_MODEL_LIGHT": "decoy/template-light"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_MODEL", "decoy/ambient-main")

    def _manifest(run_root):
        return json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))

    measured = tmp_path / "measured"
    assert run_tb.main([
        "--model", "anthropic/claude-fable-5",
        "--light-model", "google/gemini-3.5-flash",
        "--run-root", str(measured),
        "--submission-root", str(tmp_path / "submission"),
        "--settings-path", str(settings),
    ]) == 0
    manifest = _manifest(measured)
    slots = manifest["model_slots"]
    # The measured model, NOT the ambient env decoy and NOT the settings-template decoy.
    assert slots["OUROBOROS_MODEL"] == "anthropic/claude-fable-5"
    assert slots["OUROBOROS_MODEL_LIGHT"] == "google/gemini-3.5-flash"
    # New manifests keep legacy Heavy absent.
    assert "OUROBOROS_MODEL_HEAVY" not in slots
    assert slots["OUROBOROS_MODEL_FALLBACKS"] == "anthropic/claude-fable-5"
    assert "decoy/ambient-main" not in slots.values()
    assert "decoy/template-main" not in slots.values()
    # `model_slots` means the same thing here as in GAIA's manifest: MODEL_SLOT_KEYS only.
    assert set(slots).issubset(set(MODEL_SLOT_KEYS))
    # ...and the same fact is on disk from admission onward, in TB's established `extra` shape.
    assert manifest["extra"]["model"] == "anthropic/claude-fable-5"
    assert manifest["extra"]["light_model"] == "google/gemini-3.5-flash"

    # --all-model rewrites --model AFTER parsing; the manifest must follow the override, not the
    # (here empty) --model it was parsed with.
    single = tmp_path / "single"
    assert run_tb.main([
        "--all-model", "openai/gpt-5.6-sol",
        "--run-root", str(single),
        "--submission-root", str(tmp_path / "submission"),
        "--settings-path", str(settings),
    ]) == 0
    single_manifest = _manifest(single)
    assert single_manifest["model_slots"]["OUROBOROS_MODEL"] == "openai/gpt-5.6-sol"
    assert single_manifest["model_slots"]["OUROBOROS_MODEL_LIGHT"] == "openai/gpt-5.6-sol"
    assert single_manifest["extra"]["model"] == "openai/gpt-5.6-sol"
    # Every forwarded slot the single-model run pinned is recorded as that one model.
    for key in run_tb._ALL_MODEL_SLOT_KEYS:
        assert single_manifest["model_slots"][key] == "openai/gpt-5.6-sol"
    # Slots the in-container adapter never forwards stay OUT: recording a model the container
    # cannot see would be as false as recording the wrong one.
    assert not set(single_manifest["model_slots"]) & set(run_tb._UNFORWARDED_MODEL_SLOT_KEYS)


def test_gaia_and_tb_launchers_add_no_runtime_attestation(tmp_path):
    """Owner Q10: TB and GAIA are structurally immune (each sample/trial starts its own server
    from the checkout under test), so they get the seed gate and NOT attestation lines."""
    tb_dir = REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench"
    gaia_dir = REPO_ROOT / "devtools" / "benchmarks" / "gaia"
    for path in (tb_dir / "run_tb.py", tb_dir / "run_harbor_smoke.py", gaia_dir / "run_gaia.py",
                 gaia_dir / "run_harness.py"):
        src = path.read_text(encoding="utf-8")
        assert "runtime_attestation" not in src, f"{path.name} must not attest a live runtime"
    for path in (tb_dir / "run_tb.py", tb_dir / "run_harbor_smoke.py", gaia_dir / "run_gaia.py"):
        assert "require_clean=not " in path.read_text(encoding="utf-8"), f"{path.name} lost its seed gate"


def test_scrubber_refuses_symlinks_instead_of_writing_through_them(tmp_path, capsys, monkeypatch):
    """A symlink under --root must stop the scrub dead, before anything is written.

    Two independent failures, both proven against the pre-fix tool:

    * `p.is_file()` and `path.write_text()` BOTH follow a file symlink, so the sweep
      rewrites the link's TARGET outside --root. A pack linking to the live settings.json
      had its real keys replaced with `<REDACTED:...>` by the tool meant to protect them.
      `cp -a` preserves symlinks, so the procedural "run this on a COPY" rule does not
      help — the copy carries the same link.
    * `rglob` does not descend through a DIRECTORY symlink, yet the verify pass still
      printed `verify_leftovers=0` and exited 0. The tool certified a tree it had never
      read, for content reachable under --root and about to be uploaded publicly.

    The second is the one that matters most: silent non-coverage reported as cleanliness
    is precisely the class of false claim this release exists to remove, and here the
    consequence is a live API key on a public leaderboard."""
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    fake = "FAKEfake-scrub-symlink-000000000000cccc"   # obviously fake; never a credential

    outside = tmp_path / "outside"
    outside.mkdir()
    live = outside / "live_settings.json"
    live.write_text(f'{{"OPENROUTER_API_KEY": "{fake}"}}\n', encoding="utf-8")

    behind_dir_link = tmp_path / "behind"
    behind_dir_link.mkdir()
    (behind_dir_link / "deep.txt").write_text(f"token={fake}\n", encoding="utf-8")

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "normal.txt").write_text(f"plain={fake}\n", encoding="utf-8")
    (pack / "linked_settings.json").symlink_to(live)
    (pack / "hidden_dir").symlink_to(behind_dir_link, target_is_directory=True)

    secrets_src = tmp_path / "secrets.txt"
    secrets_src.write_text(f"OPENROUTER_API_KEY: {fake}\n", encoding="utf-8")

    argv = ["scrub_submission_secrets.py", "--root", str(pack),
            "--secrets-from", str(secrets_src)]
    monkeypatch.setattr(sys, "argv", argv)
    rc = scrub.main()

    assert rc == 2, "a symlink under --root must be a hard refusal, not a warning"

    err = capsys.readouterr().err
    assert "REFUSING TO SCRUB" in err
    # BOTH kinds must be named, with their targets, so the operator can act.
    assert "linked_settings.json" in err and str(live) in err
    assert "hidden_dir" in err and str(behind_dir_link) in err

    # Fail CLOSED: not one byte written anywhere — not through the link, not even to the
    # ordinary file the tool could legitimately have swept.
    assert fake in live.read_text(encoding="utf-8"), "wrote through the symlink"
    assert fake in (behind_dir_link / "deep.txt").read_text(encoding="utf-8")
    assert fake in (pack / "normal.txt").read_text(encoding="utf-8"), (
        "a refusal must leave the tree untouched; a partially swept pack is worse than "
        "an unswept one"
    )
    assert (pack / "linked_settings.json").is_symlink(), "must not have been replaced"

    # ...and with the links gone the tool still does its job, so the guard is a refusal
    # of an unsafe shape rather than a loss of capability.
    (pack / "linked_settings.json").unlink()
    (pack / "hidden_dir").unlink()
    monkeypatch.setattr(sys, "argv", argv)
    assert scrub.main() == 0
    assert fake not in (pack / "normal.txt").read_text(encoding="utf-8")
