"""GAIA: the adapter it renders, the solver it runs and the leakage it must not enjoy.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
settings and solver wiring, the sanitized provider environment, the attachment staging and
its traversal refusals, the anti-lookup and epistemic instructions every solver carries,
the leakage audit that adjusts the score, and the run status the launcher may record.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


from tests._devtools_benchmarks_shared import (
    REPO_ROOT,
    _git_commit_all,
    _git_repo,
)
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


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

def test_gaia_events_serializer_carries_web_search_sources():
    src = (REPO_ROOT / "supervisor/events_budget.py").read_text("utf-8")
    assert "web_search_sources" in src

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
