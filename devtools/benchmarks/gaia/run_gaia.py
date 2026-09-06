#!/usr/bin/env python3
"""Generate GAIA predictions with the reviewed Ouroboros CLI adapter.

The adapter intentionally keeps scoring official: it prepares a run root,
records exact settings/argv, and uses an inspect-evals solver wrapper that reads
Ouroboros's structured ``final_answer`` via ``--result-json-out``.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import (
    ACTIVE_MODEL_SLOT_KEYS,
    admit_benchmark_run,
    finalize_run_manifest,
    write_json,
)
from devtools.benchmarks.common.model_slots import (
    configured_subagents_snapshot,
    single_model_subagents_setting,
)
from devtools.benchmarks.common.run_roots import assert_outside_repo, run_root
from ouroboros.config import SETTINGS_DEFAULTS
from ouroboros.utils import truncate_review_artifact

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
_GAIA_PINNED_MODEL_KEYS = {
    "OUROBOROS_MODEL",
    "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_VISION",
    "OUROBOROS_MODEL_CONSCIOUSNESS",
    "OUROBOROS_MODEL_FALLBACKS",
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    "OUROBOROS_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
}
_PROVIDER_ENV_KEYS = {
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_COMPATIBLE_BASE_URL",
    "ANTHROPIC_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
    "MINIMAX_API_KEY",
    "MINIMAX_REGION",
    "DEEPSEEK_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_USER",
    "GIGACHAT_PASSWORD",
    "GITHUB_TOKEN",
}


def _free_port() -> int:
    """An OS-assigned free TCP port (bind :0 then release). Lets a dedicated bench server
    coexist with a running desktop app instead of colliding on the default ports."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _resolve_provider_keys(needed: set[str]) -> dict[str, str]:
    """Resolve REAL values for the needed provider env keys — from os.environ first, then
    the runtime ``data/settings.json`` (the bench template ``settings_base.json`` ships
    EMPTY placeholders; an empty value in the rendered settings makes the server's
    apply_settings_to_env POP the key and erase what the host env preserved)."""
    runtime: dict = {}
    try:
        runtime = json.loads((REPO.parent / "data" / "settings.json").read_text(encoding="utf-8"))
    except Exception:
        runtime = {}
    out: dict[str, str] = {}
    for k in needed:
        v = (os.environ.get(k) or "").strip() or str(runtime.get(k, "") or "").strip()
        if v:
            out[k] = v
    return out


def _credential_keys_for_model(model: str) -> set[str]:
    text = str(model or "").strip()  # tolerate "a, b"-split entries with leading spaces
    if text.startswith("openai::"):
        return {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
    if text.startswith("anthropic::"):
        return {"ANTHROPIC_API_KEY"}
    if text.startswith("cloudru::"):
        return {"CLOUDRU_FOUNDATION_MODELS_API_KEY", "CLOUDRU_FOUNDATION_MODELS_BASE_URL"}
    if text.startswith("gigachat::"):
        return {"GIGACHAT_CREDENTIALS", "GIGACHAT_USER", "GIGACHAT_PASSWORD"}
    if text.startswith("minimax::"):
        return {"MINIMAX_API_KEY", "MINIMAX_REGION"}
    if text.startswith("deepseek::"):
        return {"DEEPSEEK_API_KEY"}
    if text.startswith("openai-compatible::"):
        return {"OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"}
    return {"OPENROUTER_API_KEY"}


_WEBSEARCH_BACKEND_KEYS: dict[str, tuple[str, ...]] = {
    # Official OpenAI web_search needs an EMPTY base_url, so deliberately do NOT carry
    # OPENAI_BASE_URL (carrying it would route web_search off the official endpoint).
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    # 'auto' may try the OpenAI-first cascade; keep all three so it can reach any leg.
    "auto": ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"),
    "ddgs": (),  # pure retrieval, no provider key
}


def _sanitized_host_env(*models: str, websearch_backend: str = "") -> dict[str, str]:
    blocked = set(SETTINGS_DEFAULTS) | _PROVIDER_ENV_KEYS
    blocked.update(key for key in os.environ if key.startswith("USE_LOCAL_") or key.startswith("OUROBOROS_"))
    keep = {key: value for key, value in os.environ.items() if key not in blocked}
    # Preserve credential env for EVERY configured model-bearing knob (solve + vision +
    # review models can be different providers, e.g. sonnet main + gpt-4o vision) — else
    # a cross-provider route is written into settings but cannot authenticate.
    preserve: set[str] = set()
    for model in models:
        if str(model or "").strip():
            preserve.update(_credential_keys_for_model(model))
    # The pinned web_search backend may need a provider key unrelated to any model
    # (e.g. opus solve + 'openai' web backend) — preserve it too, else web_search fails.
    _ws = (websearch_backend or "").strip().lower()
    preserve.update(_WEBSEARCH_BACKEND_KEYS.get(_ws, ()))
    # Official OpenAI Responses web_search is disabled whenever OPENAI_BASE_URL is set,
    # so an 'openai' web pin must win over a same-provider model that carried the base_url.
    if _ws == "openai":
        preserve.discard("OPENAI_BASE_URL")
    for key in preserve:
        if os.environ.get(key):
            keep[key] = os.environ[key]
    return keep


def _render_run_settings(
    base_settings_path: pathlib.Path, solve_model: str, run_dir: pathlib.Path, *,
    vision_model: str = "", review_models: str = "", review_mode: str = "required",
    runtime_mode: str = "light", websearch_backend: str = "auto", or_provider: str = "",
    total_budget: float = 0.0, task_ceiling_sec: float = 0.0, host_service_port: int = 0,
    max_workers: int = 1, main_web_search: str = "off", main_web_search_engine: str = "auto",
    main_web_search_max_total_results: int = 10,
) -> pathlib.Path:
    settings = json.loads(base_settings_path.read_text(encoding="utf-8"))
    settings.pop("OUROBOROS_MODEL_HEAVY", None)
    settings.pop("USE_LOCAL_HEAVY", None)
    for key in ACTIVE_MODEL_SLOT_KEYS:
        if key.startswith("OUROBOROS_EFFORT_"):
            continue
        if key not in _GAIA_PINNED_MODEL_KEYS:
            continue
        if key == "OUROBOROS_REVIEW_MODELS":
            settings[key] = review_models or ",".join([solve_model] * 3)
        elif key:
            settings[key] = solve_model
    # One exact API actor preserves fixed-model methodology even when the product's
    # install defaults would otherwise add a Light scout or a session-backed row.
    settings["OUROBOROS_SUBAGENTS"] = single_model_subagents_setting(solve_model)
    # A fixed MAIN reasoner may route vision to a SEPARATE model (e.g. sonnet main +
    # gpt-4o vision, the HAL methodology) without breaking the fixed-model claim.
    if vision_model:
        settings["OUROBOROS_MODEL_VISION"] = vision_model
    settings["OUROBOROS_RUNTIME_MODE"] = runtime_mode
    settings["OUROBOROS_TASK_REVIEW_MODE"] = review_mode
    settings["OUROBOROS_MAX_WORKERS"] = max(1, int(max_workers or 1))
    settings["OUROBOROS_POST_TASK_EVOLUTION"] = "false"
    settings["OUROBOROS_WEBSEARCH_BACKEND"] = websearch_backend
    settings["OUROBOROS_MAIN_WEB_SEARCH"] = main_web_search
    settings["OUROBOROS_MAIN_WEB_SEARCH_ENGINE"] = main_web_search_engine
    settings["OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS"] = int(main_web_search_max_total_results or 10)
    settings["OUROBOROS_OR_PROVIDER"] = or_provider
    # Inject the REAL provider keys for every configured model + the pinned web backend, so
    # the rendered settings carry them (empty placeholders would be popped by the server's
    # apply_settings_to_env, erasing the host-env keys). Without this -> "No supported
    # provider configured." Keys land only in the isolated, gitignored run dir.
    needed: set[str] = set(_credential_keys_for_model(solve_model))
    if vision_model:
        needed |= _credential_keys_for_model(vision_model)
    for _m in (review_models or "").split(","):
        if _m.strip():
            needed |= _credential_keys_for_model(_m.strip())
    needed |= set(_WEBSEARCH_BACKEND_KEYS.get((websearch_backend or "").strip().lower(), ()))
    for _k, _v in _resolve_provider_keys(needed).items():
        settings[_k] = _v
    if (websearch_backend or "").strip().lower() == "openai":
        settings.pop("OPENAI_BASE_URL", None)  # official OpenAI web_search needs an EMPTY base_url
    # Dedicated Host-Service port: the default 8767 collides with a running desktop app and
    # crashes startup ("port 8767 busy") — auto-pick a free one for the bench server.
    if host_service_port and host_service_port > 0:
        settings["OUROBOROS_HOST_SERVICE_PORT"] = int(host_service_port)
    # Generous/0 budget: all samples share one server/data root, so a low shared
    # TOTAL_BUDGET would exhaust mid-run; 0 = unbounded (budget_remaining -> inf).
    settings["TOTAL_BUDGET"] = float(total_budget)
    # Server-side per-task ceiling: the solver's CLIENT timeout only kills the CLI; the
    # `--start` server-side task would otherwise ORPHAN and block the shared single-worker
    # queue for the next sample. A ceiling set BELOW the client timeout makes the server
    # reap its own task first, so the next sample finds a free server (no cascade).
    if task_ceiling_sec and task_ceiling_sec > 0:
        settings["OUROBOROS_TASK_ABS_CEILING_SEC"] = float(task_ceiling_sec)
        settings["OUROBOROS_TASK_IDLE_TIMEOUT_SEC"] = float(task_ceiling_sec)
    path = run_dir / "settings.json"
    write_json(path, settings)
    return path


def _settings_env(settings_path: pathlib.Path, solve_model: str, run_dir: pathlib.Path,
                  main_port: int = 0) -> dict[str, str]:
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    # Copy every scalar setting (web backend, OR_PROVIDER, TOTAL_BUDGET, runtime/review
    # mode) verbatim, then pin the model slots to solve_model — honoring the per-config
    # vision / review-models OVERRIDES already written into the settings file.
    env = {
        k: str(v)
        for k, v in settings.items()
        if k not in _PROVIDER_ENV_KEYS and v not in (None, "") and not isinstance(v, (list, dict))
    }
    for key in ACTIVE_MODEL_SLOT_KEYS:
        if key.startswith("OUROBOROS_EFFORT_") or key not in _GAIA_PINNED_MODEL_KEYS:
            continue
        if key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_MODEL_VISION") and settings.get(key):
            env[key] = str(settings[key])
        elif key == "OUROBOROS_REVIEW_MODELS":
            env[key] = ",".join([solve_model] * 3)
        elif key:
            env[key] = solve_model
    env["OUROBOROS_SETTINGS_PATH"] = str(settings_path)
    env["OUROBOROS_DATA_DIR"] = str(run_dir / "ouroboros_data")
    # Free main port (caller passes one) so the dedicated server doesn't collide with the
    # desktop app's 8765 and parallel configs never share a port; PID fallback if unset.
    port = int(main_port) if main_port and main_port > 0 else (19000 + (os.getpid() % 1000))
    env["OUROBOROS_SERVER_PORT"] = str(port)
    env["GAIA_OUROBOROS_URL"] = f"http://127.0.0.1:{port}"
    return env


def _requested_task_ids(args: argparse.Namespace) -> list[str]:
    """The denominator the audit sidecar records. When ``--sample-id`` selects an
    explicit subset (a failed-task rerun, mirrored into the Inspect argv by
    ``build_inspect_argv``), record those EXACT ids; otherwise fall back to the
    limit-derived level-index list. Keeping these in lockstep stops a sample-id
    rerun from writing a manifest that claims the first N level tasks instead of
    the actual selected samples."""
    sample_ids = str(getattr(args, "sample_id", "") or "").strip()
    if sample_ids:
        return [s.strip() for s in sample_ids.split(",") if s.strip()]
    return [f"{args.split}:level{args.level}:{idx}" for idx in range(1, int(args.limit) + 1)]


def _default_shared_files_root() -> pathlib.Path | None:
    for candidate in (
        pathlib.Path.home() / "Library" / "Caches" / "inspect_evals" / "gaia_dataset" / "GAIA" / "2023" / "validation",
        pathlib.Path.home() / ".cache" / "huggingface" / "datasets",
    ):
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve(strict=False)
    return None


def _admit_run(root: pathlib.Path, args: argparse.Namespace, planned_argv: list[str]) -> dict:
    """ADMISSION seam: persist the run manifest, THEN let the seed gate refuse.

    Deliberately takes no settings path and runs BEFORE ``_render_run_settings``: that
    renderer writes REAL provider keys into the run dir, so a refused run must be refused
    while the run dir still holds nothing but this manifest. The settings-derived fields are
    added afterwards by ``_augment_manifest``.
    """
    return admit_benchmark_run(
        root / "run_manifest.json",
        benchmark="gaia",
        run_root=root,
        repo_dir=REPO,
        requested_task_ids=_requested_task_ids(args),
        # The seed gate runs HERE, before the inspect subprocess spends anything (owner Q19=B).
        # GAIA needs no runtime attestation (owner Q10: the solver starts its own server from
        # this very checkout per sample -- there is no long-lived evolved volume to skew).
        require_clean=not getattr(args, "allow_dirty_seed", False),
        argv=planned_argv,
        dataset="inspect_evals/gaia",
        official_command=planned_argv,
        isolated_data_root=str(root / "ouroboros_data"),
        output_paths={"inspect_logs": str(root / "inspect_logs"), "samples": str(root / "samples")},
        harness={"solver": "inspect_solver/ouroboros_solver.py", "official_scorer": "gaia_scorer"},
        extra={
            "outcome": "started",
            "split": args.split,
            "level": args.level,
            "limit": args.limit,
            "solve_model": args.solve_model,
            "profile": str(getattr(args, "profile", "") or ""),
            "disable_tools": str(getattr(args, "disable_tools", "") or ""),
            "websearch_backend": str(getattr(args, "websearch_backend", "") or ""),
            "max_workers": int(getattr(args, "max_workers", 1) or 1),
            "main_web_search": str(getattr(args, "main_web_search", "off") or "off"),
            "main_web_search_engine": str(getattr(args, "main_web_search_engine", "auto") or "auto"),
            "worker_scaffold_disclosure": (
                "strict_baseline" if int(getattr(args, "max_workers", 1) or 1) == 1
                else "worker_pool_scaffold_change"
            ),
        },
    )


def _augment_manifest(manifest: dict, args: argparse.Namespace, root: pathlib.Path,
                      settings_path: pathlib.Path) -> None:
    """Add the facts that exist only once the run settings have been RENDERED, i.e. after
    admission. The finalization seam persists the augmented dict on every exit path.
    """
    manifest["model_slots"] = {
        k: v for k, v in _settings_env(settings_path, args.solve_model, root).items()
        if k in ACTIVE_MODEL_SLOT_KEYS
    }
    manifest["available_subagents"] = configured_subagents_snapshot(
        settings_path,
        env_overrides=False,
    )
    manifest.setdefault("extra", {})["image_input_mode"] = json.loads(
        settings_path.read_text(encoding="utf-8")
    ).get("OUROBOROS_IMAGE_INPUT_MODE", "")


# `inspect eval` has NO non-zero exit path for a task that RAISED: it records the failure in the
# log it writes and still returns 0. So the subprocess return code cannot tell a dead eval from a
# clean one, and trusting it alone is how the v6.81.0 GAIA smoke — every sample killed by
# `RuntimeError: Timed out executing setup command in sandbox`, zero samples scored — recorded
# `outcome="completed", exit_code=0`. An infra zero read as a result is exactly the fail-open this
# release exists to remove, so the outcome is decided from inspect's OWN log instead.
_INSPECT_LOG_SUFFIXES = (".json", ".eval")
# The recorded harness error is the EVIDENCE of that fail-open, so it is never silently clipped
# (BIBLE P1 / docs/DEVELOPMENT.md "No silent truncation"): a deep traceback from a sandbox that
# died is exactly the case where the tail is the informative part. Messages pass through whole up
# to this generous budget; beyond it the shared `truncate_review_artifact` seam appends its
# ⚠️ OMISSION NOTE (with the original length), and `error_log` always names the exact file holding
# the untouched message, so the reader reaches the whole thing without guessing.
_INSPECT_ERROR_DISCLOSED_LIMIT = 20000


def _inspect_log_files(log_dir: pathlib.Path) -> set[pathlib.Path]:
    """The inspect log files currently in ``log_dir``.

    Snapshotted before AND after the eval so a re-run into an existing ``--out-dir`` (the
    append-only resume flow) judges only the logs THIS eval wrote, never a predecessor's.
    """
    try:
        return {
            path.resolve(strict=False)
            for path in pathlib.Path(log_dir).iterdir()
            if path.is_file() and path.suffix in _INSPECT_LOG_SUFFIXES
        }
    except OSError:
        return set()


def read_inspect_eval_summary(log_paths: list[pathlib.Path]) -> dict[str, object]:
    """Summarize what the eval REALLY did, straight out of inspect's own logs.

    A sample counts as scored only when it carries a non-empty ``scores`` mapping, and that is
    what makes the honest distinction possible: a sample the official scorer marked incorrect IS
    scored (a genuine zero), while a sample that died before reaching the scorer is not (an infra
    zero). Tolerant by construction — an unreadable log is COUNTED, never swallowed, because
    "could not tell" must not read as "fine".
    """
    statuses: list[str] = []
    scored = sample_errors = unreadable = eval_errors = 0
    first_error = ""
    first_error_log = ""
    for path in log_paths:
        try:
            data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            unreadable += 1
            continue
        status = str(data.get("status") or "unknown")
        statuses.append(status)
        error = data.get("error") if isinstance(data.get("error"), dict) else None
        # inspect's terminal status is `success`; `error`/`cancelled`/`started` all mean the eval
        # did not finish, and only `error` additionally carries a message/traceback.
        if status != "success" or error is not None:
            eval_errors += 1
            if error is not None and not first_error:
                first_error = truncate_review_artifact(
                    str(error.get("message") or ""), limit=_INSPECT_ERROR_DISCLOSED_LIMIT
                )
                first_error_log = str(path)
        for sample in data.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            sample_scores = sample.get("scores")
            if isinstance(sample_scores, dict) and sample_scores:
                scored += 1
            if sample.get("error"):
                sample_errors += 1
    return {
        "logs": len(log_paths),
        "statuses": sorted(set(statuses)),
        "scored_samples": scored,
        "sample_errors": sample_errors,
        "unreadable_logs": unreadable,
        "eval_errors": eval_errors,
        "error": first_error,
        # The complete, untouched error (message AND traceback) lives here. Recorded even when
        # `error` was not truncated, because the traceback is never carried in the manifest.
        "error_log": first_error_log,
    }


def classify_inspect_outcome(summary: dict[str, object], returncode: int) -> tuple[str, int]:
    """The typed outcome AND the real exit status for an inspect eval.

    Keeps three different facts apart, because only the last of them is a result:

    ``eval_error``        the eval raised or never reached `success` — an INFRA zero: the
                          benchmark did not run, whatever the harness exit code claims.
    ``no_scored_samples`` the eval finished and scored NOTHING — also not a result.
    ``completed``         at least one sample reached the official scorer. A mean of 0.0 over
                          scored samples is a GENUINE zero and stays `completed`; collapsing it
                          into the failures above would hide real capability data.

    ``eval_status_unavailable`` is the fourth, deliberately fail-closed case: with no readable log
    there is no evidence of success to record, and unknown success is not success (the same rule
    the seed gate applies to unknown cleanliness).
    """
    code = int(returncode)
    if int(summary.get("eval_errors") or 0):
        return "eval_error", code or 1
    if code:
        return "harness_nonzero_exit", code
    if int(summary.get("unreadable_logs") or 0) or not int(summary.get("logs") or 0):
        return "eval_status_unavailable", 1
    if not int(summary.get("scored_samples") or 0):
        return "no_scored_samples", 1
    return "completed", 0


def build_inspect_argv(args: argparse.Namespace, run_dir: pathlib.Path) -> list[str]:
    solver = HERE / "inspect_solver" / "ouroboros_solver.py"
    argv = [
        sys.executable,
        "-m",
        "inspect_ai",
        "eval",
        "inspect_evals/gaia",
        "-T",
        f"subset={getattr(args, 'subset', '') or f'2023_level{int(args.level)}'}",
        "-T",
        f"split={args.split}",
        "--solver",
        f"{solver}@ouroboros_solver",
        "--max-samples",
        str(getattr(args, "max_samples", 1)),
        "--max-sandboxes",
        str(getattr(args, "max_sandboxes", 1)),
        "--log-format",
        "json",
        "--log-dir",
        str(run_dir / "inspect_logs"),
    ]
    sample_ids = str(getattr(args, "sample_id", "") or "").strip()
    if sample_ids:
        argv += ["--sample-id", sample_ids]
    else:
        argv += ["--limit", str(args.limit)]
    epochs = int(getattr(args, "epochs", 1) or 1)
    if epochs > 1:
        argv += ["--epochs", str(epochs)]
        reducer = str(getattr(args, "epochs_reducer", "") or "").strip()
        if reducer:
            argv += ["--epochs-reducer", reducer]
    return argv


def _apply_profile_defaults(args: argparse.Namespace) -> None:
    profile = str(getattr(args, "profile", "") or "strict_ddgs")
    explicit_disable = getattr(args, "disable_tools", None) is not None
    if profile == "web_off_baseline":
        if not explicit_disable:
            args.disable_tools = "web_search,claude_code_edit"
        if not getattr(args, "websearch_backend", ""):
            args.websearch_backend = "auto"
    elif profile == "quality_openrouter_web":
        if not explicit_disable:
            args.disable_tools = "web_search,claude_code_edit"
        args.main_web_search = "openrouter"
        args.main_web_search_engine = args.main_web_search_engine or "auto"
    else:  # strict_ddgs
        if not explicit_disable:
            args.disable_tools = "claude_code_edit"
        if not getattr(args, "websearch_backend", ""):
            args.websearch_backend = "ddgs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Ouroboros on GAIA via the official inspect_evals task.")
    parser.add_argument("--out-dir", default="", help="output run directory (outside repo/data)")
    parser.add_argument("--settings", default=str(HERE / "settings_base.json"))
    parser.add_argument("--solve-model", default="google/gemini-2.5-pro")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--subset", default="", help="override subset, e.g. 2023_all (all levels); default 2023_level{level}")
    parser.add_argument("--sample-id", default="", help="comma-separated sample ids to run (re-run a subset of failed tasks)")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1, help="inspect parallel samples")
    parser.add_argument("--max-sandboxes", type=int, default=1, help="inspect parallel sandboxes")
    # Per-config knobs (the 3 run configs are different invocations of these).
    parser.add_argument("--vision-model", default="", help="separate vision-slot model, e.g. openai/gpt-4o")
    parser.add_argument("--review-models", default="", help="comma-separated task-review models (default solve_model x3)")
    parser.add_argument("--review-mode", default="required", choices=["required", "auto", "off"])
    parser.add_argument("--runtime-mode", default="light")
    parser.add_argument("--profile", default="strict_ddgs", choices=["web_off_baseline", "strict_ddgs", "quality_openrouter_web"], help="GAIA scaffold profile. strict_ddgs keeps web_search on with pure retrieval; quality_openrouter_web uses main-model OpenRouter server web.")
    parser.add_argument("--websearch-backend", default="", help="auto|ddgs|openai|openrouter|anthropic (empty = profile default)")
    parser.add_argument("--main-web-search", default="off", choices=["off", "openrouter"], help="Opt-in main-loop web search server tool. `openrouter` injects openrouter:web_search into the solve-model request.")
    parser.add_argument("--main-web-search-engine", default="auto", help="OpenRouter web_search engine for --main-web-search=openrouter")
    parser.add_argument("--main-web-search-max-total-results", type=int, default=10)
    parser.add_argument("--or-provider", default="", help="''|resilience|repro|JSON")
    parser.add_argument("--total-budget", type=float, default=0.0, help="TOTAL_BUDGET (0 = unbounded)")
    parser.add_argument("--disable-tools", default=None, help="comma-separated tools to disable (default follows --profile)")
    parser.add_argument("--shared-files-root", default="", help="host dir holding GAIA /shared_files attachments (HF cache validation dir)")
    parser.add_argument("--user-files-root", default="", help="scratch home for the user_files sandbox (security isolation)")
    parser.add_argument("--sample-timeout-sec", type=float, default=7200.0)
    parser.add_argument("--max-workers", type=int, default=4, help="Ouroboros worker pool size inside the GAIA run (v6.55.0 default 4 — owner decision #16: the old strict 1 starved subagent decomposition; pass 1 explicitly for a strict-baseline ablation)")
    parser.add_argument("--epochs", type=int, default=1, help="pass@N best-of-N epochs (inspect)")
    parser.add_argument("--epochs-reducer", default="", help="inspect epochs reducer, e.g. pass_at_1 / mode")
    parser.add_argument("--dry-run", action="store_true", help="write manifest and planned argv without spending")
    parser.add_argument(
        "--allow-dirty-seed",
        action="store_true",
        help="record and proceed with an unclean/unidentifiable seed checkout instead of refusing",
    )
    args = parser.parse_args(argv)
    _apply_profile_defaults(args)

    out = pathlib.Path(args.out_dir).expanduser() if args.out_dir else run_root("gaia")
    out = assert_outside_repo(out, REPO)
    planned = build_inspect_argv(args, out)
    base_settings_path = pathlib.Path(args.settings).expanduser().resolve(strict=False)
    manifest_path = out / "run_manifest.json"
    # Admission is the outermost REFUSAL point, not the outermost side effect: `ensure_outside_repo`
    # above already created `out` (it mkdirs). Everything else above it is argument parsing and pure
    # path derivation, so a refused run leaves that directory holding nothing but the persisted
    # refusal -- no rendered settings.json carrying live provider keys, no bound ports, no subprocess.
    manifest = _admit_run(out, args, planned)
    with finalize_run_manifest(manifest_path, manifest) as final:
        if args.main_web_search == "openrouter":
            route_refusal = ""
            if "::" in str(args.solve_model):
                route_refusal = (
                    "--main-web-search=openrouter requires an OpenRouter-routed tool-calling solve model "
                    "(provider/model, not provider::model). Use --profile strict_ddgs or "
                    "--main-web-search=off if this route cannot use OpenRouter server tools."
                )
            elif not _resolve_provider_keys({"OPENROUTER_API_KEY"}).get("OPENROUTER_API_KEY"):
                route_refusal = (
                    "--main-web-search=openrouter requires OPENROUTER_API_KEY for the solve-model route; "
                    "the adapter assumes OpenRouter server-tool support for routed tool-calling models."
                )
            if route_refusal:
                # A config refusal is a REFUSAL, not a crash: name it in the record.
                final.update({"outcome": "refused", "exit_code": 1,
                              "refusal": {"stage": "main_web_search_route", "exit_code": 1}})
                raise SystemExit(route_refusal)
        # Auto-pick distinct free ports so the dedicated bench server coexists with a running
        # desktop app (no 8765 main / 8767 Host-Service collision) and parallel configs don't clash.
        main_port = _free_port()
        host_service_port = _free_port()
        settings_path = _render_run_settings(
            base_settings_path, args.solve_model, out,
            vision_model=args.vision_model, review_models=args.review_models, review_mode=args.review_mode,
            runtime_mode=args.runtime_mode, websearch_backend=args.websearch_backend,
            or_provider=args.or_provider, total_budget=args.total_budget, host_service_port=host_service_port,
            max_workers=args.max_workers,
            main_web_search=args.main_web_search,
            main_web_search_engine=args.main_web_search_engine,
            main_web_search_max_total_results=args.main_web_search_max_total_results,
            # Server reaps its own task well before the solver's client timeout — the buffer
            # covers BOTH the finalization grace (~120s default) AND margin, so the server is
            # idle again before the client gives up (no orphaned task blocking the next sample).
            task_ceiling_sec=max(60.0, float(args.sample_timeout_sec) - 240.0),
        )
        _augment_manifest(manifest, args, out, settings_path)
        if args.dry_run:
            print(json.dumps({"run_root": str(out), "planned_argv": planned}, indent=2))
            final["outcome"] = "dry_run"
            return 0
        _review_models = [m.strip() for m in (args.review_models or "").split(",") if m.strip()]
        env = {
            **_sanitized_host_env(args.solve_model, args.vision_model, *_review_models,
                                  websearch_backend=args.websearch_backend),
            **_settings_env(settings_path, args.solve_model, out, main_port=main_port),
            "GAIA_OUROBOROS_RUN_ROOT": str(out),
            "GAIA_OUROBOROS_SETTINGS": str(settings_path),
            "GAIA_OUROBOROS_SOLVE_MODEL": args.solve_model,
            # Solver-side knobs (run_gaia strips host OUROBOROS_* env, so pass GAIA_* through).
            "GAIA_DISABLE_TOOLS": args.disable_tools,
            "GAIA_SAMPLE_TIMEOUT_SEC": str(args.sample_timeout_sec),
        }
        shared_files_root = (
            pathlib.Path(args.shared_files_root).expanduser().resolve(strict=False)
            if args.shared_files_root else _default_shared_files_root()
        )
        if shared_files_root:
            env["GAIA_SHARED_FILES_ROOT"] = str(shared_files_root)
        scratch = (
            pathlib.Path(args.user_files_root).expanduser().resolve(strict=False)
            if args.user_files_root else (out / "user_files").resolve(strict=False)
        )
        scratch.mkdir(parents=True, exist_ok=True)
        env["OUROBOROS_USER_FILES_ROOT"] = str(scratch)
        # Keep the unnamed-deliverables container INSIDE the jail too: otherwise a bare
        # write_file(root='user_files', path='answer.txt') resolves to ~/Ouroboros/
        # Deliverables (outside the scratch home) and is blocked as outside_home.
        deliverables = scratch / "Deliverables"
        deliverables.mkdir(parents=True, exist_ok=True)
        env["OUROBOROS_DELIVERABLES_ROOT"] = str(deliverables)
        log_dir = out / "inspect_logs"
        logs_before = _inspect_log_files(log_dir)
        code = int(subprocess.run(planned, env=env).returncode)
        # THE STATUS COMES FROM THE EVAL, NOT FROM ITS EXIT CODE. `inspect eval` returns 0 for an
        # eval that raised, so `code` alone would let a dead run record `completed` (see the
        # comment on `_INSPECT_LOG_SUFFIXES`). The raw harness code is kept alongside the verdict
        # so the two are never confused for each other in an audit.
        summary = read_inspect_eval_summary(sorted(_inspect_log_files(log_dir) - logs_before))
        outcome, exit_code = classify_inspect_outcome(summary, code)
        final.update({"outcome": outcome, "exit_code": exit_code,
                      "harness_exit_code": code, "inspect_eval": summary})
        if outcome != "completed":
            print(f"[run_gaia] inspect eval did not produce a result: outcome={outcome} "
                  f"harness_exit_code={code} inspect_eval={json.dumps(summary)}", file=sys.stderr)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
