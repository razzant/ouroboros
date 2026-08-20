#!/usr/bin/env python3
"""External OSWorld step-loop adapter backed by local Ouroboros.

Unlike ``run_installed_agent.py``, this runner does not install Ouroboros inside
the VM. It keeps the official OSWorld rhythm:

    observe VM -> ask Ouroboros for next action(s) -> env.step(action) -> repeat

Every action returned by Ouroboros is passed through ``env.step(...)`` and is
therefore visible in OSWorld's normal trajectory/action history. Screenshots are
saved under ``data/uploads`` so Ouroboros can inspect them with ``vlm_query``.

Alignment target: OSWorld 2.0 (``ALIGNED_UPSTREAM`` below). The runner mirrors
the official per-example artifact contract consumed by upstream
``show_result.py`` / ``lib_run_single.py``:

    <result_dir>/<action_space>/<observation_type>/<model>/<domain>/<example_id>/
        traj.jsonl          # official per-step rows (step_num, action, response,
                            # reward, done, info, screenshot_file, ...)
        step_<n>_<ts>.png   # post-action screenshot per step (official naming)
        result.txt          # final env.evaluate() score (scoring authority)
        result.json         # full dict result when evaluate() returns one

Not implemented (be honest when comparing to official 2.0 numbers): inline
checkpoint evaluations (``--checkpoint_eval_mode inline --checkpoint_steps
150,300``), multi-phase tasks, the human-in-the-loop user simulator
(``ASK_USER`` rows), ``recording.mp4``, and cloud providers (aws/azure/gcp).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import (
    BenchmarkAdmissionRefused,
    RuntimeAttestationRefused,
    admit_benchmark_run,
    finalize_run_manifest,
    runtime_attestation,
    write_json,
)
from devtools.benchmarks.common.result_index import append_result_index, task_result_row
from devtools.benchmarks.common.run_roots import (  # noqa: F401 - repo_root_from_devtools stays re-exported
    assert_outside_repo,
    ensure_outside_repo,
    repo_root_from_devtools,
)
from devtools.benchmarks.osworld.step_agent_actions import (  # noqa: F401 - re-exported module surface
    SPECIAL_ACTIONS,
    _click_action,
    _hotkey_action,
    _json_from_text,
    _normalize_structured_action,
    _shell_action,
    _type_action,
    _wait_action,
)
from devtools.benchmarks.osworld.step_agent_claims import (  # noqa: F401 - re-exported module surface
    UNCONFIRMED_SCORE_SUFFIX,
    ClaimDirNotConfined,
    ClaimMarkerNotDurable,
    acquire_task_claim,
    claim_stale_sec,
    confined_claims_dir,
    mark_task_scored,
    record_unconfirmed_score,
    release_task_claim,
    scored_claim_state,
    task_already_scored,
    task_claim_key,
)
from devtools.benchmarks.osworld.step_agent_common import (  # noqa: F401 - re-exported module surface
    PreflightConfig,
    StepAgentConfig,
    TaskRecordConfig,
    _http_json,
    _safe_slug,
)
from devtools.benchmarks.osworld.step_agent_env import (  # noqa: F401 - re-exported module surface
    ALIGNED_UPSTREAM,
    SUPPORTED_PROVIDERS,
    VMWARE_FUSION_PATHS,
    _DEFAULT_DESKTOP_PORT,
    _LOOPBACK_HOSTS,
    _ensure_vmrun_on_path,
    _install_optional_dependency_stubs,
    _is_default_desktop_server,
    _teardown_partial_desktop_env,
    construct_desktop_env,
    osworld_checkout_info,
    provider_preflight_failures,
)
from devtools.benchmarks.osworld.step_agent_policy import (  # noqa: F401 - re-exported module surface
    OuroborosStepAgent,
    _initial_observation_with_retries,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent

DEFAULT_OSWORLD_ROOT = os.environ.get("OSWORLD_ROOT", str(_WORKSPACE_ROOT / "OSWorld"))
DEFAULT_VM = os.environ.get("OSWORLD_VM", str(Path(DEFAULT_OSWORLD_ROOT) / "vmware_vm_data" / "Ubuntu0" / "Ubuntu0.vmx"))
DEFAULT_TASK = "evaluation_examples/examples/os/f9be0997-4b7c-45c5-b05c-4612b44a6118.json"
DEFAULT_REPO = str(_REPO_ROOT)
DEFAULT_DATA = os.environ.get("OUROBOROS_OSWORLD_DATA_DIR", str(_WORKSPACE_ROOT / "bench_runs" / "osworld_data"))
DEFAULT_SETTINGS = os.environ.get("OUROBOROS_SETTINGS_PATH", str(_WORKSPACE_ROOT / "data" / "settings.json"))
DEFAULT_OUROBOROS_BIN = os.environ.get("OUROBOROS_BIN", str(_REPO_ROOT / ".venv" / "bin" / "ouroboros"))


def _persist_evaluation_result(result: Any, run_dir: Path) -> float:
    """Persist ``env.evaluate()`` output the way official lib_run_single.py does.

    Upstream (OSWorld-V2 ``_persist_evaluation_result``): the result may be a
    float (legacy) or a dict whose ``score`` field is the canonical float; dict
    results are additionally written to ``result.json``. ``result.txt`` is what
    official ``show_result.py`` scores.
    """

    if isinstance(result, dict):
        try:
            score = float(result.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    else:
        score = float(result)
    (run_dir / "result.txt").write_text(f"{score}\n", encoding="utf-8")
    return score


# --------------------------------------------------------------------------- #
# Shared OSWorld launcher helpers (imported by run_cu_bridge_agent.py, which
# already reuses this module for the live-server guard and checkout probe).
# --------------------------------------------------------------------------- #

def amend_task_manifest(base_manifest: dict[str, Any], *, output_paths: dict[str, Any],
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the per-task manifest: the ONE early-built manifest plus late facts.

    The clean-seed/provenance gate lives in ``benchmark_run_manifest`` itself, so the
    manifest has to be built BEFORE the first paid step; the outcome-time facts
    (output paths, counters, evaluator status) are merged in here instead of building
    a second manifest after the money is spent.
    """
    manifest = dict(base_manifest)
    manifest["output_paths"] = {**(manifest.get("output_paths") or {}), **output_paths}
    manifest["extra"] = {**(manifest.get("extra") or {}), **(extra or {})}
    return manifest


def admit_step_loop_run(manifest_path: Path, *, result_root: Path, repo_dir: Path,
                        settings_path: Path, example_id: str,
                        require_clean: bool) -> dict[str, Any]:
    """ADMISSION seam for the step loop: build the manifest, PERSIST it, then enforce.

    Routes through the shared `admit_benchmark_run` (v6.75.0) rather than pairing the bare
    manifest builder with a later `write_json()`: the complete gate/refusal
    payload reaches disk BEFORE a refusal propagates, so a refused run is exactly as
    auditable as an admitted one. A refusal raises `BenchmarkAdmissionRefused` carrying
    that same manifest, which the caller records and finalizes.
    """
    return admit_benchmark_run(
        manifest_path,
        benchmark="osworld",
        run_root=result_root,
        repo_dir=repo_dir,
        requested_task_ids=[example_id],
        dataset="OSWorld",
        settings_path=settings_path,
        require_clean=require_clean,
        harness={
            "adapter": "external_step_loop",
            "official_actions": True,
            "memory_mode": "empty_per_ouroboros_step",
            "action_space": "pyautogui",
            "aligned_upstream": dict(ALIGNED_UPSTREAM),
        },
    )


def _preflight(config: PreflightConfig) -> dict[str, Any]:
    failures: list[str] = []
    details: dict[str, Any] = {}
    checkout = osworld_checkout_info(config.osworld_root)
    details["osworld_checkout"] = checkout
    if not checkout["exists"]:
        failures.append(f"OSWorld checkout not found: {config.osworld_root}")
    if not (config.osworld_root / "evaluation_examples").exists():
        failures.append(f"OSWorld checkout shape not recognized: {config.osworld_root}")
    if checkout["exists"] and not checkout["has_desktop_env"]:
        failures.append(
            f"desktop_env package missing in OSWorld checkout (expected desktop_env/desktop_env.py): {config.osworld_root}"
        )
    if checkout["exists"] and not checkout["matches_aligned_commit"]:
        details["upstream_pin_warning"] = (
            f"checkout commit {checkout['git_commit'] or '<unknown>'} differs from the aligned "
            f"OSWorld 2.0 pin {ALIGNED_UPSTREAM['commit']} ({ALIGNED_UPSTREAM['repo']}); "
            "results are only comparable against the pinned protocol"
        )
    if not config.task_path.is_file():
        failures.append(f"task JSON not found: {config.task_path}")
    failures.extend(provider_preflight_failures(config.provider_name, config.path_to_vm))
    if not config.repo_dir.is_dir() or not (config.repo_dir / "VERSION").exists():
        failures.append(f"Ouroboros repo shape not recognized: {config.repo_dir}")
    if not config.settings_path.is_file():
        failures.append(f"settings.json not found: {config.settings_path}")
    else:
        try:
            settings = json.loads(config.settings_path.read_text(encoding="utf-8"))
            selected_model = str(config.model or settings.get("OUROBOROS_MODEL") or "")
            details["model"] = selected_model
            from ouroboros.provider_models import PROVIDER_ENV_KEYS, provider_for_model

            provider = provider_for_model(selected_model)
            env_key = PROVIDER_ENV_KEYS.get(provider, "OPENROUTER_API_KEY")
            # The provider key lives on the TARGET server (steps submit over
            # `ouroboros run --url`); the CLIENT settings/env cannot prove the
            # server has it, and /api/settings masks it. So a missing client-side
            # key is a WARNING, not a preflight pass/fail — the server scaffold
            # check below is the authoritative "is the executing server usable"
            # gate (adversarial review r1: client-side key check was misleading).
            if not str(os.environ.get(env_key) or settings.get(env_key) or "").strip():
                details["client_provider_key_absent"] = (
                    f"{env_key} not set client-side for provider {provider}; the TARGET server "
                    "must carry it (not verifiable here — /api/settings masks secrets)."
                )
        except Exception as exc:
            failures.append(f"settings.json unreadable: {type(exc).__name__}: {exc}")
    try:
        ensure_outside_repo(config.data_dir, config.repo_dir)
    except Exception as exc:
        failures.append(f"data dir must be outside repo/live data: {exc}")
    try:
        uploads = config.data_dir / "uploads" / "osworld" / "_preflight"
        uploads.mkdir(parents=True, exist_ok=True)
        probe = uploads / "write_probe.txt"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        failures.append(f"data/uploads not writable: {type(exc).__name__}: {exc}")
    try:
        state = _http_json(config.ouroboros_url.rstrip("/") + "/api/state", timeout=5)
        details["ouroboros_state"] = {
            "supervisor_ready": state.get("supervisor_ready"),
            "runtime_mode": state.get("runtime_mode"),
        }
        if not state.get("supervisor_ready"):
            failures.append("Ouroboros server reachable but supervisor_ready is false")
        # The adapter submits over the gateway (`ouroboros run --url`), so env
        # vars in the CLI subprocess can NOT configure the executing server.
        # The disclosed scaffold defaults are only real if the TARGET SERVER
        # already runs them — verify its effective settings and fail loudly on
        # drift (start the isolated server from osworld/settings_base.json).
        server_settings = _http_json(config.ouroboros_url.rstrip("/") + "/api/settings", timeout=5)
        expected = {
            "OUROBOROS_RUNTIME_MODE": "pro",
            "OUROBOROS_SAFETY_MODE": "light",
            "OUROBOROS_MAX_WORKERS": 4,
            # The scaffold's blocking review lane is only real if the TARGET server
            # runs it — CLI env cannot configure the executing server, so the
            # preflight must verify it (adversarial review r2 #6).
            "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
        }
        mismatches = []
        for key, want in expected.items():
            got = server_settings.get(key)
            if str(got).strip().lower() != str(want).strip().lower():
                mismatches.append(f"{key}: server={got!r} expected={want!r}")
        if config.model:
            server_model = str(server_settings.get("OUROBOROS_MODEL") or "")
            if server_model != config.model:
                mismatches.append(f"OUROBOROS_MODEL: server={server_model!r} expected={config.model!r}")
        details["server_scaffold_settings"] = {
            k: server_settings.get(k)
            for k in ("OUROBOROS_RUNTIME_MODE", "OUROBOROS_SAFETY_MODE", "OUROBOROS_MAX_WORKERS", "OUROBOROS_MODEL")
        }
        if mismatches:
            message = (
                "target server settings do not match the disclosed OSWorld scaffold "
                "(render devtools/benchmarks/osworld/settings_base.json into an isolated "
                "server and point --ouroboros-url at it): " + "; ".join(mismatches)
            )
            if config.allow_scaffold_mismatch:
                details["scaffold_mismatch_allowed"] = mismatches
            else:
                failures.append(message)
    except Exception as exc:
        failures.append(f"Ouroboros server not reachable: {type(exc).__name__}: {exc}")
    # Owner Q9=A+B / Q10: this launcher attaches to a live server URL, so its own admission
    # path attests WHAT is running (the HTTP `runtime_version`) against the checkout it was
    # started from (local HEAD + VERSION) before a single paid step. The shared helper fails
    # CLOSED by raising; translate that into a typed preflight failure so the caller still
    # gets `blocked/preflight_failed` records instead of a bare traceback.
    #
    # A refusal CARRIES the record it built (`RuntimeAttestationRefused.attestation`): the exact
    # typed reason plus `runtime_version`, `repo_head` and `repo_version`. Keeping only the
    # message threw those away at the moment they matter most — the manifest then said
    # `runtime_attestation_failed` and nothing about WHICH runtime disagreed with WHICH commit.
    try:
        details["runtime_attestation"] = runtime_attestation(config.ouroboros_url, config.repo_dir)
    except RuntimeAttestationRefused as exc:
        details["runtime_attestation"] = dict(exc.attestation)
        failures.append(f"runtime attestation failed reason={exc.attestation.get('reason') or ''}: {exc}")
    except RuntimeError as exc:
        # No record to keep (the helper raised before building one).
        details["runtime_attestation"] = {"ok": False, "reason": "runtime_attestation_failed",
                                          "error": f"{type(exc).__name__}: {exc}"}
        failures.append(f"runtime attestation failed: {exc}")
    try:
        ensure_outside_repo(config.result_root, config.repo_dir)
    except Exception as exc:
        failures.append(str(exc))
    return {"ok": not failures, "failures": failures, "details": details}


def _write_task_records(config: TaskRecordConfig) -> dict[str, Any]:
    details = dict(config.extra or {})
    outcome = {
        "ok": config.status == "completed",
        "task_id": config.example_id,
        "domain": config.domain,
        "reward": config.reward,
        "steps": config.steps,
        "status": config.status,
        "reason_code": config.reason_code,
        "error": config.error,
        "result_dir": str(config.run_dir),
        **details,
    }
    write_json(config.run_dir / "task_outcome.json", outcome)
    # Amend the ADMITTED manifest IN PLACE so the finalization seam writes one dict that
    # carries both these late facts and the run's terminal outcome. Rebuilding a manifest
    # here (the old fallback, with `require_clean=False`) recorded a WAIVED gate on the very
    # path where the real gate had refused the run.
    config.base_manifest.update(amend_task_manifest(
        config.base_manifest,
        output_paths={
            "task_outcome": str(config.run_dir / "task_outcome.json"),
            "traj": str(config.run_dir / "traj.jsonl"),
            "task_run_manifest": str(config.run_dir / "task_run_manifest.json"),
        },
        extra=details,
    ))
    # The manifest itself is NOT published here. Every caller runs INSIDE an active
    # `finalize_run_manifest` over this very path, and the seam merges the terminal
    # outcome/exit_code/refusal into this same retained dict only when its context EXITS —
    # so a write here publishes a pre-merge record (for a refusal, the admission seam's
    # generic payload saying exit_code 1 while the process will exit 2) that a concurrent
    # reader can observe and an interruption makes durable. The seam writes this path on
    # every exit path already, so the write was pure duplication with a window attached.
    # Enforced for the whole family by launcher_audit Invariant C.
    append_result_index(
        config.result_root,
        task_result_row(
            benchmark="osworld",
            instance_id=config.example_id,
            status=config.status,
            reason_code=config.reason_code,
            official_eval_status="completed" if config.reward is not None else "not_run",
            output_paths={
                "task_outcome": str(config.run_dir / "task_outcome.json"),
                "task_run_manifest": str(config.run_dir / "task_run_manifest.json"),
                "traj": str(config.run_dir / "traj.jsonl"),
            },
            error=config.error,
            details={"domain": config.domain, "reward": config.reward, "steps": config.steps, **details},
        ),
    )
    return outcome


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osworld-root", default=DEFAULT_OSWORLD_ROOT)
    parser.add_argument("--provider_name", default="vmware", help=f"VM provider; this adapter supports: {', '.join(SUPPORTED_PROVIDERS)}")
    parser.add_argument("--path_to_vm", default=DEFAULT_VM)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--result_dir", default="results/ouroboros_step_agent")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "screenshot_a11y_tree"],
        default="screenshot_a11y_tree",
        help="official observation_type path segment; also controls require_a11y_tree",
    )
    parser.add_argument("--repo-dir", default=DEFAULT_REPO)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--settings-path", default=DEFAULT_SETTINGS)
    parser.add_argument("--ouroboros-bin", default=DEFAULT_OUROBOROS_BIN)
    parser.add_argument("--ouroboros-url", default="http://127.0.0.1:8765",
                        help="Ouroboros server URL. The default is the LIVE desktop server; a real bench "
                             "run must point at an isolated server (see --allow-live-server).")
    parser.add_argument("--allow-scaffold-mismatch", action="store_true",
                        help="explicit ablation opt-in: run even when the target server's effective "
                             "settings differ from the disclosed scaffold defaults (recorded in the "
                             "preflight details; the run is then NOT comparable to default-scaffold runs).")
    parser.add_argument("--allow-live-server", action="store_true",
                        help="explicit opt-in to run against the default desktop server URL "
                             "(http://127.0.0.1:8765). Without it, real runs refuse the default: every "
                             "step submits tasks/screenshots into whichever server answers there, and a "
                             "LIVE data/ root must never absorb bench writes. Start an isolated server "
                             "on another port instead.")
    parser.add_argument("--model", default="anthropic/claude-opus-4-7")
    # OSWorld 2.0 protocol budget (official launch scripts pass 500; the paper
    # reports 150/300/500-step curves). The old 15/50 conventions are legacy.
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--disable-tools",
        default="claude_code_edit",
        help="comma-separated Ouroboros tools withheld per step (bench scaffold default: claude_code_edit)",
    )
    parser.add_argument("--step_timeout_sec", type=int, default=240)
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)
    parser.add_argument("--wait_after_reset_sec", type=float, default=8.0)
    parser.add_argument("--startup_timeout_sec", type=int, default=600)
    parser.add_argument("--reset_retries", type=int, default=3)
    parser.add_argument("--startup_retry_sleep_sec", type=float, default=5.0)
    parser.add_argument("--max_obs_chars", type=int, default=12000)
    parser.add_argument("--screenshot-check-only", action="store_true")
    parser.add_argument("--show-vm", action="store_true")
    parser.add_argument("--allow-dirty-seed", action="store_true",
                        help="run even when this Ouroboros checkout is dirty or its git identity is "
                             "unreadable (default: fail closed before the VM boots). Recorded in the "
                             "manifest; a dirty seed makes the run's provenance irreproducible.")
    return parser


def main() -> int:
    # NOTHING but argument parsing and pure local derivation until `admit_step_loop_run`
    # below. `_ensure_vmrun_on_path()` probes the filesystem for `vmrun` and mutates $PATH,
    # `_install_optional_dependency_stubs()` mutates `sys.modules`, and the `sys.path` insert
    # is process state: all three moved into `_run_step_loop`, i.e. after the run is on disk.
    args = build_arg_parser().parse_args()

    osworld_root = Path(args.osworld_root).expanduser().resolve(strict=False)
    if _is_default_desktop_server(args.ouroboros_url) and not args.allow_live_server:
        raise SystemExit(
            "refusing the default desktop server port (8765 on a loopback host): bench steps would "
            "write tasks/screenshots into the LIVE Ouroboros data root. Point --ouroboros-url at an "
            "isolated server (fresh OUROBOROS_DATA_DIR, non-default port), or pass --allow-live-server "
            "for an explicit local-debug run."
        )
    task_path = Path(args.task).expanduser()
    if not task_path.is_absolute():
        task_path = osworld_root / task_path
    domain = task_path.parent.name
    example_id = task_path.stem
    result_root = Path(args.result_dir).expanduser()
    if not result_root.is_absolute():
        result_root = osworld_root / result_root
    # ASSERT, not ensure (see run_cu_bridge_agent): no directory before the manifest.
    result_root = assert_outside_repo(result_root, Path(args.repo_dir).expanduser().resolve(strict=False))
    # Official example dir layout consumed by upstream show_result.py:
    # <result_dir>/<action_space>/<observation_type>/<model>/<domain>/<example_id>
    run_dir = (
        result_root
        / "pyautogui"
        / args.observation_type
        / _safe_slug(args.model or "default")
        / domain
        / example_id
    )
    # NO mkdir here: ADMISSION IS THE OUTER BOUNDARY (v6.75.0), so nothing touches the
    # filesystem before the manifest is persisted. The atomic manifest write creates the tree.
    manifest_path = run_dir / "task_run_manifest.json"

    repo_dir = Path(args.repo_dir).expanduser().resolve(strict=False)
    data_dir = Path(args.data_dir).expanduser().resolve(strict=False)
    settings_path = Path(args.settings_path).expanduser().resolve(strict=False)
    # ADMISSION: build the manifest, WRITE it, then let the seed gate enforce — before the
    # preflight, the VM boot and every paid step. `_write_task_records` amends this same dict
    # and `finalize_run_manifest` records how the run ended on EVERY exit path.
    #
    # The gate fails CLOSED (owner Q19). Nothing has been spent at this point, so report it as
    # this launcher's own typed refusal — `blocked/seed_gate_failed` with a ledger row —
    # instead of a bare traceback, over the manifest `admit_benchmark_run` already persisted.
    try:
        base_manifest = admit_step_loop_run(
            manifest_path,
            result_root=result_root, repo_dir=repo_dir, settings_path=settings_path,
            example_id=example_id, require_clean=not args.allow_dirty_seed,
        )
    except BenchmarkAdmissionRefused as exc:
        error = f"{type(exc).__name__}: {exc}"
        refused = exc.manifest
        # exit_code 2 is what this launcher REALLY exits with for a blocked run; the seam's
        # generic refusal payload says 1, and a record that disagrees with the process status
        # is the exact class of bug the parity test exists to catch.
        with finalize_run_manifest(manifest_path, refused, outcome="refused", exit_code=2) as final:
            final["refusal"] = {**((refused.get("extra") or {}).get("refusal") or {}),
                                "exit_code": 2}
            outcome = _write_task_records(TaskRecordConfig(
                run_dir=run_dir,
                result_root=result_root,
                repo_dir=repo_dir,
                settings_path=settings_path,
                example_id=example_id,
                domain=domain,
                reward=None,
                steps=0,
                status="blocked",
                reason_code="seed_gate_failed",
                base_manifest=refused,
                error=error,
                extra={"allow_dirty_seed": bool(args.allow_dirty_seed),
                       "seed_gate_error": error},
            ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 2
    base_manifest["extra"] = {**(base_manifest.get("extra") or {}),
                              "allow_dirty_seed": bool(args.allow_dirty_seed)}
    with finalize_run_manifest(manifest_path, base_manifest) as final:
        return _run_step_loop(args, final, StepLoopPaths(
            run_dir=run_dir, result_root=result_root, repo_dir=repo_dir, data_dir=data_dir,
            settings_path=settings_path, osworld_root=osworld_root, task_path=task_path,
            domain=domain, example_id=example_id, base_manifest=base_manifest,
        ))


@dataclass
class StepLoopPaths:
    """Resolved paths + the ADMITTED manifest handed to the post-admission body."""

    run_dir: Path
    result_root: Path
    repo_dir: Path
    data_dir: Path
    settings_path: Path
    osworld_root: Path
    task_path: Path
    domain: str
    example_id: str
    base_manifest: dict[str, Any]


def _run_step_loop(args: argparse.Namespace, final: dict[str, Any],
                   paths: StepLoopPaths) -> int:
    """Everything AFTER admission: preflight, VM boot, the step loop, official evaluate.

    Split out of `main()` so the admission function's pre-admission statements stay trivially
    auditable (the seam meta-test walks them with `ast` and denies every filesystem/docker/
    subprocess/network call there). `final` is the finalization seam's mutable record.
    """
    run_dir, result_root = paths.run_dir, paths.result_root
    repo_dir, data_dir, settings_path = paths.repo_dir, paths.data_dir, paths.settings_path
    domain, example_id = paths.domain, paths.example_id
    base_manifest = paths.base_manifest
    osworld_root, task_path = paths.osworld_root, paths.task_path
    # Process/environment preparation belongs HERE, after the persisted admission boundary:
    # each of these probes the filesystem or mutates process state, so a refusal in any of
    # them must land on a run that already has a durable record.
    _ensure_vmrun_on_path()
    _install_optional_dependency_stubs()
    sys.path.insert(0, str(osworld_root))
    preflight = _preflight(PreflightConfig(
        osworld_root=osworld_root,
        task_path=task_path,
        path_to_vm=args.path_to_vm,
        repo_dir=repo_dir,
        data_dir=data_dir,
        settings_path=settings_path,
        result_root=result_root,
        ouroboros_url=args.ouroboros_url,
        model=args.model,
        provider_name=args.provider_name,
        allow_scaffold_mismatch=bool(args.allow_scaffold_mismatch),
    ))
    write_json(run_dir / "preflight.json", preflight)
    # The attestation record travels with the manifest, not just the preflight file, so a
    # scored row can always be attributed to a runtime version + commit.
    base_manifest["extra"]["runtime_attestation"] = (
        (preflight.get("details") or {}).get("runtime_attestation") or {})
    if not preflight["ok"]:
        # The documented contract is that a launcher NAMES the exact attestation reason in its
        # typed refusal, not just that a preflight failed: `runtime_skew` and "the task JSON is
        # missing" are different operator actions. When the attestation is what refused, its
        # reason and stage are the refusal; otherwise the generic preflight one stands.
        attestation = base_manifest["extra"]["runtime_attestation"]
        attestation_reason = (str(attestation.get("reason") or "")
                              if isinstance(attestation, dict) and not attestation.get("ok", True)
                              else "")
        refusal = ({"stage": "runtime_attestation", "reason": attestation_reason, "exit_code": 2}
                   if attestation_reason
                   else {"stage": "preflight", "reason": "preflight_failed", "exit_code": 2})
        final.update({"outcome": "blocked", "exit_code": 2, "refusal": refusal})
        outcome = _write_task_records(TaskRecordConfig(
            run_dir=run_dir,
            result_root=result_root,
            repo_dir=repo_dir,
            settings_path=settings_path,
            example_id=example_id,
            domain=domain,
            reward=None,
            steps=0,
            status="blocked",
            reason_code="preflight_failed",
            base_manifest=base_manifest,
            error="; ".join(preflight["failures"]),
            extra={"preflight": preflight},
        ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 2
    example = json.loads(task_path.read_text(encoding="utf-8"))
    example_id = str(example.get("id") or task_path.stem)
    (run_dir / "task.json").write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
    from desktop_env.desktop_env import DesktopEnv

    env = None
    agent = OuroborosStepAgent(StepAgentConfig(
        ouroboros_bin=args.ouroboros_bin,
        ouroboros_url=args.ouroboros_url,
        repo_dir=repo_dir,
        data_dir=data_dir,
        settings_path=settings_path,
        result_dir=run_dir,
        task_id=example_id,
        model=args.model,
        timeout_sec=args.step_timeout_sec,
        max_obs_chars=args.max_obs_chars,
        screenshot_check_only=args.screenshot_check_only,
        disable_tools=args.disable_tools,
    ))

    try:
        # The constructor boots the VM, so a failed boot needs the SAME retry+cleanup
        # discipline the reset loop already has (see construct_desktop_env).
        env = construct_desktop_env(
            DesktopEnv,
            attempts=max(1, int(args.reset_retries)),
            deadline=time.time() + max(1, int(args.startup_timeout_sec)),
            retry_sleep_sec=max(0.1, float(args.startup_retry_sleep_sec)),
            provider_name=args.provider_name,
            path_to_vm=args.path_to_vm,
            action_space="pyautogui",
            screen_size=(1920, 1080),
            headless=not args.show_vm,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type == "screenshot_a11y_tree",
        )
        obs = _initial_observation_with_retries(
            env,
            example,
            startup_timeout_sec=args.startup_timeout_sec,
            reset_retries=args.reset_retries,
            wait_after_reset_sec=max(0.0, args.wait_after_reset_sec),
            retry_sleep_sec=max(0.1, args.startup_retry_sleep_sec),
            run_dir=run_dir,
        )
        agent.reset()
        instruction = str(example["instruction"])
        done = False
        step_idx = 0
        while not done and step_idx < args.max_steps:
            response, actions, debug = agent.predict(instruction, obs, max_steps=args.max_steps)
            (run_dir / f"debug_step_{step_idx + 1:03d}.json").write_text(
                json.dumps(debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for action_index, action in enumerate(actions, start=1):
                ts = _dt.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                obs, reward, done, info = env.step(action, args.sleep_after_execution)
                agent.record_action(
                    action=action,
                    response=response,
                    reward=float(reward),
                    done=bool(done),
                    info=dict(info or {}),
                )
                # Official convention (lib_run_single.py): save the post-action
                # screenshot for the last action of a step (or on done) under
                # step_<n>_<ts>.png and reference it from the traj row.
                screenshot_file = None
                if action_index == len(actions) or done:
                    shot = obs.get("screenshot") if isinstance(obs, dict) else None
                    if isinstance(shot, (bytes, bytearray)) and shot:
                        screenshot_file = f"step_{step_idx + 1}_{ts}.png"
                        (run_dir / screenshot_file).write_bytes(bytes(shot))
                with (run_dir / "traj.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step_num": step_idx + 1,
                        "action_timestamp": ts,
                        "action": action,
                        "response": response,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "screenshot_file": screenshot_file,
                        "adapter_debug": debug,
                    }, ensure_ascii=False, default=str) + "\n")
                if done:
                    break
            step_idx += 1
            if args.screenshot_check_only:
                break

        reward = _persist_evaluation_result(env.evaluate(), run_dir)
        evaluator_cfg = example.get("evaluator") if isinstance(example, dict) else None
        evaluator_func = evaluator_cfg.get("func") if isinstance(evaluator_cfg, dict) else None
        outcome = _write_task_records(TaskRecordConfig(
            run_dir=run_dir,
            result_root=result_root,
            repo_dir=repo_dir,
            settings_path=settings_path,
            example_id=example_id,
            domain=domain,
            reward=reward,
            steps=step_idx,
            status="completed",
            reason_code="official_evaluate",
            base_manifest=base_manifest,
            extra={
                "screenshot_check_only": bool(args.screenshot_check_only),
                "final_answer": agent.final_answer or agent.last_response,
                "terminal_action": agent.terminal_action or "max_steps_exhausted",
                "infeasible_declared": agent.terminal_action == "FAIL",
                "evaluator_func": evaluator_func,
                "observation_type": args.observation_type,
                "max_steps": args.max_steps,
            },
        ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - denominator-preserving adapter failure
        error = f"{type(exc).__name__}: {exc}"
        final.update({"outcome": "adapter_error", "exit_code": 1,
                      "error": {"type": type(exc).__name__, "message": str(exc)[:4000]}})
        outcome = _write_task_records(TaskRecordConfig(
            run_dir=run_dir,
            result_root=result_root,
            repo_dir=repo_dir,
            settings_path=settings_path,
            example_id=example_id,
            domain=domain,
            reward=None,
            steps=locals().get("step_idx", 0),
            status="adapter_error",
            reason_code=type(exc).__name__,
            base_manifest=base_manifest,
            error=error,
            extra={
                "final_answer": agent.final_answer or agent.last_response,
                "terminal_action": agent.terminal_action,
                "infeasible_declared": agent.terminal_action == "FAIL",
            },
        ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
