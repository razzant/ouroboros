#!/usr/bin/env python3
"""Launcher wrapper for the EXTERNAL CL-Bench (continual-learning-bench) runner.

CL-Bench (continual-learning-bench.com, arXiv 2606.05661) feeds a system a
STRICTLY SEQUENTIAL stream of task instances and measures continual learning:
a memoryless *stateless* baseline vs a *stateful* rollout where one persistent
system carries state across the whole ordered sequence. For Ouroboros the
carried state is its native MEMORY (scratchpad/knowledge), which is the whole
point of running this bench — so runs go against a LIVE agent with memory ON,
and cross-task parallelism is FORBIDDEN by the bench design.

This wrapper does NOT reimplement the benchmark. The runner, its tasks, its
scoring, and the in-runner Ouroboros adapter (``src/systems/ouroboros/``,
pinned at commit 56764d6 in the run handoff bundle) live in the external
``continual-learning-bench`` repository, which must be obtained separately and
pointed at via ``--runner-path``. The wrapper adds the house launch pattern:

- fresh append-only run root under ``bench_runs/continual_learning/``;
- per-run settings rendered from ``settings_base.json`` (secrets blanked in
  the on-disk copy; live keys travel via child env only);
- a ``run_manifest.json`` with provenance + an explicit template-fidelity
  report (which template knobs the pinned external adapter actually enforces);
- normalized ``results.json`` + denominator-preserving ``result_index.jsonl``
  collected from the runner's per-question trace artifacts.

Two observed entrypoints of the external repo are supported:

- ``standard``: ``python run_benchmark.py run <domain> --system ouroboros
  --system-params '{...}' --task.schedule default --runs 1 --max-workers 1
  --no-live-dashboard`` (the adapter METHODOLOGY's "clean path": per-action
  interaction, runner owns the loop, evaluate() once at the end).
- ``bridge``: ``python -m src.systems.ouroboros.run_clbench_bridge_agent ...``
  (one whole Ouroboros agent task per question via the host-side shim; this is
  what produced the reference full40 DB run of 2026-07-01).

Official scoring stays with the external runner/leaderboard scripts; the
normalized results here are audit sidecars, not replacement scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import benchmark_run_manifest, write_json
from devtools.benchmarks.common.run_roots import ensure_outside_repo, run_root
from devtools.benchmarks.common.secrets import load_secret_env, redacted_env_summary

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent

DOMAINS = (
    "database_exploration",
    "exploitable_poker",
    "codebase_adaptation",
    "cohort_studies",
    "blind_spectrum_monitoring",
    "sales_prediction",
)
BRIDGE_PHASES = ("stateless", "stateful_noevo", "stateful_evo")
BRIDGE_MODULE = "src.systems.ouroboros.run_clbench_bridge_agent"
# In-runner Ouroboros adapter subtree + the commit the reference run pinned it at
# (MANIFEST.json of the v6.52.2_full40_db handoff bundle).
ADAPTER_REL = pathlib.Path("src") / "systems" / "ouroboros"
ADAPTER_PINNED_COMMIT = "56764d61afa2860e4893bc14e6229e33fcebf06b"

# Model slots pinned to the single solve model in the rendered settings (single-model
# fairness: the external adapter pins the same set inside the container).
_PINNED_MODEL_KEYS = (
    "OUROBOROS_MODEL",
    "OUROBOROS_MODEL_HEAVY",
    "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_FALLBACKS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
    "OUROBOROS_SCOPE_REVIEW_MODELS",
)


def _bare_model(model: str) -> str:
    """litellm ``openrouter/anthropic/claude-sonnet-4.6`` -> Ouroboros ``anthropic/claude-sonnet-4.6``."""
    text = str(model or "").strip()
    return text.split("openrouter/", 1)[-1] if text.startswith("openrouter/") else text


def _litellm_model(model: str) -> str:
    """Ouroboros ``anthropic/claude-sonnet-4.6`` -> litellm ``openrouter/anthropic/claude-sonnet-4.6``.

    The pinned external adapter only strips an ``openrouter/`` prefix, so this launcher
    supports OpenRouter-routed models; pass a full litellm id to override.
    """
    text = str(model or "").strip()
    return text if text.startswith("openrouter/") else f"openrouter/{text}"


def render_run_settings(base_path: pathlib.Path, run_dir: pathlib.Path, *, solve_model: str,
                        evolution: bool, total_budget: float | None) -> dict:
    """Render the per-run settings template into ``<run>/_run_settings.json``.

    The on-disk copy has every secret-shaped value blanked (house rule: live keys
    enter only through the child environment, never through committed/persisted
    files). The rendered dict is the run's DECLARED Ouroboros configuration; see
    the manifest ``fidelity`` block for which keys the pinned external adapter
    actually enforces.
    """
    settings = json.loads(pathlib.Path(base_path).expanduser().read_text(encoding="utf-8"))
    bare = _bare_model(solve_model)
    for key in _PINNED_MODEL_KEYS:
        settings[key] = bare
    settings["OUROBOROS_REVIEW_MODELS"] = ",".join([bare] * 3)
    settings["OUROBOROS_POST_TASK_EVOLUTION"] = "true" if evolution else "false"
    if total_budget is not None:
        settings["TOTAL_BUDGET"] = float(total_budget)
    for key in list(settings):
        if any(token in key.upper() for token in ("API_KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIALS")):
            settings[key] = ""
    write_json(run_dir / "_run_settings.json", settings)
    return settings


def check_runner(runner: pathlib.Path) -> dict:
    """Presence report for the external runner checkout (nothing is vendored here)."""
    runner = pathlib.Path(runner).expanduser().resolve(strict=False)
    adapter = runner / ADAPTER_REL
    return {
        "path": str(runner),
        "present": runner.is_dir(),
        "run_benchmark_present": (runner / "run_benchmark.py").is_file(),
        "adapter_present": (adapter / "system.py").is_file(),
        "bridge_present": (adapter / "run_clbench_bridge_agent.py").is_file(),
        "adapter_pinned_commit": ADAPTER_PINNED_COMMIT,
    }


def check_clone(clone: pathlib.Path) -> None:
    """The Ouroboros clone handed to the external adapter must never be the live repo."""
    resolved = pathlib.Path(clone).expanduser().resolve(strict=False)
    if resolved == REPO.resolve(strict=False):
        raise SystemExit("--ouroboros-clone must be a dedicated CLONE, never the live repo "
                         f"({REPO}); the adapter's engines boot throwaway sub-clones from it.")
    if not (resolved / "devtools" / "benchmarks" / "common" / "server_runner.py").exists():
        raise SystemExit(f"--ouroboros-clone does not look like an Ouroboros checkout "
                         f"(missing devtools/benchmarks/common/server_runner.py): {resolved}")


def _fidelity_report(settings: dict, args: argparse.Namespace) -> dict:
    """Declared-vs-enforced template knobs, given the PINNED external adapter (56764d6).

    Everything under ``declared_only`` is recorded and exported but NOT forwarded into
    the agent container by the pinned adapter — enforcing it needs the small adapter
    patch described in METHODOLOGY.md ("Template fidelity").
    """
    disabled = list(settings.get("CLBENCH_SOLVE_DISABLED_TOOLS") or [])
    return {
        "enforced_via_runner_interface": {
            "model_slots": _bare_model(args.model) + " (adapter pins every slot in-container)",
            "OUROBOROS_MAX_WORKERS": int(settings.get("OUROBOROS_MAX_WORKERS") or 1),
            "OUROBOROS_POST_TASK_EVOLUTION": settings.get("OUROBOROS_POST_TASK_EVOLUTION"),
            "OUROBOROS_EFFORT_TASK": args.effort + " (env; adapter applies uniformly to all effort knobs)",
            "OUROBOROS_OR_PROVIDER": args.or_provider,
            "TOTAL_BUDGET": settings.get("TOTAL_BUDGET"),
            "OUROBOROS_RUNTIME_MODE": "advanced (hard-set by the adapter; template must stay 'advanced')",
            "memory": "shared memory_mode on every solve task; one persistent server per stateful rollout",
        },
        "declared_only_pinned_adapter_gap": {
            "OUROBOROS_SAFETY_MODE": {
                "declared": settings.get("OUROBOROS_SAFETY_MODE"),
                "status": ("exported in env — effective on the HOST engine path (IsolatedServer inherits env); "
                           "the docker engine forwards only an explicit -e list and DROPS it"),
            },
            "CLBENCH_SOLVE_DISABLED_TOOLS": {
                "declared": disabled,
                "status": ("exported in env as a comma list; the pinned adapter does not read it — the standard "
                           "path submits tasks WITHOUT disabled_tools, the bridge path hardcodes its own 8-tool "
                           "list (no claude_code_edit)"),
            },
            "OUROBOROS_REVIEW_ENFORCEMENT": {
                "declared": settings.get("OUROBOROS_REVIEW_ENFORCEMENT"),
                "status": ("exported in env — effective on the HOST engine path (IsolatedServer inherits env); "
                           "the docker engine forwards only an explicit -e list and DROPS it"),
            },
        },
    }


def _print_fidelity_warnings(fidelity: dict) -> None:
    for key, info in fidelity.get("declared_only_pinned_adapter_gap", {}).items():
        print(f"[clb] WARNING: template knob {key} is DECLARED but not enforced by the pinned "
              f"external adapter — {info['status']}", file=sys.stderr)


def _sanitized_child_env(run_dir: pathlib.Path, settings: dict, args: argparse.Namespace) -> dict:
    """Child env for the external runner: strip live-runtime/secret env, add the knobs the
    adapter observes, resolve provider keys env-first (never printed)."""
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("OUROBOROS_", "USE_LOCAL_"))
        and not any(token in key.upper() for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIALS"))
    }
    env.update(load_secret_env())  # OPENROUTER/OPENAI/ANTHROPIC keys + GITHUB_TOKEN, env-first
    env.pop("GITHUB_TOKEN", None)  # owner secret; the bench never needs it
    env.update({
        "OUROBOROS_BENCH_CLONE": str(pathlib.Path(args.ouroboros_clone).expanduser().resolve(strict=False)),
        # Adapter sidecar ledgers + engine run roots (obo_clbench_*/obo_dockerclbench_*)
        # land INSIDE this run dir instead of the default ~/cl_bench_runs.
        "OUROBOROS_BENCH_RUNS_ROOT": str(run_dir / "runner_state"),
        "OUROBOROS_EFFORT_TASK": args.effort,
        "OUROBOROS_TOTAL_BUDGET": str(float(settings.get("TOTAL_BUDGET") or 0.0)),
        # Declared-only knobs (see _fidelity_report): effective on the host engine path,
        # forward-channel for a patched adapter on the docker path.
        "OUROBOROS_SAFETY_MODE": str(settings.get("OUROBOROS_SAFETY_MODE") or "light"),
        "OUROBOROS_REVIEW_ENFORCEMENT": str(settings.get("OUROBOROS_REVIEW_ENFORCEMENT") or "blocking"),
        "CLBENCH_SOLVE_DISABLED_TOOLS": ",".join(settings.get("CLBENCH_SOLVE_DISABLED_TOOLS") or []),
    })
    if args.or_provider:
        env["OUROBOROS_OR_PROVIDER"] = args.or_provider
    return env


def _system_params(settings: dict, args: argparse.Namespace) -> dict:
    return {
        "model": _litellm_model(args.model),
        "engine": "ouroboros",
        "mode": "stateful",
        "docker": bool(args.docker),
        "docker_image": args.docker_image,
        "resume": not args.no_resume,
        "ouroboros_repo": str(pathlib.Path(args.ouroboros_clone).expanduser().resolve(strict=False)),
        "evolution": bool(args.evolution),
        "cadence": args.cadence,
        "max_workers": int(settings.get("OUROBOROS_MAX_WORKERS") or 1),
        "task_timeout_sec": int(args.task_timeout_sec),
    }


def build_planned_argv(args: argparse.Namespace, settings: dict, run_dir: pathlib.Path,
                       runner_python: str) -> list[list[str]]:
    """The exact external-runner invocation(s), from the runner's observed interface."""
    if args.path == "standard":
        argv = [
            runner_python, "run_benchmark.py", "run", args.domain,
            "--system", "ouroboros",
            "--system-params", json.dumps(_system_params(settings, args), sort_keys=True),
            "--task.schedule", args.schedule,
            "--runs", str(args.runs),
            "--max-workers", str(args.instance_workers),
            "--no-live-dashboard",
        ]
        return [argv]
    plans: list[list[str]] = []
    for phase in args.phases.split(","):
        phase = phase.strip()
        if phase not in BRIDGE_PHASES:
            raise SystemExit(f"unknown bridge phase {phase!r}; choose from {BRIDGE_PHASES}")
        argv = [
            runner_python, "-m", BRIDGE_MODULE,
            "--domain", args.domain,
            "--phases", phase,
            "--num-instances", str(args.num_instances),
            "--model", _litellm_model(args.model),
            "--ouroboros-repo", str(pathlib.Path(args.ouroboros_clone).expanduser().resolve(strict=False)),
            "--result-dir", str(run_dir / "traces"),
            "--task-timeout-sec", str(args.task_timeout_sec),
            "--max-workers", str(int(settings.get("OUROBOROS_MAX_WORKERS") or 1)),
            "--cadence", args.cadence,
        ]
        if args.memory_instruction:
            argv += ["--memory-instruction", args.memory_instruction]
        if args.run_index is not None:
            argv += ["--run-index", str(args.run_index)]
        if phase == "stateless" and args.instance_workers > 1:
            argv += ["--concurrency", str(args.instance_workers)]
        if args.docker:
            argv += ["--docker", "--docker-image", args.docker_image]
        plans.append(argv)
    return plans


# ------------------------------------------------------------------ collection
def _mean(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def collect_results(run_dir: pathlib.Path) -> dict:
    """Normalize bridge-layout traces (``traces/<condition>/<domain>/q###/task_outcome.json``)
    into ``results.json`` + a denominator-preserving ``result_index.jsonl``. Best-effort:
    standard-path runs keep their artifacts inside the external runner's own output tree,
    so for those this records sidecar pointers only. Official scoring stays external."""
    run_dir = pathlib.Path(run_dir)
    traces = run_dir / "traces"
    conditions: dict[str, dict] = {}
    ledger_rows: list[dict] = []
    # Denominator preservation: a requested question whose task_outcome.json never
    # appeared must surface as an explicit missing row, not silently vanish —
    # INCLUDING when a whole planned condition/phase directory is absent (runner
    # died before creating it). Requested ids and planned bridge phases come from
    # this run's own manifest (absent for foreign dirs -> best-effort as before).
    expected_ids: list[str] = []
    expected_conditions: list[str] = []
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_ids = [str(x) for x in (manifest.get("requested_task_ids") or [])]
            phases = str((manifest.get("extra") or {}).get("phases") or "")
            expected_conditions = [ph.strip() for ph in phases.split(",") if ph.strip()]
        except (OSError, ValueError):
            expected_ids, expected_conditions = [], []
    found_condition_names: list[str] = (
        sorted(p.name for p in traces.iterdir() if p.is_dir()) if traces.is_dir() else []
    )
    if not found_condition_names and not expected_conditions and expected_ids:
        # Standard-path run (no bridge trace conditions at all): the external
        # runner keeps per-question artifacts in its OWN output tree, so this
        # ledger can only assert the request — one explicit pointer row per
        # requested run keeps the denominator instead of an empty file.
        for expected in expected_ids:
            domain, _, rest = expected.partition(":")
            ledger_rows.append({
                "benchmark": "continual_learning",
                "condition": "standard",
                "domain": domain,
                "instance_index": None,
                "instance_id": rest or expected,
                "reward": None,
                "success": None,
                "ouroboros_status": "external_runner_sidecar_only",
                "cost_usd": None,
            })
    for cond_name in sorted(set(found_condition_names) | set(expected_conditions)):
        cond_dir = traces / cond_name
        rewards: dict[int, float | None] = {}
        costs: list[float] = []
        found_keys: set[tuple[str, str]] = set()
        if cond_dir.is_dir():
            for outcome_path in sorted(cond_dir.glob("*/q*/task_outcome.json")):
                try:
                    row = json.loads(outcome_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    row = {}
                idx = row.get("instance_index")
                idx = int(idx) if isinstance(idx, (int, float)) else None
                reward = row.get("reward") if isinstance(row.get("reward"), (int, float)) else None
                if idx is not None:
                    rewards[idx] = reward
                if isinstance(row.get("cost_usd"), (int, float)):
                    costs.append(float(row["cost_usd"]))
                domain = str(row.get("domain") or outcome_path.parent.parent.name)
                qid = outcome_path.parent.name
                found_keys.add((domain, qid))
                ledger_rows.append({
                    "benchmark": "continual_learning",
                    "condition": cond_name,
                    "domain": domain,
                    "instance_index": idx,
                    "instance_id": qid,
                    "reward": reward,
                    "success": row.get("success"),
                    "ouroboros_status": str(row.get("ouroboros_status") or ""),
                    "cost_usd": row.get("cost_usd"),
                })
        for expected in expected_ids:
            domain, _, qid = expected.partition(":")
            if qid and (domain, qid) not in found_keys:
                ledger_rows.append({
                    "benchmark": "continual_learning",
                    "condition": cond_name,
                    "domain": domain,
                    "instance_index": None,
                    "instance_id": qid,
                    "reward": None,
                    "success": None,
                    "ouroboros_status": "missing_outcome",
                    "cost_usd": None,
                })
        scored = [r for r in rewards.values() if r is not None]
        conditions[cond_name] = {
            "n_outcomes": len(rewards),
            "n_scored": len(scored),
            "missing_reward_indices": sorted(i for i, r in rewards.items() if r is None),
            "mean_reward": _mean(list(rewards.values())),
            "cost_usd": round(sum(costs), 2) if costs else None,
        }
    def _cond_mean(name: str) -> float | None:
        return (conditions.get(name) or {}).get("mean_reward")

    results = {
        "schema": "ouroboros.benchmark.clbench_results.v1",
        "run_root": str(run_dir),
        "conditions": conditions,
        # stateful_noevo - stateless == the memory-driven continual-learning gain.
        "memory_effect": (
            round(_cond_mean("stateful_noevo") - _cond_mean("stateless"), 4)
            if _cond_mean("stateful_noevo") is not None and _cond_mean("stateless") is not None else None
        ),
        "evolution_effect": (
            round(_cond_mean("stateful_evo") - _cond_mean("stateful_noevo"), 4)
            if _cond_mean("stateful_evo") is not None and _cond_mean("stateful_noevo") is not None else None
        ),
        "runner_sidecars": sorted(
            str(p.relative_to(run_dir)) for p in (run_dir / "runner_state").glob("cl_bench/*/run_manifest.json")
        ) if (run_dir / "runner_state").is_dir() else [],
        "scoring_note": ("raw per-instance rewards; the RANKED leaderboard metric (normalized_reward_mean vs the "
                         "fixed cross-system stateless gpt-5.4 baseline) is computed only by the external runner's "
                         "official analysis scripts"),
    }
    write_json(run_dir / "results.json", results)
    ledger_path = run_dir / "result_index.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ledger_rows), encoding="utf-8")
    return results


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Ouroboros on CL-Bench via the EXTERNAL continual-learning-bench runner "
                    "(obtain separately; nothing is vendored here).")
    parser.add_argument("--runner-path", default=os.environ.get("CLBENCH_RUNNER_PATH", ""),
                        help="path to the external continual-learning-bench checkout "
                             "(must contain run_benchmark.py and src/systems/ouroboros/)")
    parser.add_argument("--runner-python", default="",
                        help="python interpreter with the runner's deps (default: <runner>/.venv/bin/python "
                             "if present, else the current interpreter)")
    parser.add_argument("--ouroboros-clone", default=os.environ.get("OUROBOROS_BENCH_CLONE", ""),
                        help="dedicated Ouroboros CLONE for the adapter's isolated servers (NEVER the live repo)")
    parser.add_argument("--settings", default=str(HERE / "settings_base.json"))
    parser.add_argument("--out-dir", default="", help="run root (default: fresh bench_runs/continual_learning/<ts>)")
    parser.add_argument("--path", default="standard", choices=["standard", "bridge"],
                        help="standard = run_benchmark.py per-action clean path (adapter METHODOLOGY go-forward); "
                             "bridge = run_clbench_bridge_agent whole-question path (reference full40 run shape)")
    parser.add_argument("--domain", default="database_exploration", choices=DOMAINS)
    parser.add_argument("--model", default="",
                        help="solve model, litellm or bare OpenRouter id (default: settings OUROBOROS_MODEL)")
    parser.add_argument("--schedule", default="default",
                        help="standard path --task.schedule (default = canonical schema_drift schedule)")
    parser.add_argument("--runs", type=int, default=1,
                        help="standard path rollout seeds (leaderboard grade needs 5; 1 = smoke)")
    parser.add_argument("--phases", default="stateless,stateful_noevo",
                        help="bridge path conditions, comma-separated in run order "
                             f"(choices per phase: {', '.join(BRIDGE_PHASES)})")
    parser.add_argument("--num-instances", type=int, default=40, help="bridge path question count (full DB = 40)")
    parser.add_argument("--run-index", type=int, default=None,
                        help="bridge path per-seed question-shuffle index (None == CC baseline order)")
    parser.add_argument("--memory-instruction", default="tools", choices=["", "generic", "tools"],
                        help="bridge path memory nudge; 'tools' was the reference run's A/B winner")
    parser.add_argument("--instance-workers", type=int, default=1,
                        help="runner-level parallelism (standard --max-workers / bridge stateless --concurrency). "
                             "MUST stay 1 for anything stateful: cross-task order is the benchmark.")
    parser.add_argument("--allow-parallel-baseline", action="store_true",
                        help="explicitly allow --instance-workers>1 (stateless baseline arm only; "
                             "each worker boots its own container — watch disk)")
    parser.add_argument("--docker", dest="docker", action="store_true", default=True,
                        help="run the agent inside Docker (leak-proof: task+DB stay host-side; default)")
    parser.add_argument("--no-docker", dest="docker", action="store_false")
    parser.add_argument("--docker-image", default="clbench-ouroboros:dev")
    parser.add_argument("--no-resume", action="store_true",
                        help="standard path: disable within-question conversation-resume chaining")
    parser.add_argument("--evolution", action="store_true",
                        help="enable post-task self-evolution (OFF by default; a separately-disclosed condition)")
    parser.add_argument("--cadence", default="every_n:1", help="post-task evolution cadence (with --evolution)")
    parser.add_argument("--effort", default="", help="uniform reasoning effort (default: settings OUROBOROS_EFFORT_TASK)")
    parser.add_argument("--or-provider", default="", help="OpenRouter provider routing (default: settings OUROBOROS_OR_PROVIDER)")
    parser.add_argument("--task-timeout-sec", type=int, default=900)
    parser.add_argument("--total-budget", type=float, default=None,
                        help="isolated-server TOTAL_BUDGET (default: settings TOTAL_BUDGET)")
    parser.add_argument("--dry-run", action="store_true", help="write manifest + rendered settings + planned argv, spend nothing")
    parser.add_argument("--collect-only", default="",
                        help="skip launching; (re)normalize results in an EXISTING run dir")
    args = parser.parse_args(argv)

    if args.collect_only:
        results = collect_results(pathlib.Path(args.collect_only).expanduser())
        print(json.dumps(results, indent=2))
        return 0

    settings_template = json.loads(pathlib.Path(args.settings).expanduser().read_text(encoding="utf-8"))
    args.model = args.model or str(settings_template.get("OUROBOROS_MODEL") or "")
    args.effort = args.effort or str(settings_template.get("OUROBOROS_EFFORT_TASK") or "low")
    args.or_provider = args.or_provider or str(settings_template.get("OUROBOROS_OR_PROVIDER") or "")
    if not args.model:
        raise SystemExit("no solve model: pass --model or set OUROBOROS_MODEL in the settings template")

    # Strict-sequential guard: the bench design forbids cross-task parallelism. Only the
    # per-instance-independent stateless baseline may fan out, and only on explicit opt-in.
    if args.instance_workers > 1:
        if not args.allow_parallel_baseline:
            raise SystemExit("--instance-workers>1 requires --allow-parallel-baseline: CL-Bench task order is "
                             "STRICTLY SEQUENTIAL (only the stateless baseline arm may run instances in parallel).")
        if args.path == "standard":
            raise SystemExit("--instance-workers>1 is stateless-baseline-only, but the standard path always "
                             "runs mode='stateful' — a parallel stateful run would break the strict-sequential "
                             "contract. Use --path bridge --phases stateless for the parallel baseline.")
        non_stateless = [ph.strip() for ph in args.phases.split(",") if ph.strip() and ph.strip() != "stateless"]
        if non_stateless:
            raise SystemExit(f"--instance-workers>1 allows ONLY stateless phases; drop {non_stateless} "
                             "or run them in a separate sequential invocation.")
    if args.evolution and args.path == "bridge" and "stateful_evo" not in args.phases:
        print("[clb] note: --evolution on the bridge path only takes effect in the stateful_evo phase",
              file=sys.stderr)

    runner_report = check_runner(pathlib.Path(args.runner_path or "."))
    if not args.runner_path or not runner_report["present"] or not (
            runner_report["run_benchmark_present"] if args.path == "standard" else runner_report["bridge_present"]):
        message = (
            "external runner not found/incomplete at "
            f"{runner_report['path'] or '(unset)'} — the continual-learning-bench repo is REQUIRED and is "
            "NOT vendored in this repository. Obtain it separately; its Ouroboros adapter "
            f"(src/systems/ouroboros/, pinned commit {ADAPTER_PINNED_COMMIT[:7]}) ships in the run handoff "
            "bundle under bench-config/external-adapters/ouroboros/. See README.md."
        )
        if not args.dry_run:
            raise SystemExit(message)
        print(f"[clb] WARNING (dry-run): {message}", file=sys.stderr)
    if args.ouroboros_clone:
        check_clone(pathlib.Path(args.ouroboros_clone))
    elif not args.dry_run:
        raise SystemExit("--ouroboros-clone (or $OUROBOROS_BENCH_CLONE) is required for a real run")
    else:
        args.ouroboros_clone = "(unset)"

    out = pathlib.Path(args.out_dir).expanduser() if args.out_dir else run_root("continual_learning")
    out = ensure_outside_repo(out, REPO)
    rendered = render_run_settings(pathlib.Path(args.settings), out, solve_model=args.model,
                                   evolution=args.evolution, total_budget=args.total_budget)
    runner_python = args.runner_python
    if not runner_python:
        venv_python = pathlib.Path(runner_report["path"]) / ".venv" / "bin" / "python"
        runner_python = str(venv_python) if venv_python.exists() else sys.executable
    plans = build_planned_argv(args, rendered, out, runner_python)
    fidelity = _fidelity_report(rendered, args)
    _print_fidelity_warnings(fidelity)

    if args.path == "bridge":
        requested = [f"{args.domain}:q{i:03d}" for i in range(args.num_instances)]
    else:
        requested = [f"{args.domain}:{args.schedule}:run{r}" for r in range(args.runs)]
    child_env = _sanitized_child_env(out, rendered, args) if args.ouroboros_clone != "(unset)" else dict(os.environ)
    manifest = benchmark_run_manifest(
        benchmark="continual_learning",
        run_root=out,
        repo_dir=REPO,
        requested_task_ids=requested,
        metadata={
            "argv": sys.argv if argv is None else [sys.argv[0], *argv],
            "dataset": "continual-learning-bench (external)",
            "official_command": plans[0],
            "settings_path": str(out / "_run_settings.json"),
            "output_paths": {"traces": str(out / "traces"), "runner_state": str(out / "runner_state"),
                             "results": str(out / "results.json")},
            "harness": {"external_runner": runner_report, "entrypoint": args.path,
                        "planned_invocations": plans},
            "timeout_sec": args.task_timeout_sec,
            "extra": {
                "domain": args.domain,
                "path": args.path,
                "solve_model": _litellm_model(args.model),
                "phases": args.phases if args.path == "bridge" else "",
                "schedule": args.schedule if args.path == "standard" else "default",
                "runs": args.runs if args.path == "standard" else 1,
                "num_instances": args.num_instances if args.path == "bridge" else None,
                "docker": bool(args.docker),
                "evolution": bool(args.evolution),
                "memory": "ouroboros_native_shared (persistent server per stateful rollout; "
                          "conversation reset at question boundaries — memory, not ICL)",
                "strict_sequential": {
                    "instance_workers": args.instance_workers,
                    "note": "cross-task order is strictly sequential by bench design; "
                            "OUROBOROS_MAX_WORKERS is a WITHIN-task subagent pool, not cross-task parallelism",
                },
                "fidelity": fidelity,
                "provider_env_present": redacted_env_summary(child_env),
                "report_grade": "local_low_seed" if (args.runs < 5 or args.path == "bridge") else "leaderboard_shape",
            },
        },
    )
    write_json(out / "run_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps({"run_root": str(out), "planned_invocations": plans, "fidelity": fidelity}, indent=2))
        return 0

    rc = 0
    for plan in plans:
        proc = subprocess.run(plan, cwd=runner_report["path"], env=child_env)
        rc = proc.returncode or rc
        if proc.returncode != 0:
            print(f"[clb] runner invocation failed rc={proc.returncode}: {' '.join(plan[:4])} ...",
                  file=sys.stderr)
            break
    collect_results(out)
    print(f"[clb] run root: {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
