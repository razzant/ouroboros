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

from devtools.benchmarks.common.manifests import (
    admit_benchmark_run,
    finalize_run_manifest,
    model_slot_snapshot,
    repo_provenance,
    write_json,
)
from devtools.benchmarks.common.run_roots import assert_outside_repo, live_repo_roots, run_root
from devtools.benchmarks.common.secrets import load_secret_env, redacted_env_summary
from devtools.benchmarks.common.model_slots import (
    configured_subagents_snapshot,
    single_model_subagents_setting,
)
from ouroboros.context_mode_compat import normalize_context_mode_compat

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
# Honest label for an instance the bridge recovered from the shim's full
# ``instance_outcomes`` list: officially scored, but closed as a side effect of another
# question's step, so the agent never had its own turn on it. Written by the tracked
# operator patch ``clb_multi_instance_outcomes.v6746.patch``; read here for disclosure.
AUTO_RESOLVED_STATUS = "auto_resolved_no_agent_turn"
BRIDGE_MODULE = "src.systems.ouroboros.run_clbench_bridge_agent"
# In-runner Ouroboros adapter subtree + the commit the reference run pinned it at
# (MANIFEST.json of the v6.52.2_full40_db handoff bundle).
ADAPTER_REL = pathlib.Path("src") / "systems" / "ouroboros"
ADAPTER_PINNED_COMMIT = "56764d61afa2860e4893bc14e6229e33fcebf06b"

# Model slots pinned to the single solve model in the rendered settings (single-model
# fairness: the external adapter pins the same set inside the container).
_PINNED_MODEL_KEYS = (
    "OUROBOROS_MODEL",
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
    if ("OUROBOROS_CONTEXT_MODE" in settings
            and "OUROBOROS_CONTEXT_MODE_AUTO_LOW" not in settings):
        # A benchmark template mode is an explicit operator declaration.
        settings["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    settings = normalize_context_mode_compat(settings)
    settings["OUROBOROS_TASK_REVIEW_MODE"] = "required"
    settings.pop("OUROBOROS_MODEL_HEAVY", None)
    settings.pop("USE_LOCAL_HEAVY", None)
    bare = _bare_model(solve_model)
    for key in _PINNED_MODEL_KEYS:
        settings[key] = bare
    settings["OUROBOROS_SUBAGENTS"] = single_model_subagents_setting(bare)
    # Reviewer roster: honor an explicit template declaration (e.g. a single
    # low-effort reviewer for a campaign); default to the triple-slot parity
    # roster only when the template does not declare one.
    if not str(settings.get("OUROBOROS_REVIEW_MODELS") or "").strip():
        settings["OUROBOROS_REVIEW_MODELS"] = ",".join([bare] * 3)
    else:
        settings["OUROBOROS_REVIEW_MODELS"] = ",".join(
            _bare_model(x.strip()) for x in str(settings["OUROBOROS_REVIEW_MODELS"]).split(",") if x.strip())
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


# Tracked operator patches whose EFFECT this launcher's provenance block describes, each keyed
# to markers that only the applied patch puts into the ADAPTER checkout (`--runner-path`, the
# external continual-learning-bench tree) — NOT the `--ouroboros-clone` execution seed the
# seed gate binds to; the two are different trees and an operator who patches the wrong one
# gets an honest "unpatched" report. See operator_patches/README.md; a patch renamed there
# must be renamed here in the same commit.
#
# MARKER-UNIQUENESS REQUIREMENT — do not weaken this. Every token below must be something ONLY
# the applied patch writes: its own comment tag, a `def` line, the exact expression it
# introduces. A bare env-var NAME is NOT such a token. The pinned adapter may legitimately
# mention `OUROBOROS_SAFETY_MODE` or `CLBENCH_SOLVE_DISABLED_TOOLS` in a comment or a `-e`
# passthrough list without the patch being applied, and the resulting false positive errs in
# the DANGEROUS direction: the manifest reports `enforced_via_operator_patch` for a run that
# actually executed unenforced. Every token in the tuple must match, never just one.
ADAPTER_PATCH_MARKERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "clb_docker_runtime_attestation.v6746": ("_docker_launcher.py", (
        "def _attest_runtime(", "self.runtime_attestation")),
    "clb_env_campaign_overrides.v6745": ("_docker_launcher.py", (
        "Operator env overrides (campaign knobs)", "_v = os.environ.get(_k)",
        '"OUROBOROS_SUBAGENTS"):')),
    "clb_host_available_subagents.phase1d": ("_launcher.py", (
        "Canonical benchmark actor bytes are authored by run_clb.py through Phase-1A",
        'OUROBOROS_SUBAGENTS=os.environ.get("OUROBOROS_SUBAGENTS", "")')),
    "clb_disabled_tools_env.v6745": ("run_clbench_bridge_agent.py", (
        "Operator patch 2026-07-23: honor CLBENCH_SOLVE_DISABLED_TOOLS",
        '_os.environ.get("CLBENCH_SOLVE_DISABLED_TOOLS"')),
}

# Which adapter module each entrypoint actually executes. The bridge entrypoint runs
# `run_clbench_bridge_agent.py`; `--path standard` (the DEFAULT) goes through
# `run_benchmark.py` -> `system.py` -> `_docker_launcher.submit()`, which hardcodes
# `"disabled_tools": []`. A marker in a module the chosen entrypoint never imports says
# nothing about the run.
_DISABLED_TOOLS_FORWARDING_ENTRYPOINTS = ("bridge",)


def adapter_patch_probe(runner: pathlib.Path) -> dict:
    """Which tracked operator patches are ACTUALLY present in the adapter checkout.

    The provenance and fidelity blocks used to describe the patched adapter from CONSTANTS —
    "``--docker`` was passed, therefore the attestation hook is there; the adapter is the
    pinned commit, therefore these three knobs are dropped". Both statements are about a tree,
    and the tree is the thing that was never consulted: the adapter subtree is patched (or
    not) by an operator, out of band, per run.

    So the record is DERIVED, by looking for markers the applied patch leaves behind. An
    unreadable or absent adapter file yields ``False``: absent provenance recorded honestly is
    fine, a claimed one that did not happen is not.

    NOTE THE SUBJECT. This probes the ``--runner-path`` ADAPTER subtree (the external
    continual-learning-bench checkout, ``<runner>/src/systems/ouroboros``), NOT the
    ``--ouroboros-clone`` execution seed the seed gate binds to. They are different trees;
    ``adapter_path`` in the returned probe names the one that was read.

    THIS PROBE IS STATIC by design (owner-approved goal: a tree probe). It answers "is the
    patch in this tree", never "did the effect happen at runtime" — which is why the field it
    feeds is named ``runtime_attestation_available``, not ``runtime_attested``.
    """
    adapter = pathlib.Path(runner).expanduser().resolve(strict=False) / ADAPTER_REL
    probe: dict = {"adapter_path": str(adapter), "evidence": "static_source_scan", "patches": {}}
    sources: dict[str, str] = {}
    for name, (filename, markers) in ADAPTER_PATCH_MARKERS.items():
        if filename not in sources:
            try:
                sources[filename] = (adapter / filename).read_text(encoding="utf-8")
            except OSError:
                sources[filename] = ""
        text = sources[filename]
        probe["patches"][name] = bool(text) and all(marker in text for marker in markers)
    return probe


def refuse_live_repo_clone(clone: pathlib.Path) -> pathlib.Path:
    """PURE argument validation: the clone must never be the LIVE repo. Returns it resolved.

    The authority is ``live_repo_roots()`` — the runtime's configured checkout — NOT this
    launcher's own ``REPO``. Comparing against ``REPO`` asked "is this the tree I am executing
    from?", which is the same question only in the development workspace: ship the launcher
    inside a pinned seed and passing that seed as ``--ouroboros-clone``, which is exactly the
    recipe METHODOLOGY prescribes, was refused. The gate's intent is unchanged — a run must
    never be pointed at the live working repo, where it could disturb a running server.

    Deliberately free of any filesystem probe so it can run BEFORE `admit_benchmark_run` —
    "you passed the wrong path" is an argument error with nothing to record, unlike the
    shape probe below, which is a preflight and belongs inside the admitted run.
    """
    resolved = pathlib.Path(clone).expanduser().resolve(strict=False)
    for live in live_repo_roots():
        if resolved == live.expanduser().resolve(strict=False):
            raise SystemExit("--ouroboros-clone must be a dedicated CLONE, never the LIVE repo "
                             f"({live}); the adapter's engines boot throwaway sub-clones from it "
                             "and a benchmark must not touch the tree a running server serves.")
    return resolved


def check_clone(clone: pathlib.Path) -> None:
    """The Ouroboros clone handed to the external adapter must be a real checkout."""
    resolved = refuse_live_repo_clone(clone)
    if not (resolved / "devtools" / "benchmarks" / "common" / "server_runner.py").exists():
        raise SystemExit(f"--ouroboros-clone does not look like an Ouroboros checkout "
                         f"(missing devtools/benchmarks/common/server_runner.py): {resolved}")


def _fidelity_report(settings: dict, args: argparse.Namespace,
                     patch_probe: dict | None = None) -> dict:
    """Declared-vs-enforced template knobs, read off the EXECUTION CLONE rather than a constant.

    Four knobs — safety mode, review enforcement, the exact Available-subagents value and the
    solve-time disabled-tools list — are forwarded only by tracked operator patches. Whether
    they are enforced is therefore a fact about the clone this run will execute, not about
    the pinned commit id: on a patched clone the old text understated a run that was actually
    STRICTER than declared, and a
    reader trusting it would discount the numbers. ``patch_probe`` supplies the fact; when it
    is absent (callers that never resolved a runner) every knob falls back to the pinned
    adapter's unpatched behaviour, which is the conservative direction.

    Knobs land under ``enforced_via_operator_patch`` when the patch is present and under
    ``declared_only_pinned_adapter_gap`` when it is not, so a reader can tell what was
    actually in force without inspecting the clone.
    """
    disabled = list(settings.get("CLBENCH_SOLVE_DISABLED_TOOLS") or [])
    applied = dict((patch_probe or {}).get("patches") or {})
    # ONE marker for the env knobs is CORRECT, not a shortcut: the campaign-override
    # patch adds OUROBOROS_RUNTIME_MODE, OUROBOROS_SAFETY_MODE and OUROBOROS_SUBAGENTS
    # (and the retired review-enforcement key) in one loop/hunk, so they arrive
    # together or not at all. The patch probe includes the new key itself as a marker;
    # an older applied copy is conservatively reported as not enforcing actor purity.
    # The host engine boots a real `IsolatedServer`, so the ordinary scalar env knobs remain
    # effective there whether or not the docker-only patch is applied. The structured actor
    # value is different: IsolatedServer strips inherited runtime env, so it requires the
    # host adapter's explicit settings transport marker below.
    env_knobs_forwarded = (not args.docker) or bool(applied.get("clb_env_campaign_overrides.v6745"))
    subagents_forwarded = (
        bool(applied.get("clb_env_campaign_overrides.v6745"))
        if args.docker
        else bool(applied.get("clb_host_available_subagents.phase1d"))
    )
    # ENTRYPOINT-SPECIFIC. The patch lives in `run_clbench_bridge_agent.py`, which ONLY the
    # bridge entrypoint executes. On the DEFAULT `--path standard` the run goes
    # run_benchmark.py -> system.py -> `_docker_launcher.submit()`, which hardcodes
    # `"disabled_tools": []`; `system.py` never mentions the list at all. Keying purely on the
    # marker therefore filed the knob as ENFORCED on the default path against a patched clone
    # and claimed every declared tool — claude_code_edit included — was excluded from the task
    # contract, for a run whose executed path excluded nothing. Overstating enforcement is the
    # dangerous direction, and it is exactly the class this fix exists to remove.
    tools_forwarded = (bool(applied.get("clb_disabled_tools_env.v6745"))
                       and args.path in _DISABLED_TOOLS_FORWARDING_ENTRYPOINTS)
    enforced: dict = {}
    gap: dict = {}
    _env_status = ("exported in env and applied in-container by operator patch "
                   "clb_env_campaign_overrides.v6745 (docker engine `_overrides`)"
                   if args.docker else
                   "exported in env — effective on the HOST engine path "
                   "(IsolatedServer inherits env)") if env_knobs_forwarded else (
        "exported in env, but the docker engine forwards only an explicit -e list and DROPS "
        "it: operator patch clb_env_campaign_overrides.v6745 is NOT applied in this adapter "
        "checkout")
    (enforced if env_knobs_forwarded else gap)["OUROBOROS_SAFETY_MODE"] = {
        "declared": settings.get("OUROBOROS_SAFETY_MODE"),
        "status": _env_status,
    }
    gap["OUROBOROS_TASK_REVIEW_MODE"] = {
        "declared": "required",
        "status": "requested in rendered settings and child environment; executing external adapter settings and pre-review output producer are not attested by this launcher",
    }
    _subagents_status = (
        "exported in env and materialized in isolated settings by the host adapter"
        if not args.docker and subagents_forwarded
        else _env_status
    ) if subagents_forwarded else (
        "exported in env, but the selected external adapter does not materialize the "
        "structured value into isolated settings"
    )
    (enforced if subagents_forwarded else gap)["OUROBOROS_SUBAGENTS"] = {
        "declared": settings.get("OUROBOROS_SUBAGENTS"),
        "status": _subagents_status,
    }
    (enforced if tools_forwarded else gap)["CLBENCH_SOLVE_DISABLED_TOOLS"] = {
        "declared": disabled,
        "status": ("exported in env as a comma list and read by operator patch "
                   "clb_disabled_tools_env.v6745 (bridge agent `DISABLED_TOOLS`), so every "
                   "declared tool — including claude_code_edit — is excluded from the task "
                   "contract")
        if tools_forwarded else
        ((f"exported in env as a comma list, but this run uses --path {args.path} "
          "(run_benchmark.py -> system.py -> _docker_launcher.submit, which hardcodes "
          "disabled_tools=[]); operator patch clb_disabled_tools_env.v6745 patches "
          "run_clbench_bridge_agent.py, which that entrypoint never executes")
         if applied.get("clb_disabled_tools_env.v6745") else
         ("exported in env as a comma list; operator patch clb_disabled_tools_env.v6745 is NOT "
          "applied in this clone, so the pinned adapter submits tasks WITHOUT disabled_tools "
          "(no claude_code_edit exclusion)")),
    }
    return {
        # STATIC availability of each tracked patch in the `--runner-path` ADAPTER checkout —
        # NOT a claim that any of them ran. The full probe (adapter_path, evidence) is recorded
        # separately at `extra.adapter_operator_patches`; METHODOLOGY §6 names both.
        "adapter_operator_patches": applied,
        "adapter_operator_patches_evidence": "static_source_scan",
        "enforced_via_operator_patch": enforced,
        "declared_only_pinned_adapter_gap": gap,
        "enforced_via_runner_interface": {
            "model_slots": _bare_model(args.model) + " (adapter pins every active slot in-container)",
            "OUROBOROS_MAX_WORKERS": int(settings.get("OUROBOROS_MAX_WORKERS") or 1),
            "OUROBOROS_POST_TASK_EVOLUTION": settings.get("OUROBOROS_POST_TASK_EVOLUTION"),
            "OUROBOROS_EFFORT_TASK": args.effort + " (env; adapter applies uniformly to all effort knobs)",
            "OUROBOROS_OR_PROVIDER": args.or_provider,
            "TOTAL_BUDGET": settings.get("TOTAL_BUDGET"),
            # Same test as the three knobs above: the pinned adapter hard-sets `advanced`, but
            # `clb_env_campaign_overrides.v6745` lets an exported value WIN. Stating the
            # constant on a patched clone naming a different mode would misreport the run.
            "OUROBOROS_RUNTIME_MODE": (
                f"{settings.get('OUROBOROS_RUNTIME_MODE')} (exported; operator patch "
                "clb_env_campaign_overrides.v6745 lets the export override the adapter's "
                "hard-set 'advanced')"
                if applied.get("clb_env_campaign_overrides.v6745") and settings.get("OUROBOROS_RUNTIME_MODE")
                else "advanced (hard-set by the adapter; template must stay 'advanced')"
            ),
            "memory": "shared memory_mode on every solve task; one persistent server per stateful rollout",
        },
    }


def _print_fidelity_warnings(fidelity: dict) -> None:
    for key, info in fidelity.get("declared_only_pinned_adapter_gap", {}).items():
        print(f"[clb] WARNING: template knob {key} is DECLARED but not enforced by the pinned "
              f"external adapter — {info['status']}", file=sys.stderr)


def _sanitized_child_env(run_dir: pathlib.Path, settings: dict, args: argparse.Namespace) -> dict:
    """Child env for the external runner: strip live-runtime/secret env, add the knobs the
    adapter observes, resolve provider keys env-first (never printed)."""
    settings = normalize_context_mode_compat(settings)
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
        # Operator patch 2026-07-23: forward review MODE (not just enforcement) and
        # subagent depth so a submittable "review required" / no-swarm run is achievable
        # from the launcher instead of silently living on defaults (auto / depth 2).
        "OUROBOROS_TASK_REVIEW_MODE": "required",
        "OUROBOROS_MAX_SUBAGENT_DEPTH": str(settings.get("OUROBOROS_MAX_SUBAGENT_DEPTH") or 2),
        "CLBENCH_SOLVE_DISABLED_TOOLS": (
            _dt if isinstance((_dt := settings.get("CLBENCH_SOLVE_DISABLED_TOOLS") or []), str)
            else ",".join(_dt)
        ),
    })
    # Operator patch 2026-07-23 (round 2): forward the remaining campaign knobs a patched
    # adapter observes via env (see _docker_launcher._overrides operator-env block) —
    # runtime mode, reviewer list, split review efforts, context mode, workers. Only set
    # when the template declares them, so unpatched adapters keep parity defaults.
    for _knob in ("OUROBOROS_RUNTIME_MODE", "OUROBOROS_REVIEW_MODELS", "OUROBOROS_EFFORT_REVIEW",
                  "OUROBOROS_EFFORT_SCOPE_REVIEW", "OUROBOROS_CONTEXT_MODE",
                  "OUROBOROS_CONTEXT_MODE_AUTO_LOW", "OUROBOROS_MAX_WORKERS",
                  "OUROBOROS_SUBAGENTS"):
        _val = settings.get(_knob)
        if _val not in (None, ""):
            env[_knob] = (
                "false" if _knob == "OUROBOROS_CONTEXT_MODE_AUTO_LOW"
                and str(_val).strip().lower() in {"0", "false", "off"}
                else str(_val)
            )
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
                "runtime_outcome": {"available": False},
                "cost_usd": None,
            })
    for cond_name in sorted(set(found_condition_names) | set(expected_conditions)):
        cond_dir = traces / cond_name
        rewards: dict[int, float | None] = {}
        costs: list[float] = []
        found_keys: set[tuple[str, str]] = set()
        auto_resolved: list[int | None] = []
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
                if str(row.get("ouroboros_status") or "") == AUTO_RESOLVED_STATUS:
                    auto_resolved.append(idx)
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
                    # The RUNTIME's own terminal reason, written by operator patch
                    # clb_multi_instance_outcomes.v6746. `ouroboros_status` cannot carry
                    # it: a question the per-task USD rail truncated says "failed" exactly
                    # like a genuinely wrong answer. Rows written before that patch carry
                    # no such key, so the gap is STATED rather than assumed absent.
                    "runtime_outcome": (row.get("runtime_outcome")
                                        if isinstance(row.get("runtime_outcome"), dict)
                                        else {"available": False}),
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
                    "runtime_outcome": {"available": False},
                })
        scored = [r for r in rewards.values() if r is not None]
        conditions[cond_name] = {
            "n_outcomes": len(rewards),
            "n_scored": len(scored),
            "missing_reward_indices": sorted(i for i, r in rewards.items() if r is None),
            "mean_reward": _mean(list(rewards.values())),
            "cost_usd": round(sum(costs), 2) if costs else None,
            # Instances the bridge recovered from the shim's full instance_outcomes list
            # because ONE task.step() closed several at once — officially scored, but the
            # agent never had its own turn on them (see METHODOLOGY §4a). Disclose this
            # count next to any mean; it is also why n_outcomes can exceed the requested
            # --num-instances window.
            "n_auto_resolved_no_agent_turn": len(auto_resolved),
            "auto_resolved_no_agent_turn_indices": sorted(i for i in auto_resolved if i is not None),
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
    parser.add_argument("--allow-dirty-seed", action="store_true",
                        help="run even when the EXECUTION seed — the --ouroboros-clone the adapter boots "
                             "its agent servers from — is dirty or its git identity is unreadable (default: "
                             "fail closed). The escape is recorded in the manifest; a dirty seed makes the "
                             "run's provenance irreproducible and NOT submittable.")
    parser.add_argument("--dry-run", action="store_true", help="write manifest + rendered settings + planned argv, spend nothing")
    parser.add_argument("--collect-only", default="",
                        help="skip launching; (re)normalize results in an EXISTING run dir")
    args = parser.parse_args(argv)

    if args.collect_only:
        results = collect_results(pathlib.Path(args.collect_only).expanduser())
        print(json.dumps(results, indent=2))
        return 0

    # --- PURE argument validation / local derivation (everything before admission) ------
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
    if args.ouroboros_clone:
        # Pure path comparison only; the checkout-shape probe is a preflight and runs after
        # admission, so its refusal leaves a durable record.
        execution_clone = refuse_live_repo_clone(pathlib.Path(args.ouroboros_clone))
    elif not args.dry_run:
        raise SystemExit("--ouroboros-clone (or $OUROBOROS_BENCH_CLONE) is required for a real run")
    else:
        args.ouroboros_clone = "(unset)"
        execution_clone = None

    out = pathlib.Path(args.out_dir).expanduser() if args.out_dir else run_root("continual_learning")
    # Confine against EVERY checkout this run touches, not just the launcher's own. Validating
    # only against REPO let admission artefacts land inside the execution clone — the very seed
    # whose cleanliness the gate is about to attest — which both pollutes it and can make the
    # seed dirty. ASSERT, not ensure: the directory must not exist before the manifest does.
    for authority in (REPO, execution_clone):
        if authority is not None:
            out = assert_outside_repo(out, authority)
    manifest_path = out / "run_manifest.json"
    if args.path == "bridge":
        requested = [f"{args.domain}:q{i:03d}" for i in range(args.num_instances)]
    else:
        requested = [f"{args.domain}:{args.schedule}:run{r}" for r in range(args.runs)]

    # --- ADMISSION -----------------------------------------------------------------------
    # The seed gate is bound to the EXECUTION clone — the checkout the adapter actually boots
    # its agent servers from (`--ouroboros-clone`) — not to this launcher's own checkout. The
    # two are different trees by construction (the clone may never be the live repo), so
    # gating REPO let a DIRTY execution seed pass whenever the launcher happened to be clean:
    # the run's numbers came from code the manifest never described. The launcher's own
    # provenance is recorded ALONGSIDE it, never instead of it.
    manifest = admit_benchmark_run(
        manifest_path,
        benchmark="continual_learning",
        run_root=out,
        # dry-run without a clone has no execution seed; the launcher checkout stands in and
        # `seed_gate_target` records which tree the verdict is about.
        repo_dir=execution_clone if execution_clone is not None else REPO,
        requested_task_ids=requested,
        require_clean=not args.allow_dirty_seed,
        metadata={
            "argv": sys.argv if argv is None else [sys.argv[0], *argv],
            "dataset": "continual-learning-bench (external)",
            "settings_path": str(out / "_run_settings.json"),
            "output_paths": {"traces": str(out / "traces"), "runner_state": str(out / "runner_state"),
                             "results": str(out / "results.json")},
            "timeout_sec": args.task_timeout_sec,
            "extra": {
                "domain": args.domain,
                "path": args.path,
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
                "allow_dirty_seed": bool(args.allow_dirty_seed),
                "seed_gate_target": ("execution_clone" if execution_clone is not None
                                     else "launcher_checkout_dry_run"),
                "execution_clone": str(execution_clone) if execution_clone is not None else "",
                # `runtime_attestation_path` is NOT claimed here. On the docker path it is a
                # fact about the ADAPTER checkout (is the operator patch applied?), and a
                # filesystem probe may not run before `admit_benchmark_run`. It is derived
                # post-admission and attached to the retained manifest, next to `fidelity`.
                "report_grade": "local_low_seed" if (args.runs < 5 or args.path == "bridge") else "leaderboard_shape",
            },
        },
    )

    # --- POST-ADMISSION: preflight, settings rendering, the external runner ---------------
    with finalize_run_manifest(manifest_path, manifest) as final:
        # The launcher's own provenance is a GIT SUBPROCESS probe. Evaluated in the admission
        # call's argument list it would have run before the seam was even entered (Python
        # evaluates arguments first) — the same shape phase P1 was blocked on. It is attached
        # to the RETAINED manifest here instead, which `finalize_run_manifest` writes.
        manifest["extra"]["launcher_provenance"] = repo_provenance(REPO)
        settings_template = json.loads(pathlib.Path(args.settings).expanduser().read_text(encoding="utf-8"))
        args.model = args.model or str(settings_template.get("OUROBOROS_MODEL") or "")
        args.effort = args.effort or str(settings_template.get("OUROBOROS_EFFORT_TASK") or "low")
        args.or_provider = args.or_provider or str(settings_template.get("OUROBOROS_OR_PROVIDER") or "")
        if not args.model:
            raise SystemExit("no solve model: pass --model or set OUROBOROS_MODEL in the settings template")
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
                final.update({"outcome": "refused", "exit_code": 1,
                              "refusal": {"stage": "external_runner", "reason": "runner_missing",
                                          "exit_code": 1}})
                raise SystemExit(message)
            print(f"[clb] WARNING (dry-run): {message}", file=sys.stderr)
        if execution_clone is not None:
            check_clone(execution_clone)

        rendered = render_run_settings(pathlib.Path(args.settings), out, solve_model=args.model,
                                       evolution=args.evolution, total_budget=args.total_budget)
        # RE-SNAPSHOT the slots from the settings the engines are actually handed. Admission
        # declared this path, but the file did not exist yet at that moment, so `model_slots`
        # described only the launcher's environment — for the docker engine, which gets a fresh
        # container environment, that is a description of nobody. The SWE-Pro smoke found the
        # same class one file over (a manifest naming the template's model, not the run's).
        rendered_path = out / "_run_settings.json"
        manifest["model_slots"] = model_slot_snapshot(rendered_path, env_overrides=not args.docker)
        manifest["available_subagents"] = configured_subagents_snapshot(
            rendered_path,
            env_overrides=False,
        )
        runner_python = args.runner_python
        if not runner_python:
            venv_python = pathlib.Path(runner_report["path"]) / ".venv" / "bin" / "python"
            runner_python = str(venv_python) if venv_python.exists() else sys.executable
        plans = build_planned_argv(args, rendered, out, runner_python)
        patch_probe = adapter_patch_probe(pathlib.Path(runner_report["path"]))
        fidelity = _fidelity_report(rendered, args, patch_probe)
        _print_fidelity_warnings(fidelity)
        child_env = _sanitized_child_env(out, rendered, args) if args.ouroboros_clone != "(unset)" else dict(os.environ)

        # Late facts on the RETAINED manifest: the finalization seam writes this same dict.
        manifest["official_command"] = plans[0]
        manifest["harness"] = {"external_runner": runner_report, "entrypoint": args.path,
                               "planned_invocations": plans,
                               "settings_template": str(pathlib.Path(args.settings).expanduser()),
                               "settings_derived": str(rendered_path)}
        # NAME THE FACT YOU HAVE. This is a TREE probe (owner-approved goal): it establishes
        # that the attestation hook is PRESENT in the adapter checkout the run will execute,
        # which is not the same claim as "attestation ran" — a dry run establishes it without a
        # container ever starting. The field is therefore `runtime_attestation_available`, and
        # the enforcement itself still lives where it belongs: the hook fails the run closed on
        # skew / unreachable runtime / unknown commit, before any paid task.
        hook_available = bool(patch_probe["patches"]["clb_docker_runtime_attestation.v6746"])
        manifest["extra"].update({
            "solve_model": _litellm_model(args.model),
            "fidelity": fidelity,
            "adapter_operator_patches": patch_probe,
            "runtime_attestation_path": (
                ("docker engine: operator_patch clb_docker_runtime_attestation.v6746 "
                 "(DockerOuroborosEngine._attest_runtime, post-health, pre-solve)"
                 if hook_available else
                 "NONE: the docker engine never calls IsolatedServer._wait_ready() and operator "
                 "patch clb_docker_runtime_attestation.v6746 is NOT applied in the "
                 "--runner-path ADAPTER checkout (not the --ouroboros-clone seed) — "
                 "THIS RUN IS UNATTESTED")
                if args.docker else
                "host engine: IsolatedServer._wait_ready()"
            ),
            "runtime_attestation_available": (not args.docker) or hook_available,
            "runtime_attestation_evidence": (
                "static_source_scan of the --runner-path adapter checkout" if args.docker
                else "in-tree IsolatedServer._wait_ready (no operator patch involved)"),
            "provider_env_present": redacted_env_summary(child_env),
        })
        if args.docker and not hook_available:
            print("[clb] WARNING: docker path with NO runtime attestation hook — operator patch "
                  "clb_docker_runtime_attestation.v6746 is not applied in "
                  f"{patch_probe['adapter_path']}; the run's provenance records this honestly "
                  "rather than claiming a check that did not run.", file=sys.stderr)

        if args.dry_run:
            final.update({"outcome": "dry_run", "exit_code": 0})
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
        final.update({"outcome": "completed" if rc == 0 else "runner_failed", "exit_code": int(rc)})
        print(f"[clb] run root: {out}")
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
