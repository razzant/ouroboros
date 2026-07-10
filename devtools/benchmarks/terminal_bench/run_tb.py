#!/usr/bin/env python3
"""Terminal-Bench 2.1 runner/submission helper for Ouroboros.

The official leaderboard requires at least k=5 trials, default timeout/resource
settings, metadata.yaml, and full Harbor artifacts. This wrapper keeps those
methodology constraints visible instead of relying on ad-hoc shell history.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import benchmark_run_manifest, write_json
from devtools.benchmarks.common.run_roots import default_settings_path, ensure_outside_repo, repo_root_from_devtools
from devtools.benchmarks.terminal_bench.run_harbor_smoke import AGENT_IMPORT


DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"

# Every model slot the in-container adapter forwards (plus the review triad, handled specially).
# Used by --all-model for a single-model run: review STAYS ON but lightened to ONE reviewer at low
# effort by default (configurable). Exported to os.environ so the harbor subprocess and the adapter's forwarded-slot reads
# (harbor_installed_agent._container_env) pick them up. This does NOT change repo config defaults.
# Only slots the in-container adapter actually forwards (harbor_installed_agent._container_env).
# Deliberately omitted because the container already covers them: HEAVY/CONSCIOUSNESS default to
# OUROBOROS_MODEL when empty, and the adapter pins the fallback (singular AND plural) to the main
# model itself — so setting them here would be a dead host-env write. CLAUDE_CODE_MODEL IS included
# (and the adapter forwards it) so claude_code_edit cannot introduce a different model.
_ALL_MODEL_SLOT_KEYS = (
    "OUROBOROS_MODEL",
    "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    "OUROBOROS_WEBSEARCH_MODEL",
    "OUROBOROS_SCOPE_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
    "CLAUDE_CODE_MODEL",
)


def apply_all_model(model: str, review_slots: int = 1, review_effort: str = "low") -> None:
    """Force every FORWARDED model slot to ``model`` for a single-model run. Mutates only this
    process's env, which propagates to the harbor subprocess and the in-container adapter; it does
    NOT edit repo config defaults. The container's HEAVY/CONSCIOUSNESS slots default to
    OUROBOROS_MODEL and the adapter pins the fallback to the main model, so those need no explicit
    value here.

    Review for a single-model run defaults to ONE reviewer at ``low`` effort (configurable via
    --review-slots / --review-effort): three identical-model reviewers add latency/cost but no
    diversity (a monoculture), and a single-model run cannot achieve reviewer-model diversity anyway.
    This is a BENCHMARK setting, NOT a claim that the review subsystem got more reliable
    (single_reviewer_no_diversity stays loud). EFFORT_SCOPE_REVIEW is set too for completeness
    ("там и там"); scope review does not fire on a terminal-bench task (it is a commit-time gate)."""
    for key in _ALL_MODEL_SLOT_KEYS:
        os.environ[key] = model
    os.environ["OUROBOROS_REVIEW_MODELS"] = ",".join([model] * max(1, int(review_slots)))
    os.environ["OUROBOROS_EFFORT_REVIEW"] = review_effort
    os.environ["OUROBOROS_EFFORT_SCOPE_REVIEW"] = review_effort


@dataclass
class HarborCommandConfig:
    dataset: str
    model: str
    k: int
    jobs_dir: pathlib.Path
    harbor_bin: str
    n_concurrent: int
    task_filters: list[str]
    settings_path: pathlib.Path
    execute: bool
    light_model: str
    review_enforcement: str = "blocking"
    safety_mode: str = "light"
    setup_timeout_multiplier: float = 1.0
    build_timeout_multiplier: float = 1.0
    disable_agent_web: bool = True


def _effective_helper_models(measured_model: str, light_model: str, *, disable_agent_web: bool = False) -> list[tuple[str, str]]:
    """Resolve EVERY model that materially assists a measured run, with its role.

    With ``task_review_mode=required`` the host forces a multi-model
    task-acceptance review whose feedback re-enters the measured agent's
    context, so the review triad / scope / light / web-search models genuinely
    assist the run. Declaring only the measured model in metadata.yaml would
    misrepresent the submission. Values mirror what the container resolves
    (env override else the shipped config defaults) so the declared set matches
    reality. Returns ordered (model_id, role) pairs, deduped by model id.
    """
    review_default = "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.8"
    websearch_default = "gpt-5.2"
    scope_default = "openai/gpt-5.5"
    review = os.environ.get("OUROBOROS_REVIEW_MODELS", review_default) or review_default
    scope = (os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS")
             or os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL") or scope_default)
    websearch = os.environ.get("OUROBOROS_WEBSEARCH_MODEL", websearch_default) or websearch_default
    ordered: list[tuple[str, str]] = [(measured_model, "agent")]
    for m in review.split(","):
        if m.strip():
            ordered.append((m.strip(), "commit_review_triad"))
    for m in scope.split(","):
        if m.strip():
            ordered.append((m.strip(), "scope_review"))
    if light_model.strip():
        ordered.append((light_model.strip(), "light_safety"))
    # Only declare a web_search model if web tools are actually available this run. With
    # disable_agent_web (the default), the agent's web tools are blocked, so declaring a
    # web_search role would misrepresent the submission.
    if websearch.strip() and not disable_agent_web:
        ordered.append((websearch.strip(), "web_search"))
    # The adapter forwards CLAUDE_CODE_MODEL, so claude_code_edit can assist with that model;
    # declare it for honesty. In a single-model (--all-model) run it equals the measured model and
    # dedupes away, so this only adds a row when an operator runs a genuine multi-model ensemble.
    claude_code = os.environ.get("CLAUDE_CODE_MODEL", "").strip()
    if claude_code:
        ordered.append((claude_code, "claude_code_edit"))
    deduped: dict[str, str] = {}
    for model_id, role in ordered:
        if model_id in deduped:
            if role not in deduped[model_id].split("+"):
                deduped[model_id] = deduped[model_id] + "+" + role
        else:
            deduped[model_id] = role
    return list(deduped.items())


def leaderboard_metadata(*, agent_name: str, org_name: str, model: str, light_model: str = "", disable_agent_web: bool = False) -> str:
    lines = [
        "agent_url: https://github.com/razzant/ouroboros",
        f"agent_display_name: {json.dumps(agent_name)}",
        f"agent_org_display_name: {json.dumps(org_name)}",
        "models:",
    ]
    for model_id, role in _effective_helper_models(model, light_model, disable_agent_web=disable_agent_web):
        provider = model_id.split("/", 1)[0] if "/" in model_id else "openrouter"
        display = model_id.split("/", 1)[1] if "/" in model_id else model_id
        lines.append(f"  - model_name: {json.dumps(model_id)}")
        lines.append(f"    model_provider: {json.dumps(provider)}")
        lines.append(f"    model_display_name: {json.dumps(display)}")
        lines.append(f"    model_org_display_name: {json.dumps(provider)}")
        lines.append(f"    role: {json.dumps(role)}")
    return "\n".join(lines) + "\n"


def validate_methodology(
    *,
    k: int,
    timeout_multiplier: float,
    resource_overrides: list[str],
    setup_timeout_multiplier: float = 1.0,
    build_timeout_multiplier: float = 1.0,
    allow_setup_build_multipliers: bool = False,
    allow_low_k: bool = False,
) -> None:
    if int(k) < 5 and not allow_low_k:
        raise ValueError(
            "Terminal-Bench leaderboard mode requires k >= 5. Pass --allow-low-k for a LOCAL, "
            "non-leaderboard-valid measurement run (e.g. a k=1 first pass)."
        )
    if float(timeout_multiplier) != 1.0:
        raise ValueError("Terminal-Bench leaderboard mode requires timeout_multiplier == 1.0")
    if resource_overrides:
        raise ValueError(f"Terminal-Bench leaderboard mode forbids resource overrides: {resource_overrides}")
    # Harbor's leaderboard static_validation REJECTS non-1.0 agent_setup / environment_build
    # timeout multipliers (the top published submissions leave them null). Default them to 1.0;
    # a non-1.0 value is allowed only for an explicitly-flagged LOCAL, non-leaderboard run.
    if not allow_setup_build_multipliers and (
        float(setup_timeout_multiplier) != 1.0 or float(build_timeout_multiplier) != 1.0
    ):
        raise ValueError(
            "Terminal-Bench leaderboard mode requires agent-setup and environment-build timeout "
            f"multipliers == 1.0 (got setup={setup_timeout_multiplier}, build={build_timeout_multiplier}). "
            "Pass --allow-setup-build-multipliers for a LOCAL, non-leaderboard-valid run."
        )


def report_grade(*, k: int, leaderboard_valid: bool, low_k_floor: int = 5) -> tuple[str, str]:
    """Three-tier honesty grade for a TB run, reusing the existing leaderboard_valid
    vocabulary (NOT a parallel taxonomy):
      - leaderboard_valid : k >= 5 AND every leaderboard gate passed.
      - debug_only        : k == 1 (a single trial — noise, not a measurement).
      - local_low_k       : anything else not leaderboard-valid (1 < k < floor, or
                            k >= 5 with a non-faithful setting like web-on).
    Returns (grade, human_warning); warning is '' for a leaderboard-valid run.
    """
    if leaderboard_valid:
        return "leaderboard_valid", ""
    if int(k) <= 1:
        return "debug_only", (
            "⚠️ k=1: a SINGLE trial — this is noise, not a measurement. Do NOT cite this "
            "number; use k>=5 with leaderboard-faithful settings for any reported result."
        )
    if int(k) < int(low_k_floor):
        return "local_low_k", (
            f"⚠️ k={k} (< {low_k_floor}): LOCAL low-confidence run, NOT leaderboard-valid — "
            "high variance. Use k>=5 with leaderboard-faithful settings for any reported number."
        )
    # k >= floor but NOT leaderboard-valid: the reason is an off-spec leaderboard setting
    # (agent web enabled / timeout multiplier / resource override), NOT low-k variance — so the
    # warning must NOT claim "k < floor".
    return "local_low_k", (
        f"⚠️ k={k} is >= {low_k_floor} but this run is NOT leaderboard-valid — a leaderboard-faithful "
        "setting is off-spec (agent web enabled / timeout multiplier / resource override). "
        "Do not report it as a leaderboard number."
    )


def harbor_command(config: HarborCommandConfig) -> list[str]:
    cmd = [
        config.harbor_bin,
        "run",
        "--dataset",
        config.dataset,
        "--agent-import-path",
        AGENT_IMPORT,
        "--model",
        f"ouroboros-{config.model.replace('/', '-')}",
        "--agent-kwarg",
        f"ouroboros_model={config.model}",
        "--agent-kwarg",
        f"ouroboros_light_model={config.light_model}",
        "--agent-kwarg",
        f"host_settings_path={config.settings_path}",
        "--agent-kwarg",
        "task_review_mode=required",
        "--agent-kwarg",
        f"review_enforcement={config.review_enforcement}",
        "--agent-kwarg",
        f"safety_mode={config.safety_mode}",
        "--agent-kwarg",
        "install_timeout_sec=1200",
        "--agent-kwarg",
        "server_start_timeout_sec=240",
        "--agent-kwarg",
        f"disable_agent_web={str(bool(config.disable_agent_web)).lower()}",
        "--n-concurrent",
        str(int(config.n_concurrent)),
        "-k",
        str(int(config.k)),
        "--jobs-dir",
        str(config.jobs_dir),
        "--yes",
    ]
    # Optional host pip wheel cache (opt-in via OBO_TB_PIP_CACHE): bind-mount a durable host dir at
    # /opt/ouro-pip-cache in every task container so the per-trial Ouroboros pip install hits cached
    # wheels instead of the network (offline-fast, resilient to mirror drops). Read-write + shared is
    # safe: pip keys wheels by py/platform tag and writes via atomic rename of identical content, so
    # heterogeneous task images and n-concurrent trials populate/reuse one cache without conflict.
    # This is NOT a leaderboard-config field (it's a deploy mount, like --n-concurrent), so it does
    # not affect static_validation. Unset → no --mounts emitted → behavior unchanged.
    pip_cache = os.environ.get("OBO_TB_PIP_CACHE", "").strip()
    if pip_cache:
        cache_dir = ensure_outside_repo(pathlib.Path(pip_cache), repo_root_from_devtools())
        mounts = [{"type": "bind", "source": str(cache_dir), "target": "/opt/ouro-pip-cache"}]
        cmd.extend(["--mounts", json.dumps(mounts)])
    # Setup/build timeout multipliers default to 1.0 (Harbor static_validation rejects non-1.0
    # agent_setup / environment_build multipliers). Emit the flags ONLY when an explicit
    # local-only override is set; a leaderboard-valid run leaves them absent (== Harbor default).
    if float(config.setup_timeout_multiplier) != 1.0:
        cmd.extend(["--agent-setup-timeout-multiplier", str(float(config.setup_timeout_multiplier))])
    if float(config.build_timeout_multiplier) != 1.0:
        cmd.extend(["--environment-build-timeout-multiplier", str(float(config.build_timeout_multiplier))])
    for task in config.task_filters:
        cmd.extend(["--include-task-name", task])
    if config.execute:
        cmd.append("--force-build")
    return cmd


def write_disclosure_ledger(*, jobs_dir: pathlib.Path, out_path: pathlib.Path, run_meta: dict) -> dict:
    """Walk Harbor's jobs dir, read every trial result.json, and write a denominator-preserving
    disclosure ledger: reward distribution, exception_info histogram (AgentTimeoutError /
    ApiRateLimitError counts -- the latency / rate-limit failure signal Anton asked for), per-task
    pass rate, cost and wall-time. Honest-by-construction so a write-up can disclose how many
    trials failed by timeout vs rate-limit at a given concurrency. Best-effort: never raises into
    the harbor result path (the caller guards it)."""
    import collections

    trials: list[dict] = []
    for result_json in sorted(pathlib.Path(jobs_dir).rglob("result.json")):
        try:
            data = json.loads(result_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if "task_name" not in data and "verifier_result" not in data:
            continue  # not a trial result.json (e.g. a job-level file)
        reward = None
        verifier_result = data.get("verifier_result")
        if isinstance(verifier_result, dict) and isinstance(verifier_result.get("rewards"), dict):
            reward = verifier_result["rewards"].get("reward")
        exception_info = data.get("exception_info")
        exception_type = exception_info.get("exception_type") if isinstance(exception_info, dict) else None
        agent_result = data.get("agent_result") if isinstance(data.get("agent_result"), dict) else {}
        agent_meta = agent_result.get("metadata") if isinstance(agent_result.get("metadata"), dict) else {}
        # The adapter records the in-container Ouroboros outcome (reason_code/infra_failed) here.
        # Harbor's own exception_info is null for a clean agent exit even when Ouroboros failed on a
        # provider issue (e.g. reason_code="provider_unavailable"), so the truthful provider/infra
        # signal lives in this summary, NOT in Harbor's exception_info.
        adapter_summary = agent_meta.get("summary") if isinstance(agent_meta.get("summary"), dict) else {}
        # The adapter's result.json summary carries reason_code/infra_failed but NOT the
        # cancellation signal. captured_after_cancellation lives in the sibling run-summary,
        # written by the adapter's teardown-snapshot path. It is the load-bearing signal that a
        # terminal reason_code="provider_unavailable" is only a post-cancellation network-teardown
        # artifact (the harness cuts container egress at the finish line, so the agent's own
        # finalization/summary LLM calls die on DNS) rather than a real mid-run provider fault.
        captured_after_cancellation = False
        run_summary_path = result_json.parent / "agent" / "ouroboros-run-summary.json"
        try:
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            if isinstance(run_summary, dict):
                captured_after_cancellation = bool(run_summary.get("captured_after_cancellation"))
        except (OSError, ValueError):
            pass
        trials.append({
            "task_name": data.get("task_name"),
            "trial_name": data.get("trial_name"),
            "reward": reward,
            "exception_type": exception_type,
            "reason_code": adapter_summary.get("reason_code"),
            "infra_failed": bool(adapter_summary.get("infra_failed")),
            "captured_after_cancellation": captured_after_cancellation,
            "cost_usd": agent_result.get("cost_usd"),
            "turns": agent_meta.get("turns"),
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
        })

    def _reward_key(reward: object) -> str:
        if reward in (1, 1.0, True):
            return "1.0"
        if reward in (0, 0.0, False):
            return "0.0"
        return "null" if reward is None else str(reward)

    reward_distribution = collections.Counter(_reward_key(t["reward"]) for t in trials)
    exception_histogram = collections.Counter(t["exception_type"] for t in trials if t["exception_type"])
    reason_code_histogram = collections.Counter(t["reason_code"] for t in trials if t["reason_code"])
    # The honest provider/rate-limit/infra signal combines THREE sources: a Harbor exception that is
    # not a timeout (RuntimeError/ApiRateLimitError/NonZeroAgentExitCodeError), the adapter's
    # infra_failed flag, OR an Ouroboros reason_code denoting a provider fault (which Harbor records
    # as a clean reward-0 trial with exception_info=null). Without this a 429-driven failure is
    # indistinguishable from a genuine wrong answer.
    _timeout_types = {"AgentTimeoutError", "VerifierTimeoutError"}
    _provider_reasons = {"provider_unavailable", "llm_api_error", "rate_limited", "provider_error"}

    def _failure_category(t: dict) -> str:
        """Classify each trial into exactly one honest bucket (precedence matters).

        'pass'           reward 1.0 -- verifier-confirmed success, regardless of Ouroboros's own
                         terminal status (Ouroboros may self-report failed/provider_unavailable when
                         a post-work finalization call dies on the harness network teardown, yet the
                         work in the container already passed the external verifier).
        'provider_infra' a provider/infra fault that PREVENTED a fair attempt: any non-timeout Harbor
                         exception (setup timeout, nonzero exit, transport), infra_failed, OR a
                         provider reason_code seen WITHOUT the post-cancellation flag (a genuine
                         mid-run provider/network death).
        'cancelled'      the agent was stopped by the harness before producing a real terminal: a
                         Harbor wall-clock timeout, OR a provider reason_code stamped only during
                         post-cancellation network teardown (captured_after_cancellation) -- the
                         harness cut egress at the finish line, the agent's own finalization LLM call
                         died on DNS, and the real outcome was masked.
        'genuine'        reward 0 having reached a real terminal (final_message / tool_failure / ...)
                         -- a fair-shot wrong answer. NOTE: captured_after_cancellation is NOT used to
                         route here; it is set broadly on teardown, including on trials that DID emit
                         a final answer, so trusting it alone would wrongly excuse real wrong answers.
        'other'          reward is null/other and none of the above.
        """
        if t["reward"] in (1, 1.0, True):
            return "pass"
        et = t.get("exception_type")
        if et in _timeout_types:
            return "cancelled"          # Harbor wall-clock timeout: cut off before finishing.
        if et:
            return "provider_infra"     # setup timeout / nonzero exit / transport: real infra fault.
        if t.get("infra_failed"):
            return "provider_infra"
        if t.get("reason_code") in _provider_reasons:
            # Provider reason WITH the teardown flag == masked harness cancellation, not a real
            # provider fault; WITHOUT it == a genuine mid-run provider/network death.
            return "cancelled" if t.get("captured_after_cancellation") else "provider_infra"
        if t["reward"] in (0, 0.0, False):
            return "genuine"            # reached a real terminal with reward 0 -> wrong answer.
        return "other"

    categories = collections.Counter(_failure_category(t) for t in trials)
    provider_or_infra_failures = categories["provider_infra"]
    per_task: dict[str, dict] = collections.defaultdict(lambda: {"n": 0, "passed": 0})
    for trial in trials:
        bucket = per_task[trial["task_name"] or "?"]
        bucket["n"] += 1
        if trial["reward"] in (1, 1.0):
            bucket["passed"] += 1
    per_task_pass_rate = {
        task: (b["passed"] / b["n"] if b["n"] else 0.0) for task, b in sorted(per_task.items())
    }
    costs = [t["cost_usd"] for t in trials if isinstance(t["cost_usd"], (int, float))]
    ledger = {
        "schema": "tb_disclosure_ledger.v1",
        "run": dict(run_meta),
        "n_trials": len(trials),
        "n_tasks": len(per_task),
        "reward_distribution": dict(reward_distribution),
        "exception_histogram": dict(exception_histogram),
        "reason_code_histogram": dict(reason_code_histogram),
        "agent_timeout_count": int(exception_histogram.get("AgentTimeoutError", 0)),
        "api_rate_limit_error_count": int(exception_histogram.get("ApiRateLimitError", 0)),
        "provider_or_infra_failure_count": int(provider_or_infra_failures),
        "wall_clock_cancellation_count": int(categories["cancelled"]),
        "genuine_failure_count": int(categories["genuine"]),
        "exception_note": (
            "Honest taxonomy: every reward-0 trial is exactly one of provider_or_infra_failure / "
            "wall_clock_cancellation / genuine_failure (reward-1 trials are 'pass'). agent_timeout_count "
            "is the Harbor-named subset. A provider reason_code (e.g. provider_unavailable) only counts as "
            "provider_or_infra when it was NOT a post-cancellation teardown artifact: the harness cuts "
            "container egress at the finish line, so an agent's own finalization/summary LLM calls die on "
            "DNS and get stamped provider_unavailable even though the task already passed or was merely cut "
            "off by wall-clock -- those go to wall_clock_cancellation (captured_after_cancellation), NOT "
            "provider_or_infra. genuine_failure_count is reward-0 given a fair shot (real wrong answers, "
            "not provider artifacts and not wall-clock cut-offs)."
        ),
        "total_cost_usd": round(sum(costs), 4) if costs else None,
        "per_task_pass_rate": per_task_pass_rate,
        "trials": trials,
    }
    write_json(out_path, ledger)
    print(
        f"[run_tb] disclosure ledger: {len(trials)} trials, "
        f"{ledger['agent_timeout_count']} AgentTimeoutError, "
        f"{ledger['provider_or_infra_failure_count']} provider/infra, "
        f"{ledger['wall_clock_cancellation_count']} wall-clock-cancelled, "
        f"{ledger['genuine_failure_count']} genuine failures -> {out_path}"
    )
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default="", help="measured/declared model; or use --all-model to set every slot")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--low-k-floor", type=int, default=5, help="k below this is graded local/low-confidence (debug_only at k=1) in report_grade (default 5).")
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--task", action="append", default=[], help="optional include-task-name; repeatable")
    parser.add_argument("--review-enforcement", default="blocking", choices=["blocking", "advisory"],
                        help="in-task review enforcement mode forwarded to the container (default blocking)")
    parser.add_argument("--safety-mode", default="light", choices=["full", "light", "off"],
                        help="LLM safety mode inside the task container (default light; off disables the LLM safety pass, deterministic guards stay)")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--submission-root", default="")
    parser.add_argument("--settings-path", default="")
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--light-model", default="google/gemini-3.5-flash")
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--resource-override", action="append", default=[])
    parser.add_argument("--agent-name", default="Ouroboros")
    parser.add_argument("--org-name", default="Ouroboros")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--setup-timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--build-timeout-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--allow-setup-build-multipliers",
        action="store_true",
        help="LOCAL-only: permit non-1.0 setup/build multipliers (run is NOT leaderboard-valid)",
    )
    parser.add_argument(
        "--allow-low-k",
        action="store_true",
        help="LOCAL-only: permit k<5 (e.g. a k=1 first pass); run is NOT leaderboard-valid",
    )
    parser.add_argument(
        "--disable-agent-web",
        dest="disable_agent_web",
        action="store_true",
        default=True,
        help="block the in-container agent's web/browser tools (default; reward-hacking guard)",
    )
    parser.add_argument(
        "--allow-agent-web",
        dest="disable_agent_web",
        action="store_false",
        help="LOCAL-only: allow the agent's web tools (NOT leaderboard-faithful)",
    )
    parser.add_argument(
        "--all-model",
        default="",
        help="convenience: set ALL model slots (incl. review/scope/websearch) to this model",
    )
    parser.add_argument(
        "--review-slots",
        type=int,
        default=1,
        help="number of in-task acceptance reviewers for --all-model (default 1; 3 identical = monoculture, no diversity)",
    )
    parser.add_argument(
        "--review-effort",
        default="low",
        choices=["none", "low", "medium", "high"],
        help="reasoning effort for the in-task review under --all-model (default low; cuts the review-latency tax)",
    )
    args = parser.parse_args(argv)

    if args.all_model:
        apply_all_model(args.all_model, review_slots=args.review_slots, review_effort=args.review_effort)
        args.model = args.all_model
        args.light_model = args.all_model
    if not args.model:
        parser.error("either --model or --all-model is required")

    validate_methodology(
        k=args.k,
        timeout_multiplier=args.timeout_multiplier,
        resource_overrides=list(args.resource_override or []),
        setup_timeout_multiplier=args.setup_timeout_multiplier,
        build_timeout_multiplier=args.build_timeout_multiplier,
        allow_setup_build_multipliers=args.allow_setup_build_multipliers,
        allow_low_k=args.allow_low_k,
    )
    if not args.disable_agent_web:
        print(
            "WARNING: --allow-agent-web leaves the agent's web tools ENABLED; this run is NOT "
            "leaderboard-faithful (reward-hacking guard off).",
            file=sys.stderr,
        )

    repo = repo_root_from_devtools()
    run_root = ensure_outside_repo(
        pathlib.Path(args.run_root).expanduser() if args.run_root else pathlib.Path.cwd() / "tb21_ouroboros_run",
        repo,
    )
    settings_path = pathlib.Path(args.settings_path).expanduser() if args.settings_path else default_settings_path()
    submission_root = ensure_outside_repo(
        pathlib.Path(args.submission_root).expanduser()
        if args.submission_root
        else run_root / "submission",
        repo,
    )
    job_dir = submission_root / "submissions" / "terminal-bench" / "2.1" / f"ouroboros__{args.model.replace('/', '-')}" / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = job_dir.parent / "metadata.yaml"
    metadata_path.write_text(
        leaderboard_metadata(agent_name=args.agent_name, org_name=args.org_name, model=args.model, light_model=args.light_model, disable_agent_web=bool(args.disable_agent_web)),
        encoding="utf-8",
    )

    cmd = harbor_command(HarborCommandConfig(
        dataset=args.dataset,
        model=args.model,
        k=args.k,
        jobs_dir=job_dir,
        harbor_bin=args.harbor_bin,
        n_concurrent=args.n_concurrent,
        task_filters=list(args.task or []),
        settings_path=settings_path,
        execute=bool(args.execute),
        light_model=args.light_model,
        review_enforcement=args.review_enforcement,
        safety_mode=args.safety_mode,
        setup_timeout_multiplier=args.setup_timeout_multiplier,
        build_timeout_multiplier=args.build_timeout_multiplier,
        disable_agent_web=bool(args.disable_agent_web),
    ))
    leaderboard_valid = bool(
        int(args.k) >= 5
        and float(args.timeout_multiplier) == 1.0
        and float(args.setup_timeout_multiplier) == 1.0
        and float(args.build_timeout_multiplier) == 1.0
        and bool(args.disable_agent_web)
        and not list(args.resource_override or [])
    )
    # D: 3-tier honesty grade + warning, printed on ALL paths (incl. dry-run / command-gen,
    # which returns before --execute) so a low-k run is never silently cited as a measurement.
    report_grade_value, report_grade_warning = report_grade(
        k=int(args.k), leaderboard_valid=leaderboard_valid, low_k_floor=int(args.low_k_floor)
    )
    if report_grade_warning:
        print(report_grade_warning, file=sys.stderr)
    write_json(
        run_root / "run_manifest.json",
        benchmark_run_manifest(
            benchmark="terminal_bench",
            run_root=run_root,
            repo_dir=repo,
            requested_task_ids=list(args.task or []),
            metadata={
                # dataset + official_command are promoted to top-level by benchmark_run_manifest;
                # everything else must go through `extra` (the helper drops unknown top-level keys).
                "dataset": args.dataset,
                "official_command": cmd,
                "extra": {
                    "k": int(args.k),
                    "n_concurrent": int(args.n_concurrent),
                    "timeout_multiplier": float(args.timeout_multiplier),
                    "setup_timeout_multiplier": float(args.setup_timeout_multiplier),
                    "build_timeout_multiplier": float(args.build_timeout_multiplier),
                    "resource_overrides": list(args.resource_override or []),
                    "disable_agent_web": bool(args.disable_agent_web),
                    "all_model": args.all_model or "",
                    "leaderboard_valid": leaderboard_valid,
                    "report_grade": report_grade_value,
                    "report_grade_warning": report_grade_warning,
                    "leaderboard_submission_root": str(submission_root),
                    "metadata_yaml": str(metadata_path),
                },
            },
        ),
    )
    (run_root / "harbor_command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    print(shlex.join(cmd))
    if not args.execute:
        return 0
    completed = subprocess.run(cmd, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)})
    try:
        write_disclosure_ledger(
            jobs_dir=job_dir,
            out_path=run_root / "disclosure_ledger.json",
            run_meta={
                "dataset": args.dataset,
                "k": int(args.k),
                "n_concurrent": int(args.n_concurrent),
                "disable_agent_web": bool(args.disable_agent_web),
                "setup_timeout_multiplier": float(args.setup_timeout_multiplier),
                "build_timeout_multiplier": float(args.build_timeout_multiplier),
                "leaderboard_valid": leaderboard_valid,
                "report_grade": report_grade_value,
                "report_grade_warning": report_grade_warning,
                "model": args.model,
                "model_provider_prefix": (args.model.split("/", 1)[0] if "/" in args.model else ""),
                "all_model": args.all_model or "",
                "harbor_returncode": int(completed.returncode),
            },
        )
    except Exception as exc:  # disclosure is best-effort; never mask the harbor result
        print(f"[run_tb] disclosure ledger skipped: {exc!r}", file=sys.stderr)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
