#!/usr/bin/env python3
"""Run Cowork Bench with full Ouroboros inside the benchmark's own task containers.

The benchmark's official ``run_parallel.sh`` stays the execution authority (per-task network,
fresh Postgres, agent container, separate evaluator container with the groundtruth) and its
per-task ``evaluation/main.py`` stays the scoring authority. This launcher only prepares a
pinned per-run working copy, a derived image and the run config, invokes the official runner
with ``AGENT_ENTRY=main_ouroboros.py AGENT_PHASE_AWARE=1`` and writes the Ouroboros sidecars
(run manifest + denominator-preserving result ledger).
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import math
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import (
    admit_benchmark_run,
    finalize_run_manifest,
    openrouter_account_credits,
    openrouter_key_remaining,
    write_json,
)
from devtools.benchmarks.common.model_slots import fixed_model_actor_snapshot
from devtools.benchmarks.common.result_index import (
    RUNTIME_TRUNCATION_REASON_CODES,
    task_result_row,
    write_result_index,
)
from devtools.benchmarks.common.run_roots import assert_outside_repo, repo_root_from_devtools, run_root, timestamp_run_id
from devtools.benchmarks.common.secrets import credential_fingerprint
from devtools.benchmarks.cowork_bench.audit_cowork_bench import model_activity_observed
from devtools.benchmarks.cowork_bench.campaign import (
    CampaignBudget,
    CampaignPersistenceError,
    campaign_lock,
    key_usage,
    validate_usage,
)
from devtools.benchmarks.cowork_bench.eval_attempt import FLAG as DIAGNOSTIC_FLAG
from devtools.benchmarks.cowork_bench.eval_attempt import attempt_facts, claim_protocol
from devtools.benchmarks.cowork_bench.official_receipt import read_linked_runtime_result, read_official_receipt
from devtools.benchmarks.cowork_bench.resource_limits import LABEL_KEY, prepare_resource_env
from ouroboros.platform_layer import kill_process_group_id, terminate_process_group_id
from ouroboros.process_custody import spawn_supervised

BENCHMARK = "cowork_bench"
ENGINE = "ouroboros"
PINNED_BENCH_COMMIT = "d943e75bc0fc8e3b27141979300cd8cbcd1e890d"
DEFAULT_MODEL = "moonshotai/kimi-k3"
CONTAINER_DIR = pathlib.Path(__file__).with_name("container")
# Stdlib helpers the image copies beside the entrypoint (the eval phase imports them there).
CONTAINER_HELPERS = tuple(pathlib.Path(__file__).with_name(name) for name in ("eval_attempt.py", "official_receipt.py"))
SETTINGS_TEMPLATE = pathlib.Path(__file__).with_name("settings_base.json")
CONFIG_NAME = "ouroboros_bench.json"
SECRET_NAME = "ouroboros_bench.secret.json"
# The agent's own LLM-backed web/vision tools stay off: the benchmark's answer keys live in a
# public GitHub repository, and the reference engines have no such tools. Mirrors the
# Terminal-Bench adapter's web-off set; a sync test pins the mirror.
WEB_TOOLS = ("web_search", "browse_page", "browser_action", "youtube_transcript")
DELEGATED_VISION_TOOLS = ("analyze_screenshot", "vlm_query")
SETTLED_STATUSES = frozenset({"passed", "failed", "agent_failed"})
# Meter cadence. The blindness bound itself is the operator's recorded
# --meter-blindness-sec; one read's slice keeps a slow read from consuming it.
DEFAULT_METER_BLINDNESS_SEC = 30.0
METER_POLL_SEC = 15.0
METER_RETRY_SEC = 3.0
METER_READ_SLICE_SEC = 5.0
# Adapter-owned artifacts that exist only after this task's container began work.
# The benchmark pre-creates the dump/workspace; evaluator traj logs and receipts
# alone also do not prove the agent phase started.
START_EVIDENCE = ("ouroboros", "applied_settings.json", "preprocess.log", "mcp_proxy.log",
                  "ouroboros_server.log")


def dump_dir_name(model: str) -> str:
    """The benchmark's ``TaskConfig`` flattens ``<engine>/<model>`` into one dump directory."""
    return f"{ENGINE}/{model}".replace("/", "_")


def disabled_tools(*, subagents: bool) -> list[str]:
    tools = [*WEB_TOOLS, *DELEGATED_VISION_TOOLS, "claude_code_edit"]
    if not subagents:
        tools.append("schedule_subagent")
    return tools


def render_settings(template: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pin every execution route to the measured model and apply the disclosed scaffold knobs.

    Returns ``(settings, fixed_model_actor)``; the actor snapshot is compiled from the SAME
    mapping the container receives, so the manifest cannot name a model the run did not use.
    """
    settings = {key: value for key, value in template.items() if not key.endswith("_API_KEY")}
    actor = fixed_model_actor_snapshot(
        args.model, review_slots=args.review_slots, review_effort=args.review_effort, target=settings
    )
    settings.update(
        {
            "OUROBOROS_EFFORT_TASK": args.effort,
            "OUROBOROS_TASK_REVIEW_MODE": args.task_review_mode,
            "OUROBOROS_REVIEW_ENFORCEMENT": args.review_enforcement,
            "OUROBOROS_REVIEW_MAX_CYCLES": str(args.review_max_cycles),
            "OUROBOROS_SAFETY_MODE": args.safety_mode,
            "OUROBOROS_RUNTIME_MODE": args.runtime_mode,
            "OUROBOROS_MAX_ROUNDS": int(args.max_steps),
            "OUROBOROS_MAX_WORKERS": int(args.max_workers),
            "OUROBOROS_MAX_SUBAGENT_DEPTH": int(args.subagent_depth) if args.subagents else 0,
            "OUROBOROS_PER_TASK_COST_USD": float(args.per_task_cost_usd),
            "TOTAL_BUDGET": float(args.per_task_cost_usd),
            "OUROBOROS_POST_TASK_EVOLUTION": "false",
        }
    )
    if args.or_provider:
        settings["OUROBOROS_OR_PROVIDER"] = args.or_provider
    return settings, actor


def bench_config(args: argparse.Namespace, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ouroboros.cowork_bench.run_config.v1",
        "model": args.model,
        "settings": settings,
        "disabled_tools": disabled_tools(subagents=args.subagents),
        "task_timeout_sec": int(args.task_timeout),
        "mcp_tool_timeout_sec": int(args.mcp_tool_timeout),
        "proxy_port": 8096,
        "server_port": 8765,
        "truncation_reason_codes": sorted(RUNTIME_TRUNCATION_REASON_CODES),
        DIAGNOSTIC_FLAG: bool(args.diagnostic_eval_on_truncation),
    }


def with_diagnostic_default(config: Any) -> Any:
    """Runs recorded before the opt-in flag existed ran without it; nothing else is normalized."""
    return {**config, DIAGNOSTIC_FLAG: False} if isinstance(config, dict) and DIAGNOSTIC_FLAG not in config else config


def _git(path: pathlib.Path, *argv: str) -> str:
    proc = subprocess.run(["git", "-C", str(path), *argv], text=True, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(argv)} failed in {path}: {proc.stderr.strip()[:400]}")
    return proc.stdout.strip()


def bench_provenance(bench_root: pathlib.Path, expected_commit: str) -> dict[str, Any]:
    head = _git(bench_root, "rev-parse", "HEAD")
    dirty = bool(_git(bench_root, "status", "--porcelain"))
    record = {"bench_root": str(bench_root), "head": head, "expected": expected_commit, "dirty": dirty}
    if head != expected_commit or dirty:
        raise RuntimeError(f"benchmark checkout is not the clean pinned commit: {record}")
    return record


def _docker(docker_host: str, *argv: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "DOCKER_HOST": docker_host}
    return subprocess.run(["docker", *argv], env=env, text=True, capture_output=True, timeout=timeout)


def image_exists(docker_host: str, image: str) -> bool:
    return _docker(docker_host, "image", "inspect", image, timeout=60).returncode == 0


def run_preparation(command: list[str], args: argparse.Namespace, log_path: pathlib.Path, timeout: int) -> None:
    """Keep image preparation inside the same storage reserve as task execution."""
    with log_path.open("a", encoding="utf-8") as handle:
        proc = spawn_supervised(command, drive_root=log_path.parent, scope="session",
                                purpose="cowork-image-preparation", stdout=handle, stderr=subprocess.STDOUT,
                                env={**os.environ, "DOCKER_HOST": args.docker_host}, text=True)
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if (shutil.disk_usage(args.resource_root).free < args.min_free_gib * 1024**3
                        or shutil.disk_usage(pathlib.Path.home().anchor).free < args.min_root_free_gib * 1024**3):
                    raise RuntimeError("image preparation stopped at the disk reserve")
                if time.monotonic() >= deadline:
                    raise TimeoutError("image preparation exceeded its existing time limit")
                try:
                    proc.wait(timeout=min(15, max(0.1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if proc.returncode:
                raise RuntimeError(f"image preparation failed (exit {proc.returncode}); see {log_path}")
        finally:
            stop_process_group(proc)


def build_image(args: argparse.Namespace, seed_head: str, bench_head: str, log_path: pathlib.Path) -> None:
    """Build the derived image from a throwaway context holding a full clone of the seed."""
    with tempfile.TemporaryDirectory(prefix="cowork-image-", dir=log_path.parent / "tmp") as raw:
        context = pathlib.Path(raw)
        run_preparation(
            ["git", "clone", "--quiet", "--no-hardlinks", "--single-branch", str(args.repo_dir), str(context / "seed")],
            args, log_path, timeout=600,
        )
        _git(context / "seed", "checkout", "--quiet", "--detach", seed_head)
        shutil.copy2(CONTAINER_DIR / "main_ouroboros.py", context / "main_ouroboros.py")
        for helper in CONTAINER_HELPERS:
            shutil.copy2(helper, context / helper.name)
        shutil.copy2(CONTAINER_DIR / "Dockerfile", context / "Dockerfile")
        command = [
            "docker", "build", "-t", args.image,
            "--build-arg", f"BASE_IMAGE={args.base_image}",
            "--build-arg", f"SEED_SHA={seed_head}",
            "--build-arg", f"BENCH_SHA={bench_head}",
            str(context),
        ]
        run_preparation(command, args, log_path, timeout=7200)


def image_identity(docker_host: str, image: str) -> dict[str, Any]:
    """Resolve the exact local image, including builds without a registry RepoDigest."""
    proc = _docker(docker_host, "image", "inspect", "--format", "{{json .}}", image, timeout=60)
    if proc.returncode:
        raise RuntimeError(f"cannot inspect image {image!r}")
    record = json.loads(proc.stdout)
    if not isinstance(record, dict) or not record.get("Id"):
        raise RuntimeError("Docker did not return an immutable image identity")
    return {"id": record["Id"], "labels": (record.get("Config") or {}).get("Labels") or {},
            "repo_digests": record.get("RepoDigests") or []}

def key_headroom(api_key: str) -> dict[str, Any]:
    """Both bounds of an OpenRouter key: the key limit is not money, the account balance is."""
    remaining = openrouter_key_remaining(api_key)
    balance = openrouter_account_credits(api_key)
    bounds = [value for value in (remaining, balance) if value is not None]
    return {"key_limit_remaining": remaining, "account_balance": balance, "effective": min(bounds) if bounds else None}


def discover_tasks(bench_dir: pathlib.Path) -> list[str]:
    pool = bench_dir / "tasks" / "finalpool"
    return sorted(path.parent.name for path in pool.glob("*/task_config.json") if not path.parent.name.startswith("."))


def read_task_file(path: pathlib.Path) -> list[str]:
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def settled_tasks(run_roots: list[pathlib.Path]) -> set[str]:
    """Tasks a previous run already settled with a genuine outcome (infra rows stay open)."""
    settled: set[str] = set()
    for root in run_roots:
        ledger = root / "result_index.jsonl"
        try:
            contents = ledger.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"resume ledger unavailable; inspect the previous run: {ledger}") from exc
        for line in contents.splitlines():
            row = json.loads(line)
            if str(row.get("status") or "") in SETTLED_STATUSES:
                settled.add(str(row.get("instance_id") or ""))
    return settled


def read_summary_csv(log_root: pathlib.Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for summary in sorted(log_root.glob("fully_parallel_*/summary.csv")):
        with summary.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[str(row.get("task") or "")] = dict(row)
    return rows


def _load(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def ledger_row(task: str, task_dump: pathlib.Path, runner_row: dict[str, str],
               *, protocol: str, cause: str = "in_progress") -> dict[str, Any]:
    """One denominator-preserving row. The runner's exit code and CSV are NOT the status: the
    adapter summary says how the agent phase ended and ``eval_res.json`` is the verdict.

    Without that summary, only a task with no runner row and no start evidence is
    ``not_attempted``. Otherwise it is an infrastructure row whose paid activity is
    ``observed`` from token-bearing usage or else ``unknown``, never an invented cost.

    The official receipt is attached on EVERY branch, independently of that status: a
    timed-out agent phase keeps the evaluator's own record instead of a claimed ``not_run``,
    and only a literal boolean verdict on a successful agent phase is scored.
    A caller without a terminal cause sees a provisional snapshot, not an
    invented assertion that the runner already exited."""
    summary = _load(task_dump / "ouroboros_summary.json")
    receipt, eval_res = read_official_receipt(task_dump)
    runtime_result, runtime_source = read_linked_runtime_result(task_dump, summary)
    official = receipt["official_eval_status"]
    attempt = attempt_facts(task_dump)
    if attempt.get("official_run") is False:
        # This run's claimed attempt proved it never called the evaluator (for example a
        # preserved unclaimed eval_res.json), so no file present there is its verdict.
        official, eval_res = "not_run", None
    elif (attempt["state"] != "absent" and not attempt.get("file_matches_returned")) or (
            attempt["state"] == "absent" and protocol != "legacy"):
        # A missing claim in a CURRENT run may mean admission was refused before
        # entrypoint start. Only a positively identified legacy manifest permits
        # the pre-protocol file reader to supply a verdict.
        official = "unknown" if official in {"completed", "declined"} else official
        eval_res = None
    paths = {"task_dump": str(task_dump)}
    details: dict[str, Any] = {"runner": runner_row, "adapter": summary, "official_receipt": receipt,
                               "official_attempt": {**attempt, "protocol": protocol},
                               "runtime_result_source": runtime_source}
    runtime = {"runtime_result": runtime_result}
    if runner_row.get("status") == "pg_fail":
        return task_result_row(benchmark=BENCHMARK, instance_id=task, status="infra_failed",
                               reason_code="pg_fail", output_paths=paths, details=details,
                               official_eval_status=official, **runtime)
    if not summary:
        evidence = [name for name in START_EVIDENCE if (task_dump / name).exists()]
        if not (runner_row or evidence):
            return task_result_row(benchmark=BENCHMARK, instance_id=task, status="not_attempted",
                                   reason_code="missing_result", output_paths=paths, details=details,
                                   official_eval_status=official, **runtime)
        observed = model_activity_observed(task_dump / "ouroboros" / "events.jsonl")
        details.update({"start_evidence": evidence, "paid_activity": "observed" if observed else "unknown"})
        # A live diagnostic snapshot is not proof of interruption. The legacy infra
        # bucket stays provisional until the run stops; it is never a settled outcome.
        details["provisional"] = cause == "in_progress"
        return task_result_row(benchmark=BENCHMARK, instance_id=task, status="infra_failed",
                               reason_code="missing_adapter_summary" if cause == "in_progress" else f"interrupted:{cause}",
                               output_paths=paths, details=details, official_eval_status=official, **runtime)
    reason = str(summary.get("reason_code") or "")
    if summary.get("infra_failed"):
        return task_result_row(benchmark=BENCHMARK, instance_id=task, status="infra_failed",
                               reason_code=reason or "infra_failed", output_paths=paths,
                               error=str(summary.get("error") or ""), details=details,
                               official_eval_status=official, **runtime)
    if summary.get("bench_status") != "success":
        return task_result_row(benchmark=BENCHMARK, instance_id=task, status="agent_failed",
                               reason_code=reason or "agent_not_finished", output_paths=paths,
                               details=details, official_eval_status=official, **runtime)
    if eval_res is None:
        # No literal boolean verdict: never coerce a string/number `pass` into a score.
        return task_result_row(benchmark=BENCHMARK, instance_id=task, status="infra_failed",
                               reason_code="official_eval_not_run" if official == "not_run" else "missing_eval_result",
                               output_paths=paths, details=details, official_eval_status=official, **runtime)
    passed = receipt["pass"]
    return task_result_row(
        benchmark=BENCHMARK, instance_id=task, status="passed" if passed else "failed",
        reason_code="passed" if passed else "verifier_failed", official_eval_status="completed",
        output_paths=paths, details={**details, "eval": eval_res}, **runtime,
    )


def write_ledger(ledger_path: pathlib.Path, bench_dir: pathlib.Path, model: str, tasks: list[str],
                 *, cause: str = "in_progress") -> dict[str, int]:
    """``cause`` names why a started task without a summary did not finish: the run's stop
    reason, ``runner_exited``, or ``in_progress`` in a snapshot taken while the run is live."""
    runner_rows = read_summary_csv(bench_dir / "benchmark_logs")
    dumps = bench_dir / "dumps" / dump_dir_name(model)
    protocol = claim_protocol(bench_dir.parent)
    rows = [ledger_row(task, dumps / f"SingleUserTurn-{task}", runner_rows.get(task, {}),
                       cause=cause, protocol=protocol)
            for task in tasks]
    write_result_index(ledger_path, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def select_tasks(bench_dir: pathlib.Path, args: argparse.Namespace, config: dict[str, Any], seed_head: str) -> list[str]:
    """Keep recovery inside its declared selection and skip settled outcomes across its ancestry."""
    available = discover_tasks(bench_dir)
    records: dict[pathlib.Path, dict[str, Any]] = {}
    depths: dict[pathlib.Path, int] = {}
    visiting: set[pathlib.Path] = set()

    def parent(root: pathlib.Path) -> dict[str, Any]:
        root = root.expanduser().resolve()
        if root in visiting:
            raise ValueError("resume ancestry contains a cycle")
        if root in records:
            return records[root]
        visiting.add(root)
        previous = _load(root / "run_manifest.json")
        harness = previous.get("harness", {})
        if (with_diagnostic_default(harness.get("applied_config")) != with_diagnostic_default(config)
                or previous.get("source", {}).get("head") != seed_head
                or harness.get("bench", {}).get("head") != args.bench_commit
                or not getattr(args, "image_id", "") or harness.get("image_id") != args.image_id):
            raise ValueError(f"resume configuration/seed/image differs: {root}")
        ancestors = [pathlib.Path(item).expanduser().resolve() for item in harness.get("resume_from", [])]
        for ancestor in ancestors:
            parent(ancestor)
        depth = 1 + max(depths[item] for item in ancestors) if ancestors else 0
        if previous.get("extra", {}).get("outcome") == "nothing_remaining":
            depth = max((depths[item] for item in ancestors), default=0)
        if harness.get("resume_generation", depth) != depth:
            raise ValueError(f"resume depth disagrees with ancestry: {root}")
        records[root], depths[root] = previous, depth
        visiting.remove(root)
        return previous

    direct = [pathlib.Path(item).expanduser().resolve() for item in args.resume_from]
    for root in direct:
        parent(root)
    if args.task_file or args.task:
        selected = [*read_task_file(pathlib.Path(args.task_file)), *args.task] if args.task_file else list(args.task)
    elif direct:
        selected = []
        for root in direct:
            previous = records[root]
            selection = previous["harness"].get("selection_task_ids", previous.get("requested_task_ids"))
            if not isinstance(selection, list):
                raise ValueError(f"resume root has no retained task selection: {root}")
            selected.extend(task for task in selection if task not in selected)
    else:
        selected = available
    if len(selected) != len(set(selected)) or any(task not in available for task in selected):
        raise ValueError("task selection contains duplicate or unknown dataset IDs")
    args.selection_task_ids = list(selected)
    args.resume_sources = [str(root) for root in records]
    args.resume_generation = 1 + max(depths[root] for root in direct) if direct else 0
    settled = settled_tasks(list(records))
    remaining = [task for task in selected if task not in settled]
    if remaining and args.resume_generation > 2:
        raise ValueError("at most two infrastructure recovery passes are permitted")
    if not remaining and direct:
        args.resume_generation = max(depths[root] for root in direct)
        # Keep every outcome source; the terminal nothing_remaining outcome records
        # why this no-op adds no paid recovery generation.
    return remaining

def remove_run_containers(docker_host: str, run_label: str) -> None:
    """Settle only this invocation's resources, including concurrent cleanup removals."""
    selector = f"label={LABEL_KEY}={run_label}"
    for kind, listing, removal in (
        ("container", ("ps", "-aq"), ("rm", "-fv")),
        ("network", ("network", "ls", "-q"), ("network", "rm")),
    ):
        found = _docker(docker_host, *listing, "--filter", selector, timeout=60)
        if found.returncode:
            raise RuntimeError(f"cannot verify this run's {kind} custody")
        ids = found.stdout.split()
        if not ids:
            continue
        removal_notice = {}
        try:
            removed = _docker(docker_host, *removal, *ids, timeout=120)
            if removed.returncode or removed.stderr:
                removal_notice = {"returncode": removed.returncode, "stdout": removed.stdout or "",
                                  "stderr": removed.stderr or "", "exception": None}
        except (OSError, subprocess.SubprocessError) as exc:
            # The daemon may have completed removal despite a client failure.
            removal_notice = {"returncode": None, "stdout": str(getattr(exc, "stdout", "") or ""),
                              "stderr": str(getattr(exc, "stderr", "") or ""),
                              "exception": {"type": type(exc).__name__, "message": str(exc)}}
        if removal_notice:
            print(json.dumps({"event": "cowork_cleanup_remove", "resource": kind,
                              "run_label": run_label, "selected_ids": ids, **removal_notice}), flush=True)
        remaining = _docker(docker_host, *listing, "--filter", selector, timeout=60)
        if remaining.returncode:
            raise RuntimeError(f"cannot verify this run's {kind} custody after removal")
        if remaining.stdout.split():
            print(json.dumps({"event": "cowork_cleanup_remaining", "resource": kind,
                              "run_label": run_label, "remaining_ids": remaining.stdout.split()}), flush=True)
            raise RuntimeError(f"could not remove all {kind}s owned by this run")


def mark_stop(path: pathlib.Path, reason: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(reason + "\n")
    except FileExistsError:
        pass


def stop_process_group(proc: subprocess.Popen) -> None:
    """Stop the official runner and its forked task shells before Docker cleanup."""
    terminate_process_group_id(proc.pid)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The session leader may have exited before a task shell; kill the original group too.
    kill_process_group_id(proc.pid)
    proc.wait(timeout=15)


def observe_campaign_usage(api_key: str, campaign: CampaignBudget, diagnostics: list[dict[str, Any]],
                           *, phase: str, deadline: float) -> float:
    """Read the meter until one value is accepted and saved, or until ``deadline``.

    Each read gets a bounded slice and a failed or rejected read is retried after a short
    pause; rejections never move the deadline and a lower counter is never accepted.
    Returns the monotonic time at which the accepted read was requested, the conservative
    anchor of the next blindness bound. A campaign write failure propagates unretried.
    """
    previous = campaign.record["last_usage"]
    last_error: Exception = TimeoutError("meter blindness bound elapsed without a confirmed reading")
    rejected = False
    attempt = 0
    while (remaining := deadline - time.monotonic()) > 0:
        attempt += 1
        requested_at = time.monotonic()
        value = None
        try:
            value = key_usage(api_key, timeout=min(METER_READ_SLICE_SEC, remaining))
            if time.monotonic() > deadline:
                raise TimeoutError("meter reading arrived after the blindness bound")
            validate_usage(value, previous)
        except Exception as exc:  # Provider/transport/parse failures share the same bound.
            last_error = exc
            value = getattr(exc, "observed_usage", value)
            observation = {
                "observed_at": time.time(), "phase": phase, "attempt": attempt,
                "previous_usage": previous,
                "observed_usage": value if value is None or math.isfinite(value) else repr(value),
                "accepted": False, "error_type": type(exc).__name__, "error": str(exc),
            }
            diagnostics.append(observation)
            print(json.dumps({"event": "cowork_meter_observation", **observation}), flush=True)
            rejected = True
        else:
            # A persistence failure is not a suspect provider read and is never retried here.
            campaign.observe(value)
            if time.monotonic() >= deadline:
                # Accounting may have reached disk, but a slow save cannot renew
                # permission to spend after the previous blindness window expired.
                raise TimeoutError("campaign persistence completed after the blindness bound")
            if rejected:
                observation = {"observed_at": time.time(), "phase": phase, "attempt": attempt,
                               "previous_usage": previous, "observed_usage": value, "accepted": True}
                diagnostics.append(observation)
                print(json.dumps({"event": "cowork_meter_observation", **observation}), flush=True)
            return requested_at
        time.sleep(min(METER_RETRY_SEC, max(0.0, deadline - time.monotonic())))
    raise last_error


def publish_diagnostics(root: pathlib.Path, bench_dir: pathlib.Path, args: argparse.Namespace,
                        monitor: dict[str, Any], failures: dict[str, dict[str, Any]], *, cause: str) -> dict[str, int] | None:
    """Refresh the ledger snapshot and monitor. Neither is spending authority: a failed write
    is disclosed on stderr and in later records, never a reason to stop valid work."""
    def disclose(artifact: str, exc: Exception) -> None:
        entry = failures.setdefault(artifact, {"count": 0})
        entry.update(count=entry["count"] + 1, error_type=type(exc).__name__, error=str(exc))
        # Until a later write succeeds, this stderr line is the only record of the failure.
        print(json.dumps({"event": "cowork_diagnostic_write_failed", "artifact": artifact, **entry}),
              file=sys.stderr, flush=True)

    counts = None
    try:
        counts = write_ledger(root / "result_index.jsonl", bench_dir, args.model, args.selected_tasks, cause=cause)
    except Exception as exc:
        disclose("result_index.jsonl", exc)
    try:
        write_json(root / "monitor.json", {**monitor, "ledger_counts": counts, "diagnostic_write_failures": failures})
    except Exception as exc:
        disclose("monitor.json", exc)
    return counts


def supervise_run(args: argparse.Namespace, command: list[str], bench_dir: pathlib.Path,
                  run_env: dict[str, str], api_key: str, campaign: CampaignBudget,
                  *, confirmed_at: float | None = None) -> dict[str, Any]:
    """Own runner lifetime, budget meter and disk reserve until every owned resource is settled.

    Stop before the campaign limit with a reserve for work already sent to providers. API
    billing can lag: the reserve is disclosed, never a claim of a provider-enforced cap.
    ``confirmed_at`` is the request time of the saved startup reading. Once the meter has
    gone ``--meter-blindness-sec`` without another confirmed, saved reading, or the campaign
    record cannot be saved, new work and the run stop rather than continue with unknown spending.
    """
    root = bench_dir.parent
    stop_file = pathlib.Path(run_env["COWORK_STOP_FILE"])
    # A task's lifetime budget is not a reservation of unsettled provider charges.
    # The CLI validates this explicit billing allowance as nonnegative.
    reserve = args.budget_reserve_usd
    blindness = args.meter_blindness_sec
    initial_spent = campaign.spent
    if campaign.remaining <= reserve:
        raise ValueError("campaign remaining budget does not cover the in-flight reserve")
    old_handlers = {}
    proc = None
    reason = ""
    meter_error = ""
    meter_diagnostics: list[dict[str, Any]] = []
    write_failures: dict[str, dict[str, Any]] = {}
    code = 1
    confirmed_at = time.monotonic() if confirmed_at is None else confirmed_at
    def request_stop(signum, _frame):
        mark_stop(stop_file, f"signal_{signum}")

    try:
        try:
            campaign.start(root)
        except CampaignPersistenceError:
            reason = "campaign_persistence_failed"
            raise
        if time.monotonic() >= confirmed_at + blindness:
            reason = "budget_meter_unavailable"
            raise TimeoutError("startup meter observation expired before runner spawn")
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, request_stop)
        with (root / "console.log").open("w", encoding="utf-8") as stream:
            proc = spawn_supervised(command, drive_root=root, purpose="cowork-official-runner",
                                    scope="session", cwd=bench_dir, env=run_env, stdout=stream,
                                    stderr=subprocess.STDOUT, text=True)
            while proc.poll() is None:
                if stop_file.exists():
                    reason = stop_file.read_text(encoding="utf-8").strip() or "stop_requested"
                    break
                try:
                    confirmed_at = observe_campaign_usage(api_key, campaign, meter_diagnostics, phase="poll",
                                                          deadline=confirmed_at + blindness)
                except CampaignPersistenceError as exc:
                    meter_error, reason = type(exc).__name__, "campaign_persistence_failed"
                except Exception as exc:
                    meter_error, reason = type(exc).__name__, "budget_meter_unavailable"
                free = shutil.disk_usage(args.resource_root).free
                root_free = shutil.disk_usage(pathlib.Path.home().anchor).free
                if not reason and (free < args.min_free_gib * 1024**3 or root_free < args.min_root_free_gib * 1024**3):
                    reason = "disk_reserve"
                if not reason and campaign.remaining <= reserve:
                    reason = "campaign_budget_reserve"
                if not reason and campaign.spent - initial_spent >= args.budget_usd:
                    reason = "run_budget"
                if reason:
                    # Stop owned work at once; the final record is published after cleanup.
                    mark_stop(stop_file, reason)
                    break
                publish_diagnostics(root, bench_dir, args, {
                    "observed_at": time.time(), "launcher_pid": os.getpid(), "runner_pid": proc.pid,
                    "campaign_spent_usd": campaign.spent, "campaign_remaining_usd": campaign.remaining,
                    "run_spent_usd": campaign.spent - initial_spent, "inflight_reserve_usd": reserve,
                    "meter_blindness_sec": blindness,
                    "meter_confirmed_age_sec": round(time.monotonic() - confirmed_at, 3),
                    "disk_free_bytes": free, "root_free_bytes": root_free,
                    "stop_reason": reason, "meter_error": meter_error, "meter_diagnostics": meter_diagnostics,
                }, write_failures, cause="in_progress")
                if time.monotonic() >= confirmed_at + blindness:
                    reason, meter_error = "budget_meter_unavailable", "TimeoutError"
                    mark_stop(stop_file, reason)
                    break
                # Poll every 15 s, but early enough that one full read still fits in the bound.
                wait = confirmed_at + blindness - METER_READ_SLICE_SEC - time.monotonic()
                try:
                    code = int(proc.wait(timeout=min(METER_POLL_SEC, max(0.0, wait))))
                except subprocess.TimeoutExpired:
                    continue
            if proc.poll() is not None:
                code = int(proc.returncode)
    except BaseException:
        # An unexpected supervisor failure is neither a finished runner nor exhausted money.
        reason = reason or "supervisor_error"
        raise
    finally:
        # Also fence creation on unexpected exceptions. Cleanup must not race the shell
        # admitting the next task after a finished task returned its concurrency token.
        stop_marker_error = None
        try:
            mark_stop(stop_file, reason or "runner_finished")
        except OSError as exc:
            # The marker is not the kill: a full/read-only run filesystem must not
            # strand the runner and its paid containers before process custody runs.
            stop_marker_error = exc
            reason = reason or "supervisor_error"
            print(json.dumps({"event": "cowork_stop_marker_failed", "error_type": type(exc).__name__}),
                  file=sys.stderr, flush=True)
        try:
            if proc is not None:
                stop_process_group(proc)
            remove_run_containers(args.docker_host, run_env["COWORK_RUN_LABEL"])
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        # Accounting only: the runner group and exact-label resources are already gone.
        try:
            observe_campaign_usage(api_key, campaign, meter_diagnostics, phase="final",
                                   deadline=time.monotonic() + blindness)
        except Exception as exc:
            meter_error = type(exc).__name__
        cause = reason or "runner_exited"
        settlement_recorded = False
        try:
            campaign.finish(root, outcome=reason or "runner_finished", meter_error=meter_error)
            settlement_recorded = True
        finally:
            # A failed campaign settlement retains active_run and must still leave
            # per-task interruption evidence when the diagnostic store is writable.
            counts = publish_diagnostics(root, bench_dir, args, {
                "observed_at": time.time(), "finished": True, "runner_exit_code": code,
                "campaign_spent_usd": campaign.spent, "campaign_remaining_usd": campaign.remaining,
                "run_spent_usd": campaign.spent - initial_spent, "stop_reason": reason,
                "meter_error": meter_error, "meter_diagnostics": meter_diagnostics,
                "campaign_settlement": "recorded" if settlement_recorded else "unconfirmed",
            }, write_failures, cause=cause)
        if stop_marker_error is not None:
            raise stop_marker_error
    return {"stop_reason": reason, "runner_exit_code": code, "meter_error": meter_error,
            "meter_diagnostics": meter_diagnostics, "interruption_cause": cause,
            "diagnostic_write_failures": write_failures,
            "ledger_counts": counts,
            "key_spend_usd": campaign.spent - initial_spent, "campaign_spent_usd": campaign.spent,
            "campaign_remaining_usd": campaign.remaining, "inflight_reserve_usd": reserve}

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    workspace = pathlib.Path(__file__).resolve().parents[4]
    parser.add_argument("--bench-root", default=os.environ.get("COWORK_BENCH_ROOT") or str(workspace / "benchmarks" / "cowork_bench"))
    parser.add_argument("--bench-commit", default=PINNED_BENCH_COMMIT)
    parser.add_argument("--repo-dir", default=str(repo_root_from_devtools()), help="clean Ouroboros seed checkout")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--task-file", default="")
    parser.add_argument("--resume-from", action="append", default=[], help="earlier run root whose settled tasks are skipped")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--task-timeout", type=int, default=3600, help="runner wall-clock cap per agent phase (s)")
    parser.add_argument("--mcp-tool-timeout", type=int, default=300)
    parser.add_argument("--effort", default="high")
    parser.add_argument("--review-effort", default="high")
    parser.add_argument("--review-slots", type=int, default=3)
    parser.add_argument("--task-review-mode", default="required")
    parser.add_argument("--review-enforcement", default="blocking")
    parser.add_argument("--review-max-cycles", default="2")
    parser.add_argument("--safety-mode", default="off", choices=["full", "light", "off"])
    parser.add_argument("--runtime-mode", default="pro")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--subagents", action="store_true", help="allow schedule_subagent (default: single agent)")
    parser.add_argument("--subagent-depth", type=int, default=3, help="only with --subagents")
    parser.add_argument("--per-task-cost-usd", type=float, default=25.0, help="in-container runaway guard")
    parser.add_argument("--diagnostic-eval-on-truncation", action="store_true",
                        help="audit-only residual-state checker rerun after a proven time/round/budget "
                             "status-gate decline; never changes official verdicts (METHODOLOGY.md)")
    parser.add_argument("--budget-usd", type=float, default=150.0, help="additional spend limit for this invocation")
    parser.add_argument("--campaign-file", default="", help="required shared budget record for every paid invocation")
    parser.add_argument("--campaign-budget-usd", type=float, default=1000.0)
    parser.add_argument("--prior-spend-usd", type=float, default=0.0, help="already spent before the first campaign baseline")
    parser.add_argument("--budget-reserve-usd", type=float, default=100.0, help="unspent allowance for delayed/in-flight billing")
    parser.add_argument("--meter-blindness-sec", type=float, default=DEFAULT_METER_BLINDNESS_SEC,
                        help="stop once this long passes without a confirmed, saved meter reading")
    parser.add_argument("--resource-root", default="", help="heavy storage filesystem; defaults to the run root")
    parser.add_argument("--min-free-gib", type=float, default=200.0)
    parser.add_argument("--min-root-free-gib", type=float, default=40.0)
    parser.add_argument("--min-key-usd", type=float, default=20.0)
    parser.add_argument("--or-provider", default="", help="OUROBOROS_OR_PROVIDER value; empty = provider routing untouched")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--credential-setting", default="OPENROUTER_API_KEY")
    parser.add_argument("--docker-host", default=os.environ.get("DOCKER_HOST") or "")
    parser.add_argument("--base-image", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-seed", action="store_true",
                        help="record and proceed with an unclean/unidentifiable seed checkout instead of refusing")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("concurrency must be positive; qualify the selected value before a full run")
    if min(args.budget_usd, args.campaign_budget_usd, args.per_task_cost_usd, args.task_timeout) <= 0:
        parser.error("budgets and timeout must be positive")
    if min(args.min_free_gib, args.min_root_free_gib, args.budget_reserve_usd, args.prior_spend_usd) < 0:
        parser.error("reserves and prior spending cannot be negative")
    if not math.isfinite(args.meter_blindness_sec) or args.meter_blindness_sec <= METER_POLL_SEC + METER_READ_SLICE_SEC:
        parser.error(f"meter blindness bound must exceed one {METER_POLL_SEC:g}s poll plus one "
                     f"{METER_READ_SLICE_SEC:g}s read")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve(strict=False)
    bench_root = pathlib.Path(args.bench_root).expanduser().resolve(strict=False)
    out_root = assert_outside_repo(
        pathlib.Path(args.run_root).expanduser() if args.run_root else run_root(BENCHMARK, args.run_id),
        args.repo_dir,
    )
    requested_out_root = out_root
    reused_root = out_root.exists()
    if reused_root:
        # Keep refusal evidence without overwriting the previous run's manifest.
        out_root = out_root.parent / timestamp_run_id("cowork-refused-reused-root")
    bench_dir = out_root / "bench"
    manifest_output = out_root / "run_manifest.json"
    ledger_output = out_root / "result_index.jsonl"
    args.base_image = args.base_image or f"cowork-pack:{args.bench_commit[:7]}"

    manifest = admit_benchmark_run(
        manifest_output,
        benchmark=BENCHMARK,
        run_root=out_root,
        repo_dir=args.repo_dir,
        requested_task_ids=list(args.task),
        require_clean=not args.allow_dirty_seed,
        argv=sys.argv,
        dataset=f"cowork_bench:{bench_root}@{args.bench_commit}",
        timeout_sec=args.task_timeout,
        output_paths={
            "manifest": str(manifest_output),
            "ledger": str(ledger_output),
            "console_log": str(out_root / "console.log"),
            "bench_working_copy": str(bench_dir),
        },
        harness={
            "runner": "run_parallel.sh",
            "agent_entry": "main_ouroboros.py",
            "phase_aware": True,
            "concurrency": args.concurrency,
            "max_steps": args.max_steps,
            "memory_mode": "empty",
            "base_image": args.base_image,
            "meter": {"blindness_sec": args.meter_blindness_sec, "poll_sec": METER_POLL_SEC,
                      "retry_sec": METER_RETRY_SEC, "read_slice_sec": METER_READ_SLICE_SEC},
        },
        extra={"outcome": "started"},
    )

    with finalize_run_manifest(manifest_output, manifest) as final:
        if reused_root:
            final.update({"outcome": "refused", "exit_code": 2,
                          "refusal": {"stage": "run_root", "reason": "run roots are append-only",
                                      "requested_root": str(requested_out_root)}})
            return 2
        if not args.docker_host:
            final.update({"outcome": "refused", "exit_code": 2,
                          "refusal": {"stage": "docker_host", "reason": "DOCKER_HOST must be explicit on a shared host"}})
            return 2
        args.resource_root = pathlib.Path(args.resource_root).expanduser().resolve() if args.resource_root else out_root
        if shutil.disk_usage(args.resource_root).free < args.min_free_gib * 1024**3:
            raise ValueError("resource filesystem is below the free-space reserve")
        (out_root / "tmp").mkdir(exist_ok=True)
        template = json.loads(SETTINGS_TEMPLATE.read_text(encoding="utf-8"))
        settings, actor = render_settings(template, args)
        manifest["model_slots"] = actor["model_slots"]
        manifest["available_subagents"] = actor["available_subagents"]
        seed_head = str((manifest.get("source") or {}).get("head") or "")
        args.image = args.image or f"cowork-ouroboros:{seed_head[:12]}-{args.bench_commit[:7]}"
        config = bench_config(args, settings)
        manifest["harness"] = {
            **manifest["harness"], "fixed_model_actor": actor, "image": args.image,
            "applied_config": config, "bench": bench_provenance(bench_root, args.bench_commit),
        }
        # The ledger reads this run's protocol from disk while supervise_run is live;
        # waiting until finalization would label every live row unknown.
        final.checkpoint("configured")

        if args.build_image and not image_exists(args.docker_host, args.image):
            build_image(args, seed_head, args.bench_commit, out_root / "image_build.log")
        if not image_exists(args.docker_host, args.image):
            final.update({"outcome": "refused", "exit_code": 2,
                          "refusal": {"stage": "image", "reason": f"image {args.image} not found; pass --build-image"}})
            return 2
        identity = image_identity(args.docker_host, args.image)
        args.image_id = identity["id"]
        labels = identity["labels"]
        manifest["harness"].update({"image_labels": labels, "image_id": args.image_id,
                                   "image_repo_digests": identity["repo_digests"]})
        if (labels.get("org.ouroboros.cowork.seed_sha") != seed_head
                or labels.get("org.ouroboros.cowork.bench_sha") != args.bench_commit):
            final.update({"outcome": "refused", "exit_code": 2,
                          "refusal": {"stage": "image", "reason": "image seed/benchmark label differs from the pinned checkouts"}})
            return 2

        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(bench_root), str(bench_dir)], check=True, timeout=600)
        _git(bench_dir, "checkout", "--quiet", "--detach", args.bench_commit)
        tasks = select_tasks(bench_dir, args, config, seed_head)
        args.selected_tasks = tasks
        manifest["harness"].update({"selection_task_ids": args.selection_task_ids,
                                   "resume_from": args.resume_sources,
                                   "resume_generation": args.resume_generation})
        manifest["requested_task_ids"] = tasks
        manifest["requested_count"] = len(tasks)
        if not tasks:
            write_result_index(ledger_output, [])
            final.update({"outcome": "nothing_remaining", "exit_code": 0})
            return 0

        api_key = str(os.environ.get(args.api_key_env) or "").strip()
        command = ["bash", "run_parallel.sh", str(args.concurrency), *tasks]
        manifest["official_command"] = command
        if args.dry_run:
            write_json(bench_dir / "configs" / CONFIG_NAME, config)
            final["outcome"] = "dry_run"
            return 0
        if not args.campaign_file:
            raise ValueError("--campaign-file is required for a paid run")
        if not api_key:
            final.update({"outcome": "refused", "exit_code": 2,
                          "refusal": {"stage": "credential", "reason": f"${args.api_key_env} is empty"}})
            return 2
        headroom = key_headroom(api_key)
        manifest["harness"]["credential"] = {
            "env": args.api_key_env, "setting": args.credential_setting,
            "fingerprint": credential_fingerprint(api_key), "headroom_at_start": headroom,
        }
        if headroom["effective"] is not None and headroom["effective"] < args.min_key_usd:
            final.update({"outcome": "refused", "exit_code": 2,
                          "refusal": {"stage": "credential", "reason": "key headroom below --min-key-usd"}})
            return 2

        write_json(bench_dir / "configs" / CONFIG_NAME, config)
        manifest["harness"]["config_sha256"] = hashlib.sha256(
            (bench_dir / "configs" / CONFIG_NAME).read_bytes()
        ).hexdigest()
        campaign_path = assert_outside_repo(pathlib.Path(args.campaign_file).expanduser(), args.repo_dir)
        run_env = {key: value for key, value in os.environ.items()
                   if key not in {args.api_key_env, "LLM_API_KEY", "MODEL_API_KEY"}}
        run_env.update({
            "DOCKER_HOST": args.docker_host, "IMAGE": args.image_id, "AGENT_ENTRY": "main_ouroboros.py",
            "AGENT_PHASE_AWARE": "1", "MODEL_PROVIDER": ENGINE, "MODEL_NAME": args.model, "LLM_MODEL": args.model,
            "MAX_STEPS": str(args.max_steps), "TASK_TIMEOUT": str(args.task_timeout),
        })
        run_env = prepare_resource_env(
            run_env, run_root=out_root, docker_host=args.docker_host,
            resource_root=args.resource_root, min_free_bytes=int(args.min_free_gib * 1024**3),
        )
        manifest["harness"]["resource_limits"] = {
            key: value for key, value in run_env.items() if key.startswith("COWORK_")
        }
        manifest["harness"]["campaign_file"] = str(campaign_path)
        secret_path = bench_dir / "configs" / SECRET_NAME
        with campaign_lock(campaign_path):
            # The validated, saved startup reading anchors the first meter blindness bound.
            confirmed_at = time.monotonic()
            campaign = CampaignBudget(campaign_path, fingerprint=credential_fingerprint(api_key),
                                      ceiling=args.campaign_budget_usd, usage=key_usage(api_key),
                                      prior_spend=args.prior_spend_usd)
            secret_path.touch(mode=0o600)
            try:
                secret_path.write_text(json.dumps({"settings": {args.credential_setting: api_key}}), encoding="utf-8")
                result = supervise_run(args, command, bench_dir, run_env, api_key, campaign,
                                       confirmed_at=confirmed_at)
            finally:
                secret_path.write_text("{}", encoding="utf-8")
        final.update(result)
        # The supervised final publication already owns this write and its errors.
        # Repeating it here would turn a diagnostic failure back into an exception.
        counts = result["ledger_counts"]
        stopped = bool(result["stop_reason"] or result["meter_error"])
        infra = counts is None or counts.get("infra_failed", 0) + counts.get("not_attempted", 0)
        code = 3 if stopped else (1 if result["runner_exit_code"] or infra else 0)
        final.update({"outcome": result["stop_reason"] or ("infra_failed" if infra else "completed"),
                      "exit_code": code, "ledger_counts": counts})
        return code


if __name__ == "__main__":
    raise SystemExit(main())
