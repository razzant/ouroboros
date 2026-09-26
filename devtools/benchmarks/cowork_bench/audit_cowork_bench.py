#!/usr/bin/env python3
"""Offline Cowork evidence audit, never a scorer or proof of no contamination.

Read only the adapter ledger, its copied logs and eval-attempt records, never copying
task answers or evaluator contents. Findings identify log coordinates for manual
inspection without copying commands, results, or gold values. Cost is the llm_usage compatibility projection,
not provider billing authority; missing prices/logs remain explicitly unknown.
Eval-attempt and residual-diagnostic facts form a separate block that never feeds
the classification, official verdict or score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from devtools.benchmarks.cowork_bench import eval_attempt
from devtools.benchmarks.cowork_bench.resource_limits import ADMISSION_DIR_NAME

# Diagnostic source/artifact references, not a semantic cheating classifier.
_REFERENCE = re.compile(
    r"(?:groundtruth_workspace|(?:^|[/\\\s])evaluation[/\\]|eval_res\.json|ouroboros_(?:eval|diagnostic)_|"
    r"(?:github\.com|raw\.githubusercontent\.com)/0717376/cowork_bench|toolathlon)",
    re.IGNORECASE,
)
_SQL_CLIENT = re.compile(r"\b(?:psql|pgcli|psycopg2?|asyncpg)\b", re.IGNORECASE)


def _records(path: Path, gaps: list[dict[str, Any]]) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line, text in enumerate(handle, 1):
                if not text.strip():
                    continue
                try:
                    row = json.loads(text)
                    if not isinstance(row, dict):
                        raise ValueError("not an object")
                except ValueError:
                    gaps.append({"source": path.name, "line": line, "reason": "invalid_record"})
                    continue
                yield line, row
    except (OSError, UnicodeError):
        gaps.append({"source": path.name, "reason": "unreadable_or_missing"})


def _object(path: Path, gaps: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    except (OSError, ValueError, UnicodeError):
        pass
    gaps.append({"source": path.name, "reason": "unreadable_or_invalid"})
    return {}


def argument_strings(value: Any) -> Iterator[str]:
    """Unwrap dict/list arguments, JSON-encoded args, and shell argv without execution."""
    if isinstance(value, dict):
        for child in value.values():
            yield from argument_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from argument_strings(child)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            yield from argument_strings(decoded)
            return
        yield value
        try:
            yield from shlex.split(value)
        except ValueError:
            pass  # The original malformed shell text still gets inspected.


def call_findings(row: dict[str, Any]) -> list[str]:
    """Return review cues from requested arguments only; echoed tool output is ignored."""
    texts = list(argument_strings(row.get("args", row.get("arguments", {}))))
    findings = []
    if any(_REFERENCE.search(text) for text in texts):
        findings.append("answer_source_or_evaluator_reference")
    tool = str(row.get("tool") or "")
    # SQL through the benchmark's database MCP is expected. A native shell or the
    # benchmark terminal/python tool can instead bypass those tool interfaces.
    shell_or_code = tool in {"run_command", "run_script", "start_service", "python_execute"}
    shell_or_code |= tool.startswith("mcp_terminal__")
    if shell_or_code and any(_SQL_CLIENT.search(text) for text in texts):
        findings.append("possible_direct_postgres_access")
    return findings


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def usage_tokens(record: dict[str, Any]) -> dict[str, int | None]:
    """Token counts stated by one llm_usage record; ``None`` where it states none."""
    usage = record.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    counts = {key: _number(record.get(key, usage.get(key)))
              for key in ("prompt_tokens", "completion_tokens", "cached_tokens")}
    return {key: None if value is None else int(value) for key, value in counts.items()}


def token_bearing(tokens: dict[str, int | None]) -> bool:
    """Positive evidence of a model call; zero or unstated tokens prove nothing."""
    return (tokens["prompt_tokens"] or 0) + (tokens["completion_tokens"] or 0) > 0


def model_activity_observed(events: Path) -> bool:
    """Whether a copied events log holds a token-bearing llm_usage record (first one wins)."""
    return any(record.get("type") == "llm_usage" and token_bearing(usage_tokens(record))
               for _line, record in _records(events, []))


def _omission_count(value: Any) -> int:
    if isinstance(value, (list, dict)):
        return len(value)
    return int(bool(value))


def run_eval_facts(root: Path) -> dict[str, Any]:
    """Run-level stop evidence: the applied flag, proven owned-resource cleanup and the
    shim's host-only record of refused eval containers. Missing sources stay unknown."""
    applied = (_object(root / "run_manifest.json", []).get("harness") or {}).get("applied_config")
    refusals: dict[str, str] = {}
    for mapping in sorted((root / ADMISSION_DIR_NAME).glob("*.task")):
        try:
            refusals[mapping.read_text(encoding="utf-8").strip()] = (
                mapping.with_suffix(".eval_refused").read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError):
            continue
    monitor = _object(root / "monitor.json", [])
    try:
        with (root / "resource_stop").open("r", encoding="utf-8") as stream:
            first_stop = stream.readline(128).strip()
    except (OSError, UnicodeError):
        first_stop = None
    return {
        # Runs recorded before the opt-in flag existed never ran the diagnostic.
        "enabled": applied.get(eval_attempt.FLAG, False) is True if isinstance(applied, dict) else None,
        # The launcher finalizes monitor.json only after its exact-label cleanup succeeded,
        # which removed every container of the run and so every eval process in it.
        "cleanup_proven": monitor.get("finished") is True,
        "stop_reason": monitor.get("stop_reason"),
        "first_stop": first_stop,
        "eval_refusals": refusals,
    }


def eval_attempt_block(task_dump: Path, instance_id: str, official_status: str,
                       facts: dict[str, Any]) -> dict[str, Any]:
    """Official attempt state and the residual diagnostic, projected only from records.

    A task-specific eval-container refusal is ``unavailable``; a claim without
    its terminal receipt is ``unknown`` even after cleanup."""
    official = eval_attempt.attempt_facts(task_dump)
    refusal = facts.get("eval_refusals", {}).get(instance_id)
    cleanup = facts.get("cleanup_proven") is True
    if official["state"] == "terminal":
        if official["official_run"] is False:
            projection, reason = "not_run", official["cause"]
        elif official.get("file_matches_returned"):
            projection, reason = "evaluated", ""
        elif official.get("raised") and cleanup:
            projection, reason = "interrupted", "evaluator_raised_and_cleanup_proven"
        else:
            projection, reason = "unknown", "effect_without_linked_verdict"
    elif official["state"] == "claimed":
        # The claim precedes preparation; even post-cleanup it cannot prove the
        # evaluator was invoked. Never promote it to an interrupted checker.
        projection, reason = "unknown", "claim_without_terminal_receipt"
    elif official["state"] == "absent" and refusal:
        # A task-specific refusal also occurs on disk failure or a generic stop file.
        # Only the run's settled money-stop evidence warrants a campaign-specific cause.
        campaign_stop = (cleanup and refusal == "admission_stopped"
                         and facts.get("stop_reason") == facts.get("first_stop")
                         and facts.get("first_stop") in {
                             "campaign_budget_reserve", "run_budget", "budget_meter_unavailable"})
        projection = "unavailable"
        reason = "campaign_stopped_before_eval" if campaign_stop else f"eval_creation_refused:{refusal}"
    else:
        projection, reason = "unknown", ("invalid_claim" if official["state"] == "invalid" else "no_attempt_record")
    diagnostic = eval_attempt.diagnostic_facts(task_dump)
    if diagnostic["state"] == "terminal":
        outcome, why = diagnostic["outcome"], diagnostic.get("reason") or ""
    elif diagnostic["state"] in {"claimed", "invalid"}:
        # Diagnostic claim predates eligibility and spawn; it does not prove the
        # checker ran or that a container stop interrupted its effects.
        outcome = "unknown"
        why = f"diagnostic_{diagnostic['state']}_without_terminal_receipt"
    elif facts.get("enabled") is False:
        outcome, why = "disabled", ""
    elif projection == "unavailable":
        outcome, why = "unavailable", reason
    elif official_status == "declined" or official["state"] in {"claimed", "invalid"}:
        outcome, why = "unknown", "diagnostic_not_started"
    else:
        outcome, why = "not_applicable", "no_status_gate_decline"
    return {"official": {**official, "projection": projection, "reason": reason, "eval_refusal": refusal},
            "residual_diagnostic": {**diagnostic, "outcome": outcome, "reason": why}}


def audit_task(task_dump: Path, ledger: dict[str, Any], facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report copied runtime evidence separately from the existing official verdict."""
    gaps: list[dict[str, Any]] = []
    summary = _object(task_dump / "ouroboros_summary.json", gaps)
    events = task_dump / "ouroboros" / "events.jsonl"
    tools = task_dump / "ouroboros" / "tools.jsonl"
    activity = {"usage_records": 0, "nonempty_usage_records": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
                "tool_calls": 0, "mcp_calls": 0, "mcp_error_calls": 0}
    known_cost = 0.0
    unknown_cost = estimated_cost = 0
    billing_providers: set[str] = set()
    response_providers: set[str] = set()
    models: set[str] = set()
    findings: list[dict[str, Any]] = []
    omission_refs: list[dict[str, Any]] = []
    for source in (events, tools):
        for line, record in _records(source, gaps):
            if record.get("capability_omissions"):
                omission_refs.append({"source": source.name, "line": line,
                                      "count": _omission_count(record["capability_omissions"])})
            if record.get("response_provider"):
                response_providers.add(str(record["response_provider"]))
            if source == events and record.get("type") == "llm_usage":
                activity["usage_records"] += 1
                usage = record.get("usage")
                usage = usage if isinstance(usage, dict) else {}
                tokens = usage_tokens(record)
                for key, value in tokens.items():
                    if value is None:
                        gaps.append({"source": source.name, "line": line, "reason": f"unknown_{key}"})
                    activity[key] += value or 0
                activity["nonempty_usage_records"] += int(token_bearing(tokens))
                cost = _number(record.get("cost", usage.get("cost")))
                if cost is None or record.get("cost_known") is False:
                    unknown_cost += 1
                else:
                    known_cost += cost
                    estimated_cost += int(bool(record.get("cost_estimated")))
                if record.get("provider"):
                    billing_providers.add(str(record["provider"]))
                if record.get("model"):
                    models.add(str(record["model"]))
                if usage.get("response_provider"):
                    response_providers.add(str(usage["response_provider"]))
            elif source == tools and record.get("type") == "tool_call":
                activity["tool_calls"] += 1
                if str(record.get("tool") or "").startswith("mcp_"):
                    activity["mcp_calls"] += 1
                    activity["mcp_error_calls"] += int(record.get("is_error") is True)
                for reason in call_findings(record):
                    findings.append({"source": source.name, "line": line, "reason": reason})
    if summary.get("capability_omissions"):
        omission_refs.append({"source": "ouroboros_summary.json",
                              "count": _omission_count(summary["capability_omissions"])})
    status = str(ledger.get("status") or "unknown")
    classification = {
        "infra_failed": "infrastructure", "agent_failed": "genuine_failure",
        "failed": "genuine_failure", "passed": "passed", "not_attempted": "not_attempted",
    }.get(status, "unknown")
    # Preserve the typed ledger, including genuine timeouts with model activity.
    # Lack of telemetry is an audit gap, never a reason to rewrite its verdict.
    log_incomplete = any(g["source"] == "events.jsonl" for g in gaps)
    cost_complete = bool(activity["usage_records"]) and not unknown_cost and not log_incomplete
    official_status = str(ledger.get("official_eval_status") or "unreported")
    # Literal evaluator truth is independent of agent execution and the scoring classification.
    # Legacy rows without receipt metadata retain their previously recorded scored verdict.
    details = ledger.get("details") if isinstance(ledger.get("details"), dict) else {}
    receipt = details.get("official_receipt") if isinstance(details.get("official_receipt"), dict) else {}
    receipt_bytes = receipt.get("bytes")
    return {
        "instance_id": str(ledger.get("instance_id") or task_dump.name.removeprefix("SingleUserTurn-")),
        "ledger_status": status,
        "classification": classification,
        "reason_code": str(ledger.get("reason_code") or summary.get("reason_code") or ""),
        "official_eval_status": official_status,
        "official_pass": (receipt.get("pass") if receipt else status == "passed")
        if official_status == "completed" and (isinstance(receipt.get("pass"), bool)
                                                or not receipt and status in {"passed", "failed"}) else None,
        "official_receipt": {
            "available": bool(receipt),
            "pass": receipt.get("pass") if isinstance(receipt.get("pass"), bool) else None,
            "bytes": receipt_bytes if isinstance(receipt_bytes, int) and not isinstance(receipt_bytes, bool) else None,
            "sha256": str(receipt.get("sha256") or "") or None,
        },
        "activity": activity,
        "model_activity_observed": bool(activity["nonempty_usage_records"]),
        "mcp_activity_observed": bool(activity["mcp_calls"]),
        "cost": {"known_usd": round(known_cost, 9),
                 "total_usd": round(known_cost, 9) if cost_complete else None,
                 "unknown_usage_records": unknown_cost, "estimated_usage_records": estimated_cost,
                 "complete": cost_complete, "source": "llm_usage_projection"},
        "models": sorted(models), "billing_providers": sorted(billing_providers),
        "observed_response_providers": sorted(response_providers),
        "response_provider_coverage": "observed_subset" if response_providers else "unavailable",
        "capability_omissions": {"reported_count": sum(r["count"] for r in omission_refs),
                                 "references": omission_refs, "absence_proves_none": False},
        "manual_review": findings, "gaps": gaps,
        "eval_attempt": eval_attempt_block(task_dump, str(ledger.get("instance_id") or ""), official_status,
                                           facts or {}),
    }


def audit_run(run_root: Path | str) -> dict[str, Any]:
    """Audit each ledger row, retaining not-attempted tasks in the denominator."""
    root = Path(run_root).expanduser().resolve()
    gaps: list[dict[str, Any]] = []
    facts = run_eval_facts(root)
    rows = []
    for _, ledger in _records(root / "result_index.jsonl", gaps):
        raw_path = (ledger.get("output_paths") or {}).get("task_dump")
        if not raw_path:
            gaps.append({"source": "result_index.jsonl", "reason": "missing_task_dump"})
            raw_path = root / "missing" / str(ledger.get("instance_id") or "unknown")
        task_dump = Path(raw_path)
        if not task_dump.is_absolute():
            task_dump = root / task_dump
        rows.append(audit_task(task_dump, ledger, facts))
    complete = bool(rows) and not gaps and all(row["cost"]["complete"] for row in rows)
    known = round(sum(row["cost"]["known_usd"] for row in rows), 9)
    diagnostics = [row["eval_attempt"]["residual_diagnostic"] for row in rows]
    return {
        "schema": "ouroboros.cowork.audit.v1",
        "scoring_authority": "official evaluator; this audit never changes scores",
        "limitations": ["Diagnostic argument references require manual review, not automatic disqualification.",
                        "No flags do not prove no contamination; copied logs may omit full arguments or responses.",
                        "llm_usage cost is a compatibility projection, not authoritative provider billing.",
                        "Billing provider names do not identify OpenRouter upstream endpoints.",
                        "A receipt verdict on an agent/infrastructure-failed row is disclosure, never a scored pass.",
                        "Residual diagnostics rerun the checker on post-stop state, not an exact-deadline "
                        "verdict; they never enter the ledger, summary.csv or the score."],
        "task_count": len(rows), "classifications": dict(Counter(row["classification"] for row in rows)),
        "manual_review_tasks": [row["instance_id"] for row in rows if row["manual_review"]],
        "cost": {"known_usd": known, "total_usd": known if complete else None, "complete": complete},
        "gaps": gaps, "tasks": rows,
        "eval_attempts": {
            "official_projection": dict(Counter(row["eval_attempt"]["official"]["projection"] for row in rows)),
            "cleanup_proven": facts["cleanup_proven"],
            "residual_diagnostic": {
                "enabled": facts["enabled"], "semantics": eval_attempt.SEMANTICS,
                "cap_sec": eval_attempt.DIAGNOSTIC_CAP_SEC, "task_count": len(rows),
                "official_declined": sum(row["official_eval_status"] == "declined" for row in rows),
                "outcomes": dict(Counter(item["outcome"] for item in diagnostics)),
                "eligible": sum(item.get("eligible") is True for item in diagnostics),
                "checked": sum(item["outcome"] in {"checks_passed", "checks_failed"} for item in diagnostics),
                "ineligible_reasons": dict(Counter(item["reason"] for item in diagnostics
                                                   if item["outcome"] == "ineligible")),
                "limitations": [
                    "Eligible only after a proven same-attempt status-gate decline for deadline_local, "
                    "wall_clock_timeout, round_limit or budget_exhausted, observed model activity and a "
                    "host-confirmed removed agent container; other declines are not covered.",
                    "The checker sees state after the agent stopped and after the official evaluator "
                    "ran, possibly minutes past the limit; the cap is a ceiling, not a promised window.",
                    "A claim alone never proves evaluation started; only a terminal exception with proven cleanup may be interrupted.",
                ],
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", help="Optional JSON output; stdout otherwise. Existing files are not replaced.")
    args = parser.parse_args(argv)
    report = audit_run(args.run_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
