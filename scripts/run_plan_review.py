#!/usr/bin/env python3
"""Standalone Ouroboros multi-model plan-review dry-run.

Runs the SAME plan-review engine ``plan_task`` uses (``ouroboros/tools/plan_review.py``)
over an operator-supplied envelope — goal, plan prose, spec JSON, optional evidence
locators — against the configured reviewer slots, in an ISOLATED drive root, and prints
the recorded wave (every slot's outcome, the validated findings, the host aggregate) plus
the coordinated tool output, untruncated. Nothing here is a second engine.

Usage (from anywhere):
    python scripts/run_plan_review.py --goal "..." --plan /path/to/plan.md \\
        --spec-json /path/to/spec.json [--evidence path/or/task:<id> ...] \\
        [--subject-root /path/to/workspace] [--drive-root /tmp/x] [--output out.txt]

``spec.json`` follows the plan_task spec schema: in_scope, non_goals, acceptance_claims,
invariants, decisions[{choice, rejected, why}], deferred[{what, why_safe_to_defer}],
affected_paths (REQUIRED — the files the work will CHANGE, ``[]`` when it changes none; this
is the only list resolved as paths, and a path under the system repo makes the plan
constitutional), affected_resources (the same question in words: systems, services, projects,
people), evidence (``--evidence`` values are appended to it).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_settings_into_env() -> None:
    """Load runtime settings through the shared config path; never print secrets."""
    try:
        from ouroboros.config import apply_settings_to_env, load_settings
        from ouroboros.server_runtime import apply_runtime_provider_defaults

        settings, _changed, _changed_keys = apply_runtime_provider_defaults(load_settings())
        apply_settings_to_env(settings)
    except Exception as exc:  # pragma: no cover - operator script
        print(f"WARN: could not load/apply Ouroboros settings: {exc}", file=sys.stderr)


def _read_text_file(path_text: str, *, label: str) -> str:
    path = pathlib.Path(path_text).expanduser().resolve(strict=False)
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise SystemExit(f"ERROR: could not read {label} at {path}: {exc}") from exc


def _read_spec(path_text: str) -> dict:
    if not path_text:
        return {}
    try:
        payload = json.loads(_read_text_file(path_text, label="spec JSON"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: spec JSON is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("ERROR: spec JSON must be an object")
    return payload


async def _run(args: argparse.Namespace) -> str:
    import tempfile

    from ouroboros.task_results import load_plan_review_state
    from ouroboros.tools.plan_review import _PlanRequest, _run_plan_review_async
    from ouroboros.tools.plan_review_runtime import plan_review_slots
    from ouroboros.tools.registry import ToolContext

    plan = _read_text_file(args.plan, label="plan")
    spec = _read_spec(getattr(args, "spec_json", "") or "")
    evidence = [str(v).strip() for v in (args.evidence or []) if str(v or "").strip()]
    if evidence:
        spec["evidence"] = list(spec.get("evidence") or []) + evidence
    goal = str(args.goal or "").strip() or str(spec.get("goal") or "")

    drive_root = (
        pathlib.Path(args.drive_root).expanduser().resolve(strict=False)
        if args.drive_root
        else pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-plan-review-"))
    )
    drive_root.mkdir(parents=True, exist_ok=True)
    (drive_root / "logs").mkdir(parents=True, exist_ok=True)
    subject_root = (
        pathlib.Path(getattr(args, "subject_root", "")).expanduser().resolve(strict=False)
        if getattr(args, "subject_root", "") else REPO
    )
    task_id = str(getattr(args, "task_id", "") or "plan-review-cli")
    ctx = ToolContext(
        repo_dir=REPO,
        system_repo_dir=REPO,
        workspace_root=subject_root if subject_root != REPO else None,
        workspace_mode="external" if subject_root != REPO else "",
        drive_root=drive_root,
        task_id=task_id,
        task_metadata={"root_task_id": task_id},
    )
    slots = [
        {"slot_id": s.slot_id, "model": s.model, "route": getattr(s.route, "value", str(s.route)),
         "effort": s.effort}
        for s in plan_review_slots()
    ]
    if getattr(args, "dry_run", False):
        # Assemble exactly what a paid cycle would send, then stop. The operator entry point
        # runs OUTSIDE a task, where `review_wave_budget_gate` has no usage scope and therefore
        # no cost ceiling — so the cheap way to see the bill first is to build and measure.
        from ouroboros.tools.plan_review import build_plan_review_packet_for_dry_run

        packet = build_plan_review_packet_for_dry_run(ctx, _PlanRequest(goal=goal, plan=plan, spec=spec))
        sep = "=" * 80
        return "\n".join([
            sep, "PLAN REVIEW DRY RUN (packet assembled; NOTHING dispatched, no cycle consumed)", sep,
            json.dumps({
                "slots": slots, "drive_root": str(drive_root),
                "constitutional": packet["constitutional"],
                "constitutional_note": packet["constitutional_note"],
                "system_prompt_chars": len(packet["system_prompt"]),
                "user_content_chars": len(packet["user_content"]),
                "packet_chars_per_slot": len(packet["system_prompt"]) + len(packet["user_content"]),
                "approx_tokens_per_slot": (len(packet["system_prompt"]) + len(packet["user_content"])) // 4,
                "evidence_attached": [
                    {k: v for k, v in item.items() if k != "text"}
                    for item in packet["manifest"].get("attached") or []
                ],
                "evidence_omissions": packet["manifest"].get("omissions") or [],
                "fingerprint": packet["fingerprint"],
            }, ensure_ascii=False, indent=2, default=str),
        ])
    coordinated = await _run_plan_review_async(ctx, _PlanRequest(goal=goal, plan=plan, spec=spec))
    try:
        state = load_plan_review_state(drive_root, task_id)
    except ValueError:
        state = {}
    waves = state.get("waves") or []
    wave = waves[-1] if waves else {}

    sep = "=" * 80
    return "\n".join([
        sep, "RESOLVED PLAN REVIEW CONFIG", sep,
        json.dumps({
            "slots": slots,
            "drive_root": str(drive_root),
            "system_repo_root": str(REPO),
            "subject_root": str(subject_root),
            "plan": str(pathlib.Path(args.plan).expanduser().resolve(strict=False)),
            "spec_json": str(getattr(args, "spec_json", "") or ""),
            "evidence": evidence,
            "cycles_paid": state.get("cycles_paid"),
        }, ensure_ascii=False, indent=2, default=str),
        sep, "PLAN REVIEW WAVE (recorded state; every slot, validated findings, host aggregate)", sep,
        json.dumps(wave, ensure_ascii=False, indent=2, default=str),
        sep, "PLAN REVIEW COORDINATED OUTPUT", sep,
        coordinated,
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the configured Ouroboros plan-review panel over a goal + plan + spec envelope."
    )
    parser.add_argument("--goal", default="", help="Goal under review (falls back to spec.goal).")
    parser.add_argument("--plan", required=True, help="Path to the plan prose file.")
    parser.add_argument("--spec-json", default="", help="Path to the spec JSON object (plan_task spec schema).")
    parser.add_argument(
        "--evidence", action="append", default=[],
        help="Evidence locator (path relative to the subject root, absolute path, task:<id>, or URL); repeatable.",
    )
    parser.add_argument(
        "--subject-root", default="",
        help="Active repository/workspace under review; governance remains the Ouroboros repository.",
    )
    parser.add_argument(
        "--drive-root", default=os.environ.get("OUROBOROS_REVIEW_DRIVE_ROOT", ""),
        help="Drive root for review state/observability writes. Prefer a temp dir.",
    )
    parser.add_argument("--task-id", default="", help=(
        "State/task id for this run (default plan-review-cli). Distinct ids keep separate\n"
        "cycle counters, so two operator runs do not share one cap."))
    parser.add_argument("--dry-run", action="store_true", help=(
        "Assemble the packet and print its size per slot, then stop BEFORE any paid dispatch."))
    parser.add_argument("--output", default="", help="Optional path to also write the full output.")
    args = parser.parse_args()

    _load_settings_into_env()
    output = asyncio.run(_run(args))
    print(output)
    if args.output:
        pathlib.Path(args.output).expanduser().write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
