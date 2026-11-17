#!/usr/bin/env python3
"""Fail-closed OSWorld adapter skeleton.

This file intentionally does not implement OSWorld scoring. It verifies that a
runnable official OSWorld environment and Ouroboros computer-use surface exist
before a future adapter is allowed to proceed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
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
)
from devtools.benchmarks.common.result_index import task_result_row, write_result_index
from devtools.benchmarks.common.run_roots import live_data_roots
from devtools.benchmarks.osworld.run_step_agent import (
    ALIGNED_UPSTREAM,
    amend_task_manifest,
    osworld_checkout_info,
)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = DEFAULT_REPO_ROOT.parent / "data"
DEFAULT_ISOLATED_DATA_DIRNAME = "isolated_data"


def _http_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip().startswith("{") else {"raw": raw, "status": getattr(resp, "status", None)}


def _outside(path: Path, forbidden: list[Path]) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    for root in forbidden:
        try:
            resolved.relative_to(root.expanduser().resolve(strict=False))
            return False
        except ValueError:
            continue
    return True


def _overlaps(path: Path, root: Path) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
        return True
    except ValueError:
        pass
    try:
        resolved_root.relative_to(resolved)
        return True
    except ValueError:
        return False


def _forbidden_data_roots() -> list[Path]:
    roots = [Path(root) for root in live_data_roots()]
    default_root = DEFAULT_DATA_ROOT.expanduser().resolve(strict=False)
    if all(root.expanduser().resolve(strict=False) != default_root for root in roots):
        roots.append(DEFAULT_DATA_ROOT)
    return roots


def _unix_computer_use_state_failures(data_root: Path, state_dir: Path, payload_dir: Path) -> list[str]:
    failures: list[str] = []
    expected_state_dir = data_root.expanduser().resolve(strict=False) / "state" / "skills" / payload_dir.name
    if state_dir.expanduser().resolve(strict=False) != expected_state_dir:
        failures.append(f"unix_computer_use state dir must be the runtime canonical isolated state dir: {expected_state_dir}")
    try:
        from ouroboros.skill_loader import load_skill
        from ouroboros.skill_readiness import skill_readiness_for_execution

        loaded = load_skill(payload_dir, data_root)
        if loaded is None:
            failures.append(f"unix_computer_use payload is not a valid skill: {payload_dir}")
            return failures
        # The benchmark reads the verdict itself: a skill with blocker findings is no longer refused by any gate.
        stale = loaded.review.is_stale_for(loaded.content_hash)
        status = str(loaded.review.status or "")
        if stale or status not in {"clean", "warnings"}:
            failures.append(
                "unix_computer_use review must be fresh executable pass/advisory_pass "
                f"(clean/warnings): {'review_stale' if stale else status or 'review_not_executable'}"
            )
        else:
            readiness = skill_readiness_for_execution(data_root, loaded, require_enabled=True, require_grants=True)
            if not readiness.ready:
                failures.append("unix_computer_use is not executable in isolated state: " + ", ".join(readiness.blockers))
    except Exception as exc:
        failures.append(f"unix_computer_use runtime readiness could not be verified: {type(exc).__name__}: {exc}")
    return failures


def preflight(
    *,
    osworld_root: Path,
    ouroboros_url: str,
    osworld_server_url: str,
    unix_computer_use_payload: Path,
    unix_computer_use_state_dir: Path,
    output_root: Path,
    repo_root: Path,
    data_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    attestation: dict[str, Any] = {}
    checkout = osworld_checkout_info(osworld_root)
    if not osworld_root.is_dir():
        failures.append(f"official OSWorld checkout not found: {osworld_root}")
    if not (osworld_root / "run.py").exists() and not (osworld_root / "evaluation_examples").exists():
        failures.append(f"OSWorld checkout shape is not recognized: {osworld_root}")
    if not unix_computer_use_payload.exists():
        failures.append(f"unix_computer_use payload is missing: {unix_computer_use_payload}")
    isolation_failures: list[str] = []
    try:
        unix_computer_use_state_dir.expanduser().resolve(strict=False).relative_to(data_root.expanduser().resolve(strict=False))
    except ValueError:
        isolation_failures.append(f"unix_computer_use state dir must be under isolated data root: {unix_computer_use_state_dir}")
    forbidden_data_roots = _forbidden_data_roots()
    if not _outside(output_root, [repo_root, data_root, *forbidden_data_roots]):
        isolation_failures.append(f"output root must be outside repo and runtime data: {output_root}")
    if not _outside(data_root, [repo_root]):
        isolation_failures.append(f"isolated data root must be outside repo: {data_root}")
    if any(_overlaps(data_root, live_root) for live_root in forbidden_data_roots):
        isolation_failures.append(f"isolated data root must not overlap live Ouroboros data root: {data_root}")
    failures.extend(isolation_failures)
    if not isolation_failures:
        failures.extend(_unix_computer_use_state_failures(data_root, unix_computer_use_state_dir, unix_computer_use_payload))
    try:
        _http_json(ouroboros_url.rstrip("/") + "/api/state")
    except Exception as exc:
        failures.append(f"Ouroboros server is not reachable: {type(exc).__name__}: {exc}")
    # Owner Q9=A+B / Q10: every launcher that attaches to a live server URL attests WHAT is
    # running against the checkout it was started from. This one only preflights, but it is
    # still a URL-attaching entry point, so the same shared helper runs here — as a typed
    # failure (it fails closed by raising) rather than a traceback.
    # A refusal CARRIES its record (typed `reason`, `runtime_version`, `repo_head`,
    # `repo_version`); keeping only the message discarded exactly the evidence this preflight
    # exists to surface, and the manifest amended from `result["details"]` inherited the loss.
    try:
        attestation = runtime_attestation(ouroboros_url, repo_root)
    except RuntimeAttestationRefused as exc:
        attestation = dict(exc.attestation)
        failures.append(f"runtime attestation failed reason={exc.attestation.get('reason') or ''}: {exc}")
    except RuntimeError as exc:
        attestation = {"ok": False, "reason": "runtime_attestation_failed",
                       "error": f"{type(exc).__name__}: {exc}"}
        failures.append(f"runtime attestation failed: {exc}")
    try:
        urllib.request.urlopen(osworld_server_url.rstrip("/") + "/", timeout=5).read(1)
    except Exception as exc:
        failures.append(f"OSWorld desktop/control server is not reachable: {type(exc).__name__}: {exc}")
    return {"ok": not failures, "failures": failures,
            "details": {"osworld_checkout": checkout, "runtime_attestation": attestation}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osworld-root", required=True)
    parser.add_argument("--ouroboros-url", default="http://127.0.0.1:8765")
    parser.add_argument("--osworld-server-url", required=True)
    parser.add_argument("--unix-computer-use-payload", required=True)
    parser.add_argument("--unix-computer-use-state-dir", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--task", action="append", default=[], help="requested OSWorld task id for manifest/ledger")
    parser.add_argument("--settings-path", default="")
    parser.add_argument("--ledger-output", default="")
    parser.add_argument("--manifest-output", default="")
    parser.add_argument("--allow-dirty-seed", action="store_true",
                        help="run even when this Ouroboros checkout is dirty or its git identity is "
                             "unreadable (default: fail closed). Recorded in the manifest.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser() if args.repo_root else DEFAULT_REPO_ROOT
    output_root = Path(args.output_root).expanduser()
    data_root = Path(args.data_root).expanduser() if args.data_root else output_root / DEFAULT_ISOLATED_DATA_DIRNAME
    settings_path = Path(args.settings_path).expanduser() if args.settings_path else data_root / "settings.json"
    state_dir = (
        Path(args.unix_computer_use_state_dir).expanduser()
        if args.unix_computer_use_state_dir
        else data_root / "state" / "skills" / "unix_computer_use"
    )
    requested = args.task or ["osworld_preflight"]
    ledger_path = Path(args.ledger_output).expanduser() if args.ledger_output else output_root / "osworld_preflight.ledger.jsonl"
    manifest_path = Path(args.manifest_output).expanduser() if args.manifest_output else output_root / "osworld_preflight.run_manifest.json"
    # Confinement FIRST, and it is a PURE predicate (path arithmetic only, no mkdir, no write):
    # admission persists a manifest, so the place it would persist it has to be inside the
    # declared boundary before anything is written there.
    can_write_artifacts = (
        _outside(output_root, [repo_root, data_root, *_forbidden_data_roots()])
        and _outside(ledger_path.parent, [repo_root, data_root, *_forbidden_data_roots()])
        and _outside(manifest_path.parent, [repo_root, data_root, *_forbidden_data_roots()])
    )
    if not can_write_artifacts:
        # Nothing may be written outside the boundary — including the admission record — so
        # this refusal is reported WITHOUT running the preflight: probing a live server before
        # a run can even be admitted is exactly the pre-admission work the seams forbid, and
        # there is nothing here to record it against.
        print(json.dumps({
            "ok": False,
            "failures": [
                "output paths are not isolated from the repo/live data roots: "
                f"output_root={output_root} ledger={ledger_path} manifest={manifest_path}"
            ],
            "details": {"confinement": {"output_root": str(output_root),
                                        "ledger": str(ledger_path),
                                        "manifest": str(manifest_path),
                                        "repo_root": str(repo_root),
                                        "data_root": str(data_root)}},
        }, ensure_ascii=False, indent=2))
        return 2

    # ADMISSION: build the manifest, PERSIST it, then let the clean-seed gate enforce (owner
    # Q19). This entry point spends nothing, so a refusal still runs the preflight afterwards
    # and folds into its OWN typed refusal shape rather than a bare traceback — but the refusal
    # is now DURABLE. Writing no manifest at all (the previous behaviour) meant the one path
    # where provenance was refused was also the one path that left no evidence of the refusal.
    seed_gate_error = ""
    admission_refused = False
    try:
        base_manifest = admit_benchmark_run(
            manifest_path,
            benchmark="osworld",
            run_root=output_root,
            repo_dir=repo_root,
            requested_task_ids=requested,
            argv=sys.argv,
            dataset="OSWorld",
            require_clean=not args.allow_dirty_seed,
            harness={
                "osworld_root": str(Path(args.osworld_root).expanduser()),
                "osworld_server_url": args.osworld_server_url,
                "unix_computer_use_payload": str(Path(args.unix_computer_use_payload).expanduser()),
                "unix_computer_use_state_dir": str(state_dir),
                "runnable_adapter": False,
                "aligned_upstream": dict(ALIGNED_UPSTREAM),
            },
            official_command=[],
            isolated_data_root=str(data_root),
            settings_path=settings_path,
        )
    except BenchmarkAdmissionRefused as exc:
        base_manifest = exc.manifest
        seed_gate_error = f"{type(exc).__name__}: {exc}"
        admission_refused = True

    if admission_refused:
        # SHORT-CIRCUIT. The documented contract (osworld/METHODOLOGY.md, docs/ARCHITECTURE.md)
        # is that a dirty or unidentifiable seed stops the run BEFORE the preflight — and the
        # preflight is not free: it probes the filesystem and reaches the live server and the
        # OSWorld control server over the network. Running it anyway after refusing contradicted
        # the contract in the direction that spends work on a run that was already refused.
        # The refusal is finalized DURABLY first, then we return.
        with finalize_run_manifest(manifest_path, base_manifest,
                                   outcome="refused", exit_code=2) as final:
            final["refusal"] = {**((base_manifest.get("extra") or {}).get("refusal") or {}),
                                "exit_code": 2}
            base_manifest.update(amend_task_manifest(
                base_manifest,
                output_paths={"manifest": str(manifest_path)},
                extra={"allow_dirty_seed": bool(args.allow_dirty_seed),
                       "seed_gate_error": seed_gate_error,
                       "preflight": {"ok": False, "failures": [f"seed gate refused: {seed_gate_error}"],
                                     "details": {"seed_gate_error": seed_gate_error,
                                                 "skipped": "preflight not run: admission refused"}}},
            ))
        print(json.dumps({"ok": False,
                          "failures": [f"seed gate refused: {seed_gate_error}"],
                          "details": {"seed_gate_error": seed_gate_error,
                                      "skipped": "preflight not run: admission refused"}},
                         ensure_ascii=False, indent=2))
        return 2

    with finalize_run_manifest(manifest_path, base_manifest,
                               outcome="completed", exit_code=0) as final:
        result = preflight(
            osworld_root=Path(args.osworld_root).expanduser(),
            ouroboros_url=args.ouroboros_url,
            osworld_server_url=args.osworld_server_url,
            unix_computer_use_payload=Path(args.unix_computer_use_payload).expanduser(),
            unix_computer_use_state_dir=state_dir,
            output_root=output_root,
            repo_root=repo_root,
            data_root=data_root,
        )
        # `extra.runtime_attestation` at the TOP level, exactly like the other two launchers:
        # the documented contract is a single place to read the carried record from, and
        # burying it inside `extra.preflight.details` made this the one site that did not
        # honour it.
        attestation = (result.get("details") or {}).get("runtime_attestation") or {}
        base_manifest.update(amend_task_manifest(
            base_manifest,
            output_paths={"ledger": str(ledger_path), "manifest": str(manifest_path)},
            extra={"preflight": result, "fail_closed": True,
                   "runtime_attestation": attestation,
                   "allow_dirty_seed": bool(args.allow_dirty_seed)},
        ))
        write_result_index(
            ledger_path,
            [
                task_result_row(
                    benchmark="osworld",
                    instance_id=task,
                    status="preflight_passed" if result["ok"] else "blocked",
                    reason_code="preflight_passed" if result["ok"] else "preflight_failed",
                    official_eval_status="not_run",
                    output_paths={"manifest": str(manifest_path), "ledger": str(ledger_path)},
                    error="; ".join(result["failures"]),
                    details={"failures": result["failures"], "fail_closed": True},
                )
                for task in requested
            ],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ok"]:
            # Name the exact attestation reason when that is what refused (same contract as the
            # other two launchers); fall back to the generic preflight reason otherwise.
            attestation_reason = (str(attestation.get("reason") or "")
                                  if isinstance(attestation, dict) and not attestation.get("ok", True)
                                  else "")
            refusal = ({"stage": "runtime_attestation", "reason": attestation_reason,
                        "exit_code": 2} if attestation_reason
                       else {"stage": "preflight", "reason": "preflight_failed", "exit_code": 2})
            final.update({"outcome": "blocked", "exit_code": 2, "refusal": refusal})
            return 2
        print("OSWorld runnable adapter is not implemented in this release; preflight passed only.")
        # 3 == "preflight passed, but there is no runnable adapter to run" — a deliberate,
        # documented status, recorded so the manifest agrees with the process exit code.
        final.update({"outcome": "preflight_only", "exit_code": 3})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
