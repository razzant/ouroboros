#!/usr/bin/env python3
"""Phase 0 — evolution smoke (verify the fix FIRST), server-driven + isolated.

On a THROWAWAY clone + ISOLATED data root + a REAL isolated server.py (the live
Ouroboros is never touched), this exercises the production worker pipeline (unlike a
headless `ouroboros run`, which silently attaches to a live server if one is up).

HARD acceptance gates (the command's exit status):
  1. the per-task budget resets between tasks so a fresh task is NOT falsely flagged
     `budget: emergency` (via the guarded supervisor.state.reset_per_task_budget), and
  2. the live repo working tree is untouched.
DIAGNOSTICS (recorded in the ledger, NOT gating exit status): growth of the isolated
memory/knowledge_history.jsonl (memory-carry signal: the mind authors durable lessons
itself through knowledge_write, so growth is informational, not a hard proof — a task
may teach nothing worth writing) and --self-mod absorb (see below).

  --self-mod additionally exercises the real supervisor evolution loop (os.execvpe
  restart + absorb machinery). Project facts remain in the per-project store; absorb
  may occur when an enabled promotion chooses a concrete improvement. Proofs #1 and
  #2 hold regardless of whether a promotion is chosen.

Usage (from repo/):
  python -m devtools.benchmarks.evolve_smoke --tasks 2 --timeout 300 [--self-mod] [--keep]

Nothing here writes under repo/ or the live data dir; outputs go to a temp run root
(validated by devtools.benchmarks.common.run_roots).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from devtools.benchmarks.common.run_roots import ensure_outside_repo
from devtools.benchmarks.common.server_runner import (
    IsolatedServer,
    absorbed_cycles_done,
    build_isolated_settings,
    seed_owner_state,
)

REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
LIVE_DATA = pathlib.Path.home() / "Ouroboros" / "data"


def _log(msg: str) -> None:
    print(f"[evolve_smoke] {msg}", flush=True)


def _git(args: list[str], cwd: pathlib.Path) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _knowledge_revisions_count(data_root: pathlib.Path) -> int:
    path = data_root / "memory" / "knowledge_history.jsonl"
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8")) if path.exists() else 0
    except OSError:
        return 0


def _make_workspace(run_root: pathlib.Path, idx: int) -> pathlib.Path:
    ws = run_root / f"ws{idx}"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "notes.md").write_text(f"# scratch workspace {idx}\n", encoding="utf-8")
    _git(["init", "-q"], ws)
    _git(["add", "-A"], ws)
    _git(["-c", "user.email=smoke@local", "-c", "user.name=smoke", "commit", "-q", "-m", "seed"], ws)
    return ws


def _seed_settings(data_root: pathlib.Path, self_mod: bool) -> pathlib.Path:
    settings_path = data_root / "settings.json"
    live_cfg: dict = {}
    live = LIVE_DATA / "settings.json"
    if live.exists():
        try:
            live_cfg = json.loads(live.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            live_cfg = {}
    # Copy ONLY provider/model/budget keys (no owner secrets), then apply isolated overrides.
    overrides = {
        "OUROBOROS_RUNTIME_MODE": "advanced",
        "OUROBOROS_POST_TASK_EVOLUTION": "true" if self_mod else "false",
    }
    if self_mod:
        overrides["OUROBOROS_POST_TASK_EVOLUTION_CADENCE"] = "every_n:1"
    cfg = build_isolated_settings(live_cfg, **overrides)
    cfg.setdefault("TOTAL_BUDGET", 25.0)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return settings_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 evolution smoke (server-driven, isolated).")
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--self-mod", action="store_true", help="enable post-task evolution + check one absorbed cycle")
    ap.add_argument("--keep", action="store_true", help="keep the temp run root")
    args = ap.parse_args()

    run_root = pathlib.Path(tempfile.mkdtemp(prefix="evolve_smoke_"))
    ensure_outside_repo(run_root, REPO_DIR)
    clone = run_root / "clone"
    data_root = run_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    ensure_outside_repo(data_root, REPO_DIR)
    _log(f"run root: {run_root}")

    rc, out = _git(["clone", "--no-hardlinks", "-q", str(REPO_DIR), str(clone)], run_root)
    if rc != 0:
        _log(f"clone failed: {out}")
        return 2
    # -B guarantees the 'ouroboros' branch (safe_restart's BRANCH_DEV) exists at the
    # cloned HEAD even from a tag/detached/other-branch source; fail loudly if git errs.
    rc, out = _git(["checkout", "-B", "ouroboros"], clone)
    if rc != 0:
        _log(f"checkout -B ouroboros failed: {out}")
        return 2
    # Isolation: drop origin (== live REPO_DIR) so an isolated evolution self-mod
    # commit's _auto_push -> push_to_remote can NEVER push back to the live repo.
    _git(["remote", "remove", "origin"], clone)
    settings_path = _seed_settings(data_root, args.self_mod)
    # Seed only owner_chat_id; the post-task loop enables the one-shot campaign itself.
    seed_owner_state(data_root)
    os.environ["OUROBOROS_DATA_DIR"] = str(data_root)

    from supervisor import state as sstate

    # Mark this throwaway data root as an isolated benchmark root so the guarded
    # reset_per_task_budget will operate on it — and refuse any live root, which lacks it.
    (data_root / sstate.ISOLATED_BENCHMARK_SENTINEL).write_text("isolated benchmark data root\n", encoding="utf-8")

    live_status_before = _git(["status", "--porcelain"], REPO_DIR)[1]
    refl_before = _knowledge_revisions_count(data_root)
    emergency_seen = False
    budget_resets: list[bool] = []
    absorbed_before = absorbed_cycles_done(data_root)

    server = IsolatedServer(clone, data_root, settings_path)
    try:
        _log(f"starting isolated server on {server.base_url} …")
        server.start(ready_timeout=240)
        for i in range(1, int(args.tasks) + 1):
            ws = _make_workspace(run_root, i)
            prompt = (f"In the workspace file notes.md, append a single concise factual bullet "
                      f"about task #{i}. Keep it tiny. Do not modify anything else.")
            _log(f"task {i}: submitting (workspace, forked) …")
            task_id = server.submit(prompt, workspace_root=str(ws), memory_mode="forked", timeout_sec=args.timeout)
            result = server.wait_task(task_id, timeout=args.timeout + 300)
            if str(result.get("status") or "") == "timeout":
                # Mirror benchmark drivers: cancel a timed-out task and wait for a real terminal
                # status BEFORE budget reset / next task, so neither races a live worker.
                server.cancel_task(task_id)
                result = server.wait_task(task_id, timeout=300)
                if str(result.get("status") or "") not in ("completed", "failed", "cancelled", "rejected_duplicate"):
                    _log(f"task {i}: did not terminate after cancel (still {result.get('status')}); aborting smoke")
                    return 2
            text = json.dumps(result).lower()
            emerg = ("budget: emergency" in text) or ("budget exhausted" in text)
            emergency_seen = emergency_seen or emerg
            _log(f"task {i}: status={result.get('status')} emergency={emerg} knowledge_revisions={_knowledge_revisions_count(data_root)}")
            did_reset = sstate.reset_per_task_budget(data_root, confirm_isolated=True)
            budget_resets.append(bool(did_reset))
            if args.self_mod and i < int(args.tasks):
                absorb = server.wait_for_absorb(server.current_sha(), absorbed_cycles_done(data_root), timeout=args.timeout)
                _log(f"task {i}: self-evolution absorbed={absorb.get('absorbed')}")
    finally:
        teardown_unconfirmed = server.stop()

    refl_after = _knowledge_revisions_count(data_root)
    live_status_after = _git(["status", "--porcelain"], REPO_DIR)[1]
    acceptance = {
        "knowledge_revised": refl_after > refl_before,
        "no_budget_emergency": not emergency_seen,
        "budget_reset_worked": all(budget_resets) if budget_resets else False,
        "live_repo_untouched": live_status_before == live_status_after,
        "knowledge_rows_before": refl_before, "knowledge_rows_after": refl_after,
        "absorbed_cycles": absorbed_cycles_done(data_root) - absorbed_before,
    }
    ledger_path = run_root / "evolve_smoke_ledger.json"
    ledger_path.write_text(json.dumps({"run_root": str(run_root), "acceptance": acceptance}, indent=2), encoding="utf-8")
    _log(f"acceptance: {json.dumps(acceptance)}")
    _log(f"ledger: {ledger_path}")

    ok = acceptance["no_budget_emergency"] and acceptance["budget_reset_worked"] and acceptance["live_repo_untouched"]
    if not args.keep and not teardown_unconfirmed:  # an unconfirmed stop keeps its root for inspection
        shutil.rmtree(run_root, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
