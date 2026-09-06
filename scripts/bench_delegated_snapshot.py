#!/usr/bin/env python3
"""Wall-clock benchmark for C1 delegated execution-snapshot provisioning.

Phase-C acceptance criterion: the per-run snapshot cost on a LARGE WARM tree
must be known before landing. Run it against a THROWAWAY clone (the benchmark
creates and deletes baseline refs and worktree registrations in the target's
.git — never point it at a live tree you care about):

    git clone --no-hardlinks /path/to/big-repo /tmp/bench-target
    python scripts/bench_delegated_snapshot.py /tmp/bench-target --runs 5

Reports per-run provision (baseline build + worktree checkout) and teardown
wall-clock, plus tree shape (file count, on-disk size). The delegated-snapshot
lifecycle this measures is documented in docs/ARCHITECTURE.md,
"Delegated subagents".
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _tree_shape(target: pathlib.Path) -> tuple[int, int]:
    files = 0
    size = 0
    for path in target.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            files += 1
            try:
                size += path.stat().st_size
            except OSError:
                pass
    return files, size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="throwaway git working tree to snapshot")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    target = pathlib.Path(args.target).resolve()
    if not (target / ".git").exists():
        print(f"error: {target} is not a git working tree", file=sys.stderr)
        return 2

    from ouroboros.subagent_worktrees import (
        provision_execution_snapshot,
        remove_execution_snapshot,
    )

    files, size = _tree_shape(target)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(target),
                           capture_output=True, text=True).stdout.count("\n")
    print(f"target: {target}")
    print(f"tree: {files} files, {size / 1e6:.1f} MB (excl. .git), {dirty} dirty entries")

    with tempfile.TemporaryDirectory(prefix="bench_snap_") as scratch:
        scratch_path = pathlib.Path(scratch)
        provisions: list[float] = []
        removals: list[float] = []
        for i in range(max(1, args.runs)):
            snap_id = f"bench{i}"
            t0 = time.perf_counter()
            handle = provision_execution_snapshot(
                target_root=target, task_id="bench", snapshot_id=snap_id,
                worktree_root=scratch_path / "snaps", data_dir=scratch_path / "data",
            )
            t1 = time.perf_counter()
            remove_execution_snapshot(
                snap_id, worktree_root=scratch_path / "snaps",
                data_dir=scratch_path / "data",
            )
            t2 = time.perf_counter()
            provisions.append(t1 - t0)
            removals.append(t2 - t1)
            print(f"run {i}: provision={t1 - t0:.3f}s  teardown={t2 - t1:.3f}s  "
                  f"baseline={handle.baseline_sha[:12]}  entries={handle.entry_count}")
        warm = provisions[1:] or provisions
        print(f"provision: cold={provisions[0]:.3f}s  "
              f"warm avg={sum(warm) / len(warm):.3f}s  "
              f"warm max={max(warm):.3f}s over {len(warm)} run(s)")
        print(f"teardown:  avg={sum(removals) / len(removals):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
