#!/usr/bin/env python3
"""Isolated old/new PUBLIC find_child_tasks benchmark; never reads live data.

Example: python -I -S scripts/safe_test.py --temp-parent /tmp -- \
  /path/to/python -B scripts/benchmark_child_task_lookup.py --temp-parent /tmp
The parent creates a new fixture under --temp-parent and retains it for audit.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _row(task_id: str, payload: str, *, child: bool = False) -> str:
    return json.dumps({"_schema_version": 1, "task_id": task_id,
                       "status": "completed", "ts": "2026-10-04T00:00:00Z",
                       "delegation_role": "subagent", "parent_task_id": "parent" if child else "other",
                       "root_task_id": "parent" if child else "other", "result": payload})


def _make_fixture(root: Path, count: int, mib: int) -> None:
    results = root / "task_results"
    results.mkdir(parents=True)
    payload = "x" * (mib * 1024 * 1024 // count)
    for i in range(count):
        tid = f"task{i:04d}"
        (results / f"{tid}.json").write_text(_row(tid, payload, child=i < 3), encoding="utf-8")


def _rss_mib() -> float | None:
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _worker(root: Path, mode: str, restart: bool) -> None:
    import ouroboros.task_status as task_status

    if mode == "old":
        # Compile the exact public function from the frozen upstream base in
        # the current module's namespace. This measures its original full-list
        # call and prefilter, without charging it for the new memo navigation.
        base = subprocess.run(
            ["git", "show", "b8233b671:ouroboros/task_status.py"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout
        functions = [node for node in ast.parse(base).body
                     if isinstance(node, ast.FunctionDef) and node.name == "find_child_tasks"]
        if len(functions) != 1:
            raise RuntimeError("frozen upstream has no unique public lookup")
        module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
        # The frozen function names the complete reader the current module no
        # longer imports; bind it so old mode measures the original full scan.
        from ouroboros.task_results import list_task_results
        task_status.__dict__.setdefault("list_task_results", list_task_results)
        exec(compile(module, "b8233b671:ouroboros/task_status.py", "exec"), task_status.__dict__)

    reads = []
    decodes = []
    original_read = Path.read_text
    original_loads = json.loads

    def read_spy(path, *args, **kwargs):
        value = original_read(path, *args, **kwargs)
        if path.parent.name == "task_results" and path.suffix == ".json":
            reads.append(len(value.encode("utf-8")))
        return value

    def loads_spy(value, *args, **kwargs):
        if len(value) >= 100_000:
            decodes.append(len(value))
        return original_loads(value, *args, **kwargs)

    Path.read_text = read_spy
    json.loads = loads_spy

    def measure(label: str, expected: int) -> None:
        reads.clear()
        decodes.clear()
        cpu = time.process_time()
        wall = time.perf_counter()
        rows = task_status.find_child_tasks(root, parent_task_id="parent", scope="direct",
                                            materialize_artifacts=False)
        rss = _rss_mib()
        result = {"mode": mode, "phase": label, "wall_s": round(time.perf_counter() - wall, 6),
                  "cpu_s": round(time.process_time() - cpu, 6),
                  "rss_highwater_mib": round(rss, 2) if rss is not None else None,
                  "body_reads": len(reads), "body_bytes": sum(reads),
                  "large_json_decodes": len(decodes), "large_json_chars": sum(decodes),
                  "children": len(rows)}
        if len(rows) != expected:
            raise AssertionError(result)
        print(json.dumps(result), flush=True)

    if restart:
        measure("restart_cold", 3)
        return
    measure("cold", 3)
    for i in range(1, 4):
        measure(f"warm{i}", 3)
    target = sorted((root / "task_results").glob("task*.json"))[-1]
    row = original_loads(original_read(target))
    row["updated_at"] = "2026-10-04T00:00:01Z"
    replacement = target.with_suffix(".replacement")
    replacement.write_text(json.dumps(row), encoding="utf-8")
    os.replace(replacement, target)
    measure("sparse_update", 3)
    appeared = root / "task_results" / "new_child.json"
    appeared.write_text(_row("new_child", "appeared", child=True), encoding="utf-8")
    measure("appearance", 4)
    appeared.unlink()
    measure("disappearance", 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-parent", type=Path)
    parser.add_argument("--files", type=int, default=1000)
    parser.add_argument("--mib", type=int, default=170)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--mode", choices=("old", "new"))
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.worker_root is not None:
        if args.mode is None:
            parser.error("--worker-root requires --mode")
        _worker(args.worker_root, args.mode, args.restart)
        return
    if args.temp_parent is None or not args.temp_parent.is_dir():
        parser.error("supply an existing --temp-parent for a NEW isolated fixture")
    if args.files < 4 or args.mib < 1:
        parser.error("fixture needs at least 4 files and 1 MiB")
    run = Path(tempfile.mkdtemp(prefix="ob-child-lookup-", dir=args.temp_parent)).resolve()
    source = run / "source"
    _make_fixture(source, args.files, args.mib)
    rows = []
    for mode in ("old", "new"):
        fixture = run / mode
        shutil.copytree(source, fixture, copy_function=os.link)
        for restart in (False, True):
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker-root", str(fixture),
                   "--mode", mode]
            if restart:
                cmd.append("--restart")
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"worker {mode} restart={restart} exited {completed.returncode}:\n"
                                   f"{completed.stdout}\n{completed.stderr}")
            rows.extend(json.loads(line) for line in completed.stdout.splitlines())
    old_warm = statistics.median(row["cpu_s"] for row in rows if row["mode"] == "old" and row["phase"].startswith("warm"))
    new_warm = statistics.median(row["cpu_s"] for row in rows if row["mode"] == "new" and row["phase"].startswith("warm"))
    print(json.dumps({"python": sys.version.split()[0], "executable": sys.executable,
                      "fixture_root": str(run), "files": args.files, "target_mib": args.mib,
                      "warm_cpu_speedup": round(old_warm / new_warm, 2), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
