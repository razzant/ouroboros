#!/usr/bin/env python3
"""Copy ProgramBench submissions into a separate run dir for targeted eval.

Creates a minimal ``programbench eval``-compatible tree::

    <dest_run>/<instance_id>/submission.tar.gz

Only instances with a non-empty ``submission.tar.gz`` in the source run are copied.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import write_json
from devtools.benchmarks.common.run_roots import ensure_outside_repo, run_root, safe_join_under
from devtools.benchmarks.programbench.run_programbench_e2e import _load_instances


def _parse_instance_ids(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def export_submissions(
    *,
    source_run: pathlib.Path,
    dest_run: pathlib.Path,
    instance_ids: list[str],
    overwrite: bool,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    skipped: list[str] = []
    missing: list[str] = []

    for instance_id in instance_ids:
        src_sub = safe_join_under(source_run, instance_id) / "submission.tar.gz"
        if not src_sub.is_file() or src_sub.stat().st_size <= 0:
            missing.append(instance_id)
            continue
        dest_dir = safe_join_under(dest_run, instance_id)
        dest_sub = dest_dir / "submission.tar.gz"
        if dest_sub.is_file() and dest_sub.stat().st_size > 0 and not overwrite:
            skipped.append(instance_id)
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_sub, dest_sub)
        copied[instance_id] = str(dest_sub)

    return {
        "source_run": str(source_run),
        "dest_run": str(dest_run),
        "requested": len(instance_ids),
        "copied": len(copied),
        "skipped_existing": len(skipped),
        "missing_submission": len(missing),
        "copied_ids": sorted(copied.keys()),
        "skipped_ids": sorted(skipped),
        "missing_ids": sorted(missing),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ProgramBench submissions to a separate run dir")
    parser.add_argument("--repo-dir", default=str(pathlib.Path(__file__).resolve().parents[3]))
    parser.add_argument("--source-run-id", required=True, help="existing run under bench_runs/programbench/")
    parser.add_argument("--dest-run-id", required=True, help="export run id (created under bench_runs/programbench/)")
    parser.add_argument("--instance-id", action="append", default=[], help="repeatable instance id filter")
    parser.add_argument("--instance-ids", default="", help="comma-separated instance ids")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--slice", default="", dest="slice_spec", help="e.g. 0:25 or 50:65")
    parser.add_argument("--filter", default="", dest="filter_spec")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve(strict=False)
    source_run = ensure_outside_repo(run_root("programbench", args.source_run_id), repo_dir)
    dest_run = ensure_outside_repo(run_root("programbench", args.dest_run_id), repo_dir)

    explicit = list(args.instance_id or []) + _parse_instance_ids(args.instance_ids)
    if explicit:
        instance_ids = explicit
    else:
        instances = _load_instances(
            difficulty="",
            filter_spec=str(args.filter_spec or ""),
            slice_spec=str(args.slice_spec or ""),
            instance_id="",
            shuffle=bool(args.shuffle),
        )
        instance_ids = [str(item["instance_id"]) for item in instances]

    if not instance_ids:
        raise SystemExit("no instance ids selected")

    summary = export_submissions(
        source_run=source_run,
        dest_run=dest_run,
        instance_ids=instance_ids,
        overwrite=bool(args.overwrite),
    )
    write_json(dest_run / "export_manifest.json", summary)
    write_json(
        dest_run / "instance_order.json",
        {
            "count": len(instance_ids),
            "shuffle": bool(args.shuffle),
            "shuffle_seed": 42 if args.shuffle else None,
            "instance_ids": instance_ids,
            "exported_from": str(source_run),
        },
    )
    print(json.dumps(summary, indent=2))
    print(dest_run)
    return 0 if summary["copied"] or summary["skipped_existing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
