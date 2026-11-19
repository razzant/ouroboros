#!/usr/bin/env python3
"""Freeze one run's shard weights: the single table every sharded leg and manifest of that run packs by.

    python3 -I -S scripts/ci_shard_plan.py --out DIR --repo OWNER/NAME --run-id ID --run-attempt N --sha SHA

A run prepares its plan once, before its first leg (`.github/actions/shard-plan`, called by the
`shard-plan` job of ci.yml and of the manual full UI entry); a re-run keeps what its first attempt
published, so legs of different attempts pack alike. No leg ever reads the history itself.

The history is the repository's `ci-durations` artifacts (tests/ci_verdicts.py `collect_durations`),
read through `gh api` page by page, newest first, only until the plan holds what it wants. Prior only:
a table counts if another run's artifact existed before THIS run's original creation, so neither this
run, nor any of its attempts, nor a table that appeared while it waited weighs it. A table speaks only
for its artifact's own run: its `source` must name the run id and the `head_sha` the listing gives that
artifact (never the commit being planned, which no earlier run measured). DIR receives:

- `ci-durations.json`, the format-1 table `durations_table` reads, when some table weighs anything:
  `platforms` exactly as the newest usable table with file times has them (so a table that holds only
  UI nodes cannot displace them), and `ui_nodes` the per-node `median_low` of the newest three usable
  tables that record UI nodes, one table per run. A node's median is over the tables that time it; each
  weight is an integer millisecond some table observed. `source.run_id` is this plan,
  `shard-plan-<run id>` (the artifact holding `plan.json`), since no single run measured these values.
- `plan.json`, what the table was made of: the target run, each source's artifact, run, commit and
  creation time, how many tables timed each node (`ui_node_observations`) and every artifact it
  looked at and passed over, with why. With nothing usable `durations` is null, `reason` says why
  and no table is written: that is the plan, frozen and shared like any other, and the legs keep
  their positions or node counts.

The search is bounded, so that the plan reaches its upload inside its job's 10 minutes: BUDGET_SEC for
all its reads together (the run's own record, the listing and the tables; a read still running when
they end is stopped), PAGES listing pages and READS tables. A search cut short uses what it verified
and says why in `plan.json`'s `stopped_early` (null when it ended by itself). Stdlib only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # -I leaves the checkout off sys.path.
    sys.path.insert(0, str(REPO))

from tests.ci_verdicts import _table_shape_holds, write_json  # noqa: E402  -- stdlib only

ARTIFACT, TABLE, PLAN = "ci-durations", "ci-durations.json", "plan.json"
UI_TABLES = 3
PAGE, PAGES = 100, 5  # GitHub's largest page, and how many the search may list: 90 days of tables are far fewer.
READS = 30  # Tables it may download: the plan wants at most four, the others pass over unusable ones.
BUDGET_SEC = 300  # All of one plan's reads: half of its job's 10 minutes, the rest is for checkout and upload.
_GH_SECONDS = 120  # One read.
_UNREADABLE = (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError,
               zipfile.BadZipFile, zlib.error)


def gh(path: str, seconds: float) -> bytes:
    """The body of one `gh api` GET: GitHub's own client, which follows an artifact's storage redirect."""
    return subprocess.run(["gh", "api", path], capture_output=True, check=True, timeout=seconds).stdout


def _when(value) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _candidates(listed, *, run_id: str, cutoff: datetime) -> tuple[list[dict], list[dict]]:
    """(this run's usable artifact rows newest first, the rest with why)."""
    rows, skipped = [], []
    for artifact in listed:
        try:
            run = artifact["workflow_run"]
            row = {"artifact": artifact["id"], "run_id": str(run["id"]), "sha": str(run.get("head_sha") or ""),
                   "created_at": artifact["created_at"]}
            created = _when(row["created_at"])
        except _UNREADABLE as error:
            skipped.append({"reason": f"an unreadable listing row ({type(error).__name__})"})
            continue
        reason = ("expired" if artifact.get("expired") else "this run's own" if row["run_id"] == run_id
                  else "created after this run" if created >= cutoff else "")
        if reason:
            skipped.append({**row, "reason": reason})
        else:
            rows.append((created, row))
    rows.sort(key=lambda item: (item[0], item[1]["artifact"]), reverse=True)
    return [row for _, row in rows], skipped


def prepare(repo: str, run_id: str, sha: str, *, attempt: str = "", read=gh,
            clock=time.monotonic) -> tuple[dict, dict | None]:
    """(plan.json, the frozen table or None): the module's contract, over `read(api path, seconds) -> bytes`."""
    target = {"run_id": run_id, "run_attempt": attempt, "sha": sha}
    deadline = clock() + BUDGET_SEC

    def fetch(path):
        return read(path, min(_GH_SECONDS, max(deadline - clock(), 0)))

    def artifacts_on(page):
        path = f"repos/{repo}/actions/artifacts?name={ARTIFACT}&per_page={PAGE}&page={page}"
        return json.loads(fetch(path))["artifacts"]

    def positional(reason, skipped=(), stopped=None):
        return {"format": 1, "durations": None, "reason": reason, "target": target, "skipped": list(skipped),
                "stopped_early": stopped}, None
    try:
        created = json.loads(fetch(f"repos/{repo}/actions/runs/{run_id}"))["created_at"]
        cutoff = _when(created)
        listed = artifacts_on(1)
        rows, skipped = _candidates(listed, run_id=run_id, cutoff=cutoff)
    except _UNREADABLE as error:
        return positional(f"the artifact history is unavailable ({type(error).__name__})")
    target["created_at"] = created
    ui, platform, seen, page, reads, stopped = [], None, set(), 1, 0, None
    while len(ui) < UI_TABLES or platform is None:
        if clock() >= deadline:
            stopped = f"its {BUDGET_SEC} s of reads ran out"
            break
        if not rows:  # GitHub lists newest first; a row a new artifact pushed onto the next page meets `seen`.
            if len(listed) < PAGE:
                break
            if page == PAGES:
                stopped = f"the {PAGES} listing pages it may read are used up"
                break
            page += 1
            try:
                listed = artifacts_on(page)
                rows, passed = _candidates(listed, run_id=run_id, cutoff=cutoff)
            except _UNREADABLE as error:
                stopped = f"listing page {page} is unreadable ({type(error).__name__})"
                break
            skipped += passed
            continue
        row = rows.pop(0)
        if row["run_id"] in seen:
            skipped.append({**row, "reason": "a newer artifact of its run was read"})
            continue
        if reads == READS:
            stopped = f"its {READS} table reads are used up"
            break
        reads += 1
        seen.add(row["run_id"])
        try:
            with zipfile.ZipFile(io.BytesIO(fetch(f"repos/{repo}/actions/artifacts/{row['artifact']}/zip"))) as archive:
                table = json.loads(archive.read(TABLE).decode("utf-8"))
        except _UNREADABLE as error:
            skipped.append({**row, "reason": f"unreadable ({type(error).__name__})"})
            continue
        # Only the artifact's own run and commit identify a distinct measurement: a copy or a plan is not one.
        if not _table_shape_holds(table) or table["source"]["run_id"] != row["run_id"]:
            skipped.append({**row, "reason": "not a table this run recorded"})
            continue
        if not row["sha"] or table["source"].get("sha") != row["sha"]:
            skipped.append({**row, "reason": "not a table of this run's commit"})
            continue
        wants_ui = bool(table.get("ui_nodes")) and len(ui) < UI_TABLES
        wants_files = platform is None and any(table["platforms"].values())
        if wants_ui:
            ui.append((row, table))
        if wants_files:
            platform = (row, table)
        if not wants_ui and not wants_files:
            skipped.append({**row, "reason": "no weight this plan still needs"})
    if not ui and platform is None:
        return positional(f"no earlier {ARTIFACT} table is usable", skipped, stopped)
    observed: dict[str, list[int]] = {}
    for _, table in ui:
        for node, milliseconds in table["ui_nodes"].items():
            observed.setdefault(node, []).append(milliseconds)
    frozen = {"format": 1, "platforms": platform[1]["platforms"] if platform else {},
              "source": {"run_id": f"shard-plan-{run_id}", "sha": sha},
              "ui_nodes": {node: statistics.median_low(observed[node]) for node in sorted(observed)}}
    return {"format": 1, "durations": TABLE, "target": target,
            "platform_source": platform[0] if platform else None,
            "ui_sources": [{**row, "ui_nodes": len(table["ui_nodes"])} for row, table in ui],
            "ui_node_observations": {node: len(observed[node]) for node in sorted(observed)},
            "skipped": skipped, "stopped_early": stopped}, frozen


def _summary(plan: dict) -> str:
    stopped = f" Its search stopped early: {plan['stopped_early']}." if plan["stopped_early"] else ""
    if plan["durations"] is None:
        return f"Shard plan: {plan['reason']}; shards keep their positions and node counts.{stopped}"
    sources, counts = plan["ui_sources"], Counter(plan["ui_node_observations"].values())
    ui = ("UI weights: median_low of runs " + ", ".join(row["run_id"] for row in sources) + " ("
          + ", ".join(f"{counts[k]} node(s) timed by {k}" for k in sorted(counts, reverse=True)) + ")"
          if sources else "no earlier table records UI nodes")
    platform = plan["platform_source"]
    files = f"file weights: run {platform['run_id']}" if platform else "no earlier table records file times"
    return f"Shard plan shard-plan-{plan['target']['run_id']}: {ui}; {files}.{stopped}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ci_shard_plan.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    for flag in ("--repo", "--run-id", "--run-attempt", "--sha"):
        parser.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    plan, table = prepare(args.repo, args.run_id, args.sha, attempt=args.run_attempt)
    if table is not None:
        write_json(args.out / TABLE, table)
    write_json(args.out / PLAN, plan)
    line = _summary(plan)
    print(line)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
