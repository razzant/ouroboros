#!/usr/bin/env python3
"""Generate tests/protected_tests.json: the tests no retirement rule may take.

    GH_TOKEN=... python -m devtools.protected_tests [--since 2026-10-22] [--keep-list keep.csv]

The file has three sections:

files      Whole files: the gate floor (`test_selection.FLOOR_TESTS`, read live), then the release,
           security, process-safety and system witnesses of a keep list. A rerun without
           --keep-list carries these rows over, so the file is its own source.
functions  `file::qualname` -> issue numbers: every function named in a post-merge-red, FLAKY or
           ACCEPTED record created since `since`, from issue bodies (read-only) and
           `tests/accepted_failures.json`. Such a function keeps all its cases: it is never deleted
           or trimmed, and folded only with every case and its real process intact.
witnesses  `file::qualname` -> issue numbers: functions that cite an issue in their docstring or
           body at this tree. A row of a keep list that only cites an issue is one of these, not
           a whole file. The closest witness of a cited defect stays; its cartesian rows may go.
           The census finds citations in the source itself; this section is for a reader.

A retirement step subtracts `files` and `functions` mechanically, and the census marks what it
spared. The file is rewritten in place and bounded by `since` (about 140 KB for one release
cycle). `gh` must see GH_TOKEN; a failed query fails the run rather than protect less.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from devtools.test_census import function_key, scan_tests
from ouroboros.test_selection import FLOOR_TESTS

REPOSITORY = "razzant/ouroboros"
FLOOR_REASON = "gate floor"
NODE_ID = re.compile(r"((?:\w+/)*test_\w+\.py)::((?:\w+::)*test_\w+)")


def fetch_records(repository: str, since: str) -> list:
    """Issue bodies of the records since `since`: post-merge reds, ACCEPTED and FLAKY reports."""
    records = {}
    for label, words in (("post-merge-red", ""), ("not-a-product-bug", ""), ("", "flaky ")):
        command = ["gh", "issue", "list", "--repo", repository, "--state", "all", "--limit", "1000",
                   "--json", "number,body", "--search", f"{words}created:>={since}"]
        done = subprocess.run(command + (["--label", label] if label else []), capture_output=True, text=True,
                              encoding="utf-8", check=True)
        records.update({record["number"]: record for record in json.loads(done.stdout)})
    return [records[number] for number in sorted(records)]


def named_functions(records: list, accepted: list) -> dict:
    """`file::qualname` -> issue numbers, whatever parametrize case a record named."""
    named = defaultdict(set)
    for record in records:
        for match in NODE_ID.finditer(record.get("body") or ""):
            named[f"tests/{match[1].rpartition('tests/')[2]}::{match[2]}"].add(record["number"])
    for entry in accepted:
        named["::".join(function_key(entry["nodeid"]))].add(entry["issue"])
    return {key: sorted(numbers) for key, numbers in named.items()}


def keep_list_files(path: Path) -> dict:
    """Whole-file rows of a keep list (columns file, reason); a row citing an issue is no whole file."""
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["file"]: row["reason"].split(" (")[0] for row in csv.DictReader(stream)
                if not row["reason"].startswith("cites issue")}


def protected_files(root: Path, previous: dict, keep_list: dict) -> dict:
    """The whole-file section; a file that left the floor or the tree leaves with it."""
    files = {path: why for path, why in previous.items() if not why.startswith(FLOOR_REASON)}
    files.update(keep_list)
    files.update({path: f"{FLOOR_REASON}: {why}" for path, why in FLOOR_TESTS.items()})
    return {path: why for path, why in files.items() if (root / path).is_file()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="checkout the set is generated for")
    parser.add_argument("--since", help="oldest record date, default: the date in the existing file")
    parser.add_argument("--keep-list", type=Path, help="CSV of whole-file keep rows to add (columns file, reason)")
    parser.add_argument("--repository", default=REPOSITORY)
    args = parser.parse_args(argv)
    output = args.root / "tests" / "protected_tests.json"
    previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    since = args.since or previous.get("since")
    if not since:
        parser.error("--since is required until the file exists")
    accepted = json.loads((args.root / "tests" / "accepted_failures.json").read_text(encoding="utf-8"))["entries"]
    records = fetch_records(args.repository, since)
    functions = named_functions(records, accepted)
    rows = scan_tests(args.root)
    witnesses = {f"{row['file']}::{row['qual']}": [int(number) for number in row["cited"].split()]
                 for row in rows if row["cited"] and not row["lane_marked"]}
    keep_list = keep_list_files(args.keep_list) if args.keep_list else {}
    files = protected_files(args.root, previous.get("files", {}), keep_list)
    output.write_text(json.dumps({"format": 1, "since": since, "files": files, "functions": functions,
                                  "witnesses": witnesses}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    known = {f"{row['file']}::{row['qual']}" for row in rows}
    print(f"{len(records)} records since {since}; {len(files)} files, {len(functions)} named functions "
          f"({len(set(functions) - known)} not statically visible at this tree), {len(witnesses)} witnesses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
