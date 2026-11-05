#!/usr/bin/env python3
"""Would a retirement lose a product line? Per-test coverage, reduced in the cloud.

A class of tests retires by a rule plus a read (DEVELOPMENT "Retiring and merging tests"). This
is the mechanical half of the read: a deletion candidate that covers a live product line no
survivor covers moves to merge. The raw data is gigabytes, so it never leaves the runner:

1. Measure, on a runner. This module is also a pytest plugin. With `coverage` and `pytest-cov`
   installed (dev-only, by the job itself; nothing here imports them):
       python -m pytest -p devtools.coverage_census --cov=ouroboros --cov=supervisor --cov=launcher \
           --cov=server --cov=scripts --cov=skills --cov=devtools \
           --cov-context=test --cov-config=RC --cov-report= tests/ ...
   RC sets `parallel = true`, an absolute `data_file`, `source_pkgs`, `core = ctrace` and
   `context = ${PYTEST_CURRENT_TEST}`, which is how a child process learns its test.
2. `fold --data DIR --out fold.json.gz` shrinks one shard's data on that runner.
3. `digest fold*.json.gz --out digest.json` merges the shards, still in the cloud.
4. `python -m devtools.coverage_census query digest.json --decisions decisions.csv --out returned.csv`
   runs anywhere: which candidates of `devtools/test_census.py` must stay or merge.

The digest stores a line's covering tests only when no other line has a strict subset of them.
That loses nothing for the question asked: if a kept set keeps a witness, every superset of it
keeps that witness too. A test in no kept set is redundant for line coverage. A set of one is a
line only that test covers.

Only lines inside function bodies count. An import-time line says which test imported first.
What this cannot see, and how it errs:
- A child that is killed, or whose environment a grandchild scrubs, reports nothing. A test whose
  only unique reach is there looks deletable: this errs toward deletion. A `python -S` child too.
- A fixture or a long-lived child shared across tests, and any once-only line (a cache fill), is
  credited to the first test that reached it. That test looks unique: this errs toward keeping.
- A covered line is not a checked line. Coverage finds lost reach, never lost assertions.
- A root it was not told to measure is blind: a test whose only reach is there looks deletable. The
  05.11.2026 digest measured `ouroboros/` and `supervisor/` only; `PRODUCT_ROOTS` now names the
  launcher, the server entry, `scripts/`, `skills/` and `devtools/` too, and the run must pass the
  same roots to `--cov`.
"""
from __future__ import annotations

import argparse
import ast
import csv
import functools
import gzip
import json
import os
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path, PurePosixPath

# Stdlib only at import: pytest loads this before tests/conftest.py draws its isolation boundary.
PRODUCT_ROOTS = ("ouroboros", "supervisor", "launcher.py", "server.py", "scripts", "skills", "devtools")
CHILD_KEYS = ("COVERAGE_PROCESS_START", "PYTEST_CURRENT_TEST")
TEST_PHASE = re.compile(r"(?:\|(?:setup|run|teardown)| \((?:setup|call|teardown)\))$")
LINE_BITS = ("SELECT file.path, context.context, line_bits.numbits FROM line_bits "
             "JOIN file ON file.id = line_bits.file_id JOIN context ON context.id = line_bits.context_id")

_popen_init = subprocess.Popen.__init__


def _carried(env: dict) -> dict:
    return {**env, **{key: os.environ[key] for key in CHILD_KEYS if key in os.environ}}


@functools.wraps(_popen_init)
def _measured_init(self, *args, **kwargs):
    """A test's custom `env={...}` still starts coverage in the child and names the test."""
    if len(args) > 10 and args[10] is not None:
        args = (*args[:10], _carried(args[10]), *args[11:])
    elif kwargs.get("env") is not None:
        kwargs["env"] = _carried(kwargs["env"])
    _popen_init(self, *args, **kwargs)


def pytest_load_initial_conftests():
    """Patch first: tests/conftest.py then wraps this and its restore returns to it."""
    subprocess.Popen.__init__ = _measured_init


def pytest_sessionstart(session):
    """Arm the children. Never before xdist starts its workers: pytest-cov measures those itself."""
    config = session.config
    if hasattr(config, "workerinput") or config.getoption("dist", "no") == "no":
        os.environ["COVERAGE_PROCESS_START"] = os.path.abspath(config.getoption("cov_config"))


def _product_file(root: Path, measured: str) -> str:
    """`ouroboros/x.py` for a file of the checkout or of a copy of it (a hermetic run measures its copy); coverage
    records the measuring OS's own separators."""
    parts = PurePosixPath(measured.replace("\\", "/")).parts
    for start in range(len(parts) - 1, -1, -1):
        if parts[start] in PRODUCT_ROOTS and root.joinpath(*parts[start:]).is_file():
            return "/".join(parts[start:])
    return ""


def _body_lines(source: str) -> set:
    lines = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.update(range(node.body[0].lineno, node.body[-1].end_lineno + 1))
    return lines


def _numbers(numbits: bytes):
    for index, byte in enumerate(numbits):
        if byte:
            yield from (index * 8 + bit for bit in range(8) if byte >> bit & 1)


def _ranges(lines) -> str:
    """`[3, 4, 5, 9]` as `3-5,9`."""
    spans = []
    for line in sorted(lines):
        if spans and line == spans[-1][1] + 1:
            spans[-1][1] = line
        else:
            spans.append([line, line])
    return ",".join(str(first) if first == last else f"{first}-{last}" for first, last in spans)


def _lines(ranges: str):
    for span in ranges.split(","):
        first, _, last = span.partition("-")
        yield from range(int(first), int(last or first) + 1)


def fold(data: Path, root: Path) -> dict:
    """One shard's coverage files as covering sets: {nodes, groups: [[members, {file: ranges}]], unreadable}."""
    covers, nodes, unreadable = defaultdict(set), {}, 0
    files: dict = {}
    bodies: dict = {}
    for database in sorted(path for path in data.rglob(".coverage*") if path.is_file()):
        try:
            with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
                for measured, context, numbits in connection.execute(LINE_BITS):
                    phase = TEST_PHASE.search(context)
                    if measured not in files:
                        files[measured] = _product_file(root, measured)
                    rel = files[measured]
                    if rel and phase and context.startswith("tests/"):
                        node = nodes.setdefault(context[:phase.start()], len(nodes))
                        if rel not in bodies:
                            bodies[rel] = _body_lines((root / rel).read_text(encoding="utf-8"))
                        for line in _numbers(numbits):
                            if line in bodies[rel]:
                                covers[rel, line].add(node)
        except sqlite3.DatabaseError:  # a child killed while it wrote
            unreadable += 1
    groups: dict = defaultdict(lambda: defaultdict(list))
    for (rel, line), members in covers.items():
        groups[frozenset(members)][rel].append(line)
    return {"nodes": list(nodes), "unreadable": unreadable,
            "groups": [[sorted(members), {rel: _ranges(lines) for rel, lines in by_file.items()}]
                       for members, by_file in groups.items()]}


def _mask(members, width: int) -> int:
    raw = bytearray(width)
    for member in members:
        raw[member >> 3] |= 1 << (member & 7)
    return int.from_bytes(raw, "little")


def _members(mask: int) -> list:
    return list(_numbers(mask.to_bytes(mask.bit_length() // 8 + 1, "little")))


def minimal_sets(masks) -> list:
    """The bitmasks with no strict subset among `masks`, in order of size then value."""
    kept, by_lowest = [], defaultdict(list)
    for mask in sorted(masks, key=lambda mask: (mask.bit_count(), mask)):
        members = _members(mask)
        if not any(other & mask == other for member in members for other in by_lowest.get(member, ())):
            kept.append(mask)
            by_lowest[members[0]].append(mask)  # a kept subset of `mask` has its lowest member inside `mask`
    return kept


def digest(folds: list) -> dict:
    """The shards' folds as the small exact file of the module docstring."""
    names = sorted({name for one in folds for name in one["nodes"]})
    positions = {name: position for position, name in enumerate(names)}
    width = len(names) // 8 + 1
    covers: dict = defaultdict(int)
    for one in folds:
        for members, files in one["groups"]:
            mask = _mask((positions[one["nodes"][member]] for member in members), width)
            for rel, ranges in files.items():
                for line in _lines(ranges):
                    covers[rel, line] |= mask
    by_set: dict = defaultdict(lambda: defaultdict(list))
    for (rel, line), mask in covers.items():
        by_set[mask][rel].append(line)
    kept = minimal_sets(by_set)
    witnesses = sorted({member for mask in kept for member in _members(mask)})
    renumbered = {member: position for position, member in enumerate(witnesses)}
    groups = [{"nodes": [renumbered[member] for member in _members(mask)],
               "lines": {rel: _ranges(lines) for rel, lines in sorted(by_set[mask].items())}} for mask in kept]
    return {"format": 1, "nodes": [names[member] for member in witnesses], "groups": groups, "totals": {
        "tests_measured": len(names), "lines_covered": len(covers), "covering_sets": len(by_set),
        "kept_sets": len(kept), "single_witness_lines": sum(
            len(list(_lines(ranges))) for group in groups if len(group["nodes"]) == 1
            for ranges in group["lines"].values()),
        "unreadable_files": sum(one["unreadable"] for one in folds)}}


def query(report: dict, deleted: set) -> list:
    """What must stay if `deleted` node ids go: [{node, lines}], the witness of the most lost sets first.

    A greedy hitting set: every kept set whose witnesses are all deleted gets one witness back.
    """
    names = report["nodes"]
    gone = {position for position, name in enumerate(names) if name in deleted}
    lost = [group for group in report["groups"] if gone.issuperset(group["nodes"])]
    returned = []
    while lost:
        counts = Counter(member for group in lost for member in group["nodes"])
        best = min(counts, key=lambda member: (-counts[member], names[member]))
        files: dict = defaultdict(list)
        for group in lost:
            if best in group["nodes"]:
                for rel, ranges in group["lines"].items():
                    files[rel] += _lines(ranges)
        returned.append({"node": names[best],
                         "lines": " ".join(f"{rel}:{_ranges(files[rel])}" for rel in sorted(files))})
        lost = [group for group in lost if best not in group["nodes"]]
    return returned


def _candidates(report: dict, decisions: Path | None, deleted: Path | None) -> dict:
    """node id -> what proposed deleting it: a census row (all of a trimmed function's cases) or a list."""
    from devtools.test_census import DELETED, function_key

    candidates: dict = {}
    if decisions:
        with decisions.open(newline="", encoding="utf-8") as stream:
            rows = {(row["file"], row["qual"]): row["cluster"] for row in csv.DictReader(stream)
                    if row["cluster"] in DELETED}
        candidates.update({name: rows[function_key(name)] for name in report["nodes"] if function_key(name) in rows})
    if deleted:
        candidates.update(dict.fromkeys(deleted.read_text(encoding="utf-8").splitlines(), "listed"))
    return candidates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    folding = commands.add_parser("fold", help="one shard's coverage files to fold.json.gz")
    folding.add_argument("--data", type=Path, required=True)
    folding.add_argument("--root", type=Path, default=Path("."))
    folding.add_argument("--out", type=Path, required=True)
    merging = commands.add_parser("digest", help="every shard's fold to the small digest")
    merging.add_argument("folds", type=Path, nargs="+")
    merging.add_argument("--out", type=Path, required=True)
    asking = commands.add_parser("query", help="what deletion candidates must stay")
    asking.add_argument("digest", type=Path)
    asking.add_argument("--decisions", type=Path, help="decisions.csv of devtools/test_census.py")
    asking.add_argument("--deleted", type=Path, help="node ids deleted, one per line")
    asking.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "query" and not (args.decisions or args.deleted):
        parser.error("query needs --decisions or --deleted: an empty question has a reassuring answer")
    if args.command == "fold":
        with gzip.open(args.out, "wt", encoding="utf-8") as stream:
            json.dump(fold(args.data, args.root), stream)
    elif args.command == "digest":
        folds = []
        for path in args.folds:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                folds.append(json.load(stream))
        report = digest(folds)
        args.out.write_text(json.dumps(report) + "\n", encoding="utf-8")
        print(json.dumps(report["totals"]))
    else:
        report = json.loads(args.digest.read_text(encoding="utf-8"))
        candidates = _candidates(report, args.decisions, args.deleted)
        returned = query(report, set(candidates))
        with args.out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("node", "cluster", "lines"), lineterminator="\n")
            writer.writeheader()
            writer.writerows({**row, "cluster": candidates[row["node"]]} for row in returned)
        print(f"{len(returned)} of {len(candidates)} candidates return:",
              dict(Counter(candidates[row["node"]] for row in returned)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
