#!/usr/bin/env python3
"""Export the usage store as the journal an older Ouroboros release reads.

Usage: python scripts/export_usage_journal.py <data_root>

Run it OFFLINE, with the Ouroboros server stopped, as the step before checking
out a release older than the usage store (docs/USAGE_STORE.md "Downgrade"). It
writes ``state/usage_attempts.jsonl`` from ``state/usage.sqlite`` (a chain the
older validator accepts: each attempt's minimal legal transition chain ending in
its current row, imported aggregates under their baseline header, dense seq),
aligns the ``state.json`` freshness marker with it, and renames the store aside
so the next upgrade imports the journal again. A git revert alone is not a data
rollback. Prints one JSON report.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    root = pathlib.Path(argv[0]).expanduser().resolve()
    from ouroboros.usage_store import export_journal

    print(json.dumps(export_journal(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
