#!/usr/bin/env python3
"""Span-substitution helper for version-carrier conflicts during tactical
rebases of the v7 branch (owner-ratified: spec §1.9-10, batch №8 answer 6=A).

Standalone operator tooling — NOT runtime. When a `git rebase` (or merge) of
the v7 branch stops on the release carriers (VERSION, pyproject.toml, uv.lock,
web/package.json, web/modules/api_types.js GATEWAY_CONTRACT_VERSION, the
README badge, the README Version History block, the docs/ARCHITECTURE.md
header), this helper resolves each conflicted carrier file by span
substitution: the preferred side — 'ours' by default, which during a rebase is
the side being rebased ONTO (index stage 2) — wins INSIDE the declared carrier
spans, and everything else in the file merges as an ordinary textual 3-way.
A file whose anchors are malformed or duplicated, or which conflicts OUTSIDE
its carrier spans, is left exactly as git left it, for manual resolution.
Non-carrier conflicted files are never touched.

The engine and the span descriptors are the SAME ones the managed-update
runtime uses: supervisor/update_carriers.py reading the SSOT in
ouroboros/tools/release_sync.py. The one liberty this launcher takes is
loading release_sync straight from its file and pre-registering it under its
canonical module name, so a standalone operator invocation never executes the
`ouroboros.tools` package __init__ (which drags in the full tool registry and
its runtime configuration).

Exit codes: 0 — every conflicted carrier file was resolved (non-carrier
conflicts may remain; they are the operator's ordinary rebase work);
1 — at least one carrier file degraded to manual resolution; 2 — git or
usage failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_engine():
    """Import the shared resolver without executing ouroboros.tools.__init__."""
    spec = importlib.util.spec_from_file_location(
        "ouroboros.tools.release_sync",
        REPO_ROOT / "ouroboros" / "tools" / "release_sync.py",
    )
    assert spec is not None and spec.loader is not None
    release_sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release_sync)
    sys.modules.setdefault("ouroboros.tools.release_sync", release_sync)
    from supervisor.update_carriers import resolve_carrier_conflicts

    return resolve_carrier_conflicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--worktree", default=".",
        help="the mid-rebase checkout to operate on (default: current directory)",
    )
    parser.add_argument(
        "--prefer", choices=("ours", "theirs"), default="ours",
        help="which side wins INSIDE the carrier spans (default: ours — during "
             "a rebase, the side being rebased onto)",
    )
    args = parser.parse_args(argv)
    worktree = str(pathlib.Path(args.worktree).resolve())

    listing = subprocess.run(
        ["git", "-C", worktree, "diff", "--name-only", "--diff-filter=U"],
        capture_output=True, text=True,
    )
    if listing.returncode != 0:
        print(f"error: could not list unmerged paths: {listing.stderr.strip()}",
              file=sys.stderr)
        return 2
    conflicted = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if not conflicted:
        print("nothing to do: no unmerged paths")
        return 0

    resolve_carrier_conflicts = _load_engine()
    outcome = resolve_carrier_conflicts(worktree, conflicted, prefer=args.prefer)
    resolved = list(outcome["resolved"])
    kept = dict(outcome["kept"])
    non_carrier = sorted(p for p, reason in kept.items() if reason == "not_a_carrier")
    degraded = {p: reason for p, reason in kept.items() if reason != "not_a_carrier"}

    if resolved:
        print(f"resolved by span substitution ({args.prefer} inside the spans, "
              f"3-way for the rest): {', '.join(sorted(resolved))}")
    if degraded:
        for path, reason in sorted(degraded.items()):
            print(f"left for manual resolution: {path} ({reason})")
    if non_carrier:
        print(f"not carrier files — ordinary rebase work: {', '.join(non_carrier)}")
    if not resolved and not degraded:
        print("no carrier files among the conflicts")
    return 1 if degraded else 0


if __name__ == "__main__":
    sys.exit(main())
