#!/usr/bin/env python3
"""Regenerate the deterministic module/function/byte size debt manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ouroboros.review import (  # noqa: E402
    SIZE_RATCHET_MANIFEST_PATH,
    SizeRatchetManifest,
    candidate_repo_paths,
    collect_size_ratchet_inventory,
    collect_size_ratchet_inventory_at_ref,
    parse_size_ratchet_manifest,
    resolve_committed_manifest_text,
    validate_manifest_transition,
    validate_size_ratchet,
    validate_size_ratchet_candidate,
)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _tracked_paths() -> tuple[str, ...]:
    return candidate_repo_paths(REPO_ROOT)


def _parse_rationales(items: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--band-rationale must use PATH=TEXT")
        raw_path, rationale = item.split("=", 1)
        path = pathlib.PurePosixPath(raw_path.replace("\\", "/"))
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"band rationale path is not canonical repo-relative: {raw_path!r}")
        if not rationale.strip():
            raise ValueError(f"band rationale must be nonblank: {path.as_posix()}")
        rel = path.as_posix()
        if rel in parsed:
            raise ValueError(f"duplicate --band-rationale path: {rel}")
        parsed[rel] = rationale.strip()
    return parsed


def _tuple_lines(name: str, values: Iterable[str]) -> list[str]:
    lines = [f"{name} = ("]
    lines.extend(f"    {json.dumps(value)}," for value in values)
    lines.append(")")
    return lines


def _render(manifest: SizeRatchetManifest) -> str:
    lines = [
        '"""Generated data-only size debt manifest. Regenerate with scripts/regenerate_size_ratchet.py."""',
        "",
        f"BASELINE_SOURCE_SHA = {json.dumps(manifest.baseline_source_sha)}",
        "",
    ]
    lines.extend(_tuple_lines("GIANT_PATHS", sorted(manifest.giant_paths)))
    lines.extend(("", "FUNCTION_DEBT = ("))
    lines.extend(
        f"    ({json.dumps(path)}, {json.dumps(qualname)})," for path, qualname in sorted(manifest.function_debt)
    )
    lines.extend((")", ""))
    lines.extend(_tuple_lines("BAND_BASELINE_PATHS", sorted(manifest.band_baseline_paths)))
    lines.extend(("", "BAND_PATHS = {"))
    lines.extend(
        f"    {json.dumps(path)}: "
        f"{json.dumps(manifest.band_paths[path]) if manifest.band_paths[path] is not None else 'None'},"
        for path in sorted(manifest.band_paths)
    )
    lines.extend(("}", "", "BYTE_BASELINE_DEBT = {"))
    lines.extend(
        f"    {json.dumps(path)}: {manifest.byte_baseline_debt[path]}," for path in sorted(manifest.byte_baseline_debt)
    )
    lines.extend(("}", "", "BYTE_DEBT = {"))
    lines.extend(f"    {json.dumps(path)}: {manifest.byte_debt[path]}," for path in sorted(manifest.byte_debt))
    lines.extend(("}", ""))
    return "\n".join(lines)


def _next_manifest(
    rationales: dict[str, str],
    *,
    written: SizeRatchetManifest | None = None,
) -> SizeRatchetManifest:
    """The manifest the live tree needs. A band entry's rationale is the one named on the command
    line, else the one ``written`` on disk (an earlier run or a hand edit), else the committed one."""
    if _git("ls-files", "-u").stdout.strip():
        raise ValueError("merge in progress: resolve conflicts first")
    head = _git("rev-parse", "HEAD").stdout.strip()
    prior_text = resolve_committed_manifest_text(REPO_ROOT)
    previous = parse_size_ratchet_manifest(prior_text) if prior_text is not None else None

    unused = set(rationales)
    if previous is None:
        inventory = collect_size_ratchet_inventory_at_ref(REPO_ROOT, head)
        band_paths = {path: rationales.get(path) for path in sorted(inventory.band_paths)}
        unused -= set(inventory.band_paths)
        current = SizeRatchetManifest(
            baseline_source_sha=head,
            giant_paths=inventory.giant_paths,
            function_debt=inventory.function_debt,
            band_baseline_paths=inventory.band_paths,
            band_paths=band_paths,
            byte_baseline_debt=dict(inventory.byte_debt),
            byte_debt=dict(inventory.byte_debt),
        )
    else:
        inventory = collect_size_ratchet_inventory(REPO_ROOT, repo_paths=_tracked_paths())
        on_disk = written.band_paths if written is not None else {}
        # Every 1001-1500 module has an entry; one at 1501-1600 lines keeps its entry or takes one named here.
        entries = inventory.band_paths | (
            inventory.above_band_paths & (set(previous.band_paths) | set(on_disk) | set(rationales)))
        band_paths = {path: rationales.get(path) or on_disk.get(path) or previous.band_paths.get(path)
                      for path in sorted(entries)}
        unused -= entries
        current = SizeRatchetManifest(
            baseline_source_sha=previous.baseline_source_sha,
            giant_paths=inventory.giant_paths,
            function_debt=inventory.function_debt,
            band_baseline_paths=previous.band_baseline_paths,
            band_paths=band_paths,
            byte_baseline_debt=previous.byte_baseline_debt,
            byte_debt=dict(inventory.byte_debt),
        )
        transition_errors = validate_manifest_transition(current, previous, inventory.line_counts())
        if transition_errors:
            raise ValueError("\n".join(transition_errors))

    if unused:
        raise ValueError(f"unused --band-rationale paths: {', '.join(sorted(unused))}")
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail unless the checked-in manifest is exact")
    parser.add_argument(
        "--band-rationale",
        action="append",
        default=[],
        metavar="PATH=TEXT",
        help="set the reason of one 1001-1600-line module: a new or re-entered entry, a touched module "
             "without one, or a replacement of the recorded reason (repeatable)",
    )
    args = parser.parse_args(argv)
    try:
        rationales = _parse_rationales(args.band_rationale)
        path = REPO_ROOT / SIZE_RATCHET_MANIFEST_PATH
        written = None
        if path.exists():
            try:
                written = parse_size_ratchet_manifest(path.read_text(encoding="utf-8"))
            except ValueError:
                if args.check:
                    raise  # regeneration replaces an unreadable manifest; a check reports it
        rendered = _render(_next_manifest(rationales, written=written))
        # Candidate mode: prove the rendered manifest against the live tree
        # BEFORE overwriting the checked-in one, so a red regeneration never
        # clobbers the on-disk manifest with an invalid candidate.
        candidate_errors = [] if args.check else validate_size_ratchet_candidate(REPO_ROOT, rendered)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"size-ratchet regeneration failed: {exc}", file=sys.stderr)
        return 2
    if candidate_errors:
        print(
            "size-ratchet candidate validation failed (manifest not written):\n" + "\n".join(candidate_errors)
            + "\n(a module's reason is set with --band-rationale PATH=TEXT)",
            file=sys.stderr,
        )
        return 2

    if args.check:
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if actual != rendered:
            print(f"{SIZE_RATCHET_MANIFEST_PATH} is stale; run {pathlib.Path(__file__).name}", file=sys.stderr)
            return 1
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    try:
        validation_errors = validate_size_ratchet(REPO_ROOT)
    except (OSError, SyntaxError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"size-ratchet validation failed: {exc}", file=sys.stderr)
        return 2
    if validation_errors:
        print("size-ratchet validation failed:\n" + "\n".join(validation_errors), file=sys.stderr)
        if not args.check and all(e.startswith("staged: ") for e in validation_errors):
            # Mid-merge resolver case: the regenerated manifest WAS written and
            # is correct — only the merge-staged copy is stale. One command fixes it.
            print(
                f"note: {SIZE_RATCHET_MANIFEST_PATH} WAS regenerated on disk; run "
                f"`git add {SIZE_RATCHET_MANIFEST_PATH}` and re-run to finish",
                file=sys.stderr,
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
