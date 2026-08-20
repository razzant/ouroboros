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
    validate_manifest_transition,
    validate_size_ratchet,
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
    if manifest.module_debt_1500 is not None:
        lines.append("")
        lines.extend(_tuple_lines("MODULE_DEBT_1500", sorted(manifest.module_debt_1500)))
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
    activate_1500_layer: bool = False,
    checked_candidate: SizeRatchetManifest | None = None,
) -> SizeRatchetManifest:
    head = _git("rev-parse", "HEAD").stdout.strip()
    prior_result = _git("show", f"HEAD:{SIZE_RATCHET_MANIFEST_PATH}", check=False)
    previous = parse_size_ratchet_manifest(prior_result.stdout) if prior_result.returncode == 0 else None

    # --check inherits only the activation *state* from the checked-in candidate;
    # the active set contents always derive from the production inventory below.
    activate = activate_1500_layer or (
        checked_candidate is not None and checked_candidate.module_debt_1500 is not None
    )
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
            module_debt_1500=inventory.module_debt_1500 if activate else None,
        )
    else:
        if activate_1500_layer and previous.module_debt_1500 is not None:
            raise ValueError("MODULE_DEBT_1500 is already active; --activate-1500-layer is one-time")
        inventory = collect_size_ratchet_inventory(REPO_ROOT, repo_paths=_tracked_paths())
        if previous.module_debt_1500 is not None or activate:
            module_debt_1500 = inventory.module_debt_1500
        else:
            module_debt_1500 = None
        band_paths: dict[str, str | None] = {}
        for path in sorted(inventory.band_paths):
            if path in previous.band_paths:
                band_paths[path] = previous.band_paths[path]
            else:
                checked_rationale = checked_candidate.band_paths.get(path) if checked_candidate is not None else None
                band_paths[path] = rationales.get(path, checked_rationale)
                unused.discard(path)
        current = SizeRatchetManifest(
            baseline_source_sha=previous.baseline_source_sha,
            giant_paths=inventory.giant_paths,
            function_debt=inventory.function_debt,
            band_baseline_paths=previous.band_baseline_paths,
            band_paths=band_paths,
            byte_baseline_debt=previous.byte_baseline_debt,
            byte_debt=dict(inventory.byte_debt),
            module_debt_1500=module_debt_1500,
        )
        parent_inventory_1500 = None
        if previous.module_debt_1500 is None and module_debt_1500 is not None:
            parent_inventory_1500 = collect_size_ratchet_inventory_at_ref(REPO_ROOT, head).module_debt_1500
        transition_errors = validate_manifest_transition(
            current, previous, parent_inventory_1500=parent_inventory_1500
        )
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
        help="authorize one new or re-entered 1001-1500-line path",
    )
    parser.add_argument(
        "--activate-1500-layer",
        action="store_true",
        help="one-time activation of the v7 MODULE_DEBT_1500 layer from the exact first-parent >1500 inventory",
    )
    args = parser.parse_args(argv)
    try:
        rationales = _parse_rationales(args.band_rationale)
        path = REPO_ROOT / SIZE_RATCHET_MANIFEST_PATH
        checked_candidate = (
            parse_size_ratchet_manifest(path.read_text(encoding="utf-8")) if args.check and path.exists() else None
        )
        rendered = _render(
            _next_manifest(
                rationales,
                activate_1500_layer=args.activate_1500_layer,
                checked_candidate=checked_candidate,
            )
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"size-ratchet regeneration failed: {exc}", file=sys.stderr)
        return 2

    if args.check:
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if actual != rendered:
            print(f"{SIZE_RATCHET_MANIFEST_PATH} is stale; run {pathlib.Path(__file__).name}", file=sys.stderr)
            return 1
    else:
        path.write_text(rendered, encoding="utf-8")
    try:
        validation_errors = validate_size_ratchet(REPO_ROOT)
    except (OSError, SyntaxError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"size-ratchet validation failed: {exc}", file=sys.stderr)
        return 2
    if validation_errors:
        print("size-ratchet validation failed:\n" + "\n".join(validation_errors), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
