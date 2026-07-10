#!/usr/bin/env python3
"""Summarize official GAIA inspect logs plus diagnostic lenient normalization."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import unicodedata

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.run_roots import ensure_file_output_outside_repo, latest_run_root, repo_root_from_devtools


def lenient_normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", value).strip().lower()


def _json_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(root.rglob("*.json"))


def _rows_from_inspect_json(data: dict, path: pathlib.Path) -> list[dict]:
    samples = data.get("samples")
    if not isinstance(samples, list):
        return []
    rows = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        output = sample.get("output")
        if isinstance(output, dict):
            answer = output.get("completion") or output.get("text") or output.get("answer")
        else:
            answer = output
        score = None
        scores = sample.get("scores")
        if isinstance(scores, dict) and scores:
            first = next(iter(scores.values()))
            if isinstance(first, dict):
                score = first.get("value", first.get("score", first.get("correct")))
            else:
                score = first
        rows.append({
            "path": str(path),
            "sample_id": str(sample.get("id")) if sample.get("id") is not None else "",
            "raw_answer": answer,
            "local_normalized": lenient_normalize(str(answer or "")),
            "official_score": score,
        })
    return rows


def _rows_from_eval_logs(root: pathlib.Path) -> list[dict]:
    try:
        from inspect_ai.log import read_eval_log
    except Exception:
        return []
    rows = []
    for path in sorted(root.rglob("*.eval")):
        try:
            log = read_eval_log(path)
        except Exception:
            continue
        for sample in list(getattr(log, "samples", []) or []):
            output = getattr(sample, "output", None)
            answer = getattr(output, "completion", "") if output is not None else ""
            score = None
            scores = getattr(sample, "scores", None)
            if isinstance(scores, dict) and scores:
                first = next(iter(scores.values()))
                score = getattr(first, "value", first)
            rows.append({
                "path": str(path),
                "sample_id": str(getattr(sample, "id", "") or ""),
                "raw_answer": answer,
                "local_normalized": lenient_normalize(str(answer or "")),
                "official_score": score,
            })
    return rows


def _leakage_flagged_ids(audit_path: pathlib.Path) -> set[str]:
    """STRONG-flagged sample ids from a leakage_audit.jsonl (deterministic_flag)."""
    flagged: set[str] = set()
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return flagged
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if row.get("deterministic_flag"):
            sid = str(row.get("sample_id") or "")
            if sid:
                flagged.add(sid)
    return flagged


def _official_score_value(score: object) -> bool | None:
    if isinstance(score, bool):
        return score
    if isinstance(score, (int, float)):
        return bool(score)
    if isinstance(score, str):
        value = score.strip().lower()
        if value in {"c", "correct", "true", "1"}:
            return True
        if value in {"i", "incorrect", "false", "0"}:
            return False
    return None


def summarize(run_dir: pathlib.Path, leakage_audit: pathlib.Path | None = None) -> dict:
    rows = []
    for path in _json_files(run_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        inspect_rows = _rows_from_inspect_json(data, path)
        if inspect_rows:
            rows.extend(inspect_rows)
            continue
        answer = data.get("final_answer") or data.get("answer") or data.get("completion")
        score = data.get("score") if "score" in data else data.get("correct")
        if answer is None and score is None:
            continue
        rows.append({
            "path": str(path),
            "raw_answer": answer,
            "local_normalized": lenient_normalize(str(answer or "")),
            "official_score": score,
        })
    eval_rows = _rows_from_eval_logs(run_dir)
    if eval_rows:
        rows.extend(eval_rows)
    scored_rows = [(row, _official_score_value(row.get("official_score"))) for row in rows]
    scored_rows = [(row, value) for row, value in scored_rows if value is not None]
    correct = sum(1 for _row, value in scored_rows if value)
    summary = {
        "run_dir": str(run_dir),
        "rows": rows,
        "official_scored": len(scored_rows),
        "official_correct": correct,
        "official_accuracy": (correct / len(scored_rows)) if scored_rows else None,
        "note": "official_score is authoritative; local_normalized is diagnostic only",
    }
    # Leakage-adjusted headline (pre-registered rule, applied identically to every
    # harness): a STRONG-flagged sample counts as INCORRECT even if it scored
    # correct, because the answer was obtained by looking up the published key
    # rather than solved. The official inspect score is never mutated — this is a
    # second, disclosed number reported alongside it (raw + adjusted + flag count).
    if leakage_audit is not None:
        flagged = _leakage_flagged_ids(leakage_audit)
        adj_correct = sum(
            1 for row, value in scored_rows
            if value and str(row.get("sample_id") or "") not in flagged
        )
        flagged_scored = sum(
            1 for row, _value in scored_rows if str(row.get("sample_id") or "") in flagged
        )
        summary["leakage_audit_path"] = str(leakage_audit)
        summary["leakage_flagged_total"] = len(flagged)
        summary["leakage_flagged_among_scored"] = flagged_scored
        summary["leakage_adjusted_correct"] = adj_correct
        summary["leakage_adjusted_accuracy"] = (adj_correct / len(scored_rows)) if scored_rows else None
        summary["note"] = (
            "official_score is authoritative; local_normalized is diagnostic. "
            "leakage_adjusted_accuracy zeroes STRONG-flagged samples (pre-registered "
            "anti-lookup rule); report raw + adjusted + flag count together."
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize GAIA official scorer outputs.")
    parser.add_argument("--run-dir", default="", help="GAIA run root; defaults to latest bench_runs/gaia")
    parser.add_argument("--output", default="", help="optional summary JSON path")
    parser.add_argument(
        "--leakage-audit", default="", nargs="?", const="__auto__",
        help="also report leakage_adjusted_accuracy from a leakage_audit.jsonl "
             "(bare flag => <run-dir>/leakage_audit.jsonl)",
    )
    args = parser.parse_args(argv)
    run_dir = pathlib.Path(args.run_dir).expanduser() if args.run_dir else latest_run_root("gaia")
    if run_dir is None:
        raise SystemExit("no GAIA run directory found")
    run_dir = run_dir.resolve(strict=False)
    leakage_audit = None
    if args.leakage_audit:
        leakage_audit = (run_dir / "leakage_audit.jsonl") if args.leakage_audit == "__auto__" \
            else pathlib.Path(args.leakage_audit).expanduser()
    summary = summarize(run_dir, leakage_audit=leakage_audit)
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        ensure_file_output_outside_repo(pathlib.Path(args.output).expanduser(), repo_root_from_devtools()).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
