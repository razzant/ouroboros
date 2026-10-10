#!/usr/bin/env python3
"""Measure wait-only cognition by (task_id, round_id), not adjacent rows.

Reads only explicit JSONL inputs; never discovers owner logs. Usage rows are
llm_usage/round_usage, tool rows name a tool, and child task_done rows must carry
parent_task_id or delegation_role=subagent. Missing/conflicting evidence stays
unknown. A micro-scenario cannot certify a three-day workload percentage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WAIT_TOOLS = frozenset({"wait_task", "wait_tasks", "await_messages", "delegate_wait", "peek_task"})


def measure(rows: list[dict]) -> dict:
    tools, usage, terminals, conflicts, gaps = {}, {}, set(), set(), []
    for i, row in enumerate(rows):
        task, round_id = row.get("task_id"), row.get("round_id")
        if row.get("type") == "task_done" and task:
            if row.get("parent_task_id") or row.get("delegation_role") == "subagent":
                terminals.add(str(task))
            else:
                gaps.append({"row": i, "reason": "terminal_child_lineage_missing"})
        name = row.get("fn_name") or row.get("name") or row.get("tool")
        if not name and row.get("type") not in {"llm_usage", "round_usage"}:
            continue
        if task is None or round_id is None:
            gaps.append({"row": i, "reason": "join_identity_missing"})
            continue
        key = (str(task), str(round_id))
        if name:
            tools.setdefault(key, set()).add(str(name))
        else:
            counts = (row.get("prompt_tokens"), row.get("cached_tokens"))
            if key in usage and usage[key] != counts:
                conflicts.add(key)
                gaps.append({"row": i, "reason": "conflicting_round_usage"})
            usage.setdefault(key, counts)
    for key in sorted(tools.keys() - usage.keys()):
        gaps.append({"key": list(key), "reason": "tool_round_usage_missing"})
    only = {k for k, names in tools.items() if names and names <= WAIT_TOOLS and k in usage}

    def total(keys, column):
        values = [usage[k][column] for k in keys]
        if any(k in conflicts for k in keys) or any(type(v) is not int or v < 0 for v in values):
            return None
        return sum(values)

    def ratio(a, b):
        return a / b if a is not None and b else None

    prompt, wait_prompt, cached = total(usage, 0), total(only, 0), total(only, 1)
    fractions = (ratio(len(only), len(usage)), ratio(wait_prompt, prompt), ratio(len(only), len(terminals)))
    complete = not gaps and not conflicts
    return {"schema": 1, "usage_rounds": len(usage), "wait_only_rounds": len(only),
            "prompt_tokens": prompt, "wait_prompt_tokens": wait_prompt,
            "wait_cached_tokens": cached, "child_terminal_events": len(terminals),
            "wait_round_fraction": fractions[0], "wait_prompt_fraction": fractions[1],
            "wait_rounds_per_terminal": fractions[2],
            "targets": {name: value <= limit if complete and value is not None else None
                        for name, value, limit in zip(
                            ("round_fraction_le_3pct", "prompt_fraction_le_5pct", "rounds_per_terminal_le_1_5"),
                            fractions, (.03, .05, 1.5))},
            "coverage_gaps": gaps, "coverage_complete": complete,
            "denominator": "observed joined model rounds and prompt tokens, including legitimate waits",
            "wait_round_keys": [list(k) for k in sorted(only)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, sources = [], []
    for path in args.inputs:
        data = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{n}: expected a JSON object")
                rows.append(row)
    body = json.dumps({**measure(rows), "sources": sources}, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
