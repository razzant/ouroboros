"""Build ATIF (Agent Trajectory Interchange Format) trajectories for TB trials.

Harbor Hub leaderboard static validation requires every passing trial to carry
``agent/trajectory.json`` (ATIF, schema ``ATIF-v1.7``). This module reconstructs
an honest trajectory from the artifacts the installed adapter already writes:

- ``agent/instruction.txt``                        -> the user step
- ``agent/ouroboros-data/logs/tools.jsonl``        -> tool calls + observations
- ``agent/ouroboros-data/logs/progress.jsonl``     -> agent narration between calls
- ``agent/ouroboros-data/logs/chat.jsonl``         -> final answer
- ``agent/ouroboros-task-result.json``             -> final answer fallback
- ``agent/ouroboros-run-summary.json``             -> total cost
- ``agent/ouroboros-data/logs/events.jsonl``       -> llm_usage token totals, version

IMPORTANT: stdlib-only. The same builder runs inside task containers (where
harbor is not installed) and host-side in the offline converter. Nothing here
invents content: every step is derived verbatim from the recorded logs.

ATIF constraints honored (see harbor ``models/trajectories``):
- ``step_id`` strictly sequential starting at 1;
- ``observation.results[].source_call_id`` only references a ``tool_call`` of
  the SAME step;
- agent-only fields never appear on user steps;
- no extra keys (harbor models are ``extra="forbid"``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ATIF_SCHEMA_VERSION = "ATIF-v1.7"
_PREVIEW_LIMIT = 20_000


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _clip(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} chars omitted]"


def _json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return repr(value)


def _detect_version(events: list[dict[str, Any]]) -> str | None:
    for row in events:
        if row.get("type") == "startup_verification":
            sync = (row.get("checks") or {}).get("version_sync") or {}
            version = sync.get("version_file")
            if isinstance(version, str) and version:
                return version
    return None


def _final_answer(agent_dir: Path) -> str:
    chat = _read_jsonl(agent_dir / "ouroboros-data" / "logs" / "chat.jsonl")
    for row in reversed(chat):
        if row.get("direction") == "out" and isinstance(row.get("text"), str):
            return row["text"]
    try:
        record = json.loads(_read_text(agent_dir / "ouroboros-task-result.json") or "{}")
    except ValueError:
        record = {}
    for key in ("final_answer", "result"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def build_trajectory(
    agent_dir: Path,
    *,
    agent_name: str = "Ouroboros",
    agent_version: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build an ATIF trajectory dict from a Harbor trial ``agent/`` directory."""
    agent_dir = Path(agent_dir)
    logs_dir = agent_dir / "ouroboros-data" / "logs"

    events = _read_jsonl(logs_dir / "events.jsonl")
    tool_rows = [
        r for r in _read_jsonl(logs_dir / "tools.jsonl") if r.get("type") == "tool_call"
    ]
    narration_rows = [
        r
        for r in _read_jsonl(logs_dir / "progress.jsonl")
        if r.get("type") == "send_message" and isinstance(r.get("text"), str)
    ]

    steps: list[dict[str, Any]] = []

    instruction = _read_text(agent_dir / "instruction.txt").strip()
    steps.append(
        {
            "step_id": 1,
            "source": "user",
            "message": instruction or "(instruction.txt missing)",
        }
    )

    # Merge narration and tool calls chronologically; narration since the last
    # tool call becomes the message of the step wrapping the next tool call.
    merged: list[tuple[str, str, dict[str, Any]]] = []
    for row in narration_rows:
        merged.append((str(row.get("ts") or ""), "narration", row))
    for row in tool_rows:
        merged.append((str(row.get("ts") or ""), "tool_call", row))
    merged.sort(key=lambda item: item[0])

    call_counter = 0
    pending_narration: list[str] = []
    for ts, kind, row in merged:
        if kind == "narration":
            pending_narration.append(row["text"])
            continue
        call_counter += 1
        call_id = f"call_{call_counter}"
        preview = row.get("result_preview")
        content = preview if isinstance(preview, str) else json.dumps(_json_safe(preview))
        if row.get("is_error") or (row.get("status") not in (None, "ok")):
            content = f"[status={row.get('status')} is_error={bool(row.get('is_error'))}]\n{content}"
        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "source": "agent",
            "message": _clip("\n\n".join(pending_narration)),
            "tool_calls": [
                {
                    "tool_call_id": call_id,
                    "function_name": str(row.get("tool") or "unknown_tool"),
                    "arguments": _json_safe(row.get("args") or {}),
                }
            ],
            "observation": {
                "results": [
                    {"source_call_id": call_id, "content": _clip(content)}
                ]
            },
        }
        if ts:
            step["timestamp"] = ts
        steps.append(step)
        pending_narration = []

    final_parts = list(pending_narration)
    answer = _final_answer(agent_dir)
    if answer:
        final_parts.append(answer)
    steps.append(
        {
            "step_id": len(steps) + 1,
            "source": "agent",
            "message": _clip("\n\n".join(final_parts)) or "(no final message recorded)",
        }
    )

    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    saw_usage = False
    for row in events:
        if row.get("type") != "llm_usage":
            continue
        saw_usage = True
        prompt_tokens += int(row.get("prompt_tokens") or 0)
        completion_tokens += int(row.get("completion_tokens") or 0)
        cached_tokens += int(row.get("cached_tokens") or 0)

    total_cost: float | None = None
    try:
        summary = json.loads(_read_text(agent_dir / "ouroboros-run-summary.json") or "{}")
        cost = summary.get("cost_usd")
        if isinstance(cost, (int, float)):
            total_cost = float(cost)
    except ValueError:
        pass

    final_metrics: dict[str, Any] = {"total_steps": len(steps)}
    if saw_usage:
        final_metrics["total_prompt_tokens"] = prompt_tokens
        final_metrics["total_completion_tokens"] = completion_tokens
        final_metrics["total_cached_tokens"] = cached_tokens
    if total_cost is not None:
        final_metrics["total_cost_usd"] = total_cost

    agent_block: dict[str, Any] = {
        "name": agent_name,
        "version": agent_version or _detect_version(events) or "unknown",
    }
    if model_name:
        agent_block["model_name"] = model_name

    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "agent": agent_block,
        "steps": steps,
        "final_metrics": final_metrics,
    }


def write_trajectory(agent_dir: Path, trajectory: dict[str, Any]) -> Path:
    out = Path(agent_dir) / "trajectory.json"
    out.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


def main() -> int:
    """In-container / standalone CLI: build one trial's trajectory.json."""
    import argparse

    parser = argparse.ArgumentParser(description="Build ATIF trajectory for one trial")
    parser.add_argument("agent_dir", type=Path, help="trial agent/ directory")
    parser.add_argument("--model", default=None)
    parser.add_argument("--agent-name", default="Ouroboros")
    parser.add_argument("--agent-version", default=None)
    args = parser.parse_args()

    trajectory = build_trajectory(
        args.agent_dir,
        agent_name=args.agent_name,
        agent_version=args.agent_version,
        model_name=args.model,
    )
    out = write_trajectory(args.agent_dir, trajectory)
    print(f"wrote {out} ({len(trajectory['steps'])} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
