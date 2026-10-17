"""Measure what a live call sends Gemini on any data directory, with no provider and no MCP server.

The agent is built on the data directory (an APFS clone of a real one, never the live install),
the call's context is prepared exactly as for a call (``OuroborosAgent._prepare_task_context``:
the realtime route plan, then the ordinary preparation with the call view), and the instruction
the driver's session config carries (the system message text, one ``systemInstruction``) is
written out and measured: each identified part with the o200k tokenizer when ``tiktoken`` is
importable (here or through ``--tokenizer-python``), else the host's estimate; the seconds from
the start to the instruction being ready; the host sections the view addresses; and counters
(MCP listings asked for, chronicle index sessions, health and memory-view builds). No provider
key is read and no request leaves the machine: MCP refreshes and the Claudexor gateway are
replaced by refusals that count themselves.

usage: <app python> -m devtools.call_context_measure --data <data dir> --out <dir outside any git tree>
       [--repo <checkout>] [--model gemini-3.8-live] [--voice Kore] [--tokenizer-python <python with tiktoken>]
       [--tiktoken-cache <dir holding o200k_base>]
Run it under scripts/safe_test.py; the data directory is modified (retained captures, events), so
point it at a disposable clone.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

COUNTERS: Counter = Counter()


def _count(name: str, original: Optional[Callable[..., Any]] = None, *, refuse: bool = False) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        COUNTERS[name] += 1
        if refuse:
            return {"status": "skipped_by_measurement"}
        return original(*args, **kwargs)
    return wrapper


def _tokenizer(python: str, cache: str = "") -> Tuple[str, Callable[[List[str]], List[int]]]:
    """``(name, count)``: o200k here, o200k through another interpreter, or the host's estimate."""
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("o200k_base")
        return "o200k_base", lambda texts: [len(encoding.encode(text, disallowed_special=())) for text in texts]
    except ImportError:
        pass
    if python:
        def count(texts: List[str]) -> List[int]:
            script = ("import json,sys,tiktoken; e=tiktoken.get_encoding('o200k_base'); "
                      "print(json.dumps([len(e.encode(t, disallowed_special=())) for t in json.load(sys.stdin)]))")
            env = {**os.environ, **({"TIKTOKEN_CACHE_DIR": cache} if cache else {})}  # an existing BPE cache: no fetch
            done = subprocess.run([python, "-I", "-c", script], input=json.dumps(texts), text=True,
                                  capture_output=True, timeout=600, check=True, env=env)
            return json.loads(done.stdout)
        return "o200k_base", count
    from ouroboros.utils import estimate_tokens

    return "host_estimate", lambda texts: [estimate_tokens(text) for text in texts]


def _parts(plan: Any, mode: str) -> List[Tuple[str, str]]:
    """``(section id, shown text)`` of the identified parts of one projection, in render order."""
    from ouroboros import context_fit
    from ouroboros.section_floor import HostSection, relocated_text

    core = plan.core
    rows = {row["section_id"]: row for row in plan.projection(mode).memory_facts.get("host_sections_addressed", [])}
    form = "low" if core.compact_reference_docs or mode == "nano" else mode
    governance, development = context_fit._governance_blocks(None, core, mode=form)

    def shown(parts: Any, prefix: str) -> List[Tuple[str, str]]:
        out = []
        for i, part in enumerate(context_fit._parts_of(parts)):
            ident = part.section_id if isinstance(part, HostSection) else f"{prefix}_{i}"
            out.append((ident, relocated_text(part, rows[ident]["reader"]) if ident in rows else str(part)))
        return out

    return [*shown(governance, "governance"), *([("development_handbook", development)] if development else []),
            *shown(core.semi_stable_sections or core.semi_stable_text, "semi"),
            *shown(core.dynamic_head_sections or core.dynamic_head_text, "head"),
            *shown(core.dynamic_sections or core.dynamic_text, "dynamic")]


def _section_table(instruction: str, parts: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Locate each identified part in the instruction in order; what lies between is the memory view."""
    table, cursor = [], 0
    for ident, text in parts:
        if not text:
            continue
        at = instruction.find(text, cursor)
        if at < 0:
            table.append({"section": ident, "chars": len(text), "located": False})
            continue
        gap = instruction[cursor:at].strip("\n")
        if gap:
            table.append({"section": "memory_view (story or room)", "chars": len(gap), "text": gap, "located": True})
        table.append({"section": ident, "chars": len(text), "text": text, "located": True})
        cursor = at + len(text)
    rest = instruction[cursor:].strip("\n")
    if rest:
        table.append({"section": "memory_view (story or room)", "chars": len(rest), "text": rest, "located": True})
    return table


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--model", default="gemini-3.8-live")
    parser.add_argument("--voice", default="Kore")
    parser.add_argument("--tokenizer-python", default="")
    parser.add_argument("--tiktoken-cache", default="", help="TIKTOKEN_CACHE_DIR holding o200k_base already")
    args = parser.parse_args(argv)
    data, repo, out = pathlib.Path(args.data).resolve(), pathlib.Path(args.repo).resolve(), pathlib.Path(args.out)
    if (data / ".git").exists() or (data.parent / "ouroboros.pid").exists() or (data.parent / "repo" / ".git").exists():
        raise SystemExit("refusing a live install's or a versioned data directory; measure a disposable clone")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["OUROBOROS_DATA_DIR"] = str(data)
    os.environ["OUROBOROS_SETTINGS_PATH"] = str(data / "settings.json")
    sys.path.insert(0, str(repo))
    started = time.monotonic()

    from unittest.mock import patch

    from ouroboros import chronicle_store, context_health, mcp_client, memory_view

    patches = [
        patch.object(mcp_client.MCPManager, "refresh_server", _count("mcp_refresh_server", refuse=True)),
        patch.object(mcp_client.MCPManager, "refresh_all", _count("mcp_refresh_all", refuse=True)),
        patch.object(mcp_client.MCPManager, "refresh_all_background", _count("mcp_refresh_background", refuse=True)),
        patch("ouroboros.claudexor_daemon.ensure_owned_gateway", side_effect=RuntimeError("measurement: no daemon")),
        patch.object(chronicle_store.ChronicleStore, "_index",
                     _count("chronicle_index_sessions", chronicle_store.ChronicleStore._index)),
        patch("ouroboros.context.build_health_invariants", _count("health_builds", context_health.build_health_invariants)),
        patch.object(memory_view, "capture_memory_view", _count("memory_view_captures", memory_view.capture_memory_view)),
    ]
    for item in patches:
        item.start()
    from ouroboros.agent import make_agent
    from ouroboros.call_protocol import CallSpec
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.live_call_route import initial_call_schemas
    from ouroboros.live_tool_batch import text_of
    from supervisor import message_bus, state, workers

    state.init(data, total_budget_limit=10)
    state.init_state()
    message_bus.init(data, 10, 0, message_bus.LocalChatBridge())
    workers.init(repo, data, 0)
    agent = make_agent(str(repo), str(data))
    agent_ready = time.monotonic() - started
    spec = CallSpec(uuid.uuid4().hex, WEB_UI_CHAT_ID, "google", args.model, args.voice, opened_by="owner",
                    presence_binding_id="", max_minutes=0, label="Measured call", client_message_id="measure")
    task = {"id": "measure-" + uuid.uuid4().hex[:12], "type": "task", "chat_id": WEB_UI_CHAT_ID,
            "_is_direct_chat": True, "text": "[Live call] A live voice call is open in this room. "
            "The conversation is spoken; its transcript is written to this chat.",
            "metadata": {"live_call": spec.as_metadata()}}
    ctx, messages, _cap = agent._prepare_task_context(task)
    ready = time.monotonic() - started
    plan = ctx.context_fit_plan
    mode = plan.initial_mode
    instruction = text_of(messages[0]["content"])  # what the driver's session config carries
    tools = initial_call_schemas(agent.tools, spec)  # the call's first tool view, as its plan built it
    (out / "instruction.txt").write_text(instruction, encoding="utf-8")
    table = _section_table(instruction, _parts(plan, mode))
    name, count = _tokenizer(args.tokenizer_python, args.tiktoken_cache)
    texts = [instruction, json.dumps(messages[1:], ensure_ascii=False), *[row.get("text", "") for row in table]]
    totals = count(texts)
    for row, tokens in zip(table, totals[2:]):
        row[name] = tokens
        row.pop("text", None)
    from ouroboros.context_fit import tool_schema_tokens

    summary = {
        "engine": "google", "model": args.model, "mode": mode, "window_tokens": plan.window_tokens,
        "reply_allowance_tokens": plan.output_reserve_tokens, "calibration_ratio": plan.projection(mode).calibration_ratio,
        "route_fp": plan.route_fp, "tokenizer": name, "instruction_chars": len(instruction),
        "instruction_tokens": totals[0], "turn_tokens": totals[1],
        "estimated_request_tokens": plan.projection(mode).estimated_tokens,
        "tool_schema_estimate_tokens": tool_schema_tokens(tools), "tools": len(tools),
        "seconds_agent_ready": round(agent_ready, 2), "seconds_instruction_ready": round(ready, 2),
        "host_sections_addressed": [{key: row[key] for key in ("section_id", "tokens_saved")}
                                    for row in plan.projection(mode).memory_facts.get("host_sections_addressed", [])],
        "memory_floor_steps": plan.projection(mode).memory_facts.get("floor", {}).get("steps", {}),
        "counters": dict(COUNTERS), "sections": table,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "sections"}, ensure_ascii=False, indent=1))
    for row in table:
        print(f"{row.get(name, '?'):>8} {row['chars']:>8}  {row['section']}{'' if row['located'] else '  (not located)'}")
    sys.stdout.flush()
    os._exit(0)  # the agent's background threads are not this measurement's to wait for


if __name__ == "__main__":
    sys.exit(main())
