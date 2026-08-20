"""Shared primitives for the OSWorld cu_bridge runner leaves.

Verbatim extraction from ``run_cu_bridge_agent.py`` (v7 stream W). A leaf may
never import the launcher (cycle), so the values the gate, budget and tool-policy
leaves share with it are owned here. ``run_cu_bridge_agent.py`` re-exports every
name, so its module surface and behaviour are unchanged.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

SKILL_NAME = "unix_computer_use"


def _api(server: str, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(server.rstrip("/") + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip().startswith(("{", "[")) else {"raw": raw}


def _text_declares_infeasible(value: Any) -> bool:
    return isinstance(value, str) and any(
        line.strip() == "TASK_INFEASIBLE" for line in value.splitlines()
    )


def _terminal_answer_text(latest: dict[str, Any] | None) -> str:
    """The agent's terminal answer, with the documented fallback.

    ``final_answer`` is empty on this runner's tasks while the answer text lands in
    ``result``; an artefact whose ``final_answer`` is null for an agent that answered
    misreports what happened, which is exactly what METHODOLOGY §4 exists to prevent.
    """
    if not isinstance(latest, dict):
        return ""
    for key in ("final_answer", "result"):
        value = latest.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _final_answer_declares_infeasible(latest: dict[str, Any]) -> bool:
    """True iff the agent's FINAL ANSWER is a standalone TASK_INFEASIBLE line.

    OSWorld's infeasible evaluators check the official action history for FAIL; a
    chat marker alone is not enough, so the bridge translates this into an
    official ``env.step("FAIL")`` before evaluate(). Inspect ONLY the terminal
    answer fields of the task result (``final_answer``, ``result``) — never the
    whole result tree, or a marker quoted in intermediate reasoning/tool output
    would spuriously flip a feasible task to a FAIL (reward 0) or fake an
    infeasible pass.
    """
    if not isinstance(latest, dict):
        return False
    # The AUTHORITATIVE terminal answer only. This used to OR over both fields, so a
    # retracted mention in the result body ("I considered TASK_INFEASIBLE but solved it"
    # on its own line) could step FAIL and zero a feasible task while the published
    # final_answer said the opposite. In practice final_answer is empty on this runner and
    # the fallback picks the same text as before; the narrowing only removes the case where
    # the two fields disagree, and there the explicit answer must win.
    return _text_declares_infeasible(_terminal_answer_text(latest))
