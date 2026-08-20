"""Action translation for the OSWorld step loop.

Verbatim extraction from ``run_step_agent.py`` (v7 stream W): the official
``WAIT``/``DONE``/``FAIL`` specials, the tolerant JSON reader for a model turn,
and the pyautogui snippet emitters the structured actions normalize into.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

SPECIAL_ACTIONS = {"WAIT", "DONE", "FAIL"}


def _json_from_text(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _shell_action(command: str, cwd: str = "", timeout: int = 300) -> str:
    """Render a structured shell action as an OSWorld pyautogui/Python snippet.

    OSWorld records the resulting Python snippet as the official action and runs
    the command through a non-interactive bash. We deliberately do NOT fabricate
    ``~/.bash_history`` entries: writing the command into the history file to
    satisfy a terminal-task evaluator is hidden-verifier-knowledge / answer
    fitting (forbidden by the audit's methodology rules — the command's real
    execution path simply does not produce interactive history).
    """

    command = str(command or "").strip()
    cwd = str(cwd or "").strip()
    try:
        timeout = max(1, int(timeout))
    except Exception:
        timeout = 300
    encoded = base64.b64encode(command.encode("utf-8", errors="replace")).decode("ascii")
    return (
        "import base64, pathlib, subprocess, tempfile\n"
        f"cmd = base64.b64decode({encoded!r}).decode('utf-8', errors='replace')\n"
        f"cwd = {cwd!r} or None\n"
        f"timeout = {timeout!r}\n"
        "with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as script:\n"
        "    script.write('set -e\\n' + cmd + '\\n')\n"
        "    script_path = script.name\n"
        "try:\n"
        "    result = subprocess.run(['/bin/bash', script_path], cwd=cwd, text=True, capture_output=True, timeout=timeout)\n"
        "finally:\n"
        "    pathlib.Path(script_path).unlink(missing_ok=True)\n"
        "print(result.stdout)\n"
        "print(result.stderr)\n"
        "result.check_returncode()\n"
    )


def _click_action(x: Any, y: Any) -> str:
    return (
        "import pyautogui, time\n"
        f"pyautogui.click({int(float(x))}, {int(float(y))})\n"
        "time.sleep(0.5)\n"
    )


def _type_action(text: str, interval: float = 0.01) -> str:
    return (
        "import pyautogui, time\n"
        f"pyautogui.typewrite({str(text or '')!r}, interval={float(interval)!r})\n"
        "time.sleep(0.2)\n"
    )


def _hotkey_action(keys: Any) -> str:
    if isinstance(keys, str):
        key_list = [part.strip() for part in keys.split("+") if part.strip()]
    elif isinstance(keys, list):
        key_list = [str(part).strip() for part in keys if str(part).strip()]
    else:
        key_list = []
    return (
        "import pyautogui, time\n"
        f"pyautogui.hotkey(*{key_list!r})\n"
        "time.sleep(0.3)\n"
    )


def _wait_action(seconds: Any = 1.0) -> str:
    try:
        seconds = max(0.0, float(seconds))
    except Exception:
        seconds = 1.0
    return f"import time\ntime.sleep({seconds!r})\n"


def _normalize_structured_action(item: Any) -> str:
    """Convert a model action object to one OSWorld action string."""

    if isinstance(item, str):
        text = item.strip()
        return text.upper() if text.upper() in SPECIAL_ACTIONS else text
    if not isinstance(item, dict):
        return ""
    kind = str(item.get("type") or item.get("action") or "").strip().lower()
    if kind in {"done", "finish"}:
        return "DONE"
    if kind in {"fail", "infeasible"}:
        return "FAIL"
    if kind == "wait":
        if "seconds" in item:
            return _wait_action(item.get("seconds"))
        return "WAIT"
    if kind == "shell":
        return _shell_action(
            str(item.get("command") or item.get("cmd") or ""),
            cwd=str(item.get("cwd") or ""),
            timeout=int(item.get("timeout_sec") or item.get("timeout") or 300),
        )
    if kind == "click":
        return _click_action(item.get("x", 0), item.get("y", 0))
    if kind == "type":
        return _type_action(str(item.get("text") or ""), interval=float(item.get("interval") or 0.01))
    if kind == "hotkey":
        return _hotkey_action(item.get("keys") or item.get("key") or "")
    if kind in {"press", "key"}:
        return _hotkey_action([item.get("key") or item.get("keys") or ""])
    if kind == "python":
        return str(item.get("code") or "").strip()
    return ""
