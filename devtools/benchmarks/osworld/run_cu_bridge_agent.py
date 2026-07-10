#!/usr/bin/env python3
"""OSWorld runner: ONE Ouroboros agentic run per task, host-side computer-use bridge.

Unlike ``run_step_agent.py`` (host drives ``env.step`` and Ouroboros is a stateless
per-step action selector with ``--memory-mode empty``), this runner gives Ouroboros
the wheel:

    host: reset VM -> publish VM_IP -> submit ONE task -> wait -> evaluate()
    agent (one run, full memory): screenshot -> reason -> click/type -> screenshot -> ... -> done

The agent acts through the bundled ``unix_computer_use`` skill, whose additive
OSWorld HTTP backend routes ``screenshot``/``click``/``type``/``key``/``scroll``
to the in-VM OSWorld server (GET /screenshot, POST /execute) — the SAME guest
channel ``env.step`` uses. The backend is activated by the ``connections.json`` +
``active_connection.txt`` this runner publishes into the bench data dir's skill
state (see ``_publish_target``); there is no env-var activation path. The brain
stays on the host; only translated pyautogui mutates the guest. ``reset()`` and
``evaluate()`` are the official OSWorld ones.

Protocol note: GUI actions go straight to the guest ``/execute`` server and thus
do NOT populate the official ``DesktopEnv.action_history`` / ``traj.jsonl``; only
the translated ``FAIL`` (for a declared-infeasible task) is an official action.
See ``METHODOLOGY.md`` §7 for the full comparability disclosures.

This is the Terminal-Bench / Pointer shape (persistent agent + computer-use tool),
without installing Ouroboros inside the VM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import benchmark_run_manifest, write_json
from devtools.benchmarks.common.result_index import append_result_index, task_result_row
from devtools.benchmarks.common.run_roots import ensure_outside_repo

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent
VMWARE_FUSION_PATHS = (
    "/Applications/VMware Fusion.app/Contents/Public",
    "/Applications/VMware Fusion.app/Contents/Library",
)

SKILL_NAME = "unix_computer_use"

# The OSWorld task instruction is UNTRUSTED and the VM is driven ONLY through the
# unix_computer_use skill (ext_* tools). Rather than a fragile per-tool DENYLIST
# (which silently misses any host tool added later), the runner keeps a small
# ALLOWLIST of core tools the task legitimately needs and DENIES every other core
# tool — so any host execution/mutation/VCS/GitHub/service/self-mod/chat surface,
# present or future, is blocked by construction. The skill's ext_* tools are not
# core tools, so they are never on the computed denylist and always available.
# `enable_tools` is kept (the agent must enable the computer-use skill), which in
# principle could enable OTHER extensions — but the runner seeds and enables ONLY
# unix_computer_use into a FRESH isolated bench data dir (append-only per task per
# the runbook), so there is no other extension to reach; a reused multi-extension
# data dir is out of the supported bench setup.
# Deliberately NO host filesystem/code read tools (read_file/list_files/search_code/
# query_code): the isolated bench settings.json holds provider API keys, and a
# prompt-injected task is a normal root task that could read_file(root="runtime_data",
# "settings.json") to exfiltrate them. The agent inspects the VM through the skill
# (remote_exec/screenshot), never the host filesystem.
_ALLOWED_CORE_TOOLS = frozenset({
    "list_available_tools", "enable_tools",   # discover + enable the computer-use skill
    "view_image",                             # the vision channel (SEE screenshots)
    "compact_context", "set_tool_timeout",    # agent self-management (no host access)
})


def _core_tool_names() -> set[str]:
    """All built-in (non-extension) core tool names, for the computed denylist."""
    import tempfile

    from ouroboros.tools.registry import ToolRegistry

    tmp = Path(tempfile.mkdtemp(prefix="cu_bridge_toolscan_"))
    reg = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    return {t["function"]["name"] for t in reg.schemas()}


def _host_denied_tools() -> list[str]:
    """Deny every core tool the OSWorld task does not need (allowlist-complement)."""
    return sorted(_core_tool_names() - _ALLOWED_CORE_TOOLS)

# GUI action tools (short skill names) counted for the budget disclosure.
_GUI_ACTION_TOOLS = frozenset({
    "click", "move", "left_click_drag", "mouse_down", "mouse_up",
    "type_text", "key", "hold_key", "scroll",
})


# unix_computer_use ext tools the untrusted task must NOT reach. The runner pins
# the active connection to the published OSWorld VM; a task that could switch the
# backend (use_local/activate_connection local) or retarget it (add_connection)
# would drive the HOST desktop instead — defeating the host lockdown AND the
# fail-closed guarantee. Read-only introspection (list_connections/test_connection)
# stays; the mutating connection-management surface is denied.
_DENIED_SKILL_EXT_TOOLS = ("add_connection", "activate_connection", "use_local", "clear_active_connection")


def _effective_disabled_tools(allow_a11y: bool) -> list[str]:
    """Per-task disabled-tool list = the host-tool complement of the allowlist,
    plus the skill's connection-switching ext tools (the runner pins the VM
    connection), plus ``ax_tree`` unless ``--allow-a11y`` is given (screenshot-only
    by default; enabling it must disclose "a11y tree used"). ext names must be the
    provider-safe full surface names — disabled_tools matches exact names."""
    from ouroboros.extension_loader import extension_surface_name

    disabled = _host_denied_tools()
    disabled += [extension_surface_name(SKILL_NAME, t) for t in _DENIED_SKILL_EXT_TOOLS]
    if not allow_a11y:
        disabled.append(extension_surface_name(SKILL_NAME, "ax_tree"))
    return disabled


def _refuse_live_data_dir(data_dir: Path) -> None:
    """Never publish a bench connection into the owner's LIVE skill state — it
    would hijack the real unix_computer_use skill and point it at a bench VM."""
    live = (Path.home() / "Ouroboros" / "data").expanduser().resolve(strict=False)
    resolved = Path(data_dir).expanduser().resolve(strict=False)
    if resolved == live or live in resolved.parents:
        raise SystemExit(
            f"refusing --data-dir inside the live Ouroboros data root ({live}); "
            "use an isolated bench data dir"
        )


def _dataset_name(variant: str) -> str:
    return {"v2": "OSWorld-V2", "v1": "OSWorld"}.get(variant, f"OSWorld-{variant}")


def _effective_max_rounds(settings_path: Path) -> dict[str, Any]:
    """Report the round budget the bench server actually honors, with provenance.

    The server applies settings.json over env at startup, so settings wins; this
    is best-effort disclosure, not enforcement (there is no per-task step cap)."""
    try:
        settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        if isinstance(settings, dict) and settings.get("OUROBOROS_MAX_ROUNDS") is not None:
            return {"value": int(settings["OUROBOROS_MAX_ROUNDS"]), "source": "settings"}
    except Exception:
        pass
    env_val = os.environ.get("OUROBOROS_MAX_ROUNDS")
    if env_val:
        try:
            return {"value": int(env_val), "source": "env"}
        except ValueError:
            pass
    return {"value": 200, "source": "default"}


def _collect_budget_counters(data_dir: Path, latest: dict[str, Any], ouro_task_id: str) -> dict[str, Any]:
    """Disclosure counters for leaderboard comparability (never raises).

    A leaderboard "step" is one model turn; our rounds are not step-equivalent,
    so we publish the raw counts: llm rounds (authoritative, from the task
    result) plus per-tool call counts parsed from the task's own tools.jsonl.
    """
    from ouroboros.extension_loader import extension_name_prefix

    counters: dict[str, Any] = {"llm_rounds": int(latest.get("total_rounds") or 0)}
    prefix = extension_name_prefix(SKILL_NAME)
    child = latest.get("child_drive_root")
    log_path = (Path(child) / "logs" / "tools.jsonl") if child else (
        data_dir / "state" / "headless_tasks" / ouro_task_id / "data" / "logs" / "tools.jsonl"
    )
    fallback = data_dir / "logs" / "tools.jsonl"
    screenshots = gui = remote_exec = total = 0
    src = log_path if log_path.is_file() else (fallback if fallback.is_file() else None)
    if src is not None:
        for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or row.get("type") != "tool_call":
                continue
            if src is fallback and str(row.get("task_id") or "") != ouro_task_id:
                continue
            tool = str(row.get("tool") or "")
            if not tool.startswith(prefix):
                continue
            short = tool[len(prefix):]
            total += 1
            if short == "screenshot":
                screenshots += 1
            elif short == "remote_exec":
                remote_exec += 1
            elif short in _GUI_ACTION_TOOLS:
                gui += 1
    counters.update({
        "screenshots": screenshots,
        "gui_action_calls": gui,
        "remote_exec_calls": remote_exec,
        "skill_tool_calls": total,
        "tools_log": str(src) if src is not None else "",
    })
    return counters

OSWORLD_PREAMBLE = (
    "You are operating a real Ubuntu desktop inside an OSWorld VM, by yourself, to "
    "completion. Drive the VM like a skilled human user: look at the screen, click "
    "menus/buttons, type into dialogs, use keyboard shortcuts, save/confirm, and verify.\n"
    "The 'unix_computer_use' skill is enabled with an active OSWorld HTTP backend; its tools act on THIS VM. Call "
    "list_available_tools (or enable_tools) to get the names (ext_<n>_r_unix_computer_use_"
    "screenshot, _click, _type_text, _key, _scroll, _left_click_drag, _move, _wait, "
    "_remote_exec) and enable them.\n"
    "\n"
    "FIRST, DO A FEASIBILITY CHECK: before executing a long plan, decide "
    "whether the task is possible on this VM with the installed apps, hardware, accounts, "
    "and allowed tools. Use at most 1-2 concise remote_exec probes if needed (e.g. hardware "
    "exists? app feature exists? required file exists?). If the requested result requires "
    "missing hardware, unavailable accounts/cloud/collaboration infrastructure, or a feature "
    "the installed application genuinely does not provide, do not keep trying — end your "
    "final message with only: TASK_INFEASIBLE.\n"
    "\n"
    "PRIMARY RULE — HUMAN GUI CONTROL:\n"
    "- For application tasks (Thunderbird, Chrome, LibreOffice, VS Code, GIMP, VLC, OS "
    "settings), solve through the visible application UI unless the task explicitly says "
    "\"command line\" or is obviously file/media batch processing.\n"
    "- Treat GUI actions as the official action surface: screenshot/view_image, click, "
    "type_text, key, scroll, drag. This should be MOST of your actions, like a human using "
    "the VM. Do not replace a GUI workflow with prefs.js edits, UNO/Basic macros, "
    "python-pptx, profile hacks, XML edits, or other behind-the-back mutations.\n"
    "- remote_exec is NOT your main problem-solving channel for app tasks. It is allowed only "
    "for a quick read-only check (for example, verify a saved file or exact setting) or "
    "for tasks whose wording explicitly asks for command-line work/conversion/media tools.\n"
    "\n"
    "VISION LOOP — do exactly this for GUI work:\n"
    "  1. screenshot (returns a 'path') -> immediately view_image(path) so you SEE the desktop.\n"
    "  2. Read coordinates off that viewed image, then act with click/key/type_text/scroll.\n"
    "  3. Take another screenshot+view_image only after a meaningful UI state change.\n"
    "view_image is your visual channel. (vlm_query, analyze_screenshot and browser tools are "
    "DISABLED — do not look for them.)\n"
    "\n"
    "BE FAST — every tool call costs ~30s, so MINIMIZE calls:\n"
    "- Do not spend more than 2 calls on investigation before taking a real GUI action.\n"
    "- Batch 2-4 confident GUI actions before the next screenshot+view_image. Do NOT screenshot "
    "after every single keystroke.\n"
    "- Prefer keyboard shortcuts when faster (menus via Alt, Ctrl+S to save, etc.).\n"
    "- If remote_exec is legitimately needed, use at most 1-2 concise read-only checks before the "
    "next GUI action. Do not repeatedly grep/probe internals. NEVER use remote_exec to see the "
    "screen, pixel-analyze screenshots, or run ImageGrab/scrot/numpy screen analysis.\n"
    "\n"
    "Anti-loop: if the same action fails twice, change approach (different menu path, "
    "keyboard), but stay in the GUI for app tasks; never fall back to pixel analysis or profile "
    "hacking.\n"
    "OSWorld evaluates the VM state, not your chat answer. Unless the task explicitly asks you "
    "to write an answer in a document/app, a textual answer in chat is not success: leave the "
    "requested browser tab, file, setting, app state, or saved artifact in the VM.\n"
    "BEFORE YOUR VERDICT, VERIFY THE FINAL ENVIRONMENT STATE: re-check that the VM state right now "
    "genuinely satisfies EVERY requirement of the task. Judge by the real, observed state — re-open and "
    "look at the relevant file/app/setting — not by your belief that you performed the steps. If any "
    "requirement is not fully met (including a change made but not saved/applied), keep working; declare "
    "done only when the observed state matches the task. If the task is genuinely impossible on this VM, "
    "end with TASK_INFEASIBLE.\n"
    "Be decisive and efficient. When the task is verifiably complete in the real app, stop. "
    "If genuinely infeasible, end your final message with only: TASK_INFEASIBLE\n\nTask:\n"
)

_COMPUTER_USE_SHORT_TOOLS = (
    "list_connections", "test_connection", "screenshot", "click", "move",
    "left_click_drag", "mouse_down", "mouse_up", "type_text", "key", "hold_key",
    "scroll", "wait", "window_list", "ax_tree", "cursor_position", "remote_exec",
)


def _ensure_vmrun_on_path() -> None:
    parts = os.environ.get("PATH", "").split(os.pathsep)
    changed = False
    for cand in VMWARE_FUSION_PATHS:
        if Path(cand, "vmrun").exists() and cand not in parts:
            parts.insert(0, cand)
            changed = True
    if changed:
        os.environ["PATH"] = os.pathsep.join(parts)


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
    return _text_declares_infeasible(latest.get("final_answer")) or _text_declares_infeasible(latest.get("result"))


def _enable_skill(repo_dir: Path, data_dir: Path) -> str:
    """Controlled-seed + native-trust + enable unix_computer_use.

    Launcher auto-seeding won't pick up a brand-new bundled skill on an already
    bootstrapped data dir, and an existing native seed may be stale for this
    worktree. Re-copy the repo skill into THIS isolated bench data dir and stamp
    native trust against the current hash. Idempotent: re-copies each run so repo
    edits are reflected. The ``net`` permission needs no owner grant, but it does
    remove the skill from the launcher's native auto-enable class — this runner
    therefore enables it explicitly via ``save_enabled``.
    """
    import logging
    import shutil
    from ouroboros.launcher_bootstrap import _stamp_native_seed_trust
    from ouroboros.skill_loader import find_skill, save_enabled

    src = repo_dir / "skills" / SKILL_NAME
    if not src.is_dir():
        raise RuntimeError(f"{SKILL_NAME} not found in repo skills: {src}")
    dest = data_dir / "skills" / "native" / SKILL_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    (dest / ".seed-origin").write_text("seeded_from=bench_cu_bridge\n", encoding="utf-8")
    shutil.rmtree(dest / "__pycache__", ignore_errors=True)
    _stamp_native_seed_trust(data_dir, dest, logging.getLogger("osworld_bridge"))
    skill = find_skill(data_dir, SKILL_NAME)
    if skill is None or getattr(skill, "load_error", None):
        raise RuntimeError(f"{SKILL_NAME} unavailable after seed: {getattr(skill, 'load_error', None)}")
    save_enabled(data_dir, SKILL_NAME, True)
    review = getattr(getattr(skill, "review", None), "status", "?")
    return f"{skill.name} ({skill.source}) review={review} enabled=True"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write+rename so a crash can't leave a torn registry file (matches the
    skill's own atomic writer; the disclosed safety property must hold here too)."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _publish_target(data_dir: Path, target: str) -> Path:
    """Activate an osworld_http connection in unix_computer_use skill state.

    The skill worker may not inherit the server's custom env, so the robust
    channel is shared skill state: <data>/state/skills/unix_computer_use/connections.json.
    Registry first, active pointer last (both atomic) so a lost second write still
    names a connection that exists in the registry.
    """
    from ouroboros.skill_loader import skill_state_dir

    sdir = Path(skill_state_dir(data_dir, SKILL_NAME))
    sdir.mkdir(parents=True, exist_ok=True)
    target_path = sdir / "osworld_target.txt"
    _atomic_write_text(target_path, target)
    registry = {
        "active": "osworld-current",
        "connections": {
            "local": {"backend": "local", "enabled": True},
            "osworld-current": {"backend": "osworld_http", "target_file": str(target_path), "enabled": True},
        },
    }
    _atomic_write_text(sdir / "connections.json", json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(sdir / "active_connection.txt", "osworld-current")
    return target_path


def main() -> int:
    _ensure_vmrun_on_path()
    p = argparse.ArgumentParser(description="OSWorld via host-side Ouroboros computer-use bridge (one run per task).")
    p.add_argument("--osworld-root", default=os.environ.get("OSWORLD_ROOT", str(_WORKSPACE_ROOT / "OSWorld")))
    p.add_argument("--provider_name", default="vmware")
    p.add_argument("--path_to_vm", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--result_dir", default="results/osworld_cu_bridge")
    p.add_argument("--repo-dir", default=str(_REPO_ROOT))
    p.add_argument("--data-dir", required=True, help="bench server data dir (skill enablement target)")
    p.add_argument("--settings-path", default="",
                   help="settings.json the bench server was started with (for the max_rounds disclosure); "
                        "defaults to <data-dir>/settings.json — NOT the live workspace settings")
    p.add_argument("--ouroboros-url", default="http://127.0.0.1:8780")
    p.add_argument("--target-file", required=True, help="informational copy of the VM HTTP target URL the runner writes (also recorded in bridge.json); the published osworld_http connection reads a SEPARATE state-confined copy under the skill state dir, since target_file reads are confined there")
    # NOTE: the solve model is set by the Ouroboros server's settings (OUROBOROS_MODEL);
    # this runner does not accept a --model flag so provenance can't be misreported.
    p.add_argument("--task_timeout_sec", type=int, default=3600)
    p.add_argument("--startup_timeout_sec", type=int, default=900)
    p.add_argument("--reset_retries", type=int, default=3)
    p.add_argument("--wait_after_reset_sec", type=float, default=12.0)
    p.add_argument("--show-vm", action="store_true")
    p.add_argument("--allow-a11y", action="store_true",
                   help="expose the ax_tree (accessibility) tool; the run is then NOT screenshot-only "
                        "(disclose 'Additional a11y tree used: Yes'). Off by default.")
    p.add_argument("--allow-live-server", action="store_true",
                   help="permit pointing --ouroboros-url at the live desktop server port 8765 (debug only).")
    args = p.parse_args()

    # Guards: never drive the live desktop server or publish a bench connection
    # into the owner's live skill state (mirrors run_step_agent.py).
    from devtools.benchmarks.osworld.run_step_agent import (
        _is_default_desktop_server,
        osworld_checkout_info,
    )
    if _is_default_desktop_server(args.ouroboros_url) and not args.allow_live_server:
        raise SystemExit(
            f"refusing the live desktop server URL {args.ouroboros_url}; point at an isolated "
            "bench server (fresh OUROBOROS_DATA_DIR, non-default port) or pass --allow-live-server"
        )
    _refuse_live_data_dir(Path(args.data_dir))

    osworld_root = Path(args.osworld_root).expanduser().resolve(strict=False)
    sys.path.insert(0, str(osworld_root))
    checkout = osworld_checkout_info(osworld_root)
    dataset_name = _dataset_name(str(checkout.get("variant") or "unknown"))
    task_path = Path(args.task).expanduser()
    if not task_path.is_absolute():
        task_path = osworld_root / task_path
    domain = task_path.parent.name
    example_id = task_path.stem
    repo_dir = Path(args.repo_dir).expanduser().resolve(strict=False)
    data_dir = Path(args.data_dir).expanduser().resolve(strict=False)
    # Default the settings path INTO the isolated bench data dir, not the live
    # workspace settings, so the max_rounds disclosure reflects THIS server.
    settings_path = Path(args.settings_path).expanduser().resolve(strict=False) if args.settings_path else (data_dir / "settings.json")
    result_root = Path(args.result_dir).expanduser()
    if not result_root.is_absolute():
        result_root = osworld_root / result_root
    result_root = ensure_outside_repo(result_root, repo_dir)
    run_dir = result_root / domain / example_id
    run_dir.mkdir(parents=True, exist_ok=True)

    example = json.loads(task_path.read_text(encoding="utf-8"))
    example_id = str(example.get("id") or example_id)
    instruction = str(example["instruction"])
    (run_dir / "task.json").write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_outcome(reward: float | None, status: str, reason: str, error: str = "", extra: dict[str, Any] | None = None) -> dict[str, Any]:
        outcome = {
            "ok": status == "completed",
            "task_id": example_id, "domain": domain, "reward": reward,
            "status": status, "reason_code": reason, "error": error,
            "result_dir": str(run_dir), **(extra or {}),
        }
        write_json(run_dir / "task_outcome.json", outcome)
        write_json(run_dir / "task_run_manifest.json", benchmark_run_manifest(
            benchmark="osworld", run_root=result_root, repo_dir=repo_dir,
            requested_task_ids=[example_id], dataset=dataset_name, settings_path=settings_path,
            output_paths={"task_outcome": str(run_dir / "task_outcome.json")},
            harness={
                # HONEST contract: reset()/evaluate() are official, but GUI actions
                # go to the guest /execute channel and are NOT recorded in
                # DesktopEnv.action_history/traj.jsonl (only a translated FAIL is).
                "adapter": "host_cu_bridge", "one_run_per_task": True,
                "official_actions": False, "official_reset_evaluate": True,
                "action_channel": "guest_execute_not_env_step",
                "osworld_checkout": checkout,
                "a11y_enabled": bool(args.allow_a11y),
                "max_rounds_effective": _effective_max_rounds(settings_path),
            },
            extra=(extra or {}),
        ))
        append_result_index(result_root, task_result_row(
            benchmark="osworld", instance_id=example_id, status=status, reason_code=reason,
            official_eval_status="completed" if reward is not None else "not_run",
            output_paths={"task_outcome": str(run_dir / "task_outcome.json")},
            error=error, details={"domain": domain, "reward": reward, **(extra or {})},
        ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return outcome

    # Enable the computer-use skill in the server's data dir.
    try:
        enabled = _enable_skill(repo_dir, data_dir)
    except Exception as exc:  # noqa: BLE001
        _write_outcome(None, "blocked", "skill_enable_failed", f"{type(exc).__name__}: {exc}")
        return 2

    # Wire OSWorld's proxy pool (e.g. DataImpulse residential) for tasks flagged
    # "proxy": true. Only enable when a proxy config file actually exists, else
    # OSWorld raises "No proxy available" and hard-fails those tasks. Non-proxy
    # tasks are unaffected (OSWorld gates on task_config["proxy"] AND enable_proxy).
    # PROXY_CONFIG_FILE must be set BEFORE importing desktop_env: setup.py loads
    # the pool at import time.
    _proxy_cfg = os.environ.get("PROXY_CONFIG_FILE") or str(
        osworld_root / "evaluation_examples" / "settings" / "proxy" / "dataimpulse.json"
    )
    _enable_proxy = os.path.exists(_proxy_cfg)
    if _enable_proxy:
        os.environ["PROXY_CONFIG_FILE"] = os.path.abspath(_proxy_cfg)
    print(f"[bridge] enable_proxy={_enable_proxy} "
          f"proxy_cfg={os.environ['PROXY_CONFIG_FILE'] if _enable_proxy else '(none)'}", flush=True)

    from desktop_env.desktop_env import DesktopEnv

    env = None
    try:
        env = DesktopEnv(
            provider_name=args.provider_name, path_to_vm=args.path_to_vm,
            action_space="pyautogui", screen_size=(1920, 1080),
            headless=not args.show_vm, os_type="Ubuntu", require_a11y_tree=False,
            enable_proxy=_enable_proxy,
        )
        # Reset with retries to a usable screenshot.
        deadline = time.time() + max(1, int(args.startup_timeout_sec))
        last_err = ""
        ok = False
        for attempt in range(1, max(1, int(args.reset_retries)) + 1):
            if time.time() >= deadline:
                break
            try:
                env.reset(task_config=example)
                if args.wait_after_reset_sec > 0:
                    time.sleep(args.wait_after_reset_sec)
                obs = env._get_obs()
                if isinstance(obs, dict) and isinstance(obs.get("screenshot"), (bytes, bytearray)) and obs["screenshot"]:
                    ok = True
                    break
                last_err = f"attempt {attempt}: no screenshot"
            except Exception as exc:  # noqa: BLE001
                last_err = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            time.sleep(5)
        if not ok:
            raise RuntimeError(f"OSWorld startup failed: {last_err}")

        target = f"http://{env.vm_ip}:{env.server_port}"
        Path(args.target_file).expanduser().write_text(target, encoding="utf-8")
        state_target = _publish_target(data_dir, target)
        (run_dir / "bridge.json").write_text(json.dumps({
            "target": target, "skill": enabled, "target_file": args.target_file,
            "state_target_file": str(state_target),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        prompt = OSWORLD_PREAMBLE + instruction + (
            "\n\nunix_computer_use tools (enable then use; discover exact ext_<n>_ names via "
            "list_available_tools): " + ", ".join(_COMPUTER_USE_SHORT_TOOLS) + ". They act on THIS VM "
            "because the runner activated the osworld-current connection."
            f"\n\nVM CREDENTIALS: the desktop user is 'user' and its sudo password is "
            f"'{env.client_password}'. When a task GENUINELY needs root (create system users, "
            f"start/enable a service, install packages) or a GUI dialog prompts for a password, "
            f"use it — e.g. run privileged commands as: echo '{env.client_password}' | sudo -S <cmd>. "
            f"Still prefer the visible GUI for application tasks per the rules above; sudo is for "
            f"the OS/CLI steps that truly require root."
        )
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        created = _api(args.ouroboros_url, "POST", "/api/tasks", {
            "description": prompt, "memory_mode": "empty",
            "disabled_tools": _effective_disabled_tools(args.allow_a11y),
        })
        task_id = str(created.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"task creation returned no task_id: {created!r}")
        (run_dir / "ouroboros_task_id.txt").write_text(task_id, encoding="utf-8")

        final_statuses = {"completed", "failed", "cancelled", "rejected_duplicate"}
        t_deadline = time.time() + max(60, int(args.task_timeout_sec))
        latest: dict[str, Any] = {}
        while True:
            if time.time() >= t_deadline:
                try:
                    _api(args.ouroboros_url, "POST", f"/api/tasks/{task_id}/cancel", {})
                except Exception:
                    pass
                latest = {"status": "timeout"}
                break
            try:
                result = _api(args.ouroboros_url, "GET", "/api/tasks/" + task_id, timeout=30)
            except Exception:
                time.sleep(5)
                continue
            latest = result if isinstance(result, dict) else {}
            if str(latest.get("status") or "") in final_statuses:
                break
            time.sleep(8)
        (run_dir / "ouroboros_task_final.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")

        infeasible_declared = _final_answer_declares_infeasible(latest)
        fail_info: dict[str, Any] = {}
        if infeasible_declared:
            try:
                _obs_after_fail, _reward_after_fail, _done_after_fail, fail_info = env.step("FAIL")
            except Exception as exc:  # noqa: BLE001 - keep denominator-preserving evaluation
                fail_info = {"error": f"{type(exc).__name__}: {exc}"}
            (run_dir / "osworld_fail_action.json").write_text(
                json.dumps({"declared": True, "info": fail_info}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        try:
            budget_counters: dict[str, Any] = _collect_budget_counters(data_dir, latest, task_id)
        except Exception as exc:  # noqa: BLE001 - counters are disclosure-only, never fail the run
            budget_counters = {"budget_counters_error": f"{type(exc).__name__}: {exc}"}

        reward = float(env.evaluate())
        (run_dir / "result.txt").write_text(f"{reward}\n", encoding="utf-8")
        _write_outcome(reward, "completed", "official_evaluate", extra={
            "ouroboros_status": str(latest.get("status") or ""),
            "task_id_ouroboros": task_id,
            "infeasible_declared": infeasible_declared,
            "a11y_enabled": bool(args.allow_a11y),
            "budget_counters": budget_counters,
            "max_rounds_effective": _effective_max_rounds(settings_path),
            **({"osworld_fail_info": fail_info} if infeasible_declared else {}),
        })
        return 0
    except Exception as exc:  # noqa: BLE001 - denominator-preserving adapter failure
        _write_outcome(None, "adapter_error", type(exc).__name__, f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
