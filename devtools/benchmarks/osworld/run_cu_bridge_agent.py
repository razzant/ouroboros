#!/usr/bin/env python3
"""OSWorld runner: ONE Ouroboros agentic run per task, host-side computer-use bridge.

Unlike ``run_step_agent.py`` (host drives ``env.step`` and Ouroboros is a stateless
per-step action selector with ``--memory-mode empty``), this runner gives Ouroboros
the wheel:

    host: reset VM -> publish VM_IP -> submit ONE task -> wait -> evaluate()
    agent (one run, full memory): screenshot -> reason -> click/type -> screenshot -> ... -> done

The agent acts through the bundled ``unix_computer_use`` skill, whose additive
OSWorld HTTP backend (active when ``OUROBOROS_CU_HTTP_TARGET[_FILE]`` is set) routes
``screenshot``/``click``/``type``/``key``/``scroll`` to the in-VM OSWorld server
(GET /screenshot, POST /execute) — the SAME guest channel ``env.step`` uses. The
brain stays on the host; only translated pyautogui mutates the guest. ``reset()`` and
``evaluate()`` are the official OSWorld ones, so scoring stays comparable.

This is the Terminal-Bench / Pointer shape (persistent agent + computer-use tool),
without installing Ouroboros inside the VM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
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

# Tools hard-disabled per task (stronger than prompt nudging): redundant/expensive
# vision (vlm_query/analyze_screenshot spawn an extra model call per look),
# irrelevant browser tools, host shell (run_command/run_script run on the HOST, not
# the VM — useless here; remote_exec is the VM shell), and chat photo noise. The lean
# loop is screenshot -> view_image -> GUI action(s) -> (sparingly) remote_exec verify.
DISABLED_TOOLS = [
    "vlm_query", "analyze_screenshot", "browse_page", "browser_action",
    "run_command", "run_script", "send_photo",
]

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
    "BEFORE DONE, REVIEW CAREFULLY: compare the original task wording against the exact "
    "evaluator-facing state you changed. Verify the final state once (screen state or a concise "
    "read-only remote_exec check). Do not stop if you changed a similar-but-wrong setting, wrong "
    "file/path, wrong browser tab/URL, wrong extension path, unsaved document, or only described "
    "the answer in chat. If evidence is missing, keep working. If it is impossible, use "
    "TASK_INFEASIBLE.\n"
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


def _install_optional_dependency_stubs() -> None:
    """Avoid heavy optional evaluator imports for tasks that do not need them."""
    if "easyocr" not in sys.modules:
        easyocr = types.ModuleType("easyocr")

        class _UnavailableReader:
            def __init__(self, *_a: Any, **_k: Any) -> None:
                raise RuntimeError("easyocr is not installed; OCR metrics unavailable")

        easyocr.Reader = _UnavailableReader  # type: ignore[attr-defined]
        sys.modules["easyocr"] = easyocr
    if "fastdtw" not in sys.modules:
        fastdtw_mod = types.ModuleType("fastdtw")

        def _fastdtw_unavailable(*_a: Any, **_k: Any):
            raise RuntimeError("fastdtw is not installed; audio metrics unavailable")

        fastdtw_mod.fastdtw = _fastdtw_unavailable  # type: ignore[attr-defined]
        sys.modules["fastdtw"] = fastdtw_mod


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


def _contains_task_infeasible(value: Any) -> bool:
    """True iff the final agent result contains TASK_INFEASIBLE as a standalone line.

    OSWorld's infeasible evaluators check the official action history for FAIL; a
    chat text marker alone is not enough. This detects the marker in the task
    result so the bridge can emit the equivalent official OSWorld FAIL action
    before evaluate().
    """
    if isinstance(value, str):
        return any(line.strip() == "TASK_INFEASIBLE" for line in value.splitlines())
    if isinstance(value, dict):
        return any(_contains_task_infeasible(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_task_infeasible(v) for v in value)
    return False


def _enable_skill(repo_dir: Path, data_dir: Path) -> str:
    """Controlled-seed + native-trust + enable unix_computer_use.

    Launcher auto-seeding won't pick up a brand-new bundled skill on an already
    bootstrapped data dir, and an existing native seed may be stale for this
    worktree. Re-copy the repo skill into THIS isolated bench data dir and stamp
    native trust against the current hash. Idempotent: re-copies each run so repo
    edits are reflected. ``net`` needs no owner grant.
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


def _publish_target(data_dir: Path, target: str) -> Path:
    """Activate an osworld_http connection in unix_computer_use skill state.

    The skill worker may not inherit the server's custom env, so the robust
    channel is shared skill state: <data>/state/skills/unix_computer_use/connections.json.
    """
    from ouroboros.skill_loader import skill_state_dir

    sdir = Path(skill_state_dir(data_dir, SKILL_NAME))
    sdir.mkdir(parents=True, exist_ok=True)
    target_path = sdir / "osworld_target.txt"
    target_path.write_text(target, encoding="utf-8")
    registry = {
        "active": "osworld-current",
        "connections": {
            "local": {"backend": "local", "enabled": True},
            "osworld-current": {"backend": "osworld_http", "target_file": str(target_path), "enabled": True},
        },
    }
    (sdir / "connections.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (sdir / "active_connection.txt").write_text("osworld-current", encoding="utf-8")
    return target_path


def main() -> int:
    _ensure_vmrun_on_path()
    _install_optional_dependency_stubs()
    p = argparse.ArgumentParser(description="OSWorld via host-side Ouroboros computer-use bridge (one run per task).")
    p.add_argument("--osworld-root", default=os.environ.get("OSWORLD_ROOT", str(_WORKSPACE_ROOT / "OSWorld")))
    p.add_argument("--provider_name", default="vmware")
    p.add_argument("--path_to_vm", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--result_dir", default="results/osworld_cu_bridge")
    p.add_argument("--repo-dir", default=str(_REPO_ROOT))
    p.add_argument("--data-dir", required=True, help="bench server data dir (skill enablement target)")
    p.add_argument("--settings-path", default=str(_WORKSPACE_ROOT / "data" / "settings.json"))
    p.add_argument("--ouroboros-url", default="http://127.0.0.1:8780")
    p.add_argument("--target-file", required=True, help="file the skill reads for the VM HTTP target (OUROBOROS_CU_HTTP_TARGET_FILE)")
    p.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    p.add_argument("--task_timeout_sec", type=int, default=3600)
    p.add_argument("--startup_timeout_sec", type=int, default=900)
    p.add_argument("--reset_retries", type=int, default=3)
    p.add_argument("--wait_after_reset_sec", type=float, default=12.0)
    p.add_argument("--show-vm", action="store_true")
    args = p.parse_args()

    osworld_root = Path(args.osworld_root).expanduser().resolve(strict=False)
    sys.path.insert(0, str(osworld_root))
    task_path = Path(args.task).expanduser()
    if not task_path.is_absolute():
        task_path = osworld_root / task_path
    domain = task_path.parent.name
    example_id = task_path.stem
    repo_dir = Path(args.repo_dir).expanduser().resolve(strict=False)
    data_dir = Path(args.data_dir).expanduser().resolve(strict=False)
    settings_path = Path(args.settings_path).expanduser().resolve(strict=False)
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
            requested_task_ids=[example_id], dataset="OSWorld", settings_path=settings_path,
            output_paths={"task_outcome": str(run_dir / "task_outcome.json")},
            harness={"adapter": "host_cu_bridge", "official_actions": True, "one_run_per_task": True},
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

    from desktop_env.desktop_env import DesktopEnv

    env = None
    try:
        env = DesktopEnv(
            provider_name=args.provider_name, path_to_vm=args.path_to_vm,
            action_space="pyautogui", screen_size=(1920, 1080),
            headless=not args.show_vm, os_type="Ubuntu", require_a11y_tree=False,
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
        )
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        created = _api(args.ouroboros_url, "POST", "/api/tasks", {
            "description": prompt, "memory_mode": "empty", "disabled_tools": DISABLED_TOOLS,
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

        infeasible_declared = _contains_task_infeasible(latest)
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

        reward = float(env.evaluate())
        (run_dir / "result.txt").write_text(f"{reward}\n", encoding="utf-8")
        _write_outcome(reward, "completed", "official_evaluate", extra={
            "ouroboros_status": str(latest.get("status") or ""),
            "task_id_ouroboros": task_id,
            "infeasible_declared": infeasible_declared,
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
