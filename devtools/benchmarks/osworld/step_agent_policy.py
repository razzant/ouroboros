"""The Ouroboros step-loop policy: one CLI call per observation.

Verbatim extraction from ``run_step_agent.py`` (v7 stream W): the bounded
initial-observation retry and ``OuroborosStepAgent`` itself — screenshot
persistence, accessibility prioritisation, prompt construction, prediction and
action recording.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from devtools.benchmarks.osworld.step_agent_actions import (
    SPECIAL_ACTIONS,
    _json_from_text,
    _normalize_structured_action,
)
from devtools.benchmarks.osworld.step_agent_common import StepAgentConfig

def _initial_observation_with_retries(
    env: Any,
    example: dict[str, Any],
    *,
    startup_timeout_sec: int,
    reset_retries: int,
    wait_after_reset_sec: float,
    retry_sleep_sec: float,
    run_dir: Path,
) -> dict[str, Any]:
    """Reset OSWorld and wait for a usable first observation.

    VM reset, in-VM server readiness, screenshot capture, and accessibility-tree
    availability are startup concerns, not agent reasoning steps. Keep retrying
    them within a dedicated startup budget so transient VM/controller slowness does
    not become a task failure.
    """

    deadline = time.time() + max(1, int(startup_timeout_sec))
    attempts = max(1, int(reset_retries))
    errors: list[str] = []
    last_obs: dict[str, Any] = {}

    for attempt in range(1, attempts + 1):
        if time.time() >= deadline:
            break
        try:
            obs = env.reset(task_config=example)
            if wait_after_reset_sec > 0:
                time.sleep(wait_after_reset_sec)
            while time.time() < deadline:
                try:
                    obs = env._get_obs()
                    last_obs = obs if isinstance(obs, dict) else {}
                    screenshot = last_obs.get("screenshot")
                    if isinstance(screenshot, (bytes, bytearray)) and screenshot:
                        (run_dir / "startup_readiness.json").write_text(
                            json.dumps(
                                {
                                    "ok": True,
                                    "attempt": attempt,
                                    "has_screenshot": True,
                                    "has_accessibility_tree": bool(last_obs.get("accessibility_tree")),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        return last_obs
                    errors.append(f"attempt {attempt}: observation missing screenshot")
                except Exception as exc:  # noqa: BLE001 - startup retry diagnostics
                    errors.append(f"attempt {attempt}: _get_obs {type(exc).__name__}: {exc}")
                time.sleep(max(0.1, retry_sleep_sec))
            break
        except Exception as exc:  # noqa: BLE001 - reset retry diagnostics
            errors.append(f"attempt {attempt}: reset {type(exc).__name__}: {exc}")
            time.sleep(max(0.1, retry_sleep_sec))

    (run_dir / "startup_readiness.json").write_text(
        json.dumps(
            {
                "ok": False,
                "errors": errors[-20:],
                "last_obs_keys": sorted(last_obs.keys()),
                "startup_timeout_sec": startup_timeout_sec,
                "reset_retries": reset_retries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    raise RuntimeError(
        f"OSWorld startup did not produce a usable screenshot within {startup_timeout_sec}s; "
        f"last errors: {errors[-3:]}"
    )


class OuroborosStepAgent:
    def __init__(
        self,
        config: StepAgentConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = StepAgentConfig(**kwargs)
        self.ouroboros_bin = config.ouroboros_bin
        self.ouroboros_url = config.ouroboros_url
        self.repo_dir = config.repo_dir
        self.data_dir = config.data_dir
        self.settings_path = config.settings_path
        self.result_dir = config.result_dir
        self.model = config.model
        self.timeout_sec = config.timeout_sec
        self.max_obs_chars = config.max_obs_chars
        self.screenshot_check_only = config.screenshot_check_only
        self.disable_tools = config.disable_tools
        self.step_idx = 0
        self.history: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.final_answer = ""
        self.terminal_action = ""
        self.last_response = ""

    def reset(self) -> None:
        self.step_idx = 0
        self.history.clear()
        self.notes.clear()
        self.final_answer = ""
        self.terminal_action = ""
        self.last_response = ""

    def _save_screenshot(self, obs: dict[str, Any]) -> tuple[str, str]:
        screenshot = obs.get("screenshot")
        if not isinstance(screenshot, (bytes, bytearray)):
            return "", ""
        self.step_idx += 1
        name = f"step_{self.step_idx:03d}.png"
        local_path = self.result_dir / f"obs_{name}"
        local_path.write_bytes(bytes(screenshot))
        return str(local_path), str(local_path.name)

    @staticmethod
    def _prioritize_a11y(tree: str, budget: int) -> str:
        """Budget-bounded a11y view that PRIORITIZES interactive elements with
        coordinates instead of a blind head-slice (WS-9.6).

        A head-slice (the previous behavior) routinely cut the tree before the
        actionable widgets, so the agent never saw the controls it needed to
        click and resorted to blind/CLI moves. Here, when over budget, lines
        that name an interactive role AND/OR carry coordinates are kept first,
        then the rest in document order until the budget is spent.
        """
        if len(tree) <= budget:
            return tree
        lines = tree.splitlines()
        interactive = ("button", "menu", "entry", "text", "link", "check", "radio",
                       "tab", "combo", "field", "item", "toggle", "slider", "icon", "edit")
        coord_markers = ("coord", "position", "x=", "cp:", "screencoord", "bbox", "point")

        def score(line: str) -> int:
            low = line.lower()
            s = 0
            if any(k in low for k in interactive):
                s += 2
            if any(k in low for k in coord_markers):
                s += 2
            return s

        kept: list[tuple[int, str]] = []
        total = 0
        for _s, idx, line in sorted(((score(ln), i, ln) for i, ln in enumerate(lines)),
                                    key=lambda t: (-t[0], t[1])):
            if _s == 0 and total > 0:
                continue  # only spend budget on signal-bearing lines once we have some
            if total + len(line) + 1 > budget:
                continue
            kept.append((idx, line))
            total += len(line) + 1
        kept.sort()
        body = "\n".join(line for _i, line in kept)
        return body + "\n...[a11y prioritized: interactive/coordinate nodes kept, low-signal nodes dropped]"

    def _prompt(self, instruction: str, obs: dict[str, Any], screenshot_path: str, *, max_steps: int) -> str:
        a11y_tree = self._prioritize_a11y(str(obs.get("accessibility_tree") or ""), self.max_obs_chars)

        history_json = json.dumps(self.history[-12:], ensure_ascii=False, indent=2)
        notes_json = json.dumps(self.notes[-8:], ensure_ascii=False, indent=2)
        screenshot_instruction = (
            f'The current VM screenshot is attached to this Ouroboros run and also saved at "{screenshot_path}". '
            "Use the image directly when choosing GUI actions. If image input is unavailable, "
            "fall back to vlm_query(file_path=that path, prompt='Describe the Ubuntu desktop state and relevant controls')."
            if screenshot_path
            else "No screenshot bytes were available in this observation."
        )
        if self.screenshot_check_only:
            task_directive = (
                "This is a screenshot visibility smoke test. Use vlm_query on the "
                "screenshot path, then return WAIT with a short description of what "
                "you saw."
            )
        else:
            task_directive = (
                f"Choose the next OSWorld action(s). You are on step {self.step_idx} of at most {max_steps}. "
                "Prefer structured actions, not raw "
                "Python. Supported action objects: "
                '{"type":"shell","command":"...","cwd":"/home/user/Desktop"} (runs via non-interactive bash); '
                '{"type":"click","x":100,"y":200}; '
                '{"type":"type","text":"..."}; '
                '{"type":"hotkey","keys":["ctrl","l"]}; '
                '{"type":"wait","seconds":1}; '
                '{"type":"done"}; {"type":"fail"}. '
                'Use {"type":"python","code":"..."} only when no structured action fits. '
                "THE GRADER INSPECTS VM STATE ONLY. The OSWorld evaluator scores the virtual machine's "
                "state after your final step: files saved at the exact requested paths, in-application "
                "document state, the browser's ACTIVE TAB URL, and OS configuration. Text you write in "
                "chat is NEVER read by the evaluator. If the task asks a question, navigate the GUI until "
                "the answer is shown in the expected application/page and LEAVE the environment in that "
                "state (for example the browser tab open on the page that answers the question) before done. "
                "If the task edits a document or spreadsheet, SAVE the file to the exact expected path "
                "before done — an unsaved buffer or a chat answer scores zero. "
                "In app-named tasks, work in the named app first; if you edit files directly, reopen/verify in that app before done. "
                "Use done only after independently checking the evaluator-facing state. "
                "Use fail when demonstrably infeasible (missing hardware/resource, blocked permissions, feature absent); an out-of-app workaround is not success for an in-app task. "
                'When you return done or fail, ALSO set "final_answer" to your definitive short answer '
                "(for question-style tasks) or a one-line completion/infeasibility summary — it is recorded "
                "in the audit ledger, but it never replaces the required VM state. "
                "Do NOT claim a screenshot or VLM 'confirmed' / 'shows' anything unless you actually called vlm_query (or were given image input) THIS step; otherwise describe only what the accessibility tree and action history establish."
            )

        return f"""You are Ouroboros acting as an external OSWorld step-loop agent.
Return ONLY a JSON object, with no markdown and no prose outside JSON.

JSON schema:
{{"response": "short rationale", "notes": "optional cross-step note for yourself", "final_answer": "REQUIRED with done/fail: definitive short answer or completion summary", "actions": [{{"type": "shell", "command": "..."}}]}}

{task_directive}
{screenshot_instruction}

Task:
{instruction}

Recent official OSWorld action history:
{history_json}

Cross-step notes:
{notes_json}

Accessibility tree (may be empty/truncated):
{a11y_tree}
"""

    def predict(self, instruction: str, obs: dict[str, Any], *, max_steps: int) -> tuple[str, list[str], dict[str, Any]]:
        screenshot_path, local_screenshot = self._save_screenshot(obs)
        prompt = self._prompt(instruction, obs, screenshot_path, max_steps=max_steps)
        step = self.step_idx
        prompt_path = self.result_dir / f"prompt_step_{step:03d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        env = os.environ.copy()
        # NB: `ouroboros run --url` submits over the gateway, so these env vars
        # configure only the CLI subprocess, NOT the executing server — the
        # disclosed scaffold defaults are ENFORCED by the preflight check of the
        # target server's /api/settings (see _preflight). Kept here so any
        # CLI-local behavior matches the scaffold too.
        env.update({
            "OUROBOROS_REPO_DIR": str(self.repo_dir),
            "OUROBOROS_DATA_DIR": str(self.data_dir),
            "OUROBOROS_SETTINGS_PATH": str(self.settings_path),
            "OUROBOROS_RUNTIME_MODE": "pro",
            "OUROBOROS_MAX_WORKERS": "4",
            "OUROBOROS_SAFETY_MODE": "light",
            "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
            "PYTHONUNBUFFERED": "1",
        })
        if self.model:
            env.update({
                "OUROBOROS_MODEL": self.model,
                "OUROBOROS_MODEL_HEAVY": self.model,
                "OUROBOROS_MODEL_LIGHT": self.model,
                "OUROBOROS_MODEL_FALLBACKS": self.model,
            })

        cmd = [
            self.ouroboros_bin,
            "run",
            "--url",
            self.ouroboros_url,
            "--memory-mode",
            "empty",
            "--quiet",
            *(["--disable-tools", self.disable_tools] if self.disable_tools else []),
            *([ "--attach", screenshot_path ] if screenshot_path else []),
            # E2BIG hygiene (C5): the per-step prompt (a11y tree + history) can be
            # huge; it already lives on disk above, so it travels as a file, never
            # as an argv tail.
            "--prompt-file",
            str(prompt_path),
        ]
        timed_out = False
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.repo_dir),
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
            )
            returncode = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            stderr = (stderr + "\n" if stderr else "") + (
                f"OSWorld adapter: Ouroboros step timed out after {self.timeout_sec}s"
            )
        (self.result_dir / f"ouroboros_step_{step:03d}.stdout.txt").write_text(stdout, encoding="utf-8")
        (self.result_dir / f"ouroboros_step_{step:03d}.stderr.txt").write_text(stderr, encoding="utf-8")

        payload = _json_from_text(stdout.strip())
        response = str(payload.get("response") or stdout.strip() or stderr.strip() or "")
        note = str(payload.get("notes") or "").strip()
        if note:
            self.notes.append(note[:1000])
        raw_actions = payload.get("actions")
        _known_kinds = {"done", "finish", "fail", "wait", "shell", "click", "type", "hotkey", "key", "python"}
        actions = []
        unknown_kinds: list[str] = []
        if isinstance(raw_actions, list):
            for item in raw_actions:
                translated = _normalize_structured_action(item)
                if translated.strip():
                    actions.append(translated)
                elif isinstance(item, dict):
                    k = str(item.get("type") or item.get("action") or "").strip().lower()
                    if k and k not in _known_kinds:
                        unknown_kinds.append(k)
        if unknown_kinds:
            # Feed unknown/dropped action types back to the model (was a silent
            # drop) so it stops re-emitting them and picks a supported action.
            self.notes.append(
                f"[adapter] dropped unsupported action type(s) {sorted(set(unknown_kinds))}; "
                "use only the supported action objects listed in the directive."
            )
        if returncode != 0:
            response = (
                f"Ouroboros step timed out after {self.timeout_sec}s: {response}"
                if timed_out
                else f"ouroboros exited {returncode}: {response}"
            )
            actions = actions or ["WAIT"]
        actions = [action.upper() if action.upper() in SPECIAL_ACTIONS else action for action in actions]
        actions = actions or ["WAIT"]
        if self.screenshot_check_only and "DONE" not in actions and "FAIL" not in actions:
            actions = ["WAIT"]

        # Terminal-message capture (the cu_bridge sample-60 defect: agents that
        # answered "chat-style" left final_answer empty and the run's own
        # objective ledger degraded to not_evaluated). When the agent ends the
        # episode, persist its explicit final_answer — falling back to the
        # terminal response text — so the audit trail always carries the
        # agent's answer even though official scoring stays VM-state-only.
        if response.strip():
            self.last_response = response.strip()
        if "DONE" in actions or "FAIL" in actions:
            self.terminal_action = "FAIL" if "FAIL" in actions else "DONE"
            explicit = str(payload.get("final_answer") or "").strip()
            self.final_answer = explicit or response.strip()

        debug = {
            "step": step,
            "returncode": returncode,
            "timed_out": timed_out,
            "screenshot_upload_path": screenshot_path,
            "screenshot_file": local_screenshot,
            "payload": payload,
            "normalized_actions": actions,
        }
        return response, actions, debug

    def record_action(self, *, action: str, response: str, reward: float, done: bool, info: dict[str, Any]) -> None:
        self.history.append({
            "action": action,
            "response": response,
            "reward": reward,
            "done": done,
            "info": info,
        })
