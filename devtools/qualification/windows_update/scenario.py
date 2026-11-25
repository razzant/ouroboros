"""Packaged Windows desktop qualification without system Git: UIA + owned native input + the app's Git traces.

Exercises candidate A's ``Ouroboros.exe`` on a disposable hosted runner without WebView2 CDP/Playwright,
system-wide auditing (no ``auditpol``/``wevtutil``), screenshots, raw UI dumps, or product changes:
unpacks A's ZIP, builds fast-forward commit B in a local bare mirror, removes host Git, checks the
trace instrument once on the package's own Git, passes the Setup wizard via candidate A's
``WindowsComputerUse`` UIA skill + owned ``SendInput`` fallback, verifies fresh ``Main`` HWND handoff,
runs Turn 1, applies plain ``Check for updates`` -> ``#update-pill`` -> ``Update to ...`` -> single
``Update now``, verifies boot-concluded B + retained Turn 1 + Turn 2 + one attempt carried across both,
then judges the Git traces (``git_traces``).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from typing import Any, Callable

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fixture_model as fm  # noqa: E402
import git_traces as gt  # noqa: E402
import prepare_build as pb  # noqa: E402
import windows_host as wh  # noqa: E402

FIXTURE_DOC = "docs/qualification/windows-update-fixture-b.md"
MAIN_SOURCE_LABEL = "Main Model source"
COMPATIBLE_OPTION_LABEL = "OpenAI-compatible endpoint"
DROPPED_ENV_PREFIXES = (
    "GITHUB_", "RUNNER_", "ACTIONS_", "INPUT_", "STATE_", "PYTHON", "PIP_", "UV_",
    "VIRTUAL_ENV", "GIT_", "OUROBOROS_", "PLAYWRIGHT_", "WEBVIEW2_", "NODE_OPTIONS",
)
DROPPED_ENV_NAMES = {"CI", "pythonLocation", "Python_ROOT_DIR", "Python2_ROOT_DIR", "Python3_ROOT_DIR"}
EXIT = {"passed": 0, "failed": 1, "unconfirmed": 2, "error": 3}
PUBLISHED_FILES = frozenset({
    "result.json", "build-binding.json", "package.json",
    "host-git.json", "git-traces.json", "fixture-model.json",
})
READ_API_PATHS = frozenset({"/api/health", "/api/update/status", "/api/chat/history"})
FOOTER_BUTTON_IDS = ("next-btn", "back-btn", "quick-start-btn", "skip-presets-btn", "check-save-btn")
UIA_REFUSAL_REASONS = ("ambiguous", "unreadable_candidates", "max_nodes", "timeout", "max_depth", "incomplete")


class Outcome(Exception):
    def __init__(self, phase: str, code: str, /, **fields: Any) -> None:
        super().__init__(f"{phase}:{code}")
        self.phase, self.code = phase, code
        self.fields = {k: pb._scalar(v) for k, v in fields.items()}


class Unconfirmed(Outcome):
    """Observation gap, UIA limitation, focus block, or fixture refusal — never product failure."""


class Failure(Outcome):
    """Positive observation of a contrary product outcome."""


def poll_until(timeout_sec: float, predicate: Callable[[], Any], *, interval: float = 0.35) -> Any:
    deadline = time.monotonic() + timeout_sec
    while True:
        if val := predicate():
            return val
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


def assert_privacy_clean(payload: Any, *, tag: str, where: str = "root") -> None:
    try:
        pb.assert_no_host_paths(payload, where=where, forbidden_substrings=(fm.prompt_text(1, tag), fm.prompt_text(2, tag)))
    except pb.Refusal as ref:
        code = "prompt_text_blocked" if "prompt_text" in ref.reason else "path_leak_blocked"
        raise Unconfirmed("privacy_guard", code, **ref.fields) from ref


def parse_hwnd(raw: Any) -> int:
    if isinstance(raw, int):
        return raw
    text = str(raw or "").strip()
    return int(text, 16) if text.lower().startswith("0x") else (int(text) if text.isdigit() else 0)


def flatten_tree_nodes(root: dict[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [root] if isinstance(root, dict) else []
    while stack:
        node = stack.pop()
        out.append(node)
        children = [c for c in (node.get("children") or []) if isinstance(c, dict)]
        stack.extend(reversed(children))
    return out


def visible_rect(node: dict[str, Any]) -> dict[str, Any] | None:
    """The node's rectangle when UIA said ``IsOffscreen == false``; an unread or true flag is not visible."""
    if (node.get("state") or {}).get("offscreen") is not False:
        return None
    rect = node.get("rect")
    if not isinstance(rect, dict) or int(rect.get("width") or 0) <= 0 or int(rect.get("height") or 0) <= 0:
        return None
    center = rect.get("center")
    return rect if isinstance(center, dict) and "x" in center and "y" in center else None


def visible_leaf_texts(root: dict[str, Any] | None) -> list[str]:
    """Extract rendered visible exact leaf ``text`` nodes; excludes ``document``, ``edit``, ``combo_box``, and ``value``."""
    leaves: list[str] = []

    def _walk(node: Any, in_input: bool) -> None:
        if not isinstance(node, dict):
            return
        role = str(node.get("role") or "")
        now_in_input = in_input or role in ("edit", "combo_box")
        if children := (node.get("children") or []):
            for child in children:
                _walk(child, now_in_input)
            return
        if now_in_input or role != "text":
            return
        if node.get("children_unread") or node.get("text_truncated") or "name" in (node.get("unread") or []):
            return
        if visible_rect(node) is not None and isinstance(node.get("name"), str) and node["name"].strip():
            leaves.append(node["name"].strip())

    _walk(root, False)
    return leaves


def has_exact_subsequence(items: list[Any], expected: list[Any]) -> bool:
    rest = iter(items)  # each expected item, in order, among what follows the previous match
    return all(any(item == want for item in rest) for want in expected)


def wizard_error_text(root: dict[str, Any] | None) -> str:
    for node in flatten_tree_nodes(root):
        if str(node.get("class") or "") == "wizard-error":
            if direct := str(node.get("name") or "").strip():
                return direct
            parts = [str(c.get("name") or "").strip() for c in flatten_tree_nodes(node)[1:] if c.get("role") == "text"]
            if joined := " ".join(p for p in parts if p):
                return joined
    return ""


def wizard_footer_top(nodes: list[dict[str, Any]], win_bottom: int) -> int:
    for node in nodes:
        if str(node.get("class") or "") == "wizard-footer" and (r := visible_rect(node)):
            return int(r["top"])
    tops = [int(r["top"]) - 28 for n in nodes if n.get("automation_id") in ("next-btn", "back-btn") and (r := visible_rect(n))]
    return min(tops) if tops else win_bottom


def _build_uia_select(query: dict[str, Any], node: dict[str, Any] | None = None) -> dict[str, Any]:
    sel: dict[str, Any] = {}
    if query.get("automation_id"):
        sel["automation_id"] = str(query["automation_id"])
    if query.get("role"):
        sel["role"] = str(query["role"])
    elif query.get("roles") and node and node.get("role"):
        sel["role"] = str(node["role"])
    if query.get("name"):
        sel["name"] = str(query["name"])
    elif query.get("name_contains") or query.get("name_prefix"):
        sel["name_contains"] = str(query.get("name_contains") or query["name_prefix"])
    return sel


class UIADesktopDriver:
    """Scoped UIA reader/actor over candidate A's ``WindowsComputerUse`` + owned native input."""

    def __init__(self, *, cu: Any, proc: subprocess.Popen[Any], host_mod: Any = wh) -> None:
        self.cu, self.proc, self.wh = cu, proc, host_mod
        self.setup_hwnd = self.main_hwnd = self.update_now_clicks = 0

    @property
    def root_pid(self) -> int:
        return int(self.proc.pid)

    def ensure_alive(self, phase: str) -> None:
        if (rc := self.proc.poll()) is not None:
            raise Failure(phase, "launcher_exited_unexpectedly", exit_code=rc)

    def _owned_ouroboros_windows(self, phase: str) -> list[wh.TopWindow]:
        self.ensure_alive(phase)
        wins = self.wh.list_own_windows(self.root_pid)
        for win in wins:
            if "Setup Required" in win.title:
                raise Failure(phase, "launcher_requires_system_git")
            if "Startup Failed" in win.title:
                raise Failure(phase, "launcher_startup_failed")
        return wins

    def _acquire_window(self, *, phase: str, timeout_sec: float, not_found_code: str, ambiguous_code: str, picker: Callable[[list[wh.TopWindow]], list[wh.TopWindow]]) -> int:
        def _probe() -> int | None:
            wins = self._owned_ouroboros_windows(phase)
            hits = picker(wins)
            if len(hits) > 1:
                raise Unconfirmed(phase, ambiguous_code, count=len(hits))
            return hits[0].hwnd if len(hits) == 1 else None

        hwnd = poll_until(timeout_sec, _probe, interval=0.5)
        if not hwnd:
            self._owned_ouroboros_windows(phase)
            raise Unconfirmed(phase, not_found_code)
        return int(hwnd)

    def acquire_setup_window(self, *, timeout_sec: float = 300.0) -> int:
        def _pick_setup(wins: list[wh.TopWindow]) -> list[wh.TopWindow]:
            setups = [w for w in wins if w.title == "Ouroboros \u2014 Setup" or "Setup" in w.title]
            if not setups and any(w.title.startswith("Ouroboros v") for w in wins):
                raise Failure("wizard", "first_run_wizard_not_shown")
            return setups

        self.setup_hwnd = self._acquire_window(phase="wizard", timeout_sec=timeout_sec, not_found_code="setup_window_not_observed", ambiguous_code="ambiguous_setup_window", picker=_pick_setup)
        return self.setup_hwnd

    def acquire_main_window(self, *, phase: str, exclude_hwnds: set[int] | None = None, timeout_sec: float = 180.0) -> int:
        excluded = exclude_hwnds or set()
        self.main_hwnd = self._acquire_window(
            phase=phase, timeout_sec=timeout_sec, not_found_code="main_window_not_observed", ambiguous_code="ambiguous_main_window",
            picker=lambda wins: [w for w in wins if w.title.startswith("Ouroboros v") and w.hwnd not in excluded],
        )
        return self.main_hwnd

    def transition_setup_to_fresh_main(self, *, close_timeout: float = 90.0, open_timeout: float = 180.0) -> int:
        if not self.setup_hwnd:
            raise Unconfirmed("wizard", "setup_hwnd_missing")
        closed = poll_until(close_timeout, lambda: not self.wh.is_window_open(self.setup_hwnd) and not self.wh.window_owned_by(self.setup_hwnd, self.root_pid), interval=0.35)
        if not closed:
            if wizard_error_text(self.read_tree(self.setup_hwnd, phase="wizard", allow_truncated=True).get("root")):
                raise Failure("wizard", "onboarding_completion_rejected")
            raise Unconfirmed("wizard", "setup_window_did_not_close")
        return self.acquire_main_window(phase="wizard", exclude_hwnds={self.setup_hwnd}, timeout_sec=open_timeout)

    def read_tree(self, hwnd: int, *, phase: str, allow_truncated: bool = False) -> dict[str, Any]:
        self.ensure_alive(phase)
        if not hwnd or not self.wh.window_owned_by(hwnd, self.root_pid):
            raise Unconfirmed(phase, "window_ownership_lost")
        try:
            reading = self.cu.window_read(hwnd=f"0x{hwnd:X}", shape="tree", visible_only=False, max_nodes=4500, max_depth=64, max_text_chars=2500, timeout_ms=10000)
        except Exception as exc:
            raise Unconfirmed(phase, "uia_window_read_exception", error_type=type(exc).__name__) from exc
        if not self.wh.window_owned_by(hwnd, self.root_pid):
            raise Unconfirmed(phase, "window_ownership_lost_after_read")
        if not isinstance(reading, dict) or not reading.get("ok"):
            raise Unconfirmed(phase, "uia_window_read_failed")
        if reading.get("backend") != "uia":
            raise Unconfirmed(phase, "uia_win32_fallback", fallback_reason=reading.get("fallback_reason"))
        win_info = reading.get("window") or {}
        if parse_hwnd(win_info.get("hwnd")) != hwnd or int(win_info.get("pid") or 0) != self.root_pid:
            raise Unconfirmed(phase, "uia_window_identity_mismatch")
        if reading.get("truncated") and not allow_truncated:
            raise Unconfirmed(phase, "uia_tree_truncated", truncated_by=reading.get("truncated_by"))
        return reading

    def _element_act(self, hwnd: int, *, phase: str, **kwargs: Any) -> dict[str, Any]:
        if not hwnd or not self.wh.window_owned_by(hwnd, self.root_pid):
            raise Unconfirmed(phase, "window_ownership_lost_before_act")
        try:
            res = self.cu.element_act(hwnd=f"0x{hwnd:X}", **kwargs)
        except Exception as exc:
            raise Unconfirmed(phase, "uia_element_act_exception", error_type=type(exc).__name__) from exc
        if not isinstance(res, dict):
            raise Unconfirmed(phase, "uia_invalid_act_receipt")
        return res

    def find_nodes(
        self, hwnd: int, *, phase: str, automation_id: str = "", role: str = "",
        roles: set[str] | frozenset[str] | None = None, name: str = "", name_prefix: str = "",
        name_contains: str = "", require_visible: bool = True, allow_truncated: bool = False,
    ) -> list[dict[str, Any]]:
        reading = self.read_tree(hwnd, phase=phase, allow_truncated=allow_truncated)
        matches: list[dict[str, Any]] = []
        for node in flatten_tree_nodes(reading.get("root")):
            unread = set(node.get("unread") or [])
            for req_field, active in (("automation_id", bool(automation_id)), ("role", bool(role or roles is not None)), ("name", bool(name or name_prefix or name_contains))):
                if active and req_field in unread:
                    raise Unconfirmed(phase, "uia_unreadable_candidates", field=req_field)
            node_role, node_name = str(node.get("role") or ""), str(node.get("name") or "")
            if (automation_id and node.get("automation_id") != automation_id) or (role and node_role.lower() != role.lower()):
                continue
            if (roles is not None and node_role not in roles) or (name and node_name.strip() != name):
                continue
            if (name_prefix and not node_name.strip().lower().startswith(name_prefix.lower())) or (name_contains and name_contains.lower() not in node_name.lower()):
                continue
            if require_visible and visible_rect(node) is None:
                continue
            matches.append(node)
        return matches

    def wait_unique_node(self, hwnd: int, *, phase: str, label: str, timeout_sec: float = 30.0, **query: Any) -> dict[str, Any]:
        last_count = 0

        def _look() -> dict[str, Any] | None:
            nonlocal last_count
            found = self.find_nodes(hwnd, phase=phase, **query)
            last_count = len(found)
            if len(found) > 1:
                raise Unconfirmed(phase, "uia_ambiguous_element", control=label, matched=len(found))
            return found[0] if len(found) == 1 else None

        if (node := poll_until(timeout_sec, _look, interval=0.35)) is None:
            raise Unconfirmed(phase, "uia_element_not_found", control=label, matched=last_count)
        return node

    def reveal_wizard_control(self, *, label: str, query: dict[str, Any], select: dict[str, Any] | None = None) -> dict[str, Any]:
        hwnd = self.setup_hwnd
        for step in range(14):
            reading = self.read_tree(hwnd, phase="wizard")
            win_left, win_top, win_right, win_bottom = self.wh.window_rect(hwnd)
            f_top = wizard_footer_top(flatten_tree_nodes(reading.get("root")), win_bottom)
            matches = self.find_nodes(hwnd, phase="wizard", require_visible=False, **query)
            if len(matches) > 1:
                raise Unconfirmed("wizard", "uia_ambiguous_element", control=label, matched=len(matches))
            if len(matches) == 1:
                node, rect = matches[0], visible_rect(matches[0])
                if rect is not None and (node.get("automation_id") in FOOTER_BUTTON_IDS or (int(rect["top"]) >= win_top + 40 and int(rect["bottom"]) <= f_top - 8)):
                    return node
                if step == 0:
                    res = self._element_act(hwnd, phase="wizard", select=select or _build_uia_select(query, node), do="scroll_into_view", timeout_ms=2500)
                    if res.get("reason") in ("ambiguous", "unreadable_candidates"):
                        raise Unconfirmed("wizard", f"uia_scroll_{res['reason']}", control=label)
                    if res.get("ok"):
                        time.sleep(0.18)
                        continue
                direction = 2 if rect is None or int(rect["bottom"]) > f_top - 8 else -2
            else:
                direction = 2
            self.wh.wheel(int(win_left + (win_right - win_left) * 0.62), int(win_top + max(80, min((f_top - win_top) // 2, 320))), hwnd, self.root_pid, direction)
        raise Unconfirmed("wizard", "wizard_control_obscured_by_sticky_footer", control=label)

    def act_or_click(
        self, hwnd: int, *, phase: str, label: str, query: dict[str, Any], select: dict[str, Any] | None = None,
        verbs: tuple[str, ...] = ("invoke",), wizard: bool = False,
        post_condition: Callable[[], bool] | None = None, single_press_strict: bool = False,
        post_timeout: float = 8.0,
    ) -> None:
        node = (self.reveal_wizard_control(label=label, query=query, select=select) if wizard
                else self.wait_unique_node(hwnd, phase=phase, label=label, **query))
        if (node.get("state") or {}).get("enabled") is False:
            raise Unconfirmed(phase, "uia_control_disabled", control=label)
        eff_sel = select or _build_uia_select(query, node)

        for verb in verbs:
            res = self._element_act(hwnd, phase=phase, select=eff_sel, do=verb, timeout_ms=3500)
            if res.get("ok"):
                self.update_now_clicks += int(label == "Update now")
                if post_condition is not None and not poll_until(post_timeout, post_condition, interval=0.25):
                    raise Unconfirmed(phase, "uia_action_postcondition_unmet", control=label, verb=verb)
                return
            if (reason := str(res.get("reason") or "")) in UIA_REFUSAL_REASONS:
                raise Unconfirmed(phase, f"uia_act_{reason}", control=label)
            if res.get("action_outcome") == "unknown":
                self.update_now_clicks += int(label == "Update now")
                if post_condition is not None and poll_until(post_timeout, post_condition, interval=0.25):
                    return
                raise Unconfirmed(phase, "uia_action_outcome_unknown", control=label, verb=verb)
            if "does not support" not in str(res.get("error") or ""):
                if single_press_strict:
                    raise Unconfirmed(phase, "uia_act_refused", control=label)
                break

        if wizard:
            fresh = self.reveal_wizard_control(label=label, query=query, select=eff_sel)
        elif visible_rect(fresh := self.wait_unique_node(hwnd, phase=phase, label=label, timeout_sec=5.0, **query)) is None:
            self._element_act(hwnd, phase=phase, select=eff_sel, do="scroll_into_view", timeout_ms=2500)
            fresh = self.wait_unique_node(hwnd, phase=phase, label=label, timeout_sec=5.0, **{**query, "require_visible": True})
        if (rect := visible_rect(fresh)) is None:
            raise Unconfirmed(phase, "control_not_visible_for_native_click", control=label)
        self.wh.click(int(rect["center"]["x"]), int(rect["center"]["y"]), hwnd, self.root_pid)
        self.update_now_clicks += int(label == "Update now")
        if post_condition is not None and not poll_until(post_timeout, post_condition, interval=0.25):
            raise Unconfirmed(phase, "native_click_postcondition_unmet", control=label)

    def set_exact_text(
        self, hwnd: int, *, phase: str, automation_id: str, expected: str,
        wizard: bool = False, ready_check: Callable[[], bool] | None = None,
    ) -> None:
        sel = {"automation_id": automation_id}
        node = self.reveal_wizard_control(label=automation_id, query=sel, select=sel) if wizard else self.wait_unique_node(hwnd, phase=phase, label=automation_id, **sel)

        def _exact_readback() -> bool:
            found = self.find_nodes(hwnd, phase=phase, automation_id=automation_id, require_visible=False)
            if len(found) != 1:
                return False
            if found[0].get("text_truncated") or "value" in (found[0].get("unread") or []):
                raise Unconfirmed(phase, "uia_value_readback_truncated_or_unread", field=automation_id)
            return str(found[0].get("value") or "") == expected

        res = self._element_act(hwnd, phase=phase, select=sel, do="set_value", text=expected, timeout_ms=3500)
        if res.get("reason") in UIA_REFUSAL_REASONS:
            raise Unconfirmed(phase, f"uia_set_value_{res['reason']}", field=automation_id)
        if res.get("action_outcome") == "unknown" and not poll_until(2.0, _exact_readback, interval=0.2):
            raise Unconfirmed(phase, "uia_set_value_outcome_unknown", field=automation_id)
        if poll_until(1.5, _exact_readback, interval=0.2) and (ready_check is None or poll_until(2.0, ready_check, interval=0.25)):
            return

        if not self._element_act(hwnd, phase=phase, select=sel, do="focus", timeout_ms=2500).get("ok"):
            if (rect := visible_rect(node)) is None:
                raise Unconfirmed(phase, "input_not_visible_for_focus", field=automation_id)
            self.wh.click(int(rect["center"]["x"]), int(rect["center"]["y"]), hwnd, self.root_pid)
        if not poll_until(4.0, lambda: len(f := self.find_nodes(hwnd, phase=phase, automation_id=automation_id, require_visible=False)) == 1 and (f[0].get("state") or {}).get("focused") is True, interval=0.2):
            raise Unconfirmed(phase, "input_focus_not_confirmed", field=automation_id)
        self.wh.select_all_and_replace_text(expected, hwnd, self.root_pid)
        if not poll_until(3.0, _exact_readback, interval=0.2):
            raise Unconfirmed(phase, "field_exact_readback_mismatch", field=automation_id)
        if ready_check is not None and not poll_until(4.0, ready_check, interval=0.25):
            raise Unconfirmed(phase, "field_ready_state_not_confirmed", field=automation_id)


class QualificationRun:
    def __init__(
        self, args: argparse.Namespace, *, uia_factory: Callable[[pathlib.Path], Any] | None = None,
        host_mod: Any = wh, launcher_spawn: Callable[..., subprocess.Popen[Any]] | None = None,
        app_root_override: pathlib.Path | None = None,
    ) -> None:
        self.args, self.candidate, self.tag, self.wh = args, args.candidate.strip().lower(), args.tag, host_mod
        self._uia_factory, self._launcher_spawn = uia_factory, launcher_spawn
        self.evidence = pathlib.Path(args.evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.app_root = app_root_override or (pathlib.Path.home() / "Ouroboros")
        self.data, self.mirror = self.app_root / "data", pathlib.Path(args.mirror_url)
        self.qa_root = self.mirror.parent
        self.extract_dir, self.control_dir = self.qa_root / "unpacked-app", self.qa_root / "git-control"
        self.trace1, self.trace2_dir = self.qa_root / "git-trace1.log", self.qa_root / "git-trace2"
        self.result: dict[str, Any] = {
            "schema": 5, "candidate_sha": self.candidate, "run_tag": pb._scalar(self.tag),
            "verdict": "error", "phase": "init", "code": "not_finished",
            "reasons": [], "marks": {}, "checks": {}, "facts": {}, "declared": dict(pb.DECLARED),
        }
        self._decided = False
        self.step = "init"  # where the run is; the published phase is the first settled outcome's own
        self.proc: subprocess.Popen[Any] | None = None
        self.driver: UIADesktopDriver | None = None
        self.model: fm.FixtureModel | None = None
        self.gits: dict[str, str] = {}
        self.b_sha = self.marker = self.init_id = ""
        self.baseline_ops: set[str] = set()
        self.attempt: dict[str, Any] = {}  # the A->B attempt as the boot concluded it, before the second chat
        self.control_ok = False

    def mark(self, name: str) -> None:
        self.result["marks"][name] = pb._utc_now()

    def check(self, name: str, ok: bool, *, phase: str, failure: str = "", **detail: Any) -> None:
        row: dict[str, Any] = {"ok": bool(ok), **({"detail": {k: pb._scalar(v) for k, v in detail.items()}} if detail else {})}
        self.result["checks"][name] = row
        if not ok and failure:
            raise Failure(phase, failure, check=name, **detail)

    def save(self, name: str, payload: dict[str, Any]) -> None:
        if name not in PUBLISHED_FILES:
            raise RuntimeError(f"unlisted evidence file: {name}")
        assert_privacy_clean(payload, tag=self.tag, where=name)
        pb.write_json_atomic(self.evidence / name, payload, forbidden_substrings=(fm.prompt_text(1, self.tag), fm.prompt_text(2, self.tag)))

    def api(self, path: str, *, timeout: float = 10.0) -> dict[str, Any] | None:
        """The JSON object a passive local GET answers with 200 (``/api/update/status`` never fetches); else None."""
        if path.split("?", 1)[0] not in READ_API_PATHS:
            raise RuntimeError(f"disallowed API read path: {path}")
        try:
            raw_port = (self.data / "state" / "server_port").read_text(encoding="utf-8").strip()
            port = int(raw_port) if raw_port.isdigit() else int(json.loads(raw_port).get("port"))
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8")) if resp.status == 200 else None
        except Exception:
            return None
        return body if isinstance(body, dict) else None

    def durable_history(self) -> tuple[bool, bool | None, list[tuple[str, str]]]:
        body = self.api("/api/chat/history?limit=100") or {}
        comp = body["window"].get("complete") if isinstance(body.get("window"), dict) else None
        if not isinstance(body.get("messages"), list):
            return False, comp if isinstance(comp, bool) else None, []
        rows = [(str(m.get("role") or ""), str(m.get("text") or "").strip()) for m in body["messages"] if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        return True, comp if isinstance(comp, bool) else None, rows

    def state_doc(self) -> dict[str, Any] | None:
        return doc if isinstance(doc := pb.read_json_retry(self.data / "state" / "state.json"), dict) else None

    def initialization_id(self) -> str:
        return str((pb.read_json_retry(self.data / "state" / "state.initialized.json") or {}).get("initialization_id") or "")

    def _load_uia_skill(self, candidate_src: pathlib.Path) -> Any:
        if self._uia_factory is not None:
            return self._uia_factory(candidate_src)
        for entry in (candidate_src / "skills", candidate_src):
            if str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        from windows_computer_use.plugin import WindowsComputerUse

        return WindowsComputerUse(api=None)

    def _build_launch_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in DROPPED_ENV_NAMES and not k.upper().startswith(DROPPED_ENV_PREFIXES)}
        reg = self.wh.registry_path_dirs()
        return {**env, "PATH": os.pathsep.join(reg["machine"] + reg["user"])}

    def _probe_host_git(self, candidate_src: pathlib.Path, extra_tools: list[str] | None = None) -> tuple[list[str], dict[str, Any]]:
        tools = self.wh.candidate_tool_dirs(candidate_src, sys.executable, self.launch_env)
        dirs = self.wh.lookup_dirs(self.launch_env["PATH"].split(os.pathsep), tools + (extra_tools or []), self.exe.parent)
        return tools, self.wh.git_lookup_report(dirs, self.package_root, self.launch_env, self.exe.parent)

    def prepare_host_and_package(self) -> None:
        self.step = "host_check"
        if not self.wh.is_windows() or not self.wh.disposable_hosted_runner():
            raise Unconfirmed("host_check", "not_windows" if not self.wh.is_windows() else "not_a_disposable_hosted_runner")
        self.result["facts"].update(host_image=self.wh.host_image(), dpi_awareness=self.wh.make_dpi_aware())
        if not (screen := self.wh.screen_info()).get("foreground_present"):
            raise Unconfirmed("host_check", "no_interactive_desktop")
        if not (wv2 := self.wh.webview2_runtime()):
            raise Unconfirmed("host_check", "webview2_runtime_absent")
        self.result["facts"].update(screen=screen, webview2_runtime=wv2)
        for occupied in (self.app_root, self.mirror, self.extract_dir, self.control_dir, self.trace1, self.trace2_dir):
            if occupied.exists():
                raise Unconfirmed("host_check", "profile_not_fresh", target=occupied.name)

        self.step = "unpack_package"
        archive = pathlib.Path(self.args.archive)
        if self.wh.sha256_file(archive) != self.args.archive_sha256.lower():
            raise Unconfirmed("unpack_package", "archive_digest_mismatch")
        self.package_root, manifest, self.gits = pb.unpack_candidate_archive(archive, self.extract_dir)
        self.exe = self.package_root / "Ouroboros.exe"
        self.bundle = pathlib.Path(manifest["_bundle_path"]) if manifest.get("_bundle_path") else None
        self.release_tag = str(manifest.get("release_tag") or "")
        pkg_ok = manifest.get("source_sha") == self.candidate and manifest.get("managed_remote_url") == self.args.mirror_url
        self.check("package_matches_candidate", pkg_ok, phase="unpack_package", failure="package_not_candidate", source_sha=manifest.get("source_sha"))
        mains = sorted(rel for rel in self.gits if pathlib.PurePosixPath(rel).name.lower() == "git.exe")
        self.check("package_carries_git_standalone", bool(mains) and self.exe.is_file(), phase="unpack_package", failure="package_lacks_git_standalone", git_exe_entries=mains)

    def prepare_fixture_mirror_and_remove_host_git(self) -> None:
        self.step = "prepare_mirror"
        mingit = [self.package_root / pathlib.Path(*rel.split("/")) for rel in self.gits if rel.lower().endswith("cmd/git.exe")]
        harness_git = shutil.which("git") or (str(mingit[0]) if mingit else "")
        if not harness_git or self.bundle is None:
            raise Unconfirmed("prepare_mirror", "no_git_for_fixture_preparation")
        self.marker = f"OUROBOROS-QA-FIXTURE-B-{fm._clean_tag(self.tag)}"
        try:
            self.b_sha, tags, parent_sha = pb.create_fixture_mirror_b(
                harness_git=harness_git, mirror=self.mirror, qa_root=self.qa_root,
                bundle=self.bundle, candidate=self.candidate, doc_rel_path=FIXTURE_DOC, marker=self.marker,
            )
        except pb.Refusal as ref:
            raise Unconfirmed("prepare_mirror", ref.reason, **ref.fields) from ref
        self.check("mirror_tags_isolated", tags == [f"refs/tags/{self.release_tag}"] and bool(self.release_tag), phase="prepare_mirror", failure="mirror_tags_not_isolated", tags=tags)
        self.check("fixture_b_is_fast_forward", parent_sha == self.candidate, phase="prepare_mirror", failure="fixture_b_not_child_of_a")
        self.result["facts"]["fixture"] = {"a": self.candidate, "b": self.b_sha, "doc": FIXTURE_DOC, "local_bare_mirror": True}

        self.step = "remove_system_git"
        candidate_src = pathlib.Path(self.args.candidate_src)
        self.launch_env = self._build_launch_env()
        tools_before, before = self._probe_host_git(candidate_src)
        host_git_record: dict[str, Any] = {"before": before, "uninstall": None}
        self.save("host-git.json", host_git_record)
        if before["unreadable_dir_count"]:
            raise Unconfirmed("remove_system_git", "system_git_lookup_folder_unreadable", unreadable=before["unreadable_dir_count"])
        if before["absent"]:
            condition = "absent_on_disposable_hosted_runner"
        elif (installation := self.wh.runner_git_installation()) is None:
            raise Unconfirmed("remove_system_git", "unexpected_foreign_git_installation", git_hits=before["git_hit_count"])
        else:
            uninstall = host_git_record["uninstall"] = self.wh.uninstall_runner_git(installation)
            self.save("host-git.json", host_git_record)
            if not uninstall["done"]:
                raise Unconfirmed("remove_system_git", "runner_git_not_uninstalled", problem=uninstall["problem"])
            condition = "uninstalled_on_disposable_hosted_runner"
            self.launch_env = self._build_launch_env()

        _tools_after, after = self._probe_host_git(candidate_src, tools_before)
        host_git_record.update(after=after, condition=condition)
        self.save("host-git.json", host_git_record)
        self.result["facts"]["system_git_condition"] = condition
        if after["unreadable_dir_count"] or not after["absent"]:
            code = "system_git_lookup_folder_unreadable" if after["unreadable_dir_count"] else "system_git_still_reachable"
            raise Unconfirmed("remove_system_git", code, count=after["unreadable_dir_count"] or after["git_hit_count"])
        self.check("system_git_absent_on_lookup_paths", True, phase="remove_system_git", condition=condition)

    def _package(self) -> dict[str, Any]:
        """Where ``git_traces.identity`` looks: the package, its bundled Git, the ZIP's digests, the disk's."""
        return {"package_root": str(self.package_root), "package_exes": self.gits,
                "git_root": str(self.package_root / "_internal" / "git-standalone"),
                "folder_exes": lambda rel: self.wh.exe_digests(self.package_root, rel)}

    def control_git_identity(self) -> None:
        """The instrument's positive control on this very build: the package's own ``git version`` must leave one
        identity line inside the bundled Git and one complete ``version`` session. A miss is recorded, not a refusal:
        the run goes on and its Git traces then settle as unconfirmed."""
        self.step = "git_identity_control"
        entry = self.package_root / "_internal" / "git-standalone" / "cmd" / "git.exe"  # None when it does not run
        code = self.wh.traced_git_version(entry, self.control_dir, self.launch_env)
        try:
            text, texts, discarded = self.wh.read_trace_sinks(self.control_dir / "trace1.log", self.control_dir / "trace2")
        except wh.Unread as unread:  # the control is unconfirmed, the scenario goes on
            text, texts, discarded = "", [], False
            self.result["facts"]["git_identity_control_unread"] = unread.code
        found = gt.sessions(texts, discarded=discarded)
        claim = gt.identity(gt.identity_lines(text), found, **self._package())
        self.control_ok = code == 0 and gt.control_passed(claim, found)
        self.result["facts"]["git_identity_control"] = {"exit_code": code, **claim}
        self.check("git_identity_control_passed", self.control_ok, phase="git_identity_control")

    def launch_and_pass_wizard(self) -> None:
        assert self.model is not None
        self.step = "wizard"
        cu = self._load_uia_skill(pathlib.Path(self.args.candidate_src))
        self.trace2_dir.mkdir(parents=True, exist_ok=True)
        app_env = {**self.launch_env, **self.wh.trace_env(self.trace1, self.trace2_dir)}
        self.mark("launched")
        spawn = self._launcher_spawn or (
            lambda cmd, **kw: subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x00000200 if self.wh.is_windows() else 0, **kw)
        )
        self.proc = spawn([str(self.exe)], cwd=str(self.exe.parent), env=app_env)
        self.driver = UIADesktopDriver(cu=cu, proc=self.proc, host_mod=self.wh)
        setup_hwnd = self.driver.acquire_setup_window()
        self.mark("setup_window_ready")
        self.driver.wait_unique_node(setup_hwnd, phase="wizard", label="next-btn", automation_id="next-btn", require_visible=False, timeout_sec=60.0)

        if not self.driver.find_nodes(setup_hwnd, phase="wizard", name_prefix="Custom OpenAI-compatible servers", require_visible=False):
            self.driver.act_or_click(
                setup_hwnd, phase="wizard", label="API keys and local models", verbs=("expand", "invoke"), wizard=True,
                query={"role": "button", "name_prefix": "API keys and local models"},
                post_condition=lambda: bool(self.driver.find_nodes(
                    setup_hwnd, phase="wizard", name_prefix="Custom OpenAI-compatible servers", require_visible=False)))
        if not self.driver.find_nodes(setup_hwnd, phase="wizard", automation_id="s-openai-compatible-base-url", require_visible=True):
            self.driver.act_or_click(setup_hwnd, phase="wizard", label="Custom OpenAI-compatible servers", query={"role": "button", "name_prefix": "Custom OpenAI-compatible servers"}, verbs=("expand", "invoke"), wizard=True, post_condition=lambda: bool(self.driver.find_nodes(setup_hwnd, phase="wizard", automation_id="s-openai-compatible-base-url", require_visible=False)))

        _next_enabled = lambda: len(b := self.driver.find_nodes(setup_hwnd, phase="wizard", automation_id="next-btn", require_visible=False)) == 1 and (b[0].get("state") or {}).get("enabled") is not False  # noqa: E731
        self.driver.set_exact_text(setup_hwnd, phase="wizard", automation_id="s-openai-compatible-base-url", expected=self.model.base_url, wizard=True, ready_check=_next_enabled)
        self.driver.act_or_click(setup_hwnd, phase="wizard", label="Accounts Continue", query={"automation_id": "next-btn"}, wizard=True, post_condition=lambda: bool(self.driver.find_nodes(setup_hwnd, phase="wizard", role="combo_box", name=MAIN_SOURCE_LABEL, require_visible=False)))

        self._select_main_model_source_option(setup_hwnd)
        self.driver.set_exact_text(setup_hwnd, phase="wizard", automation_id="main-model", expected=fm.MODEL_ID, wizard=True, ready_check=_next_enabled)
        self.driver.act_or_click(setup_hwnd, phase="wizard", label="Models Continue", query={"automation_id": "next-btn"}, wizard=True, post_condition=lambda: not self.driver.find_nodes(setup_hwnd, phase="wizard", automation_id="main-model", require_visible=False))

        # Keep all remaining wizard step defaults (including Safety / Access level) unchanged until Summary -> Start Ouroboros.
        for _ in range(8):
            btn = self.driver.wait_unique_node(setup_hwnd, phase="wizard", label="next-btn", automation_id="next-btn", require_visible=False, timeout_sec=30.0)
            label = str(btn.get("name") or "").strip()
            if label == "Start Ouroboros":
                break
            if label != "Continue":
                raise Unconfirmed("wizard", "unexpected_wizard_footer_button", label=label)
            step_groups = self.driver.find_nodes(setup_hwnd, phase="wizard", role="group", name_prefix="Step ", require_visible=False)
            prev_step = str(step_groups[0].get("name") or "") if step_groups else ""
            self.driver.act_or_click(setup_hwnd, phase="wizard", label=f"Wizard Continue ({prev_step or 'step'})", query={"automation_id": "next-btn"}, wizard=True, post_condition=lambda p=prev_step: self._wizard_step_changed(setup_hwnd, p))
        else:
            raise Unconfirmed("wizard", "wizard_summary_not_reached")

        self.driver.wait_unique_node(
            setup_hwnd, phase="wizard", label="Summary Main source and model", role="text",
            name=f"{COMPATIBLE_OPTION_LABEL} · {fm.MODEL_ID}", timeout_sec=8.0,
        )
        self.check("main_model_source_selected_via_option_row", True, phase="wizard", witness="summary_main_binding")
        self._invoke_start_ouroboros(setup_hwnd)
        main_hwnd = self.driver.transition_setup_to_fresh_main()
        self.mark("main_window_ready")
        self.check("setup_closed_and_fresh_main_hwnd_opened", bool(main_hwnd and main_hwnd != setup_hwnd), phase="wizard", failure="fresh_main_window_not_opened")

    def _select_main_model_source_option(self, setup_hwnd: int) -> None:
        assert self.driver is not None
        self.driver.act_or_click(
            setup_hwnd, phase="wizard", label=MAIN_SOURCE_LABEL, query={"role": "combo_box", "name": MAIN_SOURCE_LABEL}, verbs=("expand", "invoke"), wizard=True,
            post_condition=lambda: bool(self.driver.find_nodes(setup_hwnd, phase="wizard", roles={"list_item", "menu_item"}, name=COMPATIBLE_OPTION_LABEL, require_visible=True)),
        )
        option = {"roles": {"list_item", "menu_item"}, "name": COMPATIBLE_OPTION_LABEL}
        # Dismissal only clears the menu for typing; Summary proves the actual choice.
        self.driver.act_or_click(
            setup_hwnd, phase="wizard", label=COMPATIBLE_OPTION_LABEL, query=option, verbs=(),
            post_condition=lambda: not self.driver.find_nodes(setup_hwnd, phase="wizard", require_visible=False, **option),
        )

    def _wizard_step_changed(self, setup_hwnd: int, prev_step: str) -> bool:
        assert self.driver is not None
        if wizard_error_text(self.driver.read_tree(setup_hwnd, phase="wizard").get("root")):
            raise Failure("wizard", "wizard_validation_refused")
        groups = self.driver.find_nodes(setup_hwnd, phase="wizard", role="group", name_prefix="Step ", require_visible=False)
        cur_step = str(groups[0].get("name") or "") if groups else ""
        if prev_step and cur_step and cur_step != prev_step:
            return True
        btns = self.driver.find_nodes(setup_hwnd, phase="wizard", automation_id="next-btn", require_visible=False)
        return len(btns) == 1 and str(btns[0].get("name") or "").strip() == "Start Ouroboros"

    def _invoke_start_ouroboros(self, setup_hwnd: int) -> None:
        assert self.driver is not None
        deadline = time.monotonic() + 35.0
        while time.monotonic() < deadline:
            if not self.wh.is_window_open(setup_hwnd):
                return
            btns = self.driver.find_nodes(setup_hwnd, phase="wizard", automation_id="next-btn", require_visible=False)
            if len(btns) == 1 and (btns[0].get("state") or {}).get("enabled") is not False:
                self.driver.act_or_click(setup_hwnd, phase="wizard", label="Start Ouroboros", query={"automation_id": "next-btn"}, wizard=True)
            if poll_until(3.0, lambda: not self.wh.is_window_open(setup_hwnd), interval=0.25):
                return
            err = wizard_error_text(self.driver.read_tree(setup_hwnd, phase="wizard", allow_truncated=True).get("root"))
            if "still updating from your latest setup choices" in err:
                time.sleep(0.5)
                continue
            if err:
                raise Failure("wizard", "onboarding_save_error")
            return
        raise Unconfirmed("wizard", "start_ouroboros_not_accepted")

    def ensure_main_chat_open(self, *, phase: str, click_sidebar: bool) -> None:
        assert self.driver is not None
        hwnd = self.driver.main_hwnd
        nav = self.driver.wait_unique_node(hwnd, phase=phase, label="Ouroboros Main Chat", role="button", name_contains="Main Chat", timeout_sec=45.0)
        nav_label = " ".join(str(nav.get("name") or "").split())
        if not (nav_label.startswith("Ouroboros") and "Main Chat" in nav_label):
            raise Failure(phase, "unexpected_main_chat_sidebar_label")
        if click_sidebar:
            self.driver.act_or_click(hwnd, phase=phase, label="Ouroboros Main Chat", query={"role": "button", "name_contains": "Main Chat"}, post_condition=lambda: bool(self.driver.find_nodes(hwnd, phase=phase, automation_id="chat-input")))
        else:
            self.driver.wait_unique_node(hwnd, phase=phase, label="chat-input", automation_id="chat-input", timeout_sec=45.0)

    def _observe_chat_state(self, hwnd: int, *, phase: str, expected_durable: list[tuple[str, str]], expected_leaves: list[str], state: dict[str, Any]) -> None:
        assert self.driver is not None
        read_ok, _complete, dur_rows = self.durable_history()
        if read_ok:
            state["durable"] = has_exact_subsequence(dur_rows, expected_durable)
        try:
            reading = self.driver.read_tree(hwnd, phase=phase)
            state["visible_leaf"] = has_exact_subsequence(visible_leaf_texts(reading.get("root")), expected_leaves)
            state["uia_gap"] = ""
        except Unconfirmed as gap:
            state["uia_gap"] = gap.code

    def send_chat_turn(self, slot: int, *, expected_durable: list[tuple[str, str]], expected_visible_leaves: list[str], phase: str) -> None:
        assert self.driver is not None and self.model is not None
        hwnd = self.driver.main_hwnd
        self.driver.set_exact_text(hwnd, phase=phase, automation_id="chat-input", expected=fm.prompt_text(slot, self.tag), wizard=False)
        self.driver.wait_unique_node(hwnd, phase=phase, label="chat-send", automation_id="chat-send", timeout_sec=15.0)

        cursor_before = self.model.cursor()
        self.mark(f"prompt_{slot}_before_click")
        self.driver.act_or_click(hwnd, phase=phase, label=f"Send prompt {slot}", query={"automation_id": "chat-send"})
        settled: dict[str, Any] = {"answered": False, "durable": False, "visible_leaf": False, "uia_gap": ""}

        def _poll_turn() -> bool:
            assert self.driver is not None and self.model is not None
            self.driver.ensure_alive(phase)
            for ev in self.model.events_since(cursor_before):  # a refusal of another request (slot None) waits
                if ev["kind"] == "completion" and ev["slot"] == slot and ev["status"] == 200:
                    settled["answered"] = True
                elif ev["kind"] == "completion" and ev["slot"] == slot:
                    raise Unconfirmed(phase, "fixture_model_refused_owner_turn", reason=ev["reason"])
            self._observe_chat_state(hwnd, phase=phase, expected_durable=expected_durable, expected_leaves=expected_visible_leaves, state=settled)
            return bool(settled["answered"] and settled["durable"] and settled["visible_leaf"])

        if poll_until(180.0, _poll_turn, interval=0.5):
            self.mark(f"reply_{slot}_verified")
            self.check(f"turn_{slot}_durable_history_exact", True, phase=phase)
            self.check(f"turn_{slot}_visible_leaf_reply_confirmed", True, phase=phase)
            return
        if not settled["answered"]:
            if refusals := [ev["reason"] for ev in self.model.events_since(cursor_before) if ev["kind"] == "completion" and ev["status"] != 200]:
                raise Unconfirmed(phase, "prompt_unanswered_while_fixture_refused_service_calls", reasons=sorted(set(refusals)))
            read_ok, _complete, rows = self.durable_history()
            if not read_ok or ("user", fm.prompt_text(slot, self.tag)) not in rows:  # Send never took: a harness gap
                raise Unconfirmed(phase, "durable_history_unreadable" if not read_ok else "owner_message_not_accepted", slot=slot)
            raise Failure(phase, "prompt_never_reached_model", slot=slot)
        read_ok, complete, _ = self.durable_history()
        if not read_ok or (complete is False and not settled["durable"]):
            raise Unconfirmed(phase, "durable_history_unreadable" if not read_ok else "durable_history_window_incomplete", slot=slot)
        self.check(f"turn_{slot}_durable_history_exact", settled["durable"], phase=phase, failure="exchange_missing_from_durable_history")
        gap_code = settled["uia_gap"] or "leaf_text_not_observed"
        self.check(f"turn_{slot}_visible_leaf_reply_confirmed", False, phase=phase, uia_gap=gap_code)
        raise Unconfirmed(phase, "chat_visible_leaf_text_unconfirmed", slot=slot, durable_history_exact=True, uia_gap=gap_code)

    def run_first_chat_and_check_a(self) -> None:
        self.step = "turn_1"
        self.ensure_main_chat_open(phase="turn_1", click_sidebar=False)
        p1, r1 = fm.prompt_text(1, self.tag), fm.reply_text(1, self.tag)
        self.send_chat_turn(1, expected_durable=[("user", p1), ("assistant", r1)], expected_visible_leaves=[p1, r1], phase="turn_1")
        self.init_id = self.initialization_id()
        if not self.init_id:
            raise Unconfirmed("turn_1", "initialization_id_unreadable")
        self.check("initialization_id_recorded", True, phase="turn_1")

        body, state, head = self.api("/api/update/status"), self.state_doc(), pb.read_git_head(self.app_root / "repo" / ".git")
        if not body or not state or not head or not body.get("running_sha") or not body.get("current_sha") or not state.get("current_sha"):
            raise Unconfirmed("turn_1", "revision_unreadable_before_update")
        revs = {head, str(state.get("current_sha")), str(body.get("running_sha")), str(body.get("current_sha"))}
        self.check("running_on_a_before_update", revs == {self.candidate}, phase="turn_1", failure="not_running_a_before_update")

    def run_plain_check_and_update_a_to_b(self) -> None:
        assert self.driver is not None
        self.step = "update_check"
        hwnd = self.driver.main_hwnd
        checked = lambda **want: gt.completed_check(self.api("/api/update/status"), **want)  # noqa: E731
        # The boot's own fetching check (server._boot_managed_update_tasks) settles on Stable's A before the channel moves.
        if poll_until(180.0, lambda: checked(target=self.candidate, development=False), interval=1.0) is None:
            raise Unconfirmed("update_check", "startup_stable_check_unobserved")

        # Dashboard opens on Logs (hundreds of rows): only positive lookups there, read from a cut tree if need be.
        tab = {"automation_id": "dashboard-tab-updates", "allow_truncated": True}
        dev = {"role": "button", "name_prefix": "Development", "require_visible": False, "allow_truncated": True}
        saved = lambda: (pb.read_json_retry(self.data / "settings.json") or {}).get("OUROBOROS_UPDATE_CHANNEL") == "development"  # noqa: E731
        self.driver.act_or_click(hwnd, phase="update_check", label="Dashboard", query={"role": "button", "name": "Dashboard"},
                                 post_condition=lambda: bool(self.driver.find_nodes(hwnd, phase="update_check", **tab)))
        self.driver.act_or_click(hwnd, phase="update_check", label="Dashboard > Updates", query=tab, verbs=("select", "invoke"),
                                 post_condition=lambda: bool(self.driver.find_nodes(hwnd, phase="update_check", **dev)))
        self.driver.act_or_click(hwnd, phase="update_check", label="Which updates > Development", query=dev,
                                 verbs=("toggle", "invoke"), post_condition=saved)
        self.check("development_channel_saved", True, phase="update_check")
        self.ensure_main_chat_open(phase="update_check", click_sidebar=True)
        if self.driver.find_nodes(hwnd, phase="update_check", automation_id="update-pill"):  # it hides the plain check
            raise Unconfirmed("update_check", "update_pill_visible_before_plain_check")

        # Sidebar plain Check for updates (#nav-version-check) -> #update-pill (NEVER 'Check and describe').
        # The check window of the Git traces runs from this click to the "Update to" click, whose preflight is later.
        self.driver.wait_unique_node(hwnd, phase="update_check", label="nav-version-check", automation_id="nav-version-check", name_prefix="Check for updates", timeout_sec=30.0)
        self.mark("plain_check_clicked")
        self.driver.act_or_click(
            hwnd, phase="update_check", label="Sidebar Check for updates", select={"automation_id": "nav-version-check"},
            query={"automation_id": "nav-version-check", "name_prefix": "Check for updates"}, post_timeout=120.0,
            post_condition=lambda: bool(self.driver.find_nodes(hwnd, phase="update_check", automation_id="update-pill")))
        clicked = self.result["marks"]["plain_check_clicked"]
        if poll_until(120.0, lambda: checked(target=self.b_sha, development=True, after=clicked), interval=1.0) is None:
            raise Unconfirmed("update_check", "plain_check_of_b_unobserved")
        self.driver.act_or_click(hwnd, phase="update_check", label="Sidebar update pill", query={"automation_id": "update-pill"}, select={"automation_id": "update-pill"}, post_condition=lambda: bool(self.driver.find_nodes(hwnd, phase="update_check", automation_id="btn-update-primary")))
        self.check("plain_update_check_used_without_describe", True, phase="update_check")

        if not poll_until(45.0, lambda: len(b := self.driver.find_nodes(hwnd, phase="update_check", automation_id="btn-update-primary")) == 1 and str(b[0].get("name") or "").startswith("Update to ") and (b[0].get("state") or {}).get("enabled") is not False, interval=0.35):
            btns = self.driver.find_nodes(hwnd, phase="update_check", automation_id="btn-update-primary")
            raise (Unconfirmed("update_check", "update_primary_button_unobserved", count=len(btns)) if len(btns) != 1 else Failure("update_check", "update_to_b_not_offered"))
        self.mark("update_to_clicked")
        self.driver.act_or_click(  # the dialog opens only after the server's preflight (a fetch) returns
            hwnd, phase="update_check", label="Update to B", query={"automation_id": "btn-update-primary", "name_prefix": "Update to "}, select={"automation_id": "btn-update-primary"},
            post_condition=lambda: bool(self.driver.find_nodes(hwnd, phase="update_check", role="button", name="Update now") or self.driver.find_nodes(hwnd, phase="update_check", role="button", name="Start reviewed resolution")),
            post_timeout=120.0,
        )

        self.step = "update_apply"
        if self.driver.find_nodes(hwnd, phase="update_apply", role="button", name="Start reviewed resolution"):
            raise Failure("update_apply", "unexpected_conflicting_update_dialog")
        self.driver.wait_unique_node(hwnd, phase="update_apply", label="Update now", role="button", name="Update now", timeout_sec=15.0)

        range_label = f"{self.candidate[:8]} \u2192 {self.b_sha[:8]}"  # a detail only; the status API names the target
        dialog_shows_range = bool(poll_until(8.0, lambda: self.driver.find_nodes(hwnd, phase="update_apply", name_contains=range_label, require_visible=False), interval=0.25))
        tie, state_before = self.api("/api/update/status"), self.state_doc()
        if tie is None or state_before is None:
            raise Unconfirmed("update_apply", "status_or_state_unreadable_at_confirmation")
        self.baseline_ops = {str(op) for op in ((state_before.get("managed_update_attempt") or {}).get("operation_id"), (tie.get("update_progress") or {}).get("operation_id")) if op}
        targets_b = tie.get("latest_sha") == self.b_sha and tie.get("running_sha") == self.candidate and tie.get("update_channel") == "development" and tie.get("available") is True and tie.get("safe_to_apply") is True
        self.check("confirmation_targets_b", targets_b, phase="update_apply", failure="confirmation_not_a_to_b", dialog_range_shown=dialog_shows_range)

        self.mark("update_now_clicked")
        self.driver.act_or_click(hwnd, phase="update_apply", label="Update now", query={"role": "button", "name": "Update now"}, single_press_strict=True)
        self.check("update_now_clicked_once", self.driver.update_now_clicks == 1, phase="update_apply", failure="update_now_not_clicked_once", clicks=self.driver.update_now_clicks)

        def _concluded_attempt() -> dict[str, Any] | None:
            assert self.driver is not None
            self.driver.ensure_alive("update_apply")
            att = (self.state_doc() or {}).get("managed_update_attempt")
            return att if isinstance(att, dict) and att.get("operation_id") and str(att.get("operation_id")) not in self.baseline_ops and att.get("outcome") else None

        concluded = poll_until(600.0, _concluded_attempt, interval=1.0)
        self.mark("update_concluded_observed")
        if concluded is None:
            raise Unconfirmed("update_apply", "update_attempt_conclusion_timeout")
        if (outcome := str(concluded.get("outcome") or "")) == "unknown":
            raise Unconfirmed("update_apply", "update_attempt_outcome_unknown", reason=concluded.get("reason"))
        applied_ok = outcome == "applied" and concluded.get("concluded_by") == "boot" and concluded.get("base_sha") == self.candidate and concluded.get("target_sha") == self.b_sha
        self.check("update_concluded_by_boot_applied_a_to_b", applied_ok, phase="update_apply", failure="update_outcome_not_applied", outcome=outcome, concluded_by=concluded.get("concluded_by"))
        self.attempt = dict(concluded)
        self.result["facts"]["update_attempt"] = {key: pb._scalar(concluded.get(key)) for key in pb.SAME_ATTEMPT_KEYS}

    def verify_after_update_and_second_chat(self) -> None:
        assert self.driver is not None
        self.step = "after_update"
        hwnd = self.driver.acquire_main_window(phase="after_update", exclude_hwnds={self.driver.setup_hwnd}, timeout_sec=180.0)
        self.ensure_main_chat_open(phase="after_update", click_sidebar=True)

        p1, r1 = fm.prompt_text(1, self.tag), fm.reply_text(1, self.tag)
        retained: dict[str, Any] = {"durable": False, "visible_leaf": False, "uia_gap": ""}
        poll_until(60.0, lambda: (self._observe_chat_state(hwnd, phase="after_update", expected_durable=[("user", p1), ("assistant", r1)], expected_leaves=[p1, r1], state=retained), bool(retained["durable"] and retained["visible_leaf"]))[1], interval=0.5)
        read_ok, complete, _ = self.durable_history()
        if not read_ok or (complete is False and not retained["durable"]):
            raise Unconfirmed("after_update", "history_unreadable_after_update" if not read_ok else "history_window_incomplete_after_update")
        self.check("post_update_turn_1_durable_history_retained", retained["durable"], phase="after_update", failure="durable_history_not_retained")
        if not retained["visible_leaf"]:
            gap_code = retained["uia_gap"] or "leaf_text_not_observed"
            self.check("post_update_turn_1_visible_leaf_retained", False, phase="after_update", uia_gap=gap_code)
            raise Unconfirmed("after_update", "retained_visible_leaf_text_unconfirmed", durable_history_retained=True, uia_gap=gap_code)
        self.check("post_update_turn_1_visible_leaf_retained", True, phase="after_update")

        self.step = "turn_2"
        p2, r2 = fm.prompt_text(2, self.tag), fm.reply_text(2, self.tag)
        self.send_chat_turn(2, expected_durable=[("user", p1), ("assistant", r1), ("user", p2), ("assistant", r2)], expected_visible_leaves=[p2, r2], phase="turn_2")

        state, head, init_after, status = self.state_doc(), pb.read_git_head(self.app_root / "repo" / ".git"), self.initialization_id(), self.api("/api/update/status")
        if state is None or status is None or not state.get("current_sha") or not head or not init_after or not status.get("running_sha"):
            raise Unconfirmed("turn_2", "post_update_witness_unreadable")

        doc = self.app_root / "repo" / FIXTURE_DOC
        self.check("state_and_checkout_on_b", state.get("current_sha") == self.b_sha and head == self.b_sha, phase="turn_2", failure="not_on_b_after_update")
        self.check("fixture_b_in_managed_tree", doc.is_file() and self.marker in doc.read_text(encoding="utf-8"), phase="turn_2", failure="fixture_b_marker_missing")
        self.check("same_initialization_id", init_after == self.init_id, phase="turn_2", failure="profile_reinitialized")
        self.check("running_sha_and_current_sha_b", status.get("running_sha") == self.b_sha and status.get("current_sha") == self.b_sha, phase="turn_2", failure="not_running_b_after_update")
        self.check("update_transaction_idle_after_update", (status.get("update_tx") or {}).get("active") is False, phase="turn_2", failure="update_transaction_still_active")

        try:
            same_ok, stored, public = pb.correlate_boot_update_attempt(
                state, status, candidate=self.candidate, b_sha=self.b_sha, baseline_ops=self.baseline_ops, concluded=self.attempt)
        except pb.Refusal as ref:
            raise Unconfirmed("turn_2", ref.reason) from ref
        self.check("same_attempt_before_and_after_turn_2", same_ok, phase="turn_2", failure="attempt_records_disagree_or_invalid", state_attempt=stored, status_attempt=public)

    def judge_git_traces(self, *, require: bool) -> None:
        """Read the app's sinks until they settle and publish ``git_traces.judge``'s counts. Only a completed run is
        settled by it: diagnostics after a failure never change that failure."""
        self.step = "git_traces"
        bounds = gt.bounds(self.result["marks"], self.attempt)
        for _attempt in range(5):  # a Git starting during the read shows as a count mismatch: read again
            sinks = self.wh.read_trace_sinks(self.trace1, self.trace2_dir)
            summary, outcome = gt.judge(sinks, bounds, control_ok=self.control_ok, target=self.b_sha, **self._package())
            if summary["identity"]["state"] != "unconfirmed":
                break
            time.sleep(3.0)
        self.save("git-traces.json", summary)
        self.result["facts"]["git_traces"] = {key: summary[key] for key in ("identity", "control_passed", "requirements")}
        if require and outcome is not None:
            raise (Failure if outcome[0] == "failed" else Unconfirmed)("git_traces", outcome[1], **outcome[2])
        if require:
            self.check("git_identity_and_phase_operations_verified", True, phase="git_traces", **summary["requirements"])

    def settle(self, verdict: str, phase: str, code: str, fields: dict[str, Any]) -> None:
        clean_fields = {k: pb._scalar(v) for k, v in fields.items()}
        self.result["reasons"].append({"verdict": verdict, "phase": phase, "code": code, **({"fields": clean_fields} if clean_fields else {})})
        if not self._decided:
            self.result.update(verdict=verdict, phase=phase, code=code)
            self._decided = True

    def _guarded(self, fn: Callable[[], None]) -> bool:
        try:
            fn()
            return True
        except Unconfirmed as oc:
            self.settle("unconfirmed", oc.phase, oc.code, oc.fields)
        except wh.Unread as ur:
            self.settle("unconfirmed", self.step, ur.code, ur.fields)
        except Failure as fl:
            self.settle("failed", fl.phase, fl.code, fl.fields)
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
            where = f"{pathlib.Path(tb.filename).name}:{tb.lineno}:{tb.name}" if tb else ""
            self.settle("error", self.step, "harness_error", {"type": type(exc).__name__, "where": where})
        return False

    def _run_all_steps(self) -> None:
        self.prepare_host_and_package()
        self.prepare_fixture_mirror_and_remove_host_git()
        self.control_git_identity()
        with fm.FixtureModel(self.tag) as model:
            self.model = model
            self.launch_and_pass_wizard()
            self.run_first_chat_and_check_a()
            self.run_plain_check_and_update_a_to_b()
            self.verify_after_update_and_second_chat()

    def run(self) -> int:
        try:
            ok = self._guarded(self._run_all_steps)
            if ok or self.trace1.exists() or self.trace2_dir.is_dir():  # a passed run is judged even with its sinks gone
                ok = self._guarded(lambda: self.judge_git_traces(require=ok)) and ok
            if ok and not self._decided:
                self.result.update(verdict="passed", phase="completed", code="all_checks_passed")
        finally:
            if self.model is not None:
                mod_summary = self.model.summary()
                self.save("fixture-model.json", mod_summary)
                self.result["facts"]["fixture_model"] = {k: v for k, v in mod_summary.items() if k != "events"}
            self.result["recorded_at"] = pb._utc_now()
            self.save("result.json", self.result)
            for pub in sorted(PUBLISHED_FILES):
                if (path := self.evidence / pub).is_file():
                    assert_privacy_clean(json.loads(path.read_text(encoding="utf-8")), tag=self.tag, where=pub)
        return EXIT[self.result["verdict"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run",))  # the summary is prepare_build's, before any venv exists
    for flag in ("--candidate", "--archive", "--archive-sha256", "--candidate-src", "--evidence", "--mirror-url", "--tag"):
        parser.add_argument(flag, required=True)
    return QualificationRun(parser.parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
