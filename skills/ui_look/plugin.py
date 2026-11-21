"""ui_look: Ouroboros looks at its own interface, read-only, with the ui_design skill's live stand.

One tool, ``look``. It gathers what the stand needs: the owner's window from the task's ``owner_client`` fact and a
phone, both themes, the pinned Chromium of the browser tools and a WebKit that is already installed, the served
``web/`` and a draft one. It then runs ``skills/ui_design/references/live_stand.py look`` in a child process group of
its own and answers that child's report briefly. The stand plans the places, measures and draws the frames; nothing
of it is copied here. Stop, the time budget and the work's own deadline end the child, and the report keeps what was
measured until then. Frames and the report are flat files in ``uploads/screenshots`` named after the work, which the
existing age-out of that folder removes.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

TOOL_TIMEOUT_SEC = 1800
BUDGET_SEC, BUDGET_MIN_SEC, BUDGET_MAX_SEC = 480, 30, 1500
GRACE_SEC = 90  # after the budget the child finishes its screen and writes its report, then it is ended
DESK = {"name": "desk", "width": 1440, "height": 900, "scale": 2, "touch": False}
PHONE = {"name": "phone", "width": 390, "height": 844, "scale": 3, "touch": True}
ENGINES = ("chromium", "webkit")
THEMES = ("dark", "light")
# The findings a summary row names by example: covered or overlapping, small targets, overflow and cut text.
EXAMPLE_KINDS = ("text-covered", "controls-overlap", "pinned-over-content", "target-small", "page-overflow",
                 "container-overflow", "text-clipped", "text-clipped-by-ancestor", "text-truncated")
SUMMARY_ROWS, EXAMPLES_PER_ROW, FRAMES_LISTED = 60, 3, 80
NOTE = ("Measurements for your judgement, never a verdict: open frames with view_image before you conclude. A frame "
        "is captioned with the engine that drew it; neither engine is the owner's own window (the macOS app draws "
        "with the system WKWebView).")

DESCRIPTION = (
    "Look at Ouroboros's own interface, read-only, on the live data. By default the places a change touches: the "
    "diff between the draft web/ (this task's body candidate, or web_root) and the served one, or the files in "
    "changed; every place when web/ui.css, a token or the shell changes. Looks at the owner's window (owner_client) "
    "and a 390 px phone, in both themes, with the pinned Chromium and, where already installed, WebKit emulation. "
    "Answers the measurements per screen (covered text, small targets, overflow, cut text, uneven rows, long text): "
    "facts to judge, never a verdict. Frames only when asked, as PNG files in uploads/screenshots captioned with "
    "their engine. Every write is answered locally; nothing on the install changes. Takes minutes."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "web_root": {"type": "string", "description": "A draft web/ folder to show over the live data instead of "
                                                      "the served one (default: this task's body candidate, if any)."},
        "changed": {"type": "array", "items": {"type": "string"},
                    "description": "Changed interface files (e.g. web/modules/skills.js), instead of the diff."},
        "places": {"type": "array", "items": {"type": "string"},
                   "description": "Places to look at instead of the diff's (ids of web/modules/pages.js), or [\"all\"]."},
        "themes": {"type": "array", "items": {"type": "string", "enum": list(THEMES)}},
        "engines": {"type": "array", "items": {"type": "string", "enum": list(ENGINES)}},
        "frames": {"type": "string", "enum": ["none", "first", "all"],
                   "description": "none (default); first: one frame per screen; all: with its scrolled frames."},
        "rooms": {"type": "integer", "minimum": 0, "maximum": 5, "description": "Also the first N Project rooms."},
        "budget_sec": {"type": "integer", "minimum": BUDGET_MIN_SEC, "maximum": BUDGET_MAX_SEC,
                       "description": f"Time for the whole look (default {BUDGET_SEC})."},
    },
    "additionalProperties": False,
}


def _answer(**fields: Any) -> str:
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def _first_line(exc: BaseException) -> str:
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0][:300]


def _helper_python() -> str:
    """The interpreter for the stand: the host's agent Python when it names one, else this one (it has Playwright)."""
    return os.environ.get("OUROBOROS_AGENT_PYTHON") or sys.executable


def stand_path(skill_dir: str = "") -> pathlib.Path | None:
    """The ui_design skill's stand of the running app, else the installed copy beside this skill."""
    found = []
    try:
        from ouroboros.skill_builtin import bundled_skills_dir

        found.append(bundled_skills_dir() / "ui_design" / "references" / "live_stand.py")
    except Exception:
        pass
    if skill_dir:
        found.append(pathlib.Path(skill_dir).parent / "ui_design" / "references" / "live_stand.py")
    return next((path for path in found if path.is_file()), None)


def owner_client(ctx: Any) -> dict:
    try:
        from ouroboros.client_surface import owner_client_fact

        return owner_client_fact(getattr(ctx, "task_metadata", None)) or {}
    except Exception:
        return {}


def viewports(fact: dict) -> list:
    """The owner's window as the owner_client fact measured it, then a phone; a desk window when none is known."""
    view = fact.get("viewport") if isinstance(fact.get("viewport"), dict) else {}
    width, height = int(view.get("w") or 0), int(view.get("h") or 0)
    if width >= 280 and height >= 280:
        scale = min(4.0, max(1.0, float(fact.get("device_pixel_ratio") or 1)))
        owner = {"name": "owner", "width": width, "height": height, "scale": scale,
                 "touch": bool(fact.get("coarse_pointer")), "source": "owner_client"}
    else:
        owner = dict(DESK, source="no owner window in this work's owner_client fact")
    return [owner] if (owner["width"], owner["touch"]) == (PHONE["width"], True) else [owner, dict(PHONE, source="phone")]


def _owner_engine_is_webkit(fact: dict) -> bool:
    agent = str(fact.get("ua") or "")
    return "AppleWebKit" in agent and not any(mark in agent for mark in ("Chrome/", "Chromium/", "Edg/"))


def _default_browsers_cache() -> pathlib.Path:
    home = pathlib.Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "ms-playwright"
    if os.name == "nt":
        return pathlib.Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local") / "ms-playwright"
    return pathlib.Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache") / "ms-playwright"


def pinned_chromium() -> tuple[dict | None, str]:
    """The browser tools' pinned Chromium if it is installed: (engine row, "") or (None, why not). Never installs."""
    try:
        from ouroboros import chromium_runtime
        from ouroboros.tools.browser import _pinned_chromium

        return {"engine": "chromium", "executable_path": str(_pinned_chromium(allow_install=False)),
                "sandbox": chromium_runtime.sandbox_enabled()}, ""
    except Exception as exc:
        version = getattr(sys.modules.get("ouroboros.chromium_runtime"), "CHROMIUM_VERSION", "") or "build"
        if "is not installed" in str(exc):  # the browser tools' own lookup found no pinned build
            return None, (f"the pinned Chromium {version} is not installed where the browser tools look; a "
                          "browse_page call with engine 'chromium' installs it, this tool never installs a browser")
        return None, f"the pinned Chromium {version} cannot run here: {_first_line(exc)}"


def installed_webkit(data_dir: pathlib.Path) -> str | None:
    """The browsers root (a PLAYWRIGHT_BROWSERS_PATH value) holding a WebKit build, where the browser tools look."""
    try:
        from ouroboros import chromium_runtime
        from ouroboros.tools.browser import _has_platform_webkit
    except Exception:
        return None
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    roots = [(configured, chromium_runtime.browsers_root(configured))] if configured else []
    roots += [("0", chromium_runtime.browsers_root("0")),
              (str(data_dir / "playwright-browsers"), data_dir / "playwright-browsers"),
              (str(_default_browsers_cache()), _default_browsers_cache())]
    for value, root in roots:
        if root is not None and _has_platform_webkit(pathlib.Path(root)):
            return value
    return None


def engine_rows(asked: list, fact: dict, data_dir: pathlib.Path) -> tuple[list, list, dict]:
    """(engine rows for the stand in the owner's engine first, the ones not available with why, child environment)."""
    order = ("webkit", "chromium") if _owner_engine_is_webkit(fact) else ENGINES
    rows, missing, env = [], [], {}
    for name in (engine for engine in order if engine in asked):
        if name == "chromium":
            row, why = pinned_chromium()
        else:
            root = installed_webkit(data_dir)
            row, why = ({"engine": "webkit"}, "") if root else (None, "no WebKit build is installed; this tool never downloads one")
            if root:
                env["PLAYWRIGHT_BROWSERS_PATH"] = root
        (rows.append(row) if row else missing.append({"engine": name, "why": why}))
    return rows, missing, env


def web_roots(ctx: Any, web_root: str) -> tuple[pathlib.Path | None, pathlib.Path, str]:
    """(the draft web/ to show or None for the served one, the served web/, a refusal or "")."""
    try:
        from ouroboros import body_candidate

        served = pathlib.Path(body_candidate.serving_repo_dir_for(ctx)) / "web"
        bound = body_candidate.descriptor(ctx) if body_candidate.is_bound(ctx) else {}
    except Exception:
        served, bound = pathlib.Path(getattr(ctx, "repo_dir", ".")) / "web", {}
    if web_root:
        draft = pathlib.Path(str(web_root)).expanduser()
        if not draft.is_absolute():
            base = getattr(ctx, "active_repo_dir", None)
            draft = (pathlib.Path(base()) if callable(base) else pathlib.Path(getattr(ctx, "repo_dir", "."))) / draft
    elif bound:
        draft = pathlib.Path(str(bound["path"])) / "web"
    else:
        return None, served, ""
    if not (draft / "index.html").is_file():
        return None, served, f"{draft} has no index.html, so it is not a web/ folder"
    return (None if draft.resolve() == served.resolve() else draft), served, ""


def budget(asked: Any) -> tuple[float, str]:
    """The look's time: what was asked within bounds, and never past the work's own deadline."""
    seconds = float(min(BUDGET_MAX_SEC, max(BUDGET_MIN_SEC, int(asked or BUDGET_SEC))))
    try:
        from ouroboros.model_wait import dispatch_deadline_remaining_sec
        from ouroboros.runtime_limits import NESTED_SETTLEMENT_MARGIN_SEC as margin

        remaining = dispatch_deadline_remaining_sec()
    except Exception:
        remaining, margin = None, 0
    if remaining is not None:
        seconds = min(seconds, remaining - margin - GRACE_SEC)
        if seconds < BUDGET_MIN_SEC:
            return seconds, f"this work's deadline leaves {int(max(0.0, remaining))} s, too little for a look"
    return seconds, ""


def _child_env(extra: dict) -> dict:
    env = {**os.environ, **extra}
    try:
        from ouroboros.node_runtime import forward_playwright_node

        forward_playwright_node(env)  # Playwright's driver runs on the pinned Node, as in every browser child
    except Exception:
        pass
    return env


def _group_kwargs() -> dict:
    try:
        from ouroboros.platform_layer import merge_hidden_kwargs, subprocess_new_group_kwargs

        return merge_hidden_kwargs(subprocess_new_group_kwargs())
    except Exception:
        return {} if os.name == "nt" else {"start_new_session": True}


def _end_tree(proc: subprocess.Popen) -> None:
    """End the stand and every browser it started: its whole process tree, by its own PID."""
    try:
        from ouroboros.platform_layer import kill_process_tree

        kill_process_tree(proc)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _examples(screen: dict) -> list:
    out = [f"{f['kind']}: {f.get('message', '')} ({str(f.get('where', ''))[:90]})"[:220]
           for f in screen.get("findings") or [] if f.get("kind") in EXAMPLE_KINDS]
    out += [f"uneven-rows: gaps {row.get('gaps')} px between {len(row.get('items') or [])} controls at y {row.get('y')}"
            for row in screen.get("uneven_rows") or []]
    out += [f"long-text: {text.get('chars')} characters in {text.get('el')}" for text in screen.get("long_texts") or []]
    return out[:EXAMPLES_PER_ROW]


def summary(report: dict, report_path: pathlib.Path, outcome: dict, missing: list) -> str:
    """The child's report, briefly: why these places, then one row per screen with findings, then the rest."""
    if not report.get("playwright", True):
        return _answer(ok=False, reason="no_playwright", message="Playwright is not importable in the interpreter "
                       f"that runs the stand ({_helper_python()}); nothing was looked at.", notes=report.get("notes"))
    ran = [row for row in report.get("engines") or [] if row.get("status") == "ran"]
    tried = [row for row in report.get("engines") or [] if row.get("status") == "unavailable"]
    if not ran and tried:
        return _answer(ok=False, reason="no_browser", message="No browser could start; nothing was looked at.",
                       engines=tried + missing, report=str(report_path))
    short = {"chromium": "Chromium", "webkit": "WebKit emulation"}
    rows, clean = [], []
    for screen in report.get("screens") or []:
        counts = {kind: n for kind, n in (screen.get("counts") or {}).items() if n}
        name = f"{screen['screen']} {screen['theme']} {screen['viewport']} {short.get(screen['engine'], screen['engine'])}"
        if not counts:
            clean.append(name)
            continue
        rows.append({"screen": screen["screen"], "engine": short.get(screen["engine"], screen["engine"]),
                     "theme": screen["theme"], "viewport": f"{screen['viewport']} {screen['size']}",
                     "counts": counts, "examples": _examples(screen)})
    frames = [{"file": path, "engine": screen["label"], "screen": screen["screen"], "theme": screen["theme"],
               "viewport": screen["viewport"]} for screen in report.get("screens") or [] for path in screen.get("frames") or []]
    plan = report.get("plan") or {}
    answer = {"ok": True, "summary": rows[:SUMMARY_ROWS], "screens_looked": len(report.get("screens") or []),
              "clean_screens": clean[:40], "places": plan.get("why") or [],
              "engines": [{"engine": row["engine"], "label": row.get("label", "")} for row in ran]
                         + [{"engine": row["engine"], "why": row.get("error", "")} for row in tried] + missing,
              "viewports": report.get("viewports"), "themes": report.get("themes"), "panels": report.get("panels"),
              "frames": frames[:FRAMES_LISTED], "report": str(report_path),
              "not_looked": (report.get("not_looked") or []) + ([outcome["ended"]] if outcome.get("ended") else []),
              "notes": (report.get("notes") or []) + (plan.get("unplaced") or []),
              "writes_answered_locally": report.get("writes_answered_locally", 0),
              "elapsed_sec": report.get("elapsed_sec"), "note": NOTE}
    if len(rows) > SUMMARY_ROWS or len(clean) > 40 or len(frames) > FRAMES_LISTED:
        answer["omitted"] = (f"{max(0, len(rows) - SUMMARY_ROWS)} rows, {max(0, len(clean) - 40)} clean screens and "
                             f"{max(0, len(frames) - FRAMES_LISTED)} frames are in the report")
    if not report.get("finished"):
        answer["not_looked"].append(f"the stand ended before its report was complete (exit {outcome.get('exit')})")
        if not outcome.get("ended"):
            answer["log"] = outcome.get("log", "")
    return _answer(**answer)


class Look:
    """The ``look`` tool: one child process per call, watched for Stop, budget and its own end."""

    def __init__(self, api: Any):
        self.api = api

    def _stop_reason(self) -> str:
        try:
            state = self.api.get_stop_state()
        except Exception:
            return ""  # a host without the reader: its own checks between calls still hold
        status = str((state or {}).get("status") or "")
        return "" if status == "continue" else (str((state or {}).get("reason") or status) or "stop")

    def run(self, ctx: Any, web_root: str = "", changed: list | None = None, places: list | None = None,
            themes: list | None = None, engines: list | None = None, frames: str = "none", rooms: int = 0,
            budget_sec: int = BUDGET_SEC) -> str:
        info = self.api.get_runtime_info() or {}
        data_dir = pathlib.Path(info.get("data_dir") or getattr(ctx, "drive_root", "."))
        asked_themes = [theme for theme in (themes or THEMES) if theme in THEMES] or list(THEMES)
        asked_engines = [engine for engine in (engines or ENGINES) if engine in ENGINES] or list(ENGINES)
        frames = frames if frames in ("none", "first", "all") else "none"
        stand = stand_path(str(info.get("skill_dir") or ""))
        if stand is None:
            return _answer(ok=False, reason="no_stand", message="The ui_design skill's live_stand.py is not here.")
        draft, served, refusal = web_roots(ctx, web_root)
        if refusal:
            return _answer(ok=False, reason="bad_web_root", message=refusal)
        fact = owner_client(ctx)
        rows, missing, env = engine_rows(asked_engines, fact, data_dir)
        if not rows:
            return _answer(ok=False, reason="no_browser", message="No browser to look with; nothing was started.",
                           engines=missing)
        seconds, refusal = budget(budget_sec)
        stopped = refusal or self._stop_reason()
        if stopped:
            return _answer(ok=False, reason="stopped" if not refusal else "no_time", message=stopped)
        work = re.sub(r"[^A-Za-z0-9_-]+", "_", str(getattr(ctx, "task_id", "") or "work"))[:48]
        shots = data_dir / "uploads" / "screenshots"
        shots.mkdir(parents=True, exist_ok=True)
        prefix = f"look-{work}-{time.strftime('%Y%m%dT%H%M%S')}-"
        report_path = shots / f"{prefix}report.json"
        spec = {"url": f"http://127.0.0.1:{int(info.get('server_port') or 8765)}", "web_root": str(draft) if draft else None,
                "base": str(served), "changed": list(changed) if changed else None, "places": list(places) if places else None,
                "viewports": viewports(fact), "themes": asked_themes, "engines": rows, "frames": frames,
                "frames_dir": str(shots), "frame_prefix": prefix, "rooms": max(0, min(5, int(rooms or 0))),
                "budget_sec": seconds}
        outcome = self._run_child(stand, spec, report_path, env, seconds)
        try:
            from ouroboros.maintenance_retirement import artifact_changed

            artifact_changed(data_dir)  # the frames and the report age out with uploads/screenshots
        except Exception:
            pass
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _answer(ok=False, reason="no_report", message=f"The stand left no report (exit {outcome['exit']}"
                           + (f"; {outcome['ended']}" if outcome["ended"] else "") + ").", log=outcome["log"])
        return summary(report, report_path, outcome, missing)

    def _run_child(self, stand: pathlib.Path, spec: dict, report_path: pathlib.Path, env: dict, seconds: float) -> dict:
        with tempfile.TemporaryDirectory(prefix="ui-look-") as tmp:
            spec_path, log_path = pathlib.Path(tmp) / "spec.json", pathlib.Path(tmp) / "stand.log"
            spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            command = [_helper_python(), "-I", "-B", str(stand), "look", "--spec", str(spec_path), "--report", str(report_path)]
            ended, deadline = "", time.monotonic() + seconds + GRACE_SEC
            with open(log_path, "wb") as log:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        env=_child_env(env), **_group_kwargs())
                try:
                    while proc.poll() is None:
                        try:
                            proc.wait(timeout=1.0)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                        reason = self._stop_reason()
                        if reason or time.monotonic() > deadline:
                            ended = f"stopped ({reason})" if reason else f"ended after {int(seconds + GRACE_SEC)} s"
                            break
                finally:
                    if proc.poll() is None:
                        _end_tree(proc)
            return {"exit": proc.returncode, "ended": ended,
                    "log": log_path.read_text(encoding="utf-8", errors="replace")[-600:]}


def register(api: Any) -> None:
    look = Look(api)
    api.register_tool("look", look.run, description=DESCRIPTION, schema=SCHEMA, timeout_sec=TOOL_TIMEOUT_SEC)
