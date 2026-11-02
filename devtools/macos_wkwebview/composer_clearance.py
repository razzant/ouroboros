#!/usr/bin/env python3
"""Check in the macOS system WKWebView that the Main and Project composers clear the newest message.

    python3 -I -B devtools/macos_wkwebview/composer_clearance.py --out DIR

DIR (new or empty) receives report.json, observation.json, server.log, driver.log and the WKWebView's own snapshots.
The verdict is PASS, FAIL (the product let a message end under its composer), INCONCLUSIVE (the run could not
establish its conditions) or UNSUPPORTED (no macOS, PyObjC WebKit or display here), with exit codes 0, 1, 3 and 4.
``--identity-only`` prints the source identity a report records, for any checkout. README.md explains the scenario.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import plistlib
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "ouroboros.macos-wkwebview-composer-clearance/1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAGE_SCRIPT = HERE / "composer_clearance.js"
EXIT_CODES = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3, "UNSUPPORTED": 4}

MAIN_FEED = "#chat-messages"
PROJECT_FEED = "#project-panel .chat-messages"
PROJECT_ID = "composer-room"
PROJECT_ROW = f'.nav-project-row[data-project-id="{PROJECT_ID}"]'
MAIN_TEXT = "Main saved answer"
PROJECT_TEXT = "Project saved answer"
MULTILINE = "one\ntwo\nthree\nfour\nfive"
# The desktop window's default width, then the phone-width stack layout of the same engine.
WIDTHS = (1100, 390)
START_HEIGHT, LOWEST_HEIGHT = 734, 360
SURFACES = ("main", "project")
PHASES = ("history", "reopen", "growth")
COMPOSERS = ("empty", "multiline")
# Negatives over the product's stylesheet: the layout before the spacer (end padding), and no reserve at all.
VARIANTS = {
    "old-padding": "#chat-messages,.chat-messages{padding-bottom:var(--chat-input-reserve,108px)!important}"
                   "#chat-messages::after,.chat-messages::after{content:none!important}",
    "no-reserve": "#chat-messages,.chat-messages{padding-bottom:0!important}"
                  "#chat-messages::after,.chat-messages::after{content:none!important}",
}
SHOTS = {("history", "multiline"), ("growth", "multiline")}
# What the copy holds and the identity hashes: the source entries the desktop app ships as runtime resources
# (Ouroboros.spec `datas`) without its tests, read from the working tree as they are, uncommitted edits included.
RUNTIME_INPUTS = ("server.py", "VERSION", "BIBLE.md", "README.md", "pyproject.toml", "requirements-runtime.lock",
                  "uv.lock", ".gitignore", "ouroboros", "supervisor", "prompts", "web", "docs", "assets", "skills")
GENERATED = frozenset({"__pycache__", "node_modules", ".DS_Store"})
ROOT_PARENT = Path("/tmp")  # short: nested AF_UNIX socket paths stay within macOS's 104 bytes
SERVED = ("style.css", "ui.css", "theme.js", "app.js", "modules/chat.js", "modules/chat_input_sizing.js",
          "modules/composer.js")


# --- judgement (pure) ---------------------------------------------------------------------------------------------

def clears(geometry: dict) -> bool | None:
    """The newest row ends at or above the composer's top (1 px of rounding); None when either is missing."""
    last, top = geometry.get("lastBottom"), geometry.get("composerTop")
    return None if last is None or top is None else last <= top + 1


def _extent(end: dict) -> float:
    """Where the newest row ends, measured from the transcript's top edge with the transcript scrolled to its top."""
    return end["lastBottom"] - end["feedTop"] + end["scrollTop"]


def short_history(sample: dict) -> bool:
    """Only the reserve creates the scroll range clearance needs: the rows fit the transcript, yet scrolled to the
    top the newest one would end under the composer."""
    end = sample["end"]
    if end.get("lastBottom") is None or end.get("composerTop") is None:
        return False
    return end["composerTop"] - end["feedTop"] + 1 < _extent(end) <= end["clientHeight"]


def fitted_height(sample: dict, lowest: int, highest: int) -> int | None:
    """The viewport height that puts the newest row's end midway between the composer's top and the transcript's
    bottom with the transcript at its top; None when the composer does not cover the transcript."""
    end = sample["end"]
    if end.get("lastBottom") is None or end.get("composerTop") is None:
        return None
    above, client = end["composerTop"] - end["feedTop"], end["clientHeight"]
    if client - above < 4:
        return None
    return max(lowest, min(highest, round(sample["innerHeight"] + _extent(end) - (above + client) / 2)))


def _px(value) -> float:
    try:
        return float(str(value).strip().removesuffix("px"))
    except ValueError:
        return 0.0


def at_newest(geometry: dict) -> bool:
    """The transcript sits at the end of its scroll range: where the owner reads the newest message."""
    return geometry["scrollTop"] >= geometry["scrollHeight"] - geometry["clientHeight"] - 1


def handed_back(sample: dict) -> bool:
    """A negative returned the transcript to the position the positive flow had left it at."""
    restored = sample.get("restored")
    return restored is not None and abs(restored["after"] - restored["before"]) <= 1


def judge(observation: dict) -> dict:
    """The verdict of one native run from its recorded observation; a missing condition is never a pass."""
    if observation.get("status") == "unsupported":
        return {"verdict": "UNSUPPORTED", "failures": [], "gaps": [observation.get("reason") or "unsupported"]}
    failures, gaps, old = [], [], []
    if observation.get("status") != "observed":
        detail = "; ".join(observation.get("errors") or []) or "no detail"
        gaps.append(f"the native run ended {observation.get('status')!r}: {detail}")
    if observation.get("served_matches_source") is not True:
        gaps.append("the server's assets were not shown to be the tested source")
    samples = {(s["width"], s["surface"], s["phase"], s["composer"], s["variant"]): s
               for s in observation.get("samples", [])}
    gaps.extend(f"{s['id']}: layout still moving after {s['settle']['frames']} frames"
                for s in samples.values() if not s.get("settle", {}).get("settled"))
    gaps.extend(f"{s['id']}: the negative did not hand back its starting position {s.get('restored')}"
                for s in samples.values() if s["variant"] != "current" and not handed_back(s))
    returns = {item["width"]: item for item in observation.get("returns", [])}
    probe = (observation.get("engine_probe") or {}).get("flex_column_container") or {}
    unscrollable = None if "endPaddingScrolls" not in probe else not probe["endPaddingScrolls"]
    for width in observation.get("widths") or WIDTHS:
        for surface in SURFACES:
            for phase in PHASES:
                for composer in COMPOSERS:
                    name = f"{surface} {phase}, {composer} composer, {width} px"
                    sample = samples.get((width, surface, phase, composer, "current"))
                    if sample is None:
                        gaps.append(f"{name}: not measured")
                        continue
                    clear, end = clears(sample["end"]), sample["end"]
                    if clear is None:
                        gaps.append(f"{name}: no newest row or composer")
                    elif not clear:
                        failures.append(f"{name}: newest row ends at {end['lastBottom']:.1f} px, under the "
                                        f"composer's top at {end['composerTop']:.1f} px")
                    if phase != "growth" and not short_history(sample):
                        gaps.append(f"{name}: not a short history whose reserve alone creates the scroll range")
            for composer in COMPOSERS:
                name = f"{surface} history, {composer} composer, {width} px"
                broken = samples.get((width, surface, "history", composer, "no-reserve"))
                if broken is None or clears(broken["end"]) is not False:
                    gaps.append(f"no-reserve negative, {name}: the deliberate overlap was "
                                + ("not measured" if broken is None else "not detected"))
                before = samples.get((width, surface, "history", composer, "old-padding"))
                cleared = None if before is None else clears(before["end"])
                old.append(cleared)
                if before is None:
                    gaps.append(f"old-padding negative, {name}: not measured")
                elif unscrollable and cleared is not False:
                    gaps.append(f"old-padding negative, {name}: this engine does not scroll end padding, "
                                "yet the old layout cleared the composer")
        back = returns.get(width)
        if back is None:
            gaps.append(f"Main's return from another page, {width} px: not measured")
            continue
        if back["hidden"] != back["shown"]:
            failures.append(f"Main behind a page, {width} px: its reserve moved {back['shown']} -> {back['hidden']}")
        if _px(back["shown"]) <= 92:
            gaps.append(f"Main's return, {width} px: the multiline composer left the reserve at its 92 px floor")
        # The first frame back is judged as presented: Main as the first rendering update that shows it lays it out,
        # after that update's own callbacks (composer_clearance.js `leaveAndReturn`). It is evidence only when Main
        # was left at its newest message.
        left = back.get("before")
        if left is None or not at_newest(left):
            where = "an unrecorded position" if left is None else (
                f"scrollTop {left['scrollTop']:.0f} of {left['scrollHeight'] - left['clientHeight']:.0f}")
            gaps.append(f"Main's return, {width} px: Main was left at {where}, not its newest message, "
                        "so its first frame back is no evidence")
            continue
        presented = back.get("presented")
        first = None if presented is None else clears(presented)
        if first is None:
            gaps.append(f"Main's first frame back, {width} px: no layout of the shown Main with a newest row and "
                        "composer was observed")
        elif not first:
            failures.append(f"Main's first frame back, {width} px: as presented (frame "
                            f"{presented.get('presentedFrame')}), the newest row ends at "
                            f"{presented['lastBottom']:.1f} px, under the composer's top at "
                            f"{presented['composerTop']:.1f} px")
    if not old or None in old:
        historical = "unmeasured"
    else:
        historical = "reproduced" if not any(old) else "not reproduced" if all(old) else "partly reproduced"
    verdict = "FAIL" if failures else "INCONCLUSIVE" if gaps else "PASS"
    return {"verdict": verdict, "failures": failures, "gaps": gaps,
            "historical_defect": {"engine_leaves_end_padding_unscrollable": unscrollable,
                                  "old_padding_layout": historical}}


# --- source identity -----------------------------------------------------------------------------------------------

def product_files(root: Path) -> list[Path]:
    """The runtime inputs under `root`, relative and sorted; caches generated inside them stay out."""
    found = []
    for name in RUNTIME_INPUTS:
        if (root / name).is_file():
            found.append(Path(name))
            continue
        for directory, names, files in os.walk(root / name):
            names[:] = [entry for entry in names if entry not in GENERATED]
            relative = Path(directory).relative_to(root)
            found.extend(relative / entry for entry in files if entry not in GENERATED and not entry.endswith(".pyc"))
    return sorted(found)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(root: Path) -> dict:
    digests = {path.as_posix(): _sha256(root / path) for path in product_files(root)}
    tree = hashlib.sha256("".join(f"{digest}  {path}\n" for path, digest in digests.items()).encode()).hexdigest()
    version = root / "VERSION"
    return {"version": version.read_text(encoding="utf-8").strip() if version.is_file() else None,
            "files": len(digests), "tree_sha256": tree,
            "layout_files": {f"web/{name}": digests.get(f"web/{name}") for name in SERVED}}


def instrument_identity() -> dict:
    return {path.name: _sha256(path) for path in (Path(__file__).resolve(), PAGE_SCRIPT)}


# --- runner: disposable copy, isolated data, real server, native driver ------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _command(*argv: str) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def host_facts() -> dict:
    facts = {"platform": sys.platform, "machine": platform.machine(), "python": sys.version,
             "executable": sys.executable, "mac_ver": platform.mac_ver()[0] or None}
    if sys.platform == "darwin":
        facts.update(sw_vers=_command("/usr/bin/sw_vers"), translated=_command("/usr/sbin/sysctl", "-n",
                     "sysctl.proc_translated"), session=_command("/bin/launchctl", "managername"))
        try:
            with open("/Applications/Safari.app/Contents/Info.plist", "rb") as handle:
                facts["safari"] = plistlib.load(handle).get("CFBundleShortVersionString")
        except OSError:
            facts["safari"] = None
    return facts


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(url: str, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - the check's own loopback server
        return response.read()


def _wait_ready(url: str, proc, seconds: float) -> dict:
    """Poll the server's state until its supervisor is ready, the process exits or `seconds` pass."""
    deadline, last = time.monotonic() + seconds, ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before it was ready")
        try:
            state = json.loads(_get(f"{url}/api/state", 2.0))
            if state.get("supervisor_ready") is True:
                return state
            last = f"supervisor_ready={state.get('supervisor_ready')!r}"
        except (OSError, ValueError) as exc:
            last = repr(exc)
        time.sleep(0.25)
    raise TimeoutError(f"server not ready in {seconds:.0f} s ({last})")


def _history(lead: str, question: str) -> list[dict]:
    """A short saved exchange: an owner question and one answer a few lines long."""
    return [{"ts": "2026-09-17T09:59:00Z", "direction": "in", "text": question,
             "client_message_id": f"composer-check-{lead.split()[0].lower()}"},
            {"ts": "2026-09-17T10:00:00Z", "direction": "out", "format": "markdown",
             "text": f"{lead}.\n\nThe transcript reserves room for the composer, so the newest line stays readable:"
                     "\n\n- the desktop window\n- a narrow window\n- a reopened room\n- a growing composer"}]


def _answer(chat_id: int) -> dict:
    return {"type": "chat", "role": "assistant", "chat_id": chat_id, "ts": "2026-09-18T10:00:00Z",
            "content": "Growing answer.\n\n" + "\n".join(f"- New line {index}" for index in range(40))}


def _pyobjc_present() -> bool:
    probe = "import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in " \
            "('objc', 'AppKit', 'Foundation', 'WebKit', 'PyObjCTools')) else 1)"
    return subprocess.run([sys.executable, "-I", "-c", probe], timeout=30, check=False).returncode == 0


# Its container's SIGTERM stops the server: its own handler exits 143, the signal itself reads -15.
STOPPED_SERVER = frozenset({143, -15})


def _stop(proc, container, seconds: float) -> dict:
    """Stop one child through its container: whether it still ran, how it ended, and anything of it left."""
    running = proc.poll() is None
    if running:
        proc.terminate()
        try:
            proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            pass
    try:
        left = container.reap()
    except Exception as exc:
        left = f"reap unconfirmed: {exc!r}"
    finally:
        container.close()
    return {"running_at_stop": running, "exit": proc.poll(), "left_running": left or None}


def lifecycle_gaps(report: dict, observation: dict) -> list[str]:
    """Why the run's own processes do not vouch for its observation: the driver's exit, a server that did not end
    by its stop, or a child its container could not confirm gone."""
    gaps = []
    driver = report.get("driver")
    if driver is not None:
        expected = EXIT_CODES["UNSUPPORTED"] if observation.get("status") == "unsupported" else 0
        if driver.get("exit") != expected:
            gaps.append(f"the native driver exited {driver.get('exit')}, not {expected}")
    teardown = report.get("teardown") or {}
    gaps.extend(f"{name}: teardown unconfirmed ({child['left_running']})"
                for name, child in teardown.items() if child.get("left_running"))
    server = teardown.get("server")
    if server is not None and not (server["running_at_stop"] and server["exit"] in STOPPED_SERVER):
        gaps.append(f"the server exited {server['exit']} " + ("when stopped" if server["running_at_stop"] else
                                                               "before it was stopped"))
    return gaps


def _finish(out: Path, report: dict, observation: dict) -> int:
    """The observation's verdict, unless the run's own lifecycle withholds a PASS; FAIL and UNSUPPORTED stand."""
    judgement = judge(observation)
    lifecycle = lifecycle_gaps(report, observation)
    if lifecycle:
        judgement["gaps"] = judgement["gaps"] + lifecycle
        if judgement["verdict"] == "PASS":
            judgement["verdict"] = "INCONCLUSIVE"
    report.update(observation=observation, judgement=judgement, finished_at=_now())
    _write_json(out / "report.json", report)
    print(f"composer clearance: {judgement['verdict']} ({out / 'report.json'})")
    for line in (judgement["failures"] + judgement["gaps"])[:12]:
        print(f"  - {line}")
    return EXIT_CODES[judgement["verdict"]]


def observe(root: Path, out: Path, timeout: float, report: dict, teardown: list, children: list) -> dict:
    """Copy, isolate, serve and drive one native run under `root`; children and closers are handed back as made."""
    repo = root / "repo"
    for relative in product_files(REPO):
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, repo / relative)
    report["source"] = source_identity(REPO)
    if source_identity(repo)["tree_sha256"] != report["source"]["tree_sha256"]:
        raise RuntimeError("the disposable copy differs from the source")
    # The runner's own product imports read only the isolated root, never this account's settings or data.
    spec = importlib.util.spec_from_file_location("composer_check_environment", repo / "ouroboros/test_environment.py")
    environment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(environment)
    env = environment.isolated_environment(root / "iso", repo)
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(repo))
    from ouroboros import dialogue_store
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.projects_registry import create_project

    data = Path(env["OUROBOROS_DATA_DIR"])
    inert = socket.socket()  # bound and never listening: a provider call is refused at once
    inert.bind(("127.0.0.1", 0))
    teardown.append(inert.close)
    settings = dict.fromkeys(("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"),
                             "openai-compatible::inert-loopback")
    settings.update(OPENAI_COMPATIBLE_API_KEY="composer-check-inert",
                    OPENAI_COMPATIBLE_BASE_URL=f"http://127.0.0.1:{inert.getsockname()[1]}/v1",
                    OUROBOROS_MAX_WORKERS=1, OUROBOROS_RUNTIME_MODE="light",
                    OUROBOROS_STARTER_SKILLS='["builtin/telegram"]', OUROBOROS_STARTER_SKILLS_TURN_ON=False)
    (data / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    # Git stays out of the copy (a clean macOS has only the Command Line Tools installer's stub).
    tools = root / "bin"
    tools.mkdir()
    (tools / "git").write_text("#!/bin/sh\necho 'git is not available to this check' >&2\nexit 1\n")
    (tools / "git").chmod(0o755)
    port = _free_port()
    server_env = {**env, "PATH": f"{tools}{os.pathsep}{env.get('PATH', '')}", "OUROBOROS_SERVER_HOST": "127.0.0.1",
                  "OUROBOROS_SERVER_PORT": str(port), "OUROBOROS_HOST_SERVICE_PORT": str(_free_port())}
    url = f"http://127.0.0.1:{port}"
    server_log = (out / "server.log").open("wb")
    teardown.append(server_log.close)
    server = ProcessContainer()
    proc = server.spawn([sys.executable, "server.py"], cwd=repo, env=server_env, stdout=server_log,
                        stderr=subprocess.STDOUT)
    children.append(("server", proc, server, 20.0))
    state = _wait_ready(url, proc, 180.0)
    report["server"] = {"url": url, "pid": proc.pid, "served_ui": state.get("served_ui"),
                        "git": "stub that exits 1", "provider": "openai-compatible, loopback port with no listener"}
    served = {name: hashlib.sha256(_get(f"{url}/static/{name}")).hexdigest() for name in SERVED}
    report["served"] = served
    matches = all(served[name] == report["source"]["layout_files"][f"web/{name}"] for name in SERVED)
    project = create_project(data, PROJECT_ID, name="Composer room")
    for chat_id, lead in ((1, MAIN_TEXT), (project["chat_id"], PROJECT_TEXT)):
        for row in _history(lead, "Where does the newest line end?"):
            dialogue_store.append(data, "chat", {**row, "chat_id": chat_id})
    session = {"url": url, "out": str(out), "project_chat_id": project["chat_id"],
               "deadline": max(60.0, timeout - 30.0)}
    _write_json(root / "session.json", session)
    driver_log = (out / "driver.log").open("wb")
    teardown.append(driver_log.close)
    driver = ProcessContainer()
    child = driver.spawn([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--driver",
                          str(root / "session.json")], env=env, stdout=driver_log, stderr=subprocess.STDOUT)
    children.append(("driver", child, driver, 10.0))
    try:
        report["driver"] = {"exit": child.wait(timeout=timeout)}
    except subprocess.TimeoutExpired:
        report["driver"] = {"exit": None, "timed_out_after_s": timeout}
    recorded = out / "observation.json"
    observation = json.loads(recorded.read_text(encoding="utf-8")) if recorded.is_file() else {
        "status": "error", "errors": [f"the driver wrote no observation (exit {report['driver']['exit']})"]}
    if report["driver"]["exit"] is None:
        observation.setdefault("errors", []).append(f"the driver exceeded {timeout:.0f} s and was stopped")
        observation["status"] = "deadline"
    observation["served_matches_source"] = matches
    return observation


def run(out: Path, keep_root: bool, timeout: float) -> int:
    report = {"schema": SCHEMA, "started_at": _now(), "host": host_facts(), "instrument": instrument_identity()}
    if sys.platform != "darwin":
        return _finish(out, report, {"status": "unsupported", "reason": f"needs macOS; this is {sys.platform}"})
    if not _pyobjc_present():
        return _finish(out, report, {"status": "unsupported",
                                     "reason": f"{sys.executable} has no PyObjC with WebKit (pywebview's dependencies)"})
    root = Path(tempfile.mkdtemp(prefix="ouroboros-wkwebview-", dir=ROOT_PARENT))
    observation = {"status": "error", "errors": []}
    teardown, children = [], []
    try:
        observation = observe(root, out, timeout, report, teardown, children)
    except Exception as exc:  # the run's own failure is evidence, never a pass
        observation["errors"].append(f"{type(exc).__name__}: {exc}")
        report["runner_traceback"] = traceback.format_exc()
    finally:
        for name, proc, container, seconds in reversed(children):
            report.setdefault("teardown", {})[name] = stopped = _stop(proc, container, seconds)
            keep_root = keep_root or bool(stopped["left_running"])
        for close in reversed(teardown):
            close()
        if keep_root:
            (root / "OUROBOROS_RETAINED.txt").write_text("composer clearance check root kept\n", encoding="utf-8")
            report["root"] = str(root)
        else:
            shutil.rmtree(root, ignore_errors=True)
    return _finish(out, report, observation)


# --- driver: one system WKWebView window over the real UI ---------------------------------------------------------------

def _rect(rect) -> list[float]:
    return [float(rect.origin.x), float(rect.origin.y), float(rect.size.width), float(rect.size.height)]


def _page_error(error) -> str:
    info = error.userInfo() or {}
    message = info.get("WKJavaScriptExceptionMessage") or error.localizedDescription()
    line = info.get("WKJavaScriptExceptionLineNumber")
    return f"{message}" + (f" (line {line})" if line else "")


class NativeHost:
    """The NSWindow and WKWebView, touched only on the main thread; the scenario thread waits on bounded calls."""

    def __init__(self, objc, AppKit, Foundation, WebKit, AppHelper):  # noqa: N803 - framework module names
        self.objc, self.AppKit, self.Foundation, self.WebKit, self.AppHelper = objc, AppKit, Foundation, WebKit, AppHelper
        self.trace, self.navigation = [], queue.Queue()
        self.app = AppKit.NSApplication.sharedApplication()
        self.app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
        self.screen = AppKit.NSScreen.mainScreen()
        config = WebKit.WKWebViewConfiguration.alloc().init()
        config.setWebsiteDataStore_(WebKit.WKWebsiteDataStore.nonPersistentDataStore())
        config.userContentController().addUserScript_(
            WebKit.WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
                PAGE_SCRIPT.read_text(encoding="utf-8"), WebKit.WKUserScriptInjectionTimeAtDocumentStart, True))
        rect = Foundation.NSMakeRect(0, 0, WIDTHS[0], START_HEIGHT)
        mask = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
                | AppKit.NSWindowStyleMaskMiniaturizable | AppKit.NSWindowStyleMaskResizable)
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, mask, AppKit.NSBackingStoreBuffered, False)
        self.window.setReleasedWhenClosed_(False)
        self.window.setTitle_("Ouroboros composer clearance check")
        # Above other windows: WebKit pauses animation frames in a covered window.
        self.window.setLevel_(AppKit.NSFloatingWindowLevel)
        self.window.setContentMinSize_(Foundation.NSMakeSize(320, LOWEST_HEIGHT))
        self.webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(rect, config)
        self.delegate = self._navigation_delegate()
        self.webview.setNavigationDelegate_(self.delegate)
        self.window.setContentView_(self.webview)
        visible = self.screen.visibleFrame()
        chrome = self.window.frame().size.height - self.window.contentRectForFrameRect_(self.window.frame()).size.height
        self.widest, self.highest = int(visible.size.width), int(visible.size.height - chrome)
        self.window.makeKeyAndOrderFront_(None)
        self.app.activateIgnoringOtherApps_(True)

    def _navigation_delegate(self):
        events = self.navigation
        try:
            protocols = [self.objc.protocolNamed("WKNavigationDelegate")]
        except Exception:
            protocols = []

        class ComposerCheckNavigation(self.Foundation.NSObject, protocols=protocols):
            def webView_didFinishNavigation_(self, webview, navigation):
                events.put(("finished", None))

            def webView_didFailNavigation_withError_(self, webview, navigation, error):
                events.put(("failed", str(error.localizedDescription())))

            def webView_didFailProvisionalNavigation_withError_(self, webview, navigation, error):
                events.put(("failed", str(error.localizedDescription())))

            def webViewWebContentProcessDidTerminate_(self, webview):
                events.put(("terminated", "the WebContent process ended"))

        return ComposerCheckNavigation.alloc().init()

    def engine(self) -> dict:
        """The engine and its configuration, read on the main thread."""
        bundle = self.Foundation.NSBundle.bundleForClass_(self.WebKit.WKWebView)
        info = bundle.infoDictionary() or {}
        return {"engine": "system WKWebView", "webkit_bundle": str(bundle.bundlePath()),
                "webkit_version": str(info.get("CFBundleVersion")),
                "webkit_short_version": str(info.get("CFBundleShortVersionString")),
                "webview_classes": [cls.__name__ for cls in type(self.webview).__mro__[:3]],
                "data_store_persistent": bool(self.webview.configuration().websiteDataStore().isPersistent()),
                "pyobjc": getattr(self.objc, "__version__", None), "page_zoom": float(self.webview.pageZoom())}

    def window_state(self) -> dict:
        """Screen and window as the window server reports them, read on the main thread inside the run loop: before
        it, a window just ordered front is not yet key and has no visible occlusion state."""
        screen = self.window.screen() or self.screen
        return {"screen": _rect(screen.frame()), "visible": _rect(screen.visibleFrame()),
                "backing_scale": float(screen.backingScaleFactor()), "window": _rect(self.window.frame()),
                "key_window": bool(self.window.isKeyWindow()),
                "occlusion_visible": bool(self.window.occlusionState() & self.AppKit.NSWindowOcclusionStateVisible)}

    def on_main(self, call, timeout: float = 15.0):
        done, box = threading.Event(), {}

        def run():
            try:
                box["value"] = call()
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        self.AppHelper.callAfter(run)
        if not done.wait(timeout):
            raise TimeoutError("the main thread did not take the call")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def page(self, name: str, timeout: float = 45.0, **args):
        """Await window.__composerCheck[name](args) in the page; its JSON answer, or the page's error."""
        done, box, started = threading.Event(), {}, time.monotonic()

        def finished(result, error):
            box.update(result=result, error=error)
            done.set()

        def start():
            # One plain string argument: the call crosses to the WebContent process as JSON text.
            string = self.Foundation.NSString.stringWithString_
            arguments = self.Foundation.NSDictionary.dictionaryWithObject_forKey_(
                string(json.dumps({"name": name, "args": args})), string("call"))
            self.webview.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_(
                "const { name, args } = JSON.parse(call);\n"
                "return JSON.stringify(await window.__composerCheck[name](args));",
                arguments, None, self.WebKit.WKContentWorld.pageWorld(), finished)

        try:
            self.on_main(start)
            if not done.wait(timeout):
                raise TimeoutError(f"page call {name} gave no answer in {timeout:.0f} s")
            if box["error"] is not None:
                raise RuntimeError(f"page call {name}: {_page_error(box['error'])}")
            return None if box["result"] is None else json.loads(str(box["result"]))
        finally:
            self.trace.append([name, round((time.monotonic() - started) * 1000)])

    def load(self, url: str, timeout: float = 60.0) -> None:
        while not self.navigation.empty():
            self.navigation.get_nowait()
        request = self.Foundation.NSURLRequest.requestWithURL_(self.Foundation.NSURL.URLWithString_(url))
        self.on_main(lambda: self.webview.loadRequest_(request))
        try:
            kind, detail = self.navigation.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{url} did not finish loading in {timeout:.0f} s") from None
        if kind != "finished":
            raise RuntimeError(f"loading {url}: {kind}: {detail}")

    def place(self, width: int, height: int) -> None:
        """Content size `width` x `height`, the window's top-left at the screen's visible top-left."""
        def apply():
            visible = self.screen.visibleFrame()
            frame = self.window.frameRectForContentRect_(self.Foundation.NSMakeRect(0, 0, width, height))
            frame.origin.x = visible.origin.x
            frame.origin.y = visible.origin.y + visible.size.height - frame.size.height
            self.window.setFrame_display_(frame, True)
        self.on_main(apply)

    def snapshot(self, path: Path, timeout: float = 20.0) -> dict:
        """The WKWebView's own rendering of what it shows, as PNG."""
        done, box = threading.Event(), {}

        def finished(image, error):
            try:
                if image is None:
                    box["error"] = _page_error(error) if error is not None else "no image"
                    return
                rep = self.AppKit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes(rep.representationUsingType_properties_(self.AppKit.NSBitmapImageFileTypePNG, {})))
                box["pixels"] = [int(rep.pixelsWide()), int(rep.pixelsHigh())]
            except Exception as exc:
                box["error"] = repr(exc)
            finally:
                done.set()

        self.on_main(lambda: self.webview.takeSnapshotWithConfiguration_completionHandler_(None, finished))
        if not done.wait(timeout):
            raise TimeoutError("the snapshot gave no image")
        if "error" in box:
            raise RuntimeError(f"snapshot: {box['error']}")
        return {"file": f"screenshots/{path.name}", "pixels": box["pixels"], "sha256": _sha256(path)}

    def stop(self) -> None:
        """End the application's run loop: stop, then one event so the loop sees it."""
        def end():
            try:
                self.window.orderOut_(None)
            finally:
                self.app.stop_(None)
                self.app.postEvent_atStart_(self.AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(  # noqa: E501
                    self.AppKit.NSEventTypeApplicationDefined, self.Foundation.NSMakePoint(0, 0), 0, 0.0, 0, None, 0, 0,
                    0), True)
        self.AppHelper.callAfter(end)


class Walk:
    """One width's way through Main and the Project room; every measurement lands in the observation at once."""

    def __init__(self, host: NativeHost, report: dict, save, out: Path, width: int):
        self.host, self.report, self.save, self.out, self.width = host, report, save, out, width

    def type(self, feed: str, text: str) -> None:
        if self.host.page("draft", feed=feed) != text:
            self.host.page("fill", feed=feed, text=text)

    def sample(self, surface: str, feed: str, phase: str, composer: str, variant: str = "current") -> None:
        sample_id = f"{self.width}-{surface}-{phase}-{composer}-{variant}"
        start = None
        if variant != "current":
            # A negative collapses the scroll range, so the engine clamps the position; the walk continues from
            # where the positive flow left the transcript, never from that clamp.
            start = self.host.page("position", feed=feed)["scrollTop"]
            self.host.page("variant", css=VARIANTS[variant])
        try:
            settled = self.host.page("settle", feed=feed)
            record = {"id": sample_id, "width": self.width, "surface": surface, "phase": phase, "composer": composer,
                      "variant": variant, "settle": settled, **self.host.page("measure", feed=feed)}
            if (phase, composer) in SHOTS:
                record["screenshot"] = self.host.snapshot(self.out / "screenshots" / f"{sample_id}.png")
        finally:
            if start is not None:
                self.host.page("variant", css="")
                self.host.page("settle", feed=feed)
                self.host.page("restore", feed=feed, scrollTop=start)
                self.host.page("settle", feed=feed)
        if start is not None:
            record["restored"] = {"before": start, "after": self.host.page("position", feed=feed)["scrollTop"]}
        self.report["samples"].append(record)
        self.save()

    def fit(self, surface: str, feed: str) -> None:
        """Size the window so this short history needs exactly the reserve's scroll range (verified, not assumed)."""
        attempts = []
        for _ in range(4):
            self.host.page("settle", feed=feed)
            measured = self.host.page("measure", feed=feed)
            fits = short_history(measured)
            attempts.append({"innerHeight": measured["innerHeight"], "short_history": fits, "end": measured["end"]})
            height = None if fits else fitted_height(measured, LOWEST_HEIGHT, self.host.highest)
            if height is None or height == measured["innerHeight"]:
                break
            self.host.place(self.width, height)
            self.host.page("viewport", width=self.width, height=height)
        self.report["fits"].append({"width": self.width, "surface": surface, "attempts": attempts})

    def history(self, surface: str, feed: str) -> None:
        self.fit(surface, feed)
        for composer in COMPOSERS:
            self.type(feed, MULTILINE if composer == "multiline" else "")
            for variant in ("current", *VARIANTS):
                self.sample(surface, feed, "history", composer, variant)

    def growth(self, surface: str, feed: str, chat_id: int) -> None:
        self.host.page("emit", message=_answer(chat_id))
        for composer in COMPOSERS:
            self.type(feed, MULTILINE if composer == "multiline" else "")
            self.sample(surface, feed, "growth", composer)
        self.type(feed, "")

    def main(self) -> None:
        self.host.page("click", selector='[data-nav-page="chat"]')
        self.host.page("showing", feed=MAIN_FEED, text=MAIN_TEXT)
        self.history("main", MAIN_FEED)
        # The multiline composer stays: Main goes behind Settings with its raised reserve and comes back.
        back = self.host.page("leaveAndReturn", away='[data-nav-page="settings"]', page="#page-settings")
        self.report["returns"].append({"width": self.width, **back})
        for composer in reversed(COMPOSERS):
            self.type(MAIN_FEED, MULTILINE if composer == "multiline" else "")
            self.sample("main", MAIN_FEED, "reopen", composer)
        self.growth("main", MAIN_FEED, 1)

    def project(self, chat_id: int) -> None:
        self.host.page("click", selector=PROJECT_ROW)
        self.host.page("showing", feed=PROJECT_FEED, text=PROJECT_TEXT)
        self.history("project", PROJECT_FEED)
        # Closing the room retires its instance; opening it again renders the saved history anew.
        self.host.page("click", selector="#project-panel-close")
        self.host.page("gone", selector=PROJECT_FEED)
        self.host.page("click", selector=PROJECT_ROW)
        self.host.page("showing", feed=PROJECT_FEED, text=PROJECT_TEXT)
        self.report.setdefault("project_drafts", []).append(
            {"width": self.width, "after_reopen": self.host.page("draft", feed=PROJECT_FEED)})
        for composer in reversed(COMPOSERS):
            self.type(PROJECT_FEED, MULTILINE if composer == "multiline" else "")
            self.sample("project", PROJECT_FEED, "reopen", composer)
        self.growth("project", PROJECT_FEED, chat_id)


def scenario(host: NativeHost, session: dict, report: dict, save) -> None:
    widths = [min(WIDTHS[0], host.widest), WIDTHS[1]]
    if widths[0] < 800:
        raise RuntimeError(f"the screen is {host.widest} px wide; the desktop window needs at least 800")
    report["widths"] = widths
    for index, width in enumerate(widths):
        height = min(START_HEIGHT, host.highest)
        host.place(width, height)
        host.load(session["url"])
        page = host.page("ready", text=MAIN_TEXT, timeout=90.0)
        if index == 0:
            report["page"], report["engine_probe"] = page, host.page("probe")
        report["windows"].append({"width": width, **host.on_main(host.window_state),
                                  **host.page("viewport", width=width, height=height)})
        save()
        walk = Walk(host, report, save, Path(session["out"]), width)
        walk.main()
        walk.project(session["project_chat_id"])


def drive(session_path: Path) -> int:
    session = json.loads(session_path.read_text(encoding="utf-8"))
    out = Path(session["out"])
    report = {"status": "starting", "pid": os.getpid(), "widths": [], "windows": [], "samples": [], "fits": [],
              "returns": [], "errors": []}

    def save():
        report["page_calls"] = list(host.trace) if host is not None else []
        _write_json(out / "observation.json", report)

    host = None
    try:
        import AppKit
        import Foundation
        import objc
        import WebKit
        from PyObjCTools import AppHelper
    except Exception as exc:
        report.update(status="unsupported", reason=f"PyObjC with WebKit is not importable: {exc!r}")
        save()
        return EXIT_CODES["UNSUPPORTED"]
    AppKit.NSApplication.sharedApplication()  # connects to the window server before any screen is asked for
    if AppKit.NSScreen.mainScreen() is None:
        report.update(status="unsupported", reason="no display: NSScreen.mainScreen() is None")
        save()
        return EXIT_CODES["UNSUPPORTED"]
    try:
        host = NativeHost(objc, AppKit, Foundation, WebKit, AppHelper)
        report["native"] = host.engine()
    except Exception as exc:
        report.update(status="error", errors=[f"window setup: {exc!r}"], traceback=traceback.format_exc())
        save()
        return EXIT_CODES["INCONCLUSIVE"]
    save()

    def walk():
        try:
            scenario(host, session, report, save)
            if report["status"] == "starting":
                report["status"] = "observed"
        except Exception as exc:
            if report["status"] != "deadline":
                report["status"] = "error"
            report["errors"].append(f"{type(exc).__name__}: {exc}")
            report["traceback"] = traceback.format_exc()
            try:
                report["failure_screenshot"] = host.snapshot(out / "screenshots" / "failure.png")
            except Exception as shot:
                report["errors"].append(f"no failure screenshot: {shot!r}")
        finally:
            save()
            host.stop()

    def expired():
        if report["status"] == "starting":
            report["status"] = "deadline"
            report["errors"].append(f"the driver's {session['deadline']:.0f} s deadline passed")
            save()
            host.stop()

    AppHelper.callLater(session["deadline"], expired)
    threading.Thread(target=walk, name="composer-check", daemon=True).start()
    host.app.run()
    host.window.close()
    save()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="a new or empty directory for the report and evidence")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds the native window may run")
    parser.add_argument("--keep-root", action="store_true", help="keep the disposable copy and data root")
    parser.add_argument("--identity-only", action="store_true", help="print this checkout's source identity")
    parser.add_argument("--driver", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.driver:
        return drive(args.driver)
    if args.identity_only:
        print(json.dumps({"source": source_identity(REPO), "instrument": instrument_identity()}, indent=2))
        return 0
    if args.out is None:
        parser.error("--out is required")
    # The devtools output boundary, before anything is made: a report inside the checkout would change its identity.
    spec = importlib.util.spec_from_file_location("composer_check_run_roots",
                                                  REPO / "devtools/benchmarks/common/run_roots.py")
    run_roots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_roots)
    try:
        out = run_roots.assert_outside_repo(args.out, REPO)
    except ValueError as exc:
        parser.error(str(exc))
    if out.exists() and any(out.iterdir()):
        parser.error(f"--out {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    return run(out, args.keep_root, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
