#!/usr/bin/env python3
"""Local checks of probe_first_frame.py: no window, no network, no host setting. None of this is a native pixel.

  python -I -B selfcheck_probe.py CANDIDATE     (an interpreter with Pillow and the candidate's runtime dependencies)

The plan's counts; run_case end to end through the candidate's real seams and pages and a real held loopback server,
with a stand-in pywebview and GTK whose engine fetches what a real one would; the close watchdog in its own process;
both OS judgements over synthetic screens, one per failure mode; failed renderer scans; a matrix killed as a step's
ceiling kills it; the workflow's ceilings; bind over fixture repositories. Exit 1 if any check fails.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.parse
import urllib.request

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
RESULTS: list = []


def check(name: str, ok, detail="") -> None:
    RESULTS.append({"check": name, "ok": bool(ok), **({} if ok else {"detail": str(detail)[:3000]})})


def load_probe():
    spec = importlib.util.spec_from_file_location("probe_first_frame", HERE / "probe_first_frame.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.HOST_SETTLE_SECONDS, probe.TAIL_GRACE_SECONDS = 0, 0.5
    return probe


def flavour(probe, windows: bool) -> None:
    probe.WINDOWS = windows
    probe.HOSTS, probe.RUNS = probe.plan_for(windows)
    probe.RENDERER = "edgechromium" if windows else "gtkwebkit2"


# ---- a stand-in pywebview: real HTTP to the held server, frames painted from what the engine has seen ----------

class Event:
    def __init__(self) -> None:
        self.handlers, self.flag = [], threading.Event()

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def set(self) -> None:
        self.flag.set()
        for handler in self.handlers:
            handler()

    def wait(self, timeout=None) -> bool:
        return self.flag.wait(timeout)


class Rgba:
    """``Gdk.RGBA`` as the candidate's GTK path reads and parses it."""

    def __init__(self, colour: str = "#000000") -> None:
        self.parse(colour)

    def parse(self, colour: str) -> bool:
        self.text = colour
        self.red, self.green, self.blue = (int(colour[index:index + 2], 16) / 255 for index in (1, 3, 5))
        return True


def fake_webview(probe, fill: str, painted: str, report: dict, hang_on_close: bool = False, shown: str = None,
                 theme: str = "#f6f5f4", frames_before_request: int = 2, request_after: float = 60):
    """``create_window``/``start`` as run_case uses them. The engine asks once that many frames were grabbed, or that
    many seconds passed, so a loaded machine changes nothing. A frame is ``shown`` until it asks, then what the web
    view's own background was set to, else ``fill``, and ``painted`` once the held fetch returns."""
    state = {"released": False, "requested": False, "web_view": None, "windows": [], "grabs": 0}

    class Window:
        def __init__(self, url=None, html=None, background_color="#FFFFFF", **kwargs) -> None:
            self.url, self.html, self.background_color, self.uid = url, html, background_color, "master"
            self.events = types.SimpleNamespace(**{name: Event() for name in (
                "before_show", "shown", "closed", "closing", "before_load")})
            view = types.SimpleNamespace(set_background_color=lambda colour: state.update(web_view=colour.text))
            self.gui = types.SimpleNamespace(
                renderer=probe.RENDERER, Gdk=types.SimpleNamespace(RGBA=Rgba),
                BrowserView=types.SimpleNamespace(instances={self.uid: types.SimpleNamespace(webview=view)}))
            style = types.SimpleNamespace(lookup_color=lambda name: (name == "theme_bg_color", Rgba(theme)))
            self.native, self.destroyed = types.SimpleNamespace(get_style_context=lambda: style), threading.Event()

        def destroy(self) -> None:
            self.destroyed.set()

    def create_window(title, url=None, html=None, js_api=None, **kwargs):
        state["windows"].append(Window(url, html, **kwargs))
        return state["windows"][-1]

    def start(func=None, **kwargs):
        window = state["windows"][-1]

        def engine() -> None:
            deadline = time.monotonic() + request_after
            while state["grabs"] < frames_before_request and time.monotonic() < deadline:
                time.sleep(0.01)
            target = window.url or re.search(r'src="(http://127\.0\.0\.1:\d+/held\.js)"', window.html).group(1)
            state["requested"] = True
            urllib.request.urlopen(target, timeout=60).read()  # blocks until the probe releases it
            state["released"] = True
            base = target.rsplit("/", 1)[0]
            urllib.request.urlopen(f"{base}/painted?d=" + urllib.parse.quote(json.dumps(report)), timeout=60).read()

        window.events.before_show.set()
        window.events.shown.set()
        threading.Thread(target=engine, daemon=True).start()
        threading.Thread(target=func, daemon=True).start()
        window.destroyed.wait(120)
        if hang_on_close:
            threading.Event().wait()

    def grab(path) -> None:
        image = Image.new("RGB", (640, 400), "#336699")
        colour = painted if state["released"] else (state["web_view"] or fill) if state["requested"] else (shown or fill)
        image.paste(colour, (24, 24, 544, 324))
        image.save(path)
        state["grabs"] += 1

    probe.grab = grab
    probe.window_rects = lambda window, hwnd: {"frame": [24, 24, 544, 324], "client": [24, 24, 544, 324]}
    return types.SimpleNamespace(create_window=create_window, start=start)


def run_case_checks(probe, source: pathlib.Path, tmp: pathlib.Path) -> None:
    flavour(probe, False)
    sys.path.insert(0, str(source))
    from ouroboros import launcher_window_chrome

    launcher_window_chrome._MACOS, launcher_window_chrome._LINUX = False, True  # this Mac stands in for GTK's path
    room, light, dark = dict(launcher_window_chrome.ROOM), "#f6f5f4", "#353535"
    for case, choice, host, theme, fill, painted, report, want in (
            ("control", "system", "light", light, "#ffffff", probe.SENTINEL, {"theme": "", "prefersDark": False}, "#ffffff"),
            ("control", "system", "dark", dark, "#ffffff", probe.SENTINEL, {"theme": "", "prefersDark": True}, "#ffffff"),
            ("control-room", "system", "dark", dark, "#ffffff", probe.SENTINEL, {"theme": "", "prefersDark": True},
             probe.MAGENTA),
            ("control-html", "system", "dark", dark, dark, probe.SENTINEL, {"theme": "dark", "prefersDark": True}, dark),
            ("main", "dark", "dark", dark, "#ffffff", "#0f0f11", {"theme": "dark", "prefersDark": True}, "#0f0f11"),
            ("main", "dark", "light", light, "#808080", "#0f0f11", {"theme": "dark", "prefersDark": False}, "#ffffff"),
            ("preserver", "system", "dark", dark, "#ffffff", "#0f0f11", {"theme": "dark", "prefersDark": True}, "#0f0f11"),
            ("setup", "system", "dark", dark, "#ffffff", "#0f0f11", {"theme": "dark", "prefersDark": True}, "#0f0f11")):
        name, out = f"{case} on a {host} host", tmp / f"case-{case}-{host}"
        webview = fake_webview(probe, fill, painted, report, shown=theme, theme=theme)
        code = probe.run_case(types.SimpleNamespace(out=str(out), source=str(source), case=case, choice=choice), webview)
        launcher_window_chrome.ROOM = dict(room)  # each real case has its own process
        record, teardown = probe.read_json(out / "case.json"), probe.read_json(out / "teardown.json")
        entry = probe.judge_run(case, choice, host, record, {"exit": code, "teardown": teardown}, out)
        check(f"run_case {name}: exit 0, a clean close, frames that can speak", code == 0 and teardown["closed"] and
              not entry["invalid"], {"code": code, "teardown": teardown, "invalid": entry["invalid"]})
        stages = [sample["stage"] for sample in record["samples"]]
        loading = [sample for sample in record["samples"] if sample["stage"] == "loading"]
        check(f"run_case {name}: shown, loading across the request, three held after it, transition, painted",
              [stage for stage in stages if stage != "loading"][:4] == ["shown", "held", "held", "held"] and
              stages[1] == "loading" and stages[-1] == "painted" and "transition" in stages and
              loading[0]["requests"] == 0 and loading[-1]["requests"] >= 1 and
              all(sample["requests"] for sample in record["samples"] if sample["stage"] == "held"), stages)
        check(f"run_case {name}: shown, loading, held and painted each keep their own colours, the page's report kept",
              entry["shown_seen"] == [theme] and entry["pre_document_seen"] == [want] and
              set(entry["loading_seen"]) == {theme, want} and entry["painted_seen"] == [painted] and
              entry["page_reported"] == report and [item["colours"] for item in entry["timeline"]][0] == [theme], entry)
        if case not in probe.CONTROLS:
            check(f"run_case {name}: the candidate's seam opened the web view in the room its theme resolved",
                  not (entry["contradictions"] or entry["opposite_appearance"] or entry["held_contradictions"]), entry)

    def control(case: str) -> dict:
        out = tmp / f"case-{case}-dark"
        return probe.judge_run(case, "system", "dark", probe.read_json(out / "case.json"), {"exit": 0}, out)

    check("run_case control: proves the capture on a host with no magenta to show, its white kept a canvas",
          probe.control_gap(control("control")) == ""
          and probe.held_provenance(control("control")).startswith(f"{probe.HELD_CANVAS}: ['#ffffff']"))
    check("run_case control-room: the candidate's own paint, told magenta, reads as that path's native fill",
          probe.held_provenance(control("control-room")) == probe.NATIVE_FILL)
    frames, probe.LOADING_FRAMES = probe.LOADING_FRAMES, 2
    out = tmp / "case-bounded"
    probe.run_case(types.SimpleNamespace(out=str(out), source=str(source), case="control", choice="system"),
                   fake_webview(probe, "#ffffff", probe.SENTINEL, {"theme": "", "prefersDark": False},
                                frames_before_request=6, request_after=0.8))
    probe.LOADING_FRAMES = frames
    labels = [sample["label"] for sample in probe.read_json(out / "case.json")["samples"]]
    check("run_case: the loading frames before the request are bounded, and the request is still waited for",
          [label for label in labels if re.match(r"loading-\d", label)] == ["loading-00", "loading-01"]
          and "held-2" in labels, labels)
    for case in probe.CASES:  # the documents run_case serves: the candidate's own appearance, held where it must be
        page = probe.page_html(source, 1234, case, "dark")
        wants = ([probe.SENTINEL] if case in probe.CONTROLS else []) + (
            [] if case in ("control", "control-room") else ["ouroboros.theme", "--bg-primary"]) + (
            ['127.0.0.1:1234/held.js"></script>', ".window {"] if probe.CASES[case]["page"] == "html" else [])
        check(f"page_html {case}: {', '.join(wants)}", all(want in page for want in wants) and "1234/painted" in page,
              [want for want in wants if want not in page])
    same_document = probe.page_html(source, 1234, "control-html", "system").replace(
        f"html, body {{ background: {probe.SENTINEL} !important; }}", "")
    check("page_html control-html: the preserver's own document but for the sentinel's one rule",
          same_document == probe.page_html(source, 1234, "preserver", "system"))


def watchdog_check(probe, source: pathlib.Path, tmp: pathlib.Path) -> None:
    out = tmp / "case-hung"
    runner = tmp / "hung_runner.py"
    runner.write_text(
        "import sys, types\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import selfcheck_probe as s\n"
        "probe = s.load_probe(); s.flavour(probe, False); probe.CLOSE_SECONDS = 2\n"
        f"sys.path.insert(0, {str(source)!r})\n"
        "from ouroboros import launcher_window_chrome; launcher_window_chrome._MACOS = False\n"
        "webview = s.fake_webview(probe, '#ffffff', '#0f0f11', {'theme': 'dark', 'prefersDark': False}, True)\n"
        f"args = types.SimpleNamespace(out={str(out)!r}, source={str(source)!r}, case='main', choice='dark')\n"
        "sys.exit(probe.run_case(args, webview))\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", "-B", str(runner)], capture_output=True, text=True, timeout=180)
    teardown = probe.read_json(out / "teardown.json") or {}
    check("watchdog: a close that never returns exits 4, never 0", result.returncode == probe.EXIT_CLOSE_HUNG,
          (result.returncode, result.stderr[-1500:]))
    check("watchdog: teardown.json says the close hung, and the frames were kept before it",
          teardown.get("closed") is False and "had not returned" in teardown.get("note", "") and
          (out / "case.json").is_file(), teardown)
    entry = probe.judge_run("main", "dark", "light", probe.read_json(out / "case.json") or {},
                            {"exit": result.returncode, "teardown": teardown}, out)
    check("watchdog: the run says so in unobserved", any("close hung" in reason for reason in entry["unobserved"]),
          entry["unobserved"])


# ---- synthetic screens for the judgement ------------------------------------------------------------------------

def screen(path: pathlib.Path, client: str, caption) -> dict:
    """A 1024x768 desktop holding one window: a 31 px caption band (Windows) over its client area."""
    image = Image.new("RGB", (1024, 768), "#336699")
    frame, band = [24, 24, 544, 324], 31 if caption else 0
    if caption:
        image.paste(caption, (frame[0], frame[1], frame[2], frame[1] + band))
    image.paste(client, (frame[0], frame[1] + band, frame[2], frame[3]))
    image.save(path)
    return {"frame": frame, "client": [frame[0], frame[1] + band, frame[2], frame[3]]}


def write_case(probe, case_dir: pathlib.Path, case: str, choice: str, host: str, change=None) -> int:
    """The case directory a perfect run leaves. On Linux as GTK 3.24 and WebKitGTK 2.50 showed them and the candidate
    means to: the toolkit's colour when shown and on an html hold, WebKit's white on a base url, the room or a control's
    magenta through the candidate's paint. ``change(record, frames)`` alters it."""
    want = probe.expected(case, choice, host)
    windows, palette = probe.WINDOWS, probe.HOSTS[host]
    toolkit = "#f6f5f4" if palette == "light" else "#353535"
    shown = want["pre_document"] if windows else toolkit
    fill = want["pre_document"] if windows or (probe.CASES[case]["page"] == "url" and case != "control") else (
        "#ffffff" if case == "control" else toolkit)
    frames = {"shown": shown, "loading": fill, "held": fill, "painted": want["painted"], "transition": want["painted"],
              "caption_held": want["caption_held"], "caption_painted": want["caption_painted"]}
    record = {"renderer": probe.RENDERER, "samples": [], "errors": [], "windows_build": 26100 if windows else None,
              "host_palette": palette if windows else "", "background_color_after_seams": fill,
              "page": {"theme": "" if case in ("control", "control-room") else want["page_palette"],
                       "prefersDark": palette == "dark"}}
    if change:
        change(record, frames)
    for index, (label, stage) in enumerate((("shown", "shown"), ("loading-00", "loading"), ("held-0", "held"),
                                            ("held-1", "held"), ("held-2", "held"), ("transition-00", "transition"),
                                            ("painted", "painted"))):
        dark = frames["caption_painted" if stage in ("transition", "painted") else "caption_held"]
        name = f"{index:02d}-{label}.png"
        rects = screen(case_dir / name, frames[stage], ("#202020" if dark else "#f3f3f3") if windows else None)
        record["samples"].append({"label": label, "stage": stage, "png": name,
                                  "requests": int(stage not in ("shown", "loading")),
                                  **rects, **({"caption_attribute": dark} if windows else {})})
    (case_dir / "case.json").write_text(json.dumps(record), encoding="utf-8")
    (case_dir / "teardown.json").write_text(json.dumps({"closed": True, "note": "webview.start returned"}), "utf-8")
    return 0


def changed(**frames):
    """A scenario: the perfect run's case directory, with these frames instead."""
    return lambda probe, case_dir, case, choice, host: write_case(
        probe, case_dir, case, choice, host, lambda record, kept: kept.update(frames))


def run_matrix(probe, tmp: pathlib.Path, name: str, scenario=None, restore_takes=True, tail=None, scan=None):
    """probe.matrix with its processes, registry and renderers stood in for; (exit code, summary, kills asked).
    ``tail["alive"]``: renderers a scenario leaves behind; each disappears once the probe asks to kill it.
    ``scan(tail, kills)``: the renderer scan to use instead of reading ``tail["alive"]``."""
    out, kills, registry, tail = tmp / name, [], {"value": (1, 4), "calls": 0}, tail if tail is not None else {"alive": []}

    def kill(pid: str) -> None:
        kills.append(pid)
        tail["alive"] = [line for line in tail["alive"] if line.split()[0] != pid]

    def launch(command, env, log_path):
        args = dict(zip(command[5::2], command[6::2]))
        case_dir = pathlib.Path(args["--out"])
        return (scenario or {}).get((case_dir.parent.name, args["--case"], args["--choice"]), lambda *a: write_case(*a))(
            probe, case_dir, args["--case"], args["--choice"], case_dir.parent.name)

    def set_app_mode(value):
        before, registry["calls"] = registry["value"], registry["calls"] + 1
        if restore_takes or registry["calls"] <= len(probe.HOSTS):
            registry["value"] = None if value is None else (value, 4)
        return before

    probe.set_app_mode = set_app_mode
    probe.read_app_mode = lambda: "absent" if registry["value"] is None else repr(registry["value"])
    renderers = scan(tail, kills) if scan else lambda: list(tail["alive"])
    code = probe.matrix(types.SimpleNamespace(out=str(out), source=str(tmp)), launch, renderers, kill)
    return code, json.loads((out / "summary.json").read_text(encoding="utf-8")), kills


def judgement_checks(probe, tmp: pathlib.Path) -> None:
    flavour(probe, True)
    code, summary, _ = run_matrix(probe, tmp, "w-good")
    verdict = summary["verdict"]
    check("windows: every run as contracted is a mechanical match, exit 0, the registry restored",
          code == 0 and verdict["status"] == "mechanical_match_visual_review_pending" and len(summary["runs"]) == 18
          and summary["app_mode_restored"] is True, verdict)
    check("windows: each page path's control says its held frames are the window's native fill",
          all(paths == {"url": probe.NATIVE_FILL, "html": probe.NATIVE_FILL}
              for paths in verdict["held_provenance"].values()), verdict["held_provenance"])

    near_black = changed(held="#121212", loading="#121212")  # run 41575443436's dark host, on a url
    code, summary, _ = run_matrix(probe, tmp, "w-dark", {
        ("dark", "control", "system"): changed(shown=probe.MAGENTA, held="#121212", loading="#121212"),
        ("dark", "baseline-main", "system"): near_black, ("dark", "main", "light"): near_black,
        ("dark", "main", "dark"): near_black, ("dark", "main", "system"): near_black,
        ("dark", "setup", "system"): near_black})
    verdict, control = summary["verdict"], summary["runs"][9]
    check("windows: a control shown magenta that the renderer takes over while held stops nothing, its canvas kept unproven",
          code == 3 and len(summary["runs"]) == 18 and "stopped" not in summary and verdict["capture_proof"]["dark"] == "proven"
          and control["shown_seen"] == [probe.MAGENTA] and control["pre_document_seen"] == ["#121212"]
          and verdict["held_provenance"]["dark"]["url"] == f"{probe.HELD_CANVAS}: ['#121212']; shown ['#ff00ff']"
          and verdict["held_provenance"]["dark"]["html"] == probe.NATIVE_FILL
          and (tmp / "w-dark" / "dark" / "setup-system" / "02-held-0.png").is_file()
          and any(text.startswith("dark: frames held on the url path are held canvas") for text in verdict["unobserved"]),
          (code, verdict))
    check("windows: that canvas supports a dark appearance and no exact fill: tolerance 2 keeps #121212 from #0f0f11",
          verdict["status"] == "unknown" and not verdict["contradictions"] and probe.TOLERANCE == 2
          and not probe.close("#121212", "#0f0f11")
          and any("dark/main-dark: held ['#121212'] is the dark host's appearance" in text for text in verdict["appearance"])
          and any("dark/main-dark: held frames no control showed" in text for text in verdict["unobserved"]), verdict)
    code, summary, _ = run_matrix(probe, tmp, "w-dark-white", {
        ("dark", "control", "system"): changed(shown=probe.MAGENTA, held="#121212"),
        ("dark", "main", "dark"): changed(held="#ffffff")})
    check("windows: over that same canvas a url window that holds white on a dark host fails: its appearance was seen",
          code == 1 and any("dark/main-dark: held frames show ['#ffffff'] on a dark host" in text
                            for text in summary["verdict"]["contradictions"]), summary["verdict"])
    code, summary, _ = run_matrix(probe, tmp, "w-crossover", {("dark", "main", "dark"): changed(loading="#ffffff"),
                                                              ("dark", "setup", "system"): changed(shown="#ffffff")})
    verdict = summary["verdict"]
    check("windows: a white loading frame on a dark host keeps the run from a match, and a white shown frame only advises",
          code == 3 and not verdict["contradictions"] and any(
              "dark/main-dark: loading frames show ['#ffffff']" in text for text in verdict["unobserved"])
          and any("dark/setup-system: shown frames show ['#ffffff']" in text for text in verdict["advisories"])
          and not any("setup-system" in text for text in verdict["unobserved"]), verdict)
    for name, change in (("saw no page", {"painted": "#ffffff"}), ("did not hold", {"held": probe.SENTINEL}),
                         ("showed its page early", {"loading": probe.SENTINEL})):
        code, summary, _ = run_matrix(probe, tmp, f"w-stop-{name.split()[-1]}", {("light", "control", "system"):
                                                                                changed(**change)})
        verdict = summary["verdict"]
        check(f"windows: a url control that {name} stops the matrix, unknown, its images kept",
              code == 3 and len(summary["runs"]) == 1 and "stopped" in summary and verdict["capture_proof"]["light"] != "proven"
              and (tmp / f"w-stop-{name.split()[-1]}" / "light" / "control-system" / "02-held-0.png").is_file(), verdict)

    def old_colour(probe, case_dir, case, choice, host):
        return write_case(probe, case_dir, case, choice, host,
                          lambda record, frames: frames.update(held="#ffffff", caption_held=False))
    code, summary, _ = run_matrix(probe, tmp, "w-failed", {("dark", "main", "dark"): old_colour})
    verdict = summary["verdict"]
    check("windows: a main window that opens white on a dark host fails, naming the frame and the caption",
          code == 1 and verdict["status"] == "failed" and any("pre-document frames show ['#ffffff']" in text
                                                              for text in verdict["contradictions"])
          and any("held-2: caption dark=True" in text for text in verdict["contradictions"]), verdict)
    code, summary, _ = run_matrix(probe, tmp, "w-baseline", {("light", "baseline-main", "system"): old_colour})
    verdict = summary["verdict"]
    check("windows: a baseline that misbehaves leaves the comparison unproven, never a failure",
          code == 3 and not verdict["contradictions"] and any("old-colour comparison unproven" in text
                                                              for text in verdict["unobserved"]), verdict)

    def hung(probe, case_dir, case, choice, host):
        write_case(probe, case_dir, case, choice, host)
        (case_dir / "teardown.json").write_text(json.dumps({"closed": False, "note": "had not returned 30 s"}))
        return probe.EXIT_CLOSE_HUNG
    code, summary, _ = run_matrix(probe, tmp, "w-hung", {("light", "preserver", "system"): hung})
    check("windows: a hung close is unknown, named twice (exit and teardown), never clean",
          code == 3 and any("exit 4: its window's close hung" in text for text in summary["verdict"]["unobserved"])
          and any("window close hung" in text for text in summary["verdict"]["unobserved"]), summary["verdict"])

    tail = {"alive": []}

    def timeout(probe, case_dir, case, choice, host):  # no record, and a renderer the killed tree left behind
        tail["alive"] = ["4242 msedgewebview2.exe"]
        return "timeout after 420 s; its process tree was killed"
    code, summary, kills = run_matrix(probe, tmp, "w-timeout", {("dark", "main", "light"): timeout}, tail=tail)

    unobserved = summary["verdict"]["unobserved"]
    check("windows: a timeout, its missing record and a renderer tail stay three facts, unknown",
          code == 3 and any("exit timeout" in text for text in unobserved) and any("no case record" in text
                                                                                   for text in unobserved)
          and any("renderer processes outlived" in text for text in unobserved) and kills == ["4242"], (unobserved, kills))
    code, summary, _ = run_matrix(probe, tmp, "w-restore", restore_takes=False)
    check("windows: an app mode that was not restored is unknown", code == 3 and summary["app_mode_restored"] is False
          and any("not restored" in text for text in summary["verdict"]["unobserved"]), summary["verdict"])

    def wrong_renderer(probe, case_dir, case, choice, host):
        return write_case(probe, case_dir, case, choice, host,
                          lambda record, frames: (record.update(renderer="mshtml"), frames.update(held="#ffffff")))
    code, summary, _ = run_matrix(probe, tmp, "w-renderer", {("dark", "setup", "system"): wrong_renderer})
    check("windows: wrong frames on the wrong renderer are unknown, not failed",
          code == 3 and not summary["verdict"]["contradictions"]
          and any("renderer 'mshtml'" in text for text in summary["verdict"]["unobserved"]), summary["verdict"])

    white = changed(held="#ffffff")  # as a preserver's held frames, and as a control's canvas
    code, summary, _ = run_matrix(probe, tmp, "w-html-native", {("dark", "preserver", "system"): white})
    verdict = summary["verdict"]
    check("windows: under its own native-fill control, a preserver that holds white on a dark host fails",
          code == 1 and any("dark/preserver-system: pre-document frames show ['#ffffff']" in text
                            for text in verdict["contradictions"]), verdict)
    code, summary, _ = run_matrix(probe, tmp, "w-html-canvas", {("dark", "control-html", "system"): white,
                                                                ("dark", "preserver", "system"): white})
    verdict = summary["verdict"]
    check("windows: the same frames when that path's control shows a canvas: unknown provenance kept, never failed",
          code == 3 and not verdict["contradictions"] and len(summary["runs"]) == 18 and "stopped" not in summary
          and verdict["held_provenance"]["dark"]["html"] == f"{probe.HELD_CANVAS}: ['#ffffff']; shown ['#ff00ff']"
          and verdict["held_provenance"]["light"]["html"] == probe.NATIVE_FILL
          and verdict["capture_proof"]["dark"] == "proven"
          and any("dark/preserver-system: held frames no control showed" in text for text in verdict["unobserved"])
          and any(text.startswith("dark: frames held on the html path are held canvas")
                  for text in verdict["unobserved"]), verdict)
    code, summary, _ = run_matrix(probe, tmp, "w-html-canvas-match", {("dark", "control-html", "system"): white})
    verdict = summary["verdict"]
    check("windows: a preserver whose frames match over a canvas control is still unknown: the fill was not seen",
          code == 3 and not verdict["contradictions"]
          and "dark/preserver-system" not in json.dumps(verdict["unobserved"])
          and any(text.startswith("dark: frames held on the html path") for text in verdict["unobserved"])
          and any("dark/preserver-system: held ['#0f0f11'] is the dark host's appearance" in text
                  for text in verdict["appearance"]), verdict)
    code, summary, _ = run_matrix(probe, tmp, "w-html-canvas-painted", {("dark", "control-html", "system"): white,
                                                                        ("dark", "preserver", "system"):
                                                                        changed(painted="#ff0000")})
    check("windows: a canvas control leaves a painted frame's contradiction a failure: it read the page, not the fill",
          code == 1 and any("dark/preserver-system: painted frame shows" in text
                            for text in summary["verdict"]["contradictions"]), summary["verdict"])
    code, summary, _ = run_matrix(probe, tmp, "w-html-blind", {("dark", "control-html", "system"):
                                                               changed(painted="#ffffff"),
                                                               ("dark", "preserver", "system"):
                                                               changed(painted="#ff0000"),
                                                               ("dark", "main", "dark"): changed(held="#ffffff")})
    verdict = summary["verdict"]
    check("windows: an html= control that saw no page unproves that path alone: the url main still fails",
          code == 1 and verdict["held_provenance"]["dark"]["html"].startswith(probe.UNPROVEN)
          and verdict["contradictions"] and all("dark/main-dark" in text for text in verdict["contradictions"])
          and any("dark/preserver-system: on an unproven capture" in text for text in verdict["unobserved"]), verdict)

    flavour(probe, False)
    code, summary, _ = run_matrix(probe, tmp, "l-good")
    verdict = summary["verdict"]
    check("linux: every run observed is exit 0, with one observation line per run and nothing contradicted",
          code == 0 and verdict["status"] == "observed_visual_review_pending" and len(summary["runs"]) == 24
          and len(verdict["observations"]) == 24 and not verdict["contradictions"] and not verdict["unobserved"], verdict)
    check("linux: the base url window's white and an html hold's toolkit colour stay canvases; the candidate's paint is the fill",
          all(paths["url"].startswith(f"{probe.HELD_CANVAS}: ['#ffffff']") and paths["html"].startswith(probe.HELD_CANVAS)
              and paths["url-room"] == probe.NATIVE_FILL for paths in verdict["held_provenance"].values())
          and any("dark/preserver-system: held ['#353535'] is the dark host's appearance" in text
                  for text in verdict["appearance"])
          and not any("main-" in text for text in verdict["appearance"]), verdict)
    unpainted = {(host, case, choice): changed(held="#ffffff", loading="#ffffff") for host in probe.HOSTS
                 for case, choice in (("control-room", "system"), ("main", "light"), ("main", "dark"),
                                      ("main", "system"), ("setup", "system"))}
    code, summary, _ = run_matrix(probe, tmp, "l-unpainted", unpainted)
    verdict = summary["verdict"]
    check("linux: a candidate whose paint changes nothing fails on the dark hosts and says its control saw no fill",
          code == 1 and verdict["status"] == "failed"
          and any("dark/main-dark: held frames show ['#ffffff'] on a dark host" in text for text in verdict["contradictions"])
          and any("prefer-dark/setup-system: held frames show ['#ffffff']" in text for text in verdict["contradictions"])
          and not any(text.startswith("light/") for text in verdict["contradictions"])
          and any(text.startswith("light: frames held on the url-room path are held canvas") for text in verdict["unobserved"]),
          verdict)
    only_light = {key: scenario for key, scenario in unpainted.items() if key[0] == "light"}
    code, summary, _ = run_matrix(probe, tmp, "l-unpainted-light", only_light)
    check("linux: white on a light host is that host's appearance, so without its control's magenta the run is unknown",
          code == 3 and not summary["verdict"]["contradictions"], summary["verdict"])
    code, summary, _ = run_matrix(probe, tmp, "l-wrong-room", {("dark", "main", "dark"): changed(held="#353535")})
    check("linux: under the paint's own magenta a url window's held frames must be its room, within tolerance",
          code == 1 and any("dark/main-dark: pre-document frames show ['#353535'], expected #0f0f11" in text
                            for text in summary["verdict"]["contradictions"]), summary["verdict"])
    code, summary, _ = run_matrix(probe, tmp, "l-html-magenta", {("light", "control-html", "system"):
                                                                 changed(held=probe.MAGENTA)})
    check("linux: an html= control that shows its magenta reads as that path's native fill, the url path unchanged",
          code == 1 and summary["verdict"]["held_provenance"]["light"]["html"] == probe.NATIVE_FILL
          and summary["verdict"]["held_provenance"]["light"]["url"].startswith(probe.HELD_CANVAS)
          and any("light/preserver-system: pre-document frames show ['#f6f5f4']" in text
                  for text in summary["verdict"]["contradictions"]), summary["verdict"])
    code, summary, _ = run_matrix(probe, tmp, "l-html-white", {("dark", "preserver", "system"): changed(held="#ffffff")})
    check("linux: an html hold that shows white on a dark host, over a canvas, is unknown and said: the probe adds that hold",
          code == 3 and not summary["verdict"]["contradictions"]
          and any("dark/preserver-system: on a hold the probe adds" in text for text in summary["verdict"]["unobserved"]),
          summary["verdict"])
    code, summary, _ = run_matrix(probe, tmp, "l-html-sentinel", {("light", "control-html", "system"):
                                                                  changed(held=probe.SENTINEL)})
    verdict = summary["verdict"]
    check("linux: an html= control whose hold did not hold is unknown, the matrix going on through the url path",
          code == 3 and len(summary["runs"]) == 24
          and verdict["held_provenance"]["light"]["html"].startswith(probe.UNPROVEN)
          and any("html path are unproven" in text for text in verdict["unobserved"]), verdict)
    for name, change in (("painted frame without the sentinel", lambda record, frames: frames.update(painted="#ffffff")),
                         ("held frames showing the sentinel", lambda record, frames: frames.update(held=probe.SENTINEL))):
        def control(probe, case_dir, case, choice, host, change=change):
            return write_case(probe, case_dir, case, choice, host, change)
        code, summary, _ = run_matrix(probe, tmp, f"l-{name.split()[0]}-{len(RESULTS)}", {("light", "control", "system"): control})
        check(f"linux: a control with {name} stops the matrix, unknown", code == 3 and len(summary["runs"]) == 1,
              summary["verdict"])


def failing_scan(from_call: int):
    """``scan`` for run_matrix: reads the tail, and from this call on fails as tasklist would."""
    def scan(tail, _kills):
        calls = itertools.count(1)

        def read():
            if next(calls) >= from_call:
                raise subprocess.CalledProcessError(1, ["tasklist"], "", "ERROR: scan unavailable")
            return list(tail["alive"])
        return read
    return scan


def scan_failure_checks(probe, tmp: pathlib.Path) -> None:
    """A renderer scan that cannot be read: no run is lost, nothing reads clean, no later case runs, the trace kept."""
    flavour(probe, True)
    trace = io.StringIO()
    with contextlib.redirect_stderr(trace):
        code, summary, _ = run_matrix(probe, tmp, "w-scan-before", scan=failing_scan(1))
    verdict = summary["verdict"]
    check("scan before the matrix: unknown, no case and no host change, the error kept, never an empty list",
          code == 3 and summary["runs"] == [] and summary["renderers_before"] is None
          and "scan unavailable" in summary["stopped"] and "app_mode_original" not in summary
          and not (tmp / "w-scan-before" / "light").exists() and "CalledProcessError" in trace.getvalue()
          and any("0 of 18 runs recorded: no case ran" in text for text in verdict["unobserved"])
          and not any("not restored" in text for text in verdict["unobserved"]), (code, summary))

    trace = io.StringIO()
    with contextlib.redirect_stderr(trace):  # scan 1 is before the matrix, 2 after the control, 3 after control-html
        code, summary, _ = run_matrix(probe, tmp, "w-scan-after", scan=failing_scan(3))
    verdict, run, out = summary["verdict"], summary["runs"][-1], tmp / "w-scan-after" / "light"
    check("scan after a case: its exit and judgement kept, its tail unread and not empty, the matrix stops, unknown",
          code == 3 and verdict["status"] == "unknown" and len(summary["runs"]) == 2 and run["exit"] == 0
          and run["case"] == "control-html" and run["pre_document_seen"] == [probe.MAGENTA]
          and run["tail"]["alive"] is None and run["tail"]["left"] is None
          and "scan unavailable" in run["tail"]["scan_error"] and "renderers not read" in summary["stopped"]
          and any("renderer scan failed" in text for text in verdict["unobserved"])
          and not (out / "baseline-main-system").exists(), (code, summary))
    check("scan after a case: the frames, case record and trace stay, and the host is restored",
          (out / "control-html-system" / "02-held-0.png").is_file()
          and (out / "control-html-system" / "case.json").is_file()
          and "CalledProcessError" in trace.getvalue() and summary["app_mode_restored"] is True, trace.getvalue())

    with contextlib.redirect_stderr(io.StringIO()):  # dark's main-dark is run 16; scan 18 follows the run after it
        code, summary, _ = run_matrix(probe, tmp, "w-scan-failed", {("dark", "main", "dark"):
                                                                    changed(held="#ffffff", caption_held=False)},
                                      scan=failing_scan(18))
    verdict = summary["verdict"]
    check("scan after a case: an earlier contradiction stays failed beside the unread scan",
          code == 1 and verdict["status"] == "failed" and len(summary["runs"]) == 17
          and any("dark/main-dark: pre-document frames show ['#ffffff']" in text for text in verdict["contradictions"])
          and any("renderer scan failed" in text for text in verdict["unobserved"]), (code, verdict))

    def after_a_kill(tail, kills):
        def read():
            if kills:
                raise subprocess.TimeoutExpired(["tasklist"], 30, stderr="ERROR: still busy")
            return list(tail["alive"])
        return read

    tail = {"alive": []}

    def leaves_a_renderer(probe, case_dir, case, choice, host):
        tail["alive"] = ["4242 msedgewebview2.exe"]
        return write_case(probe, case_dir, case, choice, host)
    with contextlib.redirect_stderr(io.StringIO()):
        code, summary, kills = run_matrix(probe, tmp, "w-scan-kill",
                                          {("light", "baseline-main", "system"): leaves_a_renderer},
                                          tail=tail, scan=after_a_kill)
    run, unobserved = summary["runs"][-1], summary["verdict"]["unobserved"]
    check("scan after a kill: the renderer named, the kill asked, what survived it unread, never a clean kill",
          code == 3 and kills == ["4242"] and run["tail"]["alive"] == ["4242 msedgewebview2.exe"]
          and run["tail"]["left"] is None and "still busy" in run["tail"]["scan_error"] and len(summary["runs"]) == 3
          and any("alive after kill: unread" in text for text in unobserved), (run, unobserved))

    flavour(probe, False)  # no restore to wait for, so a last run's checkpoint alone must keep this from clean
    seen = []

    def watching(tail, _kills):
        def read():
            seen.append(probe.read_json(tmp / "l-scan-order" / "verdict.json"))
            return list(tail["alive"])
        return read
    code, summary, _ = run_matrix(probe, tmp, "l-scan-order", scan=watching)
    last = seen[-1] or {}
    check("scan order: each run is on disk before its scan, and not clean until that scan ends",
          code == 0 and seen[0] is None and "1 of 24 runs recorded" in json.dumps(seen[1])
          and last.get("status") == "unknown"
          and any("renderer scan after this case had not finished" in text for text in last.get("unobserved", []))
          and summary["verdict"]["status"] == "observed_visual_review_pending", (code, last))


def killed(*_) -> None:
    os._exit(137)


def killed_step_checks(probe, tmp: pathlib.Path) -> None:
    """A matrix killed as a step's ceiling kills it runs no ``finally``: what it left is all its report can read."""
    for name, earlier, status, wants in (
            ("w-killed", "{}", "unknown", ["Matrix step: failure.", "17 of 18 runs recorded: the matrix ended early"]),
            ("w-killed-failed", "{('dark', 'main', 'dark'): s.changed(held='#ffffff', caption_held=False)}", "failed",
             ["dark/main-dark: pre-document frames show", "17 of 18 runs recorded: the matrix ended early"])):
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c",
             "import pathlib, sys\n"
             f"sys.path.insert(0, {str(HERE)!r})\n"
             "import selfcheck_probe as s\n"
             "probe = s.load_probe(); s.flavour(probe, True)\n"
             f"s.run_matrix(probe, pathlib.Path({str(tmp)!r}), {name!r}, "
             f"{{**{earlier}, ('dark', 'setup', 'system'): s.killed}})\n"],
            capture_output=True, text=True, timeout=180)
        flavour(probe, True)
        with contextlib.redirect_stdout(io.StringIO()):
            probe.summarize(types.SimpleNamespace(out=str(tmp / name), name="windows", outcome="failure"))
        report = (tmp / name / "result-summary.md").read_text(encoding="utf-8")
        check(f"step killed at its ceiling: the report reads {status} from what the matrix left, no verdict of its own",
              result.returncode == 137 and f"First frame, windows: {status}" in report
              and "says nothing of the product" in report and all(want in report for want in wants)
              and "had not finished" not in report, (result.returncode, result.stderr[-1500:], report))


def workflow_checks() -> None:
    """Every step has a ceiling, inner deadlines fire first, and the job budgets stay what they were."""
    text = (HERE.parents[1] / ".github" / "workflows" / "first-frame-qualification.yml").read_text(encoding="utf-8")
    for job in re.split(r"(?m)^  (?=[\w-]+:\n    runs-on:)", text)[1:]:
        name = job.split(":")[0]
        budget = int(re.search(r"(?m)^    timeout-minutes: (\d+)$", job).group(1))
        ceilings = [int(minutes) for minutes in re.findall(r"(?m)^        timeout-minutes: (\d+)$", job)]
        check(f"workflow {name}: every step has a ceiling, and their sum leaves its {budget} minutes a margin",
              len(ceilings) == len(re.findall(r"(?m)^      - ", job)) and sum(ceilings) + 2 <= budget
              and budget == {"windows-webview2": 75, "linux-gtk-xvfb": 90}[name], (budget, ceilings))

    def ceiling(step: str) -> int:
        return int(re.search(rf"(?ms)^      - name: {step}.*?^        timeout-minutes: (\d+)$", text).group(1))
    inner = int(re.search(r"timeout --kill-after=30s (\d+)m dbus-run-session", text).group(1))
    deadline, per_test = map(int, re.search(r"--timeout (\d+) --.*?--timeout=(\d+)", text, re.S).groups())
    check("workflow linux: the matrix's process group ends inside its ceiling, safe_test's deadline inside the wheel's",
          inner + 1 < ceiling("Native matrix on X11") and deadline + 60 < 60 * ceiling("Card-aimed wheel guard")
          and 2 * per_test < deadline, (inner, deadline, per_test))


def renderer_checks(probe) -> None:
    from unittest.mock import patch

    for windows, code in ((True, 1), (False, 2)):
        flavour(probe, windows)
        surfaced = False
        def invoke(command, **kwargs):
            result = subprocess.CompletedProcess(command, code, "", "scan unavailable")
            if kwargs.get("check"):
                result.check_returncode()
            return result
        with patch.object(probe.subprocess, "run", side_effect=invoke):
            try:
                probe.list_renderers()
            except subprocess.CalledProcessError as exc:
                surfaced = exc.returncode == code
        check(f"renderer scan: {'tasklist' if windows else 'pgrep'} failure is not empty", surfaced)
    flavour(probe, False)
    with patch.object(probe.subprocess, "run", return_value=subprocess.CompletedProcess(["pgrep"], 1, "", "")):
        check("renderer scan: pgrep exit 1 means no match", probe.list_renderers() == [])


def bind_checks(probe, tmp: pathlib.Path) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "GITHUB_"))}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

    def git(repo, *arguments) -> str:
        return subprocess.run(["git", "-c", "user.name=selfcheck", "-c", "user.email=selfcheck@invalid",
                               "-c", "commit.gpgsign=false", *arguments], cwd=repo, env=env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def fixture(name: str, touch_product: bool):
        repo = tmp / name
        (repo / "ouroboros").mkdir(parents=True)
        (repo / "ouroboros" / "x.py").write_text("x = 1\n")
        git(repo, "init", "--quiet")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "candidate")
        sha, tree = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "HEAD^{tree}")
        equipment = repo / "devtools" / "first_frame_qualification"
        equipment.mkdir(parents=True)
        (equipment / "probe_first_frame.py").write_bytes((HERE / "probe_first_frame.py").read_bytes())
        (equipment / "inputs.json").write_text(json.dumps({"candidate": {"sha": sha, "tree": tree}}))
        if touch_product:
            (repo / "ouroboros" / "x.py").write_text("x = 2\n")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "equipment")
        result = subprocess.run([sys.executable, "-I", "-B", str(equipment / "probe_first_frame.py"), "bind",
                                 "--equipment-repo", str(repo), "--candidate", str(tmp / f"{name}-candidate"),
                                 "--evidence", str(tmp / f"{name}-evidence")], env=env, capture_output=True, text=True)
        return result, sha

    bound = probe.read_json(HERE / "inputs.json") or {}
    check("bind: the inputs beside this selfcheck bind a commit and a tree: it runs on the real ones, not a placeholder",
          all(probe.HEX40.match(str(bound.get("candidate", {}).get(key, ""))) for key in ("sha", "tree")), bound)
    unfilled = tmp / "unfilled"  # its own probe beside deliberately unfilled inputs: this copy's own are bound
    unfilled.mkdir()
    (unfilled / "probe_first_frame.py").write_bytes((HERE / "probe_first_frame.py").read_bytes())
    (unfilled / "inputs.json").write_text(json.dumps({"candidate": {"sha": "FILL_IN", "tree": "FILL_IN"}}))
    result = subprocess.run([sys.executable, "-I", "-B", str(unfilled / "probe_first_frame.py"), "bind",
                             "--equipment-repo", str(tmp), "--candidate", str(tmp / "unbound"),
                             "--evidence", str(tmp / "unbound-ev")], env=env, capture_output=True, text=True)
    check("bind: an unfilled inputs.json refuses with 2 and runs nothing", result.returncode == 2 and
          "does not bind" in result.stderr and not (tmp / "unbound").exists() and not (tmp / "unbound-ev").exists(),
          result.stderr)
    result, sha = fixture("bound", False)
    identity = probe.read_json(tmp / "bound-evidence" / "identity.json") or {}
    check("bind: the candidate alone, at its commit and tree, with identity.json", result.returncode == 0 and
          identity.get("candidate", {}).get("sha") == sha and git(tmp / "bound-candidate", "rev-parse", "HEAD") == sha
          and not (tmp / "bound-candidate" / "devtools").exists(), (result.returncode, result.stderr))
    result, _ = fixture("touched", True)
    check("bind: an equipment commit that changes product refuses with 2", result.returncode == 2 and
          "does more than add equipment: M ouroboros/x.py" in result.stderr, result.stderr)


def main() -> int:
    source = pathlib.Path(sys.argv[1]).resolve()
    probe = load_probe()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ff-selfcheck-"))
    plans = [probe.plan_for(flag) for flag in (True, False)]
    check("plan: Windows 2 hosts x 9 runs, Linux 3 hosts x 8, each host's url control first, then its html= control",
          [len(hosts) * len(runs) for hosts, runs in plans] == [18, 24] and all(
              [case for case, _ in runs[:2]] == ["control", "control-html"] and all(
                  case in probe.CASES for case, _ in runs) for _, runs in plans), plans)
    for section, run in (("run_case", lambda: run_case_checks(probe, source, tmp)),
                         ("watchdog", lambda: watchdog_check(probe, source, tmp)),
                         ("judgement", lambda: judgement_checks(probe, tmp)),
                         ("renderer scans", lambda: renderer_checks(probe)),
                         ("failed renderer scans", lambda: scan_failure_checks(probe, tmp)),
                         ("killed step", lambda: killed_step_checks(probe, tmp)),
                         ("workflow ceilings", workflow_checks), ("bind", lambda: bind_checks(probe, tmp))):
        try:
            run()
        except Exception:  # a broken section is a failed check, never a skipped one
            import traceback

            check(f"{section}: the section ran", False, traceback.format_exc())
    failed = [result for result in RESULTS if not result["ok"]]
    print(json.dumps(RESULTS, indent=1))
    print(f"{len(RESULTS)} checks, {len(failed)} failed; scratch {tmp}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
