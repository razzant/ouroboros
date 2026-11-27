"""Card gallery: Ouroboros's real card modules in their rare states, on one loopback test page.

A developer picks a card family and a state (an error, a long wait, an empty answer, a long body) and sees what the
owner would see: the product's own modules and stylesheets draw it from recorded producer output. Only this test
module serves the page; the product has no route, no extra static folder and no setting for it.

What it serves, on 127.0.0.1 only:

- ``/`` the stand: ``web/index.html``'s own head (its ``<html class="ouro-ui">`` and the actual order of its sheets
  and scripts) with the stand's chrome instead of the app; ``?scene=<family>.<state>&theme=dark|light`` opens one
  state, ``?bare=1`` the same document with nothing mounted (for a test that mounts its own component). A scene of
  the setup wizard (``SETUP``) gets the head of the wizard's own document, ``web/onboarding_template.html``;
- ``/static/<path>`` and ``/onboarding/static/<path>`` the files of ``web/`` as the product serves them
  (``web/tests`` excluded), no listing;
- ``/gallery/stand.js`` and ``/gallery/stand.css`` from ``tests/card_gallery/``;
- ``/gallery/corpus/<name>.json`` for the three named corpora below and nothing else;
- ``/api/...`` the producers' recorded answers for the open scene (its ``X-Gallery-Scene`` and ``X-Gallery-Step``
  request headers, which the page sets on its own calls; two tabs therefore never share a scene), a 404 for any other
  read, and a 405 for any write: nothing but a read is answered, the sign-in card's start answered from its recorded
  job inside the page, never posted here. Every request is kept in ``GalleryServer.requests`` as ``{"method", "path",
  "scene", "step", "status"}``: the header values, ``scene`` verbatim (empty when absent) and ``step`` the parsed
  integer (0 when absent or not one).

Corpora (never copied here): ``tests/fixtures/card_gallery.json`` (``scripts/generate_card_gallery_fixture.py``),
``web/tests/fixtures/work_projection_cases.json`` and ``web/tests/fixtures/consciousness_contract.json``. ``card_corpus``
(``--card-corpus``) reads one explicitly named file instead of the first, to check a producer's draft before it lands.
Every corpus is read once when the server starts: the corpus endpoint and the API answers serve that one snapshot for the
server's lifetime, so an edited draft shows up on the next ``serve``/restart, never inside a running stand.

Commands (Python with Playwright; the browser is Playwright's own headless build, never an installed one)::

    python -m tests.card_gallery_stand serve [--port 0] [--card-corpus PATH]
    python -m tests.card_gallery_stand look --out DIR [--scenes a.b,...] [--engines chromium,webkit]
        [--themes dark,light] [--viewports 1440x900,390x844]

``look`` writes, per scene, engine, theme and viewport, a first-screen PNG at device scale 2, a crop of the card and
the ``ui_design`` skill's ``references/measure.js`` report for it (read from the skill, not copied), plus
``report.json``. It validates every asked scene id against the stand's own list before it takes any labelled image, and
a scene that faults, shows something else than asked or measures another viewport ends the run instead of labelling a
frame — the page itself faults an address naming no scene, saying so on it, and never draws another scene in its place;
each frame records the scene the page reports and the viewport measure.js measured, and the measurement is taken at the
initial screen, before the card crop scrolls. Its frames are browser frames, not the macOS app window. The ``ui_look``
tool looks at the stand through its own entry point when its runtime facts name the stand's port: the ``All scenes``
links are the places its stand walks (``data-nav-page``; only a scene drawn and ready carries
``aria-current="page"``).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ASSETS = Path(__file__).resolve().parent / "card_gallery"
MEASURE_JS = REPO / "skills" / "ui_design" / "references" / "measure.js"
CORPORA = {
    "card_gallery": REPO / "tests" / "fixtures" / "card_gallery.json",
    "work_projection_cases": WEB / "tests" / "fixtures" / "work_projection_cases.json",
    "consciousness_contract": WEB / "tests" / "fixtures" / "consciousness_contract.json",
}
STAND_FILES = {"stand.js": "text/javascript; charset=utf-8", "stand.css": "text/css; charset=utf-8"}
# The card corpus's families whose scenes read their own recorded answers.
RECORDED = ("quiz", "living_card", "reviews", "widgets", "login", "skills", "task_card", "agents", "share", "connection",
            "provider_card")
# Families drawn in the setup wizard's document: its head (its sheets: onboarding.css too), not the app's.
SETUP = ("connection", "provider_card")
STATIC_PREFIXES = ("/static/", "/onboarding/static/")  # both are web/, as server_web.py serves them
DEFAULT_VIEWPORTS = ((1440, 900), (390, 844))
# Frames that read the same on every machine: the recorded data's language and clock, not the host's.
FRAME_FACTS = {"locale": "en-US", "timezone_id": "UTC"}


def shell_html(page: str = "index.html") -> str:
    """The stand's document: ``web/<page>`` (the app's index, or the setup wizard's template) through its ``</head>``
    unchanged, then the stand instead of the app."""
    index = (WEB / page).read_text(encoding="utf-8")
    head, sep, _ = index.partition("</head>")
    if not sep or 'class="ouro-ui"' not in head:
        raise RuntimeError(f"web/{page} no longer has the head the stand repeats")
    head = re.sub(r"<title>[^<]*</title>", "<title>Ouroboros card gallery (test stand)</title>", head, count=1)
    return (f'{head}    <link rel="stylesheet" href="/gallery/stand.css">\n</head>\n'
            '<body class="gallery-body">\n<div id="gallery" class="gallery-root" data-gallery-root></div>\n'
            '<script type="module" src="/gallery/stand.js"></script>\n</body>\n</html>\n')


class RawText:
    """A stand answer whose body is exact text, not JSON (a letter's whole words as the task-file route serves them)."""

    def __init__(self, data: bytes) -> None:
        self.data = data


class Corpus:
    """The recorded answers the stand gives, looked up by the open scene.

    One immutable snapshot per server lifetime: every corpus file is read once here, and both the corpus endpoint and
    the API answers serve these bytes; a changed file on disk needs a new server (a ``serve`` restart) to show up."""

    def __init__(self, card_corpus: Path | None = None):
        self.paths = dict(CORPORA)
        if card_corpus is not None:
            self.paths["card_gallery"] = Path(card_corpus).resolve()
        self._snapshot = {name: path.read_bytes() for name, path in self.paths.items()}
        self.cards = json.loads(self._snapshot["card_gallery"])
        self.consciousness = json.loads(self._snapshot["consciousness_contract"])

    def body(self, name: str) -> bytes:
        return self._snapshot[name]

    def _entry(self, scene: str) -> dict:
        family, _, state = scene.partition(".")
        entry = self.cards.get(family, {}).get(state) if family in RECORDED else None
        return entry if isinstance(entry, dict) else {}

    @staticmethod
    def _at(responses: list, step: int):
        return responses[max(0, min(step, len(responses) - 1))] if responses else None

    def answer(self, method: str, path: str, scene: str, step: int) -> tuple[int, object]:
        """(status, body) for one API request of the open scene; (404, …) for a read the corpus has no answer for and
        (405, …) for any write — nothing but a read is answered, the sign-in card's start answered inside the page."""
        family = scene.partition(".")[0]
        entry = self._entry(scene)
        if method == "GET":
            card = re.fullmatch(r"/api/cards/([^/]+)", path)
            if card:
                for state in self.cards.get("living_card", {}).values():
                    frames = state.get("frames") or []
                    if frames and str(frames[0].get("card", {}).get("card_id")) == card.group(1):
                        return int(state.get("status") or 200), self._at(state.get("responses") or [], step)
            reviews = re.fullmatch(r"/api/tasks/([^/]+)/reviews", path)
            if reviews:
                for state in self.cards.get("reviews", {}).values():
                    if state.get("task_id") == reviews.group(1):
                        return 200, self._at(state.get("responses") or [], step)
            if path == "/api/extensions" and family == "skills":
                return 200, self._at(entry.get("responses") or [], step)
            if path == "/api/widgets" and family == "widgets":
                return 200, self._at(entry.get("responses") or [], step)
            if family == "login" and re.fullmatch(r"/api/claudexor/login/[^/]+", path):
                return 200, self._at(entry.get("responses") or [], step)
            if family in ("task_card", "share"):  # a task's room: its history, the census and the task's detail
                words = re.fullmatch(r"/api/tasks/([^/]+)/artifacts/([^/]+)\.(letter|answer)\.txt", path)
                if words:
                    whole = (entry.get("letter_words") or {}).get(words.group(2))
                    if whole is None:
                        return 409, {"reason_code": "reply_not_given"}
                    return 200, RawText(whole.encode("utf-8"))
                read = {"/api/chat/history": "history", "/api/state": "state",
                        f"/api/tasks/{entry.get('task_id')}": "detail"}.get(path)
                if read:
                    status = int(entry.get("detail_status") or 200) if read == "detail" else 200
                    return status, self._at(entry.get(read) or [], step)
            if family == "agents":
                if path == "/api/settings":
                    return 200, entry.get("settings") or {}
                if path == "/api/review-pool":
                    return 200, self._at(entry.get("responses") or [], step)
            if family == "connection" and path == "/api/onboarding/starter-skills":
                return 200, self._at(entry.get("responses") or [], step)
            if family == "consciousness":
                if path == "/api/consciousness":
                    if scene.endswith(".read_failed"):
                        return 503, {"error": "consciousness facts are unavailable"}
                    return 200, self.consciousness["consciousness"]
                facts = {"/api/state": "state", "/api/places": "places", "/api/settings": "settings"}.get(path)
                if facts:
                    return 200, self.consciousness[facts]
            return 404, {"error": "not in the gallery corpus", "path": path}
        return 405, {"error": "the gallery answers no writes", "path": path}


class GalleryServer:
    """The stand on a loopback port of its own; ``url`` is its page, and ``requests`` what the page asked: one entry
    per API request with its ``method``, ``path``, the open ``scene`` and ``step`` its headers named, and the
    ``status`` answered."""

    def __init__(self, port: int = 0, card_corpus: Path | None = None):
        self.corpus = Corpus(card_corpus)
        self.requests: list = []
        self.shell, self.setup_shell = shell_html(), shell_html("onboarding_template.html")
        stand = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _send(self, status: int, body: bytes, kind: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _route(self) -> None:
                path = unquote(urlsplit(self.path).path)
                # The page names its scene per request; a cookie cannot, so two tabs never share one.
                scene = self.headers.get("X-Gallery-Scene") or ""
                try:
                    step = int(self.headers.get("X-Gallery-Step") or 0)
                except ValueError:
                    step = 0
                if path.startswith("/api/"):
                    status, body = stand.corpus.answer(self.command, path, scene, step)
                    stand.requests.append({"method": self.command, "path": path, "scene": scene, "step": step,
                                           "status": status})
                    if isinstance(body, RawText):
                        return self._send(status, body.data, "text/plain; charset=utf-8")
                    return self._send(status, json.dumps(body).encode(), "application/json")
                if self.command not in ("GET", "HEAD"):
                    return self._send(405, b"", "text/plain")
                if path in ("/", "/index.html"):
                    asked = parse_qs(urlsplit(self.path).query).get("scene", [""])[0].partition(".")[0]
                    shell = stand.setup_shell if asked in SETUP else stand.shell
                    return self._send(200, shell.encode(), "text/html; charset=utf-8")
                if path.startswith("/gallery/corpus/") and path.endswith(".json"):
                    name = path[len("/gallery/corpus/"):-len(".json")]
                    if name in stand.corpus.paths:
                        return self._send(200, stand.corpus.body(name), "application/json")
                if path.startswith("/gallery/") and path[len("/gallery/"):] in STAND_FILES:
                    name = path[len("/gallery/"):]
                    return self._send(200, (ASSETS / name).read_bytes(), STAND_FILES[name])
                prefix = next((each for each in STATIC_PREFIXES if path.startswith(each)), "")
                if prefix and not path.startswith(f"{prefix}tests/"):
                    target = (WEB / path[len(prefix):]).resolve()
                    if target.is_file() and WEB.resolve() in target.parents:
                        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                        if kind.startswith("text/") or kind.endswith("javascript"):
                            kind += "; charset=utf-8"
                        return self._send(200, target.read_bytes(), kind)
                return self._send(404, b"not found", "text/plain")

            do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = _route

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._serving = False

    def __enter__(self) -> GalleryServer:
        try:
            self._thread.start()
        except BaseException:
            self._server.server_close()  # a setup fault leaves no bound port behind
            raise
        self._serving = True
        return self

    def __exit__(self, *_exc) -> None:
        if self._serving:
            self._server.shutdown()  # only a serve_forever that runs can be shut down; never one that never started
            self._serving = False
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def scene_url(self, scene: str, theme: str = "dark") -> str:
        return f"{self.url}/?scene={scene}&theme={theme}"


def bundled_browser(pw, engine: str):
    """Launch Playwright's own headless build of ``engine``; refuse an executable outside its browser cache."""
    browser_type = getattr(pw, engine)
    executable = Path(browser_type.executable_path).resolve()
    cache = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if cache and cache != "0" and Path(cache).resolve() not in executable.parents:
        raise RuntimeError(f"{engine} would start {executable}, outside PLAYWRIGHT_BROWSERS_PATH ({cache})")
    if "/Applications/" in str(executable):
        raise RuntimeError(f"{engine} would start an installed application: {executable}")
    return browser_type.launch(headless=True)


def open_scene(context, url: str, timeout_ms: int = 15000):
    """A page on one stand URL, ready when the stand says so; page errors are kept in ``page.gallery_errors``.

    The stand has two terminals: ``ready`` once the scene mounted, ``failed`` (with a visible ``.gallery-fault`` row)
    when it could not draw one. A fault fails here, before any frame is taken, instead of waiting forever or passing a
    broken page off as mounted."""
    page = context.new_page()
    errors: list = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.gallery_errors = errors
    page.goto(url)
    page.wait_for_function("() => { const g = window.gallery; return g && (g.ready === true || g.failed); }",
                          timeout=timeout_ms)
    failed = page.evaluate("() => (window.gallery && window.gallery.failed) || ''")
    if failed:
        raise RuntimeError(f"the stand could not draw {url}: {failed}")
    return page


def scene_names(page) -> list:
    return page.evaluate("() => window.gallery.scenes()")


def _check_scenes(asked: list | None, available: list) -> list:
    """The scenes to capture: ``asked`` validated against the stand's own list, or all of the stand's scenes."""
    if not asked:
        return list(available)
    unknown = [name for name in asked if name not in available]
    if unknown:
        raise ValueError(f"unknown scene id(s) {', '.join(unknown)}; this stand offers: {', '.join(available)}")
    return list(asked)


def look(out: Path, *, card_corpus: Path | None = None, scenes: list | None = None, engines=("chromium", "webkit"),
         themes=("dark", "light"), viewports=DEFAULT_VIEWPORTS) -> dict:
    """Frames and measure.js facts for every asked scene on the stand; returns the report it writes."""
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    measure = MEASURE_JS.read_text(encoding="utf-8")
    report = {"stand": "tests/card_gallery_stand.py", "kind": "browser frames, not the macOS app window",
              "corpora": {name: str(path) for name, path in Corpus(card_corpus).paths.items()}, "frames": []}
    with GalleryServer(card_corpus=card_corpus) as server, sync_playwright() as pw:
        for engine in engines:
            browser = bundled_browser(pw, engine)
            report.setdefault("executables", {})[engine] = getattr(pw, engine).executable_path
            try:
                for width, height in viewports:
                    for theme in themes:
                        context = browser.new_context(viewport={"width": width, "height": height},
                                                      device_scale_factor=2, color_scheme=theme,
                                                      has_touch=width < 700, is_mobile=False, **FRAME_FACTS)
                        try:
                            index = open_scene(context, server.scene_url("quiz.open", theme))
                            names = _check_scenes(scenes, scene_names(index))
                            index.close()
                            for scene in names:
                                page = open_scene(context, server.scene_url(scene, theme))
                                shown = page.evaluate("() => window.gallery.scene || ''")
                                if shown != scene:
                                    page.close()
                                    raise RuntimeError(f"the stand drew {shown or 'nothing'} for the asked scene {scene}")
                                stem = f"{scene}-{engine}-{theme}-{width}x{height}"
                                page.screenshot(path=str(out / f"{stem}.png"), animations="disabled")
                                # Measured at the initial screen, before the card crop scrolls the page.
                                facts = page.evaluate(f"({measure})({json.dumps({'subject': '[data-gallery-subject]'})})")
                                measured = [facts["viewport"]["w"], facts["viewport"]["h"]]
                                if measured != [width, height]:
                                    page.close()
                                    raise RuntimeError(f"scene {scene} measured at {measured[0]}x{measured[1]}, "
                                                       f"not the asked {width}x{height}")
                                frame = {"scene": scene, "shown": shown, "engine": engine, "theme": theme,
                                         "viewport": [width, height], "measured_viewport": measured,
                                         "frame": f"{stem}.png", "page_errors": page.gallery_errors,
                                         "findings": facts.get("findings", [])}
                                subject = page.locator("[data-gallery-subject]")
                                if subject.count() and subject.first.is_visible():
                                    subject.first.screenshot(path=str(out / f"{stem}-card.png"), animations="disabled")
                                    frame["card"] = f"{stem}-card.png"
                                report["frames"].append(frame)
                                page.close()
                        finally:
                            context.close()
            finally:
                browser.close()
        report["requests"] = server.requests
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def _pairs(text: str) -> tuple:
    return tuple(tuple(int(n) for n in item.split("x", 1)) for item in text.split(",") if item)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.card_gallery_stand", description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve the stand on 127.0.0.1 until interrupted")
    serve.add_argument("--port", type=int, default=0)
    shot = sub.add_parser("look", help="frames and measure.js facts of the stand's scenes")
    shot.add_argument("--out", type=Path, required=True)
    shot.add_argument("--scenes", default="")
    shot.add_argument("--engines", default="chromium,webkit")
    shot.add_argument("--themes", default="dark,light")
    shot.add_argument("--viewports", default="1440x900,390x844")
    for command in (serve, shot):
        command.add_argument("--card-corpus", type=Path, default=None,
                             help="read this one card_gallery.json instead of tests/fixtures/card_gallery.json")
    args = parser.parse_args(argv)
    if args.command == "serve":
        with GalleryServer(args.port, card_corpus=args.card_corpus) as server:
            print(f"Card gallery: {server.url}/  (Ctrl+C stops it)", flush=True)
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                pass
        return 0
    report = look(args.out, card_corpus=args.card_corpus, scenes=[s for s in args.scenes.split(",") if s] or None,
                  engines=tuple(e for e in args.engines.split(",") if e),
                  themes=tuple(t for t in args.themes.split(",") if t), viewports=_pairs(args.viewports))
    errors = [frame for frame in report["frames"] if frame["page_errors"]]
    print(f"{len(report['frames'])} frames in {args.out}; {len(errors)} with page errors; report.json beside them")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
