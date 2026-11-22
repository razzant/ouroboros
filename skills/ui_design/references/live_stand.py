"""Look at Ouroboros's own live interface without changing anything — an optional instrument of the ui_design skill.

When it helps: you changed a shared part of the interface (a control, a card, a status line, a list) and want every place
that uses it before and after, on the owner's real data, in both themes and at desk and phone widths; or the owner asks
you to look over the whole interface. It returns pictures and numbers, never a verdict. Nothing runs it on its own: it
runs when someone asks (a command below, or Ouroboros's `ui_look` tool). Its own tests run in the UI browser lane; no
test, CI job, release step or schedule checks the product's screens with it.

Safety, enforced here (do not open the live interface any other way while using it):
  * only plain reads (GET/HEAD) reach the live server; reads that ask it for fresh work (?fresh= ?refresh= ?force=
    ?probe= ?checkLatest= ...) are answered locally;
  * every write is answered locally: --mutations ok fakes success (to see a state after Save), abort shows the failure state;
  * every frame the page sends on its WebSocket is dropped (server -> page frames still flow, so the page stays live).
  So pressing Save, Stop, Restart or Panic inside it changes nothing on the install.

Before/after of code you have not merged: --web-root <worktree>/web serves the interface (index.html and /static/*; a
file your branch deleted answers 404) from your branch while all data still comes from the live server (the running
server's API must match your branch's).

CLI (run with a Python that imports Playwright; Playwright's own browsers unless --executable-path names one):
  python live_stand.py capture --out DIR [--places all|chat,dashboard,...] [--themes dark,light] [--widths 1440,390]
                               [--web-root PATH/web] [--mutations ok|abort] [--rooms N] [--url URL]
      -> DIR/<theme>-<width>/<place>[-<tab>].png, __pNN.png scroll frames, .measure.json (uneven rows of controls,
         long texts, measure.js findings) ; places and tabs are discovered from the page ([data-nav-page], [role="tab"]).
  python live_stand.py diff --base PATH/web --cand PATH/web --out DIR [--themes dark] [--widths 1440] [--places ...]
      -> DIR/<theme>-<width>/<place>.json computed-style differences, DIR/SUMMARY.md. One page and one moment of data:
         on every screen the stylesheets are swapped from --base to --cand and back while live updates are held, so a
         difference is caused by the stylesheets alone. It measures CSS changes; a JavaScript change shows in captures.
  python live_stand.py look --spec SPEC.json --report OUT.json
      -> the places a change touches (place_plan.py beside this file reads its diff against the served web/) or the
         places SPEC names, at each of its viewports, themes and engines: per screen the measurements (measure.js
         beside this file: covered text, small targets, overflow, cut text; plus uneven rows and long texts) in
         OUT.json, rewritten after every screen, and on request frames as flat files captioned with the engine that
         drew them; a place or view that does not show when pressed is named, not measured under its name. With a draft
         and frames a second page shows the served interface on the same data: every screen it also has is a pair,
         served then draft, the changed classes are looked for on both, and a finding the served page lacks (matched
         one by one) is marked new. Ouroboros's `ui_look` tool writes SPEC; look() reads its keys.
  --engine webkit|chromium (default webkit, the owner's macOS app engine; falls back to chromium when WebKit is missing).
  --executable-path PATH launches exactly that Chromium (Ouroboros's pinned headless shell, as browse_page) and nothing
  else: no fallback to another engine or build.
Library: from live_stand import Stand; with Stand(theme="dark", width=1440) as s: s.open(); s.nav("settings"); s.shot("x").
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

SIDE_EFFECT_GET = re.compile(r"[?&](force|fresh|refresh|checkLatest|check_latest|probe|recheck|sync)=", re.I)
OK_BODIES = {"/api/settings": {"ok": True, "immediate_changed": True, "next_task_changed": False,
                               "restart_required": False, "no_changes": False, "warnings": []}}


def live_url() -> str:
    """The running server: OUROBOROS_PORT_FILE or <data>/state/server_port, else OUROBOROS_SERVER_PORT, else 8765."""
    candidates = [os.environ.get("OUROBOROS_PORT_FILE", "").strip()]
    data = os.environ.get("OUROBOROS_DATA_DIR", "").strip() or str(Path.home() / "Ouroboros" / "data")
    candidates.append(str(Path(data) / "state" / "server_port"))
    for c in candidates:
        try:
            port = int(Path(c).read_text(encoding="utf-8").strip()) if c else 0
        except (OSError, ValueError):
            port = 0
        if port:
            return f"http://127.0.0.1:{port}"
    return f"http://127.0.0.1:{int(os.environ.get('OUROBOROS_SERVER_PORT', '8765'))}"


def _sibling_text(name: str) -> str:
    try:
        return Path(__file__).resolve().with_name(name).read_text(encoding="utf-8")
    except OSError:
        return ""


MEASURE_JS = _sibling_text("measure.js")  # the ui_design snippet beside this file; measure() calls it on every screen

def _sibling_module(name: str):
    """A Python file beside this one, loaded by its path, never by a search of sys.path."""
    spec = importlib.util.spec_from_file_location(f"ui_design_{name}", Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


place_plan = _sibling_module("place_plan").place_plan   # the places a change touches: place_plan.py beside this file


_JS = r"""
(() => {
  if (window.__look) return;
  const vis = (n) => { if (!(n instanceof Element)) return false; const r = n.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false; const cs = getComputedStyle(n);
    return cs.visibility !== 'hidden' && cs.display !== 'none' && Number(cs.opacity) !== 0 && !n.closest('[hidden]'); };
  const label = (n) => { const cls = (typeof n.className === 'string' ? n.className : '').split(/\s+/).filter(Boolean).slice(0, 3).join('.');
    const t = (n.innerText || n.getAttribute('aria-label') || '').trim().replace(/\s+/g, ' ').slice(0, 32);
    return `${n.tagName.toLowerCase()}${n.id ? '#' + n.id : ''}${cls ? '.' + cls : ''}${t ? ' "' + t + '"' : ''}`; };
  const ownText = (n) => [...n.childNodes].filter((c) => c.nodeType === 3).map((c) => c.textContent).join(' ').trim().replace(/\s+/g, ' ');
  // Controls on one visual line, whatever their DOM parents; the gaps a reader sees between them.
  const visualRows = (maxGap) => {
    const SEL = 'button, [role="button"], a[href], input, select, textarea, summary, [class*="chip"], [class*="pill"], [class*="capsule"]';
    let atoms = [...document.querySelectorAll(SEL)].filter(vis); atoms = atoms.filter((n) => !atoms.some((o) => o !== n && o.contains(n)));
    const region = (n) => n.closest('[role="dialog"], dialog, nav, aside, main') || document.body;
    const bands = [];
    for (const n of atoms) { const r = n.getBoundingClientRect(); if (r.height > 80) continue; const cy = r.top + r.height / 2, g = region(n);
      let b = bands.find((x) => x.g === g && Math.abs(x.cy - cy) <= 4 && r.height <= x.h * 1.6 && x.h <= r.height * 1.6);
      if (!b) { b = { g, cy, h: r.height, items: [] }; bands.push(b); } b.items.push({ n, r }); }
    const out = [];
    for (const b of bands) { const it = b.items.sort((p, q) => p.r.left - q.r.left); let grp = [it[0]];
      const flush = () => { if (grp.length < 3) return; const gaps = []; for (let i = 1; i < grp.length; i++) gaps.push(Math.round((grp[i].r.left - grp[i - 1].r.right) * 10) / 10);
        out.push({ y: Math.round(b.cy), items: grp.map((o) => label(o.n)), heights: grp.map((o) => Math.round(o.r.height)), gaps,
                   spread: Math.round((Math.max(...gaps) - Math.min(...gaps)) * 10) / 10 }); };
      for (let i = 1; i < it.length; i++) { const g = it[i].r.left - it[i - 1].r.right; if (g < -1 || g > (maxGap || 64)) { flush(); grp = [it[i]]; } else grp.push(it[i]); }
      flush(); }
    return out.sort((p, q) => q.spread - p.spread);
  };
  const texts = (min) => [...document.querySelectorAll('body *')].filter(vis).map((n) => ({ n, t: ownText(n) }))
    .filter((o) => o.t.length >= (min || 120)).map((o) => ({ el: label(o.n), chars: o.t.length, text: o.t.slice(0, 300) }));
  const scroller = () => { let best = null, area = 0; for (const n of document.querySelectorAll('*')) { if (!vis(n)) continue;
      if (!/(auto|scroll)/.test(getComputedStyle(n).overflowY) || n.scrollHeight <= n.clientHeight + 8) continue;
      const a = n.clientWidth * n.clientHeight; if (a > area) { area = a; best = n; } } return best; };
  const snap = (props) => { const out = {}; for (const n of document.querySelectorAll('body *')) { if (!vis(n)) continue;
      if (!n.dataset.lookId) { window.__lookSeq = (window.__lookSeq || 0) + 1; n.dataset.lookId = String(window.__lookSeq); }
      const cs = getComputedStyle(n); const o = {}; for (const p of props) o[p] = cs.getPropertyValue(p);
      out[n.dataset.lookId] = { el: label(n), s: o }; } return out; };
  const swap = async (toCand) => { const links = [...document.querySelectorAll('link[rel="stylesheet"]')].filter((l) => /\/static\//.test(l.href));
    await Promise.all(links.map((l) => new Promise((res) => { const path = new URL(l.href).pathname;
      const want = toCand ? path.replace(/^\/static\//, '/__cand/static/') : path.replace(/^\/__cand\/static\//, '/static/');
      if (path === want) return res(); const nl = l.cloneNode(); nl.href = want; nl.onload = () => { l.remove(); res(); };
      nl.onerror = () => res(); l.after(nl); })));
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))); };
  const places = () => [...new Set([...document.querySelectorAll('[data-nav-page]')].map((n) => n.getAttribute('data-nav-page')))];
  const placeRow = (id) => [...document.querySelectorAll('[data-nav-page]')].find((n) => n.getAttribute('data-nav-page') === id);
  const go = (id) => { const b = placeRow(id); if (b) b.click(); return Boolean(b); };
  // "" when the place a press asked for is the one shown: its page-<id> section, else its row chosen.
  const placeShown = (id) => { const page = document.getElementById(`page-${id}`);
    if (page) return vis(page) ? '' : 'did not open';
    return (placeRow(id) || document.body).getAttribute('aria-current') === 'page' ? '' : 'did not open'; };
  const shownPage = () => [...document.querySelectorAll('.page.active, .page:not([hidden])')].find(vis) || document;
  // A page's views: its tabs and the rows of its section list (a Dashboard view, a Settings or Extensions page). Each
  // is marked once with data-look-tab, named by its title and keyed by what it opens (its panel or page), so a list a
  // phone hides after the first press, a row whose answer line changes or a renamed tab still reaches the same view.
  const tabName = (t) => { const title = t.querySelector('.section-row-title');
    return ((title && title.textContent) || t.getAttribute('aria-label') || ownText(t) || t.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 32); };
  const opens = (t) => t.getAttribute('aria-controls') || t.dataset.settingsOpen || t.dataset.extensionsOpen || '';
  const tabs = () => { const page = shownPage();
    document.querySelectorAll('[data-look-tab]').forEach((n) => n.removeAttribute('data-look-tab'));   // a page left before
    return [...page.querySelectorAll('[role="tab"], .section-rail .section-row:not([data-section-out])')].filter((t) => vis(t) && !t.disabled)
      .map((t, i) => { t.setAttribute('data-look-tab', String(i)); return { i, name: tabName(t) || String(i), target: opens(t) }; }); };
  // The list a place stands on before any of its pages is chosen (a phone's Settings, Extensions or Dashboard list).
  const landing = () => [...shownPage().querySelectorAll('.section-shell')].some((s) => {
    const rail = s.querySelector(':scope > .section-rail'), pane = s.querySelector(':scope > .section-pane');
    return Boolean(rail && vis(rail) && !(pane && vis(pane))); });
  const press = (i) => { const t = document.querySelector(`[data-look-tab="${i}"]`); if (t) t.click(); return Boolean(t); };
  // "" when the view a press asked for is on screen: the panel it declares shown (a panel that does not exist is not),
  // else its row chosen beside (or over) its page.
  const shown = (i) => { const t = document.querySelector(`[data-look-tab="${i}"]`); if (!t) return 'its control is gone';
    const id = t.getAttribute('aria-controls');
    if (id) { const panel = document.getElementById(id); return !panel ? `its panel ${id} does not exist` : vis(panel) ? '' : 'its panel is not shown'; }
    if (t.getAttribute('role') === 'tab') return t.getAttribute('aria-selected') === 'true' ? '' : 'it is not selected';
    const pane = t.closest('.section-shell') && t.closest('.section-shell').querySelector(':scope > .section-pane');
    return t.getAttribute('aria-current') === 'page' && pane && vis(pane) ? '' : 'its page is not shown'; };
  // Project rooms a press can open (a room being deleted cannot), and whether a room is the open one.
  const rooms = () => [...document.querySelectorAll('.nav-project-row[data-project-id]')].filter((b) => !b.disabled)
    .map((b) => ({ id: b.dataset.projectId, name: ((b.querySelector('.nav-row-label') || b).textContent || '').trim().slice(0, 40) }));
  const openRoom = (id) => { const b = [...document.querySelectorAll('.nav-project-row[data-project-id]')].find((n) => n.dataset.projectId === id);
    if (b && !b.disabled) b.click(); return Boolean(b && !b.disabled); };
  const roomShown = (id) => ([...document.querySelectorAll('[data-project-id][aria-current="page"]')]
    .some((n) => n.dataset.projectId === id) ? '' : 'did not open');
  // The classes that stand on this screen: shown, in the page shown or outside every page (the shell).
  const probe = (classes) => classes.filter((c) => [...document.getElementsByClassName(c)].some((n) => vis(n)
    && (!n.closest('.page') || n.closest('.page').classList.contains('active'))));
  const panels = () => { const inline = document.documentElement.style;
    const width = (id) => { const n = document.getElementById(id); return n && vis(n) ? Math.round(n.getBoundingClientRect().width) : null; };
    return { saved_sidebar: inline.getPropertyValue('--sidebar-width').trim() || null,
             saved_project_panel: inline.getPropertyValue('--project-panel-width').trim() || null,
             sidebar: width('primary-sidebar'), project_panel: width('project-panel') }; };
  window.__look = { vis, visualRows, texts, scroller, places, go, placeShown, tabs, landing, press, shown, rooms, openRoom,
                    roomShown, snap, swap, probe, panels };
})();
"""


class ShellError(RuntimeError):
    """The interface did not load: no place to go to appeared on the page."""


class Stand:
    """One headless browser on the live interface with the safety above. Use as a context manager.

    With ``twin`` and a ``web_root``, a second page in a context of its own shows the served interface on the same live
    data: each press lands on both pages and is waited for once, so ``twin`` is the "before" of ``page``. A twin that
    fails is closed and named in ``twin_error``; the page goes on alone."""

    hold = False   # True keeps server -> page socket frames (live updates) back until it is set to False again

    def __init__(self, theme="dark", width=1440, height=900, scale=2, engine="webkit", mutations="ok",
                 out=None, web_root=None, url=None, executable_path=None, sandbox=True, fallback=True, touch=None,
                 keep_intercepted=True, launch_args=(), twin=False):
        self.theme, self.width, self.height, self.scale, self.engine = theme, width, height, scale, engine
        self.mutations = mutations
        self.out = Path(out) if out else Path.cwd() / "live-stand"
        self.out.mkdir(parents=True, exist_ok=True)
        self.web_root = Path(web_root) if web_root else None
        self.url = url or live_url()
        self.executable_path = str(executable_path or "")   # an exact Chromium build: launched alone, never replaced
        if self.executable_path:
            self.engine = "chromium"
        self.launch_args = [str(arg) for arg in launch_args or ()]   # browse_page's own, for that build
        self.sandbox, self.fallback, self.keep_intercepted = bool(sandbox), bool(fallback), bool(keep_intercepted)
        self.touch = self.width < 700 if touch is None else bool(touch)
        self.want_twin, self.twin, self.twin_error = bool(twin and self.web_root), None, ""
        self.version = ""
        self.served_from_root = 0   # requests answered from --web-root: the draft interface the page really ran
        self.missing_from_root: list[str] = []   # files the page asked for that the draft does not have (404)
        self.caption_errors: dict = {}   # frame path -> why its caption could not be drawn
        self.intercepted: list[dict] = []
        self.dropped_ws: list[str] = []

    @property
    def pages(self):
        return [self.page] + ([self.twin] if self.twin is not None else [])

    def __enter__(self):
        from playwright.sync_api import sync_playwright   # here: the place plan and the CLI help need no browser

        self._pw, self._browser = sync_playwright().start(), None
        try:
            self._browser = self._launch()
            self.version = str(getattr(self._browser, "version", "") or "")
            self.ctx = self._context(draft=True)
            self.page = self.ctx.new_page()
            self.page.set_default_timeout(30000)
            if self.want_twin:
                self.twin_ctx = self._context(draft=False)
                self.twin = self.twin_ctx.new_page()
                self.twin.set_default_timeout(30000)
        except BaseException:
            self._close()
            raise
        return self

    def _context(self, draft):
        ctx = self._browser.new_context(viewport={"width": self.width, "height": self.height},
                                        device_scale_factor=self.scale, color_scheme=self.theme,
                                        service_workers="block", has_touch=self.touch)
        ctx.route("**/*", lambda route: self._http(route, draft))
        ctx.route_web_socket(re.compile(r".*"), self._ws)
        return ctx

    def _launch(self):
        if self.executable_path:   # Ouroboros's pinned headless shell, sandboxed and with browse_page's arguments
            return self._pw.chromium.launch(headless=True, executable_path=self.executable_path,
                                            chromium_sandbox=self.sandbox, args=self.launch_args)
        try:
            return getattr(self._pw, self.engine).launch()
        except Exception as exc:   # the engine is not installed in this browsers root
            if self.engine == "chromium" or not self.fallback:
                raise
            print(f"[live_stand] {self.engine} unavailable ({str(exc).splitlines()[0][:120]}); using chromium", flush=True)
            self.engine = "chromium"
            return self._pw.chromium.launch()

    def _close(self):
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            self._pw.stop()

    def __exit__(self, *exc):
        try:
            if self.keep_intercepted:
                (self.out / "_intercepted.json").write_text(json.dumps(
                    {"writes_answered_locally": self.intercepted, "dropped_ws_frames": self.dropped_ws[:200]}, indent=1),
                    encoding="utf-8")
        finally:
            self._close()

    def _http(self, route, draft=True):
        req = route.request
        path = "/" + req.url.split("://", 1)[-1].split("/", 1)[-1].split("?", 1)[0].split("#", 1)[0]
        own = urlsplit(req.url)[:2] == urlsplit(self.url)[:2]   # the interface's origin; another one's files are its own
        if draft and own and self.web_root and req.method in ("GET", "HEAD") and (path.startswith("/static/") or path in ("/", "/index.html")):
            local = self.web_root / (unquote(path[len("/static/"):]) if path.startswith("/static/") else "index.html")
            if _inside(local, self.web_root) and local.is_file():
                route.fulfill(status=200, path=str(local))
                self.served_from_root += 1
            else:   # the draft is the whole interface: a file it lacks is missing, not the installed copy's
                route.fulfill(status=404, body="")
                if path not in self.missing_from_root and len(self.missing_from_root) < 50:
                    self.missing_from_root.append(path)
            return
        if req.method in ("GET", "HEAD") and not SIDE_EFFECT_GET.search(req.url):
            route.continue_()
            return
        self.intercepted.append({"method": req.method, "url": req.url[:300]})
        if self.mutations == "ok" and req.method not in ("GET", "HEAD"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(OK_BODIES.get(path, {"ok": True})))
        else:
            route.abort()

    def _ws(self, ws):
        server = ws.connect_to_server()
        server.on_message(lambda m: None if self.hold else ws.send(m))   # forwarded by hand so hold can pause them
        ws.on_message(lambda m: self.dropped_ws.append(str(m)[:200]))    # the page's own frames never leave

    # the twin ----------------------------------------------------------------------------------------------------
    def _drop_twin(self, why):
        self.twin_error = self.twin_error or why
        twin, self.twin = self.twin, None
        try:
            if twin is not None:
                self.twin_ctx.close()
        except Exception:
            pass

    def on_twin(self, act):
        """act(twin) when there is a twin, else None; a twin that fails is closed and named instead of raising."""
        if self.twin is None:
            return None
        try:
            return act(self.twin)
        except Exception as exc:
            self._drop_twin(f"{type(exc).__name__}: {_first_line(exc)}")
            return None

    def _each(self, act):
        act(self.page)
        self.on_twin(act)

    # navigation ------------------------------------------------------------------------------------------------
    def open(self, wait_ms=8000):
        try:
            self.page.goto(self.url + "/", wait_until="domcontentloaded")
            self.on_twin(lambda twin: twin.goto(self.url + "/", wait_until="domcontentloaded"))
            self.page.wait_for_selector("[data-nav-page]", state="attached", timeout=60000)
        except Exception as exc:   # no place to go to: the shell did not load, and nothing here is the interface
            raise ShellError(f"the interface did not load ({_first_line(exc)})") from exc
        self.on_twin(lambda twin: twin.wait_for_selector("[data-nav-page]", state="attached", timeout=30000))
        self.page.wait_for_timeout(wait_ms)
        self._each(lambda page: page.evaluate(_JS))
        return self

    def places(self):
        self.page.evaluate(_JS)
        return self.page.evaluate("() => window.__look.places()")

    def nav(self, place, wait_ms=5000):
        """Go to a place on the page and the twin: (why the page does not show it or "", the twin's or None)."""
        return self._press("(id) => window.__look.go(id)", "(id) => window.__look.placeShown(id)", place,
                           place if self.twin is not None else None, wait_ms, "place")

    def tabs(self, page=None):
        """The views of the page shown: [{i, name, target}], each marked on that page so a press finds it again."""
        return (page or self.page).evaluate("() => window.__look.tabs()")

    def landing(self):
        """Whether the place shown stands on its list before any of its pages is chosen (a phone's section list)."""
        return bool(self.page.evaluate("() => window.__look.landing()"))

    def _press(self, press, shown, arg, twin_arg, wait_ms, what="view"):
        """Press on the page (and on the twin with ``twin_arg``), wait once, then answer (why the page does not show
        what was pressed, or "") and the same for the twin (None without a twin)."""
        pressed = self.page.evaluate(press, arg)
        twin_pressed = self.on_twin(lambda twin: twin.evaluate(press, twin_arg)) if twin_arg is not None else None
        self.page.wait_for_timeout(wait_ms)
        self._each(lambda page: page.evaluate(_JS))
        why = self.page.evaluate(shown, arg) if pressed else "cannot be pressed"
        if self.twin is None:
            return why, None
        if twin_arg is None or not twin_pressed:
            return why, f"the served interface has no such {what}"
        served = self.on_twin(lambda twin: twin.evaluate(shown, twin_arg))
        return why, (None if self.twin is None else f"on the served page {served}" if served else "")

    def tab(self, index, twin_index=None, wait_ms=4000):
        """Press view ``index`` (and the twin's ``twin_index``): (why it is not on screen or "", the twin's)."""
        return self._press("(i) => window.__look.press(i)", "(i) => window.__look.shown(i)", index, twin_index, wait_ms)

    def rooms(self):
        """The Project rooms a press can open: [{id, name}] in the order of the sidebar."""
        self.page.evaluate(_JS)
        return self.page.evaluate("() => window.__look.rooms()")

    def open_room(self, room_id, wait_ms=6000):
        """Open a Project room by its id: (why it is not the open room or "", the twin's)."""
        return self._press("(id) => window.__look.openRoom(id)", "(id) => window.__look.roomShown(id)", room_id,
                           room_id if self.twin is not None else None, wait_ms, "room")

    def probe(self, classes, page=None):
        """Which of ``classes`` stand on this screen (shown, on the page shown or in the shell around it)."""
        page = page or self.page
        page.evaluate(_JS)
        return page.evaluate("(c) => window.__look.probe(c)", sorted(classes))

    def panels(self):
        """The panel widths the page uses: the saved ones it read from the server, and the drawn ones."""
        self.page.evaluate(_JS)
        return self.page.evaluate("() => window.__look.panels()")

    # capture and measure ---------------------------------------------------------------------------------------------
    def shot(self, name, caption="", page=None):
        p = self.out / f"{name}.png"
        # a busy live server can serve a web font slowly; the screenshot waits for fonts, so allow it time
        png = (page or self.page).screenshot(timeout=120000)
        if caption:
            png, error = caption_frame(png, caption, self.scale)
            if error:
                self.caption_errors[str(p)] = error
        p.write_bytes(png)
        return p

    def shot_scroll(self, name, max_frames=8, caption="", page=None):
        page = page or self.page
        n = page.evaluate("() => { const s = window.__look.scroller(); if (!s) return 0; s.setAttribute('data-look-scroller', '1');"
                          " s.scrollTop = 0; return Math.ceil(s.scrollHeight / Math.max(200, s.clientHeight - 80)); }")
        frames = []
        for i in range(1, min(n, max_frames)):
            page.evaluate("(i) => { const s = document.querySelector('[data-look-scroller]'); s.scrollTop = i * Math.max(200, s.clientHeight - 80); }", i)
            page.wait_for_timeout(350)
            frames.append(self.shot(f"{name}__p{i:02d}", f"{caption} | scrolled {i}" if caption else "", page=page))
        page.evaluate("() => { const s = document.querySelector('[data-look-scroller]'); if (s) { s.scrollTop = 0; s.removeAttribute('data-look-scroller'); } }")
        return frames

    def measure(self, target=None, limit=6, page=None):
        """Uneven rows and long texts, then measure.js: a coarse pointer under touch, whose targets are 44 px
        (DESIGN "Touch and containment"). A failure of the snippet is named and leaves the rest of the screen."""
        page = page or self.page
        page.evaluate(_JS)
        found = {"uneven_rows": [r for r in page.evaluate("() => window.__look.visualRows(64)") if r["spread"] >= 3],
                 "long_texts": page.evaluate("() => window.__look.texts(160)")}
        if MEASURE_JS:
            options = {"limit": limit, "coarse": self.touch, "keys": True, **({"target": target} if target else {})}
            try:
                found["layout"] = page.evaluate(f"({MEASURE_JS})({json.dumps(options)})")
            except Exception as exc:
                found["layout_error"] = f"{type(exc).__name__}: {_first_line(exc)}"
        return found


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def caption_frame(png: bytes, text: str, scale=2):
    """(the frame with a band under it naming what drew it, as text and as a PNG note, "") or, when Pillow is missing
    or cannot read the frame, (the frame as captured, why it has no caption)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        from PIL.PngImagePlugin import PngInfo

        frame = Image.open(io.BytesIO(png)).convert("RGB")
        size, band = max(10, int(round(12 * float(scale or 1)))), max(18, int(round(20 * float(scale or 1))))
        out = Image.new("RGB", (frame.width, frame.height + band), (28, 30, 38))
        out.paste(frame, (0, 0))
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:   # Pillow before 10.1 has one bitmap face
            font = ImageFont.load_default()
        ImageDraw.Draw(out).text((band // 2, frame.height + (band - size) // 2), text, fill=(236, 238, 243), font=font)
        note = PngInfo()
        note.add_text("Description", text)
        buffer = io.BytesIO()
        out.save(buffer, "PNG", pnginfo=note)
        return buffer.getvalue(), ""
    except Exception as exc:
        return png, f"no caption: {type(exc).__name__}: {str(exc).strip()[:160]}"


def engine_label(stand) -> str:
    """What drew a frame, as its caption says it: never the owner's own window."""
    if stand.engine == "chromium":
        return f"Chromium {stand.version} headless" if stand.version else "Chromium headless"
    return f"WebKit emulation (Playwright WebKit {stand.version})" if stand.version else "WebKit emulation (Playwright)"


def _slug(text) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()


def _walk(stand, places, frames, collect):
    """Visit every place and view; collect(name) is called on each screen (a place's list before its pages too), a
    place or view that does not show is skipped."""
    stand.open()
    for place in (places or stand.places()):
        why, _ = stand.nav(place)
        if why:
            print(f"[live_stand] the place '{place}' {why}; skipped", flush=True)
            continue
        tabs = stand.tabs()
        if not tabs or stand.landing():
            collect(place)
        for t in tabs:
            why, _ = stand.tab(t["i"])
            if why:
                print(f"[live_stand] {place}: the view '{t['name']}' {why}; skipped", flush=True)
                continue
            collect(f"{place}-{_slug(t['name']) or t['i']}")


def cmd_capture(a):
    for theme in a.themes.split(","):
        for w in (int(x) for x in a.widths.split(",")):
            out = Path(a.out) / f"{theme}-{w}"
            with Stand(theme=theme, width=w, height=900 if w > 700 else 844, mutations=a.mutations, out=out,
                       web_root=a.web_root, url=a.url, engine=a.engine, executable_path=a.executable_path) as s:
                def collect(name, s=s, out=out):
                    s.shot(name)
                    s.shot_scroll(name, a.frames)
                    (out / f"{name}.measure.json").write_text(json.dumps(s.measure(), ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
                _walk(s, a.places.split(",") if a.places != "all" else None, a.frames, collect)
                for i, room in enumerate(s.rooms()[:a.rooms]):
                    s.nav("chat")
                    why, _ = s.open_room(room["id"])
                    if why:
                        print(f"[live_stand] the Project room '{room['name']}' {why}; skipped", flush=True)
                        continue
                    collect(f"room-{i}-{_slug(room['name'])}")
            print(theme, w, "->", out)


def cmd_diff(a):
    """Same page, same data: per screen, hold live updates, snapshot, swap the stylesheets to --cand, snapshot, swap back."""
    props = ["display", "position", "width", "height", "min-height", "margin-top", "margin-right", "margin-bottom",
             "margin-left", "padding-top", "padding-right", "padding-bottom", "padding-left", "border-top-width",
             "border-top-style", "border-top-color", "border-top-left-radius", "background-color", "color", "font-size",
             "font-weight", "line-height", "box-shadow", "row-gap", "column-gap", "opacity"]
    cand = Path(a.cand)
    lines = ["# Computed-style differences (one page, one moment of data; only the stylesheets differ)",
             f"base: {a.base}", f"cand: {a.cand}", ""]
    for theme in a.themes.split(","):
        for w in (int(x) for x in a.widths.split(",")):
            vdir = Path(a.out) / f"{theme}-{w}"
            vdir.mkdir(parents=True, exist_ok=True)
            with Stand(theme=theme, width=w, height=900 if w > 700 else 844, mutations="abort", out=vdir / "_frames",
                       web_root=a.base, url=a.url, engine=a.engine, executable_path=a.executable_path) as s:
                def serve_cand(route, request=None):
                    path = "/" + route.request.url.split("://", 1)[-1].split("/", 1)[-1].split("?", 1)[0]
                    f = cand / path[len("/__cand/static/"):]
                    if f.is_file():
                        route.fulfill(status=200, path=str(f), content_type="text/css")
                    else:
                        route.fulfill(status=404, body="")
                s.page.route("**/__cand/static/**", serve_cand)

                def collect(name, s=s):
                    s.page.add_style_tag(content="*,*::before,*::after{transition:none!important;animation:none!important}")
                    s.page.evaluate(_JS)
                    s.hold = True
                    s.page.wait_for_timeout(400)
                    before = s.page.evaluate("(p) => window.__look.snap(p)", props)
                    s.page.evaluate("() => window.__look.swap(true)")
                    s.page.wait_for_timeout(300)
                    after = s.page.evaluate("(p) => window.__look.snap(p)", props)
                    s.page.evaluate("() => window.__look.swap(false)")
                    s.page.wait_for_timeout(300)
                    s.hold = False
                    diffs = [{"el": before[k]["el"], "prop": p, "base": before[k]["s"][p], "cand": after[k]["s"][p]}
                             for k in before.keys() & after.keys() for p in props if before[k]["s"][p] != after[k]["s"][p]]
                    (vdir / f"{name}.json").write_text(json.dumps(diffs, indent=0, ensure_ascii=False), encoding="utf-8")
                    moved = len(before.keys() ^ after.keys())
                    lines.append(f"- {theme}-{w} {name}: {len(diffs)} differences on {len({d['el'] for d in diffs})} elements"
                                 + (f" ({moved} elements re-rendered between the snapshots, not compared)" if moved else ""))
                _walk(s, a.places.split(",") if a.places != "all" else None, 0, collect)
            (Path(a.out) / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# look: the places a change touches, measured ---------------------------------------------------------------------------
def _first_line(exc) -> str:
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0][:300]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:200]


COUNTERS = ("writes_answered_locally", "dropped_ws_frames", "draft_files_served")
EXAMPLES, MATCHED = 4, 1000   # examples a screen keeps per kind; in a pair, findings kept until the two are matched


def _facts(found: dict) -> dict:
    """One screen's measurements as the report keeps them: counts per kind first, then a few examples of each (those a
    pair marked new first)."""
    layout = found.get("layout") or {}
    facts = layout.get("facts") or {}
    counts = dict(facts.get("counts") or {})
    if found["uneven_rows"]:
        counts["uneven-rows"] = len(found["uneven_rows"])
    if found["long_texts"]:
        counts["long-text"] = len(found["long_texts"])
    unmeasured = (f"measure.js failed: {found['layout_error']}" if found.get("layout_error")
                  else "" if layout else "measure.js was not beside the stand")
    kept, per_kind = [], Counter()
    for item in sorted(layout.get("findings") or [], key=lambda item: not item.get("new")):
        per_kind[item.get("kind")] += 1
        if per_kind[item.get("kind")] <= EXAMPLES:
            kept.append(item)
    return {"counts": counts, "findings": kept, "uneven_rows": found["uneven_rows"][:5],
            "long_texts": [dict(t, text=t["text"][:160]) for t in found["long_texts"][:5]],
            "page": facts.get("page"), "text_sizes": facts.get("textSizes"),
            **({"unmeasured": unmeasured} if unmeasured else {})}


def _key(item) -> tuple:   # a finding's kind and its elements' identities (measure.js keys), else where it stands
    return item.get("kind"), item.get("key") or item.get("where")


def _observed(found: dict):
    """A screen's observations keyed by kind and the identity of what each is about (what measure.js counted beyond
    the examples it listed, by kind alone), or None when measure.js did not measure the screen."""
    layout = found.get("layout")
    if not layout or found.get("layout_error"):
        return None
    findings = layout.get("findings") or []
    keys, listed = Counter(map(_key, findings)), Counter(item.get("kind") for item in findings)
    for kind, total in ((layout.get("facts") or {}).get("counts") or {}).items():
        keys[(kind, None)] += max(0, total - listed[kind])
    keys.update(("uneven-rows", tuple(row.get("items") or ())) for row in found["uneven_rows"])
    keys.update(("long-text", text.get("el")) for text in found["long_texts"])
    return keys


def _mark_new(found: dict, served: dict) -> dict:
    """What the draft shows beyond the served page on the same screen, matched observation by observation, so a fixed
    defect does not hide a new one of its kind; each example marked. Unknown when either page went unmeasured."""
    mine, was = _observed(found), _observed(served)
    if mine is None or was is None:
        return {"new_unknown": f"measure.js did not measure the {'draft' if mine is None else 'served page'}, so "
                               "what the draft adds is unknown"}
    added = mine - was
    left = Counter(added)
    for item in (found.get("layout") or {}).get("findings") or []:
        key = _key(item)
        item["new"] = left[key] > 0
        left[key] -= item["new"]
    return {"new": dict(Counter(kind for kind, _where in added.elements()))}


def _note(notes, text):
    if text not in notes:
        notes.append(text)


def _dotted(names, keep=4) -> str:   # a single name stands where "and 1 more" would
    names = sorted(names)
    keep = len(names) if len(names) <= keep + 1 else keep
    return "." + ", .".join(names[:keep]) + (f" and {len(names) - keep} more" if len(names) > keep else "")


def _counterparts(views, twin_views) -> dict:
    """{index of a view: index of the twin's view that opens the same panel or page, of the same name when neither
    names what it opens}; a renamed tab keeps its counterpart, a view the twin lacks has none."""
    key = lambda view: ("opens", view["target"]) if view.get("target") else ("name", view["name"])   # noqa: E731
    free, out = list(twin_views or []), {}
    for view in views:
        match = next((other for other in free if key(other) == key(view)), None)
        if match is not None:
            free.remove(match)
            out[view["i"]] = match["i"]
    return out


def _screens(stand, plan, known, discover, rooms, notes, combo, missed):
    """(place, view, screen, served) for each screen this pass visits, after going there: ``view`` is the view or
    room pressed; ``served`` is "" when the twin shows the same screen, why it does not, or None without a twin. A
    place, view or room that does not show when pressed is never measured under its name: it goes to ``missed``."""
    present = stand.places()
    if plan.get("named") and not plan["all"]:
        wanted = [place for place in plan["pages"] if place in present]
        for place in sorted(set(plan["pages"]) - set(present)):
            _note(notes, f"no place '{place}' on the page (its places: {', '.join(present)})")
    elif plan["all"] or discover:
        wanted = present
    else:
        wanted = [place for place in present if place in plan["pages"] or any(seen["place"] == place for seen in known.values())]
    for place in wanted:
        why, served = stand.nav(place)
        if why:
            missed.append(f"{combo}: the place '{place}' {why}, so it was not measured")
            continue
        views = stand.tabs()
        if not views or stand.landing():   # the place's own screen, or the list it stands on before its pages
            yield place, None, place, served
        twin_views = stand.on_twin(stand.tabs) if views and served == "" else None
        counterpart = _counterparts(views, twin_views) if twin_views is not None else {}
        for view in views:
            why, twin = stand.tab(view["i"], counterpart.get(view["i"]))
            if why:
                missed.append(f"{combo}: the {place} view '{view['name']}' {why}, so it was not measured")
                continue
            yield place, dict(view, shows=view.get("target") or ""), f"{place}-{_slug(view['name']) or view['i']}", served or twin
    listed = stand.rooms() if rooms else []
    if len(listed) < rooms:
        _note(notes, f"{len(listed)} Project room(s) can be opened on the page, fewer than the {rooms} asked for")
    for i, room in enumerate(listed[:rooms]):
        stand.nav("chat")
        why, served = stand.open_room(room["id"])
        if why:
            missed.append(f"{combo}: the Project room '{room['name']}' {why}, so it was not measured")
            continue
        yield "room", dict(room, room=True), f"room-{i}-{_slug(room['name']) or 'project'}", served


def _why(stand, plan, place, view, screen, known, discover, served=None) -> str:
    """Why this screen is looked at, or "" when the change does not touch it. The search for the changed classes looks
    on both pages of a pair, so a class the draft hides is still looked at where it stood."""
    if discover:
        hits = set(stand.probe(plan["classes"]))
        was = set(stand.on_twin(lambda twin: stand.probe(plan["classes"], page=twin)) or ()) - hits if served == "" else set()
        if hits or was:
            known[screen] = {"place": place, "classes": sorted(hits | was), "why": "; ".join(
                ([f"shows {_dotted(hits)}"] if hits else []) + ([f"the served page shows {_dotted(was)}"] if was else []))}
    if plan.get("named") or (view and view.get("room")):
        return "asked for"
    if plan["all"]:
        return "every place"
    if place in plan["pages"]:
        return f"the {place} page imports a changed module"
    return known.get(screen, {}).get("why", "")


def _version(stand, page, theme, screen, spec, version, limit=EXAMPLES, written=lambda path: None):
    """One page's raw measurements of this screen and, when asked, its frames named and captioned with ``version``:
    (found, {"frames", "uncaptioned"?}); written(path) is told of the first frame before any scroll frame is drawn."""
    label, size = engine_label(stand), f"{stand.width}x{stand.height}"
    png = page.screenshot(timeout=120000)   # the paint measure.js waits for, and the first frame
    found = stand.measure(limit=limit, page=page)
    frames, uncaptioned = [], {}
    if spec["frames"] in ("first", "all"):
        window = f"{size}-{stand.scale:g}x-{'touch' if stand.touch else 'mouse'}"   # two windows never share a file
        name = _safe(f"{spec['prefix']}{version}-{stand.engine}-{theme}-{window}-{screen}")
        caption = f"{label} | {version} | {theme} | {size} @{stand.scale:g}x | {screen}"
        path = stand.out / f"{name}.png"
        data, error = caption_frame(png, caption, stand.scale)
        path.write_bytes(data)
        written(path)
        frames.append(str(path))
        if error:
            uncaptioned[str(path)] = error
        if spec["frames"] == "all":
            frames += [str(p) for p in stand.shot_scroll(name, caption=caption, page=page)]
            uncaptioned.update({p: stand.caption_errors[p] for p in frames if p in stand.caption_errors})
    return found, {"frames": frames, **({"uncaptioned": uncaptioned} if uncaptioned else {})}


def _look_screen(stand, viewport, theme, place, view, screen, why, spec, served, hold=lambda line, keep=False: None) -> dict:
    """One screen: the twin's served version first when it shows the same screen (held as an unfinished pair from its
    first frame on, until the draft beside it is measured), then the page's own; a screen the served interface lacks
    is named as such, and so is a served frame left on disk by a twin that stopped among its scroll frames."""
    pair, held = served == "" and stand.twin is not None, []

    def first(path):
        held.append(path.name)
        hold(f"{screen}: the served version was measured (its frame {path.name}), the draft beside it was not, so the "
             "pair is unfinished")
    before = stand.on_twin(lambda twin: _version(stand, twin, theme, screen, spec, "served", MATCHED, first)) if pair else None
    if before is None and held:
        hold(f"{screen}: the served page stopped after its frame {held[0]} was written, so the draft has no served "
             "version beside it", keep=True)
    version = "draft" if stand.web_root else "served"
    found, part = _version(stand, stand.page, theme, screen, spec, version, MATCHED if pair else EXAMPLES)
    marks = _mark_new(found, before[0]) if before is not None else {}
    out = {"screen": screen, "place": place, "tab": view["name"] if view else None,
           "shows": (view or {}).get("shows") or (view or {}).get("id"), "engine": stand.engine,
           "label": engine_label(stand), "theme": theme, "viewport": viewport["name"],
           "size": f"{stand.width}x{stand.height}@{stand.scale:g}", "touch": stand.touch, "why": why, "version": version,
           **_facts(found), **part, **marks}
    if before is not None:
        out["served"] = {**_facts(before[0]), **before[1]}
    elif served:
        out["served"] = {"absent": served}
    return out


def _look_once(report, plan, engine, viewport, theme, known, deadline, spec, save) -> str:
    """One engine, viewport and theme, its screens measured into ``report``: "measured", "failed" (it broke before a
    screen, named in the notes) or "unavailable" (its browser did not start)."""
    combo, state = f"{engine['engine']} {theme} {viewport['name']}", report["engines"][-1]
    stand = Stand(theme=theme, width=int(viewport["width"]), height=int(viewport["height"]),
                  scale=float(viewport.get("scale") or 2), engine=engine["engine"], mutations="ok", out=spec["out"],
                  web_root=spec["web_root"], url=report["url"], executable_path=engine.get("executable_path"),
                  sandbox=engine.get("sandbox", True), fallback=False, touch=viewport.get("touch"), keep_intercepted=False,
                  launch_args=engine.get("args") or (), twin=spec["twin"])
    try:
        stand.__enter__()
    except Exception as exc:
        state.update(status="unavailable", error=_first_line(exc))
        return "unavailable"
    base, before, outcome, twin_named = {key: report[key] for key in COUNTERS}, len(report["screens"]), "measured", False
    misses, pending = [], []   # what did not show when pressed; a pair whose draft is not measured yet

    def checkpoint():   # the counters as they stand now, so a look ended between two screens keeps them
        nonlocal twin_named
        report["writes_answered_locally"] = base["writes_answered_locally"] + len(stand.intercepted)
        report["dropped_ws_frames"] = base["dropped_ws_frames"] + len(stand.dropped_ws)
        report["draft_files_served"] = base["draft_files_served"] + stand.served_from_root
        for path in stand.missing_from_root:
            _note(report["draft_files_missing"], path)
        for line in misses:
            _note(report["not_looked"], line)
        if stand.twin_error and not twin_named:
            twin_named = True
            report["not_looked"].append(f"{combo}: the served page beside the draft stopped ({stand.twin_error}), so "
                                        "the screens after that have no served version to compare with")
        save()

    def hold(line, keep=False):   # on disk at once (replacing the line held before): an interruption leaves it named
        while pending:
            report["not_looked"].remove(pending.pop())
        report["not_looked"].append(f"{combo}: {line}")
        pending.extend([] if keep else report["not_looked"][-1:])
        save()

    discover = bool(plan["classes"]) and not (plan.get("named") or plan["all"]) and viewport["name"] not in report["discovered"]
    try:
        state.update(status="ran", label=engine_label(stand))
        stand.open()
        report["panels"].setdefault(viewport["name"], stand.panels())
        for place, view, screen, served in _screens(stand, plan, known, discover, spec["rooms"], report["notes"], combo, misses):
            if time.monotonic() > deadline:
                report["not_looked"].append(f"{combo}: {screen} and the screens after it, the time budget was spent")
                break
            why = _why(stand, plan, place, view, screen, known, discover, served)
            if why:
                report["screens"].append(_look_screen(stand, viewport, theme, place, view, screen, why, spec, served, hold))
                while pending:
                    report["not_looked"].remove(pending.pop())
                checkpoint()
        else:
            if discover and misses:   # a view that did not show was not searched: a later theme searches again
                report["notes"].append(f"{combo}: the search for the changed classes did not finish, "
                                       f"{len(misses)} place(s), view(s) or room(s) did not show")
            elif discover:   # every screen was searched, so later themes of this window walk only the places found
                report["discovered"].append(viewport["name"])
    except ShellError as exc:
        report["notes"].append(f"{combo}: {exc}, so nothing was measured there")
        outcome = "failed"
    except Exception as exc:
        report["notes"].append(f"{combo}: stopped by {type(exc).__name__}: {_first_line(exc)}"
                               + ("; the search for the changed classes did not finish" if discover else ""))
        outcome = "failed" if len(report["screens"]) == before else "measured"
    finally:
        try:
            checkpoint()
        finally:
            stand.__exit__(None, None, None)
    return outcome


def look(spec: dict, report_path) -> dict:
    """One look, kept at ``report_path`` from its start and rewritten after every screen, so a stopped run keeps what
    it saw.

    Spec keys: url; web_root (the folder to show, else what the server serves); base (the served web/ the diff is read
    against); changed (files instead of the diff); places (names instead of the plan, ["all"] for every place);
    viewports [{name, width, height, scale, touch}]; themes; engines [{engine, executable_path?, sandbox?, args?}];
    frames none|first|all (with a web_root and frames, the served page is looked at beside it); frames_dir;
    frame_prefix; rooms (Project rooms, looked at whatever the plan); budget_sec."""
    started, report_path = time.monotonic(), Path(report_path)
    deadline = started + max(10.0, float(spec.get("budget_sec") or 480))
    web_root, served, names = spec.get("web_root") or None, spec.get("base") or None, spec.get("places") or None
    frames, rooms = spec.get("frames") or "none", max(0, int(spec.get("rooms") or 0))
    report = {"schema": 1, "url": spec.get("url") or live_url(), "web_root": web_root, "served_web": served, "plan": None,
              "viewports": spec["viewports"], "themes": spec["themes"], "frames": frames, "rooms": rooms,
              "engines": [], "screens": [], "panels": {}, "discovered": [], "not_looked": [], "notes": [],
              "writes_answered_locally": 0, "dropped_ws_frames": 0, "draft_files_served": 0, "draft_files_missing": [],
              "playwright": True, "finished": False}

    def save():
        report["elapsed_sec"] = round(time.monotonic() - started, 1)
        part = report_path.with_name(report_path.name + ".part")
        part.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(part, report_path)

    save()   # before the diff is read: a Stop while it is read still finds a report
    if names:
        plan = {"all": list(names) == ["all"], "pages": [] if list(names) == ["all"] else list(names), "classes": [],
                "why": ["the places asked for"], "changed": [], "unplaced": [], "named": True}
    elif web_root or served:
        try:
            plan = place_plan(web_root or served, served if web_root else None, spec.get("changed"))
        except OSError as exc:   # an unreadable web/ folder: say so instead of guessing places
            plan = {"all": False, "pages": [], "classes": [], "changed": [], "unplaced": [],
                    "why": [f"the diff could not be read ({_first_line(exc)})"]}
    else:
        plan = {"all": True, "pages": [], "classes": [], "why": ["no web/ folder to read a diff from, so every place"],
                "changed": [], "unplaced": []}
    if not (plan["all"] or plan["pages"] or plan["classes"]):
        plan["why"].append(f"the {rooms} Project room(s) asked for are looked at all the same" if rooms else "nothing to look at")
    report["plan"] = plan
    save()
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError as exc:
        report.update(playwright=False, finished=True, notes=[f"Playwright is not importable here: {_first_line(exc)}"])
        save()
        return report
    look_spec = {"out": Path(spec.get("frames_dir") or report_path.parent), "web_root": web_root, "rooms": rooms,
                 "prefix": str(spec.get("frame_prefix") or "look-"), "frames": frames,
                 "twin": bool(web_root) and frames in ("first", "all")}
    if web_root and not look_spec["twin"]:
        report["notes"].append("new and earlier findings are not told apart: a call with frames 'first' also looks at "
                               "the served interface beside the draft and marks what the draft adds")
    if not rooms and (plan["all"] or "chat" in plan["pages"] or plan["classes"]):
        report["notes"].append("no Project room was opened (rooms is 0), though a room draws the chat and its parts too")
    known: dict = {}
    outcomes = []
    if plan["all"] or plan["pages"] or plan["classes"] or rooms:
        for engine in spec["engines"]:
            report["engines"].append({"engine": engine["engine"], "status": "not started"})
            for viewport in spec["viewports"]:
                for theme in spec["themes"]:
                    if report["engines"][-1]["status"] == "unavailable":
                        break
                    if time.monotonic() > deadline:
                        report["not_looked"].append(f"{engine['engine']} {theme} {viewport['name']}: the time budget was spent first")
                        continue
                    outcomes.append(_look_once(report, plan, engine, viewport, theme, known.setdefault(viewport["name"], {}),
                                               deadline, look_spec, save))
                    save()
    stood = {name for window in known.values() for seen in window.values() for name in seen.get("classes", ())}
    unseen = sorted(set(plan["classes"]) - stood) if report["discovered"] and not (plan["all"] or plan.get("named")) else []
    if unseen:   # a class the change touches that no searched screen showed: where it stands was not looked at
        where = "the draft or the served page beside it" if look_spec["twin"] else "the draft" if web_root else "the interface"
        report["not_looked"].append(
            f"{_dotted(unseen, len(unseen))} stood on no screen of {where} in the windows searched ({', '.join(report['discovered'])}), "
            "so where it stands was not looked at" + ("; a call with frames 'first' also looks at the served interface "
                                                      "beside the draft, which shows where it was" if web_root and not look_spec["twin"] else ""))
    named = plan["pages"] if plan.get("named") and not plan["all"] else []
    seen = {screen["place"] for screen in report["screens"]}
    report["not_looked"] += [f"the place '{place}' was asked for, yet no screen of it was measured" for place in named if place not in seen]
    tried = [outcome for outcome in outcomes if outcome != "unavailable"]
    if tried and all(outcome == "failed" for outcome in tried):
        report["failed"] = "every engine, theme and window tried broke before a screen was measured; the notes say why"
    elif tried and named and not report["screens"]:
        report["failed"] = "no screen of the places asked for was measured; not_looked and the notes say why"
    report["finished"] = True
    save()
    return report


def cmd_look(a):
    report = look(json.loads(Path(a.spec).read_text(encoding="utf-8")), a.report)
    print(json.dumps({"screens": len(report["screens"]), "finished": report["finished"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "diff"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--places", default="all")
        p.add_argument("--themes", default="dark,light" if name == "capture" else "dark")
        p.add_argument("--widths", default="1440,390" if name == "capture" else "1440")
        p.add_argument("--url", default=None)
        p.add_argument("--engine", choices=("webkit", "chromium"), default="webkit")
        p.add_argument("--executable-path", default=None)
        if name == "capture":
            p.add_argument("--web-root", default=None)
            p.add_argument("--mutations", choices=("ok", "abort"), default="ok")
            p.add_argument("--rooms", type=int, default=0)
            p.add_argument("--frames", type=int, default=6)
        else:
            p.add_argument("--base", required=True)
            p.add_argument("--cand", required=True)
    p = sub.add_parser("look")
    p.add_argument("--spec", required=True)
    p.add_argument("--report", required=True)
    a = ap.parse_args()
    t0 = time.time()
    {"capture": cmd_capture, "diff": cmd_diff, "look": cmd_look}[a.cmd](a)
    print(f"done in {round(time.time() - t0)} s")


if __name__ == "__main__":
    main()
