"""Look at Ouroboros's own live interface without changing anything — an optional instrument of the ui_design skill.

When it helps: you changed a shared part of the interface (a control, a card, a status line, a list) and want every place
that uses it before and after, on the owner's real data, in both themes and at desk and phone widths; or the owner asks
you to look over the whole interface. It returns pictures and numbers, never a verdict. Nothing runs it on its own: not
tests, not CI, not a release step, not a schedule.

Safety, enforced here (do not open the live interface any other way while using it):
  * only plain reads (GET/HEAD) reach the live server; reads that ask it for fresh work (?fresh= ?refresh= ?force=
    ?probe= ?checkLatest= ...) are answered locally;
  * every write is answered locally: --mutations ok fakes success (to see a state after Save), abort shows the failure state;
  * every frame the page sends on its WebSocket is dropped (server -> page frames still flow, so the page stays live).
  So pressing Save, Stop, Restart or Panic inside it changes nothing on the install.

Before/after of code you have not merged: --web-root <worktree>/web serves the interface (index.html and /static/*) from
your branch while all data still comes from the live server (the running server's API must match your branch's).

CLI (run with a Python that imports Playwright; the same browsers browse_page uses):
  python live_stand.py capture --out DIR [--places all|chat,dashboard,...] [--themes dark,light] [--widths 1440,390]
                               [--web-root PATH/web] [--mutations ok|abort] [--rooms N] [--url URL]
      -> DIR/<theme>-<width>/<place>[-<tab>].png, __pNN.png scroll frames, .measure.json (uneven rows of controls,
         long texts) ; places and tabs are discovered from the page itself ([data-nav-page], [role="tab"]).
  python live_stand.py diff --base PATH/web --cand PATH/web --out DIR [--themes dark] [--widths 1440] [--places ...]
      -> DIR/<theme>-<width>/<place>.json computed-style differences, DIR/SUMMARY.md. One page and one moment of data:
         on every screen the stylesheets are swapped from --base to --cand and back while live updates are held, so a
         difference is caused by the stylesheets alone. It measures CSS changes; a JavaScript change shows in captures.
  --engine webkit|chromium (default webkit, the owner's macOS app engine; falls back to chromium when WebKit is missing).
Library: from live_stand import Stand; with Stand(theme="dark", width=1440) as s: s.open(); s.nav("settings"); s.shot("x").
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

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
            port = int(Path(c).read_text().strip()) if c else 0
        except (OSError, ValueError):
            port = 0
        if port:
            return f"http://127.0.0.1:{port}"
    return f"http://127.0.0.1:{int(os.environ.get('OUROBOROS_SERVER_PORT', '8765'))}"


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
  const tabs = () => { const page = [...document.querySelectorAll('.page.active, .page:not([hidden])')].find(vis) || document;
    return [...page.querySelectorAll('[role="tab"]')].filter(vis).map((t, i) => ({ i, name: (t.innerText || t.getAttribute('aria-label') || String(i)).trim().slice(0, 24) })); };
  window.__look = { vis, visualRows, texts, scroller, places, tabs, snap, swap };
})();
"""


class Stand:
    """One headless browser on the live interface with the safety above. Use as a context manager."""

    hold = False   # True keeps server -> page socket frames (live updates) back until it is set to False again

    def __init__(self, theme="dark", width=1440, height=900, scale=2, engine="webkit", mutations="ok",
                 out=None, web_root=None, url=None):
        self.theme, self.width, self.height, self.scale, self.engine = theme, width, height, scale, engine
        self.mutations = mutations
        self.out = Path(out) if out else Path.cwd() / "live-stand"
        self.out.mkdir(parents=True, exist_ok=True)
        self.web_root = Path(web_root) if web_root else None
        self.url = url or live_url()
        self.intercepted: list[dict] = []
        self.dropped_ws: list[str] = []

    def __enter__(self):
        self._pw = sync_playwright().start()
        try:
            self._browser = getattr(self._pw, self.engine).launch()
        except Exception as exc:   # the engine is not installed in this browsers root
            if self.engine == "chromium":
                raise
            print(f"[live_stand] {self.engine} unavailable ({str(exc).splitlines()[0][:120]}); using chromium", flush=True)
            self.engine = "chromium"
            self._browser = self._pw.chromium.launch()
        self.ctx = self._browser.new_context(viewport={"width": self.width, "height": self.height},
                                             device_scale_factor=self.scale, color_scheme=self.theme,
                                             service_workers="block", has_touch=self.width < 700)
        self.ctx.route("**/*", self._http)
        self.ctx.route_web_socket(re.compile(r".*"), self._ws)
        self.page = self.ctx.new_page()
        self.page.set_default_timeout(30000)
        return self

    def __exit__(self, *exc):
        try:
            (self.out / "_intercepted.json").write_text(json.dumps(
                {"writes_answered_locally": self.intercepted, "dropped_ws_frames": self.dropped_ws[:200]}, indent=1))
        finally:
            self._browser.close()
            self._pw.stop()

    def _http(self, route):
        req = route.request
        path = "/" + req.url.split("://", 1)[-1].split("/", 1)[-1].split("?", 1)[0].split("#", 1)[0]
        if self.web_root and req.method in ("GET", "HEAD"):
            local = self.web_root / path[len("/static/"):] if path.startswith("/static/") else (
                self.web_root / "index.html" if path in ("/", "/index.html") else None)
            if local is not None and local.is_file():
                route.fulfill(status=200, path=str(local))
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

    # navigation ------------------------------------------------------------------------------------------------
    def open(self, wait_ms=8000):
        self.page.goto(self.url + "/", wait_until="domcontentloaded")
        try:
            self.page.wait_for_selector("[data-nav-page]", state="attached", timeout=60000)
        except Exception:
            pass
        self.page.wait_for_timeout(wait_ms)
        self.page.evaluate(_JS)
        return self

    def places(self):
        self.page.evaluate(_JS)
        return self.page.evaluate("() => window.__look.places()")

    def nav(self, place, wait_ms=5000):
        self.page.evaluate("(id) => document.querySelector(`[data-nav-page=\"${id}\"]`)?.click()", place)
        self.page.wait_for_timeout(wait_ms)
        self.page.evaluate(_JS)
        return self

    def tabs(self):
        return self.page.evaluate("() => window.__look.tabs()")

    def tab(self, index, wait_ms=4000):
        self.page.evaluate("(i) => { const page = [...document.querySelectorAll('.page.active, .page:not([hidden])')].find(window.__look.vis) || document;"
                           " const t = [...page.querySelectorAll('[role=\"tab\"]')].filter(window.__look.vis)[i]; t && t.click(); }", index)
        self.page.wait_for_timeout(wait_ms)
        return self

    def open_room(self, index, wait_ms=6000):
        self.page.evaluate("(i) => { const r = [...document.querySelectorAll('.nav-project-row')][i]; r && r.click(); }", index)
        self.page.wait_for_timeout(wait_ms)
        return self

    # capture and measure ---------------------------------------------------------------------------------------------
    def shot(self, name):
        p = self.out / f"{name}.png"
        # a busy live server can serve a web font slowly; the screenshot waits for fonts, so allow it time
        self.page.screenshot(path=str(p), timeout=120000)
        return p

    def shot_scroll(self, name, max_frames=8):
        n = self.page.evaluate("() => { const s = window.__look.scroller(); if (!s) return 0; s.setAttribute('data-look-scroller', '1');"
                               " s.scrollTop = 0; return Math.ceil(s.scrollHeight / Math.max(200, s.clientHeight - 80)); }")
        for i in range(1, min(n, max_frames)):
            self.page.evaluate("(i) => { const s = document.querySelector('[data-look-scroller]'); s.scrollTop = i * Math.max(200, s.clientHeight - 80); }", i)
            self.page.wait_for_timeout(350)
            self.shot(f"{name}__p{i:02d}")
        self.page.evaluate("() => { const s = document.querySelector('[data-look-scroller]'); if (s) { s.scrollTop = 0; s.removeAttribute('data-look-scroller'); } }")

    def measure(self):
        self.page.evaluate(_JS)
        return {"uneven_rows": [r for r in self.page.evaluate("() => window.__look.visualRows(64)") if r["spread"] >= 3],
                "long_texts": self.page.evaluate("() => window.__look.texts(160)")}


def _walk(stand, places, frames, collect):
    """Visit every place and tab; collect(name) is called on each screen."""
    stand.open()
    for place in (places or stand.places()):
        stand.nav(place)
        tabs = stand.tabs()
        if not tabs:
            collect(place)
            continue
        for t in tabs:
            stand.tab(t["i"])
            collect(f"{place}-{re.sub(r'[^A-Za-z0-9]+', '_', t['name']).strip('_').lower() or t['i']}")


def cmd_capture(a):
    for theme in a.themes.split(","):
        for w in (int(x) for x in a.widths.split(",")):
            out = Path(a.out) / f"{theme}-{w}"
            with Stand(theme=theme, width=w, height=900 if w > 700 else 844, mutations=a.mutations, out=out,
                       web_root=a.web_root, url=a.url, engine=a.engine) as s:
                def collect(name, s=s, out=out):
                    s.shot(name)
                    s.shot_scroll(name, a.frames)
                    (out / f"{name}.measure.json").write_text(json.dumps(s.measure(), ensure_ascii=False, indent=1))
                _walk(s, a.places.split(",") if a.places != "all" else None, a.frames, collect)
                for i in range(a.rooms):
                    s.nav("chat")
                    s.open_room(i)
                    collect(f"room-{i}")
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
                       web_root=a.base, url=a.url, engine=a.engine) as s:
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
                    (vdir / f"{name}.json").write_text(json.dumps(diffs, indent=0, ensure_ascii=False))
                    moved = len(before.keys() ^ after.keys())
                    lines.append(f"- {theme}-{w} {name}: {len(diffs)} differences on {len({d['el'] for d in diffs})} elements"
                                 + (f" ({moved} elements re-rendered between the snapshots, not compared)" if moved else ""))
                _walk(s, a.places.split(",") if a.places != "all" else None, 0, collect)
            (Path(a.out) / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


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
        if name == "capture":
            p.add_argument("--web-root", default=None)
            p.add_argument("--mutations", choices=("ok", "abort"), default="ok")
            p.add_argument("--rooms", type=int, default=0)
            p.add_argument("--frames", type=int, default=6)
        else:
            p.add_argument("--base", required=True)
            p.add_argument("--cand", required=True)
    a = ap.parse_args()
    t0 = time.time()
    (cmd_capture if a.cmd == "capture" else cmd_diff)(a)
    print(f"done in {round(time.time() - t0)} s")


if __name__ == "__main__":
    main()
