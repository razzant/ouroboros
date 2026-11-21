"""Look at Ouroboros's own live interface without changing anything — an optional instrument of the ui_design skill.

When it helps: you changed a shared part of the interface (a control, a card, a status line, a list) and want every place
that uses it before and after, on the owner's real data, in both themes and at desk and phone widths; or the owner asks
you to look over the whole interface. It returns pictures and numbers, never a verdict. Nothing runs it on its own or as
a check: it runs when someone asks (a command below, or Ouroboros's `ui_look` tool), never as a gate of tests, CI, a
release step or a schedule.

Safety, enforced here (do not open the live interface any other way while using it):
  * only plain reads (GET/HEAD) reach the live server; reads that ask it for fresh work (?fresh= ?refresh= ?force=
    ?probe= ?checkLatest= ...) are answered locally;
  * every write is answered locally: --mutations ok fakes success (to see a state after Save), abort shows the failure state;
  * every frame the page sends on its WebSocket is dropped (server -> page frames still flow, so the page stays live).
  So pressing Save, Stop, Restart or Panic inside it changes nothing on the install.

Before/after of code you have not merged: --web-root <worktree>/web serves the interface (index.html and /static/*) from
your branch while all data still comes from the live server (the running server's API must match your branch's).

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
      -> the places a change touches (place_plan: by its diff against the served web/) or the places SPEC names, at
         each of its viewports, themes and engines: per screen the measurements (measure.js beside this file: covered
         text, small targets, overflow, cut text; plus uneven rows and long texts) in OUT.json, rewritten after every
         screen, and on request frames as flat files captioned with the engine that drew them. Ouroboros's `ui_look`
         tool writes SPEC; its keys are read in look().
  --engine webkit|chromium (default webkit, the owner's macOS app engine; falls back to chromium when WebKit is missing).
  --executable-path PATH launches exactly that Chromium (Ouroboros's pinned headless shell, as browse_page) and nothing
  else: no fallback to another engine or build.
Library: from live_stand import Stand; with Stand(theme="dark", width=1440) as s: s.open(); s.nav("settings"); s.shot("x").
"""
from __future__ import annotations

import argparse
import io
import json
import os
import posixpath
import re
import time
from collections import Counter
from pathlib import Path

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


def _sibling_text(name: str) -> str:
    try:
        return Path(__file__).resolve().with_name(name).read_text(encoding="utf-8")
    except OSError:
        return ""


MEASURE_JS = _sibling_text("measure.js")  # the ui_design snippet beside this file; measure() calls it on every screen

# The places a change touches ---------------------------------------------------------------------------------------
# A place is one [data-nav-page] of the shell (web/modules/pages.js). The shell and its tokens touch every place. A
# module touches the pages whose own module imports it, and the screens that show a class only it draws. A stylesheet
# touches the screens that show a class of a rule it changed. Classes are looked for on the screens themselves (look()),
# so a part an app-level module draws (a Dashboard panel) is found where it stands.
SHELL_FILES = ("index.html", "app.js", "theme.js", "ui.css")
_NOT_INTERFACE_DIRS = ("tests", "node_modules", "__pycache__")
_NOT_INTERFACE_FILES = ("package.json", "package-lock.json", "eslint.config.js", "__init__.py")
_IMPORT = re.compile(r"""(?:\bfrom|\bimport)\s*\(?\s*['"](\.{1,2}/[^'"\s]+)['"]""")
_NAMED_IMPORT = re.compile(r"""import\s*\{([^}]*)\}\s*from\s*['"](\.{1,2}/[^'"]+)['"]""")
_PAGE_ROW = re.compile(r"""\bid:\s*['"]([\w-]+)['"][^}]*?\binit:\s*['"](\w+)['"]""")
_CLASS_ATTR = re.compile(r"""\bclass(?:Name)?\s*=\s*\\?["'`]([^"'`]*)""")
_CLASS_LIST = re.compile(r"""classList\.(?:add|remove|toggle|replace)\(([^)]*)\)""")
_QUOTED = re.compile(r"""["'`]([A-Za-z][\w-]*)["'`]""")
_CLASS_TOKEN = re.compile(r"^[A-Za-z][\w-]+$")
_SELECTOR_CLASS = re.compile(r"\.(-?[_A-Za-z][\w-]*)")
_CUSTOM_PROPERTY = re.compile(r"(?:^|[;\s])--[\w-]+\s*:")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _interface_files(web) -> dict:
    """The interface's files under ``web``: relative POSIX path -> path (its tests and package files left out)."""
    out = {}
    for folder, dirs, files in os.walk(web):
        dirs[:] = sorted(d for d in dirs if d not in _NOT_INTERFACE_DIRS and not d.startswith("."))
        for name in files:
            if name not in _NOT_INTERFACE_FILES and not name.startswith(".") and not name.endswith((".md", ".py", ".pyc")):
                path = Path(folder) / name
                out[path.relative_to(web).as_posix()] = path
    return out


def changed_files(web, base) -> list:
    """Interface files that differ between two web/ folders, added and removed ones included."""
    old, new = _interface_files(Path(base)), _interface_files(Path(web))

    def same(rel):
        a, b = old.get(rel), new.get(rel)
        return a is not None and b is not None and a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()
    return sorted(rel for rel in set(old) | set(new) if not same(rel))


def _web_relative(item, roots):
    """A changed path as a path under web/, or None when it lies outside the interface."""
    text = str(item).replace("\\", "/").strip()
    if Path(text).is_absolute():
        for root in roots:
            try:
                return Path(text).resolve().relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                continue
        return None
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if "web" in parts:
        return "/".join(parts[len(parts) - parts[::-1].index("web"):]) or None
    rel = "/".join(parts)
    return rel if rel and any((Path(root) / rel).exists() for root in roots) else None


def _drawn_classes(text: str) -> set:
    """Classes a script or page draws: class attributes, className values and classList calls (template parts dropped)."""
    found = set()
    for value in _CLASS_ATTR.findall(text):
        found.update(t for t in re.sub(r"\$\{[^}]*\}?", " ", value).split() if _CLASS_TOKEN.match(t))
    for args in _CLASS_LIST.findall(text):
        found.update(_QUOTED.findall(args))
    return found


def _css_rules(text: str) -> list:
    """A stylesheet's leaf rules as (selector, declarations), whitespace-normalised; group rules are flattened."""
    text = _CSS_COMMENT.sub("", text)
    rules, opened, start = [], [], 0
    for i, ch in enumerate(text):
        if ch == "{":
            opened.append((text[start:i], i + 1))
            start = i + 1
        elif ch in "};":
            if ch == "}" and opened:
                prelude, at = opened.pop()
                if "{" not in text[at:i]:
                    rules.append((" ".join(prelude.split()), " ".join(text[at:i].split())))
            start = i + 1
    return rules


def _page_modules(scripts: dict) -> dict:
    """{place id: its page module}: the places of modules/pages.js, each by the module app.js imports its init from."""
    names = {}
    for group, spec in _NAMED_IMPORT.findall(scripts.get("app.js", "")):
        for item in group.split(","):
            local = item.split(" as ")[-1].strip()
            if local:
                names[local] = posixpath.normpath(spec)
    return {page: names[init] for page, init in _PAGE_ROW.findall(scripts.get("modules/pages.js", "")) if init in names}


def _reach(graph: dict, start: str) -> set:
    seen, todo = set(), [start]
    while todo:
        item = todo.pop()
        if item not in seen:
            seen.add(item)
            todo.extend(graph.get(item, ()))
    return seen


def _scripts(web: Path) -> dict:
    """The interface's scripts read once: which pages each imports into, what each draws, what the shell reaches."""
    texts = {rel: path.read_text(encoding="utf-8", errors="replace")
             for rel, path in _interface_files(web).items() if rel.endswith((".js", ".mjs"))}
    graph = {rel: {posixpath.normpath(posixpath.join(posixpath.dirname(rel), spec)) for spec in _IMPORT.findall(text)}
             for rel, text in texts.items()}
    reach = {page: _reach(graph, module) for page, module in _page_modules(texts).items()}
    drawn = {rel: _drawn_classes(text) for rel, text in texts.items()}
    by_class: dict = {}
    for rel, names in drawn.items():
        for name in names:
            by_class.setdefault(name, set()).add(rel)
    index = web / "index.html"
    return {"pages_of": {rel: sorted(page for page, mods in reach.items() if rel in mods) for rel in texts},
            "drawn": drawn, "by_class": by_class, "shell": _reach(graph, "app.js"),
            "index": _drawn_classes(index.read_text(encoding="utf-8", errors="replace")) if index.is_file() else set()}


def _changed_classes(old: str, new: str):
    """(classes of the rules that differ between two stylesheets, "") or (None, why) when they reach every place."""
    before, after = Counter(_css_rules(old)), Counter(_css_rules(new))
    named = set()
    for selector, body in ((before - after) + (after - before)).elements():
        for one in (part.strip() for part in selector.split(",")):
            found = _SELECTOR_CLASS.findall(one)
            if one and not found:   # :root, an element: a token or a rule for everything
                return None, (f"a token changed in '{one[:60]}'" if _CUSTOM_PROPERTY.search(body)
                              else f"'{one[:60]}' names no class")
            named.update(found)   # a custom property on a class reaches only where that class stands
    return named, ""


def _names(names) -> str:
    names = sorted(names)
    return ".".join([""] + names[:1]) + "".join(f", .{n}" for n in names[1:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")


def place_plan(web, base=None, changed=None) -> dict:
    """What a change of the interface touches: {"all", "pages", "classes", "why", "changed", "unplaced"}.

    ``web`` is the folder that will be served (a candidate, or the served one), ``base`` the served one to diff it
    against; ``changed`` names the changed files instead (under web/, absolute or relative). With neither there is no
    diff to read and every place is touched. "pages" are looked at whole; "classes" are looked for on every screen."""
    web = Path(web)
    plan = {"all": False, "pages": [], "classes": [], "why": [], "changed": [], "unplaced": []}
    if changed is None and (base is None or Path(base).resolve() == web.resolve()):
        plan.update(all=True, why=["no diff to read, so every place"])
        return plan
    roots = [root for root in (web, base) if root is not None]
    files = changed_files(web, base) if changed is None else [_web_relative(item, roots) for item in changed]
    scripts, pages, probes = _scripts(web), set(), set()

    def every(reason):
        plan["all"] = True
        if len(plan["why"]) < 24:
            plan["why"].append(reason)

    for item, rel in zip(changed if changed is not None else files, files):
        if rel is None:
            plan["unplaced"].append(f"{item}: outside web/")
            continue
        plan["changed"].append(rel)
        old, new = (Path(base) / rel if base is not None else None), web / rel
        if rel in SHELL_FILES:
            every(f"{rel}: the shell or its tokens, so every place")
        elif rel.endswith(".css") and (old is None or not old.is_file() or not new.is_file()):
            every(f"{rel}: its rules were not compared with the served ones, so every place")
        elif rel.endswith(".css"):
            named, why = _changed_classes(old.read_text(encoding="utf-8", errors="replace"),
                                          new.read_text(encoding="utf-8", errors="replace"))
            if named is None:
                every(f"{rel}: {why}, so every place")
                continue
            for name in named:   # a class is looked for on the screens unless every script drawing it is a page's
                drawers = scripts["by_class"].get(name, set())
                pages.update(page for drawer in drawers for page in scripts["pages_of"].get(drawer, ()))
                if not drawers or any(not scripts["pages_of"].get(drawer) for drawer in drawers):
                    probes.add(name)
            if named:
                plan["why"].append(f"{rel}: changed rules of {_names(named)}")
        elif rel.endswith((".js", ".mjs")):
            hit = scripts["pages_of"].get(rel, [])
            own = [] if hit else [name for name in scripts["drawn"].get(rel, ())
                                  if scripts["by_class"][name] == {rel} and name not in scripts["index"]]
            pages.update(hit)
            probes.update(own)
            if hit:
                plan["why"].append(f"{rel}: imported by the {', '.join(hit)} page" + ("s" if len(hit) > 1 else ""))
            elif own:
                plan["why"].append(f"{rel}: draws {_names(own)}, looked for on the screens")
            elif rel in scripts["shell"]:
                every(f"{rel}: the shell imports it and it draws no class of its own, so every place")
            else:
                plan["unplaced"].append(f"{rel}: the interface does not import it")
        else:
            every(f"{rel}: not placed by its name or rules, so every place")
    plan["pages"], plan["classes"] = sorted(pages), sorted(probes)
    if not (plan["all"] or pages or probes):
        plan["why"].append("no changed file draws a place, so nothing to look at")
    return plan


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
  // The classes that stand on this screen: shown, in the page shown or outside every page (the shell).
  const probe = (classes) => classes.filter((c) => [...document.getElementsByClassName(c)].some((n) => vis(n)
    && (!n.closest('.page') || n.closest('.page').classList.contains('active'))));
  const panels = () => { const inline = document.documentElement.style;
    const width = (id) => { const n = document.getElementById(id); return n && vis(n) ? Math.round(n.getBoundingClientRect().width) : null; };
    return { saved_sidebar: inline.getPropertyValue('--sidebar-width').trim() || null,
             saved_project_panel: inline.getPropertyValue('--project-panel-width').trim() || null,
             sidebar: width('primary-sidebar'), project_panel: width('project-panel') }; };
  window.__look = { vis, visualRows, texts, scroller, places, tabs, snap, swap, probe, panels };
})();
"""


class Stand:
    """One headless browser on the live interface with the safety above. Use as a context manager."""

    hold = False   # True keeps server -> page socket frames (live updates) back until it is set to False again

    def __init__(self, theme="dark", width=1440, height=900, scale=2, engine="webkit", mutations="ok",
                 out=None, web_root=None, url=None, executable_path=None, sandbox=True, fallback=True, touch=None,
                 keep_intercepted=True):
        self.theme, self.width, self.height, self.scale, self.engine = theme, width, height, scale, engine
        self.mutations = mutations
        self.out = Path(out) if out else Path.cwd() / "live-stand"
        self.out.mkdir(parents=True, exist_ok=True)
        self.web_root = Path(web_root) if web_root else None
        self.url = url or live_url()
        self.executable_path = str(executable_path or "")   # an exact Chromium build: launched alone, never replaced
        if self.executable_path:
            self.engine = "chromium"
        self.sandbox, self.fallback, self.keep_intercepted = bool(sandbox), bool(fallback), bool(keep_intercepted)
        self.touch = self.width < 700 if touch is None else bool(touch)
        self.version = ""
        self.intercepted: list[dict] = []
        self.dropped_ws: list[str] = []

    def __enter__(self):
        from playwright.sync_api import sync_playwright   # here: the place plan and the CLI help need no browser

        self._pw, self._browser = sync_playwright().start(), None
        try:
            self._browser = self._launch()
            self.version = str(getattr(self._browser, "version", "") or "")
            self.ctx = self._browser.new_context(viewport={"width": self.width, "height": self.height},
                                                 device_scale_factor=self.scale, color_scheme=self.theme,
                                                 service_workers="block", has_touch=self.touch)
            self.ctx.route("**/*", self._http)
            self.ctx.route_web_socket(re.compile(r".*"), self._ws)
            self.page = self.ctx.new_page()
            self.page.set_default_timeout(30000)
        except BaseException:
            self._close()
            raise
        return self

    def _launch(self):
        if self.executable_path:   # Ouroboros's pinned headless shell, in its own sandbox as browse_page runs it
            return self._pw.chromium.launch(headless=True, executable_path=self.executable_path,
                                            chromium_sandbox=self.sandbox)
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
                    {"writes_answered_locally": self.intercepted, "dropped_ws_frames": self.dropped_ws[:200]}, indent=1))
        finally:
            self._close()

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

    def probe(self, classes):
        """Which of ``classes`` stand on this screen (shown, on the page shown or in the shell around it)."""
        self.page.evaluate(_JS)
        return self.page.evaluate("(c) => window.__look.probe(c)", sorted(classes))

    def panels(self):
        """The panel widths the page uses: the saved ones it read from the server, and the drawn ones."""
        self.page.evaluate(_JS)
        return self.page.evaluate("() => window.__look.panels()")

    # capture and measure ---------------------------------------------------------------------------------------------
    def shot(self, name, caption=""):
        p = self.out / f"{name}.png"
        # a busy live server can serve a web font slowly; the screenshot waits for fonts, so allow it time
        png = self.page.screenshot(timeout=120000)
        p.write_bytes(caption_frame(png, caption, self.scale) if caption else png)
        return p

    def shot_scroll(self, name, max_frames=8, caption=""):
        n = self.page.evaluate("() => { const s = window.__look.scroller(); if (!s) return 0; s.setAttribute('data-look-scroller', '1');"
                               " s.scrollTop = 0; return Math.ceil(s.scrollHeight / Math.max(200, s.clientHeight - 80)); }")
        frames = []
        for i in range(1, min(n, max_frames)):
            self.page.evaluate("(i) => { const s = document.querySelector('[data-look-scroller]'); s.scrollTop = i * Math.max(200, s.clientHeight - 80); }", i)
            self.page.wait_for_timeout(350)
            frames.append(self.shot(f"{name}__p{i:02d}", f"{caption} | scrolled {i}" if caption else ""))
        self.page.evaluate("() => { const s = document.querySelector('[data-look-scroller]'); if (s) { s.scrollTop = 0; s.removeAttribute('data-look-scroller'); } }")
        return frames

    def measure(self, target=None, limit=6):
        """Uneven rows and long texts, then measure.js (a 44 px target under touch: DESIGN "a coarse pointer")."""
        self.page.evaluate(_JS)
        found = {"uneven_rows": [r for r in self.page.evaluate("() => window.__look.visualRows(64)") if r["spread"] >= 3],
                 "long_texts": self.page.evaluate("() => window.__look.texts(160)")}
        if MEASURE_JS:
            options = {"limit": limit, **({"target": target} if target else {})}
            found["layout"] = self.page.evaluate(f"({MEASURE_JS})({json.dumps(options)})")
        return found


def caption_frame(png: bytes, text: str, scale=2) -> bytes:
    """The frame with a band under it naming what drew it (text and PNG note); unchanged when Pillow is missing."""
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
        return buffer.getvalue()
    except Exception:   # no Pillow, or a frame it cannot read: the frame as captured, named by its file
        return png


def engine_label(stand) -> str:
    """What drew a frame, as its caption says it: never the owner's own window."""
    if stand.engine == "chromium":
        return f"Chromium {stand.version} headless" if stand.version else "Chromium headless"
    return f"WebKit emulation (Playwright WebKit {stand.version})" if stand.version else "WebKit emulation (Playwright)"


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
                       web_root=a.web_root, url=a.url, engine=a.engine, executable_path=a.executable_path) as s:
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
                    (vdir / f"{name}.json").write_text(json.dumps(diffs, indent=0, ensure_ascii=False))
                    moved = len(before.keys() ^ after.keys())
                    lines.append(f"- {theme}-{w} {name}: {len(diffs)} differences on {len({d['el'] for d in diffs})} elements"
                                 + (f" ({moved} elements re-rendered between the snapshots, not compared)" if moved else ""))
                _walk(s, a.places.split(",") if a.places != "all" else None, 0, collect)
            (Path(a.out) / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


# look: the places a change touches, measured ---------------------------------------------------------------------------
def _first_line(exc) -> str:
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0][:300]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:180]


def _facts(found: dict) -> dict:
    """One screen's measurements as the report keeps them: counts per kind first, then their examples."""
    layout = found.get("layout") or {}
    facts = layout.get("facts") or {}
    counts = dict(facts.get("counts") or {})
    if found["uneven_rows"]:
        counts["uneven-rows"] = len(found["uneven_rows"])
    if found["long_texts"]:
        counts["long-text"] = len(found["long_texts"])
    return {"counts": counts, "findings": layout.get("findings") or [], "uneven_rows": found["uneven_rows"][:5],
            "long_texts": [dict(t, text=t["text"][:160]) for t in found["long_texts"][:5]],
            "page": facts.get("page"), "text_sizes": facts.get("textSizes"),
            **({} if layout else {"unmeasured": "measure.js was not beside the stand"})}


def _note(notes, text):
    if text not in notes:
        notes.append(text)


def _screens(stand, plan, known, discover, rooms, notes):
    """(place, tab, screen) for each screen this pass visits; it navigates there before yielding."""
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
        stand.nav(place)
        tabs = stand.tabs()
        if not tabs:
            yield place, None, place
        for tab in tabs:
            stand.tab(tab["i"])
            yield place, tab, f"{place}-{re.sub(r'[^A-Za-z0-9]+', '_', tab['name']).strip('_').lower() or tab['i']}"
    for i in range(rooms):
        if not stand.page.evaluate("(i) => document.querySelectorAll('.nav-project-row').length > i", i):
            _note(notes, f"only {i} Project room(s) on the page, so fewer rooms were looked at")
            break
        stand.nav("chat")
        stand.open_room(i)
        yield "chat", None, f"room-{i}"


def _why(stand, plan, place, screen, known, discover) -> str:
    """Why this screen is looked at, or "" when the change does not touch it."""
    if plan["all"]:
        return "every place"
    if plan.get("named"):
        return "asked for"
    if place in plan["pages"]:
        return f"the {place} page imports a changed module"
    if discover:
        hits = stand.probe(plan["classes"])
        if hits:
            known[screen] = {"place": place, "why": "shows ." + ", .".join(hits[:4]) + (f" and {len(hits) - 4} more" if len(hits) > 4 else "")}
    return known.get(screen, {}).get("why", "")


def _look_screen(stand, viewport, theme, place, tab, screen, why, spec) -> dict:
    label, size = engine_label(stand), f"{stand.width}x{stand.height}"
    png = stand.page.screenshot(timeout=120000)   # the paint measure.js waits for, and the first frame
    found = stand.measure(target=44 if stand.touch else None, limit=4)
    frames = []
    if spec["frames"] in ("first", "all"):
        name, caption = _safe(f"{spec['prefix']}{stand.engine}-{theme}-{size}-{screen}"), f"{label} | {theme} | {size} @{stand.scale:g}x | {screen}"
        path = stand.out / f"{name}.png"
        path.write_bytes(caption_frame(png, caption, stand.scale))
        frames.append(str(path))
        if spec["frames"] == "all":
            frames += [str(p) for p in stand.shot_scroll(name, caption=caption)]
    return {"screen": screen, "place": place, "tab": tab["name"] if tab else None, "engine": stand.engine, "label": label,
            "theme": theme, "viewport": viewport["name"], "size": f"{size}@{stand.scale:g}", "touch": stand.touch, "why": why,
            **_facts(found), "frames": frames}


def _look_once(report, plan, engine, viewport, theme, known, deadline, spec, save):
    """One engine, viewport and theme: its screens measured into ``report``; a browser that cannot start says so."""
    combo, state = f"{engine['engine']} {theme} {viewport['name']}", report["engines"][-1]
    stand = Stand(theme=theme, width=int(viewport["width"]), height=int(viewport["height"]),
                  scale=float(viewport.get("scale") or 2), engine=engine["engine"], mutations="ok", out=spec["out"],
                  web_root=spec["web_root"], url=report["url"], executable_path=engine.get("executable_path"),
                  sandbox=engine.get("sandbox", True), fallback=False, touch=viewport.get("touch"), keep_intercepted=False)
    try:
        stand.__enter__()
    except Exception as exc:
        state.update(status="unavailable", error=_first_line(exc))
        return
    discover = bool(plan["classes"]) and not plan.get("named") and viewport["name"] not in report["discovered"]
    try:
        state.update(status="ran", label=engine_label(stand))
        stand.open()
        report["panels"].setdefault(viewport["name"], stand.panels())
        if discover:
            report["discovered"].append(viewport["name"])
        for place, tab, screen in _screens(stand, plan, known, discover, spec["rooms"], report["notes"]):
            if time.monotonic() > deadline:
                report["not_looked"].append(f"{combo}: {screen} and the screens after it, the time budget was spent")
                break
            why = _why(stand, plan, place, screen, known, discover)
            if why:
                report["screens"].append(_look_screen(stand, viewport, theme, place, tab, screen, why, spec))
                save()
    except Exception as exc:
        report["notes"].append(f"{combo}: stopped by {type(exc).__name__}: {_first_line(exc)}")
    finally:
        report["writes_answered_locally"] += len(stand.intercepted)
        report["dropped_ws_frames"] += len(stand.dropped_ws)
        stand.__exit__(None, None, None)


def look(spec: dict, report_path) -> dict:
    """One look, kept at ``report_path`` and rewritten after every screen so a stopped run keeps what it saw.

    Spec keys: url; web_root (the folder to show, else what the server serves); base (the served web/ the diff is read
    against); changed (files instead of the diff); places (names instead of the plan, ["all"] for every place);
    viewports [{name, width, height, scale, touch}]; themes; engines [{engine, executable_path?, sandbox?}]; frames
    none|first|all; frames_dir; frame_prefix; rooms; budget_sec."""
    started, report_path = time.monotonic(), Path(report_path)
    deadline = started + max(10.0, float(spec.get("budget_sec") or 480))
    web_root, served, names = spec.get("web_root") or None, spec.get("base") or None, spec.get("places") or None
    if names:
        plan = {"all": list(names) == ["all"], "pages": [] if list(names) == ["all"] else list(names), "classes": [],
                "why": ["the places asked for"], "changed": [], "unplaced": [], "named": True}
    elif web_root or served:
        plan = place_plan(web_root or served, served if web_root else None, spec.get("changed"))
    else:
        plan = {"all": True, "pages": [], "classes": [], "why": ["no web/ folder to read a diff from, so every place"],
                "changed": [], "unplaced": []}
    report = {"schema": 1, "url": spec.get("url") or live_url(), "web_root": web_root, "served_web": served, "plan": plan,
              "viewports": spec["viewports"], "themes": spec["themes"], "frames": spec.get("frames") or "none",
              "engines": [], "screens": [], "panels": {}, "discovered": [], "not_looked": [], "notes": [],
              "writes_answered_locally": 0, "dropped_ws_frames": 0, "playwright": True, "finished": False}

    def save():
        report["elapsed_sec"] = round(time.monotonic() - started, 1)
        part = report_path.with_name(report_path.name + ".part")
        part.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(part, report_path)

    try:
        import playwright.sync_api  # noqa: F401
    except ImportError as exc:
        report.update(playwright=False, finished=True, notes=[f"Playwright is not importable here: {_first_line(exc)}"])
        save()
        return report
    look_spec = {"out": Path(spec.get("frames_dir") or report_path.parent), "web_root": web_root,
                 "prefix": str(spec.get("frame_prefix") or "look-"), "frames": report["frames"],
                 "rooms": max(0, int(spec.get("rooms") or 0))}
    known: dict = {}
    if plan["all"] or plan["pages"] or plan["classes"]:
        for engine in spec["engines"]:
            report["engines"].append({"engine": engine["engine"], "status": "not started"})
            for viewport in spec["viewports"]:
                for theme in spec["themes"]:
                    if report["engines"][-1]["status"] == "unavailable":
                        break
                    if time.monotonic() > deadline:
                        report["not_looked"].append(f"{engine['engine']} {theme} {viewport['name']}: the time budget was spent first")
                        continue
                    _look_once(report, plan, engine, viewport, theme, known.setdefault(viewport["name"], {}), deadline,
                               look_spec, save)
                    save()
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
