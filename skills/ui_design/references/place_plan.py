"""Which places of Ouroboros's own interface a change of its web/ files touches, read from the files themselves.

A place is one [data-nav-page] of the shell (web/modules/pages.js). The shell and its tokens touch every place. A module
touches the pages whose own module imports it, and the screens that show a class only it draws. A stylesheet touches
the screens that show a class of a rule it changed (a rule with the conditions around it, so a moved breakpoint
changes it), and every place when the rules it keeps change their order. Classes are looked for on the screens
themselves, so a part an app-level module draws (a Dashboard panel) is found where it stands. No browser: the ui_design
skill's live_stand.py (beside this file) walks the plan this answers.
"""
from __future__ import annotations

import os
import posixpath
import re
from collections import Counter
from pathlib import Path

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
    """A stylesheet's leaf rules in order as (conditions, selector, declarations), whitespace-normalised; the
    conditions are the preludes of the group rules around it (@media, @container, @supports), outermost first."""
    text = _CSS_COMMENT.sub("", text)
    rules, opened, start = [], [], 0
    for i, ch in enumerate(text):
        if ch == "{":
            opened.append((" ".join(text[start:i].split()), i + 1))
            start = i + 1
        elif ch in "};":
            if ch == "}" and opened:
                prelude, at = opened.pop()
                if "{" not in text[at:i]:
                    rules.append((tuple(outer for outer, _ in opened), prelude, " ".join(text[at:i].split())))
            start = i + 1
    return rules


def _in_order(rules: list, kept: Counter) -> list:
    """``rules`` in their order, keeping only as many of each as ``kept`` holds."""
    left, out = Counter(kept), []
    for rule in rules:
        if left[rule] > 0:
            left[rule] -= 1
            out.append(rule)
    return out


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
    """(classes of the rules that differ between two stylesheets, "") or (None, why) when they reach every place.

    A rule is compared with the conditions around it, so a changed breakpoint changes its rule. The rules both sheets
    keep must stand in the same order: the later of two equal rules wins, so a reordering reaches every place."""
    old_rules, new_rules = _css_rules(old), _css_rules(new)
    before, after = Counter(old_rules), Counter(new_rules)
    if _in_order(old_rules, before & after) != _in_order(new_rules, before & after):
        return None, "the order of its rules changed"
    named = set()
    for _conditions, selector, body in ((before - after) + (after - before)).elements():
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
        plan["why"].append("no changed file draws a place")
    return plan
