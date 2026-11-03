"""Generated inventory of FIXED control text in the web UI (docs/DESIGN.md "References and actions").

Design drift in this UI happens at the level of owner INTENTS, not tokens: one intent
("take me to that Project") was once drawn six ways, and every copy truthfully "reused a
shared primitive". No per-diff check can see that, because the siblings of a new control
are not in the diff. This inventory puts them there: it is SORTED BY TEXT, so a new
`Retry` lands in the same hunk as the `Retry` rows that already exist, with their classes
and files beside it.

It is review evidence, never a gate on wording: the only red is staleness
(``tests/test_generated_inventories.py``). Extraction is a plain scan of source text for
countable facts — a ``<button>`` or ``<summary>`` literal whose text is fixed (including ``tr`` fallbacks), a created button with a
literal ``textContent``, the fixed accessible name of an icon-only button, a module's own
button factory (an arrow function whose template is one ``<button>`` with a parameter as
its text) called with a literal label, and every ``ouro:*`` CustomEvent with the modules
that raise it. Interpolated text is counted, not listed; it never judges what a control means.
"""
from __future__ import annotations

import html
import pathlib
import re
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
UI_CONTROLS_OUT = REPO_ROOT / "docs" / "inventories" / "UI_CONTROL_TEXT_INVENTORY.md"

# A `>` inside a `${…}` expression of an attribute (`${rows.length >= MAX ? 'disabled' : ''}`) is not the tag's end.
_BUTTON = re.compile(r"<(?:button|summary)\b((?:\$\{[^}]*\}|[^>])*)>(.*?)</(?:button|summary)>", re.S)
_ATTR = {name: re.compile(name + r'="([^"]*)"') for name in ("class", "aria-label", "title")}
# Literal tr fallbacks are fixed host words even though the language is selected at runtime.
_TRANSLATED = re.compile(r"\$\{(?:escapeHtml\()?tr\((['\"])[^'\"]+\1,\s*(['\"])([^'\"]+)\2\)\)?\}")
_TAG = re.compile(r"<[^>]+>", re.S)
_CREATED = re.compile(r"(?:const|let)\s+(\w+)\s*=\s*document\.createElement\('button'\)")
_EVENT = re.compile(r"new CustomEvent\(\s*['\"`](ouro:[\w:.-]+)['\"`]")
_LITERAL = r"\s*=\s*(['\"])((?:(?!\1).)+)\1"
_WINDOW = 900  # characters after `createElement('button')` in which its own text and class are set
# A module's own button factory: `const button = (label, data, tone = 'btn-default') => `<button … class="btn ${tone}"
# ${data}>${text(label)}</button>``. Its calls with a literal label are that module's controls; the template is none.
_FACTORY = re.compile(r"(?:const|let)\s+(\w+)\s*=\s*\(([^()]*)\)\s*=>\s*`<button\b")
_PARAM = re.compile(r"^\s*(\w+)\s*(?:=\s*(['\"])((?:(?!\2).)*)\2)?\s*$")
_TEXT_PARAM = re.compile(r"^\$\{(?:\w+\()?(\w+)\)?\}$")  # `${label}` or `${text(label)}`
_STRING = re.compile(r"^(['\"])((?:(?!\1).)*)\1$")


def _sources() -> list[pathlib.Path]:
    web = REPO_ROOT / "web"
    shell = [path for path in sorted(web.glob("*.js")) if not path.name.endswith((".min.js", ".config.js"))]
    return [*sorted((web / "modules").glob("*.js")), *shell, web / "index.html"]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", text))).strip()


def _classes(value: str) -> str:
    return re.sub(r"\$\{[^}]*\}", "…", value).strip() or "(no class)"


def _arguments(text: str, start: int) -> list[str]:
    """The top-level arguments of the call whose opening parenthesis ends at START."""
    depth, quote, begin, args = 0, "", start, []
    index = start
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = ""
        elif char in "'\"`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                return [*args, text[begin:index]]
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[begin:index])
            begin = index + 1
        index += 1
    return args


def _factories(text: str) -> list[tuple[str, list[tuple[str, str | None]], int, str, tuple[int, int]]]:
    """Each button factory of one source: name, parameters with their literal defaults, the index
    of the label parameter, the class template and the span of its `<button>` template."""
    found = []
    for match in _FACTORY.finditer(text):
        tag = _BUTTON.match(text, match.end() - len("<button"))
        params = [_PARAM.match(part) for part in match.group(2).split(",")] if match.group(2).strip() else []
        text_param = _TEXT_PARAM.match(tag.group(2).strip()) if tag else None
        names = [param.group(1) for param in params if param]
        if tag is None or not params or not all(params) or text_param is None or text_param.group(1) not in names:
            continue
        defaults = [(param.group(1), param.group(3)) for param in params]
        classes = (_ATTR["class"].search(tag.group(1)) or [None, ""])[1]
        found.append((match.group(1), defaults, names.index(text_param.group(1)), classes, tag.span()))
    return found


def scan_controls(text: str) -> tuple[list[tuple[str, str]], int]:
    """``([(label, classes)], interpolated)`` for one source text."""
    rows: list[tuple[str, str]] = []
    interpolated = 0
    factories = _factories(text)
    for match in _BUTTON.finditer(text):
        if any(start <= match.start() < end for *_rest, (start, end) in factories):
            continue  # a factory's template: its calls are the controls
        attrs, inner = match.group(1), match.group(2)
        found = {name: (rx.search(attrs) or [None, ""])[1] for name, rx in _ATTR.items()}
        label = _clean(_TRANSLATED.sub(lambda match: match[3], inner))
        if not label:  # an icon-only control is named by its fixed accessible name
            spoken = found["aria-label"] or found["title"]
            label = f"[icon] {spoken}" if spoken else ""
        if not label or "${" in label:
            interpolated += 1
            continue
        rows.append((label, _classes(found["class"])))
    for match in _CREATED.finditer(text):
        var = re.escape(match.group(1))
        window = text[match.end(): match.end() + _WINDOW]
        label = re.search(rf"\b{var}\.textContent{_LITERAL}", window)
        classes = re.search(rf"\b{var}\.className{_LITERAL}", window)
        if not label:
            interpolated += 1
            continue
        rows.append((_clean(label.group(2)), _classes(classes.group(2) if classes else "")))
    for name, params, label_at, template, _span in factories:
        for call in re.finditer(rf"(?<![\w.]){re.escape(name)}\(", text):
            args = [_STRING.match(arg.strip()) for arg in _arguments(text, call.end())]
            label = args[label_at] if label_at < len(args) else None
            if label is None:
                interpolated += 1
                continue
            classes = template
            for index, (param, default) in enumerate(params):
                bound = args[index].group(2) if index < len(args) and args[index] else default if index >= len(args) else None
                if bound is not None:
                    classes = classes.replace("${" + param + "}", bound)
            rows.append((_clean(label.group(2)), _classes(classes)))
    return rows, interpolated


def _sort_key(row: tuple[str, str, str]) -> tuple[str, str, str, str]:
    label, classes, path = row
    return (re.sub(r"^[^\w]+", "", label.replace("[icon] ", "")).casefold(), label, classes, path)


def _cell(value: str) -> str:
    return value.replace("|", "\\|")


def build_ui_control_inventory() -> tuple[str, list[str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    events: dict[str, Counter[str]] = {}
    interpolated = 0
    for path in _sources():
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        rel = path.relative_to(REPO_ROOT / "web").as_posix()
        rows, skipped = scan_controls(text)
        interpolated += skipped
        for label, classes in rows:
            counts[(label, classes, rel)] += 1
        for name in _EVENT.findall(text):
            events.setdefault(name, Counter())[rel] += 1
    listed = sum(counts.values())
    lines = [
        "# UI control text inventory",
        "",
        "Generated by `python scripts/regenerate_inventories.py` — do not edit. Why it exists and how to read",
        "it: `docs/DESIGN.md` \"References and actions\". Rows are sorted by text so that the controls which say",
        "the same thing sit together: when your diff adds a row, read its neighbours.",
        "",
        f"{listed} controls with fixed text in {len(counts)} rows, {len({key[0] for key in counts})} distinct texts;",
        f"{interpolated} more controls build their text at run time and are not listed.",
        "",
        "## Fixed control text",
        "",
        "| text | classes | file | n |",
        "|---|---|---|---|",
    ]
    for label, classes, rel in sorted(counts, key=_sort_key):
        lines.append(f"| {_cell(label)} | `{_cell(classes)}` | {rel} | {counts[(label, classes, rel)]} |")
    lines += [
        "",
        "## `ouro:*` events and the modules that raise them",
        "",
        "An owner intent that has a door is raised by exactly one module (`ouro:open-project` →",
        "`modules/project_reference.js`).",
        "",
        "| event | raised by |",
        "|---|---|",
    ]
    for name in sorted(events):
        raisers = ", ".join(f"{rel} ×{n}" if n > 1 else rel for rel, n in sorted(events[name].items()))
        lines.append(f"| `{name}` | {raisers} |")
    return "\n".join(lines) + "\n", []
