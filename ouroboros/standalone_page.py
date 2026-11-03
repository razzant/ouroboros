"""A page that answers before any served asset may load still wears the interface's appearance.

Two kinds of page live outside the interface: the builtin network login form, which a
browser sees before the gate lets anything else through, and the desktop windows the
launcher shows before its server exists. Each inlines the builtin interface's own
appearance owner (``web/theme.js``) and its tokens and recipes (``web/ui.css``), so it
paints the saved Light / Dark / System choice where its document can read one (the
login form, which names the installation's appearance scope as the interface does) and
the device's appearance where it cannot (a pre-server window has no origin that
remembers a choice). No second store, resolver or palette exists. A file that cannot be
read or inlined leaves the page readable in the system's own colours, never unusable.
"""

from __future__ import annotations

import html
import logging
import pathlib

log = logging.getLogger(__name__)


def _inline(path: pathlib.Path, tag: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        log.warning("%s was not inlined; the page keeps the system's colours: %s", path, exc)
        return ""
    if f"</{tag}" in text.lower():  # would end the element early
        log.warning("%s cannot be inlined into <%s>; the page keeps the system's colours", path, tag)
        return ""
    return f"<{tag}>{text}</{tag}>"


def themed_page(web_dir: pathlib.Path, title: str, body: str, css: str = "", head: str = "") -> str:
    """One self-contained document: ``theme.js`` before any style, then ``ui.css`` and the page's own ``css``.

    ``css`` maps the page's elements onto the tokens; it gives each a system-colour fallback
    (``var(--bg-primary, Canvas)``) for a tree whose files were not inlined. ``head`` is markup
    ``theme.js`` reads (the appearance scope a served page names)."""
    web_dir = pathlib.Path(web_dir)
    return (
        '<!DOCTYPE html>\n<html lang="en" class="ouro-ui">\n<head>\n<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{html.escape(title)}</title>\n{head}"
        f"{_inline(web_dir / 'theme.js', 'script')}\n{_inline(web_dir / 'ui.css', 'style')}\n"
        f"<style>{css}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
