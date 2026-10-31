"""docs/WIDGET_HOST_PROTOCOL.md is the public host side of module widgets; the builtin host
(``web/modules/widget_module.js`` page side, ``web/modules/widget_frame.js`` and its classic
``web/modules/widget_frame_bridge.js`` frame bootstrap) is
its reference implementation. Every message, attribute, policy directive and number the document
states is checked against that code, so an independent host built from the document alone keeps
the promises widget authors rely on."""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
DOC = (REPO / "docs" / "WIDGET_HOST_PROTOCOL.md").read_text(encoding="utf-8")
PAGE = (REPO / "web" / "modules" / "widget_module.js").read_text(encoding="utf-8")
FRAME = ((REPO / "web" / "modules" / "widget_frame.js").read_text(encoding="utf-8")
         + (REPO / "web" / "modules" / "widget_frame_bridge.js").read_text(encoding="utf-8"))
JOB = (REPO / "web" / "modules" / "widget_job.js").read_text(encoding="utf-8")


def _section(title: str) -> str:
    match = re.search(rf"^\*\*{re.escape(title)}\*\*\n(?P<body>.*?)(?=^\*\*|^## )", DOC, re.S | re.M)
    assert match, title
    return match.group("body")


def _js_constant(text: str, name: str) -> str:
    match = re.search(rf"export const {name} = (?:'(?P<text>[^']*)'|(?P<number>\d+));", text)
    assert match, name
    return match.group("text") if match.group("text") is not None else match.group("number")


def test_the_documented_grammar_is_exactly_the_bridge_grammar():
    in_code = set(re.findall(r"'(ouro-widget-[a-z-]+)'", PAGE + FRAME))
    in_doc = set(re.findall(r"`(ouro-widget-[a-z-]+)`", DOC))
    assert in_code and in_doc == in_code, (sorted(in_code - in_doc), sorted(in_doc - in_code))
    # Direction: what the frame posts is what the page handles, and the reverse.
    frame_posts = set(re.findall(r"post\(\{\s*type: '(ouro-widget-[a-z-]+)'", FRAME))
    frame_posts |= set(re.findall(r"type: '(ouro-widget-resize)'", FRAME))
    page_posts = set(re.findall(r"(?:post|reply)\(\{\s*type: '(ouro-widget-[a-z-]+)'", PAGE))
    page_posts |= {"ouro-widget-fetch-chunk"}  # posted through frame()
    to_page = set(re.findall(r"`(ouro-widget-[a-z-]+)`", _section("Frame to page:")))
    to_frame = set(re.findall(r"`(ouro-widget-[a-z-]+)`", _section("Page to frame:")))
    assert frame_posts == to_page, (sorted(frame_posts ^ to_page))
    # The one window message: the hello that hands the page its channel, documented beside the tables.
    assert "postMessage({ type: 'ouro-widget-hello'" in FRAME and "`ouro-widget-hello`" in _section("The handshake:")
    assert page_posts == to_frame, (sorted(page_posts ^ to_frame))
    assert re.findall(r"phase === '([a-z]+)'", FRAME) == ["headers", "data", "end", "error"]
    for phase in ("headers", "data", "end", "error"):
        assert f"`{phase}`" in _section("Page to frame:")


def test_the_frame_attributes_and_document_policy_are_the_builtin_ones():
    assert f"| `sandbox` | `{_js_constant(PAGE, 'WIDGET_FRAME_SANDBOX')}` |" in DOC
    assert f"| `allow` | `{_js_constant(PAGE, 'WIDGET_FRAME_ALLOW')}` |" in DOC
    assert "iframe.setAttribute('allowfullscreen', '')" in PAGE
    templates = re.findall(r"return \[\n(.*?)\]\.join\('; '\);", PAGE, re.S)
    policies = []
    for template in templates:
        directives = [literal[1].replace("${prefix}", "ORIGIN/api/extensions/SKILL/")
                      .replace("${assetBase}", "ASSET_BASE")
                      .replace("${passive}", "data: blob: RESOURCE_BASE")
                      for literal in re.findall(r"""^\s*(["'`])(.*?)\1,?$""", template, re.M)]
        policies.append('; '.join(directives))
    documented = re.findall(r"```\n\s*(default-src[^\n]+)\n\s*```", DOC)
    # The final policy belongs to the separate HTML living-card renderer.
    assert len(policies) == 2 and set(documented[:2]) == set(policies)
    assert "allow-same-origin" not in _js_constant(PAGE, "WIDGET_FRAME_SANDBOX")


def test_the_srcdoc_layout_and_script_escaping_are_the_builtin_ones():
    builtin = re.search(r"const srcdoc = `(?P<doc>[^`]+)`;", PAGE).group("doc")
    shaped = (builtin.replace("${csp}", "POLICY").replace("${bridge}", "BRIDGE")
              .replace("${resizeBridge}", "RESIZE").replace("${base}", "BASE").replace("${entryScript}", "ENTRY"))
    assert f"```html\n{shaped}\n```" in DOC
    assert r".replace(/<\/script/gi, '<\\/script')" in PAGE and r".replace(/<!--/g, '<\\!--')" in PAGE
    assert r"`</script` written as `<\/script`" in DOC and r"`<!--` as `<\!--`" in DOC


def test_the_documented_numbers_are_the_builtin_ones():
    assert _js_constant(PAGE, "WIDGET_FRAME_DEFAULT_HEIGHT") == "320"
    assert _js_constant(PAGE, "WIDGET_FRAME_MAX_HEIGHT") == "8192"
    assert _js_constant(PAGE, "WIDGET_FRAME_BORDER_RESERVE") == "2"
    assert "`min(ceiling, max(320, height + 2))`" in DOC and "default 8192" in DOC
    assert _js_constant(JOB, "WIDGET_DISPOSE_ACK_TIMEOUT_MS") == "1000"
    assert "after 1000 milliseconds" in DOC
    assert _js_constant(JOB, "WIDGET_REQUEST_TIMEOUT_MS") == "25000"
    assert "after 25 seconds" in DOC
    assert "faultCount >= 10" in FRAME and "at most 10 faults" in DOC
    assert _js_constant(PAGE, "WIDGET_FAULT_REPORTS_PER_MOUNT") == "10" and "at most 10 per mount" in DOC


def test_the_in_frame_api_is_the_one_authors_are_promised():
    api = re.search(r"window\.OuroborosWidget = \{(?P<keys>[^}]+)\};", FRAME).group("keys")
    keys = [part.split(":")[0].strip() for part in api.split(",")]
    assert "`window.OuroborosWidget = {" + ", ".join(keys) + "}`" in DOC
    for hook in ("window.__ouroWidgetOnDispose = onDispose", "window.fetch = request", "window.open = (url)"):
        assert hook in FRAME
    authoring = (REPO / "docs" / "CREATING_SKILLS.md").read_text(encoding="utf-8")
    assert "WIDGET_HOST_PROTOCOL.md" in authoring, "the author guide links the host side"


def test_the_fetch_relay_rules_are_the_builtin_ones():
    assert "redirect: 'error'" in PAGE and '`redirect: "error"`' in DOC
    assert "credentials: asset ? 'omit' : 'same-origin'" in PAGE and '`credentials: "same-origin"`' in DOC
    assert '`credentials: "omit"`' in DOC
    for text in ("module widget fetch outside extension route prefix", "widget request timed out"):
        assert text in PAGE and f"`{text}`" in DOC
    assert "[204, 205, 304]" in FRAME and "204, 205 or 304" in DOC


def test_downloads_stream_routes_and_save_declared_files_as_bytes():
    assets = (REPO / "web" / "modules" / "widget_assets.js").read_text(encoding="utf-8")
    assert "downloadBlobViaHostBridge(await readWidgetAsset(" in PAGE and "{ streaming: true }" in PAGE
    assert "credentials: 'omit'" in assets.split("export async function readWidgetAsset", 1)[1]
    assert 'streams a skill-route URL and reads an asset URL with `credentials: "omit"`, saving its finite' in DOC
