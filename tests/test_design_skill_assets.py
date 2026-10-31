"""The browser-side files the design skills ship: what they promise without a browser.

`ui_design/assets/compare.html` is the one-file comparison page, and `ui_design/references`
holds `measure.js` (observations about a rendered page) and `freeze.js` (a rendered page as one
self-contained document). These checks pin the contract the skills' prose relies on: one file that
fetches nothing, a data block that parses, function expressions a browser tool can evaluate, and
measurement that reports evidence rather than verdicts. Behaviour in real engines is covered by
`tests/test_design_skill_assets_browser.py`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMPARE = REPO / "skills" / "ui_design" / "assets" / "compare.html"
MEASURE = REPO / "skills" / "ui_design" / "references" / "measure.js"
FREEZE = REPO / "skills" / "ui_design" / "references" / "freeze.js"
ASSETS = (COMPARE, MEASURE, FREEZE)
DATA_BLOCK = re.compile(r'<script type="application/json" id="compare-data">(.*?)</script>', re.S)


def compare_data(html: str) -> dict:
    """The template's JSON block, parsed exactly as the page parses it."""
    match = DATA_BLOCK.search(html)
    assert match, "compare.html must carry one JSON data block"
    return json.loads(match.group(1))


def fill_compare(data: dict) -> str:
    """The template with its data block replaced, as an author fills it (``</`` cannot close the block)."""
    html = COMPARE.read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return DATA_BLOCK.sub(lambda _m: f'<script type="application/json" id="compare-data">{blob}</script>', html, count=1)


def _function_body(path: Path) -> str:
    """The source after its header comment: what must be one parenthesized function expression."""
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("//")):
        lines.pop(0)
    return "\n".join(lines).strip()


def test_files_exist():
    for path in ASSETS:
        assert path.is_file(), path


def test_compare_page_is_one_file_that_fetches_nothing():
    html = COMPARE.read_text(encoding="utf-8")
    assert not re.search(r"https?://", html), "no external URL, not even in a comment"
    for attribute in re.findall(r'\b(?:src|href|action)\s*=\s*["\']?([^"\'\s>]*)', html):
        assert attribute == "" or attribute.startswith(("data:", "#", "about:")), attribute
    for loader in ("fetch(", "XMLHttpRequest", "import(", "<link", "@import", "localStorage", "sendBeacon"):
        assert loader not in html, loader


def test_compare_data_block_parses_and_names_the_contract():
    data = compare_data(COMPARE.read_text(encoding="utf-8"))
    assert {"title", "lang", "viewport", "variants", "scenes", "frames", "questions"} <= set(data)
    assert {"width", "height"} <= set(data["viewport"])
    variant_ids = [variant["id"] for variant in data["variants"]]
    assert all({"id", "label", "note"} <= set(variant) for variant in data["variants"])
    assert sum(bool(variant.get("recommended")) for variant in data["variants"]) == 1
    for scene in data["scenes"]:
        for variant in variant_ids:
            assert f"{variant}/{scene['id']}" in data["frames"]
    for question in data["questions"]:
        assert question["text"] and all({"key", "text"} <= set(option) for option in question["options"])


def test_filled_page_keeps_a_frame_that_contains_script_close_tags():
    data = compare_data(COMPARE.read_text(encoding="utf-8"))
    data["frames"]["a/main"] = "<!doctype html><p>frame</p><script>let x = 1;</script>"
    assert compare_data(fill_compare(data))["frames"]["a/main"].endswith("</script>")


def test_compare_page_chrome_rules():
    html = COMPARE.read_text(encoding="utf-8")
    assert ":focus-visible" in html, "keyboard focus must be visible"
    assert "prefers-color-scheme" in html
    notes = re.search(r"\.notes p \{([^}]*)\}", html)
    assert notes and "ellipsis" not in notes.group(1) and "nowrap" not in notes.group(1), "captions wrap"
    assert "text-overflow" not in html
    assert "None of these" in html, "every question offers a way to reject all variants"


@pytest.mark.parametrize("path", [MEASURE, FREEZE], ids=["measure", "freeze"])
def test_scripts_are_single_parenthesized_function_expressions(path):
    body = _function_body(path)
    assert body.startswith("(") and body.endswith(")"), path.name
    assert re.match(r"\((?:async\s*)?\(\w*\)\s*=>\s*\{", body), "one arrow function, evaluated by a browser tool"


@pytest.mark.parametrize("path", [MEASURE, FREEZE], ids=["measure", "freeze"])
def test_scripts_evaluate_to_functions_in_node(path):
    node = shutil.which("node")
    if not node:
        if os.environ.get("OUROBOROS_PREFLIGHT_REQUIRE_NODE"):
            pytest.fail("Node is required to evaluate the browser snippets")
        pytest.skip("Node is not installed")
    from tests.group_run import run_in_group

    probe = "const f = eval(require('fs').readFileSync(process.argv[1], 'utf8')); console.log(typeof f, f.constructor.name)"
    result = run_in_group([node, "-e", probe, str(path)], timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[0] == "function"


def test_measure_reports_observations_never_verdicts():
    source = MEASURE.read_text(encoding="utf-8")
    header = source.split("((opts) =>", 1)[0]
    assert "never a gate" in header
    for mutation in (".setAttribute(", ".removeAttribute(", ".appendChild(", ".append(", ".remove(",
                     ".replaceWith(", "innerHTML", ".style.", "classList.add", "classList.toggle",
                     ".scrollTop =", ".scrollTop +="):
        assert mutation not in source, f"measure.js must not change the page: {mutation}"
    kinds = set(re.findall(r"add\('([a-z-]+)'", source))
    documented = set(re.findall(r"\b([a-z]+(?:-[a-z]+)+)\b", header))
    assert kinds <= documented, f"every reported kind is listed in the header: {kinds - documented}"


def test_freeze_strips_scripts_and_lists_gaps():
    source = FREEZE.read_text(encoding="utf-8")
    assert "script" in source and "missing" in source and "data-freeze-scroll" in source
    assert "data-freeze-scroll" in COMPARE.read_text(encoding="utf-8"), "the template restores frozen scroll offsets"
