"""A stored 0 survives the Settings form round trip.

The load path applied ``fallback && !s[key] ? fallback : s[key]`` to every
INPUT_FIELDS row, so a saved ``0`` (``OUROBOROS_CONSCIOUSNESS_DAILY_USD``, the
documented "may not spend" value) rendered as its fallback ``20`` and the next
save of any tab wrote ``20`` back. Only a missing value may take the fallback.

Source pin (the pattern of the other *_static tests) plus, when ``node`` is on
PATH, an executed check of the helper itself.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest

SETTINGS_JS = pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "settings.js"


def _source() -> str:
    return SETTINGS_JS.read_text(encoding="utf-8")


def test_input_fields_load_through_the_helper():
    src = _source()
    assert "applyInputValue(id, storedOrFallback(s[key], fallback))" in src
    assert "fallback && !s[key]" not in src


def _helper_source() -> str:
    match = re.search(r"export function storedOrFallback\(value, fallback\) \{.*?\n\}", _source(), re.S)
    assert match, "storedOrFallback helper is missing"
    return match.group(0).replace("export ", "", 1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_helper_keeps_zero_and_fills_only_absence():
    cases = [
        [0, "20", 0],
        ["0", "20", "0"],
        [0.0, "20", 0],
        [None, "20", "20"],
        ["", "20", "20"],
        ["5", "20", "5"],
        ["", "", ""],
    ]
    script = _helper_source() + "\n" + (
        "const cases = " + json.dumps(cases) + ";\n"
        "const out = cases.map(([v, f]) => storedOrFallback(v, f));\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [expected for _, _, expected in cases]
