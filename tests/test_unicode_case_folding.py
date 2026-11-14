"""The chat search's case folding (``web/modules/unicode_case_folding.js``) is Unicode 17.0.0's own table, complete.

The module is generated from the vendored ``scripts/unicode/CaseFolding-17.0.0.txt`` by ``scripts/generate_case_folding.py``:
the source must still be the pinned bytes, the committed module exactly what they generate, and every C and F line of
the source in it while no S or T line is. Python's ``str.casefold`` is the same full default folding at its own (older)
Unicode version, so for every code point that version assigns the module folds exactly as Python does; the only
differences are code points Python's version leaves unassigned (letters Unicode added since).
"""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import re
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _generator():
    spec = importlib.util.spec_from_file_location("generate_case_folding", ROOT / "scripts" / "generate_case_folding.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_table() -> dict[int, str]:
    text = (ROOT / "web" / "modules" / "unicode_case_folding.js").read_text(encoding="utf-8")
    entries = ";".join(re.findall(r"^    '([^']*)',$", text, flags=re.M)).split(";")
    table = {}
    for entry in entries:
        code, folded = entry.split(":")
        table[int(code, 16)] = "".join(chr(int(part, 16)) for part in folded.split())
    assert f"export const CASE_FOLDING_ENTRIES = {len(entries)};" in text
    return table


def test_the_source_is_the_pinned_unicode_table_and_the_module_is_what_it_generates():
    generator = _generator()
    data = generator.SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == generator.SOURCE_SHA256
    assert data.startswith(b"# CaseFolding-17.0.0.txt")
    assert (ROOT / "scripts" / "unicode" / "LICENSE.txt").read_text(encoding="utf-8").startswith("UNICODE LICENSE V3")
    assert generator.main(["generate_case_folding.py", "--check"]) == 0


def test_every_c_and_f_line_is_in_the_module_and_no_s_or_t_line_is():
    table = _module_table()
    statuses: dict[str, dict[int, str]] = {"C": {}, "F": {}, "S": {}, "T": {}}
    for line in (ROOT / "scripts" / "unicode" / "CaseFolding-17.0.0.txt").read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split("#", 1)[0].split(";")]
        if len(fields) >= 3:
            statuses[fields[1]][int(fields[0], 16)] = "".join(chr(int(part, 16)) for part in fields[2].split())
    assert table == {**statuses["C"], **statuses["F"]} and len(table) == 1585
    assert not set(statuses["C"]) & set(statuses["F"]), "one default folding per code point"
    assert table[0x49] == "i" and table[0x130] == "i̇", "C and F, never the Turkic T"
    assert table[0x1E9E] == "ss", "F, never its simple S alternative"


def test_the_module_folds_every_code_point_python_assigns_exactly_as_python_does():
    table = _module_table()
    differences = []
    for code in range(0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        char = chr(code)
        if char.casefold() != table.get(code, char):
            differences.append(code)
    # Python may run an older Unicode (its unassigned code points differ) or Unicode 17 itself (none differ).
    assert all(unicodedata.category(chr(code)) == "Cn" for code in differences), [
        (hex(code), unicodedata.name(chr(code), "?")) for code in differences if unicodedata.category(chr(code)) != "Cn"]
    assert len(differences) < 100, f"{len(differences)} code points newer than Python's Unicode {unicodedata.unidata_version}"
