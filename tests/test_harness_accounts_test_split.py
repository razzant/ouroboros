"""Structural contract for the harness-accounts Node test split (v7 stream W).

`web/tests/harness_accounts.test.js` was 1882 lines. It is now four sibling
`*.test.js` files plus one non-test helper module. `npm test` runs
`node --test tests/*.test.js`, so the only thing that can silently go wrong is a
file dropping out of that glob or a test title disappearing with it — which is
exactly what these assertions pin, from Python, without a Node runtime.
"""

from __future__ import annotations

import json
import pathlib
import re


REPO = pathlib.Path(__file__).parents[1]
WEB = REPO / "web"
TESTS = WEB / "tests"
FAMILY = (
    "harness_accounts.test.js",
    "harness_accounts_cards.test.js",
    "harness_accounts_custody.test.js",
    "harness_accounts_panel.test.js",
)
HELPERS = "harness_accounts_helpers.js"
# The count is the pre-split total: 59 `test(...)` registrations in the one file
# at the split commit. It is pinned, not derived, because a title that silently
# stops being registered is precisely the regression a split can cause.
EXPECTED_TESTS = 59
# Each fixture is declared exactly once in the family; where it is used by more
# than one file it is owned by the helper module and imported.
_HELPER_OWNERS = {
    "fakeResponse": HELPERS,
    "CREDENTIAL_PROFILES_RESPONSE": "harness_accounts.test.js",
    "cardWithUrl": "harness_accounts_cards.test.js",
    "fakeCodeInput": "harness_accounts_cards.test.js",
    "fakeCardHost": "harness_accounts_cards.test.js",
    "storeWithReads": "harness_accounts_custody.test.js",
    "fakeElement": "harness_accounts_panel.test.js",
    "mountSection": "harness_accounts_panel.test.js",
    "captureCardControls": "harness_accounts_panel.test.js",
    "WAKE_STILL_DOWN": "harness_accounts_panel.test.js",
    "WAKE_UP": "harness_accounts_panel.test.js",
}


def _sources() -> dict[str, str]:
    return {name: (TESTS / name).read_text(encoding="utf-8") for name in FAMILY}


def test_every_split_file_is_discovered_by_the_npm_test_glob():
    """`node --test tests/*.test.js` is the discovery contract; a name outside it
    would take its tests out of the suite without failing anything."""
    command = json.loads((WEB / "package.json").read_text(encoding="utf-8"))["scripts"]["test"]
    assert command == "node --test tests/*.test.js"
    for name in FAMILY:
        assert (TESTS / name).is_file(), name
        assert name.endswith(".test.js"), name
        assert (TESTS / name) in set(TESTS.glob("*.test.js")), name
    # The shared fixtures must NOT be discovered as a test file: it registers no
    # tests, and node --test would report an empty file rather than nothing.
    assert (TESTS / HELPERS).is_file()
    assert not HELPERS.endswith(".test.js")


def test_the_split_family_still_registers_every_test_exactly_once():
    titles: list[str] = []
    for source in _sources().values():
        titles += re.findall(r"^test\((['\"])(.*?)\1", source, re.M)
    names = [title for _quote, title in titles]
    assert len(names) == EXPECTED_TESTS, len(names)
    assert len(set(names)) == EXPECTED_TESTS, "a test title is registered twice"


def test_each_moved_fixture_has_exactly_one_owner_in_the_family():
    sources = _sources()
    sources[HELPERS] = (TESTS / HELPERS).read_text(encoding="utf-8")
    for fixture, owner in _HELPER_OWNERS.items():
        declaration = re.compile(rf"^(?:const|function)\s+{re.escape(fixture)}\b", re.M)
        owners = [name for name, text in sources.items() if declaration.search(text)]
        assert owners == [owner], (fixture, owners)
        for name, text in sources.items():
            if name == owner or not re.search(rf"\b{re.escape(fixture)}\b", text):
                continue
            assert f"from './{HELPERS}'" in text, (fixture, name)


def test_the_pre_split_module_surface_is_preserved_by_a_facade_reexport():
    """`fakeResponse` was declared in harness_accounts.test.js. It is owned by the
    helper module now, and the original file re-exports it, which is the facade
    the v7 migration ledger binds."""
    source = (TESTS / FAMILY[0]).read_text(encoding="utf-8")
    assert f"import {{ fakeResponse }} from './{HELPERS}';" in source
    assert "export { fakeResponse };" in source


def test_split_files_have_meaningful_size_headroom():
    counts = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in [*(TESTS / name for name in FAMILY), TESTS / HELPERS]
    }
    assert all(count <= 1000 for count in counts.values()), counts
