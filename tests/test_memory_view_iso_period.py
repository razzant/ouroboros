"""ISO display compatibility: one formatter (``memory_inventory.minute``) for the view, memory_read(node_id) and
the fold prompt, and a real 3.12 publication replayed by 3.10 through all three.

Set OURO_TEST_TRANSFER_WRITER_PYTHON to a Python 3.12 executable with this repo's
dependencies for the cross-version test. The child only writes synthetic memory
under tmp_path; it never handcrafts journal transactions or bypasses validation.
``_ORACLE`` was recorded from native CPython 3.12.13 ``datetime.fromisoformat`` (stdlib only, its C parser);
on 3.12 ``test_port_matches_native_312_on_a_generated_corpus`` derives the same answers from the native parser.
"""
from __future__ import annotations

import datetime
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import chat_chain, memory_fallback, memory_inventory as mi, memory_view as mv
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tools.chronicle import _memory_read
from tests import _memory_inventory_shared as shared
from tests.test_chronicle_tools import LIMIT, _parse_window
from tests.test_chronicle_transfer import MOVED, STATED, _ctx, _moved, _publish, _view
from tests.test_memory_reader_bounds import whole


@pytest.mark.parametrize("value, expected", [
    ("2026-07-01", "2026-07-01 00:00"),
    ("2026-07-01 00:30:12", "2026-07-01 00:30"),
    ("2026-07-01T00:30:12Z", "2026-07-01 00:30"),
    ("2026-07-01T00:30:12.123456+03:30", "2026-06-30 21:00"),
    ("2026-07-01T00:30:12-03:30", "2026-07-01 04:00"),
    # Native parsing remains first; the fallback must keep these offsets too.
    ("2026-07-01T00:00:00+00:00:30", "2026-06-30 23:59"),
    ("2026-07-01T00:00:00+00:00:30.500000", "2026-06-30 23:59"),
    ("20260701T003012Z", "2026-07-01 00:30"),
    ("2026-W27-3T00:30:12+03:30", "2026-06-30 21:00"),
    ("2026W273T003012-0330", "2026-07-01 04:00"),
    ("2026-W27", "2026-06-29 00:00"),
    ("2026W273", "2026-07-01 00:00"),
])
def test_minute_formats_recognized_iso_dates_in_utc(value, expected):
    assert mv._minute(value) == expected


@pytest.mark.parametrize("digits", [1, 2, 3, 4, 6, 7, 8, 100_000, 100_013])
@pytest.mark.parametrize("prefix, suffix, expected", [
    ("2026-07-01T00:00:00.", "+03:30", "2026-06-30 20:30"),
    ("20260701T000000,", "-0330", "2026-07-01 03:30"),
    ("2026-W27-3T00:00:00.", "Z", "2026-07-01 00:00"),
    ("2026-07-01T00:00:00.", "+00:00:30", "2026-06-30 23:59"),
    ("2026-07-01T00:00:00.", "+00:00:30.5", "2026-06-30 23:59"),
    ("20260701T000000,", "+000030.5", "2026-06-30 23:59"),
    ("2026-W27-3T00:00:00.", "-00:00:59.95", "2026-07-01 00:01"),
])
def test_fraction_precision_does_not_become_display_length(digits, prefix, suffix, expected):
    assert mv._minute(prefix + "1" * digits + suffix) == expected


@pytest.mark.parametrize("digits", [1, 4, 7, 100_013])
def test_offset_fraction_uses_microseconds_without_rounding(digits):
    fraction = "5" * digits
    assert mv._minute("2026-07-01T00:00:00-00:00:59." + fraction) == "2026-07-01 00:00"
    # Padding or truncation in the time and the offset must agree at a minute boundary.
    assert mv._minute("2026-07-01T00:00:30." + fraction + "+00:00:30.5") == "2026-07-01 00:00"
    assert mv._minute("2026-07-01T00:00:30.5+00:00:30." + fraction) == (
        "2026-07-01 00:00" if digits == 1 else "2026-06-30 23:59")


@pytest.mark.parametrize("value", [
    "?", "before summer", "2026", "2026-07", "202607", "2026-02-30",
    "2026-W54-1", "2026-W27-8", "2026-07-01T25:00:00",
    "2026-07-01T00:00:00.1234567garbage", "2026-07-01T00:00:00+25:00",
    "2026-07-01T00:00:00." + "1" * 100_013 + "garbage",
    "2026-07-01T00:00:00." + "1" * 100_013 + "+00:00:30.5garbage",
    "2026-07-01T00:00:00." + "1" * 100_013 + "+00:00:30.5.6",
    "2026-07-01T00:00:00." + "1" * 100_013 + "+00:0030.5",
    "2026-07-01T00:00:00." + "1" * 100_013 + "+0000:30.5",
    "2026-07-01T00:00:00." + "1" * 100_013 + "+25:00:30.5",
    "2026-W27-8T00:00:00.1234567", "2026-0701T00:00:00.1234567",
    "2026-07-01T00:00:00.", "2026-07-01T00:00:00.1234567+00:00:30.",
# The test id names the long run: pytest puts the node id in PYTEST_CURRENT_TEST, longer than Windows admits.
], ids=lambda value: value.replace("1" * 100_013, "1x100013"))
def test_unrecognized_or_incomplete_dates_are_not_invented_or_clipped(value):
    assert mv._minute(value) == value


@pytest.mark.parametrize("value", ["2026-08-10", "20260810", "2026-W27", "2026-W27-3", "2026W27", "2026W273"])
def test_written_date_without_a_time_keeps_its_source_precision(value):
    record = {"metadata": {"transfer": {"operator": None, "tool": None, "task_id": None}, "legacy_written_at": value}}
    assert mi.transfer_words(record).endswith("; written " + value)
    assert record["metadata"]["legacy_written_at"] == value


def test_every_reader_prints_moments_with_the_one_formatter():
    assert (mv._minute, mv._period) == (mi.minute, mi.period_text)
    assert mi.period_text({"start": "2026-W27-120000", "end": None, "incomplete": True}) == (
        "2026-06-29 12:00 → ? (incomplete)")


# (text, minute in UTC as native 3.12.13 reads it; None: refused; OVERFLOW: leaves the calendar in UTC)
_ORACLE = [
    ("2026-W27-120000", "2026-06-29 12:00"),
    ("2026-W27-1200", "2026-06-29 12:00"),
    ("2026-W27-12", "2026-06-29 12:00"),
    ("2026-W27-31200", None),
    ("2026-W27-3", "2026-07-01 00:00"),
    ("2026-W27-3T12", "2026-07-01 12:00"),
    ("2026-W27T12", "2026-06-29 12:00"),
    ("2026-W27 12:30", "2026-06-29 12:30"),
    ("2026W27120000", "2026-06-29 00:00"),
    ("2026W271200", "2026-06-29 00:00"),
    ("2026W2712", None),
    ("2026W27T1200", "2026-06-29 12:00"),
    ("2026W273T12", "2026-07-01 12:00"),
    ("2026W2731200", "2026-06-29 12:00"),
    ("2026W27312000", "2026-07-01 20:00"),
    ("2026-07-01T00.5", "2026-07-01 00:00"),
    ("2026-07-01T00,5", "2026-07-01 00:00"),
    ("2026-07-01T12.5", "2026-07-01 12:00"),
    ("2026-07-01T1230.5", "2026-07-01 12:30"),
    ("2026-07-01T12:30.5", "2026-07-01 12:30"),
    ("2026-07-01T12:30:15.5", "2026-07-01 12:30"),
    ("2026-07-01T00.5+03:30", "2026-06-30 20:30"),
    ("2026-07-01T00.1234567+03:30", "2026-06-30 20:30"),
    ("2026-07-01T12.", None),
    ("2026-07-01T12:00:00.", None),
    ("2026-07-01T12:00:00.+00:00", "2026-07-01 12:00"),
    ("2026-07-01T00:00:00.1234567garbage", None),
    ("2026-07-01T00:00:00.1234567garbage+00:00", "2026-07-01 00:00"),
    ("2026-07-01T00:00:00.1234567+00:00garbage", None),
    ("2026-07-01T00:00:00+00:0030", None),
    ("2026-07-01T00:00:00+0000:30", None),
    ("2026-07-01T00:00:00+00", "2026-07-01 00:00"),
    ("2026-07-01T00:00:00+0030", "2026-06-30 23:30"),
    ("2026-07-01T00:00:00+003", None),
    ("2026-07-01T00:00:00+00:30.5", "2026-06-30 23:29"),
    ("2026-07-01T00:00:00+000030.5", "2026-06-30 23:59"),
    ("2026-07-01T00:00:00+00:00:30.5", "2026-06-30 23:59"),
    ("2026-07-01T00:00:00+00.5", "2026-07-01 00:00"),
    ("2026-07-01T00:00:00Z", "2026-07-01 00:00"),
    ("2026-07-01T00:00:00z", None),
    ("2026-07-01T00Z", "2026-07-01 00:00"),
    ("2026-07-01T0000Z", "2026-07-01 00:00"),
    ("2026-07-01T00:00Z", "2026-07-01 00:00"),
    ("2026-07-01T00:00:00+24:00", None),
    ("2026-07-01T00:00:00+23:59", "2026-06-30 00:01"),
    ("2026-07-01T24:00:00", None),
    ("2026-07-01T23:59:60", None),
    ("2026-07-01X00:00", "2026-07-01 00:00"),
    ("2026-07-01 00:00", "2026-07-01 00:00"),
    ("2026-07-01", "2026-07-01 00:00"),
    ("20260701", "2026-07-01 00:00"),
    ("2026-0701", None),
    ("202607-01", None),
    ("2026-07", None),
    ("2026W27", "2026-06-29 00:00"),
    ("2026W53", "2026-12-28 00:00"),
    ("2020W53", "2020-12-28 00:00"),
    ("2026-W53", "2026-12-28 00:00"),
    ("2026-W00", None),
    ("2026-W27-0", None),
    ("2026-W27-8", None),
    ("2026-W27-", None),
    ("2026W27-3", None),
    ("2026-W273", None),
    ("2026-07-01T00:00:00.123", "2026-07-01 00:00"),
    ("2026-07-01T00:00:00,123456789", "2026-07-01 00:00"),
    ("20260701T000000.5-0330", "2026-07-01 03:30"),
    ("2026-07-01T000000", "2026-07-01 00:00"),
    ("2026-07-01T00:0000", None),
    ("2026-07-01T0000:00", None),
    ("2026-07-01T00:00:00-00:00", "2026-07-01 00:00"),
    ("2026-07-01T00:00:00+00:00:00.000001", "2026-07-01 00:00"),
    ("+026-07-01", None),
    ("\uff12\uff10\uff12\uff16-07-01", None),
    ("2026-07-01T\uff11\uff12:00", None),
    ("2026-07-01T12:00:00+0\uff13:00", None),
    ("2026-07-01T12é+00:00", None),
    ("2026-07-01T12:30éZ", None),
    ("2026-07-01T12:00:00:5", "2026-07-01 12:00"),
    ("2026-07-01T1200005", None),
    ("2026-07-01T120000:5", None),
    ("2026-07-01T12:+00:00", "2026-07-01 12:00"),
    ("2026-07-01T12.+00:00", "2026-07-01 12:00"),
    ("2026-07-01T12:00:00+00:75", "2026-07-01 10:45"),
    ("2026-07-01T12:00:00+0075", "2026-07-01 10:45"),
    ("2026-07-01T12:00:00.123456garbage+00:00", "2026-07-01 12:00"),
    ("2026-07-01T12:00:00.12345garbage+00:00", None),
    ("2026-07-01T12:00:00.1234567\u00e9+00:00", "2026-07-01 12:00"),
    ("2026-07-01T12:00:00.1234567garbageZ", "2026-07-01 12:00"),
    ("2026-07-01T12:00:00.1234567garbage-03:30", "2026-07-01 15:30"),
    ("2026-07-01T12:00:00.1234567:+00:00", "2026-07-01 12:00"),
    ("2026-07-01\u00e912:00", "2026-07-01 12:00"),
    ("2026-07-01\U0001f60012:00", "2026-07-01 12:00"),
    ("2026-W27\u00e912:00", "2026-06-29 12:00"),
    ("2026-07-01T12:00:00+03", "2026-07-01 09:00"),
    ("2026-07-01T12:00:00+03:30:15.5", "2026-07-01 08:29"),
    ("2026-07-01T12:00:00Z+00:00", None),
    ("2026-07-01T12:00:00-", None),
    ("2026-07-01T12:00:00+", None),
    ("2026-07-01T", None),
    ("2026-07-01T+00:00", None),
    ("2026-07-01TZ", None),
    ("2026-07-01T12Z", "2026-07-01 12:00"),
    ("2026-07-01T12:30:15.5-00:00:00.5", "2026-07-01 12:30"),
    ("2026-07-01T12:30:15+00:00:00.5", "2026-07-01 12:30"),
    ("2026-07-01T00:00:00-00:00:00.5", "2026-07-01 00:00"),
    ("0001-01-01T00:00:00+01:00", "OVERFLOW"),
    ("9999-12-31T23:59:59-01:00", "OVERFLOW"),
    ("9999-W52-5", "9999-12-31 00:00"),
    ("9999-W52-6", None),
    ("2026-W27-7T23:59:59.999999-23:59:59.999999", "2026-07-06 23:59"),
    ("2026-W27-120000+03:00", "2026-06-29 09:00"),
    ("2026-W27-1200Z", "2026-06-29 12:00"),
    ("2026W2731200Z", "2026-06-29 12:00"),
    ("2026W27312000-0330", "2026-07-01 23:30"),
    ("2026-W27-12.5", "2026-06-29 12:00"),
    ("2026-W27-3-12", "2026-07-01 12:00"),
]


def _port_minute(text):
    try:
        moment = mi._iso_312(text)
    except ValueError:
        return None
    try:
        moment = moment.astimezone(datetime.timezone.utc) if moment.tzinfo is not None else moment
    except OverflowError:
        return "OVERFLOW"
    return moment.strftime("%Y-%m-%d %H:%M")


def _native(text):
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None


@pytest.mark.parametrize("text, expected", _ORACLE)
def test_port_reads_each_form_as_native_312_does(text, expected):
    """Includes 2026-W27-120000 (a digit after YYYY-Www- starts the time: Monday 12:00, not weekday 1 at 00:00),
    a fraction after the hour or minute (T00.5), junk after six fraction digits before an offset, which 3.12 admits,
    a zero offset whose fraction 3.12 drops, and a non-ASCII separator."""
    assert _port_minute(text) == expected
    shown = mi.minute(text)
    here = _native(text)  # this reader's own parser still answers first: 3.10 takes "...Z+00:00", 3.12 refuses it
    if expected is None and here is not None:
        assert sys.version_info[:2] != (3, 12) and shown == mi.minute(here.isoformat())
    elif expected in (None, "OVERFLOW"):
        assert shown == text
    else:
        assert shown == expected


def _corpus():
    dates = ["2026-07-01", "20260701", "2026-W27", "2026W27", "2026-W27-3", "2026W273", "2026-W53", "2020-W53-4",
             "2026-02-30", "2026-0701", "9999-W52-5", "0001-01-01"]
    seps = ["T", " ", "-", "1", "3", "\u00e9", ":"]
    clocks = ["", "12", "1230", "12:30", "123015", "12:30:15", "12:30:15:5", "12.5", "12:30.5", "1230,25", "1",
              "12:30:15.1234567", "12:30:15.123456x", "12:30:15.1234567x", "12:", "12.", "24:00", "12:60", "1200005"]
    zones = ["", "Z", "z", "+03", "+0330", "+03:30", "+033015.5", "+03:30:15.5", "-00:00:00.5", "+00.5", "+24:00",
             "+00:75", "+3", "+003", "Z+00:00", "+03:30x", "-23:59:59.999999", "+00:00:00.000001"]
    yield from dates
    for parts in product(dates, seps, clocks, zones):
        yield "".join(parts)
    rng, alphabet = random.Random(312), "0123456789-:.,+ZWT "
    for _ in range(20_000):
        head = rng.choice(dates + [""]) + rng.choice(seps + [""])
        yield head + "".join(rng.choice(alphabet) for _ in range(rng.randint(0 if head else 7, 24)))


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="the port follows CPython 3.12's own parser")
def test_port_matches_native_312_on_a_generated_corpus():
    def key(moment):
        return None if moment is None else (moment.replace(tzinfo=None), moment.utcoffset())

    mismatched, seen = [], 0
    for text in _corpus():
        seen += 1
        try:
            ported = mi._iso_312(text)
        except ValueError:
            ported = None
        if key(ported) != key(_native(text)):
            mismatched.append((text, key(ported), key(_native(text))))
    assert seen > 40_000 and not mismatched[:20], mismatched[:20]


def test_display_compatibility_does_not_expand_fresh_transfer_admission(tmp_path):
    shared.world(tmp_path)
    start = "2026-07-01T00:00:00." + "1" * 100_013 + "+00:00"
    record = _moved(tmp_path, "iso-fresh", "Unchanged words.", stated={**STATED, "start": start})
    result = ChronicleStore(tmp_path).publish([record])
    if sys.version_info[:2] == (3, 10):
        assert not result.ok and result.reason == "invalid"
        assert ChronicleStore(tmp_path).get(record["id"]) is None
    else:
        assert result.ok, result
        assert ChronicleStore(tmp_path).get(record["id"])["metadata"] == record["metadata"]
    assert mv._minute(start) == "2026-07-01 00:00"


def _publish_history(root):
    """Cloud writer entry point: actual publish on 3.12, retaining originals first. Each record states one period
    start and is written at the next record's start, so every form is read in both places."""
    assert sys.version_info[:2] == (3, 12), sys.version
    shared.world(root)
    long, wide = "1" * 100_000, "1" * 100_013
    forms = [
        ("2026-07-01T00:00:00." + long + "+00:00", "2026-07-01 00:00"),
        ("2026-07-01T00:00:00." + wide + "+03:30", "2026-06-30 20:30"),
        ("20260701T000000,1111111-0330", "2026-07-01 03:30"),
        ("2026-W27-3T00:00:00.11111111Z", "2026-07-01 00:00"),
        ("2026-07-01T00:00:00." + long + "+00:00:30", "2026-06-30 23:59"),
        ("2026-07-01T00:00:00." + wide + "+00:00:30.5", "2026-06-30 23:59"),
        ("20260701T000000," + wide + "+000030.5", "2026-06-30 23:59"),
        ("2026W273T000000." + long + "-000059.95", "2026-07-01 00:01"),
        # Both the time and the offset can carry a longer-than-microsecond fraction.
        ("2026-07-01T00:00:30." + long + "+00:00:30." + wide, "2026-07-01 00:00"),
        # 3.12's own readings: a digit after YYYY-Www- starts the time; a fraction may follow the hour; an odd run
        # of digits after YYYYWww is a time without a weekday; junk after six fraction digits before an offset.
        ("2026-W27-120000", "2026-06-29 12:00"),
        ("2026-07-01T00.5+03:30", "2026-06-30 20:30"),
        ("2026W2731200Z", "2026-06-29 12:00"),
        ("2026-07-01T00:00:00." + long + "garbage+00:00", "2026-07-01 00:00"),
    ]
    entries = []
    for number, (start, minute) in enumerate(forms):
        written, written_minute = forms[(number + 1) % len(forms)]
        for value, expected in ((start, minute), (written, written_minute)):
            native = datetime.datetime.fromisoformat(value)
            native = native.astimezone(datetime.timezone.utc) if native.tzinfo else native
            assert native.strftime("%Y-%m-%d %H:%M") == expected
        ident = f"iso-transfer-{number}"
        stated = {**STATED, "start": start}
        words = f"Original chapter {number}.\nСохранённые слова.\n"
        record = _moved(root, ident, words, room="1", stated=stated, written=written)
        original = json.dumps({"period": stated, "written": written, "words": words}, ensure_ascii=False).encode()
        record["source_refs"] = [chat_chain.retain_memory_source(
            SimpleNamespace(drive_root=root, task_id=MOVED["task_id"]), ident, original, "json")]
        _publish(root, record)
        entries.append({"record": ChronicleStore(root).get(ident), "minute": minute, "written_minute": written_minute,
                        "source_sha256": hashlib.sha256(original).hexdigest()})
    (root / "writer.json").write_text(json.dumps({"python": sys.version, "entries": entries}), encoding="utf-8")


def _whole_source(root, ref):
    parts, start = [], 0
    while True:
        page = _memory_read(_ctx(root), source_ref=ref, start=start)
        first, end, total, next_start = _parse_window(page)
        assert first == start and start < end <= total and len(page) <= LIMIT
        parts.append(page.split("\n", 2)[2])
        if next_start is None:
            assert end == total
            return "".join(parts)
        assert int(next_start) == end
        start = end


def test_real_312_publication_replays_with_short_full_and_f3_periods(tmp_path):
    writer = os.environ.get("OURO_TEST_TRANSFER_WRITER_PYTHON")
    if not writer:
        if sys.version_info[:2] != (3, 12):
            pytest.skip("cross-version replay needs OURO_TEST_TRANSFER_WRITER_PYTHON=Python-3.12")
        writer = sys.executable
    subprocess.run([writer, "-m", "tests.test_memory_view_iso_period", str(tmp_path)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=120)
    receipt = json.loads((tmp_path / "writer.json").read_text(encoding="utf-8"))
    assert receipt["python"].startswith("3.12.")
    store = ChronicleStore(tmp_path)
    journal = store.log_path.read_bytes()
    store.index_path.unlink()  # exercise replay, not the writer's materialized index
    for entry in receipt["entries"]:
        record = entry["record"]
        assert store.get(record["id"]) == record
        ref = record["source_refs"][0]
        original = read_actor_source_bytes(tmp_path, ref["task_id"], ref)
        assert hashlib.sha256(original).hexdigest() == entry["source_sha256"]
        meta = record["metadata"]
        assert json.loads(original) == {"period": meta["stated_period"], "written": meta["legacy_written_at"],
                                        "words": record["text"]}
        assert _whole_source(tmp_path, ref).encode("utf-8") == original
        address = {"kind": "record", "id": record["id"]}
        for field in (["metadata", "stated_period", "start"], ["metadata", "legacy_written_at"], ["text"]):
            exact = record
            for key in field:
                exact = exact[key]
            assert whole(tmp_path, {**address, "field": field}) == exact  # the exact original, by its address
        # memory_read(node_id): the period and the writing date in minutes, so one page holds the whole document.
        page = _memory_read(_ctx(tmp_path), node_id=record["id"])
        head, (first, end, total, next_start) = page.split("\n", 1)[0], _parse_window(page)
        assert (first, end, next_start) == (0, total, None) and len(page) <= LIMIT and len(head) < 2_000
        assert f"; stated period {entry['minute']} → 2026-07-31 00:00; " in head
        assert f"; written {entry['written_minute']}; " in head
        assert f"\ntext:\n{record['text']}" in page and "retained source: memory_read(source_ref=" in page
        assert meta["stated_period"]["start"] not in page and meta["legacy_written_at"] not in page
    snapshot = _view(tmp_path)
    full = mv.render_story(snapshot)
    f3 = mv.render_story(snapshot, mv.FloorLevel((("F3", ("1",)),)))
    for name, text in (("full", full), ("F3", f3)):
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    (tmp_path / "reader.json").write_text(json.dumps({"python": sys.version,
        "full_chars": len(full), "F3_chars": len(f3), "journal_sha256": hashlib.sha256(journal).hexdigest()}))
    ids = tuple(entry["record"]["id"] for entry in receipt["entries"])
    fold = memory_fallback._part_input(tmp_path, store, memory_fallback.FallbackUnit(
        "part", "1", "part:1:probe", member_ids=ids, head_sequence=store.room_head("1")), None, len).prompt
    for entry in receipt["entries"]:
        record, meta = entry["record"], entry["record"]["metadata"]
        period = entry["minute"] + " → 2026-07-31 00:00 (stated by its source)"
        assert f"#### {record['id']} — {period} — " in full and f"written {entry['written_minute']}" in full
        assert mv._indented(record["text"]) in full
        assert record["id"] in f3 and record["text"] not in f3
        assert f"### legacy {record['id']} — by " in fold and f" — period {period} — " in fold
        assert f"; written {entry['written_minute']}: keep its stated author" in fold
        for shown in (full, f3, fold):
            assert meta["stated_period"]["start"] not in shown and meta["legacy_written_at"] not in shown
    earliest = min(entry["minute"] for entry in receipt["entries"])
    assert f"{earliest} → 2026-07-31 00:00 (stated by its source)" in f3 and earliest == "2026-06-29 12:00"
    assert "Main was quiet." in full and "retold record" in f3  # old helper memory still present
    assert "1" * 100 not in full + f3 + fold
    assert store.log_path.read_bytes() == journal
    assert all(store.get(e["record"]["id"]) == e["record"] for e in receipt["entries"])


if __name__ == "__main__":
    _publish_history(Path(sys.argv[1]))
