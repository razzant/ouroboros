"""A knowledge history row stores a version as its change against a hash-named base, and every version reads back exactly.

The journals here are built the way the writer builds them (``knowledge_rows.encode`` against the previous row,
the byte rule deciding when a text is stored whole) and then damaged on purpose: a returned text always hashes to
its version, and a version that cannot be rebuilt is a typed miss that names where the path broke, never a guess.
"""
from __future__ import annotations

import hashlib
import json
import time

import pytest

from ouroboros import knowledge_rows as rows
from ouroboros import text_change

TOPIC = "notes/река"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(journal, texts, topic=TOPIC):
    """Each text a version of one note, written as the writer writes it: its base is the previous row."""
    offsets, previous = [], None
    with journal.open("ab") as stream:
        for text in texts:
            offset = stream.tell()
            old = "" if previous is None else previous[2]
            fields = rows.encode(old, text, None if previous is None else previous[:2])
            row = {"topic": topic, "address": {"scope": "global"}, "publication": "source_capture",
                   "old_sha256": "" if previous is None else sha(old), "new_sha256": sha(text), **fields}
            stream.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            offsets.append(offset)
            previous = (offset, row, text)
    return offsets


def stored(journal):
    return [json.loads(line) for line in journal.read_bytes().splitlines()]


def damage(journal, offset):
    """The line at ``offset`` becomes bytes that are not JSON, of the same length: every other offset holds."""
    raw = journal.read_bytes()
    end = raw.index(b"\n", offset)
    journal.write_bytes(raw[:offset] + b"#" * (end - offset) + raw[end:])


def read(journal, offset, side, live=None):
    return rows.Texts(journal, live).text(offset, side)


VERSIONS = {
    "crlf": ["# Река\r\nСтарый берег.\r\nДо встречи.\r\n", "# Река\r\nНовый берег.\r\nДо встречи.\r\n",
             "# Река\r\nНовый берег.\r\nДо встречи!\r\n"],
    "no final newline": ["one\ntwo\nthree", "one\n2\nthree", "one\n2\nthree\nfour"],
    "multibyte": ["Привет, мир 🌍!\n" * 20, "Привет, дивный мир 🌏!\n" + "Привет, мир 🌍!\n" * 19,
                  "Привет, дивный мир 🌏!\n" + "Привет, мир 🌍!\n" * 18 + "Ёлка 🎄\n"],
    "empty to text and back": ["", "# Note\n\nFirst.\n", "# Note\n\nFirst, and more.\n", ""],
}


@pytest.mark.parametrize("name", sorted(VERSIONS))
def test_every_version_reads_back_exactly_from_its_row_on_both_sides(tmp_path, name):
    journal, texts = tmp_path / "knowledge_history.jsonl", VERSIONS[name]
    offsets = build(journal, texts)
    written = stored(journal)
    for index, offset in enumerate(offsets):
        assert read(journal, offset, "new") == texts[index]
        if index:
            assert read(journal, offset, "old") == texts[index - 1]
            # Every row with a base carries its change; a whole row carries it too.
            assert written[index]["base"] == {"offset": offsets[index - 1], "sha256": sha(texts[index - 1])}
            assert text_change.apply_change(texts[index - 1], written[index]["change"]) == texts[index]
            assert "old_content" not in written[index] and "old_content_ref" not in written[index]
    with pytest.raises(rows.TextUnavailable, match="names no old version"):
        read(journal, offsets[0], "old")  # the first row's note did not exist: no version, never a guessed ""


def test_the_byte_rule_stores_a_whole_text_when_the_changes_outweigh_it_and_keeps_the_change(tmp_path):
    lines = [f"line {index}: {'word ' * 8}\n" for index in range(40)]
    base = "".join(lines)
    texts = [base]
    for index in range(40):  # one line rewritten per version: small changes that add up past the text
        lines[index] = f"line {index}: {'other ' * 8}\n"
        texts.append("".join(lines))
    texts.append("Rewritten from scratch.\n")  # a change at least as large as the new text
    journal = tmp_path / "knowledge_history.jsonl"
    offsets = build(journal, texts)
    written = stored(journal)
    running, wholes = 0, []
    for index, row in enumerate(written[1:], 1):
        change = rows.change_bytes(row["change"])
        running += change
        if "new_content" in row:  # stored whole: the change was not smaller, or the chain since a whole text outgrew it
            wholes.append(index)
            assert change >= len(texts[index].encode("utf-8")) or running > len(texts[index].encode("utf-8"))
            assert "chain_bytes" not in row
            running = 0
        else:
            assert row["chain_bytes"] == running <= len(texts[index].encode("utf-8"))
    assert wholes[-1] == len(texts) - 1 and len(wholes) >= 2, wholes
    assert all(read(journal, offset, "new") == texts[index] for index, offset in enumerate(offsets))


def test_a_change_that_does_not_rebuild_both_texts_keeps_both_texts_whole():
    base = (0, {"new_content": "A\n", "new_sha256": sha("A\n")})
    assert rows.encode("A\n", "B\n", base, change=[(0, "Z", "B")]) == {"old_content": "A\n", "new_content": "B\n"}
    assert rows.encode("A\n", "B\n", None) == {"old_content": "A\n", "new_content": "B\n"}  # no verified base
    # A base row whose running sum cannot be read starts the chain again, still carrying its change.
    unread = (0, {"new_sha256": sha("A\n" * 50), "base": {"offset": 0, "sha256": "0" * 64}, "change": []})
    fields = rows.encode("A\n" * 50, "A\n" * 49 + "B\n", unread)
    assert fields["new_content"] == "A\n" * 49 + "B\n" and fields["change"] == [[98, "A", "B"]]


def test_a_wrong_base_address_is_caught_by_the_hash_and_never_applied(tmp_path):
    journal = tmp_path / "knowledge_history.jsonl"
    texts = ["# A\n\nalpha\n" * 30, "# A\n\nbeta\n" + "# A\n\nalpha\n" * 29, "# A\n\nbeta\n" + "# A\n\nalpha\n" * 28 + "gamma\n"]
    offsets = build(journal, texts)
    raw = journal.read_bytes().decode("utf-8").splitlines(keepends=True)
    wrong = json.loads(raw[2])
    wrong["base"]["offset"] = offsets[0]  # names the first row, which holds another version
    raw[2] = json.dumps(wrong, ensure_ascii=False) + "\n"
    journal.write_text("".join(raw), encoding="utf-8")
    # The rows link by hash: the version the base names is found where it is, and the text is exact.
    assert read(journal, offsets[2], "new") == texts[2] and read(journal, offsets[2], "old") == texts[1]
    damage(journal, offsets[1])  # now only the wrong address names that version: caught, never applied
    with pytest.raises(rows.TextUnavailable) as caught:
        read(journal, offsets[2], "new")
    assert (f"the base the row at byte offset {offsets[2]} names does not hash to its version: the row at byte offset "
            f"{offsets[0]} holds {sha(texts[0])}, not {sha(texts[1])}") in str(caught.value), caught.value
    # A base whose version is wrong too: the change cannot rebuild the text it names, and nothing is guessed.
    wrong["base"]["sha256"] = wrong["old_sha256"] = sha(texts[0])
    raw = journal.read_bytes().decode("utf-8").splitlines(keepends=True)
    raw[2] = json.dumps(wrong, ensure_ascii=False) + "\n"
    journal.write_text("".join(raw), encoding="utf-8")
    with pytest.raises(rows.TextUnavailable, match="does not rebuild its version"):
        read(journal, offsets[2], "new")


def _chain(tmp_path, count=5):
    journal = tmp_path / "knowledge_history.jsonl"
    body = "".join(f"- fact {index}: {'value ' * 6}\n" for index in range(30))
    texts = [body]
    for index in range(1, count):
        texts.append(texts[-1].replace(f"- fact {index}: value", f"- fact {index}: revised", 1))
    offsets = build(journal, texts)
    assert all("new_content" not in row for row in stored(journal)[1:])  # one segment: a whole text, then changes
    return journal, texts, offsets


def test_one_damaged_middle_row_loses_no_landed_version_and_its_own_read_is_a_typed_gap(tmp_path):
    journal, texts, offsets = _chain(tmp_path)  # A whole, then B, C, D, E as changes; the note is E
    damage(journal, offsets[2])  # the row that wrote C
    live = texts[-1]
    for index in range(len(texts)):  # every version, through any row that names it, from either end of the segment
        if index != 2:
            assert read(journal, offsets[index], "new", live) == texts[index]
        if index > 2:
            assert read(journal, offsets[index], "old", live) == texts[index - 1]
    assert read(journal, offsets[1], "old", live) == texts[0]
    with pytest.raises(rows.TextUnavailable, match=f"the line at byte offset {offsets[2]} is not a JSON row"):
        read(journal, offsets[2], "new", live)
    # Without the live note the segment's newer end is gone: its versions are a typed gap that names the break.
    with pytest.raises(rows.TextUnavailable) as gap:
        read(journal, offsets[3], "new")
    assert f"the line at byte offset {offsets[2]} is not a JSON row" in str(gap.value)
    assert "no other stored text of this note connects to it" in str(gap.value)


def test_two_damaged_rows_lose_only_the_versions_between_them_and_never_return_a_wrong_text(tmp_path):
    journal, texts, offsets = _chain(tmp_path, count=6)  # A whole, then B..F as changes; the note is F
    damage(journal, offsets[1])  # the row that wrote B
    damage(journal, offsets[3])  # the row that wrote D
    live, kept, lost = texts[-1], {}, set()
    for index, offset in enumerate(offsets):
        for side, version in (("new", index), ("old", index - 1)):
            if index in (1, 3) or version < 0:
                continue
            try:
                kept[version] = read(journal, offset, side, live)
            except rows.TextUnavailable:
                lost.add(version)
    assert all(sha(text) == sha(texts[version]) for version, text in kept.items())
    assert sorted(kept) == [0, 3, 4, 5] and lost == {1, 2}  # B and C lay between the two damaged rows


def test_a_whole_row_carrying_its_change_keeps_a_damaged_segment_readable_from_its_newer_end(tmp_path):
    """Astra's chain: whole A, change A→B, change B→C, whole D carrying C→D. Damaging the A→B row loses no version:
    C is D minus its change, B is C minus its own."""
    journal = tmp_path / "knowledge_history.jsonl"
    a = "".join(f"- item {index}\n" for index in range(60))
    b = a.replace("- item 7\n", "- item seven\n")
    c = b.replace("- item 9\n", "- item nine\n")
    d = "Rewritten whole.\n" + c[:40]
    offsets = build(journal, [a, b, c, d])
    written = stored(journal)
    assert "new_content" in written[3] and written[3]["base"]["offset"] == offsets[2] and written[3]["change"]
    damage(journal, offsets[1])
    assert read(journal, offsets[2], "new") == c and read(journal, offsets[2], "old") == b
    assert read(journal, offsets[3], "old") == c and read(journal, offsets[0], "new") == a


def test_a_legacy_reference_reads_the_earlier_row_it_names(tmp_path):
    """A row written before changes were stored names its replaced text by ``old_content_ref``; it reads the
    same bytes as before, and a change row may take a legacy row as its base."""
    journal = tmp_path / "knowledge_history.jsonl"
    first, second, third = "# Old\n\nOne.\n", "# Old\n\nTwo.\n", "# Old\n\nTwo, then three.\n"
    legacy = {"topic": TOPIC, "address": {"scope": "global"}, "publication": "source_capture"}
    head = json.dumps({**legacy, "old_sha256": "", "new_sha256": sha(first), "old_content": "", "new_content": first},
                      ensure_ascii=False) + "\n"
    ref = {"kind": "entry", "journal": "knowledge:global", "offset": 0, "field": ["new_content"]}
    named = json.dumps({**legacy, "old_sha256": sha(first), "new_sha256": sha(second), "old_content_ref": ref,
                        "new_content": second}, ensure_ascii=False) + "\n"
    journal.write_text(head + named, encoding="utf-8")
    second_row = json.loads(named)
    fields = rows.encode(second, third, (len(head.encode("utf-8")), second_row))
    journal.write_text(head + named + json.dumps({**legacy, "old_sha256": sha(second), "new_sha256": sha(third),
                                                  **fields}, ensure_ascii=False) + "\n", encoding="utf-8")
    at = len((head + named).encode("utf-8"))
    assert read(journal, len(head.encode("utf-8")), "old") == first
    assert read(journal, at, "old") == second and read(journal, at, "new") == third
    elsewhere = {**second_row, "old_content_ref": {**ref, "journal": "knowledge:project:p"}}  # another journal's address
    texts = rows.Texts(journal)
    texts.rows[len(head.encode("utf-8"))] = elsewhere
    assert texts.text(len(head.encode("utf-8")), "old") == first  # not followed, yet the version is found by its hash


def test_the_longest_chain_reads_its_newest_version_forward_and_its_oldest_backward(tmp_path, capsys):
    """Timings, reported and not asserted tightly: the byte rule allows a chain as long as its changes stay
    smaller than the text, so a large note edited a little at a time builds the longest chain."""
    journal = tmp_path / "knowledge_history.jsonl"
    lines = [f"{index:05d} {'stable words of a long note ' * 3}\n" for index in range(1200)]
    texts = ["".join(lines)]
    for index in range(1, 600):
        lines[index] = lines[index].replace("stable", "edited", 1)
        texts.append("".join(lines))
    started = time.perf_counter()
    offsets = build(journal, texts)
    built = time.perf_counter() - started
    assert all("new_content" not in row for row in stored(journal)[1:])  # one chain of 599 changes
    started = time.perf_counter()
    assert read(journal, offsets[-1], "new") == texts[-1]  # forward from the whole text through every change
    forward = time.perf_counter() - started
    damage(journal, offsets[0])  # the whole text is gone: the oldest version comes back from the live note
    started = time.perf_counter()
    assert read(journal, offsets[1], "old", texts[-1]) == texts[0]
    backward = time.perf_counter() - started
    with capsys.disabled():
        print(f"\n[knowledge_rows timing] {len(texts)} versions of a {len(texts[0].encode())}-byte note, "
              f"journal {journal.stat().st_size} bytes: encoded in {built:.2f}s; newest read forward through "
              f"{len(offsets) - 1} changes in {forward:.3f}s; oldest read backward from the live note in {backward:.3f}s")
    assert forward < 60 and backward < 60
