"""A knowledge history row stores a version as its change against a hash-named base, and every version reads back exactly.

The journals here are written by the writer (``knowledge.write_knowledge_note``), or built the way it builds them
(``knowledge_rows.encode`` against the previous row, the byte rule deciding when a text is stored whole), and then
damaged on purpose: a returned text always hashes to its version, and a version that cannot be rebuilt is a typed
miss that names where the path broke, never a guess.
"""
from __future__ import annotations

import hashlib
import json
import time

import pytest

from ouroboros import knowledge as store
from ouroboros import knowledge_rows as rows
from ouroboros import knowledge_versions as versions
from ouroboros import text_change
from ouroboros.memory_reader import read_memory
from tests._knowledge_history_shared import changed_rows

TOPIC = "notes/река"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(journal, texts):
    """Each text a version of one note, written as the writer writes it: its base is the previous row."""
    return changed_rows(journal, texts, TOPIC)


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


def test_the_byte_rule_at_its_two_boundaries():
    """A change exactly as large as the new text is not smaller than it: the text is stored whole; a running sum exactly
    as large keeps the row a change, and one byte more stores it whole. Two byte counts, compared as signed."""
    whole_base = (0, {"new_content": "AB", "new_sha256": sha("AB")})
    equal = rows.encode("AB", "AX", whole_base)  # change ["B" → "X"]: 2 bytes; new text: 2 bytes
    assert rows.change_bytes(equal["change"]) == 2 and equal["new_content"] == "AX" and "chain_bytes" not in equal
    smaller = rows.encode("ABC", "AXC", (0, {"new_content": "ABC", "new_sha256": sha("ABC")}))  # 2 bytes < 3
    assert "new_content" not in smaller and smaller["chain_bytes"] == 2
    at_limit = rows.encode("ABC", "AXC", (0, {"chain_bytes": 1}))  # a change row's sum 1, plus 2: exactly 3
    assert "new_content" not in at_limit and at_limit["chain_bytes"] == 3
    past = rows.encode("ABC", "AXC", (0, {"chain_bytes": 2}))  # 2 + 2 = 4: one byte past the text
    assert past["new_content"] == "AXC" and "chain_bytes" not in past and past["change"] == [[1, "B", "X"]]


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
    lines = journal.read_bytes().splitlines(keepends=True)  # byte-exact: every offset is the file's own
    wrong = json.loads(lines[2])
    wrong["base"]["offset"] = offsets[0]  # names the first row, which holds another version
    lines[2] = (json.dumps(wrong, ensure_ascii=False) + "\n").encode("utf-8")
    journal.write_bytes(b"".join(lines))
    # The rows link by hash: the version the base names is found where it is, and the text is exact.
    assert read(journal, offsets[2], "new") == texts[2] and read(journal, offsets[2], "old") == texts[1]
    damage(journal, offsets[1])  # now only the wrong address names that version: caught, never applied
    with pytest.raises(rows.TextUnavailable) as caught:
        read(journal, offsets[2], "new")
    assert (f"the base the row at byte offset {offsets[2]} names does not hash to its version: the row at byte offset "
            f"{offsets[0]} holds {sha(texts[0])}, not {sha(texts[1])}") in str(caught.value), caught.value
    # A base whose version is wrong too: the change cannot rebuild the text it names, and nothing is guessed.
    wrong["base"]["sha256"] = wrong["old_sha256"] = sha(texts[0])
    lines = journal.read_bytes().splitlines(keepends=True)
    lines[2] = (json.dumps(wrong, ensure_ascii=False) + "\n").encode("utf-8")
    journal.write_bytes(b"".join(lines))
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


def test_a_row_is_read_only_where_a_journal_line_starts(tmp_path):
    """``Texts.row`` reads a row only at a line start: after a base row whose line ends with CRLF, the change row at its
    exact offset reads both its versions; an offset between that CR and its LF, or inside a line, names no row."""
    journal = tmp_path / "knowledge_history.jsonl"
    old, new = "# Река\r\nOld.\r\n", "# Река\r\nNew.\r\n"
    head = {"topic": TOPIC, "address": {"scope": "global"}, "publication": "source_capture", "old_sha256": "",
            "new_sha256": sha(old), "old_content": "", "new_content": old}
    first = (json.dumps(head, ensure_ascii=False) + "\r\n").encode("utf-8")  # a line some other tool ended with CRLF
    after = {**head, "old_sha256": sha(old), "new_sha256": sha(new)}
    del after["old_content"], after["new_content"]
    after.update(rows.encode(old, new, (0, head)))
    journal.write_bytes(first + (json.dumps(after, ensure_ascii=False) + "\n").encode("utf-8"))
    at = len(first)
    assert "new_content" not in after and after["base"]["offset"] == 0
    assert read(journal, at, "new") == new and read(journal, at, "old") == old
    assert rows.Texts(journal).row(0)["new_sha256"] == sha(old)
    for offset in (at - 1, 3):  # between the CR and its LF; inside the first line
        with pytest.raises(rows.TextUnavailable, match=f"^no journal line starts at byte offset {offset}$"):
            rows.Texts(journal).row(offset)


def test_a_base_naming_its_own_row_or_a_later_one_is_never_followed(tmp_path):
    """A base names an earlier row. One that names the row itself or a row after it is no base: neither the forward
    path nor the hash search follows it (no loop, no forward link), so that row's version is a typed miss; the same row
    with its real base reads."""
    journal, texts, offsets = _chain(tmp_path, count=4)
    assert read(journal, offsets[2], "new") == texts[2]
    lines = journal.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[2])
    for later in (False, True):
        at = offsets[2]  # its own row
        while later:  # the row after it, where that row starts once this one is rewritten
            row["base"]["offset"] = at
            after = offsets[2] + len((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            if after == at:
                break
            at = after
        row["base"]["offset"] = at
        journal.write_bytes(b"".join(lines[:2]) + (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                            + b"".join(lines[3:]))
        assert at == offsets[2] or json.loads(journal.read_bytes()[at:].split(b"\n", 1)[0])["new_sha256"] == sha(texts[3])
        with pytest.raises(rows.TextUnavailable, match=f"the row at byte offset {offsets[2]} holds no text of version "
                                                      f"{sha(texts[2])} and names no base"):
            read(journal, offsets[2], "new")


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
    journal.write_bytes((head + named).encode("utf-8"))
    second_row = json.loads(named)
    fields = rows.encode(second, third, (len(head.encode("utf-8")), second_row))
    journal.write_bytes((head + named + json.dumps({**legacy, "old_sha256": sha(second), "new_sha256": sha(third),
                                                    **fields}, ensure_ascii=False) + "\n").encode("utf-8"))
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


# --- the writer --------------------------------------------------------------------------------------------------


def note(tmp_path, topic=TOPIC):
    return store.resolve_knowledge_address(tmp_path, topic, "global")


def journal_rows(target):
    """``(offset, row)`` of every JSON line of the note's shelf journal; a line that is not JSON is left out."""
    raw, out, at = (target.shelf.parent / "knowledge_history.jsonl").read_bytes(), [], 0
    for line in raw.splitlines(keepends=True):
        try:
            out.append((at, json.loads(line)))
        except ValueError:
            pass
        at += len(line)
    return out


def every_version_reads(target, texts):
    for text in texts:
        found, _row = versions.find_version(target, sha(text))
        assert found == text


def test_a_torn_last_row_makes_the_next_write_whole_and_every_version_reads_back(tmp_path):
    """The journal lost the end of the row of the version the note now holds (no fsync): no row reads back as the
    current version, so the next row keeps both texts whole; that row is one whole row, and no landed version is lost."""
    target = note(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    b = store.write_knowledge_note(target, "# Река\n\nOne, two.\n", expected_revision=a.revision).current
    c = store.write_knowledge_note(target, "# Река\n\nOne, two, three.\n", expected_revision=b.revision).current
    journal = target.shelf.parent / "knowledge_history.jsonl"
    torn_at = journal_rows(target)[-1][0]
    assert "base" in journal_rows(target)[-1][1]
    journal.write_bytes(journal.read_bytes()[:torn_at + 40])  # the row of `c` is torn mid-line
    d = store.write_knowledge_note(target, "# Река\n\nFour.\n", expected_revision=c.revision).current
    offset, row = journal_rows(target)[-1]
    assert offset == torn_at + 41 and "base" not in row and "change" not in row  # the boundary byte, then the row
    assert (row["old_content"], row["new_content"]) == (c.text, d.text)
    every_version_reads(target, [a.text, b.text, c.text, d.text])
    with pytest.raises(rows.TextUnavailable, match=f"the line at byte offset {torn_at} is not a JSON row"):
        read(journal, torn_at, "new")
    e = store.write_knowledge_note(target, "# Река\n\nFour, five.\n", expected_revision=d.revision).current
    assert journal_rows(target)[-1][1]["base"]["offset"] == offset  # the whole row is the next base
    every_version_reads(target, [a.text, b.text, c.text, d.text, e.text])


def test_a_note_deleted_or_edited_outside_the_writer_makes_the_next_row_whole_with_the_old_text_inline(tmp_path):
    target = note(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    b = store.write_knowledge_note(target, "# Река\n\nOne, two.\n", expected_revision=a.revision).current
    target.path.write_bytes(b"# \xd0\xa0\xd0\xb5\xd0\xba\xd0\xb0\r\n\r\nEdited by hand.")  # CRLF, no final newline
    outside = store.read_knowledge_note(target)
    c = store.write_knowledge_note(target, "# Река\n\nThree.\n", expected_revision=outside.revision).current
    row = journal_rows(target)[-1][1]
    assert {"base", "change"}.isdisjoint(row) and (row["old_content"], row["new_content"]) == (outside.text, c.text)
    target.path.unlink()
    d = store.write_knowledge_note(target, "# Река\n\nAgain.\n").current
    row = journal_rows(target)[-1][1]
    assert {"base", "change"}.isdisjoint(row) and (row["old_sha256"], row["old_content"]) == ("", "")
    every_version_reads(target, [a.text, b.text, outside.text, c.text, d.text])


@pytest.mark.parametrize("closed_by", ["an outside edit", "a deleted note", None])
def test_a_segment_closed_outside_the_writer_reads_only_forward_so_a_damaged_row_loses_the_versions_after_it(
        tmp_path, closed_by):
    """The live note is a segment's newer end only while it still holds the segment's last version. After an outside
    edit or a deleted note the next row has no verified base and keeps the outside text inline, so the segment whole
    A, change B, change C reads only forward from A: damaging B's row loses B and C, as typed misses, never a wrong
    text. While the live note still holds C (and the next write takes C's row as its base), nothing is lost."""
    target = note(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    b = store.write_knowledge_note(target, "# Река\n\nOne, two.\n", expected_revision=a.revision).current
    c = store.write_knowledge_note(target, "# Река\n\nOne, two, three.\n", expected_revision=b.revision).current
    (_, _), (b_at, b_row), (_, c_row) = journal_rows(target)
    assert "new_content" not in b_row and "new_content" not in c_row  # the segment's newer end is the live note alone
    outside = []
    if closed_by == "an outside edit":
        target.path.write_bytes("# Река\n\nEdited by hand.\n".encode("utf-8"))
        outside.append(store.read_knowledge_note(target).text)
    elif closed_by == "a deleted note":
        target.path.unlink()
    d = store.write_knowledge_note(target, "# Река\n\nFour.\n", expected_revision=(
        None if closed_by == "a deleted note" else sha(outside[0]) if outside else c.revision)).current
    assert ("base" in journal_rows(target)[-1][1]) is (closed_by is None)
    damage(target.shelf.parent / "knowledge_history.jsonl", b_at)
    every_version_reads(target, [a.text, d.text, *outside])
    if closed_by is None:
        every_version_reads(target, [b.text, c.text])  # from the live note back through D's and C's changes
        return
    for lost in (b.text, c.text):
        with pytest.raises(versions.VersionUnavailable, match=f"the line at byte offset {b_at} is not a JSON row"):
            versions.find_version(target, sha(lost))


def test_a_capture_that_never_landed_is_nobodys_base_and_lives_only_in_its_own_row(tmp_path, monkeypatch):
    target = note(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    landed_at = journal_rows(target)[-1][0]
    real_write = store.write_bytes_atomic

    def no_source(path, data):
        if path == target.path:
            raise OSError("disk")
        return real_write(path, data)

    monkeypatch.setattr(store, "write_bytes_atomic", no_source)
    lost = store.write_knowledge_note(target, "# Река\n\nNever landed.\n", expected_revision=a.revision)
    assert not lost.ok and target.path.read_bytes() == a.raw
    monkeypatch.undo()
    capture_at, capture = journal_rows(target)[-1]
    assert capture["base"]["offset"] == landed_at
    c = store.write_knowledge_note(target, "# Река\n\nOne, then two.\n", expected_revision=a.revision).current
    row = journal_rows(target)[-1][1]
    assert row["base"] == {"offset": landed_at, "sha256": a.revision}  # the landed version, not the capture
    every_version_reads(target, [a.text, c.text])
    journal = target.shelf.parent / "knowledge_history.jsonl"
    never = read(journal, capture_at, "new")
    assert sha(never) == capture["new_sha256"] and never.endswith("# Река\n\nNever landed.\n")
    damage(journal, capture_at)  # its own row was its only home; the landed versions keep theirs
    every_version_reads(target, [a.text, c.text])
    with pytest.raises(versions.VersionUnavailable):
        versions.find_version(target, capture["new_sha256"])


def test_a_base_whose_chain_is_broken_is_passed_over_for_an_older_row_that_reads_forward(tmp_path):
    """A revert put the first version back; then the row between them is damaged. The revert's row names the current
    version but no longer reads forward, so it is no base: the first row, which holds that version whole, is. Every
    version still reads, the damaged row's own version included (the revert's change, unapplied from the first)."""
    target = note(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    b = store.write_knowledge_note(target, "# Река\n\nOne, two.\n", expected_revision=a.revision).current
    assert store.write_knowledge_note(target, "", "revert", b.revision, version=a.revision).ok
    (first, _), (second, _), (revert, revert_row) = journal_rows(target)
    assert revert_row["base"]["offset"] == second and revert_row["new_sha256"] == a.revision
    damage(target.shelf.parent / "knowledge_history.jsonl", second)
    c = store.write_knowledge_note(target, "# Река\n\nOne, three.\n", expected_revision=a.revision).current
    assert journal_rows(target)[-1][1]["base"] == {"offset": first, "sha256": a.revision}
    every_version_reads(target, [a.text, b.text, c.text])


def test_rows_written_before_changes_then_change_rows_read_every_version_and_their_lines_as_stored(tmp_path):
    """A journal of whole rows as an earlier body wrote them (an inline old text, then an ``old_content_ref``),
    followed by change rows this writer adds: every version reads, and the earlier rows' lines, stored values and
    references read byte for byte as before."""
    target = note(tmp_path)
    journal = target.shelf.parent / "knowledge_history.jsonl"
    first, second = "# Река\n\nOld one.\n", "# Река\n\nOld two.\n"
    base = {"topic": TOPIC, "address": target.as_dict(), "publication": "source_capture", "mode": "overwrite"}
    head = json.dumps({**base, "old_sha256": "", "new_sha256": sha(first), "old_content": "", "new_content": first},
                      ensure_ascii=False) + "\n"
    ref = {"kind": "entry", "journal": "knowledge:global", "offset": 0, "field": ["new_content"]}
    named = json.dumps({**base, "old_sha256": sha(first), "new_sha256": sha(second), "old_content_ref": ref,
                        "new_content": second}, ensure_ascii=False) + "\n"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_bytes((head + named).encode("utf-8"))
    target.path.parent.mkdir(parents=True, exist_ok=True)
    target.path.write_bytes(second.encode("utf-8"))
    before = journal.read_bytes()
    third = store.write_knowledge_note(target, "# Река\n\nNew three.\n", expected_revision=sha(second)).current
    fourth = store.write_knowledge_note(target, "# Река\n\nNew three, four.\n", expected_revision=third.revision).current
    assert journal.read_bytes().startswith(before)  # no row is rewritten
    added = journal_rows(target)[2:]
    assert added[0][1]["base"] == {"offset": len(head.encode("utf-8")), "sha256": sha(second)}
    assert added[1][1]["base"]["offset"] == added[0][0] and "new_content" not in added[1][1]
    every_version_reads(target, [first, second, third.text, fourth.text])

    def entry(offset, field):
        return read_memory(tmp_path, "document", {"address": {"kind": "entry", "journal": "knowledge:global",
                                                              "offset": offset, "field": field}})["text"]

    assert entry(0, []) == head.rstrip("\n") and entry(len(head.encode("utf-8")), []) == named.rstrip("\n")
    assert read_memory(tmp_path, "document", {"address": ref})["text"] == first
    assert entry(added[1][0], ["new_content"]) == fourth.text and entry(added[1][0], ["old_content"]) == third.text
    assert entry(added[0][0], ["old_content"]) == second


def test_a_full_rewrite_of_a_300_kb_note_with_repeated_lines_writes_and_reads_back(tmp_path, capsys):
    """Timing, reported and not asserted tightly: the diff's worst case (repeated lines) on a large note, whose change
    is not smaller than the text, so the row stores the whole text and still carries the change."""
    target = note(tmp_path, "notes/large")
    old = "".join(f"- item {index % 7}: the same few lines again\n" for index in range(9000))
    texts = {"new repeated lines": "".join(f"* entry {index % 5}: another set of repeated lines\n" for index in range(7500)),
             "the same lines reordered": "".join(f"- item {index * 3 % 7}: the same few lines again\n" for index in range(9100))}
    first = store.write_knowledge_note(target, old).current
    previous, seen = first, [first.text]
    for name, new in texts.items():
        started = time.perf_counter()
        rewritten = store.write_knowledge_note(target, new, expected_revision=previous.revision)
        took = time.perf_counter() - started
        assert rewritten.ok and len(previous.raw) > 300_000 and len(rewritten.current.raw) > 300_000
        row = journal_rows(target)[-1][1]
        assert row["new_content"] == rewritten.current.text and row["change"] and "old_content" not in row
        previous = rewritten.current
        seen.append(previous.text)
        with capsys.disabled():
            print(f"\n[knowledge_rows timing] full rewrite ({name}) of a {len(first.raw)}-byte note with repeated lines "
                  f"into {len(previous.raw)} bytes: written in {took:.2f}s; row {len(json.dumps(row, ensure_ascii=False))} bytes")
        assert took < 120
    every_version_reads(target, seen)
