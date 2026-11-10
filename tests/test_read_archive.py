"""read_archive: a ZIP read in place through read_file's own admission, nothing ever extracted.

Every archive here is synthetic, built by the test. A member read emits at most its cap whatever the
archive's ratio, names that cannot be trusted stay escaped text, and a member read has its own identity
instead of read_file's read stamp of the whole file.
"""
from __future__ import annotations

import hashlib
import pathlib
import stat
import struct
import tracemalloc
import zipfile

import pytest

from ouroboros import archive_reader
from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def world(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id="archive-reader")
    registry.set_context(ctx)
    return repo, registry, ctx


def _zip(path, members, method=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", method) as archive:
        for name, data in members:
            archive.writestr(name, data)
    return path


def _patch_central_and_local(path, name, *, flag=0, method=None):
    """Set a general-purpose flag bit and/or the method of one member in both its headers."""
    raw = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        local = archive.getinfo(name).header_offset
    raw[local + 6] |= flag
    if method is not None:
        raw[local + 8:local + 10] = struct.pack("<H", method)
    at, encoded = raw.find(b"PK\x01\x02"), name.encode()
    while at >= 0:
        size = struct.unpack("<H", raw[at + 28:at + 30])[0]
        if raw[at + 46:at + 46 + size] == encoded:
            raw[at + 8] |= flag
            if method is not None:
                raw[at + 10:at + 12] = struct.pack("<H", method)
            break
        at = raw.find(b"PK\x01\x02", at + 4)
    path.write_bytes(bytes(raw))


def test_listing_pages_entries_and_names_every_omission(world):
    repo, registry, _ctx = world
    _zip(repo / "many.zip", [(f"dir/file-{index:03d}.txt", f"line {index:03d}\n") for index in range(450)])

    first = registry.execute("read_archive", {"path": "many.zip"})
    assert first.startswith("ZIP archive active_workspace:many.zip: 450 entries (450 files, 0 folders)")
    assert '1. "dir/file-000.txt" (9 bytes' in first and '200. "dir/file-199.txt"' in first
    assert "201." not in first
    assert "- entries 201-450: not listed here; continue with offset=200" in first

    last = registry.execute("read_archive", {"path": "many.zip", "offset": 400})
    assert '401. "dir/file-400.txt"' in last and '450. "dir/file-449.txt"' in last
    assert "- entries 1-400: before this page" in last and "not listed here" not in last
    assert registry.execute("read_archive", {"path": "many.zip", "offset": 450}).startswith("⚠️ TOOL_ARG_ERROR")


def test_a_member_read_is_capped_resumable_and_has_its_own_identity(world):
    repo, registry, ctx = world
    body = "".join(f"{index:07d} a line of text\n" for index in range(15_000)).encode()  # between one and two caps
    path = _zip(repo / "notes.zip", [("readme.txt", "short\n"), ("long.txt", body)])

    whole = registry.execute_result("read_archive", {"path": "notes.zip", "member": "readme.txt"})
    assert "bytes [0:6) as utf-8 text, the whole member." in whole.text and whole.text.endswith("short\n")
    view = whole.meta["archive_member_view"]
    with zipfile.ZipFile(path) as archive:
        crc = archive.getinfo("readme.txt").CRC
    assert view == {"archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "entry": 1,
                    "member_name_sha256": hashlib.sha256(b"readme.txt").hexdigest(), "crc32": f"{crc:08x}",
                    "start_byte": 0, "end_byte": 6, "member_bytes": 6, "complete": True}
    # A member read is never read_file's stamp of the archive: no read view of the .zip is recorded.
    assert getattr(ctx, "last_read_view", None) is None and "read_view" not in whole.meta

    first = registry.execute_result("read_archive", {"path": "notes.zip", "entry": 2})
    cap = archive_reader.TEXT_CAP
    assert f"bytes [0:{cap}) as utf-8 text." in first.text
    assert f"- bytes [{cap}:{len(body)}) as declared: not read here; continue with entry=2, start_byte={cap}" in first.text
    assert first.meta["archive_member_view"]["complete"] is False
    rest = registry.execute_result("read_archive", {"path": "notes.zip", "entry": 2, "start_byte": cap})
    assert f"bytes [{cap}:{len(body)}) as utf-8 text." in rest.text and "- bytes [0:200000): before this read" in rest.text
    joined = first.text.split("\n\n", 1)[1].split("\n\nGaps:")[0] + rest.text.split("\n\n", 1)[1].split("\n\nGaps:")[0]
    assert joined.encode() == body


def test_a_high_ratio_member_decompresses_only_what_it_emits(world, monkeypatch):
    repo, registry, _ctx = world
    path = repo / "bomb.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive, archive.open("bomb.txt", "w") as member:
        for _ in range(64):
            member.write(b"A" * (1 << 20))  # 64 MiB of one letter
    assert path.stat().st_size < 512 * 1024  # a ratio above 100:1
    returned = []
    original = zipfile.ZipExtFile.read

    def counted(self, n=-1):
        data = original(self, n)
        returned.append(len(data))
        return data

    monkeypatch.setattr(zipfile.ZipExtFile, "read", counted)
    tracemalloc.start()
    try:
        text = registry.execute("read_archive", {"path": "bomb.zip", "entry": 1})
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert sum(returned) <= archive_reader.TEXT_CAP + 1  # the cap, plus one byte to know more follows
    assert peak < 24 << 20
    assert "bytes [0:200000)" in text and "continue with entry=1, start_byte=200000" in text

    returned.clear()
    tail = registry.execute("read_archive", {"path": "bomb.zip", "entry": 1, "start_byte": (64 << 20) - 5})
    assert tail.endswith("AAAAA\n\nGaps:\n- bytes [0:67108859): before this read")
    assert sum(returned) <= 64 << 20


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32", "iso2022_jp"])
def test_resuming_text_keeps_the_encoding_state_from_the_skipped_prefix(world, encoding, monkeypatch):
    repo, registry, _ctx = world
    monkeypatch.setattr(archive_reader, "TEXT_CAP", 64)
    body = ("日本語の記録\n" * 32).encode(encoding)
    _zip(repo / "encoded.zip", [("note.txt", body)])
    parts, offset = [], 0
    while True:
        result = registry.execute_result("read_archive", {"path": "encoded.zip", "entry": 1,
                                                          "encoding": encoding, "start_byte": offset})
        assert result.status == "ok", result.text
        view = result.meta["archive_member_view"]
        parts.append(result.text.split("\n\n", 1)[1].split("\n\nGaps:", 1)[0])
        assert view["end_byte"] > offset
        offset = view["end_byte"]
        if offset == len(body):
            break
    assert hashlib.sha256("".join(parts).encode()).digest() == hashlib.sha256(body.decode(encoding).encode()).digest()


def test_a_codec_that_cannot_give_a_page_is_typed_unavailable_never_an_empty_page(world, monkeypatch):
    """UTF-7 holds an unterminated run undecoded: a window that decodes nothing alone, or a skipped run longer than a
    window, is ARCHIVE_MEMBER_UNAVAILABLE, never an empty successful page whose continuation is its own start."""
    repo, registry, _ctx = world
    run = ("日本語" * 40000).encode("utf-7")  # one base64 run past the cap
    _zip(repo / "u7.zip", [("run.txt", run)])
    for start in (0, 50, archive_reader.TEXT_CAP + 1000):  # the window, then the skipped prefix, holds it undecoded
        result = registry.execute_result("read_archive", {"path": "u7.zip", "entry": 1, "encoding": "utf-7",
                                                          "start_byte": start})
        assert result.status == "unavailable" and result.text.startswith("⚠️ ARCHIVE_MEMBER_UNAVAILABLE:"), result.text
        assert f"has no utf-7 page from byte {start}" in result.text and "archive_member_view" not in result.meta
    assert "another encoding may read it" in registry.execute("read_archive", {"path": "u7.zip", "entry": 1,
                                                                               "encoding": "utf-7"})

    monkeypatch.setattr(archive_reader, "TEXT_CAP", 64)  # short runs page through with progress every time
    body = "".join(f"line {index} 日本\n" for index in range(40)).encode("utf-7")
    _zip(repo / "short.zip", [("note.txt", body)])
    parts, offset = [], 0
    while offset < len(body):
        result = registry.execute_result("read_archive", {"path": "short.zip", "entry": 1, "encoding": "utf-7",
                                                          "start_byte": offset})
        assert result.status == "ok", result.text
        assert result.meta["archive_member_view"]["end_byte"] > offset
        parts.append(result.text.split("\n\n", 1)[1].split("\n\nGaps:", 1)[0])
        offset = result.meta["archive_member_view"]["end_byte"]
    assert "".join(parts) == body.decode("utf-7")


def test_damaged_data_empty_names_and_strict_codecs_stay_typed(world):
    """Damaged deflate is ARCHIVE_READ_FAILED, an empty member name lists and reads on Python 3.10 too, and idna, which
    has no lenient mode, is not that text: never an uncaught zlib.error, IndexError or UnicodeError."""
    repo, registry, _ctx = world
    path = _zip(repo / "damaged.zip", [("a.txt", b"hello world " * 1000)])
    raw = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        start = archive.getinfo("a.txt").header_offset + 30 + len("a.txt")
    raw[start:start + 20] = bytes(byte ^ 0xFF for byte in raw[start:start + 20])
    path.write_bytes(bytes(raw))
    damaged = registry.execute_result("read_archive", {"path": "damaged.zip", "entry": 1})
    assert damaged.status == "error" and damaged.text.startswith('⚠️ ARCHIVE_READ_FAILED: active_workspace:damaged.zip')
    assert "(zlib.error: " in damaged.text

    data, crc = b"data", zipfile.crc32(b"data")
    local = struct.pack("<4s5H3L2H", b"PK\x03\x04", 20, 0, 0, 0, 0, crc, 4, 4, 0, 0) + data
    central = struct.pack("<4s6H3L5H2L", b"PK\x01\x02", 20, 20, 0, 0, 0, 0, crc, 4, 4, 0, 0, 0, 0, 0, 0, 0)
    (repo / "nameless.zip").write_bytes(local + central + struct.pack(
        "<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, len(central), len(local), 0))
    assert '1. "" (4 bytes)' in registry.execute("read_archive", {"path": "nameless.zip"})
    assert registry.execute("read_archive", {"path": "nameless.zip", "entry": 1}).endswith("\n\ndata")

    _zip(repo / "idna.zip", [("host.txt", b"xn--" + b"\xff" * 10 + b".com" * 100)])
    for start, fact in ((0, "is not idna text ('"), (8, "has no idna page from byte 8: idna cannot carry its state")):
        refused = registry.execute_result("read_archive", {"path": "idna.zip", "entry": 1, "encoding": "idna",
                                                           "start_byte": start})  # the window's decode, the skip's
        assert refused.status == "unavailable" and refused.text.startswith("⚠️ ARCHIVE_MEMBER_UNAVAILABLE:"), refused.text
        assert fact in refused.text, refused.text
        assert start or "Unsupported error handling" not in refused.text  # the strict refusal is the fact named


def test_a_long_non_latin_name_lists_and_reads_with_a_bounded_identity(world):
    """ToolResult meta holds 8192 ASCII-escaped bytes: the identity names the member by its digest, the text whole."""
    repo, registry, _ctx = world
    name = "отчёт-" * 400 + ".txt"  # 2404 characters, past 14 KiB as escaped JSON
    path = _zip(repo / "long.zip", [(name, "содержимое")])
    assert f'1. "{name}" (20 bytes' in registry.execute("read_archive", {"path": "long.zip"})
    result = registry.execute_result("read_archive", {"path": "long.zip", "entry": 1})
    assert result.status == "ok" and f'entry 1 "{name}"' in result.text and result.text.endswith("содержимое")
    view = result.meta["archive_member_view"]
    assert view["member_name_sha256"] == hashlib.sha256(name.encode()).hexdigest()
    assert view["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert name not in str(view) and str(path) not in str(view)


def test_the_directory_byte_cap_bounds_parsed_entries_whatever_the_count_declares(world, monkeypatch):
    """zipfile parses every header inside the declared directory bytes; at 46 bytes at the least, the cap is the real
    entry bound. Long names are read while the directory fits: no rule on bytes per declared entry."""
    repo, registry, _ctx = world
    assert archive_reader.MAX_DIRECTORY_BYTES == zipfile.sizeCentralDir * archive_reader.MAX_ENTRIES
    monkeypatch.setattr(archive_reader, "MAX_ENTRIES", 5)
    monkeypatch.setattr(archive_reader, "MAX_DIRECTORY_BYTES", zipfile.sizeCentralDir * 5)

    def understated(count):  # empty stored members, each with the smallest central header, declaring one entry
        local = struct.pack("<4s5H3L2H", b"PK\x03\x04", 20, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        central = b"".join(struct.pack("<4s6H3L5H2L", b"PK\x01\x02", 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       index * len(local)) for index in range(count))
        return local * count + central + struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, len(central),
                                                     len(local) * count, 0)

    (repo / "five.zip").write_bytes(understated(5))
    (repo / "six.zip").write_bytes(understated(6))
    assert registry.execute("read_archive", {"path": "five.zip"}).startswith("ZIP archive active_workspace:five.zip: 5 entries")
    refused = registry.execute("read_archive", {"path": "six.zip"})
    assert refused.startswith('⚠️ ARCHIVE_UNAVAILABLE: "six.zip" declares 1 entry in a 276 bytes directory')
    assert "at most 5 entries in at most 230 bytes of directory" in refused
    _zip(repo / "named.zip", [("n" * 60, "a"), ("m" * 60, "b")], method=zipfile.ZIP_STORED)  # 212 bytes, 2 entries
    assert registry.execute("read_archive", {"path": "named.zip"}).startswith("ZIP archive active_workspace:named.zip: 2 entries")


def test_a_skill_assets_archive_reads_through_the_selectors_its_hint_names(world, tmp_path):
    """read_archive takes read_file's skill_payload selectors, and read_file's refusal names them in its call."""
    import json
    import re

    _repo, registry, ctx = world
    skill = pathlib.Path(ctx.drive_root) / "skills" / "external" / "atlas"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: atlas\ndescription: x\nversion: '1.0'\ntype: instruction\n---\nbody\n")
    _zip(skill / "assets.zip", [("inside.txt", "hello")])
    hint = registry.execute("read_file", {"root": "skill_payload", "bucket": "external", "skill_name": "atlas",
                                          "path": "assets.zip"})
    call = re.search(r"read_archive\((.*?)\) lists it", hint)
    assert call and 'bucket="external", skill_name="atlas"' in call[1], hint
    arguments = {key: json.loads(value) for key, value in re.findall(r'(\w+)=("[^"]*")', call[1])}
    assert registry.execute("read_archive", arguments).startswith("ZIP archive skill_payload:assets.zip: 1 entry")
    assert registry.execute("read_archive", {**arguments, "entry": 1}).endswith("\n\nhello")
    assert "skill_name" in registry.execute("read_archive", {"root": "skill_payload", "path": "assets.zip"})


def test_a_binary_codec_is_not_accepted_as_a_text_encoding(world):
    repo, registry, _ctx = world
    _zip(repo / "encoded.zip", [("note.txt", "aGVsbG8=")])
    result = registry.execute("read_archive", {"path": "encoded.zip", "entry": 1, "encoding": "base64_codec"})
    assert result.startswith("⚠️ TOOL_ARG_ERROR:") and "text encoding" in result


def test_names_that_point_outside_or_link_elsewhere_stay_text_and_nothing_is_written(world, tmp_path):
    repo, registry, _ctx = world
    path = repo / "tricky.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../../escape.txt", "outside")
        archive.writestr("/abs/root.txt", "absolute")
        drive = zipfile.ZipInfo("C:\\windows\\drive.txt")
        # ZipInfo normalizes Windows separators; preserve this hostile stored name.
        drive.filename = drive.orig_filename
        archive.writestr(drive, "drive")
        link = zipfile.ZipInfo("link-to-secret")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "/etc/passwd")
        archive.writestr("invoice\u202etxt.exe", "bidi")
        archive.writestr("two\nlines.txt", "newline")
        archive.writestr("folder/", "")
    raw = path.read_bytes()
    # Both the local header and central directory must contain the exact name.
    assert raw.count(b"C:\\windows\\drive.txt") == 2
    assert b"C:/windows/drive.txt" not in raw

    def tree():
        return sorted(str(rel) for rel in (item.relative_to(tmp_path) for item in tmp_path.rglob("*")) if rel.parts[0] != "data")

    before = tree()

    listing = registry.execute("read_archive", {"path": "tricky.zip"})
    outside = "an absolute or '..' path: only a name here, nothing is extracted"
    for line in ('1. "../../escape.txt"', '2. "/abs/root.txt"', '3. "C:\\\\windows\\\\drive.txt"'):
        assert f"{line} (" in listing and outside in listing.split(line, 1)[1].splitlines()[0]
    assert '4. "link-to-secret" (11 bytes); a symbolic link: its stored target is its text, nothing is followed' in listing
    assert '5. "invoice\\u202etxt.exe"' in listing and "\u202e" not in listing
    assert '6. "two\\u000alines.txt"' in listing and "two\nlines" not in listing
    assert '7. "folder/" (folder)' in listing
    assert registry.execute("read_archive", {"path": "tricky.zip", "entry": 1}).endswith("\n\noutside")
    drive_read = registry.execute_result("read_archive", {"path": "tricky.zip", "member": "C:\\windows\\drive.txt"})
    assert drive_read.text.endswith("\n\ndrive")
    assert drive_read.meta["archive_member_view"]["entry"] == 3
    assert drive_read.meta["archive_member_view"]["member_name_sha256"] == hashlib.sha256(b"C:\\windows\\drive.txt").hexdigest()
    assert registry.execute("read_archive", {"path": "tricky.zip", "entry": 4}).endswith("\n\n/etc/passwd")
    assert registry.execute("read_archive", {"path": "tricky.zip", "member": "invoice\u202etxt.exe"}).endswith("bidi")
    assert "is a folder" in registry.execute("read_archive", {"path": "tricky.zip", "entry": 7})
    assert tree() == before  # nothing was extracted anywhere
    assert not list(tmp_path.rglob("escape.txt")) and not list(tmp_path.rglob("root.txt"))


def test_legacy_names_are_read_as_their_encoding_and_say_so(world):
    repo, registry, _ctx = world
    path = _zip(repo / "legacy.zip", [("AAAAA.txt", "cp866"), ("BBBBBBBBBB.txt", "utf8"), ("plain.txt", "ascii")])
    raw = path.read_bytes()
    raw = raw.replace(b"AAAAA.txt", "отчёт.txt".encode("cp866")).replace(b"BBBBBBBBBB.txt", "отчёт.txt".encode())
    path.write_bytes(raw)

    listing = registry.execute("read_archive", {"path": "legacy.zip"})
    assert '1. "отчёт.txt" (5 bytes' in listing and "name read as cp866 (no UTF-8 flag)" in listing
    assert '2. "отчёт.txt" (4 bytes' in listing and "name read as UTF-8 bytes stored without the UTF-8 flag" in listing
    assert '3. "plain.txt" (5 bytes' in listing and "name read as" not in listing.split('3. "plain.txt"', 1)[1].splitlines()[0]
    duplicate = registry.execute("read_archive", {"path": "legacy.zip", "member": "отчёт.txt"})
    assert duplicate.startswith("⚠️ TOOL_ARG_ERROR: 2 entries are named") and "(entries 1, 2)" in duplicate
    assert registry.execute("read_archive", {"path": "legacy.zip", "entry": 1}).endswith("cp866")


def test_the_end_record_is_read_by_its_fields(tmp_path):
    """Entries and directory size come from the end record's own fields, never its offset."""
    path = _zip(tmp_path / "three.zip", [("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma")])
    raw = path.read_bytes()
    first, end = raw.find(b"PK\x01\x02"), raw.rfind(b"PK\x05\x06")
    assert archive_reader.declared_directory(path) == (raw.count(b"PK\x01\x02"), end - first) == (3, end - first)


def test_an_archive_whose_directory_starts_past_64_mib_lists(world):
    repo, registry, _ctx = world
    path = repo / "big.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name in ("part-1.bin", "part-2.bin"):
            with archive.open(name, "w") as member:
                for _ in range(33):
                    member.write(bytes(1 << 20))
    raw_end = path.read_bytes()[-22:]
    assert struct.unpack("<4s4H2LH", raw_end)[6] > archive_reader.MAX_DIRECTORY_BYTES  # the directory's offset
    listing = registry.execute("read_archive", {"path": "big.zip"})
    assert listing.startswith("ZIP archive active_workspace:big.zip: 2 entries (2 files, 0 folders), 66.0 MiB")
    assert '1. "part-1.bin" (33.0 MiB)' in listing and '2. "part-2.bin" (33.0 MiB)' in listing


def test_an_oversized_classic_directory_is_refused_before_it_is_parsed(world):
    repo, registry, _ctx = world
    (repo / "wide.zip").write_bytes(struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, 100 << 20, 0, 0))
    refused = registry.execute("read_archive", {"path": "wide.zip"})
    assert refused.startswith('⚠️ ARCHIVE_UNAVAILABLE: "wide.zip" declares 1 entry in a 100.0 MiB directory')


def test_encrypted_unsupported_and_nested_members_are_named_not_read(world):
    repo, registry, _ctx = world
    inner = repo / "inner.zip"
    _zip(inner, [("deep.txt", "deep")])
    path = _zip(repo / "outer.zip", [("secret.txt", "classified"), ("inner.zip", inner.read_bytes()),
                                     ("ppmd.txt", "odd method")])
    _patch_central_and_local(path, "secret.txt", flag=0x1)
    _patch_central_and_local(path, "ppmd.txt", method=98)
    with zipfile.ZipFile(path, "a", zipfile.ZIP_BZIP2) as archive:
        archive.writestr("packed.txt", "bzip2 text")

    listing = registry.execute("read_archive", {"path": "outer.zip"})
    assert "secret.txt" in listing and "encrypted: it needs a password, not readable here" in listing
    assert "an archive inside: listed, not opened" in listing.split('2. "inner.zip"', 1)[1].splitlines()[0]
    assert "compressed with PPMd, which this reader does not decode" in listing
    assert "compressed with bzip2, which this reader does not decode" in listing
    assert "deep.txt" not in listing  # the inner archive is never opened
    secret = registry.execute("read_archive", {"path": "outer.zip", "entry": 1})
    assert secret.startswith("⚠️ ARCHIVE_MEMBER_UNAVAILABLE:") and "encrypted" in secret and "classified" not in secret
    nested = registry.execute("read_archive", {"path": "outer.zip", "entry": 2})
    assert nested.startswith("⚠️ ARCHIVE_MEMBER_UNAVAILABLE:") and "binary (a ZIP archive)" in nested
    assert "An archive inside is listed, never opened." in nested
    for entry in (3, 4):
        refused = registry.execute("read_archive", {"path": "outer.zip", "entry": entry})
        assert refused.startswith("⚠️ ARCHIVE_MEMBER_UNAVAILABLE:") and "does not decode" in refused


@pytest.mark.parametrize("comment_size", [0, 65535])
def test_zip64_bounds_use_the_directory_that_zipfile_will_read(world, comment_size):
    """ZIP64 may exist with nonsaturated classic fields, also outside the classic comment scan."""
    repo, registry, _ctx = world
    path = _zip(repo / "zip64.zip", [("note.txt", "hello")])
    raw = path.read_bytes()
    end = list(struct.unpack("<4s4H2LH", raw[-22:]))
    directory_size, directory_start = end[5:7]
    zip64_start = len(raw) - 22
    record = struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                         300_000, 300_000, directory_size, directory_start)
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, zip64_start, 1)
    end[7] = comment_size  # keep the ordinary entry counts, size and offset valid
    path.write_bytes(raw[:-22] + record + locator + struct.pack("<4s4H2LH", *end) + b"c" * comment_size)
    assert archive_reader.declared_directory(path) == (300_000, directory_size)
    assert "declares 300000 entries" in registry.execute("read_archive", {"path": "zip64.zip"})


def test_read_file_names_the_archive_reader_and_read_archive_keeps_read_file_admission(world, tmp_path):
    repo, registry, _ctx = world
    _zip(repo / "bundle.zip", [("a.txt", "a")])
    (repo / "note.pdf").write_bytes(b"%PDF-1.4\n\xff\xfe binary")
    refused = registry.execute("read_file", {"path": "bundle.zip"})
    assert refused.startswith("⚠️ READ_FILE_ERROR: active_workspace:bundle.zip is not UTF-8 text (it is a ZIP archive); ")
    assert 'read_archive(root="active_workspace", path="bundle.zip")' in refused
    assert 'enable_tools(tools="read_archive")' in refused
    assert "extract_pdf_text(path=" in registry.execute("read_file", {"path": "note.pdf"})

    outside = tmp_path / "outside.zip"
    _zip(outside, [("b.txt", "b")])
    blocked = registry.execute("read_archive", {"path": str(outside)})
    assert blocked.startswith("⚠️") and "ZIP archive" not in blocked and "b.txt" not in blocked
    assert registry.execute("read_archive", {"path": "missing.zip"}).startswith("⚠️ FILE_NOT_FOUND")
    (repo / "plain.txt").write_text("not an archive\n")
    assert registry.execute("read_archive", {"path": "plain.txt"}).startswith(
        '⚠️ ARCHIVE_UNAVAILABLE: "plain.txt" is not a ZIP archive.')
    assert registry.execute("read_archive", {"path": "note.pdf"}).startswith(
        '⚠️ ARCHIVE_UNAVAILABLE: "note.pdf" is not a ZIP archive (it looks like a PDF)')


def test_read_archive_is_a_read_only_on_demand_tool_wherever_tools_are_catalogued(world):
    from ouroboros import tool_capabilities as caps
    from ouroboros.consciousness_authority import disabled_tools_for

    _repo, registry, _ctx = world
    entry = registry._entries["read_archive"]
    assert entry.safety_policy == "skip" and not entry.mutates_worktree and not entry.is_code_tool
    assert "read_archive" not in caps.CORE_TOOL_NAMES  # loaded on demand, as read_file's refusal says
    for group in (caps.LOCAL_READONLY_SUBAGENT_TOOL_NAMES, caps.ACTING_SUBAGENT_TOOL_NAMES,
                  caps.READ_ONLY_PARALLEL_TOOLS, caps.PURE_READ_TOOLS):
        assert "read_archive" in group
    assert "read_archive" not in caps.OBSERVE_WORLD_MUTATION_TOOLS
    assert "read_archive" not in disabled_tools_for("observe")


def test_read_archive_reaches_every_actor_that_extract_pdf_text_reaches(tmp_path, monkeypatch):
    """From the compiled request lists and dispatch, not the constants: Main (Max; Nano loads it on demand), a
    Presence place turn at its level and at Observe (withheld names bind at dispatch only), and both children."""
    from dataclasses import replace

    from ouroboros import places as p
    from ouroboros.consciousness_authority import disabled_tools_for
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_runner import PresenceTurnEvent, _build_task
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools import tool_discovery

    system, work, data = (tmp_path / name for name in ("system", "work", "data"))
    for folder in (system, work, data):
        folder.mkdir()
    for folder in (system, work):
        _zip(folder / "a.zip", [("x.txt", "x")])
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    monkeypatch.setattr(tool_discovery, "_registry", registry)

    def compiled(ctx, mode="max", schema_names=None):
        registry.set_context(ctx)
        return {schema["function"]["name"] for schema in initial_tool_schemas(registry, mode, schema_names=schema_names)}

    main = ToolContext(repo_dir=system, drive_root=data, task_id="main", workspace_root=work, workspace_mode="external")
    assert {"read_archive", "extract_pdf_text"} <= compiled(main)
    assert "read_archive" not in compiled(main, "nano")  # not a round-one core tool
    registry.set_context(main)
    assert "read_archive" in registry.execute("enable_tools", {"tools": "read_archive"})
    assert "read_archive" in compiled(main, "nano", ["read_archive"])

    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                        lambda _root: {"status": "available", "remaining_usd": 2, "settled_usd": 1})
    monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "")
    p.connect_place(data, connector="bridge", provider="chat", account_id="a", conversation_id="r", wakes=["any_message"])
    admission = replace(admit_presence_turn(drive_root=data, authenticated_transport_skill="bridge", provider="chat",
                                            account_id="a", conversation_id="r"),
                        allowance={"status": "available", "remaining_usd": 2}, workspace_root=str(work))
    event = PresenceTurnEvent("e1", "chat", "a", "r", "", "chat:a:r", {"platform_actor_id": "person"}, {}, {}, "hi",
                              facts=("any_message",))
    task = _build_task(admission, event, drive_root=data, staged_files=())
    observe = {**task["metadata"], "consciousness_autonomy": "observe", "disabled_tools": disabled_tools_for("observe"),
               "runtime_mode_cap": "light"}
    actors = {
        "place turn": ToolContext(repo_dir=system, drive_root=data, task_id=task["id"], task_metadata=task["metadata"],
                                  task_contract=task.get("task_contract") or {}, workspace_root=work,
                                  workspace_mode="external"),
        "place turn at Observe": ToolContext(repo_dir=system, drive_root=data, task_id="observe", task_metadata=observe,
                                             workspace_root=work, workspace_mode="external"),
        "acting child": ToolContext(repo_dir=system, drive_root=data, task_id="acting", workspace_root=work,
                                    workspace_mode="external", task_constraint=TaskConstraint(
                                        mode="acting_subagent", surface="external_workspace", write_root=str(work),
                                        allow_enable=False)),
    }
    for label, ctx in actors.items():
        assert {"read_archive", "extract_pdf_text"} <= compiled(ctx), label
        assert registry.execute("read_archive", {"path": "a.zip"}).startswith(
            "ZIP archive active_workspace:a.zip: 1 entry (1 file, 0 folders)"), label
    registry.set_context(actors["place turn at Observe"])
    assert "withholds 'run_command'" in registry.execute("run_command", {"cmd": ["true"]})  # the level's gate is live
    readonly = ToolContext(repo_dir=system, drive_root=data, task_id="readonly",
                           task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    assert {"read_archive", "extract_pdf_text"} <= compiled(readonly)
    assert registry.execute("read_archive", {"root": "system_repo", "path": "a.zip"}).startswith(
        "ZIP archive system_repo:a.zip")
