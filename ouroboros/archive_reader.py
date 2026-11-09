"""Read a ZIP archive in place: a paged listing and one member's text, nothing extracted.

``read_archive`` (``tools/media.py``) admits the file through read_file's own roots and checks, then
calls here. The archive is opened by path (``zipfile.ZipFile``), never loaded whole, and nothing is
written anywhere: a member named by an absolute or ``..`` path, or stored as a symlink, names nothing
outside the archive, it is only text here. A nested archive is listed, never opened; an encrypted
member or a compression this reader does not decode is named, never guessed. Only stored and deflate
members are decoded: ``zipfile`` bounds the output of those two per read, while its bzip2, LZMA and
zstd paths decompress a whole compressed chunk at once. A member read decompresses only up to the bytes
it emits past its offset: that cap, not a compression ratio, is the bomb defence. A read with more to
give always advances: a codec that holds a whole window, or more than one window of the skipped part,
undecoded (UTF-7's unterminated run) or cannot decode leniently (idna) makes the read typed unavailable. The
central directory's declared size is checked before it is parsed: zipfile parses every header inside
those bytes whatever count the end record declares, and each takes at least 46 bytes, so
``MAX_DIRECTORY_BYTES`` bounds a listing at ``MAX_ENTRIES`` entries.

Names are shown quoted with control, format (bidi) and separator characters escaped. A name stored
without the UTF-8 flag is read as UTF-8 when its bytes are valid UTF-8, else as cp866 when that reads
as Cyrillic, else as cp437 (the ZIP default), and the listing says which.

A member read carries its own bounded identity (``archive_member_view``: the archive's SHA-256, the
entry, the SHA-256 of its name as listed, its CRC-32, the byte range; the text shows the name whole),
never read_file's ``read_view``: reading one member is never a read of the whole archive.
"""
from __future__ import annotations

import codecs
import hashlib
import pathlib
import re
import stat
import struct
import unicodedata
import zipfile
import zlib
from typing import Any

LIST_PAGE = 200  # entries per listing page
TEXT_CAP = 200_000  # decompressed bytes of one member emitted per call
MAX_START_BYTE = 1 << 30  # how far into a member a read may begin
MAX_ENTRIES = 250_000  # entries a listing parses at most
MAX_DIRECTORY_BYTES = zipfile.sizeCentralDir * MAX_ENTRIES  # 46-byte headers at the least: no more entries fit
_CHUNK = 1 << 16
_ESCAPED = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})
_NESTED = (".zip", ".jar", ".war", ".apk", ".aar", ".whl", ".nupkg", ".7z", ".rar", ".tar", ".tgz", ".tbz2",
           ".txz", ".gz", ".bz2", ".xz", ".zst", ".lz4", ".cab", ".iso", ".dmg")
_METHODS = {0: "stored", 1: "shrink", 6: "implode", 8: "deflate", 9: "deflate64", 12: "bzip2", 14: "LZMA",
            93: "zstandard", 95: "xz", 96: "jpeg", 97: "wavpack", 98: "PPMd", 99: "AES encryption"}
_DECODED = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_MAGIC = ((b"PK\x03\x04", "a ZIP archive"), (b"PK\x05\x06", "a ZIP archive"), (b"%PDF", "a PDF"),
          (b"\x89PNG", "a PNG image"), (b"\xff\xd8\xff", "a JPEG image"), (b"GIF8", "a GIF image"),
          (b"\x1f\x8b", "gzip data"), (b"7z\xbc\xaf\x27\x1c", "a 7z archive"), (b"Rar!", "a RAR archive"),
          (b"BZh", "bzip2 data"), (b"\xfd7zXZ\x00", "xz data"))
_SHA_CACHE: dict[tuple, str] = {}


def display_name(text: Any) -> str:
    """``text`` quoted, with every control, format (bidi), private-use, unassigned or line/paragraph separator
    character escaped, so an untrusted name can neither reorder nor break the line it is shown in."""
    out = []
    for char in str(text):
        if char in '"\\':
            out.append("\\" + char)
        elif unicodedata.category(char) in _ESCAPED:
            code = ord(char)
            out.append(f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def size_words(size: Any) -> str:
    """Bytes in binary units, the way a connector states its size limit (20 MiB)."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return "size unknown"
    if size < 1024:
        return f"{size} byte{'' if size == 1 else 's'}"
    value, unit = float(size), "B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024
        if value < 1024:
            break
    return f"{value:.1f} {unit}"


def _count(number: int, one: str, many: str) -> str:
    return f"{number} {one if number == 1 else many}"


def file_kind(head: bytes) -> str:
    """What a file's first bytes say it is ("a ZIP archive", "a PDF", ...), or ``""``."""
    return next((kind for magic, kind in _MAGIC if head.startswith(magic)), "")


def declared_directory(path: pathlib.Path) -> tuple[int, int] | None:
    """Bounds from the same bounded end-record parser ZipFile uses before reading its directory.

    Re-parsing ZIP64 separately can disagree on nonsaturated classic fields, maximum comments or prepended
    bytes. This stdlib-private parser reads only end records (not the directory); the real-ZIP tests cover
    its contract on supported Python versions.
    """
    with path.open("rb") as handle:
        record = zipfile._EndRecData(handle)
    return (record[zipfile._ECD_ENTRIES_TOTAL], record[zipfile._ECD_SIZE]) if record else None


def _opened(path: pathlib.Path) -> zipfile.ZipFile | str:
    """The open archive, or the typed reason it is not read."""
    try:
        declared = declared_directory(path)
        if declared is None:
            with path.open("rb") as handle:
                kind = file_kind(handle.read(8))
            return (f"⚠️ ARCHIVE_UNAVAILABLE: {display_name(path.name)} is not a ZIP archive"
                    + (f" (it looks like {kind}); only ZIP archives are read for now." if kind else "."))
        entries, directory = declared
        if entries > MAX_ENTRIES or directory > MAX_DIRECTORY_BYTES:
            return (f"⚠️ ARCHIVE_UNAVAILABLE: {display_name(path.name)} declares {entries} "
                    f"entr{'y' if entries == 1 else 'ies'} in a {size_words(directory)} directory; this reader reads "
                    f"at most {MAX_ENTRIES} entries in at most {size_words(MAX_DIRECTORY_BYTES)} of directory, the "
                    f"room {MAX_ENTRIES} entries take at their smallest ({zipfile.sizeCentralDir} bytes each).")
        return zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, ValueError, struct.error, EOFError, NotImplementedError) as exc:
        return (f"⚠️ ARCHIVE_UNAVAILABLE: {display_name(path.name)} could not be opened as a ZIP archive "
                f"({type(exc).__name__}: {exc}).")


def _is_dir(info: zipfile.ZipInfo) -> bool:
    """``ZipInfo.is_dir`` as Python 3.11 defines it: 3.10's indexes the name and raises on an empty one."""
    return info.filename.endswith("/")


def _member_name(info: zipfile.ZipInfo) -> tuple[str, str]:
    """``(name, how it was decoded)``: the stored name, read again when it carries no UTF-8 flag."""
    name = info.orig_filename
    if info.flag_bits & 0x800:
        return name, ""
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name, ""
    if raw.isascii():
        return name, ""
    try:
        return raw.decode("utf-8"), "UTF-8 bytes stored without the UTF-8 flag"
    except UnicodeDecodeError:
        pass
    cyrillic = raw.decode("cp866")
    wide = [char for char in cyrillic if not char.isascii()]
    if sum("Ѐ" <= char <= "ӿ" for char in wide) >= 0.8 * len(wide):
        return cyrillic, "cp866 (no UTF-8 flag)"
    return name, "cp437 (no UTF-8 flag)"


def _unreadable(info: zipfile.ZipInfo) -> str:
    """Why this member's bytes are not decoded here, else ``""``."""
    if info.flag_bits & 0x1:
        return "encrypted: it needs a password, not readable here"
    if info.compress_type not in _DECODED:
        method = _METHODS.get(info.compress_type, f"method {info.compress_type}")
        return f"compressed with {method}, which this reader does not decode"
    return ""


def _facts(info: zipfile.ZipInfo) -> list[str]:
    """What this entry is beside its size; empty for an ordinary readable file."""
    name, decoded = _member_name(info)
    facts = [why] if (why := _unreadable(info)) else []
    if info.create_system == 3 and stat.S_ISLNK(info.external_attr >> 16):
        facts.append("a symbolic link: its stored target is its text, nothing is followed")
    if not _is_dir(info) and name.lower().endswith(_NESTED):
        facts.append("an archive inside: listed, not opened")
    parts = name.replace("\\", "/").split("/")
    if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name) or ".." in parts:
        facts.append("an absolute or '..' path: only a name here, nothing is extracted")
    if decoded:
        facts.append(f"name read as {decoded}")
    return facts


def listing(path: pathlib.Path, label: str, *, offset: int = 0) -> str:
    """One page of the archive's entries, every omission named."""
    opened = _opened(path)
    if isinstance(opened, str):
        return opened
    with opened as archive:
        infos = archive.infolist()
    total = len(infos)
    files = [info for info in infos if not _is_dir(info)]
    if offset and offset >= total:
        return f"⚠️ TOOL_ARG_ERROR: offset={offset} is past the last of {total} entries."
    page = infos[offset:offset + LIST_PAGE]
    lines = [f"ZIP archive {label}: {_count(total, 'entry', 'entries')} ({_count(len(files), 'file', 'files')}, "
             f"{_count(total - len(files), 'folder', 'folders')}), "
             f"{size_words(sum(info.file_size for info in files))} uncompressed as declared. Nothing is "
             "extracted; names below are only text."]
    for number, info in enumerate(page, start=offset + 1):
        what = "folder" if _is_dir(info) else size_words(info.file_size) + (
            f", {size_words(info.compress_size)} compressed" if info.compress_type != zipfile.ZIP_STORED else "")
        lines.append(f"{number}. {display_name(_member_name(info)[0])} ({what})"
                     + "".join(f"; {fact}" for fact in _facts(info)))
    gaps = ([f"entries 1-{offset}: before this page"] if offset else []) + (
        [f"entries {offset + len(page) + 1}-{total}: not listed here; continue with offset={offset + len(page)}"]
        if offset + len(page) < total else [])
    if gaps:
        lines += ["", "Gaps:", *(f"- {gap}" for gap in gaps)]
    return "\n".join([*lines, "", "Read one member's text with entry=<number> (or member=<its name as listed>)."])


def _sha256(path: pathlib.Path) -> str:
    witness = path.stat()
    key = (str(path.resolve()), witness.st_size, witness.st_mtime_ns, witness.st_ino)
    if key not in _SHA_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if len(_SHA_CACHE) >= 32:
            _SHA_CACHE.clear()
        _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]


def _select(infos: list[zipfile.ZipInfo], entry: int, member: str) -> tuple[int, zipfile.ZipInfo] | str:
    if entry:
        if not 1 <= entry <= len(infos):
            return f"⚠️ TOOL_ARG_ERROR: entry={entry} is not between 1 and {len(infos)}."
        return entry, infos[entry - 1]
    found = [(number, info) for number, info in enumerate(infos, start=1)
             if member in {_member_name(info)[0], info.orig_filename, info.filename}]
    if not found:
        return f"⚠️ TOOL_ARG_ERROR: no entry is named {display_name(member)}; list the archive for its entry numbers."
    if len(found) > 1:
        return (f"⚠️ TOOL_ARG_ERROR: {len(found)} entries are named {display_name(member)} (entries "
                f"{', '.join(str(number) for number, _info in found)}); read one by its entry number.")
    return found[0]


class _NotPaged(Exception):
    """Why this codec cannot give the member's text a page from here: the read is typed unavailable."""


def _read_window(archive: zipfile.ZipFile, info: zipfile.ZipInfo, start_byte: int, codec):
    """Skip in bounded steps, retaining the text decoder's state (BOM and shift sequences), then return the
    window and that state. Reading a later window neither forgets its encoding nor retains the skipped text:
    a decoder holding more than a window of it undecoded, or refusing lenient decoding, raises ``_NotPaged``."""
    with archive.open(info) as stream:
        skipped, prefix = 0, codec.incrementaldecoder("replace")
        while skipped < start_byte:
            block = stream.read(min(_CHUNK, start_byte - skipped))
            if not block:
                break
            try:
                prefix.decode(block, final=False)
            except UnicodeError as exc:  # idna decodes strictly or not at all
                raise _NotPaged(f"{codec.name} cannot carry its state across the bytes before it ({exc})") from exc
            skipped += len(block)
            if len(prefix.getstate()[0]) > TEXT_CAP:
                raise _NotPaged(f"more than {size_words(TEXT_CAP)} before it stay undecoded in one {codec.name} run")
        data = stream.read(TEXT_CAP)
        return skipped, data, len(data) == TEXT_CAP and bool(stream.read(1)), prefix.getstate()


def member_text(path: pathlib.Path, label: str, *, entry: int = 0, member: str = "", start_byte: int = 0,
                encoding: str = "utf-8") -> tuple[str, dict[str, Any] | None]:
    """``(text, identity)`` of one member's decoded bytes from ``start_byte``, at most ``TEXT_CAP`` of them;
    ``identity`` is None when nothing was read."""
    if type(start_byte) is not int or not 0 <= start_byte <= MAX_START_BYTE:
        return f"⚠️ TOOL_ARG_ERROR: start_byte must be an integer from 0 to {MAX_START_BYTE}.", None
    try:
        codec = codecs.lookup(str(encoding or "utf-8"))
    except LookupError:
        return f"⚠️ TOOL_ARG_ERROR: unknown encoding {display_name(encoding)}.", None
    if not codec._is_text_encoding or codec.incrementaldecoder is None:
        return f"⚠️ TOOL_ARG_ERROR: {display_name(encoding)} is not an incremental text encoding.", None
    opened = _opened(path)
    if isinstance(opened, str):
        return opened, None
    with opened as archive:
        chosen = _select(archive.infolist(), entry, member)
        if isinstance(chosen, str):
            return chosen, None
        number, info = chosen
        name = _member_name(info)[0]
        where = f"{label} › entry {number} {display_name(name)}"
        if _is_dir(info):
            return f"⚠️ ARCHIVE_MEMBER_UNAVAILABLE: {where} is a folder; it has no text.", None
        if why := _unreadable(info):
            return f"⚠️ ARCHIVE_MEMBER_UNAVAILABLE: {where} is {why}.", None
        try:
            skipped, data, more, decoder_state = _read_window(archive, info, start_byte, codec)
        except _NotPaged as exc:
            return f"⚠️ ARCHIVE_MEMBER_UNAVAILABLE: {where} has no {codec.name} page from byte {start_byte}: {exc}.", None
        except (zipfile.BadZipFile, zlib.error, OSError, EOFError, NotImplementedError, RuntimeError, ValueError) as exc:
            kind = "zlib.error" if isinstance(exc, zlib.error) else type(exc).__name__  # damaged deflate data
            return f"⚠️ ARCHIVE_READ_FAILED: {where} could not be read ({kind}: {exc}).", None
    if skipped < start_byte:
        return f"⚠️ TOOL_ARG_ERROR: start_byte={start_byte} is past the end of {where} ({skipped} bytes).", None
    if not codec.name.startswith(("utf-16", "utf-32")) and b"\x00" in data:
        kind = file_kind(data[:8]) if not start_byte else ""
        inside = " An archive inside is listed, never opened." if kind.endswith("archive") else ""
        return (f"⚠️ ARCHIVE_MEMBER_UNAVAILABLE: {where} is binary ({kind or 'it holds NUL bytes'}), not "
                f"{codec.name} text.{inside}"), None
    replaced, decoder = "", codec.incrementaldecoder("strict")
    decoder.setstate(decoder_state)
    try:
        text = decoder.decode(data, final=not more)
    except UnicodeDecodeError as strict:
        decoder = codec.incrementaldecoder("replace")
        decoder.setstate(decoder_state)
        try:
            text = decoder.decode(data, final=not more)
        except UnicodeError:  # idna decodes strictly or not at all: the strict refusal is the fact
            return f"⚠️ ARCHIVE_MEMBER_UNAVAILABLE: {where} is not {codec.name} text ({strict}).", None
        replaced = (f" Bytes that are not valid {codec.name} show as U+FFFD; another encoding (for example "
                    "encoding=\"cp1251\") may read them.")
    except UnicodeError as exc:
        return f"⚠️ ARCHIVE_MEMBER_UNAVAILABLE: {where} is not {codec.name} text ({exc}).", None
    end = start_byte + len(data) - (len(decoder.getstate()[0]) if more else 0)
    if more and end <= start_byte:  # nothing in the window decodes alone: an empty page would continue from here forever
        return (f"⚠️ ARCHIVE_MEMBER_UNAVAILABLE: {where} has no {codec.name} page from byte {start_byte}: no "
                f"character ends within the {size_words(TEXT_CAP)} read; another encoding may read it."), None
    complete = start_byte == 0 and not more
    identity = {"archive_sha256": _sha256(path), "entry": number,
                "member_name_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(), "crc32": f"{info.CRC:08x}",
                "start_byte": start_byte, "end_byte": end, "member_bytes": info.file_size, "complete": complete}
    head = (f"{where} ({size_words(info.file_size)}, CRC-32 {info.CRC:08x}): bytes [{start_byte}:{end}) as "
            f"{codec.name} text{', the whole member' if complete else ''}.{replaced}")
    gaps = ([f"bytes [0:{start_byte}): before this read"] if start_byte else []) + (
        [f"bytes [{end}:{info.file_size}) as declared: not read here; continue with entry={number}, "
         f"start_byte={end}"] if more else [])
    tail = ("\n\nGaps:\n" + "\n".join(f"- {gap}" for gap in gaps)) if gaps else ""
    return f"{head}\n\n{text}{tail}", identity
