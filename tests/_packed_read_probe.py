"""Packed-member read counting shared by retention consumer tests."""

import os
from collections import Counter


def count_packed_reads(monkeypatch) -> Counter:
    """Count reads by (pack name, member offset), independently of decoder buffering."""
    from ouroboros import evidence_packs as packs

    reads, read_at = Counter(), packs._PackReader.read_at

    def counted(reader, offset, size, sink):
        reads[(reader.path.name, int(offset))] += 1
        return read_at(reader, offset, size, sink)

    packs._READERS.clear()
    monkeypatch.setattr(packs._PackReader, "read_at", counted)
    return reads


def release_pack_readers(tree) -> int:
    """Release this process's readers beneath a test-created pack root; return the number closed.

    These fixtures explicitly pack a child drive, unlike the maintenance producer's canonical-root pass.
    Before deleting that fixture on Windows, close the handles its own reads opened; do not reopen on every
    member read or add this to the helper that models the product's ordinary SQLite-only drive release."""
    from ouroboros import evidence_packs as packs

    prefix = os.path.join(os.path.normcase(os.path.realpath(tree)), "")
    with packs._READERS_LOCK:
        held = [packs._READERS.pop(key) for key in list(packs._READERS)
                if os.path.normcase(os.path.realpath(key)).startswith(prefix)]
    closed = 0
    for reader in held:
        with reader._lock:
            if reader._stream is not None:
                reader._stream.close()
                reader._stream = None
                closed += 1
    return closed


def count_decoding(monkeypatch) -> dict:
    """Count real XZ decoder starts and the bytes they produce, including what seeks decode and discard."""
    import lzma

    counts, real = {"starts": 0, "decoded": 0}, lzma.LZMADecompressor

    class Counting:
        def __init__(self, *args, **kwargs):
            counts["starts"] += 1
            self._decoder = real(*args, **kwargs)

        def decompress(self, *args, **kwargs):
            output = self._decoder.decompress(*args, **kwargs)
            counts["decoded"] += len(output)
            return output

        def __getattr__(self, name):
            return getattr(self._decoder, name)

    monkeypatch.setattr(lzma, "LZMADecompressor", Counting)
    return counts
