"""Packed-member read counting shared by retention consumer tests."""

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
