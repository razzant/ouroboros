"""The version a knowledge history row replaced, read the way an extension reads it; rows that store changes."""
from __future__ import annotations


def changed_rows(journal, texts, topic, scope="global"):
    """Successive versions of one note appended to ``journal`` as the writer stores them: each row names the
    previous row as its base and carries its change (``knowledge_rows.encode``, the byte rule deciding whole texts).
    Returns each row's byte offset."""
    import json

    from ouroboros import knowledge_rows

    journal.parent.mkdir(parents=True, exist_ok=True)
    offsets, previous = [], None
    with journal.open("ab") as stream:
        for text in texts:
            offset, old = stream.tell(), "" if previous is None else previous[2]
            row = {"topic": topic, "address": {"scope": scope, "topic": topic}, "publication": "source_capture",
                   "mode": "overwrite", "old_sha256": "" if previous is None else knowledge_rows.sha256_text(old),
                   "new_sha256": knowledge_rows.sha256_text(text),
                   **knowledge_rows.encode(old, text, None if previous is None else previous[:2])}
            stream.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            offsets.append(offset)
            previous = (offset, row, text)
    return offsets


def _read(root, address):
    from ouroboros.memory_reader import MAX_LENGTH, read_memory

    return read_memory(root, "document", {"address": address, "length": MAX_LENGTH})["text"]


def _field(root, row, field):
    """One field of a history row read through the memory reader, at the row's exact line in its shelf's journal."""
    import json
    import pathlib

    scope = row["address"]["scope"]
    shelf = pathlib.Path(root) / ("memory" if scope == "global" else f"projects/{scope[len('project:'):]}")
    raw = (shelf / "knowledge_history.jsonl").read_bytes()
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    offset = raw.index(line)
    assert offset == 0 or raw[offset - 1:offset] == b"\n" and raw.count(line) == 1
    return _read(root, {"kind": "entry", "journal": f"knowledge:{scope}", "offset": offset, "field": [field]})


def replaced(root, row):
    """The version a row replaced, through the memory reader: inline ``old_content``, the earlier row a legacy
    ``old_content_ref`` names, or a change row's ``old_content`` (the text of the version its base holds)."""
    if "old_content" in row:
        return row["old_content"]
    if "old_content_ref" in row:
        return _read(root, row["old_content_ref"])
    return _field(root, row, "old_content")


def published(root, row):
    """The version a row published: its whole ``new_content``, or a change row's, through the memory reader."""
    return row["new_content"] if "new_content" in row else _field(root, row, "new_content")


def applied_blocks(held, reply):
    """``held`` with each whole-line block of a stale-revision reply applied: its ``+`` lines in place of its ``-`` lines.

    Each block's ``- `` lines must be exactly the held version's lines where the block says they begin, and
    each side must hold as many lines as its header says; the result is what the reply says the current note is."""
    import re

    lines = re.findall(r"[^\n]*\n|[^\n]+\Z", held)  # each line with its end; a last line may have none
    blocks, block = [], None
    for line in reply.split("\n"):
        header = re.match(r"@ yours: (\d+) line\(s\) from line (\d+) → current: (\d+) line\(s\) from line (\d+), ", line)
        if header:
            block = {"counts": (int(header[1]), int(header[3])), "first": int(header[2]), "old": [], "new": []}
            blocks.append(block)
        elif block is not None and line[:2] in ("- ", "+ "):
            block["old" if line[0] == "-" else "new"].append(line[2:] + "\n")
        elif block is not None and line == "\\ (no line end)":
            side = block["new"] or block["old"]
            side[-1] = side[-1][:-1]
        else:
            block = None
    for block in reversed(blocks):
        start, count = block["first"] - 1, block["counts"][0]
        assert (len(block["old"]), len(block["new"])) == block["counts"], block
        assert lines[start:start + count] == block["old"], (block, lines[start:start + count])
        lines[start:start + count] = block["new"]
    return "".join(lines)
