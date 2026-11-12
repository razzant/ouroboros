"""The version a knowledge history row replaced, read the way an extension reads it."""
from __future__ import annotations


def replaced(root, row):
    """Inline ``old_content``, or the earlier row ``old_content_ref`` names, through the memory reader."""
    if "old_content" in row:
        return row["old_content"]
    from ouroboros.memory_reader import MAX_LENGTH, read_memory

    return read_memory(root, "document", {"address": row["old_content_ref"], "length": MAX_LENGTH})["text"]


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
