"""The version a knowledge history row replaced, read the way an extension reads it."""
from __future__ import annotations


def replaced(root, row):
    """Inline ``old_content``, or the earlier row ``old_content_ref`` names, through the memory reader."""
    if "old_content" in row:
        return row["old_content"]
    from ouroboros.memory_reader import MAX_LENGTH, read_memory

    return read_memory(root, "document", {"address": row["old_content_ref"], "length": MAX_LENGTH})["text"]
