"""Helpers for tests whose case lists were cut or folded (DEVELOPMENT "Retiring and merging tests").

`chosen_rows` keeps the named rows of a computed list: the list stays the one source of every case, and the
test runs the rows that each stand for a distinct class. `case_folders` and `plain_rows` serve a test that
checks every case of a list in one run, each case reported as a subtest: a case gets its own folder, as it
did as its own test, and a `pytest.param` row gives the loop its values.
"""
from __future__ import annotations

import itertools
from pathlib import Path


def chosen_rows(rows, *indices):
    """The rows of `rows` at `indices`, in that order."""
    listed = list(rows)
    return [listed[index] for index in indices]


def case_folders(root: Path):
    """An endless supply of fresh folders under `root`, one per case."""
    for number in itertools.count():
        folder = Path(root) / f"case-{number}"
        folder.mkdir()
        yield folder


def plain_rows(rows, width: int = 1):
    """The values of each row, a `pytest.param` unwrapped: one value, or a tuple of `width` values."""
    plain = []
    for row in rows:
        values = getattr(row, "values", None)
        if values is not None and hasattr(row, "marks") and hasattr(row, "id"):
            plain.append(values[0] if width == 1 else tuple(values))
        else:
            plain.append(row)
    return plain
