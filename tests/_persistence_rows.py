"""Read inventory cells, including Path labels that begin with prose."""
import re

_LIFETIME_RE = re.compile(r"^\[(record|bounded|disposable: ([\w.]+))\]")


def _cells(line: str) -> list[str]:
    cells, current, code = [], [], False
    for char in line.strip().strip("|"):
        if char == "`":
            code = not code
        if char == "|" and not code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    return [*cells, "".join(current).strip()]


def inventory_rows(text: str, *, whole: bool = False) -> list[tuple[str, ...]]:
    """Path and Retention of every real inventory row, never an example table.

    By default the Path is its first code span (the lifetime check's legacy key).
    The file-kind check needs the whole cell, including explicitly named siblings,
    and the Writer cell as a third item.
    """
    rows, retention = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            retention = None
            continue
        cells = _cells(stripped)
        if retention is None:
            retention = cells.index("Retention") if "Retention" in cells else None
            writer = cells.index("Writer") if "Writer" in cells else None
        elif not set(stripped) <= set("|-: "):
            spans = cells[0].split("`")
            cell = cells[retention] if retention < len(cells) else ""
            if whole:
                rows.append((cells[0], cell, cells[writer] if writer is not None and writer < len(cells) else ""))
            else:
                rows.append((cells[0] if len(spans) < 3 else spans[1], cell))
    return rows


def lifetime(cell: str):
    """``(class, retirer)`` named at the start of the Retention cell, or ``None``."""
    match = _LIFETIME_RE.match(cell)
    if not match:
        return None
    return ("disposable", match.group(2)) if match.group(2) else (match.group(1), None)
