"""Capture/replay storage: preserve wire text, neutralize paths, share exact bodies."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact_rows(rows):
    """Run-length storage for byte-identical consecutive polls; replay keeps their count."""
    compact = []
    for row in rows:
        if compact and {key: value for key, value in compact[-1].items() if key != "repeat"} == row:
            compact[-1]["repeat"] = compact[-1].get("repeat", 1) + 1
        else:
            compact.append(dict(row))
    return compact


def write_capture(folder, name, value, *, prefixes):
    """Keep raw evidence beside the portable fixture; substitute only path prefixes."""
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "raw"
    raw.mkdir(exist_ok=True)
    (raw / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def neutralize(item):
        if isinstance(item, str):
            for original, neutral in prefixes:
                item = item.replace(original, neutral)
            return item
        if isinstance(item, list):
            return [neutralize(child) for child in item]
        if isinstance(item, dict):
            return {key: neutralize(child) for key, child in item.items()}
        return item

    value = neutralize(value)
    tree = value["engine_tree"]
    counts = Counter(row["body_utf8"] for row in value["wire"])
    for row in value["wire"]:
        body = row["body_utf8"]
        catalog = row["path"] == "/v2/operations"
        if not catalog and (len(body) < 512 or counts[body] < 2):
            continue
        sha = digest(body)
        ref = f"operations-{tree[:8]}.json" if catalog else f"body-{tree[:8]}-{sha[:16]}.json"
        shared = {"engine_tree": tree, "sha256": sha, "body_utf8": body}
        path = folder / ref
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == shared, f"Changed body at {ref}"
        else:
            path.write_text(json.dumps(shared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row.pop("body_utf8")
        row.update(body_ref=ref, body_sha256=sha)
    value["wire"] = compact_rows(value["wire"])
    (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_capture(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in value["wire"]:
        if "body_ref" in row:
            ref = row["body_ref"]
            assert Path(ref).name == ref
            shared = json.loads((path.parent / ref).read_text(encoding="utf-8"))
            assert shared["engine_tree"] == value["engine_tree"]
            assert digest(shared["body_utf8"]) == shared["sha256"] == row["body_sha256"]
            row["body_utf8"] = shared["body_utf8"]
        repeats = row.pop("repeat", 1)
        assert type(repeats) is int and 1 <= repeats <= 10000
        rows.extend(deepcopy(row) for _ in range(repeats))
    value["wire"] = rows
    return deepcopy(value)
