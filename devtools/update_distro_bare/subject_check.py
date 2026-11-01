"""Bind this equipment to one subject commit, by its Git objects. Stdlib and git only; the subject is only read.

    subject_check.py check  MANIFEST CHECKOUT      exit 1: the tree or a consumer's bytes are not the manifest's
    subject_check.py rebind MANIFEST CHECKOUT REV  prints a new ``subject`` block and every consumer that changed

``rebind`` changes nothing and decides nothing. It also reports each selected node missing from the new tree, and a
changed consumer means its row's meaning must be reread before the new block is used.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys


def _git(checkout: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", checkout, *args], check=True, capture_output=True).stdout


def facts(manifest: dict, checkout: str, rev: str) -> dict:
    commit = _git(checkout, "rev-parse", f"{rev}^{{commit}}").decode().strip()
    return {"commit": commit, "tree": _git(checkout, "rev-parse", f"{commit}^{{tree}}").decode().strip(),
            "consumers_sha256": {path: hashlib.sha256(_git(checkout, "show", f"{commit}:{path}")).hexdigest()
                                 for path in sorted(manifest["subject"]["consumers_sha256"])}}


def missing_nodes(manifest: dict, checkout: str, commit: str) -> list:
    """Selected subject nodes the new tree no longer defines. Overlay nodes travel with this equipment."""
    wanted = sorted({node for cell in manifest["cells"].values() for phase in cell["phases"].values()
                     for node in phase.get("nodes", []) if not node.startswith(manifest["overlay_test"])})
    gone = []
    for node in wanted:
        path, name = node.split("::", 1)
        tree = ast.parse(_git(checkout, "show", f"{commit}:{path}"))
        if name not in {item.name for item in tree.body if isinstance(item, ast.FunctionDef)}:
            gone.append(node)
    return gone


def main(argv: list) -> int:
    with open(argv[1], encoding="utf-8") as fh:
        manifest = json.load(fh)
    pinned = manifest["subject"]
    if argv[0] == "check":
        rev = pinned["commit"]
        found = facts(manifest, argv[2], rev)
        head = _git(argv[2], "rev-parse", "HEAD").decode().strip()
        problems = [] if found == {key: pinned[key] for key in found} and head == pinned["commit"] else [
            f"checkout is {head}", {key: value for key, value in found.items() if pinned.get(key) != value}]
        problems += missing_nodes(manifest, argv[2], rev)
        print(json.dumps({"ok": not problems, "problems": problems, "commit": rev}, indent=1))
        return 0 if not problems else 1
    if argv[0] == "rebind":
        found = facts(manifest, argv[2], argv[3])
        changed = sorted(path for path, digest in found["consumers_sha256"].items()
                         if pinned["consumers_sha256"].get(path) != digest)
        print(json.dumps({"subject": found, "consumers_changed_from_pinned": changed,
                          "selected_nodes_missing": missing_nodes(manifest, argv[2], found["commit"])}, indent=1))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
