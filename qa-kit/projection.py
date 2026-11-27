"""The numbered projection: the reviewed candidate, plus a LOCAL carrier commit that differs from it only in VERSION.

A candidate whose VERSION is a development marker (for example 7.12.2-alpha.0) is refused by the candidate's own release guard
(scripts/build_repo_bundle.py ``_read_version``), so the ordinary build never starts on it. There is one narrow way
through, inside a disposable QA stand only: keep the reviewed SHA as the candidate under review, and build from a carrier commit on top
of it that carries the plain release number. Both SHAs and the file difference are recorded separately (projection.json).

BOUNDARY (stated, enforced by this file, checked by tests):
  * The carrier lives only in the isolated fixture checkout of a disposable runner. That checkout has no remote when this runs; nothing is
    pushed, no branch is created (the one local branch ``ouroboros`` moves to the carrier), and THIS FILE CREATES NO TAG.
  * The release guard is the candidate's own and is not touched or bypassed: it is run on the reviewed VERSION (it must refuse, or no
    projection is needed) and on the carrier (it must accept, or the run stops here: no workaround exists in this kit).
  * The carrier may differ from the reviewed SHA in the allowed files only (VERSION). Any other difference is refused as
    ``projection_extra_difference``; a negative control proves that refusal on the very host that runs it.
  * The next step, the unchanged #4220 helper ``prepare_build.py bind``, creates the ONE local annotated test tag v<release> in this same
    fixture checkout, as in #4220: not a project tag, never pushed, never published. Nothing else in the kit touches a tag.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import runpy
import shutil
import sys
import tempfile
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "vendor" / "windows_update_4220_e21fee6")]

import prepare_build as pb  # noqa: E402

ALLOWED = ("VERSION",)
NO_EOL_MAGIC = ("-c", "core.autocrlf=false", "-c", "core.safecrlf=false")  # the carrier's VERSION is written and committed byte for byte


def guard(repo: pathlib.Path) -> tuple[str | None, str | None]:
    """The candidate's own release guard, unchanged: (version, None) when it accepts VERSION, else (None, its refusal as one scrubbed line)."""
    scope = runpy.run_path(str(repo / "scripts" / "build_repo_bundle.py"))
    try:
        return scope["_read_version"](repo), None
    except SystemExit as refused:
        return None, pb._scrub_text(str(refused), 160)  # noqa: SLF001


def changed_files(repo: pathlib.Path, reviewed: str, carrier: str) -> list[str]:
    return sorted(pb._git(repo, "diff", "--name-only", "--no-renames", reviewed, carrier).splitlines())  # noqa: SLF001


def check_difference(repo: pathlib.Path, reviewed: str, carrier: str, allowed: tuple[str, ...] = ALLOWED) -> list[str]:
    """Tree to tree, never the working copy: exactly the allowed files differ, and at least one does."""
    files = changed_files(repo, reviewed, carrier)
    extra = [f for f in files if f not in allowed]
    if extra:
        raise pb.Refusal("projection_extra_difference", verdict="failed", count=len(extra), extra=extra[:5])
    if not files:
        raise pb.Refusal("projection_carrier_equals_the_reviewed_tree", verdict="failed")
    return files


def release_number(version_text: str, explicit: str = "") -> str:
    """The plain release number of the one approved development marker, X.Y.Z-alpha.N, or the operator's explicit one: an rc,
    a typo or any other suffix is refused rather than rebuilt as the release."""
    if explicit:
        return explicit.strip()
    found = re.fullmatch(r"(\d+\.\d+\.\d+)-alpha\.\d+", version_text.strip())
    if not found:
        raise pb.Refusal("release_number_not_derivable", version_text=version_text[:40])
    return found.group(1)


def project(repo: pathlib.Path, reviewed: str, *, release: str = "", allowed: tuple[str, ...] = ALLOWED) -> dict[str, Any]:
    """Prepare and bind the carrier. ``repo`` is the isolated fixture checkout at the reviewed SHA (``prepare_build.py checkout``)."""
    if not pb.SHA_RE.match(reviewed):
        raise pb.Refusal("reviewed_sha_invalid")
    if pb._git(repo, "rev-parse", "HEAD") != reviewed:  # noqa: SLF001
        raise pb.Refusal("checkout_is_not_the_reviewed_sha")
    if pb._git(repo, "status", "--porcelain"):  # noqa: SLF001
        raise pb.Refusal("reviewed_tree_dirty")
    refs = pb._refs(repo)  # noqa: SLF001
    if refs != [(f"refs/heads/{pb.BRANCH}", reviewed)]:
        raise pb.Refusal("ref_area_not_isolated", refs=[name for name, _sha in refs])
    if pb._git(repo, "remote").split():  # noqa: SLF001
        raise pb.Refusal("fixture_checkout_has_a_remote")
    text = (repo / "VERSION").read_text(encoding="utf-8").strip()
    accepted, _refusal = guard(repo)
    facts: dict[str, Any] = {"schema": 1, "reviewed_sha": reviewed, "version_before": pb._scrub_text(text, 40),  # noqa: SLF001
                             "guard_before": "accepted" if accepted else "refused"}  # the guard's own message stays out of published facts
    if accepted is not None:  # already packageable: the reviewed SHA is the carrier, and nothing is written
        return {**facts, "carrier_sha": reviewed, "projected": False, "files_changed": [], "version_after": accepted}
    number = release_number(text, release)
    (repo / "VERSION").write_bytes(number.encode("utf-8") + b"\n")
    pb._git(repo, *NO_EOL_MAGIC, "add", "--", "VERSION")  # noqa: SLF001
    pb._git(repo, *pb.FIXTURE_IDENTITY, *NO_EOL_MAGIC, "commit", "-q", "-m",  # noqa: SLF001
            f"QA carrier: VERSION {pb._scrub_text(text, 40)} -> {number} (local disposable stand; never pushed, never tagged here)")  # noqa: SLF001
    carrier = pb._git(repo, "rev-parse", "HEAD")  # noqa: SLF001
    now_accepted, _refusal_after = guard(repo)  # the unchanged guard decides; there is no way around it here
    if now_accepted is None:
        raise pb.Refusal("carrier_version_not_packageable", verdict="failed")
    files = check_difference(repo, reviewed, carrier, allowed)
    if pb._git(repo, "rev-parse", "HEAD^") != reviewed:  # noqa: SLF001
        raise pb.Refusal("carrier_not_a_child_of_the_reviewed_sha", verdict="failed")
    if pb._git(repo, "show", f"{carrier}:VERSION") != number:  # noqa: SLF001
        raise pb.Refusal("carrier_version_blob_differs", verdict="failed")
    if pb._git(repo, "status", "--porcelain") or pb._refs(repo) != [(f"refs/heads/{pb.BRANCH}", carrier)]:  # noqa: SLF001
        raise pb.Refusal("carrier_left_the_checkout_unclean", verdict="failed")
    return {**facts, "carrier_sha": carrier, "projected": True, "files_changed": files, "version_after": now_accepted, "guard_after": "accepted"}


def negative_control(parent: str | None = None) -> str:
    """Prove, on this very host, that an extra difference is refused: a throwaway repository (never the checkout) whose second commit changes
    VERSION and one more file. Commits only; no tag, no remote."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="qa-projection-control-", dir=parent))
    try:
        pb._git(root, "init", "-q")  # noqa: SLF001
        (root / "VERSION").write_text("1.2.3-alpha.0\n", encoding="utf-8")
        (root / "other.txt").write_text("one\n", encoding="utf-8")
        pb._git(root, "add", ".")  # noqa: SLF001
        pb._git(root, *pb.FIXTURE_IDENTITY, "commit", "-q", "-m", "reviewed")  # noqa: SLF001
        first = pb._git(root, "rev-parse", "HEAD")  # noqa: SLF001
        (root / "VERSION").write_text("1.2.3\n", encoding="utf-8")
        (root / "other.txt").write_text("two\n", encoding="utf-8")
        pb._git(root, "add", ".")  # noqa: SLF001
        pb._git(root, *pb.FIXTURE_IDENTITY, "commit", "-q", "-m", "carrier with an extra difference")  # noqa: SLF001
        second = pb._git(root, "rev-parse", "HEAD")  # noqa: SLF001
        try:
            check_difference(root, first, second)
        except pb.Refusal as refusal:
            return "refused" if refusal.reason == "projection_extra_difference" else "wrong_refusal"
        return "not_refused"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("project")
    one.add_argument("--repo", default=".")
    one.add_argument("--reviewed", required=True, help="the reviewed candidate SHA (candidate.json): the exact checkout HEAD")
    one.add_argument("--release", default="", help="an explicit release number; default: X.Y.Z of the development marker")
    one.add_argument("--allow", action="append", default=[], help="one more file the carrier may change (default: VERSION only)")
    one.add_argument("--evidence", required=True)
    one.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    args = parser.parse_args(argv)
    evidence, reviewed = pathlib.Path(args.evidence), args.reviewed.strip().lower()
    try:
        control = negative_control()
        if control != "refused":
            raise pb.Refusal("negative_control_not_refused", verdict="failed", control=control)
        facts = project(pathlib.Path(args.repo).resolve(), reviewed, release=args.release, allowed=(*ALLOWED, *args.allow))
        facts["negative_control"] = control
        facts["carrier_is_local_only"] = True
        pb.write_json_atomic(evidence / "projection.json", facts)
        pb._outputs(args.github_output, {"reviewed_sha": facts["reviewed_sha"], "carrier_sha": facts["carrier_sha"], "projected": str(facts["projected"]).lower()})  # noqa: SLF001
        print(json.dumps({k: facts[k] for k in ("reviewed_sha", "carrier_sha", "projected", "files_changed", "version_before", "version_after")}, indent=2))
        return 0
    except pb.Refusal as refusal:
        pb.record_structured_result(evidence, verdict=refusal.verdict, phase="project_candidate", code=refusal.reason, candidate_sha=reviewed, fields=refusal.fields)
        sys.stderr.write(f"::error::{refusal}\n")
        return 2 if refusal.verdict == "unconfirmed" else 1
    except Exception as exc:  # a harness defect still leaves a structured result, never a bare traceback
        pb.record_structured_result(evidence, verdict="error", phase="project_candidate", code="harness_error", candidate_sha=reviewed, fields={"type": type(exc).__name__})
        sys.stderr.write(f"::error::harness_error {type(exc).__name__}\n")
        return 3


if __name__ == "__main__":
    sys.exit(main())
