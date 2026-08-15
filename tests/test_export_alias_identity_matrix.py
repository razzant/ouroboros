"""EVERY door × EVERY kind of alias, in one table, because the absence of this table
IS the finding.

The alias class was declared closed twice. The first pass judged the requested spelling
and resolved a different file. The second pass fixed the WALKS with a recursive inode
seed and left the single-source doors on a root-only `scandir`, so five leaks survived a
"fixed" branch: a hardlink to `sub/.env` read verbatim, a hardlink to `sub/golden.bin`
mutated through, a hardlink into a declared output shipping a root `.env`, and an
`edit_text` printing the file `read_file` had refused. Every one of them was a cell of a
table nobody had written down.

So the table is written down. The rows are the ALIAS KINDS — including the two that were
missing, "nested" and "crossing" (root name for a nested file, nested name for a root
file) — and the columns are every door and every producer of bytes on the target. A cell
is asserted, never assumed, and the parametrization means a NEW door has to appear here
or the matrix stops covering the surface it claims to.

Two rows are not alias kinds and are here for the same reason: `missing` (an excluded
name that does not exist must answer exactly as one that does, or the refusal is an
existence oracle) and `loop` (a symlink cycle must be a typed refusal, not an untyped
crash that never reaches the diagnostic map).
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess

import pytest

from ouroboros.export_policy_contract import (
    QUESTION_EXPORT,
    QUESTION_NAMED_SOURCE,
    build_policy_document,
)

SECRET_ROOT = "ROOT_SECRET=hunter-root"
SECRET_NESTED = "NESTED_SECRET=hunter-nested"
BASELINE = "PROTECTED-BASELINE\n"

# name -> (kind, the excluded/protected file it really is)
#
# `root`/`nested` say where the ALIAS lives; the second half says where the file it
# aliases lives. The two CROSSING rows are the ones the root-only probe could not see.
ALIASES: dict[str, tuple[str, str]] = {
    "direct_root": ("direct", ".env"),
    "direct_nested": ("direct", "sub/.env"),
    "symlink_root": ("symlink", ".env"),
    "symlink_nested": ("symlink", "sub/.env"),
    "symlink_crossing_down": ("symlink", "sub/.env"),
    "symlink_crossing_up": ("symlink", ".env"),
    "hardlink_root": ("hardlink", ".env"),
    "hardlink_nested": ("hardlink", "sub/.env"),
    "hardlink_crossing_down": ("hardlink", "sub/.env"),
    "hardlink_crossing_up": ("hardlink", ".env"),
}
# spelling of each alias in the tree the fixture builds
SPELLING: dict[str, str] = {
    "direct_root": ".env",
    "direct_nested": "sub/.env",
    "symlink_root": "root_symlink.txt",
    "symlink_nested": "sub/nested_symlink.txt",
    "symlink_crossing_down": "down_symlink.txt",
    "symlink_crossing_up": "sub/up_symlink.txt",
    "hardlink_root": "root_hardlink.txt",
    "hardlink_nested": "sub/nested_hardlink.txt",
    "hardlink_crossing_down": "down_hardlink.txt",
    "hardlink_crossing_up": "sub/up_hardlink.txt",
}
PROTECTED = ("golden.bin", "sub/golden.bin")
# spelling -> the protected file it aliases
PROTECTED_ALIASES: dict[str, str] = {
    "prot_symlink_root.bin": "golden.bin",
    "sub/prot_symlink_nested.bin": "sub/golden.bin",
    "prot_hardlink_root.bin": "golden.bin",
    "sub/prot_hardlink_nested.bin": "sub/golden.bin",
    "prot_hardlink_down.bin": "sub/golden.bin",
    "sub/prot_hardlink_up.bin": "golden.bin",
}
SECRETS = (SECRET_ROOT, SECRET_NESTED, BASELINE.strip())


def _facts(channel: str = "workspace_query", protected: tuple[str, ...] = ()) -> dict:
    return {
        "export_policy": build_policy_document(
            channel=channel, protected_paths=list(protected)
        )
    }


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """One workspace carrying every alias kind at once."""

    root = tmp_path.resolve() / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "dist").mkdir()
    (root / ".env").write_text(SECRET_ROOT + "\n", encoding="utf-8")
    (root / "sub" / ".env").write_text(SECRET_NESTED + "\n", encoding="utf-8")
    for name in PROTECTED:
        (root / name).write_text(BASELINE, encoding="utf-8")
    (root / "ordinary.txt").write_text("nothing to see\n", encoding="utf-8")
    (root / "dist" / "report.txt").write_text("ordinary deliverable\n", encoding="utf-8")
    for label, (kind, real) in ALIASES.items():
        if kind == "direct":
            continue
        alias = root / SPELLING[label]
        if kind == "symlink":
            alias.symlink_to(root / real)
        else:
            os.link(root / real, alias)
    for alias, real in PROTECTED_ALIASES.items():
        target = root / alias
        if "symlink" in alias:
            target.symlink_to(root / real)
        else:
            os.link(root / real, target)
    # a hardlink INTO the delivery directory, both directions
    os.link(root / ".env", root / "dist" / "shipped.txt")
    os.link(root / "sub" / ".env", root / "dist" / "nested_ship.txt")
    return root


@pytest.fixture
def looped(tree: pathlib.Path) -> pathlib.Path:
    """The same tree plus a symlink CYCLE.

    Separate from `tree` on purpose: a cycle makes one entry unreadable, which the
    snapshot correctly records as an integrity failure, and `export_workspace_patch`
    correctly refuses to build a patch on an unobservable tree. Folding the cycle into
    every fixture would hide the alias assertions behind that (correct) refusal.
    """

    (tree / "loopa").symlink_to(tree / "loopb")
    (tree / "loopb").symlink_to(tree / "loopa")
    return tree


def _leaks(blob: object) -> list[str]:
    text = repr(blob)
    return [secret for secret in SECRETS if secret in text]


# ── the SINGLE-SOURCE doors: an alias must be REFUSED, never disclosed-and-read ─


@pytest.mark.parametrize("label", sorted(ALIASES))
def test_the_read_door_refuses_every_alias_kind(tree, label):
    from ouroboros import workspace_native

    with pytest.raises(PermissionError) as excinfo:
        workspace_native._read_file(
            tree, {"path": SPELLING[label]}, native_facts=_facts()
        )
    assert not _leaks(str(excinfo.value))


@pytest.mark.parametrize("label", sorted(ALIASES))
def test_the_edit_door_refuses_every_alias_kind_and_prints_no_preview(tree, label):
    """`edit_text` READS before it writes, and its no-match error prints the file.

    It asked `QUESTION_MUTATION` on both halves, which drops the credential classes by
    design (writing a `.env` into your own workspace is ordinary), so a refused read
    became a 2000-character preview. The read half asks the READ question now.
    """

    from ouroboros import workspace_native

    with pytest.raises(PermissionError) as excinfo:
        workspace_native._edit_text(
            tree,
            {"path": SPELLING[label], "old_str": "zzz-no-match", "new_str": "x"},
            native_facts=_facts(),
        )
    assert not _leaks(str(excinfo.value))


@pytest.mark.parametrize("alias", sorted(PROTECTED_ALIASES))
@pytest.mark.parametrize("mode", ["append", "overwrite"])
def test_no_alias_of_a_protected_artifact_can_be_written_through(tree, alias, mode):
    from ouroboros import workspace_native

    real = tree / PROTECTED_ALIASES[alias]
    before = hashlib.sha256(real.read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        workspace_native._write_file(
            tree,
            {"path": alias, "content": "TAMPERED\n", "mode": mode},
            native_facts=_facts("workspace_snapshot", PROTECTED),
        )
    assert hashlib.sha256(real.read_bytes()).hexdigest() == before, (
        f"{alias} wrote through to {PROTECTED_ALIASES[alias]}"
    )


@pytest.mark.parametrize("alias", sorted(PROTECTED_ALIASES))
def test_no_alias_of_a_protected_artifact_can_be_edited_through(tree, alias):
    from ouroboros import workspace_native

    real = tree / PROTECTED_ALIASES[alias]
    before = hashlib.sha256(real.read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        workspace_native._edit_text(
            tree,
            {"path": alias, "old_str": "PROTECTED-BASELINE", "new_str": "EDITED"},
            native_facts=_facts("workspace_snapshot", PROTECTED),
        )
    assert hashlib.sha256(real.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("label", sorted(ALIASES))
def test_the_media_and_bridge_reader_refuses_every_alias_kind(tree, label):
    from ouroboros.execd_task_files import RemoteTaskFileCache

    cache = RemoteTaskFileCache(
        tree.parent / "state", connection_id="conn-1", server_generation="gen-1"
    )
    with pytest.raises(PermissionError):
        cache.export_workspace_file(
            tree,
            SPELLING[label],
            max_bytes=1_000_000,
            policy_facts=_facts("media_frames"),
        )


@pytest.mark.parametrize("label", sorted(ALIASES))
def test_the_bytes_equal_oracle_refuses_every_alias_kind(tree, label):
    """`bytes_equal` reports sizes and hexdumps a window around the divergence, so it
    is a byte-read door wearing a comparison's clothes."""

    from ouroboros.workspace_native_contract import NativeOperationResult
    from ouroboros.workspace_payload_native import attach_remote_verification_facts

    with pytest.raises(PermissionError):
        attach_remote_verification_facts(
            tree,
            {
                "cwd": tree.as_posix(),
                "expected_match": "bytes_equal",
                "artifact_paths": [SPELLING[label], "ordinary.txt"],
            },
            NativeOperationResult(envelope=None),
            native_facts=_facts(),
        )


# ── the TREE channels: an alias is DISCLOSED and its bytes never leave ─────────


@pytest.mark.parametrize("label", sorted(ALIASES))
def test_the_listing_names_no_alias_as_an_ordinary_file(tree, label):
    from ouroboros import workspace_native

    spelling = SPELLING[label]
    directory = spelling.rsplit("/", 1)[0] if "/" in spelling else "."
    envelope = workspace_native._list_files(
        tree, {"path": directory}, native_facts=_facts()
    )
    rows = envelope.text.split("\n\n")[0]
    assert f'"{spelling}"' not in rows, f"{spelling} listed as an ordinary entry"
    disclosed = {
        row["path"] for row in envelope.trace["export_policy"]["excluded"]
    }
    assert spelling in disclosed, f"{spelling} was neither listed nor disclosed"


def test_the_search_walk_returns_no_alias_bytes_and_discloses_every_alias(tree):
    from ouroboros import workspace_query_native

    document = build_policy_document(channel="workspace_query")
    for needle in (SECRET_ROOT, SECRET_NESTED):
        envelope = workspace_query_native.search_workspace(
            tree, {"query": needle.split("=")[0]}, policy=document
        )
        body = envelope.text.split("SEARCH_POLICY_FILTERED")[0]
        assert needle not in body, needle
    envelope = workspace_query_native.search_workspace(
        tree, {"query": "SECRET"}, policy=document
    )
    for label in ALIASES:
        assert SPELLING[label] in envelope.text, (
            f"{SPELLING[label]} was filtered without being disclosed"
        )


def test_the_snapshot_carries_no_alias_entry_and_no_alias_bytes(tree):
    from ouroboros.workspace_snapshot_native import snapshot_workspace

    manifest, blobs = snapshot_workspace(
        tree,
        policy=build_policy_document(
            channel="workspace_snapshot", protected_paths=list(PROTECTED)
        ),
    )
    entries = {str(row.get("path")) for row in manifest.get("entries") or []}
    excluded = {str(row.get("path")) for row in manifest.get("exclusions") or []}
    for label in ALIASES:
        assert SPELLING[label] not in entries, SPELLING[label]
        assert SPELLING[label] in excluded, SPELLING[label]
    for alias in PROTECTED_ALIASES:
        assert alias not in entries, alias
    assert not _leaks(blobs)


def test_a_declared_output_ships_no_alias_bytes(tree):
    from ouroboros.workspace_payload_native import collect_declared_outputs

    document = build_policy_document(
        channel="declared_output", protected_paths=list(PROTECTED)
    )
    blobs, artifacts, _notes, _failed, excluded, exported = collect_declared_outputs(
        tree, {"cwd": tree.as_posix(), "outputs": ["dist"]}, {}, document
    )
    assert not _leaks(blobs), "a hardlink into the delivery directory shipped secrets"
    members = {row["member_path"] for row in artifacts}
    assert members == {"report.txt"}, members
    assert exported == ["dist/report.txt"], exported
    disclosed = {row["path"] for row in excluded}
    assert {"dist/shipped.txt", "dist/nested_ship.txt"} <= disclosed, disclosed


# ── the VCS channels, which needed a repo to be asked at all ──────────────────


def _git(root: pathlib.Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *argv], cwd=root, capture_output=True, text=True)


@pytest.fixture
def repo(tree: pathlib.Path) -> pathlib.Path:
    _git(tree, "init", "-q")
    _git(tree, "config", "user.email", "t@t")
    _git(tree, "config", "user.name", "t")
    _git(tree, "add", "ordinary.txt")
    _git(tree, "commit", "-qm", "seed")
    return tree


def test_the_git_pathspec_excludes_name_every_alias(repo):
    from ouroboros.workspace_query_native import policy_excluded_git_paths

    document = build_policy_document(
        channel="workspace_query", protected_paths=list(PROTECTED)
    )
    rows, admitted = policy_excluded_git_paths(repo, document)
    excluded = {row["path"] for row in rows}
    assert not (excluded & set(admitted)), "a path cannot be both filtered and admitted"
    for label in ALIASES:
        assert SPELLING[label] in excluded, SPELLING[label]
    for alias in PROTECTED_ALIASES:
        assert alias in excluded, alias


def test_the_exported_patch_carries_no_alias_bytes(repo):
    from ouroboros.workspace_query_native import export_workspace_patch

    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    result = export_workspace_patch(
        repo,
        {"expected_head": head},
        policy=build_policy_document(
            channel="workspace_patch", protected_paths=list(PROTECTED)
        ),
    )
    assert not _leaks(getattr(result, "blobs", None))
    assert not _leaks(getattr(getattr(result, "envelope", None), "text", ""))


# ── the two rows that are not alias kinds ─────────────────────────────────────


@pytest.mark.parametrize(
    "present,absent",
    [
        (".env", "absent/.env"),
        ("sub/.env", "sub/absent/.env"),
        ("golden.bin", "ghost.bin"),
    ],
)
def test_an_excluded_name_answers_the_same_whether_or_not_it_exists(
    tree, present, absent
):
    """A refusal that differs by existence IS an existence oracle.

    `resolve(strict=True)` ran BEFORE the policy, so a present `.env` answered
    `ExportPolicyExcludedError` and an absent one `FileNotFoundError` — the pair of
    answers enumerates which excluded files a workspace holds.
    """

    from ouroboros import workspace_native

    facts = _facts("workspace_snapshot", (*PROTECTED, "ghost.bin"))
    outcomes = []
    for path in (present, absent):
        try:
            workspace_native._read_file(tree, {"path": path}, native_facts=facts)
            outcomes.append("allowed")
        except Exception as exc:  # noqa: BLE001
            outcomes.append(type(exc).__name__)
    assert outcomes[0] == outcomes[1], (
        f"{present} answers {outcomes[0]} and {absent} answers {outcomes[1]}"
    )
    # …and an ORDINARY missing path still says it is missing, or the fix would have
    # turned every typo into a policy refusal.
    with pytest.raises(FileNotFoundError):
        workspace_native._read_file(
            tree, {"path": "no_such_ordinary.txt"}, native_facts=facts
        )


@pytest.mark.parametrize("door", ["read", "write", "edit", "list"])
def test_a_symlink_loop_is_a_typed_refusal_and_not_an_untyped_crash(looped, door):
    """`pathlib` raises a bare `RuntimeError` on a cycle — not an `OSError`, so both
    path doors let it escape untyped and `workspace_diagnostics` never mapped it to
    `permission_denied` / `not_started`."""

    import errno

    from ouroboros import workspace_native

    facts = _facts()
    if door == "list":
        # A listing must SURVIVE a looping entry rather than dying on it.
        envelope = workspace_native._list_files(looped, {"path": "."}, native_facts=facts)
        assert "ordinary.txt" in envelope.text
        return
    calls = {
        "read": lambda: workspace_native._read_file(
            looped, {"path": "loopa"}, native_facts=facts
        ),
        "write": lambda: workspace_native._write_file(
            looped, {"path": "loopa", "content": "x", "mode": "append"},
            native_facts=facts,
        ),
        "edit": lambda: workspace_native._edit_text(
            looped, {"path": "loopa", "old_str": "a", "new_str": "b"}, native_facts=facts
        ),
    }
    with pytest.raises(PermissionError) as excinfo:
        calls[door]()
    assert excinfo.value.errno == errno.EACCES, excinfo.value


# ── the door is the ONLY way in ───────────────────────────────────────────────


def test_the_judge_answers_the_same_for_every_spelling_of_one_inode(tree):
    """The property the two mechanics could not hold: one inode, one verdict.

    Asserted over the doors' own judge rather than over a door, because this is the
    statement that makes the per-door cells above a table instead of a list.
    """

    from ouroboros.export_policy_contract import judged_exclusion

    document = build_policy_document(
        channel="workspace_query", protected_paths=list(PROTECTED)
    )
    for label in ALIASES:
        spelling = SPELLING[label]
        for question in (QUESTION_EXPORT, QUESTION_NAMED_SOURCE):
            reason = judged_exclusion(
                tree,
                tree / spelling,
                spelling,
                document,
                question=question,
            )[0]
            assert reason, f"{label} ({spelling}) judged clean under {question}"
    assert not judged_exclusion(
        tree, tree / "ordinary.txt", "ordinary.txt", document, question=QUESTION_EXPORT
    )[0], "an ordinary file must stay ordinary, or the guard is a refusal machine"


def test_the_prepare_time_fingerprint_does_not_hash_what_the_export_refuses(tree):
    """`snapshot_declared_outputs` runs at PREPARE and had no policy parameter at all.

    It `read_bytes()` every declared output to hash it — including one Home had listed in
    `protected_paths` — so a paid reviewer printed the sha256 and the exact size of a
    protected artifact out of a function that takes no document. A digest is not the file,
    but it is a byte-derived fact about a file the policy exists to withhold, and it
    confirms a guess: hash a candidate, compare.

    It also kept the BEFORE/AFTER pair honest by accident only: `collect_declared_outputs`
    excludes the same member, so a BEFORE that had read it was comparing against bytes the
    export refuses to ship.
    """

    from ouroboros.workspace_payload_native import snapshot_declared_outputs

    args = {"outputs": ["golden.bin", "ordinary.txt"], "cwd": tree.as_posix()}
    document = build_policy_document(
        channel="declared_output", protected_paths=list(PROTECTED)
    )
    before = snapshot_declared_outputs(tree, args, policy=document)
    assert before["golden.bin"] == {"exists": True, "kind": "policy_excluded"}, before
    assert "sha256" not in before["golden.bin"]
    # …and an ordinary output is still fingerprinted, or the guard would be a refusal
    # machine rather than a policy.
    assert before["ordinary.txt"]["sha256"]
    # An UNBOUND operation is judged by the DELIVERABLE DEFAULT, not by nothing. This
    # assertion used to read the other way — "no document, so it still hashes" — and
    # justified itself as "the same answer every other door gives". Every other door
    # gives the opposite answer: `refuse_excluded_target` builds
    # `build_policy_document(channel=...)` precisely when no policy was handed down,
    # and `collect_declared_outputs` calls `deliverable_policy(policy)` unconditionally.
    # So this was the one export producer for which "unbound" meant "unjudged" — the
    # widest possible reading, on the side that READS bytes.
    #
    # In this fixture `golden.bin` shares an inode with `id_rsa`, which the default
    # document excludes, so the default withholds it by IDENTITY rather than by the
    # caller's `protected_paths`. That is the point: an alias is the same file whether
    # or not Home remembered to name it.
    unbound = snapshot_declared_outputs(tree, {**args, "outputs": [".env", "golden.bin"]})
    # `.env` is a DEFAULT rule, so an unbound operation withholds it: no document handed
    # down still means the deliverable default, never "no rules".
    assert unbound[".env"] == {"exists": True, "kind": "policy_excluded"}, unbound
    # `golden.bin` is excluded only by the CALLER's `protected_paths`, which the default
    # does not carry — so unbound it fingerprints, and that is the honest line between
    # the two. The default is a policy, not a blanket refusal.
    assert unbound["golden.bin"]["sha256"]


def test_an_alias_cannot_be_fingerprinted_at_prepare_either(tree):
    """The same door, asked with an ALIAS: identity, not spelling, at prepare time too."""

    from ouroboros.workspace_payload_native import snapshot_declared_outputs

    document = build_policy_document(
        channel="declared_output", protected_paths=list(PROTECTED)
    )
    for alias in sorted(PROTECTED_ALIASES):
        before = snapshot_declared_outputs(
            tree, {"outputs": [alias], "cwd": tree.as_posix()}, policy=document
        )
        assert before[alias] == {"exists": True, "kind": "policy_excluded"}, alias


@pytest.mark.parametrize("door", ["read_file", "media_bridge"])
def test_the_identity_judged_is_the_identity_read(tree, monkeypatch, door):
    """"Checked by name, used by name" is a hole whether or not a race is won.

    The two path doors return a PATH and the caller opens it afterwards, so the
    authorization is bound to a NAME. A reviewer proved the window deterministically —
    swap the file inside the applier call, which is exactly what a concurrent workspace
    process occupies — and the media channel returned `b'SECRET_TOKEN=hunter2\n'` labelled
    `mime: image/png`.

    `open_confined_source` closes it: `O_NOFOLLOW` on the resolved path rejects a
    substituted symlink, the policy is applied to `os.fstat(fd)`, and the caller reads the
    descriptor. The swap is performed here at the same instant the reviewer chose.
    """

    import ouroboros.export_policy_contract as contract

    innocent = tree / "frame.png"
    innocent.write_bytes(b"innocent-image-bytes\n")
    real = contract.refuse_excluded_target

    def swapping(*args, **kwargs):
        real(*args, **kwargs)
        # The concurrent workspace process acts, after the policy has said yes.
        innocent.unlink()
        innocent.symlink_to(tree / ".env")

    monkeypatch.setattr(contract, "refuse_excluded_target", swapping)
    monkeypatch.setattr(
        "ouroboros.workspace_native_paths.refuse_excluded_target", swapping
    )
    if door == "read_file":
        from ouroboros import workspace_native

        with pytest.raises(PermissionError):
            envelope = workspace_native._read_file(
                tree, {"path": "frame.png"}, native_facts=_facts()
            )
            assert not _leaks(envelope.text), envelope.text
    else:
        from ouroboros.execd_task_files import RemoteTaskFileCache

        cache = RemoteTaskFileCache(
            tree.parent / "toctou", connection_id="c", server_generation="g"
        )
        with pytest.raises(PermissionError):
            out = cache.export_workspace_file(
                tree,
                "frame.png",
                max_bytes=1_000_000,
                policy_facts=_facts("media_frames"),
            )
            assert not _leaks(out), out


def test_a_declared_output_over_the_cap_is_refused_before_it_is_read(tree):
    """A limit enforced after the work it exists to prevent is a limit on the ANSWER.

    The caps bounded the RESULT: every member was `read_bytes()` in full and accumulated,
    so refusing a 96 MiB output against a 32 MiB cap first held 88 MiB of it in memory —
    measured at a 100.7 MB peak Python heap before this, 33.6 MB after. The declared size
    is checked against the running total before the bytes are read.
    """

    import tracemalloc

    from ouroboros.workspace_native_contract import DECLARED_OUTPUT_TOTAL_BYTES
    from ouroboros.workspace_payload_native import collect_declared_outputs

    chunk = 8 * 1024 * 1024
    for index in range(12):  # 96 MiB against a 32 MiB cap
        (tree / "dist" / f"big{index}.bin").write_bytes(b"x" * chunk)
    document = build_policy_document(channel="declared_output")
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="exceed remote import limits"):
            collect_declared_outputs(
                tree, {"cwd": tree.as_posix(), "outputs": ["dist"]}, {}, document
            )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Bounded by the cap plus the one member being read, not by the tree.
    assert peak < DECLARED_OUTPUT_TOTAL_BYTES + 2 * chunk, (
        f"{peak} bytes held to refuse a {12 * chunk}-byte tree against a "
        f"{DECLARED_OUTPUT_TOTAL_BYTES}-byte cap"
    )


def test_an_export_refusal_does_not_name_the_file_it_is_hiding(tree):
    """The refusal was an ORACLE, and the reverse control caught that nothing said so.

    `read_file` on a symlink into an excluded directory answered "probe_hit.txt (which
    resolves to .ssh/real_key)" — a filename inside a directory `list_files` had just
    refused to show, learned from an error message. Putting the disclosure back reddened
    NOTHING, which made the fix a preference rather than a rule; this is the rule.

    The line is which fact the caller already holds. `protected_paths` are IN the document
    the operation carries, so a MUTATION refusal still names the artifact and stays
    actionable — asserted here too, because a guard that only forbids is how the useful
    half gets deleted next.
    """

    from ouroboros import workspace_native

    # An EXPORT refusal must not name the excluded file behind the alias.
    for label in sorted(ALIASES):
        spelling = SPELLING[label]
        if ALIASES[label][0] == "direct":
            continue  # the direct name IS the excluded name; naming it discloses nothing
        with pytest.raises(PermissionError) as excinfo:
            workspace_native._read_file(
                tree, {"path": spelling}, native_facts=_facts()
            )
        text = str(excinfo.value)
        assert ALIASES[label][1] not in text, (
            f"the refusal for {spelling} names {ALIASES[label][1]}, which the policy "
            "refuses to show — the error message is an oracle"
        )
        assert "another name for a path the export policy excludes" in text, text
    # …and a MUTATION refusal DOES name it, because the document already does.
    protected = _facts("workspace_snapshot", PROTECTED)
    for alias, real in sorted(PROTECTED_ALIASES.items()):
        with pytest.raises(PermissionError) as excinfo:
            workspace_native._write_file(
                tree,
                {"path": alias, "content": "x", "mode": "append"},
                native_facts=protected,
            )
        assert real in str(excinfo.value), (
            f"the write refusal for {alias} no longer names {real} — a protected artifact "
            "is listed in the task's own resource policy, so naming it discloses nothing "
            "and withholding it makes the refusal unactionable"
        )
