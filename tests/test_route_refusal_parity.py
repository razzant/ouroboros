"""ROUTE PARITY — the native route must refuse everywhere the local route refuses.

The failure class this file closes is the one the PR 79 postmortem called "one policy ×
N doors", returning on a new axis: not two copies of a rule that drifted, but ONE rule
that exists on one route and not the other. Four instances were confirmed by review and
all four are the same sentence — *a check the local route performs is absent on the
native route, or a guard opens when it fails*:

* `write_file` with ``mode="append"`` followed a symlink in the final path component out
  of the workspace on the target. The local route resolves the whole spelling and
  refuses (`tool_access.resolve_resource_path`). Reproduced live: the file outside the
  workspace grew.
* `start_service` SANITIZED a service name the local route REFUSES, silently merging two
  services into one log file.
* `native_relative_spelling` accepted NUL and control characters that `utils.safe_relpath`
  has rejected on the local route since long before there was a remote one.
* the public argument-schema refusal ran AFTER `prepare_operation` on the native branch,
  so a malformed call reserved a token on the target before Home answered.

Why the parity is the property and not each instance: a per-instance test proves the
instance. What made this a class is that nobody was comparing the two routes at all, so
the rule that has to be written down is the COMPARISON. Hence `docs/DEVELOPMENT.md`'s
**Route Parity Rule**, and hence this file, which asks the same refusable question of
both routes and requires the same verdict.

BOUNDARY. Parity is asserted about the REFUSAL — that both routes say no — and not about
the message text. The two routes legitimately word things differently (a local
`⚠️ TOOL_ERROR`, a native typed diagnostic) and pinning text here would make the file a
spelling gate, which the Guard Proof Rule already warns is the weaker half. Where a
message IS the contract, the SSOT is imported rather than restated.

WHICH AXIS THIS COVERS — the Guard Proof Rule's own clause, asked of this file after it
let a fifth instance through. `_PATH_CASES` is a CONFINEMENT table: every row asks
whether a spelling lands inside the root, and every symlink row is a link pointing OUT.
Not one row asked what happens when the link stays inside and points at a file the export
POLICY excludes — and that is what shipped: `read_file("safe.txt")` returned `.env`'s
bytes on the native route while the local route refuses the same read, and a symlink
alias let `write_file` overwrite a protected artifact the local route protects. A door
that resolves inside the root satisfied every row here and still diverged.

So there are two tables now, and the second one is `_ALIAS_CASES`: for each native entry,
an ALIAS (symlink, hardlink, nested link) onto an excluded or protected path must get the
SAME refusal the direct spelling gets. The axis is named rather than assumed, per the
Document Truth Rule's first clause.
"""

from __future__ import annotations

import pathlib

import pytest


# ── the shared confinement question, asked of both path resolvers ────────────

def _local_refuses_path(base: pathlib.Path, rel: str) -> bool:
    """Does the LOCAL route refuse this workspace-relative spelling?

    The local resolver is exercised through its own two gates in the order a handler
    meets them (`utils.safe_relpath`, then resolve + `relative_to`), rather than through
    a whole tool call: a tool call drags in access profiles and protected-artifact rules
    that would let a DIFFERENT refusal stand in for the confinement one — exactly the
    masking `test_target_confinement_and_disclosure` keeps apart by hand.
    """

    from ouroboros.utils import safe_relpath

    try:
        resolved = (base.resolve(strict=False) / safe_relpath(rel)).resolve(strict=False)
        resolved.relative_to(base.resolve(strict=False))
    except (OSError, ValueError):
        return True
    return False


def _native_refuses_path(base: pathlib.Path, rel: str) -> bool:
    """Does the NATIVE route refuse it, through the mutation door?

    The MUTATION door specifically: it is the permissive one of the two by design (a
    write may name a path that does not exist yet), so if parity holds here it holds for
    `native_target` too.

    ``facts=None`` asks the CONFINEMENT question only, which is what this table is
    about. The policy axis is its own table below — mixing them here would let a policy
    refusal stand in for a confinement one, the masking `_local_refuses_path` already
    avoids in the other direction.
    """

    from ouroboros.workspace_native_paths import native_mutation_target

    try:
        native_mutation_target(base, rel, facts=None)
    except (OSError, ValueError):
        return True
    return False


def _workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('in')\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("OUTSIDE-SENTINEL\n", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside / "secret.txt")
    (root / "linked").symlink_to(outside)
    (root / "src" / "alias.py").symlink_to(root / "src" / "app.py")
    return root


_PATH_CASES = [
    pytest.param("../outside/secret.txt", True, id="lexical_traversal"),
    pytest.param("src/../../outside/secret.txt", True, id="traversal_mid_path"),
    pytest.param("escape.txt", True, id="symlink_final_component_out"),
    pytest.param("linked/secret.txt", True, id="symlink_directory_out"),
    pytest.param("linked/new.txt", True, id="new_file_under_symlinked_dir"),
    pytest.param("bad\x00name.txt", True, id="nul_byte"),
    pytest.param("bad\x01name.txt", True, id="control_character"),
    pytest.param("src/app.py", False, id="plain_inside"),
    pytest.param("src/new.txt", False, id="new_file_inside"),
    pytest.param("./src/app.py", False, id="tidy_spelling_inside"),
    pytest.param("src/alias.py", False, id="symlink_inside_is_followed"),
]


@pytest.mark.parametrize("rel,refused", _PATH_CASES)
def test_both_routes_agree_about_a_workspace_relative_path(tmp_path, rel, refused):
    """Same spelling, same verdict, on both routes — refusals AND acceptances.

    The acceptances matter as much as the refusals and are half the cases on purpose. A
    native door that refused everything would satisfy a refusal-only table while making
    the remote route useless, and the in-root-symlink case is the one that actually
    caught a wrong first draft of the fix: refusing an in-root link would have been a
    NEW asymmetry pointing the other way.
    """

    root = _workspace(tmp_path)
    local = _local_refuses_path(root, rel)
    native = _native_refuses_path(root, rel)

    assert local == refused, f"local route verdict changed for {rel!r}"
    assert native == local, (
        f"route parity broken for {rel!r}: "
        f"local {'refuses' if local else 'accepts'}, "
        f"native {'refuses' if native else 'accepts'}"
    )


def test_the_one_declared_asymmetry_is_the_native_route_being_stricter(tmp_path):
    """An ABSOLUTE spelling: native refuses it, local silently rebases it. Declared, not fixed.

    `utils.safe_relpath` does `lstrip("/")`, so the local route reads `/etc/passwd` as
    `<workspace>/etc/passwd` — a different file than the caller named, chosen without
    saying so. `native_relative_spelling` refuses instead. The two routes therefore
    disagree, and this is the ONE case in the table where they are allowed to, for two
    reasons: the divergence runs in the SAFE direction (the stricter side is the remote
    one, where the blast radius is someone else's machine), and closing it means changing
    a local resolver that predates the remote route and is relied on by every handler
    plus `normalize_root_relative`. That is a separate change with its own blast radius,
    not a rider on this one.

    Written as an assertion rather than a comment so the day someone aligns them, this
    test fails and the alignment is a decision instead of a surprise.
    """

    root = _workspace(tmp_path)
    assert _native_refuses_path(root, "/etc/passwd") is True
    assert _local_refuses_path(root, "/etc/passwd") is False, (
        "if the local route now refuses an absolute spelling too, delete this test and "
        "put `/etc/passwd` back in the parity table above"
    )


def test_the_parity_check_can_actually_disagree(tmp_path):
    """The comparison must be shown FAILING, or it proves nothing (Guard Proof Rule).

    A parity assertion between two functions that happen to be the same function would
    be green forever. So one side is replaced by a resolver that accepts everything, and
    the disagreement is asserted.
    """

    root = _workspace(tmp_path)

    def permissive(_base, _rel):
        return False

    assert _local_refuses_path(root, "escape.txt") is True
    assert permissive(root, "escape.txt") != _local_refuses_path(root, "escape.txt")


# ── the same rule at the operation level ─────────────────────────────────────

_MUTATION_OPERATIONS = [
    pytest.param(
        "write_file",
        {"path": "escape.txt", "content": "X\n", "mode": "append"},
        id="write_file_append",
    ),
    pytest.param(
        "write_file",
        {"path": "escape.txt", "content": "X\n", "mode": "overwrite"},
        id="write_file_overwrite",
    ),
    pytest.param(
        "write_file",
        {"files": [{"path": "escape.txt", "content": "X\n"}]},
        id="write_file_batch_row",
    ),
    pytest.param(
        "edit_text",
        {"path": "escape.txt", "old_str": "OUTSIDE-SENTINEL", "new_str": "Y"},
        id="edit_text",
    ),
]


@pytest.mark.parametrize("tool,args", _MUTATION_OPERATIONS)
def test_every_native_mutation_operation_refuses_what_the_local_route_refuses(
    tmp_path, tool, args
):
    """Operation level, not helper level: the refusal has to reach the caller.

    A confined helper whose refusal is swallowed into a success envelope is the same
    defect from the model's point of view. So the assertion is on the ENVELOPE a remote
    caller receives — a typed `permission_denied` reporting that nothing ran — and on
    the bytes of the file outside the workspace, which is the only unfalsifiable half.
    """

    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    victim = tmp_path / "outside" / "secret.txt"
    before = victim.read_bytes()

    assert _local_refuses_path(root, "escape.txt"), "the local route must refuse this"
    envelope = execute_native_operation(root, tool, args).envelope

    assert victim.read_bytes() == before, "the file outside the workspace was modified"
    assert envelope.diagnostic is not None, envelope.text
    assert envelope.diagnostic.code == "permission_denied", envelope.diagnostic
    assert envelope.diagnostic.completion == "not_started", (
        "a refused mutation must report that nothing ran"
    )


# ── the POLICY axis: an alias must be refused wherever the direct name is ────
#
# `_PATH_CASES` above is confinement only. This table is the axis that was missing, and
# it is written as (how the alias is built) × (which native entry reads or writes it), so
# a new native entry has to appear here or the sweep below notices.

# The nesting axis has TWO ENDS, and this table only ever had one. Every kind nested the
# ALIAS and left the SECRET at `root / <name>`, so `nested_hardlink` proved only that a deep
# alias to a ROOT secret is caught — which the root-bounded probe managed anyway. The cells
# that actually leaked are the ones where the SECRET is nested: on a branch whose version of
# this table was green, `read_file` on a root-level hardlink to `sub/.env` returned the bytes
# verbatim. A kind therefore names TWO positions now, and `crossing` is each direction of
# the two differing.
_ALIAS_KINDS = (
    "symlink", "hardlink",
    "nested_symlink", "nested_hardlink",
    "nested_secret_symlink", "nested_secret_hardlink",
    "crossing_down_symlink", "crossing_down_hardlink",
    "crossing_up_symlink", "crossing_up_hardlink",
)

# Two tables, because the document answers two DIFFERENT questions and conflating them
# would assert a divergence rather than parity. The credential-name classes govern what
# may LEAVE the host, so they judge reads; the protected-artifact class governs what may
# be WRITTEN. A task writing its own `.env` is ordinary on a local placement, so refusing
# it remotely would be a new asymmetry — the mistake this file's own docstring warns about.
#
# Every one of these four operations is in `REMOTE_NATIVE_OPERATION_MODULE`, so every one
# is live on any ssh placement under the default profile.
# The multi-file editors are in BOTH tables, and that correction is the finding. Their
# ACT is a write, so they belong to the protected-artifact half; their MEANS is a read —
# a hunk is LOCATED by matching the file's bytes, a batch row is VALIDATED by counting
# them — so they belong to the credential-name half too. Listing them under `write` alone
# is exactly the model that let the defect ship: the table said what the tools DO instead
# of what they TOUCH, so a green suite sat over an editor answering "context is ambiguous
# in .env — matches at line 3, line 7" about a file whose `read_file` the same task had
# just been refused. One-file patches keep each cell about the alias, not about parsing.
_ALIAS_EDITOR_ENTRIES = {
    "apply_patch": lambda rel: {
        "patch": (
            "*** Begin Patch\n"
            f"*** Update File: {rel}\n"
            "@@\n"
            "-SENTINEL-SECRET\n"
            "+TAMPERED\n"
            "*** End Patch\n"
        ),
    },
    "edit_batch": lambda rel: {
        "edits": [{"path": rel, "old_str": "SENTINEL", "new_str": "X", "count": 1}],
    },
}
_ALIAS_READ_ENTRIES = {
    "read_file": lambda rel: {"path": rel},
    # The alias's OWN directory, not a hardcoded `"."`: with the alias nested, listing the
    # root asks about a directory the alias is not in, and the cell passes by looking
    # somewhere else. That is the same blindness as nesting only one end of the axis.
    "list_files": lambda rel: {"path": rel.rsplit("/", 1)[0] if "/" in rel else "."},
    **_ALIAS_EDITOR_ENTRIES,
}
_ALIAS_WRITE_ENTRIES = {
    "write_file": lambda rel: {"path": rel, "content": "TAMPERED\n", "mode": "overwrite"},
    "edit_text": lambda rel: {"path": rel, "old_str": "SENTINEL", "new_str": "X"},
    **_ALIAS_EDITOR_ENTRIES,
}
_ALIAS_ENTRIES = {**_ALIAS_READ_ENTRIES, **_ALIAS_WRITE_ENTRIES}


def _alias_workspace(tmp_path: pathlib.Path, kind: str, secret_name: str):
    """A workspace whose `<alias>` is another name for `<secret_name>`, inside the root.

    Every alias stays INSIDE the workspace on purpose: an alias that pointed out would be
    refused by the confinement rule and prove nothing about the policy — the masking this
    file's docstring warns about, in the direction that flatters the code.

    Both POSITIONS vary, and that is the correction. The kind decides where the alias
    lives AND where the secret lives:

    * ``symlink``/``hardlink`` — both at the root;
    * ``nested_*`` — the alias is nested, the secret is at the root;
    * ``nested_secret_*`` — both nested, in the same directory;
    * ``crossing_down_*`` — the alias at the root, the secret nested;
    * ``crossing_up_*`` — the alias nested, the secret at the root's own depth but in a
      different subtree, so neither is a sibling of the other.

    The last three groups did not exist, and they are exactly where the shipped leak was.
    """

    import os

    root = tmp_path / "workspace"
    (root / "sub").mkdir(parents=True)
    (root / "other").mkdir(parents=True)
    nested_secret = "nested_secret" in kind or "crossing_down" in kind
    nested_alias = kind.startswith("nested") or "crossing_up" in kind
    if "crossing_up" in kind:
        secret = root / "other" / secret_name
    elif nested_secret:
        secret = root / "sub" / secret_name
    else:
        secret = root / secret_name
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("SENTINEL-SECRET\n", encoding="utf-8")
    alias = root / ("sub/alias.txt" if nested_alias else "alias.txt")
    if "hardlink" in kind:
        os.link(secret, alias)
    else:
        os.symlink(secret, alias)
    # The secret's own workspace-relative spelling comes back too, because a caller that
    # hardcoded `protected_paths=["golden.bin"]` was silently asserting the root position.
    return (
        root,
        alias.relative_to(root).as_posix(),
        secret,
        secret.relative_to(root).as_posix(),
    )


def _native_envelope(root: pathlib.Path, tool: str, args: dict, facts: dict):
    from ouroboros.workspace_native import execute_native_operation

    return execute_native_operation(root, tool, args, native_facts=facts).envelope


@pytest.mark.parametrize("kind", _ALIAS_KINDS)
@pytest.mark.parametrize("tool", sorted(_ALIAS_READ_ENTRIES))
def test_an_alias_to_an_excluded_path_is_refused_like_the_direct_name(tmp_path, kind, tool):
    """`safe.txt -> .env` must answer the way `.env` answers, on every reading entry.

    Asserted on the BYTES in the answer, not only on the envelope: the shipped defect
    returned a SUCCESSFUL envelope whose text WAS the secret, so an envelope-only
    assertion would have passed on it. This is the exact live reproduction — the direct
    read was refused and the alias returned `SECRET_TOKEN=hunter2`.
    """

    from ouroboros.export_policy_contract import build_policy_document

    root, alias_rel, _secret, secret_rel = _alias_workspace(tmp_path, kind, ".env")
    facts = {"export_policy": build_policy_document(channel="workspace_query")}

    # The secret's OWN spelling, not a hardcoded `.env`: half these layouts nest it, so
    # the hardcoded name asked about a path that is not there — and the answer a tool
    # gives about an ABSENT file cannot establish what it gives about the real one.
    direct = _native_envelope(root, tool, _ALIAS_READ_ENTRIES[tool](secret_rel), facts)
    aliased = _native_envelope(root, tool, _ALIAS_READ_ENTRIES[tool](alias_rel), facts)

    assert "SENTINEL-SECRET" not in aliased.text, (
        f"{tool} through a {kind} alias returned the excluded file's bytes"
    )
    if tool == "list_files":
        # A tree channel discloses instead of refusing, so the assertion is that the
        # alias is NOT offered as an ordinary name a later read could follow.
        assert "POLICY_FILTERED" in aliased.text, (
            "the listing offered an alias to an excluded file as an ordinary entry"
        )
        return
    # BOTH refused, not merely AGREEING. Equality alone is satisfied by two successes,
    # which is what the editors did before they asked the read question: the direct name
    # and the alias both went through, identically, and the cell was green.
    assert direct.diagnostic is not None, (
        f"{tool} answered about the excluded file under its own name: {direct.text}"
    )
    assert aliased.diagnostic is not None, (
        f"{tool}: the direct spelling was refused and the {kind} alias was not"
    )


@pytest.mark.parametrize("kind", _ALIAS_KINDS)
@pytest.mark.parametrize("tool", sorted(_ALIAS_WRITE_ENTRIES))
def test_an_alias_to_a_protected_artifact_cannot_be_written_through(tmp_path, kind, tool):
    """The write half, which had no policy applier behind the door at all.

    `refuse_protected_mutation` judged `args["path"]` and `native_mutation_target` then
    followed the in-root link and wrote at the far end — so `golden.bin` really became
    `TAMPERED` under a name the policy had never heard of. Overwrite, append and a batch
    row are all here because the three took different code paths to the same door.
    """

    from ouroboros.export_policy_contract import build_policy_document
    from ouroboros.workspace_native import execute_native_operation

    root, alias_rel, secret, secret_rel = _alias_workspace(tmp_path, kind, "golden.bin")
    facts = {
        "export_policy": build_policy_document(
            channel="workspace_snapshot", protected_paths=[secret_rel]
        )
    }
    variants = [_ALIAS_WRITE_ENTRIES[tool](alias_rel)]
    if tool == "write_file":
        variants += [
            {"path": alias_rel, "content": "TAMPERED\n", "mode": "append"},
            {"files": [{"path": alias_rel, "content": "TAMPERED\n"}]},
        ]
    for args in variants:
        envelope = execute_native_operation(root, tool, args, native_facts=facts).envelope
        assert secret.read_text(encoding="utf-8") == "SENTINEL-SECRET\n", (
            f"a protected artifact was mutated through a {kind} alias: {args}"
        )
        assert envelope.diagnostic is not None, envelope.text


def test_every_native_path_entry_is_in_the_alias_table():
    """The table may not fall behind the operation registry.

    A per-entry test only ever covers the entries somebody thought of, and the shipped
    defect was on four entries at once. So the set is derived from the routing table: EVERY
    native operation must appear in the alias table, or be named in one of the two
    registries below with the reason it does not belong there.

    This test was a TAUTOLOGY and a paid reviewer proved it: `path_entries` was built by
    FILTERING the registry down to the names already classified, so `missing` — the names
    in `path_entries` that are not classified — was the empty set by construction, for any
    registry. A newly registered path-bearing operation passed it. The completeness claim
    is now over the WHOLE registry, and the reviewer's own counter-case is asserted below:
    an unclassified name must make this fail.
    """

    from ouroboros.workspace_native_contract import REMOTE_NATIVE_OPERATION_MODULE

    # Path-taking native operations whose alias question is answered elsewhere, with the
    # reason — a registry of decisions, not an accumulating exemption list.
    answered_elsewhere = {
        "search_code": "the walk skips every symlink (`_search_skippable`) and judges "
                       "each real path it reads through `judged_exclusion`",
        "query_code": "same walk, same judge",
        "extract_video_frames": "judged at prepare AND at execute through `native_target` "
                                "on the `media_frames` channel; see test_remote_export_policy",
        "classify_ambiguous_workspace_path": "classifies an absolute spelling and reads "
                                            "no bytes",
        "verify_remote_check": "`bytes_equal` is judged through `refuse_excluded_target` "
                               "AFTER resolution; see test_export_alias_identity_matrix",
        "snapshot_manifest_and_blob_export": "the snapshot walk judges every entry with "
                                             "`judged_exclusion`; see the same matrix",
        "guarded_patch_apply": "`_validated_changes` refuses a change naming an aliased "
                               "policy path before any row is restored",
        "vcs_status": "filtered by `policy_excluded_git_paths`, which judges identity",
        "vcs_diff": "same pathspec excludes, same judge",
        "execute_reviewed_payload": "stages under a fresh mkdtemp root, not the workspace",
    }
    # Operations whose arguments carry no workspace SOURCE path at all. Declared, because
    # "not in the alias table" has to be a decision rather than an omission.
    no_source_path = {
        "run_command": "a cwd, judged with QUESTION_NONE — a process running in a "
                       "directory exports no bytes",
        "run_script": "same cwd; the script body is inline, not a workspace path",
        "start_service": "same cwd plus a service name",
        "service_status": "a service id",
        "service_logs": "a service id; the log path is execd-owned, not model-named",
        "stop_service": "a service id",
    }
    classified = {*_ALIAS_ENTRIES, *answered_elsewhere, *no_source_path}
    missing = sorted(set(REMOTE_NATIVE_OPERATION_MODULE) - classified)
    assert not missing, (
        f"native operations with no alias classification: {missing} — add them to "
        "_ALIAS_ENTRIES, or name the reason in `answered_elsewhere` / `no_source_path`"
    )
    assert set(_ALIAS_ENTRIES) <= set(REMOTE_NATIVE_OPERATION_MODULE), (
        "the alias table names an operation the routing table does not"
    )
    stale = sorted((set(answered_elsewhere) | set(no_source_path)) - set(
        REMOTE_NATIVE_OPERATION_MODULE
    ))
    assert not stale, (
        f"a reason is registered for an operation that no longer exists: {stale} — a dead "
        "exemption pardons the next operation that inherits the name"
    )


def test_the_completeness_gate_would_notice_an_unclassified_operation():
    """The gate above, shown FAILING — because it did not, for any registry at all.

    `verify_gates.py` in the review evidence put it plainly: "the set is filtered to the
    classified names BEFORE the check, so `missing` cannot be non-empty for any registry.
    The assertion is a tautology." A completeness gate that cannot fail is the Guard Proof
    Rule's own vacuous guard, sitting on top of the alias table it was meant to protect.
    """

    from ouroboros.workspace_native_contract import REMOTE_NATIVE_OPERATION_MODULE

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(
            REMOTE_NATIVE_OPERATION_MODULE,
            "brand_new_path_operation",
            "ouroboros.workspace_native",
        )
        with pytest.raises(AssertionError, match="brand_new_path_operation"):
            test_every_native_path_entry_is_in_the_alias_table()


# ── parity in the ARGUMENT-SHAPE rules, not only in path confinement ─────────

def test_the_service_name_rule_has_exactly_one_owner():
    """Both routes read the SAME pattern object, so they cannot drift again.

    Not "both regexes match" — both routes must be the same object. `tools/services`
    kept its own copy and refused, while the target's `re.sub` rewrote; two spellings of
    one rule is how that happened, so identity is the assertion.
    """

    from ouroboros.tools.services import _SERVICE_NAME_RE
    from ouroboros.workspace_native_contract import SERVICE_NAME_PATTERN

    assert _SERVICE_NAME_RE is SERVICE_NAME_PATTERN


@pytest.mark.parametrize("name", ["a/b", "x" * 81, "bad name", "../etc", "a\tb"])
def test_a_service_name_is_refused_identically_on_both_routes(tmp_path, name):
    """The native route refuses the illegal name instead of sanitizing it."""

    from ouroboros.tools.services import _sanitize_service_name
    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    _, local_error = _sanitize_service_name(name)
    assert local_error, f"the local route must refuse {name!r}"

    envelope = execute_native_operation(
        root, "start_service", {"name": name, "cmd": ["true"], "cwd": "."}
    ).envelope
    assert envelope.diagnostic is not None, envelope.text
    assert "[A-Za-z0-9_.-]" in envelope.text, envelope.text


@pytest.mark.parametrize("value", [-1, -0.5])
def test_a_negative_readiness_timeout_is_refused_on_both_routes(tmp_path, value):
    """Clamping a nonsensical timeout to zero looks like a service that came up instantly.

    The local route refuses it (`services._readiness_timeout`); the native route used to
    clamp it into `max(0.0, ...)`, so the readiness check silently never waited. The 25 s
    CEILING is a real bound and stays a clamp on both sides — only the negative case is a
    refusal.
    """

    from ouroboros.tools.services import _readiness_timeout
    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    _, local_error = _readiness_timeout({"timeout_sec": value})
    assert local_error, "the local route must refuse a negative readiness timeout"

    envelope = execute_native_operation(
        root,
        "start_service",
        {"name": "svc", "cmd": ["true"], "cwd": ".", "readiness": {"timeout_sec": value}},
    ).envelope
    assert envelope.diagnostic is not None, envelope.text
    assert "non-negative" in envelope.text, envelope.text
