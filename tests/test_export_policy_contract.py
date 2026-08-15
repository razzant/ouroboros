"""The ONE export policy: its document, its hash, and its source-side application.

These tests are about the property the module exists for — that every export door
in `ouroboros/` reaches the same predicate, so a rule cannot hold on one channel
and lapse on the next (the postmortem's "one policy × N doors" class).
"""

from __future__ import annotations

import ast
import pathlib
import subprocess

import pytest

from ouroboros.export_policy_contract import (
    EXPORT_CHANNELS,
    EXPORT_POLICY_VERSION,
    MAX_DISCLOSED_EXCLUSIONS,
    PROFILE_DELIVERABLE,
    REASON_EXCLUDED_DIRECTORY,
    REASON_PROTECTED_ARTIFACT,
    REASON_SENSITIVE_COMPONENT,
    REASON_SENSITIVE_FILE,
    ExportChannelUnknownError,
    ExportPolicyExcludedError,
    build_policy_document,
    canonical_policy_bytes,
    QUESTION_EXPORT,
    judged_exclusion,
    channel_profile,
    export_disclosure_block,
    export_policy_hash,
    normalize_export_policy,
    policy_from_facts,
    refuse_excluded_target,
    unaliased_exclusion,
)

REPO = pathlib.Path(__file__).resolve().parent.parent


def _tree_policy(*protected: str) -> dict:
    return build_policy_document(channel="workspace_snapshot", protected_paths=protected)


# ── the closed channel registry ──────────────────────────────────────────────


def test_an_undeclared_blob_kind_fails_closed():
    """A blob kind nobody declared has no policy attached, so it cannot cross."""

    with pytest.raises(ExportChannelUnknownError):
        channel_profile("mystery_blob")
    with pytest.raises(ExportChannelUnknownError):
        build_policy_document(channel="")
    for channel in EXPORT_CHANNELS:
        assert channel_profile(channel) in {"tree", PROFILE_DELIVERABLE}


def test_every_declared_channel_has_a_complete_row():
    for channel, row in EXPORT_CHANNELS.items():
        assert set(row) == {"profile", "path_bearing"}, channel
        assert isinstance(row["path_bearing"], bool), channel


# ── the document and its hash ────────────────────────────────────────────────


def test_the_hash_binds_the_rules_and_the_protected_paths():
    a = _tree_policy("docs/BIBLE.md")
    b = _tree_policy("docs/BIBLE.md")
    c = _tree_policy("docs/OTHER.md")
    assert export_policy_hash(a) == export_policy_hash(b)
    assert export_policy_hash(a) != export_policy_hash(c)
    # Order and duplication cannot change the identity of one policy.
    assert export_policy_hash(_tree_policy("b", "a", "a")) == export_policy_hash(
        _tree_policy("a", "b")
    )
    assert canonical_policy_bytes(a) == canonical_policy_bytes(b)


@pytest.mark.parametrize(
    "bad",
    [
        {"version": 99, "channel": "workspace_snapshot"},
        {"version": EXPORT_POLICY_VERSION, "channel": "not_a_channel"},
        {"version": EXPORT_POLICY_VERSION, "channel": "workspace_snapshot", "surprise": 1},
        {"version": EXPORT_POLICY_VERSION, "channel": "workspace_snapshot", "profile": "loose"},
        {
            "version": EXPORT_POLICY_VERSION,
            "channel": "workspace_snapshot",
            "protected_paths": "docs",
        },
        {
            "version": EXPORT_POLICY_VERSION,
            "channel": "workspace_snapshot",
            "protected_paths": ["../escape"],
        },
    ],
)
def test_a_document_this_build_cannot_claim_to_have_applied_is_refused(bad):
    with pytest.raises(ValueError):
        normalize_export_policy(bad)


def test_an_unbound_operation_is_visibly_unbound_not_quietly_defaulted():
    assert policy_from_facts({}) is None
    assert policy_from_facts(None) is None
    document = _tree_policy()
    assert policy_from_facts({"export_policy": document}) == document


# ── the evaluator ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path,reason",
    [
        (".env", REASON_SENSITIVE_FILE),
        ("app/.env.production", REASON_SENSITIVE_FILE),
        ("keys/id_ed25519", REASON_SENSITIVE_FILE),
        ("keys/id_rsa.pub", REASON_SENSITIVE_FILE),
        ("home/.ssh/config", REASON_SENSITIVE_FILE),
        ("conf/credentials.json", REASON_SENSITIVE_FILE),
        ("certs/server.pem", REASON_SENSITIVE_FILE),
        (".netrc", REASON_SENSITIVE_FILE),
        (".git/config", REASON_EXCLUDED_DIRECTORY),
        ("pkg/__pycache__/x.pyc", REASON_EXCLUDED_DIRECTORY),
        ("src/main.py", ""),
        ("README.md", ""),
    ],
)
def test_the_tree_profile_answers_one_way_for_every_channel(path, reason):
    assert unaliased_exclusion(path, _tree_policy())[0] == reason


def test_a_protected_path_is_named_as_protected_even_when_it_also_looks_sensitive():
    document = _tree_policy("secrets/.env")
    assert unaliased_exclusion("secrets/.env", document)[0] == REASON_PROTECTED_ARTIFACT
    assert (
        unaliased_exclusion("secrets/.env/inner", document)[0]
        == REASON_PROTECTED_ARTIFACT
    )
    # A prefix that is not a path component boundary must not match.
    assert unaliased_exclusion("secretsother/file.txt", document)[0] == ""


def test_an_unrepresentable_path_is_not_guessed_about():
    assert unaliased_exclusion("", _tree_policy())[0] == REASON_SENSITIVE_FILE
    assert unaliased_exclusion("../outside", _tree_policy())[0] == REASON_SENSITIVE_FILE


def test_the_deliverable_profile_is_strictly_stronger_than_the_tree_profile():
    tree = _tree_policy()
    deliverable = build_policy_document(channel="declared_output")
    # A credential-shaped component is incidental in a tree walk and disqualifying
    # in something the model chose to publish.
    assert unaliased_exclusion("app/secrets/notes.txt", tree)[0] == ""
    assert (
        unaliased_exclusion("app/secrets/notes.txt", deliverable)[0]
        == REASON_SENSITIVE_COMPONENT
    )
    assert unaliased_exclusion("build/api_key_dump", deliverable)[0] == (
        REASON_SENSITIVE_COMPONENT
    )
    assert unaliased_exclusion("build/index.html", deliverable)[0] == ""
    # Everything the tree profile rejects, the deliverable profile also rejects.
    for path in (".env", "keys/id_rsa", "home/.aws/credentials", ".git/config"):
        assert unaliased_exclusion(path, deliverable)[0]


# ── the disclosure block (D7) ────────────────────────────────────────────────


def test_the_disclosure_block_is_present_even_when_nothing_was_filtered():
    facts = {"export_policy": _tree_policy()}
    clean = export_disclosure_block(facts, [])["export_policy"]
    assert clean["complete"] is True
    assert clean["integrity_complete"] is True
    assert clean["policy_scope"] == "full"
    assert clean["excluded_count"] == 0
    assert clean["policy_hash"] == export_policy_hash(_tree_policy())


def test_the_count_is_exact_while_the_list_is_bounded():
    rows = [{"path": f"f{index}/.env", "reason": REASON_SENSITIVE_FILE} for index in range(500)]
    block = export_disclosure_block({"export_policy": _tree_policy()}, rows)["export_policy"]
    assert block["excluded_count"] == 500
    assert len(block["excluded"]) == MAX_DISCLOSED_EXCLUSIONS
    assert block["excluded_disclosure_truncated"] is True
    # A policy exclusion is never an integrity failure: that distinction is the
    # whole of D7 and a caller must be able to read it off the block.
    assert block["complete"] is False
    assert block["integrity_complete"] is True
    assert block["policy_scope"] == "policy_filtered"


def test_a_single_source_channel_refuses_rather_than_succeeding_emptily(tmp_path):
    facts = {"export_policy": build_policy_document(channel="media_frames")}
    root = tmp_path.resolve()
    with pytest.raises(ExportPolicyExcludedError) as excinfo:
        refuse_excluded_target(
            root, None, "assets/.env", facts,
            question=QUESTION_EXPORT, channel="media_frames",
        )
    assert "REMOTE_EXPORT_POLICY_EXCLUDED" in str(excinfo.value)
    refuse_excluded_target(
        root, None, "assets/clip.mp4", facts,
        question=QUESTION_EXPORT, channel="media_frames",
    )
    with pytest.raises(ExportChannelUnknownError):
        refuse_excluded_target(
            root, None, "assets/clip.mp4", facts,
            question=QUESTION_EXPORT, channel="undeclared_channel",
        )


# ── the de-duplication proof ─────────────────────────────────────────────────

# The rule strings that used to be copied across the export doors. The scope of the
# assertion is the DOORS — the modules that construct or import cross-boundary
# blobs — not every module in the tree: Home-side guards answering a different
# question (may this task READ this Home path) legitimately name some of the same
# strings, and folding them in would make the test about spelling instead of about
# the boundary.
_POLICY_LITERALS = (".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519", ".gnupg")
# Home-side export doors, which have no single registry to derive from. Kept as a list, but
# with the two properties the reviewer's `verify_gates.py` found missing: every entry must
# EXIST (a renamed module used to be skipped silently by `if not path.exists(): continue`)
# and the target-side half is not listed here at all — it is DERIVED from
# `tool_capabilities.remote_native_import_closure`, so a new kernel module is in scope the
# moment it can run on the target, whether or not anyone remembers this file.
_HOME_EXPORT_DOOR_MODULES = (
    "ouroboros/remote_export_policy.py",
    "ouroboros/remote_transfer.py",
    "ouroboros/remote_task_files.py",
)


# Two modules in the closure are NOT export doors, and each one says why. This is a
# registry of decisions, and the second entry carries a mechanical compensating check
# below rather than a promise.
_NOT_AN_EXPORT_DOOR = {
    "ouroboros/export_policy_contract.py":
        "the table's own home — the one place the strings are allowed to live",
    "ouroboros/code_intelligence.py":
        "the LOCAL inventory's own hygiene rule (`_is_sensitive_inventory_path`), which "
        "keeps a secret out of a code index on a placement that has no export document at "
        "all. It is not an export decision: on the remote route the admitted set is "
        "computed by the policy and handed to the builder as `exclude_paths` BEFORE it "
        "reads anything — asserted by "
        "test_the_inventory_walk_is_handed_the_policys_admitted_set_not_a_private_rule",
}


def _export_door_modules() -> tuple[str, ...]:
    """Every module that can construct a cross-boundary blob, from the closure + Home."""

    from ouroboros.tool_capabilities import remote_native_import_closure

    target_side = {
        pathlib.Path(*module.split(".")).with_suffix(".py").as_posix()
        for module in remote_native_import_closure(REPO)["modules"]
        if module.startswith("ouroboros.")
    }
    return tuple(
        sorted({*target_side, *_HOME_EXPORT_DOOR_MODULES} - set(_NOT_AN_EXPORT_DOOR))
    )


def test_the_inventory_walk_is_handed_the_policys_admitted_set_not_a_private_rule():
    """`query_code` must not READ what the policy excludes, then filter the rows after.

    The reviewer's finding, and it is an ORDER defect rather than a missing check:
    `build_code_inventory` was handed `protected_paths` only, so every `id_rsa`,
    `credentials.json` and `.netrc` in the tree was opened and parsed — symbols, imports
    and routes extracted — and `visible` dropped the rows afterwards. Derived content from
    an excluded file had already been computed on the source side.

    `code_intelligence` has its own `.env` rule, which is why the leak was partial and why
    that module is exempted from the literal gate above. This test is the compensating
    half of that exemption: the policy's own excluded set reaches the builder.
    """

    import os
    import tempfile

    import ouroboros.workspace_query_native as module
    from ouroboros.workspace_query_native import query_workspace

    seen: dict[str, object] = {}
    real = module.build_code_inventory

    def capture(root_arg, **kwargs):
        seen["exclude_paths"] = [str(item) for item in (kwargs.get("exclude_paths") or [])]
        return real(root_arg, **kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp) / "ws"
        (work / "keys").mkdir(parents=True)
        (work / "keys" / "id_rsa").write_text("PRIVATE\n", encoding="utf-8")
        (work / ".env").write_text("SECRET_TOKEN=x\n", encoding="utf-8")
        (work / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
        os.system(f"cd {work} && git init -q && git add -A")  # noqa: S605 - fixture only
        module.build_code_inventory = capture
        try:
            query_workspace(
                work,
                {"op": "symbols", "path": "."},
                policy=build_policy_document(channel="workspace_query"),
            )
        finally:
            module.build_code_inventory = real
    excluded = {pathlib.Path(item).name for item in seen.get("exclude_paths") or []}
    assert {"id_rsa", ".env"} <= excluded, (
        f"the inventory builder was handed {sorted(excluded)} — a path the policy excludes "
        "must be kept out of the WALK, not filtered out of the rows it produced"
    )


def test_no_export_door_carries_rule_literals_of_its_own():
    r"""No export door keeps a private copy of the credential-name table.

    BOUNDARY: matches STRING CONSTANTS in ASSIGNMENTS and CONTAINERS only, so a literal
    assembled at runtime (`"." + "env"`, an f-string, `f".{ext}"`), a bytes literal, a
    regex spelling of the same rule (`r"\.env$"`), and a rule table living in a data file
    all pass. DOCSTRINGS and comments are excluded deliberately: a door that EXPLAINS the
    rule it delegates is documentation, and flagging it would gate prose — the previous
    version collected every `ast.Constant` and would have failed on its own explanations.

    The door set is DERIVED, not listed. `_EXPORT_DOOR_MODULES` was a hand-written list of
    nine with `if not path.exists(): continue`, so a renamed or newly added door was
    skipped in silence — a vacuous guard by this repo's own rule, which the reviewer's
    `verify_gates.py` printed in those words. It comes from the execd import closure now,
    plus the three Home-side doors that no registry can derive, and every listed path must
    exist.
    """

    doors = _export_door_modules()
    assert len(doors) >= 20, f"the door set collapsed to {len(doors)} — derivation broke"
    missing = [name for name in _HOME_EXPORT_DOOR_MODULES if not (REPO / name).exists()]
    assert not missing, (
        f"a declared Home-side export door does not exist: {missing} — a skipped entry "
        "pardons the module that replaced it"
    )
    offenders: list[str] = []
    for relative in doors:
        path = REPO / relative
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            node.body[0].value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node not in docstrings
        }
        hits = sorted(literals & set(_POLICY_LITERALS))
        if hits:
            offenders.append(f"{relative}: {hits}")
    assert not offenders, (
        "an export door grew its own copy of the rule table — one policy, one "
        "table, asked through export_policy_contract:\n" + "\n".join(offenders)
    )


def test_no_module_reintroduces_a_private_sensitive_name_table():
    """A private copy of the tables is how the doors drifted the first time."""

    proc = subprocess.run(
        [
            "grep", "-rn", "--include=*.py",
            r"_SENSITIVE_OUTPUT_NAMES\|_SENSITIVE_PATCH_NAMES\|_SENSITIVE_OUTPUT_MARKERS",
            "ouroboros/",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "", (
        "a per-channel sensitive-name table came back:\n" + proc.stdout
    )


# ── one judgement: no door may read a subset of the document ──────────────────


def test_the_verdict_pairs_the_reason_and_the_sentence_from_one_evaluation():
    """A reason and a sentence that came from different rules is the shipped defect.

    The declared-output door asked `component_exclusion_reason` for the verdict and
    `describe_component_exclusion` for the text — one rule group, twice — so a document
    carrying BOTH the credential-name rules and Home's protected paths was applied on
    neither. Both public doors return the pair, so they cannot disagree.
    """

    deliverable = build_policy_document(
        channel="declared_output", protected_paths=["dist/golden.bin"]
    )
    for path, reason in (
        ("dist/golden.bin", REASON_PROTECTED_ARTIFACT),
        ("dist/id_rsa", REASON_SENSITIVE_FILE),
        ("dist/secrets/x.txt", REASON_SENSITIVE_COMPONENT),
    ):
        verdict, sentence = unaliased_exclusion(
            path, deliverable, question=QUESTION_EXPORT
        )
        assert verdict == reason, path
        assert sentence.startswith(path + ":"), sentence
        assert judged_exclusion(
            "/nonexistent-root", None, path, deliverable, question=QUESTION_EXPORT
        )[:2] == (verdict, sentence), (
            "the spelling door and the identity door must be the same judgement, not two"
        )
    assert unaliased_exclusion("dist/report.txt", deliverable) == ("", "")


def test_no_rule_group_is_reachable_without_the_whole_ladder():
    """The structural half: a door cannot select a subset even if it wants to.

    The sub-rules AND the ladder itself are private, so the only public ways to judge a
    path are the two doors. This is asserted rather than trusted because "everyone asks
    the right function" was true of six doors and false of the seventh.
    """

    import ouroboros.export_policy_contract as contract

    public = {name for name in contract.__all__ if not name.startswith("_")}
    assert "component_exclusion_reason" not in public
    assert "describe_component_exclusion" not in public
    assert "named_source_exclusion_reason" not in public, (
        "a second evaluator that answers a NARROWER question is a rule group with its "
        "own public door; the narrowing is a `question` argument now"
    )
    for private in ("_exclusion_reason", "_offending_component", "_marker_scoped_name"):
        assert private not in public and not hasattr(contract, private.lstrip("_")), (
            f"{private} is the ladder or one of its rule groups; a public spelling of it "
            "is the way around the mandatory `question` argument — eight call sites "
            "reached `exclusion_reason` directly while the doors were compulsory"
        )
    judging = {
        name for name in public
        if "exclusion" in name or "verdict" in name or "excluded" in name
    }
    assert judging == {
        "judged_exclusion", "unaliased_exclusion", "describe_exclusion",
        "refuse_excluded_target",
    }, (
        f"a new public judging entry point appeared: {sorted(judging)} — it must delegate "
        "to `_exclusion_reason` or it is the seventh door again"
    )


# ── ONE mechanic for identity: the gate that makes the weaker door unreachable ─


def _judging_names_referenced(path: pathlib.Path) -> set[str]:
    """Every judging identifier a module NAMES, at any scope, imported or attributed."""

    watched = {
        "judged_exclusion", "unaliased_exclusion", "refuse_excluded_target",
        "_exclusion_reason", "exclusion_reason", "exclusion_verdict",
        "excluded_inodes", "_hardlink_aliases",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in watched:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in watched:
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update(alias.name for alias in node.names if alias.name in watched)
    return found


def test_the_spelling_only_door_is_unreachable_from_the_target():
    """The gate that makes ONE mechanic structural rather than a habit.

    The shipped defect was not a missing check — it was TWO answers to one question. The
    walks seeded a recursive inode set; the single-source doors used a root-only
    `scandir`; and the declared-output door called the spelling evaluator with no identity
    check at all. Every leak lived in the weaker mechanic.

    So the rule is: a module that can run on the TARGET may name exactly one judging
    function, `judged_exclusion` (or `refuse_excluded_target`, which is that plus a
    raise). `unaliased_exclusion` — the spelling-only door — may only be named by modules
    that CANNOT run on the target, and "cannot" is computed, not asserted: the authority
    is `tool_capabilities.remote_native_import_closure`, the same closure the bundle
    isolation gate uses. A future door that reaches for the weaker question either fails
    HERE, or it moves itself out of the closure and fails the isolation gate instead.
    """

    from ouroboros.tool_capabilities import remote_native_import_closure

    closure = set(remote_native_import_closure(REPO)["modules"])
    contract_module = "ouroboros.export_policy_contract"
    assert contract_module in closure, (
        "the contract itself must travel to the target, or none of this applies"
    )
    offenders: dict[str, set[str]] = {}
    spelling_callers: set[str] = set()
    for path in sorted((REPO / "ouroboros").rglob("*.py")):
        module = ".".join(path.relative_to(REPO).with_suffix("").parts)
        if module == contract_module:
            continue
        named = _judging_names_referenced(path)
        if not named:
            continue
        if "unaliased_exclusion" in named:
            spelling_callers.add(module)
        stale = named & {
            "exclusion_reason", "exclusion_verdict", "excluded_inodes",
            "_exclusion_reason", "_hardlink_aliases",
        }
        forbidden = stale | (
            named & {"unaliased_exclusion"} if module in closure else set()
        )
        if forbidden:
            offenders[module] = forbidden
    assert not offenders, (
        "a judging function outside the one target-side door:\n"
        + "\n".join(f"  {name}: {sorted(names)}" for name, names in offenders.items())
        + "\nA module in the execd closure may name only `judged_exclusion` / "
        "`refuse_excluded_target`; `unaliased_exclusion` answers a WEAKER question and is "
        "for Home, which holds no workspace."
    )
    assert spelling_callers, "the Home-side door has no callers — the gate is vacuous"
    assert not (spelling_callers & closure)
