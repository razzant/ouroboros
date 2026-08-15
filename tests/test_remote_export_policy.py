"""The Home export-policy authority, source-side application, and D7 disclosure.

The behaviours under test are the ones that were only on paper before:

* the policy is decided on Home, hash-bound into the prepare, and applied
  MECHANICALLY at the source — the excluded bytes are never in any blob;
* Home revalidates the returned manifest against the SAME document and refuses
  loudly on any disagreement;
* a policy exclusion does NOT fail the operation (the donor's 4.0 bug), while an
  IO failure or an unstable tree still does — the tests must be able to tell those
  two apart, because conflating them is what took plan review down;
* the partial fact, the exact count and the disclosed list reach `artifact_bundle`,
  the verification ledger and the CLI.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess

import pytest

from ouroboros.export_policy_contract import build_policy_document, export_policy_hash
from ouroboros.remote_export_policy import (
    ExportPolicy,
    ExportPolicyViolation,
    apply_export_ledger_entry,
    bundle_export_fields,
    channel_for_operation,
    disclosure_summary_line,
    export_disclosure,
    merge_export_disclosures,
    policy_for_operation,
    validate_operation_trace,
    validate_returned_manifest,
    workspace_relative_protected_paths,
)
from ouroboros.workspace_payload_native import collect_declared_outputs
from ouroboros.workspace_query_native import export_workspace_patch
from ouroboros.workspace_snapshot_native import snapshot_workspace

REPO = pathlib.Path(__file__).resolve().parent.parent
SECRET = "SECRET_VALUE_THAT_MUST_NEVER_TRANSFER"


class _Ctx:
    """The minimum a policy build reads: the task contract's resource policy."""

    def __init__(self, protected: list[str]):
        self.task_metadata = {
            "task_contract": {"resource_policy": {"protected_artifacts": [{"paths": protected}]}}
        }


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    for argv in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *argv], cwd=root, check=True)
    (root / "app.py").write_text("print(1)\n", encoding="utf-8")
    (root / ".env").write_text(f"TOKEN={SECRET}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)
    return root


def _head(root: pathlib.Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


# ── Home decides ─────────────────────────────────────────────────────────────


def test_the_policy_reuses_protected_artifacts_and_spells_it_for_the_target():
    ctx = _Ctx(["docs/BIBLE.md", "/srv/project/private/keys.txt", "/elsewhere/other.txt"])
    relatives = workspace_relative_protected_paths(ctx, workspace_root="/srv/project")
    # A relative entry travels as-is; an absolute one only when it is under the
    # remote root; a Home-side absolute path is meaningless on the target and is
    # dropped rather than tail-matched onto some unrelated remote file.
    assert relatives == ("docs/BIBLE.md", "private/keys.txt")


def test_every_native_operation_maps_to_a_declared_channel():
    from ouroboros.workspace_native_contract import MANDATORY_REMOTE_NATIVE_OPERATIONS

    for operation in sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS):
        assert channel_for_operation(operation)


# The declared channels nothing produces for yet. The registry is CLOSED and grows
# with the implementation, so a declared-but-unproduced kind is a promise the code
# has not kept: `channel_profile` would happily hand out a policy for it, and a
# reader of the table cannot tell which half of it is live. Naming the remainder keeps
# the gap a RECORDED fact instead of silence, and the assertions below fail in BOTH
# directions — a new producer must move its channel off this list, and a new unproduced
# channel must be justified by being added to it.
#
# Two channels left this list when their HOME halves landed (D12): `attachment_stage`
# and `subagent_patch`. (A third, `file_bridge`, left the REGISTRY entirely with the
# external-coder bridge — naming it here as a producerless channel implied a live
# declaration that no longer exists.) The two that remain are the wire-internal kinds:
# process output and the externalized envelope are produced by the TRANSPORT, which
# names them itself and has no export door to route through.
_CHANNELS_WITH_NO_PRODUCER = frozenset({
    "operation_envelope",
    "process_stream",
})


def test_the_channel_registry_says_which_half_of_it_is_live():
    from ouroboros.export_policy_contract import EXPORT_CHANNELS
    from ouroboros.remote_export_policy import (
        HOME_CHANNEL_PRODUCERS,
        OPERATION_EXPORT_CHANNEL,
    )

    declared = set(EXPORT_CHANNELS)
    # Two producer tables: the tool dispatch's operation map, and the Home-initiated
    # doors that bind their channel at the call site because they have no tool name.
    produced = set(OPERATION_EXPORT_CHANNEL.values()) | set(HOME_CHANNEL_PRODUCERS)
    assert produced <= declared, sorted(produced - declared)
    assert all(HOME_CHANNEL_PRODUCERS.values()), "every Home producer must name its door"
    assert declared - produced == set(_CHANNELS_WITH_NO_PRODUCER), (
        "the set of declared-but-unproduced export channels changed; a channel that "
        "gained a producer must leave _CHANNELS_WITH_NO_PRODUCER, and a new channel "
        "with no producer must be added to it with a reason: "
        f"{sorted(declared - produced)}"
    )
    # An unproduced channel must still be a well-formed, fail-closed declaration —
    # not a half-entry that would blow up the day something starts using it.
    for channel in sorted(_CHANNELS_WITH_NO_PRODUCER):
        document = build_policy_document(channel=channel)
        assert document["channel"] == channel
        assert export_policy_hash(document)


def test_the_policy_carried_into_prepare_is_hash_bound():
    policy = policy_for_operation(
        _Ctx(["docs/BIBLE.md"]), "snapshot_manifest_and_blob_export", workspace_root="/srv/p"
    )
    payload = policy.arg_payload()
    # Underscore-prefixed, so the target strips it from execution_args: the policy
    # shapes the export, it is not an argument the guards authorize.
    assert set(payload) == {"_export_policy"}
    assert export_policy_hash(payload["_export_policy"]) == policy.policy_hash
    assert "docs/BIBLE.md" in payload["_export_policy"]["protected_paths"]


# ── the source applies, and the bytes never leave ────────────────────────────


def test_a_credential_file_is_excluded_at_the_source_and_the_blobs_prove_it(tmp_path):
    root = _repo(tmp_path)
    document = build_policy_document(channel="workspace_snapshot")
    manifest, blobs = snapshot_workspace(root, policy=document)

    assert manifest["complete"] is False
    assert manifest["integrity_complete"] is True
    assert manifest["policy_scope"] == "policy_filtered"
    assert manifest["excluded_count"] == 1
    assert manifest["policy_exclusions"] == [
        {"path": ".env", "reason": "sensitive_file", "judged": ".env"}
    ]
    assert manifest["policy_hash"] == export_policy_hash(document)
    # The property that matters: not "the manifest omits it" but "the bytes are not
    # here". A blob set that still carried them would only be filtered later, which
    # is already the leak.
    assert all(SECRET.encode() not in payload for payload in blobs.values())
    assert ".env" not in {row["path"] for row in manifest["entries"]}
    assert "app.py" in {row["path"] for row in manifest["entries"]}


def test_a_protected_directory_is_one_disclosed_row_and_is_not_descended_into(tmp_path):
    """The snapshot walk's prune-and-single-disclosure contract, which was comment-only.

    Stated at `workspace_snapshot_native._snapshot_once` — "a whole protected subtree is
    disclosed as ONE exclusion and pruned" — and pinned by nothing: disabling the branch
    (`if False and directory_reason == REASON_PROTECTED_ARTIFACT:`) left the FULL suite
    green while the walk descended into every protected subtree and the owner's omission
    note grew from one inspectable name to one row per contained file, truncated at 20
    with "+N more". No bytes leak either way — protected matching is prefix-based, so
    children are excluded individually — which is exactly why nothing noticed. The
    sibling `excluded_directory` branch IS pinned; every `protected_paths` fixture in
    the suite named FILES, so no protected DIRECTORY had ever been through this walk.
    """

    root = _repo(tmp_path)
    vault = root / "vault"
    (vault / "deep").mkdir(parents=True)
    for index in range(6):
        (vault / f"key{index}.pem").write_text(f"KEY-{index}\n", encoding="utf-8")
        (vault / "deep" / f"key{index}.pem").write_text(f"DEEP-{index}\n", encoding="utf-8")
    document = build_policy_document(
        channel="workspace_snapshot", protected_paths=["vault"]
    )
    manifest, blobs = snapshot_workspace(root, policy=document)

    protected = [
        row for row in manifest["policy_exclusions"] if row["reason"] == "protected_artifact"
    ]
    assert protected == [{"path": "vault", "reason": "protected_artifact", "judged": "vault"}]
    assert not [row for row in manifest["entries"] if row["path"].startswith("vault/")]
    assert all(b"KEY-" not in payload for payload in blobs.values())


def test_the_plan_review_patch_channel_keeps_working_with_a_dotenv_present(tmp_path):
    """The donor 4.0 bug: one `.env` refused the whole export.

    Plan review of a remote project, the claude_code_edit bridge and the file
    bridge all ride this channel, so a refusal here took all three down. D7 says
    the export proceeds and discloses.
    """

    root = _repo(tmp_path)
    (root / "app.py").write_text("print(2)\n", encoding="utf-8")
    (root / ".env").write_text(f"TOKEN={SECRET}_rotated\n", encoding="utf-8")
    (root / "notes.txt").write_text("new untracked file\n", encoding="utf-8")

    result = export_workspace_patch(
        root,
        {"expected_head": _head(root)},
        policy=build_policy_document(channel="workspace_patch"),
    )
    export = result.envelope.trace["patch_export"]

    assert export["status"] == "ready_with_changes"
    assert export["complete"] is False
    assert export["integrity_complete"] is True
    assert export["policy_scope"] == "policy_filtered"
    assert export["excluded"] == [
        {"path": ".env", "reason": "sensitive_file", "judged": ".env"}
    ]
    assert export["excluded_count"] == 1
    patches = list(result.blobs.values())
    assert patches, "the export must still produce a patch"
    assert all(SECRET.encode() not in payload for payload in patches)
    assert any(b"app.py" in payload for payload in patches)
    assert any(b"notes.txt" in payload for payload in patches)


def test_a_declared_output_loses_the_member_not_the_artifact(tmp_path):
    root = (tmp_path / "ws").resolve()
    (root / "site" / "secrets").mkdir(parents=True)
    (root / "site" / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
    (root / "site" / "secrets" / "token.txt").write_text(SECRET, encoding="utf-8")

    blobs, artifacts, notes, failed, excluded, exported = collect_declared_outputs(
        root, {"cwd": str(root), "outputs": ["site"]}, {}
    )

    assert failed is False, "a policy exclusion must not fail the declared output"
    assert [row["member_path"] for row in artifacts] == ["index.html"]
    assert excluded == [
        {
            "path": "site/secrets/token.txt",
            "reason": "sensitive_component",
            "judged": "site/secrets/token.txt",
        }
    ]
    # Excluded on the SOURCE, ahead of read_bytes — not filtered after transfer.
    assert all(SECRET.encode() not in payload for payload in blobs.values())
    assert any("excluded from output by export policy" in note for note in notes)
    # The other half of the same disclosure, and the half Home's leak check reads: the
    # channel declares what it SHIPPED, so `validate_returned_manifest` re-evaluates the
    # policy over a non-empty list instead of passing on arithmetic alone.
    assert exported == ["site/index.html"]


def test_the_query_channels_no_longer_return_the_bytes_the_snapshot_excludes(tmp_path):
    """The leak this change closes, stated as the bytes that used to come back.

    `search_code` returns matched LINE CONTENT and `vcs_diff` returns diff hunks.
    Before the one policy both filtered only the explicitly protected paths — and
    nothing on Home ever populated those — so a `.env` on the remote host had its
    contents returned to Home while the snapshot channel was excluding that very
    same file two doors over.
    """

    from ouroboros.workspace_query_native import (
        execute_git_workspace_operation,
        execute_workspace_query_operation,
    )

    root = _repo(tmp_path)
    (root / ".env").write_text(f"TOKEN={SECRET}_rotated\n", encoding="utf-8")
    facts = {"export_policy": build_policy_document(channel="workspace_query")}

    search = execute_workspace_query_operation(root, "search_code", {"query": SECRET}, facts)
    # The caller's own query string is echoed back; what must not come back is the
    # matched LINE, and the file must not have been opened at all.
    assert f"{SECRET}_rotated" not in search.text
    assert "1 files searched" in search.text

    for operation in ("vcs_diff", "vcs_status"):
        envelope = execute_git_workspace_operation(root, operation, {}, facts).envelope
        body, _, note = envelope.text.partition("VCS_POLICY_FILTERED")
        assert SECRET not in envelope.text, operation
        # The name may appear in the DISCLOSURE and must not appear in the projection —
        # the same split `search_code` above is asserted on, for the same reason. These two
        # operations used to emit no disclosure at all, so a filtered diff read as an
        # authoritative complete one and Home's own check ran over an empty field set.
        assert ".env" not in body, operation
        assert ".env" in note, f"{operation} filtered .env without disclosing it"
        block = envelope.trace["export_policy"]
        assert block["policy_scope"] == "policy_filtered"
        assert block["excluded"] == [
            {"path": ".env", "reason": "sensitive_file", "judged": ".env"}
        ]
        assert block["exported"] and ".env" not in block["exported"]
        assert envelope.trace["completion"] == "partial"


def test_a_filtered_query_says_so_instead_of_reporting_no_matches(tmp_path):
    """Filtering without disclosure moved the leak from bytes to CONCLUSIONS.

    Reproduced before this landed: with a `.env` holding `SECRET_TOKEN`,
    `search_code("SECRET_TOKEN")` on a policy-bound (remote) workspace returned
    "No matches found for literal `SECRET_TOKEN` … (2 files searched)." with
    `completion=complete`, while the LOCAL `search_code` over the identical tree
    returned the matching line — local `_code_search` has no export policy and
    `.env` is not in its skip globs. The model reasons over the text, so a silent
    skip made it conclude "the key is not in this workspace" from a false premise.
    That is the D7 partial-disclosure rule and the §9 cross-placement parity rule
    failing together, and no byte had to move for the harm to land.
    """

    from ouroboros.workspace_query_native import execute_workspace_query_operation

    root = (tmp_path / "ws").resolve()
    root.mkdir()
    (root / ".env").write_text("SECRET_TOKEN=abc123\n", encoding="utf-8")
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    facts = {"export_policy": build_policy_document(channel="workspace_query")}

    env = execute_workspace_query_operation(root, "search_code", {"query": "SECRET_TOKEN"}, facts)

    # The bytes still do not come back — that part must not regress.
    assert "abc123" not in env.text
    # But the omission is now IN the answer the model reads, by count and by path.
    assert "1 path excluded by the export policy" in env.text
    assert ".env: sensitive_file" in env.text
    assert "not authoritative" in env.text.casefold()
    # …and on the wire, in the same block the declared-output channel emits.
    assert env.trace["completion"] == "partial"
    block = env.trace["export_policy"]
    assert block["complete"] is False
    assert block["policy_scope"] == "policy_filtered"
    assert block["excluded_count"] == 1
    # `judged` names the spelling the policy excluded — the same as `path` here, and the
    # alias's real target when an alias is the finding. It is what makes Home able to
    # re-derive the claim instead of trusting it.
    assert block["excluded"] == [
        {"path": ".env", "reason": "sensitive_file", "judged": ".env"}
    ]
    # The exclusion is a DECISION, not a breakage: integrity is still whole.
    assert block["integrity_complete"] is True


def test_an_unfiltered_query_stays_complete_and_quiet(tmp_path):
    """The disclosure keys are always present; only their VALUES encode filtering.

    Infrastructure directories (`.git`, `__pycache__`) are excluded by both
    placements — local `search_code` prunes them via `SKIP_DIRS` with no
    disclosure — so `excluded_directory` must NOT be reported. Otherwise every
    ordinary query in a git repo would read as partial and the signal that matters
    would be buried.
    """

    from ouroboros.workspace_query_native import execute_workspace_query_operation

    root = (tmp_path / "ws").resolve()
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00binary")
    facts = {"export_policy": build_policy_document(channel="workspace_query")}

    env = execute_workspace_query_operation(root, "search_code", {"query": "hello"}, facts)

    assert "Found 1 match" in env.text
    assert "POLICY_FILTERED" not in env.text
    assert env.trace["completion"] == "complete"
    block = env.trace["export_policy"]
    assert block["complete"] is True
    assert block["policy_scope"] == "full"
    assert block["excluded_count"] == 0


def test_query_code_discloses_what_it_removed_from_the_inventory(tmp_path):
    """`query_code` filtered its inventory silently too — same class, same fix."""

    from ouroboros.workspace_query_native import execute_workspace_query_operation

    root = (tmp_path / "ws").resolve()
    (root / "secrets").mkdir(parents=True)
    (root / "app.py").write_text("def handler():\n    return 1\n", encoding="utf-8")
    (root / "secrets" / "creds.py").write_text(
        "def handler():\n    return 'api_key'\n", encoding="utf-8"
    )
    facts = {
        "export_policy": build_policy_document(
            channel="workspace_query", protected_paths=["secrets"]
        )
    }

    env = execute_workspace_query_operation(
        root, "query_code", {"op": "definition", "query": "handler"}, facts
    )

    assert "secrets/creds.py" not in env.text
    assert "app.py" in env.text
    assert "QUERY_POLICY_FILTERED" in env.text
    assert env.trace["completion"] == "partial"
    block = env.trace["export_policy"]
    assert block["policy_scope"] == "policy_filtered"
    assert block["excluded_count"] >= 1
    assert any(row["path"] == "secrets" for row in block["excluded"])


def test_an_excluded_search_scope_is_not_an_empty_workspace(tmp_path):
    """Pointing search_code AT an excluded path returned a confident empty result."""

    from ouroboros.workspace_query_native import execute_workspace_query_operation

    root = (tmp_path / "ws").resolve()
    root.mkdir()
    (root / ".env").write_text("SECRET_TOKEN=abc123\n", encoding="utf-8")
    facts = {"export_policy": build_policy_document(channel="workspace_query")}

    env = execute_workspace_query_operation(
        root, "search_code", {"query": "SECRET_TOKEN", "path": ".env"}, facts
    )

    assert "abc123" not in env.text
    assert "SEARCH_POLICY_FILTERED" in env.text
    assert env.trace["export_policy"]["excluded_count"] == 1
    assert env.trace["completion"] == "partial"


def test_the_spool_log_and_media_channels_go_through_the_same_policy(tmp_path):
    """A path-free channel is still DECLARED; a media source is still judged."""

    from ouroboros.export_policy_contract import (
        EXPORT_CHANNELS,
        QUESTION_EXPORT,
        ExportPolicyExcludedError,
        refuse_excluded_target,
    )

    assert EXPORT_CHANNELS["process_spool_log"]["path_bearing"] is False
    assert EXPORT_CHANNELS["media_frames"]["path_bearing"] is True
    facts = {"export_policy": build_policy_document(channel="media_frames")}
    root = (tmp_path / "ws").resolve()
    root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ExportPolicyExcludedError):
        refuse_excluded_target(
            root, None, "clips/.env", facts,
            question=QUESTION_EXPORT, channel="media_frames",
        )
    refuse_excluded_target(
        root, None, "clips/demo.mp4", facts,
        question=QUESTION_EXPORT, channel="media_frames",
    )


# ── policy exclusion vs. breakage: DIFFERENT things ──────────────────────────


def test_an_unreadable_entry_is_fail_closed_and_names_itself_as_such(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    (root / "app.py").write_text("print(2)\n", encoding="utf-8")
    real_read = pathlib.Path.read_bytes

    def exploding_read(self, *args, **kwargs):
        if self.name == "app.py":
            raise OSError("simulated IO failure")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_bytes", exploding_read)
    manifest, _blobs = snapshot_workspace(
        root, policy=build_policy_document(channel="workspace_snapshot")
    )
    # A broken read is NOT a policy exclusion: integrity fails, and the patch
    # channel refuses on it rather than exporting a partial nobody can bound.
    assert manifest["integrity_complete"] is False
    assert manifest["failure_count"] >= 1
    assert {row["reason"] for row in manifest["failures"]} <= {
        "entry_read_error", "changed_during_read", "walk_error", "unstable_observation",
    }
    with pytest.raises(RuntimeError, match="fail-closed|could not be observed"):
        export_workspace_patch(
            root,
            {"expected_head": _head(root)},
            policy=build_policy_document(channel="workspace_patch"),
        )


def test_the_two_conditions_are_distinguishable_from_the_manifest_alone(tmp_path):
    """Nothing downstream should have to guess which kind of partial it has."""

    root = _repo(tmp_path)
    filtered, _ = snapshot_workspace(
        root, policy=build_policy_document(channel="workspace_snapshot")
    )
    assert (filtered["complete"], filtered["integrity_complete"]) == (False, True)
    assert filtered["policy_scope"] == "policy_filtered"
    assert filtered["failures"] == []


# ── Home revalidates ─────────────────────────────────────────────────────────


def _policy() -> ExportPolicy:
    document = build_policy_document(channel="workspace_snapshot")
    return ExportPolicy(
        channel="workspace_snapshot", document=document, policy_hash=export_policy_hash(document)
    )


def test_a_manifest_with_no_policy_hash_cannot_be_vouched_for():
    with pytest.raises(ExportPolicyViolation, match="declares no policy hash"):
        validate_returned_manifest(_policy(), {"entries": [], "excluded_count": 0})


def test_a_manifest_from_a_different_policy_is_refused():
    other = build_policy_document(channel="workspace_snapshot", protected_paths=["docs"])
    with pytest.raises(ExportPolicyViolation, match="Home authorized"):
        validate_returned_manifest(
            _policy(), {"policy_hash": export_policy_hash(other), "entries": []}
        )


def test_a_returned_entry_the_policy_excludes_proves_the_filter_never_ran():
    policy = _policy()
    with pytest.raises(ExportPolicyViolation, match="source-side filtering did not run"):
        validate_returned_manifest(
            policy,
            {
                "policy_hash": policy.policy_hash,
                "entries": [{"path": "app.py"}, {"path": ".env"}],
                "excluded_count": 0,
                "policy_scope": "full",
            },
        )


def test_a_disclosure_that_does_not_describe_its_own_omission_is_refused():
    policy = _policy()
    with pytest.raises(ExportPolicyViolation, match="describe the same omission"):
        validate_returned_manifest(
            policy,
            {
                "policy_hash": policy.policy_hash,
                "entries": [],
                "excluded": [{"path": ".env", "reason": "sensitive_file"}],
                "excluded_count": 7,
            },
        )
    with pytest.raises(ExportPolicyViolation, match="unknown exclusion reasons"):
        validate_returned_manifest(
            policy,
            {
                "policy_hash": policy.policy_hash,
                "entries": [],
                "excluded": [{"path": ".env", "reason": "because_i_said_so"}],
            },
        )
    with pytest.raises(ExportPolicyViolation, match="policy_scope"):
        validate_returned_manifest(
            policy,
            {
                "policy_hash": policy.policy_hash,
                "entries": [],
                "excluded": [{"path": ".env", "reason": "sensitive_file"}],
                "excluded_count": 1,
                "policy_scope": "full",
            },
        )


def test_home_refuses_a_manifest_whose_exported_evidence_was_truncated():
    """A SAMPLE of the exported set cannot prove the filter ran.

    `exported[]` shared the exclusion bound of 200, and `exported_disclosure_truncated`
    was computed and read by nobody. A paid reviewer put a policy-excluded path at index
    200 of a 201-path export and this check PASSED — the backstop for the source-side
    filter was silently a spot check. The exported list is bounded at
    `MAX_EXPORTED_PATHS` now, above every per-channel result cap, and reaching it is a
    refusal rather than a shrug.
    """

    from ouroboros.export_policy_contract import (
        MAX_EXPORTED_PATHS,
        export_disclosure_block,
    )

    policy = _policy()
    facts = {"export_policy": policy.document}
    # The reviewer's exact case: the 201st path is the excluded one. It must be SEEN.
    just_over_the_old_bound = [f"ok/f{index}.txt" for index in range(200)] + [".env"]
    block = export_disclosure_block(facts, [], just_over_the_old_bound)["export_policy"]
    block["policy_hash"] = policy.policy_hash
    assert ".env" in block["exported"], "the evidence list dropped the offending path"
    with pytest.raises(ExportPolicyViolation, match="source-side filtering did not run"):
        validate_returned_manifest(policy, block)
    # And above the real bound, the truncation itself is the refusal.
    huge = [f"ok/f{index}.txt" for index in range(MAX_EXPORTED_PATHS + 1)]
    block = export_disclosure_block(facts, [], huge)["export_policy"]
    block["policy_hash"] = policy.policy_hash
    assert block["exported_disclosure_truncated"] is True
    with pytest.raises(ExportPolicyViolation, match="truncated its exported-path evidence"):
        validate_returned_manifest(policy, block)


def test_home_refuses_an_exclusion_the_policy_does_not_produce():
    """A FALSE omission is a lie the owner cannot detect, because nothing is missing.

    Home checked that a disclosed reason code was in the closed set and never that the
    policy actually gives that reason for that path, so a target could claim `src/app.py`
    had been excluded as `sensitive_file` and the owner would read a filtering that never
    happened. Re-derived now — and an honest ALIAS row still passes, which is why the row
    carries `judged`: `notes.txt` is innocent by construction and only the spelling it IS
    can be re-evaluated.
    """

    from ouroboros.export_policy_contract import export_disclosure_block

    policy = _policy()
    facts = {"export_policy": policy.document}
    invented = export_disclosure_block(
        facts, [{"path": "src/app.py", "reason": "sensitive_file"}], ["ok.txt"]
    )["export_policy"]
    invented["policy_hash"] = policy.policy_hash
    with pytest.raises(ExportPolicyViolation, match="this policy does not produce"):
        validate_returned_manifest(policy, invented)
    honest_alias = export_disclosure_block(
        facts,
        [{"path": "notes.txt", "reason": "sensitive_file", "judged": ".env"}],
        ["ok.txt"],
    )["export_policy"]
    honest_alias["policy_hash"] = policy.policy_hash
    assert validate_returned_manifest(policy, honest_alias)["excluded_count"] == 1


def test_source_and_home_apply_the_SAME_projection_of_one_document():
    """`read_file('.git/config')` is legal at the source and used to fail at Home.

    `QUESTION_NAMED_SOURCE` drops the bulk-only excluded-dirs rule on purpose — that rule
    exists so a TREE export does not ship `.git`, not to forbid a path the model named by
    hand, and the local route allows the read. Home re-evaluated every returned path under
    the DEFAULT question, so the legitimate read raised `ExportPolicyViolation` after its
    bytes had already crossed the boundary. Two projections of one policy is the same class
    as two mechanics for one question, one layer out.
    """

    from ouroboros.export_policy_contract import (
        QUESTION_NAMED_SOURCE,
        export_disclosure_block,
    )

    policy = _policy()
    facts = {"export_policy": policy.document}
    named = export_disclosure_block(
        facts, [], [".git/config"], question=QUESTION_NAMED_SOURCE
    )["export_policy"]
    named["policy_hash"] = policy.policy_hash
    assert named["policy_question"] == QUESTION_NAMED_SOURCE
    validate_returned_manifest(policy, named)
    # The SAME path claimed under the tree question is still a violation: the projection
    # travels, it does not become a way to opt out of a rule.
    tree = export_disclosure_block(facts, [], [".git/config"])["export_policy"]
    tree["policy_hash"] = policy.policy_hash
    with pytest.raises(ExportPolicyViolation, match="source-side filtering did not run"):
        validate_returned_manifest(policy, tree)
    # …and a question this build does not understand fails closed rather than defaulting.
    forged = dict(tree, policy_question="whatever_i_like")
    with pytest.raises(ExportPolicyViolation, match="does not understand"):
        validate_returned_manifest(policy, forged)


def test_the_listing_and_the_query_walk_declare_what_they_exported(tmp_path):
    """The two disclosure sites that passed no `exported` list at all.

    `_list_files` and `_PolicyExclusions.disclosure` were the last two vacuous cells in
    Home's backstop: it derives its field list from `MANIFEST_EXPORTED_PATH_FIELDS`, found
    `exported` present and empty, and re-evaluated the policy over nothing — passing on
    hash and arithmetic alone for every listing and every search, which is exactly the
    hole that had just been closed for reads and declared outputs.
    """

    from ouroboros.export_policy_contract import build_policy_document
    from ouroboros.workspace_native import execute_native_operation

    root = _repo(tmp_path)
    facts = {"export_policy": build_policy_document(channel="workspace_query")}
    listing = execute_native_operation(
        root, "list_files", {"path": "."}, native_facts=facts
    ).envelope
    exported = listing.trace["export_policy"]["exported"]
    assert exported, "the listing declared nothing it had handed over"
    assert ".env" not in exported
    assert "app.py" in exported or "src" in exported, exported
    search = execute_native_operation(
        root, "search_code", {"query": "SECRET_TOKEN"}, native_facts=facts
    ).envelope
    searched = search.trace["export_policy"]["exported"]
    assert ".env" not in searched
    # Every declared path must survive Home's own re-evaluation, which is the point.
    policy = ExportPolicy(
        channel="workspace_query",
        document=facts["export_policy"],
        policy_hash=export_policy_hash(facts["export_policy"]),
    )
    for envelope in (listing, search):
        block = dict(envelope.trace["export_policy"])
        validate_returned_manifest(policy, block)


def test_a_real_source_manifest_validates_against_the_policy_that_produced_it(tmp_path):
    root = _repo(tmp_path)
    document = build_policy_document(channel="workspace_snapshot")
    policy = ExportPolicy(
        channel="workspace_snapshot", document=document, policy_hash=export_policy_hash(document)
    )
    manifest, _blobs = snapshot_workspace(root, policy=document)
    disclosure = validate_returned_manifest(policy, manifest)
    assert disclosure["partial"] is True
    assert disclosure["excluded_count"] == 1
    assert disclosure["excluded"][0]["path"] == ".env"
    assert disclosure["excluded"][0]["disclosure"]
    assert disclosure["full_manifest_sha256"] == manifest["fingerprint"]


def test_a_trace_with_no_manifest_yields_the_empty_disclosure_not_an_assumed_clean_one():
    disclosure = validate_operation_trace(_policy(), {"completion": "complete"})
    assert disclosure["excluded_count"] == 0
    assert disclosure["partial"] is False
    assert disclosure["channels"] == []


def test_a_manifest_that_omits_its_hash_is_refused_rather_than_skipped():
    """The sibling case the test above states in its NAME and never exercised.

    Selection used to read `isinstance(manifest, Mapping) and manifest.get("policy_hash")`,
    which handed the TARGET the choice of whether to be checked: a manifest that simply
    left the field out was skipped WHOLE, so it reached neither the refusal written for
    exactly this case nor the exported-path leak re-check behind it. The less attested
    manifest was treated more leniently than a wrong one, and it merged to
    `policy_scope: "full"` — a positive claim that nothing was withheld. "Exported
    nothing" and "exported without a policy" must not read the same downstream.
    """

    leaking = {"entries": [{"path": ".env"}], "exported": [".env"], "excluded_count": 0}
    with pytest.raises(ExportPolicyViolation, match="declares no policy hash"):
        validate_operation_trace(_policy(), {"snapshot": dict(leaking)})
    # The same manifest one field richer: now the LEAK check speaks, which is the check
    # the omission was skipping past.
    with pytest.raises(ExportPolicyViolation, match="the policy excludes"):
        validate_operation_trace(
            _policy(), {"snapshot": {**leaking, "policy_hash": _policy().policy_hash}}
        )


def test_a_manifest_under_an_undeclared_trace_key_is_refused():
    """`guarded_patch_apply` shipped two manifests under keys nobody had declared.

    Its FAILURE arms answer under `snapshot` and were validated; its success arm — the
    only arm that MUTATES the target — invented `before`/`after`, and a judge reading
    only the declared list cannot see an undeclared key BY CONSTRUCTION, so it passed in
    silence while Home recorded a clean export. Both keys are declared now; reading the
    trace itself is what makes the NEXT invented one loud instead of invisible.
    """

    with pytest.raises(ExportPolicyViolation, match="undeclared key 'mirror'"):
        validate_operation_trace(
            _policy(), {"mirror": {"policy_hash": _policy().policy_hash, "entries": []}}
        )


def test_one_operation_crossing_several_channels_gives_the_owner_one_number():
    merged = merge_export_disclosures([
        export_disclosure(
            {
                "policy_hash": "a" * 64,
                "excluded": [{"path": ".env", "reason": "sensitive_file"}],
                "excluded_count": 1,
            },
            channel="workspace_snapshot",
        ),
        export_disclosure(
            {
                "policy_hash": "a" * 64,
                "excluded": [{"path": "site/secrets/t", "reason": "sensitive_component"}],
                "excluded_count": 1,
            },
            channel="declared_output",
        ),
    ])
    assert merged["excluded_count"] == 2
    assert merged["channels"] == ["declared_output", "workspace_snapshot"]
    assert merged["partial"] is True
    assert {row["channel"] for row in merged["excluded"]} == {
        "workspace_snapshot", "declared_output"
    }


# ── the disclosure reaches every surface D7 names ────────────────────────────


def _filtered_result() -> dict:
    return {
        "status": "completed",
        "artifacts": [],
        "remote_export": export_disclosure(
            {
                "policy_hash": "b" * 64,
                "excluded": [{"path": ".env", "reason": "sensitive_file"}],
                "excluded_count": 1,
                "fingerprint": "c" * 64,
            },
            channel="workspace_snapshot",
        ),
    }


def test_the_artifact_bundle_discloses_the_omission_without_touching_the_status():
    from ouroboros.outcomes import artifact_bundle_from_result

    clean = artifact_bundle_from_result({"status": "completed", "artifacts": []})
    filtered = artifact_bundle_from_result(_filtered_result())
    # THE additive property: same terminal status, extra disclosure.
    assert filtered["status"] == clean["status"]
    assert filtered["partial"] is True
    assert filtered["excluded_count"] == 1
    assert filtered["excluded"][0]["path"] == ".env"
    assert filtered["export_policy_hash"] == "b" * 64
    assert "partial" not in clean and "excluded_count" not in clean


def test_the_verification_ledger_names_the_omission_and_does_not_call_it_a_failure():
    from ouroboros.outcomes import (
        artifact_bundle_from_result,
        refresh_verification_ledger_artifacts,
    )

    bundle = artifact_bundle_from_result(_filtered_result())
    ledger = refresh_verification_ledger_artifacts({"entries": []}, bundle)
    rows = [row for row in ledger["entries"] if row.get("kind") == "remote_export"]
    assert len(rows) == 1
    assert rows[0]["status"] == "policy_filtered"
    assert rows[0]["excluded_count"] == 1
    assert ledger["summary"]["has_failures"] is False
    # Refreshing twice must not accumulate a second row.
    again = refresh_verification_ledger_artifacts(ledger, bundle)
    assert len([row for row in again["entries"] if row.get("kind") == "remote_export"]) == 1


def test_a_clean_export_adds_no_ledger_row():
    assert apply_export_ledger_entry([], {"status": "ready"}) == []
    assert bundle_export_fields({"status": "completed"}) == {}


def test_the_cli_and_the_model_both_get_a_sentence_not_a_silent_absence():
    from ouroboros.cli import export_policy_note

    bundle = {"excluded_count": 2, "excluded": [{"path": ".env"}, {"path": "keys/id_rsa"}]}
    note = export_policy_note({"artifact_bundle": bundle})
    assert "export policy excluded 2 path(s)" in note
    assert ".env" in note
    assert export_policy_note({"artifact_bundle": {}}) == ""
    line = disclosure_summary_line(_filtered_result()["remote_export"])
    assert "REMOTE_EXPORT_POLICY_FILTERED" in line and ".env" in line
    assert disclosure_summary_line({"excluded_count": 0}) == ""


# ── the closed registry, proved by grep ──────────────────────────────────────


# Every site in the scanned tree that names the transport's blob primitive or hands a
# blob mapping to a call, keyed by (file, enclosing function) with the reason it is
# there. WHY per-symbol and not per-file: the previous shape excluded eleven whole
# FILES, including the three most-edited ones (`workspace_executor.py`,
# `remote_workspace.py`, `execd.py`) — so a new bypass added anywhere in them was
# pre-approved, and the gate's own claim ("every crossing routes through the
# service") was untrue of exactly the code most likely to break it. Five of those
# eleven entries had also gone STALE and exempted files with no site left at all,
# which the staleness test below now forbids.
_BLOB_CHANNEL_ALLOWLIST: dict[tuple[str, str], str] = {
    # The SOURCE side of the wire. execd runs on the target, where the transfer
    # service does not exist and cannot: `remote_transfer` is on execd's
    # forbidden-import list (asserted by the import-closure gate), so these sites
    # are structurally incapable of routing through it — they hand bytes to the
    # envelope and import nothing.
    ("ouroboros/execd.py", "__init__"): "execd wires its own state stores",
    ("ouroboros/execd.py", "prepare"): "target-side prepare receives staged blobs",
    ("ouroboros/execd.py", "continue_prepared"): "target-side continuation of a staged prepare",
    ("ouroboros/execd.py", "_execute_task_file_operation"): "target-side task-file kernel call",
    ("ouroboros/execd.py", "_prepare_operation"): "target-side prepare kernel call",
    ("ouroboros/execd.py", "_revalidate_prepared_target"): "target-side re-prepare for integrity",
    ("ouroboros/execd.py", "_receive_control"): "the wire frame handler itself",
    # The native kernel BUILDS blobs; it never imports one.
    ("ouroboros/workspace_native.py", "execute_native_operation"): "kernel dispatch passes staged blobs",
    # The transport primitive and the broker/facade that forward it.
    ("ouroboros/remote_ssh.py", "execute_prepared"): "the transport primitive's one call site",
    ("ouroboros/remote_workspace.py", "<module>"): "broker protocol declarations of the primitive",
    ("ouroboros/remote_workspace.py", "_fetch_blob_on_broker"): "the broker's forwarding shim",
    # The broker RPC is name-addressed, so the primitive's name appears as a STRING on
    # both ends of it. These three are the whole of that: the two client stubs that name
    # the method they are calling, and the one handler table it is looked up in. A bare
    # string is a finding by default precisely because dynamic dispatch is how a bypass
    # hides a rename — so each of these three is a decision, not a category.
    ("ouroboros/remote_workspace.py", "fetch_blob"): "broker client stub names its own RPC method",
    ("ouroboros/remote_workspace.py", "_dispatch"): "the broker's one RPC handler table",
    ("ouroboros/remote_worker_proxy.py", "fetch_blob"): "worker-proxy stub names its own RPC method",
    ("ouroboros/workspace_executor.py", "prepare_native_operation"): "facade forwards staged blobs to prepare",
    ("ouroboros/workspace_executor.py", "fetch_native_blob"): "facade's named forward of the primitive",
    # Verify-then-hand-to-Home: fetches, proves size and hash, and passes the
    # result to the injected HomeImporter — i.e. to the service.
    ("ouroboros/remote_reconciliation.py", "prefetch_remote_result_import"): "verified prefetch into the importer",
    ("ouroboros/remote_reconciliation.py", "_import_completed_result"): "verified import of a completed result",
    # The Home import/export executor itself: this IS the service.
    ("ouroboros/remote_transfer.py", "export_operation"): "the service's own export door",
}

# Calling the service's OWN export door is the routed case, not a bypass: the method
# is defined by `RemoteTransferService` and by nothing else, so naming it is exactly
# the evidence this gate looks for. Without the exemption an OUTGOING channel
# (attachment staging carries Home bytes to the target) could satisfy the gate only
# by hiding the keyword, which would make it select for evasion rather than routing.
_SERVICE_DOORS = frozenset({"export_operation"})
_BLOB_KEYWORDS = frozenset({"blobs", "input_blobs", "process_blobs"})
_BLOB_PRIMITIVE = "fetch_blob"


def _call_target_name(node: ast.Call) -> str:
    """The attribute/name being called, THROUGH the indirect forms.

    `getattr(svc, "fetch_blob")(…)` and `svc.__dict__["fetch_blob"](…)` are the same
    call as `svc.fetch_blob(…)`; a gate that only reads `ast.Attribute` says they are
    not. The platform gate already closed these two forms for `os.*`, so the shapes
    are known — only literals are judged, because a computed name is outside a static
    scan's reach and pretending otherwise would be worse than saying so.
    """

    func = node.func
    if isinstance(func, ast.Attribute):
        return str(func.attr)
    if isinstance(func, ast.Name):
        return str(func.id)
    if isinstance(func, ast.Call):
        return _indirect_attribute_name(func)
    if isinstance(func, ast.Subscript):
        return _subscript_attribute_name(func)
    return ""


def _indirect_attribute_name(node: ast.Call) -> str:
    """`getattr(obj, "name")` → `"name"`, for a LITERAL name only.

    The three-argument form `getattr(obj, "name", None)` reads the same, which is why
    the arity is a lower bound rather than an equality.
    """

    if (
        isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return str(node.args[1].value)
    return ""


def _subscript_attribute_name(node: ast.Subscript) -> str:
    """`obj.__dict__["name"]` / `vars(obj)["name"]` → `"name"`, literals only."""

    if not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
        return ""
    value = node.value
    if isinstance(value, ast.Attribute) and value.attr == "__dict__":
        return str(node.slice.value)
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "vars":
        return str(node.slice.value)
    return ""


def _blob_dict_names(tree: ast.AST) -> frozenset[str]:
    """Names bound to a dict LITERAL that carries a blob key, so `**payload` is visible.

    `payload = {"blobs": rows}` followed by `send(**payload)` is the keyword form with
    one extra hop; without this the splat carried the mapping past a keyword-only gate.
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        keys = {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if keys & _BLOB_KEYWORDS:
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return frozenset(names)


def _splatted_blob_keys(node: ast.Call, blob_dict_names: frozenset[str]) -> set[str]:
    """The blob keys arriving by `**` splat — the keyword form spelled as a dict.

    `svc.ship(**{"blobs": payload})` and `svc.ship(**already_a_blob_dict)` both pass a
    blob mapping and carry no `keyword.arg`, so the keyword-only gate never saw either.
    """

    keys: set[str] = set()
    for keyword in node.keywords:
        if keyword.arg is not None:
            continue
        value = keyword.value
        if isinstance(value, ast.Dict):
            keys.update(
                str(key.value)
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
        elif isinstance(value, ast.Name) and value.id in blob_dict_names:
            keys.update(_BLOB_KEYWORDS)
    return keys


def blob_channel_sites(relative: str, source: str) -> list[tuple[str, str, int, str]]:
    """Every blob-channel site in one module as `(file, function, line, why)`.

    Forms this reads — each one beyond the literal `x.fetch_blob(...)` the first
    version saw, and each verified undetected before the two independent hardenings
    this function merges:

    * ``svc.fetch_blob`` and ``ouroboros.remote_ssh.fetch_blob`` — the direct spelling.
    * ``getattr(t, "fetch_blob")`` and ``getattr(t, "fetch_blob", None)``.
    * ``svc.__dict__["fetch_blob"]`` and ``vars(svc)["fetch_blob"]``.
    * ``from ouroboros.remote_ssh import fetch_blob as _fb`` — the ALIAS then makes the
      call site spell `_fb(...)`, so the import itself is the finding.
    * any string literal naming the primitive, because dynamic dispatch through
      ``sys.modules`` / a handler table is how the name survives a rename.
    * a blob keyword arriving by SPLAT: ``f(**{"blobs": rows})`` and ``f(**payload)``
      where a dict literal in the same module bound `payload` to a blob key.
    * the plain keyword forms ``blobs=`` / ``input_blobs=`` / ``process_blobs=``.

    BOUNDARY — deliberately not caught, so the claim stays honest:
    * a keyword name assembled at runtime (``**{key: rows}`` with `key` a variable) and
      a dict built key-by-key: there is no literal to read;
    * a blob that travels as an opaque field of a larger object several frames away —
      following that needs real dataflow, and a gate that reports maybes gets muted;
    * modules outside the scanned roots. ``ouroboros/``, ``supervisor/`` and
      ``server.py`` are all in scope (the latter two were not before), but ``scripts/``
      is not: it is build tooling that never holds a transport.

    Sites are keyed by the enclosing FUNCTION, not by line: line numbers drift with
    every edit above them, so an allowlist keyed by line is one that silently slides
    onto a different call. The function name is the unit a reviewer can judge.
    """

    tree = ast.parse(source)
    blob_dicts = _blob_dict_names(tree)
    sites: list[tuple[str, str, int, str]] = []

    def visit(node: ast.AST, function: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        why = ""
        if isinstance(node, (ast.Name, ast.Attribute)) and (
            getattr(node, "id", "") == _BLOB_PRIMITIVE
            or getattr(node, "attr", "") == _BLOB_PRIMITIVE
        ):
            why = "names fetch_blob"
        elif isinstance(node, ast.Subscript) and _subscript_attribute_name(node) == _BLOB_PRIMITIVE:
            why = "reaches fetch_blob through __dict__/vars"
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            alias.name == _BLOB_PRIMITIVE or alias.asname == _BLOB_PRIMITIVE
            for alias in node.names
        ):
            why = "imports fetch_blob (possibly under an alias)"
        elif isinstance(node, ast.Constant) and node.value == _BLOB_PRIMITIVE:
            why = "names fetch_blob as a string (dynamic dispatch)"
        elif isinstance(node, ast.Call):
            if _indirect_attribute_name(node) == _BLOB_PRIMITIVE:
                why = "reaches fetch_blob through getattr"
            else:
                keys = {keyword.arg for keyword in node.keywords if keyword.arg}
                keys |= _splatted_blob_keys(node, blob_dicts)
                door = _call_target_name(node)
                if keys & _BLOB_KEYWORDS and door not in _SERVICE_DOORS:
                    why = f"passes a blob channel to {door or '<expr>'}()"
        if why:
            sites.append((relative, function, int(getattr(node, "lineno", 0)), why))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, "<module>")
    return sites


def blob_channel_offenders(relative: str, source: str) -> list[str]:
    """The sites of `blob_channel_sites` that no allowlist row accounts for."""

    return [
        f"{relative}:{function}:{line}: {why}"
        for relative, function, line, why in blob_channel_sites(relative, source)
        if (relative, function) not in _BLOB_CHANNEL_ALLOWLIST
    ]


def _blob_scanned_sources() -> list[tuple[str, str]]:
    """The roots this gate reads, as `(relative, source)`.

    `supervisor/` and `server.py` are in scope now: they hold the broker supervision
    and the panic path, so "no Python outside ouroboros/ touches a blob" was an
    assumption rather than a checked fact.
    """

    paths = sorted((REPO / "ouroboros").rglob("*.py")) + sorted((REPO / "supervisor").rglob("*.py"))
    if (REPO / "server.py").exists():
        paths.append(REPO / "server.py")
    return [(path.relative_to(REPO).as_posix(), path.read_text(encoding="utf-8")) for path in paths]


def test_every_blob_crossing_the_boundary_goes_through_the_transfer_service():
    """No `fetch_blob` / blob-mapping site may bypass the transfer service (§3.2).

    The registry is only closed if nothing reaches around it, and "nothing reaches
    around it" is a property of the call sites, not of the registry — so it is
    asserted over the source rather than trusted. See `blob_channel_sites` for the
    forms it reads and the BOUNDARY it admits.
    """

    offenders: list[str] = []
    for relative, source in _blob_scanned_sources():
        offenders.extend(blob_channel_offenders(relative, source))
    assert not offenders, (
        "a blob channel bypasses the transfer service — every crossing must route "
        "through it so the closed registry actually closes:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param("svc.fetch_blob(blob_id, 10)", id="direct_attribute"),
        pytest.param("ouroboros.remote_ssh.fetch_blob(a, b)", id="dotted_module_path"),
        pytest.param('getattr(svc, "fetch_blob")(blob_id, 10)', id="getattr_literal"),
        pytest.param('getattr(svc, "fetch_blob", None)', id="getattr_three_arg"),
        pytest.param('svc.__dict__["fetch_blob"](blob_id, 10)', id="dunder_dict"),
        pytest.param('vars(svc)["fetch_blob"](blob_id, 10)', id="vars_subscript"),
        pytest.param('table["fetch_blob"](svc)', id="string_literal_dispatch"),
        pytest.param("svc.ship(blobs=payload)", id="keyword"),
        pytest.param("svc.ship(input_blobs=payload)", id="keyword_alias"),
        pytest.param('svc.ship(**{"blobs": payload})', id="literal_dict_splat"),
        pytest.param('svc.ship(**{"process_blobs": payload})', id="literal_dict_splat_alias"),
    ],
)
def test_the_gate_sees_the_indirect_spellings_of_the_same_reach(spelling):
    """The evasions the keyword/attribute-only gate let through.

    `getattr(svc, 'fetch_blob')(…)` and `svc.ship(**{'blobs': …})` are the SAME
    crossing as the direct spellings and were both invisible: the scan matched a
    literal `Name`/`Attribute` named `fetch_blob` and a literal `keyword.arg`, and
    neither form has either.
    """

    source = f"def leak(svc, blob_id, payload, a, b):\n    return {spelling}\n"
    assert blob_channel_offenders("ouroboros/newly_added.py", source)


def test_the_gate_sees_an_aliased_import_and_a_splat_of_a_named_blob_dict():
    """Two forms that need statements, not one expression, to spell.

    An aliased import renames the primitive at the door, so the call site never spells
    it; a splat of a previously-built dict moves the keyword one statement away.
    """

    aliased = "from ouroboros.remote_ssh import fetch_blob as _fb\n\n\ndef leak(a):\n    return _fb(a)\n"
    assert blob_channel_offenders("ouroboros/newly_added.py", aliased)
    splatted = 'def leak(send, rows):\n    payload = {"blobs": rows}\n    return send(**payload)\n'
    assert blob_channel_offenders("ouroboros/newly_added.py", splatted)


@pytest.mark.parametrize(
    "relative",
    [
        "ouroboros/execd.py",
        "ouroboros/execd_state.py",
        "ouroboros/remote_reconciliation.py",
        "ouroboros/remote_service_leases.py",
        "ouroboros/remote_ssh.py",
        "ouroboros/remote_transfer.py",
        "ouroboros/remote_worker_proxy.py",
        "ouroboros/remote_workspace.py",
        "ouroboros/workspace_executor.py",
        "ouroboros/workspace_native.py",
        "ouroboros/workspace_payload_native.py",
        "ouroboros/workspace_snapshot_native.py",
    ],
)
def test_a_new_bypass_in_a_previously_EXEMPT_file_now_reddens_the_gate(relative):
    """The point of narrowing: these eleven files were exempt WHOLESALE.

    A synthetic bypass is planted in each of them, in a function name none of them
    has, and the gate must name it. Under the old file-level allowlist every one of
    these came back clean — including in the three files the remote work edits most.
    """

    source = "def newly_added_bypass(svc, blob_id):\n    return svc.fetch_blob(blob_id, 1)\n"
    assert blob_channel_offenders(relative, source) == [
        f"{relative}:newly_added_bypass:2: names fetch_blob"
    ]


def test_the_gate_leaves_the_routed_call_alone():
    """The service's own door must stay legal, or the gate selects for evasion."""

    routed = "def routed(svc, ref, op, args, rows):\n    return svc.export_operation(ref, op, args, blobs=rows)\n"
    assert blob_channel_offenders("ouroboros/newly_added.py", routed) == []


def test_the_gate_states_its_own_boundary():
    """The admitted residue stays named, and stays real.

    If one of the named blind spots starts being caught, the BOUNDARY paragraph is
    wrong and must be narrowed — asserting the blindness is what keeps the admission
    from going stale in either direction.
    """

    doc = blob_channel_sites.__doc__ or ""
    assert "BOUNDARY" in doc and "assembled at runtime" in doc
    assert (
        blob_channel_offenders("ouroboros/newly_added.py", "send(**{key: rows})\n") == []
    ), "a runtime-assembled keyword is now caught — narrow the BOUNDARY paragraph"


def test_the_allowlist_has_no_stale_rows():
    """An exemption that no longer describes real code is an exemption nobody reads.

    Five of the eleven file-level entries had gone stale — they exempted files with
    no blob site left at all — which is how an allowlist stops being a list of
    decisions and becomes background noise.
    """

    live = {
        (relative, function)
        for path, source in _blob_scanned_sources()
        for relative, function, _line, _why in blob_channel_sites(path, source)
    }
    stale = sorted(set(_BLOB_CHANNEL_ALLOWLIST) - live)
    assert not stale, f"allowlist rows describing no live blob site: {stale}"
    assert live == set(_BLOB_CHANNEL_ALLOWLIST), (
        "every live blob site must be an ALLOWLISTED one or a gate failure; "
        f"unaccounted: {sorted(live - set(_BLOB_CHANNEL_ALLOWLIST))}"
    )


def test_every_allowlisted_row_carries_a_justification():
    assert all(str(reason).strip() for reason in _BLOB_CHANNEL_ALLOWLIST.values())


def test_the_import_side_refuses_a_policy_home_never_issued():
    from ouroboros.remote_transfer import _validated_export_disclosure

    trace = {"snapshot": {"policy_hash": "d" * 64, "entries": []}}
    with pytest.raises(ExportPolicyViolation, match="Home never issued"):
        _validated_export_disclosure(None, trace)
    assert _validated_export_disclosure(None, {})["excluded_count"] == 0


# ── F2: the returned-manifest leak check must read EVERY channel ──────────────


def test_the_leak_check_reads_every_declared_manifest_channel():
    """The field list is DERIVED, not restated, and the declaration is complete.

    The shipped defect: `_manifest_entry_paths` restated three field names — the
    snapshot's `entries` and the patch export's two — and the `export_policy`
    disclosure block that `read_file`, `list_files` and every declared output emit
    carried none of them. So the leak check re-evaluated the policy over an EMPTY list
    and `validate_returned_manifest` passed on hash and arithmetic alone: a vacuous
    guard sitting behind the source-side hole it was there to catch.

    Two assertions, in both directions. The check must read every declared field, and
    every declared field must really be produced by the site the declaration names —
    an entry with no producer is the same silence with a different spelling.

    BOUNDARY. The first half is BEHAVIOURAL and complete: each declared field is fed to
    the real `_manifest_entry_paths` and its path must come back. The second half reads
    SOURCE, and its reading is narrow on purpose — it looks for the literal `"<field>"`
    in the declared producer's module, so a field emitted under an ASSEMBLED key
    (`{f"{prefix}_included": rows}`) or written by a module the declaration misnames
    would pass. It also cannot see a field some OTHER module emits without declaring it;
    that direction is covered behaviourally by
    `test_a_read_declares_the_path_it_exported_so_home_can_re_evaluate_it` for the block
    every channel shares, and by the manifest tests above for the snapshot and patch
    shapes. What no reading here can do is prove a FUTURE channel declared itself — the
    same "silence is unreadable" limit the Document Truth Rule states. The compensating
    control is that the leak check no longer has a field list of its own to fall behind
    with, which is what actually failed.
    """

    import ast
    import inspect

    from ouroboros.export_policy_contract import MANIFEST_EXPORTED_PATH_FIELDS
    from ouroboros.remote_export_policy import _manifest_entry_paths

    for field in MANIFEST_EXPORTED_PATH_FIELDS:
        paths = _manifest_entry_paths({field: ["sentinel/path.txt"]})
        assert paths == ["sentinel/path.txt"], (
            f"the leak check does not read the declared manifest field {field!r}"
        )
    source = inspect.getsource(_manifest_entry_paths)
    literals = {
        node.value
        for node in ast.walk(ast.parse(source.lstrip()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not (literals & set(MANIFEST_EXPORTED_PATH_FIELDS)), (
        "the leak check restates a declared field name instead of deriving the list; "
        "a restated list is the second authority that went stale the first time"
    )
    modules = {
        producer.split(".")[0] for producer in MANIFEST_EXPORTED_PATH_FIELDS.values()
    }
    for module in modules:
        text = (REPO / "ouroboros" / f"{module}.py").read_text(encoding="utf-8")
        emitted = [
            field
            for field, producer in MANIFEST_EXPORTED_PATH_FIELDS.items()
            if producer.startswith(module + ".")
        ]
        for field in emitted:
            assert f'"{field}"' in text, (
                f"{module} is declared as the producer of {field!r} and never writes it"
            )


def test_a_read_declares_the_path_it_exported_so_home_can_re_evaluate_it(tmp_path):
    """The `exported[]` half of the disclosure, end to end through the real door.

    Without it the Home backstop has nothing to judge. With it, a manifest that claims
    to have exported an excluded path is refused — which is the property F2 was supposed
    to have all along, asserted here by FORCING the claim rather than by trusting it.
    """

    from ouroboros.export_policy_contract import build_policy_document
    from ouroboros.remote_export_policy import (
        ExportPolicy,
        ExportPolicyViolation,
        export_policy_hash,
        validate_returned_manifest,
    )
    from ouroboros.workspace_native import execute_native_operation

    root = (tmp_path / "ws").resolve()
    root.mkdir(parents=True)
    (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
    document = build_policy_document(channel="workspace_query")
    facts = {"export_policy": document}
    envelope = execute_native_operation(
        root, "read_file", {"path": "app.py"}, native_facts=facts
    ).envelope

    block = envelope.trace["export_policy"]
    assert block["exported"] == ["app.py"], block
    policy = ExportPolicy(
        channel="workspace_query",
        document=document,
        policy_hash=export_policy_hash(document),
    )
    validate_returned_manifest(policy, block)
    with pytest.raises(ExportPolicyViolation):
        validate_returned_manifest(policy, {**block, "exported": ["app.py", ".env"]})


def test_a_declared_output_applies_EVERY_rule_the_document_carries(tmp_path):
    """The door read ONE rule group out of two, and this is the case that proves it.

    `test_a_declared_output_loses_the_member_not_the_artifact` above uses a `secrets/`
    component, which the deliverable COMPONENT rule catches on its own — so it stayed
    green while the door was asking that rule and nothing else. The two members here are
    exactly the ones that group cannot see:

    * `id_rsa` — caught by the credential-name PREFIX rule, which lives in the ladder;
    * `golden.bin` — caught only because Home listed it in `protected_paths`, and those
      are WORKSPACE-relative, so a member judged relative to its declared output could
      never match one even if the rule had been asked.

    Both shipped. The second shipped with no disclosure at all, which is the worse half.
    """

    from ouroboros.export_policy_contract import build_policy_document

    root = (tmp_path / "ws").resolve()
    (root / "dist").mkdir(parents=True)
    (root / "dist" / "id_rsa").write_text("PRIVATE-KEY-SENTINEL\n", encoding="utf-8")
    (root / "dist" / "golden.bin").write_text("PROTECTED-SENTINEL\n", encoding="utf-8")
    (root / "dist" / "report.txt").write_text("ordinary\n", encoding="utf-8")
    document = build_policy_document(
        channel="declared_output", protected_paths=["dist/golden.bin"]
    )

    blobs, artifacts, notes, failed, excluded, exported = collect_declared_outputs(
        root, {"cwd": str(root), "outputs": ["dist"]}, {}, document
    )

    assert [row["member_path"] for row in artifacts] == ["report.txt"], (
        "a declared output exported a member the document excludes"
    )
    assert exported == ["dist/report.txt"]
    assert sorted(excluded, key=lambda row: row["path"]) == [
        {
            "path": "dist/golden.bin",
            "reason": "protected_artifact",
            "judged": "dist/golden.bin",
        },
        {"path": "dist/id_rsa", "reason": "sensitive_file", "judged": "dist/id_rsa"},
    ], "every exclusion must be DISCLOSED, not merely performed"
    for sentinel in (b"PRIVATE-KEY-SENTINEL", b"PROTECTED-SENTINEL"):
        assert all(sentinel not in payload for payload in blobs.values())
    assert failed is False, "a policy exclusion is disclosed work, not a failure"
    assert any("golden.bin" in note for note in notes)


def test_a_walk_channel_cannot_return_an_excluded_inode_under_a_clean_name(tmp_path):
    """`search_code` returned the secret LINE through a hardlink. One walk got this right.

    The snapshot seeded excluded inodes inline and was therefore the only channel that
    judged identity; the query walk judged spellings, and a `notes.txt` sharing `.env`'s
    inode passes every rule a document can state. So the same query that disclosed `.env`
    as excluded returned `SECRET_TOKEN=hunter2` from its alias two lines above.

    Asserted on the returned TEXT for the matched line, and on the disclosure naming
    every alias — a walk that dropped them silently would be the D7 half of the same bug.
    Both aliases are inside the root, so the confinement rule cannot stand in for the
    policy one.
    """

    import os

    from ouroboros.export_policy_contract import build_policy_document
    from ouroboros.workspace_native import execute_native_operation

    root = (tmp_path / "ws").resolve()
    (root / "sub").mkdir(parents=True)
    (root / ".env").write_text("SECRET_TOKEN=hunter2\n", encoding="utf-8")
    os.link(root / ".env", root / "notes.txt")
    os.link(root / ".env", root / "sub" / "deep.txt")
    os.symlink(root / ".env", root / "safe.txt")
    facts = {"export_policy": build_policy_document(channel="workspace_query")}

    envelope = execute_native_operation(
        root, "search_code", {"query": "SECRET_TOKEN", "path": "."}, native_facts=facts
    ).envelope

    assert "SECRET_TOKEN=hunter2" not in envelope.text, (
        "a walk channel returned an excluded file's line content through an alias"
    )
    for alias in (".env", "notes.txt", "sub/deep.txt", "safe.txt"):
        assert alias in envelope.text, (
            f"{alias} was filtered without being disclosed; a silent omission makes a "
            "no-match answer read as authoritative (D7)"
        )
    rows = envelope.trace["export_policy"]["excluded"]
    assert {row["path"] for row in rows} == {".env", "notes.txt", "safe.txt", "sub/deep.txt"}


def test_the_identity_judgement_has_exactly_one_owner(tmp_path, monkeypatch):
    """Two mechanics for one question is how the weaker one became the hole.

    Proved by SUBSTITUTION rather than by reading the source: `judged_exclusion` is
    replaced with one that judges nothing, and EVERY channel — both walks AND the
    single-source read door — must lose its identity check together. If any one of them
    survives the substitution, it is answering the identity question from somewhere else,
    which is precisely the shape that shipped: the walks seeded a recursive inode set and
    the doors used a root-only `scandir`, so a hardlink to `sub/.env` was excluded from
    `search_code` and returned verbatim by `read_file`.
    """

    import os

    from ouroboros import export_policy_contract, workspace_query_native
    from ouroboros import workspace_native, workspace_snapshot_native
    from ouroboros.export_policy_contract import AliasIndex, build_policy_document
    from ouroboros.workspace_native import execute_native_operation

    root = (tmp_path / "ws").resolve()
    (root / "sub").mkdir(parents=True)
    (root / "sub" / ".env").write_text("SECRET_TOKEN=hunter2\n", encoding="utf-8")
    # NESTED, and aliased from both depths: the two cells the root-only probe missed.
    os.link(root / "sub" / ".env", root / "sub" / "notes.txt")
    os.link(root / "sub" / ".env", root / "rootlevel.txt")
    facts = {"export_policy": build_policy_document(channel="workspace_query")}

    def blind(*_args, **_kwargs):
        return "", "", ""

    for module in (
        workspace_query_native, workspace_snapshot_native, workspace_native,
        export_policy_contract,
    ):
        monkeypatch.setattr(module, "judged_exclusion", blind, raising=False)
    envelope = execute_native_operation(
        root, "search_code", {"query": "SECRET_TOKEN", "path": "."}, native_facts=facts
    ).envelope
    assert "SECRET_TOKEN=hunter2" in envelope.text, (
        "the query walk did not lose its identity check when the shared judge was "
        "blinded, so it is judging identity somewhere else"
    )
    read = execute_native_operation(
        root, "read_file", {"path": "sub/notes.txt"}, native_facts=facts
    ).envelope
    assert "SECRET_TOKEN=hunter2" in read.text, (
        "the read door did not lose its identity check either — a SECOND mechanic"
    )
    monkeypatch.undo()

    # …and unblinded, the one owner really answers for every depth and direction,
    # including an explicit protected path under a PRUNED directory.
    (root / ".git").mkdir()
    (root / ".git" / "hidden.bin").write_text("x", encoding="utf-8")
    os.link(root / ".git" / "hidden.bin", root / "innocent.bin")
    document = build_policy_document(
        channel="workspace_snapshot", protected_paths=[".git/hidden.bin"]
    )
    index = AliasIndex(root, document)
    for alias, real in (
        ("sub/notes.txt", "sub/.env"),
        ("rootlevel.txt", "sub/.env"),
        ("innocent.bin", ".git/hidden.bin"),
    ):
        assert index.alias(root / alias)[0] == real, alias
    assert index.alias(root / "sub" / ".env")[0] == "sub/.env"
