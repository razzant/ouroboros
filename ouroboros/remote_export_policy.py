"""The HOME authority over what may leave a remote host (RWS v2 §3.2).

One policy, two ends, and a hash in between.

Home DECIDES: only here does anything read a task's contract, its resource
policy and `protected_artifacts` to work out which paths are off-limits for a
given operation, because deciding needs Home authorities and there is one brain
(the inherited standing decision — execd carries no policy authority). The
outcome is a document, hashed, that rides with the prepared operation.

The SOURCE APPLIES: execd evaluates that document mechanically, before it
constructs any blob. That ordering is the whole point. Filtering a blob after it
has been fetched is not filtering — the bytes already crossed, they are already
in a Home temp file, and the only thing "filtering" removes at that point is the
evidence. So the excluded bytes are never read on the target, and the manifest
that comes back says which paths were omitted and under which policy hash.

"Never READ", not "never exported", and the distinction is load-bearing: a door
that reads in order to write is still reading. `apply_patch` and `edit_batch`
asked only the WRITE question, which by design narrows to protected artifacts —
so they located hunks and counted occurrences inside files whose `read_file` the
same task had just been refused, and answered about them. Both ask both questions
now (`workspace_edit_native`), which is what makes the sentence above true.

Home then REVALIDATES: `validate_returned_manifest` re-evaluates the same
document over the paths that actually came back. This is not ceremony. It is the
only way Home can tell "the source applied my policy" from "the source said it
did": a returned entry the policy excludes means the filter did not run, and that
is a loud typed `ExportPolicyViolation` rather than a partial import, because an
import that quietly accepts unpoliced bytes is worse than a failed one.

Finally, D7: a policy exclusion is DISCLOSED WORK, not a failure. The manifest
keeps the donor's own wire semantics (`complete=false`, `integrity_complete=true`,
`policy_scope=policy_filtered`) with no schema bump, and `export_disclosure`
projects that into the additive block that travels to `artifact_bundle`, the
verification ledger, task-acceptance evidence, the CLI and the owner-facing
result. IO failures and unstable snapshots remain fail-closed and are a DIFFERENT
condition — the donor conflated them, and one `.env` in a remote repo therefore
took plan review, the claude_code_edit bridge and the file bridge down with it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from ouroboros.export_policy_contract import (
    MANIFEST_TRACE_KEYS,
    EXPORT_QUESTIONS,
    EXPORT_REASONS,
    MANIFEST_EXPORTED_PATH_FIELDS,
    MAX_DISCLOSED_EXCLUSIONS,
    RESTRICTED_MARKER_COMPLETIONS,
    RESTRICTED_MARKER_SUFFIXES,
    QUESTION_EXPORT,
    RESTRICTED_SENSITIVE_SUFFIXES,
    build_policy_document,
    channel_profile,
    describe_exclusion,
    unaliased_exclusion,
    export_policy_hash,
    normalize_export_policy,
)

# Which channel's profile an operation defaults to. A single operation can emit
# several blob kinds (a `run_command` produces process output AND declared
# outputs), so this names the operation's COARSEST channel; a door that emits a
# stricter kind tightens the profile itself (declared-output collection does).
# An operation missing from this map gets the tree profile, which is the
# conservative direction: it filters, it just does not apply the deliverable
# rules meant for paths the model chose to publish.
OPERATION_EXPORT_CHANNEL: dict[str, str] = {
    "snapshot_manifest_and_blob_export": "workspace_snapshot",
    "guarded_patch_apply": "workspace_snapshot",
    "vcs_diff": "workspace_patch",
    "vcs_status": "workspace_query",
    "search_code": "workspace_query",
    "query_code": "workspace_query",
    "list_files": "workspace_query",
    "read_file": "workspace_query",
    "run_command": "declared_output",
    "run_script": "declared_output",
    "start_service": "declared_output",
    "stop_service": "declared_output",
    "service_logs": "process_spool_log",
    "extract_video_frames": "media_frames",
}
_DEFAULT_OPERATION_CHANNEL = "workspace_query"

# The channels a HOME-initiated door produces for, and the door that produces each
# (D12). `OPERATION_EXPORT_CHANNEL` above maps only the operations the tool DISPATCH
# routes; these are exports Home starts itself — task admission, a vision faculty, a
# patch integration — and they bind their channel at the call site because there is no
# tool name to look up. Both tables together are the answer to "which half of the closed
# registry is live", which is asserted in tests/test_remote_export_policy.py: a channel
# declared with no producer anywhere is a contract with one end missing, and that is
# exactly what these five were before their Home halves landed.
HOME_CHANNEL_PRODUCERS: dict[str, str] = {
    "attachment_stage": "remote_task_files.export_task_attachments",
    "media_frames": "remote_task_files.import_media_from_target",
    "subagent_patch": "tools/subagent_integration._integrate_remote_subagent_patch",
    "workspace_snapshot": "remote_transfer.materialize_remote_snapshot",
    "workspace_patch": "remote_plan_review.remote_turn_diff",
    "declared_output": "tools/verify._verify_on_remote_target",
}


class ExportPolicyViolation(RuntimeError):
    """A returned manifest the Home policy cannot vouch for. Loud, never quiet."""

    code = "REMOTE_EXPORT_POLICY_VIOLATION"

    def __init__(self, detail: str):
        super().__init__(f"{self.code}: {detail}")


@dataclasses.dataclass(frozen=True)
class ExportPolicy:
    """One channel's policy document plus the hash that binds it."""

    channel: str
    document: dict[str, Any]
    policy_hash: str

    def arg_payload(self) -> dict[str, Any]:
        """The underscore-prefixed prepare argument the target reads.

        Underscore-prefixed keys are stripped from `execution_args` by the target's
        own prepare, so the policy shapes the operation's FACTS without becoming
        part of the argv the guards authorize — it is an input to the export, not
        an argument of the tool.
        """

        return {"_export_policy": dict(self.document)}


def channel_for_operation(tool: str) -> str:
    """The declared channel for a native operation; unknown channels fail closed."""

    channel = OPERATION_EXPORT_CHANNEL.get(str(tool or ""), _DEFAULT_OPERATION_CHANNEL)
    channel_profile(channel)
    return channel


def workspace_relative_protected_paths(
    ctx: Any, *, workspace_root: str = ""
) -> tuple[str, ...]:
    """Project the task's protected artifacts onto TARGET-relative spellings.

    `protected_artifacts` stays the single source of the rules — this reads its
    records and does not restate them. What it does do is spell them the way the
    target will see them, which Home has to, because the target has no Home paths
    to compare against:

    * a relative entry is already workspace-relative and travels as-is;
    * an absolute entry is kept only when it lies under the remote workspace root,
      because an absolute Home path has no meaning on the target and matching it
      loosely would exclude an unrelated remote file with a similar tail.
    """

    from ouroboros.protected_artifacts import _artifact_records

    root = str(workspace_root or "").replace("\\", "/").rstrip("/")
    relatives: list[str] = []
    for record in _artifact_records(ctx):
        for raw in record.get("paths") or []:
            text = str(raw or "").strip().replace("\\", "/")
            if not text:
                continue
            if not text.startswith("/"):
                relatives.append(text.strip("/"))
                continue
            if root and text == root:
                continue
            if root and text.startswith(root + "/"):
                relatives.append(text[len(root) + 1 :].strip("/"))
    return tuple(sorted({item for item in relatives if item}))


def restricted_reader_rules(ctx: Any) -> dict[str, list[str]]:
    """The rule tokens a RESTRICTED SUBAGENT's operations are tightened with.

    `is_restricted_subagent_profile` is the fail-closed SSOT for "this child may
    not read owner secrets or control state", and its predicate covers classes the
    default export tables do not: a `secrets/`, `auth/` or `tokens/` component, and
    the owner-control filenames (`settings.json`, `trust.json`, …). Before this, the
    denial existed only as a Home-side refusal — so a subagent's remote `search_code`
    over an ordinary directory still read those files ON THE TARGET and shipped the
    matching lines, while the identical local call never opened them.

    Expressed as rule TOKENS rather than as a second predicate on purpose (§3.2):
    the source keeps exactly one mechanical evaluator over one hashed document, and
    "this operation has a stricter reader" changes the document, not the engine.
    """

    from ouroboros.tools.core import (
        _SKILL_OWNER_STATE_FILENAMES,
        _SUBAGENT_SECRET_FILE_NAMES,
        is_restricted_subagent_profile,
    )

    if not is_restricted_subagent_profile(ctx):
        return {}
    return {
        "sensitive_components": ["auth", "credentials", "secrets", "tokens"],
        "sensitive_names": sorted(
            _SUBAGENT_SECRET_FILE_NAMES | _SKILL_OWNER_STATE_FILENAMES
        ),
        # The PATTERN half of the same denial. Names and components are enumerable;
        # `db_password.conf` / `api_key.yaml` / `x.env` are not — Home catches those with
        # a delimited-marker regex scoped to config-ish suffixes, and until now the
        # document had no field of that shape. So a restricted subagent's remote
        # `search_code` read the contents of a `db_password.conf` its own `read_file`
        # refused: the pipeline's spelling rule cannot cover a WALK, because a walk names
        # no path for it to judge.
        #
        # Both tables come from `export_policy_contract`, which owns every rule table:
        # spelling them here would be a private copy of one.
        "credential_markers": list(RESTRICTED_MARKER_COMPLETIONS),
        "marker_scoped_suffixes": list(RESTRICTED_MARKER_SUFFIXES),
        "sensitive_suffixes": list(RESTRICTED_SENSITIVE_SUFFIXES),
    }


def build_export_policy(
    ctx: Any,
    *,
    channel: str,
    workspace_root: str = "",
) -> ExportPolicy:
    """Compute the hash-bound policy for ONE operation on ONE channel."""

    document = build_policy_document(
        channel=channel,
        protected_paths=workspace_relative_protected_paths(
            ctx, workspace_root=workspace_root
        ),
        extra_rules=restricted_reader_rules(ctx),
    )
    return ExportPolicy(
        channel=channel, document=document, policy_hash=export_policy_hash(document)
    )


def policy_for_operation(ctx: Any, tool: str, *, workspace_root: str = "") -> ExportPolicy:
    """The policy a native operation is prepared under."""

    return build_export_policy(
        ctx, channel=channel_for_operation(tool), workspace_root=workspace_root
    )


def _manifest_exclusions(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    """The disclosed exclusion rows, under whichever of the wire names is present.

    The snapshot manifest calls them `policy_exclusions`, the patch export and the
    declared-output disclosure call them `excluded`; both are the donor's existing
    shapes and D7 explicitly declines to bump the schema over the difference.
    """

    for key in ("excluded", "policy_exclusions", "exclusions"):
        rows = manifest.get(key)
        if isinstance(rows, list):
            return [
                {
                    "path": str(row.get("path") or ""),
                    "reason": str(row.get("reason") or ""),
                    # The spelling the policy actually excluded. It differs from `path`
                    # exactly when an ALIAS was the finding — `notes.txt` dropped because
                    # it IS `.env` — and without it Home has a claim it cannot check.
                    "judged": str(row.get("judged") or row.get("path") or ""),
                }
                for row in rows
                if isinstance(row, Mapping)
            ]
    return []


def _manifest_entry_paths(manifest: Mapping[str, Any]) -> list[str]:
    """The paths the source actually EXPORTED, whatever the manifest shape.

    The field list is DERIVED from the contract's declaration
    (`export_policy_contract.MANIFEST_EXPORTED_PATH_FIELDS`), not restated here, for the
    same reason `remote_task_files._WIRE_FIELDS` is derived from the attachment
    contract: a restated list is a second authority that goes stale silently. This one
    did. It knew `entries`, `tracked_changed` and `untracked_included` — the snapshot's
    and the patch export's fields — and the `export_policy` disclosure block that
    `read_file`, `list_files` and every declared output emit carried none of them, so
    this returned `[]` and the leak check downstream re-evaluated the policy over
    nothing. `validate_returned_manifest` then passed on the hash and the arithmetic,
    which is exactly the vacuous guard the Guard Proof Rule names — and it was the
    Home-side backstop for the source-side hole that shipped alongside it.
    """

    paths: list[str] = []
    for key in sorted(MANIFEST_EXPORTED_PATH_FIELDS):
        rows = manifest.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                paths.append(str(row.get("path") or ""))
            elif isinstance(row, str):
                paths.append(row)
    return [item for item in paths if item]


def validate_returned_manifest(
    policy: ExportPolicy, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-evaluate the SAME policy over what came back; refuse on any disagreement.

    Four things are checked, and each one exists because its absence has a silent
    failure mode:

    1. the hash — a manifest produced under a different document describes a
       filtering nobody can name;
    2. the exported paths — a returned entry the policy excludes proves the filter
       did not run, whatever the manifest claims;
    3. the arithmetic — an `excluded_count` that disagrees with an untruncated
       `excluded[]` means the disclosure is not a description of the omission;
    4. the reason codes — an unknown reason is an unknown policy.

    Returns the validated disclosure. Raises :class:`ExportPolicyViolation`
    otherwise: a violation is a refusal, never a partial import, because accepting
    unpoliced bytes is worse than failing to accept any.
    """

    claimed = str(manifest.get("policy_hash") or "")
    if not claimed:
        raise ExportPolicyViolation(
            f"channel={policy.channel}: the returned manifest declares no policy hash, "
            "so nothing can say which rules produced it"
        )
    if claimed != policy.policy_hash:
        raise ExportPolicyViolation(
            f"channel={policy.channel}: manifest was produced under policy {claimed[:16]} "
            f"but Home authorized {policy.policy_hash[:16]}"
        )
    # The QUESTION the source applied, so Home applies the same projection of one
    # document. Without it Home judged every returned path under `QUESTION_EXPORT` while
    # `read_file('.git/config')` is legal at the source under `QUESTION_NAMED_SOURCE` —
    # so a legitimate read raised a violation AFTER its bytes had crossed the boundary.
    # An unrecognized question fails closed rather than defaulting.
    question = str(manifest.get("policy_question") or QUESTION_EXPORT)
    if question not in EXPORT_QUESTIONS:
        raise ExportPolicyViolation(
            f"channel={policy.channel}: manifest was produced under policy question "
            f"{question!r}, which this build does not understand"
        )
    if bool(manifest.get("exported_disclosure_truncated")):
        # A TRUNCATED evidence list makes this whole check a sample, and a sample cannot
        # prove the filter ran. The bound is `MAX_EXPORTED_PATHS`, above every per-channel
        # result cap, so reaching it means the source exceeded a limit it already fails
        # closed on. Accepting unpoliced bytes is worse than accepting none.
        raise ExportPolicyViolation(
            f"channel={policy.channel}: the manifest truncated its exported-path evidence "
            f"({manifest.get('exported_count')} paths), so source-side filtering cannot be "
            "re-checked; refusing rather than accepting an unverified export"
        )
    leaked = sorted(
        path
        for path in _manifest_entry_paths(manifest)
        if unaliased_exclusion(path, policy.document, question=question)[0]
    )
    if leaked:
        raise ExportPolicyViolation(
            f"channel={policy.channel}: the manifest exports {len(leaked)} path(s) the "
            f"policy excludes, so source-side filtering did not run: {leaked[:10]}"
        )
    rows = _manifest_exclusions(manifest)
    unknown = sorted({row["reason"] for row in rows} - EXPORT_REASONS)
    if unknown:
        raise ExportPolicyViolation(
            f"channel={policy.channel}: manifest discloses unknown exclusion reasons {unknown}"
        )
    # Every disclosed EXCLUSION is re-derived too, and this half was missing entirely: Home
    # checked that a reason code was in the closed set and never that the policy actually
    # gives that reason for that path. A target could therefore claim `src/app.py` had been
    # excluded as `sensitive_file` and Home would show the owner an omission that never
    # happened — a false disclosure is a lie in the direction the owner cannot detect,
    # because there is nothing missing to notice. The `judged` spelling is what makes this
    # checkable for an ALIAS row, whose own `path` is innocent by construction.
    invented = sorted(
        f"{row['path']} claimed {row['reason']}"
        for row in rows
        if unaliased_exclusion(row["judged"], policy.document, question=question)[0]
        != row["reason"]
    )
    if invented:
        raise ExportPolicyViolation(
            f"channel={policy.channel}: the manifest discloses {len(invented)} exclusion(s) "
            f"this policy does not produce, so the disclosure is not a description of what "
            f"was omitted: {invented[:10]}"
        )
    declared_count = manifest.get("excluded_count", manifest.get("policy_excluded_count"))
    count = len(rows) if declared_count is None else int(declared_count)
    truncated = bool(manifest.get("excluded_disclosure_truncated"))
    if not truncated and count != len(rows):
        raise ExportPolicyViolation(
            f"channel={policy.channel}: manifest claims {count} exclusions but discloses "
            f"{len(rows)}; the count and the list must describe the same omission"
        )
    if bool(manifest.get("integrity_complete", True)):
        scope = str(manifest.get("policy_scope") or ("policy_filtered" if count else "full"))
        expected_scope = "policy_filtered" if count else "full"
        if scope != expected_scope:
            raise ExportPolicyViolation(
                f"channel={policy.channel}: policy_scope={scope!r} disagrees with "
                f"{count} disclosed exclusions"
            )
    return export_disclosure(manifest, channel=policy.channel, policy=policy)


def export_disclosure(
    manifest: Mapping[str, Any],
    *,
    channel: str = "",
    policy: ExportPolicy | None = None,
) -> dict[str, Any]:
    """The ADDITIVE D7 block every Home surface reads.

    Additive is the operative word: terminal statuses are untouched. A
    policy-filtered export is a completed operation whose omission is written
    down, so the block adds `partial`, the exact `excluded_count`, a bounded
    disclosed `excluded[]`, the applied `policy_hash` and the full-manifest hash —
    and changes nothing that decides success.
    """

    rows = _manifest_exclusions(manifest)
    declared = manifest.get("excluded_count", manifest.get("policy_excluded_count"))
    count = len(rows) if declared is None else int(declared)
    integrity_complete = bool(manifest.get("integrity_complete", True))
    return {
        "schema_version": 1,
        "channel": str(channel or manifest.get("channel") or ""),
        # `partial` is the DERIVED state D7 names: filtered but whole, as opposed
        # to an integrity failure, which is fail-closed and never lands here.
        "partial": bool(count) and integrity_complete,
        "integrity_complete": integrity_complete,
        "policy_scope": str(manifest.get("policy_scope") or ("policy_filtered" if count else "full")),
        "policy_hash": str(manifest.get("policy_hash") or (policy.policy_hash if policy else "")),
        "excluded_count": count,
        "excluded": [
            {
                "path": row["path"],
                "reason": row["reason"],
                "disclosure": describe_exclusion(row["path"], row["reason"]),
            }
            for row in rows[:MAX_DISCLOSED_EXCLUSIONS]
        ],
        "excluded_disclosure_truncated": bool(
            manifest.get("excluded_disclosure_truncated") or count > len(rows)
        ),
        "full_manifest_sha256": str(
            manifest.get("fingerprint") or manifest.get("sha256") or ""
        ),
    }


def merge_export_disclosures(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold one operation's per-channel disclosures into one owner-facing block.

    An operation crosses several channels (a `run_command` exports process output
    and declared outputs), and the owner is owed ONE number. Counts add; the list
    is bounded once, at the end, so a channel with many exclusions cannot crowd out
    the disclosure of another channel's few.
    """

    merged: list[dict[str, Any]] = []
    total = 0
    channels: list[str] = []
    truncated = False
    integrity_complete = True
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        total += int(row.get("excluded_count") or 0)
        truncated = truncated or bool(row.get("excluded_disclosure_truncated"))
        integrity_complete = integrity_complete and bool(row.get("integrity_complete", True))
        if row.get("channel"):
            channels.append(str(row["channel"]))
        for item in row.get("excluded") or []:
            if isinstance(item, Mapping):
                merged.append({**dict(item), "channel": str(row.get("channel") or "")})
    return {
        "schema_version": 1,
        "channels": sorted(set(channels)),
        "partial": bool(total) and integrity_complete,
        "integrity_complete": integrity_complete,
        "policy_scope": "policy_filtered" if total else "full",
        "excluded_count": total,
        "excluded": merged[:MAX_DISCLOSED_EXCLUSIONS],
        "excluded_disclosure_truncated": truncated or len(merged) > MAX_DISCLOSED_EXCLUSIONS,
    }


def validate_operation_trace(
    policy: ExportPolicy, trace: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every policy-bearing manifest in one result trace, then merge.

    The manifests live under the keys the donor already used — `snapshot`,
    `patch_export`, `export_policy` — and each is validated against the SAME
    document before any of it is believed. A trace that carries no manifest at all
    yields the empty disclosure rather than an assumed-clean one, because "this
    operation exported nothing" and "this operation exported without a policy" must
    not read the same downstream.

    PRESENCE selects, not the hash. Gating on `policy_hash` truthiness handed the
    target the choice: a manifest that simply omitted the field was skipped whole,
    so it never reached `validate_returned_manifest` — including that function's
    OWN refusal for exactly this case ("declares no policy hash, so nothing can say
    which rules produced it") and, worse, its exported-path leak re-check. The
    strictly less attested manifest was treated strictly more leniently than a
    wrong one, and it read downstream as `policy_scope: "full"`: a positive claim
    that nothing was withheld. Membership in `MANIFEST_TRACE_KEYS` is already the
    signal that the operation is policy-bearing, so presence is the selector and
    the refusal below is the judge.
    """

    disclosures: list[dict[str, Any]] = []
    for key, manifest in trace.items():
        if not isinstance(manifest, Mapping):
            continue
        if key in MANIFEST_TRACE_KEYS:
            disclosures.append(validate_returned_manifest(policy, manifest))
        elif "policy_hash" in manifest:
            # A manifest under a key nobody declared. `guarded_patch_apply` shipped
            # exactly this — two full manifests under `before`/`after` while its own
            # failure arms used the declared `snapshot` — and the loop that reads only
            # the declared list cannot see such a key BY CONSTRUCTION, so it passed in
            # silence and Home recorded "nothing was withheld". Reading the trace
            # itself is what turns "declared nowhere" from invisible into loud.
            raise ExportPolicyViolation(
                f"the trace carries a policy manifest under the undeclared key {key!r}: "
                "a manifest that is not in MANIFEST_TRACE_KEYS is judged by nothing"
            )
    return merge_export_disclosures(disclosures)


def bundle_export_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """The additive D7 keys an `artifact_bundle` carries, or ``{}``.

    Lives here rather than in `outcomes` because this is where the shape of a
    disclosure is decided; `outcomes` owns statuses, and the whole of D7 is that a
    disclosure is NOT a status.
    """

    remote_export = result.get("remote_export")
    if not isinstance(remote_export, Mapping):
        return {}
    count = int(remote_export.get("excluded_count") or 0)
    if not count:
        return {}
    return {
        "partial": bool(remote_export.get("partial")),
        "excluded_count": count,
        "excluded": list(remote_export.get("excluded") or []),
        "excluded_disclosure_truncated": bool(remote_export.get("excluded_disclosure_truncated")),
        "export_policy_hash": str(remote_export.get("policy_hash") or ""),
    }


def apply_export_ledger_entry(
    entries: list[dict[str, Any]],
    artifact_bundle: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Append the filtered-export ledger entry, if this export was filtered.

    `policy_filtered` is a NON-FAILURE ledger status on purpose: the owner's own
    policy omitting a file is honest telemetry, not a verification failure, and
    conflating the two is exactly what D7 forbids. The entry exists so the evidence
    NAMES the omission — a reader should not have to notice an absence.
    """

    bundle = artifact_bundle or {}
    count = int(bundle.get("excluded_count") or 0)
    if not count:
        return entries
    return [*entries, {
        "kind": "remote_export",
        "status": "policy_filtered",
        "excluded_count": count,
        "excluded": list(bundle.get("excluded") or []),
        "export_policy_hash": str(bundle.get("export_policy_hash") or ""),
    }]


def disclosure_summary_line(disclosure: Mapping[str, Any]) -> str:
    """One honest sentence for the model, the CLI and the owner — or nothing."""

    count = int(disclosure.get("excluded_count") or 0)
    if not count:
        return ""
    names = [
        str(row.get("path") or "")
        for row in list(disclosure.get("excluded") or [])[:5]
        if isinstance(row, Mapping)
    ]
    tail = f": {', '.join(name for name in names if name)}" if names else ""
    more = " (list bounded)" if disclosure.get("excluded_disclosure_truncated") else ""
    return (
        f"ℹ️ REMOTE_EXPORT_POLICY_FILTERED: {count} path(s) were excluded from this "
        f"export by the owner's policy; everything else transferred and verified"
        f"{tail}{more}."
    )


__all__ = [
    "HOME_CHANNEL_PRODUCERS",
    "OPERATION_EXPORT_CHANNEL",
    "ExportPolicy",
    "ExportPolicyViolation",
    "apply_export_ledger_entry",
    "build_export_policy",
    "bundle_export_fields",
    "channel_for_operation",
    "disclosure_summary_line",
    "export_disclosure",
    "merge_export_disclosures",
    "export_policy_hash",
    "normalize_export_policy",
    "policy_for_operation",
    "restricted_reader_rules",
    "validate_operation_trace",
    "validate_returned_manifest",
    "workspace_relative_protected_paths",
]
