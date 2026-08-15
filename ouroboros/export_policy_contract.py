"""The export-policy DOCUMENT and its one mechanical evaluator (RWS v2 §3.2).

This module is deliberately the dumbest thing in the export boundary: a
serializable rule document, a canonical hash over it, and one predicate that
answers "may this relative path leave the host?".  It holds no authority — it
decides nothing about a task, reads no context, and imports nothing but the
standard library and the equally dependency-free contract register
(``remote_contracts``, which names this document as one of the contracts a build
pair must share and supplies the typed refusal it fails closed with) — which is
exactly why BOTH sides of the boundary may import it.

One judge here does touch the filesystem, and it is named so the exception is
visible: ``judged_exclusion``.  An ALIAS is not a spelling, so no document can
state the rule that catches it — the rule has to ask which file this is.  It still
decides nothing new: it applies the same document to the name the bytes really have.

That judge is also the ONLY one on the target side, and that is the point rather than
tidiness.  There used to be four appliers (``refuse_excluded_named_source``,
``refuse_aliased_named_source``, ``refuse_protected_mutation``,
``refuse_excluded_source``), each judging the REQUESTED spelling and each free to ask a
different subset of the document.  Two of them then resolved the path and read or wrote
the target — so ``safe.txt -> .env`` returned the secret the direct spelling was
refused, and a symlink alias overwrote a protected artifact with no policy check on the
write side at all.  A spelling is what the caller asked for; the IDENTITY is what the
filesystem will hand over, and only the second one can be judged honestly.

Collapsing four appliers into one was HALF the fix, and the half that shipped left the
other half open for a paid reviewer to find.  Identity was then answered by TWO
mechanics: the WALK channels seeded a recursive inode set, while the single-source doors
used a root-only ``scandir`` probe — so a hardlink to ``sub/.env`` was excluded from
``search_code`` and returned verbatim by ``read_file``, and one to a nested protected
artifact was mutated through.  There is now ONE function that answers "which file is
this, and does the policy exclude it" — :func:`judged_exclusion`, recursive, used by
every door and every producer of bytes.  Two mechanics for one question is a guarantee
that the weaker one becomes the hole; the general form of that lesson is in
docs/DEVELOPMENT.md under the axis clause of the Guard Proof Rule.

The one entry point that judges a SPELLING alone is :func:`unaliased_exclusion`, and it
is named for what it CANNOT do.  It exists because Home holds no workspace: a returned
manifest is a list of strings, and re-deriving identity from them is impossible, which
is why Home's leak check is a backstop and the source's judgement is the guard.  Every
module allowed to call it is absent from the execd import closure by construction
(``tests/test_export_policy_contract.py`` proves that from
``tool_capabilities.remote_native_import_closure``), so a target-side door cannot reach
the weaker question even by asking for it.

The split matters. ``ouroboros/remote_export_policy.py`` is the HOME AUTHORITY:
it is the only place that decides what a document says, because deciding
requires the task contract, the resource policy and the protected-artifact
machinery, none of which may exist on a remote host (there is one brain).  The
source side receives the finished document over the wire and APPLIES it
mechanically, before it constructs any blob.  ``remote_export_policy`` is on the
forbidden-import list for execd kernels for that reason; this module is not.

Why the rules travel as data instead of living as code on both sides. If the
predicate were code in two places, a rule change would have to land on both
sides simultaneously to stay honest, and every channel that forgot to call it
would silently export more than the last one — which is precisely the "one
policy × N doors" class from the postmortem (five separate copies of these rule
tables existed before this module). As a document with a hash, the policy is one
object: Home computes it, the token carries its hash, execd applies the exact
bytes it was handed, and Home revalidates the returned manifest against the same
document. A mismatch is a loud typed refusal rather than a quiet acceptance.

Two PROFILES, because two questions are being asked:

* ``tree`` — bulk enumeration channels (workspace snapshot, patch export, query
  walks): the SOURCE chooses which files to include, so anything sensitive must
  be dropped by rule. Exclusion is a normal, disclosed outcome (D7).
* ``deliverable`` — named-artifact channels (declared outputs, media): the paths
  were declared by the model, so the stricter rules apply (hidden components and
  credential-shaped path components are not deliverables).

Both profiles are evaluated by the SAME functions over the SAME document; the
profile only selects which rule groups participate.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
from collections.abc import Mapping, Sequence
from typing import Any

from ouroboros.remote_contracts import (
    ContractDriftError,
    refuse_unknown_members,
)

CONTRACT_EXPORT_POLICY = "export_policy"

EXPORT_POLICY_VERSION = 1

# Reason codes. These are wire values: they appear in snapshot manifests, in
# import receipts, in the verification ledger and in owner-facing disclosure, so
# they are a closed set and adding one is a deliberate edit here.
REASON_EXCLUDED_DIRECTORY = "excluded_directory"
REASON_SENSITIVE_FILE = "sensitive_file"
REASON_SENSITIVE_COMPONENT = "sensitive_component"
REASON_PROTECTED_ARTIFACT = "protected_artifact"

EXPORT_REASONS = frozenset({
    REASON_EXCLUDED_DIRECTORY,
    REASON_SENSITIVE_FILE,
    REASON_SENSITIVE_COMPONENT,
    REASON_PROTECTED_ARTIFACT,
})

PROFILE_TREE = "tree"
PROFILE_DELIVERABLE = "deliverable"
EXPORT_PROFILES = frozenset({PROFILE_TREE, PROFILE_DELIVERABLE})

# ── the closed set of QUESTIONS a door may ask of one document ────────────────
# A profile selects which rule GROUPS the document carries; a question selects which
# of them this DOOR is entitled to act on. They are separate because the same document
# answers a read and a write differently: only the protected-artifact class may stop a
# write, or an export rule would be answering a question nobody asked.
#
# Closed, and `QUESTION_NONE` is spelled out rather than expressed as "no argument",
# because the two path doors in `workspace_native_paths` REQUIRE a question: a door
# that resolves a workspace path without naming one does not compile, which is how the
# next door is prevented from re-opening the alias hole by simply not asking.
QUESTION_EXPORT = "export"
QUESTION_NAMED_SOURCE = "named_source"
QUESTION_MUTATION = "mutation"
QUESTION_NONE = "none"
EXPORT_QUESTIONS = frozenset({
    QUESTION_EXPORT, QUESTION_NAMED_SOURCE, QUESTION_MUTATION, QUESTION_NONE,
})

# ── every manifest field that can carry an EXPORTED source path ───────────────
# The DECLARATION, in one place, keyed by the producer that writes it. Home's returned-
# manifest leak check derives its field list from this table rather than restating one
# (`remote_export_policy._manifest_entry_paths`), because restating it is exactly how
# that check came to read three fields out of four and pass on an empty list: the
# declared-output/read/list disclosure block carried none of the three names it knew, so
# the backstop for the source side was arithmetic over an empty set — a vacuous guard by
# the Guard Proof Rule, sitting behind the source-side hole it was there to catch.
# ── the shape of ONE disclosed exclusion row, declared once ───────────────────
# Both sides build this row and both sides hash it, so a field added on one side and not
# the other is a fingerprint mismatch that reads as corruption. Adding `judged` did exactly
# that: `workspace_snapshot_native` put three keys in `policy_exclusions` while
# `remote_transfer._verify_materialized` reconstructed two, and a legitimately filtered
# mirror failed verification with "does not match the manifest". The tuple is the authority;
# `judged` falls back to `path` because a row whose alias half is absent is describing a
# direct exclusion.
MANIFEST_EXCLUSION_ROW_FIELDS = ("path", "reason", "judged")

MANIFEST_EXPORTED_PATH_FIELDS: dict[str, str] = {
    "entries": "workspace_snapshot_native._snapshot_once",
    "tracked_changed": "workspace_query_native.execute_git_workspace_operation",
    "untracked_included": "workspace_query_native.execute_git_workspace_operation",
    "exported": "export_policy_contract.export_disclosure_block",
}

# ── the CLOSED channel registry (Appendix C-1) ───────────────────────────────
# Every blob kind that may cross the remote boundary, with the profile its paths
# are judged under. `path_bearing=False` means the blob has no source path at all
# (process output is generated, not read from a file), so the path rules cannot
# apply to it — such a channel still has to be DECLARED here, because a blob kind
# nobody declared has no policy attached and exporting it would be unpoliced.
#
# An unknown kind fails closed everywhere it is checked. Adding a channel is a
# deliberate edit HERE, once, and the grep-proof test
# (tests/test_remote_export_policy.py) asserts every blob-producing site in
# `ouroboros/` names a channel from this table.
EXPORT_CHANNELS: dict[str, dict[str, Any]] = {
    "workspace_snapshot": {"profile": PROFILE_TREE, "path_bearing": True},
    "workspace_patch": {"profile": PROFILE_TREE, "path_bearing": True},
    "workspace_query": {"profile": PROFILE_TREE, "path_bearing": True},
    "declared_output": {"profile": PROFILE_DELIVERABLE, "path_bearing": True},
    "media_frames": {"profile": PROFILE_DELIVERABLE, "path_bearing": True},
    # DELIVERABLE, and the only channel that pushes bytes Home->target: the owner's
    # and the model's named attachments, not a tree. So the deliverable component rule
    # applies — a leading-dot or credential-marker component anywhere in the path is
    # judged, which is what a named-file channel needs and what a code tree could not
    # tolerate. (The comment that used to sit here belonged to the excised `file_bridge`
    # and said the opposite of the line it annotated; acting on it would have dropped
    # that rule from the one channel that uploads.)
    "attachment_stage": {"profile": PROFILE_DELIVERABLE, "path_bearing": True},
    "subagent_patch": {"profile": PROFILE_TREE, "path_bearing": True},
    "process_stream": {"profile": PROFILE_TREE, "path_bearing": False},
    "process_spool_log": {"profile": PROFILE_TREE, "path_bearing": False},
    "operation_envelope": {"profile": PROFILE_TREE, "path_bearing": False},
}


# The trace keys a returned envelope may carry a policy manifest under, and the reason
# every export producer must emit one: Home's judge treats an ABSENT manifest as
# `policy_scope: "full"` — a positive claim that nothing was withheld. So a door that
# says nothing and a door that filtered nothing are indistinguishable on the wire, and
# the media channel was silent for both of its producers. Emitting the block
# unconditionally is what makes "nobody asked" impossible to mistake for "all clear".
#
# ONE authority:
# this list was hand-restated at `remote_export_policy.validate_operation_trace` and
# again at `remote_transfer._trace_policy_disclosures`, and two hand-lists of the same
# thing is how a channel comes to be validated by one of them and not the other. A new
# manifest-bearing channel is added HERE, beside the channel registry it belongs to.
#
# `before`/`after` are `guarded_patch_apply`'s success arm, and they are the proof that
# one authority is not the same as one list. Its FAILURE arms return their manifest
# under `snapshot` and were validated; its success arm — the only arm that MUTATES the
# target — invented two key names and was validated by nothing, so every remote
# `integrate_subagent_patch` into a policy-filtered workspace wrote a Home record
# asserting `policy_scope: "full"`. The keys were declared nowhere and therefore
# guarded nowhere.
MANIFEST_TRACE_KEYS: tuple[str, ...] = (
    "snapshot",
    "patch_export",
    "export_policy",
    "before",
    "after",
)


class ExportChannelUnknownError(ContractDriftError):
    """A blob kind that is not in the closed registry — fail closed.

    A ``ContractDriftError`` because that is what an unknown channel almost always
    IS across a build pair: one half declared a blob kind the other has never heard
    of. It keeps its own wire ``code`` (that spelling appears in receipts and
    ledgers) and inherits the contract/member/action fields, so the refusal now
    says what to do rather than only that something was unrecognized. Still a
    ``ValueError`` two bases up, so every door that guards a channel lookup with
    ``except ValueError`` is unchanged.
    """

    code = "REMOTE_EXPORT_CHANNEL_UNKNOWN"


# ── the default rule tables: the ONE place these strings live ────────────────
# Consolidated from five previous copies (workspace_snapshot_native ×3 inline,
# workspace_query_native._sensitive_patch_path,
# workspace_payload_native._SENSITIVE_OUTPUT_*, tools/shell.py's duplicate of
# the same). Every one of those doors is now this table.
_DEFAULT_EXCLUDED_DIRS = (
    ".git", ".mypy_cache", ".ouroboros", ".pytest_cache", "__pycache__",
)
_DEFAULT_SENSITIVE_NAMES = (
    ".env", ".env.local", ".git-credentials", ".netrc", ".npmrc", ".pgpass",
    ".pypirc", "credentials", "credentials.json", "secrets.json", "token.json",
    "tokens.json",
)
_DEFAULT_SENSITIVE_NAME_PREFIXES = (".env.", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa")
_DEFAULT_SENSITIVE_SUFFIXES = (".key", ".p12", ".pem", ".pfx")
_DEFAULT_SENSITIVE_COMPONENTS = (".aws", ".gnupg", ".ssh")
_DEFAULT_CREDENTIAL_MARKERS = (
    "access_token", "api_key", "apikey", "bearer_token", "credential",
    "password", "refresh_token", "secret",
)
_DEFAULT_DELIVERABLE_COMPONENT_NAMES = (
    "credential", "credentials", "secret", "secrets", "token", "tokens",
)
# The RESTRICTED-READER tightening, as tables — here, with every other rule table,
# and not in the Home authority that decides WHEN to apply them. `remote_export_
# policy.restricted_reader_rules` imports these: a door that spelled `".env"` itself
# would be a private copy of the rule table, which is the drift this module ended.
#
# The suffixes bound the marker rule (see `_marker_scoped_name`); the completions
# finish the marker list against Home's own alternation, whose bare `token` and
# hyphenated `api-key` the default table does not carry.
RESTRICTED_MARKER_SUFFIXES = (
    ".cfg", ".conf", ".env", ".ini", ".json", ".key", ".p12", ".pem", ".pfx",
    ".toml", ".yaml", ".yml",
)
RESTRICTED_MARKER_COMPLETIONS = ("api-key", "token")
# `x.env` / `db.env`: Home's rule is `name.endswith(".env")` and the default suffix
# table has only the key/cert extensions, so this anchored form had no rule at all on
# the target. The `.env.` INFIX form (`staging.env.old`) is a stated residue — the
# document's fields are prefix, suffix, exact-name, component and marker shaped, an
# infix is none of them, and inventing a field for one spelling would be a rule no
# reader could predict. The two anchored forms (`.env*` by prefix, `*.env` by suffix)
# are the ones a real tree carries.
RESTRICTED_SENSITIVE_SUFFIXES = (".env",)

_RULE_FIELDS = (
    "excluded_dirs",
    "sensitive_names",
    "sensitive_name_prefixes",
    "sensitive_suffixes",
    "sensitive_components",
    "credential_markers",
    "deliverable_component_names",
    # EMPTY by default, so the marker rule below is inert unless a caller declares
    # the suffixes it applies to. It exists because the restricted-subagent denial is
    # a PATTERN (`db_password.conf`, `api_key.yaml`) and the tightening seam of
    # `build_policy_document` takes tokens: without a rule field for it, the pattern
    # half of that denial could only travel as code on both sides of the boundary —
    # the very thing this module exists to prevent.
    "marker_scoped_suffixes",
)
_DOCUMENT_FIELDS = frozenset({"version", "channel", "profile", "protected_paths", *_RULE_FIELDS})

# A policy is applied per path; a document that could name an unbounded number of
# protected paths would make prepare itself a denial-of-service surface.
MAX_POLICY_PROTECTED_PATHS = 1000
# The disclosed exclusion LIST is bounded (it rides in a wire trace); the disclosed
# COUNT never is, because D7 promises the owner an exact number.
MAX_DISCLOSED_EXCLUSIONS = 200
# The EXPORTED list is bounded separately and far higher, because its job is different:
# the exclusion list is a sample for a human to read, while the exported list is the
# EVIDENCE Home re-evaluates, and a sample of evidence is not evidence. Sharing the
# 200 bound made Home's leak check silently partial — a policy-excluded path at index 200
# of a 201-path export passed it. The value is the largest result any ONE channel can
# produce (`list_files` caps at 10 000 entries; declared outputs at
# `DECLARED_OUTPUT_FILE_CAP`; the snapshot at `MAX_SNAPSHOT_FILES`, whose own manifest
# `entries` list Home checks instead), so reaching it means a channel exceeded a cap it
# already fails closed on — and Home fails closed on the truncation flag regardless.
MAX_EXPORTED_PATHS = 10_000


def default_export_rules() -> dict[str, list[str]]:
    """The rule tables as a fresh mutable document fragment."""

    return {
        "excluded_dirs": list(_DEFAULT_EXCLUDED_DIRS),
        "sensitive_names": list(_DEFAULT_SENSITIVE_NAMES),
        "sensitive_name_prefixes": list(_DEFAULT_SENSITIVE_NAME_PREFIXES),
        "sensitive_suffixes": list(_DEFAULT_SENSITIVE_SUFFIXES),
        "sensitive_components": list(_DEFAULT_SENSITIVE_COMPONENTS),
        "credential_markers": list(_DEFAULT_CREDENTIAL_MARKERS),
        "deliverable_component_names": list(_DEFAULT_DELIVERABLE_COMPONENT_NAMES),
        "marker_scoped_suffixes": [],
    }


def channel_profile(channel: str) -> str:
    """The profile a channel's paths are judged under; unknown fails closed."""

    row = EXPORT_CHANNELS.get(str(channel or ""))
    if row is None:
        refuse_unknown_members(
            CONTRACT_EXPORT_POLICY,
            unknown=[channel],
            understood=EXPORT_CHANNELS,
            member="export channels",
            error_type=ExportChannelUnknownError,
        )
    return str(row["profile"])


def build_policy_document(
    *,
    channel: str,
    protected_paths: Sequence[str] = (),
    profile: str = "",
    extra_rules: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Assemble a normalized document for one channel.

    ``protected_paths`` are WORKSPACE-RELATIVE target spellings — the projection
    is Home's job (it owns the path-mapping table); by the time a document exists
    the spellings are already the ones the target will see.

    ``extra_rules`` TIGHTENS the default tables additively for one operation — the
    seam a caller with a stricter reader (a restricted subagent) uses so the source
    declines those bytes instead of Home filtering them after they arrived. It can
    only add tokens to the declared rule fields; it cannot remove a default one, so
    no caller can loosen the policy through it.
    """

    selected = str(profile or channel_profile(channel))
    rules = default_export_rules()
    for field, tokens in (extra_rules or {}).items():
        if field not in _RULE_FIELDS:
            raise ValueError(f"export policy cannot be tightened on unknown field {field!r}")
        rules[field] = [*rules[field], *[str(token) for token in tokens]]
    return normalize_export_policy({
        "version": EXPORT_POLICY_VERSION,
        "channel": str(channel),
        "profile": selected,
        "protected_paths": list(protected_paths),
        **rules,
    })


def normalize_export_policy(raw: Any) -> dict[str, Any]:
    """Validate and canonicalize a policy document; fail closed on anything odd.

    An unknown field is an error rather than something to ignore: a document with
    a field this build does not understand is a policy this build cannot claim to
    have applied.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("export policy must be a mapping")
    unknown = sorted(set(map(str, raw.keys())) - _DOCUMENT_FIELDS)
    if unknown:
        # Still fail closed — but say WHOSE contract disagreed and what fixes it.
        # A bare `ValueError: export policy has unknown fields: [...]` was the exact
        # shape the owner met when a Home that had added one rule field talked to a
        # target built before it: correct, and unactionable.
        refuse_unknown_members(
            CONTRACT_EXPORT_POLICY,
            unknown=unknown,
            understood=_DOCUMENT_FIELDS,
            member="document fields",
        )
    version = raw.get("version")
    if version != EXPORT_POLICY_VERSION:
        refuse_unknown_members(
            CONTRACT_EXPORT_POLICY,
            unknown=[version],
            understood=[EXPORT_POLICY_VERSION],
            member="document version",
        )
    channel = str(raw.get("channel") or "")
    expected_profile = channel_profile(channel)
    profile = str(raw.get("profile") or expected_profile)
    if profile not in EXPORT_PROFILES:
        refuse_unknown_members(
            CONTRACT_EXPORT_POLICY,
            unknown=[profile],
            understood=EXPORT_PROFILES,
            member="document profiles",
        )
    document: dict[str, Any] = {
        "version": EXPORT_POLICY_VERSION,
        "channel": channel,
        "profile": profile,
    }
    for field in _RULE_FIELDS:
        document[field] = _normalized_tokens(raw.get(field), field)
    document["protected_paths"] = _normalized_protected_paths(raw.get("protected_paths"))
    return document


def _normalized_tokens(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"export policy field {field!r} must be a list of strings")
    tokens = {str(item).strip().casefold() for item in value if str(item or "").strip()}
    return sorted(tokens)


def _normalized_protected_paths(value: Any) -> list[str]:
    """Workspace-relative, slash-normalized, escape-free, deduplicated, sorted."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("export policy protected_paths must be a list of strings")
    if len(value) > MAX_POLICY_PROTECTED_PATHS:
        raise ValueError(
            f"export policy declares more than {MAX_POLICY_PROTECTED_PATHS} protected paths"
        )
    paths: set[str] = set()
    for item in value:
        parts = [part for part in str(item or "").replace("\\", "/").split("/") if part not in {"", "."}]
        if not parts:
            continue
        if any(part == ".." for part in parts):
            # A policy path that escapes the workspace cannot be applied against a
            # workspace-relative entry, and silently dropping it would understate
            # the policy the hash claims to bind.
            raise ValueError(f"export policy protected path escapes the workspace: {item!r}")
        paths.add("/".join(parts))
    return sorted(paths)


def canonical_policy_bytes(document: Mapping[str, Any]) -> bytes:
    """The exact bytes both sides hash. Normalization happens first, always."""

    return json.dumps(
        normalize_export_policy(document),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def export_policy_hash(document: Mapping[str, Any]) -> str:
    """The hash that rides with the prepared operation and binds ONE policy."""

    return hashlib.sha256(canonical_policy_bytes(document)).hexdigest()


def path_under_any(path: str, prefixes: tuple[str, ...]) -> bool:
    """Is this workspace-relative posix path AT, or inside, any of these prefixes?

    ONE owner for one rule. It existed twice, byte-identical, as
    `workspace_query_native._path_matches_any` and
    `workspace_snapshot_native._path_under_any` — two spellings of the same
    containment arithmetic, each answering about the same export-policy prefixes, in
    two modules that both already import this one. Two copies of a containment rule
    is how a `/` boundary comes to be checked in one place and not the other; both
    call sites are policy EXCLUSION checks, so the rule belongs beside
    `_exclusion_reason`, which they also both reach.

    The `+ "/"` is the whole point and is not `startswith(prefix)`: `src` must not
    match `srcfile.txt`. An empty prefix is skipped rather than matching everything.
    """

    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes if prefix)


def _exclusion_reason(relative_path: Any, document: Mapping[str, Any]) -> str:
    """Why this workspace-relative path may not leave the host, or ``""``.

    THE mechanical evaluator, and the ONLY function that reads a rule group. Every
    export door — snapshot walk, patch export, query walk, declared-output member,
    media source — reaches the rules through here, and PRIVATELY: the two public ways
    in are :func:`judged_exclusion` (a file, judged with its identity) and
    :func:`unaliased_exclusion` (a bare spelling, for Home, which has no filesystem).

    This docstring used to say "asks this and nothing else" while the
    declared-output door asked `component_exclusion_reason` instead — one rule group
    out of a document that carried two — so `dist/id_rsa` exported under the
    credential-PREFIX rule this ladder applies and the component rule does not, and a
    path Home had listed in `protected_paths` exported with it, undisclosed. The
    sentence is true now because the sub-rules are private: `_offending_component` is
    reachable only from here, so a door cannot select a subset of the document.

    It is private for the same reason and one more: while it was public, EIGHT call
    sites reached it directly and skipped the mandatory ``question`` argument the doors
    were given — the door was made compulsory and the way around the door was left
    exported. A rule nobody can bypass has to be a rule nobody can NAME.

    Precedence is fixed so the disclosed reason is stable across channels: an
    explicitly protected path is named as such even when it also looks sensitive.
    """

    parts = [part for part in str(relative_path or "").replace("\\", "/").split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        # An unrepresentable path is not something to guess about.
        return REASON_SENSITIVE_FILE
    joined = "/".join(parts)
    for protected in document.get("protected_paths") or []:
        text = str(protected)
        if joined == text or joined.startswith(text + "/"):
            return REASON_PROTECTED_ARTIFACT
    folded = [part.casefold() for part in parts]
    excluded_dirs = frozenset(document.get("excluded_dirs") or ())
    if any(part in excluded_dirs for part in folded):
        return REASON_EXCLUDED_DIRECTORY
    components = frozenset(document.get("sensitive_components") or ())
    if any(part in components for part in folded):
        return REASON_SENSITIVE_FILE
    name = folded[-1]
    if name in frozenset(document.get("sensitive_names") or ()):
        return REASON_SENSITIVE_FILE
    if any(name.startswith(prefix) for prefix in document.get("sensitive_name_prefixes") or ()):
        return REASON_SENSITIVE_FILE
    if any(name.endswith(suffix) for suffix in document.get("sensitive_suffixes") or ()):
        return REASON_SENSITIVE_FILE
    if _marker_scoped_name(name, document):
        return REASON_SENSITIVE_FILE
    if str(document.get("profile") or "") == PROFILE_DELIVERABLE and _offending_component(
        parts, document
    ):
        return REASON_SENSITIVE_COMPONENT
    return ""


def _marker_scoped_name(name: str, document: Mapping[str, Any]) -> bool:
    """A credential-MARKER filename under both profiles, or ``False``.

    Inert unless the document declares ``marker_scoped_suffixes`` — the tree
    profile's default behaviour is unchanged, because "contains the word secret" is
    too broad a reason to drop a file out of an ordinary code walk.

    The match is DELIMITED (`[._-]` or an end of the name) and suffix-scoped, which
    is not decoration: those two conditions are what make this predicate the same one
    `tools/core._is_subagent_secret_repo_path` applies on Home. A plain substring test
    would exclude `token_bucket.py` on the target and include it on Home — a
    divergence in the strict direction, which is still a divergence and would show up
    as a remote `search_code` that quietly sees less than the local one.
    """

    suffixes = frozenset(document.get("marker_scoped_suffixes") or ())
    if not suffixes:
        return False
    dot = name.rfind(".")
    # `.env` has no suffix by this reckoning, matching `PurePosixPath(".env").suffix`.
    suffix = name[dot:] if dot > 0 else ""
    # An EXTENSIONLESS name is always in scope, and cannot be expressed as a token
    # (the normalizer drops the empty string): Home's own rule admits `""` in its
    # suffix set, so a file called `db_password` must not become the one spelling that
    # is refused on Home and read on the target.
    if suffix and suffix not in suffixes:
        return False
    delimiters = {".", "_", "-"}
    for marker in document.get("credential_markers") or ():
        text = str(marker)
        start = name.find(text)
        while start >= 0:
            before_ok = start == 0 or name[start - 1] in delimiters
            after = start + len(text)
            after_ok = after == len(name) or name[after] in delimiters
            if before_ok and after_ok:
                return True
            start = name.find(text, start + 1)
    return False


def _offending_component(parts: Sequence[str], document: Mapping[str, Any]) -> str:
    """The FIRST path component the deliverable rules reject, or ``""``.

    A declared output is a path the model chose to publish, so "looks like a
    credential" is enough to keep it out — unlike a tree walk, nothing here is
    incidental. The component itself is returned rather than just a verdict because
    the owner-facing sentence has always named the offending component, and a
    message that only says "somewhere in this path" is not actionable.
    """

    markers = tuple(document.get("credential_markers") or ())
    names = frozenset(document.get("deliverable_component_names") or ())
    suffixes = tuple(document.get("sensitive_suffixes") or ())
    for part in parts:
        text = str(part or "")
        if not text:
            continue
        lowered = text.casefold()
        if (
            text.startswith(".")
            or lowered in names
            or lowered.endswith(suffixes)
            or any(marker in lowered for marker in markers)
        ):
            return text
    return ""


def unaliased_exclusion(
    relative_path: Any,
    document: Mapping[str, Any],
    *,
    question: str = QUESTION_EXPORT,
) -> tuple[str, str]:
    """A SPELLING's reason code and owner-facing sentence, from ONE evaluation.

    Named for the half it CANNOT do. It judges the string it is handed and asks no
    filesystem, so it cannot see a symlink onto an excluded file or a second hardlink
    name for one inode. On the target that is a hole, which is why the target-side door
    is :func:`judged_exclusion` and why every module that names THIS function is, by
    construction, one that cannot run on the target: `remote_export_policy` and
    `remote_transfer`'s attachment filter and `tools/shell` are all outside
    `tool_capabilities.remote_native_import_closure`, and
    `tests/test_export_policy_contract.py` recomputes that closure and fails if a
    caller ever appears inside it.

    Home genuinely needs it. A returned manifest is a list of STRINGS that arrived over
    a wire; there is no inode behind `dist/shipped.txt` for Home to stat, so Home's
    leak check can only re-evaluate spellings. That makes it a backstop — it proves the
    source applied SOME filtering to the names it admitted — and it is why the identity
    guard has to stand at the source. `probe_declared.py` is the demonstration: a
    hardlink into a declared output shipped a root `.env`, and Home's check PASSED,
    because the source honestly declared the innocent spelling.

    It also exists because reason and sentence used to be separable: a door could ask a
    rule GROUP for a reason and a different helper for a sentence, and the
    declared-output door did exactly that — `component_exclusion_reason` for the verdict
    and `describe_component_exclusion` for the text, neither of which consults the rest
    of the document. Neither caller can hold a reason and a sentence from different
    rules now.

    ``question`` narrows what this DOOR may act on, out of the closed set:

    * ``QUESTION_EXPORT`` — every rule the document carries (tree walks, media, bridge);
    * ``QUESTION_NAMED_SOURCE`` — the same minus the bulk-only excluded-dirs rule. That
      rule exists so a TREE export does not ship `.git` or `__pycache__`; it is not a
      statement that a path the MODEL named by hand may not be read, and
      `read_file('.git/config')` is legal on a local placement, so applying it to a
      named source would manufacture a cross-placement divergence;
    * ``QUESTION_MUTATION`` — the protected-artifact class only, and it is a question
      about WRITING. The credential-name classes deliberately do not participate: a task
      writing a `.env` into its own workspace is ordinary on a local placement, and
      refusing it here would turn a rule about what may LEAVE the host into a rule about
      what may be written. That narrowing is only sound for a door that WRITES, and the
      reviewer found the proof: `edit_text` asked it on the READ half too, so
      `edit_text('.env', old_str='zzz')` printed 2000 chars of the file that
      `read_file('.env')` refused, and the same for `id_rsa`, `credentials.json` and
      `.env.prod`. A door that both reads and writes asks BOTH questions now — reading
      in order to write is still reading;
    * ``QUESTION_NONE`` — a resolution that exports nothing (a command cwd, a fact being
      assembled at prepare). Spelled, not implied.
    """

    if question not in EXPORT_QUESTIONS:
        refuse_unknown_members(
            CONTRACT_EXPORT_POLICY,
            unknown=[question],
            understood=EXPORT_QUESTIONS,
            member="policy questions",
        )
    if question == QUESTION_NONE:
        return "", ""
    reason = _exclusion_reason(relative_path, document)
    if question == QUESTION_NAMED_SOURCE and reason == REASON_EXCLUDED_DIRECTORY:
        reason = ""
    if question == QUESTION_MUTATION and reason != REASON_PROTECTED_ARTIFACT:
        reason = ""
    if not reason:
        return "", ""
    return reason, describe_exclusion(str(relative_path or ""), reason, document)


def describe_exclusion(
    relative_path: str, reason: str, document: Mapping[str, Any] | None = None
) -> str:
    """One owner-facing sentence per reason code. Disclosure, not a refusal.

    Split from the verdict because Home ALSO renders reasons it did not derive: a
    returned manifest carries the source's own reason codes, and re-deriving them from
    Home's document would describe a filtering that did not happen
    (`remote_export_policy.export_disclosure`). One decider, one renderer.

    ``document`` is only needed for the deliverable-component reason, whose sentence
    names the offending component; without one the sentence stays general rather than
    inventing a component.
    """

    if reason == REASON_PROTECTED_ARTIFACT:
        return f"{relative_path}: protected artifact, excluded from export by policy"
    if reason == REASON_EXCLUDED_DIRECTORY:
        return f"{relative_path}: build/VCS directory, never exported"
    if reason == REASON_SENSITIVE_COMPONENT:
        parts = [
            part
            for part in str(relative_path).replace("\\", "/").split("/")
            if part not in {"", "."}
        ]
        component = _offending_component(parts, document or {})
        if not component:
            return f"{relative_path}: credential-shaped path component is not a deliverable"
        kind = "hidden/control" if component.startswith(".") else "credential-like"
        return (
            f"{relative_path}: {kind} output path component {component} is not a "
            "deliverable artifact"
        )
    return f"{relative_path}: credential-like file, excluded from export by policy"


class ExportPolicyExcludedError(PermissionError):
    """A single-source export channel whose only source the policy omits."""

    code = "REMOTE_EXPORT_POLICY_EXCLUDED"


def identity_spelling(
    workspace_root: Any,
    resolved: Any,
    *,
    base: str = "",
    real_dirs: dict[str, str] | None = None,
) -> str:
    """The workspace-relative spelling of what will ACTUALLY be touched, or ``""``.

    The requested spelling and this one differ exactly when an alias is in play, and
    the difference is the whole finding: the policy judged `safe.txt` and then read
    `.env`. Both are judged now, and this is the second one.

    ``realpath`` on BOTH sides rather than ``Path.relative_to`` on the caller's root:
    a workspace under a symlinked temp dir (`/var` -> `/private/var`) makes the two
    spellings incomparable, and a containment question that raises is a question that
    did not get asked. ``""`` means "no second spelling to judge" — an unresolvable
    path, or one outside the root, which the confinement door has already refused.

    ``base`` and ``real_dirs`` are the caller's memos, and they are here for a MEASURED
    reason rather than for tidiness. ``realpath`` walks a path component by component, so
    this function cost 28.6 µs per file — 15.2 µs resolving the target and 11.8 µs
    re-deriving the root, once per entry, as a constant. Judging identity on every walk
    entry (which the walks did not do before) then added 38 % to a whole-tree
    `search_code` over 24 000 files. ``base`` is the root already canonicalized;
    ``real_dirs`` memoizes ``realpath`` per PARENT directory, and a file whose parent is
    already canonical and whose own final component is not a symlink IS its own
    ``realpath`` — one ``lstat`` instead of one per component. :class:`AliasIndex` carries
    both, because it is the per-operation identity context.
    """

    if resolved is None:
        return ""
    try:
        base = base or os.path.realpath(str(workspace_root))
        text = str(resolved)
        parent, name = os.path.split(text)
        if real_dirs is None or not name:
            target = os.path.realpath(text)
        else:
            real_parent = real_dirs.get(parent)
            if real_parent is None:
                real_parent = os.path.realpath(parent)
                real_dirs[parent] = real_parent
            target = (
                text
                if real_parent == parent and not os.path.islink(text)
                else os.path.realpath(text)
            )
    except (OSError, ValueError):
        return ""
    if target == base:
        return "."
    prefix = base.rstrip(os.sep) + os.sep
    if not target.startswith(prefix):
        return ""
    return target[len(prefix) :].replace(os.sep, "/")


class AliasIndex:
    """Every excluded spelling in one tree, keyed by the inode it names — built LAZILY.

    A hardlink is a second NAME for one inode, and POSIX offers no way to walk from an
    inode back to its names: the only honest answer is to enumerate the excluded names
    and compare. That enumeration is what this holds, and holding it is the point —
    it is built at most ONCE per operation, on the first question that actually needs
    it, and shared by every subsequent one.

    Why lazily, and why this is the shape that makes cost proportionate to the QUESTION:
    a file with ``st_nlink < 2`` has exactly one name, which the door has already judged
    twice (as requested and as resolved), so there is no alias to look for and the walk
    never happens. That short-circuit is not an optimization on a rare path — it is the
    ordinary path. The previous shape got this exactly backwards: the walk channels
    seeded the whole tree UP FRONT, so a `search_code(path="scope")` over one file paid
    a 24 000-file traversal before reading anything, and the single-source doors, told
    that a whole-tree scan would be a denial-of-service door, used a root-only
    ``scandir`` that answered the question wrong. Deferring it dissolves the trade:
    ordinary reads and ordinary walks pay one ``stat`` they were already paying, and only
    a tree that really contains a multiply-named file pays a traversal, once.

    Deferring also retires an ORDERING hazard the seeding shape had to reason about. The
    seed had to run before anything was read, because an alias sorting earlier than the
    file it aliases would otherwise be admitted first. A per-file question has no order
    to get wrong.

    DIRECTORIES are excluded from the alias question, and that is required rather than
    thrifty: every directory has ``st_nlink >= 2`` from `.` and `..`, so without this the
    first directory a walk judged would trigger the traversal on every tree. POSIX has no
    directory hardlinks, so a directory's link count says nothing about aliasing.

    Only paths the policy ALREADY excludes are stat'ed, so a build costs one traversal
    plus a stat per excluded entry — never a stat per file. The explicit protected paths
    are stat'ed directly as well, because one may live under a directory ordinary
    traversal prunes.

    ``root`` and ``real_dirs`` are public because this object is not only the alias index
    — it is the per-operation IDENTITY CONTEXT, and :func:`identity_spelling` reads both:
    the canonical root (re-deriving it per entry was 41 % of the per-file cost) and the
    per-directory ``realpath`` memo that lets an ordinary file cost one ``lstat`` instead
    of one per path component.
    """

    __slots__ = ("root", "real_dirs", "_document", "_names", "_listings")

    def __init__(self, workspace_root: Any, document: Mapping[str, Any]) -> None:
        self.root = os.path.realpath(os.fspath(workspace_root))
        self.real_dirs: dict[str, str] = {}
        self._document = document
        self._names: dict[tuple[int, int], tuple[str, str]] | None = None
        self._listings: dict[str, dict[tuple[int, int], tuple[str, str]]] = {}

    def alias(
        self, resolved: Any, stat_result: Any = None, spelling: str = ""
    ) -> tuple[str, str]:
        """The excluded spelling that IS this file, and its reason, or ``("", "")``.

        ``stat_result`` is the caller's own ``stat`` when it already has one (the walks
        do), so judging identity adds no syscall to a channel that was stat'ing anyway.
        ``spelling`` is the file's workspace-relative identity, which the caller has
        already computed; it only chooses where to look FIRST and never what to answer.
        """

        try:
            target = os.stat(resolved) if stat_result is None else stat_result
        except OSError:
            # Nothing there to be an alias of. A MISSING path is judged on its spellings
            # alone, which is what keeps a refusal from becoming an existence oracle.
            return "", ""
        if target.st_nlink < 2 or stat_module.S_ISDIR(target.st_mode):
            return "", ""
        key = (target.st_dev, target.st_ino)
        # NEAR FIRST, and this is what keeps the cost proportionate to the question
        # rather than to the tree. Two directory listings — the file's OWN directory and
        # the workspace root — answer every shape that occurs: an alias and the file it
        # aliases are almost always siblings, and the adversarial shape (thousands of
        # excluded names in the root) is answered by the root's own listing. Without it a
        # single read of a multiply-named file paid a whole-tree traversal, measured at
        # 43 ms per read on 24 000 files, which is a denial-of-service door inside a
        # guard — the same mistake the root-ONLY probe was invented to avoid, in the
        # opposite direction. This is a positive-only fast path: a hit is authoritative
        # (it compared inodes), a miss falls through to the complete index below, so
        # narrowing it can never admit an alias.
        near = spelling.rsplit("/", 1)[0] if "/" in spelling else ""
        for prefix in dict.fromkeys((near, "")):
            listing = self._listings.get(prefix)
            if listing is None:
                # The listing is MEMOIZED per directory, not re-scanned per entry, and
                # that is a correctness-shaped cost rather than a micro-optimization: a
                # `list_files` over a directory holding 3 000 excluded hardlinks asked the
                # question 3 000 times and each answer re-listed the same directory —
                # 10.6 s of quadratic work, a denial-of-service door inside the guard that
                # was just added to close one. One scan per directory per operation.
                listing = {}
                directory = (
                    os.path.join(self.root, *prefix.split("/")) if prefix else self.root
                )
                try:
                    entries = list(os.scandir(directory))
                except OSError:
                    entries = []
                for entry in entries:
                    name = f"{prefix}/{entry.name}" if prefix else entry.name
                    reason = _exclusion_reason(name, self._document)
                    if not reason:
                        continue
                    try:
                        found = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    listing[(found.st_dev, found.st_ino)] = (name, reason)
                self._listings[prefix] = listing
            hit = listing.get(key)
            if hit:
                return hit
        if self._names is None:
            names: dict[tuple[int, int], tuple[str, str]] = {}
            pending = [
                (os.path.join(self.root, *str(path).split("/")), str(path))
                for path in self._document.get("protected_paths") or []
            ]
            # DESCEND EVERYWHERE. This walk used to prune any directory the policy
            # excludes — `.git`, `node_modules`, `.ssh` — which meant the files inside
            # them were never indexed as excluded NAMES. A hardlink to `.git/config`
            # therefore missed the index and passed the filter: the index answered "not
            # an alias of anything excluded" about a file whose other name is the one
            # thing the policy is surest about. Pruning by the exclusion predicate is
            # exactly backwards — an excluded directory is where the aliases WORTH
            # finding live.
            #
            # `entry.inode()` is what makes covering the whole tree cheaper than the
            # pruned version was: it comes from the readdir result, so this costs one
            # readdir plus one stat PER DIRECTORY and zero stats per file, where the old
            # code stat'ed every excluded entry. Symlinked directories are still never
            # followed (`follow_symlinks=False` decides the descent), and the traversal
            # stays behind the `st_nlink < 2` short-circuit and the near+root fast path
            # above, so it runs at most once per operation and only for a tree that
            # really holds a multiply-named file.
            stack = [(self.root, "")]
            while stack:
                directory, prefix = stack.pop()
                try:
                    entries = sorted(os.scandir(directory), key=lambda item: item.name)
                except OSError:
                    continue
                try:
                    # Re-read per directory rather than hoisted: a mount point inside the
                    # workspace has its own device, and an inode number is only unique
                    # within one.
                    device = os.stat(directory).st_dev
                except OSError:
                    continue
                for entry in entries:
                    spelling = prefix + entry.name
                    try:
                        is_directory = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_directory:
                        stack.append((entry.path, spelling + "/"))
                        continue
                    reason = _exclusion_reason(spelling, self._document)
                    if not reason:
                        continue
                    try:
                        entry_key = (device, entry.inode())
                    except OSError:
                        continue
                    if entry_key not in names or reason == REASON_PROTECTED_ARTIFACT:
                        names[entry_key] = (spelling, reason)
            for path, spelling in pending:
                try:
                    entry = os.stat(path)
                except OSError:
                    continue
                # `entry_key`, NOT `key`: the outer `key` is the TARGET being asked
                # about, bound at the top of this method and read by the `return` below.
                # Rebinding it here made the answer depend on which entry the loop
                # happened to index last — and it masked the pruning leak on the first
                # question per index, which is why the two fixes have to land together.
                entry_key = (entry.st_dev, entry.st_ino)
                reason = _exclusion_reason(spelling, self._document)
                # A protected artifact WINS the naming, because it is the one reason a
                # mutation may be refused for: an inode named both `.env` and by
                # `protected_paths` must report the protected reason or the write door
                # narrows the verdict away and lets the write through.
                if entry_key not in names or reason == REASON_PROTECTED_ARTIFACT:
                    names[entry_key] = (spelling, reason)
            self._names = names
        return self._names.get(key, ("", ""))


def judged_exclusion(
    workspace_root: Any,
    resolved: Any,
    relative: Any,
    document: Mapping[str, Any],
    *,
    question: str,
    aliases: AliasIndex | None = None,
    stat_result: Any = None,
) -> tuple[str, str, str]:
    """THE judgement: ``(reason, sentence, judged spelling)`` for ONE file, ONE question.

    The single authority on judged identity, and every door and every producer of bytes
    on the target reaches the policy through here — read, list, write, append, edit,
    declared-output member, media source, file bridge, snapshot walk, query walk, git
    pathspec, patch export, rollback admission. There is deliberately no second way in:
    `tests/test_export_policy_contract.py` recomputes the execd import closure and fails
    if any module inside it names a judging function other than this one (or
    :func:`refuse_excluded_target`, which is this function plus a raise).

    That gate is the finding, not a tidiness preference. Identity used to be answered by
    two mechanics — a recursive inode seed for the walks, a root-only ``scandir`` probe
    for the single doors — and the weaker one was a hole in five places at once: a
    hardlink to `sub/.env` was excluded from `search_code` and returned verbatim by
    `read_file`, a hardlink to `sub/golden.bin` mutated a protected artifact through an
    innocent name, and a hardlink into a declared output shipped a root `.env` past a
    Home backstop that could only see the innocent string.

    Three spellings are judged, in this order, and the first refusal wins:

    1. the REQUESTED spelling, which is what a document can state rules about;
    2. the resolved IDENTITY (:func:`identity_spelling`), which is what a symlink hands
       over. A symlink raises no link count, so the alias probe cannot see it, and
       `safe.txt -> .env` returned the secret bytes `.env` itself was refused;
    3. the HARDLINK alias (:class:`AliasIndex`), which neither spelling can see — a
       second name for one inode resolves to itself and reads as an ordinary file. The
       search is RECURSIVE, at every depth, in both directions (a nested alias to a root
       file and a root alias to a nested file), and its cost is bounded by the
       ``st_nlink`` short-circuit rather than by a boundary drawn around the root.

    ``stat_result`` lets a walk that already stat'ed the entry lend it, so identity costs
    a walk nothing. ``aliases`` shares one index across an operation; without one, a
    throwaway index is built, which is correct and costs a traversal only if the named
    file actually has a second name.

    The third element is the spelling the POLICY actually excluded — the requested one, the
    resolved identity, or the alias's own name. It is the honest answer rather than the
    convenient one, because two callers need it and they need it for opposite reasons:
    `_raise_policy_refusal` decides from it whether the refusal may NAME the other spelling
    (it may for a protected artifact, which the document already lists, and may not for a
    credential file, whose location only the filesystem knows), and every disclosure ROW
    carries it as `judged` so Home can re-evaluate the claim. Home holds no workspace, so a
    row that said only "notes.txt: sensitive_file" is a claim Home cannot check — and a
    claim Home cannot check is one a target can invent.
    """

    requested = str(relative or ".")
    # The REQUESTED spelling first, and the identity only if it survives: an excluded
    # name is the early exit, and resolving a path costs a `lstat` per component while
    # reading the ladder costs 1.5 µs.
    #
    # The workspace ROOT itself is skipped in both slots. `_exclusion_reason` answers
    # `sensitive_file` for an empty component list on purpose ("an unrepresentable path is
    # not something to guess about"), and the root is the one representable path that
    # normalizes to none — listing it is the ordinary case, not an exclusion.
    if requested and requested != ".":
        reason, sentence = unaliased_exclusion(requested, document, question=question)
        if reason:
            return reason, sentence, requested
    identity = identity_spelling(
        workspace_root,
        resolved,
        base=aliases.root if aliases is not None else "",
        real_dirs=aliases.real_dirs if aliases is not None else None,
    )
    if identity and identity != "." and identity != requested:
        reason, sentence = unaliased_exclusion(identity, document, question=question)
        if reason:
            return reason, sentence, identity
    if question == QUESTION_NONE or resolved is None or requested == ".":
        return "", "", ""
    index = aliases if aliases is not None else AliasIndex(workspace_root, document)
    name, _found = index.alias(
        resolved, stat_result=stat_result, spelling=identity or requested
    )
    if not name:
        return "", "", ""
    reason = unaliased_exclusion(name, document, question=question)[0]
    if not reason:
        return "", "", ""
    return (
        reason,
        f"{requested}: the same file as {name}, which the export policy excludes",
        name,
    )


def refuse_excluded_target(
    workspace_root: Any,
    resolved: Any,
    relative: Any,
    native_facts: Mapping[str, Any] | None,
    *,
    question: str,
    channel: str = "",
    aliases: AliasIndex | None = None,
    stat_result: Any = None,
) -> None:
    """:func:`judged_exclusion` plus the raise, for the doors that refuse rather than
    disclose.

    ``stat_result`` is how :func:`workspace_native_paths.open_confined_source` binds the
    judgement to the DESCRIPTOR it opened rather than to a fresh stat of the name: an
    `os.fstat` of an open fd cannot be swapped between the check and the read.

    The two path doors in `workspace_native_paths` call it for their callers, so a door
    that resolves a workspace path cannot fail to ask.

    A channel, when given, is checked against the closed registry FIRST: a blob kind
    nobody declared has no policy attached, so it fails closed before any path is
    judged. Where a tree channel discloses a partial result, a channel with exactly
    one model-named source refuses — there is nothing left to deliver once the source
    is out, and a silent empty success would read as "the file had no frames".

    An UNBOUND read/export operation falls back to the channel's DEFAULT document,
    exactly as the tree channels do: "no policy handed down" is not "no rules". A
    mutation does not, because the class it enforces exists only in a document Home
    issued — there is no default protected-path list to fall back to.
    """

    if channel:
        channel_profile(channel)
    document = policy_from_facts(native_facts)
    if document is None:
        if question in {QUESTION_MUTATION, QUESTION_NONE}:
            return
        document = build_policy_document(channel=channel or "workspace_query")
    requested = str(relative or ".")
    reason, sentence, judged = judged_exclusion(
        workspace_root,
        resolved,
        requested,
        document,
        question=question,
        aliases=aliases,
        stat_result=stat_result,
    )
    if reason:
        _raise_policy_refusal(question, channel, requested, judged, sentence)


def _raise_policy_refusal(
    question: str,
    channel: str,
    requested: str,
    judged: str,
    sentence: str,
) -> None:
    """One refusal per question. Whether it NAMES the other spelling is a policy question.

    The mutation question raises its own typed code because a refused write reports
    `not_started` through a different diagnostic than a refused export.

    A MUTATION refusal names the resolved target, an EXPORT refusal does not, and the line
    between them is which fact the caller already holds. `protected_paths` are literally in
    the document this operation carries — a task can read its own resource policy — so
    `innocent.bin (which resolves to src/golden.bin)` discloses nothing new and is what
    makes the refusal actionable.

    The credential classes are NOT in the document: the rules are NAME SHAPES, so the
    location of a `.env` or of `.ssh/real_key` is a fact only the filesystem holds. Naming
    it turned the refusal into an oracle a paid reviewer demonstrated: `list_files` refuses
    to list `.ssh` at all, while `read_file` on a symlink into it answered "probe_hit.txt
    (which resolves to .ssh/real_key)" — the model learned a filename inside a directory
    the policy had just declined to show it, purely from an error message. The export
    refusal now says that the path IS another spelling of an excluded file, which is the
    actionable half, and does not say which one. The resolved spelling still reaches the
    owner: it rides in the operation's disclosure block, not in the model-facing text.
    """

    if question == QUESTION_MUTATION:
        trail = "" if judged == requested else f" (which resolves to {judged})"
        raise PermissionError(
            f"REMOTE_PROTECTED_ARTIFACT_BLOCKED: {requested}{trail} is a protected "
            "artifact under this task's resource policy; mutation is not allowed."
        )
    detail = (
        sentence.split(": ", 1)[-1]
        if judged == requested
        else "another name for a path the export policy excludes"
    )
    raise ExportPolicyExcludedError(
        f"{ExportPolicyExcludedError.code}: {channel or 'workspace_query'}: "
        f"{requested}: {detail}"
    )


def policy_filtered_note(marker: str, rows: Sequence[Mapping[str, str]]) -> str:
    """The owner-facing sentence: exact count, bounded list, never a bare absence.

    Same shape as the query walk's note (`workspace_query_native._PolicyExclusions.
    note`) because it is the same promise; kept here rather than imported because
    that one is bound to a walk's collector and this door has a plain list.
    """

    shown = list(rows)[:MAX_DISCLOSED_EXCLUSIONS]
    text = (
        f"⚠️ {marker}: {len(rows)} entr{'ies' if len(rows) != 1 else 'y'} excluded by "
        "the export policy and NOT listed — this listing is not a complete answer "
        "about the directory:\n"
        + "\n".join(f"- {row['path']}: {row['reason']}" for row in shown)
    )
    if len(rows) > len(shown):
        text += (
            f"\n… and {len(rows) - len(shown)} more "
            f"(list bounded at {MAX_DISCLOSED_EXCLUSIONS}; the count is exact)"
        )
    return text


def export_disclosure_block(
    native_facts: Mapping[str, Any] | None,
    excluded: Sequence[Mapping[str, str]],
    exported: Sequence[str] = (),
    *,
    question: str = QUESTION_EXPORT,
) -> dict[str, Any]:
    """The additive D7 block: exact count, bounded list, the policy that ran.

    Emitted on EVERY operation, exclusions or not, so the presence of the keys
    never encodes whether something was filtered — an absent block would let a
    quiet omission look like an unfiltered export. The disclosed EXCLUSION list is
    bounded because it travels in a wire trace and a remote envelope must not size a
    Home record; the COUNT is always exact, which is what D7 requires.

    ``exported`` is the other half, and it is why Home's returned-manifest check is
    no longer vacuous. This block carried the OMISSION and never the ADMISSION, so
    `_manifest_entry_paths` — which understands `entries`, `tracked_changed` and
    `untracked_included` — found none of its field names here and re-evaluated the
    policy over an empty list: it passed on hash and arithmetic alone for every read,
    every listing and every declared output. The field is declared in
    `MANIFEST_EXPORTED_PATH_FIELDS`, which is where Home derives its list from, so a
    fourth channel cannot appear without appearing there too.

    ``exported`` is bounded SEPARATELY, at :data:`MAX_EXPORTED_PATHS`, and that is not
    tidiness — sharing the exclusion bound made Home's check silently partial. A reviewer
    put a policy-excluded path at index 200 of a 201-path export and
    `validate_returned_manifest` PASSED: it reads only the truncated list, and nothing read
    `exported_disclosure_truncated`, which the block had been computing all along. The two
    lists have different jobs. The excluded list is a SAMPLE for a human to read, so a
    bound of 200 is right and the exact count carries the rest. The exported list is
    EVIDENCE for a mechanical re-check, so a sample is worthless: it is bounded by the
    largest result any one channel can produce, and Home fails closed if the bound is ever
    reached (see `remote_export_policy.validate_returned_manifest`).

    ``question`` travels because the source and Home must apply the SAME projection of one
    document. `QUESTION_NAMED_SOURCE` deliberately drops the bulk-only excluded-dirs rule,
    so `read_file('.git/config')` is legal at the source — and Home, re-evaluating under
    the default question, raised `ExportPolicyViolation` after the bytes had already
    crossed. Two projections of one policy is the same class as two mechanics for one
    question, one layer out.
    """

    document = policy_from_facts(native_facts)
    if question not in EXPORT_QUESTIONS:
        refuse_unknown_members(
            CONTRACT_EXPORT_POLICY,
            unknown=[question],
            understood=EXPORT_QUESTIONS,
            member="policy questions",
        )
    rows = [dict(row) for row in excluded]
    shipped = [str(path) for path in exported if str(path or "")]
    return {
        "export_policy": {
            "policy_hash": export_policy_hash(document) if document is not None else "",
            "policy_question": question,
            "complete": not rows,
            "integrity_complete": True,
            "policy_scope": "policy_filtered" if rows else "full",
            "excluded_count": len(rows),
            "excluded": rows[:MAX_DISCLOSED_EXCLUSIONS],
            "excluded_disclosure_truncated": len(rows) > MAX_DISCLOSED_EXCLUSIONS,
            "exported": shipped[:MAX_EXPORTED_PATHS],
            "exported_count": len(shipped),
            "exported_disclosure_truncated": len(shipped) > MAX_EXPORTED_PATHS,
        }
    }


def policy_from_facts(native_facts: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The policy a prepared operation was bound to, or ``None`` if unbound.

    Kernels read the document from the operation's own facts rather than
    reaching for a module-level default: an operation that was prepared without
    a policy must be visibly unbound, not quietly given one.
    """

    raw = (native_facts or {}).get("export_policy")
    if raw is None:
        return None
    return normalize_export_policy(raw)


__all__ = [
    "path_under_any",
    "EXPORT_CHANNELS",
    "MANIFEST_TRACE_KEYS",
    "EXPORT_POLICY_VERSION",
    "EXPORT_PROFILES",
    "EXPORT_QUESTIONS",
    "EXPORT_REASONS",
    "MANIFEST_EXCLUSION_ROW_FIELDS",
    "MANIFEST_EXPORTED_PATH_FIELDS",
    "MAX_DISCLOSED_EXCLUSIONS",
    "MAX_EXPORTED_PATHS",
    "MAX_POLICY_PROTECTED_PATHS",
    "PROFILE_DELIVERABLE",
    "PROFILE_TREE",
    "QUESTION_EXPORT",
    "QUESTION_MUTATION",
    "QUESTION_NAMED_SOURCE",
    "QUESTION_NONE",
    "REASON_EXCLUDED_DIRECTORY",
    "REASON_PROTECTED_ARTIFACT",
    "REASON_SENSITIVE_COMPONENT",
    "REASON_SENSITIVE_FILE",
    "RESTRICTED_MARKER_COMPLETIONS",
    "RESTRICTED_MARKER_SUFFIXES",
    "RESTRICTED_SENSITIVE_SUFFIXES",
    "AliasIndex",
    "ExportChannelUnknownError",
    "ExportPolicyExcludedError",
    "build_policy_document",
    "canonical_policy_bytes",
    "channel_profile",
    "default_export_rules",
    "describe_exclusion",
    "export_disclosure_block",
    "identity_spelling",
    "judged_exclusion",
    "policy_filtered_note",
    "refuse_excluded_target",
    "unaliased_exclusion",
    "export_policy_hash",
    "normalize_export_policy",
    "policy_from_facts",
]
