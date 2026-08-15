"""Placement-blind path policies of the AUTHORIZE phase (RWS v2 §3.1 step 3).

Why this module exists at all. A guard that lives INSIDE a Home handler is a
guard that only local placements have. The native route does not call the Home
handler — it replaces it (`registry._dispatch`, the `native_routed` branch) —
so every policy the handler body owned silently stopped applying the moment the
task was pointed at a remote workspace. That is the postmortem's "one policy × N
doors" class wearing a different hat: here the second door is not another channel
but another PLACEMENT.

The subagent secret/control denial was exactly that. `is_restricted_subagent_
profile` is the fail-closed SSOT for "this child may not read owner secrets", and
`tools/core._repo_read`/`_repo_list` enforced it — in their bodies. On an ssh
placement a restricted subagent read `.env`, `credentials.json` and
`secrets/db.txt` out of the remote workspace and listed `secrets/`, because the
handler those checks lived in was never called. Nothing detected it: both sides
were internally consistent, and no test in the repo put a subagent profile and an
ssh placement in the same room.

So the DECISION is lifted here, into the pipeline, ahead of the placement read —
which is what makes it placement-blind by construction rather than by care. The
predicate is the same one (`_is_subagent_secret_repo_path`, a pure function of a
workspace-relative spelling), the refusal texts are byte-identical to the ones
the Home handlers already returned, and the handlers keep their own checks as
DEFENCE IN DEPTH for their non-dispatch callers with one addition each can make
and this cannot: an FS-alias probe (`samefile` against a hardlink to `.env`),
which needs the filesystem the file is actually on. That split is deliberate and
declared: the pure spelling rule is the pipeline's, the alias probe is Home-only
because only Home can perform it, and the message a reader sees is the same
string from either.

`ROOT SCOPE`: the lift covers `active_workspace`, the ONE target-native root
(root matrix, ratified Q2а). `system_repo` and the data roots are Home-native
under every placement — their handler guards can never be bypassed by a native
route, so they stay where they are rather than being duplicated into a second
authority for no gain.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping

from ouroboros.tool_capabilities import IMPLICIT_RESOURCE_ROOT, dispatch_resource_root

# The interpreters `run_script` may name. Public because `tools/shell._run_script`
# imports it: the allowlist is one list, in one place, or it is two lists that will
# disagree. The set is the handler's own, unchanged, and the refusal text below is
# byte-identical to the one the handler returned.
SCRIPT_INTERPRETERS = frozenset({
    "python", "python3", "python.exe", "python3.exe", "bash", "sh", "node", "ruby",
})

# The tools whose ONE path argument the denial judges, with the refusal text each
# already returned on a local placement. The texts are quoted from the handlers
# they were lifted out of and must not drift: a reader who learned the local
# wording must not have to learn a second one for the remote case, and the golden
# traces pin them.
_PATH_DENIAL_MESSAGES: dict[str, str] = {
    "read_file": "⚠️ REPO_READ_BLOCKED: this subagent cannot read repo secret or control files.",
    "list_files": "⚠️ REPO_LIST_BLOCKED: this subagent cannot list repo secret or control paths.",
    "search_code": "⚠️ SEARCH_BLOCKED: this subagent cannot access repo secret or control paths.",
    # `query_code` had NO whole-call refusal on either placement: a restricted
    # subagent asking about `.env` got "No results", which is the false-premise
    # answer D7 forbids elsewhere in this codebase — the model concludes the file
    # holds no symbols rather than that it was not allowed to look. It is refused
    # here, in the same family as its three siblings, on both placements.
    "query_code": "⚠️ QUERY_BLOCKED: this subagent cannot access repo secret or control paths.",
}

# Which argument carries the path the denial is about. `search_code`/`query_code`
# scope by `path` too, so one key serves all four.
_PATH_ARG = "path"


def subagent_secret_path_refusal(ctx: Any, tool: str, args: Mapping[str, Any]) -> str:
    """The restricted-subagent secret/control refusal for this call, or ``""``.

    Pure over (profile, tool, root, workspace-relative spelling): no filesystem,
    no placement, no target RPC. It therefore runs BEFORE prepare — a refused call
    never reaches the target at all, and there is no ordering in which a placement
    could change the answer.
    """

    name = str(tool or "").strip()
    message = _PATH_DENIAL_MESSAGES.get(name, "")
    if not message:
        return ""
    if dispatch_resource_root(name, args) != IMPLICIT_RESOURCE_ROOT:
        # Home-native roots keep their handler guards (see the module docstring).
        return ""
    from ouroboros.tools.core import _is_subagent_secret_repo_path, is_restricted_subagent_profile

    if not is_restricted_subagent_profile(ctx):
        return ""
    raw = args.get(_PATH_ARG)
    if raw is None and name == "list_files":
        raw = args.get("dir")
    return message if _is_subagent_secret_repo_path(str(raw or "")) else ""


def script_interpreter_refusal(tool: str, args: Mapping[str, Any]) -> str:
    """The `run_script` interpreter allowlist, or ``""`` — on BOTH routes.

    `run_script` carries no root label, so it routes natively on every ssh dispatch
    and its Home handler is replaced. The handler was where the allowlist lived, and
    the target's `workspace_payload_native.execute_inline_script` takes the
    interpreter VERBATIM into its argv — so on a remote task the model could name an
    arbitrary executable and the check simply did not run.

    Judged on the RAW argument the model supplied, before prepare. That is what makes
    it placement-blind, and it also makes the resolver-attestation clause the handler
    needs unnecessary here: the resolver only ever attests a request that WAS
    `python`/`python3` (rewriting it to an absolute venv path), and both of those are
    in the allowlist. The handler keeps its own check — over the possibly-rewritten
    value, with the attestation — as defence in depth for its non-dispatch callers.
    """

    if str(tool or "").strip() != "run_script":
        return ""
    interpreter = str(args.get("interpreter") or "python3").strip()
    if pathlib.PurePath(interpreter).name in SCRIPT_INTERPRETERS:
        return ""
    return f"⚠️ RUN_SCRIPT_BLOCKED: interpreter must be one of {sorted(SCRIPT_INTERPRETERS)}."


def filter_native_listing(ctx: Any, tool: str, text: str) -> str:
    """Hide secret/control entries from a restricted subagent's remote listing.

    The whole-call refusal above covers a listing whose TARGET is secret; a listing
    of an ordinary directory that CONTAINS secrets is the other half, and locally
    it is a filtered array plus an exact hidden count (`_filter_subagent_secret_
    repo_listing`). This applies the identical filter to the array a native
    `list_files` returned, so the two placements produce the same bytes.

    Home-side, and deliberately so: this policy is about what the MODEL may see,
    not about what may leave the host — the export boundary (§3.2) owns the latter
    and filters the same classes at the source, so the bytes a subagent must not
    see are already declined there. Doing it here as well means the model-facing
    shape does not depend on which of the two policies fired first.
    """

    if str(tool or "").strip() != "list_files":
        return text
    from ouroboros.tools.core import (
        _filter_subagent_secret_repo_listing,
        is_restricted_subagent_profile,
    )

    if not is_restricted_subagent_profile(ctx):
        return text
    # A native listing is the JSON array optionally followed by the export policy's
    # own disclosure paragraph (D7). The array is what this filters; the paragraph is
    # the SOURCE's statement about what IT declined and is carried through unchanged,
    # because rewriting another authority's disclosure would make the count a
    # negotiation between two filters instead of two facts.
    head, separator, tail = text.partition("\n\n")
    try:
        items = json.loads(head)
    except (TypeError, ValueError):
        return text
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        return text
    filtered = _filter_subagent_secret_repo_listing(items, None)
    return json.dumps(filtered, ensure_ascii=False, indent=2) + separator + tail


__all__ = [
    "SCRIPT_INTERPRETERS",
    "filter_native_listing",
    "script_interpreter_refusal",
    "subagent_secret_path_refusal",
]
