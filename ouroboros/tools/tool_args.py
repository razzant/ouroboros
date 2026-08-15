"""The PUBLIC tool-call argument contract: aliases, accepted parameters, refusal.

One tool call arrives as a bag of names the model chose, and exactly one authority
decides what that bag means: which names are accepted (the public schema's
properties, falling back to the handler signature), which near-misses are renamed
rather than rejected, and what the refusal SAYS when neither applies.

It lives beside the dispatcher rather than inside it because the answer must be
the same on both routes. A remote dispatch never calls the Home handler, so if the
validation stayed inside the handler invocation an ssh task would ship the target
an argument set the public schema rejects — and the model would get a different
answer to the same malformed call depending on where its workspace happens to
live. "The same faculty surface, schemas byte-identical at equal inputs" has to
include the refusals.

``entry`` is duck-typed (`.name`, `.schema`, `.handler`) so this module does not
import the registry it serves.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

# Near-miss argument names that are RENAMED rather than refused. Renaming happens
# before prepare (the target is told which paths and cwd the operation is about),
# so an alias still spelled the model's way at that point would have the target
# resolve a path the operation never named.
_TOOL_ARG_ALIASES: dict[str, dict[str, str]] = {
    "*": {"max_entries": "max_results"},
}
# Tools whose public schema has no `root`: the model reflexively supplies one, and
# dropping it is kinder than refusing a call that is otherwise exactly right.
_IGNORE_ROOT_ARG_TOOLS = frozenset({
    "vcs_status",
    "vcs_diff",
    "vcs_pull_ff",
    "vcs_restore",
    "vcs_revert",
    "commit_reviewed",
    "vcs_commit_reviewed",
})


def _handler_public_params(handler: Callable[..., Any]) -> list[str]:
    try:
        params = list(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return []
    return [name for name in params if name not in {"ctx", "_resolved_binding"}]


def _entry_public_params(entry: Any) -> list[str]:
    try:
        params = entry.schema.get("parameters") or {}
        props = params.get("properties")
        if isinstance(props, dict):
            return [str(name) for name in props]
    except Exception:
        pass
    return _handler_public_params(entry.handler)


def _entry_has_public_param_schema(entry: Any) -> bool:
    try:
        params = entry.schema.get("parameters") or {}
        return isinstance(params.get("properties"), dict)
    except Exception:
        return False


def _normalize_tool_call_args(entry: Any, args: dict[str, Any]) -> None:
    tool_name = entry.name
    accepted = set(_entry_public_params(entry))
    aliases: dict[str, str] = {}
    aliases.update(_TOOL_ARG_ALIASES.get("*", {}))
    aliases.update(_TOOL_ARG_ALIASES.get(tool_name, {}))
    for alias, canonical in aliases.items():
        if alias in args and canonical in accepted and alias not in accepted and canonical not in args:
            args[canonical] = args.pop(alias)
    if tool_name in _IGNORE_ROOT_ARG_TOOLS and "root" in args and "root" not in accepted:
        args.pop("root", None)




def _format_tool_arg_error(entry: Any) -> str:
    params = _entry_public_params(entry)
    accepted = ", ".join(params) if params else "none"
    return (
        f"⚠️ TOOL_ARG_ERROR ({entry.name}): invalid arguments for {entry.name}. "
        f"Accepted parameters: {accepted}."
    )


__all__ = [
    "_entry_has_public_param_schema",
    "_entry_public_params",
    "_format_tool_arg_error",
    "_normalize_tool_call_args",
]
