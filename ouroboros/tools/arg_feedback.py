"""What a tool says about an argument it did not obey (DEVELOPMENT "LLM-first affordances").

Models may fill optional keys (OpenAI's Responses API tries strict mode when a
function tool leaves `strict` unset, so the OpenRouter lane sends `strict: false`). A value
that asks for nothing takes the omitted path and the result says so in one line; a
value that asks for something the call cannot serve is refused ONCE, typed, naming
the field, the value received and the repair — a refusal that only restates the
rule is retried unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ouroboros.tools.tool_result import (
    ToolResult, _publish_tool_result, _published_tool_result, _replace_tool_result,
)


def ignored_argument_note(name: str, value: Any, why: str) -> str:
    """One result line for an argument that took the omitted path.

    A blank text says no more than its length, so it is echoed by its length where that is shorter
    (``<9000 whitespace characters>``); a short one keeps its exact spelling (``' '``).
    """
    shown = repr(value)
    if isinstance(value, str) and not value.strip():
        shown = min(shown, f"<{len(value)} whitespace characters>", key=len)
    return f"{name}={shown} ignored: {why}"


def with_argument_notes(ctx: Any, result: str, notes: list[str]) -> str:
    """Keep an inner producer's typed outcome when disclosing omitted arguments."""
    if not notes:
        return result
    text = result + "\n" + "\n".join(notes)
    prior = _published_tool_result(ctx, None)
    if isinstance(prior, ToolResult) and prior.text == result:
        return _publish_tool_result(ctx, _replace_tool_result(prior, text=text))
    return text


def payload_item_feedback(
    ctx: Any, items: Any, properties: dict, *, item_label: str, options: dict,
) -> tuple[str, list[str]]:
    """Validate file/edit payloads before writes; harmless extras take omission.

    The item schema owns the consumed keys. Other keys cannot override the call:
    empty values and repeats of call-wide options are disclosed, requested changes
    are refused together so no sibling is written under an unintended contract.
    """
    problems, notes = [], []
    if not isinstance(items, list):
        problems.append(f"{item_label} payload={items!r}: use an array of objects")
    else:
        for idx, item in enumerate(items, 1):
            name = f"{item_label} {idx}"
            if not isinstance(item, dict):
                problems.append(f"{name}={item!r}: not an object; use an object with {', '.join(properties)}")
                continue
            for key, value in item.items():
                if key in properties:
                    continue
                repeated = key in options and type(value) is type(options[key]) and value == options[key]
                empty = value is None or value == "" or value == [] or value == {}
                if repeated or empty:
                    notes.append(ignored_argument_note(
                        f"{name}.{key}", value,
                        "same as the call-wide option" if repeated else "empty optional value",
                    ))
                else:
                    repair = (f"set {key}={value!r} on a separate call" if key in options
                              else f"remove {key}; item fields are {', '.join(properties)}")
                    problems.append(f"{name}.{key}={value!r}: cannot override this call; {repair}")
    refusal = argument_refusal(ctx, "TOOL_ARG_ERROR", problems, effect="Nothing was written.") if problems else ""
    return refusal, notes


def argument_refusal(
    ctx: Any, identifier: str, problems: Iterable[str], *, effect: str = "",
) -> str:
    """Publish one typed refusal naming every violated constraint (the W2 shape).

    ``identifier`` stays the first-line marker so each domain keeps its own code
    in the text; the typed ``TOOL_ARG_ERROR`` is what status and outcome readers
    read. ``effect`` states what the refused call did NOT do.
    """
    text = f"⚠️ {identifier}: " + "; ".join(str(item).rstrip(". ") for item in problems) + "."
    if effect:
        text += f" {effect}"
    return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=text))


def repair_call_available(ctx: Any, tool: str) -> bool:
    """Whether a refusal may name ``tool`` as the call that works for this caller: not withheld by the task
    contract's ``disabled_tools`` (dispatch refuses it even where a consciousness schema still shows it), in a
    subagent profile's set, and granted by a Presence ceiling with no host-bound arguments (a bound call is not
    the call the text would name). For a caller outside those profiles it answers True for any tool that is not
    withheld, which is true only of core tools (``tool_capabilities.CORE_TOOL_NAMES``): name only those."""
    from ouroboros.contracts.task_contract import canonical_tool_name
    from ouroboros.presence_authority import presence_ceiling_from_context
    from ouroboros.tool_access_roots import active_tool_profile
    from ouroboros.tool_capabilities import ACTING_SUBAGENT_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    from ouroboros.tools.registry_guards import _disabled_tools

    try:
        profile = {"local_readonly_subagent": LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
                   "acting_subagent": ACTING_SUBAGENT_TOOL_NAMES}.get(active_tool_profile(ctx))
        ceiling = presence_ceiling_from_context(ctx)
        grants = [] if ceiling is None else [grant for grant in ceiling.tool_grants if canonical_tool_name(grant.name) == tool]
        return (tool not in _disabled_tools(ctx) and (profile is None or tool in profile)
                and (ceiling is None or len(grants) == 1 and not grants[0].bindings))
    except ValueError:  # an unreadable carrier proves nothing
        return False



ABSENT: Any = object()  # a handler default that tells an omitted key from an explicit null


def read_end_spelling(ctx: Any, call: dict, max_lines: Any, end_line: Any, end_char: Any) -> tuple[Any, list, str]:
    """``(max_lines, notes, refusal)`` for read_file's ``end_line``/``end_char``, outside its schema.

    ``end_line`` is the inclusive last line, counted from the start the renderer uses (``start_line``
    floored at 1). ``end_char`` could count from the window or from the file, so it is refused with the
    call that reads the window's characters. A null takes the omitted path; ``call`` keeps the read's
    other arguments, and each named call keeps them and the call's other limit, reading the same file from
    the same cursor no further than asked. The default ``max_lines`` (2000) asks for nothing within it.
    """
    try:
        start = max(1, int(call.get("start_line", 1)))
    except (TypeError, ValueError):
        start = 1

    def ready(limit: str) -> str:
        kept = "".join(f", {key}={json.dumps(call[key])}" for key in (
            "root", "bucket", "skill_name", "start_char", limit) if call.get(key) not in (None, "", 0))
        return f"read_file(path={json.dumps(call['path'])}{kept}, start_line={start}, "

    text = str(end_line).strip() if isinstance(end_line, (int, str)) and not isinstance(end_line, bool) else ""
    try:  # str.isdigit() also admits digits int() refuses ("²")
        end = int(text) if text[text.startswith("-"):].isdigit() else None
    except ValueError:
        end = None
    want = None if end is None else end - start + 1
    notes = [ignored_argument_note(name, None, "empty optional value")
             for name, value in (("end_line", end_line), ("end_char", end_char)) if value is None]
    if end_char not in (None, ABSENT):  # the named call keeps the line window this call would have read
        try:
            lines = want if want and want >= 1 else None if max_lines is None else max(1, int(max_lines))
        except (TypeError, ValueError):
            lines = None
        call = {**call, "max_lines": lines}
        return max_lines, notes, argument_refusal(ctx, "TOOL_ARG_ERROR", [
            f"end_char={end_char!r} could count from the window or from the file; max_chars counts the window's "
            f"characters after start_char: {ready('max_lines')}max_chars=<characters>), "
            "with start_line=1 for the file's"],
            effect="Nothing was read.")
    if end_line in (None, ABSENT):
        return max_lines, notes, ""
    if want is None or want < 1 or max_lines not in (None, want) and not (max_lines == 2000 and want <= 2000):
        conflict = ("is not a whole line number" if want is None else f"is before start_line={start}" if want < 1
                    else f"asks for {want} lines from start_line={start}, max_lines={max_lines!r} for another number")
        return max_lines, notes, argument_refusal(ctx, "TOOL_ARG_ERROR", [
            f"end_line={end_line!r} {conflict}; {ready('max_chars')}max_lines=<lines>)"], effect="Nothing was read.")
    return want, notes + [f"end_line={end} read as max_lines={want} from start_line={start}"], ""
