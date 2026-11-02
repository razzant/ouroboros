"""What a tool says about an argument it did not obey (DEVELOPMENT "LLM-first affordances").

Models may fill optional keys (OpenAI's Responses API tries strict mode when a
function tool leaves `strict` unset, so the OpenRouter lane sends `strict: false`). A value
that asks for nothing takes the omitted path and the result says so in one line; a
value that asks for something the call cannot serve is refused ONCE, typed, naming
the field, the value received and the repair — a refusal that only restates the
rule is retried unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable

from ouroboros.tools.tool_result import (
    ToolResult, _publish_tool_result, _published_tool_result, _replace_tool_result,
)


def ignored_argument_note(name: str, value: Any, why: str) -> str:
    """One result line for an argument that took the omitted path."""
    return f"{name}={value!r} ignored: {why}"


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
