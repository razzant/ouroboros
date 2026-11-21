"""The lead instruction and its delivery fact, shared by tool schemas and send receipts.

Only the description of the chronicle tool in the final host projection proves delivery.
For an external engine that projection is the list handed to its interface; its own
internal transformations are unobserved. No instruction in messages proves this fact.
"""
from collections.abc import Mapping

LEAD_RULE_VERSION = 1
LEAD_RULE = (
    "A note's, page's or part's first sentence is its lead, my shortest telling of it: "
    "in a tight window my later self and my other foci read that sentence in place of "
    "the record, so it says what the record holds, settled and left open."
)


def delivered_lead_rule(candidate: Mapping) -> int | None:
    """The revision delivered by this attempt, or None; missing delivery never forbids a write."""
    projections = [candidate]
    for event in candidate.get("events") or ():
        if isinstance(event, Mapping):
            projections.extend(event[key] for key in ("setup", "session") if isinstance(event.get(key), Mapping))
    for projection in projections:
        schemas = [*(projection.get("tools") or ()), *(projection.get("functions") or ())]
        for tool in schemas:
            if not isinstance(tool, Mapping):
                continue
            if "functionDeclarations" in tool:
                schemas.extend(tool["functionDeclarations"])
            schema = tool.get("function") or tool.get("custom") or tool
            if (isinstance(schema, Mapping) and schema.get("name") == "chronicle_write"
                    and LEAD_RULE in str(schema.get("description") or "")):
                return LEAD_RULE_VERSION
    return None
