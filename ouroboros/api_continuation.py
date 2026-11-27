"""An accepted task API admission is Continue's source: the request its door saved, never owner words.

The opening is copied by value from the predecessor's record and chained by value into a Continue of a Continue;
``owner_continue`` collects it beside its other openings and renders it in the successor's work order. A letter's
root keeps the agent author its door recorded (``peer_letters.opening_author``), so that work order quotes the
letter under its linked Ouroboros (``peer_letters.continued_letter_lines``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ouroboros.peer_letters import opening_author

API_OPENING = "api_request"
# The request words the API door saved on the root (``gateway/tasks._complete_api_task_admission``).
API_REQUEST_FIELDS = ("description", "context", "expected_output", "constraints")


def accepted_api_opening(result: Dict[str, Any], metadata: Dict[str, Any], task_id: str,
                         gaps: List[str]) -> Optional[Dict[str, Any]]:
    """The request the API door saved with its accepted receipt, copied by value, or a named gap. The words are its
    caller's under the channel it declared (``client_surface``, the door's one stamp of it); the receipt marks the
    opening, it proves no caller identity. Its inputs are the work's declared inputs
    (``owner_continue.declared_inputs``)."""
    receipt = result.get("api_admission")
    surface = metadata.get("client_surface") if isinstance(metadata.get("client_surface"), dict) else {}
    opening = {"source": API_OPENING, "task_id": task_id, "channel": surface.get("channel"),
               **{key: result[key] for key in API_REQUEST_FIELDS if isinstance(result.get(key), str) and result[key]},
               **opening_author(metadata)}  # a linked Ouroboros's letter keeps its agent author
    if not (isinstance(receipt, dict) and receipt.get("status") == "accepted" and valid_api(opening)):
        gaps.append("api_request_unreadable")
        return None
    return opening


def valid_api(api: Any) -> bool:
    """An API opening exactly as ``accepted_api_opening`` builds it (also its chained copy)."""
    return (isinstance(api, dict) and api.get("source") == API_OPENING
            and all(isinstance(api.get(key), str) and api[key].strip() for key in ("task_id", "channel", "description"))
            and all(isinstance(api.get(key, ""), str) for key in API_REQUEST_FIELDS))
