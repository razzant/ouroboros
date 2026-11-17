"""The phone's one-line source label: the host's Project or named-work name, nothing else."""
from __future__ import annotations

# These carry their own typed completion/refusal heading; a second one would only repeat it.
_SELF_HEADED = {"project_completion_summary"}


def origin_label(event, lang=""):
    """The host's label as literal text; an older producer's or an unknown source's message stays unheaded."""
    del lang  # a name is never translated
    origin = event.get("message_origin") if isinstance(event, dict) else None
    if (not isinstance(origin, dict) or event.get("annotation_type") == "routing_ack" or event.get("routing_notice_carrier")
            or event.get("role") == "system" and event.get("system_type") in _SELF_HEADED):
        return ""
    return " ".join(str(origin.get("label") or "").replace("\x00", "").split())
