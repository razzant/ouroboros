"""Pure engine-pin rule shared by Apply disclosure, the Restart Claudexor preview and every restart teardown.

One rule for every restart door, the owner's Restart included (owner 3A, 2026-11-06): a forward
pin replaces the serving engine, a lower pin keeps the newer live one, an unknown comparison
preserves the handoff. Restart restarts Ouroboros, never the shared engine by itself; the
engine's own restart is the Accounts action (``claudexor_restart``).
"""

from ouroboros.claudexor_runtime import release_below


def engine_pin_transition(pin, serving: tuple[str, str] | None) -> dict:
    """Unknown comparisons preserve the handoff; a lower pin keeps a newer live engine.

    Keeping it does not make its forward-only data readable on a later cold start.
    """
    target = {"version": pin.version, "build_sha": pin.build_sha} if pin is not None else None
    result = {"target": target, "serving": {"version": serving[0], "build_sha": serving[1]} if serving else None,
              "action": "unknown", "direction": "unknown", "data_direction": "forward_only"}
    if pin is None or serving is None:
        return result
    if serving == (pin.version, pin.build_sha):
        return {**result, "action": "keep", "direction": "unchanged"}
    if release_below(pin.version, serving[0]):
        return {**result, "action": "keep", "direction": "backward"}
    direction = ("forward" if release_below(serving[0], pin.version)
                 else "same_release" if pin.version == serving[0] else "unordered")
    return {**result, "action": "replace", "direction": direction}
