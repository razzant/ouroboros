"""Pure engine-pin rule shared by Apply disclosure and restart teardown."""

from ouroboros.claudexor_runtime import release_below


def engine_pin_transition(pin, serving: tuple[str, str] | None, *, owner_restart: bool = False) -> dict:
    """Unknown comparisons preserve the handoff; a lower pin keeps a newer live engine.

    Keeping it does not make its forward-only data readable on a later cold start.
    Manual Restart has already stopped the engine and keeps its attested retry.
    """
    target = {"version": pin.version, "build_sha": pin.build_sha} if pin is not None else None
    result = {"target": target, "serving": {"version": serving[0], "build_sha": serving[1]} if serving else None,
              "action": "unknown", "direction": "unknown", "data_direction": "forward_only"}
    if pin is None or serving is None:
        return result
    if serving == (pin.version, pin.build_sha):
        return {**result, "action": "keep", "direction": "unchanged"}
    if release_below(pin.version, serving[0]):
        return {**result, "action": "replace" if owner_restart else "keep", "direction": "backward"}
    direction = ("forward" if release_below(serving[0], pin.version)
                 else "same_release" if pin.version == serving[0] else "unordered")
    return {**result, "action": "replace", "direction": direction}
