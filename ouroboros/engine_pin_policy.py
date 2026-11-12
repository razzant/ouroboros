"""Pin compatibility for Apply disclosure and the Accounts engine-restart preview.

This compares selected pins, not physical process lifetime: application Exit/Restart
always ends the old owned engine. A lower pin still cannot read a newer engine's data.
"""

from ouroboros.claudexor_runtime import release_below


def engine_pin_transition(pin, serving: tuple[str, str] | None) -> dict:
    """Compare pins for disclosure, not the application's unconditional process teardown.

    Compatibility field action='keep' means no newer engine is selected by this pin;
    it never promises process survival or makes forward-only data readable by an older pin.
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
