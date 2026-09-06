"""Every typed worker event — literal or module-constant typed — must have a supervisor handler.

The supervisor drops an event whose type is not in EVENT_HANDLERS: it is
downgraded to a truncated ``unknown_worker_event`` repr in supervisor.jsonl
(see ``dispatch_event``), so its payload is lost to every typed consumer.
v6.69.0 registered one branch of review_helpers' if/else and missed the
other; this scan closes the class: it walks the emitter call sites and
asserts each literal event type is either registered or on the explicit
allowlist below, with a reason.
"""

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

# Types that are deliberately NOT in EVENT_HANDLERS. Every entry carries the
# reason it is safe; an unexplained entry is the exact failure mode this test
# exists to end (a place unknown events go to be silently blessed).
ALLOWLIST = {
    # Intercepted by server.py's dispatch loop BEFORE the registry lookup
    # (the restart path must run even when handler dispatch is degraded).
    "restart_request",
}

# Emitter shapes covered: queue puts, agent pending-event appends, and the
# review-helpers wrapper (whose queue put forwards a caller-built literal).
_QUEUE_HINTS = ("event_q", "event_queue", "out_q", "eq")
_WRAPPER_FUNCS = {"emit_review_event"}


def _dict_type(node, constants=None):
    """The event type of a dict literal: a literal string, or a module-level
    string constant given by bare name (`DISCLOSURE_X`) or attribute
    (`task_pacing.DISCLOSURE_X`) — the R36 pacing facts were emitted through
    constants and a literal-only scan blessed them unregistered."""
    if not isinstance(node, ast.Dict):
        return None
    constants = constants or {}
    for key, value in zip(node.keys, node.values):
        if not (isinstance(key, ast.Constant) and key.value == "type"):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.Name):
            return constants.get(value.id)
        if isinstance(value, ast.Attribute):
            return constants.get(value.attr)
    return None


def _string_constants(trees):
    """UPPER_CASE module-level `NAME = "literal"` assignments across the scanned
    files, keyed by bare name (an attribute access resolves by its last part)."""
    constants = {}
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target, value = node.targets[0], node.value
            if (
                isinstance(target, ast.Name) and target.id.isupper()
                and isinstance(value, ast.Constant) and isinstance(value.value, str)
            ):
                constants.setdefault(target.id, value.value)
    return constants


def _receiver_name(func):
    parts = []
    node = func.value
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _emitted_types():
    emitted = {}
    files = [REPO / "server.py"]
    files += list((REPO / "ouroboros").rglob("*.py"))
    files += list((REPO / "supervisor").rglob("*.py"))
    trees = {}
    for path in files:
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    constants = _string_constants(trees.values())
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            candidates = []
            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                receiver = _receiver_name(node.func).lower()
                if attr == "append" and "pending_events" in receiver:
                    candidates = node.args
                elif attr in ("put", "put_nowait"):
                    # Receiver-AGNOSTIC (the narrow name hints missed a real
                    # registered emitter, `q.put_nowait` in delegate_progress):
                    # any `.put`/`.put_nowait` whose argument is a dict with a
                    # literal string `type` is scanned. A non-supervisor queue
                    # that happens to match falls to the reasoned allowlist.
                    candidates = node.args
            if _call_name(node.func) in _WRAPPER_FUNCS:
                candidates = node.args
            for arg in candidates:
                event_type = _dict_type(arg, constants)
                if event_type:
                    emitted.setdefault(event_type, []).append(
                        f"{path.relative_to(REPO)}:{node.lineno}"
                    )
    return emitted


def _registered_types():
    registered = set()
    for name in (
        "supervisor/events.py",
        "supervisor/cognitive_operations.py",
        "supervisor/events_chat_delivery.py",
        "supervisor/telemetry_events.py",
    ):
        tree = ast.parse((REPO / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                getattr(target, "id", "").endswith("EVENT_HANDLERS")
                for target in node.targets
            ):
                continue
            if isinstance(node.value, ast.Dict):
                registered |= {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant)
                }
    return registered


def test_every_emitted_literal_type_has_a_handler():
    emitted = _emitted_types()
    registered = _registered_types()
    # Sanity: the scan sees real emitters and the registry parse worked.
    assert "task_done" in emitted
    assert "task_done" in registered
    missing = {
        event_type: sites
        for event_type, sites in emitted.items()
        if event_type not in registered and event_type not in ALLOWLIST
    }
    assert not missing, (
        "Emitted worker event types without a registered handler (they would "
        f"be dropped as unknown_worker_event): {missing}. Register them in "
        "supervisor/events.py EVENT_HANDLERS or add an ALLOWLIST entry with "
        "a reason."
    )


def test_allowlist_entries_stay_emitted():
    """An allowlist row for a type nobody emits any more is stale."""
    emitted = _emitted_types()
    stale = ALLOWLIST - set(emitted)
    assert not stale, f"ALLOWLIST rows no longer emitted anywhere: {stale}"


def test_previously_dropped_types_are_registered():
    """The concrete v6.69.0-class omissions stay fixed."""
    registered = _registered_types()
    for event_type in (
        "review_wave_budget_partial_unknown",
        "task_message_injected",
        "advisory_suspect_result",
        "advisory_contract_warning",
        "plan_task_deadline_skip",
    ):
        assert event_type in registered, event_type


def test_the_scan_resolves_an_event_type_given_as_a_module_constant():
    """A literal-only scan once blessed constant-typed emitters unregistered and
    the supervisor dropped them as unknown_worker_event. No production emitter
    names its type through a constant today (the R36 pacing fact that did is
    deleted), so the resolver is pinned on its own source instead of on a live
    subject that can disappear again — losing the capability silently is
    exactly the failure this file exists to prevent."""
    source = (
        'DISCLOSURE_X = "typed_fact_by_name"\n'
        'def emit(event_q, ctx):\n'
        '    event_q.put({"type": DISCLOSURE_X})\n'
        '    ctx.pending_events.append({"type": pacing.DISCLOSURE_X})\n'
        '    emit_review_event(ctx, {"type": "typed_fact_by_literal"})\n'
    )
    tree = ast.parse(source)
    constants = _string_constants([tree])
    assert constants["DISCLOSURE_X"] == "typed_fact_by_name"
    dicts = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)]
    assert [_dict_type(node, constants) for node in dicts] == [
        "typed_fact_by_name",  # bare name
        "typed_fact_by_name",  # attribute access, resolved by its last part
        "typed_fact_by_literal",
    ]
    assert _dict_type(dicts[0], {}) is None  # without the constant table: unresolvable
