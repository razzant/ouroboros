"""Ollama load windows for a model on a given connection and endpoint.

Role windows still cap their own host views. The largest explicitly configured
window on a shared runner (one connection and model) is requested by every role,
avoiding load/cache churn, and is capped by the maximum the model's /api/show
states; a model that states none takes no explicit load (pull it, or use Auto).
No role setting (Auto) means no num_ctx override; /api/ps is the capacity witness.
"""
from __future__ import annotations


def native_target_key(target):
    """Only an account-bound physical target may reuse a native protocol or load receipt."""
    from ouroboros.context_window import bound_target_scope
    return "ollama:" + bound_target_scope(target, target.get("requested_model") or target.get("resolved_model", ""))


def thinking_mode_from_show(shown):
    """Only an explicit value domain distinguishes named levels from an on/off switch."""
    capabilities = shown.get("capabilities") if isinstance(shown, dict) else None
    fallback = "legacy" if isinstance(capabilities, list) and "thinking" in capabilities else None
    descriptor = shown.get("thinking") if isinstance(shown, dict) else None
    values = descriptor.get("values") if isinstance(descriptor, dict) else None
    if not isinstance(values, list) or not values or not all(
            isinstance(value, bool) or isinstance(value, str) and value.strip() for value in values):
        return fallback
    # Ollama uses legacy aliases when the advertised descriptor has no valid default.
    default = descriptor.get("default")
    if not isinstance(default, (str, bool)) or default not in values or len(set(values)) != len(values):
        return fallback
    if all(isinstance(value, bool) for value in values) and True in values:
        return "boolean"
    # As on /v1, submit the chosen name and let the model validate it.
    return "named"


def _window_route(target):
    """Use the capability address captured when this endpoint was resolved, as the planner does."""
    from ouroboros.capability_evidence import route_fingerprint

    identity = target.get("capability_route") or target
    model = identity.get("model") or target.get("requested_model") or target["resolved_model"]
    fp = route_fingerprint(provider=target["provider"], model=model, base_url=identity.get("base_url", ""),
                           headers=target.get("default_headers"))
    return model, fp


def forget_load(data, target):
    """Retire load-dependent statements and outcomes together, preserving the route's sequence."""
    from ouroboros.context_window import bound_target_scope
    held = data.get("probes", {}).pop(native_target_key(target), None)
    scope = bound_target_scope(target, target.get("requested_model") or target["resolved_model"])
    frontier = data.get("main_windows", {}).get(scope)
    if frontier and ("reads" in frontier or "observations" in frontier):
        frontier.pop("reads", None)
        frontier.pop("observations", None)
        return True
    return held is not None


def load_version(target, root=None):
    """The order of the scope's current statement, taken before a ``/api/ps`` read so that its publication
    (``record_window(read_sequence=...)``) can tell a statement published since: an overtaken read writes nothing."""
    from ouroboros import capability_evidence as ce
    from ouroboros.context_window import bound_target_scope, window_read_sequence
    scope = bound_target_scope(target, _window_route(target)[0])
    record = ce._read(root or ce.canonical_evidence_root()).get("main_windows", {}).get(scope) or {}
    return window_read_sequence(record) if scope else None


def record_window(target, window, *, requested=0, root=None, previous=..., read_sequence=None, replay=False):
    """Publish one runner statement and its cached witness together; return its scoped order.

    ``read_sequence``: the order the statement was read against (``load_version``, or a send's own dispatch
    order). A scope that has moved on since holds a newer statement: this one is not written and has no order.
    ``replay``: a read made before a send that finds the scope already saying the same is that statement, and
    takes its order; an answer's closing read never does (it would bind the answer to a load it was not sent under)."""
    if window is None:
        return None
    from ouroboros import capability_evidence as ce
    from ouroboros.context_window import WindowLimit, _put_read, _window_age, bound_target_scope, window_read_sequence
    model, fp = _window_route(target)
    root, key = root or ce.canonical_evidence_root(), native_target_key(target)
    scope = bound_target_scope(target, model)
    if not scope:
        return None
    detail = f"Ollama /api/show ({requested})" if requested else "Ollama /api/ps"
    try:
        with ce._store_transaction(root):
            data = ce._load(root)
            record = data.setdefault("main_windows", {}).setdefault(scope, {})
            probes = data.setdefault("probes", {})
            held = probes.get(key, {})
            if read_sequence is not None and read_sequence != window_read_sequence(record):
                stated = next(iter(record.get("reads", [])), None)
                same = replay and (stated["tokens"] if stated else 0) == window and (not stated or (
                    stated["kind"], stated["source"], stated.get("instance_id", "")) == ("total", ce.SOURCE_SERVER_METADATA, ""))
                return window_read_sequence(record) if same else None
            if previous is not ... and probes.get(key) != previous:
                return None
            if window == 0:
                # An answered Auto read found no load; an unavailable read never erases one.
                if forget_load(data, target) and not ce._save(root, data):
                    return None
                return window_read_sequence(record)
            if not (held.get("window_tokens") == window and held.get("detail") == detail
                    and held.get("source") == ce.SOURCE_SERVER_METADATA and held.get("status") == ce.STATUS_CONFIRMED
                    and _window_age(str(held.get("ts") or "")) < ce._CONFIRMED_TTL_SEC / 2):
                evidence = ce.CapabilityEvidence(window, ce.STATUS_CONFIRMED, ce.SOURCE_SERVER_METADATA, fp, model,
                                                 target["provider"], ts=ce.utc_now_iso(), detail=detail)
                held = ce.stamp_receipt(evidence).to_json()
            limit, changed = _put_read(record, WindowLimit(window, "total", ce.SOURCE_SERVER_METADATA,
                                      held["ts"], read_id=held.get("window_read_id", "")), scope)
            probe = {**held, "window_sequence": limit.sequence, "window_sequence_route": scope}
            if changed or probes.get(key) != probe:
                ce._drop_expired_probes(data)
                probes[key] = probe
                if not ce._save(root, data):
                    return None
            return limit.sequence
    except OSError:
        return None  # publication is best effort; the request's own refusal remains useful
