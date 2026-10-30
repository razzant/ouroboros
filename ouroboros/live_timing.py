"""Renewal timing assembly behind the admitted host port and its existing journal."""
import time

from ouroboros.contracts.realtime import canonical_json


def record_timing(host, event, *, at=None):
    """One journal fact per renewal; absent boundaries remain explicitly unknown."""
    if event.startswith('offer_wait:') or event in {'offer_idle', 'offer_wait_done'}:
        _offer_wait(host, event, time.monotonic() if at is None else at)
        return
    stamp = time.time() if at is None else at
    current = getattr(host, "_renewal_timing", None)
    if event in {"requested", "closed"} and current is not None:
        host._renewal_timing = None
        host.observe("host", canonical_json({"type": "call.timing", **current,
            "ended_at": stamp, "ended_by": event}))
    if event == "requested":
        host._renewal_number = getattr(host, "_renewal_number", 0) + 1
        host._renewal_timing = {"renewal": host._renewal_number, "requested_at": stamp,
            "quiet_at": None, "fit_done_at": None, "provider_ready_at": None, "first_audio_at": None,
            "prepared_ready_at": None, "switched_at": None}
        return
    if event == "closed" or current is None:
        return
    field = {"quiet": "quiet_at", "fit_done": "fit_done_at", "ready": "provider_ready_at",
             "prepared_ready": "prepared_ready_at", "switched": "switched_at", "first_audio": "first_audio_at"}.get(event)
    if field is None:
        return
    # A requested renewal may still hear the retiring connection before replacement fit/ready.
    if ((event == "ready" and current["fit_done_at"] is None)
            or (event == "first_audio" and (current["provider_ready_at"] is None
                or current.get('prepared_ready_at') is not None and current.get('switched_at') is None))):
        return
    # An account rebind can fit twice before open: the last completed fit is the sent one.
    if current[field] is None or event == "fit_done":
        current[field] = stamp
    if event == "first_audio":
        host._renewal_timing = None
        host.observe("host", canonical_json({"type": "call.timing", **current}))


def _offer_wait(host, event, stamp):
    """Boundary measurements, no timer policy. Concurrent blockers overlap; total does not."""
    state = getattr(host, '_offer_wait_timing', None)
    if state is None:
        state = host._offer_wait_timing = dict(at=stamp, pending=False, blockers=(), total=0.0, seconds={}, done=False)
    if state['done']:
        return
    elapsed = max(0.0, stamp - state['at'])
    if state['pending']:
        state['total'] += elapsed
        for name in state['blockers']:
            state['seconds'][name] = state['seconds'].get(name, 0.0) + elapsed
    state.update(at=stamp, pending=event.startswith('offer_wait:'), blockers=tuple(event.partition(':')[2].split(',')))
    state['blockers'] = tuple(name for name in state['blockers'] if name)
    if event == 'offer_wait_done':
        state['done'] = True
        times = state['seconds']
        host.observe('host', canonical_json({'type': 'call.offer_wait',
            'pending_seconds': round(state['total'], 6),
            'longest_blocker': max(sorted(times), key=times.get) if times else None,
            'seconds_by_blocker': {k: round(v, 6) for k, v in sorted(times.items())}}))
