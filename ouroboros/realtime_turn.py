"""Process-local authority and route facts of an admitted call.

Durable intent can identify a call but cannot recreate its live host port.
"""
from __future__ import annotations

from dataclasses import asdict, replace

def recovery_refusal(task, turn):
    """A persisted transport intent cannot recreate its process-local authority."""
    request = (task.get("metadata") or {}).get("execution_transport")
    if request is None:
        return ""
    if isinstance(turn, RealtimeTurn) and turn.matches_live_request(request):
        return ""
    task["_skip_post_task_synthesis"] = True
    return "realtime_restart_required"


class RealtimeTurn:
    """An admitted actor, never reconstructed from serializable metadata."""

    def matches_live_request(self, request):
        with self._lock:
            return not self._closed and request == self.request

    def observation_fingerprint(self):
        """Route/account evidence from the provider's granted settings, retaining only a hash."""
        from ouroboros.realtime_credentials import capture_session_credentials

        return capture_session_credentials(self).scope

    def resolve_route(self, task, *, allow_fetch):
        """Ordinary fitting against this transport's exact cached route evidence.

        Neither a Main-role window nor a chat-completions endpoint describes a
        realtime socket. Unknown evidence stays unknown; no model/window table.
        """
        from ouroboros.config import DATA_DIR
        from ouroboros.response_limits import resolve_response_limit

        route = dict(provider=self.route.accounting_provider, model=self.model,
                     base_url=self.route.endpoint, use_local=False,
                     model_role="main", options={})
        args = {key: route[key] for key in ("provider", "model", "base_url")}
        from ouroboros.capability_evidence import record_route_facts
        evidence = record_route_facts(DATA_DIR, **args, facts=self.catalog_model.facts)
        limit = resolve_response_limit(DATA_DIR, **args, allow_fetch=False)
        scope = self.observation_fingerprint()
        route["window_observation_fp"] = scope
        return route, replace(evidence, route_fp=scope, response_limit=asdict(limit))
