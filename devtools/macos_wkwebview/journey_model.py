"""The keyless loopback model of one window-renewal arm, with generation-bound request attribution.

Each tagged task reads its probe file for real, then every later request of that task is held
until released, so a restart cuts the round and the restored worker sends the same saved read
again. Untagged calls (Main) answer at once with the words to reply to.

A held request is bound to a server generation only through the requesting process's measured
parent chain and birth (``host_processes``) — never through the binding file's arrival view and
never through a receive time. A request whose requester cannot be resolved stays unattributed;
the arm's checks treat an untied arrival as missing evidence, never as silence or continuity.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from devtools.macos_wkwebview.host_processes import connection_owner, process_ancestry, ps

TASKS = ("live", "paused")


def task_description(key: str) -> str:
    """The one description the driver submits to the task ingress; its leading marker is the task's own identity."""
    return f"[E2E-LINEAGE:{key}] Read probe-{key}.txt and retain its exact content."


def ingress_key(body: dict, message_text) -> str:
    """The task named by its ORIGINATING user message: the first user row that opens with its own marker.

    The host's roster note (a later user-role TAIL row opening ``[System task message]``), a Main or Project turn
    and a transcript quoting another task's title all carry marker text without being that task's input, so no
    marker found elsewhere in the request ever names a task.
    """
    for message in body.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = message_text(message).lstrip()
        for key in TASKS:
            if text.startswith(f"[E2E-LINEAGE:{key}]"):
                return key
    return ""


def continuation(resumed, live, task_id, original_reads):
    """Only a tied request proves continuation. Neither a quiet timeout nor an untied request proves a stop."""
    seen_id, status, reads = live.get("task_id"), live.get("status"), live.get("read_requests")
    if seen_id and (seen_id != task_id or status in ("failed", "cancelled", "completed")):
        return False
    if type(original_reads) is int and type(reads) is int:
        if reads > original_reads:
            return False
        if resumed and seen_id == task_id and status and reads == original_reads:
            return True
    return None


def pause_kept(paused, task_id, original=None):
    """Owner-pause proof requires silence in both counters, even when task activity is still paused."""
    unchanged = None
    if original:
        counters = [(paused.get(k), original.get(k)) for k in ("held_requests", "read_requests")]
        known = [(after, before) for after, before in counters if type(after) is int and type(before) is int]
        if any(after > before for after, before in known) or (paused.get("requests_in_generation") or 0) > 0:
            return False
        if known == counters and all(after == before for after, before in known):
            unchanged = True
    if not paused.get("phase") and paused.get("activities_complete") is not True:
        return None
    # The gateway's own typed "unknown": the owner-Pause fence or its reason could not be read. That is missing
    # evidence (counters above already ruled out any advance), never a demonstrated change of Pause.
    if paused.get("phase") == "unknown" or paused.get("pause_cause") == "unknown":
        return None
    if (paused.get("activity_id"), paused.get("phase"), paused.get("pause_cause")) != (task_id, "budget_paused", "owner"):
        return False
    return unchanged if original else True


# Checks the page does not own: a late successor view cannot re-measure them, so only the others are demoted by it.
SERVER_SIDE = ("generation_new", "client_health", "update_landed", "diagnostic_lookup_recorded", "old_draft",
               "project_active_before", "live_continued", "pause_kept", "accepted_once", "no_replay")


def reconcile_view(label, old, sampled, final, checks, measured, same_view):
    """A settle that timed out on the still-observed old view is provisional, not a verdict. Only a final
    observation of that same view confirms it as the arm's own negative (the baseline's control, a candidate's
    failure to renew). A different or unreadable final view means the steps in between measured the old
    document: view_new and every page-bound check in ``measured`` stay unestablished, never green. Returns the
    gap to record, else None."""
    if not (old and sampled and sampled["view"] == old["view"]):
        return None
    if final and same_view(old, final):
        checks["view_new"] = False
        return None
    checks["view_new"] = None
    for name in measured:
        if name not in SERVER_SIDE and checks.get(name) is not None:
            checks[name] = None
    return (f"{label}: the window still showed its old view when its settle time ended, and a different "
            "or unreadable view was seen afterwards; the steps in between measured the old document")


FIXTURE_BUTTON = "Open the sample page"


def fixture_page(port: int, tag: str, page_path: str | None = None) -> str:
    path = page_path or f"http://127.0.0.1:{port}/p-{tag}"
    target = f"{path}?k-{tag}=q-{tag}#f-{tag}"
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Window renewal navigation"
            "</title><style>body{font:18px system-ui;margin:48px}button{font-size:20px;padding:16px 28px}"
            "</style></head><body><h1>Window renewal navigation</h1>"
            f"<button type=\"button\" onclick=\"this.disabled=true;location.assign('{target}')\">"
            f"{FIXTURE_BUTTON}</button></body></html>\n")


def journey_model(workspaces: dict, tag: str, binding: Path, known_generations):
    """Build this arm's JourneyModel over the shared stub (tests/system_e2e/harness.py).

    ``known_generations`` is a callable listing every ``(pid, birth)`` server generation the
    driver has already confirmed; the binding file only proposes the arrival-time candidate.
    """
    from tests.system_e2e.harness import ScriptedStubModel, classify_call, message_text, scripted_completion

    class JourneyModel(ScriptedStubModel):
        def __init__(self):
            self.tokens = {key: f"{key.upper()}-{tag}" for key in TASKS}
            for key, token in self.tokens.items():
                (workspaces[key] / f"probe-{key}.txt").write_text(token + "\n", encoding="utf-8")
            self.held = {key: {} for key in TASKS}
            self.read_requests = dict.fromkeys(TASKS, 0)
            self.release = {key: threading.Event() for key in TASKS}
            self.count_lock = threading.Lock()
            self.unattributed = dict.fromkeys(TASKS, 0)
            self.basis = {}
            self._peer = threading.local()
            super().__init__(gate=self._gate, final_answer=f"QUOTE-{tag}: a synthetic answer to reply to.", select_retained=True)
            # Record the requesting connection on its own handler thread, so the gate can
            # measure WHO asks: the per-instance handler class is this stub's own.
            handler = self._server.RequestHandlerClass
            self._serve_post = handler.do_POST
            model = self

            def do_POST(request):  # noqa: N802 - the stdlib callback's own name
                try:
                    model._peer.value = request.connection.getpeername()
                except OSError:
                    model._peer.value = None
                try:
                    model._serve_post(request)
                finally:
                    model._peer.value = None

            handler.do_POST = do_POST

        def _key(self, body):
            if classify_call(body) not in ("agent", "finalization"):
                return ""
            for key in TASKS:
                if self._read(body, key):
                    return key
            return ingress_key(body, message_text)

        def _read(self, body, key):
            results = "\n".join(message_text(m) for m in body.get("messages", []) if m.get("role") == "tool")
            return self.tokens[key] in results

        def _attribute(self):
            """``(generation, basis)`` for this thread's request: the binding file proposes a
            candidate, the requester's measured parent chain and the process birth decide."""
            try:
                server = json.loads(binding.read_text())["main"]
                candidates = [(int(server["pid"]), server["fingerprint"]["start_time"])]
            except (OSError, ValueError, KeyError, TypeError):
                candidates = []
            peer = getattr(self._peer, "value", None)
            owner = connection_owner(peer[1]) if peer else None
            if owner is None:
                return None, "requester_unresolved"
            chain = set(process_ancestry(owner))
            for pid, birth in dict.fromkeys([*candidates, *known_generations()]):
                if pid in chain and ps(pid, "lstart") == birth:
                    return (pid, birth), "parent_chain"
            return None, "generation_not_in_chain"

        def _gate(self, body):
            key = self._key(body)
            if key and self._read(body, key):
                generation, basis = self._attribute()
                with self.count_lock:
                    counts = self.held[key]
                    counts[generation] = counts.get(generation, 0) + 1
                    self.basis[basis] = self.basis.get(basis, 0) + 1
                    if generation is None:
                        self.unattributed[key] += 1
                if not self.release[key].wait(3600):
                    raise TimeoutError(f"the {key} request was never released")

        def _answer(self, body, seq):
            key = self._key(body)
            if not key:
                return super()._answer(body, seq)
            def next_read(_):
                if self._read(body, key):
                    return None
                with self.count_lock:
                    self.read_requests[key] += 1
                return {"tool": "read_file", "arguments": {"root": "active_workspace", "path": f"probe-{key}.txt"}}
            return scripted_completion(body, seq, next_read,
                                       f"Finished: {self.tokens[key]}", select_retained=True)

        def count(self, key, generation=None):
            with self.count_lock:
                if generation is None:
                    return sum(self.held[key].values())
                return self.held[key].get((generation["pid"], generation["birth"]), 0)

        def activity(self, key, generation):
            with self.count_lock:
                held = self.held[key]
                return {"held_requests": sum(held.values()), "read_requests": self.read_requests[key],
                        "requests_in_generation": held.get((generation["pid"], generation["birth"]), 0)}

        def attribution(self):
            """Aggregate attribution facts for the report: counts by basis, never request data."""
            with self.count_lock:
                return {"basis": dict(self.basis),
                        "unattributed": dict(self.unattributed)}

    return JourneyModel()
