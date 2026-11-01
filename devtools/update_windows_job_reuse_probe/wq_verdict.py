"""Case verdicts and the pure predicates they are made of (stdlib only, portable).

A case is PASS only when it recorded at least one check and every check is literally ``True``.
A check is ``False`` (the expectation was measured and refuted) or ``None`` (it could not be
measured: a deadline, an unreadable record, a holder that never acknowledged). Any ``False``
makes the case FAIL; otherwise any ``None`` makes it UNKNOWN. A case that never started is
NOT_RUN with the reason. The process exit code is 0 only for the complete required selection
with every case PASS and every mode's process custody confirmed; nothing else maps to 0.
"""
from __future__ import annotations

import json
import ntpath
import re

PASS, FAIL, UNKNOWN, NOT_RUN = "PASS", "FAIL", "UNKNOWN", "NOT_RUN"
# Cases in execution order, grouped by the private root (mode) they run in.
MODES = {
    "U": ("entry.U", "A", "B", "C", "E.U"),   # released app, user site
    "D": ("entry.D", "D", "E.D"),             # released app, --runtime-venv with spaces
    "F": ("F", "E.F"),                        # bare Windows Python, private environment
}
REQUIRED = tuple(case for cases in MODES.values() for case in cases)
EXIT = {"qualified": 0, "fail": 1, "unknown_or_not_run": 2, "partial": 3, "custody": 4, "usage": 5}


class CaseStop(Exception):
    """A required step of a case was refuted or could not be measured; the case ends here."""


class Case:
    """``required``: the check names this case must have recorded to PASS (a skipped path cannot pass)."""

    def __init__(self, name, emit=None, required=()):
        self.name, self.checks, self.forced, self.reason = name, [], None, ""
        self.required = tuple(required)
        self._emit = emit or (lambda *args, **kwargs: None)

    def check(self, name, ok, **evidence):
        """Record one check. Only ``True`` passes; ``None`` is unmeasured, anything else refuted."""
        value = True if ok is True else (None if ok is None else False)
        row = {"case": self.name, "check": name, "ok": value, **evidence}
        self.checks.append(row)
        self._emit("check", **row)
        return value

    def require(self, name, ok, **evidence):
        if self.check(name, ok, **evidence) is not True:
            raise CaseStop(f"{self.name}: {name}")
        return True

    def not_run(self, reason):
        self.forced, self.reason = NOT_RUN, reason

    def unknown(self, reason):
        self.forced, self.reason = UNKNOWN, reason

    @property
    def missing(self):
        recorded = {row["check"] for row in self.checks}
        return [name for name in self.required if name not in recorded]

    @property
    def status(self):
        values = [row["ok"] for row in self.checks]
        if False in values:
            return FAIL  # a refuted expectation outranks a forced state
        if self.forced:
            return self.forced
        if not values or self.missing:
            return UNKNOWN  # a case that measured nothing, or not everything it claims, proved nothing
        return UNKNOWN if None in values else PASS

    def as_dict(self):
        return {"case": self.name, "status": self.status, "reason": self.reason, "missing": self.missing,
                "checks": self.checks}


def select(spec):
    """``(cases, partial)`` for ``--cases``. ``None``/``all`` is the full required selection; a subset is a
    partial run whose exit code is never 0. Unknown or repeated names refuse before anything runs."""
    if spec in (None, "", "all"):
        return REQUIRED, False
    names = [part.strip() for part in str(spec).split(",") if part.strip()]
    unknown = [name for name in names if name not in REQUIRED]
    if unknown or not names or len(set(names)) != len(names):
        raise ValueError(f"cases must be distinct names from {', '.join(REQUIRED)}; got {spec!r}")
    ordered = tuple(case for case in REQUIRED if case in names)
    return ordered, ordered != REQUIRED


def conjoin(*values):
    """Three-valued AND of check values: ``False`` when any is refuted, else ``None`` when any is unmeasured,
    else ``True`` (only a literal ``True`` is true). Python's ``and`` would let an unmeasured part hide a
    refuted one, or an unmeasured equality of two missing readings pass."""
    values = [True if value is True else (None if value is None else False) for value in values]
    return False if False in values else (None if None in values else True)


def exit_code(statuses, selected, partial, custody_confirmed):
    """The one mapping from results to the process exit code (see module docstring). A selection that is
    not exactly the required one is partial whatever the caller says."""
    partial = partial or tuple(selected) != REQUIRED
    if custody_confirmed is not True:
        return EXIT["custody"]
    found = [statuses.get(case, NOT_RUN) for case in selected]
    if FAIL in found:
        return EXIT["fail"]
    if any(status != PASS for status in found) or not found:
        return EXIT["unknown_or_not_run"]
    return EXIT["partial"] if partial else EXIT["qualified"]


# --------------------------------------------------------------------------- predicates over records
def same_path(a, b):
    """Windows path equality as the file system resolves names: case-insensitive, either slash, no
    ``\\\\?\\`` prefix, no trailing separator. ``False`` for an empty or missing side."""
    if not a or not b:
        return False

    def norm(path):
        text = str(path)
        if text.startswith("\\\\?\\"):
            text = text[4:]
        return ntpath.normcase(ntpath.normpath(text)).rstrip("\\")
    return norm(a) == norm(b)


def under(path, root):
    """``path`` is ``root`` itself or inside it (Windows semantics)."""
    if not path or not root:
        return False
    child, parent = ntpath.normcase(ntpath.normpath(str(path))), ntpath.normcase(ntpath.normpath(str(root)))
    return child == parent or child.startswith(parent.rstrip("\\") + "\\")


ADOPTION_ID = re.compile(r"^[0-9a-f]{12}$")


def activation_request(status, body, candidate_id):
    """``(adoption_id or "", facts)`` of an activation POST: accepted only as HTTP 200 with
    ``restarting`` true for exactly ``candidate_id`` and a well-formed adoption id. A 409/500 or any
    other shape is a rejected activation, never something a later status read can repair."""
    body = body if isinstance(body, dict) else {}
    candidate = body.get("candidate") if isinstance(body.get("candidate"), dict) else {}
    adoption = str((candidate.get("activation") or {}).get("adoption_id") or "")
    ok = (status == 200 and body.get("status") == "ok" and body.get("restarting") is True
          and candidate.get("candidate_id") == candidate_id and bool(ADOPTION_ID.match(adoption)))
    return (adoption if ok else ""), {"http": status, "reason": body.get("reason"), "error": body.get("error"),
                                      "adoption_id": adoption, "candidate_id": candidate.get("candidate_id")}


def closed_adoption(rows, adoption_id):
    """The one ``body_adoption_closed`` row of THIS attempt; ``None`` when absent. Rows of other
    attempts (an earlier refusal, a retained Last attempt) never stand in for it."""
    if not ADOPTION_ID.match(str(adoption_id or "")):
        return None
    matches = [row for row in rows if isinstance(row, dict) and row.get("type") == "body_adoption_closed"
               and row.get("adoption_id") == adoption_id]
    return matches[-1] if len(matches) == 1 else None


CLOSED_AS = {"adopted": "adopted", "returned_retry": "returned:environment_selection_failed",
             "returned": "returned:import_probe_failed"}


def settled(record, closed, expected):
    """Whether the candidate record and this attempt's closed handoff agree on ``expected``:
    ``adopted`` (record ``activated``), ``returned_retry`` (record ``ready`` again, the refused rename
    returned everything) or ``returned`` (record ``failed``; a probe failure returned the tree). ``None``
    while either is missing, or while the record still reads ``activating`` (the deadline came before it
    settled) unless this attempt's closed handoff already names another outcome."""
    if expected not in CLOSED_AS:
        raise ValueError(expected)
    if not isinstance(record, dict) or not isinstance(closed, dict):
        return None
    activation = record.get("activation") if isinstance(record.get("activation"), dict) else {}
    outcome, state = activation.get("outcome"), record.get("state")
    decision = activation.get("decision") if isinstance(activation.get("decision"), dict) else {}
    closed_outcome = str(closed.get("outcome") or "")
    if state == "activating":
        return None if closed_outcome == CLOSED_AS[expected] else False
    if expected == "adopted":
        return outcome == "adopted" and state == "activated" and closed_outcome == "adopted"
    if expected == "returned_retry":
        return (outcome == "returned" and state == "ready" and decision.get("retry") is True
                and decision.get("trigger") == "environment_selection_failed"
                and closed_outcome == "returned:environment_selection_failed")
    return (outcome == "returned" and state == "failed" and decision.get("trigger") == "import_probe_failed"
            and closed_outcome == "returned:import_probe_failed")


UNSETTLED = (None, "none", "preparing", "checking")  # candidate states that are not an answer yet


def prepared_ready(view):
    """``prepared_ready`` from the last status view of THIS preparation: ``True`` when it reads ``ready``;
    ``False`` for a measured refusal, another terminal state, or the serving process measured gone; ``None``
    when no answer of this attempt was observed (no reply, still preparing at the deadline, or only an
    earlier attempt's candidate)."""
    if not isinstance(view, dict) or view.get("stale"):
        return None
    if view.get("state") == "ready":
        return True
    if view.get("refusal") or view.get("server_ended") is True or view.get("state") not in UNSETTLED:
        return False
    return None


def generation_identity(value):
    """Is ``value`` the identity preparation marks a generation with (``body_switch.mark_generation`` at
    015ce40b): ``[device, inode, token]``, two integers and a fresh 32-hex token?"""
    return (isinstance(value, list) and len(value) == 3
            and all(isinstance(number, int) and not isinstance(number, bool) for number in value[:2])
            and isinstance(value[2], str) and bool(TOKEN32.match(value[2])))


def prepared_environment(record, kind, current):
    """The candidate prepared its own packages of ``kind`` beside the fixed path ``current``, marked with its
    generation identity (``generation_identity``); ``None`` when the record could not be read (unmeasured, never
    refuted)."""
    if not isinstance(record, dict):
        return None, {"generation": None}
    environment = record.get("environment")
    generation = environment.get("generation") if isinstance(environment, dict) else None
    if not isinstance(generation, dict):
        return False, {"generation": None}
    path = str(generation.get("user_base") or "")
    marked = generation_identity(generation.get("identity"))
    ok = (environment.get("same") is False and generation.get("kind") == kind
          and same_path(generation.get("current"), current) and bool(path) and not same_path(path, current)
          and same_path(ntpath.dirname(path), ntpath.dirname(str(current))) and marked)
    return ok, {"generation": generation, "same": environment.get("same"), "identity": generation.get("identity"),
                "identity_marked": marked}


HEX40 = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
TOKEN32 = re.compile(r"^[0-9a-f]{32}$")


def layers_measured(value):
    """``""`` when ``value`` is a complete reading of the layers (``wq_product.layers``): a 40-hex HEAD, the
    branch read, index and status digests and a digest for EVERY sentinel asked for; else why it is not."""
    if not isinstance(value, dict):
        return "no reading"
    if "error" in value:
        return f"read failed: {value['error']}"
    files = value.get("files")
    if not HEX40.match(str(value.get("head") or "")) or not isinstance(value.get("branch"), str) \
            or not DIGEST.match(str(value.get("index") or "")) or not DIGEST.match(str(value.get("status") or "")) \
            or not isinstance(files, dict) or not all(DIGEST.match(str(digest or "")) for digest in files.values()):
        return "incomplete reading"
    return ""


def layers_equal(before, after, keys=("head", "branch", "index", "status", "files")):
    """``(same, facts)``: serving checkout layers unchanged (H, branch, I, W/U status and the sentinel bytes).
    ``same`` is ``None`` (unmeasured) unless both readings are complete; never a bare ``None``."""
    gaps = {side: why for side, why in (("before", layers_measured(before)), ("after", layers_measured(after))) if why}
    if gaps:
        return None, {"unmeasured": gaps}
    changed = [key for key in keys if before.get(key) != after.get(key)]
    return not changed, {"changed": changed}


def witness_for(witnesses, identity, where=None, since_ns=0):
    """The newest witness written by exactly that process (pid AND birth) after ``since_ns``."""
    if not isinstance(identity, dict):
        return None
    rows = [row for row in witnesses if isinstance(row, dict) and row.get("pid") == identity.get("pid")
            and str(row.get("birth")) == str(identity.get("birth")) and int(row.get("time_ns") or 0) > since_ns
            and (where is None or row.get("where") == where)]
    return max(rows, key=lambda row: int(row.get("time_ns") or 0)) if rows else None


def generation_seen(witness, version, site_root, prefix=None):
    """A witness that imported fixture ``version`` from inside ``site_root`` (and runs with ``prefix``)."""
    if not isinstance(witness, dict):
        return None
    package = witness.get("package") if isinstance(witness.get("package"), dict) else {}
    return (package.get("version") == version and under(package.get("realpath"), site_root)
            and (prefix is None or same_path(witness.get("prefix"), prefix)))


def dumps(value):
    return json.dumps(value, sort_keys=True, default=str)
