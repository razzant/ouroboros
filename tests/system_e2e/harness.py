"""SystemHarness — the Ф0 skeleton of the v7next deep-integration suite (plan §8).

Not a test module (pytest collects ``test_*.py`` only) — this is the machinery the
``tests/system_e2e/test_*`` scenario modules drive: a KEYLESS isolated real-server
stack (roast F21), a scriptable loopback stub model whose review-organ branch sits
BEFORE the finalization-turn check (roast F22 / plan §8), and readers for the durable
artifacts every scenario asserts against. The direct precedent is
``tests/fixtures_e2e_cancellation.py`` on the ``ouroboros_v7_wip`` reference branch
(same split, same stub idiom, same 0600 settings write); this file generalizes it from
the cancellation protocol to the whole system surface and hardens the egress story.

KEYLESS LANE CONTRACT (F21). The mock lane must be structurally unable to spend money
or leak an operator credential into a child the scenarios do not control:

* the isolated ``settings.json`` is built from scratch (never copied from live
  settings), pins EVERY model-slot key the tree declares
  (``provider_models.ACTIVE_MODEL_SETTING_KEYS`` + legacy) so a new upstream slot is
  pinned by construction, and carries exactly one "credential" — the loopback stub's
  non-secret placeholder pair;
* ``KeylessIsolatedServer`` strips every provider credential the tree knows about
  (``server_runner._PROVIDER_ENV_KEYS`` ∪ ``provider_models.ALL_PROVIDER_CREDENTIAL_KEYS``)
  plus all proxy variables from the child environment, ON TOP of the base
  ``IsolatedServer`` sanitization — the base ``_is_secret_env_key`` deliberately
  EXEMPTS provider keys (benchmark servers need them), which for this lane is exactly
  the ANTHROPIC_API_KEY hole the plan names;
* an un-pinned slot therefore routes to a slug whose provider has no credential and
  fails loudly instead of silently reaching a paid provider.

Full egress interception (socket-level deny + evidence) is Ф4 scope, not Ф0.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.benchmarks.common.server_runner import (  # noqa: E402
    IsolatedServer,
    _PROVIDER_ENV_KEYS,
    _is_secret_env_key,
    supervisor_state_is_ready,  # noqa: F401 (re-export: THE readiness contract)
)
from ouroboros.provider_models import (  # noqa: E402
    ACTIVE_MODEL_SETTING_KEYS,
    ALL_PROVIDER_CREDENTIAL_KEYS,
    LEGACY_MODEL_SETTING_KEYS,
)
from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS  # noqa: E402

LANE_MOCK = "mock"

# The scenario inventory of this suite (plan §8) — the manifest is DATA and the pin
# is a gen/verify pair: an id losing its test is red (a scenario must be retired
# deliberately, not by deletion), and a NEW ``test_s<N>_*`` test without a manifest
# row is red too (an undeclared scenario is invisible to the lane budget).
# Scenarios land WITH their phases (roast F22); Ф0 carried S1-S2, Ф4 lane 1 adds the
# first mandatory surfaces (plan §8: boot/identity/WS, egress hardening first,
# typed tools + safety, cost-truth).
SCENARIOS = {
    "S1": ("boot / identity / WS chat / port-file / task contract smoke", LANE_MOCK),
    "S2": ("review-organ smoke: commit_reviewed triad+scope on a doc-only diff", LANE_MOCK),
    "S3": ("egress hardening: poisoned parent credentials never reach the server tree", LANE_MOCK),
    "S4": ("typed tools + safety: protected-path denial has zero side effects", LANE_MOCK),
    "S5": ("cost-truth (ABI-3): public task projections carry honest-only cost names", LANE_MOCK),
    # Ф4 wave 2 (plan §8: subagent-дерево, cancellation, managed update core).
    "S6": ("subagent tree: lineage truth, wait_tasks quiescence, child-result handoff, root cost rollup", LANE_MOCK),
    "S7": ("cancellation: live task -> typed cancelled terminal, owed answer, honest cost, drained intents", LANE_MOCK),
    "S8": ("cancellation cascade: parent+child torn down with no orphan processes in the live tree", LANE_MOCK),
    "S9": ("managed update ff on a local managed repo: dirty-work stash insurance + honest boot-finalize", LANE_MOCK),
    "S10": ("managed update rollback: typed future/null-marker refusals + byte-for-byte tree restore", LANE_MOCK),
    # Ф4 wave 3b (plan §8: delegated transport + skills lifecycle).
    "S11": ("delegated transport: full nanny run over FakeClaudexorDaemon, wire/custody truth, typed refusals", LANE_MOCK),
    "S12": ("delegated no-orphans: SIGKILL mid-run -> restart -> boot custody sweep settles the run, one physical attempt", LANE_MOCK),
    # S13 carries TWO tests: the full lifecycle across a restart, and the
    # hot-adoption variant that removes the restart (W3B-F1) — enable after boot
    # must reach the workers already running, not only the pool's next generation.
    "S13": ("skills lifecycle E2E: payload -> stub review -> grants -> enable -> dispatch -> disable/delete + Model Experience; hot-adoption variant: enable after boot dispatches in an ALREADY-SPAWNED worker with no restart", LANE_MOCK),
    # Ф4 wave 3a (plan §8: plan review; commit triad+scope BOTH enforcement
    # classes + stale-rejection; acceptance loop). Renumbered S14-S17 at
    # integration: the parallel wave-3b lane claimed S11-S13 first.
    "S14": ("plan review: scripted REVISE->ACCEPT cycle, honest durable chronicle, cycle-cap refusal", LANE_MOCK),
    "S15": ("commit triad+scope, ADVISORY class: red verdicts recorded + waved through with durable override, commit lands", LANE_MOCK),
    "S16": ("commit triad+scope, BLOCKING class: red blocks (HEAD unmoved), identical resubmit refused free, green lands; freshness stale-rejection", LANE_MOCK),
    "S17": ("acceptance loop (required+blocking): reject -> rework -> accept; paid-identity / free-replay invariants", LANE_MOCK),
    # Ф4 wave 4 (plan §8 remainder: update variations, chat-lineage cancel,
    # absorb kill-recovery, delegated interactive answer).
    "S18": ("managed update carrier path: diverged fork, span-confined VERSION conflict auto-resolved, carriers transferred to the official version", LANE_MOCK),
    "S19": ("managed update conflicting: typed budget refusal at the assisted gate, tree byte-identical, dirty work restored", LANE_MOCK),
    "S20": ("managed update crash mid-apply: boot-finalize honesty across stash-crash / half-written tx / applied-but-unrestarted", LANE_MOCK),
    "S21": ("cancellation with chat lineage: outbox-delivered receipt, chat.jsonl row, cancel_receipt block, intent forensics", LANE_MOCK),
    "S22": ("evolution absorb kill-recovery: SIGKILL after the reviewed commit, markerless boot reconcile absorbs once, never twice", LANE_MOCK),
    "S23": ("delegated interactive answer: waiting_on_user -> delegate_answer -> run continues; wire + custody truth", LANE_MOCK),
    # Ф4 wave 5: the MUTATING delegated runs the earlier waves carried and never
    # landed (ADOPTION row DEFER-E2E-DELEG-MUT). The one delegation branch that
    # changes the owner's tree on behalf of an external harness.
    "S24": ("delegated MUTATING run, clean pull-in: private snapshot provisioned, harness edits it, containment facts read from the attempt record, integrate_delegated_patch(apply) stages into the live workspace — which is untouched until that call", LANE_MOCK),
    "S25": ("delegated MUTATING run, conflicting pull-in: the live tree drifts on a patched path, apply is REFUSED typed, and snapshot + patch survive as the nanny's own resolution material", LANE_MOCK),
    # v7 follow-up Ф1 (sprint plan §5.1 Ф1-B, D-05 chat-turn addressability): the
    # in-process DIRECT-CHAT turn — no queue row, no worker process — stopped by
    # the owner through the same cancel endpoint the UI drives, caught MID-ROUND
    # on an event-gated model hold (ModelGate), never a timed race.
    "S26": ("direct-chat owner stop: an in-flight direct turn is addressable (running list + activity snapshot), stop-now mid-round answers the typed 'still live' with the cooperative control armed ONCE (a repeat is idempotent), the turn ends at its next step with ZERO further model rounds under the owner-stop reason, the chat concludes, custody settles already_settled against the turn's own terminal, and a later stop is the typed 404", LANE_MOCK),
}

MOCK_SLUG = "openai-compatible::mock-model"

# ---------------------------------------------------------------------------
# Prompt markers the stub classifies review-organ calls by (roast F22).
#
# These are VERBATIM literals from the tree under test and WILL drift with upstream:
#   REVIEWER_SLOT_MARKER   — ouroboros/review_execution.py::_render_prompt_parts
#   ACCEPTANCE_KEYS_MARKER — same function, the task_acceptance criteria_used key list
#   TRIAD_USER_MARKER      — ouroboros/tools/review.py::_dispatch_unified_review
#   SCOPE_USER_MARKER      — ouroboros/tools/scope_review.py::_call_scope_llm
#   PLAN_REVIEW_MARKER     — ouroboros/tools/plan_packet.py::build_plan_review_system_prompt
#   NATIVE_EPISODE_MARKER  — ouroboros/review_native_episode.py::episode_prompt
# The default-lane marker-pin test greps them out of the source files so drift is a
# named test failure, not a silently mute stub.
# ---------------------------------------------------------------------------
REVIEWER_SLOT_MARKER = "You are an independent Ouroboros reviewer slot."
ACCEPTANCE_KEYS_MARKER = "criteria_used (the acceptance criteria you re-derived"
TRIAD_USER_MARKER = "Review the staged diff and context provided in the instructions above."
SCOPE_USER_MARKER = "Review the staged change and context above. Output ONLY a JSON array."
SKILL_REVIEW_MARKER = "You are performing a SKILL review, not a repo-commit review."
PLAN_REVIEW_MARKER = (
    "You are one independent reviewer of an INTENTION — a plan spec — "
    "before the work starts."
)
# Deliberately the FIRST source line of the two-literal prompt head only: the
# marker-pin test greps the SOURCE file, where the concatenated sentence is
# split across adjacent string literals.
NATIVE_EPISODE_MARKER = (
    "You are an independent Ouroboros reviewer slot running a bounded"
)
FINALIZATION_MARKERS = ("[OWNER_STOP]", "[FINALIZE_NOW]")

# The native inspection episode names its surface on a dedicated prompt line
# (review_native_episode.episode_prompt); the stub parses it so an advisory
# pre-review episode and any future native surface classify by NAME.
_SURFACE_LINE_RE = re.compile(r"^Surface: ([A-Za-z_]+)$", re.MULTILINE)

MARKER_SOURCES = {
    REVIEWER_SLOT_MARKER: "ouroboros/review_execution.py",
    ACCEPTANCE_KEYS_MARKER: "ouroboros/review_execution.py",
    TRIAD_USER_MARKER: "ouroboros/tools/review_multi_model.py",   # TRIAD_USER_TURN: the one literal the send and the admission share
    SCOPE_USER_MARKER: "ouroboros/tools/scope_review.py",
    SKILL_REVIEW_MARKER: "ouroboros/skill_review_prompt.py",
    PLAN_REVIEW_MARKER: "ouroboros/tools/plan_packet.py",
    NATIVE_EPISODE_MARKER: "ouroboros/review_native_episode.py",
}


# ---------------------------------------------------------------------------
# Opt-in gate
# ---------------------------------------------------------------------------

def lane_enabled(lane: str) -> bool:
    selected = str(os.environ.get("OUROBOROS_E2E_DEEP") or "").strip().lower()
    return selected == lane


def require_lane(lane: str) -> None:
    if not lane_enabled(lane):
        pytest.skip(
            f"set OUROBOROS_E2E_DEEP={lane} to run the {lane} deep-integration lane "
            "(spawns a real isolated server; see tests/system_e2e/)"
        )


# ---------------------------------------------------------------------------
# Message flattening: review prompts arrive as block lists (cached_prompt_blocks),
# agent-loop prompts as plain strings — marker checks must see both.
# ---------------------------------------------------------------------------

def message_text(message) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict)
        )
    return ""


def body_text(body: dict) -> str:
    return "\n".join(message_text(m) for m in (body.get("messages") or []))


def classify_call(body: dict) -> str:
    """Name the branch a chat-completion body belongs to.

    Returns one of: ``safety``, ``skill_review``, ``scope_review``, ``triad_review``,
    ``acceptance``, ``reviewer_slot``, ``plan_review``, ``advisory_review``,
    ``native_episode``, ``finalization``, ``agent``. ORDER MATTERS (roast F22):
    every review-organ branch is checked BEFORE the finalization-turn check,
    because a review packet may quote a transcript that itself contains a
    finalization marker — a stub that answered such a packet with a final chat
    answer would silently break the review organ mid-scenario. The native
    inspection episode (a TOOL-BEARING review call — review_native_episode)
    must also be classified here, or it would fall through to the agent branch
    and eat a scenario's script steps.
    """
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_object":
        return "safety"
    user_tail = "\n".join(
        message_text(m) for m in (body.get("messages") or [])
        if isinstance(m, dict) and m.get("role") == "user"
    )
    full = body_text(body)
    # Skill review FIRST among the marker branches: its pack embeds whole
    # governance docs and the payload under review, either of which could quote
    # another branch's marker; its own opening sentence is the most specific.
    if SKILL_REVIEW_MARKER in full:
        return "skill_review"
    # Scope before triad: both user messages start with "Review the staged".
    if SCOPE_USER_MARKER in user_tail:
        return "scope_review"
    if TRIAD_USER_MARKER in user_tail:
        return "triad_review"
    if PLAN_REVIEW_MARKER in full:
        return "plan_review"
    if NATIVE_EPISODE_MARKER in full:
        match = _SURFACE_LINE_RE.search(full)
        surface = match.group(1) if match else ""
        return "advisory_review" if surface == "advisory_review" else "native_episode"
    if REVIEWER_SLOT_MARKER in full:
        return "acceptance" if ACCEPTANCE_KEYS_MARKER in full else "reviewer_slot"
    if any(marker in full for marker in FINALIZATION_MARKERS):
        return "finalization"
    return "agent"


# Every kind the review-organ/safety branch owns: calls of these kinds never
# consume an agent script step or a ReplayModel fixture row, and a scenario's
# ReviewScript may override their canned answers.
REVIEW_KINDS = frozenset({
    "safety", "scope_review", "triad_review", "acceptance", "reviewer_slot",
    "plan_review", "advisory_review", "native_episode",
})


# ---------------------------------------------------------------------------
# Canned review-organ verdicts (all-clean). Shapes come from the tree's own parsers:
# triad — triad_review.REVIEW_JSON_ARRAY_CONTRACT ([] + NO_FINDINGS sentinel);
# scope — scope_review_contract.normalize_scope_items (required matrix, PASS reasons
# must be non-terse); reviewer slot — review_execution's "Return JSON with keys" list.
# ---------------------------------------------------------------------------

TRIAD_CLEAN_TEXT = "[]\nNO_FINDINGS"


def canned_review_answer(kind: str) -> dict | None:
    """The canned assistant message for a review-organ/safety ``kind``, else None.

    ONE derivation shared by ``ScriptedStubModel`` (via ``scripted_completion``) and
    ``ReplayModel``: both models must answer the review organ identically, and the
    review-organ branch must sit BEFORE any script/fixture consultation (roast F22).
    """
    if kind == "safety":
        return {"role": "assistant",
                "content": json.dumps({"status": "SAFE", "reason": "stub"})}
    if kind == "skill_review":
        return {"role": "assistant", "content": skill_review_clean_text()}
    if kind == "scope_review":
        return {"role": "assistant", "content": scope_clean_text()}
    if kind == "triad_review":
        return {"role": "assistant", "content": TRIAD_CLEAN_TEXT}
    if kind in ("acceptance", "reviewer_slot"):
        return {"role": "assistant", "content": reviewer_slot_clean_text(kind)}
    if kind in ("plan_review", "advisory_review", "native_episode"):
        # One shared clean shape: the plan-review findings contract
        # (plan_spec.parse_findings) and the advisory episode's clean predicate
        # (triad_review.empty_array_is_verified_clean) both accept the bare
        # empty array + NO_FINDINGS sentinel as a verified-clean verdict.
        return {"role": "assistant", "content": TRIAD_CLEAN_TEXT}
    return None


def skill_review_clean_text() -> str:
    """All-PASS skill-review verdict over the tree's OWN checklist items.

    The item list is imported from the prompt module (its ``_SKILL_REVIEW_ITEMS``
    is both the prompt's expected-items contract and the parser's required set),
    so an upstream checklist change re-derives the canned verdict instead of
    silently failing coverage. Lazy import: the skill-review module tree is heavy
    and only skill scenarios pay for it.
    """
    from ouroboros.skill_review_prompt import _SKILL_REVIEW_ITEMS

    return json.dumps([
        {
            "item": item,
            "verdict": "PASS",
            "severity": "advisory",
            "reason": "Stub skill reviewer: checked and clean for the scripted E2E payload.",
        }
        for item in _SKILL_REVIEW_ITEMS
    ])


def scope_clean_text() -> str:
    return json.dumps([
        {
            "item": item,
            "verdict": "PASS",
            "severity": "advisory",
            "reason": "Stub scope reviewer: checked and clean for this scripted smoke diff.",
        }
        for item in sorted(SCOPE_REQUIRED_ITEMS)
    ])


def reviewer_slot_clean_text(kind: str) -> str:
    verdict = {"verdict": "PASS", "findings": [], "summary": "stub reviewer slot: clean."}
    if kind == "acceptance":
        verdict["outcome_tier"] = "solved"
        verdict["dialogue_status"] = "continue_actionable"
        verdict["criteria_used"] = []
    return json.dumps(verdict)


class ReviewScript:
    """Ordered per-kind verdict queues for review-organ calls (wave 3a).

    The wave-1 stub answers every review-organ call with the canned ALL-CLEAN
    verdict; the review-surface scenarios need the organ to say something else
    first (a REVISE plan finding, a critical triad FAIL, an acceptance reject)
    and only then converge. A ReviewScript maps ``kind`` (a ``REVIEW_KINDS``
    member) to an ORDERED queue of verdicts; each review call of that kind
    consumes one entry, and an exhausted or absent queue falls back to the
    canned all-clean answer — so a scenario scripts exactly the red rounds it
    means to and the organ converges by construction.

    Entry forms: ``str`` (assistant content), ``dict`` (full assistant
    message), or ``callable(body) -> str | dict | None`` — the same
    dynamic-argument contract script steps use (a hook may also carry a
    scenario side effect, e.g. mutating the staged tree to prove the
    post-verdict freshness gate; ``None`` falls back to canned).
    ``served`` records ``(kind, message)`` in consumption order;
    ``assert_consumed()`` is the integrity gate a scripted-review scenario must
    end with (an unserved red verdict = the organ never ran = red).
    """

    def __init__(self, steps: dict) -> None:
        unknown = sorted(set(steps or {}) - REVIEW_KINDS)
        if unknown:
            raise ValueError(f"ReviewScript kinds must be review-organ kinds: {unknown}")
        self.steps = {kind: list(queue) for kind, queue in (steps or {}).items()}
        self.served: list = []

    def __call__(self, kind: str, body: dict):
        queue = self.steps.get(kind)
        if not queue:
            return None
        step = queue.pop(0)
        if callable(step):
            step = step(body)
        if step is None:
            return None
        message = ({"role": "assistant", "content": str(step)}
                   if isinstance(step, str) else dict(step))
        self.served.append((kind, message))
        return message

    def consumed(self) -> bool:
        return not any(self.steps.values())

    def assert_consumed(self) -> None:
        leftover = {kind: len(queue) for kind, queue in self.steps.items() if queue}
        assert not leftover, f"ReviewScript verdicts never served: {leftover}"


def scripted_completion(body: dict, seq: int, script_next, final_answer: str,
                        review_next=None) -> tuple[str, dict]:
    """The stub's whole decision function, pure so the default lane can pin it.

    ``script_next`` is a callable returning the next scripted tool step (or None when
    the script is exhausted); it is only consulted on plain agent turns. A step may
    itself be a CALLABLE ``step(body) -> step-dict`` — the wave-2 dynamic-argument
    contract: a scenario cannot know a server-minted child task id statically, so the
    step derives its arguments from the prompt the server actually sent (e.g. parse
    the ``schedule_subagent`` receipt out of the transcript to build ``wait_tasks``
    arguments). ``review_next(kind, body)`` (wave 3a, usually a ``ReviewScript``)
    may override the canned answer of a review-organ call; it is consulted ONLY for
    ``REVIEW_KINDS`` — the review-organ branch still sits BEFORE the finalization
    check and review calls still never consume agent script steps. Returns
    ``(kind, message)`` where message is the OpenAI-style assistant message.
    """
    kind = classify_call(body)
    if review_next is not None and kind in REVIEW_KINDS:
        scripted = review_next(kind, body)
        if scripted is not None:
            return kind, scripted
    canned = canned_review_answer(kind)
    if canned is not None:
        return kind, canned
    if kind == "finalization":
        return kind, {"role": "assistant", "content": final_answer}
    step = script_next(body) if body.get("tools") else None
    if callable(step):
        step = step(body)
    if step is None:
        return "final", {"role": "assistant", "content": final_answer}
    if "final" in step:
        return "final", {"role": "assistant", "content": str(step["final"])}
    call = {"name": str(step["tool"]),
            "arguments": json.dumps(step.get("arguments") or {})}
    return "agent", {
        "role": "assistant", "content": "still working",
        "tool_calls": [{"id": f"call_{seq}", "type": "function", "function": call}],
    }


class ModelGate:
    """An event-gated HOLD on the loopback model: "the model is still thinking".

    An owner-control scenario needs a turn that is provably INSIDE a model round
    when the owner acts, and that stays there until the scenario says otherwise —
    a stub answering in microseconds cannot be caught mid-round, and a latency
    sleep would only turn the catch into a race. ``match(body)`` selects the
    call to hold: the FIRST match sets ``arrived`` (the scenario's proof that the
    round is in flight) and blocks that request thread until the scenario sets
    ``release`` — bounded by ``timeout``: expiry sets ``timed_out`` and raises
    ``TimeoutError`` in the request thread (the held call fails loudly), so a
    broken scenario fails by name instead of hanging the lane or silently
    resuming the round. Every later match passes through untouched; ``matched``
    counts them all (the scenario's "how many rounds did this turn get" fact)
    and ``held`` stays at one. The hold runs OUTSIDE the model's call lock (see
    ``LoopbackModelServer.do_POST``), so unrelated calls and the scenario's own
    readers (``kinds()``) never wait behind it.
    """

    def __init__(self, match, *, timeout: float = 180.0) -> None:
        self.match = match
        self.timeout = float(timeout)
        self.arrived = threading.Event()
        self.release = threading.Event()
        self.matched = 0
        self.held = 0
        self.timed_out = False
        self._lock = threading.Lock()

    def __call__(self, body: dict) -> None:
        if not self.match(body):
            return
        with self._lock:
            self.matched += 1
            if self.held:
                return
            self.held += 1
        self.arrived.set()
        if not self.release.wait(self.timeout):
            self.timed_out = True
            raise TimeoutError(
                f"ModelGate: the scenario never released the held round within {self.timeout:.0f}s")


class LoopbackModelServer:
    """Shared HTTP plumbing of the loopback OpenAI-compatible model servers.

    Subclasses implement ``_answer(body, seq) -> (kind, message)``; this base owns the
    socket, the /models capability answer, the call ledger and the completion
    envelope. One base, two models (``ScriptedStubModel`` / ``ReplayModel``), so the
    wire shape and the window evidence can never drift between them.
    """

    def __init__(self, *, latency_sec: float = 0.0, gate: "ModelGate | None" = None) -> None:
        self.latency_sec = latency_sec
        self.gate = gate
        self.calls: list = []          # (kind, body) in arrival order
        self._lock = threading.Lock()
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib callback name
                if self.path.rstrip("/").endswith("/models"):
                    # >=1M ON PURPOSE, and it is load-bearing twice: the capability-
                    # evidence /models probe stores this as a CONFIRMED window, which
                    # (a) sizes the triad fit budget (a 400K window under the cold
                    # 1.65 density floor caps input at ~202K — BELOW the ~226K
                    # governance pack, blocking every commit_reviewed before
                    # dispatch), and (b) satisfies the BIBLE P3 >=1M floor that
                    # scope review's BLOCKING authority requires.
                    return self._send({"data": [
                        {"id": model_id, "max_model_len": 2_000_000}
                        for model_id in outer._model_ids()
                    ]})
                self.send_error(404)

            def do_POST(self):  # noqa: N802 - stdlib callback name
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads((self.rfile.read(length) or b"{}").decode("utf-8"))
                except ValueError:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                if outer.gate is not None:
                    # Before the call lock on purpose: a held round must not
                    # block the model's other callers or the scenario's readers.
                    outer.gate(body)
                if outer.latency_sec:
                    time.sleep(outer.latency_sec)
                return self._send(outer._completion(body))

            def _send(self, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _model_ids(self) -> list[str]:
        return ["mock-model"]

    def _answer(self, body: dict, seq: int) -> tuple[str, dict]:
        raise NotImplementedError

    def _completion(self, body: dict) -> dict:
        with self._lock:
            seq = len(self.calls) + 1
            kind, message = self._answer(body, seq)
            self.calls.append((kind, body))
        return {
            "id": f"stub-{seq}",
            "object": "chat.completion",
            "model": str(body.get("model") or "mock-model"),
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def kinds(self) -> list[str]:
        with self._lock:
            return [kind for kind, _ in self.calls]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()


class ScriptedStubModel(LoopbackModelServer):
    """Keep-alive OpenAI-compatible stub model with an ordered per-scenario script.

    Extends the ``StubModelServer`` idiom of the cancellation harness: instead of one
    fixed keepalive tool, a scenario hands the stub an ORDERED list of tool steps
    (``{"tool": name, "arguments": {...}}``); each plain agent turn consumes one step,
    and an exhausted script yields the tool-less final answer. Review-organ calls
    (triad / scope / reviewer-slot / acceptance) NEVER consume script steps — they are
    classified by prompt markers and answered with canned all-clean verdicts, and that
    classification runs BEFORE the finalization-turn check (roast F22). Safety
    supervisor calls (json_object response_format) always get a SAFE verdict.

    Every call is recorded as ``(kind, body)`` in ``self.calls``; ``self.kinds()``
    gives the observed branch sequence a scenario asserts against.
    """

    def __init__(self, script=None, *, final_answer: str = "Final answer: scripted scenario complete.",
                 latency_sec: float = 0.0, review_script: "ReviewScript | None" = None,
                 gate: "ModelGate | None" = None) -> None:
        super().__init__(latency_sec=latency_sec, gate=gate)
        self.script = list(script or [])
        self.final_answer = final_answer
        self.review_script = review_script
        self._script_index = 0

    def _next_step(self, _body) -> dict | None:
        if self._script_index >= len(self.script):
            return None
        step = self.script[self._script_index]
        self._script_index += 1
        return step

    def _answer(self, body: dict, seq: int) -> tuple[str, dict]:
        return scripted_completion(body, seq, self._next_step, self.final_answer,
                                   review_next=self.review_script)

    def script_consumed(self) -> bool:
        with self._lock:
            return self._script_index >= len(self.script)


# ---------------------------------------------------------------------------
# ReplayModel — deterministic (lineage, slot, attempt)-bound fixtures (plan §8).
# ---------------------------------------------------------------------------

# Scenario prompts plant the lineage tag in the text the model sees (a task
# description flows into the agent prompt verbatim); the LAST occurrence wins so a
# child task's own tag beats a parent transcript quoting it.
LINEAGE_TAG_RE = re.compile(r"\[E2E-LINEAGE:([A-Za-z0-9_.-]+)\]")
REPLAY_ROOT_LINEAGE = "root"


def default_lineage_binder(body: dict) -> str:
    matches = LINEAGE_TAG_RE.findall(body_text(body))
    return matches[-1] if matches else REPLAY_ROOT_LINEAGE


def default_slot_binder(body: dict) -> str:
    """The slot identity is the requested model id — scenarios pin DISTINCT stub
    slugs per model slot in settings, so the wire's ``model`` field names the slot
    that made the call (no guessing from prompt shape)."""
    return str(body.get("model") or "")


class ReplayModel(LoopbackModelServer):
    """Loopback model whose every non-review answer is BOUND to (lineage, slot, attempt).

    The fixture is a mapping ``{(lineage, slot, attempt): step}`` where ``attempt`` is
    the 1-based ordinal of the fixture-consulted call for that (lineage, slot) pair,
    and ``step`` is one of::

        {"tool": name, "arguments": {...}}   # scripted tool call
        {"final": "answer text"}             # tool-less final answer
        {"message": {...}}                   # raw OpenAI-style assistant message
        callable(body) -> one of the above   # dynamic step (server-minted ids)

    A CALLABLE row is the wave-2 dynamic-argument contract shared with the scripted
    stub: it receives the request body and returns the concrete step, so a fixture
    can reference values only the server mints at runtime (a child task id, an exact
    result hash) by parsing them out of the transcript the model was actually shown.

    Review-organ and safety calls are answered canned (same branch order as the
    scripted stub, review BEFORE finalization — roast F22) and NEVER consume the
    fixture or advance attempt counters. A call with no fixture row is a MISS: it is
    recorded and answered with a loud text (so the server under test cannot hang),
    and ``assert_consumed()`` — which every ReplayModel scenario must call — fails on
    ANY miss and on ANY unconsumed fixture row (недоеденная фикстура = красный).

    ``model_ids`` overrides the /models advertisement for scenarios whose
    ``slot_binder`` returns compound slot names that are NOT wire model ids (the
    default derivation would then advertise garbage and omit the real ids the
    capability-evidence window probe needs).
    """

    def __init__(self, fixture, *, lineage_binder=None, slot_binder=None,
                 latency_sec: float = 0.0, model_ids=None) -> None:
        super().__init__(latency_sec=latency_sec)
        self.fixture = dict(fixture or {})
        for key in self.fixture:
            if not (isinstance(key, tuple) and len(key) == 3 and isinstance(key[2], int)):
                raise ValueError(f"ReplayModel fixture key must be (lineage, slot, attempt): {key!r}")
        self._lineage_binder = lineage_binder or default_lineage_binder
        self._slot_binder = slot_binder or default_slot_binder
        self._explicit_model_ids = [str(m) for m in model_ids] if model_ids else None
        self._attempts: dict = {}      # (lineage, slot) -> calls consulted so far
        self.consumed: list = []       # keys served, in order
        self.misses: list = []         # keys asked for but absent from the fixture

    def _model_ids(self) -> list[str]:
        if self._explicit_model_ids is not None:
            return sorted(set(self._explicit_model_ids) | {"mock-model"})
        # Advertise every slot slug the fixture names (plus the default), so the
        # capability-evidence /models probe confirms a window for each slot route.
        ids = {"mock-model"} | {str(slot) for _, slot, _ in self.fixture}
        return sorted(ids)

    def _answer(self, body: dict, seq: int) -> tuple[str, dict]:
        kind = classify_call(body)
        canned = canned_review_answer(kind)
        if canned is not None:
            return kind, canned
        lineage = str(self._lineage_binder(body))
        slot = str(self._slot_binder(body))
        attempt = self._attempts.get((lineage, slot), 0) + 1
        self._attempts[(lineage, slot)] = attempt
        key = (lineage, slot, attempt)
        step = self.fixture.get(key)
        if step is None:
            self.misses.append(key)
            return "replay_miss", {
                "role": "assistant",
                "content": f"REPLAY_MISS: no fixture row for {key!r} — the scenario "
                           "fixture and the server's call pattern disagree.",
            }
        self.consumed.append(key)
        if callable(step):
            step = step(body)
        if "message" in step:
            return "replay", dict(step["message"])
        if "final" in step:
            return "replay_final", {"role": "assistant", "content": str(step["final"])}
        call = {"name": str(step["tool"]),
                "arguments": json.dumps(step.get("arguments") or {})}
        return "replay", {
            "role": "assistant", "content": "still working",
            "tool_calls": [{"id": f"call_{seq}", "type": "function", "function": call}],
        }

    def assert_consumed(self) -> None:
        """The fixture-integrity gate every ReplayModel scenario must end with."""
        with self._lock:
            leftover = sorted(set(self.fixture) - set(self.consumed))
            misses = list(self.misses)
        problems = []
        if misses:
            problems.append(f"missed keys (no fixture row): {misses}")
        if leftover:
            problems.append(f"unconsumed fixture rows: {leftover}")
        assert not problems, "ReplayModel fixture mismatch: " + "; ".join(problems)


# ---------------------------------------------------------------------------
# Egress-hardening probes (Ф4: the mock lane must PROVE, not assume, that no
# provider credential reaches the child server process).
# ---------------------------------------------------------------------------

# Every way the runtime tree reads an environment variable by literal name.
_ENV_READ_RE = re.compile(
    r"""os\.(?:environ(?:\.get)?[\[(]|getenv\()\s*["']([A-Z0-9_]+)["']"""
)
_CREDENTIAL_SHAPE_RE = re.compile(r"(API_KEY|CREDENTIALS|TOKEN|SECRET|PASSWORD)")
RUNTIME_TREE_GLOBS = ("ouroboros/**/*.py", "supervisor/**/*.py", "server.py")


def runtime_credential_env_key_reads() -> set:
    """Every credential-shaped env key the RUNTIME TREE actually reads.

    Built from the source (not from a hand-kept list), so a provider credential
    added upstream tomorrow lands in the strip-coverage pin automatically instead of
    silently reaching a keyless child. Includes reads through ``os.environ[...]``,
    ``os.environ.get`` and ``os.getenv``.
    """
    keys: set = set()
    for pattern in RUNTIME_TREE_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for name in _ENV_READ_RE.findall(source):
                if _CREDENTIAL_SHAPE_RE.search(name):
                    keys.add(name)
    return keys


def proc_environ(pid: int) -> dict:
    """The live environment of a running process (Linux /proc; POSIX-only probe)."""
    raw = pathlib.Path(f"/proc/{pid}/environ").read_bytes()
    env: dict = {}
    for chunk in raw.split(b"\x00"):
        if b"=" in chunk:
            key, _, value = chunk.partition(b"=")
            env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def process_tree_pids(root_pid: int) -> list:
    """``root_pid`` plus every live descendant (Linux /proc children walk)."""
    pids, queue = [], [int(root_pid)]
    while queue:
        pid = queue.pop()
        pids.append(pid)
        for children_file in pathlib.Path(f"/proc/{pid}/task").glob("*/children"):
            try:
                queue.extend(int(child) for child in children_file.read_text().split())
            except (OSError, ValueError):
                continue
    return pids


def _read_proc_environ_bytes(pid) -> bytes:
    """Raw ``/proc/<pid>/environ`` bytes, or ``b""`` when it cannot be read.

    The single read seam of both /proc environ oracles below, so the empty-window
    behaviour they must survive can be pinned deterministically.
    """
    try:
        return pathlib.Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return b""


def wait_pid_env_value(pid: int, value: str, timeout: float = 10.0) -> bool:
    """True once ``/proc/<pid>/environ`` carries *value*, within a bounded window.

    THE POSITIVE ORACLE. ``Popen`` returns as soon as the exec SUCCEEDED — the
    CLOEXEC error pipe closes inside ``execve`` — but the kernel publishes the NEW
    image's ``env_start``/``env_end`` later in that same exec path, so a read that
    lands in that window sees an EMPTY environ for a live, correctly marked child.
    A positive claim ("this child carries the marker") must therefore poll THE ONE
    pid until its environ becomes readable or a deadline passes; scanning all of
    /proc again would only re-roll the same window against a moving target.

    The negative oracle (``pids_with_env_value`` as the no-orphans postcondition)
    deliberately keeps its SINGLE scan: an orphan that survived a teardown was
    execed long before the scan, so the post-exec window cannot hide it, while a
    wait there would only slow every clean teardown down.
    """
    needle = str(value).encode()
    deadline = time.monotonic() + float(timeout)
    while True:
        if needle in _read_proc_environ_bytes(pid):
            return True
        if not pathlib.Path(f"/proc/{pid}").exists():
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def pids_with_env_value(value: str) -> list:
    """Every live pid whose /proc environ carries *value* (readable procs only).

    The no-orphans oracle of the cancellation scenarios: every isolated-server
    descendant carries the scenario's unique ``OUROBOROS_DATA_DIR`` in its
    environment, so after a (cascade) cancel every pid this scan finds must still
    be INSIDE the live server tree — a killed worker's subprocess that survived
    teardown would show up here reparented outside it. Same-uid procs only by
    construction (/proc environ of other users is unreadable), which covers the
    whole tree an isolated server can have spawned.

    NEGATIVE-USE ORACLE: one scan, no waiting (see ``wait_pid_env_value`` for why
    a positive claim about a just-spawned child needs a bounded wait instead).
    """
    needle = str(value).encode()
    found = []
    for pid_dir in pathlib.Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        if needle in _read_proc_environ_bytes(pid_dir.name):
            found.append(int(pid_dir.name))
    return found


def secret_values_in_parent_env() -> dict:
    """{env key: value} for every non-trivial credential-shaped value the PARENT
    (pytest) environment currently carries — the values that must never be observed
    in a keyless child, regardless of what variable name they might travel under."""
    out = {}
    for key, value in os.environ.items():
        if not _CREDENTIAL_SHAPE_RE.search(key):
            continue
        if key in STRIPPED_PROVIDER_ENV_KEYS or _is_secret_env_key(key):
            if len(str(value or "").strip()) >= 8:  # too-short values collide by chance
                out[key] = str(value)
    return out


# ---------------------------------------------------------------------------
# Zero-side-effects snapshots (typed tools + safety surface)
# ---------------------------------------------------------------------------

def repo_tree_fingerprint(clone: pathlib.Path, tracked_paths: tuple = ()) -> dict:
    """A comparable snapshot of an isolated clone: HEAD, porcelain status, and the
    exact bytes-hash of every named tracked path. Denial-of-write scenarios take it
    before and after and require EQUALITY — zero side effects, not "looks clean"."""
    clone = pathlib.Path(clone)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(clone),
                          check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(clone),
                            check=True, capture_output=True, text=True).stdout
    hashes = {
        rel: hashlib.sha256((clone / rel).read_bytes()).hexdigest()
        for rel in tracked_paths
    }
    return {"head": head, "status": status, "hashes": hashes}


# ---------------------------------------------------------------------------
# Cost-truth (ABI-3): deep scan for the retired alias spellings in a PUBLIC
# projection. Keys come from the tree's own COST_ALIAS_PAIRS, never a literal.
# ---------------------------------------------------------------------------

def retired_cost_alias_paths(payload) -> list:
    from ouroboros.cost_projection import COST_ALIAS_PAIRS

    retired = {legacy for _honest, legacy in COST_ALIAS_PAIRS}
    found: list = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in retired:
                    found.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload, "$")
    return found


# ---------------------------------------------------------------------------
# Keyless isolated server (roast F21)
# ---------------------------------------------------------------------------

# Everything the child environment must NOT carry in the keyless lane. The env union
# closes the documented hole: IsolatedServer._is_secret_env_key EXEMPTS provider keys,
# so an inherited ANTHROPIC_API_KEY survives the base sanitization by design.
STRIPPED_PROVIDER_ENV_KEYS = frozenset(_PROVIDER_ENV_KEYS) | frozenset(ALL_PROVIDER_CREDENTIAL_KEYS)
PROXY_ENV_KEYS = frozenset({
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
})


class KeylessIsolatedServer(IsolatedServer):
    """``IsolatedServer`` whose child env can never carry a provider credential.

    The base class keeps ``_PROVIDER_ENV_KEYS`` in the child on purpose (benchmark
    servers authenticate from them). This lane's contract is the opposite: the ONLY
    provider config a scenario server may see is what the scenario's settings.json
    says, and that file only ever names the loopback stub.
    """

    def _env(self) -> dict:
        env = super()._env()
        for key in list(env):
            if key in STRIPPED_PROVIDER_ENV_KEYS or key in PROXY_ENV_KEYS:
                env.pop(key, None)
        return env


def keyless_reviewer_slots(*, advisory: bool = False) -> str:
    """The structured ``OUROBOROS_REVIEWER_SLOTS`` value pinning every reviewer row
    to the loopback stub.

    ABI 7.0 (ABI-10): the comma-list reviewer settings keys are RETIRED —
    ``load_settings`` drops them from the file, so pinning them there is a silent
    no-op and the review organ falls back to the shipped OpenRouter default panel
    (observed live on this tree: S2's triad dispatched gemini/terra/opus with no
    credential and deterministically blocked at pack assembly). The structured key
    is the ONE configuration surface, so the keyless lane pins THAT.

    ``advisory=True`` additionally pins the ONE optional advisory reviewer row to
    the stub (wave 3a): the advisory pre-review then runs the bounded NATIVE
    inspection episode against the loopback model instead of being unavailable
    keyless (which the commit gate compensates with an audited bypass).
    """
    row = {"kind": "api_chat", "target_id": MOCK_SLUG}
    payload = {
        "triad": [{"slot_id": f"t{i}", "route": dict(row)} for i in (1, 2, 3)],
        "scope": [{"slot_id": "s1", "route": dict(row)}],
    }
    if advisory:
        payload["advisory"] = {"enabled": True, "route": dict(row)}
    return json.dumps(payload)


def keyless_settings(stub: ScriptedStubModel, **overrides) -> dict:
    """The isolated settings.json for a keyless scenario server.

    Every model-slot key the TREE declares is pinned — un-listed keys default to the
    empty string (slot disabled / no fallback), the live loop slots to the stub slug,
    and the review organ through the structured ``OUROBOROS_REVIEWER_SLOTS`` (the one
    ABI-10 configuration surface; the retired comma keys in the ACTIVE list are pinned
    empty for hygiene but are dropped by ``load_settings`` either way). Deriving the
    slot list from ``provider_models`` (instead of an enumerated literal, as the
    cancellation-harness precedent did) means an upstream slot added tomorrow is
    pinned by construction rather than silently defaulting to a live OpenRouter
    route. Overrides carrying a real provider credential are a scenario bug and are
    refused loudly.
    """
    stub_pair = {"OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"}
    forbidden = (set(ALL_PROVIDER_CREDENTIAL_KEYS) - stub_pair) & set(overrides)
    if forbidden:
        raise ValueError(
            f"keyless lane: overrides must not carry provider credentials: {sorted(forbidden)}"
        )
    cfg: dict = {key: "" for key in (*ACTIVE_MODEL_SETTING_KEYS, *LEGACY_MODEL_SETTING_KEYS)}
    cfg.update({
        # Disk-authored keys: config.apply_settings_to_env cannot author these from
        # the environment, so they have to be in the file, written fresh.
        "OUROBOROS_SAFETY_MODE": "off",
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_RUNTIME_MODE": "light",
        "OUROBOROS_TASK_REVIEW_MODE": "off",
        "OUROBOROS_POST_TASK_EVOLUTION": "false",
        "OUROBOROS_MAX_WORKERS": 4,
        "TOTAL_BUDGET": 10.0,
        "OUROBOROS_PER_TASK_COST_USD": 10.0,
        "OPENAI_COMPATIBLE_BASE_URL": stub.base_url,
        "OPENAI_COMPATIBLE_API_KEY": "stub-key-not-a-credential",
        "OUROBOROS_REVIEWER_SLOTS": keyless_reviewer_slots(),
    })
    for slot in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT"):
        cfg[slot] = MOCK_SLUG
    cfg.update(overrides)
    return cfg


def assert_settings_keyless(settings: dict) -> None:
    """Fail loudly if a scenario's settings smuggle a provider credential."""
    stub_pair = {"OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"}
    offending = sorted(
        key for key in settings
        if key in ALL_PROVIDER_CREDENTIAL_KEYS and key not in stub_pair and str(settings[key] or "").strip()
    )
    assert not offending, f"keyless settings carry provider credentials: {offending}"
    base = str(settings.get("OPENAI_COMPATIBLE_BASE_URL") or "")
    assert base.startswith("http://127.0.0.1:"), f"stub base_url is not loopback: {base!r}"


def clone_repo(destination: pathlib.Path) -> pathlib.Path:
    """One throwaway clone of the checkout under test.

    A clone (not the working tree) is what the runtime is allowed to run against: the
    server owns its repo directory, so an E2E server must never be pointed at a live
    worktree. The commit identity is pinned locally so reviewed-commit scenarios never
    depend on the operator's global git config.
    """
    clone = pathlib.Path(destination) / "clone"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(REPO_ROOT), str(clone)],
                   check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-B", "ouroboros"], cwd=str(clone),
                   check=True, capture_output=True)
    subprocess.run(["git", "remote", "remove", "origin"], cwd=str(clone),
                   check=False, capture_output=True)
    subprocess.run(["git", "config", "user.name", "SystemHarness"], cwd=str(clone),
                   check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "system-harness@e2e.invalid"],
                   cwd=str(clone), check=True, capture_output=True)
    return clone


def write_settings_file(settings_path: pathlib.Path, settings: dict) -> None:
    """0600-before-content settings write (carried over from the v7_wip harness: a
    default-umask write_text once briefly published a live key world-readable; this
    lane never holds a live key, but the shape must not regress when a paid lane
    reuses it)."""
    fd = os.open(settings_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)  # O_CREAT's mode only applies on creation
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(settings, indent=2))
    if not hasattr(os, "fchmod"):
        os.chmod(settings_path, 0o600)


def start_server(clone, root, settings: dict, *, ready_timeout: float = 300) -> KeylessIsolatedServer:
    assert_settings_keyless(settings)
    data_root = pathlib.Path(root) / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    settings_path = data_root / "settings.json"
    write_settings_file(settings_path, settings)
    server = KeylessIsolatedServer(clone, data_root, settings_path)
    server.start(ready_timeout=ready_timeout)
    return server


# ---------------------------------------------------------------------------
# ArtifactOracle: readers of the durable artifacts every scenario asserts against.
# Never an HTTP 200 on its own, never a harness exit code (AGENTS.md: the exit code
# is not the run status) — scenarios read back what the owner and watchdog read.
# ---------------------------------------------------------------------------

class ArtifactOracle:
    def __init__(self, data_root) -> None:
        self.data_root = pathlib.Path(data_root)

    def task_drive(self, task_id: str) -> "ArtifactOracle":
        """The oracle for a HEADLESS task's forked drive root.

        On this tree a headless task's ToolContext drive root is
        ``state/headless_tasks/<task_id>/data`` under the server's data root, so the
        durable review evidence (state/advisory_review.json, the
        advisory_review_bypassed / scope_review_complete events) lands THERE, not in
        the server-level files. Falls back to the server root when the task has no
        forked drive (e.g. a direct-chat turn)."""
        forked = self.data_root / "state" / "headless_tasks" / str(task_id) / "data"
        return ArtifactOracle(forked) if forked.is_dir() else self

    # -- json state files ---------------------------------------------------

    def _json(self, relpath: str) -> dict:
        path = self.data_root / relpath
        if not path.exists():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}

    def queue_snapshot(self) -> dict:
        return self._json("state/queue_snapshot.json")

    def state(self) -> dict:
        return self._json("state/state.json")

    def advisory_review(self) -> dict:
        return self._json("state/advisory_review.json")

    def cancel_intents(self) -> dict:
        blob = self._json("state/cancel_intents.json")
        return blob.get("intents") if isinstance(blob.get("intents"), dict) else {}

    def terminal_deliveries(self) -> dict:
        """The owed-answer outbox (state/terminal_deliveries.json): the durable
        registry a cancel settles AGAINST — an unowed terminal answer is a contract
        violation, so cancellation scenarios read this file, not the chat."""
        return self._json("state/terminal_deliveries.json")

    def task_result(self, task_id: str) -> dict:
        return self._json(f"task_results/{task_id}.json")

    def task_result_bytes(self, task_id: str) -> bytes:
        return (self.data_root / "task_results" / f"{task_id}.json").read_bytes()

    def child_task_ids(self, parent_task_id: str) -> list:
        """Direct children of *parent_task_id* per the durable task_results rows.

        The parent's own stored row deliberately does NOT list children ids (only a
        derived swarm rollup), so lineage enumeration reads the children's rows —
        the same truth ``find_child_tasks`` derives from.
        """
        results_dir = self.data_root / "task_results"
        if not results_dir.is_dir():
            return []
        children = []
        for path in sorted(results_dir.glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(row, dict) and str(row.get("parent_task_id") or "") == str(parent_task_id):
                children.append(str(row.get("task_id") or path.stem))
        return children

    def tree_blackboard(self, root_task_id: str) -> list:
        """Rows of the task-tree ledger (task_trees/<root>/blackboard.jsonl)."""
        return self._jsonl(f"task_trees/{root_task_id}/blackboard.jsonl")

    # -- jsonl logs -----------------------------------------------------------

    def _jsonl(self, relpath: str, *, type_filter: str = "") -> list:
        path = self.data_root / relpath
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if type_filter and type_filter not in line:
                continue  # cheap pre-filter, exact check below
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if type_filter and str(row.get("type") or "") != type_filter:
                continue
            rows.append(row)
        return rows

    def events(self, event_type: str = "") -> list:
        return self._jsonl("logs/events.jsonl", type_filter=event_type)

    def supervisor_rows(self, row_type: str = "") -> list:
        return self._jsonl("logs/supervisor.jsonl", type_filter=row_type)

    def tools_rows(self) -> list:
        return self._jsonl("logs/tools.jsonl")

    def chat_bytes(self) -> bytes:
        path = self.data_root / "logs" / "chat.jsonl"
        return path.read_bytes() if path.exists() else b""

    def running_ids(self) -> set:
        return {
            str(row.get("id") or "")
            for row in (self.queue_snapshot().get("running") or [])
            if isinstance(row, dict)
        }

    # -- boot surface ---------------------------------------------------------

    def server_port(self) -> int:
        """The loopback port the server DURABLY claims (state/server_port).

        The port-file honesty check: this must equal the port the driver actually
        talks to — a stale or absent file strands every ``ouroboros`` CLI attach.
        """
        path = self.data_root / "state" / "server_port"
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0


def ws_url(server: IsolatedServer) -> str:
    """The WS chat endpoint of an isolated server (the SAME surface the SPA opens)."""
    return f"ws://{server.host}:{server.port}/ws"


# ---------------------------------------------------------------------------
# Small drivers
# ---------------------------------------------------------------------------

def wait_until(predicate, timeout: float, interval: float = 0.5):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    return last


def submit_running(server: IsolatedServer, description: str, *,
                   workspace_root: str = "", timeout: float = 120) -> str:
    """Submit a task and wait until the supervisor actually has it RUNNING.

    ``workspace_root`` submits the task as an EXTERNAL-WORKSPACE task (the server's
    own submit sets ``workspace_mode=external`` with it), which is what gives its
    root agent the mutating delegated shape the delegation-mutation scenarios need.
    """
    task_id = server.submit(description, workspace_root=str(workspace_root or ""))
    assert task_id, "submit returned no task id"
    oracle = ArtifactOracle(server.data_root)
    running = wait_until(lambda: task_id in oracle.running_ids(), timeout)
    assert running, f"task {task_id} never reached the RUNNING set"
    return task_id


def wait_durable_result(oracle: ArtifactOracle, task_id: str, *, timeout: float = 180) -> dict:
    """Wait for ``task_results/<id>.json`` to reach a TERMINAL status and return it.

    The HTTP task view can report ``completed`` while the durable terminal write is
    still in flight behind post-task processing (observed live on this tree: the
    stored row said ``scheduled`` seconds after the API said ``completed``). A
    scenario that asserts the durable record must wait for the record, not for the
    HTTP answer.
    """
    terminal = {"completed", "failed", "cancelled", "rejected_duplicate"}
    stored = wait_until(
        lambda: (
            oracle.task_result(task_id)
            if str(oracle.task_result(task_id).get("status") or "") in terminal
            else None
        ),
        timeout,
    )
    assert stored, (
        f"task {task_id} durable result never reached a terminal status: "
        f"{oracle.task_result(task_id)!r}"
    )
    return stored
