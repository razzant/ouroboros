"""``--stub``: the $0 rehearsal of every scenario against the loopback stub model.

Reuses the system-E2E harness (``tests/system_e2e/harness.py``: the loopback model server, the
review-organ classification with its canned parse-clean verdicts, ``keyless_settings``) instead
of a second stub; the runner imports it lazily and only in stub mode. The one thing added here
is ROUTING: a swarm scenario interleaves
router, parent, child and admission-probe calls on one wire, so the script is a map of
per-role queues rather than one ordered list (``scenarios.<id>_stub_script``).
"""
from __future__ import annotations

import json

STUB_MODEL_SLUG = "openai-compatible::mock-model"   # == harness.MOCK_SLUG (asserted in stub_settings)
STUB_CHILD_SLUG = "openai-compatible::mock-child"
STUB_MODEL_SLOTS = {"OUROBOROS_MODEL": STUB_MODEL_SLUG, "OUROBOROS_MODEL_LIGHT": STUB_MODEL_SLUG}
ROUTER_PROMPT_KEY = '"promoted_task_toolset"'  # only the Swarm router turn's runtime context carries it


def routed_stub_model(script: dict):
    """A loopback model serving ``{role: [steps]}``; review-organ calls stay canned."""
    from tests.system_e2e import harness

    class RoutedStubModel(harness.LoopbackModelServer):
        def __init__(self) -> None:
            super().__init__()
            self.queues = {role: list(steps) for role, steps in script.items()}
            self.roles: list[str] = []

        def _model_ids(self) -> list[str]:
            return ["mock-model", "mock-child"]

        def _route(self, body: dict) -> str:
            if not body.get("tools"):
                return "probe"
            if "mock-child" in str(body.get("model") or ""):
                return "child"
            if ROUTER_PROMPT_KEY in harness.body_text(body):
                return "router"
            return "agent"

        def _answer(self, body: dict, seq: int) -> tuple[str, dict]:
            kind = harness.classify_call(body)
            canned = harness.canned_review_answer(kind)
            if canned is not None:
                return kind, canned
            role = self._route(body)
            self.roles.append(role)
            queue = self.queues.get(role) or []
            step = queue[0] if queue else None
            if callable(step):
                step = step(harness.body_text(body))
            if step is None:
                return "final", {"role": "assistant", "content": f"Stub: no scripted step left for role {role}."}
            if "final" in step:
                if len(queue) > 1 or role in ("agent",):
                    queue.pop(0)  # a final closes ONE task; a child/probe final repeats for every caller
                return "final", {"role": "assistant", "content": str(step["final"])}
            queue.pop(0)
            call = {"name": str(step["tool"]), "arguments": json.dumps(step.get("arguments") or {})}
            return role, {"role": "assistant", "content": "still working",
                          "tool_calls": [{"id": f"call_{seq}", "type": "function", "function": call}]}

        def consumed(self) -> dict:
            return {role: len(queue) for role, queue in self.queues.items()}

    return RoutedStubModel()


def stub_settings(stub, template: dict) -> dict:
    """The keyless lane settings: every slot the tree declares pinned (the loop slots to the
    loopback stub, the rest empty), the review panel and the advisory row on the stub, then the
    run template's knobs (budget, workers, evolution) on top. Refuses a template that would
    smuggle a paid slot or a credential."""
    from tests.system_e2e import harness

    if harness.MOCK_SLUG != STUB_MODEL_SLUG:
        raise RuntimeError("stub slug drifted from the harness MOCK_SLUG")
    cfg = harness.keyless_settings(
        stub,
        OUROBOROS_REVIEWER_SLOTS=harness.keyless_reviewer_slots(advisory=True),
        OUROBOROS_RUNTIME_MODE="advanced",
        # The tree's default context mode, declared explicitly: the suite's keyless Low would
        # either be normalized to Max at boot (a persist the strict snapshot pin refuses) or,
        # with the owner marker, SKIP whole-repository scope review by design.
        OUROBOROS_CONTEXT_MODE="max",
        OUROBOROS_CONTEXT_MODE_AUTO_LOW="false",
    )
    paid = {k: v for k, v in template.items()
            if k.startswith("OUROBOROS_MODEL") and v and str(v) != STUB_MODEL_SLUG}
    if paid:
        raise RuntimeError(f"stub template carries paid model slots: {sorted(paid)}")
    cfg.update(template)
    cfg["OPENAI_COMPATIBLE_BASE_URL"] = stub.base_url
    harness.assert_settings_keyless(cfg)
    return cfg
