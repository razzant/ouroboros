"""Two linked installations in one test process, for the peer-link and letter suites.

Each installation is its own data root and install identity (``install_identity.acting_as``); the
introduction runs the real owner functions on both sides with the network hop replaced by a direct call
(``peer_links._post_introduce``), recording every body that would cross it. The receiver's letter routes are
served by the real gateway behind ``NetworkAuthGate`` through a Starlette ``TestClient``.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class Pair:
    """A (sender) and B (receiver): B lets A start work, A only lets B answer."""

    root_a: pathlib.Path
    root_b: pathlib.Path
    repo: pathlib.Path
    link_id: str
    place_a: str  # A's place for B
    place_b: str  # B's place for A
    invite: str
    wire: list = field(default_factory=list)

    def headers(self, **override: Any) -> dict:
        from ouroboros.peer_links import outgoing_link

        headers = outgoing_link(self.place_a, drive_root=self.root_a).auth_headers()
        headers.update(override)
        return {key: value for key, value in headers.items() if value is not None}

    def secret_a(self) -> str:
        from ouroboros.peer_links import token_path

        return json.loads(token_path(self.root_a, self.link_id).read_text(encoding="utf-8"))["token"]

    def secret_b(self) -> str:
        from ouroboros.peer_links import token_path

        return json.loads(token_path(self.root_b, self.link_id).read_text(encoding="utf-8"))["token"]

    def task_id(self, letter_id: str) -> str:
        from ouroboros.peer_letters import letter_task_id

        return letter_task_id(self.place_b, letter_id)

    def row(self, letter_id: str) -> dict:
        from ouroboros.install_identity import acting_as
        from ouroboros.task_results import load_task_result

        with acting_as(self.root_b):
            return load_task_result(self.root_b, self.task_id(letter_id), strict=True) or {}


def init_control_state(root: pathlib.Path, *, enabled: bool = False) -> None:
    """Write a readable state.json so switch_state(root) reads 'off'/'on' instead of 'unknown'."""
    from supervisor import state as st

    prev = (st.DRIVE_ROOT, st.STATE_PATH, st.STATE_LAST_GOOD_PATH, st.STATE_LOCK_PATH)
    try:
        st.init(root)
        st.save_state(st.ensure_state_defaults({"bg_consciousness_enabled": bool(enabled)}))
    finally:
        st.DRIVE_ROOT, st.STATE_PATH, st.STATE_LAST_GOOD_PATH, st.STATE_LOCK_PATH = prev


def make_roots(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    root_a, root_b, repo = tmp_path / "a" / "data", tmp_path / "b" / "data", tmp_path / "repo"
    for path in (root_a / "memory", root_b / "memory", repo):
        path.mkdir(parents=True)
    init_control_state(root_a, enabled=False)
    init_control_state(root_b, enabled=False)
    return root_a, root_b, repo


def introduce(tmp_path: pathlib.Path, monkeypatch, *, b_lets_a_start: bool = True, a_lets_b_start: bool = False,
              note: str = "Velm may count files for me.", confirm: bool = True) -> Pair:
    """B invites (its owner door), A accepts (its owner door), B records A through ``introduce``; then my human
    confirms the six digits on both sides (``confirm=False`` leaves both links waiting for that check)."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as

    root_a, root_b, repo = make_roots(tmp_path)
    wire: list = []
    with acting_as(root_b):
        invite = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", label="Laptop", own_label="Velm",
                                          note=note, may_start=b_lets_a_start)

    def post(url: str, token: str, body: dict) -> dict:
        wire.append({"url": url, "token": token, "body": dict(body)})
        with acting_as(root_b):
            answer = peer_links.introduce(root_b, token, body)
        wire.append({"answer": dict(answer)})
        return answer

    monkeypatch.setattr(peer_links, "_post_introduce", post)
    with acting_as(root_a):
        view = peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8", label="Velm",
                                        own_label="Laptop", may_start=a_lets_b_start)
    with acting_as(root_b):
        place_b = peer_links.read_registry(root_b)["links"][view["link_id"]]["place_id"]
    for root in (root_a, root_b) if confirm else ():
        with acting_as(root):
            peer_links.confirm(root, view["link_id"])
    return Pair(root_a, root_b, repo, view["link_id"], view["place_id"], place_b, invite["invite"], wire)


def ready_queue(monkeypatch, drive_root: pathlib.Path) -> None:
    """The queue and worker pool of a ready server, in memory (no worker runs)."""
    import supervisor.queue as queue
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace(busy_task_id=None, reaping=False)})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(queue, "DRIVE_ROOT", drive_root)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda *args, **kwargs: True)


@contextlib.contextmanager
def receiver_client(pair: Pair, monkeypatch, *, routes=None):
    """B's gateway behind its auth gate, as its own install."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway.router import collect_routes
    from ouroboros.install_identity import acting_as
    from ouroboros.server_auth import NetworkAuthGate

    ready_queue(monkeypatch, pair.root_b)
    app = Starlette(routes=routes if routes is not None else collect_routes(data_dir=pair.root_b))
    app.state.drive_root, app.state.repo_dir = pair.root_b, pair.repo
    gate = NetworkAuthGate(app)

    async def as_receiver(scope, receive, send):
        with acting_as(pair.root_b):
            await gate(scope, receive, send)

    # A direct connection from this computer, as the sender's own loopback request arrives (server_auth admits a
    # link header only so).
    with TestClient(as_receiver, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        yield client


def letter(**fields: Any) -> dict:
    return {"letter_id": fields.pop("letter_id", uuid.uuid4().hex[:12]), "text": "Count the files in /tmp.",
            "protocol": 1, **fields}


def published(place_id: str, text: str, *, root: Any = None, link_id: str = "", kind: str = "consciousness") -> dict:
    """The publication fact ``peer_letters.admit_words`` gives words it let through (place, sha256, and when ``root``
    and ``link_id`` are provided, the exact link id and current rule basis): what a test recording a question directly
    stands in for."""
    import hashlib

    fact: dict[str, Any] = {"place_id": place_id, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    if root is not None and link_id:
        from ouroboros.install_identity import acting_as
        from ouroboros.peer_letters import speech_basis

        with acting_as(root):
            basis = speech_basis(root, place_id, kind=kind)
        fact.update(link_id=link_id, basis=basis, coverage_kind=kind)
    return fact


def admitted(pair: Pair, text: str, *, kind: str = "consciousness") -> dict:
    """The terminal stamp of B's letter root whose final was admitted for A (``peer_letters.REPLY_KEY``), as the
    terminal projection writes it beside ``result``."""
    return published(pair.place_b, text, root=pair.root_b, link_id=pair.link_id, kind=kind)


def safety_reads(monkeypatch, *replies: tuple) -> list:
    """The Supervisor's reading stood in for (``safety.check_safety``): each call's tool and subject recorded, then the
    next of ``replies`` (``(allowed, message)``) answered, ``(True, "")`` once they run out."""
    from ouroboros import safety

    calls: list = []
    queue = list(replies)

    def check(tool_name, arguments, **_kwargs):
        calls.append((tool_name, dict(arguments)))
        return queue.pop(0) if queue else (True, "")
    monkeypatch.setattr(safety, "check_safety", check)
    return calls


def events(root: pathlib.Path, kind: str = "") -> list[dict]:
    path = root / "logs" / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    rows = [json.loads(line) for line in lines if line.strip()]
    return [row for row in rows if not kind or row.get("type") == kind]


def task_count(root: pathlib.Path) -> int:
    """How many task rows the store holds (an absent store holds none)."""
    from ouroboros.install_identity import acting_as
    from ouroboros.task_store import newest_ids

    with acting_as(root):
        if not (root / "state").exists():
            return 0
        return len(newest_ids(root))
