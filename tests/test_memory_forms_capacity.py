"""The physical floor keeps the capacity and readers of 87bf33574e (v7 R11).

Ceilings and reader fingerprints are measured in the detached baseline with
these exact synthetic bytes, not in the candidate branch. No owner data.
"""
import hashlib
import re

import pytest

from ouroboros import memory_floor as mf, memory_view as mv
from tests import _memory_view_synthetic as syn


def capacity_case(name):
    if name in ("owner", "read-people"):
        groups = [syn.conversation(i, owner=name == "owner", words=6, word_chars=1500) for i in range(99)]
        for group in groups:
            room = group["rooms"][0]
            if name == "read-people":
                for word in room["words"]:
                    word["rank"] = 1
            room["facts"].update(rows=6, people=6, mine=0, task_facts=0, chars=9000,
                                 earliest=room["words"][0]["ts"], latest=room["words"][-1]["ts"])
            group["latest"] = room["facts"]["latest"]
        snap = syn.snapshot(room_facts=syn.room("1"), conversations=groups)
    elif name.startswith("replies"):
        rows = [syn.spoken(i, "ouroboros", 6000) for i in range(1 if name.endswith("one") else 100)]
        snap = syn.snapshot(room_facts=syn.room("1", lane1=rows))
    else:
        pages = syn.pages(1 if name.endswith("one") else 80, 6000)
        for page in pages:
            page["text"] = syn.prose(6000)  # an unmarked first sentence is an emergency opening
        snap = syn.snapshot(room_facts=syn.room("1"), story=pages)
    return snap, dict(mode="low", owner_mode="max", window_tokens=200_000, known_window=True,
                      output_reserve=10_000, ratio=1.0, non_memory_tokens=150_000)


def readers(text):
    return sorted(set(re.findall(r"row:\d+@[^\s;'\])]+#[a-f0-9]+|memory_read\([^\n]*?\)", text)))


def fingerprint(items):
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


# Replaced only from an explicit baseline worktree measurement, never from a failure.
BASELINE = {
    "owner": {"snapshot": "caa4cce36ad5b360244e511cf54994f4d7ca289254045da2bee3d762cf3c1f69", "tokens": 18771,
              "readers": "bd260c811e47493b0a948fed576942b929d4ae2745c50a4e90c18cfbbde2248b"},
    "read-people": {"snapshot": "b98fc234a49cf80e86ace1b7d532b105638c733c211cc8614574b89982b0f241", "tokens": 486,
                    "readers": "ca54eb34a3a6cbf517646706f45dd8f05d45471a7490c5e0db1345b549a897b4"},
    "replies-one": {"snapshot": "5407b6111bfdfb1d6e1dfc6df14652168a52d79b814cebca59039016cd3b034d", "tokens": 303,
                    "readers": "1a7e15e2a43df8abc06ec27b06f44a3bdd8d67b56566ed2fe2beb4a3b634cdbd"},
    "replies-many": {"snapshot": "a59312f879d90f0b118d04acf8f3139546111ee6a36b7ef57ac81aa428ceffaa", "tokens": 357,
                     "readers": "902a6267da74588ff5106fac2e2e9df67dffe8a7e0a0444d9ca72933b3d84f2d"},
    "story-one": {"snapshot": "b70cfbb6b90eb431133f1808ac73bcb4fde2b2e4e7096807c72c26c095689769", "tokens": 270,
                  "readers": "77e5f47672af9461c401f40520cade332e89214c7c17cc72c669d8d1376a4c20"},
    "story-many": {"snapshot": "015f8c1d1954035470764b7ac385db3b889dcc3e162919dc236c39bc54684cfd", "tokens": 311,
                   "readers": "41b604b7807b05c71bf72f03083ec05f5456df3e19aeddcb1e4075ed548abe3c"},
}


@pytest.mark.parametrize("name", list(BASELINE))
def test_physical_floor_preserves_baseline_capacity_and_readers(name):
    snap, frame = capacity_case(name)
    baseline = BASELINE[name]
    assert hashlib.sha256(mv.snapshot_json(snap).encode()).hexdigest() == baseline["snapshot"]
    shown = mf.render_view_for_mode(snap, **frame, shortest=True)
    assert mf.view_tokens(shown.story) + mf.view_tokens(shown.room) <= baseline["tokens"]
    assert fingerprint(readers(shown.story + "\n" + shown.room)) == baseline["readers"]
