"""The card gallery replays current producer bytes, not hand-maintained wire shapes."""
import json
import subprocess
import sys

import pytest

from scripts.generate_card_gallery_fixture import FIXTURE, _normalize_data_paths


def test_card_gallery_recording_matches_producers():
    recorded = FIXTURE.read_bytes()
    modified = FIXTURE.stat().st_mtime_ns
    # Do not let imports and aliases of the recorder's temporary clocks escape into the collection process.
    result = subprocess.run([sys.executable, "-B", "-m", "scripts.generate_card_gallery_fixture", "--stdout"],
                            cwd=FIXTURE.parents[2], capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stdout == recorded, (
        "Card gallery data changed; regenerate with `python -I -S scripts/safe_test.py -- "
        "<dependency-python> -m scripts.generate_card_gallery_fixture`"
    )
    assert FIXTURE.stat().st_mtime_ns == modified and FIXTURE.read_bytes() == recorded


def test_every_recorded_room_reads_the_history_its_producers_wrote():
    """A task card's room replays what its producers stored. A read refused as still importing (an older body's export
    left in a root no import ever reads) would record an empty room that the stand then draws as a state."""
    corpus = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for group in ("task_card", "share"):
        for name, case in corpus[group].items():
            for answer in case["history"]:
                assert "error" not in answer and answer["messages"], (group, name, answer.get("reason_code"))


@pytest.mark.parametrize(("root", "alias", "descendant"), [
    (r"C:\Temp Files\card-gallery-data", "c:/Temp Files/card-gallery-data", r"\logs\trace.json"),
    ("/var/tmp/card-gallery-data", "/private/var/tmp/card-gallery-data", "/logs/trace.json"),
], ids=["windows", "posix"])
def test_gallery_root_normalizes_only_at_path_boundaries(root, alias, descendant):
    prose = f"Read `{alias}{descendant}` before checking the report."
    inputs = {"case": [root, prose, {"nested": alias}, alias + ".sibling", "other" + root,
                       r"leave this unrelated \n alone", False]}
    assert _normalize_data_paths(inputs, root, alias) == {"case": [
        "/gallery-data", "Read `/gallery-data/logs/trace.json` before checking the report.",
        {"nested": "/gallery-data"}, alias + ".sibling", "other" + root, r"leave this unrelated \n alone", False,
    ]}
