"""Checkpoint bounded, fixture-only frames as the real browser takes each step."""
import json
import os
from pathlib import Path

from tests.test_room_work_browser import settle_frames

BOX = """el => {
    const r = el.getBoundingClientRect();
    const at = document.elementFromPoint(r.x + r.width/2, r.y + r.height/2);
    return {x:r.x, y:r.y, w:r.width, h:r.height, hit:el === at || el.contains(at)};
}"""
MAX_PNG = 768 * 1024


class Frames:
    def __init__(self, page, theme, scenario):
        self.page = page
        self.directory = Path(os.environ["OUROBOROS_UI_EVIDENCE_DIR"]) / theme / scenario
        self.directory.mkdir(parents=True, exist_ok=False)
        self.frames = []

    def _save(self, **facts):
        dest = self.directory / "frames.json"
        temp = dest.with_suffix(".tmp")
        temp.write_text(json.dumps({"frames": self.frames, **facts}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(dest)

    def add(self, description, button=None, action=None):
        settle_frames(self.page)
        # On a geometry failure save the real screen but do not advertise a clickable control.
        spot = button.evaluate(BOX) if button is not None and button.count() == 1 else None
        hit = spot and spot.pop("hit") and spot["w"] > 0 and spot["h"] > 0
        assert len(self.frames) < 5
        image = f"{len(self.frames):02}.png"
        png = self.page.screenshot(animations="disabled")
        assert len(png) <= MAX_PNG, "Preview frame exceeds the publication bound"
        (self.directory / image).write_bytes(png)
        self.frames.append({"image": image, "description": description, "action": action if hit else None,
                            "box": spot if hit else None, "viewport": self.page.viewport_size})
        self._save(recorded_all_steps=False)
        assert button is None or hit, "The expected control is not hit-testable"

    def finish(self, *, anchor, resume=None):
        # Export only test-proven, synthetic facts; never the request log, state, or a fixture's console.
        self._save(recorded_all_steps=True, navigation_posts=0, anchor=anchor, resume=resume)

    def finish_find(self, offsets):
        self._save(recorded_all_steps=True, from_header_px=dict(zip(("full_height", "stock_short", "overlay_short"), offsets)))
