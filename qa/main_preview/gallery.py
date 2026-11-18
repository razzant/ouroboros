"""Assemble one offline replay. Read only the fixed, bounded capture layout."""
import base64
import json
import os
import sys
from pathlib import Path

ENGINES = ("chromium", "webkit")
THEMES = ("dark", "light")
SCENES = ("now", "a", "b", "a-archive", "b-archive", "a-helper", "b-helper", "b-absent",
          "evo-held", "evo-held-design", "evo-queue", "evo-queue-design")
PNG = b"\x89PNG\r\n\x1a\n"


def read_scene(folder, passed):
    meta = folder / "frames.json"
    if not meta.is_file() or meta.stat().st_size > 16 * 1024:
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        frames = []
        for i, frame in enumerate(data["frames"][:5]):
            filename = f"{i:02}.png"
            image = folder / filename
            if frame["image"] != filename or not image.is_file() or image.stat().st_size > 768 * 1024:
                break
            png = image.read_bytes()
            if not png.startswith(PNG):
                break
            frames.append({**frame, "image": "data:image/png;base64," + base64.b64encode(png).decode("ascii")})
        if frames:
            complete = bool(passed and data.get("recorded_all_steps") and len(frames) == len(data["frames"]))
            return {"frames": frames, "complete": complete,
                    **({key: data[key] for key in ("anchor", "resume", "navigation_posts")} if complete else {})}
    except (KeyError, ValueError, OSError, TypeError):
        return None  # A missing or interrupted case cannot invalidate the other cases.
    return None


def build(directory):
    data = {}
    for engine in ENGINES:
        try:
            receipt = json.loads((Path(os.environ['QA_RAW']) / f'main-{engine}.json').read_text(encoding='utf-8'))
            passed = receipt['status'] == 'passed' and receipt['counts']['passed'] == 30
        except (OSError, KeyError, ValueError, TypeError):
            passed = False
        data[engine] = {theme: {name: read_scene(directory / engine / theme / name, passed) for name in SCENES}
                        for theme in THEMES}
    if not any(row for engine in data.values() for theme in engine.values() for row in theme.values()):
        raise SystemExit("No preview case reached a captured frame.")
    template = Path(__file__).with_name("compare.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    # Cap the standalone HTML independently of the per-frame limits.
    if len(payload.encode("utf-8")) > 48 * 1024 * 1024:
        raise ValueError("Preview exceeds the standalone HTML bound")
    (directory / "index.html").write_text(template.replace("__PREVIEW_DATA__", payload), encoding="utf-8")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
