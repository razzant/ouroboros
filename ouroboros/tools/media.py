"""Media tools: ocr_pdf (text-layer PDF extraction) + youtube_transcript (captions).

Both are lightweight and DEPENDENCY-OPTIONAL: each returns a typed `⚠️ *_UNAVAILABLE`
string instead of raising when its optional dependency or data is absent, so a missing
dep degrades gracefully rather than burning rounds. `ocr_pdf` reuses the view_image
local-file trust boundary; `youtube_transcript` is web-gated like web_search
(`registry._WEB_TOOLS`). `extract_video_frames` optionally uses `ffmpeg` from PATH
when available and returns typed `EXTRACT_VIDEO_FRAMES_UNAVAILABLE` otherwise.
"""
from __future__ import annotations

import html as _html
import json as _json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import List

from ouroboros.tools.registry import ToolContext, ToolEntry

_OCR_PDF_MAX_BYTES = 25 * 1024 * 1024
_OCR_PDF_MAX_PAGES = 50
_OCR_PDF_MAX_CHARS = 200_000
_YT_HTTP_TIMEOUT_SEC = 30
_YT_MAX_CHARS = 200_000
_YT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OuroborosMedia/1.0)"}
_VIDEO_MAX_BYTES = 512 * 1024 * 1024
_VIDEO_MAX_FRAMES = 12
_VIDEO_FRAME_TIMEOUT_SEC = 120


def _resolve_local_file(ctx: ToolContext, path: str, *, max_bytes: int) -> tuple[pathlib.Path | None, str]:
    """Resolve a local file path through the SAME trust boundary as view_image/read_file:
    it must sit under an allowed file root and pass the protected-artifact read guard.
    Returns (path, "") on success or (None, error_message)."""
    from ouroboros.tools.vision import _allowed_file_roots
    from ouroboros.tool_access import path_is_relative_to

    text = str(path or "").strip()
    if not text:
        return None, "⚠️ TOOL_ARG_ERROR: `path` is required."
    roots = _allowed_file_roots(ctx)
    raw = pathlib.Path(text).expanduser()
    # Resolve against EVERY allowed root (not just the first) so a RELATIVE path from the
    # staged-attachment manifest — e.g. `attachments/doc.pdf`, which lives under artifact_store,
    # not the uploads root — is found wherever it actually is, matching the
    # read_file(root='artifact_store', path='attachments/...') contract the manifest advertises.
    if raw.is_absolute():
        candidates = [raw.resolve(strict=False)]
    else:
        candidates = [(r / text).resolve(strict=False) for r in roots]
    confined = [c for c in candidates if any(c == r or path_is_relative_to(c, r) for r in roots)]
    if not confined:
        return None, (
            f"⚠️ PATH_BLOCKED: {text} is outside the allowed file roots "
            "(uploads / active workspace / artifact store / task drive)."
        )
    fp = next((c for c in confined if c.is_file()), confined[0])
    try:
        from ouroboros.protected_artifacts import block_reason_for_path

        reason = block_reason_for_path(ctx, fp, "read_bytes")
        if reason:
            return None, f"⚠️ PATH_BLOCKED: {reason}"
    except Exception:  # noqa: BLE001 — guard is best-effort; the root check above is the floor
        pass
    if not fp.exists() or not fp.is_file():
        return None, f"⚠️ FILE_NOT_FOUND: {text}"
    if fp.stat().st_size > max_bytes:
        return None, f"⚠️ FILE_TOO_LARGE: {fp.stat().st_size} bytes (max {max_bytes})."
    return fp, ""


def _ocr_pdf(ctx: ToolContext, path: str = "", max_pages: int = 0) -> str:
    """Extract the embedded TEXT layer of a PDF (digital PDFs). Scanned/image-only PDFs
    have no text layer → typed `⚠️ OCR_PDF_SCANNED_UNAVAILABLE` (true OCR of scanned pages
    is a deferred follow-up; use vlm_query on a page image meanwhile). Reuses the view_image
    local-file trust boundary."""
    fp, err = _resolve_local_file(ctx, path, max_bytes=_OCR_PDF_MAX_BYTES)
    if err:
        return err
    try:
        from pypdf import PdfReader
    except Exception:  # noqa: BLE001
        return "⚠️ OCR_PDF_UNAVAILABLE: the 'pypdf' dependency is not installed in this build."
    try:
        reader = PdfReader(str(fp))
        pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ OCR_PDF_UNAVAILABLE: could not parse PDF ({type(exc).__name__})."
    total = len(pages)
    cap = int(max_pages) if int(max_pages or 0) > 0 else _OCR_PDF_MAX_PAGES
    chunks: List[str] = []
    for page in pages[:cap]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — one bad page must not sink the whole extraction
            chunks.append("")
    text = "\n\n".join(c for c in chunks if c).strip()
    if not text:
        return (
            "⚠️ OCR_PDF_SCANNED_UNAVAILABLE: this PDF has no extractable text layer (likely "
            "scanned/image-only). True OCR of scanned pages is not available in this build — "
            "render a page to an image and call vlm_query on it instead."
        )
    note = "" if total <= cap else f"\n\n[disclosed: showed first {cap} of {total} pages]"
    if len(text) > _OCR_PDF_MAX_CHARS:
        text = text[:_OCR_PDF_MAX_CHARS]
        note += "\n[disclosed: text truncated]"
    return f"PDF text ({min(total, cap)} page(s)):\n\n{text}{note}"


def _youtube_video_id(url: str) -> str:
    """Best-effort extraction of an 11-char YouTube video id from a URL or a bare id."""
    text = str(url or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", text):
        return text
    for pat in (r"[?&]v=([0-9A-Za-z_-]{11})", r"youtu\.be/([0-9A-Za-z_-]{11})", r"/(?:embed|shorts|v)/([0-9A-Za-z_-]{11})"):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def _extract_json_array(text: str, key: str) -> str | None:
    """Depth-balanced extraction of the JSON array assigned to `"key":[ ... ]` (handles
    nested arrays like a caption track's `name.runs`, which a non-greedy regex would split)."""
    idx = text.find(f'"{key}":[')
    if idx < 0:
        return None
    start = text.find("[", idx)
    depth = 0
    for j in range(start, len(text)):
        c = text[j]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None


def _youtube_transcript(ctx: ToolContext, url: str = "", lang: str = "en") -> str:
    """Fetch a YouTube video's caption transcript (timed-text) over plain HTTP. Web-gated
    like web_search (registry._WEB_TOOLS). Returns `⚠️ YOUTUBE_TRANSCRIPT_UNAVAILABLE` when
    captions are absent or the (unofficial, no-SLA) endpoint shape changes — never raises."""
    vid = _youtube_video_id(url)
    if not vid:
        return "⚠️ TOOL_ARG_ERROR (youtube_transcript): provide a YouTube URL or 11-char video id."
    try:
        import requests
    except Exception:  # noqa: BLE001
        return "⚠️ YOUTUBE_TRANSCRIPT_UNAVAILABLE: the 'requests' dependency is not installed."
    try:
        watch = requests.get(f"https://www.youtube.com/watch?v={vid}", headers=_YT_HEADERS, timeout=_YT_HTTP_TIMEOUT_SEC)
        watch.raise_for_status()
        raw = _extract_json_array(watch.text, "captionTracks")
        if not raw:
            return "⚠️ YOUTUBE_TRANSCRIPT_UNAVAILABLE: no caption tracks for this video."
        tracks = _json.loads(raw)
        if not isinstance(tracks, list) or not tracks:
            return "⚠️ YOUTUBE_TRANSCRIPT_UNAVAILABLE: no caption tracks for this video."
        chosen = next((t for t in tracks if isinstance(t, dict) and str(t.get("languageCode") or "").startswith(str(lang or "en"))), tracks[0])
        base_url = str(chosen.get("baseUrl") or "")
        if not base_url:
            return "⚠️ YOUTUBE_TRANSCRIPT_UNAVAILABLE: caption track has no fetch URL."
        xml = requests.get(base_url, headers=_YT_HEADERS, timeout=_YT_HTTP_TIMEOUT_SEC).text
        parts = re.findall(r"<text[^>]*>(.*?)</text>", xml, flags=re.DOTALL)
        text = _html.unescape(re.sub(r"<[^>]+>", "", "\n".join(parts))).strip()
        if not text:
            return "⚠️ YOUTUBE_TRANSCRIPT_UNAVAILABLE: the caption track was empty."
        note = "" if len(text) <= _YT_MAX_CHARS else "\n[disclosed: transcript truncated]"
        return f"YouTube transcript ({vid}, lang={chosen.get('languageCode') or '?'}):\n\n{text[:_YT_MAX_CHARS]}{note}"
    except Exception as exc:  # noqa: BLE001 — unofficial endpoint; fail soft + typed
        return f"⚠️ YOUTUBE_TRANSCRIPT_UNAVAILABLE: fetch failed ({type(exc).__name__})."


def _resolve_ffmpeg() -> str | None:
    """ffmpeg resolver chain (v6.56.0): an ffmpeg binary BESIDE the current
    interpreter (venv/bundled bin — benchmark servers often start without
    `activate`, so a venv-installed ffmpeg never reaches PATH) → the
    imageio-ffmpeg wheel binary (the Terminal-Bench agent-prefix install) →
    plain PATH lookup. Returns None when no candidate exists."""
    sibling = pathlib.Path(sys.executable).resolve(strict=False).parent / (
        "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    )
    try:
        if sibling.exists() and os.access(sibling, os.X_OK):
            return str(sibling)
    except OSError:
        pass
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        exe = str(imageio_ffmpeg.get_ffmpeg_exe() or "")
        if exe:
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _extract_video_frames(ctx: ToolContext, path: str = "", timestamps: str = "", max_frames: int = 5) -> str:
    """Extract selected video frames with ffmpeg when available."""
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return (
            "⚠️ EXTRACT_VIDEO_FRAMES_UNAVAILABLE: no ffmpeg found (venv/bundled bin, "
            "imageio-ffmpeg, or PATH). Workaround: extract frames yourself with "
            "python+cv2 (`cv2.VideoCapture` + `cv2.imwrite`) via run_script, then "
            "inspect the saved frames with view_image."
        )
    fp, err = _resolve_local_file(ctx, path, max_bytes=_VIDEO_MAX_BYTES)
    if err:
        return err
    try:
        count = max(1, min(_VIDEO_MAX_FRAMES, int(max_frames or 5)))
    except (TypeError, ValueError):
        count = 5
    raw_times = [t.strip() for t in re.split(r"[,\s]+", str(timestamps or "")) if t.strip()]
    if not raw_times:
        # Deterministic low-cost default: first few seconds, useful for thumbnails/UI tasks.
        raw_times = [str(i) for i in range(count)]
    raw_times = raw_times[:count]
    from ouroboros.tool_access import resource_root_path

    out_dir = (resource_root_path(ctx, "artifact_store") / "video_frames").resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    errors: list[str] = []
    for idx, ts in enumerate(raw_times, start=1):
        try:
            float(ts)
        except ValueError:
            errors.append(f"invalid timestamp {ts!r}")
            continue
        out = out_dir / f"{fp.stem}_frame_{idx:02d}.jpg"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", ts, "-i", str(fp), "-frames:v", "1", "-y", str(out)]
        try:
            from ouroboros.tools.shell import _tracked_subprocess_run

            res = _tracked_subprocess_run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=_VIDEO_FRAME_TIMEOUT_SEC
            )
        except subprocess.TimeoutExpired:
            errors.append(f"{ts}s: timeout")
            continue
        if res.returncode != 0 or not out.exists():
            errors.append(f"{ts}s: {(res.stderr or 'ffmpeg failed').strip()[:200]}")
            continue
        written.append(str(out))
    if not written:
        detail = "; ".join(errors[:5]) or "no frames produced"
        return f"⚠️ EXTRACT_VIDEO_FRAMES_UNAVAILABLE: {detail}"
    note = f"\nWarnings: {'; '.join(errors[:5])}" if errors else ""
    return "Extracted video frame(s):\n" + "\n".join(written) + note


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="ocr_pdf",
            schema={
                "name": "ocr_pdf",
                "description": (
                    "Extract the text of a local PDF file (the embedded text layer of a digital PDF). "
                    "Use for reading PDFs attached to the task (see the [ATTACHMENTS] manifest) or produced "
                    "during work. Scanned/image-only PDFs have no text layer and return a typed "
                    "OCR_PDF_SCANNED_UNAVAILABLE notice — for those, render a page and use vlm_query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Local PDF path (inside the active workspace / task drive / artifact store / attachments / uploads)."},
                        "max_pages": {"type": "integer", "description": f"Cap pages read (default {_OCR_PDF_MAX_PAGES}; over-cap is disclosed)."},
                    },
                    "required": ["path"],
                },
            },
            handler=_ocr_pdf,
            timeout_sec=120,
        ),
        ToolEntry(
            name="youtube_transcript",
            schema={
                "name": "youtube_transcript",
                "description": (
                    "Fetch the caption transcript of a YouTube video by URL or video id. Returns the "
                    "transcript text, or a typed YOUTUBE_TRANSCRIPT_UNAVAILABLE notice when the video has "
                    "no captions. Requires web access."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "YouTube URL or 11-char video id."},
                        "lang": {"type": "string", "description": "Preferred caption language code prefix (default 'en'); falls back to the first available track."},
                    },
                    "required": ["url"],
                },
            },
            handler=_youtube_transcript,
            timeout_sec=90,
        ),
        ToolEntry(
            name="extract_video_frames",
            schema={
                "name": "extract_video_frames",
                "description": (
                    "Extract still frames from a local video file using ffmpeg when available. "
                    "Frames are written under artifact_store/video_frames and can be inspected with view_image. "
                    "PREFER this + view_image for any question about a video's VISUAL content (what is shown, "
                    "colors, counts, text on screen) — a clean extracted frame beats a browser screenshot of a "
                    "compressed player, and transcripts carry no visuals at all. "
                    "Returns a typed EXTRACT_VIDEO_FRAMES_UNAVAILABLE notice when ffmpeg or the file is unavailable."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Local video path inside an allowed file root."},
                        "timestamps": {"type": "string", "description": "Optional comma/space-separated seconds to extract. Defaults to the first N integer seconds."},
                        "max_frames": {"type": "integer", "description": f"Maximum frames to extract (default 5, hard cap {_VIDEO_MAX_FRAMES})."},
                    },
                    "required": ["path"],
                },
            },
            handler=_extract_video_frames,
            timeout_sec=_VIDEO_FRAME_TIMEOUT_SEC + 10,
        ),
    ]
