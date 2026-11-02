"""Media tools: extract_pdf_text / ocr_pdf (text-layer PDF extraction) + youtube_transcript (captions).

Both are lightweight and DEPENDENCY-OPTIONAL: each returns a typed `⚠️ *_UNAVAILABLE`
string instead of raising when its optional dependency or data is absent, so a missing
dep degrades gracefully rather than burning rounds. The PDF readers (one handler;
`ocr_pdf` is the historical name and does not OCR) reuse the view_image local-file trust
boundary; `youtube_transcript` is web-gated like web_search
(`registry._WEB_TOOLS`). `extract_video_frames` optionally uses `ffmpeg` from PATH
when available, writes each call's frames into a new directory, and returns typed
`EXTRACT_VIDEO_FRAMES_UNAVAILABLE` otherwise.
"""
from __future__ import annotations

import html as _html
import json as _json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List

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
    from ouroboros.tools.vision import _allowed_file_roots, _read_file_parity_block
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
    # SC-6: the SAME per-path rules read_file applies on the matrix-derived roots
    # (user_files confinement and the
    # project-store guard) — shared with view_image/vlm_query so root admission
    # never out-reaches read_file here either.
    parity_block = _read_file_parity_block(ctx, fp)
    if parity_block:
        return None, parity_block
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


def _page_spans(numbers: List[int]) -> str:
    """``[1, 2, 3, 7]`` -> ``"1-3,7"``: the same spelling ``pages`` accepts."""
    spans: List[List[int]] = []
    for number in numbers:
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def _select_pages(spec: str, total: int) -> tuple[List[int], str, str]:
    """1-based inclusive ``pages`` ('3', '2-5', '7-', '1,4-6') -> (in-range pages, spans past the end, error)."""
    if not str(spec or "").strip():
        return list(range(1, total + 1)), "", ""
    wanted: set[int] = set()
    beyond: List[str] = []
    for part in filter(None, re.split(r"[,\s]+", re.sub(r"\s*-\s*", "-", str(spec).strip()))):
        match = re.fullmatch(r"(\d+)?(-)?(\d+)?", part)
        if not match or not (match.group(1) or match.group(3)):
            return [], "", f"pages must look like '3', '2-5', '7-' or '1,4-6' (1-based); got {spec!r}."
        first = int(match.group(1) or 1)
        last = int(match.group(3)) if match.group(3) else (max(total, first) if match.group(2) else first)
        if first < 1 or last < first:
            return [], "", f"page range {part!r} is empty or below page 1."
        wanted.update(range(first, min(last, total) + 1))  # bounded by the document, never by the spec
        if last > total:
            low = max(first, total + 1)
            beyond.append(str(low) if low == last else f"{low}-{last}")
    return sorted(wanted), ",".join(beyond), ""


def _ocr_pdf(ctx: ToolContext, path: str = "", max_pages: int = 0, pages: str = "", start_char: int = 0) -> str:
    """Extract the embedded TEXT layer of selected PDF pages (shared by ocr_pdf and
    extract_pdf_text). This is not OCR: a page without a text layer is a named gap, and
    a selection with no text at all is the typed OCR_PDF_SCANNED_UNAVAILABLE notice
    (view_image/vlm_query a rendered page instead). Every page the result does not
    deliver is named; a partly delivered page can resume with ``pages`` and
    ``start_char``. Reuses the view_image local-file trust boundary."""
    fp, err = _resolve_local_file(ctx, path, max_bytes=_OCR_PDF_MAX_BYTES)
    if err:
        return err
    try:
        from pypdf import PdfReader
    except Exception:  # noqa: BLE001
        return "⚠️ OCR_PDF_UNAVAILABLE: the 'pypdf' dependency is not installed in this build."
    try:
        reader = PdfReader(str(fp))
        all_pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ OCR_PDF_UNAVAILABLE: could not parse PDF ({type(exc).__name__})."
    total = len(all_pages)
    if not total:
        return "⚠️ OCR_PDF_UNAVAILABLE: the PDF has no pages."
    selected, past_end, problem = _select_pages(pages, total)
    if type(start_char) is not int or start_char < 0 or (start_char and len(selected) != 1):
        return "⚠️ TOOL_ARG_ERROR: start_char must be nonnegative and a nonzero offset requires exactly one selected page."
    if problem or not selected:
        return f"⚠️ TOOL_ARG_ERROR: {problem or f'pages {past_end} are past the last page ({total}).'}"
    try:
        cap = int(max_pages) if int(max_pages or 0) > 0 else _OCR_PDF_MAX_PAGES
    except (TypeError, ValueError):
        cap = _OCR_PDF_MAX_PAGES
    sections: List[str] = []
    delivered: List[int] = []
    empty: List[int] = []
    failed: Dict[str, List[int]] = {}  # exception class -> pages: every failing page stays named
    budget, cut_in, next_char = _OCR_PDF_MAX_CHARS, 0, 0
    for number in selected[:cap]:
        if budget <= 0:
            break
        try:  # one bad page must not sink the whole extraction
            text = all_pages[number - 1].extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            failed.setdefault(type(exc).__name__, []).append(number)
            continue
        if not text.strip():
            empty.append(number)
            continue
        if start_char > len(text):
            return f"⚠️ TOOL_ARG_ERROR: start_char={start_char} is past page {number}'s {len(text)} extracted characters."
        total_chars = len(text)
        text = text[start_char:]
        if len(text) > budget:
            text, cut_in, next_char = text[:budget], number, start_char + budget
        budget -= len(text)
        delivered.append(number)
        range_note = f"Characters [{start_char}:{start_char + len(text)}) of {total_chars} extracted characters."
        if start_char:
            range_note += f" Prefix [0:{start_char}) omitted from this view."
        if not text:
            range_note += " Empty end-of-page slice; no new characters delivered."
        sections.append(f"{range_note}\n--- page {number} ---\n{text}")
    attempted = len(delivered) + len(empty) + sum(len(numbers) for numbers in failed.values())

    def counted(numbers: List[int]) -> str:
        return f"{len(numbers)} page(s) {_page_spans(numbers)}"

    gaps: List[str] = []
    if cut_in:
        gaps.append(f"page {cut_in}: text cut at the {_OCR_PDF_MAX_CHARS}-char cap; the rest of this page is not delivered; "
                    f"continue with pages=\"{cut_in}\", start_char={next_char}")
    if empty and delivered:
        gaps.append(f"{counted(empty)}: no text layer (scanned/image-only?); render and view_image/vlm_query them")
    gaps += [f"{counted(numbers)}: extraction failed ({kind})" for kind, numbers in failed.items()]
    if unread := selected[attempted:]:
        why = " and ".join(([f"max_pages={cap}"] if len(selected) > cap else [])
                           + ([f"the {_OCR_PDF_MAX_CHARS}-char cap"] if attempted < min(cap, len(selected)) else []))
        gaps.append(f"{counted(unread)}: requested but not read ({why}); continue with pages=\"{_page_spans(unread)}\"")
    if outside := sorted(set(range(1, total + 1)) - set(selected)):
        gaps.append(f"{counted(outside)}: not requested")
    if past_end:
        gaps.append(f"page(s) {past_end}: requested past the last page ({total})")
    tail = ("\n\nGaps:\n" + "\n".join(f"- {gap}" for gap in gaps)) if gaps else ""
    if delivered:
        head = f"PDF text layer of {fp.name} (embedded text, not OCR): page(s) {_page_spans(delivered)} of {total} delivered."
        return f"{head}\n\n" + "\n\n".join(sections) + tail
    if empty:
        return (f"⚠️ OCR_PDF_SCANNED_UNAVAILABLE: no extractable text layer on {counted(empty)} of {total} "
                "(likely scanned/image-only). This tool reads the embedded text layer only; it does not OCR — "
                "render a page to an image and view_image it (or vlm_query it) instead." + tail)
    return (f"⚠️ OCR_PDF_UNAVAILABLE: text extraction failed on each of the {counted(selected[:attempted])} "
            f"attempted (of {total}); no text was delivered." + tail)


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

    frames_root = (resource_root_path(ctx, "artifact_store") / "video_frames").resolve(strict=False)
    frames_root.mkdir(parents=True, exist_ok=True)
    # One new directory per call: repeating a video and its timestamps never
    # overwrites the frames an earlier call returned.
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", fp.stem)[:60] or "video"
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"{stem}_", dir=frames_root))
    written: list[str] = []
    errors: list[str] = []
    for idx, ts in enumerate(raw_times, start=1):
        try:
            valid = math.isfinite(float(ts)) and float(ts) >= 0
        except ValueError:
            valid = False
        if not valid:
            errors.append(f"invalid timestamp {ts!r}")
            continue
        out = out_dir / f"frame_{idx:02d}_{ts}s.jpg"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", ts, "-i", str(fp), "-frames:v", "1", "-n", str(out)]
        try:
            from ouroboros.tools.shell import _tracked_subprocess_run

            res = _tracked_subprocess_run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=_VIDEO_FRAME_TIMEOUT_SEC
            )
        except subprocess.TimeoutExpired:
            errors.append(f"{ts}s: timeout")
            out.unlink(missing_ok=True)  # only this call's incomplete frame
            continue
        if res.returncode != 0 or not out.exists():
            errors.append(f"{ts}s: {(res.stderr or 'ffmpeg failed').strip()[:200]}")
            out.unlink(missing_ok=True)
            continue
        written.append(str(out))
    if not written:
        try:
            out_dir.rmdir()  # nothing was produced: leave no empty per-call directory behind
        except OSError:
            pass
        detail = "; ".join(errors[:5]) or "no frames produced"
        return f"⚠️ EXTRACT_VIDEO_FRAMES_UNAVAILABLE: {detail}"
    note = f"\nWarnings: {'; '.join(errors[:5])}" if errors else ""
    return (f"Extracted video frame(s) into {out_dir} (a new directory for this call; earlier extractions are kept):\n"
            + "\n".join(written) + note)


def _pdf_text_schema(name: str, lead: str) -> dict:
    """ocr_pdf (historical name) and extract_pdf_text share one handler and contract."""
    return {
        "name": name,
        "description": (
            lead + " Reports every omitted page and its reason. Resume a partial page with pages and the returned "
            "start_char; reads use current PDF bytes. This is not OCR: no text layer returns "
            "OCR_PDF_SCANNED_UNAVAILABLE. Render scanned pages and use view_image or vlm_query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local PDF path (inside the active workspace / task drive / artifact store / attachments / uploads)."},
                "pages": {"type": "string", "description": "Optional 1-based pages, e.g. '3', '2-5', '7-' or '1,4-6'. Default: all pages up to max_pages."},
                "start_char": {"type": "integer", "minimum": 0, "default": 0, "description": "Resume at this character of one selected page's extracted text. A nonzero offset requires pages selecting exactly one page."},
                "max_pages": {"type": "integer", "description": f"Cap pages read from the selection (default {_OCR_PDF_MAX_PAGES}; pages past it are named)."},
            },
            "required": ["path"],
        },
    }


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(name="ocr_pdf", schema=_pdf_text_schema("ocr_pdf", (
            "Extract the embedded text layer of a local PDF (historical alias of "
            "extract_pdf_text).")), handler=_ocr_pdf, timeout_sec=120, alias_for="extract_pdf_text"),
        # ocr_pdf's read-only skip lives in safety.TOOL_POLICY; one declaration per tool, never both.
        ToolEntry(name="extract_pdf_text", schema=_pdf_text_schema(
            "extract_pdf_text", "Extract the embedded text layer of a local PDF."),
            handler=_ocr_pdf, timeout_sec=120, safety_policy="skip"),
        ToolEntry(
            name="youtube_transcript",
            schema={
                "name": "youtube_transcript",
                "description": (
                    "Fetch a YouTube video's caption track as text by URL or video id; captions contain no visual evidence. Returns the "
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
                    "Each call writes its frames into a new directory under artifact_store/video_frames (earlier "
                    "extractions are kept) and returns their paths for view_image. "
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
