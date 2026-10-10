"""Byte-only image validation and representation; callers own file admission.

The one byte check for owner images (initial composition, each drained running-task
delivery, the legacy inline image) and for ``view_image``/``vlm_query``. Keep decoded
detail unless a caller supplies a concrete byte/dimension bound: no preview size is
imposed, because a 9000x24 screenshot at a 1600-pixel side keeps four rows. Original
bytes are never modified. A representation change carries a human-readable note;
callers retain the original (``retain_original``) and show that note beside the pixels.

- A strictly decodable PNG, JPEG (an MPO gain-map JPEG included), GIF or WebP within
  the byte budget keeps its bytes and dimensions.
- BMP and other formats the installed decoder reads become PNG with the same pixels
  and alpha.
- A successful tolerant decode is re-encoded with a "partially recovered" note: the
  image may be incomplete or contain decoder fill, so recovered detail is uncertain;
  uniform pixels cannot prove loss, and missing details are not restored.
- Unreadable (``IMAGE_UNDECODABLE``), unrecognized (``IMAGE_FORMAT_UNSUPPORTED``,
  evidence of neither damage nor model capability) and over-budget
  (``VLM_IMAGE_TOO_LARGE``) bytes are refused; callers keep the original and write a
  per-file note instead of dropping it or sending corrupt pixels.
- A byte-recognized format without a local codec (HEIC, for example) passes within
  budget unvalidated, with a codec-gap note: codec absence proves neither damage nor
  remote incapability. Without the decoder at all, recognized bytes within budget
  pass unchanged with an unverified note.

Host budgets belong to callers: 8 MiB per automatic attachment, 20 MiB per explicit
read and 6 MiB per explicit representation. Every strict and tolerant decode here pins
Pillow's process-global truncation switch under one lock; a decoder outside this
helper could still observe the tolerant window.
"""
from __future__ import annotations

from hashlib import sha256
import io
import pathlib
import threading
from dataclasses import dataclass

_DECODE_LOCK = threading.RLock()
_NATIVE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


class ImagePayloadTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime: str
    note: str = ""


def image_mime(raw: bytes) -> str:
    for magic, mime in ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"),
                        (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"), (b"BM", "image/bmp"),
                        (b"II*\x00", "image/tiff"), (b"MM\x00*", "image/tiff")):
        if raw.startswith(magic):
            return mime
    from ouroboros.chat_uploads import detect_media

    mime, kind = detect_media(raw, "")
    return mime if kind == "image" else ""


def retain_original(directory: pathlib.Path, raw: bytes, name: str) -> pathlib.Path:
    """Keep admitted bytes under a content-addressed name before any conversion.

    A reused copy is touched: the ``uploads/views`` age sweep then measures its latest use.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sha256(raw).hexdigest()}_{pathlib.Path(name).name or 'image'}"
    if path.exists():
        if path.read_bytes() != raw:
            raise OSError("retained image identity differs") from None
        path.touch()
    else:
        from ouroboros.utils import write_bytes_atomic
        write_bytes_atomic(path, raw)
    return path


def _decode(Image, ImageFile, raw: bytes, *, tolerant: bool):
    """One decode with Pillow's process-global truncation switch pinned and restored."""
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = tolerant
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            # MPO is a JPEG stream with secondary images (e.g. an HDR gain map), not animation.
            fmt = opened.format
            frames = 1 if fmt == "MPO" else getattr(opened, "n_frames", 1)
            return opened.copy(), "image/jpeg" if fmt == "MPO" else Image.MIME.get(fmt, ""), frames > 1
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def prepare_image_bytes(raw: bytes, *, max_bytes: int, max_side: int | None = None,
                        reason: str = "host image payload budget") -> PreparedImage:
    """Validate, convert available codecs, and fit only explicitly supplied bounds.

    Without a decoder, recognized bytes remain usable on an unknown route within
    the byte budget; codec availability is not evidence about a remote provider,
    and an unmeasured side is not evidence that it exceeds ``max_side``.
    Pillow's existing decompression guard remains a refusal, not corruption recovery.
    """
    mime = image_mime(raw)
    try:
        from PIL import Image, ImageFile, UnidentifiedImageError
    except ImportError:
        if not mime:
            raise ValueError("IMAGE_UNDECODABLE: image type is unknown and the image decoder is unavailable")
        if len(raw) > max_bytes:
            raise ImagePayloadTooLarge(f"VLM_IMAGE_TOO_LARGE: cannot fit {reason} ({max_bytes} bytes); "
                                       "image decoder unavailable")
        unmeasured = f" Dimensions were not checked against {reason}." if max_side is not None else ""
        return PreparedImage(raw, mime, "Image decoder unavailable; bytes preserved without validation "
                             "or conversion." + unmeasured)

    notes = []
    # Serialize this helper's strict and recovery reads around the global switch.
    with _DECODE_LOCK:
        try:
            decoded, decoded_mime, animated = _decode(Image, ImageFile, raw, tolerant=False)
            mime = decoded_mime or mime
        except Image.DecompressionBombError as exc:
            raise ValueError(f"IMAGE_MEMORY_LIMIT: {exc}; original retained") from exc
        except Exception as strict_error:  # noqa: BLE001 - decoder plugins raise arbitrary types on bad bytes
            if mime and isinstance(strict_error, UnidentifiedImageError) and mime not in Image.MIME.values():
                if len(raw) > max_bytes:
                    raise ImagePayloadTooLarge(f"VLM_IMAGE_TOO_LARGE: cannot fit {reason} ({max_bytes} bytes); "
                                               "local image codec unavailable") from strict_error
                unmeasured = f" Dimensions were not checked against {reason}." if max_side is not None else ""
                return PreparedImage(raw, mime, f"Local codec unavailable for {mime}; original bytes preserved "
                                     "without validation or conversion; damage is unknown." + unmeasured)
            if not mime and isinstance(strict_error, UnidentifiedImageError):
                # Without a recognized signature or decoder, neither image type
                # nor damage is established. Do not infer remote capability.
                raise ValueError("IMAGE_FORMAT_UNSUPPORTED: the installed image decoder cannot read this "
                                 "format; the file is not shown to be damaged; original retained") from strict_error
            try:
                decoded, decoded_mime, _animated = _decode(Image, ImageFile, raw, tolerant=True)
            except Exception as exc:  # noqa: BLE001 - same decoder boundary
                raise ValueError(f"IMAGE_UNDECODABLE: {strict_error}; original retained") from exc
            mime, animated = decoded_mime or mime, False
            notes.append("Partially recovered image from tolerant decoding; it may be incomplete or contain "
                         "decoder fill. Recovered detail is uncertain; missing details were not restored.")

    exceeds_side = max_side is not None and max(decoded.size) > max_side
    if not notes and mime in _NATIVE_MIMES and len(raw) <= max_bytes and not exceeds_side:
        return PreparedImage(raw, mime)
    if mime not in _NATIVE_MIMES:
        notes.append(f"Converted {mime or 'decoded image'} to PNG with the available decoder.")
    if animated:
        notes.append("Only the first frame is represented; the original animation remains available.")
    original_size = decoded.size
    if exceeds_side:
        decoded.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        notes.append(f"Reduced dimensions for {reason} (maximum side {max_side} pixels).")
    # Lossless conversion first, preserving alpha and the decoded pixels.
    if decoded.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
        decoded = decoded.convert("RGBA" if "A" in decoded.getbands() else "RGB")
    output = io.BytesIO()
    decoded.save(output, format="PNG")
    if len(output.getvalue()) <= max_bytes:
        return PreparedImage(output.getvalue(), "image/png", " ".join(notes))

    notes.append(f"Reduced representation to fit {reason} ({max_bytes} bytes).")
    has_alpha = "A" in decoded.getbands() or "transparency" in decoded.info
    while True:
        if has_alpha:
            output = io.BytesIO()
            decoded.save(output, format="PNG", optimize=True)
            data, out_mime = output.getvalue(), "image/png"
            if len(data) <= max_bytes:
                break
        else:
            for quality in (95, 85, 75, 65, 55):
                output = io.BytesIO()
                decoded.convert("RGB").save(output, format="JPEG", quality=quality, optimize=True)
                data, out_mime = output.getvalue(), "image/jpeg"
                if len(data) <= max_bytes:
                    break
            if len(data) <= max_bytes:
                break
        if max(decoded.size) <= 1:
            raise ImagePayloadTooLarge(f"VLM_IMAGE_TOO_LARGE: cannot fit {reason} ({max_bytes} bytes)")
        decoded.thumbnail(tuple(max(1, int(side * .75)) for side in decoded.size), Image.Resampling.LANCZOS)
    if decoded.size != original_size:
        notes.append(f"Dimensions: {original_size[0]}×{original_size[1]} → {decoded.width}×{decoded.height}.")
    return PreparedImage(data, out_mime, " ".join(notes))
