"""Pixel evidence of native typing: what the compositor and the renderer each showed.

Moved whole from ``native_window.py`` so both instrument modules stay inside the size
band. These pure measurements run driver-side only; the launcher observer never calls
them, so the hooks copy of ``native_window.py`` does not need this file.
"""
from __future__ import annotations


def pixels_readable(shot: dict | None) -> bool | None:
    """Not a uniform dark surface: anti-aliased text and controls leave many colours. People still read the PNG."""
    if not shot or not shot.get("ok"):
        return None
    return shot["distinct_colors"] >= 32 and shot["dark_fraction"] < 0.995


def drawn_rgb(image):
    """Fully transparent pixels have no visible colour, even if their stored RGB channels change."""
    from PIL import Image

    rgba = image.convert("RGBA")
    rgb = Image.new("RGB", rgba.size)
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    return rgb


def typing_delta(paths, box, window=None):
    """Map the CSS field to the WK bitmap or the client area of the shadow-free NSWindow image."""
    from PIL import Image, ImageChops

    left, top, right, bottom, css_w, css_h, start, width, advance = box
    if not (0 <= left < right <= css_w and 0 <= top < bottom <= css_h):
        raise ValueError("the composer was not fully visible")
    frames = []
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            if i == 0:
                size = image.size
                if window:
                    fw, fh = window["frame"][2:]
                    x, y, vw, vh = window["viewport"]
                else:
                    fw, fh = vw, vh = css_w, css_h
                    x = y = 0
                if min(fw, fh, vw, vh) <= 0 or abs(size[0] * fh / fw - size[1]) > 2:
                    raise ValueError("capture aspect ratio disagrees with its native bounds")
                sx, sy = size[0] / fw, size[1] / fh
                scale = sx * vw / css_w
                crop = (round(sx * (x + left * vw / css_w)), round(sy * (y + top * vh / css_h)),
                        round(sx * (x + right * vw / css_w)), round(sy * (y + bottom * vh / css_h)))
                if not (0 <= crop[0] < crop[2] <= size[0] and 0 <= crop[1] < crop[3] <= size[1]):
                    raise ValueError("the field could not be located in this image")
            if image.size != size:
                raise ValueError("the capture dimensions changed around input")
            frames.append(drawn_rgb(image.crop(crop)))
    def difference(a, b):
        return ImageChops.difference(a, b).convert("L").point(lambda level: 255 if level >= 12 else 0)
    quiet = frames[:3]
    noise = max(difference(a, b).histogram()[255] for i, a in enumerate(quiet) for b in quiet[i+1:])
    ink = difference(quiet[0], frames[3])
    for previous in quiet[1:]:
        ink = ImageChops.darker(ink, difference(previous, frames[3]))
    css_size = (max(1, round(right - left)), max(1, round(bottom - top)))
    def css_mask(mask):
        return mask.resize(css_size, Image.Resampling.BOX).point(lambda level: 255 if level >= 32 else 0)
    # The quiet field holds one word. Its already drawn strokes, translated by one measured
    # word and space, predict what the second copy should add in this very capture.
    x, dest, length = (round(v * scale) for v in (start, start + advance, width))
    first = quiet[0].crop((x, 0, x + length, ink.height))
    blank = quiet[0].crop((dest, 0, dest + length, ink.height))
    reference = Image.new("L", ink.size)
    reference.paste(difference(first, blank), (dest, 0))
    changed = css_mask(ink)
    match = echo_overlap(changed, css_mask(reference), tolerance=1)
    count, reference_count = ink.histogram()[255], reference.histogram()[255]
    above_noise = count > 3 * noise + 40 * scale * scale
    same_word = 0.6 * reference_count <= count <= 1.6 * reference_count and match >= 0.70
    return {"noise": noise, "change": count, "scale": scale, "image_size": size, "crop": crop,
            "reference_pixels": reference_count, "word_overlap": match,
            "visible_word": above_noise and same_word}, changed


def echo_overlap(window, renderer, tolerance=2):
    """Fraction of changed ink near the other image's changed ink, measured in both directions in CSS pixels."""
    from PIL import ImageChops, ImageFilter

    def fraction(a, b):
        count = a.histogram()[255]
        near = b.filter(ImageFilter.MaxFilter(2 * tolerance + 1))
        return ImageChops.darker(a, near).histogram()[255] / count if count else 0.0
    return min(fraction(window, renderer), fraction(renderer, window))


def window_colors(path):
    from PIL import Image

    with Image.open(path) as image:
        # Exclude the title bar, border and shadow. Count only central client content.
        crop = drawn_rgb(image.crop((image.width // 12, image.height // 6,
                                     11 * image.width // 12, 11 * image.height // 12)))
        crop.thumbnail((500, 350))
        dark = sum(crop.convert("L").histogram()[:18])
        return {"ok": True, "distinct_colors": len(crop.getcolors(crop.width * crop.height)),
                "dark_fraction": round(dark / (crop.width * crop.height), 5)}
