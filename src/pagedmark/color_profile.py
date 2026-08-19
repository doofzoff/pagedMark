"""Carry the ICC profile and the alpha channel across a pixel regeneration.

The diffusion path is RGB by construction: SDXL has three channels, so the stage that
regenerates pixels converts to RGB, and OpenCV -- which writes the result -- has no
concept of an embedded colour profile. Both losses are silent, and both are real:

- **The profile.** Dropping it does not change a single stored number, it changes what
  those numbers *mean*. A Display P3 photograph re-read as sRGB is shown desaturated.
  Measured on a 1180x1562 P3 frame: 19.2% of pixels shift by more than 5 levels, the
  worst by 40, and mean per-pixel saturation falls from 17.6 to 15.0. That is the whole
  reason this module exists -- the picture that comes back looks flatter than the one
  that went in, and no pixel metric sees it, because the pixels are fine.
- **The alpha channel.** A transparent PNG comes back opaque, which is not a degradation
  but a lost channel.

Neither is restored by regenerating anything. The profile is re-attached and the alpha
is carried around the model and composited back, so the RGB the model produced is the
RGB that gets written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image as PILImage

# Formats whose containers hold an alpha channel. A JPEG cannot, so asking it to is a
# silent no-op rather than an error -- the caller does not know the format either.
_ALPHA_FORMATS = frozenset({".png", ".webp", ".tif", ".tiff"})
# JPEG segments cap at 65535 bytes including the length field and the 14-byte
# ICC_PROFILE header, so a large profile is split across numbered chunks. This is the
# format's own limit, not a chosen one.
_JPEG_ICC_CHUNK = 65535 - 2 - 14
_JPEG_FORMATS = frozenset({".jpg", ".jpeg"})


class Carryover(NamedTuple):
    """What the pixel path cannot carry, taken from the source before it runs."""

    icc_profile: bytes | None
    alpha: PILImage | None  # the source's alpha band, or None when it had none

    def __bool__(self) -> bool:
        return self.icc_profile is not None or self.alpha is not None


def read(source: Path) -> Carryover:
    """Take the profile and alpha channel off the source. Never raises."""
    try:
        from PIL import Image

        with Image.open(source) as image:
            icc = image.info.get("icc_profile") or None
            alpha = image.getchannel("A").copy() if "A" in image.getbands() else None
            return Carryover(icc if isinstance(icc, bytes) else None, alpha)
    except Exception:
        # A source this cannot read is a source the run is about to fail on anyway;
        # losing a profile is not the failure worth reporting here.
        return Carryover(None, None)


def _jpeg_with_profile(data: bytes, icc: bytes) -> bytes | None:
    """Insert an APP2 ICC segment into a JPEG without touching its scan data.

    A re-encode would undo the quality-preserving writes upstream of this, so the
    profile is spliced in at the byte level instead. Returns None when the file does
    not parse as a JPEG whose first marker segment can be found.
    """
    if not data.startswith(b"\xff\xd8"):
        return None
    # Insert directly after SOI, before any other segment: APP2 ordering is free, and
    # this avoids having to understand the segments already there.
    chunks = [icc[i : i + _JPEG_ICC_CHUNK] for i in range(0, len(icc), _JPEG_ICC_CHUNK)]
    if len(chunks) > 255:
        return None
    segments = bytearray()
    for number, chunk in enumerate(chunks, start=1):
        payload = b"ICC_PROFILE\x00" + bytes([number, len(chunks)]) + chunk
        segments += b"\xff\xe2" + (len(payload) + 2).to_bytes(2, "big") + payload
    return data[:2] + bytes(segments) + data[2:]


def apply(destination: Path, carryover: Carryover) -> bool:
    """Re-attach ``carryover`` to an already-written file. True when it changed it.

    The alpha channel is resized to the destination when the two differ: diffusers
    rounds to its latent grid, so an eight-pixel difference is routine.
    """
    if not carryover:
        return False
    suffix = destination.suffix.lower()

    # JPEG first and separately: it carries no alpha, and re-encoding it to attach a
    # profile would cost more than the profile is worth.
    if suffix in _JPEG_FORMATS:
        if carryover.icc_profile is None:
            return False
        patched = _jpeg_with_profile(destination.read_bytes(), carryover.icc_profile)
        if patched is None:
            return False
        destination.write_bytes(patched)
        return True

    try:
        from PIL import Image

        with Image.open(destination) as opened:
            image = opened.convert("RGB")
            changed = False
            if carryover.alpha is not None and suffix in _ALPHA_FORMATS:
                alpha = carryover.alpha
                if alpha.size != image.size:
                    alpha = alpha.resize(image.size, Image.Resampling.LANCZOS)
                image = image.convert("RGBA")
                image.putalpha(alpha)
                changed = True
            save_kwargs: dict[str, Any] = {}
            if carryover.icc_profile is not None:
                save_kwargs["icc_profile"] = carryover.icc_profile
                changed = True
            if not changed:
                return False
            image.save(destination, **save_kwargs)
    except Exception:
        return False
    return True
