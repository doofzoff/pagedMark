"""Detect Adobe TrustMark invisible watermarks.

TrustMark (github.com/adobe/trustmark, MIT) is the open, keyless image watermark
behind Adobe "Durable Content Credentials": when a C2PA manifest is stripped, a
TrustMark soft binding can still re-link the asset to its manifest in a
repository. Unlike SynthID it has a PUBLIC decoder with no secret key, so a
TrustMark-stamped image can be identified locally. Adobe's shipping products use
Variant P (the ``com.adobe.trustmark.P`` soft-binding ``alg``); this wrapper
loads that model.

Optional dependency (extra: ``trustmark``); the model weights download on first
use. ``detect_trustmark`` returns None when the package is absent. This detects
provenance (Adobe Content Credentials), NOT AI generation as such -- TrustMark
also marks human-authored content -- so callers should treat it as a watermark
signal, not proof of AI origin.
"""

# trustmark ships no type stubs; relax untyped-library diagnostics for this thin
# wrapper module only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Adobe ships Variant P in production (com.adobe.trustmark.P).
_MODEL_TYPE = "P"
# Schema 3 (BCH_3) is below the detector's measured precision threshold; the
# calibration history is canonical in docs/module-internals.md.
_SUPPORTED_SCHEMAS = frozenset({0, 1, 2})
# Lazily constructed singleton -- model load + first-use download is expensive.
# Guarded by a lock so concurrent callers don't double-construct/double-download.
_tm: Any = None
_tm_lock = threading.Lock()


def is_available() -> bool:
    """True if the optional ``trustmark`` package is installed."""
    from .optional_deps import module_available

    return module_available("trustmark")


def _decoder() -> Any:
    global _tm
    if _tm is None:
        with _tm_lock:
            if _tm is None:
                from trustmark import TrustMark

                _tm = TrustMark(verbose=False, model_type=_MODEL_TYPE)
    return _tm


# JPEG quality for the false-positive durability gate (see detect_trustmark).
# Deliberately mild: a genuine TrustMark survives far harsher. The round-trip
# still needs payload and schema validation because content-correlated false
# positives can survive this compression level.
_REENCODE_QUALITY = 95


def detect_trustmark(image_path: Path) -> str | None:
    """Return a TrustMark scheme note if a *durable* TrustMark watermark is
    decoded, else None.

    Returns e.g. ``"Adobe TrustMark (variant P, schema 0)"`` when the decoder
    reports the watermark present AND it survives a mild JPEG re-encode, or None
    if it is absent, the optional ``trustmark`` package is not installed, or the
    image cannot be read/decoded.

    **False-positive gate.** TrustMark's ``wm_present`` flag is a BCH
    error-correction validity check, which spuriously validates on a small
    fraction of un-watermarked images -- content-correlated, so AI-generated
    textures trip it more often than camera photos (verified 2026-05-29 on real
    files: the false "detections" were on Gemini / OpenAI / Doubao output that
    cannot carry Adobe's watermark, and decoded a random-bytes secret). A genuine
    TrustMark is a *durable* soft binding engineered to survive re-encoding (that
    is its entire purpose once C2PA is stripped), so we re-decode after a mild
    JPEG round-trip and require the same binary payload and schema both times.
    Only the calibrated schemas 0-2 count as high-precision positives.
    """
    if not is_available():
        return None
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            cover = img.convert("RGB")
        decoder = _decoder()
        wm_secret, wm_present, wm_schema = decoder.decode(cover, "binary")
        if not wm_present:
            return None
        if wm_schema not in _SUPPORTED_SCHEMAS:
            logger.debug(
                "TrustMark decode for %s used weak schema %s; treating as false positive",
                image_path,
                wm_schema,
            )
            return None
        if not _survives_reencode(decoder, cover, wm_secret, wm_schema):
            logger.debug("TrustMark decode for %s did not survive re-encode; treating as false positive", image_path)
            return None
    except Exception as exc:  # model download / decode failure / unreadable image
        logger.debug("TrustMark decode failed for %s: %s", image_path, exc)
        return None
    return f"Adobe TrustMark (variant {_MODEL_TYPE}, schema {wm_schema})"


def _survives_reencode(decoder: Any, cover: Any, payload: str, schema: int) -> bool:
    """True if the same watermark re-decodes after a mild JPEG round-trip."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    cover.save(buffer, "JPEG", quality=_REENCODE_QUALITY)
    buffer.seek(0)
    with Image.open(buffer) as reencoded:
        reencoded_payload, present, reencoded_schema = decoder.decode(reencoded.convert("RGB"), "binary")
    return bool(present) and reencoded_schema == schema and reencoded_payload == payload
