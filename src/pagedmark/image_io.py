"""Unicode-safe cv2 image IO (issue #17).

``cv2.imread`` / ``cv2.imwrite`` pass the path to the platform C runtime, which
on Windows uses the narrow (ANSI) code-page API and therefore fails on paths
containing non-ASCII characters (Chinese, Cyrillic, ...). The symptom is a
``can't open/read file`` warning and a ``None`` decode even though the file
exists.

These wrappers route through numpy buffers instead: ``np.fromfile`` /
``ndarray.tofile`` open the path in Python (full Unicode), and
``cv2.imdecode`` / ``cv2.imencode`` do the codec work. The decoded/encoded
bytes are byte-for-byte identical to ``imread`` / ``imwrite``. On macOS/Linux
cv2 already accepts UTF-8 paths, so the wrappers are behavior-neutral there.

cv2/numpy are imported lazily inside the functions so importing this module
stays cheap in a bare environment (matching the rest of the package).
"""

# cv2 ships no type stubs; mirror the pragma used by the other cv2-using modules.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


def imread(path: str | Path, flags: int | None = None) -> NDArray[Any] | None:
    """Unicode-safe ``cv2.imread`` with a Pillow fallback for HEIC/AVIF.

    ``flags`` defaults to ``cv2.IMREAD_COLOR`` (same as ``cv2.imread``). Returns
    ``None`` when the file is missing or cannot be decoded, matching ``cv2.imread``
    semantics so existing ``if img is None`` checks keep working.

    OpenCV cannot decode HEIC/AVIF (and some other containers), so when its decode
    returns None we fall back to Pillow (:func:`_pil_read`): AVIF is native in modern
    Pillow, HEIC works when the optional ``pillow-heif`` plugin is installed. This lets
    the pixel path (visible removal) read the same formats the metadata path already
    scans; normal PNG/JPEG/WebP never reach the fallback, so they are unaffected.
    """
    import cv2
    import numpy as np

    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    # cv2.imdecode returns None on an undecodable container (HEIC/AVIF); the type stub
    # omits that, hence the ignore.
    if img is not None:  # pyright: ignore[reportUnnecessaryComparison]
        return img
    return _pil_read(path, flags)


_heif_registered = False


def _pil_read(path: str | Path, flags: int) -> NDArray[Any] | None:
    """Decode via Pillow (HEIC/AVIF and any other Pillow-readable container) into the
    cv2 layout ``flags`` implies: grayscale, 3-channel BGR, or BGRA when the source has
    alpha and ``IMREAD_UNCHANGED`` was requested. Returns None if Pillow (with the
    optional HEIF plugin) still cannot open it. No EXIF auto-rotation, matching cv2."""
    import cv2
    import numpy as np

    try:
        from PIL import Image
    except Exception:
        return None
    _register_heif()
    try:
        with Image.open(path) as im:
            im.load()
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.asarray(im.convert("L"))
            has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            if flags == cv2.IMREAD_UNCHANGED and has_alpha:
                return cv2.cvtColor(np.asarray(im.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
            return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read ``path`` as a BGR ndarray, raising instead of returning ``None``.

    :func:`imread` keeps cv2's ``None``-on-failure contract because the removal paths
    branch on it. Scripts and tests want the opposite -- fail loudly at the read -- so
    they call this. Each vendor engine used to carry its own verbatim copy.
    """
    image = imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def to_bgr(image: NDArray[Any]) -> NDArray[Any]:
    """Return a 3-channel BGR view of ``image``, promoting grayscale and BGRA.

    The cv2-based engines (sparkle + the text-mark detectors/localizers) assume a
    3-channel BGR array for their channel reductions (``mean(axis=2)``, the top-hat
    glyph extraction). A 2D grayscale or 4-channel BGRA input -- a real Gemini-app
    export is opaque RGBA -- would otherwise crash or mis-broadcast.
    Centralizes the shape coercion that was inlined across the engines. A 3-channel
    input is returned unchanged (no copy).
    """
    import cv2

    if image.ndim == 2 or image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _register_heif() -> None:
    """Register the HEIF+AVIF Pillow opener/saver via libheif (idempotent, best-effort)."""
    global _heif_registered
    if _heif_registered:
        return
    _heif_registered = True
    import contextlib

    with contextlib.suppress(Exception):
        import pillow_heif  # pyright: ignore[reportMissingImports]

        pillow_heif.register_heif_opener()


# Containers cv2 cannot encode -> written via Pillow (pillow-heif).
_HEIF_WRITE_EXTS = {".heic", ".heif", ".avif"}


def _encode_params(ext: str) -> list[int]:
    """cv2 encode params that PRESERVE quality. The removal only touches the mark's
    footprint, so the container re-encode must not degrade the untouched pixels:
    JPEG at quality 100 with 4:4:4 chroma (no subsampling), WebP at max. Lossless
    containers (PNG/BMP/TIFF) need no params. getattr-guarded so an older OpenCV
    build without the chroma/subsampling flags still gets quality 100."""
    import cv2

    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        cq = getattr(cv2, "IMWRITE_JPEG_CHROMA_QUALITY", None)
        if cq is not None:
            params += [cq, 100]
        sf = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        sf444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
        if sf is not None and sf444 is not None:
            params += [sf, sf444]
        return params
    if ext == ".webp":
        # cv2 WebP: quality 1-100 is LOSSY; a value > 100 selects LOSSLESS mode.
        # "work with originals" requires lossless so a mark-removal re-encode does not
        # degrade the untouched pixels the fill composites over (regression: q100
        # round-tripped a random image at maxdiff ~230, q101 at 0).
        return [cv2.IMWRITE_WEBP_QUALITY, 101]
    return []


def _pil_write(path: str | Path, img: NDArray[Any]) -> bool:
    """Encode HEIC/AVIF via Pillow (+pillow-heif) at high quality -- cv2 has no encoder
    for them. BGR / BGRA in; returns False if Pillow (with the plugin) cannot save."""
    import cv2
    import numpy as np
    from PIL import Image

    _register_heif()
    if img.ndim == 3 and img.shape[2] == 4:
        arr, mode = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA), "RGBA"
    else:
        arr, mode = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2RGB), "RGB"
    try:
        Image.fromarray(np.ascontiguousarray(arr), mode).save(str(path), quality=100)
        return True
    except Exception:
        return False


def imwrite(path: str | Path, img: NDArray[Any]) -> bool:
    """Unicode-safe image write that PRESERVES the input format at max quality.

    Format is taken from the path extension. HEIC/AVIF (which cv2 cannot encode) go
    through Pillow; everything else through cv2 with quality-preserving params (see
    :func:`_encode_params`) so a lossy re-encode of the untouched pixels stays near-
    lossless. Returns ``True`` on success, ``False`` if the codec rejects the image or
    the path cannot be written (matching ``cv2.imwrite``, never raising)."""
    import cv2

    ext = (Path(path).suffix or ".png").lower()
    if ext in _HEIF_WRITE_EXTS:
        return _pil_write(path, img)
    try:
        ok, buf = cv2.imencode(ext, img, _encode_params(ext))
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


# Container extensions that carry an alpha channel (for read/write-with-alpha).
ALPHA_FORMATS = {".png", ".webp", ".heic", ".heif", ".avif"}


def read_bgr_and_alpha(path: str | Path) -> tuple[NDArray[Any] | None, NDArray[Any] | None]:
    """Read an image preserving its alpha channel separately.

    Returns ``(bgr, alpha)`` where ``alpha`` is a single-channel ndarray when the
    source has transparency, else ``None``. Grayscale inputs are promoted to BGR.
    Returns ``(None, None)`` if the image cannot be decoded.
    """
    import cv2

    image = imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        return image[:, :, :3].copy(), image[:, :, 3].copy()
    return image, None


def write_bgr_with_alpha(path: str | Path, bgr: NDArray[Any], alpha: NDArray[Any] | None) -> bool:
    """Write BGR (with optional alpha) to ``path``. Returns ``imwrite``'s success flag.

    When ``alpha`` is provided and the output extension supports it, the original
    alpha plane is rejoined unchanged. The watermark region is NOT made transparent:
    the fill reconstructs real pixels there, so zeroing alpha would punch a
    transparent hole that renders as a white box on any non-transparent viewer
    (issue #30). Preserving the input alpha keeps genuinely transparent backgrounds
    intact without inventing new holes.

    Returning the flag is load-bearing: :func:`imwrite` is contractually non-raising, so
    this is the ONLY signal a caller gets that the file was not created. Discarding it let
    a failed write (read-only directory, full disk) run on to ``output.stat()`` and die
    with a bare ``FileNotFoundError`` traceback instead of a readable error.
    Regression: ``tests/test_cli_robustness.py::TestFailedWriteIsReported``.
    """
    import numpy as np

    if alpha is None or Path(path).suffix.lower() not in ALPHA_FORMATS:
        return imwrite(path, bgr)
    return imwrite(path, np.dstack([bgr, alpha]))
