# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false
"""The complete pixel-forensics layer for one image.

STATUS

Independent from provenance verdicts, removal, and the CLI. Consumers use the
versioned :meth:`PixelEvidence.to_dict` boundary; feature extraction failures are
reported per family without discarding successful measurements.

WHAT IS MEASURED

One decode, then six families of scale-robust statistics over it:

* ``dct`` -- AC coefficient histograms over the 8x8 block DCT, plus the deviation of
  leading digits from Benford's law.
* ``fft`` -- radial band energies of the log-magnitude spectrum, plus the
  color-filter-array periodicity peaks a demosaiced camera capture leaves.
* ``noise`` -- standard deviation and kurtosis of a high-pass residual.
* ``ela`` -- error level after a quality-90 JPEG re-save.
* ``gradient`` -- gradient-magnitude histogram and Laplacian variance.
* ``color`` -- 4x4x4 RGB histogram, mean saturation, mean value.

and, in ``artifacts``, the spatial layer those statistics are computed from: a
64-bit perceptual hash, a 128px JPEG thumbnail, and coarse ELA, noise-residual and
FFT-phase maps.

THE ARTIFACTS ARE NOT AGGREGATES

Everything above ``artifacts`` is a scalar or a fixed-length histogram, and an image
cannot be reconstructed from those. ``artifacts`` is different in kind: a thumbnail
is a picture, a perceptual hash identifies one, and the coarse maps carry layout.
Collecting them makes a record that identifies the source image, so a caller storing
or forwarding them is handling image content, not statistics about it. That is why
they are a separate field and not merged into the families.

REQUIREMENTS

Needs the ``pixels`` extra (numpy). Guard a call with :func:`is_available` when the
caller must not hard-depend on it.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pagedmark._internal.schema import require_schema_version

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Analysis resolution. Every statistic here is scale-robust, and a 2048px cap keeps
# the FFT and the sliding-window residual bounded on a 100 MP input.
MAX_SIDE = 2048
# The eight lowest-frequency AC positions of the 8x8 block DCT, zig-zag order.
AC_POSITIONS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 1), (1, 2), (0, 3))
FFT_BANDS = 8
# A Bayer CFA shows as symmetric peaks at half the Nyquist on the diagonals.
BAYER_OFFSETS = ((1, 1), (1, -1))
INSTALL_HINT = "install the pixel extra: uv add 'pagedmark[pixels]'"
PIXEL_EVIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PixelEvidence:
    """Pixel statistics for one image, and the spatial artifacts behind them.

    ``decode`` carries the source dimensions, or ``{"error": ...}`` when the image
    could not be decoded -- in which case every other field is empty. A family is also
    empty when the image is too small for it (the block DCT needs 8x8, the FFT 32x32,
    the residual 3x3), so a caller must treat every field as optional rather than
    assume a fixed feature width.
    """

    path: Path
    decode: dict[str, Any]
    dct: dict[str, Any] = field(default_factory=dict[str, Any])
    fft: dict[str, Any] = field(default_factory=dict[str, Any])
    noise: dict[str, Any] = field(default_factory=dict[str, Any])
    ela: dict[str, Any] = field(default_factory=dict[str, Any])
    gradient: dict[str, Any] = field(default_factory=dict[str, Any])
    color: dict[str, Any] = field(default_factory=dict[str, Any])
    # Identifies the source image; see the module note. Empty unless asked for.
    artifacts: dict[str, Any] = field(default_factory=dict[str, Any])
    # Opt-in timings for callers measuring pipeline latency. Empty by default so
    # repeated evidence collection remains value-deterministic.
    timing_ms: dict[str, float] = field(default_factory=dict[str, float])

    @property
    def decoded(self) -> bool:
        """False when the source could not be decoded at all."""
        return "error" not in self.decode

    @property
    def status(self) -> str:
        """``complete``, ``partial`` for a failed family, or ``error`` on decode."""
        if not self.decoded:
            return "error"
        sections = (self.dct, self.fft, self.noise, self.ela, self.gradient, self.color, self.artifacts)
        return "partial" if any("error" in section for section in sections) else "complete"

    def to_dict(
        self,
        *,
        schema_version: int = PIXEL_EVIDENCE_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        """Return the selected JSON-safe transport schema without a local path."""
        schema_version = require_schema_version(
            schema_version,
            contract="pixel evidence",
            supported=(1,),
        )
        return {
            "schema_version": schema_version,
            "status": self.status,
            "decode": dict(self.decode),
            "dct": dict(self.dct),
            "fft": dict(self.fft),
            "noise": dict(self.noise),
            "ela": dict(self.ela),
            "gradient": dict(self.gradient),
            "color": dict(self.color),
            "artifacts": dict(self.artifacts),
            "timing_ms": dict(self.timing_ms),
        }


def is_available() -> bool:
    """True when the optional pixel dependencies are installed."""
    from pagedmark.optional_deps import module_available

    return module_available("numpy")


def _numpy() -> Any:
    from pagedmark.optional_deps import module_available

    if not module_available("numpy"):
        raise RuntimeError(f"Pixel evidence needs numpy -- {INSTALL_HINT}")
    import numpy as np

    return np


def _dct_matrix(np: Any, n: int = 8) -> Any:
    """Orthonormal n x n DCT-II basis: M[i, j] = cos(pi (2j + 1) i / 2n)."""
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * j + 1) * i / (2 * n))
    m[0, :] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


def read_gray(image_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Decode to float32 grayscale (and RGB for color stats), downscaled.

    Pillow, not cv2, and the source dimensions are recorded BEFORE the downscale.
    """
    np = _numpy()
    from PIL import Image

    from pagedmark import image_io

    try:
        image_io._register_heif()  # pyright: ignore[reportPrivateUsage]
        with Image.open(image_path) as img:
            info: dict[str, Any] = {"width": img.width, "height": img.height}
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
            gray = np.asarray(img.convert("L"), dtype=np.float32)
    except Exception as exc:
        logger.debug("pixel decode failed for %s: %s", image_path, exc)
        # Exception text from Pillow commonly embeds the absolute source path.
        # Keep that detail in the log, not in the pathless transport contract.
        return None, None, {"error": type(exc).__name__}
    return gray, rgb, info


def dct_features(gray: Any) -> dict[str, Any]:
    """AC coefficient histograms over the 8x8 block DCT + Benford deviation."""
    np = _numpy()
    height, width = gray.shape
    h8, w8 = height // 8 * 8, width // 8 * 8
    if h8 < 8 or w8 < 8:
        return {}
    basis = _dct_matrix(np)
    bins = np.linspace(-20.5, 20.5, 22)
    blocks = gray[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2)
    rows = basis[[row for row, _ in AC_POSITIONS]]
    columns = basis[[column for _, column in AC_POSITIONS]]
    coeff = np.einsum("ki,abij,kj->abk", rows, blocks, columns)
    hists = []
    lead_vals: list[Any] = []
    for index in range(len(AC_POSITIONS)):
        values = coeff[:, :, index].ravel()
        hists.append(np.histogram(values, bins=bins)[0].tolist())
        lead_vals.append(np.abs(values))
    out: dict[str, Any] = {"dct_ac_hist": hists}
    flat = np.abs(np.concatenate(lead_vals))
    flat = flat[flat >= 1]
    if flat.size > 100:
        leading = (flat / 10 ** np.floor(np.log10(flat))).astype(int)
        leading = leading[(leading >= 1) & (leading <= 9)]
        if leading.size > 100:
            observed = np.bincount(leading, minlength=10)[1:10] / leading.size
            benford = np.log10(1 + 1 / np.arange(1, 10))
            out["benford_mad"] = float(np.abs(observed - benford).mean())
    return out


def noise_residual_map(gray: Any) -> Any:
    """High-pass residual, the map the noise statistics are computed from."""
    np = _numpy()
    from numpy.lib.stride_tricks import sliding_window_view

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return None
    kernel = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
    height, width = gray.shape
    # kernel is float64, so the residual is float64 like the unchunked form
    out = np.empty((height - 2, width - 2), dtype=np.float64)
    # Row-chunked: the (window * kernel) temporary is ~150 MB at 2048px if
    # materialized whole. Per-element 9-tap sums are computed in the same order,
    # so the result is bit-identical to the unchunked form.
    for y0 in range(0, height - 2, 256):
        y1 = min(y0 + 256, height - 2)
        window = sliding_window_view(gray[y0 : y1 + 2], (3, 3))
        out[y0:y1] = (window * kernel).sum(axis=(-1, -2))
    return out


def noise_features(residual: Any) -> dict[str, Any]:
    """High-pass residual std and kurtosis."""
    flat = residual.ravel()
    std = float(flat.std())
    if std < 1e-9:
        return {"noise_std": 0.0, "noise_kurtosis": 0.0}
    z = (flat - flat.mean()) / std
    return {"noise_std": std, "noise_kurtosis": float((z**4).mean() - 3.0)}


def fft_decompose(gray: Any) -> tuple[Any, Any] | None:
    """Log-magnitude (fftshifted) and phase of the image spectrum."""
    np = _numpy()
    if min(gray.shape) < 32:
        return None
    spectrum = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    return np.log1p(np.abs(spectrum)), np.angle(spectrum)


def fft_features(mag: Any) -> dict[str, Any]:
    """Radial magnitude band energies (no phase) + CFA periodicity peaks."""
    np = _numpy()
    height, width = mag.shape
    cy, cx = height // 2, width // 2
    # 1D broadcast instead of an mgrid: saves ~160 MB of int64 temporaries at
    # 2048px. The squares are exact in float64 (values < 2^53), so band means
    # are identical to the mgrid form.
    r2y = (np.arange(height, dtype=np.float64) - cy) ** 2
    r2x = (np.arange(width, dtype=np.float64) - cx) ** 2
    radius = np.sqrt(r2y[:, None] + r2x[None, :])
    r_max = radius.max()
    bands = []
    for index in range(FFT_BANDS):
        mask = (radius >= r_max * index / FFT_BANDS) & (radius < r_max * (index + 1) / FFT_BANDS)
        bands.append(float(mag[mask].mean()) if mask.any() else 0.0)
    peaks = []
    for dy, dx in BAYER_OFFSETS:
        y, x = cy + dy * (height // 4), cx + dx * (width // 4)
        neighborhood = mag[y - 2 : y + 3, x - 2 : x + 3]
        peaks.append(float(neighborhood.max() - mag.mean()))
    return {"fft_band_energy": bands, "cfa_peaks": peaks, "cfa_peak": max(peaks)}


def ela_map(rgb: Any) -> Any:
    """Absolute per-pixel error after a quality-90 JPEG re-save."""
    np = _numpy()
    from PIL import Image

    try:
        buffer = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(buffer, "JPEG", quality=90)
        buffer.seek(0)
        resaved = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
    except Exception as exc:
        logger.debug("ELA re-save failed: %s", exc)
        return None
    if resaved.shape != rgb.shape:
        return None
    return np.abs(rgb - resaved).mean(axis=-1)


def ela_features(err: Any) -> dict[str, Any]:
    """Error-level stats after a quality-90 JPEG re-save."""
    np = _numpy()
    return {"ela_mean": float(err.mean()), "ela_p95": float(np.percentile(err, 95))}


def gradient_features(gray: Any) -> dict[str, Any]:
    np = _numpy()
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx**2 + gy**2)
    hist = np.histogram(mag, bins=10, range=(0, 255))[0].tolist()
    laplacian = np.gradient(gy, axis=0) + np.gradient(gx, axis=1)
    return {"gradient_hist": hist, "laplacian_var": float(laplacian.var())}


def color_features(rgb: Any) -> dict[str, Any]:
    np = _numpy()
    small = rgb[::4, ::4]  # decimate; the histogram is position-blind anyway
    bins = (small / 256 * 4).astype(int).clip(0, 3)
    index = bins[..., 0] * 16 + bins[..., 1] * 4 + bins[..., 2]
    hist = np.bincount(index.ravel(), minlength=64).tolist()
    mx = small.max(axis=-1)
    mn = small.min(axis=-1)
    saturation = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return {
        "color_hist_4x4x4": hist,
        "saturation_mean": float(saturation.mean()),
        "value_mean": float(mx.mean() / 255),
    }


def perceptual_hash(gray: Any) -> str:
    """64-bit DCT perceptual hash. Identifies an image; see the module note."""
    np = _numpy()
    from PIL import Image

    small = np.asarray(Image.fromarray(gray.astype(np.float32), mode="F").resize((32, 32), Image.Resampling.LANCZOS))
    basis = _dct_matrix(np, 32)
    low_basis = basis[:8]
    low = (low_basis @ small @ low_basis.T).ravel()[1:]  # drop DC
    bits = low > np.median(low)
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def _coarse(np: Any, arr: Any, side: int = 64) -> Any:
    """Downscale a 2D map to at most ``side`` on the long edge."""
    from PIL import Image

    height, width = arr.shape
    if max(height, width) <= side:
        return arr
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img.thumbnail((side, side), Image.Resampling.BILINEAR)
    return np.asarray(img)


def _array_payload(arr: Any) -> dict[str, Any]:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "base64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def spatial_artifacts(gray: Any, rgb: Any, *, ela: Any, residual: Any, phase: Any) -> dict[str, Any]:
    """Perceptual hash, thumbnail, and coarse ELA / residual / phase maps.

    These identify the source image rather than describe it -- see the module note.
    The maps are the ones the statistics were computed from, passed in rather than
    recomputed.
    """
    np = _numpy()
    from PIL import Image

    out: dict[str, Any] = {"phash": perceptual_hash(gray)}
    thumbnail = Image.fromarray(rgb.astype(np.uint8))
    thumbnail.thumbnail((128, 128), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, "JPEG", quality=70)
    out["thumbnail_jpeg_b64"] = base64.b64encode(buffer.getvalue()).decode("ascii")

    if ela is not None:
        out["ela_map"] = _array_payload(_coarse(np, ela))
    if residual is not None:
        clipped = np.clip(residual / 4.0, -1, 1)
        out["noise_residual"] = _array_payload(_coarse(np, (clipped * 127).astype(np.int8)))
    if phase is not None:
        out["fft_phase"] = _array_payload(_coarse(np, phase.astype(np.float32), 32))
    return out


def extract_pixel_evidence(image_path: Path, *, artifacts: bool = False, timings: bool = False) -> PixelEvidence:
    """Measure every pixel-statistic family for one image in a single decode.

    The image is decoded ONCE and the intermediate maps (high-pass residual, ELA
    error, FFT magnitude and phase) are computed once and shared, because the
    residual's sliding window and the ELA re-save are the two expensive steps and
    each family would otherwise redo them.

    A family that fails or does not apply is left empty rather than raising: an
    undecodable file, or one too small for the block DCT, still returns a
    :class:`PixelEvidence` whose ``decoded`` / empty fields say so. Missing numpy is
    the one hard error, since then nothing can be measured at all.

    Args:
        image_path: Path to the image. Any container Pillow can open.
        artifacts: Also return the spatial layer -- perceptual hash, thumbnail and
            coarse maps. Off by default: those identify the source image, so asking
            for them is a decision the caller makes explicitly.
        timings: Measure each stage and include rounded milliseconds in
            :attr:`PixelEvidence.timing_ms`.

    Returns:
        A :class:`PixelEvidence`.
    """
    started = time.perf_counter()
    stage_started = started
    measured: dict[str, float] = {}

    gray, rgb, info = read_gray(image_path)
    measured["decode"] = time.perf_counter() - stage_started
    if gray is None or rgb is None:
        measured["total"] = time.perf_counter() - started
        timing_ms = {name: round(seconds * 1000, 1) for name, seconds in measured.items()} if timings else {}
        return PixelEvidence(path=image_path, decode=info, timing_ms=timing_ms)

    families: dict[str, dict[str, Any]] = {}

    residual = None
    stage_started = time.perf_counter()
    try:
        residual = noise_residual_map(gray)
        families["noise"] = noise_features(residual) if residual is not None else {}
    except Exception as exc:
        logger.debug("pixel family noise failed for %s: %s", image_path, exc)
        families["noise"] = {"error": type(exc).__name__}
    measured["noise"] = time.perf_counter() - stage_started

    spectrum = None
    stage_started = time.perf_counter()
    try:
        spectrum = fft_decompose(gray)
        families["fft"] = fft_features(spectrum[0]) if spectrum is not None else {}
    except Exception as exc:
        logger.debug("pixel family fft failed for %s: %s", image_path, exc)
        families["fft"] = {"error": type(exc).__name__}
    measured["fft"] = time.perf_counter() - stage_started

    error = None
    stage_started = time.perf_counter()
    try:
        error = ela_map(rgb)
        families["ela"] = ela_features(error) if error is not None else {}
    except Exception as exc:
        logger.debug("pixel family ela failed for %s: %s", image_path, exc)
        families["ela"] = {"error": type(exc).__name__}
    measured["ela"] = time.perf_counter() - stage_started

    for name, compute in (
        ("dct", lambda: dct_features(gray)),
        ("gradient", lambda: gradient_features(gray)),
        ("color", lambda: color_features(rgb)),
    ):
        stage_started = time.perf_counter()
        try:
            families[name] = compute()
        except Exception as exc:  # one bad family must not lose the other five
            logger.debug("pixel family %s failed for %s: %s", name, image_path, exc)
            families[name] = {"error": type(exc).__name__}
        measured[name] = time.perf_counter() - stage_started

    if artifacts:
        stage_started = time.perf_counter()
        try:
            families["artifacts"] = spatial_artifacts(
                gray, rgb, ela=error, residual=residual, phase=spectrum[1] if spectrum is not None else None
            )
        except Exception as exc:
            logger.debug("pixel artifacts failed for %s: %s", image_path, exc)
            families["artifacts"] = {"error": type(exc).__name__}
        measured["full_artifacts"] = time.perf_counter() - stage_started

    measured["total"] = time.perf_counter() - started
    timing_ms = {name: round(seconds * 1000, 1) for name, seconds in measured.items()} if timings else {}
    return PixelEvidence(path=image_path, decode=info, timing_ms=timing_ms, **families)
