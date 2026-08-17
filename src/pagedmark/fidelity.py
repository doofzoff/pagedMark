"""What a removal run cost the picture, measured rather than asserted.

Regeneration is not payload deletion: it rewrites pixels, and the honest question is
how much. These are the measurements this project's own operating points were chosen
with, so a user can run them on their own file instead of trusting a number from a
README.

Three quantities, because one is not enough:

- **PSNR over the frame.** Cheap, familiar, and dominated by whatever occupies the most
  area. It moves too little to notice a ruined face.
- **PSNR per detected face.** Identity is what people look at first, and a face is a
  small fraction of the pixels, so the frame average hides it.
- **Invented texture.** The one that caught a real regression. A diffusion pass with
  nothing to condition on -- flat, dark cloth carries no edges -- fills the gap from its
  prior, and the result reads as coloured camouflage. Per-pixel measures are blind to
  it: the source's own sensor grain has MORE per-pixel variance than the invented
  blotches, so a naive chroma statistic reports the artifact as an improvement.

  What distinguishes the two is the SHAPE of the spectrum, not its size. Grain is flat
  across frequencies; invented blotches are low-pass. So the measure is a mid-band
  energy normalised by fine-detail energy, compared against the same ratio in the
  source: scaling the grain moves both terms and cancels, while filling a flat region
  with 8-32 px structure moves only the numerator. A plain band ratio cannot do this --
  any linear filter preserves a factor of two in the input, so it reports "twice the
  grain" and "twice the blotches" identically.
"""

# cv2/numpy boundary: the same relaxation the other array modules carry, for the same
# reason -- neither library ships usable element types.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

# The band the artifact occupies, and the two windows that isolate it: average over
# _BAND_LOW_PX to drop the grain, subtract the average over _BAND_HIGH_PX to drop the
# tone. Measured: an affected frame reads 1.10x the source at 32 px and its mean luma
# 1.09x, so what remains between those windows is invented detail, not exposure.
_BAND_LOW_PX = 8
_BAND_HIGH_PX = 32
# The normaliser: everything above this window is where grain lives, and dividing by it
# is what makes the ratio blind to how noisy either image happens to be.
_FINE_PX = 3
# "Flat and dark" in the SOURCE decides where to look, because the question is what the
# run added where the input had nothing. Percentiles rather than absolute levels, so the
# region tracks the exposure of the actual photograph.
_DARK_PERCENTILE = 20.0
_FLAT_PERCENTILE = 40.0
_FLAT_WINDOW_PX = 7
# Below this many qualifying pixels the ratio is noise, not a measurement.
_MIN_REGION_PIXELS = 2048


class FaceFidelity(NamedTuple):
    """One detected face and how far the output moved it."""

    box: tuple[int, int, int, int]
    psnr: float


class FidelityReport(NamedTuple):
    """The full comparison. ``invented_texture`` is None when the source has no
    flat, dark region large enough to measure one in."""

    width: int
    height: int
    psnr: float
    faces: tuple[FaceFidelity, ...]
    invented_texture: float | None

    @property
    def face_psnr(self) -> float | None:
        """Mean PSNR over detected faces, or None when none were found."""
        if not self.faces:
            return None
        return sum(face.psnr for face in self.faces) / len(self.faces)


def psnr(reference: NDArray, candidate: NDArray) -> float:
    """Peak signal-to-noise ratio in dB. ``inf`` for identical input."""
    import numpy as np

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes")
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def _luma(image: NDArray) -> NDArray:
    """Rec.601 luma from a BGR array, as float32."""
    import numpy as np

    array = image.astype(np.float32)
    if array.ndim == 2:
        return array
    blue, green, red = array[..., 0], array[..., 1], array[..., 2]
    return 0.114 * blue + 0.587 * green + 0.299 * red


def flat_dark_mask(reference: NDArray) -> NDArray:
    """Where the source is both dark and featureless -- the region a prior fills in."""
    import cv2
    import numpy as np

    luma = _luma(reference)
    window = (_FLAT_WINDOW_PX, _FLAT_WINDOW_PX)
    local_variance = cv2.blur(luma**2, window) - cv2.blur(luma, window) ** 2
    dark = luma < np.percentile(luma, _DARK_PERCENTILE)
    flat = local_variance < np.percentile(local_variance, _FLAT_PERCENTILE)
    return dark & flat


def _band_shape(image: NDArray, mask: NDArray) -> float:
    """Mid-band energy over fine-detail energy inside ``mask``, across luma and chroma.

    A dimensionless number, which is the point: it describes how the energy is
    distributed rather than how much there is, so it survives a change in noise level.
    """
    import cv2
    import numpy as np

    array = image.astype(np.float32)
    luma = _luma(array)
    channels = (luma, array[..., 0] - luma, array[..., 2] - luma)
    low, high, fine = (_BAND_LOW_PX,) * 2, (_BAND_HIGH_PX,) * 2, (_FINE_PX,) * 2
    band_variance = fine_variance = 0.0
    for channel in channels:
        band = cv2.blur(channel, low) - cv2.blur(channel, high)
        band_variance += float(band[mask].std()) ** 2
        fine_variance += float((channel - cv2.blur(channel, fine))[mask].std()) ** 2
    return float(np.sqrt(band_variance) / max(np.sqrt(fine_variance), 1e-6))


def invented_texture(reference: NDArray, candidate: NDArray) -> float | None:
    """How much mid-band structure the candidate holds where the source held none.

    1.0 means the output's mid-band is distributed like the input's. Above it, structure
    was added where there was none: the regression this metric was built for measured
    1.24x, and the fix brought it to 1.07x. Below it, the region came back smoother than
    the source. None when the source offers no flat, dark region big enough to be worth
    a ratio.
    """
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes")
    mask = flat_dark_mask(reference)
    if int(mask.sum()) < _MIN_REGION_PIXELS:
        return None
    baseline = _band_shape(reference, mask)
    if baseline <= 0.0:
        return None
    return _band_shape(candidate, mask) / baseline


def _detected_faces(reference_path: Path) -> list[tuple[int, int, int, int]]:
    """Face boxes from the source, or none when the detector is unavailable.

    A missing detector degrades the report by one row rather than failing it: the frame
    and texture numbers are still the answer to most questions.
    """
    try:
        from PIL import Image

        from pagedmark._internal.qwen_zimage_pipeline import detect_faces

        with Image.open(reference_path) as opened:
            return detect_faces(opened.convert("RGB"))
    except Exception:
        return []


def compare(reference_path: Path, candidate_path: Path, *, faces: bool = True) -> FidelityReport:
    """Measure one before/after pair.

    A candidate at a different size is resized to the reference before comparison --
    diffusers rounds to its latent grid, so an 8-pixel difference is routine and
    refusing to compare over it would be pedantry.
    """
    import cv2

    from pagedmark import image_io

    source = image_io.imread(reference_path, cv2.IMREAD_COLOR)
    output = image_io.imread(candidate_path, cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Could not decode {reference_path}")
    if output is None:
        raise ValueError(f"Could not decode {candidate_path}")
    if output.shape != source.shape:
        output = cv2.resize(output, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    boxes = _detected_faces(reference_path) if faces else []
    face_rows: list[FaceFidelity] = []
    for x1, y1, x2, y2 in boxes:
        crop_reference = source[y1:y2, x1:x2]
        if crop_reference.size == 0:
            continue
        face_rows.append(FaceFidelity((x1, y1, x2, y2), psnr(crop_reference, output[y1:y2, x1:x2])))

    return FidelityReport(
        width=source.shape[1],
        height=source.shape[0],
        psnr=psnr(source, output),
        faces=tuple(face_rows),
        invented_texture=invented_texture(source, output),
    )
