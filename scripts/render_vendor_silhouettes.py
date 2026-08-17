"""Render synthetic detection silhouettes for vendor text marks.

Committed assets must be font-rendered and contain no source-image pixels. Local
evaluation inputs may be used only to learn glyphs, weight, layout, and detector
thresholds. Candidate assets stay outside the installed package until calibrated.

Regenerate with:
    uv run python scripts/render_vendor_silhouettes.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from pagedmark.watermark_registry import mark_keys  # noqa: E402

_PACKAGE_ASSETS = _ROOT / "src" / "pagedmark" / "assets"
_CANDIDATE_ASSETS = _ROOT / "scripts" / "assets" / "visible-mark-candidates"
# STHeiti Medium approximates the semibold CJK sans these marks are set in; the exact
# family is unpublished for every vendor (GB 45438-2025 only requires a legible face).
_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"

MARKS = {
    "qwen_alpha.png": "千问AI生成",
    "xinghui_alpha.png": "星绘AI生成",
    # Yuanbao's stamp is a TWO-LINE block (元宝 over AI生成), left-aligned, tightly
    # stacked and ITALIC-SLANTED. A rare one-line variant exists, but the stacked block
    # is dominant.
    "yuanbao_alpha.png": "元宝\nAI生成",
    # Kling (可灵) stamps a thin light-gray one-line "可灵AI 3.0" bottom-right (an
    # "Omni" suffix variant and a latin "KlingAI 3.0" variant also exist; the CJK
    # run without the suffix is the common core). The leading spiral logo is NOT
    # rendered (logos vary; the text run discriminates).
    "kling_alpha.png": "可灵AI 3.0",
    # The "cat-logo" candidate stamps an outline cat-head plus bold "AI生成",
    # bottom-right. It remains unregistered pending sufficient calibration coverage.
    "catlogo_alpha.png": "CATLOGO",  # sentinel: drawn by draw_catlogo(), not font-rendered
    # RunningHub top-left text mark.
    "runninghub_alpha.png": "RunningHub AI生成",
    # LibLibAI bottom-center wordmark.
    "liblib_alpha.png": "LibLibAI",
    # Zhipu Qingyan candidate text mark.
    "qingyan_alpha.png": "清言·AI生成",
    # MiniMax / Hailuo candidate wordmark.
    "hailuo_alpha.png": "Hailuo AI",
    # Baidu bottom-right text run.
    "baidu_alpha.png": "百度",
}
_REGISTERED = {f"{key}_alpha.png" for key in mark_keys()} & MARKS.keys()

# Per-mark post-processing for the multi-line / slanted stamps (see render()).
MARK_OPTS: dict[str, dict[str, Any]] = {
    # Hiragino Sans GB W6, tight leading, dilation, and negative shear match the
    # standard Yuanbao stamp without clipping the lower line.
    "yuanbao_alpha.png": {
        "gap_frac": 0.05,
        "dilate": 2,
        "shear": -0.60,
        "font": "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "font_index": 2,
    },
    # Qingyan uses a heavier weight than STHeiti Medium.
    "qingyan_alpha.png": {"font": "/System/Library/Fonts/Hiragino Sans GB.ttc", "font_index": 2},
    # LibLibAI uses an Arial-class grotesque.
    "liblib_alpha.png": {"font": "/System/Library/Fonts/Supplemental/Arial.ttf"},
}


def render(text: str, width: int = 335, opts: dict[str, Any] | None = None) -> np.ndarray:
    """Binary glyph silhouette (255 = glyph), sized to the doubao asset's convention.

    Matching doubao's 335px asset width keeps the `alpha_*_frac` numbers transferable,
    since these marks are the same house style and scale. A "\n" in ``text`` renders a
    multi-line block: lines drawn left-aligned at one shared font size with a tight
    gap, then optional stroke dilation and an italic shear (see MARK_OPTS).
    """
    opts = opts or {}
    gap_frac = float(opts.get("gap_frac", 0.15))
    dilate = int(opts.get("dilate", 0))
    shear_k = float(opts.get("shear", 0.0))
    font_path = str(opts.get("font", _FONT))
    font_index = int(opts.get("font_index", 0))
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    lines = text.split("\n")
    size = 8
    while size < 200:  # grow until the LONGEST line fills the target width
        f = ImageFont.truetype(font_path, size, index=font_index)
        if max(d0.textbbox((0, 0), ln, font=f)[2] for ln in lines) >= width * 0.98:
            break
        size += 1
    font = ImageFont.truetype(font_path, size, index=font_index)
    boxes = [d0.textbbox((0, 0), ln, font=font) for ln in lines]
    line_h = max(bb[3] - bb[1] for bb in boxes)
    gap = max(1, int(line_h * gap_frac))
    w = max(bb[2] - bb[0] for bb in boxes)
    h = line_h * len(lines) + gap * (len(lines) - 1)
    pad = max(2, int(line_h * 0.12))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    draw = ImageDraw.Draw(im)
    y = pad
    for ln, bb in zip(lines, boxes, strict=True):
        draw.text((pad - bb[0], y - bb[1]), ln, font=font, fill=255)
        y += line_h + gap
    sil = np.array(im)
    if dilate or shear_k:
        import cv2

        if dilate:
            sil = cv2.dilate(sil, np.ones((dilate, dilate), np.uint8))
        if shear_k:
            hh, ww = sil.shape
            extra = int(abs(shear_k) * hh)
            offset = extra if shear_k < 0 else 0
            sil = cv2.warpAffine(
                sil,
                np.float32([[1, shear_k, offset], [0, 1, 0]]),
                (ww + extra, hh),
            )
            ys, xs = np.where(sil > 0)
            if xs.size:
                sil = sil[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return sil


def draw_catlogo(width: int = 335) -> np.ndarray:
    """The cat-logo mark: an outline cat-head (integrated pointy ears, two dot eyes)
    + a bold "AI生成" run, drawn synthetically from the calibrated layout. The outline
    form is the parked candidate described in MARKS."""
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    text = "AI生成"
    size = 8
    while size < 200:
        f = ImageFont.truetype(_FONT, size)
        if d0.textbbox((0, 0), text, font=f)[2] >= width * 0.60:
            break
        size += 1
    font = ImageFont.truetype(_FONT, size)
    bb = d0.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    cs = int(th * 1.08)
    stroke = max(2, int(th * 0.09))
    gap = int(th * 0.35)

    def head(s: int) -> Image.Image:
        im = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(im)
        f = float(s)
        pts = [
            (0.12 * f, 0.95 * f),
            (0.10 * f, 0.45 * f),
            (0.12 * f, 0.30 * f),
            (0.20 * f, 0.05 * f),  # left ear tip
            (0.40 * f, 0.24 * f),  # left ear valley
            (0.60 * f, 0.24 * f),  # right ear valley
            (0.80 * f, 0.05 * f),  # right ear tip
            (0.88 * f, 0.30 * f),
            (0.90 * f, 0.45 * f),
            (0.88 * f, 0.95 * f),
        ]
        d.line([*pts, pts[0]], fill=255, width=stroke, joint="curve")
        r = max(1.5, stroke * 0.7)
        d.ellipse([0.35 * f - r, 0.60 * f - r, 0.35 * f + r, 0.60 * f + r], fill=255)
        d.ellipse([0.65 * f - r, 0.60 * f - r, 0.65 * f + r, 0.60 * f + r], fill=255)
        return im

    w = cs + gap + tw
    h = max(th, cs)
    pad = max(2, int(h * 0.12))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    im.paste(head(cs), (pad, pad + (h - cs) // 2))
    ImageDraw.Draw(im).text((pad + cs + gap - bb[0], pad + (h - th) // 2 - bb[1]), text, font=font, fill=255)
    return np.array(im)


def main() -> None:
    try:
        for name, text in MARKS.items():
            sil = draw_catlogo() if text == "CATLOGO" else render(text, opts=MARK_OPTS.get(name))
            output_dir = _PACKAGE_ASSETS if name in _REGISTERED else _CANDIDATE_ASSETS
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / name
            Image.fromarray(sil).save(output)
            print(f"wrote {output}  ({sil.shape[1]}x{sil.shape[0]})  text={text!r}")
    except OSError as e:
        print(f"Font not found ({e}); install a CJK font or edit _FONT.", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
