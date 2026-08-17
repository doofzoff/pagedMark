# Jimeng (即梦AI) visible watermark capture

> **Status (completed 2026-05-30):** solid black and gray Jimeng captures were
> obtained and the detection asset was solved. The asset now supports detection
> and mask geometry only. Current removal localizes the wordmark and uses the
> shared fill backend. The text below records the capture and calibration work.

Goal: capture the Jimeng / Dreamina "★ 即梦AI" visible wordmark over known flat
backgrounds so we can rebuild and validate the silhouette used by
`src/pagedmark/jimeng_engine.py`.

## What we learned (verified from the captures, 2026-05-30)

- Mark: a four-point sparkle icon followed by the "即梦AI" characters, near-white
  semi-transparent overlay, bottom-right corner.
- Blend model: **alpha compositing with a pure-white logo** `watermarked =
  a*255 + (1-a)*original`, confirmed in sRGB (a linear-light solve made the
  black/gray cross-residual much worse, so the compositing is plain sRGB). An
  L-pair-solve (independent of the L assumption) lands at ~254.6, confirming white.
- **Alpha is solved from the GRAY capture**, not black: `a = (I - B)/(255 - B)`
  with B a per-capture CUBIC background fit over the non-glyph pixels, averaged
  over channels, at FULL halo extent (down to a~0.02) and UNBLURRED. Gray (bg ~132)
  is the best proxy for real content (the mark sits on bright photo areas, not on
  black). This careful build drops the gray self-residual to ~1.3; an earlier
  max-channel / quadratic-bg / blurred / halo-truncated build (and a black-dominated
  least-squares solve) left a visible outline -- the mask quality, not the method,
  was the limit.
- Geometry (fraction of image WIDTH, at the captured 2048): asset width ~0.211,
  height ~0.068, right margin ~0.023, bottom margin ~0.023. The mark scales with
  width; a real 1440-wide download matched width_frac ~0.21.
- **Per-image render variation:** the alpha maps solved independently from the
  black and the gray capture correlate 0.998 but not 1.0 (mean |Δa| ~0.02). Jimeng
  re-rasterizes the mark per generation and jitters its position a few pixels.
  This was one reason the old reverse-alpha path needed alignment and residual
  inpainting. The current path instead detects the silhouette, localizes the
  glyph footprint, and fills that footprint.

## How to capture (image-edit path, most reliable)

For each solid-color seed:

1. Open Jimeng image generation, use the image-edit / reference mode, upload the seed.
2. Prompt (Chinese preferred):
   `请完全按照原图重新生成这张图片，保持完全一致，不要添加或修改任何内容`
3. Download the ORIGINAL output file (not a screenshot). Do not crop / edit / re-save.

The black capture is the key one (white logo on black -> `captured ~= a*255`); the
gray capture refines the alpha at mid-tones; the white capture confirms the logo is
pure white (the mark is nearly invisible on white, as expected).

## Hygiene

- Original download, never a screenshot. PNG preferred; if Jimeng only gives JPEG, note it.
- No crop / edit / re-save. Default settings, watermark left ON.

## Committed inputs

```
jimeng_cap_A.png   # black seed run through Jimeng
jimeng_cap_C.png   # gray seed
```

The A and C captures feed
`scripts/visible_alpha_solve.py jimeng`. Rebuild the detection asset with:

```
uv run python scripts/visible_alpha_solve.py jimeng   # or: all
```
