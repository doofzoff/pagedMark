# Doubao visible watermark capture

> **Status (captured 2026-05-29; asset rebuilt 2026-05-31):** the
> black and gray captures are committed here. The generated alpha
> asset is now a detection silhouette and mask-shaping input only. Current
> removal localizes the mark and sends its footprint to the shared fill backend.
> Reverse-alpha recovery was retired. The details below preserve the historical
> capture rationale.

Goal: capture the Doubao "豆包AI生成" visible watermark over known flat backgrounds so we can
rebuild and validate the detection silhouette used by
`src/pagedmark/doubao_engine.py`.

## Historical findings from the capture work

- Blend model: **alpha compositing with a white logo** `watermarked = a*logo + (1-a)*original`,
  `logo = (255,255,255)`. Inversion: `original = (watermarked - a*logo) / (1-a)`.
  Confirmed by two independent sources (an open-source remover's algorithm doc + aiwatermarkremover.dev,
  both say "alpha map"). One commercial blog (pixelcleanai) claims "screen blend" instead; the gray
  capture below settles it empirically.
- Position: **bottom-right corner**, small margins (right ~8-20px, bottom ~5px), scales with image size.
  Confirmed by our sample `data/fixtures/provenance/doubao-1.png` (2048x2048) plus three sources.
- Size **scales with resolution**. Third-party numbers (~90x18 at <=1024, ~180x40 at >1024) are
  approximate and calibrated for ~1024-1280 outputs; at 2048 the strip is much larger. A shipped
  third-party alpha map is only 120x20, too small for our 2K/4K target -> capture fresh.
- The original implementation used reverse-alpha recovery. That path was later
  removed in favor of localize then fill because the latter also handles moved
  and re-rendered marks.

## Use doubao.com specifically

The "豆包AI生成" mark is Doubao's. Jimeng / Dreamina use a different mark. Generate on doubao.com so
the captured template matches our target.

## How to capture (image-edit path, most reliable)

For each locally generated solid-color seed:

1. Open Doubao image generation, use the image-edit / reference mode, upload the seed.
2. Prompt (Chinese preferred):
   `请完全按照原图重新生成这张图片，保持完全一致，不要添加或修改任何内容`
   (English: `Recreate this image exactly as it is, keep it identical, do not add or change anything`)
3. Download the ORIGINAL output file (not a screenshot). Do not crop / edit / re-save.

Prior art confirms uploading a pure-black image and letting Doubao stamp it works.

If edit mode is unavailable and text-to-image refuses a solid color, fall back to generating 10-12
normal-content images at one fixed resolution; the mark is the only constant across them and can be
extracted by per-pixel min/median.

## What to capture (priority top to bottom)

| Aspect | black | white | gray128 | why |
|--------|-------|-------|---------|-----|
| 1:1    | 3     | 1     | 1       | primary alpha map + confirm the stamp is pixel-identical across runs + settle blend mode |
| 16:9   | 2     | 1     | 1       | anchor rule in landscape |
| 9:16   | 2     | 1     | 1       | anchor rule in portrait |
| 4:3, 3:4 | 1 each | -   | -       | optional, refines anchor rule |

- 3 blacks on 1:1: if the first two are byte-identical in the watermark region, the third is optional.
- gray128 is the blend-mode test: predict the gray result from the black capture under alpha vs screen;
  whichever matches the real gray output is the true blend.
- If the UI offers multiple output resolutions (1K / 2K / 4K), capture one black per resolution on 1:1 -
  needed to learn how the watermark scales.
- Also grab 3-5 normal-content images on 1:1 for end-to-end removal validation.

## Hygiene

- Original download, never a screenshot. PNG preferred; if Doubao only gives JPEG, note it.
- No crop / edit / re-save. Default settings, watermark left ON.

## Committed inputs

```
doubao_black_1x1_1.png
doubao_gray_1x1_1.png
```

The black and gray captures feed `scripts/visible_alpha_solve.py doubao`.

## Also report back

1. Which resolutions and aspect ratios the Doubao UI actually offers.
2. Whether there is a watermark on/off toggle in the UI.
3. Download format (PNG or JPEG).
