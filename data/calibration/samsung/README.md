# Samsung Galaxy AI visible watermark capture

> **Status (built 2026-06-05):** flat black and gray Samsung Galaxy AI captures
> were obtained and the detection asset was solved. The asset now supports
> detection and mask geometry only. Current removal localizes the wordmark and
> uses the shared fill backend. The text below records the capture plan and its
> locale and resolution limits.

Goal: capture the Samsung Galaxy AI "✦ Contenuti generati dall'AI" visible wordmark
over known flat backgrounds so we can rebuild and validate the silhouette used
by `src/pagedmark/samsung_engine.py`.

## What we learned (verified from the captures, 2026-06-05)

- Mark: a sparkle icon followed by the locale string "Contenuti generati dall'AI"
  (Italian), a light low-opacity (peak alpha ~0.38) semi-transparent **white**
  overlay, anchored **bottom-LEFT** (Doubao/Jimeng are bottom-right). The string is
  locale-specific, so the alpha template is per-locale; this build is the Italian
  variant. Other locales need their own captured template.
- Blend model: alpha compositing with a pure-white logo, `watermarked =
  a*255 + (1-a)*original`, solved from the GRAY capture (same careful recipe as
  Doubao/Jimeng: cubic-background fit, mean over channels, full halo extent,
  unblurred). The white capture confirms the logo is white; on white the mark is
  white-on-white and not detectable (no contrast), which is fine -- there is nothing
  to recover there.
- Geometry (fraction of image WIDTH): asset width ~0.32, height ~0.038, left margin
  ~0.011, bottom margin ~0.006. The mark scales with width: a 1086-wide flat capture
  and a 2958-wide real photo both measure width_frac ~0.31.
- **Resolution caveat (open quality follow-up):** the flat black/gray/white captures
  arrived at the phone's flat-edit size (1086 wide and a landscape 1920 set), while
  the real photos are ~3000 wide, so the captured glyph (~334 px) is ~2.7x smaller
  than on a real photo (~900 px). The alpha is solved at the capture size and
  width-scaled + NCC-aligned per image, which removes the mark cleanly (verified on a
  real 2958-wide photo: re-detect 0.79 -> 0.00, no readable text or outline), but a
  flat capture taken at the real photo resolution (~3000 wide) would let the alpha be
  pixel-sharp instead of upscaled. Not a blocker; a quality upgrade if a full-res
  flat capture is provided.

## Capture protocol (to re-capture or add a locale)

On a Samsung Galaxy AI device (set the UI language to the target locale):

1. Run the AI edit (Generative Edit / Sketch to Image) on a solid black image, so
   the overlay lands on a flat black background. Download the ORIGINAL output file
   (not a screenshot, no crop or re-save).
2. Repeat over solid white and solid gray (those pin the exact glyph color).
3. Ideally run all three flat edits at the same resolution as real photos (~3000
   wide) so the alpha map is pixel-sharp rather than upscaled.
4. Plus 3-5 real outputs with the visible mark over normal content for validation.

## Files

- `samsung_black_1.png` and `samsung_gray_1.png` are the retained solver inputs.
- White, secondary-resolution, and real-content captures were validation
  material and are not required to rebuild the asset.

Rebuild the detection asset with:

```
uv run python scripts/visible_alpha_solve.py samsung
```
