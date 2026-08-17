# Doubao clean-reverse-alpha distillation (re-investigated 2026-05-29)

> Research archive. Reverse-alpha pixel recovery is no longer part of the
> current visible-removal pipeline. The current implementation uses
> localize-then-fill; see `docs/module-internals.md`.

> Relocated verbatim from `CLAUDE.md` on 2026-06-11 to keep the always-loaded
> context small. Long single-line entries were reformatted into paragraphs;
> no content was changed or summarized.

**RESOLVED 2026-05-29: black+gray Doubao captures were obtained and a reverse-alpha was built.**
That historical method, `doubao_engine.remove_watermark_reverse_alpha`, has since
been removed. Its detection silhouette remains at
`src/pagedmark/assets/doubao_alpha.png`. The committed captures in
`data/calibration/doubao/` confirmed the alpha-composite model: on black
`captured = a*logo`, logo pure white.

**UPDATE 2026-05-31 (issue #13 follow-up): the first build was NOT "exact"** — it left a readable "豆包AI生成" outline on the real sample (the detector was fooled, conf 0.0). The alpha is now rebuilt by `scripts/visible_alpha_solve.py` (the careful gray-self solve shared with Jimeng), removal always-aligns + thin-inpaints, and the locate box was widened; see the `doubao_engine.py` section in `docs/module-internals.md`. The notes below (the failed content-image distillation) are retained as the record of why controlled captures were necessary.

**Conclusion (historical): pure reverse-alpha distilled from content images does NOT work, and the blocker is the WRONG kind of data, not too little of it.**

Curate same-resolution originals with `DoubaoEngine.detect` and an NCC filter against
a clean glyph template, keeping only aligned marks. Even with aligned inputs,
LaMa-clean `O` plus weighted least squares and per-pixel regression for `α` and logo
color still leaves a persistent ghost outline.

Diagnosed why, empirically (cached stacks, `/tmp/doubao_distill`): (1) the mark is a clean white overlay with **no dark halo** -- over glyph pixels ~54% are brighter than the clean bg, only ~4% darker -- so the white-logo model `I=(1-α)O+α·255` is correct; (2) but content backgrounds are almost never dark *under* the mark (median darkest available bg over glyph pixels = **58/255**; only ~13% of mark pixels are ever observed on a bg < 40), so on bright backgrounds the equation is ill-conditioned and `α` is unidentifiable; (3) LaMa's `O` is a plausible **hallucination**, not the true pre-mark background, which compounds the error, and per-pixel regression on ~15 obs overfits into color noise.

**Why Gemini's engine is clean: its alpha map is the watermark stamped on a PURE-BLACK background**, where `watermarked = α·255 + (1-α)·0 = α·255`, so `alpha = capture/255` exactly -- no estimation. (`gemini_bg_*.png` is literally the sparkle in gray on black.) So the real Doubao unlock is the same controlled capture, **not more content images**. The retained black and gray outputs live in `data/calibration/doubao/`; local solid-color seeds are regenerable and are not committed.

**Until black captures arrive, the shipped direction is precise canonical glyph mask + inpaint (cv2 default, lama optional), NOT reverse-alpha.**

The consensus glyph silhouette across the aligned marks distills cleanly (proto: a tight "豆包AI生成" strip, width ≈ 0.156 × image-width) and is good both as an exact inpaint mask and as an NCC localiser -- the latter also fixes the #23 detector false-positives (match the real glyph shape, not any bright low-saturation corner). Do **not** retry content-image reverse-alpha: it is data-limited by physics (no dark-background observations), not by effort.
