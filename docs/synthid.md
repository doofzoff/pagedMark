# SynthID: technical reference

> Technical research reference. Current package behavior is defined by the
> [supported signals](supported-signals.md), [known limitations](known-limitations.md),
> and [module internals](module-internals.md). Dated measurements below are
> historical evidence and should not be read as current CLI defaults.

This document covers how Google SynthID for images works mechanically, what it
survives, what removes it, the external video-verification workflow, and the
current deployment landscape. It is written for engineers working on watermark
detection and removal -- specifically to inform decisions about strength
settings, test methodology, and what oracle results mean.

Primary sources are cited inline. Marketing-only claims are flagged separately
from independently-verified results.

---

## 1. Mechanism

### 1.1 Post-hoc, model-independent design

SynthID-Image is **not** baked into a diffusion model's weights. It is a
post-hoc, model-independent system: a separate encoder `f` is applied to an
already-generated image, and a separate decoder `g` reads it back.

> "We deliberately designed SynthID-Image as a post-hoc, model-independent
> approach, a choice largely based on deployment considerations."
> -- Gowal et al., arXiv:2510.09263

The formal definition from the paper:

> "A post-hoc watermarking scheme is a pair f, g consisting of an encoder
> function f: X -> X, which adds an identification mark, and a decoder
> function g: X -> {+-1}, which tries to detect if the mark is present."

This is the key architectural fact: **the generative model (Imagen, Gemini's
image model) is not modified**. The watermark is stamped onto the pixel output
after generation, by a separate neural network. This means:

- The watermark is in **pixel space**, not in the model's latent activations.
- Replacing the generative model does not remove the watermarking capability.
- The encoder/decoder pair can be updated independently of the generative model.

The paper does not disclose the internal architecture of the encoder/decoder
networks (layer types, capacity). The external variant SynthID-O is available
to partners; the production internal variant is not published.

### 1.2 How it differs from classical DWT-DCT watermarks

The open watermarks used by Stable Diffusion / SDXL / FLUX (via the
`imwatermark` library) use classical **DWT-DCT** frequency-domain embedding: a
fixed bit pattern is added to specific frequency coefficients of the image's
wavelet transform. This is fast, key-free, and locally detectable with a public
decoder.

SynthID-Image uses **jointly-trained deep learning models**:

> "SynthID uses two deep learning models -- for watermarking and identifying --
> that have been trained together on a diverse set of images. The combined model
> is optimised on a range of objectives, including correctly identifying
> watermarked content and improving imperceptibility by visually aligning the
> watermark to the original content."
> -- Google DeepMind blog, 2023

The practical difference for robustness: the deep learning encoder learns to
spread the signal across the image in a way that is optimized to survive a
specific perturbation distribution seen during training. Classical DWT-DCT
embeds in fixed, predictable frequency bins, making it brittle to any
operation that hits those bins (e.g., JPEG re-quantization wipes it cleanly at
quality <= 90).

### 1.3 Payload capacity

SynthID-O (the external/partnership variant) encodes:

- **136 bits** within a **512x512 pixel image**

For comparison (from the same paper):

| Method      | Bits | Resolution |
|-------------|------|------------|
| SynthID-O   | 136  | 512x512    |
| StegaStamp  | 100  | 400x400    |
| TrustMark   | 100  | 256x256    |
| WAM         | 32   | 256x256    |

The payload carries an identification mark (not a user-readable secret). The
paper separates watermark **detection** (is this watermarked?) from payload
**recovery** (what does the payload say?): the detection path is what oracles
like the Gemini app's "Verify with SynthID" exercise.

### 1.4 Where in the pipeline it lives

```
[Diffusion model]
       |
  raw pixel output
       |
  [SynthID encoder f]   <-- separate neural net, stamps the watermark
       |
  watermarked image
       |
  [served / downloaded]
       |
  [SynthID decoder g]   <-- separate neural net, run by Google's verifier only
       |
  present / not present
```

The VAE decoder of the diffusion model is **not** involved in watermarking.
Some in-generation watermark approaches (like the research method "Tree Ring")
inject the signal into the initial noise latent so it propagates through the
diffusion process and appears in the final image; SynthID-Image does not do
this -- it is applied after the VAE has already decoded latents to pixels.

---

## 2. Robustness

### 2.1 What the paper claims it survives (primary-source verified)

The SynthID-Image paper (arXiv:2510.09263) evaluates SynthID-O against **30
image transformations** grouped into 6 categories:

| Category    | Examples                                      |
|-------------|-----------------------------------------------|
| Color       | brightness, contrast, saturation, hue shifts  |
| Combination | combinations of multiple transforms           |
| Noise       | Gaussian noise, impulse noise, median filter  |
| Overlay     | text overlays, logos, stickers                |
| Quality     | JPEG compression, WebP, format conversion     |
| Spatial     | crop, resize, rotate, flip, padding           |

**TPR at 0.1% FPR -- SynthID-O vs. baselines (resized to 512x512):**

| Category         | SynthID-O | Best baseline (WAM) | Worst baseline (StegaStamp spatial) |
|------------------|-----------|---------------------|--------------------------------------|
| Identity (none)  | 100.00%   | 100.00%             | 100.00%                              |
| Aggregated       | 99.98%    | 90.62%              | ~70%                                 |
| Color            | 100.00%   | 81.29%              | ~75%                                 |
| Combination      | 99.96%    | 96.08%              | ~22%                                 |
| Noise            | 99.98%    | 100.00%             | ~92%                                 |
| Overlay          | 100.00%   | 100.00%             | 100.00%                              |
| Quality          | 99.99%    | --                  | ~89%                                 |
| Spatial (worst)  | 99.97%    | 76.04%              | 15.25%                               |

The "Spatial worst" row is the hardest case (aggressive crop + resize).
SynthID-O retains 99.97% TPR; StegaStamp collapses to 15.25%. This is where
the deep-learning approach gains the most over classical methods.

Google's marketing page states the watermark is:

> "designed to stand up to modifications like cropping, adding filters, changing
> frame rates, or lossy compression."
> -- deepmind.google/models/synthid/

The marketing claim is broadly consistent with the paper's numbers for these
specific categories.

**JPEG and format conversion specifically** fall under the "Quality" category,
where SynthID-O achieves 99.99% TPR. This is the empirical basis for the fact
that **GitHub-recompressed JPEGs from issue attachments are valid SynthID test
subjects**: the re-encoding does not remove the pixel watermark.

### 2.2 Stated limits (vendor claim, not independently verified)

> "SynthID isn't foolproof against extreme image manipulations."
> -- Google DeepMind blog, 2023

This is the only public failure-mode statement Google has made. No specific
perturbation type, threshold, or quantitative boundary is named. The
Limitations section of the paper (Section 10) was not recoverable from the
public HTML version of arXiv:2510.09263v1 due to a rendering failure in the
conversion (the body text of Section 10 is absent from the HTML).

**What is known empirically from our own oracle-verified testing.**

A controlled study (June 2026, clean v0.8.6 with text/face protection OFF,
native resolution on this repo's default SDXL pipeline) measured the minimum
img2img strength that removes the SynthID pixel watermark, verified per image on
the vendor's own oracle (openai.com/verify for OpenAI, the Gemini app "Verify
with SynthID" for Google). The reusable originals are stored once in
`data/synthid/originals/`, with their input verification in `manifest.csv`.
Generated cleaned outputs are not committed; the table below is the durable
record of the historical oracle verdicts. One third-party image from issue #14
was oracle-verified but is not committed.

**Oracle validation order: start with OpenAI.** When validating removal across
vendors, run the OpenAI arm first. `openai.com/verify` is more accessible than the
Gemini app -- fewer per-check restrictions, so it gives the fastest signal and is
the strongest candidate for automation (Playwright / Chrome MCP driving
`openai.com/verify`); the Gemini "Verify with SynthID" flow is more manual. This is
an ordering/throughput choice, not a substitution: each oracle only reads its own
vendor's SynthID (`openai.com/verify` is OpenAI-scoped), so Google content still
needs the Gemini app.

| Vendor | Images | Resolution(s) | Pipeline | Removed at |
|--------|--------|---------------|----------|------------|
| OpenAI (gpt-image) | n=4 (3 archived + 1 external-only) | 1024x1536 .. 1600x1600 | native | **0.05** |
| Google (Gemini)    | n=4 | 2816x1536 -> capped 1536 | `--max-resolution 1536` | **0.15** (0.05 and 0.10 do NOT clear) |

**Two findings, both oracle-verified:**

1. **Vendor is the dominant factor, not resolution.** Google's SynthID is
   roughly 3x more robust than OpenAI's: at a comparable (small) working
   resolution, OpenAI clears at 0.05 while Google needs 0.15. This matches
   Google having hardened SynthID more aggressively over time.

2. **OpenAI SynthID removal is resolution-independent in the tested range.**
   All four OpenAI images (including a 1600x1600) cleared at 0.05.

**CORRECTION (supersedes the earlier "resolution dependence" claim).** A prior
version of this doc and CLAUDE.md stated that strength 0.30 failed to remove
SynthID on 1600x1600 gpt-image and that removal was resolution-dependent. That
was a **measurement artifact of a since-removed per-region re-scrub step** (issue
#14): on the dense-text infographics tested, that step could reconstitute SynthID
in text regions. Re-running the *same* 1600x1600 image on the clean current
pipeline removes SynthID at **0.05**. The "large images resist removal" conclusion
was false; the resistance was that region-rescrub shielding, since removed.

**Open / not locally testable:**

- **Native large Gemini (2816x1536, ~4.3 MP).** The Gemini floor of 0.15 was
  measured on the *capped* (`--max-resolution 1536`) path, which is the
  practical local route on Apple-Silicon (native 2816 OOMs / falls back to slow
  CPU on a 32 GB M-series). Native large Gemini was not measured here; the
  vendor and resolution effects would stack, so it plausibly needs >= 0.30 or a
  discrete GPU. Confirm on a CUDA box if needed.
- **Heavy JPEG compression** (quality < ~50-60): not oracle-tested; the DL
  approach is more robust than DWT-DCT but Google acknowledges limits at
  "extreme" manipulation.

### 2.3 Removal attacks and forensic detectability

The paper arXiv:2605.09203 ("Removing the Watermark Is Not Enough",
Goonatilake & Ateniese, 2026) evaluates 6 removal attacks against a ResNet-50
forensic detector. All attacks defeat the watermark verifier but are detected
by the forensic classifier:

| Attack          | Family           | AUROC  | TPR @ 1% FPR | TPR @ 0.1% FPR |
|-----------------|------------------|--------|--------------|----------------|
| UnMarker        | Distortion       | 0.9994 | 99.81%       | 98.28%         |
| WatermarkAttacker| Regeneration    | 0.9997 | 99.95%       | 99.38%         |
| CtrlRegen+      | Regeneration     | 0.9999 | 99.97%       | 99.64%         |
| NFPA            | Inversion/Pert.  | 0.9984 | 99.24%       | 62.10%         |
| Boundary Leak.  | Inversion/Pert.  | 0.9991 | 99.24%       | 88.34%         |
| WiTS            | Erosion          | 0.9999 | 99.80%       | 99.55%         |

The forensic detector is a standard ResNet-50 fine-tuned end-to-end; no exotic
architecture needed. The key finding:

> "These removers do not return images to a clean forensic state. They often
> trade an explicit watermark for an implicit watermark: a detectable artifact
> introduced by the removal process itself."

This means: even when our SDXL img2img pass defeats the SynthID pixel
watermark (oracle reads negative), the output may still be classifiable as
"an image that went through a removal pipeline" by an independent detector --
even if that detector is not trained on SynthID specifically. **Defeating the
verifier does not restore forensic deniability.**

CtrlRegen+ is the most detectable removal method (AUROC 0.9999), which is
notable because it is also the most powerful removal attack. The paper notes
that diffusion regeneration "leaves a strong reconstruction signature from the
diffusion prior."

---

## 3. Detectability and verifier access

### 3.1 No public local detector

The SynthID decoder is proprietary and not released:

> "SynthID-Image has been used to watermark over ten billion images and video
> frames across Google's services and its corresponding verification service is
> available to trusted testers."
> -- Gowal et al., arXiv:2510.09263

There is no public API, no released decoder weights, and no reproducible
algorithm for local detection. The verification service (SynthID Detector) is:

> "a verification portal" in early testing with "journalists and media
> professionals" on a waitlist
> -- deepmind.google/models/synthid/

The external variant SynthID-O is available "through partnerships" only. Our
tool cannot locally detect SynthID presence or absence -- this is by design,
not a gap we can fill.

### 3.2 How our tool recognizes SynthID from provenance

We recognize SynthID indirectly from supported C2PA evidence; this is not a
pixel watermark decode. Google states that all media generated by its tools is
watermarked, so Google AI C2PA establishes SynthID. OpenAI C2PA predates its
SynthID rollout, but current manifests add the signed
`c2pa.watermarked.unbound` action; OpenAI provenance establishes SynthID only
when that action is present. Legacy OpenAI C2PA without it remains valid origin
evidence but does not assert a pixel watermark.

This works while the C2PA manifest is intact and is silent once the manifest is
stripped or the image is re-encoded without C2PA (e.g., a screenshot, a
social-media re-upload, or after `metadata --remove`).

This is why:
- `identify` on a GitHub-recompressed issue attachment returns Unknown (C2PA is
  gone) even though the pixel SynthID is still present and detectable by
  openai.com/verify.
- A quiet `identify` output is not proof that SynthID was removed -- it only
  means the metadata signal is gone.

### 3.3 Oracle scope: each vendor detects only their own

From openai.com/research/verify (verbatim, verified 2026-05-31):

> "OpenAI generation signals will only be detected if the image was generated
> with our tools."
> "Content could also still be AI-generated by another company's model, which
> the tool currently does not detect."

SynthID technology is used by multiple vendors, but each verifier is keyed to
its own payload:

| Oracle                        | Detects          | Does NOT detect         |
|-------------------------------|------------------|-------------------------|
| Gemini app "Verify with SynthID" | Google SynthID | OpenAI SynthID          |
| openai.com/research/verify    | OpenAI SynthID   | Google SynthID          |

A Google-SynthID image reads clean on openai.com/verify. An OpenAI image reads
clean in the Gemini oracle. They are different payloads within the same
framework.

### 3.4 Video verification and attack harness

Gemini's built-in verification flow reports whether and where it detects Google
SynthID in a video. This remains a proprietary oracle: invoke `@synthid`, use
the supported content-verification question, and keep every file in a separate
new chat. A normal Gemini answer that discusses visual clues or metadata is not
an oracle verdict. Nor is an adversarial follow-up that asks the chat model to
ignore and reinterpret a completed verifier result.

The research harness `scripts/video_synthid_sweep.py` tests a VAE regeneration
attack without pretending to detect success locally. It emits:

1. a re-encode control using the same sampled frames, dimensions, frame rate,
   and codec as the candidates;
2. VAE round-trip candidates with one spatial latent-noise field shared across
   time;
3. paired PSNR and motion-compensated temporal-residual metrics;
4. an empty oracle column for the external verdict.

The control is the first oracle submission. If it is not SynthID-positive, stop:
the surrounding transcode already changed the verifier result. Only a
control-positive, candidate-negative pair is evidence about the regeneration
attack. PSNR and temporal residual measure fidelity and flicker, never watermark
presence.

The shipped `video invisible` command and `remove_video_invisible` API reuse the
same VAE regeneration mechanism for a complete input sequence. The shipped
default is oracle-certified and does not expose a separate verification-status
flag. In the 2026-07-29
two-carrier calibration, both matched controls were positive in the built-in
verifier; the stronger candidate was negative on both, while a weaker
candidate was negative on one. A 2026-07-30 `UNAVAILABLE` response came from an
ordinary-model follow-up that asked Gemini to reinterpret the already returned
verdict and therefore did not invalidate it. The default is a calibrated,
content-dependent operating point. A per-file provider check remains an
optional audit after provider changes or for unusually important files.

The 2026-07-31 full-clip calibration used Google's public eight-second Veo
off-road sample through the complete product command. The original was detected
across 00:00-00:07, the `noise_std=0.10` output remained detected, and the
`0.15` output was not detected. The positive `0.10` result proves that the
surrounding 512 px / 12 fps / H.264 path did not create the negative result by
itself. `0.15` is therefore the shipped default. The tracked manifest
`data/evaluations/video-synthid-oracle.csv` records the public source URL,
hashes, fidelity metrics, and verdicts without committing generated videos.

What that calibration does and does not constrain, the ranked experiment program
for trading less quality for the same removal, and the change candidates that were
refuted along the way are recorded in
[`video-synthid-quality-research.md`](video-synthid-quality-research.md).

The VAE perturbation follows the general regeneration-attack construction from
Zhao et al. The video-specific control and temporal metric are local additions.
VideoMarkBench motivates testing frame aggregation and matched perturbations,
but it does not evaluate Google's proprietary SynthID, so its findings cannot
stand in for the Gemini oracle.

---

## 4. Adoption and current state (as of June 2026)

### 4.1 Google products

Google has watermarked **over 10 billion** images and video frames. The
deployment split by surface matters for our tool:

| Surface                              | SynthID pixel | C2PA metadata | Visible sparkle |
|--------------------------------------|---------------|---------------|-----------------|
| Gemini app (generated images)        | YES           | YES (Google)  | YES             |
| Gemini API / AI Studio / Nano Banana | YES           | NO            | YES             |

The Gemini API surface is a key blind spot: it embeds the pixel watermark and
the visible sparkle but **no C2PA or IPTC at all**. Our `identify` returns
Unknown on API-generated images unless the visible sparkle is detected (via
`check_visible=True`) or the user runs the Gemini app oracle.

### 4.2 OpenAI

OpenAI confirmed SynthID adoption (Help Center, updated 2026-05-21):

> "ChatGPT images include both C2PA metadata and SynthID watermarks."

This is time-gated: pre-rollout ChatGPT/gpt-image images carry C2PA without
SynthID. Current OpenAI manifests distinguish the watermarked output with the
signed `c2pa.watermarked.unbound` action. The detector requires that action, so
old OpenAI C2PA remains an origin signal without becoming a SynthID claim.

### 4.3 Other vendors

- **Kakao** (South Korea): SynthID adopter as of May 2026 (Google announcement)
- **NVIDIA Cosmos**: SynthID for video (not still images; different pipeline)
- **Meta AI**: does NOT use SynthID; uses IPTC `digitalSourceType` marker instead

### 4.4 Version evolution (v1 vs v2 hardening)

Google has not publicly documented version numbers for the SynthID image
watermark in a way that maps to our testing observations. What is known
empirically from oracle tests:

- **Before May 2026 (Gemini)**: strength 0.05 removed the watermark
- **May 2026 (Gemini)**: strength 0.05 insufficient; 0.10 required
- **Current (Gemini, June 2026)**: on the capped 1536 path, 0.05 and 0.10 do
  NOT clear; 0.15 clears (n=4, Gemini app oracle). See section 2.2.
- **OpenAI (June 2026)**: clears at 0.05 across 1024-1600 (n=4, clean v0.8.6).
  The earlier "0.30 still detected on 1600x1600" report (issue #14) was the
  text-protection bug, not a hardening of the watermark -- see the correction in
  section 2.2.

Google has hardened SynthID relative to OpenAI's (vendor gap measured at ~3x
strength), but the year-over-year "0.05 -> 0.10 -> 0.30" progression above
conflates a real hardening trend with the now-debunked region-rescrub artifact;
treat only the section 2.2 controlled numbers as authoritative.

---

## 5. Practical implications for this tool

### 5.1 Preserving content means regenerating it, never copying it

**Core rule:** SynthID is a pixel-amplitude pattern, so any approach that FREEZES
or RESTORES original pixels in a region re-introduces the watermark there. Early
region-based text/face "protection" (since removed) proved this: restoring the
original face pixels guaranteed SynthID survived in faces, and even a per-region
high-resolution re-scrub from an upscaled crop could be insufficient to destroy
the payload, reconstituting SynthID in text. The lesson held and shaped the
current design: **content is preserved by REGENERATING it under structural
conditioning, never by copying original pixels.**

> **Superseded in 0.24.0.** The `controlnet`, `sdxl`, `qwen` and `default` profiles
> were removed, and a retired name is now rejected at parse time rather than routed
> onward. `--pipeline` accepts only `qwen-zimage` (the default) and `sdxl-zimage`;
> both are CUDA-only, both condition their global stage on a canny edge map, and both
> run the same Z-Image face stage. The bullets below are a record of what was measured
> on the removed profiles, kept because the oracle verdicts are still the evidence the
> current design rests on.

- **Text + structure:** `--pipeline controlnet` (SDXL img2img + a canny ControlNet) is
  **THE DEFAULT pipeline since 2026-06-09** (`--pipeline default` opts down to plain
  SDXL img2img for inputs without text/faces). It conditions the regeneration on the
  edge map, so text and structure stay sharp while every pixel is still regenerated. Text legibility is
  better than plain img2img at the same strength (text stays readable where plain
  garbles it). **BUT removal efficacy at the low vendor-adaptive strength is CONTENT ×
  PIPELINE dependent and NEITHER pipeline clears all content -- oracle-validated
  2026-06-04 (8 OpenAI images, strength 0.10/0.15, max-res 1536).** The survivors FLIP
  by content type: **photoreal** (a 9-face grid, a bracelet product photo) SURVIVES
  controlnet but CLEARS `default`; **flat graphic** (a logo/poster with large flat
  color fills) SURVIVES `default` but CLEARS controlnet; a flat **text** card cleared
  under both. Why: controlnet's dense edge map keeps the regen too close to the
  original on photoreal (so SynthID survives) but freely repaints flat fills (so it
  clears them); plain img2img at low strength perturbs photoreal texture enough but
  barely touches flat fills. **Root cause = insufficient STRENGTH, not the pipeline:
  the vendor-adaptive 0.10 is NOT universally sufficient (the June numbers below held
  for the content they were measured on). The robust fix is a HIGHER strength,
  oracle-revalidated per content type (controlnet can be cranked harder without losing
  structure; a lower `controlnet_conditioning_scale` also frees the regen on
  photoreal).** So neither `--pipeline controlnet` nor plain `default` is a drop-in
  removal guarantee at today's strength -- pick by what you must PRESERVE (controlnet
  for text/structure), then raise strength until the oracle reads clean. (The earlier
  "reads clean on the oracle" claim held only for the one flat/text-background case it
  was checked on; it does not generalize.) **UPDATE 2026-06-09: the default strengths
  were raised and made pipeline-aware (controlnet ladder = the certified
  0.20/0.30/0.30 floors, applied to BOTH pipelines as a single ladder -- see §5.2 for
  why one ladder covers plain `sdxl` too) and controlnet is now the default pipeline.
  The plain-SDXL profile was also renamed `default` -> `sdxl` (`default` stays as an
  alias). The 0.10/0.15 numbers in this analysis are the PRE-raise values it was
  measured at. See §5.2.**
- **Highest-fidelity CUDA option:** `--pipeline qwen-zimage` is the recommended
  quality mode when preserving face identity matters more than latency, model size,
  and GPU cost. ControlNet was then the default, because it was much cheaper and ran on
  CUDA, XPU, MPS, and CPU, but canny conditioning preserves edges rather than identity.
  On two direct upstream comparisons, `qwen-zimage` retained substantially more
  ArcFace identity than polished ControlNet. On 2026-07-25 the exact six-output
  `visible -> qwen-zimage -> metadata` candidate was negative in the corresponding
  OpenAI and Gemini oracles. This is a quality recommendation for the measured content,
  not broad removal certification; very small text can still degrade.
  See `docs/qwen-improvement-research.md` for the identity and text metrics and the
  validation scope of those comparisons.
- **Face identity:** canny holds face *structure* but not *identity*. The removed
  SDXL and ControlNet profiles did not run a separate face-restoration stage, and
  earlier GFPGAN, PhotoMaker, and FaceID experiments were dropped after they
  degraded identity or risked reintroducing source pixels. Both shipped profiles
  now run the same face-specific stage: YuNet and SAM locate faces, then Z-Image
  regenerates the selected original face crops before a feathered composite. See
  `docs/controlnet-removal-pipeline-research.md` for the historical experiments.

### 5.2 Strength setting

There is no single permanent correct strength, but the controlled June 2026
study (section 2.2) gives empirical floors:

- **OpenAI**: 0.05 clears across 1024-1600 (n=4) -- **but content-dependent, NOT
  universal.** The follow-up oracle pass (2026-06-04, 8 images) found a flat-graphic
  OpenAI logo/poster still SynthID-detected after `default` at 0.10, and photoreal
  images still detected after controlnet at 0.10/0.15: at low strength the
  low-change regions (large flat fills under `default`, dense edges under controlnet)
  are not perturbed enough. So the 0.05 floor held only for the n=4 content it was
  measured on; treat it as a lower bound, not a guarantee, and raise + oracle-recheck
  per content type (see §5.1 controlnet bullet).
- **Google (capped 1536)**: 0.15 (n=4); 0.05 and 0.10 do not clear.
- **Google native 2816**: 0.15 clears (n=2, deployed controlnet worker, 2026-06-14) --
  the same rung as capped 1536, so no resolution penalty was observed.

> **Superseded in 0.24.0.** The `sdxl`, `controlnet`, `qwen` and `default` profiles
> were removed, and `OPENAI_STRENGTH` / `GEMINI_STRENGTH` / `UNKNOWN_STRENGTH` went
> with them. Everything from here to the end of this section is a record of what was
> measured on those profiles, kept because the oracle verdicts are still the evidence
> base. For the strength policy in force now see `module-internals.md`: `qwen-zimage`
> uses `resolution_adaptive_denoise`, `sdxl-zimage` a flat vendor ladder.

The default was **vendor-adaptive** (`watermark_profiles.resolve_strength` +
`vendor_for_strength`): the tool read the C2PA issuer on the original input and picked
`OPENAI_STRENGTH` 0.10 / `GEMINI_STRENGTH` 0.15 / `UNKNOWN_STRENGTH` 0.15 **(LOWERED
2026-06-14 from the 2026-06-04 cert floors 0.20/0.30/0.30)**. **The SAME ladder applied
to both pipelines** (`sdxl` and `controlnet`). The 2026-06-14 re-test on the deployed
Modal controlnet worker (v0.10.0) cleared SynthID on the oracle at OpenAI 0.10 (2
photoreal) and Google 0.15 (2 NATIVE 2816x1536, contradicting the "native >= 0.30" guess
on line above), and a pixel sweep showed 0.20/0.30 over-regenerated for no efficacy gain.
**This re-opens a genuine tension with the 2026-06-04 pass, which found photoreal STILL
detected after controlnet at 0.10/0.15 (lines above):** either the v0.10.0 controlnet
default improved the floor, or n=2 landed on the lucky side of the seed-non-determinism
(§5.5). So a SERVICE on this ladder MUST pin a fixed, oracle-verified seed (not random),
and flat-graphic hard cases (NOT in the n=2 re-test) still need a per-content oracle
recheck -- raise `--strength` there. The prior cert floors are the §5.5 record. Why one ladder
covers plain `sdxl` too: the certification was run on controlnet and does NOT transfer
by symmetry (the two pipelines have OPPOSITE hard cases -- controlnet leaves SynthID on
photoreal, `sdxl` on flat graphics, the §5.1 content-x-pipeline table), BUT on its own
hard case (flat fills) `sdxl` is the WEAKER remover (plain img2img barely perturbs a
flat region at low strength), so it needs AT LEAST controlnet's strength -- the
certified floor is therefore the right floor for `sdxl` too. This is a MARGIN argument
for `sdxl`, not a separate certification (no local SynthID detector to self-verify).
The higher strength costs little quality where it matters, because `controlnet` is now
the default pipeline, so `sdxl` is reached only via an explicit `--pipeline sdxl` (a
deliberate opt-down), where over-regeneration has no faces/text to damage.
This uses the vendor signal we DO have locally (the C2PA SynthID provenance) to avoid the
overkill of a single high default on OpenAI images, without needing a local pixel
detector. An explicit `--strength` always wins. If the watermark still survives (e.g. a
large native Gemini beyond the capped-1536 validation), raise toward 0.35-0.40 (0.40
visibly corrupts dense text), using the lowest value that reads clean on the oracle.

**qwen-zimage global denoise, Gemini boundary bracketed (2026-08-02).** The profile does
not use the vendor ladder above; `resolution_adaptive_denoise` maps megapixels onto
roughly 0.084 (sub-0.3 MP) to 0.154 (>= 3.7 MP). A ladder on one native 2816x1536 Gemini
original, seed 0, everything else at profile defaults, verified through the Gemini app:

| global denoise | Gemini app | whole-image PSNR | face-box PSNR | edge IoU |
|---|---|---|---|---|
| 0.154 (profile top) | clean | 24.72 | 31.19 | 0.188 |
| 0.12 | clean | 25.65 | 31.82 | 0.202 |
| 0.10 | **clean** | 26.26 | 32.17 | 0.212 |
| 0.08 | **SynthID FOUND** | 26.95 | 32.51 | 0.227 |

So the boundary sits between 0.08 and 0.10 for this image, and the profile's shipped
0.154 carries roughly half a rung more strength than that content needed. Fidelity rises
monotonically all the way down - dropping to 0.10 buys **+1.54 dB whole-image and
+0.98 dB inside the face boxes** - which is exactly why the temptation is to move the
ceiling, and exactly why one fixture is not enough to do it.

Two constraints on reading this:

- **It brackets, it does not calibrate.** One image, one seed. Shipping the lowest clean
  rung means shipping at the measured cliff edge; another sample, seed, or content class
  can sit on the other side of it. Note §5.2's flat-graphic hard cases were not in this
  set at all.
- **The bottom of the curve was the untested end. It has now been measured, and it
  holds.** Every Gemini oracle fixture is 2816x1536, so the Google-side certification
  only ever covered 0.154, while `resolution_adaptive_denoise` sends sub-1 MP images to
  0.084-0.094 - at or below the rung that failed at 4.33 MP. Downscaling a Gemini
  original (valid test material: SynthID survives it by design) and running the deployed
  worker on it gives, through the Gemini app:

  | processing size | profile denoise | Gemini app |
  |---|---|---|
  | 1024x559 (0.57 MP) | 0.0896 | clean |
  | 1600x873 (1.40 MP) | 0.1066 | clean |

  So production's own low end clears Google, and the "small images are under-processed"
  failure mode is ruled out at these two sizes. **Read this as validation of the shipped
  curve, not as proof that the boundary moves with resolution.** 0.0896 sits inside the
  untested gap at 4.33 MP, where only 0.08 (found) and 0.10 (clean) were probed, so it
  may well clear at both sizes. The direction of any resolution dependence remains
  unproven (§5.5). Also note these are downscales of a 2816x1536 original rather than
  natively small Gemini outputs, which have still never been tested.

### 5.3 Test methodology

- **GitHub-recompressed JPEGs from issue attachments are valid SynthID test
  subjects.** JPEG re-encoding removes C2PA metadata but does NOT remove the
  SynthID pixel watermark (verified June 2026 on issue #14 pic3). Do not
  dismiss these as "not faithful originals" for SynthID-removal tests.
- **The correct oracle for OpenAI images is openai.com/verify**, not the Gemini
  app. The two oracles detect different payloads.
- **A quiet `identify` output after processing is not proof of removal.** It
  means the provenance evidence is gone. The pixel watermark state is unknown without
  an oracle check.
- **After removal, the output may carry forensic artifacts** detectable by an
  independent classifier even if the vendor oracle reads negative. Defeating the
  verifier is not the same as being forensically indistinguishable from clean
  content (arXiv:2605.09203).

### 5.4 Strength vs forensic detectability: the tradeoff

Higher img2img strength removes the watermark but introduces detectable
regeneration artifacts. The Goonatilake & Ateniese paper shows the strongest
diffusion-based removers are simultaneously the most forensically detectable
(AUROC up to 0.9999). The tradeoff is unavoidable with current diffusion-based
approaches: defeating the vendor's verifier is not the same as being clean.

### 5.5 Oracle validation log -- 2026-06-04 OpenAI pass

Eight OpenAI `gpt-image` originals run through both pipelines and checked on
openai.com/verify (the OpenAI SynthID oracle). `--max-resolution 1536`; strength
is the vendor-adaptive default (`vendor_for_strength`): images with an OpenAI C2PA
manifest get `OPENAI_STRENGTH` 0.10, the one without C2PA falls to
`UNKNOWN_STRENGTH` 0.15. "detected" = SynthID still found (removal FAILED);
"clean" = SynthID not detected.

| image | content type | size | strength | `--auto`/controlnet | `default` |
|---|---|---|---|---|---|
| typography card | flat text | 1122x1402 | 0.10 | clean | clean |
| Flat poster | flat graphic (logo + flat fills) | 1024x1536 | 0.10 | clean | **detected** |
| 9-face grid | photoreal | 1448x1086 | 0.10 | **detected** | clean |
| bracelet product photo | photoreal | 1600x1600 | 0.15 | **detected** | clean |

(The other four cleared under both and are omitted.) **Reading:** at this strength
NEITHER pipeline removes SynthID on all content -- the survivors flip by content
type. Photoreal survives controlnet / clears `default`; flat graphic survives
`default` / clears controlnet; flat text clears both.

**Follow-up: removal near the threshold is NON-DETERMINISTIC (seed-dependent).**
Re-running the two photoreal survivors through controlnet at an explicit
`--strength 0.15` (`--auto`, same `--max-resolution 1536`) cleared BOTH on the
oracle (SynthID not detected). But the bracelet had SURVIVED controlnet at the
SAME 0.15 in the first pass (it was the no-C2PA image, so its vendor-adaptive
strength was already 0.15) -- same pipeline + strength + resolution, only the
random (unset) seed differed between runs. So **0.15 is the borderline floor for
controlnet photoreal, not a robust guarantee**: at the threshold the same
image+settings can pass or fail run-to-run. img2img runs with `seed=None` (random)
unless `--seed` is passed, so a removal SERVICE gets a coin-flip near threshold and
has no local SynthID detector to self-verify.

**Controlnet strength ladder on the two photoreal images (oracle, `--auto`,
`--max-resolution 1536`):**

| controlnet strength | 9-face grid | bracelet photo |
|---|---|---|
| 0.10 | detected | (was 0.15) |
| 0.15 | clean | **non-deterministic** (survived pass 1, clean pass 2) |
| **0.20** | **clean** | **clean** |

**Recommended robust controlnet strength = 0.20** (0.05 of margin above the 0.15
non-deterministic borderline); both photoreal survivors cleared at 0.20. Honest
caveat: 0.20 is one confirming run WITH margin, not an N-run repeatability proof --
for a removal service, add a little more margin or validate repeatability, since
there is no local SynthID detector to self-check. **Implications:** (1) the
content×pipeline table above conflates a borderline/non-deterministic 0.15 result
with deterministic content behavior -- the photoreal-survives-controlnet effect is
solid at 0.10 but at 0.15 it is near-threshold noise; (2) for reliable removal pick
a strength with MARGIN above the borderline (controlnet >= 0.20), not exactly on
it; (3) **historical engineering conclusion:** this dated run argued for a
higher ControlNet strength than the then-current default. That proposal was
later superseded. The current resolver intentionally shares the 0.10/0.15
ladder between SDXL and ControlNet and uses a separate Qwen ladder; see
`_internal/watermark_profiles.py`.
Source images are private (faces / product shots), not committed; reproduce on any
photoreal + flat-graphic gpt-image pair, varying the seed, and re-checking the
oracle.

**Gemini pass + the face-restore re-introduction (2026-06-04).** Four Gemini
originals via the then-current `--auto` ControlNet path at `--max-resolution 1024`,
checked on the
Gemini "Verify with SynthID" oracle (Google content needs the Google oracle, not
openai.com/verify):
- Most cleared at controlnet 0.15-0.25; `gemini_3` (a large central FACE, +restore)
  stayed **SynthID-detected at controlnet 0.15, 0.20 AND 0.25** -- raising strength
  did not crack it.
- **Root cause was the face-restore pass, not strength/resolution.** `gemini_3` at
  controlnet 0.20 with `--no-restore-faces` read **SynthID-NOT-detected** (clean
  A/B, only restore differed). GFPGAN runs on the ORIGINAL watermarked face and at
  weight 0.5 blends ~half its pixels back, re-introducing SynthID into the
  composited face over the diffusion-cleaned result (see §5.1 face-identity bullet).
- (Side note: reducing the processing resolution does NOT weaken SynthID -- it is
  robust to downscaling by design, so 1024 was never the wall. Whether a lower
  processing resolution then needs more or less removal strength is NOT established;
  see the note below.)

**Historical controlnet certification, superseded by the current vendor-adaptive
defaults (isolated GPU sweep + oracle,
restore OFF, <= 1536, each vendor on its own oracle):** OpenAI **0.20** (2 photoreal x
seed {1,2,3} = 6/6 clean; the 0.15-flipper is seed-robust at 0.20) and Gemini **0.30**
(0.20 detected -> 0.30 clean on 2/2 seeds). Both were measured at <= 1536 only. See
`docs/controlnet-removal-pipeline-research.md` for the table.

**Whether Gemini removal is resolution-sensitive is UNPROVEN, in either direction.**
This document previously asserted it was, and recommended capping Gemini at 1536 with
0.30 or "native-calibrating" to ~0.35+. Nothing measured that. The one relevant
measurement points the other way: the 2026-06-14 deployed-worker re-test cleared Gemini
at **0.15 on two NATIVE 2816x1536 images**, the same rung as capped 1536. So there is no
observed native-resolution penalty, and no observed benefit either -- the low-resolution
end has simply never been through the Gemini oracle on any pipeline. Do not reason from
a resolution trend here; measure it.

**Current implication:** the old floor table remains evidence about the dated
test set, not the current resolver. The SDXL and ControlNet profiles it measured
no longer exist; the shipped defaults are defined in `watermark_profiles.py`, and
both surviving profiles run face repair as a built-in second stage rather than as
an optional restore. Removal near a threshold remains seed dependent, so
reproducible verification requires a fixed seed.

---

## References

1. Gowal et al. (2025). **SynthID-Image: Image watermarking at internet scale.**
   arXiv:2510.09263. https://arxiv.org/abs/2510.09263

2. Google DeepMind. **Identifying AI-generated images with SynthID.** Blog post,
   2023. https://deepmind.google/blog/identifying-ai-generated-images-with-synthid/

3. Google DeepMind. **SynthID.** Product page.
   https://deepmind.google/models/synthid/

4. Goonatilake & Ateniese (2026). **Removing the Watermark Is Not Enough:
   Forensic Stealth in Generative-AI Watermark Removal.** arXiv:2605.09203.
   https://arxiv.org/abs/2605.09203

5. OpenAI. **Verify tool for AI-generated images.** openai.com/research/verify.
   Accessed 2026-05-31.

6. Google. **Verify AI-generated images, videos, and audio.**
   https://support.google.com/gemini/answer/16722517

7. Zhao et al. (2024). **Invisible Image Watermarks Are Provably Removable
   Using Generative AI.** NeurIPS 2024, arXiv:2306.01953.
   https://arxiv.org/abs/2306.01953

8. Jiang et al. (2025). **VideoMarkBench: Benchmarking Robustness of Video
   Watermarking.** arXiv:2505.21620.
   https://arxiv.org/abs/2505.21620
