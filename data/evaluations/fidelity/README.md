# Pipeline fidelity evaluation

This evaluation compares diffusion pipelines with
`scripts/fidelity_metrics.py`. The images themselves have one canonical home
in `data/synthid/originals/`; this directory stores only evaluation-specific
ground truth and instructions.

| Original | Provider | Content | Exercises |
| --- | --- | --- | --- |
| `ChatGPT Image May 31, 2026, 02_02_23 PM.png` | OpenAI | Light multilingual typography | Text preservation |
| `ChatGPT Image May 31, 2026, 02_03_55 PM.png` | OpenAI | Multilingual typography | Text preservation |
| `Gemini_Generated_Image_633uuy633uuy633u.png` | Google | Landscape with a Chinese sign | CJK text preservation |
| `Gemini_Generated_Image_y48j3cy48j3cy48j.png` | Google | Portrait grid | Face identity and skin texture |

## Text ground truth

`ground-truth.json` contains hand-verified OCR for the three text-bearing
originals. To regenerate an OCR seed:

```bash
uv run scripts/fidelity_metrics.py ocr \
  "data/synthid/originals/ChatGPT Image May 31, 2026, 02_03_55 PM.png" \
  "data/synthid/originals/ChatGPT Image May 31, 2026, 02_02_23 PM.png" \
  data/synthid/originals/Gemini_Generated_Image_633uuy633uuy633u.png \
  --langs en,ru,ch \
  --out data/evaluations/fidelity/ground-truth.json
```

Verify and correct the generated text by hand before using it as ground truth.
`text-lines.json` contains the verified per-line strings and source-space boxes
used by the evaluation-only selective-restoration experiment. It is not an
automatic scene-text annotation set.

`scripts/infer_text_lines.py` can generate a draft from source pixels without
existing annotations. On the two posters it proposed 20 and 18 lines at the
default threshold, but exact-text precision was only 90.0% and 94.4% because
high-confidence OCR still dropped an English comma and replaced a Chinese comma
with ASCII. Its output therefore requires manual verification of every line;
`accepted` means crop-stable, not ground-truth-correct.

## Text-preservation benchmark

`text-preservation-2026-08-13.csv` records a fixed-seed comparison of the two
current profiles and a global Z-Image Turbo prototype on all three text
fixtures. Each candidate ran through the
complete `visible -> invisible -> metadata` route with its profile default
strength and adaptive-polish setting. The Z-Image prototype instead sweeps
0.08, 0.10, 0.15, 0.20, and 0.25 without polish because its provider-specific
removal floor is not known. The output hash identifies the exact bytes
measured; generated outputs remain outside the repository.

The character-weighted added CER is 0.262 for `qwen-zimage` and 0.256 for
`sdxl-zimage`. That 0.006 absolute difference is not a
stable ordering: SDXL wins the light poster, Qwen wins the dark poster, and the
Chinese sign is tied. A paired sign test on the two non-ties is 1-1 (`p=1.0`).
The measured sample therefore does not support a general text-preservation
winner. Both profiles substantially degrade the smallest multilingual poster
text and preserve the larger Chinese sign.

Z-Image Turbo is the clear fidelity lead. At strength 0.10 its
character-weighted CER is 0.093, against the unchanged sources' 0.124 OCR
floor, while whole-image LPIPS stays between 0.047 and 0.086. Visual inspection
still finds substitutions in the smallest Cyrillic and Chinese poster lines,
so a favorable OCR score does not mean pixel-exact text. Quality drops quickly
above 0.15; at 0.25 the weighted CER is 0.428.

Provider-oracle checks on 2026-08-13 bracket the OpenAI requirement at no more
than 0.10 for this sample. Both original OpenAI controls were detected by
`openai.com/verify`; the light poster was clean at 0.08, while the dark poster
was detected at 0.08 and clean at 0.10. The light 0.10 output was not separately
checked. Google fails the entire tested sweep. The original Gemini image was
detected through C2PA, and a pixel-identical copy with AI metadata stripped was
separately detected by Gemini's built-in SynthID verifier. Z-Image outputs at
0.08, 0.10, 0.15, 0.20, and 0.25 were all still detected. At 0.25 the weighted
CER has already risen to 0.428, worse than the raw weighted CER of both current
profiles (Qwen 0.387, SDXL 0.381). Increasing strength beyond the measured grid
would therefore no longer serve the text-preservation objective without a new
mechanism or hypothesis. Z-Image is not a viable global replacement on this
evidence: no clean Google operating point was found before it lost its fidelity
advantage.

Qwen-Image-2.0 was not added to the numeric comparison. Its weights are not
published, and its hosted edit API exposes an editing instruction and seed but
no low-strength denoise control. It can be evaluated as a separate hosted edit
strategy when credentials are available, but it is not a drop-in replacement
for the partial-regeneration mechanism measured here.

The OCR floor is the source image scored against the hand-verified text. Use
`added_cer = text_cer - ocr_floor` when interpreting pipeline damage, because the
unchanged poster sources already score 0.127 CER. `oracle_rechecked=true` marks
the exact Z-Image bytes checked above; the remaining rows were not rechecked.
The table does not certify other seeds, content classes, or strengths beyond
the recorded provider verdicts.

## Text-restoration prototype

`text-restoration-2026-08-13.csv` evaluates an OCR-driven post-pass on the exact
Qwen outputs above. The prototype recognizes English and Russian with macOS
Vision and CJK with PaddleOCR, derives glyph masks independently from the
source and Qwen output, removes both sets of glyphs with block-wise LaMa, and
draws the recognized strings with new system-font pixels. It never composites
source pixels back into the result.

On the two multilingual posters, character-weighted CER fell from 0.338 and
0.305 to 0.007 on both. OpenAI Verify reported no OpenAI signals for both Qwen
controls and both restored outputs in the same run. The improvement comes with
a substantial whole-image fidelity cost: LPIPS rose from 0.107 to 0.174 and
from 0.095 to 0.162, while PSNR fell by about 10.5-11.4 dB. Visual inspection
found one residual shadow in the smallest English line of the light poster;
the dark poster was clean but the substitute fonts visibly changed typography.

The Chinese sign did not improve: CER rose from 0.074 to 0.111 because the OCR
and renderer changed punctuation. The Gemini verifier returned detected for
both the restored output and its byte-identical Qwen control on a third work
account, although that Qwen hash had previously returned clean on another work
account. This run therefore does not isolate a restoration-stage regression;
its Google verdict is inconclusive until a source-positive, Qwen-negative
control can be reproduced in the same available account.

This prototype is not ready to ship. Its strongest result establishes that
fresh-glyph reconstruction can recover literal text without reintroducing an
OpenAI signal, but portable OCR, font/style reconstruction, a tighter mask, and
a reproducible Google oracle control remain prerequisites.

### Selective restoration follow-up

`selective-text-restoration-2026-08-13.csv` compares that full compositor with
a selective prototype on the same two posters. The prototype leaves a Qwen line
unchanged when padded source and output recognition agree, and applies the same
LaMa plus fresh-system-font reconstruction only to lines whose recognized text
changed. This reduced the edited area from 15.4% and 17.0% to 5.5% on both
posters.

Under one consistent Paddle `en+ru+ch` measurement route, selective restoration
reduced Qwen CER from 0.378 to 0.101 on the light poster and from 0.413 to 0.112
on the dark poster. Its image LPIPS was 0.120 and 0.103, substantially closer to
Qwen than the full compositor's 0.174 and 0.162. These CER values must not be
mixed with the preceding table's Vision/Paddle hybrid values: the comparison
file remeasures all three variants through Paddle so their relative result is
valid on one OCR route.

OpenAI Verify returned `No OpenAI signals detected` for both selective outputs,
then detected the original light poster as `Generated with OpenAI tools` in the
same Chrome sequence. Visual inspection found the dark output clean, but the
light output still retained a local shadow around one replaced fine-text line.
Selective restoration is therefore the strongest current direction, not a
production-ready default. The next implementation needs automatic line matching
and a tighter source-plus-candidate glyph mask before it can be proposed for the
pipeline.

A mask-only follow-up added two pixels of dilation around every selected glyph.
It visually removed the light poster's shadow and improved its CER from 0.101 to
0.081 with effectively unchanged LPIPS, but increased the edited area from 5.5%
to 7.0%. Applying the same rule to the dark poster worsened CER from 0.112 to
0.147. These expanded-mask rows were not resubmitted to the provider oracle.
The opposing result rules out one global dilation constant and makes per-line
residual detection the next fidelity gate.

An automatic follow-up removed the hand-selected line list. For each known
source line, it runs a recognition-only model on padded, upscaled source and
Qwen crops. It preserves a line only when both confidence scores are at least
0.75, the source recognition is within 0.25 normalized edit distance of the
verified line, and normalized source and candidate recognition match exactly.
This reproduced the manual dark-poster line selection and additionally kept one
correct large Russian line on the light poster.

The automatic prototype also replaced uniform dilation with a second measured
LaMa pass. After the first erase, it extracts contrast remaining at the original
glyph positions, dilates only that residual, and inpaints it before drawing fresh
text. Visual inspection found no remaining double glyphs on either output. The
effective edited fractions were 4.1% and 5.7%. Image LPIPS was 0.113 and 0.104,
against 0.120 and 0.103 for the manual selective version. Detector-based page CER
was 0.123 and 0.119; the light value is inflated by the detector splitting the
preserved Cyrillic line into Latin-like fragments even though the padded Russian
recognizer reads it exactly. This is another reason that line decisions use
recognition-only crops rather than the page detector's aggregate CER.

Both exact automatic output hashes returned `No OpenAI signals detected`, and
the original light poster returned `Generated with OpenAI tools` immediately
afterward in the same Chrome sequence. The tracked script reproduced the dark
hash exactly. Its light output differed from the initial prototype in only 386
pixels by at most one channel value because separate ONNX environments rounded
differently; the script's exact `900def5a...` hash was therefore submitted and
also returned clean before the positive source control.

The script and its required verified line annotations are tracked as
`scripts/selective_text_restoration.py` and `text-lines.json`. For example:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  uv run scripts/selective_text_restoration.py \
  data/synthid/originals/Gemini_Generated_Image_633uuy633uuy633u.png \
  path/to/qwen-candidate.png \
  --restoration vae-glyphs \
  --glyph-donor path/to/qwen-vae-reconstruction.png \
  --glyph-feather 0.5 \
  --output path/to/selective-output.png \
  --manifest path/to/manifest.json
```

The restoration mode is deliberately explicit. `vae-glyphs` requires a
separately generated VAE reconstruction in `--glyph-donor`; it takes exact
thresholded glyph cores from that donor, uses a fresh silhouette beneath them,
and defaults to a narrow 0.5-pixel donor edge. The donor and the resulting exact bytes still require
full-pipeline oracle evaluation. `rerender` reproduces the tracked
fresh-system-font experiment. `source-glyphs` is only for compositing a
separately regenerated, oracle-evaluated layer; feeding it the watermarked
original would paste provenance-bearing pixels back into the result. The
`--keep-background --composite-mask boxes` combination is an aligned-layer
experiment and is not a production text restorer.

The `source-silhouette` follow-up keeps the source glyph geometry but discards
its pixel amplitudes: it thresholds each source line to a binary shape, samples
one foreground color, and synthesizes new antialiasing over the scrubbed image.
On a 24-image matrix spanning serif, Latin-diacritic, Cyrillic, CJK, tiny-UI,
and rotated text, the original outer-feather compositor preserved all 34
source-readable lines exactly under crop OCR, with median text-box SSIM 0.859.
An inner-antialias variant raised median text-box SSIM to 0.902 and left
whole-image SSIM unchanged at 0.672, but preserved 33/34 lines: OCR read one
middle dot as a colon. The higher-fidelity antialiasing is retained in the
evaluation script, with that punctuation miss recorded as a caveat. Its exact
outputs were not submitted to the provider oracle because the
public verifier reached its request limit; a clean verdict from the earlier
outer-feather bytes does not transfer to the new hashes.

Follow-up visual review on a typography-rich control rejected both silhouette
compositors despite their OCR and SSIM scores. They preserved literal content
and approximate glyph geometry, but changed stroke weight, color variation,
edge antialiasing, and small decorative details enough to be plainly different
from the source. OCR exactness and text-box SSIM are therefore screening metrics,
not acceptance gates for source-typography preservation. A candidate must also
retain source-like edge pixels and pass direct visual comparison at native size.

A narrower Qwen-VAE donor follow-up keeps the scrubbed background, then copies
only VAE-reconstructed pixels through the source-silhouette mask with a
0.5-pixel feather. Across the 48-case typography matrix (548 annotated lines),
text-box SSIM improved in 47/48 cases and glyph-edge error improved in 48/48.
Median text-box SSIM rose from 0.854 to 0.914, while median glyph-edge MAE fell
from 37.59 to 32.42. The median nonzero alpha area was 3.62%; on dense or large
text this is still substantially wider than the target control. These are
fidelity results only; the 48 matrix outputs were not submitted to the provider
oracle. On a separate untracked dense-typography control, the exact core-only
Qwen-VAE donor with a 1.43% source-silhouette area returned `No OpenAI signals
detected`. Its 0.5-pixel feathered sibling, which raises the nonzero donor area
to 2.75%, returned the same verdict and improved mean text-box SSIM from 0.957
to 0.960. Crop OCR recovered from 7/15 exact lines on the raw pass to 14/15,
matching the source OCR floor. Whole-image LPIPS was 0.082, but only 0.108% of
pixels were exactly unchanged and the detected face retained 0.670 of source
Laplacian variance. The tracked script reproduced the feathered file byte for
byte. These two exact-byte verdicts do not certify other images or the larger
matrix masks, and the global smoothing fails a strict unchanged-image criterion.

The opt-in production port was rechecked separately on 2026-08-15. Its current
LaMa runtime did not reproduce the earlier evaluation PNG byte for byte, but all
changed pixels were confined to the erased background outside the donor glyph
core. The exact production artifact returned `No OpenAI signals detected` in
3/3 OpenAI Verify runs, while the matched source control returned `Generated
with OpenAI tools` in 2/2 runs in the same Chrome session; expanded details
identified SynthID and no C2PA manifest on the control. The private control and
artifact hashes remain outside the public repository. This certifies only that
runtime, verified manifest, and output, not arbitrary text masks or images.

The Google result is negative. On the synthetic CJK sign case, two separate
work-account runs both detected SynthID in the resaved source control and in the
exact Qwen-VAE donor output. The candidate improved mean text-box SSIM from
0.791 to 0.835 and glyph-edge MAE from 35.48 to 22.49 with a 3.67% donor area,
but Gemini still detected it. The intermediate Qwen silhouette base and the
earlier original-sign Qwen baseline were also detected in the same account.
This isolates the current blocker upstream of text restoration: the global
Google pass did not reach the SynthID removal floor, so the OpenAI-clean
`vae-glyphs` configuration cannot be used as a Google operating point.

On the large Chinese-sign control, source and candidate recognition agreed on
all three lines. The script selected no changed lines, emitted a zero mask, and
copied the candidate byte for byte instead of re-encoding it. This corrects the
earlier unnecessary CJK rerender and proves the no-edit branch. It does not add
a Google-negative oracle result: the available Google account still cannot
reproduce a source-positive, Qwen-negative control sequence.

The earlier automatic rerender was the first variant in the experiment to pass
the recorded visual, fidelity, and OpenAI-oracle gates on its two poster
fixtures. The later typography-rich control shows that result does not
generalize to source-typography preservation. It remains evaluation-only: it
depends on verified source text and source line boxes, uses macOS system fonts,
and has not been validated on natural scene text, rotated text, or automatic
line-box discovery.

An opt-in `--detect-boxes` follow-up tested automatic geometry. Grouping Paddle
word detections by vertical overlap found exactly 20/20, 20/20, and 3/3 lines;
mean IoU with verified boxes was 0.857, 0.847, and 1.000. Reusing the annotation
crop padding was unstable and preserved only 4/20 dark-poster lines. Reducing
vertical recognition padding to 10% restored the exact 8/20 and 7/20 selection
decisions, but detector CER was 0.127 and 0.154 rather than 0.123 and 0.119. The
dark regression failed the fidelity gate, so these hashes were not submitted to
the provider oracle. The flag is retained only to reproduce that negative
evaluation and still requires verified strings and an exact line-count match.

### AnyText2 glyph-conditioned follow-up

`anytext2-restoration-2026-08-13.csv` tests the official Apache-2.0
AnyText2 checkpoint as a local text-editing pass over the exact Qwen Chinese-sign
output. The checkpoint hash was verified against ModelScope. Its own edit example
successfully replaced a masked blackboard line with the requested `DADDY`, so the
runtime and checkpoint were functional before the tracked fixture was measured.

AnyText2 failed the fidelity gate. The standard full-image detector scored the
default edit at CER 0.185 and the source-font mimic at 0.222, against 0.074 for
Qwen; image LPIPS moved from 0.289 to 0.338 and 0.345. A padded crop-recognition
cross-check removed the detector's punctuation misses: Qwen scored 0.000, the
default edit still scored 0.074 with two substituted characters, and the mimic
edit stayed at 0.222. Visual inspection agreed, with additional line-shape and
punctuation errors in the mimic output. No provider-oracle submission was made
because neither candidate preserved the requested content better than its
already-clean Qwen input.

This result covers the published SD1.5 checkpoint, three Chinese lines of no more
than 20 characters, one seed, and one fixture. The repository still describes
AnyText2XL as unreleased, and the inference wrapper truncates each requested line
after 20 characters, so the published model cannot directly cover the longer
English and Russian poster lines.

The padded cross-check also changes the interpretation of this fixture's earlier
restoration result. Its Qwen image already contains all three correct Chinese
lines; the reported 0.074 came from the detector cropping the final punctuation
from two lines. A deterministic rerender with horizontally padded recognition
and a mask-matched Hiragino Sans GB W6 font reached detector CER 0.000, but image
LPIPS worsened from 0.289 to 0.327 and the font was visibly heavier. The correct
policy on this large-text fixture is therefore to leave a Qwen line unchanged
when padded recognition confirms the source and candidate strings agree.

## Face-stage isolation

`face-stage-isolation-2026-08-13.csv` separates the two stages on the 18-face
Gemini portrait-grid fixture. All variants use the same visible-cleaned,
metadata-stripped input and seed 0. The prepared input remained positive in the
Gemini SynthID verifier, so the surrounding visible and metadata stages did not
create the clean verdicts.

| Route | Gemini oracle | ID cosine | Face LPIPS | Laplacian ratio | Image LPIPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Prepared control | detected | - | - | - | - |
| Z-Image face stage only | detected | 0.927 | 0.028 | 0.612 | 0.013 |
| Qwen global stage only | clean | 0.589 | 0.217 | 0.331 | 0.288 |
| Full Qwen then Z-Image | clean | 0.852 | 0.050 | 0.492 | 0.252 |

This isolates the roles: the Qwen global pass removes the watermark, while the
masked Z-Image pass restores face identity and detail without reintroducing a
detectable whole-image signal. Z-Image face-only does not remove SynthID. The
result explains why the face stage succeeded while global Z-Image failed: the
earlier clean verdict belonged to the two-stage pipeline, not to Z-Image as a
standalone remover. This is one fixture, one seed, and one oracle pass per
variant; it does not establish a general reintroduction threshold for mask size.

## Upstream Synthid-Bypass v2 reproduction

`upstream-v2-reproduction-2026-08-13.csv` records a source-level audit and a
close reproduction of the public
[`Synthid-Bypass-v2.0.json`](https://github.com/cebeuq/Synthid-Bypass/blob/3007d0351596ae0a78b7074dae7ad179710b1e48/Synthid-Bypass-v2.0.json).
The audited workflow is SHA-256
`41911b3b8e01bf51450361dc8beccd84c9513f78ddb160430cdd9bebc294adf6` at
upstream commit `3007d0351596ae0a78b7074dae7ad179710b1e48`.

Tracing links backward from `SaveImage` confirms that the active global stage is
Qwen-Image-2512 Q4 with the Lightning four-step LoRA at 0.8, DiffSynth Canny at
1.0, AuraFlow shift 3, `dpmpp_2m` plus `sgm_uniform`, CFG 1, and four steps. Its
resolution helper uses adaptive level 6 over 0.08..0.15, which resolves to the
0.154 ceiling for the 2816x1536 portrait fixture. Z-Image Turbo is used only by
the face detailer: eight steps, CFG 1, `res_2s` plus `bong_tangent`, a 768 px
guide, 1024 px cap, and direct adaptive denoise
`clamp(0.10 * largest_face_ratio / 0.03, 0.05, 0.28)`. The active face regions
come from YOLOv8-face plus SAM. Although the README describes MediaPipe as part
of the strict path, the MediaPipe nodes do not reach the saved output. The
1.2-megapixel scaler is also bypassed (`mode=4`).

The published upstream pair 12 changed from SynthID-positive to clean in the
Gemini verifier and scored 0.975 face identity. On the project portrait fixture,
the prepared control was positive, while both the close global reproduction and
the close full reproduction were clean. The global stage scored 0.589 identity
and 0.217 face LPIPS; the upstream-strength face stage improved those to 0.783
and 0.083. The maintained profile's weaker face pass scored 0.852 and 0.050 on
the same control, so copying upstream's roughly doubled face strength would be a
quality regression on this fixture.

The close reproduction is intentionally labeled rather than presented as an
exact ComfyUI run. It uses full-precision DiffSynth bf16 weights instead of the
Q4 GGUF files, the maintained DiffSynth Lightning scheduler approximation
instead of the ComfyUI sampler pair, YuNet plus SAM instead of YOLOv8-face plus
SAM, and fixed seed 0 instead of randomized seeds. The two independent oracle
controls and the published upstream pair establish the direction of the result;
they do not make the local output byte-equivalent to upstream.

## Compare

```bash
uv run scripts/fidelity_metrics.py compare \
  --original data/synthid/originals/Gemini_Generated_Image_y48j3cy48j3cy48j.png \
  --variant qwen-zimage=<qwen-out>.png \
  --variant sdxl-zimage=<sdxl-out>.png \
  --ocr-langs ""
```
