# Supported signals

This page describes the current support boundary. A check mark means that the
repository contains a corresponding code path. It does not guarantee detection
or removal on every future vendor version.

## Visible marks

The `visible` command registers these mark keys:

| Key | Mark | Expected area | Important limit |
| --- | --- | --- | --- |
| `gemini` | Google Gemini sparkle | Usually bottom right | Detection includes a false positive gate. |
| `doubao` | `豆包AI生成` | Bottom right | Vendor specific text detector. |
| `jimeng` | `★ 即梦AI` | Bottom right | Vendor specific text detector. |
| `qwen` | `千问AI生成` | Bottom right | Strict visual gate. |
| `kling` | `可灵AI 3.0` | Bottom right | Only calibrated variants are covered. |
| `yuanbao` | `元宝` over `AI生成` | Bottom right | Standard two-line variant only. |
| `samsung` | `✦ Contenuti generati dall'AI` | Bottom left | Calibrated for the Italian text variant. |
| `runninghub` | `RunningHub AI生成` | Top left | Strict visual and position gates. |
| `baidu` | `百度 AI生成` | Bottom right | Detector and extended removal footprint. |
| `liblib` | `LibLibAI` | Bottom center | Includes a minimum image size gate. |
| `jimeng_pill` | `AI生成` pill | Top left | Weak detector with additional product and background gates. |

`--mark auto` evaluates all registered marks and removes every selected match.
Known marks are localized to a mask, then the selected fill backend reconstructs
the masked area.

Marks from other vendors are not detected automatically. Use `erase --region`
when you can select the affected area yourself.

### Visible video marks

| Key | Mark | Motion | Important limit |
| --- | --- | --- | --- |
| `sora` | Sora 2 mascot and wordmark | Moves among frame positions | Requires a temporally recurring visual match; the older Sora Turbo corner swirl is a different unsupported mark. |
| `veo` | Current four-point diamond and legacy `Veo` text | Fixed bottom-right corner | Uses separate silhouettes and requires a recurring match; learned fill is preferable on structured backgrounds. |
| `seedance` | Boxed `AI` label | Fixed bottom-right corner | Requires an anchored recurring match; the full localized box is filled because a thinner synthetic shape mask leaves the real translucent rim behind. |
| `dola` | `Dola AI` text | Fixed bottom-right corner | Requires an anchored recurring match; ByteDance or BytePlus provenance can relax only an existing visual run. |
| `hailuo` | `MINIMAX \| hailuo AI` composite label | Fixed lower edge | Uses a synthetic waveform, text, separator, and ring silhouette; the complete recurring label box is filled. |
| `kling` | Kling swirl, `KLING AI`, version, and optional `PRO` suffix | Fixed bottom-right edge | Combines a synthetic logo rescue with font variants, an edge gate, a white-label gate, and anchored temporal recurrence. |

`video identify`, `video visible`, and `video all` share this registry and the
same temporal arbiter. It is separate from the image registry because selection
is made over a sequence rather than one raster. The default `auto` mode scans
all six entries in one decode pass and selects the first temporally stable
match in table order; an explicit mark restricts the scan to that row.
Accepted fills are motion-aligned across adjacent frames by default. The prior
fill contributes only where its warped mask covers the current removal mask and
nearby source context agrees. Scene cuts or disjoint marks retain the
independent frame fill.

## Fill backends

| Backend | Install | Behavior |
| --- | --- | --- |
| `cv2` | `pagedmark[visible]` | Classical OpenCV inpainting |
| `migan` | `pagedmark[migan]` | MI-GAN through ONNX Runtime; practical learned CPU video tier |
| `lama` | `pagedmark[lama]` | big-LaMa through ONNX Runtime; offline video quality tier |
| `auto` | Depends on installed extras | Selects LaMa, then MI-GAN, then OpenCV |

The learned backends download model files on first use.

## Metadata and provenance

The inspection and stripping code handles signals in these groups:

- C2PA Content Credentials and supported cloud manifest references;
- EXIF and XMP generator fields;
- exact app-export provenance and AIGC disclosures from supported
  ByteDance-family products, with product-only provenance excluded from the
  generated-image verdict;
- IPTC AI disclosure fields;
- PNG text chunks and embedded generation parameters;
- China TC260 AIGC labels in supported image placements and the normative
  MP4/MOV `moov.udta.meta.keys/ilst`, MKV/WebM
  `Segment.Tags.Tag.SimpleTag`, AVI `LIST/INFO/AIGC`, and FLV
  `script.onMetaData.AIGC` placements;
- xAI and Grok EXIF signature fields;
- Samsung AI editing markers;
- Hugging Face job metadata;
- open Stable Diffusion style DWT-DCT watermarks with the `detect` extra;
- Adobe TrustMark Variant P schemas 0-2 with the `trustmark` extra. Variant Q
  needs a different model, while schema 3 is deliberately rejected because it
  produced persistent false positives on unrelated generators.

`identify` combines detected signals into a `ProvenanceReport`. It reports
unknown when evidence is absent. It never treats missing metadata as proof that
an image is human made. C2PA presence alone is not a verified identity: the
report distinguishes asset binding, claim signature, signer trust, and signer
validity. High-confidence C2PA attribution requires all four; intact but
untrusted or fallback claims are medium-confidence, while a failed binding or
signature contributes no origin verdict.

## File and container formats

Pixel based image commands discover these extensions:

- PNG;
- JPEG;
- WebP;
- HEIC and HEIF;
- AVIF.

HEIC, HEIF, and AVIF pixel decoding requires the independent `heif` extra in
addition to the selected pixel feature. Metadata scanning does not.

Metadata inspection and removal additionally have container paths for:

- JPEG XL metadata;
- MP4, MOV, M4V, and M4A;
- WebM, MKV, MKA, AVI, FLV, MP3, WAV, FLAC, OGG, OGA, Opus, and AAC when
  ffmpeg is available.

JPEG image metadata stripping removes targeted metadata segments without
re-encoding the entropy coded image scan. PNG and WebP removal preserves pixel
values through lossless output paths. HEIC, HEIF, AVIF, and other containers
use their format specific paths.

## Invisible watermarks

The `invisible` command uses diffusion regeneration. It targets watermark
patterns by changing the image rather than decoding and deleting a known
payload.

Current pipeline values, both CUDA-only:

- `qwen-zimage`, the default;
- `sdxl-zimage`, the same recipe and the same face stage on an SDXL global pass.

The `controlnet`, `sdxl`, `qwen` and `default` values were removed. A retired name
is rejected at parse time rather than remapped onto a surviving profile.

SynthID does not have a public local pixel decoder in this project. The tool
recognizes presence from supported provenance: Google AI C2PA under Google's
all-media watermark policy, and current OpenAI C2PA carrying an explicit
`c2pa.watermarked.*` action. Legacy OpenAI C2PA without that action does not
assert SynthID. After provenance metadata is removed, a local negative result
is still inconclusive.

For MP4, MOV, and M4V, `video invisible` or the explicit
`video all --invisible` option can regenerate the video through a VAE and strip
source metadata. The shipped profile is oracle-certified, but it is not a local
decoder. A fresh source-positive, output-negative pair from Gemini's built-in
SynthID verifier is an optional per-file audit. A normal Gemini answer may instead
infer from a visible logo or metadata; asking it to reinterpret a completed
verifier result is not a second oracle run.

The optional `detect` extra is different: it provides a local decoder for the
open DWT-DCT watermark used by some Stable Diffusion, SDXL, and FLUX workflows.
That signal is carrier and transformation sensitive, so a negative is still
not a universal clean verdict.

## Provider overview

| Provider or family | Visible | Invisible path | Metadata or provenance |
| --- | --- | --- | --- |
| Google Gemini | Sparkle | Diffusion regeneration for SynthID | C2PA and related source signals |
| Google Veo video | Veo diamond and legacy text | Oracle-certified VAE removal for SynthID | C2PA and related source signals |
| OpenAI image generators | None registered | Diffusion regeneration for supported invisible signals | C2PA and generator provenance |
| Stable Diffusion and SDXL | None registered | Diffusion regeneration; optional open decoder | Embedded parameters and text metadata |
| FLUX | None registered | Diffusion regeneration; optional open decoder | C2PA for supported sources |
| Adobe Firefly | None registered | Optional TrustMark Variant P decoder | C2PA |
| Midjourney | None registered | No registered pixel decoder | EXIF, XMP, and IPTC signals |
| ByteDance generators | Doubao and Jimeng marks | No registered pixel decoder | TC260 AIGC, supported C2PA, and exact app-export AIGC disclosures |
| Qwen | Qwen mark | No registered pixel decoder | TC260 AIGC |
| Kling | Kling image and video marks | No registered pixel decoder | TC260 AIGC |
| Hailuo / MiniMax video | Hailuo composite video label | No registered pixel decoder | TC260 AIGC where present |
| Baidu | Baidu mark | No registered pixel decoder | TC260 AIGC |
| LibLibAI | LibLibAI mark | No registered pixel decoder | TC260 AIGC |
| RunningHub | RunningHub mark | No registered pixel decoder | TC260 AIGC |
| Samsung Galaxy AI | One locale specific mark | No registered pixel decoder | C2PA and Samsung markers |

For detector thresholds, measured limits, and incident history, see
[module internals](module-internals.md) and
[known limitations](known-limitations.md).
