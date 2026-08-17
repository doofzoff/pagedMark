# Watermarking landscape (research 2026-05-24)

> Research and signal inventory. Code-facing statements are updated with the
> implementation; dated vendor observations remain snapshots and may change
> independently of the package.

Who embeds what, and whether it is locally detectable (so we know which gaps are fillable). See `identify.py` for what we read.
- **Locally detectable (open decoder, no key/API):** Stable Diffusion / SDXL / FLUX via `imwatermark` DWT-DCT (now covered by `invisible_watermark.py`). FLUX uses the same library (upstream `black-forest-labs/flux2`, file `src/flux2/watermark.py`, 48-bit `0b001010101111111010000111100111001111010100101110`); SDXL is the diffusers `WATERMARK_MESSAGE` (`0b101100111110110010010000011110111011000110011110`). **Caveat: the `imwatermark` dwtDct decode is carrier-fragile on a broad class of real images, NOT just re-encode-fragile, and it is a POSITIVE-ONLY signal.** A clean encode->decode round-trip (no re-encode at all) recovers 48/48 bits on some carriers (random noise, chatgpt-1.png 48/48, firefly-1.png 45/48) but FAILS on many others — verified 2026-06-19 that a *known-embedded* watermark only round-trips 28-39/48 (below the safe `_MATCH_48` = 44 gate, random baseline ~24) on the FLUX fox sample (28), doubao-1.png (39), a 1024² minimalist-flat FLUX image (28), AND a **clean synthetic bright-flat fill with NO watermark at all (28)**. The failure does NOT track texture (firefly lapvar ~11 passes; the flat FLUX lapvar ~56 fails); it correlates with a degenerate decode where the raw bits read **all-ones (48/48 ones)** — which a clean synthetic image reproduces, so **all-ones is a CARRIER ARTIFACT, NOT a watermark signal** (a double-embed test also showed a pre-existing embed does not corrupt a second embed — no interference). Net: trust a `detect_invisible_watermark` hit, but treat a `None`/no-match as **inconclusive** whenever a positive-control embed on the same carrier does not first recover >=44/48. The 44 gate is a deliberate precision choice (lowering it would admit false positives).

  **Root cause and external confirmation (deep-research 2026-06-19, adversarially verified).** This is the SCHEME's ceiling, not our usage — there is no better decoder to adopt. The imwatermark maintainers state verbatim (both the ShieldMnt and Stability-AI READMEs) that the algorithm "cannot guarantee to decode the original watermarks 100% accurately even though we don't apply any attack." Independent measurement (WMAdapter, arXiv:2406.08337 Table 2) puts dwtDct at only **~0.79 bit accuracy on CLEAN images (~38/48 bits — already below our 44 gate)**, collapsing to ~0.50 (chance) under crop/JPEG. Two code-verified + locally-reproduced mechanisms drive the content-dependent failures: (1) the decoder reads each bit as the **highest-magnitude DCT coefficient per block**, so any content coefficient exceeding the encoded target flips the bit; (2) the default embed is in the **YUV chroma channel, which 8-bit-clamps on white/bright pixels** (a +36 chroma delta survives a white-fill round-trip as only +4, ~89% loss) — this is the mechanism behind the bright-flat / minimalist failures and the all-ones degenerate decode. No maintained fork or detector decodes this scheme reliably: the WAVES benchmark (arXiv:2401.08573) relegates DWT-DCT to supplementary appendix G.5 and targets Stable Signature / Tree-Ring / StegaStamp instead; learned encoder/decoder schemes reach ~0.98-0.99 clean but are a DIFFERENT watermark class (not what SDXL/FLUX stamp). `dwtDctSvd` does not help (SDXL embeds `dwtDct`; dwtDctSvd cannot decode it, and its clean accuracy ~0.72 is lower). **Authoritative conclusion: the open DWT-DCT mark cannot be turned from positive-only into a reliable real-world detector; keep it positive-only and rely on C2PA.** (Refuted along the way: that the library is unmaintained, and that it is robust to JPEG but only fails on geometric attacks — both did not survive verification.)

  Consequence for the FLUX hosted-output question (BFL Playground, FLUX.2 [pro] + FLUX.1 [dev], 2026-06-19): all samples carry the signed C2PA manifest (issuer "Black Forest Labs"); the open DWT-DCT decode returned `None`, but every available FLUX carrier (textured fox AND a minimalist-flat generation) failed the positive control (28/48), so the detector is blind on them and **whether BFL hosted output embeds the open pixel watermark is UNRESOLVED** (an earlier note here wrongly asserted it absent — overstated; a later note blamed "high texture" — also wrong, flat carriers fail too). What IS established: C2PA is the reliable FLUX identifier; the `_BITS_48` pattern is correct (round-trips on chatgpt/firefly/random). Resolving the hosted question needs a hosted FLUX carrier that first passes a >=44/48 positive control, which neither a textured nor a flat prompt produced — low priority (the open mark is only a stripped-metadata fallback).
- **C2PA / IPTC (covered by the issuer/marker scan):** OpenAI, Google, Adobe Firefly, Microsoft (Designer + **Bing Image Creator** — collected 2026-05-24; Bing now runs Microsoft's own **MAI-Image** model, signs C2PA as "Microsoft", NOT OpenAI/DALL-E), **Stability AI** (collected from Brand Studio / DreamStudio successor; signs C2PA as "Stability AI Ltd", no SynthID, no imwatermark on its current Stable Image model — issuer added to `C2PA_ISSUERS`), and **Canva** (Magic Media signs C2PA as "Canva" + `trainedAlgorithmicMedia` with a generic `c2pa-rs` claim generator, no SynthID — issuer `b"Canva"` → "Canva (Magic Media)"; verified samples disproved the earlier assumption that Canva downloads always strip C2PA). Still unsampled: Getty, Shutterstock. Midjourney embeds NO C2PA and no invisible watermark (our `mj-*` sample carried only the IPTC tag).

**Samsung Galaxy AI** signs supported edits with C2PA and may carry the
proprietary `genAIType` marker. The registered visible detector covers the
Italian `✦ Contenuti generati dall'AI` bottom-left variant. Removal follows the
same localize-then-fill path as other registered text marks. Other locales and
icon-only variants need separate calibrated silhouettes.

**ASUS Gallery** also signs edited photos as C2PA (`com.asus.gallery`) but with no AI source type — a signer, not an AI marker.

**Black Forest Labs (FLUX)** API output signs C2PA: `claim_generator_info "Black Forest Labs API"` + a `c2pa.ai_generated_content` assertion + `trainedAlgorithmicMedia` (issuer `b"Black Forest Labs"` added to `C2PA_ISSUERS`, platform "Black Forest Labs (FLUX)").

Some applications sign C2PA through an upstream model or infrastructure
provider. For AI claims, exact product mappings in `claim_generator` therefore
take precedence over issuer attribution. Supported mappings include Higgsfield
AI, Topaz Labs Image API, and TikTok Ad Creative Toolbox; an unknown claim
generator still falls back to the certificate issuer.

**ByteDance Volcano Engine (Volcengine)** — the cloud behind Doubao / Jimeng — signs its AI image output with a cert from `certificate_center@volcengine.com` + `trainedAlgorithmicMedia` (issuer `b"volcengine"` → "ByteDance (Volcano Engine)", platform "ByteDance (Doubao / Jimeng / Volcano Engine)"); note this is the C2PA-signed surface, distinct from the XMP/PNG TC260 `AIGC` label Doubao also uses. ByteDance's **international brand (BytePlus / Seedream / Seededit)** signs the same content as **"Byteplus Pte. Ltd."**. The bare `volcengine` needle missed it, so BytePlus output was mis-attributed to "Adobe Firefly" through an incidental "Adobe XMP" toolkit string. Issuer `b"Byteplus"` now maps directly to "BytePlus (ByteDance)". ByteDance's consumer app **Dreamina** (the international Jimeng brand) signs as **"Bytedance Pte. Ltd."** with a `Dreamina/x.y` claim generator but, unlike the Volcano Engine surface, ships **no `trainedAlgorithmicMedia`**. Issuer `b"Dreamina"` maps to "ByteDance (Dreamina)" with **`asserts_ai=True`**. Registering the broader **issuer** `b"Bytedance Pte"` was deliberately avoided because that same entity also signs non-AI CapCut edits; keying on the `Dreamina` generator token is precise.
- **EXIF/XMP/PNG-text generator tag (caught by `exif_generator`):** **Ideogram** writes EXIF `Make="Ideogram AI"` (collected 2026-05-24 — no C2PA, no SynthID, no imwatermark; the Make tag is the only signal). Additional verified generator stamps include **NovelAI** (`Software`, `Source`, and `Title` PNG text chunks), **Reve** (`Software` or XMP `CreatorTool` = `reve.com`), and **Aphrodite AI** (`Make` or `Software` = `Aphrodite AI`).
- **App-export provenance and AIGC JSON:** supported ByteDance-family exports can place a JSON object in EXIF `ImageDescription` or `UserComment`, independently of C2PA or TC260. Exact `product` values for Doubao, Xinghui, and Dreamina are removable product provenance, but do not alone prove that the pixels were generated. Dreamina additionally requires `exportType=generation` for that verdict. A nested Aweme `aigc_type=1` or private ByteDance `aigc_label_type=1` / `2` is an AIGC disclosure; `0` is inconclusive and can occur on a Dreamina generation export. Plain Aweme, retouch, and `lv` exports are preserved. The lower-case private field is deliberately not interpreted as the normative TC260 `Label`, whose values `1` / `2` / `3` mean generated / possibly generated / suspected generated under [GB 45438-2025](https://www.tc260.org.cn/upload/2025-03-15/1742009439794081593.pdf).
- **xAI / Grok — its own EXIF signature scheme, NOT C2PA (DETECTED by `metadata.xai_signature`, built 2026-05-26).**

Grok JPEG downloads (Aurora model) carry **no C2PA, no XMP, no SynthID, no IPTC** — only EXIF `Artist` = a UUID and EXIF `ImageDescription` = `Signature: <base64>` (a crypto signature, unverifiable locally without xAI's public key). This empirically kills the earlier unverified "xAI signs C2PA as xAI" lead — xAI is not even a C2PA member. `exif_generator` misses it (neither field holds an `AI_GENERATOR_TOKENS` token), so a dedicated detector `xai_signature(path)` matches the pair (`ImageDescription ~ ^Signature: [A-Za-z0-9+/=]{64,}` AND UUID `Artist`); wired into `has_ai_metadata`, `get_ai_metadata` (key `xai_signature`), and `identify` (signal `xai_signature`, platform "xAI (Grok / Aurora)").

**Format confirmed stable across n=3 genuine generations:** exactly three EXIF tags (`Artist`, `ExifOffset`, `ImageDescription`), `Signature:` prefix constant, base64 payload 300-1004 chars. Two capture facts: (a) the `Artist` UUID **equals the public image id** in the asset URL (`https://imagine-public.x.ai/imagine-public/images/<uuid>.jpg`), so it is NOT a private per-user secret — only the `Signature` blob is; (b) the Grok web-UI image is a re-encoded **WebP with no signature** — the EXIF survives only in the *original* JPEG (download button or that public tokenless URL), which is why screenshots / re-encodes are metadata-stripped. A real fixture `data/fixtures/provenance/grok-1.jpg` plus **synthetic** JPEG fixtures (fake UUID + fake `Signature:` blob) cover the detector; never add a real Grok image carrying private content (the repo is public).

**Stripped on removal too:** `remove_ai_metadata` calls `_scrub_ai_exif` on
JPEG EXIF, which deletes the xAI Signature and UUID Artist pair plus supported
AI generator values while retaining unrelated camera and editor EXIF. The
shared `xai_signature_pair` helper is the single source of truth for the
pair. On the ISOBMFF path, `blank_ai_exif_tokens` provides the corresponding
in-place scrub for supported EXIF values, TC260 AIGC blocks, and the xAI pair.
- **China TC260 AIGC label (caught by `AIGC_MARKERS` / `metadata.aigc_label`, surfaced by `identify` as the `aigc` signal):** China-served generators embed an XMP `<TC260:AIGC>{"Label":"1","ContentProducer":...}` block — China's mandatory AI-content labeling (TC260 namespace `tc260.org.cn/ns/AIGC`). The label says only "this is AI", but its `ContentProducer` names the signing entity — `001` + `1` + an 18-char Unified Social Credit Code + a 5-digit product suffix, normalized by `metadata.uscc_of`, or for a few generators a bare product name. `KnownMark.tc260_producer_codes` maps the codes settled per vendor by `scripts/vendor_cohort_harvest.py` to registry mark keys, so an AIGC image relaxes the detector of the vendor it actually carries rather than a guessed pair; an unmapped or absent producer falls back to ByteDance's two products. A code identifies a legal entity, not necessarily one brand, so a hosting or aggregating platform that signs for several apps is a recall bet rather than a proof.

**Doubao** (ByteDance) uses it (verified on a public issue sample; `ContentProducer` `001191110102MACQD9K64010000`, no C2PA/SynthID/imwatermark — the XMP block is the only signal; GitHub attachment upload did NOT strip it). The same standard is mandatory for Jimeng/Kling/Qwen/Ernie etc., so the one marker covers the whole China-AIGC-labeled ecosystem. `aigc_label` reads **four image serializations** through a shared `_parse` helper: the HTML-entity-encoded XMP `TC260:AIGC` block in **either RDF form** — the nested element `<TC260:AIGC>{...}</TC260:AIGC>` (Doubao) or the attribute `TC260:AIGC="{...}"` (**PicWish**, `ContentProducer="picwish"`, verified on compatible samples) — via a container-agnostic raw-byte scan (any JSON object accepted), a raw-JSON PNG `AIGC` tEXt chunk (Doubao also writes the label this way, no namespaced marker at all — confirmed on compatible samples, `ContentProducer="doubao"`), a bare raw-JSON `{"AIGC":{...}}` object embedded in **JPEG EXIF (UserComment)** by some China-served generators, brace-matched from the scan head with `json.JSONDecoder().raw_decode` (no namespaced marker, no PNG chunk — confirmed on compatible samples, `ContentProducer="001191440300708461136T1308L"`), **and** a bare `AIGC{...}` blob (the label glued straight to its JSON, no `"AIGC":` key wrapper) embedded in a **JPEG APP segment near the JFIF header** — confirmed on compatible samples. The two raw-JSON forms are scanned in one loop (`'"AIGC"'` then `AIGC{`) that **falls through on a non-TC260 / undecodable hit instead of returning** — a quoted `"AIGC"` can appear later in an XMP packet while the real label is a bare `AIGC{...}` earlier in the file, so an unconditional early return on the quoted form would shadow the bare form (the exact bug behind the 06-10 misses). Native MP4/MOV is a fifth serialization: TC260-PG-20257A stores an `AIGC` key in `moov.udta.meta.keys` and the raw JSON in the matching `ilst` item. The seeking parser reaches a tail `moov` without reading `mdat`; removal replaces the key with `free` and blanks the validated value at the same length so every box size and stream offset stays fixed. All generic forms are gated on at least one TC260 field (`TC260_AIGC_FIELDS`) so a generic `AIGC` key cannot false-positive; the namespaced XMP element is unambiguous and needs no gate. `TC260_AIGC_FIELDS` covers **two schemas**: the producer-side one (`Label` / `ContentProducer` / `ProduceID` / `ContentPropagator` / `PropagateID`, Doubao and most China gens) and the **service-provider** one (`ServiceProvider` / `ServiceUser`, plus generic `Time` / `ContentId` which are NOT gated on) — **Tencent Cloud's** AIGC variant (`ServiceProvider` = `腾讯云`), embedded in **EXIF `ImageDescription`**, verified on compatible samples. In `identify`, `aigc` fires on the parsed label **or** the `AIGC_MARKERS` byte scan (the latter preserves the laundering-tell case where the JSON payload is truncated).

Native MKV/WebM is a sixth serialization. TC260-PG-20257A stores
`TagName=AIGC` and the raw JSON `TagString` in
`Segment.Tags.Tag.SimpleTag`. The bounded EBML reader skips cluster payloads;
the existing ffmpeg stream-copy path removes the tags without transcoding.

The same [TC260 video guide](https://www.tc260.org.cn/portal/article/303/4061772dcf684d8a96f395a4298e9e53)
defines two more native serializations. AVI stores an `AIGC` child in
`LIST/INFO`; FLV stores an AMF0 `AIGC` string under `script.onMetaData`. The
bounded RIFF and FLV readers validate the JSON field set and skip media
payloads. Removal remuxes either container through ffmpeg with stream copy.

- **HuggingFace-hosted job (caught by `metadata.huggingface_job`, surfaced by `identify` as the `hf_job` signal, MEDIUM confidence):** HuggingFace Jobs / Spaces can stamp generated PNGs with an `hf-job-id` tEXt chunk holding the job UUID. It marks the *hosting job*, not a model, so it lifts an Unknown verdict to a tentative AI via `hf_only` but never overrides a hard metadata signal. `_HF_JOB_CAVEAT` states the limit. Removal drops the chunk through the PNG metadata whitelist.
- **No detectable signal on some downloads:** Recraft exports and some hosted
  FLUX surfaces can arrive without a supported local signal. Midjourney samples
  may carry IPTC metadata but no registered C2PA or pixel watermark. The open
  DWT-DCT decoder only applies when the producing pipeline actually ran its
  encoder and the carrier remains decodable.
- **Invisible but NOT locally detectable (proprietary, API/oracle only — same wall as SynthID):** Amazon Titan Image Generator + Nova Canvas (Bedrock `DetectGeneratedContent` API), Kakao (new SynthID image adopter, May 2026), NVIDIA Cosmos (SynthID video). No local detector possible; treat like SynthID.
- **C2PA 2.4 "Durable Content Credentials" (April 2026; verified against the spec) raise the bar for metadata stripping.** 2.4 defines soft bindings (an invisible watermark or a content fingerprint) plus a server-side manifest repository and a new `c2pa.repository-receipt` assertion. Per the spec: "if a C2PA manifest is removed from an asset, but a copy of that manifest remains in a provenance store elsewhere, the manifest and asset may be matched using available soft bindings." So our local `metadata --remove` deletes the *embedded* manifest, but a fingerprint/watermark soft binding can still re-link the image to its manifest in a repository server-side. Stripping the file is becoming necessary-but-not-sufficient against durable provenance. (Our parsers target the stable embedded-manifest format documented in C2PA 2.1 §11; that format is unchanged in 2.4 -- the new pieces are repository/soft-binding infra, not the on-file box layout, so no parser change is implied.) Spec: https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html We now READ the soft-binding `alg` (`C2PA_SOFT_BINDINGS` / `soft_binding_vendors_in`) to name the forensic-watermark vendor, and locally DECODE the one open scheme, Adobe TrustMark (`trustmark_detector`); the rest (Digimarc/Imatag/Steg.AI/...) stay name-only (proprietary decoders).
- **Built in the dated batch:** soft-binding vendor detection, IPTC Photo
  Metadata AI-disclosure fields, C2PA detection and stripping for supported
  ISOBMFF video, the optional Adobe TrustMark decoder, and temporally stabilized
  visible Sora, Veo, Seedance, Dola, Hailuo, and Kling removal. Other visible
  video logos and proprietary audio-watermark detection remain outside the
  package.
  Metadata stripping for supported audio containers is a separate implemented
  path.

**Box detection window — now handled (v0.6.8):** detection no longer relies on a fixed first-MB read. `metadata.scan_head(path, size)` reads the first `size` bytes and, for ISOBMFF, appends the payloads of late provenance boxes found by `isobmff.scan_c2pa_region` (a file-seeking top-level box walker that skips past `mdat` by size without reading it), so a C2PA/AIGC/IPTC manifest placed AFTER a large `mdat` in a streaming/non-faststart MP4 is now caught. Every C2PA/marker byte scan (`has_ai_metadata`, `aigc_label`, `iptc_ai_system`, `synthid_source`, `exif_generator` XMP, `get_ai_metadata` soft-binding, and `identify`) goes through `scan_head`; for PNG it likewise appends the payloads of `tEXt` / `iTXt` / `zTXt` / `eXIf` / `iCCP` chunks that start beyond the window (`_png_late_metadata`, seeking past `IDAT`), which is how a TC260 AIGC label appended after the pixel stream is caught; for WebP it appends the `EXIF` / `XMP ` / `ICCP` / `C2PA` chunks past the window (`_riff_late_metadata`, stepping over the coded image), which is how an IPTC "Made with AI" tag stored after the pixels is caught; and for a file at least `size` bytes long it finally appends the metadata text the decoder reaches but a raw read cannot (`_decoder_visible_text`), which covers a compressed PNG `zTXt` packet no byte scan can spell; for any file that fits inside `size`, it is exactly `f.read(size)`.

Native TC260 MP4/MOV tags do not live in those top-level provenance boxes.
`tc260_aigc_payloads` separately seeks through `moov.udta.meta.keys/ilst`, so
the normative tag is also found when a large `mdat` precedes `moov`.

**Meta-box XMP and EXIF removal are handled in place:** an AI-label XMP packet
stored as a meta-box `mime` item is blanked by
`isobmff.blank_ai_xmp_packets`. Supported EXIF items are handled by
`blank_ai_exif_tokens`. Both paths preserve box sizes and coded media offsets.

For current scope and legal context, see
[scope, safety, and legal notes](legal-and-safety.md). Re-verify legal facts
against primary sources before adding jurisdiction-specific claims.

## Visible AI-generation marks + detection methods (deep-research 2026-07-10, adversarially verified)

**Google Gemini visible "sparkle" -- tier-dependent, and spec-undocumented by Google.** Google primary sources (the Nano Banana Pro blog and the gemini.google image-generation page, both WebFetch-verified) confirm Gemini images carry BOTH the invisible SynthID (on ALL Google-AI media) AND a visible sparkle, but the visible mark is **tier-gated**: applied for FREE and Google AI **Pro** users, and **REMOVED** for Google AI **Ultra** subscribers, inside **Google AI Studio**, and on **API / dev** output. So a Google-C2PA image with NO visible sparkle is expected (Ultra / API), not evidence it is clean -- this reinforces the `identify` "no visible mark != clean" rule. The ONLY official verifier is the SynthID flow (upload to the Gemini app, ask if it is AI-generated), which reads the INVISIBLE mark; there is **no official visible-sparkle detector**, and Google publishes **no** glyph geometry / size / opacity / color / locale / placement spec. So our capture-based sparkle template is the only source of truth and cannot be validated against a vendor spec -- keep reverse-engineering from real captures (do not expect a published spec).

**Google Veo video marks use two incompatible visible designs.** Public raw
clips verify the current four-point diamond and the legacy bottom-right `Veo`
text. The independent
[VeoWatermarkRemover](https://github.com/allenk/VeoWatermarkRemover) project
reports the same current-versus-legacy split, multiple output layouts, and the
need for cross-frame position agreement. Our implementation copies no logo
pixels or alpha maps from that project: it uses two synthetic silhouettes,
known-layout searches plus a strong relocated-diamond fallback, and a separate
temporal arbiter calibrated against raw watermarked clips and clean API exports.

**ByteDance video surfaces use distinct visible labels.** Public Seedance
showcase clips contain a fixed rounded box with `AI`, while the Dola sample in
an inherited research note records that it uses
fixed `Dola AI` text. The independent
[Seedance remover](https://github.com/SamurAIGPT/seedance-2.0-watermark-remover)
estimates a static corner from a temporal mean frame and edge density. Our
implementation instead matches provider-specific synthetic silhouettes on
every frame, then requires an anchored temporal run. This extra anchor check
was necessary because a moving clean scene detail could retain enough adjacent
overlap to pass a recurrence-only gate.

**Hailuo and Kling use larger fixed composite labels.** Verified Hailuo exports
carry a lower-edge waveform, `MINIMAX`, separator, Hailuo ring, and
`hailuo AI` text. Verified Kling exports carry a bottom-right swirl,
`KLING AI`, a changing version suffix, and sometimes `PRO`. The detectors use
only synthetic primitives and fonts. Hailuo expands the matched core to cover
the complete composite. Kling combines a version-independent text core with a
synthetic ring rescue, then requires the recurring candidate to reach the
expected frame edge and contain enough bright low-saturation pixels. Those
extra gates were added after clean Luma and PixVerse scene details passed shape
and temporal recurrence alone. The generic
[WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI) project
instead uses Florence-2 to identify arbitrary watermarks before LaMa
inpainting. That is broader, but it carries a much heavier model and a less
auditable detection boundary than the provider-specific synthetic path here.

**The faint-visible-mark precision/recall wall is fundamental, not a heuristic artifact.** The visible-watermark-detection literature has moved to LEARNED segmentation / object-detection (WDNet WACV'21 arXiv:2012.07616; SLBR ACM MM'21, open code+weights; the PRCV'18 large-scale detector; Su et al. survey 2025), but three verified findings bound what a learned detector actually buys: (1) a claim that a confidence threshold "cleanly separates" true from false matches even with a learned CNN front-end was **REFUTED** in verification (arXiv:1705.08593) -- the precision/recall wall persists even with learned features. (2) Learned detectors need a LARGE, pattern-diverse labeled dataset trained on synthetic composites (PRCV'18: 60k images / 80 watermark classes; CLWD: 60k / 160 marks), and off-distribution degradation is a documented real axis (models trained on limited-pattern LVW transfer worse; diversity of training patterns drives generalization). (3) Inference is cheap (WDNet ~8 ms at 256x256) -- the cost is the data pipeline, not runtime. Net: a learned detector shifts the frontier but does NOT remove the wall; for a SINGLE mark the cheapest next step is a small patch classifier (real-sparkle vs false-positive) on top of the existing NCC localizer, not a full segmentation model. SLBR is a ready baseline. The current NCC + false-positive gate (core-ring brightness margin + gradient-NCC crispness + white-core saturation) is a sound operating point, and the residual miss is the information-theoretic wall the literature confirms.

**Visible-mark landscape beyond the registry.** Meta stamps a visible "Imagined with AI" mark (bottom-LEFT, a small symbol) on its OWN Meta AI / "Imagine" output; for third-party images it relies on C2PA / IPTC, not a visible mark. Samsung Galaxy AI additionally uses a **four-star icon** variant in a corner alongside the localized text wordmark `samsung_engine` calibrates (only the Italian text variant is covered) -- the icon is a distinct, uncovered variant. Every source agrees visible + metadata marks are trivially removable (crop / screenshot, ~2 s), which is the tool's premise.

**Regulatory driver -- China GB 45438-2025 is the strongest VISIBLE-mark mandate.** The CAC / TC260 "Measures for Labeling AI-Generated Synthesized Content" (issued March 2025, **effective 2025-09-01**, technical standard **GB 45438-2025**, building on the TC260 Aug-2023 practice guide) MANDATE a **visible** label for AI images -- a visible textual mark whose height must be **>= 5% of the image's shortest side** -- plus the metadata (implicit) label. Several such CJK text marks are now registered; see [supported signals](supported-signals.md) for the current list. By contrast EU AI Act Article 50 mandates only the MACHINE-READABLE mark (enforceable 2026-08-02, grace to 2026-12-02); a visible label is proposed and modality-specific (visible for images) but is NOT a hard "fixed icon" mandate -- a claim that Art 50 requires a clearly-visible fixed icon for images was refuted in verification. Primary-source dates verified against the article/standard text, not search summaries.

## Uncovered visible marks: implementation specs (deep-research 2026-07-18)

Compatibility testing showed that TC260-labelled images can still produce no visible-mark
detection. The main causes were a fixed Doubao localization defect and genuinely
uncovered vendors. Verification status is labelled per claim; treat (b)/(c) as leads,
not ground truth.

**GB 45438-2025 clause 5.2, the binding constraint for every Chinese mark (VERIFIED (a) -- full standard text extracted from the TC260-hosted PDF).** Verbatim requirements for an image's explicit label:
- 应采用文字提示 (must be a TEXT prompt);
- must contain BOTH an AI element (`人工智能` or `AI`) AND a generation element (`生成` and/or `合成`);
- 应位于图片的边或角 -- **edge OR corner**, so bottom-right is NOT mandated (Annex C.2's bottom-right figure is a non-normative example);
- 字型应清晰可辨 (legible typeface -- no family named);
- **文字高度不应低于画面最短边长度的 5%** -- glyph HEIGHT >= 5% of the frame's SHORTEST side, with a note defining the shortest side for non-rectangular images.

Two consequences we can exploit: (1) the 5% floor is a **scale prior** -- a compliant CN mark's glyphs are large (>= 51 px on a 1024² image), so a CN silhouette ladder can be anchored at ~5-10% of the short side instead of swept broadly, which should cut false fires; (2) every compliant string shares the tail `AI生成` / `AI合成`, so a shared suffix silhouette plus a per-vendor prefix may beat five independent templates. NOT established: whether 文字高度 means cap height, em box, or rendered bounding box (a ~1.3x spread in template scale). Sources: `https://www.tc260.org.cn/upload/2025-03-15/1742009439794081593.pdf`, parent CAC measure `https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm` (the CAC text itself specifies no size or corner).

**Alibaba Qwen -- two surfaces that differ (API tier VERIFIED (a)).** Model Studio docs state verbatim that the API adds a `Qwen-Image` watermark 在图像右下角 and 默认值为 false -- so API output is **unwatermarked by default**, and when enabled the mark is a LATIN wordmark, not CJK, and not GB-compliant in wording. The consumer app's `千问AI生成` (bottom-right) is (b) secondary only -- no Alibaba primary page states it. So Qwen needs TWO templates, and its absence is never evidence of a clean image. Source: `https://help.aliyun.com/zh/model-studio/qwen-image-api`.

**星绘 is ByteDance (VERIFIED (a): Baidu Baike + App Store listing, now branded 豆包旗下, team folded into Doubao April 2025).** So `星绘AI生成` is very likely the Doubao house style -- same typeface, same corner, possibly the same top-left `AI生成` pill. Starting from the Doubao `TextMarkConfig` and swapping the two lead glyphs is the cheap path. String/position themselves are (c) inferred.

**Baidu: RESOLVED 2026-07-22, registered (`baidu_engine.py`).** The mark is a white bold "百度" text run + a separate white rounded tag with dark "AI生成", bottom-right -- settled by the TC260 USCC cohort harvest (16 frames, USCC 91110000802100433B), not by web research. Detection keys on the text run only; details in `docs/module-internals.md`.

**Tencent Yuanbao: RESOLVED 2026-07-25, registered (`yuanbao_engine.py`).** The standard mark is a compact two-line italic `元宝` over `AI生成` block at bottom-right. It switches between light and dark strokes with the scene, so detection uses polarity-independent local contrast rather than a white top-hat. The separate one-line overlay variant remains evidence-limited to one example.

**Meta `Imagined with AI` (string VERIFIED (a) from Meta's own newsroom; POSITION NOT VERIFIED).** Sources conflict on placement. Do not encode a corner without a verified sample. `identify` reads the supported IPTC disclosure; it does not decode Meta's proprietary invisible watermark. Source: `https://about.fb.com/news/2024/02/labeling-ai-generated-images-on-facebook-instagram-and-threads/`.

**Samsung English/other locales: still not established.** Samsung's own support page says only that "A Galaxy AI watermark will appear on AI-generated images" -- no string, no corner. Every community thread carrying the exact English string returned HTTP 403 to WebFetch, so the search paraphrase (bottom-left) is deliberately NOT recorded as fact. Feature-tier detail (b): the mark is applied by Generative Edit / sketch-to-image but reportedly NOT by Object Eraser, so Samsung absence is feature-dependent. The four-star icon variant: nothing found.

**The one document that would settle ByteDance placement is BLOCKED.** Douyin's 《抖音关于人工智能生成内容标识的水印与元数据规范》 aims to give AI tools a unified watermark style and position, which would cover Doubao / Jimeng / 星绘 at once. Both mirrors return HTTP 403 to WebFetch; a secondary report (b, unconfirmed) says the watermark is `AI生成` + tool name + company name placed **top-left** -- which would explain the Jimeng pill's top-left position but contradicts the GB annex's bottom-right example. Worth one retry through Chrome MCP with a real browser session.

**No vendor publishes typeface, color, opacity, plate, or margin for ANY of these marks.** The only font-adjacent requirement anywhere is GB's "legible typeface". So each synthetic silhouette's font must be calibrated against corpus positives exactly as the Jimeng pill was; candidate CJK families by platform convention (inference): HarmonyOS Sans / Source Han Sans / Noto Sans CJK SC for Android-origin apps, PingFang SC for iOS-origin.
