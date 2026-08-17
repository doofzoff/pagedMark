# Deep research: SynthID-safe face-identity recovery for SDXL (2026-06-08)

> Research archive. This dated follow-up records evidence available during the
> study. It is not the current command reference. The `instantid` and `photomaker`
> extras it discusses, the `--restore-faces` flag, and their modules were all removed
> from the library afterwards.

**Stats:** {"angles": 6, "sourcesFetched": 28, "claimsExtracted": 104, "claimsVerified": 25, "confirmed": 19, "killed": 6, "afterSynthesis": 6, "urlDupes": 1, "budgetDropped": 7, "agentCalls": 111}

## Summary

For SDXL-based SynthID-removal pipelines, the entire ArcFace-class identity-adapter ecosystem (PhotoMaker-V2, InstantID, PuLID, IP-Adapter FaceID, Arc2Face) is blocked from commercial use by a single chokepoint: InsightFace's pretrained model packs (buffalo_l, antelopev2, buffalo_s, buffalo_m) are explicitly non-commercial research-only, regardless of the adapter weights' Apache-2.0/MIT license. The InstantX maintainers themselves acknowledged this on HuggingFace ("cannot be Apache 2.0 if it is using Insight Face") and stated intent to retrain on commercially-licensed embedders — work that as of the verified sources has not shipped. The only verified commercial-safe SDXL adapter surfaced is MS-Diffusion (ICLR 2025), which uses CLIP-ViT-bigG-14 instead of ArcFace and runs on SDXL-base-1.0, but it is an IP-Adapter-class subject-similarity method, not a face-identity method — CLIP image embeddings are empirically weaker for face identity than ArcFace (~81% vs ~88% face-ID accuracy), so it solves the license problem but likely not the identity-fidelity problem. Arc2Face is SD1.5-only and so not a drop-in regardless of license; StyleGAN2-based methods (e4e) terminate in a GAN generator with no SDXL wiring. Net: no fully commercial-safe SynthID-robust SDXL identity-preservation stack with ArcFace-grade fidelity exists today — the space is genuinely blocked by InsightFace's grip on commercial ArcFace embeddings, and AdaFace as a permissive ArcFace alternative was REFUTED in verification.

## Findings

### 1. The InsightFace pretrained model packs (buffalo_l, antelopev2, buffalo_s, buffalo_m) are non-commercial research-only and this restriction propagates at runtime to any adapter that calls FaceAnalysis(), regardless of the adapter's own license tag.

**Confidence:** high  
**Vote:** 3-0 unanimous across 4 supporting claims


InsightFace upstream: code is MIT but 'The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.' Licensing page enumerates buffalo_l, antelopev2, buffalo_s, buffalo_m as needing separate commercial licensing. InstantX maintainer on HF: 'cannot be Apache 2.0 if it is using Insight Face... we plan to train on other face encoders that support commercial license.' Mechanical propagation: the runtime FaceAnalysis() call pulls these packs.

- https://github.com/deepinsight/insightface
- https://www.insightface.ai/solutions/face-recognition-licensing
- https://huggingface.co/InstantX/InstantID/discussions/2

### 2. InstantID is Apache-2.0 on the adapter weights and architecturally a clean plugin to SDXL-base-1.0 (no UNet training, semantic IdentityNet conditioning + landmark weak-spatial, no pixel copying) — but at runtime it instantiates FaceAnalysis(name='antelopev2'), inheriting the InsightFace non-commercial restriction.

**Confidence:** high  
**Vote:** 3-0 across 5 claims


HF model card: 'License: apache-2.0' AND 'For face encoder, you need to manutally download via this URL to models/antelopev2' with 'from insightface.app import FaceAnalysis' in usage. Diffusers community pipeline shows literal `app = FaceAnalysis(name='antelopev2', ...)` then `face_emb = face_info['embedding']`. Paper: 'IdentityNet by imposing strong semantic and weak spatial conditions, integrating facial and landmark images with textual prompts' + 'seamlessly integrates with SD1.5 and SDXL'. So mechanism is semantic (good for SynthID-no-pixel-leak) but license is blocked.

- https://huggingface.co/InstantX/InstantID
- https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_stable_diffusion_xl_instantid.py
- https://arxiv.org/pdf/2401.07519
- https://instantid.github.io/

### 3. Arc2Face is SD1.5-only (not SDXL) AND requires InsightFace antelopev2 with arcface.onnx replacing glintr100.onnx — so it is doubly blocked: not a drop-in for the SDXL pipeline AND non-commercial via the model pack.

**Confidence:** high  
**Vote:** 3-0 across 3 claims


Arc2Face README verbatim: 'Arc2Face is built upon SD1.5' with stable-diffusion-v1-5 checkpoint as base. Same README: 'manually download the antelopev2 package and place the checkpoints under models/antelopev2', 'Download arcface.onnx from HuggingFace', 'delete glintr100.onnx (the default backbone from insightface)'. Code is MIT but the runtime antelopev2 dependency carries the non-commercial restriction. No SDXL adaptation exists in the official repo.

- https://github.com/foivospar/Arc2Face

### 4. MS-Diffusion (ICLR 2025) is the one verified candidate that combines SDXL-base-1.0 as its foundation with a non-ArcFace image encoder (CLIP-ViT-bigG-14), so it avoids the InsightFace non-commercial blocker — but it is an IP-Adapter-class multi-subject personalization method, NOT a face-identity-recognition method, so license safety does not equal face-identity fidelity.

**Confidence:** medium  
**Vote:** 3-0 on the narrow factual claims (SDXL base + CLIP-G encoder)


GitHub README explicitly instructs 'Download the pretrained base models from SDXL-base-1.0 and CLIP-G' (CLIP-ViT-bigG-14-laion2B-39B-b160k). Neither README nor arXiv 2406.07209 mention ArcFace/InsightFace/antelopev2/buffalo_l. Architecturally descended from IP-Adapter (CLIP-image-embedding family), not from FaceID/InstantID/PhotoMaker-V2. Verifier caveat (high confidence on the license-narrow claim, medium on suitability): CLIP-image face-ID accuracy ~80.95% vs specialized face recognition ~87.61% — license-safe but probably not identity-grade for portraits. Confidence is medium because suitability for portrait and group-photo inputs has not been validated empirically.

- https://proceedings.iclr.cc/paper_files/paper/2025/file/ed4df1609bf7d8602435341c9ce2ab5f-Paper-Conference.pdf
- https://github.com/MS-Diffusion/MS-Diffusion

### 5. ID-Aligner and the StyleGAN2/e4e-based identity method (arXiv 2510.25084) do not solve the problem: the former does not disclose embedder/license in the primary source; the latter terminates identity in a StyleGAN2 generator with no SDXL adapter, so it cannot be wired into the SDXL+canny ControlNet pipeline.

**Confidence:** high  
**Vote:** 3-0 across 3 claims


ID-Aligner paper: no license terms, no weight-release status, no specific face embedder named in the primary source — commercial-safety unresolved. arXiv 2510.25084: 'facial identity features... mapped into the W+ latent space of StyleGAN2 using the e4e encoder' — identity pixels come from StyleGAN2's generator, not from any SDXL-compatible adapter, so architecturally incompatible with our SDXL+canny ControlNet pipeline. Abstract also does not name the face embedder or disclose weights license.

- https://arxiv.org/pdf/2404.15449
- https://arxiv.org/pdf/2510.25084

### 6. The honest verdict: no fully commercial-safe SynthID-robust ArcFace-grade face-identity stack for SDXL exists today; the space is blocked by InsightFace's grip on ArcFace-class pretrained packs, and the verified permissive alternative (AdaFace as a drop-in replacement) was REFUTED in this verification round.

**Confidence:** high  
**Vote:** 3-0 on the chokepoint claim, 0-3 against the AdaFace escape hatch


Every ArcFace-grade SDXL adapter audited (PhotoMaker-V2, InstantID, PuLID, IP-Adapter FaceID, Arc2Face) instantiates an InsightFace FaceAnalysis pack at runtime. The maintainer-acknowledged commercial-retrain path (InstantX) has not shipped. AdaFace as a permissive ArcFace alternative was refuted 0-3 by the adversarial verifier (MIT-licensed claim and drop-in-replacement claim both failed). Outcome: commercial-safe option for SDXL today is CLIP-image-embedding-based (MS-Diffusion class), which is weaker for face identity than ArcFace. For non-commercial / research-only deployments, InstantID on SDXL is the strongest semantic-only (no-pixel-leak) candidate.

- https://github.com/deepinsight/insightface
- https://huggingface.co/InstantX/InstantID/discussions/2
- https://github.com/mk-minchul/AdaFace

## Caveats

Six claims were refuted in adversarial verification, two of them load-bearing: AdaFace as a permissive ArcFace drop-in (both the MIT license and the drop-in characterization failed 0-3) — so the most attractive escape hatch from the InsightFace chokepoint did not survive verification. PuLID's superiority claim and ID-Aligner's embedder claim also failed, leaving those methods uncharacterized at the mechanism level. None of the verified claims directly answered the diffusers-0.38 compat question — InstantID's compat with our shipped diffusers version is unverified by primary source. MS-Diffusion's identity-fidelity for portraits is not empirically validated for the SynthID-removal use case; it is verified as a CLIP-G/SDXL adapter, not as a face-identity method. No 2025-2026 candidate other than MS-Diffusion was both surfaced AND verified as license-safe and SDXL-compatible — the verification round did not produce a confirmed CCSR-V2/ConsistencyID/OmniGen-V2/ConsistentID/MagicID candidate. The multi-face scenario (group photos) was not addressed by any verified claim — MS-Diffusion is the only multi-subject candidate but its face-identity strength is unmeasured. Time-sensitivity: InstantX's stated intent to retrain on commercial embedders ("We agree, we plan to train on other face encoders that support commercial license") is dated and unfulfilled per the verified sources; this could change.

## Open questions

- Does MS-Diffusion (or any CLIP-image-embedding SDXL adapter) achieve usable face-identity fidelity on portraits and group photos, or is the ArcFace gap (~7 pp face-ID accuracy) visually disqualifying — and can a face-specific CLIP fine-tune close it?
- Has InstantX (or any community fork) actually shipped an InstantID variant retrained on a commercially-licensed face embedder since the maintainer's 2024 commitment, and if so what is its identity-fidelity vs the antelopev2 original?
- What is the exact diffusers-0.38 compat status of InstantID, MS-Diffusion, and PuLID-FLUX inference scripts — does any need a fork the way PhotoMaker-V1 did, and if so what specifically breaks?
- Is there a single-pipeline multi-subject identity-preservation method (mask-guided regional ID-adapters, multi-subject InstantID, MS-Diffusion multi-subject mode) that handles group photos without the per-face crop+composite patchwork that PhotoMaker-V2 produced?

## Refuted claims

- **AdaFace code is released under the MIT license, making it permissively licensed for commercial use (in contrast to InsightFace's research-only model packs).** — vote 0-3 — source: https://github.com/mk-minchul/AdaFace
- **AdaFace produces 512-dim face recognition embeddings comparable to ArcFace, positioning it as a drop-in alternative for ID-conditioning adapters that currently depend on InsightFace ArcFace.** — vote 0-3 — source: https://github.com/mk-minchul/AdaFace
- **The paper claims PuLID achieves superior performance in both ID fidelity and editability compared to prior ID-customization methods.** — vote 0-3 — source: https://arxiv.org/pdf/2404.16022
- **Identity consistency in ID-Aligner is enforced by reward feedback from face detection and recognition models, implying dependency on an external face-recognition embedder (typically ArcFace/InsightFace-class) rather than a CLIP-only path.** — vote 1-2 — source: https://arxiv.org/pdf/2404.15449
- **The pipeline decouples face detection from inference by accepting a pre-computed embedding via image_embeds, so any ArcFace-class embedder could be swapped in if a commercially-licensed equivalent existed.** — vote 1-2 — source: https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_stable_diffusion_xl_instantid.py
- **The model card does not explicitly prohibit commercial use, but the license of the required InsightFace antelopev2 embedder is not specified on this page.** — vote 1-2 — source: https://huggingface.co/InstantX/InstantID

## Sources

- [source](https://huggingface.co/InstantX/InstantID/discussions/2)
- [source](https://www.insightface.ai/solutions/face-recognition-licensing)
- [source](https://github.com/mk-minchul/AdaFace)
- [source](https://github.com/foivospar/Arc2Face)
- [source](https://apatero.com/blog/instantid-vs-pulid-vs-faceid-ultimate-face-swap-comparison-2025)
- [source](https://arxiv.org/pdf/2404.16022)
- [source](https://instantid.github.io/)
- [source](https://arxiv.org/pdf/2404.15449)
- [source](https://arxiv.org/pdf/2511.11989)
- [source](https://arxiv.org/pdf/2510.25084)
- [source](https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_stable_diffusion_xl_instantid.py)
- [source](https://huggingface.co/InstantX/InstantID)
- [source](https://github.com/huggingface/diffusers/issues/9158)
- [source](https://github.com/huggingface/diffusers/issues/5904)
- [source](https://github.com/Mikubill/sd-webui-controlnet/discussions/2589)
- [source](https://arxiv.org/pdf/2401.07519)
- [source](https://proceedings.iclr.cc/paper_files/paper/2025/file/ed4df1609bf7d8602435341c9ce2ab5f-Paper-Conference.pdf)
- [source](https://github.com/huggingface/diffusers/issues/8626)
- [source](https://arxiv.org/pdf/2404.04243)
- [source](https://huggingface.co/OmniGen2/OmniGen2)
- [source](https://github.com/VectorSpaceLab/OmniGen)
- [source](https://github.com/ToTheBeginning/PuLID)
- [source](https://arc2face.github.io/)
- [source](https://github.com/mk-minchul/AdaFace/blob/master/LICENSE)
- [source](https://github.com/IrvingMeng/MagFace/blob/main/LICENSE)
- [source](https://github.com/askerlee/AdaFace-dev)
- [source](https://openreview.net/forum?id=Hc2ZwCYgmB)
- [source](https://github.com/tencent-ailab/IP-Adapter/wiki/IP%E2%80%90Adapter%E2%80%90Face)

## Empirical follow-up (2026-06-08, end of session)

After the research synthesis above, InstantID was integrated end-to-end and cert-swept
on Modal A100 in two phases:

1. **Phase 1: InstantID txt2img per-face crop + composite.** Per-face InstantID
   txt2img with the upstream `pipeline_stable_diffusion_xl_instantid`, ArcFace
   embedding from the original face, landmark stick figure. Three composite
   iterations:
   - v1 (rectangular Gaussian alpha on the 2x square_box around each face):
     visible patchwork on group photos, generated 1024 backgrounds clashing.
   - v2 (tight crop on YuNet-detected face in the generated 1024 + elliptical
     alpha 0.45*bw x 0.55*bh + soft feather): ellipse axis exceeded bbox
     vertically, clipped forehead/chin on single portrait, group still had
     visible elliptical seams + cool-vs-warm tone clash with scene.
   - v3 (tighter ellipse 0.32*bw x 0.42*bh + per-channel mean color match to
     local cleaned canvas + softer feather): patchwork visually softened; faces
     still read as studio portraits inserted into the scene, not as people
     shot in the scene. Single portrait identity drifted (the single-portrait
     fixture regenerated as a generic "round Asian male" vs the original's thin
     face).
2. **Phase 2: InstantID img2img on cleaned crop.** Switched to the upstream
   `pipeline_stable_diffusion_xl_instantid_img2img` (downloaded at first use
   from raw.githubusercontent.com; requires `trust_remote_code=True`). Same
   ArcFace + landmark conditioning but the SDXL diffusion source is the
   CLEANED face crop, so the diffusion sees scene lighting / shoulders /
   shadow direction directly. Multi-face composition jumped substantially:
   faces sit in the bar scene with matching warm tone, no more elliptical
   seams. Single-portrait identity at the default (`strength=0.55`,
   `ip_adapter_scale=0.8`, `controlnet_conditioning_scale=0.8`) was "similar
   person, not exactly the original"; raising to `strength=0.7`,
   `ip_adapter_scale=1.0`, `controlnet_scale=1.0` brought identity closer to
   original but introduced more "SDXL gloss / clean skin" aesthetic.

**Net finding for a commercial deployment (load-bearing).** The fundamental issue is structural:
ArcFace encodes "this person's general look" (ethnicity, gender, basic facial
geometry) at 512 dimensions; SDXL decodes that embedding into pixels with the
inherent SDXL aesthetic (smooth skin, symmetric pores, AI-photoreal look).
Stronger identity push (higher strength / IP-Adapter scale) makes the face
CLOSER to the embedded identity but MORE AI-looking; weaker push leaves
identity to drift but face looks less AI-generated. There is no parameter
setting that simultaneously recovers original identity AND looks less AI than
the cleaned image, because the cleaned image is itself a controlnet-light
denoise of the original (closer to original pixels) while a restore pass is a
full SDXL regeneration (further from original pixels).

**Operational conclusion.** Do not ship `--restore-faces` in any monetized
deployment. The cleaned image from the main controlnet 0.20 pass is the
LEAST-AI state we can reach without re-introducing SynthID; every restore
method tested (GFPGAN-on-cleaned, PhotoMaker-V2, InstantID txt2img,
InstantID img2img-on-cleaned at three parameter sweeps) trades original-look
for embedding-driven regeneration and makes the face read as "AI-generated"
rather than "the original person". The `instantid` and `photomaker` extras
stay in the library as opt-in for research / personal use where users
explicitly want identity regeneration; the CLI flag and module docstrings
state the trade-off at every entry point.
