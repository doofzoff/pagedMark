"""ControlNet-as-removal-pipeline prototype sweep (issue #35 / Jacob).

Research prototype, NOT a shipped pipeline. It tests whether a full-image
SDXL-native ControlNet-conditioned img2img can REPLACE plain SDXL img2img as the
watermark remover: a single structure-guided regeneration that scrubs an invisible
robust watermark (SynthID) everywhere while keeping fine detail and small/CJK text
legible. See docs/controlnet-removal-pipeline-research.md for the full rationale.

The make-or-break tension (from the watermark-removal-attack literature): the
denoise strength high enough to scrub the watermark deforms text, while the
conditioning strong enough to keep text may spare the watermark. There is no local
SynthID detector, so this script CANNOT decide removal on its own -- it produces
one output per (control, strength, conditioning-scale) cell plus an index, and YOU
verify each output by hand in the Gemini app ("Verify with SynthID") and judge text
legibility visually. Fill the verdict columns in the emitted index, then read off
the Pareto cell (oracle clean AND text legible).

Pipeline: stabilityai/stable-diffusion-xl-base-1.0 +
  - canny: xinsir/controlnet-canny-sdxl-1.0  (control = cv2.Canny(gray, 100, 200))
  - tile:  xinsir/controlnet-tile-sdxl-1.0   (control = the resized original, no preproc)
StableDiffusionXLControlNetImg2ImgPipeline (image=init, control_image=control).

Needs the gpu extra (torch + diffusers) and cv2. Runs locally on 32 GB MPS in
fp32 (MPS fp16 decodes to all-black NaN -- issue #29 -- so fp32 is the default on
mps/cpu, fp16 only on cuda/xpu); a dedicated GPU is not required for 1024 px. Run:

    uv run python scripts/controlnet_sweep.py path/to/watermarked.png -o sweep_out
    uv run python scripts/controlnet_sweep.py img.png --control canny tile \\
        --strength 0.3 0.5 0.7 1.0 --scale 0.6 1.0 --size 1024
"""

from __future__ import annotations

# torch/diffusers/cv2 ship no usable types; relax the unknown-type + private-import
# rules for this boundary script (mirrors scripts/visible_alpha_solve.py and the
# cv2/torch engine modules). Pure-logic helpers here stay correct regardless.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportPrivateImportUsage=false
import argparse
import contextlib
import csv
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _plain_console import Console, Table

if TYPE_CHECKING:
    from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNETS = {
    "canny": "xinsir/controlnet-canny-sdxl-1.0",
    "tile": "xinsir/controlnet-tile-sdxl-1.0",
}

# A neutral quality prompt: the goal is faithful regeneration, not creative edits.
PROMPT = "best quality, high quality, sharp, detailed, photographic"
NEGATIVE_PROMPT = "blurry, lowres, deformed, distorted text, garbled text, watermark, jpeg artifacts"


def pick_device(requested: str) -> str:
    """Resolve the inference device without the CUDA-reinstaller side effect.

    Deliberately does NOT call the package ``get_device`` (which can trigger a
    torch-CUDA reinstall+restart). A research script should never do that.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, requested: str) -> Any:
    """fp16 only on cuda/xpu; fp32 on cpu AND mps, unless overridden.

    MPS fp16 produces all-black NaN output here (the SDXL UNet/VAE overflows on
    the Metal backend -- issue #29; even the fp16-fix VAE does not save it), so the
    production pipeline runs fp32 on MPS and so do we. fp32 SDXL + an SDXL ControlNet
    at 1024 fits 32 GB unified memory with vae-tiling + attention-slicing.
    """
    import torch

    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    return torch.float16 if device in {"cuda", "xpu"} else torch.float32


def fit_size(image: Image.Image, long_side: int) -> Image.Image:
    """Resize so the long side is ``long_side``, each dim a multiple of 8 (SDXL)."""
    from PIL import Image as PILImage

    w, h = image.size
    scale = long_side / max(w, h)
    nw = max(8, round(w * scale) // 8 * 8)
    nh = max(8, round(h * scale) // 8 * 8)
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), PILImage.Resampling.LANCZOS)


def make_control_image(init: Image.Image, control: str) -> Image.Image:
    """Build the ControlNet conditioning image for the given control type.

    canny: cv2.Canny(gray, 100, 200) -> 3-channel edge map (xinsir canny recipe).
    tile:  the init image itself, no preprocessing (xinsir tile recipe).
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    if control == "tile":
        return init
    rgb = np.array(init.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return PILImage.fromarray(edges_rgb)


def psnr(a: Image.Image, b: Image.Image) -> float:
    """Coarse global fidelity proxy vs the original (NOT a text or watermark metric)."""
    import numpy as np

    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        return 99.0
    return float(10.0 * np.log10((255.0**2) / mse))


def load_pipeline(control: str, device: str, dtype: Any) -> Any:
    """Load SDXL base + the chosen SDXL ControlNet as an img2img pipeline."""
    import torch
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionXLControlNetImg2ImgPipeline,
    )

    console.print(f"Loading {CONTROLNETS[control]} ({control}) ...")
    controlnet = ControlNetModel.from_pretrained(CONTROLNETS[control], torch_dtype=dtype)
    load_kwargs: dict[str, Any] = {"controlnet": controlnet, "torch_dtype": dtype}
    if dtype == torch.float16:
        # The stock SDXL VAE decodes to NaN/black in fp16; the fp16-fix VAE is the
        # same swap the production pipeline uses (_SDXL_FP16_VAE_ID).
        load_kwargs["vae"] = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(BASE_MODEL, **load_kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device != "cpu":
        # Keep the 1024 px + extra-ControlNet peak inside 32 GB unified memory.
        with contextlib.suppress(Exception):
            pipe.enable_vae_tiling()
        with contextlib.suppress(Exception):
            pipe.enable_attention_slicing()
    return pipe


def run_cell(
    pipe: Any,
    init: Image.Image,
    control_image: Image.Image,
    strength: float,
    scale: float,
    steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    """Run one ControlNet img2img cell and return the regenerated image.

    The generator is created on CPU intentionally: a CPU generator is portable
    across the mps/cuda/cpu backends (diffusers rejects a device-mismatched one),
    matching the production runner's fallback behavior.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=init,
        control_image=control_image,
        controlnet_conditioning_scale=float(scale),
        strength=float(strength),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    return result.images[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ControlNet-as-removal-pipeline prototype sweep.")
    p.add_argument("image", type=Path, help="Watermarked input image.")
    p.add_argument("-o", "--out", type=Path, default=Path("controlnet_sweep_out"), help="Output directory.")
    p.add_argument("--control", nargs="+", choices=list(CONTROLNETS), default=["canny", "tile"])
    p.add_argument("--strength", nargs="+", type=float, default=[0.3, 0.5, 0.7, 1.0])
    p.add_argument("--scale", nargs="+", type=float, default=[0.6, 1.0], help="controlnet_conditioning_scale values.")
    p.add_argument("--size", type=int, default=1024, help="Long-side resolution (multiple of 8).")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        log.error("Input image not found: %s", args.image)
        return 1

    try:
        from PIL import Image as PILImage
    except ImportError:
        log.error("Pillow is required. Install the gpu extra: uv sync --extra gpu --extra dev")
        return 1
    if importlib.util.find_spec("diffusers") is None or importlib.util.find_spec("torch") is None:
        log.error("diffusers/torch are required. Install: uv sync --extra gpu --extra dev")
        return 1

    device = pick_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    console.print(f"Device: {device} | dtype: {str(dtype).split('.')[-1]}")

    init_full = PILImage.open(args.image).convert("RGB")
    init = fit_size(init_full, args.size)
    console.print(f"Input: {args.image.name} {init_full.size[0]}x{init_full.size[1]} -> {init.size[0]}x{init.size[1]}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    init_path = args.out / f"{stem}__INPUT.png"
    init.save(init_path)

    rows: list[dict[str, Any]] = []
    table = Table(title="ControlNet sweep")
    for col in ("control", "strength", "scale", "psnr_vs_input", "file"):
        table.add_column(col)

    # Group by control so SDXL + the ControlNet load once per control type.
    for control in args.control:
        pipe = load_pipeline(control, device, dtype)
        control_image = make_control_image(init, control)
        if control == "canny":
            control_image.save(args.out / f"{stem}__canny_edges.png")
        for strength in args.strength:
            for scale in args.scale:
                tag = f"{control}_s{strength:g}_c{scale:g}"
                console.print(f"Running {tag} ...")
                try:
                    out = run_cell(
                        pipe,
                        init,
                        control_image,
                        strength,
                        scale,
                        args.steps,
                        args.guidance,
                        args.seed,
                    )
                except Exception as exc:
                    log.warning("Cell %s failed: %s", tag, exc)
                    continue
                fname = f"{stem}__{tag}.png"
                out.save(args.out / fname)
                quality = psnr(init, out)
                rows.append(
                    {
                        "control": control,
                        "strength": strength,
                        "scale": scale,
                        "psnr_vs_input": round(quality, 2),
                        "file": fname,
                        "synthid_oracle": "",  # fill: clean / present
                        "text_legible": "",  # fill: yes / no / partial
                    }
                )
                table.add_row(control, f"{strength:g}", f"{scale:g}", f"{quality:.2f}", fname)
        del pipe
        _free_memory(device)

    _write_index(args.out, stem, rows, init_path.name)
    console.print(table)
    console.print(f"\nWrote {len(rows)} cells to {args.out}/")
    console.print(f"Next: open {args.out}/sweep_index.csv, run each PNG through the Gemini SynthID oracle,")
    console.print("fill synthid_oracle (clean/present) + text_legible (yes/no/partial), find the Pareto cell.")
    return 0


def _free_memory(device: str) -> None:
    import gc

    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()


def _write_index(out: Path, stem: str, rows: list[dict[str, Any]], input_name: str) -> None:
    """Write the CSV index (with empty verdict columns) and a README."""
    fields = ["control", "strength", "scale", "psnr_vs_input", "file", "synthid_oracle", "text_legible"]
    with (out / "sweep_index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    readme = (
        f"# ControlNet sweep for {stem}\n\n"
        f"Input (resized): {input_name}\n\n"
        "Each row in sweep_index.csv is one (control, strength, scale) cell. PSNR vs the\n"
        "resized input is a COARSE global-fidelity proxy only -- it does NOT measure text\n"
        "legibility or watermark presence. Decide those two by hand:\n\n"
        "1. synthid_oracle: open the PNG in the Gemini app, 'Verify with SynthID'. Mark\n"
        "   'clean' if no SynthID is detected, 'present' if it still is. (No local detector\n"
        "   exists; this manual check is the only valid SynthID oracle.)\n"
        "2. text_legible: eyeball the small/CJK text. Mark yes / partial / no.\n\n"
        "The Pareto cell is the one where synthid_oracle=clean AND text_legible=yes at the\n"
        "lowest strength. If no cell satisfies both, the canny/tile-ControlNet middle path\n"
        "is dead for text and a glyph re-render is required (see\n"
        "docs/text-protection-research.md). Record the outcome in\n"
        "docs/controlnet-removal-pipeline-research.md.\n"
    )
    (out / "sweep_README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
