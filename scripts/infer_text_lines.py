# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "huggingface-hub>=0.20.0",
#   "numpy",
#   "onnxruntime>=1.24.0",
#   "opencv-python-headless<5",
#   "paddleocr>=3.3.3",
#   "paddlepaddle",
#   "pillow",
# ]
# ///
"""Infer stable source-text lines without modifying an image.

This evaluation-only dry run proposes line annotations for selective text
restoration. Every proposal still needs human verification: stable OCR can lose
punctuation with high confidence. It separately flags lines whose recognition
changes under crop jitter or whose minimum confidence is below the threshold.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

import click
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESTORATION_SCRIPT = ROOT / "scripts/selective_text_restoration.py"

from scripts._text_eval import normalize_text  # noqa: E402


def _load_restoration_module() -> Any:
    spec = importlib.util.spec_from_file_location("selective_text_restoration_for_inference", RESTORATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {RESTORATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _has_script(text: str, script: str) -> bool:
    return any(script in unicodedata.name(character, "") for character in text)


def choose_language(probes: dict[str, tuple[str, float]]) -> str:
    if _has_script(probes["ch"][0], "CJK"):
        return "ch"
    if _has_script(probes["ru"][0], "CYRILLIC"):
        return "ru"
    return "en"


def stable_recognition(reads: list[tuple[str, float]], min_score: float = 0.85) -> str | None:
    normalized = {normalize_text(text) for text, _score in reads}
    if len(normalized) != 1 or min(score for _text, score in reads) < min_score:
        return None
    return reads[0][0]


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--min-score", default=0.85, show_default=True, type=click.FloatRange(0.0, 1.0))
def main(source: Path, out: Path, min_score: float) -> None:
    """Write draft line text for SOURCE; manually verify every proposal."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR, TextRecognition

    restoration = _load_restoration_module()
    source_rgb = np.asarray(Image.open(source).convert("RGB"))
    detector = PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    engines = {
        "en": TextRecognition(model_name="en_PP-OCRv5_mobile_rec"),
        "ru": TextRecognition(model_name="eslav_PP-OCRv5_mobile_rec"),
        "ch": TextRecognition(model_name="PP-OCRv5_server_rec"),
    }
    boxes = restoration.detect_line_boxes(detector, source_rgb)
    accepted = []
    rejected = []
    for box in boxes:
        probes = {}
        for language, engine in engines.items():
            script = "cjk" if language == "ch" else "alphabetic"
            line = restoration.TextLine(box, "", script)
            probes[language] = restoration._recognize(engine, source_rgb, line, 0.1)
        language = choose_language(probes)
        script = "cjk" if language == "ch" else "alphabetic"
        line = restoration.TextLine(box, "", script)
        reads = [restoration._recognize(engines[language], source_rgb, line, ratio) for ratio in (0.08, 0.12, 0.2)]
        text = stable_recognition(reads, min_score)
        result = {
            "box": box,
            "script": script,
            "language": language,
            "reads": [{"text": value, "score": score} for value, score in reads],
        }
        if text is None:
            rejected.append(result)
        else:
            accepted.append({"box": box, "text": text, "script": script, "min_score": min(score for _, score in reads)})
    payload = {"source": source.name, "accepted": accepted, "rejected": rejected}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Accepted %s lines and rejected %s uncertain lines", len(accepted), len(rejected))


if __name__ == "__main__":
    main()
