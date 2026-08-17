from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/infer_text_lines.py"
SPEC = importlib.util.spec_from_file_location("infer_text_lines", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_choose_language_prefers_observed_unicode_script() -> None:
    probes = {"en": ("gibberish", 0.9), "ru": ("пример", 0.9), "ch": ("example", 0.9)}
    assert module.choose_language(probes) == "ru"

    probes["ch"] = ("示例", 0.9)
    assert module.choose_language(probes) == "ch"


def test_stable_recognition_requires_agreement_and_confidence() -> None:
    assert module.stable_recognition([("Sample text", 0.9), ("sample  text", 0.95)]) == "Sample text"
    assert module.stable_recognition([("Sample", 0.9), ("Simple", 0.95)]) is None
    assert module.stable_recognition([("Sample", 0.8), ("Sample", 0.95)]) is None
