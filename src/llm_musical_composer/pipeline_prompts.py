"""段階別プロンプトを固定テンプレートから構築する。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

PromptStage = Literal["piece_plan", "score_spec", "performance_spec"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = {
    "piece_plan": PROJECT_ROOT / "prompts/pipeline-piece-plan.md",
    "score_spec": PROJECT_ROOT / "prompts/pipeline-score-spec.md",
    "performance_spec": PROJECT_ROOT / "prompts/pipeline-performance-spec.md",
}
PLACEHOLDERS = {
    "piece_plan": frozenset({"REQUEST_JSON", "REFERENCE_TARGET_JSON", "CALIBRATION_CONTRACT"}),
    "score_spec": frozenset(
        {
            "PIECE_PLAN_DSL",
            "REFERENCE_TARGET_JSON",
            "SOURCE_MATERIALS_DSL",
            "IDENTITY_CUES",
        }
    ),
    "performance_spec": frozenset(
        {"PIECE_PLAN_DSL", "REFERENCE_TARGET_JSON", "SCORE_SUMMARY", "PROFILE_CATALOG"}
    ),
}
PLACEHOLDER_PATTERN = re.compile(r"{{([A-Z_]+)}}")


class PipelinePromptError(ValueError):
    """段階別プロンプトの入力またはテンプレートが不完全であることを表す。"""


def build_stage_prompt(stage: PromptStage, values: dict[str, str]) -> str:
    """指定段階のテンプレートを、過不足のない値だけで展開する。"""
    expected = PLACEHOLDERS[stage]
    supplied = set(values)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        raise PipelinePromptError(f"prompt values mismatch; missing={missing}, unknown={unknown}")
    template = TEMPLATES[stage].read_text(encoding="utf-8")
    if frozenset(PLACEHOLDER_PATTERN.findall(template)) != expected:
        raise PipelinePromptError("prompt template placeholders do not match its stage")
    result = template
    for name in sorted(expected):
        result = result.replace("{{" + name + "}}", values[name])
    if PLACEHOLDER_PATTERN.search(result):
        raise PipelinePromptError("prompt contains an unresolved placeholder")
    return result
