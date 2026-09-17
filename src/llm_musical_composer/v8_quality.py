"""採用済みv8基準曲の品質検査を、宣言主音へ追従させて再利用する。"""

from __future__ import annotations

from llm_musical_composer.brightness_control_run import _quality_gate


def quality_gate_v8(plan, score, performance, rendered, base_score) -> dict[str, object]:
    """既存v8ゲートを実行し、終止の期待音だけを宣言主音から一般化する。"""

    result = _quality_gate(plan, score, performance, rendered, base_score)
    failures = list(result["failures"])
    if "final-tonic-pitches" not in failures:
        return result
    third = 4 if plan.mode == "major" else 3
    expected = {
        plan.tonal_center,
        (plan.tonal_center + third) % 12,
        (plan.tonal_center + 7) % 12,
    }
    actual = set(result["ending"]["final_pitch_classes"])
    if actual != expected:
        return result
    failures.remove("final-tonic-pitches")
    return {**result, "failures": failures, "passes": not failures}

