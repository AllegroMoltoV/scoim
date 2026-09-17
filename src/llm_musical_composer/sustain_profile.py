"""新規生成素材のサステイン保持と踏み替えを決定的に検査する。"""

from __future__ import annotations

from llm_musical_composer.composition_ir import Composition, Material

PEDAL_OFF = 0
PEDAL_ON = 127
SUSTAIN_ON_MINIMUM = 64
MINIMUM_PEDAL_ON_RATIO = 0.85
MINIMUM_REPEDAL_GAP_MS = 100
MAXIMUM_REPEDAL_GAP_MS = 400
REPEDAL_REQUIRED_DURATION_MS = 8_000
FINAL_RELEASE_WINDOW_MS = 500


class SustainContractError(ValueError):
    """新規生成向けのサステイン契約違反。"""


class SustainRepairScopeError(ValueError):
    """サステイン修復がペダル以外を変更した場合のエラー。"""


def _material_report(material: Material) -> dict[str, object]:
    pedals = sorted(material.pedals, key=lambda event: (event.at_ms, event.event_id))
    issues: list[str] = []
    invalid_values = [
        f"{pedal.event_id}={pedal.value}"
        for pedal in pedals
        if pedal.value not in {PEDAL_OFF, PEDAL_ON}
    ]
    if invalid_values:
        issues.append("pedal values must be 0 or 127: " + ", ".join(invalid_values))

    if not pedals or pedals[0].at_ms != 0 or pedals[0].value != PEDAL_ON:
        issues.append("pedal events must start at 0 ms with value 127")
    if not pedals or pedals[-1].value != PEDAL_OFF:
        issues.append("pedal events must end with value 0")
    elif pedals[-1].at_ms < material.duration_ms - FINAL_RELEASE_WINDOW_MS:
        issues.append("final pedal release must be inside the final 500 ms")

    pedal_on = False
    cursor_ms = 0
    pedal_on_ms = 0
    pending_release_ms: int | None = None
    repedal_gaps_ms: list[int] = []
    for pedal in pedals:
        if pedal_on:
            pedal_on_ms += pedal.at_ms - cursor_ms
        next_pedal_on = pedal.value >= SUSTAIN_ON_MINIMUM
        if pedal_on and not next_pedal_on:
            pending_release_ms = pedal.at_ms
        elif next_pedal_on and not pedal_on and pending_release_ms is not None:
            repedal_gaps_ms.append(pedal.at_ms - pending_release_ms)
            pending_release_ms = None
        pedal_on = next_pedal_on
        cursor_ms = pedal.at_ms
    if pedal_on:
        pedal_on_ms += material.duration_ms - cursor_ms

    pedal_on_ratio = pedal_on_ms / material.duration_ms
    if pedal_on_ratio < MINIMUM_PEDAL_ON_RATIO:
        issues.append(
            f"pedal must be on for at least 85% of the material; got {pedal_on_ratio:.3f}"
        )
    if material.duration_ms >= REPEDAL_REQUIRED_DURATION_MS and len(repedal_gaps_ms) < 1:
        issues.append("materials of at least 8000 ms require at least one repedal")
    invalid_gaps = [
        gap
        for gap in repedal_gaps_ms
        if not MINIMUM_REPEDAL_GAP_MS <= gap <= MAXIMUM_REPEDAL_GAP_MS
    ]
    if invalid_gaps:
        issues.append(
            "repedal gaps must be between 100 and 400 ms: "
            + ", ".join(str(gap) for gap in invalid_gaps)
        )
    return {
        "material_id": material.material_id,
        "status": "pass" if not issues else "fail",
        "pedal_on_ms": pedal_on_ms,
        "pedal_on_ratio": round(pedal_on_ratio, 6),
        "repedal_count": len(repedal_gaps_ms),
        "repedal_gaps_ms": repedal_gaps_ms,
        "issues": issues,
    }


def evaluate_sustain_profile(materials: tuple[Material, ...]) -> dict[str, object]:
    """新規候補の全素材へサステイン契約を適用する。"""
    reports = [_material_report(material) for material in materials]
    issues = [
        f"material {report['material_id']}: {issue}"
        for report in reports
        for issue in report["issues"]
    ]
    return {
        "status": "pass" if reports and not issues else "fail",
        "issues": issues,
        "materials": reports,
    }


def validate_sustain_repair_scope(before: Composition, after: Composition) -> list[str]:
    """サステイン修復でペダル以外が変わっていないことを検査する。"""
    issues: list[str] = []
    for field_name in ("title", "form", "tonal_center", "mode", "ending"):
        if getattr(before, field_name) != getattr(after, field_name):
            issues.append(f"{field_name} changed")
    before_materials = before.material_by_id
    after_materials = after.material_by_id
    if set(before_materials) != set(after_materials):
        return [*issues, "material set changed"]
    if tuple(material.material_id for material in before.materials) != tuple(
        material.material_id for material in after.materials
    ):
        issues.append("material order changed")
    for material_id, before_material in before_materials.items():
        after_material = after_materials[material_id]
        if before_material.duration_ms != after_material.duration_ms:
            issues.append(f"material {material_id} duration changed")
        if before_material.notes != after_material.notes:
            issues.append(f"material {material_id} notes changed")
    return issues
