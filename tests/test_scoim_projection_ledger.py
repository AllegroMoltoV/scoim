import pytest

from scoim.projection_ledger import (
    ProjectionLedgerEntry,
    ProjectionLedgerValidationError,
    validate_expected_projection_targets,
    validate_projection_ledger,
)


def test_projection_ledger_rejects_an_unchecked_mechanical_target() -> None:
    entry = ProjectionLedgerEntry(
        source_kind="section",
        source_id="statement",
        target_kind="score_unit",
        target_id="score-unit-statement",
        target_stage="score_spec",
        verification="direct_id_equality",
        status="unchecked",
        evidence="",
    )

    with pytest.raises(ProjectionLedgerValidationError, match="unchecked mechanical target"):
        validate_projection_ledger((entry,))


def test_projection_ledger_allows_an_unverified_natural_language_claim() -> None:
    entry = ProjectionLedgerEntry(
        source_kind="flow_scene",
        source_id="scene-001",
        target_kind="generation_context",
        target_id="scene-001",
        target_stage="script_relations",
        verification="natural_language_claim",
        status="unverified",
        evidence="description forwarded without an automatic musical judgment",
    )

    validate_projection_ledger((entry,))


def test_projection_ledger_rejects_a_missing_expected_target() -> None:
    with pytest.raises(ProjectionLedgerValidationError, match="missing projection target"):
        validate_expected_projection_targets(
            (),
            {"score_unit": frozenset({"score-unit-statement"})},
        )


def test_projection_ledger_rejects_an_unexpected_target() -> None:
    entry = ProjectionLedgerEntry(
        source_kind="section",
        source_id="statement",
        target_kind="score_unit",
        target_id="score-unit-unexpected",
        target_stage="piece_plan",
        verification="direct_id_equality",
        status="passed",
        evidence="identifier projected",
    )

    with pytest.raises(ProjectionLedgerValidationError, match="unexpected projection target"):
        validate_expected_projection_targets(
            (entry,),
            {"score_unit": frozenset()},
        )


def test_projection_ledger_rejects_a_duplicate_target() -> None:
    entry = ProjectionLedgerEntry(
        source_kind="section",
        source_id="statement",
        target_kind="score_unit",
        target_id="score-unit-statement",
        target_stage="piece_plan",
        verification="direct_id_equality",
        status="passed",
        evidence="identifier projected",
    )

    with pytest.raises(ProjectionLedgerValidationError, match="duplicate projection target"):
        validate_expected_projection_targets(
            (entry, entry),
            {"score_unit": frozenset({"score-unit-statement"})},
        )
