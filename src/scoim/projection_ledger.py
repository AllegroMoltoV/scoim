"""Projection evidence shared across SCoIM generation phases."""

from dataclasses import dataclass


class ProjectionLedgerValidationError(ValueError):
    """A projection ledger cannot support the claimed result."""


@dataclass(frozen=True, slots=True)
class ProjectionLedgerEntry:
    source_kind: str
    source_id: str
    target_kind: str
    target_id: str
    target_stage: str
    verification: str
    status: str
    evidence: str


def validate_projection_ledger(entries: tuple[ProjectionLedgerEntry, ...]) -> None:
    """Require evidence before a mechanical projection can be treated as complete."""
    allowed_statuses = {"planned", "passed", "failed", "unchecked", "unverified"}
    for entry in entries:
        if entry.status not in allowed_statuses:
            raise ProjectionLedgerValidationError(f"unknown projection status: {entry.status}")
        if entry.verification != "natural_language_claim" and entry.status in {
            "planned",
            "unchecked",
            "unverified",
        }:
            raise ProjectionLedgerValidationError(
                f"unchecked mechanical target: {entry.source_kind}:{entry.source_id}"
            )
        if entry.status in {"passed", "failed"} and not entry.evidence:
            raise ProjectionLedgerValidationError(
                "a checked projection target must include evidence"
            )


def validate_expected_projection_targets(
    entries: tuple[ProjectionLedgerEntry, ...],
    expected_targets: dict[str, frozenset[str]],
) -> None:
    """Require every expected target identifier to appear exactly in its target kind."""
    actual_targets: dict[str, set[str]] = {target_kind: set() for target_kind in expected_targets}
    for entry in entries:
        if entry.target_kind in actual_targets:
            if entry.target_id in actual_targets[entry.target_kind]:
                raise ProjectionLedgerValidationError(
                    f"duplicate projection target: {entry.target_kind}:{entry.target_id}"
                )
            actual_targets[entry.target_kind].add(entry.target_id)
    for target_kind, expected_ids in expected_targets.items():
        missing = expected_ids - actual_targets[target_kind]
        if missing:
            raise ProjectionLedgerValidationError(
                f"missing projection target: {target_kind}:{min(missing)}"
            )
        unexpected = actual_targets[target_kind] - expected_ids
        if unexpected:
            raise ProjectionLedgerValidationError(
                f"unexpected projection target: {target_kind}:{min(unexpected)}"
            )
