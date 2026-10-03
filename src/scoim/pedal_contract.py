"""Versioned pedal semantics shared by generation, saved runs, and replay."""

LEGACY_PEDAL = "legacy-pedal-v1"
HARMONY_RELEASE_PEDAL = "harmony-release-attack-v1"


def pedal_vocabulary_version(contract: str) -> str:
    if contract == LEGACY_PEDAL:
        return "0.1.0"
    if contract == HARMONY_RELEASE_PEDAL:
        return "0.2.0"
    raise ValueError(f"unsupported pedal contract: {contract}")
