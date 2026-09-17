from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import Any

from llm_musical_composer.run_state import sha256_json

PRESET = "solo_piano_3m_v1"
CONTROL_IDS = {
    "あかるさ": "brightness",
    "高さ": "height",
    "発音頻度": "attack_frequency",
}
PUBLISHED_CONTROLS: frozenset[str] = frozenset()
REQUEST_FIELDS = frozenset({"schema_version", "preset", "reference", "controls"})


class CompositionRequestError(ValueError):
    """Raised when a public composition request is not valid or not yet supported."""


@dataclass(frozen=True)
class NormalizedCompositionRequest:
    value: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class ResolvedCompositionRequest:
    value: dict[str, Any]
    sha256: str


def _normalize_reference(request: Mapping[str, object]) -> dict[str, str]:
    if "reference" not in request:
        return {"mode": "unresolved_default"}
    reference = request["reference"]
    if not isinstance(reference, str):
        raise CompositionRequestError("reference must be a file name string")
    name = reference.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise CompositionRequestError("reference must be a non-empty file name without a path")
    return {"mode": "explicit", "name": name}


def _normalize_controls(
    raw_controls: object,
    *,
    published_controls: Set[str],
) -> dict[str, dict[str, float | str]]:
    if not isinstance(raw_controls, Mapping):
        raise CompositionRequestError("controls must be a JSON object")
    unknown_publications = set(published_controls) - set(CONTROL_IDS)
    if unknown_publications:
        raise CompositionRequestError(
            f"published_controls contains unknown labels: {sorted(unknown_publications)}"
        )
    normalized: list[tuple[str, dict[str, float | str]]] = []
    for label, value in raw_controls.items():
        if not isinstance(label, str) or label not in CONTROL_IDS:
            raise CompositionRequestError(f"unknown control: {label!r}")
        if label not in published_controls:
            raise CompositionRequestError(f"control is not published: {label}")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -1.0 <= float(value) <= 1.0
        ):
            raise CompositionRequestError(
                f"control {label!r} must be a finite number from -1.0 through 1.0"
            )
        if label == "あかるさ" and float(value) not in {-1.0, 0.0, 1.0}:
            raise CompositionRequestError("control 'あかるさ' must be -1, 0, or 1")
        control_id = CONTROL_IDS[label]
        normalized.append((control_id, {"label": label, "value": float(value)}))
    return dict(sorted(normalized))


def normalize_composition_request(
    raw_request: object,
    *,
    published_controls: Set[str] = PUBLISHED_CONTROLS,
) -> NormalizedCompositionRequest:
    if not isinstance(raw_request, Mapping):
        raise CompositionRequestError("composition request must be a JSON object")
    unknown = set(raw_request) - REQUEST_FIELDS
    if unknown:
        raise CompositionRequestError(f"unknown field(s): {sorted(unknown)}")
    version = raw_request.get("schema_version")
    if isinstance(version, bool) or version != 1:
        raise CompositionRequestError("schema_version must be 1")
    if raw_request.get("preset") != PRESET:
        raise CompositionRequestError(f"preset must be {PRESET!r}")
    normalized = {
        "schema_version": 1,
        "preset": PRESET,
        "reference": _normalize_reference(raw_request),
        "controls": _normalize_controls(
            raw_request.get("controls", {}),
            published_controls=published_controls,
        ),
    }
    return NormalizedCompositionRequest(value=normalized, sha256=sha256_json(normalized))


def _validated_reference_hashes(raw_hashes: Mapping[str, object]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for name, digest in raw_hashes.items():
        if not isinstance(name, str) or not name:
            raise CompositionRequestError("reference hash name must be non-empty")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise CompositionRequestError(f"reference hash is invalid: {name}")
        folded = name.casefold()
        if folded in result:
            raise CompositionRequestError(f"duplicate reference name: {name}")
        result[folded] = (name, digest.lower())
    return result


def _unavailable_reason(reference_summary: Mapping[str, object], requested_name: str) -> str:
    for field in ("excluded", "unable_to_investigate"):
        records = reference_summary.get(field, [])
        if not isinstance(records, list):
            raise CompositionRequestError(f"reference summary {field} must be an array")
        for record in records:
            if not isinstance(record, Mapping):
                raise CompositionRequestError(f"reference summary {field} is invalid")
            if str(record.get("name", "")).casefold() == requested_name.casefold():
                return str(record.get("status") or record.get("error") or field)
    return "reference name is not in validated corpus"


def resolve_composition_request(
    normalized: NormalizedCompositionRequest,
    *,
    reference_summary: Mapping[str, object],
    reference_hashes: Mapping[str, object],
) -> ResolvedCompositionRequest:
    hashes = _validated_reference_hashes(reference_hashes)
    reference_request = normalized.value["reference"]
    if reference_summary.get("status") != "pass":
        reference = {
            "state": "unresolved_default",
            "reason": "reference profile v1 has not passed its controls",
        }
    elif reference_request["mode"] == "explicit":
        requested_name = reference_request["name"]
        found = hashes.get(requested_name.casefold())
        if found is None:
            reference = {
                "state": "unavailable",
                "requested_name": requested_name,
                "reason": _unavailable_reason(reference_summary, requested_name),
            }
        else:
            name, digest = found
            reference = {
                "state": "resolved",
                "name": name,
                "sha256": digest,
                "selection_method": "explicit",
            }
    else:
        try:
            selected = reference_summary["default_reference"]["selected"]
            if selected["kind"] != "real_medoid":
                raise KeyError("default reference is not a real medoid")
            default_name = str(selected["name"])
            name, digest = hashes[default_name.casefold()]
        except (KeyError, TypeError) as error:
            raise CompositionRequestError("validated default reference is unavailable") from error
        reference = {
            "state": "resolved",
            "name": name,
            "sha256": digest,
            "selection_method": "automatic_medoid",
        }
    resolved = {
        "schema_version": 1,
        "preset": PRESET,
        "normalized_request_sha256": normalized.sha256,
        "reference": reference,
        "controls": normalized.value["controls"],
    }
    return ResolvedCompositionRequest(value=resolved, sha256=sha256_json(resolved))
