import json
from pathlib import Path

import pytest

from llm_musical_composer.composition_request import (
    CONTROL_IDS,
    CompositionRequestError,
    normalize_composition_request,
    resolve_composition_request,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_candidate_control_ids_match_canonical_design() -> None:
    assert CONTROL_IDS == {
        "あかるさ": "brightness",
        "高さ": "height",
        "発音頻度": "attack_frequency",
    }


def test_minimum_request_keeps_default_reference_unresolved() -> None:
    normalized = normalize_composition_request({"schema_version": 1, "preset": "solo_piano_3m_v1"})

    assert normalized.value == {
        "schema_version": 1,
        "preset": "solo_piano_3m_v1",
        "reference": {"mode": "unresolved_default"},
        "controls": {},
    }
    assert len(normalized.sha256) == 64


def test_explicit_reference_is_preserved_without_resolving_it() -> None:
    normalized = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "reference": "WhiteCurtain.mid",
        }
    )

    assert normalized.value["reference"] == {
        "mode": "explicit",
        "name": "WhiteCurtain.mid",
    }


@pytest.mark.parametrize("key", ["candidate_count", "repair_count", "neighbor_count", "extra"])
def test_unknown_and_internal_keys_are_rejected(key: str) -> None:
    with pytest.raises(CompositionRequestError, match="unknown field"):
        normalize_composition_request({"schema_version": 1, "preset": "solo_piano_3m_v1", key: 1})


@pytest.mark.parametrize("raw_request", [[], "request", None])
def test_non_object_request_is_rejected(raw_request: object) -> None:
    with pytest.raises(CompositionRequestError, match="JSON object"):
        normalize_composition_request(raw_request)


@pytest.mark.parametrize(
    "raw_request, message",
    [
        ({"schema_version": 2, "preset": "solo_piano_3m_v1"}, "schema_version"),
        ({"schema_version": True, "preset": "solo_piano_3m_v1"}, "schema_version"),
        ({"schema_version": 1, "preset": "other"}, "preset"),
    ],
)
def test_version_and_preset_are_fixed(raw_request: dict[str, object], message: str) -> None:
    with pytest.raises(CompositionRequestError, match=message):
        normalize_composition_request(raw_request)


def test_unpublished_control_is_rejected() -> None:
    with pytest.raises(CompositionRequestError, match="not published"):
        normalize_composition_request(
            {
                "schema_version": 1,
                "preset": "solo_piano_3m_v1",
                "controls": {"高さ": 0.0},
            }
        )


def test_unknown_control_label_is_rejected() -> None:
    with pytest.raises(CompositionRequestError, match="unknown control"):
        normalize_composition_request(
            {
                "schema_version": 1,
                "preset": "solo_piano_3m_v1",
                "controls": {"速さ": 0.0},
            },
            published_controls={"高さ"},
        )


@pytest.mark.parametrize("controls", [[], "controls", None])
def test_controls_must_be_an_object(controls: object) -> None:
    with pytest.raises(CompositionRequestError, match=r"controls.*JSON object"):
        normalize_composition_request(
            {
                "schema_version": 1,
                "preset": "solo_piano_3m_v1",
                "controls": controls,
            }
        )


def test_unknown_published_control_configuration_is_rejected() -> None:
    with pytest.raises(CompositionRequestError, match="published_controls"):
        normalize_composition_request(
            {"schema_version": 1, "preset": "solo_piano_3m_v1"},
            published_controls={"速さ"},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.01, 1.01])
def test_invalid_control_value_is_rejected(value: float) -> None:
    with pytest.raises(CompositionRequestError, match=r"finite.*-1.0.*1.0"):
        normalize_composition_request(
            {
                "schema_version": 1,
                "preset": "solo_piano_3m_v1",
                "controls": {"高さ": value},
            },
            published_controls={"高さ"},
        )


@pytest.mark.parametrize("value", [True, False, "0.0", None])
def test_non_numeric_control_value_is_rejected(value: object) -> None:
    with pytest.raises(CompositionRequestError, match=r"finite.*-1.0.*1.0"):
        normalize_composition_request(
            {
                "schema_version": 1,
                "preset": "solo_piano_3m_v1",
                "controls": {"高さ": value},
            },
            published_controls={"高さ"},
        )


def test_control_order_does_not_change_normalized_value_or_hash() -> None:
    one = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "controls": {"発音頻度": 0.25, "高さ": -0.5},
        },
        published_controls={"高さ", "発音頻度"},
    )
    two = normalize_composition_request(
        {
            "controls": {"高さ": -0.5, "発音頻度": 0.25},
            "preset": "solo_piano_3m_v1",
            "schema_version": 1,
        },
        published_controls={"発音頻度", "高さ"},
    )

    assert one.value == two.value
    assert one.sha256 == two.sha256
    assert list(one.value["controls"]) == ["attack_frequency", "height"]


def test_omitted_control_is_not_silently_added_as_zero() -> None:
    normalized = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "controls": {"高さ": 0.5},
        },
        published_controls={"高さ", "発音頻度"},
    )

    assert normalized.value["controls"] == {"height": {"label": "高さ", "value": 0.5}}


@pytest.mark.parametrize("value", [-0.5, 0.5, 1.0 - 1e-9])
def test_brightness_accepts_only_three_integer_values(value: float) -> None:
    with pytest.raises(CompositionRequestError, match="-1, 0, or 1"):
        normalize_composition_request(
            {
                "schema_version": 1,
                "preset": "solo_piano_3m_v1",
                "controls": {"あかるさ": value},
            },
            published_controls={"あかるさ"},
        )


@pytest.mark.parametrize("value", [-1, 0, 1])
def test_brightness_accepts_three_integer_values(value: int) -> None:
    normalized = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "controls": {"あかるさ": value},
        },
        published_controls={"あかるさ"},
    )

    assert normalized.value["controls"]["brightness"]["value"] == float(value)


@pytest.mark.parametrize("reference", ["", "../WhiteCurtain.mid", "folder/song.mid", 42])
def test_invalid_reference_is_rejected(reference: object) -> None:
    with pytest.raises(CompositionRequestError, match="reference"):
        normalize_composition_request(
            {
                "schema_version": 1,
                "preset": "solo_piano_3m_v1",
                "reference": reference,
            }
        )


def test_public_json_schemas_are_strict_and_publish_no_faders_yet() -> None:
    request_schema = json.loads(
        (PROJECT_ROOT / "schemas" / "composition-request.schema.json").read_text(encoding="utf-8")
    )
    resolved_schema = json.loads(
        (PROJECT_ROOT / "schemas" / "resolved-composition-request.schema.json").read_text(
            encoding="utf-8"
        )
    )
    result_schema = json.loads(
        (PROJECT_ROOT / "schemas" / "composition-result.schema.json").read_text(encoding="utf-8")
    )

    assert request_schema["additionalProperties"] is False
    assert request_schema["properties"]["controls"]["maxProperties"] == 0
    assert resolved_schema["additionalProperties"] is False
    assert result_schema["additionalProperties"] is False
    assert set(result_schema["$defs"]["axisResult"]["properties"]["status"]["enum"]) == {
        "achieved",
        "not_achieved",
        "unverified",
        "not_applicable",
    }


def test_default_reference_resolves_to_validated_real_medoid() -> None:
    normalized = normalize_composition_request({"schema_version": 1, "preset": "solo_piano_3m_v1"})
    resolved = resolve_composition_request(
        normalized,
        reference_summary={
            "status": "pass",
            "default_reference": {"selected": {"kind": "real_medoid", "name": "Default.mid"}},
            "excluded": [],
            "unable_to_investigate": [],
        },
        reference_hashes={"Default.mid": "a" * 64, "Other.mid": "b" * 64},
    )

    assert resolved.value["reference"] == {
        "state": "resolved",
        "name": "Default.mid",
        "sha256": "a" * 64,
        "selection_method": "automatic_medoid",
    }
    assert len(resolved.sha256) == 64


def test_explicit_reference_resolution_distinguishes_unavailable_from_unknown() -> None:
    normalized = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "reference": "rut.mid",
        }
    )
    resolved = resolve_composition_request(
        normalized,
        reference_summary={
            "status": "pass",
            "default_reference": {"selected": {"kind": "real_medoid", "name": "Default.mid"}},
            "excluded": [{"name": "rut.mid", "status": "excluded"}],
            "unable_to_investigate": [],
        },
        reference_hashes={"Default.mid": "a" * 64},
    )

    assert resolved.value["reference"] == {
        "state": "unavailable",
        "requested_name": "rut.mid",
        "reason": "excluded",
    }


def test_failed_reference_profile_keeps_default_unresolved() -> None:
    normalized = normalize_composition_request({"schema_version": 1, "preset": "solo_piano_3m_v1"})
    resolved = resolve_composition_request(
        normalized,
        reference_summary={"status": "fail"},
        reference_hashes={},
    )

    assert resolved.value["reference"] == {
        "state": "unresolved_default",
        "reason": "reference profile v1 has not passed its controls",
    }


def test_explicit_available_reference_resolves_case_insensitively() -> None:
    normalized = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "reference": "default.mid",
        }
    )
    resolved = resolve_composition_request(
        normalized,
        reference_summary={
            "status": "pass",
            "default_reference": {"selected": {"kind": "real_medoid", "name": "Default.mid"}},
            "excluded": [],
            "unable_to_investigate": [],
        },
        reference_hashes={"Default.mid": "A" * 64},
    )

    assert resolved.value["reference"]["selection_method"] == "explicit"
    assert resolved.value["reference"]["name"] == "Default.mid"
    assert resolved.value["reference"]["sha256"] == "a" * 64


@pytest.mark.parametrize(
    "hashes, message",
    [
        ({"": "a" * 64}, "name"),
        ({"a.mid": "bad"}, "hash is invalid"),
        ({"a.mid": "a" * 64, "A.MID": "b" * 64}, "duplicate"),
    ],
)
def test_reference_hash_contract_is_validated(hashes: dict[str, str], message: str) -> None:
    normalized = normalize_composition_request({"schema_version": 1, "preset": "solo_piano_3m_v1"})
    with pytest.raises(CompositionRequestError, match=message):
        resolve_composition_request(
            normalized,
            reference_summary={"status": "fail"},
            reference_hashes=hashes,
        )


def test_unknown_reference_and_malformed_reference_summary_are_distinct() -> None:
    normalized = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "reference": "missing.mid",
        }
    )
    summary = {
        "status": "pass",
        "default_reference": {"selected": {"kind": "real_medoid", "name": "Default.mid"}},
        "excluded": [],
        "unable_to_investigate": [],
    }
    resolved = resolve_composition_request(
        normalized, reference_summary=summary, reference_hashes={"Default.mid": "a" * 64}
    )
    assert resolved.value["reference"]["reason"] == "reference name is not in validated corpus"

    for bad_records in ("bad", ["bad"]):
        bad_summary = {**summary, "excluded": bad_records}
        with pytest.raises(CompositionRequestError, match="reference summary"):
            resolve_composition_request(
                normalized,
                reference_summary=bad_summary,
                reference_hashes={"Default.mid": "a" * 64},
            )


def test_validated_default_must_be_a_real_available_medoid() -> None:
    normalized = normalize_composition_request({"schema_version": 1, "preset": "solo_piano_3m_v1"})
    for selected in (
        {"kind": "diagnostic_synthetic", "name": "Default.mid"},
        {"kind": "real_medoid", "name": "Missing.mid"},
    ):
        with pytest.raises(CompositionRequestError, match="default reference"):
            resolve_composition_request(
                normalized,
                reference_summary={
                    "status": "pass",
                    "default_reference": {"selected": selected},
                },
                reference_hashes={"Default.mid": "a" * 64},
            )
