import copy
import json
from pathlib import Path

from scoim.projection import (
    check_solo_piano_3m_structure,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "scoim" / "fixed-aba" / "approved-script.json"


def _draft() -> dict[str, object]:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    document["status"] = "draft"
    document["approval"] = None
    document["script"].setdefault("requirements", {})
    return copy.deepcopy(document)


def test_profile_structure_accepts_a_compatible_draft() -> None:
    result = check_solo_piano_3m_structure(_draft())

    assert result.valid is True
    assert result.issues == ()


def test_profile_structure_rejects_an_unplaced_material() -> None:
    document = _draft()
    document["script"]["materials"]["unused"] = {
        "kind": "theme",
        "description": "配置されない素材。",
    }

    result = check_solo_piano_3m_structure(document)

    assert result.valid is False
    assert result.issues[0].path == "/script/materials/unused"


def test_profile_structure_requires_a_final_release() -> None:
    document = _draft()
    document["script"]["sections"]["release"]["role"] = "statement"

    result = check_solo_piano_3m_structure(document)

    assert result.valid is False
    assert result.issues[0].path == "/script/sections/release/role"


def test_profile_structure_requires_one_ending_material() -> None:
    document = _draft()
    document["script"]["materials"]["contrast"]["kind"] = "ending"

    result = check_solo_piano_3m_structure(document)

    assert result.valid is False
    assert result.issues[0].path == "/script/materials"


def test_profile_structure_requires_music_before_the_ending() -> None:
    document = _draft()
    script = document["script"]
    script["sections"] = {
        "whole": script["sections"]["whole"],
        "release": {**script["sections"]["release"], "order": 0},
    }
    script["materials"] = {"cadence": script["materials"]["cadence"]}
    script["placements"] = {"cadence_last": script["placements"]["cadence_last"]}
    script["variations"] = {}
    script["transitions"] = {}
    script["performance_setup"]["performance_directions"] = {}

    result = check_solo_piano_3m_structure(document)

    assert result.valid is False
    assert result.issues[0].path == "/script/sections"


def test_profile_structure_requires_ordinary_material_before_the_ending() -> None:
    document = _draft()
    materials = document["script"]["materials"]
    materials["theme"]["kind"] = "transition"
    materials["contrast"]["kind"] = "transition"

    result = check_solo_piano_3m_structure(document)

    assert result.valid is False
    assert result.issues[0].path == "/script/sections/release"


def test_profile_structure_requires_a_two_second_nominal_ending() -> None:
    document = _draft()
    document["script"]["sections"]["release"]["relative_length"] = 0.01

    result = check_solo_piano_3m_structure(document)

    assert result.valid is False
    assert result.issues[0].path == "/script/sections/release/relative_length"


def test_profile_structure_matches_connectors_to_transition_materials() -> None:
    document = _draft()
    document["script"]["materials"]["connector"]["kind"] = "theme"

    result = check_solo_piano_3m_structure(document)

    assert result.valid is False
    assert result.issues[0].path == "/script/placements/bridge_ab"


def test_profile_structure_rejects_an_unsupported_material_kind() -> None:
    document = _draft()
    document["script"]["materials"]["contrast"]["kind"] = "episode"

    result = check_solo_piano_3m_structure(document)

    assert result.valid is False
    assert result.issues[0].path == "/script/materials/contrast/kind"
