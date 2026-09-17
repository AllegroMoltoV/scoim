from scoim.profile_capabilities import (
    generation_profile_capabilities_from_json,
    generation_profile_capabilities_to_json,
    solo_piano_3m_v2_capabilities,
)


def test_solo_piano_v2_capabilities_define_the_fixed_instrument_and_duration() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    assert capabilities.instrumentation == "solo_piano"
    assert capabilities.target_duration_seconds == 180


def test_solo_piano_v2_capabilities_judge_supported_and_unsupported_roles() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    assert capabilities.supports_material_placement_role("foreground") is True
    assert capabilities.supports_material_placement_role("accompaniment") is True
    assert capabilities.supports_material_placement_role("percussion") is False


def test_solo_piano_v2_capabilities_allow_any_positive_placement_count() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    assert capabilities.allows_material_placement_count(1) is True
    assert capabilities.allows_material_placement_count(20) is True
    assert capabilities.allows_material_placement_count(0) is False


def test_solo_piano_v2_capabilities_reject_placement_level_performance_directions() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    assert capabilities.supports_performance_direction_target("section") is True
    assert capabilities.supports_performance_direction_target("material_placement") is False


def test_solo_piano_v2_capabilities_support_the_role_neutral_velocity_policy() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    assert capabilities.supports_velocity_policy("legacy-unison-v1") is True


def test_solo_piano_v2_capabilities_reject_role_aware_velocity_policies() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    assert (
        capabilities.supports_velocity_policy("foreground-accompaniment-harmony-shape-v1") is False
    )


def test_solo_piano_v2_capabilities_define_performance_aspects_and_choices() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    assert capabilities.performance_choice_vocabulary_version == "0.1.0"
    assert capabilities.performance_aspect_ids == (
        "timing",
        "dynamics",
        "articulation",
        "coordination",
        "pedal",
    )
    assert capabilities.performance_fields_for_aspect("timing") == (
        "timing_profile",
        "timing_amount",
    )
    assert capabilities.performance_fields_for_aspect("pedal") == ("pedal_profile",)
    assert capabilities.performance_choices_for_field("coordination_profile") == (
        "score",
        "rolled",
        "aligned",
    )
    assert capabilities.performance_aspect_description("articulation")


def test_generation_profile_capabilities_round_trip_through_stable_json() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    encoded = generation_profile_capabilities_to_json(capabilities)
    decoded = generation_profile_capabilities_from_json(encoded)

    assert decoded == capabilities
    assert encoded["material_placement_roles"] == ["accompaniment", "foreground"]
    assert encoded["performance_direction_target_types"] == ["section"]
    assert encoded["velocity_policy_ids"] == ["legacy-unison-v1"]
    assert encoded["performance_choice_vocabulary_version"] == "0.1.0"
    assert decoded.performance_aspect_ids == capabilities.performance_aspect_ids
    assert "score_relations" not in encoded
    assert decoded.score_relations == capabilities.score_relations
