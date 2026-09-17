"""Typed generation-profile capabilities shared by prompts and projection checks."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import cast

from .validation import CheckResult, IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class PerformanceChoice:
    value: str
    description: str


@dataclass(frozen=True, slots=True)
class PerformanceFieldVocabulary:
    field_name: str
    choices: tuple[PerformanceChoice, ...]


@dataclass(frozen=True, slots=True)
class PerformanceAspectCapability:
    aspect_id: str
    description: str
    fields: tuple[PerformanceFieldVocabulary, ...]


@dataclass(frozen=True, slots=True)
class ScoreRelationCapabilities:
    """Internal score-relation limits derived from a generation profile ID."""

    explicit_variation_roles: frozenset[str]
    transition_roles: frozenset[str]
    accompaniment_reuse: str
    instructions: tuple[str, ...]

    def to_prompt_json(self) -> dict[str, object]:
        return {
            "explicit_variation_roles": sorted(self.explicit_variation_roles),
            "transition_roles": sorted(self.transition_roles),
            "accompaniment_reuse": self.accompaniment_reuse,
            "instructions": list(self.instructions),
        }


@dataclass(frozen=True, slots=True)
class GenerationProfileCapabilities:
    profile_id: str
    instrumentation: str
    target_duration_seconds: int
    material_placement_roles: frozenset[str]
    minimum_material_placements_per_leaf: int
    maximum_material_placements_per_leaf: int | None
    performance_direction_target_types: frozenset[str]
    performance_choice_vocabulary_version: str
    performance_aspects: tuple[PerformanceAspectCapability, ...]
    velocity_policy_ids: frozenset[str]

    @property
    def performance_aspect_ids(self) -> tuple[str, ...]:
        return tuple(aspect.aspect_id for aspect in self.performance_aspects)

    def performance_aspect_description(self, aspect_id: str) -> str:
        for aspect in self.performance_aspects:
            if aspect.aspect_id == aspect_id:
                return aspect.description
        raise KeyError(aspect_id)

    def performance_fields_for_aspect(self, aspect_id: str) -> tuple[str, ...]:
        for aspect in self.performance_aspects:
            if aspect.aspect_id == aspect_id:
                return tuple(field.field_name for field in aspect.fields)
        raise KeyError(aspect_id)

    def performance_choices_for_field(self, field_name: str) -> tuple[str, ...]:
        for aspect in self.performance_aspects:
            for field in aspect.fields:
                if field.field_name == field_name:
                    return tuple(choice.value for choice in field.choices)
        raise KeyError(field_name)

    def supports_material_placement_role(self, role: str) -> bool:
        return role in self.material_placement_roles

    def allows_material_placement_count(self, count: int) -> bool:
        if count < self.minimum_material_placements_per_leaf:
            return False
        maximum = self.maximum_material_placements_per_leaf
        return maximum is None or count <= maximum

    def supports_performance_direction_target(self, target_type: str) -> bool:
        return target_type in self.performance_direction_target_types

    def supports_performance_aspect(self, aspect_id: str) -> bool:
        return aspect_id in self.performance_aspect_ids

    def supports_velocity_policy(self, velocity_policy_id: str) -> bool:
        return velocity_policy_id in self.velocity_policy_ids

    @property
    def score_relations(self) -> ScoreRelationCapabilities:
        if self.profile_id != "solo_piano_3m_v2":
            raise ValueError("score relation capabilities are undefined for this profile")
        return ScoreRelationCapabilities(
            explicit_variation_roles=frozenset({"foreground"}),
            transition_roles=frozenset({"foreground"}),
            accompaniment_reuse="implicit_previous_same_material",
            instructions=(
                "明示的な変奏は、伴奏として使うマテリアルまたはマテリアル配置を含めない。",
                "遷移の元、遷移専用、先は、すべて前景のマテリアル配置にする。",
                "伴奏で同じマテリアルを再利用するときは、明示的な変奏を作らない。",
            ),
        )


def generation_profile_capabilities_to_json(
    capabilities: GenerationProfileCapabilities,
) -> dict[str, object]:
    """Return the stable JSON representation used by immutable run inputs."""
    return {
        "profile_id": capabilities.profile_id,
        "instrumentation": capabilities.instrumentation,
        "target_duration_seconds": capabilities.target_duration_seconds,
        "material_placement_roles": sorted(capabilities.material_placement_roles),
        "minimum_material_placements_per_leaf": (capabilities.minimum_material_placements_per_leaf),
        "maximum_material_placements_per_leaf": (capabilities.maximum_material_placements_per_leaf),
        "performance_direction_target_types": sorted(
            capabilities.performance_direction_target_types
        ),
        "performance_choice_vocabulary_version": (
            capabilities.performance_choice_vocabulary_version
        ),
        "performance_aspects": [
            {
                "aspect_id": aspect.aspect_id,
                "description": aspect.description,
                "fields": [
                    {
                        "field_name": field.field_name,
                        "choices": [
                            {"value": choice.value, "description": choice.description}
                            for choice in field.choices
                        ],
                    }
                    for field in aspect.fields
                ],
            }
            for aspect in capabilities.performance_aspects
        ],
        "velocity_policy_ids": sorted(capabilities.velocity_policy_ids),
    }


def generation_profile_capabilities_from_json(
    value: Mapping[str, object],
) -> GenerationProfileCapabilities:
    """Read the stable JSON representation without stringifying set-valued fields."""
    expected = {
        "profile_id",
        "instrumentation",
        "target_duration_seconds",
        "material_placement_roles",
        "minimum_material_placements_per_leaf",
        "maximum_material_placements_per_leaf",
        "performance_direction_target_types",
        "performance_choice_vocabulary_version",
        "performance_aspects",
        "velocity_policy_ids",
    }
    if set(value) != expected:
        raise ValueError("generation profile capabilities contain unexpected fields")
    list_fields = (
        "material_placement_roles",
        "performance_direction_target_types",
        "velocity_policy_ids",
    )
    if any(
        not isinstance(value[field], list)
        or any(not isinstance(item, str) for item in cast(list[object], value[field]))
        for field in list_fields
    ):
        raise ValueError("generation profile capability sets must be JSON string arrays")
    maximum = value["maximum_material_placements_per_leaf"]
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int)):
        raise ValueError("maximum material placement count must be an integer or null")
    scalar_types = (
        isinstance(value["profile_id"], str),
        isinstance(value["instrumentation"], str),
        isinstance(value["performance_choice_vocabulary_version"], str),
        isinstance(value["target_duration_seconds"], int)
        and not isinstance(value["target_duration_seconds"], bool),
        isinstance(value["minimum_material_placements_per_leaf"], int)
        and not isinstance(value["minimum_material_placements_per_leaf"], bool),
    )
    if not all(scalar_types):
        raise ValueError("generation profile capability scalar fields are invalid")
    performance_aspects = _performance_aspects_from_json(value["performance_aspects"])
    return GenerationProfileCapabilities(
        profile_id=cast(str, value["profile_id"]),
        instrumentation=cast(str, value["instrumentation"]),
        target_duration_seconds=cast(int, value["target_duration_seconds"]),
        material_placement_roles=frozenset(cast(list[str], value["material_placement_roles"])),
        minimum_material_placements_per_leaf=cast(
            int, value["minimum_material_placements_per_leaf"]
        ),
        maximum_material_placements_per_leaf=cast(int | None, maximum),
        performance_direction_target_types=frozenset(
            cast(list[str], value["performance_direction_target_types"])
        ),
        performance_choice_vocabulary_version=cast(
            str, value["performance_choice_vocabulary_version"]
        ),
        performance_aspects=performance_aspects,
        velocity_policy_ids=frozenset(cast(list[str], value["velocity_policy_ids"])),
    )


def _performance_aspects_from_json(value: object) -> tuple[PerformanceAspectCapability, ...]:
    if not isinstance(value, list):
        raise ValueError("performance aspects must be a JSON array")
    aspects: list[PerformanceAspectCapability] = []
    for raw_aspect in value:
        if not isinstance(raw_aspect, dict) or set(raw_aspect) != {
            "aspect_id",
            "description",
            "fields",
        }:
            raise ValueError("performance aspect fields are invalid")
        if not isinstance(raw_aspect["aspect_id"], str) or not isinstance(
            raw_aspect["description"], str
        ):
            raise ValueError("performance aspect metadata is invalid")
        raw_fields = raw_aspect["fields"]
        if not isinstance(raw_fields, list):
            raise ValueError("performance aspect fields must be a JSON array")
        fields: list[PerformanceFieldVocabulary] = []
        for raw_field in raw_fields:
            if not isinstance(raw_field, dict) or set(raw_field) != {"field_name", "choices"}:
                raise ValueError("performance field vocabulary is invalid")
            raw_choices = raw_field["choices"]
            if not isinstance(raw_field["field_name"], str) or not isinstance(raw_choices, list):
                raise ValueError("performance field vocabulary metadata is invalid")
            choices: list[PerformanceChoice] = []
            for raw_choice in raw_choices:
                if (
                    not isinstance(raw_choice, dict)
                    or set(raw_choice) != {"value", "description"}
                    or not isinstance(raw_choice["value"], str)
                    or not isinstance(raw_choice["description"], str)
                ):
                    raise ValueError("performance choice is invalid")
                choices.append(
                    PerformanceChoice(
                        value=raw_choice["value"], description=raw_choice["description"]
                    )
                )
            fields.append(
                PerformanceFieldVocabulary(
                    field_name=raw_field["field_name"], choices=tuple(choices)
                )
            )
        aspects.append(
            PerformanceAspectCapability(
                aspect_id=raw_aspect["aspect_id"],
                description=raw_aspect["description"],
                fields=tuple(fields),
            )
        )
    return tuple(aspects)


def _choices(*values: tuple[str, str]) -> tuple[PerformanceChoice, ...]:
    return tuple(
        PerformanceChoice(value=value, description=description) for value, description in values
    )


def check_generation_profile_document(
    document: Mapping[str, object], capabilities: GenerationProfileCapabilities
) -> CheckResult:
    """List every profile capability that a structurally valid script exceeds."""
    script = cast(dict[str, object], document["script"])
    setup = cast(dict[str, object], script["performance_setup"])
    issues: list[ValidationIssue] = []
    if setup["instrumentation"] != capabilities.instrumentation:
        issues.append(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                "The instrumentation is unsupported by the generation profile",
                "/script/performance_setup/instrumentation",
            )
        )
    if setup["target_duration_seconds"] != capabilities.target_duration_seconds:
        issues.append(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                "The target duration is unsupported by the generation profile",
                "/script/performance_setup/target_duration_seconds",
            )
        )
    placements = cast(dict[str, dict[str, object]], script["material_placements"])
    for placement_id, placement in placements.items():
        if not capabilities.supports_material_placement_role(cast(str, placement["role"])):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "The material placement role is unsupported by the generation profile",
                    f"/script/material_placements/{placement_id}/role",
                )
            )
    directions = cast(dict[str, dict[str, object]], setup["performance_directions"])
    for direction_id, direction in directions.items():
        target = cast(dict[str, object], direction["target"])
        if not capabilities.supports_performance_direction_target(cast(str, target["type"])):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "The performance direction target is unsupported by the generation profile",
                    f"/script/performance_setup/performance_directions/{direction_id}/target/type",
                )
            )
        raw_aspects = direction.get("performance_aspects")
        if raw_aspects is None:
            continue
        for aspect_index, aspect_id in enumerate(cast(list[str], raw_aspects)):
            if not capabilities.supports_performance_aspect(aspect_id):
                issues.append(
                    ValidationIssue(
                        IssueCode.UNREPRESENTABLE,
                        "The performance aspect is unsupported by the generation profile",
                        "/script/performance_setup/performance_directions/"
                        f"{direction_id}/performance_aspects/{aspect_index}",
                    )
                )
    sections = cast(dict[str, dict[str, object]], script["sections"])
    parent_ids = {
        cast(str, section["parent_section_id"])
        for section in sections.values()
        if section["parent_section_id"] is not None
    }
    placement_count_by_section: dict[str, int] = {}
    for placement in placements.values():
        section_id = cast(str, placement["section_id"])
        placement_count_by_section[section_id] = placement_count_by_section.get(section_id, 0) + 1
    for section_id in sections:
        if section_id not in parent_ids and not capabilities.allows_material_placement_count(
            placement_count_by_section.get(section_id, 0)
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "The leaf section placement count is unsupported by the generation profile",
                    f"/script/sections/{section_id}",
                )
            )
    return CheckResult(not issues, tuple(issues))


@cache
def solo_piano_3m_v2_capabilities() -> GenerationProfileCapabilities:
    """Return the single source of profile capability truth for the v2 vertical slice."""
    return GenerationProfileCapabilities(
        profile_id="solo_piano_3m_v2",
        instrumentation="solo_piano",
        target_duration_seconds=180,
        material_placement_roles=frozenset({"foreground", "accompaniment"}),
        minimum_material_placements_per_leaf=1,
        maximum_material_placements_per_leaf=None,
        performance_direction_target_types=frozenset({"section"}),
        performance_choice_vocabulary_version="0.1.0",
        performance_aspects=(
            PerformanceAspectCapability(
                aspect_id="timing",
                description="曲の流れを保ったまま、溜めや前進感を変える。",
                fields=(
                    PerformanceFieldVocabulary(
                        "timing_profile",
                        _choices(
                            ("neutral", "楽譜の時間をそのまま使う。"),
                            ("savor", "句の始めと終わりに溜めを作る。"),
                            ("flow", "揺れを抑えて流れよく進める。"),
                            ("build", "後半へ向かって前進させる。"),
                            ("release", "終わりへ向かって緩める。"),
                        ),
                    ),
                    PerformanceFieldVocabulary(
                        "timing_amount",
                        _choices(
                            ("subtle", "時間変化を控えめにする。"),
                            ("moderate", "時間変化を明確にする。"),
                        ),
                    ),
                ),
            ),
            PerformanceAspectCapability(
                aspect_id="dynamics",
                description="音符を変えずに、音量の流れを変える。",
                fields=(
                    PerformanceFieldVocabulary(
                        "dynamics_profile",
                        _choices(
                            ("steady", "音量を安定させる。"),
                            ("shape", "句の中に音量の山を作る。"),
                            ("build", "後半へ向かって強める。"),
                            ("release", "終わりへ向かって弱める。"),
                        ),
                    ),
                ),
            ),
            PerformanceAspectCapability(
                aspect_id="articulation",
                description="音符を保持する長さと音のつながり方を変える。",
                fields=(
                    PerformanceFieldVocabulary(
                        "articulation_profile",
                        _choices(
                            ("score", "楽譜上の音価を基準にする。"),
                            ("legato", "音を長めに保ってつなげる。"),
                            ("light", "音を短めにして軽くする。"),
                        ),
                    ),
                ),
            ),
            PerformanceAspectCapability(
                aspect_id="coordination",
                description="楽譜上で同時の音を、揃えるかずらすかを変える。",
                fields=(
                    PerformanceFieldVocabulary(
                        "coordination_profile",
                        _choices(
                            ("score", "楽譜上で同時の音を揃える。"),
                            ("rolled", "低い音から短くずらして弾く。"),
                            ("aligned", "縦の線を明確に揃える。"),
                        ),
                    ),
                ),
            ),
            PerformanceAspectCapability(
                aspect_id="pedal",
                description="音符を変えずに、ペダルによる響きの保持を変える。",
                fields=(
                    PerformanceFieldVocabulary(
                        "pedal_profile",
                        _choices(
                            ("none", "サステインペダルを使わない。"),
                            ("phrase_legato", "フレーズのまとまりで踏み替える。"),
                            ("harmony_legato", "和声の変化に合わせて踏み替える。"),
                        ),
                    ),
                ),
            ),
        ),
        velocity_policy_ids=frozenset({"legacy-unison-v1"}),
    )
