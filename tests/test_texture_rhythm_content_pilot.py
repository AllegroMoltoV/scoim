from __future__ import annotations

from pathlib import Path

import pytest

from llm_musical_composer.performance_pipeline import ScoreHarmony, ScoreNote
from llm_musical_composer.texture_rhythm_content_pilot import (
    TextureContentDraftV0,
    TextureContentEventV0,
    TextureRhythmContentPilotError,
    TextureRhythmDraftV0,
    TextureRhythmGroupV0,
    dump_texture_content_draft,
    dump_texture_rhythm_draft,
    execute_texture_rhythm_content_pilot,
    join_texture_rhythm_content,
    parse_texture_content_draft,
    parse_texture_rhythm_draft,
    validate_texture_rhythm_draft,
)


class _FakeRunner:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    @property
    def call_number(self) -> int:
        return len(self.calls)

    def run(self, step_id: str, prompt: str, input_hashes: dict[str, str]):
        self.calls.append((step_id, prompt, input_hashes))
        return {
            "composition_source": self.responses.pop(0),
            "intent_summary": "fixture",
        }


def _harmonies() -> tuple[ScoreHarmony, ...]:
    return (
        ScoreHarmony("h1", 0, 4, 0, "major"),
        ScoreHarmony("h2", 4, 4, 7, "major"),
    )


def _melody() -> tuple[ScoreNote, ...]:
    return (
        ScoreNote("m1", 0, 2, 72, "upper"),
        ScoreNote("m2", 4, 2, 74, "upper"),
    )


def _budget() -> dict[str, object]:
    return {
        "required_texture_event_count": 4,
        "combined_attack_group_count": 3,
        "combined_note_event_count": 6,
        "attack_size_counts": {
            "one": 1,
            "two": 1,
            "three": 1,
            "four_or_more": 0,
        },
    }


def _feasibility() -> dict[str, object]:
    return {
        "onset_capacities": [
            {
                "harmony_index": 0,
                "start_units": 0,
                "end_units": 4,
                "maximum_new_accompaniment_attack_count": 2,
            },
            {
                "harmony_index": 1,
                "start_units": 4,
                "end_units": 8,
                "maximum_new_accompaniment_attack_count": 2,
            },
        ]
    }


def _rhythm() -> TextureRhythmDraftV0:
    return TextureRhythmDraftV0(
        (
            TextureRhythmGroupV0(0, 1),
            TextureRhythmGroupV0(2, 1),
            TextureRhythmGroupV0(4, 2),
        )
    )


def _content() -> TextureContentDraftV0:
    return TextureContentDraftV0(
        (
            (TextureContentEventV0(2, "root", "bass", ()),),
            (TextureContentEventV0(2, "fifth", "low", ("tenuto",)),),
            (
                TextureContentEventV0(4, "root", "bass", ()),
                TextureContentEventV0(4, "fifth", "low", ()),
            ),
        )
    )


def _validation_arguments() -> dict[str, object]:
    return {
        "material_length_units": 8,
        "harmonies": _harmonies(),
        "melody": _melody(),
        "budget": _budget(),
        "feasibility": _feasibility(),
        "maximum_group_size": 4,
    }


def test_rhythm_and_content_dsl_round_trip_and_join() -> None:
    rhythm_source = dump_texture_rhythm_draft(_rhythm())
    content_source = dump_texture_content_draft(_content())

    rhythm = parse_texture_rhythm_draft(rhythm_source)
    content = parse_texture_content_draft(content_source)
    validate_texture_rhythm_draft(rhythm, **_validation_arguments())
    joined = join_texture_rhythm_content(rhythm, content, _harmonies())

    assert rhythm == _rhythm()
    assert content == _content()
    assert [(item.harmony_index, item.at_units) for item in joined.events] == [
        (0, 0),
        (0, 2),
        (1, 4),
        (1, 4),
    ]


@pytest.mark.parametrize(
    "source",
    [
        "unknown(groups=[])",
        "texture_rhythm_draft([])",
        "texture_rhythm_draft(groups=[], extra=1)",
        "texture_rhythm_draft(groups=[texture_rhythm_group("
        "at_units=True, new_accompaniment_attack_count=1)])",
    ],
)
def test_rhythm_dsl_rejects_unsafe_or_ambiguous_source(source: str) -> None:
    with pytest.raises(TextureRhythmContentPilotError):
        parse_texture_rhythm_draft(source)


@pytest.mark.parametrize(
    ("rhythm", "message"),
    [
        (TextureRhythmDraftV0((TextureRhythmGroupV0(0, 1),) * 2), "strictly ordered"),
        (TextureRhythmDraftV0((TextureRhythmGroupV0(-1, 1),)), "material range"),
        (TextureRhythmDraftV0((TextureRhythmGroupV0(0, 0),)), "positive"),
        (
            TextureRhythmDraftV0(
                (TextureRhythmGroupV0(0, 1), TextureRhythmGroupV0(2, 3), TextureRhythmGroupV0(4, 0))
            ),
            "positive",
        ),
    ],
)
def test_rhythm_validation_rejects_invalid_group_shape(
    rhythm: TextureRhythmDraftV0, message: str
) -> None:
    with pytest.raises(TextureRhythmContentPilotError, match=message):
        validate_texture_rhythm_draft(rhythm, **_validation_arguments())


def test_rhythm_validation_rejects_budget_harmony_start_and_capacity() -> None:
    missing_harmony_start = TextureRhythmDraftV0(
        (
            TextureRhythmGroupV0(0, 1),
            TextureRhythmGroupV0(2, 1),
            TextureRhythmGroupV0(5, 2),
        )
    )
    with pytest.raises(TextureRhythmContentPilotError, match="harmony interval start"):
        validate_texture_rhythm_draft(missing_harmony_start, **_validation_arguments())

    over_capacity = TextureRhythmDraftV0(
        (
            TextureRhythmGroupV0(0, 1),
            TextureRhythmGroupV0(2, 3),
            TextureRhythmGroupV0(4, 2),
        )
    )
    with pytest.raises(TextureRhythmContentPilotError, match="onset capacity"):
        validate_texture_rhythm_draft(
            over_capacity,
            **{
                **_validation_arguments(),
                "budget": {
                    **_budget(),
                    "required_texture_event_count": 6,
                    "combined_note_event_count": 8,
                },
            },
        )


def test_rhythm_allows_melody_led_harmony_start_without_simultaneous_texture() -> None:
    harmonies = (ScoreHarmony("h1", 0, 8, 4, "diminished"),)
    melody = (ScoreNote("root", 0, 2, 52, "upper"),)
    feasibility = {
        "schema_version": 3,
        "onset_capacities": [
            {
                "harmony_index": 0,
                "start_units": 0,
                "end_units": 2,
                "maximum_new_accompaniment_attack_count": 0,
            },
            {
                "harmony_index": 0,
                "start_units": 2,
                "end_units": 8,
                "maximum_new_accompaniment_attack_count": 2,
            },
        ],
        "harmony_start_accompaniment": {
            "schema_version": 1,
            "policy_id": "melody-led-harmony-start-v1",
            "minimum_new_accompaniment_attacks": {"0": 0},
            "melody_led_starts": [
                {
                    "harmony_index": 0,
                    "harmony_id": "h1",
                    "at_units": 0,
                    "maximum_new_accompaniment_attack_count": 0,
                    "melody_pitches": [52],
                    "chord_pitch_classes": [4, 7, 10],
                    "reason": "capacity_zero_and_melody_attacks_chord_tones",
                }
            ],
        },
    }
    draft = TextureRhythmDraftV0((TextureRhythmGroupV0(2, 2),))

    result = validate_texture_rhythm_draft(
        draft,
        material_length_units=8,
        harmonies=harmonies,
        melody=melody,
        budget={
            "required_texture_event_count": 2,
            "combined_attack_group_count": 2,
            "combined_note_event_count": 3,
            "attack_size_counts": {
                "one": 1,
                "two": 1,
                "three": 0,
                "four_or_more": 0,
            },
        },
        feasibility=feasibility,
        maximum_group_size=4,
    )

    assert result["status"] == "pass"


def test_content_shape_is_fixed_by_rhythm() -> None:
    with pytest.raises(TextureRhythmContentPilotError, match="group count"):
        join_texture_rhythm_content(
            _rhythm(), TextureContentDraftV0(_content().groups[:-1]), _harmonies()
        )
    wrong = TextureContentDraftV0(
        ((_content().groups[0][0], _content().groups[0][0]), *_content().groups[1:])
    )
    with pytest.raises(TextureRhythmContentPilotError, match="event count"):
        join_texture_rhythm_content(_rhythm(), wrong, _harmonies())


def test_release_rhythm_rejects_late_attack_and_thin_final_chord() -> None:
    arguments = {
        "material_length_units": 8,
        "harmonies": _harmonies(),
        "melody": _melody(),
        "budget": _budget(),
        "feasibility": _feasibility(),
        "maximum_group_size": 4,
        "required_final_attack_units": 4,
        "minimum_required_final_new_accompaniment_attack_count": 2,
    }
    late = TextureRhythmDraftV0(
        (
            TextureRhythmGroupV0(0, 1),
            TextureRhythmGroupV0(2, 1),
            TextureRhythmGroupV0(4, 1),
            TextureRhythmGroupV0(6, 1),
        )
    )
    with pytest.raises(TextureRhythmContentPilotError, match="after shared ending"):
        validate_texture_rhythm_draft(late, **arguments)

    thin = TextureRhythmDraftV0(
        (
            TextureRhythmGroupV0(0, 2),
            TextureRhythmGroupV0(2, 1),
            TextureRhythmGroupV0(4, 1),
        )
    )
    with pytest.raises(TextureRhythmContentPilotError, match="final accompaniment"):
        validate_texture_rhythm_draft(thin, **arguments)


def test_two_stage_executor_calls_each_stage_once(tmp_path: Path) -> None:
    runner = _FakeRunner(
        [dump_texture_rhythm_draft(_rhythm()), dump_texture_content_draft(_content())]
    )
    validated = []

    result = execute_texture_rhythm_content_pilot(
        tmp_path,
        runner,
        rhythm_prompt="rhythm prompt",
        content_prompt=lambda rhythm: f"content prompt {dump_texture_rhythm_draft(rhythm)}",
        input_hashes={"fixture": "a" * 64},
        rhythm_validation_arguments=_validation_arguments(),
        harmonies=_harmonies(),
        validate_joined=lambda draft: validated.append(draft) or {"status": "pass"},
    )

    assert [item[0] for item in runner.calls] == ["texture-rhythm", "texture-content"]
    assert len(validated) == 1
    assert result["status"] == "completed"
    assert result["external_call_count"] == 2


def test_two_stage_executor_stops_after_invalid_rhythm(tmp_path: Path) -> None:
    invalid = TextureRhythmDraftV0((TextureRhythmGroupV0(0, 1),))
    runner = _FakeRunner([dump_texture_rhythm_draft(invalid)])

    with pytest.raises(TextureRhythmContentPilotError):
        execute_texture_rhythm_content_pilot(
            tmp_path,
            runner,
            rhythm_prompt="rhythm prompt",
            content_prompt=lambda rhythm: "unused",
            input_hashes={"fixture": "a" * 64},
            rhythm_validation_arguments=_validation_arguments(),
            harmonies=_harmonies(),
            validate_joined=lambda draft: {"status": "pass"},
        )

    assert runner.call_number == 1
    assert (tmp_path / "failures/texture-rhythm.json").is_file()


def test_two_stage_executor_resumes_after_content_failure_without_rhythm_call(
    tmp_path: Path,
) -> None:
    first = _FakeRunner(
        [
            dump_texture_rhythm_draft(_rhythm()),
            "texture_content_draft(groups=[])",
        ]
    )
    with pytest.raises(TextureRhythmContentPilotError):
        execute_texture_rhythm_content_pilot(
            tmp_path,
            first,
            rhythm_prompt="rhythm prompt",
            content_prompt=lambda rhythm: "content prompt",
            input_hashes={"fixture": "a" * 64},
            rhythm_validation_arguments=_validation_arguments(),
            harmonies=_harmonies(),
            validate_joined=lambda draft: {"status": "pass"},
        )

    second = _FakeRunner([dump_texture_content_draft(_content())])
    result = execute_texture_rhythm_content_pilot(
        tmp_path,
        second,
        rhythm_prompt="rhythm prompt",
        content_prompt=lambda rhythm: "content prompt",
        input_hashes={"fixture": "a" * 64},
        rhythm_validation_arguments=_validation_arguments(),
        harmonies=_harmonies(),
        validate_joined=lambda draft: {"status": "pass"},
    )

    assert [item[0] for item in second.calls] == ["texture-content"]
    assert result["status"] == "completed"
