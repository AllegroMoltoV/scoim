import json

from test_scoim_phase3_realization import SequencedRunner
from test_scoim_phase6_state import _phase6_run

from scoim.phase7_realization import Phase7Request, realize_phase7


def _response() -> dict[str, object]:
    return {
        "performances": [
            {
                "timing_profile": "savor",
                "timing_amount": "subtle",
                "dynamics_profile": "shape",
                "articulation_profile": None,
                "coordination_profile": None,
                "pedal_profile": None,
                "handled_directions": [
                    {
                        "direction_id": "statement-expression",
                        "field_names": [
                            "timing_profile",
                            "timing_amount",
                            "dynamics_profile",
                        ],
                    }
                ],
                "unhandled_directions": [],
            }
        ]
    }


def test_phase7_records_selected_performance_and_renders_the_phase6_score(tmp_path) -> None:
    phase6_dir = _phase6_run(tmp_path)
    runner = SequencedRunner([_response()])
    run_dir = tmp_path / "phase7"

    result = realize_phase7(Phase7Request(phase6_dir), runner, run_dir)

    assert result.realized is True
    assert result.outcome == "complete"
    assert len(runner.prompts) == 1
    assert "演奏選択語彙" in runner.prompts[0]
    performance = json.loads(
        (run_dir / "outputs" / "performance-spec.json").read_text(encoding="utf-8")
    )
    assert performance["section_performances"] == [
        {
            "section_id": "statement",
            "timing_profile": "savor",
            "timing_amount": "subtle",
            "dynamics_profile": "shape",
            "articulation_profile": None,
            "coordination_profile": None,
            "pedal_profile": None,
        }
    ]
    rendered = json.loads(
        (run_dir / "outputs" / "rendered-performance.json").read_text(encoding="utf-8")
    )
    assert rendered["notes"]
    ledger = json.loads(
        (run_dir / "outputs" / "projection-ledger.json").read_text(encoding="utf-8")
    )
    assert ledger[-1]["source_id"] == "statement-expression"
    assert ledger[-1]["status"] == "unverified"
