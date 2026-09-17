from __future__ import annotations

import json
import random

from llm_musical_composer.listening_evaluation import create_listening_package


def test_listening_package_is_blind_and_has_no_total_score(tmp_path) -> None:
    before = tmp_path / "before.mid"
    after = tmp_path / "after.mid"
    before.write_bytes(b"before")
    after.write_bytes(b"after")

    result = create_listening_package(
        before,
        after,
        tmp_path / "listen",
        target_axis="relative_timing",
        target_material="A",
        rng=random.Random(7),
    )

    form = json.loads(result.evaluation_form.read_text(encoding="utf-8"))
    mapping = json.loads(result.blind_map.read_text(encoding="utf-8"))
    assert {item["id"] for item in form["blind_axes"]} == {1, 4, 5, 6, 7, 8, 9, 10}
    assert {item["id"] for item in form["revealed_axes"]} == {2, 3}
    assert form["allowed_values"] == [1, 2, 3, 4, 5, "判定不能", "該当なし"]
    assert form["general_scale"]["3"] == "許容できるが改善が必要"
    assert form["revealed_scale"]["2"]["5"] == "ほぼ解消した"
    assert "total_score" not in form
    assert set(mapping["samples"]) == {"sample-X.mid", "sample-Y.mid"}
    assert {path.name for path in result.samples} == {"sample-X.mid", "sample-Y.mid"}
