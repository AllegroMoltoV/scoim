"""修正前後を匿名化した聴感評価用成果物を作る。"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

RUBRIC = (
    (1, "再生上の正常性", "音の鳴りっぱなし、突然の途切れ、不自然な無音、極端な音量がないか"),
    (2, "修正対象の改善", "指定された問題が修正前より改善したか"),
    (3, "修正対象外の保持", "変更する必要がなかった部分が壊れていないか"),
    (4, "全体構成と展開", "始まり、展開、終わり、反復と変化の釣り合いが自然か"),
    (5, "旋律", "音のつながり、方向性、まとまりが自然か"),
    (6, "和声", "音の重なり、緊張と解決、衝突が自然か"),
    (7, "リズムと時間感覚", "音価、間、流れ、生演奏らしいタイミングが自然か"),
    (8, "ピアノ表現", "音域、同時発音、強弱、ペダル、演奏可能性が自然か"),
    (9, "作風への適合", "参照曲群と関係する作風を感じ、特定曲の複製にはなっていないか"),
    (10, "総合的な音楽品質", "楽曲として成立し、もう一度聴きたいと思えるか"),
)


@dataclass(frozen=True)
class ListeningPackage:
    samples: tuple[Path, Path]
    evaluation_form: Path
    blind_map: Path


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_listening_package(
    before_path: Path,
    after_path: Path,
    output_dir: Path,
    *,
    target_axis: str,
    target_material: str,
    rng: random.Random | None = None,
) -> ListeningPackage:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random_source = rng or random.SystemRandom()
    labels = ["sample-X.mid", "sample-Y.mid"]
    random_source.shuffle(labels)
    source_by_role = {"before": Path(before_path), "after": Path(after_path)}
    role_by_label = {labels[0]: "before", labels[1]: "after"}
    samples: list[Path] = []
    for label in ("sample-X.mid", "sample-Y.mid"):
        destination = output_dir / label
        shutil.copyfile(source_by_role[role_by_label[label]], destination)
        samples.append(destination)

    items = [
        {"id": number, "name": name, "question": question} for number, name, question in RUBRIC
    ]
    form_path = output_dir / "evaluation-form.json"
    map_path = output_dir / "blind-map.json"
    _write_json(
        form_path,
        {
            "schema_version": 1,
            "allowed_values": [1, 2, 3, 4, 5, "判定不能", "該当なし"],
            "general_scale": {
                "1": "明確に問題がある",
                "2": "問題が目立つ",
                "3": "許容できるが改善が必要",
                "4": "良い",
                "5": "とても良い",
            },
            "revealed_scale": {
                "2": {
                    "1": "悪化した",
                    "3": "部分的に改善した",
                    "5": "ほぼ解消した",
                },
                "3": {
                    "1": "対象外の部分が大きく壊れた",
                    "3": "小さな不要変更がある",
                    "5": "意図した部分だけが変わった",
                },
                "note": "2 と 4 は両隣の中間とする",
            },
            "blind_axes": [item for item in items if item["id"] not in {2, 3}],
            "revealed_axes": [item for item in items if item["id"] in {2, 3}],
            "procedure": [
                "対応表を見ずに両方の試料について blind_axes を評価する",
                "対応表と修正対象を開示して revealed_axes を評価する",
                "各値を合算しない",
            ],
        },
    )
    _write_json(
        map_path,
        {
            "schema_version": 1,
            "samples": role_by_label,
            "target_axis": target_axis,
            "target_material": target_material,
        },
    )
    return ListeningPackage(tuple(samples), form_path, map_path)
