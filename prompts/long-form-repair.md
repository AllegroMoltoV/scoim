あなたは、確定済みの三分ピアノ曲にある素材 {{material_id}} だけを修正します。全体構成と他の素材は出力できません。

現在の素材は次のとおりです。

```python
{{current_material}}
```

機械評価と必須の修正結果は次のとおりです。

{{evaluation}}

素材 ID と duration_ms は変更しません。既存のモチーフ、音価の多様性、ペダルの踏み替えを保ちながら、宣言した最高潮が発音密度、ベロシティ中央値、平均同時発音数のうち少なくとも二つで全大区分の単独最大になるようにしてください。単に全音を同時刻へ集めたり、同じ音型を機械的に埋めたりしないでください。

出力する制限付き記法は、次の形だけです。

```python
material_batch(
    "repair-climax",
    materials=[
        material("{{material_id}}", duration_ms={{duration_ms}}, notes=[...], pedals=[...]),
    ],
)
```

JSON Schema に従い、composition_source と短い intent_summary だけを返してください。
