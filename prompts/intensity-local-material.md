あなたは、確定済みの3分ピアノ曲にある素材`{{material_id}}`の`はげしさ`だけを調整します。全体構成、ほかの素材、題名、調性、終止は出力できません。

操作値は`{{control_value}}`です。現在の素材は次のとおりです。この素材を比較の`0.0`とし、別の曲として作り直さないでください。

```python
{{current_material}}
```

目標は次のとおりです。

```json
{{target}}
```

`target_counts.note_count`は音符数、`target_counts.attack_count`は異なる発音時刻の数です。両方を満たし、和音へ音を足すだけ、または単音を細かく刻むだけで片方を攻略しないでください。`targets.velocity_level`に合わせて強弱も動かしてください。`holds.mean_active_polyphony`は同時発音数の許容範囲であり、厚い和音や単旋律への退化を防ぐ固定条件です。

和声と音域の固定条件は次のとおりです。

```json
{{tonal_context}}
```

次をすべて守ってください。

- 素材ID、duration_ms、derived_from、pedalsは一切変更しない。
- 許可済みpitch classと音域だけを使い、転調、新しい和音語彙、不協和音の追加を行わない。
- 元素材の上声の音程輪郭と、下声の伴奏上の役割を聞き取れる形で残す。
- すべてのnoteに`voice="upper"`または`voice="lower"`を付け、両声を含める。
- 上声と下声を全発音で同時に動かさず、独立した応答と節目での合流を両方残す。
- 上声だけを下声から数msから50ms遅らせる一律の人間化を行わない。
- 同じ短いセルの機械的な反復、全音符だけ、同じ音価だけ、過密和音へ退化させない。
- ペダルイベントを文字単位で同じ内容に保ち、素材末尾の余韻を切らない。

出力する制限付き記法は次の形だけです。

```python
material_batch(
    "{{batch_id}}",
    materials=[
        material("{{material_id}}", duration_ms=..., notes=[...], pedals=[...]),
    ],
)
```

音符は`note("X-n001", at_ms=0, duration_ms=400, pitch=60, velocity=80, voice="upper")`、ペダルは`pedal("X-p001", at_ms=0, value=127)`の完全なリテラルで書いてください。ループ、変数、計算式、内包表記、全体compositionは禁止です。

JSON Schemaに従い、composition_sourceと短いintent_summaryだけを返してください。
