あなたは、確定済みの3分ピアノ曲について、全曲の`はげしさ={{control_value}}`を作るための短い音符差分を生成します。全体構成、題名、調性、終止、素材全体は出力できません。

今回のバッチIDと素材仕様は次のとおりです。

```text
{{batch_id}}
```

```json
{{material_specs}}
```

各仕様の`current_material`は、この値の直接の比較元です。旋律の輪郭、下声の伴奏上の役割、和声語彙、素材境界を聞き取れる形で残してください。`source_material`がある場合は、`derived_from`で示した変奏関係も残してください。

各`target`について、次を守ってください。

- `target_counts.note_count`は音符数である。
- `target_counts.attack_count`は異なる発音時刻の数である。
- `targets.velocity_level`は1から127のvelocityを8区分して求める0から1の水準である。
- `holds.mean_active_polyphony`は同時発音数の許容範囲である。
- 音符数だけを和音で増やす方法、発音群数だけを短い同音連打で増やす方法を使わない。
- 素材間の活動量順位を平坦化しない。`usages`ではなく`current_material`の密度と強弱の差を保つ。

各`tonal_context`について、次を守ってください。

- `allowed_pitch_classes`以外のpitch classを追加しない。
- `minimum_pitch`から`maximum_pitch`の音域を越えない。
- 新しい転調や不協和な同時音程を追加しない。

全素材で次を固定してください。これらは差分記法から変更できません。

- material_id、duration_ms、derived_from、pedals
- 上声と下声の両方
- フレーズ末尾の余韻とペダル解放位置
- 素材固有の旋律と伴奏の役割

上声の発音時刻は変更しないでください。下声に新しい発音時刻を作る場合は、既存発音から100ms以上離し、聞き取れる伴奏リズムとして配置してください。数msから50msの一律の人間化を行わないでください。全発音を上下声で同期させず、独立した動きと節目での合流を両方残してください。全音符だけ、同じ音価だけ、単旋律だけ、過密和音だけの素材へ退化させないでください。

出力は、指定されたバッチIDと全素材の差分をちょうど一度ずつ含む次の形だけです。

```python
material_patch_batch(
    "{{batch_id}}",
    patches=[
        material_patch(
            "A",
            remove_ids=["A-n003"],
            edits=[note_edit("A-n004", at_ms=800, duration_ms=350)],
            additions=[
                note_clone(
                    "A-n002",
                    "A-intensity-001",
                    at_ms=1200,
                    duration_ms=300,
                    pitch=60,
                ),
            ],
            velocity_offset=-4,
        ),
    ],
)
```

`remove_ids`は削除する既存音符IDです。`note_edit`は既存音符の発音時刻と音価だけを置き換えます。`note_clone`は既存音符のvelocityとvoiceを引き継ぎ、指定したID、発音時刻、音価、pitchで新しい音符を作ります。`velocity_offset`は残した音符と追加音符すべてへ加える整数です。適用後の音符数と異なる発音時刻数を`target_counts`へ正確に一致させてください。

素材全体を再出力しないでください。`material`、`note`、`pedal`、ループ、変数、計算式、内包表記、composition、part、phrase、use、tonic_holdは禁止です。

JSON Schemaに従い、composition_sourceと短いintent_summaryだけを返してください。
