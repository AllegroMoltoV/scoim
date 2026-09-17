# 曲全体の和声素材生成

次の固定された曲構成と匿名の参照目標に対して、全素材の長さと和声列を一括生成してください。既存曲の音符や和声を再現するのではなく、曲全体で一貫した新しい素材を作ります。

## 入力

{{STAGE_CONTEXT_JSON}}

## 出力

`piece.material_count`で指定された件数の素材を、`material_order`の順に一つずつ、次の制限付きDSLだけで`composition_source`へ返してください。たとえば`material_count`が8なら、必ず8素材を返してください。`piece.occurrence_count`は素材の再利用を展開した回数であり、11回の出現ごとではありません。説明をDSL内へ入れないでください。

```python
harmonic_collection(materials=[
    harmonic_material(
        length_units=48,
        events=[
            harmonic_event(at_units=0, duration_units=24, root_pitch_class=2, quality='major'),
            harmonic_event(at_units=24, duration_units=24, root_pitch_class=7, quality='major'),
        ],
    ),
])
```

- `length_units`は正整数です。12 unitsを基準の一まとまりとして考えます。
- 通常素材は2〜8和音、終止素材だけは1和音です。
- 和声は素材の0から始め、隙間や重複なく`length_units`を埋めます。
- `root_pitch_class`は0〜11です。
- `quality`は`major`、`minor`、`diminished`、`major-seventh`のいずれかです。他の表記や和音種は使いません。
- 終止素材は入力で指定された主音根音とmodeの単一和音にします。
- 低域伴奏で濁りやすい和声を連続させず、全区間を複雑な和音だけで埋めないでください。
- 発音頻度は和声数だけで作らず、後段の旋律と伴奏が十分に動ける素材長を確保してください。
- `semantic_targets`に`tonal_hierarchy`がある場合、すべての根音を`piece.tonal_center`からの相対度数として解釈し、`scale_degrees`と`optional_scale_degrees`の範囲で曲全体を一貫させてください。`characteristic_harmony`がある場合は、指定されたstatement、contrast、returnの各役割に必要和音を少なくとも指定回数置きます。終止素材は`ending`の相対根音とqualityへ一致させ、区分ごとにmajorとminorを交互配置して数値だけを満たさないでください。
