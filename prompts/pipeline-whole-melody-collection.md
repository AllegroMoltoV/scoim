# 曲全体の前景旋律生成

確定済みの曲構成、素材長、和声と、匿名の参照目標に対して、指定素材の前景旋律を一括生成してください。伴奏は後段で生成します。

## 入力

{{STAGE_CONTEXT_JSON}}

## 出力

指定順と同じ件数の素材を、次の制限付きDSLだけで`composition_source`へ返してください。

```python
melody_collection(materials=[
    melody_draft(
        foreground_voice='upper',
        events=[
            melody_event(at_units=0, duration_units=6, pitch=62, articulations=['tenuto']),
            melody_event(at_units=6, duration_units=6, pitch=64, articulations=[]),
        ],
    ),
])
```

- 各素材は4〜`maximum_event_count`イベントです。4つ以上の異なる発音位置と3音高以上を使います。
- 音高は21〜108、音価は正整数で、素材の範囲内に置きます。
- `foreground_voice`は`upper`または`lower`です。前景がupperなら後段の伴奏を下へ、lowerなら上へ配置します。
- 主な音は和音構成音にします。非和声音は、同じ和声区間内の経過音または隣接音だけに限定します。直前と直後を和音構成音とし、非和声音への導入と離脱の両方を半音または全音にしてください。
- transition素材では入力の前後アンカーへ5半音以内で接続し、内部跳躍も5半音以内、4発音以上にします。末尾を次素材の初音と同音で打ち直さないでください。
- `is_release`がtrueの素材では、最後の旋律音を`required_final_pitch_classes`の先頭にある主音とし、`required_final_attack_units`から`required_final_duration_units`だけ保持して素材末尾へ合わせます。
- 派生素材は初頭輪郭またはリズム骨格を残しつつ完全コピーを避けます。
- 音数だけで区分差や発音頻度を作らず、リズム、音価、輪郭、休止にも変化を持たせます。
- `register_envelope`は全曲の音域幅です。各素材で上下端を使い切らず、少数の孤立した極端音だけで幅を作らないでください。`allowed_pitch_range`がある場合、新しい音はその範囲内に置きます。
- `semantic_targets`に`tonal_hierarchy`がある場合、音階は`piece.tonal_center`からの`scale_degrees`を使ってください。主な音は宣言和音に従い、音階内であっても遠い音を理由なく挿入しないでください。非和声音は前後関係を持つ経過音または隣接音に限定し、調性規則を区分ごとに切り替えないでください。
