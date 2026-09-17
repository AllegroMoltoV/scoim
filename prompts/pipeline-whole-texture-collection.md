# 曲全体のピアノ伴奏生成

確定済みの曲構成、和声、前景旋律、素材ごとの配置助言に対して、入力された現在のバッチに含まれる素材だけの伴奏を生成してください。バッチが1つだけなら、それが全素材です。

## 入力

{{STAGE_CONTEXT_JSON}}

## 出力

`material_order`の指定順と同じ件数の素材を、次の制限付きDSLだけで`composition_source`へ返してください。別バッチの素材は返しません。

```python
texture_collection(materials=[
    texture_draft(events=[
        texture_event(harmony_index=0, at_units=0, duration_units=6, degree='root', register_zone='bass', articulations=[]),
        texture_event(harmony_index=0, at_units=0, duration_units=6, degree='fifth', register_zone='low', articulations=[]),
    ]),
])
```

- 各素材は`minimum_event_count`〜`maximum_event_count`イベントです。両者が同じ場合は、その件数を厳密に守ります。
- `harmony_index`と発音位置は入力の和声区間に対応させます。
- `degree`は`root`、`third`、`fifth`、`seventh`です。和音にないdegreeを使いません。
- `texture_feasibility.harmonies`の`whole_harmony_universal_safety_advisory`は、和声区間のどこで使っても安全な保守的候補です。原則としてこの候補を優先しますが、個別eventの必須条件や共同配置の保証ではありません。
- `texture_feasibility.onset_capacities`はhard必要条件です。同じ`at_units`で新しく打鍵する伴奏event数を、対応区間の`maximum_new_accompaniment_attack_count`以下にしてください。前の位置から保持中の伴奏音はこの数へ含めませんが、別の配置検査を受けます。上限以下でも、選んだdegree、zone、保持音の組合せによっては配置不能です。
- `minimum_new_accompaniment_attacks`は和声開始位置ごとのhard必要条件です。値が1の位置には伴奏eventを最低1つ置きます。値が0の位置は、確定旋律が新しい和声を示すため、伴奏を同時に打鍵せず後から入れてかまいません。終止素材の最低2音という条件は、この値より優先します。
- `register_zone`は、和声音、前景との上下関係、低域間隔、`allowed_pitch_range`を守れる値にします。助言表にないzoneでも、event自身の発音区間でこれらを満たす候補音がある場合は使用できます。
- 前景がupperなら、左手らしい根音・五度・オクターブまたはアルペジオを中心にします。低域では三度の密集を避け、根音・五度・オクターブを優先します。
- すべての素材で、同じ発音位置に2イベント以上を置く和音打鍵を少なくとも1回作るか、同じ和声区間内に3つ以上の異なる発音位置と2種類以上のdegreeを持つアルペジオを作ります。
- `semantic_targets`の`attack_texture`は全素材を合わせた30 ms発音群の配分です。現在のバッチ内でも、`per_25_groups`を各素材のノルマにせず、3音以上の打鍵を展開素材や頂点へ多めに配分します。transitionとreleaseは相対的に薄くできます。
- `register_enforcement`を先に確認します。`normal_allowed_pitch_range`は通常の配置範囲、`placement_allowed_pitch_range`はこのrunで実際に強制する配置境界です。範囲外の配置を求めず、低域の完全5度、オクターブ、間隔品質を優先します。
- `register_span_goal_role`が`diagnostic_only`の場合、`register_envelope`の音域幅は診断値であり、`placement_allowed_pitch_range`は実験上の配置許容境界です。境界の下端から上端までを使い切る指示ではなく、使用を目指す音域として扱いません。
- `texture_budget`は、確定旋律と伴奏を結合した後の整数予算です。各固有素材はDSLへ一度だけ返し、再利用箇所を重複生成しません。全曲予算は、各素材の予算にその`occurrence_count`を掛けた合計です。素材別予算を単純合計した値とは一致しなくてよく、矛盾ではありません。全曲と各素材の`combined_attack_group_count`、`combined_note_event_count`、`attack_size_counts`、`required_texture_event_count`を厳密に守ります。発音群を減らして厚みを合わせたり、発音群を増やして発音頻度だけを合わせたりしません。
- `attack_size_counts`の`one`、`two`、`three`、`four_or_more`は、確定旋律を含む同じ`at_units`の総音数です。伴奏eventだけの音数ではありません。
- 3音以上の打鍵を増やすときは、確定旋律の発音位置へ伴奏の2音以上を重ねる形を含めます。伴奏だけの和音を増やして達成せず、句頭、和声転換、強調点、終止で両手を合流させます。
- 合流していない位置では伴奏の独立した動きを残します。全伴奏を旋律と同じリズムへそろえません。
- `key_held_texture`はnote-onからnote-offまでの鍵盤保持の厚みです。ペダルや発音頻度の追加で達成したことにしません。
- 同時発音だけでなく時間方向の動きも作り、参照発音頻度の全曲目標へ寄与させます。ただし単純な同音連打で数値だけを増やしません。
- transitionとreleaseは疎くてよく、climaxと展開素材は相対的に活動量を増やせます。
- `is_release`がtrueの素材では、`required_final_attack_units`に最後の同時打鍵を置き、`required_final_pitch_classes`に対応する`root`と`fifth`を含む最低`minimum_required_final_new_accompaniment_attack_count`音を置きます。これは厳密な音数ではないため、素材予算が許せば`third`などを追加できます。各音を`required_final_duration_units`だけ保持して素材末尾へ合わせます。
- `required_final_pitch_assignment.pitches`は、確定旋律と実効音域から検証済みの根音・完全五度の実配置例です。DSLでは実音高を直接指定できないため、`suggested_register_zones`を第一候補にします。zone自体は推奨であり、配置器が別zoneへ投影しても、根音・完全五度、音域、声部順序、低域間隔を満たせば許容されます。
- `is_release`がtrueの素材では、`required_final_attack_units`より前に始まる全イベントをその位置までに終了させます。直前のアルペジオや和音を共同終止打鍵後まで持ち越しません。
