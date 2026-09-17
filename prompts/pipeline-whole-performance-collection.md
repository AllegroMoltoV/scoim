# 曲全体の演奏指定生成

確定済みの曲構成と楽譜に対して、演奏順の各出現へ演奏表情を割り当ててください。同じ素材の再現でも、1回目と2回目を同じ演奏にしないでください。

## 入力

{{STAGE_CONTEXT_JSON}}

## 出力

指定順と同じ件数の出現を、次の制限付きDSLだけで`composition_source`へ返してください。

```python
performance_collection(occurrences=[
    performance_occurrence(
        timing_profile='savor',
        timing_amount='subtle',
        dynamics_profile='shape',
        articulation_profile='legato',
        coordination_profile='rolled',
        pedal_profile='harmony_legato',
    ),
])
```

- 値は入力に示した固定語彙または`None`だけを使います。
- 初回提示は必要に応じて溜めや小さなテンポの揺れを使い、再現はより流れる演奏にします。
- `semantic_targets`の`coordination_preservation`に従い、厚い発音群を保つ区間では`score`または`aligned`を使います。
- `velocity_shape`は全曲の強弱分布です。全区間を`steady`へ固定せず、提示、頂点、回帰、終止の役割に合わせて強弱形を分けます。既定velocityの数値はrunnerが後で校正します。
- `texture_budget`は楽譜段階で確定した発音群予算です。複数音の同時打鍵へ`rolled`を指定して30 ms群を分裂させないでください。`rolled`が必要な出現でも、厚い打鍵を保つことを優先します。
- `rolled`を全区間へ一律適用せず、崩しを意図する限定区間だけに使います。縦線をそろえる再現では`aligned`を使います。
- `key_held_texture`は鍵盤保持時間の目標です。ペダル保持率の目標と混同しません。
- ペダルは旋律のまとまりを保持し、和声または句の切れ目で踏み替える`harmony_legato`を基本にします。
- `harmony_count`が2以上の出現では、複数の和声をまたいで旧伴奏音を残す`phrase_legato`を使いません。`harmony_legato`または`none`を使います。
- transitionと区分境界でペダルを唐突に切らず、releaseでは終止長音を十分に残します。
- releaseでは`rolled`を使いません。`coordination_profile`は`score`または`aligned`、`pedal_profile`は`harmony_legato`、`articulation_profile`は`score`または`legato`にします。
