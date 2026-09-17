# 伴奏リズム骨格の生成

確定済みの1素材について、伴奏の発音位置と各位置の新規打鍵数だけを生成してください。音価、音高、和音度数、音域、奏法はまだ決めません。

## 入力

{{STAGE_CONTEXT_JSON}}

## 出力

次の制限付きDSLだけを`composition_source`へ返してください。

```python
texture_rhythm_draft(groups=[
    texture_rhythm_group(at_units=0, new_accompaniment_attack_count=2),
])
```

- `at_units`は0以上`length_units`未満の整数で、昇順かつ重複なしにします。
- `texture_feasibility.harmony_start_accompaniment.minimum_new_accompaniment_attacks`が1の和声開始位置には、伴奏打鍵を最低1つ置きます。値が0の位置は、確定旋律が新しい和声を示すため、伴奏を同時に打鍵せず後から入れてかまいません。終止素材の最低2音という条件は、この値より優先します。
- 各位置の新規伴奏打鍵数は正の整数とし、対応する`onset_capacities`の上限以下にします。
- 確定旋律と結合した`texture_budget`の発音群数、総音数、`attack_size_counts`、伴奏event数をすべて厳密に満たします。
- 前景と常時同じ位置へそろえず、伴奏だけが動く位置も残します。
- 同じ和声内でも、単純な等間隔連打だけにしません。
- `harmony_index`、音価、degree、register zone、articulationは出力しません。
