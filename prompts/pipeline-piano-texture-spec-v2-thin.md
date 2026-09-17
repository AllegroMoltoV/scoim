あなたは独奏ピアノ曲の対象素材について、確定済み和声と前景旋律を支える伴奏だけを設計します。

固定情報、確定済み和声、確定済み旋律は次のとおりです。対象素材の旧伴奏は含まれていません。

{{STAGE_CONTEXT_JSON}}

`composition_source`には次の形だけを出力してください。説明文とMarkdownコードフェンスは不要です。

```text
texture_draft(events=[
    texture_event(harmony_index=0, at_units=0, duration_units=8, degree="root", register_zone="low", articulations=[]),
])
```

- `harmony_index`は固定情報にある0始まりの和声順です。
- eventは参照和声区間内に収めます。`texture_feasibility.harmony_start_accompaniment.minimum_new_accompaniment_attacks`が1の開始位置には最低1発音を置きます。値が0の位置は、確定旋律が新しい和声を示すため、伴奏を同時に打鍵せず後から入れてかまいません。
- `degree`は`root`、`third`、`fifth`、`seventh`です。`seventh`は`major-seventh`だけで使えます。
- `register_zone`は`bass`、`low`、`middle`、`high`です。
- 固定情報の`texture_feasibility.harmonies`は、和声順とdegreeごとに`individually_feasible_register_zones`を示す保守的な助言です。この薄いパイロットでは、各eventの`register_zone`を、そのeventの`harmony_index`と`degree`に対応する一覧から選んでください。
- `texture_feasibility.onset_capacities`はhard必要条件です。同じ`at_units`で新しく打鍵する伴奏event数を、対応区間の`maximum_new_accompaniment_attack_count`以下にしてください。前から保持中の伴奏音はこの数へ含めませんが、別の配置検査を受けます。上限以下でも配置成功は保証されません。
- `accompaniment_voice`が`upper`なら、伴奏は重なる前景音より上です。`lower`なら前景音より下です。必要条件一覧は単一event用であり、同時発音どうしの重複や低域の濁りも避けてください。
- event数は固定情報にある上限以下です。
- 重音打鍵を1回以上、または一つの和声内で3発音以上・2度数以上の分散和音を作ってください。
- ID、material ID、harmony ID、role、voice、絶対pitch、octave、velocity、pedalは出力しません。
