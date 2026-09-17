あなたは独奏ピアノ曲の対象素材について、前景旋律だけを設計します。

固定情報と確定済み和声は次のとおりです。対象素材の旧旋律と旧伴奏は含まれていません。

{{STAGE_CONTEXT_JSON}}

`composition_source`には次の形だけを出力してください。説明文とMarkdownコードフェンスは不要です。

```text
melody_draft(foreground_voice="upper", events=[
    melody_event(at_units=0, duration_units=4, pitch=60, articulations=[]),
])
```

- `foreground_voice`は`upper`または`lower`です。
- eventは固定material内に収め、各和声区間に最低1発音を置いてください。
- 異なる発音位置を最低4箇所、異なるpitchを最低3種類使ってください。
- event数は固定情報にある上限以下です。
- 同じpitchの音価を重ねません。
- 和声外音は、同一和声内の短い経過音または刺繍音として理由を持つ場合だけ使ってください。0個でも構いません。
- ID、material ID、role、eventごとのvoice、伴奏、velocity、pedalは出力しません。
