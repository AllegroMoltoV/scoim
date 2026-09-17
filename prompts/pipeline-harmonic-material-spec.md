あなたは独奏ピアノ曲の対象素材について、局所和声だけを設計します。

固定情報は次のとおりです。対象素材の旧和声、旧旋律、旧伴奏は含まれていません。

{{FIXED_CONTEXT_JSON}}

`composition_source`には次の形だけを出力してください。説明文とMarkdownコードフェンスは不要です。

```text
harmonic_draft(events=[
    harmonic_event(at_units=0, duration_units=8, root_pitch_class=0, quality="major"),
    harmonic_event(at_units=8, duration_units=8, root_pitch_class=7, quality="major"),
])
```

- 先頭を0とし、隙間と重複なく固定material長の末尾まで被覆してください。
- eventは2個以上8個以下です。
- `duration_units`は正整数、`root_pitch_class`は0から11です。
- `quality`は`major`、`minor`、`diminished`、`major-seventh`のいずれかです。
- ID、material ID、note、voice、pitch、奏法、演奏指定は出力しません。
