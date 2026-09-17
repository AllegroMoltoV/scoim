あなたは、既存のピアノ楽譜について、音符の長さだけを匿名の参照分布へ近づけます。

次の制約付き Python 風 DSL だけを `composition_source` に返してください。

```python
score_duration_patch(edits=[
    duration_edit(event_id="既存のevent ID", duration_units=正整数),
])
```

規則は次のとおりです。

- 変更が必要な音符だけを列挙します。少なくとも一つは変更します。
- 入力に存在する `event_id` だけを一度ずつ使います。
- `release_fixed=true` の素材を変更しません。
- 音符を素材の `length_units` より後まで延ばしません。
- 発音位置、音高、声部、音符数、和声、奏法、素材長、IDを変更しません。
- 素材ごとの `occurrence_count` を掛けた曲全体の配分として目標へ近づけます。各素材へ同じ配分を強制しません。
- 同じ実音高の再打鍵、和声境界、素材境界を越える保持を避けます。
- 上下声の同じ実音高を同時に保持する新しい重複を作りません。
- `very_long` が大きい目標でも、それ自体を理由に拒否しません。ただし、不自然な再打鍵や和声をまたぐ保持を作りません。
- 説明文、Markdown、完全な ScoreSpec、コメントを `composition_source` に含めません。

匿名目標と素材情報:

```json
{{CONTEXT_JSON}}
```

基準 ScoreSpec:

```python
{{SCORE_SPEC}}
```
