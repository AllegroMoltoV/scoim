# 確定伴奏リズムへの内容生成

発音位置と各位置の新規伴奏打鍵数は確定済みです。位置と件数を変えず、各音の音価、和音度数、音域帯、奏法だけを生成してください。

## 入力

{{STAGE_CONTEXT_JSON}}

## 出力

`fixed_rhythm.groups`と同じ順、同じgroup数、各groupの`new_accompaniment_attack_count`と同じevent数を、次の制限付きDSLだけで`composition_source`へ返してください。

```python
texture_content_draft(groups=[
    [
        texture_content_event(duration_units=6, degree='root', register_zone='bass', articulations=[]),
        texture_content_event(duration_units=6, degree='fifth', register_zone='low', articulations=[]),
    ],
])
```

- `duration_units`は正の整数とし、そのgroupの発音位置から現在の和声区間と素材末尾を越えません。
- `degree`は`root`、`third`、`fifth`、`seventh`だけです。
- `register_zone`は`bass`、`low`、`middle`、`high`だけです。
- 前景がupperなら、左手らしい根音・五度・オクターブまたはアルペジオを中心にします。
- 低域では第三音の密集を避け、根音・五度・オクターブを優先します。
- 旋律と独立した伴奏の動きを保ちつつ、和声転換や強調点では両手の合流を活用します。
- 単純な同音・同音量連打を前提にせず、音価と奏法に変化を付けます。
- `at_units`と`harmony_index`は出力しません。groupの追加、削除、並べ替えもしません。
