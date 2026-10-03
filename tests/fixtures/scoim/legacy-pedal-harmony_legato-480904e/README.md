# 旧和声ペダルの互換fixture

2026-10-02に、ペダル修正前の製品コードで生成した完成bundleである。固定応答を使い、外部モデルを呼んでいない。bundle版5、フェーズ7版3、演奏選択語彙版0.1.0を保存する。

冒頭区分の共有和声を24 unitsずつのC majorとG majorへ分け、30 unitsに休符後の打鍵を置いた。`harmony_legato`が和音変更後80 msで踏み直し、次の打鍵を待たない旧動作を記録する。変更前にMusicXMLとMIDIの再生成bytes一致を確認した。

`bundle.zip`のSHA-256:

```text
577a8ca2480d23ef8db9dbcbfefe0c184d9b0c9416d1037da3ae7ad8dfc63937
```

既存の固定応答用helperで構成・楽譜・演奏を生成した。新処理から版番号を付け替えた保存物ではない。旧版検証と再生のテストは`test_scoim_phase8_bundle.py`にある。
