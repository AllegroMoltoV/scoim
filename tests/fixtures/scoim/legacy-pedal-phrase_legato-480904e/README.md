# 旧フレーズペダルの互換fixture

2026-10-02に、ペダル修正前の製品コードで生成した完成bundleである。固定応答を使い、外部モデルを呼んでいない。bundle版5、フェーズ7版3、演奏選択語彙版0.1.0を保存する。

冒頭区分の共有和声を24 unitsずつのC majorとG majorへ分け、30 unitsに休符後の打鍵を置いた。`phrase_legato`が途中の和音変更を越えて保持する旧動作を記録する。変更前にMusicXMLとMIDIの再生成bytes一致を確認した。

`bundle.zip`のSHA-256:

```text
e67df37940722d3a44aa42aca8303f8c02d88008805362355e3aec48b669eefc
```

既存の固定応答用helperで構成・楽譜・演奏を生成した。新処理から版番号を付け替えた保存物ではない。旧版検証と再生のテストは`test_scoim_phase8_bundle.py`にある。
