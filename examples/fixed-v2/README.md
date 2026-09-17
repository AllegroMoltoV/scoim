# 既定 v2 経路の適合 fixture

この例は、公開既定の`solo_piano_3m_v2`を外部通信なしで検査するための[適合 fixture](../../docs/design/glossary.md#適合fixture)である。

- `approved-flow.json`: 一場面だけを持つ承認済み楽曲台本
- `responses.json`: 楽曲設計データ、共有和声、前景を作る最小の固定応答

配布物検査では、生成プロファイルを明示せずに既定の v2 経路を完走する。生成した MusicXML と SMF を再読込みし、生成試行 bundle の通信なし再生で成果物の SHA-256 が一致することも確認する。
