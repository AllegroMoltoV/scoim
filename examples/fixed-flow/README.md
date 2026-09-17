# 公開経路を通信なしで検査する適合fixture

この例は、承認済みの楽曲台本と保存済みのモデル応答を使い、`solo_piano_3m_v1`互換経路を検査する[適合fixture](../../docs/design/glossary.md#適合fixture)である。

- `approved-flow.json`: 利用者が承認する楽曲台本
- `responses.json`: 楽曲設計データへの変換と演奏生成に使う固定応答

配布物検査では`scripts/smoke-installed-public.py`から読み込み、楽曲台本の検査と表示、3つのSMF生成、生成試行bundleの通信なし再生を確認する。公開既定の`solo_piano_3m_v2`は`examples/fixed-v2/`で別に検査する。
