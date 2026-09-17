# 設計書

`docs/design/`は、現在有効なSCoIM設計の正本である。最初に[用語集](glossary.md)と[決定状態](scoim-decision-status.md)を読み、次に必要な領域の文書だけを参照する。将来像は現在との差分だけを`docs/roadmap/`、実験結果と変更経緯はローカル専用の`docs/reports/`、実行手順は`.prompts/PLANS/`、現在の作業状態はBeadsへ分ける。

## 読む順序

### 1. 全体像

- [用語集](glossary.md): 利用者向けと実装向けの主要用語
- [決定状態](scoim-decision-status.md): 長期的に確定、初版では確定、仮置きの区別
- [SCoIMシステム](scoim-system.md): 目的、利用者の流れ、責務の境界
- [生成工程](scoim-generation-workflow.md): 人間、LLM、決定的処理のやり取りと失敗処理

### 2. SCoIM中核

- [楽曲台本](scoim-flow.md): 人間が読み、修正し、承認する構想
- [楽曲設計データ](scoim-script.md): 機械が扱う構成、マテリアル、マテリアル配置、変奏、遷移
- [公開インターフェース](scoim-public-interface.md): CLI、Python API、配布と互換性

### 3. 3分ピアノ実証

- [生成プロファイル](controllable-composition-system.md): 既定の`solo_piano_3m_v2`と互換経路`solo_piano_3m_v1`の境界、目的、成功条件
- [生成制御](generation-control-interface.md): `あかるさ`、`高さ`、`発音頻度`
- [ピアノ生成の全体像](performance-generation-pipeline.md): 構成から楽譜、演奏、SMFまでの入口
- [ピアノ楽譜](solo-piano-score.md): `PiecePlan`、`ScoreSpec`、MusicXML
- [ピアノ演奏](solo-piano-performance.md): 演奏選択語彙、`PerformanceSpec`、演奏イベント、SMF
- [ピアノ生成ランタイム](solo-piano-generation-runtime.md): LLM境界、作業状態、保存、再開
- [あかるさ制御](brightness-control-resolution.md)
- [高さ制御](height-control-resolution.md)
- [発音頻度制御](attack-frequency-control-resolution.md)
- [重なりの内部観測](overlap-control-resolution.md)

生成制御と四つの解決設計は現行`solo_piano_3m_v1`を対象とする。新しい対象関係を使う`solo_piano_3m_v2`には接続されていない。

### 4. 参照曲

- [参照曲を基準にした新規生成](reference-conditioned-baseline-generation.md): 参照特徴を生成目標へ渡す契約
- [参照SMFの逆分解と再合成](reference-smf-reverse-decomposition.md): 生SMFから構成仮説までを分ける凍結済みの再現性契約

`reference-smf-reverse-decomposition.md`は実行時の凍結manifestに含まれている。参照するmanifestを改版するまで、パスと内容を変更しない。

## 文書の責務

一つのルールは一つの文書だけが所有する。別の文書で必要になった場合は要約を繰り返さず、所有文書へリンクする。主要用語の意味は[用語集](glossary.md)だけが所有し、ほかの文書では定義を再掲しない。設計判断が長期的に確定しているか、初版だけで確定しているか、まだ仮置きかという分類は[決定状態](scoim-decision-status.md)だけが所有する。各領域の文書には現在のデータ契約や処理条件を記載し、その決定状態を再掲しない。

| 保存先 | 内容 |
|---|---|
| `docs/design/` | 現在有効な設計 |
| `docs/roadmap/` | 現行設計と将来像の差分。設計の正本ではない |
| `docs/reports/` | ローカル専用の調査、実験、過去の判断根拠。Git管理対象外 |
| `.prompts/PLANS/` | 承認済み設計を実行する手順 |
| `.prompts/DISCUSSIONS/` | 利用者判断が必要な選択肢と回答 |
| Beads | 担当、依存関係、ブロッカー、次の作業 |

## 更新規則

設計変更が必要な場合は、実装より先に該当する正本を更新する。現在状態の文書へ変更履歴や不採用案を積み重ねず、将来との差分はroadmap、経緯はローカルreport、Git履歴、Beadsへ移す。実装完了時は、設計、roadmap、テスト、実装、索引の整合を確認する。roadmapの項目を実装した場合は、必要な内容を正本へ移し、roadmapから削除する。

実装を始める前に毎回、該当するシーケンス図、責務を所有する設計書、承認済み計画の処理順、入力、出力、検査、失敗処理を照合する。主要用語の意味が[用語集](glossary.md)以外で再定義されていないことと、設計判断の状態が[決定状態](scoim-decision-status.md)以外で再宣言されていないことも確認する。不整合がある場合は実装へ進まず、正本または計画を先に修正する。
