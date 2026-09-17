# SCoIM

SCoIM（Scripted Co-creation of Impromptu Music、スコイム）は、人間と大規模言語モデル（LLM）が楽曲の台本を共同編集し、楽譜と演奏データへ変換するための実験的なOSSです。

利用者は「明るい感じの曲」のような短い希望を伝えます。LLMは音符を一度に書くのではなく、構成、和声、旋律、伴奏、演奏表現を段階的に提案します。人間は音符をすべて直さず、上位の台本を確認して局所修正できます。

```text
自然言語の希望
  → LLMが楽曲台本を提案
  → 人間が確認・局所修正・承認
  → LLMと決定的処理が段階的に具体化
  → MusicXML、SMF、診断、再現用記録
```

SMFは楽曲そのものではなく、タイミング、強弱、ペダルを含む演奏結果として扱います。

## 現在の状態

`v0.2.0`は、一般公開に向けた最初の候補です。WindowsとPython 3.13で、3分程度のピアノ独奏を対象とする一連の処理を検証しています。`v0.1.0`は、非公開リポジトリ内で縦断モックを固定した開発到達点です。

現在の開発版では、公開CLIの既定生成プロファイルは`solo_piano_3m_v2`です。承認済み楽曲台本から、楽曲設計データ、和声、前景、伴奏、楽譜、演奏を順に作り、最終MusicXMLとSMFを一つの生成試行bundleへ保存します。`solo_piano_3m_v1`は、`--profile solo_piano_3m_v1`を明示した場合に使える互換経路として残しています。

現在できることは次のとおりです。

- 自然言語から、場面の流れを読める楽曲台本のdraftを提案する
- JSON Patchで一部分だけ変更し、内容hash付きの台本として承認する
- 承認済み台本から、検証済みの楽曲設計データと構成bundleを作る
- 既存の段階生成資産で、構成、和声、旋律、伴奏、演奏表現を具体化する
- 保存済みbundleから、外部通信なしで同じMusicXMLとSMFを再生してhashを照合する
- 失敗を成功として隠さず、原因と途中までの記録を残す

短い依頼から最終SMFまでを公開CLIで通せます。`v2`は各モデル応答を検査し、内容だけに問題がある場合は同じ工程を追加1回まで修正します。中断後は、同じ入力、ID、モデル実行条件であれば検査済みの工程を再利用します。ピアノ独奏以外の編成、GUI、高い作風再現性、任意のLLMプロバイダーには対応していません。生成品質は研究途中であり、生成物が完成作品になることを保証しません。

## 必要な環境

- Windows
- Python 3.13
- モデル生成を使う場合は、インストール済みで認証済みのCodex CLI

リポジトリから開発環境を作る場合は、次を実行します。

```powershell
.\scripts\setup.ps1
```

## 短い依頼から演奏を作る

最初に、短い依頼から楽曲台本のdraftを作ります。

```powershell
scoim propose "明るく始まり、途中で広がって穏やかに戻るピアノ曲" `
  --document-id bright-piano `
  --instrumentation solo_piano `
  --duration 180 `
  --model gpt-5.6-sol `
  --output .tmp/bright-piano-proposal
```

draftは`.tmp/bright-piano-proposal/outputs/flow-draft.json`です。全体検査と一場面の前後確認は次のように行います。

```powershell
scoim check .tmp/bright-piano-proposal/outputs/flow-draft.json
scoim show .tmp/bright-piano-proposal/outputs/flow-draft.json scene-002
```

修正する場合は、revisionを1増やすJSON Patchを用意して`apply`します。修正しない場合はdraftをそのまま`approve`できます。

```powershell
scoim apply .tmp/bright-piano-proposal/outputs/flow-draft.json revision-2.patch.json `
  --output .tmp/bright-piano-draft-2.json
scoim approve .tmp/bright-piano-draft-2.json `
  --output .tmp/bright-piano-approved.json
```

承認済み楽曲台本から、構成bundle、生成試行bundle、診断用SMF、最終SMFを一括生成します。

```powershell
scoim realize .tmp/bright-piano-approved.json `
  --trial-id bright-piano-001 `
  --model gpt-5.6-sol `
  --output .tmp/bright-piano-output
```

最終SMFは`.tmp/bright-piano-output/trial/artifacts/final.mid`です。途中の前景と楽譜は、`.tmp/bright-piano-output/realization-work/phase4/outputs/foreground-preview.mid`と`.tmp/bright-piano-output/realization-work/phase6/outputs/score-preview.mid`で確認できます。

生成済みの実現試行bundleは、モデルを呼び出さずに再生できます。

```powershell
scoim realize .tmp/bright-piano-output/trial `
  --output .tmp/bright-piano-replay
```

通信なしの[適合fixture](docs/design/glossary.md#適合fixture)は`examples/fixed-flow/`にあります。生成した約3分の曲のSMFとMusicXMLは[公開音楽例](examples/public-music/bright-return/README.md)で確認できます。実装済みのコマンドと現在地は[SCoIM初版の公開インターフェース](docs/design/scoim-public-interface.md)と[SCoIM生成工程](docs/design/scoim-generation-workflow.md)を参照してください。

## 公開インターフェース

CLIは次の6操作です。

- `propose`: 自然言語からdraft台本を作る
- `check`: 台本を検証する
- `show`: 場面IDを使って、その場面と直前・直後の場面を表示する
- `apply`: JSON Patchで新しいdraft revisionを作る
- `approve`: 内容hash付きのapproved台本を作る
- `realize`: 承認済み楽曲台本または構成bundleから演奏を作り、生成試行bundleを通信なしで再生する

Python APIでは、`scoim`パッケージのルートから公開される名前だけを互換対象とします。`scoim`の下位モジュールと`llm_musical_composer`は実装詳細です。

契約の詳細は[SCoIM初版の公開インターフェース](docs/design/scoim-public-interface.md)と[SCoIM楽曲台本契約](docs/design/scoim-flow.md)を参照してください。設計全体は[設計書索引](docs/design/README.md)、確定事項と仮置きは[決定状態](docs/design/scoim-decision-status.md)にまとめています。

## 開発と配布検査

公開機能と現在の実行時依存を守る標準テスト、Ruffによる静的検査と整形検査、公開文書のローカルリンク検査を実行します。過去の研究や実験に固有のテストは、この標準テストへ含めません。

```powershell
.\scripts\test.ps1
```

branch coverageを確認する場合は`.\scripts\test.ps1 -Coverage`を実行します。coverage率だけでは合否を決めません。

配布検査では、sdistからwheelを作り直し、新規Python 3.13環境へ取得済み依存だけで導入します。その環境でCLI、固定例、固定Runnerによる段階生成を確認します。

```powershell
.\scripts\test-package.ps1
```

## リポジトリ構成

- `src/scoim/`: 公開中核とCLI
- `src/llm_musical_composer/`: ピアノ実証で使う内部生成資産
- `examples/`: [適合fixture](docs/design/glossary.md#適合fixture)と[公開音楽例](docs/design/glossary.md#公開音楽例)
- `docs/design/`: 現在有効な設計の正本
- `docs/roadmap/`: 現行設計と将来像の差分
- `tests/`: 挙動検査

`.appendix`、`.logs`、`.tmp`、`.prompts`、`docs/reports`はローカル作業用であり、Git管理と配布物には含めません。利用者に必要な変更履歴は`CHANGELOG.md`とGitHub Releaseへ記録します。

## コントリビューションとライセンス

開発への参加方法は[CONTRIBUTING.md](CONTRIBUTING.md)、利用者から見える変更は[CHANGELOG.md](CHANGELOG.md)に記録しています。

本プロジェクトは[MIT License](LICENSE)で公開します。`v0.1.0`は非公開リポジトリ内の開発到達点であり、一般公開は`v0.2.0`から始めます。
