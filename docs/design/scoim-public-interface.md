# SCoIM初版の公開インターフェース

初版の主な入口は`scoim`コマンドと`scoim`Pythonパッケージである。利用者は短い依頼から楽曲台本を作り、承認後に構成と実現を進める。内部フェーズ、モデル転送形式、`llm_musical_composer`配下の実装は公開互換性の対象にしない。

| 項目 | 内容 |
| --- | --- |
| 文書の役割 | 初版で利用者が依存できるCLI、Python API、配布物の正本 |
| 設計版 | 1.0 |
| 最終更新日 | 2026-09-17 |
| 対象範囲 | `scoim`コマンド、`scoim`ルートのPython名、固定例、互換性 |

## 結論

初版の主な入口は`scoim`コマンドと[楽曲台本](glossary.md#楽曲台本)である。利用者は短い依頼から楽曲台本を作り、必要なら修正し、承認後に楽譜と演奏を生成する。機械用の楽曲設計データを直接編集または承認しない。楽曲台本のデータ契約は[楽曲台本契約](scoim-flow.md)に記載する。

公開する六つの操作名は維持する。`propose`から`approve`までは楽曲台本を扱う。生成工程と呼び出し回数は[生成工程](scoim-generation-workflow.md)を正本とする。

## CLI

公開する操作は次の6個に限定する。

| 操作 | 入力と結果 |
| --- | --- |
| `propose` | 短い自然言語指示からdraftの楽曲台本と提案記録を作る |
| `check` | 楽曲台本のSchemaと意味を検査する |
| `show` | 場面IDを使って一場面と直接関係を表示する |
| `apply` | JSON Patchで新しいdraft revisionを作る |
| `approve` | 内容hashを固定したapprovedの楽曲台本を作る |
| `realize` | 承認済み楽曲台本、構成bundle、生成試行bundleのいずれかから、定義済みの処理を開始する |

上流5操作は次の形とする。

```text
scoim propose INSTRUCTION --document-id ID --instrumentation NAME
              --duration SECONDS --model MODEL --output OUTPUT
scoim check FLOW
scoim show FLOW SCENE_ID
scoim apply FLOW PATCH --output OUTPUT [--new-draft]
scoim approve FLOW --output OUTPUT
```

`propose`の`OUTPUT`は新しい提案記録directoryであり、楽曲台本draftは`OUTPUT/outputs/flow-draft.json`へ保存する。`check`は楽曲台本だけを検査する。`show`は指定場面、その演奏順、直前の場面、直後の場面を返す。`apply`と`approve`は、入力と出力が同じpathなら保存済みrevisionとの一致を確認して置換し、別pathなら既存の別文書を上書きしない。

`realize`は入力のSchema版と種類を明示的に判定する。未知の種類を拒否し、同じ入力を状態から推測して別の意味へ分岐させない。

1. 承認済み楽曲台本: 新しい構成bundleを作り、その検証済み楽曲設計データから実現試行を作る。
2. 既存の構成bundle: 検証済み楽曲設計データを固定し、実現試行だけを作る。
3. 既存の生成試行bundle: 外部通信を行わず、保存済み入力から成果物を再生する。

公開CLIは次の形とする。

```text
scoim realize INPUT --output OUTPUT [--model MODEL] [--trial-id ID]
              [--composition-id ID] [--profile PROFILE]
```

`INPUT`が承認済み楽曲台本または構成bundleの場合は、`--model`と`--trial-id`を必須とする。`--profile`を省略した新規生成では`solo_piano_3m_v2`を使う。`solo_piano_3m_v1`は明示指定できる互換経路として残す。承認済み楽曲台本では、`--composition-id`があればその値、なければ楽曲台本の`document_id`を構成IDに使う。

生成試行bundleの再生では、bundleのmanifestから版を判定する。モデル、試行ID、構成ID、生成profileを受け取らず、外部通信を行わない。したがって、CLIの既定profileが変わっても、既知のv1生成試行bundleをprofile指定なしで再生できる。旧`--frozen-response`と`--run-dir`は公開CLIへ戻さない。

`OUTPUT`は一回の公開処理の出力ルートである。承認済み楽曲台本からv2生成を始める場合は`public-run.json`、`composition`、`realization-work/phase2`から`phase7`、`trial`を作る。v2構成bundleから始める場合は`public-run.json`、`realization-work/phase3`から`phase7`、`trial`を作る。生成試行bundleを再生する場合は`artifacts`を作る。失敗時も、開始済みのモデル要求、応答、検査、確定済みの上流bundleを残す。

新しいv2生成で既存`OUTPUT`を受け入れるのは、`public-run.json`があり、入力内容hash、入力種類、生成profile、構成ID、試行ID、[モデル実行条件](glossary.md#モデル実行条件)が今回の要求と一致する場合だけである。検証済みの完成工程を再利用し、最初の未完成工程から再開する。識別記録がない、要求が異なる、保存物が改変されている、モデル呼び出しの終了状態が不明である、のいずれかでは続行しない。v1生成とbundle再生の出力先には既存directoryを指定できない。

v2の新規`OUTPUT`は、一時的な兄弟directoryへ`public-run.json`を書いて検査し、directory単位で確定配置してからモデルを呼ぶ。`OUTPUT`だけを作成して識別記録がない状態を、通常の中断結果として残さない。

公開v2生成に渡すrunnerは、モデル呼び出し前に[モデル実行条件](glossary.md#モデル実行条件)を提示できなければならない。各モデル応答でrunnerが記録した情報をその条件と照合し、不一致の応答を採用しない。終了状態など呼出し後に確定する事実は各attemptへ記録するが、事前条件として扱わない。

モデル生成と通信なし再生は、成功可否、判定した入力種類、構成bundle、実現試行bundle、再生先、成果物の相対path、型付き問題を同じ結果型で返す。相対pathの基準は`OUTPUT`とする。

`realize`は、最終SMFに加えて、生成できたフェーズ4と6の診断用SMFの相対pathを結果へ含める。診断用SMFを作るための公開操作は増やさない。通常は最終SMFだけを提示し、利用者が問題の原因を調べる場合に限って診断用SMFを案内する。

新しい構成bundleと生成試行bundleは、manifestへbundle種類、Schema版、対象profileを明記する。v2構成bundleは、承認済み楽曲台本、`script-0.4.0`の検証済み楽曲設計データ、投影台帳、フェーズ2記録を持つ。v2生成試行bundleは、試行ID、構成ID、検証済み楽曲設計データ、元の構成manifestのbytesとそのSHA-256を持つ。生成時は元の構成bundle全体を検証し、通信なし再生時は複写した構成manifestのbytesからSHA-256を再計算する。元の構成bundleがない通信なし再生では、元bundle内の全ファイルまで再検証したとは扱わない。

既存の種類fieldを持たない生成試行bundleは、既知の旧Schema版としてだけ通信なし再生できる。未知の版をfield構成から推測して受け入れない。v1の型や保存物をv2として読み替えない。

## Python API

Python APIの公開操作は`realize`一つとする。承認済み楽曲台本、構成bundle、生成試行bundleを同じ関数へ渡し、入力の種類とSchema版に応じた処理を行う。構成、実現、通信なし再生を別の公開操作へ分けない。

`scoim`ルートで公開する名前は次の五つに限定する。

- `realize`: 三種類の入力を処理する公開操作
- `PublicRealizationResult`: `realize`の結果型
- `CodexStructuredRunner`: Codex CLIを使う新規生成のモデル実行手段
- `ValidationIssue`: 結果に含まれる型付き問題
- `IssueCode`: 型付き問題の分類

`compose_flow`、`create_model_trial`、`replay_trial_bundle`、`realize_solo_piano_3m`などは、旧境界または下位処理の実装として必要な範囲で残す。ただし、`scoim`ルートの公開互換性には含めない。独自Runner用の抽象と転送結果型も、初版の公開互換性には含めない。

`scoim`の下位モジュールと`llm_musical_composer`は実装詳細であり、互換性を保証しない。公開名は`src/scoim/__init__.py`の`__all__`と一致させる。

## 配布物

配布名、Pythonパッケージ名、CLI名は`scoim`とする。wheelとsdistには次を含める。

- `scoim`パッケージと公開Schema
- SCoIMが実行時に必要とする`llm_musical_composer`の内部実装
- README、LICENSE、CHANGELOG
- sdistには外部通信なしで実行できる公開固定例

wheelには実行時パッケージだけを含め、top-levelの固定例は含めない。旧実験用console script、テスト、実行ログ、公開前資料、生成済みSMF、利用者の参照曲は含めない。

`examples/fixed-flow/`には[適合fixture](glossary.md#適合fixture)を置く。

[公開音楽例](glossary.md#公開音楽例)は適合fixtureと同じ成果物へ統合しない。複数の生成結果が合格した場合は、最良の一例へ絞らず、由来と評価を分けて掲載できる。

公開音楽例はソースリポジトリの`examples/public-music/`へ置き、wheelとsdistには含めない。最終SMF、MusicXML、承認済み楽曲台本、生成時Gitコミット、bundleの識別情報、成果物hash、人間評価を含む来歴を添える。モデルの生ログ、認証情報、ローカル絶対pathは含めない。

第三者がSMF再生環境を用意せずに試聴できるよう、圧縮した音声プレビューを添えてよい。音声プレビューはSCoIMが生成した正本ではなく、最終SMFを別の再生環境で音声化した派生物として扱う。来歴には、元の音声ファイルと公開ファイルのhash、音声形式、変換条件を記録する。再生に使ったソフトウェアや音源を確認できない場合は推測で補わず、不明であることを明記する。容量の大きい無圧縮音声は公開リポジトリへ含めない。

## 互換性

初版はWindowsとPython 3.13で検証する。他OSは未検証であり、非対応とは断定しない。

最初の公開release以後は、次を互換対象とする。

- 楽曲台本Schemaの版と、承認済み文書を上書きしない移行
- 楽曲設計データとbundleの再生に必要な版
- 六つのCLI名、必須引数、JSON結果の意味
- `scoim`ルートの`realize`と、結果型、Codex用Runner、型付き問題に必要な四つの公開名
- 対応済み実現器版のbundle再生

`1.0.0`未満でも破壊変更はCHANGELOGへ記録する。内部IR、モデル転送Schema、段階数は互換対象にしない。

既存の`approved-script.json`を持つ旧bundleは、旧bundle版と対応済み実現器版で再生する。新しい構成bundleへ暗黙に移行しない。
