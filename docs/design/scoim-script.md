# SCoIM楽曲設計データ契約

この文書は、承認済みの[楽曲台本](glossary.md#楽曲台本)から作る[楽曲設計データ](glossary.md#楽曲設計データ)のデータ契約を定める。一般利用者は直接編集または承認せず、検査に合格した不変なJSON文書として保存する。元の楽曲台本は同じ文書へ埋め込まず、`source_flow`で参照する。形式を含む設計判断の状態は[決定状態](scoim-decision-status.md)を参照する。

## 役割

楽曲設計データは音符列を持たず、次を表す。

- 曲の区分と包含関係
- 再利用するマテリアル
- マテリアルをどの区分へ置くか
- 先行要素を何として保ち、何を変えるか
- 前後の区分を遷移する専用マテリアル
- 一つの編成と希望演奏時間
- 区分またはマテリアル配置に対する演奏指示

人間が承認する音楽意図の正本は[楽曲台本](scoim-flow.md)である。楽曲設計データは`status = validated`の不変な派生物であり、同じ台本から作り直す場合は新しい構成試行と`composition_id`を使う。

## 文書の外形

新規文書は`document_type = "script"`、`schema_version = "0.4.0"`として次を持つ。

```text
ScriptDocument
├── document_type: script
├── schema_version: 0.4.0
├── document_id: composition_id
├── revision: 1
├── status: validated
├── source_flow
│   ├── document_id
│   ├── revision
│   └── content_sha256
└── script
    ├── title
    ├── brief
    ├── performance_setup
    ├── root_section_id
    ├── sections{}
    ├── materials{}
    ├── material_placements{}
    ├── script_element_variation_relations{}
    └── material_placement_transitions{}
```

機械用のJSON名は次を使う。

| 日本語 | JSON |
|---|---|
| 楽曲設計データ | `script` |
| 区分 | `sections` |
| マテリアル | `materials` |
| マテリアル配置 | `material_placements` |
| 楽曲設計要素の変奏関係 | `script_element_variation_relations` |
| マテリアル配置遷移 | `material_placement_transitions` |
| 演奏条件 | `performance_setup` |
| 演奏指示 | `performance_directions` |

配列の位置やJSON objectの記述順を、識別子または演奏順の意味に使わない。

## 識別子

- collectionは、要素IDをkeyにしたJSON objectとする。
- `document_id`と要素IDは`^[a-z][a-z0-9_-]*$`に一致させる。
- 要素IDはcollectionごとに一意とする。
- 参照先の種類はfieldまたは`type`で明示し、文字列だけから推測しない。
- 表示名や説明文をIDに使わない。
- 同じ意味の要素はrevision間でIDを維持し、削除したIDを別の意味に再利用しない。

初回の安定ID、Schema版、revision、状態、楽曲台本の内容hashはPythonが設定し、LLMに選ばせない。

## 区分と時間

`Section`は順序付きの包含木を作る。

- 根は一つで、`root_section_id`から全区分へ到達できる。
- 根以外は存在する親を参照し、循環を禁止する。
- 同じ親の子は`order`を0から欠番なく持つ。
- 子を持たない葉だけが正の`relative_length`を持つ。
- 枝の長さは子から導出し、重複保存しない。

楽曲設計データは、兄弟順、葉の相対長、希望する全体時間だけを持つ。音符の位置、テンポ曲線、絶対ミリ秒は持たない。

## マテリアルとマテリアル配置

[マテリアル](glossary.md#マテリアル)は`materials`へ保存する。各要素は抽象的な音楽の考えを表す説明を持ち、音符、MIDI音高、左右の手、ペダル値は持たない。区分の働きと配置時の役割を混在させる`kind`は持たない。

[マテリアル配置](glossary.md#マテリアル配置)は`material_placements`へ保存する。各要素は`material_id`、子区分を持たない区分の`section_id`、配置時の`role`を持つ。`role`はSCoIM中核では識別子として検査し、許可する語彙と組み合わせは生成プロファイルが定める。

同じ子区分を持たない区分にある複数のマテリアル配置は、その区分で併用できる。各配置から作る音符は、区分の途中から始まったり、区分の途中で終わったりしてよく、すべての配置が区分全体で鳴り続けるとは扱わない。局所的な発音の開始と終了だけを理由に、構成上の区分を分割しない。

区分内で前後する内容が構成上の順序を表す場合は、マテリアル配置へ順序情報を追加せず、順序と相対長を持つ子区分へ分ける。一つのマテリアルを複数の区分で使う場合も、使用箇所ごとに別のマテリアル配置を作る。

## 楽曲設計要素の変奏関係

[楽曲設計要素の変奏関係](glossary.md#楽曲設計要素の変奏関係)は`script_element_variation_relations`へ保存する。

```json
{
  "source": {"type": "section", "id": "a1"},
  "target": {"type": "section", "id": "a2"},
  "preserve": ["主題の特徴"],
  "change": ["演奏の揃い方", "終結への向かい方"],
  "description": "冒頭と分かる形で、より明確に再現する。"
}
```

- `source`と`target`の`type`は`section`、`material`、`material_placement`のいずれかとし、同じ種類の別IDにする。
- 区分またはマテリアル配置では、元を演奏順で先に置く。
- マテリアルの変奏関係は循環させない。
- `preserve`と`change`は空でない文字列配列とし、同じ項目を両方へ置かない。
- 同じtargetへ複数の変奏を定義しない。

楽曲設計要素の変奏関係が正しいだけでは、聴感上の同一性と差を証明しない。下位投影で保持対象と変更対象を別々に検査する。

## マテリアル配置遷移

[マテリアル配置遷移](glossary.md#マテリアル配置遷移)は`material_placement_transitions`へ保存する。

- `source_material_placement_id`、`transition_material_placement_id`、`target_material_placement_id`で三つのマテリアル配置を明示し、互いに異なるものとする。
- 遷移用マテリアルも通常のマテリアル配置として、正の相対長を持ち、子区分を持たない専用区分へ置く。その区分には、前景の遷移用マテリアル配置を一つだけ置き、伴奏など他の配置を置かない。
- 演奏順は前、遷移、次とする。
- 時間、マテリアル、音符をマテリアル配置遷移へ重複保存しない。
- 専用の遷移が不要な隣接には作らない。

構造を提案するモデル操作は、非公開の転送情報で遷移専用区分を明示する。Pythonは検査済み構造から、演奏順を満たす元、遷移専用、先の候補だけを作る。関係を提案するモデル操作は、任意の三つの配置を組み立てず、遷移専用区分ごとにPythonが作った候補を一つ選ぶ。候補IDと非公開の用途は、この文書へ保存しない。

遷移専用のマテリアル配置は、楽曲設計要素の変奏関係の元または先にしない。遷移は前後の配置を文脈として生成し、別の変奏関係を重ねない。

## 演奏条件と編成

初版は一つの`performance_setup`を必須とし、次を持つ。

- `instrumentation`: 一つの[編成](glossary.md#編成)を示す識別子
- `target_duration_seconds`: 希望演奏時間
- `performance_directions`: 区分またはマテリアル配置への自然言語の演奏指示

演奏指示は対象、説明、一つ以上の[演奏要素](glossary.md#演奏要素)を必須とし、必要なら比較対象を持てる。MIDIの絶対時刻、velocity、CC64は持たない。音域、旋律、伴奏音、和音など、楽譜を変更する内容も持たない。編成、時間、演奏要素、対象の対応範囲と失敗状態は、選択した[生成プロファイル](glossary.md#生成プロファイル)が定める。`solo_piano_3m_v2`の能力は[3分ピアノ生成プロファイル](controllable-composition-system.md#v2の生成能力)を正本とする。

```json
{
  "performance-direction-001": {
    "target": {"type": "section", "id": "section-005"},
    "relative_to": {"type": "section", "id": "section-003"},
    "performance_aspects": ["timing", "coordination"],
    "description": "前の高まりを受け、少し落ち着いて迷う気配を作る。"
  }
}
```

`target`は指示を適用する対象、`relative_to`は比較元である。比較元を必要としない指示では`relative_to`を省略する。対象と比較元に許可する種類は生成プロファイルが定める。

一つの楽曲台本と楽曲設計データは、一つの編成専用とする。複数化する場合はSchema版を上げ、単数fieldをその場で別の意味に読み替えない。

`performance_aspects`は、説明が対象にする弾き方の種類を示す。値は生成プロファイルが対応する演奏要素に限り、空配列や重複を許さない。Pythonは値と対応能力を検査するが、自由記述が本当にその演奏要素だけを述べているかまでは自動判定しない。

演奏指示の比較元は、どの区分を参考に変えるかを示す。何をどれだけ増減するかは自然言語の説明に含め、知覚上の特徴名、検査式、絶対値へ置き換えない。比較元を持つ演奏指示は、比較元から対象への向きが循環しないようにする。

演奏指示を具体的な演奏方法へ変換する契約は[ピアノ演奏](solo-piano-performance.md)を正本とする。楽曲設計データには、生成プロファイル固有の演奏方法、MIDIの値、知覚上の比較検査を保存しない。

## 検査

[検査](glossary.md#検査)では、JSON Schema Draft 2020-12を使って必須field、型、列挙値、数値範囲、ID形式、未知fieldを調べる。Pythonの意味検査では、参照切れ、区分木、順序、マテリアル配置、変奏、遷移、演奏指示、revisionとhashを調べる。

Schema不正と意味不正を区別し、内容上の不正を例外だけで返さない。問題は型付きの`code`、人間向けの`message`、該当位置を持つ。内容修正と失敗処理は[生成工程](scoim-generation-workflow.md)を正本とする。

## 正規化、版、hash

- 文字コードはUTF-8とする。
- JSON objectの記述順を意味に使わない。
- RFC 8785で正規化した内容のSHA-256を記録する。
- 内容hash対象は`document_type`、`schema_version`、`document_id`、`revision`、`source_flow`、`script`とする。
- `status`を内容hashから除外する。
- 人間向け保存は2空白インデントとLFを既定にする。
- 未対応Schema版と未知fieldを拒否する。
- 旧`0.1.0`、`0.2.0`、`0.3.0`は、その版を使う現行経路または既存bundleの通信なし再生に限って読み、その場で`0.4.0`へ書き換えない。

## 保存

検査に合格した文書は、構成bundleの`validated-script.json`へ不変snapshotとして保存する。成功した構成bundleだけがこのファイルを持つ。固定配置、原子的保存、成功と失敗の記録は[生成工程](scoim-generation-workflow.md)に従う。

## 下位投影

楽曲設計データから、構造骨格と[投影台帳](glossary.md#投影台帳)を決定的に作る。調性、和声、音符のように楽曲設計データが持たない値を勝手に補わず、後続の音楽選択へ残す。

投影台帳の各記録は、変換元の種類とID、変換先の種類とID、対象段階、検査方法、状態、検査証拠を持つ。状態は少なくとも`planned`、`passed`、`failed`、`unchecked`、`unverified`を区別する。機械的に検査すべき対象は、検査を実行して証拠が一致した場合だけ`passed`にする。検査器がない場合は`unchecked`として投影を失敗させる。自然言語の達成だけを`unverified`にできる。

`solo_piano_3m_v2`では、区分を`PiecePlan`のノードへ、子区分を持たない区分を楽譜生成単位へ、マテリアル配置を楽譜生成単位レイヤーへ一対一で対応させる。楽譜生成単位レイヤーはマテリアル配置だけを出所として参照し、区分とマテリアルはマテリアル配置からたどる。同じ出所情報を下位要素へ重複保存しない。

新しい対象関係を使う生成では、本書の`0.4.0`、`status = "validated"`の文書を、構成後の段階生成、作業状態、実行記録、成果物生成、通信なし再生まで一貫して使う。下流処理のために旧Schemaまたは別の状態へ変換しない。旧Schemaは対応する経路とhash契約のまま扱う。

ID、親子関係、順序、相対長、マテリアル配置、変奏、遷移、編成、時間は直接検査する。自然言語の説明は該当する生成段階へ欠落なく渡し、達成を自動判定できない場合は`unverified`とする。対象、形式、生成能力など、機械的な処理に必要な情報を表せない場合は、黙って省略せず`unrepresentable`を返す。

ピアノ向けの具体的な投影は[ピアノ生成の全体像](performance-generation-pipeline.md)以下を正本とする。作成途中状態、LLM操作、依存hash、再開、修正方式を楽曲設計データへ追加しない。

## 設計を見直す条件

- 固定例を表すためにcollection外の例外fieldが必要になる。
- 楽曲台本の項目を投影先へ欠落なく対応付けられない。
- マテリアル配置遷移が時間やマテリアルを重複して持たないと投影できない。
- 同じマテリアルの複数のマテリアル配置をSCoIM中核で禁止しなければ検査できない。
- ピアノ固有の音高、声部、ペダルをこの文書へ追加する必要が生じる。

この契約に関する設計判断の状態は[決定状態](scoim-decision-status.md)を正本とする。
