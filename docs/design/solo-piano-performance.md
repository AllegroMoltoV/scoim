# ピアノ演奏設計

この文書は、`solo_piano_3m_v2`の[演奏選択語彙](glossary.md#演奏選択語彙)、[`PerformanceSpec`](glossary.md#performancespec)、[`RenderedPerformance`](glossary.md#renderedperformance)、[SMF](glossary.md#smf)変換のデータ契約を定める。楽譜を変えずに、時間の揺れ、上下声のそろい方、強弱、実音長、ペダルをどう実現するかを扱う。実装の未完了箇所は[roadmap](../roadmap/scoim-next-generation-path.md)に記載する。

## 演奏選択語彙

`solo_piano_3m_v2`は、[演奏選択語彙](glossary.md#演奏選択語彙)の各候補に、機械が読む値、人間が読める平易な説明、変更する`PerformanceSpec`のfieldを持たせる。プロンプト、モデル応答Schema、`PerformanceSpec`検査は同じ語彙を参照し、別々の固定値を持たない。

初期の`solo_piano_3m_v2`は、次のfieldを扱う。

- `timing_profile`: `neutral`、`savor`、`flow`、`build`、`release`
- `timing_amount`: `subtle`、`moderate`
- `dynamics_profile`: `steady`、`shape`、`build`、`release`
- `articulation_profile`: `score`、`legato`、`light`
- `coordination_profile`: `score`、`rolled`、`aligned`
- `pedal_profile`: `none`、`phrase_legato`、`harmony_legato`

[演奏要素](glossary.md#演奏要素)とfieldの対応は次のとおりである。

| 演奏要素 | 選択できるfield |
|---|---|
| `timing` | `timing_profile`、`timing_amount` |
| `dynamics` | `dynamics_profile` |
| `articulation` | `articulation_profile` |
| `coordination` | `coordination_profile` |
| `pedal` | `pedal_profile` |

語彙には、懐かしい、楽しい、盛り上がるなどの抽象的な印象語を登録しない。これらは自然言語の演奏指示としてLLMへ渡す。LLMは、完成した楽譜と前後関係を読み、表現に使う具体的な演奏方法を有限語彙から選ぶ。同じ言葉を常に同じ演奏方法へ決め打ちしない。

## 演奏指示から演奏方法への変換

Pythonは、演奏指示の対象と比較元から、比較元が先に確定する順序でフェーズ7のオペレーションを作る。同じ区分を対象にする複数の演奏指示は一つのオペレーションにまとめる。LLMへは、対象区分の説明と楽譜要約、演奏指示、比較元の確定済み演奏方法、演奏選択語彙を渡す。

LLMは、オペレーションの対象順に次を返す。

- `SectionPerformance`へ変換する六つの演奏field
- 各演奏指示をどのfieldで扱ったか
- 現在の演奏選択語彙では扱わなかった演奏指示と理由

LLMが変更できるfieldは、その対象にある演奏指示の`performance_aspects`に対応するfieldだけである。該当しないfieldは`null`とし、Pythonが最も近い上位区分の値または中立値を適用する。音域、旋律、伴奏音、和音などの楽譜上の内容を、演奏方法で代用しない。

モデル応答では対象IDを返させず、Pythonが応答位置を事前登録済みの対象へ対応付ける。各演奏指示は、扱ったものと扱わなかったもののどちらかに一度だけ現れなければならない。扱った演奏指示は、その演奏要素に対応する一つ以上の非`null`fieldを参照する。扱わなかった演奏指示はfieldを参照せず、理由を持つ。

扱わなかった演奏指示があっても、機械的な不合格がなければ生成を続ける。Pythonは、元の演奏指示、比較元、選択した方法、扱わなかった理由を投影台帳へ保存し、知覚上の達成を`unverified`とする。意図どおりに聞こえたと自動判定せず、指示を別の意味へ黙って置き換えない。

応答の型、語彙、対象への対応、演奏指示の過不足が不正な場合は内容不合格とする。修正と停止条件は[生成工程](scoim-generation-workflow.md)に従う。

## `PerformanceSpec`

`PerformanceSpec`は、目標演奏時間と、`PiecePlan`の区分へ適用する次の演奏値を持つ。

- `timing_profile`と変化量
- `dynamics_profile`
- `articulation_profile`
- `coordination_profile`
- `pedal_profile`

値を省略した場合は、構成木の最も近い祖先の明示指定を使う。時間の形だけは根から葉までを一つの予算内で合成し、階層を深くしただけで揺れが増えないようにする。

`solo_piano_3m_v2`では、本書の`PerformanceSpec`を演奏指定の正本型とする。旧`v1`の同名型または`NodePerformance`へ変換して検査せず、検査済みの`SectionPerformance`から直接構築する。

演奏指示の対象は、選択した生成プロファイルの能力で生成前に検査する。能力外の対象を広い区分へ読み替えない。`solo_piano_3m_v2`の対応範囲は[生成プロファイルの能力](controllable-composition-system.md#v2の生成能力)を正本とする。

## 時間と声部協調

時間プロファイルは、楽譜上の局所位置を演奏時間へ写す単調な曲線である。LLMは任意のミリ秒列を返さず、対応済みプロファイルと変化量を選ぶ。同じ楽譜時刻にある上下声、note-off、ペダルには同じ時間写像を使う。

`savor`は句頭と句末に溜めを作り、途中を前へ進める。`flow`は揺れを減らし、同じマテリアルを再度使う演奏を軽く流す。`build`は後半へ進み、`release`は終端へ向けて解放する。具体的な曲線と量は版付きの内部設定であり、普遍的な人間らしさの閾値にはしない。

`coordination_profile`は、楽譜上で同時の音をそのままそろえるか、短いロールとして低音から順に鳴らすかを選ぶ。乱数、声部ごとの継続的な遅延、メロディーだけを常に遅らせる処理は使わない。回帰部で縦線をそろえる演奏と、初回提示で崩す演奏を、同じマテリアルから作った別の楽譜生成単位で区別できる。

## 強弱と実音長

楽譜上のdynamicを基準velocityへ写し、演奏プロファイルを相対変形として適用する。最終velocityは1から127へ収め、clip件数を記録する。左手伴奏が同じvelocityを長く連打するなどの単調さは、平均値だけでなく、値の種類、範囲、隣接同値率、最長同値列を診断する。[強弱方針](glossary.md#強弱方針)の用語は用語集を参照する。

前景と伴奏の区別は、[`ScoreUnitLayer`](glossary.md#楽譜生成単位レイヤー)の出所である[マテリアル配置](glossary.md#マテリアル配置)から取得する。`upper`または`lower`という声部から推測しない。一つの前景レイヤーが上下両方の声部を含む場合も、一方を伴奏へ読み替えない。

`solo_piano_3m_v2`の現在の対応能力では、前景と伴奏を区別してvelocityを補正する強弱方針を扱わない。対応済みの強弱方針は、すべての音符へ同じ基準を適用する`legacy-unison-v1`だけである。ほかの強弱方針を旧レンダラーの単一の`foreground_voice`へ読み替えず、演奏を生成する前に`unrepresentable`とする。この制限は、前景と伴奏のデータ契約を声部単位へ変更するものではない。設計判断の状態は[決定状態](scoim-decision-status.md)を参照する。

楽譜上のstaccatoやtenutoは、実際のnote-off時刻へ決定的に写す。同じ音高の次のnote-onを越えてnote-offを延ばさない。キーを離した後の響きはサステインペダルと分けて扱う。

## ペダル

ペダルは、楽譜生成単位ごとの機械的な一定間隔ではなく、フレーズまたは共有和声のまとまりに合わせて踏み替える。和声が変わる境界では古い響きを解放し、次の打鍵後に踏み直す。ペダル時刻にも音符と同じ時間写像を使い、場面移行で旋律だけが唐突に切れないようにする。

最終主和音を長く保持する場合は、導入部の非主和音を同じペダルで持ち越さない。曲末ではすべての音符を終了し、CC64を0へ戻す。ペダルだけの修正は、楽譜音符、時間写像、velocityを変更しない。

`clear`は保存済み成果物を再生する互換用の名前としてのみ維持する。新しい生成の既定候補には含めない。

## `RenderedPerformance`

`RenderedPerformance`は、演奏の絶対時間を表す正本である。少なくとも次を持つ。

- 曲名と演奏時間
- `performed_note_id`、`source_score_note_id`、発音時刻、実音長、音高、velocityを持つ演奏音符
- 対応する楽譜生成単位IDと、時刻と値を持つペダルイベント
- 各区分の開始・終了時刻
- `PiecePlan`、`ScoreSpec`、`PerformanceSpec`のhash

演奏音符は`source_score_note_id`から所属する楽譜生成単位レイヤーをたどる。区分、マテリアル配置、マテリアルのIDを演奏音符へ重複保存しない。ペダルは区分全体の共有演奏なので、楽譜生成単位を参照する。

再現と変奏の診断では、曲中の絶対開始位置と全体伸縮を正規化して比較する。イベントIDと絶対配置だけの違いを可聴差に数えない。楽譜差、演奏差、同一性の手掛かり、完全コピーを別々に記録する。

## SMF

SMFは`RenderedPerformance`からだけ生成する。現行方式は1 tickを1 msに対応させるため、時刻0に`set_tempo=500000`、ticks per beat 500を使う。この設定は楽曲上のBPMを意味しない。

同時刻のイベントはnote-off、CC64、note-onの安定順に並べる。曲末で全note-offとCC64=0を保証する。生成後にSMFを再読込みし、発音時刻、音価、音高、velocity、CC64、曲末時刻が`RenderedPerformance`と一致することを確認する。

SMFはSCoIMの出所IDを保存しない。再読込みでは保存可能な演奏イベントの値と件数を照合し、SMFから楽譜音符または楽譜生成単位レイヤーを復元できるとは扱わない。

## 診断の限界

形式、和声持越し、声部のそろい方、velocityの単調さなどは機械的に検査できる。一方、演奏が自然か、表情が意図どおりか、曲として魅力的かは自動検査だけでは確定できない。承認済みの観測契約がない`loudness`や`onset_alignment`の大小を、必須の知覚条件として追加しない。人間の試聴は最終SMFを基本とし、問題がある場合だけ診断用SMFへ遡る。
