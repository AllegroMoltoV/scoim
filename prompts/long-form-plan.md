あなたは独奏ピアノ曲の全体構成を設計します。音符とペダルはまだ作らず、制限付き記法の `composition` だけを返してください。

総尺は `ending` を含めて 180,000 ms とします。`ending=tonic_hold(duration_ms=4000)` を指定してください。

曲は大区分、フレーズ、素材の 3 層で設計します。`part` は全曲上の大区分、`phrase` は大区分内で聞き取れる A、A'、B、A などのまとまり、`material` は 2,000 ms から 6,000 ms の短い音素材です。

`parts` は 3 個以上とし、個数と長さを四等分へ固定しないでください。最初の `part` は `role="opening"` とします。`role="climax"` の `part` はひとつだけにし、最初と最後には置かないでください。climax の `energy` は単独最大とし、その後により低い `energy` の `part` を置いてください。part の role は opening、development、climax、return、release から選びます。

各 `part` は `phrases` を 2 個以上持ちます。各 phrase の展開後時間は 6,000 ms から 18,000 ms とします。phrase の role は statement、variation、contrast、return から選びます。

variation と return の phrase は、先に現れた phrase ID を `derived_from` に指定します。variation には `variation_kind` も必須です。値は rhythmic、textural、registral のいずれかとし、変奏の違いを一種類へ限定します。少なくともひとつの statement の後に variation を置いてください。少なくともひとつは、statement と return の間に contrast を挟み、`A -> B -> A` に相当する非隣接回帰を作ってください。すべての大区分を同じフレーズ配列にせず、固定した `A -> A' -> B -> B' -> C` を曲全体へ機械的に反復しないでください。

variation phrase の通常素材は、元 phrase の素材と一件ずつ対応させます。変奏素材の `material` に、元素材 ID を `derived_from` として指定してください。元素材の完全な再利用は variation では禁止です。return phrase は、元 phrase の素材を正確に再利用するか、`derived_from` を持つ変奏素材を使います。part 間の transition 素材にはこの対応を要求しません。

全体で `use` を 32 個以上置いてください。同じ `material_id` はひとつの part で最大 2 回まで使えます。全 use 数の 60% 以上を固有の `material_id` にしてください。同じ素材を隣り合わせてはいけません。冒頭と終盤を結ぶ回帰素材を除き、同じ素材を全曲で何度も繰り返してはいけません。

最後の part を除き、各 part の最後の use は `role="transition"` とします。transition 用 material_id は全曲で一度だけ使い、次の part へ向かう短い fill-in を担当させます。最後の part には transition を置きません。

各 use にも role と energy を指定してください。use の role は opening、contrast、transition、return、climax、release から選びます。最初の use は opening、climax の use はひとつだけとし、climax の part 内へ置いてください。同じ material_id の再利用では energy を同じにしてください。冒頭の material_id を最後の part で 2 個以上正確に再利用し、全曲単位の回帰を作ってください。

`materials` には material_id、duration_ms、必要な場合だけ derived_from を置き、`notes=[]` としてください。`pedals` は省略してください。元素材は派生素材より先に宣言してください。素材時間と use 回数から求めた本文を 176,000 ms にしてください。素材数、大区分数、大区分比率を参照曲の境界候補から決めないでください。

素材生成は最大4バッチで、1バッチは同じ派生深さの素材を最大8個処理します。派生素材からさらに派生させず、`derived_from`は必ず元素材を直接参照してください。元素材は最大16個、派生素材も最大16個とし、各群を8個ずつに分けても合計4バッチ以内に収めてください。

生成モデルへ渡してよい作風目標は次の数値範囲だけです。

{{style_target}}

この候補では、次の独立性要件を満たしてください。

{{plan_brief}}

この要件は、大区分数、フレーズ配列、素材時間を既存例から複写しないための条件です。音符やペダルは指定せず、上記の構造契約の範囲で独自の計画を作ってください。

この数値範囲は作曲判断の参考情報です。schema_version、scope、axis_targets を composition の引数へ入れないでください。composition の引数は title、tonal_center、mode、ending、parts、materials だけです。tonal_center は音名ではなく、C=0、C#=1、D=2、D#=3、E=4、F=5、F#=6、G=7、G#=8、A=9、A#=10、B=11 の整数で指定してください。part_id、phrase_id、material_id、use の material_id は位置引数で書いてください。energy は 1 から 5 の整数です。コードフェンス、コメント、変数代入は出力しません。

値を固定しない構文骨格は次のとおりです。省略可能な引数は、必要な場合だけ追加してください。

- `composition(title=..., tonal_center=..., mode=..., ending=tonic_hold(duration_ms=4000), parts=[...], materials=[...])`
- `part("PART_ID", role=..., energy=..., phrases=[...])`
- `phrase("PHRASE_ID", role=..., uses=[use(...)])`
- `phrase("PHRASE_ID", role="variation", derived_from="EARLIER_PHRASE_ID", variation_kind=..., uses=[use(...)])`
- `use("MATERIAL_ID", role=..., energy=...)`
- `material("MATERIAL_ID", duration_ms=..., notes=[], derived_from="EARLIER_MATERIAL_ID")`

以下の完全な構文例は内部のパーサー回帰テスト専用であり、実際のモデル呼び出し前に除外されます。モデルはこの配列を複写せず、上の制約から計画を組み立ててください。

<!-- INTERNAL_VALIDATION_EXAMPLE_BEGIN -->
```python
composition(
    title="曲名",
    tonal_center=9,
    mode="minor",
    ending=tonic_hold(duration_ms=4000),
    parts=[
        part(
            "P1",
            role="opening",
            energy=2,
            phrases=[
                phrase(
                    "A1",
                    role="statement",
                    uses=[
                        use("A", role="opening", energy=2),
                        use("B", role="contrast", energy=2),
                    ],
                ),
                phrase(
                    "A2",
                    role="variation",
                    derived_from="A1",
                    variation_kind="rhythmic",
                    uses=[
                        use("Av", role="return", energy=2),
                        use("Bv", role="contrast", energy=2),
                    ],
                ),
                phrase(
                    "X1",
                    role="contrast",
                    uses=[
                        use("X", role="contrast", energy=2),
                        use("T1", role="transition", energy=2),
                    ],
                ),
            ],
        ),
        part(
            "P2",
            role="development",
            energy=3,
            phrases=[
                phrase(
                    "B1",
                    role="statement",
                    uses=[
                        use("C", role="contrast", energy=3),
                        use("D", role="contrast", energy=3),
                    ],
                ),
                phrase(
                    "B2",
                    role="variation",
                    derived_from="B1",
                    variation_kind="textural",
                    uses=[
                        use("Cv", role="return", energy=3),
                        use("Dv", role="contrast", energy=3),
                    ],
                ),
                phrase(
                    "A3",
                    role="return",
                    derived_from="A1",
                    uses=[
                        use("A", role="return", energy=2),
                        use("B", role="return", energy=2),
                        use("T2", role="transition", energy=3),
                    ],
                ),
            ],
        ),
        part(
            "P3",
            role="development",
            energy=4,
            phrases=[
                phrase(
                    "C1",
                    role="statement",
                    uses=[
                        use("E", role="contrast", energy=4),
                        use("F", role="contrast", energy=4),
                    ],
                ),
                phrase(
                    "C2",
                    role="variation",
                    derived_from="C1",
                    variation_kind="rhythmic",
                    uses=[
                        use("Ev", role="return", energy=4),
                        use("Fv", role="contrast", energy=4),
                    ],
                ),
                phrase(
                    "B3",
                    role="return",
                    derived_from="B1",
                    uses=[
                        use("C", role="return", energy=3),
                        use("D", role="return", energy=3),
                        use("T3", role="transition", energy=4),
                    ],
                ),
            ],
        ),
        part(
            "P4",
            role="climax",
            energy=5,
            phrases=[
                phrase(
                    "D1",
                    role="statement",
                    uses=[
                        use("G", role="climax", energy=5),
                        use("H", role="contrast", energy=4),
                    ],
                ),
                phrase(
                    "D2",
                    role="variation",
                    derived_from="D1",
                    variation_kind="textural",
                    uses=[
                        use("Gv", role="return", energy=4),
                        use("Hv", role="contrast", energy=4),
                    ],
                ),
                phrase(
                    "K1",
                    role="contrast",
                    uses=[
                        use("I", role="contrast", energy=4),
                        use("T4", role="transition", energy=4),
                    ],
                ),
            ],
        ),
        part(
            "P5",
            role="release",
            energy=2,
            phrases=[
                phrase(
                    "A4",
                    role="return",
                    derived_from="A1",
                    uses=[
                        use("A", role="return", energy=2),
                        use("B", role="return", energy=2),
                    ],
                ),
                phrase(
                    "R1",
                    role="contrast",
                    uses=[
                        use("J", role="contrast", energy=2),
                        use("K", role="contrast", energy=2),
                    ],
                ),
                phrase(
                    "R2",
                    role="contrast",
                    uses=[
                        use("N", role="contrast", energy=2),
                        use("U", role="release", energy=2),
                    ],
                ),
            ],
        ),
    ],
    materials=[
        material("A", duration_ms=5500, notes=[]),
        material("B", duration_ms=5500, notes=[]),
        material("Av", duration_ms=5500, notes=[], derived_from="A"),
        material("Bv", duration_ms=5500, notes=[], derived_from="B"),
        material("X", duration_ms=5500, notes=[]),
        material("T1", duration_ms=5500, notes=[]),
        material("C", duration_ms=5500, notes=[]),
        material("D", duration_ms=5500, notes=[]),
        material("Cv", duration_ms=5500, notes=[], derived_from="C"),
        material("Dv", duration_ms=5500, notes=[], derived_from="D"),
        material("T2", duration_ms=5500, notes=[]),
        material("E", duration_ms=5500, notes=[]),
        material("F", duration_ms=5500, notes=[]),
        material("Ev", duration_ms=5500, notes=[], derived_from="E"),
        material("Fv", duration_ms=5500, notes=[], derived_from="F"),
        material("T3", duration_ms=5500, notes=[]),
        material("G", duration_ms=5500, notes=[]),
        material("H", duration_ms=5500, notes=[]),
        material("Gv", duration_ms=5500, notes=[], derived_from="G"),
        material("Hv", duration_ms=5500, notes=[], derived_from="H"),
        material("I", duration_ms=5500, notes=[]),
        material("T4", duration_ms=5500, notes=[]),
        material("J", duration_ms=5500, notes=[]),
        material("K", duration_ms=5500, notes=[]),
        material("N", duration_ms=5500, notes=[]),
        material("U", duration_ms=5500, notes=[]),
    ],
)
```
<!-- INTERNAL_VALIDATION_EXAMPLE_END -->

JSON Schema に従い、`composition_source` と短い `intent_summary` だけを返してください。
