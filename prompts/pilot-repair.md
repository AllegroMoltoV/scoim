次の制限付き作曲記法は機械検査に失敗しました。音楽的意図を保ち、示した失敗だけを直してください。tonal_center、mode、ending が既にある場合は変更しないでください。ない場合は、本文の調性に合う tonal_center=0..11、mode="major" または mode="minor"、ending=tonic_hold(duration_ms=4000) を追加してください。本文は 41,000 から 54,000 ms とします。

各 use には role と energy が必要です。最初の role は opening、climax はひとつだけ、climax の energy は単独最大です。同じ material を再利用する use では energy を同じにしてください。attack_style は使わないでください。

隣接区間の少なくともひとつでは、3 音以上がまとまって始まる音符の割合を 0.20 以上変えてください。さらに、1 秒当たりの音符数を 1.35 倍以上変える、平均同時発音数を 1.35 倍以上変える、中央音高を 6 半音以上変える、中央 velocity を 8 以上変える、のいずれかも満たしてください。energy が高い区間では、1 秒当たりの音符数、中央 velocity、平均同時発音数のうち二つ以上を実際に高くしてください。climax はこの三指標のうち二つ以上で単独の最大にしてください。

各 material では、音価、連続する発音間隔、同じ時点にまとまる音数のすべてを一定にしないでください。無作為な時刻のずれではなく、聞き取れる音楽的な理由を持つ変化にしてください。

pedal は MIDI CC64 です。一般的な MIDI 1.0 再生では 0 から 63 が解放、64 から 127 が踏下です。この記法の新規生成では value=0 と value=127 だけを使ってください。各 material の at_ms=0 で踏下し、素材時間の 85% 以上を踏下状態にします。8,000 ms 以上の material では、旋律または和声のひとまとまりが終わる位置で少なくとも一度踏み替えてください。曲中の解放から再踏下までを 100 から 400 ms とし、最後の 500 ms 以内で value=0 にしてください。無作為な一定間隔で踏み替えないでください。

失敗が SustainContractError で始まる場合は、pedals だけを修正してください。title、tonal_center、mode、ending、form、material の duration_ms、notes、role、energy を変更しないでください。

次は文法だけを示す完全例です。音符、題名、調性、構成、変化の位置をコピーしないでください。

```python
composition(
    title="example",
    tonal_center=0,
    mode="minor",
    ending=tonic_hold(duration_ms=4000),
    form=[
        use("A", role="opening", energy=2),
        use("B", role="climax", energy=5),
        use("A", role="return", energy=2),
    ],
    materials=[
        material(
            "A",
            duration_ms=14000,
            notes=[
                note("a1", at_ms=0, duration_ms=900, pitch=48, velocity=50),
                note("a2", at_ms=0, duration_ms=1100, pitch=55, velocity=54),
                note("a3", at_ms=0, duration_ms=1300, pitch=60, velocity=58),
                note("a4", at_ms=2200, duration_ms=500, pitch=65, velocity=60),
                note("a5", at_ms=3400, duration_ms=700, pitch=50, velocity=52),
                note("a6", at_ms=3400, duration_ms=1000, pitch=57, velocity=56),
                note("a7", at_ms=5200, duration_ms=600, pitch=52, velocity=53),
                note("a8", at_ms=5200, duration_ms=900, pitch=59, velocity=57),
                note("a9", at_ms=5200, duration_ms=1200, pitch=64, velocity=61),
                note("a10", at_ms=7600, duration_ms=400, pitch=67, velocity=62),
                note("a11", at_ms=8800, duration_ms=800, pitch=55, velocity=54),
                note("a12", at_ms=8800, duration_ms=1400, pitch=62, velocity=59),
            ],
            pedals=[
                pedal("p1", at_ms=0, value=127),
                pedal("p2", at_ms=5200, value=0),
                pedal("p3", at_ms=5400, value=127),
                pedal("p4", at_ms=13800, value=0),
            ],
        ),
        material(
            "B",
            duration_ms=14000,
            notes=[
                note("b1", at_ms=0, duration_ms=400, pitch=60, velocity=78),
                note("b2", at_ms=0, duration_ms=650, pitch=67, velocity=82),
                note("b3", at_ms=0, duration_ms=900, pitch=72, velocity=86),
                note("b4", at_ms=900, duration_ms=300, pitch=75, velocity=88),
                note("b5", at_ms=1700, duration_ms=600, pitch=77, velocity=90),
                note("b6", at_ms=3000, duration_ms=500, pitch=65, velocity=80),
                note("b7", at_ms=3000, duration_ms=800, pitch=72, velocity=84),
                note("b8", at_ms=4100, duration_ms=350, pitch=67, velocity=82),
                note("b9", at_ms=4100, duration_ms=700, pitch=74, velocity=88),
                note("b10", at_ms=5600, duration_ms=450, pitch=81, velocity=94),
                note("b11", at_ms=6500, duration_ms=500, pitch=69, velocity=84),
                note("b12", at_ms=6500, duration_ms=900, pitch=76, velocity=90),
                note("b13", at_ms=8200, duration_ms=300, pitch=83, velocity=96),
                note("b14", at_ms=9300, duration_ms=400, pitch=71, velocity=86),
                note("b15", at_ms=9300, duration_ms=750, pitch=78, velocity=92),
                note("b16", at_ms=10500, duration_ms=550, pitch=84, velocity=98),
                note("b17", at_ms=11500, duration_ms=350, pitch=79, velocity=91),
                note("b18", at_ms=12600, duration_ms=700, pitch=86, velocity=100),
            ],
            pedals=[
                pedal("bp1", at_ms=0, value=127),
                pedal("bp2", at_ms=6500, value=0),
                pedal("bp3", at_ms=6700, value=127),
                pedal("bp4", at_ms=13800, value=0),
            ],
        ),
    ],
)
```

各関数では、例にある引数名だけを使ってください。id、events、start_ms、素材の絶対開始時刻、attack_style は禁止です。composition_source には上記形式の式だけを入れてください。Beads、実装、テスト、作業報告に関する文章を入れないでください。

失敗: {{error}}

入力:

{{source}}

JSON Schema に従い、修正版 composition_source と短い intent_summary だけを返してください。
