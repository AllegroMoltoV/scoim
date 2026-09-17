# PerformanceSpec生成

確定済みPiecePlan:
{{PIECE_PLAN_DSL}}

この段階の匿名参照目標:
{{REFERENCE_TARGET_JSON}}

確定済みScoreSpecの要約:
{{SCORE_SUMMARY}}

利用可能なprofile:
{{PROFILE_CATALOG}}

performance_spec(...)を一つだけ出力してください。直接関数呼出し、文字列、整数、None、リストだけを使ってください。

正確な署名:
- performance_spec(performance_id=..., target_duration_ms=..., default_velocity=..., timing_budget_id=..., node_performances=[...])
- node_performance(node_id=..., timing_profile=..., timing_amount=..., dynamics_profile=..., articulation_profile=..., coordination_profile=..., pedal_profile=...)

`target_duration_ms`は180000です。node_idは確定済みPiecePlanのノードだけを1回以下で指定してください。省略可能なprofileは、不要なら引数自体を書かないでください。値は利用可能なprofileのselectable_valuesからだけ選んでください。

この段階では、構成ノードごとの時間形状、強弱実現、実音長、声部協調、ペダルだけを選びます。同じ楽譜の再現では、指定された演奏差を作ってください。rolledは崩す提示、alignedは縦線を保持する回帰にだけ使ってください。

SCORE_SUMMARYの全葉と局所和声境界を確認してください。各葉の実効pedal_profileは、その葉から祖先へ遡った最も近い明示指定です。rootの指定は全曲へ継承され得ます。harmony_legato以外で、同じ所有ノードと同じprofileが続く葉は一つのペダル区間へ統合されます。

和声を持つ葉ではharmony_legatoを基本とし、旧伴奏音が次の局所和声へ残らないようにしてください。単一和声素材どうしの境界も対象です。和声を持たない素材へharmony_legatoを明示または継承させないでください。rootへ広い指定を一つ置くだけで、局所和声境界を処理したことにはなりません。

最後の葉の実効coordination_profileはscoreまたはalignedにし、rolledを使わないでください。最終主和音の打鍵時にペダルが実際にdownとなり、その後2,000 ms以上保持して曲末でupにしてください。最終主和音の実音も2,000 ms以上保持してください。曲末のupだけを置くことや、終止和音をロールして打鍵時刻を分散させることは禁止します。

禁止フィールド: pitch、score_note、音符列、MusicXML、SMF、絶対イベント列、個別音符のat_ms。

説明文やMarkdownコードフェンスを付けないでください。
