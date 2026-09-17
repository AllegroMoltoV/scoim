# PiecePlan生成

利用者要求:
{{REQUEST_JSON}}

参照目標:
{{REFERENCE_TARGET_JSON}}

校正条件:
{{CALIBRATION_CONTRACT}}

piece_plan(...)を一つだけ出力してください。直接関数呼出し、文字列、整数、None、リストだけを使ってください。

正確な署名:
- piece_plan(plan_id=..., title=..., tonal_center=..., mode=..., root_node_id=..., ending_intent=..., nodes=[...])
- plan_node(node_id=..., parent_id=..., order=..., role=..., derived_from=..., contrasts_with=..., harmonic_focus=..., duration_weight=..., score_material_id=...)

`tonal_center`は0から11の整数です。0=C、1=C#、2=D、3=D#、4=E、5=F、6=F#、7=G、8=G#、9=A、10=A#、11=Bとして指定し、"D"などの音名文字列は使わないでください。各`plan_node`の`harmonic_focus`は0から11の整数またはNoneとし、和声の説明文は入れないでください。

`mode`はmajorまたはminor、`ending_intent`はtonicです。`role`はwhole、opening、statement、variation、contrast、transition、climax、return、releaseから選びます。rootは1件だけにし、親を子より先に置き、兄弟のorderを0から連番にしてください。葉だけに正のduration_weightとscore_material_idを設定します。duration_weightは1以上の整数です。0.09のような小数や比率は使わず、相対的な長さを9、13、14のような整数比で表してください。returnは同じ深さの先行ノードをderived_fromで参照し、対比には同じ深さの先行ノードをcontrasts_withで参照してください。異なる素材IDは8件以内にしてください。transition葉のscore_material_idはその葉だけで使い、別のtransition葉や通常葉へ再利用しないでください。

自動評価できる回帰だけをderived_fromで宣言してください。derived_fromを持つ葉は、派生元の葉と同じscore_material_idを再利用します。derived_fromを持つ非葉は、派生元と派生先の両部分木に、同じscore_material_idを使う葉を少なくとも一つ置きます。葉と非葉の間ではderived_fromを使いません。新しい素材による変奏はroleをvariationなどにし、derived_fromを設定しません。最終葉は専用の終止素材を使い、最終葉にはderived_fromを設定しないでください。

この段階では、構成木、役割、派生元、対比対象、和声焦点、相対的な長さ、素材IDだけを決めます。対比には、比較する先行ノードをcontrasts_withで明示してください。

`semantic_targets`に`tonal_hierarchy`がある場合、`mode`を`plan_mode`へ一致させてください。`tonic_policy=piece_plan_relative`は主音を固定する意味ではありません。新しい曲に適した`tonal_center`を0〜11から選び、後段が同じ主音相対規則を使えるよう、和声焦点をその主音と曲全体の調性階層に整合させてください。区分ごとにmajorとminorを交互配置して数値だけを満たす構成にはしないでください。

禁止フィールド: 音符、pitch、at_ms、duration_ms、velocity、pedal、演奏profile、MusicXML、SMF。

説明文やMarkdownコードフェンスを付けないでください。
