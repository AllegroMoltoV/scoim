# ScoreSpec生成

確定済みPiecePlan:
{{PIECE_PLAN_DSL}}

この段階の匿名参照目標:
{{REFERENCE_TARGET_JSON}}

派生元素材:
{{SOURCE_MATERIALS_DSL}}

保持する手掛かり:
{{IDENTITY_CUES}}

score_spec(...)を一つだけ出力してください。直接関数呼出し、文字列、整数、None、リストだけを使ってください。

正確な署名:
- score_spec(score_id=..., divisions=..., materials=[...])
- score_material(material_id=..., length_units=..., notes=[...], derived_from=..., harmonies=[...], foreground_voice=..., directions=[...])
- score_note(event_id=..., at_units=..., duration_units=..., pitch=..., voice=..., tie=..., articulations=[...])
- score_harmony(harmony_id=..., at_units=..., duration_units=..., root_pitch_class=..., quality=...)
- score_direction(direction_id=..., at_units=..., kind=..., value=...)

この応答のmaterialsは、保持する手掛かりのrequested_material_idsと同じ順、同じ集合にしてください。`voice`はupperまたはlower、`pitch`は21から108です。`quality`はmajor、minor、diminished、major-seventhから選びます。`articulations`はnormal、staccato、tenuto、accent、dynamicの値はpp、p、mp、mf、f、ff、breathの値はlightまたはfullです。省略可能な引数は、不要なら引数自体を書かないでください。

この段階では、相対整数時間、局所和声、前景声部、音高、上下声、tie、articulation、dynamic、breathだけを決めます。派生素材は、指定された手掛かりを保ちながら楽譜上の差を作ってください。

和声を使う素材では、score_harmony(...)を重複や欠落なく素材全体へ並べ、foreground_voiceをupperまたはlowerで指定してください。詳細和声の正本はscore_harmonyだけです。伴奏声部を先に和音構成音から作り、その後に前景声部を重ねてください。伴奏の非和声音は使わず、MIDI 48未満で同時に鳴る伴奏の最低2音は7半音以上離してください。

伴奏の全score_noteを、次の手順で出力前に検算してください。和音種別ごとの相対音高クラスはmajor={0,4,7}、minor={0,3,7}、diminished={0,3,6}、major-seventh={0,4,7,11}です。各値へroot_pitch_classを加えて12で割った余りを、その和声の構成音クラスにします。各伴奏音について、`harmony.at_units < note.at_units + note.duration_units`かつ`note.at_units < harmony.at_units + harmony.duration_units`を満たす、音価上で重なるすべての和声を列挙してください。重なる和声は一つ以上必要です。`note.pitch % 12`が、重なる各和声の構成音クラスに含まれることを確認してください。和声境界をまたぐ音に共通の構成音クラスがない場合は、音価を和声境界までに短くするか、境界後を別のscore_noteとして作ってください。発音時点の和声だけを確認して終わらず、全素材の全伴奏音を同じ手順で検算してからscore_spec(...)を返してください。

低域間隔は、すべての伴奏note-on位置をonsetとして一つずつ検算してください。各onsetで`note.at_units <= onset < note.at_units + note.duration_units`を満たす伴奏音を集め、同じ実音高を重複なしの集合にして昇順に並べます。2音以上あり最低音がMIDI 48未満なら、2番目の音高から最低音を引いた値を計算し、7以上にしてください。和音構成音どうしでも実音高差が5半音なら不合格です。転回名や音高クラスだけで判断しないでください。完全五度の音高クラスが根音より下にあっても、実音高差が7半音以上なら禁止しません。

前景の和声外音は、各発音位置の代表旋律音についてpassingまたはneighborとして説明できる場合だけ使ってください。前後の代表旋律音は同じscore_harmony内の和音構成音とし、対象音は和声区間の開始位置を避け、`duration_units * 2 <= divisions`を満たします。passingは前後とも半音または全音で同じ方向へ進み、neighborは半音または全音で進んで反対方向に元の音へ戻ります。同時刻の別音や直前の音だけを解決とみなさないでください。

PiecePlanの最後の葉には、他の葉が使わない単一和声の専用ScoreMaterialを割り当ててください。終止導入をこの素材へ含めないでください。最後の同時発音はPiecePlanのtonal_centerに対する主音と完全五度を含み、全音を素材終端まで保持します。先行音を最後の打鍵より後まで残さないでください。全葉出現のScoreMaterial.length_units合計を`total_occurrence_units`とするとき、最後の同時発音の最短音価は`minimum_final_duration_units * 180000 >= total_occurrence_units * 2000`を満たしてください。

禁止フィールド: at_ms、duration_ms、velocity、pedal、timing_profile、dynamics_profile、MusicXML、SMF。

説明文やMarkdownコードフェンスを付けないでください。
