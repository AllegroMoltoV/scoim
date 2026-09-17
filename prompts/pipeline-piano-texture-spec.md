あなたは、独奏ピアノ曲の伴奏声部だけを設計します。

和声、素材長、前景声部、前景音、方向は確定済みです。これらを変更せず、旧伴奏を参照せずに新しい伴奏を作ってください。

固定情報は次のとおりです。

{{FIXED_MATERIAL_JSON}}

`composition_source`には、次の形のPianoTextureSpecだけを出力してください。Markdownコードフェンスや説明文を含めないでください。

```text
piano_texture_spec(
    material_id="material_4",
    events=[
        piano_texture_note(
            event_id="pt_01",
            harmony_id="m4_h1",
            role="accompaniment",
            voice="lower",
            at_units=0,
            duration_units=8,
            degree="root",
            octave=2,
            articulations=["tenuto"],
        ),
    ],
)
```

制約は次のとおりです。

- `material_id`は`material_4`とします。
- eventは8個、異なる`at_units`は4個とします。
- 0以上8未満、8以上16未満、16以上24未満、24以上32未満の各区間に、少なくとも一つの発音位置を置きます。
- `event_id`は`pt_`で始まる新しい一意なIDにします。
- `harmony_id`は固定情報にあるIDだけを使います。
- `role`は`accompaniment`、`voice`は前景声部の反対だけを使います。
- `degree`は`root`、`third`、`fifth`、`seventh`のいずれかです。三和音では`seventh`を使いません。
- `octave`は変換後音の科学的ピッチ表記上のオクターブ番号です。絶対MIDI音高の`pitch`は書きません。
- eventは一つの和声区間内に収め、和声境界をまたがせません。
- 伴奏音は、同時に鳴る前景音より低くします。
- 低域では密集した3度を避け、根音、完全5度、オクターブを優先します。ただし、全体から3度を禁止しません。
- 伴奏を最後の1音だけへ退化させず、4区間を通してピアノ伴奏として支えます。
- `articulations`は`normal`、`staccato`、`tenuto`、`accent`の重複しない一覧です。空の一覧も許可します。
- `tie`、velocity、pedal、絶対時刻、MusicXML、SMFは書きません。

出力前に、event数、発音位置数、各区間の被覆、和音度数、声部、音域を再確認してください。
