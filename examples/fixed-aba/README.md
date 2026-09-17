# 旧形式の互換性を検査する固定例

この例は、旧`approved-script.json`と旧凍結応答の再生互換性を検査するために残している。現在の楽曲台本へ暗黙変換せず、旧形式用のPython APIで扱う。新しい利用方法は`examples/fixed-flow/`を参照する。

リポジトリのルートで開発環境を準備した後、次を実行する。

旧互換経路は配布物検査で自動実行する。公開CLIの`check`と`realize`へこの旧文書を渡してはいけない。

生成される主なファイルは次のとおりである。

- `score.musicxml`
- `final.mid`
- `realization-diagnostics.json`
- `manifest.json`

現行実現器版で期待する`final.mid`のSHA-256は`752e8379b32d9a0908a424d9c20caaca8703d349c40e0f60f6b3d456c5c3a80a`である。hashが変わった場合は、意図した実現器版変更か、再現性の破損かを調べる。
