# app/backend/schemas

pydantic を用いたモデルスキーマ定義
ユーザーへのレスポンス型式 をそれぞれ指定

# app/backend/models

DB に保存される型式

# app/backend/utils

ロガー・ヘルパーなどの汎用的な実装のみ

# 擬音語とパラメータの関係

- エンベロープをセットしない場合 -　ビー、ブーという音はでる。
- attack,decay,sustain を固定する場合はっきりとした音がでる
  - "attack": get_random_f(0, 0),
    "decay": get_random_f(0, 0),
    "sustain": get_random_f(1, 1),
    "release": get_random_f(0, 0.3),

# modulation Index

- 2 オペレータの場合

  - 1,18 といった値では音色に大きな差はない（論文：https://dke.maastrichtuniversity.nl/westra/PhDMaBa-teaching/GraduationStudents/Ba/LaurensAlbertsBaFINAL2005.pdf）に反する
  - 5000 ぐらいまで上げても音色に変化ある
  - サイン音として利用を考えるのであれば、1000 ぐらいでよい気がする。

  - 変調周波数とエンベロープセットの時間を入れ替えると音色が変わる

  - モジュレータの ADSR を変化させても音色に大差がないように聞こえた（再検討の余地あり）
  - release があると［ビッ］という音が出ない可能性

# Todo

- 未入力の適応度がある際にエラーを吐く
- 中心周波数を考慮して遺伝子を作成
- 周波数ゆらぎを用いた破裂音の作成アルゴリズム（パン、バン）
- sustain に振幅変調をかけることで［呼び出し］の機能イメージを持った音を生成する手法を検討
- ratioToCarrierFrequency が 3 の場合は偶数倍音のみが出るか調査
- ratioToCarrierFrequency の値の出現確率が等しいか確認

# アルゴリズム

- 直列のオシレータが多いほど「ブー」という音が出にくい気がする。

# Issue

- 入力を行わずに回答を送信すると世代数が変わらない（実験には影響ない）

# 実験 URL

http://localhost:8000/pi
http://localhost:8000/bu-
http://localhost:8000/fa-n
