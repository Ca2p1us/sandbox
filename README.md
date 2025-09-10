# シミュレータの動かし方
## appディレクトリに移動して下記のコマンドで実行
```tarminal
python -m backend.engine.run_iga_simulation
```
# パラメータの検討
- [Chatgptによる回答](https://chatgpt.com/s/t_68be5ddb84a08191a54ae9eadee2b8c5)
- 各パラメータの定義域
    - アタック 0～0.25sec
    - ディケイ 0～0.25sec
    - サスティン 0～1
    - 持続時間 0～0.3sec
    - リリース 0～0.35sec
    - 周波数 200～1300Hz

# 疑似ユーザモデルの代表的分類

| 分類名 | 概要 | 主な特徴 | 代表的研究例 |
|--------|------|----------|--------------|
| **理想解ベースモデル（Ideal-solution based model）** | パラメータ空間に「理想解」を設定し、その距離や差異に基づいて適応度を計算する | 評価が安定しやすく、数学的モデル化が容易 | Takagi (2001) など |
| **確率分布ベースモデル（Probability distribution model）** | 理想解周辺に正規分布・ガウス分布などを設定し、その確率密度で評価値を生成する | ランダム性を含みつつ一貫した傾向を持つ | Cho & Lee (2009) など |
| **ノイズ付加モデル（Noise-added model）** | 理想評価にランダムノイズを付加して疑似的な人間のばらつきを再現 | ユーザの疲労や一貫性低下をシミュレーション可能 | Takagi (2001) 派生研究 |
| **ペアワイズ比較モデル（Pairwise comparison model）** | 個体間の比較のみを行い順位や勝敗を決定 | 評価負荷が低いが、データが間接的 | Kim & Cho (2000) など |
| **クラスタリング利用モデル（Clustering-based model）** | 表現型空間をクラスタリングして、クラスタ代表に基づき評価を行う | 評価回数削減が可能 | Cho & Lee (2009) |
| **機械学習予測モデル（Machine learning surrogate model）** | 初期評価データを使い回帰・分類モデル（SVM, NN等）で未評価個体の適応度を推定 | 高速だが学習誤差が影響 | Ishibuchi et al. (2003) など |
| **ユーザ特性模倣モデル（User profile mimic model）** | 過去のユーザ評価履歴を模倣して評価関数を構築 | 個人差や嗜好再現が可能 | Lee et al. (2012) など |
| **疲労進行モデル（Fatigue progression model）** | 評価回数が増えるにつれてノイズや評価変動幅を増加させる | 長期実験シミュレーションに有効 | Takagi (2001) 派生研究 |

# 使っている論文
- [先輩の修論(GithubのURL)](https://github.com/mocoatsu/Research)
- ["Reducing user fatigue in interactive genetic algorithms by evaluation of population subsets." ](https://www.cse.unr.edu/~quiroz/inc/docs/trans2009.pdf)
# 読むだけ読んだ論文
- [自己組織化マップを用いた遺伝的アルゴリズムについて](https://doi.org/10.1299/jsmeoptis.2008.8.93)
- [擬音的発話のニュアンスを反映するインタラクティブ効果音合成](https://www.interaction-ipsj.org/proceedings/2024/data/pdf/1B-34.pdf)
- ["Voice Recognition based on vote-SOM."](https://www.researchgate.net/publication/281284888_Voice_Recognition_based_on_vote-SOM)
- ["A spiking neural network framework for robust sound classification."](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00836/full)
- ["対話履歴からの獲得知識に基づく模擬評価関数を用いた対話型進化計算"](https://www.jstage.jst.go.jp/article/jjske/14/4/14_TJSKE-D-15-00069/_pdf)
- ["Interactive Evolutionary Computation withEvaluation Characteristics of Multi-IEC Users"](https://catalog.lib.kyushu-u.ac.jp/opac_download_md/4488101/IntConf101.pdf)
    - 過去の好みが似ている別ユーザの評価特性を現在のユーザの特性を学習するまで用いることで収束を早める？
    - 評価特性の学習はパラメータと評価値をセットにしたデータからNNを用いていた、残念
    - 根本的には時間稼ぎ的な手法

# IGAの評価指標
- 最良個体の評価値の推移を見る
    - 何回も実験して各世代の最高評価値の平均を見る
    - 収束の早さ?
    - 使っていた論文
        - ["対話履歴からの獲得知識に基づく模擬評価関数を用いた対話型進化計算"](https://www.jstage.jst.go.jp/article/jjske/14/4/14_TJSKE-D-15-00069/_pdf)
        - ["対話型進化計算における YUKI アルゴリズムの適用"](https://www.jstage.jst.go.jp/article/jsoft/37/1/37_553/_pdf/-char/ja)
        - ["Reducing user fatigue in interactive genetic algorithms by evaluation of population subsets."](https://www.cse.unr.edu/~quiroz/inc/docs/trans2009.pdf)
- 収束特性
    - よくわからん
    - 使っていた論文
        - ["推定収束点を用いた対話型進化計算高速化の可能性"](https://api.lib.kyushu-u.ac.jp/opac_download_md/1810697/FSS2017.pdf)