# シミュレータの動かし方
## appディレクトリに移動して下記のコマンドで実行
```tarminal
python -m backend.engine.run_iga_simulation
```

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

# IGAの評価指標
- 最良個体の評価値の推移を見る
    - 収束の早さ?
- 収束特性
    - よくわからん