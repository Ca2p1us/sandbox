# シミュレータの動かし方
## appディレクトリに移動して下記のコマンドで実行
```tarminal
python main.py
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

# 評価関数
1. $f(x) = \sum_{i}x_i^2 \quad -5.12<x_i<5.12$
2. $f(x) = \sum_{i}(ix_i^4 - Gauss(0,1)) \quad -1.28<x_i<1.28$
3. $f(x) = 20A + \sum_{i}(x_i^2 - 10cos(2\pi x_i)) \quad A = 10,-5.12<x_i<5.12$
各$x_i$を理想解の値だけずらせばOKか

# 疑似ユーザモデルの代表的分類

- ["chatGPTの履歴](https://chatgpt.com/share/68c77ce7-6394-8009-95de-de186ff53d2d)

# 使っている論文
- [先輩の修論(GithubのURL)](https://github.com/mocoatsu/Research)
- ["Reducing user fatigue in interactive genetic algorithms by evaluation of population subsets." ](https://www.cse.unr.edu/~quiroz/inc/docs/trans2009.pdf)
- ["An Experimental Study of Benchmarking Functions for Genetic Algorithms"](https://www.researchgate.net/publication/220662178_An_Experimental_Study_of_Benchmarking_Functions_for_Genetic_Algorithms)
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

# 作成中のメモ
 - 個体の選択
    - pre_evaluationが小数になる場合を考えてなかった。