# シミュレータの動かし方
## パラメータの定義域を変更
**config.py**には各パラメータの定義域を(最小値,最大値)の形式でタプルとして保存している
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
    - Sphere関数
2. $f(x) = \sum_{i}exp(\frac{x_i^2}{2\sigma^2})$
    - ガウス関数
3. $f(x) = 20A + \sum_{i}(x_i^2 - 10cos(2\pi x_i)) \quad A = 10,-5.12<x_i<5.12$
    - Rastrigin関数
4. $f(x) = 418.9829 * d - \sum_{i}(x_i sin(\sqrt(|x_i|))) \quad d:次元数,$
    - Schwefel関数
5. $f(x) = 20 - 20exp(-0.2\sqrt{}\frac{1}{n}\sum_{i}x_i^2) + e - exp(\frac{1}{n} \sum_{i} cos(2\pi x_i))$

各$`x_i`$を理想解の値だけずらせばOKか

# 疑似ユーザモデルの代表的分類

- [chatGPTの履歴](https://chatgpt.com/share/68c77ce7-6394-8009-95de-de186ff53d2d)

## 自分の研究について
# 課題について
- 擬音語(オノマトペ)からそれが表していそうな音を生成
    - この路線なら単峰性を仮定して進めればいい
    - 想定する音はあるけどどうやったらその音が出せるかがわからないときにそれを教えてくれるみたいな感じ
- 何らかのシチュエーションを想像したときにより良い音を生成
    - これなら多峰性で考えなければならない、ユーザが何もない状態で想像するのは一つのシチュエーションであるから
    - ユーザが思いもよらない音を生成するのが強みとなる
# 他の研究のIGA,GAが目的としている課題の整理
- [先輩の修論(GithubのURL)](https://github.com/mocoatsu/Research)
    - 単峰性を仮定
    - 擬音語に合った音を生成
- [山本先輩の修論]
    - 単峰性を仮定
    - 個人解を生成
- ["対話型遺伝的アルゴリズムの評価操作におけるユーザの負担軽減の検討"](https://www.jstage.jst.go.jp/article/jsmecmd/2007.20/0/2007.20_315/_pdf/-char/ja)
    - 単峰性を仮定
    - T-シャツのデザインを個体としたシステム
    - 提示されたデザインから好みのものを2個選ぶ評価を提案
    - T-シャツのデザインの嗜好を探る＋ユーザ疲労の軽減
    - 嗜好を探る点では点数をつける方が良かった
        - 仮定は単峰性だが結果から見えるのは多峰性か
- ["対話型遺伝的アルゴリズムにおける嗜好の多峰性に対応可能な個体生成方法"](https://www.jstage.jst.go.jp/article/tjsai/24/1/24_1_127/_pdf)
    - 多峰性を仮定
    - ECサイトの商品推薦システム
    - 従来のIGAでは複数の最適解の内の一つに収束する単峰性
    - 他峰性である嗜好のピークに至りつつ広い空間を探索
        - 嗜好の一つだけを推薦するのもストレスと書いている
    - 途中のある世代でクラスタリングを行い、個体の生成範囲を絞る
        - クラスタリングの手法によってはよさげな方法である
    - 評価方法
        - ユーザが選択したか否かの二値
        - 必ず母集団の半数が親個体となるように、足りない場合は選ばれたものとユークリッド距離で近いものを親とする
    - 交叉
        - BLX-αに似ているが、拡張はなし
- ["対話型遺伝的アルゴリズムによる人間の感性を反映した音楽作曲"](https://www.jstage.jst.go.jp/article/jsoft/17/6/17_KJ00003983705/_pdf/-char/ja)
    - 単峰性を仮定
    - 博士論文紹介であったため詳細は不明
- ["対話型遺伝的アルゴリズムを用いたユーザの目的に合った楽曲推薦システム"](https://db-event.jpn.org/deim2010/proceedings/files/A4-1.pdf)
    - 単峰性を仮定
    - 目的に合っているか否かの二値評価
- ["Interactive Genetic Algorithms for User Interface Design"](https://www.cse.unr.edu/~dascalus/CEC2007_Quiroz.pdf)
    - 多峰性か単峰性かわからん
    - UIのデザイン
        - デザインの原則やガイドラインがあると言及しているから単峰性を仮定している可能性が高い？
        - 結果とか人間の美的感覚的には多峰性
- ["IGAOD: An online design framework for interactive genetic algorithms"](https://www.researchgate.net/publication/363741083_IGAOD_An_online_design_framework_for_interactive_genetic_algorithms#:~:text=In%20order%20to%20prompt%20the,IGA%29%2C%20the)
    - 多峰性を仮定
    - 花瓶のデザインを0～10で評価
    - そもそもフレームワークの開発の研究だから入れない方がいいかも
- ["3D Vase Design Based on Interactive Genetic Algorithm and Enhanced XGBoost Model"](https://www.mdpi.com/2227-7390/12/13/1932#:~:text=The%20human%E2%80%93computer%20interaction%20attribute%20of,three%20parts%3A%20the%20vase%20control)
    - 単峰性を仮定？
    - PSOっちゅう粒子群最適化アルゴリズムを使って最適解に移動するらしい
- ["遺伝的アルゴリズムを利用した最適トラス形態決定法"](https://www.jstage.jst.go.jp/article/kikaia1979/59/562/59_562_1568/_pdf/-char/ja)
    - 多峰性を仮定
    - トラス構造の設計で問題となる位相決定問題は多峰性らしい
    - 従来の最適化とGAを組み合わせることで局所解に陥るのを防いでいるらしい
- ["遺伝的アルゴリズムの道路整備順位決定問題への適用"](https://www.jstage.jst.go.jp/article/jscej1984/1994/482/1994_482_37/_pdf/-char/ja)
    - 多峰性を仮定
    - 道路の整備順を決める問題
    - どう見ても局所解がたくさんある

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

## 使っている論文
- [先輩の修論(GithubのURL)](https://github.com/mocoatsu/Research)
- ["Reducing user fatigue in interactive genetic algorithms by evaluation of population subsets." ](https://www.cse.unr.edu/~quiroz/inc/docs/trans2009.pdf)
- ["An Experimental Study of Benchmarking Functions for Genetic Algorithms"](https://www.researchgate.net/publication/220662178_An_Experimental_Study_of_Benchmarking_Functions_for_Genetic_Algorithms)
    - F1～F5
    - ["An analysis of the behaviour of a class of genetic adaptive systems"](https://deepblue.lib.umich.edu/handle/2027.42/4507)
- [最適化アルゴリズムを評価するベンチマーク関数まとめ](https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda)
- ["Virtual Library of Simulation Experiments:Test Functions and Datasets"](https://www.sfu.ca/~ssurjano/optimization.html)
- [Test Functions for Unconstrained Global Optimization](http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm)
- ["Test Functions for Unconstrained Global Optimization"](http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm)
    - エジプトの教授が作ってた評価関数のまとめサイト
    - [その教授のサイト](http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/AboutMe.htm)
    - [その教授の所属大学のサイト](https://www.aun.edu.eg/fci/abdel-rahman-hedar-abdel-rahman-ahmed)
- ["Accelerating the Evolutionary Algorithms by Gaussian Process Regression with $\epsilon$ -greedy acquisition function"](https://arxiv.org/pdf/2210.06814)
    - ガウス過程回帰を用いたエリート個体の推定
    - 個体群から分布を推定して次世代を生成するEDAと個体群を選択交叉して次世代を生成するEAのハイブリッド
    - 基本はEAアルゴリズムで個体を生成したりする
    - エリート個体の推定のためにガウス過程回帰で今ある個体から分布の推定をする



## 読むだけ読んだ論文
- [自己組織化マップを用いた遺伝的アルゴリズムについて](https://doi.org/10.1299/jsmeoptis.2008.8.93)
- [擬音的発話のニュアンスを反映するインタラクティブ効果音合成](https://www.interaction-ipsj.org/proceedings/2024/data/pdf/1B-34.pdf)
- ["Voice Recognition based on vote-SOM."](https://www.researchgate.net/publication/281284888_Voice_Recognition_based_on_vote-SOM)
- ["A spiking neural network framework for robust sound classification."](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00836/full)
- ["対話履歴からの獲得知識に基づく模擬評価関数を用いた対話型進化計算"](https://www.jstage.jst.go.jp/article/jjske/14/4/14_TJSKE-D-15-00069/_pdf)
- ["Interactive Evolutionary Computation withEvaluation Characteristics of Multi-IEC Users"](https://catalog.lib.kyushu-u.ac.jp/opac_download_md/4488101/IntConf101.pdf)
    - 過去の好みが似ている別ユーザの評価特性を現在のユーザの特性を学習するまで用いることで収束を早める？
    - 評価特性の学習はパラメータと評価値をセットにしたデータからNNを用いていた、残念
    - 根本的には時間稼ぎ的な手法
- ["Automated Design of a Genetic Algorithm for Image Segmentation Usingthe Iterated Local Search"](https://www.scitepress.org/Papers/2024/129085/129085.pdf)
    - エントロピーを使うのは面白そうだけど音声なので自分の研究には使えないかも

    


# 作成中のメモ
 - 個体の選択
    - pre_evaluationが小数になる場合を考えてなかった。
- 事前評価の補間
    - 10/13 fitness参照でベストとワースト選んだのに補間ではpre_evaluationを使っててカス、選んだ意味がなかったので修正。
- 結果の表示方法
    - 10/13 5回連続でシミュレータを動かしてグラフとかを表示するようにした。そのため5回分を重ねたグラフのみ実行時に表示させるようにした。
- 評価関数のパラメータ
    - 10/15 定義域を変えた場合のパラメータ設定についてgeogebraで1次元の場合の関数を眺めながら変えた
    - 10/16 Rastrigin関数は定義域が小さいから成り立っている関数なんだなぁと思ったのででかくした
    - 11/10 ガウス関数の標準偏差を一括で設定できるのに各呼び出しで設定していてカス
- 乱数について
    - numpy.randomのdefault_rngってやつで生成器を定義する(シード値を固定できる)
    - 各パーツのシード値を書いておかなければならない
        - interpolation.py seed=10
        - tournament.py seed=20
        - BLX-alpha.py seed=30
        - mutate.py seed=40
        - make_chromosome_params.py seed=42
- 補間関連
    - ガウス補間を上手くいかせるために上位と下位から実際の評価個体を選択するようにした
- そもそも
    - 来年以降もIGAを触るならXGBoostとか別の部分からのアプローチも調べないとなぁ
    - GA単体で何とかする研究は少ないんだなぁ
    - ほかの最適化手法やシンプルなGAに特殊な処理を加えるとよりよくなる(当たり前)