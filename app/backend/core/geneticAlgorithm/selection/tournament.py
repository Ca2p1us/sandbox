from itertools import permutations
from typing import List
from heapq import nlargest
from random import choices,sample
import numpy as np
Chromosomes = List[dict]

RNG = np.random.default_rng(seed=20)

# def exec_tournament_selection(chromosomes_params:dict) ->  List[Chromosomes]:
def exec_tournament_selection(chromosomes_params, participants_num=3):
        """
        トーナメント選択を行い、交叉などで利用するための2つの個体
        （染色体）を取得する。

        Returns
        -------
        selected_chromosomes : list of Chromosome
            選択された2つの個体（染色体）を格納したリスト。トーナメント
            用に引数で指定された件数分抽出された中から上位の2つの個体が
            設定される。
        """
        # participants_num: int = len(chromosomes_params) // 2
        # participants: List = sample(list(chromosomes_params.values()), k=participants_num)
        # participants = sample(chromosomes_params, k=participants_num)
        participants = RNG.choice(chromosomes_params, size=participants_num, replace=False, shuffle=True)
        selected_chromosomes: List[Chromosomes] = nlargest(n=2, iterable=participants,key=lambda params: params["fitness"])
        return selected_chromosomes
