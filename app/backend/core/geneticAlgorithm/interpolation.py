#事前評価を行う

from typing import List, Dict
import math

def interpolate_fitness_by_distance(population: List[dict], best: dict, worst: dict, param_keys: List[str] = None):
    """
    best, worstを基準に、個体群のfitnessを距離に応じて線形補間で再評価する。
    param_keys: 距離計算に使うパラメータ名リスト（Noneならoperator1のfrequencyのみ）
    """
    if not best or not worst:
        print("bestまたはworstがNoneです")
        return

    if param_keys is None:
        # デフォルトはoperator1のfrequencyのみ
        param_keys = ["fmParamsList.operator1.frequency"]

    def get_param(ind, key):
        # "fmParamsList.operator1.frequency" のようなドット区切りでアクセス
        val = ind
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        return val

    # ベクトル化
    def to_vec(ind):
        return [get_param(ind, k) for k in param_keys]

    best_vec = to_vec(best)
    worst_vec = to_vec(worst)

    # 距離計算
    def euclidean(vec1, vec2):
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(vec1, vec2) if a is not None and b is not None))

    max_dist = euclidean(best_vec, worst_vec)
    if max_dist == 0:
        # 全員同じ場合
        for ind in population:
            ind["fitness"] = best["fitness"]
        return

    best_fitness = float(best["fitness"])
    worst_fitness = float(worst["fitness"])

    for ind in population:
        ind_vec = to_vec(ind)
        dist_best = euclidean(ind_vec, best_vec)
        # 距離が近いほどbest_fitnessに近づく
        ratio = dist_best / max_dist
        fitness = best_fitness * (1 - ratio) + worst_fitness * ratio
        ind["fitness"] = str(int(round(fitness)))