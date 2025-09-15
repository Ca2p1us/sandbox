#事前評価・適応度評価を行う

from typing import List, Dict
import math
import random

def interpolate_by_distance(
    population: List[dict],
    best: dict = None,
    worst: dict = None,
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation"
):
    """
    best, worstを基準に、個体群のtarget_key（pre_evaluationやfitnessなど）を距離に応じて線形補間で再評価する。
    best/worstがNoneの場合はランダムな値を付与する。
    param_keys: 距離計算に使うパラメータ名リスト（Noneならoperator1のfrequencyのみ）
    target_key: 補間して格納するキー名（"pre_evaluation"または"fitness"など）
    """
    if not best or not worst:
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = random.randint(1, 10)
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
            value = best.get(target_key, best.get("fitness", 1))
            ind[target_key] = int(float(value))
        return

    best_val = float(best.get(target_key, best.get("fitness", 1)))
    worst_val = float(worst.get(target_key, worst.get("fitness", 1)))

    for ind in population:
        ind_vec = to_vec(ind)
        dist_best = euclidean(ind_vec, best_vec)
        # 距離が近いほどbest_valに近づく
        ratio = dist_best / max_dist
        value = best_val * (1 - ratio) + worst_val * ratio
        ind[target_key] = str(value)