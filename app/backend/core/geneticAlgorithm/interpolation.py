#事前評価・適応度評価を行う

from typing import List, Dict
import math
import random
import numpy as np

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
    print(f"補間を開始します。")
    if not best or not worst:
        rng = np.random.default_rng(seed=10)
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = rng.uniform(1.0, 10.0)
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
            ind[target_key] = float(value)
        print(f"距離が同一です。\nbest {best[target_key]}, worst {worst[target_key]}")
        return

    best_val = float(best.get("fitness", 1))
    worst_val = float(worst.get("fitness", 1))

    for ind in population:
        ind_vec = to_vec(ind)
        dist_best = euclidean(ind_vec, best_vec)
        # 距離が近いほどbest_valに近づく
        ratio = dist_best / max_dist
        value = best_val * (1 - ratio) + worst_val * ratio
        ind[target_key] = value
    print(f"{target_key}を補間しました。\nbest {best_val}, worst {worst_val}")
    return


def interpolate_by_Gaussian(
    population: List[dict],
    best: dict = None,
    worst: dict = None,
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation",
    eps_ratio: float = 0.02
):
    """
    ガウス関数 f(x) = A exp(-(x-μ)^2/(2σ^2)) + C のパラメータ推定
    C ≈ y2 として、わずかに ε を下げた近似で計算。
    
    eps_ratio: 全振幅に対する補正割合（例: 0.02 = 2%）
    """
    print(f"ガウス補間を開始します。")
    if not best or not worst:
        rng = np.random.default_rng(seed=10)
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = rng.uniform(1.0, 10.0)
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

    best_val = float(best.get(target_key, 1))
    worst_val = float(worst.get(target_key, 1))
    eps = (best_val - worst_val) * eps_ratio
    C = worst_val - eps
    A = best_val - C
    print(best[target_key] - C)
    ratio = (worst[target_key] - C) / (best[target_key] - C)
    if  ratio <= 0 or ratio >= 1:
        print("ガウス補間の計算に失敗しました。bestとworstの値を確認してください。")
        return
    best_params = [get_param(best, k) for k in param_keys]
    worst_params = [get_param(worst, k) for k in param_keys]
    sigma = (sum(abs(b - w) for b, w in zip(best_params, worst_params)) / math.sqrt(-2 * math.log(ratio)))
    mu = best_params
    print(f"ガウス補間のパラメータ: A={A}, mu={mu}, sigma={sigma}, C={C}")

    # 個体群のガウス補間
    for ind in population:
        ind_params = [get_param(ind, k) for k in param_keys]
        dist_sq = sum((ip - mp) ** 2 for ip, mp in zip(ind_params, mu))
        value = A * math.exp(-dist_sq / (2 * sigma ** 2)) + C
        ind[target_key] = value
    print(f"{target_key}をガウス補間しました。\nbest {best_val}, worst {worst_val}")
    return